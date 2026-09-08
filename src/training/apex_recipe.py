"""Versioned Apex recipes and safe continuation checks.

A digest signs a descriptor, but it is not evidence that an optimizer state was
not changed after that descriptor was written.  Continuations check both.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

import torch.optim

from src.core.game_config import get_config
from src.core.reward_contract import current_reward_contract
from src.core.runtime_contract import canonical_digest


def optimizer_descriptor(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Capture every optimizer group option except nonportable parameter ids."""
    groups = []
    for group in optimizer.param_groups:
        options = {key: value for key, value in group.items() if key != "params"}
        options["parameter_count"] = len(group["params"])
        groups.append(options)
    return {"class": type(optimizer).__name__, "param_groups": groups}


def serialized_optimizer_descriptor(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a serialized state for comparison with its declared groups."""
    groups = state.get("param_groups")
    if not isinstance(groups, list):
        raise ValueError("optimizer continuation has no param_groups")
    result = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("optimizer continuation has malformed param_groups")
        options = {key: value for key, value in group.items() if key != "params"}
        options["parameter_count"] = len(group["params"])
        result.append(options)
    return {"param_groups": result}


@dataclass(frozen=True)
class ApexRecipe:
    """Numerical and replay semantics that must agree before a continuation."""

    mode: str
    optimizer: str
    grad_clip_norm: float
    beta_clock: str
    gamma: float
    n_step: int
    priority_alpha: float
    priority_eps: float
    target_clip: float = 50.0
    schema_version: int = 2
    optimizer_contract: Mapping[str, Any] = field(default_factory=dict)
    loss_contract: Mapping[str, Any] = field(default_factory=dict)
    target_contract: Mapping[str, Any] = field(default_factory=dict)
    replay_contract: Mapping[str, Any] = field(default_factory=dict)
    observation_contract: Mapping[str, Any] = field(default_factory=dict)
    world_contract: Mapping[str, Any] = field(default_factory=dict)
    runtime_contract: Mapping[str, Any] = field(default_factory=dict)
    reward_contract: Mapping[str, Any] = field(default_factory=dict)
    exploration_contract: Mapping[str, Any] = field(default_factory=dict)
    seeding_contract: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"local", "distributed"}:
            raise ValueError(f"Unsupported Apex recipe mode {self.mode!r}")
        if self.optimizer not in {"Adam", "AdamW"}:
            raise ValueError(f"Unsupported Apex optimizer {self.optimizer!r}")
        if self.schema_version != 2:
            raise ValueError("Unsupported Apex recipe schema_version")
        if self.n_step < 1 or self.grad_clip_norm <= 0 or self.target_clip <= 0:
            raise ValueError("Apex recipe clipping and n_step values must be positive")
        canonical_digest(self.semantic_dict())

    def semantic_dict(self) -> dict[str, Any]:
        """Return every persisted semantic field in one canonical mapping."""
        return asdict(self)

    @property
    def digest(self) -> str:
        return canonical_digest(self.semantic_dict())

    def to_metadata(self) -> dict[str, Any]:
        return {"apex_recipe": self.semantic_dict(), "apex_recipe_digest": self.digest}

    @classmethod
    def local(
        cls, *, optimizer: torch.optim.Optimizer, input_size: int, output_size: int,
        gamma: float, n_step: int, target_clip: float = 50.0,
        seed_identity: Optional[Mapping[str, Any]] = None,
    ) -> "ApexRecipe":
        """Build the complete descriptor for the actual local AdamW path."""
        cfg = get_config()
        return cls._build(
            mode="local", optimizer=optimizer, input_size=input_size, output_size=output_size,
            gamma=gamma, n_step=n_step, target_clip=target_clip,
            grad_clip_norm=10.0,
            priority_alpha=cfg.apex.priority_alpha, priority_eps=cfg.apex.priority_epsilon,
            beta_clock="successful_local_sample", world=asdict(cfg.game),
            runtime={"mode": "local_apex_policy", "training": True, "distributed": False,
                     "episode_lifecycle": "GameState-owned"},
            exploration={"epsilon_start": cfg.training.epsilon_start,
                         "epsilon_end": cfg.training.epsilon_end,
                         "epsilon_decay": cfg.training.epsilon_decay,
                         "selection": "epsilon_greedy_numpy_global"},
            seed_identity=seed_identity or {"seed_source": "not-owned-by-local-policy"},
        )

    @classmethod
    def distributed(
        cls, *, optimizer: torch.optim.Optimizer, input_size: int, output_size: int,
        gamma: float, n_step: int, priority_alpha: float, priority_eps: float,
        world: Mapping[str, Any], runtime: Mapping[str, Any], actor_scaling: Mapping[str, Any],
        seed_identity: Mapping[str, Any], target_clip: float = 100.0, grad_clip_norm: Optional[float] = None,
    ) -> "ApexRecipe":
        """Build a coordinator-bound distributed descriptor, never ambient world values."""
        cfg = get_config()
        return cls._build(
            mode="distributed", optimizer=optimizer, input_size=input_size, output_size=output_size,
            gamma=gamma, n_step=n_step, target_clip=target_clip,
            grad_clip_norm=float(cfg.training.grad_clip_norm if grad_clip_norm is None else grad_clip_norm),
            priority_alpha=priority_alpha, priority_eps=priority_eps,
            beta_clock="successful_distributed_sample", world=world, runtime=runtime,
            exploration={"epsilon_base": cfg.apex.epsilon_base, "epsilon_alpha": cfg.apex.epsilon_alpha,
                         "selection": "epsilon_greedy_actor_namespace", "actor_scaling": dict(actor_scaling)},
            seed_identity=seed_identity,
        )

    @classmethod
    def _build(
        cls, *, mode: str, optimizer: torch.optim.Optimizer, input_size: int, output_size: int,
        gamma: float, n_step: int, target_clip: float, grad_clip_norm: float,
        priority_alpha: float, priority_eps: float,
        beta_clock: str, world: Mapping[str, Any], runtime: Mapping[str, Any],
        exploration: Mapping[str, Any], seed_identity: Mapping[str, Any],
    ) -> "ApexRecipe":
        cfg = get_config()
        return cls(
            mode, type(optimizer).__name__, float(grad_clip_norm), beta_clock,
            float(gamma), int(n_step), float(priority_alpha), float(priority_eps), float(target_clip),
            optimizer_contract=optimizer_descriptor(optimizer),
            loss_contract={"name": "smooth_l1_loss", "beta": 1.0,
                           "reduction": "none_then_importance_weighted_mean"},
            target_contract={"algorithm": "double_dqn", "bootstrap": "gamma_to_actual_n",
                             "terminal": "actual_done_no_bootstrap", "network": "hard_target",
                             "cadence_updates": cfg.apex.target_update_freq,
                             "target_q_clip": target_clip, "gradient_clip_norm": grad_clip_norm},
            replay_contract={"kind": "prioritized_n_step", "alpha": priority_alpha,
                             "priority_epsilon": priority_eps, "beta_start": cfg.apex.priority_beta_start,
                             "beta_end": cfg.apex.priority_beta_end,
                             "beta_increment": cfg.training.priority_beta_increment,
                             "beta_clock": beta_clock},
            observation_contract={"spec": "vector61" if input_size == 61 else "vector58",
                                  "input_size": input_size, "output_size": output_size,
                                  "layout": "relative6_vector",
                                  "mask_mode": "legacy_advisory_rowwise_legal_intersection"},
            world_contract=dict(world), runtime_contract=dict(runtime),
            reward_contract=current_reward_contract(), exploration_contract=dict(exploration),
            seeding_contract=dict(seed_identity),
        )


def validate_recipe_continuation(
    checkpoint: Mapping[str, Any], recipe: ApexRecipe, *, weights_only: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Reject incompatible resume state before optimizer or child construction."""
    if weights_only:
        return
    stored = checkpoint.get("apex_recipe")
    stored_digest = checkpoint.get("apex_recipe_digest")
    if not isinstance(stored, Mapping) or not isinstance(stored_digest, str):
        raise ValueError("optimizer continuation requires a verified Apex recipe")
    if canonical_digest(dict(stored)) != stored_digest:
        raise ValueError("checkpoint Apex recipe digest does not match its descriptor")
    if stored_digest != recipe.digest:
        raise ValueError("checkpoint Apex recipe conflicts with requested continuation")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("optimizer continuation requires optimizer_state_dict")
    stored_optimizer = stored.get("optimizer_contract")
    if not isinstance(stored_optimizer, Mapping):
        raise ValueError("optimizer continuation recipe lacks optimizer_contract")
    actual_optimizer = serialized_optimizer_descriptor(optimizer_state)
    if actual_optimizer["param_groups"] != stored_optimizer.get("param_groups"):
        raise ValueError("optimizer continuation state conflicts with its Apex recipe")
    if optimizer is not None and optimizer_descriptor(optimizer) != dict(stored_optimizer):
        raise ValueError("optimizer continuation conflicts with receiving optimizer")
