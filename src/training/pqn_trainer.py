"""PQN Q(lambda) trainer for the dual-scale raster network (blueprint P3 §3).

PQN (Parallelized Q-Network) is synchronous Q(lambda) on a batch of vectorized
environments — **no replay buffer, no target network, no PER**. Each update:

1. Roll out ``T`` steps on :class:`~src.simd_env.batch_sim.BatchSim` (E envs x S
   snakes) with pool self-play (:mod:`src.training.pqn_selfplay`), collecting
   per-slot ``(obs, action, reward, mask, done, trapped)`` and the hero Q-values.
   Episodes end at the v2 train-mode population floor or ``max_frames``; steps
   taken after an env's episode ends are not transitions and never train.
2. After the rollout, boot the network once more on the final observation to get
   the bootstrap Q(s') for the truncation case.
3. Compute the per-agent Q(lambda) return **backward** over each hero slot's
   rollout with blueprint termination handling:
     - **DEATH** (``done``): return is the reward alone (no bootstrap).
     - **TRAPPED** non-terminal (no valid next action, but not dead this step):
       bootstrap to the DEATH VALUE (a config constant), not 0.
     - **TRUNCATION** (rollout edge, or the last step of an episode): bootstrap
       from the masked-max Q(s') over valid next actions.
     - Interior alive steps: standard Q(lambda) mixing
       ``G_t = r_t + gamma * ((1-lambda) * max_a' Q(s',a') + lambda * G_{t+1})``.
4. Take several minibatch SGD steps (Huber loss on ``Q(s,a) - G``), grad-norm
   clip 10, **no TD-target clipping**. Optional horizontal-flip augmentation.

Only HERO-slot transitions train (frozen-opponent slots are ignored). The
network sees ε-greedy behavior with the blueprint ladder (1.0 -> 0.02 over the
first ~50M agent-steps, scaled down for smoke).

Telemetry (:class:`PQNTelemetry`) and tripwires (NaN/inf, max|Q| alarm,
action-collapse) are emitted per update so an outer loop can halt-and-flag.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, fields
from numbers import Complex, Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.reward_events import DEATH_REWARD, KILL_REWARD_PER_VICTIM_LENGTH
from src.core.runtime_contract import (
    EffectiveWorldConfig,
    ModelHeadContract,
    RunProvenance,
    RuntimeModeContract,
    canonical_digest,
)
from src.core.seeding import derive_seed
from src.model.checkpoint_io import atomic_torch_save
from src.model.obs_spec import (
    OBS_SPEC_KEY,
    RASTER31V2,
    RASTER31V2_SHAPES,
    RASTER31V3,
    RASTER31V3_CONTRACT,
)
from src.model.raster_network import (
    SCALARS_DIM,
    STRATEGIC_SHAPE,
    TACTICAL_SHAPE,
    RasterDuelingNetwork,
    raster_tensors_from_obs,
)
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.featurizer import build_observations, obs_inputs_from_batch_sim
from src.simd_env.gpu_featurizer import build_observations_gpu, obs_inputs_to_torch
from src.training.pqn_lifecycle import (
    EPISODE_RESET_PER_ENV_AUTORESET,
    EPISODE_SEED_DERIVED,
    POOL_ADMISSION_DISABLED,
    build_pqn_episode_lifecycle_contract,
)
from src.training.pqn_selfplay import (
    HERO_POLICY_ID,
    OpponentLease,
    OpponentPool,
    PerEnvExplorationDecisions,
    PinnedOpponentPool,
    assign_policy_ids,
    assign_policy_ids_per_env,
    batched_act,
    sample_per_env_exploration,
)
from src.training.rollout_policies import FixedPolicySource

__all__ = [
    "PQNConfig",
    "PQNTrainer",
    "PQNTelemetry",
    "PQNPerEnvTelemetry",
    "TripwireError",
    "flip_augment",
    "validate_checkpoint_numeric_state",
]


# Lateral scalar indices in the 26-D scalar vector that flip SIGN under a
# horizontal (left<->right) mirror of the world. Derived from featurizer §_build_scalars:
#   9  wall dist RIGHT  <-> 11 wall dist LEFT   (swap, handled separately)
#   12 nearest-food ego dx (lateral)            -> negate
#   15 enemy1 ego dx (lateral)                  -> negate
#   19 enemy2 ego dx (lateral)                  -> negate
# The world-x scalar (23) also mirrors, but it is an absolute arena coordinate
# with no canonical flip center under an ego mirror, so it is left untouched (the
# raster/ego lateral signals carry the mirrored geometry). Ahead/behind and
# distances are flip-invariant.
_LATERAL_NEGATE_IDX = (12, 15, 19)
# Wall-distance left/right pair to SWAP under the mirror.
_WALL_LR_SWAP = (9, 11)

# Action layout: 0=turn-left, 1=straight, 2=turn-right (normal),
#                3=turn-left, 4=straight, 5=turn-right (boost).
# A horizontal mirror swaps left<->right: 0<->2 and 3<->5.
_FLIP_ACTION = np.array([2, 1, 0, 5, 4, 3], dtype=np.int64)


@dataclass
class PQNConfig:
    """Configuration for :class:`PQNTrainer` (blueprint P3 §3 defaults).

    Attributes:
        num_envs: E parallel environments.
        num_snakes: S snakes per env.
        rollout_len: T steps collected per update.
        gamma: Discount (blueprint: 0.997).
        lambda_: Q(lambda) trace decay (blueprint: 0.65).
        lr: Adam learning rate (blueprint: 5e-4).
        adam_eps: Adam epsilon (blueprint: 1.5e-4).
        grad_clip: Grad-norm clip (blueprint: 10.0).
        minibatches: Number of fixed-size SGD steps in legacy sampling mode.
        minibatch_size: Maximum transitions per SGD minibatch. Legacy mode
            independently samples this many without replacement per batch;
            exact-coverage mode uses it as the balanced-batch size cap.
        sgd_epochs: Optional exact-coverage sampler. ``None`` retains the
            historical fixed-minibatch sampler; a positive value shuffles every
            eligible hero transition once per epoch in balanced minibatches.
        pad_sgd_batches: Pad undersized SGD batches to ``minibatch_size`` for
            the network forward only, then discard padded predictions before
            loss and telemetry. This is opt-in for fixed accelerator shapes.
        sgd_seed: Optional independent RNG seed for sampling and flip
            augmentation. ``None`` preserves the historical shared RNG stream.
        action_collapse_patience: Consecutive low-entropy updates required
            before the collapse guard stops training.
        action_collapse_min_samples: Real eligible hero samples required across
            a low-entropy streak before the collapse guard stops training.
        action_collapse_raw_actions: Measure collapse from pre-augmentation
            rollout actions instead of sampled SGD actions.
        eps_start / eps_end: ε-greedy endpoints (blueprint: 1.0 -> 0.02).
        eps_decay_steps: Agent-steps over which ε decays (blueprint ~50M; scale
            down for smoke).
        hero_frac: Probability a slot is the hero (blueprint: 0.8).
        pool_capacity: Max frozen opponents resident (blueprint: <=10).
        pool_add_interval: Add a hero snapshot to the pool every N updates.
        death_value: Bootstrap value for TRAPPED non-terminal states (blueprint:
            "TRAPPED -> the DEATH VALUE, not 0"). Defaults to the sim's terminal
            death reward (:data:`~src.core.reward_events.DEATH_REWARD` = -3.0) so
            a trapped-but-still-alive state is trained toward the forced-death
            outcome on the following step, not toward 0.
        flip_augment: Enable horizontal-flip augmentation.
        max_frames: Episode-length cap. Denominator of the episode-progress
            observation scalar, and the hard episode end (see
            :meth:`PQNTrainer._episode_over`) for a batch that never reaches the
            population floor.
        max_abs_q_alarm: Tripwire threshold on max|Q|.
        seed: Base RNG seed.
        arena_type: Arena type for the sim config.
        mechanics_version: Sim mechanics version (blueprint target: 2).
        reward_version: Reward version recorded in the checkpoint metadata.
    """

    num_envs: int = 8
    num_snakes: int = 6
    rollout_len: int = 32
    gamma: float = 0.997
    lambda_: float = 0.65
    lr: float = 5e-4
    adam_eps: float = 1.5e-4
    grad_clip: float = 10.0
    minibatches: int = 4
    minibatch_size: int = 256
    sgd_epochs: Optional[int] = None
    pad_sgd_batches: bool = False
    sgd_seed: Optional[int] = None
    action_collapse_patience: int = 1
    action_collapse_min_samples: int = 0
    action_collapse_raw_actions: bool = False
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_steps: int = 50_000_000
    hero_frac: float = 0.8
    pool_capacity: int = 10
    pool_add_interval: int = 50
    rollout_policy_mode: str = "snapshot_pool"
    fixed_policy_identity: Optional[str] = None
    death_value: float = DEATH_REWARD
    kill_scale: float = KILL_REWARD_PER_VICTIM_LENGTH
    flip_augment: bool = True
    max_frames: int = 5000
    max_abs_q_alarm: float = 1e3
    seed: int = 0
    arena_type: str = "rectangular"
    mechanics_version: int = 2
    reward_version: int = 2
    profile: bool = False  # print a CUDA-synced per-phase time breakdown each update
    # Resolved world and observation normalization. Defaults preserve the old
    # PQN recipe; P1's CLI resolver supplies explicit values for named recipes.
    game_width: int = 1450
    game_height: int = 830
    segment_size: int = 10
    wall_thickness: int = 10
    initial_food: int = 250
    max_food: int = 300
    min_boost_length: int = 5
    boost_length_cost_frames: int = 3
    frame_rate: int = 1
    max_capacity: int = 400
    starvation_max: int = 500
    max_length: int = 400
    obs_spec: str = RASTER31V2
    recipe: str = "legacy"
    field_sources: Optional[Dict[str, str]] = None
    source_revision: str = "unknown"
    requested_device: str = "auto"
    effective_device: str = "cpu"
    episode_reset_mode: str = "batch_barrier_v1"
    episode_seed_mode: str = "continuous_env_rng_v1"
    pool_admission_mode: str = "scheduled_v1"
    initial_opponent_checkpoint_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        """Reject an invalid opt-in exact-coverage epoch count."""
        if self.sgd_epochs is not None and (
            isinstance(self.sgd_epochs, bool)
            or not isinstance(self.sgd_epochs, int)
            or self.sgd_epochs <= 0
        ):
            raise ValueError("sgd_epochs must be a positive integer when set")
        for name, value, minimum in (
            ("action_collapse_patience", self.action_collapse_patience, 1),
            ("action_collapse_min_samples", self.action_collapse_min_samples, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                comparator = "positive" if minimum else "non-negative"
                raise ValueError(f"{name} must be a {comparator} integer")
        if self.reward_version != 2:
            raise ValueError("PQN supports reward_version=2 only")
        if self.mechanics_version not in {1, 2}:
            raise ValueError("PQN supports mechanics_version 1 or 2 only")
        if self.arena_type != "rectangular":
            raise ValueError("PQN supports rectangular arenas only")
        if self.obs_spec not in {RASTER31V2, RASTER31V3}:
            raise ValueError(f"Unsupported PQN observation spec {self.obs_spec!r}")
        if self.recipe not in {"legacy", "corrected-v3"}:
            raise ValueError(f"Unsupported PQN recipe {self.recipe!r}")
        if self.recipe == "corrected-v3" and self.obs_spec != RASTER31V3:
            raise ValueError("corrected-v3 recipe requires obs_spec='raster31v3'")
        if self.recipe == "legacy" and self.obs_spec != RASTER31V2:
            raise ValueError("legacy recipe requires obs_spec='raster31v2'")
        if self.recipe == "corrected-v3" and (self.mechanics_version != 2 or self.flip_augment):
            raise ValueError("corrected-v3 requires mechanics_version=2 and flip_augment=False")
        if self.episode_reset_mode not in {"batch_barrier_v1", "per_env_autoreset_v1"}:
            raise ValueError("unsupported episode_reset_mode")
        if self.episode_seed_mode not in {"continuous_env_rng_v1", "derived_env_episode_v1"}:
            raise ValueError("unsupported episode_seed_mode")
        if self.pool_admission_mode not in {"scheduled_v1", "disabled_v1"}:
            raise ValueError("unsupported pool_admission_mode")
        lifecycle_defaults = (
            self.episode_reset_mode == "batch_barrier_v1"
            and self.episode_seed_mode == "continuous_env_rng_v1"
            and self.pool_admission_mode == "scheduled_v1"
        )
        if self.recipe == "legacy" and not lifecycle_defaults:
            raise ValueError("legacy recipe only accepts default lifecycle compatibility values")
        if self.episode_reset_mode == "per_env_autoreset_v1" and (
            self.recipe != "corrected-v3" or self.episode_seed_mode != "derived_env_episode_v1"
        ):
            raise ValueError(
                "per_env_autoreset_v1 requires corrected-v3 and derived_env_episode_v1"
            )
        if self.initial_opponent_checkpoint_sha256 is not None:
            value = self.initial_opponent_checkpoint_sha256
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(
                    "initial_opponent_checkpoint_sha256 must be 64 lowercase hex characters"
                )
            if (
                self.recipe != "corrected-v3"
                or self.rollout_policy_mode != "snapshot_pool"
                or self.pool_capacity < 1
            ):
                raise ValueError(
                    "initial opponent checkpoint requires corrected-v3 snapshot pool capacity >= 1"
                )
        if self.rollout_policy_mode not in {"snapshot_pool", "fixed"}:
            raise ValueError("rollout_policy_mode must be 'snapshot_pool' or 'fixed'")
        if self.rollout_policy_mode == "fixed" and self.recipe != "corrected-v3":
            raise ValueError("fixed rollout policy requires recipe='corrected-v3'")
        if self.rollout_policy_mode == "fixed" and not self.fixed_policy_identity:
            raise ValueError("fixed rollout policy requires fixed_policy_identity")
        if self.rollout_policy_mode == "snapshot_pool" and self.fixed_policy_identity is not None:
            raise ValueError("snapshot_pool rollout policy cannot declare fixed_policy_identity")
        if self.game_width % self.segment_size or self.game_height % self.segment_size:
            raise ValueError("PQN world width and height must align to segment_size")
        if self.wall_thickness % self.segment_size:
            raise ValueError("PQN world wall_thickness must align to segment_size")


def pqn_action_mask_contract(config: PQNConfig) -> Dict[str, object]:
    """Describe the mask actually selected by this observation recipe."""
    if config.obs_spec == RASTER31V3:
        return {
            "version": "legal-advisory-resolved-v1",
            "action_count": 6,
            "resolution": "row_local_intersection_else_legal",
            "dead_rows": "all_false",
        }
    return {
        "version": "legacy-advisory-v1",
        "action_count": 6,
        "resolution": "advisory_only",
        "dead_rows": "all_false",
    }


def pqn_reward_contract(config: PQNConfig) -> Dict[str, object]:
    """Describe BatchSim's actual unclipped reward-v2 arithmetic."""
    return {
        "version": "pqn-potential-reward-v2",
        "gamma": config.gamma,
        "potential": "logical_length_divided_by_10",
        "terminal_potential": 0.0,
        "shaping": "gamma_times_next_potential_minus_previous_potential",
        "death_bonus": config.death_value,
        "kill_scale": config.kill_scale,
        "kill_sum": "victim_order_left_to_right",
        "clipping": None,
    }


def pqn_target_contract(config: PQNConfig) -> Dict[str, object]:
    """Describe the target and episode semantics selected by ``config``."""
    if config.recipe == "corrected-v3":
        contract: Dict[str, object] = {
            "version": "pqn-qlambda-corrected-v3-lifecycle-v1",
            "gamma": config.gamma,
            "lambda": config.lambda_,
            "death": "actual_done_reward_only",
            "trapped": "unsupported_alive_empty_resolved_mask_fails",
            "empty_successor_bootstrap": "error_for_valid_alive_row",
            "truncation": "masked_max_q_of_successor",
            "lambda_carry": "next_in_rollout_valid_transition_including_death",
            "validity": "env_transition_valid_and_active_episode_env",
            "inactive_worlds": "active_env_mask_freezes_world_rng_and_events",
            "reset": (
                "selected_envs_at_next_rollout_boundary"
                if config.episode_reset_mode == "per_env_autoreset_v1"
                else "whole_batch_at_rollout_boundary_after_all_floor_or_frame_cap"
            ),
            "population_floor": config.mechanics_version == 2 and config.num_snakes >= 3,
            "bootstrap_network": "rollout_frozen_online_network",
            "loss_eligibility": "valid_and_episode_assigned_hero",
            "reward_digest": canonical_digest(pqn_reward_contract(config)),
            "action_mask": pqn_action_mask_contract(config),
        }
        lifecycle = build_pqn_episode_lifecycle_contract(
            config.episode_reset_mode, config.episode_seed_mode
        )
        contract["episode_lifecycle_contract_digest"] = canonical_digest(lifecycle)
        return contract
    return {
        "version": "pqn-qlambda-pre-p2-v1",
        "gamma": config.gamma,
        "lambda": config.lambda_,
        "death": "actual_done_reward_only",
        "trapped": "alive_and_current_selected_mask_empty",
        "trapped_bootstrap": config.death_value,
        "empty_successor_bootstrap": config.death_value,
        "truncation": "masked_max_q_of_successor",
        "lambda_carry": "next_in_rollout_valid_transition_including_death",
        "validity": "env_transition_valid_and_prefloor_live_env",
        "inactive_worlds": "continue_stepping_but_exclude_rows",
        "reset": "whole_batch_at_rollout_boundary_after_all_floor_or_frame_cap",
        "population_floor": config.mechanics_version == 2 and config.num_snakes >= 3,
        "bootstrap_network": "rollout_frozen_online_network",
        "loss_eligibility": "valid_and_rollout_assigned_hero",
        "reward_digest": canonical_digest(pqn_reward_contract(config)),
        "action_mask": pqn_action_mask_contract(config),
    }


def pqn_sampler_contract(config: PQNConfig) -> Dict[str, object]:
    """Describe sampling and policy assignment without inventing resume state."""
    corrected = config.recipe == "corrected-v3"
    fixed = config.rollout_policy_mode == "fixed"
    contract: Dict[str, object] = {
        "version": (
            "pqn-sampler-corrected-v3-lifecycle-v1"
            if corrected
            else "pqn-sampler-corrected-v3" if corrected else "pqn-sampler-pre-p2-v1"
        ),
        "mode": "exact_coverage" if config.sgd_epochs is not None else "minibatch",
        "minibatches": config.minibatches,
        "minibatch_size": config.minibatch_size,
        "sgd_epochs": config.sgd_epochs,
        "pad_sgd_batches": config.pad_sgd_batches,
        "sgd_seed": config.sgd_seed,
        "sampling": (
            "independent_permutation_prefix_per_batch"
            if config.sgd_epochs is None
            else "per_epoch_permutation_balanced_array_split"
        ),
        "eligible": "valid_transitions_of_rollout_assigned_hero_slots",
        "flip_augment": config.flip_augment,
        "augmentation": (
            "legacy_horizontal_flip_per_minibatch_probability_0.5"
            if config.flip_augment
            else "disabled"
        ),
        "sgd_rng": "shared_with_rollout" if config.sgd_seed is None else "independent_seed",
        "num_envs": config.num_envs,
        "num_snakes": config.num_snakes,
        "rollout_len": config.rollout_len,
        "hero_frac": config.hero_frac,
        "pool_capacity": 0 if fixed else config.pool_capacity,
        "pool_add_interval": None if fixed else config.pool_add_interval,
        "requested_pool_capacity": config.pool_capacity if fixed else None,
        "requested_pool_add_interval": config.pool_add_interval if fixed else None,
        "policy_assignment": (
            "common_fixed_policy_for_nonhero_slots"
            if fixed
            else (
                "episode_pinned_bernoulli_hero_else_immutable_snapshot_pool"
                if corrected
                else "resample_each_rollout_bernoulli_hero_else_uniform_pool"
            )
        ),
        "forced_hero_slot": 0,
        "empty_pool": "not_applicable_fixed_source" if fixed else "all_heroes",
        "snapshot_identity": (
            "not_applicable_fixed_source"
            if fixed
            else (
                "immutable_content_hash_stable_id" if corrected else "mutable_fifo_indices_unpinned"
            )
        ),
        "assignment_lifetime": (
            "environment_episode"
            if config.episode_reset_mode == "per_env_autoreset_v1"
            else "batch_episode" if fixed or corrected else "rollout"
        ),
        "episode_pinning": corrected and not fixed,
        "pool_mutation": (
            "not_applicable_fixed_source"
            if fixed
            else (
                "admission_deferred_when_all_snapshots_pinned" if corrected else "between_rollouts"
            )
        ),
        "rollout_policy_source": {
            "mode": config.rollout_policy_mode,
            "identity": (
                config.fixed_policy_identity
                if config.rollout_policy_mode == "fixed"
                else "episode-assigned"
            ),
        },
        "snapshot_admission": (
            "disabled_fixed_source"
            if fixed or config.pool_admission_mode == "disabled_v1"
            else "after_sgd_positive_update_index_divisible_by_interval"
        ),
        "exploration": {
            "policy": "hero_only_epsilon_greedy_constant_within_rollout",
            "clock": "valid_hero_agent_steps",
            "start": config.eps_start,
            "end": config.eps_end,
            "decay_steps": config.eps_decay_steps,
        },
    }
    if corrected:
        lifecycle = build_pqn_episode_lifecycle_contract(
            config.episode_reset_mode, config.episode_seed_mode
        )
        contract["episode_lifecycle_contract_digest"] = canonical_digest(lifecycle)
        contract["policy_source_contract_digest"] = canonical_digest(
            _policy_source_contract(config, lifecycle)
        )
    return contract


def _policy_source_contract(
    config: PQNConfig,
    lifecycle: Mapping[str, object],
    *,
    initial_model_head_digest: Optional[str] = None,
    initial_snapshot_state_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Static source identity for new corrected-v3 lifecycle checkpoints."""
    environment_episode = config.episode_reset_mode == "per_env_autoreset_v1"
    return {
        "schema_version": "pqn-rollout-policy-source/v1",
        "rollout_policy_mode": config.rollout_policy_mode,
        "fixed_policy_identity": config.fixed_policy_identity,
        "assignment_lifetime": "environment_episode" if environment_episode else "batch_episode",
        "lease_lifetime": "environment_episode" if environment_episode else "batch_episode",
        "pool_admission_mode": config.pool_admission_mode,
        "initial_opponent_checkpoint_sha256": config.initial_opponent_checkpoint_sha256,
        "initial_opponent_model_head_digest": initial_model_head_digest,
        "initial_opponent_snapshot_state_sha256": initial_snapshot_state_sha256,
        "episode_lifecycle_contract_digest": canonical_digest(lifecycle),
    }


def pqn_optimizer_contract(
    config: PQNConfig, optimizer_state: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """Describe expected Adam, or the actual serialized groups for a checkpoint."""
    expected_group = {
        "lr": config.lr,
        "eps": config.adam_eps,
        "betas": [0.9, 0.999],
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }
    groups = [expected_group]
    if optimizer_state is not None:
        raw_groups = optimizer_state.get("param_groups")
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError("continuation requires optimizer param_groups")
        groups = []
        for raw in raw_groups:
            if not isinstance(raw, dict) or set(raw) - set(expected_group) - {"params"}:
                raise ValueError("unknown optimizer param_group fields")
            if not isinstance(raw.get("betas"), (tuple, list)):
                raise ValueError("invalid optimizer param_group betas")
            group = {key: raw.get(key) for key in expected_group}
            group["betas"] = list(raw["betas"])
            groups.append(group)
    return {
        "version": "pqn-adam-huber-v1",
        "algorithm": "Adam",
        "param_groups": groups,
        "loss": {"name": "smooth_l1", "beta": 1.0, "reduction": "mean"},
        "grad_clip": {"norm_type": 2.0, "max_norm": config.grad_clip},
        "td_target_clip": None,
        "target_gradient": "detached",
        "zero_grad_set_to_none": True,
    }


@dataclass
class PQNTelemetry:
    """Per-update telemetry snapshot.

    Attributes:
        update: Update index (0-based).
        agent_steps: Cumulative hero agent-steps seen.
        loss: Mean Huber loss over this update's minibatches.
        grad_norm: Mean pre-clip grad norm over minibatches.
        mean_abs_q: Mean |Q| over hero transitions.
        max_abs_q: Max |Q| over hero transitions.
        epsilon: ε used during the rollout.
        mean_reward: Mean hero per-step reward over the rollout.
        action_entropy: Entropy (nats) of the hero action histogram.
        hero_kills: Exact valid hero kill credits in this rollout.
        hero_deaths: Exact valid hero deaths in this rollout.
        completed_episodes: Environments newly completed during this rollout.
        episode_reset_count: Cumulative batch resets before collecting this rollout.
        boost_fraction: Fraction of hero steps that engaged boost.
        pool_size: Current opponent-pool size.
        eligible_hero_transitions: Real hero transitions eligible for SGD.
        sampled_transition_draws: Total transition draws across SGD batches.
        unique_sampled_transitions: Distinct eligible transitions drawn by SGD.
        optimizer_steps: SGD optimizer steps completed this update.
        valid_slot_fraction: Fraction of rollout slots that were real transitions.
        raw_action_entropy: Entropy of unique valid hero rollout actions before augmentation.
        raw_action_mode: Most frequent valid hero rollout action, or ``-1`` when empty.
        action_collapse_streak: Consecutive low-entropy update count.
        action_collapse_evidence_samples: Real eligible samples in that streak.
    """

    update: int
    agent_steps: int
    loss: float
    grad_norm: float
    mean_abs_q: float
    max_abs_q: float
    epsilon: float
    mean_reward: float
    action_entropy: float
    legacy_kills_per_rollout: Optional[float]
    boost_fraction: float
    pool_size: int
    eligible_hero_transitions: int = 0
    sampled_transition_draws: int = 0
    unique_sampled_transitions: int = 0
    optimizer_steps: int = 0
    valid_slot_fraction: float = 0.0
    raw_action_entropy: float = 0.0
    raw_action_mode: int = -1
    action_collapse_streak: int = 0
    action_collapse_evidence_samples: int = 0
    hero_kills: int = 0
    hero_deaths: int = 0
    completed_episodes: int = 0
    valid_transitions: int = 0
    rollout_capacity: int = 0
    hero_eligible_fraction: float = 0.0
    policy_exposure: Optional[Dict[str, int]] = None
    raw_action_counts: Optional[List[int]] = None
    episode_reset_count: int = 0


@dataclass
class PQNPerEnvTelemetry(PQNTelemetry):
    """Corrected-v3 lifecycle telemetry, separate from the legacy row schema."""

    episode_reset_mode: str = "batch_barrier_v1"
    episode_seed_mode: str = "continuous_env_rng_v1"
    episode_ids: Optional[List[int]] = None
    episode_reset_counts: Optional[List[int]] = None
    reset_env_indices: Optional[List[int]] = None
    episode_world_seeds: Optional[List[int]] = None
    episode_policy_ids: Optional[List[List[int]]] = None
    episode_policy_identities: Optional[Dict[str, str]] = None


class TripwireError(RuntimeError):
    """Raised when a training tripwire fires, retaining its class and evidence."""

    def __init__(
        self,
        message: str,
        telemetry: Optional[PQNTelemetry] = None,
        *,
        incident_class: str = "unknown",
        incident: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.telemetry = telemetry
        self.incident_class = incident_class
        self.incident = incident or {}

    @property
    def is_finite_alarm(self) -> bool:
        """Whether the flagged model remains finite and is safe to archive separately."""
        return self.incident_class in {"max_abs_q", "action_collapse"}


def flip_augment(
    tactical: torch.Tensor,
    strategic: torch.Tensor,
    scalars: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Horizontal-flip augmentation for a batch of raster transitions.

    Mirrors the ego frame left<->right: flip the raster COLUMN axis (last dim),
    swap the lateral wall-distance scalars, negate the lateral ego scalars, and
    remap actions (0<->2, 3<->5). Ahead/behind and distance signals are
    flip-invariant and untouched.

    Args:
        tactical: ``(N, 9, 31, 31)`` float tensor.
        strategic: ``(N, 3, 25, 25)`` float tensor.
        scalars: ``(N, 26)`` float tensor.
        actions: ``(N,)`` long action indices.

    Returns:
        Flipped ``(tactical, strategic, scalars, actions)``.
    """
    tac = torch.flip(tactical, dims=[-1])
    strat = torch.flip(strategic, dims=[-1])
    scal = scalars.clone()
    # Swap left/right wall-distance scalars.
    i, j = _WALL_LR_SWAP
    scal[:, [i, j]] = scal[:, [j, i]]
    # Negate lateral ego scalars.
    for k in _LATERAL_NEGATE_IDX:
        scal[:, k] = -scal[:, k]
    flip_action = torch.as_tensor(_FLIP_ACTION, device=actions.device)
    acts = flip_action[actions]
    return tac, strat, scal, acts


class PQNTrainer:
    """Synchronous Q(lambda) trainer with pool self-play (blueprint P3).

    Owns the network, optimizer, sim, and opponent pool. Call :meth:`update`
    repeatedly (or :meth:`train` for a loop). Each :meth:`update` performs one
    rollout + several minibatch SGD steps and returns a :class:`PQNTelemetry`.

    Args:
        config: Trainer configuration.
        network: Optional pre-built network (default: fresh
            :class:`RasterDuelingNetwork`).
        device: Compute device (default: CPU — the raster net is small enough to
            train on CPU for smoke; use CUDA for full runs).
        fixed_policy: Optional immutable common opponent source for non-hero
            slots. Its sparse action protocol and identity are recorded.
    """

    def __init__(
        self,
        config: PQNConfig,
        network: Optional[RasterDuelingNetwork] = None,
        device: Optional[torch.device] = None,
        fixed_policy: Optional[FixedPolicySource] = None,
    ) -> None:
        self.cfg = config
        self.fixed_policy = fixed_policy
        if (fixed_policy is None) != (config.rollout_policy_mode != "fixed"):
            raise ValueError("fixed_policy injection must match rollout_policy_mode")
        if fixed_policy is not None and fixed_policy.identity != config.fixed_policy_identity:
            raise ValueError("fixed_policy identity must match fixed_policy_identity")
        self.device = device or torch.device("cpu")
        self.network = (network or RasterDuelingNetwork()).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.lr, eps=config.adam_eps
        )
        # Legacy checkpoints retain the historical mutable FIFO pool. Corrected
        # v3 uses stable identities and a lease that prevents a live assignment
        # from being replaced by a later snapshot admission.
        pool_capacity = 0 if config.rollout_policy_mode == "fixed" else config.pool_capacity
        self.pool = (
            PinnedOpponentPool(capacity=pool_capacity, device=self.device)
            if config.recipe == "corrected-v3"
            else OpponentPool(capacity=pool_capacity, device=self.device)
        )
        self.rng = np.random.default_rng(config.seed)
        # The default deliberately aliases the rollout generator: existing runs
        # retain their exact RNG ordering. An explicit seed isolates optimizer
        # sampling/augmentation from rollout policy assignment and exploration.
        self.sgd_rng = (
            self.rng if config.sgd_seed is None else np.random.default_rng(config.sgd_seed)
        )
        self._initial_rng_identity = (
            canonical_digest(self.rng.bit_generator.state),
            canonical_digest(self.sgd_rng.bit_generator.state),
        )

        sim_cfg = BatchSimConfig(
            num_envs=config.num_envs,
            num_snakes=config.num_snakes,
            game_width=config.game_width,
            game_height=config.game_height,
            segment_size=config.segment_size,
            wall_thickness=config.wall_thickness,
            initial_food=config.initial_food,
            max_food=config.max_food,
            min_boost_length=config.min_boost_length,
            boost_length_cost_frames=config.boost_length_cost_frames,
            mechanics_version=config.mechanics_version,
            gamma=config.gamma,
            arena_type=config.arena_type,
            frame_rate=config.frame_rate,
            max_capacity=config.max_capacity,
            kill_scale=config.kill_scale,
            death_value=config.death_value,
        )
        derived_episode_rng = config.episode_seed_mode == EPISODE_SEED_DERIVED
        initial_world_seeds = (
            [
                derive_seed(config.seed, f"pqn/world/env/{env}/episode/0")
                for env in range(config.num_envs)
            ]
            if derived_episode_rng
            else [config.seed + env for env in range(config.num_envs)]
        )
        self.sim = BatchSim(
            sim_cfg,
            seeds=initial_world_seeds,
            train_mode=True,
        )

        self.update_idx = 0
        self.agent_steps = 0
        self._resume_mode = "fresh"
        self._resume_state: Dict[str, object] = {
            "environment": "fresh",
            "pool": "fresh",
            "rng": "fresh",
        }
        # Corrected-v3 assigns policy sources once per batch episode. A world
        # that ends early is frozen by BatchSim's active mask until the batch
        # reaches the shared reset boundary.
        self._episode_policy_ids: Optional[np.ndarray] = None
        self._episode_lease: Optional[OpponentLease] = None
        self._episode_ids = np.zeros(
            config.num_envs, dtype=np.uint64 if derived_episode_rng else np.int64
        )
        self._episode_finished_env = np.zeros(config.num_envs, dtype=bool)
        self._episode_leases: List[Optional[OpponentLease]] = [None] * config.num_envs
        self._episode_reset_counts = np.zeros(config.num_envs, dtype=np.uint64)
        self._episode_world_seeds = np.asarray(initial_world_seeds, dtype=np.uint64)
        self._action_rngs: Optional[Dict[int, np.random.Generator]] = (
            {
                env: np.random.default_rng(
                    derive_seed(config.seed, f"pqn/action/env/{env}/episode/0")
                )
                for env in range(config.num_envs)
            }
            if derived_episode_rng
            else None
        )
        self._initial_action_rng_identities: Optional[Dict[int, str]] = (
            {
                env: canonical_digest(generator.bit_generator.state)
                for env, generator in self._action_rngs.items()
            }
            if self._action_rngs is not None
            else None
        )
        self._initial_assignment_complete = False
        self._initial_snapshot_preloaded = False
        self._last_reset_env_indices = np.empty(0, dtype=np.int64)
        self._initial_opponent_model_head_digest: Optional[str] = None
        self._initial_opponent_snapshot_state_sha256: Optional[str] = None
        self._last_policy_source = self._policy_source_descriptor()
        self._episode_reset_count = 0
        self._last_sgd_sampling = {
            "eligible_hero_transitions": 0,
            "sampled_transition_draws": 0,
            "unique_sampled_transitions": 0,
            "optimizer_steps": 0,
        }
        self._action_collapse_streak = 0
        self._action_collapse_evidence_samples = 0
        self._action_collapse_raw_action_mode: Optional[int] = None
        self.last_telemetry: Optional[PQNTelemetry] = None
        self._last_known_good_state: Dict[str, object] = {}
        self._last_known_good_snapshot_bytes = 0
        self.refresh_numeric_recovery_state()

    def _policy_source_descriptor(self) -> Dict[str, object]:
        """Return the realized non-hero rollout source without serializing code."""
        if self.fixed_policy is not None:
            return {"mode": "fixed", "identity": self.fixed_policy.identity}
        return {"mode": "snapshot_pool", "identity": "episode-assigned"}

    def _static_policy_source_contract(self) -> Dict[str, object]:
        """Build the checkpoint/telemetry source contract from realized trainer state."""
        lifecycle = build_pqn_episode_lifecycle_contract(
            self.cfg.episode_reset_mode, self.cfg.episode_seed_mode
        )
        return _policy_source_contract(
            self.cfg,
            lifecycle,
            initial_model_head_digest=self._initial_opponent_model_head_digest,
            initial_snapshot_state_sha256=self._initial_opponent_snapshot_state_sha256,
        )

    def _derived_policy_identities(self, policy_ids: np.ndarray) -> Dict[str, str]:
        """Return exact pinned identities for the active derived assignment grid."""
        if self.fixed_policy is not None:
            return {"0": self.fixed_policy.identity}
        if not isinstance(self.pool, PinnedOpponentPool):
            raise RuntimeError("derived corrected-v3 rollout requires PinnedOpponentPool")
        identities: Dict[str, str] = {}
        for env, policy_row in enumerate(policy_ids):
            lease = self._episode_leases[env]
            frozen = {int(value) for value in policy_row if int(value) != HERO_POLICY_ID}
            if frozen and lease is None:
                raise RuntimeError("derived frozen policy row has no active opponent lease")
            for policy_id in frozen:
                assert lease is not None
                identity = lease.identities.get(policy_id)
                if identity is None:
                    raise RuntimeError("derived policy row names an identity outside its lease")
                previous = identities.setdefault(str(policy_id), identity)
                if previous != identity:
                    raise RuntimeError(
                        "derived policy identity disagrees across active lane leases"
                    )
        return identities

    def _episode_policy_identities(self) -> Dict[str, str]:
        """Return active assignment identities without fabricating legacy labels."""
        if self._episode_policy_ids is None:
            return {}
        if self._uses_derived_episode_rng:
            return self._derived_policy_identities(self._episode_policy_ids)
        if self._episode_lease is not None:
            return {
                str(policy_id): identity
                for policy_id, identity in self._episode_lease.identities.items()
            }
        if self.fixed_policy is not None:
            return {"0": self.fixed_policy.identity}
        return {}

    @property
    def _uses_derived_episode_rng(self) -> bool:
        return self.cfg.episode_seed_mode == EPISODE_SEED_DERIVED

    @property
    def _uses_per_env_autoreset(self) -> bool:
        return self.cfg.episode_reset_mode == EPISODE_RESET_PER_ENV_AUTORESET

    def _assign_episode_rows(self, env_indices: Sequence[int]) -> None:
        """Assign and pin selected corrected-v3 environment episode rows."""
        if self._episode_policy_ids is None:
            self._episode_policy_ids = np.full(
                (self.cfg.num_envs, self.cfg.num_snakes), HERO_POLICY_ID, dtype=np.int64
            )
        pool_ids = [0] if self.fixed_policy is not None else self.pool.policy_ids()
        self._episode_policy_ids = assign_policy_ids_per_env(
            self._episode_policy_ids,
            env_indices=env_indices,
            episode_ids=self._episode_ids,
            pool_ids=pool_ids,
            hero_frac=self.cfg.hero_frac,
            run_seed=self.cfg.seed,
            hero_slot0=True,
        )
        if self.fixed_policy is None:
            if not isinstance(self.pool, PinnedOpponentPool):
                raise RuntimeError("corrected-v3 requires PinnedOpponentPool")
            for env in env_indices:
                self._episode_leases[env] = self.pool.acquire(self._episode_policy_ids[env])

    def _prepare_derived_rollout(self) -> int:
        """Apply delayed selected-lane reset/reassignment before a derived rollout."""
        if not self._initial_assignment_complete:
            if self.cfg.initial_opponent_checkpoint_sha256 and not self._initial_snapshot_preloaded:
                raise RuntimeError("configured initial opponent checkpoint was not preloaded")
            self._assign_episode_rows(range(self.cfg.num_envs))
            self._initial_assignment_complete = True
            self._last_reset_env_indices = np.empty(0, dtype=np.int64)
            return 0
        if self._uses_per_env_autoreset:
            selected = np.flatnonzero(self._episode_finished_env)
        elif self._episode_over():
            selected = np.arange(self.cfg.num_envs, dtype=np.int64)
        else:
            selected = np.empty(0, dtype=np.int64)
        if not selected.size:
            self._last_reset_env_indices = np.empty(0, dtype=np.int64)
            return 0
        selected_indices = [int(value) for value in selected]
        completed_now = self.sim.population_floor_reached() | (
            self.sim.frame >= self.cfg.max_frames
        )
        if not bool(completed_now[selected].all()):
            raise RuntimeError(
                "pending per-environment reset lane is not at a completed final state"
            )
        for env in selected_indices:
            lease = self._episode_leases[env]
            if lease is not None:
                lease.close()
                self._episode_leases[env] = None
        self._episode_ids[selected] += np.uint64(1)
        self._episode_reset_counts[selected] += np.uint64(1)
        seeds = [
            derive_seed(self.cfg.seed, f"pqn/world/env/{env}/episode/{int(self._episode_ids[env])}")
            for env in selected_indices
        ]
        self.sim.reset_envs(self._episode_finished_env, seeds=seeds)
        for env, seed in zip(selected_indices, seeds):
            self._episode_world_seeds[env] = np.uint64(seed)
            assert self._action_rngs is not None
            self._action_rngs[env] = np.random.default_rng(
                derive_seed(
                    self.cfg.seed, f"pqn/action/env/{env}/episode/{int(self._episode_ids[env])}"
                )
            )
        self._assign_episode_rows(selected_indices)
        self._episode_finished_env[selected] = False
        self._episode_reset_count += 1
        self._last_reset_env_indices = selected.astype(np.int64, copy=True)
        return len(selected_indices)

    def preload_opponent_snapshot(
        self,
        source_network: RasterDuelingNetwork,
        *,
        checkpoint_sha256: str,
        model_head_digest: str,
    ) -> int:
        """Preload the B5 immutable opponent before episode-zero assignment.

        The caller proves checkpoint-byte identity.  This method verifies the
        declared source identity, model head, and fresh trainer state before
        adding the detached snapshot to the normal pinned pool.
        """
        if self.cfg.initial_opponent_checkpoint_sha256 is None:
            raise ValueError("no initial opponent checkpoint is configured")
        if checkpoint_sha256 != self.cfg.initial_opponent_checkpoint_sha256:
            raise ValueError("preload checkpoint hash differs from configured identity")
        expected_head = ModelHeadContract("pqn", "dueling_q", 6).digest
        if model_head_digest != expected_head or source_network.output_size != 6:
            raise ValueError("preload source does not have the raster PQN six-action model head")
        if (
            not isinstance(self.pool, PinnedOpponentPool)
            or len(self.pool) != 0
            or self.update_idx != 0
            or self.agent_steps != 0
            or self._initial_assignment_complete
            or bool(self._episode_ids.any())
            or bool(self._episode_finished_env.any())
            or bool(self.sim.frame.any())
            or any(lease is not None for lease in self._episode_leases)
        ):
            raise RuntimeError(
                "initial opponent preload is only valid before episode-zero assignment"
            )
        policy_id = self.pool.add_snapshot(source_network)
        if policy_id is None:
            raise RuntimeError("initial opponent snapshot could not be admitted")
        self._initial_snapshot_preloaded = True
        self._initial_opponent_model_head_digest = model_head_digest
        self._initial_opponent_snapshot_state_sha256 = self.pool.snapshot_hash(policy_id)
        return policy_id

    def close(self) -> None:
        """Release every owned episode lease, attempting all cleanup on failure."""
        leases: List[OpponentLease] = []
        if self._episode_lease is not None:
            leases.append(self._episode_lease)
            self._episode_lease = None
        for index, lease in enumerate(self._episode_leases):
            if lease is not None:
                leases.append(lease)
                self._episode_leases[index] = None
        errors: List[BaseException] = []
        for lease in leases:
            try:
                lease.close()
            except BaseException as exc:  # cleanup must attempt later leases
                errors.append(exc)
        if isinstance(self.pool, PinnedOpponentPool) and self.pool.active_pin_count != 0:
            errors.append(RuntimeError("opponent-pool pins remain after trainer cleanup"))
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("PQN trainer lease cleanup failed", errors)

    # -- numeric recovery --------------------------------------------------
    @staticmethod
    def _clone_for_recovery(value: Any) -> Any:
        """Clone nested training state onto CPU so failed device state is replaceable."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: PQNTrainer._clone_for_recovery(item) for key, item in value.items()}
        if isinstance(value, list):
            return [PQNTrainer._clone_for_recovery(item) for item in value]
        if isinstance(value, tuple):
            return tuple(PQNTrainer._clone_for_recovery(item) for item in value)
        return value

    @staticmethod
    def _state_tensor_bytes(value: Any) -> int:
        """Return tensor storage bytes in a nested state snapshot."""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(PQNTrainer._state_tensor_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(PQNTrainer._state_tensor_bytes(item) for item in value)
        return 0

    @staticmethod
    def _nonfinite_tensor_count(value: Any) -> int:
        """Count non-finite numeric values recursively without assuming a schema."""
        if isinstance(value, torch.Tensor):
            if not (torch.is_floating_point(value) or torch.is_complex(value)):
                return 0
            return int((~torch.isfinite(value)).sum().item())
        if isinstance(value, bool):
            return 0
        if isinstance(value, Real):
            return int(not np.isfinite(value))
        if isinstance(value, Complex):
            return int(not np.isfinite(value.real) or not np.isfinite(value.imag))
        if isinstance(value, dict):
            return sum(PQNTrainer._nonfinite_tensor_count(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(PQNTrainer._nonfinite_tensor_count(item) for item in value)
        return 0

    @classmethod
    def validate_checkpoint_numeric_state(cls, checkpoint: Dict[str, object]) -> None:
        """Reject a resume payload with non-finite weights or optimizer moments.

        This runs before a checkpoint is installed, so a failed resume cannot
        mutate a fresh trainer and then turn poisoned tensors into its rollback
        point.
        """
        for key in ("dqn_state_dict", "optimizer_state_dict"):
            nonfinite = cls._nonfinite_tensor_count(checkpoint.get(key))
            if nonfinite:
                raise ValueError(
                    f"resume checkpoint {key} has {nonfinite} non-finite tensor values"
                )

    def refresh_numeric_recovery_state(self) -> None:
        """Capture a finite network plus optimizer state as the rollback point.

        The snapshot is CPU-resident.  It contains weights and Adam moments so a
        post-step failure cannot leave either half of training advanced.  Its
        byte count is exposed in incident/checkpoint metadata rather than being
        an undocumented accelerator-memory cost.
        """
        state = {
            "network": self._clone_for_recovery(self.network.state_dict()),
            "optimizer": self._clone_for_recovery(self.optimizer.state_dict()),
        }
        nonfinite = self._nonfinite_tensor_count(state)
        if nonfinite:
            raise RuntimeError(
                f"cannot capture a numeric recovery state with {nonfinite} non-finite tensor values"
            )
        self._last_known_good_state = state
        self._last_known_good_snapshot_bytes = self._state_tensor_bytes(state)

    def _restore_numeric_recovery_state(self) -> None:
        """Restore the last successful weights and optimizer moments after a fault."""
        if not self._last_known_good_state:
            raise RuntimeError("numeric recovery requested before a good state was captured")
        self.network.load_state_dict(self._last_known_good_state["network"])
        self.optimizer.load_state_dict(self._last_known_good_state["optimizer"])

    def _raise_numeric_tripwire(self, stage: str, nonfinite_count: int) -> None:
        """Roll back and raise a classified numeric incident for the current minibatch."""
        self._restore_numeric_recovery_state()
        raise TripwireError(
            f"non-finite {stage} during SGD at update {self.update_idx}",
            incident_class=f"nonfinite_{stage}",
            incident={
                "stage": stage,
                "recovered": True,
                "recovery_snapshot_bytes": self._last_known_good_snapshot_bytes,
                "nonfinite_counts": {stage: nonfinite_count},
            },
        )

    def numeric_recovery_metadata(self) -> Dict[str, object]:
        """Return measured recovery strategy metadata for durable incident evidence."""
        return {
            "strategy": "cpu_last_known_good_network_and_optimizer",
            "snapshot_bytes": self._last_known_good_snapshot_bytes,
        }

    # -- ε schedule ---------------------------------------------------------
    def epsilon(self) -> float:
        """Current ε from the linear ladder (``eps_start`` -> ``eps_end``)."""
        frac = min(1.0, self.agent_steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    # -- observation --------------------------------------------------------
    def _sync(self) -> float:
        """Return a timestamp after flushing pending CUDA work (accurate timing)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def _current_obs(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Featurize the sim's current state into device float tensors + mask.

        Returns:
            ``(obs, mask)``: obs dict with ``tactical`` ``(E, S, 9, 31, 31)``,
            ``strategic`` ``(E, S, 3, 25, 25)``, ``scalars`` ``(E, S, 26)``; mask
            ``(E, S, 6)`` bool tensor on ``device``.
        """
        E, S = self.cfg.num_envs, self.cfg.num_snakes
        # Legacy v2 learned under the historical advisory mask. Corrected v3
        # records and consumes ENV's legal/advisory row-local resolution.
        mask_np = (
            self.sim.get_resolved_action_mask()
            if self.cfg.obs_spec == RASTER31V3
            else self.sim.get_action_mask()
        )  # (E, S, 6)
        if self.device.type == "cuda":
            # GPU featurizer: transfer only the compact sim state, build the
            # rasters on the (idle) GPU. Byte-identical tactical/strategic planes
            # and <=1e-4 scalars vs the NumPy path (parity-tested), so a
            # GPU-trained policy sees the same inputs the web app serves via the
            # NumPy featurizer. Returns (E, S, ...) tensors directly.
            state = obs_inputs_to_torch(
                self.sim,
                self.device,
                max_frames=self.cfg.max_frames,
                starvation_max=self.cfg.starvation_max,
                max_length=self.cfg.max_length,
            )
            obs_es = build_observations_gpu(state, obs_spec=self.cfg.obs_spec)
            obs_es = {
                "tactical": obs_es["tactical"],
                "strategic": obs_es["strategic"],
                "scalars": obs_es["scalars"],
            }
        else:
            inp = obs_inputs_from_batch_sim(
                self.sim,
                max_frames=self.cfg.max_frames,
                starvation_max=self.cfg.starvation_max,
                max_length=self.cfg.max_length,
            )
            obs = build_observations(inp, mask=mask_np, obs_spec=self.cfg.obs_spec)
            tensors = raster_tensors_from_obs(obs, device=self.device)
            # Reshape flat (E*S, ...) back to (E, S, ...).
            obs_es = {
                "tactical": tensors["tactical"].reshape(E, S, *TACTICAL_SHAPE),
                "strategic": tensors["strategic"].reshape(E, S, *STRATEGIC_SHAPE),
                "scalars": tensors["scalars"].reshape(E, S, SCALARS_DIM),
            }
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
        return obs_es, mask

    # -- episode boundary ---------------------------------------------------
    def _episode_over(self) -> bool:
        """True when the batch's episode has ended and the sim must be reset.

        Two blueprint episode-end conditions, evaluated over the whole batch
        because :meth:`BatchSim.reset` has no per-env form:

        * The v2 train-mode POPULATION FLOOR (``population_floor_reached``, i.e.
          fewer than 3 snakes alive) — the same signal the Apex actor ends an
          episode on. Requires ALL envs to have floored: the reset is batch-wide,
          so firing on the first env would truncate every other env's still-live
          episode and couple episode lengths across the batch. Steps taken in an
          env between its own floor and the batch reset are excluded from
          training instead (see ``valid`` in :meth:`_rollout`).
        * ``cfg.max_frames``. Envs step in lockstep from a shared reset, so the
          frame counter is identical across the batch and this fires for all envs
          at once.

        Returns:
            True iff the sim should be reset before the next rollout.
        """
        if int(self.sim.frame.max()) >= self.cfg.max_frames:
            return True
        return bool(self.sim.population_floor_reached().all())

    # -- rollout ------------------------------------------------------------
    def _rollout(self) -> Dict[str, object]:
        """Collect a ``T``-step self-play rollout of hero + frozen transitions.

        Resets the sim first if the previous episode ended. Resetting only on a
        rollout BOUNDARY (never mid-rollout) keeps every step of a rollout inside
        one episode, so no stored ``s_{t+1}`` obs/mask, bootstrap, or backward
        return can ever cross a reset.

        Returns:
            Dict of stacked rollout tensors/arrays (see below). Spatial obs are
            kept per-slot; termination bookkeeping is per (t, e, s).
        """
        cfg = self.cfg
        E, S = cfg.num_envs, cfg.num_snakes
        eps = self.epsilon()

        completed_episodes = 0
        if self._uses_derived_episode_rng:
            completed_episodes = self._prepare_derived_rollout()
        elif self._episode_over():
            if self._episode_lease is not None:
                self._episode_lease.close()
                self._episode_lease = None
            self.sim.reset()
            self._episode_policy_ids = None
            self._episode_reset_count += 1
            self._episode_ids += 1
            if self.cfg.recipe == "corrected-v3":
                self._episode_reset_counts += np.uint64(1)
                self._last_reset_env_indices = np.arange(E, dtype=np.int64)
            self._episode_finished_env[:] = False
            completed_episodes = E
        elif not self._uses_derived_episode_rng:
            self._last_reset_env_indices = np.empty(0, dtype=np.int64)

        # A rollout may not cross the frame-cap episode boundary.  In
        # particular, when ``max_frames`` is not divisible by ``rollout_len``,
        # collect the shorter tail here rather than stepping once past the cap
        # and repairing the buffers afterwards.  The latter leaks an
        # episode-progress scalar above 1 and creates a transition that belongs
        # to neither episode.
        remaining_frames = cfg.max_frames - int(self.sim.frame.max())
        T = (
            cfg.rollout_len
            if self._uses_per_env_autoreset
            else min(cfg.rollout_len, remaining_frames)
        )
        if T <= 0:
            raise RuntimeError("rollout started at or beyond the episode frame cap")

        if self.cfg.obs_spec == RASTER31V3 and self._episode_policy_ids is not None:
            policy_ids = self._episode_policy_ids
        else:
            pool_ids = [0] if self.fixed_policy is not None else self.pool.policy_ids()
            policy_ids = assign_policy_ids(E, S, pool_ids, cfg.hero_frac, self.rng, hero_slot0=True)
            if self.cfg.obs_spec == RASTER31V3:
                self._episode_policy_ids = policy_ids.copy()
                if not isinstance(self.pool, PinnedOpponentPool):
                    raise RuntimeError("corrected-v3 requires PinnedOpponentPool")
                if self.fixed_policy is None:
                    self._episode_lease = self.pool.acquire(policy_ids.reshape(-1))

        acting_pool = self._episode_lease or self.pool

        tac_buf = torch.zeros((T, E, S, *TACTICAL_SHAPE), dtype=torch.float32, device=self.device)
        strat_buf = torch.zeros(
            (T, E, S, *STRATEGIC_SHAPE), dtype=torch.float32, device=self.device
        )
        scal_buf = torch.zeros((T, E, S, SCALARS_DIM), dtype=torch.float32, device=self.device)
        act_buf = np.zeros((T, E, S), dtype=np.int64)
        rew_buf = np.zeros((T, E, S), dtype=np.float64)
        done_buf = np.zeros((T, E, S), dtype=bool)
        trapped_buf = np.zeros((T, E, S), dtype=bool)
        # valid_step[t,e,s]: step t is a REAL transition for this slot — the slot
        # was ALIVE at the start of step t AND its env's episode was still live.
        # True on the death step itself (a real terminal transition), False for
        # every post-death "zombie" step (train_mode never respawns) and for
        # every step an env takes after its population floor fired (its episode
        # is over; the batch cannot reset until all envs have floored). Invalid
        # steps are dropped from the loss, from the telemetry, and from the
        # backward-return carry.
        valid_buf = np.zeros((T, E, S), dtype=bool)
        # next-mask per step for masked-max bootstrap (mask of s_{t+1}).
        next_mask_buf = torch.zeros((T, E, S, 6), dtype=torch.bool, device=self.device)
        boost_buf = np.zeros((T, E, S), dtype=bool)
        # Hero Q(s_t) over the whole grid, kept from the acting forward. The
        # learner's bootstrap for step t is the masked max of Q(s_{t+1}), and
        # s_{t+1}'s obs IS step t+1's stored obs — already forwarded here under
        # the same weights, since the only optimizer step runs after the rollout.
        # _compute_targets reuses these instead of re-forwarding the rollout.
        hero_q_buf = torch.zeros((T, E, S, 6), dtype=torch.float32, device=self.device)
        kill_buf = np.zeros((T, E, S), dtype=np.int64)
        death_buf = np.zeros((T, E, S), dtype=bool)

        prof = self.cfg.profile
        feat_t = fwd_t = sim_t = 0.0
        newly_completed_total = 0

        for t in range(T):
            if prof:
                t0 = self._sync()
            obs, mask = self._current_obs()
            if prof:
                t1 = self._sync()
                feat_t += t1 - t0
            # ``batched_act`` owns the hero epsilon path. A fixed common source
            # is deliberately invoked through its sparse protocol for non-hero
            # rows, never smuggled into the mutable snapshot pool.
            act_ids = (
                np.where(policy_ids == HERO_POLICY_ID, HERO_POLICY_ID, HERO_POLICY_ID)
                if self.fixed_policy is not None
                else policy_ids
            )
            if self._uses_derived_episode_rng:
                assert self._action_rngs is not None
                alive_at_entry = self.sim.get_alive()
                live_env_at_entry = ~self.sim.population_floor_reached()
                live_env_at_entry &= self.sim.frame < cfg.max_frames
                live_env_at_entry &= ~self._episode_finished_env
                decisions: PerEnvExplorationDecisions = sample_per_env_exploration(
                    policy_ids,
                    mask.detach().cpu().numpy(),
                    live_env_at_entry[:, None] & alive_at_entry,
                    eps,
                    self._action_rngs,
                )
                actions, hero_q = batched_act(
                    self.network,
                    acting_pool,
                    act_ids,
                    obs,
                    mask,
                    eps,
                    None,
                    self.device,
                    exploration=decisions,
                )
            else:
                actions, hero_q = batched_act(
                    self.network,
                    acting_pool,
                    act_ids,
                    obs,
                    mask,
                    0.0 if self.fixed_policy else eps,
                    self.rng,
                    self.device,
                )
            if self.fixed_policy is not None and not self._uses_derived_episode_rng:
                hero_slots = np.argwhere(policy_ids == HERO_POLICY_ID)
                if hero_slots.size:
                    explore = self.rng.random(len(hero_slots)) < eps
                    for row, is_exploring in zip(hero_slots, explore):
                        if is_exploring:
                            choices = np.flatnonzero(mask[row[0], row[1]].cpu().numpy())
                            if choices.size:
                                actions[row[0], row[1]] = choices[self.rng.integers(len(choices))]
            if self.fixed_policy is not None:
                frozen_slots = np.argwhere(policy_ids != HERO_POLICY_ID)
                if frozen_slots.size:
                    frozen_masks = mask.cpu().numpy()[frozen_slots[:, 0], frozen_slots[:, 1]]
                    actions[frozen_slots[:, 0], frozen_slots[:, 1]] = self.fixed_policy.actions(
                        frozen_masks, self.sim, frozen_slots
                    )
            if prof:
                t2 = self._sync()
                fwd_t += t2 - t1

            hero_q_buf[t] = hero_q
            tac_buf[t] = obs["tactical"]
            strat_buf[t] = obs["strategic"]
            scal_buf[t] = obs["scalars"]
            act_buf[t] = actions
            # A slot is TRAPPED if it currently has no valid action (mask all
            # False) but is alive (so it is not yet a death transition).
            alive = self.sim.get_alive()
            # Read the floor BEFORE step(): an env whose episode ended on an
            # EARLIER step contributes no transitions, but the step that trips
            # the floor is itself the episode's last real transition.
            live_env = ~self.sim.population_floor_reached()
            live_env &= self.sim.frame < cfg.max_frames
            live_env &= ~self._episode_finished_env
            # ``trapped`` is a legacy target branch. A corrected-v3 resolved
            # legal mask cannot turn an alive row into a synthetic terminal.
            if self.cfg.obs_spec != RASTER31V3:
                no_valid = ~mask.any(dim=2).cpu().numpy()
                trapped_buf[t] = no_valid & alive

            if prof:
                t3 = self._sync()
            if self.cfg.obs_spec == RASTER31V3:
                self.sim.step(actions, active_env_mask=live_env)
            else:
                self.sim.step(actions)
            if prof:
                sim_t += self._sync() - t3

            rew_buf[t] = self.sim.get_reward()
            # ENV reports validity for the step just executed. It includes a
            # death-causing action and excludes rows dead at entry.
            valid_buf[t] = self.sim.get_transition_valid() & live_env[:, None]
            done_buf[t] = self.sim.get_done()
            boost_buf[t] = self.sim.get_boosted_this_step()
            kill_buf[t] = self.sim.get_kill_credit()
            death_buf[t] = self.sim.get_done()

            completed_now = self.sim.population_floor_reached() | (self.sim.frame >= cfg.max_frames)
            newly_completed = completed_now & ~self._episode_finished_env
            self._episode_finished_env |= completed_now
            newly_completed_total += int(newly_completed.sum())

            # Mask of the NEXT state s_{t+1} for the bootstrap of non-terminal
            # transitions (already updated by step()).
            next_mask_buf[t] = torch.as_tensor(
                (
                    self.sim.get_resolved_action_mask()
                    if self.cfg.obs_spec == RASTER31V3
                    else self.sim.get_action_mask()
                ),
                dtype=torch.bool,
                device=self.device,
            )
            if (
                self.cfg.obs_spec == RASTER31V3
                and not self._uses_per_env_autoreset
                and bool(self._episode_finished_env.all())
            ):
                T = t + 1
                break

        # Final observation (s_T) for truncation bootstrap of the last step.
        final_obs, final_mask = self._current_obs()

        if prof:
            self._rollout_prof = {"featurize": feat_t, "forward": fwd_t, "sim": sim_t}

        return {
            "tactical": tac_buf[:T],
            "strategic": strat_buf[:T],
            "scalars": scal_buf[:T],
            "actions": act_buf[:T],
            "rewards": rew_buf[:T],
            "dones": done_buf[:T],
            "trapped": trapped_buf[:T],
            "valid": valid_buf[:T],
            "next_mask": next_mask_buf[:T],
            "hero_q": hero_q_buf[:T],
            "boost": boost_buf[:T],
            "kills": kill_buf[:T],
            "deaths": death_buf[:T],
            "policy_ids": policy_ids,
            "policy_identities": (
                self._derived_policy_identities(policy_ids)
                if self._uses_derived_episode_rng
                else (
                    {
                        str(policy_id): identity
                        for policy_id, identity in self._episode_lease.identities.items()
                    }
                    if self._episode_lease is not None
                    else (
                        {"0": self.fixed_policy.identity}
                        if self.fixed_policy is not None
                        else {
                            str(policy_id): f"legacy:{policy_id}"
                            for policy_id in self.pool.policy_ids()
                        }
                    )
                )
            ),
            "rollout_policy_source": (
                {
                    **self._policy_source_descriptor(),
                    "policy_source_contract_digest": canonical_digest(
                        self._static_policy_source_contract()
                    ),
                }
                if self.cfg.recipe == "corrected-v3"
                else self._policy_source_descriptor()
            ),
            "episode_reset_count": self._episode_reset_count,
            "episode_ids": self._episode_ids.copy(),
            "episode_reset_counts": self._episode_reset_counts.copy(),
            "episode_world_seeds": self._episode_world_seeds.copy(),
            "reset_env_indices": self._last_reset_env_indices.copy(),
            "completed_episodes": completed_episodes,
            "newly_completed_episodes": newly_completed_total,
            "batch_episode_finished": bool(self._episode_finished_env.all()),
            "final_obs": final_obs,
            "final_mask": final_mask,
            "epsilon": eps,
        }

    # -- Q(lambda) targets --------------------------------------------------
    def _compute_targets(self, roll: Dict[str, object]) -> torch.Tensor:
        """Compute per-slot Q(lambda) targets ``(T, E, S)`` (blueprint §3).

        Requires exactly ONE forward, on the final obs: the masked next-max for
        every other step reuses the hero Q the rollout already computed (see
        ``hero_q`` in :meth:`_rollout`). Backward recursion per agent stream:

            done_t:      G_t = r_t                       (death: no bootstrap)
            trapped_t:   G_t = r_t + gamma * death_value (trapped non-terminal)
            else:        boot = max_{valid} Q(s_{t+1})
                         next_G = G_{t+1} if t<T-1 and step t+1 is a REAL (valid)
                                  transition — including when step t+1 is the
                                  death step, whose G_{t+1}=r_{t+1} then flows
                                  back through the lambda channel;
                                  else boot   (truncation bootstrap at the
                                  rollout edge or across a post-death zombie step)
                         G_t = r_t + gamma * ((1-lambda)*boot + lambda*next_G)

        Non-transitions (``valid[t]`` False: a slot already dead at step entry, or
        any step in an env whose episode already ended) are excluded from training
        (:meth:`_sgd`) and their return is NOT carried into the backward scan:
        ``next_g`` for a step whose successor is one of them falls back to the
        truncation bootstrap. This is what makes an episode's last real step
        truncate rather than chain into the steps that follow it.

        Args:
            roll: Rollout dict from :meth:`_rollout`.

        Returns:
            ``(T, E, S)`` float tensor of Q(lambda) targets on ``device``.
        """
        cfg = self.cfg
        E, S = cfg.num_envs, cfg.num_snakes
        T = int(np.asarray(roll["rewards"]).shape[0])
        gamma, lam = cfg.gamma, cfg.lambda_
        neg_inf = torch.finfo(torch.float32).min

        rewards = torch.as_tensor(roll["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(roll["dones"], dtype=torch.bool, device=self.device)
        trapped = torch.as_tensor(roll["trapped"], dtype=torch.bool, device=self.device)
        valid = torch.as_tensor(roll["valid"], dtype=torch.bool, device=self.device)
        next_mask = roll["next_mask"]  # (T, E, S, 6) bool
        if self.cfg.obs_spec == RASTER31V3:
            empty_live_successor = valid & ~dones & ~next_mask.any(dim=3)
            if bool(empty_live_successor.any()):
                raise RuntimeError(
                    "corrected-v3 received an alive successor with no resolved legal action"
                )

        # Masked-max Q(s_{t+1}) for every stored step. s_{t+1}'s obs == the stored
        # obs of step t+1 (for t<T-1), which the rollout already forwarded under
        # these same weights, so its Q comes from roll["hero_q"][t+1] rather than
        # a second forward. Only s_T (final_obs, for t==T-1) is unseen.
        with torch.no_grad():
            q_final = self.network(
                roll["final_obs"]["tactical"].reshape(E * S, *TACTICAL_SHAPE),
                roll["final_obs"]["strategic"].reshape(E * S, *STRATEGIC_SHAPE),
                roll["final_obs"]["scalars"].reshape(E * S, SCALARS_DIM),
            ).reshape(1, E, S, 6)
            q_next = torch.cat([roll["hero_q"][1:], q_final], dim=0)  # (T, E, S, 6)
            q_masked = torch.where(next_mask, q_next, torch.full_like(q_next, neg_inf))
            mm = q_masked.max(dim=3).values
            # If no valid next action, masked-max is -inf; fall back to death
            # value (handled by the trapped branch below, but guard here too).
            boot = torch.where(next_mask.any(dim=3), mm, torch.full_like(mm, cfg.death_value))

        targets = torch.zeros((T, E, S), dtype=torch.float32, device=self.device)
        death_value = float(cfg.death_value)
        # Backward recursion. ``carry[e,s]`` holds ``G_{t+1}`` for the current
        # agent stream and ``succ_valid[e,s]`` records whether that successor step
        # (t+1) was a REAL (alive-at-entry) transition. ``next_g`` for step t uses
        # ``carry`` iff the successor is real — this INCLUDES a death successor,
        # whose ``G_{t+1}=r_{t+1}`` (reward alone) then propagates the terminal
        # return one step back through the lambda channel. When the successor is a
        # post-death "zombie" step (``succ_valid`` False), its return is discarded
        # and step t bootstraps (truncation-like). Death (``done_t``) and trapped
        # branches ignore ``next_g`` entirely.
        carry = torch.zeros((E, S), dtype=torch.float32, device=self.device)
        succ_valid = torch.zeros((E, S), dtype=torch.bool, device=self.device)
        for t in reversed(range(T)):
            r = rewards[t]
            done_t = dones[t]
            trapped_t = (
                trapped[t] & ~done_t
                if self.cfg.obs_spec != RASTER31V3
                else torch.zeros_like(done_t)
            )

            # boot_t is the masked-max Q(s_{t+1}) computed above.
            boot_t = boot[t]

            # next_G: G_{t+1} when the successor is a real (valid) in-rollout
            # transition, else the truncation bootstrap.
            if t < T - 1:
                next_g = torch.where(succ_valid, carry, boot_t)
            else:
                # Rollout edge: truncation -> bootstrap.
                next_g = boot_t

            g_interior = r + gamma * ((1.0 - lam) * boot_t + lam * next_g)
            g_trapped = r + gamma * death_value
            g_death = r

            g = torch.where(
                done_t,
                g_death,
                torch.where(trapped_t, g_trapped, g_interior),
            )
            targets[t] = g
            # Carry g down to step t-1: step t is t-1's successor. A zombie step
            # (not valid) does not seed the carry, so its garbage return never
            # chains into an earlier real transition.
            carry = g
            # A death transition is valid: its reward-only return is precisely
            # the successor return that a preceding live transition may mix
            # through lambda. Only a dead-at-entry/invalid row breaks carry.
            succ_valid = valid[t]
        return targets

    # -- SGD ----------------------------------------------------------------
    def _sgd(
        self, roll: Dict[str, object], targets: torch.Tensor
    ) -> Tuple[float, float, float, float, float]:
        """Minibatch SGD over HERO transitions (Huber, grad clip, flip aug).

        Args:
            roll: Rollout dict.
            targets: ``(T, E, S)`` Q(lambda) targets.

        Returns:
            ``(mean_loss, mean_grad_norm, mean_abs_q, max_abs_q, entropy)``.
        """
        cfg = self.cfg
        E, S = cfg.num_envs, cfg.num_snakes
        T = int(np.asarray(roll["actions"]).shape[0])
        policy_ids = roll["policy_ids"]  # (E, S)

        # Flatten hero transitions across (T, E, S). A slot is a hero transition
        # for a step iff its policy_id == HERO (fixed per rollout) AND it was
        # ALIVE at step entry (``valid``) — post-death zombie steps of a hero
        # slot carry bogus rewards/targets and must not enter the loss.
        hero_es = policy_ids == HERO_POLICY_ID  # (E, S)
        hero_mask_tes = (np.broadcast_to(hero_es[None], (T, E, S)) & roll["valid"]).reshape(-1)
        hero_idx = np.nonzero(hero_mask_tes)[0]
        if hero_idx.size == 0:
            self._last_sgd_sampling = {
                "eligible_hero_transitions": 0,
                "sampled_transition_draws": 0,
                "unique_sampled_transitions": 0,
                "optimizer_steps": 0,
            }
            return 0.0, 0.0, 0.0, 0.0, 0.0

        tac = roll["tactical"].reshape(T * E * S, *TACTICAL_SHAPE)
        strat = roll["strategic"].reshape(T * E * S, *STRATEGIC_SHAPE)
        scal = roll["scalars"].reshape(T * E * S, SCALARS_DIM)
        acts = torch.as_tensor(roll["actions"].reshape(-1), dtype=torch.long, device=self.device)
        tgt = targets.reshape(-1)

        hero_idx_t = torch.as_tensor(hero_idx, device=self.device)
        tac_h = tac[hero_idx_t]
        strat_h = strat[hero_idx_t]
        scal_h = scal[hero_idx_t]
        acts_h = acts[hero_idx_t]
        tgt_h = tgt[hero_idx_t]

        n = hero_idx.size
        losses: List[float] = []
        grad_norms: List[float] = []
        abs_q_vals: List[torch.Tensor] = []
        action_hist = torch.zeros(6, device=self.device)
        sampled_indices = set()
        sampled_draws = 0

        def run_minibatch(indices: np.ndarray) -> None:
            """Apply one optimizer step to the supplied eligible-index positions."""
            nonlocal action_hist, sampled_draws
            sampled_indices.update(indices.tolist())
            sampled_draws += int(indices.size)
            perm = torch.as_tensor(indices, device=self.device)
            mtac = tac_h[perm]
            mstrat = strat_h[perm]
            mscal = scal_h[perm]
            macts = acts_h[perm]
            mtgt = tgt_h[perm]
            real_count = int(indices.size)

            if cfg.pad_sgd_batches and real_count < cfg.minibatch_size:
                pad_count = cfg.minibatch_size - real_count
                mtac = torch.cat(
                    [mtac, torch.zeros((pad_count, *TACTICAL_SHAPE), device=self.device)], dim=0
                )
                mstrat = torch.cat(
                    [mstrat, torch.zeros((pad_count, *STRATEGIC_SHAPE), device=self.device)], dim=0
                )
                mscal = torch.cat(
                    [mscal, torch.zeros((pad_count, SCALARS_DIM), device=self.device)], dim=0
                )

            if cfg.flip_augment and self.sgd_rng.random() < 0.5:
                mtac, mstrat, mscal, macts = flip_augment(mtac, mstrat, mscal, macts)

            target_nonfinite = self._nonfinite_tensor_count(mtgt)
            if target_nonfinite:
                self._raise_numeric_tripwire("target", target_nonfinite)
            q = self.network(mtac, mstrat, mscal)[:real_count]  # discard forward-only padding
            q_taken = q.gather(1, macts.view(-1, 1)).squeeze(1)
            loss = F.smooth_l1_loss(q_taken, mtgt.detach())
            if not torch.isfinite(loss):
                self._raise_numeric_tripwire("loss", self._nonfinite_tensor_count(loss))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_nonfinite = sum(
                self._nonfinite_tensor_count(parameter.grad)
                for parameter in self.network.parameters()
                if parameter.grad is not None
            )
            if gradient_nonfinite:
                self._raise_numeric_tripwire("gradient", gradient_nonfinite)
            gnorm = nn.utils.clip_grad_norm_(self.network.parameters(), cfg.grad_clip)
            if not torch.isfinite(gnorm):
                self._raise_numeric_tripwire("gradient", self._nonfinite_tensor_count(gnorm))
            self.optimizer.step()

            parameter_nonfinite = sum(
                self._nonfinite_tensor_count(parameter) for parameter in self.network.parameters()
            )
            if parameter_nonfinite:
                self._raise_numeric_tripwire("post_optimizer_parameters", parameter_nonfinite)
            optimizer_nonfinite = self._nonfinite_tensor_count(self.optimizer.state_dict())
            if optimizer_nonfinite:
                self._raise_numeric_tripwire("post_optimizer_state", optimizer_nonfinite)
            self.refresh_numeric_recovery_state()

            losses.append(float(loss.detach()))
            grad_norms.append(float(gnorm))
            abs_q_vals.append(q.detach().abs().reshape(-1))
            action_hist += torch.bincount(macts, minlength=6).float()

        if cfg.sgd_epochs is None:
            # Keep this loop and its RNG calls in their historical order. The
            # opt-in coverage sampler below deliberately does not share it.
            for _ in range(cfg.minibatches):
                run_minibatch(self.sgd_rng.permutation(n)[: cfg.minibatch_size])
        else:
            for _ in range(cfg.sgd_epochs):
                epoch_perm = self.sgd_rng.permutation(n)
                # Equalize the last epoch shard across all optimizer steps. For
                # example, five samples with a size-three cap become 3+2, while
                # four become 2+2 rather than 3+1. Every shard remains nonempty
                # and never exceeds minibatch_size.
                batches = int(np.ceil(n / cfg.minibatch_size))
                for indices in np.array_split(epoch_perm, batches):
                    run_minibatch(indices)

        self._last_sgd_sampling = {
            "eligible_hero_transitions": int(n),
            "sampled_transition_draws": sampled_draws,
            "unique_sampled_transitions": len(sampled_indices),
            "optimizer_steps": len(losses),
        }

        abs_q = torch.cat(abs_q_vals)
        probs = action_hist / action_hist.sum().clamp_min(1.0)
        entropy = float(-(probs * (probs + 1e-12).log()).sum())
        return (
            float(np.mean(losses)),
            float(np.mean(grad_norms)),
            float(abs_q.mean()),
            float(abs_q.max()),
            entropy,
        )

    # -- tripwires ----------------------------------------------------------
    def _update_action_collapse_evidence(
        self, epsilon: float, entropy: float, eligible_samples: int, raw_action_mode: int
    ) -> Tuple[int, int]:
        """Advance or reset collapse evidence for one completed rollout update."""
        # Raw-action evidence with no eligible rows is not evidence of action
        # collapse. Keep the legacy sampled-action default untouched.
        if self.cfg.action_collapse_raw_actions and (eligible_samples == 0 or raw_action_mode < 0):
            self._action_collapse_streak = 0
            self._action_collapse_evidence_samples = 0
            self._action_collapse_raw_action_mode = None
            return self._action_collapse_streak, self._action_collapse_evidence_samples
        if epsilon < 0.5 and entropy < 1e-3:
            if (
                self.cfg.action_collapse_raw_actions
                and self._action_collapse_streak > 0
                and raw_action_mode != self._action_collapse_raw_action_mode
            ):
                self._action_collapse_streak = 0
                self._action_collapse_evidence_samples = 0
            self._action_collapse_streak += 1
            self._action_collapse_evidence_samples += eligible_samples
            if self.cfg.action_collapse_raw_actions:
                self._action_collapse_raw_action_mode = raw_action_mode
        else:
            self._action_collapse_streak = 0
            self._action_collapse_evidence_samples = 0
            self._action_collapse_raw_action_mode = None
        return self._action_collapse_streak, self._action_collapse_evidence_samples

    def _check_tripwires(self, tel: PQNTelemetry) -> None:
        """Raise :class:`TripwireError` on NaN/inf, max|Q| alarm, action-collapse.

        Args:
            tel: The telemetry snapshot for the just-finished update.

        Raises:
            TripwireError: When any tripwire condition is met.
        """
        if not np.isfinite(tel.loss) or not np.isfinite(tel.grad_norm):
            raise TripwireError(
                f"non-finite loss/grad at update {tel.update}",
                telemetry=tel,
                incident_class="nonfinite_telemetry",
            )
        if not np.isfinite(tel.max_abs_q) or tel.max_abs_q > self.cfg.max_abs_q_alarm:
            incident_class = (
                "nonfinite_max_abs_q" if not np.isfinite(tel.max_abs_q) else "max_abs_q"
            )
            raise TripwireError(
                f"max|Q|={tel.max_abs_q:.3g} exceeded alarm "
                f"{self.cfg.max_abs_q_alarm:.3g} at update {tel.update}",
                telemetry=tel,
                incident_class=incident_class,
            )
        collapse_entropy = (
            tel.raw_action_entropy if self.cfg.action_collapse_raw_actions else tel.action_entropy
        )
        if (
            tel.epsilon < 0.5
            and collapse_entropy < 1e-3
            and tel.action_collapse_streak >= self.cfg.action_collapse_patience
            and tel.action_collapse_evidence_samples >= self.cfg.action_collapse_min_samples
        ):
            raise TripwireError(
                f"action collapse (entropy={collapse_entropy:.3g}, "
                f"streak={tel.action_collapse_streak}, "
                f"samples={tel.action_collapse_evidence_samples}) at update {tel.update}",
                telemetry=tel,
                incident_class="action_collapse",
            )

    # -- public API ---------------------------------------------------------
    def update(self) -> PQNTelemetry:
        """Run one rollout + SGD update; return telemetry.

        Returns:
            The :class:`PQNTelemetry` for this update.

        Raises:
            TripwireError: If a tripwire fires (caller may halt-and-flag).
        """
        cfg = self.cfg
        if cfg.profile:
            p0 = self._sync()
            roll = self._rollout()
            p1 = self._sync()
            targets = self._compute_targets(roll)
            p2 = self._sync()
            loss, gnorm, mean_abs_q, max_abs_q, entropy = self._sgd(roll, targets)
            p3 = self._sync()
            rp = self._rollout_prof
            total = p3 - p0
            steps = cfg.num_envs * cfg.num_snakes * int(np.asarray(roll["actions"]).shape[0])
            print(
                f"[profile] total {total*1e3:6.0f}ms ({steps/total:7.0f} step/s) | "
                f"rollout {(p1-p0)*1e3:5.0f}ms [featurize {rp['featurize']*1e3:5.0f} "
                f"forward {rp['forward']*1e3:5.0f} sim {rp['sim']*1e3:5.0f}] | "
                f"targets {(p2-p1)*1e3:5.0f}ms | sgd {(p3-p2)*1e3:5.0f}ms",
                flush=True,
            )
        else:
            roll = self._rollout()
            targets = self._compute_targets(roll)
            loss, gnorm, mean_abs_q, max_abs_q, entropy = self._sgd(roll, targets)

        E, S = cfg.num_envs, cfg.num_snakes
        T = int(np.asarray(roll["actions"]).shape[0])
        hero_es = roll["policy_ids"] == HERO_POLICY_ID
        # Only alive-at-entry hero steps are real transitions (zombie steps of a
        # dead hero slot are excluded, matching the loss/target selection).
        hero_mask_tes = np.broadcast_to(hero_es[None], (T, E, S)) & roll["valid"]
        hero_steps = int(hero_mask_tes.sum())
        self.agent_steps += hero_steps
        valid = np.asarray(roll["valid"], dtype=bool)
        valid_slot_fraction = float(valid.mean())

        # Hero-only reward / boost means.
        hero_rewards = roll["rewards"][hero_mask_tes]
        hero_boost = roll["boost"][hero_mask_tes]
        raw_actions = roll["actions"][hero_mask_tes]
        mean_reward = float(hero_rewards.mean()) if hero_rewards.size else 0.0
        boost_fraction = float(hero_boost.mean()) if hero_boost.size else 0.0
        raw_hist = np.bincount(raw_actions, minlength=6).astype(np.float64)
        raw_probs = raw_hist / max(1.0, raw_hist.sum())
        raw_action_entropy = float(-(raw_probs * np.log(raw_probs + 1e-12)).sum())
        raw_action_mode = int(raw_hist.argmax()) if raw_actions.size else -1
        guard_entropy = raw_action_entropy if cfg.action_collapse_raw_actions else entropy
        collapse_streak, collapse_evidence_samples = self._update_action_collapse_evidence(
            roll["epsilon"], guard_entropy, hero_steps, raw_action_mode
        )

        hero_kills = int(np.asarray(roll["kills"])[hero_mask_tes].sum())
        hero_deaths = int(np.asarray(roll["deaths"], dtype=bool)[hero_mask_tes].sum())
        policy_exposure: Dict[str, int] = {}
        policy_identities = roll["policy_identities"]
        for policy_id in np.unique(roll["policy_ids"]):
            slot_mask = np.broadcast_to(roll["policy_ids"] == policy_id, (T, E, S))
            identity = (
                "hero:live"
                if policy_id == HERO_POLICY_ID
                else policy_identities[str(int(policy_id))]
            )
            policy_exposure[identity] = int((slot_mask & valid).sum())

        tel = PQNTelemetry(
            update=self.update_idx,
            agent_steps=self.agent_steps,
            loss=loss,
            grad_norm=gnorm,
            mean_abs_q=mean_abs_q,
            max_abs_q=max_abs_q,
            epsilon=roll["epsilon"],
            mean_reward=mean_reward,
            action_entropy=entropy,
            # Only legacy history has this compatibility field. Corrected
            # telemetry exposes the exact hero_kills count instead.
            legacy_kills_per_rollout=(float(hero_kills) if cfg.recipe == "legacy" else None),
            boost_fraction=boost_fraction,
            pool_size=len(self.pool),
            valid_slot_fraction=valid_slot_fraction,
            **self._last_sgd_sampling,
            raw_action_entropy=raw_action_entropy,
            raw_action_mode=raw_action_mode,
            action_collapse_streak=collapse_streak,
            action_collapse_evidence_samples=collapse_evidence_samples,
            hero_kills=hero_kills,
            hero_deaths=hero_deaths,
            completed_episodes=int(roll["newly_completed_episodes"]),
            valid_transitions=int(valid.sum()),
            rollout_capacity=T * E * S,
            hero_eligible_fraction=float(hero_steps / max(1, int(valid.sum()))),
            policy_exposure=policy_exposure,
            raw_action_counts=[int(value) for value in raw_hist],
            episode_reset_count=int(roll["episode_reset_count"]),
        )
        if cfg.recipe == "corrected-v3":
            tel = PQNPerEnvTelemetry(
                **tel.__dict__,
                episode_reset_mode=cfg.episode_reset_mode,
                episode_seed_mode=cfg.episode_seed_mode,
                episode_ids=[int(value) for value in np.asarray(roll["episode_ids"])],
                episode_reset_counts=[
                    int(value) for value in np.asarray(roll["episode_reset_counts"])
                ],
                reset_env_indices=[int(value) for value in np.asarray(roll["reset_env_indices"])],
                episode_world_seeds=[
                    int(value) for value in np.asarray(roll["episode_world_seeds"])
                ],
                episode_policy_ids=np.asarray(roll["policy_ids"], dtype=np.int64).tolist(),
                episode_policy_identities=dict(roll["policy_identities"]),
            )
        self.last_telemetry = tel
        self._last_policy_source = dict(roll["rollout_policy_source"])
        self._check_tripwires(tel)

        # Once every environment has ended, the lease's identities are already
        # captured in this telemetry row. Release before snapshot admission so a
        # due admission can evict an old completed-episode snapshot.
        if (
            not self._uses_derived_episode_rng
            and roll["batch_episode_finished"]
            and self._episode_lease is not None
        ):
            self._episode_lease.close()
            self._episode_lease = None

        # Snapshot the hero into the pool at the configured cadence.
        if (
            cfg.pool_capacity > 0
            and self.update_idx > 0
            and self.update_idx % cfg.pool_add_interval == 0
            and self.fixed_policy is None
            and cfg.pool_admission_mode != POOL_ADMISSION_DISABLED
        ):
            self.pool.add_snapshot(self.network)

        self.update_idx += 1
        return tel

    def train(self, num_updates: int) -> List[PQNTelemetry]:
        """Run ``num_updates`` updates, returning the telemetry list.

        Args:
            num_updates: Number of :meth:`update` calls.

        Returns:
            List of per-update telemetry.
        """
        out: List[PQNTelemetry] = []
        for _ in range(num_updates):
            out.append(self.update())
        return out

    # -- checkpoint ---------------------------------------------------------
    def checkpoint_state(self) -> Dict[str, object]:
        """Build a raster checkpoint dict (obs_spec ``raster31v2``, algo ``pqn``).

        The metadata coordinates with the checkpoint-contract module
        (:mod:`src.model.obs_spec`): it records :data:`OBS_SPEC_KEY` =
        ``raster31v2`` and the canonical flat shape keys from
        :meth:`RasterObsShapes.to_metadata` (``tactical_channels`` /
        ``tactical_size`` / ``strategic_channels`` / ``strategic_size`` /
        ``scalars``), so :meth:`InferenceAgent.from_checkpoint` reloads a PQN
        checkpoint and rebuilds a matching :class:`RasterDuelingNetwork` without
        a config. It also records the algorithm/training knobs
        (``gamma``/``lambda``/``algo``/``mechanics_version``/``reward_version``).

        Returns:
            A ``torch.save``-able dict.
        """
        world = EffectiveWorldConfig(
            width=self.cfg.game_width,
            height=self.cfg.game_height,
            segment_size=self.cfg.segment_size,
            wall_thickness=self.cfg.wall_thickness,
            arena_type=self.cfg.arena_type,
            mechanics_version=self.cfg.mechanics_version,
            num_snakes=self.cfg.num_snakes,
            max_frames=self.cfg.max_frames,
            initial_food=self.cfg.initial_food,
            max_food=self.cfg.max_food,
            min_boost_length=self.cfg.min_boost_length,
            boost_length_cost_frames=self.cfg.boost_length_cost_frames,
            frame_rate=self.cfg.frame_rate,
            max_length=self.cfg.max_length,
            starvation_max_frames=self.cfg.starvation_max,
            max_capacity=self.cfg.max_capacity,
            kill_scale=self.cfg.kill_scale,
            death_value=self.cfg.death_value,
            normalization={
                "max_frames": float(self.cfg.max_frames),
                "starvation_max": float(self.cfg.starvation_max),
                "max_length": float(self.cfg.max_length),
            },
        )
        runtime = RuntimeModeContract(
            mode="pqn_train",
            training=True,
            respawn=False,
            hero_terminal=True,
            population_floor=self.cfg.mechanics_version == 2 and self.cfg.num_snakes >= 3,
            reset_strategy=(
                "per_env_rollout_boundary" if self._uses_per_env_autoreset else "batch_episode"
            ),
        )
        mask_contract = pqn_action_mask_contract(self.cfg)
        reward_contract = pqn_reward_contract(self.cfg)
        target_contract = pqn_target_contract(self.cfg)
        sampler_contract = pqn_sampler_contract(self.cfg)
        optimizer_state = self.optimizer.state_dict()
        optimizer_contract = pqn_optimizer_contract(self.cfg, optimizer_state)
        observation_digest = (
            RASTER31V3_CONTRACT.digest if self.cfg.obs_spec == RASTER31V3 else RASTER31V2
        )
        effective_world = {field.name: getattr(world, field.name) for field in fields(world)}
        effective_world["normalization"] = dict(world.normalization)
        state: Dict[str, object] = {
            "dqn_state_dict": self.network.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "optimizer_contract": optimizer_contract,
            "reward_contract": reward_contract,
            "reward_contract_digest": canonical_digest(reward_contract),
            "target_contract": target_contract,
            "sampler_contract": sampler_contract,
            OBS_SPEC_KEY: self.cfg.obs_spec,
            "output_size": self.network.output_size,
            "gamma": self.cfg.gamma,
            "lambda": self.cfg.lambda_,
            "lr": self.cfg.lr,
            "adam_eps": self.cfg.adam_eps,
            "eps_start": self.cfg.eps_start,
            "eps_end": self.cfg.eps_end,
            "eps_decay_steps": self.cfg.eps_decay_steps,
            "algo": "pqn",
            "mechanics_version": int(self.cfg.mechanics_version),
            "reward_version": int(self.cfg.reward_version),
            # Optimization-mode provenance. These do not affect inference or
            # the promotion contract, but identify how the checkpoint was fit.
            "sgd_epochs": self.cfg.sgd_epochs,
            "pad_sgd_batches": self.cfg.pad_sgd_batches,
            "sgd_seed": self.cfg.sgd_seed,
            "action_collapse_patience": self.cfg.action_collapse_patience,
            "action_collapse_min_samples": self.cfg.action_collapse_min_samples,
            "action_collapse_raw_actions": self.cfg.action_collapse_raw_actions,
            "action_collapse_streak": self._action_collapse_streak,
            "action_collapse_evidence_samples": self._action_collapse_evidence_samples,
            "action_collapse_raw_action_mode": self._action_collapse_raw_action_mode,
            "update_counter": self.update_idx,
            "agent_steps": self.agent_steps,
            "numeric_recovery": self.numeric_recovery_metadata(),
            "recipe": self.cfg.recipe,
            "device": {
                "requested": self.cfg.requested_device,
                "effective": self.cfg.effective_device,
            },
            "field_sources": dict(self.cfg.field_sources or {}),
            "effective_world": effective_world,
            "effective_world_digest": world.digest,
            "runtime_contract": runtime.__dict__,
            "runtime_contract_digest": runtime.digest,
            "action_mask_contract": mask_contract,
            "action_mask_contract_digest": canonical_digest(mask_contract),
            "resume_mode": self._resume_mode,
            "resume_state": dict(self._resume_state),
            "rollout_policy_source": dict(self._last_policy_source),
        }
        if self.cfg.recipe == "corrected-v3":
            lifecycle = build_pqn_episode_lifecycle_contract(
                self.cfg.episode_reset_mode, self.cfg.episode_seed_mode
            )
            policy_source = self._static_policy_source_contract()
            lifecycle_digest = canonical_digest(lifecycle)
            policy_source_digest = canonical_digest(policy_source)
            state.update(
                {
                    "episode_lifecycle_contract": lifecycle,
                    "episode_lifecycle_contract_digest": lifecycle_digest,
                    "policy_source_contract": policy_source,
                    "policy_source_contract_digest": policy_source_digest,
                    "episode_reset_mode": self.cfg.episode_reset_mode,
                    "episode_seed_mode": self.cfg.episode_seed_mode,
                    "pool_admission_mode": self.cfg.pool_admission_mode,
                    "initial_opponent_checkpoint_sha256": (
                        self.cfg.initial_opponent_checkpoint_sha256
                    ),
                    "episode_ids": np.asarray(self._episode_ids, dtype=np.uint64).copy(),
                    "episode_reset_counts": self._episode_reset_counts.copy(),
                    "episode_world_seeds": self._episode_world_seeds.copy(),
                    "episode_policy_ids": (
                        None
                        if self._episode_policy_ids is None
                        else self._episode_policy_ids.copy()
                    ),
                    "episode_policy_identities": self._episode_policy_identities(),
                    "restorable_environment_state": False,
                }
            )
            target_contract["episode_lifecycle_contract_digest"] = lifecycle_digest
            sampler_contract["episode_lifecycle_contract_digest"] = lifecycle_digest
            sampler_contract["policy_source_contract_digest"] = policy_source_digest
            state["rollout_policy_source"] = {
                **dict(self._last_policy_source),
                "policy_source_contract_digest": policy_source_digest,
            }
        state.update(ModelHeadContract("pqn", "dueling_q", 6).to_metadata())
        state["target_contract_digest"] = canonical_digest(state["target_contract"])
        state["sampler_contract_digest"] = canonical_digest(state["sampler_contract"])
        state["optimizer_contract_digest"] = canonical_digest(state["optimizer_contract"])
        provenance = RunProvenance(
            effective_seed=self.cfg.seed,
            observation_digest=observation_digest,
            world_digest=world.digest,
            runtime_digest=runtime.digest,
            reward_digest=canonical_digest(reward_contract),
            target_digest=canonical_digest(target_contract),
            sampler_digest=canonical_digest(sampler_contract),
            optimizer_digest=canonical_digest(optimizer_contract),
            model_head_digest=ModelHeadContract("pqn", "dueling_q", 6).digest,
            source_revision=self.cfg.source_revision,
        )
        state.update(provenance.to_metadata())
        if self.cfg.obs_spec == RASTER31V3:
            state.update(RASTER31V3_CONTRACT.to_metadata())
        state.update(RASTER31V2_SHAPES.to_metadata())
        return state

    def save_checkpoint(self, path: str) -> None:
        """Write a raster checkpoint to ``path`` atomically.

        Serializes to a temp file in the DESTINATION directory (so the rename
        stays on one filesystem), fsyncs it, then renames it into place. A run
        writes a single rolling artifact, and ``torch.save`` truncates its target
        the moment it opens it — so a failure part-way through a direct write
        (preemption, OOM, full disk) would destroy the previous good checkpoint
        rather than merely fail to replace it.

        Args:
            path: Destination file path.
        """
        atomic_torch_save(self.checkpoint_state(), path)

    def save_incident_checkpoint(self, path: str, incident: TripwireError) -> None:
        """Save a finite flagged state without replacing the accepted rolling checkpoint."""
        if not incident.is_finite_alarm:
            raise ValueError("only finite tripwires may be written as incident checkpoints")
        state = self.checkpoint_state()
        state["incident"] = {
            "class": incident.incident_class,
            "message": str(incident),
            "telemetry": None if incident.telemetry is None else incident.telemetry.__dict__,
            "details": incident.incident,
        }
        atomic_torch_save(state, path)


def validate_checkpoint_numeric_state(checkpoint: Dict[str, object]) -> None:
    """Reject a non-finite PQN resume payload without relying on a trainer instance."""
    PQNTrainer.validate_checkpoint_numeric_state(checkpoint)
