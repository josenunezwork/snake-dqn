"""Policy boundary for the deterministic ExIt-B8 teacher.

The engine deliberately depends on this small interface instead of a checkpoint
loader.  That keeps label semantics independent of whether the rollout policy
is the released B8 network or a later zero-initialised tower student.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class PolicyEvaluation:
    """The exact decision-time Q values and the selectable action domain."""

    q: np.ndarray
    legal: np.ndarray
    select: np.ndarray | None = None

    def __post_init__(self) -> None:
        if np.asarray(self.q).shape != (6,) or np.asarray(self.legal).shape != (6,):
            raise ValueError("policy evaluation requires six Q values and six legal flags")
        if not np.isfinite(self.q).all():
            raise ValueError("policy returned non-finite Q values")

    def greedy_action(self) -> int:
        """Return the stable float32 greedy legal action (first index wins ties)."""
        select = np.asarray(self.legal if self.select is None else self.select, dtype=bool).copy()
        if not select.any():
            select[:3] = np.asarray(self.legal, dtype=bool)[:3]
        if not select.any():
            raise ValueError("policy has no legal action")
        return int(np.where(select, np.asarray(self.q, dtype=np.float32), -np.inf).argmax())


class RolloutPolicy(Protocol):
    """Pluggable fp32 student-policy interface consumed by :class:`ExitTeacher`."""

    def evaluate(self, world: Any) -> PolicyEvaluation:
        """Return decision-time float32 Q values and the legal action mask."""


class _Fp32ModelCache:
    """Immutable source-model copies keyed only by weights identity and device."""

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self._models: dict[tuple[int, str], torch.nn.Module] = {}

    def model_for(self, source: torch.nn.Module) -> torch.nn.Module:
        key = (id(source), str(self.device))
        model = self._models.get(key)
        if model is None:
            model = copy.deepcopy(source).to(self.device, dtype=torch.float32).eval()
            self._models[key] = model
        return model


class B8Policy:
    """B8-compatible fp32 policy adapter for T3/T33 captured worlds."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = str(device)
        self._models = _Fp32ModelCache(self.device)

    def _forward(self, net: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
        from src.model.ego2s_network import obs_tensors

        model = self._models.model_for(net)
        with torch.inference_mode():
            value = model(*obs_tensors(obs, self._models.device)).float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def evaluate(self, world: Any) -> PolicyEvaluation:
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        sim = world.sim
        obs = build_ego_raster(
            sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
        )
        q = self._forward(world.hero.net, obs)[0]
        legal = np.asarray(sim.get_legal_action_mask()[0, 0], dtype=bool)
        if not legal.any():
            legal[:3] = True
        select = np.asarray(sim.get_advisory_action_mask()[0, 0], dtype=bool)
        if not select.any():
            select[:3] = legal[:3]
        return PolicyEvaluation(q=q, legal=legal, select=select)

    def evaluate_rows(self, sim: Any, rows: np.ndarray, net: Any) -> np.ndarray:
        """Return fp32 B8 Qs for compacted rows of a merged T33 simulator."""
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        obs = build_ego_raster(sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
        return self._forward(net, obs)

    def dispatch(
        self, sim: Any, runtime: Any, opponents: list[list[Any]], grid: np.ndarray
    ) -> None:
        """Fill only opponent columns, batched by policy object and fp32 device model.

        Hero inference is intentionally absent: the caller already selected its
        action.  Vector61's advisory mask and v8 hook are reproduced after the
        device Q forward through its own ``_apply_veto`` implementation.
        """
        del runtime
        alive = np.asarray(sim.get_alive(), dtype=bool)
        groups: dict[tuple[Any, ...], tuple[Any, list[tuple[int, int]]]] = {}
        for slot in range(1, int(sim.S)):
            for env, row in enumerate(opponents):
                if not alive[env, slot]:
                    continue
                policy = row[slot - 1]
                # Only a deliberately shared policy object may share a batched
                # call.  Anchor kind and mutable veto state are not safely
                # represented by a checkpoint-only key.
                key = (id(policy),)
                prior = groups.get(key)
                if prior is None:
                    groups[key] = (policy, [(env, slot)])
                else:
                    prior[1].append((env, slot))
        for policy, pairs in groups.values():
            slots = np.asarray(pairs, dtype=np.int64)
            masks = sim.get_resolved_action_mask()[slots[:, 0], slots[:, 1]]
            if hasattr(policy, "runtime") and hasattr(policy, "q_values"):
                grid[slots[:, 0], slots[:, 1]] = _vector_actions(policy, sim, slots, self._models)
            else:
                grid[slots[:, 0], slots[:, 1]] = policy.actions(masks, sim, slots)

    def action_grid(self, world: Any, hero_action: int) -> np.ndarray:
        """Build an opponent-only forced-hero grid using this policy's cache."""
        return action_grid(world, hero_action, device=self.device, cache=self)


def _vector_actions(policy: Any, sim: Any, slots: np.ndarray, cache: _Fp32ModelCache) -> np.ndarray:
    """Vector61 fp32 device forward followed by its exact advisory/v8 logic."""
    from src.training.action_mask import INVALID_Q_VALUE

    states, mask = policy.runtime.selection(slots)
    mask = np.asarray(mask, dtype=bool).copy()
    mask[~mask.any(axis=1), :3] = True
    model = cache.model_for(policy.network)
    with torch.inference_mode():
        state_t = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32)).to(cache.device)
        q = model(state_t).float()
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=cache.device)
        masked = torch.where(mask_t, q, torch.full_like(q, INVALID_Q_VALUE))
        actions = masked.argmax(dim=1).cpu().numpy().astype(np.int64)
    if getattr(policy, "veto_slots", frozenset()):
        actions = policy._apply_veto(sim, slots, masked, mask, actions)
    return np.clip(actions, 0, 5)


def action_grid(
    world: Any, hero_action: int, device: str = "cpu", cache: B8Policy | None = None
) -> np.ndarray:
    """Build one exact six-snake action grid while overriding only the hero.

    T3's dispatcher preserves the frozen-vector61 batch path and scripted host
    anchors.  Callers that already own a prepared T33 world can use this helper
    to avoid an unnecessary hero forward before applying the teacher action.
    """
    sim = world.sim
    if not 0 <= int(hero_action) < 6:
        raise ValueError("hero action must be in [0, 5]")
    actions = np.ones((1, sim.S), dtype=np.int64)
    runner = cache or B8Policy(device=device)
    runner.dispatch(sim, world.runtime, [world.opponents], actions)
    actions[0, 0] = int(hero_action)
    return actions
