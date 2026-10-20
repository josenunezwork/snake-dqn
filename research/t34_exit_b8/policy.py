"""Policy boundary for the deterministic ExIt-B8 teacher.

The engine deliberately depends on this small interface instead of a checkpoint
loader.  That keeps label semantics independent of whether the rollout policy
is the released B8 network or a later zero-initialised tower student.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

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

    def evaluate_rows(self, sim: Any, rows: np.ndarray, net: Any) -> np.ndarray:
        """Return float32 (N, 6) rollout Qs at prepared compacted rows."""


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
    """B8-compatible fp32 policy adapter for T3/T33 captured worlds.

    ``batch_device_vectors`` is deliberately opt-in for T34 v2 shadow
    collection.  The older v1 paths retain their established native vector61
    forward recipe even when their hero policy uses CUDA.
    """

    def __init__(self, device: str = "cpu", *, batch_device_vectors: bool = False) -> None:
        self.device = str(device)
        self.hero_forward_seconds = 0.0
        self.vector_forward_seconds = 0.0
        self.batch_device_vectors = bool(batch_device_vectors)
        if self.device.startswith("cuda"):
            # Float32 tensors alone do not exclude CUDA's reduced-mantissa
            # TF32 convolution/matmul paths. Labels use the full fp32 recipe.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self._models = _Fp32ModelCache(self.device)

    def _forward(self, net: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
        from src.model.ego2s_network import obs_tensors

        model = self._models.model_for(net)
        started = time.perf_counter()
        with torch.inference_mode():
            value = model(*obs_tensors(obs, self._models.device)).float().cpu().numpy()
        self.hero_forward_seconds += time.perf_counter() - started
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

    def evaluate_worlds(
        self, worlds: Sequence[Any], *, bucket_size: int = 64
    ) -> list[PolicyEvaluation]:
        """Evaluate independent prepared roots in bounded fp32 inference buckets.

        Physical state and RNG remain private to each world. This is an explicit
        batching recipe: batch-N float32 forwards need not match batch-one bits.
        All worlds in one invocation must carry the same immutable hero model.
        """
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        if type(bucket_size) is not int or not 1 <= bucket_size <= 1536:
            raise ValueError("inference bucket size must lie in [1, 1536]")
        if not worlds:
            return []
        if any(world.hero.net is not worlds[0].hero.net for world in worlds):
            raise ValueError("batched roots must share the immutable hero model")
        results = []
        for start in range(0, len(worlds), bucket_size):
            chunk = worlds[start : start + bucket_size]
            observations = [
                build_ego_raster(
                    world.sim,
                    EgoRasterConfig(version="bv8"),
                    backend="numba",
                    rows=np.asarray([[0, 0]], dtype=np.int64),
                )
                for world in chunk
            ]
            obs = {
                key: np.concatenate([item[key] for item in observations], axis=0)
                for key in observations[0]
            }
            values = self._forward(chunk[0].hero.net, obs)
            for world, q in zip(chunk, values):
                legal = np.asarray(world.sim.get_legal_action_mask()[0, 0], dtype=bool).copy()
                if not legal.any():
                    legal[:3] = True
                select = np.asarray(world.sim.get_advisory_action_mask()[0, 0], dtype=bool).copy()
                if not select.any():
                    select[:3] = legal[:3]
                results.append(PolicyEvaluation(q=q, legal=legal, select=select))
        return results

    def dispatch(
        self, sim: Any, runtime: Any, opponents: list[list[Any]], grid: np.ndarray
    ) -> None:
        """Fill only opponent columns, preserving each policy's native action hook.

        Hero inference is intentionally absent: the caller already selected its
        action.  CPU retains the historical per-policy path because batch-N and
        batch-one network forwards need not agree bit-for-bit.  CUDA collection
        batches compatible vector61 state rows per immutable network, but still
        enters every policy's own ``actions`` hook so its v5/v7/v8 carry stays
        private to that policy.
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
        if self.batch_device_vectors and self.device.startswith("cuda"):
            self._dispatch_cuda_vectors(sim, groups, grid)
            return
        for policy, pairs in groups.values():
            slots = np.asarray(pairs, dtype=np.int64)
            masks = sim.get_resolved_action_mask()[slots[:, 0], slots[:, 1]]
            # ``Vector61SimdPolicy.actions`` owns exact veto dispatch, including
            # per-environment live-hook carry.  The cached fp32 path is safe
            # only for policies with no serving veto state.
            if getattr(policy, "veto_slots", frozenset()):
                grid[slots[:, 0], slots[:, 1]] = policy.actions(masks, sim, slots)
            elif hasattr(policy, "runtime") and hasattr(policy, "q_values"):
                grid[slots[:, 0], slots[:, 1]] = _vector_actions(policy, sim, slots, self._models)
            else:
                grid[slots[:, 0], slots[:, 1]] = policy.actions(masks, sim, slots)

    def _dispatch_cuda_vectors(
        self,
        sim: Any,
        groups: dict[tuple[Any, ...], tuple[Any, list[tuple[int, int]]]],
        grid: np.ndarray,
    ) -> None:
        """Dispatch CUDA vector61 forwards once per immutable source network.

        ``Vector61SimdPolicy.actions`` contains the precise live advisory-mask,
        veto, and per-environment carry contract.  Replacing it with a local
        argmax loses v8's normalized-Q tie behaviour.  This method therefore
        only supplies an already-computed device Q slice to that native method.
        The temporary override is restored before the next policy dispatch.
        """
        vectors: dict[int, list[tuple[Any, np.ndarray]]] = {}
        fallback: list[tuple[Any, np.ndarray, np.ndarray]] = []
        for policy, pairs in groups.values():
            slots = np.asarray(pairs, dtype=np.int64)
            masks = sim.get_resolved_action_mask()[slots[:, 0], slots[:, 1]]
            if _is_vector61_device_policy(policy):
                vectors.setdefault(id(policy.network), []).append((policy, slots))
            else:
                fallback.append((policy, slots, masks))

        # Anchors and any non-native policy retain their established dispatch.
        for policy, slots, masks in fallback:
            grid[slots[:, 0], slots[:, 1]] = policy.actions(masks, sim, slots)

        for entries in vectors.values():
            states_by_policy: list[np.ndarray] = []
            counts: list[int] = []
            for policy, slots in entries:
                states, _exact = policy.runtime.selection(slots)
                states = np.ascontiguousarray(states, dtype=np.float32)
                if states.ndim != 2 or states.shape[1] != int(policy.input_size):
                    raise ValueError(
                        "checkpoint input_size does not match the active vector61 featurizer width"
                    )
                states_by_policy.append(states)
                counts.append(len(states))
            joined = np.concatenate(states_by_policy, axis=0)
            model = self._models.model_for(entries[0][0].network)
            started = time.perf_counter()
            with torch.inference_mode():
                # Native hooks do masking, argmax, and v8 normalization on the
                # host.  One slab transfer avoids a CUDA synchronization for
                # each independently carried policy group.
                q_values = model(torch.from_numpy(joined).to(self._models.device)).float().cpu()
            self.vector_forward_seconds += time.perf_counter() - started
            offset = 0
            for (policy, slots), count in zip(entries, counts):
                q_slice = q_values[offset : offset + count]
                offset += count
                masks = sim.get_resolved_action_mask()[slots[:, 0], slots[:, 1]]
                grid[slots[:, 0], slots[:, 1]] = _native_vector_actions(
                    policy, masks, sim, slots, q_slice
                )

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


def _is_vector61_device_policy(policy: Any) -> bool:
    """Return whether ``policy`` can accept a cached native vector61 Q slice."""
    return all(
        hasattr(policy, name)
        for name in ("actions", "input_size", "network", "q_values", "runtime")
    )


def _native_vector_actions(
    policy: Any, masks: np.ndarray, sim: Any, slots: np.ndarray, q_values: torch.Tensor
) -> np.ndarray:
    """Run a vector policy's native action hook with a transient device Q source.

    The hook can mutate veto counters and live-v8 objects, so it must execute on
    its original policy instance.  ``q_values`` is captured per policy before
    the hook runs; restoring the original callable in ``finally`` prevents a
    cached forward from escaping this one dispatch.
    """
    instance_values = vars(policy)
    had_local_q_values = "q_values" in instance_values
    original_q_values = instance_values.get("q_values")

    def cached_q_values(states: np.ndarray) -> torch.Tensor:
        del states
        return q_values

    policy.q_values = cached_q_values
    try:
        return np.asarray(policy.actions(masks, sim, slots), dtype=np.int64)
    finally:
        if had_local_q_values:
            policy.q_values = original_q_values
        else:
            del policy.q_values


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
