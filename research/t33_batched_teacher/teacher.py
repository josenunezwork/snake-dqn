"""Fixed-shape, many-source continuation teacher for T33.

The teacher deliberately has no strict-gate or serving role.  Its resident
layout is always ``W * 6 * N`` rows in source/action/rollout order; actions
which are illegal in a source world remain resident but are frozen.  This is
important for the CUDA path, where changing the model batch shape after a
death would invalidate the intended compiled graph.
"""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterable

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

from research.phase0_search_20261019.teacher import (  # noqa: E402
    SearchConfig,
    _margin,
    _mask_argmax,
)
from research.t3_rollout_labels_20261015.engine import _restore_grid_views  # noqa: E402

NUM_ACTIONS = 6


@dataclass(frozen=True)
class BranchLayout:
    """Stable row metadata for a resident T33 continuation batch."""

    source: np.ndarray
    action: np.ndarray
    rollout: np.ndarray
    active: np.ndarray


def _policy_key(policy: Any) -> tuple[Any, ...]:
    """Return the immutable identity used to share frozen policies across worlds."""
    checkpoint = getattr(policy, "checkpoint_path", None)
    # A missing checkpoint identity is not evidence of equal weights.  Retain
    # object identity in that case instead of accidentally coalescing models.
    return (
        type(policy),
        checkpoint if checkpoint is not None else id(policy),
        getattr(policy, "forward", None),
        getattr(policy, "kind", None),
        getattr(policy, "mask_mode", None),
    )


def _repeat_value(value: Any, source_index: np.ndarray) -> Any:
    """Gather an environment-leading mutable value without sharing source state."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            out = np.empty((len(source_index), *value.shape[1:]), dtype=value.dtype)
            for destination, source in enumerate(source_index):
                out[destination] = copy.deepcopy(value[int(source)])
            return out
        return value[source_index].copy()
    if isinstance(value, list):
        return [copy.deepcopy(value[int(source)]) for source in source_index]
    return copy.deepcopy(value)


def _clone_policy(policy: Any, runtime: Any) -> Any:
    clone = copy.copy(policy)
    if hasattr(clone, "__dict__"):
        clone.__dict__ = dict(clone.__dict__)
    if hasattr(clone, "runtime"):
        clone.runtime = runtime
    if hasattr(clone, "forward"):
        clone.forward = "batched"
    return clone


def _model_digest(net: Any) -> str:
    """Hash a model's parameter bytes once per choose call for batch safety."""
    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _runtime_for(sources: list[Any], sim: Any, controlled: np.ndarray) -> Any:
    """Bind a fresh T33 runtime, then restore only environment-indexed carry."""
    from research.t33_batched_teacher.runtime import BatchedVector61Runtime

    first = sources[0].runtime
    params = getattr(first, "_params", None)
    if params is None and getattr(first, "featurizer", None) is not None:
        params = first.featurizer.p
    runtime = BatchedVector61Runtime(params=params)
    runtime.bind(sim, controlled)
    return runtime


def merge_worlds(
    sources: Iterable[Any], rollouts: int, legal: Iterable[np.ndarray] | None = None
) -> Any:
    """Clone prepared sources into one fixed ``W * 6 * N`` resident batch.

    No field from a source simulator, runtime memory, or policy instance is
    reused mutably.  The returned ``layout.active`` marks legal first-action
    rows; inactive rows are intentionally present and must always be passed as
    false in the simulator active mask.
    """
    source_list = list(sources)
    if not source_list or rollouts < 1:
        raise ValueError("merge_worlds requires sources and a positive rollout count")
    if any(int(world.sim.E) != 1 for world in source_list):
        raise ValueError("each captured source must contain exactly one environment")
    if any(not getattr(world.sim, "_policy_phase_prepared", False) for world in source_list):
        raise ValueError("sources must be captured at a prepared Watch boundary")
    first_sim = source_list[0].sim
    if any(
        (world.sim.S, type(world.sim)) != (first_sim.S, type(first_sim)) for world in source_list
    ):
        raise ValueError("sources must have matching simulator type and slot count")
    scalar = (first_sim.s, first_sim.cap, first_sim.train_mode, first_sim.allow_respawn)
    if any(
        (world.sim.s, world.sim.cap, world.sim.train_mode, world.sim.allow_respawn) != scalar
        for world in source_list
    ):
        raise ValueError("sources must agree on segment scale, capacity, and runtime mode")
    from dataclasses import replace

    expected_cfg = replace(first_sim.cfg, num_envs=1)
    if any(replace(world.sim.cfg, num_envs=1) != expected_cfg for world in source_list):
        raise ValueError("sources must have matching simulator configuration")
    runtime_params = getattr(source_list[0].runtime.featurizer, "p", None)
    if any(getattr(world.runtime.featurizer, "p", None) != runtime_params for world in source_list):
        raise ValueError("sources must have matching Vector61 featurizer parameters")
    source_ids = np.repeat(np.arange(len(source_list), dtype=np.int64), NUM_ACTIONS * rollouts)
    actions = np.tile(np.repeat(np.arange(NUM_ACTIONS, dtype=np.int64), rollouts), len(source_list))
    rollout_ids = np.tile(np.arange(rollouts, dtype=np.int64), len(source_list) * NUM_ACTIONS)
    legal_rows = (
        list(legal) if legal is not None else [np.ones(NUM_ACTIONS, dtype=bool)] * len(source_list)
    )
    if len(legal_rows) != len(source_list):
        raise ValueError("legal masks must align one-to-one with sources")
    active = np.array([bool(legal_rows[int(w)][int(a)]) for w, a in zip(source_ids, actions)])

    # Clone one shape-bearing simulator, then gather every E-leading field from
    # its corresponding source.  This deliberately includes RNG/food lists and
    # object arrays, which are the common accidental source-mutation paths.
    from research.b8_death_census_20261014 import census

    sim = census.clone_sim(first_sim)
    for name, destination in list(vars(sim).items()):
        if isinstance(destination, np.ndarray) and destination.ndim and destination.shape[0] == 1:
            values = [getattr(world.sim, name) for world in source_list]
            if not all(isinstance(value, np.ndarray) and value.shape[0] == 1 for value in values):
                raise ValueError(f"source simulator field {name} is incompatible")
            joined = np.concatenate(values, axis=0)
            setattr(sim, name, _repeat_value(joined, source_ids))
        elif isinstance(destination, list) and len(destination) == 1:
            values = []
            for world in source_list:
                value = getattr(world.sim, name)
                if not isinstance(value, list) or len(value) != 1:
                    raise ValueError(f"source simulator list {name} is incompatible")
                values.extend(value)
            setattr(sim, name, _repeat_value(values, source_ids))
    sim.E = len(source_ids)
    if hasattr(sim, "cfg"):
        try:
            sim.cfg = replace(sim.cfg, num_envs=sim.E)
        except TypeError:
            pass
    sim._env_index = np.arange(sim.E, dtype=np.int64)[:, None]
    sim._snake_index = np.arange(sim.S, dtype=np.int64)[None, :]
    sim._det = None
    sim._policy_phase_prepared = True
    sim._policy_step_in_progress = False
    if hasattr(sim, "_active_env_mask"):
        del sim._active_env_mask
    _restore_grid_views(sim)

    # The runtime's controlled rows are discovered from the strict opponent
    # roster.  Hero rows use the T33 ego2s inference path and are forced here.
    controlled = np.zeros((sim.E, sim.S), dtype=bool)
    for env, source in enumerate(source_ids):
        for slot, policy in enumerate(source_list[int(source)].opponents, start=1):
            if hasattr(policy, "runtime"):
                controlled[env, slot] = True
    runtime = _runtime_for(source_list, sim, controlled)

    # Carry has one leading environment axis.  Replace the freshly allocated
    # values with source snapshots, retaining fresh featurizer caches bound to
    # the merged simulator.
    for name, value in vars(runtime).items():
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == sim.E:
            values = [getattr(world.runtime, name, None) for world in source_list]
            if all(isinstance(item, np.ndarray) and item.shape[0] == 1 for item in values):
                setattr(runtime, name, _repeat_value(np.concatenate(values, axis=0), source_ids))
    if getattr(runtime, "memory", None) is not None:
        for name, value in vars(runtime.memory).items():
            if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == sim.E:
                values = [getattr(world.runtime.memory, name, None) for world in source_list]
                if all(isinstance(item, np.ndarray) and item.shape[0] == 1 for item in values):
                    setattr(
                        runtime.memory,
                        name,
                        _repeat_value(np.concatenate(values, axis=0), source_ids),
                    )

    # Shared policy identities make eval_engine dispatch each frozen checkpoint
    # once per frame across all source worlds.  Anchors use their own constructor
    # because their seed vector is part of their deterministic state.
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    cache: dict[tuple[Any, ...], Any] = {}
    opponents: list[list[Any]] = []
    for source in source_ids:
        row: list[Any] = []
        for policy in source_list[int(source)].opponents:
            if not isinstance(policy, (FastProfileAnchorSimdPolicy, Vector61SimdPolicy)):
                raise TypeError("T33 supports only strict frozen Vector61 or fast-anchor opponents")
            if isinstance(policy, Vector61SimdPolicy) and getattr(
                policy, "veto_slots", frozenset()
            ):
                raise TypeError("T33 frozen opponents must not carry a serving veto")
            key = _policy_key(policy)
            if isinstance(policy, FastProfileAnchorSimdPolicy):
                # random_safe indexes this vector by merged env id; a shallow
                # clone would retain the source's one-item seed vector.
                cached = cache.get(key)
                if cached is None:
                    cached = FastProfileAnchorSimdPolicy(
                        policy.kind,
                        [int(source_list[int(index)].seed) for index in source_ids],
                    )
                    cache[key] = cached
                row.append(cached)
            else:
                row.append(cache.setdefault(key, _clone_policy(policy, runtime)))
        opponents.append(row)
    grouped: dict[Any, list[tuple[int, int]]] = {}
    for env, row in enumerate(opponents):
        for slot, policy in enumerate(row, start=1):
            grouped.setdefault(policy, []).append((env, slot))
    policy_groups = {policy: np.asarray(rows, dtype=np.int64) for policy, rows in grouped.items()}
    layout = BranchLayout(source_ids, actions, rollout_ids, active)
    return SimpleNamespace(
        sim=sim,
        runtime=runtime,
        opponents=opponents,
        policy_groups=policy_groups,
        layout=layout,
        sources=source_list,
    )


class BatchedTeacher:
    """Many-source one-ply search teacher with common random numbers by default."""

    def __init__(
        self,
        config: SearchConfig = SearchConfig(),
        *,
        device: str = "cpu",
        common_random: bool = True,
        leaf: str = "max",
        inference: Callable[[Any, dict[str, np.ndarray]], Any] | None = None,
        opponent_inference: Any | None = None,
    ) -> None:
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps or cuda")
        if leaf not in ("max", "value"):
            raise ValueError("leaf must be max or value")
        self.config, self.device, self.common_random, self.leaf = (
            config,
            device,
            common_random,
            leaf,
        )
        self.inference = inference
        self.opponent_inference = opponent_inference
        self._profile: dict[str, float | int] = {
            "assemble_s": 0.0,
            "obs_host_s": 0.0,
            "hero_forward_s": 0.0,
            "opponent_dispatch_s": 0.0,
            "runtime_prepare_s": 0.0,
            "runtime_observe_s": 0.0,
            "sim_step_s": 0.0,
            "decisions": 0,
            "steps": 0,
            "resident_rows": 0,
        }
        self._observation_buffers: dict[int, Any] = {}

    def profile(self, *, reset: bool = False) -> dict[str, float | int]:
        """Return additive phase timing counters for a benchmark receipt."""
        result = dict(self._profile)
        if reset:
            for key, value in self._profile.items():
                self._profile[key] = 0 if isinstance(value, int) else 0.0
        return result

    def _q(self, sim: Any, rows: np.ndarray, net: Any) -> np.ndarray:
        import time

        from research.t33_batched_teacher.observations import ObservationBuffer
        from src.simd_env.ego2s_policy import forward_q

        started = time.perf_counter()
        buffer = self._observation_buffers.get(len(rows))
        if buffer is None:
            buffer = ObservationBuffer(len(rows), pinned=self.device == "cuda")
            self._observation_buffers[len(rows)] = buffer
        obs = buffer.build(sim, rows)
        self._profile["obs_host_s"] += time.perf_counter() - started
        started = time.perf_counter()
        if self.inference is None:
            value = forward_q(net, obs, "batched")
        elif hasattr(self.inference, "q"):
            value = self.inference.q(buffer)
        else:
            value = self.inference(net, obs)
        self._profile["hero_forward_s"] += time.perf_counter() - started
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def choose(self, worlds: list[Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Choose one action per prepared world without mutating its source state."""
        if not worlds:
            return np.empty(0, dtype=np.int64), []
        networks = {id(world.hero.net): world.hero.net for world in worlds}
        digests = [_model_digest(net) for net in networks.values()]
        if any(digest != digests[0] for digest in digests[1:]):
            raise ValueError("all source worlds must use byte-identical hero network weights")
        legal, select, decisions = [], [], []
        for world in worlds:
            domain = world.sim.get_legal_action_mask()[0, 0].astype(bool)
            advisory = world.sim.get_advisory_action_mask()[0, 0].astype(bool)
            if not advisory.any():
                advisory[:3] = domain[:3]
            legal.append(domain)
            select.append(advisory)
            decisions.append(int(world.sim.frame[0]) - 1)
        import time

        started = time.perf_counter()
        merged = merge_worlds(worlds, self.config.rollouts, legal)
        self._profile["assemble_s"] += time.perf_counter() - started
        self._profile["decisions"] += len(worlds)
        self._profile["resident_rows"] = int(merged.sim.E)
        hero_rows = np.column_stack(
            (np.arange(merged.sim.E, dtype=np.int64), np.zeros(merged.sim.E, dtype=np.int64))
        )
        q_all = self._q(merged.sim, hero_rows, worlds[0].hero.net)
        initial_q = q_all.reshape(len(worlds), NUM_ACTIONS, self.config.rollouts, NUM_ACTIONS)[
            :, 0, 0
        ]
        return self._search(merged, initial_q, legal, select, decisions)

    def _dispatch_opponents(
        self, sim: Any, runtime: Any, grid: np.ndarray, groups: dict[Any, np.ndarray]
    ) -> None:
        """Forward each frozen checkpoint over its static rows, then mask dead rows."""
        for policy, slots in groups.items():
            if not (hasattr(policy, "q_values") and hasattr(policy, "runtime")):
                continue
            states = runtime.sel_state[slots[:, 0], slots[:, 1]]
            q = (
                self.opponent_inference.vector_q(policy, states)
                if self.opponent_inference is not None
                else policy.q_values(states)
            )
            if hasattr(q, "detach"):
                q = q.detach().cpu().numpy()
            mask = runtime.sel_mask[slots[:, 0], slots[:, 1]].copy()
            missing = ~mask.any(axis=1)
            mask[missing, :3] = True
            picked = np.where(mask, np.asarray(q, dtype=np.float64), -np.inf).argmax(axis=1)
            live = sim.alive[slots[:, 0], slots[:, 1]]
            grid[slots[live, 0], slots[live, 1]] = picked[live]
        for policy, slots in groups.items():
            if hasattr(policy, "q_values") and hasattr(policy, "runtime"):
                continue
            result = policy.actions(
                sim.get_resolved_action_mask()[slots[:, 0], slots[:, 1], :],
                sim,
                slots,
            )
            live = sim.alive[slots[:, 0], slots[:, 1]]
            grid[slots[live, 0], slots[live, 1]] = np.asarray(result, dtype=np.int64)[live]

    def _search(
        self,
        merged: Any,
        initial_q: np.ndarray,
        legal: list[np.ndarray],
        select: list[np.ndarray],
        decisions: list[int],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        import time

        sim, layout = merged.sim, merged.layout
        W = len(merged.sources)
        base = np.asarray(
            [_mask_argmax(q, mask) for q, mask in zip(initial_q, select)], dtype=np.int64
        )
        eligible = np.asarray(
            [
                int(world.sim.length[0, 0]) >= self.config.length_gate
                or _margin(q, m) < self.config.margin_threshold
                for world, q, m in zip(merged.sources, initial_q, legal)
            ],
            dtype=bool,
        )
        active = layout.active & eligible[layout.source]
        grid = np.ones((sim.E, sim.S), dtype=np.int64)
        started = time.perf_counter()
        self._dispatch_opponents(sim, merged.runtime, grid, merged.policy_groups)
        self._profile["opponent_dispatch_s"] += time.perf_counter() - started
        grid[:, 0] = layout.action
        started = time.perf_counter()
        sim.step(grid, active_env_mask=active)
        self._profile["sim_step_s"] += time.perf_counter() - started
        started = time.perf_counter()
        merged.runtime.observe_step(sim)
        self._profile["runtime_observe_s"] += time.perf_counter() - started
        self._profile["steps"] += 1
        mass = np.where(active & sim.alive[:, 0], sim.length[:, 0], 0).astype(np.float64)
        live = active & sim.alive[:, 0]
        rngs = [
            np.random.default_rng(
                np.random.SeedSequence(
                    [
                        int(merged.sources[int(w)].seed),
                        int(decisions[int(w)]),
                        0 if self.common_random else int(a),
                        int(r),
                        20261019,
                    ]
                )
            )
            for w, a, r in zip(layout.source, layout.action, layout.rollout)
        ]
        hero_rows = np.column_stack(
            (np.arange(sim.E, dtype=np.int64), np.zeros(sim.E, dtype=np.int64))
        )
        for _ in range(1, self.config.horizon):
            if not live.any():
                break

            selector_elapsed = [0.0]

            def choose(prepared: Any) -> np.ndarray:
                selector_started = time.perf_counter()
                started = time.perf_counter()
                merged.runtime.prepare(prepared)
                self._profile["runtime_prepare_s"] += time.perf_counter() - started
                grid.fill(1)
                started = time.perf_counter()
                self._dispatch_opponents(prepared, merged.runtime, grid, merged.policy_groups)
                self._profile["opponent_dispatch_s"] += time.perf_counter() - started
                q = self._q(prepared, hero_rows, merged.sources[0].hero.net)
                masks = prepared.get_advisory_action_mask()[:, 0].astype(bool)
                masks[~masks.any(axis=1), :3] = prepared.get_legal_action_mask()[
                    ~masks.any(axis=1), 0, :3
                ]
                picked = np.where(masks, q.astype(np.float64), -np.inf).argmax(axis=1)
                for env in np.flatnonzero(live):
                    if self.config.epsilon and float(rngs[int(env)].random()) < self.config.epsilon:
                        picked[env] = int(
                            rngs[int(env)].choice(
                                np.flatnonzero(prepared.get_legal_action_mask()[env, 0])
                            )
                        )
                grid[live, 0] = picked[live]
                selector_elapsed[0] += time.perf_counter() - selector_started
                return grid

            started = time.perf_counter()
            sim.step_with_policy(choose, active_env_mask=live)
            self._profile["sim_step_s"] += time.perf_counter() - started - selector_elapsed[0]
            started = time.perf_counter()
            merged.runtime.observe_step(sim)
            self._profile["runtime_observe_s"] += time.perf_counter() - started
            self._profile["steps"] += 1
            mass += np.where(live & sim.alive[:, 0], sim.length[:, 0], 0)
            live &= sim.alive[:, 0]
        leaf_values = np.zeros(sim.E, dtype=np.float64)
        if live.any():
            captured: dict[str, np.ndarray] = {}

            class _LeafReady(Exception):
                pass

            def leaf_selector(prepared: Any) -> np.ndarray:
                started = time.perf_counter()
                merged.runtime.prepare(prepared)
                self._profile["runtime_prepare_s"] += time.perf_counter() - started
                captured["q"] = self._q(prepared, hero_rows, merged.sources[0].hero.net)
                captured["masks"] = prepared.get_legal_action_mask()[:, 0].astype(bool)
                # This is a Watch-boundary read.  Do not consume a movement or
                # advance any inactive row merely to obtain the leaf value.
                raise _LeafReady

            try:
                sim.step_with_policy(leaf_selector, active_env_mask=live)
            except _LeafReady:
                pass
            q = captured["q"]
            masks = captured["masks"]
            if self.leaf == "value":
                leaf_values[live] = q[live].mean(axis=1)
            else:
                leaf_values[live] = np.max(np.where(masks[live], q[live], -np.inf), axis=1)
        totals = (mass + self.config.leaf_scale * leaf_values).reshape(
            W, NUM_ACTIONS, self.config.rollouts
        )
        mean_scores = totals.mean(axis=2)
        actions = base.copy()
        diagnostics: list[dict[str, Any]] = []
        for source in range(W):
            scores = {
                int(action): float(mean_scores[source, action])
                for action in np.flatnonzero(legal[source])
            }
            if eligible[source]:
                actions[source] = max(scores, key=scores.__getitem__)
            diagnostics.append(
                {
                    "q": initial_q[source].copy(),
                    "mask": legal[source].copy(),
                    "scores": scores,
                    "searched": bool(eligible[source]),
                    "common_random": self.common_random,
                    "resident_rows": int(sim.E),
                }
            )
        return actions, diagnostics
