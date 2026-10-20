"""One-row-per-world GridBatchSim stepping for the T34 v2 collector.

The public collector owns long-lived single-world objects because their exact
digests, export traces, and shadow rings are part of the evidence contract.
This module temporarily assembles those prepared objects into a real batched
``GridBatchSim`` for a single transition, then projects every environment and
its Vector61/v8 carry back onto its original owner.  It is intentionally a
transition seam, not a second simulator.
"""

from __future__ import annotations

import copy
import types
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


def _copy_environment(source: Any, source_env: int, destination: Any) -> None:
    """Copy one merged environment into an E=1 source-world object."""
    for from_owner, to_owner in (
        (source.sim, destination.sim),
        (source.runtime, destination.runtime),
        (getattr(source.runtime, "memory", None), getattr(destination.runtime, "memory", None)),
    ):
        if from_owner is None or to_owner is None:
            continue
        for name, value in vars(from_owner).items():
            if name in {"_env_index", "_snake_index", "_det", "_active_env_mask"}:
                continue
            if from_owner is source.sim and name == "_food_array":
                continue
            if name in {"bodies", "_owner_pad", "_slot_pad", "_food_pad"}:
                # Direct assembly preserves the original ring capacity and
                # padded grid shape, so this is an exact E=1 slice.
                if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == source.sim.E:
                    setattr(to_owner, name, value[source_env : source_env + 1].copy())
                continue
            if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == source.sim.E:
                copied = value[source_env : source_env + 1].copy()
                setattr(to_owner, name, copied)
            elif isinstance(value, list) and len(value) == source.sim.E:
                setattr(to_owner, name, [copy.deepcopy(value[source_env])])
    if hasattr(source, "food_before"):
        width = int(source.food_widths[source_env])
        destination.sim._food_array = source.sim._food_array[
            source_env : source_env + 1, :width
        ].copy()
    # Restoring views is required after copying the padded grid arrays.
    from research.t3_rollout_labels_20261015.engine import _restore_grid_views

    destination.sim._det = None
    if hasattr(source.sim, "_policy_phase_prepared"):
        destination.sim._policy_phase_prepared = source.sim._policy_phase_prepared
    elif hasattr(destination.sim, "_policy_phase_prepared"):
        del destination.sim._policy_phase_prepared
    destination.sim._policy_step_in_progress = False
    _restore_grid_views(destination.sim)


def _restore_policy_carry(merged: Any, env: int, destination: Any) -> None:
    """Project native Vector61/v8 carry from the private merged policy clone."""
    for source_policy, target_policy in zip(merged.opponents[env], destination.opponents):
        for name in ("veto_counters", "boost_counters", "live_vetoes"):
            value = getattr(source_policy, name, None)
            if isinstance(value, dict):
                restored = getattr(target_policy, name, None)
                if not isinstance(restored, dict):
                    restored = {}
                    setattr(target_policy, name, restored)
                restored.clear()
                if env in value:
                    restored[0] = copy.deepcopy(value[env])
        if hasattr(source_policy, "last_veto"):
            last_veto = copy.deepcopy(source_policy.last_veto)
            if isinstance(last_veto, dict) and isinstance(last_veto.get("rows"), np.ndarray):
                last_veto["rows"][:, 0] = 0
            target_policy.last_veto = last_veto


def _merged_roster(merged: Any, worlds: Sequence[Any]) -> None:
    """Install one independent carried opponent roster for each merged world."""
    from research.t34_exit_b8.engine import _clone_merged_policy, _merged_policy_key
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

    cache: dict[tuple[Any, ...], Any] = {}
    roster = []
    for env, world in enumerate(worlds):
        row = []
        for slot, policy in enumerate(world.opponents, 1):
            if isinstance(policy, FastProfileAnchorSimdPolicy):
                key = ("anchor", policy.kind)
                if key not in cache:
                    cache[key] = FastProfileAnchorSimdPolicy(
                        policy.kind, [int(item.seed) for item in worlds]
                    )
            else:
                key = _merged_policy_key(policy, env, slot)
                if key not in cache:
                    cache[key] = _clone_merged_policy(
                        policy, merged.runtime, np.asarray([env], dtype=np.int64)
                    )
            row.append(cache[key])
        roster.append(row)
    merged.opponents = roster


def _merge_prepared_worlds(worlds: Sequence[Any]) -> Any:
    """Assemble one physical environment per source without ring re-capping.

    T33's continuation assembler deliberately grows every body ring for a
    future horizon.  That is correct for branches but changes ring pointers,
    making it unsuitable for a serial-identity root projection.  This smaller
    assembler performs only an E-axis concatenate, retaining every source's
    original capacity, pointers, padding, food order, and RNG object.
    """
    from research.b8_death_census_20261014 import census
    from research.t3_rollout_labels_20261015.engine import _restore_grid_views
    from research.t33_batched_teacher.teacher import BranchLayout, _runtime_for

    first = worlds[0]
    sim0 = first.sim
    if any(type(world.sim) is not type(sim0) or world.sim.S != sim0.S for world in worlds):
        raise ValueError("lockstep sources must share a simulator shape")
    expected = replace(sim0.cfg, num_envs=1)
    if any(replace(world.sim.cfg, num_envs=1) != expected for world in worlds):
        raise ValueError("lockstep sources must share simulator configuration")
    if any(int(world.sim.cap) != int(sim0.cap) for world in worlds):
        raise ValueError("lockstep sources must share body ring capacity")
    sim = census.clone_sim(sim0)
    food_before = [world.sim._food_array.copy() for world in worlds]
    food_widths = [int(value.shape[1]) for value in food_before]
    for name, destination in list(vars(sim).items()):
        if isinstance(destination, np.ndarray) and destination.ndim and destination.shape[0] == 1:
            values = [getattr(world.sim, name) for world in worlds]
            if all(isinstance(value, np.ndarray) and value.shape[0] == 1 for value in values):
                if name == "_food_array":
                    width = max(value.shape[1] for value in values)
                    joined = np.zeros((len(values), width, 2), dtype=destination.dtype)
                    for env, value in enumerate(values):
                        joined[env, : value.shape[1]] = value[0]
                    setattr(sim, name, joined)
                else:
                    setattr(sim, name, np.concatenate(values, axis=0))
        elif isinstance(destination, list) and len(destination) == 1:
            values = [getattr(world.sim, name) for world in worlds]
            if all(isinstance(value, list) and len(value) == 1 for value in values):
                setattr(sim, name, [copy.deepcopy(value[0]) for value in values])
    sim.E = len(worlds)
    sim.cfg = replace(sim.cfg, num_envs=sim.E)
    sim._env_index = np.arange(sim.E, dtype=np.int64)[:, None]
    sim._snake_index = np.arange(sim.S, dtype=np.int64)[None, :]
    sim._det = None
    sim._policy_phase_prepared = True
    sim._policy_step_in_progress = False
    if hasattr(sim, "_active_env_mask"):
        del sim._active_env_mask
    _restore_grid_views(sim)
    # The merged simulator must allocate a common food mirror.  Track what
    # each E=1 serial simulator would have allocated so scatter restores its
    # original capacity and untouched tail bytes exactly.
    food_context: list[int | None] = [None]
    original_ensure = sim._ensure_food_array_capacity

    def ensure(self: Any, count: int) -> None:
        env = food_context[0]
        if env is not None and count > food_widths[env]:
            food_widths[env] = max(int(count), 2 * food_widths[env])
        original_ensure(count)

    def wrapped(method: Any):
        def call(self: Any, env: int, *args: Any, **kwargs: Any) -> Any:
            food_context[0] = int(env)
            try:
                return method(env, *args, **kwargs)
            finally:
                food_context[0] = None

        return call

    sim._ensure_food_array_capacity = types.MethodType(ensure, sim)
    sim._sync_food_array = types.MethodType(wrapped(sim._sync_food_array), sim)
    sim._spawn = types.MethodType(wrapped(sim._spawn), sim)
    sim._drop_corpse_v2 = types.MethodType(wrapped(sim._drop_corpse_v2), sim)
    controlled = np.zeros((sim.E, sim.S), dtype=bool)
    for env, world in enumerate(worlds):
        for slot, policy in enumerate(world.opponents, 1):
            controlled[env, slot] = hasattr(policy, "runtime")
    runtime = _runtime_for(list(worlds), sim, controlled)
    for owner_name in ("runtime", "memory"):
        target = runtime if owner_name == "runtime" else getattr(runtime, "memory", None)
        if target is None:
            continue
        source_values = [
            getattr(world.runtime, owner_name, None) if owner_name == "memory" else world.runtime
            for world in worlds
        ]
        for name, value in vars(target).items():
            if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == sim.E:
                values = [getattr(source, name, None) for source in source_values]
                if all(isinstance(item, np.ndarray) and item.shape[0] == 1 for item in values):
                    setattr(target, name, np.concatenate(values, axis=0))
    indices = np.arange(len(worlds), dtype=np.int64)
    return SimpleNamespace(
        sim=sim,
        runtime=runtime,
        opponents=[],
        layout=BranchLayout(
            indices,
            np.zeros(len(worlds), dtype=np.int64),
            indices,
            np.ones(len(worlds), dtype=bool),
        ),
        food_before=food_before,
        food_widths=food_widths,
    )


def step_prepared_worlds(
    worlds: Sequence[Any], hero_actions: Sequence[int], teacher: Any
) -> list[list[int]]:
    """Advance prepared roots in one compacted GridBatchSim step.

    ``hero_actions`` is supplied by the caller, so root, recorded-control, and
    shadow greedy transitions share one exact action-dispatch path.  A single
    legal action per source makes ``merge_worlds`` emit one active row per
    source; stable compaction discards its audit-only inactive action rows.
    """
    if len(worlds) != len(hero_actions):
        raise ValueError("worlds and hero actions must align")
    if not worlds:
        return []
    if not all(hasattr(world.sim, "cfg") and hasattr(world.sim, "E") for world in worlds):
        grids = []
        for world, action in zip(worlds, hero_actions):
            grid = teacher.actions(world, int(action))
            world.sim.step(grid)
            world.runtime.observe_step(world.sim)
            grids.append(np.asarray(grid[0], dtype=np.int64).tolist())
        return grids
    if any(int(world.sim.E) != 1 for world in worlds):
        raise ValueError("lockstep stepping requires one environment per source world")
    if any(not getattr(world.sim, "_policy_phase_prepared", False) for world in worlds):
        raise ValueError("lockstep stepping requires prepared source worlds")
    if len(worlds) == 1:
        world = worlds[0]
        grid = teacher.actions(world, int(hero_actions[0]))
        world.sim.step(grid)
        world.runtime.observe_step(world.sim)
        return [np.asarray(grid[0], dtype=np.int64).tolist()]

    for action in hero_actions:
        if not 0 <= int(action) < 6:
            raise ValueError("hero action must be in [0, 5]")
    merged = _merge_prepared_worlds(worlds)
    _merged_roster(merged, worlds)
    grid = np.ones((merged.sim.E, merged.sim.S), dtype=np.int64)
    teacher._dispatch_merged(merged, grid, np.ones(merged.sim.E, dtype=bool))
    grid[:, 0] = np.asarray(hero_actions, dtype=np.int64)
    merged.sim.step(grid)
    merged.runtime.observe_step(merged.sim)
    for env, world in enumerate(worlds):
        _copy_environment(merged, env, world)
        _restore_policy_carry(merged, env, world)
    return [np.asarray(row, dtype=np.int64).tolist() for row in grid]
