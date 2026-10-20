"""Chronological branch-point batches with exact independent v8 carry.

There are at most eight source decisions and six forced actions per source.
The prepared worlds are reused throughout a batch. Native Grid kernels operate
over all active rows together; policy forwards remain rowwise for exact B8
action identity. All scratch belongs to the calling worker thread.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

import numpy as np

from research.t34_shadow.shadow import clone_world_exact

MAX_BRANCH_POINTS = 8


def evaluate_decisions(
    decisions: Sequence[Any],
    horizons: Sequence[int],
    *,
    cancel_event: Any = None,
    on_complete: Callable[[int, dict[int, Any]], bool] | None = None,
) -> list[dict[int, Any]]:
    """Evaluate one bounded chronological window in one native simulator batch."""
    from research.t33_batched_teacher.teacher import merge_worlds
    from research.t34_pathb.engine_v2 import NonFiniteQ, SearchOutcome, _workspace
    from src.simd_env.ego2s_policy import forward_q
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    if not 1 <= len(decisions) <= MAX_BRANCH_POINTS or len(decisions) != len(horizons):
        raise ValueError("branch-point batch must contain 1..8 decisions and matching horizons")
    if any(not 1 <= int(horizon) <= 500 for horizon in horizons):
        raise ValueError("continuation horizon must be in 1..500")
    if any(decisions[i].frame >= decisions[i + 1].frame for i in range(len(decisions) - 1)):
        raise ValueError("branch-point window must be chronological")
    sources, legal_masks = [], []
    for decision in decisions:
        source = clone_world_exact(decision.world)
        for policy in source.opponents:
            if hasattr(policy, "veto_slots"):
                policy.veto_slots = frozenset()
        sources.append(source)
        mask = np.asarray(decision.legal, dtype=bool).copy()
        mask[decision.bare_action] = True
        legal_masks.append(mask)
    merged = merge_worlds(sources, 1, legal_masks, horizon=max(horizons))
    sim, runtime, layout = merged.sim, merged.runtime, merged.layout
    source_ids = np.asarray(layout.source, dtype=np.int64)
    groups: dict[Any, list[tuple[int, int]]] = {}
    # Keep T33's expanded-seed anchor policies, filtering the exact source rows.
    anchor_groups = merged.policy_groups
    for source_index, decision in enumerate(decisions):
        envs = np.flatnonzero(source_ids == source_index).tolist()
        for slot, original in enumerate(decision.world.opponents, 1):
            if not hasattr(original, "veto_slots"):
                for policy, slots in anchor_groups.items():
                    rows = [
                        tuple(map(int, row))
                        for row in slots
                        if int(row[0]) in envs and int(row[1]) == slot
                    ]
                    if rows:
                        groups.setdefault(policy, []).extend(rows)
                continue
            policy = copy.copy(original)
            policy.__dict__ = dict(vars(original))
            policy.runtime = runtime
            for name in ("veto_counters", "boost_counters", "live_vetoes"):
                carried = getattr(original, name, None)
                if isinstance(carried, dict):
                    setattr(
                        policy,
                        name,
                        {env: copy.deepcopy(carried[0]) for env in envs if 0 in carried},
                    )
            groups[policy] = [(env, slot) for env in envs]
            for env in envs:
                merged.opponents[env][slot - 1] = policy
    policy_groups = {policy: np.asarray(rows, dtype=np.int64) for policy, rows in groups.items()}
    grid, active = _workspace(sim.E, sim.S)
    active[...] = np.asarray(layout.active, dtype=bool)
    wanted = active.copy()
    row_horizons = np.asarray(horizons, dtype=np.int64)[source_ids]
    alive_steps = np.zeros(sim.E, dtype=np.int64)
    executed = np.zeros(sim.E, dtype=np.int64)
    masses = np.zeros(sim.E, dtype=np.float64)
    results: list[dict[int, Any]] = [{} for _ in decisions]
    source_rows = [
        np.flatnonzero(wanted & (source_ids == source)) for source in range(len(decisions))
    ]
    reported = np.zeros(len(decisions), dtype=bool)

    def complete_source(source: int) -> dict[int, Any]:
        outcomes: dict[int, Any] = {}
        for env in source_rows[source]:
            action = int(layout.action[env])
            survives = bool(sim.alive[env, 0] and executed[env] == row_horizons[env])
            outcomes[action] = SearchOutcome(
                action,
                survives,
                int(alive_steps[env]),
                float(masses[env]),
                int(row_horizons[env]),
                death_frame=None if survives else decisions[source].frame + int(executed[env]) - 1,
            )
        return outcomes

    def opponents() -> None:
        for policy, slots in policy_groups.items():
            live_rows = slots[active[slots[:, 0]] & sim.alive[slots[:, 0], slots[:, 1]]]
            if len(live_rows):
                masks = sim.get_resolved_action_mask()[live_rows[:, 0], live_rows[:, 1]]
                grid[live_rows[:, 0], live_rows[:, 1]] = policy.actions(masks, sim, live_rows)

    for offset in range(int(max(horizons))):
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError("Path B v2 branch window cancelled")
        if not active.any():
            break
        grid.fill(1)
        if offset == 0:
            opponents()
            grid[:, 0] = layout.action
            sim.step(grid, active_env_mask=active)
        else:

            def choose(prepared: Any) -> np.ndarray:
                runtime.prepare(prepared, rows_mask=active)
                opponents()
                envs = np.flatnonzero(active)
                rows = np.column_stack((envs, np.zeros(len(envs), dtype=np.int64)))
                obs = build_ego_raster(
                    prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=rows
                )
                q = forward_q(decisions[0].world.hero.net, obs, "rowwise").numpy()
                if not np.isfinite(q).all():
                    raise NonFiniteQ("nonfinite B8 continuation Q")
                mask = prepared.get_advisory_action_mask()[envs, 0].astype(bool)
                mask[~mask.any(axis=1), :3] = True
                grid[envs, 0] = np.where(mask, q.astype(np.float64), -np.inf).argmax(axis=1)
                return grid

            sim.step_with_policy(choose, active_env_mask=active)
        runtime.observe_step(sim, rows_mask=active)
        executed += active.astype(np.int64)
        alive_steps += (active & sim.alive[:, 0]).astype(np.int64)
        masses += np.where(active & sim.alive[:, 0], sim.length[:, 0], 0)
        active &= sim.alive[:, 0] & (executed < row_horizons)
        for source in range(len(decisions)):
            if reported[source] or active[source_rows[source]].any():
                continue
            reported[source] = True
            results[source] = complete_source(source)
            if on_complete is not None and on_complete(source, results[source]):
                return results
    return results
