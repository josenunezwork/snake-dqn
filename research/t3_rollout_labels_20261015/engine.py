"""Bitwise-faithful batched continuations of a prepared T1 :class:`Replay`.

The first action is applied to one independent clone per candidate.  Later
actions are selected by the unmodified T1 B8/opponent dispatch path.  Network
inference therefore remains rowwise, as it was for the census and T1.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Sequence

import numpy as np

from research.b8_death_census_20261014 import census
from research.t1_commitment_20261015.replay import Replay


def verify_expand(
    prepared: Any, bare: int, q: Sequence[float], mask: Sequence[bool], depth: int
) -> dict[str, Any]:
    """Reuse T3's forced-first-action continuations with a clone/step adapter.

    The adapter owns the simulator, policy and private RNG. Verify the bare
    action first; only on failure expand alternatives in lockstep. This seam
    supports the real live simulator without duplicating any game dynamics.
    """
    if depth < 1 or len(q) != 6 or len(mask) != 6 or not np.isfinite(q).all():
        raise ValueError("invalid verify/expand input")
    choices = [a for a in range(6) if mask[a]]
    if bare not in choices:
        raise ValueError("bare action must belong to the resolved mask")

    def advance(branches: dict[int, Any]) -> dict[int, dict[str, Any]]:
        counts = {a: 0 for a in branches}
        for frame in range(depth):
            for action, branch in branches.items():
                if branch.alive:
                    branch.step(action if frame == 0 else None)
                    counts[action] += int(branch.alive)
        return {
            a: {"survives": branch.alive, "frames_survived": counts[a]}
            for a, branch in branches.items()
        }

    outcomes = advance({bare: prepared.clone()})
    if outcomes[bare]["survives"]:
        return {"proposed_action": bare, "outcomes": outcomes}
    outcomes.update(advance({a: prepared.clone() for a in choices if a != bare}))
    survivors = [a for a in choices if outcomes[a]["survives"]]
    pool = survivors or choices
    if not survivors:
        longest = max(outcomes[a]["frames_survived"] for a in pool)
        pool = [a for a in pool if outcomes[a]["frames_survived"] == longest]
    best_q = max(float(q[a]) for a in pool)
    tied = [a for a in pool if float(q[a]) == best_q]
    chosen = bare if bare in tied else min(tied)
    return {"proposed_action": chosen, "outcomes": outcomes}


def _restore_grid_views(sim: Any) -> None:
    """Restore GridBatchSim's padded-grid views after changing E."""
    if getattr(sim, "_grids_ready", False) and hasattr(sim, "_owner_pad"):
        pad, height, width = sim._grid_pad, sim.grid_h, sim.grid_w
        sim._owner = sim._owner_pad[:, pad : pad + height, pad : pad + width]
        sim._slot = sim._slot_pad[:, pad : pad + height, pad : pad + width]
        sim._food = sim._food_pad[:, pad : pad + height, pad : pad + width]


def _repeat_envs(prepared: Replay, count: int) -> Replay:
    """Make ``count`` independent copies of a prepared E=1 T1 world."""
    if prepared.sim.E != 1:
        raise ValueError("batched_rollout requires one prepared world (sim.E == 1)")
    if count < 1:
        raise ValueError("at least one first action is required")
    sim = census.clone_sim(prepared.sim)
    for name, value in list(vars(sim).items()):
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == 1:
            if value.dtype.hasobject:
                copied = np.empty((count, *value.shape[1:]), dtype=value.dtype)
                for env in range(count):
                    copied[env] = copy.deepcopy(value[0])
                setattr(sim, name, copied)
            else:
                setattr(sim, name, np.repeat(value, count, axis=0))
        elif isinstance(value, list) and len(value) == 1:
            setattr(sim, name, [copy.deepcopy(value[0]) for _ in range(count)])
    sim.E = count
    # These are derived from E/S, while _det can cache a shape-specialized helper.
    sim._env_index = np.arange(count, dtype=np.int64)[:, None]
    sim._snake_index = np.arange(sim.S, dtype=np.int64)[None, :]
    sim._det = None
    # T1 observer snapshots occur inside step_with_policy.  This batch takes
    # ownership of its clone and consumes the already prepared first transition.
    sim._policy_phase_prepared = True
    sim._policy_step_in_progress = False
    _restore_grid_views(sim)
    # Fast anchors are immutable after construction.  Stateful opponent runtime
    # is deliberately rejected until it has an independent batched contract.
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

    if not all(isinstance(policy, FastProfileAnchorSimdPolicy) for policy in prepared.opponents[0]):
        raise TypeError("batched_rollout currently supports only stateless fast-anchor opponents")
    # Reconstruct once per opponent slot with K seed entries.  This retains
    # dispatch grouping while making random_safe's env-indexed hash exact.
    anchors = [
        FastProfileAnchorSimdPolicy(policy.kind, [prepared.seeds[0]] * count)
        for policy in prepared.opponents[0]
    ]
    return Replay(
        sim,
        prepared.hero,
        [anchors] * count,
        [prepared.seeds[0]] * count,
        [copy.deepcopy(prepared.kinds[0]) for _ in range(count)],
    )


def extract_prepared_env(prepared: Replay, env: int) -> Replay:
    """Copy one prepared environment from an E>1 replay as a prepared E=1 replay."""
    if not 0 <= env < prepared.sim.E:
        raise IndexError(f"environment {env} is outside [0, {prepared.sim.E})")
    sim = census.clone_sim(prepared.sim)
    old_e = prepared.sim.E
    for name, value in list(vars(sim).items()):
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == old_e:
            setattr(sim, name, value[env : env + 1].copy())
        elif isinstance(value, list) and len(value) == old_e:
            setattr(sim, name, [copy.deepcopy(value[env])])
    sim.E = 1
    sim._env_index = np.zeros((1, 1), dtype=np.int64)
    sim._snake_index = np.arange(sim.S, dtype=np.int64)[None, :]
    sim._det = None
    _restore_grid_views(sim)
    sim._policy_phase_prepared = True
    sim._policy_step_in_progress = False
    if hasattr(sim, "_active_env_mask"):
        del sim._active_env_mask
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

    if not all(
        isinstance(policy, FastProfileAnchorSimdPolicy) for policy in prepared.opponents[env]
    ):
        raise TypeError(
            "extract_prepared_env currently supports only stateless fast-anchor opponents"
        )
    anchors = [
        FastProfileAnchorSimdPolicy(policy.kind, [prepared.seeds[env]])
        for policy in prepared.opponents[env]
    ]
    return Replay(
        sim,
        prepared.hero,
        [anchors],
        [prepared.seeds[env]],
        [copy.deepcopy(prepared.kinds[env])],
    )


def batched_rollout(
    prepared: Replay,
    actions: Sequence[int],
    decision: int,
    depth: int = 300,
    *,
    record_outcomes: bool = False,
) -> dict[int, dict]:
    """Roll every candidate first action through B8 greedy continuations.

    Returns the same outcome fields as ``t1.study.rollout`` (apart from T1's
    optional death-reproduction control), keyed by integer first action.  The
    receipt fields on the result are intentionally per-action so callers can
    sum T1-comparable environment frames.  ``_batch_receipt`` carries timing.
    """
    choices = [int(action) for action in actions]
    if len(set(choices)) != len(choices):
        raise ValueError("candidate first actions must be unique")
    if any(action < 0 or action > 5 for action in choices):
        raise ValueError("actions must be in [0, 5]")
    if depth < 1:
        raise ValueError("depth must be positive")
    horizon = min(int(depth), census.HORIZON - int(decision))
    if horizon < 1:
        raise ValueError("decision is outside the rollout horizon")
    started = time.perf_counter()
    branch = _repeat_envs(prepared, len(choices))
    sim = branch.sim
    # The input is a pre-action T1 snapshot: direct step consumes precisely that
    # prepared policy phase, with one forced action in each independently cloned row.
    sim.step(branch.actions(choices))
    executed = np.ones(sim.E, dtype=np.int64)
    alive_transitions = sim.alive[:, 0].astype(np.int64)
    active = sim.alive[:, 0].copy()
    while int(executed.max()) < horizon and bool(active.any()):
        sim.step_with_policy(lambda _sim: branch.actions(), active_env_mask=active)
        executed[active] += 1
        alive_transitions[active] += sim.alive[active, 0].astype(np.int64)
        active &= sim.alive[:, 0]
    wall_seconds = time.perf_counter() - started
    outcomes: dict[int, dict] = {}
    for env, action in enumerate(choices):
        end = int(executed[env])
        alive = bool(sim.alive[env, 0])
        outcome = {
            "action": action,
            "survives": alive and end == horizon,
            "horizon": horizon,
            "frames_survived": int(alive_transitions[env]),
            "frames_executed": end,
            "death_decision": None if alive else int(decision) + end - 1,
            "control_reproduced": None,
            "env_frames": end,
        }
        if record_outcomes:
            outcome["final_length"] = int(sim.length[env, 0]) if alive else 0
        outcomes[action] = outcome
    outcomes["_batch_receipt"] = {
        "candidate_count": len(choices),
        "env_frames": int(executed.sum()),
        "wall_seconds": wall_seconds,
        "env_frames_per_second": float(executed.sum() / wall_seconds) if wall_seconds else 0.0,
        "decisions_per_second": float(1.0 / wall_seconds) if wall_seconds else 0.0,
    }
    return outcomes
