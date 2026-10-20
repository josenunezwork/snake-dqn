"""Calibration-only rowwise oracle, adapted from Phase 0 at 38258c42.

Never used as a production fallback. CRN shares epsilon draws across first
moves; food streams clone the source RNG identically and scripted opponents
are keyed by seed/slot/frame. Q.mean equals dueling V (roundoff aside).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.phase0_search_20261019.teacher import (
    SearchConfig,
    _mask_argmax,
    _q_and_legal,
    _replicate,
    _rng,
)
from research.t3_rollout_labels_20261015 import bank_engine


def prefix_trace(
    source: bank_engine.CapturedWorld,
    actions: list[int],
    config: SearchConfig,
    decision: int,
    *,
    common_random: bool = True,
    horizons: list[int] | None = None,
) -> dict:
    """Record exact CRN mass/Q prefixes during ONE full continuation.

    Prefix Q is read at the next Watch selection boundary, after lifecycle
    preparation but before movement. That same read selects the next action;
    lifecycle preparation is never repeated just to collect a prefix.
    """
    horizons = sorted(set(horizons or [config.horizon]) | {config.horizon})
    if min(horizons) < 1 or max(horizons) > config.horizon:
        raise ValueError("prefix horizons must be within the full horizon")
    checkpoints = {}
    from src.simd_env.eval_engine import _dispatch_actions

    branches = [(action, rollout) for action in actions for rollout in range(config.rollouts)]
    replica = _replicate(source, len(branches))
    sim, runtime = replica.sim, replica.runtime
    rngs = [
        _rng(source.seed, decision, 0 if common_random else action, rollout)
        for action, rollout in branches
    ]
    forced = np.asarray([action for action, _ in branches], dtype=np.int64)

    actions_grid = np.ones((sim.E, sim.S), dtype=np.int64)
    _dispatch_actions(
        sim, sim.get_resolved_action_mask(), actions_grid, replica.heroes, replica.opponents
    )
    actions_grid[:, 0] = forced
    sim.step(actions_grid)
    runtime.observe_step(sim)
    mass = np.where(sim.alive[:, 0], sim.length[:, 0], 0).astype(np.float64)
    active = sim.alive[:, 0].copy()
    for completed in range(1, config.horizon):
        if not active.any():
            break
        prefix_q = np.zeros((sim.E, 6), dtype=np.float64)
        prefix_legal = np.zeros((sim.E, 6), dtype=bool)

        def choose(prepared: Any) -> np.ndarray:
            runtime.prepare(prepared)
            grid = np.ones((sim.E, sim.S), dtype=np.int64)
            _dispatch_actions(
                prepared,
                prepared.get_resolved_action_mask(),
                grid,
                replica.heroes,
                replica.opponents,
            )
            # Hero evaluation is intentionally rowwise; frozen opponents remain
            # batched through their shared runtime/policy instance.
            for env in np.flatnonzero(active):
                row = bank_engine.CapturedWorld(
                    prepared,
                    runtime,
                    replica.heroes[int(env)],
                    replica.opponents[int(env)],
                    source.seed,
                    source.mix,
                    source.kinds,
                )
                q, legal, select_mask = _q_and_legal(row, int(env))
                prefix_q[int(env)] = q
                prefix_legal[int(env)] = legal
                choice = _mask_argmax(q, select_mask)
                if config.epsilon and float(rngs[int(env)].random()) < config.epsilon:
                    choice = int(rngs[int(env)].choice(np.flatnonzero(legal)))
                grid[int(env), 0] = choice
            return grid

        prefix_mass = mass.copy() if completed in horizons else None
        sim.step_with_policy(choose, active_env_mask=active)
        if completed in horizons:
            checkpoints[completed] = _checkpoint(prefix_mass, prefix_q, prefix_legal)
        runtime.observe_step(sim)
        mass += np.where(sim.alive[:, 0] & active, sim.length[:, 0], 0)
        active &= sim.alive[:, 0]
    # Read the leaf at the real next Watch boundary, then abort before movement.
    leaf_q = np.zeros((sim.E, 6), dtype=np.float64)
    leaf_legal = np.zeros((sim.E, 6), dtype=bool)
    live = sim.alive[:, 0].copy()
    if live.any():

        class _LeafReady(Exception):
            pass

        def leaf_selector(prepared: Any) -> np.ndarray:
            runtime.prepare(prepared)
            for env in np.flatnonzero(live):
                row = bank_engine.CapturedWorld(
                    prepared,
                    runtime,
                    replica.heroes[int(env)],
                    replica.opponents[int(env)],
                    source.seed,
                    source.mix,
                    source.kinds,
                )
                leaf_q[int(env)], leaf_legal[int(env)], _ = _q_and_legal(row, int(env))
            raise _LeafReady

        try:
            sim.step_with_policy(leaf_selector, active_env_mask=live)
        except _LeafReady:
            pass
    final = _checkpoint(mass, leaf_q, leaf_legal)
    for horizon in horizons:
        # An all-dead batch terminates early: remaining mass is zero, so every
        # subsequent requested prefix has this same cumulative total.
        checkpoints.setdefault(horizon, final)
    return {
        "branches": branches,
        "checkpoints": checkpoints,
        "horizon": config.horizon,
        "rollouts": config.rollouts,
    }


def _checkpoint(mass: np.ndarray, q: np.ndarray, legal: np.ndarray) -> dict:
    live = legal.any(axis=1)
    maximum = np.zeros(len(q), dtype=np.float64)
    maximum[live] = np.max(np.where(legal[live], q[live], -np.inf), axis=1)
    return {"mass": mass.copy(), "max": maximum, "value": np.where(live, q.mean(axis=1), 0.0)}


def trace_scores(trace: dict, horizon: int, rollouts: int, leaf: str, scale: float) -> dict:
    """Reduce a stored prefix in segment-frames; scale has segment-frames/Q units."""
    if leaf not in ("max", "value") or not 0 < rollouts <= trace["rollouts"]:
        raise ValueError("invalid leaf or rollout prefix")
    checkpoint = trace["checkpoints"][horizon]
    totals = checkpoint["mass"] + scale * checkpoint[leaf]
    return {
        action: float(
            np.mean(
                [
                    totals[i]
                    for i, (a, n) in enumerate(trace["branches"])
                    if a == action and n < rollouts
                ]
            )
        )
        for action in dict.fromkeys(a for a, _ in trace["branches"])
    }


def scores(
    source: bank_engine.CapturedWorld,
    actions: list[int],
    config: SearchConfig,
    decision: int,
    *,
    common_random: bool = True,
    leaf: str = "max",
) -> dict[int, float]:
    """Independent-search compatibility oracle, retaining the original RNG draws."""
    trace = prefix_trace(source, actions, config, decision, common_random=common_random)
    return trace_scores(trace, config.horizon, config.rollouts, leaf, config.leaf_scale)
