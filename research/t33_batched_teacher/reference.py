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


def scores(
    source: bank_engine.CapturedWorld,
    actions: list[int],
    config: SearchConfig,
    decision: int,
    *,
    common_random: bool = True,
    leaf: str = "max",
) -> dict[int, float]:
    """Evaluate ``action x rollout`` independent rows in one SIMD continuation."""
    if leaf not in ("max", "value"):
        raise ValueError("leaf must be max or value")
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
    for _ in range(1, config.horizon):
        if not active.any():
            break

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
                choice = _mask_argmax(q, select_mask)
                if config.epsilon and float(rngs[int(env)].random()) < config.epsilon:
                    choice = int(rngs[int(env)].choice(np.flatnonzero(legal)))
                grid[int(env), 0] = choice
            return grid

        sim.step_with_policy(choose, active_env_mask=active)
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
    totals = mass + config.leaf_scale * np.where(
        live,
        np.array(
            [
                (q.mean() if leaf == "value" else np.max(q[m])) if m.any() else 0.0
                for q, m in zip(leaf_q, leaf_legal)
            ]
        ),
        0.0,
    )
    return {
        action: float(
            np.mean([totals[index] for index, pair in enumerate(branches) if pair[0] == action])
        )
        for action in actions
    }
