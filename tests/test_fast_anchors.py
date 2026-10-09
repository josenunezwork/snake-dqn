"""FastProfileAnchorSimdPolicy makes exactly the gate anchor's decisions (redesign M1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.eval_engine import _ProfileAnchorSimdPolicy
from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
from src.simd_env.grid_scenarios import inject_serpentines
from src.simd_env.grid_sim import GridBatchSim


@pytest.mark.parametrize("kind", ["greedy_food", "random_safe"])
@pytest.mark.parametrize("big", [False, True])
def test_fast_anchor_equals_profile_anchor_every_frame(kind, big):
    seeds = [101, 202, 303, 404]
    cfg = BatchSimConfig(
        num_envs=len(seeds),
        num_snakes=6,
        game_width=600,
        game_height=600,
        initial_food=60,
        max_food=80,
        mechanics_version=2,
        max_capacity=1200,
    )
    sim = GridBatchSim(cfg, seeds=seeds, train_mode=False, allow_respawn=True)
    if big:
        inject_serpentines([sim], [300, 120, 60, 30, 10, 3], box_height=6)
    ref = _ProfileAnchorSimdPolicy(kind, seeds)
    fast = FastProfileAnchorSimdPolicy(kind, seeds)
    rng = np.random.default_rng(5)
    stats = {"rows": 0, "no_safe": 0, "turns": 0, "deaths": 0}

    def choose(prepared):
        masks = prepared.get_resolved_action_mask()
        rows = np.argwhere(prepared.get_alive())
        # Shuffle row order: decisions must not depend on it.
        rows = rows[rng.permutation(len(rows))]
        m = masks[rows[:, 0], rows[:, 1]]
        a = ref.actions(m, prepared, rows)
        b = fast.actions(m, prepared, rows)
        np.testing.assert_array_equal(a, b)
        stats["rows"] += len(rows)
        stats["no_safe"] += int((~m[:, :3].any(axis=1)).sum())
        stats["turns"] += int((a != 1).sum())
        actions = np.ones((prepared.E, prepared.S), dtype=np.int64)
        actions[rows[:, 0], rows[:, 1]] = a
        # Occasional boosts keep the world changing (decisions still compared above).
        boost = rng.random(len(rows)) < 0.1
        actions[rows[boost, 0], rows[boost, 1]] += 3 * m[boost, a[boost] + 3]
        return actions

    for _ in range(250):
        sim.step_with_policy(choose)
        stats["deaths"] += int(sim.get_done().sum())
    assert stats["rows"] > 1000 and stats["turns"] > 50
    if big:
        assert stats["deaths"] > 0


def test_fast_anchor_no_food_and_empty_rows():
    cfg = BatchSimConfig(num_envs=1, num_snakes=2, game_width=200, game_height=200)
    sim = GridBatchSim(cfg, seeds=[3], train_mode=False, allow_respawn=True)
    for e in range(sim.E):
        for cell in list(sim.food_cells[e]):
            sim.food_cells[e].remove(cell)
            sim.food_set[e].discard(cell)
        sim._rebuild_env_grids(e)
    sim.frame[:] = 1
    ref = _ProfileAnchorSimdPolicy("greedy_food", [3])
    fast = FastProfileAnchorSimdPolicy("greedy_food", [3])
    rows = np.array([[0, 0], [0, 1]])
    for bits in range(64):
        m = np.array([[(bits >> k) & 1 for k in range(6)]] * 2, dtype=bool)
        np.testing.assert_array_equal(ref.actions(m, sim, rows), fast.actions(m, sim, rows))
    assert fast.actions(np.zeros((0, 6), bool), sim, np.zeros((0, 2), np.int64)).shape == (0,)
