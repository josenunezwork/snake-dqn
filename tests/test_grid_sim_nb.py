"""Compiled GridBatchSim kernels == their NumPy reference versions, call by call.

The lockstep suite (``tests/test_grid_sim_parity.py``) already runs every scenario with
``jit`` on and off against BatchSim. This file compares the two implementations directly
at every call, including on paused worlds and on the death-heavy big-body regime.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simd_env import grid_sim_nb
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_scenarios import GreedySafePolicy, inject_serpentines
from src.simd_env.grid_sim import GridBatchSim


@pytest.mark.parametrize("mechanics", [1, 2])
@pytest.mark.parametrize("train_mode", [True, False])
def test_kernels_equal_numpy_every_call(monkeypatch, mechanics, train_mode):
    calls = {"detect": 0, "masks": 0, "events": 0}
    real_detect = grid_sim_nb.detect_collisions
    real_masks = grid_sim_nb.action_masks

    def detect(sim, n_trav):
        got = real_detect(sim, n_trav)
        ref = sim._detect_collisions(n_trav)
        for key in ref:
            np.testing.assert_array_equal(got[key], ref[key], err_msg=key)
        calls["detect"] += 1
        calls["events"] += int(ref["event_env"].sum())
        return got

    def masks(sim):
        got = real_masks(sim)
        np.testing.assert_array_equal(got, sim._compute_action_masks_numpy())
        calls["masks"] += 1
        return got

    monkeypatch.setattr(grid_sim_nb, "detect_collisions", detect)
    monkeypatch.setattr(grid_sim_nb, "action_masks", masks)
    cfg = BatchSimConfig(num_envs=4, num_snakes=6, mechanics_version=mechanics, max_capacity=1200)
    sim = GridBatchSim(cfg, seeds=[1, 2, 3, 4], train_mode=train_mode, allow_respawn=True, jit=True)
    inject_serpentines([sim], [800, 400, 150, 60, 20, 5], box_height=8)
    pol = GreedySafePolicy(11, boost_prob=0.3)
    active = np.ones(sim.E, dtype=bool)
    for frame in range(200):
        if frame == 50:
            active = np.array([True, False, True, True])
        if frame == 120:
            active = np.ones(sim.E, dtype=bool)
        sim.step(pol.actions(sim, sim.get_action_mask()), active_env_mask=active)
    assert calls["detect"] == 200 and calls["masks"] >= 200 and calls["events"] > 0


def test_jit_flag_selects_kernels():
    cfg = BatchSimConfig(num_envs=1, num_snakes=2)
    assert GridBatchSim(cfg, jit=True)._kernels is grid_sim_nb
    assert GridBatchSim(cfg, jit=False)._kernels is None
