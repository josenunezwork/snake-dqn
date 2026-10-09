"""Ego2sNet checkpoint contract and the greedy ego2s SIMD policy (redesign M2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint, save_ego2s_checkpoint
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.ego2s_policy import Ego2sSimdPolicy
from src.simd_env.ego_raster import build_ego_raster
from src.simd_env.grid_sim import GridBatchSim


def test_checkpoint_roundtrip_and_contract(tmp_path):
    torch.manual_seed(0)
    net = Ego2sNet().eval()
    path = tmp_path / "s.pth"
    save_ego2s_checkpoint(path, net, {"note": 1})
    loaded = load_ego2s_checkpoint(str(path))
    x = (
        torch.randint(0, 255, (3, 6, 31, 31), dtype=torch.uint8),
        torch.randint(0, 255, (3, 4, 37, 37), dtype=torch.uint8),
        torch.rand(3, 12),
    )
    with torch.no_grad():
        assert torch.equal(net(*x), loaded(*x))
    torch.save({"obs_spec": "vector61", "state_dict": {}}, tmp_path / "bad.pth")
    with pytest.raises(ValueError):
        load_ego2s_checkpoint(str(tmp_path / "bad.pth"))


def test_policy_acts_inside_the_mask(tmp_path):
    torch.manual_seed(1)
    path = tmp_path / "s.pth"
    save_ego2s_checkpoint(path, Ego2sNet())
    policy = Ego2sSimdPolicy(str(path))
    cfg = BatchSimConfig(num_envs=3, num_snakes=4, mechanics_version=2)
    sim = GridBatchSim(cfg, seeds=[1, 2, 3])
    for _ in range(30):
        masks = sim.get_resolved_action_mask()
        rows = np.argwhere(sim.get_alive())
        m = masks[rows[:, 0], rows[:, 1]]
        a = policy.actions(m, sim, rows)
        ok = m[np.arange(len(a)), a] | ~m.any(axis=1)
        assert ok.all()
        # Same rows, same observation -> same actions (no hidden state).
        assert np.array_equal(a, policy.actions(m, sim, rows))
        obs = build_ego_raster(sim, rows=rows)
        assert obs["local"].shape[0] == len(rows)
        acts = np.ones((sim.E, sim.S), dtype=np.int64)
        acts[rows[:, 0], rows[:, 1]] = a
        sim.step(acts)
