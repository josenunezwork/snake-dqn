"""Focused CPU oracle for the device B8 observation port."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.ego_device import build_observation
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_sim import GridBatchSim


def _fixture() -> GridBatchSim:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=3,
        game_width=160,
        game_height=120,
        segment_size=10,
        initial_food=3,
        max_food=5,
        mechanics_version=2,
        max_capacity=32,
    )
    sim = GridBatchSim(cfg, seeds=[41], train_mode=True, allow_respawn=False, jit=False)
    # Exercise headings, a boosted enemy, food classes, and nontrivial TTLs.
    sim.step(np.array([[1, 4, 2]], dtype=np.int64))
    return sim


def _wrapped_ring_fixture() -> GridBatchSim:
    """A rotated ring makes physical slot order differ from segment age order."""
    sim = _fixture()
    for snake, (x0, y0) in enumerate(((9, 3), (9, 6), (9, 9))):
        sim.head_ptr[0, snake] = 2
        sim.seg_count[0, snake] = 6
        sim.length[0, snake] = 6
        sim.direction[0, snake] = 1
        for age in range(6):
            slot = (2 - age) % sim.cap
            sim.bodies[0, snake, slot] = (x0 - age, y0)
    sim._rebuild_env_grids(0, food=True)
    return sim


@pytest.mark.parametrize("factory", (_fixture, _wrapped_ring_fixture))
def test_device_b8_observation_matches_numba_reference_fixture(factory) -> None:
    ref = factory()
    sim = DeviceSim.from_reference(ref, device="cpu")
    rows = torch.tensor([[0, 0], [0, 1], [0, 2]], dtype=torch.long)
    got = build_observation(sim, rows)
    expected = build_ego_raster(
        ref, EgoRasterConfig(version="b"), backend="numba", rows=rows.numpy()
    )
    np.testing.assert_array_equal(got[0].numpy(), expected["local"])
    np.testing.assert_array_equal(got[1].numpy(), expected["global"])
    # Device float math is deliberately judged as measured divergence, rather
    # than a cross-backend bit-identity claim.  Spatial planes remain bytes.
    scalar_error = np.abs(got[2].numpy() - expected["scalars"])
    assert float(scalar_error.max()) <= 1e-6
    np.testing.assert_allclose(got[2].numpy(), expected["scalars"], rtol=0.0, atol=1e-6)


def test_device_b8_observation_appends_v8_features_without_changing_base() -> None:
    sim = DeviceSim.from_reference(_fixture(), device="cpu")
    rows = torch.tensor([[0, 0]], dtype=torch.long)
    local, glob, base = build_observation(sim, rows)
    v8 = torch.arange(30, dtype=torch.float32).reshape(1, 30)
    appended = build_observation(sim, rows, v8)
    assert base.shape == (1, 30)
    assert appended[2].shape == (1, 60)
    assert torch.equal(appended[0], local) and torch.equal(appended[1], glob)
    assert torch.equal(appended[2][:, :30], base)
    assert torch.equal(appended[2][:, 30:], v8)
