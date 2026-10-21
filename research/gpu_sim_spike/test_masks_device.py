"""Direct oracle for tail-vacating masks, including wrapped long rings."""

from __future__ import annotations

import numpy as np

from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.masks_device import advisory_mask
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_scenarios import inject_serpentines
from src.simd_env.grid_sim import GridBatchSim


def test_short_played_masks_against_reference() -> None:
    ref = GridBatchSim(BatchSimConfig(num_envs=2, num_snakes=6, max_capacity=128), seeds=[41, 42])
    rng = np.random.default_rng(93)
    for _ in range(12):
        sim = DeviceSim.from_reference(ref)
        np.testing.assert_array_equal(advisory_mask(sim).numpy(), ref.get_advisory_action_mask())
        ref.step(rng.integers(0, 6, size=(2, 6)))


def test_long_masks_against_reference_and_rotated_ring() -> None:
    ref = GridBatchSim(BatchSimConfig(num_envs=1, num_snakes=6, max_capacity=1024), seeds=[41])
    inject_serpentines([ref], [900, 600, 400, 300, 150, 150], box_height=8)
    for _ in range(3):
        sim = DeviceSim.from_reference(ref)
        np.testing.assert_array_equal(advisory_mask(sim).numpy(), ref.get_advisory_action_mask())
        ref.step(np.ones((1, 6), dtype=np.int64))
