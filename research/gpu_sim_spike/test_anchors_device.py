"""Resident anchors use the reference clock, ordering and safe-list ranking."""

import numpy as np
import torch

from research.gpu_sim_spike.anchors_device import choose_anchor, random_table
from research.gpu_sim_spike.device_sim import DeviceSim
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
from src.simd_env.grid_sim import GridBatchSim


def test_both_scripted_anchors_match_reference_across_prepared_frames() -> None:
    ref = GridBatchSim(BatchSimConfig(num_envs=1, num_snakes=3), seeds=[41])
    table = random_table(41, 1, torch.device("cpu"))
    for _ in range(12):
        ref.frame += 1
        sim = DeviceSim.from_reference(ref)
        for kind in ("greedy_food", "random_safe"):
            policy = FastProfileAnchorSimdPolicy(kind, [41])
            expected = policy.actions(ref.get_resolved_action_mask()[:, 1], ref, np.array([[0, 1]]))
            actual = choose_anchor(sim, 1, kind, table)
            np.testing.assert_array_equal(actual.numpy(), expected)
        ref.step(np.ones((1, 3), dtype=np.int64))
