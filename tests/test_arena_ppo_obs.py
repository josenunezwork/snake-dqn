"""Tests for `research.arena_ppo.obs` proving GPU/MPS obs == CPU obs bit-for-bit."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.arena_ppo import ARENA_GRID_H, ARENA_GRID_W, EGO_CROP_SIZE, SCALAR_DIM
from research.arena_ppo.obs import (
    CH_BOOST_FLAGS,
    CH_ENEMY_HEAD_CURR,
    CH_ENEMY_HEAD_PREV,
    CH_FOOD_CORPSE,
    CH_OWN_HEAD_CURR,
    CH_OWN_HEAD_PREV,
    CH_WALL,
    NUM_OBS_CHANNELS,
    build_arena_ego_obs_cpu,
    build_arena_ego_obs_from_tensors,
    build_arena_ego_obs_gpu,
)
from research.arena_ppo.sim import TrainSim, arena_v2_config
from research.devicesim_v2_parity.harness import install_splitmix_rng_proxies
from src.simd_env.grid_sim import GridBatchSim


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_obs_gpu_equals_cpu_exact_64_worlds_x_200_ticks(device_name: str) -> None:
    """Prove GPU/MPS obs == CPU reference obs bit-for-bit on 64 worlds x 200 ticks."""
    num_envs = 64
    ticks = 200
    train_sim = TrainSim.create(num_envs, device=device_name, seed=20261110, max_capacity=384)
    # Seed mature lengths so boost, trail pellets, and corpses occur throughout the 200 ticks.
    init_lengths = torch.tensor([1, 5, 8, 12, 18, 25], dtype=torch.long, device=train_sim.device)
    train_sim.sim.length.copy_(init_lengths[None, :].expand(num_envs, -1))
    train_sim.sim._reward_prev_length.copy_(train_sim.sim.length)

    gen = torch.Generator(device="cpu").manual_seed(11223344)
    saw_corpse = False
    saw_boost = False
    saw_prev_head = False
    saw_wall_crop = False

    for tick in range(ticks):
        train_sim.prepare()
        gpu_obs = build_arena_ego_obs_gpu(train_sim, ring_chunk=128)
        arena_gpu = gpu_obs.arena.detach().cpu().numpy()
        ego_gpu = gpu_obs.ego.detach().cpu().numpy()
        scalars_gpu = gpu_obs.scalars.detach().cpu().numpy()

        arena_cpu, ego_cpu, scalars_cpu = build_arena_ego_obs_cpu(
            train_sim.sim,
            train_sim.prev_heads.detach().cpu().numpy(),
            train_sim.prev_alive.detach().cpu().numpy(),
        )

        assert arena_gpu.shape == (num_envs * 6, NUM_OBS_CHANNELS, ARENA_GRID_H, ARENA_GRID_W)
        assert ego_gpu.shape == (num_envs * 6, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE)
        assert scalars_gpu.shape == (num_envs * 6, SCALAR_DIM)

        assert np.array_equal(
            arena_gpu, arena_cpu
        ), f"Arena mismatch at tick={tick} ({device_name})"
        assert np.array_equal(ego_gpu, ego_cpu), f"Ego mismatch at tick={tick} ({device_name})"
        assert np.array_equal(
            scalars_gpu, scalars_cpu
        ), f"Scalars mismatch at tick={tick} ({device_name})"

        saw_corpse = saw_corpse or bool((arena_gpu[:, CH_FOOD_CORPSE] > 0).any())
        saw_boost = saw_boost or bool((arena_gpu[:, CH_BOOST_FLAGS] > 0).any())
        saw_prev_head = saw_prev_head or bool(
            (arena_gpu[:, CH_OWN_HEAD_PREV] > 0).any()
            and (arena_gpu[:, CH_ENEMY_HEAD_PREV] > 0).any()
        )
        saw_wall_crop = saw_wall_crop or bool((ego_gpu[:, CH_WALL] > 0).any())

        act = torch.randint(0, 6, (num_envs, 6), generator=gen, dtype=torch.long)
        train_sim.step(act.to(train_sim.device))

    assert saw_corpse, "Expected corpse food planes to be populated during 64x200 run"
    assert saw_boost, "Expected boost head flags to be populated during 64x200 run"
    assert saw_prev_head, "Expected 2-frame previous-head planes to be populated"
    assert saw_wall_crop, "Expected ego wall border channel to be active near arena boundaries"


def test_obs_cpu_from_gridbatchsim_matches_gpu_and_ordered_bodies() -> None:
    """Verify CPU builder on live GridBatchSim and ordered_bodies path match ring GPU obs."""
    cfg = arena_v2_config(8, max_capacity=256)
    grid_sim = GridBatchSim(cfg, train_mode=False, allow_respawn=True, seeds=list(range(10, 18)))
    grid_sim.reset()
    grid_sim.length[:] = np.array([[2, 6, 9, 12, 5, 15]] * 8, dtype=np.int64)
    grid_sim._reward_prev_length[:] = grid_sim.length

    train_sim = TrainSim.from_reference(grid_sim, device="cpu")
    install_splitmix_rng_proxies(grid_sim, train_sim.sim)

    prev_heads_np = np.zeros((8, 6, 2), dtype=np.int64)
    prev_alive_np = np.zeros((8, 6), dtype=bool)
    rng = np.random.default_rng(7)

    for tick in range(25):
        act_np = rng.integers(0, 6, size=(8, 6), dtype=np.int64)
        grid_sim.step(act_np)
        train_sim.step(torch.as_tensor(act_np, dtype=torch.long))
        survived_np = grid_sim.alive.copy()
        pushes = np.where(grid_sim._boosted_this_step, 2, 1)
        prev_ring = (grid_sim.head_ptr - pushes) % grid_sim.cap
        e_grid = np.arange(8)[:, None]
        s_grid = np.arange(6)[None, :]
        decision_heads_before = grid_sim.bodies[e_grid, s_grid, prev_ring]
        prev_heads_np = np.where(
            survived_np[..., None], decision_heads_before, np.zeros_like(decision_heads_before)
        )
        prev_alive_np = survived_np

        arena_grid, ego_grid, scalars_grid = build_arena_ego_obs_cpu(
            grid_sim, prev_heads_np, prev_alive_np
        )
        obs_dev = build_arena_ego_obs_gpu(train_sim)
        assert np.array_equal(obs_dev.arena.numpy(), arena_grid), f"GridBatchSim arena diff@{tick}"
        assert np.array_equal(obs_dev.ego.numpy(), ego_grid), f"GridBatchSim ego diff@{tick}"
        assert np.array_equal(obs_dev.scalars.numpy(), scalars_grid)

        # Also test ordered_bodies path against ring_bodies path
        sim = train_sim.sim
        max_k = int(sim.seg_count.max().item()) + 2
        k_vec = torch.arange(max_k, dtype=torch.long)[None, None, :]
        rings = (sim.head_ptr[..., None] - k_vec) % sim.cap
        e_idx = torch.arange(sim.E)[:, None, None]
        s_idx = torch.arange(sim.S)[None, :, None]
        ordered = sim.bodies[e_idx, s_idx, rings]
        obs_ord = build_arena_ego_obs_from_tensors(
            ordered_bodies=ordered,
            seg_count=sim.seg_count,
            length=sim.length,
            alive=sim.alive,
            direction=sim.direction,
            boosted=sim._boosted_this_step,
            prev_heads=train_sim.prev_heads,
            prev_alive=train_sim.prev_alive,
            food_cells=sim.food_cells,
            food_present=sim.food_present,
            corpse_food=sim.corpse_food,
            min_boost_length=train_sim.min_boost_length,
        )
        assert torch.equal(obs_ord.arena, obs_dev.arena)
        assert torch.equal(obs_ord.ego, obs_dev.ego)
        assert torch.equal(obs_ord.scalars, obs_dev.scalars)
        assert (obs_dev.ego[:, CH_OWN_HEAD_CURR, 23, 23] == sim.alive.reshape(-1).float()).all()
        assert (obs_dev.arena[:, CH_ENEMY_HEAD_CURR].sum(dim=(1, 2)) >= 0).all()
