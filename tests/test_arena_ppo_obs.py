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


def test_minor3_arena_wall_channel_border_and_ego_wall_exclusivity() -> None:
    """Regression MINOR 3: arena CH_WALL marks 1-cell perimeter; ego CH_WALL is OOB-only."""
    train_sim = TrainSim.create(4, device="cpu", seed=17, max_capacity=128)
    train_sim.prepare()
    # Kill one seat to verify dead seats remain all-zero on arena CH_WALL too.
    train_sim.sim.alive[0, 2] = False
    # Place seat (0, 0) right at (x=1, y=1) so its 47x47 ego crop sees both perimeter & OOB cells.
    train_sim.sim.bodies[0, 0, train_sim.sim.head_ptr[0, 0]] = torch.tensor(
        [1, 1], dtype=torch.long
    )

    obs = build_arena_ego_obs_gpu(train_sim)
    alive_flat = train_sim.sim.alive.reshape(-1)
    for row_idx in range(obs.arena.shape[0]):
        wall = obs.arena[row_idx, CH_WALL]
        if bool(alive_flat[row_idx].item()):
            assert (wall[0, :] == 1.0).all()
            assert (wall[-1, :] == 1.0).all()
            assert (wall[:, 0] == 1.0).all()
            assert (wall[:, -1] == 1.0).all()
            assert (wall[1:-1, 1:-1] == 0.0).all()
        else:
            assert (wall == 0.0).all()
            assert (obs.ego[row_idx] == 0.0).all()

    arena_cpu, ego_cpu, scalars_cpu = build_arena_ego_obs_cpu(
        train_sim.sim,
        train_sim.prev_heads.numpy(),
        train_sim.prev_alive.numpy(),
    )
    assert np.array_equal(obs.arena.numpy(), arena_cpu)
    assert np.array_equal(obs.ego.numpy(), ego_cpu)
    assert np.array_equal(obs.scalars.numpy(), scalars_cpu)
    # Seat (0, 0) at (1, 1) facing right (dir=1): head at ego (23, 23); left of head (col=22) is
    # y=0 (in-bounds perimeter -> ego CH_WALL must be 0.0!), col=21 is y=-1 (OOB -> CH_WALL == 1.0).
    assert float(obs.ego[0, CH_WALL, 23, 22].item()) == 0.0
    assert float(obs.ego[0, CH_WALL, 23, 21].item()) == 1.0


def test_m1_obs_avoids_large_float32_zeros_like_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression M1: obs builder masks in-place without multi-tensor float32 zeros_like copies."""
    orig_zeros_like = torch.zeros_like
    float_zeros_like_elems: list[int] = []

    def _tracked_zeros_like(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        out = orig_zeros_like(tensor, *args, **kwargs)
        if out.dtype == torch.float32:
            float_zeros_like_elems.append(int(out.numel()))
        return out

    monkeypatch.setattr(torch, "zeros_like", _tracked_zeros_like)
    train_sim = TrainSim.create(8, device="cpu", seed=33, max_capacity=128)
    train_sim.prepare()
    _ = build_arena_ego_obs_gpu(train_sim)
    assert float_zeros_like_elems == []


def test_parameterized_obs_gpu_equals_cpu_custom_board_and_snakes() -> None:
    """Coordinator #7: verify exact GPU==CPU obs at custom (48x64, S=4) and S=1."""
    train_sim = TrainSim.create(
        8, device="cpu", seed=9001, grid_h=48, grid_w=64, num_snakes=4, max_capacity=192
    )
    gen = torch.Generator(device="cpu").manual_seed(77)
    for _ in range(15):
        train_sim.prepare()
        gpu_obs = build_arena_ego_obs_gpu(train_sim)
        cpu_arena, cpu_ego, cpu_scalars = build_arena_ego_obs_cpu(
            train_sim.sim,
            train_sim.prev_heads.numpy(),
            train_sim.prev_alive.numpy(),
        )
        assert gpu_obs.arena.shape == (32, NUM_OBS_CHANNELS, 48, 64)
        assert gpu_obs.ego.shape == (32, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE)
        assert np.array_equal(gpu_obs.arena.numpy(), cpu_arena)
        assert np.array_equal(gpu_obs.ego.numpy(), cpu_ego)
        assert np.array_equal(gpu_obs.scalars.numpy(), cpu_scalars)
        train_sim.step(torch.randint(0, 6, (8, 4), generator=gen, dtype=torch.long))

    solo_sim = TrainSim.create(
        2, device="cpu", seed=12, grid_h=32, grid_w=40, num_snakes=1, max_capacity=64
    )
    solo_sim.prepare()
    solo_obs = build_arena_ego_obs_gpu(solo_sim)
    solo_a, solo_e, solo_s = build_arena_ego_obs_cpu(
        solo_sim.sim, solo_sim.prev_heads.numpy(), solo_sim.prev_alive.numpy()
    )
    assert np.array_equal(solo_obs.arena.numpy(), solo_a)
    assert np.array_equal(solo_obs.ego.numpy(), solo_e)
    assert np.array_equal(solo_obs.scalars.numpy(), solo_s)


def test_minor9_r3_respawn_after_fatal_boost_clears_stale_boost_flag_gpu_and_cpu() -> None:
    """MINOR 9: respawned snake (`alive=True, prev_alive=False`) must NOT show stale boost flag."""
    train_sim = TrainSim.create(2, device="cpu", seed=2026, max_capacity=128)
    train_sim.prepare()
    # Seat (0, 0) boosts straight into the right border wall and dies; seat (0, 1) boosts safely
    hp00 = int(train_sim.sim.head_ptr[0, 0].item())
    train_sim.sim.bodies[0, 0, hp00] = torch.tensor([train_sim.grid_w - 1, 20], dtype=torch.long)
    train_sim.sim.direction[0, 0] = 1
    train_sim.sim.length[0, 0] = 10

    hp01 = int(train_sim.sim.head_ptr[0, 1].item())
    train_sim.sim.bodies[0, 1, hp01] = torch.tensor([40, 40], dtype=torch.long)
    train_sim.sim.direction[0, 1] = 1
    train_sim.sim.length[0, 1] = 10

    actions = torch.ones((2, 6), dtype=torch.long)
    actions[0, 0] = 4  # boost straight -> fatal wall collision
    actions[0, 1] = 4  # boost straight -> survives
    info = train_sim.step(actions)
    assert bool(info.died[0, 0].item()) is True
    assert bool(info.survived[0, 1].item()) is True

    # Force immediate respawn on next prepare() while _boosted_this_step[0, 0] is still True
    train_sim.sim.respawn_timer[0, 0] = 0
    train_sim.prepare()
    assert bool(train_sim.sim.alive[0, 0].item()) is True
    assert bool(train_sim.prev_alive[0, 0].item()) is False
    assert bool(train_sim.sim._boosted_this_step[0, 0].item()) is True
    assert bool(train_sim.prev_alive[0, 1].item()) is True
    assert bool(train_sim.sim._boosted_this_step[0, 1].item()) is True

    gpu_obs = build_arena_ego_obs_gpu(train_sim)
    cpu_arena, cpu_ego, cpu_scalars = build_arena_ego_obs_cpu(
        train_sim.sim,
        train_sim.prev_heads.numpy(),
        train_sim.prev_alive.numpy(),
    )
    assert np.array_equal(gpu_obs.arena.numpy(), cpu_arena)
    assert np.array_equal(gpu_obs.ego.numpy(), cpu_ego)
    assert np.array_equal(gpu_obs.scalars.numpy(), cpu_scalars)

    respawn_head = train_sim.sim.bodies[0, 0, int(train_sim.sim.head_ptr[0, 0].item())]
    rx, ry = int(respawn_head[0].item()), int(respawn_head[1].item())
    survive_head = train_sim.sim.bodies[0, 1, int(train_sim.sim.head_ptr[0, 1].item())]
    sx, sy = int(survive_head[0].item()), int(survive_head[1].item())

    # Respawned seat (0, 0) must NOT appear on CH_BOOST_FLAGS for self (row 0) or enemy (row 1),
    # whereas surviving boosting seat (0, 1) MUST appear as 1.0!
    assert float(gpu_obs.arena[0, CH_BOOST_FLAGS, ry, rx].item()) == 0.0
    assert float(gpu_obs.arena[1, CH_BOOST_FLAGS, ry, rx].item()) == 0.0
    assert float(cpu_arena[0, CH_BOOST_FLAGS, ry, rx]) == 0.0
    assert float(cpu_arena[1, CH_BOOST_FLAGS, ry, rx]) == 0.0
    assert float(gpu_obs.ego[0, CH_BOOST_FLAGS, 23, 23].item()) == 0.0
    assert float(cpu_ego[0, CH_BOOST_FLAGS, 23, 23]) == 0.0

    assert float(gpu_obs.arena[0, CH_BOOST_FLAGS, sy, sx].item()) == 1.0
    assert float(cpu_arena[0, CH_BOOST_FLAGS, sy, sx]) == 1.0
