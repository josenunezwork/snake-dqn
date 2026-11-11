"""Exact parity tests for GameV3Sim vs ReferenceWorld and build_obs vs NumPy reference."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from research.game_v3.config import GameV3Config
from research.game_v3.obs import (
    CH_ENEMY_DX_T0,
    CH_ENEMY_DY_T0,
    CH_ENEMY_HEADS_T0,
    CH_ENEMY_HEADS_T1,
    CH_OUT_OF_ARENA,
    CH_OWN_BODY,
    build_obs,
    build_obs_reference,
)
from research.game_v3.reference import ReferenceWorld
from research.game_v3.rng import PURPOSE_SCRIPTED, rng_int_torch
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import GameV3Sim


def _assert_world_parity_every_tick(sim: GameV3Sim, refs: list[ReferenceWorld], tick: int) -> None:
    """Assert exact discrete-state equality (including all 2D grids) and CPU bitwise floats."""
    occ_sim = sim.active_occupancy_grid().numpy()
    amb_sim = sim.ambient_grid.numpy()
    cor_sim = sim.corpse_grid.numpy()
    alive_sim = sim.alive.numpy()
    len_sim = sim.length.numpy()
    amb_cnt_sim = sim.ambient_count.numpy()
    pos_sim_u32 = sim.head_pos.numpy().view(np.uint32)
    th_sim_u32 = sim.theta.numpy().view(np.uint32)

    for e, ref in enumerate(refs):
        if not np.array_equal(alive_sim[e], ref.alive):
            raise AssertionError(f"tick={tick} env={e}: alive mismatch")
        if not np.array_equal(len_sim[e], ref.length):
            raise AssertionError(f"tick={tick} env={e}: length mismatch")
        if int(amb_cnt_sim[e]) != ref.ambient_count:
            raise AssertionError(
                f"tick={tick} env={e}: ambient_count {amb_cnt_sim[e]} != {ref.ambient_count}"
            )
        if not np.array_equal(pos_sim_u32[e], ref.head_pos.view(np.uint32)):
            raise AssertionError(f"tick={tick} env={e}: bitwise head_pos mismatch")
        if not np.array_equal(th_sim_u32[e], ref.theta.view(np.uint32)):
            raise AssertionError(f"tick={tick} env={e}: bitwise theta mismatch")
        if not np.array_equal(occ_sim[e], ref.occ_grid):
            raise AssertionError(f"tick={tick} env={e}: occupancy grid mismatch")
        if not np.array_equal(amb_sim[e], ref.ambient_grid):
            raise AssertionError(f"tick={tick} env={e}: ambient_grid mismatch")
        if not np.array_equal(cor_sim[e], ref.corpse_grid):
            raise AssertionError(f"tick={tick} env={e}: corpse_grid mismatch")


def _assert_obs_parity(sim: GameV3Sim, refs: list[ReferenceWorld]) -> None:
    """Assert exact observation equality between PyTorch build_obs and NumPy reference."""
    obs_th = build_obs(sim)
    s1_th = obs_th.raster_s1.numpy()
    s4_th = obs_th.raster_s4.numpy()
    s16_th = obs_th.raster_s16.numpy()
    sc_th = obs_th.scalars.numpy()
    am_th = obs_th.action_mask.numpy()

    for e, ref in enumerate(refs):
        obs_ref = build_obs_reference(ref)
        np.testing.assert_allclose(s1_th[e], obs_ref.raster_s1, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(s4_th[e], obs_ref.raster_s4, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(s16_th[e], obs_ref.raster_s16, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(sc_th[e], obs_ref.scalars, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(am_th[e], obs_ref.action_mask)


def test_obs_builder_vs_numpy_reference_and_ego_geometry() -> None:
    """GPU/Torch obs builder matches NumPy reference across scales, channels, and 2-frame heads."""
    cfg = GameV3Config(arena_radius=300.0, num_snakes=12, respawn_delay=8)
    seeds = [3, 17, 29, 43]
    sim = GameV3Sim(cfg, num_envs=4, seeds=seeds, device="cpu")
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]

    # Tick 0: T1 previous-frame head channels must be all zero.
    obs0 = build_obs(sim)
    assert float(obs0.raster_s1[:, :, CH_ENEMY_HEADS_T1].abs().max().item()) == 0.0
    _assert_obs_parity(sim, refs)

    for t in range(1, 61):
        actions = food_seeking_actions(sim) if t % 2 == 1 else random_safe_actions(sim)
        act_np = actions.numpy()
        sim.step(actions)
        for e, ref in enumerate(refs):
            ref.step(act_np[e])
        if t in (1, 2, 20, 40, 60):
            _assert_obs_parity(sim, refs)

    # Verify explicit ego-frame geometry: observer at origin heading +y (theta = pi/2),
    # enemy 10 units ahead at (0, +10) also heading +y -> appears in upper half (row < 32),
    # centered horizontally (col == 32), with ego heading (dx=0, dy=+1).
    g_cfg = GameV3Config(arena_radius=100.0, num_snakes=6)
    g_sim = GameV3Sim(g_cfg, num_envs=1, seeds=[5], device="cpu")
    hg = g_sim.half_grid
    g_sim.alive.fill_(False)
    g_sim.tail_seq.fill_(10)
    g_sim.occ_packed.fill_(-1)

    # Observer slot 0 at cell (hg, hg) heading +y (pi/2).
    g_sim.alive[0, 0] = True
    g_sim.length[0, 0] = 20
    g_sim.head_pos[0, 0] = torch.tensor([0.5, 0.5], dtype=torch.float32)
    g_sim.theta[0, 0] = float(np.float32(0.5 * math.pi))
    g_sim.tail_seq[0, 0] = 1
    g_sim._stamp_discs(
        torch.tensor([[hg] * 6], dtype=torch.int32),
        torch.tensor([[hg] * 6], dtype=torch.int32),
        torch.ones((1, 6), dtype=torch.int32),
        g_sim.alive,
        seq=1,
    )

    # Enemy slot 1 at (0.5, 10.5) heading +y (pi/2) -> 10 units straight ahead.
    g_sim.alive[0, 1] = True
    g_sim.length[0, 1] = 20
    g_sim.head_pos[0, 1] = torch.tensor([0.5, 10.5], dtype=torch.float32)
    g_sim.theta[0, 1] = float(np.float32(0.5 * math.pi))

    obs_g = build_obs(g_sim)
    r1 = obs_g.raster_s1[0, 0]
    assert float(r1[CH_OWN_BODY, 31:33, 31:33].max().item()) == 1.0
    head_locs = (r1[CH_ENEMY_HEADS_T0] > 0.5).nonzero(as_tuple=False)
    assert head_locs.shape == (1, 2)
    row_h, col_h = int(head_locs[0, 0].item()), int(head_locs[0, 1].item())
    assert row_h == 22 and col_h == 32  # floor(32 - 10) == 22 (ahead/up), floor(0 + 32) == 32
    assert pytest.approx(float(r1[CH_ENEMY_DX_T0, row_h, col_h].item()), abs=1e-5) == 0.0
    assert pytest.approx(float(r1[CH_ENEMY_DY_T0, row_h, col_h].item()), abs=1e-5) == 1.0
    # At scale 16 (covering 1024x1024 around radius-100 arena), corners must be out-of-arena.
    assert float(obs_g.raster_s16[0, 0, CH_OUT_OF_ARENA, 0, 0].item()) == 1.0
    assert float(obs_g.raster_s16[0, 0, CH_OUT_OF_ARENA, 32, 32].item()) == 0.0
    # Dead snake slot 2 emits exact zeros.
    assert float(obs_g.raster_s1[0, 2].abs().max().item()) == 0.0
    assert float(obs_g.scalars[0, 2].abs().max().item()) == 0.0


def test_sim_vs_reference_fast_smoke_8x120() -> None:
    """Fast non-slow parity check across 8 worlds x 120 ticks checking every tick."""
    cfg = GameV3Config(arena_radius=300.0, num_snakes=12, respawn_delay=10)
    seeds = [10 + i * 13 for i in range(8)]
    sim = GameV3Sim(cfg, num_envs=8, seeds=seeds, device="cpu")
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]

    _assert_world_parity_every_tick(sim, refs, tick=0)
    for t in range(1, 121):
        actions = food_seeking_actions(sim) if t % 3 != 0 else random_safe_actions(sim)
        act_np = actions.numpy()
        info_s = sim.step(actions)
        for e, ref in enumerate(refs):
            info_r = ref.step(act_np[e])
            assert np.array_equal(info_s.died[e].numpy(), info_r.died)
            assert np.array_equal(info_s.killer_id[e].numpy(), info_r.killer_id)
            assert np.array_equal(info_s.kill_credit[e].numpy(), info_r.kill_credit)
            assert np.array_equal(info_s.mass_eaten[e].numpy(), info_r.mass_eaten)
        _assert_world_parity_every_tick(sim, refs, tick=t)
    _assert_obs_parity(sim, refs)


def _run_32x1000_parity(*, scripted: bool) -> None:
    cfg = GameV3Config()  # Default R=600, N=24
    num_envs = 32
    num_ticks = 1000
    base = 5000 if scripted else 1000
    seeds = [base + e * 37 for e in range(num_envs)]

    sim = GameV3Sim(cfg, num_envs=num_envs, seeds=seeds, device="cpu")
    refs = [ReferenceWorld(cfg, seed=s) for s in seeds]

    _assert_world_parity_every_tick(sim, refs, tick=0)

    slots = torch.arange(cfg.num_snakes, dtype=torch.int64)[None, :]
    half_n = cfg.num_snakes // 2

    for t in range(1, num_ticks + 1):
        if scripted:
            a_food = food_seeking_actions(sim)
            a_rand = random_safe_actions(sim)
            actions = torch.cat((a_food[:, :half_n], a_rand[:, half_n:]), dim=1)
        else:
            actions = rng_int_torch(sim.seeds[:, None], t, PURPOSE_SCRIPTED, slots, 6).to(
                torch.int64
            )
        actions_np = actions.numpy()

        info_sim = sim.step(actions)
        died_sim = info_sim.died.numpy()
        killer_sim = info_sim.killer_id.numpy()
        credit_sim = info_sim.kill_credit.numpy()
        eaten_sim = info_sim.mass_eaten.numpy()

        for e, ref in enumerate(refs):
            info_ref = ref.step(actions_np[e])
            if not np.array_equal(died_sim[e], info_ref.died):
                raise AssertionError(f"tick={t} env={e}: died mismatch")
            if not np.array_equal(killer_sim[e], info_ref.killer_id):
                raise AssertionError(f"tick={t} env={e}: killer_id mismatch")
            if not np.array_equal(credit_sim[e], info_ref.kill_credit):
                raise AssertionError(f"tick={t} env={e}: kill_credit mismatch")
            if not np.array_equal(eaten_sim[e], info_ref.mass_eaten):
                raise AssertionError(f"tick={t} env={e}: mass_eaten mismatch")

        _assert_world_parity_every_tick(sim, refs, tick=t)

    _assert_obs_parity(sim, refs)


@pytest.mark.slow
def test_sim_vs_reference_random_tape_32x1000_n24() -> None:
    """Exact discrete + bitwise CPU parity every tick over 32 worlds x 1,000 ticks (random)."""
    _run_32x1000_parity(scripted=False)


@pytest.mark.slow
def test_sim_vs_reference_scripted_tape_32x1000_n24() -> None:
    """Exact discrete + bitwise CPU parity every tick over 32 worlds x 1,000 ticks (scripted)."""
    _run_32x1000_parity(scripted=True)
