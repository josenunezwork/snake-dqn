"""Gameplay invariant, determinism, checkpoint, and zero-host-sync tests for Game V3."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from research.game_v3.config import GameV3Config
from research.game_v3.obs import build_obs
from research.game_v3.rng import PURPOSE_SCRIPTED, rng_int_torch
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import GameV3Sim


def _place_custom_snake(
    sim: GameV3Sim,
    env: int,
    slot: int,
    gx: int,
    gy: int,
    theta: float,
    length: int,
    *,
    alive: bool = True,
) -> None:
    """Place a single snake cleanly at cell (gx, gy) and rebuild occ_packed for test setup."""
    sim.alive[env, slot] = alive
    if not alive:
        sim.length[env, slot] = 0
        sim.seg_count[env, slot] = 0
        sim.tail_seq[env, slot] = sim.tick + 2
        sim.respawn_timer[env, slot] = 9999
        return
    c = float(np.float32(sim.cfg.cell_size))
    hg = float(np.float32(sim.half_grid))
    px = (float(gx) - hg + 0.5) * c
    py = (float(gy) - hg + 0.5) * c
    sim.head_pos[env, slot, 0] = px
    sim.head_pos[env, slot, 1] = py
    sim.theta[env, slot] = float(np.float32(theta))
    sim.length[env, slot] = length
    sim.seg_count[env, slot] = 1
    sim.head_ptr[env, slot] = 0
    sim.tail_seq[env, slot] = sim.tick + 1
    sim.boost_ticks[env, slot] = 0
    sim.ring_pos[env, slot, 0, 0] = px
    sim.ring_pos[env, slot, 0, 1] = py
    sim.ring_cell[env, slot, 0, 0] = gx
    sim.ring_cell[env, slot, 0, 1] = gy


def _restamp_test_env(sim: GameV3Sim) -> None:
    """Clear and restamp occ_packed from current 1-segment custom test positions."""
    sim.occ_packed.fill_(-1)
    gx = sim.ring_cell[:, :, 0, 0].to(torch.int32)
    gy = sim.ring_cell[:, :, 0, 1].to(torch.int32)
    r_sq = sim._r_sq_table[sim.length.clamp(0, sim.cfg.max_length).to(torch.int64)]
    sim._stamp_discs(gx, gy, r_sq, sim.alive, seq=sim.tick + 1)


def test_never_dies_from_own_body() -> None:
    """A long snake circling tightly across its own body for hundreds of ticks never dies."""
    cfg = GameV3Config(arena_radius=200.0, num_snakes=6, start_length=150)
    sim = GameV3Sim(cfg, num_envs=1, seeds=[7])
    hg = sim.half_grid
    _place_custom_snake(sim, 0, 0, hg, hg, 0.0, length=200, alive=True)
    for s in range(1, 6):
        _place_custom_snake(sim, 0, s, hg, hg, 0.0, length=0, alive=False)
    _restamp_test_env(sim)

    # Constant max left turn forms a small ~15-unit radius circle, crossing its 200-unit
    # body repeatedly every ~95 ticks.
    actions = torch.zeros((1, 6), dtype=torch.int64)
    crossed_own = False
    for _ in range(350):
        # Check if next step's head lands on an already-occupied cell of slot 0.
        occ = sim.active_occupancy_grid()[0]
        info = sim.step(actions)
        assert bool(info.alive[0, 0].item())
        assert not bool(info.died[0, 0].item())
        gx = int(sim.ring_cell[0, 0, sim.head_ptr[0, 0], 0].item())
        gy = int(sim.ring_cell[0, 0, sim.head_ptr[0, 0], 1].item())
        if int(occ[gy, gx].item()) == 0:
            crossed_own = True

    assert crossed_own, "Snake must actually cross its own occupied body cells during the test"
    assert int(sim.seg_count[0, 0].item()) >= 200


def test_boundary_kills_with_no_killer_credit() -> None:
    """Touching or overlapping the circular arena boundary kills the snake."""
    cfg = GameV3Config(arena_radius=100.0, num_snakes=6)
    sim = GameV3Sim(cfg, num_envs=1, seeds=[11])
    hg = sim.half_grid
    # Place snake 0 heading +x at x = 94.5 (radius 100, w(10) ~ 1.13 -> dies within ~5 ticks).
    _place_custom_snake(sim, 0, 0, hg + 94, hg, 0.0, length=10, alive=True)
    for s in range(1, 6):
        _place_custom_snake(sim, 0, s, hg, hg, 0.0, length=0, alive=False)
    _restamp_test_env(sim)

    actions = torch.ones((1, 6), dtype=torch.int64)  # straight normal speed
    died_tick = None
    for t in range(1, 12):
        info = sim.step(actions)
        if bool(info.died[0, 0].item()):
            died_tick = t
            assert int(info.killer_id[0, 0].item()) == -1
            assert int(info.kill_credit.sum().item()) == 0
            assert not bool(info.alive[0, 0].item())
            break
    assert died_tick is not None and 3 <= died_tick <= 6


def test_head_on_collision_strict_winner_and_tie() -> None:
    """Head-on kills the smaller snake if >15% larger, or kills both within 15%."""
    cfg = GameV3Config(arena_radius=200.0, num_snakes=6, head_on_tie_margin=0.15)
    sim = GameV3Sim(cfg, num_envs=2, seeds=[1, 2])
    hg = sim.half_grid

    for env in range(2):
        for s in range(6):
            _place_custom_snake(sim, env, s, hg, hg, 0.0, length=0, alive=False)

    # Env 0: Snake 0 (L=25) vs Snake 1 (L=20) -> 25 > 1.15 * 20 = 23 -> Snake 0 wins!
    _place_custom_snake(sim, 0, 0, hg - 2, hg, 0.0, length=25, alive=True)
    _place_custom_snake(sim, 0, 1, hg + 2, hg, math.pi, length=20, alive=True)

    # Env 1: Snake 0 (L=23) vs Snake 1 (L=20) -> 23 <= 1.15 * 20 = 23 -> Mutual death!
    _place_custom_snake(sim, 1, 0, hg - 2, hg, 0.0, length=23, alive=True)
    _place_custom_snake(sim, 1, 1, hg + 2, hg, math.pi, length=20, alive=True)
    _restamp_test_env(sim)
    sim.ambient_grid.zero_()
    sim.ambient_grid_4x.zero_()
    sim.ambient_grid_16x.zero_()
    sim.ambient_count.fill_(cfg.target_ambient_pellets)

    actions = torch.ones((2, 6), dtype=torch.int64)
    collided_0 = False
    collided_1 = False
    for _ in range(5):
        info = sim.step(actions)
        if bool(info.died[0, 1].item()):
            collided_0 = True
            assert bool(info.alive[0, 0].item())
            assert not bool(info.died[0, 0].item())
            assert int(info.killer_id[0, 1].item()) == 0
            assert int(info.kill_credit[0, 0].item()) == 1
        if bool(info.died[1, 0].item()) or bool(info.died[1, 1].item()):
            collided_1 = True
            assert bool(info.died[1, 0].item()) and bool(info.died[1, 1].item())
            assert int(info.killer_id[1, 0].item()) == -1
            assert int(info.killer_id[1, 1].item()) == -1
            assert int(info.kill_credit[1].sum().item()) == 0
        if collided_0 and collided_1:
            break
    assert collided_0 and collided_1


def test_enemy_body_collision_awards_kill_and_drops_corpse() -> None:
    """Running into an enemy snake's body kills the attacker, credits victim, drops corpse."""
    cfg = GameV3Config(arena_radius=200.0, num_snakes=6, corpse_mass_fraction=0.5)
    sim = GameV3Sim(cfg, num_envs=1, seeds=[21])
    hg = sim.half_grid
    for s in range(6):
        _place_custom_snake(sim, 0, s, hg, hg, 0.0, length=0, alive=False)

    # Snake 0 moves vertically along x = hg + 5 from y = hg - 10 upward so its body spans y=hg.
    _place_custom_snake(sim, 0, 0, hg + 5, hg - 10, 0.5 * math.pi, length=30, alive=True)
    # Snake 1 starts at x = hg - 12, y = hg moving +x straight into Snake 0's mid-body.
    _place_custom_snake(sim, 0, 1, hg - 12, hg, 0.0, length=20, alive=True)
    _restamp_test_env(sim)
    sim.ambient_grid.zero_()
    sim.ambient_grid_4x.zero_()
    sim.ambient_grid_16x.zero_()
    sim.ambient_count.fill_(cfg.target_ambient_pellets)

    actions = torch.ones((1, 6), dtype=torch.int64)
    hit = False
    for _ in range(25):
        info = sim.step(actions)
        if bool(info.died[0, 1].item()):
            hit = True
            assert bool(info.alive[0, 0].item())
            assert int(info.killer_id[0, 1].item()) == 0
            assert int(info.kill_credit[0, 0].item()) == 1
            # 20 * 0.5 = 10 corpse mass dropped on board.
            assert int(sim.corpse_grid.sum().item()) == 10
            break
    assert hit


def test_boost_sheds_mass_drops_trail_and_respects_min_length() -> None:
    """Boosting moves at boost_speed, sheds mass into trail pellets, stops at min_boost_length."""
    cfg = GameV3Config(
        arena_radius=300.0,
        num_snakes=6,
        min_boost_length=15,
        boost_shed_ticks=4,
        boost_shed_mass=1,
    )
    sim = GameV3Sim(cfg, num_envs=1, seeds=[99])
    hg = sim.half_grid
    for s in range(6):
        _place_custom_snake(sim, 0, s, hg, hg, 0.0, length=0, alive=False)
    _place_custom_snake(sim, 0, 0, hg - 100, hg, 0.0, length=17, alive=True)
    _restamp_test_env(sim)
    sim.ambient_grid.zero_()
    sim.ambient_grid_4x.zero_()
    sim.ambient_grid_16x.zero_()
    sim.corpse_grid.zero_()
    sim.corpse_grid_4x.zero_()
    sim.corpse_grid_16x.zero_()
    sim.ambient_count.fill_(cfg.target_ambient_pellets)

    # Action 4 = straight + boost.
    actions = torch.full((1, 6), 4, dtype=torch.int64)
    x0 = float(sim.head_pos[0, 0, 0].item())

    # Ticks 1..8: length starts at 17 (> 15), sheds 1 at tick 4 (->16) and 1 at tick 8 (->15).
    for _ in range(8):
        sim.step(actions)
    x8 = float(sim.head_pos[0, 0, 0].item())
    assert pytest.approx(x8 - x0, abs=1e-4) == 8 * cfg.boost_speed
    assert int(sim.length[0, 0].item()) == 15
    assert int(sim.corpse_grid.sum().item()) == 2

    # Tick 9..12: now length == 15 == min_boost_length, so boost is disallowed (moves at speed=1.0).
    for _ in range(4):
        sim.step(actions)
    x12 = float(sim.head_pos[0, 0, 0].item())
    assert pytest.approx(x12 - x8, abs=1e-4) == 4 * cfg.speed
    assert int(sim.length[0, 0].item()) == 15
    assert int(sim.corpse_grid.sum().item()) == 2


def test_respawn_after_delay_and_exact_mass_conservation() -> None:
    """Dead snakes respawn after respawn_delay and total mass is conserved up to stated sinks."""
    cfg = GameV3Config(arena_radius=250.0, num_snakes=12, respawn_delay=8, start_length=18)
    sim = GameV3Sim(cfg, num_envs=4, seeds=[101, 202, 303, 404])

    saw_death = False
    saw_respawn = False
    prev_dead = torch.zeros_like(sim.alive)

    for t in range(300):
        acts = rng_int_torch(
            sim.seeds[:, None],
            t + 1,
            PURPOSE_SCRIPTED,
            torch.arange(cfg.num_snakes)[None, :],
            6,
        ).to(torch.int64)
        info = sim.step(acts)
        if bool(info.died.any().item()):
            saw_death = True
        if bool((prev_dead & info.alive).any().item()):
            saw_respawn = True
        prev_dead = ~info.alive

        # Exact per-world mass conservation check!
        live_mass = sim.length.to(torch.int64).sum(dim=1)
        amb_mass = sim.ambient_grid.to(torch.int64).sum(dim=(1, 2))
        cor_mass = sim.corpse_grid.to(torch.int64).sum(dim=(1, 2))
        expected_mass = (
            sim.total_spawned_snake_mass
            + sim.total_spawned_ambient_mass
            - sim.total_corpse_decay_sink
            - sim.total_length_cap_sink
        )
        torch.testing.assert_close(live_mass + amb_mass + cor_mass, expected_mass)

        # Coarse food pyramid consistency check!
        torch.testing.assert_close(sim.ambient_grid_4x.to(torch.int64).sum(dim=(1, 2)), amb_mass)
        torch.testing.assert_close(sim.ambient_grid_16x.to(torch.int64).sum(dim=(1, 2)), amb_mass)
        torch.testing.assert_close(sim.corpse_grid_4x.to(torch.int64).sum(dim=(1, 2)), cor_mass)
        torch.testing.assert_close(sim.corpse_grid_16x.to(torch.int64).sum(dim=(1, 2)), cor_mass)

    assert saw_death, "Expected at least one snake death over 300 ticks"
    assert saw_respawn, "Expected at least one snake respawn over 300 ticks"


def test_determinism_repeat_and_midrun_save_restore() -> None:
    """Same seed repeats bit-identically and mid-run save/restore matches uninterrupted run."""
    cfg = GameV3Config(arena_radius=300.0, num_snakes=16, respawn_delay=10)
    seeds = [11, 22, 33, 44]
    tape = [
        rng_int_torch(
            torch.tensor(seeds, dtype=torch.int64)[:, None],
            t + 1,
            PURPOSE_SCRIPTED,
            torch.arange(cfg.num_snakes, dtype=torch.int64)[None, :],
            6,
        ).to(torch.int64)
        for t in range(200)
    ]

    sim_a = GameV3Sim(cfg, num_envs=4, seeds=seeds)
    sim_b = GameV3Sim(cfg, num_envs=4, seeds=seeds)
    ckpt = None

    for t in range(200):
        info_a = sim_a.step(tape[t])
        info_b = sim_b.step(tape[t])
        torch.testing.assert_close(info_a.alive, info_b.alive)
        torch.testing.assert_close(info_a.length, info_b.length)
        if t == 99:
            ckpt = sim_a.state_dict()

    assert ckpt is not None
    sim_resumed = GameV3Sim(cfg, num_envs=4, seeds=[0, 0, 0, 0])
    sim_resumed.load_state_dict(ckpt)
    for t in range(100, 200):
        sim_resumed.step(tape[t])

    obs_a = build_obs(sim_a)
    obs_resumed = build_obs(sim_resumed)
    torch.testing.assert_close(obs_a.raster_s1, obs_resumed.raster_s1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(obs_a.raster_s4, obs_resumed.raster_s4, rtol=0.0, atol=0.0)
    torch.testing.assert_close(obs_a.raster_s16, obs_resumed.raster_s16, rtol=0.0, atol=0.0)
    torch.testing.assert_close(obs_a.scalars, obs_resumed.scalars, rtol=0.0, atol=0.0)
    torch.testing.assert_close(obs_a.action_mask, obs_resumed.action_mask)

    for sim_other in (sim_b, sim_resumed):
        assert np.array_equal(
            sim_a.head_pos.numpy().view(np.uint32),
            sim_other.head_pos.numpy().view(np.uint32),
        )
        assert np.array_equal(
            sim_a.theta.numpy().view(np.uint32),
            sim_other.theta.numpy().view(np.uint32),
        )
        torch.testing.assert_close(sim_a.alive, sim_other.alive)
        torch.testing.assert_close(sim_a.length, sim_other.length)
        torch.testing.assert_close(sim_a.occ_packed, sim_other.occ_packed)
        torch.testing.assert_close(sim_a.ambient_grid, sim_other.ambient_grid)
        torch.testing.assert_close(sim_a.corpse_grid, sim_other.corpse_grid)
        torch.testing.assert_close(sim_a.prev_head_pos, sim_other.prev_head_pos)
        torch.testing.assert_close(sim_a.prev_alive, sim_other.prev_alive)


def test_step_obs_and_scripted_zero_host_syncs(monkeypatch: pytest.MonkeyPatch) -> None:
    """step, build_obs, and scripted policies never call .item/.nonzero/.tolist/.numpy."""
    cfg = GameV3Config(arena_radius=200.0, num_snakes=12)
    sim = GameV3Sim(cfg, num_envs=4, seeds=[1, 2, 3, 4])
    _ = build_obs(sim)  # Prime one-time static WeakKeyDictionary cache before monkeypatching.

    def _forbid(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Forbidden host synchronization inside device loop")

    monkeypatch.setattr(torch.Tensor, "item", _forbid)
    monkeypatch.setattr(torch.Tensor, "nonzero", _forbid)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbid)
    monkeypatch.setattr(torch.Tensor, "numpy", _forbid)

    for t in range(10):
        actions = food_seeking_actions(sim) if t % 2 == 0 else random_safe_actions(sim)
        sim.step(actions)
        _ = build_obs(sim)
