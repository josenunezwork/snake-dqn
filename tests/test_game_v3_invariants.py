"""Core gameplay, geometry, occupancy, episode-hook, and zero-sync invariants for Game V3.1."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from research.game_v3.config import FP_ONE, FP_SHIFT, NUM_ACTIONS, GameV3Config
from research.game_v3.obs import (
    CH_ENEMY_BODY,
    CH_ENEMY_BOOST_T0,
    CH_ENEMY_HEADS_T0,
    CH_OUT_OF_ARENA,
    CH_OWN_BODY,
    NUM_RASTER_CHANNELS,
    NUM_SCALAR_FEATURES,
    SCALAR_BOUNDARY_DIST,
    SCALAR_OWN_BOOST,
    allocate_obs_batch,
    build_obs,
)
from research.game_v3.rng import PURPOSE_RESPAWN, rng_int_torch
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import GameV3Sim


def _clear_world(sim: GameV3Sim) -> None:
    """Clear all snakes, occupancy grids, food grids, and mass ledgers for synthetic tests."""
    sim.alive.fill_(False)
    sim.boosting.fill_(False)
    sim.length.zero_()
    sim.seg_count.zero_()
    sim.head_ptr.zero_()
    sim.oldest_active_subtick.fill_(2)
    sim.dist_accum_fp.zero_()
    sim.boost_ticks.zero_()
    sim.boost_drop_fp_accum.zero_()
    sim.respawn_timer.fill_(10_000)
    sim.life_age.zero_()
    sim.life_start.fill_(False)
    sim.truncated_pending.fill_(False)

    sim.ring_pos.zero_()
    sim.ring_cell.zero_()
    sim.ring_r_sq.zero_()
    sim.ring_subtick.zero_()

    sim.prev_alive.fill_(False)
    sim.prev_boosting.fill_(False)
    sim.prev_head_pos.zero_()
    sim.prev_theta.zero_()
    sim.prev_length.zero_()

    sim.occ_hi.fill_(-1)
    sim.occ_lo.fill_(-1)
    sim.occ_subtick_4x.fill_(-1)
    sim.occ_subtick_16x.fill_(-1)

    sim.ambient_grid.zero_()
    sim.ambient_grid_4x.zero_()
    sim.ambient_grid_16x.zero_()
    sim.corpse_grid.zero_()
    sim.corpse_grid_4x.zero_()
    sim.corpse_grid_16x.zero_()
    sim.ambient_count.zero_()

    sim.total_spawned_snake_mass.zero_()
    sim.total_spawned_ambient_mass.zero_()
    sim.total_corpse_decay_sink.zero_()
    sim.total_boost_decay_sink.zero_()
    sim.total_length_cap_sink.zero_()
    sim.total_truncation_sink.zero_()
    sim.tick = 0


def _world_to_fp(coord: float) -> int:
    """Convert world-unit coordinate (at cell_size=1.0) to fixed-point int32."""
    return int(round(coord * FP_ONE))


def _place_snake(
    sim: GameV3Sim,
    env: int,
    slot: int,
    points_xy_world: list[tuple[float, float]],
    theta_idx: int,
    length: int | None = None,
) -> None:
    """Place a live snake with explicit tail-to-head points in world units."""
    seg_cnt = len(points_xy_world)
    snake_len = seg_cnt if length is None else int(length)
    r_sq = int(sim._r_sq_table[min(snake_len, sim.cfg.max_length)].item())

    sim.alive[env, slot] = True
    sim.boosting[env, slot] = False
    sim.respawn_timer[env, slot] = 0
    sim.length[env, slot] = snake_len
    sim.seg_count[env, slot] = seg_cnt
    sim.head_ptr[env, slot] = seg_cnt - 1
    sim.oldest_active_subtick[env, slot] = 1
    sim.dist_accum_fp[env, slot] = 0
    sim.boost_ticks[env, slot] = 0
    sim.boost_drop_fp_accum[env, slot] = 0
    sim.life_age[env, slot] = 0
    sim.life_start[env, slot] = True
    sim.truncated_pending[env, slot] = False
    sim.theta[env, slot] = theta_idx & (sim.cfg.angle_steps - 1)

    for k, (wx, wy) in enumerate(points_xy_world):
        xfp = _world_to_fp(wx)
        yfp = _world_to_fp(wy)
        gx = (xfp >> FP_SHIFT) + sim.half_grid
        gy = (yfp >> FP_SHIFT) + sim.half_grid
        sim.ring_pos[env, slot, k, 0] = xfp
        sim.ring_pos[env, slot, k, 1] = yfp
        sim.ring_cell[env, slot, k, 0] = gx
        sim.ring_cell[env, slot, k, 1] = gy
        sim.ring_r_sq[env, slot, k] = r_sq
        sim.ring_subtick[env, slot, k] = 1

    sim.head_pos[env, slot] = sim.ring_pos[env, slot, seg_cnt - 1]
    sim.total_spawned_snake_mass[env] += snake_len


def _stamp_placed_snakes(
    sim: GameV3Sim, slot_order: torch.Tensor | None = None, subtick_val: int = 1
) -> None:
    """Stamp all active ring segments of placed snakes into 1x, 4x, and 16x occupancy grids."""
    max_seg = int(sim.seg_count.max().item())
    if max_seg <= 0:
        return
    idx_k = torch.arange(max_seg, dtype=torch.int32, device=sim.device)[None, None, :]
    active = sim.alive[:, :, None] & (idx_k < sim.seg_count[:, :, None])
    gx = sim.ring_cell[:, :, :max_seg, 0].to(torch.int32)
    gy = sim.ring_cell[:, :, :max_seg, 1].to(torch.int32)
    r_sq = sim.ring_r_sq[:, :, :max_seg].to(torch.int32)
    sub = torch.full_like(gx, subtick_val)
    sim._stamp_discs(gx, gy, r_sq, active, sub, slot_order=slot_order)


def _place_food_cell(
    sim: GameV3Sim, env: int, gx: int, gy: int, mass: int, *, corpse: bool = False
) -> None:
    """Place ambient or corpse food mass directly into 1x, 4x, and 16x grids."""
    if corpse:
        sim.corpse_grid[env, gy, gx] += mass
        sim.corpse_grid_4x[env, gy >> 2, gx >> 2] += mass
        sim.corpse_grid_16x[env, gy >> 4, gx >> 4] += mass
    else:
        sim.ambient_grid[env, gy, gx] += mass
        sim.ambient_grid_4x[env, gy >> 2, gx >> 2] += mass
        sim.ambient_grid_16x[env, gy >> 4, gx >> 4] += mass
        sim.ambient_count[env] += 1
        sim.total_spawned_ambient_mass[env] += mass


def test_self_crossing_never_kills_self() -> None:
    """A long snake steering tightly across its own body must never kill itself."""
    cfg = GameV3Config(arena_radius=80.0, num_snakes=6, ambient_food_density=0.0)
    sim = GameV3Sim(cfg, num_envs=1, device="cpu")
    _clear_world(sim)

    pts = [(-25.0 + 0.5 * i, 0.0) for i in range(60)]
    _place_snake(sim, 0, 0, pts, theta_idx=0, length=60)
    _stamp_placed_snakes(sim)

    # Action 0 = full hard left turn (-1.0 * turn_limit), normal speed.
    actions = torch.zeros((1, cfg.num_snakes), dtype=torch.int64)
    for _ in range(180):
        info = sim.step(actions)
        assert not bool(info.died[0, 0].item())
        assert bool(sim.alive[0, 0].item())


def test_boundary_kills_and_no_kill_credit() -> None:
    """Moving into the circular boundary kills the snake with killer_id=-1 and 0 kill credit."""
    cfg = GameV3Config(arena_radius=30.0, num_snakes=6, ambient_food_density=0.0)
    sim = GameV3Sim(cfg, num_envs=1, device="cpu")
    _clear_world(sim)

    _place_snake(sim, 0, 0, [(27.5, 0.0), (28.5, 0.0)], theta_idx=0, length=20)
    _stamp_placed_snakes(sim)

    # Action 2 = straight normal speed.
    actions = torch.full((1, cfg.num_snakes), 2, dtype=torch.int64)
    died = False
    for _ in range(5):
        info = sim.step(actions)
        if bool(info.died[0, 0].item()):
            died = True
            assert int(info.killer_id[0, 0].item()) == -1
            assert int(info.kill_credit.sum().item()) == 0
            assert int(sim.corpse_grid.sum().item()) > 0
            break
    assert died


def test_geometric_head_on_mutual_death_regardless_of_size() -> None:
    """Rule 1: Head-to-head is purely geometric; both snakes die even at L=300 vs L=10."""
    cfg = GameV3Config(arena_radius=80.0, num_snakes=6, ambient_food_density=0.0)
    sim = GameV3Sim(cfg, num_envs=2, device="cpu")
    _clear_world(sim)

    half_turn = cfg.angle_steps // 2  # facing -x
    # Env 0: equal lengths (20 vs 20) colliding head-on.
    _place_snake(sim, 0, 0, [(-1.5, 0.0), (-0.5, 0.0)], theta_idx=0, length=20)
    _place_snake(sim, 0, 1, [(1.5, 0.0), (0.5, 0.0)], theta_idx=half_turn, length=20)

    # Env 1: huge snake (L=300) vs tiny start-length snake (L=10) colliding head-on.
    # Under v3.1 Rule 1 there is NO size rule and NO 15% tie margin: both die.
    _place_snake(sim, 1, 0, [(-1.5, 0.0), (-0.5, 0.0)], theta_idx=0, length=300)
    _place_snake(sim, 1, 1, [(1.5, 0.0), (0.5, 0.0)], theta_idx=half_turn, length=10)
    _stamp_placed_snakes(sim)

    actions = torch.full((2, cfg.num_snakes), 2, dtype=torch.int64)
    info = sim.step(actions)

    for e in (0, 1):
        assert bool(info.died[e, 0].item()), f"Env {e} slot 0 should die in head-on"
        assert bool(info.died[e, 1].item()), f"Env {e} slot 1 should die in head-on"
        assert int(info.killer_id[e, 0].item()) == 1
        assert int(info.killer_id[e, 1].item()) == 0
        assert int(info.kill_credit[e, 0].item()) == 1
        assert int(info.kill_credit[e, 1].item()) == 1


def test_small_snake_kills_large_snake_cutting_across_head() -> None:
    """Rule 1: A small snake (L=10) kills a giant snake (L=300) by cutting across its head."""
    cfg = GameV3Config(arena_radius=80.0, num_snakes=6, ambient_food_density=0.0)
    sim = GameV3Sim(cfg, num_envs=1, device="cpu")
    _clear_world(sim)

    quarter_turn = cfg.angle_steps // 4  # facing +y
    # Giant snake (slot 0, L=300, body radius ~ 2.40) at (-0.8, 0.0) moving +x -> (0.2, 0.0).
    _place_snake(sim, 0, 0, [(-2.8, 0.0), (-1.8, 0.0), (-0.8, 0.0)], theta_idx=0, length=300)
    # Small snake (slot 1, L=10, body radius ~ 1.13) cutting across x = 0.5 heading +y:
    # trailing body sits directly in front of giant's path at (0.5, 0.0..3.0), while small's
    # head starts at (0.5, 4.0) and moves to (0.5, 5.0), cleanly outside giant's head disc!
    small_pts = [(0.5, float(y)) for y in (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0)]
    _place_snake(sim, 0, 1, small_pts, theta_idx=quarter_turn, length=10)
    _stamp_placed_snakes(sim)

    actions = torch.full((1, cfg.num_snakes), 2, dtype=torch.int64)
    info = sim.step(actions)

    # Giant dies hitting small's body; small survives and earns kill credit + corpse mass!
    assert bool(info.died[0, 0].item())
    assert not bool(info.died[0, 1].item())
    assert bool(sim.alive[0, 1].item())
    assert int(info.killer_id[0, 0].item()) == 1
    assert int(info.kill_credit[0, 1].item()) == 1
    assert int(sim.corpse_grid[0].sum().item()) == int(math.floor(cfg.corpse_mass_fraction * 300))


def test_simultaneous_multi_owner_collision_awards_smallest_other_id() -> None:
    """Rule 9: Own body never hides an enemy, and kill credit goes to smallest other owner id."""
    cfg = GameV3Config(arena_radius=80.0, num_snakes=8, ambient_food_density=0.0)
    sim = GameV3Sim(cfg, num_envs=1, device="cpu")
    _clear_world(sim)

    quarter_turn = cfg.angle_steps // 4
    # Target cell (0.5, 0.5) is covered simultaneously by:
    # - Victim snake 3's OWN tail loop (must never mask enemies!)
    # - Enemy snake 5's body
    # - Enemy snake 2's body (smallest other owner id -> must receive kill credit!)
    _place_snake(
        sim,
        0,
        3,
        [(0.5, 0.5), (-3.5, 0.5), (-3.5, 4.5), (-1.5, 4.5), (-0.5, 0.5)],
        theta_idx=0,
        length=25,
    )
    three_quarters = (3 * cfg.angle_steps) // 4  # facing -y
    _place_snake(
        sim, 0, 5, [(0.5, 0.5), (0.5, -4.5), (0.5, -8.5)], theta_idx=three_quarters, length=20
    )
    _place_snake(sim, 0, 2, [(0.5, 0.5), (0.5, 4.5), (0.5, 8.5)], theta_idx=quarter_turn, length=20)

    # Stamp in an adversarial slot order where slot 5 and own slot 3 write AFTER slot 2.
    perm = torch.tensor([2, 5, 3, 0, 1, 4, 6, 7], dtype=torch.int64)
    _stamp_placed_snakes(sim, slot_order=perm)

    actions = torch.full((1, cfg.num_snakes), 2, dtype=torch.int64)
    info = sim.step(actions)

    assert bool(info.died[0, 3].item())
    assert not bool(info.died[0, 2].item())
    assert not bool(info.died[0, 5].item())
    assert int(info.killer_id[0, 3].item()) == 2
    assert int(info.kill_credit[0, 2].item()) == 1
    assert int(info.kill_credit[0, 5].item()) == 0


def test_boost_does_not_stretch_body_and_ring_size_default() -> None:
    """Rule 3: Boosting emits 1 point per speed unit (no stretching); ring size defaults to 2048."""
    default_cfg = GameV3Config()
    assert default_cfg.max_length == 2048

    # High boost_shed_ticks isolates pure kinematics so length stays constant at 25.
    cfg = GameV3Config(
        arena_radius=150.0,
        num_snakes=6,
        boost_shed_ticks=1000,
        ambient_food_density=0.0,
    )
    sim = GameV3Sim(cfg, num_envs=1, device="cpu")
    assert sim.ring_pos.shape == (1, 6, 2048, 2)
    _clear_world(sim)

    # Both start as single-point seeds of target length 25 heading +x (theta=0).
    _place_snake(sim, 0, 0, [(-60.0, -10.0)], theta_idx=0, length=25)
    _place_snake(sim, 0, 1, [(-60.0, 10.0)], theta_idx=0, length=25)
    _stamp_placed_snakes(sim)

    actions = torch.full((1, cfg.num_snakes), 2, dtype=torch.int64)
    actions[0, 1] = 7  # straight + boost (2x speed -> 2 points/tick)

    for _ in range(30):
        sim.step(actions)

    # Both have grown to full 25 stored body points.
    assert int(sim.seg_count[0, 0].item()) == 25
    assert int(sim.seg_count[0, 1].item()) == 25

    # Boosting snake travelled twice as far (60 units vs 30 units).
    x_norm = int(sim.head_pos[0, 0, 0].item())
    x_boost = int(sim.head_pos[0, 1, 0].item())
    assert x_norm - _world_to_fp(-60.0) == 30 * FP_ONE
    assert x_boost - _world_to_fp(-60.0) == 60 * FP_ONE

    # Extract active ring points from oldest to head for both snakes: every consecutive step
    # must equal exactly speed_fp = 1024 (1.0 world unit), proving zero body stretching!
    for slot in (0, 1):
        hp = int(sim.head_ptr[0, slot].item())
        idxs = [(hp - k) % cfg.max_length for k in range(25)]
        pts = sim.ring_pos[0, slot, idxs]  # head -> tail
        diffs = pts[:-1] - pts[1:]
        assert torch.all(diffs[:, 0] == FP_ONE)
        assert torch.all(diffs[:, 1] == 0)


def test_boost_in_circle_loses_mass_and_masks_at_min_length() -> None:
    """Rule 4: Dropped boost food is boost_drop_fraction=0.5, so circle-boosting loses mass."""
    cfg = GameV3Config(
        arena_radius=100.0,
        num_snakes=6,
        min_boost_length=15,
        boost_shed_ticks=4,
        boost_shed_mass=1,
        boost_drop_fraction=0.5,
        ambient_food_density=0.0,
    )
    sim = GameV3Sim(cfg, num_envs=1, device="cpu")
    _clear_world(sim)

    pts = [(-10.0 + 1.0 * i, 0.0) for i in range(45)]
    _place_snake(sim, 0, 0, pts, theta_idx=0, length=45)
    _stamp_placed_snakes(sim)

    # Action 5 = full hard left (-1.0) + boost, looping over its own dropped pellets.
    actions = torch.full((1, cfg.num_snakes), 2, dtype=torch.int64)
    actions[0, 0] = 5

    for _ in range(320):
        sim.step(actions)

    # Even when eating part of its own trail, 0.5x drop value guarantees net mass loss
    # all the way down to min_boost_length (15), where boost becomes masked out.
    final_len = int(sim.length[0, 0].item())
    assert final_len == cfg.min_boost_length
    assert int(sim.total_boost_decay_sink[0].item()) >= 15
    assert not bool(sim.boosting[0, 0].item())

    obs = build_obs(sim)
    assert torch.all(obs.action_mask[0, 0, :5])
    assert not torch.any(obs.action_mask[0, 0, 5:])


def test_food_pickup_radius_and_contested_pellet_lowest_slot() -> None:
    """Rules 5 & 9: Pickup uses integer d^2 <= (1.5 * r)^2; contested pellet goes to lowest slot."""
    cfg = GameV3Config(
        arena_radius=80.0,
        num_snakes=6,
        food_pickup_mult=1.5,
        ambient_food_density=0.0,
    )
    sim = GameV3Sim(cfg, num_envs=1, device="cpu")
    _clear_world(sim)

    # Part A: Snake 0 (L=10) moves straight +x from (-0.5, 0.5) to (0.5, 0.5) = cell (half, half).
    # At L=10, body_radius ~ 1.131 -> pickup_r = 1.5 * 1.131 ~ 1.697 units.
    # Cell (half+1, half+1) center is at (1.5, 1.5) -> distance sqrt(2) = 1.414 <= 1.697 (eaten).
    # Cell (half+2, half) center is at (2.5, 0.5) -> distance 2.0 > 1.697 (not eaten).
    hg = sim.half_grid
    _place_snake(sim, 0, 0, [(-1.5, 0.5), (-0.5, 0.5)], theta_idx=0, length=10)
    _place_food_cell(sim, 0, hg + 1, hg + 1, mass=3, corpse=False)
    _place_food_cell(sim, 0, hg + 2, hg, mass=5, corpse=False)

    # Part B: Contested pellet at cell (hg, hg + 20) = (0.5, 20.5) worth 9 mass.
    # Surviving Snake 4 (moving to (-1.1, 20.5)) and surviving Snake 1 (moving to (2.1, 20.5))
    # stand 3.2 units apart (> head-collision sum ~ 2.48) while each is 1.6 units from (0.5, 20.5)
    # (<= pickup_r ~ 1.86); Rule 9 mandates lowest slot (slot 1) wins the entire pellet.
    half_turn = cfg.angle_steps // 2
    _place_snake(sim, 0, 4, [(-3.1, 20.5), (-2.1, 20.5)], theta_idx=0, length=20)
    _place_snake(sim, 0, 1, [(4.1, 20.5), (3.1, 20.5)], theta_idx=half_turn, length=20)
    _place_food_cell(sim, 0, hg, hg + 20, mass=9, corpse=True)
    _stamp_placed_snakes(sim)

    actions = torch.full((1, cfg.num_snakes), 2, dtype=torch.int64)
    info = sim.step(actions)

    # Part A assertions:
    assert int(info.mass_eaten[0, 0].item()) == 3
    assert int(sim.ambient_grid[0, hg + 1, hg + 1].item()) == 0
    assert int(sim.ambient_grid[0, hg, hg + 2].item()) == 5

    # Part B assertions:
    assert bool(sim.alive[0, 1].item()) and bool(sim.alive[0, 4].item())
    assert int(info.mass_eaten[0, 1].item()) == 9
    assert int(info.mass_eaten[0, 4].item()) == 0
    assert int(sim.corpse_grid[0, hg + 20, hg].item()) == 0


def test_occupancy_order_independence_under_permuted_slot_writes() -> None:
    """Rule 9: Permuting slot write order leaves occupancy pyramids, kills, and obs identical."""
    cfg = GameV3Config(arena_radius=60.0, num_snakes=10, ambient_food_density=0.005)
    sim_a = GameV3Sim(cfg, num_envs=2, device="cpu", seeds=[11, 22])
    sim_b = GameV3Sim(cfg, num_envs=2, device="cpu", seeds=[11, 22])
    _clear_world(sim_a)
    _clear_world(sim_b)

    # Place 6 overlapping snakes touching shared cells across both worlds.
    quarter = cfg.angle_steps // 4
    placements = [
        (0, [(-3.0, 0.5), (-2.0, 0.5), (-1.0, 0.5)], 0, 35),
        (1, [(0.5, -2.0), (0.5, 0.5), (0.5, 2.0)], quarter, 20),
        (4, [(0.5, -2.0), (0.5, 0.5), (0.5, 2.0)], quarter, 50),
        (7, [(1.5, -2.0), (0.5, 0.5), (-0.5, 2.0)], quarter, 15),
        (8, [(-4.0, 8.5), (-3.0, 8.5), (-2.0, 8.5)], 0, 25),
        (9, [(2.0, -8.5), (3.0, -8.5), (4.0, -8.5)], 0, 40),
    ]
    for sim in (sim_a, sim_b):
        for e in range(2):
            for slot, pts, th, length in placements:
                _place_snake(sim, e, slot, pts, theta_idx=th, length=length)

    # Stamp sim_a in canonical order [0..9] and sim_b in reversed/permuted order [9,7,4,1,8,0,...].
    perm = torch.tensor([9, 7, 4, 1, 8, 0, 6, 5, 3, 2], dtype=torch.int64)
    _stamp_placed_snakes(sim_a, slot_order=None)
    _stamp_placed_snakes(sim_b, slot_order=perm)

    assert torch.equal(sim_a.occ_hi, sim_b.occ_hi)
    assert torch.equal(sim_a.occ_lo, sim_b.occ_lo)
    assert torch.equal(sim_a.occ_subtick_4x, sim_b.occ_subtick_4x)
    assert torch.equal(sim_a.occ_subtick_16x, sim_b.occ_subtick_16x)

    obs_a0 = build_obs(sim_a)
    obs_b0 = build_obs(sim_b)
    for k in ("raster_s1", "raster_s4", "raster_s16", "scalars", "action_mask"):
        assert torch.equal(getattr(obs_a0, k), getattr(obs_b0, k))

    # Step forward and verify collisions, kill credits, and observations remain bitwise identical.
    actions = torch.full((2, cfg.num_snakes), 2, dtype=torch.int64)
    info_a = sim_a.step(actions)
    info_b = sim_b.step(actions)
    for k in info_a.__dataclass_fields__:
        assert torch.equal(getattr(info_a, k), getattr(info_b, k))


def test_respawn_k_candidate_determinism_fallback_and_mass_conservation() -> None:
    """Rule 6: Counter-RNG K-candidate respawns, deterministic fallback, and exact mass ledger."""
    # Part A & C: Determinism + exact mass conservation over 300 ticks incl. truncations.
    cfg = GameV3Config(
        arena_radius=90.0,
        num_snakes=16,
        respawn_delay=5,
        respawn_candidates=8,
        respawn_fallback_candidates=8,
        max_life_ticks=90,
        ambient_food_density=0.008,
    )
    sim1 = GameV3Sim(cfg, num_envs=4, device="cpu", seeds=[101, 202, 303, 404])
    sim2 = GameV3Sim(cfg, num_envs=4, device="cpu", seeds=[101, 202, 303, 404])

    rng = np.random.default_rng(77)
    respawns_seen = 0
    for t in range(300):
        act = torch.from_numpy(
            rng.integers(0, NUM_ACTIONS, size=(4, cfg.num_snakes), dtype=np.int64)
        )
        info1 = sim1.step(act)
        info2 = sim2.step(act)
        respawns_seen += int(info1.life_start.sum().item())

        assert torch.equal(sim1.head_pos, sim2.head_pos)
        assert torch.equal(sim1.theta, sim2.theta)
        assert torch.equal(info1.life_start, info2.life_start)

        if t % 25 == 0 or t == 299:
            snake_mass = (
                torch.where(sim1.alive, sim1.length, torch.zeros_like(sim1.length))
                .to(torch.int64)
                .sum(dim=1)
            )
            amb_mass = sim1.ambient_grid.to(torch.int64).sum(dim=(1, 2))
            cor_mass = sim1.corpse_grid.to(torch.int64).sum(dim=(1, 2))
            lhs = snake_mass + amb_mass + cor_mass
            rhs = (
                sim1.total_spawned_snake_mass
                + sim1.total_spawned_ambient_mass
                - sim1.total_corpse_decay_sink
                - sim1.total_boost_decay_sink
                - sim1.total_length_cap_sink
                - sim1.total_truncation_sink
            )
            assert torch.equal(lhs, rhs), f"Mass mismatch at tick {t}: {lhs} != {rhs}"

    assert respawns_seen >= 20

    # Part B: Force all K=8 counter-RNG candidates for slot 0 on tick 1 to be blocked by slot 1,
    # verifying that deterministic fallback spot selection places slot 0 cleanly with zero loops.
    cfg_fb = GameV3Config(
        arena_radius=80.0,
        num_snakes=6,
        respawn_candidates=8,
        respawn_fallback_candidates=8,
        ambient_food_density=0.0,
    )
    sim_fb = GameV3Sim(cfg_fb, num_envs=1, device="cpu", seeds=[999])
    _clear_world(sim_fb)
    sim_fb.seeds[0] = 999
    sim_fb.respawn_timer[0, 0] = 1  # Slot 0 will attempt respawn during step() at new_tick = 1

    # Compute the exact K=8 RNG candidate cells that slot 0 will draw at new_tick = 1.
    k_cand = cfg_fb.respawn_candidates
    slots = torch.arange(k_cand, dtype=torch.int64)
    seed_t = torch.tensor([999], dtype=torch.int64)[:, None]
    rng_gx = rng_int_torch(seed_t, 1, PURPOSE_RESPAWN, slots[None, :] * 3, sim_fb.G)[0]
    rng_gy = rng_int_torch(seed_t, 1, PURPOSE_RESPAWN, slots[None, :] * 3 + 1, sim_fb.G)[0]

    # Stamp enemy snake 1 onto every one of those K=8 RNG cells while leaving fallback ring clear.
    block_pts = [
        (
            float(int(gx.item()) - sim_fb.half_grid) + 0.5,
            float(int(gy.item()) - sim_fb.half_grid) + 0.5,
        )
        for gx, gy in zip(rng_gx, rng_gy)
    ]
    _place_snake(sim_fb, 0, 1, block_pts, theta_idx=0, length=20)
    _stamp_placed_snakes(sim_fb)

    info_fb = sim_fb.step(torch.full((1, cfg_fb.num_snakes), 2, dtype=torch.int64))
    assert bool(info_fb.life_start[0, 0].item())
    assert bool(sim_fb.alive[0, 0].item())

    # First fallback entry for (slot=0, tick=1, offset=0) is index (0*7 + 1*13 + 0) & 63 = 13.
    fb_idx0 = (0 * 7 + 1 * 13 + 0) & (sim_fb._fallback_spawns.shape[0] - 1)
    fb0_gx = int(sim_fb._fallback_spawns[fb_idx0, 0].item())
    fb0_gy = int(sim_fb._fallback_spawns[fb_idx0, 1].item())
    fb0_th = int(sim_fb._fallback_spawns[fb_idx0, 2].item())
    exp_pos = sim_fb._cell_center_fp(torch.tensor(fb0_gx), torch.tensor(fb0_gy))
    assert torch.equal(sim_fb.head_pos[0, 0], exp_pos)
    assert int(sim_fb.theta[0, 0].item()) == fb0_th


def test_episode_hooks_and_per_life_truncation() -> None:
    """Rule 12: Emit all 10 StepInfo hooks (no reward) and support per-life truncation."""
    cfg = GameV3Config(
        arena_radius=120.0,
        num_snakes=6,
        respawn_delay=4,
        max_life_ticks=12,
        ambient_food_density=0.0,
    )
    sim = GameV3Sim(cfg, num_envs=1, device="cpu", seeds=[7])
    _clear_world(sim)

    # Place snake 0 safe near origin so it survives to max_life_ticks=12 without dying.
    _place_snake(sim, 0, 0, [(-20.0, 0.0)], theta_idx=0, length=15)
    # Place snake 1 dead in respawn countdown (timer=2).
    sim.respawn_timer[0, 1] = 2
    _stamp_placed_snakes(sim)

    expected_fields = {
        "alive",
        "life_start",
        "died",
        "truncated",
        "in_respawn_delay",
        "length",
        "mass_eaten",
        "killer_id",
        "kill_credit",
        "life_age",
    }
    actions = torch.full((1, cfg.num_snakes), 2, dtype=torch.int64)

    for t in range(1, 16):
        info = sim.step(actions)
        assert set(info.__dataclass_fields__.keys()) == expected_fields
        assert not hasattr(info, "reward")

        # Mutual exclusivity of slot status: alive + died + in_respawn_delay == 1 everywhere.
        status_sum = (
            info.alive.to(torch.int32)
            + info.died.to(torch.int32)
            + info.in_respawn_delay.to(torch.int32)
        )
        assert torch.all(status_sum == 1)

        if t == 12:
            # Snake 0 hits per-life time limit: truncated=True, died=False, still alive for V(s_T)!
            assert bool(info.truncated[0, 0].item())
            assert not bool(info.died[0, 0].item())
            assert bool(info.alive[0, 0].item())
            assert int(info.life_age[0, 0].item()) == 12
            assert int(sim.corpse_grid.sum().item()) == 0
            obs_trunc = build_obs(sim)
            assert float(obs_trunc.scalars[0, 0, SCALAR_BOUNDARY_DIST].item()) > 0.0
        elif t == 13:
            # Next tick: truncated life is cleanly retired (no corpse) and immediately respawned.
            assert bool(info.life_start[0, 0].item())
            assert not bool(info.died[0, 0].item())
            assert not bool(info.truncated[0, 0].item())
            assert int(info.life_age[0, 0].item()) == 0
            assert int(sim.total_truncation_sink[0].item()) == 15
            assert int(sim.corpse_grid.sum().item()) == 0


def test_obs_channels_learner_mask_and_zero_host_syncs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rules 10 & 11: Verify uint8 rasters, owner size buckets, learner_mask, and 0 host syncs."""
    cfg = GameV3Config(arena_radius=120.0, num_snakes=24, ambient_food_density=0.006)
    sim = GameV3Sim(cfg, num_envs=4, device="cpu", seeds=[1, 2, 3, 4])

    # Advance 5 ticks with boosting enabled on several snakes so T0/T1 head stack & boost populate.
    for _ in range(5):
        act = food_seeking_actions(sim)
        sim.step(act)

    full_obs = build_obs(sim)
    assert full_obs.raster_s1.dtype == torch.uint8
    assert full_obs.raster_s4.dtype == torch.uint8
    assert full_obs.raster_s16.dtype == torch.uint8
    assert full_obs.scalars.dtype == torch.float32
    assert full_obs.action_mask.dtype == torch.bool

    assert full_obs.raster_s1.shape == (4, 24, NUM_RASTER_CHANNELS, 64, 64)
    assert full_obs.raster_s4.shape == (4, 24, NUM_RASTER_CHANNELS, 32, 32)
    assert full_obs.raster_s16.shape == (4, 24, NUM_RASTER_CHANNELS, 32, 32)
    assert full_obs.scalars.shape == (4, 24, NUM_SCALAR_FEATURES)
    assert full_obs.action_mask.shape == (4, 24, NUM_ACTIONS)

    # Verify enemy_body values match owner size_bucket values (>= 1) and out_of_arena uses 255.
    valid_buckets = set(sim._size_bucket_table.tolist()) | {0}
    assert set(torch.unique(full_obs.raster_s1[:, :, CH_ENEMY_BODY]).tolist()) <= valid_buckets
    assert set(torch.unique(full_obs.raster_s1[:, :, CH_ENEMY_HEADS_T0]).tolist()) <= valid_buckets
    assert set(torch.unique(full_obs.raster_s1[:, :, CH_OWN_BODY]).tolist()) <= {0, 255}
    assert set(torch.unique(full_obs.raster_s16[:, :, CH_OUT_OF_ARENA]).tolist()) == {0, 255}
    assert set(torch.unique(full_obs.raster_s1[:, :, CH_ENEMY_BOOST_T0]).tolist()) <= {0, 255}
    assert set(torch.unique(full_obs.scalars[:, :, SCALAR_OWN_BOOST]).tolist()) <= {0.0, 1.0}

    # Test learner_mask parity: 8-of-24 structured, random irregular, all-False, and all-True.
    rng = np.random.default_rng(2026)
    masks = [
        torch.tensor([[j < 8 for j in range(24)] for _ in range(4)], dtype=torch.bool),
        torch.from_numpy(rng.random((4, 24)) < 0.31),
        torch.zeros((4, 24), dtype=torch.bool),
        torch.ones((4, 24), dtype=torch.bool),
    ]
    for mask in masks:
        sub_obs = build_obs(sim, learner_mask=mask)
        pre_out = allocate_obs_batch(sim, learner_mask=mask)
        sub_pre = build_obs(sim, learner_mask=mask, out=pre_out)
        for field_name in ("raster_s1", "raster_s4", "raster_s16", "scalars", "action_mask"):
            expected = getattr(full_obs, field_name)[mask]
            assert torch.equal(getattr(sub_obs, field_name), expected)
            assert torch.equal(getattr(sub_pre, field_name), expected)

    # Verify zero host-sync operations during steady-state step + obs + scripted policies.
    full_out = allocate_obs_batch(sim)
    mask_8 = masks[0]
    sub_out_8 = allocate_obs_batch(sim, learner_mask=mask_8)

    def _forbid_sync(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Forbidden host synchronization in steady-state rollout!")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_sync)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbid_sync)
    monkeypatch.setattr(torch.Tensor, "numpy", _forbid_sync)
    monkeypatch.setattr(torch.Tensor, "nonzero", _forbid_sync)
    monkeypatch.setattr(torch, "nonzero", _forbid_sync)

    for _ in range(4):
        act1 = random_safe_actions(sim)
        sim.step(act1)
        build_obs(sim, out=full_out)
        build_obs(sim, learner_mask=mask_8, out=sub_out_8)
        act2 = food_seeking_actions(sim)
        sim.step(act2)


def test_mid_run_save_and_restore_bitwise_identity() -> None:
    """Saving and restoring state_dict mid-run reproduces bitwise-identical future trajectories."""
    cfg = GameV3Config(
        arena_radius=100.0,
        num_snakes=16,
        respawn_delay=6,
        max_life_ticks=35,
        ambient_food_density=0.01,
    )
    sim_a = GameV3Sim(cfg, num_envs=3, device="cpu", seeds=[501, 502, 503])
    rng = np.random.default_rng(888)

    for _ in range(45):
        act = torch.from_numpy(
            rng.integers(0, NUM_ACTIONS, size=(3, cfg.num_snakes), dtype=np.int64)
        )
        sim_a.step(act)

    saved = copy.deepcopy(sim_a.state_dict())
    sim_b = GameV3Sim(cfg, num_envs=3, device="cpu", seeds=[0, 0, 0])
    sim_b.load_state_dict(saved)

    for _ in range(50):
        act = torch.from_numpy(
            rng.integers(0, NUM_ACTIONS, size=(3, cfg.num_snakes), dtype=np.int64)
        )
        info_a = sim_a.step(act)
        info_b = sim_b.step(act)
        for k in info_a.__dataclass_fields__:
            assert torch.equal(getattr(info_a, k), getattr(info_b, k))

    assert torch.equal(sim_a.head_pos, sim_b.head_pos)
    assert torch.equal(sim_a.theta, sim_b.theta)
    assert torch.equal(sim_a.length, sim_b.length)
    assert torch.equal(sim_a.occ_hi, sim_b.occ_hi)
    assert torch.equal(sim_a.occ_lo, sim_b.occ_lo)
    assert torch.equal(sim_a.ambient_grid, sim_b.ambient_grid)
    assert torch.equal(sim_a.corpse_grid, sim_b.corpse_grid)

    obs_a = build_obs(sim_a)
    obs_b = build_obs(sim_b)
    for k in ("raster_s1", "raster_s4", "raster_s16", "scalars", "action_mask"):
        assert torch.equal(getattr(obs_a, k), getattr(obs_b, k))
