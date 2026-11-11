"""Core gameplay, geometry, occupancy, episode-hook, and zero-sync invariants for Game V3.1."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from research.game_v3 import cuda_check
from research.game_v3 import sim as sim_mod
from research.game_v3.bench import _tensor_bytes_of_sim_and_obs
from research.game_v3.config import FP_ONE, FP_SHIFT, NUM_ACTIONS, GameV3Config
from research.game_v3.obs import (
    CH_ENEMY_BODY,
    CH_ENEMY_BOOST_T0,
    CH_ENEMY_HEADS_T0,
    CH_ENEMY_HEADS_T1,
    CH_OUT_OF_ARENA,
    CH_OWN_BODY,
    NUM_RASTER_CHANNELS,
    NUM_SCALAR_FEATURES,
    SCALAR_BOUNDARY_DIST,
    SCALAR_OWN_BOOST,
    LearnerIndex,
    allocate_obs_batch,
    build_obs,
    build_obs_reference,
)
from research.game_v3.reference import ReferenceWorld
from research.game_v3.rng import PURPOSE_RESPAWN, rng_int_torch
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import (
    AUTO_REBASE_SUBTICK_THRESHOLD,
    CORPSE_GRID_MAX,
    MAX_OCC_SUBTICK,
    GameV3Sim,
)


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
    sim.world_ticks.zero_()
    sim.tick = 0
    sim._subtick_offset = 0
    sim._clean_step_streak = 0


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
    """A long snake (L=150) steering tightly across its own body crosses discs and survives."""
    cfg = GameV3Config(arena_radius=180.0, num_snakes=6, ambient_food_density=0.0)
    sim = GameV3Sim(cfg, num_envs=1, device="cpu")
    _clear_world(sim)

    # At L=150, body_radius = 2.0 units (r_sq = 4 -> 13-cell disc) and turn circle diameter
    # is ~33 units, while the 150-point body trunk spans 149 units along y = 0.
    pts = [(-120.0 + 1.0 * i, 0.0) for i in range(150)]
    _place_snake(sim, 0, 0, pts, theta_idx=0, length=150)
    _stamp_placed_snakes(sim)

    disc_mask = sim._disc_dist_sq.numpy() <= 4
    dx_np = sim._disc_dx.numpy()[disc_mask]
    dy_np = sim._disc_dy.numpy()[disc_mask]

    # Action 0 = full hard left (+turn_limit), looping +x -> +y -> -x -> -y across y = 0.
    actions = torch.zeros((1, cfg.num_snakes), dtype=torch.int64)
    min_head_to_body_dist = float("inf")
    max_shared_own_cells = 0

    for _ in range(95):
        info = sim.step(actions)
        assert not bool(info.died[0, 0].item())
        assert bool(sim.alive[0, 0].item())

        seg_cnt = int(sim.seg_count[0, 0].item())
        hp = int(sim.head_ptr[0, 0].item())
        # Compare current head against older active body segments (>= 20 segments behind head).
        older_idxs = [(hp - k) % cfg.max_length for k in range(20, seg_cnt)]
        hx, hy = (
            float(sim.head_pos[0, 0, 0].item()) / FP_ONE,
            float(sim.head_pos[0, 0, 1].item()) / FP_ONE,
        )
        older_pos = sim.ring_pos[0, 0, older_idxs].numpy().astype(np.float64) / FP_ONE
        dists = np.hypot(older_pos[:, 0] - hx, older_pos[:, 1] - hy)
        min_head_to_body_dist = min(min_head_to_body_dist, float(dists.min()))

        hgx, hgy = (
            int(sim.ring_cell[0, 0, hp, 0].item()),
            int(sim.ring_cell[0, 0, hp, 1].item()),
        )
        head_cells = {(hgx + int(dx), hgy + int(dy)) for dx, dy in zip(dx_np, dy_np)}
        older_cells_arr = sim.ring_cell[0, 0, older_idxs].numpy().astype(np.int32)
        own_older_cells = {
            (int(cx) + int(dx), int(cy) + int(dy))
            for cx, cy in older_cells_arr
            for dx, dy in zip(dx_np, dy_np)
        }
        max_shared_own_cells = max(max_shared_own_cells, len(head_cells & own_older_cells))

    # Prove the test is non-vacuous: the head center passed well inside its 2.0-unit body radius
    # and overlapped >= 8 grid cells of its own active body while surviving cleanly.
    assert (
        min_head_to_body_dist < 1.0
    ), f"Expected real crossing, got min_dist={min_head_to_body_dist}"
    assert max_shared_own_cells >= 8, f"Expected disc overlap, got {max_shared_own_cells} cells"


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


def test_multi_hit_kill_credit_uses_counter_rng_draw_and_matches_reference() -> None:
    """Own body never hides enemies; multi-hitter credit uses counter RNG matching reference."""
    cfg = GameV3Config(arena_radius=80.0, num_snakes=8, ambient_food_density=0.0)
    num_envs = 32
    seeds = [100 + e for e in range(num_envs)]
    sim_a = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    sim_b = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)
    _clear_world(sim_a)
    _clear_world(sim_b)
    sim_a.seeds.copy_(torch.tensor(seeds, dtype=torch.int64))
    sim_b.seeds.copy_(torch.tensor(seeds, dtype=torch.int64))

    quarter_turn = cfg.angle_steps // 4
    three_quarters = (3 * cfg.angle_steps) // 4  # facing -y
    victim_pts = [(0.5, 0.5), (-3.5, 0.5), (-3.5, 4.5), (-1.5, 4.5), (-0.5, 0.5)]
    enemy5_pts = [(0.5, 0.5), (0.5, -4.5), (0.5, -8.5)]
    enemy2_pts = [(0.5, 0.5), (0.5, 4.5), (0.5, 8.5)]

    refs: list[ReferenceWorld] = []
    for e, s in enumerate(seeds):
        for sim in (sim_a, sim_b):
            _place_snake(sim, e, 3, victim_pts, theta_idx=0, length=25)
            _place_snake(sim, e, 5, enemy5_pts, theta_idx=three_quarters, length=20)
            _place_snake(sim, e, 2, enemy2_pts, theta_idx=quarter_turn, length=20)

        ref = ReferenceWorld(cfg, seed=s)
        ref.alive.fill(False)
        ref.length.fill(0)
        ref.seg_count.fill(0)
        ref.respawn_timer.fill(10_000)
        ref.ambient_grid.fill(0)
        ref.corpse_grid.fill(0)
        ref.ambient_count = 0
        for slot, pts, th, slen in (
            (3, victim_pts, 0, 25),
            (5, enemy5_pts, three_quarters, 20),
            (2, enemy2_pts, quarter_turn, 20),
        ):
            ref.alive[slot] = True
            ref.life_start[slot] = True
            ref.respawn_timer[slot] = 0
            ref.length[slot] = slen
            ref.seg_count[slot] = len(pts)
            ref.head_ptr[slot] = len(pts) - 1
            ref.theta[slot] = th
            rsq = int(ref.tables["r_sq_cells"][slen])
            for k, (wx, wy) in enumerate(pts):
                xfp, yfp = _world_to_fp(wx), _world_to_fp(wy)
                ref.ring_pos[slot, k] = (xfp, yfp)
                ref.ring_cell[slot, k] = (
                    (xfp >> FP_SHIFT) + ref.half_grid,
                    (yfp >> FP_SHIFT) + ref.half_grid,
                )
                ref.ring_r_sq[slot, k] = rsq
            ref.head_pos[slot] = ref.ring_pos[slot, len(pts) - 1]
        ref._rebuild_occ_mask_and_grid()
        refs.append(ref)

    perm = torch.tensor([2, 5, 3, 0, 1, 4, 6, 7], dtype=torch.int64)
    _stamp_placed_snakes(sim_a, slot_order=None)
    _stamp_placed_snakes(sim_b, slot_order=perm)

    actions = torch.full((num_envs, cfg.num_snakes), 2, dtype=torch.int64)
    info_a = sim_a.step(actions)
    info_b = sim_b.step(actions)
    ref_infos = [r.step(np.full((cfg.num_snakes,), 2, dtype=np.int32)) for r in refs]

    assert torch.all(info_a.died[:, 3])
    assert not torch.any(info_a.died[:, 2])
    assert not torch.any(info_a.died[:, 5])
    assert torch.equal(info_a.killer_id, info_b.killer_id)
    assert torch.equal(info_a.kill_credit, info_b.kill_credit)

    killers = info_a.killer_id[:, 3].tolist()
    assert set(killers) == {2, 5}, f"Expected both hitters {2, 5} credited, got {set(killers)}"
    assert killers.count(2) >= 8 and killers.count(5) >= 8

    ref_killers = [int(ri.killer_id[3]) for ri in ref_infos]
    assert killers == ref_killers


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
    fb0_th_raw = int(sim_fb._fallback_spawns[fb_idx0, 2].item())
    fb0_th = int(
        sim_fb._orient_spawn_heading_torch(
            torch.tensor(fb0_gx), torch.tensor(fb0_gy), torch.tensor(fb0_th_raw)
        ).item()
    )
    exp_pos = sim_fb._cell_center_fp(torch.tensor(fb0_gx), torch.tensor(fb0_gy))
    assert torch.equal(sim_fb.head_pos[0, 0], exp_pos)
    assert int(sim_fb.theta[0, 0].item()) == fb0_th


def test_episode_hooks_and_per_life_truncation() -> None:
    """Rule 12: Emit all 11 StepInfo hooks (incl. prev_length; no reward) and life truncation."""
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
        "prev_length",
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
        idx = LearnerIndex(sim, mask)
        sub_obs = build_obs(sim, idx)
        pre_out = allocate_obs_batch(sim, idx)
        sub_pre = build_obs(sim, idx, out=pre_out)
        for field_name in ("raster_s1", "raster_s4", "raster_s16", "scalars", "action_mask"):
            expected = getattr(full_obs, field_name)[mask]
            assert torch.equal(getattr(sub_obs, field_name), expected)
            assert torch.equal(getattr(sub_pre, field_name), expected)

    # Verify zero host-sync operations during steady-state step + obs + scripted policies.
    full_out = allocate_obs_batch(sim)
    idx_8 = LearnerIndex(sim, masks[0])
    sub_out_8 = allocate_obs_batch(sim, idx_8)

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
        build_obs(sim, idx_8, out=sub_out_8)
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


def test_learner_index_address_reuse_and_out_mismatch_refusal() -> None:
    """MAJOR 1: LearnerIndex avoids stale address-reuse rows and refuses mismatched out batches."""
    cfg = GameV3Config(arena_radius=100.0, num_snakes=12, ambient_food_density=0.01)
    sim = GameV3Sim(cfg, num_envs=2, device="cpu", seeds=[11, 22])
    for _ in range(4):
        sim.step(food_seeking_actions(sim))
    full_obs = build_obs(sim)

    # Part A: Repeatedly allocate/free masks selecting [0, 1] then [5, 6] to trigger CPU allocator
    # address reuse at _version == 0; build_obs must never return stale [0, 1] rows.
    for _ in range(48):
        m_a = torch.zeros((2, 12), dtype=torch.bool)
        m_a[0, 0] = True
        m_a[0, 1] = True
        obs_a = build_obs(sim, m_a)
        assert torch.equal(obs_a.scalars, full_obs.scalars[m_a])
        del m_a, obs_a

        m_b = torch.zeros((2, 12), dtype=torch.bool)
        m_b[0, 5] = True
        m_b[0, 6] = True
        obs_b = build_obs(sim, m_b)
        assert torch.equal(obs_b.scalars, full_obs.scalars[m_b])
        assert torch.equal(obs_b.raster_s1, full_obs.raster_s1[m_b])
        del m_b, obs_b

    # Part B: Explicit LearnerIndex + strict refusal of any mismatched mask/index on out batch.
    mask_01 = torch.zeros((2, 12), dtype=torch.bool)
    mask_01[0, 0] = True
    mask_01[0, 1] = True
    mask_56 = torch.zeros((2, 12), dtype=torch.bool)
    mask_56[0, 5] = True
    mask_56[0, 6] = True

    idx_01 = LearnerIndex(sim, mask_01)
    idx_56 = LearnerIndex(sim, mask_56)
    out_01 = allocate_obs_batch(sim, idx_01)
    out_full = allocate_obs_batch(sim)

    got_01 = build_obs(sim, idx_01, out=out_01)
    assert torch.equal(got_01.raster_s1, full_obs.raster_s1[mask_01])

    with pytest.raises(ValueError, match="differs"):
        build_obs(sim, idx_56, out=out_01)
    with pytest.raises(ValueError, match="differs"):
        build_obs(sim, mask_56, out=out_01)
    with pytest.raises(ValueError):
        build_obs(sim, None, out=out_01)
    with pytest.raises(ValueError):
        build_obs(sim, idx_01, out=out_full)

    # Mutating caller's original mask tensor after LearnerIndex creation must not corrupt idx_01,
    # and passing the mutated tensor with out_01 must be refused.
    mask_01.zero_()
    mask_01[1, 3] = True
    mask_01[1, 4] = True
    with pytest.raises(ValueError, match="differs"):
        build_obs(sim, mask_01, out=out_01)
    still_ok = build_obs(sim, idx_01, out=out_01)
    assert torch.equal(still_ok.scalars, full_obs.scalars[idx_01.mask])


def test_subtick_overflow_guard_and_forced_rebase_bitwise_identity() -> None:
    """MAJOR 2a/2b: Host overflow check raises pre-limit; forced mid-run rebase is bitwise-exact."""
    cfg = GameV3Config(arena_radius=80.0, num_snakes=10, respawn_delay=5, ambient_food_density=0.01)

    # Part A: With auto_rebase=False, stepping into MAX_OCC_SUBTICK raises OverflowError on host.
    sim_guard = GameV3Sim(cfg, num_envs=1, device="cpu", seeds=[7], auto_rebase=False)
    sim_guard.tick = MAX_OCC_SUBTICK // sim_guard.max_pts
    with pytest.raises(OverflowError, match="overflow"):
        sim_guard.step(torch.full((1, cfg.num_snakes), 2, dtype=torch.int64))

    # Part B: With auto_rebase=True (default), crossing AUTO_REBASE_SUBTICK_THRESHOLD rebases
    # automatically and preserves collisions past 17M ticks without overflow.
    sim_auto = GameV3Sim(cfg, num_envs=1, device="cpu", seeds=[7])
    jump_ticks = (AUTO_REBASE_SUBTICK_THRESHOLD // sim_auto.max_pts) - 2
    st_bump = jump_ticks * sim_auto.max_pts
    sim_auto.tick = jump_ticks
    sim_auto.world_ticks.fill_(jump_ticks)
    sim_auto.ring_subtick = torch.where(
        sim_auto.ring_subtick > 0, sim_auto.ring_subtick + st_bump, 0
    )
    sim_auto.oldest_active_subtick = sim_auto.oldest_active_subtick + st_bump
    sim_auto.occ_hi = torch.where(
        sim_auto.occ_hi >= 0,
        (((sim_auto.occ_hi >> 6) + st_bump) << 6) | (sim_auto.occ_hi & 63),
        -1,
    )
    sim_auto.occ_lo = torch.where(
        sim_auto.occ_lo >= 0,
        (((sim_auto.occ_lo >> 6) + st_bump) << 6) | (sim_auto.occ_lo & 63),
        -1,
    )
    for blk in sim_auto._occ_4x_blocks:
        blk.copy_(torch.where(blk >= 0, blk + st_bump, -1))
    sim_auto.occ_subtick_16x = torch.where(
        sim_auto.occ_subtick_16x >= 0, sim_auto.occ_subtick_16x + st_bump, -1
    )

    for _ in range(6):
        sim_auto.step(food_seeking_actions(sim_auto))
    assert sim_auto._subtick_offset > 0
    assert int(sim_auto.occ_hi.max().item()) >= 0
    assert int((sim_auto.occ_hi >> 6).max().item()) < AUTO_REBASE_SUBTICK_THRESHOLD

    # Part C: Forced mid-way rebase produces bitwise-identical outcomes to an unrebased run.
    sim_plain = GameV3Sim(cfg, num_envs=3, device="cpu", seeds=[101, 202, 303])
    sim_rebased = GameV3Sim(cfg, num_envs=3, device="cpu", seeds=[101, 202, 303])
    rng = np.random.default_rng(4242)

    for t in range(1, 61):
        act = torch.from_numpy(
            rng.integers(0, NUM_ACTIONS, size=(3, cfg.num_snakes), dtype=np.int64)
        )
        info_p = sim_plain.step(act)
        info_r = sim_rebased.step(act)
        if t in (20, 40):
            shifted = sim_rebased.rebase_subticks()
            assert shifted > 0

        for k in info_p.__dataclass_fields__:
            assert torch.equal(getattr(info_p, k), getattr(info_r, k)), f"Mismatch on {k} at {t}"
        assert torch.equal(sim_plain.active_occupancy_grid(), sim_rebased.active_occupancy_grid())

    obs_p = build_obs(sim_plain)
    obs_r = build_obs(sim_rebased)
    for k in ("raster_s1", "raster_s4", "raster_s16", "scalars", "action_mask"):
        assert torch.equal(getattr(obs_p, k), getattr(obs_r, k))


def test_reset_worlds_mask_isolation_and_reference_parity() -> None:
    """MAJOR 2c: reset_worlds(mask) resets selected worlds cleanly without disturbing others."""
    cfg = GameV3Config(arena_radius=90.0, num_snakes=10, respawn_delay=4, ambient_food_density=0.01)
    init_seeds = [400, 401, 402, 403]
    sim = GameV3Sim(cfg, num_envs=4, device="cpu", seeds=init_seeds)
    refs = [ReferenceWorld(cfg, seed=s) for s in init_seeds]

    for _ in range(25):
        act = food_seeking_actions(sim)
        act_np = act.numpy()
        sim.step(act)
        for e in range(4):
            refs[e].step(act_np[e])

    pre_head_0 = sim.head_pos[0].clone()
    pre_head_2 = sim.head_pos[2].clone()
    pre_amb_0 = sim.ambient_grid[0].clone()

    mask = torch.tensor([False, True, False, True], dtype=torch.bool)
    sim.reset_worlds(mask, seeds=[9101, 9103])
    refs[1] = ReferenceWorld(cfg, seed=9101)
    refs[3] = ReferenceWorld(cfg, seed=9103)

    assert torch.equal(sim.head_pos[0], pre_head_0)
    assert torch.equal(sim.head_pos[2], pre_head_2)
    assert torch.equal(sim.ambient_grid[0], pre_amb_0)
    assert sim.world_ticks.tolist() == [25, 0, 25, 0]

    for _ in range(30):
        act = random_safe_actions(sim)
        act_np = act.numpy()
        info = sim.step(act)
        ref_infos = [refs[e].step(act_np[e]) for e in range(4)]
        for k in info.__dataclass_fields__:
            exp = np.stack([getattr(ri, k) for ri in ref_infos], axis=0)
            assert np.array_equal(getattr(info, k).numpy(), exp)

    obs_sim = build_obs(sim)
    for e in range(4):
        obs_ref = build_obs_reference(refs[e])
        assert np.array_equal(obs_sim.raster_s1[e].numpy(), obs_ref.raster_s1)
        assert np.array_equal(
            obs_sim.scalars[e].view(torch.uint32).numpy(), obs_ref.scalars.view(np.uint32)
        )

    if torch.backends.mps.is_available():
        sim_mps = GameV3Sim(cfg, num_envs=4, device="mps", seeds=init_seeds)
        for _ in range(25):
            sim_mps.step(food_seeking_actions(sim_mps))
        sim_mps.reset_worlds(mask.to("mps"), seeds=[9101, 9103])
        for _ in range(30):
            sim_mps.step(random_safe_actions(sim_mps))
        assert torch.equal(sim_mps.head_pos.cpu(), sim.head_pos)
        assert torch.equal(sim_mps.ambient_grid.cpu(), sim.ambient_grid)


def test_default_max_life_ticks_off_and_rollout_truncation_keeps_snakes() -> None:
    """MAJOR 3: Default max_life_ticks=0 keeps large snakes; rollout truncate is non-destructive."""
    cfg = GameV3Config(arena_radius=120.0, num_snakes=6, ambient_food_density=0.0)
    assert cfg.max_life_ticks == 0

    sim = GameV3Sim(cfg, num_envs=2, device="cpu", seeds=[1, 2])
    _clear_world(sim)
    _place_snake(sim, 0, 0, [(-30.0, 0.0)], theta_idx=0, length=200)
    _place_snake(sim, 1, 0, [(-30.0, 0.0)], theta_idx=0, length=200)
    sim.life_age[0, 0] = 3599
    sim.life_age[1, 0] = 3599
    _stamp_placed_snakes(sim)

    actions = torch.full((2, cfg.num_snakes), 2, dtype=torch.int64)
    # Cross tick 3600 with default max_life_ticks=0: snake must NOT be truncated or retired!
    info_3600 = sim.step(actions)
    assert not bool(info_3600.truncated.any().item())
    assert int(sim.length[0, 0].item()) == 200
    assert int(info_3600.life_age[0, 0].item()) == 3600

    # Flag rollout-boundary truncation on world 0 only via step(..., truncate=world_mask).
    w_mask = torch.tensor([True, False], dtype=torch.bool)
    info_roll = sim.step(actions, truncate=w_mask)
    assert bool(info_roll.truncated[0, 0].item())
    assert not bool(tuple(info_roll.truncated[0, 1:].tolist())[0])  # dead slots stay False
    assert not bool(info_roll.truncated[1, 0].item())
    assert int(sim.length[0, 0].item()) == 200

    # Next tick after rollout truncation: snake in world 0 MUST still be alive with length=200!
    info_after = sim.step(actions)
    assert bool(info_after.alive[0, 0].item())
    assert not bool(info_after.life_start[0, 0].item())
    assert int(sim.length[0, 0].item()) == 200
    assert int(sim.total_truncation_sink.sum().item()) == 0


def test_compact_food_grid_dtypes_saturation_and_memory_budget() -> None:
    """MAJOR 4: ambient_grid is uint8, corpse_grid is int16 saturating at 32767, saving memory."""
    cfg_small = GameV3Config(arena_radius=60.0, num_snakes=6, ambient_food_density=0.0)
    sim = GameV3Sim(cfg_small, num_envs=1, device="cpu", seeds=[5])
    ref = ReferenceWorld(cfg_small, seed=5)
    assert sim.ambient_grid.dtype == torch.uint8
    assert sim.corpse_grid.dtype == torch.int16
    assert ref.ambient_grid.dtype == np.uint8
    assert ref.corpse_grid.dtype == np.int16

    # Pre-load a border cell near the right wall with 32,760 corpse mass and walk a length-100
    # snake into the wall so its 70-mass single-orb corpse saturates that cell at 32,767.
    _clear_world(sim)
    ref.alive.fill(False)
    ref.length.fill(0)
    ref.seg_count.fill(0)
    ref.respawn_timer.fill(10_000)
    ref.ambient_grid.fill(0)
    ref.corpse_grid.fill(0)
    ref.ambient_count = 0

    wx = 58.5
    xfp = _world_to_fp(wx)
    gx = (xfp >> FP_SHIFT) + sim.half_grid
    gy = sim.half_grid
    _place_food_cell(sim, 0, gx, gy, mass=32760, corpse=True)
    ref.corpse_grid[gy, gx] = np.int16(32760)

    _place_snake(sim, 0, 0, [(wx, 0.5)], theta_idx=0, length=100)
    _stamp_placed_snakes(sim)
    ref.alive[0] = True
    ref.life_start[0] = True
    ref.respawn_timer[0] = 0
    ref.length[0] = 100
    ref.seg_count[0] = 1
    ref.head_ptr[0] = 0
    ref.theta[0] = 0
    ref.head_pos[0] = (xfp, _world_to_fp(0.5))
    ref.ring_pos[0, 0] = ref.head_pos[0]
    ref.ring_cell[0, 0] = (gx, gy)
    ref.ring_r_sq[0, 0] = int(ref.tables["r_sq_cells"][100])
    ref._rebuild_occ_mask_and_grid()

    info = sim.step(torch.full((1, 6), 2, dtype=torch.int64))
    ref_info = ref.step(np.full((6,), 2, dtype=np.int32))
    assert bool(info.died[0, 0].item()) and bool(ref_info.died[0])
    assert int(sim.corpse_grid[0, gy, gx].item()) == CORPSE_GRID_MAX == 32767
    assert int(ref.corpse_grid[gy, gx]) == 32767

    # Coarse 4x and 16x pyramids and mass ledger remain exact after int16 saturation.
    cor_1x_i32 = sim.corpse_grid.to(torch.int32)
    assert torch.equal(
        sim.corpse_grid_4x, cor_1x_i32.reshape(1, sim.G4, 4, sim.G4, 4).sum(dim=(2, 4))
    )
    assert torch.equal(
        sim.corpse_grid_16x, cor_1x_i32.reshape(1, sim.G16, 16, sim.G16, 16).sum(dim=(2, 4))
    )
    lhs = int(sim.corpse_grid.to(torch.int64).sum().item())
    rhs = (
        32760
        + int(sim.total_spawned_snake_mass[0].item())
        - int(sim.total_corpse_decay_sink[0].item())
    )
    assert lhs == rhs == 32767

    obs_s = build_obs(sim)
    obs_r = build_obs_reference(ref)
    assert np.array_equal(obs_s.raster_s1[0].numpy(), obs_r.raster_s1)

    # Verify R=600, N=24 per-world sim footprint dropped well below the old 34.87 MiB/world.
    sim_r600 = GameV3Sim(GameV3Config(), num_envs=2, device="cpu")
    mib_per_world = _tensor_bytes_of_sim_and_obs(sim_r600, None) / (2 * 1024.0 * 1024.0)
    assert mib_per_world < 28.0, f"Expected < 28 MiB/world at R=600, got {mib_per_world:.2f}"


def test_surviving_discs_never_overlap_long_random_run() -> None:
    """SMALLER 1: debug_assert_invariants verifies <= 1 surviving owner/cell over a long run."""
    cfg = GameV3Config(
        arena_radius=90.0,
        num_snakes=20,
        respawn_delay=4,
        ambient_food_density=0.012,
    )
    sim = GameV3Sim(
        cfg, num_envs=4, device="cpu", seeds=[11, 22, 33, 44], debug_assert_invariants=True
    )
    rng = np.random.default_rng(31337)
    for t in range(250):
        if t % 3 == 0:
            act = food_seeking_actions(sim)
        elif t % 3 == 1:
            act = random_safe_actions(sim)
        else:
            act = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(4, cfg.num_snakes), dtype=np.int64)
            )
        sim.step(act)

    # Also confirm assert_surviving_discs_disjoint catches synthetic overlapping survivors.
    sim_bad = GameV3Sim(GameV3Config(arena_radius=60.0, num_snakes=6), num_envs=1, device="cpu")
    _clear_world(sim_bad)
    _place_snake(sim_bad, 0, 0, [(0.5, 0.5)], theta_idx=0, length=20)
    _place_snake(sim_bad, 0, 1, [(0.5, 0.5)], theta_idx=0, length=20)
    _stamp_placed_snakes(sim_bad)
    with pytest.raises(AssertionError, match="Surviving-disc overlap"):
        sim_bad.assert_surviving_discs_disjoint()


def test_spawn_wall_margin_20_and_headings_point_away_from_wall() -> None:
    """SMALLER 4: Spawn margin is 20 units and initial/respawn headings point inward (dot <= 0)."""
    cfg = GameV3Config(arena_radius=80.0, num_snakes=24, respawn_delay=2, ambient_food_density=0.01)
    assert cfg.spawn_wall_margin == 20.0

    sim = GameV3Sim(cfg, num_envs=6, device="cpu", seeds=[51, 52, 53, 54, 55, 56])
    refs = [ReferenceWorld(cfg, seed=s) for s in [51, 52, 53, 54, 55, 56]]
    max_r_fp = (cfg.arena_radius - cfg.spawn_wall_margin) * FP_ONE

    def _assert_spawn_pose(mask: torch.Tensor) -> None:
        es, slots = torch.where(mask)
        if int(es.shape[0]) == 0:
            return
        pos = sim.head_pos[es, slots].to(torch.int64)
        th = sim.theta[es, slots].to(torch.int64)
        d_sq = pos[:, 0] * pos[:, 0] + pos[:, 1] * pos[:, 1]
        assert bool((d_sq < max_r_fp * max_r_fp).all().item())
        dot = pos[:, 0] * sim._cos_table[th].to(torch.int64) + pos[:, 1] * sim._sin_table[th].to(
            torch.int64
        )
        assert bool((dot <= 0).all().item()), f"Found outward spawn heading: max dot={dot.max()}"

    _assert_spawn_pose(sim.life_start)
    for e in range(6):
        assert np.array_equal(sim.theta[e].numpy(), refs[e].theta)

    # Drive snakes straight so many hit the wall and respawn repeatedly.
    straight = torch.full((6, cfg.num_snakes), 2, dtype=torch.int64)
    total_respawns = 0
    for _ in range(90):
        info = sim.step(straight)
        for e in range(6):
            refs[e].step(np.full((cfg.num_snakes,), 2, dtype=np.int32))
        _assert_spawn_pose(info.life_start)
        total_respawns += int(info.life_start.sum().item())
        for e in range(6):
            assert np.array_equal(sim.theta[e].numpy(), refs[e].theta)
    assert total_respawns >= 60


def test_obs_prev_enemy_heads_projected_into_current_ego_frame() -> None:
    """SMALLER 5: Previous-tick (t-1) enemy heads are projected into the CURRENT observer frame."""
    # Set turn_rate = pi/2 so action 0 turns a start-length snake by +90 deg (+x -> +y) in 1 tick.
    cfg = GameV3Config(
        arena_radius=80.0,
        num_snakes=6,
        turn_rate=math.pi / 2.0,
        ambient_food_density=0.0,
    )
    sim = GameV3Sim(cfg, num_envs=1, device="cpu", seeds=[1])
    ref = ReferenceWorld(cfg, seed=1)
    _clear_world(sim)
    ref.alive.fill(False)
    ref.length.fill(0)
    ref.seg_count.fill(0)
    ref.respawn_timer.fill(10_000)
    ref.ambient_grid.fill(0)
    ref.corpse_grid.fill(0)
    ref.ambient_count = 0

    # At t-1: observer slot 0 is at (0.5, 0.5) facing +x (theta=0); enemy slot 1 is at (0.5, 12.5).
    # In the OLD code (previous ego frame facing +x), (0.5, 12.5) was to the LEFT (col < 32).
    # On tick 1, observer turns +90 deg to face +y: in the CURRENT ego frame facing +y, enemy's t-1
    # head at (0.5, 12.5) is STRAIGHT AHEAD (+ego_y -> row < 25, col ~ 32)!
    _place_snake(sim, 0, 0, [(0.5, 0.5)], theta_idx=0, length=10)
    _place_snake(sim, 0, 1, [(0.5, 12.5)], theta_idx=0, length=20)
    _stamp_placed_snakes(sim)

    for slot, (wx, wy), slen in ((0, (0.5, 0.5), 10), (1, (0.5, 12.5), 20)):
        xfp, yfp = _world_to_fp(wx), _world_to_fp(wy)
        ref.alive[slot] = True
        ref.life_start[slot] = True
        ref.respawn_timer[slot] = 0
        ref.length[slot] = slen
        ref.seg_count[slot] = 1
        ref.head_ptr[slot] = 0
        ref.theta[slot] = 0
        ref.head_pos[slot] = (xfp, yfp)
        ref.ring_pos[slot, 0] = (xfp, yfp)
        ref.ring_cell[slot, 0] = (
            (xfp >> FP_SHIFT) + ref.half_grid,
            (yfp >> FP_SHIFT) + ref.half_grid,
        )
        ref.ring_r_sq[slot, 0] = int(ref.tables["r_sq_cells"][slen])
    ref._rebuild_occ_mask_and_grid()

    act = torch.full((1, 6), 2, dtype=torch.int64)
    act[0, 0] = 0  # +90 deg left turn -> now facing +y
    sim.step(act)
    ref.step(act[0].numpy())

    obs = build_obs(sim)
    obs_ref = build_obs_reference(ref)
    t1_ch = obs.raster_s1[0, 0, CH_ENEMY_HEADS_T1].numpy()
    rows, cols = np.nonzero(t1_ch)
    assert len(rows) == 1
    assert (
        int(rows[0]) < 25
    ), f"Expected t-1 head ahead (row < 25), got row={rows[0]}, col={cols[0]}"
    assert 30 <= int(cols[0]) <= 33, f"Expected t-1 head centered, got row={rows[0]}, col={cols[0]}"
    assert np.array_equal(obs.raster_s1[0].numpy(), obs_ref.raster_s1)


def test_step_info_prev_length_on_death_and_survival() -> None:
    """SMALLER 6: StepInfo.prev_length preserves pre-tick length even when length=0 on death."""
    cfg = GameV3Config(arena_radius=30.0, num_snakes=6, ambient_food_density=0.0)
    sim = GameV3Sim(cfg, num_envs=1, device="cpu", seeds=[3])
    ref = ReferenceWorld(cfg, seed=3)
    _clear_world(sim)
    ref.alive.fill(False)
    ref.length.fill(0)
    ref.seg_count.fill(0)
    ref.respawn_timer.fill(10_000)
    ref.ambient_grid.fill(0)
    ref.corpse_grid.fill(0)
    ref.ambient_count = 0

    # Slot 0 (L=73) steps into the wall at x=28.5 -> dies on tick 1 (length=0, prev_length=73).
    # Slot 1 (L=25) near origin eats a 4-mass pellet -> survives (length=29, prev_length=25).
    _place_snake(sim, 0, 0, [(28.5, 0.5)], theta_idx=0, length=73)
    _place_snake(sim, 0, 1, [(-0.5, 0.5)], theta_idx=0, length=25)
    _place_food_cell(sim, 0, sim.half_grid, sim.half_grid, mass=4, corpse=False)
    _stamp_placed_snakes(sim)

    for slot, wx, slen in ((0, 28.5, 73), (1, -0.5, 25)):
        xfp, yfp = _world_to_fp(wx), _world_to_fp(0.5)
        ref.alive[slot] = True
        ref.life_start[slot] = True
        ref.respawn_timer[slot] = 0
        ref.length[slot] = slen
        ref.seg_count[slot] = 1
        ref.head_ptr[slot] = 0
        ref.theta[slot] = 0
        ref.head_pos[slot] = (xfp, yfp)
        ref.ring_pos[slot, 0] = (xfp, yfp)
        ref.ring_cell[slot, 0] = (
            (xfp >> FP_SHIFT) + ref.half_grid,
            (yfp >> FP_SHIFT) + ref.half_grid,
        )
        ref.ring_r_sq[slot, 0] = int(ref.tables["r_sq_cells"][slen])
    ref.ambient_grid[ref.half_grid, ref.half_grid] = np.uint8(4)
    ref.ambient_count = 1
    ref._rebuild_occ_mask_and_grid()

    info = sim.step(torch.full((1, 6), 2, dtype=torch.int64))
    ref_info = ref.step(np.full((6,), 2, dtype=np.int32))

    assert bool(info.died[0, 0].item())
    assert int(info.length[0, 0].item()) == 0
    assert int(info.prev_length[0, 0].item()) == 73

    assert not bool(info.died[0, 1].item())
    assert int(info.prev_length[0, 1].item()) == 25
    assert int(info.length[0, 1].item()) == 29

    assert np.array_equal(info.prev_length[0].numpy(), ref_info.prev_length)
    assert np.array_equal(info.length[0].numpy(), ref_info.length)


def test_boosting_cross_path_head_only_death_and_head_on_symmetry() -> None:
    """SMALLER 7: Boosting snake crossing ahead survives; follower hitting trailing point dies."""
    cfg = GameV3Config(
        arena_radius=80.0,
        num_snakes=6,
        boost_shed_ticks=100,
        ambient_food_density=0.0,
    )
    sim = GameV3Sim(cfg, num_envs=2, device="cpu", seeds=[10, 11], debug_assert_invariants=True)
    refs = [ReferenceWorld(cfg, seed=10), ReferenceWorld(cfg, seed=11)]
    _clear_world(sim)
    for r in refs:
        r.alive.fill(False)
        r.length.fill(0)
        r.seg_count.fill(0)
        r.respawn_timer.fill(10_000)
        r.ambient_grid.fill(0)
        r.corpse_grid.fill(0)
        r.ambient_count = 0

    quarter = cfg.angle_steps // 4  # +y
    half = cfg.angle_steps // 2  # -x

    # Env 0 (asymmetric crossing):
    # - Snake 0 (L=20, r_sq=1) boosts +x from (0.5, 0.5) -> mid-tick pt_0=(1.5, 0.5) [s=0],
    #   end-of-tick head pt_1=(2.5, 0.5) [s=1].
    # - Snake 1 (L=20, r_sq=1) moves +y at normal speed from (1.5, -2.5) -> head=(1.5, -1.5) [s=1],
    #   which overlaps Snake 0's mid-tick body pt_0 (s=0 <= 1) but NOT Snake 0's head pt_1!
    # Snake 0 MUST survive with kill_credit=1; Snake 1 MUST die with killer_id=0.
    _place_snake(sim, 0, 0, [(-0.5, 0.5), (0.5, 0.5)], theta_idx=0, length=20)
    _place_snake(sim, 0, 1, [(1.5, -3.5), (1.5, -2.5)], theta_idx=quarter, length=20)

    # Env 1 (simultaneous boost head-on):
    # Both boost toward each other along y=0.5 -> sub-step heads meet at same substep, both die.
    _place_snake(sim, 1, 0, [(-2.5, 0.5), (-1.5, 0.5)], theta_idx=0, length=20)
    _place_snake(sim, 1, 1, [(2.5, 0.5), (1.5, 0.5)], theta_idx=half, length=20)
    _stamp_placed_snakes(sim)

    env_placements = [
        [(0, [(-0.5, 0.5), (0.5, 0.5)], 0, 20), (1, [(1.5, -3.5), (1.5, -2.5)], quarter, 20)],
        [(0, [(-2.5, 0.5), (-1.5, 0.5)], 0, 20), (1, [(2.5, 0.5), (1.5, 0.5)], half, 20)],
    ]
    for e, pl in enumerate(env_placements):
        r = refs[e]
        for slot, pts, th, slen in pl:
            r.alive[slot] = True
            r.life_start[slot] = True
            r.respawn_timer[slot] = 0
            r.length[slot] = slen
            r.seg_count[slot] = len(pts)
            r.head_ptr[slot] = len(pts) - 1
            r.theta[slot] = th
            rsq = int(r.tables["r_sq_cells"][slen])
            for k, (wx, wy) in enumerate(pts):
                xfp, yfp = _world_to_fp(wx), _world_to_fp(wy)
                r.ring_pos[slot, k] = (xfp, yfp)
                r.ring_cell[slot, k] = (
                    (xfp >> FP_SHIFT) + r.half_grid,
                    (yfp >> FP_SHIFT) + r.half_grid,
                )
                r.ring_r_sq[slot, k] = rsq
            r.head_pos[slot] = r.ring_pos[slot, len(pts) - 1]
        r._rebuild_occ_mask_and_grid()

    actions = torch.full((2, 6), 2, dtype=torch.int64)
    actions[0, 0] = 7  # Env 0 slot 0: straight + boost
    actions[0, 1] = 2  # Env 0 slot 1: straight normal speed
    actions[1, 0] = 7  # Env 1 slot 0: straight + boost
    actions[1, 1] = 7  # Env 1 slot 1: straight + boost

    info = sim.step(actions)
    ref_infos = [refs[e].step(actions[e].numpy()) for e in range(2)]

    # Env 0: crossing booster survives; follower hitting trailing mid-tick segment dies.
    assert not bool(info.died[0, 0].item())
    assert bool(sim.alive[0, 0].item())
    assert bool(info.died[0, 1].item())
    assert int(info.killer_id[0, 1].item()) == 0
    assert int(info.kill_credit[0, 0].item()) == 1

    # Env 1: simultaneous head-on kills both.
    assert bool(info.died[1, 0].item()) and bool(info.died[1, 1].item())

    for e in range(2):
        for k in info.__dataclass_fields__:
            assert np.array_equal(getattr(info, k)[e].numpy(), getattr(ref_infos[e], k))


def test_ready_only_respawn_skips_work_and_n64_food_tiebreak_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SMALLER 9: Respawn evaluates ONLY ready slots; key-sort food tie-break matches N=64."""
    cfg = GameV3Config(arena_radius=120.0, num_snakes=24, respawn_delay=5, ambient_food_density=0.0)
    sim = GameV3Sim(cfg, num_envs=4, device="cpu", seeds=[70, 71, 72, 73])

    respawn_slot_rows: list[int] = []
    orig_rng_int = sim_mod.rng_int_torch

    def _spy_rng_int(
        seed: torch.Tensor | int,
        tick: torch.Tensor | int,
        purpose: int,
        slot: torch.Tensor | int,
        high: torch.Tensor | int,
    ) -> torch.Tensor:
        if purpose == PURPOSE_RESPAWN and isinstance(slot, torch.Tensor):
            respawn_slot_rows.append(int(slot.shape[0]))
        return orig_rng_int(seed, tick, purpose, slot, high)

    monkeypatch.setattr(sim_mod, "rng_int_torch", _spy_rng_int)

    # All 4 * 24 = 96 snakes alive -> 0 ready slots -> 0 PURPOSE_RESPAWN evaluations.
    sim.step(food_seeking_actions(sim))
    assert (
        respawn_slot_rows == []
    ), f"Expected 0 respawn evaluations when 0 ready, got {respawn_slot_rows}"

    # Mark 2 specific slots ready on next tick (timer=1) and 1 dead but NOT ready (timer=4).
    sim.alive[0, 3] = False
    sim.length[0, 3] = 0
    sim.seg_count[0, 3] = 0
    sim.oldest_active_subtick[0, 3] = sim.max_pts * (sim.tick + 1)
    sim.respawn_timer[0, 3] = 1

    sim.alive[2, 7] = False
    sim.length[2, 7] = 0
    sim.seg_count[2, 7] = 0
    sim.oldest_active_subtick[2, 7] = sim.max_pts * (sim.tick + 1)
    sim.respawn_timer[2, 7] = 1

    sim.alive[1, 5] = False
    sim.length[1, 5] = 0
    sim.seg_count[1, 5] = 0
    sim.oldest_active_subtick[1, 5] = sim.max_pts * (sim.tick + 1)
    sim.respawn_timer[1, 5] = 4

    info_r = sim.step(food_seeking_actions(sim))
    assert respawn_slot_rows == [
        2,
        2,
    ], f"Expected only 2 ready rows evaluated, got {respawn_slot_rows}"
    assert bool(info_r.life_start[0, 3].item()) and bool(info_r.life_start[2, 7].item())
    assert not bool(info_r.life_start[1, 5].item())

    # N=64 contested food tie-break parity against ReferenceWorld (key-sort vs pairwise oracle).
    cfg_64 = GameV3Config(arena_radius=120.0, num_snakes=64, ambient_food_density=0.0)
    sim64 = GameV3Sim(cfg_64, num_envs=1, device="cpu", seeds=[99])
    ref64 = ReferenceWorld(cfg_64, seed=99)
    _clear_world(sim64)
    ref64.alive.fill(False)
    ref64.length.fill(0)
    ref64.seg_count.fill(0)
    ref64.respawn_timer.fill(10_000)
    ref64.ambient_grid.fill(0)
    ref64.corpse_grid.fill(0)
    ref64.ambient_count = 0

    # Place 6 snakes at non-overlapping nearby positions whose L=120 pickup discs (~4.2 units)
    # overlap a shared 5x5 cluster of ambient & corpse pellets around the origin.
    contestants = [
        (5, (-3.5, 0.5), 0, 120),
        (17, (4.5, 0.5), cfg_64.angle_steps // 2, 120),
        (29, (0.5, -3.5), cfg_64.angle_steps // 4, 120),
        (41, (0.5, 4.5), (3 * cfg_64.angle_steps) // 4, 120),
        (52, (-3.5, -3.5), 0, 120),
        (63, (4.5, 4.5), cfg_64.angle_steps // 2, 120),
    ]
    for slot, (wx, wy), th, slen in contestants:
        _place_snake(sim64, 0, slot, [(wx, wy)], theta_idx=th, length=slen)
        xfp, yfp = _world_to_fp(wx), _world_to_fp(wy)
        ref64.alive[slot] = True
        ref64.life_start[slot] = True
        ref64.respawn_timer[slot] = 0
        ref64.length[slot] = slen
        ref64.seg_count[slot] = 1
        ref64.head_ptr[slot] = 0
        ref64.theta[slot] = th
        ref64.head_pos[slot] = (xfp, yfp)
        ref64.ring_pos[slot, 0] = (xfp, yfp)
        ref64.ring_cell[slot, 0] = (
            (xfp >> FP_SHIFT) + ref64.half_grid,
            (yfp >> FP_SHIFT) + ref64.half_grid,
        )
        ref64.ring_r_sq[slot, 0] = int(ref64.tables["r_sq_cells"][slen])

    c0 = sim64.half_grid
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            is_corpse = (dx + dy) % 2 == 0
            mass = 7 if is_corpse else 1
            _place_food_cell(sim64, 0, c0 + dx, c0 + dy, mass=mass, corpse=is_corpse)
            if is_corpse:
                ref64.corpse_grid[c0 + dy, c0 + dx] = np.int16(mass)
            else:
                ref64.ambient_grid[c0 + dy, c0 + dx] = np.uint8(mass)
                ref64.ambient_count += 1
    _stamp_placed_snakes(sim64)
    ref64._rebuild_occ_mask_and_grid()

    info64 = sim64.step(torch.full((1, 64), 2, dtype=torch.int64))
    ref_info64 = ref64.step(np.full((64,), 2, dtype=np.int32))
    assert int(info64.mass_eaten.sum().item()) > 0
    assert np.array_equal(info64.mass_eaten[0].numpy(), ref_info64.mass_eaten)
    assert np.array_equal(info64.length[0].numpy(), ref_info64.length)
    assert np.array_equal(sim64.ambient_grid[0].numpy(), ref64.ambient_grid)
    assert np.array_equal(sim64.corpse_grid[0].numpy(), ref64.corpse_grid)


def test_cuda_check_cli_smoke_on_cpu(tmp_path: Path) -> None:
    """SMALLER 10: cuda_check.py CLI runs parity + benchmark and writes one valid JSON result."""
    out_json = tmp_path / "cuda_check_cpu_smoke.json"
    rc = cuda_check.main(
        [
            "--device",
            "cpu",
            "--parity-envs",
            "4",
            "--parity-ticks",
            "16",
            "--parity-snakes",
            "8",
            "--parity-arena-radius",
            "80.0",
            "--bench-envs",
            "4",
            "--bench-snakes",
            "8",
            "--learners",
            "4",
            "--bench-arena-radius",
            "80.0",
            "--warmup-steps",
            "1",
            "--bench-steps",
            "3",
            "--stage-steps",
            "2",
            "--json-out",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "game-v3-cuda-check/v1"
    assert payload["parity"]["passed"] is True
    assert payload["parity"]["num_envs"] == 4
    assert payload["parity"]["num_ticks"] == 16
    assert len(payload["benchmarks"]) == 1
    assert payload["benchmarks"][0]["cases"][0]["agent_steps_per_sec"] > 0.0
