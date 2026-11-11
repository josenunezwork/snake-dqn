"""Tensorised zero-host-sync scripted opponents for Game V3."""

from __future__ import annotations

import numpy as np
import torch

from research.game_v3.config import PI_F32, TWO_PI_F32
from research.game_v3.rng import PURPOSE_SCRIPTED, rng_uniform_f32_torch
from research.game_v3.sim import GameV3Sim, poly_sincos_torch

_SAFETY_HORIZON_TICKS: tuple[float, ...] = (1.5, 4.0, 8.0, 14.0, 22.0)
_FOOD_DISTANCES: tuple[float, ...] = (6.0, 14.0, 28.0, 52.0)
_FOOD_WEIGHTS: tuple[float, ...] = (1.0, 0.65, 0.40, 0.20)


def _wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.where(
        theta >= PI_F32,
        theta - TWO_PI_F32,
        torch.where(theta < -PI_F32, theta + TWO_PI_F32, theta),
    )


def _evaluate_arc_safety(sim: GameV3Sim, *, speed_mult: float = 1.0) -> torch.Tensor:
    """Score safety of {left, straight, right} steering arcs in [0, len(horizons)].

    Probes points along constant-curvature steering arcs ``(E, N, 3, H)`` against
    both the circular boundary and active enemy occupancy cells without host syncs.
    """
    cfg = sim.cfg
    len_i64 = sim.length.clamp(0, cfg.max_length).to(torch.int64)
    turn_lim = sim._turn_table[len_i64]
    w_self = sim._w_table[len_i64]

    # Steering deltas for actions {0: +turn_lim, 1: 0, 2: -turn_lim}: shape (E, N, 3).
    dtheta = torch.stack((turn_lim, torch.zeros_like(turn_lim), -turn_lim), dim=-1)

    h_ticks = torch.tensor(_SAFETY_HORIZON_TICKS, dtype=torch.float32, device=sim.device)
    dists = h_ticks * float(np.float32(cfg.speed * speed_mult))

    # Chord midpoint angle theta + 0.5 * h * dtheta: shape (E, N, 3, H).
    mid_theta = _wrap_angle(
        sim.theta[:, :, None, None] + dtheta[:, :, :, None] * (0.5 * h_ticks[None, None, None, :])
    )
    sin_m, cos_m = poly_sincos_torch(mid_theta)

    px = sim.head_pos[:, :, 0, None, None] + cos_m * dists[None, None, None, :]
    py = sim.head_pos[:, :, 1, None, None] + sin_m * dists[None, None, None, :]

    # Boundary safety margin along arc.
    r_eff = float(np.float32(cfg.arena_radius)) - w_self[:, :, None, None] - 1.5
    r_eff_sq = r_eff.clamp_min(0.0).square()
    in_arena = (px * px + py * py) < r_eff_sq

    # Enemy body occupancy check along arc.
    inv_c = float(np.float32(1.0 / cfg.cell_size))
    gx = torch.floor(px * inv_c).to(torch.int32) + sim.half_grid
    gy = torch.floor(py * inv_c).to(torch.int32) + sim.half_grid
    in_grid = (gx >= 0) & (gx < sim.G) & (gy >= 0) & (gy < sim.G)

    gx_c = gx.clamp(0, sim.G - 1).to(torch.int64)
    gy_c = gy.clamp(0, sim.G - 1).to(torch.int64)
    flat = (gy_c * sim.G + gx_c).view(sim.E, -1)

    occ = sim.occ_packed.view(sim.E, -1).gather(1, flat).view_as(gx)
    owner = occ & 63
    owner_tail = sim.tail_seq.gather(
        1, owner.clamp(0, sim.N - 1).to(torch.int64).view(sim.E, -1)
    ).view_as(gx)
    enemy_occ = (
        (occ >= 0) & (owner != sim._snake_ids[:, :, None, None]) & ((occ >> 6) >= owner_tail)
    )

    step_safe = in_arena & in_grid & (~enemy_occ)
    # Cumulative prefix survival: a blocked step invalidates all farther steps on that arc.
    cum_safe = step_safe.to(torch.int32).cumprod(dim=-1)
    return cum_safe.to(torch.float32).sum(dim=-1)


def random_safe_actions(sim: GameV3Sim, *, stream_offset: int = 0) -> torch.Tensor:
    """Choose random safe steering actions ``(E, N)`` in ``0..5`` without host syncs."""
    safety = _evaluate_arc_safety(sim, speed_mult=1.0)

    base_slot = (
        int(stream_offset) + torch.arange(sim.N * 4, dtype=torch.int64, device=sim.device)[None, :]
    )
    draws = rng_uniform_f32_torch(
        sim.seeds[:, None], sim.tick + 1, PURPOSE_SCRIPTED, base_slot
    ).view(sim.E, sim.N, 4)

    # Safety dominates (weight 10.0 per clear horizon step); uniform [0, 1) breaks ties.
    scores = safety * 10.0 + draws[:, :, :3]
    steer = scores.argmax(dim=-1)

    max_safety = float(len(_SAFETY_HORIZON_TICKS))
    can_boost = (
        sim.alive
        & (sim.length > int(sim.cfg.min_boost_length) + 4)
        & (safety.gather(2, steer[:, :, None]).squeeze(2) >= max_safety)
        & (draws[:, :, 3] < 0.05)
    )
    actions = steer + torch.where(can_boost, torch.full_like(steer, 3), torch.zeros_like(steer))
    return torch.where(sim.alive, actions, torch.ones_like(actions))


def food_seeking_actions(sim: GameV3Sim, *, stream_offset: int = 1000) -> torch.Tensor:
    """Choose food-seeking wall/body-avoiding actions ``(E, N)`` in ``0..5`` on device."""
    cfg = sim.cfg
    safety = _evaluate_arc_safety(sim, speed_mult=1.0)

    # Probe 3 forward cones {+0.42 rad, 0.0 rad, -0.42 rad} across coarse 4x and 16x food grids.
    cone_offsets = torch.tensor([0.42, 0.0, -0.42], dtype=torch.float32, device=sim.device)
    cone_angles = _wrap_angle(sim.theta[:, :, None] + cone_offsets[None, None, :])
    sin_c, cos_c = poly_sincos_torch(cone_angles)

    f_dists = torch.tensor(_FOOD_DISTANCES, dtype=torch.float32, device=sim.device)
    f_weights = torch.tensor(_FOOD_WEIGHTS, dtype=torch.float32, device=sim.device)

    fx = sim.head_pos[:, :, 0, None, None] + cos_c[:, :, :, None] * f_dists[None, None, None, :]
    fy = sim.head_pos[:, :, 1, None, None] + sin_c[:, :, :, None] * f_dists[None, None, None, :]

    inv_c = float(np.float32(1.0 / cfg.cell_size))
    gx = torch.floor(fx * inv_c).to(torch.int32) + sim.half_grid
    gy = torch.floor(fy * inv_c).to(torch.int32) + sim.half_grid
    in_g = (gx >= 0) & (gx < sim.G) & (gy >= 0) & (gy < sim.G)

    gx_c = gx.clamp(0, sim.G - 1).to(torch.int64)
    gy_c = gy.clamp(0, sim.G - 1).to(torch.int64)

    flat4 = ((gy_c >> 2) * sim.G4 + (gx_c >> 2)).view(sim.E, -1)
    amb4 = sim.ambient_grid_4x.view(sim.E, -1).gather(1, flat4).view_as(gx).to(torch.float32)
    cor4 = sim.corpse_grid_4x.view(sim.E, -1).gather(1, flat4).view_as(gx).to(torch.float32)

    flat16 = ((gy_c >> 4) * sim.G16 + (gx_c >> 4)).view(sim.E, -1)
    amb16 = sim.ambient_grid_16x.view(sim.E, -1).gather(1, flat16).view_as(gx).to(torch.float32)
    cor16 = sim.corpse_grid_16x.view(sim.E, -1).gather(1, flat16).view_as(gx).to(torch.float32)

    local_val = amb4 + 3.0 * cor4 + 0.15 * (amb16 + 3.0 * cor16)
    local_val = torch.where(in_g, local_val, torch.zeros_like(local_val))
    food_score = (local_val * f_weights[None, None, None, :]).sum(dim=-1)

    # Gentle inward bias near outer 15% of arena radius to prevent wall hugging.
    r_lim = float(np.float32(0.85 * cfg.arena_radius))
    end_x = fx[:, :, :, -1]
    end_y = fy[:, :, :, -1]
    end_r = torch.sqrt(end_x * end_x + end_y * end_y)
    wall_pen = (end_r - r_lim).clamp_min(0.0) * 0.25

    jitter_slots = (
        int(stream_offset) + torch.arange(sim.N * 3, dtype=torch.int64, device=sim.device)[None, :]
    )
    jitter = (
        rng_uniform_f32_torch(
            sim.seeds[:, None], sim.tick + 1, PURPOSE_SCRIPTED, jitter_slots
        ).view(sim.E, sim.N, 3)
        * 1e-3
    )

    total_score = safety * 25.0 + food_score - wall_pen + jitter
    steer = total_score.argmax(dim=-1)

    # Boost straight ahead toward rich corpse spoils when path is completely clear.
    max_safety = float(len(_SAFETY_HORIZON_TICKS))
    rich_corpse_ahead = cor4[:, :, 1, :2].sum(dim=-1) >= 2.0
    boost = (
        sim.alive
        & (steer == 1)
        & (sim.length > int(cfg.min_boost_length) + 8)
        & (safety[:, :, 1] >= max_safety)
        & rich_corpse_ahead
    )
    actions = steer + torch.where(boost, torch.full_like(steer, 3), torch.zeros_like(steer))
    return torch.where(sim.alive, actions, torch.ones_like(actions))
