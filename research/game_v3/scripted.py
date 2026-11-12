"""Tensorised zero-host-sync fixed-point scripted opponents for Game V3.1 (10 actions)."""

from __future__ import annotations

import math

import torch

from research.game_v3.config import (
    FP_ONE,
    FP_SHIFT,
    NUM_TURN_LEVELS,
    decode_occ_owner_hi,
    decode_occ_owner_lo,
    decode_occ_subtick,
)
from research.game_v3.rng import PURPOSE_SCRIPTED, rng_int_torch
from research.game_v3.sim import GameV3Sim, step_delta_fp_torch

_SAFETY_HORIZONS_INT: tuple[int, ...] = (2, 4, 8, 14, 22)
_FOOD_DISTS_CELLS: tuple[int, ...] = (6, 14, 28, 52)
_FOOD_WEIGHTS_BP: tuple[int, ...] = (100, 65, 40, 20)


def _evaluate_arc_safety_int(sim: GameV3Sim) -> torch.Tensor:
    """Score integer safety [0, 5] for all 5 steering levels {0, 1, 2, 3, 4} on device.

    Uses only exact integer tables and integer squared distances (zero floats) so
    closed-loop scripted rollouts are bit-identical across CPU, MPS, and CUDA.
    """
    cfg = sim.cfg
    angle_mask = int(cfg.angle_steps - 1)
    len_i64 = sim.length.clamp(0, cfg.max_length).to(torch.int64)
    # Shape: (E, N, 5)
    dtheta = sim._turn_deltas_table[len_i64]
    spd_fp = sim._speed_fp_table[len_i64]
    rad_fp = sim._radius_fp_table[len_i64]

    h_ticks = torch.tensor(_SAFETY_HORIZONS_INT, dtype=torch.int32, device=sim.device)
    # Chord midpoint angle index: (theta + (h * dtheta) // 2) & angle_mask, shape (E, N, 5, H).
    mid_ang = (
        sim.theta[:, :, None, None] + ((dtheta[:, :, :, None] * h_ticks[None, None, None, :]) >> 1)
    ) & angle_mask
    mid_i64 = mid_ang.to(torch.int64)
    sin_m = sim._sin_table[mid_i64]
    cos_m = sim._cos_table[mid_i64]

    dists_fp = spd_fp[:, :, None, None] * h_ticks[None, None, None, :]
    px_fp = sim.head_pos[:, :, 0, None, None] + step_delta_fp_torch(dists_fp, cos_m)
    py_fp = sim.head_pos[:, :, 1, None, None] + step_delta_fp_torch(dists_fp, sin_m)

    margin_fp = int(round((1.5 / cfg.cell_size) * FP_ONE))
    r_eff_fp = (int(cfg.arena_radius_fp) - rad_fp - margin_fp).clamp_min(0).to(torch.int64)
    r_eff_sq = r_eff_fp * r_eff_fp
    d_sq = px_fp.to(torch.int64).square() + py_fp.to(torch.int64).square()
    in_arena = d_sq < r_eff_sq[:, :, None, None]

    gx = (px_fp >> FP_SHIFT) + sim.half_grid
    gy = (py_fp >> FP_SHIFT) + sim.half_grid
    in_grid = (gx >= 0) & (gx < sim.G) & (gy >= 0) & (gy < sim.G)

    gx_c = gx.clamp(0, sim.G - 1).to(torch.int64)
    gy_c = gy.clamp(0, sim.G - 1).to(torch.int64)
    flat = (gy_c * sim.G + gx_c).view(sim.E, -1)

    occ_hi = sim.occ_hi.view(sim.E, -1).gather(1, flat).view_as(gx)
    ow_hi = decode_occ_owner_hi(occ_hi)
    st_hi = sim.oldest_active_subtick.gather(
        1, ow_hi.clamp(0, sim.N - 1).to(torch.int64).view(sim.E, -1)
    ).view_as(gx)
    enemy_hi = (
        (occ_hi >= 0)
        & (ow_hi != sim._snake_ids[:, :, None, None])
        & (decode_occ_subtick(occ_hi) >= st_hi)
    )

    occ_lo = sim.occ_lo.view(sim.E, -1).gather(1, flat).view_as(gx)
    ow_lo = decode_occ_owner_lo(occ_lo)
    st_lo = sim.oldest_active_subtick.gather(
        1, ow_lo.clamp(0, sim.N - 1).to(torch.int64).view(sim.E, -1)
    ).view_as(gx)
    enemy_lo = (
        (occ_lo >= 0)
        & (ow_lo != sim._snake_ids[:, :, None, None])
        & (decode_occ_subtick(occ_lo) >= st_lo)
    )

    step_safe = in_arena & in_grid & (~(enemy_hi | enemy_lo))
    cum_safe = step_safe.to(torch.int32).cumprod(dim=-1)
    return cum_safe.sum(dim=-1)


def random_safe_actions(sim: GameV3Sim, *, stream_offset: int = 0) -> torch.Tensor:
    """Choose deterministic safe steering actions ``(E, N)`` in ``0..9`` without host syncs."""
    safety = _evaluate_arc_safety_int(sim)

    n_draws = NUM_TURN_LEVELS + 1
    base_slot = (
        int(stream_offset)
        + torch.arange(sim.N * n_draws, dtype=torch.int64, device=sim.device)[None, :]
    )
    draws = rng_int_torch(
        sim.seeds[:, None], sim.world_ticks[:, None] + 1, PURPOSE_SCRIPTED, base_slot, 1000
    ).view(sim.E, sim.N, n_draws)

    # Safety dominates (weight 10,000 per clear horizon step); uniform [0, 1000) breaks ties
    # across all 5 turn levels {-1, -0.5, 0, +0.5, +1}.
    scores = safety * 10_000 + draws[:, :, :NUM_TURN_LEVELS]
    steer = scores.argmax(dim=-1)

    max_safety = len(_SAFETY_HORIZONS_INT)
    chosen_safety = safety.gather(2, steer[:, :, None]).squeeze(2)
    can_boost = (
        sim.alive
        & (sim.length > int(sim.cfg.min_boost_length) + 4)
        & (chosen_safety >= max_safety)
        & (draws[:, :, NUM_TURN_LEVELS] < 120)
    )
    actions = steer + torch.where(
        can_boost, torch.full_like(steer, NUM_TURN_LEVELS), torch.zeros_like(steer)
    )
    return torch.where(sim.alive, actions, torch.full_like(actions, 2))


def food_seeking_actions(sim: GameV3Sim, *, stream_offset: int = 1000) -> torch.Tensor:
    """Choose food-seeking wall/body-avoiding actions ``(E, N)`` in ``0..9`` on device."""
    cfg = sim.cfg
    angle_mask = int(cfg.angle_steps - 1)
    safety = _evaluate_arc_safety_int(sim)

    # Probe 5 forward cones {+0.44, +0.22, 0.0, -0.22, -0.44} rad matching the 5 turn levels.
    full_cone = max(2, int(round(0.44 * cfg.angle_steps / (2.0 * math.pi))))
    half_cone = max(1, full_cone // 2)
    cone_steps = torch.tensor(
        [full_cone, half_cone, 0, -half_cone, -full_cone],
        dtype=torch.int32,
        device=sim.device,
    )
    cone_ang = (sim.theta[:, :, None] + cone_steps[None, None, :]) & angle_mask
    cone_i64 = cone_ang.to(torch.int64)
    sin_c = sim._sin_table[cone_i64]
    cos_c = sim._cos_table[cone_i64]

    cell_fp = max(1, int(round((1.0 / cfg.cell_size) * FP_ONE)))
    f_dists_fp = torch.tensor(_FOOD_DISTS_CELLS, dtype=torch.int32, device=sim.device) * cell_fp
    f_weights = torch.tensor(_FOOD_WEIGHTS_BP, dtype=torch.int32, device=sim.device)

    fx_fp = sim.head_pos[:, :, 0, None, None] + step_delta_fp_torch(
        f_dists_fp[None, None, None, :], cos_c[:, :, :, None]
    )
    fy_fp = sim.head_pos[:, :, 1, None, None] + step_delta_fp_torch(
        f_dists_fp[None, None, None, :], sin_c[:, :, :, None]
    )

    gx = (fx_fp >> FP_SHIFT) + sim.half_grid
    gy = (fy_fp >> FP_SHIFT) + sim.half_grid
    in_g = (gx >= 0) & (gx < sim.G) & (gy >= 0) & (gy < sim.G)

    gx_c = gx.clamp(0, sim.G - 1).to(torch.int64)
    gy_c = gy.clamp(0, sim.G - 1).to(torch.int64)

    flat4 = ((gy_c >> 2) * sim.G4 + (gx_c >> 2)).view(sim.E, -1)
    amb4 = sim.ambient_grid_4x.view(sim.E, -1).gather(1, flat4).view_as(gx)
    cor4 = sim.corpse_grid_4x.view(sim.E, -1).gather(1, flat4).view_as(gx)

    flat16 = ((gy_c >> 4) * sim.G16 + (gx_c >> 4)).view(sim.E, -1)
    amb16 = sim.ambient_grid_16x.view(sim.E, -1).gather(1, flat16).view_as(gx)
    cor16 = sim.corpse_grid_16x.view(sim.E, -1).gather(1, flat16).view_as(gx)

    # Integer food value in centi-units: 100*(amb4 + 3*cor4) + 15*(amb16 + 3*cor16).
    local_val = 100 * (amb4 + 3 * cor4) + 15 * (amb16 + 3 * cor16)
    local_val = torch.where(in_g, local_val.clamp_max(50_000), torch.zeros_like(local_val))
    food_score = (local_val.to(torch.int64) * f_weights[None, None, None, :].to(torch.int64)).sum(
        dim=-1
    )

    # Gentle inward bias near outer 15% of arena radius using squared distance.
    r85_fp = int(round(0.85 * cfg.arena_radius_fp))
    r85_sq = int(r85_fp) * int(r85_fp)
    end_x = fx_fp[:, :, :, -1].to(torch.int64)
    end_y = fy_fp[:, :, :, -1].to(torch.int64)
    over_sq = (end_x * end_x + end_y * end_y - r85_sq).clamp_min(0)
    wall_pen = over_sq >> 14

    jitter_slots = (
        int(stream_offset)
        + torch.arange(sim.N * NUM_TURN_LEVELS, dtype=torch.int64, device=sim.device)[None, :]
    )
    jitter = rng_int_torch(
        sim.seeds[:, None], sim.world_ticks[:, None] + 1, PURPOSE_SCRIPTED, jitter_slots, 64
    ).view(sim.E, sim.N, NUM_TURN_LEVELS)

    total_score = safety.to(torch.int64) * 2_500_000 + food_score * 10 - wall_pen + jitter
    steer = total_score.argmax(dim=-1)

    max_safety = len(_SAFETY_HORIZONS_INT)
    chosen_safety = safety.gather(2, steer[:, :, None]).squeeze(2)
    chosen_cor4 = cor4[:, :, :, :2].sum(dim=-1).gather(2, steer[:, :, None]).squeeze(2)
    boost = (
        sim.alive
        & (sim.length > int(cfg.min_boost_length) + 6)
        & (chosen_safety >= max_safety)
        & (chosen_cor4 >= 2)
    )
    actions = steer + torch.where(
        boost, torch.full_like(steer, NUM_TURN_LEVELS), torch.zeros_like(steer)
    )
    return torch.where(sim.alive, actions, torch.full_like(actions, 2))


_CARDINAL6_ACTION_MAP: tuple[int, ...] = (0, 2, 4, 5, 7, 9)


def cardinal6_safe_actions(sim: GameV3Sim, *, stream_offset: int = 2000) -> torch.Tensor:
    """Choose deterministic 6-cardinal actions ``{L, S, R} x {normal, boost}`` on device.

    Scores the classic 6-action subset ``[0, 2, 4, 5, 7, 9]`` in an ``(E, N, 6)`` integer
    tensor and selects via ``argmax(dim=-1)`` with zero host syncs.
    """
    safety5 = _evaluate_arc_safety_int(sim)
    cardinal_idx = torch.tensor([0, 2, 4], dtype=torch.int64, device=sim.device)
    safety3 = safety5.index_select(2, cardinal_idx)

    n_cardinal = 6
    base_slot = (
        int(stream_offset)
        + torch.arange(sim.N * n_cardinal, dtype=torch.int64, device=sim.device)[None, :]
    )
    draws6 = rng_int_torch(
        sim.seeds[:, None], sim.world_ticks[:, None] + 1, PURPOSE_SCRIPTED, base_slot, 1000
    ).view(sim.E, sim.N, n_cardinal)

    max_safety = len(_SAFETY_HORIZONS_INT)
    boost_ok = (
        sim.alive[:, :, None]
        & (sim.length[:, :, None] > int(sim.cfg.min_boost_length) + 4)
        & (safety3 >= max_safety)
    )
    normal_scores = safety3.to(torch.int64) * 10_000 + draws6[:, :, :3]
    boost_raw = safety3.to(torch.int64) * 10_000 + draws6[:, :, 3:] + 50
    boost_scores = torch.where(boost_ok, boost_raw, torch.full_like(boost_raw, -1_000_000))
    scores6 = torch.cat((normal_scores, boost_scores), dim=-1)
    choice6 = scores6.argmax(dim=-1)
    action_map = torch.tensor(_CARDINAL6_ACTION_MAP, dtype=torch.int64, device=sim.device)
    actions = action_map[choice6]
    return torch.where(sim.alive, actions, torch.full_like(actions, 2))
