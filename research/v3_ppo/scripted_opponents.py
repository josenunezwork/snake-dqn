"""Stronger tensorised fixed-point scripted opponents for Game V3.1 (`research/v3_ppo`).

Built on the public helpers of :mod:`research.game_v3.scripted` (and public fixed-point tables in
:mod:`research.game_v3.config`, :mod:`research.game_v3.rng`, and :mod:`research.game_v3.sim`), with
zero per-snake Python loops and zero device-to-host synchronizations:

- ``head_evader`` (:func:`head_evader_actions`): Integrates curved multi-horizon normal/boost
  trajectories with 3-point head-disc body/wall safety, obstacle-gated food/corpse valuation, and
  pairwise enemy-head ``k``-tick reachability-cone avoidance before seeking food.
- ``cut_off`` (:func:`cut_off_actions`): Stalks nearby enemy heads onto a flanking shoulder offset
  and boosts across their projected forward path whenever its boost arc crosses ``3..13`` cells
  ahead of an enemy nose with safe post-crossing clearance, while avoiding losing head-ons and
  harvesting dropped corpses.
"""

from __future__ import annotations

from typing import Any

import torch

from research.game_v3.config import (
    TRIG_SHIFT,
    GameV3Config,
    build_length_tables_np,
    build_trig_tables_np,
)
from research.game_v3.scripted import (
    FP_ONE,
    FP_SHIFT,
    NUM_TURN_LEVELS,
    PURPOSE_SCRIPTED,
    cardinal6_safe_actions,
    decode_occ_owner_hi,
    decode_occ_owner_lo,
    decode_occ_subtick,
    food_seeking_actions,
    random_safe_actions,
    rng_int_torch,
    step_delta_fp_torch,
)
from research.game_v3.sim import GameV3Sim

__all__ = [
    "cardinal6_safe_actions",
    "cut_off_actions",
    "food_seeking_actions",
    "head_evader_actions",
    "random_safe_actions",
]

# Incremental segment durations (ticks); cumulative horizons = (2, 4, 7, 11, 16, 22).
_ARC_SEGMENT_TICKS: tuple[int, ...] = (2, 2, 3, 4, 5, 6)
_ARC_HORIZONS: tuple[int, ...] = (2, 4, 7, 11, 16, 22)
_THREAT_HORIZONS: tuple[int, ...] = (2, 4, 7, 11, 16)
_STRIKE_HORIZONS: tuple[int, ...] = (2, 4, 7, 11)
_FAR_FOOD_EXTRA_CELLS: tuple[int, ...] = (12, 30)
_FOOD_SAMPLE_WEIGHTS: tuple[int, ...] = (110, 95, 80, 65, 50, 38, 26, 16)
_MAX_NEAREST_ENEMIES: int = 6
_INF_DIST_SQ: int = 1 << 60

_STATIC_TABLES_CACHE: dict[tuple[GameV3Config, torch.device, int], dict[str, torch.Tensor]] = {}


def _get_static_tables(sim: GameV3Sim) -> dict[str, torch.Tensor]:
    """Return cached public integer trig/length lookup tensors on ``sim.device``."""
    key = (sim.cfg, sim.device, int(sim.N))
    cached = _STATIC_TABLES_CACHE.get(key)
    if cached is not None:
        return cached

    sin_np, cos_np = build_trig_tables_np(sim.cfg.angle_steps)
    lt_np = build_length_tables_np(sim.cfg)
    n_snakes = int(sim.N)
    tables: dict[str, torch.Tensor] = {
        "sin_table": torch.as_tensor(sin_np, dtype=torch.int32, device=sim.device),
        "cos_table": torch.as_tensor(cos_np, dtype=torch.int32, device=sim.device),
        "speed_fp": torch.as_tensor(lt_np["speed_fp"], dtype=torch.int32, device=sim.device),
        "boost_speed_fp": torch.as_tensor(
            lt_np["boost_speed_fp"], dtype=torch.int32, device=sim.device
        ),
        "turn_deltas": torch.as_tensor(lt_np["turn_deltas"], dtype=torch.int32, device=sim.device),
        "radius_fp": torch.as_tensor(lt_np["radius_fp"], dtype=torch.int32, device=sim.device),
        "snake_ids": torch.arange(n_snakes, dtype=torch.int32, device=sim.device)[None, :],
        "eye_mask": torch.eye(n_snakes, dtype=torch.bool, device=sim.device)[None, :, :],
        "food_weights": torch.tensor(_FOOD_SAMPLE_WEIGHTS, dtype=torch.int64, device=sim.device),
        "threat_horizons": torch.tensor(_THREAT_HORIZONS, dtype=torch.int64, device=sim.device),
        "threat_cone_weights": torch.tensor(
            (3_400_000, 2_000_000, 950_000, 420_000, 160_000),
            dtype=torch.int64,
            device=sim.device,
        ),
        "threat_direct_weights": torch.tensor(
            (3_200_000, 1_800_000, 750_000, 200_000, 0),
            dtype=torch.int64,
            device=sim.device,
        ),
        "strike_horizons": torch.tensor(_STRIKE_HORIZONS, dtype=torch.int64, device=sim.device),
        "strike_weights": torch.tensor(
            (2_900_000, 2_500_000, 2_000_000, 1_450_000),
            dtype=torch.int64,
            device=sim.device,
        ),
    }
    if len(_STATIC_TABLES_CACHE) >= 32:
        _STATIC_TABLES_CACHE.clear()
    _STATIC_TABLES_CACHE[key] = tables
    return tables


def _rollout_curved_arcs(
    sim: GameV3Sim,
    tables: dict[str, torch.Tensor],
    *,
    boost: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate 5 curved steering arcs across 6 horizons with center + 3-point disc safety."""
    cfg = sim.cfg
    dev = sim.device
    e_cnt = int(sim.E)
    n_cnt = int(sim.N)
    h_cnt = len(_ARC_SEGMENT_TICKS)
    angle_mask = int(cfg.angle_steps - 1)

    len_i64 = sim.length.clamp(0, cfg.max_length).to(torch.int64)
    dtheta = tables["turn_deltas"][len_i64]  # (E, N, 5)
    spd_tbl = tables["boost_speed_fp"] if boost else tables["speed_fp"]
    spd_fp = spd_tbl[len_i64]  # (E, N)
    rad_fp = tables["radius_fp"][len_i64]  # (E, N)
    sin_tbl = tables["sin_table"]
    cos_tbl = tables["cos_table"]

    cur_theta = sim.theta[:, :, None].expand(e_cnt, n_cnt, NUM_TURN_LEVELS).clone()
    cur_x = sim.head_pos[:, :, 0, None].expand(e_cnt, n_cnt, NUM_TURN_LEVELS).clone()
    cur_y = sim.head_pos[:, :, 1, None].expand(e_cnt, n_cnt, NUM_TURN_LEVELS).clone()

    px_all = torch.empty((e_cnt, n_cnt, NUM_TURN_LEVELS, h_cnt), dtype=torch.int32, device=dev)
    py_all = torch.empty((e_cnt, n_cnt, NUM_TURN_LEVELS, h_cnt), dtype=torch.int32, device=dev)
    th_all = torch.empty((e_cnt, n_cnt, NUM_TURN_LEVELS, h_cnt), dtype=torch.int32, device=dev)

    for k_step, dh in enumerate(_ARC_SEGMENT_TICKS):
        mid_ang = (cur_theta + ((dtheta * dh) >> 1)) & angle_mask
        cur_theta = (cur_theta + dtheta * dh) & angle_mask
        mid_i64 = mid_ang.to(torch.int64)
        seg_dist = spd_fp[:, :, None] * dh
        cur_x = cur_x + step_delta_fp_torch(seg_dist, cos_tbl[mid_i64])
        cur_y = cur_y + step_delta_fp_torch(seg_dist, sin_tbl[mid_i64])
        px_all[:, :, :, k_step] = cur_x
        py_all[:, :, :, k_step] = cur_y
        th_all[:, :, :, k_step] = cur_theta

    margin_fp = int(round((1.75 / cfg.cell_size) * FP_ONE))
    r_eff_fp = (int(cfg.arena_radius_fp) - rad_fp - margin_fp).clamp_min(0).to(torch.int64)
    r_eff_sq = r_eff_fp * r_eff_fp
    d_sq = px_all.to(torch.int64).square() + py_all.to(torch.int64).square()
    in_arena = d_sq < r_eff_sq[:, :, None, None]

    th_i64 = th_all.to(torch.int64)
    nx_q14 = -sin_tbl[th_i64]
    ny_q14 = cos_tbl[th_i64]
    lat_fp = rad_fp[:, :, None, None]
    off_x = step_delta_fp_torch(lat_fp, nx_q14)
    off_y = step_delta_fp_torch(lat_fp, ny_q14)

    probe_x = torch.stack((px_all, px_all + off_x, px_all - off_x), dim=-1)  # (E, N, 5, H, 3)
    probe_y = torch.stack((py_all, py_all + off_y, py_all - off_y), dim=-1)

    gx = (probe_x >> FP_SHIFT) + sim.half_grid
    gy = (probe_y >> FP_SHIFT) + sim.half_grid
    in_grid = (gx >= 0) & (gx < sim.G) & (gy >= 0) & (gy < sim.G)

    gx_c = gx.clamp(0, sim.G - 1).to(torch.int64)
    gy_c = gy.clamp(0, sim.G - 1).to(torch.int64)
    flat = (gy_c * sim.G + gx_c).view(e_cnt, -1)

    self_ids = tables["snake_ids"][:, :, None, None, None]
    occ_hi = sim.occ_hi.view(e_cnt, -1).gather(1, flat).view_as(gx)
    ow_hi = decode_occ_owner_hi(occ_hi)
    st_hi = sim.oldest_active_subtick.gather(
        1, ow_hi.clamp(0, n_cnt - 1).to(torch.int64).view(e_cnt, -1)
    ).view_as(gx)
    enemy_hi = (occ_hi >= 0) & (ow_hi != self_ids) & (decode_occ_subtick(occ_hi) >= st_hi)

    occ_lo = sim.occ_lo.view(e_cnt, -1).gather(1, flat).view_as(gx)
    ow_lo = decode_occ_owner_lo(occ_lo)
    st_lo = sim.oldest_active_subtick.gather(
        1, ow_lo.clamp(0, n_cnt - 1).to(torch.int64).view(e_cnt, -1)
    ).view_as(gx)
    enemy_lo = (occ_lo >= 0) & (ow_lo != self_ids) & (decode_occ_subtick(occ_lo) >= st_lo)

    pt_clear = in_grid & (~(enemy_hi | enemy_lo))  # (E, N, 5, H, 3)
    center_step_safe = in_arena & pt_clear[:, :, :, :, 0]
    disc_step_safe = in_arena & pt_clear.all(dim=-1)

    cum_center_safe = center_step_safe.to(torch.int32).cumprod(dim=-1)  # (E, N, 5, H)
    cum_disc_safe = disc_step_safe.to(torch.int32).cumprod(dim=-1)
    center_safety = cum_center_safe.sum(dim=-1)  # (E, N, 5)
    disc_safety = cum_disc_safe.sum(dim=-1)  # (E, N, 5)
    return px_all, py_all, th_all, cum_center_safe, center_safety, disc_safety


def _score_reachable_food(
    sim: GameV3Sim,
    px_norm: torch.Tensor,
    py_norm: torch.Tensor,
    th_norm: torch.Tensor,
    cum_center_safe: torch.Tensor,
    tables: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score obstacle-gated ambient + corpse food along each curved steering arc."""
    cfg = sim.cfg
    e_cnt = int(sim.E)
    cell_fp = max(1, int(round((1.0 / cfg.cell_size) * FP_ONE)))
    end_th_i64 = th_norm[:, :, :, -1].to(torch.int64)
    cos_end = tables["cos_table"][end_th_i64]
    sin_end = tables["sin_table"][end_th_i64]

    d1_fp = _FAR_FOOD_EXTRA_CELLS[0] * cell_fp
    d2_fp = _FAR_FOOD_EXTRA_CELLS[1] * cell_fp
    far1_x = px_norm[:, :, :, -1] + step_delta_fp_torch(d1_fp, cos_end)
    far1_y = py_norm[:, :, :, -1] + step_delta_fp_torch(d1_fp, sin_end)
    far2_x = px_norm[:, :, :, -1] + step_delta_fp_torch(d2_fp, cos_end)
    far2_y = py_norm[:, :, :, -1] + step_delta_fp_torch(d2_fp, sin_end)

    fx_fp = torch.cat((px_norm, far1_x.unsqueeze(-1), far2_x.unsqueeze(-1)), dim=-1)  # (E,N,5,8)
    fy_fp = torch.cat((py_norm, far1_y.unsqueeze(-1), far2_y.unsqueeze(-1)), dim=-1)
    far_gate = cum_center_safe[:, :, :, -1:]
    reach_gate = torch.cat((cum_center_safe, far_gate, far_gate), dim=-1).to(torch.int64)

    gx = (fx_fp >> FP_SHIFT) + sim.half_grid
    gy = (fy_fp >> FP_SHIFT) + sim.half_grid
    in_g = (gx >= 0) & (gx < sim.G) & (gy >= 0) & (gy < sim.G)

    gx_c = gx.clamp(0, sim.G - 1).to(torch.int64)
    gy_c = gy.clamp(0, sim.G - 1).to(torch.int64)

    flat4 = ((gy_c >> 2) * sim.G4 + (gx_c >> 2)).view(e_cnt, -1)
    amb4 = sim.ambient_grid_4x.view(e_cnt, -1).gather(1, flat4).view_as(gx)
    cor4 = sim.corpse_grid_4x.view(e_cnt, -1).gather(1, flat4).view_as(gx)

    flat16 = ((gy_c >> 4) * sim.G16 + (gx_c >> 4)).view(e_cnt, -1)
    amb16 = sim.ambient_grid_16x.view(e_cnt, -1).gather(1, flat16).view_as(gx)
    cor16 = sim.corpse_grid_16x.view(e_cnt, -1).gather(1, flat16).view_as(gx)

    local_val = 100 * (amb4 + 4 * cor4) + 18 * (amb16 + 4 * cor16)
    local_val = torch.where(in_g, local_val.clamp_max(60_000), torch.zeros_like(local_val))
    gated_val = local_val.to(torch.int64) * reach_gate
    food_score = (gated_val * tables["food_weights"][None, None, None, :]).sum(dim=-1)
    near_cor4 = (cor4[:, :, :, :4].to(torch.int64) * reach_gate[:, :, :, :4]).sum(dim=-1)

    r84_fp = int(round(0.84 * cfg.arena_radius_fp))
    r84_sq = int(r84_fp) * int(r84_fp)
    end_x = px_norm[:, :, :, -1].to(torch.int64)
    end_y = py_norm[:, :, :, -1].to(torch.int64)
    wall_pen = (end_x * end_x + end_y * end_y - r84_sq).clamp_min(0) >> 13
    return food_score, near_cor4, wall_pen


def _gather_nearest_enemies(
    sim: GameV3Sim,
    tables: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Gather state of the ``K`` nearest alive enemy heads per snake ``(E, N, K)``."""
    cfg = sim.cfg
    n_cnt = int(sim.N)
    k_near = min(_MAX_NEAREST_ENEMIES, max(1, n_cnt - 1))

    hx = sim.head_pos[:, :, 0].to(torch.int64)  # (E, N)
    hy = sim.head_pos[:, :, 1].to(torch.int64)
    dx_ij = hx[:, :, None] - hx[:, None, :]  # (E, N, N)
    dy_ij = hy[:, :, None] - hy[:, None, :]
    d2_ij = dx_ij * dx_ij + dy_ij * dy_ij

    valid_enemy_ij = sim.alive[:, None, :] & (~tables["eye_mask"])
    inf_d2 = torch.full_like(d2_ij, _INF_DIST_SQ)
    d2_masked = torch.where(valid_enemy_ij, d2_ij, inf_d2)
    nn_d2, nn_idx = torch.topk(d2_masked, k=k_near, dim=-1, largest=False)
    nn_valid = nn_d2 < _INF_DIST_SQ  # (E, N, K)

    th_i64 = sim.theta.to(torch.int64)
    ux = tables["cos_table"][th_i64].to(torch.int64)
    uy = tables["sin_table"][th_i64].to(torch.int64)
    len_clamped = sim.length.clamp(0, cfg.max_length).to(torch.int64)
    spd = tables["speed_fp"][len_clamped].to(torch.int64)
    bst = tables["boost_speed_fp"][len_clamped].to(torch.int64)
    rad = tables["radius_fp"][len_clamped].to(torch.int64)

    def _gather_k(vec_en: torch.Tensor) -> torch.Tensor:
        return vec_en[:, None, :].expand(-1, n_cnt, -1).gather(2, nn_idx)

    return {
        "k_near": k_near,
        "nn_d2": nn_d2,
        "nn_valid": nn_valid,
        "hx_i": hx,
        "hy_i": hy,
        "ux_i": ux,
        "uy_i": uy,
        "rad_i": rad,
        "hx_k": _gather_k(hx),
        "hy_k": _gather_k(hy),
        "ux_k": _gather_k(ux),
        "uy_k": _gather_k(uy),
        "spd_k": _gather_k(spd),
        "bst_k": _gather_k(bst),
        "rad_k": _gather_k(rad),
    }


def _compute_head_reach_threat(
    px_arcs: torch.Tensor,
    py_arcs: torch.Tensor,
    nn: dict[str, Any],
    tables: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute enemy-head ``k``-tick reachability cone & direct-collision threat per steer."""
    m_cnt = len(_THREAT_HORIZONS)
    px_m = px_arcs[:, :, :, :m_cnt].to(torch.int64).unsqueeze(-1)  # (E, N, 5, M, 1)
    py_m = py_arcs[:, :, :, :m_cnt].to(torch.int64).unsqueeze(-1)

    hx_k = nn["hx_k"][:, :, None, None, :]  # (E, N, 1, 1, K)
    hy_k = nn["hy_k"][:, :, None, None, :]
    ux_k = nn["ux_k"][:, :, None, None, :]
    uy_k = nn["uy_k"][:, :, None, None, :]
    spd_k = nn["spd_k"][:, :, None, None, :]
    bst_k = nn["bst_k"][:, :, None, None, :]
    rad_k = nn["rad_k"][:, :, None, None, :]
    rad_i = nn["rad_i"][:, :, None, None, None]
    valid_k = nn["nn_valid"][:, :, None, None, :]
    taus = tables["threat_horizons"][None, None, None, :, None]  # (1, 1, 1, M, 1)

    rx = px_m - hx_k
    ry = py_m - hy_k
    proj_fwd = (rx * ux_k + ry * uy_k) >> TRIG_SHIFT

    # Forward-shifted boost reachability cone at horizon tau.
    shift_fp = (taus * spd_k) >> 1
    cx = hx_k + ((shift_fp * ux_k) >> TRIG_SHIFT)
    cy = hy_k + ((shift_fp * uy_k) >> TRIG_SHIFT)
    dx_c = px_m - cx
    dy_c = py_m - cy
    cone_r = ((taus * bst_k * 3) >> 2) + rad_i + rad_k + (3 * FP_ONE // 2)
    in_cone = valid_k & (proj_fwd >= -2 * FP_ONE) & (dx_c * dx_c + dy_c * dy_c <= cone_r * cone_r)

    # Direct straight-line enemy head projection (both normal and boost speed).
    ex_norm = hx_k + ((taus * spd_k * ux_k) >> TRIG_SHIFT)
    ey_norm = hy_k + ((taus * spd_k * uy_k) >> TRIG_SHIFT)
    ex_bst = hx_k + ((taus * bst_k * ux_k) >> TRIG_SHIFT)
    ey_bst = hy_k + ((taus * bst_k * uy_k) >> TRIG_SHIFT)
    direct_r = rad_i + rad_k + (5 * FP_ONE // 2)
    direct_r_sq = direct_r * direct_r
    in_direct = valid_k & (
        ((px_m - ex_norm).square() + (py_m - ey_norm).square() <= direct_r_sq)
        | ((px_m - ex_bst).square() + (py_m - ey_bst).square() <= direct_r_sq)
    )

    cone_hits_m = in_cone.to(torch.int64).sum(dim=-1)  # (E, N, 5, M)
    direct_hits_m = in_direct.to(torch.int64).sum(dim=-1)
    cone_w = tables["threat_cone_weights"][None, None, None, :]
    direct_w = tables["threat_direct_weights"][None, None, None, :]
    threat_pen = (cone_hits_m * cone_w + direct_hits_m * direct_w).sum(dim=-1)  # (E, N, 5)
    near_hits = (cone_hits_m[:, :, :, :2] + direct_hits_m[:, :, :, :2]).sum(dim=-1)
    return threat_pen, near_hits


def head_evader_actions(sim: GameV3Sim, *, stream_offset: int = 3000) -> torch.Tensor:
    """Avoid arcs reachable by any enemy head within ``k`` ticks, then harvest food/corpses."""
    if int(sim.N) <= 1:
        return food_seeking_actions(sim, stream_offset=stream_offset)

    cfg = sim.cfg
    tables = _get_static_tables(sim)
    px_n, py_n, th_n, cum_c_n, c_safe_n, d_safe_n = _rollout_curved_arcs(sim, tables, boost=False)
    px_b, py_b, _th_b, _cum_c_b, c_safe_b, d_safe_b = _rollout_curved_arcs(sim, tables, boost=True)
    food_score, near_cor4, wall_pen = _score_reachable_food(sim, px_n, py_n, th_n, cum_c_n, tables)
    nn = _gather_nearest_enemies(sim, tables)
    threat_norm, near_hits_norm = _compute_head_reach_threat(px_n, py_n, nn, tables)
    threat_bst, near_hits_bst = _compute_head_reach_threat(px_b, py_b, nn, tables)

    jitter_slots = (
        int(stream_offset)
        + torch.arange(sim.N * NUM_TURN_LEVELS, dtype=torch.int64, device=sim.device)[None, :]
    )
    jitter = rng_int_torch(
        sim.seeds[:, None], sim.world_ticks[:, None] + 1, PURPOSE_SCRIPTED, jitter_slots, 64
    ).view(sim.E, sim.N, NUM_TURN_LEVELS)

    # Immediate (t=2, 4) body/wall survival takes absolute priority over everything; beyond that,
    # enemy-head k-tick reachability cones strongly steer away from contested space before food.
    imm_safe_n = cum_c_n[:, :, :, 0].to(torch.int64) * 40_000_000 + (
        cum_c_n[:, :, :, 1].to(torch.int64) * 20_000_000
    )
    arc_safe_score = (
        imm_safe_n + c_safe_n.to(torch.int64) * 4_200_000 + d_safe_n.to(torch.int64) * 1_900_000
    )
    total_score = arc_safe_score - threat_norm + food_score * 12 - wall_pen + jitter
    steer = total_score.argmax(dim=-1)

    max_safety = len(_ARC_HORIZONS)
    chosen_c_bst = c_safe_b.gather(2, steer[:, :, None]).squeeze(2)
    chosen_d_bst = d_safe_b.gather(2, steer[:, :, None]).squeeze(2)
    chosen_near_bst = near_hits_bst.gather(2, steer[:, :, None]).squeeze(2)
    chosen_near_norm = near_hits_norm.gather(2, steer[:, :, None]).squeeze(2)
    chosen_threat_bst = threat_bst.gather(2, steer[:, :, None]).squeeze(2)
    chosen_threat_norm = threat_norm.gather(2, steer[:, :, None]).squeeze(2)
    chosen_cor4 = near_cor4.gather(2, steer[:, :, None]).squeeze(2)

    corpse_sprint = (
        (sim.length > int(cfg.min_boost_length) + 6)
        & (chosen_c_bst >= max_safety)
        & (chosen_d_bst >= max_safety - 1)
        & (chosen_near_bst == 0)
        & (chosen_cor4 >= 2)
    )
    close_enemy = nn["nn_valid"][:, :, 0] & (nn["nn_d2"][:, :, 0] <= (13 * FP_ONE) ** 2)
    escape_boost = (
        close_enemy
        & (sim.length > int(cfg.min_boost_length) + 3)
        & (chosen_c_bst >= max_safety - 1)
        & (chosen_d_bst >= max_safety - 2)
        & (
            (chosen_near_bst < chosen_near_norm)
            | (chosen_threat_bst + 800_000 < chosen_threat_norm)
        )
    )
    boost = sim.alive & (corpse_sprint | escape_boost)
    actions = steer + torch.where(
        boost, torch.full_like(steer, NUM_TURN_LEVELS), torch.zeros_like(steer)
    )
    return torch.where(sim.alive, actions, torch.full_like(actions, 2))


def cut_off_actions(sim: GameV3Sim, *, stream_offset: int = 4000) -> torch.Tensor:
    """Stalk nearby enemy heads onto a flanking line and boost across their path to cut them off."""
    if int(sim.N) <= 1:
        return food_seeking_actions(sim, stream_offset=stream_offset)

    cfg = sim.cfg
    tables = _get_static_tables(sim)
    px_n, py_n, th_n, cum_c_n, c_safe_n, d_safe_n = _rollout_curved_arcs(sim, tables, boost=False)
    px_b, py_b, _th_b, cum_c_b, c_safe_b, d_safe_b = _rollout_curved_arcs(sim, tables, boost=True)
    food_score, near_cor4, wall_pen = _score_reachable_food(sim, px_n, py_n, th_n, cum_c_n, tables)
    nn = _gather_nearest_enemies(sim, tables)
    threat_norm, _ = _compute_head_reach_threat(px_n, py_n, nn, tables)
    threat_bst, near_hits_bst = _compute_head_reach_threat(px_b, py_b, nn, tables)

    # Evaluate bow-crossing cut-off strikes across the 4 boost horizons tau = (2, 4, 7, 11).
    s_cnt = len(_STRIKE_HORIZONS)
    px_bs = px_b[:, :, :, :s_cnt].to(torch.int64).unsqueeze(-1)  # (E, N, 5, S, 1)
    py_bs = py_b[:, :, :, :s_cnt].to(torch.int64).unsqueeze(-1)

    hx_i = nn["hx_i"][:, :, None]  # (E, N, 1)
    hy_i = nn["hy_i"][:, :, None]
    ux_i = nn["ux_i"][:, :, None]
    uy_i = nn["uy_i"][:, :, None]
    hx_k = nn["hx_k"]  # (E, N, K)
    hy_k = nn["hy_k"]
    ux_k = nn["ux_k"]
    uy_k = nn["uy_k"]
    spd_k = nn["spd_k"]
    rad_k = nn["rad_k"]
    rad_i = nn["rad_i"][:, :, None]

    cur_dx = hx_i - hx_k  # (E, N, K): our head in enemy k's frame
    cur_dy = hy_i - hy_k
    cur_s = (cur_dx * ux_k + cur_dy * uy_k) >> TRIG_SHIFT
    cur_lat = (-cur_dx * uy_k + cur_dy * ux_k) >> TRIG_SHIFT
    heading_dot = (ux_i * ux_k + uy_i * uy_k) >> TRIG_SHIFT

    hunt_range = nn["nn_valid"] & (nn["nn_d2"] <= (46 * FP_ONE) ** 2)
    not_narrow_head_on = ~((cur_s > 0) & (cur_lat.abs() < 3 * FP_ONE) & (heading_dot < -5500))
    eligible_k = hunt_range & not_narrow_head_on  # (E, N, K)

    rx_b = px_bs - hx_k[:, :, None, None, :]
    ry_b = py_bs - hy_k[:, :, None, None, :]
    s_along = (rx_b * ux_k[:, :, None, None, :] + ry_b * uy_k[:, :, None, None, :]) >> TRIG_SHIFT
    lat_abs = (
        (-rx_b * uy_k[:, :, None, None, :] + ry_b * ux_k[:, :, None, None, :]) >> TRIG_SHIFT
    ).abs()

    s_taus = tables["strike_horizons"][None, None, None, :, None]  # (1, 1, 1, S, 1)
    d_enemy = s_taus * spd_k[:, :, None, None, :]
    lead_fp = s_along - d_enemy
    sum_rad = rad_i[:, :, None, None, :] + rad_k[:, :, None, None, :]
    min_lead_fp = sum_rad + (5 * FP_ONE // 2)
    max_lead_fp = sum_rad + 13 * FP_ONE
    max_lat_fp = 4 * FP_ONE + rad_i[:, :, None, None, :]

    m_idx = torch.arange(s_cnt, dtype=torch.int32, device=sim.device)[None, None, None, :, None]
    safe_after_cross = (
        (cum_c_b[:, :, :, :s_cnt, None] == 1)
        & (c_safe_b[:, :, :, None, None] >= m_idx + 2)
        & (d_safe_b[:, :, :, None, None] >= m_idx + 1)
    )
    strike_hit = (
        eligible_k[:, :, None, None, :]
        & safe_after_cross
        & (lead_fp >= min_lead_fp)
        & (lead_fp <= max_lead_fp)
        & (lat_abs <= max_lat_fp)
    )  # (E, N, 5, S, K)

    center_tightness = ((5 * FP_ONE - lat_abs).clamp_min(0) * 60).to(torch.int64)
    strike_val = torch.where(
        strike_hit,
        tables["strike_weights"][None, None, None, :, None] + center_tightness,
        torch.zeros((1,), dtype=torch.int64, device=sim.device),
    )
    # Best strike opportunity across horizons S and nearby enemies K per steer (E, N, 5).
    can_hunt_boost = sim.alive & (sim.length > int(cfg.min_boost_length) + 4)
    strike_score = torch.where(
        can_hunt_boost[:, :, None],
        strike_val.amax(dim=(-2, -1)),
        torch.zeros((sim.E, sim.N, NUM_TURN_LEVELS), dtype=torch.int64, device=sim.device),
    )
    has_strike = strike_score > 0  # (E, N, 5)

    # Flanking stalk shaping: pull alongside nearest eligible enemy's shoulder (11 cells ahead,
    # 6 cells laterally on our current side) so the next ticks set up a clean bow crossing.
    side_sign = torch.where(cur_lat >= 0, torch.ones_like(cur_lat), -torch.ones_like(cur_lat))
    lead_dist = 11 * spd_k
    flank_off = 6 * FP_ONE * side_sign
    target_x = hx_k + ((lead_dist * ux_k - flank_off * uy_k) >> TRIG_SHIFT)  # (E, N, K)
    target_y = hy_k + ((lead_dist * uy_k + flank_off * ux_k) >> TRIG_SHIFT)

    # Evaluate distance reduction at normal horizon m=2 (t=7) toward nearest eligible enemy k=0.
    k0_stalk = eligible_k[:, :, 0] & can_hunt_boost & (near_cor4.amax(dim=-1) < 3)
    p7_x = px_n[:, :, :, 2].to(torch.int64)  # (E, N, 5)
    p7_y = py_n[:, :, :, 2].to(torch.int64)
    cur_d2_t0 = (hx_i[:, :, 0] - target_x[:, :, 0]).square() + (
        hy_i[:, :, 0] - target_y[:, :, 0]
    ).square()
    arc_d2_t0 = (p7_x - target_x[:, :, 0, None]).square() + (
        p7_y - target_y[:, :, 0, None]
    ).square()
    stalk_gain = ((cur_d2_t0[:, :, None] - arc_d2_t0) >> 12).clamp(-150_000, 350_000)
    stalk_score = torch.where(k0_stalk[:, :, None], stalk_gain, torch.zeros_like(stalk_gain))

    jitter_slots = (
        int(stream_offset)
        + torch.arange(sim.N * NUM_TURN_LEVELS, dtype=torch.int64, device=sim.device)[None, :]
    )
    jitter = rng_int_torch(
        sim.seeds[:, None], sim.world_ticks[:, None] + 1, PURPOSE_SCRIPTED, jitter_slots, 64
    ).view(sim.E, sim.N, NUM_TURN_LEVELS)

    imm_safe_n = cum_c_n[:, :, :, 0].to(torch.int64) * 40_000_000 + (
        cum_c_n[:, :, :, 1].to(torch.int64) * 20_000_000
    )
    base_safe_score = (
        imm_safe_n + c_safe_n.to(torch.int64) * 4_000_000 + d_safe_n.to(torch.int64) * 1_750_000
    )
    # On steers where a boost strike is active, evaluate boost-arc head-on safety; otherwise normal.
    eff_threat = torch.where(has_strike, threat_bst, threat_norm)
    total_score = (
        base_safe_score
        - eff_threat
        + strike_score
        + stalk_score
        + food_score * 11
        - wall_pen
        + jitter
    )
    steer = total_score.argmax(dim=-1)

    max_safety = len(_ARC_HORIZONS)
    chosen_strike = has_strike.gather(2, steer[:, :, None]).squeeze(2)
    chosen_c_bst = c_safe_b.gather(2, steer[:, :, None]).squeeze(2)
    chosen_d_bst = d_safe_b.gather(2, steer[:, :, None]).squeeze(2)
    chosen_near_bst = near_hits_bst.gather(2, steer[:, :, None]).squeeze(2)
    chosen_cor4 = near_cor4.gather(2, steer[:, :, None]).squeeze(2)

    strike_boost = chosen_strike & (chosen_near_bst == 0) & (chosen_c_bst >= 3)
    corpse_boost = (
        (sim.length > int(cfg.min_boost_length) + 5)
        & (chosen_c_bst >= max_safety)
        & (chosen_d_bst >= max_safety - 1)
        & (chosen_near_bst == 0)
        & (chosen_cor4 >= 2)
    )
    boost = sim.alive & (strike_boost | corpse_boost)
    actions = steer + torch.where(
        boost, torch.full_like(steer, NUM_TURN_LEVELS), torch.zeros_like(steer)
    )
    return torch.where(sim.alive, actions, torch.full_like(actions, 2))
