"""Frozen pre-data {O_0..O_5} spatial macro-option trackers and Mac 1T timing probe.

Implements research-t7-1 §1-2 with review-7 §2-3's exact pending-growth / boost-burn
legality fix and NO ``!spacious`` interrupt:
1. Exact pending-growth TTL (``max(n_o, L_o) - k``) and exact 1-step / 2-step boost-burn
   action legality matching ``GridBatchSim._compute_action_masks_numpy`` / ``CTDeviceSim``,
   plus 1-step head-on size-ratio guard (``1.15 * L_j >= L_0``).
2. Pocket component ``C_{m0}``, cyclic 2-core ``C_2`` via linear degree-<=1 queue peeling,
   earliest release gate ``g*`` and external expansion gate ``g+``, and reverse BFS ``d_{g*}``.
3. Multi-source opponent flood ``d_opp``, 3-heading hero no-wait Voronoi ``V_dir`` and
   corpse-weighted discounted food ``F_dir``, per-action surviving volume ``V_surv``,
   2-core ``V_{2,surv}``, pocket split ``orphan``, parity-matched coil certificate ``T(a)``,
   and bounded depth-28 2-cell/tick boost sprint flood ``V_bst``.
4. Six deterministic closed-loop option trackers ``{O_0..O_5}`` (``K=15`` ticks).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numba as nb
import numpy as np
import torch

from research.r7_macro_fallback import OPTION_NAMES, R7_KILL_THRESHOLDS
from research.sandbox_a_resident.ct_snapshots import CompactWorldSnapshot


@dataclass(frozen=True)
class WorldGridState:
    """Extracted CPU world arrays for single-tick option evaluation."""

    grid_h: int
    grid_w: int
    min_boost_length: int
    boost_cost_frames: int
    alive: np.ndarray  # (S,) uint8
    length: np.ndarray  # (S,) int32
    seg_count: np.ndarray  # (S,) int32
    direction: np.ndarray  # (S,) int32 (0:up, 1:right, 2:down, 3:left)
    boost_frames: np.ndarray  # (S,) int32
    body_offsets: np.ndarray  # (S + 1,) int32
    body_xy: np.ndarray  # (total_segs, 2) int32 (col, row) ordered k=0..seg_count-1
    food_xy: np.ndarray  # (F, 2) int32 (col, row)
    food_corpse: np.ndarray  # (F,) uint8


def state_from_compact_snapshot(
    snap: CompactWorldSnapshot,
    *,
    grid_h: int = 83,
    grid_w: int = 145,
    min_boost_length: int = 10,
    boost_cost_frames: int = 3,
) -> WorldGridState:
    """Convert a ``CompactWorldSnapshot`` into a ``WorldGridState``."""
    st = snap.small_tensors
    return WorldGridState(
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        min_boost_length=int(min_boost_length),
        boost_cost_frames=int(boost_cost_frames),
        alive=st["alive"].numpy().astype(np.uint8),
        length=st["length"].numpy().astype(np.int32),
        seg_count=st["seg_count"].numpy().astype(np.int32),
        direction=st["direction"].numpy().astype(np.int32),
        boost_frames=st["boost_frames"].numpy().astype(np.int32),
        body_offsets=snap.body_offsets.astype(np.int32),
        body_xy=snap.body_segments_xy.astype(np.int32),
        food_xy=snap.food_xy.astype(np.int32),
        food_corpse=snap.food_corpse.astype(np.uint8),
    )


def extract_states_from_ct_sim(
    sim: Any,
    rows: Sequence[int] | None = None,
) -> list[WorldGridState]:
    """Batch-copy selected rows from ``CTDeviceSim`` to CPU ``WorldGridState`` records."""
    if rows is None:
        row_idx = list(range(int(sim.E)))
    else:
        row_idx = [int(r) for r in rows]
    if not row_idx:
        return []

    idx_t = torch.tensor(row_idx, device=sim.device, dtype=torch.long)
    alive_np = sim.alive.index_select(0, idx_t).detach().cpu().numpy().astype(np.uint8)
    length_np = sim.length.index_select(0, idx_t).detach().cpu().numpy().astype(np.int32)
    seg_np = sim.seg_count.index_select(0, idx_t).detach().cpu().numpy().astype(np.int32)
    dir_np = sim.direction.index_select(0, idx_t).detach().cpu().numpy().astype(np.int32)
    bf_np = sim.boost_frames.index_select(0, idx_t).detach().cpu().numpy().astype(np.int32)
    hp_np = sim.head_ptr.index_select(0, idx_t).detach().cpu().numpy().astype(np.int64)
    bodies_np = sim.bodies.index_select(0, idx_t).detach().cpu().numpy()
    f_pres_np = sim.food_present.index_select(0, idx_t).detach().cpu().numpy()
    f_cells_np = sim.food_cells.index_select(0, idx_t).detach().cpu().numpy()
    f_corpse_np = sim.corpse_food.index_select(0, idx_t).detach().cpu().numpy()

    gh = int(sim.grid_h)
    gw = int(sim.grid_w)
    min_bst = int(sim.cfg.min_boost_length)
    bst_frames = int(sim.cfg.boost_length_cost_frames)
    cap = int(sim.cap)
    S = int(sim.S)

    out: list[WorldGridState] = []
    for b in range(len(row_idx)):
        offsets = np.zeros(S + 1, dtype=np.int32)
        tot = int(seg_np[b].sum())
        b_xy = np.empty((tot, 2), dtype=np.int32)
        cur = 0
        for s_idx in range(S):
            sc = int(seg_np[b, s_idx])
            if sc > 0 and alive_np[b, s_idx]:
                rings = (int(hp_np[b, s_idx]) - np.arange(sc, dtype=np.int64)) % cap
                b_xy[cur : cur + sc] = bodies_np[b, s_idx, rings].astype(np.int32)
                cur += sc
            offsets[s_idx + 1] = cur
        b_xy = b_xy[:cur]
        f_active = np.flatnonzero(f_pres_np[b])
        if f_active.size > 0:
            f_xy = f_cells_np[b, f_active].astype(np.int32)
            f_cp = f_corpse_np[b, f_active].astype(np.uint8)
        else:
            f_xy = np.empty((0, 2), dtype=np.int32)
            f_cp = np.empty((0,), dtype=np.uint8)
        out.append(
            WorldGridState(
                grid_h=gh,
                grid_w=gw,
                min_boost_length=min_bst,
                boost_cost_frames=bst_frames,
                alive=alive_np[b],
                length=length_np[b],
                seg_count=seg_np[b],
                direction=dir_np[b],
                boost_frames=bf_np[b],
                body_offsets=offsets,
                body_xy=b_xy,
                food_xy=f_xy,
                food_corpse=f_cp,
            )
        )
    return out


@nb.njit(cache=True)
def _evaluate_options_nb(
    gh: int,
    gw: int,
    min_boost_len: int,
    boost_cost_frames: int,
    alive: np.ndarray,
    length: np.ndarray,
    seg_count: np.ndarray,
    direction: np.ndarray,
    boost_frames: np.ndarray,
    body_offsets: np.ndarray,
    body_xy: np.ndarray,
    food_xy: np.ndarray,
    food_corpse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (chosen_actions[6], option_scores[6, 6], state_diag[4]) in one flat Numba pass.

    state_diag returns ``(any_certified, max_v2_surv, max_v_surv, safe_count)`` for Ply-1
    top-3 ranking without touching future suffixes.
    """
    chosen = np.ones(6, dtype=np.int64)
    scores = np.full((6, 6), -1e18, dtype=np.float64)
    diag = np.zeros(4, dtype=np.float64)

    if alive.shape[0] == 0 or alive[0] == 0:
        return chosen, scores, diag

    n_cells = gh * gw
    owner_grid = np.full(n_cells, -1, dtype=np.int16)
    offset_grid = np.zeros(n_cells, dtype=np.int32)
    tau_own = np.zeros(n_cells, dtype=np.int32)
    tau_opp = np.zeros(n_cells, dtype=np.int32)
    food_grid = np.zeros(n_cells, dtype=np.int8)

    # Cardinal headings: 0=up(0,-1), 1=right(1,0), 2=down(0,1), 3=left(-1,0) in (col, row).
    dc_arr = np.array([0, 1, 0, -1], dtype=np.int32)
    dr_arr = np.array([-1, 0, 1, 0], dtype=np.int32)

    S = alive.shape[0]
    for s in range(S):
        if alive[s] == 0:
            continue
        start = body_offsets[s]
        end = body_offsets[s + 1]
        sc = end - start
        if sc <= 0:
            continue
        L_s = length[s]
        eff_L = L_s if L_s > sc else sc
        drop_tail_opp = (sc >= 2) and (sc >= L_s)
        # Tail-to-head write so head/newer segment owns any self-overlap.
        for k in range(sc - 1, -1, -1):
            col = body_xy[start + k, 0]
            row = body_xy[start + k, 1]
            if 0 <= col < gw and 0 <= row < gh:
                u = row * gw + col
                owner_grid[u] = s
                offset_grid[u] = k
                ttl = eff_L - k
                if ttl < 1:
                    ttl = 1
                if s == 0:
                    tau_own[u] = ttl
                    tau_opp[u] = 0
                else:
                    tau_own[u] = 0
                    # If opponent is not dropping tail due to pending growth, tail stays >1 tick.
                    if k == sc - 1 and drop_tail_opp:
                        tau_opp[u] = 1
                    else:
                        tau_opp[u] = ttl if ttl > 1 else 2

    for f_i in range(food_xy.shape[0]):
        fc = food_xy[f_i, 0]
        fr = food_xy[f_i, 1]
        if 0 <= fc < gw and 0 <= fr < gh:
            food_grid[fr * gw + fc] = 2 if food_corpse[f_i] != 0 else 1

    h0_start = body_offsets[0]
    if body_offsets[1] <= h0_start:
        return chosen, scores, diag
    h_col = body_xy[h0_start, 0]
    h_row = body_xy[h0_start, 1]
    if h_col < 0 or h_col >= gw or h_row < 0 or h_row >= gh:
        return chosen, scores, diag
    h_u = h_row * gw + h_col

    L_0 = length[0]
    n_0 = seg_count[0]
    if body_offsets[1] - h0_start < n_0:
        n_0 = body_offsets[1] - h0_start
    bf_0 = boost_frames[0]
    dir_0 = direction[0] & 3

    # Exact pending-growth & boost-burn legality matching _compute_action_masks_numpy.
    normal_last = (n_0 + 1 if (n_0 + 1) < L_0 else L_0) - 2
    l1 = n_0 + 1 if (n_0 + 1) < L_0 else L_0
    l2 = l1 + 1 if (l1 + 1) < L_0 else L_0
    burn = (bf_0 + 1) >= boost_cost_frames
    lm1 = L_0 - 1 if (L_0 - 1) > 1 else 1
    f_len = (l2 if l2 < lm1 else lm1) if burn else l2
    boost_last = f_len - 3
    can_boost = L_0 >= min_boost_len

    exact_legal = np.zeros(6, dtype=np.uint8)
    safe_mask = np.zeros(6, dtype=np.uint8)
    c1_u = np.full(6, -1, dtype=np.int32)
    cend_u = np.full(6, -1, dtype=np.int32)
    cend_col = np.zeros(6, dtype=np.int32)
    cend_row = np.zeros(6, dtype=np.int32)

    for d_rel in range(3):
        hd = (dir_0 + d_rel - 1) & 3
        dc = dc_arr[hd]
        dr = dr_arr[hd]
        c1c = h_col + dc
        c1r = h_row + dr
        c2c = c1c + dc
        c2r = c1r + dr

        ok_norm = False
        if 0 <= c1c < gw and 0 <= c1r < gh:
            u1 = c1r * gw + c1c
            o1 = owner_grid[u1]
            k1 = offset_grid[u1]
            own_hit_n = (o1 == 0) and (k1 >= 2) and (k1 <= normal_last)
            other_hit_1 = False
            if o1 > 0:
                sc_o = body_offsets[o1 + 1] - body_offsets[o1]
                L_o = length[o1]
                drop_o = (sc_o >= 2) and (sc_o >= L_o)
                olast = (sc_o - 2) if drop_o else (sc_o - 1)
                other_hit_1 = k1 <= olast
            if (not own_hit_n) and (not other_hit_1):
                ok_norm = True
                exact_legal[d_rel] = 1
                c1_u[d_rel] = u1
                cend_u[d_rel] = u1
                cend_col[d_rel] = c1c
                cend_row[d_rel] = c1r

            if ok_norm and can_boost and 0 <= c2c < gw and 0 <= c2r < gh:
                u2 = c2r * gw + c2c
                o2 = owner_grid[u2]
                k2 = offset_grid[u2]
                own_hit_b1 = (o1 == 0) and (k1 >= 1) and (k1 <= boost_last)
                own_hit_b2 = (o2 == 0) and (k2 >= 1) and (k2 <= boost_last)
                other_hit_2 = False
                if o2 > 0:
                    sc_o2 = body_offsets[o2 + 1] - body_offsets[o2]
                    L_o2 = length[o2]
                    drop_o2 = (sc_o2 >= 2) and (sc_o2 >= L_o2)
                    olast2 = (sc_o2 - 2) if drop_o2 else (sc_o2 - 1)
                    other_hit_2 = k2 <= olast2
                if (
                    (not own_hit_b1)
                    and (not own_hit_b2)
                    and (not other_hit_1)
                    and (not other_hit_2)
                ):
                    exact_legal[3 + d_rel] = 1
                    c1_u[3 + d_rel] = u1
                    cend_u[3 + d_rel] = u2
                    cend_col[3 + d_rel] = c2c
                    cend_row[3 + d_rel] = c2r

    # Prune by 1-step head-on risk against opponents with 1.15 * L_j >= L_0.
    for a in range(6):
        if exact_legal[a] == 0:
            continue
        risky = False
        ac = cend_col[a]
        ar = cend_row[a]
        c1c_a = c1_u[a] % gw
        c1r_a = c1_u[a] // gw
        for s in range(1, S):
            if alive[s] == 0 or body_offsets[s + 1] <= body_offsets[s]:
                continue
            if 1.15 * float(length[s]) >= float(L_0):
                ec = body_xy[body_offsets[s], 0]
                er = body_xy[body_offsets[s], 1]
                rad = 2 if length[s] >= min_boost_len else 1
                if abs(ac - ec) + abs(ar - er) <= rad:
                    risky = True
                    break
                if a >= 3 and abs(c1c_a - ec) + abs(c1r_a - er) <= rad:
                    risky = True
                    break
        if not risky:
            safe_mask[a] = 1

    has_safe = False
    for a in range(6):
        if safe_mask[a] == 1:
            has_safe = True
            break
    if not has_safe:
        for a in range(6):
            safe_mask[a] = exact_legal[a]
            if safe_mask[a] == 1:
                has_safe = True
    if not has_safe:
        return chosen, scores, diag

    # Step-1 enterable set Omega_1 (using exact normal_last for own body and vacating opp tail).
    omega1 = np.zeros(n_cells, dtype=np.uint8)
    for u in range(n_cells):
        o = owner_grid[u]
        if o < 0:
            omega1[u] = 1
        elif o == 0:
            k = offset_grid[u]
            if k > normal_last:
                omega1[u] = 1
                tau_own[u] = 1
            elif tau_own[u] <= 1:
                tau_own[u] = 2
        else:
            if tau_opp[u] <= 1:
                omega1[u] = 1
    omega1[h_u] = 1

    # Connected component labeling on Omega_1.
    comp_id = np.full(n_cells, -1, dtype=np.int32)
    comp_sizes = np.zeros(n_cells + 1, dtype=np.int32)
    queue = np.empty(n_cells, dtype=np.int32)
    n_comps = 0
    for start_u in range(n_cells):
        if omega1[start_u] == 1 and comp_id[start_u] < 0:
            cid = n_comps
            n_comps += 1
            q_head = 0
            q_tail = 0
            queue[q_tail] = start_u
            q_tail += 1
            comp_id[start_u] = cid
            sz = 0
            while q_head < q_tail:
                u = queue[q_head]
                q_head += 1
                sz += 1
                ur = u // gw
                uc = u - ur * gw
                if ur > 0 and omega1[u - gw] == 1 and comp_id[u - gw] < 0:
                    comp_id[u - gw] = cid
                    queue[q_tail] = u - gw
                    q_tail += 1
                if ur + 1 < gh and omega1[u + gw] == 1 and comp_id[u + gw] < 0:
                    comp_id[u + gw] = cid
                    queue[q_tail] = u + gw
                    q_tail += 1
                if uc > 0 and omega1[u - 1] == 1 and comp_id[u - 1] < 0:
                    comp_id[u - 1] = cid
                    queue[q_tail] = u - 1
                    q_tail += 1
                if uc + 1 < gw and omega1[u + 1] == 1 and comp_id[u + 1] < 0:
                    comp_id[u + 1] = cid
                    queue[q_tail] = u + 1
                    q_tail += 1
            comp_sizes[cid] = sz

    hero_cid = comp_id[h_u]
    hero_pocket_sz = comp_sizes[hero_cid]

    # Degree-<=1 queue peeling on Omega_1 for cyclic 2-core C_2.
    deg = np.zeros(n_cells, dtype=np.int8)
    in_c2 = omega1.copy()
    in_c2[h_u] = 0
    q_head = 0
    q_tail = 0
    for u in range(n_cells):
        if in_c2[u] == 1:
            ur = u // gw
            uc = u - ur * gw
            d = 0
            if ur > 0 and in_c2[u - gw] == 1:
                d += 1
            if ur + 1 < gh and in_c2[u + gw] == 1:
                d += 1
            if uc > 0 and in_c2[u - 1] == 1:
                d += 1
            if uc + 1 < gw and in_c2[u + 1] == 1:
                d += 1
            deg[u] = d
            if d <= 1:
                queue[q_tail] = u
                q_tail += 1

    while q_head < q_tail:
        u = queue[q_head]
        q_head += 1
        if in_c2[u] == 0:
            continue
        in_c2[u] = 0
        ur = u // gw
        uc = u - ur * gw
        for nb_idx in range(4):
            nc = uc + dc_arr[nb_idx]
            nr = ur + dr_arr[nb_idx]
            if 0 <= nc < gw and 0 <= nr < gh:
                v = nr * gw + nc
                if in_c2[v] == 1:
                    deg[v] -= 1
                    if deg[v] == 1:
                        queue[q_tail] = v
                        q_tail += 1

    # Exit gates g* (earliest own-body release on boundary of hero pocket) and g+ (expansion gate).
    g_star_u = -1
    g_star_tau = 1_000_000_000
    g_plus_u = -1
    g_plus_score = -1_000_000_000
    for u in range(n_cells):
        if owner_grid[u] == 0 and omega1[u] == 0 and u != h_u:
            ur = u // gw
            uc = u - ur * gw
            touches_hero = False
            ext_gain = 0
            seen_c0 = -1
            seen_c1 = -1
            seen_c2 = -1
            for nb_idx in range(4):
                nc = uc + dc_arr[nb_idx]
                nr = ur + dr_arr[nb_idx]
                if 0 <= nc < gw and 0 <= nr < gh:
                    v = nr * gw + nc
                    vcid = comp_id[v]
                    if vcid == hero_cid and v != h_u:
                        touches_hero = True
                    elif vcid >= 0 and vcid != hero_cid:
                        if vcid != seen_c0 and vcid != seen_c1 and vcid != seen_c2:
                            ext_gain += comp_sizes[vcid]
                            if seen_c0 < 0:
                                seen_c0 = vcid
                            elif seen_c1 < 0:
                                seen_c1 = vcid
                            else:
                                seen_c2 = vcid
            if touches_hero:
                t_u = tau_own[u]
                if t_u < g_star_tau:
                    g_star_tau = t_u
                    g_star_u = u
                sc_plus = ext_gain - 3 * t_u
                if sc_plus > g_plus_score:
                    g_plus_score = sc_plus
                    g_plus_u = u

    if g_star_u < 0:
        g_star_u = h_u
        g_star_tau = 0
    if g_plus_u < 0:
        g_plus_u = g_star_u
    gp_r = g_plus_u // gw
    gp_c = g_plus_u - gp_r * gw

    # Reverse BFS distance d_{g*}(u) in hero pocket from neighbors of g*.
    INF = 1_000_000
    d_gstar = np.full(n_cells, INF, dtype=np.int32)
    q_head = 0
    q_tail = 0
    gs_r = g_star_u // gw
    gs_c = g_star_u - gs_r * gw
    for nb_idx in range(4):
        nc = gs_c + dc_arr[nb_idx]
        nr = gs_r + dr_arr[nb_idx]
        if 0 <= nc < gw and 0 <= nr < gh:
            v = nr * gw + nc
            if comp_id[v] == hero_cid and v != h_u and d_gstar[v] == INF:
                d_gstar[v] = 1
                queue[q_tail] = v
                q_tail += 1
    while q_head < q_tail:
        u = queue[q_head]
        q_head += 1
        du = d_gstar[u] + 1
        ur = u // gw
        uc = u - ur * gw
        for nb_idx in range(4):
            nc = uc + dc_arr[nb_idx]
            nr = ur + dr_arr[nb_idx]
            if 0 <= nc < gw and 0 <= nr < gh:
                v = nr * gw + nc
                if comp_id[v] == hero_cid and v != h_u and d_gstar[v] == INF:
                    d_gstar[v] = du
                    queue[q_tail] = v
                    q_tail += 1

    # Multi-source opponent flood d_opp(u) respecting max(tau_own, tau_opp) <= d.
    d_opp = np.full(n_cells, INF, dtype=np.int32)
    q_head = 0
    q_tail = 0
    for s in range(1, S):
        if alive[s] == 1 and body_offsets[s + 1] > body_offsets[s]:
            ec = body_xy[body_offsets[s], 0]
            er = body_xy[body_offsets[s], 1]
            if 0 <= ec < gw and 0 <= er < gh:
                eu = er * gw + ec
                d_opp[eu] = 0
                queue[q_tail] = eu
                q_tail += 1
    while q_head < q_tail:
        u = queue[q_head]
        q_head += 1
        nd = d_opp[u] + 1
        if nd > 64:
            continue
        ur = u // gw
        uc = u - ur * gw
        for nb_idx in range(4):
            nc = uc + dc_arr[nb_idx]
            nr = ur + dr_arr[nb_idx]
            if 0 <= nc < gw and 0 <= nr < gh:
                v = nr * gw + nc
                if d_opp[v] == INF:
                    req = tau_own[v] if tau_own[v] > tau_opp[v] else tau_opp[v]
                    if nd >= req:
                        d_opp[v] = nd
                        queue[q_tail] = v
                        q_tail += 1

    # 3-heading hero no-wait flood d_nw(u) -> directional Voronoi V_dir[3] & food F_dir[3].
    d_nw = np.full(n_cells, INF, dtype=np.int32)
    br_origin = np.full(n_cells, -1, dtype=np.int8)
    v_dir = np.zeros(3, dtype=np.float64)
    f_dir = np.zeros(3, dtype=np.float64)
    q_head = 0
    q_tail = 0
    for d_rel in range(3):
        if exact_legal[d_rel] == 1:
            u1 = c1_u[d_rel]
            d_nw[u1] = 1
            br_origin[u1] = d_rel
            queue[q_tail] = u1
            q_tail += 1
    while q_head < q_tail:
        u = queue[q_head]
        q_head += 1
        du = d_nw[u]
        b_orig = br_origin[u]
        if du < d_opp[u]:
            v_dir[b_orig] += 1.0
            fg = food_grid[u]
            if fg > 0 and du <= 40:
                wf = 3.0 if fg == 2 else 1.0
                f_dir[b_orig] += wf * (0.92 ** float(du))
        if du >= 48:
            continue
        nd = du + 1
        ur = u // gw
        uc = u - ur * gw
        for nb_idx in range(4):
            nc = uc + dc_arr[nb_idx]
            nr = ur + dr_arr[nb_idx]
            if 0 <= nc < gw and 0 <= nr < gh:
                v = nr * gw + nc
                if v != h_u and d_nw[v] == INF:
                    req = tau_own[v] if tau_own[v] > tau_opp[v] else tau_opp[v]
                    if nd >= req:
                        d_nw[v] = nd
                        br_origin[v] = b_orig
                        queue[q_tail] = v
                        q_tail += 1

    # Per-action visit_tag surviving pocket & 2-core BFS + depth-28 boost sprint flood.
    visit_tag = np.zeros(n_cells, dtype=np.int32)
    tag_counter = 0

    v_surv = np.zeros(6, dtype=np.float64)
    v2_surv = np.zeros(6, dtype=np.float64)
    orphan = np.zeros(6, dtype=np.float64)
    gate_reach = np.zeros(6, dtype=np.float64)
    delta_par = np.zeros(6, dtype=np.int32)
    t_cert = np.zeros(6, dtype=np.uint8)
    v_bst = np.zeros(6, dtype=np.float64)
    free_deg_end = np.zeros(6, dtype=np.float64)
    tau_adj_end = np.zeros(6, dtype=np.float64)

    base_pocket = hero_pocket_sz - 1  # exclude old head cell h_u
    if base_pocket < 1:
        base_pocket = 1

    for a in range(6):
        if safe_mask[a] == 0:
            continue
        u1 = c1_u[a]
        ue = cend_u[a]
        step_cells = 2 if a >= 3 else 1

        # Surviving Omega_1 \ {h_u, u1, ue} reachable from neighbors of ue (plus ue itself).
        tag_counter += 1
        cur_tag = tag_counter
        visit_tag[h_u] = cur_tag
        visit_tag[u1] = cur_tag
        visit_tag[ue] = cur_tag

        q_head = 0
        q_tail = 0
        ue_r = ue // gw
        ue_c = ue - ue_r * gw
        fdeg = 0
        max_adj_tau = 0
        for nb_idx in range(4):
            nc = ue_c + dc_arr[nb_idx]
            nr = ue_r + dr_arr[nb_idx]
            if 0 <= nc < gw and 0 <= nr < gh:
                v = nr * gw + nc
                if owner_grid[v] == 0 and tau_own[v] > max_adj_tau:
                    max_adj_tau = tau_own[v]
                if omega1[v] == 1 and visit_tag[v] != cur_tag:
                    fdeg += 1
                    visit_tag[v] = cur_tag
                    queue[q_tail] = v
                    q_tail += 1
        free_deg_end[a] = float(fdeg)
        tau_adj_end[a] = float(max_adj_tau) / float(L_0 if L_0 > 1 else 1)

        vs = 1
        v2s = 1 if in_c2[ue] == 1 else 0
        g_ok = 1.0 if d_gstar[ue] < INF else 0.0
        while q_head < q_tail:
            u = queue[q_head]
            q_head += 1
            vs += 1
            if in_c2[u] == 1:
                v2s += 1
            if u == g_star_u or d_gstar[u] == 1:
                g_ok = 1.0
            ur = u // gw
            uc = u - ur * gw
            for nb_idx in range(4):
                nc = uc + dc_arr[nb_idx]
                nr = ur + dr_arr[nb_idx]
                if 0 <= nc < gw and 0 <= nr < gh:
                    v = nr * gw + nc
                    if omega1[v] == 1 and visit_tag[v] != cur_tag:
                        visit_tag[v] = cur_tag
                        queue[q_tail] = v
                        q_tail += 1

        v_surv[a] = float(vs)
        v2_surv[a] = float(v2s)
        orph = float(base_pocket - step_cells - (vs - 1))
        if orph < 0.0:
            orph = 0.0
        orphan[a] = orph
        gate_reach[a] = g_ok

        pop1 = 2 + (1 if burn else 0) if a >= 3 else 1
        ate_end = 1 if food_grid[ue] > 0 else 0
        d_gs = d_gstar[ue] if d_gstar[ue] < INF else base_pocket
        raw_wait = g_star_tau - pop1 + ate_end - d_gs
        if raw_wait < 0:
            raw_wait = 0
        dp = raw_wait + (raw_wait & 1)
        delta_par[a] = dp
        rem_c2 = v2s - d_gs
        if rem_c2 < 0:
            rem_c2 = 0
        cap_coil = 2 * (rem_c2 // 2)
        if v2s >= L_0 or (g_ok > 0.5 and cap_coil >= dp):
            t_cert[a] = 1

        # Bounded depth-28 2-cell/tick sprint flood V_bst(a) with burn relief.
        tag_counter += 1
        b_tag = tag_counter
        visit_tag[h_u] = b_tag
        visit_tag[u1] = b_tag
        visit_tag[ue] = b_tag
        q_head = 0
        q_tail = 0
        queue[q_tail] = ue
        q_tail += 1
        # Track BFS depth via level boundaries on the queue.
        lvl_end = 1
        cur_d = step_cells
        vbst_cnt = 0
        while q_head < q_tail and cur_d < 28:
            next_d = cur_d + 1
            t_arr = (next_d + 1) // 2
            relief = t_arr + (t_arr // boost_cost_frames)
            while q_head < lvl_end:
                u = queue[q_head]
                q_head += 1
                ur = u // gw
                uc = u - ur * gw
                for nb_idx in range(4):
                    nc = uc + dc_arr[nb_idx]
                    nr = ur + dr_arr[nb_idx]
                    if 0 <= nc < gw and 0 <= nr < gh:
                        v = nr * gw + nc
                        if visit_tag[v] != b_tag:
                            ok_cell = False
                            ov = owner_grid[v]
                            if ov < 0:
                                ok_cell = True
                            elif ov == 0 and relief >= tau_own[v]:
                                ok_cell = True
                            elif ov > 0 and t_arr >= tau_opp[v]:
                                ok_cell = True
                            if ok_cell and t_arr < d_opp[v]:
                                visit_tag[v] = b_tag
                                queue[q_tail] = v
                                q_tail += 1
                                vbst_cnt += 1
            lvl_end = q_tail
            cur_d = next_d
        v_bst[a] = float(vbst_cnt)

    sum_t = 0
    max_v2 = 0.0
    max_vs = 0.0
    safe_cnt = 0.0
    for a in range(6):
        if safe_mask[a] == 1:
            safe_cnt += 1.0
            sum_t += int(t_cert[a])
            if v2_surv[a] > max_v2:
                max_v2 = v2_surv[a]
            if v_surv[a] > max_vs:
                max_vs = v_surv[a]
    diag[0] = 1.0 if sum_t > 0 else 0.0
    diag[1] = max_v2
    diag[2] = max_vs
    diag[3] = safe_cnt

    # Nearest opponent within Chebyshev 28 for O_5 (prioritizing 1.15 * L_j < L_0).
    has_o5_target = False
    proj_c = 0
    proj_r = 0
    best_opp_rank = 1_000_000
    for s in range(1, S):
        if alive[s] == 0 or body_offsets[s + 1] <= body_offsets[s]:
            continue
        ec = body_xy[body_offsets[s], 0]
        er = body_xy[body_offsets[s], 1]
        cheb = abs(ec - h_col) if abs(ec - h_col) > abs(er - h_row) else abs(er - h_row)
        if cheb <= 28:
            smaller = 1.15 * float(length[s]) < float(L_0)
            rank = cheb + (0 if smaller else 100)
            if rank < best_opp_rank:
                best_opp_rank = rank
                has_o5_target = True
                ed = direction[s] & 3
                proj_c = ec + 4 * dc_arr[ed]
                proj_r = er + 4 * dr_arr[ed]

    has_normal_safe = (safe_mask[0] == 1) or (safe_mask[1] == 1) or (safe_mask[2] == 1)
    has_boost_safe = (safe_mask[3] == 1) or (safe_mask[4] == 1) or (safe_mask[5] == 1)

    for a in range(6):
        if safe_mask[a] == 0:
            continue
        d_rel = a % 3
        ue = cend_u[a]
        u1 = c1_u[a]
        u1_r = u1 // gw
        u1_c = u1 - u1_r * gw
        b_bonus = 1.0e6 if (t_cert[a] == 1 or sum_t == 0) else 0.0
        d_gs = float(d_gstar[ue] if d_gstar[ue] < INF else 0)
        has_f = 1.0 if food_grid[ue] > 0 else 0.0
        dopp_c = float(d_opp[ue] if d_opp[ue] < 12 else 12)

        # O_0: Serpentine Delay-Coil Holding Orbit (normal moves preferred when available).
        if a < 3 or (not has_normal_safe):
            if delta_par[a] > 2:
                scores[0, a] = (
                    1.0e5 * gate_reach[a]
                    + 0.1 * b_bonus
                    - 1.0e4 * orphan[a]
                    - 50.0 * has_f
                    + 8.0 * d_gs
                    - 4.0 * free_deg_end[a]
                    + 2.0 * tau_adj_end[a]
                    + 0.5 * v2_surv[a]
                )
            else:
                scores[0, a] = (
                    1.0e5 * gate_reach[a] - 12.0 * d_gs + 0.5 * v_surv[a] - 100.0 * orphan[a]
                )

        # O_1: Gate-Breach / Perimeter Trench Orbit.
        if a < 3 or (not has_normal_safe):
            rs = delta_par[a] // 2
            if rs < 2:
                rs = 2
            elif rs > 14:
                rs = 14
            dist_gp = abs(u1_c - gp_c) + abs(u1_r - gp_r)
            scores[1, a] = (
                b_bonus
                + 1.0e4 * gate_reach[a]
                - 2000.0 * orphan[a]
                - 8.0 * float(abs(dist_gp - rs))
                + 4.0 * tau_adj_end[a]
                + 0.4 * v_surv[a]
            )

        # O_2: Open-Voronoi 2-Core Expansion.
        scores[2, a] = (
            b_bonus
            + (200.0 if a < 3 else 0.0)
            + v_dir[d_rel]
            + 0.5 * v2_surv[a]
            - 8.0 * orphan[a]
            + 2.0 * dopp_c
        )

        # O_3: Trail-Certified Corpse-Cluster Forage.
        scores[3, a] = (
            b_bonus
            - (50.0 if a >= 3 else 0.0)
            + 40.0 * f_dir[d_rel]
            + 0.2 * v2_surv[a]
            - 20.0 * orphan[a]
        )

        # O_4: Boost-Neck Evacuation (boost moves when available, else fallback to safe normal).
        if a >= 3 or (not has_boost_safe):
            scores[4, a] = (
                (1.0e5 if a >= 3 else 0.0)
                + 0.2 * b_bonus
                + 4.0 * v_bst[a]
                + 0.5 * v2_surv[a]
                + 0.3 * v_dir[d_rel]
                - 10.0 * orphan[a]
            )

        # O_5: Smaller-Snake Voronoi Cutoff (falls back to O_2 if no target within Chebyshev 28).
        if has_o5_target:
            l1_p = abs(cend_col[a] - proj_c) + abs(cend_row[a] - proj_r)
            va = 2 if a >= 3 else 1
            t_p = (l1_p + va - 1) // va
            bst_close = 120.0 if (a >= 3 and l1_p <= 12) else 0.0
            scores[5, a] = (
                b_bonus
                + bst_close
                - 10.0 * float(t_p)
                + 2.0 * v_bst[a]
                + 0.4 * v_dir[d_rel]
                + 0.2 * v2_surv[a]
            )
        else:
            scores[5, a] = scores[2, a]

    for opt in range(6):
        best_a = 1
        best_sc = -1e19
        for a in range(6):
            if scores[opt, a] > best_sc:
                best_sc = scores[opt, a]
                best_a = a
        if best_sc <= -1e17:
            for a in range(6):
                if safe_mask[a] == 1:
                    best_a = a
                    break
        chosen[opt] = best_a

    return chosen, scores, diag


def evaluate_all_options(state: WorldGridState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(chosen_actions[6], option_scores[6, 6], state_diag[4])`` for ``state``."""
    return _evaluate_options_nb(
        state.grid_h,
        state.grid_w,
        state.min_boost_length,
        state.boost_cost_frames,
        state.alive,
        state.length,
        state.seg_count,
        state.direction,
        state.boost_frames,
        state.body_offsets,
        state.body_xy,
        state.food_xy,
        state.food_corpse,
    )


def select_batch_option_actions(
    sim: Any,
    option_ids: Sequence[int] | np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    """Evaluate closed-loop option actions for ``len(option_ids) == sim.E`` active worlds.

    Does NOT interrupt on ``!spacious`` (per review-7 §3).
    """
    opt_arr = np.asarray(option_ids, dtype=np.int64).reshape(-1)
    if opt_arr.size != int(sim.E):
        raise ValueError(f"option_ids length {opt_arr.size} != sim.E={sim.E}")
    states = extract_states_from_ct_sim(sim)
    acts = np.ones(opt_arr.size, dtype=np.int64)
    diags = np.zeros((opt_arr.size, 4), dtype=np.float64)
    for idx, st in enumerate(states):
        if st.alive[0] != 0:
            chosen, _, dg = evaluate_all_options(st)
            acts[idx] = int(chosen[int(opt_arr[idx])])
            diags[idx] = dg
    return torch.from_numpy(acts).to(device=sim.device, dtype=torch.long), diags


def run_mac_timing_probe(
    states: Sequence[WorldGridState],
    *,
    ticks: int = 50_000,
    warmup_ticks: int = 200,
) -> dict[str, Any]:
    """Run 1-thread Mac CPU latency probe over ``ticks`` warm option-tracker evaluations."""
    if not states:
        raise ValueError("states must be nonempty for timing probe")
    if ticks < 1:
        raise ValueError("ticks must be >= 1")

    n_states = len(states)
    for i in range(warmup_ticks):
        evaluate_all_options(states[i % n_states])

    latencies_ms = np.empty(ticks, dtype=np.float64)
    action_hist = np.zeros((6, 6), dtype=np.int64)
    t_wall0 = time.perf_counter()
    for i in range(ticks):
        st = states[i % n_states]
        t0 = time.perf_counter_ns()
        chosen, _, _ = evaluate_all_options(st)
        t1 = time.perf_counter_ns()
        latencies_ms[i] = float(t1 - t0) * 1e-6
        for opt in range(6):
            action_hist[opt, int(chosen[opt])] += 1
    wall_sec = time.perf_counter() - t_wall0

    mean_ms = float(latencies_ms.mean())
    p50_ms = float(np.percentile(latencies_ms, 50.0))
    p95_ms = float(np.percentile(latencies_ms, 95.0))
    p99_ms = float(np.percentile(latencies_ms, 99.0))
    max_ms = float(latencies_ms.max())
    max_p95_bar = float(R7_KILL_THRESHOLDS["max_mac_1thread_tracker_p95_ms_per_tick"])

    return {
        "schema": "t34-r7-mac-tracker-timing/v1",
        "ticks": int(ticks),
        "warmup_ticks": int(warmup_ticks),
        "num_distinct_states": n_states,
        "options_evaluated_simultaneously": list(OPTION_NAMES),
        "spacious_interrupt_enabled": False,
        "wall_seconds": round(wall_sec, 4),
        "mean_ms_per_tick": round(mean_ms, 4),
        "p50_ms_per_tick": round(p50_ms, 4),
        "p95_ms_per_tick": round(p95_ms, 4),
        "p99_ms_per_tick": round(p99_ms, 4),
        "max_ms_per_tick": round(max_ms, 4),
        "p95_limit_ms_per_tick": max_p95_bar,
        "passes_latency_bar": bool(p95_ms < max_p95_bar),
        "action_histogram_6x6": action_hist.tolist(),
    }
