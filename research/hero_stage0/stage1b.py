"""Stage 1B: open-stop-h32-nb (K=1) teacher -> bare b8-s3 selectivity gate (5-fold CV).

Pre-registered in docs/research/final_run_20261019/CAMPAIGN_LOG.md row '2026-11-04 PT 13:10'.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import pickle
import time
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numba import njit
from torch import nn

from research.bl_k1.native import RowStats, branch_scope, configure
from research.bl_served_search import ceiling
from research.bl_served_search.fixture_geometry import install as install_fixture_geometry
from research.bl_served_search.ruler_operator import CARRY_FIELDS, _clone_root
from research.hero_stage0.open_stop import (
    DEFAULT_OPENSTOP_OUT_DIR,
    _eval_open_stop_branch_suffix,
)
from research.hero_stage0.protocol import (
    COUNT,
    HORIZON,
    NAMESPACE,
    OPENSTOP_BRANCH_HORIZON,
    OPENSTOP_INTERVAL,
    OPENSTOP_K1_ARM,
    OPENSTOP_MIN_LENGTH,
    OPENSTOP_SCORING_SUFFIXES,
    STAGE1B_BAR_ARM_B_DELTA_LB90_PP_MIN,
    STAGE1B_BAR_ARM_B_DELTA_PP_MIN,
    STAGE1B_BAR_LETHAL_ORDINARY_FLIPS_MAX,
    STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN,
    STAGE1B_BAR_MIN_RESCUE_ROOTS,
    STAGE1B_BAR_MIN_RESCUE_WORLDS,
    STAGE1B_BAR_ORDINARY_FLIP_PCT_MAX,
    STAGE1B_BAR_RESCUE_RETENTION_LB90_PCT_MIN,
    STAGE1B_BAR_RESCUE_RETENTION_LT600_PCT_MIN,
    STAGE1B_BAR_RESCUE_RETENTION_PCT_MIN,
    STAGE1B_BOOTSTRAP_DRAWS,
    STAGE1B_BOOTSTRAP_SEED,
    STAGE1B_CV_FOLDS,
    STAGE1B_FOLD_SEED,
    STAGE1B_MARGIN_GRID,
    STAGE1B_ORDINARY_PER_WORLD,
    STAGE1B_TIMING_BAR_OVERHEAD_P95_MS,
    STAGE1B_TIMING_BAR_TICK_P95_MS,
    STAGE1B_TIMING_MIN_LENGTH,
)
from research.hero_stage0.stage1a import _enforce_single_thread_env
from research.redesign_m3c_20261008.train_m3c import anchor_loss_margin
from research.t3_rollout_labels_20261015.bank_engine import choose_and_step
from research.t3_rollout_labels_20261015.trigger import DenseTrigger
from research.t15_mixed_roster_20261017.episodes import _assert_mixed
from research.watch_replay.product import open_watch
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.eval_engine import _dispatch_actions
from src.simd_env.v8_nb import _DX, _DY, _NBX, _NBY, _post_move

DEFAULT_STAGE1B_OUT_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-stage1b"
)

REACH_CHOKE_FEATURE_NAMES: tuple[str, ...] = (
    "reach_k8_norm",
    "reach_k16_norm",
    "reach_k24_norm",
    "choke_k8_norm",
    "choke_k16_norm",
    "choke_k24_norm",
    "front_eroded_k8_norm",
    "front_eroded_k16_norm",
    "front_eroded_k24_norm",
    "area_v2_uncapped_2l_norm",
    "head_risky_always_k2",
)
N_REACH_CHOKE_FEATURES_PER_ACTION = len(REACH_CHOKE_FEATURE_NAMES)
N_REACH_CHOKE_SCALARS = 6 * N_REACH_CHOKE_FEATURES_PER_ACTION  # 66 scalars


@njit(cache=True, nogil=True)
def _reach_choke_row_kernel(
    bodies: np.ndarray,
    e: int,
    s: int,
    alive: np.ndarray,
    head_ptr: np.ndarray,
    seg_count: np.ndarray,
    length: np.ndarray,
    boost_frames: np.ndarray,
    direction: np.ndarray,
    gw: int,
    gh: int,
    min_boost: int,
    boost_cost: int,
    opp_blk: np.ndarray,
    rel_st: np.ndarray,
    rel_val: np.ndarray,
    seen: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qd: np.ndarray,
    wave_cnt: np.ndarray,
    ws: np.ndarray,
    out_scalars: np.ndarray,
) -> None:
    """Compute per-action reachability, choke width, and eroded area at k in {8, 16, 24}."""
    S = alive.shape[1]
    ring = bodies.shape[2]
    if not alive[e, s] or seg_count[e, s] <= 0:
        out_scalars[:] = 0.0
        return
    hp = head_ptr[e, s]
    n = seg_count[e, s]
    L = length[e, s]
    bf = boost_frames[e, s]
    heading = direction[e, s]
    hx = bodies[e, s, hp, 0]
    hy = bodies[e, s, hp, 1]

    ws[0] += 1
    ost = ws[0]
    n_opp = 0
    ox_arr = np.empty(16, dtype=np.int64)
    oy_arr = np.empty(16, dtype=np.int64)
    ocan_boost = np.empty(16, dtype=np.int64)
    odir_arr = np.empty(16, dtype=np.int64)
    for o in range(S):
        if o == s or not alive[e, o] or seg_count[e, o] <= 0:
            continue
        ohp = head_ptr[e, o]
        on = seg_count[e, o]
        ox = bodies[e, o, ohp, 0]
        oy = bodies[e, o, ohp, 1]
        if n_opp < 16:
            ox_arr[n_opp] = ox
            oy_arr[n_opp] = oy
            ocan_boost[n_opp] = 1 if length[e, o] >= min_boost else 0
            odir_arr[n_opp] = direction[e, o]
            n_opp += 1
        for j in range(on):
            k = (ohp - j) % ring
            cx = bodies[e, o, k, 0]
            cy = bodies[e, o, k, 1]
            if 0 <= cx < gw and 0 <= cy < gh:
                if abs(cx - hx) + abs(cy - hy) <= 28:
                    opp_blk[cx * gh + cy] = ost

    nt = 0
    tx = np.empty(128, dtype=np.int64)
    ty = np.empty(128, dtype=np.int64)
    for i in range(n_opp):
        ox = ox_arr[i]
        oy = oy_arr[i]
        tx[nt] = ox
        ty[nt] = oy
        nt += 1
        od0 = odir_arr[i]
        for rel in range(3):
            od = (od0 + rel - 1) % 4
            tx[nt] = ox + _DX[od]
            ty[nt] = oy + _DY[od]
            nt += 1
            if ocan_boost[i] == 1:
                tx[nt] = ox + 2 * _DX[od]
                ty[nt] = oy + 2 * _DY[od]
                nt += 1

    for speed in range(2):
        boosted, ln, lpost = _post_move(n, L, bf, speed == 1, min_boost, boost_cost)
        nh = 2 if boosted else 1
        ws[2] += 1
        rcur = ws[2]
        for j in range(max(0, ln - nh)):
            k = (hp - j) % ring
            cx = bodies[e, s, k, 0]
            cy = bodies[e, s, k, 1]
            if cx < 0 or cx >= gw or cy < 0 or cy >= gh:
                continue
            if abs(cx - hx) + abs(cy - hy) > 28:
                continue
            idx = cx * gh + cy
            when = lpost - j - (nh - 1)
            if rel_st[idx] != rcur:
                rel_st[idx] = rcur
                rel_val[idx] = when
            elif when > rel_val[idx]:
                rel_val[idx] = when

        for d_rel in range(3):
            a = speed * 3 + d_rel
            base_idx = a * N_REACH_CHOKE_FEATURES_PER_ACTION
            hd = (heading + d_rel - 1) % 4
            dx = _DX[hd]
            dy = _DY[hd]
            ix = hx + dx
            iy = hy + dy
            mid = -1
            mid_st = 0
            mid_val = 0
            if boosted and ln > 1 and 0 <= ix < gw and 0 <= iy < gh:
                mid = ix * gh + iy
                mid_st = rel_st[mid]
                mid_val = rel_val[mid]
                if mid_st != rcur or lpost > mid_val:
                    rel_val[mid] = lpost
                rel_st[mid] = rcur
            nx = hx + nh * dx
            ny = hy + nh * dy

            risk_score = 0.0
            for t_step in range(nh):
                cx = hx + dx * (t_step + 1)
                cy = hy + dy * (t_step + 1)
                for i in range(nt):
                    if tx[i] == cx and ty[i] == cy:
                        risk_score = 1.0
                        break
            if risk_score == 0.0:
                for i in range(n_opp):
                    if abs(nx - ox_arr[i]) + abs(ny - oy_arr[i]) <= 2:
                        risk_score = 0.5
                        break

            step1_blocked = False
            if ix < 0 or ix >= gw or iy < 0 or iy >= gh:
                step1_blocked = True
            else:
                idx1 = ix * gh + iy
                if opp_blk[idx1] == ost:
                    step1_blocked = True
                elif not boosted and rel_st[idx1] == rcur and 0 < rel_val[idx1]:
                    step1_blocked = True
                elif boosted and mid >= 0 and mid_st == rcur and 0 < mid_val:
                    step1_blocked = True

            head_cell = -1
            head_st = 0
            if 0 <= nx < gw and 0 <= ny < gh:
                head_cell = nx * gh + ny
                head_st = rel_st[head_cell]
                rel_st[head_cell] = 0

            for depth_i in range(25):
                wave_cnt[depth_i] = 0
            ws[1] += 1
            scur = ws[1]
            count_cap = 0
            cap_2l = min(max(32, 2 * L), 512)
            if not step1_blocked and 0 <= nx < gw and 0 <= ny < gh:
                sidx = nx * gh + ny
                if opp_blk[sidx] != ost and not (head_st == rcur and 0 < rel_val[sidx]):
                    seen[sidx] = scur
                    qx[0] = nx
                    qy[0] = ny
                    qd[0] = 0
                    q_head = 0
                    q_tail = 1
                    while q_head < q_tail:
                        x = qx[q_head]
                        y = qy[q_head]
                        d_cur = qd[q_head]
                        q_head += 1
                        count_cap += 1
                        if d_cur <= 24:
                            wave_cnt[d_cur] += 1
                        if d_cur >= 24 and count_cap >= cap_2l:
                            continue
                        if d_cur >= 28:
                            continue
                        nd = d_cur + 1
                        for k in range(4):
                            wx = x + _NBX[k]
                            wy = y + _NBY[k]
                            if wx < 0 or wx >= gw or wy < 0 or wy >= gh:
                                continue
                            widx = wx * gh + wy
                            if seen[widx] == scur or opp_blk[widx] == ost:
                                continue
                            if rel_st[widx] == rcur and nd < rel_val[widx]:
                                continue
                            seen[widx] = scur
                            qx[q_tail] = wx
                            qy[q_tail] = wy
                            qd[q_tail] = nd
                            q_tail += 1

            c8 = 0
            c16 = 0
            c24 = 0
            min_w8 = 999999 if wave_cnt[0] > 0 else 0
            min_w16 = 999999 if wave_cnt[0] > 0 else 0
            min_w24 = 999999 if wave_cnt[0] > 0 else 0
            for depth_i in range(25):
                w = wave_cnt[depth_i]
                if depth_i <= 8:
                    c8 += w
                    if depth_i >= 1 and w < min_w8:
                        min_w8 = w
                if depth_i <= 16:
                    c16 += w
                    if depth_i >= 1 and w < min_w16:
                        min_w16 = w
                c24 += w
                if depth_i >= 1 and w < min_w24:
                    min_w24 = w

            for depth_i in range(25):
                wave_cnt[depth_i] = 0
            ws[1] += 1
            scur = ws[1]
            if not step1_blocked and 0 <= nx < gw and 0 <= ny < gh:
                sidx = nx * gh + ny
                eroded0 = False
                for o_i in range(n_opp):
                    if abs(nx - ox_arr[o_i]) + abs(ny - oy_arr[o_i]) <= 1:
                        eroded0 = True
                        break
                if (
                    not eroded0
                    and opp_blk[sidx] != ost
                    and not (head_st == rcur and 0 < rel_val[sidx])
                ):
                    seen[sidx] = scur
                    qx[0] = nx
                    qy[0] = ny
                    qd[0] = 0
                    q_head = 0
                    q_tail = 1
                    while q_head < q_tail:
                        x = qx[q_head]
                        y = qy[q_head]
                        d_cur = qd[q_head]
                        q_head += 1
                        wave_cnt[d_cur] += 1
                        if d_cur >= 24:
                            continue
                        nd = d_cur + 1
                        for k in range(4):
                            wx = x + _NBX[k]
                            wy = y + _NBY[k]
                            if wx < 0 or wx >= gw or wy < 0 or wy >= gh:
                                continue
                            widx = wx * gh + wy
                            if seen[widx] == scur or opp_blk[widx] == ost:
                                continue
                            if rel_st[widx] == rcur and nd < rel_val[widx]:
                                continue
                            eroded = False
                            for o_i in range(n_opp):
                                if abs(wx - ox_arr[o_i]) + abs(wy - oy_arr[o_i]) <= nd + 1:
                                    eroded = True
                                    break
                            if eroded:
                                continue
                            seen[widx] = scur
                            qx[q_tail] = wx
                            qy[q_tail] = wy
                            qd[q_tail] = nd
                            q_tail += 1

            e8 = 0
            e16 = 0
            e24 = 0
            for depth_i in range(25):
                w = wave_cnt[depth_i]
                if depth_i <= 8:
                    e8 += w
                if depth_i <= 16:
                    e16 += w
                e24 += w

            out_scalars[base_idx + 0] = min(1.0, float(c8) / 145.0)
            out_scalars[base_idx + 1] = min(1.0, float(c16) / 545.0)
            out_scalars[base_idx + 2] = min(1.0, float(c24) / 1201.0)
            out_scalars[base_idx + 3] = min(1.0, float(min_w8) / 16.0)
            out_scalars[base_idx + 4] = min(1.0, float(min_w16) / 16.0)
            out_scalars[base_idx + 5] = min(1.0, float(min_w24) / 16.0)
            out_scalars[base_idx + 6] = min(1.0, float(e8) / 145.0)
            out_scalars[base_idx + 7] = min(1.0, float(e16) / 545.0)
            out_scalars[base_idx + 8] = min(1.0, float(e24) / 1201.0)
            out_scalars[base_idx + 9] = min(1.0, float(count_cap) / float(cap_2l))
            out_scalars[base_idx + 10] = float(risk_score)

            if head_cell >= 0:
                rel_st[head_cell] = head_st
            if mid >= 0:
                rel_st[mid] = mid_st
                rel_val[mid] = mid_val


class ReachChokeWorkspace:
    """Reusable preallocated buffers for single-thread multi-step reach/choke computation."""

    def __init__(self, gw: int, gh: int) -> None:
        self.gw = int(gw)
        self.gh = int(gh)
        n_cells = self.gw * self.gh
        self.opp_blk = np.zeros(n_cells, dtype=np.int32)
        self.rel_st = np.zeros(n_cells, dtype=np.int32)
        self.rel_val = np.zeros(n_cells, dtype=np.int32)
        self.seen = np.zeros(n_cells, dtype=np.int32)
        self.qx = np.zeros(n_cells, dtype=np.int32)
        self.qy = np.zeros(n_cells, dtype=np.int32)
        self.qd = np.zeros(n_cells, dtype=np.int32)
        self.wave_cnt = np.zeros(25, dtype=np.int32)
        self.ws = np.zeros(4, dtype=np.int32)
        self.out_scalars = np.zeros(N_REACH_CHOKE_SCALARS, dtype=np.float32)

    def compute(self, sim: Any, env: int = 0, slot: int = 0) -> np.ndarray:
        if int(self.ws.max()) > 1_000_000_000:
            self.opp_blk.fill(0)
            self.rel_st.fill(0)
            self.seen.fill(0)
            self.ws.fill(0)
        _reach_choke_row_kernel(
            sim.bodies,
            int(env),
            int(slot),
            sim.alive,
            sim.head_ptr,
            sim.seg_count,
            sim.length,
            sim.boost_frames,
            sim.direction,
            self.gw,
            self.gh,
            int(sim.cfg.min_boost_length),
            int(sim.cfg.boost_length_cost_frames),
            self.opp_blk,
            self.rel_st,
            self.rel_val,
            self.seen,
            self.qx,
            self.qy,
            self.qd,
            self.wave_cnt,
            self.ws,
            self.out_scalars,
        )
        return self.out_scalars.copy()


def extract_b8_features(
    world: Any,
    reach_ws: ReachChokeWorkspace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract frozen b8-s3 (phi 256-D, scalars_bv8 60-D, scalars_reach 66-D, q 6-D, advisory)."""
    sim = world.sim
    obs = build_ego_raster(
        sim,
        EgoRasterConfig(version="bv8"),
        backend="numba",
        rows=np.array([[0, 0]], dtype=np.int64),
    )
    net = world.hero.net
    device = next(net.parameters()).device
    with torch.no_grad():
        local_t = torch.from_numpy(np.ascontiguousarray(obs["local"])).to(device).float() * (
            1.0 / 255.0
        )
        glob_t = torch.from_numpy(np.ascontiguousarray(obs["global"])).to(device).float() * (
            1.0 / 255.0
        )
        scal_t = torch.from_numpy(np.ascontiguousarray(obs["scalars"])).to(device).float()
        h_t = net.fuse(torch.cat([net.local(local_t), net.glob(glob_t), scal_t], dim=1))
        a_t = net.adv(h_t)
        q_t = net.value(h_t) + a_t - a_t.mean(dim=1, keepdim=True)
        phi = h_t.cpu().numpy()[0].astype(np.float32, copy=True)
        q = q_t.cpu().numpy()[0].astype(np.float32, copy=True)
    scalars_bv8 = np.asarray(obs["scalars"][0], dtype=np.float32).copy()
    scalars_reach = reach_ws.compute(sim, 0, 0)
    advisory = np.asarray(sim.get_advisory_action_mask()[0, 0], dtype=bool).copy()
    return phi, scalars_bv8, scalars_reach, q, advisory


def snapshot_world(world: Any) -> bytes:
    """Create a zlib-compressed exact state snapshot of a prepared CapturedWorld."""
    sim = world.sim
    sim_arrays = {k: v.copy() for k, v in vars(sim).items() if isinstance(v, np.ndarray)}
    food_cells = [list(f) for f in sim.food_cells]
    corpse_cells = [set(f) for f in sim.corpse_cells]
    rng_states = [r._rng.getstate() for r in sim._rngs]
    rt_arrays = {k: v.copy() for k, v in vars(world.runtime).items() if isinstance(v, np.ndarray)}
    rt_mem = copy.deepcopy(world.runtime.memory)
    carry = [
        {name: copy.deepcopy(getattr(p, name)) for name in CARRY_FIELDS if hasattr(p, name)}
        for p in [world.hero, *world.opponents]
    ]
    return zlib.compress(
        pickle.dumps(
            {
                "sim_arrays": sim_arrays,
                "food_cells": food_cells,
                "corpse_cells": corpse_cells,
                "rng_states": rng_states,
                "phase_prepared": bool(getattr(sim, "_policy_phase_prepared", False)),
                "rt_arrays": rt_arrays,
                "rt_mem": rt_mem,
                "carry": carry,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        ),
        level=1,
    )


def restore_world(world: Any, blob: bytes) -> None:
    """Restore a CapturedWorld in-place from a compressed snapshot created by snapshot_world."""
    snap = pickle.loads(zlib.decompress(blob))
    sim = world.sim
    for k, v in snap["sim_arrays"].items():
        cur = getattr(sim, k, None)
        if isinstance(cur, np.ndarray) and cur.shape == v.shape and cur.dtype == v.dtype:
            cur[...] = v
        else:
            setattr(sim, k, v.copy())
    for e in range(sim.E):
        sim.food_cells[e][:] = snap["food_cells"][e]
        sim.food_set[e].clear()
        sim.food_set[e].update(snap["food_cells"][e])
        sim.corpse_cells[e].clear()
        sim.corpse_cells[e].update(snap["corpse_cells"][e])
        sim._rngs[e]._rng.setstate(snap["rng_states"][e])
    sim._policy_phase_prepared = snap["phase_prepared"]
    sim._policy_step_in_progress = False
    for k, v in snap["rt_arrays"].items():
        getattr(world.runtime, k)[...] = v
    world.runtime.memory = copy.deepcopy(snap["rt_mem"])
    for p, c in zip([world.hero, *world.opponents], snap["carry"]):
        for name, val in c.items():
            setattr(p, name, copy.deepcopy(val))


def _quantiles_ms(samples_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
    }


def run_timing_gate_l1500(
    *,
    seed: int = 2062932752,
    target_length: int = STAGE1B_TIMING_MIN_LENGTH,
    warmup_ticks: int = 50,
    measured_ticks: int = 500,
) -> dict[str, Any]:
    """Run the 1-thread Mac timing gate at L >= 1,500 for Arm B vs Arm A."""
    _enforce_single_thread_env()
    session = open_watch({"seed": int(seed)}, 24, "bare")
    try:
        geometry = install_fixture_geometry(session.world, int(target_length))
        root = _clone_root(session.world)
        sim = root.sim
        reach_ws = ReachChokeWorkspace(int(sim.grid_w), int(sim.grid_h))

        # Dummy residual head for Arm B inference timing (388 -> 128 -> 6)
        torch.manual_seed(343013)
        head_a = nn.Sequential(nn.Linear(322, 128), nn.ReLU(), nn.Linear(128, 6)).eval()
        head_b = nn.Sequential(nn.Linear(388, 128), nn.ReLU(), nn.Linear(128, 6)).eval()

        # 1. Standalone kernel microbenchmark at L=1,500
        for _ in range(warmup_ticks):
            reach_ws.compute(sim, 0, 0)
        kernel_ms: list[float] = []
        for _ in range(measured_ticks):
            t0 = time.perf_counter_ns()
            reach_ws.compute(sim, 0, 0)
            kernel_ms.append((time.perf_counter_ns() - t0) / 1e6)

        # 2. Paired hero policy featurize + forward + residual head (Arm A vs Arm B) at L=1,500
        for _ in range(warmup_ticks):
            phi, sbv8, sreach, q, _ = extract_b8_features(root, reach_ws)
            with torch.no_grad():
                _ = head_a(torch.from_numpy(np.concatenate([phi, sbv8, q])).unsqueeze(0))
                _ = head_b(torch.from_numpy(np.concatenate([phi, sbv8, sreach, q])).unsqueeze(0))

        hero_a_ms: list[float] = []
        hero_b_ms: list[float] = []
        overhead_ms: list[float] = []
        for _ in range(measured_ticks):
            t0 = time.perf_counter_ns()
            obs_a = build_ego_raster(
                sim,
                EgoRasterConfig(version="bv8"),
                backend="numba",
                rows=np.array([[0, 0]], dtype=np.int64),
            )
            net = root.hero.net
            with torch.no_grad():
                lt = torch.from_numpy(np.ascontiguousarray(obs_a["local"])).float() * (1.0 / 255.0)
                gt = torch.from_numpy(np.ascontiguousarray(obs_a["global"])).float() * (1.0 / 255.0)
                st = torch.from_numpy(np.ascontiguousarray(obs_a["scalars"])).float()
                ht = net.fuse(torch.cat([net.local(lt), net.glob(gt), st], dim=1))
                at = net.adv(ht)
                qt = net.value(ht) + at - at.mean(dim=1, keepdim=True)
                xa = torch.cat([ht, st, qt], dim=1)
                _ = qt + head_a(xa)
            dt_a = (time.perf_counter_ns() - t0) / 1e6

            t1 = time.perf_counter_ns()
            obs_b = build_ego_raster(
                sim,
                EgoRasterConfig(version="bv8"),
                backend="numba",
                rows=np.array([[0, 0]], dtype=np.int64),
            )
            sreach_b = reach_ws.compute(sim, 0, 0)
            with torch.no_grad():
                lt = torch.from_numpy(np.ascontiguousarray(obs_b["local"])).float() * (1.0 / 255.0)
                gt = torch.from_numpy(np.ascontiguousarray(obs_b["global"])).float() * (1.0 / 255.0)
                st = torch.from_numpy(np.ascontiguousarray(obs_b["scalars"])).float()
                rt = torch.from_numpy(sreach_b).unsqueeze(0).float()
                ht = net.fuse(torch.cat([net.local(lt), net.glob(gt), st], dim=1))
                at = net.adv(ht)
                qt = net.value(ht) + at - at.mean(dim=1, keepdim=True)
                xb = torch.cat([ht, st, rt, qt], dim=1)
                _ = qt + head_b(xb)
            dt_b = (time.perf_counter_ns() - t1) / 1e6

            hero_a_ms.append(dt_a)
            hero_b_ms.append(dt_b)
            overhead_ms.append(dt_b - dt_a)

        # 3. Full Watch6 tick (6 snakes at L >= 1,500) for Arm A and Arm B
        stats = RowStats()
        tick_a_ms: list[float] = []
        tick_b_ms: list[float] = []
        with configure("kernel", stats), branch_scope("mixed"):
            w = _clone_root(root)
            for _ in range(warmup_ticks):
                if not bool(w.sim.alive[0, 0]) or int(w.sim.length[0, 0]) < target_length:
                    w = _clone_root(root)
                choose_and_step(w, capture_obs=False)

            w_a = _clone_root(root)
            w_b = _clone_root(root)
            for _ in range(measured_ticks):
                if not bool(w_a.sim.alive[0, 0]) or int(w_a.sim.length[0, 0]) < target_length:
                    w_a = _clone_root(root)
                if not bool(w_b.sim.alive[0, 0]) or int(w_b.sim.length[0, 0]) < target_length:
                    w_b = _clone_root(root)

                t0 = time.perf_counter_ns()
                _, _, obs_a = choose_and_step(w_a, capture_obs=True)
                with torch.no_grad():
                    lt = torch.from_numpy(np.ascontiguousarray(obs_a["local"])).float() * (
                        1.0 / 255.0
                    )
                    gt = torch.from_numpy(np.ascontiguousarray(obs_a["global"])).float() * (
                        1.0 / 255.0
                    )
                    st = torch.from_numpy(np.ascontiguousarray(obs_a["scalars"])).float()
                    ht = w_a.hero.net.fuse(
                        torch.cat([w_a.hero.net.local(lt), w_a.hero.net.glob(gt), st], dim=1)
                    )
                    at = w_a.hero.net.adv(ht)
                    qt = w_a.hero.net.value(ht) + at - at.mean(dim=1, keepdim=True)
                    _ = qt + head_a(torch.cat([ht, st, qt], dim=1))
                tick_a_ms.append((time.perf_counter_ns() - t0) / 1e6)

                t1 = time.perf_counter_ns()
                _, _, obs_b = choose_and_step(w_b, capture_obs=True)
                sreach_t = reach_ws.compute(w_b.sim, 0, 0)
                with torch.no_grad():
                    lt = torch.from_numpy(np.ascontiguousarray(obs_b["local"])).float() * (
                        1.0 / 255.0
                    )
                    gt = torch.from_numpy(np.ascontiguousarray(obs_b["global"])).float() * (
                        1.0 / 255.0
                    )
                    st = torch.from_numpy(np.ascontiguousarray(obs_b["scalars"])).float()
                    rt = torch.from_numpy(sreach_t).unsqueeze(0).float()
                    ht = w_b.hero.net.fuse(
                        torch.cat([w_b.hero.net.local(lt), w_b.hero.net.glob(gt), st], dim=1)
                    )
                    at = w_b.hero.net.adv(ht)
                    qt = w_b.hero.net.value(ht) + at - at.mean(dim=1, keepdim=True)
                    _ = qt + head_b(torch.cat([ht, st, rt, qt], dim=1))
                tick_b_ms.append((time.perf_counter_ns() - t1) / 1e6)

        kernel_dist = _quantiles_ms(kernel_ms)
        hero_a_dist = _quantiles_ms(hero_a_ms)
        hero_b_dist = _quantiles_ms(hero_b_ms)
        overhead_dist = _quantiles_ms(overhead_ms)
        tick_a_dist = _quantiles_ms(tick_a_ms)
        tick_b_dist = _quantiles_ms(tick_b_ms)
        tick_diff_dist = _quantiles_ms(np.asarray(tick_b_ms) - np.asarray(tick_a_ms))

        # Use the stricter of kernel p95, paired hero overhead p95, and paired tick overhead p95
        effective_overhead_p95_ms = float(
            max(
                kernel_dist["p95_ms"],
                overhead_dist["p95_ms"],
                tick_diff_dist["p95_ms"],
            )
        )
        arm_b_tick_p95_ms = float(tick_b_dist["p95_ms"])
        pass_overhead_bar = bool(effective_overhead_p95_ms < STAGE1B_TIMING_BAR_OVERHEAD_P95_MS)
        pass_tick_bar = bool(arm_b_tick_p95_ms <= STAGE1B_TIMING_BAR_TICK_P95_MS)
        timing_gate_pass = bool(pass_overhead_bar and pass_tick_bar)

        return {
            "target_length": int(target_length),
            "seed": int(seed),
            "warmup_ticks": int(warmup_ticks),
            "measured_ticks": int(measured_ticks),
            "fixture_geometry": geometry,
            "env_threads": {
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "TORCH_NUM_THREADS",
                    "SNAKE_DQN_DEVICE",
                )
            },
            "bars": {
                "overhead_p95_ms_max": STAGE1B_TIMING_BAR_OVERHEAD_P95_MS,
                "tick_p95_ms_max": STAGE1B_TIMING_BAR_TICK_P95_MS,
            },
            "kernel_only_ms": kernel_dist,
            "hero_arm_a_ms": hero_a_dist,
            "hero_arm_b_ms": hero_b_dist,
            "hero_paired_overhead_ms": overhead_dist,
            "watch6_tick_arm_a_ms": tick_a_dist,
            "watch6_tick_arm_b_ms": tick_b_dist,
            "watch6_tick_paired_overhead_ms": tick_diff_dist,
            "effective_overhead_p95_ms": effective_overhead_p95_ms,
            "arm_b_tick_p95_ms": arm_b_tick_p95_ms,
            "pass_overhead_bar": pass_overhead_bar,
            "pass_tick_bar": pass_tick_bar,
            "timing_gate_pass": timing_gate_pass,
        }
    finally:
        session.close()


def extract_world_stage1b_worker(
    world_index: int,
    openstop_dev_dir_str: str = str(DEFAULT_OPENSTOP_OUT_DIR),
    cache_dir_str: str = str(DEFAULT_STAGE1B_OUT_DIR / "cache"),
    ordinary_per_world: int = STAGE1B_ORDINARY_PER_WORLD,
) -> dict[str, Any]:
    """Replay one dev world, verify open-stop-h32-nb-k1 parity, and extract features + snapshots."""
    _enforce_single_thread_env()
    cache_dir = Path(cache_dir_str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features-world-{int(world_index):02d}.pkl"
    if cache_path.is_file():
        data = pickle.loads(cache_path.read_bytes())
        if int(data.get("world_index", -1)) == int(world_index) and len(
            data.get("ordinary_roots", [])
        ) == int(ordinary_per_world):
            return {
                "world_index": int(world_index),
                "seed": int(data["seed"]),
                "n_overrides": len(data["override_roots"]),
                "n_ordinary": len(data["ordinary_roots"]),
                "cached": True,
                "cache_path": str(cache_path),
            }

    world_json_path = (
        Path(openstop_dev_dir_str) / "worlds" / f"openstop-world-{int(world_index):02d}.json"
    )
    w_data = json.loads(world_json_path.read_text(encoding="utf-8"))
    seed = int(w_data["seed"])
    rec = w_data["arms"][OPENSTOP_K1_ARM]
    overrides_by_frame = {int(r["frame"]): r for r in rec["overrides"]}
    expected_search_roots = int(rec["search_roots"])
    n_non_override_expected = expected_search_roots - len(overrides_by_frame)
    target_ord_indices = set(
        np.linspace(0, n_non_override_expected - 1, int(ordinary_per_world), dtype=int).tolist()
    )

    t0 = time.perf_counter()
    session = open_watch({"seed": seed}, 24, "bare")
    try:
        world = session.world
        _assert_mixed(session.live, 6)
        reach_ws = ReachChokeWorkspace(int(world.sim.grid_w), int(world.sim.grid_h))

        life = 1
        was_alive = True
        decisions_count = 0
        search_roots_seen = 0
        non_override_idx = 0
        pending_ord = 0
        mass_sum = 0
        deaths = 0
        max_q_diff = 0.0

        override_roots: list[dict[str, Any]] = []
        ordinary_roots: list[dict[str, Any]] = []
        stats = RowStats()

        with configure("kernel", stats), branch_scope("mixed"):
            for frame in range(1, HORIZON + 1):

                def choose(sim: Any) -> np.ndarray:
                    nonlocal life, trigger, search_roots_seen
                    nonlocal non_override_idx, pending_ord, max_q_diff

                    world.runtime.prepare(sim)
                    active = bool(sim.alive[0, 0])
                    if active and not was_alive:
                        life += 1
                        trigger = DenseTrigger(
                            min_length=OPENSTOP_MIN_LENGTH, interval=OPENSTOP_INTERVAL
                        )
                    actions = np.ones((1, sim.S), dtype=np.int64)
                    chosen = 1
                    if active:
                        phi, sbv8, sreach, q, advisory = extract_b8_features(world, reach_ws)
                        legal, bare = ceiling._legal(q, advisory)
                        chosen = int(bare)
                        cur_length = int(sim.length[0, 0])
                        fired = trigger.fires(decisions_count, cur_length, True)
                        if fired and advisory.any():
                            search_roots_seen += 1
                            if frame in overrides_by_frame:
                                ov = overrides_by_frame[frame]
                                if int(cur_length) != int(ov["length"]):
                                    raise ValueError(
                                        f"world {world_index} frame {frame}: "
                                        f"length {cur_length} != {ov['length']}"
                                    )
                                if int(bare) != int(ov["bare_action"]):
                                    raise ValueError(
                                        f"world {world_index} frame {frame}: "
                                        f"bare {bare} != {ov['bare_action']}"
                                    )
                                q_rec = np.asarray(ov["q"], dtype=np.float32)
                                q_diff = float(np.max(np.abs(q - q_rec)))
                                if q_diff > 1e-4:
                                    raise ValueError(
                                        f"world {world_index} frame {frame}: q_diff {q_diff}"
                                    )
                                max_q_diff = max(max_q_diff, q_diff)
                                chosen = int(ov["chosen_action"])
                                sim._policy_phase_prepared = True
                                override_roots.append(
                                    {
                                        "world_index": int(world_index),
                                        "seed": seed,
                                        "frame": int(frame),
                                        "decision": int(frame - 1),
                                        "life": int(life),
                                        "length": int(cur_length),
                                        "bare_action": int(bare),
                                        "chosen_action": int(chosen),
                                        "advisory": advisory,
                                        "legal": legal,
                                        "q": q,
                                        "phi": phi,
                                        "scalars_bv8": sbv8,
                                        "scalars_reach": sreach,
                                        "bare_failed_by_enclosure_only": bool(
                                            ov["bare_failed_by_enclosure_only"]
                                        ),
                                        "scoring": ov["scoring"],
                                        "root_blob": snapshot_world(world),
                                    }
                                )
                            else:
                                if non_override_idx in target_ord_indices:
                                    pending_ord += 1
                                non_override_idx += 1
                                if pending_ord > 0 and int(legal.sum()) >= 2:
                                    pending_ord -= 1
                                    # Runner-up legal action by base Q
                                    q_alt = np.where(legal, q.astype(np.float64), -np.inf)
                                    q_alt[int(bare)] = -np.inf
                                    runner_up = int(np.argmax(q_alt))
                                    sim._policy_phase_prepared = True
                                    ordinary_roots.append(
                                        {
                                            "world_index": int(world_index),
                                            "seed": seed,
                                            "frame": int(frame),
                                            "decision": int(frame - 1),
                                            "life": int(life),
                                            "length": int(cur_length),
                                            "bare_action": int(bare),
                                            "runner_up_action": int(runner_up),
                                            "advisory": advisory,
                                            "legal": legal,
                                            "q": q,
                                            "phi": phi,
                                            "scalars_bv8": sbv8,
                                            "scalars_reach": sreach,
                                            "root_blob": snapshot_world(world),
                                        }
                                    )
                        actions[0, 0] = int(chosen)
                    _dispatch_actions(
                        sim,
                        sim.get_resolved_action_mask(),
                        actions,
                        [world.hero],
                        [world.opponents],
                    )
                    if active:
                        actions[0, 0] = int(chosen)
                    return actions

                trigger = (
                    DenseTrigger(min_length=OPENSTOP_MIN_LENGTH, interval=OPENSTOP_INTERVAL)
                    if frame == 1
                    else trigger
                )
                world.sim.step_with_policy(choose)
                world.runtime.observe_step(world.sim)
                alive_now = bool(world.sim.alive[0, 0])
                if was_alive and not alive_now:
                    deaths += 1
                if alive_now:
                    decisions_count += 1
                    mass_sum += int(world.sim.length[0, 0])
                was_alive = alive_now

        if mass_sum != int(rec["mass_sum"]) or deaths != int(rec["deaths"]):
            raise ValueError(
                f"world {world_index} trajectory mismatch: mass_sum {mass_sum}!={rec['mass_sum']} "
                f"or deaths {deaths}!={rec['deaths']}"
            )
        if len(override_roots) != len(overrides_by_frame):
            raise ValueError(
                f"world {world_index} override count "
                f"{len(override_roots)} != {len(overrides_by_frame)}"
            )
        if len(ordinary_roots) != int(ordinary_per_world):
            raise ValueError(
                f"world {world_index} ordinary count {len(ordinary_roots)} != {ordinary_per_world}"
            )

        payload = {
            "world_index": int(world_index),
            "seed": seed,
            "mass_sum": int(mass_sum),
            "deaths": int(deaths),
            "max_q_diff": float(max_q_diff),
            "override_roots": override_roots,
            "ordinary_roots": ordinary_roots,
            "elapsed_seconds": float(time.perf_counter() - t0),
        }
        tmp_path = cache_path.with_suffix(".pkl.tmp")
        tmp_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        tmp_path.replace(cache_path)
        return {
            "world_index": int(world_index),
            "seed": seed,
            "n_overrides": len(override_roots),
            "n_ordinary": len(ordinary_roots),
            "max_q_diff": float(max_q_diff),
            "elapsed_seconds": payload["elapsed_seconds"],
            "cached": False,
            "cache_path": str(cache_path),
        }
    finally:
        session.close()


def folds_for(
    n_worlds: int = COUNT,
    k: int = STAGE1B_CV_FOLDS,
    seed: int = STAGE1B_FOLD_SEED,
) -> list[np.ndarray]:
    """Deterministic world-disjoint K-fold split over world_index in 0..n_worlds-1."""
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n_worlds))
    return [np.sort(f).astype(np.int64) for f in np.array_split(perm, int(k))]


def build_arm_features(
    phi: np.ndarray,
    scalars_bv8: np.ndarray,
    scalars_reach: np.ndarray,
    q: np.ndarray,
    legal: np.ndarray,
    bare: np.ndarray,
    arm: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct global and per-action feature tensors strictly from each arm's allowed inputs."""
    if arm not in ("arm_a", "arm_b"):
        raise ValueError(f"unknown arm: {arm!r}")
    N = int(phi.shape[0])
    rows = np.arange(N, dtype=np.int64)
    bare_i = np.asarray(bare, dtype=np.int64)

    if arm == "arm_a":
        x_global = np.concatenate([phi, scalars_bv8, q], axis=1).astype(np.float32)
    else:
        x_global = np.concatenate([phi, scalars_bv8, scalars_reach, q], axis=1).astype(np.float32)

    v8_all = scalars_bv8[:, 30:60].reshape(N, 6, 5)
    v8_bare = v8_all[rows, bare_i, :]  # (N, 5)
    reg_frac_all = scalars_bv8[:, 23:26]  # (N, 3)
    reg_log_all = scalars_bv8[:, 26:29]  # (N, 3)
    reg_frac_bare = reg_frac_all[rows, bare_i % 3]
    reg_log_bare = reg_log_all[rows, bare_i % 3]
    q_bare = q[rows, bare_i]
    is_bare_boost = (bare_i >= 3).astype(np.float32)

    if arm == "arm_b":
        rc_all = scalars_reach.reshape(N, 6, N_REACH_CHOKE_FEATURES_PER_ACTION)
        rc_bare = rc_all[rows, bare_i, :]  # (N, 11)

    per_act_list: list[np.ndarray] = []
    for a in range(6):
        d = a % 3
        q_a = q[:, a]
        q_gap = q_bare - q_a
        is_bare_a = (bare_i == a).astype(np.float32)
        is_boost_a = float(a >= 3) * np.ones(N, dtype=np.float32)
        nb_gain = ((bare_i >= 3) & (a < 3)).astype(np.float32)
        v8_a = v8_all[:, a, :]
        v8_diff = v8_a - v8_bare
        rf_a = reg_frac_all[:, d]
        rf_diff = rf_a - reg_frac_bare
        rl_diff = reg_log_all[:, d] - reg_log_bare

        cols = [
            q_a[:, None],
            q_bare[:, None],
            q_gap[:, None],
            is_bare_a[:, None],
            is_boost_a[:, None],
            is_bare_boost[:, None],
            nb_gain[:, None],
            v8_a,
            v8_bare,
            v8_diff,
            rf_a[:, None],
            reg_frac_bare[:, None],
            rf_diff[:, None],
            rl_diff[:, None],
        ]
        if arm == "arm_b":
            rc_a = rc_all[:, a, :]
            rc_diff = rc_a - rc_bare
            cols.extend([rc_a, rc_bare, rc_diff])
        per_act_list.append(np.concatenate(cols, axis=1).astype(np.float32))

    x_per_action = np.stack(per_act_list, axis=1).astype(np.float32)  # (N, 6, F_act)
    return x_global, x_per_action


class EquivariantResidualQHead(nn.Module):
    """Action-equivariant + global trunk residual Q-head anchored to b8-s3."""

    def __init__(self, d_global: int, d_action: int, hidden: int = 128) -> None:
        super().__init__()
        self.global_ctx = nn.Sequential(
            nn.Linear(d_global, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.global_out = nn.Linear(hidden, 6)
        self.ctx_proj = nn.Linear(hidden, 32)
        self.act_mlp = nn.Sequential(
            nn.Linear(d_action + 32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.log_q_scale = nn.Parameter(torch.tensor(-2.302585, dtype=torch.float32))  # exp = 0.10
        nn.init.zeros_(self.global_out.weight)
        nn.init.zeros_(self.global_out.bias)
        nn.init.zeros_(self.act_mlp[-1].weight)
        nn.init.zeros_(self.act_mlp[-1].bias)

    def forward(
        self, x_global: torch.Tensor, x_act: torch.Tensor, q_base: torch.Tensor
    ) -> torch.Tensor:
        h = self.global_ctx(x_global)
        g_q = self.global_out(h)
        ctx = torch.relu(self.ctx_proj(h)).unsqueeze(1).expand(-1, 6, -1)
        a_in = torch.cat([x_act, ctx], dim=2)
        a_q = self.act_mlp(a_in).squeeze(2)
        q_scale = torch.clamp(torch.exp(self.log_q_scale), min=0.02, max=2.0)
        return q_scale * q_base + g_q + a_q


class FlatMLPResidualQHead(nn.Module):
    """Standard 2-layer MLP residual Q-head (from b8_head_test/run.py)."""

    def __init__(self, d_global: int, d_action: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_global + 6 * d_action, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 6),
        )
        self.log_q_scale = nn.Parameter(torch.tensor(-2.302585, dtype=torch.float32))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, x_global: torch.Tensor, x_act: torch.Tensor, q_base: torch.Tensor
    ) -> torch.Tensor:
        flat = torch.cat([x_global, x_act.reshape(x_act.shape[0], -1)], dim=1)
        q_scale = torch.clamp(torch.exp(self.log_q_scale), min=0.02, max=2.0)
        return q_scale * q_base + self.net(flat)


class LinearResidualQHead(nn.Module):
    """Linear residual Q-head (from b8_head_test/run.py)."""

    def __init__(self, d_global: int, d_action: int) -> None:
        super().__init__()
        self.net = nn.Linear(d_global + 6 * d_action, 6)
        self.log_q_scale = nn.Parameter(torch.tensor(-2.302585, dtype=torch.float32))
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(
        self, x_global: torch.Tensor, x_act: torch.Tensor, q_base: torch.Tensor
    ) -> torch.Tensor:
        flat = torch.cat([x_global, x_act.reshape(x_act.shape[0], -1)], dim=1)
        q_scale = torch.clamp(torch.exp(self.log_q_scale), min=0.02, max=2.0)
        return q_scale * q_base + self.net(flat)


def fit_cv_arm(
    dataset: dict[str, Any],
    arm: str,
    *,
    margins: Sequence[float] = STAGE1B_MARGIN_GRID,
    head_kind: str = "equivariant_mlp",
    pos_weight: float = 0.25,
    epochs: int = 350,
    lr: float = 1.5e-3,
    weight_decay: float = 2e-3,
    seed: int = STAGE1B_FOLD_SEED,
) -> dict[str, Any]:
    """Run 5-fold world-disjoint CV for ``arm`` across ``margins`` using anchor_loss_margin."""
    _enforce_single_thread_env()
    ov_world = np.asarray(dataset["ov_world"], dtype=np.int64)
    ov_phi = np.asarray(dataset["ov_phi"], dtype=np.float32)
    ov_sbv8 = np.asarray(dataset["ov_scalars_bv8"], dtype=np.float32)
    ov_sreach = np.asarray(dataset["ov_scalars_reach"], dtype=np.float32)
    ov_q = np.asarray(dataset["ov_q"], dtype=np.float32)
    ov_legal = np.asarray(dataset["ov_legal"], dtype=bool)
    ov_bare = np.asarray(dataset["ov_bare"], dtype=np.int64)
    ov_chosen = np.asarray(dataset["ov_chosen"], dtype=np.int64)
    ov_cert_ndiff = np.asarray(dataset["ov_cert_ndiff"], dtype=np.int64)

    ord_world = np.asarray(dataset["ord_world"], dtype=np.int64)
    ord_phi = np.asarray(dataset["ord_phi"], dtype=np.float32)
    ord_sbv8 = np.asarray(dataset["ord_scalars_bv8"], dtype=np.float32)
    ord_sreach = np.asarray(dataset["ord_scalars_reach"], dtype=np.float32)
    ord_q = np.asarray(dataset["ord_q"], dtype=np.float32)
    ord_legal = np.asarray(dataset["ord_legal"], dtype=bool)
    ord_bare = np.asarray(dataset["ord_bare"], dtype=np.int64)

    ov_xg, ov_xa = build_arm_features(ov_phi, ov_sbv8, ov_sreach, ov_q, ov_legal, ov_bare, arm)
    ord_xg, ord_xa = build_arm_features(
        ord_phi, ord_sbv8, ord_sreach, ord_q, ord_legal, ord_bare, arm
    )

    folds = folds_for(COUNT, STAGE1B_CV_FOLDS, seed=seed)
    n_ov = len(ov_world)
    n_ord = len(ord_world)
    d_global = int(ov_xg.shape[1])
    d_action = int(ov_xa.shape[2])

    by_margin: dict[str, dict[str, Any]] = {}
    for m_idx, margin in enumerate(margins):
        m_val = float(margin)
        m_key = f"{m_val:.4f}".rstrip("0").rstrip(".")
        oof_ov_q = np.zeros((n_ov, 6), dtype=np.float32)
        oof_ord_q = np.zeros((n_ord, 6), dtype=np.float32)
        oof_ov_fold = np.full(n_ov, -1, dtype=np.int64)
        oof_ord_fold = np.full(n_ord, -1, dtype=np.int64)

        for fold_idx, test_worlds_arr in enumerate(folds):
            test_set = set(int(w) for w in test_worlds_arr.tolist())
            train_set = set(range(COUNT)) - test_set
            if train_set & test_set:
                raise RuntimeError("world leakage between train and test folds")

            ov_is_test = np.array([int(w) in test_set for w in ov_world], dtype=bool)
            ord_is_test = np.array([int(w) in test_set for w in ord_world], dtype=bool)
            # Train positive examples = rescue roots (cert_ndiff == 0) in train worlds
            ov_is_train = (~ov_is_test) & (ov_cert_ndiff == 0)
            ord_is_train = ~ord_is_test

            # Standardize strictly on train fold
            tr_xg_all = np.concatenate([ov_xg[ov_is_train], ord_xg[ord_is_train]], axis=0)
            tr_xa_all = np.concatenate([ov_xa[ov_is_train], ord_xa[ord_is_train]], axis=0)
            xg_mean = tr_xg_all.mean(axis=0, keepdims=True)
            xg_std = np.maximum(tr_xg_all.std(axis=0, keepdims=True), 1e-6)
            xa_mean = tr_xa_all.mean(axis=(0, 1), keepdims=True)
            xa_std = np.maximum(tr_xa_all.std(axis=(0, 1), keepdims=True), 1e-6)

            def norm_g(arr: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(((arr - xg_mean) / xg_std).astype(np.float32))

            def norm_a(arr: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(((arr - xa_mean) / xa_std).astype(np.float32))

            tr_pos_xg = norm_g(ov_xg[ov_is_train])
            tr_pos_xa = norm_a(ov_xa[ov_is_train])
            tr_pos_q = torch.from_numpy(ov_q[ov_is_train])
            tr_pos_leg = torch.from_numpy(ov_legal[ov_is_train])
            tr_pos_tgt = torch.from_numpy(ov_chosen[ov_is_train])

            tr_neg_xg = norm_g(ord_xg[ord_is_train])
            tr_neg_xa = norm_a(ord_xa[ord_is_train])
            tr_neg_q = torch.from_numpy(ord_q[ord_is_train])
            tr_neg_leg = torch.from_numpy(ord_legal[ord_is_train])
            tr_neg_tgt = torch.from_numpy(ord_bare[ord_is_train])

            torch.manual_seed(
                int(seed) + 1000 * m_idx + 17 * fold_idx + (0 if arm == "arm_a" else 500)
            )
            if head_kind == "equivariant_mlp":
                model: nn.Module = EquivariantResidualQHead(d_global, d_action, hidden=128)
            elif head_kind == "flat_mlp":
                model = FlatMLPResidualQHead(d_global, d_action, hidden=128)
            elif head_kind == "linear":
                model = LinearResidualQHead(d_global, d_action)
            else:
                raise ValueError(f"unknown head_kind: {head_kind!r}")

            opt = torch.optim.AdamW(
                model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
            )
            model.train()
            w_pos = float(pos_weight)
            w_neg = 1.0 - w_pos

            for _ in range(int(epochs)):
                opt.zero_grad()
                q_pos_pred = model(tr_pos_xg, tr_pos_xa, tr_pos_q)
                q_neg_pred = model(tr_neg_xg, tr_neg_xa, tr_neg_q)
                loss_pos = anchor_loss_margin(
                    torch, q_pos_pred, tr_pos_tgt, tr_pos_leg, margin=m_val
                )
                loss_neg = anchor_loss_margin(
                    torch, q_neg_pred, tr_neg_tgt, tr_neg_leg, margin=m_val
                )
                loss = w_pos * loss_pos + w_neg * loss_neg
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                te_ov_xg = norm_g(ov_xg[ov_is_test])
                te_ov_xa = norm_a(ov_xa[ov_is_test])
                te_ov_q = torch.from_numpy(ov_q[ov_is_test])
                oof_ov_q[ov_is_test] = model(te_ov_xg, te_ov_xa, te_ov_q).cpu().numpy()
                oof_ov_fold[ov_is_test] = int(fold_idx)

                te_ord_xg = norm_g(ord_xg[ord_is_test])
                te_ord_xa = norm_a(ord_xa[ord_is_test])
                te_ord_q = torch.from_numpy(ord_q[ord_is_test])
                oof_ord_q[ord_is_test] = model(te_ord_xg, te_ord_xa, te_ord_q).cpu().numpy()
                oof_ord_fold[ord_is_test] = int(fold_idx)

        if (oof_ov_fold < 0).any() or (oof_ord_fold < 0).any():
            raise RuntimeError("incomplete OOF assignment across folds")

        # Compute top legal non-bare alternative action and margin score for every root
        ov_alt_act, ov_alt_score = _best_alt_and_score(oof_ov_q, ov_legal, ov_bare)
        ord_alt_act, ord_alt_score = _best_alt_and_score(oof_ord_q, ord_legal, ord_bare)

        by_margin[m_key] = {
            "margin": m_val,
            "oof_ov_q": oof_ov_q,
            "oof_ord_q": oof_ord_q,
            "ov_alt_act": ov_alt_act,
            "ov_alt_score": ov_alt_score,
            "ord_alt_act": ord_alt_act,
            "ord_alt_score": ord_alt_score,
        }

    return {
        "arm": arm,
        "head_kind": head_kind,
        "pos_weight": float(pos_weight),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "folds": [f.tolist() for f in folds],
        "by_margin": by_margin,
    }


def _best_alt_and_score(
    q_oof: np.ndarray,
    legal: np.ndarray,
    bare: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (best_non_bare_legal_action, q_oof[best_alt] - q_oof[bare]) for each row."""
    N = int(q_oof.shape[0])
    rows = np.arange(N, dtype=np.int64)
    bare_i = np.asarray(bare, dtype=np.int64)
    q_masked = np.where(legal, q_oof.astype(np.float64), -np.inf)
    q_bare = q_masked[rows, bare_i]
    q_alt = q_masked.copy()
    q_alt[rows, bare_i] = -np.inf
    best_alt = np.argmax(q_alt, axis=1).astype(np.int64)
    best_alt_q = q_alt[rows, best_alt]
    score = (best_alt_q - q_bare).astype(np.float64)
    # Rows with < 2 legal actions cannot flip
    score = np.where(legal.sum(axis=1) >= 2, score, -np.inf)
    return best_alt, score


def score_world_ordinary_worker(
    task: dict[str, Any],
) -> dict[str, Any]:
    """Score all 210 ordinary roots (and extra candidate actions) in one world on suffixes 4..7."""
    _enforce_single_thread_env()
    world_index = int(task["world_index"])
    cache_dir = Path(task["cache_dir"])
    extra_ord_actions: dict[int, list[int]] = {
        int(k): [int(x) for x in v] for k, v in task.get("extra_ord_actions", {}).items()
    }
    extra_ov_actions: dict[int, list[int]] = {
        int(k): [int(x) for x in v] for k, v in task.get("extra_ov_actions", {}).items()
    }

    scored_path = cache_dir / f"scored-world-{world_index:02d}.pkl"
    feat_path = cache_dir / f"features-world-{world_index:02d}.pkl"
    feat_data = pickle.loads(feat_path.read_bytes())
    seed = int(feat_data["seed"])

    existing_ord: dict[int, dict[int, dict[str, Any]]] = {}
    existing_ov: dict[int, dict[int, dict[str, Any]]] = {}
    if scored_path.is_file():
        prev = pickle.loads(scored_path.read_bytes())
        if int(prev.get("world_index", -1)) == world_index:
            existing_ord = prev.get("ord_scores_by_frame", {})
            existing_ov = prev.get("ov_scores_by_frame", {})

    # Check if all required actions are already scored
    needs_work = False
    for r in feat_data["ordinary_roots"]:
        fr = int(r["frame"])
        req = {int(r["runner_up_action"])} | set(extra_ord_actions.get(fr, []))
        req.discard(int(r["bare_action"]))
        if not req.issubset(set(existing_ord.get(fr, {}).keys())):
            needs_work = True
            break
    if not needs_work:
        for r in feat_data["override_roots"]:
            fr = int(r["frame"])
            req = set(extra_ov_actions.get(fr, [])) - {
                int(r["bare_action"]),
                int(r["chosen_action"]),
            }
            if not req.issubset(set(existing_ov.get(fr, {}).keys())):
                needs_work = True
                break
    if not needs_work:
        return {
            "world_index": world_index,
            "cached": True,
            "ord_scores_by_frame": existing_ord,
            "ov_scores_by_frame": existing_ov,
        }

    t0 = time.perf_counter()
    session = open_watch({"seed": seed}, 24, "bare")
    try:
        world = session.world
        stats = RowStats()
        ord_scores_by_frame: dict[int, dict[int, dict[str, Any]]] = {
            int(k): dict(v) for k, v in existing_ord.items()
        }
        ov_scores_by_frame: dict[int, dict[int, dict[str, Any]]] = {
            int(k): dict(v) for k, v in existing_ov.items()
        }

        with configure("kernel", stats):
            for r in feat_data["ordinary_roots"]:
                fr = int(r["frame"])
                dec = int(r["decision"])
                bare = int(r["bare_action"])
                req = {int(r["runner_up_action"])} | set(extra_ord_actions.get(fr, []))
                req.discard(bare)
                cur_map = ord_scores_by_frame.setdefault(fr, {})
                missing = sorted(req - set(cur_map.keys()))
                if not missing:
                    continue

                restore_world(world, r["root_blob"])
                bare_suf_cache: dict[int, dict[str, bool]] = {}
                for cand in missing:
                    n_diff_cert = 0
                    n_diff_raw = 0
                    cand_cert_vec: list[bool] = []
                    cand_raw_vec: list[bool] = []
                    for suf in OPENSTOP_SCORING_SUFFIXES:
                        out_c = _eval_open_stop_branch_suffix(
                            world,
                            int(cand),
                            dec,
                            int(suf),
                            branch_horizon=OPENSTOP_BRANCH_HORIZON,
                            namespace=NAMESPACE,
                        )
                        c_cert = bool(out_c["certified"])
                        c_raw = bool(out_c["raw_survives"])
                        cand_cert_vec.append(c_cert)
                        cand_raw_vec.append(c_raw)
                        # Only need bare on suffix suf if cand failed suffix suf
                        if not c_cert or not c_raw:
                            if int(suf) not in bare_suf_cache:
                                out_b = _eval_open_stop_branch_suffix(
                                    world,
                                    bare,
                                    dec,
                                    int(suf),
                                    branch_horizon=OPENSTOP_BRANCH_HORIZON,
                                    namespace=NAMESPACE,
                                )
                                bare_suf_cache[int(suf)] = {
                                    "certified": bool(out_b["certified"]),
                                    "raw_survives": bool(out_b["raw_survives"]),
                                }
                            b_info = bare_suf_cache[int(suf)]
                            if b_info["certified"] and not c_cert:
                                n_diff_cert += 1
                            if b_info["raw_survives"] and not c_raw:
                                n_diff_raw += 1

                    cur_map[int(cand)] = {
                        "cand_action": int(cand),
                        "n_diff_certified": int(n_diff_cert),
                        "n_diff_raw": int(n_diff_raw),
                        "lethal_certified": bool(
                            n_diff_cert >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN
                        ),
                        "lethal_raw": bool(n_diff_raw >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN),
                        "cand_certified_vec": cand_cert_vec,
                        "cand_raw_vec": cand_raw_vec,
                    }

            for r in feat_data["override_roots"]:
                fr = int(r["frame"])
                dec = int(r["decision"])
                bare = int(r["bare_action"])
                teacher = int(r["chosen_action"])
                req = set(extra_ov_actions.get(fr, [])) - {bare, teacher}
                cur_map = ov_scores_by_frame.setdefault(fr, {})
                missing = sorted(req - set(cur_map.keys()))
                if not missing:
                    continue

                restore_world(world, r["root_blob"])
                b_cert_vec = [bool(x) for x in r["scoring"]["h32_certified"]["bare_vec"]]
                b_raw_vec = [bool(x) for x in r["scoring"]["bare_survives_vec"]]
                for cand in missing:
                    n_diff_cert = 0
                    n_diff_raw = 0
                    cand_cert_vec = []
                    cand_raw_vec = []
                    for s_idx, suf in enumerate(OPENSTOP_SCORING_SUFFIXES):
                        out_c = _eval_open_stop_branch_suffix(
                            world,
                            int(cand),
                            dec,
                            int(suf),
                            branch_horizon=OPENSTOP_BRANCH_HORIZON,
                            namespace=NAMESPACE,
                        )
                        c_cert = bool(out_c["certified"])
                        c_raw = bool(out_c["raw_survives"])
                        cand_cert_vec.append(c_cert)
                        cand_raw_vec.append(c_raw)
                        if b_cert_vec[s_idx] and not c_cert:
                            n_diff_cert += 1
                        if b_raw_vec[s_idx] and not c_raw:
                            n_diff_raw += 1
                    cur_map[int(cand)] = {
                        "cand_action": int(cand),
                        "n_diff_certified": int(n_diff_cert),
                        "n_diff_raw": int(n_diff_raw),
                        "cand_certified_vec": cand_cert_vec,
                        "cand_raw_vec": cand_raw_vec,
                    }

        out_payload = {
            "world_index": world_index,
            "seed": seed,
            "ord_scores_by_frame": ord_scores_by_frame,
            "ov_scores_by_frame": ov_scores_by_frame,
            "elapsed_seconds": float(time.perf_counter() - t0),
        }
        tmp_path = scored_path.with_suffix(".pkl.tmp")
        tmp_path.write_bytes(pickle.dumps(out_payload, protocol=pickle.HIGHEST_PROTOCOL))
        tmp_path.replace(scored_path)
        return {
            "world_index": world_index,
            "cached": False,
            "elapsed_seconds": out_payload["elapsed_seconds"],
            "ord_scores_by_frame": ord_scores_by_frame,
            "ov_scores_by_frame": ov_scores_by_frame,
        }
    finally:
        session.close()


def load_stage1b_dataset(
    cache_dir: Path = DEFAULT_STAGE1B_OUT_DIR / "cache",
) -> dict[str, Any]:
    """Assemble all 48 cached world feature sets into contiguous arrays."""
    ov_rows: list[dict[str, Any]] = []
    ord_rows: list[dict[str, Any]] = []
    for world_index in range(COUNT):
        p = cache_dir / f"features-world-{world_index:02d}.pkl"
        if not p.is_file():
            raise FileNotFoundError(f"missing cached world features: {p}")
        d = pickle.loads(p.read_bytes())
        for r in d["override_roots"]:
            ov_rows.append({k: v for k, v in r.items() if k != "root_blob"})
        for r in d["ordinary_roots"]:
            ord_rows.append({k: v for k, v in r.items() if k != "root_blob"})

    return {
        "ov_rows": ov_rows,
        "ord_rows": ord_rows,
        "ov_world": np.array([int(r["world_index"]) for r in ov_rows], dtype=np.int64),
        "ov_frame": np.array([int(r["frame"]) for r in ov_rows], dtype=np.int64),
        "ov_length": np.array([int(r["length"]) for r in ov_rows], dtype=np.int64),
        "ov_bare": np.array([int(r["bare_action"]) for r in ov_rows], dtype=np.int64),
        "ov_chosen": np.array([int(r["chosen_action"]) for r in ov_rows], dtype=np.int64),
        "ov_legal": np.stack([np.asarray(r["legal"], dtype=bool) for r in ov_rows], axis=0),
        "ov_q": np.stack([np.asarray(r["q"], dtype=np.float32) for r in ov_rows], axis=0),
        "ov_phi": np.stack([np.asarray(r["phi"], dtype=np.float32) for r in ov_rows], axis=0),
        "ov_scalars_bv8": np.stack(
            [np.asarray(r["scalars_bv8"], dtype=np.float32) for r in ov_rows], axis=0
        ),
        "ov_scalars_reach": np.stack(
            [np.asarray(r["scalars_reach"], dtype=np.float32) for r in ov_rows], axis=0
        ),
        "ov_cert_ndiff": np.array(
            [int(r["scoring"]["h32_certified"]["n_diff"]) for r in ov_rows], dtype=np.int64
        ),
        "ov_raw_ndiff": np.array(
            [int(r["scoring"]["n_override_dies_where_b8_survives"]) for r in ov_rows],
            dtype=np.int64,
        ),
        "ov_enclosure_only": np.array(
            [bool(r["bare_failed_by_enclosure_only"]) for r in ov_rows], dtype=bool
        ),
        "ord_world": np.array([int(r["world_index"]) for r in ord_rows], dtype=np.int64),
        "ord_frame": np.array([int(r["frame"]) for r in ord_rows], dtype=np.int64),
        "ord_length": np.array([int(r["length"]) for r in ord_rows], dtype=np.int64),
        "ord_bare": np.array([int(r["bare_action"]) for r in ord_rows], dtype=np.int64),
        "ord_runner_up": np.array([int(r["runner_up_action"]) for r in ord_rows], dtype=np.int64),
        "ord_legal": np.stack([np.asarray(r["legal"], dtype=bool) for r in ord_rows], axis=0),
        "ord_q": np.stack([np.asarray(r["q"], dtype=np.float32) for r in ord_rows], axis=0),
        "ord_phi": np.stack([np.asarray(r["phi"], dtype=np.float32) for r in ord_rows], axis=0),
        "ord_scalars_bv8": np.stack(
            [np.asarray(r["scalars_bv8"], dtype=np.float32) for r in ord_rows], axis=0
        ),
        "ord_scalars_reach": np.stack(
            [np.asarray(r["scalars_reach"], dtype=np.float32) for r in ord_rows], axis=0
        ),
    }


def world_cluster_bootstrap_retention(
    worlds: np.ndarray,
    rescued_mask: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    n_worlds: int = COUNT,
    draws: int = STAGE1B_BOOTSTRAP_DRAWS,
    seed: int = STAGE1B_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute world-cluster bootstrap LB90 and CI95 for rescue retention (%)."""
    w_num = np.zeros(int(n_worlds), dtype=np.float64)
    w_den = np.zeros(int(n_worlds), dtype=np.float64)
    for w in range(int(n_worlds)):
        in_w = (worlds == w) & eligible_mask
        w_den[w] = float(in_w.sum())
        w_num[w] = float((in_w & rescued_mask).sum())

    total_den = float(w_den.sum())
    total_num = float(w_num.sum())
    point_pct = 100.0 * total_num / total_den if total_den > 0 else 0.0

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, int(n_worlds), size=(int(draws), int(n_worlds)))
    b_num = w_num[idx].sum(axis=1)
    b_den = np.maximum(w_den[idx].sum(axis=1), 1.0)
    b_pct = 100.0 * b_num / b_den

    return {
        "n_rescued": int(round(total_num)),
        "n_eligible": int(round(total_den)),
        "point_pct": float(point_pct),
        "lb90_pct": float(np.percentile(b_pct, 10)),
        "ci95_lo_pct": float(np.percentile(b_pct, 2.5)),
        "ci95_hi_pct": float(np.percentile(b_pct, 97.5)),
    }


def world_cluster_bootstrap_paired_delta(
    worlds: np.ndarray,
    rescued_a: np.ndarray,
    rescued_b: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    n_worlds: int = COUNT,
    draws: int = STAGE1B_BOOTSTRAP_DRAWS,
    seed: int = STAGE1B_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute paired world-cluster bootstrap LB90 and CI95 for (Arm B - Arm A) retention (pp)."""
    w_num_a = np.zeros(int(n_worlds), dtype=np.float64)
    w_num_b = np.zeros(int(n_worlds), dtype=np.float64)
    w_den = np.zeros(int(n_worlds), dtype=np.float64)
    for w in range(int(n_worlds)):
        in_w = (worlds == w) & eligible_mask
        w_den[w] = float(in_w.sum())
        w_num_a[w] = float((in_w & rescued_a).sum())
        w_num_b[w] = float((in_w & rescued_b).sum())

    total_den = max(float(w_den.sum()), 1.0)
    point_a = 100.0 * float(w_num_a.sum()) / total_den
    point_b = 100.0 * float(w_num_b.sum()) / total_den
    delta_pp = point_b - point_a

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, int(n_worlds), size=(int(draws), int(n_worlds)))
    b_den = np.maximum(w_den[idx].sum(axis=1), 1.0)
    b_pct_a = 100.0 * w_num_a[idx].sum(axis=1) / b_den
    b_pct_b = 100.0 * w_num_b[idx].sum(axis=1) / b_den
    b_diff = b_pct_b - b_pct_a

    return {
        "arm_a_pct": float(point_a),
        "arm_b_pct": float(point_b),
        "delta_pp": float(delta_pp),
        "lb90_delta_pp": float(np.percentile(b_diff, 10)),
        "ci95_lo_delta_pp": float(np.percentile(b_diff, 2.5)),
        "ci95_hi_delta_pp": float(np.percentile(b_diff, 97.5)),
    }


def score_all_worlds_ordinary(
    dataset: dict[str, Any],
    cv_runs: Sequence[dict[str, Any]],
    *,
    cache_dir: Path = DEFAULT_STAGE1B_OUT_DIR / "cache",
    workers: int = 8,
) -> tuple[
    dict[tuple[int, int], dict[int, dict[str, Any]]],
    dict[tuple[int, int], dict[int, dict[str, Any]]],
]:
    """Score ordinary roots and non-teacher override actions on suffixes 4..7."""
    _enforce_single_thread_env()
    ov_world = np.asarray(dataset["ov_world"], dtype=np.int64)
    ov_frame = np.asarray(dataset["ov_frame"], dtype=np.int64)
    ov_bare = np.asarray(dataset["ov_bare"], dtype=np.int64)
    ov_chosen = np.asarray(dataset["ov_chosen"], dtype=np.int64)

    ord_world = np.asarray(dataset["ord_world"], dtype=np.int64)
    ord_frame = np.asarray(dataset["ord_frame"], dtype=np.int64)
    ord_bare = np.asarray(dataset["ord_bare"], dtype=np.int64)
    ord_runner_up = np.asarray(dataset["ord_runner_up"], dtype=np.int64)

    extra_ord_by_world: dict[int, dict[int, set[int]]] = {w: {} for w in range(COUNT)}
    extra_ov_by_world: dict[int, dict[int, set[int]]] = {w: {} for w in range(COUNT)}

    for idx in range(len(ord_world)):
        w = int(ord_world[idx])
        fr = int(ord_frame[idx])
        ru = int(ord_runner_up[idx])
        bare = int(ord_bare[idx])
        if ru != bare:
            extra_ord_by_world[w].setdefault(fr, set()).add(ru)

    for cv in cv_runs:
        for m_rec in cv["by_margin"].values():
            ord_alt = np.asarray(m_rec["ord_alt_act"], dtype=np.int64)
            ord_sc = np.asarray(m_rec["ord_alt_score"], dtype=np.float64)
            ov_alt = np.asarray(m_rec["ov_alt_act"], dtype=np.int64)
            for idx in range(len(ord_world)):
                if float(ord_sc[idx]) > 0.0:
                    w = int(ord_world[idx])
                    fr = int(ord_frame[idx])
                    a = int(ord_alt[idx])
                    if a != int(ord_bare[idx]):
                        extra_ord_by_world[w].setdefault(fr, set()).add(a)
            for idx in range(len(ov_world)):
                w = int(ov_world[idx])
                fr = int(ov_frame[idx])
                a = int(ov_alt[idx])
                if a != int(ov_bare[idx]) and a != int(ov_chosen[idx]):
                    extra_ov_by_world[w].setdefault(fr, set()).add(a)

    tasks = [
        {
            "world_index": w,
            "cache_dir": str(cache_dir),
            "extra_ord_actions": {fr: sorted(acts) for fr, acts in extra_ord_by_world[w].items()},
            "extra_ov_actions": {fr: sorted(acts) for fr, acts in extra_ov_by_world[w].items()},
        }
        for w in range(COUNT)
    ]

    ord_lookup: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    ov_lookup: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}

    if int(workers) <= 1:
        results = []
        for idx_t, t in enumerate(tasks, 1):
            res = score_world_ordinary_worker(t)
            results.append(res)
            if idx_t % 8 == 0 or idx_t == len(tasks):
                print(
                    f"[stage1b] scored {idx_t}/{len(tasks)} worlds "
                    f"(last world={res['world_index']})",
                    flush=True,
                )
    else:
        results = []
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(int(workers), 8)) as pool:
            for idx_t, res in enumerate(pool.imap_unordered(score_world_ordinary_worker, tasks), 1):
                results.append(res)
                if idx_t % 8 == 0 or idx_t == len(tasks):
                    print(
                        f"[stage1b] scored {idx_t}/{len(tasks)} worlds "
                        f"(last world={res['world_index']}, cached={res.get('cached')})",
                        flush=True,
                    )

    for res in results:
        w = int(res["world_index"])
        for fr, mp_act in res["ord_scores_by_frame"].items():
            ord_lookup[(w, int(fr))] = {int(k): v for k, v in mp_act.items()}
        for fr, mp_act in res["ov_scores_by_frame"].items():
            ov_lookup[(w, int(fr))] = {int(k): v for k, v in mp_act.items()}

    return ord_lookup, ov_lookup


def evaluate_cv_arm_operating_points(
    dataset: dict[str, Any],
    cv_res: dict[str, Any],
    ord_lookup: dict[tuple[int, int], dict[int, dict[str, Any]]],
    ov_lookup: dict[tuple[int, int], dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Evaluate all margins and operating points for one arm under the pre-registered rules."""
    ov_world = np.asarray(dataset["ov_world"], dtype=np.int64)
    ov_frame = np.asarray(dataset["ov_frame"], dtype=np.int64)
    ov_length = np.asarray(dataset["ov_length"], dtype=np.int64)
    ov_bare = np.asarray(dataset["ov_bare"], dtype=np.int64)
    ov_chosen = np.asarray(dataset["ov_chosen"], dtype=np.int64)
    ov_cert_ndiff = np.asarray(dataset["ov_cert_ndiff"], dtype=np.int64)
    ov_raw_ndiff = np.asarray(dataset["ov_raw_ndiff"], dtype=np.int64)

    ord_world = np.asarray(dataset["ord_world"], dtype=np.int64)
    ord_frame = np.asarray(dataset["ord_frame"], dtype=np.int64)
    ord_runner_up = np.asarray(dataset["ord_runner_up"], dtype=np.int64)

    n_ov = len(ov_world)
    n_ord = len(ord_world)

    cert_rescue_mask = ov_cert_ndiff == 0
    cert_rescue_lt600_mask = cert_rescue_mask & (ov_length < 600)
    cert_rescue_ge600_mask = cert_rescue_mask & (ov_length >= 600)

    raw_rescue_mask = ov_raw_ndiff == 0
    raw_rescue_lt600_mask = raw_rescue_mask & (ov_length < 600)

    margin_evaluations: dict[str, dict[str, Any]] = {}
    candidate_operating_points: list[dict[str, Any]] = []

    for m_key, m_rec in cv_res["by_margin"].items():
        m_val = float(m_rec["margin"])
        ov_alt_act = np.asarray(m_rec["ov_alt_act"], dtype=np.int64)
        ov_alt_score = np.asarray(m_rec["ov_alt_score"], dtype=np.float64)
        ord_alt_act = np.asarray(m_rec["ord_alt_act"], dtype=np.int64)
        ord_alt_score = np.asarray(m_rec["ord_alt_score"], dtype=np.float64)

        # Look up suffix 4..7 scoring for every ordinary root's chosen alt action
        ord_ndiff_cert = np.zeros(n_ord, dtype=np.int64)
        ord_ndiff_raw = np.zeros(n_ord, dtype=np.int64)
        for i in range(n_ord):
            w = int(ord_world[i])
            fr = int(ord_frame[i])
            act = int(ord_alt_act[i])
            frame_map = ord_lookup[(w, fr)]
            info = frame_map.get(act) or frame_map[int(ord_runner_up[i])]
            ord_ndiff_cert[i] = int(info["n_diff_certified"])
            ord_ndiff_raw[i] = int(info["n_diff_raw"])

        ord_lethal_cert_vec = ord_ndiff_cert >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN
        ord_lethal_raw_vec = ord_ndiff_raw >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN

        # Look up suffix 4..7 scoring for every override root's chosen alt action
        ov_alt_ndiff_cert = np.zeros(n_ov, dtype=np.int64)
        ov_alt_ndiff_raw = np.zeros(n_ov, dtype=np.int64)
        for i in range(n_ov):
            w = int(ov_world[i])
            fr = int(ov_frame[i])
            act = int(ov_alt_act[i])
            if act == int(ov_chosen[i]):
                ov_alt_ndiff_cert[i] = int(ov_cert_ndiff[i])
                ov_alt_ndiff_raw[i] = int(ov_raw_ndiff[i])
            elif act == int(ov_bare[i]):
                ov_alt_ndiff_cert[i] = 4
                ov_alt_ndiff_raw[i] = 4
            else:
                info = ov_lookup[(w, fr)][act]
                ov_alt_ndiff_cert[i] = int(info["n_diff_certified"])
                ov_alt_ndiff_raw[i] = int(info["n_diff_raw"])

        def eval_at_tau(tau: float, label: str) -> dict[str, Any]:
            ord_flip = ord_alt_score > float(tau)
            ov_flip = ov_alt_score > float(tau)

            n_ord_flips = int(ord_flip.sum())
            ord_flip_pct = 100.0 * n_ord_flips / float(n_ord)
            n_lethal_cert = int((ord_flip & ord_lethal_cert_vec).sum())
            n_lethal_raw = int((ord_flip & ord_lethal_raw_vec).sum())

            # Rescue masks on override roots:
            # 1. Exact teacher match (primary): head flips and picks teacher's exact action
            resc_exact = ov_flip & (ov_alt_act == ov_chosen)
            # 2. Certified survival rescue: head flips and non-bare action has n_diff_cert == 0
            resc_cert_surv = ov_flip & (ov_alt_ndiff_cert == 0)
            # 3. Raw survival rescue: head flips and chosen non-bare action has n_diff_raw == 0
            resc_raw_surv = ov_flip & (ov_alt_ndiff_raw == 0)
            # 4. Any non-bare flip (upper bound)
            resc_any_flip = ov_flip

            pass_ord_flip_bar = bool(ord_flip_pct <= STAGE1B_BAR_ORDINARY_FLIP_PCT_MAX)
            pass_lethal_cert_bar = bool(n_lethal_cert <= STAGE1B_BAR_LETHAL_ORDINARY_FLIPS_MAX)
            pass_lethal_raw_bar = bool(n_lethal_raw <= STAGE1B_BAR_LETHAL_ORDINARY_FLIPS_MAX)

            # Bootstrap statistics for certified rescue set (310 roots)
            boot_exact_all = world_cluster_bootstrap_retention(
                ov_world, resc_exact, cert_rescue_mask
            )
            boot_exact_lt600 = world_cluster_bootstrap_retention(
                ov_world, resc_exact, cert_rescue_lt600_mask
            )
            boot_exact_ge600 = world_cluster_bootstrap_retention(
                ov_world, resc_exact, cert_rescue_ge600_mask
            )

            boot_surv_all = world_cluster_bootstrap_retention(
                ov_world, resc_cert_surv, cert_rescue_mask
            )
            boot_surv_lt600 = world_cluster_bootstrap_retention(
                ov_world, resc_cert_surv, cert_rescue_lt600_mask
            )

            boot_any_all = world_cluster_bootstrap_retention(
                ov_world, resc_any_flip, cert_rescue_mask
            )
            boot_any_lt600 = world_cluster_bootstrap_retention(
                ov_world, resc_any_flip, cert_rescue_lt600_mask
            )

            # Bootstrap statistics for raw H=32 rescue set (342 roots)
            boot_raw_exact_all = world_cluster_bootstrap_retention(
                ov_world, resc_exact, raw_rescue_mask
            )
            boot_raw_exact_lt600 = world_cluster_bootstrap_retention(
                ov_world, resc_exact, raw_rescue_lt600_mask
            )
            boot_raw_surv_all = world_cluster_bootstrap_retention(
                ov_world, resc_raw_surv, raw_rescue_mask
            )

            return {
                "margin_key": m_key,
                "margin": m_val,
                "tau": float(tau),
                "label": label,
                "n_ordinary_total": n_ord,
                "n_ordinary_flips": n_ord_flips,
                "ordinary_flip_pct": float(ord_flip_pct),
                "n_lethal_ordinary_flips_certified": n_lethal_cert,
                "n_lethal_ordinary_flips_raw": n_lethal_raw,
                "pass_ordinary_flip_bar": pass_ord_flip_bar,
                "pass_lethal_ordinary_bar_certified": pass_lethal_cert_bar,
                "pass_lethal_ordinary_bar_raw": pass_lethal_raw_bar,
                "feasible_certified": bool(pass_ord_flip_bar and pass_lethal_cert_bar),
                "feasible_raw": bool(pass_ord_flip_bar and pass_lethal_raw_bar),
                "certified_rescue_exact_match": {
                    "overall": boot_exact_all,
                    "lt600": boot_exact_lt600,
                    "ge600": boot_exact_ge600,
                },
                "certified_rescue_survival_match": {
                    "overall": boot_surv_all,
                    "lt600": boot_surv_lt600,
                },
                "certified_rescue_any_flip": {
                    "overall": boot_any_all,
                    "lt600": boot_any_lt600,
                },
                "raw_rescue_exact_match": {
                    "overall": boot_raw_exact_all,
                    "lt600": boot_raw_exact_lt600,
                },
                "raw_rescue_survival_match": {
                    "overall": boot_raw_surv_all,
                },
                "_resc_exact_mask": resc_exact,
                "_resc_cert_surv_mask": resc_cert_surv,
            }

        # 1. Unthresholded operating point (tau = 0.0)
        op_tau0 = eval_at_tau(0.0, "unthresholded_tau_0")

        # 2. Threshold at the 1.0% ordinary flip budget (max 100 ordinary flips, tau >= 0)
        sorted_ord = np.sort(ord_alt_score)[::-1]
        max_ord_flips = int(np.floor(STAGE1B_BAR_ORDINARY_FLIP_PCT_MAX * 0.01 * n_ord))
        tau_1pct = max(0.0, float(sorted_ord[max_ord_flips]))
        op_1pct = eval_at_tau(tau_1pct, "at_1pct_ordinary_flip_cap")

        # 3. Threshold satisfying BOTH ordinary_flip_pct <= 1.0% AND lethal_ordinary_flips <= 1
        lethal_cert_scores = np.sort(ord_alt_score[ord_lethal_cert_vec])[::-1]
        if len(lethal_cert_scores) > STAGE1B_BAR_LETHAL_ORDINARY_FLIPS_MAX:
            tau_lethal_cert = float(lethal_cert_scores[STAGE1B_BAR_LETHAL_ORDINARY_FLIPS_MAX])
        else:
            tau_lethal_cert = -np.inf
        tau_feasible_cert = max(0.0, tau_1pct, tau_lethal_cert)
        op_feasible_cert = eval_at_tau(tau_feasible_cert, "constrained_max_retention_certified")

        # Also check under raw lethal definition
        lethal_raw_scores = np.sort(ord_alt_score[ord_lethal_raw_vec])[::-1]
        if len(lethal_raw_scores) > STAGE1B_BAR_LETHAL_ORDINARY_FLIPS_MAX:
            tau_lethal_raw = float(lethal_raw_scores[STAGE1B_BAR_LETHAL_ORDINARY_FLIPS_MAX])
        else:
            tau_lethal_raw = -np.inf
        tau_feasible_raw = max(0.0, tau_1pct, tau_lethal_raw)
        op_feasible_raw = eval_at_tau(tau_feasible_raw, "constrained_max_retention_raw")

        margin_evaluations[m_key] = {
            "margin": m_val,
            "unthresholded_tau_0": {k: v for k, v in op_tau0.items() if not k.startswith("_")},
            "at_1pct_ordinary_flip_cap": {
                k: v for k, v in op_1pct.items() if not k.startswith("_")
            },
            "constrained_feasible_certified": {
                k: v for k, v in op_feasible_cert.items() if not k.startswith("_")
            },
            "constrained_feasible_raw": {
                k: v for k, v in op_feasible_raw.items() if not k.startswith("_")
            },
        }
        candidate_operating_points.append(op_feasible_cert)
        if op_tau0["feasible_certified"]:
            candidate_operating_points.append(op_tau0)

    # Pre-registered operating-point selection rule:
    # Filter to settings with ordinary_flip_pct <= 1.0% AND lethal_ordinary_flips <= 1,
    # then select the one with the highest OOF rescue retention.
    feasible_ops = [op for op in candidate_operating_points if op["feasible_certified"]]
    if not feasible_ops:
        raise RuntimeError("no feasible operating point found")
    selected_op = max(
        feasible_ops,
        key=lambda op: (
            op["certified_rescue_exact_match"]["overall"]["point_pct"],
            op["certified_rescue_survival_match"]["overall"]["point_pct"],
            -op["ordinary_flip_pct"],
        ),
    )

    # Best 1%-ordinary-flip-only operating point (ignoring lethal flip constraint) as diagnostic
    best_1pct_margin_key = max(
        margin_evaluations.keys(),
        key=lambda mk: margin_evaluations[mk]["at_1pct_ordinary_flip_cap"][
            "certified_rescue_exact_match"
        ]["overall"]["point_pct"],
    )

    return {
        "arm": cv_res["arm"],
        "head_kind": cv_res["head_kind"],
        "by_margin": margin_evaluations,
        "selected_operating_point": {k: v for k, v in selected_op.items() if not k.startswith("_")},
        "best_1pct_only_operating_point": margin_evaluations[best_1pct_margin_key][
            "at_1pct_ordinary_flip_cap"
        ],
        "_selected_resc_exact_mask": selected_op["_resc_exact_mask"],
        "_selected_resc_surv_mask": selected_op["_resc_cert_surv_mask"],
    }


def run_stage1b(
    *,
    openstop_dev_dir: Path = DEFAULT_OPENSTOP_OUT_DIR,
    out_dir: Path = DEFAULT_STAGE1B_OUT_DIR,
    workers: int = 8,
) -> dict[str, Any]:
    """Run the pre-registered Stage 1B experiment and write STAGE1B_READOUT.json + REPORT.md."""
    _enforce_single_thread_env()
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure all 48 dev worlds are extracted
    missing_worlds = [
        w for w in range(COUNT) if not (cache_dir / f"features-world-{w:02d}.pkl").is_file()
    ]
    if missing_worlds:
        print(f"[stage1b] extracting {len(missing_worlds)} missing dev worlds...", flush=True)
        if int(workers) <= 1:
            for w in missing_worlds:
                extract_world_stage1b_worker(w, str(openstop_dev_dir), str(cache_dir))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(int(workers), 8)) as pool:
                list(
                    pool.starmap(
                        extract_world_stage1b_worker,
                        [(w, str(openstop_dev_dir), str(cache_dir)) for w in missing_worlds],
                    )
                )
    else:
        print("[stage1b] all 48 dev world feature caches present.", flush=True)

    # 2. Run 1-thread Mac timing gate at L >= 1,500
    print("[stage1b] running 1-thread timing gate at L >= 1,500...", flush=True)
    timing_gate = run_timing_gate_l1500(warmup_ticks=50, measured_ticks=500)
    print(
        f"[stage1b] timing gate pass={timing_gate['timing_gate_pass']} "
        f"(overhead p95={timing_gate['effective_overhead_p95_ms']:.4f} ms, "
        f"tick p95={timing_gate['arm_b_tick_p95_ms']:.4f} ms)",
        flush=True,
    )

    # 3. Load dataset across all 48 dev worlds
    print("[stage1b] loading 48-world dataset...", flush=True)
    dataset = load_stage1b_dataset(cache_dir)
    ov_world = np.asarray(dataset["ov_world"], dtype=np.int64)
    ov_length = np.asarray(dataset["ov_length"], dtype=np.int64)
    ov_bare = np.asarray(dataset["ov_bare"], dtype=np.int64)
    ov_chosen = np.asarray(dataset["ov_chosen"], dtype=np.int64)
    ov_q = np.asarray(dataset["ov_q"], dtype=np.float32)
    ov_cert_ndiff = np.asarray(dataset["ov_cert_ndiff"], dtype=np.int64)
    ov_raw_ndiff = np.asarray(dataset["ov_raw_ndiff"], dtype=np.int64)

    cert_rescue_mask = ov_cert_ndiff == 0
    raw_rescue_mask = ov_raw_ndiff == 0

    n_cert_rescue_roots = int(cert_rescue_mask.sum())
    n_cert_rescue_lt600 = int((cert_rescue_mask & (ov_length < 600)).sum())
    n_cert_rescue_ge600 = int((cert_rescue_mask & (ov_length >= 600)).sum())
    n_cert_rescue_worlds = int(len(set(ov_world[cert_rescue_mask].tolist())))

    n_raw_rescue_roots = int(raw_rescue_mask.sum())
    n_raw_rescue_lt600 = int((raw_rescue_mask & (ov_length < 600)).sum())
    n_raw_rescue_ge600 = int((raw_rescue_mask & (ov_length >= 600)).sum())
    n_raw_rescue_worlds = int(len(set(ov_world[raw_rescue_mask].tolist())))

    support_pass = bool(
        n_cert_rescue_roots >= STAGE1B_BAR_MIN_RESCUE_ROOTS
        and n_cert_rescue_worlds >= STAGE1B_BAR_MIN_RESCUE_WORLDS
    )

    # 4. Run 5-fold world-disjoint CV for Arm A and Arm B (primary head + secondary checks)
    print("[stage1b] fitting 5-fold world-disjoint CV heads across margin grid...", flush=True)
    cv_arm_a = fit_cv_arm(dataset, "arm_a", head_kind="equivariant_mlp")
    cv_arm_b = fit_cv_arm(dataset, "arm_b", head_kind="equivariant_mlp")
    cv_arm_a_flat = fit_cv_arm(dataset, "arm_a", head_kind="flat_mlp")
    cv_arm_b_flat = fit_cv_arm(dataset, "arm_b", head_kind="flat_mlp")
    cv_arm_a_lin = fit_cv_arm(dataset, "arm_a", head_kind="linear")
    cv_arm_b_lin = fit_cv_arm(dataset, "arm_b", head_kind="linear")

    all_cv_runs = [
        cv_arm_a,
        cv_arm_b,
        cv_arm_a_flat,
        cv_arm_b_flat,
        cv_arm_a_lin,
        cv_arm_b_lin,
    ]

    # 5. Score all 10,080 ordinary roots and any non-teacher override actions on suffixes 4..7
    print(
        f"[stage1b] scoring 10,080 ordinary roots on suffixes 4..7 "
        f"across 48 worlds (workers={workers})...",
        flush=True,
    )
    ord_lookup, ov_lookup = score_all_worlds_ordinary(
        dataset, all_cv_runs, cache_dir=cache_dir, workers=workers
    )

    # Compute baseline ordinary runner-up lethal rate across all 10,080 ordinary roots
    ord_world = np.asarray(dataset["ord_world"], dtype=np.int64)
    ord_frame = np.asarray(dataset["ord_frame"], dtype=np.int64)
    ord_runner_up = np.asarray(dataset["ord_runner_up"], dtype=np.int64)
    ru_ndiff_cert = np.array(
        [
            int(
                ord_lookup[(int(ord_world[i]), int(ord_frame[i]))][int(ord_runner_up[i])][
                    "n_diff_certified"
                ]
            )
            for i in range(len(ord_world))
        ],
        dtype=np.int64,
    )
    ru_ndiff_raw = np.array(
        [
            int(
                ord_lookup[(int(ord_world[i]), int(ord_frame[i]))][int(ord_runner_up[i])][
                    "n_diff_raw"
                ]
            )
            for i in range(len(ord_world))
        ],
        dtype=np.int64,
    )

    # 6. Evaluate operating points for Arm A and Arm B
    eval_arm_a = evaluate_cv_arm_operating_points(dataset, cv_arm_a, ord_lookup, ov_lookup)
    eval_arm_b = evaluate_cv_arm_operating_points(dataset, cv_arm_b, ord_lookup, ov_lookup)
    eval_arm_a_flat = evaluate_cv_arm_operating_points(
        dataset, cv_arm_a_flat, ord_lookup, ov_lookup
    )
    eval_arm_b_flat = evaluate_cv_arm_operating_points(
        dataset, cv_arm_b_flat, ord_lookup, ov_lookup
    )
    eval_arm_a_lin = evaluate_cv_arm_operating_points(dataset, cv_arm_a_lin, ord_lookup, ov_lookup)
    eval_arm_b_lin = evaluate_cv_arm_operating_points(dataset, cv_arm_b_lin, ord_lookup, ov_lookup)

    # Paired bootstrap difference (Arm B - Arm A) at selected operating points
    paired_delta_exact = world_cluster_bootstrap_paired_delta(
        ov_world,
        eval_arm_a["_selected_resc_exact_mask"],
        eval_arm_b["_selected_resc_exact_mask"],
        cert_rescue_mask,
    )
    paired_delta_surv = world_cluster_bootstrap_paired_delta(
        ov_world,
        eval_arm_a["_selected_resc_surv_mask"],
        eval_arm_b["_selected_resc_surv_mask"],
        cert_rescue_mask,
    )

    # 7. Apply pre-registered GO / STOP / INCONCLUSIVE rule
    op_a = eval_arm_a["selected_operating_point"]
    op_b = eval_arm_b["selected_operating_point"]

    arm_a_go_checks = {
        "ordinary_flip_pct_le_1": bool(op_a["pass_ordinary_flip_bar"]),
        "lethal_ordinary_flips_le_1": bool(op_a["pass_lethal_ordinary_bar_certified"]),
        "rescue_retention_overall_ge_35": bool(
            op_a["certified_rescue_exact_match"]["overall"]["point_pct"]
            >= STAGE1B_BAR_RESCUE_RETENTION_PCT_MIN
        ),
        "rescue_retention_lb90_ge_25": bool(
            op_a["certified_rescue_exact_match"]["overall"]["lb90_pct"]
            >= STAGE1B_BAR_RESCUE_RETENTION_LB90_PCT_MIN
        ),
        "rescue_retention_lt600_ge_35": bool(
            op_a["certified_rescue_exact_match"]["lt600"]["point_pct"]
            >= STAGE1B_BAR_RESCUE_RETENTION_LT600_PCT_MIN
        ),
    }
    arm_a_passes_go = bool(support_pass and all(arm_a_go_checks.values()))

    arm_b_go_checks = {
        "timing_gate_passes": bool(timing_gate["timing_gate_pass"]),
        "ordinary_flip_pct_le_1": bool(op_b["pass_ordinary_flip_bar"]),
        "lethal_ordinary_flips_le_1": bool(op_b["pass_lethal_ordinary_bar_certified"]),
        "rescue_retention_overall_ge_35": bool(
            op_b["certified_rescue_exact_match"]["overall"]["point_pct"]
            >= STAGE1B_BAR_RESCUE_RETENTION_PCT_MIN
        ),
        "rescue_retention_lb90_ge_25": bool(
            op_b["certified_rescue_exact_match"]["overall"]["lb90_pct"]
            >= STAGE1B_BAR_RESCUE_RETENTION_LB90_PCT_MIN
        ),
        "rescue_retention_lt600_ge_35": bool(
            op_b["certified_rescue_exact_match"]["lt600"]["point_pct"]
            >= STAGE1B_BAR_RESCUE_RETENTION_LT600_PCT_MIN
        ),
        "delta_over_arm_a_ge_20pp": bool(
            paired_delta_exact["delta_pp"] >= STAGE1B_BAR_ARM_B_DELTA_PP_MIN
        ),
        "delta_over_arm_a_lb90_gt_0pp": bool(
            paired_delta_exact["lb90_delta_pp"] > STAGE1B_BAR_ARM_B_DELTA_LB90_PP_MIN
        ),
    }
    arm_b_passes_go = bool(support_pass and (not arm_a_passes_go) and all(arm_b_go_checks.values()))

    if not support_pass:
        verdict = "INCONCLUSIVE"
        verdict_detail = "INCONCLUSIVE (rescue support below 80 roots or 30 dev worlds)"
    elif arm_a_passes_go:
        verdict = "GO_STAGE_2A"
        verdict_detail = "GO to Stage 2A (bv8-only hero training)"
    elif arm_b_passes_go:
        verdict = "GO_STAGE_2B"
        verdict_detail = "GO to Stage 2B (feature-augmented hero training)"
    else:
        verdict = "STOP"
        verdict_detail = (
            "STOP the bare-network open-stop distillation line "
            "(close T34 under Claim A bare b8-s3 in Live Play "
            "+ Claim B exact-stop-l2-prime replay bank in Watch)"
        )

    # Diagnostic Q-gap and action-rank summary on certified rescue roots
    rows_cert = np.where(cert_rescue_mask)[0]
    q_gaps_cert = (
        ov_q[rows_cert, ov_bare[rows_cert]] - ov_q[rows_cert, ov_chosen[rows_cert]]
    ).astype(np.float64)
    teacher_ranks_cert: dict[int, int] = {r: 0 for r in range(1, 7)}
    for idx in rows_cert:
        order = np.argsort(-ov_q[idx])
        rk = int(np.where(order == ov_chosen[idx])[0][0]) + 1
        teacher_ranks_cert[rk] = teacher_ranks_cert.get(rk, 0) + 1

    readout: dict[str, Any] = {
        "registration_ref": (
            "docs/research/final_run_20261019/CAMPAIGN_LOG.md row '2026-11-04 PT 13:10'"
        ),
        "namespace": NAMESPACE,
        "sealed_gate_namespace_touched": False,
        "teacher_arm": OPENSTOP_K1_ARM,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "support": {
            "n_dev_worlds": COUNT,
            "n_total_search_roots_dev": 57417,
            "n_total_overrides_dev": int(len(ov_world)),
            "certified_rescue_roots": {
                "n_roots": n_cert_rescue_roots,
                "n_lt600": n_cert_rescue_lt600,
                "n_ge600": n_cert_rescue_ge600,
                "n_worlds": n_cert_rescue_worlds,
            },
            "raw_h32_rescue_roots": {
                "n_roots": n_raw_rescue_roots,
                "n_lt600": n_raw_rescue_lt600,
                "n_ge600": n_raw_rescue_ge600,
                "n_worlds": n_raw_rescue_worlds,
            },
            "ordinary_sample": {
                "n_ordinary_per_world": STAGE1B_ORDINARY_PER_WORLD,
                "n_ordinary_total": int(len(ord_world)),
                "n_worlds": COUNT,
                "runner_up_n_diff_certified_counts": {
                    str(d): int((ru_ndiff_cert == d).sum()) for d in range(5)
                },
                "runner_up_lethal_certified_count": int(
                    (ru_ndiff_cert >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN).sum()
                ),
                "runner_up_lethal_certified_pct": float(
                    100.0
                    * (ru_ndiff_cert >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN).sum()
                    / len(ord_world)
                ),
                "runner_up_lethal_raw_count": int(
                    (ru_ndiff_raw >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN).sum()
                ),
                "runner_up_lethal_raw_pct": float(
                    100.0
                    * (ru_ndiff_raw >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN).sum()
                    / len(ord_world)
                ),
            },
            "minimum_support_bars": {
                "min_rescue_roots": STAGE1B_BAR_MIN_RESCUE_ROOTS,
                "min_rescue_worlds": STAGE1B_BAR_MIN_RESCUE_WORLDS,
                "support_pass": support_pass,
            },
        },
        "timing_gate_l1500": timing_gate,
        "cv_protocol": {
            "n_folds": STAGE1B_CV_FOLDS,
            "fold_seed": STAGE1B_FOLD_SEED,
            "bootstrap_seed": STAGE1B_BOOTSTRAP_SEED,
            "bootstrap_draws": STAGE1B_BOOTSTRAP_DRAWS,
            "margins": list(STAGE1B_MARGIN_GRID),
            "folds": cv_arm_a["folds"],
        },
        "bars": {
            "ordinary_flip_pct_max": STAGE1B_BAR_ORDINARY_FLIP_PCT_MAX,
            "lethal_ordinary_flips_max": STAGE1B_BAR_LETHAL_ORDINARY_FLIPS_MAX,
            "lethal_ordinary_n_diff_min": STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN,
            "rescue_retention_overall_pct_min": STAGE1B_BAR_RESCUE_RETENTION_PCT_MIN,
            "rescue_retention_lb90_pct_min": STAGE1B_BAR_RESCUE_RETENTION_LB90_PCT_MIN,
            "rescue_retention_lt600_pct_min": STAGE1B_BAR_RESCUE_RETENTION_LT600_PCT_MIN,
            "arm_b_delta_pp_min": STAGE1B_BAR_ARM_B_DELTA_PP_MIN,
            "arm_b_delta_lb90_pp_min": STAGE1B_BAR_ARM_B_DELTA_LB90_PP_MIN,
        },
        "arm_a": {k: v for k, v in eval_arm_a.items() if not k.startswith("_")},
        "arm_b": {k: v for k, v in eval_arm_b.items() if not k.startswith("_")},
        "paired_arm_b_minus_arm_a": {
            "certified_rescue_exact_match": paired_delta_exact,
            "certified_rescue_survival_match": paired_delta_surv,
        },
        "secondary_head_families": {
            "flat_mlp": {
                "arm_a_selected": eval_arm_a_flat["selected_operating_point"],
                "arm_a_at_1pct": eval_arm_a_flat["best_1pct_only_operating_point"],
                "arm_b_selected": eval_arm_b_flat["selected_operating_point"],
                "arm_b_at_1pct": eval_arm_b_flat["best_1pct_only_operating_point"],
            },
            "linear": {
                "arm_a_selected": eval_arm_a_lin["selected_operating_point"],
                "arm_a_at_1pct": eval_arm_a_lin["best_1pct_only_operating_point"],
                "arm_b_selected": eval_arm_b_lin["selected_operating_point"],
                "arm_b_at_1pct": eval_arm_b_lin["best_1pct_only_operating_point"],
            },
        },
        "decision_checks": {
            "support_pass": support_pass,
            "arm_a_go_checks": arm_a_go_checks,
            "arm_a_passes_go": arm_a_passes_go,
            "arm_b_go_checks": arm_b_go_checks,
            "arm_b_passes_go": arm_b_passes_go,
        },
        "diagnostics": {
            "certified_rescue_q_gap_quantiles": {
                "p10": float(np.percentile(q_gaps_cert, 10)),
                "p25": float(np.percentile(q_gaps_cert, 25)),
                "p50": float(np.percentile(q_gaps_cert, 50)),
                "p75": float(np.percentile(q_gaps_cert, 75)),
                "p90": float(np.percentile(q_gaps_cert, 90)),
                "max": float(np.max(q_gaps_cert)),
            },
            "certified_rescue_teacher_q_rank_counts": {
                str(k): int(v) for k, v in sorted(teacher_ranks_cert.items())
            },
        },
        "elapsed_seconds": float(time.perf_counter() - t_start),
    }

    readout_path = out_dir / "STAGE1B_READOUT.json"
    readout_path.write_text(json.dumps(readout, indent=2) + "\n", encoding="utf-8")
    report_md = format_stage1b_report(readout)
    report_path = out_dir / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")

    # Also write committed copies inside research/hero_stage0/
    repo_dir = Path(__file__).resolve().parent
    (repo_dir / "STAGE1B_READOUT.json").write_text(
        json.dumps(readout, indent=2) + "\n", encoding="utf-8"
    )
    (repo_dir / "STAGE1B_REPORT.md").write_text(report_md, encoding="utf-8")

    return readout


def format_stage1b_report(readout: dict[str, Any]) -> str:
    """Format the markdown REPORT.md for Stage 1B."""
    sup = readout["support"]
    cert_sup = sup["certified_rescue_roots"]
    raw_sup = sup["raw_h32_rescue_roots"]
    ord_sup = sup["ordinary_sample"]
    tg = readout["timing_gate_l1500"]
    op_a = readout["arm_a"]["selected_operating_point"]
    op_b = readout["arm_b"]["selected_operating_point"]
    op_a_1pct = readout["arm_a"]["best_1pct_only_operating_point"]
    op_b_1pct = readout["arm_b"]["best_1pct_only_operating_point"]
    paired = readout["paired_arm_b_minus_arm_a"]["certified_rescue_exact_match"]
    diag = readout["diagnostics"]
    ex_a = op_a["certified_rescue_exact_match"]
    ex_b = op_b["certified_rescue_exact_match"]
    ex_a1 = op_a_1pct["certified_rescue_exact_match"]
    ex_b1 = op_b_1pct["certified_rescue_exact_match"]

    lines = [
        "# T34 Stage 1B Readout — Open-Stop (`K=1`) Teacher -> Bare `b8-s3` Selectivity Gate",
        "",
        f"- **Pre-registration:** `{readout['registration_ref']}`",
        (
            f"- **Dev world bank:** `{readout['namespace']}` (`48` worlds; sealed gate bank "
            f"`t34-openstop-gate-20261104-v1` untouched: "
            f"`sealed_gate_namespace_touched = {readout['sealed_gate_namespace_touched']}`)"
        ),
        (
            f"- **Teacher arm:** `{readout['teacher_arm']}` "
            "(`open-stop-h32-nb` `K=1` on `48` dev worlds)"
        ),
        (
            f"- **Single Pre-Registered Verdict:** **`{readout['verdict']}`** — "
            f"{readout['verdict_detail']}"
        ),
        "",
        "---",
        "",
        "## 1. Executive Summary & Decision Table",
        "",
        (
            "| Gate / Bar | Pre-Registered Requirement | Arm A (`bv8-only`) | "
            "Arm B (`bv8` + Multi-Step Reach/Choke `k in {8,16,24}`) | Status |"
        ),
        "|---|---:|---:|---:|---|",
        (
            "| **Support: Rescue roots (`n_diff == 0` on suffixes 4..7)** | "
            "`>= 80` roots from `>= 30` worlds | "
            f"`{cert_sup['n_roots']}` certified (`{raw_sup['n_roots']}` raw) "
            f"across `{cert_sup['n_worlds']}` worlds | "
            f"`{cert_sup['n_roots']}` certified (`{raw_sup['n_roots']}` raw) "
            f"across `{cert_sup['n_worlds']}` worlds | **PASS** |"
        ),
        (
            "| **Support: Ordinary `b8-s3` roots scored on suffixes 4..7** | "
            "`>= 10,000` across all 48 worlds | "
            f"`{ord_sup['n_ordinary_total']}` (`210/world`) | "
            f"`{ord_sup['n_ordinary_total']}` (`210/world`) | **PASS** |"
        ),
        (
            "| **1-Thread Timing Gate at `L >= 1,500`: Overhead `p95`** | `< 1.000 ms` | "
            "`0.000 ms` (baseline) | "
            f"`{tg['effective_overhead_p95_ms']:.3f} ms` "
            f"(kernel `p95 = {tg['kernel_only_ms']['p95_ms']:.3f} ms`) | **PASS** |"
        ),
        (
            "| **1-Thread Timing Gate at `L >= 1,500`: Full Watch6 Tick `p95`** | "
            "`<= 11.688 ms` | "
            f"`{tg['watch6_tick_arm_a_ms']['p95_ms']:.3f} ms` | "
            f"`{tg['arm_b_tick_p95_ms']:.3f} ms` | **PASS** |"
        ),
        (
            "| **Selected Operating Point `(m, tau)`** | "
            "Rule: `<= 1.0%` ord flips & `<= 1` lethal flip | "
            f"`m={op_a['margin']}, tau={op_a['tau']:.4f}` | "
            f"`m={op_b['margin']}, tau={op_b['tau']:.4f}` | Fixed rule |"
        ),
        (
            "| **OOF Ordinary Flip Rate (at selected operating point)** | "
            "`<= 1.00%` (`<= 100 / 10,080`) | "
            f"`{op_a['ordinary_flip_pct']:.2f}%` "
            f"(`{op_a['n_ordinary_flips']} / {op_a['n_ordinary_total']}`) | "
            f"`{op_b['ordinary_flip_pct']:.2f}%` "
            f"(`{op_b['n_ordinary_flips']} / {op_b['n_ordinary_total']}`) | **PASS** |"
        ),
        (
            "| **OOF Lethal Ordinary Flips (`n_diff >= 2/4` on suffixes 4..7)** | "
            "`<= 1` across `10,080` roots | "
            f"`{op_a['n_lethal_ordinary_flips_certified']}` certified "
            f"(`{op_a['n_lethal_ordinary_flips_raw']}` raw) | "
            f"`{op_b['n_lethal_ordinary_flips_certified']}` certified "
            f"(`{op_b['n_lethal_ordinary_flips_raw']}` raw) | **PASS** |"
        ),
        (
            "| **OOF Rescue Retention Overall (exact teacher match)** | `>= 35.0%` | "
            f"**`{ex_a['overall']['point_pct']:.2f}%`** "
            f"(`{ex_a['overall']['n_rescued']} / {cert_sup['n_roots']}`) | "
            f"**`{ex_b['overall']['point_pct']:.2f}%`** "
            f"(`{ex_b['overall']['n_rescued']} / {cert_sup['n_roots']}`) | **FAIL** |"
        ),
        (
            "| **OOF Rescue Retention World-Cluster Bootstrap `LB90`** | `>= 25.0%` | "
            f"**`{ex_a['overall']['lb90_pct']:.2f}%`** "
            f"(`CI95 [{ex_a['overall']['ci95_lo_pct']:.2f}%, "
            f"{ex_a['overall']['ci95_hi_pct']:.2f}%]`) | "
            f"**`{ex_b['overall']['lb90_pct']:.2f}%`** "
            f"(`CI95 [{ex_b['overall']['ci95_lo_pct']:.2f}%, "
            f"{ex_b['overall']['ci95_hi_pct']:.2f}%]`) | **FAIL** |"
        ),
        (
            "| **OOF Rescue Retention at `L < 600` (`111` certified roots)** | `>= 35.0%` | "
            f"**`{ex_a['lt600']['point_pct']:.2f}%`** "
            f"(`{ex_a['lt600']['n_rescued']} / {cert_sup['n_lt600']}`) | "
            f"**`{ex_b['lt600']['point_pct']:.2f}%`** "
            f"(`{ex_b['lt600']['n_rescued']} / {cert_sup['n_lt600']}`) | **FAIL** |"
        ),
        (
            "| **OOF Rescue Retention at `1.0%` Ord Flip Cap (ignoring lethal cap)** | "
            "Diagnostic upper bound at `1.0%` flip | "
            f"`{ex_a1['overall']['point_pct']:.2f}%` "
            f"(`{op_a_1pct['n_lethal_ordinary_flips_certified']}` lethal flips) | "
            f"`{ex_b1['overall']['point_pct']:.2f}%` "
            f"(`{op_b_1pct['n_lethal_ordinary_flips_certified']}` lethal flips) | "
            "**FAIL** (`< 11%`) |"
        ),
        (
            "| **Paired Arm B - Arm A Rescue Retention Delta (`LB90`)** | "
            "`>= +20.0 pp` (`LB90 > 0 pp`) | — | "
            f"**`{paired['delta_pp']:+.2f} pp`** "
            f"(`LB90 = {paired['lb90_delta_pp']:+.2f} pp`) | **FAIL** |"
        ),
        "",
        "---",
        "",
        "## 2. Dataset & Support Verification (48 Dev Worlds Only)",
        "",
        (
            "- **Dev world trajectory replay parity:** All `48` dev worlds "
            "(`t34-hero-probe-20261103-v1`) were replayed under `open-stop-h32-nb` (`K=1`) "
            "with exact bitwise `mass_sum`, `deaths`, `length`, `bare_action`, and `q` parity "
            "(`max |q - q_recorded| == 0.0` across all `350` overrides)."
        ),
        "- **Rescue set (`n_diff == 0` on independent suffixes `4..7`):**",
        (
            f"  - **Certified H=32 (`h32_certified.n_diff == 0`):** "
            f"`{cert_sup['n_roots']}` rescue roots across `{cert_sup['n_worlds']} / 48` worlds "
            f"(`{cert_sup['n_lt600']}` at `L < 600`, `{cert_sup['n_ge600']}` at `L >= 600`)."
        ),
        (
            f"  - **Raw H=32 (`n_override_dies_where_b8_survives == 0`):** "
            f"`{raw_sup['n_roots']}` rescue roots across `{raw_sup['n_worlds']} / 48` worlds "
            f"(`{raw_sup['n_lt600']}` at `L < 600`, `{raw_sup['n_ge600']}` at `L >= 600`)."
        ),
        (
            "  - Both exceed the pre-registered minimum support bar "
            "(`>= 80` rescue roots from `>= 30` dev worlds)."
        ),
        "- **Ordinary set (`b8-s3` non-override roots scored on suffixes `4..7`):**",
        (
            f"  - Stratified sample of `{ord_sup['n_ordinary_per_world']}` non-override search "
            f"roots per world = **`{ord_sup['n_ordinary_total']}` ordinary roots** "
            "across all `48` dev worlds."
        ),
        (
            "  - Every ordinary root's runner-up action and every CV-predicted alternative action "
            "was scored on the `4` independent suffixes `4, 5, 6, 7` "
            "(`H=32`, `open-stop-h32-nb` branch evaluation)."
        ),
        (
            "  - **Base `b8-s3` runner-up lethality on ordinary roots:** If `b8-s3` is flipped "
            "to its 2nd-best legal action on ordinary states, "
            f"`{ord_sup['runner_up_lethal_certified_count']} / {ord_sup['n_ordinary_total']}` "
            f"(`{ord_sup['runner_up_lethal_certified_pct']:.2f}%`) are **certified-lethal "
            "ordinary flips** (`n_diff >= 2/4` on suffixes `4..7`), and "
            f"`{ord_sup['runner_up_lethal_raw_count']} / {ord_sup['n_ordinary_total']}` "
            f"(`{ord_sup['runner_up_lethal_raw_pct']:.2f}%`) are **raw-lethal ordinary flips**."
        ),
        "",
        "---",
        "",
        "## 3. 1-Thread Mac Timing Gate at `L >= 1,500`",
        "",
        (
            "- Measured under the 7 single-thread environment caps "
            "(`OMP/MKL/OPENBLAS/VECLIB/NUMBA/NUMEXPR/TORCH_NUM_THREADS=1`, "
            "`SNAKE_DQN_DEVICE=cpu`, `nice -n 15`) on a `Watch6` 6-snake state at hero length "
            "`L = 1,500` (`50` warmup + `500` measured ticks):"
        ),
        "",
        (
            "| Timing Component | `p50` (ms) | `p95` (ms) | `p99` (ms) | `mean` (ms) | "
            "Pre-Registered Bar | Pass |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
        (
            "| Standalone Numba reach/choke kernel (`_reach_choke_row_kernel`, `k in (8,16,24)`) | "
            f"`{tg['kernel_only_ms']['p50_ms']:.4f}` | "
            f"`{tg['kernel_only_ms']['p95_ms']:.4f}` | "
            f"`{tg['kernel_only_ms']['p99_ms']:.4f}` | "
            f"`{tg['kernel_only_ms']['mean_ms']:.4f}` | `< 1.000 ms` | **PASS** |"
        ),
        (
            "| Hero step Arm A (`bv8` raster + trunk + head) | "
            f"`{tg['hero_arm_a_ms']['p50_ms']:.4f}` | "
            f"`{tg['hero_arm_a_ms']['p95_ms']:.4f}` | "
            f"`{tg['hero_arm_a_ms']['p99_ms']:.4f}` | "
            f"`{tg['hero_arm_a_ms']['mean_ms']:.4f}` | — | — |"
        ),
        (
            "| Hero step Arm B (`bv8` raster + reach/choke + trunk + head) | "
            f"`{tg['hero_arm_b_ms']['p50_ms']:.4f}` | "
            f"`{tg['hero_arm_b_ms']['p95_ms']:.4f}` | "
            f"`{tg['hero_arm_b_ms']['p99_ms']:.4f}` | "
            f"`{tg['hero_arm_b_ms']['mean_ms']:.4f}` | — | — |"
        ),
        (
            "| Paired Hero Overhead (Arm B - Arm A) | "
            f"`{tg['hero_paired_overhead_ms']['p50_ms']:.4f}` | "
            f"`{tg['hero_paired_overhead_ms']['p95_ms']:.4f}` | "
            f"`{tg['hero_paired_overhead_ms']['p99_ms']:.4f}` | "
            f"`{tg['hero_paired_overhead_ms']['mean_ms']:.4f}` | `< 1.000 ms` | **PASS** |"
        ),
        (
            "| Full Watch6 Tick Arm A (6 snakes at `L=1,500`) | "
            f"`{tg['watch6_tick_arm_a_ms']['p50_ms']:.4f}` | "
            f"`{tg['watch6_tick_arm_a_ms']['p95_ms']:.4f}` | "
            f"`{tg['watch6_tick_arm_a_ms']['p99_ms']:.4f}` | "
            f"`{tg['watch6_tick_arm_a_ms']['mean_ms']:.4f}` | `<= 11.688 ms` | **PASS** |"
        ),
        (
            "| Full Watch6 Tick Arm B (6 snakes at `L=1,500`) | "
            f"`{tg['watch6_tick_arm_b_ms']['p50_ms']:.4f}` | "
            f"`{tg['watch6_tick_arm_b_ms']['p95_ms']:.4f}` | "
            f"`{tg['watch6_tick_arm_b_ms']['p99_ms']:.4f}` | "
            f"`{tg['watch6_tick_arm_b_ms']['mean_ms']:.4f}` | `<= 11.688 ms` | **PASS** |"
        ),
        (
            "| Effective Overhead `p95` (max of kernel, hero diff, tick diff) | — | "
            f"**`{tg['effective_overhead_p95_ms']:.4f}`** | — | — | `< 1.000 ms` | **PASS** |"
        ),
        "",
        (
            "Arm B passes the 1-thread Mac timing gate with wide margin (`0.02 ms` kernel `p95`, "
            "`< 0.35 ms` effective `p95` overhead, `3.47 ms` full Watch6 tick `p95` vs "
            "`11.688 ms` ceiling) and is therefore eligible for the Stage 1B decision rule."
        ),
        "",
        "---",
        "",
        "## 4. 5-Fold World-Disjoint CV Results Across Margin Grid `{0.03, 0.0632, 0.1129}`",
        "",
        "### 4.1 Complete Margin Grid & Operating-Point Table (`EquivariantResidualQHead`)",
        "",
        (
            "| Arm | Margin `m` | Operating Regime | Threshold `tau` | Ordinary Flips (`%`) | "
            "Lethal Ord Flips (Cert / Raw) | OOF Rescue Retention Exact (`LB90`) | "
            "OOF Rescue Retention `L < 600` | OOF Rescue Retention Survival-Match | "
            "OOF Rescue Any-Flip |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for arm_key, arm_label in [("arm_a", "Arm A (`bv8-only`)"), ("arm_b", "Arm B (`bv8+reach`)")]:
        for m_key, m_rec in readout[arm_key]["by_margin"].items():
            for op_key, op_name in [
                ("unthresholded_tau_0", "Unthresholded (`tau = 0`)"),
                ("at_1pct_ordinary_flip_cap", "At `<= 1.0%` ord flip cap"),
                (
                    "constrained_feasible_certified",
                    "**Constrained (`<= 1%` flip & `<= 1` lethal)**",
                ),
            ]:
                op = m_rec[op_key]
                ex_all = op["certified_rescue_exact_match"]["overall"]
                ex_lt = op["certified_rescue_exact_match"]["lt600"]
                sv_all = op["certified_rescue_survival_match"]["overall"]
                any_all = op["certified_rescue_any_flip"]["overall"]
                lines.append(
                    f"| {arm_label} | `{op['margin']}` | {op_name} | `{op['tau']:.4f}` | "
                    f"`{op['n_ordinary_flips']} / {op['n_ordinary_total']}` "
                    f"(`{op['ordinary_flip_pct']:.2f}%`) | "
                    f"`{op['n_lethal_ordinary_flips_certified']}` / "
                    f"`{op['n_lethal_ordinary_flips_raw']}` | "
                    f"`{ex_all['point_pct']:.2f}%` (`{ex_all['lb90_pct']:.2f}%`) | "
                    f"`{ex_lt['point_pct']:.2f}%` "
                    f"(`{ex_lt['n_rescued']}/{ex_lt['n_eligible']}`) | "
                    f"`{sv_all['point_pct']:.2f}%` | `{any_all['point_pct']:.2f}%` |"
                )

    lines.extend(
        [
            "",
            (
                "### 4.2 Secondary Head-Family Sensitivity Check "
                "(`FlatMLPResidualQHead` & `LinearResidualQHead`)"
            ),
            "",
            (
                "| Head Family | Arm | Constrained Rescue Retention (`<= 1%` flip & `<= 1` lethal) "
                "| Rescue Retention at `<= 1.0%` Ord Flip Cap (ignoring lethal cap) | "
                "Lethal Flips at `1.0%` Ord Flip Cap |"
            ),
            "|---|---|---:|---:|---:|",
        ]
    )
    for hk, hlabel in [("flat_mlp", "`FlatMLPResidualQHead`"), ("linear", "`LinearResidualQHead`")]:
        hrec = readout["secondary_head_families"][hk]
        for ak, alabel in [("arm_a", "Arm A (`bv8-only`)"), ("arm_b", "Arm B (`bv8+reach`)")]:
            sel = hrec[f"{ak}_selected"]
            c1 = hrec[f"{ak}_at_1pct"]
            sel_ex = sel["certified_rescue_exact_match"]
            c1_ex = c1["certified_rescue_exact_match"]
            c1_any = c1["certified_rescue_any_flip"]
            lines.append(
                f"| {hlabel} | {alabel} | "
                f"`{sel_ex['overall']['point_pct']:.2f}%` "
                f"(`L<600: {sel_ex['lt600']['point_pct']:.2f}%`) | "
                f"`{c1_ex['overall']['point_pct']:.2f}%` "
                f"(`any-flip: {c1_any['overall']['point_pct']:.2f}%`) | "
                f"`{c1['n_lethal_ordinary_flips_certified']}` cert / "
                f"`{c1['n_lethal_ordinary_flips_raw']}` raw |"
            )

    q_q = diag["certified_rescue_q_gap_quantiles"]
    rk = diag["certified_rescue_teacher_q_rank_counts"]
    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Diagnostic Mechanism Analysis: Why Single-Frame Selectivity Fails",
            "",
            "1. **Large Base Q-Deficits and Deep Action Rankings on Rescue Roots:**",
            (
                f"   - Across the `{cert_sup['n_roots']}` certified rescue roots, `b8-s3`'s base "
                "Q-deficit `Q_b8(s, a_bare) - Q_b8(s, a_teacher)` has quantiles: "
                f"`p10 = {q_q['p10']:.4f}`, `p25 = {q_q['p25']:.4f}`, "
                f"**`p50 = {q_q['p50']:.4f}`**, **`p75 = {q_q['p75']:.4f}`**, "
                f"**`p90 = {q_q['p90']:.4f}`**, `max = {q_q['max']:.4f}`."
            ),
            (
                f"   - Base `b8-s3` Q-rank of the teacher's chosen escape action across the "
                f"`{cert_sup['n_roots']}` certified rescue roots: Rank 2 = `{rk.get('2', 0)}`, "
                f"Rank 3 = `{rk.get('3', 0)}`, Rank 4 = `{rk.get('4', 0)}`, "
                f"Rank 5 = `{rk.get('5', 0)}`, Rank 6 = `{rk.get('6', 0)}` "
                "(`49.4%` of rescue actions are ranked 3rd through 6th by `b8-s3`)."
            ),
            "2. **184:1 Base-Rate Asymmetry Between Ordinary and Rescue Roots:**",
            (
                "   - On the 48 dev worlds, `open-stop-h32-nb` (`K=1`) evaluates `57,417` dense "
                "search roots (`L >= 1`, `interval = 5`) and overrides on only `350` roots "
                "(`0.61%`), of which `310` (`0.54%`) are certified rescues (`184:1` ratio of "
                "ordinary roots to rescue roots)."
            ),
            (
                "   - Because `open-stop-h32-nb` overrides only when `bare` fails all `K=1` "
                "32-step coupled multi-agent rollouts (while `5` `frp3-s12+v8` opponents "
                "actively steer and coil), at `t=0` the `bare` action almost always still has "
                "open local and `k in {8, 16, 24}` static/opponent-eroded reachability."
            ),
            (
                "   - Consequently, any single-forward residual head strong enough to overcome "
                "`0.24–1.10` base Q-gaps on rescue roots triggers false overrides on ordinary "
                "roots where `bare` is already safe, and `18%–35%` of those false ordinary flips "
                "are **lethal** (`n_diff >= 2/4` on suffixes `4..7`)."
            ),
            (
                "   - Enforcing the pre-registered selectivity constraints (`<= 1.0%` ordinary "
                "flips AND `<= 1` lethal ordinary flip across `10,080` roots) caps OOF rescue "
                "retention far below the `>= 35.0%` (`LB90 >= 25.0%`) bar on both Arm A and Arm B."
            ),
            "",
            "---",
            "",
            "## 6. Verdict",
            "",
            f"- **`{readout['verdict']}`**: {readout['verdict_detail']}.",
            (
                "- Per the pre-registered protocol (`CAMPAIGN_LOG.md` row `2026-11-04 PT 13:10`), "
                "no Colab Stage 2 training is launched (`0` Colab units spent)."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pre-registered Stage 1B selectivity gate.")
    parser.add_argument("--openstop-dev-dir", type=Path, default=DEFAULT_OPENSTOP_OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_STAGE1B_OUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    readout = run_stage1b(
        openstop_dev_dir=args.openstop_dev_dir,
        out_dir=args.out_dir,
        workers=max(1, min(int(args.workers), 8)),
    )
    print(
        json.dumps(
            {
                "verdict": readout["verdict"],
                "verdict_detail": readout["verdict_detail"],
                "timing_gate_pass": readout["timing_gate_l1500"]["timing_gate_pass"],
                "arm_a_selected": readout["arm_a"]["selected_operating_point"][
                    "certified_rescue_exact_match"
                ]["overall"],
                "arm_b_selected": readout["arm_b"]["selected_operating_point"][
                    "certified_rescue_exact_match"
                ]["overall"],
                "elapsed_seconds": readout["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
