"""Causal t=0 root feature extraction for R6 Watch6-v2 L<600 separability census.

Extracts:
1. Standard ``ego2s-bv8`` features: 256-D ``phi`` (``Ego2sNet.fuse``), 6-D ``q_b8``,
   and 60-D ``scalars_bv8`` (30 ``ego2s-b`` scalars + 30 ``v8`` features).
2. ``+2`` ego-rotated ``37x37`` (``8x8``-cell block) global planes separating
   FIFO-age-weighted ``DEATH_DROP`` corpse mass from ``BOOST_PELLET`` trail mass
   using exact ``food_cells`` FIFO order and ``ProvenanceObserver`` side state.
3. ``4`` FIFO/cluster/race scalars:
   - ``fifo_fill``: active corpse-FIFO occupancy ``n_corpse / 150.0``.
   - ``top_cluster_mass``: richest ``2x2`` coarse (``16x16``-cell) FIFO-weighted
     ``DEATH_DROP`` cluster mass.
   - ``top_cluster_forward_cos``: ego-frame forward cosine to that cluster centroid
     (plus auxiliary ``ego_dx_norm``, ``ego_dy_norm``).
   - ``enemy_arrival_margin``: ``(min_enemy_manhattan - hero_manhattan) / 60.0``
     to the top ``DEATH_DROP`` cluster centroid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from research.t34_o1_harvest.provenance import BOOST_PELLET, DEATH_DROP, ProvenanceObserver
from src.core.mechanics_constants import corpse_food_cap
from src.model.ego2s_network import Ego2sNet, obs_tensors
from src.simd_env.ego_raster import (
    CARDINAL,
    EgoRasterConfig,
    _crop_index,
    _offset_table,
    build_ego_raster,
)


def extract_fifo_corpse_global_planes_and_scalars(
    sim: Any,
    observer: ProvenanceObserver,
    *,
    env: int = 0,
    hero: int = 0,
    global_size: int = 37,
    global_cell: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ego-rotated ``(2, 37, 37)`` FIFO-weighted corpse/trail planes + 4 scalars.

    Args:
        sim: Prepared ``GridBatchSim`` instance.
        observer: Attached ``ProvenanceObserver`` tracking ``DEATH_DROP`` vs ``BOOST_PELLET``.
        env: Environment index (default 0).
        hero: Hero snake slot (default 0).
        global_size: Ego global grid width/height (default 37).
        global_cell: Coarse block size in arena cells (default 8).

    Returns:
        Tuple ``(planes_2_37_37, pooled_28, scalars_4, aux_4)`` of float32 arrays.
    """
    H = int(sim.grid_h)
    W = int(sim.grid_w)
    hc = -(-H // global_cell)
    wc = -(-W // global_cell)
    fifo_cap = max(1, int(corpse_food_cap(sim.cfg.max_food)))

    corpse_set = sim.corpse_cells[env]
    # Exact FIFO order in GridBatchSim: food_cells list order restricted to corpse_cells.
    ordered_corpse = [cell for cell in sim.food_cells[env] if cell in corpse_set]
    n_corpse = len(ordered_corpse)
    fifo_fill = float(n_corpse) / float(fifo_cap)

    drop_coarse = np.zeros((hc, wc), dtype=np.float32)
    trail_coarse = np.zeros((hc, wc), dtype=np.float32)
    drop_sum_xy = np.zeros((hc, wc, 2), dtype=np.float32)
    n_drop = 0
    n_trail = 0

    for fifo_idx, cell in enumerate(ordered_corpse):
        col, row = int(cell[0]), int(cell[1])
        # Remaining FIFO slots before eviction normalized to (0, 1].
        rem_slots = max(1, fifo_cap - n_corpse + 1 + fifo_idx)
        fifo_weight = min(1.0, float(rem_slots) / float(fifo_cap))
        src = observer._sources.get((env, (col, row)))
        br = min(hc - 1, max(0, row // global_cell))
        bc = min(wc - 1, max(0, col // global_cell))
        if src == DEATH_DROP:
            drop_coarse[br, bc] += fifo_weight
            drop_sum_xy[br, bc, 0] += fifo_weight * float(col)
            drop_sum_xy[br, bc, 1] += fifo_weight * float(row)
            n_drop += 1
        elif src == BOOST_PELLET:
            trail_coarse[br, bc] += fifo_weight
            n_trail += 1

    # Rotate coarse grids into ego (2, 37, 37) with exact Ego2sNet _global_planes geometry.
    half = global_size // 2
    pw = ((half, half), (half, half))
    drop_pad = np.pad(drop_coarse, pw, mode="constant", constant_values=0.0)
    trail_pad = np.pad(trail_coarse, pw, mode="constant", constant_values=0.0)
    stride = wc + 2 * half
    table = _offset_table(global_size, half, half, stride)

    heads = sim.heads()
    heads_c = (heads[env : env + 1, hero : hero + 1] // global_cell).astype(np.int64)
    alive_arr = np.asarray(sim.alive[env : env + 1, hero : hero + 1], dtype=bool)
    dir_arr = np.asarray(sim.direction[env : env + 1, hero : hero + 1], dtype=np.int64)
    idx = _crop_index(heads_c, alive_arr, dir_arr, half, stride, table, (wc // 2, hc // 2))[0, 0]

    drop_ego = drop_pad.ravel()[idx].reshape(global_size, global_size).astype(np.float32)
    trail_ego = trail_pad.ravel()[idx].reshape(global_size, global_size).astype(np.float32)
    if not bool(alive_arr[0, 0]):
        drop_ego.fill(0.0)
        trail_ego.fill(0.0)
    planes = np.stack([drop_ego, trail_ego], axis=0)

    # Richest 2x2 coarse (16x16-cell) DEATH_DROP cluster + ego projection + enemy race.
    hero_head = heads[env, hero].astype(np.float32)  # (col, row)
    heading = int(sim.direction[env, hero])
    dx_h, dy_h = float(CARDINAL[heading, 0]), float(CARDINAL[heading, 1])
    rx_h, ry_h = -dy_h, dx_h

    top_cluster_mass = 0.0
    forward_cos = 0.0
    ego_dx_norm = 0.0
    ego_dy_norm = 0.0
    enemy_arrival_margin = 0.0

    if drop_coarse.sum() > 0.0:
        if hc >= 2 and wc >= 2:
            win = (
                drop_coarse[:-1, :-1]
                + drop_coarse[1:, :-1]
                + drop_coarse[:-1, 1:]
                + drop_coarse[1:, 1:]
            )
            flat_best = int(np.argmax(win))
            r0, c0 = divmod(flat_best, wc - 1)
            r1, c1 = r0 + 2, c0 + 2
        else:
            r0, r1, c0, c1 = 0, hc, 0, wc
        sub_mass = drop_coarse[r0:r1, c0:c1]
        top_cluster_mass = float(sub_mass.sum())
        if top_cluster_mass > 1e-6:
            sub_xy = drop_sum_xy[r0:r1, c0:c1].sum(axis=(0, 1)) / top_cluster_mass
            d_col = float(sub_xy[0] - hero_head[0])
            d_row = float(sub_xy[1] - hero_head[1])
            ahead = d_col * dx_h + d_row * dy_h
            right = d_col * rx_h + d_row * ry_h
            norm = float(np.hypot(ahead, right))
            if norm > 1e-6:
                forward_cos = float(ahead / norm)
            ego_dx_norm = float(ahead / 60.0)
            ego_dy_norm = float(right / 60.0)

            hero_l1 = abs(d_col) + abs(d_row)
            min_enemy_l1 = 240.0
            for slot in range(1, int(sim.S)):
                if bool(sim.alive[env, slot]):
                    eh = heads[env, slot]
                    l1 = abs(float(sub_xy[0]) - float(eh[0])) + abs(float(sub_xy[1]) - float(eh[1]))
                    if l1 < min_enemy_l1:
                        min_enemy_l1 = l1
            enemy_arrival_margin = float((min_enemy_l1 - hero_l1) / 60.0)

    scalars_4 = np.array(
        [fifo_fill, top_cluster_mass, forward_cos, enemy_arrival_margin], dtype=np.float32
    )
    aux_4 = np.array([ego_dx_norm, ego_dy_norm, float(n_drop), float(n_trail)], dtype=np.float32)
    pooled_28 = _pool_ego_planes(planes, half=half)
    return planes, pooled_28, scalars_4, aux_4


def _pool_ego_planes(planes: np.ndarray, *, half: int = 18) -> np.ndarray:
    """Compute 14 directional/radial/cone spatial summaries per 37x37 ego plane (28 total).

    In Ego2sNet's ego coordinates, pixel ``(r, c)`` has ``ahead = half - r`` and
    ``right = c - half`` (in 8-cell coarse blocks).
    """
    size = planes.shape[1]
    rr, cc = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    ahead = (half - rr).astype(np.float32)
    right = (cc - half).astype(np.float32)
    dist = np.maximum(np.abs(ahead), np.abs(right))

    is_ahead = (ahead > 0) & (np.abs(right) <= ahead)
    is_left = (right < 0) & (-right >= np.abs(ahead))
    is_right = (right > 0) & (right >= np.abs(ahead))
    is_behind = ahead < -np.abs(right)

    near = (dist >= 0) & (dist <= 2)
    mid = (dist > 2) & (dist <= 5)
    far = dist > 5

    # Action-aligned inverse-distance cones for turn-left (0/3), straight (1/4), right (2/5).
    inv_d = 1.0 / (1.0 + np.hypot(ahead, right))
    cone_left = (right < 0) & (ahead >= -2)
    cone_straight = (ahead > 0) & (np.abs(right) <= np.maximum(1.0, 0.75 * ahead))
    cone_right = (right > 0) & (ahead >= -2)

    feats: list[float] = []
    for ch in range(planes.shape[0]):
        p = planes[ch]
        feats.extend(
            [
                float(p.sum()),
                float(p[is_ahead & near].sum()),
                float(p[is_ahead & mid].sum()),
                float(p[is_ahead & far].sum()),
                float(p[is_left & near].sum()),
                float(p[is_left & mid].sum()),
                float(p[is_left & far].sum()),
                float(p[is_right & near].sum()),
                float(p[is_right & mid].sum()),
                float(p[is_right & far].sum()),
                float(p[is_behind].sum()),
                float((p * inv_d)[cone_left].sum()),
                float((p * inv_d)[cone_straight].sum()),
                float((p * inv_d)[cone_right].sum()),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


@torch.inference_mode()
def extract_root_features(world: Any, observer: ProvenanceObserver) -> dict[str, Any]:
    """Extract causal t=0 ``ego2s-bv8`` + corpse/FIFO features for hero slot 0."""
    sim = world.sim
    obs = build_ego_raster(
        sim,
        EgoRasterConfig(version="bv8"),
        backend="numba",
        rows=np.array([[0, 0]], dtype=np.int64),
    )
    net: Ego2sNet = world.hero.net
    dev = next(net.parameters()).device
    local_t, glob_t, scalars_t = obs_tensors(obs, dev)
    local_f = local_t.float() * (1.0 / 255.0)
    glob_f = glob_t.float() * (1.0 / 255.0)
    h_loc = net.local(local_f)
    h_glob = net.glob(glob_f)
    phi = net.fuse(torch.cat([h_loc, h_glob, scalars_t.float()], dim=1))
    adv = net.adv(phi)
    q = net.value(phi) + adv - adv.mean(dim=1, keepdim=True)

    planes, pooled_28, scalars_4, aux_4 = extract_fifo_corpse_global_planes_and_scalars(
        sim, observer, env=0, hero=0
    )
    return {
        "hero_length": int(sim.length[0, 0]),
        "phi": phi[0].detach().cpu().numpy().astype(np.float32),
        "q_b8": q[0].detach().cpu().numpy().astype(np.float32),
        "scalars_bv8": np.asarray(obs["scalars"][0], dtype=np.float32).copy(),
        "corpse_trail_planes": planes,
        "corpse_trail_pooled": pooled_28,
        "corpse_fifo_scalars": scalars_4,
        "corpse_fifo_aux": aux_4,
    }
