"""Multi-scale heading-aligned ego-raster and scalar observations for Game V3.

Provides both the zero-host-sync batched PyTorch GPU builder (:func:`build_obs`)
and an independent plain-NumPy single-world reference (:func:`build_obs_reference`).

Sampling & Coordinate Specification (identical on CPU and GPU):
- Ego frame is centered on each snake's head ``(hx, hy)`` with **up** (decreasing
  raster ``row``, ``row = 0`` ahead) aligned to ``fwd = (cos(theta), sin(theta))``
  and **right** (increasing raster ``col``, ``col = S - 1`` right) aligned to
  ``right = (sin(theta), -cos(theta))``.
- For raster size ``S = 64`` and scale ``s in {1, 4, 16}`` units/cell, pixel
  ``(row, col)`` has ego-center offset:
  ``ego_x = (col - 31.5) * s``, ``ego_y = (31.5 - row) * s``,
  and world sample point ``(wx, wy) = head_pos + ego_x * right + ego_y * fwd``.
- ``own_body`` and ``enemy_body`` use nearest-cell sampling on the ``1x`` occupancy
  grid at ``gx = floor(wx / c) + half_grid``, ``gy = floor(wy / c) + half_grid``.
- ``ambient_food`` and ``corpse_food`` use nearest-cell sampling on scale-matched
  exact block-summed food grids (``1x1`` at scale 1, ``4x4`` sum at scale 4,
  ``16x16`` sum at scale 16) normalized via ``m / (1.0 + m)``.
- ``out_of_arena`` is ``1.0`` where ``wx^2 + wy^2 >= arena_radius^2`` else ``0.0``.
- Enemy head channels (current tick ``t`` in channels ``5..8`` and previous tick
  ``t - 1`` in channels ``9..12``) bin each alive enemy head directly into its ego
  pixel ``col = floor(ego_x / s + S / 2)``, ``row = floor(S / 2 - ego_y / s)``;
  when multiple enemy heads map to the same coarse pixel, the lowest slot index
  wins deterministically.
- Dead observer lanes emit all zeros across rasters and scalars.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass

import numpy as np
import torch

from research.game_v3.config import (
    COS_C2,
    COS_C4,
    COS_C6,
    COS_C8,
    COS_C10,
    HALF_PI_F32,
    PI_F32,
    SIN_C3,
    SIN_C5,
    SIN_C7,
    SIN_C9,
    GameV3Config,
)
from research.game_v3.reference import ReferenceWorld
from research.game_v3.sim import GameV3Sim, poly_sincos_torch

CH_OWN_BODY: int = 0
CH_ENEMY_BODY: int = 1
CH_AMBIENT_FOOD: int = 2
CH_CORPSE_FOOD: int = 3
CH_OUT_OF_ARENA: int = 4
CH_ENEMY_HEADS_T0: int = 5
CH_ENEMY_DX_T0: int = 6
CH_ENEMY_DY_T0: int = 7
CH_ENEMY_SIZE_T0: int = 8
CH_ENEMY_HEADS_T1: int = 9
CH_ENEMY_DX_T1: int = 10
CH_ENEMY_DY_T1: int = 11
CH_ENEMY_SIZE_T1: int = 12

NUM_RASTER_CHANNELS: int = 13
NUM_SCALAR_FEATURES: int = 5


@dataclass
class ObservationBatch:
    """Batched multi-scale observations on a single Torch device."""

    raster_s1: torch.Tensor
    raster_s4: torch.Tensor
    raster_s16: torch.Tensor
    scalars: torch.Tensor
    action_mask: torch.Tensor


@dataclass
class ReferenceObservation:
    """Single-world NumPy reference observations matching :class:`ObservationBatch`."""

    raster_s1: np.ndarray
    raster_s4: np.ndarray
    raster_s16: np.ndarray
    scalars: np.ndarray
    action_mask: np.ndarray


class _ObsStaticCache:
    """Device-resident static lookup tables cached per GameV3Sim instance."""

    def __init__(self, sim: GameV3Sim) -> None:
        cfg = sim.cfg
        dev = sim.device
        sz = cfg.obs_raster_size
        half_r = float(np.float32(0.5 * sz - 0.5))
        idx_1d = torch.arange(sz, dtype=torch.float32, device=dev)

        self.ego_x_by_scale: list[torch.Tensor] = []
        self.ego_y_by_scale: list[torch.Tensor] = []
        for s in cfg.obs_scales:
            s_f32 = float(np.float32(s))
            self.ego_x_by_scale.append((idx_1d - half_r) * s_f32)
            self.ego_y_by_scale.append((half_r - idx_1d) * s_f32)

        denom_log = math.log2(1.0 + cfg.max_length)
        len_log_np = np.array(
            [
                np.float32(math.log2(1.0 + length) / denom_log)
                for length in range(cfg.max_length + 1)
            ],
            dtype=np.float32,
        )
        self.len_log_table = torch.as_tensor(len_log_np, device=dev)
        self.tril_nnn = torch.tril(
            torch.ones((sim.N, sim.N), dtype=torch.bool, device=dev), diagonal=-1
        )[None, None, :, :]


_OBS_CACHE: weakref.WeakKeyDictionary[GameV3Sim, _ObsStaticCache] = weakref.WeakKeyDictionary()


def _get_obs_cache(sim: GameV3Sim) -> _ObsStaticCache:
    cached = _OBS_CACHE.get(sim)
    if cached is None:
        cached = _ObsStaticCache(sim)
        _OBS_CACHE[sim] = cached
    return cached


def allocate_obs_batch(sim: GameV3Sim) -> ObservationBatch:
    """Preallocate an :class:`ObservationBatch` on ``sim.device`` for reuse."""
    sz = sim.cfg.obs_raster_size
    r_shape = (sim.E, sim.N, NUM_RASTER_CHANNELS, sz, sz)
    return ObservationBatch(
        raster_s1=torch.empty(r_shape, dtype=torch.float32, device=sim.device),
        raster_s4=torch.empty(r_shape, dtype=torch.float32, device=sim.device),
        raster_s16=torch.empty(r_shape, dtype=torch.float32, device=sim.device),
        scalars=torch.empty(
            (sim.E, sim.N, NUM_SCALAR_FEATURES), dtype=torch.float32, device=sim.device
        ),
        action_mask=torch.empty((sim.E, sim.N, 6), dtype=torch.bool, device=sim.device),
    )


def _splat_heads_torch(
    dst_4ch: torch.Tensor,
    *,
    obs_alive: torch.Tensor,
    obs_pos: torch.Tensor,
    obs_sin: torch.Tensor,
    obs_cos: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_pos: torch.Tensor,
    enemy_sin: torch.Tensor,
    enemy_cos: torch.Tensor,
    enemy_size_norm: torch.Tensor,
    not_eye: torch.Tensor,
    tril_nnn: torch.Tensor,
    scale: float,
    raster_size: int,
) -> None:
    """Project and splat 4 enemy head channels into ``dst_4ch`` ``(E, N, 4, S, S)``."""
    dst_4ch.zero_()
    e_cnt, n_cnt = obs_alive.shape
    num_pix = raster_size * raster_size
    inv_s = float(np.float32(1.0 / scale))
    half_sz = float(np.float32(0.5 * raster_size))

    rx = enemy_pos[:, None, :, 0] - obs_pos[:, :, None, 0]
    ry = enemy_pos[:, None, :, 1] - obs_pos[:, :, None, 1]

    sin_i = obs_sin[:, :, None]
    cos_i = obs_cos[:, :, None]
    ego_x = rx * sin_i - ry * cos_i
    ego_y = rx * cos_i + ry * sin_i

    col = torch.floor(ego_x * inv_s + half_sz).to(torch.int32)
    row = torch.floor(half_sz - ego_y * inv_s).to(torch.int32)

    valid = (
        obs_alive[:, :, None]
        & enemy_alive[:, None, :]
        & not_eye
        & (col >= 0)
        & (col < raster_size)
        & (row >= 0)
        & (row < raster_size)
    )
    pix = row.clamp(0, raster_size - 1).to(torch.int64) * raster_size + col.clamp(
        0, raster_size - 1
    ).to(torch.int64)

    # Deduplicate same-pixel collisions deterministically: lowest enemy slot j wins.
    same_pix = (
        valid[:, :, :, None]
        & valid[:, :, None, :]
        & (pix[:, :, :, None] == pix[:, :, None, :])
        & tril_nnn
    )
    keep = valid & (~same_pix.any(dim=-1))

    sin_j = enemy_sin[:, None, :]
    cos_j = enemy_cos[:, None, :]
    ego_dx = cos_j * sin_i - sin_j * cos_i
    ego_dy = cos_j * cos_i + sin_j * sin_i
    sz_j = enemy_size_norm[:, None, :].expand_as(ego_dx)

    zeros = torch.zeros_like(ego_dx)
    ones = torch.ones_like(ego_dx)

    flat_dst = dst_4ch.view(e_cnt * n_cnt, 4, num_pix)
    pix_flat = pix.view(e_cnt * n_cnt, n_cnt)

    flat_dst[:, 0, :].scatter_add_(
        1, pix_flat, torch.where(keep, ones, zeros).view(e_cnt * n_cnt, n_cnt)
    )
    flat_dst[:, 1, :].scatter_add_(
        1, pix_flat, torch.where(keep, ego_dx, zeros).view(e_cnt * n_cnt, n_cnt)
    )
    flat_dst[:, 2, :].scatter_add_(
        1, pix_flat, torch.where(keep, ego_dy, zeros).view(e_cnt * n_cnt, n_cnt)
    )
    flat_dst[:, 3, :].scatter_add_(
        1, pix_flat, torch.where(keep, sz_j, zeros).view(e_cnt * n_cnt, n_cnt)
    )


def _fill_scale_raster_torch(
    dst: torch.Tensor,
    sim: GameV3Sim,
    cache: _ObsStaticCache,
    scale_idx: int,
    sin_t0: torch.Tensor,
    cos_t0: torch.Tensor,
    size_norm_t0: torch.Tensor,
    sin_t1: torch.Tensor,
    cos_t1: torch.Tensor,
    size_norm_t1: torch.Tensor,
) -> None:
    """Populate all 13 channels of one scale raster ``(E, N, 13, S, S)`` in place."""
    cfg = sim.cfg
    sz = cfg.obs_raster_size
    num_pix = sz * sz
    scale = cfg.obs_scales[scale_idx]

    ego_x_1d = cache.ego_x_by_scale[scale_idx]
    ego_y_1d = cache.ego_y_by_scale[scale_idx]

    hx = sim.head_pos[:, :, 0]
    hy = sim.head_pos[:, :, 1]
    right_x = sin_t0
    right_y = -cos_t0
    fwd_x = cos_t0
    fwd_y = sin_t0

    wx_row = hx[:, :, None] + fwd_x[:, :, None] * ego_y_1d[None, None, :]
    wx_col = right_x[:, :, None] * ego_x_1d[None, None, :]
    wy_row = hy[:, :, None] + fwd_y[:, :, None] * ego_y_1d[None, None, :]
    wy_col = right_y[:, :, None] * ego_x_1d[None, None, :]

    wx = wx_row[:, :, :, None] + wx_col[:, :, None, :]
    wy = wy_row[:, :, :, None] + wy_col[:, :, None, :]

    alive_mask = sim.alive[:, :, None, None]
    r_sq_world = float(np.float32(cfg.arena_radius * cfg.arena_radius))
    out_arena = alive_mask & ((wx * wx + wy * wy) >= r_sq_world)
    dst[:, :, CH_OUT_OF_ARENA, :, :] = out_arena.to(torch.float32)

    inv_c = float(np.float32(1.0 / cfg.cell_size))
    gx = torch.floor(wx * inv_c).to(torch.int32) + sim.half_grid
    gy = torch.floor(wy * inv_c).to(torch.int32) + sim.half_grid
    in_grid = alive_mask & (gx >= 0) & (gx < sim.G) & (gy >= 0) & (gy < sim.G)

    gx_c = gx.clamp(0, sim.G - 1).to(torch.int64)
    gy_c = gy.clamp(0, sim.G - 1).to(torch.int64)
    flat_1x = (gy_c * sim.G + gx_c).view(sim.E, sim.N * num_pix)

    occ_sample = sim.occ_packed.view(sim.E, -1).gather(1, flat_1x).view(sim.E, sim.N, sz, sz)
    owner = occ_sample & 63
    owner_tail = sim.tail_seq.gather(
        1, owner.clamp(0, sim.N - 1).to(torch.int64).view(sim.E, -1)
    ).view(sim.E, sim.N, sz, sz)
    occ_active = in_grid & (occ_sample >= 0) & ((occ_sample >> 6) >= owner_tail)

    own_id = sim._snake_ids[:, :, None, None]
    dst[:, :, CH_OWN_BODY, :, :] = (occ_active & (owner == own_id)).to(torch.float32)
    dst[:, :, CH_ENEMY_BODY, :, :] = (occ_active & (owner != own_id)).to(torch.float32)

    # Scale-matched block-summed food lookup.
    if scale >= 16.0:
        flat_f = ((gy_c >> 4) * sim.G16 + (gx_c >> 4)).view(sim.E, sim.N * num_pix)
        amb_raw = sim.ambient_grid_16x.view(sim.E, -1).gather(1, flat_f)
        cor_raw = sim.corpse_grid_16x.view(sim.E, -1).gather(1, flat_f)
    elif scale >= 4.0:
        flat_f = ((gy_c >> 2) * sim.G4 + (gx_c >> 2)).view(sim.E, sim.N * num_pix)
        amb_raw = sim.ambient_grid_4x.view(sim.E, -1).gather(1, flat_f)
        cor_raw = sim.corpse_grid_4x.view(sim.E, -1).gather(1, flat_f)
    else:
        amb_raw = sim.ambient_grid.view(sim.E, -1).gather(1, flat_1x)
        cor_raw = sim.corpse_grid.view(sim.E, -1).gather(1, flat_1x)

    amb_f = amb_raw.view(sim.E, sim.N, sz, sz).to(torch.float32)
    cor_f = cor_raw.view(sim.E, sim.N, sz, sz).to(torch.float32)
    zeros_f = torch.zeros_like(amb_f)
    dst[:, :, CH_AMBIENT_FOOD, :, :] = torch.where(in_grid, amb_f / (1.0 + amb_f), zeros_f)
    dst[:, :, CH_CORPSE_FOOD, :, :] = torch.where(in_grid, cor_f / (1.0 + cor_f), zeros_f)

    # 2-frame stack of enemy head channels: T0 (current) and T1 (previous tick).
    _splat_heads_torch(
        dst[:, :, CH_ENEMY_HEADS_T0 : CH_ENEMY_SIZE_T0 + 1, :, :],
        obs_alive=sim.alive,
        obs_pos=sim.head_pos,
        obs_sin=sin_t0,
        obs_cos=cos_t0,
        enemy_alive=sim.alive,
        enemy_pos=sim.head_pos,
        enemy_sin=sin_t0,
        enemy_cos=cos_t0,
        enemy_size_norm=size_norm_t0,
        not_eye=sim._not_eye_nn,
        tril_nnn=cache.tril_nnn,
        scale=scale,
        raster_size=sz,
    )
    _splat_heads_torch(
        dst[:, :, CH_ENEMY_HEADS_T1 : CH_ENEMY_SIZE_T1 + 1, :, :],
        obs_alive=sim.alive & sim.prev_alive,
        obs_pos=sim.prev_head_pos,
        obs_sin=sin_t1,
        obs_cos=cos_t1,
        enemy_alive=sim.prev_alive,
        enemy_pos=sim.prev_head_pos,
        enemy_sin=sin_t1,
        enemy_cos=cos_t1,
        enemy_size_norm=size_norm_t1,
        not_eye=sim._not_eye_nn,
        tril_nnn=cache.tril_nnn,
        scale=scale,
        raster_size=sz,
    )


def build_obs(sim: GameV3Sim, *, out: ObservationBatch | None = None) -> ObservationBatch:
    """Build heading-aligned multi-scale observations on ``sim.device`` with zero host syncs."""
    cache = _get_obs_cache(sim)
    batch = out if out is not None else allocate_obs_batch(sim)
    cfg = sim.cfg

    sin_t0, cos_t0 = poly_sincos_torch(sim.theta)
    sin_t1, cos_t1 = poly_sincos_torch(sim.prev_theta)

    inv_max_w = float(np.float32(1.0 / cfg.max_body_radius))
    len_i64 = sim.length.clamp(0, cfg.max_length).to(torch.int64)
    prev_len_i64 = sim.prev_length.clamp(0, cfg.max_length).to(torch.int64)
    w_t0 = sim._w_table[len_i64]
    size_norm_t0 = w_t0 * inv_max_w
    size_norm_t1 = sim._w_table[prev_len_i64] * inv_max_w

    _fill_scale_raster_torch(
        batch.raster_s1, sim, cache, 0, sin_t0, cos_t0, size_norm_t0, sin_t1, cos_t1, size_norm_t1
    )
    _fill_scale_raster_torch(
        batch.raster_s4, sim, cache, 1, sin_t0, cos_t0, size_norm_t0, sin_t1, cos_t1, size_norm_t1
    )
    _fill_scale_raster_torch(
        batch.raster_s16, sim, cache, 2, sin_t0, cos_t0, size_norm_t0, sin_t1, cos_t1, size_norm_t1
    )

    # Scalar features: [log_length, body_radius, boost_allowed, wall_dist_norm, turn_rate_limit].
    boost_ok = sim.alive & (sim.length > int(cfg.min_boost_length))
    hx = sim.head_pos[:, :, 0]
    hy = sim.head_pos[:, :, 1]
    dist_center = torch.sqrt(hx * hx + hy * hy)
    r_f32 = float(np.float32(cfg.arena_radius))
    inv_r = float(np.float32(1.0 / cfg.arena_radius))
    wall_dist = ((r_f32 - dist_center - w_t0) * inv_r).clamp(0.0, 1.0)

    alive_f = sim.alive.to(torch.float32)
    batch.scalars[:, :, 0] = cache.len_log_table[len_i64] * alive_f
    batch.scalars[:, :, 1] = w_t0 * alive_f
    batch.scalars[:, :, 2] = boost_ok.to(torch.float32)
    batch.scalars[:, :, 3] = wall_dist * alive_f
    batch.scalars[:, :, 4] = sim._turn_table[len_i64] * alive_f

    batch.action_mask[:, :, :3] = True
    batch.action_mask[:, :, 3:] = boost_ok[:, :, None]
    return batch


# ==============================================================================
# Plain-NumPy single-world reference observation builder (independent of sim.py)
# ==============================================================================


def _sincos_ref_np(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f32 = np.float32
    ab = np.abs(theta)
    over = ab > f32(HALF_PI_F32)
    xs = np.where(over, np.where(theta >= f32(0.0), f32(PI_F32) - ab, ab - f32(PI_F32)), theta)
    xc = np.where(over, f32(PI_F32) - ab, ab)

    x2s = xs * xs
    p = x2s * f32(SIN_C9)
    p = p + f32(SIN_C7)
    p = p * x2s
    p = p + f32(SIN_C5)
    p = p * x2s
    p = p + f32(SIN_C3)
    sin_val = xs + xs * (x2s * p)

    x2c = xc * xc
    q = x2c * f32(COS_C10)
    q = q + f32(COS_C8)
    q = q * x2c
    q = q + f32(COS_C6)
    q = q * x2c
    q = q + f32(COS_C4)
    q = q * x2c
    q = q + f32(COS_C2)
    cos_val = f32(1.0) + x2c * q
    cos_val = np.where(over, -cos_val, cos_val)
    return sin_val.astype(np.float32), cos_val.astype(np.float32)


def _splat_heads_ref_np(
    dst_4ch: np.ndarray,
    *,
    obs_alive: np.ndarray,
    obs_pos: np.ndarray,
    obs_sin: np.ndarray,
    obs_cos: np.ndarray,
    enemy_alive: np.ndarray,
    enemy_pos: np.ndarray,
    enemy_sin: np.ndarray,
    enemy_cos: np.ndarray,
    enemy_size_norm: np.ndarray,
    scale: float,
    raster_size: int,
) -> None:
    dst_4ch.fill(0.0)
    n_cnt = obs_alive.shape[0]
    inv_s = np.float32(1.0 / scale)
    half_sz = np.float32(0.5 * raster_size)

    for i in range(n_cnt):
        if not obs_alive[i]:
            continue
        occupied_pixels: set[tuple[int, int]] = set()
        si = obs_sin[i]
        ci = obs_cos[i]
        for j in range(n_cnt):
            if j == i or not enemy_alive[j]:
                continue
            rx = np.float32(enemy_pos[j, 0] - obs_pos[i, 0])
            ry = np.float32(enemy_pos[j, 1] - obs_pos[i, 1])
            ego_x = np.float32(rx * si - ry * ci)
            ego_y = np.float32(rx * ci + ry * si)
            col = int(np.floor(np.float32(ego_x * inv_s + half_sz)))
            row = int(np.floor(np.float32(half_sz - ego_y * inv_s)))
            if not (0 <= col < raster_size and 0 <= row < raster_size):
                continue
            key = (row, col)
            if key in occupied_pixels:
                continue
            occupied_pixels.add(key)
            sj = enemy_sin[j]
            cj = enemy_cos[j]
            dst_4ch[i, 0, row, col] = np.float32(1.0)
            dst_4ch[i, 1, row, col] = np.float32(cj * si - sj * ci)
            dst_4ch[i, 2, row, col] = np.float32(cj * ci + sj * si)
            dst_4ch[i, 3, row, col] = enemy_size_norm[j]


def build_obs_reference(ref: ReferenceWorld) -> ReferenceObservation:
    """Build single-world NumPy reference observations directly from :class:`ReferenceWorld`."""
    cfg: GameV3Config = ref.cfg
    sz = cfg.obs_raster_size
    n_cnt = ref.N
    g_sz = ref.G

    sin_t0, cos_t0 = _sincos_ref_np(ref.theta)
    sin_t1, cos_t1 = _sincos_ref_np(ref.prev_theta)

    inv_max_w = np.float32(1.0 / cfg.max_body_radius)
    w_t0 = np.array(
        [np.float32(cfg.body_radius_f64(int(ref.length[i]))) for i in range(n_cnt)],
        dtype=np.float32,
    )
    w_t1 = np.array(
        [np.float32(cfg.body_radius_f64(int(ref.prev_length[i]))) for i in range(n_cnt)],
        dtype=np.float32,
    )
    size_t0 = (w_t0 * inv_max_w).astype(np.float32)
    size_t1 = (w_t1 * inv_max_w).astype(np.float32)

    amb_4x = ref.ambient_grid.reshape(g_sz // 4, 4, g_sz // 4, 4).sum(axis=(1, 3))
    cor_4x = ref.corpse_grid.reshape(g_sz // 4, 4, g_sz // 4, 4).sum(axis=(1, 3))
    amb_16x = ref.ambient_grid.reshape(g_sz // 16, 16, g_sz // 16, 16).sum(axis=(1, 3))
    cor_16x = ref.corpse_grid.reshape(g_sz // 16, 16, g_sz // 16, 16).sum(axis=(1, 3))

    half_r = np.float32(0.5 * sz - 0.5)
    idx_1d = np.arange(sz, dtype=np.float32)
    inv_c = np.float32(1.0 / cfg.cell_size)
    r_sq_world = np.float32(cfg.arena_radius * cfg.arena_radius)

    rasters: list[np.ndarray] = []
    for scale in cfg.obs_scales:
        s_f32 = np.float32(scale)
        ego_x_1d = ((idx_1d - half_r) * s_f32).astype(np.float32)
        ego_y_1d = ((half_r - idx_1d) * s_f32).astype(np.float32)

        ras = np.zeros((n_cnt, NUM_RASTER_CHANNELS, sz, sz), dtype=np.float32)
        for i in range(n_cnt):
            if not ref.alive[i]:
                continue
            hx = ref.head_pos[i, 0]
            hy = ref.head_pos[i, 1]
            rx = sin_t0[i]
            ry = -cos_t0[i]
            fx = cos_t0[i]
            fy = sin_t0[i]

            wx = (hx + fx * ego_y_1d)[:, None] + (rx * ego_x_1d)[None, :]
            wy = (hy + fy * ego_y_1d)[:, None] + (ry * ego_x_1d)[None, :]
            wx = wx.astype(np.float32)
            wy = wy.astype(np.float32)

            ras[i, CH_OUT_OF_ARENA] = ((wx * wx + wy * wy) >= r_sq_world).astype(np.float32)

            gx = np.floor(wx * inv_c).astype(np.int32) + ref.half_grid
            gy = np.floor(wy * inv_c).astype(np.int32) + ref.half_grid
            in_g = (gx >= 0) & (gx < g_sz) & (gy >= 0) & (gy < g_sz)

            gx_c = np.clip(gx, 0, g_sz - 1)
            gy_c = np.clip(gy, 0, g_sz - 1)
            owners = ref.occ_grid[gy_c, gx_c]

            ras[i, CH_OWN_BODY] = (in_g & (owners == i)).astype(np.float32)
            ras[i, CH_ENEMY_BODY] = (in_g & (owners >= 0) & (owners != i)).astype(np.float32)

            if scale >= 16.0:
                af = amb_16x[gy_c >> 4, gx_c >> 4].astype(np.float32)
                cf = cor_16x[gy_c >> 4, gx_c >> 4].astype(np.float32)
            elif scale >= 4.0:
                af = amb_4x[gy_c >> 2, gx_c >> 2].astype(np.float32)
                cf = cor_4x[gy_c >> 2, gx_c >> 2].astype(np.float32)
            else:
                af = ref.ambient_grid[gy_c, gx_c].astype(np.float32)
                cf = ref.corpse_grid[gy_c, gx_c].astype(np.float32)

            ras[i, CH_AMBIENT_FOOD] = np.where(in_g, af / (np.float32(1.0) + af), np.float32(0.0))
            ras[i, CH_CORPSE_FOOD] = np.where(in_g, cf / (np.float32(1.0) + cf), np.float32(0.0))

        _splat_heads_ref_np(
            ras[:, CH_ENEMY_HEADS_T0 : CH_ENEMY_SIZE_T0 + 1, :, :],
            obs_alive=ref.alive,
            obs_pos=ref.head_pos,
            obs_sin=sin_t0,
            obs_cos=cos_t0,
            enemy_alive=ref.alive,
            enemy_pos=ref.head_pos,
            enemy_sin=sin_t0,
            enemy_cos=cos_t0,
            enemy_size_norm=size_t0,
            scale=scale,
            raster_size=sz,
        )
        _splat_heads_ref_np(
            ras[:, CH_ENEMY_HEADS_T1 : CH_ENEMY_SIZE_T1 + 1, :, :],
            obs_alive=ref.alive & ref.prev_alive,
            obs_pos=ref.prev_head_pos,
            obs_sin=sin_t1,
            obs_cos=cos_t1,
            enemy_alive=ref.prev_alive,
            enemy_pos=ref.prev_head_pos,
            enemy_sin=sin_t1,
            enemy_cos=cos_t1,
            enemy_size_norm=size_t1,
            scale=scale,
            raster_size=sz,
        )
        rasters.append(ras)

    denom_log = math.log2(1.0 + cfg.max_length)
    scalars = np.zeros((n_cnt, NUM_SCALAR_FEATURES), dtype=np.float32)
    action_mask = np.zeros((n_cnt, 6), dtype=bool)
    action_mask[:, :3] = True

    r_f32 = np.float32(cfg.arena_radius)
    inv_r = np.float32(1.0 / cfg.arena_radius)
    for i in range(n_cnt):
        boost_ok = bool(ref.alive[i] and int(ref.length[i]) > cfg.min_boost_length)
        action_mask[i, 3:] = boost_ok
        if not ref.alive[i]:
            continue
        li = int(ref.length[i])
        hx = ref.head_pos[i, 0]
        hy = ref.head_pos[i, 1]
        dist_c = np.sqrt(np.float32(hx * hx + hy * hy))
        wall_d = np.clip(
            np.float32((r_f32 - dist_c - w_t0[i]) * inv_r), np.float32(0.0), np.float32(1.0)
        )
        scalars[i, 0] = np.float32(math.log2(1.0 + li) / denom_log)
        scalars[i, 1] = w_t0[i]
        scalars[i, 2] = np.float32(1.0 if boost_ok else 0.0)
        scalars[i, 3] = wall_d
        scalars[i, 4] = np.float32(cfg.turn_rate_f64(max(1, li)))

    return ReferenceObservation(
        raster_s1=rasters[0],
        raster_s4=rasters[1],
        raster_s16=rasters[2],
        scalars=scalars,
        action_mask=action_mask,
    )
