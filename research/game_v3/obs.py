"""Multi-scale pooled uint8 ego-raster and scalar observations for Game V3.1.

Provides both the zero-host-sync batched PyTorch GPU builder (:func:`build_obs`)
and an independent plain-NumPy single-world reference (:func:`build_obs_reference`),
both supporting an optional boolean ``learner_mask`` ``(E, N)`` so observations
are built ONLY for learner slots.

Specification (v3.1 Rules 10 & 11):
- Three uint8 rasters of 13 channels each:
  - ``raster_s1``: ``(..., 13, 64, 64)`` at ``1`` unit/cell (nearest lookup on ``1x``).
  - ``raster_s4``: ``(..., 13, 32, 32)`` at ``4`` units/cell (nearest lookup on
    ``4x4`` max-pooled body & count/mass-pooled food pyramid levels).
  - ``raster_s16``: ``(..., 13, 32, 32)`` at ``16`` units/cell (nearest lookup on
    ``16x16`` max-pooled body & count/mass-pooled food pyramid levels).
- Channels ``0..12`` (all ``uint8``):
  - ``0``: ``own_body`` (``0`` or ``255``)
  - ``1``: ``enemy_body`` (``0`` or max active enemy ``size_bucket`` in ``1..255``)
  - ``2``: ``ambient_food`` (count/mass density LUT ``round(255 * m / (1 + m))``)
  - ``3``: ``corpse_food`` (count/mass density LUT ``round(255 * m / (4 + m))``)
  - ``4``: ``out_of_arena`` (``0`` inside arena, ``255`` outside)
  - ``5..8``: current-tick ``t`` enemy heads (``size_bucket``, ``ego_dx``, ``ego_dy``,
    ``boosting`` flag ``0/255``)
  - ``9..12``: previous-tick ``t - 1`` enemy heads projected into the current ``t``
    observer ego frame (2-frame head stack)
- Scalars ``(..., 8)`` in ``float32`` (exact dyadic / table values, bitwise identical
  across NumPy, CPU, MPS, and CUDA):
  ``[log_length, body_radius, boost_allowed, own_boost_state, turn_rate,
  boundary_dist, boundary_ego_dx, boundary_ego_dy]``.
- ``action_mask`` ``(..., 10)`` in ``bool`` over the 10 discrete actions.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field

import numpy as np
import torch

from research.game_v3.config import (
    FOOD_LUT_MAX,
    FP_ONE,
    FP_SHIFT,
    INV_FP_ONE_F32,
    NUM_ACTIONS,
    NUM_TURN_LEVELS,
    TRIG_ONE,
    TRIG_SHIFT,
    GameV3Config,
    build_food_luts_np,
    decode_occ_owner_hi,
    decode_occ_owner_lo,
    occ_entry_active,
)
from research.game_v3.reference import ReferenceWorld
from research.game_v3.sim import GameV3Sim

CH_OWN_BODY: int = 0
CH_ENEMY_BODY: int = 1
CH_AMBIENT_FOOD: int = 2
CH_CORPSE_FOOD: int = 3
CH_OUT_OF_ARENA: int = 4
CH_ENEMY_HEADS_T0: int = 5
CH_ENEMY_DX_T0: int = 6
CH_ENEMY_DY_T0: int = 7
CH_ENEMY_BOOST_T0: int = 8
CH_ENEMY_HEADS_T1: int = 9
CH_ENEMY_DX_T1: int = 10
CH_ENEMY_DY_T1: int = 11
CH_ENEMY_BOOST_T1: int = 12

# Alias kept for clear head-channel range indexing.
CH_ENEMY_LAST_T0: int = CH_ENEMY_BOOST_T0
CH_ENEMY_LAST_T1: int = CH_ENEMY_BOOST_T1

SCALAR_LOG_LENGTH: int = 0
SCALAR_BODY_RADIUS: int = 1
SCALAR_BOOST_ALLOWED: int = 2
SCALAR_OWN_BOOST: int = 3
SCALAR_TURN_RATE: int = 4
SCALAR_BOUNDARY_DIST: int = 5
SCALAR_BOUNDARY_DX: int = 6
SCALAR_BOUNDARY_DY: int = 7

NUM_RASTER_CHANNELS: int = 13
NUM_SCALAR_FEATURES: int = 8
OBS_LEARNER_CHUNK: int = 1536


def _build_obs_luts_np(
    sin_table: np.ndarray, cos_table: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact uint8 lookup tables for food density and relative head orientation."""
    amb_lut, cor_lut = build_food_luts_np()

    # Relative angle idx = (obs_theta - enemy_theta) & (angle_steps - 1):
    # ego_dx = sin(obs_theta - enemy_theta), ego_dy = cos(obs_theta - enemy_theta).
    dx_u8 = np.empty(len(sin_table), dtype=np.uint8)
    dy_u8 = np.empty(len(cos_table), dtype=np.uint8)
    for a in range(len(sin_table)):
        dx_u8[a] = min(255, max(1, 128 + int(round(127.0 * int(sin_table[a]) / TRIG_ONE))))
        dy_u8[a] = min(255, max(1, 128 + int(round(127.0 * int(cos_table[a]) / TRIG_ONE))))
    return amb_lut, cor_lut, dx_u8, dy_u8


@dataclass
class ObservationBatch:
    """Batched multi-scale observations on a single Torch device.

    Shapes are ``(E, N, ...)`` when built without a learner selection, or ``(M, ...)``
    for the ``M`` ``True`` entries of a :class:`LearnerIndex` / ``learner_mask`` in
    row-major ``(e, n)`` order.
    """

    raster_s1: torch.Tensor
    raster_s4: torch.Tensor
    raster_s16: torch.Tensor
    scalars: torch.Tensor
    action_mask: torch.Tensor
    _learner_index: LearnerIndex | None = field(default=None, repr=False, compare=False)
    _sim_shape: tuple[int, int] | None = field(default=None, repr=False, compare=False)


@dataclass
class ReferenceObservation:
    """Single-world NumPy reference observations matching :class:`ObservationBatch`."""

    raster_s1: np.ndarray
    raster_s4: np.ndarray
    raster_s16: np.ndarray
    scalars: np.ndarray
    action_mask: np.ndarray


def _build_learner_chunks_from_env_np(
    env_np: np.ndarray,
    env_block: int,
    max_learners: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Partition monotonic learner env indices into ``(e0, e1, m0, m1)`` chunks ``< 2^30``."""
    m_total = int(env_np.shape[0])
    if m_total == 0:
        return ()
    chunks: list[tuple[int, int, int, int]] = []
    m0 = 0
    while m0 < m_total:
        e0 = int(env_np[m0])
        blk_end = ((e0 // env_block) + 1) * env_block
        limit_m = min(m_total, m0 + max_learners)
        hi_m = int(np.searchsorted(env_np, blk_end, side="left"))
        m1 = min(limit_m, max(m0 + 1, hi_m))
        e1 = int(env_np[m1 - 1]) + 1
        chunks.append((e0, e1, m0, m1))
        m0 = m1
    return tuple(chunks)


class LearnerIndex:
    """Immutable precomputed learner-slot index built once by the caller and passed in.

    Replaces implicit ``(data_ptr, _version)`` tensor caching so address reuse on CPU or
    CUDA can never return observations for stale slots.
    """

    def __init__(self, sim: GameV3Sim, learner_mask: torch.Tensor) -> None:
        if learner_mask.dtype != torch.bool or learner_mask.shape != (sim.E, sim.N):
            raise ValueError(
                f"learner_mask must be bool of shape ({sim.E}, {sim.N}), "
                f"got {learner_mask.dtype} {tuple(learner_mask.shape)}"
            )
        if learner_mask.device != sim.device:
            raise ValueError(
                f"learner_mask device {learner_mask.device} != sim device {sim.device}"
            )
        self.mask: torch.Tensor = learner_mask.detach().clone()
        self.sim_shape: tuple[int, int] = (sim.E, sim.N)
        self.device: torch.device = sim.device
        self.env_block: int = sim.env_block_size

        env_idx, slot_idx = torch.where(self.mask)
        self.env_idx: torch.Tensor = env_idx.contiguous()
        self.slot_idx: torch.Tensor = slot_idx.contiguous()
        self.count: int = int(self.env_idx.shape[0])
        self.chunks: tuple[tuple[int, int, int, int], ...] = _build_learner_chunks_from_env_np(
            self.env_idx.detach().cpu().numpy(), self.env_block, OBS_LEARNER_CHUNK
        )

    def matches(self, other: LearnerIndex | torch.Tensor | None) -> bool:
        """Return True iff ``other`` selects the exact same ``(E, N)`` mask on ``device``.

        Note:
            Passing the exact same :class:`LearnerIndex` instance (``other is self``) returns
            ``True`` in ``O(1)`` on the host with zero device synchronization. Passing a raw
            boolean ``torch.Tensor`` mask (or a distinct :class:`LearnerIndex` instance with
            matching shape/count) verifies content via ``torch.equal(..., self.mask)``, which
            triggers a device-to-host synchronization on CUDA/MPS.
        """
        if other is self:
            return True
        if other is None:
            return False
        if isinstance(other, LearnerIndex):
            if (
                other.sim_shape != self.sim_shape
                or other.device != self.device
                or other.count != self.count
                or other.env_block != self.env_block
            ):
                return False
            return bool(torch.equal(other.mask, self.mask))
        if isinstance(other, torch.Tensor):
            if (
                other.dtype != torch.bool
                or other.shape != self.sim_shape
                or other.device != self.device
            ):
                return False
            return bool(torch.equal(other, self.mask))
        return False


class _ObsStaticCache:
    """Device-resident static lookup tables cached per :class:`GameV3Sim` instance."""

    def __init__(self, sim: GameV3Sim) -> None:
        cfg = sim.cfg
        dev = sim.device
        self.sizes: tuple[int, int, int] = (
            cfg.obs_fine_size,
            cfg.obs_coarse_size,
            cfg.obs_coarse_size,
        )
        self.ego_x_2x: list[torch.Tensor] = []
        self.ego_y_2x: list[torch.Tensor] = []
        for scale, sz in zip(cfg.obs_scales, self.sizes):
            idx = torch.arange(sz, dtype=torch.int32, device=dev)
            # 2 * ego_coord in grid-cell units (exact integers).
            s_cells = max(1, int(round(scale / cfg.cell_size)))
            self.ego_x_2x.append((2 * idx - (sz - 1)) * s_cells)
            self.ego_y_2x.append(((sz - 1) - 2 * idx) * s_cells)

        sin_np = sim._sin_table.detach().cpu().numpy()
        cos_np = sim._cos_table.detach().cpu().numpy()
        amb_lut_np, cor_lut_np, dx_u8_np, dy_u8_np = _build_obs_luts_np(sin_np, cos_np)
        self.amb_lut = torch.as_tensor(amb_lut_np, device=dev)
        self.cor_lut = torch.as_tensor(cor_lut_np, device=dev)
        self.head_dx_u8 = torch.as_tensor(dx_u8_np, device=dev)
        self.head_dy_u8 = torch.as_tensor(dy_u8_np, device=dev)

        self.env_block: int = sim.env_block_size

        self.all_env_idx = (
            torch.arange(sim.E, dtype=torch.int64, device=dev)[:, None]
            .expand(sim.E, sim.N)
            .reshape(-1)
        )
        self.all_slot_idx = (
            torch.arange(sim.N, dtype=torch.int64, device=dev)[None, :]
            .expand(sim.E, sim.N)
            .reshape(-1)
        )
        all_env_np = np.repeat(np.arange(sim.E, dtype=np.int64), sim.N)
        self.all_chunks: tuple[tuple[int, int, int, int], ...] = _build_learner_chunks_from_env_np(
            all_env_np, self.env_block, OBS_LEARNER_CHUNK
        )
        self.eye_n = torch.eye(sim.N, dtype=torch.bool, device=dev)
        self.tril_nn = torch.tril(
            torch.ones((sim.N, sim.N), dtype=torch.bool, device=dev), diagonal=-1
        )[None, :, :]


_OBS_CACHE: weakref.WeakKeyDictionary[GameV3Sim, _ObsStaticCache] = weakref.WeakKeyDictionary()


def _get_obs_cache(sim: GameV3Sim) -> _ObsStaticCache:
    cached = _OBS_CACHE.get(sim)
    if cached is None:
        cached = _ObsStaticCache(sim)
        _OBS_CACHE[sim] = cached
    return cached


def _coerce_learner_spec(
    sim: GameV3Sim,
    learner_mask: LearnerIndex | torch.Tensor | None,
    learner_index: LearnerIndex | None,
) -> LearnerIndex | torch.Tensor | None:
    if learner_index is not None and learner_mask is not None:
        if not isinstance(learner_index, LearnerIndex):
            raise TypeError(f"learner_index must be a LearnerIndex, got {type(learner_index)}")
        if not learner_index.matches(learner_mask):
            raise ValueError("Conflicting learner_mask and learner_index passed simultaneously")
        return learner_index
    spec = learner_index if learner_index is not None else learner_mask
    if spec is None:
        return None
    if isinstance(spec, LearnerIndex):
        if (
            spec.sim_shape != (sim.E, sim.N)
            or spec.device != sim.device
            or spec.env_block != sim.env_block_size
        ):
            raise ValueError(
                f"LearnerIndex built for {spec.sim_shape} on {spec.device} "
                f"(env_block={spec.env_block}) does not match sim {(sim.E, sim.N)} "
                f"on {sim.device} (env_block={sim.env_block_size})"
            )
        return spec
    if isinstance(spec, torch.Tensor):
        if spec.dtype != torch.bool or spec.shape != (sim.E, sim.N):
            raise ValueError(
                f"learner_mask must be bool of shape ({sim.E}, {sim.N}), "
                f"got {spec.dtype} {tuple(spec.shape)}"
            )
        if spec.device != sim.device:
            raise ValueError(f"learner_mask device {spec.device} != sim device {sim.device}")
        return spec
    raise TypeError(f"Expected LearnerIndex, bool Tensor, or None, got {type(spec)}")


def allocate_obs_batch(
    sim: GameV3Sim,
    learner_mask: LearnerIndex | torch.Tensor | None = None,
    *,
    learner_index: LearnerIndex | None = None,
) -> ObservationBatch:
    """Preallocate an :class:`ObservationBatch` on ``sim.device`` for zero-sync reuse."""
    cache = _get_obs_cache(sim)
    s1_sz, s4_sz, s16_sz = cache.sizes
    spec = _coerce_learner_spec(sim, learner_mask, learner_index)
    if spec is None:
        lead: tuple[int, ...] = (sim.E, sim.N)
        resolved_idx: LearnerIndex | None = None
    elif isinstance(spec, LearnerIndex):
        resolved_idx = spec
        lead = (resolved_idx.count,)
    else:
        resolved_idx = LearnerIndex(sim, spec)
        lead = (resolved_idx.count,)

    return ObservationBatch(
        raster_s1=torch.empty(
            (*lead, NUM_RASTER_CHANNELS, s1_sz, s1_sz), dtype=torch.uint8, device=sim.device
        ),
        raster_s4=torch.empty(
            (*lead, NUM_RASTER_CHANNELS, s4_sz, s4_sz), dtype=torch.uint8, device=sim.device
        ),
        raster_s16=torch.empty(
            (*lead, NUM_RASTER_CHANNELS, s16_sz, s16_sz), dtype=torch.uint8, device=sim.device
        ),
        scalars=torch.empty((*lead, NUM_SCALAR_FEATURES), dtype=torch.float32, device=sim.device),
        action_mask=torch.empty((*lead, NUM_ACTIONS), dtype=torch.bool, device=sim.device),
        _learner_index=resolved_idx,
        _sim_shape=(sim.E, sim.N),
    )


def _isqrt_i64_torch(val: torch.Tensor) -> torch.Tensor:
    """Compute exact integer floor(sqrt(val)) for val in [0, 2^44) via 22-bit binary search."""
    res = torch.zeros_like(val)
    for bit in range(21, -1, -1):
        cand = res | (1 << bit)
        res = torch.where(cand * cand <= val, cand, res)
    return res


def _splat_heads_u8_torch(
    *,
    obs_alive: torch.Tensor,
    obs_pos: torch.Tensor,
    obs_theta: torch.Tensor,
    obs_sin: torch.Tensor,
    obs_cos: torch.Tensor,
    obs_slot: torch.Tensor,
    enemy_alive: torch.Tensor,
    enemy_pos: torch.Tensor,
    enemy_theta: torch.Tensor,
    enemy_boost: torch.Tensor,
    enemy_bucket: torch.Tensor,
    cache: _ObsStaticCache,
    angle_steps: int,
    scale_fp: int,
    raster_size: int,
) -> torch.Tensor:
    """Project and splat 4 uint8 enemy head channels ``(M_c, 4, S, S)`` for learner chunk."""
    m_cnt = obs_alive.shape[0]
    num_pix = raster_size * raster_size
    half_sz_fp = (raster_size // 2) * scale_fp
    angle_mask = angle_steps - 1

    rx = enemy_pos[:, :, 0].to(torch.int64) - obs_pos[:, None, 0].to(torch.int64)
    ry = enemy_pos[:, :, 1].to(torch.int64) - obs_pos[:, None, 1].to(torch.int64)

    sin_i = obs_sin[:, None].to(torch.int64)
    cos_i = obs_cos[:, None].to(torch.int64)
    ego_x_fp = ((rx * sin_i - ry * cos_i) >> TRIG_SHIFT).to(torch.int32)
    ego_y_fp = ((rx * cos_i + ry * sin_i) >> TRIG_SHIFT).to(torch.int32)

    col = torch.div(ego_x_fp + half_sz_fp, scale_fp, rounding_mode="floor")
    row = torch.div(half_sz_fp - ego_y_fp, scale_fp, rounding_mode="floor")

    not_self = ~cache.eye_n.index_select(0, obs_slot)
    valid = (
        obs_alive[:, None]
        & enemy_alive
        & not_self
        & (col >= 0)
        & (col < raster_size)
        & (row >= 0)
        & (row < raster_size)
    )
    pix = row.clamp(0, raster_size - 1).to(torch.int64) * raster_size + col.clamp(
        0, raster_size - 1
    ).to(torch.int64)

    # Lowest enemy slot index j wins deterministically on same-pixel collisions.
    same_pix = (
        valid[:, :, None] & valid[:, None, :] & (pix[:, :, None] == pix[:, None, :]) & cache.tril_nn
    )
    keep = valid & (~same_pix.any(dim=-1))

    rel_ang = ((obs_theta[:, None] - enemy_theta) & angle_mask).to(torch.int64)
    dx_u8 = cache.head_dx_u8[rel_ang]
    dy_u8 = cache.head_dy_u8[rel_ang]

    # Pack the 4 disjoint uint8 head channels into a single non-negative int32 (<= 0x01FFFFFF)
    # so 1D int32 index_put_(accumulate=True) avoids uint8 3D scatter_add_ overhead on MPS.
    packed = (
        enemy_bucket.to(torch.int32)
        | (dx_u8.to(torch.int32) << 8)
        | (dy_u8.to(torch.int32) << 16)
        | (enemy_boost.to(torch.int32) << 24)
    )
    z_i32 = torch.zeros((1,), dtype=torch.int32, device=obs_pos.device)
    src_flat = torch.where(keep, packed, z_i32).reshape(-1)
    row_base = torch.arange(m_cnt, dtype=torch.int64, device=obs_pos.device)[:, None] * num_pix
    flat_pix = (row_base + pix).reshape(-1)

    buf_i32 = torch.zeros((m_cnt * num_pix,), dtype=torch.int32, device=obs_pos.device)
    buf_i32.index_put_((flat_pix,), src_flat, accumulate=True)
    buf_2d = buf_i32.view(m_cnt, raster_size, raster_size)

    return torch.stack(
        (
            (buf_2d & 255).to(torch.uint8),
            ((buf_2d >> 8) & 255).to(torch.uint8),
            ((buf_2d >> 16) & 255).to(torch.uint8),
            ((buf_2d >> 24) & 1).to(torch.uint8) * 255,
        ),
        dim=1,
    )


def _fill_scale_raster_learners_torch(
    dst_flat: torch.Tensor,
    sim: GameV3Sim,
    cache: _ObsStaticCache,
    scale_idx: int,
    env_idx: torch.Tensor,
    slot_idx: torch.Tensor,
    chunks: tuple[tuple[int, int, int, int], ...],
    size_bucket_t0: torch.Tensor,
    size_bucket_t1: torch.Tensor,
) -> None:
    """Populate all 13 uint8 channels of ``dst_flat`` ``(M, 13, S, S)`` for learner rows."""
    if not chunks:
        return

    cfg = sim.cfg
    sz = cache.sizes[scale_idx]
    num_pix = sz * sz
    scale_cells = max(1, int(round(cfg.obs_scales[scale_idx] / cfg.cell_size)))
    scale_fp = scale_cells * FP_ONE
    arena_r_sq_fp = int(cfg.arena_radius_fp) * int(cfg.arena_radius_fp)

    ego_x_2x = cache.ego_x_2x[scale_idx]
    ego_y_2x = cache.ego_y_2x[scale_idx]

    z_u8 = torch.zeros((1,), dtype=torch.uint8, device=sim.device)
    max_u8 = torch.full((1,), 255, dtype=torch.uint8, device=sim.device)

    for e0, e1, m0, m1 in chunks:
        mc = m1 - m0
        le = env_idx[m0:m1]
        le_local = le - e0
        ln = slot_idx[m0:m1]
        ln_i32 = ln.to(torch.int32)

        obs_alive = sim.alive[le, ln]
        obs_pos = sim.head_pos[le, ln]
        obs_theta = sim.theta[le, ln]
        obs_th_i64 = obs_theta.to(torch.int64)
        sin0 = sim._sin_table[obs_th_i64]
        cos0 = sim._cos_table[obs_th_i64]

        # Exact integer ego-to-world sample coordinates (in 1/1024 cell units).
        # wx_fp = hx_fp + ((cos0 * ego_y_2x + sin0 * ego_x_2x + 16) >> 5)
        # wy_fp = hy_fp + ((sin0 * ego_y_2x - cos0 * ego_x_2x + 16) >> 5)
        rot_x = (
            cos0[:, None, None] * ego_y_2x[None, :, None]
            + sin0[:, None, None] * ego_x_2x[None, None, :]
            + 16
        ) >> 5
        rot_y = (
            sin0[:, None, None] * ego_y_2x[None, :, None]
            - cos0[:, None, None] * ego_x_2x[None, None, :]
            + 16
        ) >> 5
        wx_fp = obs_pos[:, 0, None, None] + rot_x
        wy_fp = obs_pos[:, 1, None, None] + rot_y
        del rot_x, rot_y

        alive_pix = obs_alive[:, None, None]
        d_sq = wx_fp.to(torch.int64).square() + wy_fp.to(torch.int64).square()
        ch_out = torch.where(alive_pix & (d_sq >= arena_r_sq_fp), max_u8, z_u8)
        del d_sq

        gx = (wx_fp >> FP_SHIFT) + sim.half_grid
        gy = (wy_fp >> FP_SHIFT) + sim.half_grid
        del wx_fp, wy_fp
        in_grid = alive_pix & (gx >= 0) & (gx < sim.G) & (gy >= 0) & (gy < sim.G)
        in_flat = in_grid.view(mc, num_pix)

        oldest_e = sim.oldest_active_subtick.index_select(0, le)
        bkt_e = size_bucket_t0.index_select(0, le)

        if scale_idx == 0:
            gx_c = gx.clamp(0, sim.G - 1).to(torch.int64)
            gy_c = gy.clamp(0, sim.G - 1).to(torch.int64)
            del gx, gy
            g_idx = (le_local[:, None, None] * (sim.G * sim.G) + gy_c * sim.G + gx_c).view(-1)
            del gx_c, gy_c

            occ_hi_sub = sim.occ_hi[e0:e1].view(-1)
            v_hi = occ_hi_sub.index_select(0, g_idx).view(mc, num_pix)
            ow_hi = decode_occ_owner_hi(v_hi)
            ow_hi_i64 = ow_hi.clamp(0, sim.N - 1).to(torch.int64)
            act_hi = in_flat & occ_entry_active(v_hi, oldest_e.gather(1, ow_hi_i64))

            occ_lo_sub = sim.occ_lo[e0:e1].view(-1)
            v_lo = occ_lo_sub.index_select(0, g_idx).view(mc, num_pix)
            ow_lo = decode_occ_owner_lo(v_lo)
            ow_lo_i64 = ow_lo.clamp(0, sim.N - 1).to(torch.int64)
            act_lo = in_flat & occ_entry_active(v_lo, oldest_e.gather(1, ow_lo_i64))

            self_col = ln_i32[:, None]
            is_own = (act_hi & (ow_hi == self_col)) | (act_lo & (ow_lo == self_col))
            ch_own = torch.where(is_own, max_u8, z_u8).view(mc, sz, sz)

            eb_hi = torch.where(act_hi & (ow_hi != self_col), bkt_e.gather(1, ow_hi_i64), z_u8)
            eb_lo = torch.where(act_lo & (ow_lo != self_col), bkt_e.gather(1, ow_lo_i64), z_u8)
            ch_enemy = torch.maximum(eb_hi, eb_lo).view(mc, sz, sz)
            del v_hi, v_lo, ow_hi, ow_lo, act_hi, act_lo, eb_hi, eb_lo

            amb_1x_sub = sim.ambient_grid[e0:e1].view(-1)
            cor_1x_sub = sim.corpse_grid[e0:e1].view(-1)
            amb_m = (
                amb_1x_sub.index_select(0, g_idx)
                .to(torch.int64)
                .view(mc, sz, sz)
                .clamp(0, FOOD_LUT_MAX)
            )
            cor_m = (
                cor_1x_sub.index_select(0, g_idx)
                .to(torch.int64)
                .view(mc, sz, sz)
                .clamp(0, FOOD_LUT_MAX)
            )
            del g_idx
        elif scale_idx == 1:
            gx4 = (gx >> 2).clamp(0, sim.G4 - 1).to(torch.int64)
            gy4 = (gy >> 2).clamp(0, sim.G4 - 1).to(torch.int64)
            del gx, gy
            g_idx4 = (le_local[:, None, None] * (sim.G4 * sim.G4) + gy4 * sim.G4 + gx4).view(-1)
            del gx4, gy4

            occ_4x_sub = sim.occ_4x_slice(e0, e1).view(-1, sim.N)
            st_n = occ_4x_sub.index_select(0, g_idx4).view(mc, num_pix, sim.N)
            act_n = in_flat[:, :, None] & (st_n >= oldest_e[:, None, :])
            del st_n

            own_act = act_n.gather(2, ln[:, None, None].expand(mc, num_pix, 1)).squeeze(2)
            ch_own = torch.where(own_act, max_u8, z_u8).view(mc, sz, sz)

            not_self = ~cache.eye_n.index_select(0, ln)[:, None, :]
            ch_enemy = (
                torch.where(act_n & not_self, bkt_e[:, None, :], z_u8).amax(dim=-1).view(mc, sz, sz)
            )
            del act_n, own_act

            amb_4x_sub = sim.ambient_grid_4x[e0:e1].view(-1)
            cor_4x_sub = sim.corpse_grid_4x[e0:e1].view(-1)
            amb_m = amb_4x_sub.index_select(0, g_idx4).view(mc, sz, sz).clamp(0, FOOD_LUT_MAX)
            cor_m = cor_4x_sub.index_select(0, g_idx4).view(mc, sz, sz).clamp(0, FOOD_LUT_MAX)
            del g_idx4
        else:
            gx16 = (gx >> 4).clamp(0, sim.G16 - 1).to(torch.int64)
            gy16 = (gy >> 4).clamp(0, sim.G16 - 1).to(torch.int64)
            del gx, gy
            g_idx16 = (le_local[:, None, None] * (sim.G16 * sim.G16) + gy16 * sim.G16 + gx16).view(
                -1
            )
            del gx16, gy16

            occ_16x_sub = sim.occ_subtick_16x[e0:e1].view(-1, sim.N)
            st_n = occ_16x_sub.index_select(0, g_idx16).view(mc, num_pix, sim.N)
            act_n = in_flat[:, :, None] & (st_n >= oldest_e[:, None, :])
            del st_n

            own_act = act_n.gather(2, ln[:, None, None].expand(mc, num_pix, 1)).squeeze(2)
            ch_own = torch.where(own_act, max_u8, z_u8).view(mc, sz, sz)

            not_self = ~cache.eye_n.index_select(0, ln)[:, None, :]
            ch_enemy = (
                torch.where(act_n & not_self, bkt_e[:, None, :], z_u8).amax(dim=-1).view(mc, sz, sz)
            )
            del act_n, own_act

            amb_16x_sub = sim.ambient_grid_16x[e0:e1].view(-1)
            cor_16x_sub = sim.corpse_grid_16x[e0:e1].view(-1)
            amb_m = amb_16x_sub.index_select(0, g_idx16).view(mc, sz, sz).clamp(0, FOOD_LUT_MAX)
            cor_m = cor_16x_sub.index_select(0, g_idx16).view(mc, sz, sz).clamp(0, FOOD_LUT_MAX)
            del g_idx16

        ch_amb = torch.where(in_grid, cache.amb_lut[amb_m.to(torch.int64)], z_u8)
        ch_cor = torch.where(in_grid, cache.cor_lut[cor_m.to(torch.int64)], z_u8)
        del amb_m, cor_m, in_grid, in_flat

        heads_t0 = _splat_heads_u8_torch(
            obs_alive=obs_alive,
            obs_pos=obs_pos,
            obs_theta=obs_theta,
            obs_sin=sin0,
            obs_cos=cos0,
            obs_slot=ln,
            enemy_alive=sim.alive.index_select(0, le),
            enemy_pos=sim.head_pos.index_select(0, le),
            enemy_theta=sim.theta.index_select(0, le),
            enemy_boost=sim.boosting.index_select(0, le),
            enemy_bucket=bkt_e,
            cache=cache,
            angle_steps=cfg.angle_steps,
            scale_fp=scale_fp,
            raster_size=sz,
        )
        # Project previous-tick (t - 1) enemy heads into the CURRENT observer ego frame.
        heads_t1 = _splat_heads_u8_torch(
            obs_alive=obs_alive,
            obs_pos=obs_pos,
            obs_theta=obs_theta,
            obs_sin=sin0,
            obs_cos=cos0,
            obs_slot=ln,
            enemy_alive=sim.prev_alive.index_select(0, le),
            enemy_pos=sim.prev_head_pos.index_select(0, le),
            enemy_theta=sim.prev_theta.index_select(0, le),
            enemy_boost=sim.prev_boosting.index_select(0, le),
            enemy_bucket=size_bucket_t1.index_select(0, le),
            cache=cache,
            angle_steps=cfg.angle_steps,
            scale_fp=scale_fp,
            raster_size=sz,
        )

        torch.cat(
            (
                ch_own[:, None],
                ch_enemy[:, None],
                ch_amb[:, None],
                ch_cor[:, None],
                ch_out[:, None],
                heads_t0,
                heads_t1,
            ),
            dim=1,
            out=dst_flat[m0:m1],
        )


def _signed_div_q10_torch(num: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """Compute signed trunc(num / denom) clamped to [-1024, +1024] without Tensor.abs().

    Note: PyTorch MPS lowers integer ``Tensor.abs()`` through ``float32`` (corrupting
    values ``|x| > 2**24``), whereas ``torch.where(num >= 0, num, -num)`` + integer
    ``rounding_mode="floor"`` is 64-bit exact across CPU, MPS, and CUDA.
    """
    abs_num = torch.where(num >= 0, num, -num)
    abs_q = torch.div(abs_num, denom, rounding_mode="floor").clamp_max(1024)
    return torch.where(num >= 0, abs_q, -abs_q)


def build_obs(
    sim: GameV3Sim,
    learner_mask: LearnerIndex | torch.Tensor | None = None,
    *,
    learner_index: LearnerIndex | None = None,
    out: ObservationBatch | None = None,
) -> ObservationBatch:
    """Build multi-scale uint8 pyramid observations on ``sim.device``.

    For zero-host-sync steady-state rollouts, construct a :class:`LearnerIndex` once,
    allocate ``out = allocate_obs_batch(sim, idx)``, and reuse that same ``idx`` instance in
    ``build_obs(sim, idx, out=out)`` (fast ``other is self`` identity check). Passing a raw
    boolean ``torch.Tensor`` ``learner_mask`` (or a newly constructed :class:`LearnerIndex`)
    with ``out=`` validates mask equality via ``torch.equal(..., out._learner_index.mask)``,
    which incurs a device-to-host synchronization on CUDA/MPS. Passing an ``out`` batch built
    with a different mask/index raises ``ValueError``.
    """
    cache = _get_obs_cache(sim)
    spec = _coerce_learner_spec(sim, learner_mask, learner_index)

    if out is None:
        batch = allocate_obs_batch(sim, spec)
    else:
        if out.raster_s1.device != sim.device:
            raise ValueError(
                f"out ObservationBatch device {out.raster_s1.device} != sim device {sim.device}"
            )
        if out._sim_shape is not None and out._sim_shape != (sim.E, sim.N):
            raise ValueError(
                f"out ObservationBatch was allocated for sim shape {out._sim_shape}, "
                f"got {(sim.E, sim.N)}"
            )
        if out._learner_index is None:
            if spec is not None:
                raise ValueError(
                    "out ObservationBatch was allocated for all slots (learner_index=None), "
                    "but build_obs received a non-None learner_mask/learner_index"
                )
            if out.raster_s1.shape[:2] != (sim.E, sim.N):
                raise ValueError(
                    f"out ObservationBatch leading shape {tuple(out.raster_s1.shape[:2])} "
                    f"does not match ({sim.E}, {sim.N})"
                )
        else:
            if spec is None:
                raise ValueError(
                    "out ObservationBatch was allocated with a LearnerIndex/learner_mask, "
                    "but build_obs received None"
                )
            if not out._learner_index.matches(spec):
                raise ValueError(
                    "learner_mask/learner_index passed to build_obs differs from the one "
                    "used to build out ObservationBatch"
                )
        batch = out

    cfg = sim.cfg
    s1_sz, s4_sz, s16_sz = cache.sizes

    if batch._learner_index is None:
        env_idx = cache.all_env_idx
        slot_idx = cache.all_slot_idx
        chunks = cache.all_chunks
        m_cnt = sim.E * sim.N
    else:
        env_idx = batch._learner_index.env_idx
        slot_idx = batch._learner_index.slot_idx
        chunks = batch._learner_index.chunks
        m_cnt = batch._learner_index.count

    len_all_i64 = sim.length.clamp(0, cfg.max_length).to(torch.int64)
    prev_len_all_i64 = sim.prev_length.clamp(0, cfg.max_length).to(torch.int64)
    z_u8 = torch.zeros((1,), dtype=torch.uint8, device=sim.device)
    size_bucket_t0 = torch.where(sim.alive, sim._size_bucket_table[len_all_i64], z_u8)
    size_bucket_t1 = torch.where(sim.prev_alive, sim._size_bucket_table[prev_len_all_i64], z_u8)

    r1_flat = batch.raster_s1.view(m_cnt, NUM_RASTER_CHANNELS, s1_sz, s1_sz)
    r4_flat = batch.raster_s4.view(m_cnt, NUM_RASTER_CHANNELS, s4_sz, s4_sz)
    r16_flat = batch.raster_s16.view(m_cnt, NUM_RASTER_CHANNELS, s16_sz, s16_sz)

    _fill_scale_raster_learners_torch(
        r1_flat, sim, cache, 0, env_idx, slot_idx, chunks, size_bucket_t0, size_bucket_t1
    )
    _fill_scale_raster_learners_torch(
        r4_flat, sim, cache, 1, env_idx, slot_idx, chunks, size_bucket_t0, size_bucket_t1
    )
    _fill_scale_raster_learners_torch(
        r16_flat, sim, cache, 2, env_idx, slot_idx, chunks, size_bucket_t0, size_bucket_t1
    )

    sc_flat = batch.scalars.view(m_cnt, NUM_SCALAR_FEATURES)
    am_flat = batch.action_mask.view(m_cnt, NUM_ACTIONS)
    if m_cnt == 0:
        return batch

    l_alive = sim.alive[env_idx, slot_idx]
    l_boost = sim.boosting[env_idx, slot_idx]
    l_len = sim.length[env_idx, slot_idx]
    l_len_i64 = l_len.clamp(0, cfg.max_length).to(torch.int64)
    l_pos = sim.head_pos[env_idx, slot_idx]
    l_th_i64 = sim.theta[env_idx, slot_idx].to(torch.int64)

    boost_ok = l_alive & (l_len > int(cfg.min_boost_length))
    alive_f = l_alive.to(torch.float32)

    hx64 = l_pos[:, 0].to(torch.int64)
    hy64 = l_pos[:, 1].to(torch.int64)
    dist_c_fp = _isqrt_i64_torch(hx64 * hx64 + hy64 * hy64)

    arena_r_fp = int(cfg.arena_radius_fp)
    rad_fp = sim._radius_fp_table[l_len_i64].to(torch.int64)
    clear_fp = (arena_r_fp - dist_c_fp - rad_fp).clamp(0, arena_r_fp)
    wall_dist_q10 = torch.div(clear_fp * 1024, arena_r_fp, rounding_mode="floor")

    sin_i = sim._sin_table[l_th_i64].to(torch.int64)
    cos_i = sim._cos_table[l_th_i64].to(torch.int64)
    denom = dist_c_fp.clamp_min(1) * TRIG_ONE
    dx_q10_raw = _signed_div_q10_torch((hx64 * sin_i - hy64 * cos_i) * 1024, denom)
    dy_q10_raw = _signed_div_q10_torch((hx64 * cos_i + hy64 * sin_i) * 1024, denom)
    at_origin = dist_c_fp == 0
    z_i64 = torch.zeros_like(dx_q10_raw)
    dx_q10 = torch.where(l_alive & (~at_origin), dx_q10_raw, z_i64)
    dy_q10 = torch.where(
        l_alive,
        torch.where(at_origin, torch.full_like(dy_q10_raw, 1024), dy_q10_raw),
        z_i64,
    )

    sc_flat[:, SCALAR_LOG_LENGTH] = sim._len_log_f32_table[l_len_i64] * alive_f
    sc_flat[:, SCALAR_BODY_RADIUS] = sim._radius_norm_f32_table[l_len_i64] * alive_f
    sc_flat[:, SCALAR_BOOST_ALLOWED] = boost_ok.to(torch.float32)
    sc_flat[:, SCALAR_OWN_BOOST] = (l_alive & l_boost).to(torch.float32)
    sc_flat[:, SCALAR_TURN_RATE] = sim._turn_norm_f32_table[l_len_i64] * alive_f
    sc_flat[:, SCALAR_BOUNDARY_DIST] = (
        torch.where(l_alive, wall_dist_q10, z_i64).to(torch.float32) * INV_FP_ONE_F32
    )
    sc_flat[:, SCALAR_BOUNDARY_DX] = dx_q10.to(torch.float32) * INV_FP_ONE_F32
    sc_flat[:, SCALAR_BOUNDARY_DY] = dy_q10.to(torch.float32) * INV_FP_ONE_F32

    am_flat[:, :NUM_TURN_LEVELS] = True
    am_flat[:, NUM_TURN_LEVELS:] = boost_ok[:, None]
    return batch


# ==============================================================================
# Plain-NumPy single-world reference observation builder (independent of sim.py)
# ==============================================================================


def _splat_heads_ref_u8_np(
    dst_4ch: np.ndarray,
    *,
    learner_slots: list[int],
    obs_alive: np.ndarray,
    obs_pos: np.ndarray,
    obs_theta: np.ndarray,
    sin_table: np.ndarray,
    cos_table: np.ndarray,
    enemy_alive: np.ndarray,
    enemy_pos: np.ndarray,
    enemy_theta: np.ndarray,
    enemy_boost: np.ndarray,
    enemy_bucket: np.ndarray,
    dx_u8_lut: np.ndarray,
    dy_u8_lut: np.ndarray,
    angle_steps: int,
    scale_fp: int,
    raster_size: int,
) -> None:
    dst_4ch.fill(0)
    n_cnt = enemy_alive.shape[0]
    half_sz_fp = (raster_size // 2) * scale_fp
    angle_mask = angle_steps - 1

    for out_row, i in enumerate(learner_slots):
        if not obs_alive[i]:
            continue
        occupied_pixels: set[tuple[int, int]] = set()
        th_i = int(obs_theta[i])
        si = int(sin_table[th_i])
        ci = int(cos_table[th_i])
        ox, oy = int(obs_pos[i, 0]), int(obs_pos[i, 1])

        for j in range(n_cnt):
            if j == i or not enemy_alive[j]:
                continue
            rx = int(enemy_pos[j, 0]) - ox
            ry = int(enemy_pos[j, 1]) - oy
            ego_x_fp = (rx * si - ry * ci) >> TRIG_SHIFT
            ego_y_fp = (rx * ci + ry * si) >> TRIG_SHIFT
            col = (ego_x_fp + half_sz_fp) // scale_fp
            row = (half_sz_fp - ego_y_fp) // scale_fp
            if not (0 <= col < raster_size and 0 <= row < raster_size):
                continue
            key = (row, col)
            if key in occupied_pixels:
                continue
            occupied_pixels.add(key)
            rel_ang = (th_i - int(enemy_theta[j])) & angle_mask
            dst_4ch[out_row, 0, row, col] = enemy_bucket[j]
            dst_4ch[out_row, 1, row, col] = dx_u8_lut[rel_ang]
            dst_4ch[out_row, 2, row, col] = dy_u8_lut[rel_ang]
            dst_4ch[out_row, 3, row, col] = np.uint8(255 if enemy_boost[j] else 0)


def _max_enemy_bucket_from_mask_grid_np(
    mask_grid: np.ndarray, self_slot: int, size_buckets: np.ndarray
) -> np.ndarray:
    """Compute max active enemy size_bucket per cell directly from uint64 owner bitmasks."""
    out = np.zeros(mask_grid.shape, dtype=np.uint8)
    for j in range(len(size_buckets)):
        if j == self_slot or size_buckets[j] == 0:
            continue
        bit_j = np.uint64(1) << np.uint64(j)
        active_j = (mask_grid & bit_j) != 0
        out = np.where(active_j, np.maximum(out, size_buckets[j]), out)
    return out


def build_obs_reference(
    ref: ReferenceWorld,
    learner_mask: np.ndarray | None = None,
) -> ReferenceObservation:
    """Build single-world NumPy reference observations directly from :class:`ReferenceWorld`."""
    cfg: GameV3Config = ref.cfg
    n_cnt = ref.N
    g_sz = ref.G
    max_len = cfg.max_length

    if learner_mask is None:
        learner_slots = list(range(n_cnt))
    else:
        mask_arr = np.asarray(learner_mask, dtype=bool)
        if mask_arr.shape != (n_cnt,):
            raise ValueError(f"learner_mask must have shape ({n_cnt},), got {mask_arr.shape}")
        learner_slots = [int(i) for i in np.flatnonzero(mask_arr)]

    m_cnt = len(learner_slots)
    sizes = (cfg.obs_fine_size, cfg.obs_coarse_size, cfg.obs_coarse_size)
    amb_lut, cor_lut, dx_u8_lut, dy_u8_lut = _build_obs_luts_np(ref.sin_table, ref.cos_table)

    bkt_t0 = np.array(
        [
            (
                ref.tables["size_bucket"][min(max(int(ref.length[j]), 0), max_len)]
                if ref.alive[j]
                else np.uint8(0)
            )
            for j in range(n_cnt)
        ],
        dtype=np.uint8,
    )
    bkt_t1 = np.array(
        [
            (
                ref.tables["size_bucket"][min(max(int(ref.prev_length[j]), 0), max_len)]
                if ref.prev_alive[j]
                else np.uint8(0)
            )
            for j in range(n_cnt)
        ],
        dtype=np.uint8,
    )

    # Exact 4x4 and 16x16 max-pooled body bitmask pyramids and sum-pooled food pyramids!
    occ_4x = np.bitwise_or.reduce(ref.occ_mask.reshape(g_sz // 4, 4, g_sz // 4, 4), axis=(1, 3))
    occ_16x = np.bitwise_or.reduce(
        ref.occ_mask.reshape(g_sz // 16, 16, g_sz // 16, 16), axis=(1, 3)
    )
    amb_i32 = ref.ambient_grid.astype(np.int32)
    cor_i32 = ref.corpse_grid.astype(np.int32)
    amb_4x = amb_i32.reshape(g_sz // 4, 4, g_sz // 4, 4).sum(axis=(1, 3))
    cor_4x = cor_i32.reshape(g_sz // 4, 4, g_sz // 4, 4).sum(axis=(1, 3))
    amb_16x = amb_i32.reshape(g_sz // 16, 16, g_sz // 16, 16).sum(axis=(1, 3))
    cor_16x = cor_i32.reshape(g_sz // 16, 16, g_sz // 16, 16).sum(axis=(1, 3))

    arena_r_sq_fp = int(cfg.arena_radius_fp) * int(cfg.arena_radius_fp)
    obs_alive_t0 = ref.alive

    rasters: list[np.ndarray] = []
    for scale_idx, (scale, sz) in enumerate(zip(cfg.obs_scales, sizes)):
        s_cells = max(1, int(round(scale / cfg.cell_size)))
        scale_fp = s_cells * FP_ONE
        idx_1d = np.arange(sz, dtype=np.int32)
        ego_x_2x = (2 * idx_1d - (sz - 1)) * s_cells
        ego_y_2x = ((sz - 1) - 2 * idx_1d) * s_cells

        ras = np.zeros((m_cnt, NUM_RASTER_CHANNELS, sz, sz), dtype=np.uint8)
        for out_row, i in enumerate(learner_slots):
            if not ref.alive[i]:
                continue
            hx, hy = int(ref.head_pos[i, 0]), int(ref.head_pos[i, 1])
            th_i = int(ref.theta[i])
            si = int(ref.sin_table[th_i])
            ci = int(ref.cos_table[th_i])

            rot_x = ((ci * ego_y_2x[:, None] + si * ego_x_2x[None, :] + 16) >> 5).astype(np.int32)
            rot_y = ((si * ego_y_2x[:, None] - ci * ego_x_2x[None, :] + 16) >> 5).astype(np.int32)
            wx_fp = hx + rot_x
            wy_fp = hy + rot_y

            d_sq = wx_fp.astype(np.int64) ** 2 + wy_fp.astype(np.int64) ** 2
            ras[out_row, CH_OUT_OF_ARENA] = np.where(
                d_sq >= arena_r_sq_fp, np.uint8(255), np.uint8(0)
            )

            gx = (wx_fp >> FP_SHIFT) + ref.half_grid
            gy = (wy_fp >> FP_SHIFT) + ref.half_grid
            in_g = (gx >= 0) & (gx < g_sz) & (gy >= 0) & (gy < g_sz)

            self_bit = np.uint64(1) << np.uint64(i)
            if scale_idx == 0:
                gx_c = np.clip(gx, 0, g_sz - 1)
                gy_c = np.clip(gy, 0, g_sz - 1)
                m_grid = ref.occ_mask[gy_c, gx_c]
                af = np.clip(amb_i32[gy_c, gx_c], 0, FOOD_LUT_MAX)
                cf = np.clip(cor_i32[gy_c, gx_c], 0, FOOD_LUT_MAX)
            elif scale_idx == 1:
                gx4 = np.clip(gx >> 2, 0, (g_sz // 4) - 1)
                gy4 = np.clip(gy >> 2, 0, (g_sz // 4) - 1)
                m_grid = occ_4x[gy4, gx4]
                af = np.clip(amb_4x[gy4, gx4], 0, FOOD_LUT_MAX)
                cf = np.clip(cor_4x[gy4, gx4], 0, FOOD_LUT_MAX)
            else:
                gx16 = np.clip(gx >> 4, 0, (g_sz // 16) - 1)
                gy16 = np.clip(gy >> 4, 0, (g_sz // 16) - 1)
                m_grid = occ_16x[gy16, gx16]
                af = np.clip(amb_16x[gy16, gx16], 0, FOOD_LUT_MAX)
                cf = np.clip(cor_16x[gy16, gx16], 0, FOOD_LUT_MAX)

            ras[out_row, CH_OWN_BODY] = np.where(
                in_g & ((m_grid & self_bit) != 0), np.uint8(255), np.uint8(0)
            )
            eb_raw = _max_enemy_bucket_from_mask_grid_np(m_grid, i, bkt_t0)
            ras[out_row, CH_ENEMY_BODY] = np.where(in_g, eb_raw, np.uint8(0))
            ras[out_row, CH_AMBIENT_FOOD] = np.where(in_g, amb_lut[af], np.uint8(0))
            ras[out_row, CH_CORPSE_FOOD] = np.where(in_g, cor_lut[cf], np.uint8(0))

        _splat_heads_ref_u8_np(
            ras[:, CH_ENEMY_HEADS_T0 : CH_ENEMY_BOOST_T0 + 1, :, :],
            learner_slots=learner_slots,
            obs_alive=obs_alive_t0,
            obs_pos=ref.head_pos,
            obs_theta=ref.theta,
            sin_table=ref.sin_table,
            cos_table=ref.cos_table,
            enemy_alive=ref.alive,
            enemy_pos=ref.head_pos,
            enemy_theta=ref.theta,
            enemy_boost=ref.boosting,
            enemy_bucket=bkt_t0,
            dx_u8_lut=dx_u8_lut,
            dy_u8_lut=dy_u8_lut,
            angle_steps=cfg.angle_steps,
            scale_fp=scale_fp,
            raster_size=sz,
        )
        # Project previous-tick (t - 1) enemy heads into the CURRENT observer ego frame.
        _splat_heads_ref_u8_np(
            ras[:, CH_ENEMY_HEADS_T1 : CH_ENEMY_BOOST_T1 + 1, :, :],
            learner_slots=learner_slots,
            obs_alive=obs_alive_t0,
            obs_pos=ref.head_pos,
            obs_theta=ref.theta,
            sin_table=ref.sin_table,
            cos_table=ref.cos_table,
            enemy_alive=ref.prev_alive,
            enemy_pos=ref.prev_head_pos,
            enemy_theta=ref.prev_theta,
            enemy_boost=ref.prev_boosting,
            enemy_bucket=bkt_t1,
            dx_u8_lut=dx_u8_lut,
            dy_u8_lut=dy_u8_lut,
            angle_steps=cfg.angle_steps,
            scale_fp=scale_fp,
            raster_size=sz,
        )
        rasters.append(ras)

    scalars = np.zeros((m_cnt, NUM_SCALAR_FEATURES), dtype=np.float32)
    action_mask = np.zeros((m_cnt, NUM_ACTIONS), dtype=bool)
    action_mask[:, :NUM_TURN_LEVELS] = True

    arena_r_fp = int(cfg.arena_radius_fp)
    inv_1024 = np.float32(INV_FP_ONE_F32)

    for out_row, i in enumerate(learner_slots):
        li = min(max(int(ref.length[i]), 0), max_len)
        boost_ok = bool(ref.alive[i] and li > cfg.min_boost_length)
        action_mask[out_row, NUM_TURN_LEVELS:] = boost_ok
        if not ref.alive[i]:
            continue

        hx, hy = int(ref.head_pos[i, 0]), int(ref.head_pos[i, 1])
        dist_c_fp = math.isqrt(hx * hx + hy * hy)
        rad_fp = int(ref.tables["radius_fp"][li])
        clear_fp = min(max(arena_r_fp - dist_c_fp - rad_fp, 0), arena_r_fp)
        wall_dist_q10 = (clear_fp * 1024) // arena_r_fp

        if dist_c_fp == 0:
            dx_q10, dy_q10 = 0, 1024
        else:
            th_i = int(ref.theta[i])
            si = int(ref.sin_table[th_i])
            ci = int(ref.cos_table[th_i])
            denom = dist_c_fp * TRIG_ONE
            num_dx = (hx * si - hy * ci) * 1024
            num_dy = (hx * ci + hy * si) * 1024
            dx_q10 = min(1024, abs(num_dx) // denom) * (1 if num_dx >= 0 else -1)
            dy_q10 = min(1024, abs(num_dy) // denom) * (1 if num_dy >= 0 else -1)

        scalars[out_row, SCALAR_LOG_LENGTH] = ref.tables["len_log_f32"][li]
        scalars[out_row, SCALAR_BODY_RADIUS] = ref.tables["radius_norm_f32"][li]
        scalars[out_row, SCALAR_BOOST_ALLOWED] = np.float32(1.0 if boost_ok else 0.0)
        scalars[out_row, SCALAR_OWN_BOOST] = np.float32(1.0 if ref.boosting[i] else 0.0)
        scalars[out_row, SCALAR_TURN_RATE] = ref.tables["turn_norm_f32"][li]
        scalars[out_row, SCALAR_BOUNDARY_DIST] = np.float32(wall_dist_q10) * inv_1024
        scalars[out_row, SCALAR_BOUNDARY_DX] = np.float32(dx_q10) * inv_1024
        scalars[out_row, SCALAR_BOUNDARY_DY] = np.float32(dy_q10) * inv_1024

    return ReferenceObservation(
        raster_s1=rasters[0],
        raster_s4=rasters[1],
        raster_s16=rasters[2],
        scalars=scalars,
        action_mask=action_mask,
    )
