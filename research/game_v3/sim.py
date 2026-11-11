"""GPU-first batched fixed-point PyTorch simulator for Game V3.1 (E worlds x N snakes)."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from research.game_v3.config import (
    BOOST_ACCUM_ONE,
    BOOST_ACCUM_SHIFT,
    FALLBACK_RING_SIZE,
    FP_HALF,
    FP_SHIFT,
    NUM_ACTIONS,
    NUM_TURN_LEVELS,
    TRIG_HALF,
    TRIG_SHIFT,
    GameV3Config,
    build_fallback_spawns_np,
    build_length_tables_np,
    build_trig_tables_np,
)
from research.game_v3.rng import (
    PURPOSE_AMBIENT_FOOD,
    PURPOSE_INIT_FOOD,
    PURPOSE_INIT_SNAKE,
    PURPOSE_RESPAWN,
    rng_angle_idx_torch,
    rng_int_torch,
)

_INIT_SNAKE_CANDIDATES: int = 32
_INIT_FOOD_BATCH_SIZE: int = 128


@dataclass
class StepInfo:
    """Per-slot per-tick episode and physical signals emitted by :class:`GameV3Sim`.

    No reward is computed inside the simulator. Every slot-tick partitions cleanly
    into ``alive``, ``died``, or ``in_respawn_delay`` (mutually exclusive), with
    ``life_start`` marking newly spawned lives and ``truncated`` marking live
    snakes that reached ``cfg.max_life_ticks`` (for value bootstrapping before
    retiring on the next tick).
    """

    alive: torch.Tensor
    life_start: torch.Tensor
    died: torch.Tensor
    truncated: torch.Tensor
    in_respawn_delay: torch.Tensor
    length: torch.Tensor
    mass_eaten: torch.Tensor
    killer_id: torch.Tensor
    kill_credit: torch.Tensor
    life_age: torch.Tensor


def step_delta_fp_torch(dist_fp: torch.Tensor, trig_q14: torch.Tensor) -> torch.Tensor:
    """Compute symmetric nearest-integer fixed-point step ``round(dist_fp * trig_q14 / 2^14)``."""
    prod = dist_fp * trig_q14
    pos = (prod + TRIG_HALF) >> TRIG_SHIFT
    neg = -((-prod + TRIG_HALF) >> TRIG_SHIFT)
    return torch.where(prod >= 0, pos, neg)


def _build_static_tables(cfg: GameV3Config, device: torch.device) -> dict[str, Any]:
    """Precompute shared integer trig, length, disc-overlap, pickup, and spawn tables."""
    c = cfg.cell_size
    sin_np, cos_np = build_trig_tables_np(cfg.angle_steps)
    lt_np = build_length_tables_np(cfg)
    fb_np = build_fallback_spawns_np(cfg)

    r_sq_np = lt_np["r_sq_cells"]
    max_r_cells = int(math.floor(cfg.max_body_radius / c + 1e-12))
    offsets: list[tuple[int, int, int]] = []
    for dy in range(-max_r_cells, max_r_cells + 1):
        for dx in range(-max_r_cells, max_r_cells + 1):
            d2 = dx * dx + dy * dy
            if d2 <= max_r_cells * max_r_cells:
                offsets.append((d2, dy, dx))
    offsets.sort()
    disc_dist_sq_np = np.array([o[0] for o in offsets], dtype=np.int32)
    disc_dy_np = np.array([o[1] for o in offsets], dtype=np.int32)
    disc_dx_np = np.array([o[2] for o in offsets], dtype=np.int32)

    # Compact radius-tier LUT for exact cell-disc intersection between post-move points.
    unique_r_sq = sorted({int(x) for x in r_sq_np.tolist()})
    tier_map = {val: idx for idx, val in enumerate(unique_r_sq)}
    r_tier_np = np.array([tier_map[int(v)] for v in r_sq_np], dtype=np.int64)

    max_diam = 2 * max_r_cells
    span = 2 * max_diam + 1
    num_tiers = len(unique_r_sq)
    overlap_lut_np = np.zeros((num_tiers, num_tiers, span, span), dtype=bool)
    tier_masks = [
        set(
            zip(
                disc_dx_np[disc_dist_sq_np <= r_sq].tolist(),
                disc_dy_np[disc_dist_sq_np <= r_sq].tolist(),
            )
        )
        for r_sq in unique_r_sq
    ]
    for ti in range(num_tiers):
        pts_i = tier_masks[ti]
        for tj in range(num_tiers):
            pts_j = tier_masks[tj]
            for dgx in range(-max_diam, max_diam + 1):
                for dgy in range(-max_diam, max_diam + 1):
                    shifted_j = {(x + dgx, y + dgy) for (x, y) in pts_j}
                    if pts_i & shifted_j:
                        overlap_lut_np[ti, tj, dgx + max_diam, dgy + max_diam] = True

    # Candidate cell stencil for exact integer squared-distance food pickup.
    max_eat_r_cells = int(math.ceil((cfg.food_pickup_mult * cfg.max_body_radius) / c)) + 1
    eat_offsets: list[tuple[int, int, int]] = []
    for dy in range(-max_eat_r_cells, max_eat_r_cells + 1):
        for dx in range(-max_eat_r_cells, max_eat_r_cells + 1):
            d2 = dx * dx + dy * dy
            if d2 <= max_eat_r_cells * max_eat_r_cells:
                eat_offsets.append((d2, dy, dx))
    eat_offsets.sort()
    eat_dy_np = np.array([o[1] for o in eat_offsets], dtype=np.int32)
    eat_dx_np = np.array([o[2] for o in eat_offsets], dtype=np.int32)

    r_clear = max(
        int(math.ceil(cfg.body_radius_f64(cfg.start_length) / c)) + 2,
        cfg.respawn_clearance_cells,
    )
    clear_offsets: list[tuple[int, int]] = []
    for dy in range(-r_clear, r_clear + 1):
        for dx in range(-r_clear, r_clear + 1):
            if dx * dx + dy * dy <= r_clear * r_clear:
                clear_offsets.append((dx, dy))
    clear_dx_np = np.array([o[0] for o in clear_offsets], dtype=np.int32)
    clear_dy_np = np.array([o[1] for o in clear_offsets], dtype=np.int32)

    start_r_fp = int(lt_np["radius_fp"][cfg.start_length])
    spawn_margin_fp = int(round((cfg.spawn_wall_margin / c) * (1 << FP_SHIFT)))
    spawn_lim_fp = max(1 << FP_SHIFT, cfg.arena_radius_fp - start_r_fp - spawn_margin_fp)
    spawn_wall_sq_fp = int(spawn_lim_fp) * int(spawn_lim_fp)

    food_margin_fp = int(round((cfg.food_wall_margin / c) * (1 << FP_SHIFT)))
    food_lim_fp = max(1 << FP_SHIFT, cfg.arena_radius_fp - food_margin_fp)
    food_wall_sq_fp = int(food_lim_fp) * int(food_lim_fp)

    out: dict[str, Any] = {
        "sin_table": torch.as_tensor(sin_np, device=device),
        "cos_table": torch.as_tensor(cos_np, device=device),
        "fallback_spawns": torch.as_tensor(fb_np, device=device),
        "r_tier_table": torch.as_tensor(r_tier_np, device=device),
        "disc_dx": torch.as_tensor(disc_dx_np, device=device),
        "disc_dy": torch.as_tensor(disc_dy_np, device=device),
        "disc_dist_sq": torch.as_tensor(disc_dist_sq_np, device=device),
        "overlap_lut": torch.as_tensor(overlap_lut_np, device=device),
        "eat_dx": torch.as_tensor(eat_dx_np, device=device),
        "eat_dy": torch.as_tensor(eat_dy_np, device=device),
        "clear_dx": torch.as_tensor(clear_dx_np, device=device),
        "clear_dy": torch.as_tensor(clear_dy_np, device=device),
        "max_diam": max_diam,
        "r_clear": r_clear,
        "spawn_wall_sq_fp": spawn_wall_sq_fp,
        "food_wall_sq_fp": food_wall_sq_fp,
    }
    for k, arr in lt_np.items():
        out[k] = torch.as_tensor(arr, device=device)
    return out


class GameV3Sim:
    """Batched GPU-first Game V3.1 fixed-point simulator with order-independent occupancy.

    All world state uses exact integer representations (``int32`` coordinates in
    ``1/1024`` units, ``int32`` angle indices in ``[0, angle_steps)``, shared ``int32``
    trig/length tables, integer mass/pellet grids) so trajectories match bit-for-bit
    across plain NumPy, Torch CPU, Torch MPS, and Torch CUDA.
    """

    def __init__(
        self,
        cfg: GameV3Config | None = None,
        num_envs: int = 1,
        *,
        seeds: Sequence[int] | torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.cfg: GameV3Config = cfg if cfg is not None else GameV3Config()
        req_dev = torch.device(device)
        self.device: torch.device = (
            torch.device(req_dev.type, 0)
            if req_dev.type in ("cuda", "mps") and req_dev.index is None
            else req_dev
        )
        self.E: int = int(num_envs)
        self.N: int = int(self.cfg.num_snakes)
        self.G: int = self.cfg.grid_size
        self.G4: int = self.G // 4
        self.G16: int = self.G // 16
        self.half_grid: int = self.cfg.half_grid
        self.max_pts: int = self.cfg.max_points_per_tick

        tables = _build_static_tables(self.cfg, self.device)
        self._sin_table: torch.Tensor = tables["sin_table"]
        self._cos_table: torch.Tensor = tables["cos_table"]
        self._speed_fp_table: torch.Tensor = tables["speed_fp"]
        self._boost_speed_fp_table: torch.Tensor = tables["boost_speed_fp"]
        self._turn_q_table: torch.Tensor = tables["turn_quantum"]
        self._turn_deltas_table: torch.Tensor = tables["turn_deltas"]
        self._radius_fp_table: torch.Tensor = tables["radius_fp"]
        self._r_sq_table: torch.Tensor = tables["r_sq_cells"]
        self._r_tier_table: torch.Tensor = tables["r_tier_table"]
        self._pickup_r_fp_table: torch.Tensor = tables["pickup_r_fp"]
        self._pickup_r_sq_fp_table: torch.Tensor = tables["pickup_r_sq_fp"]
        self._wall_limit_sq_fp_table: torch.Tensor = tables["wall_limit_sq_fp"]
        self._corpse_mass_table: torch.Tensor = tables["corpse_mass"]
        self._size_bucket_table: torch.Tensor = tables["size_bucket"]
        self._len_log_f32_table: torch.Tensor = tables["len_log_f32"]
        self._radius_norm_f32_table: torch.Tensor = tables["radius_norm_f32"]
        self._turn_norm_f32_table: torch.Tensor = tables["turn_norm_f32"]
        self._fallback_spawns: torch.Tensor = tables["fallback_spawns"]

        self._disc_dx: torch.Tensor = tables["disc_dx"]
        self._disc_dy: torch.Tensor = tables["disc_dy"]
        self._disc_dist_sq: torch.Tensor = tables["disc_dist_sq"]
        self._overlap_lut: torch.Tensor = tables["overlap_lut"]
        self._eat_dx: torch.Tensor = tables["eat_dx"]
        self._eat_dy: torch.Tensor = tables["eat_dy"]
        self._clear_dx: torch.Tensor = tables["clear_dx"]
        self._clear_dy: torch.Tensor = tables["clear_dy"]
        self._max_diam: int = int(tables["max_diam"])
        self._r_clear: int = int(tables["r_clear"])
        self._spawn_wall_sq_fp: int = int(tables["spawn_wall_sq_fp"])
        self._food_wall_sq_fp: int = int(tables["food_wall_sq_fp"])

        # Preallocated constant index helpers.
        self._snake_ids = torch.arange(self.N, dtype=torch.int32, device=self.device)[None, :]
        self._snake_ids_i64 = self._snake_ids.to(torch.int64)
        self._not_eye_nn = ~torch.eye(self.N, dtype=torch.bool, device=self.device)[None, :, :]
        self._tril_nn = torch.tril(
            torch.ones((self.N, self.N), dtype=torch.bool, device=self.device), diagonal=-1
        )[None, :, :]
        self._pts_idx = torch.arange(self.max_pts, dtype=torch.int32, device=self.device)[
            None, None, :
        ]

        m_amb = self.cfg.ambient_spawn_budget_per_tick
        self._amb_slots = torch.arange(m_amb, dtype=torch.int64, device=self.device)[None, :]
        self._tril_amb = torch.tril(
            torch.ones((m_amb, m_amb), dtype=torch.bool, device=self.device), diagonal=-1
        )[None, :, :]
        self._corpse_k = torch.arange(
            self.cfg.corpse_max_pellets, dtype=torch.int32, device=self.device
        )[None, None, :]

        k_resp = self.cfg.respawn_candidates
        resp_slots = (
            torch.arange(self.N, dtype=torch.int64, device=self.device)[:, None] * k_resp
            + torch.arange(k_resp, dtype=torch.int64, device=self.device)[None, :]
        )[None, :, :]
        self._resp_slot_x = resp_slots * 3
        self._resp_slot_y = resp_slots * 3 + 1
        self._resp_slot_t = resp_slots * 3 + 2
        f_resp = self.cfg.respawn_fallback_candidates
        self._fb_f_offsets = torch.arange(f_resp, dtype=torch.int64, device=self.device)[
            None, None, :
        ] * max(1, FALLBACK_RING_SIZE // max(1, f_resp))

        env_idx = torch.arange(self.E, dtype=torch.int64, device=self.device)[:, None]
        self._env_base_1x = env_idx * (self.G * self.G)
        self._env_base_4x = env_idx * (self.G4 * self.G4)
        self._env_base_16x = env_idx * (self.G16 * self.G16)

        # Mutable fixed-point simulation state tensors (zero floating-point state).
        es = (self.E, self.N)
        max_len = self.cfg.max_length
        self.seeds = torch.zeros((self.E,), dtype=torch.int64, device=self.device)
        self.tick: int = 0

        self.head_pos = torch.zeros((self.E, self.N, 2), dtype=torch.int32, device=self.device)
        self.theta = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.alive = torch.zeros(es, dtype=torch.bool, device=self.device)
        self.boosting = torch.zeros(es, dtype=torch.bool, device=self.device)
        self.length = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.seg_count = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.head_ptr = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.oldest_active_subtick = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.dist_accum_fp = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.boost_ticks = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.boost_drop_fp_accum = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.respawn_timer = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.life_age = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.life_start = torch.zeros(es, dtype=torch.bool, device=self.device)
        self.truncated_pending = torch.zeros(es, dtype=torch.bool, device=self.device)

        self.ring_pos = torch.zeros(
            (self.E, self.N, max_len, 2), dtype=torch.int32, device=self.device
        )
        self.ring_cell = torch.zeros(
            (self.E, self.N, max_len, 2), dtype=torch.int16, device=self.device
        )
        self.ring_r_sq = torch.zeros(
            (self.E, self.N, max_len), dtype=torch.int16, device=self.device
        )
        self.ring_subtick = torch.zeros(
            (self.E, self.N, max_len), dtype=torch.int32, device=self.device
        )

        # Previous-tick head state for 2-frame head observation stacking.
        self.prev_alive = torch.zeros(es, dtype=torch.bool, device=self.device)
        self.prev_boosting = torch.zeros(es, dtype=torch.bool, device=self.device)
        self.prev_head_pos = torch.zeros((self.E, self.N, 2), dtype=torch.int32, device=self.device)
        self.prev_theta = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.prev_length = torch.zeros(es, dtype=torch.int32, device=self.device)

        # Order-independent dual-bound occupancy grids at 1x:
        # - occ_hi stores max((subtick << 6) | slot_id)
        # - occ_lo stores max((subtick << 6) | (63 - slot_id))
        # Both updated exclusively via commutative scatter_reduce_("amax") so slot write
        # order never affects results and own body never hides an enemy on the same cell.
        self.occ_hi = torch.full(
            (self.E, self.G, self.G), -1, dtype=torch.int32, device=self.device
        )
        self.occ_lo = torch.full(
            (self.E, self.G, self.G), -1, dtype=torch.int32, device=self.device
        )
        # Exact max-pooled coarse body occupancy pyramids: (E, G4, G4, N) & (E, G16, G16, N)
        # stored N-last so ego-pixel index_select(0, ...) reads contiguous N-element vectors.
        self.occ_subtick_4x = torch.full(
            (self.E, self.G4, self.G4, self.N), -1, dtype=torch.int32, device=self.device
        )
        self.occ_subtick_16x = torch.full(
            (self.E, self.G16, self.G16, self.N), -1, dtype=torch.int32, device=self.device
        )

        # Scratch grid for contested-pellet tie-breaking (lowest surviving slot wins).
        self._claim_grid = torch.full(
            (self.E, self.G * self.G), self.N, dtype=torch.int32, device=self.device
        )

        # Incremental count/mass-pooled food grids (1x, 4x, 16x) in int32.
        self.ambient_grid = torch.zeros(
            (self.E, self.G, self.G), dtype=torch.int32, device=self.device
        )
        self.ambient_grid_4x = torch.zeros(
            (self.E, self.G4, self.G4), dtype=torch.int32, device=self.device
        )
        self.ambient_grid_16x = torch.zeros(
            (self.E, self.G16, self.G16), dtype=torch.int32, device=self.device
        )
        self.corpse_grid = torch.zeros(
            (self.E, self.G, self.G), dtype=torch.int32, device=self.device
        )
        self.corpse_grid_4x = torch.zeros(
            (self.E, self.G4, self.G4), dtype=torch.int32, device=self.device
        )
        self.corpse_grid_16x = torch.zeros(
            (self.E, self.G16, self.G16), dtype=torch.int32, device=self.device
        )
        self.ambient_count = torch.zeros((self.E,), dtype=torch.int32, device=self.device)

        # Exact integer mass-accounting ledgers per world (sources and sinks).
        self.total_spawned_snake_mass = torch.zeros(
            (self.E,), dtype=torch.int64, device=self.device
        )
        self.total_spawned_ambient_mass = torch.zeros(
            (self.E,), dtype=torch.int64, device=self.device
        )
        self.total_corpse_decay_sink = torch.zeros((self.E,), dtype=torch.int64, device=self.device)
        self.total_boost_decay_sink = torch.zeros((self.E,), dtype=torch.int64, device=self.device)
        self.total_length_cap_sink = torch.zeros((self.E,), dtype=torch.int64, device=self.device)
        self.total_truncation_sink = torch.zeros((self.E,), dtype=torch.int64, device=self.device)

        self._timing_enabled: bool = False
        self._stage_seconds: dict[str, float] = {
            "move": 0.0,
            "collide": 0.0,
            "rasterise": 0.0,
            "food": 0.0,
            "respawn": 0.0,
        }

        self.reset(seeds=seeds)

    @property
    def num_actions(self) -> int:
        """Number of discrete actions exposed by the simulator (10)."""
        return NUM_ACTIONS

    def set_stage_timing(self, enabled: bool) -> None:
        """Enable or disable synchronized per-stage wall-clock timing."""
        self._timing_enabled = bool(enabled)
        for k in self._stage_seconds:
            self._stage_seconds[k] = 0.0

    def consume_stage_timing(self) -> dict[str, float]:
        """Return accumulated stage seconds and reset counters to zero."""
        out = dict(self._stage_seconds)
        for k in self._stage_seconds:
            self._stage_seconds[k] = 0.0
        return out

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _cell_center_fp(self, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        cx = ((gx.to(torch.int32) - self.half_grid) << FP_SHIFT) + FP_HALF
        cy = ((gy.to(torch.int32) - self.half_grid) << FP_SHIFT) + FP_HALF
        return torch.stack((cx, cy), dim=-1)

    def _cell_center_sq_fp(self, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        cx = ((gx.to(torch.int64) - self.half_grid) << FP_SHIFT) + FP_HALF
        cy = ((gy.to(torch.int64) - self.half_grid) << FP_SHIFT) + FP_HALF
        return cx * cx + cy * cy

    def _is_cell_occupied(
        self, flat_idx: torch.Tensor, oldest_subtick: torch.Tensor
    ) -> torch.Tensor:
        """Return bool mask of shape ``flat_idx.shape`` true where any snake is active."""
        e_cnt = flat_idx.shape[0]
        flat_2d = flat_idx.reshape(e_cnt, -1)
        val_hi = self.occ_hi.view(e_cnt, -1).gather(1, flat_2d)
        ow_hi = (val_hi & 63).clamp(0, self.N - 1).to(torch.int64)
        act_hi = (val_hi >= 0) & ((val_hi >> 6) >= oldest_subtick.gather(1, ow_hi))

        val_lo = self.occ_lo.view(e_cnt, -1).gather(1, flat_2d)
        ow_lo = (63 - (val_lo & 63)).clamp(0, self.N - 1).to(torch.int64)
        act_lo = (val_lo >= 0) & ((val_lo >> 6) >= oldest_subtick.gather(1, ow_lo))
        return (act_hi | act_lo).view_as(flat_idx)

    def _stamp_discs(
        self,
        gx: torch.Tensor,
        gy: torch.Tensor,
        r_sq: torch.Tensor,
        active: torch.Tensor,
        subtick: torch.Tensor,
        *,
        slot_order: torch.Tensor | None = None,
    ) -> None:
        """Order-independently stamp body discs into 1x (occ_hi/occ_lo), 4x, and 16x pyramids.

        Args:
            gx: Int32 cell x tensor of shape ``(E, N, P)`` (``P >= 1`` points per snake).
            gy: Int32 cell y tensor of shape ``(E, N, P)``.
            r_sq: Int32 squared cell radius tensor of shape ``(E, N, P)``.
            active: Bool validity tensor of shape ``(E, N, P)``.
            subtick: Int32 subtick sequence tensor of shape ``(E, N, P)``.
            slot_order: Optional permutation ``(N,)`` of ``0..N-1`` to verify slot-write
                order independence in tests.
        """
        if slot_order is not None:
            perm = slot_order.to(device=self.device, dtype=torch.int64)
            gx = gx.index_select(1, perm)
            gy = gy.index_select(1, perm)
            r_sq = r_sq.index_select(1, perm)
            active = active.index_select(1, perm)
            subtick = subtick.index_select(1, perm)
            slot_ids = self._snake_ids.index_select(1, perm)
        else:
            slot_ids = self._snake_ids

        sx = gx[..., None] + self._disc_dx[None, None, None, :]
        sy = gy[..., None] + self._disc_dy[None, None, None, :]
        in_b = (sx >= 0) & (sx < self.G) & (sy >= 0) & (sy < self.G)
        in_r = self._disc_dist_sq[None, None, None, :] <= r_sq[..., None]
        valid = active[..., None] & in_b & in_r

        sx_c = sx.clamp(0, self.G - 1).to(torch.int64)
        sy_c = sy.clamp(0, self.G - 1).to(torch.int64)
        flat_1x = (sy_c * self.G + sx_c).reshape(self.E, -1)

        st_shifted = subtick[..., None] << 6
        sid = slot_ids[:, :, None, None]
        neg_one = torch.full((1,), -1, dtype=torch.int32, device=self.device)

        src_hi = torch.where(valid, st_shifted | sid, neg_one).reshape(self.E, -1)
        src_lo = torch.where(valid, st_shifted | (63 - sid), neg_one).reshape(self.E, -1)
        self.occ_hi.view(self.E, -1).scatter_reduce_(
            1, flat_1x, src_hi, reduce="amax", include_self=True
        )
        self.occ_lo.view(self.E, -1).scatter_reduce_(
            1, flat_1x, src_lo, reduce="amax", include_self=True
        )

        # Stamp per-snake 4x and 16x max-pooled subtick pyramids (N-last contiguous layout).
        src_st = torch.where(valid, subtick[..., None].expand_as(sx), neg_one).reshape(self.E, -1)
        sid_i64 = sid.to(torch.int64)
        flat_4x_n = (((sy_c >> 2) * self.G4 + (sx_c >> 2)) * self.N + sid_i64).reshape(self.E, -1)
        flat_16x_n = (((sy_c >> 4) * self.G16 + (sx_c >> 4)) * self.N + sid_i64).reshape(self.E, -1)
        self.occ_subtick_4x.view(self.E, -1).scatter_reduce_(
            1, flat_4x_n, src_st, reduce="amax", include_self=True
        )
        self.occ_subtick_16x.view(self.E, -1).scatter_reduce_(
            1, flat_16x_n, src_st, reduce="amax", include_self=True
        )

    def _spawn_ambient_batch(
        self,
        purpose: int,
        tick_val: int,
        batch_offset: int,
        budget: int,
    ) -> None:
        """Deterministically spawn up to ``budget`` ambient pellets per world."""
        if budget == self.cfg.ambient_spawn_budget_per_tick and batch_offset == 0:
            slots = self._amb_slots
            tril = self._tril_amb
        else:
            slots = torch.arange(budget, dtype=torch.int64, device=self.device)[None, :] + int(
                batch_offset
            )
            tril = torch.tril(
                torch.ones((budget, budget), dtype=torch.bool, device=self.device), diagonal=-1
            )[None, :, :]

        seeds_col = self.seeds[:, None]
        gx = rng_int_torch(seeds_col, tick_val, purpose, slots * 2, self.G)
        gy = rng_int_torch(seeds_col, tick_val, purpose, slots * 2 + 1, self.G)

        in_grid = (gx >= 0) & (gx < self.G) & (gy >= 0) & (gy < self.G)
        in_arena = in_grid & (self._cell_center_sq_fp(gx, gy) < self._food_wall_sq_fp)

        gx_c = gx.clamp(0, self.G - 1).to(torch.int64)
        gy_c = gy.clamp(0, self.G - 1).to(torch.int64)
        flat = gy_c * self.G + gx_c

        occupied = self._is_cell_occupied(flat, self.oldest_active_subtick)

        amb_val = self.ambient_grid.view(self.E, -1).gather(1, flat)
        cor_val = self.corpse_grid.view(self.E, -1).gather(1, flat)
        food_free = (amb_val == 0) & (cor_val == 0)

        same = flat[:, :, None] == flat[:, None, :]
        unique = ~(same & tril).any(dim=2)

        cand_valid = in_arena & (~occupied) & food_free & unique
        deficit = (int(self.cfg.target_ambient_pellets) - self.ambient_count).clamp_min(0)
        rank = cand_valid.to(torch.int32).cumsum(dim=1) - 1
        accept = cand_valid & (rank < deficit[:, None])

        pellet_i32 = int(self.cfg.ambient_pellet_value)
        add_i32 = torch.where(accept, pellet_i32, 0).to(torch.int32)
        add_flat = add_i32.reshape(-1)
        self.ambient_grid.view(-1).index_put_(
            ((self._env_base_1x + flat).reshape(-1),), add_flat, accumulate=True
        )

        flat4 = (gy_c >> 2) * self.G4 + (gx_c >> 2)
        self.ambient_grid_4x.view(-1).index_put_(
            ((self._env_base_4x + flat4).reshape(-1),), add_flat, accumulate=True
        )
        flat16 = (gy_c >> 4) * self.G16 + (gx_c >> 4)
        self.ambient_grid_16x.view(-1).index_put_(
            ((self._env_base_16x + flat16).reshape(-1),), add_flat, accumulate=True
        )

        added_count = accept.to(torch.int32).sum(dim=1)
        self.ambient_count = self.ambient_count + added_count
        self.total_spawned_ambient_mass = (
            self.total_spawned_ambient_mass + added_count.to(torch.int64) * pellet_i32
        )

    def reset(self, seeds: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset all E worlds deterministically from per-world int64 seeds."""
        if seeds is None:
            seed_tensor = torch.arange(self.E, dtype=torch.int64, device=self.device)
        elif isinstance(seeds, torch.Tensor):
            seed_tensor = seeds.to(device=self.device, dtype=torch.int64)
        else:
            seed_tensor = torch.as_tensor(list(seeds), dtype=torch.int64, device=self.device)
        if seed_tensor.shape != (self.E,):
            raise ValueError(f"seeds must have shape ({self.E},), got {tuple(seed_tensor.shape)}")

        self.seeds.copy_(seed_tensor)
        self.tick = 0

        self.occ_hi.fill_(-1)
        self.occ_lo.fill_(-1)
        self.occ_subtick_4x.fill_(-1)
        self.occ_subtick_16x.fill_(-1)
        self._claim_grid.fill_(self.N)

        self.ambient_grid.zero_()
        self.ambient_grid_4x.zero_()
        self.ambient_grid_16x.zero_()
        self.corpse_grid.zero_()
        self.corpse_grid_4x.zero_()
        self.corpse_grid_16x.zero_()
        self.ambient_count.zero_()

        self.total_spawned_snake_mass.zero_()
        self.total_spawned_ambient_mass.zero_()
        self.total_corpse_decay_sink.zero_()
        self.total_boost_decay_sink.zero_()
        self.total_length_cap_sink.zero_()
        self.total_truncation_sink.zero_()

        self.ring_pos.zero_()
        self.ring_cell.zero_()
        self.ring_r_sq.zero_()
        self.ring_subtick.zero_()
        self.dist_accum_fp.zero_()
        self.boost_ticks.zero_()
        self.boost_drop_fp_accum.zero_()
        self.respawn_timer.zero_()
        self.head_ptr.zero_()
        self.boosting.fill_(False)
        self.life_age.zero_()
        self.truncated_pending.fill_(False)

        # Initial snake placement: K=32 counter-RNG candidates + 64 deterministic fallback spots.
        k_init = _INIT_SNAKE_CANDIDATES
        slots = (
            torch.arange(self.N, dtype=torch.int64, device=self.device)[:, None] * k_init
            + torch.arange(k_init, dtype=torch.int64, device=self.device)[None, :]
        )[None, :, :]
        seeds_3d = self.seeds[:, None, None]
        rng_gx = rng_int_torch(seeds_3d, 0, PURPOSE_INIT_SNAKE, slots * 3, self.G)
        rng_gy = rng_int_torch(seeds_3d, 0, PURPOSE_INIT_SNAKE, slots * 3 + 1, self.G)
        rng_th = rng_angle_idx_torch(
            seeds_3d, 0, PURPOSE_INIT_SNAKE, slots * 3 + 2, self.cfg.angle_steps
        )

        fb_idx = (
            torch.arange(self.N, dtype=torch.int64, device=self.device)[:, None] * 7
            + torch.arange(FALLBACK_RING_SIZE, dtype=torch.int64, device=self.device)[None, :]
        ) & (FALLBACK_RING_SIZE - 1)
        fb_sel = self._fallback_spawns[fb_idx][None, :, :, :].expand(
            self.E, self.N, FALLBACK_RING_SIZE, 3
        )
        cand_gx = torch.cat((rng_gx, fb_sel[..., 0]), dim=2)
        cand_gy = torch.cat((rng_gy, fb_sel[..., 1]), dim=2)
        cand_th = torch.cat((rng_th, fb_sel[..., 2]), dim=2)

        in_wall = (
            (cand_gx >= 0)
            & (cand_gx < self.G)
            & (cand_gy >= 0)
            & (cand_gy < self.G)
            & (self._cell_center_sq_fp(cand_gx, cand_gy) < self._spawn_wall_sq_fp)
        )

        placed_gx = torch.zeros((self.E, self.N), dtype=torch.int32, device=self.device)
        placed_gy = torch.zeros((self.E, self.N), dtype=torch.int32, device=self.device)
        placed_th = torch.zeros((self.E, self.N), dtype=torch.int32, device=self.device)
        placed_alive = torch.zeros((self.E, self.N), dtype=torch.bool, device=self.device)
        min_sep_sq = (2 * self._r_clear) * (2 * self._r_clear)

        for i in range(self.N):
            valid_i = in_wall[:, i, :]
            if i > 0:
                dx = cand_gx[:, i, :, None] - placed_gx[:, None, :i]
                dy = cand_gy[:, i, :, None] - placed_gy[:, None, :i]
                already = placed_alive[:, None, :i]
                clear = ((~already) | ((dx * dx + dy * dy) > min_sep_sq)).all(dim=-1)
                valid_i = valid_i & clear
            has_i = valid_i.any(dim=-1)
            best_m = valid_i.to(torch.int32).argmax(dim=-1, keepdim=True)
            placed_gx[:, i] = cand_gx[:, i, :].gather(1, best_m).squeeze(1)
            placed_gy[:, i] = cand_gy[:, i, :].gather(1, best_m).squeeze(1)
            placed_th[:, i] = cand_th[:, i, :].gather(1, best_m).squeeze(1)
            placed_alive[:, i] = has_i

        self.head_pos.copy_(self._cell_center_fp(placed_gx, placed_gy))
        self.theta.copy_(placed_th)
        self.alive.copy_(placed_alive)
        self.life_start.copy_(placed_alive)
        self.length = torch.where(
            placed_alive,
            torch.full_like(self.length, int(self.cfg.start_length)),
            torch.zeros_like(self.length),
        )
        self.seg_count = torch.where(
            placed_alive, torch.ones_like(self.seg_count), torch.zeros_like(self.seg_count)
        )
        self.respawn_timer = torch.where(
            placed_alive,
            torch.zeros_like(self.respawn_timer),
            torch.full_like(self.respawn_timer, int(self.cfg.respawn_delay)),
        )
        self.oldest_active_subtick = torch.where(
            placed_alive,
            torch.ones_like(self.oldest_active_subtick),
            torch.full_like(self.oldest_active_subtick, 2),
        )

        init_r_sq = self._r_sq_table[int(self.cfg.start_length)].expand(self.E, self.N)
        self.ring_pos[:, :, 0, :] = self.head_pos
        self.ring_cell[:, :, 0, 0] = placed_gx.to(torch.int16)
        self.ring_cell[:, :, 0, 1] = placed_gy.to(torch.int16)
        self.ring_r_sq[:, :, 0] = init_r_sq.to(torch.int16)
        self.ring_subtick[:, :, 0] = torch.where(
            placed_alive,
            torch.ones_like(self.oldest_active_subtick),
            torch.zeros_like(self.oldest_active_subtick),
        )

        self._stamp_discs(
            placed_gx[:, :, None],
            placed_gy[:, :, None],
            init_r_sq[:, :, None],
            placed_alive[:, :, None],
            torch.ones((self.E, self.N, 1), dtype=torch.int32, device=self.device),
        )
        self.total_spawned_snake_mass = placed_alive.to(torch.int64).sum(dim=1) * int(
            self.cfg.start_length
        )

        # No t = -1 frame exists immediately after reset.
        self.prev_alive.fill_(False)
        self.prev_boosting.fill_(False)
        self.prev_head_pos.copy_(self.head_pos)
        self.prev_theta.copy_(self.theta)
        self.prev_length.copy_(self.length)

        # Populate initial ambient food up to target_ambient_pellets.
        num_batches = max(
            1,
            int(math.ceil(self.cfg.target_ambient_pellets * 2.0 / _INIT_FOOD_BATCH_SIZE)) + 2,
        )
        for b in range(num_batches):
            self._spawn_ambient_batch(
                PURPOSE_INIT_FOOD,
                0,
                b * _INIT_FOOD_BATCH_SIZE,
                _INIT_FOOD_BATCH_SIZE,
            )

    def step(self, actions: torch.Tensor) -> StepInfo:
        """Advance all E worlds by one tick without any host synchronization.

        Exact within-tick order (Rule 9):
          0. Retire any snakes flagged ``truncated_pending`` on the previous tick
             (so their terminal state was observable for value bootstrapping).
          1. Move all alive snakes in fixed-point and emit distance-spaced body points
             (one point per ``speed`` unit travelled; boost shedding & fractional drop).
          2. Lay bodies from post-move positions including snakes dying this tick.
          3. Test every head against walls and all other snakes' retained + post-move
             bodies/heads (geometric head-to-head; both die on simultaneous hit).
          4. Apply all deaths at once; award kill credit to smallest other owner id.
          5. Resolve food eating on integer squared pickup distances (contested pellet
             goes to lowest surviving slot); drop boost pellets and corpses AFTER deaths;
             refill ambient food.
          6. Advance respawn timers and spawn ready slots from K counter-RNG candidates
             plus deterministic fallback ring; emit episode hooks.
        """
        if actions.shape != (self.E, self.N):
            raise ValueError(
                f"actions must have shape ({self.E}, {self.N}), got {tuple(actions.shape)}"
            )
        if actions.device != self.device:
            raise ValueError(f"actions device {actions.device} != sim device {self.device}")

        timed = self._timing_enabled
        if timed:
            self._sync_device()
            t0 = time.perf_counter()

        self.tick += 1
        tick_base_subtick = self.max_pts * self.tick + 2
        expired_sentinel_st = tick_base_subtick + self.max_pts + 1

        # Retire snakes truncated at the end of the previous tick (not deaths; no corpse).
        retire = self.truncated_pending & self.alive
        retire_mass = torch.where(retire, self.length, 0).to(torch.int64).sum(dim=1)
        self.total_truncation_sink = self.total_truncation_sink + retire_mass
        self.alive = self.alive & (~retire)
        self.boosting = self.boosting & (~retire)
        self.length = torch.where(retire, 0, self.length)
        self.seg_count = torch.where(retire, 0, self.seg_count)
        self.oldest_active_subtick = torch.where(
            retire, expired_sentinel_st, self.oldest_active_subtick
        )
        self.dist_accum_fp = torch.where(retire, 0, self.dist_accum_fp)
        self.boost_ticks = torch.where(retire, 0, self.boost_ticks)
        self.boost_drop_fp_accum = torch.where(retire, 0, self.boost_drop_fp_accum)
        self.life_age = torch.where(retire, 0, self.life_age)
        # Set timer=1 so Step 5 decrements to 0 and immediately spawns a fresh life this tick.
        self.respawn_timer = torch.where(retire, 1, self.respawn_timer)
        self.truncated_pending.fill_(False)

        # Snapshot pre-step head pose for 2-frame head observation stacking.
        self.prev_alive.copy_(self.alive)
        self.prev_boosting.copy_(self.boosting)
        self.prev_head_pos.copy_(self.head_pos)
        self.prev_theta.copy_(self.theta)
        self.prev_length.copy_(self.length)

        # ---------------- Stage 1: move ----------------
        act = actions.to(torch.int32).clamp(0, NUM_ACTIONS - 1)
        steer = act.remainder(NUM_TURN_LEVELS).to(torch.int64)
        want_boost = act >= NUM_TURN_LEVELS

        cur_len_i64 = self.length.clamp(0, self.cfg.max_length).to(torch.int64)
        dtheta = self._turn_deltas_table[cur_len_i64].gather(2, steer[:, :, None]).squeeze(2)
        angle_mask = int(self.cfg.angle_steps - 1)
        new_theta = torch.where(self.alive, (self.theta + dtheta) & angle_mask, self.theta)

        boosting = self.alive & want_boost & (self.length > int(self.cfg.min_boost_length))
        base_spd_fp = self._speed_fp_table[cur_len_i64]
        boost_spd_fp = self._boost_speed_fp_table[cur_len_i64]
        step_fp = torch.where(boosting, boost_spd_fp, base_spd_fp)

        th_i64 = new_theta.to(torch.int64)
        sin_t = self._sin_table[th_i64]
        cos_t = self._cos_table[th_i64]

        old_hx = self.head_pos[:, :, 0]
        old_hy = self.head_pos[:, :, 1]
        cand_hx = old_hx + step_delta_fp_torch(step_fp, cos_t)
        cand_hy = old_hy + step_delta_fp_torch(step_fp, sin_t)

        # Emit distance-spaced body points: 1 point per `speed` unit travelled.
        old_accum = self.dist_accum_fp
        tot_dist = old_accum + step_fp
        num_pts = torch.where(
            self.alive,
            torch.div(tot_dist, base_spd_fp, rounding_mode="floor"),
            torch.zeros_like(tot_dist),
        )
        new_accum = tot_dist - num_pts * base_spd_fp

        pt_valid = self.alive[:, :, None] & (self._pts_idx < num_pts[:, :, None])
        d_pts = ((self._pts_idx + 1) * base_spd_fp[:, :, None] - old_accum[:, :, None]).clamp_min(0)
        pt_x = old_hx[:, :, None] + step_delta_fp_torch(d_pts, cos_t[:, :, None])
        pt_y = old_hy[:, :, None] + step_delta_fp_torch(d_pts, sin_t[:, :, None])
        pt_gx = (pt_x >> FP_SHIFT) + self.half_grid
        pt_gy = (pt_y >> FP_SHIFT) + self.half_grid
        pt_subtick = tick_base_subtick + self._pts_idx

        # Capture pre-step tail cell before boost shedding / tail advance.
        old_seg = self.seg_count
        old_ptr = self.head_ptr
        max_len = int(self.cfg.max_length)
        tail_ring_idx = (old_ptr - (old_seg - 1).clamp_min(0)).remainder(max_len).to(torch.int64)
        old_tail_cell = self.ring_cell.gather(
            2, tail_ring_idx[:, :, None, None].expand(-1, -1, 1, 2)
        ).squeeze(2)
        old_tail_gx = old_tail_cell[:, :, 0].to(torch.int64)
        old_tail_gy = old_tail_cell[:, :, 1].to(torch.int64)

        # Boost mass shedding & fractional pellet accumulator (Rule 4).
        inc_boost = torch.where(boosting, self.boost_ticks + 1, self.boost_ticks)
        shed = boosting & (inc_boost >= int(self.cfg.boost_shed_ticks))
        self.boost_ticks = torch.where(shed, torch.zeros_like(inc_boost), inc_boost)
        shed_mass = int(self.cfg.boost_shed_mass)
        len_post_shed = torch.where(shed, self.length - shed_mass, self.length)
        self.length = torch.where(self.alive, len_post_shed, self.length)

        drop_fp_inc = int(self.cfg.boost_drop_fp_per_shed)
        acc_next = self.boost_drop_fp_accum + drop_fp_inc
        boost_drop_int = torch.where(
            shed, acc_next >> BOOST_ACCUM_SHIFT, torch.zeros_like(acc_next)
        )
        self.boost_drop_fp_accum = torch.where(
            shed, acc_next & (BOOST_ACCUM_ONE - 1), self.boost_drop_fp_accum
        )
        boost_sink = torch.where(shed, shed_mass - boost_drop_int, torch.zeros_like(boost_drop_int))
        self.total_boost_decay_sink = self.total_boost_decay_sink + boost_sink.to(torch.int64).sum(
            dim=1
        )

        # Retained pre-move segment count (including snakes that will die in Stage 2).
        target_seg = torch.minimum(
            torch.full_like(old_seg, max_len),
            torch.minimum(old_seg + num_pts, self.length),
        )
        keep_old = (target_seg - num_pts).clamp_min(0)
        kept_tail_idx = (old_ptr - (keep_old - 1).clamp_min(0)).remainder(max_len).to(torch.int64)
        kept_tail_st = self.ring_subtick.gather(2, kept_tail_idx[:, :, None]).squeeze(2)
        pre_death_oldest_st = torch.where(
            self.alive & (keep_old > 0),
            kept_tail_st,
            torch.full_like(kept_tail_st, expired_sentinel_st),
        )

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["move"] += t1 - t0
            t0 = t1

        # ---------------- Stage 2: lay post-move bodies & test heads ----------------
        len_idx = self.length.clamp(0, max_len).to(torch.int64)
        r_sq = self._r_sq_table[len_idx]
        r_tier = self._r_tier_table[len_idx]
        wall_lim_sq = self._wall_limit_sq_fp_table[len_idx]

        in_grid_pt = (pt_gx >= 0) & (pt_gx < self.G) & (pt_gy >= 0) & (pt_gy < self.G)
        pt_d_sq_fp = pt_x.to(torch.int64).square() + pt_y.to(torch.int64).square()
        died_wall = self.alive & (
            pt_valid & ((~in_grid_pt) | (pt_d_sq_fp >= wall_lim_sq[:, :, None]))
        ).any(dim=-1)

        # Geometric collision against OTHER alive snakes' post-move points (incl. heads & dying).
        ti = r_tier[:, :, None].expand(-1, -1, self.N)
        tj = r_tier[:, None, :].expand(-1, self.N, -1)
        hit_post_pair = torch.zeros((self.E, self.N, self.N), dtype=torch.bool, device=self.device)
        for ki in range(self.max_pts):
            vi = pt_valid[:, :, ki]
            gxi = pt_gx[:, :, ki]
            gyi = pt_gy[:, :, ki]
            for kj in range(self.max_pts):
                vj = pt_valid[:, :, kj]
                dgx = pt_gx[:, None, :, kj] - gxi[:, :, None]
                dgy = pt_gy[:, None, :, kj] - gyi[:, :, None]
                near = (
                    vi[:, :, None]
                    & vj[:, None, :]
                    & self._not_eye_nn
                    & (dgx.abs() <= self._max_diam)
                    & (dgy.abs() <= self._max_diam)
                )
                dgx_idx = (dgx + self._max_diam).clamp(0, 2 * self._max_diam).to(torch.int64)
                dgy_idx = (dgy + self._max_diam).clamp(0, 2 * self._max_diam).to(torch.int64)
                hit_post_pair = hit_post_pair | (near & self._overlap_lut[ti, tj, dgx_idx, dgy_idx])

        post_killer_min = torch.where(
            hit_post_pair,
            self._snake_ids[:, None, :].expand(self.E, self.N, self.N),
            torch.full((self.E, self.N, self.N), self.N, dtype=torch.int32, device=self.device),
        ).amin(dim=-1)

        # Geometric collision against all OTHER alive snakes' retained pre-move bodies (occ_hi/lo).
        hx_k = pt_gx[..., None] + self._disc_dx[None, None, None, :]
        hy_k = pt_gy[..., None] + self._disc_dy[None, None, None, :]
        in_b_k = (hx_k >= 0) & (hx_k < self.G) & (hy_k >= 0) & (hy_k < self.G)
        in_r_k = self._disc_dist_sq[None, None, None, :] <= r_sq[:, :, None, None]
        valid_k = pt_valid[..., None] & in_b_k & in_r_k

        flat_k = (
            hy_k.clamp(0, self.G - 1).to(torch.int64) * self.G
            + hx_k.clamp(0, self.G - 1).to(torch.int64)
        ).view(self.E, -1)

        occ_hi_k = self.occ_hi.view(self.E, -1).gather(1, flat_k).view_as(hx_k)
        ow_hi_k = occ_hi_k & 63
        ow_hi_i64 = ow_hi_k.clamp(0, self.N - 1).to(torch.int64)
        st_tail_hi = pre_death_oldest_st.gather(1, ow_hi_i64.view(self.E, -1)).view_as(hx_k)
        hit_hi = (
            valid_k
            & (occ_hi_k >= 0)
            & (ow_hi_k != self._snake_ids[:, :, None, None])
            & ((occ_hi_k >> 6) >= st_tail_hi)
        )

        occ_lo_k = self.occ_lo.view(self.E, -1).gather(1, flat_k).view_as(hx_k)
        ow_lo_k = 63 - (occ_lo_k & 63)
        ow_lo_i64 = ow_lo_k.clamp(0, self.N - 1).to(torch.int64)
        st_tail_lo = pre_death_oldest_st.gather(1, ow_lo_i64.view(self.E, -1)).view_as(hx_k)
        hit_lo = (
            valid_k
            & (occ_lo_k >= 0)
            & (ow_lo_k != self._snake_ids[:, :, None, None])
            & ((occ_lo_k >> 6) >= st_tail_lo)
        )

        full_n = torch.full_like(ow_hi_k, self.N)
        ret_killer_min = torch.minimum(
            torch.where(hit_hi, ow_hi_k, full_n),
            torch.where(hit_lo, ow_lo_k, full_n),
        ).amin(dim=(2, 3))

        best_enemy_killer = torch.minimum(post_killer_min, ret_killer_min)
        died_enemy = self.alive & (best_enemy_killer < self.N)
        died = self.alive & (died_wall | died_enemy)
        survives = self.alive & (~died)

        killer_id = torch.where(
            died_enemy, best_enemy_killer, torch.full_like(best_enemy_killer, -1)
        )
        kill_credit = (
            (killer_id[:, :, None] == self._snake_ids[:, None, :]).to(torch.int32).sum(dim=1)
        )

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["collide"] += t1 - t0
            t0 = t1

        # ---------------- Stage 3: apply deaths & rasterise survivors ----------------
        cand_pos = torch.stack((cand_hx, cand_hy), dim=-1)
        self.head_pos = torch.where(survives[:, :, None], cand_pos, self.head_pos)
        self.theta = torch.where(survives, new_theta, self.theta)
        self.dist_accum_fp = torch.where(
            survives, new_accum, torch.where(died, torch.zeros_like(new_accum), self.dist_accum_fp)
        )
        self.boosting = survives & boosting

        surv_pt_valid = survives[:, :, None] & (self._pts_idx < num_pts[:, :, None])
        for k in range(self.max_pts):
            vk = surv_pt_valid[:, :, k]
            w_ptr = (old_ptr + k + 1).remainder(max_len).to(torch.int64)
            w_ptr_2 = w_ptr[:, :, None, None].expand(-1, -1, 1, 2)

            cur_rp = self.ring_pos.gather(2, w_ptr_2).squeeze(2)
            new_rp = torch.stack((pt_x[:, :, k], pt_y[:, :, k]), dim=-1)
            self.ring_pos.scatter_(
                2, w_ptr_2, torch.where(vk[:, :, None], new_rp, cur_rp)[:, :, None, :]
            )

            cur_rc = self.ring_cell.gather(2, w_ptr_2).squeeze(2)
            new_rc = torch.stack(
                (pt_gx[:, :, k].to(torch.int16), pt_gy[:, :, k].to(torch.int16)), dim=-1
            )
            self.ring_cell.scatter_(
                2, w_ptr_2, torch.where(vk[:, :, None], new_rc, cur_rc)[:, :, None, :]
            )

            w_ptr_1 = w_ptr[:, :, None]
            cur_rr = self.ring_r_sq.gather(2, w_ptr_1).squeeze(2)
            self.ring_r_sq.scatter_(
                2, w_ptr_1, torch.where(vk, r_sq.to(torch.int16), cur_rr)[:, :, None]
            )
            cur_rs = self.ring_subtick.gather(2, w_ptr_1).squeeze(2)
            self.ring_subtick.scatter_(
                2, w_ptr_1, torch.where(vk, pt_subtick[:, :, k], cur_rs)[:, :, None]
            )

        new_ptr = (old_ptr + num_pts).remainder(max_len)
        self.head_ptr = torch.where(survives, new_ptr, old_ptr)

        new_seg = (keep_old + num_pts).clamp_max(max_len)
        self.seg_count = torch.where(
            survives, new_seg, torch.where(died, torch.zeros_like(old_seg), old_seg)
        )

        surv_tail_idx = (
            (self.head_ptr - (self.seg_count - 1).clamp_min(0)).remainder(max_len).to(torch.int64)
        )
        surv_oldest_st = self.ring_subtick.gather(2, surv_tail_idx[:, :, None]).squeeze(2)
        self.oldest_active_subtick = torch.where(
            survives,
            surv_oldest_st,
            torch.where(
                died,
                torch.full_like(self.oldest_active_subtick, expired_sentinel_st),
                self.oldest_active_subtick,
            ),
        )

        self.alive = survives
        self.life_age = torch.where(
            survives,
            self.life_age + 1,
            torch.where(died, torch.zeros_like(self.life_age), self.life_age),
        )
        self.respawn_timer = torch.where(
            died,
            torch.full_like(self.respawn_timer, int(self.cfg.respawn_delay)),
            self.respawn_timer,
        )
        self.boost_ticks = torch.where(died, torch.zeros_like(self.boost_ticks), self.boost_ticks)
        self.boost_drop_fp_accum = torch.where(
            died, torch.zeros_like(self.boost_drop_fp_accum), self.boost_drop_fp_accum
        )

        self._stamp_discs(
            pt_gx,
            pt_gy,
            r_sq[:, :, None].expand_as(pt_gx),
            surv_pt_valid,
            pt_subtick,
        )

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["rasterise"] += t1 - t0
            t0 = t1

        # --------- Stage 4: food (pickup -> boost drops -> corpses -> refill) ---------
        # 4a. Surviving heads eat existing pellets within pickup_r_sq_fp; lowest slot wins ties.
        hgx = (self.head_pos[:, :, 0] >> FP_SHIFT) + self.half_grid
        hgy = (self.head_pos[:, :, 1] >> FP_SHIFT) + self.half_grid
        ex_k = hgx[:, :, None] + self._eat_dx[None, None, :]
        ey_k = hgy[:, :, None] + self._eat_dy[None, None, :]
        eat_in_b = (ex_k >= 0) & (ex_k < self.G) & (ey_k >= 0) & (ey_k < self.G)

        cx_fp = ((ex_k - self.half_grid) << FP_SHIFT) + FP_HALF
        cy_fp = ((ey_k - self.half_grid) << FP_SHIFT) + FP_HALF
        dx_eat = (cx_fp - self.head_pos[:, :, 0, None]).clamp(-32768, 32767)
        dy_eat = (cy_fp - self.head_pos[:, :, 1, None]).clamp(-32768, 32767)
        pickup_r_sq = self._pickup_r_sq_fp_table[len_idx]
        can_eat_k = (
            survives[:, :, None]
            & eat_in_b
            & ((dx_eat * dx_eat + dy_eat * dy_eat) <= pickup_r_sq[:, :, None])
        )

        ex_c = ex_k.clamp(0, self.G - 1).to(torch.int64)
        ey_c = ey_k.clamp(0, self.G - 1).to(torch.int64)
        flat_eat = (ey_c * self.G + ex_c).view(self.E, -1)

        claim_src = torch.where(
            can_eat_k,
            self._snake_ids[:, :, None].expand_as(ex_k),
            torch.full_like(ex_k, self.N),
        ).view(self.E, -1)
        self._claim_grid.scatter_reduce_(1, flat_eat, claim_src, reduce="amin", include_self=True)
        winner_k = self._claim_grid.gather(1, flat_eat).view_as(ex_k)
        won_eat_k = can_eat_k & (winner_k == self._snake_ids[:, :, None])
        # Reset touched claim cells back to N in O(E * N * K_eat).
        self._claim_grid.scatter_(1, flat_eat, torch.full_like(claim_src, self.N))

        won_flat = won_eat_k.view(self.E, -1)
        amb_flat = self.ambient_grid.view(self.E, -1)
        cor_flat = self.corpse_grid.view(self.E, -1)
        eaten_amb = torch.where(won_flat, amb_flat.gather(1, flat_eat), 0)
        eaten_cor = torch.where(won_flat, cor_flat.gather(1, flat_eat), 0)

        neg_amb_1d = (-eaten_amb).reshape(-1)
        self.ambient_grid.view(-1).index_put_(
            ((self._env_base_1x + flat_eat).reshape(-1),), neg_amb_1d, accumulate=True
        )
        flat4_eat = ((ey_c >> 2) * self.G4 + (ex_c >> 2)).view(self.E, -1)
        self.ambient_grid_4x.view(-1).index_put_(
            ((self._env_base_4x + flat4_eat).reshape(-1),), neg_amb_1d, accumulate=True
        )
        flat16_eat = ((ey_c >> 4) * self.G16 + (ex_c >> 4)).view(self.E, -1)
        self.ambient_grid_16x.view(-1).index_put_(
            ((self._env_base_16x + flat16_eat).reshape(-1),), neg_amb_1d, accumulate=True
        )

        eaten_amb_count = (eaten_amb > 0).to(torch.int32).sum(dim=1)
        self.ambient_count = self.ambient_count - eaten_amb_count

        mass_eaten = (eaten_amb.view_as(ex_k) + eaten_cor.view_as(ex_k)).sum(dim=-1)
        uncapped_len = self.length + mass_eaten
        capped_len = uncapped_len.clamp_max(max_len)
        cap_overflow = (uncapped_len - capped_len).clamp_min(0)
        self.total_length_cap_sink = self.total_length_cap_sink + cap_overflow.to(torch.int64).sum(
            dim=1
        )
        self.length = torch.where(survives, capped_len, self.length)

        # 4b. Shed fractional boost trail pellets at pre-step tail cells.
        tail_gx_c = old_tail_gx.clamp(0, self.G - 1)
        tail_gy_c = old_tail_gy.clamp(0, self.G - 1)
        tail_flat = tail_gy_c * self.G + tail_gx_c
        tail_flat4 = (tail_gy_c >> 2) * self.G4 + (tail_gx_c >> 2)
        tail_flat16 = (tail_gy_c >> 4) * self.G16 + (tail_gx_c >> 4)
        shed_val = boost_drop_int.to(torch.int32)

        # 4c. Drop corpses AFTER deaths and eating.
        dead_len = torch.where(died, self.length, 0)
        corpse_mass = self._corpse_mass_table[dead_len.clamp(0, max_len).to(torch.int64)]
        corpse_sink = dead_len - corpse_mass
        self.total_corpse_decay_sink = self.total_corpse_decay_sink + corpse_sink.to(
            torch.int64
        ).sum(dim=1)
        self.length = torch.where(died, 0, self.length)

        p_cnt = torch.minimum(
            corpse_mass,
            torch.minimum(
                old_seg.clamp_min(1), torch.full_like(corpse_mass, int(self.cfg.corpse_max_pellets))
            ),
        )
        p_safe = p_cnt.clamp_min(1)
        k_idx = self._corpse_k
        active_orb = died[:, :, None] & (corpse_mass[:, :, None] > 0) & (k_idx < p_cnt[:, :, None])
        denom = (p_safe - 1).clamp_min(1)[:, :, None]
        seg_span = (old_seg - 1).clamp_min(0)[:, :, None]
        orb_offset = torch.div(k_idx * seg_span, denom, rounding_mode="floor")
        orb_ring_idx = (old_ptr[:, :, None] - orb_offset).remainder(max_len).to(torch.int64)
        orb_cells = self.ring_cell.gather(2, orb_ring_idx[..., None].expand(-1, -1, -1, 2))
        orb_gx = orb_cells[..., 0].to(torch.int64).clamp(0, self.G - 1)
        orb_gy = orb_cells[..., 1].to(torch.int64).clamp(0, self.G - 1)

        base_orb = torch.div(corpse_mass, p_safe, rounding_mode="floor")[:, :, None].to(torch.int32)
        rem_orb = corpse_mass.remainder(p_safe)[:, :, None]
        orb_val = torch.where(
            active_orb,
            base_orb + (k_idx < rem_orb).to(torch.int32),
            0,
        ).reshape(self.E, -1)
        orb_flat = (orb_gy * self.G + orb_gx).reshape(self.E, -1)
        orb_flat4 = ((orb_gy >> 2) * self.G4 + (orb_gx >> 2)).reshape(self.E, -1)
        orb_flat16 = ((orb_gy >> 4) * self.G16 + (orb_gx >> 4)).reshape(self.E, -1)

        cor_delta_1d = torch.cat((-eaten_cor, shed_val, orb_val), dim=1).reshape(-1)
        cor_idx_1x = (
            self._env_base_1x + torch.cat((flat_eat, tail_flat, orb_flat), dim=1)
        ).reshape(-1)
        cor_idx_4x = (
            self._env_base_4x + torch.cat((flat4_eat, tail_flat4, orb_flat4), dim=1)
        ).reshape(-1)
        cor_idx_16x = (
            self._env_base_16x + torch.cat((flat16_eat, tail_flat16, orb_flat16), dim=1)
        ).reshape(-1)
        self.corpse_grid.view(-1).index_put_((cor_idx_1x,), cor_delta_1d, accumulate=True)
        self.corpse_grid_4x.view(-1).index_put_((cor_idx_4x,), cor_delta_1d, accumulate=True)
        self.corpse_grid_16x.view(-1).index_put_((cor_idx_16x,), cor_delta_1d, accumulate=True)

        # 4d. Refill ambient pellets toward target density.
        self._spawn_ambient_batch(
            PURPOSE_AMBIENT_FOOD,
            self.tick,
            0,
            self.cfg.ambient_spawn_budget_per_tick,
        )

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["food"] += t1 - t0
            t0 = t1

        # ---------------- Stage 5: respawn & episode hooks ----------------
        was_dead_before = (~self.alive) & (~died)
        self.respawn_timer = torch.where(
            was_dead_before,
            (self.respawn_timer - 1).clamp_min(0),
            self.respawn_timer,
        )
        ready = was_dead_before & (self.respawn_timer == 0)

        seeds_3d = self.seeds[:, None, None]
        rgx = rng_int_torch(seeds_3d, self.tick, PURPOSE_RESPAWN, self._resp_slot_x, self.G)
        rgy = rng_int_torch(seeds_3d, self.tick, PURPOSE_RESPAWN, self._resp_slot_y, self.G)
        rth = rng_angle_idx_torch(
            seeds_3d, self.tick, PURPOSE_RESPAWN, self._resp_slot_t, self.cfg.angle_steps
        )

        # Append deterministic fallback ring candidates after the K counter-RNG candidates.
        fb_ring_idx = (
            self._snake_ids_i64[:, :, None] * 7 + int(self.tick) * 13 + self._fb_f_offsets
        ) & (FALLBACK_RING_SIZE - 1)
        fb_cands = self._fallback_spawns[fb_ring_idx].expand(
            self.E, self.N, self.cfg.respawn_fallback_candidates, 3
        )
        all_gx = torch.cat((rgx, fb_cands[..., 0]), dim=2)
        all_gy = torch.cat((rgy, fb_cands[..., 1]), dim=2)
        all_th = torch.cat((rth, fb_cands[..., 2]), dim=2)

        r_in_wall = (
            ready[:, :, None]
            & (all_gx >= 0)
            & (all_gx < self.G)
            & (all_gy >= 0)
            & (all_gy < self.G)
            & (self._cell_center_sq_fp(all_gx, all_gy) < self._spawn_wall_sq_fp)
        )

        cx_all = all_gx[..., None] + self._clear_dx[None, None, None, :]
        cy_all = all_gy[..., None] + self._clear_dy[None, None, None, :]
        c_in_grid = (cx_all >= 0) & (cx_all < self.G) & (cy_all >= 0) & (cy_all < self.G)
        c_flat = cy_all.clamp(0, self.G - 1).to(torch.int64) * self.G + cx_all.clamp(
            0, self.G - 1
        ).to(torch.int64)
        c_busy = (~c_in_grid) | self._is_cell_occupied(c_flat, self.oldest_active_subtick)

        cand_ok = r_in_wall & (~c_busy.any(dim=-1))
        has_cand = cand_ok.any(dim=-1)
        best_m = cand_ok.to(torch.int32).argmax(dim=-1, keepdim=True)
        sel_gx = all_gx.gather(2, best_m).squeeze(2)
        sel_gy = all_gy.gather(2, best_m).squeeze(2)
        sel_th = all_th.gather(2, best_m).squeeze(2)

        # Enforce mutual clearance with all earlier slots i < j having a valid candidate.
        rdx = sel_gx[:, :, None] - sel_gx[:, None, :]
        rdy = sel_gy[:, :, None] - sel_gy[:, None, :]
        min_sep_sq = (2 * self._r_clear) * (2 * self._r_clear)
        earlier_conflict = (
            self._tril_nn & has_cand[:, None, :] & ((rdx * rdx + rdy * rdy) <= min_sep_sq)
        ).any(dim=-1)
        spawn_now = has_cand & (~earlier_conflict)

        spawn_pos = self._cell_center_fp(sel_gx, sel_gy)
        spawn_st = int(tick_base_subtick + self.max_pts - 1)
        self.head_pos = torch.where(spawn_now[:, :, None], spawn_pos, self.head_pos)
        self.theta = torch.where(spawn_now, sel_th, self.theta)
        self.alive = self.alive | spawn_now
        self.boosting = torch.where(spawn_now, torch.zeros_like(self.boosting), self.boosting)
        self.length = torch.where(
            spawn_now,
            torch.full_like(self.length, int(self.cfg.start_length)),
            self.length,
        )
        self.seg_count = torch.where(spawn_now, torch.ones_like(self.seg_count), self.seg_count)
        self.head_ptr = torch.where(spawn_now, torch.zeros_like(self.head_ptr), self.head_ptr)
        self.oldest_active_subtick = torch.where(
            spawn_now,
            torch.full_like(self.oldest_active_subtick, spawn_st),
            self.oldest_active_subtick,
        )
        self.dist_accum_fp = torch.where(
            spawn_now, torch.zeros_like(self.dist_accum_fp), self.dist_accum_fp
        )
        self.boost_ticks = torch.where(
            spawn_now, torch.zeros_like(self.boost_ticks), self.boost_ticks
        )
        self.boost_drop_fp_accum = torch.where(
            spawn_now, torch.zeros_like(self.boost_drop_fp_accum), self.boost_drop_fp_accum
        )
        self.life_age = torch.where(spawn_now, torch.zeros_like(self.life_age), self.life_age)

        spawn_r_sq = self._r_sq_table[int(self.cfg.start_length)].expand(self.E, self.N)
        spawn_cell_i16 = torch.stack((sel_gx.to(torch.int16), sel_gy.to(torch.int16)), dim=-1)
        self.ring_pos[:, :, 0, :] = torch.where(
            spawn_now[:, :, None], spawn_pos, self.ring_pos[:, :, 0, :]
        )
        self.ring_cell[:, :, 0, :] = torch.where(
            spawn_now[:, :, None], spawn_cell_i16, self.ring_cell[:, :, 0, :]
        )
        self.ring_r_sq[:, :, 0] = torch.where(
            spawn_now, spawn_r_sq.to(torch.int16), self.ring_r_sq[:, :, 0]
        )
        self.ring_subtick[:, :, 0] = torch.where(
            spawn_now,
            torch.full_like(self.oldest_active_subtick, spawn_st),
            self.ring_subtick[:, :, 0],
        )

        self._stamp_discs(
            sel_gx[:, :, None],
            sel_gy[:, :, None],
            spawn_r_sq[:, :, None],
            spawn_now[:, :, None],
            torch.full((self.E, self.N, 1), spawn_st, dtype=torch.int32, device=self.device),
        )
        self.total_spawned_snake_mass = self.total_spawned_snake_mass + spawn_now.to(
            torch.int64
        ).sum(dim=1) * int(self.cfg.start_length)

        # Episode lifecycle signals (Rule 12).
        self.life_start.copy_(spawn_now)
        if self.cfg.max_life_ticks > 0:
            truncated = survives & (self.life_age >= int(self.cfg.max_life_ticks))
        else:
            truncated = torch.zeros_like(survives)
        self.truncated_pending.copy_(truncated)
        in_respawn_delay = (~self.alive) & (~died)

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["respawn"] += t1 - t0

        return StepInfo(
            alive=self.alive.clone(),
            life_start=self.life_start.clone(),
            died=died,
            truncated=truncated,
            in_respawn_delay=in_respawn_delay,
            length=self.length.clone(),
            mass_eaten=mass_eaten,
            killer_id=killer_id,
            kill_credit=kill_credit,
            life_age=self.life_age.clone(),
        )

    def active_occupancy_grid(self) -> torch.Tensor:
        """Materialize lowest active owner ID ``(E, G, G)`` in int8 (``-1`` where empty)."""
        ow_lo = 63 - (self.occ_lo & 63)
        ow_lo_i64 = ow_lo.clamp(0, self.N - 1).to(torch.int64)
        st_lo = self.oldest_active_subtick.gather(1, ow_lo_i64.view(self.E, -1)).view_as(
            self.occ_lo
        )
        act_lo = (self.occ_lo >= 0) & ((self.occ_lo >> 6) >= st_lo)

        ow_hi = self.occ_hi & 63
        ow_hi_i64 = ow_hi.clamp(0, self.N - 1).to(torch.int64)
        st_hi = self.oldest_active_subtick.gather(1, ow_hi_i64.view(self.E, -1)).view_as(
            self.occ_hi
        )
        act_hi = (self.occ_hi >= 0) & ((self.occ_hi >> 6) >= st_hi)

        full_n = torch.full_like(ow_lo, self.N)
        lowest = torch.minimum(
            torch.where(act_lo, ow_lo, full_n),
            torch.where(act_hi, ow_hi, full_n),
        )
        return torch.where(
            lowest < self.N, lowest.to(torch.int8), torch.full_like(lowest, -1, dtype=torch.int8)
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize full deterministic simulator state for checkpoint/resume."""
        tensor_names = (
            "seeds",
            "head_pos",
            "theta",
            "alive",
            "boosting",
            "length",
            "seg_count",
            "head_ptr",
            "oldest_active_subtick",
            "dist_accum_fp",
            "boost_ticks",
            "boost_drop_fp_accum",
            "respawn_timer",
            "life_age",
            "life_start",
            "truncated_pending",
            "ring_pos",
            "ring_cell",
            "ring_r_sq",
            "ring_subtick",
            "prev_alive",
            "prev_boosting",
            "prev_head_pos",
            "prev_theta",
            "prev_length",
            "occ_hi",
            "occ_lo",
            "occ_subtick_4x",
            "occ_subtick_16x",
            "ambient_grid",
            "ambient_grid_4x",
            "ambient_grid_16x",
            "corpse_grid",
            "corpse_grid_4x",
            "corpse_grid_16x",
            "ambient_count",
            "total_spawned_snake_mass",
            "total_spawned_ambient_mass",
            "total_corpse_decay_sink",
            "total_boost_decay_sink",
            "total_length_cap_sink",
            "total_truncation_sink",
        )
        return {
            "config": self.cfg.to_dict(),
            "num_envs": self.E,
            "tick": self.tick,
            "tensors": {name: getattr(self, name).detach().cpu().clone() for name in tensor_names},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore deterministic simulator state in-place onto ``self.device``."""
        loaded_cfg = GameV3Config.from_dict(state["config"])
        if loaded_cfg != self.cfg:
            raise ValueError("Checkpoint GameV3Config does not match simulator config")
        if int(state["num_envs"]) != self.E:
            raise ValueError(f"Checkpoint num_envs {state['num_envs']} != {self.E}")
        self.tick = int(state["tick"])
        tensors = state["tensors"]
        for name, tensor in tensors.items():
            dst = getattr(self, name)
            dst.copy_(tensor.to(device=self.device, dtype=dst.dtype))
