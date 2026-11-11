"""GPU-first batched PyTorch simulator for Game V3 (E worlds x N snakes)."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
    TWO_PI_F32,
    GameV3Config,
)
from research.game_v3.rng import (
    PURPOSE_AMBIENT_FOOD,
    PURPOSE_INIT_FOOD,
    PURPOSE_INIT_SNAKE,
    PURPOSE_RESPAWN,
    rng_angle_f32_torch,
    rng_int_torch,
)

_INIT_SNAKE_CANDIDATES: int = 32
_INIT_FOOD_BATCH_SIZE: int = 128


@dataclass
class StepInfo:
    """Per-snake reward and episode signals emitted each simulator tick."""

    alive: torch.Tensor
    length: torch.Tensor
    died: torch.Tensor
    killer_id: torch.Tensor
    kill_credit: torch.Tensor
    mass_eaten: torch.Tensor


def poly_sincos_torch(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute float32 (sin, cos) on [-pi, pi] via fixed-order minimax polynomials.

    Uses only IEEE-754 float32 additions, subtractions, and multiplications with no
    transcendental library calls so CPU Torch and CPU NumPy match bit-for-bit.
    """
    ab = theta.abs()
    over = ab > HALF_PI_F32
    xs = torch.where(over, torch.where(theta >= 0.0, PI_F32 - ab, ab - PI_F32), theta)
    xc = torch.where(over, PI_F32 - ab, ab)

    x2s = xs * xs
    p = x2s * SIN_C9
    p = p + SIN_C7
    p = p * x2s
    p = p + SIN_C5
    p = p * x2s
    p = p + SIN_C3
    sin_val = xs + xs * (x2s * p)

    x2c = xc * xc
    q = x2c * COS_C10
    q = q + COS_C8
    q = q * x2c
    q = q + COS_C6
    q = q * x2c
    q = q + COS_C4
    q = q * x2c
    q = q + COS_C2
    cos_val = 1.0 + x2c * q
    cos_val = torch.where(over, -cos_val, cos_val)
    return sin_val, cos_val


def _build_static_tables(cfg: GameV3Config, device: torch.device) -> dict[str, torch.Tensor | int]:
    """Precompute length-indexed geometry, collision, and stencil tables on CPU."""
    max_len = cfg.max_length
    c = cfg.cell_size

    w_np = np.empty(max_len + 1, dtype=np.float32)
    turn_np = np.empty(max_len + 1, dtype=np.float32)
    r_sq_np = np.empty(max_len + 1, dtype=np.int32)
    wall_d4_np = np.empty(max_len + 1, dtype=np.int64)
    corpse_mass_np = np.empty(max_len + 1, dtype=np.int32)

    for length in range(max_len + 1):
        w64 = cfg.body_radius_f64(length)
        w_np[length] = np.float32(w64)
        turn_np[length] = np.float32(cfg.turn_rate_f64(max(1, length)))
        r_sq_np[length] = int(math.floor((w64 / c) ** 2 + 1e-12))
        wall_r = max(0.0, cfg.arena_radius - w64)
        wall_d4_np[length] = int(math.ceil(4.0 * ((wall_r / c) ** 2) - 1e-12))
        corpse_mass_np[length] = int(math.floor(length * cfg.corpse_mass_fraction + 1e-12))

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

    # Map each length's r_sq_cells to a compact tier index for O(1) head-disc overlap lookup.
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

    spawn_limit_r = max(
        c, cfg.arena_radius - cfg.body_radius_f64(cfg.start_length) - cfg.spawn_wall_margin
    )
    spawn_wall_d4 = int(math.floor(4.0 * ((spawn_limit_r / c) ** 2)))
    food_limit_r = max(c, cfg.arena_radius - cfg.food_wall_margin)
    food_wall_d4 = int(math.floor(4.0 * ((food_limit_r / c) ** 2)))

    return {
        "w_table": torch.as_tensor(w_np, device=device),
        "turn_table": torch.as_tensor(turn_np, device=device),
        "r_sq_table": torch.as_tensor(r_sq_np, device=device),
        "r_tier_table": torch.as_tensor(r_tier_np, device=device),
        "wall_d4_table": torch.as_tensor(wall_d4_np, device=device),
        "corpse_mass_table": torch.as_tensor(corpse_mass_np, device=device),
        "disc_dx": torch.as_tensor(disc_dx_np, device=device),
        "disc_dy": torch.as_tensor(disc_dy_np, device=device),
        "disc_dist_sq": torch.as_tensor(disc_dist_sq_np, device=device),
        "overlap_lut": torch.as_tensor(overlap_lut_np, device=device),
        "clear_dx": torch.as_tensor(clear_dx_np, device=device),
        "clear_dy": torch.as_tensor(clear_dy_np, device=device),
        "max_diam": max_diam,
        "r_clear": r_clear,
        "spawn_wall_d4": spawn_wall_d4,
        "food_wall_d4": food_wall_d4,
    }


class GameV3Sim:
    """Batched GPU-first Game V3 simulator with incremental grid maintenance.

    All tensors have fixed shapes across ``(E, N, ...)``; dead lanes are masked in
    place without compaction, and ``step`` performs zero host synchronizations.
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

        tables = _build_static_tables(self.cfg, self.device)
        self._w_table: torch.Tensor = tables["w_table"]  # type: ignore[assignment]
        self._turn_table: torch.Tensor = tables["turn_table"]  # type: ignore[assignment]
        self._r_sq_table: torch.Tensor = tables["r_sq_table"]  # type: ignore[assignment]
        self._r_tier_table: torch.Tensor = tables["r_tier_table"]  # type: ignore[assignment]
        self._wall_d4_table: torch.Tensor = tables["wall_d4_table"]  # type: ignore[assignment]
        self._corpse_mass_table = tables["corpse_mass_table"]  # type: ignore[assignment]
        self._disc_dx: torch.Tensor = tables["disc_dx"]  # type: ignore[assignment]
        self._disc_dy: torch.Tensor = tables["disc_dy"]  # type: ignore[assignment]
        self._disc_dist_sq: torch.Tensor = tables["disc_dist_sq"]  # type: ignore[assignment]
        self._overlap_lut: torch.Tensor = tables["overlap_lut"]  # type: ignore[assignment]
        self._clear_dx: torch.Tensor = tables["clear_dx"]  # type: ignore[assignment]
        self._clear_dy: torch.Tensor = tables["clear_dy"]  # type: ignore[assignment]
        self._max_diam: int = int(tables["max_diam"])
        self._r_clear: int = int(tables["r_clear"])
        self._spawn_wall_d4: int = int(tables["spawn_wall_d4"])
        self._food_wall_d4: int = int(tables["food_wall_d4"])

        # Preallocated constant index helpers.
        self._snake_ids = torch.arange(self.N, dtype=torch.int32, device=self.device)[None, :]
        self._snake_ids_i64 = self._snake_ids.to(torch.int64)
        self._not_eye_nn = ~torch.eye(self.N, dtype=torch.bool, device=self.device)[None, :, :]
        self._tril_nn = torch.tril(
            torch.ones((self.N, self.N), dtype=torch.bool, device=self.device), diagonal=-1
        )[None, :, :]
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

        # Mutable simulation state tensors.
        es = (self.E, self.N)
        max_len = self.cfg.max_length
        self.seeds = torch.zeros((self.E,), dtype=torch.int64, device=self.device)
        self.tick: int = 0

        self.head_pos = torch.zeros((self.E, self.N, 2), dtype=torch.float32, device=self.device)
        self.theta = torch.zeros(es, dtype=torch.float32, device=self.device)
        self.alive = torch.zeros(es, dtype=torch.bool, device=self.device)
        self.length = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.seg_count = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.head_ptr = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.tail_seq = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.boost_ticks = torch.zeros(es, dtype=torch.int32, device=self.device)
        self.respawn_timer = torch.zeros(es, dtype=torch.int32, device=self.device)

        self.ring_pos = torch.zeros(
            (self.E, self.N, max_len, 2), dtype=torch.float32, device=self.device
        )
        self.ring_cell = torch.zeros(
            (self.E, self.N, max_len, 2), dtype=torch.int16, device=self.device
        )

        # Previous-tick head state for 2-frame head observation stacking.
        self.prev_alive = torch.zeros(es, dtype=torch.bool, device=self.device)
        self.prev_head_pos = torch.zeros(
            (self.E, self.N, 2), dtype=torch.float32, device=self.device
        )
        self.prev_theta = torch.zeros(es, dtype=torch.float32, device=self.device)
        self.prev_length = torch.zeros(es, dtype=torch.int32, device=self.device)

        # Incremental occupancy grid: packed int32 = (seq << 6) | owner_id, -1 if never stamped.
        self.occ_packed = torch.full(
            (self.E, self.G, self.G), -1, dtype=torch.int32, device=self.device
        )
        # Incremental food mass grids (1x, 4x, 16x).
        self.ambient_grid = torch.zeros(
            (self.E, self.G, self.G), dtype=torch.int16, device=self.device
        )
        self.ambient_grid_4x = torch.zeros(
            (self.E, self.G4, self.G4), dtype=torch.int16, device=self.device
        )
        self.ambient_grid_16x = torch.zeros(
            (self.E, self.G16, self.G16), dtype=torch.int16, device=self.device
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

        # Exact mass-accounting ledgers per world (sources and sinks).
        self.total_spawned_snake_mass = torch.zeros(
            (self.E,), dtype=torch.int64, device=self.device
        )
        self.total_spawned_ambient_mass = torch.zeros(
            (self.E,), dtype=torch.int64, device=self.device
        )
        self.total_corpse_Criteria_sink = torch.zeros(
            (self.E,), dtype=torch.int64, device=self.device
        )
        self.total_length_cap_sink = torch.zeros((self.E,), dtype=torch.int64, device=self.device)

        # Optional stage timing accumulator (disabled by default; zero host sync when False).
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
    def total_corpse_decay_sink(self) -> torch.Tensor:
        """Cumulative mass lost to (1 - corpse_mass_fraction) on snake deaths."""
        return self.total_corpse_Criteria_sink

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

    def _cell_d4_int(self, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        dx2 = 2 * (gx.to(torch.int64) - self.half_grid) + 1
        dy2 = 2 * (gy.to(torch.int64) - self.half_grid) + 1
        return dx2 * dx2 + dy2 * dy2

    def _cell_center_f32(self, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        c_f32 = float(np.float32(self.cfg.cell_size))
        hg_f32 = float(np.float32(self.half_grid))
        cx = (gx.to(torch.float32) - hg_f32 + 0.5) * c_f32
        cy = (gy.to(torch.float32) - hg_f32 + 0.5) * c_f32
        return torch.stack((cx, cy), dim=-1)

    def _stamp_discs(
        self,
        gx: torch.Tensor,
        gy: torch.Tensor,
        r_sq: torch.Tensor,
        active: torch.Tensor,
        seq: int,
    ) -> None:
        """Stamp head discs of ``active`` snakes into ``occ_packed`` with ``(seq << 6) | id``."""
        sx = gx[:, :, None] + self._disc_dx[None, None, :]
        sy = gy[:, :, None] + self._disc_dy[None, None, :]
        in_b = (sx >= 0) & (sx < self.G) & (sy >= 0) & (sy < self.G)
        in_r = self._disc_dist_sq[None, None, :] <= r_sq[:, :, None]
        valid = active[:, :, None] & in_b & in_r

        flat_idx = (
            sy.clamp(0, self.G - 1).to(torch.int64) * self.G
            + sx.clamp(0, self.G - 1).to(torch.int64)
        ).reshape(self.E, -1)
        stamp_val = ((int(seq) << 6) | self._snake_ids)[:, :, None].expand_as(sx)
        src = torch.where(valid, stamp_val, torch.full_like(stamp_val, -1)).reshape(self.E, -1)

        occ_flat = self.occ_packed.view(self.E, -1)
        occ_flat.scatter_reduce_(1, flat_idx, src, reduce="amax", include_self=True)

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
        in_arena = in_grid & (self._cell_d4_int(gx, gy) < self._food_wall_d4)

        gx_c = gx.clamp(0, self.G - 1).to(torch.int64)
        gy_c = gy.clamp(0, self.G - 1).to(torch.int64)
        flat = gy_c * self.G + gx_c

        occ_val = self.occ_packed.view(self.E, -1).gather(1, flat)
        owner = (occ_val & 63).clamp(0, self.N - 1).to(torch.int64)
        owner_tail = self.tail_seq.gather(1, owner)
        occupied = (occ_val >= 0) & ((occ_val >> 6) >= owner_tail)

        amb_val = self.ambient_grid.view(self.E, -1).gather(1, flat)
        cor_val = self.corpse_grid.view(self.E, -1).gather(1, flat)
        food_free = (amb_val == 0) & (cor_val == 0)

        same = flat[:, :, None] == flat[:, None, :]
        unique = ~(same & tril).any(dim=2)

        cand_valid = in_arena & (~occupied) & food_free & unique
        deficit = (int(self.cfg.target_ambient_pellets) - self.ambient_count).clamp_min(0)
        rank = cand_valid.to(torch.int32).cumsum(dim=1) - 1
        accept = cand_valid & (rank < deficit[:, None])

        pellet_i16 = int(self.cfg.ambient_pellet_value)
        add_i16 = torch.where(
            accept,
            torch.full_like(amb_val, pellet_i16),
            torch.zeros_like(amb_val),
        )
        self.ambient_grid.view(self.E, -1).scatter_add_(1, flat, add_i16)

        flat4 = (gy_c >> 2) * self.G4 + (gx_c >> 2)
        self.ambient_grid_4x.view(self.E, -1).scatter_add_(1, flat4, add_i16)
        flat16 = (gy_c >> 4) * self.G16 + (gx_c >> 4)
        self.ambient_grid_16x.view(self.E, -1).scatter_add_(1, flat16, add_i16)

        added_count = accept.to(torch.int32).sum(dim=1)
        self.ambient_count = self.ambient_count + added_count
        self.total_spawned_ambient_mass = (
            self.total_spawned_ambient_mass + added_count.to(torch.int64) * pellet_i16
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

        self.occ_packed.fill_(-1)
        self.ambient_grid.zero_()
        self.ambient_grid_4x.zero_()
        self.ambient_grid_16x.zero_()
        self.corpse_grid.zero_()
        self.corpse_grid_4x.zero_()
        self.corpse_grid_16x.zero_()
        self.ambient_count.zero_()

        self.total_spawned_snake_mass.zero_()
        self.total_spawned_ambient_mass.zero_()
        self.total_corpse_Criteria_sink.zero_()
        self.total_length_cap_sink.zero_()

        self.ring_pos.zero_()
        self.ring_cell.zero_()
        self.boost_ticks.zero_()
        self.respawn_timer.zero_()
        self.head_ptr.zero_()

        # Initial snake placement (tick = 0, PURPOSE_INIT_SNAKE).
        k_init = _INIT_SNAKE_CANDIDATES
        slots = (
            torch.arange(self.N, dtype=torch.int64, device=self.device)[:, None] * k_init
            + torch.arange(k_init, dtype=torch.int64, device=self.device)[None, :]
        )[None, :, :]
        seeds_3d = self.seeds[:, None, None]
        cand_gx = rng_int_torch(seeds_3d, 0, PURPOSE_INIT_SNAKE, slots * 3, self.G)
        cand_gy = rng_int_torch(seeds_3d, 0, PURPOSE_INIT_SNAKE, slots * 3 + 1, self.G)
        cand_th = rng_angle_f32_torch(seeds_3d, 0, PURPOSE_INIT_SNAKE, slots * 3 + 2)
        in_wall = (
            (cand_gx >= 0)
            & (cand_gx < self.G)
            & (cand_gy >= 0)
            & (cand_gy < self.G)
            & (self._cell_d4_int(cand_gx, cand_gy) < self._spawn_wall_d4)
        )

        placed_gx = torch.zeros((self.E, self.N), dtype=torch.int32, device=self.device)
        placed_gy = torch.zeros((self.E, self.N), dtype=torch.int32, device=self.device)
        placed_th = torch.zeros((self.E, self.N), dtype=torch.float32, device=self.device)
        min_sep_sq = (2 * self._r_clear) * (2 * self._r_clear)

        # Deterministic fallback ring inside 0.5 * arena_radius if crowded arena exhausts trials.
        fb_r_cells = max(1.0, 0.5 * self.cfg.arena_radius / self.cfg.cell_size)
        for i in range(self.N):
            valid_i = in_wall[:, i, :]
            if i > 0:
                dx = cand_gx[:, i, :, None] - placed_gx[:, None, :i]
                dy = cand_gy[:, i, :, None] - placed_gy[:, None, :i]
                clear = ((dx * dx + dy * dy) > min_sep_sq).all(dim=-1)
                valid_i = valid_i & clear
            has_i = valid_i.any(dim=-1)
            best_m = valid_i.to(torch.int32).argmax(dim=-1, keepdim=True)
            gx_i = cand_gx[:, i, :].gather(1, best_m).squeeze(1)
            gy_i = cand_gy[:, i, :].gather(1, best_m).squeeze(1)
            th_i = cand_th[:, i, :].gather(1, best_m).squeeze(1)

            ang_fb = 2.0 * math.pi * i / self.N
            fb_gx = int(round(self.half_grid + fb_r_cells * math.cos(ang_fb)))
            fb_gy = int(round(self.half_grid + fb_r_cells * math.sin(ang_fb)))
            fb_th = float(np.float32(ang_fb - math.pi))

            placed_gx[:, i] = torch.where(has_i, gx_i, torch.full_like(gx_i, fb_gx))
            placed_gy[:, i] = torch.where(has_i, gy_i, torch.full_like(gy_i, fb_gy))
            placed_th[:, i] = torch.where(has_i, th_i, torch.full_like(th_i, fb_th))

        self.head_pos.copy_(self._cell_center_f32(placed_gx, placed_gy))
        self.theta.copy_(placed_th)
        self.alive.fill_(True)
        self.length.fill_(int(self.cfg.start_length))
        self.seg_count.fill_(1)
        self.tail_seq.fill_(1)
        self.ring_pos[:, :, 0, :] = self.head_pos
        self.ring_cell[:, :, 0, 0] = placed_gx.to(torch.int16)
        self.ring_cell[:, :, 0, 1] = placed_gy.to(torch.int16)

        init_r_sq = self._r_sq_table[self.length.to(torch.int64)]
        self._stamp_discs(placed_gx, placed_gy, init_r_sq, self.alive, seq=1)
        self.total_spawned_snake_mass.fill_(int(self.N * self.cfg.start_length))

        # No t = -1 frame exists immediately after reset.
        self.prev_alive.fill_(False)
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

        Args:
            actions: Integer action tensor of shape ``(E, N)`` in ``0..5``
                (``{left, straight, right} x {normal, boost}``).

        Returns:
            :class:`StepInfo` containing per-snake signals on ``self.device``.
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
        seq = self.tick + 1

        # Snapshot pre-step head pose for 2-frame head observation stacking.
        self.prev_alive.copy_(self.alive)
        self.prev_head_pos.copy_(self.head_pos)
        self.prev_theta.copy_(self.theta)
        self.prev_length.copy_(self.length)

        # ---------------- Stage 1: move ----------------
        act = actions.to(torch.int32).clamp(0, 5)
        steer = act.remainder(3)
        want_boost = act >= 3

        cur_len_i64 = self.length.clamp(0, self.cfg.max_length).to(torch.int64)
        turn_lim = self._turn_table[cur_len_i64]
        dtheta = torch.where(
            steer == 0,
            turn_lim,
            torch.where(steer == 2, -turn_lim, torch.zeros_like(turn_lim)),
        )
        raw_theta = self.theta + dtheta
        wrapped_theta = torch.where(
            raw_theta >= PI_F32,
            raw_theta - TWO_PI_F32,
            torch.where(raw_theta < -PI_F32, raw_theta + TWO_PI_F32, raw_theta),
        )
        new_theta = torch.where(self.alive, wrapped_theta, self.theta)

        boosting = self.alive & want_boost & (self.length > int(self.cfg.min_boost_length))
        step_dist = torch.where(
            boosting,
            torch.full_like(new_theta, float(np.float32(self.cfg.boost_speed))),
            torch.full_like(new_theta, float(np.float32(self.cfg.speed))),
        )
        sin_t, cos_t = poly_sincos_torch(new_theta)
        dx_step = step_dist * cos_t
        dy_step = step_dist * sin_t
        cand_x = self.head_pos[:, :, 0] + dx_step
        cand_y = self.head_pos[:, :, 1] + dy_step

        inv_c = float(np.float32(1.0 / self.cfg.cell_size))
        cand_gx = torch.floor(cand_x * inv_c).to(torch.int32) + self.half_grid
        cand_gy = torch.floor(cand_y * inv_c).to(torch.int32) + self.half_grid

        # Capture current tail cell before boost shedding / tail advance.
        old_seg = self.seg_count
        old_ptr = self.head_ptr
        tail_ring_idx = (
            (old_ptr - (old_seg - 1).clamp_min(0)).remainder(self.cfg.max_length).to(torch.int64)
        )
        old_tail_cell = self.ring_cell.gather(
            2, tail_ring_idx[:, :, None, None].expand(-1, -1, 1, 2)
        ).squeeze(2)
        old_tail_gx = old_tail_cell[:, :, 0].to(torch.int64)
        old_tail_gy = old_tail_cell[:, :, 1].to(torch.int64)

        inc_boost = torch.where(boosting, self.boost_ticks + 1, self.boost_ticks)
        shed = boosting & (inc_boost >= int(self.cfg.boost_shed_ticks))
        self.boost_ticks = torch.where(shed, torch.zeros_like(inc_boost), inc_boost)
        len_post_shed = torch.where(shed, self.length - int(self.cfg.boost_shed_mass), self.length)
        self.length = torch.where(self.alive, len_post_shed, self.length)

        keep_old = torch.minimum(old_seg, (self.length - 1).clamp_min(0))
        active_body_tail_seq = torch.where(
            self.alive,
            torch.full_like(keep_old, seq) - keep_old,
            torch.full_like(keep_old, seq + 1),
        )

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["move"] += t1 - t0
            t0 = t1

        # ---------------- Stage 2: collide ----------------
        len_idx = self.length.clamp(0, self.cfg.max_length).to(torch.int64)
        r_sq = self._r_sq_table[len_idx]
        r_tier = self._r_tier_table[len_idx]
        wall_d4 = self._wall_d4_table[len_idx]

        in_grid = (cand_gx >= 0) & (cand_gx < self.G) & (cand_gy >= 0) & (cand_gy < self.G)
        d4 = self._cell_d4_int(cand_gx, cand_gy)
        died_wall = self.alive & ((~in_grid) | (d4 >= wall_d4))

        # Pairwise head-disc overlap via translation-invariant integer offset LUT.
        dgx = cand_gx[:, None, :] - cand_gx[:, :, None]
        dgy = cand_gy[:, None, :] - cand_gy[:, :, None]
        near_pair = (
            self.alive[:, :, None]
            & self.alive[:, None, :]
            & self._not_eye_nn
            & (dgx.abs() <= self._max_diam)
            & (dgy.abs() <= self._max_diam)
        )
        dgx_idx = (dgx + self._max_diam).clamp(0, 2 * self._max_diam).to(torch.int64)
        dgy_idx = (dgy + self._max_diam).clamp(0, 2 * self._max_diam).to(torch.int64)
        ti = r_tier[:, :, None].expand(-1, -1, self.N)
        tj = r_tier[:, None, :].expand(-1, self.N, -1)
        head_pair = near_pair & self._overlap_lut[ti, tj, dgx_idx, dgy_idx]

        li = self.length[:, :, None].to(torch.int64)
        lj = self.length[:, None, :].to(torch.int64)
        ratio_bp = int(self.cfg.head_on_ratio_bp)
        j_wins = (lj * 10000) > (li * ratio_bp)
        i_wins = (li * 10000) > (lj * ratio_bp)
        tie = (~j_wins) & (~i_wins)

        died_head = self.alive & (head_pair & (j_wins | tie)).any(dim=-1)
        strict_head_killers = head_pair & j_wins
        has_head_killer = strict_head_killers.any(dim=-1)
        head_killer_raw = torch.where(
            strict_head_killers,
            self._snake_ids[:, None, :].expand(self.E, self.N, self.N),
            torch.full((self.E, self.N, self.N), self.N, dtype=torch.int32, device=self.device),
        ).amin(dim=-1)
        head_killer = torch.where(
            has_head_killer, head_killer_raw, torch.full_like(head_killer_raw, -1)
        )

        # Enemy body collision by gathering occ_packed under each snake's candidate head disc.
        hx_k = cand_gx[:, :, None] + self._disc_dx[None, None, :]
        hy_k = cand_gy[:, :, None] + self._disc_dy[None, None, :]
        in_b_k = (hx_k >= 0) & (hx_k < self.G) & (hy_k >= 0) & (hy_k < self.G)
        in_r_k = self._disc_dist_sq[None, None, :] <= r_sq[:, :, None]
        valid_k = self.alive[:, :, None] & in_b_k & in_r_k

        flat_k = hy_k.clamp(0, self.G - 1).to(torch.int64) * self.G + hx_k.clamp(0, self.G - 1).to(
            torch.int64
        )
        occ_k = self.occ_packed.view(self.E, -1).gather(1, flat_k.view(self.E, -1)).view_as(hx_k)
        owner_k = occ_k & 63
        owner_k_i64 = owner_k.clamp(0, self.N - 1).to(torch.int64)
        tail_k = (
            active_body_tail_seq[:, None, :].expand(self.E, self.N, self.N).gather(2, owner_k_i64)
        )
        is_head_pair_k = head_pair.gather(2, owner_k_i64)

        hit_body_k = (
            valid_k
            & (occ_k >= 0)
            & (owner_k != self._snake_ids[:, :, None])
            & ((occ_k >> 6) >= tail_k)
            & (~is_head_pair_k)
        )
        died_body = self.alive & hit_body_k.any(dim=-1)
        body_killer_raw = torch.where(hit_body_k, owner_k, torch.full_like(owner_k, self.N)).amin(
            dim=-1
        )
        body_killer = torch.where(died_body, body_killer_raw, torch.full_like(body_killer_raw, -1))

        died = self.alive & (died_wall | died_body | died_head)
        survives = self.alive & (~died)

        killer_id = torch.where(
            ~died | died_wall,
            torch.full_like(body_killer, -1),
            torch.where(died_body, body_killer, head_killer),
        )
        kill_credit = (
            (killer_id[:, :, None] == self._snake_ids[:, None, :]).to(torch.int32).sum(dim=1)
        )

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["collide"] += t1 - t0
            t0 = t1

        # ---------------- Stage 3: rasterise ----------------
        cand_pos = torch.stack((cand_x, cand_y), dim=-1)
        self.head_pos = torch.where(survives[:, :, None], cand_pos, self.head_pos)
        self.theta = torch.where(survives, new_theta, self.theta)

        new_ptr = (old_ptr + 1).remainder(self.cfg.max_length)
        self.head_ptr = torch.where(survives, new_ptr, old_ptr)

        write_ptr_i64 = self.head_ptr.to(torch.int64)
        cur_ring_pos = self.ring_pos.gather(
            2, write_ptr_i64[:, :, None, None].expand(-1, -1, 1, 2)
        ).squeeze(2)
        upd_ring_pos = torch.where(survives[:, :, None], cand_pos, cur_ring_pos)
        self.ring_pos.scatter_(
            2, write_ptr_i64[:, :, None, None].expand(-1, -1, 1, 2), upd_ring_pos[:, :, None, :]
        )

        cand_cell_i16 = torch.stack((cand_gx.to(torch.int16), cand_gy.to(torch.int16)), dim=-1)
        cur_ring_cell = self.ring_cell.gather(
            2, write_ptr_i64[:, :, None, None].expand(-1, -1, 1, 2)
        ).squeeze(2)
        upd_ring_cell = torch.where(survives[:, :, None], cand_cell_i16, cur_ring_cell)
        self.ring_cell.scatter_(
            2,
            write_ptr_i64[:, :, None, None].expand(-1, -1, 1, 2),
            upd_ring_cell[:, :, None, :],
        )

        new_seg = keep_old + 1
        self.seg_count = torch.where(
            survives, new_seg, torch.where(died, torch.zeros_like(old_seg), old_seg)
        )
        self.tail_seq = torch.where(
            survives,
            torch.full_like(new_seg, seq) - new_seg + 1,
            torch.where(died, torch.full_like(self.tail_seq, seq + 1), self.tail_seq),
        )
        self.alive = survives
        self.respawn_timer = torch.where(
            died,
            torch.full_like(self.respawn_timer, int(self.cfg.respawn_delay)),
            self.respawn_timer,
        )
        self.boost_ticks = torch.where(died, torch.zeros_like(self.boost_ticks), self.boost_ticks)

        self._stamp_discs(cand_gx, cand_gy, r_sq, survives, seq=seq)

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["rasterise"] += t1 - t0
            t0 = t1

        # ---------------- Stage 4: food ----------------
        # 4a. Surviving snakes eat existing ambient and corpse pellets under their head discs.
        eat_valid_k = survives[:, :, None] & in_b_k & in_r_k
        flat_eat = flat_k.view(self.E, -1)
        eat_mask_flat = eat_valid_k.view(self.E, -1)

        amb_flat = self.ambient_grid.view(self.E, -1)
        cor_flat = self.corpse_grid.view(self.E, -1)
        eaten_amb = torch.where(
            eat_mask_flat,
            amb_flat.gather(1, flat_eat),
            torch.zeros_like(eat_mask_flat, dtype=torch.int16),
        )
        eaten_cor = torch.where(
            eat_mask_flat,
            cor_flat.gather(1, flat_eat),
            torch.zeros_like(eat_mask_flat, dtype=torch.int32),
        )

        # Surviving snakes' head discs are strictly pairwise disjoint, so scatter_add_ of
        # -eaten clears each eaten cell exactly once with zero write conflicts.
        amb_flat.scatter_add_(1, flat_eat, -eaten_amb)
        cor_flat.scatter_add_(1, flat_eat, -eaten_cor)

        hx4_k = hx_k.clamp(0, self.G - 1).to(torch.int64) >> 2
        hy4_k = hy_k.clamp(0, self.G - 1).to(torch.int64) >> 2
        flat4_eat = (hy4_k * self.G4 + hx4_k).view(self.E, -1)
        self.ambient_grid_4x.view(self.E, -1).scatter_add_(1, flat4_eat, -eaten_amb)
        self.corpse_grid_4x.view(self.E, -1).scatter_add_(1, flat4_eat, -eaten_cor)

        hx16_k = hx_k.clamp(0, self.G - 1).to(torch.int64) >> 4
        hy16_k = hy_k.clamp(0, self.G - 1).to(torch.int64) >> 4
        flat16_eat = (hy16_k * self.G16 + hx16_k).view(self.E, -1)
        self.ambient_grid_16x.view(self.E, -1).scatter_add_(1, flat16_eat, -eaten_amb)
        self.corpse_grid_16x.view(self.E, -1).scatter_add_(1, flat16_eat, -eaten_cor)

        eaten_amb_count = (eaten_amb > 0).to(torch.int32).sum(dim=1)
        self.ambient_count = self.ambient_count - eaten_amb_count

        mass_eaten = (eaten_amb.view_as(hx_k).to(torch.int32) + eaten_cor.view_as(hx_k)).sum(dim=-1)
        uncapped_len = self.length + mass_eaten
        capped_len = uncapped_len.clamp_max(int(self.cfg.max_length))
        cap_overflow = (uncapped_len - capped_len).clamp_min(0)
        self.total_length_cap_sink = self.total_length_cap_sink + cap_overflow.to(torch.int64).sum(
            dim=1
        )
        self.length = torch.where(survives, capped_len, self.length)

        # 4b. Drop shed boost trail pellets at pre-step tail cells.
        tail_gx_c = old_tail_gx.clamp(0, self.G - 1)
        tail_gy_c = old_tail_gy.clamp(0, self.G - 1)
        tail_flat = tail_gy_c * self.G + tail_gx_c
        shed_val = torch.where(
            shed,
            torch.full(
                (self.E, self.N),
                int(self.cfg.boost_shed_mass),
                dtype=torch.int32,
                device=self.device,
            ),
            torch.zeros((self.E, self.N), dtype=torch.int32, device=self.device),
        )
        cor_flat.scatter_add_(1, tail_flat, shed_val)
        self.corpse_grid_4x.view(self.E, -1).scatter_add_(
            1, (tail_gy_c >> 2) * self.G4 + (tail_gx_c >> 2), shed_val
        )
        self.corpse_grid_16x.view(self.E, -1).scatter_add_(
            1, (tail_gy_c >> 4) * self.G16 + (tail_gx_c >> 4), shed_val
        )

        # 4c. Convert dying snakes into up to corpse_max_pellets orbs along their ring path.
        dead_len = torch.where(died, self.length, torch.zeros_like(self.length))
        corpse_mass = self._corpse_mass_table[
            dead_len.clamp(0, self.cfg.max_length).to(torch.int64)
        ]
        corpse_sink = dead_len - corpse_mass
        self.total_corpse_Criteria_sink = self.total_corpse_Criteria_sink + corpse_sink.to(
            torch.int64
        ).sum(dim=1)
        self.length = torch.where(died, torch.zeros_like(self.length), self.length)

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
        orb_ring_idx = (
            (old_ptr[:, :, None] - orb_offset).remainder(self.cfg.max_length).to(torch.int64)
        )
        orb_cells = self.ring_cell.gather(2, orb_ring_idx[..., None].expand(-1, -1, -1, 2))
        orb_gx = orb_cells[..., 0].to(torch.int64).clamp(0, self.G - 1)
        orb_gy = orb_cells[..., 1].to(torch.int64).clamp(0, self.G - 1)

        base_orb = torch.div(corpse_mass, p_safe, rounding_mode="floor")[:, :, None].to(torch.int32)
        rem_orb = corpse_mass.remainder(p_safe)[:, :, None]
        orb_val = torch.where(
            active_orb,
            base_orb + (k_idx < rem_orb).to(torch.int32),
            torch.zeros_like(active_orb, dtype=torch.int32),
        ).reshape(self.E, -1)

        cor_flat.scatter_add_(1, (orb_gy * self.G + orb_gx).reshape(self.E, -1), orb_val)
        self.corpse_grid_4x.view(self.E, -1).scatter_add_(
            1, ((orb_gy >> 2) * self.G4 + (orb_gx >> 2)).reshape(self.E, -1), orb_val
        )
        self.corpse_grid_16x.view(self.E, -1).scatter_add_(
            1, ((orb_gy >> 4) * self.G16 + (orb_gx >> 4)).reshape(self.E, -1), orb_val
        )

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

        # ---------------- Stage 5: respawn ----------------
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
        rth = rng_angle_f32_torch(seeds_3d, self.tick, PURPOSE_RESPAWN, self._resp_slot_t)

        r_in_wall = (
            ready[:, :, None]
            & (rgx >= 0)
            & (rgx < self.G)
            & (rgy >= 0)
            & (rgy < self.G)
            & (self._cell_d4_int(rgx, rgy) < self._spawn_wall_d4)
        )

        cx_all = rgx[..., None] + self._clear_dx[None, None, None, :]
        cy_all = rgy[..., None] + self._clear_dy[None, None, None, :]
        c_in_grid = (cx_all >= 0) & (cx_all < self.G) & (cy_all >= 0) & (cy_all < self.G)
        c_flat = cy_all.clamp(0, self.G - 1).to(torch.int64) * self.G + cx_all.clamp(
            0, self.G - 1
        ).to(torch.int64)
        c_occ = self.occ_packed.view(self.E, -1).gather(1, c_flat.view(self.E, -1)).view_as(cx_all)
        c_owner = (c_occ & 63).clamp(0, self.N - 1).to(torch.int64)
        c_tail = self.tail_seq.gather(1, c_owner.view(self.E, -1)).view_as(cx_all)
        c_busy = (~c_in_grid) | ((c_occ >= 0) & ((c_occ >> 6) >= c_tail))

        cand_ok = r_in_wall & (~c_busy.any(dim=-1))
        has_cand = cand_ok.any(dim=-1)
        best_m = cand_ok.to(torch.int32).argmax(dim=-1, keepdim=True)
        sel_gx = rgx.gather(2, best_m).squeeze(2)
        sel_gy = rgy.gather(2, best_m).squeeze(2)
        sel_th = rth.gather(2, best_m).squeeze(2)

        # Enforce mutual clearance with all earlier slots i < j having a valid candidate.
        rdx = sel_gx[:, :, None] - sel_gx[:, None, :]
        rdy = sel_gy[:, :, None] - sel_gy[:, None, :]
        min_sep_sq = (2 * self._r_clear) * (2 * self._r_clear)
        earlier_conflict = (
            self._tril_nn & has_cand[:, None, :] & ((rdx * rdx + rdy * rdy) <= min_sep_sq)
        ).any(dim=-1)
        spawn_now = has_cand & (~earlier_conflict)

        spawn_pos = self._cell_center_f32(sel_gx, sel_gy)
        self.head_pos = torch.where(spawn_now[:, :, None], spawn_pos, self.head_pos)
        self.theta = torch.where(spawn_now, sel_th, self.theta)
        self.alive = self.alive | spawn_now
        self.length = torch.where(
            spawn_now,
            torch.full_like(self.length, int(self.cfg.start_length)),
            self.length,
        )
        self.seg_count = torch.where(spawn_now, torch.ones_like(self.seg_count), self.seg_count)
        self.head_ptr = torch.where(spawn_now, torch.zeros_like(self.head_ptr), self.head_ptr)
        self.tail_seq = torch.where(spawn_now, torch.full_like(self.tail_seq, seq), self.tail_seq)
        self.boost_ticks = torch.where(
            spawn_now, torch.zeros_like(self.boost_ticks), self.boost_ticks
        )

        spawn_cell_i16 = torch.stack((sel_gx.to(torch.int16), sel_gy.to(torch.int16)), dim=-1)
        self.ring_pos[:, :, 0, :] = torch.where(
            spawn_now[:, :, None], spawn_pos, self.ring_pos[:, :, 0, :]
        )
        self.ring_cell[:, :, 0, :] = torch.where(
            spawn_now[:, :, None], spawn_cell_i16, self.ring_cell[:, :, 0, :]
        )

        spawn_r_sq = self._r_sq_table[int(self.cfg.start_length)].expand(self.E, self.N)
        self._stamp_discs(sel_gx, sel_gy, spawn_r_sq, spawn_now, seq=seq)
        self.total_spawned_snake_mass = self.total_spawned_snake_mass + spawn_now.to(
            torch.int64
        ).sum(dim=1) * int(self.cfg.start_length)

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["respawn"] += t1 - t0

        return StepInfo(
            alive=self.alive.clone(),
            length=self.length.clone(),
            died=died,
            killer_id=killer_id,
            kill_credit=kill_credit,
            mass_eaten=mass_eaten,
        )

    def active_occupancy_grid(self) -> torch.Tensor:
        """Materialize active owner IDs ``(E, G, G)`` in int8 (``-1`` where empty)."""
        occ = self.occ_packed
        owner = (occ & 63).clamp(0, self.N - 1).to(torch.int64)
        owner_tail = self.tail_seq.gather(1, owner.view(self.E, -1)).view_as(occ)
        active = (occ >= 0) & ((occ >> 6) >= owner_tail)
        return torch.where(
            active, (occ & 63).to(torch.int8), torch.full_like(occ, -1, dtype=torch.int8)
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize full deterministic simulator state for checkpoint/resume."""
        tensor_names = (
            "seeds",
            "head_pos",
            "theta",
            "alive",
            "length",
            "seg_count",
            "head_ptr",
            "tail_seq",
            "boost_ticks",
            "respawn_timer",
            "ring_pos",
            "ring_cell",
            "prev_alive",
            "prev_head_pos",
            "prev_theta",
            "prev_length",
            "occ_packed",
            "ambient_grid",
            "ambient_grid_4x",
            "ambient_grid_16x",
            "corpse_grid",
            "corpse_grid_4x",
            "corpse_grid_16x",
            "ambient_count",
            "total_spawned_snake_mass",
            "total_spawned_ambient_mass",
            "total_corpse_Criteria_sink",
            "total_length_cap_sink",
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
