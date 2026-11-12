"""GPU-first batched fixed-point PyTorch simulator for Game V3.1 (E worlds x N snakes)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from research.game_v3.config import (
    BOOST_ACCUM_ONE,
    BOOST_ACCUM_SHIFT,
    CORPSE_GRID_MAX,
    FALLBACK_RING_SIZE,
    FALLBACK_SNAKE_STRIDE,
    FALLBACK_TICK_STRIDE,
    FOOD_PICKUP_CLAMP_FP,
    FP_HALF,
    FP_SHIFT,
    INIT_FOOD_BATCH_SIZE,
    INIT_SNAKE_CANDIDATES,
    MAX_OCC_SUBTICK,
    MAX_SNAKES,
    NUM_ACTIONS,
    NUM_TURN_LEVELS,
    OCC_OWNER_BITS,
    OCC_OWNER_MASK,
    OCC_SUBTICK_BITS,
    TRIG_HALF,
    TRIG_SHIFT,
    GameV3Config,
    build_clear_offsets_np,
    build_disc_offsets_np,
    build_eat_offsets_np,
    build_fallback_spawns_np,
    build_length_tables_np,
    build_trig_tables_np,
    decode_occ_owner_hi,
    decode_occ_owner_lo,
    food_wall_sq_fp,
    init_food_num_batches,
    occ_entry_active,
    pack_occ_hi,
    pack_occ_lo,
    spawn_wall_sq_fp,
)
from research.game_v3.rng import (
    PURPOSE_AMBIENT_FOOD,
    PURPOSE_INIT_FOOD,
    PURPOSE_INIT_SNAKE,
    PURPOSE_KILL_CREDIT,
    PURPOSE_RESPAWN,
    rng_angle_idx_torch,
    rng_int_torch,
)

_REBASE_ENV_CHUNK: int = 8

AUTO_REBASE_SUBTICK_THRESHOLD: int = 1 << (OCC_SUBTICK_BITS - 1)  # 16,777,216


@dataclass
class StepInfo:
    """Per-slot per-tick episode and physical signals emitted by :class:`GameV3Sim`.

    No reward is computed inside the simulator. Every slot-tick partitions cleanly
    into ``alive``, ``died``, or ``in_respawn_delay`` (mutually exclusive), with
    ``life_start`` marking newly spawned lives.

    Length convention:
      - ``length`` is the snake's post-tick length (``0`` on a death tick or while dead).
      - ``prev_length`` is the snake's length at the start of this tick before boost
        shedding, food pickup, death zeroing, or respawn (so on a death tick
        ``length == 0`` while ``prev_length`` preserves the pre-tick length).

    Truncation convention:
      - Trainer rollout boundaries flag ``truncated`` on surviving snakes via
        ``sim.step(actions, truncate=...)`` or :meth:`flag_truncation` WITHOUT
        removing the snake or its mass from the world.
      - If ``cfg.max_life_ticks > 0`` (disabled by default: ``0``), per-life age
        truncation additionally retires the snake on the next tick.
    """

    alive: torch.Tensor
    life_start: torch.Tensor
    died: torch.Tensor
    truncated: torch.Tensor
    in_respawn_delay: torch.Tensor
    length: torch.Tensor
    prev_length: torch.Tensor
    mass_eaten: torch.Tensor
    killer_id: torch.Tensor
    kill_credit: torch.Tensor
    life_age: torch.Tensor

    def flag_truncation(self, mask: bool | torch.Tensor = True) -> StepInfo:
        """Mark rollout-boundary truncation in-place on surviving snakes without world removal."""
        eligible = self.alive & (~self.life_start)
        if isinstance(mask, bool):
            if mask:
                self.truncated = self.truncated | eligible
            return self
        if mask.dtype != torch.bool:
            raise ValueError(f"truncation mask must be bool, got {mask.dtype}")
        if mask.shape == (eligible.shape[0],):
            m2 = mask.to(device=eligible.device)[:, None]
        elif mask.shape == eligible.shape:
            m2 = mask.to(device=eligible.device)
        else:
            raise ValueError(
                f"truncation mask must have shape {(eligible.shape[0],)} or "
                f"{tuple(eligible.shape)}, got {tuple(mask.shape)}"
            )
        self.truncated = self.truncated | (eligible & m2)
        return self


@dataclass(frozen=True)
class StateField:
    """Declarative specification for one simulator state tensor on :class:`GameV3Sim`."""

    name: str
    shape_kind: str
    dtype: torch.dtype
    reset_value: int | bool
    saved: bool = True
    reset_on_clear: bool = True
    sharded_4x: bool = False
    is_ring: bool = False
    per_life: bool = False
    prev_source: str | None = None
    in_reference_snake: bool = False


STATE_FIELDS: tuple[StateField, ...] = (
    StateField("seeds", "E", torch.int64, 0, reset_on_clear=False),
    StateField("world_ticks", "E", torch.int64, 0),
    StateField("head_pos", "ES2", torch.int32, 0, in_reference_snake=True),
    StateField("theta", "ES", torch.int32, 0, in_reference_snake=True),
    StateField("alive", "ES", torch.bool, False, in_reference_snake=True),
    StateField("boosting", "ES", torch.bool, False, per_life=True, in_reference_snake=True),
    StateField("length", "ES", torch.int32, 0, in_reference_snake=True),
    StateField("seg_count", "ES", torch.int32, 0, in_reference_snake=True),
    StateField("head_ptr", "ES", torch.int32, 0),
    StateField("oldest_active_subtick", "ES", torch.int32, 0),
    StateField("dist_accum_fp", "ES", torch.int32, 0, per_life=True, in_reference_snake=True),
    StateField("boost_ticks", "ES", torch.int32, 0, per_life=True, in_reference_snake=True),
    StateField("boost_drop_fp_accum", "ES", torch.int32, 0, per_life=True, in_reference_snake=True),
    StateField("respawn_timer", "ES", torch.int32, 0, in_reference_snake=True),
    StateField("life_age", "ES", torch.int32, 0, per_life=True, in_reference_snake=True),
    StateField("life_start", "ES", torch.bool, False, in_reference_snake=True),
    StateField("truncated_pending", "ES", torch.bool, False, in_reference_snake=True),
    StateField("ring_pos", "RING2", torch.int32, 0, is_ring=True),
    StateField("ring_cell", "RING2", torch.int16, 0, is_ring=True),
    StateField("ring_r_sq", "RING", torch.int16, 0, is_ring=True),
    StateField("ring_subtick", "RING", torch.int32, 0, is_ring=True),
    StateField("prev_alive", "ES", torch.bool, False, prev_source="alive", in_reference_snake=True),
    StateField(
        "prev_boosting", "ES", torch.bool, False, prev_source="boosting", in_reference_snake=True
    ),
    StateField(
        "prev_head_pos", "ES2", torch.int32, 0, prev_source="head_pos", in_reference_snake=True
    ),
    StateField("prev_theta", "ES", torch.int32, 0, prev_source="theta", in_reference_snake=True),
    StateField("prev_length", "ES", torch.int32, 0, prev_source="length", in_reference_snake=True),
    StateField("occ_hi", "G1", torch.int32, -1),
    StateField("occ_lo", "G1", torch.int32, -1),
    StateField("occ_subtick_4x", "G4N", torch.int32, -1, sharded_4x=True),
    StateField("occ_subtick_16x", "G16N", torch.int32, -1),
    StateField("ambient_grid", "G1", torch.uint8, 0),
    StateField("ambient_grid_4x", "G4", torch.int32, 0),
    StateField("ambient_grid_16x", "G16", torch.int32, 0),
    StateField("corpse_grid", "G1", torch.int16, 0),
    StateField("corpse_grid_4x", "G4", torch.int32, 0),
    StateField("corpse_grid_16x", "G16", torch.int32, 0),
    StateField("ambient_count", "E", torch.int32, 0),
    StateField("total_spawned_snake_mass", "E", torch.int64, 0),
    StateField("total_spawned_ambient_mass", "E", torch.int64, 0),
    StateField("total_corpse_decay_sink", "E", torch.int64, 0),
    StateField("total_boost_decay_sink", "E", torch.int64, 0),
    StateField("total_length_cap_sink", "E", torch.int64, 0),
    StateField("total_truncation_sink", "E", torch.int64, 0),
)

_SAVED_STATE_FIELDS: tuple[StateField, ...] = tuple(f for f in STATE_FIELDS if f.saved)
_SAVED_TENSOR_NAMES: frozenset[str] = frozenset(f.name for f in _SAVED_STATE_FIELDS)
_CLEARABLE_STATE_FIELDS: tuple[StateField, ...] = tuple(f for f in STATE_FIELDS if f.reset_on_clear)
_PER_LIFE_STATE_FIELDS: tuple[StateField, ...] = tuple(f for f in STATE_FIELDS if f.per_life)
_STAGE3_SURVIVE_HANDLED_NAMES: frozenset[str] = frozenset({"boosting", "dist_accum_fp", "life_age"})
_DEATH_ONLY_PER_LIFE_FIELDS: tuple[StateField, ...] = tuple(
    f for f in _PER_LIFE_STATE_FIELDS if f.name not in _STAGE3_SURVIVE_HANDLED_NAMES
)
_PREV_SNAPSHOT_FIELDS: tuple[StateField, ...] = tuple(
    f for f in STATE_FIELDS if f.prev_source is not None
)
_REQUIRED_STATE_DICT_KEYS: frozenset[str] = frozenset(
    {"config", "num_envs", "tick", "subtick_offset", "tensors"}
)
_OPTIONAL_STATE_DICT_KEYS: frozenset[str] = frozenset({"clean_step_streak"})
_ALL_STATE_DICT_KEYS: frozenset[str] = _REQUIRED_STATE_DICT_KEYS | _OPTIONAL_STATE_DICT_KEYS


def state_field_shape(sim: GameV3Sim, shape_kind: str) -> tuple[int, ...]:
    """Resolve a :class:`StateField` ``shape_kind`` against ``sim`` dimensions."""
    max_len = int(sim.cfg.max_length)
    if shape_kind == "E":
        return (sim.E,)
    if shape_kind == "ES":
        return (sim.E, sim.N)
    if shape_kind == "ES2":
        return (sim.E, sim.N, 2)
    if shape_kind == "RING":
        return (sim.E, sim.N, max_len)
    if shape_kind == "RING2":
        return (sim.E, sim.N, max_len, 2)
    if shape_kind == "G1":
        return (sim.E, sim.G, sim.G)
    if shape_kind == "G4":
        return (sim.E, sim.G4, sim.G4)
    if shape_kind == "G16":
        return (sim.E, sim.G16, sim.G16)
    if shape_kind == "G4N":
        return (sim.E, sim.G4, sim.G4, sim.N)
    if shape_kind == "G16N":
        return (sim.E, sim.G16, sim.G16, sim.N)
    raise ValueError(f"Unknown StateField shape_kind: {shape_kind!r}")


def step_delta_fp_torch(dist_fp: torch.Tensor, trig_q14: torch.Tensor) -> torch.Tensor:
    """Compute symmetric nearest-integer fixed-point step ``round(dist_fp * trig_q14 / 2^14)``."""
    prod = dist_fp * trig_q14
    pos = (prod + TRIG_HALF) >> TRIG_SHIFT
    neg = -((-prod + TRIG_HALF) >> TRIG_SHIFT)
    return torch.where(prod >= 0, pos, neg)


def _build_static_tables(cfg: GameV3Config, device: torch.device) -> dict[str, Any]:
    """Precompute shared integer trig, length, disc-overlap, pickup, and spawn tables."""
    sin_np, cos_np = build_trig_tables_np(cfg.angle_steps)
    lt_np = build_length_tables_np(cfg)
    fb_np = build_fallback_spawns_np(cfg)

    r_sq_np = lt_np["r_sq_cells"]
    max_r_cells, disc_dist_sq_np, disc_dy_np, disc_dx_np = build_disc_offsets_np(cfg)

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

    max_r_sq_val = int(max(unique_r_sq))
    r_sq_to_tier_np = np.zeros(max_r_sq_val + 1, dtype=np.int64)
    for val, idx in tier_map.items():
        r_sq_to_tier_np[int(val)] = idx

    # Deduplicated 4x and 16x block stencils per (tier, gy_mod, gx_mod) so coarse
    # pyramid updates have strictly unique block indices per (env, slot, subpoint).
    d4_min, d4_max = (-max_r_cells) >> 2, (3 + max_r_cells) >> 2
    d4_pairs = [
        (dy4, dx4) for dy4 in range(d4_min, d4_max + 1) for dx4 in range(d4_min, d4_max + 1)
    ]
    d4_idx_map = {pair: idx for idx, pair in enumerate(d4_pairs)}
    disc_4x_dy_np = np.array([p[0] for p in d4_pairs], dtype=np.int32)
    disc_4x_dx_np = np.array([p[1] for p in d4_pairs], dtype=np.int32)
    disc_4x_mask_np = np.zeros((num_tiers, 4, 4, len(d4_pairs)), dtype=bool)
    for ti in range(num_tiers):
        for ry in range(4):
            for rx in range(4):
                for dx, dy in tier_masks[ti]:
                    k4 = d4_idx_map[((ry + dy) >> 2, (rx + dx) >> 2)]
                    disc_4x_mask_np[ti, ry, rx, k4] = True

    d16_min, d16_max = (-max_r_cells) >> 4, (15 + max_r_cells) >> 4
    d16_pairs = [
        (dy16, dx16) for dy16 in range(d16_min, d16_max + 1) for dx16 in range(d16_min, d16_max + 1)
    ]
    d16_idx_map = {pair: idx for idx, pair in enumerate(d16_pairs)}
    disc_16x_dy_np = np.array([p[0] for p in d16_pairs], dtype=np.int32)
    disc_16x_dx_np = np.array([p[1] for p in d16_pairs], dtype=np.int32)
    disc_16x_mask_np = np.zeros((num_tiers, 16, 16, len(d16_pairs)), dtype=bool)
    for ti in range(num_tiers):
        for ry in range(16):
            for rx in range(16):
                for dx, dy in tier_masks[ti]:
                    k16 = d16_idx_map[((ry + dy) >> 4, (rx + dx) >> 4)]
                    disc_16x_mask_np[ti, ry, rx, k16] = True

    max_eat_r_cells, eat_dy_np, eat_dx_np = build_eat_offsets_np(cfg)
    r_clear, clear_dx_np, clear_dy_np = build_clear_offsets_np(cfg)

    start_r_fp = int(lt_np["radius_fp"][cfg.start_length])
    spawn_wall_sq = spawn_wall_sq_fp(cfg, start_r_fp)
    food_wall_sq = food_wall_sq_fp(cfg)

    out: dict[str, Any] = {
        "sin_table": torch.as_tensor(sin_np, device=device),
        "cos_table": torch.as_tensor(cos_np, device=device),
        "fallback_spawns": torch.as_tensor(fb_np, device=device),
        "r_tier_table": torch.as_tensor(r_tier_np, device=device),
        "r_sq_to_tier": torch.as_tensor(r_sq_to_tier_np, device=device),
        "disc_dx": torch.as_tensor(disc_dx_np, device=device),
        "disc_dy": torch.as_tensor(disc_dy_np, device=device),
        "disc_dist_sq": torch.as_tensor(disc_dist_sq_np, device=device),
        "disc_4x_dx": torch.as_tensor(disc_4x_dx_np, device=device),
        "disc_4x_dy": torch.as_tensor(disc_4x_dy_np, device=device),
        "disc_4x_mask": torch.as_tensor(disc_4x_mask_np, device=device),
        "disc_16x_dx": torch.as_tensor(disc_16x_dx_np, device=device),
        "disc_16x_dy": torch.as_tensor(disc_16x_dy_np, device=device),
        "disc_16x_mask": torch.as_tensor(disc_16x_mask_np, device=device),
        "overlap_lut": torch.as_tensor(overlap_lut_np, device=device),
        "eat_dx": torch.as_tensor(eat_dx_np, device=device),
        "eat_dy": torch.as_tensor(eat_dy_np, device=device),
        "clear_dx": torch.as_tensor(clear_dx_np, device=device),
        "clear_dy": torch.as_tensor(clear_dy_np, device=device),
        "max_diam": max_diam,
        "max_eat_r_cells": max_eat_r_cells,
        "r_clear": r_clear,
        "spawn_wall_sq_fp": spawn_wall_sq,
        "food_wall_sq_fp": food_wall_sq,
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
        auto_rebase: bool = True,
        debug_assert_invariants: bool = False,
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
        self._start_r_sq_int: int = int(self._r_sq_table[self.cfg.start_length].item())
        self._r_tier_table: torch.Tensor = tables["r_tier_table"]
        self._r_sq_to_tier: torch.Tensor = tables["r_sq_to_tier"]
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
        self._disc_4x_dx: torch.Tensor = tables["disc_4x_dx"]
        self._disc_4x_dy: torch.Tensor = tables["disc_4x_dy"]
        self._disc_4x_mask: torch.Tensor = tables["disc_4x_mask"]
        self._disc_16x_dx: torch.Tensor = tables["disc_16x_dx"]
        self._disc_16x_dy: torch.Tensor = tables["disc_16x_dy"]
        self._disc_16x_mask: torch.Tensor = tables["disc_16x_mask"]
        self._overlap_lut: torch.Tensor = tables["overlap_lut"]
        self._eat_dx: torch.Tensor = tables["eat_dx"]
        self._eat_dy: torch.Tensor = tables["eat_dy"]
        self._clear_dx: torch.Tensor = tables["clear_dx"]
        self._clear_dy: torch.Tensor = tables["clear_dy"]
        self._max_diam: int = int(tables["max_diam"])
        self._max_eat_r_cells: int = int(tables["max_eat_r_cells"])
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
        # Order-independent dual-bound occupancy grids at 1x:
        # - occ_hi stores max((subtick << 6) | slot_id)
        # - occ_lo stores max((subtick << 6) | (63 - slot_id))
        #
        # Surviving-disc non-overlap invariant (why 2 owner slots per cell suffices):
        #   1. Base case (reset / reset_worlds / Stage 5 respawn): newly placed snakes
        #      require empty clearance >= r_clear around active discs and pairwise center
        #      separation > 2 * r_clear, so active discs never overlap at spawn.
        #   2. Inductive step (Stage 2 -> Stage 3): Stage 2 tests every alive snake's new
        #      sub-step head disc(s) against all other snakes' retained segments and
        #      causally ordered new points (s_j <= s_i; total order over integers). If two
        #      snakes' discs share any cell, at least one dies in Stage 2. Stage 3 stamps
        #      ONLY surviving snakes and immediately expires dying snakes via
        #      oldest_active_subtick. Thus no grid cell ever has >= 2 active owners during
        #      gameplay (<= 1 active owner per cell), and occ_hi / occ_lo also keep
        #      synthetic multi-owner test placements order-independent without own-masking.
        #
        # Shard occ_subtick_4x into <=256-env blocks so every tensor stays strictly < INT_MAX
        # elements on MPSGraph even at E >= 1024.
        self.tick: int = 0
        self._subtick_offset: int = 0
        self._clean_step_streak: int = 0
        self.auto_rebase: bool = bool(auto_rebase)
        self.debug_assert_invariants: bool = bool(debug_assert_invariants)

        max_per_env = max(1, self.G * self.G, self.G4 * self.G4 * self.N)
        self.env_block_size: int = min(256, max(1, ((1 << 31) - 1024) // max_per_env))
        self._env_block_1x: int = (
            self.E
            if self.E * self.G * self.G < ((1 << 31) - 1024)
            else max(1, ((1 << 31) - 1024) // max(1, self.G * self.G))
        )
        self._occ_4x_blocks: list[torch.Tensor] = []
        self._allocate_state_tensors()

        self._timing_enabled: bool = False
        self._stage_seconds: dict[str, float] = {
            "move": 0.0,
            "collide": 0.0,
            "rasterise": 0.0,
            "food": 0.0,
            "respawn": 0.0,
        }

        self.reset(seeds=seeds)

    def _allocate_state_tensors(self) -> None:
        """Allocate every registered state tensor (and sharded 4x occupancy blocks) from STATE_FIELDS."""
        for field in STATE_FIELDS:
            shape = state_field_shape(self, field.shape_kind)
            if field.sharded_4x:
                tail_shape = shape[1:]
                self._occ_4x_blocks = [
                    torch.full(
                        (min(self.E, e0 + self.env_block_size) - e0, *tail_shape),
                        field.reset_value,
                        dtype=field.dtype,
                        device=self.device,
                    )
                    for e0 in range(0, self.E, self.env_block_size)
                ]
            elif field.reset_value == 0 or field.reset_value is False:
                setattr(
                    self,
                    field.name,
                    torch.zeros(shape, dtype=field.dtype, device=self.device),
                )
            else:
                setattr(
                    self,
                    field.name,
                    torch.full(shape, field.reset_value, dtype=field.dtype, device=self.device),
                )

    @property
    def clean_step_streak(self) -> int:
        """Consecutive clean full-world steps since last world reset or state load."""
        return self._clean_step_streak

    @clean_step_streak.setter
    def clean_step_streak(self, value: int) -> None:
        val = int(value)
        if val < 0:
            raise ValueError(f"clean_step_streak must be >= 0, got {val}")
        self._clean_step_streak = val

    @property
    def num_actions(self) -> int:
        """Number of discrete actions exposed by the simulator (10)."""
        return NUM_ACTIONS

    @property
    def occ_subtick_4x(self) -> torch.Tensor:
        """Return ``(E, G4, G4, N)`` on ``self.device`` (view if 1 block, else full device copy).

        Debug/test use only on multi-block sims (``len(self._occ_4x_blocks) > 1``, e.g.
        ``E > 256``): when sharded across multiple ``_occ_4x_blocks``, ``torch.cat`` allocates
        and materializes a full concatenated device copy (mutations to that copy do not write
        back to ``_occ_4x_blocks``). When ``self.E <= self.env_block_size`` (single shard),
        returns the underlying block view with zero allocation. Production code paths iterate
        ``_occ_4x_blocks`` directly or call :meth:`occ_4x_slice`.
        """
        if len(self._occ_4x_blocks) == 1:
            return self._occ_4x_blocks[0]
        total_elems = self.E * self.G4 * self.G4 * self.N
        if self.device.type == "mps" and total_elems >= (1 << 31):
            raise RuntimeError(
                f"occ_subtick_4x has {total_elems} elements (>= 2^31 MPS limit); "
                "use _occ_4x_blocks or occ_4x_slice() instead"
            )
        # Multi-block debug/test fallback: materializes a full concatenated copy on self.device.
        return torch.cat(self._occ_4x_blocks, dim=0)

    def occ_4x_slice(self, e0: int, e1: int) -> torch.Tensor:
        """Return a view into the single ``_occ_4x_blocks`` shard containing ``[e0:e1]``."""
        bs = self.env_block_size
        b_idx = e0 // bs
        b_off = e0 - b_idx * bs
        return self._occ_4x_blocks[b_idx][b_off : b_off + (e1 - e0)]

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

    def _orient_spawn_heading_torch(
        self, gx: torch.Tensor, gy: torch.Tensor, th: torch.Tensor
    ) -> torch.Tensor:
        """Flip spawn heading by half a turn when its radial dot product points toward the wall."""
        cx = ((gx.to(torch.int64) - self.half_grid) << FP_SHIFT) + FP_HALF
        cy = ((gy.to(torch.int64) - self.half_grid) << FP_SHIFT) + FP_HALF
        angle_mask = int(self.cfg.angle_steps - 1)
        th_m = th.to(torch.int32) & angle_mask
        th_i64 = th_m.to(torch.int64)
        outward_dot = cx * self._cos_table[th_i64].to(torch.int64) + cy * self._sin_table[
            th_i64
        ].to(torch.int64)
        flipped = (th_m + (int(self.cfg.angle_steps) >> 1)) & angle_mask
        return torch.where(outward_dot > 0, flipped, th_m)

    def _decode_pair_active(
        self,
        val_hi: torch.Tensor,
        val_lo: torch.Tensor,
        old_src: torch.Tensor,
    ) -> torch.Tensor:
        """Return boolean mask true where either ``occ_hi`` or ``occ_lo`` has an active owner."""
        ow_hi = decode_occ_owner_hi(val_hi).clamp(0, self.N - 1).to(torch.int64)
        act_hi = occ_entry_active(val_hi, old_src.gather(1, ow_hi))
        ow_lo = decode_occ_owner_lo(val_lo).clamp(0, self.N - 1).to(torch.int64)
        act_lo = occ_entry_active(val_lo, old_src.gather(1, ow_lo))
        return act_hi | act_lo

    def _is_cell_occupied(
        self, flat_idx: torch.Tensor, oldest_subtick: torch.Tensor
    ) -> torch.Tensor:
        """Return bool mask of shape ``flat_idx.shape`` true where any snake is active."""
        e_cnt = int(flat_idx.shape[0])
        flat_2d = flat_idx.reshape(e_cnt, -1)
        eb_1x = self._env_block_1x
        if e_cnt <= eb_1x:
            hi_src = self.occ_hi if e_cnt == self.E else self.occ_hi[:e_cnt]
            lo_src = self.occ_lo if e_cnt == self.E else self.occ_lo[:e_cnt]
            old_src = oldest_subtick if e_cnt == self.E else oldest_subtick[:e_cnt]
            val_hi = hi_src.view(e_cnt, -1).gather(1, flat_2d)
            val_lo = lo_src.view(e_cnt, -1).gather(1, flat_2d)
            return self._decode_pair_active(val_hi, val_lo, old_src).view_as(flat_idx)

        out_2d = torch.empty_like(flat_2d, dtype=torch.bool)
        for e0 in range(0, e_cnt, eb_1x):
            e1 = min(e_cnt, e0 + eb_1x)
            eb = e1 - e0
            f_sub = flat_2d[e0:e1]
            old_sub = oldest_subtick[e0:e1]
            val_hi = self.occ_hi[e0:e1].view(eb, -1).gather(1, f_sub)
            val_lo = self.occ_lo[e0:e1].view(eb, -1).gather(1, f_sub)
            out_2d[e0:e1] = self._decode_pair_active(val_hi, val_lo, old_sub)
        return out_2d.view_as(flat_idx)

    def _is_cell_occupied_rows(
        self,
        row_envs: torch.Tensor,
        flat_idx: torch.Tensor,
        oldest_subtick: torch.Tensor,
    ) -> torch.Tensor:
        """Return bool active-occupancy mask for arbitrary row world indices ``row_envs``."""
        r_cnt = int(row_envs.shape[0])
        if r_cnt == 0:
            return torch.zeros(flat_idx.shape, dtype=torch.bool, device=self.device)
        flat_2d = flat_idx.reshape(r_cnt, -1)
        g2 = self.G * self.G
        eb_1x = self._env_block_1x
        if eb_1x == self.E:
            g_idx = (row_envs[:, None] * g2 + flat_2d).reshape(-1)
            old_sub = oldest_subtick.index_select(0, row_envs)
            v_hi = self.occ_hi.view(-1).index_select(0, g_idx).view_as(flat_2d)
            v_lo = self.occ_lo.view(-1).index_select(0, g_idx).view_as(flat_2d)
            return self._decode_pair_active(v_hi, v_lo, old_sub).view_as(flat_idx)

        out_2d = torch.zeros_like(flat_2d, dtype=torch.bool)
        for e0 in range(0, self.E, eb_1x):
            e1 = min(self.E, e0 + eb_1x)
            in_blk = (row_envs >= e0) & (row_envs < e1)
            rows = torch.where(in_blk)[0]
            if rows.numel() == 0:
                continue
            local_e = row_envs.index_select(0, rows) - e0
            f_sub = flat_2d.index_select(0, rows)
            g_idx = (local_e[:, None] * g2 + f_sub).reshape(-1)
            old_sub = oldest_subtick.index_select(0, row_envs.index_select(0, rows))
            v_hi = self.occ_hi[e0:e1].view(-1).index_select(0, g_idx).view_as(f_sub)
            v_lo = self.occ_lo[e0:e1].view(-1).index_select(0, g_idx).view_as(f_sub)
            out_2d.index_copy_(0, rows, self._decode_pair_active(v_hi, v_lo, old_sub))
        return out_2d.view_as(flat_idx)

    def _gather_1x_grid_chunked(self, grid: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
        """Gather ``flat_idx`` ``(E, M)`` from a ``(E, G, G)`` grid without exceeding ``2^31``."""
        e_cnt = int(flat_idx.shape[0])
        flat_2d = flat_idx.reshape(e_cnt, -1)
        eb_1x = self._env_block_1x
        if e_cnt <= eb_1x:
            src = grid if e_cnt == self.E else grid[:e_cnt]
            return src.view(e_cnt, -1).gather(1, flat_2d).view_as(flat_idx)
        out_2d = torch.empty(flat_2d.shape, dtype=grid.dtype, device=self.device)
        for e0 in range(0, e_cnt, eb_1x):
            e1 = min(e_cnt, e0 + eb_1x)
            out_2d[e0:e1] = grid[e0:e1].view(e1 - e0, -1).gather(1, flat_2d[e0:e1])
        return out_2d.view_as(flat_idx)

    def _gather_1x_grid_rows(
        self, grid: torch.Tensor, row_envs: torch.Tensor, flat_idx: torch.Tensor
    ) -> torch.Tensor:
        """Gather ``flat_idx`` ``(R, M)`` at world indices ``row_envs`` without copying ``grid``."""
        r_cnt = int(row_envs.shape[0])
        if r_cnt == 0:
            return torch.empty(flat_idx.shape, dtype=grid.dtype, device=self.device)
        flat_2d = flat_idx.reshape(r_cnt, -1)
        g2 = self.G * self.G
        eb_1x = self._env_block_1x
        if eb_1x == self.E:
            g_idx = (row_envs[:, None] * g2 + flat_2d).reshape(-1)
            return grid.view(-1).index_select(0, g_idx).view_as(flat_idx)

        out_2d = torch.empty(flat_2d.shape, dtype=grid.dtype, device=self.device)
        for e0 in range(0, self.E, eb_1x):
            e1 = min(self.E, e0 + eb_1x)
            rows = torch.where((row_envs >= e0) & (row_envs < e1))[0]
            if rows.numel() == 0:
                continue
            local_e = row_envs.index_select(0, rows) - e0
            f_sub = flat_2d.index_select(0, rows)
            g_idx = (local_e[:, None] * g2 + f_sub).reshape(-1)
            vals = grid[e0:e1].view(-1).index_select(0, g_idx).view_as(f_sub)
            out_2d.index_copy_(0, rows, vals)
        return out_2d.view_as(flat_idx)

    def _put_1x_grid_chunked(
        self,
        grid: torch.Tensor,
        cell_flat: torch.Tensor,
        write_mask: torch.Tensor,
        values: torch.Tensor | int,
    ) -> None:
        """Write ``values`` at pairwise-disjoint ``(env, cell_flat)`` where ``write_mask`` holds."""
        g2 = self.G * self.G
        eb_1x = self._env_block_1x
        if eb_1x == self.E:
            sub_g = grid.view(-1)
            idx_1d = (self._env_base_1x + cell_flat).reshape(-1)
            m_1d = write_mask.reshape(-1)
            orig0 = sub_g[0:1].clone()
            redir = torch.where(m_1d, idx_1d, 0)
            at0 = m_1d & (idx_1d == 0)
            wrote0 = at0.any(dim=0, keepdim=True)
            if isinstance(values, int):
                v_const = torch.full((1,), values, dtype=grid.dtype, device=self.device)
                sub_g.index_put_((redir,), v_const.expand_as(redir), accumulate=False)
                sub_g[0:1] = torch.where(wrote0, v_const, orig0)
            else:
                val_1d = values.reshape(-1).to(grid.dtype)
                sub_g.index_put_((redir,), val_1d, accumulate=False)
                val0 = (
                    torch.where(at0, val_1d.to(torch.int32), 0)
                    .sum(dim=0, keepdim=True)
                    .to(grid.dtype)
                )
                sub_g[0:1] = torch.where(wrote0, val0, orig0)
            return

        for e0 in range(0, self.E, eb_1x):
            e1 = min(self.E, e0 + eb_1x)
            sub_g = grid[e0:e1].view(-1)
            eb = e1 - e0
            env_off = torch.arange(eb, dtype=torch.int64, device=self.device)[:, None] * g2
            idx_1d = (env_off + cell_flat[e0:e1]).reshape(-1)
            m_1d = write_mask[e0:e1].reshape(-1)
            orig0 = sub_g[0:1].clone()
            redir = torch.where(m_1d, idx_1d, 0)
            at0 = m_1d & (idx_1d == 0)
            wrote0 = at0.any(dim=0, keepdim=True)
            if isinstance(values, int):
                v_const = torch.full((1,), values, dtype=grid.dtype, device=self.device)
                sub_g.index_put_((redir,), v_const.expand_as(redir), accumulate=False)
                sub_g[0:1] = torch.where(wrote0, v_const, orig0)
            else:
                val_1d = values[e0:e1].reshape(-1).to(grid.dtype)
                sub_g.index_put_((redir,), val_1d, accumulate=False)
                val0 = (
                    torch.where(at0, val_1d.to(torch.int32), 0)
                    .sum(dim=0, keepdim=True)
                    .to(grid.dtype)
                )
                sub_g[0:1] = torch.where(wrote0, val0, orig0)

    @staticmethod
    def _max_scatter_(
        flat_grid: torch.Tensor,
        idx: torch.Tensor,
        cand: torch.Tensor,
        valid: torch.Tensor,
        zero_i32: torch.Tensor,
    ) -> None:
        """Order-independently update ``flat_grid[idx] = max(flat_grid[idx], cand)`` where ``valid``."""
        cur = flat_grid.index_select(0, idx)
        delta = torch.where(valid, torch.maximum(cur, cand) - cur, zero_i32)
        flat_grid.index_put_((idx,), delta, accumulate=True)

    def _stamp_rows(
        self,
        e0: int,
        e1: int,
        env_local: torch.Tensor,
        gx: torch.Tensor,
        gy: torch.Tensor,
        r_sq: torch.Tensor | int,
        tier: torch.Tensor,
        sids: torch.Tensor,
        sids_i64: torch.Tensor,
        subtick: torch.Tensor | int,
        act_mask: torch.Tensor | None,
        zero_i32: torch.Tensor,
        subtick_scalar: torch.Tensor | None = None,
    ) -> None:
        """Stamp 1D sparse rows ``(R,)`` or 2D block points ``(eb, S)`` into ``[e0:e1]``."""
        full_env = e0 == 0 and e1 == self.E
        env_exp = env_local[..., None]
        act_exp = act_mask[..., None] if act_mask is not None else None

        # 1. Fine 1x dual-bound grids (occ_hi & occ_lo).
        sx = gx[..., None] + self._disc_dx
        sy = gy[..., None] + self._disc_dy
        r_bound = r_sq if isinstance(r_sq, int) else r_sq[..., None]
        in_1x = (
            (sx >= 0) & (sx < self.G) & (sy >= 0) & (sy < self.G) & (self._disc_dist_sq <= r_bound)
        )
        valid_1x = (act_exp & in_1x if act_exp is not None else in_1x).reshape(-1)
        sx_c = sx.clamp(0, self.G - 1).to(torch.int64)
        sy_c = sy.clamp(0, self.G - 1).to(torch.int64)
        idx_1x = (env_exp * (self.G * self.G) + sy_c * self.G + sx_c).reshape(-1)

        cand_hi = pack_occ_hi(subtick, sids)[..., None].expand_as(sx).reshape(-1)
        cand_lo = pack_occ_lo(subtick, sids)[..., None].expand_as(sx).reshape(-1)
        occ_hi_1d = (self.occ_hi if full_env else self.occ_hi[e0:e1]).view(-1)
        occ_lo_1d = (self.occ_lo if full_env else self.occ_lo[e0:e1]).view(-1)
        self._max_scatter_(occ_hi_1d, idx_1x, cand_hi, valid_1x, zero_i32)
        self._max_scatter_(occ_lo_1d, idx_1x, cand_lo, valid_1x, zero_i32)

        # 2. Deduplicated 4x max-pooled per-snake subtick pyramid (N-last).
        mask_4x = self._disc_4x_mask[tier, (gy & 3).to(torch.int64), (gx & 3).to(torch.int64)]
        gx4 = (gx >> 2)[..., None] + self._disc_4x_dx
        gy4 = (gy >> 2)[..., None] + self._disc_4x_dy
        in_4x = mask_4x & (gx4 >= 0) & (gx4 < self.G4) & (gy4 >= 0) & (gy4 < self.G4)
        valid_4x = (act_exp & in_4x if act_exp is not None else in_4x).reshape(-1)
        gx4_c = gx4.clamp(0, self.G4 - 1).to(torch.int64)
        gy4_c = gy4.clamp(0, self.G4 - 1).to(torch.int64)
        idx_4x = (
            (env_exp * (self.G4 * self.G4) + gy4_c * self.G4 + gx4_c) * self.N + sids_i64[..., None]
        ).reshape(-1)
        if subtick_scalar is not None:
            cand_st4 = subtick_scalar.expand_as(idx_4x)
        else:
            assert isinstance(subtick, torch.Tensor)
            cand_st4 = subtick[..., None].expand_as(gx4).reshape(-1)
        occ_4x_1d = (
            self._occ_4x_blocks[0]
            if (full_env and len(self._occ_4x_blocks) == 1)
            else self.occ_4x_slice(e0, e1)
        ).view(-1)
        self._max_scatter_(occ_4x_1d, idx_4x, cand_st4, valid_4x, zero_i32)

        # 3. Deduplicated 16x max-pooled per-snake subtick pyramid (N-last).
        mask_16x = self._disc_16x_mask[tier, (gy & 15).to(torch.int64), (gx & 15).to(torch.int64)]
        gx16 = (gx >> 4)[..., None] + self._disc_16x_dx
        gy16 = (gy >> 4)[..., None] + self._disc_16x_dy
        in_16x = mask_16x & (gx16 >= 0) & (gx16 < self.G16) & (gy16 >= 0) & (gy16 < self.G16)
        valid_16x = (act_exp & in_16x if act_exp is not None else in_16x).reshape(-1)
        gx16_c = gx16.clamp(0, self.G16 - 1).to(torch.int64)
        gy16_c = gy16.clamp(0, self.G16 - 1).to(torch.int64)
        idx_16x = (
            (env_exp * (self.G16 * self.G16) + gy16_c * self.G16 + gx16_c) * self.N
            + sids_i64[..., None]
        ).reshape(-1)
        if subtick_scalar is not None:
            cand_st16 = subtick_scalar.expand_as(idx_16x)
        else:
            assert isinstance(subtick, torch.Tensor)
            cand_st16 = subtick[..., None].expand_as(gx16).reshape(-1)
        occ_16x_1d = (self.occ_subtick_16x if full_env else self.occ_subtick_16x[e0:e1]).view(-1)
        self._max_scatter_(occ_16x_1d, idx_16x, cand_st16, valid_16x, zero_i32)

    def _stamp_sparse_points(
        self,
        env_rows: torch.Tensor,
        slot_rows: torch.Tensor,
        gx_s: torch.Tensor,
        gy_s: torch.Tensor,
        r_sq_val: int,
        subtick_val: int,
        *,
        active_rows: torch.Tensor | None = None,
    ) -> None:
        """Stamp ``S`` pairwise-disjoint single-point discs at ``(env_rows, slot_rows)``."""
        if int(env_rows.shape[0]) == 0:
            return
        z_i32 = torch.zeros((1,), dtype=torch.int32, device=self.device)
        tier_t = self._r_sq_to_tier[r_sq_val]
        cand_st_scalar = torch.full((1,), subtick_val, dtype=torch.int32, device=self.device)

        if len(self._occ_4x_blocks) == 1 and self._env_block_1x == self.E:
            self._stamp_rows(
                0,
                self.E,
                env_rows,
                gx_s,
                gy_s,
                r_sq_val,
                tier_t,
                slot_rows.to(torch.int32),
                slot_rows.to(torch.int64),
                subtick_val,
                active_rows,
                z_i32,
                subtick_scalar=cand_st_scalar,
            )
            return

        for e0 in range(0, self.E, self.env_block_size):
            e1 = min(self.E, e0 + self.env_block_size)
            in_b = (env_rows >= e0) & (env_rows < e1)
            if active_rows is not None:
                in_b = in_b & active_rows
            rows = torch.where(in_b)[0]
            if int(rows.shape[0]) == 0:
                continue
            sids = slot_rows.index_select(0, rows).to(torch.int32)
            self._stamp_rows(
                e0,
                e1,
                env_rows.index_select(0, rows) - e0,
                gx_s.index_select(0, rows),
                gy_s.index_select(0, rows),
                r_sq_val,
                tier_t,
                sids,
                sids.to(torch.int64),
                subtick_val,
                None,
                z_i32,
                subtick_scalar=cand_st_scalar,
            )

    def _stamp_slice_point(
        self,
        e0: int,
        e1: int,
        gx_p: torch.Tensor,
        gy_p: torch.Tensor,
        r_sq_p: torch.Tensor,
        act_p: torch.Tensor,
        st_p: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> None:
        """Stamp one subpoint ``(eb, S)`` of pairwise-disjoint discs into ``[e0:e1]``."""
        eb = e1 - e0
        env_local = torch.arange(eb, dtype=torch.int64, device=self.device)[:, None]
        z_i32 = torch.zeros((1,), dtype=torch.int32, device=self.device)
        tier = self._r_sq_to_tier[r_sq_p.clamp_min(0).to(torch.int64)]
        self._stamp_rows(
            e0,
            e1,
            env_local,
            gx_p,
            gy_p,
            r_sq_p,
            tier,
            slot_ids,
            slot_ids.to(torch.int64),
            st_p,
            act_p,
            z_i32,
        )

    def _stamp_discs(
        self,
        gx: torch.Tensor,
        gy: torch.Tensor,
        r_sq: torch.Tensor,
        active: torch.Tensor,
        subtick: torch.Tensor,
        *,
        slot_order: torch.Tensor | None = None,
        disjoint_slots: bool = False,
    ) -> None:
        """Order-independently stamp body discs into 1x (occ_hi/occ_lo), 4x, and 16x pyramids.

        Uses ``256``-env chunked ``index_select`` + associative/commutative ``torch.maximum``
        + ``1D index_put_(accumulate=True)`` delta updates so every tensor stays strictly
        under ``2^31`` elements and avoids MPS full-grid ``scatter_reduce_`` copies.
        """
        r_sq = r_sq.expand_as(gx)
        active = active.expand_as(gx)
        subtick = subtick.expand_as(gx)
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

        p_cnt = int(gx.shape[2])
        n_cnt = int(gx.shape[1])
        env_block = self.env_block_size

        if disjoint_slots and slot_order is None:
            for e0 in range(0, self.E, env_block):
                e1 = min(self.E, e0 + env_block)
                sids = slot_ids.expand(e1 - e0, n_cnt)
                for p in range(p_cnt):
                    self._stamp_slice_point(
                        e0,
                        e1,
                        gx[e0:e1, :, p],
                        gy[e0:e1, :, p],
                        r_sq[e0:e1, :, p],
                        active[e0:e1, :, p],
                        subtick[e0:e1, :, p],
                        sids,
                    )
        else:
            for e0 in range(0, self.E, env_block):
                e1 = min(self.E, e0 + env_block)
                for col in range(n_cnt):
                    sid_col = slot_ids[:, col : col + 1].expand(e1 - e0, 1)
                    for p in range(p_cnt):
                        self._stamp_slice_point(
                            e0,
                            e1,
                            gx[e0:e1, col : col + 1, p],
                            gy[e0:e1, col : col + 1, p],
                            r_sq[e0:e1, col : col + 1, p],
                            active[e0:e1, col : col + 1, p],
                            subtick[e0:e1, col : col + 1, p],
                            sid_col,
                        )

    def rebase_subticks(self, shift: int | None = None, *, _sync_free_hint: bool = False) -> int:
        """Subtract a constant positive offset from all stored subticks in-place (bitwise-neutral).

        Active ring segments have ``subtick >= oldest_active_subtick >= shift + 1``, so
        subtracting ``shift`` preserves all active relative comparisons ``st >= oldest``
        and owner IDs while keeping packed ``(subtick << 6) | id`` strictly inside
        non-negative signed ``int32`` forever. Any cell with ``st <= shift`` was already
        expired and is reset to ``-1``.

        Executes strictly in-place over ``_REBASE_ENV_CHUNK``-world slices with at most one
        boolean mask temporary per slice (~17.8 MiB peak at ``R=600, N=24`` regardless of ``E``).
        """
        frontier_st = max(2, self.max_pts * self.tick + 2 - self._subtick_offset + self.max_pts + 1)
        if shift is None:
            if _sync_free_hint and self._clean_step_streak > int(self.cfg.max_length) + 4:
                cur_base = self.max_pts * self.tick + 2 - self._subtick_offset
                safe_min = cur_base - self.max_pts * (int(self.cfg.max_length) + 4)
                shift_val = max(0, safe_min - 1)
            else:
                min_active = int(
                    torch.where(self.alive, self.oldest_active_subtick.to(torch.int64), frontier_st)
                    .amin()
                    .item()
                )
                shift_val = max(0, min_active - 1)
        else:
            shift_val = int(shift)
            if shift_val < 0:
                raise ValueError(f"shift must be >= 0, got {shift_val}")
            min_active = int(
                torch.where(self.alive, self.oldest_active_subtick.to(torch.int64), frontier_st)
                .amin()
                .item()
            )
            if shift_val >= min_active:
                raise ValueError(
                    f"shift ({shift_val}) must be < minimum active subtick ({min_active})"
                )

        if shift_val == 0:
            return 0

        int32_max = (1 << 31) - 1
        if shift_val > int32_max:
            raise ValueError(f"shift ({shift_val}) exceeds signed int32 max ({int32_max})")
        new_sentinel = max(2, frontier_st - shift_val)
        if not (2 <= new_sentinel <= int32_max):
            raise ValueError(
                f"new_sentinel ({new_sentinel}) out of signed int32 range [2, {int32_max}] "
                f"(frontier_st={frontier_st}, shift={shift_val})"
            )

        self.ring_subtick.clamp_min_(shift_val)
        self.ring_subtick.sub_(shift_val)

        dead = ~self.alive
        self.oldest_active_subtick.masked_fill_(dead, shift_val)
        self.oldest_active_subtick.sub_(shift_val)
        self.oldest_active_subtick.masked_fill_(dead, new_sentinel)
        del dead

        # 1x grids pack ``(st << OCC_OWNER_BITS) | code`` (0 <= code <= OCC_OWNER_MASK) or -1 in
        # signed int32. For 1 <= shift_val < 2^OCC_SUBTICK_BITS, ``shift_6 = shift_val << OCC_OWNER_BITS``
        # and ``exp_max_1x = shift_6 | OCC_OWNER_MASK`` both fit in signed int32, and
        # ``(g >= 0) & ((g >> OCC_OWNER_BITS) > shift_val)`` holds iff ``g > exp_max_1x``.
        # Filling expired cells (``g <= exp_max_1x``) with ``shift_6 - 1`` before ``sub_(shift_6)``
        # maps expired cells to -1 and active cells to ``((st - shift_val) << OCC_OWNER_BITS) | code``
        # with a single bool mask and zero int32 overflow.
        if shift_val >= (1 << OCC_SUBTICK_BITS):
            self.occ_hi.fill_(-1)
            self.occ_lo.fill_(-1)
        else:
            shift_6 = shift_val << OCC_OWNER_BITS
            exp_max_1x = shift_6 | OCC_OWNER_MASK
            pre_fill_1x = shift_6 - 1
            for e0 in range(0, self.E, _REBASE_ENV_CHUNK):
                e1 = min(self.E, e0 + _REBASE_ENV_CHUNK)
                for grid_1x in (self.occ_hi[e0:e1], self.occ_lo[e0:e1]):
                    mask = grid_1x <= exp_max_1x
                    grid_1x.masked_fill_(mask, pre_fill_1x)
                    del mask
                    grid_1x.sub_(shift_6)

        pre_fill_pyr = shift_val - 1
        for blk in self._occ_4x_blocks:
            b_len = int(blk.shape[0])
            for b0 in range(0, b_len, _REBASE_ENV_CHUNK):
                sub_4x = blk[b0 : min(b_len, b0 + _REBASE_ENV_CHUNK)]
                mask = sub_4x <= shift_val
                sub_4x.masked_fill_(mask, pre_fill_pyr)
                del mask
                sub_4x.sub_(shift_val)

        for e0 in range(0, self.E, _REBASE_ENV_CHUNK):
            sub_16x = self.occ_subtick_16x[e0 : min(self.E, e0 + _REBASE_ENV_CHUNK)]
            mask = sub_16x <= shift_val
            sub_16x.masked_fill_(mask, pre_fill_pyr)
            del mask
            sub_16x.sub_(shift_val)

        self._subtick_offset += shift_val
        return shift_val

    def assert_surviving_discs_disjoint(self) -> None:
        """Debug check: rebuild occupancy from active ring segments and verify <= 1 owner/cell."""
        max_len = int(self.cfg.max_length)
        alive_np = self.alive.detach().cpu().numpy()
        seg_np = self.seg_count.detach().cpu().numpy()
        ptr_np = self.head_ptr.detach().cpu().numpy()
        rc_np = self.ring_cell.detach().cpu().numpy().astype(np.int32)
        rr_np = self.ring_r_sq.detach().cpu().numpy().astype(np.int32)
        dx_np = self._disc_dx.detach().cpu().numpy()
        dy_np = self._disc_dy.detach().cpu().numpy()
        d2_np = self._disc_dist_sq.detach().cpu().numpy()
        got_occ = self.active_occupancy_grid().detach().cpu().numpy()

        for e in range(self.E):
            owners = np.zeros((self.G, self.G), dtype=np.int32)
            expected_occ = np.full((self.G, self.G), -1, dtype=np.int8)
            for i in range(self.N - 1, -1, -1):
                cnt = int(seg_np[e, i])
                if not alive_np[e, i] or cnt <= 0:
                    continue
                idxs = (int(ptr_np[e, i]) - np.arange(cnt, dtype=np.int32)) % max_len
                cells = rc_np[e, i, idxs]
                rsqs = rr_np[e, i, idxs]
                snake_mask = np.zeros((self.G, self.G), dtype=bool)
                for r_val in set(rsqs.tolist()):
                    sc = cells[rsqs == r_val]
                    dm = d2_np <= r_val
                    sx = sc[:, 0:1] + dx_np[None, dm]
                    sy = sc[:, 1:2] + dy_np[None, dm]
                    in_b = (sx >= 0) & (sx < self.G) & (sy >= 0) & (sy < self.G)
                    snake_mask[sy[in_b], sx[in_b]] = True
                owners += snake_mask.astype(np.int32)
                expected_occ[snake_mask] = np.int8(i)
            max_ov = int(owners.max())
            if max_ov > 1:
                yx = tuple(int(v) for v in np.argwhere(owners > 1)[0])
                raise AssertionError(
                    f"Surviving-disc overlap invariant violated in world {e} at tick {self.tick}, "
                    f"cell (gy, gx)={yx}: {max_ov} surviving snakes occupy the same cell"
                )
            if not np.array_equal(got_occ[e], expected_occ):
                yx = tuple(int(v) for v in np.argwhere(got_occ[e] != expected_occ)[0])
                raise AssertionError(
                    f"occ_hi/occ_lo mismatch vs ring segments in world {e} at tick {self.tick}, "
                    f"cell (gy, gx)={yx}: got={got_occ[e][yx]}, expected={expected_occ[yx]}"
                )

    def _spawn_ambient_batch(
        self,
        purpose: int,
        tick_val: torch.Tensor | int,
        batch_offset: int,
        budget: int,
        *,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Deterministically spawn up to ``budget`` ambient pellets per (selected) world."""
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

        pellet_i32 = int(self.cfg.ambient_pellet_value)
        if env_ids is None:
            seeds_col = self.seeds[:, None]
            t_arg: torch.Tensor | int = (
                tick_val[:, None] if isinstance(tick_val, torch.Tensor) else int(tick_val)
            )
            gx = rng_int_torch(seeds_col, t_arg, purpose, slots * 2, self.G)
            gy = rng_int_torch(seeds_col, t_arg, purpose, slots * 2 + 1, self.G)

            in_grid = (gx >= 0) & (gx < self.G) & (gy >= 0) & (gy < self.G)
            in_arena = in_grid & (self._cell_center_sq_fp(gx, gy) < self._food_wall_sq_fp)

            gx_c = gx.clamp(0, self.G - 1).to(torch.int64)
            gy_c = gy.clamp(0, self.G - 1).to(torch.int64)
            flat = gy_c * self.G + gx_c

            occupied = self._is_cell_occupied(flat, self.oldest_active_subtick)
            amb_val = self._gather_1x_grid_chunked(self.ambient_grid, flat)
            cor_val = self._gather_1x_grid_chunked(self.corpse_grid, flat)
            food_free = (amb_val == 0) & (cor_val == 0)

            same = flat[:, :, None] == flat[:, None, :]
            unique = ~(same & tril).any(dim=2)

            cand_valid = in_arena & (~occupied) & food_free & unique
            deficit = (int(self.cfg.target_ambient_pellets) - self.ambient_count).clamp_min(0)
            rank = cand_valid.to(torch.int32).cumsum(dim=1) - 1
            accept = cand_valid & (rank < deficit[:, None])

            self._put_1x_grid_chunked(self.ambient_grid, flat, accept, pellet_i32)

            add_flat = torch.where(accept, pellet_i32, 0).to(torch.int32).reshape(-1)
            flat4 = (gy_c >> 2) * self.G4 + (gx_c >> 2)
            self.ambient_grid_4x.view(-1).index_put_(
                ((self._env_base_4x + flat4).reshape(-1),), add_flat, accumulate=True
            )
            flat16 = (gy_c >> 4) * self.G16 + (gx_c >> 4)
            self.ambient_grid_16x.view(-1).index_put_(
                ((self._env_base_16x + flat16).reshape(-1),), add_flat, accumulate=True
            )

            added_count = accept.to(torch.int32).sum(dim=1, dtype=torch.int32)
            self.ambient_count = self.ambient_count + added_count
            self.total_spawned_ambient_mass = (
                self.total_spawned_ambient_mass + added_count.to(torch.int64) * pellet_i32
            )
            return

        m_envs = int(env_ids.shape[0])
        if m_envs == 0:
            return
        seeds_col = self.seeds.index_select(0, env_ids)[:, None]
        t_arg = (
            tick_val.index_select(0, env_ids)[:, None]
            if isinstance(tick_val, torch.Tensor)
            else int(tick_val)
        )
        gx = rng_int_torch(seeds_col, t_arg, purpose, slots * 2, self.G)
        gy = rng_int_torch(seeds_col, t_arg, purpose, slots * 2 + 1, self.G)

        in_grid = (gx >= 0) & (gx < self.G) & (gy >= 0) & (gy < self.G)
        in_arena = in_grid & (self._cell_center_sq_fp(gx, gy) < self._food_wall_sq_fp)

        gx_c = gx.clamp(0, self.G - 1).to(torch.int64)
        gy_c = gy.clamp(0, self.G - 1).to(torch.int64)
        flat = gy_c * self.G + gx_c

        occupied = self._is_cell_occupied_rows(env_ids, flat, self.oldest_active_subtick)
        amb_sub = self._gather_1x_grid_rows(self.ambient_grid, env_ids, flat)
        cor_sub = self._gather_1x_grid_rows(self.corpse_grid, env_ids, flat)
        food_free = (amb_sub == 0) & (cor_sub == 0)

        same = flat[:, :, None] == flat[:, None, :]
        unique = ~(same & tril).any(dim=2)

        cand_valid = in_arena & (~occupied) & food_free & unique
        deficit = (
            int(self.cfg.target_ambient_pellets) - self.ambient_count.index_select(0, env_ids)
        ).clamp_min(0)
        rank = cand_valid.to(torch.int32).cumsum(dim=1) - 1
        accept = cand_valid & (rank < deficit[:, None])

        full_accept = torch.zeros((self.E, budget), dtype=torch.bool, device=self.device)
        full_flat = torch.zeros((self.E, budget), dtype=torch.int64, device=self.device)
        full_accept.index_copy_(0, env_ids, accept)
        full_flat.index_copy_(0, env_ids, flat)
        self._put_1x_grid_chunked(self.ambient_grid, full_flat, full_accept, pellet_i32)

        add_flat = torch.where(accept, pellet_i32, 0).to(torch.int32).reshape(-1)
        eb4 = env_ids[:, None] * (self.G4 * self.G4)
        flat4 = (gy_c >> 2) * self.G4 + (gx_c >> 2)
        self.ambient_grid_4x.view(-1).index_put_(
            ((eb4 + flat4).reshape(-1),), add_flat, accumulate=True
        )
        eb16 = env_ids[:, None] * (self.G16 * self.G16)
        flat16 = (gy_c >> 4) * self.G16 + (gx_c >> 4)
        self.ambient_grid_16x.view(-1).index_put_(
            ((eb16 + flat16).reshape(-1),), add_flat, accumulate=True
        )

        added_count = accept.to(torch.int32).sum(dim=1, dtype=torch.int32)
        self.ambient_count.index_add_(0, env_ids, added_count)
        self.total_spawned_ambient_mass.index_add_(
            0, env_ids, added_count.to(torch.int64) * pellet_i32
        )

    def _populate_initial_worlds(self, env_ids: torch.Tensor | None, *, init_subtick: int) -> None:
        """Place initial snakes and ambient pellets for all worlds or selected ``env_ids``."""
        m_envs = self.E if env_ids is None else int(env_ids.shape[0])
        if m_envs == 0:
            return

        k_init = INIT_SNAKE_CANDIDATES
        slots = (
            torch.arange(self.N, dtype=torch.int64, device=self.device)[:, None] * k_init
            + torch.arange(k_init, dtype=torch.int64, device=self.device)[None, :]
        )[None, :, :]
        seeds_1d = self.seeds if env_ids is None else self.seeds.index_select(0, env_ids)
        seeds_3d = seeds_1d[:, None, None]
        rng_gx = rng_int_torch(seeds_3d, 0, PURPOSE_INIT_SNAKE, slots * 3, self.G)
        rng_gy = rng_int_torch(seeds_3d, 0, PURPOSE_INIT_SNAKE, slots * 3 + 1, self.G)
        rng_th = rng_angle_idx_torch(
            seeds_3d, 0, PURPOSE_INIT_SNAKE, slots * 3 + 2, self.cfg.angle_steps
        )

        fb_idx = (
            torch.arange(self.N, dtype=torch.int64, device=self.device)[:, None]
            * FALLBACK_SNAKE_STRIDE
            + torch.arange(FALLBACK_RING_SIZE, dtype=torch.int64, device=self.device)[None, :]
        ) & (FALLBACK_RING_SIZE - 1)
        fb_sel = self._fallback_spawns[fb_idx][None, :, :, :].expand(
            m_envs, self.N, FALLBACK_RING_SIZE, 3
        )
        cand_gx = torch.cat((rng_gx, fb_sel[..., 0]), dim=2)
        cand_gy = torch.cat((rng_gy, fb_sel[..., 1]), dim=2)
        cand_th = self._orient_spawn_heading_torch(
            cand_gx, cand_gy, torch.cat((rng_th, fb_sel[..., 2]), dim=2)
        )

        in_wall = (
            (cand_gx >= 0)
            & (cand_gx < self.G)
            & (cand_gy >= 0)
            & (cand_gy < self.G)
            & (self._cell_center_sq_fp(cand_gx, cand_gy) < self._spawn_wall_sq_fp)
        )

        placed_gx = torch.zeros((m_envs, self.N), dtype=torch.int32, device=self.device)
        placed_gy = torch.zeros((m_envs, self.N), dtype=torch.int32, device=self.device)
        placed_th = torch.zeros((m_envs, self.N), dtype=torch.int32, device=self.device)
        placed_alive = torch.zeros((m_envs, self.N), dtype=torch.bool, device=self.device)
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

        h_pos = self._cell_center_fp(placed_gx, placed_gy)
        len_init = torch.where(
            placed_alive,
            torch.full(
                (m_envs, self.N), int(self.cfg.start_length), dtype=torch.int32, device=self.device
            ),
            torch.zeros((m_envs, self.N), dtype=torch.int32, device=self.device),
        )
        seg_init = torch.where(
            placed_alive,
            torch.ones((m_envs, self.N), dtype=torch.int32, device=self.device),
            torch.zeros((m_envs, self.N), dtype=torch.int32, device=self.device),
        )
        rt_init = torch.where(
            placed_alive,
            torch.zeros((m_envs, self.N), dtype=torch.int32, device=self.device),
            torch.full(
                (m_envs, self.N), int(self.cfg.respawn_delay), dtype=torch.int32, device=self.device
            ),
        )
        old_st_init = torch.where(
            placed_alive,
            torch.full((m_envs, self.N), init_subtick, dtype=torch.int32, device=self.device),
            torch.full((m_envs, self.N), init_subtick + 1, dtype=torch.int32, device=self.device),
        )
        init_r_sq = self._r_sq_table[int(self.cfg.start_length)].expand(m_envs, self.N)
        ring_st0 = torch.where(
            placed_alive,
            torch.full((m_envs, self.N), init_subtick, dtype=torch.int32, device=self.device),
            torch.zeros((m_envs, self.N), dtype=torch.int32, device=self.device),
        )
        spawn_mass = placed_alive.to(torch.int64).sum(dim=1) * int(self.cfg.start_length)

        if env_ids is None:
            self.head_pos.copy_(h_pos)
            self.theta.copy_(placed_th)
            self.alive.copy_(placed_alive)
            self.life_start.copy_(placed_alive)
            self.length.copy_(len_init)
            self.seg_count.copy_(seg_init)
            self.respawn_timer.copy_(rt_init)
            self.oldest_active_subtick.copy_(old_st_init)

            self.ring_pos[:, :, 0, :] = h_pos
            self.ring_cell[:, :, 0, 0] = placed_gx.to(torch.int16)
            self.ring_cell[:, :, 0, 1] = placed_gy.to(torch.int16)
            self.ring_r_sq[:, :, 0] = init_r_sq.to(torch.int16)
            self.ring_subtick[:, :, 0] = ring_st0

            self._stamp_discs(
                placed_gx[:, :, None],
                placed_gy[:, :, None],
                init_r_sq[:, :, None],
                placed_alive[:, :, None],
                torch.full(
                    (self.E, self.N, 1), init_subtick, dtype=torch.int32, device=self.device
                ),
                disjoint_slots=True,
            )
            self.total_spawned_snake_mass.copy_(spawn_mass)

            self.prev_alive.fill_(False)
            self.prev_boosting.fill_(False)
            self.prev_head_pos.copy_(h_pos)
            self.prev_theta.copy_(placed_th)
            self.prev_length.copy_(len_init)
        else:
            self.head_pos.index_copy_(0, env_ids, h_pos)
            self.theta.index_copy_(0, env_ids, placed_th)
            self.alive.index_copy_(0, env_ids, placed_alive)
            self.life_start.index_copy_(0, env_ids, placed_alive)
            self.length.index_copy_(0, env_ids, len_init)
            self.seg_count.index_copy_(0, env_ids, seg_init)
            self.respawn_timer.index_copy_(0, env_ids, rt_init)
            self.oldest_active_subtick.index_copy_(0, env_ids, old_st_init)

            self.ring_pos[env_ids, :, 0, :] = h_pos
            self.ring_cell[env_ids, :, 0, 0] = placed_gx.to(torch.int16)
            self.ring_cell[env_ids, :, 0, 1] = placed_gy.to(torch.int16)
            self.ring_r_sq[env_ids, :, 0] = init_r_sq.to(torch.int16)
            self.ring_subtick[env_ids, :, 0] = ring_st0

            full_gx = torch.zeros((self.E, self.N, 1), dtype=torch.int32, device=self.device)
            full_gy = torch.zeros((self.E, self.N, 1), dtype=torch.int32, device=self.device)
            full_act = torch.zeros((self.E, self.N, 1), dtype=torch.bool, device=self.device)
            full_gx.index_copy_(0, env_ids, placed_gx[:, :, None])
            full_gy.index_copy_(0, env_ids, placed_gy[:, :, None])
            full_act.index_copy_(0, env_ids, placed_alive[:, :, None])
            full_rsq = self._r_sq_table[int(self.cfg.start_length)].expand(self.E, self.N, 1)
            self._stamp_discs(
                full_gx,
                full_gy,
                full_rsq,
                full_act,
                torch.full(
                    (self.E, self.N, 1), init_subtick, dtype=torch.int32, device=self.device
                ),
                disjoint_slots=True,
            )
            self.total_spawned_snake_mass.index_copy_(0, env_ids, spawn_mass)

            self.prev_alive.index_fill_(0, env_ids, False)
            self.prev_boosting.index_fill_(0, env_ids, False)
            self.prev_head_pos.index_copy_(0, env_ids, h_pos)
            self.prev_theta.index_copy_(0, env_ids, placed_th)
            self.prev_length.index_copy_(0, env_ids, len_init)

        num_batches = init_food_num_batches(self.cfg)
        for b in range(num_batches):
            self._spawn_ambient_batch(
                PURPOSE_INIT_FOOD,
                0,
                b * INIT_FOOD_BATCH_SIZE,
                INIT_FOOD_BATCH_SIZE,
                env_ids=env_ids,
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
        self._subtick_offset = 0
        self._clean_step_streak = 0
        self._clear_all_worlds()

        self._populate_initial_worlds(None, init_subtick=1)
        if self.debug_assert_invariants:
            self.assert_surviving_discs_disjoint()

    def _clear_all_worlds(self) -> None:
        """Reset all clearable registered state tensors across all worlds in-place."""
        for field in _CLEARABLE_STATE_FIELDS:
            if field.sharded_4x:
                for blk in self._occ_4x_blocks:
                    blk.fill_(field.reset_value)
            elif field.reset_value == 0 and field.dtype != torch.bool:
                getattr(self, field.name).zero_()
            else:
                getattr(self, field.name).fill_(field.reset_value)

    def _clear_selected_worlds(self, env_ids: torch.Tensor) -> None:
        """Reset all clearable registered state tensors on ``env_ids`` worlds in-place."""
        for field in _CLEARABLE_STATE_FIELDS:
            if field.sharded_4x:
                for b_idx, blk in enumerate(self._occ_4x_blocks):
                    e0 = b_idx * self.env_block_size
                    e1 = e0 + int(blk.shape[0])
                    in_blk = env_ids[(env_ids >= e0) & (env_ids < e1)] - e0
                    if in_blk.numel() > 0:
                        blk.index_fill_(0, in_blk, field.reset_value)
            else:
                getattr(self, field.name).index_fill_(0, env_ids, field.reset_value)

    def reset_worlds(
        self,
        world_mask: torch.Tensor,
        *,
        seeds: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Reset only worlds selected by boolean ``world_mask`` ``(E,)`` cleanly."""
        if world_mask.dtype != torch.bool or world_mask.shape != (self.E,):
            raise ValueError(
                f"world_mask must be bool of shape ({self.E},), "
                f"got {world_mask.dtype} {tuple(world_mask.shape)}"
            )
        if world_mask.device != self.device:
            raise ValueError(f"world_mask device {world_mask.device} != sim device {self.device}")

        env_ids = torch.where(world_mask)[0]
        m_envs = int(env_ids.shape[0])

        if seeds is not None:
            if isinstance(seeds, torch.Tensor):
                seed_t = seeds.to(device=self.device, dtype=torch.int64)
            else:
                seed_t = torch.as_tensor(list(seeds), dtype=torch.int64, device=self.device)
            if seed_t.shape == (self.E,):
                self.seeds.copy_(torch.where(world_mask, seed_t, self.seeds))
            elif seed_t.shape == (m_envs,):
                if m_envs > 0:
                    self.seeds.index_copy_(0, env_ids, seed_t)
            else:
                raise ValueError(
                    f"seeds for reset_worlds must have shape ({self.E},) or ({m_envs},), "
                    f"got {tuple(seed_t.shape)}"
                )

        if m_envs == 0:
            return

        init_st = max(1, self.max_pts * self.tick + 1 - self._subtick_offset)
        if self.auto_rebase and init_st + 1 >= AUTO_REBASE_SUBTICK_THRESHOLD:
            self.rebase_subticks()
            init_st = max(1, self.max_pts * self.tick + 1 - self._subtick_offset)
        if init_st + 1 > MAX_OCC_SUBTICK:
            raise OverflowError(
                f"reset_worlds subtick {init_st + 1} exceeds int32 limit {MAX_OCC_SUBTICK}"
            )

        self._clear_selected_worlds(env_ids)

        self._populate_initial_worlds(env_ids, init_subtick=init_st)
        if self.debug_assert_invariants:
            self.assert_surviving_discs_disjoint()

    def step(
        self,
        actions: torch.Tensor,
        *,
        truncate: bool | torch.Tensor = False,
    ) -> StepInfo:
        """Advance all E worlds by one tick.

        Exact within-tick order (Rule 9):
          0. Auto-rebase subticks if approaching ``AUTO_REBASE_SUBTICK_THRESHOLD`` and
             verify on the host (0 sync) that packed ``int32`` occupancy cannot overflow;
             retire any snakes flagged ``truncated_pending`` on the previous tick (when
             ``cfg.max_life_ticks > 0``).
          1. Move all alive snakes in fixed-point and emit distance-spaced body points
             (one point per ``speed`` unit travelled; boost shedding & fractional drop).
          2. Test every alive snake's sub-step head disc(s) against walls, all other
             snakes' retained pre-move bodies, and causally ordered new points
             (``s_j <= s_i``: head hitting body/head dies; simultaneous head-on kills
             both; boosting ahead does not die when a follower hits its trailing point).
          3. Apply all deaths at once; award multi-hit kill credit by a counter-based
             RNG draw over all enemy hitters; stamp surviving snakes only (preserving the
             ``<= 1`` active owner per cell invariant).
          4. Resolve food eating on integer squared pickup distances (contested pellet
             goes to lowest surviving slot via ``O(N K log(N K))`` key sort); drop boost
             pellets and corpses AFTER deaths (saturating ``int16`` corpse cells at
             ``32767``); refill ambient food.
          5. Advance respawn timers and evaluate respawns ONLY for ready slots using
             world-local ``world_ticks``, orienting spawn headings away from the wall;
             emit episode hooks (including non-destructive rollout ``truncate``).
        """
        if actions.shape != (self.E, self.N):
            raise ValueError(
                f"actions must have shape ({self.E}, {self.N}), got {tuple(actions.shape)}"
            )
        if actions.device != self.device:
            raise ValueError(f"actions device {actions.device} != sim device {self.device}")

        if isinstance(truncate, bool):
            trunc_mask: torch.Tensor | None = None
        else:
            if truncate.dtype != torch.bool:
                raise ValueError(f"truncate tensor must be bool, got {truncate.dtype}")
            if truncate.device != self.device:
                raise ValueError(f"truncate device {truncate.device} != sim device {self.device}")
            if truncate.shape == (self.E,):
                trunc_mask = truncate[:, None].expand(self.E, self.N)
            elif truncate.shape == (self.E, self.N):
                trunc_mask = truncate
            else:
                raise ValueError(
                    f"truncate must have shape ({self.E},) or ({self.E}, {self.N}), "
                    f"got {tuple(truncate.shape)}"
                )

        # Host-side subtick overflow guard & automatic bitwise-neutral rebase (0 sync).
        next_tick = self.tick + 1
        prospective_base = self.max_pts * next_tick + 2 - self._subtick_offset
        prospective_max = prospective_base + self.max_pts + 1
        if self.auto_rebase and prospective_max >= AUTO_REBASE_SUBTICK_THRESHOLD:
            self.rebase_subticks(_sync_free_hint=True)
            prospective_base = self.max_pts * next_tick + 2 - self._subtick_offset
            prospective_max = prospective_base + self.max_pts + 1
        if prospective_max > MAX_OCC_SUBTICK:
            raise OverflowError(
                f"Occupancy subtick {prospective_max} at tick {next_tick} would overflow "
                f"packed int32 limit {MAX_OCC_SUBTICK}; call rebase_subticks() or enable "
                f"auto_rebase=True."
            )

        timed = self._timing_enabled
        if timed:
            self._sync_device()
            t0 = time.perf_counter()

        self.tick = next_tick
        self.world_ticks = self.world_ticks + 1
        self._clean_step_streak += 1
        tick_base_subtick = prospective_base
        expired_sentinel_st = tick_base_subtick + self.max_pts + 1

        # Retire snakes truncated by max_life_ticks at the end of the previous tick (no corpse).
        retire = self.truncated_pending & self.alive
        retire_mass = torch.where(retire, self.length, 0).to(torch.int64).sum(dim=1)
        self.total_truncation_sink = self.total_truncation_sink + retire_mass
        self.alive = self.alive & (~retire)
        self.length = torch.where(retire, 0, self.length)
        self.seg_count = torch.where(retire, 0, self.seg_count)
        self.oldest_active_subtick = torch.where(
            retire, expired_sentinel_st, self.oldest_active_subtick
        )
        for field in _PER_LIFE_STATE_FIELDS:
            cur = getattr(self, field.name)
            if field.dtype == torch.bool and field.reset_value is False:
                setattr(self, field.name, cur & (~retire))
            else:
                setattr(self, field.name, torch.where(retire, field.reset_value, cur))
        # Set timer=1 so Stage 5 decrements to 0 and immediately spawns a fresh life this tick.
        self.respawn_timer = torch.where(retire, 1, self.respawn_timer)
        self.truncated_pending.fill_(False)

        # Snapshot pre-step head pose and pre-tick length (StepInfo.prev_length).
        for field in _PREV_SNAPSHOT_FIELDS:
            assert field.prev_source is not None
            getattr(self, field.name).copy_(getattr(self, field.prev_source))

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

        # ---------------- Stage 2: causal sub-step head collision & RNG kill credit ------------
        len_idx = self.length.clamp(0, max_len).to(torch.int64)
        r_sq = self._r_sq_table[len_idx]
        r_tier = self._r_tier_table[len_idx]
        wall_lim_sq = self._wall_limit_sq_fp_table[len_idx]

        in_grid_pt = (pt_gx >= 0) & (pt_gx < self.G) & (pt_gy >= 0) & (pt_gy < self.G)
        pt_d_sq_fp = pt_x.to(torch.int64).square() + pt_y.to(torch.int64).square()
        died_wall = self.alive & (
            pt_valid & ((~in_grid_pt) | (pt_d_sq_fp >= wall_lim_sq[:, :, None]))
        ).any(dim=-1)

        # Causal sub-step head collision: emitted point k of snake i occurs at sub-step
        # s(i, k) = max_pts - num_pts[i] + k (0 = mid-tick boost point, 1 = end-of-tick head).
        # Sub-step head (i, ki) dies on enemy point (j, kj) iff s(j, kj) <= s(i, ki):
        # - Head hitting an earlier mid-tick boost body segment (s_j=0 < s_i=1) kills ONLY i.
        # - Simultaneous sub-step heads meeting (s_j == s_i) kills BOTH i and j (head-on rule).
        ti = r_tier[:, :, None].expand(-1, -1, self.N)
        tj = r_tier[:, None, :].expand(-1, self.N, -1)
        base_substep = self.max_pts - num_pts
        hit_post_pair = torch.zeros((self.E, self.N, self.N), dtype=torch.bool, device=self.device)
        for ki in range(self.max_pts):
            vi = pt_valid[:, :, ki]
            gxi = pt_gx[:, :, ki]
            gyi = pt_gy[:, :, ki]
            si = base_substep + ki
            for kj in range(self.max_pts):
                vj = pt_valid[:, :, kj]
                sj = base_substep + kj
                dgx = pt_gx[:, None, :, kj] - gxi[:, :, None]
                dgy = pt_gy[:, None, :, kj] - gyi[:, :, None]
                near = (
                    vi[:, :, None]
                    & vj[:, None, :]
                    & self._not_eye_nn
                    & (sj[:, None, :] <= si[:, :, None])
                    & (dgx >= -self._max_diam)
                    & (dgx <= self._max_diam)
                    & (dgy >= -self._max_diam)
                    & (dgy <= self._max_diam)
                )
                dgx_idx = (dgx + self._max_diam).clamp(0, 2 * self._max_diam).to(torch.int64)
                dgy_idx = (dgy + self._max_diam).clamp(0, 2 * self._max_diam).to(torch.int64)
                hit_post_pair = hit_post_pair | (near & self._overlap_lut[ti, tj, dgx_idx, dgy_idx])

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

        occ_hi_k = self._gather_1x_grid_chunked(self.occ_hi, flat_k).view_as(hx_k)
        ow_hi_k = decode_occ_owner_hi(occ_hi_k)
        ow_hi_i64 = ow_hi_k.clamp(0, self.N - 1).to(torch.int64)
        st_tail_hi = pre_death_oldest_st.gather(1, ow_hi_i64.view(self.E, -1)).view_as(hx_k)
        hit_hi = (
            valid_k
            & occ_entry_active(occ_hi_k, st_tail_hi)
            & (ow_hi_k != self._snake_ids[:, :, None, None])
        )

        occ_lo_k = self._gather_1x_grid_chunked(self.occ_lo, flat_k).view_as(hx_k)
        ow_lo_k = decode_occ_owner_lo(occ_lo_k)
        ow_lo_i64 = ow_lo_k.clamp(0, self.N - 1).to(torch.int64)
        st_tail_lo = pre_death_oldest_st.gather(1, ow_lo_i64.view(self.E, -1)).view_as(hx_k)
        hit_lo = (
            valid_k
            & occ_entry_active(occ_lo_k, st_tail_lo)
            & (ow_lo_k != self._snake_ids[:, :, None, None])
        )

        en_cnt = self.E * self.N
        en_base = torch.arange(en_cnt, dtype=torch.int64, device=self.device)[:, None] * self.N
        ret_1d = torch.zeros((en_cnt * self.N,), dtype=torch.int32, device=self.device)
        ret_1d.index_put_(
            ((en_base + ow_hi_i64.view(en_cnt, -1)).reshape(-1),),
            hit_hi.reshape(-1).to(torch.int32),
            accumulate=True,
        )
        ret_1d.index_put_(
            ((en_base + ow_lo_i64.view(en_cnt, -1)).reshape(-1),),
            hit_lo.reshape(-1).to(torch.int32),
            accumulate=True,
        )
        ret_hit_pair = ret_1d.view(self.E, self.N, self.N) > 0

        hit_enemy_pair = self.alive[:, :, None] & self._not_eye_nn & (hit_post_pair | ret_hit_pair)
        hitter_cnt = hit_enemy_pair.to(torch.int32).sum(dim=-1, dtype=torch.int32)
        died_enemy = self.alive & (hitter_cnt > 0)
        died = self.alive & (died_wall | died_enemy)
        survives = self.alive & (~died)

        # Deterministic counter-RNG draw among enemy hitters (mirrors ReferenceWorld).
        pick_rank = rng_int_torch(
            self.seeds[:, None],
            self.world_ticks[:, None],
            PURPOSE_KILL_CREDIT,
            self._snake_ids_i64,
            hitter_cnt.clamp_min(1),
        )
        hitter_rank = hit_enemy_pair.to(torch.int32).cumsum(dim=-1) - 1
        chosen_killer = (
            (hit_enemy_pair & (hitter_rank == pick_rank[:, :, None])).to(torch.int32)
            * self._snake_ids[:, None, :]
        ).sum(dim=-1, dtype=torch.int32)
        killer_id = torch.where(died_enemy, chosen_killer, torch.full_like(chosen_killer, -1))
        kill_credit = (
            (killer_id[:, :, None] == self._snake_ids[:, None, :])
            .to(torch.int32)
            .sum(dim=1, dtype=torch.int32)
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
        for field in _DEATH_ONLY_PER_LIFE_FIELDS:
            cur = getattr(self, field.name)
            fill = (
                torch.zeros_like(cur)
                if (field.reset_value == 0 or field.reset_value is False)
                else torch.full_like(cur, field.reset_value)
            )
            setattr(self, field.name, torch.where(died, fill, cur))

        # Stamp ONLY surviving snakes: because Stage 2 killed at least one snake in every
        # overlapping pair (s_j <= s_i is a total order), surviving discs are pairwise disjoint.
        self._stamp_discs(
            pt_gx,
            pt_gy,
            r_sq[:, :, None].expand_as(pt_gx),
            surv_pt_valid,
            pt_subtick,
            disjoint_slots=True,
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
        dx_eat = (cx_fp - self.head_pos[:, :, 0, None]).clamp(
            -FOOD_PICKUP_CLAMP_FP, FOOD_PICKUP_CLAMP_FP
        )
        dy_eat = (cy_fp - self.head_pos[:, :, 1, None]).clamp(
            -FOOD_PICKUP_CLAMP_FP, FOOD_PICKUP_CLAMP_FP
        )
        pickup_r_sq = self._pickup_r_sq_fp_table[len_idx]
        can_eat_k = (
            survives[:, :, None]
            & eat_in_b
            & ((dx_eat * dx_eat + dy_eat * dy_eat) <= pickup_r_sq[:, :, None])
        )

        ex_c = ex_k.clamp(0, self.G - 1).to(torch.int64)
        ey_c = ey_k.clamp(0, self.G - 1).to(torch.int64)
        flat_eat_3d = ey_c * self.G + ex_c
        flat_eat = flat_eat_3d.view(self.E, -1)

        raw_amb = self._gather_1x_grid_chunked(self.ambient_grid, flat_eat).to(torch.int32)
        raw_cor = self._gather_1x_grid_chunked(self.corpse_grid, flat_eat).to(torch.int32)
        can_eat_food_k = can_eat_k & ((raw_amb > 0) | (raw_cor > 0)).view_as(can_eat_k)

        # O(E * N * K_eat * log(N * K_eat)) lowest-slot tie-break: all active keys
        # ``(cell_flat << OCC_OWNER_BITS) | slot_id`` in a world are strictly distinct, so
        # sorting places lowest slot first for every contested food cell with zero host syncs.
        if self.G * self.G < (1 << OCC_SUBTICK_BITS):
            sentinel_key = 0x7FFFFFFF
            eat_key = torch.where(
                can_eat_food_k,
                (flat_eat_3d.to(torch.int32) << OCC_OWNER_BITS) | self._snake_ids[:, :, None],
                sentinel_key,
            ).view(self.E, -1)
        else:
            sentinel_key = 0x7FFFFFFFFFFFFFFF
            eat_key = torch.where(
                can_eat_food_k,
                (flat_eat_3d << OCC_OWNER_BITS) | self._snake_ids_i64[:, :, None],
                sentinel_key,
            ).view(self.E, -1)
        sk, eat_perm = torch.sort(eat_key, dim=1)
        won_sorted = sk != sentinel_key
        won_sorted[:, 1:] = won_sorted[:, 1:] & (
            (sk[:, 1:] >> OCC_OWNER_BITS) != (sk[:, :-1] >> OCC_OWNER_BITS)
        )
        won_flat = torch.zeros_like(won_sorted)
        won_flat.scatter_(1, eat_perm, won_sorted)

        eaten_amb = torch.where(won_flat, raw_amb, 0)
        eaten_cor = torch.where(won_flat, raw_cor, 0)

        self._put_1x_grid_chunked(self.ambient_grid, flat_eat, won_flat, 0)
        self._put_1x_grid_chunked(self.corpse_grid, flat_eat, won_flat, 0)

        neg_amb_1d = (-eaten_amb).reshape(-1)
        flat4_eat = ((ey_c >> 2) * self.G4 + (ex_c >> 2)).view(self.E, -1)
        self.ambient_grid_4x.view(-1).index_put_(
            ((self._env_base_4x + flat4_eat).reshape(-1),), neg_amb_1d, accumulate=True
        )
        flat16_eat = ((ey_c >> 4) * self.G16 + (ex_c >> 4)).view(self.E, -1)
        self.ambient_grid_16x.view(-1).index_put_(
            ((self._env_base_16x + flat16_eat).reshape(-1),), neg_amb_1d, accumulate=True
        )

        eaten_amb_count = (eaten_amb > 0).to(torch.int32).sum(dim=1, dtype=torch.int32)
        self.ambient_count = self.ambient_count - eaten_amb_count

        mass_eaten = (eaten_amb.view_as(ex_k) + eaten_cor.view_as(ex_k)).sum(
            dim=-1, dtype=torch.int32
        )
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
        shed_val = boost_drop_int.to(torch.int32)

        # 4c. Drop corpses AFTER deaths and eating; saturate 1x int16 cells at CORPSE_GRID_MAX.
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

        dep_flat = torch.cat((tail_flat, orb_flat), dim=1)
        dep_val = torch.cat((shed_val, orb_val), dim=1)
        if self.G * self.G < (1 << 31):
            sort_cell_raw, dep_perm = torch.sort(dep_flat.to(torch.int32), dim=1)
            sort_cell = sort_cell_raw.to(torch.int64)
        else:
            sort_cell, dep_perm = torch.sort(dep_flat, dim=1)
        sort_add = dep_val.gather(1, dep_perm).to(torch.int64)
        pref = sort_add.cumsum(dim=1)
        is_first = torch.ones_like(sort_cell, dtype=torch.bool)
        is_first[:, 1:] = sort_cell[:, 1:] != sort_cell[:, :-1]
        prev_pref = torch.zeros_like(pref)
        prev_pref[:, 1:] = pref[:, :-1]
        group_base = torch.cummax(
            torch.where(is_first, prev_pref, torch.zeros_like(pref)), dim=1
        ).values
        is_last = torch.ones_like(sort_cell, dtype=torch.bool)
        is_last[:, :-1] = sort_cell[:, :-1] != sort_cell[:, 1:]
        group_sum = pref - group_base
        write_dep = is_last & (group_sum > 0)

        cur_cor = self._gather_1x_grid_chunked(self.corpse_grid, sort_cell).to(torch.int64)
        raw_new = cur_cor + group_sum
        clamped_new = raw_new.clamp_max(CORPSE_GRID_MAX)
        sat_overflow = torch.where(write_dep, raw_new - clamped_new, 0)
        self.total_corpse_decay_sink = self.total_corpse_decay_sink + sat_overflow.sum(dim=1)
        eff_add = torch.where(write_dep, (clamped_new - cur_cor).to(torch.int32), 0)

        self._put_1x_grid_chunked(
            self.corpse_grid, sort_cell, write_dep, clamped_new.to(torch.int16)
        )

        dep_gy = torch.div(sort_cell, self.G, rounding_mode="floor")
        dep_gx = sort_cell - dep_gy * self.G
        dep_flat4 = (dep_gy >> 2) * self.G4 + (dep_gx >> 2)
        dep_flat16 = (dep_gy >> 4) * self.G16 + (dep_gx >> 4)

        cor_delta_4x16x = torch.cat((-eaten_cor, eff_add), dim=1).reshape(-1)
        cor_idx_4x = (self._env_base_4x + torch.cat((flat4_eat, dep_flat4), dim=1)).reshape(-1)
        cor_idx_16x = (self._env_base_16x + torch.cat((flat16_eat, dep_flat16), dim=1)).reshape(-1)
        self.corpse_grid_4x.view(-1).index_put_((cor_idx_4x,), cor_delta_4x16x, accumulate=True)
        self.corpse_grid_16x.view(-1).index_put_((cor_idx_16x,), cor_delta_4x16x, accumulate=True)

        # 4d. Refill ambient pellets toward target density using per-world world_ticks.
        self._spawn_ambient_batch(
            PURPOSE_AMBIENT_FOOD,
            self.world_ticks,
            0,
            self.cfg.ambient_spawn_budget_per_tick,
        )

        if timed:
            self._sync_device()
            t1 = time.perf_counter()
            self._stage_seconds["food"] += t1 - t0
            t0 = t1

        # ---------------- Stage 5: ready-only respawn & episode hooks ----------------
        was_dead_before = (~self.alive) & (~died)
        self.respawn_timer = torch.where(
            was_dead_before,
            (self.respawn_timer - 1).clamp_min(0),
            self.respawn_timer,
        )
        ready = was_dead_before & (self.respawn_timer == 0)
        # Intentional 1-count device-to-host sync via ``torch.where(ready)``:
        #   - Capping to a fixed ``B < E * N`` without reading ``r_cnt`` would drop slots when
        #     simultaneous deaths mature on the same tick (up to ``E * N``), breaking parity.
        #   - Unconditional dense ``(E, N)`` evaluation (``B = E * N``) avoids the sync cleanly
        #     only by paying 81-cell 1x clearance gathers across every slot every tick (~24.5 ms
        #     and >500 MiB temp vs ~1.1 ms on MPS at E=256, N=64; 51.2 ms vs 0.04 ms on CPU).
        raw_e, raw_i = torch.where(ready)
        r_cnt = int(raw_e.shape[0])

        spawn_now = torch.zeros((self.E, self.N), dtype=torch.bool, device=self.device)
        if r_cnt > 0:
            if self.device.type == "cpu":
                b_cnt = r_cnt
                ready_e = raw_e
                ready_i = raw_i
                valid_b = torch.ones((b_cnt,), dtype=torch.bool, device=self.device)
            else:
                # Bucket to power-of-two row counts (>= 16) so MPSGraph / CUDA caches static shapes
                # instead of recompiling kernels whenever ready count fluctuates tick-to-tick.
                b_cnt = 16
                while b_cnt < r_cnt:
                    b_cnt <<= 2
                b_cnt = min(b_cnt, max(r_cnt, self.E * self.N))
                if b_cnt == r_cnt:
                    ready_e = raw_e
                    ready_i = raw_i
                    valid_b = torch.ones((b_cnt,), dtype=torch.bool, device=self.device)
                else:
                    pad = b_cnt - r_cnt
                    z_pad = torch.zeros((pad,), dtype=torch.int64, device=self.device)
                    ready_e = torch.cat((raw_e, z_pad), dim=0)
                    ready_i = torch.cat((raw_i, z_pad), dim=0)
                    valid_b = torch.arange(b_cnt, dtype=torch.int64, device=self.device) < r_cnt

            r_seeds = self.seeds.index_select(0, ready_e)[:, None]
            r_ticks = self.world_ticks.index_select(0, ready_e)[:, None]
            k_resp = int(self.cfg.respawn_candidates)
            k_offs = torch.arange(k_resp, dtype=torch.int64, device=self.device)[None, :]
            r_slots = ready_i[:, None] * k_resp + k_offs

            rgx = rng_int_torch(r_seeds, r_ticks, PURPOSE_RESPAWN, r_slots * 3, self.G)
            rgy = rng_int_torch(r_seeds, r_ticks, PURPOSE_RESPAWN, r_slots * 3 + 1, self.G)
            rth = rng_angle_idx_torch(
                r_seeds, r_ticks, PURPOSE_RESPAWN, r_slots * 3 + 2, self.cfg.angle_steps
            )

            fb_ring_idx = (
                ready_i[:, None] * FALLBACK_SNAKE_STRIDE
                + r_ticks * FALLBACK_TICK_STRIDE
                + self._fb_f_offsets.reshape(1, -1)
            ) & (FALLBACK_RING_SIZE - 1)
            fb_cands = self._fallback_spawns[fb_ring_idx]
            all_gx = torch.cat((rgx, fb_cands[..., 0]), dim=1)
            all_gy = torch.cat((rgy, fb_cands[..., 1]), dim=1)
            all_th_raw = torch.cat((rth, fb_cands[..., 2]), dim=1)

            r_in_wall = (
                valid_b[:, None]
                & (all_gx >= 0)
                & (all_gx < self.G)
                & (all_gy >= 0)
                & (all_gy < self.G)
                & (self._cell_center_sq_fp(all_gx, all_gy) < self._spawn_wall_sq_fp)
            )

            cx_r = all_gx[..., None] + self._clear_dx[None, None, :]
            cy_r = all_gy[..., None] + self._clear_dy[None, None, :]
            c_in_grid = (cx_r >= 0) & (cx_r < self.G) & (cy_r >= 0) & (cy_r < self.G)
            c_flat = cy_r.clamp(0, self.G - 1).to(torch.int64) * self.G + cx_r.clamp(
                0, self.G - 1
            ).to(torch.int64)
            c_busy = (~c_in_grid) | self._is_cell_occupied_rows(
                ready_e, c_flat, self.oldest_active_subtick
            )

            cand_ok_r = r_in_wall & (~c_busy.any(dim=-1))
            has_cand_r = valid_b & cand_ok_r.any(dim=-1)
            best_m_r = cand_ok_r.to(torch.int32).argmax(dim=-1, keepdim=True)
            sel_gx_r = torch.where(valid_b, all_gx.gather(1, best_m_r).squeeze(1), 0)
            sel_gy_r = torch.where(valid_b, all_gy.gather(1, best_m_r).squeeze(1), 0)
            sel_th_r = torch.where(
                valid_b,
                self._orient_spawn_heading_torch(
                    sel_gx_r, sel_gy_r, all_th_raw.gather(1, best_m_r).squeeze(1)
                ),
                0,
            )

            ready_lin = torch.where(valid_b, ready_e * self.N + ready_i, 0)
            sel_gx = torch.zeros((self.E * self.N,), dtype=torch.int32, device=self.device)
            sel_gy = torch.zeros((self.E * self.N,), dtype=torch.int32, device=self.device)
            sel_th = torch.zeros((self.E * self.N,), dtype=torch.int32, device=self.device)
            sel_gx.index_put_((ready_lin,), sel_gx_r, accumulate=True)
            sel_gy.index_put_((ready_lin,), sel_gy_r, accumulate=True)
            sel_th.index_put_((ready_lin,), sel_th_r, accumulate=True)
            sel_gx = sel_gx.view(self.E, self.N)
            sel_gy = sel_gy.view(self.E, self.N)
            sel_th = sel_th.view(self.E, self.N)

            min_sep_sq = (2 * self._r_clear) * (2 * self._r_clear)
            if b_cnt <= 256:
                rdx_r = sel_gx_r[:, None] - sel_gx_r[None, :]
                rdy_r = sel_gy_r[:, None] - sel_gy_r[None, :]
                conflict_r = (
                    valid_b[:, None]
                    & valid_b[None, :]
                    & (ready_e[:, None] == ready_e[None, :])
                    & (ready_i[:, None] > ready_i[None, :])
                    & has_cand_r[None, :]
                    & ((rdx_r * rdx_r + rdy_r * rdy_r) <= min_sep_sq)
                ).any(dim=-1)
                spawn_r = has_cand_r & (~conflict_r)
                spawn_i32 = torch.zeros((self.E * self.N,), dtype=torch.int32, device=self.device)
                spawn_i32.index_put_((ready_lin,), spawn_r.to(torch.int32), accumulate=True)
                spawn_now = (spawn_i32 > 0).view(self.E, self.N)
            else:
                hc_i32 = torch.zeros((self.E * self.N,), dtype=torch.int32, device=self.device)
                hc_i32.index_put_((ready_lin,), has_cand_r.to(torch.int32), accumulate=True)
                has_cand = (hc_i32 > 0).view(self.E, self.N)
                rdx = sel_gx[:, :, None] - sel_gx[:, None, :]
                rdy = sel_gy[:, :, None] - sel_gy[:, None, :]
                earlier_conflict = (
                    self._tril_nn & has_cand[:, None, :] & ((rdx * rdx + rdy * rdy) <= min_sep_sq)
                ).any(dim=-1)
                spawn_now = has_cand & (~earlier_conflict)
                spawn_r = valid_b & spawn_now.view(-1).index_select(0, ready_lin)

            spawn_pos = self._cell_center_fp(sel_gx, sel_gy)
            spawn_st = int(tick_base_subtick + self.max_pts - 1)
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
            self.oldest_active_subtick = torch.where(
                spawn_now,
                torch.full_like(self.oldest_active_subtick, spawn_st),
                self.oldest_active_subtick,
            )
            for field in _PER_LIFE_STATE_FIELDS:
                cur = getattr(self, field.name)
                fill = (
                    torch.zeros_like(cur)
                    if (field.reset_value == 0 or field.reset_value is False)
                    else torch.full_like(cur, field.reset_value)
                )
                setattr(self, field.name, torch.where(spawn_now, fill, cur))

            spawn_r_sq_1d = self._r_sq_table[int(self.cfg.start_length)]
            spawn_r_sq = spawn_r_sq_1d.expand(self.E, self.N)
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

            self._stamp_sparse_points(
                ready_e,
                ready_i,
                sel_gx_r,
                sel_gy_r,
                self._start_r_sq_int,
                spawn_st,
                active_rows=spawn_r,
            )
            self.total_spawned_snake_mass = self.total_spawned_snake_mass + spawn_now.to(
                torch.int64
            ).sum(dim=1) * int(self.cfg.start_length)

        # Episode lifecycle signals (Rule 12).
        self.life_start.copy_(spawn_now)
        if self.cfg.max_life_ticks > 0:
            life_truncated = survives & (self.life_age >= int(self.cfg.max_life_ticks))
        else:
            life_truncated = torch.zeros_like(survives)
        self.truncated_pending.copy_(life_truncated)

        if trunc_mask is not None:
            rollout_truncated = survives & trunc_mask
        elif truncate:
            rollout_truncated = survives.clone()
        else:
            rollout_truncated = torch.zeros_like(survives)
        truncated = life_truncated | rollout_truncated
        in_respawn_delay = (~self.alive) & (~died)

        if self.debug_assert_invariants:
            self.assert_surviving_discs_disjoint()

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
            prev_length=self.prev_length.clone(),
            mass_eaten=mass_eaten,
            killer_id=killer_id,
            kill_credit=kill_credit,
            life_age=self.life_age.clone(),
        )

    def active_occupancy_grid(self) -> torch.Tensor:
        """Materialize lowest active owner ID ``(E, G, G)`` in int8 (``-1`` where empty)."""
        out = torch.empty((self.E, self.G, self.G), dtype=torch.int8, device=self.device)
        for e0 in range(0, self.E, self.env_block_size):
            e1 = min(self.E, e0 + self.env_block_size)
            eb = e1 - e0
            lo_sub = self.occ_lo[e0:e1]
            hi_sub = self.occ_hi[e0:e1]
            old_sub = self.oldest_active_subtick[e0:e1]

            ow_lo = decode_occ_owner_lo(lo_sub)
            ow_lo_i64 = ow_lo.clamp(0, self.N - 1).to(torch.int64)
            st_lo = old_sub.gather(1, ow_lo_i64.view(eb, -1)).view_as(lo_sub)
            act_lo = occ_entry_active(lo_sub, st_lo)

            ow_hi = decode_occ_owner_hi(hi_sub)
            ow_hi_i64 = ow_hi.clamp(0, self.N - 1).to(torch.int64)
            st_hi = old_sub.gather(1, ow_hi_i64.view(eb, -1)).view_as(hi_sub)
            act_hi = occ_entry_active(hi_sub, st_hi)

            full_n = torch.full_like(ow_lo, self.N)
            lowest = torch.minimum(
                torch.where(act_lo, ow_lo, full_n),
                torch.where(act_hi, ow_hi, full_n),
            )
            out[e0:e1] = torch.where(
                lowest < self.N,
                lowest.to(torch.int8),
                torch.full_like(lowest, -1, dtype=torch.int8),
            )
        return out

    def state_dict(self) -> dict[str, Any]:
        """Serialize full deterministic simulator state for checkpoint/resume."""
        tensors_out: dict[str, torch.Tensor] = {}
        for field in _SAVED_STATE_FIELDS:
            if field.sharded_4x:
                if len(self._occ_4x_blocks) == 1:
                    tensors_out[field.name] = self._occ_4x_blocks[0].detach().cpu().clone()
                else:
                    tensors_out[field.name] = torch.cat(
                        [blk.detach().cpu() for blk in self._occ_4x_blocks], dim=0
                    )
            else:
                tensors_out[field.name] = getattr(self, field.name).detach().cpu().clone()
        return {
            "config": self.cfg.to_dict(),
            "num_envs": self.E,
            "tick": self.tick,
            "subtick_offset": self._subtick_offset,
            "clean_step_streak": self._clean_step_streak,
            "tensors": tensors_out,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Strictly validate and restore deterministic simulator state onto ``self.device``."""
        if not isinstance(state, dict):
            raise TypeError(f"Expected dict for state_dict, got {type(state).__name__}")
        top_keys = set(state.keys())
        missing_top = _REQUIRED_STATE_DICT_KEYS - top_keys
        if missing_top:
            raise KeyError(f"Missing required state_dict keys: {sorted(missing_top)}")
        unknown_top = top_keys - _ALL_STATE_DICT_KEYS
        if unknown_top:
            raise KeyError(f"Unknown state_dict keys: {sorted(unknown_top)}")

        tensors = state["tensors"]
        if not isinstance(tensors, dict):
            raise TypeError(
                f"Expected dict for state_dict['tensors'], got {type(tensors).__name__}"
            )
        tensor_keys = set(tensors.keys())
        missing_tensors = _SAVED_TENSOR_NAMES - tensor_keys
        if missing_tensors:
            raise KeyError(f"Missing state_dict tensor keys: {sorted(missing_tensors)}")
        unknown_tensors = tensor_keys - _SAVED_TENSOR_NAMES
        if unknown_tensors:
            raise KeyError(f"Unknown state_dict tensor keys: {sorted(unknown_tensors)}")

        loaded_cfg = GameV3Config.from_dict(state["config"])
        if loaded_cfg != self.cfg:
            raise ValueError("Checkpoint GameV3Config does not match simulator config")
        if int(state["num_envs"]) != self.E:
            raise ValueError(f"Checkpoint num_envs {state['num_envs']} != {self.E}")
        streak = int(state.get("clean_step_streak", 0))
        if streak < 0:
            raise ValueError(f"Checkpoint clean_step_streak must be >= 0, got {streak}")

        for field in _SAVED_STATE_FIELDS:
            tensor = tensors[field.name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Checkpoint tensor {field.name!r} must be torch.Tensor, "
                    f"got {type(tensor).__name__}"
                )
            exp_shape = state_field_shape(self, field.shape_kind)
            if tuple(tensor.shape) != exp_shape:
                raise ValueError(
                    f"Checkpoint tensor {field.name!r} shape {tuple(tensor.shape)} != {exp_shape}"
                )

        self.tick = int(state["tick"])
        self._subtick_offset = int(state["subtick_offset"])
        self._clean_step_streak = streak
        for field in _SAVED_STATE_FIELDS:
            tensor = tensors[field.name]
            if field.sharded_4x:
                bs = self.env_block_size
                for b_idx, blk in enumerate(self._occ_4x_blocks):
                    e0 = b_idx * bs
                    e1 = e0 + int(blk.shape[0])
                    blk.copy_(tensor[e0:e1].to(device=self.device, dtype=blk.dtype))
            else:
                dst = getattr(self, field.name)
                dst.copy_(tensor.to(device=self.device, dtype=dst.dtype))
