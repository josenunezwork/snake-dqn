"""Configuration, fixed-point constants, and shared integer tables for Game V3 (v3.1)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Fixed-point coordinate and trigonometry scaling constants:
# - Positions and radii use 1/1024 grid-cell units (1.0 world unit at cell_size=1.0).
# - Shared integer sin/cos lookup tables use Q14 fixed-point [-16384, +16384].
FP_SHIFT: int = 10
FP_ONE: int = 1 << FP_SHIFT  # 1024
FP_HALF: int = 1 << (FP_SHIFT - 1)  # 512

TRIG_SHIFT: int = 14
TRIG_ONE: int = 1 << TRIG_SHIFT  # 16384
TRIG_HALF: int = 1 << (TRIG_SHIFT - 1)  # 8192

BOOST_ACCUM_SHIFT: int = 10
BOOST_ACCUM_ONE: int = 1 << BOOST_ACCUM_SHIFT  # 1024

INV_FP_ONE_F32: float = float(np.float32(1.0 / FP_ONE))

NUM_TURN_LEVELS: int = 5
NUM_ACTIONS: int = 10  # 5 turn levels {-1, -0.5, 0, +0.5, +1} x {normal, boost}

# Shared world-population, food-grid, fallback-spawn, and encoding constants:
# - GameV3Config.__post_init__ enforces max_pickup_fp < MAX_FOOD_PICKUP_FP < FOOD_PICKUP_CLAMP_FP,
#   and 2 * FOOD_PICKUP_CLAMP_FP**2 < 2**31 so int32 dx*dx + dy*dy in sim.py never overflows
#   signed int32 while leaving every in-radius candidate untouched.
INIT_SNAKE_CANDIDATES: int = 32
INIT_FOOD_BATCH_SIZE: int = 128
CORPSE_GRID_MAX: int = 32767
MAX_FOOD_PICKUP_FP: int = 30_000
FOOD_PICKUP_CLAMP_FP: int = 32_000

FALLBACK_RING_SIZE: int = 64
FALLBACK_SNAKE_STRIDE: int = 7
FALLBACK_TICK_STRIDE: int = 13
FALLBACK_INNER_RING_FRAC: float = 0.35
FALLBACK_OUTER_RING_FRAC: float = 0.70
FALLBACK_OUTER_ANGLE_OFFSET: float = 0.09817477

SIZE_BUCKET_LOG_SCALE: float = 32.0
SIZE_BUCKET_LEN_DIV: float = 4.0

FOOD_LUT_MAX: int = 2048
AMBIENT_FOOD_HALF_SAT: float = 1.0
CORPSE_FOOD_HALF_SAT: float = 4.0

# 1x occupancy bit layout (signed int32):
# - Empty sentinel is -1.
# - Non-empty cells pack ``(subtick << OCC_OWNER_BITS) | owner_code`` where ``occ_hi``
#   stores ``owner_code = slot_id`` and ``occ_lo`` stores ``OCC_OWNER_MASK - slot_id``.
# - Non-negative signed int32 requires ``0 <= subtick <= MAX_OCC_SUBTICK``.
OCC_OWNER_BITS: int = 6
MAX_SNAKES: int = 1 << OCC_OWNER_BITS  # 64
OCC_OWNER_MASK: int = MAX_SNAKES - 1  # 63
OCC_SUBTICK_BITS: int = 31 - OCC_OWNER_BITS  # 25
MAX_OCC_SUBTICK: int = (1 << OCC_SUBTICK_BITS) - 1  # 33,554,431


def pack_occ_hi(subtick: Any, slot_id: Any) -> Any:
    """Pack ``(subtick, slot_id)`` into signed ``int32`` ``occ_hi`` format."""
    return (subtick << OCC_OWNER_BITS) | slot_id


def pack_occ_lo(subtick: Any, slot_id: Any) -> Any:
    """Pack ``(subtick, slot_id)`` into signed ``int32`` ``occ_lo`` format."""
    return (subtick << OCC_OWNER_BITS) | (OCC_OWNER_MASK - slot_id)


def decode_occ_owner_hi(packed: Any) -> Any:
    """Decode ``slot_id`` from a non-negative ``occ_hi`` entry."""
    return packed & OCC_OWNER_MASK


def decode_occ_owner_lo(packed: Any) -> Any:
    """Decode ``slot_id`` from a non-negative ``occ_lo`` entry."""
    return OCC_OWNER_MASK - (packed & OCC_OWNER_MASK)


def decode_occ_subtick(packed: Any) -> Any:
    """Extract the packed subtick from a non-negative ``occ_hi`` or ``occ_lo`` entry."""
    return packed >> OCC_OWNER_BITS


def occ_entry_active(packed: Any, min_active_subtick: Any) -> Any:
    """Return boolean mask where ``packed`` is non-empty and its subtick >= ``min_active_subtick``."""
    return (packed >= 0) & ((packed >> OCC_OWNER_BITS) >= min_active_subtick)


@dataclass(frozen=True)
class GameV3Config:
    """Frozen configuration for the deterministic fixed-point Game V3.1 simulator.

    Every gameplay, geometry, fixed-point table, food-economy, episode, and
    observation parameter is exposed here with its v3.1 specification default
    and supports lossless JSON round-tripping.
    """

    # World & respawn parameters.
    arena_radius: float = 600.0
    num_snakes: int = 24
    cell_size: float = 1.0
    start_length: int = 10
    max_length: int = 2048
    respawn_delay: int = 30
    respawn_candidates: int = 8
    respawn_fallback_candidates: int = 8
    respawn_clearance_cells: int = 4
    spawn_wall_margin: float = 20.0
    max_life_ticks: int = 0

    # Movement, steering, fixed-point angle, and boost parameters.
    angle_steps: int = 4096
    speed: float = 1.0
    boost_speed: float = 2.0
    turn_rate: float = 0.12
    turn_exponent: float = 0.24
    min_boost_length: int = 15
    boost_shed_ticks: int = 4
    boost_shed_mass: int = 1
    boost_drop_fraction: float = 0.5

    # Body geometry & food pickup parameters:
    # w(L) = min(max_body_radius, base + scale * log2(1 + L / len_scale)).
    base_body_radius: float = 1.0
    body_radius_scale: float = 0.5
    body_radius_length_scale: float = 50.0
    max_body_radius: float = 4.0
    food_pickup_mult: float = 1.5

    # Food economy parameters.
    ambient_food_density: float = 0.002
    ambient_pellet_value: int = 1
    ambient_spawn_budget_per_tick: int = 32
    food_wall_margin: float = 2.0
    corpse_mass_fraction: float = 0.5
    corpse_max_pellets: int = 32

    # Observation pyramid parameters (uint8 rasters: 64x64 @ 1x, 32x32 @ 4x and 16x).
    obs_fine_size: int = 64
    obs_coarse_size: int = 32
    obs_scales: tuple[int, int, int] = (1, 4, 16)

    def __post_init__(self) -> None:
        """Validate configuration invariants and canonicalize tuple fields."""
        object.__setattr__(self, "obs_scales", tuple(int(round(x)) for x in self.obs_scales))

        if not (0.0 < self.arena_radius <= 2000.0):
            raise ValueError(f"arena_radius must be in (0, 2000], got {self.arena_radius}")
        if not (6 <= self.num_snakes <= MAX_SNAKES):
            raise ValueError(
                f"num_snakes must be in [6, {MAX_SNAKES}] (derived from {OCC_OWNER_BITS}-bit "
                f"occupancy owner layout), got {self.num_snakes}"
            )
        if self.cell_size <= 0.0:
            raise ValueError(f"cell_size must be positive, got {self.cell_size}")
        if self.start_length < 2:
            raise ValueError(f"start_length must be >= 2, got {self.start_length}")
        if self.max_length < self.start_length:
            raise ValueError(
                f"max_length ({self.max_length}) must be >= start_length ({self.start_length})"
            )
        if self.respawn_delay < 1:
            raise ValueError(f"respawn_delay must be >= 1, got {self.respawn_delay}")
        if self.respawn_candidates < 1:
            raise ValueError(f"respawn_candidates must be >= 1, got {self.respawn_candidates}")
        if self.respawn_fallback_candidates < 1:
            raise ValueError(
                f"respawn_fallback_candidates must be >= 1, got {self.respawn_fallback_candidates}"
            )
        if self.respawn_clearance_cells < 1:
            raise ValueError(
                f"respawn_clearance_cells must be >= 1, got {self.respawn_clearance_cells}"
            )
        if self.max_life_ticks < 0:
            raise ValueError(f"max_life_ticks must be >= 0, got {self.max_life_ticks}")
        if self.angle_steps < 64 or (self.angle_steps & (self.angle_steps - 1)) != 0:
            raise ValueError(f"angle_steps must be a power of 2 >= 64, got {self.angle_steps}")
        if self.speed <= 0.0 or self.boost_speed < self.speed:
            raise ValueError(
                f"Require 0 < speed <= boost_speed, got {self.speed}, {self.boost_speed}"
            )
        if self.turn_rate <= 0.0 or self.turn_rate >= math.pi:
            raise ValueError(f"turn_rate must be in (0, pi), got {self.turn_rate}")
        if self.turn_exponent < 0.0:
            raise ValueError(f"turn_exponent must be >= 0, got {self.turn_exponent}")
        if self.min_boost_length < 1:
            raise ValueError(f"min_boost_length must be >= 1, got {self.min_boost_length}")
        if self.boost_shed_ticks < 1 or self.boost_shed_mass < 1:
            raise ValueError("boost_shed_ticks and boost_shed_mass must be >= 1")
        if self.boost_shed_mass > self.min_boost_length:
            raise ValueError("boost_shed_mass must be <= min_boost_length")
        if not (0.0 <= self.boost_drop_fraction <= 1.0):
            raise ValueError(
                f"boost_drop_fraction must be in [0, 1], got {self.boost_drop_fraction}"
            )
        if (
            self.base_body_radius <= 0.0
            or self.body_radius_scale < 0.0
            or self.body_radius_length_scale <= 0.0
            or self.max_body_radius < self.base_body_radius
        ):
            raise ValueError("Invalid body radius parameters")
        if self.food_pickup_mult <= 0.0:
            raise ValueError(f"food_pickup_mult must be > 0, got {self.food_pickup_mult}")
        max_pickup_fp = int(
            round((self.food_pickup_mult * self.max_body_radius / self.cell_size) * FP_ONE)
        )
        if max_pickup_fp <= 0 or max_pickup_fp >= MAX_FOOD_PICKUP_FP:
            raise ValueError(
                "food_pickup_mult * max_body_radius exceeds safe local int32 squared-distance range"
            )
        if (
            self.arena_radius
            <= self.max_body_radius + self.spawn_wall_margin + 2.0 * self.cell_size
        ):
            raise ValueError("arena_radius is too small for max_body_radius and spawn_wall_margin")
        if self.ambient_food_density < 0.0 or not (1 <= self.ambient_pellet_value <= 255):
            raise ValueError(
                "Invalid ambient food parameters (require density >= 0 and pellet_value in 1..255)"
            )
        if self.ambient_spawn_budget_per_tick < 1:
            raise ValueError("ambient_spawn_budget_per_tick must be >= 1")
        if not (0.0 <= self.corpse_mass_fraction <= 1.0):
            raise ValueError(
                f"corpse_mass_fraction must be in [0, 1], got {self.corpse_mass_fraction}"
            )
        if self.corpse_max_pellets < 1:
            raise ValueError(f"corpse_max_pellets must be >= 1, got {self.corpse_max_pellets}")
        if self.obs_fine_size < 8 or self.obs_fine_size % 2 != 0:
            raise ValueError("obs_fine_size must be an even integer >= 8")
        if self.obs_coarse_size < 8 or self.obs_coarse_size % 2 != 0:
            raise ValueError("obs_coarse_size must be an even integer >= 8")
        if len(self.obs_scales) != 3 or any(s < 1 for s in self.obs_scales):
            raise ValueError(f"obs_scales must be 3 positive integers, got {self.obs_scales}")

    @property
    def num_actions(self) -> int:
        """Number of discrete actions: 5 turn levels x {normal, boost} = 10."""
        return NUM_ACTIONS

    @property
    def half_grid(self) -> int:
        """Half-width of the square cell grid, padded and aligned to 16 cells."""
        margin = max(self.max_body_radius, self.food_pickup_mult * self.max_body_radius)
        pad = int(math.ceil(self.max_body_radius / self.cell_size)) + 4
        raw = int(math.ceil((self.arena_radius + margin) / self.cell_size)) + pad
        return ((raw + 15) // 16) * 16

    @property
    def grid_size(self) -> int:
        """Side length G of the square occupancy and pellet grid (divisible by 16)."""
        return 2 * self.half_grid

    @property
    def arena_radius_fp(self) -> int:
        """Arena radius in 1/1024 cell fixed-point units."""
        return int(round((self.arena_radius / self.cell_size) * FP_ONE))

    @property
    def target_ambient_pellets(self) -> int:
        """Target count of live ambient pellets across the circular arena."""
        area = math.pi * self.arena_radius * self.arena_radius
        return int(round(self.ambient_food_density * area))

    @property
    def boost_drop_fp_per_shed(self) -> int:
        """Fixed-point pellet mass (in 1/1024 units) added per boost shed event."""
        return int(round(self.boost_shed_mass * self.boost_drop_fraction * BOOST_ACCUM_ONE))

    @property
    def max_points_per_tick(self) -> int:
        """Compile-time upper bound on distance-spaced spine points emitted in one tick."""
        speed_fp = max(1, int(round((self.speed / self.cell_size) * FP_ONE)))
        boost_fp = max(speed_fp, int(round((self.boost_speed / self.cell_size) * FP_ONE)))
        return int(((speed_fp - 1) + boost_fp) // speed_fp)

    def body_radius_f64(self, length: int) -> float:
        """Compute exact float64 body radius w(L) for integer length L."""
        eff_l = max(0, int(length))
        raw = self.base_body_radius + self.body_radius_scale * math.log2(
            1.0 + eff_l / self.body_radius_length_scale
        )
        return min(self.max_body_radius, raw)

    def turn_rate_f64(self, length: int) -> float:
        """Compute exact float64 maximum turn rate (rad/tick) for integer length L.

        Uses power-law decay ``turn_rate * (start_length / max(start_length, L)) ** turn_exponent``
        so at default ``turn_exponent = 0.24`` turn radius triples from ``L = 10`` to ``L = 1000``.
        """
        eff_l = max(self.start_length, int(length))
        return self.turn_rate * ((self.start_length / eff_l) ** self.turn_exponent)

    def turn_quantum(self, length: int) -> int:
        """Return exact integer full-turn angle steps for integer length L (>= 2)."""
        raw_steps = self.turn_rate_f64(length) * self.angle_steps / (2.0 * math.pi)
        return max(2, int(round(raw_steps)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration fields to a JSON-compatible dictionary."""
        data = asdict(self)
        data["obs_scales"] = list(self.obs_scales)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameV3Config:
        """Construct a validated configuration from a dictionary."""
        parsed = dict(data)
        if "obs_scales" in parsed:
            parsed["obs_scales"] = tuple(int(round(x)) for x in parsed["obs_scales"])
        return cls(**parsed)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> GameV3Config:
        """Deserialize configuration from a JSON string."""
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("GameV3Config JSON payload must be an object")
        return cls.from_dict(raw)


def build_trig_tables_np(angle_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Build exact quadrant-symmetric int32 Q14 (sin_table, cos_table) of length ``angle_steps``."""
    if angle_steps < 64 or (angle_steps & (angle_steps - 1)) != 0:
        raise ValueError(f"angle_steps must be a power of 2 >= 64, got {angle_steps}")
    q = angle_steps // 4
    quarter = np.empty(q + 1, dtype=np.int32)
    quarter[0] = 0
    for k in range(1, q):
        quarter[k] = int(round(TRIG_ONE * math.sin(0.5 * math.pi * k / q)))
    quarter[q] = TRIG_ONE

    sin_tbl = np.empty(angle_steps, dtype=np.int32)
    sin_tbl[0:q] = quarter[0:q]
    sin_tbl[q : 2 * q] = quarter[q:0:-1]
    sin_tbl[2 * q : 3 * q] = -quarter[0:q]
    sin_tbl[3 * q : 4 * q] = -quarter[q:0:-1]

    cos_tbl = np.roll(sin_tbl, -q).copy()
    return sin_tbl, cos_tbl


def build_length_tables_np(cfg: GameV3Config) -> dict[str, np.ndarray]:
    """Build all length-indexed integer and scalar lookup tables for ``L in 0..max_length``."""
    max_len = cfg.max_length
    c = cfg.cell_size
    arena_r_fp = cfg.arena_radius_fp

    base_speed_fp = max(1, int(round((cfg.speed / c) * FP_ONE)))
    boost_spd_fp = max(base_speed_fp, int(round((cfg.boost_speed / c) * FP_ONE)))

    speed_fp = np.full(max_len + 1, base_speed_fp, dtype=np.int32)
    boost_speed_fp = np.full(max_len + 1, boost_spd_fp, dtype=np.int32)
    turn_q = np.empty(max_len + 1, dtype=np.int32)
    turn_deltas = np.empty((max_len + 1, NUM_TURN_LEVELS), dtype=np.int32)
    radius_fp = np.empty(max_len + 1, dtype=np.int32)
    r_sq_cells = np.empty(max_len + 1, dtype=np.int32)
    pickup_r_fp = np.empty(max_len + 1, dtype=np.int32)
    pickup_r_sq_fp = np.empty(max_len + 1, dtype=np.int32)
    wall_limit_sq_fp = np.empty(max_len + 1, dtype=np.int64)
    corpse_mass = np.empty(max_len + 1, dtype=np.int32)
    size_bucket = np.zeros(max_len + 1, dtype=np.uint8)

    len_log_f32 = np.zeros(max_len + 1, dtype=np.float32)
    radius_norm_f32 = np.zeros(max_len + 1, dtype=np.float32)
    turn_norm_f32 = np.zeros(max_len + 1, dtype=np.float32)

    denom_log = math.log2(1.0 + max_len)
    start_q = float(cfg.turn_quantum(cfg.start_length))
    max_r_fp = max(1, int(round((cfg.max_body_radius / c) * FP_ONE)))

    for length in range(max_len + 1):
        w64 = cfg.body_radius_f64(length)
        r_fp = max(1, int(round((w64 / c) * FP_ONE)))
        radius_fp[length] = r_fp
        r_sq_cells[length] = max(0, int(math.floor((w64 / c) ** 2 + 1e-12)))

        pr_fp = max(1, int(round(cfg.food_pickup_mult * r_fp)))
        pickup_r_fp[length] = pr_fp
        pickup_r_sq_fp[length] = pr_fp * pr_fp

        wall_r_fp = max(0, arena_r_fp - r_fp)
        wall_limit_sq_fp[length] = int(wall_r_fp) * int(wall_r_fp)
        corpse_mass[length] = int(math.floor(length * cfg.corpse_mass_fraction + 1e-12))

        fq = cfg.turn_quantum(max(1, length))
        hq = max(1, fq // 2)
        turn_q[length] = fq
        # Actions 0..4 (and 5..9 with boost): {-1, -0.5, 0, +0.5, +1} where left increases theta.
        turn_deltas[length, 0] = fq
        turn_deltas[length, 1] = hq
        turn_deltas[length, 2] = 0
        turn_deltas[length, 3] = -hq
        turn_deltas[length, 4] = -fq

        if length >= 1:
            size_bucket[length] = min(
                255,
                max(
                    1,
                    int(
                        round(SIZE_BUCKET_LOG_SCALE * math.log2(1.0 + length / SIZE_BUCKET_LEN_DIV))
                    ),
                ),
            )
            len_log_f32[length] = np.float32(math.log2(1.0 + length) / denom_log)
        radius_norm_f32[length] = np.float32(r_fp / max_r_fp)
        turn_norm_f32[length] = np.float32(fq / start_q)

    return {
        "speed_fp": speed_fp,
        "boost_speed_fp": boost_speed_fp,
        "turn_quantum": turn_q,
        "turn_deltas": turn_deltas,
        "radius_fp": radius_fp,
        "r_sq_cells": r_sq_cells,
        "pickup_r_fp": pickup_r_fp,
        "pickup_r_sq_fp": pickup_r_sq_fp,
        "wall_limit_sq_fp": wall_limit_sq_fp,
        "corpse_mass": corpse_mass,
        "size_bucket": size_bucket,
        "len_log_f32": len_log_f32,
        "radius_norm_f32": radius_norm_f32,
        "turn_norm_f32": turn_norm_f32,
    }


def build_fallback_spawns_np(cfg: GameV3Config) -> np.ndarray:
    """Build deterministic fallback spawn table ``(64, 3)`` of ``(gx, gy, angle_idx)`` in int32."""
    c = cfg.cell_size
    hg = cfg.half_grid
    g_sz = cfg.grid_size
    w_start = cfg.body_radius_f64(cfg.start_length)
    max_safe_r_cells = max(1.0, (cfg.arena_radius - w_start - cfg.spawn_wall_margin) / c - 1.0)

    table = np.empty((FALLBACK_RING_SIZE, 3), dtype=np.int32)
    half_ring = FALLBACK_RING_SIZE // 2
    for idx in range(FALLBACK_RING_SIZE):
        ring_frac = FALLBACK_INNER_RING_FRAC if idx < half_ring else FALLBACK_OUTER_RING_FRAC
        k = idx % half_ring
        ang = 2.0 * math.pi * k / half_ring + (
            FALLBACK_OUTER_ANGLE_OFFSET if idx >= half_ring else 0.0
        )
        r_cells = ring_frac * max_safe_r_cells
        gx = min(max(int(round(hg + r_cells * math.cos(ang))), 0), g_sz - 1)
        gy = min(max(int(round(hg + r_cells * math.sin(ang))), 0), g_sz - 1)
        # Point inward toward arena origin (ang + pi).
        inward_idx = int(round(((ang + math.pi) / (2.0 * math.pi)) * cfg.angle_steps)) % (
            cfg.angle_steps
        )
        table[idx, 0] = gx
        table[idx, 1] = gy
        table[idx, 2] = inward_idx
    return table


def build_disc_offsets_np(
    cfg: GameV3Config,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted ``(max_r_cells, d2, dy, dx)`` body-disc cell offsets in ``int32``."""
    max_r_cells = int(math.floor(cfg.max_body_radius / cfg.cell_size + 1e-12))
    offsets: list[tuple[int, int, int]] = []
    for dy in range(-max_r_cells, max_r_cells + 1):
        for dx in range(-max_r_cells, max_r_cells + 1):
            d2 = dx * dx + dy * dy
            if d2 <= max_r_cells * max_r_cells:
                offsets.append((d2, dy, dx))
    offsets.sort()
    disc_dist_sq = np.array([o[0] for o in offsets], dtype=np.int32)
    disc_dy = np.array([o[1] for o in offsets], dtype=np.int32)
    disc_dx = np.array([o[2] for o in offsets], dtype=np.int32)
    return max_r_cells, disc_dist_sq, disc_dy, disc_dx


def build_eat_offsets_np(cfg: GameV3Config) -> tuple[int, np.ndarray, np.ndarray]:
    """Build sorted ``(max_eat_r_cells, dy, dx)`` candidate food-pickup cell offsets in ``int32``."""
    max_eat_r_cells = (
        int(math.ceil((cfg.food_pickup_mult * cfg.max_body_radius) / cfg.cell_size)) + 1
    )
    eat_offsets: list[tuple[int, int, int]] = []
    for dy in range(-max_eat_r_cells, max_eat_r_cells + 1):
        for dx in range(-max_eat_r_cells, max_eat_r_cells + 1):
            d2 = dx * dx + dy * dy
            if d2 <= max_eat_r_cells * max_eat_r_cells:
                eat_offsets.append((d2, dy, dx))
    eat_offsets.sort()
    eat_dy = np.array([o[1] for o in eat_offsets], dtype=np.int32)
    eat_dx = np.array([o[2] for o in eat_offsets], dtype=np.int32)
    return max_eat_r_cells, eat_dy, eat_dx


def build_clear_offsets_np(cfg: GameV3Config) -> tuple[int, np.ndarray, np.ndarray]:
    """Build ``(r_clear, dx, dy)`` respawn clearance cell offsets in ``int32``."""
    r_clear = max(
        int(math.ceil(cfg.body_radius_f64(cfg.start_length) / cfg.cell_size)) + 2,
        cfg.respawn_clearance_cells,
    )
    clear_offsets: list[tuple[int, int]] = []
    for dy in range(-r_clear, r_clear + 1):
        for dx in range(-r_clear, r_clear + 1):
            if dx * dx + dy * dy <= r_clear * r_clear:
                clear_offsets.append((dx, dy))
    clear_dx = np.array([o[0] for o in clear_offsets], dtype=np.int32)
    clear_dy = np.array([o[1] for o in clear_offsets], dtype=np.int32)
    return r_clear, clear_dx, clear_dy


def spawn_wall_sq_fp(cfg: GameV3Config, start_r_fp: int | None = None) -> int:
    """Compute exact squared fixed-point arena limit for initial/respawn placement."""
    if start_r_fp is None:
        w64 = cfg.body_radius_f64(cfg.start_length)
        start_r_fp = max(1, int(round((w64 / cfg.cell_size) * FP_ONE)))
    spawn_margin_fp = int(round((cfg.spawn_wall_margin / cfg.cell_size) * FP_ONE))
    spawn_lim_fp = max(FP_ONE, cfg.arena_radius_fp - int(start_r_fp) - spawn_margin_fp)
    return int(spawn_lim_fp) * int(spawn_lim_fp)


def food_wall_sq_fp(cfg: GameV3Config) -> int:
    """Compute exact squared fixed-point arena limit for ambient food spawning."""
    food_margin_fp = int(round((cfg.food_wall_margin / cfg.cell_size) * FP_ONE))
    food_lim_fp = max(FP_ONE, cfg.arena_radius_fp - food_margin_fp)
    return int(food_lim_fp) * int(food_lim_fp)


def init_food_num_batches(cfg: GameV3Config) -> int:
    """Return deterministic batch count for initial ambient-food population."""
    return max(1, int(math.ceil(cfg.target_ambient_pellets * 2.0 / INIT_FOOD_BATCH_SIZE)) + 2)


def build_food_luts_np() -> tuple[np.ndarray, np.ndarray]:
    """Build exact ``uint8`` lookup tables ``(ambient_lut, corpse_lut)`` over ``0..FOOD_LUT_MAX``."""
    amb_lut = np.empty(FOOD_LUT_MAX + 1, dtype=np.uint8)
    cor_lut = np.empty(FOOD_LUT_MAX + 1, dtype=np.uint8)
    for m in range(FOOD_LUT_MAX + 1):
        amb_lut[m] = min(255, int(round(255.0 * m / (AMBIENT_FOOD_HALF_SAT + m))))
        cor_lut[m] = min(255, int(round(255.0 * m / (CORPSE_FOOD_HALF_SAT + m))))
    return amb_lut, cor_lut
