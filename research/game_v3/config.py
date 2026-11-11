"""Configuration and shared numerical constants for Game V3."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Exact IEEE-754 float32 angle and minimax/Taylor polynomial constants shared by
# both the PyTorch simulator and the plain-NumPy reference so CPU float32
# trigonometry is bit-for-bit identical across runtimes without libm/SLEEF ULP drift.
PI_F32: float = float(np.float32(math.pi))
TWO_PI_F32: float = float(np.float32(2.0 * math.pi))
HALF_PI_F32: float = float(np.float32(0.5 * math.pi))

SIN_C3: float = float(np.float32(-1.666665668e-1))
SIN_C5: float = float(np.float32(8.333025139e-3))
SIN_C7: float = float(np.float32(-1.980741872e-4))
SIN_C9: float = float(np.float32(2.601903036e-6))

COS_C2: float = float(np.float32(-0.5))
COS_C4: float = float(np.float32(4.166666418313980e-2))
COS_C6: float = float(np.float32(-1.388887874782085e-3))
COS_C8: float = float(np.float32(2.480060087811244e-5))
COS_C10: float = float(np.float32(-2.752556287020700e-7))


@dataclass(frozen=True)
class GameV3Config:
    """Frozen configuration for the continuous-angle Game V3 simulator.

    Every gameplay, geometry, food-economy, and observation parameter is exposed
    here with its specification default and supports lossless JSON round-tripping.
    """

    # World parameters.
    arena_radius: float = 600.0
    num_snakes: int = 24
    cell_size: float = 1.0
    start_length: int = 10
    max_length: int = 1024
    respawn_delay: int = 30
    respawn_candidates: int = 8
    respawn_clearance_cells: int = 4
    spawn_wall_margin: float = 8.0

    # Movement, steering, and boost parameters.
    speed: float = 1.0
    boost_speed: float = 2.0
    turn_rate: float = 0.12
    turn_width_power: float = 1.0
    min_boost_length: int = 15
    boost_shed_ticks: int = 4
    boost_shed_mass: int = 1

    # Body geometry parameters: w(L) = min(max_w, base + scale * log2(1 + L / len_scale)).
    base_body_radius: float = 1.0
    body_radius_scale: float = 0.5
    body_radius_length_scale: float = 50.0
    max_body_radius: float = 4.0

    # Collision parameters.
    head_on_tie_margin: float = 0.15

    # Food economy parameters.
    ambient_food_density: float = 0.002
    ambient_pellet_value: int = 1
    ambient_spawn_budget_per_tick: int = 32
    food_wall_margin: float = 2.0
    corpse_mass_fraction: float = 0.5
    corpse_max_pellets: int = 32

    # Observation parameters.
    obs_raster_size: int = 64
    obs_scales: tuple[float, float, float] = (1.0, 4.0, 16.0)

    def __post_init__(self) -> None:
        """Validate configuration invariants and canonicalize tuple fields."""
        if isinstance(self.obs_scales, list):
            object.__setattr__(self, "obs_scales", tuple(float(x) for x in self.obs_scales))
        else:
            object.__setattr__(self, "obs_scales", tuple(float(x) for x in self.obs_scales))

        if not (0.0 < self.arena_radius <= 2000.0):
            raise ValueError(f"arena_radius must be in (0, 2000], got {self.arena_radius}")
        if not (6 <= self.num_snakes <= 64):
            raise ValueError(f"num_snakes must be in [6, 64], got {self.num_snakes}")
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
        if self.respawn_clearance_cells < 1:
            raise ValueError(
                f"respawn_clearance_cells must be >= 1, got {self.respawn_clearance_cells}"
            )
        if self.speed <= 0.0 or self.boost_speed < self.speed:
            raise ValueError(
                f"Require 0 < speed <= boost_speed, got {self.speed}, {self.boost_speed}"
            )
        if self.turn_rate <= 0.0 or self.turn_rate >= math.pi:
            raise ValueError(f"turn_rate must be in (0, pi), got {self.turn_rate}")
        if self.min_boost_length < 1:
            raise ValueError(f"min_boost_length must be >= 1, got {self.min_boost_length}")
        if self.boost_shed_ticks < 1 or self.boost_shed_mass < 1:
            raise ValueError("boost_shed_ticks and boost_shed_mass must be >= 1")
        if self.boost_shed_mass > self.min_boost_length:
            raise ValueError("boost_shed_mass must be <= min_boost_length")
        if (
            self.base_body_radius <= 0.0
            or self.body_radius_scale < 0.0
            or self.body_radius_length_scale <= 0.0
            or self.max_body_radius < self.base_body_radius
        ):
            raise ValueError("Invalid body radius parameters")
        if (
            self.arena_radius
            <= self.max_body_radius + self.spawn_wall_margin + 2.0 * self.cell_size
        ):
            raise ValueError("arena_radius is too small for max_body_radius and spawn_wall_margin")
        if self.head_on_tie_margin < 0.0:
            raise ValueError(f"head_on_tie_margin must be >= 0, got {self.head_on_tie_margin}")
        if self.ambient_food_density < 0.0 or self.ambient_pellet_value < 1:
            raise ValueError("Invalid ambient food parameters")
        if self.ambient_spawn_budget_per_tick < 1:
            raise ValueError("ambient_spawn_budget_per_tick must be >= 1")
        if not (0.0 <= self.corpse_mass_fraction <= 1.0):
            raise ValueError(
                f"corpse_mass_fraction must be in [0, 1], got {self.corpse_mass_fraction}"
            )
        if self.corpse_max_pellets < 1:
            raise ValueError(f"corpse_max_pellets must be >= 1, got {self.corpse_max_pellets}")
        if self.obs_raster_size < 8 or self.obs_raster_size % 2 != 0:
            raise ValueError("obs_raster_size must be an even integer >= 8")
        if len(self.obs_scales) != 3 or any(s <= 0.0 for s in self.obs_scales):
            raise ValueError(f"obs_scales must be 3 positive floats, got {self.obs_scales}")

    @property
    def half_grid(self) -> int:
        """Half-width of the square cell grid, padded and aligned to 16 cells."""
        raw = int(math.ceil((self.arena_radius + self.max_body_radius) / self.cell_size)) + 2
        return ((raw + 15) // 16) * 16

    @property
    def grid_size(self) -> int:
        """Side length G of the square occupancy and pellet grid (divisible by 16)."""
        return 2 * self.half_grid

    @property
    def target_ambient_pellets(self) -> int:
        """Target count of live ambient pellets across the circular arena."""
        area = math.pi * self.arena_radius * self.arena_radius
        return int(round(self.ambient_food_density * area))

    @property
    def head_on_ratio_bp(self) -> int:
        """Head-on strict-win multiplier in basis points (10000 = 1.0x)."""
        return int(round((1.0 + self.head_on_tie_margin) * 10000.0))

    def body_radius_f64(self, length: int) -> float:
        """Compute exact float64 body radius w(L) for integer length L."""
        eff_l = max(0, int(length))
        raw = self.base_body_radius + self.body_radius_scale * math.log2(
            1.0 + eff_l / self.body_radius_length_scale
        )
        return min(self.max_body_radius, raw)

    def turn_rate_f64(self, length: int) -> float:
        """Compute exact float64 maximum turn rate (rad/tick) for integer length L."""
        w0 = self.body_radius_f64(self.start_length)
        w = self.body_radius_f64(max(1, int(length)))
        return self.turn_rate * ((w0 / w) ** self.turn_width_power)

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
            parsed["obs_scales"] = tuple(float(x) for x in parsed["obs_scales"])
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
