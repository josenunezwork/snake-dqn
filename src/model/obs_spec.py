"""Shared observation-spec constants and shape descriptors (checkpoint contract v2).

An ``obs_spec`` string names *what a checkpoint's network consumes* so that a
raster-trained checkpoint and a 61-D vector champion can coexist forever behind
the same serving surface. The string is written into checkpoint metadata under
:data:`OBS_SPEC_KEY`; when absent, a checkpoint is treated as :data:`VECTOR61`
so every historical champion (``champion_a5_freespace_20260621.pth`` and all
prior ones) keeps loading unchanged.

Two specs exist today:

- :data:`VECTOR61` — the legacy 61-D free-space vector consumed by
  :class:`~src.model.apex_network.ApexNetwork` (dueling MLP). Also covers the
  58-D variant; only the input width differs and that is inferred from weights.
- :data:`RASTER31V2` — the dual-scale ego-raster (blueprint §2) consumed by
  ``RasterDuelingNetwork``: a tactical ``(9, 31, 31)`` stack, a strategic
  ``(3, 25, 25)`` stack, and a ``(26,)`` scalar vector.

The :class:`RasterObsShapes` descriptor records the exact channel/size numbers
so a checkpoint can rebuild a correctly-sized raster network without a config.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.runtime_contract import ObservationContract

__all__ = [
    "OBS_SPEC_KEY",
    "VECTOR61",
    "RASTER31V2",
    "RASTER31V3",
    "RASTER_OBS_SPECS",
    "DEFAULT_OBS_SPEC",
    "KNOWN_OBS_SPECS",
    "RasterObsShapes",
    "RASTER31V2_SHAPES",
    "RASTER31V3_CONTRACT",
]

# -- metadata key ----------------------------------------------------------
#: Checkpoint metadata key under which the obs_spec string is stored.
OBS_SPEC_KEY = "obs_spec"

# -- obs_spec string constants ---------------------------------------------
#: Legacy 61-D (or 58-D) free-space vector consumed by ``ApexNetwork``.
VECTOR61 = "vector61"
#: Dual-scale ego-raster (blueprint §2) consumed by ``RasterDuelingNetwork``.
RASTER31V2 = "raster31v2"
#: Corrected raster semantics. Serving deliberately does not dispatch this
#: value until S1 installs its matching featurizer and loader.
RASTER31V3 = "raster31v3"

#: Spec assumed when a checkpoint omits :data:`OBS_SPEC_KEY` — keeps every
#: pre-contract-v2 champion loadable unchanged.
DEFAULT_OBS_SPEC = VECTOR61

#: All obs_spec strings this build understands.
RASTER_OBS_SPECS = (RASTER31V2, RASTER31V3)

# A recognized v3 descriptor is not, by itself, a serving guarantee. The
# loader validates its semantic metadata before construction and Session adds
# the deployment-world gate. Keeping recognition here lets failures be precise
# instead of silently treating a v3 checkpoint as a vector model.
KNOWN_OBS_SPECS = (VECTOR61, *RASTER_OBS_SPECS)


@dataclass(frozen=True)
class RasterObsShapes:
    """Channel/size descriptor for a ``raster31v2`` network's three inputs.

    Attributes:
        tactical_channels: Expanded tactical channel count (9).
        tactical_size: Tactical raster edge length in cells (31).
        strategic_channels: Strategic density channel count (3).
        strategic_size: Strategic raster edge length in cells (25).
        scalars: Ego-frame scalar-vector length (26).
    """

    tactical_channels: int = 9
    tactical_size: int = 31
    strategic_channels: int = 3
    strategic_size: int = 25
    scalars: int = 26

    @property
    def tactical_shape(self) -> tuple[int, int, int]:
        """Per-sample tactical shape ``(C, H, W)``."""
        return (self.tactical_channels, self.tactical_size, self.tactical_size)

    @property
    def strategic_shape(self) -> tuple[int, int, int]:
        """Per-sample strategic shape ``(C, H, W)``."""
        return (self.strategic_channels, self.strategic_size, self.strategic_size)

    def to_metadata(self) -> dict[str, int]:
        """Flatten to the scalar checkpoint-metadata keys the trainer records."""
        return {
            "tactical_channels": self.tactical_channels,
            "tactical_size": self.tactical_size,
            "strategic_channels": self.strategic_channels,
            "strategic_size": self.strategic_size,
            "scalars": self.scalars,
        }

    @classmethod
    def from_metadata(cls, meta: dict) -> "RasterObsShapes":
        """Rebuild from checkpoint metadata, falling back to the canonical shape.

        Missing keys default to the :data:`RASTER31V2_SHAPES` canonical values,
        so a checkpoint that records nothing beyond ``obs_spec='raster31v2'``
        still rebuilds the standard network.

        Args:
            meta: Mapping of the flat shape keys (see :meth:`to_metadata`).

        Returns:
            A populated :class:`RasterObsShapes`.
        """
        d = RASTER31V2_SHAPES
        return cls(
            tactical_channels=int(meta.get("tactical_channels", d.tactical_channels)),
            tactical_size=int(meta.get("tactical_size", d.tactical_size)),
            strategic_channels=int(meta.get("strategic_channels", d.strategic_channels)),
            strategic_size=int(meta.get("strategic_size", d.strategic_size)),
            scalars=int(meta.get("scalars", d.scalars)),
        )


#: Canonical ``raster31v2`` shapes (blueprint §2 / featurizer geometry).
RASTER31V2_SHAPES = RasterObsShapes()

# The v3 tensor dimensions deliberately match v2. The ordered descriptors and
# semantic digest prevent a checkpoint with corrected painting semantics from
# being relabelled as a compatible v2 artifact.
RASTER31V3_CONTRACT = ObservationContract(
    obs_spec=RASTER31V3,
    tactical_shape=RASTER31V2_SHAPES.tactical_shape,
    strategic_shape=RASTER31V2_SHAPES.strategic_shape,
    scalar_count=26,
    tactical_channels=(
        "own_body",
        "enemy_body",
        "enemy_head",
        "enemy_prediction",
        "ambient_food",
        "corpse_food",
        "wall",
        "own_head",
        "reserved",
    ),
    strategic_channels=(
        "enemy_mass_density",
        "food_mass_density",
        "own_body_density",
    ),
    scalars=(
        "length_normalized",
        "length_log_normalized",
        "boost_available",
        "boost_cost_phase",
        "hunger",
        "episode_progress",
        "alive_fraction",
        "mass_rank_percentile",
        "wall_ahead",
        "wall_right",
        "wall_behind",
        "wall_left",
        "food_beyond_dx",
        "food_beyond_dy",
        "food_beyond_distance",
        "enemy1_dx",
        "enemy1_dy",
        "enemy1_size_ratio",
        "enemy1_boosting",
        "enemy2_dx",
        "enemy2_dy",
        "enemy2_size_ratio",
        "enemy2_boosting",
        "world_x",
        "world_y",
        "arena_type_flag",
    ),
    coordinate_frame="ego_heading",
    tactical_origin=(23, 15),
    strategic_origin=(12, 12),
    strategic_cell_size=5,
    rotation="exact_rot90_heading_to_up",
    paint_rule="wall>own_head>enemy_head>enemy_body>own_body>prediction>corpse_food>ambient_food",
    same_type_tie_rule="max_value_byte",
    out_of_world_prediction="discard",
    action_interpretation="relative6",
    supported_arena_types=("rectangular",),
    value_scales={
        "body_ttl": 255.0,
        "food_mass": 255.0,
        "enemy_size_ratio_cap": 2.0,
        "wall_distance_cells": 64.0,
    },
)
