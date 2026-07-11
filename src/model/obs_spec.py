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
The trainer records these under :data:`RASTER_SHAPE_KEYS`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "OBS_SPEC_KEY",
    "VECTOR61",
    "RASTER31V2",
    "DEFAULT_OBS_SPEC",
    "KNOWN_OBS_SPECS",
    "RasterObsShapes",
    "RASTER31V2_SHAPES",
    "RASTER_SHAPE_KEYS",
]

# -- metadata key ----------------------------------------------------------
#: Checkpoint metadata key under which the obs_spec string is stored.
OBS_SPEC_KEY = "obs_spec"

# -- obs_spec string constants ---------------------------------------------
#: Legacy 61-D (or 58-D) free-space vector consumed by ``ApexNetwork``.
VECTOR61 = "vector61"
#: Dual-scale ego-raster (blueprint §2) consumed by ``RasterDuelingNetwork``.
RASTER31V2 = "raster31v2"

#: Spec assumed when a checkpoint omits :data:`OBS_SPEC_KEY` — keeps every
#: pre-contract-v2 champion loadable unchanged.
DEFAULT_OBS_SPEC = VECTOR61

#: All obs_spec strings this build understands.
KNOWN_OBS_SPECS = (VECTOR61, RASTER31V2)


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

#: Flat metadata keys that carry the raster shape descriptor.
RASTER_SHAPE_KEYS = (
    "tactical_channels",
    "tactical_size",
    "strategic_channels",
    "strategic_size",
    "scalars",
)
