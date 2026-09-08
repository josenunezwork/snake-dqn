"""Explicit raster augmentation recipes, isolated from the training loop."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def apply_raster_augmentation(
    observation: Mapping[str, np.ndarray], *, mode: str = "none"
) -> dict[str, np.ndarray]:
    """Return an explicitly selected augmentation of an observation batch.

    ``none`` is the corrected-v3 recipe. ``legacy-horizontal-flip`` preserves
    the historical tensor operation for experiments; it makes no claim that a
    world-coordinate reflection is physically symmetric.
    """
    if mode == "none":
        return {key: np.asarray(value).copy() for key, value in observation.items()}
    if mode != "legacy-horizontal-flip":
        raise ValueError(f"Unsupported raster augmentation mode {mode!r}")
    output = {key: np.asarray(value).copy() for key, value in observation.items()}
    for key in ("tactical_uint8", "strategic_uint8", "tactical", "strategic"):
        if key in output:
            output[key] = np.flip(output[key], axis=-1).copy()
    return output
