"""Tests for explicit raster augmentation selection."""

import numpy as np
import pytest

from src.training.raster_augmentation import apply_raster_augmentation


def test_none_is_a_copying_identity():
    observation = {"tactical_uint8": np.arange(12, dtype=np.uint8).reshape(1, 1, 1, 3, 4)}
    result = apply_raster_augmentation(observation, mode="none")
    assert np.array_equal(result["tactical_uint8"], observation["tactical_uint8"])
    assert result["tactical_uint8"] is not observation["tactical_uint8"]


def test_legacy_flip_is_explicit_and_unknown_modes_fail():
    observation = {"tactical_uint8": np.array([[[[[1, 2, 3]]]]], dtype=np.uint8)}
    assert apply_raster_augmentation(observation, mode="legacy-horizontal-flip")[
        "tactical_uint8"
    ].tolist() == [[[[[3, 2, 1]]]]]
    with pytest.raises(ValueError, match="Unsupported"):
        apply_raster_augmentation(observation, mode="flip")
