"""Focused unit coverage for E3 deterministic feature reconstruction."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from research.bl_info_value.reconstruct import (
    ReconstructionError,
    _exact,
    _geometry,
    _history_state,
    _history_window,
)


class _Sim:
    grid_w = 5
    grid_h = 3
    cap = 8

    def __init__(self) -> None:
        self._owner = np.full((1, 3, 5), -1, dtype=np.int16)
        self._slot = np.zeros((1, 3, 5), dtype=np.int32)
        self.alive = np.array([[True, True, False, False, False, False]])
        self.head_ptr = np.array([[3, 5, 0, 0, 0, 0]], dtype=np.int64)
        self.seg_count = np.array([[4, 6, 0, 0, 0, 0]], dtype=np.int64)
        self.bodies = np.zeros((1, 6, 8, 2), dtype=np.int64)
        self.bodies[0, 0, 3] = (4, 2)
        self.bodies[0, 1, 5] = (1, 1)
        self.direction = np.array([[1, 0, 0, 0, 0, 0]], dtype=np.int64)
        self.length = np.array([[1200, 50, 0, 0, 0, 0]], dtype=np.int64)
        self._owner[0, 2, 4] = 0
        self._slot[0, 2, 4] = 3
        self._owner[0, 1, 1] = 1
        self._slot[0, 1, 1] = 1


def test_geometry_keeps_owner_planes_and_unclipped_release_order() -> None:
    sim = _Sim()
    geometry = _geometry(sim)
    assert geometry.shape == (7, 3, 5)
    assert geometry.dtype == np.float32
    assert geometry[0, 2, 4] == 1.0
    assert geometry[1, 1, 1] == 1.0
    # Seat 1's ring offset is (5 - 1) % 8 == 4, so it releases in two steps.
    assert geometry[6, 1, 1] == pytest.approx(0.002)
    assert geometry[6, 2, 4] == pytest.approx(0.004)


def test_history_is_predecision_fixed_unit_and_left_padded() -> None:
    state = _history_state(_Sim(), np.array([[5, 1, 0, 0, 0, 0]], dtype=np.int64))
    assert state.shape == (43,)
    assert state[0] == pytest.approx(4 / 5)
    assert state[1] == pytest.approx(2 / 3)
    assert state[2:4].tolist() == [1.0, 0.0]
    assert state[4] == pytest.approx(1.2)
    assert state[6] == 1.0
    assert state[-1] == 1.0
    window = _history_window(deque([state], maxlen=128))
    assert not window[:-1].any()
    np.testing.assert_array_equal(window[-1], state)


def test_exact_rejects_float_or_byte_drift() -> None:
    value = np.array([1.0], dtype=np.float32)
    _exact("scalars", value, value.copy(), 3)
    with pytest.raises(ReconstructionError, match="identity mismatch"):
        _exact("scalars", value, np.array([1.00001], dtype=np.float32), 3)
