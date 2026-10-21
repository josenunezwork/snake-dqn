"""Focused unit coverage for E3 deterministic feature reconstruction."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from research.bl_info_value.reconstruct import (
    ReconstructionError,
    _check_row,
    _exact,
    _geometry,
    _history_state,
    _history_window,
    _reconstruct_trace_world,
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


def test_compact_row_requires_original_dtype_shape_bytes_action_and_frame() -> None:
    from research.bl_info_value.compact import array_binding

    obs = {
        name: np.array([[1, 2]], dtype=dtype)
        for name, dtype in (
            ("local", "uint8"),
            ("global", "uint8"),
            ("scalars", "float32"),
            ("mask", "bool"),
        )
    }
    row = dict(
        obs_hashes={name: array_binding(value[0]) for name, value in obs.items()},
        action=1,
        frame=5,
    )
    _check_row(row, obs, 1, 4, 5)
    for field in obs:
        changed = dict(obs)
        changed[field] = obs[field].astype("float64")
        with pytest.raises(ReconstructionError, match="identity mismatch"):
            _check_row(row, changed, 1, 4, 5)
        changed[field] = obs[field].reshape((1, 1, 2))
        with pytest.raises(ReconstructionError, match="identity mismatch"):
            _check_row(row, changed, 1, 4, 5)
    with pytest.raises(ReconstructionError, match="action/frame"):
        _check_row(row, obs, 2, 4, 5)
    with pytest.raises(ReconstructionError, match="action/frame"):
        _check_row(row, obs, 1, 4, 6)


def test_trace_refuses_tampering_before_world_creation(tmp_path) -> None:
    from research.bl_info_value.io import sha256

    path = tmp_path / "trace.npz"
    np.savez_compressed(path, actions=np.ones((3, 6), dtype=np.uint8))
    task = dict(
        depth=300,
        indices=[0],
        row_bindings=[dict(decision=2)],
        root_state_hashes={"2": "a" * 64},
        mac_controls_verified=1,
        trace_npz=str(path),
        trace_sha256=sha256(path),
    )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ReconstructionError, match="trace SHA"):
        _reconstruct_trace_world(task)


@pytest.mark.parametrize("actions", [np.ones((3, 5), np.uint8), np.full((3, 6), 6, np.uint8)])
def test_trace_refuses_incomplete_seats_or_invalid_actions(tmp_path, actions) -> None:
    from research.bl_info_value.io import sha256

    path = tmp_path / "trace.npz"
    np.savez_compressed(path, actions=actions)
    task = dict(
        depth=300,
        indices=[0],
        row_bindings=[dict(decision=2)],
        root_state_hashes={"2": "a" * 64},
        mac_controls_verified=1,
        trace_npz=str(path),
        trace_sha256=sha256(path),
    )
    with pytest.raises(ReconstructionError, match="shape/dtype/range"):
        _reconstruct_trace_world(task)
