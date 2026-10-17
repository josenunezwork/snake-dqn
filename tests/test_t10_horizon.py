"""Focused label, identity and feature-contract checks for the T10 horizon fit."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from research.t10_certificate_horizon_20261017 import horizon  # noqa: E402


def _data() -> dict[str, np.ndarray]:
    return {
        "state_id": np.array(["s0", "s1"]),
        "scalars": np.zeros((2, 60), dtype=np.float32),
        "features": np.zeros((2, 24), dtype=np.float32),
        "legal": np.array([[True, True, False, True, True, True], [True] * 6]),
        "died": np.array([[True, True, False, True, False, False], [False] * 6]),
        "horizon": np.array([[30, 300, 0, 100, 300, 300], [300] * 6], dtype=np.int16),
    }


def test_death_within_uses_known_d300_outcomes_and_includes_boundary() -> None:
    data = _data()
    d30, d60, d100, d200, d300 = (horizon.death_within(data, value) for value in horizon.HORIZONS)
    assert d30[0, 0] == 1
    assert d60[0, 0] == 1
    assert d60[0, 1] == 0
    assert d100[0, 3] == 1
    assert d200[0, 3] == 1
    assert d300[0, 1] == 1
    assert d300[0, 4] == 0
    with pytest.raises(ValueError, match="unsupported"):
        horizon.death_within(data, 31)


def test_action_rows_uses_only_legal_actions_and_appends_identity() -> None:
    data = _data()
    matrix = np.ones((2, 3), dtype=np.float32)
    x, y, state, action = horizon.action_rows(
        matrix, data["legal"], horizon.death_within(data, 100)
    )
    assert x.shape == (11, 9)
    assert not np.any((state == 0) & (action == 2))
    np.testing.assert_array_equal(x[:, -6:], np.eye(6)[action])
    np.testing.assert_array_equal(y, horizon.death_within(data, 100)[state, action])


def test_designs_uses_arrival_and_fuse_without_action_until_row_expansion() -> None:
    data = _data()
    arrival = np.ones((2, 18), dtype=np.float32)
    fuse = np.ones((2, 262), dtype=np.float32)
    got = horizon.designs(data, arrival, {"fuse_q_B8_s0": fuse})
    assert {name: value.shape[1] for name, value in got.items()} == {
        "60": 60,
        "60+24": 84,
        "60+24+18_arrival": 102,
        "fuse_q_B8_s0": 262,
    }


def test_load_fuse_requires_canonical_state_identity(tmp_path) -> None:
    data = _data()
    path = tmp_path / "fuse.npz"
    np.savez(
        path,
        fuse=np.zeros((2, 256), np.float32),
        q=np.zeros((2, 6), np.float32),
        state_id=data["state_id"],
    )
    assert horizon.load_fuse(data, path).shape == (2, 262)
    np.savez(
        path,
        fuse=np.zeros((2, 256), np.float32),
        q=np.zeros((2, 6), np.float32),
        state_id=data["state_id"][::-1],
    )
    with pytest.raises(ValueError, match="identities"):
        horizon.load_fuse(data, path)


def test_named_fuse_paths_uses_final_directory_and_rejects_duplicates(tmp_path) -> None:
    left = tmp_path / "B8_s0" / "B8_s0.features.npz"
    right = tmp_path / "B8_s1" / "B8_s1.features.npz"
    left.parent.mkdir()
    right.parent.mkdir()
    assert horizon.named_fuse_paths([left, right]) == {
        "fuse_q_B8_s0": left,
        "fuse_q_B8_s1": right,
    }
    with pytest.raises(ValueError, match="duplicate"):
        horizon.named_fuse_paths([left, left])
