"""Focused tests for B-3's stored-raster action geometry."""

import numpy as np
import pytest

from research.sandbox_b_heads.geometry_b3 import (
    FEATURE_NAMES,
    N_ACTION_FEATURES,
    action_geometry,
    flatten_action_geometry,
    geometry_features,
)


def _row(*, boost: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    local = np.zeros((1, 7, 31, 31), dtype=np.uint8)
    scalars = np.zeros((1, 60), dtype=np.float32)
    scalars[0, 1] = 0.01  # length 10
    scalars[0, 3] = boost
    return local, scalars


def test_open_raster_has_all_legal_actions_and_unavailable_boost_matches_normal():
    local, scalars = _row()
    actual = action_geometry(local, scalars)
    assert actual.shape == (1, 6, N_ACTION_FEATURES)
    assert actual.dtype == np.float32
    assert FEATURE_NAMES[0] == "immediate_legal"
    assert np.all(actual[0, :, 0] == 1)
    assert np.array_equal(actual[0, :3, :7], actual[0, 3:, :7])
    assert np.all(actual[0, :, 6] == 0)
    assert flatten_action_geometry(local, scalars).shape == (1, 6 * N_ACTION_FEATURES)
    calls = []
    assert np.array_equal(
        geometry_features({"local": local, "scalars": scalars}, lambda: calls.append(True)), actual
    )
    assert calls == [True]


def test_boost_checks_second_landing_cell_and_marks_only_effective_boosts():
    local, scalars = _row(boost=1.0)
    # Straight's first cell is (22, 15); wall its second cell (21, 15).
    local[0, 4, 21, 15] = 255
    actual = action_geometry(local, scalars)[0]
    assert actual[1, 0] == 1  # normal straight
    assert actual[4, 0] == 0  # boost straight hits the second cell
    assert actual[3, 6] == actual[5, 6] == 1
    assert actual[4, 6] == 0  # illegal action exposes no post-move values


def test_own_ttl_releases_for_future_arrival_but_enemy_never_does():
    local, scalars = _row()
    # A TTL-one cell is usable at one step, so left is legal.  Enemy TTL is static.
    local[0, 0, 23, 14] = 1
    local[0, 1, 23, 16] = 1
    actual = action_geometry(local, scalars)[0]
    assert actual[0, 0] == 1
    assert actual[2, 0] == 0
    assert actual[0, 3] == 1  # reached a tail-release cell


def test_boosted_dead_end_does_not_flood_back_through_new_neck_or_tail():
    local, scalars = _row(boost=1.0)
    # A boosted straight action crosses (22, 15) and lands at (21, 15).
    # Its landing pocket has three walls; the fourth neighbour is the just-made
    # neck.  That neck begins as a visible TTL-one tail cell, so it must neither
    # raise the landing degree nor manufacture tail reachability.
    local[0, 0, 22, 15] = 1
    local[0, 4, 20, 15] = 255
    local[0, 4, 21, 14] = 255
    local[0, 4, 21, 16] = 255

    actual = action_geometry(local, scalars)[0, 4]

    assert actual[0] == 1
    assert actual[1] == pytest.approx(0.1)  # area 1 / stored length 10
    assert actual[2] == pytest.approx(1.0 / (31 * 31))
    assert actual[3] == 0
    assert actual[4] == 0
    assert actual[5] == 1


def test_unboosted_dead_end_does_not_flood_back_through_old_head():
    local, scalars = _row()
    # Normal straight lands at (22, 15).  The only apparent exit is the old
    # head at (23, 15), which becomes own body at t=1 and remains blocked.
    local[0, 0, 23, 15] = 1
    local[0, 4, 21, 15] = 255
    local[0, 4, 22, 14] = 255
    local[0, 4, 22, 16] = 255

    actual = action_geometry(local, scalars)[0, 1]

    assert actual[0] == 1
    assert actual[1] == pytest.approx(0.1)
    assert actual[2] == pytest.approx(1.0 / (31 * 31))
    assert actual[3] == 0
    assert actual[4] == 0


def test_float64_scalars_are_normalized_to_float32_output():
    local, scalars = _row()
    actual = action_geometry(local, scalars.astype(np.float64))
    assert actual.dtype == np.float32


def test_preserves_existing_action_major_b8_v8_scalars_without_reconstruction():
    local, scalars = _row()
    scalars[0, 30:60] = np.arange(30, dtype=np.float32)
    actual = action_geometry(local, scalars)
    assert np.array_equal(actual[0, :, 7:], np.arange(30, dtype=np.float32).reshape(6, 5))


def test_bad_input_is_rejected():
    local, scalars = _row()
    with pytest.raises(ValueError, match="B8 observations"):
        action_geometry(local[:, :6], scalars)
    with pytest.raises(ValueError, match="aligned"):
        action_geometry(local, scalars[:, :3])
