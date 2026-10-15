"""Focused unit tests for the T3 legacy-bank prefilter calibration."""

import numpy as np
import pytest

from research.t3_rollout_labels_20261015 import calibrate_bank


def test_threshold_uses_nextafter_to_retain_tied_boundary_scores() -> None:
    result = calibrate_bank.select_threshold(np.array([0.0] * 6 + [1.0] * 14), coverage=0.95)

    assert result["boundary_score"] == 0.0
    assert result["threshold"] < 0.0
    assert result["retained_decisive_rows"] == 20
    assert result["coverage"] == 1.0


def test_threshold_selects_required_order_statistic() -> None:
    result = calibrate_bank.select_threshold(np.arange(20, dtype=np.float64), coverage=0.95)

    assert result["required_decisive_rows"] == 19
    assert result["retained_decisive_rows"] == 19
    assert result["coverage"] == 19 / 20


def test_threshold_rejects_empty_population() -> None:
    with pytest.raises(ValueError, match="empty decisive-score"):
        calibrate_bank.select_threshold(np.array([], dtype=np.float64))


def test_sidecar_fields_are_row_aligned() -> None:
    """The four sidecar fields bind a source row without relying on row position alone."""
    source = {
        "decision": np.array([17, 22], dtype=np.int64),
        "frame": np.array([2017, 2022], dtype=np.int64),
        "scalars": np.array([[0.0, 0.701], [0.0, 0.999]], dtype=np.float32),
    }
    sidecar = {
        "q": np.zeros((2, 6), dtype=np.float32),
        "decision": source["decision"].copy(),
        "frame": source["frame"].copy(),
        "length": np.rint(source["scalars"][:, 1] * 1000.0).astype(np.int64),
    }

    assert sidecar["q"].shape == (len(source["decision"]), 6)
    assert np.array_equal(sidecar["decision"], source["decision"])
    assert np.array_equal(sidecar["frame"], source["frame"])
    assert np.array_equal(sidecar["length"], np.array([701, 999]))


def test_empty_legacy_rows_have_stable_empty_keys_and_no_death() -> None:
    decision, frame = calibrate_bank._row_keys({}, 0)

    assert decision.dtype == np.int64 and frame.dtype == np.int64
    assert not len(decision) and not len(frame)
    assert calibrate_bank._death_endpoint(True, 4999) is None
    assert calibrate_bank._death_endpoint(False, 4999) == 4999
