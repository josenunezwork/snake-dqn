"""Independent Part 1 artifact-audit contracts."""

from __future__ import annotations

import numpy as np
import pytest

from research.t4a_posthoc_20261016 import audit


def test_registered_capture_matrix_has_25_distinct_pairs() -> None:
    pairs = audit.expected_captures()
    assert len(pairs) == 25
    assert len(set(pairs)) == 25
    assert pairs.count(("CTRL_s10", "CTRL_s10")) == 1
    assert pairs.count(("LABELS_s10", "CTRL_s10")) == 1
    assert pairs.count(("LABELS_KEEPV8_s10", "CTRL_s10")) == 1


def test_bucket_and_action_counts_preserve_partition() -> None:
    lengths = np.array([0, 399, 400, 699, 700, 999, 1000], dtype=np.int32)
    assert audit.bucket_counts(lengths) == {
        "lt400": 2,
        "400_699": 2,
        "700_999": 2,
        "ge1000": 1,
    }
    summary = audit.action_counts(np.array([0, 3, 5, 5], dtype=np.int8))
    assert summary["counts"] == [1, 0, 0, 1, 0, 2]
    assert summary["boost_argmax"] == 3
    assert summary["boost_rate"] == pytest.approx(0.75)


def test_ctrl_comparison_joins_by_env_and_frame_and_checks_length() -> None:
    arrays = {
        "env": np.array([1, 0], dtype=np.int16),
        "frame": np.array([4, 2], dtype=np.int16),
        "length": np.array([12, 10], dtype=np.int32),
        "advisory_action": np.array([5, 3], dtype=np.int8),
    }
    prior = {
        "decision_arm": np.array([0, 0, 1, 1], dtype=np.int8),
        "decision_env": np.array([0, 1, 0, 1], dtype=np.int16),
        "decision_frame": np.array([2, 4, 8, 9], dtype=np.int16),
        "decision_length": np.array([10, 12, 20, 30], dtype=np.int32),
        "decision_arm_action": np.array([3, 5, 1, 2], dtype=np.int8),
    }
    result = audit.compare_ctrl_actions(arrays, prior, "LABELS")
    assert result == {
        "status": "pass",
        "rows": 2,
        "action_mismatches": 0,
        "length_mismatches": 0,
        "first_mismatch": None,
    }


def test_ctrl_comparison_reports_action_mismatch_without_fabricating_parity() -> None:
    arrays = {
        "env": np.array([0], dtype=np.int16),
        "frame": np.array([2], dtype=np.int16),
        "length": np.array([10], dtype=np.int32),
        "advisory_action": np.array([4], dtype=np.int8),
    }
    prior = {
        "decision_arm": np.array([0], dtype=np.int8),
        "decision_env": np.array([0], dtype=np.int16),
        "decision_frame": np.array([2], dtype=np.int16),
        "decision_length": np.array([10], dtype=np.int32),
        "decision_arm_action": np.array([3], dtype=np.int8),
    }
    result = audit.compare_ctrl_actions(arrays, prior, "LABELS")
    assert result["status"] == "fail"
    assert result["action_mismatches"] == 1
    assert result["length_mismatches"] == 0


def test_own_prelength_uses_one_frame_shift_and_leaves_initial_mass_unchecked() -> None:
    arrays = {
        "env": np.array([0, 0, 0], dtype=np.int16),
        "frame": np.array([1, 2, 3], dtype=np.int16),
        "length": np.array([99, 10, 11], dtype=np.int32),
    }
    prior = {
        "seeds": np.array([123], dtype=np.int64),
        "length": np.array([[10, 11, 0]], dtype=np.int32),
        "alive": np.array([[True, True, False]]),
    }
    result = audit.compare_own_prelength(arrays, prior)
    assert result["status"] == "pass"
    assert result["shifted_rows_checked"] == 2
    assert result["initial_prelength_rows_unchecked"] == 1
    assert result["initial_mass_comparison"].startswith("unavailable")


def test_own_prelength_rejects_post_terminal_shift() -> None:
    arrays = {
        "env": np.array([0, 0], dtype=np.int16),
        "frame": np.array([1, 3], dtype=np.int16),
        "length": np.array([99, 10], dtype=np.int32),
    }
    prior = {
        "seeds": np.array([123], dtype=np.int64),
        "length": np.array([[10, 0, 0]], dtype=np.int32),
        "alive": np.array([[True, False, False]]),
    }
    result = audit.compare_own_prelength(arrays, prior)
    assert result["status"] == "fail"
    assert result["terminal_shift_mismatches"] == 1
