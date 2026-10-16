"""Focused pure contracts for the T4a/T3 boost reconciliation tool."""

from __future__ import annotations

import numpy as np
import pytest

from research.t4a_posthoc_20261016 import boost


def test_greedy_definitions_preserve_raw_and_normal_fallback() -> None:
    q = np.array([[1, 2, 3, 90, 91, 92], [1, 2, 3, 4, 5, 6]], dtype=np.float32)
    mask = np.array([[False] * 6, [True, False, False, False, False, False]])

    np.testing.assert_array_equal(boost._greedy(q, None), np.array([5, 5], dtype=np.int8))
    np.testing.assert_array_equal(boost._greedy(q, mask), np.array([2, 0], dtype=np.int8))


def test_buckets_follow_the_registered_boundaries() -> None:
    np.testing.assert_array_equal(
        boost._bucket(np.array([0, 399, 400, 699, 700, 999, 1000])),
        np.array([0, 0, 1, 1, 2, 2, 3], dtype=np.int8),
    )


def test_summary_keeps_action_definitions_and_own_shares_separate() -> None:
    arrays = {
        "length": np.array([399, 400, 700, 1000], dtype=np.int32),
        "raw_action": np.array([3, 0, 4, 1], dtype=np.int8),
        "resolved_action": np.array([0, 3, 1, 5], dtype=np.int8),
        "advisory_action": np.array([2, 4, 1, 2], dtype=np.int8),
    }
    summary = boost.summarize_arrays(arrays)

    assert summary["decisions"] == 4
    assert summary["by_length"]["lt400"]["own_decision_share"] == pytest.approx(0.25)
    assert summary["by_length"]["lt400"]["raw"]["rate"] == 1.0
    assert summary["by_length"]["lt400"]["resolved"]["rate"] == 0.0
    assert summary["by_length"]["lt400"]["advisory"]["rate"] == 0.0
    assert summary["by_length"]["400_699"]["resolved"]["rate"] == 1.0
    assert summary["by_length"]["700_999"]["resolved"]["rate"] == 0.0
    assert summary["by_length"]["ge1000"]["resolved"]["rate"] == 1.0


def test_append_report_refuses_duplicate_section(tmp_path) -> None:
    report = tmp_path / "diagnostics.md"
    report.write_text("# Existing\n")
    summary = {"arms": {arm: {} for arm in ("CTRL", "LABELS", "LABELS_KEEPV8")}}
    # Rendering requires actual seed summaries; duplicate protection happens before rendering.
    report.write_text("# Existing\n\n## 4. Boost reconciliation (T4a post-hoc)\n")
    with pytest.raises(ValueError, match="already contains"):
        boost.append_report(report, summary)
