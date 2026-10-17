"""T16 Colab throughput comparison contract tests."""

import pytest

from research.t16_colab import compare


def test_completed_configs_are_compared_with_the_4x_threshold():
    report = compare.compare_results(
        {
            "schema": compare.SCHEMA,
            "status": "incomplete",
            "configs": [
                {"name": "hero_w128_a2", "status": "completed", "transitions_per_second": 21600},
                {"name": "hero_w128_a4", "status": "completed", "transitions_per_second": 21599.9},
                {"name": "hero_w128_a8", "status": "completed", "transitions_per_second": 1},
                {"name": "selfplay_w64_a2", "status": "completed", "transitions_per_second": 1},
                {"name": "selfplay_w128_a2", "status": "completed", "transitions_per_second": 1},
            ],
        }
    )

    assert report["threshold_transitions_per_second"] == 21600
    assert [row["verdict"] for row in report["configs"][:2]] == ["PASS", "FAIL"]


def test_failed_or_missing_configs_never_pass():
    report = compare.compare_results(
        {
            "schema": compare.SCHEMA,
            "status": "failed",
            "configs": [
                {"name": "hero_w128_a2", "status": "completed", "transitions_per_second": 999999},
                {"name": "hero_w128_a2", "status": "completed", "transitions_per_second": 999999},
            ],
        }
    )

    assert [row["verdict"] for row in report["configs"][:2]] == ["INVALID", "INVALID"]
    missing = {row["name"] for row in report["configs"] if row["runner_status"] == "missing"}
    assert set(compare.REQUIRED_CONFIG_NAMES[1:]) <= missing


def test_wrong_schema_is_not_compared():
    with pytest.raises(compare.ResultsError, match="schema"):
        compare.compare_results({"schema": "wrong", "configs": []})
