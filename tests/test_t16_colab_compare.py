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
                {
                    "name": "selfplay_w128_a4",
                    "status": "completed",
                    "transitions_per_second": 30000,
                    "tail_transitions_per_second": 29000,
                    "resources": {"gpu_util_pct_mean": 40.0},
                },
            ],
        }
    )

    assert report["threshold_transitions_per_second"] == 21600
    assert [row["verdict"] for row in report["configs"][:2]] == ["PASS", "FAIL"]
    self_play = report["configs"][5]
    assert self_play["verdict"] == "PASS" and "caveat" in self_play
    assert self_play["tail_transitions_per_second"] == 29000
    assert self_play["resources"]["gpu_util_pct_mean"] == 40.0
    assert all(row["runner_status"] != "missing" for row in report["configs"])


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


def test_other_matrix_requires_its_own_recorded_names():
    payload = {
        "schema": compare.SCHEMA,
        "status": "completed",
        "matrix": "smoke03",
        "matrix_names": ["sp_w128_a4_pack", "sp_w128_a4_raw"],
        "configs": [
            {"name": "sp_w128_a4_pack", "status": "completed", "transitions_per_second": 11000.0}
        ],
    }
    report = compare.compare_results(payload)
    rows = {row["name"]: row for row in report["configs"]}
    assert rows["sp_w128_a4_pack"]["verdict"] != "PASS"
    assert rows["sp_w128_a4_raw"]["runner_status"] == "missing"
    assert not set(compare.REQUIRED_CONFIG_NAMES) & set(rows)
    assert compare.required_config_names({"matrix": "smoke02"}) == compare.REQUIRED_CONFIG_NAMES
    assert compare.required_config_names({}) == compare.REQUIRED_CONFIG_NAMES
