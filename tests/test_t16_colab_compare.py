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
                    "actor_exitcodes": [0, 0, 0, 0],
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
    assert self_play["actor_exitcodes"] == [0, 0, 0, 0]
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
    with pytest.raises(compare.ResultsError, match="unknown suite"):
        compare.compare_results({"schema": compare.SCHEMA, "suite": "unknown", "configs": []})


def test_smoke03_requires_update_rate_and_fixed_transitions_per_update():
    configs = [
        {
            "name": name,
            "status": "completed",
            "transitions_per_second": 21_600,
            "updates_per_second": 84,
            "transitions_per_update": 256,
            "policy_lag_mean": 3.0,
            "learn_seconds_per_update": 0.01,
        }
        for name in compare.SMOKE03_REQUIRED_CONFIG_NAMES
    ]
    configs[1]["updates_per_second"] = 83.9
    configs[2]["transitions_per_update"] = 250.87  # just outside 256 +/-2%
    configs[5]["batch_scale"] = 4
    configs[5]["transitions_per_update"] = 512
    configs[5]["throughput_only"] = True
    configs[6]["throughput_only"] = True
    report = compare.compare_results(
        {"schema": compare.SCHEMA, "suite": "smoke03", "status": "completed", "configs": configs}
    )
    assert [row["verdict"] for row in report["configs"]] == [
        "PASS",
        "FAIL",
        "INVALID",
        "PASS",
        "PASS",
        "THROUGHPUT_ONLY",
        "THROUGHPUT_ONLY",
    ]
    assert report["configs"][0]["threshold_updates_per_second"] == 84
    assert report["configs"][0]["policy_lag_mean"] == 3.0


@pytest.mark.parametrize("lag", [float("nan"), -1, "bad"])
def test_smoke03_rejects_nonfinite_or_invalid_policy_lag(lag):
    configs = [
        {
            "name": name,
            "status": "completed",
            "transitions_per_second": 21_600,
            "updates_per_second": 84,
            "transitions_per_update": 256,
            "policy_lag_mean": 3.0,
        }
        for name in compare.SMOKE03_REQUIRED_CONFIG_NAMES
    ]
    configs[1]["policy_lag_mean"] = lag
    report = compare.compare_results(
        {"schema": compare.SCHEMA, "suite": "smoke03", "status": "completed", "configs": configs}
    )
    assert report["configs"][1]["verdict"] == "INVALID"


def test_smoke03_requires_a_completed_unique_control_for_lag_comparison():
    configs = [
        {
            "name": name,
            "status": "completed",
            "transitions_per_second": 21_600,
            "updates_per_second": 84,
            "transitions_per_update": 256,
            "policy_lag_mean": 3.0,
        }
        for name in compare.SMOKE03_REQUIRED_CONFIG_NAMES
    ]
    configs[0]["status"] = "failed"
    report = compare.compare_results(
        {"schema": compare.SCHEMA, "suite": "smoke03", "status": "completed", "configs": configs}
    )
    assert report["configs"][1]["verdict"] == "FAIL"
    assert "packed control" in report["configs"][1]["reason"]


def test_smoke03_registered_batch_scale_cannot_be_forged_in_results_metadata():
    configs = [
        {
            "name": name,
            "status": "completed",
            "transitions_per_second": 21_600,
            "updates_per_second": 84,
            "transitions_per_update": 256,
            "policy_lag_mean": 3.0,
        }
        for name in compare.SMOKE03_REQUIRED_CONFIG_NAMES
    ]
    configs[0]["batch_scale"] = 4
    report = compare.compare_results(
        {"schema": compare.SCHEMA, "suite": "smoke03", "status": "completed", "configs": configs}
    )
    assert report["configs"][0]["expected_transitions_per_update"] == 256
    assert report["configs"][0]["verdict"] == "PASS"
