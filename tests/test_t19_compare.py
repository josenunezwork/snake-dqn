"""N=3 timing cannot inherit an unrelated-N speed qualification."""

import copy

import pytest

from research.t16_colab import compare, run_smoke


def payload():
    return {
        "schema": compare.SCHEMA,
        "suite": "n3",
        "status": "completed",
        "configs": [
            {
                **config,
                "status": "completed",
                "transitions_per_second": 100_000,
                "updates_per_second": 100_000 / 256,
                "transitions_per_update": 256,
                "learner_decisions": 100_000,
                "replay_rows": 99_840,
                "measured_updates": 390,
                "live_rows_by_policy": {"b8-s3": 123},
                "policy_lag_mean": 2.5,
                "policy_lag_p95": 5,
                "frozen_forward_gpu_seconds": 1.25,
            }
            for config in run_smoke.N3_CONFIGS
        ],
    }


def test_n3_valid_measurements_need_same_n_baseline():
    report = compare.compare_results(payload())
    assert report["baseline_transitions_per_second"] is None
    assert {row["verdict"] for row in report["configs"]} == {"MEASURED"}
    assert all(row["speedup_vs_mac"] is None for row in report["configs"])


def test_real_telemetry_aggregation_produces_a_valid_n3_comparison():
    logs, telemetry = [], []
    for i in range(3):
        logs.append(
            {
                "wall": 40 + i * 40,
                "transitions": 1000 + i * 1024,
                "learner_decisions": 1000 + i * 1024,
                "replay_rows": 900 + i * 1024,
                "updates": 3 + i * 4,
                "per_policy_live_rows": {"learner": 1000 + i * 1024, "b8-s3": 100 + i * 100},
            }
        )
        telemetry.append(
            {
                "transitions": logs[-1]["transitions"],
                "policy_lag_mean": 2,
                "policy_lag_p95": 2,
                "policy_lag_histogram": {"2": 20},
                "frozen_forward_gpu_s": i * 0.25,
            }
        )
    measured = run_smoke.measured_rate(
        run_smoke.merge_telemetry_rows(logs, telemetry),
        warm_seconds=0,
        settle_seconds=0,
    )
    data = payload()
    for row in data["configs"]:
        row.update(measured)
    assert all(row["verdict"] == "MEASURED" for row in compare.compare_results(data)["configs"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("learner_decisions", None),
        ("replay_rows", -1),
        ("measured_updates", True),
        ("policy_lag_p95", float("nan")),
        ("frozen_forward_gpu_seconds", None),
        ("live_rows_by_policy", {}),
        ("live_rows_by_policy", {"b8": -1}),
        ("learner_seats", 6),
        ("transitions_per_update", 512),
    ],
)
def test_n3_incomplete_or_wrong_workload_rows_are_invalid(key, value):
    data = payload()
    data["configs"][0][key] = value
    assert compare.compare_results(data)["configs"][0]["verdict"] == "INVALID"


def test_n3_missing_duplicate_and_failed_rows_are_invalid():
    data = payload()
    data["configs"].pop()
    assert compare.compare_results(data)["configs"][-1]["verdict"] == "INVALID"
    data["configs"].append(copy.deepcopy(data["configs"][0]))
    assert all(row["verdict"] == "INVALID" for row in compare.compare_results(data)["configs"])
    data = payload()
    data["status"] = "failed"
    assert all(row["verdict"] == "INVALID" for row in compare.compare_results(data)["configs"])
