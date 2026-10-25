"""Synthetic-only checks for the R30 timing forecast; no scientific payloads."""

from __future__ import annotations

import copy

import pytest

from research.bl_round30 import timing, timing_forecast


def _report() -> dict:
    from research.bl_round30.timing_pack import identity_paths

    phases = {name: 1.0 for name in timing.PHASES}
    phases.update(
        {
            "adam_0_10": 2.0,
            "adam_10_20": 3.0,
            "adam_20_30": 2.5,
            "closure_0": 4.0,
            "closure_10": 5.0,
            "closure_30": 4.5,
            "bookkeeping_probe": 20.0,
        }
    )
    return {
        "schema": timing.SCHEMA,
        "complete": True,
        "synthetic": False,
        "ruling_sha256": timing.RULING_SHA,
        "config_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "source_hashes": {path: "c" * 64 for path in identity_paths()},
        "input_hashes": {
            "train.npz": "d" * 64,
            "b8.pth": "e" * 64,
            "sampler-plan.npz": "f" * 64,
            "sampler-plan.json": "0" * 64,
        },
        "control_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "schedule": timing.SCHEDULE,
        "device": "cuda",
        "threads": 1,
        "row_count": 2,
        "feature_shape": [2, 256],
        "parameter_count": 1542,
        "phase_seconds": phases,
        "worker_seconds": 100.0,
    }


def _operational(report: dict) -> dict:
    receipt = {
        "status": "PASS",
        "stop_verified": True,
        "cleanup_error": None,
        "elapsed_seconds": 130.0,
        "manifest_sha256": "3" * 64,
        "commit": report["source_commit"],
        "job_id": "r30-timing-fixture",
        "generation": "fixture-1",
        "terminal": {
            "status": "succeeded",
            "returncode": 0,
            "job_id": "r30-timing-fixture",
            "generation": "fixture-1",
        },
        "exports": [
            {
                "path": "timing.json",
                "sha256": timing_forecast.digest_bytes(report),
                "size_bytes": 1,
            }
        ],
    }
    return {
        "schema": timing_forecast.OPERATIONAL_SCHEMA,
        "ruling_sha256": timing.RULING_SHA,
        "source_commit": report["source_commit"],
        "control_sha256": report["control_sha256"],
        "input_hashes": report["input_hashes"],
        "timing_report_sha256": timing_forecast.digest_bytes(report),
        "supervisor_receipt_sha256": "5" * 64,
        "supervisor_receipt": receipt,
        "calibration_manifest_sha256": "3" * 64,
        "timing_config_sha256": report["config_sha256"],
        "calibration_required_exports": ["timing.json"],
        "main_output_transfer_seconds": 150.0,
        "teardown_upper_seconds": 10.0,
        "transfer_upper_seconds": 10.0,
        "calibration_teardown_seconds": 10.0,
        "staging_upper_seconds": 15.0,
        "independent_review_status": "ACCEPTED",
        "independent_review_sha256": "4" * 64,
        "g4_status": "PENDING",
        "g5_status": "PENDING_ACCEPT",
    }


def test_forecast_uses_frozen_conservative_formula_once():
    report = _report()
    result = timing_forecast.forecast(report, _operational(report))
    # fixed phases 13, Adam 600, closures 2500, bookkeeping 300, certificate
    # + export 2, worker residual 44, supervisor residual 30, output 150.
    assert result["measured_est_seconds"] == 3639.0
    assert result["within_main_ceiling"] is False
    assert result["g4_status"] == "PENDING_INDEPENDENT_REVIEW"
    assert result["g5_status"] == "PENDING_ACCEPT"


@pytest.mark.parametrize("key", ["main_output_transfer_seconds", "teardown_upper_seconds"])
def test_forecast_refuses_missing_independent_main_cost(key):
    report = _report()
    operational = _operational(report)
    del operational[key]
    with pytest.raises(ValueError, match="keys differ"):
        timing_forecast.forecast(report, operational)


def test_forecast_refuses_incomplete_or_unreconciled_calibration():
    report = _report()
    report["complete"] = False
    with pytest.raises(ValueError, match="identity differs"):
        timing_forecast.forecast(report, _operational(report))
    report = _report()
    operational = _operational(report)
    operational["supervisor_receipt"]["elapsed_seconds"] = 99.0
    with pytest.raises(ValueError, match="reconciled"):
        timing_forecast.forecast(report, operational)


def test_forecast_refuses_identity_or_terminal_receipt_drift():
    report = _report()
    operational = _operational(report)
    operational["input_hashes"] = copy.deepcopy(report["input_hashes"])
    operational["input_hashes"]["train.npz"] = "9" * 64
    with pytest.raises(ValueError, match="input binding"):
        timing_forecast.forecast(report, operational)
    operational = _operational(report)
    operational["supervisor_receipt"]["stop_verified"] = False
    with pytest.raises(ValueError, match="terminal PASS"):
        timing_forecast.forecast(report, operational)


def test_forecast_refuses_wrong_export_or_unreconciled_transport():
    report = _report()
    operational = _operational(report)
    operational["supervisor_receipt"]["exports"][0]["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="timing export"):
        timing_forecast.forecast(report, operational)
    operational = _operational(report)
    operational["staging_upper_seconds"] = 31.0
    with pytest.raises(ValueError, match="reconciled"):
        timing_forecast.forecast(report, operational)


def test_forecast_refuses_g4_or_g5_adoption():
    report = _report()
    operational = _operational(report)
    operational["g4_status"] = "ACCEPT"
    with pytest.raises(ValueError, match="G4"):
        timing_forecast.forecast(report, operational)
    operational = _operational(report)
    operational["g5_status"] = "ACCEPT"
    with pytest.raises(ValueError, match="G5"):
        timing_forecast.forecast(report, operational)


def test_exact_main_ceiling_is_accepted():
    report = _report()
    report["phase_seconds"] = {name: 0.0 for name in timing.PHASES}
    report["worker_seconds"] = 1.0
    operational = _operational(report)
    operational["timing_report_sha256"] = timing_forecast.digest_bytes(report)
    operational["supervisor_receipt"]["elapsed_seconds"] = 16.0
    operational["supervisor_receipt"]["exports"][0]["sha256"] = timing_forecast.digest_bytes(report)
    operational["main_output_transfer_seconds"] = 2864.0
    result = timing_forecast.forecast(report, operational)
    assert result["measured_est_seconds"] == 3180.0
    assert result["within_main_ceiling"] is True
