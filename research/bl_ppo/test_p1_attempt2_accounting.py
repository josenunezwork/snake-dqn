"""Attempt-2 P1 accounting preserves a failed receipt without faking terminal PASS."""

import json
from pathlib import Path

import pytest

from research.bl_ppo import p1_attempt2_accounting as accounting
from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.p1_attempt2_accounting import (
    cumulative_admission,
    record_failed_repair,
    verify_cumulative_admission,
)


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def failure_inputs(root: Path) -> tuple[Path, Path]:
    receipt = write_json(
        root / "failed-receipt.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "terminal": None,
            "elapsed_seconds": 384.6450653329957,
            "accounting": {"own_rate_units_per_hour": 1.54},
            "usage_before": {"balance": 1098.49},
            "usage_after": {"balance": 1098.39},
        },
    )
    report = root / "failure-report.json"
    export_json(
        report,
        {
            "schema": "bl-p1-repair-failure-independent-codex-v/v1",
            "status": "VERIFIED_NO_RETRY_UNDER_CURRENT_RULES",
            "accounting": {
                "legacy_units": 10.11,
                "failed_repair": {
                    "own_units": 0.16454261128133707,
                    "gross_units": 0.10,
                    "conservative_rounded_units": 0.17,
                },
            },
            "evidence": [{"path": str(receipt.resolve()), "sha256": digest(receipt)}],
        },
    )
    return report, receipt


def record(monkeypatch: pytest.MonkeyPatch, report: Path, receipt: Path, output: Path) -> None:
    monkeypatch.setattr(accounting, "FAILURE_REPORT_SHA256", digest(report))
    monkeypatch.setattr(accounting, "FAILED_RECEIPT_SHA256", digest(receipt))
    record_failed_repair(failure_report=report, failed_receipt=receipt, output=output)


def reconciliation(root: Path, debit: float = 0.0) -> Path:
    path = root / "reconciliation.json"
    export_json(
        path,
        {
            "schema": "bl-p1-attempt2-live-reconciliation/v1",
            "status": "PASS",
            "warning": "WARNING_UNEXPLAINED_DEBIT",
            "additional_attributable_debit_units": debit,
        },
    )
    return path


def manifest(root: Path) -> Path:
    path = root / "held-manifest.json"
    export_json(path, {"commit": "a" * 40})
    return path


def successful_canary(root: Path, *, balance_after: float = 99.77) -> Path:
    return write_json(
        root / "canary-receipt.json",
        {
            "status": "PASS",
            "stop_verified": True,
            "cleanup_error": None,
            "error": None,
            "terminal": {"status": "succeeded", "returncode": 0},
            "elapsed_seconds": 523.1326021669665,
            "accounting": {"own_rate_units_per_hour": 1.54},
            "usage_before": {"balance": 100.0},
            "usage_after": {"balance": balance_after},
        },
    )


def test_failed_repair_record_is_create_only_and_preserves_success_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, receipt = failure_inputs(tmp_path)
    output = tmp_path / "failed-accounting.json"
    record(monkeypatch, report, receipt, output)
    value = json.loads(output.read_text())
    assert value["charges"]["charged_units"] == 10.51
    assert value["failed_receipt"]["sha256"]
    with pytest.raises(FileExistsError):
        record(monkeypatch, report, receipt, output)


def test_canary_admission_uses_full_15_88_liability_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, receipt = failure_inputs(tmp_path)
    failed = tmp_path / "failed-accounting.json"
    record(monkeypatch, report, receipt, failed)
    held = manifest(tmp_path)
    output = tmp_path / "canary-admission.json"
    cumulative_admission(
        stage="canary",
        held_manifest=held,
        failure_accounting=failed,
        reconciliation=reconciliation(tmp_path),
        additional_attributable_debit_units=0,
        output=output,
    )
    value = verify_cumulative_admission(output, stage="canary", held_manifest=held)
    assert value["charged_units"] == 10.51
    assert value["outstanding_liabilities_units"] == 15.88
    assert value["charged_plus_reserved_units"] == 26.39


def test_repair_substitutes_actual_canary_charge_and_rejects_excess_debit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, receipt = failure_inputs(tmp_path)
    failed = tmp_path / "failed-accounting.json"
    record(monkeypatch, report, receipt, failed)
    held = manifest(tmp_path)
    output = tmp_path / "repair-admission.json"
    cumulative_admission(
        stage="repair",
        held_manifest=held,
        failure_accounting=failed,
        reconciliation=reconciliation(tmp_path),
        additional_attributable_debit_units=0,
        canary_receipt=successful_canary(tmp_path),
        output=output,
    )
    value = json.loads(output.read_text())
    assert value["live_stage_charge_units"] == 0.23
    assert value["outstanding_liabilities_units"] == 13.88
    assert value["charged_plus_reserved_units"] == 24.62
    with pytest.raises(ValueError, match="exceeds 34.50"):
        cumulative_admission(
            stage="repair",
            held_manifest=held,
            failure_accounting=failed,
            reconciliation=reconciliation(tmp_path / "other", 20.0),
            additional_attributable_debit_units=20.0,
            canary_receipt=successful_canary(tmp_path / "other"),
            output=tmp_path / "too-much.json",
        )


def test_reconciliation_and_manifest_binding_are_mandatory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, receipt = failure_inputs(tmp_path)
    failed = tmp_path / "failed-accounting.json"
    record(monkeypatch, report, receipt, failed)
    held = manifest(tmp_path)
    with pytest.raises(ValueError, match="differs"):
        cumulative_admission(
            stage="canary",
            held_manifest=held,
            failure_accounting=failed,
            reconciliation=reconciliation(tmp_path, 0.1),
            additional_attributable_debit_units=0,
            output=tmp_path / "bad.json",
        )
