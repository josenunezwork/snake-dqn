"""Receipt-backed Q0/Q0b accounting contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.q0q1_20261017 import q0_accounting as subject


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def pricing(root: Path) -> Path:
    return write_json(
        root / "observed-pricing.json",
        {
            "schema": subject.PRICING_SCHEMA,
            "evidence": "operator panel observation",
            "rates": {"A100": 5.3, "L4": 2.0, "cpu": 0.5},
        },
    )


def overhead(root: Path) -> Path:
    return write_json(
        root / "cpu-overhead.json",
        {
            "schema": subject.OVERHEAD_SCHEMA,
            "status": "MEASURED",
            "measurement": "real_q1_evaluate_empty_shard_with_staging_publish_and_cleanup",
            "no_selection_worlds": True,
            "cpu_threads": 1,
            "system": "Darwin test fixture",
            "q1_evaluate_sha256": subject.sha256_file(
                Path(subject.__file__).resolve().parents[2]
                / "research/q0q1_20261017/q1_evaluate.py"
            ),
            "max_seconds": 3.0,
            "runs": [{"elapsed_seconds": 1.0}, {"elapsed_seconds": 2.0}, {"elapsed_seconds": 3.0}],
        },
    )


def claim(
    root: Path,
    *,
    job_id: str = "q0-audit-a100r6",
    token: str = "a" * 32,
    created_at: str = "2026-10-18T00:00:00Z",
) -> Path:
    return write_json(
        root / "queue" / "claimed" / job_id / f"g0.worker.{token}.json",
        {
            "id": job_id,
            "generation": 0,
            "token": token,
            "worker_id": "worker",
            "created_at": created_at,
            "capabilities": {"gpu": "A100", "vcpu": 12},
        },
    )


def terminal(
    root: Path,
    *,
    job_id: str = "q0-audit-a100r6",
    token: str = "a" * 32,
    base: float = 1792281600.0,
    recorded_at: str = "2026-10-18T00:06:00Z",
) -> Path:
    return write_json(
        root / "queue" / "failed" / job_id / f"g0.{token}.json",
        {
            "id": job_id,
            "generation": 0,
            "token": token,
            "status": "failed",
            "gpu": "A100",
            "elapsed_seconds": 360.0,
            "started_at": base,
            "run_started_at": base + 10.0,
            "finished_at": base + 360.0,
            "recorded_at": recorded_at,
        },
    )


def test_completed_attempts_include_failed_setup_cleanup_time(tmp_path):
    claim(tmp_path)
    terminal(tmp_path)
    result = subject.build_accounting(
        drive_root=tmp_path,
        pricing_path=pricing(tmp_path),
        overhead_path=overhead(tmp_path),
    )
    assert result["status"] == "COMPLETE"
    assert result["q0_gpu_hours"] == pytest.approx(0.1)
    assert result["q0_units"] == pytest.approx(0.53)
    assert result["attempts"][0]["terminal_status"] == "failed"
    assert result["attempts"][0]["provenance"] == "terminal_receipt_elapsed_seconds"
    assert result["cpu_eval_job_overhead_seconds"] == pytest.approx(3.0 * 3.0 + 10.0 + 60.0)


def test_forecast_and_debit_include_claim_arbitration_before_execute(tmp_path):
    claim(tmp_path)
    path = terminal(tmp_path, base=1792281720.0)
    result = subject.build_accounting(
        drive_root=tmp_path,
        pricing_path=pricing(tmp_path),
        overhead_path=overhead(tmp_path),
    )
    row = result["attempts"][0]
    assert row["queue_prelude_seconds"] == 120.0
    assert row["queue_setup_seconds"] == 10.0
    assert row["elapsed_seconds"] == 480.0
    assert row["closure_path"] == str(path)
    assert result["q0_units"] == pytest.approx(480.0 / 3600 * 5.3)
    assert result["cpu_eval_job_overhead_seconds"] == pytest.approx(3 * 3.0 + 130.0 + 60)


def test_active_claim_without_terminal_or_explicit_closure_refuses(tmp_path):
    claim(tmp_path)
    with pytest.raises(subject.AccountingError, match="no terminal receipt or explicit closure"):
        subject.build_accounting(
            drive_root=tmp_path,
            pricing_path=pricing(tmp_path),
            overhead_path=overhead(tmp_path),
        )


def test_idle_heartbeat_is_charged_as_conservative_upper_bound(tmp_path):
    claim(tmp_path, job_id="q0-audit-cpu", token="b" * 32)
    claim_path = (
        tmp_path / "queue" / "claimed" / "q0-audit-cpu" / ("g0.worker." + "b" * 32 + ".json")
    )
    value = json.loads(claim_path.read_text())
    value["capabilities"]["gpu"] = "cpu"
    claim_path.write_text(json.dumps(value))
    write_json(
        tmp_path / "queue" / "heartbeats" / "worker.json",
        {
            "worker_id": "worker",
            "phase": "idle",
            "job_id": None,
            "token": None,
            "updated_at": "2026-10-18T00:10:00Z",
        },
    )
    claim(tmp_path, job_id="q0-setup", token="d" * 32)
    terminal(tmp_path, job_id="q0-setup", token="d" * 32)
    result = subject.build_accounting(
        drive_root=tmp_path,
        pricing_path=pricing(tmp_path),
        overhead_path=overhead(tmp_path),
    )
    assert result["status"] == "COMPLETE_WITH_CONSERVATIVE_BOUNDS"
    assert result["q0_cpu_hours"] == pytest.approx(1 / 6)
    assert (
        next(row for row in result["attempts"] if row["id"] == "q0-audit-cpu")[
            "conservative_upper_bound"
        ]
        is True
    )


def test_existing_output_is_never_replaced(tmp_path):
    claim(tmp_path)
    terminal(tmp_path)
    out = tmp_path / "observed-q0-accounting.json"
    out.write_text("existing")
    with pytest.raises(SystemExit):
        subject.main(
            [
                "--drive-root",
                str(tmp_path),
                "--pricing",
                str(pricing(tmp_path)),
                "--cpu-eval-overhead-receipt",
                str(overhead(tmp_path)),
                "--out",
                str(out),
            ]
        )


def test_cli_writes_accounting_and_prints_the_full_derivation(tmp_path, capsys):
    claim(tmp_path)
    terminal(tmp_path)
    out = tmp_path / "admission" / "observed-q0-accounting.json"
    assert (
        subject.main(
            [
                "--drive-root",
                str(tmp_path),
                "--pricing",
                str(pricing(tmp_path)),
                "--cpu-eval-overhead-receipt",
                str(overhead(tmp_path)),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    value = json.loads(out.read_text())
    printed = json.loads(capsys.readouterr().out)
    assert printed == value
    assert printed["attempts"][0]["elapsed_seconds"] == 360.0
    assert printed["attempts"][0]["rate_units_per_hour"] == 5.3
    assert printed["attempts"][0]["charged_units"] == pytest.approx(0.53)
    assert printed["cpu_eval_overhead"]["forecast_seconds"] == 79.0


def test_stale_eval_overhead_receipt_refuses_accounting(tmp_path):
    claim(tmp_path)
    terminal(tmp_path)
    path = overhead(tmp_path)
    value = json.loads(path.read_text())
    value["q1_evaluate_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(subject.AccountingError, match="current real evaluator"):
        subject.build_accounting(
            drive_root=tmp_path,
            pricing_path=pricing(tmp_path),
            overhead_path=path,
        )


def test_cancel_without_verified_runtime_closure_is_not_cost_evidence(tmp_path):
    claim(tmp_path)
    write_json(
        tmp_path / "queue" / "control" / "q0-audit-a100r6.json",
        {"id": "q0-audit-a100r6", "cancelled": True, "updated_at": "2026-10-18T00:10:00Z"},
    )
    with pytest.raises(subject.AccountingError, match="no terminal receipt or explicit closure"):
        subject.build_accounting(
            drive_root=tmp_path,
            pricing_path=pricing(tmp_path),
            overhead_path=overhead(tmp_path),
        )


def test_verified_runtime_disconnect_is_conservative_bound(tmp_path):
    claim(tmp_path)
    write_json(
        tmp_path / "queue" / "control" / "q0-audit-a100r6.json",
        {
            "id": "q0-audit-a100r6",
            "cancelled": False,
            "reason": "runtime verified disconnected in Colab UI",
            "updated_at": "2026-10-18T00:10:00Z",
        },
    )
    claim(tmp_path, job_id="q0-setup", token="d" * 32)
    terminal(tmp_path, job_id="q0-setup", token="d" * 32)
    result = subject.build_accounting(
        drive_root=tmp_path, pricing_path=pricing(tmp_path), overhead_path=overhead(tmp_path)
    )
    assert (
        next(row for row in result["attempts"] if row["id"] == "q0-audit-a100r6")["provenance"]
        == "explicit_verified_runtime_disconnect_upper_bound"
    )


def test_zero_rate_and_unmatched_today_terminal_refuse(tmp_path):
    claim(tmp_path)
    terminal(tmp_path)
    pricing_path = pricing(tmp_path)
    bad_pricing = json.loads(pricing_path.read_text())
    bad_pricing["rates"]["A100"] = 0
    pricing_path.write_text(json.dumps(bad_pricing))
    with pytest.raises(subject.AccountingError, match="positive"):
        subject.build_accounting(
            drive_root=tmp_path, pricing_path=pricing_path, overhead_path=overhead(tmp_path)
        )
    bad_pricing["rates"]["A100"] = 5.3
    pricing_path.write_text(json.dumps(bad_pricing))
    write_json(
        tmp_path / "queue" / "done" / "q0-extra" / "g0.extra.json",
        {
            "id": "q0-extra",
            "generation": 0,
            "token": "c" * 32,
            "status": "done",
            "recorded_at": "2026-10-18T00:02:00Z",
        },
    )
    with pytest.raises(subject.AccountingError, match="no matching queue claim"):
        subject.build_accounting(
            drive_root=tmp_path, pricing_path=pricing_path, overhead_path=overhead(tmp_path)
        )


def test_default_lower_bound_includes_later_utc_rollover_attempt(tmp_path):
    claim(tmp_path, created_at="2026-10-19T00:00:00Z")
    terminal(tmp_path, base=1792368000.0, recorded_at="2026-10-19T00:06:00Z")
    result = subject.build_accounting(
        drive_root=tmp_path, pricing_path=pricing(tmp_path), overhead_path=overhead(tmp_path)
    )
    assert result["since_date_utc"] == "2026-10-18"
    assert len(result["attempts"]) == 1


def test_a3_envelope_preserves_the_original_q1_headroom_and_reserve(tmp_path):
    """The higher combined cap must not turn the fixed reserve into Q1 spend."""
    for index in range(120):
        token = f"{index:032x}"
        job_id = f"q0b-capacity-a100-r{index}"
        claim(tmp_path, job_id=job_id, token=token)
        terminal(tmp_path, job_id=job_id, token=token)
    result = subject.build_accounting(
        drive_root=tmp_path, pricing_path=pricing(tmp_path), overhead_path=overhead(tmp_path)
    )
    envelope = subject.validate_a3_envelope(result)
    assert result["q0_units"] == pytest.approx(63.6)
    assert envelope["q0_units_cap"] == pytest.approx(63.6)
    assert envelope["q0_overrun_units"] == pytest.approx(23.6)
    assert envelope["combined_units_cap"] == pytest.approx(198.6)
    assert envelope["reserve_fraction"] == 0.20
    assert envelope["reserve_basis_combined_units"] == 175.0
    assert envelope["reserve_units"] == 35.0
    assert envelope["q1_dispatch_headroom_units"] == 100.0
    assert envelope["combined_dispatch_limit_units"] == pytest.approx(163.6)


def test_a3_envelope_refuses_a_tampered_reserve_or_under_cap_accounting(tmp_path):
    for index in range(120):
        token = f"{index:032x}"
        job_id = f"q0b-capacity-a100-r{index}"
        claim(tmp_path, job_id=job_id, token=token)
        terminal(tmp_path, job_id=job_id, token=token)
    result = subject.build_accounting(
        drive_root=tmp_path, pricing_path=pricing(tmp_path), overhead_path=overhead(tmp_path)
    )
    result["a3_envelope"]["reserve_units"] = 35.01
    with pytest.raises(subject.AccountingError, match="invalid reserve_units"):
        subject.validate_a3_envelope(result)

    low_result = {
        "schema": subject.SCHEMA,
        "q0_units": 39.0,
        "a3_envelope": {
            **result["a3_envelope"],
            "observed_q0_units": 39.0,
            "q0_units_cap": 39.0,
            "q0_overrun_units": -1.0,
            "combined_units_cap": 174.0,
            "combined_dispatch_limit_units": 139.0,
            "reserve_units": 35.0,
        },
    }
    with pytest.raises(subject.AccountingError, match="above the 40-unit cap"):
        subject.validate_a3_envelope(low_result)


def test_a3_envelope_accepts_a_hash_bound_study_plan_view(tmp_path):
    for index in range(120):
        token = f"{index:032x}"
        job_id = f"q0b-capacity-a100-r{index}"
        claim(tmp_path, job_id=job_id, token=token)
        terminal(tmp_path, job_id=job_id, token=token)
    accounting = subject.build_accounting(
        drive_root=tmp_path, pricing_path=pricing(tmp_path), overhead_path=overhead(tmp_path)
    )
    plan = {
        "schema": "q0q1-study-budget/v1",
        "q0_units": accounting["q0_units"],
        "a3_envelope": accounting["a3_envelope"],
    }
    assert subject.validate_a3_envelope(plan) == accounting["a3_envelope"]
