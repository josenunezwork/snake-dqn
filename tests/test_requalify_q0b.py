"""Fail-closed tests for the A3 preserved Q0b packet repair."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.q0q1_20261017 import make_q1_jobs
from research.q0q1_20261017 import requalify_q0b as subject

HASH = "a" * 64
CLAIMED_AT = 1_792_357_200.0


def _cell(schedule: str, collector: str, *, lag_mean: float = 2.0) -> dict:
    actors = 3 if collector == "n1" else 6
    return {
        "schedule": schedule,
        "collector": collector,
        "actor_count": actors,
        "warmup_seconds": 120.0,
        "compile_ready": True,
        "replay_ready": True,
        "scored_windows": [
            {"seconds": 180.0, "learner_decisions": 1_800},
            {"seconds": 180.0, "learner_decisions": 1_800},
        ],
        "clean_exit": True,
        "finite_updates": True,
        "dropped_transitions": 0,
        "ring_overruns": 0,
        "observed_launcher_seconds": 500.0,
        "counters": {
            "actual_learner_decisions": 2_560,
            "world_frames": 2_560,
            "pool_decisions": 2_560,
            "emitted_replay_rows": 2_560,
            "completed_updates": 10,
            "pending_update_debt": 0,
            "truncation_rows_dropped": 0,
            "pending_nstep_rows": 0,
            "pending_nstep_rows_by_actor": {str(actor): 0 for actor in range(actors)},
            "decision_replay_delta": 0,
            "nstep_accounting_reconciled": True,
        },
        "policy_lag": {
            "mean": lag_mean,
            "p95": 3.0,
            "age_p95": 5.0,
            "matched_control": {"mean": 2.0, "p95": 3.0, "age_p95": 5.0},
        },
    }


def _packet(source: dict[str, str]) -> dict:
    runtime = {
        "precision": "fp32",
        "batch_scale": 2,
        "raw_local_rings": True,
        "queue_depth": 2,
        "emitted_rows_per_update": 256,
        "ram_gb": 24.0,
        "cpu_cores": 12.0,
    }
    return {
        "schema": subject.qualification.PACKET_SCHEMA,
        "hardware": "A100",
        "packet_seconds": 1_950.0,
        "unit_rate_per_hour": 5.3,
        "runtime": runtime,
        "stages": {"setup_seconds": 0.0, "cleanup_seconds": 0.0},
        "source": source,
        "config": {"trainer": HASH},
        "environment": {"runtime": HASH},
        "hashes": {"b8_initializer": HASH},
        "cells": [
            _cell("restart", "n1", lag_mean=2.0),
            _cell("mature", "n1", lag_mean=2.05),
            _cell("restart", "n3", lag_mean=2.0),
            _cell("mature", "n3", lag_mean=2.0),
        ],
        "parity": {
            "target_runtime": "cuda",
            "short_body": True,
            "long_body": True,
            "t22": "PASS",
            "t24": "PASS",
        },
        "cpu_seconds_per_episode": 10.0,
        "observed_cpu_unit_rate_per_hour": 0.5,
        "cpu_worker_efficiency": 1.0,
    }


def _archive(
    path: Path,
    packet: dict,
    *,
    reference: dict | None = None,
    qualification_errors: list[str] | None = None,
) -> None:
    runtime_reference = reference or {
        "schema": "q1-parity/v1",
        "bank_sha256": HASH,
        "b8_sha256": HASH,
        "source_sha256": HASH,
    }
    runtime = {
        "reference": runtime_reference,
        "pass": True,
        "schema": "q1-runtime-parity-receipt/v1",
    }
    runtime_raw = json.dumps(runtime, sort_keys=True).encode()
    t22 = {"fixture": "t22", "status": "PASS"}
    t24 = {"fixture": "t24", "status": "PASS"}
    t22_raw = json.dumps(t22, sort_keys=True).encode()
    t24_raw = json.dumps(t24, sort_keys=True).encode()
    packet["runtime_parity_receipt_sha256"] = hashlib.sha256(runtime_raw).hexdigest()
    packet["hashes"].update(
        t22_receipt=hashlib.sha256(t22_raw).hexdigest(),
        t24_receipt=hashlib.sha256(t24_raw).hexdigest(),
    )
    packet["qualification"] = {
        "status": "INCOMPLETE",
        "errors": qualification_errors
        or ["cell mature/n1 has an unexplained policy-lag increase over its matched control"],
    }
    objects = {
        "attempt/measurement/receipt.json": packet,
        "attempt/measurement/runtime-parity.json": runtime,
        "attempt/t22-cuda.json": t22,
        "attempt/t24-cuda.json": t24,
        "attempt/receipt.json": {"schema": "q0q1-qualification-packet/v1", "status": "INCOMPLETE"},
        "attempt/queue-attempt-receipt.json": {
            "schema": "q0q1-queue-attempt/v1",
            "job_id": subject.JOB_ID,
            "status": "INCOMPLETE",
        },
    }
    with path.open("wb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, value in sorted(objects.items()):
            raw = json.dumps(value, sort_keys=True).encode()
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    path.with_name(path.name + ".sha256").write_text(f"{subject.sha256_file(path)}  {path.name}\n")


def _queue_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    drive = tmp_path / "drive"
    admission = drive / "q0q1-20261017" / "admission"
    admission.mkdir(parents=True)
    (admission / "observed-pricing.json").write_text(json.dumps({"pricing": True}))
    (admission / "mac-parity-reference.json").write_text(
        json.dumps(
            {
                "schema": "q1-parity/v1",
                "bank_sha256": HASH,
                "b8_sha256": HASH,
                "source_sha256": HASH,
            }
        )
    )
    for path in admission.glob("*.json"):
        path.with_name(path.name + ".sha256").write_text(
            f"{subject.sha256_file(path)}  {path.name}\n"
        )
    claim = tmp_path / "claim.json"
    claim.write_text(
        json.dumps(
            {
                "id": subject.JOB_ID,
                "generation": 0,
                "token": "token",
                "worker_id": "worker",
                "commit": subject.R6_COMMIT,
                "created_at": "2026-10-18T21:00:00Z",
                "capabilities": {"gpu": "A100"},
            }
        )
    )
    terminal = tmp_path / "terminal.json"
    terminal.write_text(
        json.dumps(
            {
                "id": subject.JOB_ID,
                "generation": 0,
                "token": "token",
                "worker_id": "worker",
                "commit": subject.R6_COMMIT,
                "status": "failed",
                "exit_code": 64,
                "gpu": "A100",
                "verified_commit": subject.R6_COMMIT,
                "started_at": CLAIMED_AT + 10,
                "run_started_at": CLAIMED_AT + 20,
                "finished_at": CLAIMED_AT + 2_100,
                "elapsed_seconds": 2_090,
                "run_seconds": 2_080,
            }
        )
    )
    return drive, claim, terminal


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    source = {"recipe.py": HASH}
    monkeypatch.setattr(subject.q0b_measure, "measurement_source_manifest", lambda: source)
    monkeypatch.setattr(subject, "observed_pricing", lambda value, hardware: (5.3, 0.5))
    monkeypatch.setattr(
        subject,
        "bound_input",
        lambda drive, relative: (
            drive
            / "q0q1-20261017/admission"
            / ("observed-pricing.json" if "pricing" in relative else "mac-parity-reference.json"),
            (
                {"pricing": True}
                if "pricing" in relative
                else {
                    "schema": "q1-parity/v1",
                    "bank_sha256": HASH,
                    "b8_sha256": HASH,
                    "source_sha256": HASH,
                }
            ),
        ),
    )
    monkeypatch.setattr(make_q1_jobs, "_validate_mac_reference", lambda path: None)
    monkeypatch.setattr(subject.q1_parity, "validate_runtime_receipt", lambda *args, **kwargs: None)
    monkeypatch.setattr(make_q1_jobs.q1_evaluate, "source_manifest", lambda: {})
    packet = _packet(source)
    archive = tmp_path / "original-result.tar.gz"
    _archive(archive, packet)
    drive, claim, terminal = _queue_evidence(tmp_path)
    return archive, claim, terminal, drive


def test_qualify_reaccepts_preserved_packet_with_inclusive_claim_to_terminal_accounting(prepared):
    archive, claim, terminal, drive = prepared
    receipt, payload = subject.qualify(
        archive_path=archive, claim_path=claim, terminal_path=terminal, drive_root=drive
    )
    assert receipt["qualification"]["status"] == "QUALIFIED"
    assert receipt["packet_seconds"] == pytest.approx(2_100.0)
    assert receipt["observed_units"] == pytest.approx(5.3 * 2_100 / 3_600)
    assert receipt["observed_cpu_hours"] == 0.0
    assert receipt["requalification"]["inclusive_claim_to_terminal_seconds"] == 2_100.0
    assert "attempt/original-wrapper-receipt.json" in payload
    assert "attempt/claim.json" in payload
    assert "attempt/terminal.json" in payload


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("finished_at", CLAIMED_AT + 3_700, "cannot cover measurement within cap"),
        ("started_at", CLAIMED_AT + 30, "times are unordered"),
    ],
)
def test_qualify_rejects_inclusive_cap_and_time_order(prepared, field, value, match):
    archive, claim, terminal, drive = prepared
    data = json.loads(terminal.read_text())
    data[field] = value
    if field == "finished_at":
        data["elapsed_seconds"] = value - data["started_at"]
        data["run_seconds"] = value - data["run_started_at"]
    terminal.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        subject.qualify(
            archive_path=archive, claim_path=claim, terminal_path=terminal, drive_root=drive
        )


def test_qualify_rejects_archive_sidecar_tampering(prepared):
    archive, claim, terminal, drive = prepared
    archive.with_name(archive.name + ".sha256").write_text(f"{'b' * 64}  {archive.name}\n")
    with pytest.raises(ValueError, match="SHA sidecar mismatch"):
        subject.qualify(
            archive_path=archive, claim_path=claim, terminal_path=terminal, drive_root=drive
        )


def test_qualify_rejects_another_original_failure(prepared):
    archive, claim, terminal, drive = prepared
    objects = subject.archive_objects(archive)
    packet = objects["attempt/measurement/receipt.json"]["value"]
    packet["qualification"]["errors"] = ["cell restart/n1 throughput declined"]
    _archive(archive, packet, qualification_errors=["cell restart/n1 throughput declined"])
    with pytest.raises(ValueError, match="did not fail solely"):
        subject.qualify(
            archive_path=archive, claim_path=claim, terminal_path=terminal, drive_root=drive
        )


def test_qualify_rejects_source_mismatch(prepared, monkeypatch):
    archive, claim, terminal, drive = prepared
    monkeypatch.setattr(
        subject.q0b_measure, "measurement_source_manifest", lambda: {"other.py": HASH}
    )
    with pytest.raises(ValueError, match="measurement source differs"):
        subject.qualify(
            archive_path=archive, claim_path=claim, terminal_path=terminal, drive_root=drive
        )


def test_validate_receipt_rejects_changed_measurement_cells(prepared):
    archive, claim, terminal, drive = prepared
    receipt, _ = subject.qualify(
        archive_path=archive, claim_path=claim, terminal_path=terminal, drive_root=drive
    )
    receipt["cells"][0]["scored_windows"][0]["learner_decisions"] += 1
    with pytest.raises(ValueError, match="stale A3 re-qualification provenance"):
        subject.validate_receipt(receipt)


def test_generated_archive_is_accepted_by_stage_evidence_and_qualified_receipt(prepared, tmp_path):
    archive, claim, terminal, drive = prepared
    receipt, payload = subject.qualify(
        archive_path=archive, claim_path=claim, terminal_path=terminal, drive_root=drive
    )
    generated = tmp_path / "generated-result.tar.gz"
    with generated.open("wb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, raw in sorted(payload.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    generated.with_name(generated.name + ".sha256").write_text(
        f"{subject.sha256_file(generated)}  {generated.name}\n"
    )
    staged = tmp_path / "staged.json"
    assert (
        make_q1_jobs._stage_evidence(generated, staged, member_name="attempt/receipt.json")
        == staged
    )
    staged_receipt, packet = make_q1_jobs._qualified_receipt(staged, "A100")
    assert staged_receipt["requalification"]["status"] == "QUALIFIED_FROM_PRESERVED_MEASUREMENT"
    assert packet["hardware"] == "A100"
    assert receipt["packet_seconds"] == staged_receipt["packet_seconds"]


def test_main_writes_create_only_hash_bound_archive_and_refuses_replacement(
    prepared, tmp_path, monkeypatch
):
    archive, claim, terminal, drive = prepared
    monkeypatch.setattr(
        subject.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    out = tmp_path / "new-result"
    argv = [
        "--archive",
        str(archive),
        "--claim",
        str(claim),
        "--terminal",
        str(terminal),
        "--drive-root",
        str(drive),
        "--out-dir",
        str(out),
    ]
    assert subject.main(argv) == 0
    assert (out / "result.tar.gz").is_file()
    assert (out / "result.tar.gz.sha256").is_file()
    with pytest.raises(SystemExit):
        subject.main(argv)
