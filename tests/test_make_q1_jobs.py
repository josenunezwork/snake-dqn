"""Fail-closed tests for the Mac-only approved Q1 packet generator."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from research.q0q1_20261017 import make_q1_jobs as subject


def _receipt(*, rate: float = 0.5) -> dict:
    return {
        "observed_units": 1.0,
        "observed_gpu_hours": 0.0,
        "observed_cpu_hours": 0.25,
        "observed_cpu_unit_rate_per_hour": rate,
        "cpu_seconds_per_episode": 2.0,
        "cpu_worker_efficiency": 0.8,
        "stages": {"setup_seconds": 120.0, "cleanup_seconds": 60.0},
        "q1_attempt_overhead_seconds": 45.0,
    }


def test_study_plan_requires_observed_nonzero_cpu_debit_and_matches_packets() -> None:
    l4, a100 = _receipt(), _receipt()
    accounting = {
        "q0_units": 64.0,
        "q0_gpu_hours": 1.0,
        "q0_cpu_hours": 2.0,
        "cpu_unit_rate_per_hour": 0.5,
        "cpu_eval_job_overhead_seconds": 3.0,
    }
    accounting.update(_a3_accounting())
    plan = subject._study_plan(l4, a100, accounting)
    assert plan["owner_release"] == subject.OWNER_RELEASE
    assert plan["amendment"] == "A3"
    assert plan["a3_envelope"]["q0_units_cap"] == 64.0
    assert plan["a3_envelope"]["combined_units_cap"] == 199.0
    assert plan["a3_envelope"]["reserve_units"] == 35.0
    accounting["cpu_unit_rate_per_hour"] = 0.0
    with pytest.raises(ValueError, match="cpu"):
        subject._study_plan(l4, a100, accounting)
    accounting["cpu_unit_rate_per_hour"] = 0.6
    with pytest.raises(ValueError, match="does not match"):
        subject._study_plan(l4, a100, accounting)


def _current_test_owner(tmp_path: Path) -> Path:
    """Copy the production A2 admission without repairing its binding in tests."""
    path = tmp_path / "test-owner-admission.json"
    path.write_bytes(subject.owner_admission.DEFAULT_ADMISSION.read_bytes())
    return path


def test_owner_admission_binds_current_registration_bytes(tmp_path) -> None:
    admission = subject._validate_owner_admission(_current_test_owner(tmp_path))
    assert admission["amendment"] == "A3"
    assert admission["status"] == "OWNER_ADMITTED"
    assert admission["owner_release"] == subject.OWNER_RELEASE


def test_a2_rejects_the_immutable_pre_a2_production_owner_admission() -> None:
    with pytest.raises(ValueError, match="does not bind"):
        subject._validate_owner_admission(
            Path("research/q0q1_20261017/OWNER_ADMISSION_20261018.json")
        )


@pytest.mark.parametrize("repaired_a100", [False, True])
@pytest.mark.parametrize("repaired_l4", [False, True])
@pytest.mark.parametrize("accounting_matches", [False, True])
def test_make_jobs_emits_exactly_20_training_and_100_evaluation_jobs(
    monkeypatch, tmp_path, repaired_a100, repaired_l4, accounting_matches
) -> None:
    l4 = tmp_path / "l4.json"
    a100 = tmp_path / "a100.json"
    runtime = tmp_path / "runtime.json"
    mac = tmp_path / "mac.json"
    accounting = tmp_path / "accounting.json"
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    owner = _current_test_owner(tmp_path)
    for path, value in ((l4, {}), (a100, {}), (runtime, {}), (mac, {})):
        path.write_text(json.dumps(value))
    accounting.write_text(
        json.dumps(
            {
                "q0_units": 64.0,
                "q0_gpu_hours": 1.0,
                "q0_cpu_hours": 2.0,
                "cpu_unit_rate_per_hour": 0.5,
                "cpu_eval_job_overhead_seconds": 3.0,
            }
        )
    )
    accounting.write_text(json.dumps({**json.loads(accounting.read_text()), **_a3_accounting()}))
    packet = {
        "runtime": {"cpu_cores": 12},
        "cells": [
            {"collector": "n1", "actor_count": 3},
            {"collector": "n3", "actor_count": 6},
        ],
    }
    l4_receipt = _receipt()
    l4_receipt["runtime_parity_receipt_sha256"] = subject.sha256_file(runtime)
    a100_value = _receipt()
    for index, repaired, receipt, job_id in (
        (0, repaired_l4, l4_receipt, "q0b-l4"),
        (1, repaired_a100, a100_value, "q0b-a100"),
    ):
        if repaired:
            receipt["requalification"] = {
                "job_id": job_id,
                "claim_sha256": "c" * 64,
                "terminal_sha256": "d" * 64,
            }
            accounting_value = json.loads(accounting.read_text())
            accounting_value["attempts"][index].update(
                claim_sha256="c" * 64, closure_sha256=("d" if accounting_matches else "e") * 64
            )
            accounting.write_text(json.dumps(accounting_value))
    monkeypatch.setattr(
        "research.q0q1_20261017.requalify_q0b.validate_receipt",
        lambda value: value["requalification"]["job_id"],
    )
    monkeypatch.setattr(
        subject,
        "_qualified_receipt",
        lambda path, hardware: (l4_receipt if hardware == "L4" else a100_value, packet),
    )
    monkeypatch.setattr(
        subject,
        "validate_packet",
        lambda packet: {
            "measured_cells": {
                "restart/n3": {"decisions_per_second": 2_000},
                "mature/n3": {"decisions_per_second": 2_000},
                "restart/n1": {"decisions_per_second": 2_000},
                "mature/n1": {"decisions_per_second": 2_000},
            }
        },
    )
    monkeypatch.setattr(subject, "_validate_runtime_receipt", lambda path: {"reference": {}})
    monkeypatch.setattr(subject, "_validate_mac_reference", lambda path: {})
    monkeypatch.setattr(
        "research.q0q1_20261017.qualification.validate_admission",
        lambda packet, plan: {"status": "ADMITTED_FOR_DISPATCH", "dispatch_authorized": True},
    )
    monkeypatch.setattr(
        "research.q0q1_20261017.run_queue_job.qualified_recipe_binding",
        lambda packet, command: None,
    )
    monkeypatch.setattr(subject.inventory_receipt, "load_receipt_audit", lambda *a, **kw: {})
    expected_error = not accounting_matches and (repaired_l4 or repaired_a100)
    with (
        pytest.raises(ValueError, match="accounting does not bind the re-qualified")
        if expected_error
        else nullcontext()
    ):
        jobs, plan = subject.make_jobs(
            commit="a" * 40,
            l4_receipt_path=l4,
            a100_receipt_path=a100,
            runtime_receipt_path=runtime,
            mac_reference_path=mac,
            owner_admission_path=owner.resolve(),
            q0_accounting_path=accounting,
            inventory_receipt_path=inventory,
            q0b_l4_job_id="q0b-l4",
            q0b_a100_job_id="q0b-a100",
            study_plan_drive_path=(
                "/content/drive/MyDrive/t16_colab/q0q1-20261017/q1/jobs/study-plan.json"
            ),
        )
    if expected_error:
        return
    assert len(jobs) == 120
    assert sum(job["id"].startswith("q1-train-") for job in jobs) == 20
    assert all(job["env"]["Q0Q1_OWNER_RELEASE"] == subject.OWNER_RELEASE for job in jobs)
    assert all(("q0b-l4" not in job["depends_on"]) == repaired_l4 for job in jobs)
    assert all(("q0b-a100" not in job["depends_on"]) == repaired_a100 for job in jobs)
    assert all(subject.Q0_MASS_JOB_ID in job["depends_on"] for job in jobs)
    assert all(job["env"]["Q0Q1_Q0B_A100_RECEIPT_SHA256"] for job in jobs)
    assert all(
        job["env"]["Q0Q1_BANK_INVENTORY_RECEIPT_SHA256"] == subject.sha256_file(inventory)
        for job in jobs
    )
    assert all(subject.Q0_INIT_JOB_ID in job["depends_on"] for job in jobs)
    assert all(job["env"]["Q0Q1_MASS_ARCHIVE_DIR"] == subject.Q0_MASS_ARCHIVE_DIR for job in jobs)
    assert {
        job["requirements"]["min_vcpu"] for job in jobs if job["id"].startswith("q1-train-")
    } == {12}
    assert plan["q0_units"] == 64.0
    assert plan["q1_cpu_overhead_hours"] == 100 * (4 * 2 + 3) / (3600 * 0.8)

    assert (
        jobs[0]["env"]["Q0Q1_STUDY_PLAN_SHA256"]
        == hashlib.sha256(subject.pretty_json(plan)).hexdigest()
    )


@pytest.mark.parametrize("field,value", [("amendment", "A1"), ("registration_sha256", "0" * 64)])
def test_current_digest_does_not_rescue_stale_amendment(tmp_path, field, value):
    path = _current_test_owner(tmp_path)
    admission = json.loads(path.read_text())
    admission[field] = value
    path.write_text(json.dumps(admission))
    with pytest.raises(ValueError, match="does not bind"):
        subject._validate_owner_admission(path)


def test_new_registration_amendment_invalidates_old_admission(tmp_path):
    registration = tmp_path / "registration.md"
    registration.write_bytes(subject.owner_admission.REGISTRATION.read_bytes())
    with registration.open("a") as stream:
        stream.write("\n## Amendment A4 - test only\n")
    admission = json.loads(subject.owner_admission.DEFAULT_ADMISSION.read_text())
    admission["registration_sha256"] = subject.sha256_file(registration)
    with pytest.raises(ValueError, match="approved A4"):
        subject.owner_admission.validate(admission, registration)


def _a3_accounting(l4_id="q0b-l4", a100_id="q0b-a100"):
    return {
        "schema": subject.q0_accounting.SCHEMA,
        "status": "COMPLETE",
        "q0_units": 64.0,
        "attempts": [
            {"id": job_id, "conservative_upper_bound": False} for job_id in (l4_id, a100_id)
        ],
        "a3_envelope": {
            "amendment": "A3",
            "baseline_q0_units_cap": 40.0,
            "observed_q0_units": 64.0,
            "q0_units_cap": 64.0,
            "q1_original_units_cap": 135.0,
            "baseline_combined_units_cap": 175.0,
            "reserve_fraction": 0.2,
            "reserve_basis_combined_units": 175.0,
            "reserve_units": 35.0,
            "q1_dispatch_headroom_units": 100.0,
            "q0_overrun_units": 24.0,
            "combined_units_cap": 199.0,
            "combined_dispatch_limit_units": 164.0,
            "derivation": "synthetic inclusive fixture",
        },
    }
