"""Fail-closed tests for the Mac-only approved Q1 packet generator."""

from __future__ import annotations

import json
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
    }


def test_study_plan_requires_observed_nonzero_cpu_debit_and_matches_packets() -> None:
    l4, a100 = _receipt(), _receipt()
    accounting = {
        "q0_units": 20.0,
        "q0_gpu_hours": 1.0,
        "q0_cpu_hours": 2.0,
        "cpu_unit_rate_per_hour": 0.5,
    }
    plan = subject._study_plan(l4, a100, accounting)
    assert plan["owner_release"] == subject.OWNER_RELEASE
    assert plan["amendment"] == "A1"
    assert plan["a1_envelope"] == {
        "q0_units_cap": 40.0,
        "combined_units_cap": 175.0,
        "reserve_fraction": 0.20,
    }
    accounting["cpu_unit_rate_per_hour"] = 0.0
    with pytest.raises(ValueError, match="cpu"):
        subject._study_plan(l4, a100, accounting)
    accounting["cpu_unit_rate_per_hour"] = 0.6
    with pytest.raises(ValueError, match="does not match"):
        subject._study_plan(l4, a100, accounting)


def test_owner_admission_binds_current_registration_bytes() -> None:
    admission = subject._validate_owner_admission(
        Path("research/q0q1_20261017/OWNER_ADMISSION_20261018.json")
    )
    assert admission["status"] == "OWNER_ADMITTED"
    assert admission["owner_release"] == subject.OWNER_RELEASE


def test_make_jobs_emits_exactly_20_training_and_100_evaluation_jobs(monkeypatch, tmp_path) -> None:
    l4 = tmp_path / "l4.json"
    a100 = tmp_path / "a100.json"
    runtime = tmp_path / "runtime.json"
    mac = tmp_path / "mac.json"
    accounting = tmp_path / "accounting.json"
    owner = Path("research/q0q1_20261017/OWNER_ADMISSION_20261018.json")
    for path, value in ((l4, {}), (a100, {}), (runtime, {}), (mac, {})):
        path.write_text(json.dumps(value))
    accounting.write_text(
        json.dumps(
            {"q0_units": 20.0, "q0_gpu_hours": 1.0, "q0_cpu_hours": 2.0,
             "cpu_unit_rate_per_hour": 0.5}
        )
    )
    packet = {
        "runtime": {"cpu_cores": 12},
        "cells": [
            {"collector": "n1", "actor_count": 3},
            {"collector": "n3", "actor_count": 6},
        ],
    }
    monkeypatch.setattr(subject, "_qualified_receipt", lambda path, hardware: (_receipt(), packet))
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
    monkeypatch.setattr(subject, "_validate_runtime_receipt", lambda path: {})
    monkeypatch.setattr(subject, "_validate_mac_reference", lambda path: {})
    monkeypatch.setattr(
        "research.q0q1_20261017.qualification.validate_admission",
        lambda packet, plan: {"status": "ADMITTED_FOR_DISPATCH", "dispatch_authorized": True},
    )
    monkeypatch.setattr(
        "research.q0q1_20261017.run_queue_job.qualified_recipe_binding", lambda packet, command: None
    )
    jobs, plan = subject.make_jobs(
        commit="a" * 40,
        l4_receipt_path=l4,
        a100_receipt_path=a100,
        runtime_receipt_path=runtime,
        mac_reference_path=mac,
        owner_admission_path=owner.resolve(),
        q0_accounting_path=accounting,
        q0b_l4_job_id="q0b-l4",
        q0b_a100_job_id="q0b-a100",
        study_plan_drive_path="/content/drive/MyDrive/t16_colab/q0q1-20261017/q1/jobs/study-plan.json",
    )
    assert len(jobs) == 120
    assert sum(job["id"].startswith("q1-train-") for job in jobs) == 20
    assert all(job["env"]["Q0Q1_OWNER_RELEASE"] == subject.OWNER_RELEASE for job in jobs)
    assert all({"q0b-l4", "q0b-a100"} <= set(job["depends_on"]) for job in jobs)
    assert all(job["env"]["Q0Q1_Q0B_A100_RECEIPT_SHA256"] for job in jobs)
    assert {job["requirements"]["min_vcpu"] for job in jobs if job["id"].startswith("q1-train-")} == {12}
    assert plan["q0_units"] == 20.0
