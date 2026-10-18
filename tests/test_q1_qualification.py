"""Contract tests for the fail-closed Q0b/Q1 admission validator."""

from __future__ import annotations

from research.q0q1_20261017 import qualification as subject

HASH = "a" * 64


def _cell(schedule: str, collector: str) -> dict:
    return {
        "schedule": schedule,
        "collector": collector,
        "actor_count": 6 if collector == "n3" else 3,
        "warmup_seconds": 120,
        "compile_ready": True,
        "replay_ready": True,
        "scored_windows": [
            {"seconds": 180, "learner_decisions": 1_800_000},
            {"seconds": 180, "learner_decisions": 1_782_000},
        ],
        "clean_exit": True,
        "finite_updates": True,
        "dropped_transitions": 0,
        "ring_overruns": 0,
        "counters": {
            "actual_learner_decisions": 3_582_000,
            "world_frames": 4_000_000,
            "pool_decisions": 2_000_000,
            "emitted_replay_rows": 256_100,
            "completed_updates": 1000,
            "pending_update_debt": 100,
            "truncation_rows_dropped": 3_325_836,
            "pending_nstep_rows": 64,
            "pending_nstep_rows_by_actor": {
                str(actor): 64 if actor == 0 else 0
                for actor in range(6 if collector == "n3" else 3)
            },
            "decision_replay_delta": 3_325_900,
            "nstep_accounting_reconciled": True,
        },
        "policy_lag": {
            "mean": 2.0,
            "p95": 3.0,
            "age_p95": 5.5,
            "matched_control": {"mean": 2.0, "p95": 3.0, "age_p95": 5.0},
        },
    }


def _packet() -> dict:
    return {
        "schema": subject.PACKET_SCHEMA,
        "hardware": "L4",
        "packet_seconds": 3590,
        "unit_rate_per_hour": 2.0,
        "runtime": {
            "precision": "fp32",
            "batch_scale": 2,
            "raw_local_rings": True,
            "queue_depth": 2,
            "emitted_rows_per_update": 256,
            "ram_gb": 24,
            "cpu_cores": 4,
        },
        "stages": {"setup_seconds": 100, "cleanup_seconds": 100},
        "source": {"commit": HASH},
        "config": {"trainer": HASH},
        "environment": {"lock": HASH},
        "hashes": {"b8": HASH, "t22": HASH, "t24": HASH},
        "cells": [
            _cell("restart", "n1"),
            _cell("mature", "n1"),
            _cell("restart", "n3"),
            _cell("mature", "n3"),
        ],
        "parity": {
            "target_runtime": "cuda",
            "short_body": True,
            "long_body": True,
            "t22": "PASS",
            "t24": "PASS",
        },
    }


def _plan() -> dict:
    return {
        "schema": subject.PLAN_SCHEMA,
        "owner_review_status": "DRAFT_PENDING_OWNER_REVIEW",
        "dispatch": False,
        "q1_training_hardware": "L4",
        "q0_units": 10,
        "q0_gpu_hours": 1,
        "q0_cpu_hours": 1,
        "q1_gpu_overhead_hours": 1,
        "q1_cpu_overhead_hours": 1,
        "cpu_unit_rate_per_hour": 0.5,
        "cpu_seconds_per_episode": 10,
        "cpu_workers": 1,
        "cpu_worker_efficiency": 0.8,
        "evaluation_episodes": 3200,
        "runs_per_arm": 5,
        "decisions_per_run": 8_000_000,
        "evaluation_jobs": 100,
        "parity_worlds_per_evaluation_job": 4,
    }


def _approved_a1_plan() -> dict:
    plan = _plan()
    plan.update(
        owner_review_status=subject.APPROVED_OWNER_RELEASE,
        owner_release=subject.APPROVED_OWNER_RELEASE,
        amendment=subject.A1_AMENDMENT,
        dispatch=True,
        a1_envelope={
            "q0_units_cap": 40.0,
            "combined_units_cap": 175.0,
            "reserve_fraction": 0.20,
        },
    )
    return plan


def test_packet_accepts_all_exact_q0b_constraints() -> None:
    outcome = subject.validate_packet(_packet())
    assert outcome["status"] == "QUALIFIED", outcome["errors"]
    assert set(outcome["measured_cells"]) == {"restart/n1", "mature/n1", "restart/n3", "mature/n3"}


def test_packet_refuses_unbound_or_non_cuda_evidence() -> None:
    packet = _packet()
    packet["source"] = {"commit": "not-a-hash"}
    packet["parity"]["target_runtime"] = "macos"
    outcome = subject.validate_packet(packet)
    assert outcome["status"] == "INCOMPLETE"
    assert any("SHA-256" in error for error in outcome["errors"])
    assert any("CUDA" in error for error in outcome["errors"])


def test_packet_rejects_short_windows_actor_drift_and_counter_debt() -> None:
    packet = _packet()
    packet["cells"][1]["actor_count"] = 4
    packet["cells"][2]["scored_windows"][1]["seconds"] = 179
    packet["cells"][3]["counters"]["pending_update_debt"] = 99_999
    outcome = subject.validate_packet(packet)
    assert outcome["status"] == "INCOMPLETE"
    assert any("one predeclared actor count" in error for error in outcome["errors"])
    assert any("at least 180" in error for error in outcome["errors"])
    assert any("R/U/debt" in error for error in outcome["errors"])


def test_packet_rejects_rate_decline_and_lag_regression() -> None:
    packet = _packet()
    packet["cells"][0]["scored_windows"][1]["learner_decisions"] = 1_600_000
    packet["cells"][0]["policy_lag"]["p95"] = 3.1
    packet["cells"][0]["policy_lag"]["age_p95"] = 5.6
    outcome = subject.validate_packet(packet)
    assert any("throughput declined" in error for error in outcome["errors"])
    assert any("policy-lag increase" in error for error in outcome["errors"])
    assert any("ingestion age" in error for error in outcome["errors"])


def test_packet_rejects_wall_time_that_cannot_cover_all_four_cells() -> None:
    packet = _packet()
    packet["packet_seconds"] = (
        2100  # Four 120-second warm-ups, eight 180-second windows, setup and cleanup need 2,120.
    )
    outcome = subject.validate_packet(packet)
    assert outcome["status"] == "INCOMPLETE"
    assert any("wall time" in error for error in outcome["errors"])


def test_admission_computes_complete_20_run_and_3200_episode_forecast() -> None:
    outcome = subject.validate_admission(_packet(), _plan())
    assert outcome["status"] == "ADMITTED_FOR_OWNER_REVIEW", outcome["errors"]
    assert outcome["dispatch_authorized"] is False
    assert outcome["forecast"]["q1_training_gpu_hours"] > 0
    assert outcome["forecast"]["q1_evaluation_cpu_hours"] == 3200 * 10 / (3600 * 1 * 0.8)


def test_admission_refuses_unknown_cpu_rate_and_cap_overrun() -> None:
    plan = _plan()
    plan["cpu_unit_rate_per_hour"] = None
    plan["q0_units"] = 16
    outcome = subject.validate_admission(_packet(), plan)
    assert outcome["status"] == "REFUSED"
    assert any("cpu_unit_rate" in error for error in outcome["errors"])
    assert any("Q0 exceeds" in error for error in outcome["errors"])


def test_approved_a1_admission_uses_bound_40_unit_q0_and_175_unit_envelope() -> None:
    plan = _approved_a1_plan()
    plan["q0_units"] = 40
    plan["q0_gpu_hours"] = 4
    plan["q0_cpu_hours"] = 8
    outcome = subject.validate_admission(_packet(), plan)
    assert outcome["status"] == "ADMITTED_FOR_DISPATCH", outcome["errors"]
    assert outcome["dispatch_authorized"] is True
    assert outcome["forecast"]["q0_units_cap"] == 40
    assert outcome["forecast"]["usable_combined_units_cap"] == 140


def test_approved_a1_admission_requires_every_approval_and_envelope_binding() -> None:
    plan = _approved_a1_plan()
    plan["owner_release"] = "forged"
    plan["a1_envelope"]["combined_units_cap"] = 150
    outcome = subject.validate_admission(_packet(), plan)
    assert outcome["status"] == "REFUSED"
    assert outcome["dispatch_authorized"] is False
    assert any("owner release" in error for error in outcome["errors"])
    assert any("combined_units_cap" in error for error in outcome["errors"])


def test_admission_never_turns_draft_qualification_into_dispatch() -> None:
    plan = _plan()
    plan["dispatch"] = True
    outcome = subject.validate_admission(_packet(), plan)
    assert outcome["status"] == "REFUSED"
    assert outcome["dispatch_authorized"] is False
    assert any("must not authorize dispatch" in error for error in outcome["errors"])


def test_admission_requires_a_single_l4_training_cohort() -> None:
    packet = _packet()
    packet["hardware"] = "A100"
    plan = _plan()
    plan["q1_training_hardware"] = "A100"
    outcome = subject.validate_admission(packet, plan)
    assert outcome["status"] == "REFUSED"
    assert any("exact qualified L4 cohort" in error for error in outcome["errors"])


def test_admission_counts_q0_hours_against_global_reserve() -> None:
    packet = _packet()
    for cell in packet["cells"]:
        for window in cell["scored_windows"]:
            window["learner_decisions"] = 500_000
    plan = _plan()
    plan["q0_gpu_hours"] = 18.5
    outcome = subject.validate_admission(packet, plan)
    assert outcome["status"] == "REFUSED"
    assert any("GPU forecast" in error for error in outcome["errors"])


def test_packet_rejects_zero_negative_nan_and_boolean_counters() -> None:
    packet = _packet()
    packet["cells"][0]["actor_count"] = True
    packet["cells"][0]["counters"]["emitted_replay_rows"] = 0
    packet["cells"][1]["counters"]["completed_updates"] = -1
    packet["cells"][2]["counters"]["pending_update_debt"] = float("nan")
    packet["cells"][3]["counters"]["pool_decisions"] = False
    outcome = subject.validate_packet(packet)
    assert outcome["status"] == "INCOMPLETE"
    assert any("actor_count" in error for error in outcome["errors"])
    for field in (
        "emitted_replay_rows",
        "completed_updates",
        "pending_update_debt",
        "pool_decisions",
    ):
        assert any(field in error for error in outcome["errors"])


def test_packet_accepts_zero_pending_update_debt() -> None:
    packet = _packet()
    for cell in packet["cells"]:
        cell["counters"].update(emitted_replay_rows=256_000, pending_update_debt=0)
        cell["counters"].update(truncation_rows_dropped=3_325_936, decision_replay_delta=3_326_000)
    assert subject.validate_packet(packet)["status"] == "QUALIFIED"


def test_admission_requires_explicit_nonzero_overhead_and_one_worker() -> None:
    plan = _plan()
    plan["q1_gpu_overhead_hours"] = 0
    plan["q1_cpu_overhead_hours"] = -1
    plan["cpu_workers"] = 2
    outcome = subject.validate_admission(_packet(), plan)
    assert outcome["status"] == "REFUSED"
    assert any("q1_gpu_overhead_hours" in error for error in outcome["errors"])
    assert any("q1_cpu_overhead_hours" in error for error in outcome["errors"])
    assert any("one worker" in error for error in outcome["errors"])
