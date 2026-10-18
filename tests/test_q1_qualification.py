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
            "emitted_replay_rows": 256_100,
            "completed_updates": 1000,
            "pending_update_debt": 100,
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
        "parity": {"target_runtime": "cuda", "short_body": True, "long_body": True, "t22": "PASS", "t24": "PASS"},
    }


def _plan() -> dict:
    return {
        "schema": subject.PLAN_SCHEMA,
        "owner_review_status": "DRAFT_PENDING_OWNER_REVIEW",
        "dispatch": False,
        "q0_units": 10,
        "cpu_unit_rate_per_hour": 0.5,
        "cpu_seconds_per_episode": 10,
        "cpu_workers": 2,
        "cpu_worker_efficiency": 0.8,
        "evaluation_episodes": 3200,
        "runs_per_arm": 5,
        "decisions_per_run": 8_000_000,
    }


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
    assert any("exactly 180" in error for error in outcome["errors"])
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


def test_admission_computes_complete_20_run_and_3200_episode_forecast() -> None:
    outcome = subject.validate_admission(_packet(), _plan())
    assert outcome["status"] == "ADMITTED_FOR_OWNER_REVIEW", outcome["errors"]
    assert outcome["dispatch_authorized"] is False
    assert outcome["forecast"]["q1_training_gpu_hours"] > 0
    assert outcome["forecast"]["q1_evaluation_cpu_hours"] == 3200 * 10 / (3600 * 2 * 0.8)


def test_admission_refuses_unknown_cpu_rate_and_cap_overrun() -> None:
    plan = _plan()
    plan["cpu_unit_rate_per_hour"] = None
    plan["q0_units"] = 16
    outcome = subject.validate_admission(_packet(), plan)
    assert outcome["status"] == "REFUSED"
    assert any("cpu_unit_rate" in error for error in outcome["errors"])
    assert any("Q0 exceeds" in error for error in outcome["errors"])


def test_admission_never_turns_draft_qualification_into_dispatch() -> None:
    plan = _plan()
    plan["dispatch"] = True
    outcome = subject.validate_admission(_packet(), plan)
    assert outcome["status"] == "REFUSED"
    assert outcome["dispatch_authorized"] is False
    assert any("must not authorize dispatch" in error for error in outcome["errors"])
