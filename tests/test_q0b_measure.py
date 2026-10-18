"""Focused contract tests for the Q0b real-measurement receipt adapter."""

from __future__ import annotations

import pytest

from research.q0q1_20261017 import q0b_measure as subject


def _boundary(phase: str, elapsed: float, decisions: int) -> dict:
    return {
        "phase": phase,
        "elapsed_seconds": elapsed,
        "learner_decisions": decisions,
        "world_frames": decisions * 2,
        "pool_decisions": decisions * 3,
        "emitted_replay_rows": 256_000,
        "completed_updates": 1000,
        "pending_update_debt": 0,
        "compile_ready": True,
        "replay_ready": True,
        "finite_updates": True,
        "dropped_transitions": 0,
        "ring_overruns": 0,
        "policy_lag": {"mean": 2.0, "p95": 3.0, "age_p95": 4.0},
        "matched_control": {"mean": 2.0, "p95": 3.0, "age_p95": 4.0},
    }


def test_cell_uses_real_boundary_counter_deltas_and_exact_windows() -> None:
    boundaries = {
        "warmup": _boundary("warmup", 120.0, 1_000),
        "window1": _boundary("window1", 300.0, 4_000),
        "window2": _boundary("window2", 480.0, 8_000),
    }
    cell = subject.cell_from_boundaries(
        schedule="restart",
        collector="n1",
        actor_count=2,
        boundaries=boundaries,
        summary={"stopped_for_time": True, "halt": None, "actor_exitcodes": [0]},
    )
    assert cell["scored_windows"] == [
        {"seconds": 180.0, "learner_decisions": 3000},
        {"seconds": 180.0, "learner_decisions": 4000},
    ]
    assert cell["counters"]["completed_updates"] == 1000


@pytest.mark.parametrize(
    "phase,elapsed", [("warmup", 119.0), ("window1", 299.0), ("window2", 479.0)]
)
def test_cell_refuses_shortened_or_renamed_boundaries(phase: str, elapsed: float) -> None:
    boundaries = {
        "warmup": _boundary("warmup", 120.0, 1_000),
        "window1": _boundary("window1", 300.0, 4_000),
        "window2": _boundary("window2", 480.0, 8_000),
    }
    boundaries[phase]["elapsed_seconds"] = elapsed
    with pytest.raises(ValueError, match="shorter"):
        subject.cell_from_boundaries(
            schedule="mature",
            collector="n3",
            actor_count=4,
            boundaries=boundaries,
            summary={"stopped_for_time": True, "halt": None, "actor_exitcodes": [0]},
        )


def test_cell_refuses_aggregate_or_nonadvancing_scored_counters() -> None:
    boundaries = {
        "warmup": _boundary("warmup", 120.0, 1_000),
        "window1": _boundary("window1", 300.0, 1_000),
        "window2": _boundary("window2", 480.0, 2_000),
    }
    with pytest.raises(ValueError, match="did not advance"):
        subject.cell_from_boundaries(
            schedule="restart",
            collector="n1",
            actor_count=2,
            boundaries=boundaries,
            summary={"stopped_for_time": True, "halt": None, "actor_exitcodes": [0]},
        )


def test_cell_refuses_unclean_or_uninstrumented_completion() -> None:
    boundaries = {
        "warmup": _boundary("warmup", 120.0, 1_000),
        "window1": _boundary("window1", 300.0, 4_000),
        "window2": _boundary("window2", 480.0, 8_000),
    }
    with pytest.raises(ValueError, match="cleanly"):
        subject.cell_from_boundaries(
            schedule="restart",
            collector="n1",
            actor_count=2,
            boundaries=boundaries,
            summary={"stopped_for_time": False, "halt": None, "actor_exitcodes": [0]},
        )
