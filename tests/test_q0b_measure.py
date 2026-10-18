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
        "emitted_replay_rows": decisions,
        "completed_updates": decisions // 256,
        "pending_update_debt": decisions % 256,
        "truncation_rows_dropped": 0,
        "pending_nstep_rows": 0,
        "pending_nstep_rows_by_actor": {"0": 0, "1": 0},
        "decision_replay_delta": 0,
        "nstep_accounting_reconciled": True,
        "compile_ready": True,
        "replay_ready": True,
        "finite_updates": True,
        "dropped_transitions": 0,
        "ring_overruns": 0,
        "target_lag": {"mean": 1.0, "p95": 2.0},
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
        summary={"stopped_for_time": True, "halt": None, "actor_exitcodes": [0, 0]},
    )
    assert cell["scored_windows"] == [
        {"seconds": 180.0, "learner_decisions": 3000},
        {"seconds": 180.0, "learner_decisions": 4000},
    ]
    assert cell["counters"]["completed_updates"] == 31


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
            actor_count=2,
            boundaries=boundaries,
            summary={"stopped_for_time": True, "halt": None, "actor_exitcodes": [0, 0]},
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
            summary={"stopped_for_time": True, "halt": None, "actor_exitcodes": [0, 0]},
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
            summary={"stopped_for_time": False, "halt": None, "actor_exitcodes": [0, 0]},
        )


def test_recorder_preserves_two_full_windows_after_jitter(monkeypatch, tmp_path):
    from research.redesign_m3c_20261008.q0b_timing import Q0bBoundaryRecorder

    monkeypatch.setenv("Q0Q1_BOUNDARY_OUT", str(tmp_path / "bounds.json"))
    recorder = Q0bBoundaryRecorder()
    recorder.observe(120.2, {"learner_decisions": 100})
    recorder.observe(300.1, {"learner_decisions": 200})
    assert list(recorder.records) == ["warmup"]
    recorder.observe(300.3, {"learner_decisions": 300})
    recorder.observe(480.2, {"learner_decisions": 400})
    assert not recorder.complete
    recorder.observe(480.4, {"learner_decisions": 500})
    assert recorder.complete
    assert recorder.records["window2"]["elapsed_seconds"] == 480.4


def test_recorder_cannot_manufacture_multiple_windows_after_stall(monkeypatch, tmp_path):
    from research.redesign_m3c_20261008.q0b_timing import Q0bBoundaryRecorder

    monkeypatch.setenv("Q0Q1_BOUNDARY_OUT", str(tmp_path / "bounds.json"))
    recorder = Q0bBoundaryRecorder()
    recorder.observe(700, {"learner_decisions": 100})
    assert list(recorder.records) == ["warmup"]
    recorder.observe(879, {"learner_decisions": 200})
    assert list(recorder.records) == ["warmup"]
    recorder.observe(880, {"learner_decisions": 300})
    assert list(recorder.records) == ["warmup", "window1"]


def test_stale_replay_filter_counts_each_discarded_transition_once():
    import numpy as np

    from research.redesign_m3c_20261008.train_m3c import Replay

    class Ring:
        def valid(self, ids):
            return ids >= 10

    replay = Replay(1)
    replay.add(0, 0, 0.0, 0.99, 1)
    rng = np.random.default_rng(1)
    assert not len(replay.sample(rng, 10, Ring()))
    assert replay.stale_rows_filtered == 1
    replay.sample(rng, 10, Ring())
    assert replay.stale_rows_filtered == 1
    replay.add(1, 0, 0.0, 0.99, 2)
    replay.sample(rng, 10, Ring())
    assert replay.stale_rows_filtered == 2


@pytest.mark.parametrize(
    "schedule,collector,arm",
    [
        ("restart", "n3", "A"),
        ("mature", "n3", "B"),
        ("restart", "n1", "C"),
        ("mature", "n1", "D"),
    ],
)
def test_measured_command_matches_exact_normalized_q1_recipe(schedule, collector, arm):
    import argparse
    import sys
    from pathlib import Path

    from research.q0q1_20261017 import run_queue_job

    inputs = Path("/content/q0q1-inputs")
    args = argparse.Namespace(b8=inputs / "init/B8_s3_final.pth", demo_data=inputs / "demo")
    measured = subject.registered_command(args, schedule, collector, 3, Path("/tmp/q0b-cell"))
    queued = run_queue_job.command(
        {
            "Q0Q1_JOB_KIND": "q1_train",
            "Q0Q1_INPUT_ROOT": str(inputs),
            "Q0Q1_PYTHON": sys.executable,
            "Q1_ARM": arm,
            "Q1_ACTORS": "3",
            "Q1_SEED": "261799",
            "Q1_TRANSITIONS": "8000000",
        },
        Path("/tmp/q1-training"),
    )
    assert run_queue_job.normalized_command_sha(measured) == run_queue_job.normalized_command_sha(
        queued
    )
    if collector == "n1":
        assert measured[measured.index("--namespace") + 1].startswith(
            "q1-task-aligned-n1-20261017/"
        )


def test_cuda_fixture_producer_fails_closed_without_cuda(monkeypatch, tmp_path):
    import json

    import torch

    from research.q0q1_20261017 import q0b_cuda_fixtures

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "fixture.json"
    assert q0b_cuda_fixtures.main(["--fixture", "t22", "--out", str(out)]) == 2
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "INCOMPLETE"
    assert "CUDA is unavailable" in receipt["error"]
    assert "short_body" not in receipt


def test_cell_launcher_accounts_actual_readiness_and_cleanup(monkeypatch, tmp_path):
    import argparse
    import json
    import subprocess

    args = argparse.Namespace(
        trainer_command=[
            "python",
            "--out",
            "{out}",
            "--seed",
            "1",
            "--namespace",
            "test",
            "--max-wall-seconds",
            "5100",
            "--batch-scale",
            "2",
            "--actor-queue-frames",
            "2",
            "--learner-precision",
            "fp32",
        ],
        b8=tmp_path / "b8",
        demo_data=tmp_path / "demo",
        repo_root=tmp_path,
    )
    times = iter((100.0, 700.0))
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(times))

    def run(command, *, env, cwd):
        out = tmp_path / "restart-n1"
        out.mkdir()
        records = {
            phase: _boundary(phase, elapsed, decisions)
            for phase, elapsed, decisions in (
                ("warmup", 120.0, 1000),
                ("window1", 300.0, 4000),
                ("window2", 480.0, 8000),
            )
        }
        for record in records.values():
            record["measurement_started_monotonic"] = 110.0
            record["observed_monotonic"] = 110.0 + record["elapsed_seconds"]
        (out / "q0b-boundaries.json").write_text(json.dumps(records))
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "stopped_for_time": True,
                    "halt": None,
                    "actor_exitcodes": [0, 0],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subject, "_run", run)
    cell = subject.run_cell(args, "restart", "n1", 2, tmp_path)
    assert cell["setup_seconds"] == 10.0
    assert cell["cleanup_seconds"] == 110.0
    assert cell["observed_launcher_seconds"] == 600.0


@pytest.mark.parametrize("phase", ["warmup", "window1", "window2"])
def test_registered_nstep_endpoint_rows_are_distinct_from_operational_loss(phase):
    boundaries = {
        "warmup": _boundary("warmup", 120.0, 1000),
        "window1": _boundary("window1", 300.0, 4000),
        "window2": _boundary("window2", 480.0, 8000),
    }
    boundary = boundaries[phase]
    boundary.update(
        emitted_replay_rows=boundary["learner_decisions"] - 9,
        truncation_rows_dropped=7,
        pending_nstep_rows=2,
        decision_replay_delta=9,
        pending_nstep_rows_by_actor={"0": 1, "1": 1},
    )
    cell = subject.cell_from_boundaries(
        schedule="restart",
        collector="n3",
        actor_count=2,
        boundaries=boundaries,
        summary={"stopped_for_time": True, "halt": None, "actor_exitcodes": [0, 0]},
    )
    assert cell["dropped_transitions"] == 0
    if phase == "window2":
        assert cell["counters"]["truncation_rows_dropped"] == 7
        assert cell["counters"]["pending_nstep_rows"] == 2
        assert cell["counters"]["decision_replay_delta"] == 9
    boundary["truncation_rows_dropped"] = 8
    with pytest.raises(ValueError, match="does not reconcile exactly"):
        subject.cell_from_boundaries(
            schedule="restart",
            collector="n3",
            actor_count=2,
            boundaries=boundaries,
            summary={"stopped_for_time": True, "halt": None, "actor_exitcodes": [0, 0]},
        )
