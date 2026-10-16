"""Operational boundaries for the diagnostic curve watcher."""

import json
from pathlib import Path

import pytest

from research.t4_curve_20261015 import watch


def test_jobs_cover_exact_registered_curve():
    jobs = watch.jobs(Path("/train"), Path("/ctrl"))
    assert len(jobs) == len({j["name"] for j in jobs}) == 70
    assert sum(j["run"].startswith("CTRL_") for j in jobs) == 10
    assert {j["round_index"] for j in jobs} == {64}
    assert {j["mix"] for j in jobs} == {"scripted"}
    assert {j["worlds"] for j in jobs} == {32}
    for job in jobs:
        command = watch.command(job, Path("/out/result.json"))
        for flag, expected in (
            ("--veto", "none"),
            ("--ego2s-forward", "rowwise"),
            ("--round-index", "64"),
            ("--frames", "5000"),
        ):
            assert command[command.index(flag) + 1] == expected
        assert "mps" not in command
    env = watch.environment()
    assert env["SNAKE_DQN_DEVICE"] == "cpu"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert env[key] == "1"


def test_candidate_waits_for_successful_complete_run(tmp_path):
    job = watch.jobs(tmp_path, tmp_path)[10]
    ckpt = Path(job["checkpoint"])
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"checkpoint")
    summary = ckpt.parent / "summary.json"
    summary.write_text(json.dumps({"transitions": 8_000_000, "halt": None}))
    assert not watch.ready(job, {})
    assert not watch.ready(job, {job["run"]: 1})
    assert watch.ready(job, {job["run"]: 0})
    summary.write_text(json.dumps({"transitions": 7_000_000, "halt": None}))
    with pytest.raises(ValueError, match="Incomplete"):
        watch.ready(job, {job["run"]: 0})


def test_cached_rows_cannot_bypass_terminal_training_status(tmp_path, monkeypatch):
    from argparse import Namespace

    from research.redesign_m2_20261008.dev_probe import probe_seeds

    out = tmp_path / "out"
    out.mkdir()
    (out / "seed_bank_audit.json").write_text(
        json.dumps(
            {"pass": True, "round_index": 64, "worlds": 32, "seeds": sorted(probe_seeds(32, 64))}
        )
    )
    job = watch.jobs(tmp_path, tmp_path)[10]
    checkpoint = Path(job["checkpoint"])
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parent / "summary.json").write_text(
        json.dumps({"transitions": 8_000_000, "halt": None})
    )
    monkeypatch.setattr(watch, "jobs", lambda *_: [job])
    monkeypatch.setattr(watch, "valid_row", lambda *_: True)
    monkeypatch.setattr(watch, "training_status", lambda *_: {})

    def stop_after_poll(_seconds):
        raise InterruptedError("test ended after first poll")

    monkeypatch.setattr(watch.time, "sleep", stop_after_poll)
    with pytest.raises(InterruptedError, match="first poll"):
        watch.run(Namespace(out=out, train=tmp_path, ctrl=tmp_path, poll_seconds=1))
    progress = json.loads((out / "progress.json").read_text())
    assert progress["completed"] == 0
    assert progress["active"] == []
