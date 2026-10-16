"""T4c watcher contracts: receipt-only training state and registered CPU jobs."""

import argparse

import pytest

from research.t3_screen_20261015 import watch_t4c as watch


def test_controls_first_registered_round_and_cap():
    jobs = watch.base_jobs()
    assert len(jobs) == 33
    assert {job["name"] for job in jobs[:18]} == {
        "baseline",
        *(f"CTRL_s{s}" for s in range(10, 15)),
    }
    assert {job["round_index"] for job in jobs} == {66}
    assert all(job["worlds"] == (96 if job["mix"] == "scripted" else 32) for job in jobs)
    assert watch.WORKER_LIMIT == 2


def test_statuses_read_only_t4c_receipts_and_terminal_lanes(tmp_path):
    (tmp_path / "lane0-status.txt").write_text("TWO_SIDED_T_s10 exit=0\nTWO_SIDED_T_s12 exit=7\n")
    (tmp_path / "lane1-status.txt").write_text("TWO_SIDED_T_s11 exit=0\nTWO_SIDED_T_s13 exit=0\n")
    statuses = watch.training_status(tmp_path)
    assert statuses["TWO_SIDED_T_s12"] == 7
    assert watch.training_terminal(statuses)


def test_candidate_readiness_needs_receipt_final_and_summary(tmp_path):
    name = "TWO_SIDED_T_s10"
    run = tmp_path / name
    run.mkdir()
    assert not watch.ready(name, tmp_path, tmp_path / "ctrl")
    (tmp_path / "lane0-status.txt").write_text(f"{name} exit=0\n")
    (run / "final.pth").write_bytes(b"checkpoint")
    assert not watch.ready(name, tmp_path, tmp_path / "ctrl")
    (run / "summary.json").write_text("{}")
    assert watch.ready(name, tmp_path, tmp_path / "ctrl")


def test_probe_command_is_t4c_cpu_path_without_candidate_veto(tmp_path):
    args = argparse.Namespace(train=tmp_path / "train", ctrl=tmp_path / "ctrl", out=tmp_path)
    candidate = watch.probe_command(watch.base_jobs()[-1], args)
    baseline = watch.probe_command(
        next(job for job in watch.base_jobs() if job["name"] == "baseline"), args
    )
    assert "research.t3_screen_20261015.probe_t4c" in candidate
    assert candidate[candidate.index("--veto") + 1] == "none"
    assert baseline[baseline.index("--veto") + 1] == "v8"
    assert watch.environment()["SNAKE_DQN_DEVICE"] == "cpu"


def test_t4c_rows_require_detached_final_lengths(tmp_path, monkeypatch):
    job = watch.base_jobs()[0]
    path = watch.row_path(tmp_path, job)
    path.parent.mkdir()
    path.write_text("{}")
    monkeypatch.setattr(watch, "valid_row", lambda *_args: True)
    with pytest.raises(ValueError, match="final-length"):
        watch.valid_t4c_row(path, job, None)
