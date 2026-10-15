"""T3 job ordering, fixed probe contract, and resource ceilings."""

import argparse
import subprocess

from research.t3_screen_20261015 import watch_t3 as watch


def test_controls_first_and_registered_world_counts():
    jobs = watch.base_jobs()
    assert len(jobs) == 63
    assert {j["name"] for j in jobs[:18]} == {"baseline"} | {f"CTRL_s{s}" for s in range(10, 15)}
    assert all(j["round_index"] == 62 for j in jobs)
    assert all(j["worlds"] == (96 if j["mix"] == "scripted" else 32) for j in jobs)
    assert len({watch.job_key(job) for job in jobs}) == len(jobs)


def test_worker_limit_waits_for_terminal_status_and_lane_exit(monkeypatch):
    statuses = dict.fromkeys(watch.NAMES, 0)
    monkeypatch.setattr(watch, "training_lanes", lambda: [])
    assert watch.worker_limit({}) == 2
    assert watch.worker_limit(statuses) == 4
    monkeypatch.setattr(watch, "training_lanes", lambda: ["t3-lane0"])
    assert watch.worker_limit(statuses) == 2


def test_training_lookup_error_is_conservative(monkeypatch):
    monkeypatch.setattr(
        watch.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="refused"),
    )
    assert watch.training_lanes()


def test_probe_command_reuses_t2_with_incumbent_only_veto(tmp_path):
    args = argparse.Namespace(train=tmp_path / "train", ctrl=tmp_path / "ctrl", out=tmp_path)
    jobs = watch.base_jobs()
    candidate = watch.probe_command(jobs[-1], args)
    baseline = watch.probe_command(next(j for j in jobs if j["name"] == "baseline"), args)
    assert "research.t2_screen_20261015.probe_t2" in candidate
    assert candidate[candidate.index("--veto") + 1] == "none"
    assert baseline[baseline.index("--veto") + 1] == "v8"
    assert "--student" not in baseline
