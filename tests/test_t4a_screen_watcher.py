"""T4a watcher contracts: fixed CPU cap, receipt readiness, and job ordering."""

import argparse

from research.t4a_screen_20261016 import watch_t4a as watch


def test_controls_first_and_registered_round_and_world_counts():
    jobs = watch.base_jobs()
    assert len(jobs) == 33
    assert {job["name"] for job in jobs[:18]} == {"baseline"} | {
        f"CTRL_s{seed}" for seed in range(10, 15)
    }
    assert {job["round_index"] for job in jobs} == {65}
    assert all(job["worlds"] == (96 if job["mix"] == "scripted" else 32) for job in jobs)
    assert watch.WORKER_LIMIT == 2


def test_training_status_uses_only_t4a_lane_receipts(tmp_path):
    (tmp_path / "lane0-status.txt").write_text("TWO_SIDED_s10 exit=0\nTWO_SIDED_s12 exit=7\n")
    (tmp_path / "lane1-status.txt").write_text("unrelated exit=0\nTWO_SIDED_s11 exit=0\n")
    assert watch.training_status(tmp_path) == {
        "TWO_SIDED_s10": 0,
        "TWO_SIDED_s11": 0,
        "TWO_SIDED_s12": 7,
    }


def test_failure_stops_only_its_lane_and_waits_for_the_other_lane():
    assert not watch.training_terminal({"TWO_SIDED_s10": 7})
    assert watch.training_terminal(
        {
            "TWO_SIDED_s10": 7,
            "TWO_SIDED_s11": 0,
            "TWO_SIDED_s13": 0,
        }
    )


def test_candidate_readiness_requires_t4a_receipt_final_and_summary(tmp_path):
    name = "TWO_SIDED_s10"
    candidate = tmp_path / name
    candidate.mkdir()
    assert not watch.ready(name, tmp_path, tmp_path / "ctrl")
    (tmp_path / "lane0-status.txt").write_text(f"{name} exit=0\n")
    assert not watch.ready(name, tmp_path, tmp_path / "ctrl")
    (candidate / "final.pth").write_bytes(b"checkpoint")
    assert not watch.ready(name, tmp_path, tmp_path / "ctrl")
    (candidate / "summary.json").write_text("{}")
    assert watch.ready(name, tmp_path, tmp_path / "ctrl")


def test_probe_command_has_cpu_t2_path_and_no_candidate_veto(tmp_path):
    args = argparse.Namespace(train=tmp_path / "train", ctrl=tmp_path / "ctrl", out=tmp_path)
    candidate = watch.probe_command(watch.base_jobs()[-1], args)
    baseline = watch.probe_command(
        next(job for job in watch.base_jobs() if job["name"] == "baseline"), args
    )
    assert "research.t2_screen_20261015.probe_t2" in candidate
    assert candidate[candidate.index("--veto") + 1] == "none"
    assert baseline[baseline.index("--veto") + 1] == "v8"


def test_controls_complete_requires_all_initial_control_rows():
    jobs = watch.base_jobs()
    assert not watch._controls_complete(jobs[:17], jobs)
    assert watch._controls_complete(jobs[:18], jobs)
