"""T11 terminal-failure and fixed allocation contracts."""

import argparse
import json

from research.t11_screen_20261017 import watch


def test_fixed_allocation_and_process_budget() -> None:
    jobs = watch.base_jobs()
    assert len(jobs) == 33
    assert {j["round_index"] for j in jobs} == {74}
    assert watch.WORKER_LIMIT + 1 <= 4  # watcher plus CPU children
    assert len(watch.extension_jobs()) == 10
    assert {j["round_index"] for j in watch.extension_jobs()} == {75}
    assert all(j["mix"] == "scripted" for j in watch.extension_jobs())


def test_summary_halt_overrides_zero_receipt(tmp_path) -> None:
    run = tmp_path / "LOCAL30_s10"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"halt": "boost", "transitions": 8000000, "actor_exitcodes": [0, 0]})
    )
    (tmp_path / "lane0-status.jsonl").write_text(
        json.dumps({"out": str(run), "exit_code": 0, "success": True}) + "\n"
    )
    assert watch.training_status(tmp_path) == {"LOCAL30_s10": 98}


def test_failed_seed_never_launches_gameplay(tmp_path, monkeypatch) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "lane1-status.jsonl").write_text(
        json.dumps({"out": str(train / "LOCAL30_s13"), "exit_code": 1, "success": False}) + "\n"
    )
    monkeypatch.setattr(watch, "_audit", lambda *a, **k: {"status": "PASS"})

    def forbidden(*args, **kwargs):
        raise AssertionError("gameplay launched after registered KILL")

    monkeypatch.setattr(watch.subprocess, "Popen", forbidden)
    observed = []
    monkeypatch.setattr(watch, "_finish", lambda a, r, s: observed.append((r, s)) or 0)
    args = argparse.Namespace(train=train, ctrl=tmp_path, out=tmp_path / "out", poll_seconds=0)
    assert watch.run(args) == 0
    assert observed == [([], {"LOCAL30_s13": 1})]


def test_extension_only_for_extend() -> None:
    assert not watch.extension_due({"arms": {"LOCAL30": {"decision": "KILL"}}}, False)
    assert watch.extension_due({"arms": {"LOCAL30": {"decision": "EXTEND"}}}, False)
    assert not watch.extension_due({"arms": {"LOCAL30": {"decision": "EXTEND"}}}, True)
