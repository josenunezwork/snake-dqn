"""T6 receipt-only watcher contracts."""

import argparse
import json

import pytest

from research.t6_screen_20261017 import watch


def test_registered_jobs_order_controls_before_restart_and_fix_rounds() -> None:
    jobs = watch.base_jobs()

    assert len(jobs) == 33
    assert {job["name"] for job in jobs[:18]} == {
        "baseline",
        *(f"CTRL_s{s}" for s in range(10, 15)),
    }
    assert {job["round_index"] for job in jobs} == {72}
    assert len(watch.extension_jobs()) == 10
    assert all(job["round_index"] == 73 for job in watch.extension_jobs())
    assert all(job["mix"] == "scripted" and job["worlds"] == 96 for job in watch.extension_jobs())
    assert watch.WORKER_LIMIT == 2


def test_training_status_uses_only_jsonl_receipts_and_requires_all_finals(tmp_path) -> None:
    for lane, seeds in ((0, (10, 12, 14)), (1, (11, 13))):
        (tmp_path / f"lane{lane}-status.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "out": str(tmp_path / f"RESTART_s{seed}"),
                        "exit_code": 0,
                        "success": True,
                    }
                )
                + "\n"
                for seed in seeds
            )
        )
    statuses = watch.training_status(tmp_path)

    assert statuses == {f"RESTART_s{seed}": 0 for seed in range(10, 15)}
    assert watch.training_terminal(statuses)
    assert not watch.training_terminal({"RESTART_s10": 0})


def test_restart_candidate_requires_clean_summary_and_success_receipt(tmp_path) -> None:
    name = "RESTART_s10"
    run = tmp_path / name
    run.mkdir()
    (tmp_path / "lane0-status.jsonl").write_text(
        json.dumps({"out": str(run), "exit_code": 0, "success": True}) + "\n"
    )
    (run / "final.pth").write_bytes(b"checkpoint")
    (run / "summary.json").write_text('{"halt": null, "actor_exitcodes": [0, 0]}')

    assert not watch.ready(name, tmp_path, tmp_path / "ctrl")
    (run / "summary.json").write_text(
        '{"halt": null, "actor_exitcodes": [0, 0], "transitions": 8000000}'
    )
    assert watch.ready(name, tmp_path, tmp_path / "ctrl")


def test_probe_command_is_cpu_rowwise_bare_network(tmp_path) -> None:
    args = argparse.Namespace(train=tmp_path / "train", ctrl=tmp_path / "ctrl", out=tmp_path)
    candidate = watch.probe_command(watch.base_jobs()[-1], args)
    baseline = watch.probe_command(
        next(job for job in watch.base_jobs() if job["name"] == "baseline"), args
    )

    assert "research.t6_screen_20261017.probe" in candidate
    assert candidate[candidate.index("--veto") + 1] == "none"
    assert baseline[baseline.index("--veto") + 1] == "v8"
    assert watch.environment()["SNAKE_DQN_DEVICE"] == "cpu"
    assert watch.environment()["OMP_NUM_THREADS"] == "1"


def test_mocked_scheduler_caps_at_two_and_holds_restart_until_18_controls_finish() -> None:
    jobs = watch.base_jobs()
    controls = jobs[:18]
    ready = {job["name"] for job in jobs}

    before = watch.launchable_jobs(
        jobs,
        controls[:-1],
        set(),
        controls_complete=False,
        ready_names=ready,
    )
    after = watch.launchable_jobs(
        jobs,
        controls,
        set(),
        controls_complete=True,
        ready_names=ready,
    )

    assert len(before) == 1
    assert len(after) == 2
    assert all(job["name"] not in watch.NAMES for job in before)
    assert all(job["name"] in watch.NAMES for job in after)


def test_extension_scheduler_adds_the_ten_registered_rows_only_once() -> None:
    result = {"arms": {"RESTART": {"decision": "EXTEND"}}}

    assert watch.extension_due(result, extension_checked=False)
    assert not watch.extension_due(result, extension_checked=True)
    assert len(watch.extension_jobs()) == 10


def test_row_requires_detached_terminal_masses(tmp_path, monkeypatch) -> None:
    job = watch.base_jobs()[0]
    path = watch.row_path(tmp_path, job)
    path.parent.mkdir()
    path.write_text("{}")
    monkeypatch.setattr(watch, "valid_row", lambda *_args: True)

    with pytest.raises(ValueError, match="final-length"):
        watch.valid_t6_row(path, job, None)


def test_failed_receipt_is_terminal_before_any_gameplay_probe(tmp_path, monkeypatch) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "lane0-status.jsonl").write_text(
        json.dumps(
            {
                "out": str(train / "RESTART_s10"),
                "exit_code": 9,
                "success": False,
            }
        )
        + "\n"
    )
    args = argparse.Namespace(
        train=train,
        ctrl=tmp_path / "ctrl",
        out=tmp_path / "out",
        poll_seconds=0,
    )
    finished: list[tuple[list[dict], dict[str, int]]] = []
    monkeypatch.setattr(watch, "_audit", lambda *_args, **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(
        watch,
        "_finish",
        lambda _args, rows, statuses: finished.append((rows, statuses)) or 0,
    )
    monkeypatch.setattr(
        watch.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("failed arm must not launch a probe"),
    )

    assert watch.run(args) == 0
    assert finished == [([], {"RESTART_s10": 9})]
