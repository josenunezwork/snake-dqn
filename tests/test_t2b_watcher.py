"""Watcher routing and reuse checks for the T2b extension."""

from research.t2_screen_20261015 import watch_t2b as watch


def test_jobs_keep_reusable_controls_first():
    jobs = watch.base_jobs(60)
    assert len(jobs) == 48
    assert {j["name"] for j in jobs[:18]} == {"baseline"} | {f"CTRL_s{s}" for s in range(10, 15)}
    assert {j["name"].split("_s")[0] for j in jobs[18:]} == {"HUBER10", "HUBER30"}
    assert len({watch.job_key(j) for j in jobs}) == 48


def test_only_t2b_sessions_control_concurrency(monkeypatch):
    import subprocess

    def sessions(*args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0, stdout="t3-run\nt2b-lane0\nt2p-run\nt2b-lane1\n"
        )

    monkeypatch.setattr(watch.subprocess, "run", sessions)
    assert watch.training_lanes() == ["t2b-lane0", "t2b-lane1"]


def test_halted_checkpoint_never_ready(tmp_path):
    train, ctrl = tmp_path / "train", tmp_path / "ctrl"
    run = train / "HUBER10_s10"
    run.mkdir(parents=True)
    (run / "final.pth").write_bytes(b"not eligible")
    (run / "summary.json").write_text("{}")
    (train / "status.txt").write_text("HUBER10_s10 exit=1\n")
    assert not watch.ready("HUBER10_s10", train, ctrl)
    (train / "status.txt").write_text("HUBER10_s10 exit=0\n")
    assert watch.ready("HUBER10_s10", train, ctrl)


def test_session_lookup_failure_keeps_two_worker_limit(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        watch.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="connection refused"),
    )
    assert watch.training_lanes()  # A nonempty result retains the two-worker ceiling.


def test_reused_row_rejects_nonfinite_and_impossible_survival(tmp_path):
    import json
    import pytest
    from research.redesign_m2_20261008 import dev_probe

    job = {"name": "CTRL_s10", "mix": "scripted", "worlds": 96, "round_index": 61}
    worlds = list(map(str, dev_probe.probe_seeds(96, 61)))
    original = dict(
        job,
        schema="t2-screen-probe/v1",
        frames=5000,
        veto="none",
        ego2s_forward="rowwise",
        ego2s_mask="advisory",
        v61_backend="numba",
        anchors="fast",
        mass=dict.fromkeys(worlds, 100.0),
        deaths=dict.fromkeys(worlds, 0),
        alive_frames=dict.fromkeys(worlds, 5000),
        death_frame=dict.fromkeys(worlds),
        length_at_death=dict.fromkeys(worlds),
        causes=dict.fromkeys(worlds),
    )
    path = tmp_path / "row.json"
    for field, value in [("mass", float("nan")), ("alive_frames", 9000)]:
        row = json.loads(json.dumps(original))
        row[field][next(iter(row[field]))] = value
        path.write_text(json.dumps(row))
        with pytest.raises(ValueError, match="invalid"):
            watch.valid_row(path, job, None)
    row = json.loads(json.dumps(original))
    row["worlds"] = 2
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="stale row"):
        watch.valid_row(path, job, None)
