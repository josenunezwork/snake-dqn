"""Contracts for the owner-run smoke, without requiring CUDA or Colab."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research.t16_colab import run_smoke as smoke


def _process_stopped(pid):
    import psutil

    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_recipe_matrix_and_default_off_selfplay(tmp_path):
    assert [(c["worlds"], c["actors"]) for c in smoke.CONFIGS] == [
        (128, 2),
        (128, 4),
        (128, 8),
        (64, 2),
        (128, 2),
        (128, 4),
    ]
    assert smoke.WARMUP["name"] not in {c["name"] for c in smoke.CONFIGS}
    for config in smoke.CONFIGS:
        command = smoke.trainer_command(
            config,
            tmp_path,
            tmp_path / "out",
            tmp_path / "init",
            150,
            "reduce-overhead",
            tmp_path / "ring",
        )
        assert ("--all-snakes-learner" in command) == config["self_play"]
        assert command[command.index("--device") + 1] == "cuda"
        assert command[command.index("--checkpoint-every") + 1] == str(smoke.SMOKE_CHECKPOINT_EVERY)
        assert command[command.index("--compile-learner") + 1] == "reduce-overhead"
        assert command[command.index("--batch-scale") + 1] == "2"


def test_post_warmup_rate_uses_interval_not_lifetime_average():
    rows = [
        {"wall": 10, "transitions": 100000, "updates": 10},
        {"wall": 40, "transitions": 200000, "updates": 20},
        {"wall": 90, "transitions": 1300000, "updates": 30},
    ]
    measured = smoke.measured_rate(rows)
    assert measured["transitions_per_second"] == 22000
    assert measured["measured_seconds"] == 50
    assert measured["first_update_wall"] == 10 and measured["window_start_wall"] == 40
    assert measured["tail_transitions_per_second"] == 22000
    with pytest.raises(ValueError, match="two"):
        smoke.measured_rate(rows[:2])
    rows[-1]["updates"] = 20
    with pytest.raises(ValueError, match="updates"):
        smoke.measured_rate(rows)


def test_steady_state_window_excludes_compile_and_replay_fill():
    # Actors run from the start but the learner's first update (replay fill + compile) lands
    # at 80 s; the window must open 20 s after that, not at the fixed 30 s warm-up.
    rows = [
        {"wall": 20, "transitions": 200000, "updates": 0},
        {"wall": 50, "transitions": 500000, "updates": 0},
        {"wall": 80, "transitions": 800000, "updates": 1},
        {"wall": 100, "transitions": 1000000, "updates": 100},
        {"wall": 110, "transitions": 1100000, "updates": 200},
        {"wall": 150, "transitions": 1500000, "updates": 600},
    ]
    measured = smoke.measured_rate(rows)
    assert measured["window_start_wall"] == 100
    assert measured["transitions_per_second"] == 10000
    assert measured["updates_per_second"] == 10
    with pytest.raises(ValueError, match="no telemetry row with a learner update"):
        smoke.measured_rate(rows[:2])
    with pytest.raises(ValueError, match="fewer than two"):
        smoke.measured_rate(rows[:4])


def test_checkpoint_restart_requires_receipt_hash_and_skips_aborted(tmp_path):
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    checkpoint = attempt / "ckpt_100k.pth"
    checkpoint.write_bytes(b"checkpoint")
    row = {"file": checkpoint.name, "sha256": smoke.sha256(checkpoint)}
    (attempt / "manifest.jsonl").write_text(json.dumps(row) + '\n{"partial":')
    assert smoke.latest_checkpoint(tmp_path) == checkpoint
    checkpoint.write_bytes(b"corrupt")
    assert smoke.latest_checkpoint(tmp_path) is None
    checkpoint.write_bytes(b"checkpoint")
    (attempt / "summary.json").write_text('{"halt": "NaN"}')
    assert smoke.latest_checkpoint(tmp_path) is None


def test_owned_subprocess_timeout(tmp_path):
    code = smoke.bounded_command(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "log", 0.1, {}
    )
    assert code == 124


def test_timeout_cleans_owned_descendant_even_after_parent_exits(tmp_path):
    pid_file = tmp_path / "child.pid"
    child = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
    )
    child_pid = None
    try:
        code = smoke.bounded_command([sys.executable, "-c", parent], tmp_path / "log", 1, {})
        assert code == 124
        child_pid = int(pid_file.read_text())
        for _ in range(20):
            if _process_stopped(child_pid):
                break
            time.sleep(0.05)
        else:
            pytest.fail("owned child survived timeout after its parent exited")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_outer_sigterm_cleans_separate_trainer_session(tmp_path):
    pid_file = tmp_path / "trainer.pid"
    trainer = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    runner = (
        "import os, sys, signal; from pathlib import Path; "
        "from research.t16_colab.run_smoke import bounded_command, interrupt_on_signal; "
        "signal.signal(signal.SIGTERM, interrupt_on_signal); "
        f"bounded_command([sys.executable, '-c', {trainer!r}], "
        f"Path({str(tmp_path / 'trainer.log')!r}), 30, dict(os.environ))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", runner],
        cwd=smoke.REPO,
        start_new_session=True,
        stderr=subprocess.DEVNULL,
    )
    trainer_pid = None
    try:
        for _ in range(100):
            if pid_file.exists():
                break
            time.sleep(0.05)
        trainer_pid = int(pid_file.read_text())
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=12)
        assert _process_stopped(trainer_pid)
    finally:
        for pid in (trainer_pid, process.pid):
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.wait(timeout=5)


def test_preflight_failure_still_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.subprocess, "check_output", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(
        smoke, "environment_info", lambda: (_ for _ in ()).throw(RuntimeError("no CUDA"))
    )
    assert smoke.run(tmp_path, tmp_path / "output", "a" * 40, 150, 1050, tmp_path / "s") == 1
    result = json.loads((tmp_path / "output" / "colab_throughput_results.json").read_text())
    assert result["status"] == "failed" and "no CUDA" in result["error"]
    assert result["configs"] == []


def test_parity_nodes_exist():
    for node in smoke.PARITY_TESTS:
        path, function = node.split("::")
        assert Path(smoke.REPO / path).is_file()
        assert f"def {function.split('[')[0]}(" in (smoke.REPO / path).read_text()


def test_run_completion_and_hash_checked_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "CONFIGS", [smoke.CONFIGS[0]])
    monkeypatch.setattr(smoke.subprocess, "check_output", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(
        smoke,
        "environment_info",
        lambda: {
            "gpu": "test GPU",
            "versions": {},
            "c_compiler": None,
            "cpu_count": 2,
        },
    )
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "B8_s3_final.pth").write_bytes(b"init")
    trainings = []

    def fake_command(command, log, timeout, env):
        if "--out" not in command:
            return 0
        attempt = Path(command[command.index("--out") + 1])
        trainings.append(attempt)
        attempt.mkdir(parents=True)
        (attempt / "summary.json").write_text(
            json.dumps(
                {
                    "halt": None,
                    "actor_exitcodes": [0, 0],
                    "rate": 10000,
                }
            )
        )
        (attempt / "final.pth").write_bytes(b"final")
        rows = [
            {"wall": 40, "transitions": 100000, "updates": 20},
            {"wall": 70, "transitions": 850000, "updates": 50},
            {"wall": 100, "transitions": 1600000, "updates": 80},
        ]
        (attempt / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return 0

    monkeypatch.setattr(smoke, "bounded_command", fake_command)
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    assert smoke.run(tmp_path, output, "a" * 40, 150, 1050, scratch) == 0
    # The unscored warm-up ran first (into scratch) and left its marker; then one scored run.
    assert trainings[0].parent.parent == scratch and smoke.WARMUP["name"] in str(trainings[0])
    assert (scratch / "warmup-done").exists()
    assert len(trainings) == 2
    result = json.loads((output / "colab_throughput_results.json").read_text())
    assert result["warmup"]["completed"] and result["status"] == "completed"
    row = result["configs"][0]
    published = Path(row["attempt"])
    assert published.parent.parent == output and (published / "final.pth").is_file()
    assert "resources" in row and row["transitions_per_second"] == 25000
    assert smoke.run(tmp_path, output, "a" * 40, 150, 1050, scratch) == 0
    assert len(trainings) == 2  # completed row reused; warm-up marker honoured
    (published / "final.pth").write_bytes(b"corrupt")
    assert smoke.run(tmp_path, output, "a" * 40, 150, 1050, scratch) == 0
    assert len(trainings) == 3
    result_path = output / "colab_throughput_results.json"
    saved = result_path.read_bytes()
    assert smoke.run(tmp_path, output, "a" * 40, 180, 1050, scratch) == 1
    assert result_path.read_bytes() == saved
    assert (output / "refused_identity.json").exists()
