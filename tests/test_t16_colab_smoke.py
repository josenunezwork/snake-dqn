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
        assert "--obs-pack" in command and "--learner-precision" not in command


def test_smoke03_matrix_varies_only_the_learner_host_work_and_actor_count(tmp_path):
    """smoke-03: 128-world self-play rows; raw rings, batch scale, bf16 and six actors."""
    assert smoke.matrix_configs("smoke02") == list(smoke.CONFIGS)
    assert smoke.matrix_configs("smoke03") == list(smoke.SMOKE03_CONFIGS)
    with pytest.raises(ValueError):
        smoke.matrix_configs("smoke99")
    names = [c["name"] for c in smoke.SMOKE03_CONFIGS]
    assert names == [
        "sp_w128_a2_packed_g0_bs2_fp32",
        "sp_w128_a2_raw_g0_bs2_fp32",
        "sp_w128_a3_raw_g3_bs2_fp32",
        "sp_w128_a4_raw_g3_bs2_fp32",
        "sp_w128_a6_raw_g3_bs2_fp32",
        "sp_w128_a4_raw_g3_bs4_fp32",
        "sp_w128_a4_raw_g3_bs2_bf16",
    ]
    assert not set(names) & {c["name"] for c in smoke.CONFIGS}
    for config in smoke.SMOKE03_CONFIGS:
        assert (config["worlds"], config["self_play"]) == (128, True)
        command = smoke.trainer_command(
            config, tmp_path, tmp_path / "out", tmp_path / "init", 150, "off", tmp_path / "ring"
        )
        assert "--all-snakes-learner" in command and "--compile-learner" not in command
        assert ("--obs-pack" in command) == config.get("obs_pack", True)
        assert command[command.index("--batch-scale") + 1] == str(config.get("batch_scale", 2))
        assert command[command.index("--actors") + 1] == str(config["actors"])
        if config.get("precision", "fp32") == "fp32":
            assert "--learner-precision" not in command
        else:
            assert command[command.index("--learner-precision") + 1] == config["precision"]
    raw = [c for c in smoke.SMOKE03_CONFIGS if c.get("obs_pack", True) is False]
    assert len(raw) == 6 and smoke.SMOKE03_CONFIGS[0].get("obs_pack", True) is True


def test_smoke03_matrix_is_the_preregistered_seven_row_recipe(tmp_path):
    assert [(c["actors"], c["gather_threads"]) for c in smoke.SMOKE03_CONFIGS] == [
        (2, 0),
        (2, 0),
        (3, 3),
        (4, 3),
        (6, 3),
        (4, 3),
        (4, 3),
    ]
    assert all(c["worlds"] == 128 and c["self_play"] for c in smoke.SMOKE03_CONFIGS)
    assert all(c["actor_queue_frames"] == 4 for c in smoke.SMOKE03_CONFIGS)
    assert [c.get("obs_pack", True) for c in smoke.SMOKE03_CONFIGS] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert [c.get("batch_scale", 2) for c in smoke.SMOKE03_CONFIGS] == [2, 2, 2, 2, 2, 4, 2]
    assert [c.get("precision", "fp32") for c in smoke.SMOKE03_CONFIGS] == ["fp32"] * 6 + ["bf16"]
    assert [c.get("throughput_only", False) for c in smoke.SMOKE03_CONFIGS] == [False] * 5 + [
        True,
        True,
    ]
    assert smoke.warmup_for_suite("smoke03") == smoke.SMOKE03_WARMUP
    assert smoke.SMOKE03_WARMUP["self_play"]
    command = smoke.trainer_command(
        smoke.SMOKE03_CONFIGS[-1],
        tmp_path,
        tmp_path / "out",
        tmp_path / "init",
        600,
        "reduce-overhead",
        tmp_path / "ring",
    )
    assert command[command.index("--gather-threads") + 1] == "3"
    assert command[command.index("--actor-queue-frames") + 1] == "4"
    assert "--all-snakes-learner" in command


def test_smoke03_row5_is_only_explicit_optional_omission():
    configs = smoke.configs_for_suite("smoke03", omit_smoke03_row5=True)
    assert len(configs) == 6
    assert smoke.SMOKE03_CONFIGS[4] not in configs


def test_runner_cap_is_3600_seconds(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_smoke.py",
            "--inputs",
            "/tmp/i",
            "--output",
            "/tmp/o",
            "--commit",
            "a" * 40,
            "--budget-seconds",
            "3601",
        ],
    )
    with pytest.raises(SystemExit):
        smoke.main()


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


def test_measured_rate_reports_fixed_batch_and_learner_metrics():
    rows = [
        {"wall": 40, "transitions": 1_000, "updates": 4, "learn_s": 0.1, "policy_lag_mean": 2},
        {
            "wall": 70,
            "transitions": 257_000,
            "updates": 1_004,
            "learn_s": 12.1,
            "policy_lag_mean": 4,
        },
        {
            "wall": 100,
            "transitions": 513_000,
            "updates": 2_004,
            "learn_s": 24.1,
            "policy_lag_mean": 6,
        },
    ]
    measured = smoke.measured_rate(rows)
    assert measured["transitions_per_update"] == 256
    assert measured["learn_seconds_per_update"] == pytest.approx(0.012)
    assert measured["policy_lag_mean"] == 5


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
    assert result["python"] == smoke.platform.python_version()
    assert result["python_executable"] == sys.executable


def test_parity_nodes_exist():
    for node in smoke.PARITY_TESTS:
        path, function = node.split("::")
        assert Path(smoke.REPO / path).is_file()
        assert f"def {function.split('[')[0]}(" in (smoke.REPO / path).read_text()


def test_run_completion_and_hash_checked_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.sys, "executable", "/managed-312/bin/python")
    monkeypatch.setattr(smoke, "start_sampler", lambda *args: object())
    monkeypatch.setattr(smoke, "stop_sampler", lambda *args: {})
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
        assert command[0] == "/managed-312/bin/python"
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


def test_smoke03_profile_is_warmed_and_result_identity_resumes(tmp_path, monkeypatch):
    config = dict(smoke.SMOKE03_CONFIGS[2])
    monkeypatch.setattr(smoke, "SMOKE03_CONFIGS", (config,))
    monkeypatch.setattr(smoke, "SMOKE03_PROFILE_CONFIG", config["name"])
    monkeypatch.setattr(smoke, "start_sampler", lambda *args: object())
    monkeypatch.setattr(smoke, "stop_sampler", lambda *args: {})
    monkeypatch.setattr(smoke.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(
        smoke,
        "environment_info",
        lambda: {"gpu": "test GPU", "versions": {}, "c_compiler": None, "cpu_count": 2},
    )
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "B8_s3_final.pth").write_bytes(b"init")
    calls = []

    def fake_command(command, log, timeout, env):
        if "--out" not in command:
            return 0
        attempt = Path(command[command.index("--out") + 1])
        calls.append((attempt, list(command), dict(env)))
        attempt.mkdir(parents=True)
        if env.get("M3C_PROFILE") == "cprofile":
            profile = attempt / "profile"
            profile.mkdir()
            (profile / "learner.pstats").write_bytes(b"profile")
            (profile / "learner.meta.json").write_text(
                '{"mode": "cprofile", "profiled_s": 60.0, "requested_duration_s": 60.0, '
                '"warm_from_first_tick": true, "warm_s": 60.000000000007276}'
            )
        (attempt / "summary.json").write_text(
            json.dumps({"halt": None, "actor_exitcodes": [0, 0, 0], "rate": 10_000})
        )
        (attempt / "final.pth").write_bytes(b"final")
        (attempt / "log.jsonl").write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {"wall": 40, "transitions": 1_000, "updates": 4},
                    {"wall": 70, "transitions": 257_000, "updates": 1_004},
                    {"wall": 100, "transitions": 513_000, "updates": 2_004},
                )
            )
        )
        return 0

    monkeypatch.setattr(smoke, "bounded_command", fake_command)
    output, scratch = tmp_path / "output", tmp_path / "scratch"
    assert smoke.run(tmp_path, output, "a" * 40, 150, 1_200, scratch, "smoke03") == 0
    profile_call = next(call for call in calls if call[2].get("M3C_PROFILE") == "cprofile")
    assert profile_call[2]["M3C_PROFILE_WARM_S"] == "60"
    assert profile_call[2]["M3C_PROFILE_WARM_FROM_FIRST_TICK"] == "1"
    assert profile_call[2]["M3C_PROFILE_DURATION_S"] == "60"
    assert "--max-wall-seconds" in profile_call[1]
    assert profile_call[1][profile_call[1].index("--max-wall-seconds") + 1] == "125"
    result = json.loads((output / "colab_throughput_results.json").read_text())
    assert result["suite"] == "smoke03"
    assert result["profile"]["seconds"] == 60 and result["profile"]["timed_after_warmup"]
    assert result["profile"]["profile_metadata"]["learner.meta.json"]["profiled_s"] == 60.0
    assert smoke.completed_profile_artifacts_match(result["profile"])
    first_count = len(calls)
    assert smoke.run(tmp_path, output, "a" * 40, 150, 1_200, scratch, "smoke03") == 0
    assert len(calls) == first_count


def test_smoke03_retries_a_failed_profile_before_completing(tmp_path, monkeypatch):
    config = dict(smoke.SMOKE03_CONFIGS[2])
    monkeypatch.setattr(smoke, "SMOKE03_CONFIGS", (config,))
    monkeypatch.setattr(smoke, "SMOKE03_PROFILE_CONFIG", config["name"])
    monkeypatch.setattr(smoke, "start_sampler", lambda *args: object())
    monkeypatch.setattr(smoke, "stop_sampler", lambda *args: {})
    monkeypatch.setattr(smoke.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(
        smoke,
        "environment_info",
        lambda: {"gpu": "test GPU", "versions": {}, "c_compiler": None, "cpu_count": 2},
    )
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "B8_s3_final.pth").write_bytes(b"init")
    profile_runs = 0

    def fake_command(command, log, timeout, env):
        nonlocal profile_runs
        if "--out" not in command:
            return 0
        attempt = Path(command[command.index("--out") + 1])
        attempt.mkdir(parents=True)
        if env.get("M3C_PROFILE") == "cprofile":
            profile_runs += 1
            profile = attempt / "profile"
            profile.mkdir()
            (profile / "learner.pstats").write_bytes(b"profile")
            profiled_s = 25.0 if profile_runs == 1 else 60.0
            (profile / "learner.meta.json").write_text(
                '{"mode": "cprofile", "profiled_s": '
                + str(profiled_s)
                + ', "requested_duration_s": 60.0, "warm_from_first_tick": true, "warm_s": 60.0}'
            )
        (attempt / "summary.json").write_text(
            json.dumps({"halt": None, "actor_exitcodes": [0, 0, 0], "rate": 10_000})
        )
        (attempt / "final.pth").write_bytes(b"final")
        (attempt / "log.jsonl").write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {"wall": 40, "transitions": 1_000, "updates": 4},
                    {"wall": 70, "transitions": 257_000, "updates": 1_004},
                    {"wall": 100, "transitions": 513_000, "updates": 2_004},
                )
            )
        )
        return 0

    monkeypatch.setattr(smoke, "bounded_command", fake_command)
    output, scratch = tmp_path / "output", tmp_path / "scratch"
    assert smoke.run(tmp_path, output, "a" * 40, 150, 1_200, scratch, "smoke03") == 1
    failed = json.loads((output / "colab_throughput_results.json").read_text())
    assert failed["profile"]["returncode"] == 0  # reject the truncated capture, not process failure
    assert failed["profile"]["profile_metadata"]["learner.meta.json"]["profiled_s"] == 25.0
    assert not smoke.completed_profile_artifacts_match(failed["profile"])
    assert smoke.run(tmp_path, output, "a" * 40, 150, 1_200, scratch, "smoke03") == 0
    assert profile_runs == 2


def test_sampler_uses_runner_interpreter(tmp_path, monkeypatch):
    """Sampling uses the same managed interpreter as parity and training."""
    from unittest.mock import Mock

    monkeypatch.setattr(smoke.sys, "executable", "/managed-312/bin/python")
    popen = Mock()
    monkeypatch.setattr(smoke.subprocess, "Popen", popen)
    smoke.start_sampler(tmp_path / "resources.jsonl", {})
    assert popen.call_args.args[0][:3] == [
        "/managed-312/bin/python",
        "-m",
        "research.t16_colab.sampler",
    ]


def test_copy_attempt_is_best_effort(tmp_path, monkeypatch):
    missing = tmp_path / "never-created"
    assert smoke.copy_attempt(missing, tmp_path / "drive" / "attempt-001") is False
    source = tmp_path / "local" / "attempt-001"
    source.mkdir(parents=True)
    (source / "summary.json").write_text("{}")
    destination = tmp_path / "drive" / "attempt-001"
    assert smoke.copy_attempt(source, destination) is True
    assert (destination / "summary.json").exists()

    def broken(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(smoke.shutil, "copytree", broken)
    monkeypatch.setattr(smoke.time, "sleep", lambda seconds: None)
    assert smoke.copy_attempt(source, destination) is False


def test_finite_json_replaces_non_finite_floats():
    value = {"rate": float("nan"), "nested": [1.0, float("inf"), {"ok": 2}], "text": "x"}
    assert smoke.finite_json(value) == {"rate": None, "nested": [1.0, None, {"ok": 2}], "text": "x"}
    json.dumps(smoke.finite_json(value), allow_nan=False)


def test_trainer_dying_before_output_does_not_abort_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "CONFIGS", [smoke.CONFIGS[0], smoke.CONFIGS[1]])
    monkeypatch.setattr(smoke.subprocess, "check_output", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(
        smoke,
        "environment_info",
        lambda: {"gpu": "test GPU", "versions": {}, "c_compiler": None, "cpu_count": 2},
    )
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "B8_s3_final.pth").write_bytes(b"init")

    def fake_command(command, log, timeout, env):
        if "--out" not in command:
            return 0
        attempt = Path(command[command.index("--out") + 1])
        if smoke.WARMUP["name"] in str(attempt) or "hero_w128_a4" in str(attempt):
            attempt.mkdir(parents=True)
            (attempt / "summary.json").write_text(
                json.dumps({"halt": None, "actor_exitcodes": [0, 0], "rate": float("nan")})
            )
            (attempt / "final.pth").write_bytes(b"final")
            rows = [
                {"wall": 40, "transitions": 100000, "updates": 20},
                {"wall": 70, "transitions": 850000, "updates": 50},
                {"wall": 100, "transitions": 1600000, "updates": 80},
            ]
            (attempt / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            return 0
        return 1  # hero_w128_a2 dies before creating its output directory

    monkeypatch.setattr(smoke, "bounded_command", fake_command)
    output = tmp_path / "output"
    assert smoke.run(tmp_path, output, "a" * 40, 150, 1050, tmp_path / "scratch") == 1
    result = json.loads((output / "colab_throughput_results.json").read_text())
    assert result["status"] == "incomplete"
    by_name = {row["name"]: row for row in result["configs"]}
    assert by_name["hero_w128_a2"]["status"] == "failed"
    assert by_name["hero_w128_a2"]["published"] is False
    assert by_name["hero_w128_a4"]["status"] == "completed"
    assert by_name["hero_w128_a4"]["published"] is True
    assert by_name["hero_w128_a4"]["summary"]["rate"] is None  # NaN sanitized for strict JSON
