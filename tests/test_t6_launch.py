"""Prepared T6 launch boundaries; these tests never start a training process."""

import json
import sys
from pathlib import Path

import pytest

from research.t6_restart_20261016 import launch, recipe


def test_check_prints_all_lane_commands_without_starting(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "RUNS", tmp_path)
    monkeypatch.setattr(launch, "verify_inputs", lambda: None)
    monkeypatch.setattr(
        launch,
        "command",
        lambda seed: [
            "python",
            "train",
            "--out",
            str(tmp_path / f"s{seed}"),
            "--seed",
            str(seed),
            "--device",
            "mps",
        ],
    )
    monkeypatch.setattr(
        launch.subprocess, "run", lambda *a, **k: pytest.fail("check launched a process")
    )
    monkeypatch.setattr(sys, "argv", ["launch", "--lane", "0", "--check"])
    assert launch.main() == 0
    printed = capsys.readouterr().out
    assert all(f"--seed {seed} --device mps" in printed for seed in (10, 12, 14))
    assert not list(tmp_path.iterdir())


def test_existing_output_and_kill_refuse_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "RUNS", tmp_path)
    monkeypatch.setattr(launch, "verify_inputs", lambda: None)
    monkeypatch.setattr(
        launch.subprocess, "run", lambda *a, **k: pytest.fail("refusal launched a process")
    )
    monkeypatch.setattr(sys, "argv", ["launch", "--lane", "0"])
    (tmp_path / "KILL.json").write_text("{}")
    with pytest.raises(SystemExit, match="KILL"):
        launch.main()
    (tmp_path / "KILL.json").unlink()
    monkeypatch.setattr(
        launch, "command", lambda seed: ["python", "--out", str(tmp_path / f"s{seed}")]
    )
    (tmp_path / "s10.log").write_text("prior")
    with pytest.raises(SystemExit, match="existing output/log"):
        launch.main()


def test_failed_child_records_kill_and_stops_lane(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(launch, "RUNS", tmp_path)
    monkeypatch.setattr(launch, "verify_inputs", lambda: None)
    monkeypatch.setattr(
        launch, "command", lambda seed: ["python", "--out", str(tmp_path / f"s{seed}")]
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["launch", "--lane", "0"])
    assert launch.main() == 9
    assert len(calls) == 1
    assert json.loads((tmp_path / "KILL.json").read_text())["decision"] == "KILL"
    assert not (tmp_path / "s12.log").exists()


def test_environment_removes_inherited_overrides(monkeypatch):
    monkeypatch.setenv("M3C_OPP_FORWARD", "merged-device")
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    env = recipe.environment()
    assert env["SNAKE_DQN_DEVICE"] == "mps"
    assert not any(k.startswith("M3C_") for k in env)
    smoke = recipe.command(10, smoke=True)
    full = recipe.command(10)
    changed = [(a, b) for a, b in zip(smoke, full) if a != b]
    assert len(changed) == 2
    assert ("200000", "8000000") in changed
    assert Path(changed[0][0]).name == "smoke_200k_s10"
