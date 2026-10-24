"""Synthetic operational-window refusals; no live factory access or numerical work."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_ppo.exports import digest, export_json
from research.bl_round30 import factory_check as check


@pytest.fixture
def ordinary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    started = time.time() - 7200
    state, output = tmp_path / "state", tmp_path / "output"
    state.mkdir()
    output.mkdir()
    lease = tmp_path / "lease.json"
    lease.write_text(
        json.dumps(
            {
                "owner": "bl-labels-f0-mac-factory",
                "purpose": "F0 Mac hindsight label factory",
                "until_epoch": time.time() + 1000,
            }
        )
    )
    control = tmp_path / "control.json"
    control.write_text("synthetic operational control")
    start = state / "attempt-synthetic.start.json"
    export_json(
        start,
        {
            "schema": "bl-labels-f0-mac-attempt-start/v1",
            "workers": 6,
            "owner_w12_ok": False,
            "namespace": "synthetic-mac-d1024",
            "started_epoch": started,
            "source_commit": "a" * 40,
            "operational_source_commit": check.APPROVED_OPERATIONAL_PIN,
            "output": str(output),
            "control_sha256": digest(control),
        },
    )
    export_json(
        state / "slowdown-baseline-test-w6.json",
        {
            "schema": "bl-labels-f0-slowdown-baseline/v1",
            "namespace": "synthetic-mac-d1024",
            "source_commit": "a" * 40,
            "stream": "watch6",
            "width": 6,
            "created_epoch": started + 3601,
            "baseline": {"frozen": True, "sample_count": 138},
        },
    )
    journal = state / "attempt-synthetic.factory-process.jsonl"
    journal.write_text(
        json.dumps(
            {
                "attempt_id": "synthetic",
                "decision": "observation_complete",
                "interim_ruling_sha256": check.INTERIM_SHA,
                "sample_monotonic": time.monotonic(),
            }
        )
        + "\n"
    )
    argv = [
        "python",
        "-m",
        "research.bl_labels.mac_driver",
        "--state-dir",
        str(state),
        "--workers",
        "6",
        "--output",
        str(output),
        "--control",
        str(control),
    ]
    worker = SimpleNamespace(cmdline=lambda: ["python", "--multiprocessing-fork"], nice=lambda: 10)
    process = SimpleNamespace(
        pid=123, create_time=lambda: started, cmdline=lambda: argv, children=lambda: [worker]
    )
    monkeypatch.setattr(check.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(check, "default_lease_path", lambda: lease)
    monkeypatch.setattr(check.launcher, "TIMING_ARMED_PATH", tmp_path / "timing-arm")
    return start, process, worker, journal, lease


def test_fresh_w6_window_accepts_and_preserves_lease(ordinary: tuple) -> None:
    start, _, _, _, lease = ordinary
    before = lease.read_bytes()
    window = check.OrdinaryWindow(start, 123)
    window.check()
    assert window.evidence()["lease_preserved"]
    assert lease.read_bytes() == before


def test_rearming_or_foreign_baseline_refused(ordinary: tuple) -> None:
    start, process, _, _, _ = ordinary
    process.create_time = lambda: time.time()  # A replacement attempt cannot inherit proof.
    with pytest.raises(ValueError, match="first-hour/rearm/W12"):
        check.OrdinaryWindow(start, 123)


def test_runtime_timing_stale_journal_and_priority_refuse(ordinary: tuple) -> None:
    start, _, worker, journal, _ = ordinary
    window = check.OrdinaryWindow(start, 123)
    check.launcher.TIMING_ARMED_PATH.touch()
    with pytest.raises(ValueError, match="window changed"):
        window.check()
    check.launcher.TIMING_ARMED_PATH.unlink()
    worker.nice = lambda: 14
    with pytest.raises(ValueError, match="nice10"):
        window.check()
    worker.nice = lambda: 10
    observation = json.loads(journal.read_text())
    observation["sample_monotonic"] -= 30
    journal.write_text(json.dumps(observation) + "\n")
    with pytest.raises(ValueError, match="fresh successful"):
        window.check()


def test_terminal_and_pid_reuse_refuse(ordinary: tuple) -> None:
    start, process, _, _, _ = ordinary
    window = check.OrdinaryWindow(start, 123)
    process.create_time = lambda: time.time()
    with pytest.raises(ValueError, match="identity/window"):
        window.check()


@pytest.mark.parametrize("fault", ["priority", "window", "timeout"])
def test_run_retains_lock_through_cleanup_after_spawn(tmp_path, monkeypatch, fault):
    import os
    import signal
    from contextlib import contextmanager

    locked, cleaned = [], []

    @contextmanager
    def slot():
        locked.append(True)
        try:
            yield
        finally:
            locked.pop()

    checks = []

    def window_check():
        checks.append(True)
        if fault == "timeout":
            raise check.launcher.InterimLaunchError("test route exceeded deadline")
        raise ValueError("window changed")

    child = SimpleNamespace(pid=os.getpid(), poll=lambda: None)

    def popen(command, **kwargs):
        assert locked
        assert command[:3] == ["/usr/bin/nice", "-n", "0"]
        assert kwargs["env"]["OMP_THREAD_LIMIT"] == "1"
        assert kwargs["env"]["SNAKE_DQN_DEVICE"] == "cpu"
        kwargs["preexec_fn"]()  # Child resets inherited blocked signal mask before exec.
        assert signal.SIGTERM not in signal.pthread_sigmask(signal.SIG_BLOCK, set())
        return child

    def cleanup(owned, deadline):
        assert locked and owned is child
        cleaned.append(True)

    monkeypatch.setattr(check.launcher, "serial_test_slot", slot)
    monkeypatch.setattr(check.launcher, "_reject_test_command", lambda *a, **k: None)
    monkeypatch.setattr(check.launcher, "_cleanup", cleanup)
    monkeypatch.setattr(check.subprocess, "Popen", popen)
    monkeypatch.setattr(check, "OrdinaryWindow", lambda *a: SimpleNamespace(check=window_check))
    monkeypatch.setattr(check.os, "getpriority", lambda kind, pid: 14 if pid else 15)
    if fault != "priority":
        monkeypatch.setattr(check.os, "getpriority", lambda *a: 15)
    with pytest.raises((ValueError, check.launcher.InterimLaunchError)):
        check.run(["reviewed"], tmp_path / "manifest", "a" * 64, tmp_path / "start", 123)
    assert cleaned == [True] and not locked
    assert bool(checks) == (fault != "priority")


def test_busy_shared_slot_never_spawns(tmp_path, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def busy():
        raise check.launcher.InterimLaunchError("factory serial test slot is busy")
        yield

    monkeypatch.setattr(check.launcher, "serial_test_slot", busy)
    monkeypatch.setattr(check.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(check.launcher.InterimLaunchError, match="slot is busy"):
        check.run(["reviewed"], tmp_path / "manifest", "a" * 64, tmp_path / "start", 123)
