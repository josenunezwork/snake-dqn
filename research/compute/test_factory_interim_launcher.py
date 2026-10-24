"""Focused no-factory checks for the nice-interim launcher."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from research.compute import factory_interim_launcher as launcher
from research.mac_lease import held


@contextmanager
def _lease() -> object:
    yield object()


def _admit_any_test(*_args: object, **_kwargs: object) -> None:
    return None


@pytest.fixture(autouse=True)
def _isolated_interim_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "TEST_SLOT_PATH", tmp_path / "test-slot.lock")
    monkeypatch.setattr(launcher, "TIMING_ARMED_PATH", tmp_path / "timing-armed")
    monkeypatch.setattr(launcher, "held", lambda *_args, **_kwargs: _lease())
    monkeypatch.setattr(launcher, "_reject_test_command", _admit_any_test)


def test_serial_slot_is_nonblocking_and_resolves_current_default(tmp_path: Path) -> None:
    path = tmp_path / "shared.lock"
    with launcher.serial_test_slot(path):
        with pytest.raises(launcher.InterimLaunchError, match="busy"):
            with launcher.serial_test_slot(path):
                pass


def test_test_route_cleans_a_descendant_after_leader_success(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys; "
        "code='import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'; "
        "p=subprocess.Popen([sys.executable, '-c', code]); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid))"
    )
    assert (
        launcher.run_test(
            [sys.executable, "-c", script],
            review_manifest="unused",
            review_sha256="unused",
            deadline_seconds=5,
        )
        == 0
    )
    pid = int(child_pid.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_timeout_cleans_term_resistant_owned_process_group() -> None:
    script = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    with pytest.raises(launcher.InterimLaunchError, match="exceeded"):
        launcher.run_test(
            [sys.executable, "-c", script],
            review_manifest="unused",
            review_sha256="unused",
            deadline_seconds=4,
        )


def test_timing_arm_refuses_before_command_start() -> None:
    launcher.TIMING_ARMED_PATH.write_text("timing\n")
    with pytest.raises(launcher.InterimLaunchError, match="timing"):
        launcher.run_test(
            [sys.executable, "-c", "raise SystemExit(0)"],
            review_manifest="unused",
            review_sha256="unused",
            deadline_seconds=5,
        )


def test_test_deadline_cannot_exceed_interim_cap() -> None:
    with pytest.raises(launcher.InterimLaunchError, match="no greater"):
        launcher.run_test(
            [sys.executable, "-c", "raise SystemExit(0)"],
            review_manifest="unused",
            review_sha256="unused",
            deadline_seconds=121,
        )


def test_light_rejects_arbitrary_shell_and_python_routes() -> None:
    with pytest.raises(launcher.InterimLaunchError):
        launcher.run_light(["/bin/bash", "-c", "echo unsafe"])
    with pytest.raises(launcher.InterimLaunchError):
        launcher.run_light([sys.executable, "-m", "pytest"])


def test_factory_lease_refuses_all_tests_until_factory_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = tmp_path / "mac-lease.json"
    monkeypatch.setattr(launcher, "held", lambda *args, **kwargs: held(*args, **kwargs, path=lease))
    with held("synthetic-factory", "protected baseline/rearm/W12", seconds=20, path=lease):
        with pytest.raises(launcher.InterimLaunchError, match="lease is active"):
            launcher.run_test(
                [sys.executable, "-c", "raise AssertionError('must not start')"],
                review_manifest="unused",
                review_sha256="unused",
            )


@pytest.mark.parametrize("blocked", ["admission", "release"])
def test_complete_deadline_also_bounds_lease_mutation_locks(
    monkeypatch: pytest.MonkeyPatch, blocked: str
) -> None:
    @contextmanager
    def stalled_lease(*args, **kwargs):
        if blocked == "admission":
            time.sleep(10)
        yield
        if blocked == "release":
            time.sleep(10)

    monkeypatch.setattr(launcher, "held", stalled_lease)
    started = time.monotonic()
    with pytest.raises(launcher.InterimLaunchError, match="deadline"):
        launcher.run_test(
            [sys.executable, "-c", "pass"],
            review_manifest="unused",
            review_sha256="unused",
            deadline_seconds=0.7,
        )
    assert time.monotonic() - started < 1.5
    with launcher.serial_test_slot():
        pass
