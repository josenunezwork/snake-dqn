"""Slow offline admission must fit, while every caller deadline still fences it."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident.training_continuation import run_dispatch_dry_run
from tools.ops import dispatch


def test_slow_host_admission_fits_but_hung_and_ordinary_commands_refuse(monkeypatch) -> None:
    duration = [300.0]

    def run(args, *, timeout, **kwargs):
        if duration[0] > timeout:
            raise subprocess.TimeoutExpired(args, timeout)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(dispatch.subprocess, "run", run)
    dispatch.command(
        ["offline-admission"],
        timeout_seconds=dispatch.SANDBOX_ADMISSION_VALIDATE_TIMEOUT_SECONDS,
    )
    with pytest.raises(subprocess.TimeoutExpired):
        dispatch.command(["ordinary-host-command"])
    duration[0] = 1000.0
    with pytest.raises(subprocess.TimeoutExpired):
        dispatch.command(
            ["offline-admission"],
            timeout_seconds=dispatch.SANDBOX_ADMISSION_VALIDATE_TIMEOUT_SECONDS,
        )


def test_outer_validation_covers_both_passes_and_preserves_lock_deadline(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dispatch, "read_json", lambda _: {"sandbox_admission": {}})
    monkeypatch.setattr(dispatch, "sandbox_collector_directory", lambda _: None)
    monkeypatch.setattr(dispatch, "supervisor_launcher", lambda *_: Path("offline-launcher"))
    monkeypatch.setattr(dispatch.time, "monotonic", lambda: 10.0)
    limits = []

    def run(args, *, timeout, **kwargs):
        limits.append(timeout)
        assert kwargs["check"] is True
        if timeout < 650.0:
            raise subprocess.TimeoutExpired(args, timeout)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(dispatch.subprocess, "run", run)
    args = (tmp_path, tmp_path, tmp_path / "state", tmp_path / "manifest")
    dispatch.validate(*args, tmp_path / "first.log")
    assert limits[0] <= 900.0  # Finite outer bound, never an unlimited validation.
    with pytest.raises(subprocess.TimeoutExpired):
        dispatch.validate(*args, tmp_path / "second.log", deadline=110.0)
    assert limits[-1] == 100.0


def test_builder_waits_for_bounded_outer_validation(monkeypatch, tmp_path: Path) -> None:
    def run(args, *, timeout, **kwargs):
        assert kwargs["check"] is False
        assert "--dry-run" in args
        assert 850.0 <= timeout <= 960.0
        return subprocess.CompletedProcess(args, 0, stdout="PASS_DRY_RUN allocations=0", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert "allocations=0" in run_dispatch_dry_run(
        tmp_path, tmp_path, tmp_path, tmp_path / "state", "offline-job"
    )
