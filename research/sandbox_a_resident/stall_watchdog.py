"""Fail closed when a TRAIN-01 worker stops making meaningful progress.

The native faulthandler timer attempts diagnostics even when a native extension
holds the GIL. A separate stdlib process enforces a hard stop if that best-effort
dump itself hangs. Neither termination path requires the worker's Python loop.
"""

from __future__ import annotations

import faulthandler
import json
import math
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# A diagnostic grace, not a reset or extension of the completed-work deadline.
# The original native timer still fires at timeout_seconds. Only its dump gets
# this fixed bound before the independent process sends an uncatchable SIGKILL.
DUMP_GRACE_SECONDS = 0.25


class _ExitGuard:
    """Own a private deadline pipe and reap its independent hard-stop process."""

    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(Path(__file__).with_name("ct_watchdog_guard.py")),
                str(os.getpid()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            ready, _, _ = select.select([self.process.stdout], [], [], 5)
            if not ready or os.read(self.process.stdout.fileno(), 1) != b"R":
                raise RuntimeError("stall hard-exit guard did not start")
            os.set_blocking(self.process.stdin.fileno(), False)
        except BaseException:
            self.close()
            raise
        finally:
            self.process.stdout.close()

    def arm(self, deadline: float) -> None:
        """Publish one atomic absolute deadline or refuse to claim protection."""
        if self.process.poll() is not None:
            raise RuntimeError("stall hard-exit guard exited unexpectedly")
        command = f"{deadline:.9f}\n".encode("ascii")
        assert self.process.stdin is not None
        if os.write(self.process.stdin.fileno(), command) != len(command):
            raise RuntimeError("incomplete stall hard-exit deadline")

    def close(self) -> None:
        """EOF disarms; terminate only this owned helper if it fails to reap."""
        assert self.process.stdin is not None
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


class StallWatchdog:
    """One-shot no-progress watchdog with exported all-thread diagnostics.

    ``progress`` must be called only after an observable unit of work completes;
    supervisor polling is deliberately not a source of liveness.  On expiry,
    ``faulthandler`` attempts thread stacks and ``_exit(1)`` at the deadline.
    If that dump blocks, the guard kills the worker after at most 0.25 seconds
    of diagnostic grace. The pre-written JSON survives either hard-exit path.
    """

    def __init__(self, diagnostics_dir: Path, timeout_seconds: float) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("stall timeout must be finite and positive")
        self._diagnostics_dir = diagnostics_dir
        self._timeout_seconds = float(timeout_seconds)
        self._stack_file: Any | None = None
        self._exit_guard: _ExitGuard | None = None
        self._armed = False
        self._closed = False

    @property
    def diagnostics_dir(self) -> Path:
        """Directory containing the durable last-progress record and stack dump."""
        return self._diagnostics_dir

    def start(self, stage: str = "startup") -> None:
        """Create diagnostic files and arm the first bounded liveness interval."""
        if self._closed:
            raise RuntimeError("cannot restart a closed stall watchdog")
        if self._armed:
            raise RuntimeError("stall watchdog is already armed")
        self._diagnostics_dir.mkdir(parents=True, exist_ok=True)
        for name in ("stall-watchdog.json", "faulthandler-stacks.log"):
            for suffix in (".sha256", ".complete"):
                sidecar = self._diagnostics_dir / f"{name}{suffix}"
                if sidecar.exists() or sidecar.is_symlink():
                    raise RuntimeError(f"refusing to mutate sealed stall diagnostics: {sidecar}")
        self._stack_file = (self._diagnostics_dir / "faulthandler-stacks.log").open("a")
        try:
            self._exit_guard = _ExitGuard()
        except BaseException:
            self._stack_file.close()
            raise
        self._armed = True
        try:
            self.progress(stage)
        except BaseException:
            self.close()
            raise

    def progress(self, stage: str, counters: Mapping[str, Any] | None = None) -> None:
        """Reset the deadline after a meaningful completed work unit."""
        if not self._armed or self._closed:
            raise RuntimeError("stall watchdog is not active")
        if not stage:
            raise ValueError("progress stage must be nonempty")
        if counters is not None and not isinstance(counters, Mapping):
            raise TypeError("progress counters must be a mapping")
        now = time.monotonic()
        record = {
            "schema": "sandbox-a-stall-watchdog/v1",
            "pid": os.getpid(),
            "stage": stage,
            "timeout_seconds": self._timeout_seconds,
            "progress_monotonic_seconds": now,
            "deadline_monotonic_seconds": now + self._timeout_seconds,
            "counters": dict(counters or {}),
        }
        temporary = self._diagnostics_dir / "stall-watchdog.json.partial"
        temporary.write_text(
            json.dumps(record, sort_keys=True, default=str, allow_nan=False) + "\n"
        )
        os.replace(temporary, self._diagnostics_dir / "stall-watchdog.json")
        assert self._exit_guard is not None
        self._exit_guard.arm(now + self._timeout_seconds + DUMP_GRACE_SECONDS)
        faulthandler.cancel_dump_traceback_later()
        # Best effort only: CPython walks live frame pointers without the GIL.
        # The independent guard above must be armed BEFORE cancellation/dumping,
        # since cancel itself joins the native dump thread and can also hang.
        faulthandler.dump_traceback_later(
            self._timeout_seconds, repeat=False, file=self._stack_file, exit=True
        )

    def close(self) -> None:
        """Cancel the native timer and close its export file without firing it."""
        if self._closed:
            return
        if self._armed:
            faulthandler.cancel_dump_traceback_later()
            assert self._stack_file is not None
            self._stack_file.close()
            assert self._exit_guard is not None
            self._exit_guard.close()
        self._closed = True

    def __enter__(self) -> "StallWatchdog":
        self.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
