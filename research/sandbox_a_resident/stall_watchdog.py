"""Fail closed when a TRAIN-01 worker stops making meaningful progress.

The timer is intentionally implemented with :mod:`faulthandler`, rather than a
Python ``threading.Timer``.  Its watchdog thread can dump every Python thread
and terminate the process even while the training thread is blocked in a CUDA
call or another native extension that does not return to Python.
"""

from __future__ import annotations

import faulthandler
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping


class StallWatchdog:
    """One-shot no-progress watchdog with exported all-thread diagnostics.

    ``progress`` must be called only after an observable unit of work completes;
    supervisor polling is deliberately not a source of liveness.  On expiry,
    ``faulthandler`` appends all Python thread stacks to ``faulthandler-stacks.log``
    and calls ``_exit(1)``.  The pre-written JSON records the last known unit even
    when a native stall prevents any finally block from running.
    """

    def __init__(self, diagnostics_dir: Path, timeout_seconds: float) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("stall timeout must be finite and positive")
        self._diagnostics_dir = diagnostics_dir
        self._timeout_seconds = float(timeout_seconds)
        self._stack_file: Any | None = None
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
        self._stack_file = (self._diagnostics_dir / "faulthandler-stacks.log").open("a")
        self._armed = True
        self.progress(stage)

    def progress(self, stage: str, counters: Mapping[str, Any] | None = None) -> None:
        """Reset the deadline after a meaningful completed work unit."""
        if not self._armed or self._closed:
            raise RuntimeError("stall watchdog is not active")
        if not stage:
            raise ValueError("progress stage must be nonempty")
        if counters is not None and not isinstance(counters, Mapping):
            raise TypeError("progress counters must be a mapping")
        record = {
            "schema": "sandbox-a-stall-watchdog/v1",
            "pid": os.getpid(),
            "stage": stage,
            "timeout_seconds": self._timeout_seconds,
            "progress_monotonic_seconds": time.monotonic(),
            "deadline_monotonic_seconds": time.monotonic() + self._timeout_seconds,
            "counters": dict(counters or {}),
        }
        temporary = self._diagnostics_dir / "stall-watchdog.json.partial"
        temporary.write_text(json.dumps(record, sort_keys=True, default=str) + "\n")
        os.replace(temporary, self._diagnostics_dir / "stall-watchdog.json")
        faulthandler.cancel_dump_traceback_later()
        # exit=True is essential: a traceback alone cannot release a stuck
        # allocated runtime.  faulthandler's native watchdog is independent of
        # the Python evaluation loop that may itself be stuck.
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
        self._closed = True

    def __enter__(self) -> "StallWatchdog":
        self.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()
