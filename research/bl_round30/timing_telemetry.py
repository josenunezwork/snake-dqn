"""Durable operational snapshots only; never accept scientific values or exception text."""

from __future__ import annotations

import builtins
import contextlib
import json
import math
import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Iterator

from research.bl_ppo.exports import export_json, limit_exports, verify_export

OBSERVABILITY_SHA = "bc5e471c28ab824b1ed06abc86ba72a5f7e5e2d018c3f687a167d41389a74888"
SCHEMA = "r30-timing-telemetry/v1"
MAX_EVENTS = 64
MAX_BYTES = 524_288  # All output bytes, including timing.json and sidecars.
MAX_EVENT_BYTES = 4096
ERROR_TYPES = frozenset(
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
)
CODES = {"NONE", "WORKER_DEADLINE", "CERTIFICATE_DEADLINE", "OTHER_EXCEPTION", "INTERRUPTED"}
_BOUNDARY_DEPTH = 0
_DEFERRED: BaseException | None = None


class WorkerDeadline(TimeoutError):
    """Fixed worker/session-relative cutoff; distinct from the certificate cap."""


class Interrupted(BaseException):
    """Supervisor termination; never expose signal frame or exception payload."""


def deliver_signal(error: BaseException) -> None:
    """Python handlers can run via an unmasked native thread; defer them too."""
    global _DEFERRED
    if _BOUNDARY_DEPTH:
        if _DEFERRED is None:
            _DEFERRED = error
    else:
        raise error


@contextlib.contextmanager
def atomic_boundary() -> Iterator[None]:
    """Defer deadline/termination delivery until publication and state agree."""
    global _BOUNDARY_DEPTH, _DEFERRED
    _BOUNDARY_DEPTH += 1
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        _BOUNDARY_DEPTH -= 1
        if not _BOUNDARY_DEPTH and _DEFERRED is not None:
            error, _DEFERRED = _DEFERRED, None
            raise error


def validate_event(value: dict) -> None:
    """Validate the exact operational-only vocabulary and scalar timing values."""
    from research.bl_round30.timing import PHASES

    if set(value) != {
        "schema",
        "observability_ruling_sha256",
        "sequence",
        "event",
        "phase",
        "status",
        "timestamp",
        "phase_started_at",
        "phase_elapsed_seconds",
        "elapsed_seconds",
        "completed_phase_seconds",
        "exception_type",
        "diagnostic_code",
    }:
        raise ValueError("telemetry keys differ")
    if (
        value["schema"] != SCHEMA
        or value["observability_ruling_sha256"] != OBSERVABILITY_SHA
        or type(value["sequence"]) is not int
        or not 0 <= value["sequence"] < MAX_EVENTS
        or value["phase"] not in (*PHASES, "preflight", "publication", "worker")
        or value["event"] not in {"start", "end", "failure", "complete"}
        or value["status"] not in {"running", "failed", "complete"}
        or value["diagnostic_code"] not in CODES
        or value["exception_type"] not in (*ERROR_TYPES, "OtherException", None)
    ):
        raise ValueError("telemetry identity differs")
    seconds = value["completed_phase_seconds"]
    if not isinstance(seconds, dict) or not set(seconds) <= set(PHASES):
        raise ValueError("telemetry phases differ")
    for number in (
        value["timestamp"],
        value["phase_started_at"],
        value["phase_elapsed_seconds"],
        value["elapsed_seconds"],
        *seconds.values(),
    ):
        if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
            raise ValueError("telemetry time differs")
    if value["event"] == "failure":
        if (
            value["status"] != "failed"
            or value["exception_type"] is None
            or value["diagnostic_code"] == "NONE"
            or value["phase"] in seconds
        ):
            raise ValueError("telemetry failure differs")
    elif value["exception_type"] is not None or value["diagnostic_code"] != "NONE":
        raise ValueError("telemetry error differs")
    if len((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()) > MAX_EVENT_BYTES:
        raise ValueError("telemetry event exceeds allowance")


class Telemetry:
    """One fsynced immutable export per boundary, with its completion marker last.

    Prior records remain exportable during native hangs, SIGKILL and timeouts.
    An unmatched start denotes an incomplete phase; its elapsed time is never a
    completed duration. No failure timing can become a timing.json report.
    """

    def __init__(self, output: Path) -> None:
        self.output = output
        (output / "telemetry").mkdir(exist_ok=False)
        for directory in (output, output.parent):
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        self.sequence = 0
        self.origin = time.monotonic()
        self.phase = "worker"
        self.started = self.origin
        self.wall_started = time.time()
        self.seconds: dict[str, float] = {}

    def emit(self, event: str, *, error: BaseException | None = None) -> None:
        with atomic_boundary():
            self._emit(event, error=error)

    def _emit(self, event: str, *, error: BaseException | None = None) -> None:
        code, exception_type = "NONE", None
        if error is not None:
            cls = type(error)
            # Class names can be user-controlled too. Only exact builtins escape.
            exception_type = cls.__name__ if cls in vars(builtins).values() else "OtherException"
            if isinstance(error, WorkerDeadline):
                code, exception_type = "WORKER_DEADLINE", "TimeoutError"
            elif isinstance(error, Interrupted):
                code, exception_type = "INTERRUPTED", "BaseException"
            elif isinstance(error, TimeoutError) and self.phase == "certificate":
                code = "CERTIFICATE_DEADLINE"
            else:
                code = "OTHER_EXCEPTION"
        value = {
            "schema": SCHEMA,
            "observability_ruling_sha256": OBSERVABILITY_SHA,
            "sequence": self.sequence,
            "event": event,
            "phase": self.phase,
            "status": (
                "failed"
                if error is not None
                else ("complete" if event == "complete" else "running")
            ),
            "timestamp": time.time(),
            "phase_started_at": self.wall_started,
            "phase_elapsed_seconds": time.monotonic() - self.started,
            "elapsed_seconds": time.monotonic() - self.origin,
            "completed_phase_seconds": dict(self.seconds),
            "exception_type": exception_type,
            "diagnostic_code": code,
        }
        validate_event(value)
        try:
            with limit_exports(self.output, total_bytes=MAX_BYTES, file_bytes=MAX_EVENT_BYTES):
                export_json(self.output / "telemetry" / f"{self.sequence:06d}.json", value)
        finally:
            # An I/O failure may leave a partial immutable slot. Never reuse it.
            self.sequence += 1

    def start(self, name: str) -> None:
        with atomic_boundary():
            self.phase = name
            self.started, self.wall_started = time.monotonic(), time.time()
            self.emit("start")

    def end(self) -> float:
        from research.bl_round30.timing import PHASES

        with atomic_boundary():
            duration = time.monotonic() - self.started
            if self.phase in PHASES:
                self.seconds[self.phase] = duration
            try:
                self.emit("end")
            except BaseException:
                # The operation returned, but its end publication failed. Do
                # not assert completion in a later failure snapshot/forecast.
                self.seconds.pop(self.phase, None)
                raise
            self.phase = "worker"
            self.started, self.wall_started = time.monotonic(), time.time()
            return duration


def validate_inventory(output: Path, *, complete: bool = False) -> list[dict[str, Any]]:
    """Read only bounded timing metadata, refusing scientific or partial exports."""
    directory = output / "telemetry"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("timing telemetry inventory differs")
    entries = sorted(directory.iterdir())
    if any(
        not re.fullmatch(
            r"0000[0-5][0-9]\.json(?:\.sha256|\.complete)?|"
            r"00006[0-3]\.json(?:\.sha256|\.complete)?",
            p.name,
        )
        for p in entries
    ):
        raise ValueError("timing telemetry inventory differs")
    payloads = [
        p.with_name(p.name.removesuffix(".complete"))
        for p in entries
        if p.name.endswith(".complete")
    ]
    if not payloads or any(p.is_symlink() or not p.is_file() for p in entries):
        raise ValueError("timing telemetry inventory differs")
    if complete:
        expected = {p.name + suffix for p in payloads for suffix in ("", ".sha256", ".complete")}
        if {p.name for p in entries} != expected:
            raise ValueError("timing telemetry inventory differs")
    records = []
    for sequence, path in enumerate(payloads):
        if (
            complete and path.name != f"{sequence:06d}.json"
        ) or path.stat().st_size > MAX_EVENT_BYTES:
            raise ValueError("timing telemetry sequence/size differs")
        verify_export(path)
        value = json.loads(path.read_text())
        validate_event(value)
        if path.name != f"{value['sequence']:06d}.json":
            raise ValueError("timing telemetry sequence differs")
        records.append(value)
    if complete and (
        records[-1]["event"] != "complete" or any(r["event"] == "failure" for r in records)
    ):
        raise ValueError("incomplete timing telemetry")
    return records
