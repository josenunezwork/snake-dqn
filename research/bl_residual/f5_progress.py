"""Outcome-blind stdout telemetry for bounded F5 qualification operations."""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

DEVICE_ROLES = {
    "cuda": "B8 parity and timing-only calibration",
    "cpu": "source evaluator and CPU parity reference",
}


class QualificationProgress:
    """Emit flushed, outcome-blind phase heartbeats from a daemon thread.

    The thread only observes monotonic time and immutable progress counters.  It
    never touches the evaluator, torch state, random generators, or measurements.
    """

    def __init__(self, *, total_fixtures: int = 12, interval_seconds: float = 60.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self._total = total_fixtures
        self._interval = interval_seconds
        self._started = time.monotonic()
        self._phase = "startup"
        self._completed = 0
        self._lock = threading.Lock()
        self._emission_lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def _payload(self, *, status: str) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "bl-f5-qualification-progress/v1",
                "kind": "OUTCOME_BLIND_OPERATIONAL_TELEMETRY",
                "status": status,
                "phase": self._phase,
                "fixtures_completed": self._completed,
                "fixtures_total": self._total,
                "elapsed_seconds": round(time.monotonic() - self._started, 3),
                "device_roles": DEVICE_ROLES,
            }

    def _write_locked(self, payload: dict[str, Any]) -> None:
        """Write a complete record while the caller holds the emission lock."""
        encoded = json.dumps(payload, sort_keys=True) + "\n"
        sys.stdout.write(encoded)
        sys.stdout.flush()

    def _emit(self, *, status: str) -> None:
        # Snapshot and publication share a lock so a queued heartbeat cannot
        # publish an older completed count after a newer phase record.
        with self._emission_lock:
            self._write_locked(self._payload(status=status))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._emit(status="STARTED")
        self._thread = threading.Thread(
            target=self._heartbeat, name="f5-qualification-progress", daemon=True
        )
        self._thread.start()

    def _heartbeat(self) -> None:
        while not self._stopped.wait(self._interval):
            self._emit(status="HEARTBEAT")

    def _stop_heartbeat(self) -> None:
        self._stopped.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self._interval + 1.0))

    def phase(self, name: str, *, completed: int | None = None) -> None:
        if not name or any(character.isspace() for character in name):
            raise ValueError("progress phase must be a nonempty token")
        with self._lock:
            self._phase = name
            if completed is not None:
                self._completed = completed
        self._emit(status="PHASE")

    def failure(self, error: BaseException) -> None:
        """Record a terminal failure class without emitting exception detail."""
        # Join before the terminal record so a concurrent heartbeat cannot be
        # emitted after failure has become visible to the supervisor.
        self._stop_heartbeat()
        with self._emission_lock:
            payload = self._payload(status="FAILED")
            payload["failure_class"] = type(error).__name__
            self._write_locked(payload)

    def close(self, *, success: bool) -> None:
        self._stop_heartbeat()
        self._emit(status="COMPLETE" if success else "STOPPED")
