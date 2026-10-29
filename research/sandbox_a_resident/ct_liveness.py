"""Stdlib-only C/T status heartbeat, independent of completed-work deadlines.

This module never imports torch, draws random numbers, or touches a work watchdog.
The entrypoint starts it before importing any numerical producer. Phase changes
and periodic status lines are diagnostics only, never evidence of completed work.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from research.sandbox_a_resident import ct_json as json

_OUTPUT_LOCK = threading.RLock()
_CURRENT: StatusHeartbeat | None = None


def print_status(line: str, output: Path) -> None:
    """Serialize complete history/stdout lines from main and heartbeat threads."""
    with _OUTPUT_LOCK:
        with (output / "progress-history.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)


class StatusHeartbeat:
    """Emit every 15 seconds even when the producer has completed no work."""

    def __init__(self, output: Path, *, interval: float = 15.0) -> None:
        if not 0 < interval <= 30:
            raise ValueError("status heartbeat interval must be positive and <=30 seconds")
        self.output = output
        self.interval = interval
        self.started = time.monotonic()
        self.phase = "imports"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="ct-status", daemon=True)
        self.emitted: list[float] = []

    def _emit(self) -> None:
        with _OUTPUT_LOCK:
            now = time.monotonic()
            row = {
                "schema": "t34-cf-heartbeat/v1",
                "study": "t34-cf-1028",
                "phase": self.phase,
                "elapsed_sec": round(now - self.started, 6),
            }
            print_status("CT_HEARTBEAT " + json.dumps(row, sort_keys=True), self.output)
            self.emitted.append(now)

    def start(self) -> None:
        self._emit()
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._emit()

    def set_phase(self, phase: str) -> None:
        with _OUTPUT_LOCK:
            self.phase = phase
            self._emit()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()

    def cadence(self) -> dict[str, Any]:
        """Measure actual heartbeat gaps from startup through this observation."""
        with _OUTPUT_LOCK:
            points = [self.started, *self.emitted, time.monotonic()]
            return {
                "line_count": len(self.emitted),
                "max_gap_seconds": max(b - a for a, b in zip(points, points[1:])),
                "limit_seconds": 30,
            }


def start_status(output: Path) -> StatusHeartbeat:
    """Create a fresh diagnostic directory and start before numerical imports."""
    global _CURRENT
    if _CURRENT is not None:
        raise RuntimeError("C/T status heartbeat already active")
    output.mkdir(parents=True, exist_ok=False)
    _CURRENT = StatusHeartbeat(output)
    _CURRENT.start()
    return _CURRENT


def owns_output(output: Path) -> bool:
    """Whether the entrypoint created this directory in this process."""
    return _CURRENT is not None and _CURRENT.output == output


def set_phase(phase: str) -> None:
    """Publish diagnostic phase boundaries; deliberately no work-watchdog call."""
    if _CURRENT is not None:
        _CURRENT.set_phase(phase)


def finish_status() -> dict[str, Any] | None:
    """Stop and join before publishing immutable history sidecars."""
    global _CURRENT
    if _CURRENT is None:
        return None
    current, _CURRENT = _CURRENT, None
    current.close()
    return current.cadence()
