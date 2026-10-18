"""Opt-in learner-owned Q0b timing boundaries.

Off by default so ordinary M3-C training has byte-identical control flow.  When
``Q0Q1_BOUNDARY_OUT`` is set, the learner writes cumulative counters immediately
after it has consumed actor messages and finalized due updates.  Boundaries are
minimum-duration wall-clock boundaries: a slow learner can overrun a boundary but
can never shorten a registered window.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


class Q0bBoundaryRecorder:
    """Persist the 120s warm-up and two >=180s scored-window endpoints."""

    _POINTS = (("warmup", 120.0), ("window1", 180.0), ("window2", 180.0))

    def __init__(self) -> None:
        raw = os.environ.get("Q0Q1_BOUNDARY_OUT")
        self.path = Path(raw) if raw else None
        self.records: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self.path is not None

    @property
    def complete(self) -> bool:
        return len(self.records) == len(self._POINTS)

    def due(self, elapsed_seconds: float) -> bool:
        """Whether the next endpoint is due relative to the last actual endpoint."""
        if not self.enabled or self.complete:
            return False
        _, duration = self._POINTS[len(self.records)]
        previous = next(reversed(self.records.values()))["elapsed_seconds"] if self.records else 0.0
        return elapsed_seconds - previous >= duration

    def observe(self, elapsed_seconds: float, counters: Mapping[str, Any]) -> None:
        if not self.due(elapsed_seconds):
            return
        # At most one endpoint per observation: even a long pause cannot fabricate
        # two windows with the same counters or shorten the next scored window.
        phase, _ = self._POINTS[len(self.records)]
        self.records[phase] = {
            "phase": phase,
            "elapsed_seconds": float(elapsed_seconds),
            **counters,
        }
        self._write()

    def _write(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self.path.parent, delete=False) as stream:
            json.dump(self.records, stream, sort_keys=True)
            stream.write("\n")
            name = stream.name
        Path(name).replace(self.path)
