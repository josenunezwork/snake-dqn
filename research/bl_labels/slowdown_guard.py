"""Outcome-blind, windowed host-slowdown guard for Mac factory attempts.

The guard consumes only coordinator monotonic timestamps and durable resume
metadata.  In particular it deliberately has no access to emitted life bytes,
labels, rescue decisions, or any outcome-bearing factory fields.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping

SCHEMA = "bl-labels-f0-slowdown-guard/v1"
FIRST_HOUR_SECONDS = 3600.0
MIN_BASELINE_LIVES = 60
SLOWDOWN_MULTIPLIER = 1.30
W12_W6_SANITY_MULTIPLIER = 1.911


def _number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


class SlowdownGuard:
    """Track per-stream/width baselines and one fresh-attempt median window.

    ``persisted`` is durable operational state from a previous attempt.  It
    intentionally does not restore the current-attempt window or in-flight
    admissions: every resume must collect fresh observations before arming.
    """

    def __init__(
        self,
        stream: str,
        width: int,
        started_at: float,
        *,
        persisted: Mapping[str, Any] | None = None,
        reference_baselines: Mapping[int | str, float] | None = None,
        d256_w6_baseline: float | None = None,
    ) -> None:
        if not isinstance(stream, str) or not stream:
            raise ValueError("stream must be nonempty")
        self.stream = stream
        self.width = self._width(width)
        self._clock = _number(started_at, "started_at")
        self._width_wall: dict[int, float] = {}
        self._baselines: dict[int, dict[str, Any]] = {}
        self._candidates: dict[int, list[float]] = {}
        self._reference = self._read_references(reference_baselines)
        self._d256_w6_baseline = (
            None
            if d256_w6_baseline is None
            else _number(d256_w6_baseline, "D256 W6 baseline", nonnegative=True)
        )
        self._inflight: dict[str, dict[str, Any]] = {}
        self._window: list[float] = []
        self._load(persisted)
        self._width_wall.setdefault(self.width, 0.0)
        self._candidates.setdefault(self.width, [])

    @staticmethod
    def _width(value: object) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError("width must be a positive integer")
        return value

    @staticmethod
    def _read_references(value: Mapping[int | str, float] | None) -> dict[int, float]:
        if value is None:
            return {}
        result: dict[int, float] = {}
        for raw_width, raw_baseline in value.items():
            try:
                width = int(raw_width)
            except (TypeError, ValueError) as error:
                raise ValueError("reference baseline width is invalid") from error
            result[SlowdownGuard._width(width)] = _number(
                raw_baseline, "reference baseline", nonnegative=True
            )
        return result

    def _load(self, persisted: Mapping[str, Any] | None) -> None:
        if persisted is None:
            return
        if not isinstance(persisted, Mapping):
            raise ValueError("slowdown guard state must be an object")
        if persisted.get("schema") != SCHEMA or persisted.get("stream") != self.stream:
            raise ValueError("slowdown guard state identity differs")
        for raw_width, raw_wall in persisted.get("width_wall_seconds", {}).items():
            self._width_wall[self._width(int(raw_width))] = _number(
                raw_wall, "width wall seconds", nonnegative=True
            )
        for raw_width, raw_value in persisted.get("baselines", {}).items():
            width = self._width(int(raw_width))
            if not isinstance(raw_value, Mapping) or raw_value.get("frozen") is not True:
                raise ValueError("slowdown baseline must be frozen")
            count = raw_value.get("sample_count")
            if type(count) is not int or count < MIN_BASELINE_LIVES:
                raise ValueError("slowdown baseline sample count is invalid")
            self._baselines[width] = {
                "median": _number(raw_value.get("median"), "baseline median", nonnegative=True),
                "sample_count": count,
                "frozen": True,
            }
        for raw_width, raw_values in persisted.get("candidates", {}).items():
            width = self._width(int(raw_width))
            if width in self._baselines:
                continue
            if not isinstance(raw_values, list):
                raise ValueError("slowdown candidates must be a list")
            self._candidates[width] = [
                _number(item, "candidate life duration", nonnegative=True) for item in raw_values
            ]

    def _advance(self, now: float) -> float:
        current = _number(now, "now")
        if current < self._clock:
            raise ValueError("monotonic timestamps moved backwards")
        # Count only time in which this stream owns at least one active parent.
        # A coordinator may interleave streams in one attempt.
        if self._inflight:
            self._width_wall[self.width] = (
                self._width_wall.get(self.width, 0.0) + current - self._clock
            )
        self._clock = current
        return current

    def _sanity_references(self, width: int) -> list[tuple[float, float]]:
        """Return ``(multiplier, reference)`` pairs all new baselines must clear."""
        checks: list[tuple[float, float]] = []
        same_width = self._reference.get(width)
        if same_width is not None:
            checks.append((SLOWDOWN_MULTIPLIER, same_width))
        if width == 12:
            local_w6 = self._baselines.get(6)
            w6 = float(local_w6["median"]) if local_w6 is not None else self._d256_w6_baseline
            # W12 never has an unreferenced baseline: the D256 W6 comparison
            # is an explicit fail-closed requirement of the ruling.
            if w6 is None:
                return []
            checks.append((W12_W6_SANITY_MULTIPLIER, w6))
        return checks

    def _freeze_if_ready(self, width: int) -> dict[str, Any] | None:
        if width in self._baselines:
            return None
        values = self._candidates.get(width, [])
        if (
            self._width_wall.get(width, 0.0) < FIRST_HOUR_SECONDS
            or len(values) < MIN_BASELINE_LIVES
        ):
            return None
        baseline = float(median(values))
        checks = self._sanity_references(width)
        if width == 12 and not checks:
            return {
                "event": "slowdown_baseline_sanity",
                "width": width,
                "baseline_median": baseline,
                "reference_median": None,
                "stop_reason": "slowdown baseline sanity",
            }
        for multiplier, reference in checks:
            if baseline <= multiplier * reference:
                continue
            return {
                "event": "slowdown_baseline_sanity",
                "width": width,
                "baseline_median": baseline,
                "reference_median": reference,
                "stop_reason": "slowdown baseline sanity",
            }
        self._baselines[width] = {
            "median": baseline,
            "sample_count": len(values),
            "frozen": True,
        }
        self._candidates.pop(width, None)
        return None

    def admit(self, allocation_id: str, has_prior: bool, now: float) -> None:
        """Start one life; ``has_prior`` records the durable replay exclusion."""
        if not isinstance(allocation_id, str) or not allocation_id:
            raise ValueError("allocation_id must be nonempty")
        started = self._advance(now)
        if allocation_id in self._inflight:
            raise ValueError("allocation already has an in-flight life")
        self._inflight[allocation_id] = {
            "started_at": started,
            "width": self.width,
            "replay_excluded": bool(has_prior),
            "width_changed": False,
            "stopped": False,
        }

    def boundary(
        self, allocation_id: str, now: float, stopped: bool = False
    ) -> dict[str, Any] | None:
        """Close one life and return a preemption event only when required."""
        current = self._advance(now)
        state = self._inflight.get(allocation_id)
        if state is None:
            raise ValueError("life boundary has no matching admission")
        duration = current - float(state["started_at"])
        eligible = not (
            bool(stopped)
            or bool(state.get("stopped", False))
            or state["replay_excluded"]
            or state["width_changed"]
        )
        # A boundary begins the allocation's next life.  A parent is admitted
        # once by the factory driver and may emit many life boundaries.
        self._inflight[allocation_id] = {
            "started_at": current,
            "width": self.width,
            "replay_excluded": False,
            "width_changed": False,
            # IPC can already contain more boundaries when cooperative stop
            # begins.  Keep a timer solely to drain them; none is eligible.
            "stopped": bool(stopped) or bool(state.get("stopped", False)),
        }
        if not eligible:
            return self._freeze_if_ready(self.width)
        width = int(state["width"])
        if width != self.width:
            return self._freeze_if_ready(self.width)
        self._window.append(duration)
        window_k = 4 * width
        if len(self._window) > window_k:
            del self._window[: len(self._window) - window_k]
        candidates = self._candidates.setdefault(width, [])
        if width not in self._baselines and (
            self._width_wall.get(width, 0.0) <= FIRST_HOUR_SECONDS
            or len(candidates) < MIN_BASELINE_LIVES
        ):
            candidates.append(duration)
        sanity = self._freeze_if_ready(width)
        if sanity is not None:
            return sanity
        baseline = self._baselines.get(width)
        if baseline is None:
            return None
        if len(self._window) < window_k:
            return None
        window_median = float(median(self._window))
        baseline_median = float(baseline["median"])
        if window_median > SLOWDOWN_MULTIPLIER * baseline_median:
            return {
                "event": "slowdown_guard",
                "statistic": "window_median",
                "window_k": window_k,
                "window_median": window_median,
                "baseline_median": baseline_median,
                "width": width,
                "stop_reason": "slowdown guard: window median > 1.30 x baseline",
            }
        return None

    def switch(self, width: int, now: float) -> None:
        """Change width at a boundary, excluding every life that straddles it."""
        self._advance(now)
        new_width = self._width(width)
        if new_width == self.width:
            return
        for state in self._inflight.values():
            state["width_changed"] = True
        self.width = new_width
        self._width_wall.setdefault(new_width, 0.0)
        self._candidates.setdefault(new_width, [])
        self._window.clear()

    def update_reference(self, width: int, median_value: float) -> None:
        """Publish the latest same-pin frozen baseline to this stream guard."""
        self._reference[self._width(width)] = _number(
            median_value, "reference baseline", nonnegative=True
        )

    def complete(self, allocation_id: str, now: float) -> None:
        """Stop charging a completed/preempted parent with no further boundary."""
        self._advance(now)
        self._inflight.pop(allocation_id, None)

    def tick(self, now: float) -> dict[str, Any] | None:
        """Advance width wall time when no boundary arrives at the hour edge."""
        self._advance(now)
        return self._freeze_if_ready(self.width)

    def snapshot(self, now: float) -> dict[str, Any]:
        """Return only durable per-stream baseline-collection state."""
        self._advance(now)
        return {
            "schema": SCHEMA,
            "stream": self.stream,
            "width_wall_seconds": {
                str(width): value for width, value in sorted(self._width_wall.items())
            },
            "baselines": {
                str(width): dict(value) for width, value in sorted(self._baselines.items())
            },
            "candidates": {
                str(width): list(values)
                for width, values in sorted(self._candidates.items())
                if width not in self._baselines
            },
        }
