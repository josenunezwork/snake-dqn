"""Asynchronous measured-host guard for the T34 retention runner.

``check`` is deliberately a cached, non-blocking per-tick read.  Sampling
uses host tools in a background thread; waiting is confined to
``between_rows`` so no gameplay row is paused or retried.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping


class HostContentionError(RuntimeError):
    """Measured host conditions make the registered result inconclusive."""


_REQUIRED = {
    "max_load_1m_excess",
    "other_process_cpu_percent",
    "other_process_cpu_sustained_seconds",
    "persistence_seconds",
    "check_interval_seconds",
    "own_cpu_worker_allowance",
    "known_heavy_process_patterns",
    "max_contended_fraction",
}
_POWER_REASONS = frozenset({"ac_power_lost", "lid_closed", "thermal_not_ok"})


def validate_policy(policy: Mapping[str, Any]) -> None:
    """Validate the prospective, measurable policy frozen in the profile."""
    if set(policy) != _REQUIRED:
        raise ValueError("load policy keys differ from the measured-host contract")
    numeric = (
        "max_load_1m_excess",
        "other_process_cpu_percent",
        "other_process_cpu_sustained_seconds",
        "persistence_seconds",
        "check_interval_seconds",
        "own_cpu_worker_allowance",
        "max_contended_fraction",
    )
    if any(
        type(policy[key]) not in (int, float) or not math.isfinite(policy[key]) or policy[key] <= 0
        for key in numeric
    ):
        raise ValueError("load policy numeric limits must be positive finite numbers")
    expected = {
        "max_load_1m_excess": 3.0,
        "other_process_cpu_percent": 50.0,
        "other_process_cpu_sustained_seconds": 60,
        "persistence_seconds": 120,
        "check_interval_seconds": 30,
        "own_cpu_worker_allowance": 2,
        "max_contended_fraction": 0.02,
    }
    if any(policy[key] != value for key, value in expected.items()):
        raise ValueError("load policy threshold differs from the registered contract")
    if policy["check_interval_seconds"] > 60:
        raise ValueError("host guard sampling interval must be at most 60 seconds")
    patterns = policy["known_heavy_process_patterns"]
    if (
        not isinstance(patterns, list)
        or not patterns
        or any(not isinstance(p, str) or not p for p in patterns)
    ):
        raise ValueError("known heavy process patterns must be a nonempty string list")


def _own_pids() -> set[int]:
    """Exclude only this run's descendants and its launcher ancestor chain."""
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True, timeout=2).splitlines()
    parents = {}
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and all(field.isdigit() for field in fields):
            parents[int(fields[0])] = int(fields[1])
    descendants = {os.getpid()}
    while True:
        expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    ancestors: set[int] = set()
    parent = parents.get(os.getpid())
    while parent and parent not in ancestors:
        ancestors.add(parent)
        parent = parents.get(parent)
    return descendants | ancestors


def _processes() -> list[dict[str, Any]]:
    rows = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,%cpu=,command="], text=True, timeout=2
    ).splitlines()
    records: list[dict[str, Any]] = []
    for row in rows:
        fields = row.strip().split(None, 3)
        if len(fields) != 4 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        try:
            cpu = float(fields[2])
        except ValueError:
            continue
        records.append(
            {
                "pid": int(fields[0]),
                "ppid": int(fields[1]),
                "cpu_percent": cpu,
                "command": fields[3],
            }
        )
    return records


def _power_thermal() -> tuple[bool, bool, bool]:
    if sys.platform != "darwin":
        return False, False, False
    from research.compute.thermal_guard import parse_pmset_therm, read_pmset_therm, thermal_reasons

    reading = read_pmset_therm()
    thermal_ok = bool(reading.get("available")) and not thermal_reasons(
        parse_pmset_therm(reading.get("text") or "")
    )
    try:
        power = subprocess.check_output(["pmset", "-g", "batt"], text=True, timeout=2)
        lid = subprocess.check_output(
            ["ioreg", "-r", "-k", "AppleClamshellState"], text=True, timeout=2
        )
    except (OSError, subprocess.SubprocessError):
        return False, False, thermal_ok
    return "AC Power" in power, '"AppleClamshellState" = No' in lid, thermal_ok


class HostGuard:
    """Sample asynchronously, account at ticks, and pause only between rows."""

    def __init__(
        self,
        policy: Mapping[str, Any],
        *,
        clock: Callable[[], float] = time.monotonic,
        sample: Callable[[], dict[str, Any]] | None = None,
        prior_elapsed_seconds: float = 0.0,
        prior_contended_seconds: float = 0.0,
    ) -> None:
        validate_policy(policy)
        if any(
            not math.isfinite(value) or value < 0
            for value in (prior_elapsed_seconds, prior_contended_seconds)
        ):
            raise ValueError("prior host accounting must be finite and nonnegative")
        if prior_contended_seconds > prior_elapsed_seconds:
            raise ValueError("prior contended time cannot exceed prior elapsed time")
        self.policy, self._clock, self._sample = dict(policy), clock, sample or self._measure
        self._prior_elapsed, self._prior_contended = prior_elapsed_seconds, prior_contended_seconds
        self._hot_since: dict[int, float] = {}
        self._lock = threading.Lock()
        self._updated = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] | None = None
        self._sample_error: BaseException | None = None
        self._started: float | None = None
        self._power_bad_since: float | None = None
        self._tick_count = self._contended_ticks = 0
        self._contended_seconds = 0.0
        self.records: list[dict[str, Any]] = []
        self.check_budget: Callable[[], None] = lambda: None

    def _measure(self) -> dict[str, Any]:
        ac_power, lid_open, thermal_ok = _power_thermal()
        load_1m, own, now = os.getloadavg()[0], _own_pids(), self._clock()
        other = [row for row in _processes() if row["pid"] not in own]
        hot: list[dict[str, Any]] = []
        active: set[int] = set()
        for row in other:
            if row["cpu_percent"] > self.policy["other_process_cpu_percent"]:
                active.add(row["pid"])
                since = self._hot_since.setdefault(row["pid"], now)
                if now - since > self.policy["other_process_cpu_sustained_seconds"]:
                    hot.append(row)
        self._hot_since = {pid: since for pid, since in self._hot_since.items() if pid in active}
        patterns = tuple(self.policy["known_heavy_process_patterns"])
        known = [row for row in hot if any(pattern in row["command"] for pattern in patterns)]
        excess, reasons = load_1m - float(self.policy["own_cpu_worker_allowance"]), []
        if not ac_power:
            reasons.append("ac_power_lost")
        if not lid_open:
            reasons.append("lid_closed")
        if not thermal_ok:
            reasons.append("thermal_not_ok")
        if excess > float(self.policy["max_load_1m_excess"]):
            reasons.append("load_1m_excess")
        if hot:
            reasons.append("other_cpu_sustained")
        if known:
            reasons.append("known_heavy_process")
        return {
            "timestamp_monotonic": now,
            "ac_power": ac_power,
            "lid_open": lid_open,
            "thermal_ok": thermal_ok,
            "load_1m": load_1m,
            "load_1m_excess": excess,
            "other_processes_over_cpu_limit": hot,
            "known_heavy_processes": known,
            "contended": bool(reasons),
            "reasons": reasons,
        }

    def _capture(self) -> None:
        """Sample off the gameplay path and atomically publish the result."""
        try:
            record = dict(self._sample())
            record["timestamp_monotonic"] = self._clock()
        except BaseException as exc:
            with self._updated:
                self._sample_error = exc
                self._updated.notify_all()
            return
        with self._updated:
            if self._started is not None and self._latest is not None:
                previous = float(self._latest["timestamp_monotonic"])
                elapsed = max(0.0, float(record["timestamp_monotonic"]) - previous)
                if self._latest["contended"]:
                    self._contended_seconds += elapsed
            self.records.append(record)
            self._latest, self._sample_error = record, None
            if _POWER_REASONS.intersection(record["reasons"]):
                if self._power_bad_since is None:
                    self._power_bad_since = float(record["timestamp_monotonic"])
            else:
                self._power_bad_since = None
            self._updated.notify_all()

    def _sampling_loop(self) -> None:
        while not self._stop.wait(float(self.policy["check_interval_seconds"])):
            self._capture()

    def start(self) -> None:
        """Perform admission before the first row and start the background sampler."""
        if self._thread is not None:
            return
        self.check_budget()
        self._started = self._clock()
        self._capture()
        with self._lock:
            if self._sample_error is not None:
                raise HostContentionError(f"host sampling failed: {self._sample_error}")
            if self._latest is None:
                raise HostContentionError("host sampling produced no record")
        self._thread = threading.Thread(
            target=self._sampling_loop, name="t34-host-guard", daemon=True
        )
        self._thread.start()

    def check(self) -> dict[str, Any]:
        """Read a cached sample and account for one tick without waiting or sampling."""
        self.check_budget()
        with self._lock:
            if self._started is None or self._latest is None:
                raise RuntimeError("HostGuard.start() is required before tick checks")
            if self._sample_error is not None:
                raise HostContentionError(f"host sampling failed: {self._sample_error}")
            self._tick_count += 1
            if self._latest["contended"]:
                self._contended_ticks += 1
            return dict(self._latest)

    def _summary_locked(self) -> dict[str, Any]:
        now = self._clock()
        run_elapsed = max(0.0, now - (self._started if self._started is not None else now))
        contended = self._contended_seconds
        if self._latest is not None and self._latest["contended"] and self._started is not None:
            contended += max(
                0.0, now - max(self._started, float(self._latest["timestamp_monotonic"]))
            )
        elapsed, contended = self._prior_elapsed + run_elapsed, self._prior_contended + contended
        fraction = contended / elapsed if elapsed else 0.0
        return {
            "sample_count": len(self.records),
            "tick_count": self._tick_count,
            "contended_tick_count": self._contended_ticks,
            "contended_seconds": contended,
            "elapsed_seconds": elapsed,
            "contended_fraction": fraction,
            "max_contended_fraction": self.policy["max_contended_fraction"],
            "status": (
                "INCONCLUSIVE" if fraction > self.policy["max_contended_fraction"] else "PASS"
            ),
        }

    def summary(self, *, validate: bool = True) -> dict[str, Any]:
        """Return diagnostics; the normal final call enforces the 2% budget."""
        with self._lock:
            result = self._summary_locked()
            if validate and result["status"] == "INCONCLUSIVE":
                error = HostContentionError(
                    "cumulative host contention exceeds registered fraction: "
                    f"{result['contended_fraction']:.6f}>{result['max_contended_fraction']:.6f}"
                )
                error.summary = result  # type: ignore[attr-defined]
                raise error
            return result

    def between_rows(self, on_pause: Callable[[], None] | None = None) -> dict[str, Any]:
        """Pause for power/lid/thermal only, after a complete row has finished."""
        paused = False
        while True:
            self.check_budget()
            with self._updated:
                if self._latest is None:
                    raise RuntimeError("HostGuard.start() is required before row boundaries")
                if self._sample_error is not None:
                    raise HostContentionError(f"host sampling failed: {self._sample_error}")
                record = dict(self._latest)
                power_pause = bool(_POWER_REASONS.intersection(record["reasons"]))
                if not power_pause:
                    result = self._summary_locked()
                    if result["status"] == "INCONCLUSIVE":
                        error = HostContentionError(
                            "cumulative host contention exceeds registered fraction: "
                            f"{result['contended_fraction']:.6f}>"
                            f"{result['max_contended_fraction']:.6f}"
                        )
                        error.summary = result  # type: ignore[attr-defined]
                        raise error
                    return record
                now = self._clock()
                pause_since = self._power_bad_since if self._power_bad_since is not None else now
                if now - pause_since >= float(self.policy["persistence_seconds"]):
                    raise HostContentionError(
                        "power/lid/thermal contention persisted: " + ",".join(record["reasons"])
                    )
                if not paused:
                    paused = True
                    if on_pause is not None:
                        on_pause()
                self._updated.wait(timeout=min(1.0, float(self.policy["check_interval_seconds"])))

    def close(self) -> None:
        """Stop the sampler; callers close it on every phase exit."""
        self._stop.set()
        with self._updated:
            self._updated.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=float(self.policy["check_interval_seconds"]) + 2.0)
            self._thread = None


def measured_snapshot(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Create one measured host snapshot for operator inspection, never an attestation."""
    guard = HostGuard(policy)
    try:
        guard.start()
        return guard.check()
    finally:
        guard.close()
