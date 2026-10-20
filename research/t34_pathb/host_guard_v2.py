"""Measured, continuous quiet-host guard for the T34 retention runner.

This is deliberately an operational guard rather than an operator attestation.
It samples the host before admission and at most thirty seconds apart while a
registered phase is active.  A non-quiet host pauses the phase; a continuous
non-quiet interval reaching the registered limit is INCONCLUSIVE.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


class HostContentionError(RuntimeError):
    """Measured host contention persisted through the registered pause limit."""


_REQUIRED = {
    "max_load_1m_excess",
    "other_process_cpu_percent",
    "other_process_cpu_sustained_seconds",
    "persistence_seconds",
    "check_interval_seconds",
    "own_cpu_worker_allowance",
    "known_heavy_process_patterns",
}


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
    }
    if any(policy[key] != value for key, value in expected.items()):
        raise ValueError("load policy threshold differs from the registered contract")
    if policy["check_interval_seconds"] > 60:
        raise ValueError("host guard sampling interval must be at most 60 seconds")
    patterns = policy["known_heavy_process_patterns"]
    if (
        not isinstance(patterns, list)
        or not patterns
        or any(not isinstance(pattern, str) or not pattern for pattern in patterns)
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
    ancestors = set()
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
    """Poll a registered host policy and pause until quiet or terminally contended."""

    def __init__(
        self,
        policy: Mapping[str, Any],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        sample: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        validate_policy(policy)
        self.policy = dict(policy)
        self._clock, self._sleeper = clock, sleeper
        self._sample = sample or self._measure
        self._next_check = 0.0
        self._bad_since: float | None = None
        self._hot_since: dict[int, float] = {}
        self.records: list[dict[str, Any]] = []
        self.check_budget: Callable[[], None] = lambda: None

    def _measure(self) -> dict[str, Any]:
        ac_power, lid_open, thermal_ok = _power_thermal()
        load_1m = os.getloadavg()[0]
        own = _own_pids()
        other = [row for row in _processes() if row["pid"] not in own]
        now = self._clock()
        hot: list[dict[str, Any]] = []
        active = set()
        for row in other:
            if row["cpu_percent"] > self.policy["other_process_cpu_percent"]:
                active.add(row["pid"])
                since = self._hot_since.setdefault(row["pid"], now)
                if now - since > self.policy["other_process_cpu_sustained_seconds"]:
                    hot.append(row)
        self._hot_since = {pid: since for pid, since in self._hot_since.items() if pid in active}
        patterns = tuple(self.policy["known_heavy_process_patterns"])
        known = [row for row in other if any(pattern in row["command"] for pattern in patterns)]
        excess = load_1m - float(self.policy["own_cpu_worker_allowance"])
        reasons = []
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

    def check(self, *, force: bool = False) -> dict[str, Any]:
        """Return the latest measured state and pause through a transient contention."""
        now = self._clock()
        if not force and self.records and now < self._next_check:
            return self.records[-1]
        while True:
            self.check_budget()
            record = self._sample()
            now = self._clock()
            record["timestamp_monotonic"] = now
            self.records.append(record)
            self._next_check = now + float(self.policy["check_interval_seconds"])
            if not record["contended"]:
                self._bad_since = None
                return record
            if self._bad_since is None:
                self._bad_since = now
            if now - self._bad_since >= float(self.policy["persistence_seconds"]):
                raise HostContentionError(
                    "measured host contention persisted: " + ",".join(record["reasons"])
                )
            self._sleeper(
                min(5.0, float(self.policy["persistence_seconds"]) - (now - self._bad_since))
            )
            force = True


def measured_snapshot(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Create one measured host snapshot for operator inspection, never an attestation."""
    guard = HostGuard(policy)
    return guard.check(force=True)
