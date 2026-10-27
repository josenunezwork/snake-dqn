"""Opt-in classified, interval CPU evidence; never imported by production serving."""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from research.bl_k1.m1_background import SYSTEM_EXECUTABLE_ROOTS

POLICY = "classified-v1"
SAMPLE_SECONDS = 1.0
MAX_SAMPLE_SECONDS = 5.0
FOREIGN_PERCENT = 20.0
BACKGROUND_CEILING_PERCENT = 150.0
BACKGROUND_CEILING_SECONDS = 60.0
BACKGROUND_KINDS = {"macos_service", "desktop_renderer"}


def process_kind(command: str) -> str:
    """Classify executable identity, never permissive substrings in argv.

    Workload names precede system paths: Apple's Python remains a workload.
    Unrecognized executables retain the strict foreign-load threshold.
    """
    path = Path(command)
    name = path.name.lower()
    if ".." in path.parts:
        return "unknown"
    if (
        re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name)
        or name in {"codex", "codex-exec", "agy", "claude", "node", "pytest", "uv"}
        or "/Projects/" in command
        or "/research/" in command
    ):
        return "agent_workload"
    if command.startswith(SYSTEM_EXECUTABLE_ROOTS):
        return "macos_service"
    # Electron/Chromium renderer executables inside installed application bundles.
    # CLI executables, shells, generic node and arbitrary /tmp helpers do not qualify.
    application = command.startswith("/Applications/") or bool(
        re.match(r"/Users/[^/]+/Applications/", command)
    )
    if (
        application
        and ".app/Contents/" in command
        and (name.endswith(" (renderer)") or name.endswith(" (gpu)"))
    ):
        return "desktop_renderer"
    return "unknown"


def owned_pids(rows: list[dict], owner_pid: int) -> set[int]:
    """Resolve sampled ancestry even after an owned short-lived helper exits."""
    allowed = {owner_pid}
    while True:
        expanded = allowed | {row["pid"] for row in rows if row["ppid"] in allowed}
        if expanded == allowed:
            return allowed
        allowed = expanded


def _cpu_seconds(value: str) -> float:
    """Parse macOS ps TIME (minutes:seconds.hundredths; optional hours/days)."""
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    fields = value.split(":")
    if len(fields) not in (2, 3):
        raise ValueError("invalid ps CPU time")
    total = 0.0
    for field in fields:
        total = total * 60 + float(field)
    total += days * 86400
    if not math.isfinite(total) or total < 0:
        raise ValueError("invalid ps CPU time")
    return total


def capture_cpu() -> dict[str, Any]:
    """Read cumulative process CPU, including protected macOS services.

    psutil CPU times are inaccessible for many system PIDs on macOS. ps TIME
    supplies user+system cumulative CPU at 10ms resolution without that blind spot.
    The capture bracket discloses that the inventory is not an atomic snapshot.
    """
    started = time.perf_counter_ns()
    result = subprocess.run(
        ["ps", "-ww", "-Ao", "pid,ppid,lstart,time,comm"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        env={**os.environ, "LC_ALL": "C"},
    )
    ended = time.perf_counter_ns()
    rows = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(None, 8)
        if len(parts) != 9:
            raise ValueError("unparseable cumulative CPU inventory")
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "birth": " ".join(parts[2:7]),
                "cpu_seconds": _cpu_seconds(parts[7]),
                "command": parts[8],
            }
        )
    if not rows:
        raise ValueError("empty cumulative CPU inventory")
    return {
        "time_ns": (started + ended) // 2,
        "capture_start_ns": started,
        "capture_end_ns": ended,
        "processes": rows,
    }


def cpu_interval(previous: dict, current: dict, owner_pid: int) -> dict[str, Any]:
    """CPU-time deltas keyed by PID and birth; churn is never a quiet observation."""
    start, end = previous["time_ns"], current["time_ns"]
    if end <= start:
        raise ValueError("CPU sample clock did not advance")
    seconds = (end - start) / 1e9
    before = {(row["pid"], row["birth"]): row for row in previous["processes"]}
    after = {(row["pid"], row["birth"]): row for row in current["processes"]}
    own_before = owned_pids(previous["processes"], owner_pid)
    own_after = owned_pids(current["processes"], owner_pid)
    coverage = seconds <= MAX_SAMPLE_SECONDS
    rows, churn = [], []
    totals = {"background": 0.0, "workload": 0.0, "unknown": 0.0}
    for key, row in after.items():
        if row["pid"] in own_after:
            continue
        old = before.get(key)
        if old is None or old["command"] != row["command"]:
            coverage = False
            churn.append({**row, "reason": "new_or_exec_process_no_delta"})
        delta = row["cpu_seconds"] if old is None else row["cpu_seconds"] - old["cpu_seconds"]
        if delta < 0:
            raise ValueError("process CPU counter decreased")
        kind = process_kind(row["command"])
        if old is not None and old["command"] != row["command"]:
            kinds = {kind, process_kind(old["command"])}
            if "agent_workload" in kinds:
                kind = "agent_workload"
            elif "unknown" in kinds:
                kind = "unknown"
        cpu = 100 * delta / seconds
        group = (
            "background"
            if kind in BACKGROUND_KINDS
            else ("workload" if kind == "agent_workload" else "unknown")
        )
        totals[group] += cpu
        if cpu > 0:
            rows.append({**row, "kind": kind, "cpu_percent": cpu})
    for key, row in before.items():
        if key not in after and row["pid"] not in own_before:
            coverage = False
            churn.append({**row, "reason": "exited_process_no_final_counter"})
    return {
        "start_ns": start,
        "end_ns": end,
        "capture_start_ns": current["capture_start_ns"],
        "capture_end_ns": current["capture_end_ns"],
        "coverage_complete": coverage,
        "process_churn": churn,
        "background_cpu_percent": totals["background"],
        "workload_cpu_percent": totals["workload"],
        "unknown_cpu_percent": totals["unknown"],
        "processes": rows,
    }


class LoadGuard:
    """Immediate foreign-work abort; aggregate background >150% for >60 seconds."""

    def __init__(self) -> None:
        self.high_since_ns: int | None = None
        self.last_end_ns: int | None = None

    def check(self, sample: dict) -> None:
        for row in sample["processes"]:
            if row["kind"] not in BACKGROUND_KINDS and row["cpu_percent"] >= FOREIGN_PERCENT:
                raise RuntimeError("foreign workload/unknown heavy load; attempt contaminated")
        contiguous = self.last_end_ns == sample["start_ns"]
        # No long-gap inference. Process churn limits low-window eligibility but
        # cannot hide measured high background CPU or reset its sustained guard.
        if (
            sample["background_cpu_percent"] > BACKGROUND_CEILING_PERCENT
            and sample["end_ns"] - sample["start_ns"] <= MAX_SAMPLE_SECONDS * 1e9
        ):
            if self.high_since_ns is None or not contiguous:
                self.high_since_ns = sample["start_ns"]
            if sample["end_ns"] - self.high_since_ns > BACKGROUND_CEILING_SECONDS * 1e9:
                raise RuntimeError("sustained background >150% for >60s; attempt contaminated")
        else:
            self.high_since_ns = None
        self.last_end_ns = sample["end_ns"]
