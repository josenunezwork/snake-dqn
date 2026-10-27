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
DESKTOP_POLICY = "classified-desktop-v1"
SAMPLE_SECONDS = 1.0
MAX_SAMPLE_SECONDS = 5.0
FOREIGN_PERCENT = 20.0
BACKGROUND_CEILING_PERCENT = 150.0
BACKGROUND_CEILING_SECONDS = 60.0
BACKGROUND_KINDS = {"macos_service", "desktop_renderer"}
DESKTOP_BACKGROUND_KINDS = BACKGROUND_KINDS | {"desktop_app"}


def _workload_kind(command: str, desktop_classifier: Any | None = None) -> str | None:
    """Return workload classification before any desktop admission lookup."""
    path = Path(command)
    name = path.name.lower()
    if ".." in path.parts:
        return "unknown"
    verified_gui_candidate = (
        desktop_classifier is not None
        and name in {"claude", "codex"}
        and command.startswith(("/Applications/", "/Users/"))
        and ".app/Contents/MacOS/" in command
    )
    if (
        re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name)
        or (name in {"codex", "claude"} and not verified_gui_candidate)
        or name in {"codex-exec", "agy", "node", "pytest", "uv"}
        or "/Projects/" in command
        or "/research/" in command
    ):
        return "agent_workload"
    return None


def process_kind(
    command: str, desktop_classifier: Any | None = None, row: dict | None = None
) -> str:
    """Classify executable identity, never permissive substrings in argv.

    Workload names precede system paths: Apple's Python remains a workload.
    Unrecognized executables retain the strict foreign-load threshold.
    """
    evidence = (
        desktop_classifier.classify({**(row or {}), "command": command})
        if desktop_classifier is not None
        else {}
    )
    return _kind_from_evidence(command, desktop_classifier, evidence)


def _kind_from_evidence(command: str, desktop_classifier: Any | None, evidence: dict) -> str:
    """Apply precedence without reclassifying or mutating registry state."""
    if desktop_classifier is not None:
        return str(evidence.get("kind", "unknown"))
    path = Path(command)
    name = path.name.lower()
    workload = _workload_kind(command, desktop_classifier)
    if evidence.get("kind") == "agent_workload" or workload == "agent_workload":
        return "agent_workload"
    if evidence.get("identity_binding_changed") or workload == "unknown":
        return "unknown"
    if command.startswith(SYSTEM_EXECUTABLE_ROOTS):
        return "macos_service"
    if desktop_classifier is not None:
        return str(evidence.get("kind", "unknown"))
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


def classification_evidence(kind: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep final precedence classification while retaining violation flags."""
    if evidence.get("kind") == kind:
        return evidence
    normalized = {
        "kind": kind,
        "reason": "workload_or_system_precedence",
        "classifier_decision": evidence,
    }
    for key in ("identity_binding_changed", "known_workload"):
        if evidence.get(key):
            normalized[key] = evidence[key]
    if evidence.get("reason") == "known_workload":
        normalized["known_workload"] = True
    return normalized


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


# macOS ps uses header length as column width. MAXPATHLEN is 1024; a wider
# fixed executable column preserves spaces without splitting/guessing argv.
_COMMAND_WIDTH = 2048


def capture_argv(row: dict[str, Any]) -> dict[str, Any]:
    """Read exact argv, checking birth/executable on both sides of the read."""
    import psutil

    try:
        before = psutil.Process(row["pid"])
        birth = before.create_time()
        uid = before.uids().effective
        command = before.exe()
        argv = before.cmdline()
        after = psutil.Process(row["pid"])
        if (
            birth != after.create_time()
            or uid != row.get("uid")
            or after.uids().effective != uid
            or " ".join(time.ctime(birth).split()) != row["birth"]
            or command != row["command"]
            or after.exe() != command
            or after.cmdline() != argv
            or not argv
        ):
            return {"argv": [], "identity_observation_error": "argv_identity_verification_failed"}
        return {"argv": argv}
    except (psutil.Error, OSError, ValueError):
        return {"argv": [], "identity_observation_error": "argv_identity_unavailable"}


def capture_process_table(metric: str) -> list[dict[str, Any]]:
    """Capture executable and displayed argv together, with PID/birth and CPU."""
    if metric not in {"time", "pcpu"}:
        raise ValueError("unsupported process CPU metric")
    result = subprocess.run(
        [
            "ps",
            "-ww",
            "-Ao",
            f"pid,ppid,uid,lstart,{metric}",
            "-o",
            "comm=" + "C" * _COMMAND_WIDTH,
            "-o",
            "args",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        env={**os.environ, "LC_ALL": "C"},
    )
    rows = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(None, 9)
        if len(parts) != 10 or len(parts[9]) <= _COMMAND_WIDTH:
            raise ValueError("unparseable process executable/argv inventory")
        command = parts[9][:_COMMAND_WIDTH].rstrip()
        argv = parts[9][_COMMAND_WIDTH:].strip()
        if not command or not argv:
            raise ValueError("missing process executable/argv evidence")
        row = {
            "pid": int(parts[0]),
            "ppid": int(parts[1]),
            "uid": int(parts[2]),
            "birth": " ".join(parts[3:8]),
            "command": command,
            "argv_display": argv,
            ("cpu_seconds" if metric == "time" else "cpu_percent"): (
                _cpu_seconds(parts[8]) if metric == "time" else float(parts[8])
            ),
        }
        row.update(capture_argv(row))
        rows.append(row)
    if not rows:
        raise ValueError("empty process inventory")
    return rows


def capture_cpu() -> dict[str, Any]:
    """Read cumulative macOS CPU and argv in one bracketed process inventory."""
    started = time.perf_counter_ns()
    rows = capture_process_table("time")
    ended = time.perf_counter_ns()
    return {
        "time_ns": (started + ended) // 2,
        "capture_start_ns": started,
        "capture_end_ns": ended,
        "processes": rows,
    }


def freeze_cpu_snapshot(snapshot: dict, owner_pid: int, desktop_classifier: Any) -> None:
    """Attach decisions at observation time; previous rows are never reclassified."""
    own = owned_pids(snapshot["processes"], owner_pid)
    for row in snapshot["processes"]:
        if row["pid"] not in own and "classification" not in row:
            row["observation_ns"] = snapshot["time_ns"]
            observe = getattr(desktop_classifier, "admit_process", desktop_classifier.classify)
            row["classification"] = observe(row)


def interval_classification(previous: dict, current: dict) -> dict:
    """Conservatively attribute CPU across a transition from two frozen decisions."""
    kinds = {previous["kind"], current["kind"]}
    kind = (
        "agent_workload"
        if "agent_workload" in kinds
        else "unknown" if "unknown" in kinds else current["kind"]
    )
    return classification_evidence(kind, current)


def cpu_interval(
    previous: dict, current: dict, owner_pid: int, *, desktop_classifier: Any | None = None
) -> dict[str, Any]:
    """CPU-time deltas keyed by PID and birth; churn is never a quiet observation."""
    start, end = previous["time_ns"], current["time_ns"]
    if end <= start:
        raise ValueError("CPU sample clock did not advance")
    seconds = (end - start) / 1e9
    if desktop_classifier is not None:
        for snapshot in (previous, current):
            own = owned_pids(snapshot["processes"], owner_pid)
            if any(
                row["pid"] not in own and "classification" not in row
                for row in snapshot["processes"]
            ):
                raise ValueError("desktop CPU snapshot was not classified at capture time")
    before = {(row["pid"], row["birth"]): row for row in previous["processes"]}
    after = {(row["pid"], row["birth"]): row for row in current["processes"]}
    own_before = owned_pids(previous["processes"], owner_pid)
    own_after = owned_pids(current["processes"], owner_pid)
    coverage = seconds <= MAX_SAMPLE_SECONDS
    rows, churn = [], []
    totals = {
        "background": 0.0,
        "workload": 0.0,
        "unknown": 0.0,
        "macos_service": 0.0,
        "desktop_renderer": 0.0,
        "desktop_app": 0.0,
    }
    for key, row in after.items():
        if row["pid"] in own_after:
            continue
        old = before.get(key)
        uid_changed = (
            desktop_classifier is not None and old is not None and old.get("uid") != row.get("uid")
        )
        transition = old is not None and (
            old["command"] != row["command"]
            or uid_changed
            or (desktop_classifier is not None and old.get("argv") != row.get("argv"))
        )
        if old is None or transition:
            coverage = False
            churn.append(
                {
                    **row,
                    "reason": (
                        "uid_changed_no_delta" if uid_changed else "new_or_exec_process_no_delta"
                    ),
                }
            )
        delta = row["cpu_seconds"] if old is None else row["cpu_seconds"] - old["cpu_seconds"]
        if delta < 0:
            raise ValueError("process CPU counter decreased")
        prior_evidence = None
        if desktop_classifier is not None:
            evidence = row["classification"]
            if transition:
                prior_evidence = old["classification"]
                final_evidence = interval_classification(prior_evidence, evidence)
            else:
                final_evidence = evidence
            kind = final_evidence["kind"]
        else:
            evidence = {}
            kind = _kind_from_evidence(row["command"], None, evidence)
            if transition:
                prior_kind = _kind_from_evidence(old["command"], None, {})
                prior_evidence = classification_evidence(prior_kind, {})
                final_evidence = interval_classification(prior_evidence, {"kind": kind})
                kind = final_evidence["kind"]
        cpu = 100 * delta / seconds
        background_kinds = (
            DESKTOP_BACKGROUND_KINDS if desktop_classifier is not None else BACKGROUND_KINDS
        )
        group = (
            "background"
            if kind in background_kinds
            else ("workload" if kind == "agent_workload" else "unknown")
        )
        totals[group] += cpu
        if kind in {"macos_service", "desktop_renderer", "desktop_app"}:
            totals[kind] += cpu
        current_evidence = evidence
        evidence = (
            final_evidence
            if desktop_classifier is not None
            else classification_evidence(kind, evidence)
        )
        # Preserve every measured CPU contribution and every admitted UI/violation
        # binding. Idle unrelated processes remain in raw point inventories and
        # churn, rather than repeating thousands of zero rows for up to 7200s.
        if cpu > 0 or (
            desktop_classifier is not None
            and (
                kind in {"desktop_app", "desktop_renderer"}
                or evidence.get("identity_binding_changed")
                or evidence.get("reason") == "known_workload"
            )
        ):
            exported = {**row, "kind": kind, "cpu_percent": cpu, "classification": evidence}
            if transition:
                # Replay needs both identities: this interval's conservative
                # classification is a precedence over the before/after rows,
                # not a claim that the after identity supplied all its CPU.
                exported["interval_transition"] = {
                    "reason": "uid_changed_no_delta" if uid_changed else "exec_changed_no_delta",
                    "previous": {
                        key: value for key, value in old.items() if key != "classification"
                    },
                    "current_classification": current_evidence,
                    "previous_classification": (prior_evidence),
                    "conservative_kind": kind,
                }
            rows.append(exported)
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
        "macos_service_cpu_percent": totals["macos_service"],
        "desktop_renderer_cpu_percent": totals["desktop_renderer"],
        "desktop_app_cpu_percent": totals["desktop_app"],
        "processes": rows,
    }


class LoadGuard:
    """Immediate foreign-work abort; aggregate background >150% for >60 seconds."""

    def __init__(self, *, desktop_policy: bool = False) -> None:
        self.background_kinds = DESKTOP_BACKGROUND_KINDS if desktop_policy else BACKGROUND_KINDS
        self.high_since_ns: int | None = None
        self.last_end_ns: int | None = None

    def check(self, sample: dict) -> None:
        for row in sample["processes"]:
            if (
                row.get("classification", {}).get("known_workload")
                or row.get("classification", {}).get("reason") == "known_workload"
            ):
                raise RuntimeError("prohibited workload under desktop role; attempt contaminated")
            if row.get("classification", {}).get("identity_binding_changed"):
                raise RuntimeError("desktop identity changed; attempt contaminated")
            if row["kind"] not in self.background_kinds and row["cpu_percent"] >= FOREIGN_PERCENT:
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
