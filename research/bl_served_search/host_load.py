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


def capture_cpu() -> dict[str, Any]:
    """Read cumulative process CPU, including protected macOS services.

    psutil CPU times are inaccessible for many system PIDs on macOS. ps TIME
    supplies user+system cumulative CPU at 10ms resolution without that blind spot.
    The capture bracket discloses that the inventory is not an atomic snapshot.
    """
    started = time.perf_counter_ns()
    result = subprocess.run(
        ["ps", "-ww", "-Ao", "pid,ppid,uid,lstart,time,comm"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        env={**os.environ, "LC_ALL": "C"},
    )
    ended = time.perf_counter_ns()
    rows = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(None, 9)
        if len(parts) not in (9, 10):
            raise ValueError("unparseable cumulative CPU inventory")
        has_uid = len(parts) == 10
        offset = 1 if has_uid else 0
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "uid": int(parts[2]) if has_uid else None,
                "birth": " ".join(parts[2 + offset : 7 + offset]),
                "cpu_seconds": _cpu_seconds(parts[7 + offset]),
                "command": parts[8 + offset],
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


def cpu_interval(
    previous: dict, current: dict, owner_pid: int, *, desktop_classifier: Any | None = None
) -> dict[str, Any]:
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
        transition = old is not None and (old["command"] != row["command"] or uid_changed)
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
        if transition:
            # Snapshot the previous identity exactly once, before any current
            # admission, classification, or revocation can mutate its binding.
            prior_raw = desktop_classifier.classify(old) if desktop_classifier else {}
            prior_kind = _kind_from_evidence(old["command"], desktop_classifier, prior_raw)
            prior_evidence = classification_evidence(prior_kind, prior_raw)
            if desktop_classifier is not None:
                desktop_classifier.invalidate_transition(row)
        elif desktop_classifier is not None:
            # Dedicated off-tick collector: admit only new/unchanged rows.
            admit = getattr(desktop_classifier, "admit_process", None)
            if callable(admit):
                admit(row)
        evidence = desktop_classifier.classify(row) if desktop_classifier else {}
        kind = _kind_from_evidence(row["command"], desktop_classifier, evidence)
        if transition:
            kinds = {kind, prior_kind}
            if "agent_workload" in kinds:
                kind = "agent_workload"
            elif "unknown" in kinds:
                kind = "unknown"
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
        evidence = classification_evidence(kind, evidence)
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
                    "previous": dict(old),
                    "previous_classification": (
                        prior_evidence if prior_evidence is not None else {"kind": prior_kind}
                    ),
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
