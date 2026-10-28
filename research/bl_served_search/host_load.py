"""Opt-in classified, interval CPU evidence; never imported by production serving."""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import platform
import re
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research.bl_served_search.desktop_apps import (
    _AGENT_BUNDLE_NON_UI_PATH_RE,
    _is_verified_gui_command,
    is_protected_macos_service_path,
)

POLICY = "classified-v1"
DESKTOP_POLICY = "classified-desktop-v1"
SAMPLE_SECONDS = 1.0
MAX_SAMPLE_SECONDS = 5.0
FOREIGN_PERCENT = 20.0
BACKGROUND_CEILING_PERCENT = 150.0
BACKGROUND_CEILING_SECONDS = 60.0
BACKGROUND_KINDS = {"macos_service", "desktop_renderer"}
DESKTOP_BACKGROUND_KINDS = BACKGROUND_KINDS | {"desktop_app"}
_COMMAND_WIDTH = 2048


def _argv_changed(old: dict[str, Any], row: dict[str, Any]) -> bool:
    """Return True when argument identity or its capture status changed across an interval."""
    from research.bl_served_search.desktop_apps import (
        _extract_argv_evidence,
        _extract_display_argv,
    )

    old_argv, old_status, _ = _extract_argv_evidence(old)
    row_argv, row_status, _ = _extract_argv_evidence(row)
    old_display = _extract_display_argv(old, old_argv, old_status)
    row_display = _extract_display_argv(row, row_argv, row_status)
    if old_display != row_display:
        return True
    if (old_argv, old_status) == (row_argv, row_status):
        return False
    cmd = row.get("command")
    if (
        isinstance(cmd, str)
        and cmd == old.get("command")
        and row.get("uid") == old.get("uid")
        and is_protected_macos_service_path(cmd)
        and _workload_kind(cmd) is None
        and (old_status != "ok" or row_status != "ok")
    ):
        return False
    return True


_EXPLICIT_WORKLOAD_MARKERS = (
    "/Projects/",
    "/research/",
    "/codex-cli/",
    "/claude-code/",
    "/cua_node/",
    "/.codex/",
    "/artifact-sessions/",
)
_NON_TOP_LEVEL_GUI_MARKERS = (
    "/Contents/Resources/",
    "/Contents/Helpers/",
    "/Contents/Frameworks/",
    "/Application Support/",
)


def _classifier_verified_map(desktop_classifier: Any | None) -> Mapping[str, Any] | None:
    if desktop_classifier is None:
        return None
    verified = getattr(desktop_classifier, "_verified", None)
    if isinstance(verified, Mapping):
        return verified
    if isinstance(desktop_classifier, Mapping):
        exported = desktop_classifier.get("verified")
        if isinstance(exported, Mapping):
            return exported
    return None


def _workload_kind(command: str, desktop_classifier: Any | None = None) -> str | None:
    """Return workload classification before any desktop admission lookup."""
    path = Path(command)
    name = path.name.lower()
    if ".." in path.parts:
        return "unknown"
    explicit_workload_tree = any(marker in command for marker in _EXPLICIT_WORKLOAD_MARKERS)
    non_top_level_gui = any(marker in command for marker in _NON_TOP_LEVEL_GUI_MARKERS)
    verified_map = _classifier_verified_map(desktop_classifier)
    gui_location_ok = (
        _is_verified_gui_command(command, verified_map)
        if verified_map is not None
        else command.startswith(("/Applications/", "/Users/"))
    )
    verified_gui_candidate = (
        desktop_classifier is not None
        and not is_protected_macos_service_path(command)
        and not explicit_workload_tree
        and not non_top_level_gui
        and name in {"claude", "codex"}
        and gui_location_ok
        and ".app/Contents/MacOS/" in command
    )
    if (
        re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name)
        or (name in {"codex", "claude"} and not verified_gui_candidate)
        or name in {"codex-exec", "agy", "node", "pytest", "uv"}
        or explicit_workload_tree
    ):
        return "agent_workload"
    if is_protected_macos_service_path(command):
        return None
    if _AGENT_BUNDLE_NON_UI_PATH_RE.search(command):
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
    if is_protected_macos_service_path(command):
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


_LIBPROC: Any | None = None
_LIBPROC_LOADED = False


def _proc_pidpath(pid: int) -> str | None:
    """Resolve canonical kernel executable path via libproc.proc_pidpath on Darwin."""
    global _LIBPROC, _LIBPROC_LOADED
    if platform.system() != "Darwin" or pid <= 0:
        return None
    if not _LIBPROC_LOADED:
        _LIBPROC_LOADED = True
        try:
            _LIBPROC = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        except OSError:
            _LIBPROC = None
    if _LIBPROC is None:
        return None
    buf = ctypes.create_string_buffer(4096)
    try:
        nbytes = int(_LIBPROC.proc_pidpath(ctypes.c_int(pid), buf, ctypes.c_uint32(4096)))
    except Exception:
        return None
    if nbytes <= 0:
        return None
    try:
        return buf.raw[:nbytes].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _resolve_protected_service_path(row: dict[str, Any]) -> None:
    """Resolve bare ps comm names for SIP/root protected macOS services with birth bracketing."""
    command = row.get("command")
    if not isinstance(command, str) or not command or command.startswith("/"):
        return
    import psutil

    try:
        pid = int(row["pid"])
        before = psutil.Process(pid)
        birth = before.create_time()
        uid = before.uids().effective
        pidpath = _proc_pidpath(pid)
        after = psutil.Process(pid)
        if (
            not pidpath
            or birth != after.create_time()
            or (row.get("uid") is not None and uid != row.get("uid"))
            or after.uids().effective != uid
            or " ".join(time.ctime(birth).split()) != row.get("birth")
            or _proc_pidpath(pid) != pidpath
        ):
            return
    except (psutil.Error, OSError, ValueError, KeyError):
        return
    if (
        is_protected_macos_service_path(pidpath)
        and Path(pidpath).name == command
        and _workload_kind(pidpath) is None
    ):
        row["command"] = pidpath


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
            or (row.get("uid") is not None and uid != row.get("uid"))
            or after.uids().effective != uid
            or " ".join(time.ctime(birth).split()) != row["birth"]
            or command != row["command"]
            or after.exe() != command
            or after.cmdline() != argv
            or not argv
        ):
            return {"argv": [], "identity_observation_error": "argv_identity_verification_failed"}
        return {"argv": [str(x) for x in argv]}
    except (psutil.Error, OSError, ValueError, KeyError):
        return {"argv": [], "identity_observation_error": "argv_identity_unavailable"}


def capture_process_table(metric: str) -> tuple[int, int, list[dict[str, Any]]]:
    """Capture executable and displayed argv together, with PID/birth and CPU."""
    if metric not in {"time", "pcpu"}:
        raise ValueError("unsupported process CPU metric")
    started = time.perf_counter_ns()
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
    ended = time.perf_counter_ns()
    midpoint = (started + ended) // 2
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(None, 9)
        if len(parts) not in (9, 10):
            raise ValueError(f"unparseable process inventory ({metric})")
        has_uid = len(parts) == 10
        offset = 1 if has_uid else 0
        tail = parts[8 + offset]
        if len(tail) <= _COMMAND_WIDTH:
            raise ValueError(f"unparseable process command column ({metric})")
        command = tail[:_COMMAND_WIDTH].rstrip()
        argv_display = tail[_COMMAND_WIDTH:].strip()
        if not command or not argv_display:
            raise ValueError(f"empty process command ({metric})")
        row: dict[str, Any] = {
            "pid": int(parts[0]),
            "ppid": int(parts[1]),
            "uid": int(parts[2]) if has_uid else None,
            "birth": " ".join(parts[2 + offset : 7 + offset]),
            "metric": parts[7 + offset],
            "command": command,
            "argv_display": argv_display,
            "observation_ns": midpoint,
        }
        _resolve_protected_service_path(row)
        row.update(capture_argv(row))
        rows.append(row)
    if not rows:
        raise ValueError(f"empty process inventory ({metric})")
    return started, ended, rows


def capture_cpu() -> dict[str, Any]:
    """Read cumulative process CPU, including protected macOS services.

    psutil CPU times are inaccessible for many system PIDs on macOS. ps TIME
    supplies user+system cumulative CPU at 10ms resolution without that blind spot.
    The capture bracket discloses that the inventory is not an atomic snapshot.
    """
    started, ended, raw_rows = capture_process_table("time")
    rows = []
    for raw in raw_rows:
        metric = raw.pop("metric")
        raw["cpu_seconds"] = _cpu_seconds(metric)
        rows.append(raw)
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
    use_step_row = desktop_classifier is not None and hasattr(desktop_classifier, "step_row")
    prior_snapshot = (
        previous.get("_process_history")
        if use_step_row and isinstance(previous.get("_process_history"), dict)
        else None
    )
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
        argv_changed = (
            desktop_classifier is not None and old is not None and _argv_changed(old, row)
        )
        transition = old is not None and (
            old["command"] != row["command"] or uid_changed or argv_changed
        )
        if old is None or transition:
            coverage = False
            churn.append(
                {
                    **row,
                    "reason": (
                        "uid_changed_no_delta"
                        if uid_changed
                        else (
                            "argv_changed_no_delta"
                            if argv_changed and old is not None and old["command"] == row["command"]
                            else "new_or_exec_process_no_delta"
                        )
                    ),
                }
            )
        delta = row["cpu_seconds"] if old is None else row["cpu_seconds"] - old["cpu_seconds"]
        if delta < 0:
            raise ValueError("process CPU counter decreased")
        prior_evidence = None
        prior_history_before = None
        prior_history_after = None
        row_history_before = None
        row_history_after = None
        row_admit = False
        if use_step_row:
            from research.bl_served_search.desktop_apps import (
                _extract_argv_evidence,
                _extract_display_argv,
            )

            if transition:
                prior_history_before = desktop_classifier.history_for_row(
                    old, snapshot=prior_snapshot
                )
                old_argv, old_argv_status, _ = _extract_argv_evidence(old)
                old_display = _extract_display_argv(old, old_argv, old_argv_status)
                old_input = {**old}
                if (
                    isinstance(prior_history_before, dict)
                    and (
                        prior_history_before.get("uid"),
                        prior_history_before.get("command"),
                        prior_history_before.get("argv"),
                        prior_history_before.get("argv_status"),
                        prior_history_before.get("argv_display"),
                    )
                    == (
                        old.get("uid"),
                        old.get("command"),
                        old_argv,
                        old_argv_status,
                        old_display,
                    )
                    and isinstance(prior_history_before.get("executable_status"), str)
                    and prior_history_before.get("executable_status")
                ):
                    old_input["executable_status"] = prior_history_before["executable_status"]
                prior_kind, prior_evidence, _, prior_history_after, _ = desktop_classifier.step_row(
                    old_input,
                    admit=False,
                    history_override=prior_history_before,
                    commit=True,
                    observation_ns=start,
                )
                current_hist = (
                    desktop_classifier._history.get(key)
                    if isinstance(getattr(desktop_classifier, "_history", None), dict)
                    else None
                )
                after_prior = (
                    dict(current_hist)
                    if (
                        isinstance(current_hist, dict)
                        and isinstance(prior_history_after, dict)
                        and current_hist.get("status") == "non_desktop"
                        and type(current_hist.get("sequence")) is int
                        and type(prior_history_after.get("sequence")) is int
                        and current_hist["sequence"] > prior_history_after["sequence"]
                    )
                    else prior_history_after
                )
                cur_kind, cur_evidence, _, row_history_after, _ = desktop_classifier.step_row(
                    row,
                    admit=False,
                    history_override=after_prior,
                    commit=True,
                    observation_ns=end,
                )
                kinds = {cur_kind, prior_kind}
                if "agent_workload" in kinds:
                    kind = "agent_workload"
                elif "unknown" in kinds:
                    kind = "unknown"
                else:
                    kind = cur_kind
                evidence = classification_evidence(kind, cur_evidence)
                row_history_before = after_prior
                row_admit = False
            else:
                if old is not None:
                    old_argv, old_argv_status, _ = _extract_argv_evidence(old)
                    row_argv, row_argv_status, _ = _extract_argv_evidence(row)
                    old_display = _extract_display_argv(old, old_argv, old_argv_status)
                    row_display = _extract_display_argv(row, row_argv, row_argv_status)
                    if (old_argv, old_argv_status, old_display) != (
                        row_argv,
                        row_argv_status,
                        row_display,
                    ):
                        proc_obs_map = getattr(
                            desktop_classifier, "_proc_last_observation_ns", None
                        )
                        last_proc_obs_ns = (
                            proc_obs_map.get(key) if isinstance(proc_obs_map, dict) else None
                        )
                        if last_proc_obs_ns is None:
                            last_proc_obs_ns = next(
                                (
                                    entry.get("observation_ns")
                                    for entry in reversed(
                                        getattr(desktop_classifier, "_history_log", ())
                                    )
                                    if (entry.get("pid"), entry.get("birth")) == key
                                ),
                                None,
                            )
                        if last_proc_obs_ns is None or (
                            type(last_proc_obs_ns) is int and last_proc_obs_ns <= start
                        ):
                            desktop_classifier.step_row(
                                old,
                                admit=True,
                                commit=True,
                                observation_ns=start,
                            )
                kind, evidence, row_history_before, row_history_after, _ = (
                    desktop_classifier.step_row(row, admit=True, observation_ns=end)
                )
                row_admit = True
        else:
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
            evidence = classification_evidence(kind, evidence)
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
        # Preserve every measured CPU contribution and every admitted UI/violation
        # binding. Idle unrelated processes remain in raw point inventories and
        # churn, rather than repeating thousands of zero rows for up to 7200s.
        if cpu > 0 or (
            desktop_classifier is not None
            and (
                kind in {"desktop_app", "desktop_renderer"}
                or evidence.get("identity_binding_changed")
                or evidence.get("reason") == "known_workload"
                or (
                    row_history_before is not None
                    and row_history_after is not None
                    and row_history_before != row_history_after
                )
            )
        ):
            exported = {**row, "kind": kind, "cpu_percent": cpu, "classification": evidence}
            if use_step_row and row_history_after is not None:
                exported["history_before"] = row_history_before
                exported["history_after"] = row_history_after
                exported["sequence"] = row_history_after["sequence"]
                exported["executable_status"] = row_history_after["executable_status"]
                exported["admit"] = row_admit
                exported["observation_ns"] = end
            if transition:
                # Replay needs both identities: this interval's conservative
                # classification is a precedence over the before/after rows,
                # not a claim that the after identity supplied all its CPU.
                transition_export = {
                    "reason": (
                        "uid_changed_no_delta"
                        if uid_changed
                        else (
                            "argv_changed_no_delta"
                            if argv_changed and old["command"] == row["command"]
                            else "exec_changed_no_delta"
                        )
                    ),
                    "previous": dict(old),
                    "previous_classification": (
                        prior_evidence if prior_evidence is not None else {"kind": prior_kind}
                    ),
                    "conservative_kind": kind,
                }
                if use_step_row and prior_history_after is not None:
                    transition_export["previous_history_before"] = prior_history_before
                    transition_export["previous_history_after"] = prior_history_after
                    transition_export["previous_sequence"] = prior_history_after["sequence"]
                    transition_export["previous_executable_status"] = prior_history_after[
                        "executable_status"
                    ]
                    transition_export["previous_observation_ns"] = start
                exported["interval_transition"] = transition_export
            rows.append(exported)
    for key, row in before.items():
        if key not in after and row["pid"] not in own_before:
            coverage = False
            churn.append({**row, "reason": "exited_process_no_final_counter"})
    if use_step_row:
        current["_process_history"] = desktop_classifier.history_snapshot()
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
