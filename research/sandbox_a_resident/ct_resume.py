"""Mac-side outcome-blind continuation builder with scientific-fault precedence for C/T.

Implements Ruling §4 & §5 and JUDGMENT_RESUME_POLICY.md:
- Torch-free Mac-side verification of completed ``stage-XXXX`` pairs by path,
  size, and SHA-256 sidecars only (never opening numerical ``.pt`` payloads
  unless explicitly encoding a new ``.pt.xz`` derivative on request).
- Qualified mechanical-stop acceptance aligned with ``research/colab_cli`` and
  TRAIN-01: admits verified ``RPC_EXEC_POLL_TIMEOUT``, ``CHILD_NONZERO_EXIT``
  with mechanical ``terminal.reason`` (``timeout``, ``lease_expired``,
  ``release_credential_expiring``, e.g. ``SIGUSR1`` exit ``138``), and
  mechanical ``SUPERVISOR`` wall/unit/lease/read-retry stops.
- Strict scientific-fault precedence: refuses continuation if ``summary.json``,
  ``fault.json``, non-empty ``faulthandler-stacks.log``, explicit fault/traceback
  evidence across any retained poll or terminal marker, or terminal
  ``INCOMPLETE_SUPPORT`` / ``INVALID`` stage metadata exist across any prior
  attempt directory.
- Cross-attempt stage union: merges verified stages across multiple prior
  attempt export directories without recomputation, refusing duplicate or
  conflicting stage units, and writes complete 6-file admitted continuation
  packets (``control.json``, ``registry-check.json``, ``topology-manifest.json``,
  ``verified-ledger.json``, ``dispatch-manifest.json``, ``sandbox.json``).
"""

from __future__ import annotations

import argparse
import ast
import codecs
import contextvars
import copy
import functools
import hashlib
import json
import os
import re
import stat
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, ParamSpec, Sequence, TypeVar

from research.sandbox_a_resident.ct_cuda_observed import BOOTSTRAP_LINES, CAPTURE_LABELS
from research.sandbox_a_resident.ct_prepare import STUDY, ct_endpoint_steps, write_once
from research.sandbox_a_resident.ct_progress import validate_heartbeat, validate_progress
from research.sandbox_a_resident.ct_units import (
    COMPACT_DERIVATIVE_SUFFIX,
    COMPACT_MANIFEST_SUFFIX,
    COMPACT_RESUME_FIELD,
    COMPACT_RESUME_SCHEMA,
    LEDGER_SCHEMA,
    NAMESPACE,
    SNAPSHOT_SCHEMA,
    allocation,
    encode_compact_stage_snapshot,
    unit_id,
    validate_compact_resume,
)
from research.sandbox_a_resident.units import digest
from tools.sandbox.manifest import load_manifest, validate_manifest
from tools.sandbox.registry import DEFAULT_REGISTRY, load_registry, validate_input, world_seed

RUNTIME_ROOT = "resume"
MAX_CONTINUATION_PRIVATE_INPUT_BYTES = 100_000_000
STAGING_HARD_CAP_SECONDS = 300.0
# Sample-fitted conservative upload parameters from colab-limits §4.1:
_WORST_UPLOAD_BYTES_PER_SEC = 810_734.0
_WORST_PER_FILE_OVERHEAD_SEC = 3.670
_WORST_MKDIR_PLUS_SPARSE_INSTALL_SEC = 1.82 + 3.47

MECHANICAL_TERMINAL_REASONS: frozenset[str] = frozenset(
    {
        "timeout",
        "lease_expired",
        "release_credential_expiring",
    }
)
MECHANICAL_SUPERVISOR_ERROR_MARKERS: tuple[str, ...] = (
    "whole-session wall cap reached",
    "whole-session deadline reached",
    "priced session unit cap reached",
    "study reserve reached",
    "retention lease elapsed",
    "read retry budget exhausted",
)
NON_MECHANICAL_SUPERVISOR_ERROR_MARKERS: tuple[str, ...] = (
    "keyboardinterrupt",
    "missing required export",
    "canary",
    "corrupt",
    "schema",
)
# Explicit scientific, stall-watchdog, overflow, or Python exception markers
# always override any timeout/lease/wall-cap marker. Note: "stall watchdog"
# (with space) and "stallwatchdog" do not match ct_job.sh's normal exit footer
# "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=.../stall-watchdog".
EXPLICIT_FAULT_EVIDENCE_MARKERS: tuple[str, ...] = (
    "keyboardinterrupt",
    "missing required export",
    "canary",
    "corrupt",
    "schemaerror",
    "colab-cli runtime:",
    "runtime supervisor failure",
    "runtime unassign failed",
    "floatingpointerror",
    "non-finite",
    "nonfinite ",
    '"finite": false',
    '"finite":false',
    "tripwire",
    "incomplete_support",
    "stageexportoverflowerror",
    "snapshotoverflowerror",
    "compactresumerefused",
    "ctresumerefused",
    "resumerefused",
    "stall watchdog",
    "stallwatchdog",
    "faulthandler",
    "timeout (",
    "traceback (most recent call last)",
    "assertionerror",
    "valueerror:",
    "typeerror:",
    "keyerror:",
    "indexerror:",
    "oserror:",
    "jsondecodeerror",
    "filenotfounderror",
    "permissionerror",
    "brokenpipeerror",
    "eoferror",
    "zerodivisionerror",
    "arithmeticerror",
    "overflowerror",
    "memoryerror",
    "cuda out of memory",
    "cuda error",
    "inductorerror",
    "backendcompilerfailed",
    "runtimeerror:",
)
SUPERVISOR_LOG_FAULT_MARKERS: tuple[str, ...] = (
    "colab-cli runtime:",
    "runtime supervisor failure",
    "runtime unassign failed",
    "traceback",
    "exception",
    "error:",
    "failed",
)


class CTResumeRefused(ValueError):
    """Raised when prior evidence is ineligible for outcome-blind continuation."""


@dataclass(frozen=True)
class CTResumePacket:
    """Create-only components for a manual outcome-blind C/T completion attempt."""

    ledger: dict[str, Any]
    ledger_sha256: str
    input_records: list[dict[str, str]]
    manifest: dict[str, Any]
    verified_stages: tuple[int, ...]
    staging_projection: dict[str, float | int]


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _normalize_export_roots(export_root: Path | Sequence[Path]) -> list[Path]:
    raw_list = [export_root] if isinstance(export_root, Path) else list(export_root)
    if not raw_list:
        raise CTResumeRefused("at least one prior export_root is required")
    out: list[Path] = []
    seen_resolved: set[Path] = set()
    for item in raw_list:
        p = Path(item)
        _no_symlink_path(p)
        resolved = p.resolve()
        if resolved in seen_resolved:
            raise CTResumeRefused(f"duplicate export_root supplied: {p}")
        seen_resolved.add(resolved)
        out.append(p)
    return out


def _complete(path: Path) -> bool:
    return all(
        p.is_file() for p in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    )


def _descriptor(path: Path, *, runtime_path: str) -> dict[str, Any]:
    sidecar = Path(str(path) + ".sha256")
    complete = Path(str(path) + ".complete")
    if not _complete(path):
        raise CTResumeRefused(f"incomplete export: {path.name}")
    if any(p.is_symlink() for p in (path, sidecar, complete)):
        raise CTResumeRefused(f"symlinked export refused: {path.name}")
    if sidecar.stat().st_size > 1024 or complete.stat().st_size > 4096:
        raise CTResumeRefused(f"oversized sidecar refused: {path.name}")
    raw_sidecar = sidecar.read_text()
    words = raw_sidecar.split()
    expected = words[0] if words else ""
    if not _is_hex(expected, 64):
        raise CTResumeRefused(f"malformed SHA-256 sidecar: {path.name}")
    if raw_sidecar not in {expected + "\n", f"{expected}  {path.name}\n"}:
        raise CTResumeRefused(f"malformed SHA-256 sidecar: {path.name}")
    if complete.read_bytes() != b"complete\n":
        raise CTResumeRefused(f"malformed completion marker: {path.name}")
    if _retained_digest(path) != expected:
        raise CTResumeRefused(f"corrupt export: {path.name}")
    return {"path": runtime_path, "size": path.stat().st_size, "sha256": expected}


def _refuse_if_fault_text(
    value: Any, *, field_name: str, check_supervisor_denylist: bool = False
) -> None:
    if not isinstance(value, str) or not value.strip():
        return
    lowered = value.lower()
    for marker in EXPLICIT_FAULT_EVIDENCE_MARKERS:
        if marker in lowered:
            raise CTResumeRefused(
                f"prior evidence records explicit scientific/non-mechanical fault "
                f"in {field_name}: {marker!r}"
            )
    if re.search(
        r"\b[A-Za-z_][\w.]*?(?:Error|Exception):|"
        r"^\s*[\w.]+(?:Error|Exception|Interrupt|Exit)\b|"
        r"Traceback \(most recent call last\)|Exception in thread|^\s*[\w.]+:\s",
        value,
        re.M,
    ):
        raise CTResumeRefused(
            f"prior evidence records scientific/non-mechanical exception in {field_name}"
        )
    if check_supervisor_denylist:
        for marker in NON_MECHANICAL_SUPERVISOR_ERROR_MARKERS:
            if marker in lowered:
                raise CTResumeRefused(
                    f"prior evidence records non-mechanical supervisor marker "
                    f"in {field_name}: {marker!r}"
                )


def _refuse_if_supervisor_log_fault(value: Any, *, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise CTResumeRefused(f"malformed non-string supervisor log in {field_name}")
    if not isinstance(value, str) or not value.strip():
        return
    _refuse_if_fault_text(value, field_name=field_name, check_supervisor_denylist=True)
    lowered = value.lower()
    for marker in SUPERVISOR_LOG_FAULT_MARKERS:
        if marker in lowered:
            raise CTResumeRefused(
                f"prior evidence records runtime supervisor fault in {field_name}: {marker!r}"
            )


def _refuse_if_fault_exports(exports: Any, *, source_name: str) -> None:
    if not isinstance(exports, list):
        return
    for item in exports:
        if not isinstance(item, Mapping):
            continue
        rel_path = str(item.get("path", "")).replace("\\", "/").lower()
        size = item.get("size")
        if rel_path.endswith("faulthandler-stacks.log") and isinstance(size, int) and size > 0:
            raise CTResumeRefused(
                f"{source_name} exported non-empty faulthandler-stacks.log (stall watchdog fired)"
            )
        if rel_path.endswith("summary.json") or rel_path.endswith("fault.json"):
            raise CTResumeRefused(
                f"{source_name} exported terminal diagnostic/summary artifact: {rel_path}"
            )


def _check_runtime_state_fault_evidence(state: Mapping[str, Any], *, label: str) -> None:
    """Validate qualified ``runtime.read_state()['state']`` or persisted ``state.json``."""
    rc = state.get("returncode")
    if rc is not None:
        if type(rc) is not int:
            raise CTResumeRefused(f"{label} carries malformed returncode={rc!r}")
        if rc == 124:
            raise CTResumeRefused(f"{label} records watchdog timeout exit code 124")
        if rc == 0:
            raise CTResumeRefused(f"{label} records clean child completion returncode=0")

    for err_key in ("error", "unassign_error"):
        err_val = state.get(err_key)
        _refuse_if_fault_text(
            err_val, field_name=f"{label}.{err_key}", check_supervisor_denylist=True
        )
        if err_val is not None and (not isinstance(err_val, str) or bool(err_val.strip())):
            raise CTResumeRefused(f"{label} records runtime fault {err_key}={err_val!r}")

    raw_reason = state.get("reason")
    if raw_reason is not None and not isinstance(raw_reason, str):
        raise CTResumeRefused(f"{label} carries malformed reason={raw_reason!r}")
    s_reason = (
        raw_reason.strip().lower() if isinstance(raw_reason, str) and raw_reason.strip() else None
    )
    if s_reason is not None:
        if any(
            tok in s_reason
            for tok in (
                "fault",
                "exception",
                "invalid",
                "incomplete_support",
                "runtime_error",
                "popen_failed",
            )
        ):
            raise CTResumeRefused(f"{label} reports scientific/runtime fault reason={raw_reason!r}")
        if s_reason not in MECHANICAL_TERMINAL_REASONS:
            raise CTResumeRefused(f"{label} reports non-mechanical reason={raw_reason!r}")

    raw_status = state.get("status")
    s_status = raw_status.strip().lower() if isinstance(raw_status, str) else ""
    if s_status in {"starting", "running"}:
        if rc is not None:
            raise CTResumeRefused(f"{label} reports returncode={rc!r} while status={raw_status!r}")
    elif s_status == "failed":
        if s_reason not in MECHANICAL_TERMINAL_REASONS:
            raise CTResumeRefused(
                f"{label} records a non-mechanical child failure "
                f"(reason={raw_reason!r}, returncode={rc!r})"
            )
    else:
        raise CTResumeRefused(f"{label} reports non-resumable status={raw_status!r}")

    _refuse_if_fault_exports(state.get("exports"), source_name=label)


def _check_terminal_fault_evidence(terminal: Mapping[str, Any], *, label: str) -> None:
    t_rc = terminal.get("returncode")
    if t_rc is not None:
        if type(t_rc) is not int:
            raise CTResumeRefused(f"{label} carries malformed returncode={t_rc!r}")
        if t_rc == 124:
            raise CTResumeRefused(f"{label} records watchdog timeout exit code 124")
        if t_rc == 0:
            raise CTResumeRefused(f"{label} records clean child completion returncode=0")

    for err_key in ("error", "unassign_error"):
        err_val = terminal.get(err_key)
        _refuse_if_fault_text(
            err_val, field_name=f"{label}.{err_key}", check_supervisor_denylist=True
        )
        if err_val is not None and (not isinstance(err_val, str) or bool(err_val.strip())):
            raise CTResumeRefused(f"{label} records runtime fault {err_key}={err_val!r}")

    t_status = str(terminal.get("status", "")).strip().lower()
    if t_status in {
        "completed",
        "succeeded",
        "success",
        "pass",
        "lease_expired_after_terminal",
        "fault",
        "invalid",
        "incomplete_support",
    } or ("status" in terminal and t_status != "failed"):
        raise CTResumeRefused(f"{label} reports non-resumable status={terminal.get('status')!r}")

    raw_reason = terminal.get("reason")
    if raw_reason is not None and not isinstance(raw_reason, str):
        raise CTResumeRefused(f"{label} carries malformed reason={raw_reason!r}")
    t_reason = (
        raw_reason.strip().lower() if isinstance(raw_reason, str) and raw_reason.strip() else None
    )
    if t_reason is not None and any(
        tok in t_reason
        for tok in (
            "fault",
            "exception",
            "invalid",
            "incomplete_support",
            "runtime_error",
            "popen_failed",
        )
    ):
        raise CTResumeRefused(f"{label} reports scientific/runtime fault reason={raw_reason!r}")
    if t_rc not in (None, 0) and t_reason not in MECHANICAL_TERMINAL_REASONS:
        raise CTResumeRefused(f"{label} records a non-mechanical child failure")
    if t_reason is not None and t_reason not in MECHANICAL_TERMINAL_REASONS:
        raise CTResumeRefused(f"{label} reports non-mechanical reason={raw_reason!r}")

    for key in ("last_traceback_line", "last_error_line", "job_log_tail"):
        _refuse_if_fault_text(terminal.get(key), field_name=f"{label}.{key}")
    _refuse_if_supervisor_log_fault(
        terminal.get("supervisor_log_tail"), field_name=f"{label}.supervisor_log_tail"
    )
    _refuse_if_fault_exports(terminal.get("exports"), source_name=label)
    scanner = terminal.get("export_scanner")
    if isinstance(scanner, Mapping):
        _refuse_if_fault_exports(scanner.get("inventory"), source_name=f"{label}.export_scanner")


def _check_poll_fault_evidence(poll_payload: Mapping[str, Any], *, label: str) -> None:
    """Inspect qualified nested ``state``, ``supervisor_log_tail``, ``job_log_tail`` & terminal."""
    if "state" in poll_payload:
        poll_state = poll_payload.get("state")
        if not isinstance(poll_state, Mapping):
            raise CTResumeRefused(f"{label}.state is malformed")
        _check_runtime_state_fault_evidence(poll_state, label=f"{label}.state")

    flat_state_keys = ("status", "reason", "returncode", "error", "unassign_error")
    if any(k in poll_payload for k in flat_state_keys):
        synthetic_state: dict[str, Any] = {
            k: poll_payload[k] for k in flat_state_keys if k in poll_payload
        }
        if "status" not in synthetic_state and synthetic_state.get("returncode") is None:
            synthetic_state["status"] = "running"
        _check_runtime_state_fault_evidence(synthetic_state, label=label)

    _refuse_if_fault_text(poll_payload.get("job_log_tail"), field_name=f"{label}.job_log_tail")
    _refuse_if_supervisor_log_fault(
        poll_payload.get("supervisor_log_tail"), field_name=f"{label}.supervisor_log_tail"
    )
    _refuse_if_fault_exports(poll_payload.get("exports"), source_name=label)
    poll_term = poll_payload.get("terminal")
    if poll_term is not None:
        if not isinstance(poll_term, Mapping):
            raise CTResumeRefused(f"{label}.terminal is malformed")
        _check_terminal_fault_evidence(poll_term, label=f"{label}.terminal")


def _check_retained_execution(
    record: Mapping[str, Any],
    *,
    label: str,
    expected_job_id: str | None,
    expected_generation: str | None,
    require_identity: bool = False,
    fixture_completion: bool = False,
) -> None:
    """Validate execution facts independently of human-readable fault markers."""
    for key, expected in (
        ("job_id", expected_job_id),
        ("job", expected_job_id),
        ("generation", expected_generation),
    ):
        if expected is not None and key in record and record[key] != expected:
            raise CTResumeRefused(f"{label} belongs to foreign {key}")
    if require_identity:
        for key in ("job_id", "generation"):
            item = record.get(key, record.get("job") if key == "job_id" else None)
            if not isinstance(item, str) or not item.strip():
                raise CTResumeRefused(f"{label} lacks matching {key} identity")
    status = record.get("status")
    reason = record.get("reason")
    if status is not None and not isinstance(status, str):
        raise CTResumeRefused(f"{label} records malformed execution status")
    if reason is not None and not isinstance(reason, str):
        raise CTResumeRefused(f"{label} records malformed execution reason")
    returncode = record.get("returncode")
    if not fixture_completion and (returncode == 0 or status in {"completed", "succeeded"}):
        raise CTResumeRefused(f"{label} records completed status")
    if reason is not None and reason not in MECHANICAL_TERMINAL_REASONS:
        raise CTResumeRefused(f"{label} records non-mechanical reason: {reason!r}")
    if returncode is not None and type(returncode) is not int:
        raise CTResumeRefused(f"{label} records malformed returncode")
    if returncode == 124:
        raise CTResumeRefused(f"{label} records watchdog timeout exit code 124")
    if reason in MECHANICAL_TERMINAL_REASONS and returncode not in (None, 138, -10):
        raise CTResumeRefused(f"{label} records non-signal exit under a mechanical reason")
    fixture_signal = (
        fixture_completion
        and set(record)
        == {"name", "returncode", "runtime", "checkpoint_bytes", "signal_to_exit_seconds", "marker"}
        and record.get("name") == "signal"
        and returncode == 138
    )
    if fixture_signal:
        _validate_signal_marker(record["marker"])
    if (returncode not in (None, 0) or status in {"failed", "stopped", "error"}) and (
        reason not in MECHANICAL_TERMINAL_REASONS and not fixture_signal
    ):
        raise CTResumeRefused(
            f"{label} records non-mechanical child failure or unexplained runtime failure"
        )
    if status is not None and status not in (
        {"completed", "succeeded", "PASS"} if fixture_completion else set()
    ) | {
        "running",
        "starting",
        "ready",
        "failed",
        "stopped",
        "error",
    }:
        raise CTResumeRefused(f"{label} records unknown execution status: {status!r}")


_EVIDENCE_CACHE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "ct_evidence_cache", default=None
)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _evidence_session(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Cache successful immutable-evidence work only for one admission operation."""

    @functools.wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if _EVIDENCE_CACHE.get() is not None:
            return function(*args, **kwargs)
        token = _EVIDENCE_CACHE.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _EVIDENCE_CACHE.reset(token)

    return wrapped


def _no_symlink_path(path: Path) -> None:
    if (
        not path.is_absolute()
        or str(path) != os.path.normpath(str(path))
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise CTResumeRefused(f"path must be absolute without symlink ancestors: {path}")


def _retained_digest(path: Path) -> str:
    cache = _EVIDENCE_CACHE.get()
    info = path.stat()
    key = (
        "digest",
        str(path),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )
    if cache is None:
        return digest(path)
    if key not in cache:
        cache[key] = digest(path)
    return cache[key]


def _retained_files(output_root: Path) -> list[Path]:
    """Walk every retained file without following links or accepting special files."""
    files = []
    for path in sorted(output_root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise CTResumeRefused(f"non-regular or symlinked retained evidence: {path}")
        if stat.S_ISREG(mode):
            files.append(path)
    return files


def _mechanical_supervisor_error(value: Any) -> bool:
    """Accept a whole qualified cause, never a compound error containing one."""
    if not isinstance(value, str):
        return False
    causes = set(MECHANICAL_SUPERVISOR_ERROR_MARKERS) | {
        "whole-session wall cap reached (including setup/export)",
        "whole-session deadline reached before helper",
        "whole-session deadline reached before CLI call",
        "study reserve reached during calibration",
        "retention lease elapsed; no further job",
        "priced session unit cap reached during calibration",
        "priced session unit cap reached while awaiting account broker",
    }
    cause = value
    # Parse one repr wrapper, never strip a prefix or accept a compound cause.
    match = re.fullmatch(r"(SupervisorError|Refused)\((.*)\)", value, re.S)
    if match:
        try:
            cause = ast.literal_eval(match[2])
        except (ValueError, SyntaxError, RecursionError):
            return False
    return isinstance(cause, str) and cause in causes


def _observed_signal_line(value: Any, generation: str | None = None) -> bool:
    """Exact observed benign lines; a mechanical reason never exempts arbitrary text."""
    if isinstance(value, str) and value.startswith("CT_SIGUSR1_CLEAN_EXIT ") and "\n" not in value:
        try:
            _validate_signal_marker(_load_json(value.partition(" ")[2]))
            return True
        except (ValueError, TypeError, KeyError, RecursionError):
            return False
    if isinstance(generation, str) and re.fullmatch(r"[0-9a-f]{32}", generation):
        if value == (
            f"STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/snake-cli/{generation}"
            "/output/ct/stall-watchdog"
        ):
            return True
    return (
        value
        in {
            "CT_CHILD_READY",
            "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/out/ct/stall-watchdog",
            "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/session/output/ct/stall-watchdog",
        }
        if isinstance(value, str)
        else False
    )


def _validate_signal_marker(row: Any) -> None:
    """One closed real-producer marker; false/slow/malformed SLA never qualifies."""
    if not isinstance(row, dict):
        raise ValueError("malformed signal marker")
    expected = {
        "schema",
        "study",
        "allocation",
        "status",
        "interrupted_phase",
        "latest_durable_stage",
        "received_unix",
        "shutdown_elapsed_sec",
        "within_15s_sla",
    }
    if (
        set(row) != expected
        or row["schema"] != "t34-cf-sigusr1-marker/v1"
        or row["study"] != "t34-cf-1028"
        or row["status"] != "PREEMPTED_SIGUSR1"
        or not _is_hex(row["allocation"], 64)
        or not isinstance(row["interrupted_phase"], str)
        or not re.fullmatch(r"stage-\d{4}-(collection|scoring|fit)", row["interrupted_phase"])
        or row["within_15s_sla"] is not True
        or type(row["shutdown_elapsed_sec"]) not in (int, float)
        or not 0 <= row["shutdown_elapsed_sec"] < 15
        or any(
            type(row[k]) not in (int, float) or not 0 <= row[k] < float("inf")
            for k in ("latest_durable_stage", "received_unix")
        )
    ):
        raise ValueError("unqualified stop marker")


def _scan_job_tail(
    value: Any, *, label: str, generation: str | None = None, fixture_completion: bool = False
) -> None:
    """Accept observed lines or closed C/T progress schemas, never arbitrary log text."""
    if value in (None, ""):
        return
    if not isinstance(value, str):
        raise CTResumeRefused(f"malformed job_log_tail: {label}")
    lines = value.splitlines()
    # The qualified producer slices bytes, not lines (runtime._tail). Never
    # discard a partial line: require its exact suffix in a fully validated,
    # retained producer history. An appended fault cannot obtain this witness.
    for index, line in enumerate(lines):
        if not line or line in BOOTSTRAP_LINES or _observed_signal_line(line, generation):
            continue
        prefix, _, payload = line.partition(" ")
        try:
            row = _load_json(payload)
            if not isinstance(row, dict):
                raise ValueError("unknown line")
            if prefix == "CT_PROGRESS":
                validate_progress(row, fixture_completion=fixture_completion)
            elif prefix == "CT_HEARTBEAT":
                validate_heartbeat(row)
            elif prefix == "CT_SIGUSR1_CLEAN_EXIT":
                _validate_signal_marker(row)
            else:
                raise ValueError("unknown line")
        except (ValueError, TypeError, RecursionError):
            _refuse_if_fault_text(line, field_name=label)
            witnesses = (_EVIDENCE_CACHE.get() or {}).get("progress_history_lines", set())
            if (
                index == 0
                and prefix not in {"CT_PROGRESS", "CT_HEARTBEAT", "CT_SIGUSR1_CLEAN_EXIT"}
                and len(value.encode("utf-8")) in {4096, 65536}
                and any(known.endswith(line) for known in witnesses)
            ):
                continue
            # Values can contain outcome-bearing counters. Keep refusals blind.
            raise CTResumeRefused(f"unrecognized job_log_tail line in {label}") from None


def _check_retained_receipt(record: Any, *, label: str) -> None:
    if not isinstance(record, Mapping):
        raise CTResumeRefused(f"malformed retained receipt: {label}")
    if record.get("failure_class") not in {
        "SUPERVISOR",
        "RPC_EXEC_POLL_TIMEOUT",
        "CHILD_NONZERO_EXIT",
    }:
        raise CTResumeRefused(
            f"prior receipt failure_class {record.get('failure_class')!r} "
            "is not a qualified mechanical trigger"
        )
    if (
        record.get("status") != "INCOMPLETE"
        or record.get("stop_verified") is not True
        or record.get("cleanup_error") is not None
        or record.get("failure_class")
        not in {"SUPERVISOR", "RPC_EXEC_POLL_TIMEOUT", "CHILD_NONZERO_EXIT"}
    ):
        raise CTResumeRefused(f"non-resumable retained receipt: {label}")
    error = record.get("error")
    _refuse_if_fault_text(error, field_name=label, check_supervisor_denylist=True)
    terminal = record.get("terminal")
    if isinstance(terminal, Mapping) and terminal.get("returncode") == 124:
        raise CTResumeRefused(f"{label} records watchdog timeout exit code 124")
    if (
        isinstance(terminal, Mapping)
        and terminal.get("reason") in MECHANICAL_TERMINAL_REASONS
        and terminal.get("returncode") not in (None, 138, -10)
    ):
        raise CTResumeRefused(f"{label} records non-signal exit under a mechanical reason")
    child_error = False
    if (
        record["failure_class"] == "CHILD_NONZERO_EXIT"
        and isinstance(terminal, Mapping)
        and terminal.get("reason") in MECHANICAL_TERMINAL_REASONS
        and terminal.get("returncode") in {138, -10}
    ):
        # failures.terminal_failure formats this exact detail; supervisor records repr(exc).
        line = terminal.get("last_traceback_line") or terminal.get("last_error_line")
        if not isinstance(line, str) or not line.strip():
            line = terminal["reason"]
        detail = (
            f"remote child exited {terminal.get('returncode')}: {line.strip().splitlines()[-1]}"
        )
        legacy_detail = (
            f"remote child exited {terminal.get('returncode')}: "
            "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/out/ct/stall-watchdog"
        )
        child_error = error in (detail, f"ClassifiedFailure({detail!r})", legacy_detail)
    if error not in (None, "") and not (_mechanical_supervisor_error(error) or child_error):
        raise CTResumeRefused(f"unexplained retained receipt error: {label}")
    terminal = record.get("terminal")
    if record["failure_class"] == "CHILD_NONZERO_EXIT" and (
        not isinstance(terminal, Mapping)
        or terminal.get("reason") not in MECHANICAL_TERMINAL_REASONS
        or terminal.get("returncode") not in {138, -10}
    ):
        raise CTResumeRefused(
            f"non-mechanical child failure: retained receipt lacks mechanical terminal: {label}"
        )


def _scan_json_faults(
    value: Any,
    *,
    label: str,
    expected_job_id: str | None = None,
    expected_generation: str | None = None,
    supervisor_transport: bool = False,
    supervisor_log: bool = False,
    require_execution_identity: bool = False,
    fixture_completion: bool = False,
) -> None:
    """Inspect nested diagnostic fields, including error fields in snapshot headers."""
    if isinstance(value, Mapping):
        receipt = value.get("status") == "INCOMPLETE" and any(
            key in value for key in ("stop_verified", "failure_class", "cleanup_error")
        )
        if receipt:
            _check_retained_receipt(value, label=label)
        if (
            "failure_class" in value
            and not receipt
            and not (
                fixture_completion
                and value.get("status") == "PASS"
                and value["failure_class"] is None
            )
        ):
            raise CTResumeRefused(f"{label} records failure_class outside a qualified receipt")
        # These exact qualified transport records describe completed/failed
        # RPCs and transfers, never completion of the scientific child.
        transport_record = supervisor_transport and (
            {"started_monotonic", "ended_monotonic", "elapsed_seconds"}.issubset(value)
            and ("operation" in value or "direction" in value)
            and "job_id" not in value
            and "generation" not in value
            and "returncode" not in value
            and "reason" not in value
        )
        transport_summary = (
            supervisor_transport
            and set(value) == {"job", "elapsed", "status", "terminal", "exports", "job_log_tail"}
            and type(value.get("terminal")) is bool
            and type(value.get("exports")) is int
            and isinstance(value.get("elapsed"), (int, float))
        )
        execution = not (transport_record or transport_summary) and (
            require_execution_identity
            or any(key in value for key in ("returncode", "reason", "failure_class"))
            or ("status" in value and any(key in value for key in ("job", "job_id", "generation")))
            or value.get("status") in ("failed", "stopped", "error", "completed", "succeeded")
        )
        execution_record = {k: v for k, v in value.items() if not (receipt and k == "status")}
        for key, item in value.items():
            name = f"{label}.{key}"
            mechanical_error = key == "error" and (
                receipt or supervisor_transport and _mechanical_supervisor_error(item)
            )
            if (
                key
                in {
                    "error",
                    "exception",
                    "last_error_line",
                    "last_traceback_line",
                    "unassign_error",
                    "fault",
                    "error_type",
                    "error_message",
                    "cleanup_error",
                    "traceback",
                }
                and item
                and not mechanical_error
                and not (
                    key in {"last_error_line", "last_traceback_line"}
                    and isinstance(item, str)
                    and _observed_signal_line(item, expected_generation)
                    and "\n" not in item
                )
            ):
                _refuse_if_fault_text(item, field_name=name, check_supervisor_denylist=True)
                raise CTResumeRefused(
                    f"{name} records explicit scientific/non-mechanical fault: {item!r}"
                )
            if key in {"phase", "prepare_phase", "stage"} and isinstance(item, str):
                if item.lower().endswith("failed") or "failure" in item.lower():
                    raise CTResumeRefused(f"{name} records failed stage")
            if (
                (key == "finite" and item is False)
                or (key == "material_gaps" and item)
                or (key == "scientific_fault" and item)
                or (key == "support_status" and item in {"INCOMPLETE_SUPPORT", "INVALID"})
            ):
                _refuse_if_fault_text(item, field_name=name, check_supervisor_denylist=True)
                raise CTResumeRefused(
                    f"{name} records explicit scientific/non-mechanical fault: {item!r}"
                )
            # Inventories name diagnostic files even when they are empty. Their
            # paths are checked by the export scanner; don't treat filenames as errors.
            if key == "job_log_tail":
                _scan_job_tail(
                    item,
                    label=name,
                    generation=expected_generation,
                    fixture_completion=fixture_completion,
                )
                continue
            if key == "schema" and item == "colab-family-canary/v1":
                continue
            if key not in {"path", "dest", "sha256"} or not isinstance(item, str):
                _scan_json_faults(
                    item,
                    label=name,
                    expected_job_id=expected_job_id,
                    expected_generation=expected_generation,
                    supervisor_transport=supervisor_transport,
                    supervisor_log=supervisor_log,
                    fixture_completion=fixture_completion,
                    require_execution_identity=(
                        key in {"state", "terminal"} and isinstance(item, Mapping)
                    ),
                )
        _check_retained_execution(
            (
                execution_record
                if receipt
                else (
                    value
                    if execution
                    else {k: v for k, v in value.items() if k in {"job", "job_id", "generation"}}
                )
            ),
            label=label,
            expected_job_id=expected_job_id,
            expected_generation=expected_generation,
            require_identity=require_execution_identity,
            fixture_completion=fixture_completion,
        )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_json_faults(
                item,
                label=f"{label}[{index}]",
                expected_job_id=expected_job_id,
                expected_generation=expected_generation,
                supervisor_transport=supervisor_transport,
                supervisor_log=supervisor_log,
                fixture_completion=fixture_completion,
            )
    elif isinstance(value, str):
        if fixture_completion and value in CAPTURE_LABELS:
            return
        _refuse_if_fault_text(value, field_name=label, check_supervisor_denylist=supervisor_log)
        if (
            supervisor_log
            and not label.endswith(".status")
            and not _mechanical_supervisor_error(value)
        ):
            _refuse_if_supervisor_log_fault(value, field_name=label)


def _verify_staged_source_archive(path: Path, output_root: Path) -> None:
    """Verify every member against the root seal without Git or gzip metadata identity."""
    try:
        plan_path = path.parent / "source-plan.json"
        if plan_path.stat().st_size > 1024 * 1024:
            raise ValueError("oversized source plan")
        plan = _load_json(plan_path.read_text())
        manifest = _load_json((output_root / "manifest.json").read_text())
        binding = (_EVIDENCE_CACHE.get() or {}).get("source_archive_binding", {})
        from research.colab_cli.source import _required, _safe_path

        expected = binding.get("source_files")
        commit = binding.get("source_commit")
        if (
            not isinstance(plan, Mapping)
            or not isinstance(manifest, Mapping)
            or not isinstance(manifest.get("source_paths"), list)
            or set(plan)
            != {
                "version",
                "commit",
                "tree",
                "paths",
                "fallback_archive",
                "fallback_sha256",
                "fallback_size",
                "fallback_unpacked_size",
                "files",
            }
            or plan.get("version") != 1
            or not _is_hex(plan.get("tree"), 40)
            or plan.get("tree") != binding.get("source_tree")
            or plan.get("paths")
            != sorted(
                {_safe_path(raw) for raw in [*manifest["source_paths"], *_required(manifest)]}
            )
            or not isinstance(plan.get("files"), Mapping)
            or any(
                not isinstance(row, Mapping)
                or set(row) != {"sha256", "size", "mode"}
                or type(row.get("size")) is not int
                or row["size"] < 0
                or row.get("mode") not in {"0644", "0755"}
                for row in plan["files"].values()
            )
        ):
            raise ValueError("source plan contains unqualified fields or member descriptors")
        if (
            not isinstance(expected, Mapping)
            or not expected
            or not _is_hex(commit, 40)
            or manifest.get("commit") != commit
            or plan.get("commit") != commit
            or plan.get("fallback_archive") != path.name
            or plan.get("fallback_size") != path.stat().st_size
            or path.stat().st_size > 32 * 1024 * 1024
            or plan.get("fallback_sha256") != _retained_digest(path)
            or not isinstance(plan.get("files"), Mapping)
            or {k: v.get("sha256") for k, v in plan["files"].items()} != expected
        ):
            raise ValueError(
                "source plan/archive differs from sealed source_commit/files/size/SHA-256"
            )
        seen, total = set(), 0
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                name = member.name
                if (
                    name in seen
                    or name not in expected
                    or not member.isfile()
                    or Path(name).is_absolute()
                    or ".." in Path(name).parts
                ):
                    raise ValueError("foreign/duplicate/nonregular archive member")
                seen.add(name)
                total += member.size
                row = plan["files"][name]
                if (
                    total > 32 * 1024 * 1024
                    or member.size != row["size"]
                    or f"{member.mode:04o}" != row["mode"]
                ):
                    raise ValueError("source member size/mode mismatch")
                stream = archive.extractfile(member)
                if (
                    stream is None
                    or hashlib.file_digest(stream, "sha256").hexdigest() != expected[name]
                ):
                    raise ValueError("source member SHA-256 mismatch")
        if seen != set(expected) or total != plan["fallback_unpacked_size"]:
            raise ValueError("missing sealed source members or unpacked-size mismatch")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        RecursionError,
        tarfile.TarError,
    ) as error:
        raise CTResumeRefused(
            f"unverified retained staged source archive: {path}: {error}"
        ) from error


def _publication_sidecar(path: Path) -> bool:
    if path.suffix not in {".sha256", ".complete"}:
        return False
    payload = path.with_suffix("")
    if not _complete(payload):
        return False
    _descriptor(payload, runtime_path=payload.name)
    return True


def _scan_diagnostic_bytes(path: Path, *, json_mode: bool = False, collect: bool = True) -> str:
    """Scan the entire readable stream first, then refuse invalid encoding/NULs."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    strict_decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    bad_utf8 = False
    overlap, chunks, invalid = "", [], False
    with path.open("rb") as stream:
        while raw := stream.read(65536):
            invalid |= b"\x00" in raw
            part = decoder.decode(raw).replace("\x00", "")
            if not bad_utf8:
                try:
                    strict_decoder.decode(raw)
                except UnicodeDecodeError:
                    bad_utf8 = True
            if not json_mode:
                _refuse_if_fault_text(overlap + part, field_name=str(path))
            overlap = (overlap + part)[-256:]
            if collect:
                chunks.append(part)
    tail = decoder.decode(b"", final=True)
    if not bad_utf8:
        try:
            strict_decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            bad_utf8 = True
    invalid |= bad_utf8
    if not json_mode or invalid:
        _refuse_if_fault_text(
            ("".join(chunks) if json_mode else overlap) + tail, field_name=str(path)
        )
    if invalid:
        raise CTResumeRefused(f"unreadable retained diagnostic evidence: {path}")
    return "".join(chunks) + tail


def _json_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CTResumeRefused(f"duplicate JSON key in retained evidence: {key!r}")
        result[key] = value
    return result


def _load_json(text: str | bytes) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_json_without_duplicates)
    except RecursionError as error:
        raise CTResumeRefused("excessively nested diagnostic JSON") from error
    # JSON's C decoder and Python walkers have different recursion limits.
    # Bound nesting iteratively before any recursive diagnostic inspection.
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 128:
            raise CTResumeRefused("excessively nested diagnostic JSON")
        children = (
            item.values() if isinstance(item, dict) else item if isinstance(item, list) else ()
        )
        pending.extend((child, depth + 1) for child in children if isinstance(child, (dict, list)))
    return value


def _verified_stage_payload(path: Path) -> bool:
    match = re.fullmatch(r"stage-(\d{4})\.pt(?:\.xz)?", path.name)
    if match is None:
        return False
    header = path.parent / f"stage-{match[1]}.json"
    if not _complete(header) or not _complete(path):
        raise CTResumeRefused(f"unverified retained binary stage payload: {path}")
    _descriptor(header, runtime_path=header.name)
    info = _load_json(header.read_text())
    descriptor = _descriptor(path, runtime_path=path.name)
    if path.suffix == ".xz":
        manifest = Path(str(path) + ".manifest")
        _descriptor(manifest, runtime_path=manifest.name)
        meta = _load_json(manifest.read_text())
        bound = meta.get("derivative", {})
        original = meta.get("original", {})
        if meta.get("unit") != info.get("id") or original.get("sha256") != info.get(
            "snapshot", {}
        ).get("sha256"):
            raise CTResumeRefused(f"unbound compact stage payload: {path}")
    else:
        bound = info.get("snapshot", {})
    if (
        info.get("schema") != SNAPSHOT_SCHEMA
        or info.get("stage") != int(match[1])
        or info.get("id") != unit_id(int(match[1]))
        or bound.get("sha256") != descriptor["sha256"]
        or bound.get("size") != descriptor["size"]
    ):
        raise CTResumeRefused(f"unbound retained stage payload: {path}")
    return True


def _scan_retained_text(
    path: Path,
    *,
    output_root: Path,
    expected_job_id: str | None = None,
    expected_generation: str | None = None,
    supervisor_receipt: Mapping[str, Any] | None = None,
    supervisor_receipt_path: Path | None = None,
) -> None:
    """Validate every diagnostic representation, refusing archives and incomplete writes."""
    if path.parent == output_root / "staging" and path.name in {
        "bootstrap.sh",
        "source.py",
        "runtime.py",
        "release.py",
    }:
        source = Path(__file__).resolve().parents[1] / "colab_cli" / path.name
        if _retained_digest(path) != _retained_digest(source):
            raise CTResumeRefused(
                f"retained staged executable differs from qualified source: {path}"
            )
        return
    if path == output_root / "staging/source.tar.gz":
        _verify_staged_source_archive(path, output_root)
        return
    if path == output_root / "staging/source-plan.json":
        # This is a source inventory, with filenames such as canary_observer.py.
        # Exempt only its exact sealed bytes, never an unbound diagnostic mapping.
        _verify_staged_source_archive(path.parent / "source.tar.gz", output_root)
        return
    if _verified_stage_payload(path) or _publication_sidecar(path):
        return
    with path.open("rb") as stream:
        header = stream.read(8)
    if (
        header.startswith(
            (
                b"\x1f\x8b",
                b"\xfd7zXZ\x00",
                b"PK\x03\x04",
                b"BZh",
                b"\x28\xb5\x2f\xfd",
                b"7z\xbc\xaf\x27\x1c",
            )
        )
        or any(
            part in {".gz", ".xz", ".zip", ".bz2", ".zst", ".tar", ".7z"} for part in path.suffixes
        )
        and not re.fullmatch(r"stage-\d{4}\.pt\.xz\.manifest", path.name)
    ):
        raise CTResumeRefused(f"unsupported retained diagnostic archive: {path}")
    if path.suffix in {".pt", ".pth"}:
        raise CTResumeRefused(f"unexpected retained binary payload: {path}")
    name = path.name.lstrip(".")
    if name.startswith("supervisor.log"):
        _scan_supervisor_log(
            path,
            job_id=expected_job_id,
            generation=expected_generation,
            receipt=supervisor_receipt,
            receipt_path=supervisor_receipt_path,
        )
        if path.name.startswith(".") or any(
            part.startswith((".partial", ".tmp")) for part in path.suffixes
        ):
            raise CTResumeRefused(f"incomplete retained diagnostic evidence: {path}")
        return
    is_json = any(suffix in {".json", ".jsonl"} for suffix in path.suffixes) or name.endswith(
        ".xz.manifest"
    )
    jsonl = ".jsonl" in path.suffixes
    text = _scan_diagnostic_bytes(path, json_mode=is_json, collect=not jsonl)
    if is_json:
        stream = path.open(encoding="utf-8", newline="") if jsonl else None
        try:
            records = (
                (_load_json(line) for line in stream if line.strip())
                if stream is not None
                else [_load_json(text)]
            )
            count = 0
            for record in records:
                count += 1
                execution_file = name.startswith(("state.json", "terminal.json", "poll-"))
                if (execution_file or name.startswith("receipt.json") or jsonl) and not isinstance(
                    record, Mapping
                ):
                    raise CTResumeRefused(f"malformed retained execution record: {path}")
                if name.startswith("receipt.json"):
                    _check_retained_receipt(record, label=str(path))
                _scan_json_faults(
                    record,
                    label=str(path),
                    expected_job_id=expected_job_id,
                    expected_generation=expected_generation,
                    supervisor_transport=name.startswith(
                        ("poll-", "receipt.json", "terminal.json", "rate-calibration")
                    ),
                    require_execution_identity=(execution_file and not name.startswith("poll-")),
                )
                if name.startswith("poll-"):
                    _check_poll_fault_evidence(record, label=str(path))
            if not count:
                raise ValueError("empty JSONL evidence")
        except CTResumeRefused:
            raise
        except ValueError as error:
            _refuse_if_fault_text(text, field_name=str(path))
            label = (
                "corrupt poll evidence"
                if name.startswith("poll-")
                else "unreadable or partial retained JSON evidence"
            )
            raise CTResumeRefused(f"{label}: {path}") from error
        finally:
            if stream is not None:
                stream.close()
    if path.name.startswith(".") or any(
        part.startswith((".partial", ".tmp")) for part in path.suffixes
    ):
        raise CTResumeRefused(f"incomplete retained diagnostic evidence: {path}")


def retained_inventory(root: Path) -> dict[str, Any]:
    """Bind every retained byte by path/size/hash without decoding numerical payloads."""
    _no_symlink_path(root)
    if not root.exists():
        return {"exists": False, "files": []}
    if not root.is_dir():
        raise CTResumeRefused(f"retained evidence root is not a directory: {root}")
    return {
        "exists": True,
        "files": [
            {
                "path": str(p.relative_to(root)),
                "size": p.stat().st_size,
                "sha256": _retained_digest(p),
            }
            for p in _retained_files(root)
        ],
    }


def _check_attempt_base_fault_evidence(
    base: Path,
    *,
    expected_job_id: str | None = None,
    expected_generation: str | None = None,
) -> None:
    """Inspect the entire retained diagnostic set; model payloads remain hash-only."""
    _no_symlink_path(base)
    cache = _EVIDENCE_CACHE.get()
    key = ("scan", str(base), expected_job_id, expected_generation)
    if cache is not None and key in cache:
        return
    if not base.exists():
        raise CTResumeRefused(f"missing retained evidence scan root: {base}")
    if not base.is_dir():
        raise CTResumeRefused(f"retained evidence root is not a directory: {base}")
    retained_receipt = None
    receipt_path = base / "receipt.json"
    if receipt_path.is_file():
        _no_symlink_path(receipt_path)
        if receipt_path.stat().st_size > 8 * 1024 * 1024:
            raise CTResumeRefused(f"oversized retained receipt: {receipt_path}")
        retained_receipt = _load_json(receipt_path.read_text())
        _check_retained_receipt(retained_receipt, label=str(receipt_path))
        bound_job = (cache or {}).get("bound_output_jobs", {}).get(str(base))
        expected_job_id = expected_job_id or bound_job
        if expected_job_id is not None and retained_receipt.get("job_id") != expected_job_id:
            raise CTResumeRefused("retained receipt belongs to foreign job_id")
        expected_generation = expected_generation or retained_receipt.get("generation")
    retained_files = _retained_files(base)
    for history in (p for p in retained_files if p.name == "progress-history.log"):
        text = _scan_diagnostic_bytes(history)
        _scan_job_tail(text, label=str(history), generation=expected_generation)
        if cache is not None:
            cache.setdefault("progress_history_lines", set()).update(text.splitlines())
    for path in retained_files:
        if _publication_sidecar(path):
            continue  # Sidecar bytes are not the payload's diagnostic content.
        name = path.name.lower().lstrip(".")
        if name.startswith("summary.json"):
            raise CTResumeRefused(f"scientific summary already present: {path}")
        if name.startswith("fault.json"):
            raise CTResumeRefused(f"scientific/runtime fault diagnostic present: {path}")
        if name.startswith("faulthandler-stacks.log") and path.lstat().st_size:
            raise CTResumeRefused(f"non-empty faulthandler stack dump: {path}")
        _scan_retained_text(
            path,
            output_root=base,
            expected_job_id=expected_job_id,
            expected_generation=expected_generation,
            supervisor_receipt=retained_receipt,
            supervisor_receipt_path=receipt_path,
        )
    if cache is not None:
        cache[key] = True


def _candidate_search_roots(export_root: Path) -> list[Path]:
    bound = {
        os.path.normpath(raw)
        for raw in (_EVIDENCE_CACHE.get() or {}).get("bound_output_dirs", set())
    }
    normalized = os.path.normpath(str(export_root))
    parents = [Path(raw) for raw in bound if os.path.commonpath([raw, normalized]) == raw]
    if not parents:
        raise CTResumeRefused(f"unbound retained evidence scan root: {export_root}")
    # Always scan the entire bound attempt, independent of the export basename.
    return list(dict.fromkeys([min(parents, key=lambda p: len(p.parts)), export_root]))


def _check_attempt_output_dir_fault_evidence(output_dir: Path) -> None:
    """Scan an attempt's ``output_dir`` and its export subdirectories without climbing to parent."""
    if output_dir.is_symlink():
        raise CTResumeRefused(f"prior output_dir must not be a symlink: {output_dir}")
    _check_attempt_base_fault_evidence(output_dir)


def _check_single_root_fault_evidence(export_root: Path) -> None:
    if export_root.is_symlink():
        raise CTResumeRefused(f"export_root must not be a symlink: {export_root}")
    for base in _candidate_search_roots(export_root):
        _check_attempt_base_fault_evidence(base)


def assert_no_scientific_or_fault_precedence(
    export_root: Path | Sequence[Path], receipt: Mapping[str, Any]
) -> None:
    """Refuse mechanical continuation if any scientific outcome or fault exists."""
    for root in _normalize_export_roots(export_root):
        _check_single_root_fault_evidence(root)

    _scan_json_faults(
        {k: v for k, v in receipt.items() if k != "error"},
        label="receipt",
        expected_job_id=receipt.get("job_id"),
        expected_generation=receipt.get("generation"),
        supervisor_transport=True,
    )
    _refuse_if_fault_text(
        receipt.get("error"), field_name="receipt.error", check_supervisor_denylist=True
    )
    _refuse_if_fault_exports(receipt.get("exports"), source_name="receipt")
    terminal = receipt.get("terminal")
    if isinstance(terminal, Mapping):
        _check_terminal_fault_evidence(terminal, label="terminal")


def _scan_supervisor_log(
    path: Path,
    *,
    job_id: str | None,
    generation: str | None,
    receipt: Mapping[str, Any] | None = None,
    receipt_path: Path | None = None,
    fixture_completion: bool = False,
) -> None:
    """Apply execution validation to every structured host-log record, including nested facts."""
    _no_symlink_path(path)
    if not path.is_file():
        raise CTResumeRefused(f"missing or symlinked supervisor log: {path}")
    _scan_diagnostic_bytes(path, json_mode=True, collect=False)
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            stripped = line.strip()
            label = f"{path}:{index}"
            if stripped.startswith("INCOMPLETE:"):
                # argparse.main emits this final line after Broker.run has
                # published a receipt. Admit only that entire bound cause;
                # a generic log substring whitelist would conceal appended faults.
                bound_path = receipt_path or path.parent / "receipt.json"
                _no_symlink_path(bound_path)
                bound = receipt
                if bound is None and bound_path.is_file():
                    bound = _load_json(bound_path.read_text())
                if not isinstance(bound, Mapping):
                    raise CTResumeRefused(f"supervisor CLI footer lacks bound receipt: {label}")
                _check_retained_receipt(bound, label=label)
                _scan_json_faults(
                    bound,
                    label=label,
                    expected_job_id=job_id,
                    expected_generation=generation,
                    supervisor_transport=True,
                )
                expected = f"INCOMPLETE: {bound.get('error')}; receipt: {bound_path}"
                if stripped != expected:
                    _refuse_if_supervisor_log_fault(stripped, field_name=label)
                    raise CTResumeRefused(
                        f"supervisor CLI footer differs from bound receipt: {label}"
                    )
                continue
            if stripped.startswith("{"):
                try:
                    record = _load_json(stripped)
                except ValueError as error:
                    raise CTResumeRefused(f"corrupt supervisor JSON record: {label}") from error
                _scan_json_faults(
                    record,
                    label=label,
                    expected_job_id=job_id,
                    expected_generation=generation,
                    supervisor_transport=True,
                    supervisor_log=True,
                    fixture_completion=fixture_completion,
                )
                for field in ("state", "terminal"):
                    item = record.get(field)
                    if field == "terminal" and type(item) is bool:
                        continue  # Broker stdout records presence, not terminal payload.
                    if item is not None and not (field == "terminal" and item is False):
                        if not isinstance(item, Mapping):
                            raise CTResumeRefused(f"malformed supervisor {field}: {label}")
                _refuse_if_fault_exports(record.get("exports"), source_name=label)
            else:
                _refuse_if_supervisor_log_fault(line, field_name=label)


@_evidence_session
def verify_mechanical_stop(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    *,
    export_root: Path | Sequence[Path],
    supervisor_log_path: Path | None = None,
) -> None:
    """Enforce scientific-fault precedence first, then verify clean qualified mechanical stop."""
    cache = _EVIDENCE_CACHE.get()
    if cache is not None and isinstance(manifest.get("output_dir"), str):
        cache.setdefault("bound_output_dirs", set()).add(manifest["output_dir"])
        cache.setdefault("bound_output_jobs", {})[manifest["output_dir"]] = manifest.get("job_id")
    if receipt.get("job_id") != manifest.get("job_id"):
        raise CTResumeRefused("prior receipt belongs to foreign job_id")
    _check_retained_receipt(receipt, label="receipt")
    assert_no_scientific_or_fault_precedence(export_root, receipt)
    if supervisor_log_path is not None:
        _scan_supervisor_log(
            supervisor_log_path,
            job_id=receipt.get("job_id"),
            generation=receipt.get("generation"),
            receipt=receipt,
            receipt_path=(
                Path(manifest["output_dir"]) / "receipt.json"
                if isinstance(manifest.get("output_dir"), str)
                else None
            ),
        )

    if not _is_hex(manifest_sha256, 64):
        raise CTResumeRefused("original manifest_sha256 is malformed")
    if (
        receipt.get("job_id") != manifest.get("job_id")
        or receipt.get("commit") != manifest.get("commit")
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("status") != "INCOMPLETE"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
    ):
        raise CTResumeRefused("prior receipt does not prove a verified released incomplete run")

    terminal = receipt.get("terminal")
    terminal_reason: str | None = None
    if terminal is not None:
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("job_id") != manifest.get("job_id")
            or terminal.get("generation") != receipt.get("generation")
        ):
            raise CTResumeRefused("terminal marker identity differs from receipt")
        raw_reason = terminal.get("reason")
        terminal_reason = raw_reason.lower() if isinstance(raw_reason, str) else None
        if (
            terminal.get("returncode") not in (None, 0)
            and terminal_reason not in MECHANICAL_TERMINAL_REASONS
        ):
            raise CTResumeRefused("terminal records a non-mechanical child failure")

    failure_class = receipt.get("failure_class")
    if failure_class == "RPC_EXEC_POLL_TIMEOUT":
        if terminal is not None and terminal_reason not in MECHANICAL_TERMINAL_REASONS:
            raise CTResumeRefused("RPC poll timeout receipt carries a non-mechanical terminal")
        return
    if failure_class == "CHILD_NONZERO_EXIT":
        if isinstance(terminal, Mapping) and terminal_reason in MECHANICAL_TERMINAL_REASONS:
            return
        raise CTResumeRefused(
            "CHILD_NONZERO_EXIT receipt lacks a mechanical timeout/lease terminal reason"
        )
    if failure_class == "SUPERVISOR":
        if isinstance(terminal, Mapping) and terminal_reason in MECHANICAL_TERMINAL_REASONS:
            return
        error = receipt.get("error")
        if terminal is None and isinstance(error, str):
            if _mechanical_supervisor_error(error):
                return
        raise CTResumeRefused(
            "SUPERVISOR receipt does not prove a mechanical wall/unit/lease/poll stop"
        )
    raise CTResumeRefused(
        f"prior receipt failure_class {failure_class!r} is not a qualified mechanical trigger"
    )


def _stage_row(
    export_root: Path, info_path: Path, control: Mapping[str, Any]
) -> tuple[int, dict[str, Any], list[dict[str, str]]] | None:
    if not _complete(info_path):
        return None
    _descriptor(info_path, runtime_path=f"{RUNTIME_ROOT}/{info_path.name}")
    try:
        info = _load_json(info_path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise CTResumeRefused(f"unreadable stage header: {info_path.name}") from err

    stage = info.get("stage")
    identity = info.get("id")
    if type(stage) is not int or stage < 1 or identity != unit_id(stage):
        raise CTResumeRefused(f"foreign stage identity: {info_path.name}")
    if info.get("schema") != SNAPSHOT_SCHEMA or not isinstance(info.get("metadata"), dict):
        raise CTResumeRefused(f"malformed stage header: {info_path.name}")
    meta = info["metadata"]
    if meta.get("allocation") != allocation(control) or meta.get("source_commit") != control.get(
        "source_commit"
    ):
        raise CTResumeRefused(f"stage header differs from frozen control: {info_path.name}")
    if meta.get("scientific_fault") or meta.get("support_status") in {
        "INCOMPLETE_SUPPORT",
        "INVALID",
    }:
        raise CTResumeRefused(
            f"stage-{stage:04d} recorded terminal support/fault status "
            f"{meta.get('support_status') or meta.get('scientific_fault')}; continuation refused"
        )

    stem = f"stage-{stage:04d}"
    if info_path.name != f"{stem}.json":
        raise CTResumeRefused(f"noncanonical stage header filename: {info_path.name}")
    pt_path = export_root / f"{stem}.pt"
    if not _complete(pt_path):
        return None

    paths = [pt_path, info_path]
    runtime_paths = [f"{RUNTIME_ROOT}/{p.name}" for p in paths]
    descriptors = [_descriptor(p, runtime_path=dest) for p, dest in zip(paths, runtime_paths)]
    snap_meta = info.get("snapshot")
    if (
        not isinstance(snap_meta, dict)
        or snap_meta.get("id") != identity
        or snap_meta.get("size") != descriptors[0]["size"]
        or snap_meta.get("sha256") != descriptors[0]["sha256"]
    ):
        raise CTResumeRefused(f"stage header/payload descriptor mismatch: {info_path.name}")

    # Upload only the hash-bound stage payload + JSON header (2 files per raw stage);
    # Mac completion sidecars were verified above by _descriptor and are re-checked by
    # ct_preflight, keeping continuation RPC file counts low under the 300 s staging cap.
    uploads: list[dict[str, str]] = [
        {"path": str(p.resolve()), "dest": dest, "sha256": desc["sha256"]}
        for p, dest, desc in zip(paths, runtime_paths, descriptors)
    ]
    return stage, {"id": identity, "stage": stage, "exports": descriptors}, uploads


def build_stage_ledger(
    export_root: Path | Sequence[Path], control: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]], tuple[int, ...]]:
    """Build duplicate-checked cross-attempt union ledger over all verified stages."""
    roots = _normalize_export_roots(export_root)
    rows_by_stage: dict[int, dict[str, Any]] = {}
    uploads_by_stage: dict[int, list[dict[str, str]]] = {}

    for root in roots:
        if not root.is_dir():
            raise CTResumeRefused(f"missing prior export directory: {root}")
        for info_path in sorted(root.glob("stage-*.json")):
            cand = _stage_row(root, info_path, control)
            if cand is None:
                continue
            stage, row, items = cand
            if stage in rows_by_stage:
                raise CTResumeRefused(
                    f"duplicate verified stage-{stage:04d} across attempt export directories"
                )
            rows_by_stage[stage] = row
            uploads_by_stage[stage] = items

    if not rows_by_stage:
        raise CTResumeRefused("no complete verified stage snapshots found")
    stages = tuple(sorted(rows_by_stage.keys()))
    if 1 not in stages:
        raise CTResumeRefused("cannot resume scoring/fitting without verified stage-0001")
    total_stages = 2 + int(control.get("scoring_chunks", 8))
    if any(s < 1 or s > total_stages for s in stages):
        raise CTResumeRefused("stage index outside frozen C/T stage schedule")
    if total_stages in stages:
        raise CTResumeRefused("final stage already completed; continuation unnecessary")

    rows = [rows_by_stage[s] for s in stages]
    uploads: list[dict[str, str]] = []
    for s in stages:
        uploads.extend(uploads_by_stage[s])
    ledger = {
        "schema": LEDGER_SCHEMA,
        "namespace": NAMESPACE,
        "allocation": allocation(control),
        "units": rows,
    }
    return ledger, uploads, stages


def _locate_stage_owner_dir(
    stem: str, uploads: Sequence[Mapping[str, str]], roots: Sequence[Path]
) -> Path:
    filename = stem if stem.endswith(".pt") else f"{stem}.pt"
    target_dest = f"{RUNTIME_ROOT}/{filename}"
    for rec in uploads:
        if rec.get("dest") == target_dest:
            return Path(rec["path"]).resolve().parent
    for root in roots:
        if (root / filename).is_file() or (
            root / f"{filename}{COMPACT_DERIVATIVE_SUFFIX}"
        ).is_file():
            return root.resolve()
    raise CTResumeRefused(f"unable to locate owning export directory for {filename}")


def attach_compact_stage_resume(
    ledger: Mapping[str, Any],
    uploads: list[dict[str, str]],
    export_root: Path | Sequence[Path],
    *,
    stages_to_compact: Sequence[int] | None = None,
    auto_encode_missing: bool = False,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Replace selected stage ``.pt`` uploads with verified ``.pt.xz`` + ``.pt.xz.manifest``."""
    roots = _normalize_export_roots(export_root)
    successor = copy.deepcopy(dict(ledger))
    units = successor.get("units", [])
    if not isinstance(units, list) or not units:
        raise CTResumeRefused("compact resume requires at least one verified stage unit")

    if stages_to_compact is not None:
        requested = {int(s) for s in stages_to_compact}
    else:
        requested = set()
        for u in units:
            stem = Path(u["exports"][0]["path"]).name
            owner = _locate_stage_owner_dir(stem, uploads, roots)
            if _complete(owner / f"{stem}{COMPACT_DERIVATIVE_SUFFIX}"):
                requested.add(int(u["stage"]))
    if not requested:
        raise CTResumeRefused("no stages selected or encoded for compact resume")

    available_stages = {int(u["stage"]) for u in units}
    missing_requested = requested - available_stages
    if missing_requested:
        raise CTResumeRefused(
            f"requested compact stages absent from ledger: {sorted(missing_requested)}"
        )

    entries: list[dict[str, Any]] = []
    updated_uploads = list(uploads)
    for unit in units:
        stage_num = int(unit["stage"])
        if stage_num not in requested:
            continue
        orig = dict(unit["exports"][0])
        stem = Path(orig["path"]).name
        owner_dir = _locate_stage_owner_dir(stem, updated_uploads, roots)
        deriv_file = owner_dir / f"{stem}{COMPACT_DERIVATIVE_SUFFIX}"
        man_file = owner_dir / f"{stem}{COMPACT_MANIFEST_SUFFIX}"
        if auto_encode_missing and not (_complete(deriv_file) and _complete(man_file)):
            encode_compact_stage_snapshot(owner_dir / stem, unit=str(unit["id"]))

        entry: dict[str, Any] = {
            "unit": unit["id"],
            "original": orig,
        }
        replacement_records: list[dict[str, str]] = []
        for role, suffix in (
            ("derivative", COMPACT_DERIVATIVE_SUFFIX),
            ("manifest", COMPACT_MANIFEST_SUFFIX),
        ):
            p = owner_dir / f"{stem}{suffix}"
            dest = f"{orig['path']}{suffix}"
            entry[role] = _descriptor(p, runtime_path=dest)
            replacement_records.append(
                {"path": str(p.resolve()), "dest": dest, "sha256": entry[role]["sha256"]}
            )
        entries.append(entry)
        positions = [i for i, rec in enumerate(updated_uploads) if rec["dest"] == orig["path"]]
        if len(positions) != 1:
            raise CTResumeRefused(f"missing or duplicate upload record for {orig['path']}")
        pos = positions[0]
        updated_uploads = updated_uploads[:pos] + replacement_records + updated_uploads[pos + 1 :]

    successor[COMPACT_RESUME_FIELD] = {
        "schema": COMPACT_RESUME_SCHEMA,
        "entries": entries,
    }
    validate_compact_resume(successor)
    return successor, updated_uploads


def _canonical_ledger_bytes(ledger: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(ledger), indent=2, sort_keys=True) + "\n").encode("utf-8")


def project_staging_envelope(
    inputs: Sequence[Mapping[str, str]],
    *,
    ledger_size_bytes: int,
    source_archive_bytes: int = 1_100_000,
) -> dict[str, float | int]:
    """Compute conservative whole-input staging bytes and wall time (colab-limits §4.1)."""
    resume_bytes = 0
    fixed_bytes = 0
    for row in inputs:
        dest = str(row.get("dest", ""))
        if dest == "verified-ledger.json":
            sz = int(ledger_size_bytes)
        else:
            p = Path(row["path"])
            if p.is_symlink() or not p.is_file():
                raise CTResumeRefused(f"missing or symlinked staged input: {p}")
            sz = int(p.stat().st_size)
        if dest.startswith("resume/"):
            resume_bytes += sz
        else:
            fixed_bytes += sz

    if resume_bytes > MAX_CONTINUATION_PRIVATE_INPUT_BYTES:
        raise CTResumeRefused(
            f"continuation resume payload {resume_bytes} B exceeds "
            f"MAX_CONTINUATION_PRIVATE_INPUT_BYTES={MAX_CONTINUATION_PRIVATE_INPUT_BYTES} B; "
            "use compact .pt.xz stage encoding"
        )
    # Count private inputs + bootstrap.sh + source.py + source.tar.gz (N_priv + 3)
    n_up = len(inputs) + 3
    b_up = fixed_bytes + resume_bytes + int(source_archive_bytes) + 16_384
    t_worst = (
        _WORST_MKDIR_PLUS_SPARSE_INSTALL_SEC
        + (float(b_up) / _WORST_UPLOAD_BYTES_PER_SEC)
        + (_WORST_PER_FILE_OVERHEAD_SEC * float(n_up))
    )
    if t_worst >= STAGING_HARD_CAP_SECONDS:
        raise CTResumeRefused(
            f"projected worst-case initial staging {t_worst:.2f} s exceeds "
            f"{STAGING_HARD_CAP_SECONDS:.1f} s UPLOAD_TOTAL_TIMEOUT_SECONDS cap"
        )
    return {
        "num_private_inputs": len(inputs),
        "num_total_uploads": n_up,
        "fixed_input_bytes": fixed_bytes,
        "resume_input_bytes": resume_bytes,
        "total_upload_bytes": b_up,
        "projected_worst_staging_seconds": round(t_worst, 3),
    }


def _append_unique_dir_str(target: list[str], directory: Path) -> None:
    _no_symlink_path(directory)
    if directory.is_symlink():
        raise CTResumeRefused(f"provenance directory must not be a symlink: {directory}")
    resolved_str = str(directory.resolve())
    if resolved_str not in target:
        target.append(resolved_str)


def _valid_ledger_export_desc(desc: Any, *, expected_runtime_path: str) -> bool:
    return (
        isinstance(desc, Mapping)
        and desc.get("path") == expected_runtime_path
        and type(desc.get("size")) is int
        and desc["size"] >= 1
        and _is_hex(desc.get("sha256"), 64)
    )


def _extract_prior_admitted_coverage(
    original_manifest: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    completion_index: int,
    prior_packet_dir: Path | None = None,
) -> tuple[dict[int, dict[str, Any]], str | None, list[str], list[str]]:
    """Validate prior manifest/ledger lineage and extract admitted units + provenance dirs."""
    prior_attempt = original_manifest.get("completion_attempt")
    if completion_index == 1:
        if prior_attempt is not None:
            raise CTResumeRefused(
                "completion_index=1 requires an initial manifest without completion_attempt"
            )
    elif completion_index == 2:
        if not isinstance(prior_attempt, Mapping) or prior_attempt.get("index") != 1:
            raise CTResumeRefused(
                "completion_index=2 requires a prior completion_attempt=1 manifest"
            )
    else:
        raise CTResumeRefused("only two manual outcome-blind completion attempts are permitted")

    prior_output_dirs: list[str] = []
    prior_export_roots: list[str] = []
    if isinstance(prior_attempt, Mapping):
        raw_p_outs = prior_attempt.get("prior_output_dirs", [])
        if not isinstance(raw_p_outs, list) or any(
            not isinstance(x, str) or not x.strip() for x in raw_p_outs
        ):
            raise CTResumeRefused("malformed completion_attempt.prior_output_dirs")
        for item in raw_p_outs:
            _append_unique_dir_str(prior_output_dirs, Path(item))

        raw_p_exps = prior_attempt.get("prior_export_roots", [])
        if not isinstance(raw_p_exps, list) or any(
            not isinstance(x, str) or not x.strip() for x in raw_p_exps
        ):
            raise CTResumeRefused("malformed completion_attempt.prior_export_roots")
        for item in raw_p_exps:
            _append_unique_dir_str(prior_export_roots, Path(item))

    raw_orig_out = original_manifest.get("output_dir")
    if raw_orig_out is not None:
        if not isinstance(raw_orig_out, str) or not raw_orig_out.strip():
            raise CTResumeRefused("original_manifest.output_dir is malformed")
        orig_out_path = Path(raw_orig_out)
        if not orig_out_path.is_absolute():
            raise CTResumeRefused("original_manifest.output_dir must be absolute")
        _append_unique_dir_str(prior_output_dirs, orig_out_path)
        _append_unique_dir_str(prior_export_roots, orig_out_path / "exports/ct")

    raw_inputs = original_manifest.get("inputs", [])
    if not isinstance(raw_inputs, list) or any(not isinstance(r, Mapping) for r in raw_inputs):
        raise CTResumeRefused("original_manifest.inputs must be a list of input records")
    prior_uploads: dict[str, Mapping[str, Any]] = {}
    for row in raw_inputs:
        dest = row.get("dest")
        if not isinstance(dest, str) or not dest:
            raise CTResumeRefused("original_manifest input lacks destination")
        if dest in prior_uploads:
            raise CTResumeRefused(f"original_manifest duplicates input destination: {dest}")
        prior_uploads[dest] = row

    ledger_entry = prior_uploads.get("verified-ledger.json")
    prior_ledger: Mapping[str, Any] | None = None
    prior_ledger_sha256: str | None = None
    prior_compact: dict[str, dict[str, Any]] = {}

    if completion_index == 2 or prior_packet_dir is not None:
        if ledger_entry is None:
            raise CTResumeRefused("prior manifest lacks bound verified-ledger.json input")

    if ledger_entry is not None:
        expected_lsha = ledger_entry.get("sha256")
        prior_ledger_file = Path(str(ledger_entry.get("path", "")))
        _no_symlink_path(prior_ledger_file)
        if (
            not _is_hex(expected_lsha, 64)
            or prior_ledger_file.is_symlink()
            or not prior_ledger_file.is_file()
            or _retained_digest(prior_ledger_file) != expected_lsha
        ):
            raise CTResumeRefused(
                "prior verified-ledger.json is missing, symlinked, or hash-mismatched"
            )
        if prior_packet_dir is not None:
            pkt_ledger = prior_packet_dir / "verified-ledger.json"
            if (
                pkt_ledger.is_symlink()
                or not pkt_ledger.is_file()
                or _retained_digest(pkt_ledger) != expected_lsha
            ):
                raise CTResumeRefused(
                    "prior packet verified-ledger.json differs from manifest binding"
                )
        prior_ledger_sha256 = str(expected_lsha)
        try:
            loaded_ledger = _load_json(prior_ledger_file.read_text())
        except (OSError, json.JSONDecodeError) as err:
            raise CTResumeRefused("corrupt prior verified-ledger.json") from err
        if not isinstance(loaded_ledger, Mapping):
            raise CTResumeRefused("malformed prior verified-ledger.json")
        if (
            loaded_ledger.get("schema") != LEDGER_SCHEMA
            or loaded_ledger.get("namespace") != NAMESPACE
            or loaded_ledger.get("allocation") != allocation(control)
            or not isinstance(loaded_ledger.get("units"), list)
        ):
            raise CTResumeRefused("prior verified-ledger.json differs from frozen C/T allocation")
        try:
            prior_compact = validate_compact_resume(loaded_ledger)
        except ValueError as err:
            raise CTResumeRefused(f"invalid prior compact_resume block: {err}") from err
        prior_ledger = loaded_ledger

    if completion_index == 1:
        if prior_ledger is not None and prior_ledger["units"]:
            raise CTResumeRefused("initial manifest verified-ledger.json must have empty units")
        if any(dest.startswith("resume/") for dest in prior_uploads):
            raise CTResumeRefused("initial manifest must not carry resume/* inputs")
        return {}, prior_ledger_sha256, prior_output_dirs, prior_export_roots

    assert isinstance(prior_attempt, Mapping)
    assert prior_ledger is not None
    bound_lsha = prior_attempt.get("ledger_sha256")
    if bound_lsha is not None and bound_lsha != prior_ledger_sha256:
        raise CTResumeRefused(
            "prior completion_attempt.ledger_sha256 disagrees with bound verified-ledger.json"
        )

    declared_stages = prior_attempt.get("verified_stages")
    if (
        not isinstance(declared_stages, list)
        or not declared_stages
        or any(type(s) is not int or s < 1 for s in declared_stages)
    ):
        raise CTResumeRefused(
            "prior completion_attempt.verified_stages must be a non-empty int list"
        )

    raw_units = prior_ledger["units"]
    if not raw_units:
        raise CTResumeRefused("prior completion_attempt=1 ledger has no verified stage units")

    prior_units_by_stage: dict[int, dict[str, Any]] = {}
    ordered_stages: list[int] = []
    for u in raw_units:
        if not isinstance(u, Mapping):
            raise CTResumeRefused("malformed stage unit in prior verified-ledger.json")
        stage = u.get("stage")
        u_id = u.get("id")
        if (
            type(stage) is not int
            or stage < 1
            or u_id != unit_id(stage)
            or stage in prior_units_by_stage
        ):
            raise CTResumeRefused("invalid or duplicate stage unit in prior verified-ledger.json")
        exports = u.get("exports")
        expected_paths = (
            f"{RUNTIME_ROOT}/stage-{stage:04d}.pt",
            f"{RUNTIME_ROOT}/stage-{stage:04d}.json",
        )
        if (
            not isinstance(exports, list)
            or len(exports) != 2
            or not _valid_ledger_export_desc(exports[0], expected_runtime_path=expected_paths[0])
            or not _valid_ledger_export_desc(exports[1], expected_runtime_path=expected_paths[1])
        ):
            raise CTResumeRefused(f"malformed export descriptors in prior stage-{stage:04d}")
        prior_units_by_stage[stage] = {
            "id": str(u_id),
            "stage": int(stage),
            "exports": [
                {
                    "path": str(exp["path"]),
                    "size": int(exp["size"]),
                    "sha256": str(exp["sha256"]),
                }
                for exp in exports
            ],
        }
        ordered_stages.append(stage)

    if ordered_stages != sorted(prior_units_by_stage) or ordered_stages != list(declared_stages):
        raise CTResumeRefused(
            "prior completion_attempt.verified_stages disagrees with bound verified-ledger.json"
        )

    referenced_resume: set[str] = set()
    for stage, unit_row in sorted(prior_units_by_stage.items()):
        c_entry = prior_compact.get(unit_row["id"])
        for exp in unit_row["exports"]:
            dest = exp["path"]
            if c_entry is not None and dest == c_entry["original"]["path"]:
                if dest in prior_uploads:
                    raise CTResumeRefused(
                        "prior compact resume manifest also uploaded raw stage .pt"
                    )
                orig_candidates: list[Path] = []
                for role, suffix in (
                    ("derivative", COMPACT_DERIVATIVE_SUFFIX),
                    ("manifest", COMPACT_MANIFEST_SUFFIX),
                ):
                    item = prior_uploads.get(dest + suffix)
                    bound = c_entry[role]
                    if (
                        item is None
                        or bound["path"] != dest + suffix
                        or item.get("sha256") != bound["sha256"]
                        or not str(item.get("path", "")).endswith(suffix)
                    ):
                        raise CTResumeRefused(
                            f"prior compact resume {role} upload binding mismatch"
                        )
                    item_path = Path(str(item["path"]))
                    _no_symlink_path(item_path)
                    if (
                        item_path.is_symlink()
                        or not item_path.is_file()
                        or item_path.stat().st_size != bound["size"]
                        or _retained_digest(item_path) != bound["sha256"]
                    ):
                        raise CTResumeRefused(
                            f"prior compact resume {role} file missing or corrupt"
                        )
                    orig_candidates.append(Path(str(item_path)[: -len(suffix)]))
                    referenced_resume.add(dest + suffix)
                if orig_candidates[0] != orig_candidates[1]:
                    raise CTResumeRefused("prior compact derivative and manifest are not siblings")
                orig_src = orig_candidates[0]
                for sc_suffix in (".sha256", ".complete"):
                    sc_file = Path(str(orig_src) + sc_suffix)
                    if sc_file.is_symlink() or not sc_file.is_file():
                        raise CTResumeRefused("prior compact stage original lacks Mac sidecars")
                    if (
                        sc_suffix == ".sha256"
                        and sc_file.read_text().strip().split()[0] != exp["sha256"]
                    ):
                        raise CTResumeRefused("prior compact stage original .sha256 mismatch")
                    sc_up = prior_uploads.get(dest + sc_suffix)
                    if sc_up is not None:
                        if Path(str(sc_up.get("path", ""))) != sc_file or _retained_digest(
                            sc_file
                        ) != sc_up.get("sha256"):
                            raise CTResumeRefused("prior uploaded stage sidecar mismatch")
                        referenced_resume.add(dest + sc_suffix)
                _append_unique_dir_str(prior_export_roots, orig_src.resolve().parent)
                continue

            item = prior_uploads.get(dest)
            if item is None or item.get("sha256") != exp["sha256"]:
                raise CTResumeRefused(
                    f"prior stage-{stage:04d} export {dest} lacks hash-bound upload"
                )
            src = Path(str(item.get("path", "")))
            _no_symlink_path(src)
            if (
                src.is_symlink()
                or not src.is_file()
                or src.stat().st_size != exp["size"]
                or _retained_digest(src) != exp["sha256"]
            ):
                raise CTResumeRefused(f"prior stage-{stage:04d} source file missing or corrupt")
            referenced_resume.add(dest)
            for sc_suffix in (".sha256", ".complete"):
                sc_file = Path(str(src) + sc_suffix)
                if sc_file.is_symlink() or not sc_file.is_file():
                    raise CTResumeRefused(f"prior stage-{stage:04d} source lacks Mac sidecars")
                if (
                    sc_suffix == ".sha256"
                    and sc_file.read_text().strip().split()[0] != exp["sha256"]
                ):
                    raise CTResumeRefused(f"prior stage-{stage:04d} .sha256 sidecar mismatch")
                sc_up = prior_uploads.get(dest + sc_suffix)
                if sc_up is not None:
                    if Path(str(sc_up.get("path", ""))) != sc_file or _retained_digest(
                        sc_file
                    ) != sc_up.get("sha256"):
                        raise CTResumeRefused("prior uploaded stage sidecar mismatch")
                    referenced_resume.add(dest + sc_suffix)
            _append_unique_dir_str(prior_export_roots, src.resolve().parent)

    if referenced_resume != {d for d in prior_uploads if d.startswith("resume/")}:
        raise CTResumeRefused("prior manifest carries unreferenced resume/* upload")

    return prior_units_by_stage, prior_ledger_sha256, prior_output_dirs, prior_export_roots


@_evidence_session
def completion_packet(
    original_manifest: Mapping[str, Any],
    terminal_receipt: Mapping[str, Any],
    export_root: Path | Sequence[Path],
    control: Mapping[str, Any],
    *,
    completion_index: int,
    job_id: str,
    output_dir: Path,
    original_manifest_sha256: str,
    use_compact_xz_for_stages: Sequence[int] | None = None,
    auto_encode_compact_xz: bool = False,
    packet_dir: Path | None = None,
    ledger_path: Path | None = None,
    sandbox_manifest_path: Path | None = None,
    supervisor_log_path: Path | None = None,
    prior_packet_dir: Path | None = None,
    prior_manifest_path: Path | None = None,
) -> CTResumePacket:
    """Build a complete, admitted manual outcome-blind completion attempt packet for C/T."""
    if completion_index not in (1, 2):
        raise CTResumeRefused("only two manual outcome-blind completion attempts are permitted")
    if original_manifest.get("commit") != control.get("source_commit"):
        raise CTResumeRefused("manifest commit differs from frozen control source_commit")

    (
        prior_units_by_stage,
        prior_ledger_sha256,
        prior_output_dirs,
        prior_export_roots,
    ) = _extract_prior_admitted_coverage(
        original_manifest,
        control,
        completion_index=completion_index,
        prior_packet_dir=prior_packet_dir,
    )

    cache = _EVIDENCE_CACHE.get()
    if cache is not None:
        cache.setdefault("bound_output_dirs", set()).update(prior_output_dirs)

    # Inspect all bound prior attempt output/export directories for any scientific or runtime fault
    # before admitting mechanical continuation, even if an earlier export dir was omitted by caller.
    for out_dir_str in prior_output_dirs:
        _check_attempt_output_dir_fault_evidence(Path(out_dir_str))
    for exp_dir_str in prior_export_roots:
        _check_single_root_fault_evidence(Path(exp_dir_str))

    verify_mechanical_stop(
        terminal_receipt,
        original_manifest,
        original_manifest_sha256,
        export_root=export_root,
        supervisor_log_path=supervisor_log_path,
    )
    inherited_roots = set()
    prior_attempt = original_manifest.get("completion_attempt")
    if isinstance(prior_attempt, Mapping):
        inherited_roots.update(prior_attempt.get("prior_output_dirs", []))
        inherited_roots.update(prior_attempt.get("prior_export_roots", []))
    current_roots = [Path(r) for r in prior_output_dirs if r not in inherited_roots]
    current_roots.extend(
        root
        for root in _normalize_export_roots(export_root)
        if str(root.resolve()) not in inherited_roots
    )
    for root in current_roots:
        _check_attempt_base_fault_evidence(
            root,
            expected_job_id=str(original_manifest["job_id"]),
            expected_generation=terminal_receipt.get("generation"),
        )
    if not job_id or job_id == original_manifest.get("job_id"):
        raise CTResumeRefused("completion attempt requires a fresh job_id")
    _no_symlink_path(output_dir)
    if not output_dir.is_absolute() or output_dir.is_symlink():
        raise CTResumeRefused("completion attempt requires a fresh absolute output_dir")
    resolved_new_output = str(output_dir.resolve())
    if resolved_new_output in set(prior_output_dirs):
        raise CTResumeRefused("completion attempt requires a fresh absolute output_dir")

    ledger, records, stages = build_stage_ledger(export_root, control)
    new_units_by_stage = {int(u["stage"]): u for u in ledger["units"]}
    omitted_prior = [s for s in sorted(prior_units_by_stage) if s not in new_units_by_stage]
    if omitted_prior:
        raise CTResumeRefused(
            f"successor packet omits previously verified stage(s) {omitted_prior}; "
            f"admitted coverage {sorted(prior_units_by_stage)} cannot be discarded or recomputed"
        )
    for s in sorted(prior_units_by_stage):
        if new_units_by_stage[s] != prior_units_by_stage[s]:
            raise CTResumeRefused(
                f"successor stage-{s:04d} conflicts with previously verified coverage; "
                "recomputing or replacing admitted units is forbidden"
            )

    cumulative_export_roots = list(prior_export_roots)
    for root in _normalize_export_roots(export_root):
        _append_unique_dir_str(cumulative_export_roots, root)

    if use_compact_xz_for_stages is not None:
        ledger, records = attach_compact_stage_resume(
            ledger,
            records,
            export_root,
            stages_to_compact=use_compact_xz_for_stages,
            auto_encode_missing=auto_encode_compact_xz,
        )

    ledger_bytes = _canonical_ledger_bytes(ledger)
    ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    resolved_ledger_path = (
        (packet_dir / "verified-ledger.json").resolve()
        if packet_dir is not None
        else (
            ledger_path.resolve()
            if ledger_path is not None
            else (output_dir.parent / f"{job_id}-verified-ledger.json").resolve()
        )
    )
    resolved_sandbox_path = (
        sandbox_manifest_path.resolve()
        if sandbox_manifest_path is not None
        else (
            (packet_dir / "sandbox.json").resolve()
            if packet_dir is not None
            else (output_dir.parent / f"{job_id}-sandbox.json").resolve()
        )
    )

    base_inputs = [
        row
        for row in copy.deepcopy(list(original_manifest.get("inputs", [])))
        if row["dest"] != "verified-ledger.json" and not str(row["dest"]).startswith("resume/")
    ]
    ledger_record = {
        "path": str(resolved_ledger_path),
        "dest": "verified-ledger.json",
        "sha256": ledger_sha,
    }
    all_inputs = base_inputs + [ledger_record] + records
    staging_proj = project_staging_envelope(all_inputs, ledger_size_bytes=len(ledger_bytes))

    prior_sorted_stages = sorted(prior_units_by_stage)
    completion_meta: dict[str, Any] = {
        "index": int(completion_index),
        "manual": True,
        "prior_job_id": str(original_manifest["job_id"]),
        "prior_manifest_sha256": str(original_manifest_sha256),
        "verified_stages": list(stages),
        "ledger_sha256": ledger_sha,
        "prior_verified_stages": prior_sorted_stages,
        "prior_verified_units": [
            copy.deepcopy(prior_units_by_stage[s]) for s in prior_sorted_stages
        ],
        "prior_output_dirs": prior_output_dirs,
        "prior_export_roots": cumulative_export_roots,
    }
    if prior_ledger_sha256 is not None:
        completion_meta["prior_ledger_sha256"] = prior_ledger_sha256
    if prior_manifest_path is not None:
        completion_meta["prior_manifest_path"] = str(prior_manifest_path.resolve())

    successor = copy.deepcopy(dict(original_manifest))
    successor["job_id"] = job_id
    successor["output_dir"] = resolved_new_output
    successor["completion_attempt"] = completion_meta
    if (
        "sandbox_admission" in successor
        or packet_dir is not None
        or sandbox_manifest_path is not None
    ):
        successor["sandbox_admission"] = {
            "schema": "standing-a100-development-sandbox-admission/v1",
            "manifest_path": str(resolved_sandbox_path),
        }
    successor["inputs"] = all_inputs
    return CTResumePacket(
        ledger=ledger,
        ledger_sha256=ledger_sha,
        input_records=records,
        manifest=successor,
        verified_stages=stages,
        staging_projection=staging_proj,
    )


def _sandbox_input_kind_for_dest(dest: str) -> str:
    if dest == "b8-s3.pth":
        return "frozen-checkpoint"
    if dest == "registry.json":
        return "registry"
    if dest == "verified-ledger.json":
        return "verified-ledger"
    if dest.startswith("resume/"):
        name = Path(dest).name
        if name.endswith((".sha256", ".complete")):
            return "ct-export-sidecar"
        if name.endswith((".pt.xz", ".pt.xz.manifest")):
            return "ct-export-derivative"
        return "ct-export"
    return "frozen-opponent"


@_evidence_session
def write_continuation_packet(
    packet_dir: Path,
    *,
    prior_packet_dir: Path,
    terminal_receipt: Mapping[str, Any],
    export_root: Path | Sequence[Path],
    completion_index: int,
    job_id: str,
    output_dir: Path,
    use_compact_xz_for_stages: Sequence[int] | None = (1,),
    auto_encode_compact_xz: bool = True,
    supervisor_log_path: Path | None = None,
    validate_registry_inputs: bool = True,
) -> dict[str, Any]:
    """Create-only 6-file admitted C/T continuation packet directory; never dispatches."""
    packet = packet_dir.resolve()
    _no_symlink_path(prior_packet_dir)
    _no_symlink_path(packet_dir)
    prior = prior_packet_dir.resolve()
    if packet.exists() or packet.is_symlink():
        raise FileExistsError(f"continuation packet_dir already exists: {packet}")
    if prior.is_symlink() or not prior.is_dir():
        raise CTResumeRefused(f"missing or symlinked prior_packet_dir: {prior}")

    if supervisor_log_path is None:
        raise CTResumeRefused("production continuation requires bound supervisor_log")
    _no_symlink_path(supervisor_log_path)
    if not supervisor_log_path.is_file():
        raise CTResumeRefused("production continuation requires a regular supervisor_log")
    prior_manifest_path = prior / "dispatch-manifest.json"
    prior_sandbox_path = prior / "sandbox.json"
    prior_control_path = prior / "control.json"
    prior_reg_check_path = prior / "registry-check.json"
    prior_topo_path = prior / "topology-manifest.json"
    prior_ledger_path = prior / "verified-ledger.json"
    for req in (
        prior_manifest_path,
        prior_sandbox_path,
        prior_control_path,
        prior_reg_check_path,
        prior_topo_path,
        prior_ledger_path,
    ):
        if req.is_symlink() or not req.is_file():
            raise CTResumeRefused(f"prior packet lacks required file: {req.name}")

    from research.sandbox_a_resident.ct_preflight import (
        _bounded_manifest_bytes,
        _check_lineage_graph,
    )
    from tools.ops.packet_supersession import require_current_packet

    original_bytes = _bounded_manifest_bytes(prior_manifest_path)
    prior_manifest = _load_json(original_bytes.decode("utf-8"))
    require_current_packet(
        prior_manifest,
        prior_manifest_path,
        root=Path(__file__).resolve().parents[2],
        original_bytes=original_bytes,
    )
    _check_lineage_graph(
        prior,
        prior_manifest,
        _load_json(prior_control_path.read_text()),
        original_bytes=original_bytes,
    )
    prior_manifest_sha = _retained_digest(prior_manifest_path)
    prior_sandbox = _load_json(prior_sandbox_path.read_text())
    prior_broker_binding = (
        prior_sandbox.get("outputs", {}).get("broker_manifest")
        if isinstance(prior_sandbox, Mapping) and isinstance(prior_sandbox.get("outputs"), Mapping)
        else None
    )
    if (
        not isinstance(prior_broker_binding, Mapping)
        or prior_broker_binding.get("sha256") != prior_manifest_sha
    ):
        raise CTResumeRefused(
            "prior sandbox.json broker_manifest SHA-256 differs from dispatch-manifest.json"
        )
    control = _load_json(prior_control_path.read_text())
    reg_check = _load_json(prior_reg_check_path.read_text())
    topo_manifest = _load_json(prior_topo_path.read_text())

    from research.sandbox_a_resident.ct_preflight import validate_uploaded_ledger

    # Preparation can precede the local packet commit; production admission always
    # reopens all components with require_committed=True.
    validate_uploaded_ledger(prior, copy.deepcopy(prior_manifest), control, require_committed=False)
    built = completion_packet(
        prior_manifest,
        terminal_receipt,
        export_root,
        control,
        completion_index=completion_index,
        job_id=job_id,
        output_dir=output_dir,
        original_manifest_sha256=prior_manifest_sha,
        use_compact_xz_for_stages=use_compact_xz_for_stages,
        auto_encode_compact_xz=auto_encode_compact_xz,
        packet_dir=packet,
        supervisor_log_path=supervisor_log_path,
        prior_packet_dir=prior,
        prior_manifest_path=prior_manifest_path,
    )

    provenance_roots = sorted(
        set(
            built.manifest["completion_attempt"]["prior_output_dirs"]
            + built.manifest["completion_attempt"]["prior_export_roots"]
        )
    )
    provenance = {
        "schema": "t34-cf-continuation-provenance/v1",
        "prior_manifest_path": str(prior_manifest_path),
        "prior_manifest_sha256": prior_manifest_sha,
        "receipt": dict(terminal_receipt),
        "retained": {root: retained_inventory(Path(root)) for root in provenance_roots},
        "supervisor_log": (
            None
            if supervisor_log_path is None
            else {
                "path": str(supervisor_log_path.resolve()),
                "size": supervisor_log_path.stat().st_size,
                "sha256": _retained_digest(supervisor_log_path),
            }
        ),
    }
    packet.mkdir(parents=True, exist_ok=False)
    write_once(packet / "control.json", control)
    write_once(packet / "registry-check.json", reg_check)
    write_once(packet / "topology-manifest.json", topo_manifest)
    write_once(packet / "verified-ledger.json", built.ledger)
    if _retained_digest(packet / "verified-ledger.json") != built.ledger_sha256:
        raise CTResumeRefused("written verified-ledger.json SHA-256 differs from manifest binding")

    write_once(packet / "provenance.json", provenance)
    # Preserve canonical seed/topology bindings pointing to identical sidecar bytes.
    manifest_out = copy.deepcopy(built.manifest)
    manifest_out["completion_attempt"]["provenance_sha256"] = _retained_digest(
        packet / "provenance.json"
    )
    manifest_out["sandbox_a"] = {
        **dict(manifest_out.get("sandbox_a", {})),
        "seed_audit_sha256": _retained_digest(packet / "registry-check.json"),
        "topology_manifest_sha256": _retained_digest(packet / "topology-manifest.json"),
    }
    write_once(packet / "dispatch-manifest.json", manifest_out)
    broker_sha = _retained_digest(packet / "dispatch-manifest.json")

    allocations = list(manifest_out["sandbox_a"]["seed_allocations"])
    roots = [int(row["offset"]) for row in allocations]
    sandbox_inputs: list[dict[str, Any]] = [
        {
            "kind": "worlds",
            "study": STUDY,
            "indices": roots,
            "seeds": [world_seed(STUDY, i) for i in roots],
        }
    ]
    for row in manifest_out["inputs"]:
        sandbox_inputs.append(
            {
                "kind": _sandbox_input_kind_for_dest(str(row["dest"])),
                "path": str(Path(row["path"]).resolve()),
                "sha256": str(row["sha256"]),
            }
        )

    sandbox_record = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": STUDY,
        "inputs": sandbox_inputs,
        "salt": prior_sandbox["salt"],
        "endpoint": {"kind": "steps", "value": ct_endpoint_steps(control)},
        "timeout": copy.deepcopy(prior_sandbox["timeout"]),
        "excluded_studies": copy.deepcopy(prior_sandbox["excluded_studies"]),
        "owner_rule": prior_sandbox["owner_rule"],
        "outputs": {
            "label": "simulator-D",
            "paths": [str(output_dir.resolve())],
            "broker_manifest": {
                "path": str((packet / "dispatch-manifest.json").resolve()),
                "sha256": broker_sha,
            },
        },
        "crossing": copy.deepcopy(prior_sandbox["crossing"]),
    }
    lines = [
        ("{" if i == 0 else " ") + json.dumps(k) + ": " + json.dumps(v) + ("," if i < 9 else "}")
        for i, (k, v) in enumerate(sandbox_record.items())
    ]
    with (packet / "sandbox.json").open("x") as stream:
        stream.write("\n".join(lines) + "\n")

    write_once(
        packet / "SEAL.json",
        {
            "schema": "t34-cf-continuation-seal/v1",
            "provenance_sha256": _retained_digest(packet / "provenance.json"),
            "manifest_sha256": broker_sha,
            "ledger_sha256": built.ledger_sha256,
            "prior_manifest_sha256": prior_manifest_sha,
        },
    )
    validate_manifest(load_manifest(packet / "sandbox.json"))
    if validate_registry_inputs:
        reg_entry = next((r for r in sandbox_inputs if r["kind"] == "registry"), None)
        reg_obj = load_registry(Path(reg_entry["path"]) if reg_entry else DEFAULT_REGISTRY)
        for item in sandbox_inputs:
            validate_input(item, reg_obj)

    return {
        "packet": str(packet),
        "job_id": job_id,
        "completion_index": int(completion_index),
        "verified_stages": list(built.verified_stages),
        "ledger_sha256": built.ledger_sha256,
        "manifest_sha256": broker_sha,
        "sandbox_sha256": _retained_digest(packet / "sandbox.json"),
        "staging_projection": built.staging_projection,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--prior-packet-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--export-dir",
        type=Path,
        action="append",
        required=True,
        help="prior attempt export directory (pass once per prior attempt for cross-attempt union)",
    )
    parser.add_argument("--completion-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compact-stage",
        type=int,
        action="append",
        default=None,
        help="stage index to upload via compact .pt.xz (defaults to stage 1)",
    )
    parser.add_argument("--supervisor-log", type=Path, default=None)
    args = parser.parse_args(argv)
    compact_stages = tuple(args.compact_stage) if args.compact_stage is not None else (1,)
    receipt_payload = _load_json(args.receipt.read_text())
    summary = write_continuation_packet(
        args.packet_dir,
        prior_packet_dir=args.prior_packet_dir,
        terminal_receipt=receipt_payload,
        export_root=args.export_dir,
        completion_index=args.completion_index,
        job_id=args.job_id,
        output_dir=args.output_dir,
        use_compact_xz_for_stages=compact_stages,
        auto_encode_compact_xz=True,
        supervisor_log_path=args.supervisor_log,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
