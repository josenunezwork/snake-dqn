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
import copy
import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.sandbox_a_resident.ct_prepare import STUDY, ct_endpoint_steps, write_once
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
        if p.is_symlink():
            raise CTResumeRefused(f"export_root must not be a symlink: {p}")
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
    expected = sidecar.read_text().strip().split()[0]
    if not _is_hex(expected, 64):
        raise CTResumeRefused(f"malformed SHA-256 sidecar: {path.name}")
    if digest(path) != expected:
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
        for key, expected in (("job_id", expected_job_id), ("generation", expected_generation)):
            if expected is not None and record.get(key) != expected:
                raise CTResumeRefused(f"{label} lacks matching {key} identity")
    status = record.get("status")
    reason = record.get("reason")
    returncode = record.get("returncode")
    if returncode == 0 or status in {"completed", "succeeded"}:
        raise CTResumeRefused(f"{label} records completed status")
    if reason is not None and reason not in MECHANICAL_TERMINAL_REASONS:
        raise CTResumeRefused(f"{label} records non-mechanical reason: {reason!r}")
    if returncode is not None and type(returncode) is not int:
        raise CTResumeRefused(f"{label} records malformed returncode")
    if returncode == 124:
        raise CTResumeRefused(f"{label} records watchdog timeout exit code 124")
    if (returncode not in (None, 0) or status in {"failed", "stopped", "error"}) and (
        reason not in MECHANICAL_TERMINAL_REASONS
    ):
        raise CTResumeRefused(
            f"{label} records non-mechanical child failure or unexplained runtime failure"
        )
    if status is not None and status not in {
        "running",
        "starting",
        "ready",
        "failed",
        "stopped",
        "error",
    }:
        raise CTResumeRefused(f"{label} records unknown execution status: {status!r}")


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


def _scan_json_faults(
    value: Any,
    *,
    label: str,
    expected_job_id: str | None = None,
    expected_generation: str | None = None,
) -> None:
    """Inspect nested diagnostic fields, including error fields in snapshot headers."""
    if isinstance(value, Mapping):
        for key, expected in (
            ("job_id", expected_job_id),
            ("job", expected_job_id),
            ("generation", expected_generation),
        ):
            if expected is not None and key in value and value[key] != expected:
                raise CTResumeRefused(f"{label} belongs to foreign {key}")
        for key, item in value.items():
            name = f"{label}.{key}"
            mechanical_error = (
                key == "error"
                and isinstance(item, str)
                and any(marker in item.lower() for marker in MECHANICAL_SUPERVISOR_ERROR_MARKERS)
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
                    "traceback",
                }
                and item
                and not mechanical_error
                and not (
                    key in {"last_error_line", "last_traceback_line"}
                    and isinstance(item, str)
                    and item.startswith("STALL_WATCHDOG_DIAGNOSTICS sealed_dir=")
                    and "\n" not in item
                )
            ):
                raise CTResumeRefused(
                    f"{name} records explicit scientific/non-mechanical fault: {item!r}"
                )
            if (
                key in {"state", "terminal"}
                and isinstance(item, Mapping)
                and any(field in item for field in ("status", "reason", "returncode"))
            ):
                _check_retained_execution(
                    item,
                    label=name,
                    expected_job_id=expected_job_id,
                    expected_generation=expected_generation,
                    require_identity=True,
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
                raise CTResumeRefused(
                    f"{name} records explicit scientific/non-mechanical fault: {item!r}"
                )
            # Inventories name diagnostic files even when they are empty. Their
            # paths are checked by the export scanner; don't treat filenames as errors.
            if key not in {"path", "dest", "sha256"}:
                _scan_json_faults(
                    item,
                    label=name,
                    expected_job_id=expected_job_id,
                    expected_generation=expected_generation,
                )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_json_faults(
                item,
                label=f"{label}[{index}]",
                expected_job_id=expected_job_id,
                expected_generation=expected_generation,
            )
    elif isinstance(value, str):
        _refuse_if_fault_text(value, field_name=label)


def _scan_retained_text(
    path: Path,
    *,
    output_root: Path,
    expected_job_id: str | None = None,
    expected_generation: str | None = None,
) -> None:
    """Scan all text/JSON, including incomplete telemetry and unknown filenames.

    Model payloads are binary and remain path/size/hash evidence. Text is streamed
    with overlap so a fault marker split across read boundaries is still found.
    Incomplete JSON refuses after scanning its retained text for explicit faults.
    """
    # Staged executable source contains exception definitions and canary checks,
    # not observations. Inventory its bytes, but scan diagnostic files separately.
    if path.parent == output_root / "staging" and path.name in {
        "bootstrap.sh",
        "source.py",
        "runtime.py",
        "release.py",
    }:
        qualified_source = Path(__file__).resolve().parents[1] / "colab_cli" / path.name
        if digest(path) != digest(qualified_source):
            raise CTResumeRefused(
                f"retained staged executable differs from qualified source: {path}"
            )
        return
    is_json = path.suffix == ".json" or path.name.endswith(".json.partial")
    if is_json:
        try:
            text = path.read_text()
            payload = json.loads(text)
        except (UnicodeError, json.JSONDecodeError) as error:
            if isinstance(error, json.JSONDecodeError):
                _refuse_if_fault_text(text, field_name=str(path))
            label = (
                "corrupt poll evidence"
                if path.name.startswith("poll-")
                else "unreadable or partial retained JSON evidence"
            )
            raise CTResumeRefused(f"{label}: {path}") from error
        if path.name in {
            "state.json",
            "state.json.partial",
            "terminal.json",
            "terminal.json.partial",
        }:
            if not isinstance(payload, Mapping):
                raise CTResumeRefused(f"malformed retained execution record: {path}")
            _check_retained_execution(
                payload,
                label=str(path),
                expected_job_id=expected_job_id,
                expected_generation=expected_generation,
                require_identity=True,
            )
        if path.name.startswith("poll-"):
            if not isinstance(payload, Mapping):
                raise CTResumeRefused(f"malformed poll evidence: {path}")
            _check_poll_fault_evidence(payload, label=str(path))
        if path.name in {"receipt.json", "receipt.json.partial"} and isinstance(payload, Mapping):
            if payload.get("status") != "INCOMPLETE" or payload.get("stop_verified") is not True:
                raise CTResumeRefused(f"non-resumable retained receipt: {path}")
            _refuse_if_fault_text(payload.get("error"), field_name=str(path))
            payload = {k: v for k, v in payload.items() if k != "error"}
        _scan_json_faults(
            payload,
            label=str(path),
            expected_job_id=expected_job_id,
            expected_generation=expected_generation,
        )
        return
    # Known model payload formats must never be interpreted as diagnostic text.
    if any(suffix in {".pt", ".pth", ".xz", ".gz", ".zip"} for suffix in path.suffixes):
        return
    with path.open("rb") as stream:
        sample = stream.read(8192)
        if b"\x00" in sample:
            raise CTResumeRefused(f"unreadable retained diagnostic evidence: {path}")
    with path.open(encoding="utf-8") as stream:
        overlap = ""
        while part := stream.read(65536):
            _refuse_if_fault_text(overlap + part, field_name=str(path))
            overlap = part[-256:]


def retained_inventory(root: Path) -> dict[str, Any]:
    """Bind every retained byte by path/size/hash without decoding numerical payloads."""
    if not root.is_absolute() or root.is_symlink():
        raise CTResumeRefused(f"invalid retained evidence root: {root}")
    if not root.exists():
        return {"exists": False, "files": []}
    if not root.is_dir():
        raise CTResumeRefused(f"retained evidence root is not a directory: {root}")
    return {
        "exists": True,
        "files": [
            {"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": digest(p)}
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
    inventory = retained_inventory(base.resolve() if not base.is_symlink() else base)
    for row in inventory["files"]:
        path = base / row["path"]
        name = path.name.lower()
        if name.startswith("summary.json"):
            raise CTResumeRefused(f"scientific summary already present: {path}")
        if name.startswith("fault.json"):
            raise CTResumeRefused(f"scientific/runtime fault diagnostic present: {path}")
        if name == "faulthandler-stacks.log" and row["size"]:
            raise CTResumeRefused(f"non-empty faulthandler stack dump: {path}")
        _scan_retained_text(
            path,
            output_root=base,
            expected_job_id=expected_job_id,
            expected_generation=expected_generation,
        )


def _candidate_search_roots(export_root: Path) -> list[Path]:
    roots = [export_root]
    if export_root.name == "ct" and export_root.parent.name == "exports":
        if len(export_root.parents) >= 2:
            roots.append(export_root.parents[1])
    return roots


def _check_attempt_output_dir_fault_evidence(output_dir: Path) -> None:
    """Scan an attempt's ``output_dir`` and its export subdirectories without climbing to parent."""
    if output_dir.is_symlink():
        raise CTResumeRefused(f"prior output_dir must not be a symlink: {output_dir}")
    if output_dir.exists():
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
    )
    _refuse_if_fault_text(
        receipt.get("error"), field_name="receipt.error", check_supervisor_denylist=True
    )
    _refuse_if_fault_exports(receipt.get("exports"), source_name="receipt")
    terminal = receipt.get("terminal")
    if isinstance(terminal, Mapping):
        _check_terminal_fault_evidence(terminal, label="terminal")


def verify_mechanical_stop(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    *,
    export_root: Path | Sequence[Path],
    supervisor_log_path: Path | None = None,
) -> None:
    """Enforce scientific-fault precedence first, then verify clean qualified mechanical stop."""
    assert_no_scientific_or_fault_precedence(export_root, receipt)
    if supervisor_log_path is not None:
        if supervisor_log_path.is_symlink() or not supervisor_log_path.is_file():
            raise CTResumeRefused(f"missing or symlinked supervisor log: {supervisor_log_path}")
        for idx, line in enumerate(supervisor_log_path.read_text().splitlines(), start=1):
            _refuse_if_supervisor_log_fault(line, field_name=f"supervisor_log:{idx}")

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
            lowered = error.lower()
            if any(marker in lowered for marker in MECHANICAL_SUPERVISOR_ERROR_MARKERS) and not any(
                bad in lowered for bad in NON_MECHANICAL_SUPERVISOR_ERROR_MARKERS
            ):
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
        info = json.loads(info_path.read_text())
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
        if (
            not _is_hex(expected_lsha, 64)
            or prior_ledger_file.is_symlink()
            or not prior_ledger_file.is_file()
            or digest(prior_ledger_file) != expected_lsha
        ):
            raise CTResumeRefused(
                "prior verified-ledger.json is missing, symlinked, or hash-mismatched"
            )
        if prior_packet_dir is not None:
            pkt_ledger = prior_packet_dir / "verified-ledger.json"
            if (
                pkt_ledger.is_symlink()
                or not pkt_ledger.is_file()
                or digest(pkt_ledger) != expected_lsha
            ):
                raise CTResumeRefused(
                    "prior packet verified-ledger.json differs from manifest binding"
                )
        prior_ledger_sha256 = str(expected_lsha)
        try:
            loaded_ledger = json.loads(prior_ledger_file.read_text())
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
                    if (
                        item_path.is_symlink()
                        or not item_path.is_file()
                        or item_path.stat().st_size != bound["size"]
                        or digest(item_path) != bound["sha256"]
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
                        if Path(str(sc_up.get("path", ""))) != sc_file or digest(
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
            if (
                src.is_symlink()
                or not src.is_file()
                or src.stat().st_size != exp["size"]
                or digest(src) != exp["sha256"]
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
                    if Path(str(sc_up.get("path", ""))) != sc_file or digest(sc_file) != sc_up.get(
                        "sha256"
                    ):
                        raise CTResumeRefused("prior uploaded stage sidecar mismatch")
                    referenced_resume.add(dest + sc_suffix)
            _append_unique_dir_str(prior_export_roots, src.resolve().parent)

    if referenced_resume != {d for d in prior_uploads if d.startswith("resume/")}:
        raise CTResumeRefused("prior manifest carries unreferenced resume/* upload")

    return prior_units_by_stage, prior_ledger_sha256, prior_output_dirs, prior_export_roots


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
    prior = prior_packet_dir.resolve()
    if packet.exists() or packet.is_symlink():
        raise FileExistsError(f"continuation packet_dir already exists: {packet}")
    if prior.is_symlink() or not prior.is_dir():
        raise CTResumeRefused(f"missing or symlinked prior_packet_dir: {prior}")

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

    prior_manifest = json.loads(prior_manifest_path.read_text())
    prior_manifest_sha = digest(prior_manifest_path)
    prior_sandbox = json.loads(prior_sandbox_path.read_text())
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
    control = json.loads(prior_control_path.read_text())
    reg_check = json.loads(prior_reg_check_path.read_text())
    topo_manifest = json.loads(prior_topo_path.read_text())

    from research.sandbox_a_resident.ct_preflight import validate_uploaded_ledger

    validate_uploaded_ledger(prior, copy.deepcopy(prior_manifest), control)
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
                "sha256": digest(supervisor_log_path),
            }
        ),
    }
    packet.mkdir(parents=True, exist_ok=False)
    write_once(packet / "control.json", control)
    write_once(packet / "registry-check.json", reg_check)
    write_once(packet / "topology-manifest.json", topo_manifest)
    write_once(packet / "verified-ledger.json", built.ledger)
    if digest(packet / "verified-ledger.json") != built.ledger_sha256:
        raise CTResumeRefused("written verified-ledger.json SHA-256 differs from manifest binding")

    write_once(packet / "provenance.json", provenance)
    # Preserve canonical seed/topology bindings pointing to identical sidecar bytes.
    manifest_out = copy.deepcopy(built.manifest)
    manifest_out["completion_attempt"]["provenance_sha256"] = digest(packet / "provenance.json")
    manifest_out["sandbox_a"] = {
        **dict(manifest_out.get("sandbox_a", {})),
        "seed_audit_sha256": digest(packet / "registry-check.json"),
        "topology_manifest_sha256": digest(packet / "topology-manifest.json"),
    }
    write_once(packet / "dispatch-manifest.json", manifest_out)
    broker_sha = digest(packet / "dispatch-manifest.json")

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
            "provenance_sha256": digest(packet / "provenance.json"),
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
        "sandbox_sha256": digest(packet / "sandbox.json"),
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
    receipt_payload = json.loads(args.receipt.read_text())
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
