"""Fail-closed admission and supervisor failure classification for Watch6-L1-V1."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from research.colab_cli import canary
from research.colab_cli import source as colab_source
from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.costs import (
    BASE_MINUTES_BY_GPU,
    CANARY_BASE_MINUTES,
    MAX_BASE_MINUTES,
    PRICED_CEILING_RATES_U_PER_H,
    empirical_forecast,
    verify_report_evidence,
)
from research.watch6_l1_confirm.mirror import validate as validate_mirror
from research.watch6_l1_confirm.resume import WORLD_NAME_RE
from research.watch6_l1_confirm.uploads import METADATA, check_uploads, stage_metadata
from tools.ops.package_spec import digest, read_ledger, verify_immutable

EXTERNAL_TRANSPORT_CLASSES = {
    "RPC_EXEC_POLL_TIMEOUT",
    "SUPERVISOR",
    "TRANSPORT_TIMEOUT",
    "RUNTIME_LOST",
    "ACCOUNTING",
}

LEASE_TERMINAL_REASONS = frozenset({"lease_expired", "release_credential_expiring"})
GRACEFUL_COMPUTE_TERMINAL_REASONS = LEASE_TERMINAL_REASONS | frozenset({"timeout"})
RETAINED_PENDING_CLEANUP_ERROR = "ALLOCATION_UNRESOLVED: retained pending record"

AMENDMENT_SCHEMA = "watch6-l1-hardware-reassignment-amendment/v1"
AMENDMENT_ID = "AMENDMENT_L4_REASSIGN_01"
AMENDMENT_PATH: Path = p.ROOT / "research/watch6_l1_confirm/AMENDMENT_L4_REASSIGN_01.json"
AMENDMENT_SHA256 = "71083fecc7cb34fce5ee531e7cceeca67f31fbd2515c6e865a46f694ee115659"
AMENDMENT_SHARDS: tuple[int, ...] = (8, 9, 10)
AMENDMENT_GPU = "L4"
AMENDMENT_WORKERS = 6

INCIDENT_SCHEMA = "watch6-l1-host-interrupted-attempt/v1"
INCIDENT_ID = "INCIDENT_HOST_RESTART_01"
INCIDENT_PATH: Path = p.ROOT / "research/watch6_l1_confirm/INCIDENT_HOST_RESTART_01.json"
INCIDENT_SHA256 = "6b9218fea73febbb8595af16859dda07001687c9bd39f916408638e75c77e1b9"
INCIDENT_SHARD = 11
INCIDENT_POLL_COUNT = 1083
INCIDENT_LAST_POLL_NAME = "poll-001082.json"
INCIDENT_LAST_POLL_UPDATED_AT = 1793669698
INCIDENT_HOST_BOOT_EPOCH = 1793670392
INCIDENT_RULING = "RULING_L1_V1_R5.md"
INCIDENT_EXPORTS_LISTING: tuple[str, ...] = (
    "performance.json",
    "performance.json.sha256",
    "performance.json.complete",
    "workload-started.json",
    "workload-started.json.sha256",
    "workload-started.json.complete",
)
EXPECTED_INCIDENT_KEYS: frozenset[str] = frozenset(
    {
        "schema",
        "incident_id",
        "namespace",
        "science",
        "registration_sha256",
        "shard",
        "compute_completion_attempt",
        "classification",
        "attempt_output_dir",
        "attempt_manifest_sha256",
        "manifest_commit",
        "gpu",
        "workers",
        "hardware_json_sha256",
        "last_poll",
        "poll_count",
        "supervisor_log",
        "host_boot_epoch",
        "sessions_evidence",
        "exports_listing",
        "world_outputs",
        "ruling",
        "ruling_sha256",
    }
)

# Explicit allowlist of inert macOS filesystem metadata names ignored under exports/, output_dir/,
# and ops/manifests/ directory scans. Every other dot-file (e.g. .tmp, .part) remains fail-closed.
INERT_MACOS_METADATA_NAMES: frozenset[str] = frozenset(
    {
        ".DS_Store",
        "._.DS_Store",
        ".localized",
        ".Spotlight-V100",
        ".Trashes",
        ".fseventsd",
        ".TemporaryItems",
        ".VolumeIcon.icns",
    }
)

ALLOWED_PREALLOC_FILES: frozenset[str] = frozenset(
    {
        # research/colab_cli/supervisor.py write_once before colab new (RULING_L1_V1_R7).
        "accounting-baseline.json",
        "manifest.json",
        "manifest.json.sha256",
        "manifest.json.complete",
        "receipt.json",
        "receipt.json.sha256",
        "receipt.json.complete",
        "unavailable.json",
        "unavailable.json.sha256",
        "unavailable.json.complete",
    }
)

# Local staging files written by research/colab_cli/supervisor.py (Broker.run) and
# research/colab_cli/source.py (prepare) into output_dir/staging/ before VM allocation.
ALLOWED_PREALLOC_STAGING_FILES: frozenset[str] = frozenset(
    {
        colab_source.ARCHIVE_NAME,
        colab_source.PLAN_NAME,
        "schema-validation.json",
        "runtime.py",
        "bootstrap.sh",
        "release.py",
        "source.py",
    }
)

FORBIDDEN_WATCHDOG_SUBSTRINGS = (
    "WATCHDOG",
    "FAULTHANDLER",
    "DUMP_TRACEBACK_LATER",
)

WALL_TIMEOUT_SUBSTRINGS = (
    "WALL_TIMEOUT",
    "WHOLE-SESSION WALL CAP",
    "WHOLE-SESSION DEADLINE",
    "WALL CAP REACHED",
    "TIMEOUT_MINUTES",
    "EXCEEDED WALL",
    "EXCEEDED TIMEOUT",
)

_NON_TRANSPORT_FAULT_RE = re.compile(
    r"(?:"
    r"\bnan\b"
    r"|\binf(?:inity)?\b"
    r"|non[-_\s]?finite"
    r"|\boverflow\b"
    r"|\bunderflow\b"
    r"|\bnumerical\b"
    r"|\bmirror\b"
    r"|\bmismatch\b"
    r"|\bchecksum\b"
    r"|sha[-_\s]?256.*mismatch"
    r"|digest.*mismatch"
    r"|hash.*mismatch"
    r"|\bcorrupt"
    r"|\btamper"
    r"|\bforeign\b"
    r"|\bduplicate\b"
    r"|illegal[\s_-]?action"
    r"|action[\s_-]?collapse"
    r"|fallback[\s_-]?row"
    r"|stock[\s_-]?row"
    r"|\bnoncandidate"
    r"|q[-_\s]?value"
    r"|\blogit"
    r"|\bintegrity\b"
    r"|\bscientific\b"
    r"|WATCH6_L1_SCIENTIFIC_FAULT"
    r"|\bpytest\b"
    r"|test.*fail"
    r"|kernel error despite cli exit zero"
    r"|runtime unassign failed"
    r"|out of memory"
    r"|\boom\b"
    r"|\bcuda\b"
    r"|\bcudnn\b"
    r"|\bcublas\b"
    r"|segfault"
    r"|segmentation fault"
    r"|\bsig(?:segv|abrt|bus|ill|fpe)\b"
    r"|\bwatchdog\b"
    r"|\bfaulthandler\b"
    r"|dump_traceback_later"
    r"|timeout\s*\(0:15:00\)"
    r"|\b(?:AssertionError|ValueError|TypeError|KeyError|IndexError|AttributeError"
    r"|ZeroDivisionError|ArithmeticError|FloatingPointError|OverflowError"
    r"|ImportError|ModuleNotFoundError|NameError|UnboundLocalError"
    r"|SyntaxError|IndentationError|NotImplementedError|RecursionError"
    r"|MemoryError|BufferError|LookupError|SystemError"
    r"|FileNotFoundError|FileExistsError|IsADirectoryError|NotADirectoryError"
    r"|PermissionError|JSONDecodeError|UnicodeError|UnicodeDecodeError|UnicodeEncodeError"
    r"|KeyboardInterrupt|SystemExit)\b"
    r"|(?:research/watch6_l1_confirm|research/watch6_search_efficacy"
    r"|research/bl_served_search|research/t1_commitment"
    r"|/src/|\"src/|'src/|/web/|\"web/|'web/|job\.sh)"
    r")",
    re.IGNORECASE,
)

_FRAME_PATH_RE = re.compile(r'File "([^"]+)", line \d+')
_RICH_FRAME_PATH_RE = re.compile(r"([^\s:]+\.py):\d+\s+in\s+")
_FORBIDDEN_FRAME_RE = re.compile(
    r"(?:^|/)(?:src|web|research/watch6_l1_confirm|research/watch6_search_efficacy(?:_v3)?"
    r"|research/bl_served_search|research/t1_commitment[^/]*)(?:/|$)|job\.sh$",
    re.IGNORECASE,
)
_ALLOWED_TRANSPORT_FRAME_RE = re.compile(
    r"(?:colab|httpx|httpcore|urllib|requests|websockets|http/client\.py|ssl\.py|socket\.py"
    r"|subprocess\.py|research/colab_cli/)",
    re.IGNORECASE,
)
_TRANSPORT_EXCEPTION_LINE_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_.]*\.)?"
    r"(TimeoutError|TimeoutExpired|ConnectTimeout|ReadTimeout|WriteTimeout|PoolTimeout"
    r"|TimeoutException|Timeout|ConnectionError|ConnectionResetError|ConnectionAbortedError"
    r"|ConnectionRefusedError|BrokenPipeError|EOFError|RemoteDisconnected|IncompleteRead"
    r"|SSLError|SSLEOFError|SSLCertVerificationError|gaierror|NetworkError|ConnectError"
    r"|ReadError|WriteError|RemoteProtocolError|TransportError|ChunkedEncodingError"
    r"|URLError|HTTPError|HTTPStatusError|ColabRequestError|RpcTimeoutError"
    r"|SessionLostError|GuardError|ConnectionClosed|ConnectionClosedError|WebSocketException)"
    r"(?:\s*:\s*(.+?))?\s*$"
)
_CLIENT_4XX_RE = re.compile(r"\b(?:400|401|403|404|405|406|407|409|410|422)\b")
_GENERIC_LOG_EXCEPTION_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_.]*\.)?[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Fault|Exit)\s*:",
    re.MULTILINE,
)
_UPLOAD_CAP_GATE_RE = re.compile(
    r"(?:total upload time cap reached after staging control|upload/staging cap"
    r"|StagingCapError|transfer_seconds|staging_seconds)",
    re.IGNORECASE,
)
_RENEWAL_WS_TRUNCATED_RE = re.compile(
    r"(?:renewal.*websocket|websocket.*closed|ConnectionClosed(?:Error|OK)?|keepalive.*ping)",
    re.IGNORECASE,
)
_ACCOUNTING_PROVIDER_LAG_RE = re.compile(
    r"(?:usage|balance|accounting|ccu|quota|poll).*?"
    r"(?:timeout|unavailable|stale|lag|failed|503|502|429)",
    re.IGNORECASE,
)

FORBIDDEN_PREALLOC_ENTRIES: tuple[str, ...] = (
    "terminal.json",
    "hardware.json",
    "ready.json",
    "warmup-measurement.json",
    "rate-calibration.json",
    "upload-measurement.json",
    "source-delivery.json",
    "input-hashes.json",
    "usage-allocated.json",
    "runtime-measurement.json",
    "scientific-fault.json",
    "supervisor.log",
    "job.log",
    "performance.json",
    "operational.json",
)


def _unwrap_text_variants(raw: object) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    variants = [raw]
    stripped = raw.strip()
    try:
        parsed = ast.parse(stripped, mode="eval")
    except SyntaxError:
        parsed = None
    if parsed is not None and isinstance(parsed.body, ast.Call):
        for arg in parsed.body.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value:
                variants.append(arg.value)
    return variants


def is_inert_macos_metadata(path: Path, *, root: Path | None = None) -> bool:
    """True iff path is a non-symlink entry under an explicitly listed inert macOS metadata name."""
    if path.is_symlink():
        return False
    rel_parts = path.relative_to(root).parts if root is not None else (path.name,)
    return any(part in INERT_MACOS_METADATA_NAMES for part in rel_parts)


def has_non_inert_entries(directory: Path) -> bool:
    """True iff directory is a symlink or contains any entry outside INERT_MACOS_METADATA_NAMES."""
    if directory.is_symlink():
        return True
    if not directory.exists():
        return False
    for entry in directory.rglob("*"):
        if entry.is_symlink():
            return True
        if is_inert_macos_metadata(entry, root=directory):
            continue
        return True
    return False


def has_non_transport_fault(text: object) -> bool:
    """True iff error/log text carries any scientific, integrity, child, or watchdog fault."""
    for candidate in _unwrap_text_variants(text):
        if _NON_TRANSPORT_FAULT_RE.search(candidate) is not None:
            return True
        upper = candidate.upper()
        if any(token in upper for token in FORBIDDEN_WATCHDOG_SUBSTRINGS):
            return True
    return False


def has_wall_timeout_marker(text: object) -> bool:
    """True iff text carries a whole-session wall-timeout marker (compute-only, never prestart)."""
    for candidate in _unwrap_text_variants(text):
        upper = candidate.upper()
        if any(token in upper for token in WALL_TIMEOUT_SUBSTRINGS):
            return True
    return False


def _has_failed_transport_record(receipt: dict) -> bool:
    for key in ("cli_calls", "control_operations", "contents_transfers", "transfers"):
        records = receipt.get(key)
        if not isinstance(records, list):
            continue
        for item in records:
            if not isinstance(item, dict):
                continue
            rc = item.get("returncode")
            status = item.get("status")
            err = item.get("error")
            if (
                (rc is not None and rc != 0)
                or (status is not None and status not in ("ok", "verified", "PASS", "completed"))
                or (isinstance(err, str) and bool(err.strip()))
            ):
                return True
    return False


def has_verified_transport_traceback(receipt: dict) -> bool:
    """Require explicit host/CLI transport traceback evidence free of scientific/child frames."""
    for field in ("error", "traceback", "last_traceback_line"):
        if has_non_transport_fault(receipt.get(field)):
            return False

    candidates: list[str] = []
    for field in ("error", "traceback"):
        candidates.extend(_unwrap_text_variants(receipt.get(field)))

    for text in candidates:
        if "Traceback (most recent call last)" not in text:
            continue
        frame_paths = _FRAME_PATH_RE.findall(text) + _RICH_FRAME_PATH_RE.findall(text)
        if not frame_paths:
            continue
        if any(_FORBIDDEN_FRAME_RE.search(fp) is not None for fp in frame_paths):
            return False
        if not any(_ALLOWED_TRANSPORT_FRAME_RE.search(fp) is not None for fp in frame_paths):
            continue

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        matched_terminal = False
        for line in reversed(lines):
            if line.startswith(("╰", "└", "─")):
                continue
            exc_match = _TRANSPORT_EXCEPTION_LINE_RE.match(line)
            if exc_match is None:
                break
            exc_cls = exc_match.group(1)
            detail = exc_match.group(2) or ""
            if exc_cls == "GuardError" and "tracked kernel reconnect required" not in detail:
                return False
            if exc_cls in {"HTTPError", "HTTPStatusError", "ColabRequestError"}:
                if _CLIENT_4XX_RE.search(detail) is not None:
                    return False
            matched_terminal = True
            break
        if matched_terminal:
            return True
    return False


def _read_optional_measurement(path: Path) -> tuple[bool, dict | None]:
    if not path.exists() and not path.is_symlink():
        return True, None
    if path.is_symlink() or not path.is_file():
        return False, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, None
    if not isinstance(data, dict):
        return False, None
    for key in ("error", "last_probe_error"):
        if has_non_transport_fault(data.get(key)):
            return False, None
    return True, data


def _audit_job_log_clean(output_dir: Path) -> bool:
    """Verify job.log (when present) has zero tracebacks, scientific faults, or watchdog hangs."""
    job_log = output_dir / "job.log"
    if not job_log.exists() and not job_log.is_symlink():
        return True
    if job_log.is_symlink() or not job_log.is_file():
        return False
    try:
        text = job_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return (
        "Traceback (most recent call last)" not in text
        and "WATCH6_L1_SCIENTIFIC_FAULT" not in text
        and _GENERIC_LOG_EXCEPTION_RE.search(text) is None
        and not has_non_transport_fault(text)
    )


def _audit_poll_files(
    output_dir: Path,
    *,
    allow_graceful_compute_terminal: bool = False,
    allow_precompute_fetch_rc74: bool = False,
) -> tuple[bool, int]:
    """Audit real supervisor poll-*.json snapshots for zero scientific/child/watchdog faults."""
    for legacy in sorted(output_dir.glob("telemetry-*.json")):
        try:
            raw_legacy = legacy.read_text(encoding="utf-8")
        except OSError:
            return False, 0
        if "Traceback (most recent call last)" in raw_legacy or has_non_transport_fault(raw_legacy):
            return False, 0

    poll_paths = sorted(output_dir.glob("poll-*.json"))
    for poll_path in poll_paths:
        if poll_path.is_symlink() or not poll_path.is_file():
            return False, len(poll_paths)
        try:
            poll = json.loads(poll_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, len(poll_paths)
        if not isinstance(poll, dict):
            return False, len(poll_paths)
        term = poll.get("terminal")
        if term is not None:
            if (
                allow_graceful_compute_terminal
                and isinstance(term, dict)
                and (term.get("reason") in GRACEFUL_COMPUTE_TERMINAL_REASONS)
                and term.get("returncode") != 86
            ):
                pass
            elif (
                allow_precompute_fetch_rc74
                and isinstance(term, dict)
                and (term.get("returncode") == 74)
            ):
                pass
            else:
                return False, len(poll_paths)
        state = poll.get("state")
        if not isinstance(state, dict):
            return False, len(poll_paths)
        if state.get("status") not in ("starting", "running"):
            if (
                allow_graceful_compute_terminal
                and state.get("reason") in GRACEFUL_COMPUTE_TERMINAL_REASONS
                and state.get("returncode") != 86
            ):
                pass
            elif allow_precompute_fetch_rc74 and state.get("returncode") == 74:
                pass
            else:
                return False, len(poll_paths)
        for tail_key in ("job_log_tail", "supervisor_log_tail"):
            tail = str(poll.get(tail_key) or "")
            if allow_precompute_fetch_rc74:
                tail = "\n".join(
                    ln
                    for ln in tail.splitlines()
                    if "WATCH6_L1_PRECOMPUTE_PUBLIC_INPUT_FETCH_FAILED" not in ln
                )
            if (
                "Traceback (most recent call last)" in tail
                or _GENERIC_LOG_EXCEPTION_RE.search(tail) is not None
                or has_non_transport_fault(tail)
            ):
                return False, len(poll_paths)
    return True, len(poll_paths)


def _is_precompute_public_fetch_failure(receipt: dict, output_dir: Path) -> bool:
    """True iff VM exited rc=74 during checkpoint fetch before workload-started.json or any unit."""
    if (
        receipt.get("status") != "INCOMPLETE"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
        or not isinstance(receipt.get("endpoint"), str)
        or not receipt.get("endpoint")
        or receipt.get("failure_class") != "CHILD_NONZERO_EXIT"
        or receipt.get("exports")
    ):
        return False
    exports_dir = output_dir / "exports"
    if has_non_inert_entries(exports_dir):
        return False
    terminal = receipt.get("terminal")
    if not isinstance(terminal, dict) or terminal.get("returncode") != 74:
        return False
    if "WATCH6_L1_SCIENTIFIC_FAULT" in str(receipt.get("error") or ""):
        return False
    polls_ok, _ = _audit_poll_files(output_dir, allow_precompute_fetch_rc74=True)
    if not polls_ok:
        return False
    job_log = output_dir / "job.log"
    if job_log.is_file() and not job_log.is_symlink():
        text = job_log.read_text(encoding="utf-8", errors="replace")
        return (
            "WATCH6_L1_PRECOMPUTE_PUBLIC_INPUT_FETCH_FAILED:" in text
            and "WATCH6_L1_SCIENTIFIC_FAULT" not in text
        )
    return any(
        "WATCH6_L1_PRECOMPUTE_PUBLIC_INPUT_FETCH_FAILED:"
        in str(json.loads(p_path.read_text(encoding="utf-8")).get("job_log_tail") or "")
        for p_path in sorted(output_dir.glob("poll-*.json"))
    )


def _prealloc_tree_clean(output_dir: Path) -> bool:
    """Verify a pre-allocation output_dir contains only inert macOS metadata and prealloc files."""
    if output_dir.is_symlink():
        return False
    exports_dir = output_dir / "exports"
    staging_dir = output_dir / "staging"
    has_staging_dir = False
    staging_files: set[str] = set()
    for entry in output_dir.rglob("*"):
        if entry.is_symlink():
            return False
        rel = entry.relative_to(output_dir)
        if rel.parts[0] == "staging":
            if entry == staging_dir and entry.is_dir():
                has_staging_dir = True
                continue
            if (
                len(rel.parts) != 2
                or rel.name not in ALLOWED_PREALLOC_STAGING_FILES
                or not entry.is_file()
            ):
                return False
            staging_files.add(rel.name)
            continue
        if is_inert_macos_metadata(entry, root=output_dir):
            continue
        if entry == exports_dir and entry.is_dir() and not has_non_inert_entries(exports_dir):
            continue
        if len(rel.parts) != 1 or rel.name not in ALLOWED_PREALLOC_FILES or not entry.is_file():
            return False
    if has_staging_dir and staging_files != ALLOWED_PREALLOC_STAGING_FILES:
        return False
    return True


SOURCE_DELIVERY_POLL_DEADLINE_ERROR = "Refused('deadline reached during telemetry recovery')"
ALLOWED_STAGING_PHASE_FILES: frozenset[str] = ALLOWED_PREALLOC_FILES | frozenset(
    {
        "warmup-measurement.json",
        "rate-calibration.json",
        "upload-measurement.json",
        "usage-allocated.json",
    }
)
_RATE_CALIBRATION_SAMPLE_RE = re.compile(r"^rate-calibration-\d{3}\.json$")


def _is_source_delivery_poll_deadline(receipt: dict, output_dir: Path, upload: dict | None) -> bool:
    """True iff a supervisor poll_rpc deadline fired inside deliver_source, pre-ready.json."""
    if (
        receipt.get("failure_class") != "RPC_EXEC_POLL_TIMEOUT"
        or receipt.get("error") != SOURCE_DELIVERY_POLL_DEADLINE_ERROR
        or receipt.get("traceback") is not None
        or receipt.get("last_traceback_line") is not None
        or receipt.get("exports") != []
        or receipt.get("terminal") is not None
        or not isinstance(upload, dict)
        or upload.get("staging_completed") is not False
        or not isinstance(upload.get("source_delivery_seconds"), (int, float))
        or isinstance(upload.get("source_delivery_seconds"), bool)
        or upload["source_delivery_seconds"] <= 0
    ):
        return False
    transfers = receipt.get("transfers")
    if not isinstance(transfers, list) or not transfers:
        return False
    if any(not isinstance(t, dict) or t.get("status") != "verified" for t in transfers):
        return False
    failed_ops = [
        op.get("operation")
        for op in (receipt.get("control_operations") or [])
        if isinstance(op, dict)
        and op.get("status") != "completed"
        and op.get("operation") != "renewal-heartbeat"
    ]
    if failed_ops not in (["source-fetch"], ["source-install_fallback"]):
        return False
    for name in (
        "source-delivery.json",
        "input-hashes.json",
        "hardware.json",
        "ready.json",
        "terminal.json",
        "job.log",
        "supervisor.log",
    ):
        if (output_dir / name).exists() or (output_dir / name).is_symlink():
            return False
    # Fail-closed tree allowlist for the staging phase (mirrors _prealloc_tree_clean).
    staging_dir = output_dir / "staging"
    staging_files: set[str] = set()
    for entry in output_dir.rglob("*"):
        if entry.is_symlink():
            return False
        rel = entry.relative_to(output_dir)
        if rel.parts[0] == "staging":
            if entry == staging_dir and entry.is_dir():
                continue
            if (
                len(rel.parts) != 2
                or rel.name not in ALLOWED_PREALLOC_STAGING_FILES
                or not entry.is_file()
            ):
                return False
            staging_files.add(rel.name)
            continue
        if is_inert_macos_metadata(entry, root=output_dir):
            continue
        if len(rel.parts) != 1 or not entry.is_file():
            return False
        if (
            rel.name not in ALLOWED_STAGING_PHASE_FILES
            and _RATE_CALIBRATION_SAMPLE_RE.match(rel.name) is None
        ):
            return False
    return staging_files == ALLOWED_PREALLOC_STAGING_FILES


def is_prestart_transport_failure(receipt: dict, output_dir: Path | None) -> bool:
    """True iff positive supervisor evidence proves failure occurred before unit compute began."""
    if not isinstance(output_dir, Path) or output_dir.is_symlink() or not output_dir.is_dir():
        return False
    for fault_glob in ("runtime-measurement*", "scientific-fault*"):
        if any(output_dir.glob(fault_glob)):
            return False
    if receipt.get("status") != "INCOMPLETE":
        return False
    if _is_precompute_public_fetch_failure(receipt, output_dir):
        return True
    if receipt.get("exports") or receipt.get("terminal") is not None:
        return False
    if (output_dir / "terminal.json").exists() or (output_dir / "terminal.json").is_symlink():
        return False
    exports_dir = output_dir / "exports"
    if has_non_inert_entries(exports_dir):
        return False

    if (
        (output_dir / "hardware.json").exists()
        or (output_dir / "hardware.json").is_symlink()
        or any(output_dir.glob("poll-*.json"))
        or any(output_dir.glob("telemetry-*.json"))
        or any(output_dir.glob("usage-[0-9]*.json"))
        or any(output_dir.glob("spend-*.json"))
    ):
        return False

    failure_class = str(receipt.get("failure_class") or "")
    if failure_class in {"CHILD_NONZERO_EXIT", "FOREIGN_GENERATION"}:
        return False
    for field in ("error", "traceback", "last_traceback_line"):
        if has_non_transport_fault(receipt.get(field)) or has_wall_timeout_marker(
            receipt.get(field)
        ):
            return False

    ok_w, warmup = _read_optional_measurement(output_dir / "warmup-measurement.json")
    ok_c, calib = _read_optional_measurement(output_dir / "rate-calibration.json")
    ok_u, upload = _read_optional_measurement(output_dir / "upload-measurement.json")
    if not (ok_w and ok_c and ok_u):
        return False

    has_source_delivery = (output_dir / "source-delivery.json").exists()
    has_input_hashes = (output_dir / "input-hashes.json").exists()
    endpoint = receipt.get("endpoint")

    # Proof 1: Pre-allocation refusal before any VM endpoint was assigned (including 412/503).
    if endpoint is None:
        if any((output_dir / name).exists() for name in FORBIDDEN_PREALLOC_ENTRIES):
            return False
        if not _prealloc_tree_clean(output_dir):
            return False
        return (
            receipt.get("stop_verified") is False
            and receipt.get("cleanup_error") in (None, RETAINED_PENDING_CLEANUP_ERROR)
            and not receipt.get("transfers")
            and not receipt.get("contents_transfers")
            and not receipt.get("control_operations")
            and warmup is None
            and calib is None
            and upload is None
            and not has_source_delivery
            and not has_input_hashes
            and failure_class in {"ACCOUNTING", "UPLOAD_CAP", "SUPERVISOR"}
            and isinstance(receipt.get("error"), str)
            and bool(receipt["error"].strip())
        )

    if (
        not isinstance(endpoint, str)
        or not endpoint
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
    ):
        return False

    # Proof 2: Warmup failed before runtime.py launch.
    if warmup is not None and warmup.get("status") == "INCOMPLETE":
        return (
            isinstance(warmup.get("error"), str)
            and bool(warmup["error"].strip())
            and calib is None
            and upload is None
            and not has_source_delivery
            and not has_input_hashes
        )
    if warmup is None or warmup.get("status") != "PASS":
        return False

    # Proof 3: Post-warmup, pre-calibration startup transport failure.
    if calib is None and upload is None and not has_source_delivery and not has_input_hashes:
        ops = [
            op.get("kind")
            for op in (receipt.get("control_operations") or [])
            if isinstance(op, dict)
        ]
        return (
            "mkdir" not in ops
            and "source-fetch" not in ops
            and _has_failed_transport_record(receipt)
            and has_verified_transport_traceback(receipt)
        )

    # Proof 4: Rate calibration failed before staging.
    if calib is not None and calib.get("status") == "INCOMPLETE":
        return (
            isinstance(calib.get("error"), str)
            and bool(calib["error"].strip())
            and upload is None
            and not has_source_delivery
            and not has_input_hashes
        )
    if calib is None or calib.get("status") not in ("PASS", "WARNING_RATE_UNSTABLE"):
        return False

    # Proof 5: Staging failed before source-delivery/input-hashes/ready.json.
    if upload is not None and upload.get("staging_completed") is False:
        if has_source_delivery or has_input_hashes:
            return False
        if failure_class == "UPLOAD_CAP":
            return True
        # Proof 5b (RULING_L1_V1_R8): supervisor-side poll_rpc deadline during
        # source delivery (supervisor.py deliver_source -> poll_rpc "deadline reached
        # during telemetry recovery"); failed before source-delivery.json,
        # input-hashes.json, hardware.json and ready.json, so no unit could start.
        if _is_source_delivery_poll_deadline(receipt, output_dir, upload):
            return True
        return _has_failed_transport_record(receipt) and has_verified_transport_traceback(receipt)

    # Proof 6: Post-staging upload-cap gate before ready.json.
    if (
        upload is not None
        and upload.get("staging_completed") is not False
        and has_source_delivery
        and has_input_hashes
        and failure_class == "UPLOAD_CAP"
    ):
        return _UPLOAD_CAP_GATE_RE.search(str(receipt.get("error") or "")) is not None

    return False


def is_external_transport_compute_failure(receipt: dict, output_dir: Path | None = None) -> bool:
    """True iff compute started and stopped cleanly on verified transport, lease, or wall cap."""
    if not isinstance(output_dir, Path) or output_dir.is_symlink() or not output_dir.is_dir():
        return False
    if any(output_dir.glob("scientific-fault*")):
        return False
    if (
        receipt.get("status") != "INCOMPLETE"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
        or not isinstance(receipt.get("endpoint"), str)
        or not receipt.get("endpoint")
    ):
        return False
    if is_prestart_transport_failure(receipt, output_dir):
        return False
    if not (output_dir / "hardware.json").is_file() or (output_dir / "hardware.json").is_symlink():
        return False
    if not _audit_job_log_clean(output_dir):
        return False

    failure_class = str(receipt.get("failure_class") or "")

    # Subcase A: runtime.py self-fenced on lease expiry, release-credential expiry, or 2x wall cap
    # (never faulthandler 900s watchdog, ScientificFault rc=86, or child Python exception).
    if failure_class == "CHILD_NONZERO_EXIT":
        terminal = receipt.get("terminal")
        if (
            not isinstance(terminal, dict)
            or terminal.get("reason") not in GRACEFUL_COMPUTE_TERMINAL_REASONS
            or terminal.get("returncode") in (0, 86)
            or has_non_transport_fault(terminal.get("error"))
            or has_non_transport_fault(terminal.get("last_traceback_line"))
            or has_non_transport_fault(receipt.get("error"))
        ):
            return False
        polls_clean, _ = _audit_poll_files(output_dir, allow_graceful_compute_terminal=True)
        return polls_clean

    if receipt.get("terminal") is not None or (output_dir / "terminal.json").exists():
        return False
    if failure_class not in EXTERNAL_TRANSPORT_CLASSES:
        return False

    polls_clean, _ = _audit_poll_files(output_dir)
    if not polls_clean:
        return False

    for field in ("error", "traceback", "last_traceback_line"):
        if has_non_transport_fault(receipt.get(field)):
            return False

    if failure_class == "ACCOUNTING":
        err_text = str(receipt.get("error") or "")
        return bool(err_text) and _ACCOUNTING_PROVIDER_LAG_RE.search(err_text) is not None

    # Subcase B: Mac supervisor whole-session wall cap reached cleanly with zero child faults.
    if failure_class == "SUPERVISOR" and (
        "whole-session wall cap reached" in str(receipt.get("error") or "").lower()
    ):
        tb_texts = _unwrap_text_variants(receipt.get("traceback"))
        for tb in tb_texts:
            frame_paths = _FRAME_PATH_RE.findall(tb) + _RICH_FRAME_PATH_RE.findall(tb)
            if any(_FORBIDDEN_FRAME_RE.search(fp) is not None for fp in frame_paths):
                return False
            if frame_paths and not any(
                _ALLOWED_TRANSPORT_FRAME_RE.search(fp) is not None for fp in frame_paths
            ):
                return False
        return True

    if not _has_failed_transport_record(receipt):
        return False
    if has_verified_transport_traceback(receipt):
        return True
    err_blob = f"{receipt.get('error') or ''}\n{receipt.get('traceback') or ''}"
    return _RENEWAL_WS_TRUNCATED_RE.search(err_blob) is not None


def validate_hardware_amendment(
    path: Path | None = None,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Validate the frozen L4 hardware reassignment amendment for shards 08..10."""
    target = AMENDMENT_PATH if path is None else path
    expected_sha = AMENDMENT_SHA256 if expected_sha256 is None else expected_sha256
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"missing or symlinked hardware reassignment amendment: {target}")
    actual_sha = digest(target)
    if actual_sha != expected_sha:
        raise ValueError(
            f"hardware reassignment amendment SHA-256 mismatch: {actual_sha} != {expected_sha}"
        )
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable hardware reassignment amendment {target}: {exc}") from exc
    if (
        not isinstance(data, dict)
        or data.get("schema") != AMENDMENT_SCHEMA
        or data.get("amendment_id") != AMENDMENT_ID
        or data.get("namespace") != p.NAMESPACE
        or data.get("science") != p.identity()
        or data.get("registration_sha256") != digest(p.registration())
        or not isinstance(data.get("shards"), list)
        or not all(type(s) is int for s in data["shards"])
        or tuple(data["shards"]) != AMENDMENT_SHARDS
        or data.get("from_gpu") != "A100"
        or data.get("gpu") != AMENDMENT_GPU
        or type(data.get("workers")) is not int
        or data.get("workers") != AMENDMENT_WORKERS
        or type(data.get("worlds_per_shard")) is not int
        or data.get("worlds_per_shard") != p.A100_WORLDS_PER_SHARD
        or type(data.get("min_vcpu")) is not int
        or data.get("min_vcpu") != 12
        or data.get("device") != "cpu"
        or not isinstance(data.get("reason"), str)
        or not data["reason"].strip()
    ):
        raise ValueError(f"invalid hardware reassignment amendment payload: {target}")
    return {**data, "sha256": actual_sha, "path": str(target.resolve())}


def effective_shard_spec(
    shard: int,
    *,
    amendment_path: Path | None = None,
    expected_amendment_sha256: str | None = None,
) -> dict:
    """Return shard placement spec after applying validated AMENDMENT_L4_REASSIGN_01."""
    base = dict(_ORIG_PROTOCOL_SHARD_SPEC(shard))
    amendment = validate_hardware_amendment(
        amendment_path,
        expected_sha256=expected_amendment_sha256,
    )
    if shard in amendment["shards"]:
        base["gpu"] = amendment["gpu"]
        base["workers"] = amendment["workers"]
        base["amendment_id"] = amendment["amendment_id"]
        base["amendment_sha256"] = amendment["sha256"]
    return base


def validate_host_interrupted_incident(
    path: Path | None = None,
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Validate the single-use frozen host-interrupted compute attempt incident for shard 11."""
    target = INCIDENT_PATH if path is None else path
    expected_sha = INCIDENT_SHA256 if expected_sha256 is None else expected_sha256
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"missing or symlinked host-interrupted incident file: {target}")

    pkg_dir = p.ROOT / "research/watch6_l1_confirm"
    pkg_incidents = sorted(pkg_dir.glob("INCIDENT_*"))
    if (
        any(cand.is_symlink() for cand in pkg_incidents)
        or len(pkg_incidents) != 1
        or pkg_incidents[0].name != "INCIDENT_HOST_RESTART_01.json"
    ):
        raise ValueError(
            "single-use incident mechanism violated: expected exactly one INCIDENT_* file in "
            f"{pkg_dir}, found {[c.name for c in pkg_incidents]}"
        )
    resolved_target = target.resolve()
    extra_sibling_incidents = [
        cand
        for cand in sorted(target.parent.glob("INCIDENT_*"))
        if cand.is_symlink() or cand.resolve() != resolved_target
    ]
    if extra_sibling_incidents:
        raise ValueError(
            "single-use incident mechanism violated: multiple INCIDENT_* files in "
            f"{target.parent}: {[c.name for c in extra_sibling_incidents]}"
        )

    actual_sha = digest(target)
    if actual_sha != expected_sha:
        raise ValueError(
            f"host-interrupted incident SHA-256 mismatch: {actual_sha} != {expected_sha}"
        )
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable host-interrupted incident {target}: {exc}") from exc

    spec = effective_shard_spec(INCIDENT_SHARD)
    last_poll = data.get("last_poll") if isinstance(data, dict) else None
    sup_log = data.get("supervisor_log") if isinstance(data, dict) else None
    sess = data.get("sessions_evidence") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or set(data.keys()) != EXPECTED_INCIDENT_KEYS
        or data.get("schema") != INCIDENT_SCHEMA
        or data.get("incident_id") != INCIDENT_ID
        or data.get("namespace") != p.NAMESPACE
        or data.get("science") != p.identity()
        or data.get("registration_sha256") != digest(p.registration())
        or type(data.get("shard")) is not int
        or data.get("shard") != INCIDENT_SHARD
        or type(data.get("compute_completion_attempt")) is not int
        or data.get("compute_completion_attempt") != 1
        or data.get("classification") != "external_transport_compute"
        or not isinstance(data.get("attempt_output_dir"), str)
        or not Path(data["attempt_output_dir"]).is_absolute()
        or not isinstance(data.get("attempt_manifest_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", data["attempt_manifest_sha256"])
        or not isinstance(data.get("manifest_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", data["manifest_commit"])
        or data.get("gpu") != spec["gpu"]
        or type(data.get("workers")) is not int
        or data.get("workers") != spec["workers"]
        or not isinstance(data.get("hardware_json_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", data["hardware_json_sha256"])
        or not isinstance(last_poll, dict)
        or set(last_poll.keys()) != {"name", "sha256", "state_status", "updated_at"}
        or last_poll.get("name") != INCIDENT_LAST_POLL_NAME
        or not isinstance(last_poll.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", last_poll["sha256"])
        or last_poll.get("state_status") != "running"
        or type(last_poll.get("updated_at")) is not int
        or last_poll.get("updated_at") != INCIDENT_LAST_POLL_UPDATED_AT
        or type(data.get("poll_count")) is not int
        or data.get("poll_count") != INCIDENT_POLL_COUNT
        or not isinstance(sup_log, dict)
        or set(sup_log.keys()) != {"path", "sha256", "lines", "last_write_pt"}
        or not isinstance(sup_log.get("path"), str)
        or not Path(sup_log["path"]).is_absolute()
        or not isinstance(sup_log.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", sup_log["sha256"])
        or type(sup_log.get("lines")) is not int
        or sup_log.get("lines") != INCIDENT_POLL_COUNT
        or not isinstance(sup_log.get("last_write_pt"), str)
        or not sup_log["last_write_pt"].strip()
        or type(data.get("host_boot_epoch")) is not int
        or data.get("host_boot_epoch") != INCIDENT_HOST_BOOT_EPOCH
        or not isinstance(sess, dict)
        or set(sess.keys()) != {"captured_pt", "command", "output_sha256", "output_text"}
        or not isinstance(sess.get("captured_pt"), str)
        or not sess["captured_pt"].strip()
        or sess.get("command") != "colab sessions"
        or not isinstance(sess.get("output_text"), str)
        or not isinstance(sess.get("output_sha256"), str)
        or sess.get("output_sha256")
        != hashlib.sha256(sess["output_text"].encode("utf-8")).hexdigest()
        or not isinstance(data.get("exports_listing"), list)
        or len(data["exports_listing"]) != len(INCIDENT_EXPORTS_LISTING)
        or sorted(data["exports_listing"]) != sorted(INCIDENT_EXPORTS_LISTING)
        or type(data.get("world_outputs")) is not int
        or data.get("world_outputs") != 0
        or data.get("ruling") != INCIDENT_RULING
        or not isinstance(data.get("ruling_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", data["ruling_sha256"])
    ):
        raise ValueError(f"invalid host-interrupted incident payload: {target}")
    return {**data, "sha256": actual_sha, "path": str(resolved_target)}


def is_host_interrupted_compute_attempt(
    attempt_dir: Path | None,
    *,
    incident_path: Path | None = None,
    expected_incident_sha256: str | None = None,
) -> bool:
    """True iff attempt_dir matches every condition of the shard-11 host restart incident."""
    if not isinstance(attempt_dir, Path) or attempt_dir.is_symlink() or not attempt_dir.is_dir():
        return False
    try:
        incident = validate_host_interrupted_incident(
            incident_path,
            expected_sha256=expected_incident_sha256,
        )
    except ValueError:
        return False

    inc_out = Path(incident["attempt_output_dir"])
    if inc_out.is_symlink():
        return False
    try:
        resolved_attempt = attempt_dir.resolve(strict=True)
        if resolved_attempt != inc_out.resolve(strict=True):
            return False
    except OSError:
        return False

    if any(attempt_dir.glob("scientific-fault*")) or not _audit_job_log_clean(attempt_dir):
        return False

    for forbidden in (
        "receipt.json",
        "receipt.json.sha256",
        "receipt.json.complete",
        "terminal.json",
        "terminal.json.sha256",
        "terminal.json.complete",
        "state.json",
        "state.json.sha256",
        "state.json.complete",
    ):
        cand = attempt_dir / forbidden
        if cand.exists() or cand.is_symlink():
            return False

    hw_path = attempt_dir / "hardware.json"
    if hw_path.is_symlink() or not hw_path.is_file():
        return False
    if digest(hw_path) != incident["hardware_json_sha256"]:
        return False
    try:
        hw_data = json.loads(hw_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(hw_data, dict) or hw_data.get("ready") is not True:
        return False

    polls_clean, poll_count = _audit_poll_files(attempt_dir)
    if not polls_clean or poll_count != incident["poll_count"] or poll_count != INCIDENT_POLL_COUNT:
        return False
    poll_paths = sorted(attempt_dir.glob("poll-*.json"))
    if [p_path.name for p_path in poll_paths] != [
        f"poll-{i:06d}.json" for i in range(INCIDENT_POLL_COUNT)
    ]:
        return False
    last_poll_path = poll_paths[-1]
    last_poll_spec = incident["last_poll"]
    if (
        last_poll_path.name != last_poll_spec["name"]
        or digest(last_poll_path) != last_poll_spec["sha256"]
    ):
        return False
    try:
        last_poll_data = json.loads(last_poll_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    last_state = last_poll_data.get("state") if isinstance(last_poll_data, dict) else None
    raw_updated = last_state.get("updated_at") if isinstance(last_state, dict) else None
    if (
        not isinstance(last_poll_data, dict)
        or last_poll_data.get("terminal") is not None
        or not isinstance(last_state, dict)
        or last_state.get("status") != last_poll_spec["state_status"]
        or last_state.get("status") != "running"
        or isinstance(raw_updated, bool)
        or not isinstance(raw_updated, (int, float))
        or not math.isfinite(raw_updated)
        or int(raw_updated) != last_poll_spec["updated_at"]
    ):
        return False

    sup_spec = incident["supervisor_log"]
    sup_path = Path(sup_spec["path"])
    if sup_path.is_symlink() or not sup_path.is_file():
        return False
    if digest(sup_path) != sup_spec["sha256"]:
        return False
    try:
        sup_text = sup_path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return False
    if (
        len(sup_text.splitlines()) != sup_spec["lines"]
        or sup_spec["lines"] != INCIDENT_POLL_COUNT
        or "EXIT" in sup_text
        or "SCIENTIFIC_FAULT" in sup_text
        or "Traceback" in sup_text
        or "Timeout (0:15:00)" in sup_text
        or has_non_transport_fault(sup_text)
    ):
        return False

    exports_dir = attempt_dir / "exports"
    if exports_dir.is_symlink() or not exports_dir.is_dir():
        return False
    efficacy_dir = exports_dir / "efficacy"
    if efficacy_dir.exists() or efficacy_dir.is_symlink():
        return False
    export_names: list[str] = []
    for entry in sorted(exports_dir.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            return False
        export_names.append(entry.name)
    if len(export_names) != len(INCIDENT_EXPORTS_LISTING) or sorted(export_names) != sorted(
        incident["exports_listing"]
    ):
        return False

    for dirpath, dirnames, filenames in os.walk(attempt_dir, followlinks=False):
        cur_dir = Path(dirpath)
        for dname in dirnames:
            dpath = cur_dir / dname
            if dpath.is_symlink() or dname == "efficacy" or WORLD_NAME_RE.fullmatch(dname):
                return False
        for fname in filenames:
            fpath = cur_dir / fname
            if fpath.is_symlink() or fname == "efficacy" or WORLD_NAME_RE.fullmatch(fname):
                return False

    mf_path = attempt_dir / "manifest.json"
    if mf_path.is_symlink() or not mf_path.is_file():
        return False
    if digest(mf_path) != incident["attempt_manifest_sha256"]:
        return False
    parent_mf_path = attempt_dir.parent / "manifest.json"
    if parent_mf_path.exists() or parent_mf_path.is_symlink():
        if (
            parent_mf_path.is_symlink()
            or not parent_mf_path.is_file()
            or digest(parent_mf_path) != incident["attempt_manifest_sha256"]
        ):
            return False
    try:
        mf = json.loads(mf_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(mf, dict):
        return False

    spec = effective_shard_spec(INCIDENT_SHARD)
    req = mf.get("requirements", {})
    env = mf.get("env", {})
    binding = mf.get("watch6_l1_v1", {})
    try:
        mf_out_resolved = Path(str(mf.get("output_dir") or "")).resolve(strict=True)
        mf_shard = int(env.get("WATCH6_SHARD", -1))
        mf_shards = int(env.get("WATCH6_SHARDS", -1))
        mf_workers = int(env.get("WATCH6_WORKERS", -1))
    except (OSError, TypeError, ValueError):
        return False

    if (
        mf.get("commit") != incident["manifest_commit"]
        or mf_out_resolved != resolved_attempt
        or not isinstance(req, dict)
        or req.get("gpu") != incident["gpu"]
        or req.get("gpu") != spec["gpu"]
        or req.get("allowed_gpus") != [spec["gpu"]]
        or not isinstance(env, dict)
        or env.get("WATCH6_FIXTURE") != "0"
        or mf_shards != p.SHARDS
        or mf_shard != incident["shard"]
        or mf_shard != INCIDENT_SHARD
        or mf_workers != incident["workers"]
        or mf_workers != spec["workers"]
        or not isinstance(binding, dict)
        or binding.get("namespace") != p.NAMESPACE
        or binding.get("science") != p.identity()
        or binding.get("primary_allocation") != p.allocation()
        or binding.get("shard", INCIDENT_SHARD) != INCIDENT_SHARD
        or binding.get("gpu", spec["gpu"]) != spec["gpu"]
        or binding.get("workers", spec["workers"]) != spec["workers"]
        or any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage"))
    ):
        return False

    return True


def resolve_host_interrupted_attempt_dir(
    path: Path | None,
    *,
    incident_path: Path | None = None,
    expected_incident_sha256: str | None = None,
) -> Path | None:
    """Resolve a path (attempt_dir, attempt_dir/{receipt,manifest}.json, or incident file)
    to the verified host-interrupted attempt directory iff it validates.
    """
    if not isinstance(path, Path) or path.is_symlink():
        return None
    try:
        incident = validate_host_interrupted_incident(
            incident_path,
            expected_sha256=expected_incident_sha256,
        )
        inc_path = Path(incident["path"]).resolve(strict=True)
        inc_out = Path(incident["attempt_output_dir"]).resolve(strict=True)
    except (OSError, ValueError):
        return None

    candidate_dir: Path | None = None
    if path.is_dir():
        try:
            if path.resolve(strict=True) == inc_out:
                candidate_dir = inc_out
        except OSError:
            return None
    elif path.is_file():
        try:
            resolved_file = path.resolve(strict=True)
        except OSError:
            return None
        if resolved_file == inc_path or resolved_file == (inc_out / "manifest.json"):
            candidate_dir = inc_out
    elif (
        path.name in ("receipt.json", "manifest.json")
        and path.parent.is_dir()
        and not path.parent.is_symlink()
    ):
        try:
            if path.parent.resolve(strict=True) == inc_out:
                candidate_dir = inc_out
        except OSError:
            return None

    if candidate_dir is not None and is_host_interrupted_compute_attempt(
        candidate_dir,
        incident_path=incident_path,
        expected_incident_sha256=expected_incident_sha256,
    ):
        return candidate_dir
    return None


def verify_prior_receipts_lineage(
    shard: int,
    prior_receipts: list[dict],
    prestart_used: int,
    compute_attempt: int,
    total_dispatches: int,
) -> None:
    """Verify declared prior stopped receipts match retry counters and shard placement on Mac."""
    if not isinstance(prior_receipts, list):
        raise ValueError("prior_receipts must be a list")
    if total_dispatches != 1 + prestart_used + compute_attempt or len(prior_receipts) != (
        prestart_used + compute_attempt
    ):
        raise ValueError("retry-cap counters inconsistent with prior_receipts length")
    expected_spec = effective_shard_spec(shard)
    seen_paths: set[Path] = set()
    actual_prestart = 0
    actual_compute = 0
    for entry in prior_receipts:
        if not isinstance(entry, dict):
            raise ValueError("malformed prior_receipts entry")
        raw_ref = (
            entry.get("receipt_path")
            or entry.get("attempt_output_dir")
            or entry.get("incident_path")
            or ""
        )
        interrupted_dir = (
            resolve_host_interrupted_attempt_dir(Path(raw_ref))
            if isinstance(raw_ref, str) and raw_ref
            else None
        )
        if interrupted_dir is not None:
            if shard != INCIDENT_SHARD or interrupted_dir in seen_paths:
                raise ValueError(
                    f"duplicate or wrong-shard host-interrupted attempt in lineage: {raw_ref}"
                )
            seen_paths.add(interrupted_dir)
            incident = validate_host_interrupted_incident()
            if (
                not isinstance(entry.get("receipt_path"), str)
                or not entry["receipt_path"]
                or Path(entry["receipt_path"]).resolve() != Path(incident["path"]).resolve()
            ):
                raise ValueError(f"host-interrupted prior receipt_path mismatch: {raw_ref}")
            if entry.get("classification") != "external_transport_compute":
                raise ValueError(
                    "host-interrupted prior attempt requires classification "
                    f"external_transport_compute: {raw_ref}"
                )
            if entry.get("receipt_sha256") != incident["sha256"]:
                raise ValueError(f"host-interrupted prior SHA-256 mismatch: {raw_ref}")
            if entry.get("incident_id") != incident["incident_id"]:
                raise ValueError(f"host-interrupted incident_id mismatch: {raw_ref}")
            if entry.get("incident_sha256") != incident["sha256"]:
                raise ValueError(f"host-interrupted incident_sha256 mismatch: {raw_ref}")
            if (
                not isinstance(entry.get("attempt_output_dir"), str)
                or not entry["attempt_output_dir"]
                or Path(entry["attempt_output_dir"]).resolve() != interrupted_dir
            ):
                raise ValueError(f"host-interrupted attempt_output_dir mismatch: {raw_ref}")
            actual_compute += 1
            continue

        r_path = Path(entry.get("receipt_path", ""))
        if not r_path.is_absolute() or r_path.is_symlink() or not r_path.is_file():
            raise ValueError(f"missing or symlinked prior receipt in lineage: {r_path}")
        resolved = r_path.resolve()
        if resolved in seen_paths or digest(resolved) != entry.get("receipt_sha256"):
            raise ValueError(f"duplicate or hash-mismatched prior receipt: {r_path}")
        seen_paths.add(resolved)
        receipt = json.loads(resolved.read_text(encoding="utf-8"))
        m_path = resolved.parent / "manifest.json"
        if (
            m_path.is_symlink()
            or not m_path.is_file()
            or digest(m_path) != receipt.get("manifest_sha256")
        ):
            raise ValueError(f"prior receipt manifest_sha256 mismatch: {r_path}")
        m_data = json.loads(m_path.read_text(encoding="utf-8"))
        m_bind = m_data.get("watch6_l1_v1", {})
        m_req = m_data.get("requirements", {})
        if (
            m_bind.get("namespace") != p.NAMESPACE
            or m_bind.get("science") != p.identity()
            or m_data.get("env", {}).get("WATCH6_FIXTURE") != "0"
            or int(m_data.get("env", {}).get("WATCH6_SHARD", -1)) != shard
            or int(m_data.get("env", {}).get("WATCH6_WORKERS", -1)) != expected_spec["workers"]
            or m_req.get("gpu") != expected_spec["gpu"]
            or m_req.get("allowed_gpus") != [expected_spec["gpu"]]
        ):
            raise ValueError(f"foreign shard/science/GPU in prior receipt lineage: {r_path}")
        declared_kind = entry.get("classification")
        if is_prestart_transport_failure(receipt, resolved.parent):
            if declared_kind != "prestart":
                raise ValueError(
                    f"prior receipt classification mismatch (expected prestart): {r_path}"
                )
            actual_prestart += 1
        elif is_external_transport_compute_failure(receipt, resolved.parent):
            if declared_kind != "external_transport_compute":
                raise ValueError(
                    "prior receipt classification mismatch "
                    f"(expected external_transport_compute): {r_path}"
                )
            actual_compute += 1
        else:
            raise ValueError(f"forbidden failure class in prior_receipts lineage: {r_path}")
    if actual_prestart != prestart_used or actual_compute != compute_attempt:
        raise ValueError("prior_receipts classifications do not match retry counters")


def verify_bound_receipt(receipt_path: Path) -> tuple[dict, dict]:
    """Verify a stopped Watch6-L1-V1 main-shard supervisor receipt and its executed manifest."""
    receipt_path = receipt_path.resolve(strict=True)
    if receipt_path.is_symlink():
        raise ValueError(f"refusing symlinked receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_path = receipt_path.parent / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or digest(manifest_path) != receipt.get("manifest_sha256")
    ):
        raise ValueError(f"receipt manifest_sha256 binding mismatch: {receipt_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if receipt.get("job_id") != manifest.get("job_id") or receipt.get("commit") != manifest.get(
        "commit"
    ):
        raise ValueError(f"receipt job_id/commit mismatch against manifest.json: {receipt_path}")

    prealloc_unassigned = (
        receipt.get("endpoint") is None
        and receipt.get("stop_verified") is False
        and receipt.get("cleanup_error") in (None, RETAINED_PENDING_CLEANUP_ERROR)
        and is_prestart_transport_failure(receipt, receipt_path.parent)
    )
    if not prealloc_unassigned and (
        receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None
    ):
        raise ValueError(f"runtime stop not verified or cleanup_error present: {receipt_path}")

    binding = manifest.get("watch6_l1_v1", {})
    env = manifest.get("env", {})
    shard = int(env.get("WATCH6_SHARD", -1))
    if (
        canary.is_canary(manifest)
        or env.get("WATCH6_FIXTURE") != "0"
        or binding.get("namespace") != p.NAMESPACE
        or binding.get("science") != p.identity()
        or binding.get("primary_allocation") != p.allocation()
        or any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage"))
        or int(env.get("WATCH6_SHARDS", -1)) != p.SHARDS
        or shard not in range(p.SHARDS)
    ):
        raise ValueError(
            "foreign, fixture, reserve, or canary receipt rejected "
            f"from scientific ledger: {receipt_path}"
        )
    expected_spec = effective_shard_spec(shard)
    req = manifest.get("requirements", {})
    if (
        req.get("gpu") != expected_spec["gpu"]
        or req.get("allowed_gpus") != [expected_spec["gpu"]]
        or int(env.get("WATCH6_WORKERS", -1)) != expected_spec["workers"]
    ):
        raise ValueError(f"shard {shard} GPU/worker placement mismatch in {receipt_path}")
    if "gpu" in binding and binding.get("gpu") != expected_spec["gpu"]:
        raise ValueError(f"shard {shard} watch6_l1_v1.gpu mismatch in {receipt_path}")
    if "workers" in binding and binding.get("workers") != expected_spec["workers"]:
        raise ValueError(f"shard {shard} watch6_l1_v1.workers mismatch in {receipt_path}")
    if "amendment_id" in expected_spec:
        if (
            "hardware_amendment_id" in binding
            and binding.get("hardware_amendment_id") != expected_spec["amendment_id"]
        ) or (
            "hardware_amendment_sha256" in binding
            and binding.get("hardware_amendment_sha256") != expected_spec["amendment_sha256"]
        ):
            raise ValueError(f"shard {shard} hardware amendment binding mismatch in {receipt_path}")
    elif "hardware_amendment_id" in binding or "hardware_amendment_sha256" in binding:
        raise ValueError(f"un-amended shard {shard} carries hardware amendment in {receipt_path}")
    return manifest, receipt


def completed_ledger(path: Path) -> list[dict]:
    """Read only identity/size/hash metadata from the Mac-verified Watch6-L1-V1 ledger."""
    rows = read_ledger(path, p.NAMESPACE, "watch6", {"science": p.identity()})
    if any(row["index"] not in range(p.COUNT) for row in rows):
        raise ValueError("foreign Watch6-L1-V1 allocation index in verified ledger")
    return rows


def admit(manifest: dict, *, pending_qualification: bool = False) -> dict:
    """Validate Watch6-L1-V1 manifest allocation, GPU/W placement, <=8/<=60s uploads, 2x caps."""
    binding = manifest["watch6_l1_v1"]
    if manifest.get("pending_qualification") and not pending_qualification:
        raise ValueError("main packet awaits genuine passing same-pin canary; finalize first")
    if pending_qualification and manifest.get("pending_qualification") is not True:
        raise ValueError("pending validation only accepts explicitly pending packets")
    if any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage")):
        raise ValueError("reserve seed substitution is forbidden in Watch6-L1-V1")
    if (
        binding.get("namespace") != p.NAMESPACE
        or binding.get("science") != p.identity()
        or binding.get("primary_allocation") != p.allocation()
    ):
        raise ValueError("foreign Watch6-L1-V1 source/namespace/allocation")
    if manifest.get("script") != p.SCRIPT or manifest.get("device") != "cpu":
        raise ValueError("wrong Watch6-L1-V1 entrypoint or device")
    if manifest.get("lease_seconds") != 600:
        raise ValueError("Watch6-L1-V1 requires lease_seconds == 600")

    probe = manifest.get("canary", {}).get("mode") == "probe"
    env = manifest.get("env", {})
    shard = int(env.get("WATCH6_SHARD", -1))
    if shard not in range(p.SHARDS) or int(env.get("WATCH6_SHARDS", -1)) != p.SHARDS:
        raise ValueError(f"invalid WATCH6_SHARD/WATCH6_SHARDS ({shard}/{env.get('WATCH6_SHARDS')})")

    spec = effective_shard_spec(shard)
    expected_gpu = spec["gpu"]
    expected_workers = spec["workers"]
    req = manifest.get("requirements", {})
    if (
        req.get("gpu") != expected_gpu
        or req.get("allowed_gpus") != [expected_gpu]
        or "prefer" in req
        or req.get("min_vcpu", 0) < 12
        or int(env.get("WATCH6_WORKERS", -1)) != expected_workers
    ):
        raise ValueError(
            f"shard {shard} requires exact >=12 vCPU {expected_gpu} W={expected_workers} placement"
        )
    if "gpu" in binding and binding.get("gpu") != expected_gpu:
        raise ValueError(f"shard {shard} watch6_l1_v1.gpu mismatch: {binding.get('gpu')!r}")
    if "workers" in binding and binding.get("workers") != expected_workers:
        raise ValueError(f"shard {shard} watch6_l1_v1.workers mismatch: {binding.get('workers')!r}")
    if "indices" in binding and binding.get("indices") != spec["indices"]:
        raise ValueError(f"shard {shard} watch6_l1_v1.indices mismatch")
    if "amendment_id" in spec and not probe:
        if (
            binding.get("hardware_amendment_id") != spec["amendment_id"]
            or binding.get("hardware_amendment_sha256") != spec["amendment_sha256"]
        ):
            raise ValueError(f"shard {shard} hardware amendment binding mismatch")
    elif "hardware_amendment_id" in binding or "hardware_amendment_sha256" in binding:
        raise ValueError(f"un-amended shard {shard} cannot carry hardware amendment binding")

    expected_base = CANARY_BASE_MINUTES if probe else BASE_MINUTES_BY_GPU[expected_gpu]
    base_minutes = binding.get("base_minutes")
    if (
        type(base_minutes) is not int
        or base_minutes != expected_base
        or not 0 < base_minutes <= MAX_BASE_MINUTES
    ):
        raise ValueError(f"invalid base_minutes envelope: {base_minutes!r} != {expected_base}")
    if (
        manifest.get("timeout_minutes") != 2 * base_minutes
        or req.get("est_minutes") != base_minutes
    ):
        raise ValueError("requires pre-data 2x wall guard matching est_minutes")

    budget = manifest.get("budget", {})
    rate = budget.get("rate_units_per_hour")
    expected_rate = PRICED_CEILING_RATES_U_PER_H[expected_gpu]
    if (
        type(rate) not in (int, float)
        or not math.isfinite(rate)
        or not math.isclose(float(rate), expected_rate, rel_tol=1e-9)
    ):
        raise ValueError(f"budget requires priced ceiling rate {expected_rate} for {expected_gpu}")
    expected_base_units = base_minutes / 60.0 * float(rate)
    if not math.isclose(
        budget.get("unit_cap", -1.0), 2.0 * expected_base_units, rel_tol=1e-9
    ) or not math.isclose(
        budget.get("reserve_units", -1.0), 2.0 * expected_base_units * 0.2, rel_tol=1e-9
    ):
        raise ValueError("requires pre-data 2x pricing envelope")

    prestart_used = binding.get("prestart_retries_used", 0)
    compute_attempt = binding.get("compute_completion_attempt", 0)
    total_dispatches = binding.get("total_dispatches_for_shard", 1)
    top_completion = manifest.get("completion_attempt", 0)
    phase = binding.get("phase")
    expected_phase = "canary" if probe else "primary"
    if (
        phase != expected_phase
        or type(prestart_used) is not int
        or prestart_used < 0
        or type(compute_attempt) is not int
        or not 0 <= compute_attempt <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
        or type(top_completion) is not int
        or not 0 <= top_completion <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
        or compute_attempt + top_completion > p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
        or type(total_dispatches) is not int
        or total_dispatches != 1 + prestart_used + compute_attempt
    ):
        raise ValueError("retry or phase counters violate pre-registered Watch6-L1-V1 caps")
    if probe and (
        prestart_used != 0 or compute_attempt != 0 or total_dispatches != 1 or top_completion != 0
    ):
        raise ValueError("canary probe must have zero shard retry counters")

    if env.get("WATCH6_FIXTURE") != ("1" if probe else "0"):
        raise ValueError("canary/main fixture isolation mismatch")
    if env.get("SNAKE_WATCH_SOURCE") != "live":
        raise ValueError("requires explicit SNAKE_WATCH_SOURCE=live")
    if int(env.get("WATCH6_BASE_MINUTES", -1)) != base_minutes:
        raise ValueError("runtime base_minutes env mismatch")
    if manifest.get("required_exports") != ["efficacy/operational.json", "performance.json"]:
        raise ValueError("required_exports mismatch")
    if manifest.get("resume") != {
        "unit_kind": "paired-world",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise ValueError("resume policy mismatch")

    prior_receipts = binding.get("prior_receipts", [])
    if probe:
        if prior_receipts != []:
            raise ValueError("canary probe cannot carry shard prior_receipts")
    else:
        verify_prior_receipts_lineage(
            shard,
            prior_receipts,
            prestart_used,
            compute_attempt,
            total_dispatches,
        )

    for item in manifest["inputs"]:
        in_path = Path(item["path"])
        if in_path.is_symlink() or digest(in_path) != item["sha256"]:
            raise ValueError(f"input digest mismatch or symlink: {item['dest']}")
    upload_report = check_uploads(manifest)
    meta_item = next(i for i in manifest["inputs"] if i["dest"] == METADATA)
    if meta_item["sha256"] != env.get("WATCH6_METADATA_SHA256"):
        raise ValueError("runtime metadata SHA differs from uploaded manifest input")

    with tempfile.TemporaryDirectory(prefix="watch6-l1-admission-") as temporary:
        root = Path(temporary)
        shutil.copyfile(meta_item["path"], root / METADATA)
        stage_metadata(root, env["WATCH6_METADATA_SHA256"], materialize_checkpoints=False)
        verify_runtime(root, env)

    if not probe and not pending_qualification:
        canary.check(manifest, p.ROOT)
    if not probe:
        report = json.loads(env["WATCH6_MEASUREMENTS"])
        verify_report_evidence(report)
    return upload_report


def verify_runtime(root: Path, env: dict) -> dict:
    """Execute-time guard validating contract, registration hash, mirror, and 18-shard layout."""
    contract = json.loads((root / "runtime-contract.json").read_text(encoding="utf-8"))
    expected_contract = {
        "schema": "watch6-l1-runtime-contract/v1",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "primary_allocation": p.allocation(),
    }
    if contract != expected_contract:
        raise ValueError("runtime contract source/allocation mismatch")
    if env.get("WATCH6_SCIENCE") != p.identity() or env.get("SNAKE_WATCH_SOURCE") != "live":
        raise ValueError("runtime science or SNAKE_WATCH_SOURCE env mismatch")

    shards = int(env.get("WATCH6_SHARDS", "0"))
    shard = int(env.get("WATCH6_SHARD", "-1"))
    workers = int(env.get("WATCH6_WORKERS", "0"))
    if (
        env.get("WATCH6_FIXTURE") not in ("0", "1")
        or shards != p.SHARDS
        or shard not in range(shards)
    ):
        raise ValueError("runtime shard/fixture mismatch")
    spec = effective_shard_spec(shard)
    if workers != spec["workers"]:
        raise ValueError(f"shard {shard} requires W={spec['workers']}, got W={workers}")

    completed_ledger(root / "verified-ledger.json")

    verify_immutable(root / "mac-mirror.json")
    validate_mirror(json.loads((root / "mac-mirror.json").read_text(encoding="utf-8")))

    protocol_meta = json.loads((root / "judge-approval.json").read_text(encoding="utf-8"))
    if (
        protocol_meta.get("schema") != "watch6-l1-protocol-binding/v1"
        or protocol_meta.get("implementer") != "Argon"
        or protocol_meta.get("judge") not in ("Astra", "Fable")
        or protocol_meta.get("namespace") != p.NAMESPACE
        or protocol_meta.get("science") != p.identity()
        or protocol_meta.get("registration_sha256") != digest(p.registration())
    ):
        raise ValueError("uploaded protocol/registration binding mismatch")

    if env["WATCH6_FIXTURE"] == "0":
        if "WATCH6_MEASUREMENTS" not in env:
            raise ValueError("main shard requires empirical cost measurements")
        report = json.loads(env["WATCH6_MEASUREMENTS"])
        verify_immutable(root / "cost-evidence.json")
        if json.loads((root / "cost-evidence.json").read_text(encoding="utf-8")) != report:
            raise ValueError("WATCH6_MEASUREMENTS differs from uploaded cost-evidence.json")
        forecast = empirical_forecast(report)
        gpu_env = forecast["per_gpu_envelopes"][spec["gpu"]]
        if int(env.get("WATCH6_BASE_MINUTES", "0")) != gpu_env["base_minutes"]:
            raise ValueError("runtime base_minutes differs from empirical per-GPU forecast")
    return contract


def runtime_admit(
    *,
    workers: int | None = None,
    shards: int | None = None,
    shard: int | None = None,
) -> dict:
    """Refuse direct CLI controls that differ from admitted manifest controls."""
    env = dict(os.environ)
    contract = verify_runtime(Path(env["COLAB_INPUT_DIR"]), env)
    if env["WATCH6_FIXTURE"] == "0" and (os.cpu_count() or 0) < 12:
        raise ValueError("main shards require >=12 vCPU at execution time")
    for value, key in (
        (workers, "WATCH6_WORKERS"),
        (shards, "WATCH6_SHARDS"),
        (shard, "WATCH6_SHARD"),
    ):
        if value is not None and value != int(env[key]):
            raise ValueError("runtime CLI differs from admitted manifest")
    return contract


_ORIG_PROTOCOL_SHARD_SPEC = p.shard_spec


def _install_resume_placement_hook() -> None:
    from research.watch6_l1_confirm import resume as _resume_mod

    orig_verify = _resume_mod.verify_bound_receipt
    orig_code = getattr(orig_verify, "__code__", None)

    def _resume_aware_shard_spec(shard: int) -> dict:
        frame = sys._getframe(1)
        if orig_code is not None and frame.f_code is orig_code:
            return effective_shard_spec(shard)
        return _ORIG_PROTOCOL_SHARD_SPEC(shard)

    p.shard_spec = _resume_aware_shard_spec  # type: ignore[assignment]
    _resume_mod.verify_bound_receipt = verify_bound_receipt


_install_resume_placement_hook()
