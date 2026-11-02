"""Fail-closed admission and supervisor failure classification for Watch6-L1-V1."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
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
    expected_spec = p.shard_spec(shard)
    seen_paths: set[Path] = set()
    actual_prestart = 0
    actual_compute = 0
    for entry in prior_receipts:
        if not isinstance(entry, dict):
            raise ValueError("malformed prior_receipts entry")
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

    spec = p.shard_spec(shard)
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
    spec = p.shard_spec(shard)
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
