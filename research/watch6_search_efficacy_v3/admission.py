"""Fail-closed admission and real-supervisor failure classification for Watch6 V3."""

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
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.costs import empirical_forecast, verify_report_evidence
from research.watch6_search_efficacy_v3.mirror import validate as validate_mirror
from research.watch6_search_efficacy_v3.uploads import METADATA, check_uploads, stage_metadata
from tools.ops.package_spec import digest, read_ledger, verify_immutable

EXTERNAL_TRANSPORT_CLASSES = {
    "RPC_EXEC_POLL_TIMEOUT",
    "SUPERVISOR",
    "TRANSPORT_TIMEOUT",
    "RUNTIME_LOST",
}

FORBIDDEN_TIMEOUT_SUBSTRINGS = (
    "WALL_TIMEOUT",
    "WHOLE-SESSION WALL CAP",
    "WHOLE-SESSION DEADLINE",
    "WALL CAP REACHED",
    "TIMEOUT_MINUTES",
    "EXCEEDED WALL",
    "EXCEEDED TIMEOUT",
    "WATCHDOG",
    "FAULTHANDLER",
    "DUMP_TRACEBACK_LATER",
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
    r"|\bpytest\b"
    r"|test.*fail"
    r"|child_nonzero_exit"
    r"|child\s+(?:process\s+)?exit"
    r"|kernel error despite cli exit zero"
    r"|colab-cli runtime:"
    r"|runtime supervisor failure"
    r"|runtime unassign failed"
    r"|out of memory"
    r"|\boom\b"
    r"|\bcuda\b"
    r"|\bcudnn\b"
    r"|\bcublas\b"
    r"|segfault"
    r"|segmentation fault"
    r"|\bsig(?:segv|abrt|bus|ill|fpe|kill|usr1)\b"
    r"|wall_timeout"
    r"|whole-session wall cap"
    r"|whole-session deadline"
    r"|wall cap reached"
    r"|timeout_minutes"
    r"|exceeded wall"
    r"|exceeded timeout"
    r"|\bwatchdog\b"
    r"|\bfaulthandler\b"
    r"|dump_traceback_later"
    r"|timeout\s*\(0:15:00\)"
    r"|account-wide assignment cap exceeded during job"
    r"|\b(?:AssertionError|ValueError|TypeError|KeyError|IndexError|AttributeError"
    r"|ZeroDivisionError|ArithmeticError|FloatingPointError|OverflowError"
    r"|ImportError|ModuleNotFoundError|NameError|UnboundLocalError"
    r"|SyntaxError|IndentationError|NotImplementedError|RecursionError"
    r"|MemoryError|BufferError|LookupError|SystemError"
    r"|FileNotFoundError|FileExistsError|IsADirectoryError|NotADirectoryError"
    r"|PermissionError|JSONDecodeError|UnicodeError|UnicodeDecodeError|UnicodeEncodeError"
    r"|KeyboardInterrupt|SystemExit)\b"
    r"|(?:research/watch6_search_efficacy|research/bl_served_search|research/t1_commitment"
    r"|/src/|\"src/|'src/|/web/|\"web/|'web/|job\.sh)"
    r")",
    re.IGNORECASE,
)

_FRAME_PATH_RE = re.compile(r'File "([^"]+)", line \d+')
_RICH_FRAME_PATH_RE = re.compile(r"([^\s:]+\.py):\d+\s+in\s+")
_FORBIDDEN_FRAME_RE = re.compile(
    r"(?:^|/)(?:src|web|research/watch6_search_efficacy(?:_v3)?|research/bl_served_search"
    r"|research/t1_commitment[^/]*)(?:/|$)|job\.sh$",
    re.IGNORECASE,
)
_ALLOWED_TRANSPORT_FRAME_RE = re.compile(
    r"(?:colab|httpx|httpcore|urllib|requests|http/client\.py|ssl\.py|socket\.py"
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
    r"|SessionLostError|GuardError)"
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


def _unwrap_text_variants(raw: object) -> list[str]:
    """Return raw text plus unescaped inner string arguments when wrapped via repr(exc)."""
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


def has_non_transport_fault(text: object) -> bool:
    """True iff error/log text carries any scientific, integrity, child, or wall-timeout fault."""
    for candidate in _unwrap_text_variants(text):
        if _NON_TRANSPORT_FAULT_RE.search(candidate) is not None:
            return True
        upper = candidate.upper()
        if any(token in upper for token in FORBIDDEN_TIMEOUT_SUBSTRINGS):
            return True
    return False


def _has_failed_transport_record(receipt: dict) -> bool:
    """True iff supervisor recorded at least one failed CLI/RPC/transfer operation."""
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
    """Read an optional supervisor measurement JSON, failing closed on corruption or faults."""
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


def _audit_poll_files(output_dir: Path) -> tuple[bool, int]:
    """Audit real supervisor poll-*.json snapshots for zero child/supervisor faults."""
    if any(output_dir.glob("telemetry-*.json")):
        for legacy in sorted(output_dir.glob("telemetry-*.json")):
            try:
                raw_legacy = legacy.read_text(encoding="utf-8")
            except OSError:
                return False, 0
            if "Traceback (most recent call last)" in raw_legacy or has_non_transport_fault(
                raw_legacy
            ):
                return False, 0

    poll_paths = sorted(output_dir.glob("poll-*.json"))
    for poll_path in poll_paths:
        if poll_path.is_symlink() or not poll_path.is_file():
            return False, len(poll_paths)
        try:
            poll = json.loads(poll_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, len(poll_paths)
        if not isinstance(poll, dict) or poll.get("terminal") is not None:
            return False, len(poll_paths)
        state = poll.get("state")
        if (
            not isinstance(state, dict)
            or state.get("status") not in ("starting", "running")
            or state.get("returncode") is not None
            or state.get("reason") is not None
            or state.get("error") is not None
            or state.get("unassign_error") is not None
        ):
            return False, len(poll_paths)
        for tail_key in ("job_log_tail", "supervisor_log_tail"):
            tail = str(poll.get(tail_key) or "")
            if (
                "Traceback (most recent call last)" in tail
                or _GENERIC_LOG_EXCEPTION_RE.search(tail) is not None
                or has_non_transport_fault(tail)
            ):
                return False, len(poll_paths)
    return True, len(poll_paths)


def is_prestart_transport_failure(receipt: dict, output_dir: Path | None) -> bool:
    """True iff positive supervisor.py evidence proves failure occurred pre-ready.json (line 1904).

    Missing evidence, presence of hardware.json or any poll-*.json, post-staging non-UPLOAD_CAP
    failures at the line-1904 ready.json boundary, or any scientific/integrity fault => False.
    """
    if not isinstance(output_dir, Path) or not output_dir.is_dir():
        return False
    if receipt.get("status") != "INCOMPLETE":
        return False
    if receipt.get("exports") or receipt.get("terminal") is not None:
        return False
    if (output_dir / "terminal.json").exists() or (output_dir / "terminal.json").is_symlink():
        return False
    exports_dir = output_dir / "exports"
    if exports_dir.is_symlink() or (exports_dir.exists() and any(exports_dir.rglob("*"))):
        return False

    # Any post-ready.json supervisor artifact proves execution reached or entered compute phase.
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
        if has_non_transport_fault(receipt.get(field)):
            return False

    ok_w, warmup = _read_optional_measurement(output_dir / "warmup-measurement.json")
    ok_c, calib = _read_optional_measurement(output_dir / "rate-calibration.json")
    ok_u, upload = _read_optional_measurement(output_dir / "upload-measurement.json")
    if not (ok_w and ok_c and ok_u):
        return False

    has_source_delivery = (output_dir / "source-delivery.json").exists()
    has_input_hashes = (output_dir / "input-hashes.json").exists()
    endpoint = receipt.get("endpoint")

    # Positive Proof 1: Pre-allocation refusal before any VM endpoint was assigned.
    if endpoint is None:
        return (
            receipt.get("stop_verified") is False
            and receipt.get("cleanup_error") is None
            and not receipt.get("transfers")
            and not receipt.get("contents_transfers")
            and not receipt.get("control_operations")
            and warmup is None
            and calib is None
            and upload is None
            and not has_source_delivery
            and not has_input_hashes
            and not (output_dir / "usage-allocated.json").exists()
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

    # Positive Proof 2: Warmup failed before runtime.py launch (supervisor.py:737).
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

    # Positive Proof 3: Post-warmup, pre-calibration startup transport failure (lines 1645-1719).
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

    # Positive Proof 4: Rate calibration failed before staging (supervisor.py:1574).
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

    # Positive Proof 5: Staging failed before source-delivery/input-hashes/ready.json (line 1838).
    if upload is not None and upload.get("staging_completed") is False:
        if has_source_delivery or has_input_hashes:
            return False
        if failure_class == "UPLOAD_CAP":
            return True
        return _has_failed_transport_record(receipt) and has_verified_transport_traceback(receipt)

    # Positive Proof 6: Post-staging upload-cap gate before line 1904 ready.json (lines 1865-1871).
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
    """True iff compute started (hardware.json) and stopped purely on verified transport loss.

    Requires real on-disk supervisor evidence (hardware.json, audited poll-*.json free of child or
    supervisor faults, failed transport call record, and explicit host/CLI transport traceback) and
    rejects any error/log text carrying scientific, integrity, child, or wall-timeout faults.
    """
    if not isinstance(output_dir, Path) or not output_dir.is_dir():
        return False
    if (
        receipt.get("status") != "INCOMPLETE"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
        or not isinstance(receipt.get("endpoint"), str)
        or not receipt.get("endpoint")
    ):
        return False
    if receipt.get("terminal") is not None or (output_dir / "terminal.json").exists():
        return False
    if is_prestart_transport_failure(receipt, output_dir):
        return False
    if not (output_dir / "hardware.json").is_file() or (output_dir / "hardware.json").is_symlink():
        return False

    failure_class = str(receipt.get("failure_class") or "")
    if failure_class not in EXTERNAL_TRANSPORT_CLASSES:
        return False

    polls_clean, _ = _audit_poll_files(output_dir)
    if not polls_clean:
        return False

    if not _has_failed_transport_record(receipt):
        return False
    return has_verified_transport_traceback(receipt)


def verify_prior_receipts_lineage(
    shard: int,
    prior_receipts: list[dict],
    prestart_used: int,
    compute_attempt: int,
    total_dispatches: int,
) -> None:
    """Verify declared prior stopped receipts match retry counters and cap on Mac host."""
    if not isinstance(prior_receipts, list):
        raise ValueError("prior_receipts must be a list")
    if total_dispatches != 1 + prestart_used + compute_attempt or len(prior_receipts) != (
        prestart_used + compute_attempt
    ):
        raise ValueError("retry-cap counters inconsistent with prior_receipts length")
    seen_paths: set[Path] = set()
    actual_prestart = 0
    actual_compute = 0
    for entry in prior_receipts:
        if not isinstance(entry, dict):
            raise ValueError("malformed prior_receipts entry")
        r_path = Path(entry.get("receipt_path", ""))
        if not r_path.is_absolute() or not r_path.is_file():
            raise ValueError(f"missing prior receipt in lineage: {r_path}")
        resolved = r_path.resolve()
        if resolved in seen_paths or digest(resolved) != entry.get("receipt_sha256"):
            raise ValueError(f"duplicate or hash-mismatched prior receipt: {r_path}")
        seen_paths.add(resolved)
        receipt = json.loads(resolved.read_text(encoding="utf-8"))
        m_path = resolved.parent / "manifest.json"
        if not m_path.is_file() or digest(m_path) != receipt.get("manifest_sha256"):
            raise ValueError(f"prior receipt manifest_sha256 mismatch: {r_path}")
        m_data = json.loads(m_path.read_text(encoding="utf-8"))
        m_bind = m_data.get("watch6_v3", {})
        if (
            m_bind.get("namespace") != p.NAMESPACE
            or m_bind.get("science") != p.identity()
            or m_data.get("env", {}).get("WATCH6_FIXTURE") != "0"
            or int(m_data.get("env", {}).get("WATCH6_SHARD", -1)) != shard
        ):
            raise ValueError(f"foreign shard/science in prior receipt lineage: {r_path}")
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
    """Read only identity/size/hash metadata from the Mac-verified V3 ledger."""
    rows = read_ledger(path, p.NAMESPACE, "watch6", {"science": p.identity()})
    if any(row["index"] not in range(p.COUNT) for row in rows):
        raise ValueError("foreign V3 allocation index in verified ledger")
    return rows


def admit(manifest: dict, *, pending_qualification: bool = False) -> dict:
    """Check controller, V3 primary allocation, <=12/<=90s uploads, forecast, and 2x envelopes."""
    binding = manifest["watch6_v3"]
    if manifest.get("pending_qualification") and not pending_qualification:
        raise ValueError("main packet awaits genuine passing same-pin canary; finalize first")
    if pending_qualification and manifest.get("pending_qualification") is not True:
        raise ValueError("pending validation only accepts explicitly pending packets")
    if any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage")):
        raise ValueError("reserve seed substitution is forbidden in Watch6 V3")
    if (
        binding.get("namespace") != p.NAMESPACE
        or binding.get("science") != p.identity()
        or binding.get("primary_allocation") != p.allocation()
    ):
        raise ValueError("foreign V3 source/namespace/allocation")
    if manifest.get("script") != p.SCRIPT or manifest.get("device") != "cpu":
        raise ValueError("wrong V3 entrypoint or device")
    req = manifest.get("requirements", {})
    if req.get("gpu") != "L4" or req.get("allowed_gpus") != ["L4"] or req.get("min_vcpu", 0) < 12:
        raise ValueError("only registered >=12 vCPU L4 CPU placement is admitted")
    base_minutes = binding.get("base_minutes")
    if type(base_minutes) is not int or not 0 < base_minutes <= 660:
        raise ValueError("invalid base_minutes envelope")
    if (
        manifest.get("timeout_minutes") != 2 * base_minutes
        or req.get("est_minutes") != base_minutes
    ):
        raise ValueError("requires pre-data 2x wall guard matching est_minutes")
    budget = manifest.get("budget", {})
    rate = budget.get("rate_units_per_hour")
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("budget requires positive finite rate_units_per_hour")
    expected_base_units = base_minutes / 60 * float(rate)
    if not math.isclose(
        budget.get("unit_cap", -1.0), 2 * expected_base_units, rel_tol=1e-9
    ) or not math.isclose(
        budget.get("reserve_units", -1.0), 2 * expected_base_units * 0.2, rel_tol=1e-9
    ):
        raise ValueError("requires pre-data 2x pricing envelope")

    probe = manifest.get("canary", {}).get("mode") == "probe"
    prestart_used = binding.get("prestart_retries_used", 0)
    compute_attempt = binding.get("compute_completion_attempt", 0)
    total_dispatches = binding.get("total_dispatches_for_shard", 1)
    phase = binding.get("phase")
    expected_phase = "canary" if probe else "primary"
    if (
        phase != expected_phase
        or type(prestart_used) is not int
        or prestart_used < 0
        or type(compute_attempt) is not int
        or not 0 <= compute_attempt <= p.MAX_COMPUTE_COMPLETIONS_PER_SHARD
        or type(total_dispatches) is not int
        or total_dispatches != 1 + prestart_used + compute_attempt
    ):
        raise ValueError("retry or phase counters violate pre-registered V3 caps")
    if probe and (prestart_used != 0 or compute_attempt != 0 or total_dispatches != 1):
        raise ValueError("canary probe must have zero shard retry counters")

    env = manifest.get("env", {})
    if env.get("WATCH6_FIXTURE") != ("1" if probe else "0"):
        raise ValueError("canary/main fixture isolation mismatch")
    if env.get("SNAKE_WATCH_SOURCE") != "live":
        raise ValueError("requires explicit SNAKE_WATCH_SOURCE=live")
    if int(env.get("WATCH6_BASE_MINUTES", -1)) != base_minutes:
        raise ValueError("runtime base_minutes env mismatch")
    if manifest.get("required_exports") != ["efficacy/operational.json", "performance.json"]:
        raise ValueError("required_exports mismatch")
    if manifest.get("resume") != {
        "unit_kind": "paired-session",
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
            int(env["WATCH6_SHARD"]),
            prior_receipts,
            prestart_used,
            compute_attempt,
            total_dispatches,
        )

    for item in manifest["inputs"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"input digest mismatch: {item['dest']}")
    upload_report = check_uploads(manifest)
    meta_item = next(i for i in manifest["inputs"] if i["dest"] == METADATA)
    if meta_item["sha256"] != env.get("WATCH6_METADATA_SHA256"):
        raise ValueError("runtime metadata SHA differs from uploaded manifest input")

    with tempfile.TemporaryDirectory(prefix="watch6-v3-admission-") as temporary:
        root = Path(temporary)
        shutil.copyfile(meta_item["path"], root / METADATA)
        stage_metadata(root, env["WATCH6_METADATA_SHA256"])
        verify_runtime(root, env)

    if not probe and not pending_qualification:
        canary.check(manifest, p.ROOT)
    if not probe:
        report = json.loads(env["WATCH6_MEASUREMENTS"])
        verify_report_evidence(report)
    return upload_report


def verify_runtime(root: Path, env: dict) -> dict:
    """Execute-time guard validating contract, registration hash, mirror, and 24x4 W4 sizing."""
    contract = json.loads((root / "runtime-contract.json").read_text(encoding="utf-8"))
    expected_contract = {
        "schema": "watch6-v3-runtime-contract/v2",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "primary_allocation": p.allocation(),
    }
    if contract != expected_contract:
        raise ValueError("runtime contract source/allocation mismatch")
    if env.get("WATCH6_SCIENCE") != p.identity() or env.get("SNAKE_WATCH_SOURCE") != "live":
        raise ValueError("runtime science or SNAKE_WATCH_SOURCE env mismatch")

    shards = int(env.get("WATCH6_SHARDS", "0"))
    workers = int(env.get("WATCH6_WORKERS", "0"))
    if (
        env.get("WATCH6_FIXTURE") not in ("0", "1")
        or shards != p.SHARDS
        or int(env.get("WATCH6_SHARD", "-1")) not in range(shards)
        or workers != p.WORKERS
    ):
        raise ValueError("runtime shard/fixture/workers mismatch (requires 24 shards, W4)")

    completed_ledger(root / "verified-ledger.json")

    verify_immutable(root / "mac-mirror.json")
    validate_mirror(json.loads((root / "mac-mirror.json").read_text(encoding="utf-8")))

    protocol_meta = json.loads((root / "judge-approval.json").read_text(encoding="utf-8"))
    if protocol_meta != {
        "schema": "watch6-v3-protocol-binding/v2",
        "implementer": "Argon",
        "judge": "Astra",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "registration_sha256": digest(p.registration()),
    }:
        raise ValueError("uploaded protocol/registration binding mismatch")

    if env["WATCH6_FIXTURE"] == "0":
        if "WATCH6_MEASUREMENTS" not in env:
            raise ValueError("main shard requires empirical cost measurements")
        report = json.loads(env["WATCH6_MEASUREMENTS"])
        verify_immutable(root / "cost-evidence.json")
        if json.loads((root / "cost-evidence.json").read_text(encoding="utf-8")) != report:
            raise ValueError("WATCH6_MEASUREMENTS differs from uploaded cost-evidence.json")
        selected = empirical_forecast(
            report,
            json.loads(env["WATCH6_FROZEN_LAYOUT"]) if "WATCH6_FROZEN_LAYOUT" in env else None,
        )
        if (
            workers != selected["workers"]
            or shards != selected["shards"]
            or int(env.get("WATCH6_BASE_MINUTES", "0")) != selected["base_minutes_per_shard"]
        ):
            raise ValueError("runtime sizing differs from empirical 24x4 W4 forecast")
    return contract


def runtime_admit(
    *, workers: int | None = None, shards: int | None = None, shard: int | None = None
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
