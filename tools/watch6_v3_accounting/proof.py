"""Fail-closed verifiers for PROVEN never-allocated attempts (Shard-00 Ruling Point 4) and
hash-bound renewal-WebSocket zero-export transport-evidence sidecars (Shard-12 Ruling Points 2-6).

- Never-allocated attempts: only a proven never-allocated attempt satisfying every item in
  Shard-00 Ruling Point 4 may retain ``cleanup_error == "ALLOCATION_UNRESOLVED: retained pending
  record"`` with ``stop_verified is False``. Receipts are never rewritten and
  ``stop_verified=True`` is never manufactured.
- Truncated renewal-WebSocket zero-export attempts: only an allocated, cleanly stopped,
  zero-export attempt satisfying the complete hash-bound sidecar conjunction in Shard-12 Ruling
  Points 2, 3, and 6 (prospective across shards 0..23, excluding partial-export and general
  WebSocket failures) is recognized as an external transport compute failure. Provenance
  substitutes ONLY for missing traceback header/frame paths due to the pinned supervisor's
  2,000-character CLI output truncation; every child, scientific, integrity, and timeout veto
  overrides transport.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

from tools.watch6_v3_accounting import (
    BASELINE_COMMIT,
    BASELINE_NAMESPACE,
    BASELINE_SCIENCE,
    PINNED_SUPERVISOR_REL,
    PINNED_SUPERVISOR_SHA256,
    RETAINED_PENDING_CLEANUP_ERROR,
    SHARD00_EXEC_MANIFEST_SHA256,
    SHARD00_NOTEBOOK_HASH,
    SHARD00_PENDING_SHA256,
    SHARD00_RECEIPT_SHA256,
    SHARD12_EVIDENCE_SIDECAR_REL,
    SHARD12_EXEC_MANIFEST_SHA256,
    SHARD12_RECEIPT_SHA256,
    V3_BROKER_ROOT,
    artifacts_root,
    default_artifact_dir,
    ensure_cli_dash_b,
    isolated_broker_execution,
    verify_baseline_closure,
)

# isort: split

from research.watch6_search_efficacy_v3 import admission as v3_admission
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.admission import has_non_transport_fault
from research.watch6_search_efficacy_v3.dispatch import (
    ALLOCATION_UNAVAILABLE,
    DEFAULT_OPS_DIR,
    _log_has_exit,
)
from tools.ops.package_spec import digest, publish_json, verify_immutable

PROOF_SCHEMA = "colab-allocation-unavailable/v1"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
ACCEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
SESSION_LINE_RE = re.compile(
    r"^\[([^\]]+)\]\s+([A-Za-z0-9_.-]+)\s+\|\s+Hardware:\s+[^|]+\|"
    r"\s+Shape:\s+[^|]+\|\s+Variant:\s+\S+\s*$"
)
CREATING_SESSION_RE = re.compile(r"\[colab\] Creating session '([^']+)'\.\.\.")
ASSIGN_URL_NBH_RE = re.compile(
    r"https://colab\.research\.google\.com/tun/m/assign\?nbh=([A-Za-z0-9_.-]+)"
)
ASSIGN_503_LINE_RE = re.compile(
    r"^ColabRequestError: Failed to issue request POST "
    r"https://colab\.research\.google\.com/tun/m/assign"
    r"\?nbh=([a-f0-9_.-]+)&variant=GPU&accelerator=([A-Za-z][A-Za-z0-9_-]*): "
    r"(?:Service Unavailable|503(?: Service Unavailable)?)$"
)

NO_ACTIVE_SESSIONS_LINES: frozenset[str] = frozenset(
    {
        "No active sessions.",
        "[colab] No active sessions.",
        "No active sessions found on server.",
        "[colab] No active sessions found on server.",
    }
)

REQUIRED_PROOF_KEYS: frozenset[str] = frozenset(
    {
        "classification",
        "endpoint_ever_assigned",
        "membership",
        "mode",
        "pending",
        "pending_action",
        "receipt",
        "recovery",
        "requested_accelerator",
        "schema",
        "sessions_stdout",
        "vm_compute_started",
    }
)
OPTIONAL_PROOF_KEYS: frozenset[str] = frozenset(
    {
        "auth_ok",
        "sessions_ok",
        "job_id",
        "session",
        "notebook_hash",
        "nbh",
    }
)
REQUIRED_PENDING_KEYS: frozenset[str] = frozenset({"path", "record", "sha256"})
REQUIRED_PENDING_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "assignment_record",
        "before",
        "config",
        "notebook_hash",
        "output",
        "session",
    }
)

# Complete post-allocation & fault artifact set enumerated from
# research/colab_cli/supervisor.py (lines 737, 1481, 1554, 1574, 1608, 1838,
# 1856-1858, 1901, 1905, 1945, 1975, 2018, 2022, 2068, 2133, 2270),
# research/colab_cli/runtime.py, and runtime/scientific fault outputs.
FORBIDDEN_OUTPUT_ENTRIES: tuple[str, ...] = (
    "terminal.json",
    "hardware.json",
    "ready.json",
    "warmup-measurement.json",
    "rate-calibration.json",
    "upload-measurement.json",
    "source-delivery.json",
    "input-hashes.json",
    "usage-allocated.json",
    "backup.json",
    "session-release.json",
    "runtime-measurement.json",
    "scientific-fault.json",
    "spec.json",
    "lease.json",
    "lease-heartbeat.json",
    "release-credential.json",
    "public-inputs-verified.json",
    "public-input-fetch-failed.json",
    "supervisor.log",
    "job.log",
    "performance.json",
    "operational.json",
    "canary-observations",
    "events.json",
    "exports",
)

FORBIDDEN_OUTPUT_GLOBS: tuple[str, ...] = (
    "poll-*.json",
    "telemetry-*.json",
    "usage-*.json",
    "spend-*.json",
    "rate-calibration-*.json",
    "recovery-*.json",
    "runtime-measurement*",
    "scientific-fault*",
)

ALLOWED_PREALLOC_OUTPUT_ENTRIES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "receipt.json",
        "accounting-baseline.json",
        "public-input-preflight.json",
        "staging",
        "state.json",
        "events.jsonl",
    }
)

ALLOWED_PREALLOC_STAGING_ENTRIES: frozenset[str] = frozenset(
    {
        "bootstrap.sh",
        "release.py",
        "runtime.py",
        "schema-validation.json",
        "source-plan.json",
        "source.py",
        "source.tar.gz",
    }
)

ALLOWED_PREALLOC_EVENT_TOKENS: frozenset[str] = frozenset(
    {
        "version",
        "sessions",
        "usage",
        "new",
        "refused",
        "allocation_refused",
        "preflight",
        "accounting_baseline",
    }
)


def _read_regular_json_dict(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable or corrupt {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def canonical_pending_bytes(record: dict) -> bytes:
    """Exact byte serialization written by ``research.colab_cli.supervisor.write_once``."""
    return (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def canonical_pending_sha256(record: dict) -> str:
    """Compute exact SHA-256 of the archived pending payload."""
    return hashlib.sha256(canonical_pending_bytes(record)).hexdigest()


def _validate_notebook_uuid4(raw: object, *, allow_assign_encoding: bool = False) -> str:
    """Validate an RFC-4122 UUIDv4 notebook identity and return canonical 32-char lowercase hex."""
    if not isinstance(raw, str) or not raw:
        raise ValueError("missing or non-string notebook_hash/nbh identity")
    candidate = (
        raw.translate(str.maketrans("", "", "._-")).lower() if allow_assign_encoding else raw
    )
    if HEX32_RE.fullmatch(candidate) is None:
        raise ValueError(f"malformed 32-hex notebook_hash/nbh identity: {raw!r}")
    try:
        parsed = uuid.UUID(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid UUID notebook_hash/nbh identity: {raw!r}") from exc
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122 or parsed.hex != candidate:
        raise ValueError(f"notebook_hash/nbh is not an RFC-4122 UUIDv4 identity: {raw!r}")
    return candidate


def _same_notebook_uuid(left: object, right_hex: str) -> bool:
    if not isinstance(left, str) or not left:
        return False
    cleaned = left.translate(str.maketrans("", "", "._-")).lower()
    if HEX32_RE.fullmatch(cleaned) is None:
        return False
    return cleaned == right_hex


def _is_colab_update_banner_line(line: str) -> bool:
    return (
        line.startswith("[colab] A new version of Colab CLI is available")
        or line.startswith("[colab] Run 'colab update'")
        or line.startswith('[colab] To silence this check, set "enable_update_check"')
    )


def strip_colab_update_notice(detail: str) -> str:
    """Strip optional colab-cli 0.7.5+ update banner prepended to `colab new failed (1): `."""
    prefix = "colab new failed (1): "
    if not detail.startswith(prefix):
        return detail
    lines = detail[len(prefix) :].splitlines(keepends=True)
    while lines and (_is_colab_update_banner_line(lines[0]) or lines[0] == "\n"):
        lines.pop(0)
    return prefix + "".join(lines)


def _matches_failed_new_server_refusal(
    error_raw: object,
    *,
    session: str,
    accelerator: str,
    notebook_hash: str | None = None,
) -> bool:
    if (
        not isinstance(error_raw, str)
        or not error_raw.startswith("Refused(")
        or not error_raw.endswith(")")
    ):
        return False
    if has_non_transport_fault(error_raw):
        return False
    try:
        detail = ast.literal_eval(error_raw[len("Refused(") : -1])
    except (ValueError, SyntaxError):
        return False
    if not isinstance(detail, str) or has_non_transport_fault(detail):
        return False

    creating_sessions = CREATING_SESSION_RE.findall(detail)
    if any(found != session for found in creating_sessions):
        return False

    for raw_url_nbh in ASSIGN_URL_NBH_RE.findall(detail):
        try:
            norm_url_nbh = _validate_notebook_uuid4(raw_url_nbh, allow_assign_encoding=True)
        except ValueError:
            return False
        if notebook_hash is not None and norm_url_nbh != notebook_hash:
            return False

    normalized = strip_colab_update_notice(detail)
    expected_precondition = (
        f"colab new failed (1): [colab] Creating session '{session}'...\n"
        f"{ALLOCATION_UNAVAILABLE}\n"
    )
    if normalized == expected_precondition:
        return True

    if normalized.startswith("colab new failed (1): "):
        lines = normalized.removeprefix("colab new failed (1): ").strip().splitlines()
        if lines:
            match = ASSIGN_503_LINE_RE.fullmatch(lines[-1])
            if match is not None and match.group(2) == accelerator:
                try:
                    url_nbh = _validate_notebook_uuid4(match.group(1), allow_assign_encoding=True)
                except ValueError:
                    return False
                if notebook_hash is not None and url_nbh != notebook_hash:
                    return False
                return True
    return False


def _verify_optional_state_and_events(
    directory: Path,
    *,
    job_id: str | None = None,
    session: str | None = None,
) -> None:
    """Fail closed on contradictory state.json or events.jsonl indicating allocation or exec."""
    events_json_path = directory / "events.json"
    if events_json_path.exists() or events_json_path.is_symlink():
        raise ValueError(f"forbidden events.json present: {events_json_path}")

    state_path = directory / "state.json"
    if state_path.exists() or state_path.is_symlink():
        state = _read_regular_json_dict(state_path, "state.json")
        if not state:
            raise ValueError(f"empty state.json forbidden: {state_path}")
        if state.get("status") not in {"INCOMPLETE", "REFUSED"}:
            raise ValueError(f"contradictory or allocated status in state.json: {state_path}")
        if "failure_class" in state and state["failure_class"] not in {
            None,
            "SUPERVISOR",
            "INFRA",
            "ACCOUNTING",
        }:
            raise ValueError(f"forbidden failure_class in state.json: {state_path}")
        if "stop_verified" in state and state["stop_verified"] is not False:
            raise ValueError(f"contradictory stop_verified in state.json: {state_path}")

        for null_key in (
            "endpoint",
            "allocated_endpoint",
            "assigned_endpoint",
            "runtime_endpoint",
            "release_endpoint",
            "kernel_id",
            "pid",
            "url",
            "token",
            "terminal",
            "hardware",
            "returncode",
            "usage_after",
            "calibration",
            "last_crosscheck",
            "observed_rate_units_per_hour",
        ):
            if state.get(null_key) is not None:
                raise ValueError(f"allocated/post-allocation {null_key} in state.json")

        for false_key in (
            "allocated",
            "assigned",
            "endpoint_ever_assigned",
            "vm_compute_started",
            "ready",
            "staging_completed",
        ):
            if false_key in state and state[false_key] is not False:
                raise ValueError(f"allocated/compute flag {false_key} in state.json")

        for zero_key in (
            "polls",
            "poll_count",
            "uploaded",
            "downloaded",
            "warmup_seconds",
            "usage_checks",
        ):
            if zero_key in state and (
                isinstance(state[zero_key], bool) or state[zero_key] not in (0, 0.0)
            ):
                raise ValueError(f"post-allocation counter {zero_key} nonzero in state.json")

        for empty_key in (
            "exports",
            "transfers",
            "contents_transfers",
            "control_operations",
            "unaccounted_endpoints",
        ):
            if empty_key in state and state[empty_key] != []:
                raise ValueError(f"post-allocation list {empty_key} non-empty in state.json")

        if "accounting" in state:
            acct = state["accounting"]
            if (
                not isinstance(acct, dict)
                or acct.get("allocated_endpoint") is not None
                or acct.get("calibration") is not None
                or acct.get("last_crosscheck") is not None
                or acct.get("observed_rate_units_per_hour") is not None
                or acct.get("unaccounted_endpoints", []) != []
            ):
                raise ValueError("contradictory accounting state in state.json")

        if job_id is not None and "job_id" in state and state["job_id"] != job_id:
            raise ValueError("state.json job_id mismatch")
        if session is not None and "session" in state and state["session"] != session:
            raise ValueError("state.json session mismatch")

        for key, val in state.items():
            low_k = key.lower()
            if any(tok in low_k for tok in ("alloc", "assign", "exec", "kernel", "endpoint")):
                if val not in (None, False, 0, 0.0, [], {}):
                    raise ValueError(f"allocated/exec indicator {key!r} in state.json")
            if isinstance(val, str) and has_non_transport_fault(val):
                raise ValueError(f"non-transport fault text in state.json.{key}")

    events_path = directory / "events.jsonl"
    if events_path.exists() or events_path.is_symlink():
        if events_path.is_symlink() or not events_path.is_file():
            raise ValueError(f"missing or symlinked events.jsonl: {events_path}")
        try:
            raw_events = events_path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise ValueError(f"unreadable events.jsonl: {events_path}") from exc
        lines = [ln.strip() for ln in raw_events.splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"empty events.jsonl forbidden: {events_path}")

        tag_fields = ("event", "command", "subcommand", "kind", "type", "action")
        for idx, line in enumerate(lines):
            try:
                ev = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"corrupt JSONL in {events_path} line {idx + 1}") from exc
            if not isinstance(ev, dict) or not ev:
                raise ValueError(f"non-object event in {events_path} line {idx + 1}")

            for null_key in (
                "endpoint",
                "allocated_endpoint",
                "assigned_endpoint",
                "runtime_endpoint",
                "release_endpoint",
                "kernel_id",
                "pid",
                "url",
                "token",
                "terminal",
                "hardware",
            ):
                if ev.get(null_key) is not None:
                    raise ValueError(
                        f"allocated/post-allocation {null_key} in events.jsonl line {idx + 1}"
                    )

            for false_key in (
                "allocated",
                "assigned",
                "endpoint_ever_assigned",
                "vm_compute_started",
                "ready",
                "staging_completed",
            ):
                if false_key in ev and ev[false_key] is not False:
                    raise ValueError(
                        f"allocated/compute flag {false_key} in events.jsonl line {idx + 1}"
                    )

            for zero_key in ("polls", "poll_count", "uploaded", "downloaded", "warmup_seconds"):
                if zero_key in ev and (
                    isinstance(ev[zero_key], bool) or ev[zero_key] not in (0, 0.0)
                ):
                    raise ValueError(f"nonzero counter {zero_key} in events.jsonl line {idx + 1}")

            present_tags: list[str] = []
            for tf in tag_fields:
                if tf in ev:
                    raw_tag = ev[tf]
                    if not isinstance(raw_tag, str) or not raw_tag.strip():
                        raise ValueError(f"invalid {tf} in events.jsonl line {idx + 1}")
                    norm_tag = raw_tag.strip().lower()
                    if norm_tag not in ALLOWED_PREALLOC_EVENT_TOKENS:
                        raise ValueError(
                            f"forbidden post-allocation/exec event {tf}={raw_tag!r} "
                            f"in events.jsonl line {idx + 1}"
                        )
                    present_tags.append(norm_tag)

            if "operation" in ev:
                raw_op = ev["operation"]
                if not isinstance(raw_op, str) or not raw_op.strip():
                    raise ValueError(f"invalid operation in events.jsonl line {idx + 1}")
                norm_op = raw_op.strip().lower()
                if norm_op not in (ALLOWED_PREALLOC_EVENT_TOKENS | {"control"}):
                    raise ValueError(
                        f"forbidden post-allocation operation={raw_op!r} "
                        f"in events.jsonl line {idx + 1}"
                    )
                present_tags.append(norm_op)

            if not present_tags:
                raise ValueError(f"unclassified event in events.jsonl line {idx + 1}")

            if "new" in present_tags:
                if "status" in ev and (
                    not isinstance(ev["status"], str)
                    or ev["status"].strip().lower() not in {"failed", "refused", "error"}
                ):
                    raise ValueError(f"non-failed 'new' event in events.jsonl line {idx + 1}")
                if "returncode" in ev and (
                    isinstance(ev["returncode"], bool) or ev["returncode"] in (None, 0)
                ):
                    raise ValueError(
                        f"zero/null returncode on 'new' event in events.jsonl line {idx + 1}"
                    )

            if "status" in ev:
                if not isinstance(ev["status"], str) or ev["status"].strip().lower() in {
                    "allocated",
                    "assigned",
                    "running",
                    "starting",
                    "ready",
                    "succeeded",
                    "active",
                }:
                    raise ValueError(f"allocated/running status in events.jsonl line {idx + 1}")

            if job_id is not None and "job_id" in ev and ev["job_id"] != job_id:
                raise ValueError("events.jsonl job_id mismatch")
            if session is not None and "session" in ev and ev["session"] != session:
                raise ValueError("events.jsonl session mismatch")

            for key, val in ev.items():
                if isinstance(val, str) and has_non_transport_fault(val):
                    raise ValueError(f"non-transport fault text in events.jsonl.{key}")


def verify_output_dir_prealloc_clean(
    output_dir: Path,
    *,
    job_id: str | None = None,
    session: str | None = None,
) -> None:
    """Verify zero post-allocation/fault files and non-contradictory state.json/events.jsonl."""
    for forbidden in FORBIDDEN_OUTPUT_ENTRIES:
        target = output_dir / forbidden
        if target.exists() or target.is_symlink():
            raise ValueError(f"forbidden post-allocation artifact present in output_dir: {target}")

    if output_dir.is_dir() and not output_dir.is_symlink():
        for pattern in FORBIDDEN_OUTPUT_GLOBS:
            if any(output_dir.glob(pattern)):
                raise ValueError(f"forbidden runtime/fault artifact matching {pattern}")

    _verify_optional_state_and_events(output_dir, job_id=job_id, session=session)

    if output_dir.is_dir() and not output_dir.is_symlink():
        for entry in sorted(output_dir.iterdir()):
            if entry.is_symlink():
                raise ValueError(f"symlinked entry forbidden in output_dir: {entry}")
            if entry.name not in ALLOWED_PREALLOC_OUTPUT_ENTRIES:
                raise ValueError(
                    f"unexpected/post-allocation entry in never-allocated output_dir: {entry}"
                )
            if entry.name == "staging":
                if not entry.is_dir():
                    raise ValueError(f"staging must be a directory: {entry}")
                for child in sorted(entry.iterdir()):
                    if (
                        child.is_symlink()
                        or not child.is_file()
                        or child.name not in ALLOWED_PREALLOC_STAGING_ENTRIES
                    ):
                        raise ValueError(f"unexpected file inside pre-allocation staging: {child}")


def has_forbidden_or_contradictory_output_artifacts(
    output_dir: Path | None,
    *,
    job_id: str | None = None,
    session: str | None = None,
) -> bool:
    """True iff output_dir contains any forbidden post-allocation/fault file or bad state/events."""
    if not isinstance(output_dir, Path):
        return False
    if output_dir.is_symlink():
        return True
    try:
        verify_output_dir_prealloc_clean(output_dir, job_id=job_id, session=session)
    except ValueError:
        return True
    return False


def _parse_sessions_stdout(
    sessions_stdout: object,
    forbidden_session: str | None = None,
) -> set[str]:
    if not isinstance(sessions_stdout, str) or not sessions_stdout.strip():
        raise ValueError("proof sessions_stdout must be a non-empty string")
    if forbidden_session and forbidden_session in sessions_stdout:
        raise ValueError("refused session appears in proof sessions_stdout")

    raw_lines = [ln for ln in sessions_stdout.splitlines() if ln.strip()]
    while raw_lines and _is_colab_update_banner_line(raw_lines[0].strip()):
        raw_lines.pop(0)
    if not raw_lines:
        raise ValueError("proof sessions_stdout contains no session inventory output")

    if len(raw_lines) == 1 and raw_lines[0].strip() in NO_ACTIVE_SESSIONS_LINES:
        return set()

    endpoints: list[str] = []
    for raw_line in raw_lines:
        match = SESSION_LINE_RE.match(raw_line)
        if match is None:
            raise ValueError(f"unrecognized line in proof sessions_stdout: {raw_line!r}")
        sess_label, endpoint = match.group(1), match.group(2)
        if forbidden_session and sess_label == forbidden_session:
            raise ValueError("refused session appears in parsed sessions_stdout")
        endpoints.append(endpoint)
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("duplicate endpoints in proof sessions_stdout")
    return set(endpoints)


def _validate_proof_auth_and_keys(
    proof: dict,
    *,
    require_full_schema: bool = True,
) -> None:
    keys = set(proof.keys())
    if require_full_schema and not REQUIRED_PROOF_KEYS.issubset(keys):
        missing = sorted(REQUIRED_PROOF_KEYS - keys)
        raise ValueError(f"unallocated proof missing required keys: {missing}")
    if not keys.issubset(REQUIRED_PROOF_KEYS | OPTIONAL_PROOF_KEYS):
        unknown = sorted(keys - (REQUIRED_PROOF_KEYS | OPTIONAL_PROOF_KEYS))
        raise ValueError(f"unallocated proof contains unexpected keys: {unknown}")
    if "auth_ok" in proof or "sessions_ok" in proof:
        if proof.get("auth_ok") is not True or proof.get("sessions_ok") is not True:
            raise ValueError("unallocated proof auth_ok/sessions_ok must both be boolean True")


def verify_no_receipt_unallocated_proof(
    proof_path: Path,
    output_dir: Path,
) -> bool:
    """Strict verifier for a no-receipt LIVE colab-allocation-unavailable/v1 proof."""
    if proof_path.is_symlink() or not proof_path.is_file():
        return False
    if (output_dir / "receipt.json").exists() or (output_dir / "receipt.json").is_symlink():
        return False
    if has_forbidden_or_contradictory_output_artifacts(output_dir):
        return False
    try:
        proof = _read_regular_json_dict(proof_path, "unallocated proof")
        _validate_proof_auth_and_keys(proof, require_full_schema=False)
        if (
            proof.get("schema") != PROOF_SCHEMA
            or proof.get("classification", "ALLOCATION_UNAVAILABLE") != "ALLOCATION_UNAVAILABLE"
            or proof.get("mode") != "LIVE"
            or proof.get("endpoint_ever_assigned") is not False
            or proof.get("vm_compute_started") is not False
            or proof.get("recovery") != {"endpoint": None}
        ):
            return False
        forbidden_sess: str | None = None
        if "pending" in proof:
            pending_obj = proof.get("pending")
            if (
                not isinstance(pending_obj, dict)
                or set(pending_obj.keys()) != REQUIRED_PENDING_KEYS
            ):
                return False
            record = pending_obj.get("record")
            if (
                not isinstance(record, dict)
                or set(record.keys()) != REQUIRED_PENDING_RECORD_KEYS
                or not isinstance(pending_obj.get("sha256"), str)
                or HEX64_RE.fullmatch(pending_obj["sha256"]) is None
                or canonical_pending_sha256(record) != pending_obj["sha256"]
            ):
                return False
            _validate_notebook_uuid4(record.get("notebook_hash"))
            rec_sess = record.get("session")
            if not isinstance(rec_sess, str) or not rec_sess:
                return False
            forbidden_sess = rec_sess
            pending_file = Path(str(pending_obj.get("path", "")))
            if (
                pending_file.name != f"pending-{rec_sess}.json"
                or pending_file.exists()
                or pending_file.is_symlink()
                or (pending_file.parent / f"active-{rec_sess}.json").exists()
                or (pending_file.parent / f"active-{rec_sess}.json").is_symlink()
                or Path(str(record.get("output", ""))).resolve() != output_dir.resolve()
            ):
                return False
        if "sessions_stdout" in proof:
            _parse_sessions_stdout(proof["sessions_stdout"], forbidden_sess)
    except (OSError, ValueError):
        return False
    return True


def _discover_ops_names_and_manifests(
    ops_dir: Path,
    output_dir: Path,
    ops_name: str | None = None,
) -> list[tuple[str, Path, dict]]:
    resolved_out = output_dir.resolve()
    manifests_dir = ops_dir / "manifests"
    matches: list[tuple[str, Path, dict]] = []
    if ops_name is not None:
        mf_path = manifests_dir / f"{ops_name}.json"
        if mf_path.exists() or mf_path.is_symlink():
            mf_data = _read_regular_json_dict(mf_path, "ops manifest")
            matches.append((ops_name, mf_path, mf_data))
        return matches

    if not manifests_dir.is_dir():
        return matches
    for mf_path in sorted(manifests_dir.glob("*.json")):
        if mf_path.name.endswith(".live.json"):
            continue
        mf_data = _read_regular_json_dict(mf_path, "ops manifest")
        raw_out = mf_data.get("output_dir")
        if isinstance(raw_out, str) and raw_out.strip():
            if Path(raw_out).resolve() == resolved_out:
                matches.append((mf_path.stem, mf_path, mf_data))
    return matches


def _discover_proof_paths(
    ops_dir: Path,
    receipt_path: Path,
    ops_names: list[str],
) -> list[Path]:
    resolved_receipt = receipt_path.resolve()
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add(cand: Path) -> None:
        if cand.is_symlink():
            raise ValueError(f"symlinked unallocated proof forbidden: {cand}")
        if cand.exists():
            if not cand.is_file():
                raise ValueError(f"non-regular unallocated proof path: {cand}")
            key = cand.resolve()
            if key not in seen:
                seen.add(key)
                candidates.append(cand)

    for name in ops_names:
        _add(ops_dir / "dispatch" / name / "unavailable.json")
        disp_dir = ops_dir / "dispatch" / name
        if disp_dir.is_dir():
            for item in sorted(disp_dir.glob("unallocated*.json")):
                _add(item)
        for item in sorted(ops_dir.glob(f"unallocated-{name}*.json")):
            _add(item)

    if not candidates:
        scan_globs = [
            *sorted(ops_dir.glob("unallocated-*.json")),
            *sorted(ops_dir.glob("dispatch/*/unavailable.json")),
            *sorted(ops_dir.glob("dispatch/*/unallocated*.json")),
        ]
        for item in scan_globs:
            if item.is_symlink():
                raise ValueError(f"symlinked unallocated proof forbidden: {item}")
            if not item.is_file():
                continue
            payload = _read_regular_json_dict(item, "unallocated proof")
            rec_ref = payload.get("receipt")
            rec_path_str = rec_ref.get("path") if isinstance(rec_ref, dict) else None
            if isinstance(rec_path_str, str) and rec_path_str:
                if Path(rec_path_str).resolve() == resolved_receipt:
                    _add(item)

    return candidates


def _verify_notebook_not_replayed(
    ops_dir: Path,
    own_proof_paths: list[Path],
    own_receipt_path: Path,
    notebook_hash: str,
    *,
    account_dir: Path | None = None,
) -> None:
    """Refuse any notebook_hash appearing in a foreign proof, assignment, or broker record."""
    own_proofs_resolved = {pp.resolve() for pp in own_proof_paths}
    own_receipt_resolved = own_receipt_path.resolve()

    all_proofs = [
        *sorted(ops_dir.rglob("unallocated*.json")),
        *sorted(ops_dir.rglob("unavailable.json")),
    ]
    seen_proofs: set[Path] = set()
    for item in all_proofs:
        if item.is_symlink() or not item.is_file():
            continue
        resolved_item = item.resolve()
        if resolved_item in seen_proofs or resolved_item in own_proofs_resolved:
            continue
        seen_proofs.add(resolved_item)
        try:
            other = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(other, dict):
            continue
        other_rec = other.get("receipt")
        other_rec_path = other_rec.get("path") if isinstance(other_rec, dict) else None
        if isinstance(other_rec_path, str) and other_rec_path:
            try:
                if Path(other_rec_path).resolve() == own_receipt_resolved:
                    continue
            except OSError:
                pass
        other_pend = other.get("pending")
        other_record = other_pend.get("record") if isinstance(other_pend, dict) else None
        other_nb = other_record.get("notebook_hash") if isinstance(other_record, dict) else None
        if _same_notebook_uuid(other_nb, notebook_hash) or _same_notebook_uuid(
            other.get("notebook_hash"), notebook_hash
        ):
            raise ValueError(
                f"foreign/replayed notebook_hash {notebook_hash} already bound in {item}"
            )

    if ops_dir.is_dir() and not ops_dir.is_symlink():
        for assign_file in sorted(ops_dir.rglob("*.assignment.json")):
            if assign_file.is_symlink():
                raise ValueError(f"symlinked assignment record in ops: {assign_file}")
            try:
                data = json.loads(assign_file.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                raise ValueError(f"corrupt assignment record in ops: {assign_file}") from exc
            if isinstance(data, dict) and _same_notebook_uuid(
                data.get("notebook_hash"), notebook_hash
            ):
                raise ValueError(
                    f"notebook_hash {notebook_hash} belongs to allocated assignment {assign_file}"
                )

    if account_dir is not None and account_dir.is_dir() and not account_dir.is_symlink():
        for pat in ("active-*.json", "pending-*.json"):
            for broker_file in sorted(account_dir.glob(pat)):
                if broker_file.is_symlink() or not broker_file.is_file():
                    continue
                try:
                    bdata = json.loads(broker_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(bdata, dict) and _same_notebook_uuid(
                    bdata.get("notebook_hash"), notebook_hash
                ):
                    raise ValueError(
                        f"notebook_hash {notebook_hash} belongs to live broker record {broker_file}"
                    )


def verify_never_allocated_attempt(
    output_dir: Path,
    *,
    receipt: dict | None = None,
    ops_dir: Path | None = None,
    ops_name: str | None = None,
) -> dict:
    """Verify every item in Astra Ruling Point 4 for a never-allocated attempt; else raise."""
    if not isinstance(output_dir, Path) or output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(f"invalid output_dir for never-allocated proof: {output_dir}")

    resolved_out = output_dir.resolve()
    receipt_path = output_dir / "receipt.json"
    exec_manifest_path = output_dir / "manifest.json"
    disk_receipt = _read_regular_json_dict(receipt_path, "receipt.json")
    if receipt is not None and receipt != disk_receipt:
        raise ValueError("in-memory receipt differs from immutable on-disk receipt.json")
    exec_manifest = _read_regular_json_dict(exec_manifest_path, "executed manifest.json")

    receipt_sha = digest(receipt_path)
    exec_manifest_sha = digest(exec_manifest_path)

    # 1. Immutable manifest / receipt / job / session / generation / commit bindings.
    job_id = exec_manifest.get("job_id")
    commit = exec_manifest.get("commit")
    generation = disk_receipt.get("generation")
    session = disk_receipt.get("session")
    raw_exec_out = exec_manifest.get("output_dir")
    if (
        not isinstance(job_id, str)
        or JOB_ID_RE.fullmatch(job_id) is None
        or disk_receipt.get("job_id") != job_id
        or not isinstance(commit, str)
        or HEX40_RE.fullmatch(commit) is None
        or disk_receipt.get("commit") != commit
        or disk_receipt.get("manifest_sha256") != exec_manifest_sha
        or not isinstance(raw_exec_out, str)
        or not raw_exec_out.strip()
        or Path(raw_exec_out).resolve() != resolved_out
        or not isinstance(generation, str)
        or HEX32_RE.fullmatch(generation) is None
        or not isinstance(session, str)
        or session != f"snake-{job_id}-{generation[:12]}"
    ):
        raise ValueError("manifest/receipt/job/session/generation/commit binding mismatch")

    req = exec_manifest.get("requirements")
    accelerator = req.get("gpu") if isinstance(req, dict) else None
    if not isinstance(accelerator, str) or ACCEL_RE.fullmatch(accelerator) is None:
        raise ValueError("executed manifest missing valid requirements.gpu accelerator")

    # 2. Explicit failed-new server refusal & retained-pending cleanup state (never rewritten).
    if (
        disk_receipt.get("status") != "INCOMPLETE"
        or disk_receipt.get("failure_class") != "SUPERVISOR"
        or disk_receipt.get("stop_verified") is not False
        or disk_receipt.get("cleanup_error") != RETAINED_PENDING_CLEANUP_ERROR
    ):
        raise ValueError("receipt status/failure_class/stop_verified/cleanup_error mismatch")

    for fault_field in ("error", "traceback", "last_traceback_line"):
        if has_non_transport_fault(disk_receipt.get(fault_field)):
            raise ValueError(f"non-transport fault text present in receipt.{fault_field}")

    if not _matches_failed_new_server_refusal(
        disk_receipt.get("error"), session=session, accelerator=accelerator
    ):
        raise ValueError("receipt error is not an explicit failed-new server allocation refusal")

    cli_calls = disk_receipt.get("cli_calls")
    if (
        not isinstance(cli_calls, list)
        or not cli_calls
        or not all(isinstance(call, dict) for call in cli_calls)
        or sum(call.get("command") == "new" for call in cli_calls) != 1
        or cli_calls[-1].get("command") != "new"
        or cli_calls[-1].get("status") != "failed"
        or any(
            call.get("command") not in {"version", "sessions", "usage", "new"} for call in cli_calls
        )
        or any(
            call.get("status") not in {"completed", "ok", "PASS", "verified"}
            or call.get("returncode") not in (None, 0)
            for call in cli_calls[:-1]
        )
    ):
        raise ValueError("cli_calls does not match pre-assignment failed-new-only call sequence")

    # 3. Null allocated endpoints & clean accounting state.
    accounting = disk_receipt.get("accounting")
    if (
        disk_receipt.get("endpoint") is not None
        or not isinstance(accounting, dict)
        or accounting.get("allocated_endpoint") is not None
        or accounting.get("unaccounted_endpoints") != []
        or accounting.get("calibration") is not None
        or accounting.get("last_crosscheck") is not None
        or accounting.get("observed_rate_units_per_hour") is not None
    ):
        raise ValueError("allocated endpoint or post-allocation accounting state is non-null")

    # 4. Zero transfers, control operations, terminal, hardware, polls, measurements, faults,
    #    or contradictory state.json / events.jsonl.
    if (
        disk_receipt.get("exports") != []
        or disk_receipt.get("transfers") != []
        or disk_receipt.get("contents_transfers") != []
        or disk_receipt.get("control_operations", []) != []
        or disk_receipt.get("terminal") is not None
        or disk_receipt.get("usage_after") is not None
        or disk_receipt.get("warmup_seconds") not in (0, 0.0)
    ):
        raise ValueError("post-allocation transfers, control ops, terminal, or exports recorded")

    verify_output_dir_prealloc_clean(output_dir, job_id=job_id, session=session)

    # 5. Ops manifest & log consistency + identical LIVE colab-allocation-unavailable/v1 proof(s).
    resolved_ops = (ops_dir or DEFAULT_OPS_DIR).resolve()
    if not resolved_ops.is_dir():
        raise ValueError(f"missing ops_dir for LIVE unallocated reconciliation: {resolved_ops}")

    ops_matches = _discover_ops_names_and_manifests(resolved_ops, resolved_out, ops_name)
    ops_names: list[str] = []
    ops_logs: list[Path] = []
    for matched_name, matched_mf_path, matched_mf in ops_matches:
        ops_names.append(matched_name)
        ops_core = {k: v for k, v in matched_mf.items() if k != "budget"}
        exec_core = {k: v for k, v in exec_manifest.items() if k != "budget"}
        ops_budget = matched_mf.get("budget") if isinstance(matched_mf.get("budget"), dict) else {}
        exec_budget = (
            exec_manifest.get("budget") if isinstance(exec_manifest.get("budget"), dict) else {}
        )
        if ops_core != exec_core or any(
            ops_budget.get(k) != exec_budget.get(k)
            for k in ("rate_units_per_hour", "reserve_units", "unit_cap")
        ):
            raise ValueError(f"ops manifest {matched_mf_path} diverges from executed manifest.json")
        log_path = resolved_ops / "logs" / f"{matched_name}.log"
        if not _log_has_exit(log_path, nonzero_only=True):
            raise ValueError(f"ops log lacks nonzero EXIT marker for refused job: {log_path}")
        ops_logs.append(log_path)

    if ops_name is not None and ops_name not in ops_names:
        log_path = resolved_ops / "logs" / f"{ops_name}.log"
        if log_path.exists():
            if not _log_has_exit(log_path, nonzero_only=True):
                raise ValueError(f"ops log lacks nonzero EXIT marker: {log_path}")
            ops_logs.append(log_path)
        ops_names.append(ops_name)

    proof_paths = _discover_proof_paths(resolved_ops, receipt_path, ops_names)
    if not proof_paths:
        raise ValueError(f"missing LIVE {PROOF_SCHEMA} proof for {receipt_path}")

    proof_shas = {digest(pp) for pp in proof_paths}
    proof_text_shas = {
        hashlib.sha256(pp.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for pp in proof_paths
    }
    if len(proof_shas) != 1 or len(proof_text_shas) != 1:
        raise ValueError(
            f"conflicting/non-identical unallocated proof copies for {receipt_path}: {proof_paths}"
        )
    proof_sha = next(iter(proof_shas))
    proof = _read_regular_json_dict(proof_paths[0], "unallocated proof")
    _validate_proof_auth_and_keys(proof, require_full_schema=True)

    if (
        proof.get("schema") != PROOF_SCHEMA
        or proof.get("classification") != "ALLOCATION_UNAVAILABLE"
        or proof.get("mode") != "LIVE"
        or proof.get("endpoint_ever_assigned") is not False
        or proof.get("vm_compute_started") is not False
        or proof.get("recovery") != {"endpoint": None}
        or proof.get("requested_accelerator") != accelerator
        or proof.get("pending_action") != "unlink_after_archive"
    ):
        raise ValueError("unallocated proof fails LIVE/never-assigned/no-compute contract")

    if "job_id" in proof and proof["job_id"] != job_id:
        raise ValueError("unallocated proof job_id binding mismatch")
    if "session" in proof and proof["session"] != session:
        raise ValueError("unallocated proof session binding mismatch")

    proof_rec = proof.get("receipt")
    if (
        not isinstance(proof_rec, dict)
        or set(proof_rec.keys()) != {"path", "sha256"}
        or not isinstance(proof_rec.get("path"), str)
        or not proof_rec["path"]
        or Path(proof_rec["path"]).resolve() != receipt_path.resolve()
        or proof_rec.get("sha256") != receipt_sha
    ):
        raise ValueError("unallocated proof receipt path/sha256 binding mismatch")

    # 6. Content-verified archived pending record + bound notebook/nbh identity + retired state.
    pending_obj = proof.get("pending")
    if (
        not isinstance(pending_obj, dict)
        or set(pending_obj.keys()) != REQUIRED_PENDING_KEYS
        or not isinstance(pending_obj.get("path"), str)
        or not pending_obj["path"]
        or not isinstance(pending_obj.get("sha256"), str)
        or HEX64_RE.fullmatch(pending_obj["sha256"]) is None
        or not isinstance(pending_obj.get("record"), dict)
    ):
        raise ValueError("unallocated proof missing archived pending record metadata")

    pending_path = Path(pending_obj["path"])
    if not pending_path.is_absolute() or pending_path.name != f"pending-{session}.json":
        raise ValueError("archived pending record filename does not match receipt session")
    if pending_path.exists() or pending_path.is_symlink():
        raise ValueError(f"pending allocation record was not retired from disk: {pending_path}")
    active_path = pending_path.parent / f"active-{session}.json"
    if active_path.exists() or active_path.is_symlink():
        raise ValueError(
            f"active session record exists for allegedly unallocated job: {active_path}"
        )

    record = pending_obj["record"]
    if set(record.keys()) != REQUIRED_PENDING_RECORD_KEYS:
        raise ValueError("archived pending record keyset does not match supervisor pending schema")

    if canonical_pending_sha256(record) != pending_obj["sha256"]:
        raise ValueError("archived pending record content SHA-256 does not match pending.sha256")

    nb_hash = _validate_notebook_uuid4(record.get("notebook_hash"))

    for nb_carrier in (proof, disk_receipt, exec_manifest):
        for nb_key in ("notebook_hash", "nbh"):
            if nb_key in nb_carrier:
                if (
                    _validate_notebook_uuid4(
                        nb_carrier[nb_key], allow_assign_encoding=(nb_key == "nbh")
                    )
                    != nb_hash
                ):
                    raise ValueError(f"{nb_key} binding mismatch against archived pending record")

    if (
        record.get("session") != session
        or not isinstance(record.get("output"), str)
        or record["output"] != raw_exec_out
        or Path(record["output"]).resolve() != resolved_out
        or not isinstance(record.get("config"), str)
        or not isinstance(record.get("assignment_record"), str)
    ):
        raise ValueError("archived pending record fields do not match session/output/manifest")

    if not _matches_failed_new_server_refusal(
        disk_receipt.get("error"),
        session=session,
        accelerator=accelerator,
        notebook_hash=nb_hash,
    ):
        raise ValueError("receipt refusal error nbh/session does not match bound notebook_hash")

    for log_path in ops_logs:
        if log_path.is_symlink() or not log_path.is_file():
            raise ValueError(f"missing or symlinked ops log: {log_path}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if any(found != session for found in CREATING_SESSION_RE.findall(log_text)):
            raise ValueError(f"foreign session recorded in ops log: {log_path}")
        for raw_url_nbh in ASSIGN_URL_NBH_RE.findall(log_text):
            if _validate_notebook_uuid4(raw_url_nbh, allow_assign_encoding=True) != nb_hash:
                raise ValueError(f"foreign nbh recorded in ops log: {log_path}")

    cfg_path = Path(record["config"])
    assign_path = Path(record["assignment_record"])
    if (
        not cfg_path.is_absolute()
        or cfg_path.name != "sessions.json"
        or cfg_path.parent.name != job_id
        or assign_path != cfg_path.with_suffix(".assignment.json")
    ):
        raise ValueError("archived pending record config/assignment_record not bound to job_id")

    if ops_names:
        expected_lane_cfgs = {
            (resolved_ops / "state" / name / job_id / "sessions.json").resolve()
            for name in ops_names
        }
        if cfg_path.resolve() not in expected_lane_cfgs:
            raise ValueError("archived pending record config path not bound to job ops state lane")

    if (
        cfg_path.is_symlink()
        or assign_path.exists()
        or assign_path.is_symlink()
        or (cfg_path.parent / "assignment.json").exists()
        or (cfg_path.parent / "assignment.json").is_symlink()
    ):
        raise ValueError("lane assignment record exists on disk for allegedly unallocated job")
    if cfg_path.exists():
        cfg_data = _read_regular_json_dict(cfg_path, "lane sessions.json")
        if cfg_data:
            raise ValueError("lane sessions.json contains recorded session state")
    if cfg_path.parent.is_dir() and not cfg_path.parent.is_symlink():
        _verify_optional_state_and_events(cfg_path.parent, job_id=job_id, session=session)

    _verify_notebook_not_replayed(
        resolved_ops,
        proof_paths,
        receipt_path,
        nb_hash,
        account_dir=pending_path.parent,
    )

    if receipt_sha == SHARD00_RECEIPT_SHA256:
        if (
            nb_hash != SHARD00_NOTEBOOK_HASH
            or pending_obj["sha256"] != SHARD00_PENDING_SHA256
            or exec_manifest_sha != SHARD00_EXEC_MANIFEST_SHA256
        ):
            raise ValueError(
                "notebook_hash/pending.sha256 does not match bound shard-00 session/manifest"
            )

    before = record.get("before")
    if (
        not isinstance(before, list)
        or any(not isinstance(ep, str) or not ep for ep in before)
        or len(before) != len(set(before))
    ):
        raise ValueError("archived pending record has invalid 'before' endpoint list")
    before_set = set(before)

    if "assignments" in accounting:
        acct_rows = accounting.get("assignments")
        if not isinstance(acct_rows, list) or any(not isinstance(r, dict) for r in acct_rows):
            raise ValueError("malformed receipt.accounting.assignments")
        acct_eps = [r.get("endpoint") for r in acct_rows]
        if any(not isinstance(ep, str) or not ep for ep in acct_eps) or set(acct_eps) != before_set:
            raise ValueError("pending.record.before disagrees with receipt.accounting.assignments")

    baseline_path = output_dir / "accounting-baseline.json"
    if baseline_path.exists() or baseline_path.is_symlink():
        baseline_obj = _read_regular_json_dict(baseline_path, "accounting-baseline.json")
        base_rows = baseline_obj.get("assignments")
        if (
            baseline_obj.get("unaccounted_endpoints") != []
            or not isinstance(base_rows, list)
            or any(not isinstance(r, dict) for r in base_rows)
        ):
            raise ValueError("malformed accounting-baseline.json")
        base_eps = [r.get("endpoint") for r in base_rows]
        if any(not isinstance(ep, str) or not ep for ep in base_eps) or set(base_eps) != before_set:
            raise ValueError("pending.record.before disagrees with accounting-baseline.json")

    # 7. Accounted sessions & membership reconciliation against sessions_stdout.
    current_set = _parse_sessions_stdout(proof.get("sessions_stdout"), session)
    membership = proof.get("membership")
    if not isinstance(membership, dict) or set(membership.keys()) != {
        "added",
        "removed",
        "retained",
    }:
        raise ValueError("unallocated proof missing complete membership evidence")

    added_map = membership["added"]
    removed_map = membership["removed"]
    retained_map = membership["retained"]
    if (
        not isinstance(added_map, dict)
        or not isinstance(removed_map, dict)
        or not isinstance(retained_map, dict)
        or set(added_map.keys()) != (current_set - before_set)
        or set(removed_map.keys()) != (before_set - current_set)
        or set(retained_map.keys()) != (current_set & before_set)
    ):
        raise ValueError("unallocated proof membership sets disagree with before/sessions_stdout")

    own_ops_mf_paths = {mf_path.resolve() for _, mf_path, _ in ops_matches}
    for bucket_name, bucket in (("added", added_map), ("retained", retained_map)):
        for ep, entry in bucket.items():
            if not isinstance(ep, str) or not ep or not isinstance(entry, dict):
                raise ValueError(f"malformed membership.{bucket_name} entry for {ep!r}")
            for sha_key in ("active_sha256", "executed_manifest_sha256", "manifest_sha256"):
                val = entry.get(sha_key)
                if not isinstance(val, str) or HEX64_RE.fullmatch(val) is None:
                    raise ValueError(f"invalid {sha_key} in membership.{bucket_name}[{ep!r}]")
            active_str = entry.get("active")
            if (
                not isinstance(active_str, str)
                or not active_str
                or Path(active_str).name == f"active-{session}.json"
            ):
                raise ValueError(f"invalid active record in membership.{bucket_name}[{ep!r}]")
            for mf_key, sha_key in (
                ("manifest", "manifest_sha256"),
                ("executed_manifest", "executed_manifest_sha256"),
            ):
                mf_str = entry.get(mf_key)
                if not isinstance(mf_str, str) or not mf_str:
                    raise ValueError(f"missing {mf_key} in membership.{bucket_name}[{ep!r}]")
                mf_p = Path(mf_str)
                if mf_p.is_symlink() or not mf_p.is_file():
                    raise ValueError(f"missing accounted session {mf_key} file on disk: {mf_p}")
                if digest(mf_p) != entry[sha_key]:
                    raise ValueError(f"hash mismatch on accounted session {mf_key}: {mf_p}")
                if (
                    mf_p.resolve() == exec_manifest_path.resolve()
                    or mf_p.resolve() in own_ops_mf_paths
                ):
                    raise ValueError("refused attempt owns an active session in membership")

    return {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "executed_manifest_path": str(exec_manifest_path.resolve()),
        "executed_manifest_sha256": exec_manifest_sha,
        "proof_paths": [str(pp.resolve()) for pp in proof_paths],
        "proof_sha256": proof_sha,
        "job_id": job_id,
        "session": session,
        "generation": generation,
    }


def is_proven_never_allocated_attempt(
    receipt: dict | None,
    output_dir: Path | None,
    *,
    ops_dir: Path | None = None,
    ops_name: str | None = None,
) -> bool:
    """Boolean predicate wrapper around ``verify_never_allocated_attempt``."""
    if not isinstance(output_dir, Path):
        return False
    try:
        verify_never_allocated_attempt(
            output_dir,
            receipt=receipt,
            ops_dir=ops_dir,
            ops_name=ops_name,
        )
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Hash-bound renewal-WebSocket zero-export transport evidence (Shard-12 Ruling Points 2, 3, 6)
# ---------------------------------------------------------------------------

TRANSPORT_EVIDENCE_SCHEMA = "watch6-v3-renewal-ws-transport-evidence/v1"
TRANSPORT_EVIDENCE_CLASSIFICATION = "RENEWAL_WEBSOCKET_CLOSED_ZERO_EXPORT"
OPS_MANIFEST_RELATIONSHIP = "exec_equals_ops_source_plus_accounted_other_assignments"
RENEWAL_WS_WRAPPER_PREFIX = "colab exec failed (1): "
RENEWAL_WS_TRUNCATED_BODY_LEN = 2000
EXACT_TERMINAL_WS_LINE = "WebSocketConnectionClosedException: Connection is already closed."
EXACT_WS_SEND_CODE = "if not self.sock or self.sock.send(text_data, ABNF.OPCODE_TEXT) == 0:"
EXACT_WS_RAISE_CODE = 'raise WebSocketConnectionClosedException("Connection is already closed.")'
EXPECTED_SUPERVISOR_PROVENANCE: dict[str, object] = {
    "cli_nonzero_truncation_chars": RENEWAL_WS_TRUNCATED_BODY_LEN,
    "columns_env": 240,
    "renewal_method": "renew_runtime_lease->poll_rpc(read_state|renew_lease)",
    "rpc_kernel_error_prefix": "kernel error despite CLI exit zero:",
    "substitutes_only": "missing_traceback_header_and_frame_paths",
}

ENDPOINT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")
RICH_BOTTOM_BORDER_RE = re.compile(r"^╰─{20,}╯$")
RICH_CODE_INNER_RE = re.compile(r"^(\s*)(❱\s*)?(\d+)\s+(.*)$")
EXIT_CODE_LINE_RE = re.compile(r"^EXIT=(-?\d+)\s*$")
ALLOCATED_INDEXED_JSON_RE = re.compile(
    r"^(?:poll-\d{6}|rate-calibration-\d{3}|spend-\d{6}|usage-\d{6})\.json$"
)

ALLOWED_RENEWAL_WS_FIXED_FILES: frozenset[str] = frozenset(
    {
        "manifest.json",
        "receipt.json",
        "hardware.json",
        "warmup-measurement.json",
        "rate-calibration.json",
        "upload-measurement.json",
        "source-delivery.json",
        "input-hashes.json",
        "accounting-baseline.json",
        "usage-allocated.json",
        "ready.json",
        "session-release.json",
        "state.json",
        "events.jsonl",
    }
)

FORBIDDEN_RENEWAL_WS_OUTPUT_ENTRIES: tuple[str, ...] = (
    "terminal.json",
    "runtime-measurement.json",
    "scientific-fault.json",
    "backup.json",
    "public-input-fetch-failed.json",
    "supervisor.log",
    "job.log",
    "performance.json",
    "operational.json",
    "events.json",
)

FORBIDDEN_RENEWAL_WS_OUTPUT_GLOBS: tuple[str, ...] = (
    "telemetry-*.json",
    "recovery-*.json",
    "runtime-measurement*",
    "scientific-fault*",
)

REQUIRED_TRANSPORT_EVIDENCE_KEYS: frozenset[str] = frozenset(
    {
        "schema",
        "classification",
        "identity",
        "receipt",
        "executed_manifest",
        "ops_manifests",
        "hardware",
        "polls",
        "operational_log",
        "supervisor",
        "timing",
        "error_signature",
    }
)
REQUIRED_FILE_DESC_KEYS: frozenset[str] = frozenset({"path", "sha256", "size"})
REQUIRED_IDENTITY_KEYS: frozenset[str] = frozenset(
    {
        "namespace",
        "science",
        "commit",
        "shard",
        "job_id",
        "session",
        "endpoint",
        "generation",
    }
)
REQUIRED_OPS_MANIFESTS_KEYS: frozenset[str] = frozenset(
    {
        "ops_name",
        "source_manifest",
        "live_manifest",
        "accounting_record",
        "relationship",
    }
)
REQUIRED_HARDWARE_DESC_KEYS: frozenset[str] = frozenset({"path", "sha256", "size", "ready", "kind"})
REQUIRED_POLL_DESC_KEYS: frozenset[str] = frozenset(
    {
        "index",
        "path",
        "sha256",
        "size",
        "status",
        "updated_at",
        "exports_count",
        "terminal",
    }
)
REQUIRED_OPLOG_DESC_KEYS: frozenset[str] = frozenset(
    {
        "path",
        "sha256",
        "size",
        "exit_code",
        "poll_lines_count",
        "max_reported_exports",
    }
)
REQUIRED_SUPERVISOR_DESC_KEYS: frozenset[str] = frozenset(
    {"supervisor_sha256", "supervisor_relpath", "provenance"}
)
REQUIRED_TIMING_KEYS: frozenset[str] = frozenset(
    {
        "failed_control_index",
        "failed_control_started_monotonic",
        "failed_control_ended_monotonic",
        "failed_cli_index",
        "failed_cli_started_monotonic",
        "failed_cli_ended_monotonic",
        "failed_cli_elapsed_seconds",
        "failed_cli_timeout_seconds",
        "stop_cli_index",
        "stop_cli_started_monotonic",
        "stop_cli_ended_monotonic",
        "sessions_cli_index",
        "sessions_cli_started_monotonic",
        "sessions_cli_ended_monotonic",
    }
)
REQUIRED_ERROR_SIG_KEYS: frozenset[str] = frozenset(
    {
        "error_sha256",
        "wrapper_prefix",
        "decoded_body_length",
        "terminal_exception_line",
        "websocket_send_line",
        "websocket_raise_line",
    }
)


def _is_finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))  # type: ignore[arg-type]


def _file_descriptor(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked {label}: {path}")
    resolved = path.resolve()
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError(f"empty {label} forbidden: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest(resolved),
        "size": int(size),
    }


_PINNED_SUPERVISOR_CLI_CALL_KEYS: frozenset[str] = frozenset(
    {
        "attempt",
        "command",
        "elapsed_seconds",
        "ended_monotonic",
        "operation",
        "started_monotonic",
        "status",
        "thread",
        "timeout_seconds",
    }
)
_HARDWARE_VCPU_KEYS: tuple[str, ...] = ("vcpu", "vcpus", "cpu_count", "min_vcpu")
_FORBIDDEN_NONNULL_FAULT_KEYS: frozenset[str] = frozenset(
    {
        "error",
        "reason",
        "traceback",
        "last_traceback_line",
        "scientific_fault",
        "unassign_error",
        "cleanup_error",
        "terminal_error",
        "exception",
        "fault",
    }
)
_ANYWHERE_EXCEPTION_OR_FAULT_RE = re.compile(
    r"(?:"
    r"\b(?:[A-Za-z_][A-Za-z0-9_.]*\.)?[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Fault|Exit)\s*:"
    r"|\bRuntimeError\b"
    r"|\bChildProcessError\b"
    r"|\bSubprocessError\b"
    r"|\bCalledProcessError\b"
    r"|\bBrokenProcessPool\b"
    r"|\bProcessLookupError\b"
    r"|\bchild\b[^\n]*\b(?:fail|crash|abort|error|fault|killed|died|nonzero|timeout)\b"
    r"|\bworker\b[^\n]*\b(?:fail|crash|abort|error|fault|killed|died|nonzero|timeout)\b"
    r"|\btraceback\b"
    r")",
    re.IGNORECASE,
)


def _assert_clean_text_value(text: str, label: str) -> None:
    if (
        has_non_transport_fault(text)
        or "Traceback (most recent call last)" in text
        or v3_admission._GENERIC_LOG_EXCEPTION_RE.search(text) is not None
        or _ANYWHERE_EXCEPTION_OR_FAULT_RE.search(text) is not None
    ):
        raise ValueError(f"child/scientific/integrity/timeout fault text in {label}")


def _assert_clean_operational_json(
    value: object,
    json_path: str,
    *,
    allow_root_receipt_error: bool = False,
    forbid_failure_class_and_returncode: bool = True,
) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            cur_path = f"{json_path}.{k}"
            is_allowed_receipt_err = allow_root_receipt_error and cur_path == "receipt.json.error"
            if k in _FORBIDDEN_NONNULL_FAULT_KEYS and v is not None and not is_allowed_receipt_err:
                raise ValueError(f"non-null operational fault field at {cur_path}")
            if (
                forbid_failure_class_and_returncode
                and k in {"failure_class", "returncode"}
                and v is not None
            ):
                raise ValueError(f"non-null operational fault/returncode field at {cur_path}")
            if not is_allowed_receipt_err:
                _assert_clean_operational_json(
                    v,
                    cur_path,
                    allow_root_receipt_error=False,
                    forbid_failure_class_and_returncode=forbid_failure_class_and_returncode,
                )
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _assert_clean_operational_json(
                item,
                f"{json_path}[{idx}]",
                allow_root_receipt_error=False,
                forbid_failure_class_and_returncode=forbid_failure_class_and_returncode,
            )
    elif isinstance(value, str):
        _assert_clean_text_value(value, json_path)


def _extract_rich_box_code_line(line: str) -> tuple[bool, str] | None:
    stripped = line.rstrip()
    if len(stripped) < 4 or not stripped.startswith("│") or not stripped.endswith("│"):
        return None
    inner = stripped[1:-1]
    match = RICH_CODE_INNER_RE.match(inner)
    if match is None:
        return None
    is_pointer = bool(match.group(2))
    code_raw = match.group(4)
    clean_code = re.sub(r"^(?:\s*│)*\s*", "", code_raw).strip()
    return is_pointer, clean_code


def _parse_renewal_ws_error_signature(error_raw: object) -> dict[str, object]:
    """Verify exact Refused('colab exec failed (1): <2000-char Rich WS send box + closed line>')."""
    if (
        not isinstance(error_raw, str)
        or not error_raw.startswith("Refused(")
        or not error_raw.endswith(")")
    ):
        raise ValueError("receipt.error lacks Refused(...) wrapper")
    if has_non_transport_fault(error_raw):
        raise ValueError("non-transport fault text detected in raw receipt.error")
    if "kernel error despite cli exit zero" in error_raw.lower():
        raise ValueError("RPC kernel error refusal forbidden in renewal transport signature")

    try:
        parsed = ast.parse(error_raw, mode="eval")
    except SyntaxError as exc:
        raise ValueError("malformed Refused(...) literal in receipt.error") from exc
    if (
        not isinstance(parsed.body, ast.Call)
        or not isinstance(parsed.body.func, ast.Name)
        or parsed.body.func.id != "Refused"
        or len(parsed.body.args) != 1
        or bool(parsed.body.keywords)
        or not isinstance(parsed.body.args[0], ast.Constant)
        or not isinstance(parsed.body.args[0].value, str)
    ):
        raise ValueError("receipt.error is not a single-string Refused('...') call")

    inner = parsed.body.args[0].value
    if has_non_transport_fault(inner):
        raise ValueError("non-transport fault text detected in decoded Refused payload")
    if "kernel error despite cli exit zero" in inner.lower():
        raise ValueError("RPC kernel error refusal forbidden in decoded Refused payload")
    if not inner.startswith(RENEWAL_WS_WRAPPER_PREFIX):
        raise ValueError("decoded Refused payload lacks exact 'colab exec failed (1): ' prefix")

    body = inner[len(RENEWAL_WS_WRAPPER_PREFIX) :]
    if len(body) != RENEWAL_WS_TRUNCATED_BODY_LEN:
        raise ValueError(
            f"decoded CLI failure output length {len(body)} != {RENEWAL_WS_TRUNCATED_BODY_LEN} "
            "(truncation provenance applies ONLY to 2,000-char truncated CLI output)"
        )

    visible_frames = v3_admission._FRAME_PATH_RE.findall(
        body
    ) + v3_admission._RICH_FRAME_PATH_RE.findall(body)
    for fp in visible_frames:
        if v3_admission._FORBIDDEN_FRAME_RE.search(fp) is not None:
            raise ValueError(f"forbidden runtime/scientific frame path in CLI output: {fp}")
        if (
            v3_admission._ALLOWED_TRANSPORT_FRAME_RE.search(fp) is None
            and "websocket" not in fp.lower()
        ):
            raise ValueError(f"unrecognized frame path in CLI output: {fp}")

    nonempty_lines = [ln.rstrip("\r") for ln in body.splitlines() if ln.strip()]
    if len(nonempty_lines) < 4:
        raise ValueError("truncated CLI output lacks Rich traceback box context")

    terminal_line = nonempty_lines[-1].strip()
    if terminal_line != EXACT_TERMINAL_WS_LINE:
        raise ValueError(
            f"terminal exception {terminal_line!r} != {EXACT_TERMINAL_WS_LINE!r} "
            "(general WebSocket or non-matching exceptions are excluded)"
        )

    border_line = nonempty_lines[-2].strip()
    if RICH_BOTTOM_BORDER_RE.fullmatch(border_line) is None:
        raise ValueError("terminal exception is not immediately preceded by Rich traceback border")

    first_nonempty = nonempty_lines[0].rstrip()
    if not first_nonempty.endswith(("│", "╯", "╮", "┤")):
        raise ValueError("leading truncated CLI line is outside Rich traceback box")
    for mid_line in nonempty_lines[1:-2]:
        s_mid = mid_line.strip()
        if not s_mid.startswith(("│", "╭", "├", "╰")) or not s_mid.endswith(("│", "╮", "┤", "╯")):
            raise ValueError("unboxed non-Rich line present inside truncated CLI output")

    extracted_code: list[tuple[bool, str]] = []
    for box_line in nonempty_lines[:-2]:
        parsed_code = _extract_rich_box_code_line(box_line)
        if parsed_code is not None:
            extracted_code.append(parsed_code)
        if parsed_code != (True, EXACT_WS_RAISE_CODE):
            _assert_clean_text_value(box_line, "receipt.error.rich_box")

    matched_ws_send_pair = any(
        extracted_code[idx] == (False, EXACT_WS_SEND_CODE)
        and extracted_code[idx + 1] == (True, EXACT_WS_RAISE_CODE)
        for idx in range(len(extracted_code) - 1)
    )
    if not matched_ws_send_pair:
        raise ValueError(
            "Rich traceback box lacks exact websocket-send guard and active raise context"
        )

    return {
        "error_sha256": hashlib.sha256(error_raw.encode("utf-8")).hexdigest(),
        "wrapper_prefix": RENEWAL_WS_WRAPPER_PREFIX,
        "decoded_body_length": RENEWAL_WS_TRUNCATED_BODY_LEN,
        "terminal_exception_line": EXACT_TERMINAL_WS_LINE,
        "websocket_send_line": EXACT_WS_SEND_CODE,
        "websocket_raise_line": EXACT_WS_RAISE_CODE,
    }


def _verify_allocated_zero_export_state_and_events(
    directory: Path,
    *,
    job_id: str,
    session: str,
    endpoint: str,
) -> None:
    events_json_path = directory / "events.json"
    if events_json_path.exists() or events_json_path.is_symlink():
        raise ValueError(f"forbidden events.json present: {events_json_path}")

    state_path = directory / "state.json"
    if state_path.exists() or state_path.is_symlink():
        state = _read_regular_json_dict(state_path, "state.json")
        if not state:
            raise ValueError(f"empty state.json forbidden: {state_path}")
        if "job_id" in state and state["job_id"] != job_id:
            raise ValueError("state.json job_id mismatch")
        if "session" in state and state["session"] != session:
            raise ValueError("state.json session mismatch")
        if "endpoint" in state and state["endpoint"] not in (None, endpoint):
            raise ValueError("state.json endpoint mismatch")
        if "status" in state and state["status"] not in {
            "INCOMPLETE",
            "running",
            "starting",
            "stopping",
            "stopped",
        }:
            raise ValueError(f"contradictory status in state.json: {state['status']!r}")
        if "failure_class" in state and state["failure_class"] not in {None, "SUPERVISOR"}:
            raise ValueError(f"contradictory failure_class in state.json: {state_path}")
        if state.get("terminal") is not None or state.get("returncode") is not None:
            raise ValueError("non-null terminal/returncode in state.json")
        if "exports" in state and state["exports"] not in ([], 0, None):
            raise ValueError("nonzero exports recorded in state.json")
        _assert_clean_operational_json(
            state,
            "state.json",
            forbid_failure_class_and_returncode=False,
        )

    events_path = directory / "events.jsonl"
    if events_path.exists() or events_path.is_symlink():
        if events_path.is_symlink() or not events_path.is_file():
            raise ValueError(f"missing or symlinked events.jsonl: {events_path}")
        try:
            raw_events = events_path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise ValueError(f"unreadable events.jsonl: {events_path}") from exc
        lines = [ln.strip() for ln in raw_events.splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"empty events.jsonl forbidden: {events_path}")
        for idx, line in enumerate(lines):
            try:
                ev = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"corrupt JSONL in {events_path} line {idx + 1}") from exc
            if not isinstance(ev, dict) or not ev:
                raise ValueError(f"non-object event in {events_path} line {idx + 1}")
            if "job_id" in ev and ev["job_id"] != job_id:
                raise ValueError("events.jsonl job_id mismatch")
            if "session" in ev and ev["session"] != session:
                raise ValueError("events.jsonl session mismatch")
            if "endpoint" in ev and ev["endpoint"] not in (None, endpoint):
                raise ValueError("events.jsonl endpoint mismatch")
            if ev.get("terminal") is not None:
                raise ValueError("non-null terminal in events.jsonl")
            if "exports" in ev and ev["exports"] not in ([], 0, None):
                raise ValueError("nonzero exports in events.jsonl")
            if ev.get("command") == "download" or ev.get("direction") == "download":
                raise ValueError("download event forbidden in zero-export attempt events.jsonl")
            _assert_clean_operational_json(ev, f"events.jsonl[{idx}]")


def _discover_single_ops_manifest_for_renewal_ws(
    ops_dir: Path,
    resolved_out: Path,
    ops_name: str | None = None,
) -> tuple[str, Path, dict, bool]:
    """Return unique (run_name, source_manifest_path, source_manifest_dict, is_retry_one)."""
    exact = _discover_ops_names_and_manifests(ops_dir, resolved_out, ops_name)
    if len(exact) > 1:
        raise ValueError(f"ambiguous ops manifests for {resolved_out}: {[m[1] for m in exact]}")
    if len(exact) == 1:
        matched_name, mf_path, mf_data = exact[0]
        return matched_name, mf_path, mf_data, False

    if resolved_out.name.endswith("-retry-1"):
        base_out = Path(str(resolved_out)[: -len("-retry-1")]).resolve()
        base_ops_name = (
            ops_name[: -len("-retry-1")]
            if isinstance(ops_name, str) and ops_name.endswith("-retry-1")
            else ops_name
        )
        retry_matches = _discover_ops_names_and_manifests(ops_dir, base_out, base_ops_name)
        if len(retry_matches) > 1:
            raise ValueError(
                f"ambiguous base ops manifests for retry-1 {resolved_out}: "
                f"{[m[1] for m in retry_matches]}"
            )
        if len(retry_matches) == 1:
            base_name, mf_path, mf_data = retry_matches[0]
            return f"{base_name}-retry-1", mf_path, mf_data, True

    raise ValueError(f"missing ops manifest in {ops_dir / 'manifests'} for {resolved_out}")


def _verify_ops_manifest_relationship(
    *,
    ops_dir: Path,
    run_name: str,
    source_mf_path: Path,
    source_mf: dict,
    exec_manifest: dict,
    is_retry_one: bool,
) -> dict[str, object]:
    source_desc = _file_descriptor(source_mf_path, "ops source manifest")
    expected_source = dict(source_mf)
    if is_retry_one:
        if isinstance(expected_source.get("output_dir"), str):
            expected_source["output_dir"] = f"{expected_source['output_dir']}-retry-1"
        if (
            isinstance(expected_source.get("job_id"), str)
            and exec_manifest.get("job_id") == f"{expected_source['job_id']}-retry-1"
        ):
            expected_source["job_id"] = f"{expected_source['job_id']}-retry-1"

    src_core = {k: v for k, v in expected_source.items() if k != "budget"}
    exec_core = {k: v for k, v in exec_manifest.items() if k != "budget"}
    src_budget = expected_source.get("budget")
    exec_budget = exec_manifest.get("budget")
    expected_budget_keys = {
        "other_assignments",
        "rate_units_per_hour",
        "reserve_units",
        "unit_cap",
    }
    if (
        src_core != exec_core
        or not isinstance(src_budget, dict)
        or not isinstance(exec_budget, dict)
        or set(src_budget.keys()) != expected_budget_keys
        or set(exec_budget.keys()) != expected_budget_keys
        or any(
            src_budget.get(k) != exec_budget.get(k)
            for k in ("rate_units_per_hour", "reserve_units", "unit_cap")
        )
        or not isinstance(exec_budget.get("other_assignments"), dict)
    ):
        raise ValueError(
            f"ops source manifest {source_mf_path} diverges from executed manifest beyond "
            "budget.other_assignments"
        )

    live_mf_path = ops_dir / "manifests" / f"{run_name}.live.json"
    live_desc: dict[str, object] | None = None
    if live_mf_path.is_symlink():
        raise ValueError(f"symlinked live ops manifest forbidden: {live_mf_path}")
    if live_mf_path.exists():
        live_desc = _file_descriptor(live_mf_path, "ops live manifest")
        live_data = _read_regular_json_dict(live_mf_path, "ops live manifest")
        if live_data != exec_manifest:
            raise ValueError(
                f"ops live manifest {live_mf_path} does not match executed manifest.json"
            )

    acct_rec_path = ops_dir / "dispatch" / run_name / f"{run_name}.accounting.json"
    acct_desc: dict[str, object] | None = None
    if acct_rec_path.is_symlink():
        raise ValueError(f"symlinked ops accounting record forbidden: {acct_rec_path}")
    if acct_rec_path.exists():
        acct_desc = _file_descriptor(acct_rec_path, "ops accounting record")
        acct_data = _read_regular_json_dict(acct_rec_path, "ops accounting record")
        raw_src = acct_data.get("source_manifest")
        if (
            not isinstance(raw_src, str)
            or not raw_src
            or Path(raw_src).resolve() != source_mf_path.resolve()
            or acct_data.get("source_manifest_sha256") != source_desc["sha256"]
            or acct_data.get("other_assignments") != exec_budget["other_assignments"]
        ):
            raise ValueError(f"ops accounting record binding mismatch: {acct_rec_path}")
        if live_desc is not None:
            raw_live = acct_data.get("manifest")
            if (
                not isinstance(raw_live, str)
                or not raw_live
                or Path(raw_live).resolve() != live_mf_path.resolve()
                or acct_data.get("manifest_sha256") != live_desc["sha256"]
            ):
                raise ValueError(
                    f"ops accounting record live manifest hash mismatch: {acct_rec_path}"
                )
        sanitized_acct = dict(acct_data)
        sanitized_acct.pop("source_manifest", None)
        sanitized_acct.pop("manifest", None)
        _assert_clean_operational_json(sanitized_acct, acct_rec_path.name)

    disp_run_dir = ops_dir / "dispatch" / run_name
    if disp_run_dir.is_symlink():
        raise ValueError(f"symlinked ops dispatch directory forbidden: {disp_run_dir}")
    if disp_run_dir.is_dir():
        for disp_item in sorted(disp_run_dir.iterdir()):
            if disp_item.is_symlink():
                raise ValueError(f"symlinked entry in ops dispatch directory: {disp_item}")
            if not disp_item.is_file() or disp_item == acct_rec_path:
                continue
            if disp_item.name.startswith("transport-evidence") or disp_item.name.endswith(
                (".sha256", ".complete")
            ):
                continue
            if disp_item.suffix == ".json":
                d_obj = _read_regular_json_dict(disp_item, disp_item.name)
                _assert_clean_operational_json(d_obj, f"dispatch/{run_name}/{disp_item.name}")
            else:
                d_txt = disp_item.read_text(encoding="utf-8", errors="replace")
                _assert_clean_text_value(d_txt, f"dispatch/{run_name}/{disp_item.name}")

    return {
        "ops_name": run_name,
        "source_manifest": source_desc,
        "live_manifest": live_desc,
        "accounting_record": acct_desc,
        "relationship": OPS_MANIFEST_RELATIONSHIP,
    }


def _verify_operational_and_queue_logs(
    *,
    ops_dir: Path,
    run_name: str,
    output_dir: Path,
    receipt_path: Path,
    receipt_error: str,
    job_id: str,
    session: str,
    generation: str,
) -> dict[str, object]:
    log_path = ops_dir / "logs" / f"{run_name}.log"
    log_desc = _file_descriptor(log_path, "operational log")
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"unreadable or non-UTF8 operational log: {log_path}") from exc

    exit_codes: list[int] = []
    incomplete_lines: list[str] = []
    poll_lines_count = 0
    max_reported_exports = 0
    expected_calib = (output_dir / "rate-calibration.json").resolve()
    expected_incomplete_prefix = f"INCOMPLETE: {receipt_error}; receipt: "

    for line_no, raw_line in enumerate(log_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        exit_match = EXIT_CODE_LINE_RE.match(stripped)
        if exit_match is not None:
            exit_codes.append(int(exit_match.group(1)))
            continue
        if raw_line.startswith("OPS_ALLOCATION_SUCCESS calibration="):
            calib_raw = raw_line.split("=", 1)[1].strip()
            allowed_calibs = {
                str(output_dir / "rate-calibration.json"),
                str(expected_calib),
            }
            if (
                not calib_raw
                or calib_raw not in allowed_calibs
                or Path(calib_raw).resolve() != expected_calib
            ):
                raise ValueError(
                    f"operational log calibration path mismatch at {log_path}:{line_no}"
                )
            continue
        if stripped.startswith("{"):
            try:
                jline = json.loads(stripped)
            except ValueError as exc:
                raise ValueError(f"corrupt JSON telemetry line at {log_path}:{line_no}") from exc
            if (
                not isinstance(jline, dict)
                or jline.get("job") != job_id
                or jline.get("status") not in {"starting", "running"}
                or jline.get("terminal") is not False
            ):
                raise ValueError(
                    f"contradictory telemetry status/terminal/job at {log_path}:{line_no}"
                )
            exp_val = jline.get("exports")
            if type(exp_val) is not int or exp_val != 0:
                raise ValueError(
                    f"nonzero or invalid exports={exp_val!r} in operational log "
                    f"at {log_path}:{line_no} (partial-export attempts excluded)"
                )
            max_reported_exports = max(max_reported_exports, exp_val)
            tail = jline.get("job_log_tail")
            if tail is not None and not isinstance(tail, str):
                raise ValueError(f"non-string job_log_tail at {log_path}:{line_no}")
            _assert_clean_operational_json(
                jline,
                f"{log_path.name}:{line_no}",
                forbid_failure_class_and_returncode=True,
            )
            poll_lines_count += 1
            continue
        if raw_line.startswith("INCOMPLETE: "):
            if not raw_line.startswith(expected_incomplete_prefix):
                raise ValueError(
                    f"operational log INCOMPLETE line differs from receipt.error at "
                    f"{log_path}:{line_no}"
                )
            rec_suffix = raw_line[len(expected_incomplete_prefix) :].strip()
            allowed_rec_paths = {str(receipt_path), str(receipt_path.resolve())}
            if (
                not rec_suffix
                or rec_suffix not in allowed_rec_paths
                or Path(rec_suffix).resolve() != receipt_path.resolve()
            ):
                raise ValueError(
                    f"operational log INCOMPLETE receipt path mismatch at {log_path}:{line_no}"
                )
            incomplete_lines.append(raw_line)
            continue

        raise ValueError(
            f"unexpected or fault line in operational log {log_path}:{line_no}: {stripped[:160]}"
        )

    if len(exit_codes) != 1 or exit_codes[0] == 0:
        raise ValueError(
            f"operational log requires single nonzero EXIT code, found {exit_codes!r}: {log_path}"
        )
    if len(incomplete_lines) != 1:
        raise ValueError(
            f"operational log requires exactly one matching INCOMPLETE line: {log_path}"
        )
    if poll_lines_count < 1:
        raise ValueError(f"operational log lacks running poll telemetry lines: {log_path}")

    logs_dir = ops_dir / "logs"
    if logs_dir.is_dir() and not logs_dir.is_symlink():
        manifests_dir = ops_dir / "manifests"
        dispatch_root = ops_dir / "dispatch"
        other_run_names: set[str] = set()
        if manifests_dir.is_dir() and not manifests_dir.is_symlink():
            for mf in manifests_dir.glob("*.json"):
                stem = mf.name[:-5]
                if stem.endswith(".live"):
                    stem = stem[:-5]
                if stem != run_name:
                    other_run_names.add(stem)
        if dispatch_root.is_dir() and not dispatch_root.is_symlink():
            for ddir in dispatch_root.iterdir():
                if ddir.is_dir() and ddir.name != run_name:
                    other_run_names.add(ddir.name)

        unique_needles = tuple(tok for tok in (job_id, session) if tok)
        run_name_re = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(run_name)}(?![A-Za-z0-9_-])")
        resolved_primary_log = log_path.resolve()
        for qlog in sorted(logs_dir.glob("*.log")):
            if qlog.is_symlink():
                raise ValueError(f"symlinked log file forbidden in ops/logs: {qlog}")
            if not qlog.is_file() or qlog.resolve() == resolved_primary_log:
                continue
            is_other_run_log = any(
                qlog.stem == other
                or qlog.name.startswith(f"{other}.")
                or qlog.name == f"{other}-retry-1.log"
                for other in other_run_names
            )
            qtext = qlog.read_text(encoding="utf-8", errors="replace")
            if is_other_run_log and not any(tok in qtext for tok in unique_needles):
                continue
            is_run_specific_log = not is_other_run_log and (
                qlog.stem == run_name
                or qlog.name.startswith(f"{run_name}.")
                or qlog.name.startswith(f"{run_name}-")
            )
            for qline_no, qline in enumerate(qtext.splitlines(), start=1):
                if is_other_run_log:
                    if not any(tok in qline for tok in unique_needles):
                        continue
                elif not is_run_specific_log:
                    matched_shared = any(tok in qline for tok in unique_needles) or (
                        run_name_re.search(qline) is not None
                    )
                    if (
                        not matched_shared
                        and generation
                        and generation in qline
                        and not any(other in qline for other in other_run_names)
                    ):
                        matched_shared = True
                    if not matched_shared:
                        continue
                sanitized = (
                    qline.replace(receipt_error, "")
                    .replace(str(receipt_path.resolve()), "")
                    .replace(str(receipt_path), "")
                    .replace(str(output_dir.resolve()), "")
                    .replace(str(output_dir), "")
                    .replace(str(ops_dir.resolve()), "")
                    .replace(str(ops_dir), "")
                )
                if sanitized.strip().startswith("{"):
                    try:
                        qj = json.loads(sanitized.strip())
                    except ValueError:
                        qj = None
                    if isinstance(qj, dict):
                        _assert_clean_operational_json(
                            qj,
                            f"{qlog.name}:{qline_no}",
                            forbid_failure_class_and_returncode=True,
                        )
                        continue
                _assert_clean_text_value(sanitized, f"{qlog.name}:{qline_no}")

    return {
        **log_desc,
        "exit_code": exit_codes[0],
        "poll_lines_count": poll_lines_count,
        "max_reported_exports": max_reported_exports,
    }


def _collect_verified_renewal_ws_evidence(
    output_dir: Path,
    *,
    receipt: dict | None = None,
    ops_dir: Path | None = None,
    ops_name: str | None = None,
    broker_root: Path | None = None,
) -> tuple[dict[str, object], Path, str, str]:
    """Validate all on-disk attempt artifacts and construct the canonical sidecar payload."""
    if not isinstance(output_dir, Path) or output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(f"invalid output_dir for renewal-WS transport evidence: {output_dir}")

    eff_broker = (broker_root if broker_root is not None else V3_BROKER_ROOT).resolve()
    verify_baseline_closure(eff_broker, require_head_at_baseline=True)
    sup_path = eff_broker / PINNED_SUPERVISOR_REL
    if sup_path.is_symlink() or not sup_path.is_file():
        raise ValueError(f"missing or symlinked broker supervisor.py: {sup_path}")
    if digest(sup_path) != PINNED_SUPERVISOR_SHA256:
        raise ValueError(
            f"broker supervisor.py SHA256 mismatch ({digest(sup_path)} != "
            f"{PINNED_SUPERVISOR_SHA256})"
        )
    broker_alloc = p.allocation()

    for forbidden_name in FORBIDDEN_RENEWAL_WS_OUTPUT_ENTRIES:
        f_target = output_dir / forbidden_name
        if f_target.exists() or f_target.is_symlink():
            raise ValueError(f"forbidden fault/terminal artifact present in output_dir: {f_target}")
    for pattern in FORBIDDEN_RENEWAL_WS_OUTPUT_GLOBS:
        if any(output_dir.glob(pattern)):
            raise ValueError(f"forbidden artifact matching {pattern} in {output_dir}")

    exports_dir = output_dir / "exports"
    if exports_dir.is_symlink():
        raise ValueError(f"symlinked exports directory forbidden: {exports_dir}")
    if exports_dir.exists():
        if not exports_dir.is_dir() or any(exports_dir.iterdir()):
            raise ValueError(
                f"partial or non-empty exports directory excluded from renewal-WS rule: "
                f"{exports_dir}"
            )

    for entry in sorted(output_dir.iterdir()):
        if entry.is_symlink():
            raise ValueError(f"symlinked entry forbidden in output_dir: {entry}")
        if entry.name == "staging":
            if not entry.is_dir():
                raise ValueError(f"staging must be a directory: {entry}")
            for child in sorted(entry.iterdir()):
                if (
                    child.is_symlink()
                    or not child.is_file()
                    or child.name not in ALLOWED_PREALLOC_STAGING_ENTRIES
                ):
                    raise ValueError(f"unexpected file inside staging: {child}")
        elif entry.name == "exports":
            continue
        elif entry.is_file() and (
            entry.name in ALLOWED_RENEWAL_WS_FIXED_FILES
            or ALLOCATED_INDEXED_JSON_RE.fullmatch(entry.name) is not None
        ):
            continue
        else:
            raise ValueError(f"unexpected or contradictory artifact in output_dir: {entry}")

    resolved_out = output_dir.resolve()
    receipt_path = output_dir / "receipt.json"
    exec_manifest_path = output_dir / "manifest.json"
    receipt_desc = _file_descriptor(receipt_path, "receipt.json")
    exec_mf_desc = _file_descriptor(exec_manifest_path, "executed manifest.json")
    disk_receipt = _read_regular_json_dict(receipt_path, "receipt.json")
    if receipt is not None and receipt != disk_receipt:
        raise ValueError("in-memory receipt differs from immutable on-disk receipt.json")
    exec_manifest = _read_regular_json_dict(exec_manifest_path, "executed manifest.json")

    if v3_admission.is_prestart_transport_failure(disk_receipt, output_dir):
        raise ValueError("prestart failure cannot be classified as renewal-WS compute failure")

    job_id = exec_manifest.get("job_id")
    commit = exec_manifest.get("commit")
    generation = disk_receipt.get("generation")
    session = disk_receipt.get("session")
    endpoint = disk_receipt.get("endpoint")
    raw_exec_out = exec_manifest.get("output_dir")
    accounting = disk_receipt.get("accounting")
    if (
        not isinstance(job_id, str)
        or JOB_ID_RE.fullmatch(job_id) is None
        or disk_receipt.get("job_id") != job_id
        or commit != BASELINE_COMMIT
        or disk_receipt.get("commit") != BASELINE_COMMIT
        or disk_receipt.get("manifest_sha256") != exec_mf_desc["sha256"]
        or not isinstance(raw_exec_out, str)
        or not raw_exec_out.strip()
        or Path(raw_exec_out).resolve() != resolved_out
        or not isinstance(generation, str)
        or HEX32_RE.fullmatch(generation) is None
        or not isinstance(session, str)
        or session != f"snake-{job_id}-{generation[:12]}"
        or not isinstance(endpoint, str)
        or ENDPOINT_RE.fullmatch(endpoint) is None
        or not isinstance(accounting, dict)
        or accounting.get("allocated_endpoint") != endpoint
        or accounting.get("unaccounted_endpoints") != []
        or disk_receipt.get("supervisor_sha256") != PINNED_SUPERVISOR_SHA256
    ):
        raise ValueError(
            "manifest/receipt/job/session/endpoint/generation/commit/supervisor binding mismatch"
        )

    binding = exec_manifest.get("watch6_v3")
    env = exec_manifest.get("env")
    req = exec_manifest.get("requirements")
    if (
        not isinstance(binding, dict)
        or not isinstance(env, dict)
        or not isinstance(req, dict)
        or exec_manifest.get("script") != p.SCRIPT
        or exec_manifest.get("device") != "cpu"
        or binding.get("namespace") != BASELINE_NAMESPACE
        or binding.get("science") != BASELINE_SCIENCE
        or env.get("WATCH6_SCIENCE") != BASELINE_SCIENCE
        or env.get("SNAKE_WATCH_SOURCE") != "live"
        or env.get("WATCH6_WORKERS") != str(p.WORKERS)
        or binding.get("primary_allocation") != broker_alloc
        or any(k in binding for k in ("reserve_allocation", "reserve_slots", "reserve_lineage"))
        or req.get("gpu") != "L4"
        or req.get("allowed_gpus") != ["L4"]
        or "prefer" in req
        or type(req.get("min_vcpu")) is not int
        or req["min_vcpu"] < 12
        or env.get("WATCH6_FIXTURE") != "0"
    ):
        raise ValueError("foreign science/namespace/allocation or non-L4 W4 manifest binding")

    for container_label, container in (
        ("manifest", exec_manifest),
        ("watch6_v3", binding),
        ("requirements", req),
    ):
        if "workers" in container and (
            type(container["workers"]) is not int or container["workers"] != p.WORKERS
        ):
            raise ValueError(
                f"invalid workers={container['workers']!r} in {container_label} "
                f"(expected {p.WORKERS})"
            )

    if "WATCH6_FROZEN_LAYOUT" in env:
        raw_layout = env["WATCH6_FROZEN_LAYOUT"]
        if not isinstance(raw_layout, str) or not raw_layout.strip():
            raise ValueError("malformed WATCH6_FROZEN_LAYOUT in executed manifest")
        try:
            frozen_layout = json.loads(raw_layout)
        except ValueError as exc:
            raise ValueError("corrupt JSON in WATCH6_FROZEN_LAYOUT") from exc
        if (
            not isinstance(frozen_layout, dict)
            or type(frozen_layout.get("workers")) is not int
            or frozen_layout["workers"] != p.WORKERS
            or type(frozen_layout.get("shards")) is not int
            or frozen_layout["shards"] != p.SHARDS
        ):
            raise ValueError("WATCH6_FROZEN_LAYOUT workers/shards mismatch (requires W=4, S=24)")

    try:
        total_shards = int(env.get("WATCH6_SHARDS", -1))
        shard = int(env.get("WATCH6_SHARD", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed WATCH6_SHARD/WATCH6_SHARDS in executed manifest") from exc
    if total_shards != p.SHARDS or shard not in range(p.SHARDS) or binding.get("shard") != shard:
        raise ValueError(f"invalid V3 shard index {shard} in executed manifest")

    if receipt_desc["sha256"] == SHARD12_RECEIPT_SHA256:
        if exec_mf_desc["sha256"] != SHARD12_EXEC_MANIFEST_SHA256 or shard != 12:
            raise ValueError("shard-12 receipt SHA256 paired with mismatched manifest or shard")

    if (
        disk_receipt.get("status") != "INCOMPLETE"
        or disk_receipt.get("failure_class") != "SUPERVISOR"
        or disk_receipt.get("stop_verified") is not True
        or disk_receipt.get("cleanup_error") is not None
        or disk_receipt.get("terminal") is not None
        or disk_receipt.get("exports") != []
        or disk_receipt.get("traceback") is not None
        or disk_receipt.get("last_traceback_line") is not None
    ):
        raise ValueError(
            "receipt status/failure_class/stop_verified/cleanup_error/terminal/exports vetoed"
        )

    for tx_key in ("transfers", "contents_transfers"):
        tx_list = disk_receipt.get(tx_key)
        if not isinstance(tx_list, list):
            raise ValueError(f"receipt.{tx_key} must be a list")
        for tx in tx_list:
            if not isinstance(tx, dict):
                raise ValueError(f"malformed entry in receipt.{tx_key}")
            direction = tx.get("direction") or tx.get("command")
            if direction == "download":
                raise ValueError(f"download transfer forbidden in zero-export {tx_key}")
            if tx_key == "contents_transfers" and tx.get("direction") != "upload":
                raise ValueError("non-upload contents_transfer in zero-export attempt")
            for path_field in ("remote", "local", "dest", "path"):
                pval = tx.get(path_field)
                if isinstance(pval, str) and "exports" in Path(pval).parts:
                    raise ValueError(f"export transfer path forbidden in {tx_key}: {pval}")

    hw_path = output_dir / "hardware.json"
    hw_desc = _file_descriptor(hw_path, "hardware.json")
    hw_data = _read_regular_json_dict(hw_path, "hardware.json")
    hw_info = hw_data.get("hardware")
    if (
        hw_data.get("ready") is not True
        or not isinstance(hw_info, dict)
        or hw_info.get("kind") != "L4"
    ):
        raise ValueError("hardware.json missing ready=True on L4")
    min_vcpu = req["min_vcpu"]
    present_vcpus: list[tuple[str, object]] = []
    for hw_loc, hw_obj in (("hardware.json", hw_data), ("hardware.json.hardware", hw_info)):
        for vcpu_key in _HARDWARE_VCPU_KEYS:
            if vcpu_key in hw_obj:
                present_vcpus.append((f"{hw_loc}.{vcpu_key}", hw_obj[vcpu_key]))
    if present_vcpus:
        for vcpu_loc, vcpu_val in present_vcpus:
            if type(vcpu_val) is not int or vcpu_val < min_vcpu:
                raise ValueError(f"hardware vCPU {vcpu_loc}={vcpu_val!r} fails min_vcpu={min_vcpu}")
        if len({int(v) for _, v in present_vcpus}) != 1:  # type: ignore[arg-type]
            raise ValueError("contradictory vCPU fields in hardware.json")
    else:
        hw_name = hw_info.get("name")
        hw_cc = hw_info.get("compute_cap")
        if (
            disk_receipt.get("supervisor_sha256") != PINNED_SUPERVISOR_SHA256
            or set(hw_data.keys()) != {"ready", "hardware"}
            or set(hw_info.keys()) != {"kind", "name", "compute_cap"}
            or not isinstance(hw_name, str)
            or "L4" not in hw_name
            or not isinstance(hw_cc, str)
            or not hw_cc.strip()
        ):
            raise ValueError(
                "hardware.json lacks explicit vcpu >= min_vcpu and does not match pinned "
                "supervisor GPU admission schema"
            )

    warmup_data = _read_regular_json_dict(
        output_dir / "warmup-measurement.json", "warmup-measurement.json"
    )
    if warmup_data.get("status") != "PASS":
        raise ValueError("warmup-measurement.json status != PASS")

    calib_data = _read_regular_json_dict(
        output_dir / "rate-calibration.json", "rate-calibration.json"
    )
    if calib_data.get("status") not in {"PASS", "WARNING_RATE_UNSTABLE"}:
        raise ValueError("rate-calibration.json status invalid")

    upload_data = _read_regular_json_dict(
        output_dir / "upload-measurement.json", "upload-measurement.json"
    )
    if upload_data.get("staging_completed") is False or not _is_finite_number(
        upload_data.get("elapsed_seconds")
    ):
        raise ValueError("upload-measurement.json incomplete or invalid")

    src_deliv = _read_regular_json_dict(output_dir / "source-delivery.json", "source-delivery.json")
    if src_deliv.get("commit") != BASELINE_COMMIT:
        raise ValueError("source-delivery.json commit mismatch")

    in_hashes = _read_regular_json_dict(output_dir / "input-hashes.json", "input-hashes.json")
    if not in_hashes or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or HEX64_RE.fullmatch(v) is None
        for k, v in in_hashes.items()
    ):
        raise ValueError("input-hashes.json empty or malformed")

    for spend_path in sorted(output_dir.glob("spend-*.json")):
        spend_data = _read_regular_json_dict(spend_path, spend_path.name)
        if spend_data.get("unit_cap_status") != "OK":
            raise ValueError(f"spend unit_cap_status != OK in {spend_path}")

    _verify_allocated_zero_export_state_and_events(
        output_dir,
        job_id=job_id,
        session=session,
        endpoint=endpoint,
    )

    poll_paths = sorted(output_dir.glob("poll-*.json"))
    if not poll_paths:
        raise ValueError("renewal-WS transport evidence requires nonempty clean poll-*.json files")
    polls_clean, audited_count = v3_admission._audit_poll_files(output_dir)
    if not polls_clean or audited_count != len(poll_paths):
        raise ValueError("poll-*.json audit failed or detected child/supervisor fault")

    polls_meta: list[dict[str, object]] = []
    prev_updated_at = -math.inf
    for idx, poll_path in enumerate(poll_paths):
        expected_name = f"poll-{idx:06d}.json"
        if poll_path.name != expected_name:
            raise ValueError(
                f"non-contiguous poll sequence: expected {expected_name}, found {poll_path.name}"
            )
        p_desc = _file_descriptor(poll_path, poll_path.name)
        poll_obj = _read_regular_json_dict(poll_path, poll_path.name)
        if poll_obj.get("exports") != [] or poll_obj.get("terminal") is not None:
            raise ValueError(f"poll {poll_path.name} reports nonzero exports or non-null terminal")
        p_state = poll_obj.get("state")
        if (
            not isinstance(p_state, dict)
            or p_state.get("status") not in {"starting", "running"}
            or p_state.get("job_id") != job_id
            or p_state.get("generation") != generation
            or p_state.get("returncode") is not None
            or p_state.get("reason") is not None
            or p_state.get("error") is not None
            or p_state.get("unassign_error") is not None
            or not _is_finite_number(p_state.get("started_at"))
            or not _is_finite_number(p_state.get("updated_at"))
        ):
            raise ValueError(f"poll {poll_path.name} state binding/health mismatch")
        started_at = float(p_state["started_at"])
        updated_at = float(p_state["updated_at"])
        if started_at <= 0 or updated_at < started_at or updated_at < prev_updated_at:
            raise ValueError(f"non-monotonic timestamps in {poll_path.name}")
        prev_updated_at = updated_at
        polls_meta.append(
            {
                "index": idx,
                **p_desc,
                "status": str(p_state["status"]),
                "updated_at": p_state["updated_at"],
                "exports_count": 0,
                "terminal": None,
            }
        )

    control_ops = disk_receipt.get("control_operations")
    if (
        not isinstance(control_ops, list)
        or not control_ops
        or not all(isinstance(op, dict) for op in control_ops)
    ):
        raise ValueError("receipt.control_operations missing or malformed")
    for earlier_op in control_ops[:-1]:
        if earlier_op.get("thread") == "MainThread" and earlier_op.get("status") != "completed":
            raise ValueError("earlier MainThread control operation failed before terminal renewal")
        if isinstance(earlier_op.get("error"), str):
            _assert_clean_text_value(earlier_op["error"], "control_operations.error")

    failed_control_index = len(control_ops) - 1
    failed_ctrl = control_ops[failed_control_index]
    ctrl_start = failed_ctrl.get("started_monotonic")
    ctrl_end = failed_ctrl.get("ended_monotonic")
    if (
        failed_ctrl.get("operation") != "renewal"
        or failed_ctrl.get("thread") != "MainThread"
        or failed_ctrl.get("status") != "failed"
        or not _is_finite_number(ctrl_start)
        or not _is_finite_number(ctrl_end)
        or not (0.0 < float(ctrl_start) <= float(ctrl_end))  # type: ignore[arg-type]
    ):
        raise ValueError(
            "terminal control_operations entry is not a failed MainThread renewal interval"
        )

    cli_calls = disk_receipt.get("cli_calls")
    if (
        not isinstance(cli_calls, list)
        or len(cli_calls) < 4
        or not all(isinstance(c, dict) for c in cli_calls)
    ):
        raise ValueError("receipt.cli_calls missing or malformed")

    failed_main_cli_indices = [
        idx
        for idx, call in enumerate(cli_calls)
        if call.get("thread") == "MainThread" and call.get("status") != "completed"
    ]
    if len(failed_main_cli_indices) != 1:
        raise ValueError(
            f"expected exactly 1 non-completed MainThread CLI call, found {failed_main_cli_indices}"
        )
    failed_cli_index = failed_main_cli_indices[0]
    has_any_returncode_key = any("returncode" in c for c in cli_calls)
    if has_any_returncode_key:
        for call_idx, call_obj in enumerate(cli_calls):
            if "returncode" in call_obj and type(call_obj["returncode"]) is not int:
                raise ValueError(
                    f"non-integer returncode={call_obj['returncode']!r} in cli_calls[{call_idx}]"
                )
            if call_obj.get("thread") == "MainThread":
                if "returncode" not in call_obj or type(call_obj["returncode"]) is not int:
                    raise ValueError(
                        f"missing or null returncode on MainThread cli_calls[{call_idx}]"
                    )
            elif "returncode" in call_obj and call_obj.get("status") == "completed":
                if call_obj["returncode"] != 0:
                    raise ValueError(
                        f"completed background CLI call had nonzero returncode at [{call_idx}]"
                    )
    else:
        if disk_receipt.get("supervisor_sha256") != PINNED_SUPERVISOR_SHA256:
            raise ValueError("cli_calls missing returncode under unpinned supervisor")
        for call_idx, call_obj in enumerate(cli_calls):
            c_el = call_obj.get("elapsed_seconds")
            c_to = call_obj.get("timeout_seconds")
            if (
                set(call_obj.keys()) != _PINNED_SUPERVISOR_CLI_CALL_KEYS
                or type(call_obj.get("attempt")) is not int
                or call_obj["attempt"] < 1
                or not _is_finite_number(call_obj.get("started_monotonic"))
                or not _is_finite_number(call_obj.get("ended_monotonic"))
                or not _is_finite_number(c_el)
                or not _is_finite_number(c_to)
                or not (0.0 <= float(c_el) < float(c_to))  # type: ignore[arg-type]
            ):
                raise ValueError(
                    f"cli_calls[{call_idx}] violates pinned supervisor call_records schema"
                )

    for prev_call in cli_calls[:failed_cli_index]:
        if prev_call.get("thread") == "MainThread":
            if prev_call.get("status") != "completed":
                raise ValueError("earlier MainThread CLI call was not completed")
            if has_any_returncode_key and prev_call.get("returncode") != 0:
                raise ValueError("earlier MainThread CLI call had nonzero returncode")

    failed_cli = cli_calls[failed_cli_index]
    cli_start = failed_cli.get("started_monotonic")
    cli_end = failed_cli.get("ended_monotonic")
    cli_elapsed = failed_cli.get("elapsed_seconds")
    cli_timeout = failed_cli.get("timeout_seconds")
    if (
        failed_cli.get("command") != "exec"
        or failed_cli.get("operation") != "renewal"
        or failed_cli.get("thread") != "MainThread"
        or failed_cli.get("status") != "failed"
        or failed_cli.get("attempt") != 1
        or (has_any_returncode_key and failed_cli.get("returncode") != 1)
        or not _is_finite_number(cli_start)
        or not _is_finite_number(cli_end)
        or not _is_finite_number(cli_elapsed)
        or not _is_finite_number(cli_timeout)
        or not (
            float(ctrl_start)  # type: ignore[arg-type]
            <= float(cli_start)  # type: ignore[arg-type]
            <= float(cli_end)  # type: ignore[arg-type]
            <= float(ctrl_end)  # type: ignore[arg-type]
        )
        or not (0.0 < float(cli_elapsed) < float(cli_timeout))  # type: ignore[arg-type]
    ):
        raise ValueError(
            "failed MainThread exec/renewal CLI interval is missing, lacks failing returncode, "
            "timed out, or is not nested inside failed renewal control interval"
        )

    post_calls = cli_calls[failed_cli_index + 1 :]
    if len(post_calls) not in (2, 3):
        raise ValueError(
            "failed renewal CLI call must be followed by stop, sessions, and optional usage only"
        )
    for post_call in post_calls:
        if (
            post_call.get("thread") != "MainThread"
            or post_call.get("operation") != "control"
            or post_call.get("status") != "completed"
            or (has_any_returncode_key and post_call.get("returncode") != 0)
            or not _is_finite_number(post_call.get("started_monotonic"))
            or not _is_finite_number(post_call.get("ended_monotonic"))
        ):
            raise ValueError("post-failure cleanup CLI call incomplete, non-zero, or malformed")

    stop_cli_index = failed_cli_index + 1
    sessions_cli_index = failed_cli_index + 2
    stop_cli = cli_calls[stop_cli_index]
    sessions_cli = cli_calls[sessions_cli_index]
    stop_start = float(stop_cli["started_monotonic"])
    stop_end = float(stop_cli["ended_monotonic"])
    sess_start = float(sessions_cli["started_monotonic"])
    sess_end = float(sessions_cli["ended_monotonic"])
    if (
        stop_cli.get("command") != "stop"
        or sessions_cli.get("command") != "sessions"
        or not (
            float(ctrl_end)  # type: ignore[arg-type]
            <= stop_start
            <= stop_end
            <= sess_start
            <= sess_end
        )
    ):
        raise ValueError(
            "post-failure stop -> sessions verification ordering or command sequence invalid"
        )
    if len(post_calls) == 3:
        usage_cli = post_calls[2]
        u_start = float(usage_cli["started_monotonic"])
        u_end = float(usage_cli["ended_monotonic"])
        if usage_cli.get("command") != "usage" or not (sess_end <= u_start <= u_end):
            raise ValueError("optional post-sessions cleanup call must be monotonic 'usage'")

    error_sig = _parse_renewal_ws_error_signature(disk_receipt.get("error"))
    verified_error_str = str(disk_receipt["error"])

    for json_file in sorted(output_dir.glob("*.json")):
        if json_file.name == "manifest.json":
            continue
        jdata = _read_regular_json_dict(json_file, json_file.name)
        _assert_clean_operational_json(
            jdata,
            json_file.name,
            allow_root_receipt_error=(json_file.name == "receipt.json"),
            forbid_failure_class_and_returncode=(
                json_file.name not in {"receipt.json", "state.json"}
            ),
        )

    resolved_ops = (ops_dir or DEFAULT_OPS_DIR).resolve()
    if resolved_ops.is_symlink() or not resolved_ops.is_dir():
        raise ValueError(f"missing or symlinked ops_dir: {resolved_ops}")

    run_name, source_mf_path, source_mf, is_retry_one = (
        _discover_single_ops_manifest_for_renewal_ws(
            resolved_ops,
            resolved_out,
            ops_name,
        )
    )
    ops_manifests_obj = _verify_ops_manifest_relationship(
        ops_dir=resolved_ops,
        run_name=run_name,
        source_mf_path=source_mf_path,
        source_mf=source_mf,
        exec_manifest=exec_manifest,
        is_retry_one=is_retry_one,
    )
    oplog_obj = _verify_operational_and_queue_logs(
        ops_dir=resolved_ops,
        run_name=run_name,
        output_dir=resolved_out,
        receipt_path=receipt_path,
        receipt_error=verified_error_str,
        job_id=job_id,
        session=session,
        generation=generation,
    )

    lane_state_dir = resolved_ops / "state" / run_name / job_id
    if lane_state_dir.is_symlink():
        raise ValueError(f"symlinked ops lane state directory forbidden: {lane_state_dir}")
    if lane_state_dir.is_dir():
        _verify_allocated_zero_export_state_and_events(
            lane_state_dir,
            job_id=job_id,
            session=session,
            endpoint=endpoint,
        )

    expected_payload: dict[str, object] = {
        "schema": TRANSPORT_EVIDENCE_SCHEMA,
        "classification": TRANSPORT_EVIDENCE_CLASSIFICATION,
        "identity": {
            "namespace": BASELINE_NAMESPACE,
            "science": BASELINE_SCIENCE,
            "commit": BASELINE_COMMIT,
            "shard": shard,
            "job_id": job_id,
            "session": session,
            "endpoint": endpoint,
            "generation": generation,
        },
        "receipt": receipt_desc,
        "executed_manifest": exec_mf_desc,
        "ops_manifests": ops_manifests_obj,
        "hardware": {
            **hw_desc,
            "ready": True,
            "kind": "L4",
        },
        "polls": polls_meta,
        "operational_log": oplog_obj,
        "supervisor": {
            "supervisor_sha256": PINNED_SUPERVISOR_SHA256,
            "supervisor_relpath": PINNED_SUPERVISOR_REL,
            "provenance": dict(EXPECTED_SUPERVISOR_PROVENANCE),
        },
        "timing": {
            "failed_control_index": failed_control_index,
            "failed_control_started_monotonic": ctrl_start,
            "failed_control_ended_monotonic": ctrl_end,
            "failed_cli_index": failed_cli_index,
            "failed_cli_started_monotonic": cli_start,
            "failed_cli_ended_monotonic": cli_end,
            "failed_cli_elapsed_seconds": cli_elapsed,
            "failed_cli_timeout_seconds": cli_timeout,
            "stop_cli_index": stop_cli_index,
            "stop_cli_started_monotonic": stop_cli["started_monotonic"],
            "stop_cli_ended_monotonic": stop_cli["ended_monotonic"],
            "sessions_cli_index": sessions_cli_index,
            "sessions_cli_started_monotonic": sessions_cli["started_monotonic"],
            "sessions_cli_ended_monotonic": sessions_cli["ended_monotonic"],
        },
        "error_signature": error_sig,
    }
    return expected_payload, resolved_ops, run_name, job_id


def _resolve_transport_evidence_dirs(
    ops_dir: Path,
    evidence_dir: Path | None = None,
) -> list[Path]:
    if evidence_dir is not None:
        if evidence_dir.is_symlink() or not evidence_dir.is_dir():
            raise ValueError(f"missing or symlinked transport evidence_dir: {evidence_dir}")
        return [evidence_dir.resolve()]
    env_raw = os.environ.get("WATCH6_V3_TRANSPORT_EVIDENCE_DIR")
    if env_raw:
        env_dir = Path(env_raw)
        if env_dir.is_symlink() or not env_dir.is_dir():
            raise ValueError(f"missing or symlinked WATCH6_V3_TRANSPORT_EVIDENCE_DIR: {env_dir}")
        return [env_dir.resolve()]
    if ops_dir.resolve() == DEFAULT_OPS_DIR.resolve():
        found_dirs: list[Path] = []
        seen_dirs: set[Path] = set()
        shard12_ev_parent = artifacts_root() / Path(SHARD12_EVIDENCE_SIDECAR_REL).parent
        for cand_dir in (default_artifact_dir(), shard12_ev_parent):
            if cand_dir.is_symlink():
                raise ValueError(f"symlinked default transport evidence dir forbidden: {cand_dir}")
            if cand_dir.is_dir():
                resolved_cand = cand_dir.resolve()
                if resolved_cand not in seen_dirs:
                    seen_dirs.add(resolved_cand)
                    found_dirs.append(resolved_cand)
        return found_dirs
    return []


def _discover_transport_evidence_paths(
    ops_dir: Path,
    receipt_path: Path,
    run_name: str,
    job_id: str,
    *,
    evidence_dir: Path | None = None,
) -> list[Path]:
    resolved_receipt = receipt_path.resolve()
    extra_dirs = _resolve_transport_evidence_dirs(ops_dir, evidence_dir)
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add(cand: Path) -> None:
        if cand.is_symlink():
            raise ValueError(f"symlinked transport evidence sidecar forbidden: {cand}")
        if cand.exists():
            if not cand.is_file():
                raise ValueError(f"non-regular transport evidence sidecar path: {cand}")
            key = cand.resolve()
            if key not in seen:
                seen.add(key)
                candidates.append(cand)

    disp_dir = ops_dir / "dispatch" / run_name
    if disp_dir.is_symlink():
        raise ValueError(f"symlinked ops dispatch directory forbidden: {disp_dir}")
    if disp_dir.is_dir():
        for item in sorted(disp_dir.glob("transport-evidence*.json")):
            _add(item)

    shared_dirs = [ops_dir, *extra_dirs]
    exact_name_re = re.compile(rf"^transport-evidence-{re.escape(run_name)}(?:-\d+)?\.json$")
    for sdir in shared_dirs:
        for item in sorted(sdir.glob("transport-evidence*.json")):
            if item.is_symlink():
                raise ValueError(f"symlinked transport evidence sidecar forbidden: {item}")
            if not item.is_file():
                continue
            if exact_name_re.fullmatch(item.name) is not None:
                _add(item)
                continue
            verify_immutable(item)
            payload = _read_regular_json_dict(item, "transport evidence sidecar")
            rec_obj = payload.get("receipt")
            rec_path_raw = rec_obj.get("path") if isinstance(rec_obj, dict) else None
            ops_obj = payload.get("ops_manifests")
            cand_ops = ops_obj.get("ops_name") if isinstance(ops_obj, dict) else None
            id_obj = payload.get("identity")
            cand_job = id_obj.get("job_id") if isinstance(id_obj, dict) else None
            path_matches = False
            if isinstance(rec_path_raw, str) and rec_path_raw:
                path_matches = Path(rec_path_raw).resolve() == resolved_receipt
            if path_matches or cand_ops == run_name or cand_job == job_id:
                _add(item)

    if not candidates and (ops_dir / "dispatch").is_dir():
        for item in sorted((ops_dir / "dispatch").glob("*/transport-evidence*.json")):
            if item.is_symlink():
                raise ValueError(f"symlinked transport evidence sidecar forbidden: {item}")
            if not item.is_file():
                continue
            verify_immutable(item)
            payload = _read_regular_json_dict(item, "transport evidence sidecar")
            rec_obj = payload.get("receipt")
            rec_path_raw = rec_obj.get("path") if isinstance(rec_obj, dict) else None
            if (
                isinstance(rec_path_raw, str)
                and rec_path_raw
                and Path(rec_path_raw).resolve() == resolved_receipt
            ):
                _add(item)

    return candidates


def _validate_transport_evidence_schema_keys(payload: dict) -> None:
    if set(payload.keys()) != REQUIRED_TRANSPORT_EVIDENCE_KEYS:
        raise ValueError("transport evidence sidecar top-level keyset mismatch")
    for field, expected_keys in (
        ("identity", REQUIRED_IDENTITY_KEYS),
        ("receipt", REQUIRED_FILE_DESC_KEYS),
        ("executed_manifest", REQUIRED_FILE_DESC_KEYS),
        ("ops_manifests", REQUIRED_OPS_MANIFESTS_KEYS),
        ("hardware", REQUIRED_HARDWARE_DESC_KEYS),
        ("operational_log", REQUIRED_OPLOG_DESC_KEYS),
        ("supervisor", REQUIRED_SUPERVISOR_DESC_KEYS),
        ("timing", REQUIRED_TIMING_KEYS),
        ("error_signature", REQUIRED_ERROR_SIG_KEYS),
    ):
        sub = payload.get(field)
        if not isinstance(sub, dict) or set(sub.keys()) != expected_keys:
            raise ValueError(f"transport evidence sidecar {field} keyset mismatch")
    ops_sub = payload["ops_manifests"]
    if (
        not isinstance(ops_sub.get("source_manifest"), dict)
        or set(ops_sub["source_manifest"].keys()) != REQUIRED_FILE_DESC_KEYS
    ):
        raise ValueError("transport evidence sidecar source_manifest keyset mismatch")
    for opt_field in ("live_manifest", "accounting_record"):
        opt_val = ops_sub.get(opt_field)
        if opt_val is not None and (
            not isinstance(opt_val, dict) or set(opt_val.keys()) != REQUIRED_FILE_DESC_KEYS
        ):
            raise ValueError(f"transport evidence sidecar {opt_field} keyset mismatch")
    polls = payload.get("polls")
    if not isinstance(polls, list) or not polls:
        raise ValueError("transport evidence sidecar polls must be a nonempty list")
    for poll_entry in polls:
        if not isinstance(poll_entry, dict) or set(poll_entry.keys()) != REQUIRED_POLL_DESC_KEYS:
            raise ValueError("transport evidence sidecar poll entry keyset mismatch")


def build_renewal_ws_transport_evidence(
    output_dir: Path,
    *,
    ops_dir: Path | None = None,
    ops_name: str | None = None,
    broker_root: Path | None = None,
) -> dict[str, object]:
    """Build canonical hash-bound transport-evidence payload after full fail-closed verification."""
    payload, _, _, _ = _collect_verified_renewal_ws_evidence(
        output_dir,
        ops_dir=ops_dir,
        ops_name=ops_name,
        broker_root=broker_root,
    )
    _validate_transport_evidence_schema_keys(payload)
    return payload


def publish_renewal_ws_transport_evidence(
    output_dir: Path,
    out_path: Path,
    *,
    ops_dir: Path | None = None,
    ops_name: str | None = None,
    broker_root: Path | None = None,
) -> dict[str, object]:
    """Verify attempt evidence, publish immutable sidecar (.sha256 + .complete), and self-verify."""
    if out_path.is_symlink():
        raise ValueError(f"symlinked sidecar output path forbidden: {out_path}")
    with isolated_broker_execution("proof-publish-renewal-ws", broker_root=broker_root):
        payload = build_renewal_ws_transport_evidence(
            output_dir,
            ops_dir=ops_dir,
            ops_name=ops_name,
            broker_root=broker_root,
        )
        publish_json(out_path, payload)
        return verify_renewal_ws_transport_evidence(
            output_dir,
            ops_dir=ops_dir,
            ops_name=ops_name,
            evidence_dir=out_path.resolve().parent,
            broker_root=broker_root,
        )


def verify_renewal_ws_transport_evidence(
    output_dir: Path,
    *,
    receipt: dict | None = None,
    ops_dir: Path | None = None,
    ops_name: str | None = None,
    evidence_dir: Path | None = None,
    broker_root: Path | None = None,
) -> dict[str, object]:
    """Verify immutable hash-bound renewal-WS transport-evidence sidecar and full conjunction."""
    expected_payload, resolved_ops, run_name, job_id = _collect_verified_renewal_ws_evidence(
        output_dir,
        receipt=receipt,
        ops_dir=ops_dir,
        ops_name=ops_name,
        broker_root=broker_root,
    )
    receipt_path = (output_dir / "receipt.json").resolve()
    sidecar_paths = _discover_transport_evidence_paths(
        resolved_ops,
        receipt_path,
        run_name,
        job_id,
        evidence_dir=evidence_dir,
    )
    if not sidecar_paths:
        raise ValueError(
            f"missing immutable {TRANSPORT_EVIDENCE_SCHEMA} sidecar for {receipt_path}"
        )

    sidecar_shas: set[str] = set()
    for sp in sidecar_paths:
        desc = verify_immutable(sp)
        sidecar_shas.add(str(desc["sha256"]))
        parsed_sidecar = _read_regular_json_dict(sp, "transport evidence sidecar")
        _validate_transport_evidence_schema_keys(parsed_sidecar)
        if parsed_sidecar != expected_payload:
            raise ValueError(
                f"transport evidence sidecar {sp} disagrees with verified on-disk artifacts"
            )

    if len(sidecar_shas) != 1:
        raise ValueError(
            f"ambiguous/conflicting transport evidence sidecar copies for {receipt_path}: "
            f"{sidecar_paths}"
        )

    id_info = expected_payload["identity"]
    assert isinstance(id_info, dict)
    rec_info = expected_payload["receipt"]
    assert isinstance(rec_info, dict)
    mf_info = expected_payload["executed_manifest"]
    assert isinstance(mf_info, dict)
    polls_info = expected_payload["polls"]
    assert isinstance(polls_info, list)

    return {
        "schema": TRANSPORT_EVIDENCE_SCHEMA,
        "classification": TRANSPORT_EVIDENCE_CLASSIFICATION,
        "shard": id_info["shard"],
        "ops_name": run_name,
        "job_id": job_id,
        "session": id_info["session"],
        "endpoint": id_info["endpoint"],
        "generation": id_info["generation"],
        "receipt_path": rec_info["path"],
        "receipt_sha256": rec_info["sha256"],
        "executed_manifest_path": mf_info["path"],
        "executed_manifest_sha256": mf_info["sha256"],
        "supervisor_sha256": PINNED_SUPERVISOR_SHA256,
        "polls_count": len(polls_info),
        "sidecar_paths": [str(sp.resolve()) for sp in sidecar_paths],
        "sidecar_sha256": next(iter(sidecar_shas)),
    }


def is_proven_renewal_ws_transport_failure(
    receipt: dict | None,
    output_dir: Path | None,
    *,
    ops_dir: Path | None = None,
    ops_name: str | None = None,
    evidence_dir: Path | None = None,
    broker_root: Path | None = None,
) -> bool:
    """Boolean predicate wrapper around ``verify_renewal_ws_transport_evidence``."""
    if not isinstance(output_dir, Path):
        return False
    try:
        verify_renewal_ws_transport_evidence(
            output_dir,
            receipt=receipt,
            ops_dir=ops_dir,
            ops_name=ops_name,
            evidence_dir=evidence_dir,
            broker_root=broker_root,
        )
    except ValueError:
        return False
    return True


def main() -> None:
    ensure_cli_dash_b("tools.watch6_v3_accounting.proof")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pub = sub.add_parser("publish-renewal-ws", help="Publish immutable renewal-WS sidecar")
    p_pub.add_argument("--output-dir", type=Path, required=True)
    p_pub.add_argument("--out", type=Path, required=True)
    p_pub.add_argument("--ops-dir", type=Path, default=None)
    p_pub.add_argument("--ops-name", type=str, default=None)
    p_pub.add_argument(
        "--broker-root", "--broker", dest="broker_root", type=Path, default=V3_BROKER_ROOT
    )

    p_ver = sub.add_parser("verify-renewal-ws", help="Verify immutable renewal-WS sidecar")
    p_ver.add_argument("--output-dir", type=Path, required=True)
    p_ver.add_argument("--ops-dir", type=Path, default=None)
    p_ver.add_argument("--ops-name", type=str, default=None)
    p_ver.add_argument("--evidence-dir", type=Path, default=None)
    p_ver.add_argument(
        "--broker-root", "--broker", dest="broker_root", type=Path, default=V3_BROKER_ROOT
    )

    args = parser.parse_args()
    with isolated_broker_execution("proof-cli", broker_root=args.broker_root):
        if args.cmd == "publish-renewal-ws":
            summary = publish_renewal_ws_transport_evidence(
                args.output_dir.resolve(),
                args.out.resolve(),
                ops_dir=args.ops_dir.resolve() if args.ops_dir else None,
                ops_name=args.ops_name,
                broker_root=args.broker_root,
            )
        else:
            summary = verify_renewal_ws_transport_evidence(
                args.output_dir.resolve(),
                ops_dir=args.ops_dir.resolve() if args.ops_dir else None,
                ops_name=args.ops_name,
                evidence_dir=args.evidence_dir.resolve() if args.evidence_dir else None,
                broker_root=args.broker_root,
            )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
