"""Fail-closed verifier for PROVEN never-allocated attempts (Astra Ruling Point 4).

Only a proven never-allocated attempt satisfying every item in Ruling Point 4 may retain
``cleanup_error == "ALLOCATION_UNRESOLVED: retained pending record"`` with
``stop_verified is False``. Receipts are never rewritten and ``stop_verified=True`` is never
manufactured. Allocated or ambiguous attempts continue to require verified stop.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import uuid
from pathlib import Path

from research.watch6_search_efficacy_v3.admission import has_non_transport_fault
from research.watch6_search_efficacy_v3.dispatch import (
    ALLOCATION_UNAVAILABLE,
    DEFAULT_OPS_DIR,
    _log_has_exit,
)
from tools.ops.package_spec import digest
from tools.watch6_v3_accounting import (
    RETAINED_PENDING_CLEANUP_ERROR,
    SHARD00_EXEC_MANIFEST_SHA256,
    SHARD00_NOTEBOOK_HASH,
    SHARD00_PENDING_SHA256,
    SHARD00_RECEIPT_SHA256,
)

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
