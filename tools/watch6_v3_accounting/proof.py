"""Fail-closed verifier for PROVEN never-allocated attempts (Astra Ruling Point 4).

Only a proven never-allocated attempt satisfying every item in Ruling Point 4 may retain
``cleanup_error == "ALLOCATION_UNRESOLVED: retained pending record"`` with
``stop_verified is False``. Receipts are never rewritten and ``stop_verified=True`` is never
manufactured. Allocated or ambiguous attempts continue to require verified stop.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from research.watch6_search_efficacy_v3.admission import has_non_transport_fault
from research.watch6_search_efficacy_v3.dispatch import (
    ALLOCATION_UNAVAILABLE,
    DEFAULT_OPS_DIR,
    _log_has_exit,
)
from tools.ops.package_spec import digest
from tools.watch6_v3_accounting import RETAINED_PENDING_CLEANUP_ERROR

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
    "exports",
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


def strip_colab_update_notice(detail: str) -> str:
    """Strip optional colab-cli 0.7.5+ update banner prepended to `colab new failed (1): `."""
    prefix = "colab new failed (1): "
    if not detail.startswith(prefix):
        return detail
    lines = detail[len(prefix) :].splitlines(keepends=True)
    while lines and (
        lines[0].startswith("[colab] A new version of Colab CLI is available")
        or lines[0].startswith("[colab] Run 'colab update'")
        or lines[0].startswith('[colab] To silence this check, set "enable_update_check"')
        or lines[0] == "\n"
    ):
        lines.pop(0)
    return prefix + "".join(lines)


def _matches_failed_new_server_refusal(
    error_raw: object,
    *,
    session: str,
    accelerator: str,
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

    normalized = strip_colab_update_notice(detail)
    expected_precondition = (
        f"colab new failed (1): [colab] Creating session '{session}'...\n"
        f"{ALLOCATION_UNAVAILABLE}\n"
    )
    if normalized == expected_precondition:
        return True

    if normalized.startswith("colab new failed (1): "):
        lines = normalized.removeprefix("colab new failed (1): ").strip().splitlines()
        if lines and re.fullmatch(
            r"ColabRequestError: Failed to issue request POST "
            r"https://colab\.research\.google\.com/tun/m/assign"
            r"\?nbh=[a-f0-9_.-]+&variant=GPU&accelerator="
            + re.escape(accelerator)
            + r": (?:Service Unavailable|503(?: Service Unavailable)?)",
            lines[-1],
        ):
            return True
    return False


def _parse_sessions_stdout(sessions_stdout: object, forbidden_session: str) -> set[str]:
    if not isinstance(sessions_stdout, str):
        raise ValueError("proof sessions_stdout must be a string")
    if forbidden_session in sessions_stdout:
        raise ValueError("refused session appears in proof sessions_stdout")

    endpoints: list[str] = []
    for raw_line in sessions_stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        match = SESSION_LINE_RE.match(raw_line)
        if match is None:
            raise ValueError(f"unrecognized line in proof sessions_stdout: {raw_line!r}")
        sess_label, endpoint = match.group(1), match.group(2)
        if sess_label == forbidden_session:
            raise ValueError("refused session appears in parsed sessions_stdout")
        endpoints.append(endpoint)
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("duplicate endpoints in proof sessions_stdout")
    return set(endpoints)


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
        # Also scan top-level and dispatch proof files referencing this exact receipt path.
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
    if (
        not isinstance(job_id, str)
        or JOB_ID_RE.fullmatch(job_id) is None
        or disk_receipt.get("job_id") != job_id
        or not isinstance(commit, str)
        or HEX40_RE.fullmatch(commit) is None
        or disk_receipt.get("commit") != commit
        or disk_receipt.get("manifest_sha256") != exec_manifest_sha
        or Path(str(exec_manifest.get("output_dir", ""))).resolve() != resolved_out
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

    # 4. Zero transfers, control operations, terminal, hardware, polls, measurements, or outputs.
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

    for forbidden in FORBIDDEN_OUTPUT_ENTRIES:
        target = output_dir / forbidden
        if target.exists() or target.is_symlink():
            raise ValueError(f"forbidden post-allocation artifact present in output_dir: {target}")

    for pattern in ("poll-*.json", "telemetry-*.json", "usage-[0-9]*.json", "spend-*.json"):
        if any(output_dir.glob(pattern)):
            raise ValueError(f"forbidden runtime telemetry/poll artifact matching {pattern}")

    # 5. Ops manifest & log consistency + identical LIVE colab-allocation-unavailable/v1 proof(s).
    resolved_ops = (ops_dir or DEFAULT_OPS_DIR).resolve()
    if not resolved_ops.is_dir():
        raise ValueError(f"missing ops_dir for LIVE unallocated reconciliation: {resolved_ops}")

    ops_matches = _discover_ops_names_and_manifests(resolved_ops, resolved_out, ops_name)
    ops_names: list[str] = []
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

    if ops_name is not None and ops_name not in ops_names:
        log_path = resolved_ops / "logs" / f"{ops_name}.log"
        if log_path.exists() and not _log_has_exit(log_path, nonzero_only=True):
            raise ValueError(f"ops log lacks nonzero EXIT marker: {log_path}")
        ops_names.append(ops_name)

    proof_paths = _discover_proof_paths(resolved_ops, receipt_path, ops_names)
    if not proof_paths:
        raise ValueError(f"missing LIVE {PROOF_SCHEMA} proof for {receipt_path}")

    proof_shas = {digest(pp) for pp in proof_paths}
    if len(proof_shas) != 1:
        raise ValueError(
            f"conflicting/non-identical unallocated proof copies for {receipt_path}: {proof_paths}"
        )
    proof_sha = next(iter(proof_shas))
    proof = _read_regular_json_dict(proof_paths[0], "unallocated proof")

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

    proof_rec = proof.get("receipt")
    if (
        not isinstance(proof_rec, dict)
        or not isinstance(proof_rec.get("path"), str)
        or not proof_rec["path"]
        or Path(proof_rec["path"]).resolve() != receipt_path.resolve()
        or proof_rec.get("sha256") != receipt_sha
    ):
        raise ValueError("unallocated proof receipt path/sha256 binding mismatch")

    # 6. Archived & retired pending record + pending/active/lane assignment absence on disk.
    pending_obj = proof.get("pending")
    if (
        not isinstance(pending_obj, dict)
        or not isinstance(pending_obj.get("path"), str)
        or not pending_obj["path"]
        or not isinstance(pending_obj.get("sha256"), str)
        or HEX64_RE.fullmatch(pending_obj["sha256"]) is None
        or not isinstance(pending_obj.get("record"), dict)
    ):
        raise ValueError("unallocated proof missing archived pending record metadata")

    pending_path = Path(pending_obj["path"])
    if pending_path.name != f"pending-{session}.json":
        raise ValueError("archived pending record filename does not match receipt session")
    if pending_path.exists() or pending_path.is_symlink():
        raise ValueError(f"pending allocation record was not retired from disk: {pending_path}")
    active_path = pending_path.parent / f"active-{session}.json"
    if active_path.exists() or active_path.is_symlink():
        raise ValueError(
            f"active session record exists for allegedly unallocated job: {active_path}"
        )

    record = pending_obj["record"]
    if (
        "endpoint" in record
        or record.get("session") != session
        or not isinstance(record.get("output"), str)
        or Path(record["output"]).resolve() != resolved_out
        or not isinstance(record.get("notebook_hash"), str)
        or HEX32_RE.fullmatch(record["notebook_hash"]) is None
        or not isinstance(record.get("config"), str)
        or not isinstance(record.get("assignment_record"), str)
    ):
        raise ValueError("archived pending record fields do not match session/output/notebook")

    cfg_path = Path(record["config"])
    assign_path = Path(record["assignment_record"])
    if cfg_path.name != "sessions.json" or assign_path != cfg_path.with_suffix(".assignment.json"):
        raise ValueError("archived pending record config/assignment_record path mismatch")
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
        if cfg_data.get(session):
            raise ValueError("lane sessions.json contains an assignment for refused session")

    before = record.get("before")
    if (
        not isinstance(before, list)
        or any(not isinstance(ep, str) or not ep for ep in before)
        or len(before) != len(set(before))
    ):
        raise ValueError("archived pending record has invalid 'before' endpoint list")
    before_set = set(before)

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
