"""Receipt-less c2 admission of a hash-bound pre-loss compact-export prefix.

No account queries, stop operations, payload decoding, or output writes occur.
Every suffix is scanned for poison; admission requires an offline account release capture.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.units import digest

SCHEMA = "sandbox-a-train01-retained-export-prefix/v2"
STOP_KINDS = frozenset({"host_loss", "lease_expired", "runtime_timeout"})


SELF_RELEASE_ERROR = "AccountingError('supervised assignment disappeared from sessions')"
SELF_RELEASE_REASONS = frozenset({"timeout", "lease_expired"})


def _export_inventory(root: Path) -> list[dict[str, Any]]:
    """Bind the entire export listing, including unfinished and opaque files."""
    return [
        {"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": digest(p)}
        for p in r._retained_files(root)
    ]


def _terminal(record: Any, job: str, generation: str) -> str:
    if (
        not isinstance(record, Mapping)
        or record.get("job_id") != job
        or record.get("generation") != generation
        or record.get("status") != "failed"
        or record.get("reason") not in SELF_RELEASE_REASONS
        or type(record.get("returncode")) is not int
        or record["returncode"] not in {0, -9, -10, -15, -30}
    ):
        raise r.ResumeRefused("retained-prefix terminal is not a known mechanical self-release")
    r._scan_json_faults(
        record, label="self_release_terminal", expected_job_id=job, expected_generation=generation
    )
    return record["reason"]


def _receipt(record: Any, manifest: Mapping[str, Any], manifest_sha: str, generation: str) -> None:
    """Validate a narrow producer envelope, then scan every other field recursively."""
    if (
        not isinstance(record, Mapping)
        or record.get("job_id") != manifest.get("job_id")
        or record.get("commit") != manifest.get("commit")
        or record.get("manifest_sha256") != manifest_sha
        or record.get("generation") != generation
        or record.get("status") != "INCOMPLETE"
        or record.get("failure_class") != "ACCOUNTING"
        or record.get("error") != SELF_RELEASE_ERROR
        or record.get("stop_verified") is not True
        or "cleanup_error" not in record
        or record["cleanup_error"] is not None
    ):
        raise r.ResumeRefused("retained-prefix receipt is not a known mechanical self-release")
    _terminal(record.get("terminal"), str(manifest["job_id"]), generation)
    if "returncode" in record and (
        type(record["returncode"]) is not int
        or record["returncode"] != record["terminal"]["returncode"]
    ):
        raise r.ResumeRefused("retained-prefix receipt returncode differs from terminal")
    # Only the exact already-validated producer envelope fields are omitted.
    # No exception is propagated into nested accounting, diagnostics or terminal.
    r._scan_json_faults(
        {
            k: v
            for k, v in record.items()
            if k not in {"status", "failure_class", "error", "returncode"}
        },
        label="self_release_receipt",
        expected_job_id=str(manifest["job_id"]),
        expected_generation=generation,
        supervisor_transport=True,
    )


def _event_matches(event: Any, poll: Mapping[str, Any], job: str) -> bool:
    return (
        isinstance(event, Mapping)
        and event.get("job") == job
        and event.get("status") == poll["state"].get("status")
        and event.get("terminal") is False
        and type(event.get("exports")) is int
        and event["exports"] == len(poll["exports"])
        and event.get("job_log_tail") == poll.get("job_log_tail", "")[-4096:]
    )


def _prefix_events(data: bytes, polls: list[Mapping[str, Any]], job: str) -> float:
    events = []
    for line in data.decode("utf-8").splitlines():
        try:
            event = r._json_loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping) and event.get("job") == job and "terminal" in event:
            events.append(event)
    if len(events) != len(polls) or any(
        not _event_matches(event, poll, job) for event, poll in zip(events, polls)
    ):
        raise r.ResumeRefused("retained-prefix log events differ from ordered poll prefix")
    elapsed = -1.0
    for event in events:
        current = event.get("elapsed")
        if type(current) not in (int, float) or not math.isfinite(current) or current < elapsed:
            raise r.ResumeRefused("retained-prefix log elapsed history is malformed")
        elapsed = current
    return elapsed


def _scan_incomplete_text(text: str, *, label: str, job: str, generation: str) -> bool:
    """Only EOF truncation is inert; complete retained negative facts still poison."""
    try:
        r._json_loads(text)
    except json.JSONDecodeError as error:
        stripped = text.rstrip()
        if stripped and (
            not stripped.lstrip().startswith(("{", "["))
            or not (error.pos >= len(stripped) or error.msg.startswith("Unterminated string"))
        ):
            return False
        r._refuse_if_fault_text(text, field_name=label)
        for line in text.splitlines():
            r._scan_diagnostic_record_line(
                line,
                label=label,
                expected_job_id=job,
                expected_generation=generation,
                supervisor_transport=True,
                allow_first_tail_fragment=True,
            )
        decoder = json.JSONDecoder()
        for match in re.finditer(r'"([^"\\]+)"\s*:\s*', text):
            key = r._record_key(match[1])
            try:
                fact, _ = decoder.raw_decode(text, match.end())
            except json.JSONDecodeError:
                if key in {
                    "status",
                    "reason",
                    "returncode",
                    "failure_class",
                    "error",
                    "finite",
                    "material_gaps",
                    "stage",
                    "phase",
                    "prepare_phase",
                }:
                    raise r.ResumeRefused("incomplete suffix truncates an execution/fault fact")
                continue
            if key in {"reason", "failure_class", "error"} and fact:
                raise r.ResumeRefused("incomplete suffix records unverified terminal or failure")
            r._scan_json_faults(
                {key: fact},
                label=label,
                expected_job_id=job,
                expected_generation=generation,
                supervisor_transport=True,
            )
        return True
    return False


def _scan_incomplete(path: Path, *, job: str, generation: str) -> bool:
    if path.stat().st_size > r.MAX_RETAINED_JSON_BYTES:
        raise r.ResumeRefused(f"oversized retained diagnostic: {path}")
    return _scan_incomplete_text(
        path.read_text(encoding="utf-8"), label=str(path), job=job, generation=generation
    )


def _scan_suffix_and_output(
    root: Path,
    log_path: Path,
    cutoff: int,
    retained: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha: str,
    generation: str,
    transport: r._RetainedTransport,
    latest: int,
) -> tuple[Mapping[str, Any] | None, set[str], float]:
    """All suffix/output bytes can poison admission; none bind a resume payload."""
    job = str(manifest["job_id"])
    receipt = None
    reasons: set[str] = set()
    terminals: list[Mapping[str, Any]] = []
    elapsed = 0.0
    receipt_path = root / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = r._retained_json_records(receipt_path)[0]
        _receipt(receipt, manifest, manifest_sha, generation)
        reasons.add(receipt["terminal"]["reason"])
        terminals.append(receipt["terminal"])
    for path in r._retained_files(root):
        rel = str(path.relative_to(root))
        if (
            path.name.startswith("receipt.json")
            and path != receipt_path
            and not r._publication_sidecar(path)
        ):
            raise r.ResumeRefused("unsupported retained-prefix receipt variant")
        if path == receipt_path or path.resolve() == log_path.resolve():
            continue  # Supplied supervisor log is scanned below, including its whole suffix.
        if path.name.startswith("summary.json") and path.parent == root / "exports/training":
            raise r.ResumeRefused("exported summary.json already exists; prior run completed")
        if path.name == "faulthandler-stacks.log" and path.stat().st_size:
            raise r.ResumeRefused(
                "exported faulthandler-stacks.log is non-empty (stall watchdog fired)"
            )
        suffix = rel not in retained and not Path(str(path) + ".complete").exists()
        is_json = any(s in {".json", ".jsonl"} for s in path.suffixes)
        if (
            suffix
            and not path.name.startswith("terminal.json")
            and is_json
            and not r._publication_sidecar(path)
            and _scan_incomplete(path, job=job, generation=generation)
        ):
            continue
        if is_json and not r._publication_sidecar(path):
            for record in r._retained_json_records(path):
                r._check_scientific_publication(path, root, record)
                if path.name.startswith("terminal.json"):
                    reasons.add(_terminal(record, job, generation))
                    terminals.append(record)
                if isinstance(record, Mapping) and record.get("terminal") is not None:
                    reasons.add(_terminal(record["terminal"], job, generation))
                    terminals.append(record["terminal"])
                if isinstance(record, Mapping):
                    for entry in record.get("exports", []):
                        if isinstance(entry, Mapping):
                            match = re.fullmatch(
                                r"training/iteration-(\d{4})(?:\.pt(?:\.xz)?|\.json|-actor\.pth)",
                                str(entry.get("path")),
                            )
                            if match and int(match[1]) > latest:
                                raise r.ResumeRefused(
                                    "retained inventory reveals a newer snapshot "
                                    "than recovery cutoff"
                                )
        r._scan_retained_text(
            path,
            output_root=root,
            expected_job_id=job,
            expected_generation=generation,
            transport=transport,
        )
    data = log_path.read_bytes()
    if len(data) > r.MAX_RETAINED_JSON_BYTES:
        raise r.ResumeRefused("oversized retained-prefix supervisor log")
    lines = data[cutoff:].decode("utf-8").splitlines()
    footer = [f"INCOMPLETE: {SELF_RELEASE_ERROR}; receipt: {root / 'receipt.json'}", "EXIT=1"]
    if receipt is not None and [line.strip() for line in lines[-2:]] == footer:
        lines = lines[:-2]
    for line_index, line in enumerate(lines):
        try:
            record = r._json_loads(line)
        except json.JSONDecodeError:
            record = None
        if receipt is not None and record == receipt:
            _receipt(record, manifest, manifest_sha, generation)
            continue
        if isinstance(record, Mapping):
            if isinstance(record.get("terminal"), Mapping):
                reasons.add(_terminal(record["terminal"], job, generation))
            if "reason" in record:
                reasons.add(_terminal(record, job, generation))
            current = record.get("elapsed")
            if (
                type(current) in (int, float)
                and math.isfinite(current)
                and (record.get("job") == job or record.get("job_id") == job)
            ):
                elapsed = max(elapsed, current)
        if (
            isinstance(record, Mapping)
            and record.get("terminal") is True
            and set(record) <= {"job", "elapsed", "status", "terminal", "exports", "job_log_tail"}
            and record.get("job") == job
            and type(record.get("exports")) is int
        ):
            matches = [
                t
                for t in terminals
                if record.get("status") == t.get("status")
                and isinstance(t.get("exports"), list)
                and record["exports"] == len(t["exports"])
            ]
            if matches and len({t["reason"] for t in matches}) == 1:
                # The qualified native terminal event omits generation/reason.
                # Bind those fields only from an already validated retained terminal.
                r._scan_json_faults(
                    {
                        **{k: v for k, v in record.items() if k != "job"},
                        "job_id": job,
                        "generation": generation,
                        "reason": matches[0]["reason"],
                    },
                    label="supervisor_log_suffix",
                    expected_job_id=job,
                    expected_generation=generation,
                    supervisor_transport=True,
                )
                continue
        if line_index == len(lines) - 1 and _scan_incomplete_text(
            line, label="supervisor_log_suffix", job=job, generation=generation
        ):
            continue
        r._scan_diagnostic_record_line(
            line,
            label="supervisor_log_suffix",
            expected_job_id=job,
            expected_generation=generation,
            supervisor_transport=True,
            validated_receipt=receipt,
        )
    return receipt, reasons, elapsed


def verify_prefix_recovery(
    certificate_path: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    *,
    output_root: Path,
    source_control: Mapping[str, Any],
    source_packet_id: str | None = None,
    source_packet_manifest_sha256: str | None = None,
    now_unix_seconds: float | None = None,
    max_capture_age_seconds: float | None = 86400.0,
) -> dict[str, Any]:
    """Admit only the frozen retained prefix, with every included byte verified.

    The latest locally complete snapshot must be compact and must be inventoried
    in the final retained poll. All earlier polls are mandatory. The log ends at
    that poll's native supervisor event. All suffix/output evidence can refuse;
    only the prefix binds snapshots. A post-hoc release capture is mandatory.
    """
    if (
        any(path.is_symlink() for path in (output_root, *output_root.parents))
        or not output_root.is_dir()
    ):
        raise r.ResumeRefused("retained-prefix output root is missing or symlinked")
    if certificate_path.resolve().is_relative_to(output_root.resolve()):
        raise r.ResumeRefused("prefix certificate must be outside prior output")
    desc = r._descriptor(certificate_path, runtime_path=certificate_path.name)
    cert = r._json_loads(certificate_path.read_text())
    if not isinstance(cert, Mapping) or cert.get("schema") != SCHEMA:
        raise r.ResumeRefused("unsupported retained-prefix certificate")
    identity = cert.get("identities", {})
    if not isinstance(identity, Mapping):
        raise r.ResumeRefused("retained-prefix certificate identity differs from frozen source")
    if (
        not r._is_hex(manifest_sha256, 64)
        or identity.get("job_id") != manifest.get("job_id")
        or identity.get("commit") != manifest.get("commit")
        or manifest.get("commit") != source_control.get("source_commit")
        or identity.get("manifest_sha256") != manifest_sha256
        or identity.get("output_dir") != str(output_root.resolve())
        or not r._is_hex(identity.get("generation"), 32)
        or not isinstance(identity.get("source_packet_id"), str)
        or not identity.get("source_packet_id")
        or not r._is_hex(identity.get("source_packet_manifest_sha256"), 64)
        or source_packet_id is not None
        and identity.get("source_packet_id") != source_packet_id
        or source_packet_manifest_sha256 is not None
        and identity.get("source_packet_manifest_sha256") != source_packet_manifest_sha256
    ):
        raise r.ResumeRefused("retained-prefix certificate identity differs from frozen source")
    if (
        cert.get("stop_kind") not in STOP_KINDS
        or cert.get("operator_confirmed_stopped") is not True
    ):
        raise r.ResumeRefused(
            "retained-prefix recovery requires an operator-confirmed eligible stop"
        )
    r._scan_json_faults(cert, label="prefix_certificate")
    export_root = output_root / "exports/training"
    ledger, uploads, latest = r.build_ledger(export_root, source_control)
    if latest >= source_control.get("outer_iterations", 0):
        raise r.ResumeRefused("latest verified snapshot already reaches the frozen endpoint")
    expected = {"manifest.json"}
    snapshot_bindings = []
    # Every eligible snapshot stays verified; no older/better-checkpoint selection.
    for unit in ledger["units"]:
        stem = f"iteration-{unit['iteration']:04d}"
        info = r._retained_json_records(export_root / f"{stem}.json")[0]
        compact = info.get("compact_snapshot")
        original = unit["exports"][0]
        if (
            not isinstance(compact, Mapping)
            or compact.get("schema") != r.COMPACT_SNAPSHOT_EXPORT_SCHEMA
        ):
            raise r.ResumeRefused("retained-prefix snapshot lacks compact-at-source metadata")
        for role, suffix in (("derivative", ".pt.xz"), ("manifest", ".pt.xz.manifest")):
            payload = export_root / (stem + suffix)
            descriptor = r._descriptor(payload, runtime_path=payload.name)
            expected_compact = {
                "name": payload.name,
                "size": descriptor["size"],
                "sha256": descriptor["sha256"],
            }
            if compact.get(role) != expected_compact:
                raise r.ResumeRefused(
                    "retained-prefix compact descriptor differs from snapshot header"
                )
        if compact.get("original") != {
            "name": stem + ".pt",
            "size": original["size"],
            "sha256": original["sha256"],
        }:
            raise r.ResumeRefused(
                "retained-prefix compact original differs from full-state binding"
            )
        for suffix in (".pt.xz", ".pt.xz.manifest", ".json", "-actor.pth"):
            path = export_root / (stem + suffix)
            row = r._descriptor(path, runtime_path=f"training/{path.name}")
            snapshot_bindings.append(row)
            expected.update(
                str(Path(str(path) + end).relative_to(output_root))
                for end in ("", ".sha256", ".complete")
            )
    for row in uploads:
        path = Path(row["path"])
        expected.add(str(path.relative_to(output_root)))
    last = cert.get("last_verified_poll")
    if type(last) is not int or last < 0 or last > 1000000:
        raise r.ResumeRefused("malformed retained-prefix poll cutoff")
    expected.update(f"poll-{i:06d}.json" for i in range(last + 1))
    retained = cert.get("retained_files")
    if not isinstance(retained, Mapping) or set(retained) != expected:
        raise r.ResumeRefused(
            "retained-prefix file set differs from verified snapshots and poll prefix"
        )
    for name in sorted(expected):
        path = output_root / name
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise r.ResumeRefused("retained-prefix file has a symlinked ancestor")
        record = retained[name]
        if not isinstance(record, Mapping):
            raise r.ResumeRefused("malformed retained-prefix file descriptor")
        r._verified_regular_file(output_root / name, record, label=f"retained_prefix:{name}")
    if digest(output_root / "manifest.json") != manifest_sha256:
        raise r.ResumeRefused("retained-prefix executed manifest digest differs")
    if cert.get("export_directory_inventory") != _export_inventory(output_root / "exports"):
        raise r.ResumeRefused("retained-prefix full export directory inventory differs")
    job, generation = identity["job_id"], identity["generation"]
    transport = r._RetainedTransport(output_root, job, generation)
    # Avoid transport.bindings(), which deliberately examines the entire ordinary
    # output tree. Only the hash-verified retained poll prefix may bind this path.
    transport.bounds = {}
    polls = []
    required = {(row["path"], row["size"], row["sha256"]) for row in snapshot_bindings}
    for i in range(last + 1):
        path = output_root / f"poll-{i:06d}.json"
        poll = r._retained_json_records(path)[0]
        if not isinstance(poll, Mapping):
            raise r.ResumeRefused("malformed retained-prefix poll")
        state = poll.get("state")
        if (
            not isinstance(state, Mapping)
            or state.get("job_id") != job
            or state.get("generation") != generation
        ):
            raise r.ResumeRefused("retained-prefix poll identity differs")
        r._scan_json_faults(
            poll,
            label=f"retained_poll:{i}",
            expected_job_id=job,
            expected_generation=generation,
            supervisor_transport=True,
        )
        exports = poll.get("exports")
        if not isinstance(exports, list) or any(not isinstance(e, Mapping) for e in exports):
            raise r.ResumeRefused("malformed retained-prefix export inventory")
        bound = {(e.get("path"), e.get("size"), e.get("sha256")) for e in exports}
        if i < last and required <= bound:
            raise r.ResumeRefused("retained-prefix cutoff extends past last verified export")
        if i == last and not required <= bound:
            raise r.ResumeRefused(
                "retained-prefix cutoff does not bind all compact snapshot exports"
            )
        for e in exports:
            if (
                isinstance(e.get("path"), str)
                and type(e.get("size")) is int
                and r._is_hex(e.get("sha256"), 64)
            ):
                transport.bounds.setdefault(e["path"], set()).add((e["size"], e["sha256"]))
        polls.append(poll)
    for name in sorted(expected):
        if not name.startswith("poll-"):
            r._scan_retained_text(
                output_root / name,
                output_root=output_root,
                expected_job_id=job,
                expected_generation=generation,
                transport=transport,
            )
    log = cert.get("supervisor_log_prefix")
    if not isinstance(log, Mapping) or not isinstance(log.get("path"), str):
        raise r.ResumeRefused("missing retained-prefix supervisor log")
    path, length = Path(log["path"]), log.get("size")
    if (
        any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
        or type(length) is not int
        or not 0 < length <= r.MAX_RETAINED_JSON_BYTES
    ):
        raise r.ResumeRefused("malformed retained-prefix supervisor log")
    with path.open("rb") as stream:
        prefix = stream.read(length)
    if (
        len(prefix) != length
        or not prefix.endswith(b"\n")
        or hashlib.sha256(prefix).hexdigest() != log.get("sha256")
    ):
        raise r.ResumeRefused("retained-prefix supervisor log hash or boundary differs")
    try:
        event = r._json_loads(prefix.decode("utf-8").splitlines()[-1])
    except (ValueError, UnicodeError) as error:
        raise r.ResumeRefused(
            "retained-prefix log cutoff is not a complete supervisor event"
        ) from error
    poll = polls[-1]
    if (
        not isinstance(event, Mapping)
        or event.get("job") != job
        or event.get("status") != poll["state"].get("status")
        or event.get("terminal") is not False
        or event.get("exports") != len(poll["exports"])
        or event.get("job_log_tail") != poll.get("job_log_tail", "")[-4096:]
    ):
        raise r.ResumeRefused("retained-prefix log cutoff differs from last verified poll")
    elapsed = _prefix_events(prefix, polls, job)
    receipt, reasons, suffix_elapsed = _scan_suffix_and_output(
        output_root,
        path,
        length,
        retained,
        manifest,
        manifest_sha256,
        generation,
        transport,
        latest,
    )
    elapsed = max(elapsed, suffix_elapsed)
    kind = cert["stop_kind"]
    timeout = manifest.get("timeout_minutes")
    if kind == "lease_expired" and reasons != {"lease_expired"}:
        raise r.ResumeRefused("declared lease_expired lacks matching mechanical terminal evidence")
    if kind == "runtime_timeout" and (
        reasons != {"timeout"}
        or type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or timeout <= 0
        or not timeout * 60 - 60 <= elapsed <= timeout * 60 + 600
    ):
        raise r.ResumeRefused(
            "declared runtime_timeout differs from terminal or elapsed timeout evidence"
        )
    if kind == "host_loss" and (reasons or receipt is not None):
        raise r.ResumeRefused(
            "declared host_loss conflicts with retained mechanical terminal evidence"
        )
    capture = r._verify_account_release_capture(
        cert.get("account_status_capture"),
        path,
        now_unix_seconds=now_unix_seconds,
        max_capture_age_seconds=max_capture_age_seconds,
    )
    with tempfile.TemporaryDirectory(prefix="train01-prefix-scan-") as scratch:
        copy = Path(scratch) / "supervisor.log"
        copy.write_bytes(prefix)
        audit = r._check_supervisor_log_fault_evidence(
            copy, expected_job_id=job, expected_generation=generation
        )
    if audit["matched_job_records"] < 1:
        raise r.ResumeRefused("retained-prefix supervisor log lacks same-job evidence")
    return {
        "schema": SCHEMA,
        "certificate_sha256": desc["sha256"],
        "certificate_size": desc["size"],
        "failure_class": "RECEIPTLESS_RETAINED_EXPORT_PREFIX",
        "original_receipt_present": receipt is not None,
        "original_receipt_sha256": (
            digest(output_root / "receipt.json") if receipt is not None else None
        ),
        "original_receipt_used": False,
        "receipt_less": True,
        "stop_kind": cert["stop_kind"],
        "stop_cause_basis": "operator_declaration_checked_against_retained_evidence",
        "post_hoc_release_verified": True,
        "account_status_capture_sha256": cert["account_status_capture"]["sha256"],
        "captured_unix_seconds": capture["captured_unix_seconds"],
        "suffix_scanned_for_poison": True,
        "export_directory_inventory": cert["export_directory_inventory"],
        "after_last_verified_export_trusted": False,
        "last_verified_poll": last,
        "supervisor_log_sha256": log["sha256"],
        "supervisor_log_prefix_bytes": length,
        "latest_eligible_iteration": latest,
        "retained_files": dict(retained),
    }


def build_prefix_recovery_certificate(
    output_root: Path,
    source_control: Mapping[str, Any],
    supervisor_log: Path,
    certificate_output: Path,
    *,
    account_status_capture: Path,
    stop_kind: str,
    operator_confirmed_stopped: bool,
    source_packet_id: str,
    source_packet_manifest_sha256: str,
) -> dict[str, Any]:
    """Create-only local evidence binding after an operator declares a stop.

    Copies no bytes to the old output and never queries the provider. The
    temporary certificate must pass the same admission verifier before publish.
    """
    from research.sandbox_a_resident.units import publish

    if certificate_output.exists() or certificate_output.is_symlink():
        raise FileExistsError(certificate_output)
    if certificate_output.resolve().is_relative_to(output_root.resolve()):
        raise r.ResumeRefused("prefix certificate must be outside prior output")
    manifest_path = output_root / "manifest.json"
    manifest = r._retained_json_records(manifest_path)[0]
    ledger, uploads, _ = r.build_ledger(output_root / "exports/training", source_control)
    required = set()
    files = {"manifest.json", *(str(Path(row["path"]).relative_to(output_root)) for row in uploads)}
    for unit in ledger["units"]:
        for suffix in (".pt.xz", ".pt.xz.manifest", ".json", "-actor.pth"):
            path = output_root / "exports/training" / f"iteration-{unit['iteration']:04d}{suffix}"
            row = r._descriptor(path, runtime_path=f"training/{path.name}")
            required.add((row["path"], row["size"], row["sha256"]))
            files.update(
                str(Path(str(path) + end).relative_to(output_root))
                for end in ("", ".sha256", ".complete")
            )
    for index, path in enumerate(sorted(output_root.glob("poll-*.json"))):
        if path.name != f"poll-{index:06d}.json":
            raise r.ResumeRefused("retained-prefix poll history has a gap")
        files.add(path.name)
        poll = r._retained_json_records(path)[0]
        if not isinstance(poll, Mapping) or not isinstance(poll.get("exports"), list):
            raise r.ResumeRefused("malformed retained-prefix poll")
        exports = {
            (e.get("path"), e.get("size"), e.get("sha256"))
            for e in poll["exports"]
            if isinstance(e, Mapping)
        }
        if required <= exports:
            break
    else:
        raise r.ResumeRefused("no retained poll binds all compact snapshot exports")
    prefix = bytearray()
    poll_prefix = [
        r._retained_json_records(output_root / f"poll-{i:06d}.json")[0] for i in range(index + 1)
    ]
    matched = 0
    with supervisor_log.open("rb") as stream:
        for line in stream:
            prefix.extend(line)
            if len(prefix) > r.MAX_RETAINED_JSON_BYTES:
                raise r.ResumeRefused("oversized retained-prefix supervisor log")
            try:
                event = r._json_loads(line.decode("utf-8"))
            except (ValueError, UnicodeError):
                continue
            if (
                isinstance(event, Mapping)
                and event.get("job") == manifest.get("job_id")
                and "terminal" in event
            ):
                if not _event_matches(event, poll_prefix[matched], manifest["job_id"]):
                    raise r.ResumeRefused(
                        "retained-prefix log events differ from ordered poll prefix"
                    )
                matched += 1
                if matched == len(poll_prefix):
                    break
        else:
            raise r.ResumeRefused("no supervisor event matches last verified export poll")
    cert = {
        "schema": SCHEMA,
        "identities": {
            "job_id": manifest["job_id"],
            "commit": manifest["commit"],
            "manifest_sha256": digest(manifest_path),
            "output_dir": str(output_root.resolve()),
            "generation": poll["state"]["generation"],
            "source_packet_id": source_packet_id,
            "source_packet_manifest_sha256": source_packet_manifest_sha256,
        },
        "account_status_capture": r._descriptor(
            account_status_capture, runtime_path=account_status_capture.name
        ),
        "export_directory_inventory": _export_inventory(output_root / "exports"),
        "stop_kind": stop_kind,
        "operator_confirmed_stopped": operator_confirmed_stopped,
        "last_verified_poll": index,
        "retained_files": {
            name: {
                "path": str((output_root / name).resolve()),
                "size": (output_root / name).stat().st_size,
                "sha256": digest(output_root / name),
            }
            for name in sorted(files)
        },
        "supervisor_log_prefix": {
            "path": str(supervisor_log.resolve()),
            "size": len(prefix),
            "sha256": hashlib.sha256(prefix).hexdigest(),
        },
    }
    cert["account_status_capture"]["captured_unix_seconds"] = r._json_loads(
        account_status_capture.read_text()
    ).get("captured_unix_seconds")
    cert["account_status_capture"]["path"] = str(account_status_capture.resolve())
    content = json.dumps(cert, indent=2, sort_keys=True) + "\n"
    with tempfile.TemporaryDirectory(prefix="train01-prefix-certificate-") as scratch:
        candidate = Path(scratch) / "candidate.json"
        publish(candidate, lambda dst: dst.write_text(content))
        verify_prefix_recovery(
            candidate,
            manifest,
            digest(manifest_path),
            output_root=output_root,
            source_control=source_control,
            source_packet_id=source_packet_id,
            source_packet_manifest_sha256=source_packet_manifest_sha256,
        )
    publish(certificate_output, lambda dst: dst.write_text(content))
    return verify_prefix_recovery(
        certificate_output,
        manifest,
        digest(manifest_path),
        output_root=output_root,
        source_control=source_control,
        source_packet_id=source_packet_id,
        source_packet_manifest_sha256=source_packet_manifest_sha256,
    )
