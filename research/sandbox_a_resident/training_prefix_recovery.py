"""Receipt-less c2 admission of a hash-bound pre-loss compact-export prefix.

No account queries, stop operations, payload decoding, or output writes occur.
A declared stop cause is an operator assertion, never fabricated release proof.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.units import digest

SCHEMA = "sandbox-a-train01-retained-export-prefix/v1"
STOP_KINDS = frozenset({"host_loss", "lease_expired", "runtime_timeout"})


def verify_prefix_recovery(
    certificate_path: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    *,
    output_root: Path,
    source_control: Mapping[str, Any],
    source_packet_id: str | None = None,
    source_packet_manifest_sha256: str | None = None,
    **unused: Any,
) -> dict[str, Any]:
    """Admit only the frozen retained prefix, with every included byte verified.

    The latest locally complete snapshot must be compact and must be inventoried
    in the final retained poll. All earlier polls are mandatory. The log ends at
    that poll's native supervisor event; its suffix is neither read nor trusted.
    This is continuation-input admission, not proof of account release.
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
        "original_receipt_present": None,
        "original_receipt_used": False,
        "receipt_less": True,
        "stop_kind": cert["stop_kind"],
        "stop_cause_basis": "operator_declaration",
        "post_hoc_release_verified": False,
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
                and event.get("terminal") is False
                and event.get("status") == poll.get("state", {}).get("status")
                and event.get("exports") == len(poll["exports"])
                and event.get("job_log_tail") == poll.get("job_log_tail", "")[-4096:]
            ):
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
