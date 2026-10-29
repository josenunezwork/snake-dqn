"""Mac-side, outcome-blind resume packet construction for TRAIN-01.

This module deliberately does not import torch.  The Mac verifies immutable
payload bytes through paths, sizes and SHA-256 sidecars; it only opens the JSON
envelope that says which completed full-state snapshot a payload belongs to.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import stat
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from research.sandbox_a_resident import training_units
from research.sandbox_a_resident.training_units import (
    COMPACT_DERIVATIVE_SUFFIX,
    COMPACT_MANIFEST_SUFFIX,
    COMPACT_RESUME_FIELD,
    COMPACT_RESUME_SCHEMA,
    COMPACT_RESUME_SCHEMA_V2,
    COMPACT_SNAPSHOT_EXPORT_SCHEMA,
    NAMESPACE,
    allocation,
    load_compact_allowlist,
    source_migration,
    unit_id,
    validate_compact_resume,
    validate_compact_source_packet,
)
from research.sandbox_a_resident.units import digest

LEDGER_SCHEMA = "sandbox-a-train-ledger/v1"
SNAPSHOT_SCHEMA = "sandbox-a-train-snapshot/v1"
HOST_ENOSPC_RECOVERY_SCHEMA = "sandbox-a-train01-host-enospc-recovery/v2"
ACCOUNT_RELEASE_CAPTURE_SCHEMA = "sandbox-a-colab-account-release-capture/v1"
HOST_ENOSPC_FAILURE_CLASS = "HOST_SUPERVISOR_RECEIPT_WRITE_ENOSPC"
ENOSPC_RECEIPT_LINE_RE = re.compile(
    r"^\[Errno 28\] No space left on device: '(.+)/\.receipt-[A-Za-z0-9._-]+'$"
)
TRACKED_OPTIONAL_OUTPUT_FILES = (
    "rate-calibration.json",
    "accounting-baseline.json",
    "exports/training/progress.json",
    "exports/training/summary.json",
    "exports/training/distribution-screen.json",
    "exports/training/calibrated-boundary.json",
    "exports/training/stall-watchdog/stall-watchdog.json",
    "exports/training/stall-watchdog/faulthandler-stacks.log",
)
RUNTIME_ROOT = "resume"
MECHANICAL_TERMINAL_REASONS = frozenset({"timeout", "lease_expired", "release_credential_expiring"})
MECHANICAL_SUPERVISOR_ERROR_MARKERS = (
    "whole-session wall cap reached",
    "whole-session deadline reached",
    "priced session unit cap reached",
    "study reserve reached",
    "retention lease elapsed",
    "read retry budget exhausted",
)
NON_MECHANICAL_SUPERVISOR_ERROR_MARKERS = (
    "keyboardinterrupt",
    "missing required export",
    "canary",
    "corrupt",
    "schema",
)
# Explicit scientific, preparation, distribution-screen, stall-watchdog, or
# unhandled Python exception evidence always overrides any mechanical timeout,
# lease, or wall-cap marker when a timeout races with a scientific fault.
# Note: ``"nonfinite "`` keeps its trailing space so healthy progress lines
# carrying ``"first_nonfinite_term": null`` never false-positive.
EXPLICIT_FAULT_EVIDENCE_MARKERS = (
    "keyboardinterrupt",
    "missing required export",
    "canary",
    "corrupt",
    "schemaerror",
    "floatingpointerror",
    "non-finite",
    "nonfinite ",
    '"finite": false',
    '"finite":false',
    "tripwire",
    "preparationfailure",
    "preparation-failed",
    "failed preparation",
    "distribution-screen-failed",
    "material distribution gap",
    "invalid initial food support",
    "zero-update b8 greedy",
    "greedy agreement failed",
    "collector selected an illegal action",
    "action collapse exceeds",
    "compactresumerefused",
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
    "zerodivisionerror",
    "arithmeticerror",
    "overflowerror",
    "memoryerror",
    "cuda out of memory",
    "cuda error",
    "inductorerror",
    "backendcompilerfailed",
    "runtimeerror:",
    "runtime supervisor failure",
    "systemexit:",
    "oserror:",
    "filenotfounderror:",
    "fileexistserror:",
)


class ResumeRefused(ValueError):
    """The immutable evidence is not eligible for an outcome-blind continuation."""


class _NonResumableSnapshot(ResumeRefused):
    """A complete snapshot is intact but predates a resumable trajectory boundary."""


@dataclass(frozen=True)
class ResumePacket:
    """Unwritten completion-attempt components for the caller's create-only packet."""

    ledger: dict[str, Any]
    input_records: list[dict[str, str]]
    manifest: dict[str, Any]
    latest_iteration: int


def _complete(path: Path) -> bool:
    """Return whether a completion-last triple is present, without reading it."""
    return all(
        item.is_file()
        for item in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    )


def _descriptor(path: Path, *, runtime_path: str) -> dict[str, Any]:
    """Verify one immutable payload and describe it without decoding its content."""
    sidecar = Path(str(path) + ".sha256")
    complete = Path(str(path) + ".complete")
    if not _complete(path):
        raise ResumeRefused(f"incomplete export: {path.name}")
    if path.is_symlink() or sidecar.is_symlink() or complete.is_symlink():
        raise ResumeRefused(f"symlinked export is not canonical: {path.name}")
    expected = sidecar.read_text().strip()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ResumeRefused(f"malformed SHA-256 sidecar: {path.name}")
    if digest(path) != expected:
        raise ResumeRefused(f"corrupt export: {path.name}")
    return {"path": runtime_path, "size": path.stat().st_size, "sha256": expected}


def _phase_is_resumable(metadata: Mapping[str, Any]) -> bool:
    """Reject non-trajectory preparation states without looking inside ``.pt``."""
    phase = metadata.get("phase")
    if phase is None or phase == "ppo":
        return True
    if not isinstance(phase, str):
        return False
    normalized = phase.lower()
    if "fail" in normalized or "calibrat" in normalized:
        return False
    # A pre-actor state can be exact-resumed only if its JSON explicitly says
    # preparation reached a safe continuation boundary.  This avoids treating
    # a calibration-only payload as a full PPO trajectory.
    return normalized in {"pre-actor", "preparation"} and metadata.get("prepare_phase") in {
        "ready",
        "skipped",
    }


def _snapshot_row(
    root: Path, info_path: Path, control: Mapping[str, Any]
) -> tuple[int, dict[str, Any], list[dict[str, str]]] | None:
    """Return one verified ledger row and the matching upload records.

    The JSON envelope is metadata; the state and actor payloads remain opaque.
    """
    if not _complete(info_path):
        return None
    # The envelope is the one payload we parse, but only after its own
    # completion-last evidence is verified.  In particular, a partial JSON
    # must never influence which state a completion attempt selects.
    _descriptor(info_path, runtime_path=f"{RUNTIME_ROOT}/{info_path.name}")
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeRefused(f"unreadable snapshot metadata: {info_path.name}") from error
    iteration = info.get("iteration")
    identity = info.get("id")
    if type(iteration) is not int or identity != unit_id(iteration):
        raise ResumeRefused(f"foreign snapshot identity: {info_path.name}")
    if info.get("schema") != SNAPSHOT_SCHEMA or not isinstance(info.get("metadata"), dict):
        raise ResumeRefused(f"malformed snapshot metadata: {info_path.name}")
    if info["metadata"].get("allocation") != allocation(dict(control)) or info["metadata"].get(
        "source_commit"
    ) != control.get("source_commit"):
        raise ResumeRefused(f"snapshot header differs from frozen control: {info_path.name}")
    stem = f"iteration-{iteration:04d}"
    if info_path.name != f"{stem}.json":
        raise ResumeRefused(f"noncanonical snapshot filename: {info_path.name}")
    raw_pt = root / f"{stem}.pt"
    actor_path = root / f"{stem}-actor.pth"
    derivative_path = root / f"{stem}.pt{COMPACT_DERIVATIVE_SUFFIX}"
    manifest_path = root / f"{stem}.pt{COMPACT_MANIFEST_SUFFIX}"
    paths = [raw_pt, info_path, actor_path]
    if not raw_pt.exists() and not raw_pt.is_symlink():
        if Path(f"{raw_pt}.sha256").exists() or Path(f"{raw_pt}.complete").exists():
            raise ResumeRefused(f"orphan sidecar without raw snapshot: {raw_pt.name}")
        compact_paths = [derivative_path, manifest_path, info_path, actor_path]
        if not all(_complete(path) for path in compact_paths):
            return None
        deriv_desc = _descriptor(
            derivative_path, runtime_path=f"{RUNTIME_ROOT}/{derivative_path.name}"
        )
        manifest_desc = _descriptor(
            manifest_path, runtime_path=f"{RUNTIME_ROOT}/{manifest_path.name}"
        )
        info_desc = _descriptor(info_path, runtime_path=f"{RUNTIME_ROOT}/{info_path.name}")
        actor_desc = _descriptor(actor_path, runtime_path=f"{RUNTIME_ROOT}/{actor_path.name}")
        if not _phase_is_resumable(info["metadata"]):
            raise _NonResumableSnapshot(
                f"snapshot phase is not a safe continuation: {info_path.name}"
            )
        snapshot = info.get("snapshot")
        orig_sha = snapshot.get("sha256") if isinstance(snapshot, dict) else None
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("id") != identity
            or type(snapshot.get("size")) is not int
            or snapshot["size"] < 1
            or not isinstance(orig_sha, str)
            or len(orig_sha) != 64
            or any(char not in "0123456789abcdef" for char in orig_sha)
        ):
            raise ResumeRefused(f"metadata does not bind full state: {info_path.name}")
        compact_info = info.get("compact_snapshot")
        if (
            not isinstance(compact_info, dict)
            or compact_info.get("schema") != COMPACT_SNAPSHOT_EXPORT_SCHEMA
            or compact_info.get("original")
            != {"name": raw_pt.name, "size": snapshot["size"], "sha256": orig_sha}
            or compact_info.get("derivative")
            != {
                "name": derivative_path.name,
                "size": deriv_desc["size"],
                "sha256": deriv_desc["sha256"],
            }
            or compact_info.get("manifest")
            != {
                "name": manifest_path.name,
                "size": manifest_desc["size"],
                "sha256": manifest_desc["sha256"],
            }
        ):
            raise ResumeRefused(f"compact snapshot metadata/descriptor mismatch: {info_path.name}")
        descriptors = [
            {
                "path": f"{RUNTIME_ROOT}/{raw_pt.name}",
                "size": snapshot["size"],
                "sha256": orig_sha,
            },
            info_desc,
            actor_desc,
        ]
        uploads: list[dict[str, str]] = [
            {
                "path": str(derivative_path.resolve()),
                "dest": deriv_desc["path"],
                "sha256": deriv_desc["sha256"],
            },
            {
                "path": str(manifest_path.resolve()),
                "dest": manifest_desc["path"],
                "sha256": manifest_desc["sha256"],
            },
        ]
        for path, descriptor in ((info_path, info_desc), (actor_path, actor_desc)):
            destination = descriptor["path"]
            uploads.extend(
                [
                    {
                        "path": str(path.resolve()),
                        "dest": destination,
                        "sha256": descriptor["sha256"],
                    },
                    {
                        "path": str(Path(str(path) + ".sha256").resolve()),
                        "dest": destination + ".sha256",
                        "sha256": digest(Path(str(path) + ".sha256")),
                    },
                    {
                        "path": str(Path(str(path) + ".complete").resolve()),
                        "dest": destination + ".complete",
                        "sha256": digest(Path(str(path) + ".complete")),
                    },
                ]
            )
        return iteration, {"id": identity, "iteration": iteration, "exports": descriptors}, uploads
    # A completed envelope may be visible while its other completion-last
    # exports are still arriving.  It is not a ledger unit until the whole trio
    # exists; do not parse or decode the missing members.
    if not all(_complete(path) for path in paths):
        return None
    runtime = [f"{RUNTIME_ROOT}/{path.name}" for path in paths]
    descriptors = [
        _descriptor(path, runtime_path=destination) for path, destination in zip(paths, runtime)
    ]
    # A preparation-only snapshot may be skipped for resumption, but it is
    # still immutable evidence.  Verify the complete payload trio before
    # reporting its semantic phase so corruption cannot be silently ignored.
    if not _phase_is_resumable(info["metadata"]):
        raise _NonResumableSnapshot(f"snapshot phase is not a safe continuation: {info_path.name}")
    snapshot = info.get("snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("id") != identity:
        raise ResumeRefused(f"metadata does not bind full state: {info_path.name}")
    if (
        snapshot.get("size") != descriptors[0]["size"]
        or snapshot.get("sha256") != descriptors[0]["sha256"]
    ):
        raise ResumeRefused(f"metadata/state descriptor mismatch: {info_path.name}")
    uploads = []
    for path, destination, descriptor in zip(paths, runtime, descriptors):
        # The runtime repeats ``load_ledger`` validation after upload, so both
        # completion sidecars are inputs too.  They are independently hash-bound
        # rather than treated as unverified transport metadata.
        uploads.extend(
            [
                {"path": str(path.resolve()), "dest": destination, "sha256": descriptor["sha256"]},
                {
                    "path": str(Path(str(path) + ".sha256").resolve()),
                    "dest": destination + ".sha256",
                    "sha256": digest(Path(str(path) + ".sha256")),
                },
                {
                    "path": str(Path(str(path) + ".complete").resolve()),
                    "dest": destination + ".complete",
                    "sha256": digest(Path(str(path) + ".complete")),
                },
            ]
        )
    return iteration, {"id": identity, "iteration": iteration, "exports": descriptors}, uploads


def build_ledger(
    export_root: Path,
    control: Mapping[str, Any],
    *,
    source_control: Mapping[str, Any] | None = None,
    allowlist_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]], int]:
    """Build a relative-path ledger from complete canonical snapshot triples.

    Incomplete snapshots are ignored (completion-last); a malformed *complete*
    snapshot fails closed.  The latest eligible trajectory is selected by the
    runner after it re-verifies the same copied input bytes.
    """
    target_control = dict(control)
    snapshot_control = target_control if source_control is None else dict(source_control)
    migration = (
        None
        if source_control is None
        else source_migration(snapshot_control, target_control, allowlist_path=allowlist_path)
    )
    rows: list[dict[str, Any]] = []
    uploads: list[dict[str, str]] = []
    nonresumable: list[tuple[int, Path]] = []
    if export_root.is_symlink():
        raise ResumeRefused("snapshot export root must not be a symlink")
    for info_path in sorted(export_root.glob("iteration-*.json")):
        try:
            candidate = _snapshot_row(export_root, info_path, snapshot_control)
        except _NonResumableSnapshot:
            # Preparation-only snapshots are valid immutable evidence but
            # cannot be resumed.  A later full PPO trajectory supersedes them;
            # malformed or foreign snapshots still fail closed below.
            # Canonicality and header identity were already verified by
            # _snapshot_row before it reports this semantic phase state.
            nonresumable.append((int(info_path.stem.removeprefix("iteration-")), info_path))
            continue
        if candidate is None:
            continue
        iteration, row, items = candidate
        rows.append(row)
        uploads.extend(items)
    if not rows:
        if nonresumable:
            raise ResumeRefused(
                f"snapshot phase is not a safe continuation: {nonresumable[-1][1].name}"
            )
        raise ResumeRefused("no complete full-trajectory snapshot triples")
    rows.sort(key=lambda item: item["iteration"])
    if len({item["iteration"] for item in rows}) != len(rows):
        raise ResumeRefused("duplicate snapshot iteration")
    endpoint, cadence = target_control.get("outer_iterations"), target_control.get(
        "snapshot_interval"
    )
    if cadence is not None:
        if type(endpoint) is not int or endpoint < 1 or type(cadence) is not int or cadence < 1:
            raise ResumeRefused("malformed frozen snapshot cadence")
        permitted = {0, endpoint, *range(cadence, endpoint, cadence)}
        if any(row["iteration"] not in permitted for row in rows):
            raise ResumeRefused("snapshot lies outside the frozen boundary cadence")
    latest = rows[-1]["iteration"]
    if any(iteration > latest for iteration, _path in nonresumable):
        later = max(nonresumable, key=lambda item: item[0])[1]
        raise ResumeRefused(f"snapshot phase is not a safe continuation: {later.name}")
    return (
        {
            "schema": LEDGER_SCHEMA,
            "namespace": NAMESPACE,
            "allocation": allocation(target_control),
            "units": rows,
            **({"source_migration": migration} if migration is not None else {}),
        },
        uploads,
        latest,
    )


def attach_compact_resume(
    ledger: Mapping[str, Any],
    uploads: list[dict[str, str]],
    export_root: Path,
    *,
    source_packet_id: str | None = None,
    allowlist: dict[Any, dict[str, Any]] | None = None,
    allowlist_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Bind the allowlisted compact derivative in place of the full-state upload.

    Torch-free: the derivative and manifest are verified by completion-last
    sidecars and SHA-256 only, then the ledger gains a ``compact_resume`` entry
    that must equal the frozen allowlist.  The original ``.pt`` payload upload is
    replaced by the derivative and manifest; its ``.sha256``/``.complete``
    sidecars stay uploaded so the runtime verifies the decoded original through
    the ordinary ledger path.
    """
    successor = copy.deepcopy(dict(ledger))
    units = successor.get("units")
    if not isinstance(units, list) or len(units) != 1:
        raise ResumeRefused("compact resume binds exactly one latest snapshot")
    unit = units[0]
    original = unit["exports"][0]
    if source_packet_id is None:
        table_v1 = training_units.COMPACT_RESUME_ALLOWLIST if allowlist is None else allowlist
        frozen = table_v1.get(unit.get("id"))
        if frozen is None:
            raise ResumeRefused("snapshot is not allowlisted for compact resume")
        if original != frozen["original"]:
            raise ResumeRefused("ledger full-state export differs from the allowlisted original")
        entry: dict[str, Any] = {
            "schema": COMPACT_RESUME_SCHEMA,
            "unit": unit["id"],
            "original": dict(original),
        }
    else:
        iteration = unit.get("iteration")
        try:
            validate_compact_source_packet(source_packet_id, iteration)
        except ValueError as error:
            raise ResumeRefused("snapshot is not allowlisted for compact resume") from error
        table_v2 = load_compact_allowlist(allowlist_path) if allowlist is None else allowlist
        frozen = table_v2.get((source_packet_id, iteration, original["sha256"]))
        if frozen is None:
            if any(
                isinstance(key, tuple)
                and len(key) == 3
                and key[:2] == (source_packet_id, iteration)
                for key in table_v2
            ):
                raise ResumeRefused(
                    "ledger full-state export differs from the allowlisted original"
                )
            raise ResumeRefused("snapshot is not allowlisted for compact resume")
        if original != frozen["original"]:
            raise ResumeRefused("ledger full-state export differs from the allowlisted original")
        entry = {
            "schema": COMPACT_RESUME_SCHEMA_V2,
            "source_packet_id": source_packet_id,
            "iteration": iteration,
            "unit": unit["id"],
            "original": dict(original),
        }
    stem = Path(original["path"]).name
    records: list[dict[str, str]] = []
    for role, suffix in (
        ("derivative", COMPACT_DERIVATIVE_SUFFIX),
        ("manifest", COMPACT_MANIFEST_SUFFIX),
    ):
        path = export_root / f"{stem}{suffix}"
        destination = f"{original['path']}{suffix}"
        entry[role] = _descriptor(path, runtime_path=destination)
        records.append(
            {"path": str(path.resolve()), "dest": destination, "sha256": entry[role]["sha256"]}
        )
    successor[COMPACT_RESUME_FIELD] = entry
    try:
        validate_compact_resume(successor, allowlist=allowlist, allowlist_path=allowlist_path)
    except ValueError as error:
        raise ResumeRefused(str(error)) from error
    if any(row["dest"] == original["path"] for row in uploads):
        kept = [row for row in uploads if row["dest"] != original["path"]]
        if len(kept) != len(uploads) - 1:
            raise ResumeRefused("full-state upload record is missing or duplicated")
        position = next(
            index for index, row in enumerate(uploads) if row["dest"] == original["path"]
        )
        return successor, kept[:position] + records + kept[position:]
    compact_dests = {row["dest"] for row in records}
    existing_compact = [row for row in uploads if row["dest"] in compact_dests]
    unexpected_raw = [
        row
        for row in uploads
        if row["dest"].startswith(original["path"]) and row["dest"] not in compact_dests
    ]
    if existing_compact != records or unexpected_raw:
        raise ResumeRefused("full-state upload record is missing or duplicated")
    return successor, list(uploads)


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _verified_regular_file(path: Path, descriptor: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(descriptor, Mapping):
        raise ResumeRefused(f"malformed {label} file descriptor")
    expected_path = descriptor.get("path")
    expected_size = descriptor.get("size")
    expected_sha = descriptor.get("sha256")
    if (
        not isinstance(expected_path, str)
        or type(expected_size) is not int
        or expected_size < 0
        or not _is_hex(expected_sha, 64)
    ):
        raise ResumeRefused(f"malformed {label} file descriptor")
    if path.is_symlink() or not path.is_file():
        raise ResumeRefused(f"missing or symlinked {label}: {path}")
    if str(path.resolve()) != str(Path(expected_path).resolve()):
        raise ResumeRefused(f"{label} path differs from bound descriptor: {path}")
    if path.stat().st_size != expected_size or digest(path) != expected_sha:
        raise ResumeRefused(f"{label} size or SHA-256 differs from bound descriptor: {path}")


def _refuse_if_fault_text(
    value: Any, *, field_name: str, check_supervisor_denylist: bool = False
) -> None:
    if not isinstance(value, str) or not value.strip():
        return
    lowered = value.lower()
    for marker in EXPLICIT_FAULT_EVIDENCE_MARKERS:
        if marker in lowered:
            raise ResumeRefused(
                f"prior evidence records explicit scientific/non-mechanical fault "
                f"in {field_name}: {marker!r}"
            )
    if check_supervisor_denylist:
        for marker in NON_MECHANICAL_SUPERVISOR_ERROR_MARKERS:
            if marker in lowered:
                raise ResumeRefused(
                    f"prior evidence records non-mechanical supervisor marker "
                    f"in {field_name}: {marker!r}"
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
            raise ResumeRefused(
                f"{source_name} exported non-empty faulthandler-stacks.log (stall watchdog fired)"
            )
        if "preparation-failed" in rel_path or "distribution-screen-failed" in rel_path:
            raise ResumeRefused(f"{source_name} exported failed stage artifact: {rel_path}")


def _check_terminal_fault_evidence(terminal: Mapping[str, Any], *, label: str) -> None:
    if terminal.get("returncode") == 124:
        raise ResumeRefused(f"{label} records watchdog timeout exit code 124")
    _refuse_if_fault_text(
        terminal.get("error"), field_name=f"{label}.error", check_supervisor_denylist=True
    )
    for key in ("last_traceback_line", "last_error_line", "job_log_tail"):
        _refuse_if_fault_text(terminal.get(key), field_name=f"{label}.{key}")
    _refuse_if_fault_exports(terminal.get("exports"), source_name=label)
    scanner = terminal.get("export_scanner")
    if isinstance(scanner, Mapping):
        _refuse_if_fault_exports(scanner.get("inventory"), source_name=f"{label}.export_scanner")


# All retained mappings use this vocabulary, independent of filename or depth.
NON_FAILURE_STATUSES = frozenset(
    {"running", "starting", "ready", "compiled", "pass", "ok", "healthy"}
)
MECHANICAL_RECEIPT_CLASSES = frozenset(
    {"RPC_EXEC_POLL_TIMEOUT", "SUPERVISOR", "CHILD_NONZERO_EXIT"}
)
RECORD_KEY_ALIASES = {"job": "job_id", "return_code": "returncode", "exit_code": "returncode"}


def _normalised_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    normal = {}
    for raw, value in record.items():
        key = str(raw).strip().lower().replace("-", "_")
        key = RECORD_KEY_ALIASES.get(key, key)
        if key in normal and normal[key] != value:
            raise ResumeRefused(f"{label} records conflicting job/job_id or field identities")
        normal[key] = value
    return normal


def _check_mechanical_receipt_class(record: Mapping[str, Any]) -> None:
    """The sole exception to failure admission: a validated mechanical receipt."""
    failure = record.get("failure_class")
    terminal = record.get("terminal")
    reason = terminal.get("reason") if isinstance(terminal, Mapping) else None
    if reason is not None and not isinstance(reason, str):
        raise ResumeRefused("malformed mechanical receipt terminal reason")
    if not isinstance(failure, str) or failure not in MECHANICAL_RECEIPT_CLASSES:
        raise ResumeRefused(
            f"prior receipt failure_class {failure!r} is not an eligible mechanical trigger"
        )
    if failure == "RPC_EXEC_POLL_TIMEOUT" and (
        terminal is None or reason in MECHANICAL_TERMINAL_REASONS
    ):
        return
    if reason in MECHANICAL_TERMINAL_REASONS:
        return
    if (
        failure == "SUPERVISOR"
        and terminal is None
        and _mechanical_supervisor_error(record.get("error"))
    ):
        return
    raise ResumeRefused(f"{failure} receipt lacks a mechanical timeout/lease terminal reason")


def _check_retained_execution(
    record: Mapping[str, Any],
    *,
    label: str,
    expected_job_id: str | None,
    expected_generation: str | None,
    require_identity: bool = False,
    allow_validated_incomplete: bool = False,
) -> None:
    """Normalize and validate EVERY mapping before examining its nested values.

    Missing status never suppresses identity or failure checks. Only the exact
    separately validated receipt root may carry the mechanical failure envelope;
    that exception is never propagated to a child mapping.
    """
    record = _normalised_record(record, label)
    for key, expected in (("job_id", expected_job_id), ("generation", expected_generation)):
        if key in record:
            value = record[key]
            if (
                not isinstance(value, str)
                or not value.strip()
                or (expected is not None and value != expected)
            ):
                raise ResumeRefused(f"{label} belongs to foreign {key} or malformed identity")
        elif require_identity:
            raise ResumeRefused(f"{label} lacks matching {key} identity")
    status = record.get("status")
    status = status.strip().lower() if isinstance(status, str) else status
    reason = record.get("reason")
    reason = reason.strip().lower() if isinstance(reason, str) else reason
    returncode = record.get("returncode")
    if status is not None and not isinstance(status, str):
        raise ResumeRefused(f"{label} records malformed execution status")
    if reason is not None and not isinstance(reason, str):
        raise ResumeRefused(f"{label} records malformed execution reason")
    if status in {"completed", "succeeded", "success"}:
        raise ResumeRefused(f"{label} records completed status")
    if reason is not None and reason not in MECHANICAL_TERMINAL_REASONS:
        raise ResumeRefused(f"{label} records non-mechanical reason: {reason!r}")
    if returncode is not None and type(returncode) is not int:
        raise ResumeRefused(f"{label} records malformed returncode")
    if returncode == 124:
        raise ResumeRefused(f"{label} records watchdog timeout exit code 124")
    if reason in MECHANICAL_TERMINAL_REASONS:
        # A mechanical terminal is meaningful only for the bound execution.
        for key, expected in (("job_id", expected_job_id), ("generation", expected_generation)):
            if expected is None or record.get(key) != expected:
                raise ResumeRefused(
                    f"{label} lacks matching {key} identity for mechanical terminal"
                )
    if allow_validated_incomplete:
        # Identities and exit facts still run before this narrow exception.
        if (
            status != "incomplete"
            or record.get("stop_verified") is not True
            or record.get("cleanup_error") is not None
        ):
            raise ResumeRefused(f"{label} is not a validated mechanical receipt envelope")
        _check_mechanical_receipt_class(record)
    elif (
        status is not None
        and status not in NON_FAILURE_STATUSES
        and not (status in {"failed", "stopped", "error"} and reason in MECHANICAL_TERMINAL_REASONS)
    ):
        raise ResumeRefused(
            f"{label} records unknown execution status or failed status: {status!r}"
        )
    if record.get("failure_class") not in (None, "") and not allow_validated_incomplete:
        raise ResumeRefused(f"{label} records non-receipt failure_class")
    if returncode not in (None, 0) and reason not in MECHANICAL_TERMINAL_REASONS:
        raise ResumeRefused(f"{label} records unexplained/non-mechanical child or runtime failure")


def _retained_files(output_root: Path) -> list[Path]:
    """Walk every retained file without following links or accepting special files."""
    files = []
    for path in sorted(output_root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ResumeRefused(f"non-regular or symlinked retained evidence: {path}")
        if stat.S_ISREG(mode):
            files.append(path)
    return files


def _mechanical_supervisor_error(value: Any) -> bool:
    """Match only complete qualified transport causes, never compound errors."""
    if not isinstance(value, str):
        return False
    causes = set(MECHANICAL_SUPERVISOR_ERROR_MARKERS) | {
        "whole-session wall cap reached (including setup/export)",
        "whole-session deadline reached before helper",
        "whole-session deadline reached before CLI call",
        "study reserve reached during calibration",
        "retention lease elapsed; no further job",
    }
    return value in causes or any(
        value == f"{wrapper}({cause!r})"
        for wrapper in ("SupervisorError", "Refused")
        for cause in causes
    )


def _scan_json_faults(
    value: Any,
    *,
    label: str,
    expected_job_id: str | None = None,
    expected_generation: str | None = None,
    supervisor_transport: bool = False,
    require_execution_identity: bool = False,
    validated_receipt_root: bool = False,
) -> None:
    """Inspect nested diagnostic fields, including error fields in snapshot headers."""
    pending = [
        (value, label, supervisor_transport, require_execution_identity, validated_receipt_root)
    ]
    while pending:
        current, name, transport, require_identity, receipt_root = pending.pop()
        if isinstance(current, Mapping):
            _check_retained_execution(
                current,
                label=name,
                expected_job_id=expected_job_id,
                expected_generation=expected_generation,
                require_identity=require_identity,
                allow_validated_incomplete=receipt_root,
            )
            for key, item in _normalised_record(current, name).items():
                field = f"{name}.{key}"
                if key == "stop_verified" and item is False:
                    raise ResumeRefused(f"{field} records unverified stop")
                mechanical_error = (
                    key == "error" and transport and _mechanical_supervisor_error(item)
                )
                if (
                    key
                    in {
                        "error",
                        "exception",
                        "last_error_line",
                        "last_traceback_line",
                        "cleanup_error",
                    }
                    and item
                    and not mechanical_error
                ):
                    raise ResumeRefused(
                        f"{field} records explicit scientific/non-mechanical fault: {item!r}"
                    )
                if key in {"phase", "prepare_phase", "stage"} and isinstance(item, str):
                    if item.lower().endswith("failed") or "failure" in item.lower():
                        raise ResumeRefused(f"{field} records failed stage")
                if (key == "finite" and item is False) or (key == "material_gaps" and item):
                    raise ResumeRefused(f"{field} records explicit scientific/non-mechanical fault")
                # Inventory paths are data, but mappings/lists under any key still validate.
                if key not in {"path", "dest", "sha256"} or not isinstance(item, str):
                    pending.append(
                        (
                            item,
                            field,
                            False,
                            key in {"state", "terminal"} and isinstance(item, Mapping),
                            False,
                        )
                    )
        elif isinstance(current, list):
            pending.extend(
                (item, f"{name}[{index}]", False, False, False)
                for index, item in enumerate(current)
            )
        elif isinstance(current, str):
            _refuse_if_fault_text(current, field_name=name)


def _retained_json_records(path: Path) -> list[Any]:
    """Parse JSON and JSONL even when an atomic-write suffix was retained."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        if ".jsonl" in path.suffixes:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            if not records:
                raise ValueError("empty JSONL evidence")
            return records
        try:
            return [json.loads(text)]
        except ValueError:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            if not records:
                raise ValueError("empty JSON evidence")
            return records
    except (ValueError, RecursionError) as error:
        _refuse_if_fault_text(text, field_name=str(path))
        label = (
            "corrupt poll evidence"
            if path.name.startswith("poll-")
            else ("unreadable or partial retained JSON evidence")
        )
        raise ResumeRefused(f"{label}: {path}") from error


STAGING_SUFFIX = r"\.(?:partial|part|tmp[A-Za-z0-9._-]*)"
MODEL_NAME = r"iteration-\d{4}(?:-actor\.pth|\.pt)(?:\.xz)?"


def _sidecar_target(path: Path) -> tuple[Path, str] | None:
    """Recognize canonical and suffix-staged sidecars, never their contents alone."""
    match = re.fullmatch(r"(.+)\.(sha256|complete)(?:" + STAGING_SUFFIX + r")?", path.name)
    return (path.with_name(match[1]), match[2]) if match else None


def _sidecar_bytes(path: Path, kind: str, expected: str | None = None) -> bool:
    """Allow only complete sidecars or recognizable byte prefixes of them."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024:
        raise ResumeRefused(f"non-regular or oversized retained publication sidecar: {path}")
    data = path.read_bytes()
    if kind == "complete":
        if not b"complete\n".startswith(data):
            raise ResumeRefused(f"malformed retained completion marker: {path}")
        return data == b"complete\n"
    # Producers here publish a bare lowercase digest followed by a newline.
    if not re.fullmatch(rb"[0-9a-f]{0,64}\n?", data) or (b"\n" in data and len(data) != 65):
        raise ResumeRefused(f"malformed retained publication sidecar: {path}")
    if expected is not None and not (expected.encode() + b"\n").startswith(data):
        raise ResumeRefused(f"corrupt retained publication sidecar: {path}")
    return len(data.rstrip(b"\n")) == 64


def _publication_sidecar(path: Path) -> bool:
    """Validate completion-last triples, allowing recognizable unfinished sidecars.

    A partial marker never finalizes a payload. A complete canonical marker must
    have its complete digest and payload; staging variants never finalize units.
    """
    target = _sidecar_target(path)
    if target is None:
        return False
    payload, kind = target
    sidecar, marker = Path(str(payload) + ".sha256"), Path(str(payload) + ".complete")
    try:
        if any(item.is_symlink() for item in (path, payload, sidecar, marker)):
            raise ResumeRefused(f"symlinked retained publication sidecar: {path}")
        expected = digest(payload) if payload.is_file() else None
        if path.exists():
            _sidecar_bytes(path, kind, expected)
        sha_full = sidecar.is_file() and _sidecar_bytes(sidecar, "sha256", expected)
        marker_full = marker.is_file() and _sidecar_bytes(marker, "complete")
        if marker_full:
            if not payload.is_file() or not sha_full:
                raise ResumeRefused(f"incomplete finalized publication: {payload}")
            return True
        return False
    except (OSError, UnicodeError) as error:
        raise ResumeRefused(f"unreadable retained publication sidecar: {path}") from error


class _RetainedTransport:
    """One byte-proof rule for canonical payloads and qualified transfer staging.

    supervisor._download_exports stages .transfer-*/{data,sha256,complete};
    transfers.download_contents stages .transfer-* and .transfer-*.<chunk>.
    Publication writers also leave .partial/.part/.tmp* suffixes. All use the
    same verified whole-payload or prefix proof. A final SHA alone cannot prove
    a truncated preimage. No temporary file becomes a ledger unit.
    """

    def __init__(self, output_root: Path, job: str | None, generation: str | None) -> None:
        self.root = output_root
        self.job, self.generation = job, generation
        self.bounds: dict[str, tuple[int, str]] | None = None
        self.verified: dict[Path, str] = {}

    def bindings(self) -> dict[str, tuple[int, str]]:
        if self.bounds is not None:
            return self.bounds
        bounds = {}
        for poll in sorted(self.root.glob("poll-*.json*")):
            if _sidecar_target(poll) is not None:
                continue
            if poll.is_symlink() or not poll.is_file():
                raise ResumeRefused(f"non-regular retained export inventory: {poll}")
            for record in _retained_json_records(poll):
                if not isinstance(record, Mapping):
                    raise ResumeRefused(f"malformed retained export inventory: {poll}")
                _scan_json_faults(
                    record,
                    label=str(poll),
                    expected_job_id=self.job,
                    expected_generation=self.generation,
                    supervisor_transport=True,
                )
                # Qualified saved polls put execution identity in state; event
                # logs may put it at the root. Both scopes have already passed
                # the SAME recursive validator, including every alias conflict.
                identity_scopes = [record, record.get("state")]
                matched = any(
                    isinstance(scope, Mapping)
                    and _normalised_record(scope, str(poll)).get("job_id") == self.job
                    and _normalised_record(scope, str(poll)).get("generation") == self.generation
                    for scope in identity_scopes
                )
                if self.job is None or self.generation is None or not matched:
                    continue
                exports = record.get("exports", [])
                if not isinstance(exports, list):
                    raise ResumeRefused(f"malformed retained export inventory: {poll}")
                for entry in exports:
                    if not isinstance(entry, Mapping):
                        raise ResumeRefused(f"malformed retained export descriptor: {poll}")
                    name, size, sha = entry.get("path"), entry.get("size"), entry.get("sha256")
                    if not isinstance(name, str) or not name.startswith("training/"):
                        continue
                    if Path(name).is_absolute() or any(
                        part in {"", ".", ".."} for part in name.split("/")
                    ):
                        raise ResumeRefused(f"unsafe retained export path: {poll}")
                    if type(size) is not int or size < 0 or not _is_hex(sha, 64):
                        raise ResumeRefused(f"malformed retained model export binding: {poll}")
                    if name in bounds and bounds[name] != (size, sha):
                        raise ResumeRefused(f"conflicting retained model export bindings: {name}")
                    bounds[name] = (size, sha)
        self.bounds = bounds
        return bounds

    def admit_payload(self, path: Path, payload: Path | None) -> tuple[str, str]:
        if path.is_symlink() or not path.is_file():
            raise ResumeRefused(f"non-regular retained model transport partial: {path}")
        # A canonical complete publication is an independent hash binding.
        candidates = (
            [payload] if payload is not None else sorted((self.root / "exports/training").iterdir())
        )
        verified = []
        for candidate in candidates:
            if (
                candidate is None
                or not candidate.is_file()
                or candidate.name.startswith(".")
                or _sidecar_target(candidate) is not None
            ):
                continue
            if candidate not in self.verified:
                sidecar = Path(str(candidate) + ".sha256")
                _publication_sidecar(sidecar)
                if candidate.is_file() and sidecar.is_file():
                    expected = digest(candidate)
                    if _sidecar_bytes(sidecar, "sha256", expected):
                        self.verified[candidate] = expected
            if candidate in self.verified:
                verified.append(candidate)
                if path.stat().st_size <= candidate.stat().st_size:
                    with path.open("rb") as partial, candidate.open("rb") as original:
                        while block := partial.read(1024 * 1024):
                            if block != original.read(len(block)):
                                break
                        else:
                            return self.verified[candidate], candidate.name
        matches = [
            (name, size, sha)
            for name, (size, sha) in self.bindings().items()
            if payload is None or name == f"training/{payload.name}"
        ]
        for name, size, sha in matches:
            if path.stat().st_size == size and digest(path) == sha:
                return sha, Path(name).name
        if verified:
            raise ResumeRefused(
                f"retained model transport partial differs from verified payload: {path}"
            )
        raise ResumeRefused(f"unverifiable retained model transport partial: {path}")

    def admit(self, path: Path) -> bool:
        training = self.root / "exports/training"
        try:
            relative = path.relative_to(training)
        except ValueError:
            return False
        # Qualified .transfer-* directory and file names, including chunk suffixes.
        staged_dir = any(
            part.startswith(
                (".transfer-", ".snake-transfer-", ".batch-export-", ".batch-download-")
            )
            for part in relative.parts[:-1]
        )
        opaque = path.name.startswith((".transfer-", ".snake-transfer-")) or (
            staged_dir and (path.name == "data" or re.fullmatch(r"\d{8}\.part", path.name))
        )
        model = re.fullmatch("(" + MODEL_NAME + r")(?:(?:" + STAGING_SUFFIX + r"))?", path.name)
        sidecar = _sidecar_target(path)
        if staged_dir and path.name in {"sha256", "complete"}:
            # The staging directory must contain byte-proven model data.
            data = path.with_name("data")
            expected, name = self.admit_payload(data, None)
            _sidecar_bytes(path, path.name, expected)
            if not re.fullmatch(MODEL_NAME, name):
                _scan_retained_text(
                    data,
                    output_root=self.root,
                    expected_job_id=self.job,
                    expected_generation=self.generation,
                    _skip_transport=True,
                )
            return True
        if sidecar is not None:
            _publication_sidecar(path)
            return True
        if opaque:
            # The downloader uses anonymous .transfer-* names for sidecars too.
            # Only bounded recognizable sidecar bytes may bypass payload proof.
            if path.stat().st_size <= 1024:
                data = path.read_bytes()
                if b"complete\n".startswith(data):
                    return True
                if re.fullmatch(rb"[0-9a-f]{0,64}\n?", data) and (
                    b"\n" not in data or len(data) == 65
                ):
                    return True
        if opaque or (model and path.parent == training):
            _, name = self.admit_payload(path, path.with_name(model[1]) if model else None)
            # Hash binding never excuses failure/identity signals in JSON/text.
            return bool(re.fullmatch(MODEL_NAME, name))
        return False


def _verify_staged_source_archive(path: Path, output_root: Path) -> None:
    """Exempt the expected source archive only after checking pinned Git blobs."""
    try:
        plan = json.loads((path.parent / "source-plan.json").read_text())
        manifest = json.loads((output_root / "manifest.json").read_text())
        commit = manifest["commit"]
        files = plan["files"]
        if (
            not _is_hex(commit, 40)
            or not isinstance(files, dict)
            or not files
            or plan.get("commit") != commit
            or plan.get("fallback_archive") != path.name
            or plan.get("fallback_size") != path.stat().st_size
            or path.stat().st_size > 32 * 1024 * 1024
            or plan.get("fallback_sha256") != digest(path)
        ):
            raise ValueError("source plan binding differs")
        tree = subprocess.run(
            ["git", "ls-tree", "-rz", commit, "--", *files],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=True,
            timeout=20,
        ).stdout
        blobs = {}
        for row in tree.split(b"\x00"):
            if row:
                meta, name = row.split(b"\t", 1)
                mode, kind, blob = meta.decode().split()
                if kind != "blob" or mode not in {"100644", "100755"}:
                    raise ValueError("source plan contains non-regular Git entry")
                blobs[name.decode()] = blob
        if set(blobs) != set(files):
            raise ValueError("source plan differs from pinned Git inventory")
        seen = set()
        total = 0
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                total += member.size
                if (
                    not member.isfile()
                    or member.name not in files
                    or member.name in seen
                    or member.size < 0
                    or total > 32 * 1024 * 1024
                ):
                    raise ValueError("unexpected or oversized source archive member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("unreadable source member")
                data = stream.read()
                blob = hashlib.sha1(f"blob {len(data)}\x00".encode() + data).hexdigest()
                if (
                    blob != blobs[member.name]
                    or len(data) != files[member.name]["size"]
                    or hashlib.sha256(data).hexdigest() != files[member.name]["sha256"]
                ):
                    raise ValueError("source archive differs from pinned Git bytes")
                seen.add(member.name)
        if seen != set(files):
            raise ValueError("source archive omits pinned files")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        tarfile.TarError,
        subprocess.SubprocessError,
    ) as error:
        raise ResumeRefused(
            f"unverified retained staged source archive: {path}: {error}"
        ) from error


def _scan_retained_text(
    path: Path,
    *,
    output_root: Path,
    expected_job_id: str | None = None,
    expected_generation: str | None = None,
    validated_receipt: Mapping[str, Any] | None = None,
    transport: _RetainedTransport | None = None,
    _skip_transport: bool = False,
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
            raise ResumeRefused(f"retained staged executable differs from qualified source: {path}")
        return
    transport = transport or _RetainedTransport(output_root, expected_job_id, expected_generation)
    if not _skip_transport and transport.admit(path):
        return
    model_metadata = _sidecar_target(path) is not None
    model_manifest = re.fullmatch(r"iteration-\d{4}\.pt\.xz\.manifest", path.name)
    in_training = path.parent == output_root / "exports/training"
    if path == output_root / "staging/source.tar.gz":
        _verify_staged_source_archive(path, output_root)
        return
    with path.open("rb") as stream:
        header = stream.read(8)
    if header.startswith(
        (
            b"\x1f\x8b",
            b"\xfd7zXZ\x00",
            b"PK\x03\x04",
            b"BZh",
            b"\x28\xb5\x2f\xfd",
            b"7z\xbc\xaf\x27\x1c",
        )
    ):
        raise ResumeRefused(f"unsupported retained diagnostic archive: {path}")
    if any(
        suffix in {".xz", ".gz", ".zip", ".bz2", ".zst", ".tar", ".7z"} for suffix in path.suffixes
    ) and not (in_training and (model_metadata or model_manifest)):
        raise ResumeRefused(f"unsupported retained diagnostic archive: {path}")
    if any(suffix in {".pt", ".pth"} for suffix in path.suffixes) and not (
        in_training and (model_metadata or model_manifest)
    ):
        raise ResumeRefused(f"unexpected retained binary payload: {path}")
    publication_sidecar = _sidecar_target(path) is not None
    if publication_sidecar:
        _publication_sidecar(path)
    is_json = (
        any(suffix in {".json", ".jsonl"} for suffix in path.suffixes) and not publication_sidecar
    ) or (in_training and model_manifest is not None)
    if not publication_sidecar:
        with path.open(encoding="utf-8", errors="replace") as probe:
            prefix = probe.read(4096).lstrip()
        is_json = is_json or prefix.startswith(("{", "["))
    if is_json:
        for payload in _retained_json_records(path):
            execution_file = path.name.startswith(("state.json", "terminal.json", "poll-"))
            if execution_file and not isinstance(payload, Mapping):
                raise ResumeRefused(f"malformed retained execution record: {path}")
            _scan_json_faults(
                payload,
                label=str(path),
                expected_job_id=expected_job_id,
                expected_generation=expected_generation,
                supervisor_transport=(
                    path.parent == output_root
                    and (path.name.startswith("poll-") or path.name == "receipt.json")
                ),
                require_execution_identity=(execution_file and not path.name.startswith("poll-")),
                validated_receipt_root=(
                    path == output_root / "receipt.json"
                    and validated_receipt is not None
                    and payload == validated_receipt
                ),
            )
        return
    if not publication_sidecar:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                stripped = line.replace("\x00", "").strip()
                if stripped.startswith(("{", "[")):
                    try:
                        record = json.loads(stripped)
                    except ValueError as error:
                        raise ResumeRefused(f"partial retained JSON log record: {path}") from error
                    _scan_json_faults(
                        record,
                        label=str(path),
                        expected_job_id=expected_job_id,
                        expected_generation=expected_generation,
                    )
    with path.open(encoding="utf-8", errors="replace") as stream:
        overlap = ""
        while part := stream.read(65536):
            part = part.replace("\x00", "")
            combined = overlap + part
            _refuse_if_fault_text(combined, field_name=str(path))
            overlap = combined[-256:]


def _check_output_root_fault_evidence(
    output_root: Path,
    *,
    expected_job_id: str | None = None,
    expected_generation: str | None = None,
    validated_receipt: Mapping[str, Any] | None = None,
) -> tuple[int, int]:
    """Scan ALL retained attempt telemetry and exports for scientific/non-mechanical faults.

    Returns ``(poll_count, identity_matched_poll_count)``.  Every saved
    ``poll-*.json`` is scanned from earliest to latest (never a trailing window)
    and any unreadable, non-dict, or symlinked poll/telemetry file refuses closed.
    """
    if output_root.is_symlink():
        raise ResumeRefused("prior output directory must not be a symlink")
    if not output_root.exists():
        return 0, 0
    if not output_root.is_dir():
        raise ResumeRefused("prior output_root is not a directory")

    transport = _RetainedTransport(output_root, expected_job_id, expected_generation)
    for retained_file in _retained_files(output_root):
        _scan_retained_text(
            retained_file,
            output_root=output_root,
            expected_job_id=expected_job_id,
            expected_generation=expected_generation,
            validated_receipt=validated_receipt,
            transport=transport,
        )

    terminal_file = output_root / "terminal.json"
    if terminal_file.is_symlink():
        raise ResumeRefused("symlinked terminal.json in prior output directory")
    if terminal_file.is_file():
        try:
            persisted_terminal = json.loads(terminal_file.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeRefused(
                "corrupt persisted terminal.json in prior output directory"
            ) from error
        if not isinstance(persisted_terminal, Mapping):
            raise ResumeRefused("malformed persisted terminal.json in prior output directory")
        if persisted_terminal.get("status") in {"completed", "succeeded"}:
            raise ResumeRefused("persisted terminal.json already completed; refusing continuation")
        _check_retained_execution(
            persisted_terminal,
            label="persisted_terminal",
            expected_job_id=expected_job_id,
            expected_generation=expected_generation,
            require_identity=True,
        )
        _check_terminal_fault_evidence(persisted_terminal, label="persisted_terminal")

    summary_base = output_root / "exports/training/summary.json"
    for suffix in ("", ".partial", ".sha256", ".complete"):
        probe = Path(f"{summary_base}{suffix}")
        if probe.is_symlink() or probe.exists():
            raise ResumeRefused("exported summary.json already exists; prior run completed")

    stacks_path = output_root / "exports/training/stall-watchdog/faulthandler-stacks.log"
    if stacks_path.is_symlink():
        raise ResumeRefused("symlinked faulthandler-stacks.log in prior output directory")
    if stacks_path.is_file() and stacks_path.stat().st_size > 0:
        raise ResumeRefused("exported faulthandler-stacks.log is non-empty (stall watchdog fired)")

    watchdog_json = output_root / "exports/training/stall-watchdog/stall-watchdog.json"
    if watchdog_json.is_symlink():
        raise ResumeRefused("symlinked stall-watchdog.json in prior output directory")
    if watchdog_json.is_file():
        try:
            watchdog_payload = json.loads(watchdog_json.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeRefused("corrupt exported stall-watchdog.json") from error
        if not isinstance(watchdog_payload, Mapping):
            raise ResumeRefused("malformed exported stall-watchdog.json")
        if str(watchdog_payload.get("stage", "")).lower().endswith("-failed"):
            raise ResumeRefused(
                f"exported stall-watchdog.json records failed stage: "
                f"{watchdog_payload.get('stage')!r}"
            )

    screen_path = output_root / "exports/training/distribution-screen.json"
    if screen_path.is_symlink():
        raise ResumeRefused("symlinked distribution-screen.json in prior output directory")
    if screen_path.is_file():
        try:
            screen_payload = json.loads(screen_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeRefused("corrupt exported distribution-screen.json") from error
        if not isinstance(screen_payload, Mapping):
            raise ResumeRefused("malformed exported distribution-screen.json")
        if screen_payload.get("material_gaps"):
            raise ResumeRefused(
                "exported distribution-screen.json records material distribution gaps"
            )

    progress_path = output_root / "exports/training/progress.json"
    if progress_path.is_symlink():
        raise ResumeRefused("symlinked progress.json in prior output directory")
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeRefused("corrupt exported progress.json") from error
        if not isinstance(progress, Mapping):
            raise ResumeRefused("malformed exported progress.json")
        if str(progress.get("stage", "")).lower().endswith("-failed"):
            raise ResumeRefused(
                f"exported progress.json records failed stage: {progress.get('stage')!r}"
            )
        _refuse_if_fault_text(progress.get("error"), field_name="progress.json.error")

    training_exports = output_root / "exports/training"
    if training_exports.is_symlink():
        raise ResumeRefused("symlinked training export directory")
    if training_exports.is_dir():
        header_candidates = [training_exports / "calibrated-boundary.json"] + sorted(
            training_exports.glob("iteration-*.json")
        )
        for header_path in header_candidates:
            if header_path.is_symlink():
                raise ResumeRefused(f"symlinked snapshot header: {header_path.name}")
            if not header_path.is_file():
                continue
            try:
                header_payload = json.loads(header_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                if _complete(header_path):
                    raise ResumeRefused(
                        f"corrupt completed snapshot header: {header_path.name}"
                    ) from error
                continue
            if not isinstance(header_payload, Mapping):
                if _complete(header_path):
                    raise ResumeRefused(f"malformed snapshot header: {header_path.name}")
                continue
            meta = header_payload.get("metadata")
            if isinstance(meta, Mapping):
                phase_str = str(meta.get("phase", "")).lower()
                prep_str = str(meta.get("prepare_phase", "")).lower()
                if "fail" in phase_str or prep_str == "failed":
                    raise ResumeRefused(
                        f"exported {header_path.name} records failed preparation/trajectory phase"
                    )

    poll_count = 0
    identity_matched_polls = 0
    for poll_path in sorted(output_root.glob("poll-*.json*")):
        if _sidecar_target(poll_path) is not None:
            _publication_sidecar(poll_path)
            continue
        if poll_path.is_symlink() or not poll_path.is_file():
            raise ResumeRefused(f"non-regular or symlinked poll evidence: {poll_path.name}")
        records = _retained_json_records(poll_path)
        if len(records) != 1:
            raise ResumeRefused(f"poll evidence must contain one complete poll: {poll_path.name}")
        poll_payload = records[0]
        if not isinstance(poll_payload, Mapping):
            raise ResumeRefused(f"malformed poll evidence: {poll_path.name}")
        poll_count += 1
        _check_retained_execution(
            poll_payload,
            label=poll_path.name,
            expected_job_id=expected_job_id,
            expected_generation=expected_generation,
        )
        if poll_payload.get("status") in {"completed", "succeeded"}:
            raise ResumeRefused(f"{poll_path.name} records completed status")
        matched_this_poll = False
        for key in ("job", "job_id"):
            if key in poll_payload:
                if expected_job_id is not None and poll_payload[key] != expected_job_id:
                    raise ResumeRefused(
                        f"{poll_path.name} belongs to foreign {key}: {poll_payload[key]!r}"
                    )
                matched_this_poll = True
        if "generation" in poll_payload and expected_generation is not None:
            if poll_payload["generation"] != expected_generation:
                raise ResumeRefused(
                    f"{poll_path.name} belongs to foreign generation: "
                    f"{poll_payload['generation']!r}"
                )
        poll_state = poll_payload.get("state")
        if poll_state is not None:
            if not isinstance(poll_state, Mapping):
                raise ResumeRefused(f"{poll_path.name}.state is malformed")
            if poll_state.get("status") in {"completed", "succeeded"}:
                raise ResumeRefused(f"{poll_path.name}.state records completed status")
            _check_retained_execution(
                poll_state,
                label=f"{poll_path.name}.state",
                expected_job_id=expected_job_id,
                expected_generation=expected_generation,
                require_identity=True,
            )
            if "job_id" in poll_state:
                if expected_job_id is not None and poll_state["job_id"] != expected_job_id:
                    raise ResumeRefused(
                        f"{poll_path.name}.state belongs to foreign job_id: "
                        f"{poll_state['job_id']!r}"
                    )
            if "generation" in poll_state:
                if (
                    expected_generation is not None
                    and poll_state["generation"] != expected_generation
                ):
                    raise ResumeRefused(
                        f"{poll_path.name}.state belongs to foreign generation: "
                        f"{poll_state['generation']!r}"
                    )
            if (
                expected_job_id is not None
                and expected_generation is not None
                and poll_state.get("job_id") == expected_job_id
                and poll_state.get("generation") == expected_generation
            ):
                matched_this_poll = True
            _refuse_if_fault_text(
                poll_state.get("error"), field_name=f"{poll_path.name}.state.error"
            )
        if matched_this_poll:
            identity_matched_polls += 1
        _refuse_if_fault_text(
            poll_payload.get("error"),
            field_name=f"{poll_path.name}.error",
            check_supervisor_denylist=True,
        )
        _refuse_if_fault_text(
            poll_payload.get("job_log_tail"), field_name=f"{poll_path.name}.job_log_tail"
        )
        _refuse_if_fault_text(
            poll_payload.get("supervisor_log_tail"),
            field_name=f"{poll_path.name}.supervisor_log_tail",
        )
        _refuse_if_fault_exports(poll_payload.get("exports"), source_name=poll_path.name)
        poll_terminal = poll_payload.get("terminal")
        if poll_terminal is not None:
            if not isinstance(poll_terminal, Mapping):
                raise ResumeRefused(f"{poll_path.name}.terminal is malformed")
            if poll_terminal.get("status") in {"completed", "succeeded"}:
                raise ResumeRefused(f"{poll_path.name}.terminal already completed")
            _check_retained_execution(
                poll_terminal,
                label=f"{poll_path.name}.terminal",
                expected_job_id=expected_job_id,
                expected_generation=expected_generation,
                require_identity=True,
            )
            _check_terminal_fault_evidence(poll_terminal, label=f"{poll_path.name}.terminal")
    return poll_count, identity_matched_polls


def _check_supervisor_log_fault_evidence(
    supervisor_log_path: Path,
    *,
    expected_job_id: str | None = None,
    expected_generation: str | None = None,
    allow_terminal_receipt_enospc: bool = False,
    expected_output_root: Path | None = None,
) -> dict[str, Any]:
    """Scan EVERY line of a supervisor log for scientific/non-mechanical fault evidence."""
    if supervisor_log_path.is_symlink() or not supervisor_log_path.is_file():
        raise ResumeRefused(f"missing or symlinked supervisor log: {supervisor_log_path}")
    try:
        raw_lines = (
            supervisor_log_path.read_text(encoding="utf-8", errors="replace")
            .replace("\x00", "")
            .splitlines()
        )
    except OSError as error:
        raise ResumeRefused(f"unreadable supervisor log: {supervisor_log_path}") from error

    nonempty_indices = [idx for idx, line in enumerate(raw_lines) if line.strip()]
    enospc_line_index: int | None = None
    exit_line_index: int | None = None
    if allow_terminal_receipt_enospc:
        if len(nonempty_indices) < 3 or expected_output_root is None:
            raise ResumeRefused(
                "host ENOSPC recovery requires supervisor log with poll history and terminal lines"
            )
        enospc_line_index = nonempty_indices[-2]
        exit_line_index = nonempty_indices[-1]
        enospc_text = raw_lines[enospc_line_index].strip()
        exit_text = raw_lines[exit_line_index].strip()
        match = ENOSPC_RECEIPT_LINE_RE.fullmatch(enospc_text)
        if match is None or exit_text != "EXIT=1":
            raise ResumeRefused("supervisor log does not end with receipt-write ENOSPC and EXIT=1")
        enospc_dir = Path(match.group(1)).resolve()
        if enospc_dir != expected_output_root.resolve():
            raise ResumeRefused(
                f"supervisor log ENOSPC path {enospc_dir} differs from {expected_output_root}"
            )

    matched_job_records = 0
    for idx, raw_line in enumerate(raw_lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if allow_terminal_receipt_enospc and idx in (enospc_line_index, exit_line_index):
            continue
        line_no = idx + 1
        if stripped.startswith("{"):
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ResumeRefused(
                    f"corrupt JSON record in supervisor log at line {line_no}"
                ) from error
            if not isinstance(record, Mapping):
                raise ResumeRefused(f"malformed JSON record in supervisor log at line {line_no}")
            for key in ("job", "job_id"):
                if key in record:
                    if expected_job_id is not None and record[key] != expected_job_id:
                        raise ResumeRefused(
                            f"supervisor log line {line_no} belongs to foreign {key}: "
                            f"{record[key]!r}"
                        )
                    matched_job_records += 1
            if record.get("status") in {"completed", "succeeded"}:
                raise ResumeRefused(
                    f"supervisor log line {line_no} records completed run; refusing continuation"
                )
            _check_retained_execution(
                record,
                label=f"supervisor_log:{line_no}",
                expected_job_id=expected_job_id,
                expected_generation=expected_generation,
            )
            for key in ("state", "terminal"):
                retained = record.get(key)
                # Supervisor event logs use terminal:false as a presence flag;
                # saved poll payloads carry the full terminal object instead.
                if key == "terminal" and retained is False:
                    continue
                if retained is not None:
                    if not isinstance(retained, Mapping):
                        raise ResumeRefused(f"supervisor_log:{line_no}.{key} is malformed")
                    _check_retained_execution(
                        retained,
                        label=f"supervisor_log:{line_no}.{key}",
                        expected_job_id=expected_job_id,
                        expected_generation=expected_generation,
                        require_identity=True,
                    )
            _scan_json_faults(
                record,
                label=f"supervisor_log:{line_no}",
                expected_job_id=expected_job_id,
                expected_generation=expected_generation,
                supervisor_transport=True,
            )
            _refuse_if_fault_text(
                record.get("error"),
                field_name=f"supervisor_log:{line_no}.error",
                check_supervisor_denylist=True,
            )
            _refuse_if_fault_text(
                record.get("job_log_tail"),
                field_name=f"supervisor_log:{line_no}.job_log_tail",
            )
            _refuse_if_fault_text(
                record.get("supervisor_log_tail"),
                field_name=f"supervisor_log:{line_no}.supervisor_log_tail",
            )
            if isinstance(record.get("exports"), list):
                _refuse_if_fault_exports(
                    record.get("exports"), source_name=f"supervisor_log:{line_no}"
                )
            rec_terminal = record.get("terminal")
            if isinstance(rec_terminal, Mapping):
                if rec_terminal.get("status") in {"completed", "succeeded"}:
                    raise ResumeRefused(f"supervisor_log:{line_no}.terminal already completed")
                _check_terminal_fault_evidence(
                    rec_terminal, label=f"supervisor_log:{line_no}.terminal"
                )
        _refuse_if_fault_text(stripped, field_name=f"supervisor_log:{line_no}")
    return {
        "line_count": len(raw_lines),
        "matched_job_records": matched_job_records,
        "terminal_enospc_line": (
            raw_lines[enospc_line_index].strip() if enospc_line_index is not None else None
        ),
        "terminal_exit_line": (
            raw_lines[exit_line_index].strip() if exit_line_index is not None else None
        ),
    }


def verify_mechanical_stop(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    *,
    output_root: Path | None = None,
    supervisor_log_path: Path | None = None,
) -> None:
    """Require an exact, released mechanical stop before a manual successor.

    Explicit scientific, preparation, distribution-screen, or stall-watchdog
    failure evidence in ``receipt``, ``terminal``, exported inventories,
    ``output_root`` (across ALL retained polls), or ``supervisor_log_path``
    always takes precedence over any timeout/lease/wall-cap marker and refuses
    continuation.

    Admits only mechanical timeout/recycle/poll terminations with verified VM
    release and clean cleanup:
    - ``RPC_EXEC_POLL_TIMEOUT`` (with ``terminal`` absent or bearing a
      mechanical timeout/lease reason),
    - ``SUPERVISOR`` when ``terminal.reason`` is a mechanical timeout/lease
      reason or ``terminal`` is absent and ``error`` names a host wall/unit/lease
      cap or read-retry exhaustion (never ``KeyboardInterrupt``, scientific
      tripwires, or missing/corrupt exports),
    - ``CHILD_NONZERO_EXIT`` only when ``terminal.reason`` explicitly records a
      mechanical VM timeout/lease/credential stop and no conflicting scientific
      or stall fault evidence is present.
    """
    terminal = receipt.get("terminal")
    if not _is_hex(manifest_sha256, 64):
        raise ResumeRefused("original manifest digest is malformed")
    if (
        receipt.get("job_id") != manifest.get("job_id")
        or receipt.get("commit") != manifest.get("commit")
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("status") != "INCOMPLETE"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
    ):
        raise ResumeRefused("prior receipt does not prove a verified released incomplete run")
    _refuse_if_fault_text(
        receipt.get("error"), field_name="receipt.error", check_supervisor_denylist=True
    )
    _refuse_if_fault_exports(receipt.get("exports"), source_name="receipt")
    terminal_reason: Any = None
    if terminal is not None:
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("job_id") != manifest.get("job_id")
            or terminal.get("generation") != receipt.get("generation")
        ):
            raise ResumeRefused("terminal identity differs from the incomplete receipt")
        if terminal.get("status") in {"completed", "succeeded"}:
            raise ResumeRefused("prior terminal already completed; refusing continuation")
        _check_terminal_fault_evidence(terminal, label="terminal")
        terminal_reason = terminal.get("reason")
        if (
            terminal.get("returncode") not in (None, 0)
            and terminal_reason not in MECHANICAL_TERMINAL_REASONS
        ):
            raise ResumeRefused("terminal records a non-mechanical child failure")
    expected_job_id = manifest.get("job_id") if isinstance(manifest.get("job_id"), str) else None
    expected_generation = (
        receipt.get("generation") if isinstance(receipt.get("generation"), str) else None
    )
    if output_root is not None:
        _check_output_root_fault_evidence(
            output_root,
            expected_job_id=expected_job_id,
            expected_generation=expected_generation,
            validated_receipt=receipt,
        )
    if supervisor_log_path is not None:
        _check_supervisor_log_fault_evidence(
            supervisor_log_path,
            expected_job_id=expected_job_id,
            expected_generation=expected_generation,
        )
    _check_retained_execution(
        receipt,
        label="receipt",
        expected_job_id=expected_job_id,
        expected_generation=expected_generation,
        allow_validated_incomplete=True,
    )


def _glob_inventory(output_root: Path, pattern: str) -> tuple[int, str, str | None, str | None]:
    rows: list[dict[str, Any]] = []
    for item in sorted(output_root.glob(pattern)):
        if _publication_sidecar(item):
            continue
        if item.is_symlink() or not item.is_file():
            raise ResumeRefused(f"non-regular or symlinked retained evidence file: {item.name}")
        rows.append({"name": item.name, "size": item.stat().st_size, "sha256": digest(item)})
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (
        len(rows),
        hashlib.sha256(encoded).hexdigest(),
        rows[0]["name"] if rows else None,
        rows[-1]["name"] if rows else None,
    )


def compute_retained_evidence_inventory(output_root: Path) -> dict[str, Any]:
    """Build a deterministic hash-bound inventory of all retained attempt telemetry."""
    if output_root.is_symlink() or not output_root.is_dir():
        raise ResumeRefused(f"missing or symlinked output_root: {output_root}")
    poll_count, polls_sha256, first_poll, last_poll = _glob_inventory(output_root, "poll-*.json*")
    spend_count, spend_sha256, _, _ = _glob_inventory(output_root, "spend-*.json*")
    usage_count, usage_sha256, _, _ = _glob_inventory(output_root, "usage-*.json*")
    receipt_path = output_root / "receipt.json"
    terminal_path = output_root / "terminal.json"
    optional_files: dict[str, dict[str, Any] | None] = {}
    for rel_path in TRACKED_OPTIONAL_OUTPUT_FILES:
        candidate = output_root / rel_path
        if candidate.is_symlink():
            raise ResumeRefused(f"symlinked retained output artifact: {rel_path}")
        if candidate.is_file():
            optional_files[rel_path] = {
                "size": candidate.stat().st_size,
                "sha256": digest(candidate),
            }
        else:
            optional_files[rel_path] = None
    return {
        "files": [
            {
                "path": str(path.relative_to(output_root)),
                "size": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in _retained_files(output_root)
        ],
        "poll_count": poll_count,
        "first_poll": first_poll,
        "last_poll": last_poll,
        "polls_sha256": polls_sha256,
        "spend_count": spend_count,
        "spend_sha256": spend_sha256,
        "usage_count": usage_count,
        "usage_sha256": usage_sha256,
        "receipt_present": receipt_path.exists() or receipt_path.is_symlink(),
        "terminal_present": terminal_path.exists() or terminal_path.is_symlink(),
        "optional_files": optional_files,
    }


def verify_host_enospc_recovery(
    recovery_evidence_path: Path,
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
    """Validate an explicit, create-only host-side ENOSPC recovery certificate.

    This narrow admission path covers a host disk-full failure while writing the
    supervisor receipt after remote execution stopped.  It preserves the missing
    original ``receipt.json`` as absent, preserves unknown original terminal and
    cleanup fields as ``UNKNOWN_UNAVAILABLE`` (never fabricating ``cleanup_error:
    null`` or a wall-timeout classification), verifies post-hoc server session
    disappearance and zero account assignments, reconciles broker assignment
    records, and scans EVERY retained poll, export, and supervisor log line for
    scientific or non-mechanical fault evidence.
    """
    if not _is_hex(manifest_sha256, 64):
        raise ResumeRefused("original manifest digest is malformed")
    if output_root.is_symlink() or not output_root.is_dir():
        raise ResumeRefused(f"missing or symlinked output_root: {output_root}")
    resolved_output = output_root.resolve()
    resolved_cert = recovery_evidence_path.resolve()
    if recovery_evidence_path.name == "receipt.json" or resolved_cert.parent == resolved_output:
        raise ResumeRefused(
            "recovery certificate must be separately named outside output_root (never receipt.json)"
        )
    cert_desc = _descriptor(recovery_evidence_path, runtime_path=recovery_evidence_path.name)
    try:
        certificate = json.loads(recovery_evidence_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeRefused("unreadable host ENOSPC recovery certificate") from error
    if (
        not isinstance(certificate, Mapping)
        or certificate.get("schema") != HOST_ENOSPC_RECOVERY_SCHEMA
    ):
        raise ResumeRefused("unsupported or malformed host ENOSPC recovery certificate schema")

    terminal_file = resolved_output / "terminal.json"
    if any(resolved_output.glob("receipt.json*")):
        raise ResumeRefused(
            "receipt.json exists in output_root; use ordinary receipt verification, "
            "not ENOSPC recovery"
        )
    if terminal_file.exists() or terminal_file.is_symlink():
        raise ResumeRefused(
            "persisted terminal.json exists in output_root; "
            "narrow receipt-less ENOSPC path disagrees"
        )

    attempt_state = certificate.get("original_attempt_state")
    if (
        not isinstance(attempt_state, Mapping)
        or attempt_state.get("prior_attempt_status") != "INCOMPLETE"
        or attempt_state.get("failure_class") != HOST_ENOSPC_FAILURE_CLASS
        or attempt_state.get("original_receipt_present") is not False
        or attempt_state.get("original_terminal_status") != "UNKNOWN_UNAVAILABLE"
        or attempt_state.get("original_supervisor_error") != "UNKNOWN_UNAVAILABLE"
        or attempt_state.get("original_cleanup_error") != "UNKNOWN_UNAVAILABLE"
        or attempt_state.get("post_hoc_release_verified") is not True
    ):
        raise ResumeRefused(
            "recovery certificate must preserve attempt INCOMPLETE and unknown original "
            "receipt/terminal/cleanup fields without inventing cleanup_error=null"
        )

    identities = certificate.get("identities")
    if not isinstance(identities, Mapping):
        raise ResumeRefused("recovery certificate lacks identities section")
    job_id = manifest.get("job_id")
    source_commit = manifest.get("commit")
    manifest_output_dir = manifest.get("output_dir")
    if (
        not isinstance(job_id, str)
        or not job_id
        or not _is_hex(source_commit, 40)
        or not isinstance(manifest_output_dir, str)
        or Path(manifest_output_dir).resolve() != resolved_output
    ):
        raise ResumeRefused("executed manifest identity or output_dir does not match output_root")
    if source_control.get("source_commit") != source_commit:
        raise ResumeRefused("source_control commit differs from executed manifest")
    expected_allocation = allocation(dict(source_control))
    generation = identities.get("generation")
    endpoint = identities.get("endpoint")
    session = identities.get("session")
    if (
        identities.get("study") != source_control.get("study")
        or identities.get("namespace") != NAMESPACE
        or identities.get("job_id") != job_id
        or identities.get("executed_manifest_sha256") != manifest_sha256
        or identities.get("source_commit") != source_commit
        or identities.get("control_allocation") != expected_allocation
        or identities.get("output_dir") != str(resolved_output)
        or not _is_hex(generation, 32)
        or not isinstance(endpoint, str)
        or not endpoint.strip()
        or session != f"snake-{job_id}-{generation[:12]}"
    ):
        raise ResumeRefused("recovery certificate identities do not match frozen attempt/control")
    if source_packet_id is not None and identities.get("source_packet_id") != source_packet_id:
        raise ResumeRefused("recovery certificate source_packet_id mismatch")
    if (
        not isinstance(identities.get("source_packet_id"), str)
        or not identities["source_packet_id"]
    ):
        raise ResumeRefused("recovery certificate lacks source_packet_id")
    if source_packet_manifest_sha256 is not None:
        if identities.get("source_packet_manifest_sha256") != source_packet_manifest_sha256:
            raise ResumeRefused("recovery certificate source_packet_manifest_sha256 mismatch")
    elif not _is_hex(identities.get("source_packet_manifest_sha256"), 64):
        raise ResumeRefused("recovery certificate lacks valid source_packet_manifest_sha256")

    assignment_desc = identities.get("assignment_record")
    if not isinstance(assignment_desc, Mapping) or not isinstance(assignment_desc.get("path"), str):
        raise ResumeRefused("recovery certificate lacks assignment_record binding")
    assignment_path = Path(assignment_desc["path"])
    _verified_regular_file(assignment_path, assignment_desc, label="assignment.json")
    try:
        assignment = json.loads(assignment_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeRefused("unreadable assignment.json") from error
    if (
        not isinstance(assignment, Mapping)
        or assignment.get("endpoint") != endpoint
        or assignment.get("generation") != generation
        or assignment.get("session") != session
        or not isinstance(assignment.get("output"), str)
        or Path(assignment["output"]).resolve() != resolved_output
    ):
        raise ResumeRefused("assignment.json disagrees with recovery certificate identities")

    sup_log = certificate.get("supervisor_log")
    if not isinstance(sup_log, Mapping) or not isinstance(sup_log.get("path"), str):
        raise ResumeRefused("recovery certificate lacks supervisor_log binding")
    sup_log_path = Path(sup_log["path"])
    _verified_regular_file(sup_log_path, sup_log, label="supervisor_log")
    log_audit = _check_supervisor_log_fault_evidence(
        sup_log_path,
        expected_job_id=job_id,
        expected_generation=generation,
        allow_terminal_receipt_enospc=True,
        expected_output_root=resolved_output,
    )
    if (
        sup_log.get("line_count") != log_audit["line_count"]
        or sup_log.get("terminal_enospc_line") != log_audit["terminal_enospc_line"]
        or sup_log.get("terminal_exit_line") != log_audit["terminal_exit_line"]
        or log_audit["matched_job_records"] < 1
    ):
        raise ResumeRefused("supervisor_log line audit disagrees with recovery certificate")

    retained_decl = certificate.get("retained_evidence")
    actual_inventory = compute_retained_evidence_inventory(resolved_output)
    if not isinstance(retained_decl, Mapping) or dict(retained_decl) != actual_inventory:
        raise ResumeRefused(
            "retained output_root evidence inventory/hashes drifted from recovery certificate"
        )
    if actual_inventory["poll_count"] < 1:
        raise ResumeRefused("recovery certificate requires retained same-attempt poll evidence")
    poll_count, matched_polls = _check_output_root_fault_evidence(
        resolved_output,
        expected_job_id=job_id,
        expected_generation=generation,
    )
    if poll_count != actual_inventory["poll_count"] or matched_polls < 1:
        raise ResumeRefused(
            "retained poll files lack verified same-attempt job_id and generation bindings"
        )

    release = certificate.get("release_evidence")
    if not isinstance(release, Mapping):
        raise ResumeRefused("recovery certificate lacks release_evidence")
    capture_desc = release.get("account_status_capture")
    if not isinstance(capture_desc, Mapping) or not isinstance(capture_desc.get("path"), str):
        raise ResumeRefused("recovery certificate lacks account_status_capture")
    capture_path = Path(capture_desc["path"])
    capture_sidecar_desc = _descriptor(capture_path, runtime_path=capture_path.name)
    if (
        capture_desc.get("size") != capture_sidecar_desc["size"]
        or capture_desc.get("sha256") != capture_sidecar_desc["sha256"]
    ):
        raise ResumeRefused("account_status_capture size or SHA-256 differs from certificate")
    try:
        capture_payload = json.loads(capture_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeRefused("unreadable account_status_capture file") from error
    if (
        not isinstance(capture_payload, Mapping)
        or capture_payload.get("schema") != ACCOUNT_RELEASE_CAPTURE_SCHEMA
    ):
        raise ResumeRefused("malformed account_status_capture schema")
    captured_ts = capture_payload.get("captured_unix_seconds")
    if (
        isinstance(captured_ts, bool)
        or not isinstance(captured_ts, (int, float))
        or not math.isfinite(float(captured_ts))
        or capture_desc.get("captured_unix_seconds") != captured_ts
    ):
        raise ResumeRefused("malformed or mismatched captured_unix_seconds in release capture")
    log_mtime = sup_log_path.stat().st_mtime
    now_ts = time.time() if now_unix_seconds is None else float(now_unix_seconds)
    if float(captured_ts) < log_mtime - 1.0:
        raise ResumeRefused(
            "stale account_status_capture predates supervisor_log ENOSPC termination"
        )
    if float(captured_ts) > now_ts + 60.0:
        raise ResumeRefused("account_status_capture timestamp lies in the future")
    if (
        max_capture_age_seconds is not None
        and (now_ts - float(captured_ts)) > max_capture_age_seconds
    ):
        raise ResumeRefused("stale account_status_capture exceeds maximum admission age")

    status_obj = capture_payload.get("status")
    if not isinstance(status_obj, Mapping):
        raise ResumeRefused("account_status_capture lacks status object")
    sessions_obj = status_obj.get("sessions")
    usage_obj = status_obj.get("usage")
    unresolved_obj = status_obj.get("unresolved")
    if (
        not isinstance(sessions_obj, Mapping)
        or len(sessions_obj) != 0
        or endpoint in sessions_obj
        or not isinstance(usage_obj, Mapping)
        or isinstance(usage_obj.get("assignments"), bool)
        or not isinstance(usage_obj.get("assignments"), (int, float))
        or float(usage_obj["assignments"]) != 0.0
        or isinstance(usage_obj.get("rate"), bool)
        or not isinstance(usage_obj.get("rate"), (int, float))
        or float(usage_obj["rate"]) != 0.0
        or isinstance(usage_obj.get("balance"), bool)
        or not isinstance(usage_obj.get("balance"), (int, float))
        or not math.isfinite(float(usage_obj["balance"]))
        or unresolved_obj != []
    ):
        raise ResumeRefused(
            "account_status_capture does not prove empty server sessions, 0 assignments, "
            "0.0 rate, and empty unresolved pending state"
        )

    broker_rec = release.get("broker_reconciliation")
    if not isinstance(broker_rec, Mapping) or not isinstance(broker_rec.get("account_root"), str):
        raise ResumeRefused("recovery certificate lacks broker_reconciliation")
    account_root = Path(broker_rec["account_root"])
    if account_root.is_symlink() or not account_root.is_dir():
        raise ResumeRefused(f"missing or symlinked broker account_root: {account_root}")
    pending_file = account_root / f"pending-{session}.json"
    if pending_file.exists() or pending_file.is_symlink():
        raise ResumeRefused(f"unresolved broker pending record exists: {pending_file}")
    active_file = account_root / f"active-{session}.json"
    rec_mode = broker_rec.get("mode")
    if rec_mode == "qualified_recover_receipt":
        if active_file.exists() or active_file.is_symlink():
            raise ResumeRefused(
                "qualified_recover_receipt mode requires stale active record to be absent"
            )
        rec_desc = broker_rec.get("recover_receipt")
        if not isinstance(rec_desc, Mapping) or not isinstance(rec_desc.get("path"), str):
            raise ResumeRefused("qualified_recover_receipt lacks recover_receipt descriptor")
        rec_path = Path(rec_desc["path"])
        _verified_regular_file(rec_path, rec_desc, label="recover_receipt")
        if rec_path.resolve().parent != resolved_output or not (
            rec_path.name.startswith("recovery-") and rec_path.suffix == ".json"
        ):
            raise ResumeRefused("recover_receipt must reside in output_root as recovery-*.json")
        if {
            path.resolve()
            for path in resolved_output.glob("recovery-*.json*")
            if not _publication_sidecar(path)
        } != {rec_path.resolve()}:
            raise ResumeRefused("unbound recovery-*.json* retained in output_root")
        try:
            rec_payload = json.loads(rec_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeRefused("unreadable recover_receipt") from error
        if (
            not isinstance(rec_payload, Mapping)
            or rec_payload.get("endpoint") != endpoint
            or rec_payload.get("stop_verified") is not True
        ):
            raise ResumeRefused("recover_receipt does not prove stop_verified for endpoint")
    else:
        raise ResumeRefused(f"unsupported broker_reconciliation mode: {rec_mode!r}")

    resume_binding = certificate.get("resume_binding")
    if not isinstance(resume_binding, Mapping):
        raise ResumeRefused("recovery certificate lacks resume_binding")
    verified_ledger, _uploads, latest = build_ledger(
        resolved_output / "exports/training",
        source_control,
    )
    verified_iters = [int(row["iteration"]) for row in verified_ledger["units"]]
    latest_export = verified_ledger["units"][-1]["exports"][0]
    if (
        resume_binding.get("verified_snapshot_iterations") != verified_iters
        or resume_binding.get("latest_eligible_iteration") != latest
        or resume_binding.get("latest_eligible_unit") != unit_id(latest)
        or resume_binding.get("latest_full_state_size") != latest_export["size"]
        or resume_binding.get("latest_full_state_sha256") != latest_export["sha256"]
    ):
        raise ResumeRefused("recovery certificate resume_binding disagrees with verified exports")

    return {
        "schema": HOST_ENOSPC_RECOVERY_SCHEMA,
        "certificate_path": str(resolved_cert),
        "certificate_sha256": cert_desc["sha256"],
        "certificate_size": cert_desc["size"],
        "failure_class": HOST_ENOSPC_FAILURE_CLASS,
        "prior_attempt_status": "INCOMPLETE",
        "original_receipt_present": False,
        "original_terminal_status": "UNKNOWN_UNAVAILABLE",
        "original_supervisor_error": "UNKNOWN_UNAVAILABLE",
        "original_cleanup_error": "UNKNOWN_UNAVAILABLE",
        "post_hoc_release_verified": True,
        "account_status_capture_sha256": capture_sidecar_desc["sha256"],
        "captured_at_utc": capture_payload.get("captured_at_utc"),
        "captured_unix_seconds": captured_ts,
        "broker_reconciliation_mode": rec_mode,
        "recover_receipt": dict(rec_desc),
        "operator_reconcile_command": broker_rec.get("operator_reconcile_command"),
        "stale_active_present_now": active_file.is_file(),
        "supervisor_log_sha256": sup_log["sha256"],
        "polls_scanned": poll_count,
        "polls_sha256": actual_inventory["polls_sha256"],
        "latest_eligible_iteration": latest,
        "latest_full_state_sha256": latest_export["sha256"],
    }


def completion_packet(
    original_manifest: Mapping[str, Any],
    terminal_receipt: Mapping[str, Any] | None,
    export_root: Path,
    control: Mapping[str, Any],
    *,
    completion_index: int,
    job_id: str,
    output_dir: Path,
    original_manifest_sha256: str,
    successor_manifest: Mapping[str, Any] | None = None,
    successor_control: Mapping[str, Any] | None = None,
    compact_source_packet_id: str | None = None,
    allowlist_path: Path | None = None,
    output_root: Path | None = None,
    supervisor_log_path: Path | None = None,
    recovery_evidence_path: Path | None = None,
    source_packet_id: str | None = None,
    source_packet_manifest_sha256: str | None = None,
    max_recovery_capture_age_seconds: float | None = 86400.0,
) -> ResumePacket:
    """Return a fresh, manually dispatchable successor manifest and its inputs.

    This does not write or dispatch anything.  The caller creates the ledger
    once, adds its digest as an input, then writes the returned manifest once.
    """
    if completion_index not in (1, 2):
        raise ResumeRefused("only two manual outcome-blind completion attempts are registered")
    if (terminal_receipt is None) == (recovery_evidence_path is None):
        raise ResumeRefused(
            "completion_packet requires exclusively terminal_receipt or recovery_evidence_path"
        )
    inferred_output_root = output_root
    if (
        inferred_output_root is None
        and export_root.name == "training"
        and export_root.parent.name == "exports"
    ):
        inferred_output_root = export_root.parents[1]
    if recovery_evidence_path is not None:
        if inferred_output_root is None:
            raise ResumeRefused("host ENOSPC recovery requires output_root")
        verify_host_enospc_recovery(
            recovery_evidence_path,
            original_manifest,
            original_manifest_sha256,
            output_root=inferred_output_root,
            source_control=control,
            source_packet_id=source_packet_id,
            source_packet_manifest_sha256=source_packet_manifest_sha256,
            max_capture_age_seconds=max_recovery_capture_age_seconds,
        )
    else:
        assert terminal_receipt is not None
        verify_mechanical_stop(
            terminal_receipt,
            original_manifest,
            original_manifest_sha256,
            output_root=inferred_output_root,
            supervisor_log_path=supervisor_log_path,
        )
    if not job_id or job_id == original_manifest.get("job_id"):
        raise ResumeRefused("completion attempt requires a fresh job id")
    if str(output_dir) == original_manifest.get("output_dir") or not output_dir.is_absolute():
        raise ResumeRefused("completion attempt requires a fresh absolute output directory")
    target_control = dict(control) if successor_control is None else dict(successor_control)
    target_manifest = (
        dict(original_manifest) if successor_manifest is None else dict(successor_manifest)
    )
    if target_control != dict(control):
        source_migration(dict(control), target_control, allowlist_path=allowlist_path)
    if target_manifest.get("commit") != target_control.get("source_commit"):
        raise ResumeRefused("successor manifest source pin differs from successor control")
    ledger, records, latest = build_ledger(
        export_root,
        target_control,
        source_control=None if target_control == dict(control) else dict(control),
        allowlist_path=allowlist_path,
    )
    if latest == target_control.get("outer_iterations"):
        raise ResumeRefused("latest verified snapshot already reaches the frozen endpoint")
    successor = copy.deepcopy(target_manifest)
    successor["job_id"] = job_id
    successor["output_dir"] = str(output_dir)
    successor["completion_attempt"] = {
        "index": completion_index,
        "manual": True,
        "prior_job_id": original_manifest["job_id"],
        "resume_from_iteration": latest,
    }
    # A successor resumes only the latest eligible full trajectory.  Strip all
    # prior runtime inputs here too, because callers may use this function
    # directly rather than the create-only packet writer below.
    latest_unit = ledger["units"][-1]
    ledger["units"] = [latest_unit]
    destinations = {row["path"] for row in latest_unit["exports"]}
    destinations |= {
        name + suffix for name in list(destinations) for suffix in (".sha256", ".complete")
    }
    state_runtime_path = latest_unit["exports"][0]["path"]
    destinations |= {
        state_runtime_path + COMPACT_DERIVATIVE_SUFFIX,
        state_runtime_path + COMPACT_MANIFEST_SUFFIX,
    }
    records = [row for row in records if row["dest"] in destinations]
    if compact_source_packet_id is not None:
        ledger, records = attach_compact_resume(
            ledger,
            records,
            export_root,
            source_packet_id=compact_source_packet_id,
            allowlist_path=allowlist_path,
        )
    successor["inputs"] = [
        row
        for row in copy.deepcopy(target_manifest.get("inputs", []))
        if row["dest"] != "verified-ledger.json" and not row["dest"].startswith("resume/")
    ] + records
    return ResumePacket(
        ledger=ledger, input_records=records, manifest=successor, latest_iteration=latest
    )
