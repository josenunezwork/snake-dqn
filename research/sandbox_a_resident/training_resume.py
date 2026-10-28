"""Mac-side, outcome-blind resume packet construction for TRAIN-01.

This module deliberately does not import torch.  The Mac verifies immutable
payload bytes through paths, sizes and SHA-256 sidecars; it only opens the JSON
envelope that says which completed full-state snapshot a payload belongs to.
"""

from __future__ import annotations

import copy
import json
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
    paths = [root / f"{stem}.pt", info_path, root / f"{stem}-actor.pth"]
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
    uploads: list[dict[str, str]] = []
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
    kept = [row for row in uploads if row["dest"] != original["path"]]
    if len(kept) != len(uploads) - 1:
        raise ResumeRefused("full-state upload record is missing or duplicated")
    position = next(index for index, row in enumerate(uploads) if row["dest"] == original["path"])
    return successor, kept[:position] + records + kept[position:]


def verify_mechanical_stop(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any], manifest_sha256: str
) -> None:
    """Require an exact, released mechanical stop before a manual successor.

    Admits only mechanical timeout/recycle/poll terminations with verified VM
    release and clean cleanup:
    - ``RPC_EXEC_POLL_TIMEOUT`` (with ``terminal`` absent or bearing a
      mechanical timeout/lease reason),
    - ``SUPERVISOR`` when ``terminal.reason`` is a mechanical timeout/lease
      reason or ``terminal`` is absent and ``error`` names a host wall/unit/lease
      cap or read-retry exhaustion (never ``KeyboardInterrupt`` or missing/corrupt
      exports),
    - ``CHILD_NONZERO_EXIT`` only when ``terminal.reason`` explicitly records a
      mechanical VM timeout/lease/credential stop (refusing scientific tripwires,
      ``StallWatchdog`` ``os._exit(1)``, and ordinary crashes where ``reason`` is
      ``None``).
    """
    terminal = receipt.get("terminal")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in manifest_sha256)
    ):
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
    terminal_reason: Any = None
    if terminal is not None:
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("job_id") != manifest.get("job_id")
            or terminal.get("generation") != receipt.get("generation")
        ):
            raise ResumeRefused("terminal identity differs from the incomplete receipt")
        if terminal.get("status") == "completed":
            raise ResumeRefused("prior terminal already completed; refusing continuation")
        terminal_reason = terminal.get("reason")
        if (
            terminal.get("returncode") not in (None, 0)
            and terminal_reason not in MECHANICAL_TERMINAL_REASONS
        ):
            raise ResumeRefused("terminal records a non-mechanical child failure")
    failure_class = receipt.get("failure_class")
    if failure_class == "RPC_EXEC_POLL_TIMEOUT":
        if terminal is not None and terminal_reason not in MECHANICAL_TERMINAL_REASONS:
            raise ResumeRefused("RPC poll timeout receipt carries a non-mechanical terminal record")
        return
    if failure_class == "CHILD_NONZERO_EXIT":
        if isinstance(terminal, Mapping) and terminal_reason in MECHANICAL_TERMINAL_REASONS:
            return
        raise ResumeRefused(
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
        raise ResumeRefused(
            "SUPERVISOR receipt does not prove a mechanical wall/unit/lease/poll stop"
        )
    raise ResumeRefused(
        f"prior receipt failure_class {failure_class!r} is not an eligible mechanical trigger"
    )


def completion_packet(
    original_manifest: Mapping[str, Any],
    terminal_receipt: Mapping[str, Any],
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
) -> ResumePacket:
    """Return a fresh, manually dispatchable successor manifest and its inputs.

    This does not write or dispatch anything.  The caller creates the ledger
    once, adds its digest as an input, then writes the returned manifest once.
    """
    if completion_index not in (1, 2):
        raise ResumeRefused("only two manual outcome-blind completion attempts are registered")
    verify_mechanical_stop(terminal_receipt, original_manifest, original_manifest_sha256)
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
