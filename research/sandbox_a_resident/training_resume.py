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

from research.sandbox_a_resident.training_units import NAMESPACE, allocation, unit_id
from research.sandbox_a_resident.units import digest

LEDGER_SCHEMA = "sandbox-a-train-ledger/v1"
SNAPSHOT_SCHEMA = "sandbox-a-train-snapshot/v1"
RUNTIME_ROOT = "resume"


class ResumeRefused(ValueError):
    """The immutable evidence is not eligible for an outcome-blind continuation."""


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
    if not _phase_is_resumable(info["metadata"]):
        raise ResumeRefused(f"snapshot phase is not a safe continuation: {info_path.name}")
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
    export_root: Path, control: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]], int]:
    """Build a relative-path ledger from complete canonical snapshot triples.

    Incomplete snapshots are ignored (completion-last); a malformed *complete*
    snapshot fails closed.  The latest eligible trajectory is selected by the
    runner after it re-verifies the same copied input bytes.
    """
    rows: list[dict[str, Any]] = []
    uploads: list[dict[str, str]] = []
    if export_root.is_symlink():
        raise ResumeRefused("snapshot export root must not be a symlink")
    for info_path in sorted(export_root.glob("iteration-*.json")):
        candidate = _snapshot_row(export_root, info_path, control)
        if candidate is None:
            continue
        iteration, row, items = candidate
        rows.append(row)
        uploads.extend(items)
    if not rows:
        raise ResumeRefused("no complete full-trajectory snapshot triples")
    rows.sort(key=lambda item: item["iteration"])
    if len({item["iteration"] for item in rows}) != len(rows):
        raise ResumeRefused("duplicate snapshot iteration")
    endpoint, cadence = control.get("outer_iterations"), control.get("snapshot_interval")
    if cadence is not None:
        if type(endpoint) is not int or endpoint < 1 or type(cadence) is not int or cadence < 1:
            raise ResumeRefused("malformed frozen snapshot cadence")
        permitted = {0, endpoint, *range(cadence, endpoint, cadence)}
        if any(row["iteration"] not in permitted for row in rows):
            raise ResumeRefused("snapshot lies outside the frozen boundary cadence")
    latest = rows[-1]["iteration"]
    return (
        {
            "schema": LEDGER_SCHEMA,
            "namespace": NAMESPACE,
            "allocation": allocation(dict(control)),
            "units": rows,
        },
        uploads,
        latest,
    )


def verify_mechanical_stop(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any], manifest_sha256: str
) -> None:
    """Require an exact, released mechanical stop before a manual successor."""
    terminal = receipt.get("terminal")
    if len(manifest_sha256) != 64:
        raise ResumeRefused("original manifest digest is malformed")
    if (
        receipt.get("job_id") != manifest.get("job_id")
        or receipt.get("commit") != manifest.get("commit")
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("status") != "INCOMPLETE"
        or receipt.get("failure_class") != "RPC_EXEC_POLL_TIMEOUT"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
    ):
        raise ResumeRefused("prior receipt does not prove the frozen released RPC-poll trigger")
    if terminal is not None and (
        not isinstance(terminal, Mapping)
        or terminal.get("job_id") != manifest.get("job_id")
        or terminal.get("generation") != receipt.get("generation")
    ):
        raise ResumeRefused("terminal identity differs from the incomplete receipt")


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
    ledger, records, latest = build_ledger(export_root, control)
    if latest == control.get("outer_iterations"):
        raise ResumeRefused("latest verified snapshot already reaches the frozen endpoint")
    successor = copy.deepcopy(dict(original_manifest))
    successor["job_id"] = job_id
    successor["output_dir"] = str(output_dir)
    successor["completion_attempt"] = {
        "index": completion_index,
        "manual": True,
        "prior_job_id": original_manifest["job_id"],
        "resume_from_iteration": latest,
    }
    # The preparation layer appends the create-only ledger record after it
    # serializes this payload and hashes it.  All numerical exports are already
    # exact inputs, with runtime-relative ledger paths under COLAB_INPUT_DIR.
    successor["inputs"] = [*copy.deepcopy(original_manifest.get("inputs", [])), *records]
    return ResumePacket(
        ledger=ledger, input_records=records, manifest=successor, latest_iteration=latest
    )
