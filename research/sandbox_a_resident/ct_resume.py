"""Mac-side outcome-blind continuation builder with scientific-fault precedence for C/T.

Implements Ruling §4 & §5 and JUDGMENT_RESUME_POLICY.md:
- Torch-free Mac-side verification of completed ``stage-XXXX`` triples by path,
  size, and SHA-256 sidecars only (never opening numerical ``.pt`` payloads).
- Scientific-fault precedence: refuses continuation if ``summary.json``,
  ``fault.json``, non-empty ``faulthandler-stacks.log``, terminal non-timeout
  markers, or terminal ``INCOMPLETE_SUPPORT`` / ``INVALID`` stage metadata exist,
  even when the transport receipt reports ``RPC_EXEC_POLL_TIMEOUT``.
- Supports ledger-bound compact ``.pt.xz`` + ``.pt.xz.manifest`` uploads for any
  stage snapshot without requiring a new source commit.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.sandbox_a_resident.ct_units import (
    COMPACT_DERIVATIVE_SUFFIX,
    COMPACT_MANIFEST_SUFFIX,
    COMPACT_RESUME_FIELD,
    COMPACT_RESUME_SCHEMA,
    LEDGER_SCHEMA,
    NAMESPACE,
    SNAPSHOT_SCHEMA,
    allocation,
    unit_id,
    validate_compact_resume,
)
from research.sandbox_a_resident.units import digest

RUNTIME_ROOT = "resume"
MECHANICAL_STOP_CLASSES: frozenset[str] = frozenset(
    {
        "RPC_EXEC_POLL_TIMEOUT",
        "WALL_CLOCK_TIMEOUT",
        "SESSION_LOST",
        "SIGUSR1_PREEMPTED",
    }
)


class CTResumeRefused(ValueError):
    """Raised when prior evidence is ineligible for outcome-blind continuation."""


@dataclass(frozen=True)
class CTResumePacket:
    """Create-only components for a manual outcome-blind C/T completion attempt."""

    ledger: dict[str, Any]
    input_records: list[dict[str, str]]
    manifest: dict[str, Any]
    verified_stages: tuple[int, ...]


def _complete(path: Path) -> bool:
    return all(
        p.is_file() for p in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    )


def _descriptor(path: Path, *, runtime_path: str) -> dict[str, Any]:
    sidecar = Path(str(path) + ".sha256")
    complete = Path(str(path) + ".complete")
    if not _complete(path):
        raise CTResumeRefused(f"incomplete export: {path.name}")
    if any(p.is_symlink() for p in (path, sidecar, complete)):
        raise CTResumeRefused(f"symlinked export refused: {path.name}")
    expected = sidecar.read_text().strip().split()[0]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise CTResumeRefused(f"malformed SHA-256 sidecar: {path.name}")
    if digest(path) != expected:
        raise CTResumeRefused(f"corrupt export: {path.name}")
    return {"path": runtime_path, "size": path.stat().st_size, "sha256": expected}


def assert_no_scientific_or_fault_precedence(export_root: Path, receipt: Mapping[str, Any]) -> None:
    """Refuse mechanical continuation if any scientific outcome or fault exists.

    Scientific outcomes (``summary.json``, ``INCOMPLETE_SUPPORT``, ``INVALID``)
    and runtime/watchdog faults (``fault.json``, non-empty ``faulthandler-stacks.log``)
    take strict precedence over generic ``RPC_EXEC_POLL_TIMEOUT`` or timeout receipts.
    """
    search_roots = [export_root]
    if export_root.parent != export_root:
        search_roots.append(export_root.parent)

    for base in search_roots:
        if not base.exists():
            continue
        for summary_rel in ("summary.json", "summary.json.partial", "ct/summary.json"):
            cand = base / summary_rel
            if cand.exists() or cand.is_symlink():
                raise CTResumeRefused(
                    f"scientific summary already present ({cand.name}); continuation refused"
                )
        for fault_rel in ("fault.json", "fault.json.partial", "ct/fault.json"):
            cand = base / fault_rel
            if cand.exists() or cand.is_symlink():
                raise CTResumeRefused(
                    f"scientific/runtime fault diagnostic present ({cand.name}); "
                    "continuation refused"
                )
        for stack_rel in (
            "stall-watchdog/faulthandler-stacks.log",
            "ct/stall-watchdog/faulthandler-stacks.log",
        ):
            stack_path = base / stack_rel
            if stack_path.is_file() and stack_path.stat().st_size > 0:
                raise CTResumeRefused(
                    "stall-watchdog faulthandler stack dump is non-empty; continuation refused"
                )

    terminal = receipt.get("terminal")
    if isinstance(terminal, Mapping):
        t_status = str(terminal.get("status", "")).upper()
        t_reason = str(terminal.get("reason", "")).lower()
        if t_status in {
            "COMPLETED",
            "SUCCEEDED",
            "FAILED",
            "FAULT",
            "INVALID",
            "INCOMPLETE_SUPPORT",
        }:
            raise CTResumeRefused(f"terminal marker reports non-resumable status={t_status!r}")
        if any(tok in t_reason for tok in ("fault", "exception", "invalid", "incomplete_support")):
            raise CTResumeRefused(
                f"terminal marker reports scientific/runtime fault reason={t_reason!r}"
            )


def verify_mechanical_stop(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    *,
    export_root: Path,
) -> None:
    """Enforce scientific-fault precedence first, then verify clean mechanical stop."""
    assert_no_scientific_or_fault_precedence(export_root, receipt)
    if len(manifest_sha256) != 64:
        raise CTResumeRefused("original manifest_sha256 is malformed")
    if (
        receipt.get("job_id") != manifest.get("job_id")
        or receipt.get("commit") != manifest.get("commit")
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("status") != "INCOMPLETE"
        or receipt.get("failure_class") not in MECHANICAL_STOP_CLASSES
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
    ):
        raise CTResumeRefused("prior receipt does not prove a verified released mechanical stop")
    terminal = receipt.get("terminal")
    if terminal is not None and (
        not isinstance(terminal, Mapping)
        or terminal.get("job_id") != manifest.get("job_id")
        or terminal.get("generation") != receipt.get("generation")
    ):
        raise CTResumeRefused("terminal marker identity differs from receipt")


def _stage_row(
    export_root: Path, info_path: Path, control: Mapping[str, Any]
) -> tuple[int, dict[str, Any], list[dict[str, str]]] | None:
    if not _complete(info_path):
        return None
    _descriptor(info_path, runtime_path=f"{RUNTIME_ROOT}/{info_path.name}")
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise CTResumeRefused(f"unreadable stage header: {info_path.name}") from err

    stage = info.get("stage")
    identity = info.get("id")
    if type(stage) is not int or stage < 1 or identity != unit_id(stage):
        raise CTResumeRefused(f"foreign stage identity: {info_path.name}")
    if info.get("schema") != SNAPSHOT_SCHEMA or not isinstance(info.get("metadata"), dict):
        raise CTResumeRefused(f"malformed stage header: {info_path.name}")
    meta = info["metadata"]
    if meta.get("allocation") != allocation(control) or meta.get("source_commit") != control.get(
        "source_commit"
    ):
        raise CTResumeRefused(f"stage header differs from frozen control: {info_path.name}")
    if meta.get("scientific_fault") or meta.get("support_status") in {
        "INCOMPLETE_SUPPORT",
        "INVALID",
    }:
        raise CTResumeRefused(
            f"stage-{stage:04d} recorded terminal support/fault status "
            f"{meta.get('support_status') or meta.get('scientific_fault')}; continuation refused"
        )

    stem = f"stage-{stage:04d}"
    if info_path.name != f"{stem}.json":
        raise CTResumeRefused(f"noncanonical stage header filename: {info_path.name}")
    pt_path = export_root / f"{stem}.pt"
    if not _complete(pt_path):
        return None

    paths = [pt_path, info_path]
    runtime_paths = [f"{RUNTIME_ROOT}/{p.name}" for p in paths]
    descriptors = [_descriptor(p, runtime_path=dest) for p, dest in zip(paths, runtime_paths)]
    snap_meta = info.get("snapshot")
    if (
        not isinstance(snap_meta, dict)
        or snap_meta.get("id") != identity
        or snap_meta.get("size") != descriptors[0]["size"]
        or snap_meta.get("sha256") != descriptors[0]["sha256"]
    ):
        raise CTResumeRefused(f"stage header/payload descriptor mismatch: {info_path.name}")

    uploads: list[dict[str, str]] = []
    for p, dest, desc in zip(paths, runtime_paths, descriptors):
        uploads.extend(
            [
                {"path": str(p.resolve()), "dest": dest, "sha256": desc["sha256"]},
                {
                    "path": str(Path(str(p) + ".sha256").resolve()),
                    "dest": dest + ".sha256",
                    "sha256": digest(Path(str(p) + ".sha256")),
                },
                {
                    "path": str(Path(str(p) + ".complete").resolve()),
                    "dest": dest + ".complete",
                    "sha256": digest(Path(str(p) + ".complete")),
                },
            ]
        )
    return stage, {"id": identity, "stage": stage, "exports": descriptors}, uploads


def build_stage_ledger(
    export_root: Path, control: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]], tuple[int, ...]]:
    """Build verified stage ledger over all complete contiguous/available stages."""
    if export_root.is_symlink():
        raise CTResumeRefused("export_root must not be a symlink")
    rows: list[dict[str, Any]] = []
    uploads: list[dict[str, str]] = []
    for info_path in sorted(export_root.glob("stage-*.json")):
        cand = _stage_row(export_root, info_path, control)
        if cand is None:
            continue
        _stage, row, items = cand
        rows.append(row)
        uploads.extend(items)
    if not rows:
        raise CTResumeRefused("no complete verified stage snapshots found")
    rows.sort(key=lambda r: r["stage"])
    stages = tuple(int(r["stage"]) for r in rows)
    if len(set(stages)) != len(stages):
        raise CTResumeRefused("duplicate verified stage index")
    if 1 not in stages:
        raise CTResumeRefused("cannot resume scoring/fitting without verified stage-0001")
    total_stages = 2 + int(control.get("scoring_chunks", 8))
    if any(s < 1 or s > total_stages for s in stages):
        raise CTResumeRefused("stage index outside frozen C/T stage schedule")
    if total_stages in stages:
        raise CTResumeRefused("final stage already completed; continuation unnecessary")
    ledger = {
        "schema": LEDGER_SCHEMA,
        "namespace": NAMESPACE,
        "allocation": allocation(control),
        "units": rows,
    }
    return ledger, uploads, stages


def attach_compact_stage_resume(
    ledger: Mapping[str, Any],
    uploads: list[dict[str, str]],
    export_root: Path,
    *,
    stages_to_compact: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Replace selected stage ``.pt`` uploads with verified ``.pt.xz`` + ``.pt.xz.manifest``."""
    successor = copy.deepcopy(dict(ledger))
    units = successor.get("units", [])
    if not isinstance(units, list) or not units:
        raise CTResumeRefused("compact resume requires at least one verified stage unit")

    requested = (
        {int(s) for s in stages_to_compact}
        if stages_to_compact is not None
        else {
            int(u["stage"])
            for u in units
            if _complete(
                export_root / f"{Path(u['exports'][0]['path']).name}{COMPACT_DERIVATIVE_SUFFIX}"
            )
        }
    )
    if not requested:
        raise CTResumeRefused("no stages selected or encoded for compact resume")

    entries: list[dict[str, Any]] = []
    updated_uploads = list(uploads)
    for unit in units:
        if int(unit["stage"]) not in requested:
            continue
        orig = dict(unit["exports"][0])
        stem = Path(orig["path"]).name
        entry: dict[str, Any] = {
            "unit": unit["id"],
            "original": orig,
        }
        replacement_records: list[dict[str, str]] = []
        for role, suffix in (
            ("derivative", COMPACT_DERIVATIVE_SUFFIX),
            ("manifest", COMPACT_MANIFEST_SUFFIX),
        ):
            p = export_root / f"{stem}{suffix}"
            dest = f"{orig['path']}{suffix}"
            entry[role] = _descriptor(p, runtime_path=dest)
            replacement_records.append(
                {"path": str(p.resolve()), "dest": dest, "sha256": entry[role]["sha256"]}
            )
        entries.append(entry)
        positions = [i for i, rec in enumerate(updated_uploads) if rec["dest"] == orig["path"]]
        if len(positions) != 1:
            raise CTResumeRefused(f"missing or duplicate upload record for {orig['path']}")
        pos = positions[0]
        updated_uploads = updated_uploads[:pos] + replacement_records + updated_uploads[pos + 1 :]

    successor[COMPACT_RESUME_FIELD] = {
        "schema": COMPACT_RESUME_SCHEMA,
        "entries": entries,
    }
    validate_compact_resume(successor)
    return successor, updated_uploads


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
    use_compact_xz_for_stages: Sequence[int] | None = None,
) -> CTResumePacket:
    """Build a fresh manual outcome-blind completion attempt packet for C/T."""
    if completion_index not in (1, 2):
        raise CTResumeRefused("only two manual outcome-blind completion attempts are permitted")
    verify_mechanical_stop(
        terminal_receipt,
        original_manifest,
        original_manifest_sha256,
        export_root=export_root,
    )
    if not job_id or job_id == original_manifest.get("job_id"):
        raise CTResumeRefused("completion attempt requires a fresh job_id")
    if not output_dir.is_absolute() or str(output_dir) == original_manifest.get("output_dir"):
        raise CTResumeRefused("completion attempt requires a fresh absolute output_dir")
    if original_manifest.get("commit") != control.get("source_commit"):
        raise CTResumeRefused("manifest commit differs from frozen control source_commit")

    ledger, records, stages = build_stage_ledger(export_root, control)
    if use_compact_xz_for_stages is not None:
        ledger, records = attach_compact_stage_resume(
            ledger, records, export_root, stages_to_compact=use_compact_xz_for_stages
        )

    successor = copy.deepcopy(dict(original_manifest))
    successor["job_id"] = job_id
    successor["output_dir"] = str(output_dir)
    successor["completion_attempt"] = {
        "index": int(completion_index),
        "manual": True,
        "prior_job_id": str(original_manifest["job_id"]),
        "verified_stages": list(stages),
    }
    successor["inputs"] = [
        row
        for row in copy.deepcopy(list(original_manifest.get("inputs", [])))
        if row["dest"] != "verified-ledger.json" and not str(row["dest"]).startswith("resume/")
    ] + records
    return CTResumePacket(
        ledger=ledger,
        input_records=records,
        manifest=successor,
        verified_stages=stages,
    )
