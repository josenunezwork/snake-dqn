"""Completion-last downstream stage units, separate from the collection ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from research.sandbox_a_resident.ct_units import allocation, publish_bounded_export
from research.sandbox_a_resident.units import digest

STUDY = "t34-ct-h5000-stage2-1030"
NAMESPACE = f"a100-dev/{STUDY}/stages"
STAGES = 10  # eight scoring chunks, all-three-arm predictions, registered evaluation
TAIL_BYTES = 12 * 1024**2


def empty_ledger(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the immutable bank, science, code pin and fixture flag to this ledger."""
    return {
        "schema": "ct-stage2-ledger/v1",
        "namespace": NAMESPACE,
        "allocation": allocation(config),
        "units": [],
    }


def complete_file(path: Path, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Verify exact payload bytes and completion-last sidecars before opening content."""
    sha, marker = Path(str(path) + ".sha256"), Path(str(path) + ".complete")
    for candidate in (path, sha, marker):
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("missing or symlinked immutable file: " + str(candidate))
    actual = {"size": path.stat().st_size, "sha256": digest(path)}
    words = sha.read_text().split()
    if not words or words[0] != actual["sha256"] or marker.read_bytes() != b"complete\n":
        raise ValueError("corrupt hash/completion sidecar: " + str(path))
    if expected is not None and any(actual[k] != expected[k] for k in actual):
        raise ValueError("immutable descriptor mismatch: " + str(path))
    return actual


def publish_stage(
    output: Path, stage: int, writer: Callable[[Path], None], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish payload before its header; both are atomic and recursively bounded."""
    if type(stage) is not int or not 1 <= stage <= STAGES:
        raise ValueError("foreign downstream stage")
    ctrl = config["science"]
    payload = output / f"stage-{stage:04d}.pt"

    def bounded_writer(path: Path) -> None:
        writer(path)
        if path.stat().st_size > ctrl["max_stage_export_bytes"]:
            raise ValueError("stage payload exceeds frozen stage cap")

    desc = publish_bounded_export(
        payload,
        bounded_writer,
        export_root=output,
        max_total_bytes=ctrl["max_total_export_bytes"],
        reserved_tail_bytes=TAIL_BYTES,
    )
    header = {
        "schema": "ct-stage2-snapshot/v1",
        "id": f"{NAMESPACE}/stage-{stage:04d}",
        "stage": stage,
        "allocation": allocation(config),
        "source_commit": config["source_commit"],
        "bank_sha256": config["bank"]["payload"]["sha256"],
        "payload": {"path": payload.name, "size": desc["size"], "sha256": desc["sha256"]},
    }
    publish_bounded_export(
        output / f"stage-{stage:04d}.json",
        lambda p: p.write_text(json.dumps(header, sort_keys=True) + "\n"),
        export_root=output,
        max_total_bytes=ctrl["max_total_export_bytes"],
        reserved_tail_bytes=TAIL_BYTES,
    )
    return header


def load_ledger(path: Path, config: Mapping[str, Any], input_root: Path) -> dict[int, Path]:
    """Skip exactly verified units; refuse foreign identities, paths and stage gaps."""
    ledger = json.loads(path.read_text())
    if {k: v for k, v in ledger.items() if k != "units"} != {
        k: v for k, v in empty_ledger(config).items() if k != "units"
    } or not isinstance(ledger.get("units"), list):
        raise ValueError("foreign downstream ledger namespace/allocation")
    verified: dict[int, Path] = {}
    for row in ledger["units"]:
        stage = row.get("stage")
        if type(stage) is not int or not 1 <= stage <= STAGES or stage in verified:
            raise ValueError("duplicate or foreign downstream stage")
        if row.get("id") != f"{NAMESPACE}/stage-{stage:04d}":
            raise ValueError("foreign downstream unit identity")
        paths = {}
        for role, suffix in (("payload", "pt"), ("header", "json")):
            desc = row[role]
            raw = Path(desc["path"])
            if raw.as_posix() != f"resume/stage-{stage:04d}.{suffix}":
                raise ValueError("noncanonical resume input path")
            target = input_root / raw
            if not target.resolve().is_relative_to(input_root.resolve()):
                raise ValueError("resume path escapes input root")
            complete_file(target, desc)
            paths[role] = target
        header = json.loads(paths["header"].read_text())
        if header != {
            "schema": "ct-stage2-snapshot/v1",
            "id": row["id"],
            "stage": stage,
            "allocation": allocation(config),
            "source_commit": config["source_commit"],
            "bank_sha256": config["bank"]["payload"]["sha256"],
            "payload": {
                "path": f"stage-{stage:04d}.pt",
                **{k: row["payload"][k] for k in ("size", "sha256")},
            },
        }:
            raise ValueError("stage header disagrees with ledger/bank/control")
        verified[stage] = paths["payload"]
    if sorted(verified) != list(range(1, len(verified) + 1)):
        raise ValueError("resume stage coverage has gaps")
    return verified


def build_ledger(config: Mapping[str, Any], roots: list[Path]) -> dict[str, Any]:
    """Mac path/size/hash-only ledger union; never deserialize outcome tensors."""
    ledger = empty_ledger(config)
    by_stage = {}
    for root in roots:
        for header_path in sorted(root.glob("stage-*.json")):
            # An interrupted publication is not a completed unit.
            if not Path(str(header_path) + ".complete").exists():
                continue
            header_desc = complete_file(header_path)
            header = json.loads(header_path.read_text())
            stage = header["stage"]
            payload_path = root / f"stage-{stage:04d}.pt"
            payload_desc = complete_file(payload_path, header["payload"])
            row = {
                "id": header["id"],
                "stage": stage,
                "payload": {"path": f"resume/{payload_path.name}", **payload_desc},
                "header": {"path": f"resume/{header_path.name}", **header_desc},
            }
            if stage in by_stage and by_stage[stage] != row:
                raise ValueError("conflicting prior verified coverage")
            by_stage[stage] = row
    ledger["units"] = [by_stage[s] for s in sorted(by_stage)]
    return ledger
