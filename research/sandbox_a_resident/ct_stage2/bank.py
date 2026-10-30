"""Verify the immutable supported H5000 input before downstream computation."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.sandbox_a_resident.ct_h5000.task import CollectedCTDataset, CTControl
from research.sandbox_a_resident.ct_h5000.units import allocation as collection_allocation
from research.sandbox_a_resident.ct_snapshots import LosslessBoundedSnapshotStore
from research.sandbox_a_resident.ct_stage2.units import complete_file
from research.sandbox_a_resident.ct_stats import validate_ct_support
from research.sandbox_a_resident.ct_task import AnchorArchive
from research.sandbox_a_resident.units import digest

BANK_SHA256 = "112a2aa4c369ecd0e0f735db410e9ac5a7d330aca09846a50d1a734e6b3175f0"
BANK_BYTES = 126_341_023
COLLECTION_PIN = "9fdc4cb655431686a60c6a548a03d1b9a05b1bd5"
BANK_NAME = "stage-0001"


def describe_bank(root: Path) -> dict[str, Any]:
    """Check receipt + saved manifest + summary + complete bank, read-only."""
    output = root.parents[1]
    receipt_path, manifest_path = output / "receipt.json", output / "manifest.json"
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error") is not None
        or receipt.get("commit") != COLLECTION_PIN
        or receipt.get("job_id") != manifest.get("job_id")
        or receipt.get("manifest_sha256") != digest(manifest_path)
    ):
        raise ValueError("collection receipt/manifest does not prove completed released input")
    ctrl = CTControl.from_dict(json.loads(manifest["env"]["SANDBOX_CT_CONTROL_JSON"]))
    if ctrl != CTControl(source_commit=COLLECTION_PIN):
        raise ValueError("foreign collection recipe")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.endswith((".sha256", ".complete")):
            # progress.json is live mutable telemetry, not a scientific export.
            if path.name == "progress.json":
                continue
            if Path(str(path) + ".complete").exists():
                files[path.relative_to(root).as_posix()] = complete_file(path)
    payload = complete_file(root / f"{BANK_NAME}.pt")
    header = complete_file(root / f"{BANK_NAME}.json")
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("verdict") != "SUPPORT_COMPLETE":
        raise ValueError("collection did not supply complete support")
    report = {
        "schema": "ct-stage2-bank/v1",
        "payload": payload,
        "header": header,
        "source_control": ctrl.to_dict(),
        "allocation": collection_allocation(ctrl.to_dict()),
        "support_report": summary["support_report"],
        "summary_sha256": digest(root / "summary.json"),
        "receipt_sha256": digest(receipt_path),
        "manifest_sha256": digest(manifest_path),
        "receipt_job_id": receipt["job_id"],
        "verified_files": files,
    }
    load_bank(root, report)
    return report


def load_bank(root: Path, report: dict[str, Any]) -> CollectedCTDataset:
    """Admit only this complete bank, then independently repeat the support check."""
    if report["payload"] != {"sha256": BANK_SHA256, "size": BANK_BYTES}:
        raise ValueError("foreign supported bank identity")
    complete_file(root / f"{BANK_NAME}.pt", report["payload"])
    complete_file(root / f"{BANK_NAME}.json", report["header"])
    header = json.loads((root / f"{BANK_NAME}.json").read_text())
    ctrl = CTControl.from_dict(report["source_control"])
    if (
        ctrl != CTControl(source_commit=COLLECTION_PIN)
        or collection_allocation(ctrl.to_dict()) != report["allocation"]
        or header["id"] != "a100-dev/t34-ct-h5000-1029/stages/stage-0001"
        or header["metadata"]["allocation"] != report["allocation"]
        or header["metadata"]["source_commit"] != COLLECTION_PIN
        or header["snapshot"]["size"] != BANK_BYTES
        or header["snapshot"]["sha256"] != BANK_SHA256
    ):
        raise ValueError("foreign collection header/control")
    blob = torch.load(root / f"{BANK_NAME}.pt", map_location="cpu", weights_only=False)
    if blob["schema"] != "ct-stage-0001/v1":
        raise ValueError("foreign collection payload schema")
    dataset = CollectedCTDataset.from_state_dict(blob["dataset"])
    actual = validate_ct_support(
        control=ctrl,
        store=dataset.store,
        anchors=dataset.anchors,
        world_partitions=dataset.world_partitions,
    )
    if (
        actual != blob["support_report"]
        or actual != report["support_report"]
        or actual["status"] != "PASS"
        or len(dataset.store) != 2560
        or dataset.anchors.num_rows != 50_000
        or header["metadata"]["roots"] != 2560
        or header["metadata"]["anchors"] != 50_000
        or dataset.collection_telemetry.get("completed_ticks") != 5000
    ):
        raise ValueError("bank payload, summary and independent support audit disagree")
    return dataset


def small_real_slice(dataset: CollectedCTDataset) -> CollectedCTDataset:
    """Four real roots + sixteen real anchors for plumbing only, no scientific look."""
    store = dataset.store
    indices = []
    for partition in ("train", "assessment"):
        for kind in ("death_lookback", "ordinary"):
            indices.append(
                next(
                    i
                    for i, m in enumerate(store.metadata)
                    if m.partition == partition and m.kind == kind
                )
            )
    slim = LosslessBoundedSnapshotStore(
        max_roots=4,
        max_bytes=store.max_bytes,
        max_body_capacity=store.max_body_capacity,
        grid_h=store.grid_h,
        grid_w=store.grid_w,
    )
    obs = store.stacked_observations()
    slim.snapshots = [store.snapshots[i] for i in indices]
    slim.metadata = [replace(store.metadata[i], root_index=j) for j, i in enumerate(indices)]
    slim._uid_set = {m.root_uid for m in slim.metadata}
    for attr, key in (
        ("_locals", "local"),
        ("_globs", "glob"),
        ("_scalars", "scalars"),
        ("_legal_masks", "legal_mask"),
        ("_q_b8", "q_b8"),
    ):
        setattr(slim, attr, [obs[key][i].clone() for i in indices])
    slim._finalized_topology = store.finalize_topology().slice_rows(indices)
    anchors = dataset.anchors
    ai = np.concatenate((anchors.train_indices()[:8], anchors.assessment_indices()[:8]))
    values = {}
    for field in fields(AnchorArchive):
        value = getattr(anchors, field.name)
        values[field.name] = (
            len(ai)
            if field.name == "num_rows"
            else (
                value.slice_rows(ai)
                if field.name == "topology"
                else value[ai].copy() if isinstance(value, np.ndarray) else value[ai].clone()
            )
        )
    return replace(dataset, store=slim, anchors=AnchorArchive(**values))
