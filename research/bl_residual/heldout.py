"""Mac held-out diagnostics of the actual immutable fused F3 checkpoint.

Raw held-out labels never enter the fit worker. This one-look entrypoint loads
them only after verifying final checkpoint completion. It emits no gameplay GO.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
from pathlib import Path

import numpy as np
import torch

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual.data import allocate_lives, tensor_cache, validate_rows, world_fold
from research.bl_residual.fit import FitRecipe, heldout_metrics
from research.bl_residual.model import FrozenB8Residual
from research.bl_residual.protocol import (
    B8_SHA,
    DOSES,
    allocation_hash,
    canonical_sha,
    require_judgment,
)
from src.model.ego2s_network import load_ego2s_checkpoint


def validate_candidate(meta: dict, control: dict, allocation_path: Path) -> None:
    """A canary or a foreign final must never open a real held-out outcome."""
    dose = meta.get("dose")
    if (
        meta.get("schema") != "bl-residual-fit/v1"
        or meta.get("synthetic") is not False
        or control.get("synthetic") is not False
        or dose not in DOSES
        or dose not in control.get("doses", [])
        or meta.get("index") != DOSES.index(dose)
        or meta.get("updates") != FitRecipe().updates
        or allocation_hash(control) != control.get("allocation")
        or meta.get("allocation") != control["allocation"]
        or meta.get("namespace") != control.get("namespace")
        or meta.get("recipe_sha256") != canonical_sha(dataclasses.asdict(FitRecipe()))
        or control.get("recipe_sha256") != canonical_sha(dataclasses.asdict(FitRecipe()))
        or control.get("input_hashes", {}).get("life-allocation.json") != digest(allocation_path)
    ):
        raise ValueError("foreign/synthetic/nonfinal candidate or changed held-out allocation")


def preflight_look(
    checkpoint: Path,
    allocation_path: Path,
    control_path: Path,
    approval_path: Path,
    archive_receipt: Path,
) -> dict:
    """Verify immutable R5 identities before any held-out table is opened."""
    if platform.system() != "Darwin":
        raise RuntimeError("R5 reserve looks are Mac-only")
    registration = Path(__file__).with_name("F4_REGISTRATION.md")
    require_judgment(approval_path, registration, canonical_sha(dataclasses.asdict(FitRecipe())))
    checkpoint_sha = verify_export(checkpoint)
    verify_export(control_path)
    control = json.loads(control_path.read_text())
    verify_export(allocation_path)
    verify_export(archive_receipt)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_candidate(blob.get("meta", {}), control, allocation_path)
    receipt = json.loads(archive_receipt.read_text())
    if (
        receipt.get("schema") != "bl-residual-natural-archive/v2"
        or receipt.get("complete") is not True
    ):
        raise ValueError("unsealed or foreign natural archive census")
    return {
        "dose": blob["meta"]["dose"],
        "checkpoint_sha256": checkpoint_sha,
        "control_sha256": digest(control_path),
        "allocation_sha256": digest(allocation_path),
        "archive_receipt_sha256": digest(archive_receipt),
        "registration_sha256": digest(registration),
    }


def evaluate(
    checkpoint: Path,
    b8: Path,
    table_path: Path,
    *,
    allocation_path: Path,
    control_path: Path,
    approval_path: Path,
    archive_receipt: Path | None = None,
    joint_claim_path: Path | None = None,
) -> dict:
    """Score a frozen candidate; archive completeness needs a verified census."""
    if archive_receipt is None:
        raise ValueError("R5 evaluation requires an immutable archive census")
    preflight = preflight_look(
        checkpoint, allocation_path, control_path, approval_path, archive_receipt
    )
    if preflight["dose"] in (256, 1024):
        if joint_claim_path is None or not joint_claim_path.exists():
            raise ValueError("D256/D1024 R-A outcomes may be opened only by the joint funding look")
        claim = json.loads(joint_claim_path.read_text())
        if claim.get("schema") != "bl-residual-r-a-lock/v1" or preflight not in (
            claim.get("d256"),
            claim.get("d1024"),
        ):
            raise ValueError("R-A claim does not bind this immutable fit")
        if str(joint_claim_path.parent.resolve()) != claim.get("lock_root"):
            raise ValueError("R-A claim is outside its frozen lock root")
        if joint_claim_path.name != f"F4_R_A_{canonical_sha(claim)}.lock.json":
            raise ValueError("R-A claim path is not canonical")
    candidate_sha = preflight["checkpoint_sha256"]
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    registration = Path(__file__).with_name("F4_REGISTRATION.md")
    if digest(b8) != B8_SHA:
        raise ValueError("baseline not B8 s3")
    torch.set_num_threads(1)
    candidate = load_ego2s_checkpoint(str(checkpoint))
    base = load_ego2s_checkpoint(str(b8))
    verify_export(allocation_path)
    allocation = json.loads(allocation_path.read_text())
    if allocate_lives(allocation["catalog"]) != allocation:
        raise ValueError("changed life allocator")
    heldout = set(allocation["doses"]["256"]["heldout_lives"])
    catalog = {row["life_id"]: row for row in allocation["catalog"]}
    reserve_max_index = max(catalog[life]["allocation_index"] for life in heldout)
    if preflight["dose"] == 4096:
        r_b = {
            "schema": "bl-residual-r-b-lock/v1",
            "preflight": preflight,
            "reserve_lives": sorted(heldout),
            "reserve_max_allocation_index": reserve_max_index,
        }
        lock_path = allocation_path.parent / f"F4_R_B_{canonical_sha(r_b)}.lock.json"
        try:
            with lock_path.open("x") as stream:
                stream.write(json.dumps(r_b, sort_keys=True) + "\n")
        except FileExistsError as exc:
            raise FileExistsError("R-B already opened for this canonical reserve look") from exc
    with np.load(table_path, allow_pickle=False) as values:
        table = {k: values[k] for k in values.files}
    validate_rows(table)
    if "sampling_prob" not in table:
        raise ValueError("R5 archive requires unclipped sampling_prob")
    archive_lives = {
        life
        for life, row in catalog.items()
        if world_fold(row["parent_world"]) == 0 and row["allocation_index"] <= reserve_max_index
    }
    archive_parents = {catalog[life]["parent_world"] for life in archive_lives}
    present_doom_lives = set(table["life_id"][~table["ordinary"]].tolist())
    if (
        len(heldout) != 64
        or not present_doom_lives <= heldout
        or not heldout <= set(table["life_id"].tolist())
        or not heldout <= set(table["life_id"].tolist())
        or not set(table["life_id"].tolist()) <= archive_lives
        or set(table["parent_world"].tolist()) != archive_parents
    ):
        raise ValueError("held-out reserve/ordinary archive ancestry mismatch")
    if any(
        allocation["parents"].get(life) != parent
        for life, parent in zip(table["life_id"], table["parent_world"])
    ):
        raise ValueError("held-out life crosses frozen parent allocation")
    if any(world_fold(str(parent)) != 0 for parent in table["parent_world"]):
        raise ValueError("non-heldout parent in diagnostic table")
    archive_complete = False
    if archive_receipt:
        receipt = json.loads(archive_receipt.read_text())
        natural = table["natural_stream"]
        actual = sorted(table["row_id"][natural].tolist())
        if (
            receipt.get("schema") != "bl-residual-natural-archive/v2"
            or receipt.get("table_sha256") != digest(table_path)
            or receipt.get("complete") is not True
            or receipt.get("natural_row_ids") != actual
            or not actual
            or receipt.get("parent_worlds") != sorted(archive_parents)
            or receipt.get("eligible_life_ids") != sorted(archive_lives)
            or receipt.get("reserve_max_allocation_index") != reserve_max_index
            or receipt.get("includes_non_rescue_lives") is not True
        ):
            raise ValueError("incomplete or foreign natural archive census")
        archive_complete = True
    cache = tensor_cache(table, base)
    chunks = []
    with torch.no_grad():
        # Match the evaluated rowwise forward, including float32 ties.
        for start in range(len(table["q"])):
            chunks.append(
                candidate(
                    *[
                        torch.as_tensor(table[k][start : start + 1])
                        for k in ("local", "global", "scalars")
                    ]
                )
            )
    life_codes = {life: index for index, life in enumerate(sorted(set(table["life_id"].tolist())))}
    result = heldout_metrics(
        FrozenB8Residual(base),
        cache,
        natural_archive_complete=archive_complete,
        candidate_q=torch.cat(chunks),
        reserve_life_ids=torch.as_tensor(sorted(life_codes[life] for life in heldout)),
    )
    for record in result["life_retention"]:
        record["life_id"] = sorted(life_codes)[record["life_id"]]
        record["parent_world"] = allocation["parents"][record["life_id"]]
    return {
        "schema": "bl-residual-heldout/v2",
        "checkpoint_sha256": candidate_sha,
        "dose": blob["meta"]["dose"],
        "namespace": blob["meta"]["namespace"],
        "recipe_sha256": blob["meta"]["recipe_sha256"],
        "registration_sha256": digest(registration),
        "table_sha256": digest(table_path),
        "natural_archive_complete": archive_complete,
        "archive_receipt_sha256": digest(archive_receipt) if archive_receipt else None,
        "allocation_sha256": digest(allocation_path),
        "reserve_allocation_sha256": canonical_sha(
            [
                {
                    "life_id": life,
                    "parent_world": catalog[life]["parent_world"],
                    "allocation_index": catalog[life]["allocation_index"],
                }
                for life in sorted(heldout)
            ]
        ),
        "reserve_life_ids": sorted(heldout),
        "archive_eligible_life_ids": sorted(archive_lives),
        "metrics": result,
        "gameplay_decision": "NOT_A_DECISION",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--archive-receipt", type=Path)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("one create-only held-out look")
    export_json(
        args.out,
        evaluate(
            args.checkpoint,
            args.b8,
            args.table,
            allocation_path=args.allocation,
            control_path=args.control,
            approval_path=args.approval,
            archive_receipt=args.archive_receipt,
        ),
    )


if __name__ == "__main__":
    main()
