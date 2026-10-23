"""Mac held-out diagnostics of the actual immutable fused F3 checkpoint.

Raw held-out labels never enter the fit worker. This one-look entrypoint loads
them only after verifying final checkpoint completion. It emits no gameplay GO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual.data import allocate_lives, tensor_cache, validate_rows, world_fold
from research.bl_residual.fit import FitRecipe, heldout_metrics
from research.bl_residual.model import FrozenB8Residual
from research.bl_residual.protocol import B8_SHA, DOSES, allocation_hash
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
        or meta.get("recipe_sha256") != control.get("recipe_sha256")
        or control.get("input_hashes", {}).get("life-allocation.json") != digest(allocation_path)
    ):
        raise ValueError("foreign/synthetic/nonfinal candidate or changed held-out allocation")


def evaluate(
    checkpoint: Path,
    b8: Path,
    table_path: Path,
    *,
    allocation_path: Path,
    control_path: Path,
    archive_receipt: Path | None = None,
) -> dict:
    """Score a frozen candidate; archive completeness needs a verified census."""
    candidate_sha = verify_export(checkpoint)
    verify_export(control_path)
    control = json.loads(control_path.read_text())
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_candidate(blob.get("meta", {}), control, allocation_path)
    if digest(b8) != B8_SHA:
        raise ValueError("baseline not B8 s3")
    torch.set_num_threads(1)
    candidate = load_ego2s_checkpoint(str(checkpoint))
    base = load_ego2s_checkpoint(str(b8))
    with np.load(table_path, allow_pickle=False) as values:
        table = {k: values[k] for k in values.files}
    validate_rows(table)
    verify_export(allocation_path)
    allocation = json.loads(allocation_path.read_text())
    if allocate_lives(allocation["catalog"]) != allocation:
        raise ValueError("changed life allocator")
    heldout = set(allocation["doses"]["256"]["heldout_lives"])
    parents = {allocation["parents"][life] for life in heldout}
    present_doom_lives = set(table["life_id"][~table["ordinary"]].tolist())
    if (
        len(heldout) != 64
        or not present_doom_lives <= heldout
        or not heldout <= set(table["life_id"].tolist())
        or set(table["parent_world"].tolist()) != parents
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
        verify_export(archive_receipt)
        receipt = json.loads(archive_receipt.read_text())
        natural = table["natural_stream"]
        actual = sorted(table["row_id"][natural].tolist())
        if (
            receipt.get("schema") != "bl-residual-natural-archive/v1"
            or receipt.get("table_sha256") != digest(table_path)
            or receipt.get("complete") is not True
            or receipt.get("natural_row_ids") != actual
            or not actual
            or receipt.get("parent_worlds") != sorted(set(table["parent_world"].tolist()))
            or receipt.get("eligible_life_ids") != sorted(set(table["life_id"][natural].tolist()))
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
    result = heldout_metrics(
        FrozenB8Residual(base),
        cache,
        natural_archive_complete=archive_complete,
        candidate_q=torch.cat(chunks),
    )
    return {
        "schema": "bl-residual-heldout/v1",
        "checkpoint_sha256": candidate_sha,
        "table_sha256": digest(table_path),
        "natural_archive_complete": archive_complete,
        "archive_receipt_sha256": digest(archive_receipt) if archive_receipt else None,
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
            archive_receipt=args.archive_receipt,
        ),
    )


if __name__ == "__main__":
    main()
