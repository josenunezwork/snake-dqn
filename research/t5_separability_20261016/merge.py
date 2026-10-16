#!/usr/bin/env python3
"""Merge four independently restored T5 feature partitions without relabelling rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(parts: list[Path], output: Path) -> dict:
    """Verify exact partitions and emit the canonical features.npz plus receipt."""
    if len(parts) != 4 or len(set(parts)) != 4 or any(not part.is_file() for part in parts):
        raise ValueError("T5 merge requires four distinct completed feature parts")
    arrays = []
    receipts = []
    for part in parts:
        receipt_path = part.with_suffix(".json")
        if not receipt_path.is_file():
            raise ValueError(f"missing restoration receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt["output_sha256"] != sha256(part):
            raise ValueError("feature part differs from its extraction receipt")
        receipts.append(receipt)
        with np.load(part, allow_pickle=False) as archive:
            arrays.append({key: archive[key].copy() for key in archive.files})
    if {r["shard"]["index"] for r in receipts} != set(range(4)) or any(
        r["shard"]["count"] != 4 for r in receipts
    ):
        raise ValueError("invalid world partitions")
    inputs = [r["input_sha256"] for r in receipts]
    if any(item != inputs[0] for item in inputs[1:]):
        raise ValueError("partition bank input hashes differ")
    keys = set(arrays[0])
    if any(set(item) != keys for item in arrays[1:]):
        raise ValueError("feature parts have different field sets")
    sources = [receipt["source_sha256"] for receipt in receipts]
    if any(source != sources[0] for source in sources[1:]):
        raise ValueError("feature parts were restored with different source bytes")
    merged = {key: np.concatenate([item[key] for item in arrays]) for key in sorted(keys)}
    ids = merged["state_id"]
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("feature parts overlap state identities")
    legal_branches = int(merged["legal"].sum())
    train = merged["split"] == "train"
    decisive = merged["family"] == 0
    expected = {
        "states": 19976,
        "train_states": 18924,
        "holdout_states": 1052,
        "train_decisive_states": 9462,
        "train_outside_states": 9462,
        "branches": 103264,
    }
    actual = {
        "states": int(len(ids)),
        "train_states": int(train.sum()),
        "holdout_states": int((~train).sum()),
        "train_decisive_states": int((train & decisive).sum()),
        "train_outside_states": int((train & ~decisive).sum()),
        "branches": legal_branches,
    }
    if actual != expected:
        raise ValueError(f"T5 merged counts do not match immutable T4e bank: {actual}")
    verified = sum(
        int(r["observation_proof"]["all_emitted_states_byte_verified"]) for r in receipts
    )
    if verified != len(ids):
        raise ValueError("observation parity coverage differs from state count")
    train_worlds = set(merged["world"][train])
    held_worlds = set(merged["world"][~train])
    if train_worlds & held_worlds:
        raise ValueError("worlds overlap historical splits")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite merged output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    receipt = {
        "schema": "t5-separability-features/v1",
        "output": str(output),
        "output_sha256": sha256(output),
        "counts": actual,
        "source_sha256": sources[0],
        "input_sha256": inputs[0],
        "worlds": {"train": len(train_worlds), "holdout": len(held_worlds)},
        "parts": {str(part): sha256(part) for part in parts},
        "part_receipts": {
            str(part.with_suffix(".json")): sha256(part.with_suffix(".json")) for part in parts
        },
        "validation": {
            "state_ids_disjoint": True,
            "all_part_observation_bytes_verified": sum(
                int(item["observation_proof"]["all_emitted_states_byte_verified"])
                for item in receipts
            ),
            "all_real_legal_branches_retained": legal_branches,
        },
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, nargs=4, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.parts, args.out), indent=2))


if __name__ == "__main__":
    main()
