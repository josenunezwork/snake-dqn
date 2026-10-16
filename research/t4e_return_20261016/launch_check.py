#!/usr/bin/env python3
"""Validate the immutable T4e RETURN-launch inputs without opening a training device."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Return one file's streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_bank(name: str, bank: dict[str, Any]) -> dict[str, int]:
    """Verify one exact published bank, including every declared shard."""
    root = Path(bank["path"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("collection_schema") != "t4e-return-bank/v1":
        raise ValueError(f"unexpected {name} return-bank schema")
    shards = manifest.get("shards")
    if not isinstance(shards, dict) or {path.name for path in root.glob("*.npz")} != set(shards):
        raise ValueError(f"{name} return-bank shard set differs from its manifest")
    for shard_name, expected in shards.items():
        path = root / shard_name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{name} return-bank shard changed: {path}")
    rows = manifest["publication"]["rows"]
    if rows != bank["publication_rows"]:
        raise ValueError(f"{name} return-bank row count differs from its registered value")
    return {"shards": len(shards), "rows": rows}


def verify(binding_path: Path) -> dict[str, Any]:
    """Verify immutable launch inputs and exact train/holdout return-bank publications."""
    binding = json.loads(binding_path.read_text())
    if binding.get("schema") != "t4e-return-launch-inputs/v1":
        raise ValueError("unexpected launch binding schema")
    for raw, expected in binding["files"].items():
        path = Path(raw)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"registered input changed: {path}")
    training = _verify_bank("training", binding["training_return_bank"])
    holdout = _verify_bank("holdout", binding["holdout_return_bank"])
    bank = binding["training_return_bank"]
    completion = json.loads(Path(bank["completion_path"]).read_text())
    expected_counts = bank["completion_counts"]
    for key, expected in expected_counts.items():
        if completion.get(key) != expected:
            raise ValueError(f"completion receipt mismatch for {key}")
    return {
        "files": len(binding["files"]),
        "holdout_rows": holdout["rows"],
        "holdout_shards": holdout["shards"],
        "training_rows": training["rows"],
        "training_shards": training["shards"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binding", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.binding), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
