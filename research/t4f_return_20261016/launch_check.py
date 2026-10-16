#!/usr/bin/env python3
"""Fail-closed validator for the later-bound, create-only RETURN_C launch receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

COLLECTION_SCHEMA = "t4f-return-bank/v3"
SCHEMA_VERSION = 2
MACRO_POLICY = "first_branch_boost_then_B8_direction_boost_if_legal_k_frames_then_B8"
MACRO_DEPTH = 300


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_corrected_macro(name: str, manifest: dict[str, Any]) -> None:
    """Reject banks made with addendum 2's superseded repeated-turn macro."""
    macro = manifest.get("macro")
    if not isinstance(macro, dict):
        raise ValueError(f"{name} macro-bank has no macro identity")
    if macro.get("policy") != MACRO_POLICY:
        raise ValueError(f"{name} macro-bank does not use addendum 2 boost geometry")
    if macro.get("depth") != MACRO_DEPTH:
        raise ValueError(f"{name} macro-bank has unexpected macro depth")
    period = macro.get("boost_cost_period_frames")
    if not isinstance(period, int) or isinstance(period, bool) or period < 1:
        raise ValueError(f"{name} macro-bank has invalid boost cost period")


def _verify_bank(name: str, bank: dict[str, Any]) -> dict[str, int]:
    root = Path(bank["path"])
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{name} macro bank manifest missing")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("collection_schema") != COLLECTION_SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"unexpected {name} macro-bank schema")
    _verify_corrected_macro(name, manifest)
    shards = manifest.get("shards")
    if not isinstance(shards, dict) or {p.name for p in root.glob("*.npz")} != set(shards):
        raise ValueError(f"{name} macro-bank shard set differs from its manifest")
    for shard_name, expected in shards.items():
        shard = root / shard_name
        if not shard.is_file() or sha256(shard) != expected:
            raise ValueError(f"{name} macro-bank shard changed: {shard}")
    rows = manifest["publication"]["rows"]
    if rows != bank["publication_rows"]:
        raise ValueError(f"{name} macro-bank row count differs from registered value")
    return {"shards": len(shards), "rows": rows}


def verify(binding_path: Path) -> dict[str, Any]:
    """Verify the new macro bank plus all SHA-bound production inputs."""
    binding = json.loads(binding_path.read_text())
    if binding.get("schema") != "t4f-return-launch-inputs/v1":
        raise ValueError("unexpected RETURN_C launch binding schema")
    for raw, expected in binding["files"].items():
        path = Path(raw)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"registered input changed: {path}")
    train, holdout = _verify_bank("training", binding["training_return_bank"]), _verify_bank(
        "holdout", binding["holdout_return_bank"]
    )
    outside = Path(binding["heldout_outside"]["path"])
    outside_manifest = json.loads((outside / "manifest.json").read_text())
    if outside_manifest.get("schema") != "t4f-heldout-outside-observations/v1":
        raise ValueError("unrecognized heldout outside schema")
    if outside_manifest["states"] != binding["heldout_outside"]["states"]:
        raise ValueError("heldout outside count changed")
    if set(outside_manifest["shards"]) != {p.name for p in outside.glob("*.npz")}:
        raise ValueError("heldout outside shard set changed")
    for name, expected in outside_manifest["shards"].items():
        if sha256(outside / name) != expected:
            raise ValueError("heldout outside observations changed")
    return {
        "files": len(binding["files"]),
        "training_rows": train["rows"],
        "training_shards": train["shards"],
        "holdout_rows": holdout["rows"],
        "holdout_shards": holdout["shards"],
        "heldout_outside_states": outside_manifest["states"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binding", type=Path)
    parser.add_argument(
        "--shell", action="store_true", help="print only safely constrained shell assignments"
    )
    args = parser.parse_args()
    result = verify(args.binding)
    binding = json.loads(args.binding.read_text())
    weight = binding.get("return_weight")
    if not isinstance(weight, (float, int)) or not 0 < weight <= 1:
        raise ValueError("launch binding must contain return_weight in (0, 1]")
    if args.shell:
        print(f"return_weight={float(weight):.12g}")
    else:
        result["return_weight"] = float(weight)
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
