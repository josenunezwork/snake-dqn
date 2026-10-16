#!/usr/bin/env python3
"""Independently audit the T4f Addendum 2 derived return bank.

This checker deliberately does not call the collector's source or shard
validation functions.  It verifies the immutable provenance chain and compares
every source/v2/v3 row directly.  Only the outcome fields of boost rows may
change in the v3 bank; all non-boost rows must remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_FIELDS = (
    "state_packed",
    "state_scalars",
    "action",
    "T",
    "died",
    "S",
    "L0",
    "L_T",
    "terminal_packed",
    "terminal_scalars",
    "terminal_mask",
    "row_weight",
    "family",
    "source_seed",
    "source_decision",
    "death_distance",
)
IMMUTABLE_FIELDS = (
    "state_packed",
    "state_scalars",
    "action",
    "L0",
    "row_weight",
    "family",
    "source_seed",
    "source_decision",
    "death_distance",
)
OUTCOME_FIELDS = {"T", "died", "S", "L_T", "terminal_packed", "terminal_scalars", "terminal_mask"}
EXPECTED = {
    "train": {"states": 18924, "rows": 97846, "boost": 47269},
    "holdout": {"states": 1052, "rows": 5418, "boost": 2622},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def manifest_shards(root: Path) -> tuple[dict[str, Any], list[str]]:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    listed = manifest.get("shards")
    if not isinstance(listed, dict):
        raise AssertionError(f"{manifest_path}: shards is not an object")
    actual = sorted(path.name for path in root.glob("*.npz"))
    if sorted(listed) != actual:
        raise AssertionError(f"{manifest_path}: listed shard set differs from directory")
    bad = [name for name, expected in listed.items() if sha256(root / name) != expected]
    if bad:
        raise AssertionError(f"{manifest_path}: shard hash mismatch: {bad[:3]}")
    return manifest, actual


def source_identity(source: Path) -> dict[str, Any]:
    binding_path = source / "binding.json"
    binding = read_json(binding_path)
    plans = sorted((source / "plans").glob("cycle-*.json"))
    if not plans:
        raise AssertionError("source has no cycle plans")
    banks: dict[str, Any] = {}
    for name in ("train", "holdout"):
        manifest, shards = manifest_shards(source / "banks" / name)
        banks[name] = {
            "manifest_sha256": sha256(source / "banks" / name / "manifest.json"),
            "shards": len(shards),
        }
        if manifest.get("collection_schema") != "t4e-return-bank/v1":
            raise AssertionError(f"source {name}: unexpected schema")
    finals = binding.get("finals")
    if not isinstance(finals, dict):
        raise AssertionError("source binding has no finals")
    checkpoints = {}
    for arm, value in sorted(finals.items()):
        if not isinstance(value, dict) or not Path(value["path"]).is_file():
            raise AssertionError(f"source checkpoint missing: {arm}")
        actual = sha256(Path(value["path"]))
        if actual != value["sha256"]:
            raise AssertionError(f"source checkpoint hash mismatch: {arm}")
        checkpoints[arm] = value["sha256"]
    return {
        "binding_sha256": sha256(binding_path),
        "plan_sha256": {path.name: sha256(path) for path in plans},
        "banks": banks,
        "checkpoints": checkpoints,
    }


def assert_equal(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not same_bytes(left, right):
        raise AssertionError(
            f"{label}: arrays differ ({left.dtype}/{left.shape} vs {right.dtype}/{right.shape})"
        )


def same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare dtype, shape, and serialized element bytes exactly."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()


def load_archive(path: Path) -> dict[str, np.ndarray]:
    """Materialize each compressed array once so comparisons stay vectorized."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def changed_row_count(left: np.ndarray, right: np.ndarray) -> int:
    """Count rows with any serialized-byte difference."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return int(len(left))
    if not len(left):
        return 0
    left_bytes = np.ascontiguousarray(left).reshape(len(left), -1).view(np.uint8)
    right_bytes = np.ascontiguousarray(right).reshape(len(right), -1).view(np.uint8)
    return int(np.any(left_bytes != right_bytes, axis=1).sum())


def audit_bank(source: Path, old: Path, new: Path, name: str) -> dict[str, Any]:
    source_manifest, source_shards = manifest_shards(source / "banks" / name)
    old_manifest, old_shards = manifest_shards(old / "banks" / name)
    new_manifest, new_shards = manifest_shards(new / "banks" / name)
    if source_shards != old_shards or source_shards != new_shards:
        raise AssertionError(f"{name}: shard names differ across source/v2/v3")
    if new_manifest.get("collection_schema") != "t4f-return-bank/v3":
        raise AssertionError(f"{name}: new schema is not t4f-return-bank/v3")
    source_manifest_hash = sha256(source / "banks" / name / "manifest.json")
    if old_manifest.get("source_bank", {}).get("manifest_sha256") != source_manifest_hash:
        raise AssertionError(f"{name}: v2 source-bank manifest binding mismatch")
    if new_manifest.get("source_bank", {}).get("manifest_sha256") != source_manifest_hash:
        raise AssertionError(f"{name}: v3 source-bank manifest binding mismatch")
    rows = states = boost = nonboost = 0
    changed_boost_fields: dict[str, int] = {field: 0 for field in OUTCOME_FIELDS}
    b8_mismatch = 0
    for shard_name in source_shards:
        src = load_archive(source / "banks" / name / shard_name)
        v2 = load_archive(old / "banks" / name / shard_name)
        v3 = load_archive(new / "banks" / name / shard_name)
        if len(src["action"]) != len(v2["action"]) or len(src["action"]) != len(v3["action"]):
            raise AssertionError(f"{name}/{shard_name}: row count differs")
        if not same_bytes(src["action"], v2["action"]) or not same_bytes(
            src["action"], v3["action"]
        ):
            raise AssertionError(f"{name}/{shard_name}: action order changed")
        for field in SOURCE_FIELDS:
            if field not in src or field not in v2 or field not in v3:
                raise AssertionError(f"{name}/{shard_name}: missing field {field}")
        if "b8_action" not in v2 or "b8_action" not in v3:
            raise AssertionError(f"{name}/{shard_name}: missing b8_action")
        assert_equal(v2["b8_action"], v3["b8_action"], f"{name}/{shard_name}: b8_action")
        b8_mismatch += int(np.count_nonzero(v2["b8_action"] != v2["action"]))
        actions = np.asarray(src["action"], dtype=np.int64)
        boost_mask = actions >= 3
        nonboost_mask = ~boost_mask
        boost += int(boost_mask.sum())
        nonboost += int(nonboost_mask.sum())
        for field in IMMUTABLE_FIELDS:
            if not same_bytes(src[field][boost_mask], v3[field][boost_mask]):
                raise AssertionError(f"{name}/{shard_name}: v3 changed boost {field}")
            if not same_bytes(v2[field][boost_mask], v3[field][boost_mask]):
                raise AssertionError(f"{name}/{shard_name}: v2/v3 boost {field} differs")
        for field in OUTCOME_FIELDS:
            changed_boost_fields[field] += changed_row_count(
                src[field][boost_mask], v3[field][boost_mask]
            )
        for field in SOURCE_FIELDS:
            if not same_bytes(src[field][nonboost_mask], v3[field][nonboost_mask]):
                raise AssertionError(
                    f"{name}/{shard_name}: nonboost differs from source in {field}"
                )
            if not same_bytes(v2[field][nonboost_mask], v3[field][nonboost_mask]):
                raise AssertionError(f"{name}/{shard_name}: nonboost differs from v2 in {field}")
        rows += len(actions)
        states += len(np.unique(src["source_decision"]))
    expected = EXPECTED[name]
    if {"rows": rows, "boost": boost, "states": states} != expected:
        raise AssertionError(f"{name}: totals {rows=} {boost=} {states=} expected {expected}")
    pub = new_manifest.get("publication", {})
    for key, value in (("rows", rows), ("states", states), ("boost_rows_replaced", boost)):
        if int(pub.get(key, -1)) != value:
            raise AssertionError(f"{name}: publication {key} does not match arrays")
    return {
        "shards": len(source_shards),
        "rows": rows,
        "states": states,
        "boost_rows": boost,
        "nonboost_rows": nonboost,
        "b8_action_mismatches_with_branch_action": b8_mismatch,
        "boost_outcome_fields_changed_rows": changed_boost_fields,
        "source_manifest_sha256": source_manifest_hash,
        "old_manifest_sha256": sha256(old / "banks" / name / "manifest.json"),
        "new_manifest_sha256": sha256(new / "banks" / name / "manifest.json"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    old = args.old.resolve()
    new = args.new.resolve()
    identity = source_identity(source)
    old_binding = read_json(old / "macro_binding.json")
    if old_binding.get("source_identity") != identity:
        raise AssertionError(
            "v2 macro binding source identity differs from independently recomputed source"
        )
    if int(old_binding.get("boost_cost_period_frames", -1)) != 3:
        raise AssertionError("v2 boost cost period is not 3")
    macro_binding = read_json(new / "macro_binding.json")
    if macro_binding.get("schema") != "t4f-return-bank/v3":
        raise AssertionError("v3 macro binding schema mismatch")
    if macro_binding.get("source_identity") != identity:
        raise AssertionError(
            "v3 macro binding source identity differs from independently recomputed source"
        )
    if int(macro_binding.get("boost_cost_period_frames", -1)) != 3:
        raise AssertionError("boost cost period is not 3")
    expected_policy = "first_branch_boost_then_B8_direction_boost_if_legal_k_frames_then_B8"
    if macro_binding.get("policy") != expected_policy:
        raise AssertionError("v3 macro policy binding mismatch")
    completion = read_json(new / "completion.json")
    if completion.get("source_identity") != identity:
        raise AssertionError("v3 completion source identity mismatch")
    result = {
        "schema": "t4f-addendum2-independent-validation/v1",
        "source": str(source),
        "old": str(old),
        "new": str(new),
        "source_identity": identity,
        "banks": {name: audit_bank(source, old, new, name) for name in ("train", "holdout")},
        "totals": {
            "rows": sum(EXPECTED[name]["rows"] for name in EXPECTED),
            "states": sum(EXPECTED[name]["states"] for name in EXPECTED),
            "boost_rows": sum(EXPECTED[name]["boost"] for name in EXPECTED),
            "nonboost_rows": sum(
                EXPECTED[name]["rows"] - EXPECTED[name]["boost"] for name in EXPECTED
            ),
        },
        "status": "PASS",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", type=Path, required=True)
    value.add_argument("--old", type=Path, required=True)
    value.add_argument("--new", type=Path, required=True)
    value.add_argument("--out", type=Path, required=True)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
