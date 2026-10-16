"""Derive a hash-bound v2 label bank without modifying its source bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from research.t3_rollout_labels_20261015.bank_engine import label_set, sha256


def _tree_hash(directory: Path) -> tuple[str, dict[str, str]]:
    """Hash every source bank payload in stable relative-path order."""
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    files = {path.relative_to(directory).as_posix(): sha256(path) for path in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, files


def relabel(source: Path, out: Path, *, policy: str = "prefer-nonboost") -> dict[str, Any]:
    """Write a new v2 bank from stored surviving masks and return exact statistics."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError("derived bank output must differ from its source")
    if source in out.parents:
        raise ValueError("derived bank output cannot be inside its source bank")
    if not source.is_dir() or not list(source.glob("*.npz")):
        raise ValueError("source must be a label-bank directory containing .npz shards")
    if out.exists() and any(out.iterdir()):
        raise ValueError("derived bank output directory must be empty")
    source_hash, source_files = _tree_hash(source)
    out.mkdir(parents=True, exist_ok=True)
    rows = changed = before_boost = after_boost = before_total = after_total = 0
    source_has_outcomes = False
    for shard in sorted(source.glob("*.npz")):
        with np.load(shard, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        surviving = arrays.get("surviving")
        if surviving is None or surviving.ndim != 2 or surviving.shape[1] != 6:
            raise ValueError(f"source shard has no valid surviving matrix: {shard}")
        updated = np.stack([label_set(row, policy) for row in surviving])
        source_has_outcomes |= {"outcome_recorded", "outcome_survived"}.issubset(arrays)
        if not updated.any(axis=1).all():
            raise ValueError(f"label policy produced an empty action set: {shard}")
        rows += len(updated)
        changed += int(np.any(updated != surviving, axis=1).sum())
        before_boost += int(surviving[:, 3:].sum())
        before_total += int(surviving.sum())
        after_boost += int(updated[:, 3:].sum())
        after_total += int(updated.sum())
        if policy == "prefer-nonboost":
            arrays["surviving_original"] = surviving.copy()
        arrays["surviving"] = updated
        temp = out / f".{shard.name}.tmp.npz"
        np.savez_compressed(temp, **arrays)
        os.replace(temp, out / shard.name)
    value = {
        "schema": "t4-derived-label-bank/v2",
        "source": {"path": str(source), "tree_sha256": source_hash, "files": source_files},
        "label_policy": policy,
        "rows": rows,
        "rows_changed": changed,
        "boost_share_before": before_boost / max(1, before_total),
        "boost_share_after": after_boost / max(1, after_total),
        "outcome_provenance": (
            "source includes per-action outcomes"
            if source_has_outcomes
            else "source lacks per-action outcomes; labels were derived from surviving"
        ),
        "shards": {path.name: sha256(path) for path in sorted(out.glob("*.npz"))},
    }
    if _tree_hash(source) != (source_hash, source_files):
        raise RuntimeError("source bank changed while deriving its v2 bank")
    (out / "manifest.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--label-policy", choices=("surviving-set", "prefer-nonboost"), default="prefer-nonboost"
    )
    args = parser.parse_args()
    print(json.dumps(relabel(args.source, args.out, policy=args.label_policy), sort_keys=True))


if __name__ == "__main__":
    main()
