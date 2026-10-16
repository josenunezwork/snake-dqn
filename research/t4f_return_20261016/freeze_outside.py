"""Freeze all eligible unique outside observations from preassigned holdout worlds.

No gameplay, outcomes, training rows, or model-dependent selection is introduced.
The original phase-A candidate eligibility already requires 300 future decisions
away from death/censoring. Ordering and deduplication use plan/decision order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.t4e_return_20261016.collection import masked_greedy
from research.t4e_return_20261016.publisher import load_spool


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(source: Path, out: Path) -> dict:
    """Publish create-only observations with complete source receipt bindings."""
    if out.exists():
        raise FileExistsError(out)
    plans = sorted((source / "plans").glob("cycle-*.json"))
    jobs = {j["plan_index"]: j for p in plans for j in json.loads(p.read_text())["jobs"]}
    receipts = {
        json.loads(p.read_text())["job"]["plan_index"]: p
        for p in (source / "staging/phase-a").glob("*.json")
    }
    if len(jobs) != 1200 or set(jobs) != set(receipts):
        raise ValueError("expected all 1200 original collection worlds")
    heldout = {i for i, j in jobs.items() if j["holdout"]}
    if len(heldout) != 120:
        raise ValueError("expected 120 preassigned heldout worlds")
    source_manifest = source / "banks/train/manifest.json"
    bindings = {str(p.resolve()): sha256(p) for p in [*plans, source_manifest]}
    # Prove no observation overlaps training, beyond the stronger world exclusion.
    training_keys = set()
    train_meta = json.loads(source_manifest.read_text())
    for name, expected in train_meta["shards"].items():
        p = source_manifest.parent / name
        if sha256(p) != expected:
            raise ValueError(f"training shard changed: {p}")
        with np.load(p, allow_pickle=False) as data:
            training_keys.update(
                (a.tobytes(), b.tobytes())
                for a, b in zip(data["state_packed"], data["state_scalars"])
            )
            if set(map(int, data["source_seed"])) & {jobs[i]["seed"] for i in heldout}:
                raise ValueError("training/holdout world overlap")
    kept, seen, duplicates, overlaps = [], set(), 0, 0
    for i in sorted(heldout):
        receipt_path = receipts[i]
        receipt = json.loads(receipt_path.read_text())
        spool = source / "staging" / receipt["candidate_spool"]
        if receipt["status"] != "COMPLETE" or sha256(spool) != receipt["candidate_spool_sha256"]:
            raise ValueError(f"invalid heldout source receipt: {receipt_path}")
        bindings.update({str(p.resolve()): sha256(p) for p in (receipt_path, spool)})
        for row in sorted(load_spool(spool), key=lambda r: r["source_decision"]):
            if not row["holdout"] or row["plan_index"] != i:
                raise ValueError("candidate disagrees with original world assignment")
            if row["length"] < 700 or 0 <= row["death_distance"] < 300:
                raise ValueError("ineligible heldout outside candidate")
            key = (row["state_packed"].tobytes(), row["state_scalars"].tobytes())
            if key in training_keys:
                overlaps += 1
                continue
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            mask = np.zeros(6, bool)
            mask[row["legal"]] = True
            # Outside spools use -1 for an unmaterialized action; the saved
            # source-B8 Q vector and advisory legal set determine it exactly.
            b8_action = masked_greedy(row["q"], row["legal"])
            if row["b8_action"] not in (-1, b8_action):
                raise ValueError("recorded B8 action disagrees with saved Q/legal set")
            row["b8_action"] = b8_action
            if not mask[row["b8_action"]] or mask.sum() < 2:
                continue
            kept.append({**row, "legal_mask": mask, "source_seed": jobs[i]["seed"]})
    if not kept:
        raise ValueError("empty heldout outside set")
    out.mkdir(parents=True)
    path = out / "observations.npz"
    np.savez_compressed(
        path,
        state_packed=np.stack([r["state_packed"] for r in kept]),
        state_scalars=np.stack([r["state_scalars"] for r in kept]),
        b8_action=np.array([r["b8_action"] for r in kept], np.int64),
        legal_mask=np.stack([r["legal_mask"] for r in kept]),
        source_seed=np.array([r["source_seed"] for r in kept], np.int64),
        source_decision=np.array([r["source_decision"] for r in kept], np.int64),
        source_plan_index=np.array([r["plan_index"] for r in kept], np.int64),
    )
    result = {
        "schema": "t4f-heldout-outside-observations/v1",
        "layout": train_meta["layout"],
        "selection": "all eligible unique phase-A candidates in plan/decision order; at least two legal actions",
        "source_files": bindings,
        "shards": {path.name: sha256(path)},
        "states": len(kept),
        "reserved_worlds": len(heldout),
        "represented_worlds": len({r["plan_index"] for r in kept}),
        "duplicates_excluded": duplicates,
        "training_observation_overlap_excluded": overlaps,
        "source_world_ids": sorted(heldout),
        "source_seeds": sorted(jobs[i]["seed"] for i in heldout),
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.source.resolve(), args.out.resolve())
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"source_files", "source_seeds", "source_world_ids"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
