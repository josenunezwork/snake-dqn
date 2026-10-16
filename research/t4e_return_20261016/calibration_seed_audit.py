"""Reproduce the T4e calibration seed/world exclusion receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.redesign_m2_20261008 import ni_spec  # noqa: E402

WORLD_RE = re.compile(r"world-(\d+)-.*-s(\d+)(?:-|\.)")


def sha256(p: Path) -> str:
    """Bind a provenance file's exact bytes."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(values: set[int]) -> str:
    """Hash the canonical sorted seed set."""
    return hashlib.sha256(",".join(map(str, sorted(values))).encode()).hexdigest()


def main() -> int:
    """Compare the entire collection namespace with reserved evaluation worlds."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--generic-audit", type=Path)
    args = ap.parse_args()
    root = args.artifacts.resolve()
    staged = {}
    staged_conflicts = []
    staged_files = list(root.rglob("world-*.npz"))
    for p in staged_files:
        m = WORLD_RE.match(p.name)
        if m:
            world, seed = int(m.group(1)), int(m.group(2))
            if world in staged and staged[world] != seed:
                staged_conflicts.append(
                    {"world": world, "prior_seed": staged[world], "seed": seed, "path": str(p)}
                )
            staged[world] = seed
    manifests, banks = {}, {}
    for kind in ("train", "holdout"):
        p = root / "banks" / kind / "manifest.json"
        manifests[kind] = {"path": str(p), "sha256": sha256(p)}
        d = json.loads(p.read_text())
        names = d["publication"]["world_receipts"]
        rows = [WORLD_RE.match(n) for n in names]
        banks[kind] = {
            "world_ids": {int(m.group(1)) for m in rows},
            "seeds": {int(m.group(2)) for m in rows},
            "pairs": {(int(m.group(1)), int(m.group(2))) for m in rows},
        }
    rounds = {}
    for r in (68, 69):
        seeds = set(ni_spec.distill_seeds(96, round_index=r))
        rounds[str(r)] = {
            "count": len(seeds),
            "seed_digest": digest(seeds),
            "overlap_all_1200_collection_seeds": sorted(seeds & set(staged.values())),
            "overlap_train_bank_seeds": sorted(seeds & banks["train"]["seeds"]),
            "overlap_holdout_bank_seeds": sorted(seeds & banks["holdout"]["seeds"]),
        }
    plans = sorted((root / "plans").glob("cycle-*.json"))
    plan_rows = [job for p in plans for job in json.loads(p.read_text()).get("jobs", [])]
    plan_mismatches = [
        {"plan_index": j.get("plan_index"), "cycle": j.get("cycle"), "seed": j.get("seed")}
        for j in plan_rows
        if staged.get(j.get("plan_index")) != j.get("seed")
    ]
    bank_mismatches = []
    for kind in ("train", "holdout"):
        for world, seed in banks[kind]["pairs"]:
            if staged.get(world) != seed:
                bank_mismatches.append({"bank": kind, "world": world, "seed": seed})
        if banks[kind]["seeds"] - set(staged.values()):
            bank_mismatches.append(
                {
                    "bank": kind,
                    "unseen_seed_count": len(banks[kind]["seeds"] - set(staged.values())),
                }
            )
    generic_path = args.generic_audit
    generic = json.loads(generic_path.read_text()) if generic_path else {"status": "NOT_RUN"}
    passed = (
        len(staged) == 1200
        and len(set(staged.values())) == 1200
        and set(staged) == set(range(1200))
        and not staged_conflicts
        and len(plan_rows) == 1200
        and len({job["plan_index"] for job in plan_rows}) == 1200
        and not plan_mismatches
        and not bank_mismatches
        and not (banks["train"]["world_ids"] & banks["holdout"]["world_ids"])
        and not (banks["train"]["seeds"] & banks["holdout"]["seeds"])
        and all(not x["overlap_all_1200_collection_seeds"] for x in rounds.values())
        and all(
            not x["overlap_train_bank_seeds"] and not x["overlap_holdout_bank_seeds"]
            for x in rounds.values()
        )
    )
    receipt = {
        "schema": "t4e-calibration-seed-audit/v3",
        "status": "PASS" if passed else "FAIL",
        "audit_source_sha256": sha256(Path(__file__)),
        "generator": {
            "path": "research/redesign_m2_20261008/ni_spec.py",
            "symbol": "distill_seeds(96, round_index)",
            "source_sha256": sha256(
                Path(__file__).parents[2] / "research/redesign_m2_20261008/ni_spec.py"
            ),
        },
        "collection": {
            "staged_npz_files": len(staged_files),
            "unique_world_ids": len(staged),
            "world_id_range": [min(staged), max(staged)] if staged else [],
            "world_ids_complete": set(staged) == set(range(1200)),
            "seed_digest": digest(set(staged.values())),
            "staged_conflicts": staged_conflicts,
            "plan_job_count": len(plan_rows),
            "plan_mismatches": plan_mismatches,
            "plan_files": [{"path": str(p), "sha256": sha256(p)} for p in plans],
        },
        "banks": {
            "manifests": manifests,
            "train_worlds": len(banks["train"]["world_ids"]),
            "holdout_worlds": len(banks["holdout"]["world_ids"]),
            "world_overlap": sorted(banks["train"]["world_ids"] & banks["holdout"]["world_ids"]),
            "seed_overlap": sorted(banks["train"]["seeds"] & banks["holdout"]["seeds"]),
        },
        "bank_staged_mismatches": bank_mismatches,
        "rounds": rounds,
        "mix_semantics": (
            "Round rosters are generated as 96 scripted seeds by distill_seeds; "
            "dev_probe uses prefixes of that roster for the frozen and mixed 32-world "
            "assignments. This receipt does not claim 160 distinct seeds."
        ),
        "dev_probe_source_sha256": sha256(
            Path(__file__).parents[2] / "research/redesign_m2_20261008/dev_probe.py"
        ),
        "generic_seed_audit": {
            "path": str(generic_path) if generic_path else None,
            "sha256": sha256(generic_path) if generic_path else None,
            "scope_note": (
                "Unfiltered broad audit is recorded separately from direct bank exclusion. "
                "Its hits include prior audit receipts and the collection transcript; "
                "they are not evidence of played evaluation worlds. Refresh the broad "
                "audit with exact documentary exclusions at evaluation launch."
            ),
            "status": generic.get("status"),
            "pass": generic.get("pass"),
            "hit_paths": [
                json.loads(row["match_paths"])
                for rows in generic.get("text_scan", {}).get("progress", {}).values()
                for row in rows
                if row.get("match_paths")
            ],
            "checks": {
                r: {
                    "observed_json_collisions": len(v.get("observed_json_collisions", [])),
                    "reserved_metadata": generic.get("text_scan", {}).get(
                        "reserved_round_metadata", {}
                    ),
                }
                for r, v in generic.get("checks", {}).items()
            },
        },
    }
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
