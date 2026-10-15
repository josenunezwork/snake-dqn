"""Collect a fresh death-window label bank with one student playing every branch.

This is development tooling.  It neither trains a student nor qualifies one for
serving or a gate.  The supplied checkpoint is both the played hero and the
continuation policy after each forced first action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from research.b8_death_census_20261014 import census
from research.t3_rollout_labels_20261015.bank_engine import (
    concrete_seed_hits,
    manifest,
    planned_rosters,
    sha256,
)
from research.t3_rollout_labels_20261015.collect import _atomic_json, merge_world_rows
from research.t3_rollout_labels_20261015.death_window import collect_world_death_window


def excluded_seeds(count: int) -> dict[str, set[int]]:
    """Exclude T3's generated evaluation banks even without concrete file hits."""
    from research.redesign_m2_20261008 import ni_spec

    return {"ni": set(ni_spec.ni_seeds())} | {
        f"dev_round_{index}": set(ni_spec.distill_seeds(max(96, count), round_index=index))
        for index in (52, 60, 61)
    }


def _seeds(namespace: str, count: int, roots: tuple[Path, ...]) -> tuple[list[int], dict[str, Any]]:
    """Allocate hash-derived seeds absent from both current audit roots."""
    selected: list[int] = []
    rejected: dict[int, list[str]] = {}
    exclusions = excluded_seeds(count)
    cursor = 0
    while len(selected) < count:
        batch = [
            int.from_bytes(
                hashlib.sha256(f"{namespace}:{index}".encode("utf-8")).digest()[:4], "big"
            )
            for index in range(cursor, cursor + max(2 * count, 16))
        ]
        cursor += len(batch)
        candidates = list(dict.fromkeys(seed for seed in batch if seed not in selected))
        hits_by_root = {str(root): concrete_seed_hits(root, candidates) for root in roots}
        for candidate in candidates:
            hits = [root for root, values in hits_by_root.items() if candidate in values]
            hits.extend(
                f"generated:{name}" for name, values in exclusions.items() if candidate in values
            )
            if hits:
                rejected[candidate] = hits
            else:
                selected.append(candidate)
                if len(selected) == count:
                    break
    return selected, {
        "method": "sha256-namespace-counter + rg-o-fixed-word-concrete-seed-rejection/v1",
        "namespace": namespace,
        "roots": [str(root) for root in roots],
        "candidates_examined": cursor,
        "rejected": [{"seed": seed, "roots": hits} for seed, hits in sorted(rejected.items())],
        "selected": selected,
        "excluded_namespace_counts": {name: len(values) for name, values in exclusions.items()},
        "excluded_namespace_overlap": {
            name: sorted(set(selected) & values) for name, values in exclusions.items()
        },
    }


def plan(student: Path, out: Path, namespace: str, worlds: int, mixes: list[str]) -> dict[str, Any]:
    """Create the immutable fresh-round seed audit and roster plan before collection."""
    student, out = Path(student).resolve(), Path(out).resolve()
    if not student.is_file():
        raise ValueError(f"student checkpoint does not exist: {student}")
    if worlds < 1:
        raise ValueError("world count must be positive")
    if not mixes or any(mix not in {"scripted", "mixed", "frozen"} for mix in mixes):
        raise ValueError("mixes must be scripted, mixed, or frozen")
    if out.exists() and any(out.iterdir()):
        raise ValueError("fresh round output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    roots = (repo, census.ARTIFACTS)
    for root in roots:
        result = subprocess.run(
            ["rg", "-l", "--fixed-strings", namespace, str(root)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(f"namespace audit failed for {root}: exit {result.returncode}")
        if result.stdout.strip():
            raise ValueError(f"expert-iteration namespace is stale under {root}")
    seeds, audit = _seeds(namespace, worlds, roots)
    chosen_mixes = [mixes[index % len(mixes)] for index in range(worlds)]
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    plans = planned_rosters(gid._context(1), seeds, chosen_mixes)
    jobs = [
        {"arm": "student", "mix": mix, "cycle": 0, "index": index}
        for index, mix in enumerate(chosen_mixes)
    ]
    value = {
        "schema": "t4-expert-iteration-round/v1",
        "namespace": namespace,
        "student": {"path": str(student), "sha256": sha256(student)},
        "seed_audit": audit,
        "jobs": jobs,
        "plans": plans,
        "played_policy": "student",
        "continuation_policy": "student",
    }
    _atomic_json(out / "namespace.json", value)
    return value


def collect(
    round_plan: dict[str, Any],
    out: Path,
    *,
    record_outcomes: bool,
    label_policy: str,
    max_seconds: float,
) -> dict[str, Any]:
    """Run the planned small collection serially and materialize its derived bank."""
    student = Path(round_plan["student"]["path"])
    if sha256(student) != round_plan["student"]["sha256"]:
        raise RuntimeError("student checkpoint changed before collection")
    identity = {
        "schema": "t4-expert-iteration-run/v1",
        "student_sha256": round_plan["student"]["sha256"],
        "played_policy": "student",
        "continuation_policy": "student",
        "record_outcomes": record_outcomes,
        "label_policy": label_policy,
        "source_sha256": {
            path.name: sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).parents[1] / "t3_rollout_labels_20261015" / "bank_engine.py",
                Path(__file__).parents[1] / "t3_rollout_labels_20261015" / "death_window.py",
                Path(__file__).parents[1] / "t3_rollout_labels_20261015" / "collect.py",
                Path(__file__).parents[1] / "t3_rollout_labels_20261015" / "engine.py",
            )
        },
    }
    receipts = []
    started = time.monotonic()
    for job, roster in zip(round_plan["jobs"], round_plan["plans"]):
        if receipts and time.monotonic() - started >= max_seconds:
            break
        receipt = collect_world_death_window(
            job,
            roster,
            out,
            identity,
            record_outcomes=record_outcomes,
            label_policy=label_policy,
            hero_checkpoint=student,
        )
        receipt["_resume_kind"] = "current"
        receipts.append(receipt)
    if sha256(student) != round_plan["student"]["sha256"]:
        raise RuntimeError("student checkpoint changed during collection")
    stats = merge_world_rows(out, receipts, identity, {}, counterweight_ratio=0.5)
    provenance = {
        "schema": "t4-expert-iteration-bank/v1",
        "namespace": round_plan["namespace"],
        "student": round_plan["student"],
        "played_policy": "student",
        "continuation_policy": "student",
        "record_outcomes": record_outcomes,
        "label_policy": label_policy,
        "worlds": len(receipts),
        "requested_worlds": len(round_plan["jobs"]),
        "max_seconds": max_seconds,
        "wall_seconds": time.monotonic() - started,
        "stats": stats,
        "receipt_keys": [str(receipt["key"]) for receipt in receipts],
    }
    manifest(out, provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", type=str, required=True)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--mixes", type=str, default="scripted")
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    parser.add_argument("--record-outcomes", action="store_true")
    parser.add_argument(
        "--label-policy", choices=("surviving-set", "prefer-nonboost"), default="surviving-set"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mixes = [part for part in args.mixes.split(",") if part]
    namespace = f"t4-expert-iteration-20261015/{args.round}"
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise SystemExit("--max-seconds must be positive")
    round_plan = plan(args.student, args.out, namespace, args.worlds, mixes)
    if args.dry_run:
        print(json.dumps({"status": "PLANNED", **round_plan}, sort_keys=True))
        return
    # A single serial worker stays below the two-worker resource limit.
    print(
        json.dumps(
            collect(
                round_plan,
                args.out,
                record_outcomes=args.record_outcomes,
                label_policy=args.label_policy,
                max_seconds=args.max_seconds,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
