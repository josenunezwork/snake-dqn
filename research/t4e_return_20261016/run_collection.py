#!/usr/bin/env python3
"""Run the pre-registered, disk-backed T4e return-bank collection.

The coordinator intentionally has no training mode.  It writes immutable plans before
any worlds play, asks the disk workers for only compact receipts, makes every global
selection in stable plan order, and publishes a trainer input only after exact matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.t3_rollout_labels_20261015.bank_engine import (  # noqa: E402
    MIX_CYCLE,
    planned_rosters,
    sha256,
)
from research.t4e_return_20261016 import collection, publisher, seed_audit  # noqa: E402

WORKER_LIMIT = 12
CYCLE_WORLDS = 200
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
PREREG = Path("docs/research/specs/t4e_prereg_2026-10-16.md")
REFERENCE_BANK = ARTIFACTS / "t4c-data-20261016/bank-v3"
B8_ROOT = ARTIFACTS / "redesign-m4b-mac-20261013"
BOUND_SOURCES = (
    Path("research/t4e_return_20261016/run_collection.py"),
    Path("research/t4e_return_20261016/collection.py"),
    Path("research/t4e_return_20261016/publisher.py"),
    Path("research/t4e_return_20261016/seed_audit.py"),
    Path("research/t4d_return_regression_20261016/collector.py"),
    Path("research/t4d_return_regression_20261016/bank.py"),
    Path("research/t4d_return_regression_20261016/planning.py"),
    Path("research/t3_rollout_labels_20261015/bank_engine.py"),
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one evidence JSON file without replacing prior evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite collection evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite collection evidence: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _hash(path: Path) -> str:
    return sha256(Path(path))


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
    )
    return result.stdout.strip()


def source_hashes() -> dict[str, str]:
    """Hash the exact collector stack that is frozen for this collection."""
    missing = [str(path) for path in BOUND_SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing bound collection source: {missing}")
    return {str(path): _hash(path) for path in BOUND_SOURCES}


def ac_power() -> bool:
    """Return whether macOS currently reports external power; fail closed elsewhere."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "AC Power" in result.stdout


def require_ac() -> None:
    """Stop before a cycle when the pre-registered AC condition is absent."""
    if not ac_power():
        raise RuntimeError("T4e collection requires AC power before each fresh cycle")


def finals(root: Path = B8_ROOT) -> dict[str, Path]:
    """Bind exactly the five immutable B8 final checkpoints."""
    result = {f"B8_s{i}": (Path(root) / f"B8_s{i}" / "final.pth").resolve() for i in range(5)}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required B8 final checkpoints: {missing}")
    return result


def _candidate_seed(namespace: str, counter: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{namespace}:{counter}".encode()).digest()[:4], "big")


def allocate_fresh_seeds(
    namespace: str, count: int, roots: Sequence[Path]
) -> tuple[list[int], dict]:
    """Allocate concrete unseen seeds with T4e's partitioned single-thread scanner."""
    from research.redesign_m2_20261008 import ni_spec

    _audit, generated = seed_audit._registered(Path.cwd())
    for round_index in seed_audit.ROUNDS:
        generated.update(ni_spec.distill_seeds(96, round_index=round_index))
    selected: list[int] = []
    rejected: dict[str, list[str]] = {}
    cursor = 0
    while len(selected) < count:
        candidates = list(dict.fromkeys(_candidate_seed(namespace, cursor + i) for i in range(400)))
        cursor += 400
        tokens = [str(value) for value in candidates]
        hits = {str(root): seed_audit.concrete_text_hits(root, tokens)[0] for root in roots}
        for candidate in candidates:
            reasons = [name for name, values in hits.items() if str(candidate) in values]
            if candidate in generated:
                reasons.append("generated-or-reserved-evaluation")
            if candidate in selected:
                reasons.append("already-selected")
            if reasons:
                rejected[str(candidate)] = reasons
            else:
                selected.append(candidate)
                if len(selected) == count:
                    break
        if cursor > 2_000_000:
            raise RuntimeError(
                "fresh collection seed allocation exhausted its deterministic budget"
            )
    return selected, {
        "method": "t4e-partitioned-rg-fixed-word/threads1/timeout60",
        "namespace": namespace,
        "scan_roots": [str(root) for root in roots],
        "selected": selected,
        "rejected": rejected,
        "generated_exclusion_count": len(generated),
    }


def make_cycle_jobs(
    namespace: str,
    cycle: int,
    start_index: int,
    bound_finals: Mapping[str, Path],
    roots: Sequence[Path],
) -> tuple[list[dict], dict]:
    """Create one complete 5 x 40 stable roster and assign holdout before play."""
    seeds, audit = allocate_fresh_seeds(f"{namespace}/cycle-{cycle}", CYCLE_WORLDS, roots)
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    plans: list[dict] = []
    cursor = 0
    context = gid._context(1)
    for arm_index in range(5):
        arm = f"B8_s{arm_index}"
        arm_plans = planned_rosters(context, seeds[cursor : cursor + 40], MIX_CYCLE)
        cursor += 40
        for plan in arm_plans:
            plans.append({**plan, "arm": arm, "cycle": cycle})
    jobs = collection.prepare_jobs(plans, bound_finals)
    for offset, job in enumerate(jobs):
        job["plan_index"] = start_index + offset
    collection.validate_mix_cycle(jobs)
    return jobs, audit


def provenance(prereg: Path, bound_finals: Mapping[str, Path], reference: Path) -> dict:
    """Immutable source binding embedded in every bank shard."""
    if not prereg.is_file() or not reference.is_dir():
        raise FileNotFoundError("pre-registration and T4c reference bank must exist")
    return {
        "schema": "t4e-return-collection/v1",
        "preregistration": {"path": str(prereg.resolve()), "sha256": _hash(prereg)},
        "reference_bank": {
            "path": str(reference.resolve()),
            "manifest_sha256": _hash(reference / "manifest.json"),
        },
        "finals": {
            arm: {"path": str(path), "sha256": _hash(path)}
            for arm, path in sorted(bound_finals.items())
        },
        "source_tree": {"git_head": _git_head(), "sha256": source_hashes()},
        "horizon": 5000,
        "branch_depth": 300,
        "target_training_decisive_states": collection.MIN_DECISIVE,
    }


def verify_bound_inputs(source: Mapping[str, Any]) -> None:
    """Rehash every pre-play binding before a completion receipt can be written."""
    prereg = source["preregistration"]
    reference = source["reference_bank"]
    if _hash(Path(prereg["path"])) != prereg["sha256"]:
        raise RuntimeError("pre-registration bytes changed during collection")
    if _hash(Path(reference["path"]) / "manifest.json") != reference["manifest_sha256"]:
        raise RuntimeError("reference-bank manifest changed during collection")
    for arm, value in source["finals"].items():
        if _hash(Path(value["path"])) != value["sha256"]:
            raise RuntimeError(f"B8 final changed during collection: {arm}")
    if source["source_tree"] != {"git_head": _git_head(), "sha256": source_hashes()}:
        raise RuntimeError("bound collection source tree changed before final publication")


def staged_metadata(
    staging: Path, receipts: Iterable[Mapping[str, Any]]
) -> tuple[list[dict], list[dict]]:
    """Load compact decisive/candidate metadata in stable receipt order."""
    decisive: list[dict] = []
    candidates: list[dict] = []
    for receipt in sorted(receipts, key=lambda row: int(row["job"]["plan_index"])):
        for key in ("decisive_spool", "candidate_spool"):
            spool = staging / receipt[key]
            expected = receipt[f"{key}_sha256"]
            if _hash(spool) != expected:
                raise ValueError(f"staged {key} hash mismatch: {spool}")
        for rows, destination in (
            (publisher.load_spool(staging / receipt["decisive_spool"]), decisive),
            (publisher.load_spool(staging / receipt["candidate_spool"]), candidates),
        ):
            # Final publication reloads selected full metadata directly from its per-world
            # spool. Global matching only needs these compact identity/stratum fields.
            for row in rows:
                row.pop("state_packed", None)
                row.pop("state_scalars", None)
                destination.append(row)
    return decisive, candidates


def selection(
    decisive: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], cuts, excluded
):
    """Return deduped decisive records, exact outside selection, and its stable receipt."""
    deduped = collection.dedupe_decisive(decisive)
    outside, receipt = collection.select_outside(deduped, candidates, cuts, excluded)
    return deduped, outside, receipt


def _counts(rows: Iterable[Mapping[str, Any]], jobs: Mapping[int, Mapping[str, Any]]) -> dict:
    result: Counter[str] = Counter()
    for row in rows:
        job = jobs[int(row["plan_index"])]
        result[f"{job['arm']}/{job['mix']}"] += 1
    return dict(sorted(result.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="new create-only collection namespace"
    )
    parser.add_argument("--workers", type=int, default=WORKER_LIMIT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--prereg", type=Path, default=PREREG)
    parser.add_argument("--reference-bank", type=Path, default=REFERENCE_BANK)
    parser.add_argument("--b8-root", type=Path, default=B8_ROOT)
    parser.add_argument(
        "--check", action="store_true", help="print validated resources; do not play worlds"
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    """Collect complete fresh cycles, then delegate final staged filtering/publication."""
    started = time.monotonic()
    if not 1 <= args.workers <= WORKER_LIMIT:
        raise ValueError("--workers must be from 1 through 12")
    root = Path(args.out).resolve()
    if root.exists():
        raise FileExistsError(f"T4e output must be a new namespace: {root}")
    artifacts = Path(args.artifacts).resolve()
    bound_finals = finals(args.b8_root)
    cuts, excluded = collection.historical_cuts(Path(args.reference_bank))
    source = provenance(Path(args.prereg), bound_finals, Path(args.reference_bank))
    if args.check:
        return {
            "status": "CHECK_ONLY",
            "command": " ".join(str(value) for value in os.sys.argv),
            "workers": args.workers,
            "cpu_threads_per_worker": 1,
            "cycle_worlds": CYCLE_WORLDS,
            "mix_counts_per_final": dict(Counter(MIX_CYCLE)),
            "source": source,
        }
    root.mkdir(parents=True)
    namespace = f"t4e-return-collection/2026-10-16/{root.name}"
    _atomic_json(
        root / "binding.json",
        {
            **source,
            "namespace": namespace,
            "workers": args.workers,
            "cpu_threads_per_worker": 1,
            "device": "cpu",
            "command": " ".join(str(value) for value in os.sys.argv),
            "ac_power_at_setup": ac_power(),
        },
    )

    all_receipts: list[dict] = []
    all_jobs: dict[int, dict] = {}
    next_index = 0
    cycle = 0
    final_decisive: list[dict] = []
    final_outside: list[dict] = []
    matching: dict = {}
    while True:
        require_ac()
        jobs, audit = make_cycle_jobs(
            namespace, cycle, next_index, bound_finals, (Path.cwd(), artifacts)
        )
        _atomic_json(
            root / "plans" / f"cycle-{cycle:04d}.json", {"jobs": jobs, "seed_audit": audit}
        )
        receipts = publisher.run_phase_a_to_disk(root, jobs, cuts, source, workers=args.workers)
        all_receipts.extend(receipts)
        all_jobs.update({int(job["plan_index"]): job for job in jobs})
        decisive, candidates = staged_metadata(root / "staging", all_receipts)
        try:
            final_decisive, final_outside, matching = selection(
                decisive, candidates, cuts, excluded
            )
            training_decisive = sum(not bool(row.get("holdout", False)) for row in final_decisive)
            if training_decisive < collection.MIN_DECISIVE:
                cycle += 1
                next_index += CYCLE_WORLDS
                continue
            collection.completion_receipt(final_decisive, final_outside, time.monotonic() - started)
        except RuntimeError as error:
            if "INCOMPLETE_MATCHING" not in str(error):
                raise
            cycle += 1
            next_index += CYCLE_WORLDS
            continue
        break

    selected_decisive = {
        (int(row["plan_index"]), int(row["source_decision"])) for row in final_decisive
    }
    selected_outside = {
        (int(row["plan_index"]), int(row["source_decision"])) for row in final_outside
    }
    published = publisher.publish_selected_staged(
        root, all_jobs, selected_decisive, selected_outside, source, workers=args.workers
    )
    validation = publisher.validate_completed_banks(root, published)
    verify_bound_inputs(source)
    completion = collection.completion_receipt(
        final_decisive, final_outside, time.monotonic() - started
    )
    result = {
        **completion,
        "matching": matching,
        "phase_a_worlds": len(all_receipts),
        "cycles": cycle + 1,
        "holdout_worlds": sum(bool(job["holdout"]) for job in all_jobs.values()),
        "holdout_world_fraction": sum(bool(job["holdout"]) for job in all_jobs.values())
        / len(all_jobs),
        "decisive_by_arm_mix": _counts(final_decisive, all_jobs),
        "outside_by_arm_mix": _counts(final_outside, all_jobs),
        "publication": published,
        "validation": validation,
    }
    _atomic_json(root / "completion.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
