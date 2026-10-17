"""Fail-closed freshness audit for T6 RESTART evaluation rounds 72 and 73.

Rounds 72/73 must be fresh not only relative to the registered evaluation
history, but also relative to all T4e collection worlds and every source world
whose late-game state is present in the immutable T4b snapshot store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.t4e_return_20261016 import seed_audit as t4e_audit  # noqa: E402
from research.t4f_return_20261016 import seed_audit as t4f_audit  # noqa: E402

INITIAL_ROUND = 72
EXTENSION_ROUND = 73
ROUNDS = (INITIAL_ROUND, EXTENSION_ROUND)
COLLECTION_WORLD_COUNT = 1200
SNAPSHOT_COUNT = 10_885
SNAPSHOT_SOURCE_WORLD_COUNT = 319
SNAPSHOT_MANIFEST_SHA256 = "33a8abdbc540046a5f805fe107d1d4082fa729b4ef065fac6afee7d8f0e71eaa"
SNAPSHOT_FILE_RE = re.compile(
    r"^b8_s[0-4]-(?:scripted|frozen|mixed)-c\d+-i\d+-s(?P<seed>\d+)-d\d+\.npz$"
)


def sha256(path: Path) -> str:
    """Return a bounded-block SHA256 for one evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_digest(seeds: Iterable[int]) -> str:
    """Return the canonical digest for a seed set."""
    return hashlib.sha256(",".join(str(seed) for seed in sorted(set(seeds))).encode()).hexdigest()


def _partitioned_t6_rg(
    root: Path, arguments: list[str], excludes: tuple[Path, ...], label: str
) -> tuple[set[str], list[dict[str, str]]]:
    """Search every path, including ignored and hidden evidence, in bounded partitions."""
    excluded = {path.resolve() for path in excludes}
    pending = sorted(root.iterdir(), key=lambda path: path.name)
    hits: set[str] = set()
    progress: list[dict[str, str]] = []
    while pending:
        target = pending.pop(0)
        if target.resolve() in excluded:
            progress.append({"target": str(target), "status": "excluded-exact-receipt"})
            continue
        if target.is_dir() and any(target.resolve() in path.parents for path in excluded):
            pending[0:0] = sorted(target.iterdir(), key=lambda path: path.name)
            progress.append({"target": str(target), "status": "split-exclusion-ancestor"})
            continue
        command = [
            "rg",
            "--no-config",
            "--no-ignore",
            "--hidden",
            "--no-require-git",
            "--threads",
            "1",
            "--no-filename",
            "--no-heading",
            "--color",
            "never",
            *arguments,
            "--",
            str(target),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            if not target.is_dir():
                raise RuntimeError(f"T6 {label} scan timed out on indivisible file: {target}")
            children = sorted(target.iterdir(), key=lambda path: path.name)
            if not children:
                raise RuntimeError(f"T6 {label} scan timed out on empty directory: {target}")
            pending[0:0] = children
            progress.append({"target": str(target), "status": "timeout-split"})
            continue
        if result.returncode not in (0, 1):
            raise RuntimeError(f"T6 {label} scan failed: {result.stderr[-2000:]}")
        lines = result.stdout.splitlines()
        hits.update(lines)
        progress.append(
            {"target": str(target), "status": "complete", "match_count": str(len(lines))}
        )
    return hits, progress


def concrete_text_hits(
    root: Path, tokens: list[str], excludes: tuple[Path, ...] = ()
) -> tuple[set[str], list[dict[str, str]]]:
    """Find reserved seed literals without honoring ignore rules."""
    arguments = ["--fixed-strings", "--word-regexp", "--only-matching"]
    for token in tokens:
        arguments.extend(["-e", token])
    return _partitioned_t6_rg(root, arguments, excludes, "concrete-token")


def snapshot_store_audit(store: str | Path) -> tuple[dict[str, Any], set[int]]:
    """Bind the full snapshot manifest and enumerate every source-world seed."""
    root = Path(store).resolve()
    manifest_path = root / "manifest.json"
    result: dict[str, Any] = {
        "path": str(root),
        "manifest_path": str(manifest_path),
        "exists": manifest_path.is_file(),
        "expected_manifest_sha256": SNAPSHOT_MANIFEST_SHA256,
    }
    if not manifest_path.is_file():
        result.update({"pass": False, "error": "snapshot manifest is missing"})
        return result, set()
    actual_sha = sha256(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        result.update({"pass": False, "manifest_sha256": actual_sha, "error": str(error)})
        return result, set()
    snapshots = payload.get("snapshots")
    malformed: list[str] = []
    seeds: set[int] = set()
    if not isinstance(snapshots, list):
        malformed.append("snapshots is not a list")
        snapshots = []
    for row in snapshots:
        name = row.get("file") if isinstance(row, dict) else None
        match = SNAPSHOT_FILE_RE.match(name) if isinstance(name, str) else None
        if match is None:
            malformed.append(str(name))
            continue
        seeds.add(int(match.group("seed")))
    result.update(
        {
            "schema": payload.get("schema"),
            "manifest_sha256": actual_sha,
            "manifest_sha256_matches": actual_sha == SNAPSHOT_MANIFEST_SHA256,
            "snapshot_count": len(snapshots),
            "snapshot_count_matches": len(snapshots) == SNAPSHOT_COUNT,
            "malformed_snapshot_files": malformed,
            "source_world_seed_count": len(seeds),
            "source_world_seed_count_matches": len(seeds) == SNAPSHOT_SOURCE_WORLD_COUNT,
            # This explicit roster makes the exclusion independently reviewable.
            "source_world_seeds": sorted(seeds),
            "source_world_seed_digest": seed_digest(seeds),
        }
    )
    result["pass"] = (
        result["schema"] == "t4-snapshot-store/v2"
        and result["manifest_sha256_matches"]
        and result["snapshot_count_matches"]
        and not malformed
        and result["source_world_seed_count_matches"]
    )
    return result, seeds


def t4e_collection_audit(source: str | Path) -> tuple[dict[str, Any], set[int]]:
    """Verify and enumerate all 1,200 planned T4e collection-world seeds."""
    root = Path(source).resolve()
    base = t4f_audit._source_bank_audit(root)
    plans = sorted((root / "plans").glob("cycle-*.json")) if root.is_dir() else []
    jobs: list[dict[str, Any]] = []
    malformed: list[str] = []
    for path in plans:
        try:
            payload = json.loads(path.read_text())
            rows = payload.get("jobs")
            if not isinstance(rows, list):
                raise ValueError("jobs is not a list")
            for row in rows:
                if not isinstance(row, dict) or type(row.get("seed")) is not int:
                    malformed.append(str(path))
                else:
                    jobs.append(row)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            malformed.append(f"{path}: {error}")
    seeds = {int(row["seed"]) for row in jobs}
    result = {
        "source_bank_audit": base,
        "planned_world_count": len(jobs),
        "planned_world_count_matches": len(jobs) == COLLECTION_WORLD_COUNT,
        "unique_source_world_seed_count": len(seeds),
        "source_world_seed_digest": seed_digest(seeds),
        "malformed_plan_rows": malformed,
    }
    result["pass"] = (
        base.get("pass") is True
        and result["planned_world_count_matches"]
        and len(seeds) == COLLECTION_WORLD_COUNT
        and not malformed
    )
    return result, seeds


def training_namespace_audit(
    repo: Path, artifact_root: Path, candidate_seeds: set[int]
) -> dict[str, Any]:
    """Check candidates against bounded, recorded M3/M3C training-world prefixes."""
    from research.apex_safety_20260926 import dev_screen

    records: dict[tuple[str, int, int, int, int, float, int], list[dict[str, str]]] = {}
    unsupported: list[str] = []
    for root in (repo / "research", artifact_root):
        for path in sorted(root.rglob("summary.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("schema") not in {
                "redesign-m3-train/v1",
                "redesign-m3c-train/v1",
            }:
                continue
            args = payload.get("args")
            if not isinstance(args, dict):
                unsupported.append(str(path))
                continue
            try:
                namespace, seed = str(args["namespace"]), int(args["seed"])
                worlds, requested = int(args["worlds"]), int(args["transitions"])
                segment_frames = int(args["segment_frames"])
                actors, queue = int(args.get("actors", 0)), int(args.get("actor_queue_frames", 0))
                restart = float(args.get("restart_frac", 0.0) or 0.0)
                actual = int(payload.get("transitions", 0) or 0)
            except (KeyError, TypeError, ValueError):
                unsupported.append(str(path))
                continue
            records.setdefault(
                (namespace, seed, worlds, segment_frames, max(requested, actual), restart, actors),
                [],
            ).append({"path": str(path), "sha256": sha256(path), "queue_frames": str(queue)})
    # The fixed unstarted T6 arm is a bounded planned training input, not an
    # unbounded future namespace reservation.
    for seed in (10, 11, 12, 13, 14):
        records.setdefault(
            ("redesign-m3-train/v1", seed, 128, 5_000, 8_000_000, 0.2, 2), []
        ).append(
            {
                "path": "docs/research/specs/t6_prereg_2026-10-16.md",
                "sha256": "preregistered",
                "queue_frames": "8",
            }
        )
    rows: list[dict[str, Any]] = []
    all_collisions: set[int] = set()
    for (namespace, seed, worlds, segment_frames, budget, restart, actors), sources in sorted(
        records.items()
    ):
        queue = max(int(source["queue_frames"]) for source in sources)
        workers = max(1, actors)
        fresh = worlds - round(worlds * restart)
        if fresh <= 0:
            unsupported.extend(source["path"] for source in sources)
            continue
        prefetch = workers * (queue + 1) * worlds
        # A fresh segment guarantees only its initial fresh-world decisions:
        # after deaths, respawn waits and placement failures can suppress later
        # decisions.  Do not multiply that minimum by segment_frames.
        completed_segments = (budget + workers * worlds + prefetch) // fresh
        budget_highest_index = workers * (completed_segments + 1) * worlds - 1
        generated: set[int] = set()
        collisions: list[int] = []
        first_duplicate_index: int | None = None
        first_duplicate_seed: int | None = None
        for index in range(budget_highest_index + 1):
            seed_value = dev_screen.uint32_seed(namespace, f"seed{seed}", index)
            if seed_value in generated:
                first_duplicate_index, first_duplicate_seed = index, seed_value
                break
            generated.add(seed_value)
            if seed_value in candidate_seeds:
                collisions.append(seed_value)
        all_collisions.update(collisions)
        rows.append(
            {
                "namespace": namespace,
                "run_seed": seed,
                "worlds": worlds,
                "segment_frames": segment_frames,
                "decision_budget_bound": budget,
                "restart_fraction": restart,
                "actors": workers,
                "queue_frames": queue,
                "fresh_world_decisions_per_segment_min": fresh,
                "prefetch_decision_bound": prefetch,
                "budget_highest_generated_world_index": budget_highest_index,
                "first_duplicate_index": first_duplicate_index,
                "first_duplicate_seed": first_duplicate_seed,
                "checked_unique_prefix_count": len(generated),
                "candidate_collisions": sorted(collisions),
                "sources": sources,
            }
        )
    return {
        "status": "PASS" if rows and not all_collisions else "FAIL",
        "pass": bool(rows) and not all_collisions,
        "records": rows,
        "candidate_collisions": sorted(all_collisions),
        "unsupported_summaries": sorted(set(unsupported)),
        "limit": (
            "Bounded recorded summaries and fixed T6 plans only; unrecorded histories remain "
            "outside this inventory."
        ),
    }


def _reserved_round_metadata(
    root: Path, excludes: tuple[Path, ...]
) -> tuple[bool, list[dict[str, str]]]:
    """Find already-recorded T6 fresh-round metadata in JSON evidence."""
    hits, progress = _partitioned_t6_rg(
        root,
        [
            "--pcre2",
            r'"(?:round_index|round)"\s*:\s*(?:72|73)(?![0-9])',
            "--glob",
            "*.json",
            "--glob",
            "*.jsonl",
            "--only-matching",
        ],
        excludes,
        "round-metadata",
    )
    return bool(hits), progress


def seed_bank_audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    snapshot_store: str | Path | None = None,
    t4e_source_bank: str | Path | None = None,
    exclude_out: str | Path | None = None,
) -> dict[str, Any]:
    """Audit fresh T6 rounds 72/73 with only the exact output self-excluded."""
    from research.redesign_m2_20261008 import ni_spec

    repo_path = Path(repo).resolve()
    artifact_root = (
        Path(artifacts).resolve()
        if artifacts is not None
        else repo_path.parent / "snake-dqn-artifacts"
    )
    store = (
        Path(snapshot_store).resolve()
        if snapshot_store is not None
        else artifact_root / "t4b-snapshot-restart-20261016/full"
    )
    source = (
        Path(t4e_source_bank).resolve()
        if t4e_source_bank is not None
        else artifact_root / "t4e-return-bank-20261016"
    )
    output = Path(exclude_out).resolve() if exclude_out is not None else None
    if output is not None and (output.is_dir() or output.suffix != ".json"):
        raise ValueError("exclude_out must name exactly one JSON audit receipt")
    roots = (repo_path / "research", artifact_root)
    if not all(root.is_dir() for root in roots):
        raise ValueError("repository research and artifact root must both exist")

    t2, registered, history = t4f_audit._registered(repo_path)
    consumed: dict[str, set[int]] = {
        "t4f_consumed_round_70": set(ni_spec.distill_seeds(96, round_index=70)),
        "t4f_consumed_round_71": set(ni_spec.distill_seeds(96, round_index=71)),
    }
    collection, collection_seeds = t4e_collection_audit(source)
    snapshot, snapshot_seeds = snapshot_store_audit(store)
    consumed["t4e_collection_planned_worlds"] = collection_seeds
    consumed["t4b_snapshot_source_worlds"] = snapshot_seeds
    for values in consumed.values():
        registered.update(values)

    expected = {
        str(round_index): set(ni_spec.distill_seeds(96, round_index=round_index))
        for round_index in ROUNDS
    }
    reserved = set().union(*expected.values())
    training = training_namespace_audit(repo_path, artifact_root, reserved)
    excluded = (output,) if output is not None else ()
    scans = {
        str(root): concrete_text_hits(root, [str(seed) for seed in sorted(reserved)], excluded)
        for root in roots
    }
    observed = {
        int(token) for hits, _progress in scans.values() for token in hits if token.isdigit()
    }
    metadata_scans = {str(root): _reserved_round_metadata(root, excluded) for root in roots}
    metadata_found = {name: found for name, (found, _progress) in metadata_scans.items()}

    namespace_path = artifact_root / "t3-rollout-labels-20261015/bank/namespace.json"
    namespace: dict[str, Any] = {"exists": False, "manifest_binding": {"ok": False}}
    if namespace_path.is_file():
        raw = json.loads(namespace_path.read_text())
        values = raw.get("seeds", [])
        namespace = {
            "exists": True,
            "schema": raw.get("schema"),
            "count": len(values) if isinstance(values, list) else 0,
            "unique": isinstance(values, list) and len(values) == len(set(values)),
            "manifest_binding": t4e_audit.prior_banks._manifest_binding(
                namespace_path.parent, namespace_path
            ),
        }
        registered.update(seed for seed in values if type(seed) is int and seed >= 0)
    namespace_valid = (
        namespace["exists"]
        and namespace.get("schema") == "t3-label-namespace/v1"
        and bool(namespace.get("count"))
        and namespace.get("unique") is True
        and namespace["manifest_binding"].get("ok") is True
    )
    checks: dict[str, dict[str, Any]] = {}
    for round_index, seeds in expected.items():
        other = set().union(*(values for key, values in expected.items() if key != round_index))
        checks[round_index] = {
            "count": len(seeds),
            "unique": len(seeds) == 96,
            "seed_digest": seed_digest(seeds),
            "seeds": sorted(seeds),
            "registered_collisions": sorted(seeds & registered),
            "observed_json_collisions": sorted(seeds & observed),
            "other_reserved_round_collisions": sorted(seeds & other),
        }
    passed = (
        t2["pass"]
        and namespace_valid
        and collection["pass"]
        and snapshot["pass"]
        and training["pass"]
        and not any(metadata_found.values())
        and all(
            check["unique"]
            and not check["registered_collisions"]
            and not check["observed_json_collisions"]
            and not check["other_reserved_round_collisions"]
            for check in checks.values()
        )
    )
    return {
        "schema": "t6-restart-seed-bank-audit/v1",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "audit_source_sha256": sha256(Path(__file__).resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "fresh rounds 72 and 73; prior registered T3/T4 history; consumed rounds 68--71; "
            "all 1,200 T4e planned collection worlds and all source worlds in the bound full "
            "T4b snapshot manifest"
        ),
        "excluded_output": str(output) if output is not None else None,
        "exclusion_policy": "Only the exact audit receipt being written is self-excluded.",
        "registered_t2_audit": {"pass": t2["pass"], "schema": t2["schema"]},
        "registered_history": {
            name: {"count": len(values), "seed_digest": seed_digest(values)}
            for name, values in history.items()
        },
        "consumed_namespaces": {
            name: {"count": len(values), "seed_digest": seed_digest(values)}
            for name, values in consumed.items()
        },
        "t3_rollout_training_namespace": namespace,
        "t4e_collection": collection,
        "snapshot_store": snapshot,
        "training_world_inventory": training,
        "text_scan": {
            "method": "t6-partitioned-rg-no-ignore-hidden/threads1/timeout60",
            "roots": [str(root) for root in roots],
            "candidate_count": len(reserved),
            "hits": {name: sorted(hits) for name, (hits, _progress) in scans.items()},
            "reserved_round_metadata": metadata_found,
            "progress": {name: progress for name, (_hits, progress) in scans.items()},
            "metadata_progress": {
                name: progress for name, (_found, progress) in metadata_scans.items()
            },
        },
        "checks": checks,
        "limit": "No claim is made about unrecorded experiments or external systems.",
    }


def main() -> int:
    """Write the exact T6 receipt, excluding only that path from its own scan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot-store", type=Path)
    parser.add_argument("--t4e-source-bank", type=Path)
    args = parser.parse_args()
    receipt = seed_bank_audit(
        snapshot_store=args.snapshot_store,
        t4e_source_bank=args.t4e_source_bank,
        exclude_out=args.out,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": receipt["pass"], "checks": receipt["checks"]}, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
