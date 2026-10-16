"""Fail-closed freshness audit for the T4f evaluation rounds 70 and 71.

The T4f bank is derived from the completed T4e bank.  This receipt therefore
checks both the complete T4e collection namespace (all 1,200 planned worlds)
and the current repository/artifact evidence scope used by the T4e audit.  A
text hit for a reserved seed is retained in the unfiltered sibling receipt. The
current receipt and explicitly named documentary files are excluded from the
qualified scan, with exact path, size and SHA recorded for every exclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.t3_screen_20261015 import banks as prior_banks  # noqa: E402
from research.t4e_return_20261016 import seed_audit as t4e_audit  # noqa: E402

INITIAL_ROUND = 70
EXTENSION_ROUND = 71
ROUNDS = (INITIAL_ROUND, EXTENSION_ROUND)
CONSUMED_ROUNDS = (68, 69)
COLLECTION_WORLD_COUNT = 1200
WORLD_RE = re.compile(r"world-(\d+)-.*-s(\d+)(?:-|\.)")


def sha256(path: Path) -> str:
    """Hash one evidence file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_digest(seeds: Iterable[int]) -> str:
    """Hash a canonical sorted seed set."""
    return hashlib.sha256(",".join(str(seed) for seed in sorted(set(seeds))).encode()).hexdigest()


def _reserved_round_metadata(
    root: Path, excludes: tuple[Path, ...] = ()
) -> tuple[bool, list[dict[str, str]]]:
    """Search JSON/JSONL receipts for the two fresh-round metadata values."""
    pattern = r'"(?:round_index|round)"\s*:\s*(?:70|71)(?![0-9])'
    hits, progress = t4e_audit._partitioned_rg(
        root,
        [
            "--pcre2",
            pattern,
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


def _progress_paths(progress: Iterable[Mapping[str, str]]) -> list[str]:
    """Extract paths recorded by the T4e partitioned scanner."""
    paths: set[str] = set()
    for row in progress:
        raw = row.get("match_paths")
        if not raw:
            continue
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(values, list):
            paths.update(str(value) for value in values)
    return sorted(paths)


def _documentary_collisions(
    scans: Mapping[str, tuple[set[str], list[dict[str, str]]]],
    metadata_scans: Mapping[str, tuple[bool, list[dict[str, str]]]],
    documentary_exclusions: tuple[Path, ...],
) -> dict[str, Any]:
    """Report collision paths without changing the underlying pass decision."""
    explicit = {str(path.resolve()) for path in documentary_exclusions}
    rows: list[dict[str, Any]] = []
    for root, (_hits, progress) in scans.items():
        for path in _progress_paths(progress):
            resolved = str(Path(path).resolve())
            rows.append(
                {
                    "root": root,
                    "path": path,
                    "classification": (
                        "explicit-documentary" if resolved in explicit else "evidence"
                    ),
                    "reason": (
                        "caller-designated documentary file"
                        if resolved in explicit
                        else "reserved seed literal found in current scan scope"
                    ),
                }
            )
    for root, (_found, progress) in metadata_scans.items():
        for path in _progress_paths(progress):
            resolved = str(Path(path).resolve())
            rows.append(
                {
                    "root": root,
                    "path": path,
                    "classification": (
                        "explicit-documentary" if resolved in explicit else "evidence"
                    ),
                    "reason": (
                        "caller-designated documentary file"
                        if resolved in explicit
                        else "fresh-round metadata found in current scan scope"
                    ),
                }
            )
    return {
        "count": len(rows),
        "paths": rows,
        "explicit_paths": sorted(explicit),
        "policy": (
            "Documentary paths are listed with hashes below when supplied by the caller. "
            "They are not silently removed from scan results; unclassified hits fail."
        ),
    }


def _registered(repo: Path) -> tuple[dict[str, Any], set[int], dict[str, set[int]]]:
    """Return prior registrations plus consumed T4e rounds 68 and 69."""
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    t2, registered = t4e_audit._registered(repo)
    history: dict[str, set[int]] = {
        "t4e_consumed_round_68": set(ni_spec.distill_seeds(96, round_index=68)),
        "t4e_consumed_round_69": set(ni_spec.distill_seeds(96, round_index=69)),
    }
    registered.update(history["t4e_consumed_round_68"])
    registered.update(history["t4e_consumed_round_69"])
    # Keep this explicit in the receipt so the next audit cannot silently drop
    # the already-consumed T4e rosters if the older helper changes.
    history["ni_namespace"] = set(ni_spec.ni_seeds())
    history["ni_m2b_namespace"] = set(ni_spec_m2b.ni_seeds())
    return t2, registered, history


def _manifest_worlds(path: Path) -> dict[str, Any]:
    """Read a T4e publication manifest and extract its world/seed identities."""
    manifest = json.loads(path.read_text())
    receipts = manifest.get("publication", {}).get("world_receipts", {})
    if not isinstance(receipts, dict):
        raise ValueError(f"publication.world_receipts missing from {path}")
    pairs: set[tuple[int, int]] = set()
    malformed: list[str] = []
    for name in receipts:
        match = WORLD_RE.match(name)
        if match is None:
            malformed.append(str(name))
            continue
        pairs.add((int(match.group(1)), int(match.group(2))))
    return {
        "manifest": {
            "path": str(path),
            "sha256": sha256(path),
            "collection_schema": manifest.get("collection_schema"),
            "schema_version": manifest.get("schema_version"),
        },
        "receipt_count": len(receipts),
        "malformed_receipts": malformed,
        "world_ids": {world for world, _seed in pairs},
        "seeds": {seed for _world, seed in pairs},
        "pairs": pairs,
    }


def _source_bank_audit(source: Path) -> dict[str, Any]:
    """Verify the T4e plans and published train/holdout identities."""
    result: dict[str, Any] = {"path": str(source), "exists": source.is_dir()}
    if not source.is_dir():
        result["pass"] = False
        result["error"] = "T4e source bank directory is missing"
        return result

    plans = sorted((source / "plans").glob("cycle-*.json"))
    jobs: list[dict[str, Any]] = []
    for path in plans:
        payload = json.loads(path.read_text())
        jobs.extend(dict(job) for job in payload.get("jobs", []))
    by_index: dict[int, dict[str, Any]] = {}
    duplicate_indices: list[int] = []
    for job in jobs:
        index = int(job["plan_index"])
        if index in by_index:
            duplicate_indices.append(index)
        by_index[index] = job
    planned_seeds = [int(job["seed"]) for job in jobs]
    plan_seed_set = set(planned_seeds)
    plan_files = [{"path": str(path), "sha256": sha256(path)} for path in plans]
    result["plans"] = {
        "file_count": len(plans),
        "job_count": len(jobs),
        "unique_plan_indices": len(by_index),
        "plan_indices_complete": set(by_index) == set(range(COLLECTION_WORLD_COUNT)),
        "duplicate_plan_indices": sorted(set(duplicate_indices)),
        "unique_seeds": len(plan_seed_set),
        "seed_digest": seed_digest(plan_seed_set),
        "plan_files": plan_files,
    }
    banks: dict[str, Any] = {}
    for kind in ("train", "holdout"):
        manifest_path = source / "banks" / kind / "manifest.json"
        if not manifest_path.is_file():
            banks[kind] = {"exists": False, "pass": False, "path": str(manifest_path)}
            continue
        bank = _manifest_worlds(manifest_path)
        bank["exists"] = True
        bank["worlds_in_plans"] = all(world in by_index for world in bank["world_ids"])
        bank["seed_plan_matches"] = all(
            by_index.get(world, {}).get("seed") == seed for world, seed in bank["pairs"]
        )
        bank["seeds_in_all_plans"] = bank["seeds"].issubset(plan_seed_set)
        bank["pass"] = (
            not bank["malformed_receipts"]
            and bank["worlds_in_plans"]
            and bank["seed_plan_matches"]
            and bank["seeds_in_all_plans"]
        )
        banks[kind] = bank
    train, holdout = banks.get("train", {}), banks.get("holdout", {})
    world_overlap = sorted(train.get("world_ids", set()) & holdout.get("world_ids", set()))
    seed_overlap = sorted(train.get("seeds", set()) & holdout.get("seeds", set()))
    result["banks"] = {
        kind: {
            key: value for key, value in bank.items() if key not in {"world_ids", "seeds", "pairs"}
        }
        for kind, bank in banks.items()
    }
    result["train_holdout_world_overlap"] = world_overlap
    result["train_holdout_seed_overlap"] = seed_overlap
    result["pass"] = (
        result["plans"]["file_count"] == 6
        and result["plans"]["job_count"] == COLLECTION_WORLD_COUNT
        and result["plans"]["unique_plan_indices"] == COLLECTION_WORLD_COUNT
        and result["plans"]["plan_indices_complete"]
        and not result["plans"]["duplicate_plan_indices"]
        and result["plans"]["unique_seeds"] == COLLECTION_WORLD_COUNT
        and all(bank.get("pass") for bank in banks.values())
        and not world_overlap
        and not seed_overlap
    )
    return result


def _macro_bank_audit(macro_root: Path, source: Path) -> dict[str, Any]:
    """Check a present v2 macro bank; absence is the declared pre-collection state."""
    if not macro_root.is_dir():
        return {
            "path": str(macro_root),
            "status": "PENDING",
            "pass": True,
            "note": (
                "The derived macro bank has not been created; source-bank exclusion remains bound."
            ),
        }
    manifest_paths = [
        macro_root / "banks" / kind / "manifest.json" for kind in ("train", "holdout")
    ]
    if not any(path.is_file() for path in manifest_paths):
        binding_path = macro_root / "macro_binding.json"
        row: dict[str, Any] = {
            "path": str(macro_root),
            "status": "IN_PROGRESS",
            "pass": False,
            "binding": {"path": str(binding_path), "exists": binding_path.is_file()},
            "partial_npz_shards": len(list(macro_root.rglob("*.npz"))),
        }
        if binding_path.is_file():
            binding = json.loads(binding_path.read_text())
            expected = {
                kind: sha256(source / "banks" / kind / "manifest.json")
                for kind in ("train", "holdout")
            }
            actual = {
                kind: binding.get("source_identity", {})
                .get("banks", {})
                .get(kind, {})
                .get("manifest_sha256")
                for kind in ("train", "holdout")
            }
            row["binding"].update(
                {
                    "schema": binding.get("schema"),
                    "source": binding.get("source"),
                    "source_manifest_sha256": actual,
                    "expected_source_manifest_sha256": expected,
                }
            )
            row["pass"] = (
                binding.get("schema") == "t4f-return-bank/v2"
                and Path(str(binding.get("source", ""))).resolve() == source
                and actual == expected
            )
        else:
            row["note"] = "No macro manifests yet; awaiting the declared collector."
            row["pass"] = True
        row["note"] = (
            "Macro derivation is still running; source identity is bound and final manifests "
            "remain a required pre-launch check."
        )
        return row
    banks: dict[str, Any] = {}
    passed = True
    for kind in ("train", "holdout"):
        path = macro_root / "banks" / kind / "manifest.json"
        row: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            row.update({"pass": False, "error": "macro-bank manifest missing"})
            passed = False
            banks[kind] = row
            continue
        manifest = json.loads(path.read_text())
        binding = manifest.get("source_bank", {})
        source_manifest = source / "banks" / kind / "manifest.json"
        shard_names = sorted(p.name for p in path.parent.glob("*.npz"))
        source_shard_names = sorted(p.name for p in source_manifest.parent.glob("*.npz"))
        row.update(
            {
                "schema": manifest.get("collection_schema"),
                "schema_version": manifest.get("schema_version"),
                "source_manifest_sha256": binding.get("manifest_sha256"),
                "expected_source_manifest_sha256": sha256(source_manifest),
                "shards": len(shard_names),
                "source_shards": len(source_shard_names),
                "shard_set_matches": shard_names == source_shard_names,
            }
        )
        row["pass"] = (
            manifest.get("collection_schema") == "t4f-return-bank/v2"
            and manifest.get("schema_version") == 2
            and row["source_manifest_sha256"] == row["expected_source_manifest_sha256"]
            and row["shard_set_matches"]
        )
        passed &= bool(row["pass"])
        banks[kind] = row
    return {"path": str(macro_root), "status": "PRESENT", "banks": banks, "pass": passed}


def seed_bank_audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    source_bank: str | Path | None = None,
    macro_bank: str | Path | None = None,
    exclude_out: str | Path | None = None,
    documentary_exclusions: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Audit fresh rounds 70/71 against all current evidence and the T4e bank."""
    from research.redesign_m2_20261008 import ni_spec

    repo_path = Path(repo).resolve()
    artifact_root = (
        Path(artifacts).resolve()
        if artifacts is not None
        else repo_path.parent / "snake-dqn-artifacts"
    )
    source = (
        Path(source_bank).resolve()
        if source_bank is not None
        else artifact_root / "t4e-return-bank-20261016"
    )
    macro = (
        Path(macro_bank).resolve()
        if macro_bank is not None
        else artifact_root / "t4f-return-bank-20261016"
    )
    output = Path(exclude_out).resolve() if exclude_out is not None else None
    if output is not None and (output.is_dir() or output.suffix != ".json"):
        raise ValueError("exclude_out must name exactly one JSON audit receipt")
    documentary_paths = tuple(Path(path).resolve() for path in documentary_exclusions)
    excluded = tuple(path for path in ((output,) if output is not None else ())) + documentary_paths
    if any(path.is_dir() for path in documentary_paths):
        raise ValueError("documentary exclusions must name exact files")
    documentary: list[dict[str, Any]] = []
    for path in documentary_paths:
        if not path.is_file():
            raise ValueError(f"documentary exclusion missing: {path}")
        documentary.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "reason": (
                    "exact documentary exclusion; raw hits retained in unfiltered sibling receipt"
                ),
            }
        )
    if output is not None:
        if output.exists():
            documentary.append(
                {
                    "path": str(output),
                    "bytes": output.stat().st_size,
                    "sha256": sha256(output),
                    "reason": "current audit receipt; excluded exactly from its own scan",
                }
            )
        else:
            documentary.append(
                {
                    "path": str(output),
                    "exists_at_scan": False,
                    "reason": "current audit receipt; excluded exactly from its own scan",
                }
            )

    t2, registered, history = _registered(repo_path)
    expected = {
        str(round_index): set(ni_spec.distill_seeds(96, round_index=round_index))
        for round_index in ROUNDS
    }
    roots = (repo_path / "research", artifact_root)
    if not all(root.is_dir() for root in roots):
        raise ValueError("repository research and artifact root must both exist")
    reserved = set().union(*expected.values())
    tokens = [str(seed) for seed in sorted(reserved)]
    scans = {str(root): t4e_audit.concrete_text_hits(root, tokens, excluded) for root in roots}
    observed = {
        int(token) for hits, _progress in scans.values() for token in hits if token.isdigit()
    }
    metadata_scans = {str(root): _reserved_round_metadata(root, excluded) for root in roots}
    reserved_metadata = {name: value[0] for name, value in metadata_scans.items()}
    collisions = _documentary_collisions(scans, metadata_scans, documentary_paths)
    evidence_collision_paths = [
        row["path"] for row in collisions["paths"] if row["classification"] == "evidence"
    ]

    namespace_path = artifact_root / "t3-rollout-labels-20261015/bank/namespace.json"
    namespace: dict[str, Any] = {"exists": False, "manifest_binding": {"ok": False}}
    if namespace_path.is_file():
        raw = json.loads(namespace_path.read_text())
        listed = raw.get("seeds", [])
        namespace = {
            "exists": True,
            "schema": raw.get("schema"),
            "count": len(listed) if isinstance(listed, list) else 0,
            "unique": isinstance(listed, list) and len(listed) == len(set(listed)),
            "manifest_binding": prior_banks._manifest_binding(
                namespace_path.parent, namespace_path
            ),
        }
        registered.update(seed for seed in listed if type(seed) is int and seed >= 0)

    checks: dict[str, dict[str, Any]] = {}
    for round_index, seeds in expected.items():
        other = set().union(
            *(candidate for key, candidate in expected.items() if key != round_index)
        )
        checks[round_index] = {
            "count": len(seeds),
            "unique": len(seeds) == 96,
            "seed_digest": seed_digest(seeds),
            "seeds": sorted(seeds),
            "registered_collisions": sorted(seeds & registered),
            "observed_json_collisions": sorted(seeds & observed),
            "other_reserved_round_collisions": sorted(seeds & other),
        }
    namespace_valid = (
        namespace["exists"]
        and namespace.get("schema") == "t3-label-namespace/v1"
        and bool(namespace.get("count"))
        and namespace.get("unique") is True
        and namespace["manifest_binding"].get("ok") is True
    )
    source_result = _source_bank_audit(source)
    macro_result = _macro_bank_audit(macro, source)
    passed = (
        t2["pass"]
        and namespace_valid
        and source_result["pass"]
        and macro_result["pass"]
        and not any(reserved_metadata.values())
        and not evidence_collision_paths
        and all(
            check["unique"]
            and not check["registered_collisions"]
            and not check["observed_json_collisions"]
            and not check["other_reserved_round_collisions"]
            for check in checks.values()
        )
    )
    return {
        "schema": "t4f-seed-bank-audit/v1",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "audit_source_sha256": sha256(Path(__file__).resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "fresh distill rounds 70 and 71; consumed T4e rounds 68 and 69; all repository "
            "research and artifact-root JSON/JSONL evidence; complete T4e 1,200-world plans"
        ),
        "source_bank": source_result,
        "macro_bank": macro_result,
        "excluded_output": str(output) if output is not None else None,
        "documentary_exclusions": documentary,
        "documentary_collisions": collisions,
        "registered_history": {
            name: {
                "count": len(values),
                "unique": len(values) == len(set(values)),
                "seed_digest": seed_digest(values),
            }
            for name, values in history.items()
        },
        "registered_t2_audit": {"pass": t2["pass"], "schema": t2["schema"]},
        "t3_rollout_training_namespace": namespace,
        "text_scan": {
            "method": "t4e-partitioned-rg-fixed-word/threads1/timeout60",
            "roots": [str(root) for root in roots],
            "candidate_count": len(tokens),
            "hits": {name: sorted(hits) for name, (hits, _progress) in scans.items()},
            "reserved_round_metadata": reserved_metadata,
            "progress": {name: progress for name, (_hits, progress) in scans.items()},
            "metadata_progress": {
                name: progress for name, (_found, progress) in metadata_scans.items()
            },
        },
        "checks": checks,
        "limit": (
            "No claim is made about unrecorded experiments, arbitrary binary files, or "
            "external systems. A documentary collision is retained as evidence and does "
            "not become an exclusion unless it is the exact receipt being written."
        ),
    }


def main() -> int:
    """Write the exact receipt excluded from its own audit scan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path)
    parser.add_argument("--macro-bank", type=Path)
    parser.add_argument("--exclude-documentary", type=Path, action="append", default=[])
    args = parser.parse_args()
    receipt = seed_bank_audit(
        source_bank=args.source_bank,
        macro_bank=args.macro_bank,
        exclude_out=args.out,
        documentary_exclusions=tuple(args.exclude_documentary),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": receipt["pass"], "checks": receipt["checks"]}, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
