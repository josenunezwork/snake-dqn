"""Fresh-world audit for the T4c evaluation rounds.

The audit scans repository research and all artifact-root JSON/JSONL files,
including root-level files. Only its own documentary receipt is excluded;
actual probe results, including this study's results, are always counted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.t3_screen_20261015 import banks as prior_banks


def inventory_json(
    root: Path, *, exclude: Path | None = None, prune: tuple[Path, ...] = ()
) -> dict[str, Any]:
    """Collect explicit seed fields during JSON decoding, skipping a second tree walk.

    JSONL is streamed one record at a time. Every byte still contributes to the
    inventory digest. Malformed files are reported for explicit separate byte-coverage checks;
    they are never silently skipped or counted as valid JSON. This deliberately
    inspects nested explicit seed fields even inside metric mappings, a
    conservative superset of the historical inventory helper.
    """
    if not root.is_dir():
        return {
            "exists": False,
            "file_count": 0,
            "seed_count": 0,
            "seeds": [],
            "digest": None,
            "invalid_json": [],
        }
    seeds: set[int] = set()
    metrics = {
        "mass",
        "deaths",
        "alive_frames",
        "death_frame",
        "length_at_death",
        "death_lengths",
        "causes",
    }

    def seed_fields(value: dict) -> dict:
        for key, child in value.items():
            if key in {"seed", "world_seed"} and type(child) is int and child >= 0:
                seeds.add(child)
            elif key == "world_seeds" and isinstance(child, list):
                seeds.update(item for item in child if type(item) is int and item >= 0)
            elif key in metrics and isinstance(child, dict):
                seeds.update(int(item) for item in child if str(item).isdigit())
        return value

    digest = hashlib.sha256()
    valid = count = 0
    invalid = []
    excluded = tuple(path.resolve() for path in prune)

    def omitted(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == item or item in resolved.parents for item in excluded)

    # Do not use Path.rglob here: an ignored aborted study must be pruned before
    # discovery, rather than merely omitted after its files have been enumerated.
    paths: list[Path] = []
    for parent, directories, files in os.walk(root):
        parent_path = Path(parent)
        directories[:] = sorted(name for name in directories if not omitted(parent_path / name))
        for name in sorted(files):
            path = parent_path / name
            if (
                path.suffix in {".json", ".jsonl"}
                and not omitted(path)
                and path.is_file()
                and path != exclude
            ):
                paths.append(path)
    for path in paths:
        count += 1
        file_digest = hashlib.sha256()
        bad = False
        if path.suffix == ".jsonl":
            with path.open("rb") as stream:
                for line in stream:
                    file_digest.update(line)
                    if line.strip():
                        try:
                            json.loads(line, object_hook=seed_fields)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            bad = True
        else:
            raw = path.read_bytes()
            file_digest.update(raw)
            try:
                json.loads(raw, object_hook=seed_fields)
            except (UnicodeDecodeError, json.JSONDecodeError):
                bad = True
        digest.update(str(path.relative_to(root)).encode() + b":" + file_digest.digest())
        if bad:
            invalid.append(str(path))
        else:
            valid += 1
    return {
        "exists": True,
        "file_count": count,
        "valid_json_count": valid,
        "seed_count": len(seeds),
        "seeds": sorted(seeds),
        "digest": digest.hexdigest(),
        "invalid_json": invalid,
    }


def malformed_byte_coverage(paths: list[str], reserved: set[int]) -> dict[str, Any]:
    """Check complete reserved IDs in retained malformed ASCII bytes, never missing bytes."""
    results = []
    needles = {str(seed).encode("ascii"): seed for seed in reserved}
    round_pattern = re.compile(rb'"(?:round_index|round)"\s*:\s*(?:66|67)(?![0-9])')
    for name in paths:
        raw = Path(name).read_bytes()  # one immutable bytes object binds search and digest
        try:
            json.loads(raw)
            error = None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = {"reason": str(exc), "offset": getattr(exc, "pos", None)}
        ascii_compatible = raw.isascii() and b"\x00" not in raw
        escapes = b"\\u" in raw or b"\\U" in raw
        matches = sorted(seed for literal, seed in needles.items() if literal in raw)
        round_metadata = bool(round_pattern.search(raw))
        results.append(
            {
                "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "parse_error": error,
                "ascii_without_nul": ascii_compatible,
                "unicode_escape": escapes,
                "reserved_literals": matches,
                "reserved_round_metadata": round_metadata,
                "pass": ascii_compatible and not escapes and not matches and not round_metadata,
            }
        )
    return {
        "pass": all(item["pass"] for item in results),
        "files": results,
        "limit": "Negative byte coverage proves no complete reserved integer ID in retained "
        "ASCII bytes. It does not prove that missing/truncated records never used "
        "these worlds. Malformed files are not treated as valid JSON.",
    }


def seed_bank_audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    exclude_out: str | Path | None = None,
) -> dict[str, Any]:
    """Audit reserved rounds 66/67 against the registered and observed scope."""
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    repo_path = Path(repo).resolve()
    artifact_root = (
        Path(artifacts).resolve()
        if artifacts is not None
        else repo_path.parent / "snake-dqn-artifacts"
    )
    output = Path(exclude_out).resolve() if exclude_out is not None else None
    if output is not None and (output.is_dir() or output.suffix != ".json"):
        raise ValueError("exclude_out must name one JSON audit receipt, never a directory")
    t2 = prior_banks._registered_t2(repo_path)
    expected = {
        str(round_index): set(ni_spec.distill_seeds(96, round_index=round_index))
        for round_index in (66, 67)
    }
    registered = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds())
    for values in t2["registered_rounds"].values():
        registered.update(values)
    for item in t2["historical_distillation"].values():
        registered.update(item["seeds"])
    for round_index in (62, 63, 64, 65):
        registered.update(ni_spec.distill_seeds(96, round_index=round_index))

    repository_results = repo_path / "research/t4c_20261016/results/seed_bank_audit.json"
    aborted = artifact_root / "redesign-t4c-mac-20261016/ABORTED_cpu_TWO_SIDED_T_s10"
    forbidden_worktree = Path("/Users/josenunez/Projects/ml/snake-dqn-t4c")
    roots = [repo_path / "research", artifact_root]
    inventories = {
        str(root): inventory_json(
            root,
            exclude=(repository_results if root == repo_path / "research" else output),
            prune=(aborted, forbidden_worktree),
        )
        for root in roots
    }
    observed = set().union(*(set(item["seeds"]) for item in inventories.values()))
    malformed = malformed_byte_coverage(
        [path for item in inventories.values() for path in item["invalid_json"]],
        set().union(*expected.values()),
    )
    namespace_path = artifact_root / "t3-rollout-labels-20261015/bank/namespace.json"
    binding: dict[str, Any] = {"ok": False, "reason": "namespace.json missing"}
    namespace: dict[str, Any] = {"exists": False, "seeds": []}
    if namespace_path.is_file():
        raw = json.loads(namespace_path.read_text())
        seeds = [int(seed) for seed in raw.get("seeds", []) if type(seed) is int]
        binding = prior_banks._manifest_binding(namespace_path.parent, namespace_path)
        namespace = {
            "exists": True,
            "schema": raw.get("schema"),
            "count": len(seeds),
            "unique": len(seeds) == len(set(seeds)),
            "seed_digest": prior_banks._seed_digest(seeds),
            "manifest_binding": binding,
        }
        observed.update(seeds)
    checks = {}
    for round_index, seeds in expected.items():
        other = set().union(*(value for key, value in expected.items() if key != round_index))
        checks[round_index] = {
            "count": len(seeds),
            "unique": len(seeds) == 96,
            "seed_digest": prior_banks._seed_digest(seeds),
            "seeds": sorted(seeds),
            "registered_collisions": sorted(seeds & registered),
            "observed_json_or_label_collisions": sorted(seeds & observed),
            "other_reserved_round_collisions": sorted(seeds & other),
        }
    namespace_valid = (
        namespace.get("schema") == "t3-label-namespace/v1"
        and bool(namespace.get("count"))
        and namespace.get("unique") is True
    )
    passed = (
        t2["pass"]
        and binding["ok"]
        and namespace_valid
        and malformed["pass"]
        and all(
            item["unique"]
            and not item["registered_collisions"]
            and not item["observed_json_or_label_collisions"]
            and not item["other_reserved_round_collisions"]
            for item in checks.values()
        )
    )
    return {
        "schema": "t4c-seed-bank-audit/v1",
        "status": "PASS" if passed else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "registered NI/distillation rounds, used rounds 62-65, repository research JSON, "
            "all artifact-directory JSON/JSONL except the explicitly ignored aborted CPU T4c "
            "run, and immutable T3 label namespace"
        ),
        "excluded_output": str(output) if output is not None else None,
        "excluded_repository_results": str(repository_results),
        "ignored_entirely": [str(aborted), str(forbidden_worktree)],
        "registered_t2_audit": {"pass": t2["pass"], "schema": t2["schema"]},
        "t3_rollout_training_namespace": namespace,
        "inventories": {
            name: prior_banks._compact_inventory(item) for name, item in inventories.items()
        },
        "malformed_byte_coverage": malformed,
        "checks": checks,
        "pass": passed,
        "limit": (
            "No claim is made about unrecorded experiments, arbitrary binary files, or external "
            "systems."
        ),
    }


def main() -> int:
    """Write a freshness receipt without evaluating or training."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = seed_bank_audit(exclude_out=args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": receipt["pass"], "checks": receipt["checks"]}))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
