"""Fresh E2 fixed-controller worlds and a no-peek historical collision audit.

Allocation has no gameplay. Published seed inventories cover the sealed Round 0
bank; the local scanner prunes its live artifact directory before traversal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
NAMESPACE = "bl-ppo-e2-screen-20261021/v2"
LEGACY_NAMESPACE = "bl-ppo-e2-screen-20261021/v1"
TRAINING_NAMESPACE = "bl-ppo-e2-training-20261021/v1"
MIX_COUNTS = {"scripted": 128, "frozen": 48, "mixed": 48}
LEGACY_COUNTS = {"scripted": 96, "frozen": 32, "mixed": 32}
SCHEMA = "bl-ppo-screen-bank/v2"
SEALED_DIRECTORY = "t34-round0-study-01"


def _scan_history(candidates: list[int], repo: Path, artifacts: Path) -> dict[str, Any]:
    """Scan explicit safe roots; never give ripgrep the live artifact parent.

    A basename-only negative glob is insufficient when rg receives an absolute
    directory root. os.walk prunes prohibited directories before descending;
    ripgrep receives only resolved, explicitly approved regular-file paths.
    """
    from research.t34_round0d.bank import SCAN_GLOBS, sha256

    if any(not p.is_dir() for p in (repo / "docs", repo / "research", artifacts)):
        raise ValueError("mandatory historical inventory roots unavailable")
    roots = [
        repo / name for name in ("docs", "research", "src", "scripts") if (repo / name).is_dir()
    ]
    # Do not resolve/stat the excluded child or follow artifact symlinks.
    roots.append(artifacts)
    command = [
        "rg",
        "--threads",
        "1",
        "--hidden",
        "--no-ignore",
        "--fixed-strings",
        "--word-regexp",
        "--only-matching",
        "--no-heading",
        "--no-filename",
        "--no-line-number",
    ]
    for pattern in SCAN_GLOBS:
        command += ["--glob", pattern]
    for seed in candidates:
        command += ["-e", str(seed)]
    suffixes = {pattern.removeprefix("*") for pattern in SCAN_GLOBS}
    files = []
    for root in roots:
        for current, directories, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            directories[:] = [
                name
                for name in directories
                if name not in {SEALED_DIRECTORY, "bl_ppo", ".git"}
                and not (current_path / name).is_symlink()
                and not (name == "bl-e2" and current_path.parts[-2:] == ("big-lever", "exp"))
            ]
            for name in names:
                path = current_path / name
                if path.suffix not in suffixes or path.is_symlink():
                    continue
                resolved = path.resolve()
                if SEALED_DIRECTORY in resolved.parts or not resolved.is_relative_to(
                    root.resolve()
                ):
                    raise ValueError("unsafe path escaped historical audit boundary")
                files.append(resolved)
    found: set[int] = set()
    # Explicit file batches keep Mac ARG_MAX bounded; rg never receives a
    # directory that could independently recurse past our pruning decision.
    for offset in range(0, len(files), 100):
        completed = subprocess.run(
            [*command, *map(str, files[offset : offset + 100])], capture_output=True, text=True
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError(f"safe historical scan failed: {completed.stderr[:500]}")
        found.update(int(token) for token in completed.stdout.split())
    hits = sorted(found)
    exclusions = sorted(
        path
        for path in files
        if path.name == "training_exclusions.json"
        and path.parent.name.startswith("namespaces")
        and path.parent.parent.name.startswith("frp-")
        and path.is_relative_to(artifacts)
    )
    if len(exclusions) < 7:
        raise ValueError("historical FRP training exclusion inventories incomplete")
    bindings = []
    for path in exclusions:
        numbers = json.loads(path.read_text())
        if not isinstance(numbers, list) or any(type(number) is not int for number in numbers):
            raise ValueError("malformed historical FRP exclusion inventory")
        bindings.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "count": len(numbers),
                "collisions": sorted(set(numbers) & set(candidates)),
            }
        )
    return {
        "method": "safe-root-pruned-literal-history-v1",
        "candidate_count": len(candidates),
        "scan_roots": list(map(str, roots)),
        "scan_globs": list(SCAN_GLOBS),
        "explicit_safe_file_count": len(files),
        "excluded_live_study_roots": [SEALED_DIRECTORY],
        "concrete_hits": {"safe_visible_roots": hits},
        "historical_frp_exclusions": bindings,
        "coverage": "safe visible text roots, published registered banks and FRP exclusions",
        "limits": "sealed live output root, opaque binaries, symlinks and unrecorded runs excluded",
    }


def canonical_sha(value: Any) -> str:
    """Hash portable JSON semantics, rejecting nonfinite numbers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def materialize(round_index: int = 0, *, allocation_version: int = 2) -> dict[str, Any]:
    """Allocate all decision, timing and parity rows without sampling outcomes."""
    from research.t34_round0.bank import _rows

    if type(round_index) is not int or round_index < 0:
        raise ValueError("round_index must be a nonnegative integer")
    if allocation_version not in (1, 2):
        raise ValueError("unknown prospective allocation version")
    counts = LEGACY_COUNTS if allocation_version == 1 else MIX_COUNTS
    namespace = f"{LEGACY_NAMESPACE if allocation_version == 1 else NAMESPACE}/round-{round_index}"
    value = {
        "schema": "bl-ppo-screen-bank/v1" if allocation_version == 1 else SCHEMA,
        "namespace": namespace,
        "round_index": round_index,
        "counts": counts,
        "training_namespaces": {
            arm: f"{TRAINING_NAMESPACE}/{arm}" for arm in ("R", "F", "calibration")
        },
        "profile": {
            "horizon": 5000,
            "hero_mode": "terminal",
            "hero_slot": 0,
            "hero_veto": "none",
            "frozen_veto": "v8",
            "mechanics_version": 1,
            "engine": "GridBatchSim",
            "roster_source": "research.t34_round0.bank._rows (read-only)",
        },
        "rows": _rows(namespace + "/screen", counts, 5000),
        "excluded_preflight_rows": _rows(namespace + "/preflight", dict.fromkeys(counts, 1), 40),
        "excluded_parity_rows": _rows(namespace + "/parity", dict.fromkeys(counts, 1), 5000),
    }
    seeds = all_seeds(value)
    if len(seeds) != sum(counts.values()) + 6:
        raise ValueError("whole allocation rejected: internal seed collision")
    value["allocation_sha256"] = canonical_sha(value)
    return value


def all_seeds(bank: dict[str, Any]) -> set[int]:
    """Return decision and excluded fixture seeds; training must reject these."""
    return {
        int(row["world_seed"])
        for key in ("rows", "excluded_preflight_rows", "excluded_parity_rows")
        for row in bank[key]
    }


def validate_bank(value: dict[str, Any]) -> None:
    """Require the complete deterministic bank, including every ordered roster."""
    version = {"bl-ppo-screen-bank/v1": 1, SCHEMA: 2}.get(value.get("schema"))
    if version is None or value != materialize(
        value.get("round_index"), allocation_version=version
    ):
        raise ValueError("bank differs from frozen deterministic allocation")


def audit(bank: dict[str, Any], repo: Path = ROOT, artifacts: Path = ARTIFACTS) -> dict[str, Any]:
    """Reject the entire allocation on historical collisions or missing evidence."""
    from research.t34_claim_a.seed_audit import _bank_seeds, namespace_seeds, seed_digest
    from research.t34_round0d import bank as prior

    validate_bank(bank)
    repo, artifacts = Path(repo).resolve(), Path(artifacts).resolve()
    payload = {
        "rows": bank["rows"],
        "excluded_preflight_rows": bank["excluded_preflight_rows"],
        "excluded_a5d_rows": bank["excluded_parity_rows"],
    }
    record = prior.audit(
        payload, scan=lambda seeds: _scan_history(seeds, repo=repo, artifacts=artifacts)
    )
    candidates = all_seeds(bank)
    from research.redesign_t7_phase_r_20261016.seed_audit import _registered

    historical_generated = _registered(repo)
    if candidates & historical_generated:
        raise ValueError("whole allocation collides with historical generated namespaces")
    published = []
    for path in sorted((repo / "research").rglob("*SEED_AUDIT*.json")):
        if path.parent == Path(__file__).parent:
            continue
        if path.is_symlink() or SEALED_DIRECTORY in path.resolve().parts:
            raise ValueError("unsafe published inventory path")
        value = json.loads(path.read_text())
        known = _bank_seeds(value)
        derived = set()
        if value.get("domain") == "t34-exit-b8-20261019/v1":
            for namespace, entry in value.get("namespaces", {}).items():
                values = namespace_seeds(namespace, int(entry["count"]))
                if seed_digest(values) != entry["sha256"]:
                    raise ValueError(f"published seed namespace digest mismatch: {path}")
                derived.update(values)
        if candidates & (known | derived):
            raise ValueError(f"whole allocation collides with published inventory: {path}")
        published.append(
            {
                "path": str(path.relative_to(repo)),
                "sha256": prior.sha256(path),
                "concrete_count": len(known),
                "reconstructed_count": len(derived),
            }
        )
    return {
        **record,
        "schema": "bl-ppo-screen-seed-audit/v1",
        "pass": True,
        "bank_semantic_sha256": bank["allocation_sha256"],
        "published_inventories": published,
        "historical_generated_seed_count": len(historical_generated),
        "sealed_bank_coverage": {
            "identity_source": "research/t34_round0/BANK_ROUND0.json",
            "live_source_opened": False,
            "scope": "published initial world identities and excluded fixtures only",
        },
        "limitations": [
            "No universal claim about unrecorded runs, inaccessible history or opaque binaries.",
            "Sealed live output bytes were never read; published bank identities cover that bank.",
            "Training/calibration must reject every allocated screen and excluded fixture seed.",
            "A later discovered collision invalidates this development screen.",
        ],
    }


def check_audit(bank: dict[str, Any], receipt: dict[str, Any]) -> None:
    """Validate the shipped Mac audit without reopening historical inputs."""
    validate_bank(bank)
    if (
        receipt.get("schema") != "bl-ppo-screen-seed-audit/v1"
        or receipt.get("method") != "safe-root-pruned-literal-history-v1"
        or not isinstance(receipt.get("explicit_safe_file_count"), int)
        or receipt.get("explicit_safe_file_count", 0) <= 0
        or receipt.get("pass") is not True
        or receipt.get("bank_semantic_sha256") != bank["allocation_sha256"]
        or receipt.get("collisions") != []
        or set(receipt.get("seeds", [])) != all_seeds(bank)
        or any(receipt.get("concrete_hits", {}).values())
        or len(receipt.get("historical_frp_exclusions", [])) < 7
        or any(x.get("collisions", []) for x in receipt.get("historical_frp_exclusions", []))
        or not receipt.get("published_inventories")
        or not isinstance(receipt.get("historical_generated_seed_count"), int)
        or receipt.get("historical_generated_seed_count", 0) <= 0
    ):
        raise ValueError("missing, stale or incomplete Mac seed audit")


def main(argv: list[str] | None = None) -> int:
    """Prepare immutable bank/audit exports; never play an episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args(argv)
    from research.bl_ppo.exports import export_json

    value = materialize(args.round)
    receipt = audit(value)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, item in (("bank.json", value), ("seed_audit.json", receipt)):
        export_json(args.out / name, item)
    print(
        json.dumps(
            {"bank": str(args.out / "bank.json"), "worlds": len(value["rows"]), "audit": "CLEAR"}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
