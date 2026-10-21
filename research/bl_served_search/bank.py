"""Local whole-allocation freshness audit; never opens sealed Round 0 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from research.bl_served_search.exports import digest, export_json

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
NAMESPACE = "bl-served-search-l2-20261021/v1"
COUNTS = {"frozen": 64, "mixed": 64, "scripted": 32}
SEALED = "t34-round0-study-01"
GLOBS = ("*.json", "*.jsonl", "*.md", "*.py", "*.yaml", "*.yml", "*.sh", "*.csv", "*.tsv", "*.txt")


def derive(round_index: int, counts: dict[str, int] = COUNTS) -> list[dict[str, Any]]:
    """Derive the entire uint32 allocation; rounds never replace individual seeds."""
    if type(round_index) is not int or round_index < 0:
        raise ValueError("round must be a nonnegative integer")
    rows = []
    for pool, allocation in (("selection", counts), ("preflight", dict.fromkeys(COUNTS, 1))):
        for mix, count in allocation.items():
            for index in range(count):
                name = f"{NAMESPACE}/round-{round_index}/{pool}/{mix}/{index}"
                seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
                rows.append(
                    {
                        "world_id": name,
                        "world_seed": seed,
                        "mix": mix,
                        "pool": pool,
                        "horizon": 5000 if pool == "selection" else 500,
                    }
                )
    return rows


def scan(candidates: list[int], repo: Path = ROOT, artifacts: Path = ARTIFACTS) -> dict:
    """Exact integer tokens, explicit no-peek exclusion, fail closed on read errors."""
    roots = [repo / name for name in ("research", "docs", "src", "scripts")]
    roots.append(artifacts)
    if any(not path.is_dir() for path in (repo / "research", repo / "docs", artifacts)):
        raise RuntimeError("mandatory scan roots missing")
    command = [
        "rg",
        "--threads",
        "1",
        "--hidden",
        "--no-ignore",
        "-F",
        "-w",
        "-o",
        "--no-heading",
        "--no-filename",
        "--no-line-number",
    ]
    for glob in GLOBS:
        command.extend(["--glob", glob])
    # rg prunes the directory before reading any entry or outcome bytes.
    for glob in (f"!**/{SEALED}/**", "!**/.git/**"):
        command.extend(["--glob", glob])
    for seed in candidates:
        command.extend(["-e", str(seed)])
    hits = {}
    for root in roots:
        if not root.is_dir():
            continue
        result = subprocess.run([*command, str(root)], capture_output=True, text=True, timeout=300)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"seed scan failed: {result.stderr[:500]}")
        hits[str(root)] = sorted({int(value) for value in result.stdout.split()})
    return {
        "roots": [str(x) for x in roots if x.is_dir()],
        "hits": hits,
        "globs": list(GLOBS),
        "never_opened": SEALED,
    }


def audit(rows: list[dict], repo: Path = ROOT, artifacts: Path = ARTIFACTS) -> dict:
    """Audit history plus seven concrete training inventories and bank controls."""
    seeds = [row["world_seed"] for row in rows]
    explicit = {}
    for relative in (
        "research/t34_round0/BANK_ROUND0.json",
        "research/t34_round0d/BANK_ROUND0D.json",
        "research/t34_pathb/RETENTION_BANK_V2.json",
        "research/phase0_search_20261019/WORLD_BANK.json",
    ):
        path = repo / relative
        value = json.loads(path.read_text())
        found: set[int] = set()

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                for key, entry in item.items():
                    if key in {"seed", "world_seed"} and type(entry) is int:
                        found.add(entry)
                    else:
                        visit(entry)
            elif isinstance(item, list):
                for entry in item:
                    visit(entry)

        visit(value)
        if not found:
            raise RuntimeError(f"explicit bank contains no seeds: {relative}")
        explicit[relative] = {
            "sha256": digest(path),
            "count": len(found),
            "positive_control": min(found),
            "collisions": sorted(set(seeds) & found),
        }
    controls = {entry["positive_control"] for entry in explicit.values()}
    history = scan(sorted(set(seeds) | controls), repo, artifacts)
    found = {x for values in history["hits"].values() for x in values}
    if not controls <= found:
        raise RuntimeError("freshness scanner failed positive bank controls")
    collisions = set(seeds) & found
    exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise RuntimeError("historical training exclusions incomplete")
    inventory = []
    for path in exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or any(type(x) is not int for x in values):
            raise RuntimeError(f"malformed historical exclusions: {path}")
        overlap = set(seeds) & set(values)
        collisions.update(overlap)
        inventory.append(
            {
                "path": str(path),
                "sha256": digest(path),
                "count": len(values),
                "collisions": sorted(overlap),
            }
        )
    internal = len(seeds) != len(set(seeds))
    return {
        "schema": "bl-served-search-seed-audit/v1",
        "status": "REJECTED" if collisions or internal else "CLEAR",
        "candidate_count": len(seeds),
        "unique_count": len(set(seeds)),
        "collisions": sorted(collisions),
        "internal_collision": internal,
        "explicit_banks": explicit,
        "history": history,
        "training_exclusions": inventory,
        "coverage": "all visible text banks including strict banks and T3 census; exact tokens",
        "limits": "opaque binaries, unrecorded external runs and inaccessible history; sealed "
        "Round 0 outcomes excluded, its committed bank checked explicitly",
    }


def materialize(rows: list[dict]) -> list[dict]:
    """Plan each pool's historical rosters once; do not change allocation helpers."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    context = gid._context(1)
    result = []
    for pool in ("selection", "preflight"):
        group = [row for row in rows if row["pool"] == pool]
        plans = planned_rosters(
            context, [r["world_seed"] for r in group], [r["mix"] for r in group]
        )
        result.extend({**row, "plan": plan} for row, plan in zip(group, plans))
    return result


def allocate(out: Path, *, max_rounds: int = 32) -> dict:
    """Retain every rejected round; publish first fully clear allocation only."""
    if out.exists():
        raise FileExistsError("allocation output must be fresh")
    out.mkdir(parents=True)
    for round_index in range(max_rounds):
        rows = derive(round_index)
        report = audit(rows)
        report.update(namespace=NAMESPACE, round_index=round_index)
        export_json(out / f"audit-round-{round_index:02d}.json", report)
        if report["status"] != "CLEAR":
            continue
        rows = materialize(rows)
        bank = {
            "schema": "bl-served-search-bank/v1",
            "namespace": NAMESPACE,
            "round_index": round_index,
            "development_only": True,
            "counts": COUNTS,
            "rows": [r for r in rows if r["pool"] == "selection"],
            "excluded_preflight_rows": [r for r in rows if r["pool"] == "preflight"],
            "audit_sha256": digest(out / f"audit-round-{round_index:02d}.json"),
        }
        export_json(out / "bank.json", bank)
        export_json(out / "audit.json", {**report, "bank_sha256": digest(out / "bank.json")})
        return bank
    raise RuntimeError("no clear whole allocation; preserve rejected rounds, do not dispatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    bank = allocate(args.out)
    print(json.dumps({"round": bank["round_index"], "counts": bank["counts"]}))


if __name__ == "__main__":
    main()
