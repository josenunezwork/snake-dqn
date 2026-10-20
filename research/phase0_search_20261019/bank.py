"""Immutable world allocation and fail-closed collision audit for Phase 0.

This module creates no policies and never steps a world.  Local checkpoint
paths are deliberately absent from the semantic bank: a runner resolves the
pinned SHA identities to its checked input inventory immediately before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.apex_safety_20260926 import dev_screen
from src.evaluation.strict_promotion import scripted_agent

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
NAMESPACE = "phase0-search-teacher-20261019/v1"
SCHEMA = "phase0-search-teacher-world-bank/v1"
AUDIT_SCHEMA = "phase0-search-teacher-seed-audit/v1"
PROFILE_ID = "phase0-search-teacher-terminal-h5000/v1"
HORIZON = 5000
MIX_COUNTS = {"scripted": 96, "frozen": 32, "mixed": 32}
FIXTURE_COUNTS = {
    "calibration": {"scripted": 1, "frozen": 1, "mixed": 1},
    "timing": {"scripted": 1, "frozen": 1, "mixed": 1},
    "parity": {"scripted": 2, "frozen": 1, "mixed": 1},
    "smoke": {"scripted": 2},
}
OUTPUT = Path(__file__).with_name("WORLD_BANK.json")
AUDIT_OUTPUT = Path(__file__).with_name("SEED_AUDIT.json")


def canonical(value: Any) -> str:
    """Return canonical JSON suitable for stable byte-independent bindings."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    """Return the SHA-256 semantic digest for a JSON value."""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_create(path: Path, value: Any) -> None:
    """Write a JSON artifact once; replacing a receipt is a protocol error."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"create-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _strict_lookup() -> dict[str, tuple[str, str]]:
    """Map strict roster identities to actual evaluator opponent specs."""
    lookup = {
        scripted_agent(name)["sha256"]: ("scripted", name)
        for name in ("greedy_food", "random_safe")
    }
    lookup.update({sha: ("checkpoint", sha) for _, sha in dev_screen.POOL})
    return lookup


def _one_world(row: Mapping[str, Any], *, pool: str, index: int) -> dict[str, Any]:
    lookup = _strict_lookup()
    slots = []
    opponents = []
    for raw in row["slots"]:
        member = str(raw["member_sha256"])
        if member not in lookup:
            raise ValueError(f"strict roster identity cannot be resolved: {member}")
        kind, reference = lookup[member]
        slots.append(
            {
                "slot": int(raw["slot"]),
                "member_sha256": member,
                "eligible_member_sha256s": [str(x) for x in raw["eligible_member_sha256s"]],
                "resolved_kind": kind,
                "resolved_reference": reference,
            }
        )
        opponents.append([kind, reference])
    semantic = {
        "namespace": NAMESPACE,
        "pool": pool,
        "world_id": f"{pool}-{row['mix']}-{index:04d}-{int(row['world_seed'])}",
        "world_seed": int(row["world_seed"]),
        "mix": str(row["mix"]),
        "horizon": HORIZON,
        "profile_id": PROFILE_ID,
        "terminal_hero_slot": 0,
        "opponent_slots": slots,
    }
    return {
        **semantic,
        "opponents": opponents,
        "opponent_semantic_sha256": digest({"mix": semantic["mix"], "slots": slots}),
        "world_semantic_sha256": digest(semantic),
    }


def _pool(pool: str, counts: Mapping[str, int]) -> list[dict[str, Any]]:
    worlds: list[dict[str, Any]] = []
    for mix, count in counts.items():
        seeds = dev_screen.screen_seeds(int(count), NAMESPACE, f"{pool}/{mix}")
        materialized = [row for row in dev_screen._design_rows(seeds) if row["mix"] == mix]
        if len(materialized) != count:
            raise RuntimeError(f"strict materializer did not yield {count} {mix} {pool} worlds")
        worlds.extend(
            _one_world(row, pool=pool, index=index) for index, row in enumerate(materialized)
        )
    if len({world["world_seed"] for world in worlds}) != len(worlds):
        raise RuntimeError(f"world seed collision inside {pool}")
    return worlds


def make_bank() -> dict[str, Any]:
    """Build the registered 160-world selection bank plus disjoint fixtures."""
    selection = _pool("selection", MIX_COUNTS)
    fixtures = {name: _pool(name, counts) for name, counts in FIXTURE_COUNTS.items()}
    all_worlds = selection + [world for values in fixtures.values() for world in values]
    if len({world["world_seed"] for world in all_worlds}) != len(all_worlds):
        raise RuntimeError("world seed collision between Phase 0 pools")
    profile = {
        "profile_id": PROFILE_ID,
        "base_profile_id": dev_screen.PROFILE_NAME,
        "base_profile_digest": dev_screen.PROFILE_DIGEST,
        "engine": "grid",
        "horizon": HORIZON,
        "hero_mode": "terminal",
        "veto": False,
        "source": "research.apex_safety_20260926.dev_screen._design_rows",
    }
    allocation = {"selection": selection, "fixtures": fixtures}
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "profile": profile,
        "mix_counts": MIX_COUNTS,
        "selection": selection,
        "fixtures": fixtures,
        "banks": fixtures,
        "profile_semantic_sha256": digest(profile),
        "allocation_semantic_sha256": digest(allocation),
    }


def _json_paths(
    roots: Iterable[Path], own: Path, suffixes: tuple[str, ...] = ("*.json", "*.jsonl")
) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        command = ["rg", "--files", "--hidden", "--no-ignore", "--threads", "1"]
        for suffix in suffixes:
            command.extend(["--glob", suffix])
        listed = subprocess.run(
            command, cwd=root, text=True, capture_output=True, check=False, timeout=60
        )
        if listed.returncode not in (0, 1):
            raise RuntimeError(f"inventory listing failed for {root}: {listed.stderr.strip()}")
        for relative in listed.stdout.splitlines():
            path = root / relative
            if own not in path.parents:
                paths.append(path)
    return sorted(set(paths))


def _scan_candidates(paths: Iterable[Path], candidates: set[int]) -> set[int]:
    expression = r"\b(?:" + "|".join(map(str, sorted(candidates))) + r")\b"
    found: set[int] = set()
    values = [str(path) for path in paths]
    for offset in range(0, len(values), 128):
        completed = subprocess.run(
            [
                "rg",
                "--threads",
                "1",
                "--no-heading",
                "--no-filename",
                "--only-matching",
                expression,
                *values[offset : offset + 128],
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError(f"collision scan failed: {completed.stderr.strip()}")
        found.update(int(value) for value in completed.stdout.splitlines() if value.isdigit())
    return found


def _scan_roots(
    roots: Iterable[tuple[Path, tuple[str, ...]]],
    candidates: set[int],
    progress: list | None = None,
) -> set[int]:
    """Partition visible evidence into bounded single-thread fixed-string scans."""
    from research.t4e_return_20261016.seed_audit import _partitioned_rg

    found: set[int] = set()
    for root, suffixes in roots:
        if not root.exists():
            raise FileNotFoundError(f"collision inventory root missing: {root}")
        arguments = [
            "--hidden",
            "--no-ignore",
            "--fixed-strings",
            "--word-regexp",
            "--only-matching",
        ]
        for suffix in suffixes:
            arguments.extend(["--glob", suffix])
        for seed in sorted(candidates):
            arguments.extend(["-e", str(seed)])
        hits, scans = _partitioned_rg(root, arguments, (Path(__file__).parent,), "phase0-worlds")
        found.update(int(value) for value in hits if value.isdigit())
        if progress is not None:
            progress.extend({"root": str(root), **item} for item in scans)
    return found


def audit_bank(
    bank: Mapping[str, Any],
    *,
    repo: Path = ROOT,
    artifacts: Path = ARTIFACTS,
    inventory_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Audit every allocated world against bound historical strict inventory.

    Missing historic mandatory inputs intentionally raise rather than yielding a
    plausible-looking passing receipt.
    """
    from research.redesign_t7_phase_r_20261016 import seed_audit as t7_audit

    if bank.get("schema") != SCHEMA or bank.get("allocation_semantic_sha256") != digest(
        {"selection": bank.get("selection"), "fixtures": bank.get("fixtures")}
    ):
        raise ValueError("invalid or tampered Phase 0 world bank")
    selection = list(bank["selection"])
    if {mix: sum(row["mix"] == mix for row in selection) for mix in MIX_COUNTS} != MIX_COUNTS:
        raise ValueError("selection mix allocation differs from registration")
    candidates = {
        int(row["world_seed"])
        for row in selection + [row for worlds in bank["fixtures"].values() for row in worlds]
    }
    if len(candidates) != 160 + sum(sum(counts.values()) for counts in FIXTURE_COUNTS.values()):
        raise ValueError("Phase 0 bank lacks globally unique worlds")
    repo, artifacts = Path(repo), Path(artifacts)
    mandatory = {
        "t7_seed_audit": repo / "research/redesign_t7_phase_r_20261016/SEED_AUDIT_T7.json",
        "t7_plan": repo / "research/redesign_t7_phase_r_20261016/results/plan.json",
        "frp3_strict": repo / "docs/research/frp3_strict_result_2026-10-06.md",
        "b8_release": repo / "docs/research/b8_s3_release_2026-10-17.md",
        "artifact_root": artifacts,
        "t3_namespace": artifacts / "t3-rollout-labels-20261015/bank/namespace.json",
        "t3_manifest": artifacts / "t3-rollout-labels-20261015/bank/manifest.json",
        "t4_snapshot": artifacts / "t4b-snapshot-restart-20261016/full/manifest.json",
        "t4c_bank": artifacts / "t4c-data-20261016/bank-v3/manifest.json",
    }
    missing = [name for name, path in mandatory.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"mandatory historical audit inputs missing: {missing}")
    exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise FileNotFoundError("historical FRP strict exclusion inventories incomplete")
    mandatory.update({f"frp_exclusion_{index}": path for index, path in enumerate(exclusions)})
    bindings = {
        name: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        }
        for name, path in mandatory.items()
    }
    historical = t7_audit._registered(repo)
    t7_receipt = json.loads(
        (repo / "research/redesign_t7_phase_r_20261016/SEED_AUDIT_T7.json").read_text()
    )
    if t7_receipt.get("pass") is not True:
        raise ValueError("T7 historical strict audit is not passing")
    for item in t7_receipt.get("banks", {}).values():
        historical.update(int(seed) for seed in item.get("seeds", []))
    for path in exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or not all(type(value) is int for value in values):
            raise ValueError(f"invalid FRP exclusion inventory: {path}")
        historical.update(values)
    scan_progress: list = []
    if inventory_paths is not None:
        paths = list(inventory_paths)
        historical |= _scan_candidates(paths, candidates)
        inventory_count = len(paths)
    else:
        paths = []
        historical |= _scan_roots(
            (
                (repo / "research", ("*.json", "*.jsonl", "*.md", "*.py", "*.sh")),
                (repo / "docs", ("*.json", "*.jsonl", "*.md", "*.py", "*.sh")),
                (repo / "scripts", ("*.json", "*.jsonl", "*.md", "*.py", "*.sh")),
                (artifacts, ("*.json", "*.jsonl")),
            ),
            candidates,
            scan_progress,
        )
        inventory_count = None
    namespace_hits = []
    for path in paths:
        if NAMESPACE in path.read_text(errors="ignore"):
            namespace_hits.append(str(path))
    passed = not (candidates & historical) and not namespace_hits
    return {
        "schema": AUDIT_SCHEMA,
        "namespace": NAMESPACE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "world_bank_semantic_sha256": bank["allocation_semantic_sha256"],
        "candidate_count": len(candidates),
        "internal_unique": len(candidates) == 172,
        "mandatory_evidence": bindings,
        "inventory_file_count": inventory_count,
        "scan_progress": scan_progress,
        "registered_collision_seeds": sorted(candidates & historical),
        "namespace_collision_files": namespace_hits,
        "pass": passed and len(candidates) == 172,
        "limit": (
            "Visible JSON/JSONL inventories and bound strict historical receipts; "
            "opaque binaries excluded."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--audit-out", type=Path, default=AUDIT_OUTPUT)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    value = make_bank()
    audit = audit_bank(value, repo=args.repo, artifacts=args.artifacts)
    if not audit["pass"]:
        raise SystemExit(f"Phase 0 collision audit failed: {audit['registered_collision_seeds']}")
    write_create(args.out, value)
    audit["world_bank_file"] = {
        "path": str(args.out),
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    write_create(args.audit_out, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
