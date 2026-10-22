"""Frozen C1 allocation and outcome-blind freshness proof."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from research.bl_served_search import bank
from research.bl_served_search.exports import digest

NAMESPACE = "bl-plan-v3-c1-served-watch6-20261022/v1"
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
STUDY_ROOT = ARTIFACTS / "big-lever/exp/bl-c1"
PLAN = ARTIFACTS / "big-lever/PLAN_v3.md"
PLAN_SHA = "d23690617519262bd1b1a3e6c5d8b414ef952003b5b92c86842c9f5fa4210201"
COUNT = 24
HORIZON = 14400


def allocation() -> list[dict[str, Any]]:
    """One deterministic allocation; no replacement seeds or extension."""
    return [
        {
            "unit_id": f"session-{i:02d}",
            "seed": int.from_bytes(hashlib.sha256(f"{NAMESPACE}/{i}".encode()).digest()[:4], "big"),
            "horizon": HORIZON,
        }
        for i in range(COUNT)
    ]


def freshness(repo: Path = ROOT, artifacts: Path = ARTIFACTS) -> dict:
    """Scan visible history and training inventories; never open sealed outcomes.

    Only this allocation's fixed artifact directory is excluded. Its inputs and
    units are bound separately by the source/allocation identity and ledger.
    Registered banks, including sealed studies' committed banks, are scanned.
    """
    seeds = {row["seed"] for row in allocation()}
    if len(seeds) != COUNT:
        raise ValueError("internal C1 seed collision; frozen allocation infeasible")
    roots = [repo / name for name in ("research", "docs", "src", "scripts")] + [artifacts]
    if any(not p.is_dir() for p in roots):
        raise ValueError("mandatory freshness root missing")
    controls = {}
    for name in (
        "research/t34_round0/BANK_ROUND0.json",
        "research/t34_round0d/BANK_ROUND0D.json",
        "research/t34_pathb/RETENTION_BANK_V2.json",
        "research/phase0_search_20261019/WORLD_BANK.json",
    ):
        path = repo / name
        text = path.read_text()
        values = {int(x) for x in re.findall(r'"(?:world_seed|seed)"\s*:\s*(\d+)', text)}
        if not values or values & seeds:
            raise ValueError(f"bank missing seeds or C1 collision: {name}")
        controls[name] = {"sha256": digest(path), "positive_control": min(values)}
    command = ["rg", "--threads", "1", "--hidden", "--no-ignore", "-F", "-w", "-o", "-l"]
    for glob in bank.GLOBS:
        command += ["--glob", glob]
    command += ["--glob", "!**/.git/**", "--glob", f"!**/{bank.SEALED}/**"]
    # Fixed own namespace only; no user-selectable broad exclusion.
    command += ["--glob", "!**/big-lever/exp/bl-c1/**"]
    candidates = seeds | {row["positive_control"] for row in controls.values()}
    for seed in sorted(candidates):
        command += ["-e", str(seed)]
    inventories = {}
    found = set()
    collisions = []
    for root in roots:
        result = subprocess.run([*command, str(root)], capture_output=True, text=True, timeout=300)
        if result.returncode not in (0, 1):
            raise ValueError(f"freshness read failure: {result.stderr[:500]}")
        for filename in result.stdout.splitlines():
            path = Path(filename)
            content = path.read_text()
            hits = {int(x) for x in re.findall(r"\b\d+\b", content)} & candidates
            found |= hits
            if hits & seeds:
                collisions.append({"path": str(path), "seeds": sorted(hits & seeds)})
            inventories[str(path)] = digest(path)
    if not {row["positive_control"] for row in controls.values()} <= found:
        raise ValueError("freshness positive controls failed")
    exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise ValueError("historical training inventory incomplete")
    for path in exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or any(type(x) is not int for x in values):
            raise ValueError("malformed historical training inventory")
        inventories[str(path)] = digest(path)
        if seeds & set(values):
            collisions.append({"path": str(path), "seeds": sorted(seeds & set(values))})
    if collisions:
        raise ValueError(f"frozen C1 seed allocation collides: {collisions}")
    return {
        "status": "CLEAR",
        "namespace": NAMESPACE,
        "count": COUNT,
        "roots": [str(p) for p in roots],
        "controls": controls,
        "matching_file_hashes": inventories,
        "training_inventory_count": len(exclusions),
        "excluded_own_artifact_root": str(STUDY_ROOT),
        "sealed_outcomes_never_opened": bank.SEALED,
        "scope": (
            "all visible text banks and historical training inventories; "
            "opaque or unrecorded history unavailable"
        ),
    }
