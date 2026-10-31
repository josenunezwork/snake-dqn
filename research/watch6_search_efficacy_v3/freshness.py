"""Pre-data seed disjointness audit for Watch6 V3 primary and reserve banks.

Audits all 192 V3 scientific seeds (96 primary + 96 reserve) and 3 excluded
fixtures against Watch6 V1, Watch6 V2, Watch replay banks 01-32, exploration
namespaces, committed JSON banks, visible artifact text, and historical FRP
training exclusions while strictly refusing to ever open any exports/ directory
or pair-*.json outcome payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from research.bl_served_search import bank
from research.watch6_search_efficacy_v3 import protocol as p
from tools.ops.package_spec import digest, publish_json

STUDY_ARTIFACT_GLOB = "!**/t34-sol/argon-v3-confirm/**"

EXPLICIT_DISJOINT_NAMESPACES: list[tuple[str, int]] = [
    ("t34-watch6-exact-stop-efficacy-20261030-v1", 256),
    ("t34-watch6-exact-stop-efficacy-20261030-v2", 256),
    ("t34-watch-replay-20261030-v1", 256),
    ("t34-watch-replay-parity-fixture-v1", 64),
    *[(f"t34-watch-replay-fix-20261030-bank{i:02d}", 256) for i in range(1, 33)],
    *[(f"t34-watch-replay-20261030-bank{i:02d}", 256) for i in range(1, 33)],
    ("bl-plan-v3-c1-served-watch6-20261022/v1", 512),
    ("bl-served-search-l2-20261021/v1", 512),
    ("bl-served-search-mac-identity-20261022/v1", 512),
    ("bl-c15-b2-natural-roots-20261022/v1", 512),
    ("bl-c15-m0-d300-lad/v1", 512),
    ("t15-mixed-roster-20261017/v1", 512),
    ("t15-mixed-roster-20261017/smoke-v1", 128),
    ("t12-b8-serving-20261017/v1", 512),
    ("t13-product-roster-20261017/v1", 512),
    ("t13b-six-snake-20261017/v1", 512),
    ("t34-round0-premise-20261020/v1", 512),
    ("t34-round0d-dense-20261021/v1", 512),
    ("t34-round0d-dense-r1-20261021/v1", 512),
    ("t34-pathb-deadline-20261019/v1", 512),
    ("s1-mature-h5000-terminal-hero/20261018/v5", 512),
    ("q1-mature-n1n3-h5000-terminal-hero/20261017/v1", 512),
    ("t34-watch6-alllen-screen-20261031-v1", 512),
    ("t34-watch6-alllen-screen-20261031", 512),
    ("t34-watch6-all-len-screen-20261031-v1", 512),
    ("t34-explore-watch6-alllen-screen-20261031-r1", 512),
    ("t34-explore-watch6-alllen-canary-20261031-r1", 64),
    ("t34-explore-watch6-alllen-preflight-20261031-r1", 64),
    ("t34-watch-replay-full-size-fixture-v1", 64),
    ("t34-watch-replay-mac-fixture-v1", 64),
]


def _derive_namespace_seeds(namespace: str, count: int) -> set[int]:
    derived = set()
    for index in range(count):
        for key in (
            f"{namespace}/{index}",
            f"{namespace}/primary/{index}",
            f"{namespace}/reserve/{index}",
        ):
            derived.add(int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big"))
    return derived


def _assert_not_sealed_export(path: Path) -> None:
    if "exports" in path.parts or path.name.startswith("pair-"):
        raise ValueError(f"refusing to read sealed export or pair payload: {path}")


def freshness(repo: Path = p.ROOT, artifacts: Path | None = None) -> dict:
    """Audit V3 primary + reserve seeds without opening any sealed outcome export."""
    artifacts = artifacts if artifacts is not None else (repo.parent / "snake-dqn-artifacts")
    primary = [row["seed"] for row in p.allocation()]
    reserve = [row["seed"] for row in p.reserve_allocation()]
    fixtures = list(p.FIXTURE_SEEDS)
    scientific_seeds = set(primary) | set(reserve)
    all_v3_seeds = scientific_seeds | set(fixtures)
    if (
        len(primary) != p.COUNT
        or len(reserve) != p.COUNT
        or len(all_v3_seeds) != 2 * p.COUNT + len(fixtures)
    ):
        raise ValueError("internal V3 primary/reserve/fixture seed collision")

    namespace_collisions = []
    for namespace, count in EXPLICIT_DISJOINT_NAMESPACES:
        overlap = all_v3_seeds & _derive_namespace_seeds(namespace, count)
        if overlap:
            namespace_collisions.append({"namespace": namespace, "seeds": sorted(overlap)})
    if namespace_collisions:
        raise ValueError(
            f"V3 seed collision with historical/exploration namespace: {namespace_collisions}"
        )

    roots = [repo / name for name in ("research", "docs", "src", "scripts")] + [artifacts]
    if any(not root.is_dir() for root in roots):
        raise ValueError("mandatory freshness root missing")

    controls = {}
    for name in (
        "research/t34_round0/BANK_ROUND0.json",
        "research/t34_round0d/BANK_ROUND0D.json",
        "research/t34_pathb/RETENTION_BANK_V2.json",
        "research/phase0_search_20261019/WORLD_BANK.json",
    ):
        path = repo / name
        _assert_not_sealed_export(path)
        text = path.read_text()
        values = {int(x) for x in re.findall(r'"(?:world_seed|seed)"\s*:\s*(\d+)', text)}
        if not values or values & scientific_seeds:
            raise ValueError(f"committed bank missing seeds or collides with V3: {name}")
        controls[name] = {"sha256": digest(path), "positive_control": min(values)}

    command = ["rg", "--threads", "1", "--hidden", "--no-ignore", "-F", "-w", "-o", "-l"]
    for glob in bank.GLOBS:
        command += ["--glob", glob]
    for excluded in (
        "!**/.git/**",
        f"!**/{bank.SEALED}/**",
        "!**/exports/**",
        "!**/pair-*.json",
        "!**/pair-*.meta.json",
        STUDY_ARTIFACT_GLOB,
    ):
        command += ["--glob", excluded]

    candidates = scientific_seeds | {row["positive_control"] for row in controls.values()}
    for seed in sorted(candidates):
        command += ["-e", str(seed)]

    inventories = {}
    found = set()
    collisions = []
    for root in roots:
        result = subprocess.run([*command, str(root)], capture_output=True, text=True, timeout=300)
        if result.returncode not in (0, 1):
            raise ValueError(f"freshness rg failure: {result.stderr[:500]}")
        for filename in result.stdout.splitlines():
            path = Path(filename)
            _assert_not_sealed_export(path)
            content = path.read_text()
            hits = {int(x) for x in re.findall(r"\b\d+\b", content)} & candidates
            found |= hits
            if hits & scientific_seeds:
                collisions.append({"path": str(path), "seeds": sorted(hits & scientific_seeds)})
            inventories[str(path)] = digest(path)

    if not {row["positive_control"] for row in controls.values()} <= found:
        raise ValueError("freshness positive controls failed")

    exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise ValueError("historical training inventory incomplete")
    for path in exclusions:
        _assert_not_sealed_export(path)
        values = json.loads(path.read_text())
        if not isinstance(values, list) or any(type(x) is not int for x in values):
            raise ValueError("malformed historical training inventory")
        inventories[str(path)] = digest(path)
        if scientific_seeds & set(values):
            collisions.append({"path": str(path), "seeds": sorted(scientific_seeds & set(values))})

    if collisions:
        raise ValueError(f"frozen V3 seed allocation collides: {collisions}")

    return {
        "schema": "watch6-v3-freshness/v1",
        "status": "CLEAR",
        "namespace": p.NAMESPACE,
        "primary_count": p.COUNT,
        "reserve_count": p.COUNT,
        "fixture_count": len(fixtures),
        "explicit_disjoint_namespaces_checked": len(EXPLICIT_DISJOINT_NAMESPACES),
        "roots": [str(r) for r in roots],
        "controls": controls,
        "matching_file_hashes": inventories,
        "training_inventory_count": len(exclusions),
        "sealed_exports_excluded": ["**/exports/**", "**/pair-*.json", bank.SEALED],
        "primary_allocation": p.allocation(),
        "reserve_allocation": p.reserve_allocation(),
        "fixtures": fixtures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freshness()
    publish_json(args.out, result)
    print(json.dumps({"status": result["status"], "namespace": result["namespace"]}), flush=True)


if __name__ == "__main__":
    main()
