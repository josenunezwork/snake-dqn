"""Outcome-blind seed disjointness audit; never opens Watch6 exports/ or pair-*.json."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from research.bl_served_search import bank
from research.watch6_alllen_screen import protocol as p
from tools.ops.package_spec import digest, publish_json

KNOWN_NAMESPACE_PREFIXES: tuple[tuple[str, int], ...] = (
    # Watch6 V1 and V2 confirmation banks
    ("t34-watch6-exact-stop-efficacy-20261030-v1", 64),
    ("t34-watch6-exact-stop-efficacy-20261030-v2", 256),
    # Watch6 V3 confirmation namespaces (exact primary/reserve + prospective patterns)
    ("t34-watch6-exact-stop-efficacy-20261031-v3", 512),
    ("t34-watch6-exact-stop-efficacy-20261031-v3/primary", 512),
    ("t34-watch6-exact-stop-efficacy-20261031-v3/reserve", 512),
    ("t34-watch6-exact-stop-efficacy-20261031-v3-reserve", 256),
    ("t34-watch6-exact-stop-efficacy-v3-20261031", 512),
    ("t34-watch6-exact-stop-efficacy-v3-reserve-20261031", 256),
    ("t34-watch6-search-efficacy-20261031-v3", 512),
    ("t34-watch6-search-efficacy-20261031-v3-reserve", 256),
    ("t34-watch6-v3-confirm-20261031", 512),
    ("t34-watch6-v3-confirm-20261031-reserve", 256),
    # Watch replay banks 01-04 (and 05-08 margin) + fixtures
    *((f"t34-watch-replay-fix-20261030-bank{i:02d}", 128) for i in range(1, 9)),
    *((f"t34-watch-replay-20261030-bank{i:02d}", 128) for i in range(1, 9)),
    ("t34-watch-replay-full-size-fixture-v1", 32),
    ("t34-watch-replay-mac-fixture-v1", 32),
    # C1 served-Watch6 bank
    ("bl-plan-v3-c1-served-watch6-20261022/v1", 128),
)

KNOWN_FIXTURE_SEEDS: frozenset[int] = frozenset(
    {
        34890123,
        34890124,
        34890125,
        91841001,
        91841002,
        91841003,
    }
)

COMMITTED_BANK_FILES: tuple[str, ...] = (
    "research/bl_served_search/BANK_L2.json",
    "research/t34_round0/BANK_ROUND0.json",
    "research/t34_round0d/BANK_ROUND0D.json",
    "research/t34_pathb/RETENTION_BANK_V2.json",
    "research/phase0_search_20261019/WORLD_BANK.json",
)


def _discover_watch6_namespaces(roots: list[Path]) -> dict[str, set[int]]:
    """Extract namespace strings and integer fixture lists from watch6* protocol sources."""
    discovered: dict[str, set[int]] = {}
    ns_pattern = re.compile(r"([A-Za-z0-9_./-]*watch6[A-Za-z0-9_./-]*)")
    for root in roots:
        research_dir = root / "research"
        if not research_dir.is_dir():
            continue
        for pkg in sorted(research_dir.glob("watch6*")):
            if not pkg.is_dir() or pkg.name == "watch6_alllen_screen":
                continue
            for py in sorted(pkg.glob("*.py")):
                text = py.read_text(encoding="utf-8")
                for match in ns_pattern.findall(text):
                    if "/" in match and match.startswith("research/"):
                        continue
                    if any(ch.isdigit() for ch in match) or "v3" in match or "v2" in match:
                        bucket = discovered.setdefault(f"{py}:{match}", set())
                        for prefix in (match, f"{match}/primary", f"{match}/reserve"):
                            bucket.update(p.seed_for(prefix, idx) for idx in range(512))
    return discovered


def audit(
    repo: Path = p.ROOT,
    artifacts: Path | None = None,
    extra_worktrees: list[Path] | None = None,
) -> dict[str, Any]:
    """Verify 48 main + canary + preflight + benchmark seeds are fresh without opening exports."""
    artifacts = (repo.parent / "snake-dqn-artifacts") if artifacts is None else artifacts
    default_v3 = repo.parent / "snake-dqn-argon-v3-confirm"
    worktrees = (
        list(extra_worktrees)
        if extra_worktrees is not None
        else ([default_v3] if default_v3.is_dir() else [])
    )

    main_rows = p.allocation(p.NAMESPACE, p.COUNT, p.HORIZON)["rows"]
    canary_rows = p.allocation(p.CANARY_NAMESPACE, p.CANARY_COUNT, p.CANARY_HORIZON)["rows"]
    preflight_rows = p.allocation(p.PREFLIGHT_NAMESPACE, p.PREFLIGHT_COUNT, p.PREFLIGHT_HORIZON)[
        "rows"
    ]

    main_seeds = {row["seed"] for row in main_rows}
    canary_seeds = {row["seed"] for row in canary_rows}
    preflight_seeds = {row["seed"] for row in preflight_rows}
    bench_seeds = set(p.BRANCH_BENCHMARK_SEEDS)
    own_all = main_seeds | canary_seeds | preflight_seeds | bench_seeds
    expected_total = p.COUNT + p.CANARY_COUNT + p.PREFLIGHT_COUNT + len(p.BRANCH_BENCHMARK_SEEDS)
    if len(own_all) != expected_total:
        raise ValueError("internal seed collision across main/canary/preflight/benchmark sets")

    collisions: list[dict[str, Any]] = []
    checked_namespaces: dict[str, int] = {}
    for ns, limit in KNOWN_NAMESPACE_PREFIXES:
        ns_seeds = {p.seed_for(ns, idx) for idx in range(limit)}
        checked_namespaces[ns] = limit
        overlap = own_all & ns_seeds
        if overlap:
            collisions.append({"source": f"namespace:{ns}", "seeds": sorted(overlap)})

    if own_all & KNOWN_FIXTURE_SEEDS:
        collisions.append(
            {"source": "known_fixture_seeds", "seeds": sorted(own_all & KNOWN_FIXTURE_SEEDS)}
        )

    dynamic_ns = _discover_watch6_namespaces([repo, *worktrees])
    for label, ns_seeds in sorted(dynamic_ns.items()):
        overlap = own_all & ns_seeds
        if overlap:
            collisions.append({"source": f"discovered:{label}", "seeds": sorted(overlap)})

    controls: dict[str, dict[str, Any]] = {}
    for rel in COMMITTED_BANK_FILES:
        path = repo / rel
        if not path.is_file():
            raise ValueError(f"mandatory committed bank missing: {rel}")
        text = path.read_text(encoding="utf-8")
        values = {int(x) for x in re.findall(r'"(?:world_seed|seed)"\s*:\s*(\d+)', text)}
        if not values:
            raise ValueError(f"committed bank parsed zero seeds: {rel}")
        if own_all & values:
            collisions.append({"source": rel, "seeds": sorted(own_all & values)})
        controls[rel] = {
            "sha256": digest(path),
            "count": len(values),
            "positive_control": max(values),
        }

    scan_roots = [repo / name for name in ("research", "docs", "src", "scripts")]
    if artifacts.is_dir():
        scan_roots.append(artifacts)
    for wt in worktrees:
        if (wt / "research").is_dir():
            scan_roots.append(wt / "research")

    # Strictly forbid opening any exports/, pair-*.json, *.watch.gz, or sealed directories.
    command = [
        "rg",
        "--threads",
        "4",
        "--hidden",
        "--no-ignore",
        "-F",
        "-w",
        "-o",
        "--no-heading",
        "--no-filename",
        "--no-line-number",
    ]
    for glob in bank.GLOBS:
        command += ["--glob", glob]
    for excluded in (
        "!**/.git/**",
        f"!**/{bank.SEALED}/**",
        "!**/exports/**",
        "!**/pair-*.json",
        "!**/*.watch.gz",
        "!**/t34-sol/watch6-efficacy*/**",
        "!**/t34-sol/argon-alllen-screen*/**",
        "!**/research/watch6_alllen_screen/**",
    ):
        command += ["--glob", excluded]

    positive_controls = {row["positive_control"] for row in controls.values()}
    candidates = own_all | positive_controls
    for seed in sorted(candidates):
        command += ["-e", str(seed)]

    matching_files: dict[str, str] = {}
    found_controls: set[int] = set()
    for root in scan_roots:
        if not root.is_dir():
            continue
        result = subprocess.run([*command, str(root)], capture_output=True, text=True, timeout=300)
        if result.returncode not in (0, 1):
            raise ValueError(f"seed audit rg failure on {root}: {result.stderr[:400]}")
        hits = {int(value) for value in result.stdout.split()} & candidates
        found_controls |= hits & positive_controls
        if hits & own_all:
            collisions.append({"source": str(root), "seeds": sorted(hits & own_all)})

    if not positive_controls <= found_controls:
        raise ValueError("seed audit positive controls failed")

    training_exclusions: list[Path] = []
    if artifacts.is_dir():
        training_exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
        if len(training_exclusions) < 7:
            raise ValueError("historical FRP training exclusion inventory incomplete")
        for path in training_exclusions:
            values = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(values, list) or any(type(x) is not int for x in values):
                raise ValueError(f"malformed historical training inventory: {path}")
            matching_files[str(path)] = digest(path)
            overlap = own_all & set(values)
            if overlap:
                collisions.append({"source": str(path), "seeds": sorted(overlap)})

    if collisions:
        raise ValueError(f"seed disjointness audit failed: {collisions}")

    return {
        "status": "CLEAR",
        "namespace": p.NAMESPACE,
        "canary_namespace": p.CANARY_NAMESPACE,
        "preflight_namespace": p.PREFLIGHT_NAMESPACE,
        "count": p.COUNT,
        "checked_namespaces": checked_namespaces,
        "discovered_watch6_sources": sorted(dynamic_ns.keys()),
        "controls": controls,
        "training_inventory_count": len(training_exclusions),
        "sealed_and_exports_never_opened": [
            bank.SEALED,
            "**/exports/**",
            "**/pair-*.json",
            "**/*.watch.gz",
        ],
        "matching_file_hashes": matching_files,
        "main_seeds": [row["seed"] for row in main_rows],
        "canary_seeds": [row["seed"] for row in canary_rows],
        "preflight_seeds": [row["seed"] for row in preflight_rows],
        "branch_benchmark_seeds": list(p.BRANCH_BENCHMARK_SEEDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    publish_json(args.out.resolve(), report)
    print(
        json.dumps(
            {"status": report["status"], "namespace": report["namespace"], "count": report["count"]}
        )
    )


if __name__ == "__main__":
    main()
