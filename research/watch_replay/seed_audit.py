"""Outcome-blind seed disjointness audit for Watch replay banks; never opens Watch6 exports/."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from research.bl_served_search import bank
from research.watch_replay.run import ROOT, allocation
from tools.ops.package_spec import digest, publish_json

DEFAULT_NAMESPACES: tuple[str, ...] = (
    "t34-watch-replay-20261031-bank05",
    "t34-watch-replay-20261031-bank06",
    "t34-watch-replay-20261031-bank07",
)

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
    # Watch6 all-length screen (main + canary + preflight)
    ("t34-explore-watch6-alllen-screen-20261031-r1", 256),
    ("t34-explore-watch6-alllen-canary-20261031-r1", 32),
    ("t34-explore-watch6-alllen-preflight-20261031-r1", 32),
    # Previous Watch replay banks 01-04 (plus 20261030 01-08 margin) + replay fixtures
    *((f"t34-watch-replay-fix-20261030-bank{i:02d}", 128) for i in range(1, 9)),
    *((f"t34-watch-replay-20261030-bank{i:02d}", 128) for i in range(1, 9)),
    *((f"t34-watch-replay-20261031-bank{i:02d}", 128) for i in range(1, 5)),
    ("t34-watch-replay-full-size-fixture-v1", 32),
    ("t34-watch-replay-mac-fixture-v1", 32),
    ("t34-watch-replay-parity-fixture-v1", 32),
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
        91842001,
        93841001,
        93841002,
        93841003,
    }
)

COMMITTED_BANK_FILES: tuple[str, ...] = (
    "research/bl_served_search/BANK_L2.json",
    "research/t34_round0/BANK_ROUND0.json",
    "research/t34_round0d/BANK_ROUND0D.json",
    "research/t34_pathb/RETENTION_BANK_V2.json",
    "research/phase0_search_20261019/WORLD_BANK.json",
)


def seed_for(namespace: str, index: int) -> int:
    """Standard uint32 SHA256 seed derivation shared by Watch6 and Watch replay."""
    return int.from_bytes(hashlib.sha256(f"{namespace}/{index}".encode()).digest()[:4], "big")


def _discover_watch6_sources(roots: list[Path]) -> dict[str, set[int]]:
    """Extract namespace strings and integer fixture seeds from watch6* code files only."""
    discovered: dict[str, set[int]] = {}
    ns_pattern = re.compile(r"([A-Za-z0-9_./-]*(?:watch6|alllen)[A-Za-z0-9_./-]*)")
    int_pattern = re.compile(r"\b(\d{7,10})\b")
    for root in roots:
        research_dir = root / "research"
        if not research_dir.is_dir():
            continue
        for pkg in sorted(research_dir.glob("watch6*")):
            if not pkg.is_dir():
                continue
            for py in sorted(pkg.glob("*.py")):
                text = py.read_text(encoding="utf-8")
                int_seeds = {int(value) for value in int_pattern.findall(text)}
                if int_seeds:
                    discovered.setdefault(f"{py}:literals", set()).update(int_seeds)
                for match in ns_pattern.findall(text):
                    if "/" in match and match.startswith("research/"):
                        continue
                    if any(ch.isdigit() for ch in match) or "v3" in match or "v2" in match:
                        bucket = discovered.setdefault(f"{py}:{match}", set())
                        for prefix in (match, f"{match}/primary", f"{match}/reserve"):
                            bucket.update(seed_for(prefix, idx) for idx in range(512))
    return discovered


def _live_bank_headers(live_bank: Path) -> list[dict[str, Any]]:
    """Read only the 1-line metadata headers of installed watch-replay-bank-live sessions."""
    rows: list[dict[str, Any]] = []
    if not live_bank.is_dir():
        return rows
    for path in sorted(live_bank.glob("*.watch.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            header = json.loads(stream.readline())
        rows.append(
            {
                "file": path.name,
                "unit_id": header["unit_id"],
                "seed": int(header["seed"]),
                "controller": header["controller"],
            }
        )
    return rows


def audit(
    namespaces: tuple[str, ...] | list[str] = DEFAULT_NAMESPACES,
    count: int = 12,
    controller: str = "exact-stop-l2-prime",
    repo: Path = ROOT,
    artifacts: Path | None = None,
    extra_worktrees: list[Path] | None = None,
) -> dict[str, Any]:
    """Verify replay bank seeds are disjoint from Watch6 V2/V3, alllen screen, and banks 01-04."""
    namespaces = tuple(namespaces)
    if not namespaces:
        raise ValueError("at least one namespace is required")
    artifacts = (repo.parent / "snake-dqn-artifacts") if artifacts is None else artifacts
    default_v3 = repo.parent / "snake-dqn-argon-v3-confirm"
    worktrees = (
        list(extra_worktrees)
        if extra_worktrees is not None
        else ([default_v3] if default_v3.is_dir() else [])
    )

    banks_by_ns: dict[str, list[dict[str, int]]] = {}
    bank_seeds: set[int] = set()
    for ns in namespaces:
        if ns in banks_by_ns:
            raise ValueError(f"duplicate namespace in audit: {ns}")
        rows = allocation(ns, count, controller)["rows"]
        banks_by_ns[ns] = rows
        for row in rows:
            if row["seed"] in bank_seeds:
                raise ValueError(f"internal seed collision across replay bank namespaces: {ns}")
            bank_seeds.add(row["seed"])

    canary_seed = allocation("t34-watch-replay-full-size-fixture-v1", 1, controller, 3000)["rows"][
        0
    ]["seed"]
    if canary_seed in bank_seeds:
        raise ValueError("replay bank seed collides with excluded canary fixture seed")

    collisions: list[dict[str, Any]] = []
    checked_namespaces: dict[str, int] = {}
    for ns, limit in KNOWN_NAMESPACE_PREFIXES:
        ns_seeds = {seed_for(ns, idx) for idx in range(limit)}
        checked_namespaces[ns] = limit
        overlap = bank_seeds & ns_seeds
        if overlap:
            collisions.append({"source": f"namespace:{ns}", "seeds": sorted(overlap)})

    if bank_seeds & KNOWN_FIXTURE_SEEDS:
        collisions.append(
            {"source": "known_fixture_seeds", "seeds": sorted(bank_seeds & KNOWN_FIXTURE_SEEDS)}
        )

    dynamic_ns = _discover_watch6_sources([repo, *worktrees])
    for label, ns_seeds in sorted(dynamic_ns.items()):
        overlap = bank_seeds & ns_seeds
        if overlap:
            collisions.append({"source": f"discovered:{label}", "seeds": sorted(overlap)})

    live_headers = _live_bank_headers(artifacts / "watch-replay-bank-live")
    live_seeds = {row["seed"] for row in live_headers}
    if bank_seeds & live_seeds:
        collisions.append(
            {
                "source": str(artifacts / "watch-replay-bank-live"),
                "seeds": sorted(bank_seeds & live_seeds),
            }
        )

    prior_contracts: dict[str, int] = {}
    if (artifacts / "t34-sol").is_dir():
        for contract_path in sorted(
            (artifacts / "t34-sol").glob("watch-replay*/*/main-contract.json")
        ):
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            c_seeds = {int(r["seed"]) for r in payload.get("rows", [])}
            prior_contracts[str(contract_path)] = len(c_seeds)
            if bank_seeds & c_seeds:
                collisions.append(
                    {"source": str(contract_path), "seeds": sorted(bank_seeds & c_seeds)}
                )

    controls: dict[str, dict[str, Any]] = {}
    for rel in COMMITTED_BANK_FILES:
        path = repo / rel
        if not path.is_file():
            raise ValueError(f"mandatory committed bank missing: {rel}")
        text = path.read_text(encoding="utf-8")
        values = {int(x) for x in re.findall(r'"(?:world_seed|seed)"\s*:\s*(\d+)', text)}
        if not values:
            raise ValueError(f"committed bank parsed zero seeds: {rel}")
        if bank_seeds & values:
            collisions.append({"source": rel, "seeds": sorted(bank_seeds & values)})
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
        "!**/t34-sol/argon-replay-banks/**",
        "!**/research/watch_replay/seed_audit.py",
    ):
        command += ["--glob", excluded]

    positive_controls = {row["positive_control"] for row in controls.values()}
    candidates = bank_seeds | positive_controls
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
        if hits & bank_seeds:
            collisions.append({"source": str(root), "seeds": sorted(hits & bank_seeds)})

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
            overlap = bank_seeds & set(values)
            if overlap:
                collisions.append({"source": str(path), "seeds": sorted(overlap)})

    if collisions:
        raise ValueError(f"seed disjointness audit failed: {collisions}")

    return {
        "status": "CLEAR",
        "namespaces": list(namespaces),
        "count_per_bank": count,
        "total_seeds": len(bank_seeds),
        "controller": controller,
        "excluded_canary_seed": canary_seed,
        "live_bank_sessions_checked": len(live_headers),
        "live_bank_distinct_seeds": len(live_seeds),
        "prior_replay_contracts_checked": prior_contracts,
        "checked_namespaces": checked_namespaces,
        "discovered_watch6_sources": sorted(dynamic_ns.keys()),
        "controls": controls,
        "training_inventory_count": len(training_exclusions),
        "sealed_and_exports_never_opened": [
            bank.SEALED,
            "**/exports/**",
            "**/pair-*.json",
        ],
        "matching_file_hashes": matching_files,
        "banks": banks_by_ns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespaces", nargs="+", default=list(DEFAULT_NAMESPACES))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--controller", default="exact-stop-l2-prime")
    args = parser.parse_args()
    report = audit(
        namespaces=args.namespaces,
        count=args.count,
        controller=args.controller,
    )
    publish_json(args.out.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "namespaces": report["namespaces"],
                "total_seeds": report["total_seeds"],
                "live_bank_sessions_checked": report["live_bank_sessions_checked"],
            }
        )
    )


if __name__ == "__main__":
    main()
