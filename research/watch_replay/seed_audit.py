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
from research.watch6_alllen_screen.protocol import BRANCH_BENCHMARK_SEEDS
from research.watch_replay.run import ROOT, allocation
from tools.ops.package_spec import digest, publish_json

DEFAULT_NAMESPACES: tuple[str, ...] = (
    "t34-watch-replay-20261101-bank14l1",
    "t34-watch-replay-20261101-bank15l1",
    "t34-watch-replay-20261101-bank16l1",
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
    # Previous Watch replay banks 01-13 + replay fixtures
    *((f"t34-watch-replay-fix-20261030-bank{i:02d}", 128) for i in range(1, 14)),
    *((f"t34-watch-replay-20261030-bank{i:02d}", 128) for i in range(1, 14)),
    *((f"t34-watch-replay-20261031-bank{i:02d}", 128) for i in range(1, 14)),
    *((f"t34-watch-replay-20261101-bank{i:02d}", 128) for i in range(1, 14)),
    ("t34-watch-replay-full-size-fixture-v1", 32),
    ("t34-watch-replay-mac-fixture-v1", 32),
    ("t34-watch-replay-parity-fixture-v1", 32),
    # C1 served-Watch6 bank
    ("bl-plan-v3-c1-served-watch6-20261022/v1", 128),
)

# Later studies a NEW namespace must also avoid (kept out of KNOWN_NAMESPACE_PREFIXES so
# audit() of the served L1 banks themselves is unchanged).
LATER_STUDY_NAMESPACES: tuple[tuple[str, int], ...] = (
    # Served L1 banks 14-16 (DEFAULT_NAMESPACES above)
    *((ns, 128) for ns in DEFAULT_NAMESPACES),
    # Hero Stage 0 probe + open-stop decisive gate (branch argon-hero-stage0-1103,
    # research/hero_stage0/protocol.py); seeds derived only, worlds never opened.
    ("t34-hero-probe-20261103-v1", 128),
    ("t34-openstop-gate-20261104-v1", 128),
    # Served G4 L1 banks 17-24 and their canary (watch-replay-g4-20261103).
    *((f"t34-watch-replay-20261103-g4-bank{i}l1", 128) for i in range(17, 25)),
    ("t34-watch-replay-20261103-g4-canary-l1-v1", 32),
)

# Bank-v2 (exact-stop-l1-os) namespaces, registered before any world is run.  DEV is the
# Mac canary namespace (excluded from the screen); SCREEN is the proposed 48-world
# paired dev screen.  Both are dev-only: never training, tuning, or release evidence.
BANK_V2_DEV_NAMESPACE = "t34-bank-v2-dev-20261105-v1"
BANK_V2_SCREEN_NAMESPACE = "t34-bank-v2-screen-20261105-v1"
BANK_V2_CONFIRM_NAMESPACE = "t34-bank-v2-confirm-20261105-v1"
BANK_V2_DEV_COUNT = 8
BANK_V2_SCREEN_COUNT = 48
BANK_V2_CONFIRM_COUNT = 96
BANK_V2_NAMESPACES: tuple[tuple[str, int], ...] = (
    (BANK_V2_DEV_NAMESPACE, BANK_V2_DEV_COUNT),
    (BANK_V2_SCREEN_NAMESPACE, BANK_V2_SCREEN_COUNT),
    # Prospective confirmation (registered 2026-11-05 after the screen ADOPT).
    (BANK_V2_CONFIRM_NAMESPACE, BANK_V2_CONFIRM_COUNT),
)

# Product replay generation only: distinct from every screen/confirmation and from
# each packet's excluded canary. These are never training or release-gate worlds.
L1_OS_BANK_COUNT = 44
L1_OS_BANK_NAMESPACES = tuple(
    f"t34-watch-replay-l1-os-20261106-bank{i:02d}-v1" for i in range(1, 4)
)
L1_OS_CANARY_NAMESPACES = tuple(
    f"t34-watch-replay-l1-os-20261106-canary{i:02d}-v1" for i in range(1, 4)
)
L1_OS_SMOKE_NAMESPACE = "t34-watch-replay-l1-os-20261106-smoke-v1"
L1_OS_NAMESPACES = (
    *((ns, L1_OS_BANK_COUNT) for ns in L1_OS_BANK_NAMESPACES),
    *((ns, 1) for ns in L1_OS_CANARY_NAMESPACES),
    (L1_OS_SMOKE_NAMESPACE, 8),
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
        *BRANCH_BENCHMARK_SEEDS,
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


def bank_v2_disjointness(repo: Path = ROOT) -> dict[str, Any]:
    """Outcome-blind, file-scan-free disjointness of the Bank-v2 namespaces.

    Checks the Bank-v2 dev/screen seeds against each other, every registered prior
    namespace prefix plus LATER_STUDY_NAMESPACES (served L1 banks, open-stop gate, hero
    probe), known fixture seeds,
    the excluded full-size canary seed and the committed bank files.  Never opens
    exports/, pair-*.json, *.watch.gz or any sealed directory.
    """
    seeds: dict[str, set[int]] = {}
    for ns, count in BANK_V2_NAMESPACES:
        values = {seed_for(ns, idx) for idx in range(count)}
        if len(values) != count:
            raise ValueError(f"internal seed collision in {ns}")
        seeds[ns] = values
    collisions: list[dict[str, Any]] = []
    names = [ns for ns, _ in BANK_V2_NAMESPACES]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if seeds[a] & seeds[b]:
                collisions.append(
                    {"source": f"bank-v2 {a} vs {b}", "seeds": sorted(seeds[a] & seeds[b])}
                )
    candidate = set().union(*seeds.values())
    prior = (*KNOWN_NAMESPACE_PREFIXES, *LATER_STUDY_NAMESPACES)
    for ns, limit in prior:
        if ns in {name for name, _ in BANK_V2_NAMESPACES}:
            raise ValueError(f"Bank-v2 namespace reuses a prior namespace: {ns}")
        overlap = candidate & {seed_for(ns, idx) for idx in range(limit)}
        if overlap:
            collisions.append({"source": f"namespace:{ns}", "seeds": sorted(overlap)})
    if candidate & KNOWN_FIXTURE_SEEDS:
        collisions.append(
            {"source": "known_fixture_seeds", "seeds": sorted(candidate & KNOWN_FIXTURE_SEEDS)}
        )
    canary_seed = seed_for("t34-watch-replay-full-size-fixture-v1", 0)
    if canary_seed in candidate:
        collisions.append({"source": "full-size canary fixture", "seeds": [canary_seed]})
    for rel in COMMITTED_BANK_FILES:
        text = (repo / rel).read_text(encoding="utf-8")
        values = {int(x) for x in re.findall(r'"(?:world_seed|seed)"\s*:\s*(\d+)', text)}
        if not values:
            raise ValueError(f"committed bank parsed zero seeds: {rel}")
        if candidate & values:
            collisions.append({"source": rel, "seeds": sorted(candidate & values)})
    if collisions:
        raise ValueError(f"Bank-v2 seed disjointness failed: {collisions}")
    return {
        "status": "CLEAR",
        "namespaces": {ns: sorted(seeds[ns]) for ns, _ in BANK_V2_NAMESPACES},
        "checked_namespaces": len(prior),
        "fixture_seeds": len(KNOWN_FIXTURE_SEEDS),
        "committed_bank_files": list(COMMITTED_BANK_FILES),
    }


def l1_os_bank_disjointness(repo: Path = ROOT) -> dict[str, Any]:
    """Audit only code-declared namespaces and the fixed committed seed-file allowlist.

    No glob, dynamic source discovery, artifact inventory, replay header, outcome,
    screen output, or sealed confirmation data is opened. Keep the older Bank-v2
    audit unchanged; this is a separate prospective product-bank registration.
    """
    prior = (*KNOWN_NAMESPACE_PREFIXES, *LATER_STUDY_NAMESPACES, *BANK_V2_NAMESPACES)
    prior_names = {ns for ns, _ in prior}
    candidates: dict[str, set[int]] = {}
    seen: set[int] = set()
    for ns, count in L1_OS_NAMESPACES:
        values = {seed_for(ns, i) for i in range(count)}
        if ns in prior_names or ns in candidates or len(values) != count or seen & values:
            raise ValueError(f"L1-OS namespace reuse or internal seed collision: {ns}")
        candidates[ns] = values
        seen |= values
    controls = {f"namespace:{ns}": {seed_for(ns, i) for i in range(n)} for ns, n in prior}
    controls["known_fixture_seeds"] = set(KNOWN_FIXTURE_SEEDS)
    for rel in COMMITTED_BANK_FILES:
        text = (repo / rel).read_text(encoding="utf-8")
        values = {int(x) for x in re.findall(r'"(?:world_seed|seed)"\s*:\s*(\d+)', text)}
        if not values:
            raise ValueError(f"committed bank parsed zero seeds: {rel}")
        controls[rel] = values
    collisions = {label: sorted(seen & seeds) for label, seeds in controls.items() if seen & seeds}
    if collisions:
        raise ValueError(f"L1-OS seed disjointness failed: {collisions}")
    return {
        "status": "CLEAR",
        "namespaces": {ns: sorted(seeds) for ns, seeds in candidates.items()},
        "total_seeds": len(seen),
        "checked_namespaces": len(prior),
        "fixture_seeds": len(KNOWN_FIXTURE_SEEDS),
        "committed_bank_files": list(COMMITTED_BANK_FILES),
        "scope": "code metadata and committed seed allowlist only; no outcomes or artifacts",
    }


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
    *,
    read_live_bank: bool = False,
) -> dict[str, Any]:
    """Verify replay bank seeds are disjoint from Watch6 V2/V3, alllen screen, and banks 01-13."""
    namespaces = tuple(namespaces)
    if not namespaces:
        raise ValueError("at least one namespace is required")
    artifacts = (repo.parent / "snake-dqn-artifacts") if artifacts is None else artifacts
    worktrees = list(extra_worktrees) if extra_worktrees is not None else []

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

    live_headers = (
        _live_bank_headers(artifacts / "watch-replay-bank-live") if read_live_bank else []
    )
    live_seeds = {row["seed"] for row in live_headers}
    if bank_seeds & live_seeds:
        collisions.append(
            {
                "source": str(artifacts / "watch-replay-bank-live"),
                "seeds": sorted(bank_seeds & live_seeds),
            }
        )

    prior_contracts: dict[str, int] = {}
    candidate_ns_set = set(namespaces)
    if (artifacts / "t34-sol").is_dir():
        contract_paths = sorted(
            {
                *(artifacts / "t34-sol").glob("watch-replay*/*/main-contract.json"),
                *(artifacts / "t34-sol").glob("*/packet-bank*/main-contract.json"),
            }
        )
        for contract_path in contract_paths:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            if payload.get("namespace") in candidate_ns_set:
                continue
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
        "!**/watch-replay-bank-live*/**",
        "!**/t34-sol/watch6-efficacy*/**",
        "!**/t34-sol/argon-replay-banks/**",
        "!**/t34-sol/argon-banks-l1/**",
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
