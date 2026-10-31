"""Outcome-blind seed disjointness audit & world-disjoint split allocation for Track-4 Gate 1.

Strictly forbids opening any ``exports/``, ``pair-*.json``, ``*.watch.gz`` payload bodies,
or sealed study directories.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.bl_served_search import bank
from research.t4_gate1 import CANARY_NAMESPACE, MAIN_NAMESPACE, PREFLIGHT_NAMESPACE
from tools.ops.package_spec import digest, publish_json

ROOT = Path(__file__).resolve().parents[2]

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
    # Watch replay banks 01-08 (20261030, fix, and 20261031) + replay fixtures
    *((f"t34-watch-replay-fix-20261030-bank{i:02d}", 128) for i in range(1, 9)),
    *((f"t34-watch-replay-20261030-bank{i:02d}", 128) for i in range(1, 9)),
    *((f"t34-watch-replay-20261031-bank{i:02d}", 128) for i in range(1, 9)),
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
    """Standard uint32 SHA256 seed derivation shared across Watch6 and T34 exploration banks."""
    if not namespace:
        raise ValueError(f"invalid namespace: {namespace!r}")
    if int(index) < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    val = int.from_bytes(hashlib.sha256(f"{namespace}/{int(index)}".encode()).digest()[:4], "big")
    if val <= 0:
        raise ValueError(f"degenerate zero seed for {namespace}/{index}")
    return val


def world_allocation(
    namespace: str,
    *,
    num_replicates: int,
    fit_per_replicate: int,
    val_per_replicate: int,
    test_per_replicate: int,
) -> dict[str, Any]:
    """Deterministically allocate world-disjoint fit/val/test seeds across replicates."""
    if num_replicates < 1:
        raise ValueError("num_replicates must be >= 1")
    if fit_per_replicate < 1 or val_per_replicate < 1 or test_per_replicate < 1:
        raise ValueError(
            "fit_per_replicate, val_per_replicate, and test_per_replicate must be >= 1"
        )

    worlds_per_rep = int(fit_per_replicate + val_per_replicate + test_per_replicate)
    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()

    for rep in range(int(num_replicates)):
        base = rep * worlds_per_rep
        part_specs = (
            ("fit", 0, int(fit_per_replicate)),
            ("val", int(fit_per_replicate), int(val_per_replicate)),
            ("test", int(fit_per_replicate + val_per_replicate), int(test_per_replicate)),
        )
        for partition, offset, count in part_specs:
            for local_idx in range(count):
                global_idx = base + offset + local_idx
                seed = seed_for(namespace, global_idx)
                if seed in seen_seeds:
                    raise ValueError(f"internal seed collision at {namespace}/{global_idx}: {seed}")
                seen_seeds.add(seed)
                rows.append(
                    {
                        "global_index": int(global_idx),
                        "replicate_index": int(rep),
                        "partition": partition,
                        "local_index": int(local_idx),
                        "seed": int(seed),
                    }
                )

    return {
        "schema": "t34-t4-gate1-world-allocation/v1",
        "namespace": namespace,
        "num_replicates": int(num_replicates),
        "fit_per_replicate": int(fit_per_replicate),
        "val_per_replicate": int(val_per_replicate),
        "test_per_replicate": int(test_per_replicate),
        "total_worlds": len(rows),
        "rows": rows,
    }


def _discover_watch6_sources(roots: Sequence[Path]) -> dict[str, set[int]]:
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
    *,
    main_allocation: Mapping[str, Any] | None = None,
    canary_allocation: Mapping[str, Any] | None = None,
    preflight_allocation: Mapping[str, Any] | None = None,
    repo: Path = ROOT,
    artifacts: Path | None = None,
    extra_worktrees: list[Path] | None = None,
    scan_filesystem: bool = True,
) -> dict[str, Any]:
    """Verify Gate-1 main/canary/preflight seeds are pairwise & externally disjoint."""
    main_alloc = dict(
        main_allocation
        if main_allocation is not None
        else world_allocation(
            MAIN_NAMESPACE,
            num_replicates=2,
            fit_per_replicate=102,
            val_per_replicate=26,
            test_per_replicate=64,
        )
    )
    canary_alloc = dict(
        canary_allocation
        if canary_allocation is not None
        else world_allocation(
            CANARY_NAMESPACE,
            num_replicates=1,
            fit_per_replicate=4,
            val_per_replicate=2,
            test_per_replicate=2,
        )
    )
    preflight_alloc = dict(
        preflight_allocation
        if preflight_allocation is not None
        else world_allocation(
            PREFLIGHT_NAMESPACE,
            num_replicates=1,
            fit_per_replicate=2,
            val_per_replicate=1,
            test_per_replicate=2,
        )
    )

    allocs = {
        str(main_alloc["namespace"]): main_alloc,
        str(canary_alloc["namespace"]): canary_alloc,
        str(preflight_alloc["namespace"]): preflight_alloc,
    }
    if len(allocs) != 3:
        raise ValueError("main, canary, and preflight namespaces must be distinct")

    gate1_seeds: set[int] = set()
    counts_by_ns: dict[str, int] = {}
    for ns, doc in allocs.items():
        rows = doc["rows"]
        counts_by_ns[ns] = len(rows)
        for row in rows:
            s = int(row["seed"])
            if s in gate1_seeds:
                raise ValueError(f"internal seed collision across Gate-1 namespaces at {ns}: {s}")
            gate1_seeds.add(s)

    resolved_artifacts = (repo.parent / "snake-dqn-artifacts") if artifacts is None else artifacts
    default_v3 = repo.parent / "snake-dqn-argon-v3-confirm"
    worktrees = (
        list(extra_worktrees)
        if extra_worktrees is not None
        else ([default_v3] if default_v3.is_dir() else [])
    )

    collisions: list[dict[str, Any]] = []
    checked_namespaces: dict[str, int] = {}
    for ns, limit in KNOWN_NAMESPACE_PREFIXES:
        ns_seeds = {seed_for(ns, idx) for idx in range(limit)}
        checked_namespaces[ns] = limit
        overlap = gate1_seeds & ns_seeds
        if overlap:
            collisions.append({"source": f"namespace:{ns}", "seeds": sorted(overlap)})

    # Also exclude bl_served_search L2 round-0/1/2 seeds explicitly.
    for r_idx in range(3):
        l2_seeds = {int(r["world_seed"]) for r in bank.derive(r_idx)}
        checked_namespaces[f"{bank.NAMESPACE}/round-{r_idx}"] = len(l2_seeds)
        overlap = gate1_seeds & l2_seeds
        if overlap:
            collisions.append(
                {"source": f"bl_served_search:round-{r_idx}", "seeds": sorted(overlap)}
            )

    if gate1_seeds & KNOWN_FIXTURE_SEEDS:
        collisions.append(
            {"source": "known_fixture_seeds", "seeds": sorted(gate1_seeds & KNOWN_FIXTURE_SEEDS)}
        )

    dynamic_ns = _discover_watch6_sources([repo, *worktrees])
    for label, ns_seeds in sorted(dynamic_ns.items()):
        overlap = gate1_seeds & ns_seeds
        if overlap:
            collisions.append({"source": f"discovered:{label}", "seeds": sorted(overlap)})

    live_headers = _live_bank_headers(resolved_artifacts / "watch-replay-bank-live")
    live_seeds = {row["seed"] for row in live_headers}
    if gate1_seeds & live_seeds:
        collisions.append(
            {
                "source": str(resolved_artifacts / "watch-replay-bank-live"),
                "seeds": sorted(gate1_seeds & live_seeds),
            }
        )

    prior_contracts: dict[str, int] = {}
    if (resolved_artifacts / "t34-sol").is_dir():
        for contract_path in sorted(
            (resolved_artifacts / "t34-sol").glob("watch-replay*/*/main-contract.json")
        ):
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            c_seeds = {int(r["seed"]) for r in payload.get("rows", [])}
            prior_contracts[str(contract_path)] = len(c_seeds)
            if gate1_seeds & c_seeds:
                collisions.append(
                    {"source": str(contract_path), "seeds": sorted(gate1_seeds & c_seeds)}
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
        if gate1_seeds & values:
            collisions.append({"source": rel, "seeds": sorted(gate1_seeds & values)})
        controls[rel] = {
            "sha256": digest(path),
            "count": len(values),
            "positive_control": max(values),
        }

    matching_files: dict[str, str] = {}
    training_exclusions: list[Path] = []
    if scan_filesystem:
        scan_roots = [repo / name for name in ("research", "docs", "src", "scripts")]
        if resolved_artifacts.is_dir():
            scan_roots.append(resolved_artifacts)
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
            "!**/t34-sol/argon-t4-gate1*/**",
            "!**/research/t4_gate1/**",
        ):
            command += ["--glob", excluded]

        positive_controls = {row["positive_control"] for row in controls.values()}
        candidates = gate1_seeds | positive_controls
        for seed in sorted(candidates):
            command += ["-e", str(seed)]

        found_controls: set[int] = set()
        for root in scan_roots:
            if not root.is_dir():
                continue
            result = subprocess.run(
                [*command, str(root)], capture_output=True, text=True, timeout=300
            )
            if result.returncode not in (0, 1):
                raise ValueError(f"seed audit rg failure on {root}: {result.stderr[:400]}")
            hits = {int(value) for value in result.stdout.split()} & candidates
            found_controls |= hits & positive_controls
            if hits & gate1_seeds:
                collisions.append({"source": str(root), "seeds": sorted(hits & gate1_seeds)})

        if not positive_controls <= found_controls:
            raise ValueError("seed audit positive controls failed")

        if resolved_artifacts.is_dir():
            training_exclusions = sorted(
                resolved_artifacts.glob("frp-*/namespaces*/training_exclusions.json")
            )
            if len(training_exclusions) < 7:
                raise ValueError("historical FRP training exclusion inventory incomplete")
            for path in training_exclusions:
                values = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(values, list) or any(type(x) is not int for x in values):
                    raise ValueError(f"malformed historical training inventory: {path}")
                matching_files[str(path)] = digest(path)
                overlap = gate1_seeds & set(values)
                if overlap:
                    collisions.append({"source": str(path), "seeds": sorted(overlap)})

    if collisions:
        raise ValueError(f"Track-4 Gate 1 seed disjointness audit failed: {collisions}")

    return {
        "status": "CLEAR",
        "namespaces": counts_by_ns,
        "total_seeds_audited": len(gate1_seeds),
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
            "**/*.watch.gz (payload bodies)",
        ],
        "matching_file_hashes": matching_files,
        "allocations": {
            "main": main_alloc,
            "canary": canary_alloc,
            "preflight": preflight_alloc,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = audit()
    publish_json(args.out.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "namespaces": report["namespaces"],
                "total_seeds_audited": report["total_seeds_audited"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
