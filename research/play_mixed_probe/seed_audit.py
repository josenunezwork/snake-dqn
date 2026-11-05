"""Seed disjointness audit and training-exclusion registration for the Play mixed-field probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.apex_safety_20260926 import dev_screen
from research.play_mixed_probe.protocol import (
    NAMESPACE,
    TOTAL_COUNT,
    allocation,
    seed_for,
    training_exclusions,
)
from research.watch6_alllen_screen import protocol as alllen_protocol
from research.watch6_l1_confirm import freshness as l1_freshness
from research.watch6_l1_confirm import protocol as l1_protocol
from research.watch_replay import seed_audit as wr_audit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
DEFAULT_OUT_DIR = DEFAULT_ARTIFACTS / "t34-sol" / "argon-play-mixed-probe"

C1_NAMESPACE = "bl-plan-v3-c1-served-watch6-20261022/v1"
T34_DOMAIN = "t34-exit-b8-20261019/v1"
HERO_STAGE0_NAMESPACE = "t34-hero-probe-20261103-v1"
HERO_STAGE0_COUNT = 48
OPENSTOP_GATE_NAMESPACE = "t34-openstop-gate-20261104-v1"
OPENSTOP_GATE_COUNT = 96

COMMITTED_BANK_FILES: tuple[str, ...] = (
    "research/bl_served_search/BANK_L2.json",
    "research/t34_round0/BANK_ROUND0.json",
    "research/t34_round0d/BANK_ROUND0D.json",
    "research/t34_pathb/RETENTION_BANK_V2.json",
    "research/phase0_search_20261019/WORLD_BANK.json",
    "research/bl_ppo/BANK.json",
    "research/bl_e16/BANK.json",
    "research/t34_o1_harvest/BANK_O1.json",
    "research/t34_o2_harvest/BANK_O2.json",
    "research/t34_o2_harvest/BANK_O2_V2.json",
    "research/s1_mature_20261018/SEED_BANK.json",
    "research/s1_mature_20261018/WORLD_BANK.json",
    "research/t23_r0_screen_20261017/WORLD_BANK.json",
    "research/q0q1_20261017/Q1_WORLD_BANK.json",
    "tools/sandbox/SANDBOX_INPUT_REGISTRY.json",
)

RG_FORBIDDEN_GLOBS: tuple[str, ...] = (
    "!.git/**",
    "!venv/**",
    "!**/.venv/**",
    "!**/__pycache__/**",
    "!**/node_modules/**",
    "!**/exports/**",
    "!**/pair-*.json*",
    "!**/world-*.json*",
    "!**/fixture-world-*.json*",
    "!**/*.watch.gz*",
    "!**/*.npz",
    "!**/*.npy",
    "!**/*.pth",
    "!**/*.pt",
    "!**/*.tar.gz",
    "!**/watch-replay-bank-*/**",
    "!**/t34-sol/l1v1-*/**",
    "!**/t34-sol/*alllen*/**",
    "!**/t34-sol/watch6-efficacy*/**",
    "!**/t34-round0-study-01/**",
    "!**/t34-sol/argon-openstop-gate/**",
    "!**/play_mixed_probe/**",
    "!**/t34-sol/argon-play-mixed-probe/**",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _extract_seeds(value: Any) -> set[int]:
    """Recursively extract integer values stored under any ``*seed*`` key or list."""
    found: set[int] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            k = str(key).lower()
            if "seed" in k:
                if type(item) is int:
                    found.add(item)
                elif isinstance(item, list):
                    found.update(x for x in item if type(x) is int)
            found |= _extract_seeds(item)
    elif isinstance(value, list):
        for item in value:
            found |= _extract_seeds(item)
    return found


def collect_watch6_alllen_screen_seeds(prefix_count: int = 256) -> dict[str, Any]:
    """Collect Watch6 all-length screen main, canary, preflight, and benchmark seeds."""
    main_alloc = alllen_protocol.allocation(alllen_protocol.NAMESPACE, alllen_protocol.COUNT)
    canary_alloc = alllen_protocol.allocation(
        alllen_protocol.CANARY_NAMESPACE, alllen_protocol.CANARY_COUNT
    )
    preflight_alloc = alllen_protocol.allocation(
        alllen_protocol.PREFLIGHT_NAMESPACE, alllen_protocol.PREFLIGHT_COUNT
    )
    seeds: set[int] = set()
    seeds.update(int(r["seed"]) for r in main_alloc["rows"])
    seeds.update(int(r["seed"]) for r in canary_alloc["rows"])
    seeds.update(int(r["seed"]) for r in preflight_alloc["rows"])
    seeds.update(int(s) for s in alllen_protocol.BRANCH_BENCHMARK_SEEDS)
    for ns in (
        alllen_protocol.NAMESPACE,
        alllen_protocol.CANARY_NAMESPACE,
        alllen_protocol.PREFLIGHT_NAMESPACE,
    ):
        for idx in range(prefix_count):
            seeds.add(alllen_protocol.seed_for(ns, idx))
    return {
        "namespaces": [
            alllen_protocol.NAMESPACE,
            alllen_protocol.CANARY_NAMESPACE,
            alllen_protocol.PREFLIGHT_NAMESPACE,
        ],
        "main_count": len(main_alloc["rows"]),
        "canary_count": len(canary_alloc["rows"]),
        "preflight_count": len(preflight_alloc["rows"]),
        "benchmark_seeds": list(alllen_protocol.BRANCH_BENCHMARK_SEEDS),
        "prefix_count_per_namespace": prefix_count,
        "total_unique_seeds": len(seeds),
        "seeds": seeds,
    }


def collect_watch6_l1_v1_seeds(prefix_count: int = 512) -> dict[str, Any]:
    """Collect Watch6-L1-V1 primary + reserve + fixture seeds via protocol functions only."""
    alloc_rows = l1_protocol.allocation()
    primary_seeds = {int(r["seed"]) for r in alloc_rows}
    fixture_seeds = {int(s) for s in l1_protocol.FIXTURE_SEEDS}
    derived_prefixes: set[int] = set()
    for ns in (
        l1_protocol.NAMESPACE,
        "t34-watch6-l1-confirm-canary-20261101-v1",
    ):
        derived_prefixes |= l1_freshness._derive_namespace_seeds(ns, prefix_count)
    all_seeds = primary_seeds | fixture_seeds | derived_prefixes
    return {
        "namespace": l1_protocol.NAMESPACE,
        "primary_count": len(primary_seeds),
        "fixture_count": len(fixture_seeds),
        "reserve_and_prefix_count_per_namespace": prefix_count,
        "outputs_opened": False,
        "total_unique_seeds": len(all_seeds),
        "seeds": all_seeds,
    }


def collect_watch_replay_seeds(prefix_count: int = 256) -> dict[str, Any]:
    """Collect all watch-replay bank and fixture seeds analytically without touching live banks."""
    prefix_names = {
        ns if isinstance(ns, str) else ns[0] for ns in wr_audit.KNOWN_NAMESPACE_PREFIXES
    }
    namespaces = sorted(
        set(wr_audit.DEFAULT_NAMESPACES)
        | prefix_names
        | {ns for ns in l1_freshness.KNOWN_PRIOR_NAMESPACES if "watch-replay" in ns}
    )
    seeds: set[int] = set()
    for ns in namespaces:
        for idx in range(prefix_count):
            seeds.add(wr_audit.seed_for(ns, idx))
    fixed_sets = set(wr_audit.KNOWN_FIXTURE_SEEDS)
    seeds |= {int(s) for s in fixed_sets}
    return {
        "namespace_count": len(namespaces),
        "prefix_count_per_namespace": prefix_count,
        "fixed_seed_count": len(fixed_sets),
        "live_bank_files_touched": False,
        "total_unique_seeds": len(seeds),
        "seeds": seeds,
    }


def collect_hero_stage0_and_openstop_seeds(prefix_count: int = 512) -> dict[str, Any]:
    """Collect Stage 0 probe and open-stop gate seeds analytically via seed_for."""
    stage0_allocated = {seed_for(HERO_STAGE0_NAMESPACE, idx) for idx in range(HERO_STAGE0_COUNT)}
    stage0_prefix = {seed_for(HERO_STAGE0_NAMESPACE, idx) for idx in range(prefix_count)}
    openstop_allocated = {
        seed_for(OPENSTOP_GATE_NAMESPACE, idx) for idx in range(OPENSTOP_GATE_COUNT)
    }
    openstop_prefix = {seed_for(OPENSTOP_GATE_NAMESPACE, idx) for idx in range(prefix_count)}
    all_seeds = stage0_allocated | stage0_prefix | openstop_allocated | openstop_prefix
    return {
        "stage0_namespace": HERO_STAGE0_NAMESPACE,
        "stage0_allocated_count": len(stage0_allocated),
        "openstop_gate_namespace": OPENSTOP_GATE_NAMESPACE,
        "openstop_gate_allocated_count": len(openstop_allocated),
        "prefix_count_per_namespace": prefix_count,
        "gate_worlds_opened": False,
        "total_unique_seeds": len(all_seeds),
        "seeds": all_seeds,
    }


def collect_c1_and_committed_bank_seeds(repo: Path, prefix_count: int = 512) -> dict[str, Any]:
    """Collect C1 seeds, all committed BANK_*/WORLD_BANK/SEED_BANK files, and R6 census seeds."""
    c1_seeds: set[int] = set()
    for idx in range(prefix_count):
        c1_seeds.add(seed_for(C1_NAMESPACE, idx))
        c1_seeds.add(seed_for(f"{C1_NAMESPACE}/primary", idx))
        c1_seeds.add(seed_for(f"{C1_NAMESPACE}/reserve", idx))
    c1_seeds.update(int(s) for s in dev_screen.screen_seeds(prefix_count, T34_DOMAIN, C1_NAMESPACE))

    bank_records: dict[str, Any] = {}
    all_seeds: set[int] = set(c1_seeds)
    positive_controls: list[int] = []

    for rel in COMMITTED_BANK_FILES:
        path = repo / rel
        if not path.is_file():
            raise FileNotFoundError(f"missing committed bank file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        extracted = _extract_seeds(payload)
        if not extracted:
            raise ValueError(f"extracted zero seeds from committed bank file: {rel}")
        if not positive_controls:
            positive_controls.append(min(extracted))
        all_seeds |= extracted
        bank_records[rel] = {
            "sha256": _sha256_file(path),
            "seed_count": len(extracted),
            "seeds": extracted,
        }

    from research.r6_l600_census.seeds import derive_world_seeds as r6_seeds

    r6_set = set(r6_seeds())
    all_seeds |= r6_set

    audit_files_seeds: set[int] = set()
    research_dir = repo / "research"
    if research_dir.is_dir():
        from research.t34_claim_a.seed_audit import namespace_seeds as claim_a_seeds

        for path in sorted(research_dir.rglob("*SEED_AUDIT*.json")):
            if "play_mixed_probe" in path.parts or "exports" in path.parts:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            audit_files_seeds |= _extract_seeds(payload)
            if isinstance(payload, dict) and payload.get("domain") == T34_DOMAIN:
                for ns, entry in payload.get("namespaces", {}).items():
                    if isinstance(entry, dict) and "count" in entry:
                        audit_files_seeds |= set(claim_a_seeds(str(ns), int(entry["count"])))
    all_seeds |= audit_files_seeds

    return {
        "c1_namespace": C1_NAMESPACE,
        "c1_seed_count": len(c1_seeds),
        "r6_l600_census_seed_count": len(r6_set),
        "audit_files_seed_count": len(audit_files_seeds),
        "bank_records": bank_records,
        "positive_controls": positive_controls,
        "total_unique_seeds": len(all_seeds),
        "seeds": all_seeds,
    }


def collect_frp_training_exclusions(artifacts: Path) -> list[dict[str, Any]]:
    """Load all historical FRP training_exclusions.json files under artifacts."""
    if not artifacts.is_dir():
        raise FileNotFoundError(f"artifacts directory not found: {artifacts}")
    files = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(files) < 7:
        raise ValueError(f"expected >=7 FRP training_exclusions.json files, found {len(files)}")
    records: list[dict[str, Any]] = []
    for path in files:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not values or any(type(v) is not int for v in values):
            raise ValueError(f"malformed FRP training_exclusions.json: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "count": len(values),
                "seeds": {int(v) for v in values},
            }
        )
    return records


def run_rg_disjointness_scan(
    candidate_seeds: Sequence[int],
    positive_controls: Sequence[int],
    roots: Sequence[Path],
    *,
    forbidden_globs: Sequence[str] = RG_FORBIDDEN_GLOBS,
) -> dict[str, Any]:
    """Run a single-threaded ripgrep word scan for candidate seeds + positive controls."""
    query_seeds = sorted({int(s) for s in candidate_seeds} | {int(s) for s in positive_controls})
    cmd: list[str] = [
        "rg",
        "--threads",
        "1",
        "--fixed-strings",
        "--word-regexp",
        "--only-matching",
        "--no-heading",
        "--no-filename",
        "--no-line-number",
    ]
    for glob_pat in forbidden_globs:
        cmd.extend(["--glob", glob_pat])
    for s in query_seeds:
        cmd.extend(["-e", str(s)])
    cmd.extend(str(r.resolve()) for r in roots)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"rg scan failed (code={proc.returncode}): {proc.stderr[:400]}")

    matched = {int(tok) for tok in proc.stdout.split() if tok.strip().isdigit()}
    candidate_hits = sorted(matched & set(int(s) for s in candidate_seeds))
    control_hits = sorted(matched & set(int(s) for s in positive_controls))
    if positive_controls and len(control_hits) != len(set(int(s) for s in positive_controls)):
        raise RuntimeError(
            "rg positive control failed: did not detect all known committed bank/probe seeds "
            f"(expected {sorted(set(int(s) for s in positive_controls))}, got {control_hits})"
        )
    return {
        "roots": [str(r.resolve()) for r in roots],
        "threads": 1,
        "forbidden_globs": list(forbidden_globs),
        "positive_controls": [int(s) for s in positive_controls],
        "positive_control_hits": control_hits,
        "candidate_hits": candidate_hits,
    }


def run_seed_audit(
    *,
    namespace: str = NAMESPACE,
    repo: Path = REPO_ROOT,
    artifacts: Path = DEFAULT_ARTIFACTS,
    run_rg: bool = True,
) -> dict[str, Any]:
    """Audit the 320 Play mixed-field probe seeds for disjointness against all prior banks."""
    alloc = allocation(namespace)
    seeds = [int(s) for s in alloc["seeds"]]
    seed_set = set(seeds)
    if len(seed_set) != len(seeds):
        raise ValueError("internal seed collision in allocation")

    alllen_info = collect_watch6_alllen_screen_seeds()
    alllen_collisions = sorted(seed_set & alllen_info.pop("seeds"))

    l1_info = collect_watch6_l1_v1_seeds()
    l1_collisions = sorted(seed_set & l1_info.pop("seeds"))

    wr_info = collect_watch_replay_seeds()
    wr_collisions = sorted(seed_set & wr_info.pop("seeds"))

    stage0_info = collect_hero_stage0_and_openstop_seeds()
    stage0_collisions = sorted(seed_set & stage0_info.pop("seeds"))

    c1_info = collect_c1_and_committed_bank_seeds(repo)
    positive_controls = c1_info.pop("positive_controls")
    c1_all_collisions = sorted(seed_set & c1_info.pop("seeds"))
    bank_summary: dict[str, Any] = {}
    for rel, rec in c1_info.pop("bank_records").items():
        b_hits = sorted(seed_set & rec["seeds"])
        bank_summary[rel] = {
            "sha256": rec["sha256"],
            "seed_count": rec["seed_count"],
            "collisions": b_hits,
        }

    frp_records = collect_frp_training_exclusions(artifacts)
    frp_summary: list[dict[str, Any]] = []
    frp_collisions: list[int] = []
    for rec in frp_records:
        hits = sorted(seed_set & rec["seeds"])
        frp_collisions.extend(hits)
        frp_summary.append(
            {
                "path": rec["path"],
                "sha256": rec["sha256"],
                "count": rec["count"],
                "collisions": hits,
            }
        )

    rg_summary: dict[str, Any] | None = None
    rg_collisions: list[int] = []
    if run_rg:
        scan_roots = [repo]
        if artifacts.is_dir():
            scan_roots.append(artifacts)
        rg_summary = run_rg_disjointness_scan(seeds, positive_controls, scan_roots)
        rg_collisions = rg_summary["candidate_hits"]

    all_collisions = sorted(
        set(alllen_collisions)
        | set(l1_collisions)
        | set(wr_collisions)
        | set(stage0_collisions)
        | set(c1_all_collisions)
        | set(frp_collisions)
        | set(rg_collisions)
    )
    if all_collisions:
        raise ValueError(f"seed audit failed; collisions detected: {all_collisions}")

    excluded_list = training_exclusions(namespace)
    exclusions_bytes = (json.dumps(excluded_list, indent=2) + "\n").encode("utf-8")
    exclusions_sha = _sha256_bytes(exclusions_bytes)

    return {
        "schema": "t34-play-mixed-probe-seed-audit/v1",
        "status": "CLEAR",
        "pass": True,
        "namespace": namespace,
        "candidate_count": len(seeds),
        "unique_count": len(seed_set),
        "internal_collision": False,
        "collisions": [],
        "seeds": seeds,
        "seeds_sha256": alloc["seeds_sha256"],
        "disjointness_checks": {
            "watch6_alllen_screen": {
                **alllen_info,
                "collisions": alllen_collisions,
            },
            "watch6_l1_v1_primary_and_reserve": {
                **l1_info,
                "collisions": l1_collisions,
            },
            "watch_replay_banks": {
                **wr_info,
                "collisions": wr_collisions,
            },
            "hero_stage0_and_openstop_gate": {
                **stage0_info,
                "collisions": stage0_collisions,
            },
            "c1_and_committed_banks": {
                **c1_info,
                "committed_banks": bank_summary,
                "collisions": c1_all_collisions,
            },
            "frp_training_exclusions": {
                "inventory_count": len(frp_summary),
                "inventories": frp_summary,
                "collisions": sorted(set(frp_collisions)),
            },
            "rg_literal_history_scan": rg_summary,
        },
        "training_exclusions": {
            "path": "research/play_mixed_probe/training_exclusions.json",
            "sha256": exclusions_sha,
            "count": len(excluded_list),
            "seeds": excluded_list,
            "rule": (
                f"N={TOTAL_COUNT} Play mixed-field probe worlds ({namespace}); "
                "registered as training-excluded across all training and gate banks."
            ),
        },
        "historical_frp_training_exclusions": frp_summary,
    }


def write_registered_artifacts(
    *,
    repo: Path = REPO_ROOT,
    artifacts: Path = DEFAULT_ARTIFACTS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Run seed audit and write BANK_PLAY_MIXED_PROBE.json, training_exclusions.json, and audit."""
    audit_report = run_seed_audit(repo=repo, artifacts=artifacts, run_rg=True)
    alloc = allocation(NAMESPACE)
    excl = training_exclusions(NAMESPACE)

    excl_text = json.dumps(excl, indent=2) + "\n"
    excl_sha = _sha256_bytes(excl_text.encode("utf-8"))

    alloc_with_binding = {
        **alloc,
        "training_exclusions": {
            "path": "research/play_mixed_probe/training_exclusions.json",
            "sha256": excl_sha,
            "count": len(excl),
        },
    }
    bank_text = json.dumps(alloc_with_binding, indent=2, sort_keys=True) + "\n"
    bank_sha = _sha256_bytes(bank_text.encode("utf-8"))

    audit_report["bank_path"] = "research/play_mixed_probe/BANK_PLAY_MIXED_PROBE.json"
    audit_report["bank_sha256"] = bank_sha
    audit_text = json.dumps(audit_report, indent=2, sort_keys=True) + "\n"

    pkg_dir = repo / "research" / "play_mixed_probe"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target_dir in (pkg_dir, out_dir):
        (target_dir / "training_exclusions.json").write_text(excl_text, encoding="utf-8")
        (target_dir / "BANK_PLAY_MIXED_PROBE.json").write_text(bank_text, encoding="utf-8")
        (target_dir / "SEED_AUDIT.json").write_text(audit_text, encoding="utf-8")

    return audit_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = write_registered_artifacts(
        repo=args.repo, artifacts=args.artifacts, out_dir=args.out_dir
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "namespace": report["namespace"],
                "candidate_count": report["candidate_count"],
                "seeds_sha256": report["seeds_sha256"],
                "bank_sha256": report["bank_sha256"],
                "training_exclusions_sha256": report["training_exclusions"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
