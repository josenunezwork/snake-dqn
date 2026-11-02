"""Pre-data freshness & disjointness audit for Watch6-L1-V1; never opens exports/ or pair-*.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_served_search import bank
from research.watch6_l1_confirm import protocol as p
from tools.ops.package_spec import publish_json

KNOWN_PRIOR_NAMESPACES: tuple[tuple[str, int], ...] = (
    # Watch6 V1, V2, and V3 confirmation studies
    ("t34-watch6-exact-stop-efficacy-20261030-v1", 256),
    ("t34-watch6-exact-stop-efficacy-20261030-v2", 256),
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
    ("t34-explore-watch6-alllen-canary-20261031-r1", 64),
    ("t34-explore-watch6-alllen-preflight-20261031-r1", 64),
    # R6 L>=600 census and T4 Gate 1 exploration
    ("t34-r6-watch6-v2-l600-census-20261031", 256),
    ("t34-explore-t4-gate1-1031-r4", 256),
    ("t34-explore-t4-gate1-canary-1031-r4", 64),
    ("t34-explore-t4-gate1-preflight-1031-r4", 64),
    # Watch replay banks 01..32 and L1 banks 14l1..32l1
    *((f"t34-watch-replay-fix-20261030-bank{i:02d}", 128) for i in range(1, 33)),
    *((f"t34-watch-replay-20261030-bank{i:02d}", 128) for i in range(1, 33)),
    *((f"t34-watch-replay-20261031-bank{i:02d}", 128) for i in range(1, 33)),
    *((f"t34-watch-replay-20261101-bank{i:02d}", 128) for i in range(1, 33)),
    *((f"t34-watch-replay-20261101-bank{i:02d}l1", 128) for i in range(1, 33)),
    *((f"t34-watch-replay-bank-{i:02d}", 128) for i in range(1, 33)),
    *((f"watch-replay-bank-{i:02d}", 128) for i in range(1, 33)),
    ("t34-watch-replay-full-size-fixture-v1", 64),
    ("t34-watch-replay-mac-fixture-v1", 64),
    ("t34-watch-replay-parity-fixture-v1", 64),
    # C1 / L2 / B2 served-search and T12/T13/T15/Round0/S1/Q1 banks
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
    (bank.NAMESPACE, 512),
)

KNOWN_PRIOR_FIXTURE_SEEDS: frozenset[int] = frozenset(
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


def _derive_namespace_seeds(prefix: str, count: int) -> set[int]:
    derived: set[int] = set()
    for index in range(count):
        for key in (
            f"{prefix}/{index}",
            f"{prefix}/primary/{index}",
            f"{prefix}/reserve/{index}",
        ):
            derived.add(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big"))
    return derived


def _assert_not_sealed_export(path: Path) -> None:
    if (
        "exports" in path.parts
        or path.name.startswith("pair-")
        or path.name.startswith("world-")
        or path.name.startswith("fixture-world-")
    ):
        raise ValueError(f"refusing to read sealed export or unit payload: {path}")


def _extract_json_seeds(value: Any, out: set[int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key in ("seed", "world_seed")
                and isinstance(item, int)
                and not isinstance(item, bool)
            ):
                out.add(item)
            elif key in ("seeds", "world_seeds", "excluded_seeds") and isinstance(item, list):
                out.update(x for x in item if isinstance(x, int) and not isinstance(x, bool))
            else:
                _extract_json_seeds(item, out)
    elif isinstance(value, list):
        for item in value:
            _extract_json_seeds(item, out)


def _rg_hits(seeds: list[int], target: Path, *, extra_globs: list[str] | None = None) -> list[str]:
    if not target.exists() or not seeds:
        return []
    cmd = [
        "rg",
        "--threads",
        "1",
        "--hidden",
        "--no-ignore",
        "-F",
        "-w",
        "-n",
        "--no-heading",
    ]
    for glob_pat in bank.GLOBS:
        cmd.extend(["--glob", glob_pat])
    for excluded in (
        "!**/.git/**",
        "!**/venv/**",
        "!**/node_modules/**",
        "!**/__pycache__/**",
        f"!**/{bank.SEALED}/**",
        "!**/exports/**",
        "!**/pair-*.json*",
        "!**/world-*.json*",
        "!**/fixture-world-*.json*",
        "!**/*.watch.gz*",
        "!**/watch-replay-bank-live*/**",
    ):
        cmd.extend(["--glob", excluded])
    for glob_pat in extra_globs or []:
        cmd.extend(["--glob", glob_pat])
    for seed in sorted(set(seeds)):
        cmd.extend(["-e", str(seed)])
    cmd.append(str(target))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"rg seed scan failed ({proc.returncode}): {proc.stderr.strip()}")
    lines: list[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        file_part = line.split(":", 1)[0]
        _assert_not_sealed_export(Path(file_part))
        lines.append(line)
    return lines


def audit(
    *,
    artifacts_dir: Path | None = None,
    own_artifact_subdir: Path | None = None,
) -> dict:
    """Audit Watch6-L1-V1 primary (96) and fixture (3) seeds for uniqueness and disjointness."""
    primary = [row["seed"] for row in p.allocation()]
    fixtures = list(p.FIXTURE_SEEDS)
    combined = primary + fixtures
    if len(primary) != p.COUNT or len(set(combined)) != len(combined):
        raise ValueError("internal seed collision within Watch6-L1-V1 primary/fixture sets")
    combined_set = set(combined)

    namespace_overlaps: dict[str, list[int]] = {}
    for prefix, count in KNOWN_PRIOR_NAMESPACES:
        prior = _derive_namespace_seeds(prefix, count)
        hit = sorted(combined_set & prior)
        if hit:
            namespace_overlaps[prefix] = hit

    fixture_overlaps = sorted(combined_set & KNOWN_PRIOR_FIXTURE_SEEDS)

    committed_overlaps: dict[str, list[int]] = {}
    committed_counts: dict[str, int] = {}
    positive_controls: list[int] = []
    for rel in COMMITTED_BANK_FILES:
        file_path = p.ROOT / rel
        if not file_path.is_file():
            raise ValueError(f"missing committed bank file for positive-control check: {rel}")
        _assert_not_sealed_export(file_path)
        found: set[int] = set()
        _extract_json_seeds(json.loads(file_path.read_text(encoding="utf-8")), found)
        if not found:
            raise ValueError(f"positive-control failure: extracted 0 seeds from {rel}")
        committed_counts[rel] = len(found)
        positive_controls.append(min(found))
        hit = sorted(combined_set & found)
        if hit:
            committed_overlaps[rel] = hit

    # Verify _rg_hits positive controls actually match all committed bank files.
    control_hits = _rg_hits(positive_controls, p.ROOT / "research")
    if len(control_hits) < len(COMMITTED_BANK_FILES):
        raise ValueError("freshness rg positive-control verification failed")

    resolved_artifacts = (
        artifacts_dir if artifacts_dir is not None else (p.ROOT.parent / "snake-dqn-artifacts")
    )

    frp_files: list[Path] = []
    frp_overlaps: dict[str, list[int]] = {}
    if resolved_artifacts.is_dir() and (
        artifacts_dir is None or any(resolved_artifacts.glob("frp-*"))
    ):
        frp_files = sorted(resolved_artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
        if len(frp_files) < 7:
            raise ValueError(
                f"expected >=7 FRP training_exclusions.json files, found {len(frp_files)}"
            )
        for fpath in frp_files:
            _assert_not_sealed_export(fpath)
            values = json.loads(fpath.read_text(encoding="utf-8"))
            if not isinstance(values, list) or any(type(x) is not int for x in values):
                raise ValueError(f"malformed historical training inventory: {fpath}")
            hit = sorted(combined_set & set(values))
            if hit:
                frp_overlaps[str(fpath.relative_to(resolved_artifacts))] = hit

    # Primary seeds are derived via SHA-256 and never appear as literals in code; scan entire repo.
    repo_primary_hits = _rg_hits(
        primary,
        p.ROOT,
        extra_globs=[
            "!**/.pytest_cache/**",
            "!**/.pytest_tmp*/**",
            "!**/.pytest-tmp*/**",
            "!**/research/watch6_l1_confirm/REGISTRATION_L1_V1.md",
        ],
    )
    # Fixture seeds appear literally in research/watch6_l1_confirm/ and its test; scan rest of repo.
    repo_fixture_hits = _rg_hits(
        fixtures,
        p.ROOT,
        extra_globs=[
            "!**/.pytest_cache/**",
            "!**/.pytest_tmp*/**",
            "!**/.pytest-tmp*/**",
            "!**/research/watch6_l1_confirm/**",
            "!**/tests/test_watch6_l1_confirm.py",
        ],
    )

    artifact_globs: list[str] = ["!**/t34-sol/argon-l1v1-pkg*/**"]
    if own_artifact_subdir is not None and resolved_artifacts.exists():
        try:
            rel_own = own_artifact_subdir.resolve().relative_to(resolved_artifacts.resolve())
            artifact_globs.append(f"!**/{rel_own.as_posix()}/**")
        except ValueError:
            pass
    artifact_hits = _rg_hits(combined, resolved_artifacts, extra_globs=artifact_globs)

    clear = (
        not namespace_overlaps
        and not fixture_overlaps
        and not committed_overlaps
        and not frp_overlaps
        and not repo_primary_hits
        and not repo_fixture_hits
        and not artifact_hits
    )
    report = {
        "schema": "watch6-l1-seed-freshness/v1",
        "status": "CLEAR" if clear else "COLLISION",
        "namespace": p.NAMESPACE,
        "primary_count": len(primary),
        "fixture_count": len(fixtures),
        "prior_namespaces_checked": len(KNOWN_PRIOR_NAMESPACES),
        "committed_bank_seed_counts": committed_counts,
        "frp_exclusion_files_checked": len(frp_files),
        "namespace_overlaps": namespace_overlaps,
        "fixture_overlaps": fixture_overlaps,
        "committed_overlaps": committed_overlaps,
        "frp_overlaps": frp_overlaps,
        "repo_primary_hits": repo_primary_hits,
        "repo_fixture_hits": repo_fixture_hits,
        "artifact_hits": artifact_hits,
    }
    if not clear:
        raise ValueError(f"Watch6-L1-V1 seed freshness collision detected: {json.dumps(report)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args()
    report = audit(artifacts_dir=args.artifacts_dir)
    if args.out is not None:
        publish_json(args.out.resolve(), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
