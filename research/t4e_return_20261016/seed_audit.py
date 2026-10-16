"""Fail-closed freshness audit for the T4e evaluation rounds 68 and 69.

This audit inventories every JSON/JSONL receipt below repository ``research``
and the artifact root.  It excludes only the exact audit receipt being written:
T4e results and collection artifacts remain evidence once they exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.t3_screen_20261015 import banks as prior_banks  # noqa: E402

INITIAL_ROUND = 68
EXTENSION_ROUND = 69
ROUNDS = (INITIAL_ROUND, EXTENSION_ROUND)


def _seed_digest(seeds: Iterable[int]) -> str:
    return hashlib.sha256(",".join(str(seed) for seed in sorted(set(seeds))).encode()).hexdigest()


def _partitioned_rg(
    root: Path,
    arguments: list[str],
    excludes: tuple[Path, ...],
    label: str,
    *,
    with_filenames: bool = False,
) -> tuple[set[str], list[dict[str, str]]]:
    """Search all text below ``root`` in fail-closed, 60-second partitions."""
    excluded = {path.resolve() for path in excludes}
    pending = sorted(root.iterdir(), key=lambda path: path.name)
    hits: set[str] = set()
    progress: list[dict[str, str]] = []
    while pending:
        target = pending.pop(0)
        if target.resolve() in excluded:
            progress.append({"target": str(target), "status": "excluded-exact-receipt"})
            continue
        if target.is_dir() and any(target.resolve() in path.parents for path in excluded):
            pending[0:0] = sorted(target.iterdir(), key=lambda path: path.name)
            progress.append({"target": str(target), "status": "split-exclusion-ancestor"})
            continue
        command = [
            "rg",
            "--no-config",
            "--no-ignore-global",
            "--no-require-git",
            "--threads",
            "1",
            "--no-filename",
            "--no-heading",
            "--color",
            "never",
            *arguments,
            "--",
            str(target),
        ]
        if with_filenames:
            command[10:10] = ["--with-filename"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            if not target.is_dir():
                raise RuntimeError(f"T4e {label} scan timed out on indivisible file: {target}")
            children = sorted(target.iterdir(), key=lambda path: path.name)
            if not children:
                raise RuntimeError(f"T4e {label} scan timed out on empty directory: {target}")
            pending[0:0] = children
            progress.append({"target": str(target), "status": "timeout-split"})
            continue
        if result.returncode not in (0, 1):
            raise RuntimeError(f"T4e {label} scan failed: {result.stderr[-2000:]}")
        lines = result.stdout.splitlines()
        hits.update(lines)
        entry = {"target": str(target), "status": "complete", "match_count": str(len(lines))}
        if with_filenames and lines:
            entry["match_paths"] = json.dumps(sorted({line.rsplit(":", 1)[0] for line in lines}))
        progress.append(entry)
    return hits, progress


def concrete_text_hits(
    root: Path, tokens: list[str], excludes: tuple[Path, ...] = ()
) -> tuple[set[str], list[dict[str, str]]]:
    """Search complete visible scope for exact integer tokens, partitioning on timeout."""
    arguments = ["--fixed-strings", "--word-regexp", "--only-matching"]
    for token in tokens:
        arguments.extend(["-e", token])
    rows, progress = _partitioned_rg(
        root, arguments, excludes, "concrete-token", with_filenames=True
    )
    return {row.rsplit(":", 1)[-1] for row in rows}, progress


def reserved_round_metadata(
    root: Path, excludes: tuple[Path, ...] = ()
) -> tuple[bool, list[dict[str, str]]]:
    """Search every recorded JSON/JSONL receipt for consumed T4e round metadata."""
    hits, progress = _partitioned_rg(
        root,
        [
            "--pcre2",
            '"(?:round_index|round)"\\s*:\\s*(?:68|69)(?![0-9])',
            "--glob",
            "*.json",
            "--glob",
            "*.jsonl",
            "--only-matching",
        ],
        excludes,
        "round-metadata",
    )
    return bool(hits), progress


def _registered(repo: Path) -> tuple[dict[str, Any], set[int]]:
    """Return the complete registered/generated history T4e must avoid."""
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    t2 = prior_banks._registered_t2(repo)
    values = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds())
    for seeds in t2["registered_rounds"].values():
        values.update(seeds)
    for item in t2["historical_distillation"].values():
        values.update(item["seeds"])
    # T2 registers 52 and 60/61; subsequent evaluation banks consumed 62--67.
    for round_index in (52, 60, 61, 62, 63, 64, 65, 66, 67):
        values.update(ni_spec.distill_seeds(96, round_index=round_index))
    return t2, values


def seed_bank_audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    exclude_out: str | Path | None = None,
    documentary_exclusions: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Audit T4e's fresh round-68 and round-69 rosters before any evaluation starts."""
    from research.redesign_m2_20261008 import ni_spec

    repo_path = Path(repo).resolve()
    artifact_root = (
        Path(artifacts).resolve()
        if artifacts is not None
        else repo_path.parent / "snake-dqn-artifacts"
    )
    output = Path(exclude_out).resolve() if exclude_out is not None else None
    if output is not None and (output.is_dir() or output.suffix != ".json"):
        raise ValueError("exclude_out must name exactly one JSON audit receipt")
    excluded = tuple(
        path.resolve()
        for path in ((output,) if output is not None else ())
        + tuple(Path(path) for path in documentary_exclusions)
    )
    if any(path.is_dir() for path in excluded):
        raise ValueError("documentary exclusions must name exact files")
    documentary = []
    for path in excluded:
        if output is not None and path == output and not path.exists():
            documentary.append(
                {"path": str(path), "exists_at_scan": False, "reason": "current-audit-receipt"}
            )
            continue
        if not path.is_file():
            raise ValueError(f"documentary exclusion missing: {path}")
        documentary.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "reason": (
                    "current-audit-receipt"
                    if output is not None and path == output
                    else "own-transcript-or-prior-audit-receipt"
                ),
            }
        )
    t2, registered = _registered(repo_path)
    expected = {
        str(round_index): set(ni_spec.distill_seeds(96, round_index=round_index))
        for round_index in ROUNDS
    }
    roots = (repo_path / "research", artifact_root)
    if not all(root.is_dir() for root in roots):
        raise ValueError("repository research and artifact root must both exist")
    # The artifact root contains large binary banks.  Search only the concrete 192
    # candidate integers with ripgrep's single-thread text path; it covers every
    # visible result/receipt without walking or decoding those banks.
    reserved = set().union(*expected.values())
    tokens = [str(seed) for seed in sorted(reserved)]
    scans = {str(root): concrete_text_hits(root, tokens, excluded) for root in roots}
    scanned_hits = {name: value[0] for name, value in scans.items()}
    observed = {int(token) for hits in scanned_hits.values() for token in hits if token.isdigit()}
    metadata_scans = {str(root): reserved_round_metadata(root, excluded) for root in roots}
    reserved_metadata = {name: value[0] for name, value in metadata_scans.items()}
    namespace_path = artifact_root / "t3-rollout-labels-20261015/bank/namespace.json"
    namespace: dict[str, Any] = {"exists": False, "manifest_binding": {"ok": False}}
    if namespace_path.is_file():
        raw = json.loads(namespace_path.read_text())
        listed = raw.get("seeds", [])
        namespace = {
            "exists": True,
            "schema": raw.get("schema"),
            "count": len(listed) if isinstance(listed, list) else 0,
            "unique": isinstance(listed, list) and len(listed) == len(set(listed)),
            "manifest_binding": prior_banks._manifest_binding(
                namespace_path.parent, namespace_path
            ),
        }
        observed.update(seed for seed in listed if type(seed) is int and seed >= 0)
    checks: dict[str, dict[str, Any]] = {}
    for round_index, seeds in expected.items():
        other = set().union(
            *(candidate for key, candidate in expected.items() if key != round_index)
        )
        checks[round_index] = {
            "count": len(seeds),
            "unique": len(seeds) == 96,
            "seed_digest": _seed_digest(seeds),
            "seeds": sorted(seeds),
            "registered_collisions": sorted(seeds & registered),
            "observed_json_collisions": sorted(seeds & observed),
            "other_reserved_round_collisions": sorted(seeds & other),
        }
    namespace_valid = (
        namespace["exists"]
        and namespace.get("schema") == "t3-label-namespace/v1"
        and bool(namespace.get("count"))
        and namespace.get("unique") is True
        and namespace["manifest_binding"].get("ok") is True
    )
    passed = (
        t2["pass"]
        and namespace_valid
        and not any(reserved_metadata.values())
        and all(
            check["unique"]
            and not check["registered_collisions"]
            and not check["observed_json_collisions"]
            and not check["other_reserved_round_collisions"]
            for check in checks.values()
        )
    )
    return {
        "schema": "t4e-seed-bank-audit/v1",
        "status": "PASS" if passed else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "generated NI/distillation/history rounds 52 and 60--67; all repository research "
            "and artifact-root JSON/JSONL evidence"
        ),
        "excluded_output": str(output) if output is not None else None,
        "exact_documentary_exclusions": documentary,
        "registered_t2_audit": {"pass": t2["pass"], "schema": t2["schema"]},
        "t3_rollout_training_namespace": namespace,
        "text_scan": {
            "method": "rg-fixed-word/threads1/timeout60",
            "roots": [str(root) for root in roots],
            "candidate_count": len(tokens),
            "hits": {name: sorted(hits) for name, hits in scanned_hits.items()},
            "reserved_round_metadata": reserved_metadata,
            "progress": {name: value[1] for name, value in scans.items()},
            "metadata_progress": {name: value[1] for name, value in metadata_scans.items()},
        },
        "checks": checks,
        "pass": passed,
        "limit": (
            "No claim is made about unrecorded experiments, arbitrary binary files, or "
            "external systems."
        ),
    }


def main() -> int:
    """Write the exact receipt excluded from its own audit scan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-documentary", type=Path, action="append", default=[])
    args = parser.parse_args()
    receipt = seed_bank_audit(
        exclude_out=args.out, documentary_exclusions=tuple(args.exclude_documentary)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": receipt["pass"], "checks": receipt["checks"]}, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
