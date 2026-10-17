"""Refreshable, fail-closed world-seed audit for T11 LOCAL30.

T11 reserves the ordinary 96-world development rosters for rounds 74 and 75.
It performs no training or evaluation.  The audit deliberately binds the prior
T6 evidence chain and the live T7 namespace before checking the two new
rosters, so a launch can refresh this receipt immediately before gameplay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from research.redesign_t7_phase_r_20261016 import seed_audit as t7_seed_audit
from research.t6_restart_20261016 import seed_audit as t6_seed_audit
from research.t6_screen_20261017 import audit as t6_screen_audit

INITIAL_ROUND = 74
EXTENSION_ROUND = 75
ROUNDS = (INITIAL_ROUND, EXTENSION_ROUND)
SCHEMA = "t11-local30-seed-audit/v1"
FROZEN_RECEIPT_REL = Path("research/t11_local30_20261017/seed_audit.json")
_ROUND_METADATA_RE = re.compile(rb'"(?:round_index|round)"\s*:\s*(?:74|75)(?![0-9])')
_NAMESPACE_RE = re.compile(rb'"namespace"\s*:\s*"([^"]+)"')


def sha256(path: Path) -> str:
    """Return the SHA256 of one evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_digest(seeds: Iterable[int]) -> str:
    """Return the canonical digest of a roster."""
    return hashlib.sha256(",".join(str(seed) for seed in sorted(set(seeds))).encode()).hexdigest()


def candidate_rosters() -> dict[str, set[int]]:
    """Return T11's fixed initial and one-extension development rosters."""
    from research.redesign_m2_20261008 import ni_spec

    rosters = {
        str(round_index): set(ni_spec.distill_seeds(96, round_index=round_index))
        for round_index in ROUNDS
    }
    if any(len(seeds) != 96 for seeds in rosters.values()):
        raise ValueError("T11 round generation did not produce 96 worlds per roster")
    if rosters[str(INITIAL_ROUND)] & rosters[str(EXTENSION_ROUND)]:
        raise ValueError("T11 initial and extension rosters collide")
    return rosters


def _live_t7_namespace(repo: Path) -> tuple[dict[str, Any], set[int]]:
    """Validate and return T7's running round-76 namespace."""
    result, seeds = t6_screen_audit._t7_namespace(repo)
    if result.get("pass") is not True:
        raise ValueError("T7 Phase R namespace receipt is missing or invalid")
    return result, seeds


def _prior_t7_banks() -> tuple[dict[str, Any], set[int]]:
    """Return the rejected T7 round-74/75 identities as permanent exclusions."""
    from research.apex_safety_20260926 import dev_screen

    values: set[int] = set()
    for round_index in (74, 75):
        for run_seed in range(5):
            values.update(
                dev_screen.screen_seeds(
                    t7_seed_audit.SEED_COUNT,
                    t7_seed_audit.BANK_NS,
                    f"round{round_index}-seed{run_seed}",
                )
            )
    expected = 2 * 5 * t7_seed_audit.SEED_COUNT
    if len(values) != expected:
        raise ValueError("T7 rejected namespace has duplicate or incomplete identities")
    return (
        {
            "namespace": t7_seed_audit.BANK_NS,
            "rounds": [74, 75],
            "world_count": len(values),
            "seed_digest": seed_digest(values),
        },
        values,
    )


def _registered_history(repo: Path, artifacts: Path) -> tuple[dict[str, Any], set[int]]:
    """Bind prior bank sources, including all rounds through T6 and T7 exclusions."""
    from research.redesign_m2_20261008 import ni_spec
    from research.t4f_return_20261016 import seed_audit as t4f_seed_audit

    t2, values, history = t4f_seed_audit._registered(repo)
    if t2.get("pass") is not True:
        raise ValueError("registered T2 history is incomplete")
    consumed = {
        f"development_round_{round_index}": set(ni_spec.distill_seeds(96, round_index=round_index))
        for round_index in range(70, 74)
    }
    for seeds in consumed.values():
        values.update(seeds)
    additional, additional_seeds = t6_screen_audit._additional_historical_namespaces(artifacts)
    if additional.get("pass") is not True:
        raise ValueError("additional historical namespace inventory is invalid")
    rejected, rejected_seeds = _prior_t7_banks()
    values.update(additional_seeds)
    values.update(rejected_seeds)
    return (
        {
            "registered_t2_audit": {"schema": t2.get("schema"), "pass": t2.get("pass")},
            "base_history": {
                name: {"count": len(seeds), "seed_digest": seed_digest(seeds)}
                for name, seeds in history.items()
            },
            "consumed_development_rounds": {
                name: {"count": len(seeds), "seed_digest": seed_digest(seeds)}
                for name, seeds in consumed.items()
            },
            "additional_historical_namespaces": additional,
            "t7_rejected_banks": rejected,
        },
        values,
    )


def _round_metadata_scope(root: Path, excludes: tuple[Path, ...]) -> dict[str, list[str]]:
    """Separate T7's namespaced rounds from unknown development metadata."""
    ignored = {path.resolve() for path in excludes}
    known_t7: list[str] = []
    unknown: list[str] = []
    for parent, _directories, names in os.walk(root):
        for name in sorted(names):
            path = Path(parent) / name
            if path.resolve() in ignored or path.suffix not in {".json", ".jsonl"}:
                continue
            raw = path.read_bytes()
            if not _ROUND_METADATA_RE.search(raw):
                continue
            try:
                payloads = (
                    [json.loads(line) for line in raw.splitlines() if line.strip()]
                    if path.suffix == ".jsonl"
                    else [json.loads(raw)]
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                # A malformed file can hide record boundaries or namespace scope.
                unknown.append(str(path))
                continue
            classifications: set[str] = set()

            def visit(value: Any, inherited_namespace: str | None = None) -> None:
                if isinstance(value, list):
                    for item in value:
                        visit(item, inherited_namespace)
                    return
                if not isinstance(value, dict):
                    return
                namespace = value.get("namespace", inherited_namespace)
                for key in ("round", "round_index"):
                    if value.get(key) in ROUNDS:
                        classifications.add(
                            "known_t7"
                            if namespace == t7_seed_audit.BANK_NS
                            else "unknown_or_development"
                        )
                for child in value.values():
                    visit(child, namespace if isinstance(namespace, str) else None)

            for payload in payloads:
                visit(payload)
            # A raw match that cannot be tied to integer JSON metadata is ambiguous.
            if not classifications:
                classifications.add("unknown_or_development")
            if classifications == {"known_t7"}:
                known_t7.append(str(path))
            else:
                unknown.append(str(path))
    return {"known_t7_namespace": known_t7, "unknown_or_development": unknown}


def _frozen_receipt_exclusion(
    repo: Path, output: Path | None, rosters: dict[str, set[int]]
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    """Bind the immutable preregistration receipt while allowing a fresh launch audit."""
    frozen = (repo / FROZEN_RECEIPT_REL).resolve()
    if output == frozen and frozen.exists():
        raise FileExistsError("the frozen T11 preregistration receipt must not be overwritten")
    rows: list[dict[str, Any]] = []
    if frozen.exists():
        payload = json.loads(frozen.read_text())
        checks = payload.get("checks") if isinstance(payload, dict) else None
        valid = (
            isinstance(payload, dict)
            and payload.get("schema") == SCHEMA
            and isinstance(checks, dict)
            and all(
                checks.get(key, {}).get("seeds") == sorted(seeds) for key, seeds in rosters.items()
            )
        )
        if not valid:
            raise ValueError(
                "frozen T11 preregistration receipt does not bind the expected rosters"
            )
        rows.append(
            {
                "path": str(frozen),
                "sha256": sha256(frozen),
                "kind": "frozen-preregistration-receipt",
            }
        )
    elif output != frozen:
        raise FileNotFoundError(
            "frozen T11 preregistration receipt is required for a launch refresh"
        )
    exclusions = tuple(path for path in (frozen, output) if path is not None)
    return tuple(dict.fromkeys(exclusions)), rows


def audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Audit T11 round 74/75 freshness against every known prior bank.

    ``out`` is the sole self-exclusion.  Rerunning the audit rescans the full
    visible repository and artifact roots, so any newly recorded use of either
    roster makes the receipt fail closed.
    """
    source_path = Path(__file__).resolve()
    source_hash = sha256(source_path)
    repo_path = Path(repo).resolve()
    artifact_root = (
        Path(artifacts).resolve()
        if artifacts is not None
        else repo_path.parent / "snake-dqn-artifacts"
    )
    output = Path(out).resolve() if out is not None else None
    if output is not None and (output.is_dir() or output.suffix != ".json"):
        raise ValueError("out must name exactly one JSON audit receipt")
    if not (repo_path / "research").is_dir() or not artifact_root.is_dir():
        raise FileNotFoundError("repository research and artifact root are required")

    rosters = candidate_rosters()
    excluded, frozen_receipts = _frozen_receipt_exclusion(repo_path, output, rosters)
    history, registered = _registered_history(repo_path, artifact_root)
    t7, t7_seeds = _live_t7_namespace(repo_path)
    registered.update(t7_seeds)

    reserved = set().union(*rosters.values())
    collection, collection_seeds = t6_seed_audit.t4e_collection_audit(
        artifact_root / "t4e-return-bank-20261016"
    )
    snapshot, snapshot_seeds = t6_seed_audit.snapshot_store_audit(
        artifact_root / "t4b-snapshot-restart-20261016/full"
    )
    training = t6_seed_audit.training_namespace_audit(repo_path, artifact_root, reserved)
    if not collection.get("pass") or not snapshot.get("pass") or not training.get("pass"):
        raise ValueError("a bound T6 source namespace audit does not pass")
    registered.update(collection_seeds)
    registered.update(snapshot_seeds)

    roots = (repo_path / "research", artifact_root)
    scans = {
        str(root): t6_seed_audit.concrete_text_hits(
            root, [str(seed) for seed in sorted(reserved)], excluded
        )
        for root in roots
    }
    observed = {
        int(token) for hits, _progress in scans.values() for token in hits if token.isdigit()
    }
    metadata_scans = {
        str(root): t6_seed_audit._partitioned_t6_rg(
            root,
            [
                "--pcre2",
                r'"(?:round_index|round)"\s*:\s*(?:74|75)(?![0-9])',
                "--glob",
                "*.json",
                "--glob",
                "*.jsonl",
                "--only-matching",
            ],
            excluded,
            "t11-round-metadata",
        )
        for root in roots
    }
    metadata_scan_found = {name: bool(hits) for name, (hits, _progress) in metadata_scans.items()}
    metadata_scope = {str(root): _round_metadata_scope(root, excluded) for root in roots}
    metadata_found = {
        name: bool(rows["unknown_or_development"]) for name, rows in metadata_scope.items()
    }
    checks: dict[str, dict[str, Any]] = {}
    for round_index, seeds in rosters.items():
        other = set().union(*(row for key, row in rosters.items() if key != round_index))
        checks[round_index] = {
            "count": len(seeds),
            "unique": len(seeds) == 96,
            "seed_digest": seed_digest(seeds),
            "seeds": sorted(seeds),
            "registered_collisions": sorted(seeds & registered),
            "observed_json_collisions": sorted(seeds & observed),
            "t7_running_namespace_collisions": sorted(seeds & t7_seeds),
            "other_reserved_round_collisions": sorted(seeds & other),
        }
    passed = not any(metadata_found.values()) and all(
        row["unique"]
        and not row["registered_collisions"]
        and not row["observed_json_collisions"]
        and not row["t7_running_namespace_collisions"]
        and not row["other_reserved_round_collisions"]
        for row in checks.values()
    )
    if sha256(source_path) != source_hash:
        raise RuntimeError("seed-audit source changed while the audit was running")
    return {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source_sha256": source_hash,
        "scope": (
            "fresh T11 development rounds 74 and 75; every registered prior bank through T6; "
            "T4e collection, snapshot, and bounded training namespaces; rejected T7 allocations; "
            "and the running T7 Phase R namespace"
        ),
        "excluded_output": str(output) if output is not None else None,
        "exact_documentary_exclusions": frozen_receipts,
        "exclusion_policy": (
            "Only the exact frozen preregistration receipt and exact caller output receipt are "
            "excluded; no directory is excluded."
        ),
        "registered_history": history,
        "t6_source_namespaces": {
            "t4e_collection": collection,
            "snapshot_store": snapshot,
            "training_world_inventory": training,
        },
        "t7_running_namespace": t7,
        "text_scan": {
            "method": "t6-partitioned-rg-no-ignore-hidden/threads1/timeout60",
            "roots": [str(root) for root in roots],
            "candidate_count": len(reserved),
            "hits": {name: sorted(hits) for name, (hits, _progress) in scans.items()},
            "reserved_round_metadata": metadata_found,
            "round_metadata_scope": metadata_scope,
            "round_metadata_raw_hits": metadata_scan_found,
            "progress": {name: progress for name, (_hits, progress) in scans.items()},
            "metadata_progress": {
                name: progress for name, (_hits, progress) in metadata_scans.items()
            },
        },
        "checks": checks,
        "limit": (
            "No claim is made about unrecorded experiments, opaque binary files, "
            "or external systems."
        ),
    }


def main() -> int:
    """Write one refreshable audit receipt without starting gameplay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    receipt = audit(artifacts=args.artifacts, out=args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": receipt["pass"], "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
