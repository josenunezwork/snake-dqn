"""Fail-closed freshness audit for the T12 B8 serving qualification.

This module allocates the protocol's fixed, purpose-labelled world seeds and
audits them before qualification gameplay.  It never starts a game.  The
audit extends the frozen T7 freshness boundary with T7's final and rejected
allocations, then searches all currently visible JSON/JSONL evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from research.apex_safety_20260926 import dev_screen
from research.redesign_t7_phase_r_20261016 import seed_audit as t7_seed_audit
from research.t4e_return_20261016 import seed_audit as t4e_seed_audit


NAMESPACE = "t12-b8-serving-20261017/v1"
SCHEMA = "t12-b8-serving-seed-audit/v1"
PURPOSE_COUNTS = {
    "watch": 25,
    "play": 25,
    "parity": 2,
    "latency": 4,
    "confidence": 1,
    "ui": 1,
    "rollback": 2,
    "controls": 1,
    "p2_h5000": 9,
    "p2_h10000": 3,
}
T7_RECEIPT_REL = Path("research/redesign_t7_phase_r_20261016/SEED_AUDIT_T7.json")
ARTIFACT_OUTPUT_DIRNAME = "t12-b8-serving-20261017"


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_digest(seeds: Iterable[int]) -> str:
    """Return a canonical digest for a seed collection."""
    return hashlib.sha256(",".join(str(seed) for seed in sorted(set(seeds))).encode()).hexdigest()


def seed_for(purpose: str, index: int, namespace: str = NAMESPACE) -> int:
    """Return one protocol seed from the registered uint32 big-endian recipe."""
    if purpose not in PURPOSE_COUNTS:
        raise ValueError(f"unknown T12 seed purpose: {purpose}")
    if not 0 <= index < PURPOSE_COUNTS[purpose]:
        raise ValueError(f"index {index} is outside the {purpose} roster")
    material = f"{namespace}|{purpose}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def generate_banks(namespace: str = NAMESPACE) -> dict[str, Any]:
    """Return the complete ordered qualification allocation without filesystem access."""
    banks = {
        purpose: [seed_for(purpose, index, namespace) for index in range(count)]
        for purpose, count in PURPOSE_COUNTS.items()
    }
    all_seeds = [seed for seeds in banks.values() for seed in seeds]
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("T12 purpose rosters collide; allocate a new namespace before gameplay")
    return {
        "namespace": namespace,
        "recipe": "uint32_be(sha256(f'{namespace}|{purpose}|{i}')[:4])",
        "banks": {
            purpose: {"count": len(seeds), "seeds": seeds, "seed_digest": seed_digest(seeds)}
            for purpose, seeds in banks.items()
        },
        "count": len(all_seeds),
        "seed_digest": seed_digest(all_seeds),
    }


def _t7_receipt(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate T7's frozen passing receipt and its mandatory evidence binding."""
    path = repo / T7_RECEIPT_REL
    if not path.is_file():
        raise FileNotFoundError(f"T7 freshness receipt is missing: {path}")
    receipt = json.loads(path.read_text())
    if receipt.get("pass") is not True or receipt.get("schema") != t7_seed_audit.SCHEMA:
        raise ValueError("T7 freshness receipt is not a passing T7 audit")
    if receipt.get("audit_source_sha256") != sha256(Path(t7_seed_audit.__file__)):
        raise ValueError("T7 audit source does not match its frozen receipt")
    evidence = receipt.get("mandatory_evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("T7 freshness receipt has no mandatory evidence manifest")
    checked: dict[str, Any] = {}
    for name, row in evidence.items():
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError(f"invalid T7 mandatory evidence row: {name}")
        candidate = Path(row["path"])
        if not candidate.exists():
            raise FileNotFoundError(f"T7 mandatory evidence is missing: {candidate}")
        expected_hash = row.get("sha256")
        actual_hash = sha256(candidate) if candidate.is_file() else None
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"T7 mandatory evidence hash changed: {candidate}")
        checked[name] = {
            "path": str(candidate),
            "kind": "file" if candidate.is_file() else "directory",
            "sha256": actual_hash,
        }
    return receipt, {"path": str(path), "sha256": sha256(path), "mandatory_evidence": checked}


def _t7_allocations(repo: Path) -> tuple[dict[str, Any], set[int]]:
    """Return the final and rejected T7 allocations as permanent exclusions."""
    final = t7_seed_audit.generate_banks()
    final_seeds = {seed for row in final["banks"].values() for seed in row["seeds"]}
    rejected: set[int] = set()
    for round_index in (74, 75):
        for run_seed in range(5):
            rejected.update(
                dev_screen.screen_seeds(
                    t7_seed_audit.SEED_COUNT,
                    t7_seed_audit.BANK_NS,
                    f"round{round_index}-seed{run_seed}",
                )
            )
    if len(final_seeds) != 960 or len(rejected) != 1920:
        raise ValueError("T7 allocation inventories are incomplete")
    return (
        {
            "final": {"count": len(final_seeds), "seed_digest": seed_digest(final_seeds)},
            "rejected": {"count": len(rejected), "seed_digest": seed_digest(rejected)},
        },
        final_seeds | rejected,
    )


def audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Audit all T12 qualification identities without launching gameplay.

    The new T12 research directory and its artifact output root are excluded so
    that this audit can be refreshed after qualification evidence is recorded.
    Those explicit exclusions are retained in the receipt.  Existing outputs
    are refused by ``main``; callers can use this function for a read-only
    preflight without passing ``out``.
    """
    source = Path(__file__).resolve()
    source_hash = sha256(source)
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
        raise FileNotFoundError("repository research and artifact roots are required")

    generated = generate_banks()
    candidate = {seed for row in generated["banks"].values() for seed in row["seeds"]}
    receipt, receipt_binding = _t7_receipt(repo_path)
    allocations, allocation_seeds = _t7_allocations(repo_path)
    registered = t7_seed_audit._registered(repo_path)
    registered.update(allocation_seeds)

    # T7's audit verifies the historical FRP exclusions.  Re-read them here so
    # this receipt remains explicitly bound to their current contents.
    prior_exclusions = sorted(artifact_root.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(prior_exclusions) < 7:
        raise FileNotFoundError("historical FRP training exclusion inventories incomplete")
    exclusion_binding: list[dict[str, Any]] = []
    for path in prior_exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or not all(type(value) is int for value in values):
            raise ValueError(f"invalid historical FRP training exclusion inventory: {path}")
        registered.update(values)
        exclusion_binding.append({"path": str(path), "sha256": sha256(path), "count": len(values)})

    documentary_exclusions = (
        repo_path / "research/t12_b8_serving_20261017",
        artifact_root / ARTIFACT_OUTPUT_DIRNAME,
    )
    # ``_partitioned_rg`` accepts directories and skips only these exact
    # self-owned roots; all historical evidence remains in scope.
    excluded = documentary_exclusions + ((output,) if output is not None else ())
    scan_args = [
        "--hidden",
        "--no-ignore",
        "--fixed-strings",
        "--word-regexp",
        "--glob",
        "*.json",
        "--glob",
        "*.jsonl",
        "--only-matching",
    ]
    for seed in sorted(candidate):
        scan_args.extend(["-e", str(seed)])
    roots = (repo_path / "research", artifact_root)
    scans = {
        str(root): t4e_seed_audit._partitioned_rg(
            root, scan_args, excluded, "t12-visible-json-jsonl-seed-identities"
        )
        for root in roots
    }
    observed = {
        int(token) for hits, _progress in scans.values() for token in hits if token.isdigit()
    }
    collisions = sorted(candidate & registered)
    bank_checks = {
        name: {
            "count": row["count"],
            "seed_digest": row["seed_digest"],
            "registered_collisions": sorted(set(row["seeds"]) & registered),
            "observed_json_collisions": sorted(set(row["seeds"]) & observed),
        }
        for name, row in generated["banks"].items()
    }
    passed = not collisions and not observed and all(
        not row["registered_collisions"] and not row["observed_json_collisions"]
        for row in bank_checks.values()
    )
    if sha256(source) != source_hash:
        raise RuntimeError("seed-audit source changed while the audit was running")
    return {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source_sha256": source_hash,
        "namespace": generated["namespace"],
        "recipe": generated["recipe"],
        "candidate_count": generated["count"],
        "candidate_seed_digest": generated["seed_digest"],
        "banks": generated["banks"],
        "bank_checks": bank_checks,
        "registered_count": len(registered),
        "registered_collision_seeds": collisions,
        "observed_json_jsonl_collision_seeds": sorted(observed),
        "t7_receipt": receipt_binding,
        "t7_allocations": allocations,
        "t7_receipt_generated_utc": receipt.get("generated_utc"),
        "historical_frp_training_exclusions": exclusion_binding,
        "text_scan": {
            "method": "t7-partitioned-rg-hidden-no-ignore/threads1/timeout60",
            "roots": [str(root) for root in roots],
            "candidate_count": len(candidate),
            "hits": {name: sorted(hits) for name, (hits, _progress) in scans.items()},
            "progress": {name: progress for name, (_hits, progress) in scans.items()},
            "documentary_exclusions": [str(path) for path in documentary_exclusions],
            "exact_output_exclusion": str(output) if output is not None else None,
        },
        "limitations": [
            "No claim about unrecorded experiments, opaque binary files, or external systems.",
            "Visible binary banks are covered through the T7 mandatory manifests and receipts.",
        ],
    }


def main() -> int:
    """Create exactly one new audit receipt and return nonzero on a collision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {args.out}")
    receipt = audit(artifacts=args.artifacts, out=args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": receipt["pass"], "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
