"""Fail-closed freshness audit for the T7 Phase R world banks.

This module only allocates and audits identities; it never starts gameplay.  The
T7 banks use a namespace separate from the historical distillation rounds so
each candidate seed has an independent 192/96/96 roster.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.apex_safety_20260926 import dev_screen
from research.t3_screen_20261015 import banks as prior_banks
from research.t4e_return_20261016 import seed_audit as t4e_audit

BANK_NS = "redesign-t7-phase-r-b8/v1"
BANK_KEY = "round76-seed{s}"
SEED_COUNT = 192
ROUNDS = tuple(range(50, 74))
SCHEMA = "t7-phase-r-seed-audit/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_digest(seeds: set[int] | list[int]) -> str:
    return hashlib.sha256(",".join(str(x) for x in sorted(set(seeds))).encode()).hexdigest()


def generate_banks() -> dict[str, Any]:
    """Return the fixed five candidate rosters, without touching the filesystem."""
    banks = {}
    all_seeds: set[int] = set()
    for index in range(5):
        key = BANK_KEY.format(s=index)
        seeds = dev_screen.screen_seeds(SEED_COUNT, BANK_NS, key)
        banks[f"s{index}"] = {
            "key": key,
            "count": len(seeds),
            "seeds": seeds,
            "seed_digest": seed_digest(seeds),
        }
        all_seeds.update(seeds)
    if len(all_seeds) != 5 * SEED_COUNT:
        raise RuntimeError("T7 candidate banks collide")
    return {
        "namespace": BANK_NS,
        "bank_key": BANK_KEY,
        "banks": banks,
        "count": len(all_seeds),
        "seed_digest": seed_digest(all_seeds),
    }


def _registered(repo: Path) -> set[int]:
    """Collect registered NI, historical, and all generated round identities."""
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    registered = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds())
    t2 = prior_banks._registered_t2(repo)
    if not t2["pass"] or not t2["historical_complete"]:
        raise ValueError("historical distillation registration is incomplete")
    for values in t2["registered_rounds"].values():
        registered.update(values)
    for item in t2["historical_distillation"].values():
        registered.update(item["seeds"])
    for round_index in ROUNDS:
        registered.update(ni_spec.distill_seeds(1024, round_index=round_index))
    for round_index in (74, 75):
        for s in range(5):
            registered.update(dev_screen.screen_seeds(192, BANK_NS, f"round{round_index}-seed{s}"))
    return registered


def audit(repo: str | Path = ".", artifacts: str | Path | None = None) -> dict[str, Any]:
    """Audit candidate identities against registered history and visible evidence."""
    repo = Path(repo).resolve()
    artifacts = Path(artifacts).resolve() if artifacts else repo.parent / "snake-dqn-artifacts"
    generated = generate_banks()
    if not artifacts.is_dir() or not (repo / "research").is_dir():
        raise FileNotFoundError("repository research and artifact roots are required")
    mandatory = {
        "t3_namespace": artifacts / "t3-rollout-labels-20261015/bank/namespace.json",
        "t3_manifest": artifacts / "t3-rollout-labels-20261015/bank/manifest.json",
        "t4f_manifest_train": artifacts / "t4f-return-bank-v3-20261016/banks/train/manifest.json",
        "t4f_manifest_holdout": artifacts
        / "t4f-return-bank-v3-20261016/banks/holdout/manifest.json",
        "m4b_root": artifacts / "redesign-m4b-mac-20261013",
        "m3c_root": artifacts / "redesign-m3c-gpu-20261008",
        "census_root": artifacts / "b8-death-census-20261014",
        "m3_phase_r": artifacts / "redesign-m3-phase-r-20261008/plan.json",
        "m4_phase_r": artifacts / "redesign-m4-phase-r-nv-20261013/plan.json",
        "t4e_train": artifacts / "t4e-return-bank-20261016/banks/train/manifest.json",
        "t4e_holdout": artifacts / "t4e-return-bank-20261016/banks/holdout/manifest.json",
        "t4_snapshot": artifacts / "t4b-snapshot-restart-20261016/full/manifest.json",
        "t4c_bank": artifacts / "t4c-data-20261016/bank-v3/manifest.json",
        "t4c_outside": artifacts / "t4c-data-20261016/outside-bank/manifest.json",
    }
    prior_exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(prior_exclusions) < 7:
        raise FileNotFoundError("historical FRP training exclusion inventories incomplete")
    mandatory.update({f"prior_exclusions_{i}": p for i, p in enumerate(prior_exclusions)})
    missing = [name for name, path in mandatory.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"mandatory freshness evidence missing: {missing}")
    evidence_manifest = {
        name: {
            "path": str(path),
            "kind": "directory" if path.is_dir() else "file",
            "count": len(list(path.rglob("*"))) if path.is_dir() else 1,
            "sha256": sha256(path) if path.is_file() else None,
        }
        for name, path in mandatory.items()
    }
    candidate = set().union(*(set(row["seeds"]) for row in generated["banks"].values()))
    registered = _registered(repo)
    for path in prior_exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or not all(type(v) is int for v in values):
            raise ValueError(f"invalid training exclusions: {path}")
        registered.update(values)
    namespace = json.loads(mandatory["t3_namespace"].read_text())
    t3_binding = prior_banks._manifest_binding(
        mandatory["t3_namespace"].parent, mandatory["t3_namespace"]
    )
    if namespace.get("schema") != "t3-label-namespace/v1" or not t3_binding["ok"]:
        raise ValueError("T3 namespace/manifest identity invalid")
    registered.update(namespace["seeds"])
    for name in ("m3_phase_r", "m4_phase_r"):
        plan = json.loads(mandatory[name].read_text())
        for bank in plan["banks"].values():
            registered.update(bank)
    from research.t4f_return_20261016.seed_audit import _source_bank_audit

    t4e_source = _source_bank_audit(artifacts / "t4e-return-bank-20261016")
    if not t4e_source["pass"]:
        raise ValueError("T4e source namespace/manifest identity invalid")
    collisions = sorted(candidate & registered)
    # Search exact decimal identities in JSON/JSONL receipts; binary arrays are
    # intentionally excluded because their manifests/receipts are the audit boundary.
    tokens = sorted(str(x) for x in candidate)
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
    for token in tokens:
        scan_args.extend(["-e", token])
    excludes = (repo / "research/redesign_t7_phase_r_20261016/SEED_AUDIT_T7.json",)
    hits, progress = t4e_audit._partitioned_rg(
        repo / "research", scan_args, excludes, "t7-seed-identities"
    )
    if artifacts.is_dir():
        extra, extra_progress = t4e_audit._partitioned_rg(
            artifacts, scan_args, excludes, "t7-artifact-seed-identities"
        )
        hits |= extra
        progress.extend(extra_progress)
    evidence_hits = sorted(hits)
    return {
        "schema": SCHEMA,
        "namespace": BANK_NS,
        "banks": generated["banks"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_source_sha256": sha256(Path(__file__)),
        "t3_manifest_binding": t3_binding,
        "t4e_source_audit": t4e_source,
        "registered_count": len(registered),
        "mandatory_evidence": evidence_manifest,
        "registered_rounds": list(ROUNDS),
        "candidate_count": len(candidate),
        "registered_collision_seeds": collisions,
        "evidence_hits": evidence_hits,
        "scan_progress": progress,
        "pass": not collisions and not evidence_hits,
        "limitations": [
            "No claim about unrecorded experiments or opaque binary files.",
            "NPZ/NPY are covered through visible JSON/JSONL manifests and receipts.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
