"""Offline create-only engineering closure for Astra's prospective F5 amendment.

This seals source and metadata, never a timing package, plan, claim or acceptance.
No checkpoint, selected row, scientific outcome or provider API is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_json, verify_export
from research.bl_residual.f5_package import ROOT, source_inventory, source_review

TIMING_COMMIT = "911426dbace94ca2925d737d00d5d8ca8456cb6a"
EVIDENCE_COMMIT = "93743a9acaca23c10f06acd1e10c6424db34dbec"
REDUCTION_COMMIT = "d41e2ff7ab0444441c733751138502c17b66fb7a"
OLD_PLAN_COMMIT = "75b14427"
AMENDMENT = "research/bl_residual/f5_hold_amendment_20261026"
ENVIRONMENT = "research/bl_residual/f5_adoption_environment_20261026_v6"
IMMUTABLE_TREES = (
    "research/bl_residual/ledger/F5_LONGRUN01_OUTPUT02_20261026",
    "research/bl_residual/ledger/F5_POST_LONGRUN01_VERIFICATION_20261026",
    "research/bl_residual/ledger/F5_W_PLAN_SEAL_VERIFICATION_20261026",
)
UNCHANGED_SCIENCE = (
    "research/bl_residual/f5_fit.py",
    "research/bl_residual/f5_data.py",
    "research/bl_residual/fit.py",
    "research/bl_residual/model.py",
    "research/bl_residual/gameplay.py",
    "research/t3_rollout_labels_20261015/bank_engine.py",
)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reference(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unsafe closure file: " + relative)
    raw = path.read_bytes()
    return {"path": relative, "sha256": sha(raw), "size": len(raw)}


def immutable_inventory(repo: Path) -> dict[str, str]:
    """Compare old tracked evidence byte-for-byte to its published Git objects."""
    inventory = {}
    for tree in IMMUTABLE_TREES:
        revision = EVIDENCE_COMMIT if "OUTPUT02" in tree else REDUCTION_COMMIT
        names = git(repo, "ls-tree", "-r", "--name-only", revision, "--", tree)
        if not names.strip():
            raise ValueError("missing historical evidence tree: " + tree)
        for relative in names.decode().splitlines():
            expected = sha(git(repo, "show", f"{revision}:{relative}"))
            if reference(repo, relative)["sha256"] != expected:
                raise ValueError("immutable historical evidence changed: " + relative)
            inventory[relative] = expected
    return inventory


def source_bridge(repo: Path, pins: dict[str, str]) -> dict[str, Any]:
    """Classify the full inventory delta; measured old computation must be unchanged."""
    previous_names = set(
        git(repo, "ls-tree", "-r", "--name-only", TIMING_COMMIT).decode().splitlines()
    )
    changes = {}
    for relative, current in pins.items():
        old = (
            sha(git(repo, "show", f"{TIMING_COMMIT}:{relative}"))
            if relative in previous_names
            else None
        )
        if current != old:
            changes[relative] = {"before": old, "after": current}
    allowed = {
        "research/bl_residual/f5_package.py": "inventory-only: adds four engineering entrypoints",
        "research/bl_residual/f5_protocol.py": "append-only claim lineage and sole-token refusal",
        "research/bl_residual/f5_plan_successor.py": "new label-blind identity migration",
        "research/bl_residual/f5_timing_supplement.py": "new timing; qualification required",
        "research/bl_residual/f5_amended_reduction.py": "prospective reducer; old units retained",
        "research/bl_residual/f5_engineering_seal.py": "new offline closure verification",
        "research/bl_residual/f5_timing_recovery_package.py": (
            "existing operational runaway manifest guard from 2b3cb441; no measured computation"
        ),
        "research/bl_residual/f5_timing_resume.py": (
            "existing Mac-only bound preallocation refusal alias/accounting provenance from "
            "2b3cb441; no measured computation or original unit rebinding"
        ),
    }
    if set(changes) - set(allowed):
        raise ValueError(
            "unreviewed measured source drift: " + repr(sorted(set(changes) - set(allowed)))
        )
    for relative in UNCHANGED_SCIENCE:
        current = reference(repo, relative)["sha256"]
        for revision in (OLD_PLAN_COMMIT, TIMING_COMMIT):
            if sha(git(repo, "show", f"{revision}:{relative}")) != current:
                raise ValueError("scientific recipe/loss/engine drift: " + relative)
    return {
        "timing_commit": TIMING_COMMIT,
        "changes": {
            name: {**row, "classification": allowed[name]} for name, row in changes.items()
        },
        "unchanged_scientific_files": {name: reference(repo, name) for name in UNCHANGED_SCIENCE},
        "old_measurement_reuse": "Astra must review this bridge; original source bindings retained",
        "new_supplement_application_qualification": "REQUIRED_BEFORE_STEP3",
        "transport": "CLI unchanged; no ritual transport requalification claimed or performed",
    }


def build(repo: Path = ROOT) -> dict[str, Any]:
    pins = source_inventory(repo)
    required = ("f5_plan_successor.py", "f5_timing_supplement.py", "f5_amended_reduction.py")
    if any("research/bl_residual/" + name not in pins for name in required):
        raise ValueError("incomplete engineering source inventory")
    amendments = {
        str(path.relative_to(repo)): reference(repo, str(path.relative_to(repo)))
        for path in sorted((repo / AMENDMENT).glob("*"))
        if path.is_file() and path.suffix in {".md", ".json"}
    }
    return {
        "schema": "bl-f5-engineering-source-closure/v1",
        "status": "SEALED_ENGINEERING_ONLY_HOLD",
        "source_inventory": pins,
        "source_sha256": sha((json.dumps(pins, sort_keys=True, allow_nan=False) + "\n").encode()),
        "source_bridge": source_bridge(repo, pins),
        "f4_symbol_review": source_review(repo),
        "environment": reference(repo, ENVIRONMENT + "/f5-common-environment-v6.json"),
        "schema_v6": reference(repo, "research/bl_residual/f5_environment_schema_v6.json"),
        "loss": reference(repo, "research/bl_residual/f5_fit.py"),
        "immutable_old_inventory": immutable_inventory(repo),
        "amendments": amendments,
        "actual_successor_plan": "NOT_CREATED_STEP4_REQUIRES_INDEPENDENT_REVIEW",
        "actual_claim_lineage": "NOT_ISSUED_STEP4_REQUIRES_UNCONSUMED_PREDECESSOR_PROOF",
        "supplement_package": "DRAFT_UNSEALED_NOT_ACCEPTED",
        "science_dispatch_authorized": False,
        "look_authorized": False,
    }


def verify(path: Path, repo: Path = ROOT) -> dict[str, Any]:
    verify_export(path)
    value = json.loads(path.read_text())
    if value != build(repo):
        raise ValueError("engineering seal differs from current source/amendment/evidence bytes")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("seal", "verify"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.operation == "seal":
        export_json(args.path, build())
    else:
        verify(args.path)
    print("PASS_ENGINEERING_ONLY_HOLD " + args.operation)


if __name__ == "__main__":
    main()
