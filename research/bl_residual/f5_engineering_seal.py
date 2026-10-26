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
PREDECESSOR_SEAL_PATH = (
    "research/bl_residual/ledger/F5_ENGINEERING_SEAL_20261026/ENGINEERING_SEAL.json"
)
PREDECESSOR_SEAL_SHA256 = "69e63fe86f3683aed27d9d8f15881d80b0feede7dbec63493631dce3fcddbb42"
AMENDMENT = "research/bl_residual/f5_hold_amendment_20261026"
CORRECTION_CONTRACT = "research/bl_residual/f5_hold_correction_20261026/ENGINEERING_CORRECTIONS.md"
CORRECTION_CONTRACT_SHA256 = "4859512108db90fcfa06c14c051128bccc40166a8a490f5c82df8095b41e4c4e"
ASTRA_REJECTION = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/"
    "REJECT_F5_HOLD_ENGINEERING_fc67d6bf.md"
)
ASTRA_REJECTION_SHA256 = "51cff62e7da3d0ec34ea4e37c2293331cbd1800ec6e7e38b20589caa228936d2"
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
ENGINEERING_CHANGES = {
    "research/bl_residual/f5_package.py": "closure implementation; no measured workload change",
    "research/bl_residual/f5_protocol.py": "append-only claim lineage and sole-token refusal",
    "research/bl_residual/f5_plan_successor.py": "new label-blind identity migration",
    "research/bl_residual/f5_timing_supplement.py": (
        "new timing application; qualification required"
    ),
    "research/bl_residual/f5_amended_reduction.py": "prospective reducer; old units retained",
    "research/bl_residual/f5_engineering_seal.py": "new offline closure verification",
    "research/bl_residual/f5_timing_recovery_package.py": (
        "existing operational runaway manifest guard; no measured computation"
    ),
    "research/bl_residual/f5_timing_resume.py": (
        "existing Mac-only bound preallocation refusal/accounting provenance; "
        "no measured computation"
    ),
    "research/bl_residual/f5_step3_driver.py": "new step-3 application; qualification required",
    "research/bl_residual/f5_step3_package.py": "new step-3 package; qualification required",
    "research/bl_residual/f5_step3.sh": "new step-3 shell entrypoint; qualification required",
    "research/bl_residual/f5_calibration.py": (
        "F5 calibration application; targeted qualification required"
    ),
    "research/bl_residual/f5_colab.sh": "F5 runtime entrypoint; targeted qualification required",
    "research/bl_residual/f5_environment.py": (
        "F5 environment application; targeted qualification required"
    ),
    "research/bl_residual/f5_environment_reference.py": (
        "F5 environment reference application; targeted qualification required"
    ),
    "research/bl_residual/f5_environment_schema_v6.json": (
        "native-v6 environment contract; targeted qualification required"
    ),
    "research/bl_residual/f5_environment_v5.py": (
        "F5 environment comparison application; targeted qualification required"
    ),
    "research/bl_residual/f5_job.py": "F5 job application; targeted qualification required",
    "research/bl_residual/f5_progress.py": "F5 resume application; targeted qualification required",
    "research/bl_residual/f5_qualification.py": (
        "F5 CUDA/parity application; targeted qualification required"
    ),
    "research/bl_residual/f5_raw_envelope.py": (
        "F5 envelope application; targeted qualification required"
    ),
    "research/bl_residual/f5_timing.py": "F5 timing application; targeted qualification required",
    "research/colab_cli/acceptance.py": (
        "qualified transport dependency; application review required"
    ),
    "research/colab_cli/admission.py": (
        "qualified transport dependency; application review required"
    ),
    "research/colab_cli/canary.py": ("qualified transport dependency; application review required"),
    "research/colab_cli/canary_observer.py": (
        "qualified CUDA-observation dependency; application review required"
    ),
    "research/colab_cli/supervisor.py": (
        "qualified transport dependency; application review required"
    ),
    "research/colab_cli/transfer_policy.py": (
        "qualified transport dependency; application review required"
    ),
    "research/bl_residual/f5_adoption_environment_20261026_v6/requirements-f5-l4-cu126.lock": (
        "step-3 shell requirements lock; targeted qualification required"
    ),
    "research/bl_residual/f5_adoption_environment_20261026_v6/f5-common-environment-v6.json": (
        "step-3 shell common environment; targeted qualification required"
    ),
}


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


def external_reference(path: Path) -> dict[str, Any]:
    """Bind a reviewed external ruling without treating it as repository source."""
    if (
        not path.is_absolute()
        or not path.is_file()
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise ValueError("unsafe external closure file: " + str(path))
    raw = path.read_bytes()
    return {"path": str(path), "sha256": sha(raw), "size": len(raw)}


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
    """Compare every closure member to both historic sources before classifying deltas."""
    names_by_revision = {
        revision: set(git(repo, "ls-tree", "-r", "--name-only", revision).decode().splitlines())
        for revision in (OLD_PLAN_COMMIT, TIMING_COMMIT)
    }
    comparisons: dict[str, dict[str, str | None]] = {}
    changes = {}
    for relative, current in pins.items():
        historical = {
            revision: (
                sha(git(repo, "show", f"{revision}:{relative}"))
                if relative in names_by_revision[revision]
                else None
            )
            for revision in (OLD_PLAN_COMMIT, TIMING_COMMIT)
        }
        comparisons[relative] = historical
        # A closure member that differs from either historic source cannot be
        # silently treated as inherited measurement code merely because it
        # happens to match the other historic source.
        if any(current != value for value in historical.values()):
            changes[relative] = {**historical, "current": current}
    unreviewed = set(changes) - set(ENGINEERING_CHANGES)
    if unreviewed:
        raise ValueError("unreviewed measured source drift: " + repr(sorted(unreviewed)))
    for relative in UNCHANGED_SCIENCE:
        current = reference(repo, relative)["sha256"]
        for revision in (OLD_PLAN_COMMIT, TIMING_COMMIT):
            if sha(git(repo, "show", f"{revision}:{relative}")) != current:
                raise ValueError("scientific recipe/loss/engine drift: " + relative)
    return {
        "timing_commit": TIMING_COMMIT,
        "historical_source_comparison": comparisons,
        "changes": {
            name: {**row, "classification": ENGINEERING_CHANGES[name]}
            for name, row in changes.items()
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
    predecessor_seal = reference(repo, PREDECESSOR_SEAL_PATH)
    if predecessor_seal["sha256"] != PREDECESSOR_SEAL_SHA256:
        raise ValueError("predecessor engineering seal digest changed")
    correction_contract = reference(repo, CORRECTION_CONTRACT)
    if correction_contract["sha256"] != CORRECTION_CONTRACT_SHA256:
        raise ValueError("engineering correction contract digest changed")
    astra_rejection = external_reference(ASTRA_REJECTION)
    if astra_rejection["sha256"] != ASTRA_REJECTION_SHA256:
        raise ValueError("Astra rejection digest changed")
    interim = reference(
        repo, "research/bl_residual/ledger/F5_ENGINEERING_SEAL_20261026_V2/ENGINEERING_SEAL.json"
    )
    if interim["sha256"] != "0ba82da4182b5cea69ecd8d950ef059eb41008fbe8a7b40616e549290556d9b2":
        raise ValueError("interim engineering predecessor digest changed")
    v3 = reference(
        repo, "research/bl_residual/ledger/F5_ENGINEERING_SEAL_20261026_V3/ENGINEERING_SEAL.json"
    )
    if v3["sha256"] != "76d707a9d52473a50f7f2b9b1cfa7644eddb1e2344f051d9f4bc47e703f5ff72":
        raise ValueError("V3 engineering predecessor digest changed")
    v4 = reference(
        repo, "research/bl_residual/ledger/F5_ENGINEERING_SEAL_20261026_V4/ENGINEERING_SEAL.json"
    )
    if v4["sha256"] != "e73b2d24144cd3e375aeffca1706dd5c677c3b23da95dce49db30d15ba3f9038":
        raise ValueError("V4 engineering predecessor digest changed")
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
        "predecessor_engineering_seal": predecessor_seal,
        "interim_engineering_predecessor": interim,
        "v3_engineering_predecessor": v3,
        "v4_engineering_predecessor": v4,
        "v5_engineering_correction_contract": reference(
            repo, "research/bl_residual/f5_hold_correction_20261026/ENGINEERING_V5.md"
        ),
        "attempt02_evidence": reference(
            repo, "research/bl_residual/ledger/F5_STEP1_ATTEMPT02_20261026/PUBLICATION.json"
        ),
        "v4_engineering_correction_contract": reference(
            repo, "research/bl_residual/f5_hold_correction_20261026/ENGINEERING_V4.md"
        ),
        "v3_reviews": [
            external_reference(
                Path(
                    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
                    "astra-20261024/REJECT_F5_HOLD_ENGINEERING_V3.md"
                )
            ),
            external_reference(
                Path(
                    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
                    "argon-20261026/review-f5-hold-v3/answer.md"
                )
            ),
        ],
        "export_envelope_successor": reference(
            repo, "research/bl_residual/f5_hold_correction_20261026/EXPORT_ENVELOPE_V3.md"
        ),
        "engineering_correction_contract": correction_contract,
        "astra_rejection": astra_rejection,
        "actual_successor_plan": "NOT_CREATED_STEP4_REQUIRES_INDEPENDENT_REVIEW",
        "actual_claim_lineage": "NOT_ISSUED_STEP4_REQUIRES_UNCONSUMED_PREDECESSOR_PROOF",
        "supplement_package": "SEALED_PACKAGE_REQUIRES_SEPARATE_ACCEPTANCE",
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
