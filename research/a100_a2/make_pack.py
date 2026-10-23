"""Prepare A2 dispatch manifests offline; this module never allocates Colab."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from research.a100_a2.protocol import (
    B8_SHA,
    NAMESPACE,
    allocation_hash,
    empty_ledger,
)
from research.bl_ppo.exports import digest, export_json

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
CHECKPOINT = ROOT.parent / "snake-dqn/saved_snakes/b8_s3_t7_20261016.pth"
QUALIFIED_TREE = "553af3f692515b2bf66173356fe4d099acc96942"
SOURCE_PATHS = [
    "src",
    "research/a100_a2",
    "research/colab_cli",
    "research/bl_ppo/exports.py",
    "research/t34_exit_b8",
    "research/apex_safety_20260926",
    "research/redesign_m3c_20261008",
    "research/t35_outcome_head",
    "research/t5_separability_20261016",
    "research/t5b_risk_and_fuse_20261016",
    "research/phase0_search_20261019",
    "research/t3_rollout_labels_20261015",
    "research/b8_death_census_20261014",
    "research/redesign_scope_20261007",
    "research/t1_commitment_20261015",
    "research/redesign_m2_20261008",
    "research/bl_k1",
    "research/fixed_search.py",
    "configs",
    "pyproject.toml",
]


def prepare_inputs(output: Path, existing_bundle: Path | None = None) -> dict[str, Any]:
    """Bundle only authority-allowlisted existing inputs; no fitting or Mac replay."""
    from research.a100_a2.data import build_bundle

    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "labels.tar.gz"
    if existing_bundle is None:
        bundle_receipt = build_bundle(bundle)
    else:
        shutil.copyfile(existing_bundle, bundle)
        bundle_receipt = {
            "archive_sha256": digest(bundle),
            "archive_bytes": bundle.stat().st_size,
            "source": "reused hash-bound A2 bundle; no replay",
        }
    checkpoint = output / "b8.pth"
    shutil.copyfile(CHECKPOINT, checkpoint)
    if digest(checkpoint) != B8_SHA:
        raise ValueError("checkpoint identity differs")
    bindings = {
        "labels": digest(bundle),
        "checkpoint": B8_SHA,
        "preregistration": digest(ROOT / "research/a100_a2/PREREGISTRATION.md"),
        "semantics_erratum": digest(ROOT / "research/a100_a2/SEMANTICS_ERRATUM.md"),
    }
    alloc = allocation_hash(bindings)
    export_json(output / "verified-ledger.json", empty_ledger(NAMESPACE, alloc))
    canary_alloc = hashlib.sha256((alloc + "-canary").encode()).hexdigest()
    export_json(output / "canary-ledger.json", empty_ledger(NAMESPACE + "-canary", canary_alloc))
    input_hashes = {
        name: digest(output / name)
        for name in ("labels.tar.gz", "b8.pth", "verified-ledger.json", "canary-ledger.json")
    }
    control = {
        "schema": "a2-control/v1",
        "bindings": bindings,
        "allocation": alloc,
        "input_hashes": input_hashes,
        "bundle_receipt": bundle_receipt,
    }
    export_json(output / "control.json", control)
    return control


def make_pack(inputs: Path, output: Path, commit: str, rates: Path | None = None) -> dict:
    """Pin published source and qualified tree; keep main refused until live canary."""
    from tools.ops.qualification import check

    registry = ARTIFACTS / "ops/colab-cli-qualified.json"
    if check(ROOT, registry, commit) != QUALIFIED_TREE:
        raise ValueError("unexpected qualified transport tree")
    # No mutable science implementation can be hidden behind a published pin.
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "research/a100_a2"],
        cwd=ROOT,
        text=True,
    )
    if dirty:
        raise ValueError("commit A2 implementation before preparing pinned manifests")
    remote = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True
    ).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, remote], cwd=ROOT, check=True)
    output.mkdir(parents=True, exist_ok=False)
    control = json.loads((inputs / "control.json").read_text())
    for name, sha in control["input_hashes"].items():
        if digest(inputs / name) != sha:
            raise ValueError("prepared input changed")
    names = [
        "labels.tar.gz",
        "b8.pth",
        "control.json",
        "verified-ledger.json",
        "canary-ledger.json",
    ]
    inventory = [
        {"path": str((inputs / n).resolve()), "dest": n, "sha256": digest(inputs / n)}
        for n in names
    ]
    other = json.loads(rates.read_text()) if rates else {}
    main = {
        "job_id": f"a100-a2-main-{commit[:8]}",
        "commit": commit,
        "script": "research/a100_a2/colab.sh",
        "source_paths": SOURCE_PATHS,
        "device": "cuda",
        "scientific_admission": True,
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 1,
            "est_minutes": 120,
        },
        "timeout_minutes": 258,
        "output_dir": str(output / "main-output"),
        "inputs": inventory,
        "required_exports": ["receipt.json"],
        "env": {"A2_DEVICE": "cuda", "COLAB_CLI_REQUIREMENTS": "research/a100_a2/requirements.txt"},
        "budget": {
            "unit_cap": 45,
            "rate_units_per_hour": 10,
            "reserve_units": 8,
            "other_assignments": other,
        },
        "resume": {
            "unit_kind": "final-fold-fit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "max_export_bytes": 2 * 1024**3,
        "a2": {
            "allocation": control["allocation"],
            "qualified_transport_tree": QUALIFIED_TREE,
            "base_hours": 2,
            "base_units": 20,
            "effective_series_hours": 4.8,
            "effective_series_units": 48,
            "reserve_base": 4,
            "reserve_effective": 8,
            "forecast_hours": 2,
            "forecast_units_at_5_4_per_hour": 10.8,
            "canary_in_series": True,
            "review_required": "independent Codex-V",
        },
    }
    probe = copy.deepcopy(main)
    probe.update(
        job_id=f"a100-a2-canary-{commit[:8]}",
        output_dir=str(output / "canary-output"),
        timeout_minutes=18,
    )
    probe["requirements"]["est_minutes"] = 9
    probe["budget"].update(unit_cap=3, reserve_units=0.6)
    probe["resume"]["ledger_path"] = "inputs/canary-ledger.json"
    probe["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/a100_a2")
    probe["canary"] = {
        "schema": "colab-family-canary/v1",
        "mode": "probe",
        "family": "research/a100_a2/colab.sh",
    }
    export_json(output / "canary.json", probe)
    export_json(output / "main-template.json", main)
    instructions = (
        "A2: reviewed preparation; Claude dispatches only.\n"
        "Run same-pin A100-40 canary first; verify PASS and stop_verified.\n"
        "Bind the genuine canary receipt with make_pack --admit, then dispatch main.json.\n"
        "No metrics may be opened before the complete Mac-verified 480-fit union.\n"
    )
    (output / "DISPATCH.txt").write_text(instructions)
    receipt = {
        "schema": "a2-prepared-pack/v1",
        "commit": commit,
        "transport_tree": QUALIFIED_TREE,
        "allocation": control["allocation"],
        "inputs": inventory,
        "canary": str(output / "canary.json"),
        "main_template": str(output / "main-template.json"),
        "status": "PREPARED_REQUIRES_REAL_A100_CANARY",
    }
    export_json(output / "pack.json", receipt)
    return receipt


def admit(template: Path, canary_receipt: Path, output: Path) -> dict:
    """Never synthesize canary evidence; delegate complete live proof to supervisor."""
    from research.colab_cli.supervisor import validate

    main = json.loads(template.read_text())
    main["canary"] = {
        "receipt": {"path": str(canary_receipt.resolve()), "sha256": digest(canary_receipt)}
    }
    validated = validate(main)
    export_json(output, validated)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("inputs")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--bundle", type=Path)
    pack = sub.add_parser("pack")
    pack.add_argument("--inputs", type=Path, required=True)
    pack.add_argument("--out", type=Path, required=True)
    pack.add_argument("--commit", required=True)
    pack.add_argument("--other-assignments", type=Path)
    admission = sub.add_parser("admit")
    admission.add_argument("--template", type=Path, required=True)
    admission.add_argument("--canary-receipt", type=Path, required=True)
    admission.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "inputs":
        result = prepare_inputs(args.out, args.bundle)
    elif args.action == "pack":
        result = make_pack(args.inputs, args.out, args.commit, args.other_assignments)
    else:
        result = admit(args.template, args.canary_receipt, args.out)
    print(
        json.dumps(
            {
                "action": args.action,
                "out": str(args.out),
                "allocation": result.get("allocation", result.get("a2", {}).get("allocation")),
            }
        )
    )


if __name__ == "__main__":
    main()
