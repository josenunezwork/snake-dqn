"""Prepare A2 dispatch manifests offline; this module never allocates Colab."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
REQUIREMENTS = "research/a100_a2/requirements.txt"
# Include both remote modes and the lazy replay/strict constructor/numba seams.
IMPORT_ROOTS = [
    "research.a100_a2.job",
    "research.a100_a2.run",
    "research.a100_a2.data",
    "research.t34_exit_b8.replay:_default_observation",
    "research.t34_exit_b8.trajectory:build_world",
    "research.phase0_search_20261019.teacher:build_world",
    "research.redesign_scope_20261007.grid_h5000_identity:_context",
    "research.t3_rollout_labels_20261015.bank_engine:build_world",
    "src.simd_env.ego_raster:build_ego_raster",
    "src.simd_env.grid_sim_nb",
    "src.simd_env.vector61_nb",
    "src.simd_env.v8_nb",
]
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
    # Conservative transitive lazy-import closure; typing-only branches excluded.
    "research/apex_veto_strict_20260927/__init__.py",
    "research/apex_veto_strict_20260927/strict_run.py",
    "research/bl_evidence/__init__.py",
    "research/bl_evidence/loader.py",
    "research/bl_evidence/strict_pins.py",
    "research/bl_paced_identity/__init__.py",
    "research/bl_paced_identity/authorization.py",
    "research/bl_paced_identity/events.py",
    "research/bl_paced_identity/record.py",
    "research/bl_paced_identity/replay.py",
    "research/bl_ppo/__init__.py",
    "research/compute/thermal_guard.py",
    "research/redesign_t7_phase_r_20261016/__init__.py",
    "research/redesign_t7_phase_r_20261016/seed_audit.py",
    "research/t2_screen_20261015/__init__.py",
    "research/t2_screen_20261015/banks.py",
    "research/t33_batched_teacher/__init__.py",
    "research/t33_batched_teacher/inference.py",
    "research/t33_batched_teacher/observation_kernels.py",
    "research/t33_batched_teacher/observations.py",
    "research/t33_batched_teacher/runtime.py",
    "research/t33_batched_teacher/teacher.py",
    "research/t34_pathb/__init__.py",
    "research/t34_pathb/live.py",
    "research/t34_shadow/__init__.py",
    "research/t34_shadow/lad.py",
    "research/t34_shadow/shadow.py",
    "research/t34_speed_s2/__init__.py",
    "research/t34_speed_s2/native_v8.py",
    "research/t3_screen_20261015/__init__.py",
    "research/t3_screen_20261015/banks.py",
    "research/t4e_return_20261016/__init__.py",
    "research/t4e_return_20261016/seed_audit.py",
    "research/t4f_return_20261016/__init__.py",
    "research/t4f_return_20261016/seed_audit.py",
    "web/backend/__init__.py",
    "web/backend/checkpoints.py",
    "web/backend/ego2s_policy.py",
    "web/backend/pathb.py",
    "web/backend/pathb_controller.py",
    "web/backend/raster_policy.py",
    "web/backend/release_monitoring.py",
    "web/backend/safety_veto_serving.py",
    "web/backend/serialize.py",
    "web/backend/served_checkpoint.py",
    "web/backend/serving_speed.py",
    "web/backend/session.py",
    "web/backend/state_labels.py",
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


def replacement_envelope(receipt: Path | None) -> dict:
    """Charge the failed, stopped, zero-export canary to the original series cap."""
    if receipt is None:
        return {"main_minutes": 258, "main_units": 45}
    prior = json.loads(receipt.read_text())
    if (
        prior.get("status") != "INCOMPLETE"
        or prior.get("stop_verified") is not True
        or prior.get("failure_class") != "CHILD_NONZERO_EXIT"
        or prior.get("exports") != []
        or not prior.get("job_id", "").startswith("a100-a2-canary-")
        or "ModuleNotFoundError" not in prior.get("error", "")
    ):
        raise ValueError("replacement requires stopped mechanical zero-export canary evidence")
    elapsed = float(prior["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid prior canary elapsed time")
    # Conservative ceiling rate, including setup/export/cleanup; no credit reset.
    charged_units = elapsed * 10 / 3600
    minutes = min(258, math.floor(288 - 18 - elapsed / 60))
    units = min(45, math.floor((48 - 3 - charged_units) * 100) / 100)
    if minutes < 240 or units <= 8:
        raise ValueError("remaining original envelope cannot admit the unchanged forecast")
    return {
        "main_minutes": minutes,
        "main_units": units,
        "prior_canary_receipt": str(receipt.resolve()),
        "prior_canary_receipt_sha256": digest(receipt),
        "prior_elapsed_seconds": elapsed,
        "prior_charged_units_at_ceiling": charged_units,
    }


def make_pack(
    inputs: Path,
    output: Path,
    commit: str,
    rates: Path | None = None,
    failed_canary_receipt: Path | None = None,
) -> dict:
    """Pin published source and qualified tree; keep main refused until live canary."""
    from tools.ops.import_preflight import check as check_imports
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
    envelope = replacement_envelope(failed_canary_receipt)
    # Probe the pinned sparse archive, never the developer checkout or venv.
    # This must finish before any pack directory/manifests are published.
    if output.exists():
        raise FileExistsError(output)
    preflight = check_imports(ROOT, commit, SOURCE_PATHS, REQUIREMENTS, IMPORT_ROOTS)
    output.mkdir(parents=True, exist_ok=False)
    export_json(output / "clean-import-preflight.json", preflight)
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
        "timeout_minutes": envelope["main_minutes"],
        "output_dir": str(output / "main-output"),
        "inputs": inventory,
        "required_exports": ["receipt.json"],
        "env": {"A2_DEVICE": "cuda", "COLAB_CLI_REQUIREMENTS": REQUIREMENTS},
        "budget": {
            "unit_cap": envelope["main_units"],
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
            "remaining_envelope": envelope,
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
        "clean_import_preflight_sha256": digest(output / "clean-import-preflight.json"),
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
    pack.add_argument("--failed-canary-receipt", type=Path)
    admission = sub.add_parser("admit")
    admission.add_argument("--template", type=Path, required=True)
    admission.add_argument("--canary-receipt", type=Path, required=True)
    admission.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "inputs":
        result = prepare_inputs(args.out, args.bundle)
    elif args.action == "pack":
        result = make_pack(
            args.inputs, args.out, args.commit, args.other_assignments, args.failed_canary_receipt
        )
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
