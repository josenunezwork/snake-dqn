"""Create-only offline step-3 transport package. Never allocate or dispatch.

Sealing fixes bytes for review. It cannot manufacture independent acceptance,
an L4 assignment, current-driver CUDA evidence, or unknown long-body timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export
from research.bl_residual import f5_environment, f5_package, f5_timing_supplement
from research.bl_residual.f5_environment_reference import capture05_reference
from research.bl_residual.f5_timing_recovery_package import THREADS

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "f5-step3-long-body-20261026-v3"
ALLOCATION = "1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7"
PRIOR = (
    "research/bl_residual/ledger/F5_W_TIMING_OPERATIONAL_20261026_ROUND08_02_REQUAL/manifest.json"
)
AMENDMENT = "research/bl_residual/f5_hold_amendment_20261026"
ENGINEERING_SEAL = (
    "research/bl_residual/ledger/F5_ENGINEERING_SEAL_20261026_V3/ENGINEERING_SEAL.json"
)
MAX_EXPORT_BYTES = 64 * 1024 * 1024
REJECTION = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/"
    "REJECT_F5_HOLD_ENGINEERING_fc67d6bf.md"
)


def canonical(value: Any) -> bytes:
    """Match the source inventory serialization, including its terminal newline."""
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def reference(path: Path, dest: str | None = None) -> dict[str, Any]:
    """Hash input bytes without deserializing any checkpoint."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("unsafe or absent input: " + str(path))
    value = {"path": str(path.resolve()), "sha256": digest(path), "size_bytes": path.stat().st_size}
    if dest is not None:
        value["dest"] = dest
    return value


def envelope() -> dict[str, Any]:
    """Freeze a finite prospective guard; unknown costs remain explicit HOLD."""
    return {
        "base_timeout_minutes": 60,
        "effective_timeout_minutes": 120,
        "double_applied": 1,
        "qualification_base_timeout_minutes": 15,
        "qualification_effective_timeout_minutes": 30,
        "global_completion_rounds_maximum": 2,
        "maximum_attempts": 3,
        "aggregate_effective_timeout_minutes": 360,
        "aggregate_effective_with_one_qualification_minutes": 390,
        "workers": 6,
        "host_workers": 6,
        "threads_per_worker": 1,
        "one_f5_l4_at_a_time": True,
        "forecast_status": "HOLD_CURRENT_DRIVER_QUALIFICATION_AND_UNKNOWN_COSTS",
        "historical_scan_max_seconds": 85.637473645,
        "historical_action_max_seconds": 21.37695354000016,
        "scan_forecast_seconds": 12 * 85.637473645 / 6,
        "branch_diagnostic_seconds": 12 * 7 * 21.37695354000016 / 6,
        "diagnostic_only": True,
        "unknown_costs": [
            "long-body actual bundle and individual bounds",
            "current-driver bootstrap and genuine CUDA/native-v6 qualification",
            "cache/plan preparation",
            "export/I-O and transfers",
            "imbalance and idle",
            "cleanup",
        ],
        "admission": "independent reviewed complete measured forecast must fit base before measure",
        "archive_stage_base_hours": 96,
        "archive_stage_effective_hours": 192,
        "archive_shards": 16,
        "archive_shard_base_hours": 6,
        "archive_shard_effective_hours": 12,
        "archive_new_reservation_seconds_per_shard": 1500,
        "archive_carried_seconds_per_shard": 2239.787154037055 / 16,
        "historical_timing_seconds": 10902.09342033323,
        "historical_canary_seconds": 3849.601884124102,
        "accounting": (
            "append-only all attempts/refusals/setup/qualification/idle/transfers/export/cleanup; "
            "no reset or overlap double charge"
        ),
    }


def build(output: Path, *, repo: Path = ROOT) -> dict[str, Any]:
    """Seal exact runnable inputs without claiming missing admission evidence."""
    if output.exists() or not output.is_absolute():
        raise ValueError("package directory must be absolute and fresh")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    pins = f5_package.source_inventory(repo)
    for relative, expected in pins.items():
        raw = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("source must be committed before packaging: " + relative)
    source_sha = hashlib.sha256(canonical(pins)).hexdigest()
    verify_export(repo / ENGINEERING_SEAL)
    engineering = json.loads((repo / ENGINEERING_SEAL).read_text())
    if engineering["source_inventory"] != pins or engineering["source_sha256"] != source_sha:
        raise ValueError("corrected engineering seal differs from the final package source")
    prior = json.loads((repo / PRIOR).read_text())
    output.mkdir()
    staging = output / "inputs"
    staging.mkdir()
    inputs = []
    for item in prior["inputs"]:
        if item["dest"] in {"control.json", "verified-ledger.json"}:
            continue
        source = (
            Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")
            if item["dest"] == "b8.pth"
            else Path(item["path"])
        )
        row = reference(source, item["dest"])
        if row["sha256"] != item["sha256"]:
            raise ValueError("frozen input drift: " + item["dest"])
        target = staging / item["dest"]
        export_bytes(target, source.read_bytes())
        inputs.append(reference(target, item["dest"]))
    fixture = next(row for row in inputs if row["dest"] == "timing-fixtures.json")
    fixture_value = json.loads(Path(fixture["path"]).read_text())
    f5_timing_supplement.frozen_fixtures(fixture_value["fixtures"])
    identity = {
        "namespace": NAMESPACE,
        "allocation": ALLOCATION,
        "source_sha256": source_sha,
        "fixtures_sha256": fixture["sha256"],
    }
    ledger = staging / "verified-ledger.json"
    export_json(
        ledger,
        {
            "schema": "bl-f5-step3-component-ledger/v1",
            "component_id": "f5-step3-long-body-v1",
            "identity": identity,
            "units": [],
        },
    )
    inputs.append(reference(ledger, ledger.name))
    control = {
        "schema": "bl-f5-step3-control/v1",
        "namespace": NAMESPACE,
        "allocation": ALLOCATION,
        "source_hashes": pins,
        "source_sha256": source_sha,
        "fixture_descriptor_sha256": fixture["sha256"],
        "source_provenance": {
            "source_sha256": source_sha,
            "environment_sha256": hashlib.sha256(
                canonical(f5_environment.policy_binding())
            ).hexdigest(),
        },
        "input_hashes": {
            row["dest"]: {"sha256": row["sha256"], "size": row["size_bytes"]} for row in inputs
        },
        "environment_policy": f5_environment.policy_binding(),
        "environment_reference": capture05_reference(),
        "execution_envelope": envelope(),
        "external_acceptance_required": True,
    }
    export_json(staging / "control.json", control)
    inputs.append(reference(staging / "control.json", "control.json"))
    for suffix in (".sha256", ".complete"):
        inputs.append(reference(Path(str(ledger) + suffix), ledger.name + suffix))
    manifest = {
        "job_id": "f5-step3-longbody-v3-attempt01",
        "commit": commit,
        "script": "research/bl_residual/f5_step3.sh",
        "device": "cuda:0",
        "env": {**THREADS, "F5_STEP3_MODE": "measure"},
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 6, "est_minutes": 60},
        "timeout_minutes": 120,
        "budget": {
            "unit_cap": 8.0,
            "rate_units_per_hour": 2.0,
            "reserve_units": 0.0,
            "other_assignments": {},
        },
        "output_dir": str(output / "runtime-output-attempt01"),
        "supervisor_state_dir": str(output / "state-attempt01"),
        "inputs": inputs,
        "source_paths": sorted(pins),
        "max_export_bytes": MAX_EXPORT_BYTES,
        "required_exports": ["qualification.json", "supplement-summary.json"],
        "resume": {
            "unit_kind": "fixture",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
    }
    export_json(output / "manifest.json", manifest)
    qualification = json.loads(json.dumps(manifest))
    qualification.update(
        job_id="f5-step3-qualification-v3-attempt01",
        timeout_minutes=30,
        output_dir=str(output / "qualification-output-attempt01"),
        supervisor_state_dir=str(output / "qualification-state-attempt01"),
        required_exports=["qualification.json", "environment.json"],
        resume_exemption={
            "kind": "acceptance",
            "reason": "Current-driver native-v6 CUDA parity; no completed timing unit replay",
            "timeout_minutes": 30,
        },
        canary={
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_residual/f5_step3.sh",
        },
    )
    qualification.pop("resume")
    qualification["requirements"]["est_minutes"] = 15
    qualification["env"].update(
        F5_STEP3_MODE="qualify",
        SNAKE_COLAB_CANARY="1",
        SNAKE_COLAB_CANARY_PACKAGE="research/bl_residual",
    )
    export_json(output / "qualification-manifest.json", qualification)
    export_json(output / "source-inventory.json", pins)
    export_json(output / "envelope.json", envelope())
    refs = [
        reference(REJECTION),
        reference(repo / ENGINEERING_SEAL),
        reference(
            repo / "research/bl_residual/f5_hold_correction_20261026/ENGINEERING_CORRECTIONS.md"
        ),
        reference(repo / (AMENDMENT + "/ENGINEERING_AMENDMENT.md")),
        reference(repo / (AMENDMENT + "/EXECUTION_PLAN.json")),
    ]
    seal = {
        "schema": "bl-f5-step3-package-seal/v1",
        "status": "SEALED_PENDING_INDEPENDENT_ACCEPTANCE",
        "commit": commit,
        "source_sha256": source_sha,
        "authority_references": refs,
        "files": [
            reference(output / name)
            for name in (
                "manifest.json",
                "qualification-manifest.json",
                "source-inventory.json",
                "envelope.json",
            )
        ]
        + [reference(Path(row["path"])) for row in inputs],
        "missing_admission": [
            "independent engineering acceptance",
            "independent exact-package acceptance",
            "matching genuine CUDA/native-v6 current-driver qualification",
            "complete measured finite forecast",
            "fresh account/rate/path locks",
        ],
        "dispatch_performed": False,
        "science_admission": False,
    }
    export_json(output / "PACKAGE_SEAL.json", seal)
    return seal


def verify(output: Path, *, repo: Path = ROOT) -> dict[str, Any]:
    """Dry-run: bytes and source only, no provider or application workload."""
    verify_export(output / "PACKAGE_SEAL.json")
    seal = json.loads((output / "PACKAGE_SEAL.json").read_text())
    for row in seal["files"] + seal["authority_references"]:
        observed = reference(Path(row["path"]))
        if observed != row:
            raise ValueError("sealed package input drift")
    pins = f5_package.source_inventory(repo)
    if pins != json.loads((output / "source-inventory.json").read_text()):
        raise ValueError("current source closure differs")
    from research.colab_cli.schema import validate_manifest

    validate_manifest(json.loads((output / "manifest.json").read_text()))
    from research.bl_residual.f5_step3_driver import run

    run(
        control_path=output / "inputs/control.json",
        fixtures=output / "inputs/timing-fixtures.json",
        inputs=output / "inputs",
        ledger=output / "inputs/verified-ledger.json",
        output=output / "runtime-output-attempt01",
        mode="dry-run",
    )
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("seal", "dry-run"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    value = build(args.path) if args.operation == "seal" else verify(args.path)
    print(
        json.dumps(
            {
                "status": value["status"],
                "source_sha256": value["source_sha256"],
                "dispatch_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
