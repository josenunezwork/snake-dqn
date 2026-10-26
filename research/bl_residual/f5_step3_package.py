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
NAMESPACE = "f5-step3-long-body-20261026-v5"
QUALIFICATION_MANIFEST = "qualification-manifest-attempt03.json"
ALLOCATION = "1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7"
PRIOR = (
    "research/bl_residual/ledger/F5_W_TIMING_OPERATIONAL_20261026_ROUND08_02_REQUAL/manifest.json"
)
AMENDMENT = "research/bl_residual/f5_hold_amendment_20261026"
ENGINEERING_SEAL = (
    "research/bl_residual/ledger/F5_ENGINEERING_SEAL_20261026_V5/ENGINEERING_SEAL.json"
)
MAX_EXPORT_BYTES = 64 * 1024 * 1024
REJECTION = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/"
    "REJECT_F5_HOLD_ENGINEERING_V3.md"
)
ARGON = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/argon-20261026/"
    "review-f5-hold-v3/answer.md"
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
        "family_canary_base_timeout_minutes": 15,
        "family_canary_effective_timeout_minutes": 30,
        "maximum_family_canaries": 3,
        "aggregate_effective_with_qualification_and_family_canaries_minutes": 480,
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
            row["dest"]: {"sha256": row["sha256"], "size": row["size_bytes"]}
            for row in inputs
            if row["dest"] != "verified-ledger.json"
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
        "job_id": "f5-step3-longbody-v5-attempt01",
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
        job_id="f5-step3-qualification-v5-attempt03",
        timeout_minutes=30,
        output_dir=str(output / "qualification-output-attempt03"),
        supervisor_state_dir=str(output / "qualification-state-attempt03"),
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
    export_json(output / QUALIFICATION_MANIFEST, qualification)
    export_json(output / "source-inventory.json", pins)
    export_json(output / "envelope.json", envelope())
    refs = [
        reference(REJECTION),
        reference(ARGON),
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
                QUALIFICATION_MANIFEST,
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
            "post-acceptance exact-inventory family canary for each attempt",
        ],
        "dispatch_performed": False,
        "science_admission": False,
    }
    export_json(output / "PACKAGE_SEAL.json", seal)
    return seal


def prepare_target(
    package: Path, output: Path, *, qualification: Path, forecast: Path, ledger: Path
) -> dict[str, Any]:
    """Stage reviewable measurement inputs; never invent qualification or acceptance.

    An independent reviewer binds acceptance_target_sha256 from TARGET.json.
    The dynamic ledger stays outside static science/control identity, but its
    exact bytes and every unit are uploaded by the final manifest and canary.
    """
    from research.bl_residual import f5_step3_driver as driver

    verify(package)
    if output.exists() or not output.is_absolute():
        raise ValueError("target directory must be absolute and fresh")
    verify_export(qualification)
    verify_export(forecast)
    control = json.loads((package / "inputs/control.json").read_text())
    # Validate all resume units before creating the successor target.
    completed = driver.load_verified_units(ledger, control)
    output.mkdir()
    staging = output / "inputs"
    staging.mkdir()
    manifest = json.loads((package / "manifest.json").read_text())
    for row in manifest["inputs"]:
        if row["dest"] in {
            "control.json",
            "qualification.json",
            "external-acceptance.json",
            "verified-ledger.json",
            "verified-ledger.json.sha256",
            "verified-ledger.json.complete",
        }:
            continue
        export_bytes(staging / row["dest"], Path(row["path"]).read_bytes())
    export_bytes(staging / "qualification.json", qualification.read_bytes())
    export_bytes(staging / "verified-ledger.json", ledger.read_bytes())
    for fixture_id in completed:
        source = ledger.parent / "units" / f"fixture-{fixture_id}.json"
        export_bytes(staging / "units" / source.name, source.read_bytes())
    control["qualification_sha256"] = digest(qualification)
    control["input_hashes"]["qualification.json"] = {
        "sha256": digest(qualification),
        "size": qualification.stat().st_size,
    }
    control["execution_envelope"] = json.loads(forecast.read_text())
    export_json(staging / "control.json", control)
    driver.validate_control(
        staging / "control.json", staging / "timing-fixtures.json", staging, measure=True
    )
    value = {
        "schema": "bl-f5-step3-review-target/v1",
        "package": str(package),
        "acceptance_target_sha256": driver._acceptance_target_sha256(control),
        "execution_envelope_sha256": driver._canonical_sha(control["execution_envelope"]),
        "qualification_sha256": control["qualification_sha256"],
        "source_sha256": control["source_sha256"],
        "fixture_descriptor_sha256": control["fixture_descriptor_sha256"],
        "ledger_sha256": digest(ledger),
        "science_admission": False,
    }
    export_json(output / "TARGET.json", value)
    return value


def seal_final(target: Path, output: Path, *, acceptance: Path, attempt: int) -> dict[str, Any]:
    """Seal the exact matching family-canary/measurement pair after review.

    Each mechanical completion has a new exact-inventory canary. A changed
    ledger cannot reuse an earlier receipt. No receipt is manufactured here.
    """
    from research.bl_residual import f5_step3_driver as driver
    from research.colab_cli.canary import input_inventory
    from research.colab_cli.schema import validate_manifest

    if type(attempt) is not int or not 1 <= attempt <= 3:
        raise ValueError("only original plus two mechanical completion attempts")
    if output.exists() or not output.is_absolute():
        raise ValueError("final directory must be absolute and fresh")
    verify_export(target / "TARGET.json")
    authority = json.loads((target / "TARGET.json").read_text())
    package = Path(authority["package"])
    verify(package)
    control = json.loads((target / "inputs/control.json").read_text())
    if driver._acceptance_target_sha256(control) != authority["acceptance_target_sha256"]:
        raise ValueError("review target control changed")
    if digest(target / "inputs/verified-ledger.json") != authority["ledger_sha256"]:
        raise ValueError("review target ledger changed")
    driver.load_verified_units(target / "inputs/verified-ledger.json", control)
    verify_export(acceptance)
    if acceptance.name != "external-acceptance.json":
        raise ValueError("acceptance must be named external-acceptance.json")
    control["input_hashes"]["external-acceptance.json"] = {
        "sha256": digest(acceptance),
        "size": acceptance.stat().st_size,
    }
    driver._acceptance(control, acceptance.parent, target / "inputs/control.json")
    output.mkdir()
    staging = output / "inputs"
    staging.mkdir()
    # Copy payloads through the immutable export API, regenerating exact sidecars.
    for path in sorted((target / "inputs").rglob("*")):
        if path.is_file() and not path.name.endswith((".sha256", ".complete")):
            if path.name != "control.json":
                verify_export(path)
                export_bytes(staging / path.relative_to(target / "inputs"), path.read_bytes())
    export_json(staging / "control.json", control)
    export_bytes(staging / "external-acceptance.json", acceptance.read_bytes())
    driver.validate_control(
        staging / "control.json", staging / "timing-fixtures.json", staging, measure=True
    )
    manifest = json.loads((package / "manifest.json").read_text())
    manifest.update(
        job_id=f"f5-step3-longbody-v5-attempt{attempt:02}",
        output_dir=str(output / "runtime-output"),
        supervisor_state_dir=str(output / "state"),
        inputs=[
            reference(p, p.relative_to(staging).as_posix())
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        ],
    )
    canary = json.loads(json.dumps(manifest))
    canary.update(
        job_id=f"f5-step3-final-inventory-v5-attempt{attempt:02}",
        output_dir=str(output / "canary-output"),
        supervisor_state_dir=str(output / "canary-state"),
        timeout_minutes=30,
        canary={"schema": "colab-family-canary/v1", "mode": "probe", "family": manifest["script"]},
        resume_exemption={
            "kind": "acceptance",
            "reason": "Exact final inventory; no timing replay",
            "timeout_minutes": 30,
        },
    )
    canary.pop("resume")
    canary["requirements"]["est_minutes"] = 15
    canary["env"].update(
        F5_STEP3_MODE="family-canary",
        SNAKE_COLAB_CANARY="1",
        SNAKE_COLAB_CANARY_PACKAGE="research/bl_residual",
    )
    assert input_inventory(canary) == input_inventory(manifest)
    assert canary["required_exports"] == manifest["required_exports"]
    for name, value in (("manifest.json", manifest), ("family-canary-manifest.json", canary)):
        validate_manifest(value)
        export_json(output / name, value)
    seal = {
        "schema": "bl-f5-step3-final-package/v1",
        "attempt": attempt,
        "status": "SEALED_PENDING_MATCHING_FAMILY_CANARY_AND_DISPATCH_REVIEW",
        "acceptance_target_sha256": authority["acceptance_target_sha256"],
        "files": [
            reference(output / name) for name in ("manifest.json", "family-canary-manifest.json")
        ]
        + [reference(Path(row["path"])) for row in manifest["inputs"]],
        "dispatch_performed": False,
    }
    export_json(output / "FINAL_PACKAGE_SEAL.json", seal)
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


def verify_final(output: Path) -> dict[str, Any]:
    """Check final bytes and admitted driver without CUDA or provider access."""
    from research.bl_residual.f5_step3_driver import run
    from research.colab_cli.canary import input_inventory
    from research.colab_cli.schema import validate_manifest

    verify_export(output / "FINAL_PACKAGE_SEAL.json")
    seal = json.loads((output / "FINAL_PACKAGE_SEAL.json").read_text())
    for row in seal["files"]:
        if reference(Path(row["path"])) != row:
            raise ValueError("final sealed package input drift")
    main = json.loads((output / "manifest.json").read_text())
    probe = json.loads((output / "family-canary-manifest.json").read_text())
    for manifest in (main, probe):
        validate_manifest(manifest)
    if (
        input_inventory(main) != input_inventory(probe)
        or main["required_exports"] != probe["required_exports"]
    ):
        raise ValueError("final family canary inventory or export coverage differs")
    run(
        control_path=output / "inputs/control.json",
        fixtures=output / "inputs/timing-fixtures.json",
        inputs=output / "inputs",
        ledger=output / "inputs/verified-ledger.json",
        output=Path(main["output_dir"]),
        mode="dry-run",
    )
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("seal", "dry-run", "prepare-target", "seal-final", "dry-run-final")
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--forecast", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.operation == "dry-run-final":
        value = verify_final(args.path)
    elif args.operation == "prepare-target":
        if not all((args.output, args.qualification, args.forecast, args.ledger)):
            parser.error("prepare-target requires output, qualification, forecast and ledger")
        value = prepare_target(
            args.path,
            args.output,
            qualification=args.qualification,
            forecast=args.forecast,
            ledger=args.ledger,
        )
    elif args.operation == "seal-final":
        if not all((args.output, args.acceptance)):
            parser.error("seal-final requires output and acceptance")
        value = seal_final(args.path, args.output, acceptance=args.acceptance, attempt=args.attempt)
    else:
        value = build(args.path) if args.operation == "seal" else verify(args.path)
    print(
        json.dumps(
            {
                "status": value.get("status", "TARGET_PREPARED_FOR_REVIEW"),
                "source_sha256": value.get("source_sha256"),
                "dispatch_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
