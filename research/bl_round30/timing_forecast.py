"""Freeze a conservative R30-1 main-runtime estimate from timing-only metadata.

The calibration is evidence about elapsed time only.  This module never opens a
training bundle, scientific unit, checkpoint, diagnostic, or canary payload.
It deliberately does not decide G4/G5: an estimate only records that G5 remains
pending independent review.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_round30 import timing

SCHEMA = "r30-runtime-forecast/v1"
OPERATIONAL_SCHEMA = "r30-g5-timing-operational/v1"
MAIN_CEILING_SECONDS = 53 * 60
TRANSFER_BAR_SECONDS = 150.0
STAGING_BAR_SECONDS = 300.0
FIXED_PHASES = (
    "imports",
    "unpack",
    "input_checks_load",
    "feature_cache",
    "plan_coefficients",
    "equality",
    "adam_setup",
    "solver_auxiliary",
    "diagnostics",
    "bindings",
    "fusion_serialization",
    "main_export_io",
    "cleanup",
)
ADAM_BLOCKS = ("adam_0_10", "adam_10_20", "adam_20_30")
CLOSURES = ("closure_0", "closure_10", "closure_30")


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"finite {name} required")
    number = float(value)
    if number < 0 or (positive and number <= 0):
        raise ValueError(f"nonnegative {name} required")
    return number


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"valid {name} SHA-256 required")
    return value


def _receipt(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("native supervisor receipt object required")
    required = {
        "status",
        "stop_verified",
        "elapsed_seconds",
        "manifest_sha256",
        "commit",
        "job_id",
        "generation",
        "terminal",
        "exports",
    }
    if not required <= set(value):
        raise ValueError("native supervisor receipt incomplete")
    if value["status"] != "PASS" or value["stop_verified"] is not True:
        raise ValueError("native supervisor receipt is not terminal PASS")
    if value.get("cleanup_error") is not None:
        raise ValueError("native supervisor receipt cleanup is not verified")
    _number(value["elapsed_seconds"], "supervisor elapsed_seconds", positive=True)
    _sha(value["manifest_sha256"], "supervisor manifest")
    if not isinstance(value["commit"], str) or len(value["commit"]) != 40:
        raise ValueError("native supervisor receipt source commit differs")
    if not isinstance(value["job_id"], str) or not value["job_id"]:
        raise ValueError("native supervisor receipt job identity differs")
    terminal = value["terminal"]
    if (
        not isinstance(terminal, dict)
        or terminal.get("status") != "succeeded"
        or terminal.get("returncode") != 0
        or terminal.get("job_id") != value["job_id"]
        or str(terminal.get("generation")) != str(value["generation"])
    ):
        raise ValueError("native supervisor terminal receipt differs")
    exports = value["exports"]
    if not isinstance(exports, list) or not 1 <= len(exports) <= 65:
        raise ValueError("native supervisor exports differ")
    import re

    names = [entry.get("path") for entry in exports if isinstance(entry, dict)]
    if (
        len(names) != len(exports)
        or len(set(names)) != len(names)
        or names.count("timing.json") != 1
        or any(
            name != "timing.json"
            and not re.fullmatch(
                r"telemetry/0000[0-5][0-9]\.json|telemetry/00006[0-3]\.json", str(name)
            )
            for name in names
        )
    ):
        raise ValueError("native supervisor exports differ")
    for entry in exports:
        if entry["path"] != "timing.json":
            _sha(entry.get("sha256"), "telemetry export")
            if type(entry.get("size")) is not int or not 0 < entry["size"] <= 4096:
                raise ValueError("telemetry export size differs")
    if (
        sum(entry.get("size", 0) for entry in exports if type(entry.get("size")) is int)
        > timing.TIMING_PAYLOAD_BYTES
    ):
        raise ValueError("timing output exceeds 512 KiB")
    return value


def _operational(value: object, report: dict[str, Any]) -> dict[str, Any]:
    """Validate reviewed external cost metadata and immutable receipt bindings."""
    if not isinstance(value, dict) or value.get("schema") != OPERATIONAL_SCHEMA:
        raise ValueError("timing operational metadata schema differs")
    required = {
        "schema",
        "ruling_sha256",
        "amendment_sha256",
        "source_commit",
        "control_sha256",
        "input_hashes",
        "timing_report_sha256",
        "supervisor_receipt_sha256",
        "supervisor_receipt",
        "calibration_manifest_sha256",
        "timing_config_sha256",
        "calibration_required_exports",
        "main_output_transfer_seconds",
        "teardown_upper_seconds",
        "calibration_teardown_seconds",
        "transfer_upper_seconds",
        "staging_upper_seconds",
        "independent_review_status",
        "independent_review_sha256",
        "g4_status",
        "g5_status",
    }
    if set(value) != required:
        raise ValueError("timing operational metadata keys differ")
    if (
        value["ruling_sha256"] != timing.RULING_SHA
        or value["amendment_sha256"] != timing.AMENDMENT_SHA
    ):
        raise ValueError("timing ruling binding differs")
    if value["source_commit"] != report["source_commit"]:
        raise ValueError("source commit binding differs")
    if value["control_sha256"] != report["control_sha256"]:
        raise ValueError("control binding differs")
    if value["input_hashes"] != report["input_hashes"]:
        raise ValueError("input binding differs")
    if value["independent_review_status"] != "ACCEPTED":
        raise ValueError("independent operational-cost review is required")
    _sha(value["independent_review_sha256"], "independent review")
    _sha(value["timing_report_sha256"], "timing report")
    _sha(value["supervisor_receipt_sha256"], "supervisor receipt")
    manifest_sha = _sha(value["calibration_manifest_sha256"], "calibration manifest")
    if _sha(value["timing_config_sha256"], "timing config") != report["config_sha256"]:
        raise ValueError("timing config binding differs")
    if value["calibration_required_exports"] != ["timing.json"]:
        raise ValueError("calibration required exports differ")
    receipt = _receipt(value["supervisor_receipt"])
    if receipt["commit"] != report["source_commit"]:
        raise ValueError("supervisor receipt source binding differs")
    if receipt["manifest_sha256"] != manifest_sha:
        raise ValueError("supervisor receipt manifest binding differs")
    export = next(entry for entry in receipt["exports"] if entry["path"] == "timing.json")
    if (
        not isinstance(export, dict)
        or export.get("path") != "timing.json"
        or export.get("sha256") != value["timing_report_sha256"]
        or type(export.get("size")) is not int
        or not 0 < export["size"] <= timing.TIMING_PAYLOAD_BYTES
    ):
        raise ValueError("timing export receipt binding differs")
    for name in ("teardown_upper_seconds", "calibration_teardown_seconds"):
        _number(value[name], name)
    _number(value["main_output_transfer_seconds"], "main_output_transfer_seconds", positive=True)
    transfer = _number(value["transfer_upper_seconds"], "transfer_upper_seconds")
    staging = _number(value["staging_upper_seconds"], "staging_upper_seconds")
    if transfer > TRANSFER_BAR_SECONDS or staging > STAGING_BAR_SECONDS:
        raise ValueError("standard transport bar exceeded")
    residual = _number(receipt["elapsed_seconds"], "supervisor elapsed_seconds") - _number(
        report["worker_seconds"], "worker_seconds"
    )
    if (
        _number(value["calibration_teardown_seconds"], "calibration_teardown_seconds") > residual
        or transfer > staging
        or staging > residual
    ):
        raise ValueError("operational timing is not reconciled to supervisor residual")
    if value["g4_status"] not in {"PENDING", "INDEPENDENT_REVIEW_REQUIRED"}:
        raise ValueError("calibration cannot adopt or satisfy G4")
    if value["g5_status"] != "PENDING_ACCEPT":
        raise ValueError("calibration cannot accept G5")
    return value


def forecast(report: dict[str, Any], operational: dict[str, Any]) -> dict[str, Any]:
    """Calculate the frozen conservative extrapolation from metadata only.

    The formula is fixed before numerical execution.  It charges fixed phases
    once, 2,000 times the slowest ten-update mean, 500 times the slowest
    full-plan closure, a conservative solver-bookkeeping allowance, certificate
    and export probes, then supervisor residual and independently reviewed
    main-output/teardown costs exactly once each.
    """
    timing.validate_report(report)
    if report["synthetic"]:
        raise ValueError("synthetic timing report cannot forecast a real main run")
    operational = _operational(operational, report)
    phase = report["phase_seconds"]
    worker = _number(report["worker_seconds"], "worker_seconds", positive=True)
    receipt = operational["supervisor_receipt"]
    supervisor = _number(receipt["elapsed_seconds"], "supervisor elapsed_seconds", positive=True)
    if supervisor >= timing.DEADLINES["whole_session_seconds"]:
        raise ValueError("supervisor calibration timeout")
    if supervisor < worker:
        raise ValueError("supervisor elapsed time is below reconciled worker time")

    fixed = sum(float(phase[name]) for name in FIXED_PHASES)
    worker_phase_residual = max(0.0, worker - sum(float(value) for value in phase.values()))
    adam_per_ten = max(float(phase[name]) for name in ADAM_BLOCKS)
    closure = max(float(phase[name]) for name in CLOSURES)
    bookkeeping = max(300.0, 10.0 * float(phase["bookkeeping_probe"]))
    extrapolated_worker = (
        fixed
        + 2000.0 * adam_per_ten / 10.0
        + 500.0 * closure
        + bookkeeping
        + float(phase["certificate"])
        + float(phase["metadata_export_probe"])
    )
    supervisor_residual = supervisor - worker
    main_output = _number(
        operational["main_output_transfer_seconds"], "main_output_transfer_seconds", positive=True
    )
    teardown = max(
        0.0,
        _number(operational["teardown_upper_seconds"], "teardown_upper_seconds")
        - _number(operational["calibration_teardown_seconds"], "calibration_teardown_seconds"),
    )
    estimate = (
        extrapolated_worker + worker_phase_residual + supervisor_residual + main_output + teardown
    )
    report_sha = digest_bytes(report)
    if operational["timing_report_sha256"] != report_sha:
        raise ValueError("operational metadata does not bind timing report bytes")
    return {
        "schema": SCHEMA,
        "amendment_sha256": timing.AMENDMENT_SHA,
        "deadlines": timing.DEADLINES,
        "source_commit": report["source_commit"],
        "input_control_sha256": report["control_sha256"],
        "timing_report_sha256": report_sha,
        "supervisor_receipt_sha256": operational["supervisor_receipt_sha256"],
        "operational_metadata_sha256": digest_bytes(operational),
        "measured_est_minutes": estimate / 60.0,
        "measured_est_seconds": estimate,
        "main_ceiling_seconds": MAIN_CEILING_SECONDS,
        "within_main_ceiling": estimate <= MAIN_CEILING_SECONDS,
        "g4_status": "PENDING_INDEPENDENT_REVIEW",
        "g5_status": "PENDING_ACCEPT",
        "formula": {
            "fixed_phase_seconds": fixed,
            "excluded_probe_decision_seconds": float(phase["b8_masked_action_decision"]),
            "worker_phase_residual_seconds": worker_phase_residual,
            "adam_block_seconds": adam_per_ten,
            "adam_extrapolated_seconds": 2000.0 * adam_per_ten / 10.0,
            "closure_seconds": closure,
            "closure_extrapolated_seconds": 500.0 * closure,
            "bookkeeping_allowance_seconds": bookkeeping,
            "certificate_seconds": float(phase["certificate"]),
            "metadata_export_probe_seconds": float(phase["metadata_export_probe"]),
            "supervisor_residual_seconds": supervisor_residual,
            "main_output_transfer_seconds": main_output,
            "incremental_teardown_seconds": teardown,
            "independent_review_sha256": operational["independent_review_sha256"],
            "transport_bars_seconds": {
                "transfer": float(operational["transfer_upper_seconds"]),
                "staging": float(operational["staging_upper_seconds"]),
            },
        },
    }


def digest_bytes(value: dict[str, Any]) -> str:
    """Canonical SHA-256 matching the create-only JSON exporter."""
    import hashlib

    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _read_native_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("unsafe native supervisor receipt")
    value = json.loads(path.read_text())
    _receipt(value)
    return value


def _read_object(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe {name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _validate_cli_bindings(
    report: dict[str, Any],
    operational: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, Any],
    review: dict[str, Any],
    *,
    manifest_path: Path,
    config_path: Path,
    review_path: Path,
) -> None:
    """Bind raw reviewed packet metadata without opening a scientific payload."""
    if operational["calibration_manifest_sha256"] != digest(manifest_path):
        raise ValueError("operational metadata does not bind calibration manifest bytes")
    if operational["timing_config_sha256"] != digest(config_path):
        raise ValueError("operational metadata does not bind timing config bytes")
    if operational["independent_review_sha256"] != digest(review_path):
        raise ValueError("operational metadata does not bind independent review bytes")
    if review.get("decision", review.get("status")) not in {"ACCEPT", "ACCEPTED"}:
        raise ValueError("independent review is not accepted")
    if (
        manifest.get("commit") != report["source_commit"]
        or not isinstance(manifest.get("output_dir"), str)
        or not manifest["output_dir"]
        or manifest.get("required_exports") != ["timing.json"]
        or manifest.get("env", {}).get("R30_TIMING_CONFIG_SHA256") != report["config_sha256"]
    ):
        raise ValueError("calibration manifest timing binding differs")
    from research.bl_round30.timing_pack import BASE_MINUTES, EFFECTIVE_MINUTES

    if (
        manifest.get("timeout_minutes") != EFFECTIVE_MINUTES
        or manifest.get("requirements", {}).get("est_minutes") != BASE_MINUTES
        or manifest.get("resume_exemption", {}).get("timeout_minutes") != EFFECTIVE_MINUTES
        or manifest.get("canary")
        != {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_round30/timing_colab.sh",
        }
        or manifest.get("timing_calibration", {}).get("no_automatic_retry") is not True
        or config.get("deadlines") != timing.DEADLINES
        or config.get("amendment_sha256") != timing.AMENDMENT_SHA
        or config.get("observability_ruling_sha256") != timing.OBSERVABILITY_SHA
    ):
        raise ValueError("calibration probe deadlines/amendment differ")
    if config.get("source_hashes") != report["source_hashes"]:
        raise ValueError("timing config source hash binding differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-json", type=Path, required=True)
    parser.add_argument("--supervisor-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--operational", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("forecast output is create-only")
    if args.timing_json.name != "timing.json":
        raise ValueError("timing output filename differs")
    timing.validate_output_inventory(args.timing_json.parent)
    verify_export(args.operational)
    report = json.loads(args.timing_json.read_text())
    operational = json.loads(args.operational.read_text())
    receipt = _read_native_receipt(args.supervisor_receipt)
    from research.bl_round30.timing_telemetry import validate_inventory

    records = validate_inventory(args.timing_json.parent, complete=True)
    if records[-1]["completed_phase_seconds"] != report["phase_seconds"]:
        raise ValueError("completed telemetry/report timings differ")
    local_exports = {"timing.json": digest(args.timing_json)}
    local_exports.update(
        {
            f"telemetry/{record['sequence']:06d}.json": digest(
                args.timing_json.parent / "telemetry" / f"{record['sequence']:06d}.json"
            )
            for record in records
        }
    )
    if {entry["path"]: entry["sha256"] for entry in receipt["exports"]} != local_exports:
        raise ValueError("native telemetry inventory binding differs")
    manifest = _read_object(args.manifest, "calibration manifest")
    config = _read_object(args.config, "timing config")
    review = _read_object(args.review, "independent review")
    if operational.get("supervisor_receipt") != receipt:
        raise ValueError("operational metadata does not bind native supervisor receipt")
    if operational.get("supervisor_receipt_sha256") != digest(args.supervisor_receipt):
        raise ValueError("operational metadata does not bind native supervisor receipt bytes")
    _validate_cli_bindings(
        report,
        operational,
        manifest,
        config,
        review,
        manifest_path=args.manifest,
        config_path=args.config,
        review_path=args.review,
    )
    result = forecast(report, operational)
    export_json(args.output, result)


if __name__ == "__main__":
    main()
