"""Validate the small, final-only S1 artifact transport contract.

S1 makes decisions from its final online and target weights plus the immutable
run receipt, telemetry, and logs.  It deliberately does not transfer replay or
full trainer state: those bytes cannot be used for an exact resume or for
checkpoint selection under the draft registration.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "s1-final-artifact-transport/v1"
SIZE_SCHEMA = "s1-final-artifact-size/v1"
PROTOCOL = "s1-final-artifact-export/v1"
MAX_FINAL_EXPORT_BYTES = 64 * 1024**2
REQUIRED_ARTIFACT_FILES = frozenset(
    {
        "final.pth",
        "target.pth",
        "summary.json",
        "s1-training-provenance.json",
        "log.jsonl",
        "telemetry.jsonl",
        "manifest.jsonl",
        "runtime-parity.json",
        "t22-cuda.json",
        "t24-cuda.json",
    }
)
SOURCES = (
    "research/s1_mature_20261018/final_artifacts.py",
    "research/s1_mature_20261018/manifests.py",
    "research/s1_mature_20261018/train.py",
    "research/redesign_m3c_20261008/train_m3c.py",
)
CLI_SOURCES = (
    "research/colab_cli/source.py",
    "research/colab_cli/supervisor.py",
    "research/colab_cli/transfers.py",
    "research/colab_cli/runtime.py",
    "research/colab_cli/bootstrap.sh",
    "research/colab_cli/acceptance.py",
    "research/colab_cli/acceptance_probe.py",
)


def binding(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def source_manifest(paths: tuple[str, ...]) -> dict[str, str]:
    return {relative: binding(ROOT / relative)["sha256"] for relative in paths}


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _bound_receipt(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{name} requires a path and sha256")
    path = Path(value["path"])
    if not path.is_file() or binding(path)["sha256"] != value.get("sha256"):
        raise ValueError(f"{name} receipt absent or changed")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"{name} receipt must be a JSON object")
    return result


def transport_bound(value: dict[str, Any]) -> dict[str, Any]:
    """Validate final-bundle size bounds against a CLI acceptance receipt."""
    if value.get("schema") != SCHEMA or value.get("status") != "PASS":
        raise ValueError("final-artifact transport qualification is absent")
    expected_sources = source_manifest(SOURCES)
    if value.get("source_manifest") != expected_sources:
        raise ValueError("final-artifact source binding changed")
    acceptance = _bound_receipt(value.get("cli_acceptance_receipt"), "CLI acceptance")
    transfer = acceptance.get("s1_final_artifact_transfer")
    if acceptance.get("qualified") is not True or not isinstance(transfer, dict):
        raise ValueError("CLI acceptance receipt lacks measured final-artifact throughput")
    if (
        transfer.get("status") != "PASS"
        or transfer.get("hardware") != "L4"
        or transfer.get("protocol") != PROTOCOL
        or transfer.get("verified_on_mac") is not True
        or transfer.get("transport_source_sha256")
        != expected_sources["research/s1_mature_20261018/train.py"]
        or transfer.get("transport_source_manifest") != source_manifest(CLI_SOURCES)
    ):
        raise ValueError("throughput must cover verified L4 final-only exports on the Mac")
    rate = _positive(transfer.get("bytes_per_second_lower_bound"), "measured throughput")
    sent = _positive(transfer.get("exported_bytes"), "measured exported bytes")
    seconds = _positive(transfer.get("elapsed_seconds"), "measured transfer seconds")
    if rate > sent / seconds:
        raise ValueError("throughput lower bound exceeds the measured transfer average")
    receipts = transfer.get("final_artifact_size_receipts")
    if not isinstance(receipts, dict):
        raise ValueError("CLI acceptance receipt lacks bound final-artifact size receipts")
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("n3", "n1"):
        measurement = _bound_receipt(receipts.get(arm), f"{arm} final-artifact size")
        if (
            measurement.get("schema") != SIZE_SCHEMA
            or measurement.get("status") != "PASS"
            or measurement.get("arm") != arm
            or measurement.get("source_manifest") != expected_sources
            or measurement.get("production_contract_validated") is not True
            or measurement.get("export_contract") != PROTOCOL
        ):
            raise ValueError(f"{arm} lacks a bound final-artifact size measurement")
        measured = _positive(
            measurement.get("measured_final_bundle_bytes"), f"{arm} measured bytes"
        )
        upper = _positive(measurement.get("final_bundle_bytes_upper_bound"), f"{arm} final bytes")
        allowance = measurement.get("telemetry_log_allowance_bytes")
        if isinstance(allowance, bool) or not isinstance(allowance, (int, float)) or allowance < 0:
            raise ValueError(f"{arm} telemetry/log allowance must be nonnegative")
        files = measurement.get("artifact_files")
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise ValueError(f"{arm} final-artifact file inventory is absent")
        if not REQUIRED_ARTIFACT_FILES.issubset(files):
            raise ValueError(f"{arm} final artifact inventory is incomplete")
        if upper < measured + allowance:
            raise ValueError(
                f"{arm} final-artifact bound omits its measured telemetry/log allowance"
            )
        if upper > MAX_FINAL_EXPORT_BYTES:
            raise ValueError(f"{arm} final-artifact bound exceeds the 64 MiB manifest cap")
        arms[arm] = {
            "measured_final_bundle_bytes": int(measured),
            "final_bundle_bytes_upper_bound": int(upper),
            "telemetry_log_allowance_bytes": int(allowance),
            "artifact_files": files,
            "transport_seconds_upper_bound": upper / rate,
        }
    return {
        **value,
        "measured_export_bytes_per_second_lower_bound": rate,
        "liability_formula": "final_bundle_bytes_upper_bound / "
        "measured_export_bytes_per_second_lower_bound",
        "qualified_arms": arms,
    }
