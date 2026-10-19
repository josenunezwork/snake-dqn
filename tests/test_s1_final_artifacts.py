"""Final-only S1 transport receipt validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.s1_mature_20261018 import final_artifacts


def _receipt(tmp_path: Path) -> dict:
    sources = final_artifacts.source_manifest(final_artifacts.SOURCES)
    size_receipts = {}
    for arm in ("n3", "n1"):
        path = tmp_path / f"{arm}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": final_artifacts.SIZE_SCHEMA,
                    "status": "PASS",
                    "arm": arm,
                    "source_manifest": sources,
                    "production_contract_validated": True,
                    "export_contract": final_artifacts.PROTOCOL,
                    "measured_final_bundle_bytes": 12_000_000,
                    "final_bundle_bytes_upper_bound": 64 * 1024**2,
                    "telemetry_log_allowance_bytes": 1_000_000,
                    "artifact_files": sorted(final_artifacts.REQUIRED_ARTIFACT_FILES),
                }
            )
        )
        size_receipts[arm] = {
            "path": str(path),
            "sha256": final_artifacts.binding(path)["sha256"],
        }
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "qualified": True,
                "s1_final_artifact_transfer": {
                    "status": "PASS",
                    "hardware": "L4",
                    "protocol": final_artifacts.PROTOCOL,
                    "verified_on_mac": True,
                    "transport_source_sha256": sources["research/s1_mature_20261018/train.py"],
                    "transport_source_manifest": final_artifacts.source_manifest(
                        final_artifacts.CLI_SOURCES
                    ),
                    "exported_bytes": 12_000_000,
                    "elapsed_seconds": 12,
                    "bytes_per_second_lower_bound": 1_000_000,
                    "final_artifact_size_receipts": size_receipts,
                },
            }
        )
    )
    return {
        "schema": final_artifacts.SCHEMA,
        "status": "PASS",
        "source_manifest": sources,
        "cli_acceptance_receipt": {
            "path": str(acceptance),
            "sha256": final_artifacts.binding(acceptance)["sha256"],
        },
    }


def test_final_artifact_transport_charges_bound_size_at_measured_rate(tmp_path: Path) -> None:
    result = final_artifacts.transport_bound(_receipt(tmp_path))
    assert result["qualified_arms"]["n3"]["transport_seconds_upper_bound"] == pytest.approx(
        64 * 1024**2 / 1_000_000
    )
    assert "target.pth" in result["qualified_arms"]["n1"]["artifact_files"]


def test_final_artifact_transport_rejects_size_below_measurement(tmp_path: Path) -> None:
    value = _receipt(tmp_path)
    acceptance = Path(value["cli_acceptance_receipt"]["path"])
    transfer = json.loads(acceptance.read_text())["s1_final_artifact_transfer"]
    n3 = Path(transfer["final_artifact_size_receipts"]["n3"]["path"])
    size = json.loads(n3.read_text())
    size["final_bundle_bytes_upper_bound"] = size["measured_final_bundle_bytes"] - 1
    n3.write_text(json.dumps(size))
    transfer["final_artifact_size_receipts"]["n3"]["sha256"] = final_artifacts.binding(n3)["sha256"]
    acceptance.write_text(json.dumps({"qualified": True, "s1_final_artifact_transfer": transfer}))
    value["cli_acceptance_receipt"]["sha256"] = final_artifacts.binding(acceptance)["sha256"]
    with pytest.raises(ValueError, match="omits"):
        final_artifacts.transport_bound(value)
