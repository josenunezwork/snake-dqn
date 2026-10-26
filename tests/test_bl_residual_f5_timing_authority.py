"""The F5 timing validator requires the actual recovery receipt and inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_residual import f5_timing
from research.colab_cli import transfer_policy
from tests.test_bl_residual_f5_timing import _proof
from tests.test_colab_cli_f5_timing_transfer import REPO, _manifest


def test_a_copied_binding_cannot_widen_selected_science_timing() -> None:
    proof = _proof()
    proof["transport"]["transfer_seconds"] = 222
    proof["transport"]["f5_timing_transfer_exception"] = (
        transfer_policy.f5_timing_authority_binding("0" * 64)
    )
    with pytest.raises(f5_timing.TimingHold, match="actual broker"):
        f5_timing.validate_operational_proof(
            proof,
            source_identity=proof["source_identity"],
            environment_identity=proof["environment_identity"],
            batch_identity=proof["batch_identity"],
        )


def _saved_recovery(tmp_path: Path) -> tuple[dict, dict, dict]:
    """Create the broker-owned artifacts the timing validator must consume."""
    manifest = _manifest(REPO, tmp_path)
    actual = tmp_path / "manifest.json"
    actual.write_text(json.dumps(manifest))
    receipt = {
        "commit": manifest["commit"],
        "job_id": manifest.get("job_id"),
        "manifest_sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        "status": "PASS",
        "stop_verified": True,
        "cleanup_error": None,
        "terminal": {"returncode": 0},
    }
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    measurement = {
        "elapsed_seconds": 240.0,
        "transfer_seconds": 222.0,
        "source_delivery_seconds": 16.0,
    }
    (tmp_path / "upload-measurement.json").write_text(json.dumps(measurement))
    return manifest, receipt, measurement


def _scoped_proof(manifest: dict, measurement: dict) -> dict:
    proof = _proof()
    proof["source_identity"] = {"commit": manifest["commit"]}
    proof["transport"]["transfer_seconds"] = measurement["transfer_seconds"]
    proof["transport"]["staging_seconds"] = measurement["elapsed_seconds"]
    proof["transport"]["f5_timing_transfer_exception"] = manifest["f5_timing_transfer_exception"]
    rows = manifest["f5_timing_synthetic_inventory"]["inputs"]
    proof["input_inventory"] = [
        {
            "name": row["dest"],
            "size": row["size_bytes"],
            "scope": "public_b8" if row["transport"] == "https" else "private",
            "release_qualified": True,
        }
        for row in rows
    ]
    proof["transport"]["private_input_bytes"] = sum(
        row["size_bytes"] for row in rows if row["transport"] == "upload"
    )
    return proof


@pytest.mark.parametrize("drift", [None, "source", "inventory", "receipt", "bytes"])
def test_scoped_timing_validation_uses_receipt_source_and_exact_bytes(
    tmp_path: Path, drift: str | None
) -> None:
    manifest, receipt, measurement = _saved_recovery(tmp_path)
    proof = _scoped_proof(manifest, measurement)
    if drift == "source":
        proof["source_identity"]["commit"] = "0" * 40
    elif drift == "inventory":
        proof["input_inventory"][0]["size"] += 1
    elif drift == "receipt":
        receipt["manifest_sha256"] = "0" * 64
    elif drift == "bytes":
        private = next(row for row in manifest["inputs"] if "path" in row)
        Path(private["path"]).write_bytes(b"changed")

    def validate() -> dict:
        return f5_timing.validate_operational_proof(
            proof,
            source_identity=proof["source_identity"],
            environment_identity=proof["environment_identity"],
            batch_identity=proof["batch_identity"],
            recovery_manifest=manifest,
            recovery_receipt=receipt,
            recovery_output=tmp_path,
        )

    if drift is None:
        assert validate()["status"] == "QUALIFIED"
    else:
        with pytest.raises(f5_timing.TimingHold):
            validate()


@pytest.mark.parametrize("fault", ["missing", "drift", "symlink", "over300"])
def test_scoped_timing_validation_requires_saved_matching_measurement(
    tmp_path: Path, fault: str
) -> None:
    manifest, receipt, measurement = _saved_recovery(tmp_path)
    proof = _scoped_proof(manifest, measurement)
    measured_path = tmp_path / "upload-measurement.json"
    if fault == "missing":
        measured_path.unlink()
    elif fault == "drift":
        proof["transport"]["transfer_seconds"] = 221.0
    elif fault == "symlink":
        replacement = tmp_path / "replacement.json"
        replacement.write_text(measured_path.read_text())
        measured_path.unlink()
        measured_path.symlink_to(replacement)
    else:
        measurement["elapsed_seconds"] = 301.0
        measured_path.write_text(json.dumps(measurement))
        proof["transport"]["staging_seconds"] = 301.0
    with pytest.raises(f5_timing.TimingHold):
        f5_timing.validate_operational_proof(
            proof,
            source_identity=proof["source_identity"],
            environment_identity=proof["environment_identity"],
            batch_identity=proof["batch_identity"],
            recovery_manifest=manifest,
            recovery_receipt=receipt,
            recovery_output=tmp_path,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "INCOMPLETE"),
        ("stop_verified", False),
        ("cleanup_error", "leaked"),
        ("terminal", {"returncode": 1}),
    ],
)
def test_scoped_timing_validation_refuses_nonterminal_recovery_receipt(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest, receipt, measurement = _saved_recovery(tmp_path)
    receipt[field] = value
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    proof = _scoped_proof(manifest, measurement)
    with pytest.raises(f5_timing.TimingHold):
        f5_timing.validate_operational_proof(
            proof,
            source_identity=proof["source_identity"],
            environment_identity=proof["environment_identity"],
            batch_identity=proof["batch_identity"],
            recovery_manifest=manifest,
            recovery_receipt=receipt,
            recovery_output=tmp_path,
        )
