"""Hermetic F4 D1024 transfer-exception boundaries for acceptance evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.colab_cli import acceptance, supervisor, transfer_policy

REPO = Path(__file__).resolve().parents[1]


def test_partial_staging_diagnostic_cannot_be_acceptance_evidence(tmp_path: Path) -> None:
    path = tmp_path / "upload-measurement.json"
    path.write_text(
        json.dumps(
            {
                "elapsed_seconds": 20.0,
                "transfer_seconds": 15.0,
                "source_delivery_seconds": 0.0,
                "staging_completed": False,
            }
        )
    )
    with pytest.raises(supervisor.Refused, match="partial staging"):
        acceptance._d1024_staging_measurement(path)


def _l4_plan(tmp_path: Path, authority: str | None) -> tuple[dict[str, object], dict[str, object]]:
    plan = acceptance.build_plan(
        REPO,
        tmp_path / "state",
        d1024_inventory=True,
        transfer_authority_sha256=authority,
    )["plan"]
    job = next(row for row in plan["jobs"] if row["name"] == "l4")
    return plan, job


def _staging_evidence(
    tmp_path: Path,
    plan: dict[str, object],
    job: dict[str, object],
    *,
    transfer_seconds: float,
    elapsed_seconds: float,
) -> dict[str, object]:
    manifest = json.loads(Path(job["manifest"]).read_text())
    normalized = supervisor.validate(manifest, REPO)
    output = Path(manifest["output_dir"])
    output.mkdir(parents=True)
    (output / "public-input-preflight.json").write_text("{}")
    (output / "upload-measurement.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": elapsed_seconds,
                "transfer_seconds": transfer_seconds,
                "source_delivery_seconds": 0.0,
            }
        )
    )
    transfers = [
        {
            "direction": "upload",
            "status": "verified",
            "remote": f"/content/fake/inputs/{row['dest']}",
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in manifest["inputs"]
        if row["dest"] in {"factory-verified-ledger.json", "train-1024.npz", "metadata.tar"}
    ]
    saved = output / "manifest.json"
    saved.write_text(json.dumps(normalized))
    receipt = {
        "transfers": transfers,
        "commit": manifest["commit"],
        "job_id": manifest["job_id"],
        "manifest_sha256": acceptance.sha256_file(saved),
    }
    (output / "receipt.json").write_text(json.dumps(receipt))
    return acceptance.d1024_synthetic_staging_evidence(plan, job, receipt, output)


def test_d1024_authorized_222_second_transfer_and_236_second_staging_pass(tmp_path: Path) -> None:
    plan, job = _l4_plan(tmp_path, transfer_policy.RULING_SHA256)
    manifest = json.loads(Path(job["manifest"]).read_text())
    assert manifest["transfer_exception"] == transfer_policy.authority_binding("acceptance")
    assert transfer_policy.transfer_limit(manifest) == 300.0
    evidence = _staging_evidence(tmp_path, plan, job, transfer_seconds=222.0, elapsed_seconds=236.0)
    assert evidence["status"] == "PASS"


def test_d1024_authorized_300_second_transfer_and_staging_boundaries_pass(tmp_path: Path) -> None:
    plan, job = _l4_plan(tmp_path, transfer_policy.RULING_SHA256)
    evidence = _staging_evidence(tmp_path, plan, job, transfer_seconds=300.0, elapsed_seconds=300.0)
    assert evidence["upload_measurement"]["transfer_seconds"] == 300.0


def test_d1024_staging_over_300_seconds_refuses_independently(tmp_path: Path) -> None:
    plan, job = _l4_plan(tmp_path, transfer_policy.RULING_SHA256)
    with pytest.raises(supervisor.Refused, match="staging exceeds the 300-second cap"):
        _staging_evidence(tmp_path, plan, job, transfer_seconds=222.0, elapsed_seconds=300.001)


def test_missing_authority_keeps_150_second_default(tmp_path: Path) -> None:
    plan, job = _l4_plan(tmp_path, None)
    manifest = json.loads(Path(job["manifest"]).read_text())
    assert "transfer_exception" not in manifest
    assert transfer_policy.transfer_limit(manifest) == 150.0
    with pytest.raises(supervisor.Refused, match="150-second bar"):
        _staging_evidence(tmp_path, plan, job, transfer_seconds=222.0, elapsed_seconds=236.0)


def test_combined_f5_union_keeps_d1024_subset_at_150_seconds(tmp_path: Path) -> None:
    """The F5 union may exceed 150s without silently widening D1024's subset."""
    measurement = tmp_path / "upload-measurement.json"
    measurement.write_text(
        json.dumps(
            {
                "elapsed_seconds": 240.0,
                "transfer_seconds": 221.0,
                "source_delivery_seconds": 16.0,
                "input_transfer_seconds": {
                    "factory-verified-ledger.json": 45.0,
                    "train-1024.npz": 55.0,
                    "metadata.tar": 50.0,
                    "f5-control.json": 71.0,
                },
            }
        )
    )
    evidence = acceptance._d1024_subset_transfer_measurement(
        measurement, acceptance.d1024_synthetic_inventory()
    )
    assert evidence["transfer_seconds"] == 221.0
    assert evidence["d1024_transfer_seconds"] == 150.0


def test_foreign_authority_and_non_d1024_opt_in_refuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid"):
        _l4_plan(tmp_path, "0" * 64)
    with pytest.raises(ValueError, match="only valid"):
        acceptance.build_plan(
            REPO,
            tmp_path / "other-state",
            transfer_authority_sha256=transfer_policy.RULING_SHA256,
        )


def test_verify_cli_forwards_required_transfer_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def verify(_repo: Path, _record: Path, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(acceptance, "verify_qualified_transport_record", verify)
    assert (
        acceptance.main(
            [
                "--repo",
                str(REPO),
                "--verify-qualified-record",
                str(tmp_path / "qualified.json"),
                "--required-transfer-authority-sha256",
                transfer_policy.RULING_SHA256,
            ]
        )
        == 0
    )
    assert observed["required_transfer_authority_sha256"] == transfer_policy.RULING_SHA256


def test_exception_must_be_declared_consistently_before_execution(tmp_path: Path) -> None:
    plan, job = _l4_plan(tmp_path, transfer_policy.RULING_SHA256)
    plan.pop("transfer_exception")
    with pytest.raises(ValueError, match="undeclared transfer authority"):
        transfer_policy.acceptance_limit(plan)
    with pytest.raises(supervisor.Refused, match="authority differs from plan"):
        _staging_evidence(tmp_path, plan, job, transfer_seconds=222, elapsed_seconds=236)


@pytest.mark.parametrize("fault", ["missing", "foreign", "hash", "commit", "job"])
def test_exception_requires_receipt_bound_actual_manifest(tmp_path: Path, fault: str) -> None:
    plan, job = _l4_plan(tmp_path, transfer_policy.RULING_SHA256)
    _staging_evidence(tmp_path, plan, job, transfer_seconds=222, elapsed_seconds=236)
    manifest = json.loads(Path(job["manifest"]).read_text())
    output = Path(manifest["output_dir"])
    saved = output / "manifest.json"
    actual = json.loads(saved.read_text())
    receipt = json.loads((output / "receipt.json").read_text())
    if fault == "missing":
        actual.pop("transfer_exception")
    elif fault == "foreign":
        actual["transfer_exception"]["ruling_sha256"] = "0" * 64
    elif fault == "commit":
        actual["commit"] = "0" * 40
    elif fault == "job":
        actual["job_id"] = "foreign-job"
    else:
        actual["tamper"] = True
    saved.write_text(json.dumps(actual))
    if fault != "hash":
        receipt["manifest_sha256"] = acceptance.sha256_file(saved)
        (output / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(supervisor.Refused, match="actual manifest"):
        acceptance.d1024_synthetic_staging_evidence(plan, job, receipt, output)
    with pytest.raises(ValueError, match="actual manifest"):
        transfer_policy.receipt_limit(manifest, receipt, output)
