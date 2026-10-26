"""Mac-only accounting proof for the one pre-allocation refusal alias."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from research.bl_ppo.exports import export_json
from research.bl_residual import f5_timing_resume as subject
from tests.test_bl_residual_f5_timing_accounting import _attempt, _history, _setup


def _save(root: Path, name: str, value: Any) -> Path:
    path = root / name
    export_json(path, value)
    return path


def _alias_case(root: Path, control: Path) -> tuple[Path, Path, Path, Path, str, str]:
    logical = subject._bound_metadata(subject.metadata_reference(control))["timing_attempt_id"]
    broker = "broker-02"
    control_ref = subject.metadata_reference(control)
    shared = {
        "commit": "e" * 40,
        "inputs": [{"dest": "control.json", "sha256": control_ref["sha256"]}],
        "budget": {"unit_cap": 100000.0, "reserve_units": 0.0},
        "supervisor_state_dir": "/old-state",
        "output_dir": "/old-output",
        "operational_admission": {},
    }
    old = {**shared, "job_id": logical}
    new = copy.deepcopy(old)
    new.update(job_id=broker, supervisor_state_dir="/new-state", output_dir="/new-output")
    new["budget"]["unit_cap"] = 100.0
    new["operational_admission"] = {
        "live_account_preflight": {"path": "/preflight", "sha256": "a" * 64, "size": 1}
    }
    old_raw = _save(root, "old-raw.json", old)
    old_normalized = _save(root, "old-normalized.json", {**old, "lease_seconds": 180.0})
    new_raw = _save(root, "new-raw.json", new)
    refusal = _save(
        root,
        "old-receipt.json",
        {
            "job_id": logical,
            "manifest_sha256": subject.metadata_reference(old_normalized)["sha256"],
            "commit": "e" * 40,
            "endpoint": None,
            "terminal": None,
            "exports": [],
            "error": "Refused('insufficient balance for cap plus study reserve')",
            "control_operations": [],
            "transfers": [],
            "cli_calls": [{"command": item} for item in ("version", "sessions", "usage")],
        },
    )
    alias = _save(
        root,
        "alias.json",
        {
            "schema": subject.PREALLOCATION_REFUSAL_ALIAS_SCHEMA,
            "attempt_index": 0,
            "logical_attempt_id": logical,
            "broker_attempt_id": broker,
            "predecessor": {
                "raw_manifest": subject.metadata_reference(old_raw),
                "refusal_receipt": subject.metadata_reference(refusal),
                "normalized_manifest": subject.metadata_reference(old_normalized),
            },
            "successor": {
                "raw_manifest": subject.metadata_reference(new_raw),
                "control": control_ref,
            },
            "allowed_leaf_diff": subject.PREALLOCATION_ALLOWED_LEAVES,
        },
    )
    return alias, old_raw, new_raw, old_normalized, logical, broker


def _static_alias(root: Path) -> tuple[Path, Path, Path, str, str]:
    control = _save(root, "control.json", {"timing_attempt_id": "logical-01"})
    alias, _, raw, _, logical, broker = _alias_case(root, control)
    return alias, raw, control, logical, broker


def test_valid_preallocation_refusal_alias_is_static_and_type_bound(tmp_path: Path) -> None:
    alias, raw, control, logical, broker = _static_alias(tmp_path)
    result = subject._preallocation_refusal_alias(
        subject.metadata_reference(alias),
        manifest=subject._bound_metadata(subject.metadata_reference(raw)),
        control=subject._bound_metadata(subject.metadata_reference(control)),
        attempt_index=0,
    )
    assert result["logical_attempt_id"] == logical
    assert result["broker_attempt_id"] == broker
    assert subject._typed_differences({}, {"x": {}}) == ["/x"]
    assert subject._typed_differences(False, 0) == [""]


@pytest.mark.parametrize("kind", ["control", "source", "receipt", "manifest"])
def test_alias_refuses_forged_bound_evidence(kind: str, tmp_path: Path) -> None:
    alias, raw, control, _, _ = _static_alias(tmp_path)
    value = subject._bound_metadata(subject.metadata_reference(alias), immutable=True)
    if kind == "control":
        value["successor"]["control"]["sha256"] = "0" * 64
    elif kind == "source":
        changed = subject._bound_metadata(value["successor"]["raw_manifest"])
        changed["commit"] = "f" * 40
        replacement = _save(tmp_path, "foreign-raw.json", changed)
        value["successor"]["raw_manifest"] = subject.metadata_reference(replacement)
    elif kind == "receipt":
        changed = subject._bound_metadata(value["predecessor"]["refusal_receipt"])
        changed["endpoint"] = "allocated"
        replacement = _save(tmp_path, "foreign-receipt.json", changed)
        value["predecessor"]["refusal_receipt"] = subject.metadata_reference(replacement)
    else:
        value["allowed_leaf_diff"] = ["/timeout_minutes"] * 7
    forged = _save(tmp_path, f"forged-{kind}.json", value)
    with pytest.raises(subject.ResumeRefusal):
        subject._preallocation_refusal_alias(
            subject.metadata_reference(forged),
            manifest=subject._bound_metadata(subject.metadata_reference(raw)),
            control=subject._bound_metadata(subject.metadata_reference(control)),
            attempt_index=0,
        )


def test_alias_rejects_later_attempt_and_future_normalized_manifest_drift(tmp_path: Path) -> None:
    alias, raw, control, _, broker = _static_alias(tmp_path)
    reference = subject.metadata_reference(alias)
    manifest = subject._bound_metadata(subject.metadata_reference(raw))
    control_value = subject._bound_metadata(subject.metadata_reference(control))
    normalized = _save(tmp_path, "observed-manifest.json", {**manifest, "lease_seconds": 180.0})
    receipt = {
        "job_id": broker,
        "manifest_sha256": subject.metadata_reference(normalized)["sha256"],
    }
    with pytest.raises(subject.ResumeRefusal, match="original attempt zero"):
        subject._preallocation_refusal_alias(
            reference,
            manifest=manifest,
            control=control_value,
            receipt=receipt,
            observed_normalized_manifest=normalized,
            attempt_index=1,
        )
    drifted = _save(tmp_path, "drifted-manifest.json", {**manifest, "lease_seconds": 181.0})
    with pytest.raises(subject.ResumeRefusal, match="successor normalization"):
        subject._preallocation_refusal_alias(
            reference,
            manifest=manifest,
            control=control_value,
            receipt=receipt,
            observed_normalized_manifest=drifted,
            attempt_index=0,
        )
    with pytest.raises(subject.ResumeRefusal, match="successor normalization"):
        subject._preallocation_refusal_alias(
            reference,
            manifest=manifest,
            control=control_value,
            receipt={**receipt, "manifest_sha256": "0" * 64},
            observed_normalized_manifest=normalized,
            attempt_index=0,
        )


def test_alias_seals_and_accounts_logical_control_under_successor_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject.platform, "system", lambda: "Darwin")
    ledger, environment, binding = _setup(tmp_path)
    attempts = [_attempt(tmp_path, index, binding, environment) for index in range(2)]
    control = Path(attempts[0]["control"]["path"])
    alias, _, raw, _, logical, broker = _alias_case(tmp_path, control)
    receipt = subject._bound_metadata(attempts[0]["broker_receipt"])
    for row in receipt["exports"]:
        path = tmp_path / row["path"]
        if not path.exists():
            export_json(path, subject._bound_metadata(environment))
    assert all((tmp_path / row["path"]).is_file() for row in receipt["exports"])
    receipt.update(job_id=broker, manifest_sha256="pending")
    output = tmp_path / "broker-output"
    output.mkdir()
    normalized = subject._bound_metadata(subject.metadata_reference(raw)) | {"lease_seconds": 180.0}
    normalized_path = _save(output, "manifest.json", normalized)
    receipt["manifest_sha256"] = subject.metadata_reference(normalized_path)["sha256"]
    receipt_path = _save(output, "receipt.json", receipt)
    attempts[0]["attempt_id"] = broker
    attempts[0]["manifest"] = subject.metadata_reference(raw)
    attempts[0]["broker_receipt"] = subject.metadata_reference(receipt_path)
    attempts[0]["accounting_measurement"] = _save(
        tmp_path,
        "broker-accounting.json",
        {**subject._bound_metadata(attempts[0]["accounting_measurement"]), "attempt_id": broker},
    )
    with pytest.raises(subject.ResumeRefusal, match="provenance"):
        subject.seal_attempt_accounting(
            tmp_path / "without-alias",
            binding,
            manifest=raw,
            broker_receipt=receipt_path,
            upload_measurement=Path(attempts[0]["upload_measurement"]["path"]),
            control=control,
            exports_root=tmp_path,
            attempt_index=0,
            environment_after_path="environment/after.json",
        )
    sealed = subject.seal_attempt_accounting(
        tmp_path / "sealed",
        binding,
        manifest=raw,
        broker_receipt=receipt_path,
        upload_measurement=Path(attempts[0]["upload_measurement"]["path"]),
        control=control,
        exports_root=tmp_path,
        attempt_index=0,
        environment_after_path="environment/after.json",
        preallocation_refusal_alias=alias,
    )
    attempts[0] = sealed
    history = _history(tmp_path, binding, environment, attempts)
    result = subject.reduce_ledger(ledger, binding, accounting_history=history)
    assert result["accounting_verified"] is True
    assert result["accounting_evidence"]["measured_attempts"][0]["attempt_id"] == broker
    assert logical not in {
        row["attempt_id"] for row in result["accounting_evidence"]["measured_attempts"]
    }
