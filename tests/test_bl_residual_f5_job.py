"""Focused offline checks for the F5 runtime admission boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from research.bl_ppo.exports import export_json
from research.bl_residual import f5_job


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _metadata() -> dict[str, torch.Tensor]:
    n = 8
    return {
        "status": torch.ones(n, 6, dtype=torch.bool),
        "legal": torch.ones(n, 6, dtype=torch.bool),
        "action_mask": torch.ones(n, 6, dtype=torch.bool),
        "died": torch.tensor([[False, True, False, True, False, True]] * n),
        "ordinary": torch.tensor([False, True, False, True, False, True, False, True]),
        "life_id": torch.arange(n),
    }


def _inputs(tmp_path: Path) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    torch.save(_metadata(), inputs / "metadata.pt")
    identities = {
        name: "a" * 64
        for name in (
            "source_sha256",
            "loss_sha256",
            "environment_sha256",
            "allocation_sha256",
            "corpus_sha256",
        )
    }
    control = {
        "schema": f5_job.SCHEMA,
        "namespace": f5_job.NAMESPACE,
        "identity": identities,
        "input_hashes": {"metadata.pt": f5_job.digest(inputs / "metadata.pt")},
    }
    control["allocation"] = _sha(control)
    (inputs / "control.json").write_text(json.dumps(control, sort_keys=True))
    return inputs


def test_identity_requires_all_five_pins(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    control = f5_job._control(inputs)
    assert f5_job._identity(control).loss_sha256 == "a" * 64


def test_acceptance_refuses_placeholder(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    control = f5_job._control(inputs)
    # An opaque object is sufficient: the test covers the receipt boundary and
    # deliberately does not build a scientific plan or open a payload.
    plan = type("Plan", (), {"identity": f5_job._identity(control), "sha256": "b" * 64})()
    receipt = tmp_path / "acceptance.json"
    export_json(receipt, {"decision": "ACCEPT_F5_FIT"})
    with pytest.raises(f5_job.F5FitRefusal):
        f5_job._acceptance(receipt, control, plan)


def test_verified_ledger_refuses_changed_or_duplicate_unit_without_payload_open(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    control = f5_job._control(inputs)
    ledger = tmp_path / "verified-ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema": "bl-f5-verified-ledger/v1",
                "namespace": f5_job.NAMESPACE,
                "allocation": control["allocation"],
                "units": [
                    {"unit_id": "foreign", "kind": "COMPLETE_PAIRED_FIT_UNIT"},
                    {"unit_id": "also-foreign", "kind": "COMPLETE_PAIRED_FIT_UNIT"},
                ],
            }
        )
    )
    control["ledger_sha256"] = f5_job.digest(ledger)
    with pytest.raises(f5_job.F5FitRefusal, match="duplicate"):
        f5_job._verified_pair(ledger, control)
    ledger.write_text("changed")
    with pytest.raises(f5_job.F5FitRefusal, match="differs"):
        f5_job._verified_pair(ledger, control)


def test_ledger_relative_complete_unit_skips_after_cross_host_transfer(tmp_path: Path) -> None:
    from research.bl_ppo.exports import digest

    control = f5_job._control(_inputs(tmp_path))
    transferred = tmp_path / "transferred-inputs"
    receipt = transferred / "completed-pair.json"
    record = export_json(
        receipt,
        {
            "kind": "COMPLETE_PAIRED_FIT_UNIT",
            "namespace": f5_job.NAMESPACE,
            "allocation": control["allocation"],
            "identity": asdict(f5_job._identity(control)),
            "arms": {"C": {"updates_completed": 2000}, "W": {"updates_completed": 2000}},
        },
    )
    ledger = transferred / "verified-ledger.json"
    unit = {
        "unit_id": f"{f5_job.NAMESPACE}/{control['allocation']}/0",
        "kind": "COMPLETE_PAIRED_FIT_UNIT",
        "receipt_path": receipt.name,
        "receipt_sha256": record["sha256"],
        "receipt_size": record["size"],
    }
    export_json(
        ledger,
        {
            "schema": "bl-f5-verified-ledger/v1",
            "namespace": f5_job.NAMESPACE,
            "allocation": control["allocation"],
            "units": [unit],
        },
    )
    control["ledger_sha256"] = digest(ledger)
    assert f5_job._verified_pair(ledger, control) == unit


def test_prepare_uses_fresh_child_of_existing_supervisor_output(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _inputs(tmp_path)
    (inputs / "selected-training.tar.xz").write_bytes(b"synthetic package boundary")
    control = f5_job._control(inputs)
    monkeypatch.setattr(f5_job, "_source_admission", lambda _: None)
    monkeypatch.setattr(f5_job, "_environment_admission", lambda _: {})
    monkeypatch.setattr(f5_job, "_bound_inputs", lambda *a, **k: None)
    monkeypatch.setattr(f5_job, "_payload_metadata", lambda *a: (_metadata(), {}))
    supervisor_output = tmp_path / "supervisor-output"
    supervisor_output.mkdir()
    result = f5_job.prepare(inputs, supervisor_output / "prepare")
    assert result["rows"] == 512000
    assert (supervisor_output / "prepare" / f5_job.PLAN_NAME).is_file()
    assert result["allocation"] == control["allocation"]


def test_fused_export_reloads_through_normal_ego2s_loader(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from research.bl_residual.model import FrozenB8Residual, fuse_residual
    from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint

    torch.manual_seed(15)
    net = Ego2sNet(7, 60).eval()
    model = FrozenB8Residual(net)
    identity = f5_job._identity(f5_job._control(_inputs(tmp_path)))
    plan = SimpleNamespace(identity=identity, sha256="b" * 64)
    path = tmp_path / "fused.pth"
    torch.save(f5_job._checkpoint_blob(fuse_residual(model), arm="C", plan=plan), path)
    loaded = load_ego2s_checkpoint(str(path))
    observations = (
        torch.zeros(1, 7, 31, 31, dtype=torch.uint8),
        torch.zeros(1, 4, 37, 37, dtype=torch.uint8),
        torch.zeros(1, 60),
    )
    with torch.no_grad():
        assert torch.equal(net(*observations), loaded(*observations))
