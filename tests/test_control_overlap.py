"""Exact-pin post-canary admission; synthetic control evidence only."""

from __future__ import annotations

import hashlib
import json

import pytest

from research.bl_labels.exports import publish_json
from tools.ops import control_overlap as overlap
from tools.ops import dispatch


def proof(tmp_path, monkeypatch, **change):
    evidence = {}
    for key in (
        "canary_manifest",
        "canary_receipt",
        "control_telemetry",
        "factory_process_telemetry",
    ):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({"stop_verified": True, "cleanup_error": None}))
        evidence[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    value = {
        "schema": "factory-colab-control-overlap/v1",
        "verdict": "PASS",
        "policy_id": overlap.POLICY_ID,
        "ruling_sha256": overlap.RULING_SHA256,
        "operational_pin": "a" * 40,
        "cli_pin": "b" * 40,
        "qualified_broker_source_closure": "c" * 40,
        "independent_reviewer": "focused verifier",
        "gates": dict.fromkeys(overlap.GATES, True),
        "evidence": evidence,
        **change,
    }
    path = tmp_path / "overlap.json"
    publish_json(path, value)
    monkeypatch.setattr(
        overlap,
        "_git",
        lambda root, ref: (
            "c" * 40 if ":" in ref else ("a" * 40 if root.name == "ops" else "b" * 40)
        ),
    )
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_future_main_accepts_only_exact_verified_overlap_proof(tmp_path, monkeypatch):
    binding = proof(tmp_path, monkeypatch)
    overlap.require_proof(binding, operational_root=tmp_path / "ops", broker=tmp_path / "broker")


@pytest.mark.parametrize(
    "field,value",
    [
        ("operational_pin", "0" * 40),
        ("cli_pin", "0" * 40),
        ("ruling_sha256", "0" * 64),
        ("verdict", "HOLD"),
        ("gates", {}),
        ("independent_reviewer", ""),
    ],
)
def test_wrong_pin_or_incomplete_proof_never_admits(tmp_path, monkeypatch, field, value):
    binding = proof(tmp_path, monkeypatch, **{field: value})
    with pytest.raises(ValueError):
        overlap.require_proof(
            binding, operational_root=tmp_path / "ops", broker=tmp_path / "broker"
        )


def test_prepared_status_cannot_admit_an_ordinary_main():
    value = {"colab_control_nice0": {"overlap_status": "PREPARED_NOT_QUALIFIED_FOR_OVERLAP"}}
    with pytest.raises(dispatch.Refused, match="authorized overlap canary"):
        dispatch.require_overlap_canary(value, json.dumps(value).encode())
