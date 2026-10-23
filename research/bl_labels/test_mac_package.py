"""Outcome-blind tests for the Mac equivalence and held-D256 package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_labels.exports import load_ledger, publish_json
from research.bl_labels.mac_package import (
    CANARY_WALL_SECONDS,
    F1_BANK_SHA256,
    d256_allocation,
    d256_manifest,
    make_canary_control,
    prepare,
)
from research.bl_labels.manifests import canary_allocation


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "canary-a-control.json"
    publish_json(
        path,
        {
            "schema": "bl-labels-f0-control/v1",
            "namespace": "f0-canary-a",
            "allocation": canary_allocation(),
            "workers": "nproc",
        },
    )
    return path


def test_canary_b_preserves_exact_canary_a_identity_and_single_look(tmp_path: Path) -> None:
    source = _source(tmp_path)
    paths = prepare(
        source,
        tmp_path / "mac-package",
        source_commit="a" * 40,
        expected_canary_a_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    control = json.loads(paths["control"].read_text())
    canary = json.loads(paths["canary"].read_text())
    assert control["namespace"] == "f0-canary-a"
    assert control["allocation"] == canary_allocation()
    assert control["workers"] == 6
    assert control["mac_gate"]["stage"] == "canary-b"
    assert control["mac_gate"]["status"] == "HOLD"
    assert canary["registered_look"]["count"] == 1
    assert canary["registered_look"]["rescue_floor_per_mac_worker_hour"] == 8
    assert canary["wall_cap_seconds"] == CANARY_WALL_SECONDS
    assert canary["preconditions"]["f1_bank_sha256"] == F1_BANK_SHA256
    assert load_ledger(paths["ledger"], control["namespace"], control["allocation"]) == set()


def test_mac_package_refuses_noncanonical_canary_population(tmp_path: Path) -> None:
    source = _source(tmp_path)
    value = json.loads(source.read_text())
    value["allocation"] = value["allocation"][:-1]
    bad = tmp_path / "bad.json"
    publish_json(bad, value)
    with pytest.raises(ValueError, match="exact frozen 36"):
        make_canary_control(bad, expected_sha256=hashlib.sha256(bad.read_bytes()).hexdigest())


def test_d256_is_fresh_namespace_and_held_behind_both_required_gates(tmp_path: Path) -> None:
    manifest = d256_manifest(tmp_path, source_commit="b" * 40)
    assert manifest["status"] == "HOLD"
    assert manifest["namespace"] == "f0-hindsight-labels-" + "b" * 40 + "-mac-d256"
    assert manifest["workers"] == 6
    assert manifest["operational_cap"]["effective_wall_hours"] == 16
    assert manifest["selection"]["fit_lives"] == 192
    assert manifest["selection"]["reserve_lives"] == 64
    assert manifest["gates"]["f4_approval"]["decision"] == "APPROVE"
    assert manifest["gates"]["f4_approval"]["recipe_sha256"]
    allocation = d256_allocation(manifest["namespace"])
    assert len(allocation) == 864
    assert sum(row["stream"] == "watch6" for row in allocation) == 96
    assert sum(row["stream"] == "gate-frozen" for row in allocation) == 288
    assert sum(row["stream"] == "gate-mixed" for row in allocation) == 288
    assert sum(row["stream"] == "gate-scripted" for row in allocation) == 192
    assert len({row["seed"] for row in allocation}) == 864
