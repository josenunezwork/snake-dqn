"""Outcome-blind tests for the Mac equivalence and held-D256 package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_labels import mac_gates
from research.bl_labels.exports import load_ledger, publish_json
from research.bl_labels.mac_package import (
    CANARY_WALL_SECONDS,
    F1_BANK_SHA256,
    FUTURE_PARENT_BLOCKS,
    d256_allocation,
    d256_manifest,
    future_control,
    future_manifest,
    future_seed_audit,
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


def test_future_controls_prefilter_fold_zero_and_append_after_d256() -> None:
    commit = "c" * 40
    d1024 = future_control(commit, stage="d1024")
    d4096 = future_control(commit, stage="d4096")
    assert len(d1024["allocation"]) == 9 * FUTURE_PARENT_BLOCKS["d1024"]
    assert len(d4096["allocation"]) == 9 * FUTURE_PARENT_BLOCKS["d4096"]
    assert all(row["world_fold"] in {1, 2, 3, 4} for row in d1024["allocation"])
    assert all(row["prepared_decision_shadow"] is True for row in d4096["allocation"])
    assert min(row["allocation_index"] for row in d1024["allocation"]) >= 864
    assert min(row["allocation_index"] for row in d4096["allocation"]) > max(
        row["allocation_index"] for row in d1024["allocation"]
    )
    assert (
        future_seed_audit(
            d1024["allocation"], prior_allocation_rows=d256_allocation("d256-sealed")
        )["prior_seed_collision"]
        is False
    )
    manifest = future_manifest(d4096, stage="d4096")
    assert manifest["status"] == "HOLD"
    assert manifest["selection"]["world_folds"] == [1, 2, 3, 4]


def test_d1024_readiness_uses_registered_f4_staging_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.bl_residual import job

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    control = inputs / "control.json"
    recipe = "a" * 64
    control.write_text(
        json.dumps(
            {"schema": "bl-residual-control/v1", "synthetic": False, "recipe_sha256": recipe}
        )
    )
    seen: dict[str, object] = {}

    def validate(path, value, requested, recipe_sha, skipped) -> None:
        seen.update(
            path=path, value=value, requested=requested, recipe_sha=recipe_sha, skipped=skipped
        )

    monkeypatch.setattr(mac_gates, "_f4_d1024_header", lambda *_: (recipe, set()))
    monkeypatch.setattr(job, "_validate_staging", validate)
    binding = mac_gates._future_readiness("d1024", control, inputs)
    assert binding["control_sha256"] == hashlib.sha256(control.read_bytes()).hexdigest()
    assert seen["path"] == inputs
    assert seen["requested"] == [1024]
    assert seen["recipe_sha"] == recipe


def test_d1024_readiness_refuses_missing_or_faulted_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    missing = tmp_path / "missing-control.json"
    with pytest.raises(ValueError, match="sealed F4 control"):
        mac_gates._future_readiness("d1024", missing, inputs)
    control = inputs / "control.json"
    control.write_text(
        json.dumps(
            {
                "schema": "bl-residual-control/v1",
                "synthetic": False,
                "recipe_sha256": "a" * 64,
                "stage_receipts": {},
            }
        )
    )
    monkeypatch.setattr(mac_gates, "_f4_d1024_header", lambda *_: ("a" * 64, set()))
    with pytest.raises(ValueError, match="D1024 requires passed D256"):
        mac_gates._future_readiness("d1024", control, inputs)


def test_d1024_readiness_rejects_synthetic_control_before_staging(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    control = inputs / "control.json"
    control.write_text(
        json.dumps(
            {"schema": "bl-residual-control/v1", "synthetic": True, "recipe_sha256": "a" * 64}
        )
    )
    with pytest.raises(ValueError, match="non-synthetic"):
        mac_gates._future_readiness("d1024", control, inputs)
