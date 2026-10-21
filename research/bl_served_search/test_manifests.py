"""Offline envelope and dispatch-refusal checks; no CLI provider operations."""

import json
from pathlib import Path

import pytest

from research.bl_served_search.exports import digest, export_json
from research.bl_served_search.manifests import forecast, portable_allocation, source_paths
from research.bl_served_search.readiness import check


def test_forecast_keeps_overrun_visible() -> None:
    value = forecast()
    assert value["reference_floor_allocation_minutes"] * 2 <= value["timeout_minutes"]
    assert value["conservative_allocation_minutes"] > value["absolute_wall_cap_minutes"]
    assert value["conservative_units_at_2_per_hour"] > value["unit_cap"]
    assert value["status"] == "NOT_DISPATCHABLE"
    assert value["reserve_units"] == 3.2


def test_readiness_never_lets_schema_validation_override_forecast(tmp_path: Path) -> None:
    export_json(tmp_path / "bank.json", {"counts": {"frozen": 64, "mixed": 64, "scripted": 32}})
    export_json(
        tmp_path / "audit.json", {"status": "CLEAR", "bank_sha256": digest(tmp_path / "bank.json")}
    )
    plan = {
        "schema": "bl-served-search-plan/v1",
        "dispatchable": False,
        "input_hashes": {
            "bank.json": digest(tmp_path / "bank.json"),
            "audit.json": digest(tmp_path / "audit.json"),
        },
        "forecast": forecast(),
    }
    with pytest.raises(ValueError, match="NOT_DISPATCHABLE"):
        check(plan, tmp_path)
    plan["dispatchable"] = True
    plan["runtime_parity_admitted"] = True
    plan["forecast"]["status"] = "FITS"
    with pytest.raises(ValueError, match="exceeds"):
        check(plan, tmp_path)


def test_portable_bank_changes_only_path_presentation(tmp_path: Path) -> None:
    allocation, output = tmp_path / "old", tmp_path / "portable"
    original = {
        "rows": [{"world_seed": 22, "plan": {"specs": [["checkpoint", "/Mac/a.pth"]]}}],
        "excluded_preflight_rows": [],
    }
    export_json(allocation / "bank.json", original)
    export_json(
        allocation / "audit.json",
        {"status": "CLEAR", "bank_sha256": digest(allocation / "bank.json")},
    )
    bank_path, audit_path = portable_allocation(allocation, output)
    value = json.loads(bank_path.read_text())
    assert value["rows"][0] == {"world_seed": 22, "plan": {"specs": [["checkpoint", "a.pth"]]}}
    audit = json.loads(audit_path.read_text())
    assert audit["bank_sha256"] == digest(bank_path)
    assert audit["allocation_bank_sha256"] == digest(allocation / "bank.json")


def test_static_closure_contains_t3_runtime_and_exports() -> None:
    paths = source_paths()
    assert "research/t3_rollout_labels_20261015/bank_engine.py" in paths
    assert "research/t3_rollout_labels_20261015/trigger.py" in paths
    assert "research/bl_ppo/exports.py" in paths
    assert "configs" in paths
    assert "research/colab_cli" in paths
