"""Offline envelope and dispatch-refusal checks; no CLI provider operations."""

import json
from pathlib import Path

import pytest

from research.bl_served_search.exports import digest, export_json
from research.bl_served_search.manifests import forecast, portable_allocation, source_paths
from research.bl_served_search.readiness import check


def test_forecast_does_not_admit_unmeasured_historical_sensitivity() -> None:
    value = forecast()
    assert value["historical_scripted_unchunked_minutes"] > 480
    assert value["two_times_conservative_minutes"] < 384
    assert value["status"] == "NOT_DISPATCHABLE"
    assert value["unit_cap"] == 64
    assert value["reserve_units"] == 12.8


def test_corrected_rate_bar_and_aggregate_envelope() -> None:
    low, boundary = forecast(2.179), forecast(2.18)
    assert low["status"] == "NOT_DISPATCHABLE"
    assert boundary["status"] == "FITS"
    assert boundary["total_two_times_minutes"] <= 1536
    assert boundary["two_times_conservative_minutes"] <= 384 <= 480
    assert boundary["conservative_allocation_minutes"] == pytest.approx(191.7584097859)
    for rate in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            forecast(rate)


def test_readiness_never_lets_schema_validation_override_forecast(tmp_path: Path) -> None:
    from research.bl_served_search.ceiling import validate_bank
    from research.bl_served_search.chunks import chunk_rows

    source = Path(__file__).parent / "BANK_L2.json"
    bank = json.loads(source.read_text())
    export_json(tmp_path / "bank.json", bank)
    export_json(
        tmp_path / "audit.json", {"status": "CLEAR", "bank_sha256": digest(tmp_path / "bank.json")}
    )
    plan = {
        "schema": "bl-served-search-plan/v2",
        "dispatchable": False,
        "chunk_index": 0,
        "chunk_count": 4,
        "world_ids": [row["world_id"] for row in chunk_rows(validate_bank(bank), 0)],
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
    plan["forecast"] = forecast(2.18)
    with pytest.raises(ValueError, match="separate Mac and L4"):
        check(plan, tmp_path)
    plan["ruler_admitted"] = True
    with pytest.raises(ValueError, match="hash-bound"):
        check(plan, tmp_path)
    plan["forecast"]["conservative_allocation_minutes"] = 241
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
