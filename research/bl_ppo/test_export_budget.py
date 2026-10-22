"""Arithmetic-only full-size retention and manifest admission checks."""

import copy

import pytest

from research.bl_ppo.export_budget import (
    freeze_control_budget,
    validate_control_budget,
    validate_manifest_budget,
)


def control() -> dict:
    return {
        "worlds": 128,
        "processes": 12,
        "rollout_steps": 128,
        "target": 100_000_000,
        "checkpoint_every_seconds": 900,
    }


def test_full_y3_retention_is_declared_before_data_without_allocating_banks() -> None:
    value = control()
    budget = freeze_control_budget(value, timeout_minutes=300, max_export_bytes=2 * 1024**3)
    assert budget["max_periodic_snapshots"] == 20
    assert budget["retained_closure_allowance"] == 26
    assert budget["max_total_export_bytes"] == 2 * budget["base_total_export_bytes"]
    assert budget["derived_download_closure_bytes"] <= 8 * 1024**3
    assert budget["download_closure_cap_bytes"] == 2 * budget["download_closure_base_cap_bytes"]
    assert budget["max_total_export_bytes"] > budget["max_export_bytes"]
    assert validate_control_budget(value) == budget
    manifest = {
        "bl_p1": {"control": value},
        "timeout_minutes": 300,
        "max_export_bytes": 2 * 1024**3,
        "max_total_export_bytes": budget["max_total_export_bytes"],
    }
    validate_manifest_budget(manifest)
    manifest["max_total_export_bytes"] -= 1
    with pytest.raises(ValueError, match="allowances differ"):
        validate_manifest_budget(manifest)


@pytest.mark.parametrize(
    "field", ["snapshot_byte_bounds", "export_byte_budget", "max_snapshot_bytes"]
)
def test_forged_nested_allowances_are_refused(field: str) -> None:
    value = control()
    freeze_control_budget(value, timeout_minutes=300, max_export_bytes=2 * 1024**3)
    forged = copy.deepcopy(value)
    if field == "max_snapshot_bytes":
        forged[field] -= 1
    elif field == "snapshot_byte_bounds":
        forged[field]["components"]["ordinary_bank"]["effective_bytes"] -= 1
    else:
        forged[field]["max_total_export_bytes"] -= 1
    with pytest.raises(ValueError, match="arithmetic differs"):
        validate_control_budget(forged)


def test_insufficient_per_file_cap_is_rejected_before_collection() -> None:
    with pytest.raises(ValueError, match="exceeds manifest"):
        freeze_control_budget(control(), timeout_minutes=300, max_export_bytes=160 * 1024**2)
