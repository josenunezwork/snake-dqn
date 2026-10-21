"""Synthetic admission checks; these tests play no registered worlds."""

from pathlib import Path

import pytest

from research.bl_e16.admission import (
    AdmissionRefused,
    arm_preconditions,
    forecast,
    require_execution_plan,
)
from research.bl_ppo.exports import export_json


def test_frozen_cost_does_not_omit_geometry_refit_or_shrink_bank() -> None:
    priced = forecast(2000.0, 800.0)
    assert priced["training_transitions"] == 12_000_000
    assert priced["evaluation_world_steps"] == 5_760_000
    assert priced["allocation_hours"] == pytest.approx(5 + 2 / 3)
    assert priced["units_at_2_per_hour"] == pytest.approx(11 + 1 / 3)
    assert priced["timeout_minutes"] == 680
    assert priced["status"] == "NOT_DISPATCHABLE"
    assert forecast(4000.0, 2000.0)["status"] == "FORECAST_FITS"
    assert forecast(1999.0, 4000.0, 2)["status"] == "NOT_DISPATCHABLE"
    assert forecast(4000.0, 799.0)["status"] == "NOT_DISPATCHABLE"


def test_stub_preconditions_cannot_become_available_from_a_status_string() -> None:
    facts = {
        "E2_F_nominated_before_efficacy": True,
        "E8_ready": True,
        "E12_parent": "PASS",
        "E3": "GO",
        "E14": "PASS",
        "E14_architecture": "e3-seven-planes",
        "G_architecture": "e3-seven-planes",
        "U_upper90": 39.0,
        "E7": "TRANSFER_ADMISSIBLE",
        "productive_coverage": 0.2,
        "tau_requirement": 0.9,
        "E1": "CONTINUE",
        "PLR_protocol_frozen": True,
    }
    assert not arm_preconditions("A", facts)
    assert not arm_preconditions("G", facts)
    assert any("UB90" in x for x in arm_preconditions("R", facts))
    for arm in ("R", "S", "P"):
        assert any("stub" in x for x in arm_preconditions(arm, facts))
    facts["E14_architecture"] = "wrong-architecture"
    assert arm_preconditions("G", facts)


def test_nonadmitted_or_partial_plan_refuses_before_world_loading(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    export_json(path, {"schema": "bl-e16-plan/v1", "status": "NOT_DISPATCHABLE"})
    with pytest.raises(AdmissionRefused, match="not admitted"):
        require_execution_plan(path, "G")
    path.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        require_execution_plan(path, "A")
