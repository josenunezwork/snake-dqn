"""Mechanical cap repair preserves scientific and operational scope."""

import copy

import pytest

from tools.ops.f5_unit_guard import repair_guard, require_admissible


def template() -> dict:
    return {
        "script": "research/bl_residual/f5_colab.sh",
        "budget": {"unit_cap": 9999.0, "reserve_units": 0, "other_assignments": {"x": 1.54}},
        "f5": {
            "technical_budget": "historical placeholder",
            "technical_guards": {
                "base_unit_cap": 4999.5,
                "effective_unit_cap": 9999.0,
                "base_reserve_units": 0,
                "effective_reserve_units": 0,
                "units_are_scientific_budget_gate": False,
            },
        },
        "timeout_minutes": 30,
        "inputs": [{"sha256": "a" * 64}],
        "timing_calibration": {"no_job_retry": True},
    }


def test_only_guard_changes() -> None:
    original = template()
    before = copy.deepcopy(original)
    result = repair_guard(original)
    assert original == before
    assert result["budget"]["unit_cap"] == 100
    assert result["f5"]["technical_guards"]["base_unit_cap"] * 2 == 100
    result["budget"]["unit_cap"] = original["budget"]["unit_cap"]
    result["f5"] = original["f5"]
    assert result == original


@pytest.mark.parametrize("balance", [0, float("nan"), float("inf"), True])
def test_strict_admission_refuses(balance: float) -> None:
    with pytest.raises(ValueError):
        require_admissible(balance, 5)


def test_admits_generous_guard() -> None:
    require_admissible(1065.08, 0)
    require_admissible(105.001, 5)


@pytest.mark.parametrize("field,value", [("unit_cap", 10), ("reserve_units", 1)])
def test_legacy_threshold_values_do_not_refuse(field: str, value: int) -> None:
    value_manifest = template()
    value_manifest["budget"][field] = value
    assert repair_guard(value_manifest)["budget"]["unit_cap"] == 100


def test_refuses_scientific_cap() -> None:
    value = template()
    value["f5"]["technical_guards"]["units_are_scientific_budget_gate"] = True
    with pytest.raises(ValueError):
        repair_guard(value)


@pytest.mark.parametrize("balance", [0.01, 100, 105])
def test_legacy_f5_cap_only_warns(balance, capsys):
    require_admissible(balance, 5)
    assert "WARNING" in capsys.readouterr().err
