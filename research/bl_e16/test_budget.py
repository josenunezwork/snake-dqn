import pytest

from research.bl_e16.budget import validate_reservation


def test_total_study_cannot_multiply_twenty_unit_cap_per_arm() -> None:
    jobs = [
        {
            "lane": lane,
            "job_id": lane,
            "unit_cap": 4.0,
            "timeout_minutes": 120,
            "estimate_minutes": 60,
            "rate_units_per_hour": 2.0,
            "includes_setup_export": True,
        }
        for lane in ("A", "G", "evaluation")
    ]
    value = {
        "schema": "bl-e16-budget/v1",
        "unit_cap": 20,
        "allocation_hour_cap": 10,
        "reserve_units": 4,
        "jobs": jobs,
    }
    assert set(validate_reservation(value, ["A", "G"])) == {"A", "G", "evaluation"}
    for job in jobs:
        job["unit_cap"] = 20
    with pytest.raises(ValueError, match="aggregate"):
        validate_reservation(value, ["A", "G"])
