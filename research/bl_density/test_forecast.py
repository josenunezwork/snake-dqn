"""Forecast arithmetic is descriptive, bounded, and requires real timing fields."""

import pytest

from research.bl_density.forecast import forecast, pilot_cpu_projection


def test_forecast_prices_full_decision_ceiling_and_cloud_sensitivities() -> None:
    records = [
        {
            "sampled": [{"prefilter": True, "h300_cpu_seconds": 0.3, "h300_frames_executed": 300}],
            "decisions": 8,
            "steady_live_step_cpu_seconds": 0.008,
            "steady_live_decisions": 8,
            "world_setup_cpu_seconds": 0.1,
        }
        for _ in range(3)
    ]
    value = forecast(records)
    assert value["world_decision_upper_bound"] == 8954400
    assert value["max_observed_cpu_seconds_per_world_decision"] == 0.001
    assert [row["cloud_cpu_slowdown_assumption"] for row in value["sensitivity"]] == [2, 3, 5]
    assert value["sensitivity"][1]["session_minutes"] == pytest.approx(60.018)
    assert value["whole_session_wall_cap_minutes"] == 180
    assert value["manifest_work_wall_seconds"] == 10800
    assert value["worst_case_units_at_session_cap"] == pytest.approx(4.62)
    assert value["pilot_timing_projection_cpu_seconds"]["total"] == pytest.approx(25497.6)
    assert value["pilot_timing_projection_cpu_hours"] == pytest.approx(7.0826666667)
    assert value["ten_worker_wall_minutes_expected_range"] == [70, 80]


def test_pilot_timing_projection_uses_only_frozen_per_episode_costs() -> None:
    projection = pilot_cpu_projection()
    assert projection == pytest.approx(
        {"A": 14498.4, "B70m": 2894.4, "B70p": 4348.8, "C-P1": 3756.0, "total": 25497.6}
    )


def test_forecast_refuses_fewer_than_three_timed_continuations() -> None:
    with pytest.raises(ValueError, match="three"):
        forecast([])


def test_alternative_cpu_is_priced_by_all_executed_branch_decisions() -> None:
    records = [
        {
            "sampled": [
                {
                    "prefilter": True,
                    "h300_cpu_seconds": 0.61,
                    "h300_frames_executed": 10,
                    "continuation_branch_decisions": 610,
                }
            ],
            "world_setup_cpu_seconds": 0,
        }
        for _ in range(3)
    ]
    value = forecast(records)
    assert value["max_observed_cpu_seconds_per_world_decision"] == 0.001
    assert value["measured_continuation_decisions"] == [610, 610, 610]
