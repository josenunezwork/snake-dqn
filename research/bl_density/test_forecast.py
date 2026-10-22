"""Forecast arithmetic is descriptive, bounded, and requires real timing fields."""

import pytest

from research.bl_density.forecast import forecast


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
    assert value["world_decision_upper_bound"] == 878400
    assert value["max_observed_cpu_seconds_per_world_decision"] == 0.001
    assert [row["cloud_cpu_slowdown_assumption"] for row in value["sensitivity"]] == [2, 3, 5]
    assert value["sensitivity"][1]["session_minutes"] == pytest.approx(22.38)
    assert value["whole_session_wall_cap_minutes"] == 90


def test_forecast_refuses_fewer_than_three_timed_continuations() -> None:
    with pytest.raises(ValueError, match="three"):
        forecast([])
