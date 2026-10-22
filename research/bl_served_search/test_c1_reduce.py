"""Untimed raw-evidence tests for the PLAN_v3 C1 reducer."""

from __future__ import annotations

import copy

import pytest

from research.bl_served_search import c1_reduce


def arm(*, death_frame: int | None = None, mass: int = 10, rescue: bool = False) -> dict:
    frames = [
        {
            "frame": frame,
            "mass": 0 if frame == death_frame else mass,
            "alive": frame != death_frame,
            "death": frame == death_frame,
            "life": 1,
        }
        for frame in range(1, c1_reduce.HORIZON + 1)
    ]
    decisions = [{"frame": 1, "life": 1, "length": 12, "bare": 0, "chosen": 3}]
    if death_frame is not None and death_frame != 1:
        decisions.append({"frame": death_frame, "life": 1, "length": 12, "bare": 0, "chosen": 3})
    batches = []
    if rescue:
        batches.append(
            {
                "frame": 1,
                "life": 1,
                "bare": 0,
                "chosen": 3,
                "outcomes": {
                    "0": {"survives": False, "frames_executed": 5},
                    "3": {"survives": True, "frames_executed": 300},
                },
            }
        )
    return {"frames": frames, "decisions": decisions, "batches": batches}


def units(
    *, bare_deaths: int = 0, dense_deaths: int = 0, dense_mass: int = 10, rescues: int = 0
) -> list[dict]:
    rows = []
    for index in range(24):
        rows.append(
            {
                "unit_id": f"session-{index:02d}",
                "namespace": "c1-fresh",
                "seed": index + 1,
                "horizon": c1_reduce.HORIZON,
                "arms": {
                    "bare": arm(death_frame=100 if index < bare_deaths else None),
                    "dense": arm(
                        death_frame=100 if index < dense_deaths else None,
                        mass=dense_mass,
                        rescue=index < rescues,
                    ),
                },
            }
        )
    return rows


def test_summarize_arm_reports_raw_endpoint_identifiers_and_own_denominator() -> None:
    summary = c1_reduce.summarize_arm(**arm(death_frame=100, rescue=True))
    assert summary["deaths_per_1000_decisions"] == 500
    assert summary["death_frames"] == [100]
    assert summary["death_rows"] == [{"frame": 100, "life": 1}]
    assert summary["distinct_rescue_witness_lives"] == 1
    assert summary["same_life_redeaths_within_300"] == 1
    assert summary["boost_share"] == 1
    assert summary["length_quantiles"]["p50"] == 12
    assert summary["interventions"][0]["chosen_outcome"]["survives"]


def test_summarize_arm_retains_negative_local_h300_witnesses() -> None:
    value = arm()
    value["batches"] = [
        {
            "frame": 1,
            "life": 1,
            "bare": 0,
            "chosen": 3,
            "outcomes": {
                "0": {"survives": True, "frames_executed": 300},
                "3": {"survives": False, "frames_executed": 5},
            },
        }
    ]
    summary = c1_reduce.summarize_arm(**value)
    assert summary["newly_caused_death_witnesses_h300"] == [
        {"life": 1, "frame": 1, "bare": 0, "chosen": 3}
    ]


def test_reduce_go_and_reports_all_paired_endpoints() -> None:
    result = c1_reduce.reduce_sessions(
        units(bare_deaths=24, dense_deaths=0, dense_mass=10, rescues=4)
    )
    assert result["verdict"] == "GO"
    assert result["death_rate_delta_per_1000"]["ub90"] < 0
    assert result["witness_sessions"] == 4
    assert result["sessions"][0]["paired_frame_death_disclosure"].startswith("descriptive")


def test_mass_stop_precedes_low_baseline_death_insufficiency() -> None:
    result = c1_reduce.reduce_sessions(units(dense_mass=9))
    assert result["verdict"] == "STOP"


def test_low_baseline_deaths_is_insufficient_before_death_rule() -> None:
    result = c1_reduce.reduce_sessions(units(bare_deaths=7, dense_deaths=7))
    assert result["verdict"] == "INSUFFICIENT"


def test_nonbeneficial_death_bound_stops_with_sufficient_baseline_deaths() -> None:
    result = c1_reduce.reduce_sessions(units(bare_deaths=24, dense_deaths=24))
    assert result["baseline_deaths"] == 24
    assert result["death_rate_delta_per_1000"]["lb90"] == 0
    assert result["verdict"] == "STOP"


def test_missing_witness_clause_is_inconclusive_not_go() -> None:
    result = c1_reduce.reduce_sessions(units(bare_deaths=24, dense_deaths=12))
    assert result["death_rate_delta_per_1000"]["ub90"] < 0
    assert result["witness_sessions"] == 0
    assert result["verdict"] == "INCONCLUSIVE"


def test_reducer_refuses_incomplete_duplicate_and_foreign_raw_evidence() -> None:
    value = units()
    with pytest.raises(ValueError, match="exactly 24"):
        c1_reduce.reduce_sessions(value[:-1])
    duplicate = copy.deepcopy(value)
    duplicate[-1]["unit_id"] = "session-00"
    with pytest.raises(ValueError, match="duplicate"):
        c1_reduce.reduce_sessions(duplicate)
    malformed = copy.deepcopy(value)
    malformed[0]["arms"]["dense"]["frames"].pop()
    with pytest.raises(ValueError, match="14400"):
        c1_reduce.reduce_sessions(malformed)
    mismatch = copy.deepcopy(value)
    mismatch[0]["arms"]["dense"]["batches"] = [
        {
            "frame": 1,
            "life": 1,
            "bare": 1,
            "chosen": 3,
            "outcomes": {
                "1": {"survives": False, "frames_executed": 2},
                "3": {"survives": True, "frames_executed": 2},
            },
        }
    ]
    with pytest.raises(ValueError, match="does not match"):
        c1_reduce.reduce_sessions(mismatch)


def test_uniform_exact_mass_boundary_does_not_gain_roundoff_regression() -> None:
    bounds = c1_reduce._bound([-0.05] * 24)
    assert bounds["mean"] == bounds["lb90"] == bounds["ub90"] == -0.05
    assert bounds["se"] == 0
