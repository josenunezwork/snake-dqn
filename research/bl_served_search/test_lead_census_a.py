"""Targeted M0 shard extraction controls; no production replays."""

import copy

import pytest

from research.bl_served_search.lead_census_a import rescue_record, tables


def fixture_row() -> dict:
    return {
        "world_id": "fixture",
        "world_seed": 1,
        "mix": "frozen",
        "horizon": 5000,
        "plan": {},
        "baseline": {"death_length": 800, "death_cause": 4, "death_frame": 20},
        "trajectories": {"dense_alive": [True] * 11, "dense_mass": [750] * 11},
        "dense": {
            "batch_diagnostics": [
                {
                    "decision": 10,
                    "bare": 1,
                    "chosen": 4,
                    "branches_evaluated": 3,
                    "branches_skipped": 0,
                    "evaluation_order": [1, 2, 4],
                    "outcomes": {
                        "1": {"survives": False, "death_decision": 20, "frames_executed": 11},
                        "2": {"survives": False},
                        "4": {"survives": True},
                    },
                }
            ]
        },
    }


def test_extraction_and_rank() -> None:
    record = rescue_record(fixture_row())
    assert (record["f"], record["D"], record["surviving_alternative_q_rank"]) == (10, 10, 2)
    assert record["hero_length_at_f"] == 750
    assert record["baseline_death_length"] == 800
    assert record["a"] is None


def test_all_die_is_not_a_survivor() -> None:
    row = fixture_row()
    row["dense"]["batch_diagnostics"][0]["outcomes"]["4"]["survives"] = False
    record = rescue_record(row)
    assert record["all_die_delay_ranked_at_f"]
    assert record["surviving_alternative_q_rank"] is None
    assert tables([record])["frozen+mixed"]["all_die_at_f"] == 1


def test_bad_death_coordinate_rejected() -> None:
    row = fixture_row()
    row["dense"]["batch_diagnostics"][0]["outcomes"]["1"]["frames_survived"] = 11
    with pytest.raises(ValueError, match="coordinate"):
        rescue_record(row)


def test_duplicate_batch_rejected() -> None:
    row = fixture_row()
    row["dense"]["batch_diagnostics"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        rescue_record(row)


def test_pre_f_delay_counter() -> None:
    row = fixture_row()
    before = copy.deepcopy(row["dense"]["batch_diagnostics"][0])
    before["decision"] = 5
    before["outcomes"]["1"]["survives"] = True
    row["dense"]["batch_diagnostics"].insert(0, before)
    assert rescue_record(row)["delay_only_changes_before_f"] == 1
