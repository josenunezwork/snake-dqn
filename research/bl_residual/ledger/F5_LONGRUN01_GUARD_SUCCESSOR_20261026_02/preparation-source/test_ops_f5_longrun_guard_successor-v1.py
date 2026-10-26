"""Mechanical successor must retain every non-operational manifest leaf."""

import copy

import pytest

from tools.ops.f5_longrun_guard_successor import differences, successor


def test_successor_changes_only_seven_mechanical_leaves() -> None:
    old = {
        "job_id": "initial-01",
        "budget": {"unit_cap": 100000.0, "reserve_units": 0, "rate_units_per_hour": 2},
        "output_dir": "/old/output-01",
        "supervisor_state_dir": "/old/state-01",
        "operational_admission": {"scientific_fits_authorized": False},
        "inputs": [{"dest": "control.json", "sha256": "a" * 64}],
        "timeout_minutes": 240,
        "resume": {"max_outcome_blind_retries": 2, "skip_verified": True},
    }
    before = copy.deepcopy(old)
    new = successor(old, {"path": "/new/account.json", "sha256": "b" * 64, "size": 42})
    assert old == before
    assert len(differences(old, new)) == 7
    assert new["budget"] == {"unit_cap": 100.0, "reserve_units": 0, "rate_units_per_hour": 2}
    new["inputs"][0]["dest"] = "mutated"
    assert old["inputs"] == before["inputs"]


def test_diff_descends_every_list_entry_and_added_dictionary_leaf() -> None:
    rows = differences({"a": [{"b": 1}, {"b": 2}]}, {"a": [{"b": 1}, {"b": 3}], "c": {"d": 4}})
    assert rows == [
        {"pointer": "/a/1/b", "before": 2, "after": 3},
        {"pointer": "/c/d", "before": None, "after": 4},
    ]


def test_successor_refuses_a_different_predecessor_guard() -> None:
    with pytest.raises(AssertionError):
        successor({"budget": {"unit_cap": 9999}}, {})
