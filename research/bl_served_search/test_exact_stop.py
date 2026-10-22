"""Synthetic proof contracts for the prospective exact-stop implementation."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from research.bl_served_search import ceiling
from research.bl_served_search.exact_stop import NOT_EVALUATED, evaluation_order, run
from research.bl_served_search.ruler_operator import ExactT3Operator


def _outcome(survives: bool, frames: int) -> dict[str, object]:
    return {"survives": survives, "frames_executed": frames}


def test_exact_stop_is_decision_identical_to_full_choose_exhaustively() -> None:
    """Cover every survivor pattern, Q tie ordering, and all-die delay choice."""
    legal = [0, 1, 2, 3]
    q_cases = (
        np.array([4.0, 3.0, 2.0, 1.0, 0.0, 0.0]),
        np.array([4.0, 4.0, 3.0, 3.0, 0.0, 0.0]),
        np.array([1.0, 4.0, 4.0, 4.0, 0.0, 0.0]),
    )
    for bare in legal:
        for q in q_cases:
            for survivors in itertools.product((False, True), repeat=len(legal)):
                for delays in itertools.product((1, 150, 300), repeat=len(legal)):
                    full = {
                        action: _outcome(survives, frames)
                        for action, survives, frames in zip(legal, survivors, delays)
                    }
                    actual = run(legal, bare, q, lambda action: full[action], ceiling._choose)
                    assert actual["chosen"] == ceiling._choose(full, bare, q)
                    assert actual["branches_evaluated"] + actual["branches_skipped"] == len(legal)
                    for action in legal:
                        if action not in actual["evaluation_order"]:
                            assert actual["outcomes"][action] == {"status": NOT_EVALUATED}


def test_exact_stop_is_bare_first_then_q_descending_index_ascending() -> None:
    q = np.array([2.0, 7.0, 7.0, 5.0, 0.0, 0.0])
    assert evaluation_order([0, 1, 2, 3], 0, q) == [0, 1, 2, 3]
    assert evaluation_order([0, 1, 2, 3], 3, q) == [3, 1, 2, 0]


def test_exact_stop_reports_not_evaluated_only_after_a_survivor() -> None:
    q = np.array([5.0, 4.0, 3.0, 2.0, 0.0, 0.0])
    outcomes = {
        0: _outcome(False, 90),
        1: _outcome(True, 300),
        2: _outcome(True, 300),
        3: _outcome(False, 299),
    }
    actual = run([0, 1, 2, 3], 0, q, lambda action: outcomes[action], ceiling._choose)
    assert actual["chosen"] == 1
    assert actual["evaluation_order"] == [0, 1]
    assert (actual["branches_evaluated"], actual["branches_skipped"]) == (2, 2)
    assert actual["outcomes"][2] == {"status": NOT_EVALUATED}
    assert actual["outcomes"][3] == {"status": NOT_EVALUATED}


def test_all_die_preserves_full_choose_and_reports_no_skips() -> None:
    q = np.array([1.0, 4.0, 4.0, 3.0, 0.0, 0.0])
    outcomes = {
        0: _outcome(False, 299),
        1: _outcome(False, 120),
        2: _outcome(False, 299),
        3: _outcome(False, 299),
    }
    actual = run([0, 1, 2, 3], 0, q, lambda action: outcomes[action], ceiling._choose)
    assert actual["chosen"] == ceiling._choose(outcomes, 0, q) == 0
    assert (actual["branches_evaluated"], actual["branches_skipped"]) == (4, 0)


def test_operator_requires_registered_native_and_search_modes() -> None:
    assert (
        ExactT3Operator(native_mode="kernel", search_mode="exact_stop").search_mode == "exact_stop"
    )
    with pytest.raises(ValueError, match="native mode"):
        ExactT3Operator(native_mode="unknown")
    with pytest.raises(ValueError, match="search mode"):
        ExactT3Operator(search_mode="unknown")
