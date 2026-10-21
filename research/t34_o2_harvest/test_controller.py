"""Focused O2 shared-tree controller contracts; no simulator episodes."""

from __future__ import annotations

from types import SimpleNamespace

from research.t34_o2_harvest.controller import (
    HarvestScore,
    OpportunitySchedule,
    choose_d_plus_o,
    choose_shared_tree,
)


def _search(*, bare_survives: bool, action: int = 1):
    return SimpleNamespace(
        status="NO_IMPROVEMENT",
        bare_action=1,
        action=action,
        outcomes={1: SimpleNamespace(survives=bare_survives), 2: SimpleNamespace(survives=True)},
    )


def test_harvest_uses_the_top_surviving_branch_and_never_a_lower_fallback() -> None:
    # Action 2 has the highest mass score but fails attribution.  A lower
    # qualifying candidate would be post-hoc cherry-picking and is forbidden.
    result = choose_d_plus_o(
        _search(bare_survives=True),
        [HarvestScore(2, 3000, 1, True), HarvestScore(1, 0, 0, True)],
    )
    assert result.action == 1
    assert result.kind == "bare"


def test_rescue_has_priority_over_harvest() -> None:
    search = _search(bare_survives=False, action=2)
    search.status = "IMPROVING"
    result = choose_d_plus_o(search, [HarvestScore(2, 5000, 5000, True)])
    assert result.action == 2
    assert result.kind == "rescue"


def test_dead_harvest_branch_is_never_eligible() -> None:
    assert not HarvestScore(2, 5000, 5000, False).eligible()


def test_shared_tree_imports_retained_rescue_ranking() -> None:
    outcomes = {
        "1": {"tested": True, "survives": False, "alive_steps": 10, "mass": 10},
        "2": {"tested": True, "survives": True, "alive_steps": 500, "mass": 500},
    }
    d, do = choose_shared_tree(
        outcomes,
        q=[0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
        bare_action=1,
        scores=[HarvestScore(1, 0, 0, False), HarvestScore(2, 5000, 5000, True)],
    )
    assert (d.action, d.kind, do.action, do.kind) == (2, "rescue", 2, "rescue")


def test_schedule_is_time_stratified_then_event_bounded() -> None:
    schedule = OpportunitySchedule()
    assert schedule.due(1000, 0)
    assert not schedule.due(1010, 0)
    assert schedule.due(1010, 0, nearby_death=True, last_event=0)
    assert not schedule.due(1020, 1, nearby_death=True, last_event=1010)
