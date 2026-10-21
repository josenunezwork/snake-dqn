"""The amended delivered-boost clause and Round 0's unchanged decision order.

Pure arithmetic only. Every clause except ``delivered_boost_share`` is taken
verbatim from the registered ``research.t34_round0.analyze.summarize`` output;
the decision order below is the registered order (faults, then point clauses or a
clear survival failure, then all clauses, else INCONCLUSIVE). Tests check this
order against the registered ``summarize`` itself on randomized inputs.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from research.t34_round0.analyze import MIXES, InvalidEvidence, _integer

CLAUSE = "delivered_boost_share"
CEILING = 0.10
# Floor = FLOOR_NUM / FLOOR_DEN x bare pooled share, compared exactly in integers.
FLOOR_NUM, FLOOR_DEN = 1, 2
# Registered clause names and point clauses (analyze.py summarize, unchanged).
REGISTERED_CLAUSES = (
    "scripted_no_death_ge_066",
    "scripted_no_death_gain_ge_010",
    "mass_gain_ge_40",
    "mass_lb90_positive",
    "survival_guard_a_pooled",
    "survival_guard_a_each_mix",
    "survival_guard_a_absolute_floor",
    "survival_guard_b_scripted",
    CLAUSE,
)
POINT_CLAUSES = (
    "scripted_no_death_ge_066",
    "scripted_no_death_gain_ge_010",
    "mass_gain_ge_40",
    "survival_guard_a_absolute_floor",
    "survival_guard_b_scripted",
    CLAUSE,
)


def pooled_boosts(records: list[Mapping[str, Any]], arm: str) -> tuple[int, int]:
    """Pooled (boost decisions, decisions) over all worlds; runner.py: action >= 3.

    One decision per delivered trace row. analyze.py already reconciled these
    counters against six-action counts and the exported trace for both arms.
    """
    boosts = sum(_integer(r["arms"][arm]["boost_decisions"], "boost decisions") for r in records)
    decisions = sum(_integer(r["arms"][arm]["decisions"], "decisions", 1) for r in records)
    return boosts, decisions


def amended_boost_clause(
    teacher_boosts: int, teacher_decisions: int, bare_boosts: int, bare_decisions: int
) -> bool:
    """0.5 x bare pooled share <= teacher pooled share <= 0.10, both inclusive.

    The floor is an exact integer cross-multiplication (no float rounding at the
    boundary); bare share 0 gives floor 0. The ceiling keeps the registered float
    expression ``boosts / decisions <= 0.10`` unchanged.
    """
    _integer(teacher_boosts, "teacher boosts")
    _integer(bare_boosts, "bare boosts")
    _integer(teacher_decisions, "teacher decisions", 1)
    _integer(bare_decisions, "bare decisions", 1)
    if teacher_boosts > teacher_decisions or bare_boosts > bare_decisions:
        raise InvalidEvidence("boost decisions exceed decisions")
    floor_ok = (
        FLOOR_DEN * teacher_boosts * bare_decisions >= FLOOR_NUM * bare_boosts * teacher_decisions
    )
    return floor_ok and teacher_boosts / teacher_decisions <= CEILING


def amend(
    registered: Mapping[str, Any],
    teacher: tuple[int, int],
    bare: tuple[int, int],
) -> dict[str, Any]:
    """Return a copy of a registered summary with only the boost clause re-evaluated."""
    clauses = registered["clauses"]
    if set(clauses) != set(REGISTERED_CLAUSES):
        raise InvalidEvidence("registered summary lacks the registered clause set")
    if registered["boost_share"] != teacher[0] / teacher[1]:
        raise InvalidEvidence("teacher pooled boost counts differ from the registered share")
    result = copy.deepcopy(dict(registered))
    amended = dict(clauses)
    amended[CLAUSE] = amended_boost_clause(teacher[0], teacher[1], bare[0], bare[1])
    result["clauses"] = amended
    if registered["unconditional_faults"]:
        # Faults precede every clause; the registered decision and reason stand.
        return result
    alive, per_mix = registered["pooled_alive"], registered["per_mix"]
    clear_survival_failure = alive["upper"] <= -0.05 or any(
        per_mix[m]["alive"]["upper"] <= -0.075 for m in MIXES
    )
    if any(not amended[c] for c in POINT_CLAUSES) or clear_survival_failure:
        decision, reason = "STOP", "this teacher/controller failed its registered bar"
    elif all(amended.values()):
        decision, reason = "GO", "all Round 0 development premise bars pass; no release authority"
    else:
        decision, reason = (
            "INCONCLUSIVE",
            "point bars pass but confidence bounds remain inconclusive",
        )
    result.update(decision=decision, reason=reason)
    return result
