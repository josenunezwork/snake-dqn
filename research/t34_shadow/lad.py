"""Budgeted LAD bisection with local, stride-one refinement.

The survival predicate need not be monotone. This algorithm implements the
registered heuristic, not an exhaustive proof of the globally last escape.
"""

from __future__ import annotations

from typing import Any, Callable


def _result(first: int, doom: int, tested: dict[int, dict[str, Any]]) -> dict[str, Any]:
    positive = [t for t, value in tested.items() if value["surviving_alternatives"]]
    best = max(positive) if positive else None
    value = tested[best] if best is not None else None
    return {
        "t_star": best,
        "a_star": value["surviving_alternatives"][0] if value else None,
        "F": value["surviving_alternatives"] if value else [],
        "per_action": value["outcomes"] if value else {},
        "first": first,
        "doom": doom,
        "expansion_count": len(tested),
        "probes": [{"decision": t, **value} for t, value in sorted(tested.items())],
    }


def _probe(t: int, tested: dict[int, dict[str, Any]], expand: Callable) -> bool:
    if t not in tested:
        value = expand(t)
        alternatives = [int(a) for a in value["outcomes"] if int(a) != int(value["greedy"])]
        ordered = sorted(alternatives, key=lambda a: (-float(value["q"][a]), a))
        survivors = [a for a in ordered if value["outcomes"][str(a)]["survives"]]
        tested[t] = {**value, "surviving_alternatives": survivors}
    return bool(tested[t]["surviving_alternatives"])


def search_lad_teacher(
    first: int,
    doom: int,
    expand: Callable[[int], dict[str, Any]],
) -> dict[str, Any]:
    """Amendment 1: stride ten, then +/-5 around every sampled positive-to-negative edge.

    A transition lies between adjacent sampled endpoints. The refinement centre
    is their midpoint, so +/-5 covers the entire ten-frame gap. Isolated
    positive islands between two negative samples can still be missed.
    """
    if first < 0 or doom < first:
        raise ValueError("invalid LAD interval")
    tested: dict[int, dict[str, Any]] = {}
    coarse = sorted(set([*range(first, doom + 1, 10), doom]))
    flags = {t: _probe(t, tested, expand) for t in coarse}
    transitions = []
    for left, right in zip(coarse, coarse[1:]):
        if flags[left] and not flags[right]:
            centre = (left + right) // 2
            transitions.append({"left": left, "right": right, "centre": centre})
            for t in range(max(first, centre - 5), min(doom, centre + 5) + 1):
                _probe(t, tested, expand)
    return {
        **_result(first, doom, tested),
        "rule": "teacher-stride10-transition-refine5",
        "transitions": transitions,
        "global_last_avoidable_proven": False,
    }


def search_lad_truth(
    first: int,
    doom: int,
    expand: Callable[[int], dict[str, Any]],
) -> dict[str, Any]:
    """Exact stride-one latest surviving decision, scanning backward to avoid unused work."""
    if first < 0 or doom < first:
        raise ValueError("invalid LAD interval")
    tested: dict[int, dict[str, Any]] = {}
    for t in range(doom, first - 1, -1):
        if _probe(t, tested, expand):
            break
    return {
        **_result(first, doom, tested),
        "rule": "stride1-backward-exact-LAD",
        "global_last_avoidable_proven": True,
    }


def search_lad(
    first: int,
    doom: int,
    expand: Callable[[int], dict[str, Any]],
    *,
    refinement_radius: int = 3,
) -> dict[str, Any]:
    """Bisect branch points, then inspect every decision near that boundary.

    ``expand`` returns greedy, q, and per-action outcomes. All alternatives
    must be in descending Q order (lowest action breaks exact Q ties).
    Results retain every probe so missed nonmonotone islands can be audited.
    """
    if first < 0 or doom < first or refinement_radius < 0:
        raise ValueError("invalid LAD interval or refinement radius")
    tested: dict[int, dict[str, Any]] = {}

    def probe(t: int) -> bool:
        return _probe(t, tested, expand)

    lo, hi = first, doom
    probe(first)
    probe(doom)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if probe(mid):
            lo = mid
        else:
            hi = mid - 1
    for t in range(max(first, lo - refinement_radius), min(doom, lo + refinement_radius) + 1):
        probe(t)
    return {
        **_result(first, doom, tested),
        "boundary": lo,
        "refinement_radius": refinement_radius,
        "expansion_count": len(tested),
        "probes": [{"decision": t, **value} for t, value in sorted(tested.items())],
        "global_last_avoidable_proven": False,
    }
