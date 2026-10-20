"""Budgeted LAD bisection with local, stride-one refinement.

The survival predicate need not be monotone. This algorithm implements the
registered heuristic, not an exhaustive proof of the globally last escape.
"""

from __future__ import annotations

from typing import Any, Callable


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
        if t not in tested:
            value = expand(t)
            outcomes = value["outcomes"]
            alternatives = [int(a) for a in outcomes if int(a) != int(value["greedy"])]
            ordered = sorted(alternatives, key=lambda a: (-float(value["q"][a]), a))
            survivors = [a for a in ordered if outcomes[str(a)]["survives"]]
            tested[t] = {**value, "surviving_alternatives": survivors}
        return bool(tested[t]["surviving_alternatives"])

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
        "boundary": lo,
        "refinement_radius": refinement_radius,
        "expansion_count": len(tested),
        "probes": [{"decision": t, **value} for t, value in sorted(tested.items())],
        "global_last_avoidable_proven": False,
    }
