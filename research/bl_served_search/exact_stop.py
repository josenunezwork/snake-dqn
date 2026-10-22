"""Decision-identical, branch-by-branch early stopping for the frozen T3 rule.

The T3 scorer gives every H300 survivor its maximum score and prefers the
bare action among ties.  That makes the evaluation order below safe: evaluate
bare first, then alternatives in T3's Q/index order, and stop at the first
survivor.  All-die batches remain fully evaluated and use ``ceiling._choose``
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

NOT_EVALUATED = "NOT_EVALUATED"


def evaluation_order(legal: Sequence[int], bare: int, q: np.ndarray) -> list[int]:
    """Return bare-first then descending-Q/ascending-index remaining actions."""
    actions = [int(action) for action in legal]
    if not actions or len(set(actions)) != len(actions) or bare not in actions:
        raise ValueError("exact stop requires unique legal actions containing bare")
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("exact stop requires finite six-action Q values")
    return [
        bare,
        *sorted((action for action in actions if action != bare), key=lambda a: (-float(q[a]), a)),
    ]


def run(
    legal: Sequence[int],
    bare: int,
    q: np.ndarray,
    evaluate: Callable[[int], dict[str, Any]],
    choose: Callable[[dict[int, dict[str, Any]], int, np.ndarray], int],
) -> dict[str, Any]:
    """Evaluate exact singleton branches until T3's chosen action is fixed.

    ``evaluate`` must perform a complete H300 singleton continuation.  It is
    deliberately injected so this module cannot grow a second simulator path.
    """
    order = evaluation_order(legal, bare, q)
    outcomes: dict[int, dict[str, Any]] = {}
    chosen: int | None = None
    for action in order:
        outcome = dict(evaluate(action))
        outcomes[action] = outcome
        if bool(outcome["survives"]):
            # Bare is preferred over every survivor; otherwise the first
            # survivor has the highest Q (then lowest index) by ``order``.
            chosen = action
            break
    if chosen is None:
        chosen = int(choose(outcomes, bare, q))
    evaluated_order = list(outcomes)
    skipped = [action for action in order if action not in outcomes]
    reported = {action: outcomes.get(action, {"status": NOT_EVALUATED}) for action in sorted(order)}
    return {
        "chosen": chosen,
        "outcomes": reported,
        "evaluation_order": evaluated_order,
        "branches_evaluated": len(outcomes),
        "branches_skipped": len(skipped),
    }
