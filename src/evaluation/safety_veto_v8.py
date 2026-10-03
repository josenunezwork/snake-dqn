"""Opt-in serving-time free-space veto, v8: v7's space preference, then v6's head avoidance.

This module adds a NEW veto, ``free-space-veto/v8-space-and-head(lambda=<x>)``. It changes
none of the released, retired or screened vetoes (:mod:`src.evaluation.safety_veto` v2 and
``safety_veto_v3`` to ``safety_veto_v7`` stay byte-identical; a test asserts all six). v8 is
"v7's decision at ``lambda``, then v6's layer (a)". Both parts are imported and run
unchanged:

1. **v7.** A private :class:`~src.evaluation.safety_veto_v7.SpacePreferenceVeto` at the
   same ``lambda`` makes the decision exactly as v7 does (``SpacePreferenceVeto.decide``,
   which runs v5's :func:`~src.evaluation.safety_veto_v5.boost_aware_choice` and v7's
   re-rank). Its counters (v7's probe counters and its diagnostics, with v5's nested) are
   kept beside v8's, so the v7 part of every decision is auditable on its own.
2. **Head layer** (v6 layer (a), ``docs/experiments/apex_veto_v6_2026-10-02``: head-on
   deaths 10 -> 2, +30 mass vs v5 in the scripted mix). Applies when v7's outcome is
   ``kept`` or ``vetoed`` (never on ``no_spacious``, as in v6). Reach, the head-on rule
   (mechanics v2 and ``hero_post >= 1.15 * (opp_logical + 1)``) and head risk are v6's
   :func:`~src.evaluation.safety_veto_v6.opponent_head_reach` and
   :func:`~src.evaluation.safety_veto_v6.head_risky_actions`. If v7's choice ``c`` is
   head-risky and some OTHER action is masked-legal, v5-eligible (v2-spacious; for a boost
   also v5's landing check) and not head-risky, ``c`` is replaced; otherwise ``c`` stands
   (``head_risky_kept_no_alternative``).

**Replacement** (the one new rule). Let ``R`` be the eligible non-risky alternatives in
``c``'s speed mode, else (none there) in the other mode: exactly the set v6's
:func:`~src.evaluation.safety_veto_v6.head_avoid_choice` takes its highest-Q member from
(that member is the anchor ``h``). v8 replaces ``c`` by v7's re-rank over ``R``:
:func:`~src.evaluation.safety_veto_v7.space_preference_choice` with candidates ``R``,
anchor ``h`` and the same ``lambda``, i.e. the argmax of ``Qn + lambda * g`` with ``Qn``
min-max normalized over ``R`` and ``g`` v7's capped tail-aware post-move area (v7's
``landing_area``, ``area_cap`` and ``area_score``); ties go to ``h``, then the lowest
index. So the replacement is "the best v7 score among the non-risky eligible actions, same
speed mode preferred", and at ``lambda = 0`` it is v6's highest-Q replacement exactly.

Consequences (each tested): with the head layer off (``head_avoidance=False``) v8 is v7 at
the same ``lambda``, decision for decision and counter for counter; with it on, v8 differs
from v7 only on head vetoes; v8 never returns a masked action and, outside ``no_spacious``
and v5's landing-veto same-direction replacement (both v7's own choice), never a
v5-ineligible one; every choice is deterministic (no randomness or timing).

**Cost.** The head check is v6's (an early exit when no threat cell is within Manhattan
distance 2 of the hero's head). Only on a head-risky decision does v8 recompute v2's
free-space features and v5's landing floods (v7 does not expose its own) and evaluate
areas lazily for ``R`` (at most three).

**Reference diagnostic (opt-in, never decides).** ``reference_lambda`` (default ``None``)
also runs a separate v7 at that ``lambda`` on the same inputs and counts
``action_differs_from_reference``; the v8 DEV sweep sets it to its arm A's ``lambda`` so
its activity gate reads "decisions where this arm differs from arm A". It is a diagnostic
only: it is not part of the descriptor and costs one more v7 decision per call.

Records. ``record()`` is the descriptor plus exactly v2's seven counters, so
``strict_promotion._validate_candidate_wrapper_probe`` accepts it. A head veto is a
``vetoed`` decision (also when v7 had ``kept``). The v8 counters, with v7's diagnostics
under ``"v7"`` (v5's nested in them) and v7's probe counters under ``"v7_probe_counters"``,
come from :meth:`SpaceAndHeadVeto.diagnostics_record`.

Default behavior is untouched: nothing installs this class unless a caller does so
explicitly with :func:`install_space_and_head_veto`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.evaluation.safety_veto import (
    NUM_ACTIONS,
    NUM_DIRECTIONS,
    OUTCOME_KEPT,
    OUTCOME_NO_SPACIOUS,
    OUTCOME_VETOED,
    SafetyVetoCounters,
    free_space_threshold,
    spacious_directions,
)
from src.evaluation.safety_veto_v3 import grid_for, static_blocked
from src.evaluation.safety_veto_v5 import landing_count
from src.evaluation.safety_veto_v6 import (
    HEAD_RULE,
    REASON_HEAD_KEPT,
    REASON_HEAD_VETO,
    head_avoid_choice,
    head_risky_actions,
    opponent_head_reach,
)
from src.evaluation.safety_veto_v7 import (
    SpacePreferenceVeto,
    area_cap,
    area_score,
    landing_area,
    space_preference_choice,
    validate_lambda,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.game.snake import Snake

VETO_METHOD_V8_PREFIX = "free-space-veto/v8-space-and-head"
REPLACEMENT_RULE_V8 = "v7-space-preference-then-v6-head-avoidance-with-v7-score-replacement"
HEAD_REPLACEMENT_RULE = (
    "R = masked-legal v5-eligible non-head-risky actions other than v7's choice, in its "
    "speed mode, else the other mode; argmax over R of (q-max_R q)/max(|max_R q-min_R q|,"
    "1e-6) + lambda*min(area,cap)/cap; ties -> highest-q member of R (lowest index), then "
    "lowest index"
)
REASON_V7 = "v7"


def method_for(lam: float) -> str:
    """The probe ``method`` string for ``lambda`` (``repr`` of the float, as v7)."""
    return f"{VETO_METHOD_V8_PREFIX}(lambda={validate_lambda(lam)!r})"


def head_replacement_choice(
    q_values: Sequence[float],
    eligible: Sequence[bool],
    risky: Sequence[bool],
    current: int,
    lam: float,
    score_of: Callable[[int], float],
) -> Tuple[int, str, Optional[int]]:
    """The head layer on one decision (pure given ``score_of``).

    Returns ``(action, reason, anchor)``: ``reason`` is ``"v7"`` (``current`` not risky),
    v6's ``head_kept_no_alternative`` or ``head_veto``; ``anchor`` is v6's highest-Q
    replacement (``None`` unless vetoed). ``score_of(a)`` is v7's ``g`` (called lazily,
    only for members of ``R``).
    """
    lam = validate_lambda(lam)
    anchor, reason = head_avoid_choice(q_values, eligible, risky, current)
    if reason != REASON_HEAD_VETO:
        return int(current), (REASON_HEAD_KEPT if reason == REASON_HEAD_KEPT else REASON_V7), None
    boost = anchor >= NUM_DIRECTIONS
    pool = [
        a
        for a in range(NUM_ACTIONS)
        if a != int(current) and eligible[a] and not risky[a] and (a >= NUM_DIRECTIONS) == boost
    ]
    action, _ = space_preference_choice(q_values, pool, anchor, lam, score_of)
    return int(action), REASON_HEAD_VETO, int(anchor)


@dataclass
class SpaceAndHeadCounters:
    """Per-episode v8 bookkeeping (reported beside the probe; never gates).

    Identities (``v7`` = the nested v7 diagnostics): ``decisions == kept + vetoes_applied
    + no_spacious`` (the probe's ``kept_base``/``vetoes_applied``/``fallback_no_spacious``);
    ``decisions == v7.decisions``; ``kept == v7.kept - head_vetoes_of_v7_kept``;
    ``vetoes_applied == v7.vetoes_applied + head_vetoes_of_v7_kept``;
    ``no_spacious == v7.no_spacious``; ``head_risky_decisions == head_risky_vetoes +
    head_risky_kept_no_alternative``; ``action_differs_from_v7 == head_risky_vetoes``;
    ``head_checks == 0`` when the head layer is off. ``*_seconds_*`` are wall-clock.
    """

    decisions: int = 0
    kept: int = 0
    vetoes_applied: int = 0
    no_spacious: int = 0
    head_checks: int = 0
    head_risky_decisions: int = 0
    head_risky_vetoes: int = 0
    head_risky_kept_no_alternative: int = 0
    head_risk_waived_hero_wins: int = 0
    head_vetoes_speed_switched: int = 0
    head_vetoes_of_v7_kept: int = 0
    head_vetoes_of_v7_rerank: int = 0
    head_vetoes_space_differs_from_highest_q: int = 0
    head_area_evaluations: int = 0
    action_differs_from_v7: int = 0
    reference_decisions: int = 0
    action_differs_from_reference: int = 0
    apply_seconds_total: float = 0.0
    apply_seconds_max: float = 0.0
    head_seconds_total: float = 0.0
    head_seconds_max: float = 0.0
    reference_seconds_total: float = 0.0
    reference_seconds_max: float = 0.0

    def count_outcome(self, outcome: str) -> None:
        """Count one decision's probe outcome."""
        name = {
            OUTCOME_KEPT: "kept",
            OUTCOME_VETOED: "vetoes_applied",
            OUTCOME_NO_SPACIOUS: "no_spacious",
        }.get(outcome)
        if name is None:
            raise ValueError(f"unknown veto outcome {outcome!r}")
        self.decisions += 1
        setattr(self, name, getattr(self, name) + 1)

    def add_time(self, kind: str, seconds: float) -> None:
        """Accumulate wall-clock ``seconds`` under ``apply``, ``head`` or ``reference``."""
        total, peak = f"{kind}_seconds_total", f"{kind}_seconds_max"
        setattr(self, total, getattr(self, total) + float(seconds))
        setattr(self, peak, max(getattr(self, peak), float(seconds)))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping plus the mean apply cost."""
        out: Dict[str, Any] = {
            key: float(value) if isinstance(value, float) else int(value)
            for key, value in self.__dict__.items()
        }
        out["mean_apply_seconds"] = (
            self.apply_seconds_total / self.decisions if self.decisions else None
        )
        return out


class SpaceAndHeadVeto:
    """Stateful per-snake veto hook (same surface as ``FreeSpaceVeto``).

    ``head_avoidance=False`` turns the head layer off (then v8 is v7 at ``lam``; the
    descriptor records the flag, so such a run is a different identity).
    ``reference_lambda`` adds the diagnostic-only reference v7 (module docstring).
    """

    def __init__(
        self,
        lam: float,
        head_avoidance: bool = True,
        reference_lambda: Optional[float] = None,
    ) -> None:
        self.lam = validate_lambda(lam)
        self.method = method_for(self.lam)
        self.head_avoidance = bool(head_avoidance)
        self.reference_lambda = (
            None if reference_lambda is None else validate_lambda(reference_lambda)
        )
        self.counters = SafetyVetoCounters()
        self.v8 = SpaceAndHeadCounters()
        self.v7 = SpacePreferenceVeto(self.lam)
        self._reference: Optional[SpacePreferenceVeto] = None
        if self.reference_lambda is not None and self.reference_lambda != self.lam:
            self._reference = SpacePreferenceVeto(self.reference_lambda)

    def reset(self) -> None:
        """Zero all counters (call at episode start)."""
        self.counters = SafetyVetoCounters()
        self.v8 = SpaceAndHeadCounters()
        self.v7.reset()
        if self._reference is not None:
            self._reference.reset()

    def apply(
        self,
        snake: "Snake",
        other_snakes: Sequence["Snake"],
        q_values: Any,
        action_mask: Any,
        base_action: int,
    ) -> int:
        """Return the (possibly vetoed, re-ranked or head-avoiding) action for one decision."""
        started = time.perf_counter()
        action, outcome = self.decide(
            snake,
            list(other_snakes),
            _as_list(q_values),
            [bool(x) for x in _as_list(action_mask)],
            int(base_action),
        )
        self.counters.record(int(base_action), action, outcome)
        self.v8.count_outcome(outcome)
        self.v8.add_time("apply", time.perf_counter() - started)
        return action

    def decide(
        self,
        snake: "Snake",
        others: Sequence["Snake"],
        q: Sequence[float],
        mask: Sequence[bool],
        base: int,
    ) -> Tuple[int, str]:
        """``(action, probe outcome)``; updates the v8 and nested v7 diagnostics."""
        if len(q) != NUM_ACTIONS or len(mask) != NUM_ACTIONS:
            raise ValueError(f"expected {NUM_ACTIONS} Q-values and mask entries")
        # Part 1: v7, exactly as SpacePreferenceVeto.apply (its own counters kept).
        started = time.perf_counter()
        reranks_before = self.v7.v7.rerank_changes
        v7_action, v7_outcome = self.v7.decide(snake, others, q, mask, base)
        self.v7.counters.record(base, v7_action, v7_outcome)
        self.v7.v7.count_outcome(v7_outcome)
        self.v7.v7.add_time("apply", time.perf_counter() - started)
        action, outcome = v7_action, v7_outcome
        # Part 2: v6 layer (a) on v7's choice, with the v7-score replacement.
        if self.head_avoidance and v7_outcome != OUTCOME_NO_SPACIOUS:
            head_started = time.perf_counter()
            action = self._head_layer(snake, others, q, mask, v7_action)
            if action != v7_action:
                outcome = OUTCOME_VETOED
                self.v8.head_vetoes_of_v7_kept += int(v7_outcome == OUTCOME_KEPT)
                self.v8.head_vetoes_of_v7_rerank += int(self.v7.v7.rerank_changes > reranks_before)
            self.v8.add_time("head", time.perf_counter() - head_started)
        self.v8.action_differs_from_v7 += int(action != v7_action)
        if self.reference_lambda is not None:
            ref_started = time.perf_counter()
            reference = v7_action
            if self._reference is not None:
                reference, _ = self._reference.decide(snake, others, q, mask, base)
            self.v8.reference_decisions += 1
            self.v8.action_differs_from_reference += int(action != reference)
            self.v8.add_time("reference", time.perf_counter() - ref_started)
        return action, outcome

    def _head_layer(
        self,
        snake: "Snake",
        others: Sequence["Snake"],
        q: Sequence[float],
        mask: Sequence[bool],
        current: int,
    ) -> int:
        """v6 layer (a) on ``current`` (v7's choice); replacement by v7's score over ``R``."""
        grid = grid_for(snake)
        threats = opponent_head_reach(snake, others, grid)
        self.v8.head_checks += 1
        risky, waived = head_risky_actions(snake, threats, grid)
        if waived[current]:
            self.v8.head_risk_waived_hero_wins += 1
        if not risky[current]:
            return current
        self.v8.head_risky_decisions += 1
        features = snake._get_free_space_features(list(others))
        cap, need = free_space_threshold(snake.length, snake._logical_length())
        spacious = spacious_directions(features, cap, need)
        eligible = [
            bool(mask[a])
            and bool(spacious[a % NUM_DIRECTIONS])
            and not risky[a]  # checked first: only non-risky boosts need a landing flood
            and (a < NUM_DIRECTIONS or _landing_ok(snake, others, a % NUM_DIRECTIONS, cap, need))
            for a in range(NUM_ACTIONS)
        ]
        context: Dict[str, Any] = {}

        def score_of(action: int) -> float:
            if not context:
                blocked = static_blocked(snake, others, grid)
                context.update(blocked=blocked, cap=area_cap(snake.length, grid, blocked))
            self.v8.head_area_evaluations += 1
            area = landing_area(snake, action, context["blocked"], grid, context["cap"])
            return area_score(area, context["cap"])

        action, reason, anchor = head_replacement_choice(
            q, eligible, risky, current, self.lam, score_of
        )
        if reason != REASON_HEAD_VETO:
            self.v8.head_risky_kept_no_alternative += 1
            return current
        self.v8.head_risky_vetoes += 1
        self.v8.head_vetoes_speed_switched += int(
            (action >= NUM_DIRECTIONS) != (current >= NUM_DIRECTIONS)
        )
        self.v8.head_vetoes_space_differs_from_highest_q += int(action != anchor)
        return action

    def diagnostics_record(self) -> Dict[str, Any]:
        """v8 counters plus v7's diagnostics (``"v7"``) and probe counters."""
        out = self.v8.to_dict()
        out["head_avoidance"] = self.head_avoidance
        out["reference_lambda"] = self.reference_lambda
        out["v7"] = self.v7.diagnostics_record()
        out["v7_probe_counters"] = self.v7.counters.to_dict()
        return out

    def descriptor(self) -> Dict[str, Any]:
        """Static identity of this veto (no counters): v7's at ``lambda`` plus the head layer."""
        return {
            **self.v7.descriptor(),
            "method": self.method,
            "replacement_rule": REPLACEMENT_RULE_V8,
            "head_rule": HEAD_RULE,
            "head_replacement_rule": HEAD_REPLACEMENT_RULE,
            "head_avoidance": self.head_avoidance,
        }

    def record(self) -> Dict[str, Any]:
        """Descriptor plus v2's seven counters, for rollout probes."""
        return {**self.descriptor(), "counters": self.counters.to_dict()}


def _landing_ok(snake: "Snake", others: Sequence["Snake"], direction: int, cap: int, need: int):
    """v5's landing check for a boost in ``direction`` (``None`` = the boost does not fire)."""
    count = landing_count(snake, others, direction, cap)
    return count is None or count >= need


def _as_list(values: Any) -> List[Any]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    return list(values)


def install_space_and_head_veto(
    snake: Any,
    lam: float,
    head_avoidance: bool = True,
    reference_lambda: Optional[float] = None,
) -> SpaceAndHeadVeto:
    """Attach a fresh :class:`SpaceAndHeadVeto` (``lambda = lam``) to an AISnake."""
    if not hasattr(snake, "safety_veto"):
        raise TypeError("safety veto can only be installed on an AISnake")
    veto = SpaceAndHeadVeto(lam, head_avoidance=head_avoidance, reference_lambda=reference_lambda)
    snake.safety_veto = veto
    return veto
