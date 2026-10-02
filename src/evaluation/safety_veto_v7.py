"""Opt-in serving-time free-space veto, v7: space preference among v5's eligible moves.

This module adds a NEW veto, ``free-space-veto/v7-space-preference(lambda=<x>)``. It
changes none of the released or retired vetoes (:mod:`src.evaluation.safety_veto` v2,
``safety_veto_v3``/``_v4``/``_v5``/``_v6`` stay byte-identical). v5's decision machinery
is imported and run unchanged: v2's cap/need and one-step spacious test
(:func:`~src.evaluation.safety_veto.free_space_threshold`,
:func:`~src.evaluation.safety_veto.spacious_directions`), v5's lazy two-cell landing
check (:func:`~src.evaluation.safety_veto_v5.landing_count`) and v5's choice
(:func:`~src.evaluation.safety_veto_v5.boost_aware_choice`).

Why (``docs/research/death_census_v5_2026-10-02.md``): most remaining self deaths under
v5 begin with the hero entering a region too small for it 20 to 60 frames before death,
at lengths 262 to 807. v5 only asks whether a direction reaches ``need`` (at most 160)
cells, so among several "spacious" moves it follows the policy's Q-values even when one
move leads into a much smaller region than another. v7 adds a soft preference for the
larger region, measured with a larger cap, whenever v5 had a real choice.

Rule, per greedy decision (``q`` = the masked Q row, ``c`` = v5's chosen action):

1. **v5 unchanged.** ``(c, outcome, reason) = boost_aware_choice(...)``, exactly as
   :meth:`BoostAwareFreeSpaceVeto.apply` computes it.
2. **No re-rank** when v5's outcome is ``no_spacious`` (v5's choice stands, as in v5),
   when ``lambda == 0`` (so ``lambda=0`` IS v5, decision for decision), and when v5's
   reason is a landing veto replaced by the same direction at normal speed (that is
   v5's deliberate rule, not a Q argmax, so a re-rank could not reproduce it at
   ``lambda=0``).
3. **Candidates** ``C``: the v5-eligible actions in the speed mode of ``c`` (masked-legal,
   v2-spacious direction, and for a boost a passing v5 landing check). ``c`` is the
   highest-Q member of ``C`` with lowest-index ties whenever the base action is the
   policy's masked argmax (v5 keeps the base, or picks the highest-Q eligible action of
   the base's speed mode, else of the other mode). Keeping the speed mode is v2's
   speed-preserving principle: the re-rank chooses between directions, never adds
   boosting. Only ``|C| >= 2`` is re-ranked.
4. **Score** ``s(a) = Qn(a) + lambda * g(a)`` with
   ``Qn(a) = (q[a] - max_C q) / max(|max_C q - min_C q|, 1e-6)`` in ``[-1, 0]`` (unit
   free) and ``g(a) = log1p(area(a)) / log1p(cap)`` in ``[0, 1]``. The chosen action is
   the argmax of ``s`` over ``C``; ties go to ``c``, then to the lowest action index.
   Every choice is deterministic (no randomness or timing).
5. **Area** (the v3 tail-aware model on the exact post-move state):
   :func:`~src.evaluation.safety_veto_v5.simulate_action` gives the hero's body after
   action ``a`` (one cell, or two when the boost fires, including a boost burn); the
   breadth-first count of :func:`~src.evaluation.safety_veto_v3.tail_aware_reachable`
   starts at the new head (distance 0), other live snakes are static walls
   (:func:`~src.evaluation.safety_veto_v3.static_blocked`), and the own body releases
   by steps (:func:`~src.evaluation.safety_veto_v3.own_body_release` on the post-move
   body and length, slack 1, the new head excluded). For a normal move this is v3's
   one-step model shifted by one frame, except that a move into the current tail cell
   (which the tail leaves this frame) is counted instead of being blocked.
6. **Cap** ``min(max(32, 4 * length), 4096, open cells)``, where open cells are the
   in-bounds cells not covered by another live snake (the most the count can reach).
   4096 bounds the runtime: a capped breadth-first count costs about 0.3 us per cell.
   ``4 * length`` exceeds 4096 only for heroes longer than 1024 (census lengths
   262 to 807), and ``g`` is logarithmic, so the cap mostly affects regions that are
   already large.

**Cost.** Areas are evaluated lazily and only where they can change the result: an
alternative ``a`` whose best case ``Qn(a) + lambda`` does not exceed the best score
so far is skipped (exact, since ``g <= 1``), and the anchor ``c`` is evaluated first.
:func:`eager_space_preference_choice` gives the same result from a full table (tested).

Records. ``record()`` is the descriptor plus exactly v2's seven counters, so
``strict_promotion._validate_candidate_wrapper_probe`` accepts it. A re-rank that
replaces the base action is a ``vetoed`` decision (also when v5 had ``kept`` it); the
speed mode never changes in a re-rank. The v7 counters (``rerank_changes``,
``area_evaluations``, ``area_cap_hits``, ``area_eval_seconds_*``, nested v5 counters)
come from :meth:`SpacePreferenceVeto.diagnostics_record`.

Default behavior is untouched: nothing installs this class unless a caller does so
explicitly with :func:`install_space_preference_veto`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Sequence, Tuple

from src.evaluation.safety_veto import (
    NUM_ACTIONS,
    NUM_DIRECTIONS,
    OUTCOME_KEPT,
    OUTCOME_NO_SPACIOUS,
    OUTCOME_VETOED,
    SafetyVetoCounters,
    free_space_threshold,
    spacious_directions,
    veto_choice,
)
from src.evaluation.safety_veto_v3 import (
    TAIL_RELEASE_SLACK,
    Grid,
    grid_for,
    own_body_release,
    static_blocked,
    tail_aware_reachable,
)
from src.evaluation.safety_veto_v5 import (
    BOOST_APPROXIMATION_V5,
    LANDING_RULE,
    REASON_LANDING_SAME_DIRECTION,
    REPLACEMENT_RULE_V5,
    BoostAwareCounters,
    boost_aware_choice,
    landing_count,
    simulate_action,
)
from src.game.snake_state import FREE_SPACE_BFS_CAP, FREE_SPACE_MIN_CAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.game.snake import Snake

VETO_METHOD_V7_PREFIX = "free-space-veto/v7-space-preference"
AREA_CAP_MIN = FREE_SPACE_MIN_CAP
AREA_CAP_LENGTH_FACTOR = 4
AREA_CAP_LIMIT = 4096
Q_SCALE_EPS = 1e-6
AREA_RULE = (
    "tail-aware-bfs-from-post-move-head/own-body-released-by-steps/"
    f"slack-{TAIL_RELEASE_SLACK}/other-snakes-static"
)
AREA_CAP_RULE = (
    f"min(max({AREA_CAP_MIN}, {AREA_CAP_LENGTH_FACTOR}*length), {AREA_CAP_LIMIT}, "
    "in-bounds cells not covered by other live snakes)"
)
SCORE_RULE = "(q-max_C q)/max(|max_C q-min_C q|,1e-6) + lambda*log1p(area)/log1p(cap)"
RERANK_RULE = (
    "C = v5-eligible actions in the speed mode of v5's choice; argmax score over C; "
    "ties -> v5's choice, then lowest index; unchanged when no_spacious, lambda == 0 "
    "or a landing veto's same-direction normal-speed replacement"
)


def validate_lambda(lam: Any) -> float:
    """``lambda`` as a finite non-negative float (raises ``ValueError`` otherwise)."""
    if isinstance(lam, bool):
        raise ValueError("lambda must be a number, not a bool")
    value = float(lam)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"lambda must be finite and >= 0, got {lam!r}")
    return value


def method_for(lam: float) -> str:
    """The probe ``method`` string for ``lambda`` (``repr`` of the float)."""
    return f"{VETO_METHOD_V7_PREFIX}(lambda={validate_lambda(lam)!r})"


def area_score(area: int, cap: int) -> float:
    """``g = log1p(area) / log1p(cap)``, in ``[0, 1]`` for ``0 <= area <= cap``."""
    return math.log1p(max(int(area), 0)) / math.log1p(max(int(cap), 1))


@lru_cache(maxsize=32)
def in_bounds_cells(grid: Grid) -> int:
    """Number of in-bounds cells of ``grid`` (rectangle, or the circular-arena test)."""
    if not grid.circular:
        return int(grid.width) * int(grid.height)
    return sum(1 for x in range(grid.width) for y in range(grid.height) if grid.in_bounds((x, y)))


def area_cap(length: int, grid: Grid, blocked: set) -> int:
    """The re-rank cap (see the module docstring, item 6)."""
    covered = sum(1 for cell in blocked if grid.in_bounds(cell))
    open_cells = in_bounds_cells(grid) - covered
    by_length = max(AREA_CAP_MIN, AREA_CAP_LENGTH_FACTOR * int(length))
    return max(1, min(by_length, AREA_CAP_LIMIT, open_cells))


def landing_area(
    snake: "Snake",
    action: int,
    blocked: set,
    grid: Grid,
    cap: int,
    slack: int = TAIL_RELEASE_SLACK,
) -> int:
    """Tail-aware reachable cells from ``action``'s post-move head (at most ``cap``)."""
    move = simulate_action(snake, int(action))
    head = grid.to_cell(*move.segments[0])
    release = own_body_release(move.segments, move.length, grid, slack)
    release.pop(head, None)  # the search starts on the new head (distance 0)
    return tail_aware_reachable(head, 0, blocked, release, grid.in_bounds, int(cap))


def rerank_candidates(
    v5_action: int,
    reason: str,
    action_mask: Sequence[bool],
    spacious: Sequence[bool],
    landing_ok: Callable[[int], bool],
) -> List[int]:
    """``C``: v5-eligible actions in the speed mode of v5's choice (ascending index).

    A landing veto's same-direction normal-speed replacement keeps ``[v5_action]`` (no
    re-rank). ``landing_ok`` is called only for masked-legal spacious boosts other than
    v5's choice, and only when v5's choice is itself a boost.
    """
    if reason == REASON_LANDING_SAME_DIRECTION:
        return [int(v5_action)]
    offset = NUM_DIRECTIONS if int(v5_action) >= NUM_DIRECTIONS else 0
    out: List[int] = []
    for direction in range(NUM_DIRECTIONS):
        action = offset + direction
        if action == int(v5_action):
            out.append(action)
        elif action_mask[action] and spacious[direction]:
            if offset == 0 or landing_ok(direction):
                out.append(action)
    return out


def normalized_q(q_values: Sequence[float], candidates: Sequence[int]) -> Dict[int, float]:
    """``Qn(a) = (q[a] - max_C q) / max(|max_C q - min_C q|, eps)`` over ``candidates``."""
    values = {a: float(q_values[a]) for a in candidates}
    top, bottom = max(values.values()), min(values.values())
    scale = max(abs(top - bottom), Q_SCALE_EPS)
    return {a: (value - top) / scale for a, value in values.items()}


def space_preference_choice(
    q_values: Sequence[float],
    candidates: Sequence[int],
    anchor: int,
    lam: float,
    score_of: Callable[[int], float],
) -> Tuple[int, Dict[int, float]]:
    """Argmax of ``Qn + lam * g`` over ``candidates`` (ties: ``anchor``, then lowest index).

    ``score_of(a)`` returns ``g(a)`` in ``[0, 1]`` and is called lazily: never for an
    action whose best case ``Qn(a) + lam`` cannot exceed the best score so far (exact,
    because ``g <= 1``). Returns ``(action, {action: g})`` for the evaluated actions.
    """
    anchor = int(anchor)
    if anchor not in candidates:
        raise ValueError("the anchor (v5's choice) must be a candidate")
    lam = validate_lambda(lam)
    if len(candidates) < 2:
        return anchor, {}
    qn = normalized_q(q_values, candidates)
    others = sorted(int(a) for a in candidates if int(a) != anchor)
    if all(qn[a] + lam <= qn[anchor] for a in others):  # g(anchor) >= 0: nobody can win
        return anchor, {}
    scores = {anchor: float(score_of(anchor))}
    best, best_score = anchor, qn[anchor] + lam * scores[anchor]
    for action in others:
        if qn[action] + lam <= best_score:
            continue
        scores[action] = float(score_of(action))
        value = qn[action] + lam * scores[action]
        if value > best_score:
            best, best_score = action, value
    return best, scores


def eager_space_preference_choice(
    q_values: Sequence[float],
    candidates: Sequence[int],
    anchor: int,
    lam: float,
    scores: Mapping[int, float],
) -> int:
    """Reference form of :func:`space_preference_choice` from a full ``g`` table."""
    qn = normalized_q(q_values, candidates)
    lam = validate_lambda(lam)
    return max(
        (int(a) for a in candidates),
        key=lambda a: (qn[a] + lam * float(scores[a]), a == int(anchor), -a),
    )


@dataclass
class SpacePreferenceCounters:
    """Per-episode v7 bookkeeping (reported beside the probe; never gates).

    Identities: ``decisions == kept + vetoes_applied + no_spacious`` (mirrors of the
    probe's ``kept_base``/``vetoes_applied``/``fallback_no_spacious``);
    ``kept == v5.kept - rerank_changes_of_v5_kept``;
    ``vetoes_applied == v5.vetoes_applied + rerank_changes_of_v5_kept``;
    ``no_spacious == v5.no_spacious``; ``rerank_changes <= rerank_decisions``;
    ``rerank_decisions == rerank_pruned + rerank_scored``;
    ``area_cap_hits <= area_evaluations``. ``*_seconds_*`` are wall-clock.
    """

    decisions: int = 0
    kept: int = 0
    vetoes_applied: int = 0
    no_spacious: int = 0
    rerank_decisions: int = 0
    rerank_pruned: int = 0
    rerank_scored: int = 0
    rerank_changes: int = 0
    rerank_changes_of_v5_kept: int = 0
    rerank_changes_boost_mode: int = 0
    area_evaluations: int = 0
    area_cap_hits: int = 0
    area_cap_max: int = 0
    extra_landing_checks: int = 0
    area_eval_seconds_total: float = 0.0
    area_eval_seconds_max: float = 0.0
    apply_seconds_total: float = 0.0
    apply_seconds_max: float = 0.0
    v5: BoostAwareCounters = field(default_factory=BoostAwareCounters)

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
        """Accumulate wall-clock ``seconds`` under ``apply`` or ``area_eval``."""
        total = f"{kind}_seconds_total"
        peak = f"{kind}_seconds_max"
        setattr(self, total, getattr(self, total) + float(seconds))
        setattr(self, peak, max(getattr(self, peak), float(seconds)))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping plus mean costs; ``v5`` is the nested v5 counter dict."""
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key == "v5":
                continue
            out[key] = float(value) if isinstance(value, float) else int(value)
        out["mean_apply_seconds"] = (
            self.apply_seconds_total / self.decisions if self.decisions else None
        )
        out["mean_area_eval_seconds_per_evaluation"] = (
            self.area_eval_seconds_total / self.area_evaluations if self.area_evaluations else None
        )
        out["v5"] = self.v5.to_dict()
        return out


class SpacePreferenceVeto:
    """Stateful per-snake veto hook (same surface as ``FreeSpaceVeto``)."""

    def __init__(self, lam: float) -> None:
        self.lam = validate_lambda(lam)
        self.method = method_for(self.lam)
        self.counters = SafetyVetoCounters()
        self.v7 = SpacePreferenceCounters()

    def reset(self) -> None:
        """Zero all counters (call at episode start)."""
        self.counters = SafetyVetoCounters()
        self.v7 = SpacePreferenceCounters()

    def apply(
        self,
        snake: "Snake",
        other_snakes: Sequence["Snake"],
        q_values: Any,
        action_mask: Any,
        base_action: int,
    ) -> int:
        """Return the (possibly vetoed or re-ranked) action for one greedy decision."""
        started = time.perf_counter()
        action, outcome = self.decide(
            snake,
            list(other_snakes),
            _as_list(q_values),
            [bool(x) for x in _as_list(action_mask)],
            int(base_action),
        )
        self.counters.record(int(base_action), action, outcome)
        self.v7.count_outcome(outcome)
        self.v7.add_time("apply", time.perf_counter() - started)
        return action

    def decide(
        self,
        snake: "Snake",
        others: Sequence["Snake"],
        q: Sequence[float],
        mask: Sequence[bool],
        base: int,
    ) -> Tuple[int, str]:
        """``(action, probe outcome)``; updates the v7 and nested v5 diagnostics."""
        if len(q) != NUM_ACTIONS or len(mask) != NUM_ACTIONS:
            raise ValueError(f"expected {NUM_ACTIONS} Q-values and mask entries")
        started = time.perf_counter()
        features = snake._get_free_space_features(list(others))
        cap, need = free_space_threshold(snake.length, snake._logical_length())
        spacious = spacious_directions(features, cap, need)
        memo: Dict[int, bool] = {}
        tally = {"checks": 0, "failures": 0, "extra": 0, "in_v5": 1}

        def landing_ok(direction: int) -> bool:
            if direction not in memo:
                count = landing_count(snake, others, direction, cap)
                memo[direction] = count is None or count >= need
                if tally["in_v5"]:
                    tally["checks"] += int(count is not None)
                    tally["failures"] += int(not memo[direction])
                else:
                    tally["extra"] += int(count is not None)
            return memo[direction]

        # Step 1: v5, exactly as BoostAwareFreeSpaceVeto.apply (its counters nested).
        v5_action, v5_outcome, v5_reason = boost_aware_choice(q, mask, spacious, base, landing_ok)
        v2_action, _ = veto_choice(q, mask, spacious, base)
        self.v7.v5.record(
            base,
            v5_outcome,
            v5_reason,
            v5_action != v2_action,
            tally["checks"],
            tally["failures"],
            time.perf_counter() - started,
        )
        tally["in_v5"] = 0
        if v5_outcome == OUTCOME_NO_SPACIOUS or self.lam == 0.0:
            return v5_action, v5_outcome
        candidates = rerank_candidates(v5_action, v5_reason, mask, spacious, landing_ok)
        self.v7.extra_landing_checks += tally["extra"]
        if len(candidates) < 2:
            return v5_action, v5_outcome
        action = self._rerank(snake, others, q, candidates, v5_action)
        if action == v5_action:
            return v5_action, v5_outcome
        self.v7.rerank_changes += 1
        self.v7.rerank_changes_of_v5_kept += int(v5_outcome == OUTCOME_KEPT)
        self.v7.rerank_changes_boost_mode += int(action >= NUM_DIRECTIONS)
        return action, OUTCOME_VETOED

    def _rerank(
        self,
        snake: "Snake",
        others: Sequence["Snake"],
        q: Sequence[float],
        candidates: Sequence[int],
        anchor: int,
    ) -> int:
        """Steps 3-6 of the rule for one decision with ``|C| >= 2``."""
        context: Dict[str, Any] = {}

        def score_of(action: int) -> float:
            started = time.perf_counter()
            if not context:
                grid = grid_for(snake)
                blocked = static_blocked(snake, others, grid)
                context.update(grid=grid, blocked=blocked)
                context["cap"] = area_cap(snake.length, grid, blocked)
            cap = context["cap"]
            area = landing_area(snake, action, context["blocked"], context["grid"], cap)
            self.v7.area_evaluations += 1
            self.v7.area_cap_hits += int(area >= cap)
            self.v7.area_cap_max = max(self.v7.area_cap_max, int(cap))
            self.v7.add_time("area_eval", time.perf_counter() - started)
            return area_score(area, cap)

        self.v7.rerank_decisions += 1
        action, scored = space_preference_choice(q, candidates, anchor, self.lam, score_of)
        if scored:
            self.v7.rerank_scored += 1
        else:
            self.v7.rerank_pruned += 1
        return action

    def diagnostics_record(self) -> Dict[str, Any]:
        """The v7 counters (``SpacePreferenceCounters.to_dict``), stored beside the probe."""
        return self.v7.to_dict()

    def descriptor(self) -> Dict[str, Any]:
        """Static identity of this veto (no counters)."""
        return {
            "method": self.method,
            "free_space_bfs_cap": FREE_SPACE_BFS_CAP,
            "free_space_min_cap": FREE_SPACE_MIN_CAP,
            "boost_approximation": BOOST_APPROXIMATION_V5,
            "replacement_rule": REPLACEMENT_RULE_V5,
            "landing_rule": LANDING_RULE,
            "space_preference_lambda": self.lam,
            "space_preference_rule": RERANK_RULE,
            "space_score_rule": SCORE_RULE,
            "area_rule": AREA_RULE,
            "area_cap_rule": AREA_CAP_RULE,
        }

    def record(self) -> Dict[str, Any]:
        """Descriptor plus v2's seven counters, for rollout probes."""
        return {**self.descriptor(), "counters": self.counters.to_dict()}


def _as_list(values: Any) -> List[Any]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    return list(values)


def install_space_preference_veto(snake: Any, lam: float) -> SpacePreferenceVeto:
    """Attach a fresh :class:`SpacePreferenceVeto` (``lambda = lam``) to an AISnake."""
    if not hasattr(snake, "safety_veto"):
        raise TypeError("safety veto can only be installed on an AISnake")
    veto = SpacePreferenceVeto(lam)
    snake.safety_veto = veto
    return veto
