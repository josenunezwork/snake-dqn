"""Opt-in serving-time free-space veto, v5: boost-aware landing check.

This module adds a NEW veto, ``free-space-veto/v5-boost-aware``. It changes neither
:mod:`src.evaluation.safety_veto` (v2, whose bytes the strict receipt binds) nor the
retired v3/v4 modules. It reuses v2's pure pieces unchanged:
:func:`~src.evaluation.safety_veto.free_space_threshold` (cap/need),
:func:`~src.evaluation.safety_veto.spacious_directions` (the one-step test),
:func:`~src.evaluation.safety_veto.veto_choice` (the speed-preserving replacement rule,
which never selects a masked action) and
:class:`~src.evaluation.safety_veto.SafetyVetoCounters` (the seven-counter probe shape
``strict_promotion`` validates). The grid and the "other live snakes are walls" set come
from v3's :func:`~src.evaluation.safety_veto_v3.grid_for` and
:func:`~src.evaluation.safety_veto_v3.static_blocked`, which reproduce
``SnakeStateMixin._get_free_space_features`` (``src/game/snake_state.py:548-589``).

Why (``docs/research/trap_horizon_2026-10-01.md``): in 24 of 42 self-collision deaths
under the released v2 veto, the point-of-no-return action was a boost whose two-cell
landing state had 1 to 24 reachable cells (need 160, or 50) while the same direction at
normal speed still had a way out. v2 scores a boost only from its FIRST cell (its
documented one-step approximation), so it kept those boosts.

Boost mechanics in the live game (what :func:`simulate_move` reproduces exactly):

* ``AISnake.update`` decodes the action (``src/game/ai_snake.py:512-513``: ``a >= 3``
  boosts, ``a % 3`` is the relative direction), sets the absolute direction with
  ``GameLogic.relative_to_absolute_direction`` (``ai_snake.py:516-517``;
  ``src/game/game_logic.py:99-128``), sets ``is_boosting`` only when
  ``length >= MIN_BOOST_LENGTH`` (``ai_snake.py:520-523``; default 5,
  ``src/core/game_config.py:46``) and calls ``Snake.move`` (``ai_snake.py:525``).
* ``Snake.move`` (``src/game/snake.py:134-177``): step 1 inserts ``head + direction``
  at index 0 and pops the tail while ``len(segments) > length`` (``snake.py:145-153``).
  When boosting, step 2 inserts a second head one more cell along the same direction and
  pops again under the same rule (``snake.py:156-164``). So in one frame the head
  occupies TWO cells: the first cell becomes ``segments[1]`` (the body between) and the
  second is the new head. Then the boost cost (``snake.py:166-177``): ``boost_frames``
  increments, and when it reaches ``BOOST_LENGTH_COST_FRAMES`` (default 3,
  ``game_config.py:47``) it resets to 0, ``length`` drops by one (if ``> 1``) and one
  more tail segment is popped while ``len(segments) > length`` (under mechanics v2 that
  cell becomes a trail pellet, which is food, not a wall).
* Food eaten this frame (``GameState.check_food_consumption``,
  ``src/game/game_state.py:510-516``) only calls ``grow`` (``length += 1``); it does
  not change this frame's segments. Self-collision is checked afterwards for every
  traversed head cell against ``segments[3:]`` (``game_logic.py:228-251``).
* Other snakes act on pre-move snapshots (``game_state.py:361-367``); as in v2 they are
  static walls here.

Rule, per greedy decision (``base`` = the policy's masked argmax, ``d = base % 3``):

1. **v2's one-step test, unchanged.** ``cap, need`` and ``spacious[d]`` exactly as v2
   (``round(feature * cap) >= need`` on the snake's own ``_get_free_space_features``).
   A non-boost action is eligible exactly as in v2: masked-legal and spacious.
2. **Landing test for boosts.** A boost action ``3 + d`` is eligible only if it is
   masked-legal, ``spacious[d]`` AND its landing count is ``>= need``. The landing count
   simulates the boost's one-frame movement (:func:`simulate_move`), blocks every cell of
   the resulting body except the new head (so the first cell, now ``segments[1]``, is
   blocked) plus every other live snake's cells, and flood-fills from the landing cell
   with v2's ``cap`` (the same cap and need as step 1; the count is
   ``min(cap, component size)``, as v2's). A landing cell that is out of bounds or
   blocked counts 0. If the boost would not fire (``length < MIN_BOOST_LENGTH``) the
   action moves one cell and the landing test is skipped (it equals step 1).
3. **Choice.** If the base action is eligible it is kept (v2 keeps it too). If the base
   is a boost that v2 would keep (masked-legal, spacious) but whose landing fails
   (a *landing veto*): replace it with the SAME DIRECTION at normal speed (``d``) when
   that action is masked-legal (it is spacious by step 1's test, same first cell);
   otherwise apply v2's :func:`veto_choice` over the v5-eligible actions. In every other
   case apply v2's :func:`veto_choice` over the v5-eligible actions (v2's mask with the
   landing-failing boosts cleared), so the speed-preserving rule is v2's. If no action
   is eligible the base action is left unchanged (``no_spacious``), as in v2.

So v5 differs from v2 only where a boost's landing fails: it never keeps or chooses
such a boost while an eligible action exists. For a non-boost base the result equals
v2's unless v2's replacement would be a landing-failing boost. Landing counts are
computed lazily (at most one flood fill per boost direction and decision, only when the
choice depends on it); :func:`boost_aware_choice` with an eager landing table gives the
same result (tested).

Records. ``record()`` is the descriptor plus exactly v2's seven counters, so
``strict_promotion._validate_candidate_wrapper_probe`` accepts it. In the probe a
landing veto is a ``vetoed`` decision (``vetoed_base_boost`` and, for a same-direction
normal-speed replacement, ``vetoes_speed_switched``: unlike v2 that counter is then not
limited to "no same-mode action eligible"); a landing failure with nothing eligible is
``no_spacious``. The v5 counters (``boost_landing_vetoes``,
``boost_to_normal_same_direction``, landing checks, comparison with v2) are returned by
:meth:`BoostAwareFreeSpaceVeto.diagnostics_record`, stored beside the probe (v4's
pattern).

Default behavior is untouched: nothing installs this class unless a caller does so
explicitly with :func:`install_boost_aware_veto`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.core.game_config import GameConfig
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
from src.evaluation.safety_veto_v3 import grid_for, static_blocked, tail_aware_reachable
from src.game.game_logic import GameLogic
from src.game.snake_state import FREE_SPACE_BFS_CAP, FREE_SPACE_MIN_CAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.game.snake import Snake

VETO_METHOD_V5 = "free-space-veto/v5-boost-aware"
BOOST_APPROXIMATION_V5 = "first-cell-and-two-cell-landing"
LANDING_RULE = "post-move-body-blocked-except-head/other-snakes-static/v2-cap-need"
REPLACEMENT_RULE_V5 = (
    "landing-veto-same-direction-normal-speed-first/"
    "highest-q-eligible-same-speed-mode-then-other"
)

Point = Tuple[float, float]

# Decision reasons (diagnostics only; the probe records the v2-shaped outcome).
REASON_KEPT = "kept"
REASON_V2_RULE = "v2_rule"
REASON_LANDING_SAME_DIRECTION = "landing_same_direction_normal"
REASON_LANDING_V2_RULE = "landing_v2_rule"
REASON_LANDING_NO_ELIGIBLE = "landing_no_eligible"
LANDING_REASONS = (
    REASON_LANDING_SAME_DIRECTION,
    REASON_LANDING_V2_RULE,
    REASON_LANDING_NO_ELIGIBLE,
)


@dataclass(frozen=True)
class SimulatedMove:
    """One frame of ``Snake.move`` for a hypothetical action (pixel coordinates).

    ``traversed`` is ``last_move_positions`` (one cell, or two when boosting);
    ``segments``/``length``/``boost_frames`` are the snake's values after the move;
    ``burned_tail`` is the tail popped by a boost burn (``None`` if none).
    """

    traversed: Tuple[Point, ...]
    segments: Tuple[Point, ...]
    length: int
    boost_frames: int
    boosted: bool
    burned_tail: Optional[Point]


def simulate_move(
    segments: Sequence[Point],
    direction: Tuple[int, int],
    length: int,
    boost_frames: int,
    boosting: bool,
    segment_size: int,
    min_boost_length: int,
    boost_cost_frames: int,
) -> SimulatedMove:
    """Replicate ``Snake.move`` (``src/game/snake.py:134-177``) without mutating a snake.

    ``direction`` is the absolute direction AFTER the action is applied; ``boosting`` is
    ``Snake.is_boosting`` as ``AISnake.update`` sets it (``ai_snake.py:520-523``).
    """
    body: List[Point] = list(segments)
    dx, dy = direction
    length = int(length)
    frames = int(boost_frames)
    first = (body[0][0] + dx * segment_size, body[0][1] + dy * segment_size)
    traversed: List[Point] = [first]
    body.insert(0, first)
    if len(body) > length:
        body.pop()
    burned: Optional[Point] = None
    boosted = bool(boosting) and length >= int(min_boost_length)
    if boosted:
        second = (body[0][0] + dx * segment_size, body[0][1] + dy * segment_size)
        traversed.append(second)
        body.insert(0, second)
        if len(body) > length:
            body.pop()
        frames += 1
        if frames >= int(boost_cost_frames):
            frames = 0
            if length > 1:
                length -= 1
                if len(body) > length:
                    burned = body.pop()
    return SimulatedMove(tuple(traversed), tuple(body), length, frames, boosted, burned)


def simulate_action(snake: "Snake", action: int) -> SimulatedMove:
    """The movement ``AISnake.update`` would make for ``action`` (0..5) this frame."""
    relative = int(action) % NUM_DIRECTIONS
    direction = GameLogic.relative_to_absolute_direction(tuple(snake.direction), relative)
    boosting = int(action) >= NUM_DIRECTIONS and snake.length >= GameConfig.MIN_BOOST_LENGTH
    return simulate_move(
        snake.segments,
        direction,
        int(snake.length),
        int(snake.boost_frames),
        boosting,
        max(int(snake.segment_size), 1),
        int(GameConfig.MIN_BOOST_LENGTH),
        int(GameConfig.BOOST_LENGTH_COST_FRAMES),
    )


def landing_count(
    snake: "Snake", other_snakes: Sequence["Snake"], direction: int, cap: int
) -> Optional[int]:
    """Reachable cells from boost direction ``direction``'s landing cell (at most ``cap``).

    Returns ``None`` when the boost would not fire (``length < MIN_BOOST_LENGTH``): the
    action then moves one cell and v2's one-step count already describes it.
    """
    move = simulate_action(snake, NUM_DIRECTIONS + int(direction))
    if not move.boosted:
        return None
    grid = grid_for(snake)
    blocked = static_blocked(snake, other_snakes, grid)
    blocked.update(grid.to_cell(x, y) for x, y in move.segments[1:])
    start = grid.to_cell(*move.segments[0])
    # No release map: every blocked cell stays blocked, i.e. v2's static capped count.
    return tail_aware_reachable(start, 0, blocked, {}, grid.in_bounds, int(cap))


def _cleared_boosts(
    mask: Sequence[bool], spacious: Sequence[bool], landing_ok: Callable[[int], bool]
) -> List[bool]:
    """``mask`` with every landing-failing boost cleared (floods only legal spacious ones)."""
    out = [bool(value) for value in mask]
    for direction in range(NUM_DIRECTIONS):
        action = NUM_DIRECTIONS + direction
        if out[action] and spacious[direction] and not landing_ok(direction):
            out[action] = False
    return out


def boost_aware_choice(
    q_values: Sequence[float],
    action_mask: Sequence[bool],
    spacious: Sequence[bool],
    base_action: int,
    landing_ok: Callable[[int], bool],
) -> Tuple[int, str, str]:
    """Apply the v5 rule to one greedy decision (pure given ``landing_ok``).

    Returns ``(action, outcome, reason)``: ``outcome`` is the v2-shaped probe outcome
    (``kept``/``vetoed``/``no_spacious``), ``reason`` one of the ``REASON_*`` values.
    ``landing_ok(d)`` says whether boost direction ``d``'s landing meets ``need``; it is
    called only for masked-legal boosts whose direction is spacious, and only when the
    result depends on it.
    """
    if len(q_values) != NUM_ACTIONS or len(action_mask) != NUM_ACTIONS:
        raise ValueError(f"expected {NUM_ACTIONS} Q-values and mask entries")
    if len(spacious) != NUM_DIRECTIONS:
        raise ValueError(f"expected {NUM_DIRECTIONS} spacious flags")
    mask = [bool(value) for value in action_mask]
    base = int(base_action)
    if 0 <= base < NUM_ACTIONS and mask[base] and spacious[base % NUM_DIRECTIONS]:
        direction = base % NUM_DIRECTIONS
        if base < NUM_DIRECTIONS or landing_ok(direction):
            return base, OUTCOME_KEPT, REASON_KEPT
        # Landing veto: v2 would keep this boost; its two-cell landing is a pocket.
        if mask[direction]:
            return direction, OUTCOME_VETOED, REASON_LANDING_SAME_DIRECTION
        action, outcome = veto_choice(
            q_values, _cleared_boosts(mask, spacious, landing_ok), spacious, base
        )
        if outcome == OUTCOME_NO_SPACIOUS:
            return base, OUTCOME_NO_SPACIOUS, REASON_LANDING_NO_ELIGIBLE
        return action, outcome, REASON_LANDING_V2_RULE
    # The base is not v2-eligible: v2's rule. Clearing landing-failing boosts can only
    # change the result when v2's own pick is such a boost (see module docstring).
    action, outcome = veto_choice(q_values, mask, spacious, base)
    if outcome == OUTCOME_VETOED and action >= NUM_DIRECTIONS:
        if not landing_ok(action % NUM_DIRECTIONS):
            action, outcome = veto_choice(
                q_values, _cleared_boosts(mask, spacious, landing_ok), spacious, base
            )
    return action, outcome, REASON_V2_RULE


def eager_boost_aware_choice(
    q_values: Sequence[float],
    action_mask: Sequence[bool],
    spacious: Sequence[bool],
    base_action: int,
    landing: Sequence[bool],
) -> Tuple[int, str]:
    """Reference form of the rule with a precomputed landing table (tests use it).

    Eligible = v2-eligible, minus boosts whose ``landing[d]`` is False; a boost base that
    is v2-eligible but landing-failing prefers its masked-legal normal-speed direction;
    otherwise v2's :func:`veto_choice` over the eligible actions.
    """
    mask = [bool(m) for m in action_mask]
    reduced = [
        mask[a] and (a < NUM_DIRECTIONS or bool(landing[a % NUM_DIRECTIONS]))
        for a in range(NUM_ACTIONS)
    ]
    base = int(base_action)
    direction = base % NUM_DIRECTIONS
    landing_veto = (
        NUM_DIRECTIONS <= base < NUM_ACTIONS
        and mask[base]
        and spacious[direction]
        and not landing[direction]
    )
    if landing_veto and mask[direction]:
        return direction, OUTCOME_VETOED
    return veto_choice(q_values, reduced, spacious, base)


@dataclass
class BoostAwareCounters:
    """Per-episode v5 bookkeeping (reported beside the probe; never gates).

    Identities: ``decisions == kept + vetoes_applied + no_spacious`` (these mirror the
    probe's ``kept_base``/``vetoes_applied``/``fallback_no_spacious``);
    ``base_landing_failed == boost_landing_vetoes + boost_landing_no_eligible``;
    ``boost_landing_vetoes == boost_to_normal_same_direction + boost_landing_v2_rule``.
    ``landing_checks`` counts flood fills from a landing cell (``landing_failures`` of
    them below ``need``). ``action_differs_from_v2`` compares with the released v2 rule on
    the same inputs. ``apply_seconds_*`` are wall-clock and vary between runs.
    """

    decisions: int = 0
    kept: int = 0
    vetoes_applied: int = 0
    no_spacious: int = 0
    boost_base_decisions: int = 0
    base_landing_failed: int = 0
    boost_landing_vetoes: int = 0
    boost_to_normal_same_direction: int = 0
    boost_landing_v2_rule: int = 0
    boost_landing_no_eligible: int = 0
    landing_checks: int = 0
    landing_failures: int = 0
    action_differs_from_v2: int = 0
    apply_seconds_total: float = 0.0
    apply_seconds_max: float = 0.0

    def record(
        self,
        base_action: int,
        outcome: str,
        reason: str,
        differs_from_v2: bool,
        checks: int,
        failures: int,
        seconds: float,
    ) -> None:
        """Count one decision."""
        field = {
            OUTCOME_KEPT: "kept",
            OUTCOME_VETOED: "vetoes_applied",
            OUTCOME_NO_SPACIOUS: "no_spacious",
        }.get(outcome)
        if field is None:
            raise ValueError(f"unknown veto outcome {outcome!r}")
        self.decisions += 1
        setattr(self, field, getattr(self, field) + 1)
        if int(base_action) >= NUM_DIRECTIONS:
            self.boost_base_decisions += 1
        if reason in LANDING_REASONS:
            self.base_landing_failed += 1
            if reason == REASON_LANDING_NO_ELIGIBLE:
                self.boost_landing_no_eligible += 1
            else:
                self.boost_landing_vetoes += 1
                if reason == REASON_LANDING_SAME_DIRECTION:
                    self.boost_to_normal_same_direction += 1
                else:
                    self.boost_landing_v2_rule += 1
        elif reason not in (REASON_KEPT, REASON_V2_RULE):
            raise ValueError(f"unknown v5 reason {reason!r}")
        self.action_differs_from_v2 += int(bool(differs_from_v2))
        self.landing_checks += int(checks)
        self.landing_failures += int(failures)
        self.apply_seconds_total += float(seconds)
        self.apply_seconds_max = max(self.apply_seconds_max, float(seconds))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping plus the mean per-decision wall cost."""
        out: Dict[str, Any] = {
            key: (float(value) if isinstance(value, float) else int(value))
            for key, value in self.__dict__.items()
        }
        out["mean_apply_seconds"] = (
            self.apply_seconds_total / self.decisions if self.decisions else None
        )
        return out


class BoostAwareFreeSpaceVeto:
    """Stateful per-snake veto hook (same surface as ``FreeSpaceVeto``)."""

    method = VETO_METHOD_V5

    def __init__(self) -> None:
        self.counters = SafetyVetoCounters()
        self.boost = BoostAwareCounters()

    def reset(self) -> None:
        """Zero all counters (call at episode start)."""
        self.counters = SafetyVetoCounters()
        self.boost = BoostAwareCounters()

    def apply(
        self,
        snake: "Snake",
        other_snakes: Sequence["Snake"],
        q_values: Any,
        action_mask: Any,
        base_action: int,
    ) -> int:
        """Return the (possibly vetoed) action for one greedy decision."""
        started = time.perf_counter()
        others = list(other_snakes)
        q = _as_list(q_values)
        mask = [bool(x) for x in _as_list(action_mask)]
        base = int(base_action)
        features = snake._get_free_space_features(others)
        cap, need = free_space_threshold(snake.length, snake._logical_length())
        spacious = spacious_directions(features, cap, need)
        memo: Dict[int, bool] = {}
        checks = failures = 0

        def landing_ok(direction: int) -> bool:
            nonlocal checks, failures
            if direction not in memo:
                count = landing_count(snake, others, direction, cap)
                memo[direction] = count is None or count >= need
                checks += int(count is not None)  # a flood fill actually ran
                failures += int(not memo[direction])
            return memo[direction]

        action, outcome, reason = boost_aware_choice(q, mask, spacious, base, landing_ok)
        v2_action, _ = veto_choice(q, mask, spacious, base)
        self.counters.record(base, action, outcome)
        self.boost.record(
            base,
            outcome,
            reason,
            action != v2_action,
            checks,
            failures,
            time.perf_counter() - started,
        )
        return action

    def diagnostics_record(self) -> Dict[str, Any]:
        """The v5 counters (``BoostAwareCounters.to_dict``), stored beside the probe."""
        return self.boost.to_dict()

    def descriptor(self) -> Dict[str, Any]:
        """Static identity of this veto (no counters)."""
        return {
            "method": VETO_METHOD_V5,
            "free_space_bfs_cap": FREE_SPACE_BFS_CAP,
            "free_space_min_cap": FREE_SPACE_MIN_CAP,
            "boost_approximation": BOOST_APPROXIMATION_V5,
            "replacement_rule": REPLACEMENT_RULE_V5,
            "landing_rule": LANDING_RULE,
        }

    def record(self) -> Dict[str, Any]:
        """Descriptor plus v2's seven counters, for rollout probes."""
        return {**self.descriptor(), "counters": self.counters.to_dict()}


def _as_list(values: Any) -> List[Any]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    return list(values)


def install_boost_aware_veto(snake: Any) -> BoostAwareFreeSpaceVeto:
    """Attach a fresh :class:`BoostAwareFreeSpaceVeto` to an AISnake and return it."""
    if not hasattr(snake, "safety_veto"):
        raise TypeError("safety veto can only be installed on an AISnake")
    veto = BoostAwareFreeSpaceVeto()
    snake.safety_veto = veto
    return veto
