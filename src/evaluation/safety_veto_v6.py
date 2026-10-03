"""Opt-in serving-time free-space veto, v6: opponent-head avoidance + no-spacious fallback.

This module adds a NEW veto, ``free-space-veto/v6-head-and-fallback``. It changes none of
the released or retired vetoes (:mod:`src.evaluation.safety_veto` v2,
:mod:`~src.evaluation.safety_veto_v3`, :mod:`~src.evaluation.safety_veto_v4`,
:mod:`~src.evaluation.safety_veto_v5`); it imports their pure pieces unchanged. v6 is
"v5's decision, then two layers". Both layers are ON in v6 and each is recorded in
:meth:`HeadAndFallbackVeto.diagnostics_record`.

Why (``docs/research/death_census_v5_2026-10-02.md``): under the released v5 veto,
7.8% of strict-gate episodes end in a head-on (12.3% in the scripted mix, mostly within
the first ~500 frames). In all 5 census head-ons v5 ``kept`` a move into a cell that was
empty before the move but reachable by an equal-or-larger opponent's head, while 2-5
masked-legal alternatives survived. Separately, all 36 census self deaths were
``no_spacious`` at the fatal decision, where v5 leaves the base action unchanged; 10 of
them were boosts into a 1-44 cell landing.

Layer (a), opponent-head avoidance (applies when v5's outcome is ``kept`` or ``vetoed``):

* **Reach.** For every other live snake (``is_alive`` with segments; the hero itself is
  skipped) its head can occupy next frame: its current head cell (a hero move into it is
  the head-swap case of ``GameLogic._head_paths_crossed``), and for each of its three
  relative moves (``GameLogic.relative_to_absolute_direction`` of its pre-move heading;
  it cannot reverse) the first cell, plus the second cell along the same direction when
  it can boost (``length >= MIN_BOOST_LENGTH``, ``AISnake.update``). Every cell the
  hero's head path and any reachable opponent path could share is in that set: a shared
  traversed cell is in it directly, and a swap needs one of the two heads' path starts,
  which are the opponent's current head or its first cell.
* **Head-on rule** (``GameState.handle_collisions``, ``src/game/game_state.py``): under
  ``GameConfig.MECHANICS_VERSION == 2`` a snake whose ``_logical_length()`` is
  ``>= HEADON_SIZE_RATIO`` (1.15, ``src/core/mechanics_constants.py``) times the other's
  survives; otherwise (and always at mechanics v1) both die. Lengths are read after
  movement and food consumption (``check_food_consumption`` grows by at most 1 per frame).
  So the hero is treated as the winner against an opponent only when
  ``hero_post >= 1.15 * (opp_logical + 1)``, with ``hero_post`` the hero's length after
  this action's own boost burn (:func:`~src.evaluation.safety_veto_v5.simulate_action`)
  without growth, and ``opp_logical + 1`` the opponent's largest possible length.
  This is the census's 1.15x note, which matches the code; the +1 margin is conservative.
* **Head-risky.** An action is head-risky if any cell it traverses (one, or both cells of
  a boost) is reachable by an opponent the hero would not beat.
* **Choice.** If the current (v5) action is head-risky and some OTHER action is
  masked-legal, v5-eligible (v2-spacious; for a boost also v5's landing check) and not
  head-risky, replace it by the highest-Q such action in the current action's speed mode,
  else in the other mode (ties: lowest index, as ``torch.argmax``). Otherwise keep v5's
  choice (``head_risky_kept_no_alternative``). The replacement is recorded in the
  seven-counter probe as ``vetoed``.

Layer (b), no-spacious fallback (applies only when v5's outcome is ``no_spacious``, so it
never overlaps layer (a), which needs a v5-eligible alternative):

1. **Landing.** Both counts use one model: v2's static capped flood (cap from v2) over
   the hero's body AFTER this action's one-frame move (v5's
   :func:`~src.evaluation.safety_veto_v5.simulate_action`, cells ``segments[1:]``
   blocked) and the other live snakes, from the new head. The boost count is v5's
   :func:`~src.evaluation.safety_veto_v5.landing_count` (from the second cell, the first
   cell blocked as body); the normal count is :func:`normal_post_count` (from the first
   cell). A boost uses one more cell than normal speed, so in an unchanged region its
   landing is exactly ``normal - 1``; the boost counts as "landing below normal" only when
   ``landing < normal - 1``, i.e. the second cell cuts the hero off from part of what
   the first cell reaches (the census pattern: a boost into a pocket while normal speed
   still had a way out). If the action is such a firing boost, switch to that direction
   at normal speed when it is masked-legal (``fallback_landing_switches``). A boost in a
   region the first cell does not split is NOT switched.
2. **Escape ranking.** v4's :class:`~src.evaluation.safety_veto_v4.EscapeSearch` (depth
   8, node budget 2000 for the whole decision, children straight/left/right) on
   :func:`~src.evaluation.safety_veto_v4.world_for`, judged per ACTION by
   :class:`BoostAwareEscapeSearch`: a normal-speed action (or a boost that does not
   fire) is v4's ``escape(direction)``; a firing boost must escape along its forced
   path, i.e. its first cell, then the second cell straight ahead entered at step 2 with
   the first cell on the path, and that second cell must not be next to another live
   snake's head (it is entered in the same frame as the first cell, so v4's step-1
   head-adjacency rule applies to both). If the (step-1) action escapes it is kept.
   Otherwise the other masked-legal actions are tried in descending Q order (ties: lowest
   index, as ``torch.argmax``), boosts only if their landing is not below normal as in
   step 1, and the first that escapes is chosen (``fallback_escape_switches``); this is
   the highest-Q escaping candidate. If the budget runs out at any point the result of
   step 1 stands (``fallback_budget_unknown``); if nothing escapes, too.
   The probe records ``no_spacious`` (no v2-spacious action existed) in every case.

Determinism: no randomness and no timing in any choice; the search order is fixed.

Records. ``record()`` is the descriptor plus exactly v2's seven counters, so
``strict_promotion._validate_candidate_wrapper_probe`` accepts it. The v6 counters
(``head_risky_vetoes``, ``head_risky_kept_no_alternative``, ``fallback_landing_switches``,
``fallback_escape_switches``, ``fallback_budget_unknown``, ``action_differs_from_v5``,
timing, and v5's own counters under ``"v5"``) come from :meth:`diagnostics_record`.

Default behavior is untouched: nothing installs this class unless a caller does so
explicitly with :func:`install_head_and_fallback_veto`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.core.game_config import GameConfig
from src.core.mechanics_constants import HEADON_SIZE_RATIO
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
    Cell,
    Grid,
    grid_for,
    static_blocked,
    tail_aware_reachable,
)
from src.evaluation.safety_veto_v4 import (
    BudgetExhausted,
    EscapeSearch,
    LookaheadWorld,
    turn,
    world_for,
)
from src.evaluation.safety_veto_v5 import (
    BoostAwareCounters,
    boost_aware_choice,
    landing_count,
    simulate_action,
)
from src.game.game_logic import GameLogic
from src.game.snake_state import FREE_SPACE_BFS_CAP, FREE_SPACE_MIN_CAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.game.snake import Snake

VETO_METHOD_V6 = "free-space-veto/v6-head-and-fallback"
BOOST_APPROXIMATION_V6 = "first-cell-and-two-cell-landing"
HEAD_RULE = (
    "opponent-head-reach-current-and-3-moves-plus-boost-second-cell/"
    "hero-wins-iff-mechanics-v2-and-post-burn-length-ge-1.15x-opponent-plus-1/"
    "veto-if-v5-eligible-non-risky-alternative"
)
FALLBACK_RULE = (
    "no-spacious-only/boost-landing-lt-normal-post-move-count-minus-1-to-normal-speed/"
    "v4-escape-search-depth-8-budget-2000-per-action-boost-forced-straight-second-cell-"
    "not-head-adjacent/highest-q-escaping/budget-unknown-no-change"
)
REPLACEMENT_RULE_V6 = (
    "v5-then-head-avoid-highest-q-v5-eligible-same-speed-mode-then-other/"
    "no-spacious-fallback-highest-q-escaping"
)
FALLBACK_DEPTH = 8
FALLBACK_NODE_BUDGET = 2000

# Decision reasons (diagnostics only; the probe records the v2-shaped outcome).
REASON_V5 = "v5"
REASON_HEAD_VETO = "head_veto"
REASON_HEAD_KEPT = "head_kept_no_alternative"
REASON_FALLBACK = "fallback"


def hero_wins_head_on(hero_length: int, opponent_max_length: int) -> bool:
    """Whether the game's head-on rule lets a snake of ``hero_length`` survive.

    Mechanics v2 only (``GameState.handle_collisions``): the survivor's logical length is
    ``>= HEADON_SIZE_RATIO`` times the other's. At v1 every head-on is mutual death.
    """
    if int(GameConfig.MECHANICS_VERSION) != 2:
        return False
    return max(1, int(hero_length)) >= HEADON_SIZE_RATIO * max(1, int(opponent_max_length))


def opponent_head_reach(
    snake: "Snake", other_snakes: Sequence["Snake"], grid: Grid
) -> Dict[Cell, int]:
    """Cell -> largest possible post-frame length of any opponent whose head reaches it.

    The reach is the opponent's current head cell, the first cell of each of its three
    relative moves and, if it can boost, the second cell along each. The length bound is
    ``_logical_length() + 1`` (one pellet of growth this frame).
    """
    threats: Dict[Cell, int] = {}
    for other in other_snakes:
        if other is snake or not getattr(other, "is_alive", False) or not other.segments:
            continue
        hx, hy = grid.to_cell(*other.segments[0])
        bound = int(other._logical_length()) + 1
        cells = [(hx, hy)]
        can_boost = int(other.length) >= int(GameConfig.MIN_BOOST_LENGTH)
        for relative in range(NUM_DIRECTIONS):
            dx, dy = GameLogic.relative_to_absolute_direction(tuple(other.direction), relative)
            cells.append((hx + dx, hy + dy))
            if can_boost:
                cells.append((hx + 2 * dx, hy + 2 * dy))
        for cell in cells:
            threats[cell] = max(threats.get(cell, 0), bound)
    return threats


def head_risky_actions(
    snake: "Snake", threats: Dict[Cell, int], grid: Grid
) -> Tuple[List[bool], List[bool]]:
    """Per action 0..5: ``(risky, waived)``.

    An action's traversed cells come from v5's exact one-frame movement model. A cell that
    some opponent can reach is a risk unless the hero (at this action's post-burn length)
    beats the largest such opponent under :func:`hero_wins_head_on`; ``waived[a]`` is True
    when ``a`` touches a reachable cell but every such risk was waived by that rule.
    Threat cells further than 2 (Manhattan) from the hero's head cannot be traversed, so
    when there are none nothing is simulated.
    """
    risky = [False] * NUM_ACTIONS
    waived = [False] * NUM_ACTIONS
    hx, hy = grid.to_cell(*snake.segments[0])
    if not any(abs(cx - hx) + abs(cy - hy) <= 2 for cx, cy in threats):
        return risky, waived
    for action in range(NUM_ACTIONS):
        move = simulate_action(snake, action)
        for x, y in move.traversed:
            bound = threats.get(grid.to_cell(x, y))
            if bound is None:
                continue
            if hero_wins_head_on(move.length, bound):
                waived[action] = True
                continue
            risky[action] = True
        if risky[action]:
            waived[action] = False
    return risky, waived


def _highest_q(q_values: Sequence[float], candidates: Sequence[int]) -> Optional[int]:
    """Highest Q among ``candidates`` (ascending), ties to the lowest index."""
    best: Optional[int] = None
    for action in candidates:
        if best is None or float(q_values[action]) > float(q_values[best]):
            best = action
    return best


def head_avoid_choice(
    q_values: Sequence[float],
    eligible: Sequence[bool],
    risky: Sequence[bool],
    current: int,
) -> Tuple[int, str]:
    """Layer (a) on one decision (pure). Returns ``(action, reason)``.

    ``eligible`` is v5-eligibility per action (masked-legal, v2-spacious, boost landing
    ok); ``current`` is v5's choice. A head-risky ``current`` is replaced by the
    highest-Q eligible non-risky other action in its speed mode, else the other mode.
    """
    if len(q_values) != NUM_ACTIONS or len(eligible) != NUM_ACTIONS or len(risky) != NUM_ACTIONS:
        raise ValueError(f"expected {NUM_ACTIONS} Q-values, eligibility and risk flags")
    current = int(current)
    if not risky[current]:
        return current, REASON_V5
    options = [a for a in range(NUM_ACTIONS) if a != current and eligible[a] and not risky[a]]
    if not options:
        return current, REASON_HEAD_KEPT
    boost = current >= NUM_DIRECTIONS
    same_mode = [a for a in options if (a >= NUM_DIRECTIONS) == boost]
    best = _highest_q(q_values, same_mode or options)
    assert best is not None
    return best, REASON_HEAD_VETO


@dataclass(frozen=True)
class FallbackResult:
    """Layer (b) on one decision: the action and what happened."""

    action: int
    landing_switch: bool = False
    escape_switch: bool = False
    budget_unknown: bool = False
    searched: bool = False


def normal_post_count(
    snake: "Snake", other_snakes: Sequence["Snake"], direction: int, cap: int
) -> int:
    """Reachable cells from ``direction``'s normal-speed new head (at most ``cap``).

    Same model as v5's :func:`~src.evaluation.safety_veto_v5.landing_count`: the hero's
    body after the one-frame move (``segments[1:]``) and the other live snakes are walls,
    v2's static capped flood from the new head. 0 when the first cell is off the board.
    """
    move = simulate_action(snake, int(direction))
    grid = grid_for(snake)
    blocked = static_blocked(snake, other_snakes, grid)
    blocked.update(grid.to_cell(x, y) for x, y in move.segments[1:])
    start = grid.to_cell(*move.segments[0])
    return tail_aware_reachable(start, 0, blocked, {}, grid.in_bounds, int(cap))


def landing_below_normal_rule(landing: Optional[int], normal: int) -> bool:
    """Layer (b)'s landing test: a firing boost (``landing`` not ``None``) whose landing
    count is below ``normal - 1`` (a boost always uses one more cell than normal speed)."""
    return landing is not None and int(landing) < int(normal) - 1


class BoostAwareEscapeSearch(EscapeSearch):
    """v4's :class:`EscapeSearch` judged per action (0..5), with a boost's forced path.

    ``escape_action(a)`` for a normal-speed action (or any action when ``boost_fires`` is
    False, since such a boost moves one cell) is v4's ``escape(a % 3)``. A firing boost
    enters its first cell F (v4's step-1 rule) and, in the same frame, the next cell S
    straight ahead: S must not be next to another live snake's head and must be passable
    at step 2 with F on the path; the search then continues from S at step 2 exactly as
    v4 does (same node budget and failed-subtree memo, so a boost's subtree is the same
    as normal speed's straight child at F).
    """

    def __init__(self, world: LookaheadWorld, depth: int, budget: int, boost_fires: bool):
        super().__init__(world, depth, budget)
        self.boost_fires = bool(boost_fires)

    def escape_action(self, action: int) -> bool:
        """Whether ``action`` (0..5) has an escape; may raise :class:`BudgetExhausted`."""
        action = int(action)
        direction = action % NUM_DIRECTIONS
        if action < NUM_DIRECTIONS or not self.boost_fires:
            return self.escape(direction)
        world = self.world
        heading = turn(world.heading, direction)
        first = (world.head[0] + heading[0], world.head[1] + heading[1])
        if not self.passable(first, 1):
            return False
        second = (first[0] + heading[0], first[1] + heading[1])
        self._path.append(first)
        self._on_path.add(first)
        try:
            if second in world.head_adjacent or not self.passable(second, 2):
                return False
            return self._visit(second, heading, 2)
        finally:
            self._path.pop()
            self._on_path.discard(first)


def fallback_choice(
    q_values: Sequence[float],
    mask: Sequence[bool],
    current: int,
    landing_below_normal: Callable[[int], bool],
    escape: Optional[Callable[[int], bool]],
) -> FallbackResult:
    """Layer (b) on one ``no_spacious`` decision (pure given its two callables).

    ``landing_below_normal(d)`` says whether boost direction ``d`` fires and lands on fewer
    than ``normal - 1`` cells (:func:`landing_below_normal_rule`). ``escape(a)`` is the
    shared per-ACTION escape search (:meth:`BoostAwareEscapeSearch.escape_action`; it may
    raise :class:`BudgetExhausted`); ``None`` skips step 2. Candidates are tried in
    descending Q (ties: lowest index), so the first that escapes is the highest-Q one.
    """
    mask = [bool(value) for value in mask]
    action = int(current)
    landing_switch = False
    direction = action % NUM_DIRECTIONS
    if action >= NUM_DIRECTIONS and mask[direction] and landing_below_normal(direction):
        action, landing_switch = direction, True
    if escape is None:
        return FallbackResult(action, landing_switch)
    try:
        if escape(action):
            return FallbackResult(action, landing_switch, searched=True)
        candidates = [
            a
            for a in range(NUM_ACTIONS)
            if a != action
            and mask[a]
            and (a < NUM_DIRECTIONS or not landing_below_normal(a % NUM_DIRECTIONS))
        ]
        candidates.sort(key=lambda a: (-float(q_values[a]), a))
        for candidate in candidates:
            if escape(candidate):
                return FallbackResult(candidate, landing_switch, escape_switch=True, searched=True)
    except BudgetExhausted:
        return FallbackResult(action, landing_switch, budget_unknown=True, searched=True)
    return FallbackResult(action, landing_switch, searched=True)


@dataclass
class HeadAndFallbackCounters:
    """Per-episode v6 bookkeeping (reported beside the probe; never gates).

    Identities: ``decisions == kept + vetoes_applied + no_spacious`` (the probe's
    ``kept_base``/``vetoes_applied``/``fallback_no_spacious``);
    ``head_risky_decisions == head_risky_vetoes + head_risky_kept_no_alternative``;
    ``kept == v5.kept - head_vetoes_of_v5_kept`` and ``vetoes_applied ==
    v5.vetoes_applied + head_vetoes_of_v5_kept``;
    ``fallback_decisions == no_spacious``; ``fallback_searches ==
    fallback_decisions - fallback_skipped_no_search``. ``action_differs_from_v5`` compares
    with the released v5 rule on the same inputs. ``*_seconds_*`` are wall-clock.
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
    head_vetoes_of_v5_kept: int = 0
    fallback_decisions: int = 0
    fallback_landing_switches: int = 0
    fallback_escape_switches: int = 0
    fallback_budget_unknown: int = 0
    fallback_searches: int = 0
    fallback_skipped_no_search: int = 0
    fallback_search_nodes: int = 0
    action_differs_from_v5: int = 0
    apply_seconds_total: float = 0.0
    apply_seconds_max: float = 0.0
    fallback_seconds_total: float = 0.0
    fallback_seconds_max: float = 0.0
    v5: BoostAwareCounters = field(default_factory=BoostAwareCounters)

    def count_outcome(self, outcome: str) -> None:
        field_name = {
            OUTCOME_KEPT: "kept",
            OUTCOME_VETOED: "vetoes_applied",
            OUTCOME_NO_SPACIOUS: "no_spacious",
        }.get(outcome)
        if field_name is None:
            raise ValueError(f"unknown veto outcome {outcome!r}")
        self.decisions += 1
        setattr(self, field_name, getattr(self, field_name) + 1)

    def add_time(self, prefix: str, seconds: float) -> None:
        total, peak = f"{prefix}_seconds_total", f"{prefix}_seconds_max"
        setattr(self, total, getattr(self, total) + float(seconds))
        setattr(self, peak, max(getattr(self, peak), float(seconds)))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping (v5's counters nested under ``"v5"``) plus mean costs."""
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key == "v5":
                continue
            out[key] = float(value) if isinstance(value, float) else int(value)
        out["mean_apply_seconds"] = (
            self.apply_seconds_total / self.decisions if self.decisions else None
        )
        out["mean_fallback_seconds"] = (
            self.fallback_seconds_total / self.fallback_decisions
            if self.fallback_decisions
            else None
        )
        out["v5"] = self.v5.to_dict()
        return out


class HeadAndFallbackVeto:
    """Stateful per-snake veto hook (same surface as ``FreeSpaceVeto``).

    ``head_avoidance`` and ``fallback`` switch the two layers (both True in v6; the
    descriptor records them, so a run with a layer off is a different identity).
    """

    method = VETO_METHOD_V6

    def __init__(
        self,
        head_avoidance: bool = True,
        fallback: bool = True,
        depth: int = FALLBACK_DEPTH,
        node_budget: int = FALLBACK_NODE_BUDGET,
    ) -> None:
        if int(depth) < 1 or int(node_budget) < 1:
            raise ValueError("depth and node_budget must be at least 1")
        self.head_avoidance = bool(head_avoidance)
        self.fallback = bool(fallback)
        self.depth = int(depth)
        self.node_budget = int(node_budget)
        self.counters = SafetyVetoCounters()
        self.v6 = HeadAndFallbackCounters()

    def reset(self) -> None:
        """Zero all counters (call at episode start)."""
        self.counters = SafetyVetoCounters()
        self.v6 = HeadAndFallbackCounters()

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
        action, outcome = self.decide(
            snake,
            list(other_snakes),
            _as_list(q_values),
            [bool(x) for x in _as_list(action_mask)],
            int(base_action),
        )
        self.counters.record(int(base_action), action, outcome)
        self.v6.count_outcome(outcome)
        self.v6.add_time("apply", time.perf_counter() - started)
        return action

    def decide(
        self,
        snake: "Snake",
        others: Sequence["Snake"],
        q: Sequence[float],
        mask: Sequence[bool],
        base: int,
    ) -> Tuple[int, str]:
        """``(action, probe outcome)``; updates the v6 and nested v5 diagnostics."""
        if len(q) != NUM_ACTIONS or len(mask) != NUM_ACTIONS:
            raise ValueError(f"expected {NUM_ACTIONS} Q-values and mask entries")
        started = time.perf_counter()
        features = snake._get_free_space_features(list(others))
        cap, need = free_space_threshold(snake.length, snake._logical_length())
        spacious = spacious_directions(features, cap, need)
        counts: Dict[int, Optional[int]] = {}
        tally = {"checks": 0, "failures": 0}

        def landing(direction: int) -> Optional[int]:
            if direction not in counts:
                counts[direction] = landing_count(snake, others, direction, cap)
                tally["checks"] += int(counts[direction] is not None)
                tally["failures"] += int(
                    counts[direction] is not None and counts[direction] < need  # type: ignore
                )
            return counts[direction]

        def landing_ok(direction: int) -> bool:
            count = landing(direction)
            return count is None or count >= need

        v5_action, v5_outcome, v5_reason = boost_aware_choice(q, mask, spacious, base, landing_ok)
        v2_action, _ = veto_choice(q, mask, spacious, base)
        self.v6.v5.record(
            base,
            v5_outcome,
            v5_reason,
            v5_action != v2_action,
            tally["checks"],
            tally["failures"],
            time.perf_counter() - started,
        )
        action, outcome = v5_action, v5_outcome
        if v5_outcome == OUTCOME_NO_SPACIOUS:
            if self.fallback:
                action = self._fallback(snake, others, q, mask, v5_action, cap, landing)
        elif self.head_avoidance:
            action = self._head_layer(snake, others, q, mask, spacious, v5_action, landing_ok)
            if action != v5_action:
                self.v6.head_vetoes_of_v5_kept += int(v5_outcome == OUTCOME_KEPT)
                outcome = OUTCOME_VETOED
        self.v6.action_differs_from_v5 += int(action != v5_action)
        return action, outcome

    def _head_layer(
        self,
        snake: "Snake",
        others: Sequence["Snake"],
        q: Sequence[float],
        mask: Sequence[bool],
        spacious: Sequence[bool],
        current: int,
        landing_ok: Callable[[int], bool],
    ) -> int:
        """Layer (a): avoid cells an unbeaten opponent head can reach next frame."""
        grid = grid_for(snake)
        threats = opponent_head_reach(snake, others, grid)
        self.v6.head_checks += 1
        risky, waived = head_risky_actions(snake, threats, grid)
        if waived[current]:
            self.v6.head_risk_waived_hero_wins += 1
        if not risky[current]:
            return current
        self.v6.head_risky_decisions += 1
        eligible = [
            bool(mask[a])
            and bool(spacious[a % NUM_DIRECTIONS])
            and not risky[a]  # checked first: only non-risky boosts need a landing flood
            and (a < NUM_DIRECTIONS or landing_ok(a % NUM_DIRECTIONS))
            for a in range(NUM_ACTIONS)
        ]
        action, reason = head_avoid_choice(q, eligible, risky, current)
        if reason == REASON_HEAD_VETO:
            self.v6.head_risky_vetoes += 1
            if (action >= NUM_DIRECTIONS) != (current >= NUM_DIRECTIONS):
                self.v6.head_vetoes_speed_switched += 1
        else:
            self.v6.head_risky_kept_no_alternative += 1
        return action

    def _fallback(
        self,
        snake: "Snake",
        others: Sequence["Snake"],
        q: Sequence[float],
        mask: Sequence[bool],
        current: int,
        cap: int,
        landing: Callable[[int], Optional[int]],
    ) -> int:
        """Layer (b): the no-spacious landing switch, then the budgeted escape ranking."""
        started = time.perf_counter()
        self.v6.fallback_decisions += 1
        search: List[BoostAwareEscapeSearch] = []
        normal: Dict[int, int] = {}

        def landing_below_normal(direction: int) -> bool:
            count = landing(direction)
            if count is None:
                return False
            if direction not in normal:
                normal[direction] = normal_post_count(snake, others, direction, cap)
            return landing_below_normal_rule(count, normal[direction])

        def escape(action: int) -> bool:
            if not search:
                fires = int(snake.length) >= int(GameConfig.MIN_BOOST_LENGTH)
                world = world_for(snake, others)
                search.append(BoostAwareEscapeSearch(world, self.depth, self.node_budget, fires))
            return search[0].escape_action(action)

        result = fallback_choice(q, mask, current, landing_below_normal, escape)
        self.v6.fallback_landing_switches += int(result.landing_switch)
        self.v6.fallback_escape_switches += int(result.escape_switch)
        self.v6.fallback_budget_unknown += int(result.budget_unknown)
        self.v6.fallback_searches += int(result.searched)
        self.v6.fallback_skipped_no_search += int(not result.searched)
        self.v6.fallback_search_nodes += search[0].nodes if search else 0
        self.v6.add_time("fallback", time.perf_counter() - started)
        return result.action

    def diagnostics_record(self) -> Dict[str, Any]:
        """The v6 counters (``HeadAndFallbackCounters.to_dict``), stored beside the probe."""
        return self.v6.to_dict()

    def descriptor(self) -> Dict[str, Any]:
        """Static identity of this veto (no counters)."""
        return {
            "method": VETO_METHOD_V6,
            "free_space_bfs_cap": FREE_SPACE_BFS_CAP,
            "free_space_min_cap": FREE_SPACE_MIN_CAP,
            "boost_approximation": BOOST_APPROXIMATION_V6,
            "replacement_rule": REPLACEMENT_RULE_V6,
            "head_rule": HEAD_RULE,
            "fallback_rule": FALLBACK_RULE,
            "head_avoidance": self.head_avoidance,
            "no_spacious_fallback": self.fallback,
            "fallback_depth": self.depth,
            "fallback_node_budget": self.node_budget,
        }

    def record(self) -> Dict[str, Any]:
        """Descriptor plus v2's seven counters, for rollout probes."""
        return {**self.descriptor(), "counters": self.counters.to_dict()}


def _as_list(values: Any) -> List[Any]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    return list(values)


def install_head_and_fallback_veto(snake: Any) -> HeadAndFallbackVeto:
    """Attach a fresh :class:`HeadAndFallbackVeto` (both layers on) to an AISnake."""
    if not hasattr(snake, "safety_veto"):
        raise TypeError("safety veto can only be installed on an AISnake")
    veto = HeadAndFallbackVeto()
    snake.safety_veto = veto
    return veto
