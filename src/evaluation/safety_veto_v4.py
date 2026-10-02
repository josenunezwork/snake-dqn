"""Opt-in serving-time free-space veto, v4: bounded multi-step look-ahead.

This module adds a NEW veto, ``free-space-veto/v4-lookahead``. It changes neither
:mod:`src.evaluation.safety_veto` (v2, whose bytes the strict receipt binds) nor
:mod:`src.evaluation.safety_veto_v3` (retired). It reuses their pure pieces unchanged:
v2's cap/need formulas, :func:`~src.evaluation.safety_veto.veto_choice` (the
speed-preserving replacement rule, which never selects a masked action) and
:class:`~src.evaluation.safety_veto.SafetyVetoCounters` (the seven-counter probe shape
``strict_promotion`` validates), and v3's grid, own-body release model and tail-aware
reachable count.

Why: in the v3 screen (``research/apex_veto_v3_screen_20261001``) the remaining
self-collision deaths happened after the safe-and-spacious set was already empty. The
trap was entered several moves earlier, while every one-step reachability count still
looked large enough. A count measures area, not whether the snake can traverse it: a
1-wide dead-end corridor of ``need + 3`` cells passes a one-step check and is fatal.

Rule, per greedy decision (``base`` = the policy's masked argmax, ``d = base % 3``):

1. **Trigger.** Read v2's one-step count for ``d`` (``round(feature * cap)`` from the
   snake's own ``_get_free_space_features``). If it is ``>= min(trigger_factor * need,
   cap)`` the base action is kept without a search. The ``cap`` bound is needed because
   the count saturates at ``cap``: for a snake of length >= ``cap / 2`` the literal
   ``2 * need`` exceeds every possible count and would trigger on every decision.
2. **Escape search** for ``d`` (and, only if ``d`` has none, for each other direction
   with a masked-legal action). A direction has an *escape* when some sequence of the
   snake's own future moves (three relative turns per step, one cell per step) starting
   with that direction stays passable for ``depth`` steps (default 8) and, at the leaf,
   v3's tail-aware reachable count from the leaf cell is ``>= need``. Passability of a
   cell at step ``k``: inside the arena; not a cell of another live snake (static, as in
   v2/v3); not a cell this search path already entered (the path stays blocked for the
   rest of the search, conservative: it would release only after ``length`` more moves);
   own body cells only once released (v3's model: segment ``j`` vacates after
   ``length - j + slack`` moves, ``slack`` default 1 absorbs one pellet of growth); and
   at step 1 only, not 4-adjacent to another live snake's head (conservative: that
   snake may move there now). The leaf count blocks the whole path except the leaf.
3. **Veto.** If ``d`` has an escape the base action is kept. Otherwise v2's
   :func:`veto_choice` runs with ``spacious := escape``: the highest-Q masked-legal
   escaping action in the base action's speed mode, else in the other mode.
4. **v2 fallback.** If no masked-legal action escapes (``fallback_no_escape``) or the
   node budget runs out (``budget_exhausted``), the decision is exactly the released
   v2 veto's: :func:`veto_choice` with v2's own one-step ``spacious_directions`` (the
   features read in step 1). So v4 never keeps a base action that v2 would veto in an
   untriggered or fallback state. (Untriggered: ``need <= cap``, so the base count is
   ``>= need`` and v2 keeps it too.) Without this fallback, a direction that is
   v2-spacious but fails v4's escape test (its step-1 cell next to another snake's
   head, or a pocket in ``[need, need + depth)`` under the prune) let v4 keep a
   certain-death base that v2 vetoes.

Why the count is applied at the leaf only: applying "count >= need" at depth 1 would
accept every direction v3 accepts, so v4 could never veto a move v3 allows. Requiring a
traversable ``depth``-step continuation into a spacious leaf is what makes v4 stricter
than v2/v3 on dead ends and spirals; on an open board nothing triggers and v4 equals v2.

Boost: a boost action moves two cells. It is scored by its direction's escape, i.e. its
first step is the search's first step (the second cell is one of the straight children
the search may or may not use). The action mask already rules out an immediate two-cell
collision.

Bounded cost. The search is a depth-first search with a failed-subtree memo keyed by
``(cell, heading, step, path cells)``, a bound prune at every node (a node at step ``k``
is abandoned when its tail-aware count is below ``need + depth - k``; see
:class:`EscapeSearch`) and a per-decision node budget (default 4000 nodes across all
directions searched in that decision). Each count is v3's breadth-first search stopped
at the required number of cells. **Budget fallback (deterministic):** if the budget runs
out at any point in a decision, the v2 fallback of step 4 decides and ``budget_exhausted``
is counted. Children are tried in the fixed order straight,
left, right, so results never depend on timing.

Records. ``record()`` is the descriptor plus exactly v2's seven counters, so
``strict_promotion._validate_candidate_wrapper_probe`` accepts it. The probe counts the
action actually taken: v4's escape vetoes plus the v2 fallback's vetoes in
``vetoes_applied``, and the v2 fallback's no-spacious decisions in
``fallback_no_spacious``. The v4 counters (decisions, searches, the five reason
outcomes, the v2 fallback's kept/vetoed/no-spacious split, node and time totals) are
returned by :meth:`LookaheadFreeSpaceVeto.diagnostics_record`, which a
screen stores beside the record (the timing fields are wall-clock and never gate).

Default behavior is untouched: nothing installs this class unless a caller does so
explicitly with :func:`install_lookahead_veto`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

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
    REACHABILITY_RULE,
    TAIL_RELEASE_SLACK,
    Cell,
    Grid,
    grid_for,
    own_body_release,
    static_blocked,
    tail_aware_reachable,
)
from src.game.snake_state import FREE_SPACE_BFS_CAP, FREE_SPACE_MIN_CAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.game.snake import Snake

VETO_METHOD_V4 = "free-space-veto/v4-lookahead"
DEFAULT_DEPTH = 8
DEFAULT_NODE_BUDGET = 4000
DEFAULT_TRIGGER_FACTOR = 2
LOOKAHEAD_RULE = (
    "dfs-own-moves/leaf-tail-aware-count-ge-need/path-static/"
    "other-snakes-static/other-heads-adjacent-blocked-step-1"
)
BOOST_APPROXIMATION = "direction-first-step-lookahead"
BUDGET_FALLBACK = "v2-veto"
NO_ESCAPE_FALLBACK = "v2-veto"
# Up, right, down, left (GameLogic.relative_to_absolute_direction's ordering).
CARDINAL: Tuple[Cell, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
CHILD_ORDER = (1, 0, 2)  # straight, left, right: fixed, so the search is deterministic


def turn(heading: Cell, relative: int) -> Cell:
    """Absolute heading after a relative move (0 = left/ccw, 1 = straight, 2 = right/cw).

    Same mapping as ``GameLogic.relative_to_absolute_direction``; a non-cardinal heading
    is kept unchanged, as there.
    """
    try:
        index = CARDINAL.index(tuple(heading))
    except ValueError:
        return tuple(heading)  # type: ignore[return-value]
    return CARDINAL[(index + int(relative) - 1) % 4]


@dataclass(frozen=True)
class LookaheadWorld:
    """Everything the escape search reads for one decision (built once per decision)."""

    grid: Grid
    head: Cell
    heading: Cell
    blocked: FrozenSet[Cell]
    head_adjacent: FrozenSet[Cell]
    release: Mapping[Cell, int]
    need: int


def world_for(
    snake: "Snake", other_snakes: Sequence["Snake"], slack: int = TAIL_RELEASE_SLACK
) -> LookaheadWorld:
    """Build the search world: v3's grid, statics and release map plus head adjacency."""
    grid = grid_for(snake)
    blocked = frozenset(static_blocked(snake, other_snakes, grid))
    adjacent = set()
    for other in other_snakes:
        if other is snake or not other.is_alive or not other.segments:
            continue
        hx, hy = grid.to_cell(*other.segments[0])
        adjacent.update(((hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)))
    _, need = free_space_threshold(snake.length, snake._logical_length())
    return LookaheadWorld(
        grid=grid,
        head=grid.to_cell(*snake.head),
        heading=tuple(snake.direction),  # type: ignore[arg-type]
        blocked=blocked,
        head_adjacent=frozenset(adjacent),
        release=own_body_release(snake.segments, int(snake.length), grid, slack),
        need=int(need),
    )


class BudgetExhausted(Exception):
    """Raised inside a search when the per-decision node budget is spent."""


class _Either:
    """Membership in either of two sets, without building their union per leaf."""

    __slots__ = ("first", "second")

    def __init__(self, first: FrozenSet[Cell], second: set) -> None:
        self.first = first
        self.second = second

    def __contains__(self, cell: object) -> bool:
        return cell in self.first or cell in self.second


class EscapeSearch:
    """Bounded depth-first escape search for one decision (shared node budget).

    ``escape(direction)`` returns whether that relative direction has an escape (module
    docstring) and raises :class:`BudgetExhausted` once more than ``budget`` nodes have
    been visited in this decision. A node is one passable cell on a search path (the
    leaf included); failed-subtree memo hits do not count.

    Every node at step ``k`` first runs v3's tail-aware count from its cell (the path
    before it blocked), stopped at ``need + depth - k`` cells, and is abandoned when the
    count is below that. At the leaf (``k = depth``) this is exactly the escape test
    ``count >= need``. Above the leaf it is a bound prune: the remaining ``depth - k``
    path cells are all inside the node's reachable set and all blocked for the leaf
    count, so with static occupancy no leaf below can reach ``need``. (With time-indexed
    tail release the bound is close but not a proof, since v3's breadth-first count
    admits a released cell only on first arrival; the prune is part of the defined rule
    and is deterministic either way.) A large pocket that is still too small is thus
    rejected after one count, without enumerating its paths.
    """

    def __init__(self, world: LookaheadWorld, depth: int, budget: int) -> None:
        if int(depth) < 1:
            raise ValueError("depth must be at least 1")
        if int(budget) < 1:
            raise ValueError("budget must be at least 1")
        self.world = world
        self.depth = int(depth)
        self.budget = int(budget)
        self.nodes = 0
        self.counts = 0
        self._failed: set = set()
        self._path: List[Cell] = []
        self._on_path: set = set()

    def passable(self, cell: Cell, step: int) -> bool:
        """Whether the head may enter ``cell`` on move ``step`` of the current path."""
        world = self.world
        if cell in world.blocked or cell in self._on_path or not world.grid.in_bounds(cell):
            return False
        if step == 1 and cell in world.head_adjacent:
            return False
        when = world.release.get(cell)
        return when is None or step >= when

    def escape(self, direction: int) -> bool:
        """Escape exists for relative ``direction`` (0 = left, 1 = straight, 2 = right)."""
        world = self.world
        heading = turn(world.heading, direction)
        first = (world.head[0] + heading[0], world.head[1] + heading[1])
        if not self.passable(first, 1):
            return False
        return self._visit(first, heading, 1)

    def reachable(self, cell: Cell, step: int, limit: int) -> int:
        """v3 tail-aware count from ``cell`` at ``step``, path blocked, at most ``limit``."""
        self.counts += 1
        world = self.world
        return tail_aware_reachable(
            cell,
            step,
            _Either(world.blocked, self._on_path),  # type: ignore[arg-type]
            world.release,  # type: ignore[arg-type]
            world.grid.in_bounds,
            limit,
        )

    def _visit(self, cell: Cell, heading: Cell, step: int) -> bool:
        key = (cell, heading, step, frozenset(self._on_path))
        if key in self._failed:
            return False
        self.nodes += 1
        if self.nodes > self.budget:
            raise BudgetExhausted
        required = self.world.need + self.depth - step
        if self.reachable(cell, step, required) < required:
            self._failed.add(key)
            return False
        if step >= self.depth:
            return True
        self._path.append(cell)
        self._on_path.add(cell)
        try:
            for relative in CHILD_ORDER:
                child_heading = turn(heading, relative)
                child = (cell[0] + child_heading[0], cell[1] + child_heading[1])
                if self.passable(child, step + 1) and self._visit(child, child_heading, step + 1):
                    return True
        finally:
            self._path.pop()
            self._on_path.discard(cell)
        self._failed.add(key)
        return False


OUTCOME_KEPT_UNTRIGGERED = "kept_untriggered"
OUTCOME_KEPT_ESCAPE = "kept_escape"
OUTCOME_BUDGET_EXHAUSTED = "budget_exhausted"
OUTCOME_FALLBACK_NO_ESCAPE = "fallback_no_escape"
# v4 outcome -> the v2 outcome recorded in the seven-counter probe. The two fallback
# outcomes are absent: there the v2 veto decides and its own outcome is recorded.
V2_OUTCOME = {
    OUTCOME_KEPT_UNTRIGGERED: OUTCOME_KEPT,
    OUTCOME_KEPT_ESCAPE: OUTCOME_KEPT,
    OUTCOME_VETOED: OUTCOME_VETOED,
}
FALLBACK_OUTCOMES = (OUTCOME_BUDGET_EXHAUSTED, OUTCOME_FALLBACK_NO_ESCAPE)
# v2 outcome of a fallback decision -> the v4 counter that splits it.
FALLBACK_V2_FIELD = {
    OUTCOME_KEPT: "fallback_v2_kept",
    OUTCOME_VETOED: "fallback_v2_vetoes",
    OUTCOME_NO_SPACIOUS: "fallback_v2_no_spacious",
}


@dataclass
class LookaheadCounters:
    """Per-episode v4 bookkeeping (reported beside the probe; never gates).

    ``decisions == kept_untriggered + kept_escape + vetoes_applied + budget_exhausted +
    fallback_no_escape``, ``searches == decisions - kept_untriggered`` and
    ``budget_exhausted + fallback_no_escape == fallback_v2_kept + fallback_v2_vetoes +
    fallback_v2_no_spacious`` always hold. Against the probe's v2 counters:
    ``kept_base == kept_untriggered + kept_escape + fallback_v2_kept``,
    ``vetoes_applied (probe) == vetoes_applied + fallback_v2_vetoes`` and
    ``fallback_no_spacious == fallback_v2_no_spacious``.
    ``nodes_total``/``counts_total``/``nodes_max`` are deterministic; the ``*_seconds``
    fields are wall-clock (``time.perf_counter``) and vary between runs.
    """

    decisions: int = 0
    searches: int = 0
    kept_untriggered: int = 0
    kept_escape: int = 0
    vetoes_applied: int = 0
    budget_exhausted: int = 0
    fallback_no_escape: int = 0
    fallback_v2_kept: int = 0
    fallback_v2_vetoes: int = 0
    fallback_v2_no_spacious: int = 0
    directions_searched: int = 0
    nodes_total: int = 0
    counts_total: int = 0
    nodes_max: int = 0
    apply_seconds_total: float = 0.0
    search_seconds_total: float = 0.0
    search_seconds_max: float = 0.0

    def record(
        self,
        outcome: str,
        v2_outcome: str,
        search: Optional[EscapeSearch],
        seconds: float,
    ) -> None:
        """Count one decision; ``search`` is ``None`` when the trigger did not fire.

        ``v2_outcome`` is the outcome recorded in the probe: fixed by ``V2_OUTCOME`` for
        the non-fallback outcomes, the v2 veto's own outcome for the fallbacks.
        """
        if outcome in FALLBACK_OUTCOMES:
            if v2_outcome not in FALLBACK_V2_FIELD:
                raise ValueError(f"unknown v2 outcome {v2_outcome!r}")
            field = FALLBACK_V2_FIELD[v2_outcome]
            setattr(self, field, getattr(self, field) + 1)
        elif outcome not in V2_OUTCOME:
            raise ValueError(f"unknown v4 outcome {outcome!r}")
        elif V2_OUTCOME[outcome] != v2_outcome:
            raise ValueError(f"v4 outcome {outcome!r} must record v2 {V2_OUTCOME[outcome]!r}")
        self.decisions += 1
        name = "vetoes_applied" if outcome == OUTCOME_VETOED else outcome
        setattr(self, name, getattr(self, name) + 1)
        if search is not None:
            self.searches += 1
            self.nodes_total += search.nodes
            self.counts_total += search.counts
            self.nodes_max = max(self.nodes_max, search.nodes)
            self.search_seconds_total += seconds
            self.search_seconds_max = max(self.search_seconds_max, seconds)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe mapping plus the mean per-decision and per-search wall cost."""
        out: Dict[str, Any] = {
            key: (float(value) if isinstance(value, float) else int(value))
            for key, value in self.__dict__.items()
        }
        out["mean_apply_seconds"] = (
            self.apply_seconds_total / self.decisions if self.decisions else None
        )
        out["mean_search_seconds"] = (
            self.search_seconds_total / self.searches if self.searches else None
        )
        return out


class LookaheadFreeSpaceVeto:
    """Stateful per-snake veto hook (same surface as ``FreeSpaceVeto``)."""

    method = VETO_METHOD_V4

    def __init__(
        self,
        *,
        depth: int = DEFAULT_DEPTH,
        node_budget: int = DEFAULT_NODE_BUDGET,
        trigger_factor: int = DEFAULT_TRIGGER_FACTOR,
        slack: int = TAIL_RELEASE_SLACK,
    ) -> None:
        if int(depth) < 1 or int(node_budget) < 1:
            raise ValueError("depth and node_budget must be at least 1")
        if int(trigger_factor) < 1 or int(slack) < 0:
            raise ValueError("trigger_factor must be >= 1 and slack >= 0")
        self.depth = int(depth)
        self.node_budget = int(node_budget)
        self.trigger_factor = int(trigger_factor)
        self.slack = int(slack)
        self.counters = SafetyVetoCounters()
        self.lookahead = LookaheadCounters()

    def reset(self) -> None:
        """Zero all counters (call at episode start)."""
        self.counters = SafetyVetoCounters()
        self.lookahead = LookaheadCounters()

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
        if len(q) != NUM_ACTIONS or len(mask) != NUM_ACTIONS:
            raise ValueError(f"expected {NUM_ACTIONS} Q-values and mask entries")
        base = int(base_action)
        action, outcome, v2_outcome, search, seconds = self.decide(snake, others, q, mask, base)
        self.counters.record(base, action, v2_outcome)
        self.lookahead.record(outcome, v2_outcome, search, seconds)
        self.lookahead.apply_seconds_total += time.perf_counter() - started
        return action

    def trigger_threshold(self, cap: int, need: int) -> int:
        """Base one-step counts below this run the search (``min(factor * need, cap)``)."""
        return min(self.trigger_factor * int(need), int(cap))

    def decide(
        self,
        snake: "Snake",
        others: Sequence["Snake"],
        q: Sequence[float],
        mask: Sequence[bool],
        base: int,
    ) -> Tuple[int, str, str, Optional[EscapeSearch], float]:
        """``(action, v4 outcome, probe v2 outcome, search or None, search seconds)``."""
        cap, need = free_space_threshold(snake.length, snake._logical_length())
        base_direction = base % NUM_DIRECTIONS
        features = snake._get_free_space_features(list(others))
        if round(float(features[base_direction]) * cap) >= self.trigger_threshold(cap, need):
            return base, OUTCOME_KEPT_UNTRIGGERED, OUTCOME_KEPT, None, 0.0
        started = time.perf_counter()
        search = EscapeSearch(world_for(snake, others, self.slack), self.depth, self.node_budget)
        try:
            action, outcome = self._search_choice(search, q, mask, base)
        except BudgetExhausted:
            action, outcome = base, OUTCOME_BUDGET_EXHAUSTED
        if outcome in FALLBACK_OUTCOMES:
            spacious = spacious_directions(features, cap, need)
            action, v2_outcome = veto_choice(q, mask, spacious, base)
        else:
            v2_outcome = V2_OUTCOME[outcome]
        return action, outcome, v2_outcome, search, time.perf_counter() - started

    def _search_choice(
        self, search: EscapeSearch, q: Sequence[float], mask: Sequence[bool], base: int
    ) -> Tuple[int, str]:
        base_direction = base % NUM_DIRECTIONS
        self.lookahead.directions_searched += 1
        if search.escape(base_direction):
            return base, OUTCOME_KEPT_ESCAPE
        escape = [False] * NUM_DIRECTIONS
        for direction in range(NUM_DIRECTIONS):
            legal = mask[direction] or mask[direction + NUM_DIRECTIONS]
            if direction == base_direction or not legal:
                continue
            self.lookahead.directions_searched += 1
            escape[direction] = search.escape(direction)
        action, outcome = veto_choice(q, mask, escape, base)
        if outcome == OUTCOME_NO_SPACIOUS:
            return base, OUTCOME_FALLBACK_NO_ESCAPE  # decide() hands this to the v2 veto
        return action, OUTCOME_VETOED

    def diagnostics_record(self) -> Dict[str, Any]:
        """The v4 counters (``LookaheadCounters.to_dict``), stored beside the probe."""
        return self.lookahead.to_dict()

    def descriptor(self) -> Dict[str, Any]:
        """Static identity of this veto (no counters)."""
        return {
            "method": VETO_METHOD_V4,
            "free_space_bfs_cap": FREE_SPACE_BFS_CAP,
            "free_space_min_cap": FREE_SPACE_MIN_CAP,
            "boost_approximation": BOOST_APPROXIMATION,
            "replacement_rule": "highest-q-eligible-same-speed-mode-then-other",
            "reachability": REACHABILITY_RULE,
            "lookahead": LOOKAHEAD_RULE,
            "lookahead_depth": self.depth,
            "node_budget": self.node_budget,
            "budget_fallback": BUDGET_FALLBACK,
            "no_escape_fallback": NO_ESCAPE_FALLBACK,
            "trigger": "v2-base-count-below-min(factor*need,cap)",
            "trigger_factor": self.trigger_factor,
            "tail_release_slack": self.slack,
        }

    def record(self) -> Dict[str, Any]:
        """Descriptor plus v2's seven counters, for rollout probes."""
        return {**self.descriptor(), "counters": self.counters.to_dict()}


def _as_list(values: Any) -> List[Any]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    return list(values)


def install_lookahead_veto(snake: Any, **options: Any) -> LookaheadFreeSpaceVeto:
    """Attach a fresh :class:`LookaheadFreeSpaceVeto` to an AISnake and return it."""
    if not hasattr(snake, "safety_veto"):
        raise TypeError("safety veto can only be installed on an AISnake")
    veto = LookaheadFreeSpaceVeto(**options)
    snake.safety_veto = veto
    return veto
