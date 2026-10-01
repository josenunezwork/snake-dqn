"""Opt-in serving-time free-space veto, v3: tail-aware reachability.

This module adds a NEW veto, ``free-space-veto/v3-tail-aware``. It does not change
:mod:`src.evaluation.safety_veto` (v2): that file's bytes, its descriptor and its
source-sha-bound strict receipt stay valid. v3 reuses v2's pure pieces unchanged
(:func:`~src.evaluation.safety_veto.free_space_threshold`,
:func:`~src.evaluation.safety_veto.veto_choice` and
:class:`~src.evaluation.safety_veto.SafetyVetoCounters`), so the cap/need formulas,
the veto rule and the speed-preserving replacement are v2's by construction. Only
the reachable-cell count per direction differs.

Why: v2 (like the observation feature it reads) blocks the snake's WHOLE body for
the flood fill. A long snake that coils is then told every move is cramped even
when it can follow its own tail out, and v2 falls back to the policy's own choice
(``fallback_no_spacious``). In the saved run-v3 strict records every candidate
self-collision episode had at least one such fallback.

Tail-aware reachability (one breadth-first search per direction):

* The grid, bounds (rectangular or circular arena) and the "other live snakes are
  walls" rule are exactly those of
  :meth:`SnakeStateMixin._get_free_space_features`. Other snakes are treated as
  static (conservative: their tails also move, but their heads can grow too).
* The candidate next-head cell is at BFS distance 1 (one move from now).
* Own body segment ``j`` (0 = head, list order) is gone after ``k`` moves iff
  ``j + k >= length`` (``Snake.move`` inserts a head and pops the tail only while
  ``len(segments) > length``; collisions are checked after the move). So its cell
  becomes passable at distance ``length - j + slack``. Counted from the tail
  (``i = len(segments) - 1 - j``) that is ``i + 1 + pending_growth + slack`` where
  ``pending_growth = length - len(segments)`` is the exact not-yet-filled growth.
  ``slack`` (default 1) absorbs one food eaten on the way; growth is otherwise
  ignored, and boost burns (which only shorten the body) are ignored too. This is
  a known PERMISSIVE approximation: every pellet eaten on the way delays every
  later release by one more step (``Snake.grow`` adds length and ``move`` skips
  the pop), so in dense food a path that follows the tail can be counted open
  while the real tail is still there. The opt-in diagnostics count such
  "tail-admitted" decisions so a screen can tell a rescue from an admitted
  pocket that closed (see :class:`TailAwareDiagnostics`).
* A body cell is entered when some neighbour reached at distance ``d`` has
  ``d + 1 >= release``. A body cell rejected from an early neighbour can still be
  entered later from a farther one (it is not marked seen on rejection).
* The search stops after ``cap`` accepted cells, the same bound v2 uses, so it is
  ``O(4 * cap)`` per direction.

With release disabled the count equals v2's static count, and enabling release
only adds cells, so for the default one-step mode v3 never vetoes a direction
that v2 allows; it can only (a) keep a base action v2 would veto and (b) turn a
v2 ``no_spacious`` fallback into an informed choice.

Opt-in boost refinement (``boost_two_step=True``, default off and NOT used by the
pre-registered screen): a boost action is scored from its two-step cell (distance
2) with the first-step cell blocked as the new neck. This closes v2's documented
one-step approximation for boosts and is the case where v3 is stricter than v2.

Default behavior is untouched: nothing installs this class unless a caller does so
explicitly with :func:`install_tail_aware_veto`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.core.game_config import GameConfig
from src.evaluation.safety_veto import (
    NUM_ACTIONS,
    NUM_DIRECTIONS,
    OUTCOME_NO_SPACIOUS,
    SafetyVetoCounters,
    free_space_threshold,
    veto_choice,
)
from src.game.game_logic import GameLogic
from src.game.snake_state import FREE_SPACE_BFS_CAP, FREE_SPACE_MIN_CAP

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.game.snake import Snake

VETO_METHOD_V3 = "free-space-veto/v3-tail-aware"
TAIL_RELEASE_SLACK = 1
# Diagnostics only: decisions counted back from the episode's last decision when
# asking whether a tail-admitted decision preceded a self-collision death.
CLOSING_POCKET_WINDOW = 50
REACHABILITY_RULE = "tail-aware-bfs/own-body-released-by-steps/other-snakes-static"

Cell = Tuple[int, int]


@dataclass(frozen=True)
class Grid:
    """Segment-resolution grid of one snake's free-space flood fill."""

    segment_size: int
    width: int
    height: int
    circular: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    radius_sq: float = 0.0

    def to_cell(self, x: float, y: float) -> Cell:
        """Half-open floor mapping, as in ``_get_free_space_features``."""
        return (int(x // self.segment_size), int(y // self.segment_size))

    def in_bounds(self, cell: Cell) -> bool:
        """Rectangle bounds plus the circular-arena corner test."""
        gx, gy = cell
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        if self.circular:
            px, py = gx * self.segment_size, gy * self.segment_size
            if (px - self.center_x) ** 2 + (py - self.center_y) ** 2 > self.radius_sq:
                return False
        return True


def grid_for(snake: "Snake") -> Grid:
    """The grid ``snake._get_free_space_features`` floods (same formulas)."""
    ss = max(int(snake.segment_size), 1)
    width = max(int(math.ceil(snake.game_width / ss)), 1)
    height = max(int(math.ceil(snake.game_height / ss)), 1)
    if GameConfig.ARENA_TYPE != "circular":
        return Grid(ss, width, height)
    cx, cy, radius = GameLogic.get_circular_arena(snake.game_width, snake.game_height)
    return Grid(ss, width, height, True, float(cx), float(cy), float(radius) ** 2)


def own_body_release(
    segments: Sequence[Tuple[float, float]], length: int, grid: Grid, slack: int
) -> Dict[Cell, int]:
    """Cell -> first BFS distance at which the own body no longer occupies it.

    Segment ``j`` (0 = head) vacates after ``length - j`` moves; a cell holding
    several segments releases when the one nearest the head does.
    """
    release: Dict[Cell, int] = {}
    for index, (sx, sy) in enumerate(segments):
        cell = grid.to_cell(sx, sy)
        when = int(length) - index + int(slack)
        if when > release.get(cell, -1):
            release[cell] = when
    return release


def static_blocked(snake: "Snake", other_snakes: Iterable["Snake"], grid: Grid) -> set:
    """Cells of other LIVE snakes (walls for the search, as in v2)."""
    blocked: set = set()
    for other in other_snakes:
        if other is snake or not other.is_alive:
            continue
        for sx, sy in other.segments:
            blocked.add(grid.to_cell(sx, sy))
    return blocked


def tail_aware_reachable(
    start: Cell,
    start_distance: int,
    blocked: set,
    release: Dict[Cell, int],
    in_bounds: Callable[[Cell], bool],
    cap: int,
    tail_release: bool = True,
) -> int:
    """Count cells reachable from ``start`` (at ``start_distance``), at most ``cap``.

    ``blocked`` cells are never passable. A cell in ``release`` is passable only at
    distance ``>= release[cell]`` when ``tail_release`` is on, never when it is off
    (``tail_release=False`` reproduces v2's static count exactly).
    """

    def passable(cell: Cell, distance: int) -> bool:
        if cell in blocked or not in_bounds(cell):
            return False
        when = release.get(cell)
        return when is None or (tail_release and distance >= when)

    if not passable(start, start_distance):
        return 0
    seen = {start}
    queue = deque([(start, start_distance)])
    count = 0
    while queue and count < cap:
        (gx, gy), distance = queue.popleft()
        count += 1
        for nb in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
            if nb not in seen and passable(nb, distance + 1):
                seen.add(nb)
                queue.append((nb, distance + 1))
    return count


@dataclass(frozen=True)
class TailAwareCounts:
    """Raw reachable-cell counts for one decision.

    ``one_step[d]`` is direction ``d``'s count from its next-head cell (tail-aware
    unless computed with ``tail_release=False``); ``two_step[d]`` is the boost
    count from the two-step cell, or ``None`` when the boost refinement is off.
    """

    one_step: Tuple[int, int, int]
    two_step: Optional[Tuple[int, int, int]] = None


def reachable_counts(
    snake: "Snake",
    other_snakes: Sequence["Snake"],
    *,
    slack: int = TAIL_RELEASE_SLACK,
    tail_release: bool = True,
    boost_two_step: bool = False,
) -> TailAwareCounts:
    """Per-direction counts for ``snake`` (cap as v2: length-scaled, at most 160)."""
    grid = grid_for(snake)
    blocked = static_blocked(snake, other_snakes, grid)
    release = own_body_release(snake.segments, int(snake.length), grid, slack)
    cap, _ = free_space_threshold(snake.length, snake._logical_length())
    ss = grid.segment_size
    head_x, head_y = snake.head
    one: List[int] = []
    two: List[int] = []
    for relative in range(NUM_DIRECTIONS):
        dx, dy = GameLogic.relative_to_absolute_direction(snake.direction, relative)
        first = grid.to_cell(head_x + dx * ss, head_y + dy * ss)
        one.append(
            tail_aware_reachable(first, 1, blocked, release, grid.in_bounds, cap, tail_release)
        )
        if boost_two_step:
            second = grid.to_cell(head_x + 2 * dx * ss, head_y + 2 * dy * ss)
            neck = blocked | {first}
            two.append(
                tail_aware_reachable(second, 2, neck, release, grid.in_bounds, cap, tail_release)
            )
    return TailAwareCounts(
        (one[0], one[1], one[2]), (two[0], two[1], two[2]) if boost_two_step else None
    )


def eligibility_inputs(
    counts: TailAwareCounts, action_mask: Sequence[bool], need: int
) -> Tuple[List[bool], List[bool]]:
    """``(mask, spacious)`` to hand to v2's :func:`veto_choice`.

    One-step mode: ``spacious[d] = one_step[d] >= need`` and the mask is unchanged,
    so the rule is literally v2's. Two-step mode additionally clears a boost
    action's mask entry when its two-step count is below ``need``.
    """
    spacious = [int(value) >= int(need) for value in counts.one_step]
    mask = [bool(value) for value in action_mask]
    if counts.two_step is not None:
        for direction, value in enumerate(counts.two_step):
            if int(value) < int(need):
                mask[NUM_DIRECTIONS + direction] = False
    return mask, spacious


@dataclass
class TailAwareDiagnostics:
    """Comparison against v2's static count on the same decision (opt-in, not a probe).

    Kept off the ``probes.safety_veto`` record so the probe stays the exact
    descriptor-plus-seven-counters shape ``strict_promotion`` validates.

    A decision is *tail-admitted* when the executed action's direction is spacious
    only thanks to tail release (v3 spacious, v2's static count not). That covers
    "v3 kept a base move v2 would veto", "v3 kept or chose where v2 fell back",
    and "v3 vetoed into a released direction". ``tail_admitted_last_decision`` is
    the 1-based index of the latest one (0 = none) and
    ``tail_admitted_in_final_window`` counts those within the episode's last
    :data:`CLOSING_POCKET_WINDOW` decisions. With a terminal hero (one decision per
    frame alive) and a ``self`` death cause, a non-zero final-window count flags a
    pocket v3 admitted that may have closed (the growth gap); a rescue is a
    tail-admitted episode without one. Reported only; actions never change.
    """

    decisions: int = 0
    action_differs_from_v2: int = 0
    v2_no_spacious: int = 0
    v2_no_spacious_rescued: int = 0
    v2_vetoed_v3_kept: int = 0
    directions_released_by_tail: int = 0
    tail_admitted: int = 0
    tail_admitted_last_decision: int = 0
    _recent: deque = field(default_factory=deque, repr=False)

    def admit(self) -> None:
        """Record the current decision (already counted) as tail-admitted."""
        self.tail_admitted += 1
        self.tail_admitted_last_decision = self.decisions
        self._recent.append(self.decisions)
        while self._recent and self._recent[0] <= self.decisions - CLOSING_POCKET_WINDOW:
            self._recent.popleft()

    def to_dict(self) -> Dict[str, int]:
        out = {k: int(v) for k, v in self.__dict__.items() if not k.startswith("_")}
        floor = self.decisions - CLOSING_POCKET_WINDOW
        out["tail_admitted_in_final_window"] = sum(1 for i in self._recent if i > floor)
        return out


class TailAwareFreeSpaceVeto:
    """Stateful per-snake veto hook (same surface as ``FreeSpaceVeto``)."""

    method = VETO_METHOD_V3

    def __init__(
        self,
        *,
        slack: int = TAIL_RELEASE_SLACK,
        boost_two_step: bool = False,
        diagnostics: bool = False,
    ) -> None:
        if int(slack) < 0:
            raise ValueError("slack must be non-negative")
        self.slack = int(slack)
        self.boost_two_step = bool(boost_two_step)
        self.track_diagnostics = bool(diagnostics)
        self.counters = SafetyVetoCounters()
        self.diagnostics = TailAwareDiagnostics()

    def reset(self) -> None:
        """Zero the counters and diagnostics (call at episode start)."""
        self.counters = SafetyVetoCounters()
        self.diagnostics = TailAwareDiagnostics()

    def apply(
        self,
        snake: "Snake",
        other_snakes: Sequence["Snake"],
        q_values: Any,
        action_mask: Any,
        base_action: int,
    ) -> int:
        """Return the (possibly vetoed) action for one greedy decision."""
        others = list(other_snakes)
        q = _as_list(q_values)
        raw_mask = [bool(x) for x in _as_list(action_mask)]
        if len(q) != NUM_ACTIONS or len(raw_mask) != NUM_ACTIONS:
            raise ValueError(f"expected {NUM_ACTIONS} Q-values and mask entries")
        _, need = free_space_threshold(snake.length, snake._logical_length())
        counts = reachable_counts(
            snake, others, slack=self.slack, boost_two_step=self.boost_two_step
        )
        mask, spacious = eligibility_inputs(counts, raw_mask, need)
        action, outcome = veto_choice(q, mask, spacious, base_action)
        self.counters.record(int(base_action), action, outcome)
        if self.track_diagnostics:
            static = reachable_counts(snake, others, slack=self.slack, tail_release=False)
            static_spacious = [value >= need for value in static.one_step]
            v2_action, v2_outcome = veto_choice(q, raw_mask, static_spacious, base_action)
            self._observe(spacious, static_spacious, action, outcome, v2_action, v2_outcome)
        return action

    def _observe(
        self,
        spacious: Sequence[bool],
        static_spacious: Sequence[bool],
        action: int,
        outcome: str,
        v2_action: int,
        v2_outcome: str,
    ) -> None:
        diag = self.diagnostics
        diag.decisions += 1
        diag.action_differs_from_v2 += int(action != v2_action)
        diag.directions_released_by_tail += sum(
            1 for now, before in zip(spacious, static_spacious) if now and not before
        )
        direction = int(action) % NUM_DIRECTIONS
        if spacious[direction] and not static_spacious[direction]:
            diag.admit()
        if v2_outcome == OUTCOME_NO_SPACIOUS:
            diag.v2_no_spacious += 1
            diag.v2_no_spacious_rescued += int(outcome != OUTCOME_NO_SPACIOUS)
        elif v2_outcome == "vetoed" and outcome == "kept":
            diag.v2_vetoed_v3_kept += 1

    def diagnostics_record(self) -> Optional[Dict[str, int]]:
        """The v2-comparison diagnostics, or ``None`` when they are not tracked."""
        return self.diagnostics.to_dict() if self.track_diagnostics else None

    def descriptor(self) -> Dict[str, Any]:
        """Static identity of this veto (no counters)."""
        return {
            "method": VETO_METHOD_V3,
            "free_space_bfs_cap": FREE_SPACE_BFS_CAP,
            "free_space_min_cap": FREE_SPACE_MIN_CAP,
            "boost_approximation": (
                "two-step-path-neck-blocked"
                if self.boost_two_step
                else "one-step-direction-feature"
            ),
            "replacement_rule": "highest-q-eligible-same-speed-mode-then-other",
            "reachability": REACHABILITY_RULE,
            "tail_release_slack": self.slack,
        }

    def record(self) -> Dict[str, Any]:
        """Descriptor plus the current counters, for rollout probes."""
        return {**self.descriptor(), "counters": self.counters.to_dict()}


def _as_list(values: Any) -> List[Any]:
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    return list(values)


def install_tail_aware_veto(snake: Any, **options: Any) -> TailAwareFreeSpaceVeto:
    """Attach a fresh :class:`TailAwareFreeSpaceVeto` to an AISnake and return it."""
    if not hasattr(snake, "safety_veto"):
        raise TypeError("safety veto can only be installed on an AISnake")
    veto = TailAwareFreeSpaceVeto(**options)
    snake.safety_veto = veto
    return veto
