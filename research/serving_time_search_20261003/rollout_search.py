"""Prototype (research only, opt-in): hero-only rollout check for a future "veto v9".

Feasibility-study code (``docs/research/serving_time_search_feasibility_2026-10-03.md``).
Nothing imports or installs it: no default, served veto, champion or gate changes. It
lives under ``research/`` (not ``src/``) so the source closures that the strict and serving
receipts bind are untouched.

What it does, per hero decision:

1. For each candidate action ``a`` (0..5), play ``a`` on a cell-level copy of the hero
   (v5's exact :func:`~src.evaluation.safety_veto_v5.simulate_move`, run on cell
   coordinates with ``segment_size = 1``), then ``horizon - 1`` frames of a deterministic
   normal-speed rollout policy.
2. **Opponent model (identity-blind, no oracle).** Other live snakes are static walls
   (v2/v3/v7 convention) and, on the first simulated frame only, every cell an opponent
   head can reach that the hero would not win a head-on in (v6's
   :func:`~src.evaluation.safety_veto_v6.opponent_head_reach` and
   :func:`~src.evaluation.safety_veto_v6.hero_wins_head_on`) is fatal. The model never
   reads an opponent's checkpoint, policy or kind, so it cannot overfit the gate's
   opponent pool. No food spawns and no growth (growth only delays tail release; the
   tail-aware counts use v3's slack of 1 for that).
3. **Rollout policy.** Among the three normal-speed moves that do not die this frame,
   take the one whose tail-aware reachable count (v3's model, capped at ``policy_cap``)
   is largest; ties go straight, then left, then right. Deterministic, no RNG.
4. **Score** ``(survived_all, leaf_ok, frames_survived, leaf_area)`` where ``leaf_ok``
   means the tail-aware count from the final head reaches ``need`` (v2's need).

:func:`rollout_choice` is the sketched v9 rule: keep the incumbent (v8) choice ``c``
unless ``c`` fails the check (``not (survived_all and leaf_ok)``) and some other
masked-legal candidate passes; then take the passing candidate in ``c``'s speed mode
first, highest Q among them (ties: lowest index). Sparse by design, so paired worlds stay
tied wherever the check does not fire.

Cost is bounded by node counts (``horizon * 3`` capped floods per candidate), never by
wall time, so a strict replay is bit-reproducible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.evaluation.safety_veto import NUM_ACTIONS, NUM_DIRECTIONS, free_space_threshold
from src.evaluation.safety_veto_v3 import TAIL_RELEASE_SLACK, Grid, grid_for, static_blocked
from src.evaluation.safety_veto_v5 import simulate_move
from src.evaluation.safety_veto_v6 import hero_wins_head_on, opponent_head_reach

Cell = Tuple[int, int]
CARDINAL: Tuple[Cell, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))  # GameLogic order
ROLLOUT_ORDER = (1, 0, 2)  # straight, left, right


@dataclass(frozen=True)
class HeroCells:
    """The hero in cell units: body head-first, absolute direction, length, boost counter."""

    body: Tuple[Cell, ...]
    direction: Cell
    length: int
    boost_frames: int


@dataclass(frozen=True)
class RolloutParams:
    """Static knobs (all integers; part of a future descriptor)."""

    horizon: int = 40
    policy_cap: int = 64
    min_boost_length: int = 5
    boost_cost_frames: int = 3
    slack: int = TAIL_RELEASE_SLACK


@dataclass(frozen=True)
class RolloutScore:
    """Outcome of one candidate's rollout."""

    survived_all: bool
    leaf_ok: bool
    frames_survived: int
    leaf_area: int
    nodes: int

    @property
    def passes(self) -> bool:
        """Survives the horizon and ends with at least ``need`` reachable cells."""
        return self.survived_all and self.leaf_ok


def relative_to_absolute(direction: Cell, relative: int) -> Cell:
    """``GameLogic.relative_to_absolute_direction`` (0 left, 1 straight, 2 right)."""
    idx = CARDINAL.index(tuple(direction))
    if relative == 0:
        return CARDINAL[(idx - 1) % 4]
    if relative == 2:
        return CARDINAL[(idx + 1) % 4]
    return CARDINAL[idx]


def step_hero(
    hero: HeroCells, relative: int, boost: bool, params: RolloutParams
) -> Tuple[HeroCells, Tuple[Cell, ...]]:
    """One frame of ``Snake.move`` in cell units; returns ``(hero_after, traversed)``."""
    direction = relative_to_absolute(hero.direction, relative)
    move = simulate_move(
        hero.body,
        direction,
        hero.length,
        hero.boost_frames,
        bool(boost),
        1,
        params.min_boost_length,
        params.boost_cost_frames,
    )
    after = HeroCells(tuple(move.segments), direction, int(move.length), int(move.boost_frames))
    return after, tuple(move.traversed)


def dies(
    after: HeroCells,
    traversed: Sequence[Cell],
    blocked: set,
    fatal_first_frame: set,
    in_bounds: Callable[[Cell], bool],
    first_frame: bool,
) -> bool:
    """Wall, static opponent body, first-frame head threat, or self (``segments[3:]``)."""
    tail = set(after.body[3:]) if len(after.body) > 3 else set()
    for cell in traversed:
        if not in_bounds(cell) or cell in blocked or cell in tail:
            return True
        if first_frame and cell in fatal_first_frame:
            return True
    return False


def tail_aware_count(
    hero: HeroCells, blocked: set, in_bounds: Callable[[Cell], bool], cap: int, slack: int
) -> int:
    """v3's tail-aware reachable count from the head (distance 0), at most ``cap``."""
    release: Dict[Cell, int] = {}
    for index, cell in enumerate(hero.body):
        when = int(hero.length) - index + int(slack)
        if when > release.get(cell, -1):
            release[cell] = when
    head = hero.body[0]
    release.pop(head, None)
    seen = {head}
    queue = deque([(head, 0)])
    count = 0
    while queue and count < cap:
        (gx, gy), distance = queue.popleft()
        count += 1
        for nb in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
            if nb in seen or nb in blocked or not in_bounds(nb):
                continue
            when = release.get(nb)
            if when is not None and distance + 1 < when:
                continue
            seen.add(nb)
            queue.append((nb, distance + 1))
    return count


def rollout(
    hero: HeroCells,
    action: int,
    blocked: set,
    fatal_first_frame: set,
    in_bounds: Callable[[Cell], bool],
    need: int,
    params: RolloutParams,
) -> RolloutScore:
    """Play ``action`` then the rollout policy for ``horizon - 1`` frames (no RNG)."""
    boost = int(action) >= NUM_DIRECTIONS and hero.length >= params.min_boost_length
    state, traversed = step_hero(hero, int(action) % NUM_DIRECTIONS, boost, params)
    nodes = 1
    if dies(state, traversed, blocked, fatal_first_frame, in_bounds, True):
        return RolloutScore(False, False, 0, 0, nodes)
    survived = 1
    for _ in range(params.horizon - 1):
        best: Optional[Tuple[int, HeroCells]] = None
        for relative in ROLLOUT_ORDER:
            nxt, trav = step_hero(state, relative, False, params)
            nodes += 1
            if dies(nxt, trav, blocked, fatal_first_frame, in_bounds, False):
                continue
            area = tail_aware_count(nxt, blocked, in_bounds, params.policy_cap, params.slack)
            if best is None or area > best[0]:
                best = (area, nxt)
        if best is None:
            return RolloutScore(False, False, survived, 0, nodes)
        state = best[1]
        survived += 1
    leaf = tail_aware_count(state, blocked, in_bounds, max(int(need), 1), params.slack)
    return RolloutScore(True, leaf >= need, survived, leaf, nodes)


def hero_cells(snake, grid: Grid) -> HeroCells:
    """Cell-level copy of a live ``Snake`` (no mutation)."""
    body = tuple(grid.to_cell(x, y) for x, y in snake.segments)
    return HeroCells(body, tuple(snake.direction), int(snake.length), int(snake.boost_frames))


def world_model(snake, other_snakes: Sequence) -> Tuple[Grid, set, set, int]:
    """``(grid, blocked, fatal_first_frame, need)`` for the hero's decision (read-only)."""
    grid = grid_for(snake)
    blocked = static_blocked(snake, other_snakes, grid)
    threats = opponent_head_reach(snake, other_snakes, grid)
    # Conservative: a threat cell is fatal unless the hero (pre-move length, no growth)
    # wins the head-on against the largest opponent that can reach it.
    fatal = {cell for cell, bound in threats.items() if not hero_wins_head_on(snake.length, bound)}
    _, need = free_space_threshold(snake.length, snake._logical_length())
    return grid, blocked, fatal, need


def score_actions(
    snake, other_snakes: Sequence, candidates: Sequence[int], params: RolloutParams
) -> Dict[int, RolloutScore]:
    """Rollout score for each candidate action of a live hero (read-only)."""
    grid, blocked, fatal, need = world_model(snake, other_snakes)
    hero = hero_cells(snake, grid)
    return {
        int(a): rollout(hero, int(a), blocked, fatal, grid.in_bounds, need, params)
        for a in candidates
    }


def rollout_choice(
    q_values: Sequence[float],
    action_mask: Sequence[bool],
    current: int,
    scores: Dict[int, RolloutScore],
) -> Tuple[int, str]:
    """The sketched v9 rule over precomputed scores (pure; see module docstring).

    Returns ``(action, reason)`` with reason ``kept_passes``, ``kept_no_alternative``
    or ``switched``. Never returns a masked action other than ``current`` itself.
    """
    if len(q_values) != NUM_ACTIONS or len(action_mask) != NUM_ACTIONS:
        raise ValueError(f"expected {NUM_ACTIONS} Q-values and mask entries")
    own = scores.get(int(current))
    if own is not None and own.passes:
        return int(current), "kept_passes"
    passing = [
        a
        for a in range(NUM_ACTIONS)
        if a != int(current) and bool(action_mask[a]) and a in scores and scores[a].passes
    ]
    if not passing:
        return int(current), "kept_no_alternative"
    boost = int(current) >= NUM_DIRECTIONS
    same = [a for a in passing if (a >= NUM_DIRECTIONS) == boost]
    pool: List[int] = same or passing
    best = pool[0]
    for a in pool[1:]:
        if float(q_values[a]) > float(q_values[best]):
            best = a
    return best, "switched"
