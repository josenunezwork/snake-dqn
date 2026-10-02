#!/usr/bin/env python3
"""DEVELOPMENT DIAGNOSTIC (Tier-0/1): point of no return before Apex self-collision deaths.

Not a screen, not gate evidence, no candidate and no decision rule. It measures how
many frames before each self-collision death the Apex champion (with the released v2
free-space veto) last had a way out, to decide whether a deeper search shield or the
policy/training is the next lever.

Pipeline:

1. **Simulate.** Champion + v2 veto (``run_simd_eval(vector61=True,
   hero_safety_veto=True, vector61_forward="rowwise")``, the configuration that is
   bit-exact with the live released Watch hero) at H5000, profile
   ``promotion-v2-watch-rect``, strict balanced rosters, all three mixes, on the fresh
   development namespace :data:`DOMAIN`. A read-only wrapper around
   ``Vector61SimdPolicy._apply_veto`` (it calls the original first and returns its
   result unchanged) snapshots, for every hero decision, the hero body (ordered),
   logical length, heading, boost counter, the cells of every other live snake, the
   food cells, and the v2 veto's one-step counts/need/base/final action. Snapshots
   live in a per-world ring of :data:`WINDOW` decisions, so memory is bounded.
2. **Search offline.** For each hero self-collision death, walk backwards from the
   fatal decision. At each frame run :class:`EscapeSearch` for every action: the
   hero's own moves are simulated exactly (BatchSim movement, boost two-step and burn,
   growth from the food present at that frame, self/wall collision), other snakes are
   static walls. An action *escapes* (primary criterion, :data:`COUNT_OR_DEPTH`) when
   some continuation reaches a state whose tail-aware reachable count (v3/v4 release
   model, slack 1) is ``>= need`` (v2's ``min(length, cap)``), or survives
   :data:`DEPTH` frames. It is computed exactly as the combination of two one-sided
   searches, which are also reported as sensitivities: :data:`DEPTH_ONLY` (survive
   ``DEPTH`` frames) and :data:`COUNT_ONLY` (reach a spacious state within ``DEPTH``
   frames). Depth-first search with a failed-state memo, a sound region prune
   (:func:`doomed`) and a per-action node budget decides; an exhausted budget is
   ``unknown`` and is never guessed.
3. **PNR.** The point of no return is the last frame at or before the fatal decision
   at which at least one action escapes; ``frames_before_death = fatal_frame - pnr``.
   Unknown frames after the candidate turn the result into a bounded ``unknown``.

Usage (CPU slot 1, AC power required; output directory is create-only):
  ./venv/bin/python research/trap_horizon_20261001/diagnose.py
Smoke (separate smoke namespace, short horizon, scratch output):
  ./venv/bin/python research/trap_horizon_20261001/diagnose.py --smoke --out /tmp/x \\
      --worlds-per-mix 2 --frames 600 --mixes frozen
"""

from __future__ import annotations

import os

# Thread caps must be set before torch/numpy import anywhere in the process. setdefault
# keeps an importing test process untouched; main() refuses unless they are as required.
os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
REQUIRED_ENV = {"SNAKE_DQN_DEVICE": "cpu", "OMP_NUM_THREADS": "2"}

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import (  # noqa: E402
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "trap-horizon-diagnostic/v1"
AUTHORITY = "development-diagnostic (Tier-0/1; not a screen, not gate evidence)"
DOMAIN = "trap-horizon-dev-v1"
SMOKE_DOMAIN = "trap-horizon-dev-smoke-v1"
NAMESPACE = "worlds"
WORLDS_PER_MIX = 16
WINDOW = 120  # hero decisions kept per world (the fatal one included)
DEPTH = 40  # search horizon in frames
NODE_BUDGET = 50000  # nodes per (frame, first action, criterion); exhausted -> unknown
TAIL_SLACK = 1  # v3/v4 TAIL_RELEASE_SLACK
VECTOR61_FORWARD = "rowwise"  # bit-exact batch-1 forwards (the live call shape)
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
DEFAULT_OUT = ARTIFACTS / "trap-horizon-20261001" / "run-v1"
SLOT_LOCK = ARTIFACTS / "pqn-followup-20260909" / "cpu-slot-1.lock"
EXCLUSION_PREFIX = 1000  # seeds per earlier domain/purpose checked

ESCAPE = "escape"
NO_ESCAPE = "no_escape"
UNKNOWN = "unknown"
NUM_ACTIONS = 6
NUM_DIRECTIONS = 3
# Up, right, down, left: BatchSim's CARDINAL / direction index order.
CARDINAL: Tuple[Tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
BINS = ("le_8", "9_20", "21_40", "gt_40", "unknown")
COUNT_OR_DEPTH = "count_or_depth"  # primary escape definition
DEPTH_ONLY = "depth_only"  # sensitivity: survive `depth` frames only
COUNT_ONLY = "count_only"  # sensitivity: reach a spacious state within `depth` frames
CRITERIA = (COUNT_OR_DEPTH, DEPTH_ONLY, COUNT_ONLY)

Cell = Tuple[int, int]


# ---------------------------------------------------------------------------
# Exact own-move model (pure)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StaticWorld:
    """Everything outside the hero, frozen at one decision frame (cell units)."""

    width: int
    height: int
    blocked: FrozenSet[Cell]  # cells of every other live snake (static walls)
    food: FrozenSet[Cell]
    min_boost_length: int = 5
    boost_cost_frames: int = 3
    trail_food: bool = True  # mechanics v2: a burned boost tail becomes a pellet

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height


@dataclass(frozen=True)
class HeroState:
    """The hero exactly as BatchSim holds it at a decision (head first)."""

    body: Tuple[Cell, ...]
    length: int
    direction: int
    boost_frames: int = 0
    food_added: FrozenSet[Cell] = frozenset()  # own trail pellets dropped on this path
    food_eaten: FrozenSet[Cell] = frozenset()  # pellets eaten on this path

    @property
    def key(self) -> Tuple[Any, ...]:
        return (
            self.body,
            self.length,
            self.direction,
            self.boost_frames,
            self.food_added,
            self.food_eaten,
        )


def need_for(length: int) -> int:
    """v2's ``need``: ``min(max(1, length), cap)``, ``cap = min(160, max(32, 2*length))``."""
    from src.evaluation.safety_veto import free_space_threshold

    _cap, need = free_space_threshold(int(length), max(1, int(length)))
    return int(need)


def step_hero(world: StaticWorld, state: HeroState, action: int) -> Tuple[Optional[HeroState], str]:
    """One BatchSim frame of the hero alone: ``(next_state, "alive")`` or ``(None, cause)``.

    Mirrors ``BatchSim._move_all`` (heading, one or two head pushes with the
    ``len > length`` tail pop, boost eligibility, burn cadence and the extra tail pop),
    ``_drop_trail_pellets`` (v2), ``_consume_food`` (first pellet on a traversed cell,
    ``length += 1``) and the hero's wall/self checks of ``_build_snake_frame_cache``
    (self: a traversed cell in ``segments[3:]`` when ``seg_count > 3``). Other snakes are
    the static ``blocked`` set; a traversed cell in it is cause ``"other"``. Causes are
    checked in BatchSim's order: wall, self, other.
    """
    rel = int(action) % NUM_DIRECTIONS
    direction = (state.direction + (-1 if rel == 0 else (1 if rel == 2 else 0))) % 4
    dx, dy = CARDINAL[direction]
    length = int(state.length)
    body = state.body
    first = (body[0][0] + dx, body[0][1] + dy)
    # Invariant len(body) <= length, so one push needs at most one pop.
    segs = (first,) + body
    if len(segs) > length:
        segs = segs[:length]
    traversed: Tuple[Cell, ...] = (first,)
    boost_frames = int(state.boost_frames)
    added = state.food_added
    eaten = state.food_eaten

    def has_food(cell: Cell) -> bool:
        return (cell in world.food and cell not in eaten) or cell in added

    if int(action) >= NUM_DIRECTIONS and length >= world.min_boost_length:
        second = (first[0] + dx, first[1] + dy)
        segs = (second,) + segs
        if len(segs) > length:
            segs = segs[:length]
        traversed = (first, second)
        boost_frames += 1
        if boost_frames >= world.boost_cost_frames:
            boost_frames = 0
            if length > 1:
                length -= 1
                if len(segs) > length:
                    tail = segs[-1]
                    segs = segs[:-1]
                    if world.trail_food and world.in_bounds(tail) and not has_food(tail):
                        added = added | {tail}
    for cell in traversed:
        if has_food(cell):
            length += 1
            if cell in added:
                added = added - {cell}
            else:
                eaten = eaten | {cell}
            break
    for cell in traversed:
        if not world.in_bounds(cell):
            return None, "wall"
    if len(segs) > 3:
        rest = segs[3:]
        for cell in traversed:
            if cell in rest:
                return None, "self"
    for cell in traversed:
        if cell in world.blocked:
            return None, "other"
    return (
        HeroState(
            body=segs,
            length=length,
            direction=direction,
            boost_frames=boost_frames,
            food_added=frozenset(added),
            food_eaten=frozenset(eaten),
        ),
        "alive",
    )


def own_release(state: HeroState, slack: int = TAIL_SLACK) -> Dict[Cell, int]:
    """Own body (head excluded) -> first move count at which the cell is free.

    Segment ``j`` (0 = head) vacates after ``length - j`` moves without growth (pending
    growth included, as ``len(body) <= length``); ``slack`` absorbs one pellet, exactly
    v3's ``own_body_release``. A cell held twice releases with its head-most segment.
    """
    release: Dict[Cell, int] = {}
    for index, cell in enumerate(state.body):
        if index == 0:
            continue
        when = int(state.length) - index + int(slack)
        if when > release.get(cell, -1):
            release[cell] = when
    release.pop(state.body[0], None)
    return release


class _Either:
    """Membership in either of two sets without building their union."""

    __slots__ = ("first", "second")

    def __init__(self, first: FrozenSet[Cell], second: Any) -> None:
        self.first = first
        self.second = second

    def __contains__(self, cell: object) -> bool:
        return cell in self.first or cell in self.second


def split_release(
    state: HeroState, limit: int, slack: int = TAIL_SLACK
) -> Tuple[FrozenSet[Cell], Dict[Cell, int]]:
    """``(front, release)``: own cells that stay blocked for a count of ``limit``, and the rest.

    A breadth-first count capped at ``limit`` accepted cells never reaches distance
    ``limit + 1``, so a segment whose release is ``> limit`` is a wall for that count.
    Only the tail-side segments get a release entry; the result of
    :func:`tail_aware_reachable` is identical to using :func:`own_release` (tested).
    """
    body = state.body
    n = len(body)
    length = int(state.length)
    # release(j) = length - j + slack <= limit  <=>  j >= length + slack - limit.
    cut = max(1, min(n, length + int(slack) - int(limit)))
    front = frozenset(body[1:cut])
    cells = body[cut:]
    values = range(length - cut + int(slack), length - n + int(slack), -1)
    # Head-most segment wins for a doubled cell: write tail-first so it is written last.
    release = dict(zip(reversed(cells), reversed(values)))
    release = {c: w for c, w in release.items() if c not in front and c != body[0]}
    return front, release


def tail_aware_count(
    world: StaticWorld, state: HeroState, limit: int, slack: int = TAIL_SLACK
) -> int:
    """v3's tail-aware breadth-first count from the head cell, at most ``limit``.

    Equals ``tail_aware_reachable(head, 0, blocked, own_release(state, slack), ...)``.
    """
    from src.evaluation.safety_veto_v3 import tail_aware_reachable

    front, release = split_release(state, limit, slack)
    return tail_aware_reachable(
        state.body[0],
        0,
        _Either(world.blocked, front),  # type: ignore[arg-type]
        release,
        world.in_bounds,
        int(limit),
    )


def tail_aware_count_reference(
    world: StaticWorld, state: HeroState, limit: int, slack: int = TAIL_SLACK
) -> int:
    """Unoptimized reference for :func:`tail_aware_count` (full release map)."""
    from src.evaluation.safety_veto_v3 import tail_aware_reachable

    return tail_aware_reachable(
        state.body[0],
        0,
        world.blocked,  # type: ignore[arg-type]
        own_release(state, slack),
        world.in_bounds,
        int(limit),
    )


def _region_count(world: StaticWorld, state: HeroState, free_from_index: int, cap: int) -> int:
    """Cells connected to the head (head included, at most ``cap``) ignoring timing.

    Own segments with index ``>= free_from_index`` count as free; earlier ones (head
    excluded) and every other snake's cell are walls.
    """
    body = state.body
    walls = frozenset(body[1 : max(1, min(len(body), int(free_from_index)))])
    start = body[0]
    seen = {start}
    queue = deque([start])
    count = 0
    blocked = world.blocked
    in_bounds = world.in_bounds
    while queue and count < cap:
        gx, gy = queue.popleft()
        count += 1
        for nb in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
            if nb in seen or nb in blocked or nb in walls or not in_bounds(nb):
                continue
            seen.add(nb)
            queue.append(nb)
    return count


def doomed(world: StaticWorld, state: HeroState, remaining: int, criterion: str) -> bool:
    """Sound proof that no continuation of ``state`` succeeds within ``remaining`` frames.

    Applies only to long heroes, ``length > 3 * remaining + 2``: then no cell entered
    during the next ``remaining`` frames can be vacated again inside them (at most two
    pushes per frame and one burn per three boost frames), so surviving ``remaining``
    frames needs at least ``remaining`` distinct new cells. Own segments that could be
    vacated within the horizon are counted as free from the start (an over-approximation
    of every cell the head could ever enter), so:

    * survival is impossible when that region (head excluded) has ``< remaining`` cells;
    * a count success is impossible when the region with the horizon extended by the
      count's own reach (``need`` more pushes) has fewer than the smallest reachable
      ``need`` (``need_for(length - ceil(remaining / 3))``) cells, since every cell a
      later tail-aware count admits lies in it.

    The criterion decides which successes must be ruled out.
    """
    length = int(state.length)
    if remaining <= 0 or length <= 3 * remaining + 2:
        return False
    pushes = 2 * remaining + (remaining + 2) // 3 + 1  # generous vacate horizon
    no_survival = True
    if criterion != COUNT_ONLY:
        reach = _region_count(world, state, length - pushes, remaining + 1)
        no_survival = reach - 1 < remaining
    if not no_survival:
        return False
    if criterion == DEPTH_ONLY:
        return True
    need_lo = need_for(max(1, length - (remaining + 2) // 3))
    need_hi = need_for(length + 2 * remaining)  # growth can raise need; free generously
    reach2 = _region_count(world, state, length - pushes - need_hi - TAIL_SLACK - 1, need_lo)
    return reach2 < need_lo


def memo_fails(criterion: str, failed_remaining: int, remaining: int) -> bool:
    """Whether a failure recorded with ``failed_remaining`` frames left implies failure now.

    Count-or-depth and depth-only: monotone upwards (``remaining >= failed_remaining``).
    Count-only: monotone downwards (``remaining <= failed_remaining``).
    """
    if criterion == COUNT_ONLY:
        return remaining <= failed_remaining
    return remaining >= failed_remaining


class BudgetExhausted(Exception):
    """The per-action node budget ran out (the result is unknown)."""


@dataclass
class ActionResult:
    """Escape verdict for one first action at one frame."""

    status: str  # ESCAPE / NO_ESCAPE / UNKNOWN
    nodes: int = 0
    kind: Optional[str] = None  # "count" or "depth" for an escape
    success_depth: Optional[int] = None
    immediate: str = "alive"  # first-step outcome ("alive", "self", "wall", "other")
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "nodes": int(self.nodes),
            "kind": self.kind,
            "success_depth": self.success_depth,
            "immediate": self.immediate,
            "seconds": float(self.seconds),
        }


class EscapeSearch:
    """Depth-first escape search on one frozen frame (memo shared across first actions).

    A node is a hero state after ``k >= 1`` frames. It is an escape when its tail-aware
    count reaches ``need`` (``kind="count"``) or ``k == depth`` (``kind="depth"``). Its
    children are every distinct action (boost actions only when the hero may boost),
    tried in order of decreasing child count (then action index), so open space succeeds
    after one node. A failed state is memoised with the remaining depth it failed at
    (:func:`memo_fails`): for :data:`COUNT_OR_DEPTH` and :data:`DEPTH_ONLY`, failing
    with ``r`` frames left implies failing with any ``r' >= r`` (every path died within
    ``r`` frames without a success); for :data:`COUNT_ONLY` it implies failing with any
    ``r' <= r`` (fewer frames, fewer chances). The memo is exact, not a heuristic, and
    it is shared by all first actions of the frame. Nodes are counted per first action;
    more than ``budget`` raises :class:`BudgetExhausted` and the action is ``unknown``.

    ``criterion`` :data:`COUNT_OR_DEPTH` (default, the diagnostic's definition) accepts
    either success kind. :data:`DEPTH_ONLY` (sensitivity check) accepts only surviving
    ``depth`` frames, so a dead-end corridor or pocket of at least ``need`` cells, which
    the area count accepts, is not an escape unless it can be survived for ``depth``
    frames. A depth-only escape is always a count-or-depth escape. :data:`COUNT_ONLY`
    accepts only reaching a state whose count is ``>= need`` within ``depth`` frames
    (surviving without ever reaching one is a failure); it is the proxy for a real way
    out when the hero is too long to be saved by surviving ``depth`` frames.

    ``prune`` (default on) abandons a node that :func:`doomed` proves cannot succeed; it
    never changes a verdict, only the node count (tested against ``prune=False``).
    """

    def __init__(
        self,
        world: StaticWorld,
        depth: int = DEPTH,
        budget: int = NODE_BUDGET,
        slack: int = TAIL_SLACK,
        criterion: str = "count_or_depth",
        prune: bool = True,
    ) -> None:
        if int(depth) < 1 or int(budget) < 1:
            raise ValueError("depth and budget must be at least 1")
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        self.criterion = criterion
        self.prune = bool(prune)
        self.pruned = 0
        self.world = world
        self.depth = int(depth)
        self.budget = int(budget)
        self.slack = int(slack)
        self.failed: Dict[Tuple[Any, ...], int] = {}
        self.count_cache: Dict[Tuple[Any, ...], int] = {}
        self.nodes = 0
        self._kind: Optional[str] = None
        self._success_depth: Optional[int] = None

    def count(self, state: HeroState) -> int:
        key = state.key
        cached = self.count_cache.get(key)
        if cached is None:
            cached = tail_aware_count(self.world, state, need_for(state.length), self.slack)
            self.count_cache[key] = cached
        return cached

    def actions_for(self, state: HeroState) -> Tuple[int, ...]:
        if state.length >= self.world.min_boost_length:
            return tuple(range(NUM_ACTIONS))
        return tuple(range(NUM_DIRECTIONS))

    def run(self, state: HeroState, action: int) -> ActionResult:
        started = time.perf_counter()
        self.nodes = 0
        self._kind = None
        self._success_depth = None
        child, outcome = step_hero(self.world, state, int(action))
        if child is None:
            return ActionResult(NO_ESCAPE, 0, immediate=outcome, seconds=0.0)
        try:
            found = self._visit(child, 1)
        except BudgetExhausted:
            return ActionResult(UNKNOWN, self.nodes, seconds=time.perf_counter() - started)
        return ActionResult(
            ESCAPE if found else NO_ESCAPE,
            self.nodes,
            kind=self._kind if found else None,
            success_depth=self._success_depth if found else None,
            seconds=time.perf_counter() - started,
        )

    def _visit(self, state: HeroState, step: int) -> bool:
        remaining = self.depth - step
        key = state.key
        if key in self.failed and memo_fails(self.criterion, self.failed[key], remaining):
            return False
        self.nodes += 1
        if self.nodes > self.budget:
            raise BudgetExhausted
        if self.criterion != DEPTH_ONLY and self.count(state) >= need_for(state.length):
            self._kind, self._success_depth = "count", step
            return True
        if step >= self.depth:
            if self.criterion == COUNT_ONLY:
                self._remember_failure(key, remaining)
                return False
            self._kind, self._success_depth = "depth", step
            return True
        if self.prune and doomed(self.world, state, remaining, self.criterion):
            self.pruned += 1
            self._remember_failure(key, remaining)
            return False
        children: List[Tuple[int, int, HeroState]] = []
        seen = set()
        for action in self.actions_for(state):
            child, _ = step_hero(self.world, state, action)
            if child is None or child.key in seen:
                continue
            seen.add(child.key)
            children.append((-self.count(child), action, child))
        children.sort(key=lambda item: (item[0], item[1]))
        for _, _, child in children:
            if self._visit(child, step + 1):
                return True
        self._remember_failure(key, remaining)
        return False

    def _remember_failure(self, key: Tuple[Any, ...], remaining: int) -> None:
        stored = self.failed.get(key)
        if stored is None or not memo_fails(self.criterion, stored, remaining):
            self.failed[key] = remaining


def analyze_frame(
    world: StaticWorld,
    state: HeroState,
    actions: Sequence[int] = tuple(range(NUM_ACTIONS)),
    depth: int = DEPTH,
    budget: int = NODE_BUDGET,
    slack: int = TAIL_SLACK,
    criterion: str = COUNT_OR_DEPTH,
) -> Dict[int, ActionResult]:
    """Escape verdict per first action at one frame (one shared memo)."""
    search = EscapeSearch(world, depth, budget, slack, criterion)
    return {int(a): search.run(state, int(a)) for a in actions}


# ---------------------------------------------------------------------------
# Backward PNR walk (pure)
# ---------------------------------------------------------------------------
def walk_back(
    n_frames: int,
    taken: Sequence[int],
    evaluate: Callable[[int], Mapping[int, str]],
) -> Dict[str, Any]:
    """Find the point of no return over window indices ``0 .. n_frames - 1``.

    Index ``n_frames - 1`` is the fatal decision. ``evaluate(t)`` returns the status
    (``escape``/``no_escape``/``unknown``) of every action at index ``t``. The walk goes
    backwards and stops at the first index with an escaping action (the PNR candidate).

    ``status``:
      * ``exact``: every index after the PNR was proven escape-less.
      * ``unknown``: some later index had an unknown action and no escape, so the true
        PNR is that index or the candidate; ``frames_before_bounds`` gives the range.
      * ``beyond_window``: every index in the window was proven escape-less; the PNR is
        at least ``n_frames`` frames before the death (``frames_before_lower``).
    """
    if n_frames < 1 or len(taken) != n_frames:
        raise ValueError("need at least one frame and one taken action per frame")
    fatal = n_frames - 1
    unknown_indices: List[int] = []
    statuses: Dict[int, Dict[int, str]] = {}
    pnr: Optional[int] = None
    for t in range(fatal, -1, -1):
        row = {int(a): str(s) for a, s in evaluate(t).items()}
        statuses[t] = row
        if any(s == ESCAPE for s in row.values()):
            pnr = t
            break
        if any(s == UNKNOWN for s in row.values()):
            unknown_indices.append(t)
    out: Dict[str, Any] = {
        "fatal_index": fatal,
        "frames_evaluated": len(statuses),
        "unknown_indices": unknown_indices,
        "pnr_index": pnr,
    }
    latest_unknown = max(unknown_indices) if unknown_indices else None
    if pnr is None:
        out["status"] = UNKNOWN if unknown_indices else "beyond_window"
        out["frames_before_death"] = None
        out["frames_before_lower"] = (
            fatal - latest_unknown if latest_unknown is not None else n_frames
        )
        out["frames_before_bounds"] = None
        out["taken_action"] = None
        out["taken_status"] = None
        out["escaping_actions"] = []
        out["fatal_choice"] = None
        return out
    row = statuses[pnr]
    taken_action = int(taken[pnr])
    escaping = sorted(a for a, s in row.items() if s == ESCAPE)
    taken_status = row.get(taken_action)
    out["status"] = "exact" if latest_unknown is None else UNKNOWN
    out["frames_before_death"] = fatal - pnr
    out["frames_before_lower"] = fatal - pnr if latest_unknown is None else fatal - latest_unknown
    out["frames_before_bounds"] = (
        [fatal - pnr, fatal - pnr]
        if latest_unknown is None
        else [fatal - latest_unknown, fatal - pnr]
    )
    out["taken_action"] = taken_action
    out["taken_status"] = taken_status
    out["escaping_actions"] = escaping
    # The policy's fatal choice: the action taken at the PNR had no escape while
    # another action did (unknown taken status is not counted either way).
    out["fatal_choice"] = (
        None if taken_status == UNKNOWN else bool(taken_status == NO_ESCAPE and escaping)
    )
    return out


def horizon_bin(result: Mapping[str, Any]) -> str:
    """Summary bin of one walk result (``unknown`` when it cannot be placed)."""
    if result["status"] == "exact":
        fb = int(result["frames_before_death"])
        if fb <= 8:
            return "le_8"
        if fb <= 20:
            return "9_20"
        if fb <= 40:
            return "21_40"
        return "gt_40"
    if result["status"] == "beyond_window":
        return "gt_40" if int(result["frames_before_lower"]) > 40 else "unknown"
    # Unknown but bounded: placed only when both bounds fall in the same bin.
    bounds = result.get("frames_before_bounds")
    if bounds:
        lo, hi = bounds
        for name, (a, b) in (
            ("le_8", (0, 8)),
            ("9_20", (9, 20)),
            ("21_40", (21, 40)),
            ("gt_40", (41, math.inf)),
        ):
            if a <= lo and hi <= b:
                return f"unknown_within_{name}"
    return "unknown"


def summarize_walks(deaths: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """PNR distribution over per-death results (each has ``mix`` and ``walk``)."""
    n = len(deaths)
    bins: Dict[str, int] = {name: 0 for name in BINS}
    bounded_unknown: Dict[str, int] = {}
    exact: List[int] = []
    censored = 0
    by_status: Dict[str, int] = {}
    for death in deaths:
        walk = death["walk"]
        by_status[walk["status"]] = by_status.get(walk["status"], 0) + 1
        name = horizon_bin(walk)
        if name.startswith("unknown_within_"):
            bounded_unknown[name] = bounded_unknown.get(name, 0) + 1
            name = "unknown"
        bins[name] += 1
        if walk["status"] == "exact":
            exact.append(int(walk["frames_before_death"]))
        elif walk["status"] == "beyond_window":
            censored += 1

    def quantiles(values: Sequence[float]) -> Optional[Dict[str, Any]]:
        if not values:
            return None
        arr = np.sort(np.asarray(values, dtype=float))

        def q(p: float) -> Any:
            # Lower order statistic (no interpolation, so censored values stay inf).
            v = arr[int(math.floor(p * (len(arr) - 1)))]
            return None if math.isinf(v) else float(v)

        return {"q25": q(0.25), "median": q(0.5), "q75": q(0.75), "n": int(len(arr))}

    pnr_found = [d["walk"] for d in deaths if d["walk"]["pnr_index"] is not None]
    fatal_known = [w for w in pnr_found if w["fatal_choice"] is not None]
    exact_walks = [d["walk"] for d in deaths if d["walk"]["status"] == "exact"]
    exact_fatal = [w for w in exact_walks if w["fatal_choice"] is not None]
    by_mix: Dict[str, Dict[str, int]] = {}
    for death in deaths:
        cell = by_mix.setdefault(death["mix"], {name: 0 for name in BINS})
        name = horizon_bin(death["walk"])
        cell["unknown" if name.startswith("unknown") else name] += 1
    return {
        "self_deaths": n,
        "by_status": dict(sorted(by_status.items())),
        "bins": bins,
        "bin_fractions": {k: (v / n if n else None) for k, v in bins.items()},
        "bounded_unknown_placements": dict(sorted(bounded_unknown.items())),
        "quantiles_exact_only": quantiles(exact),
        "quantiles_exact_plus_beyond_window_as_inf": quantiles(exact + [math.inf] * censored),
        "fatal_choice": {
            "pnr_found": len(pnr_found),
            "taken_status_known": len(fatal_known),
            "fatal_choice": sum(bool(w["fatal_choice"]) for w in fatal_known),
            "taken_action_escaped": sum(w["taken_status"] == ESCAPE for w in pnr_found),
            "taken_status_unknown": sum(w["taken_status"] == UNKNOWN for w in pnr_found),
            "fraction_of_self_deaths": (
                sum(bool(w["fatal_choice"]) for w in fatal_known) / n if n else None
            ),
            "exact_only": {
                "n": len(exact_walks),
                "fatal_choice": sum(bool(w["fatal_choice"]) for w in exact_fatal),
                "taken_action_escaped": sum(w["taken_status"] == ESCAPE for w in exact_walks),
            },
        },
        "by_mix": dict(sorted(by_mix.items())),
    }


# ---------------------------------------------------------------------------
# Snapshot capture (read-only wrapper around the v2 veto call)
# ---------------------------------------------------------------------------
def _ordered_cells(sim: Any, env: int, slot: int) -> np.ndarray:
    n = int(sim.seg_count[env, slot])
    ring = (int(sim.head_ptr[env, slot]) - np.arange(n)) % int(sim.cap)
    return np.asarray(sim.bodies[env, slot, ring], dtype=np.int16).reshape(n, 2)


class DecisionRecorder:
    """Per-world ring of the last ``window`` hero decisions (bounded memory)."""

    def __init__(self, window: int = WINDOW) -> None:
        self.window = int(window)
        self.buffers: Dict[int, Deque[Dict[str, Any]]] = {}
        self.decisions: Dict[int, int] = {}
        self.world_meta: Optional[Dict[str, Any]] = None

    def capture(self, sim: Any, last_veto: Mapping[str, Any], q_rows: np.ndarray, masks) -> None:
        if self.world_meta is None:
            self.world_meta = {
                "grid_width": int(sim.grid_w),
                "grid_height": int(sim.grid_h),
                "min_boost_length": int(sim.cfg.min_boost_length),
                "boost_cost_frames": int(sim.cfg.boost_length_cost_frames),
                "trail_food": bool(sim.v2),
                "segment_size": int(sim.s),
            }
        rows = np.asarray(last_veto["rows"])
        for i, (env, slot) in enumerate(rows):
            env, slot = int(env), int(slot)
            others = [
                _ordered_cells(sim, env, s)
                for s in range(int(sim.S))
                if s != slot and bool(sim.alive[env, s])
            ]
            snap = {
                "frame": int(sim.frame[env]),
                "body": _ordered_cells(sim, env, slot),
                "length": int(sim.length[env, slot]),
                "direction": int(sim.direction[env, slot]),
                "boost_frames": int(sim.boost_frames[env, slot]),
                "others": (
                    np.concatenate(others).astype(np.int16)
                    if others
                    else np.zeros((0, 2), np.int16)
                ),
                "food": np.asarray(list(sim.food_cells[env]), dtype=np.int16).reshape(-1, 2),
                "v2_counts": np.asarray(last_veto["counts"][i], dtype=np.int64),
                "v2_need": int(last_veto["need"][i]),
                "base": int(last_veto["base"][i]),
                "final": int(last_veto["final"][i]),
                "q": np.asarray(q_rows[i], dtype=np.float32),
                "mask": np.asarray(masks[i], dtype=bool),
            }
            buf = self.buffers.setdefault(env, deque(maxlen=self.window))
            buf.append(snap)
            self.decisions[env] = self.decisions.get(env, 0) + 1


@contextmanager
def recording(recorder: DecisionRecorder) -> Iterator[DecisionRecorder]:
    """Wrap ``Vector61SimdPolicy._apply_veto``: original first, then a read-only capture."""
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    original = Vector61SimdPolicy._apply_veto

    def wrapped(self, sim, slots, masked_q, mask, actions):  # type: ignore[no-untyped-def]
        out = original(self, sim, slots, masked_q, mask, actions)
        if self.veto_slots and self.last_veto is not None:
            slots_arr = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            pick = np.flatnonzero(np.isin(slots_arr[:, 1], list(self.veto_slots)))
            if len(pick):
                q_rows = masked_q.cpu().numpy()[pick]
                recorder.capture(sim, self.last_veto, q_rows, np.asarray(mask)[pick])
        return out

    Vector61SimdPolicy._apply_veto = wrapped
    try:
        yield recorder
    finally:
        Vector61SimdPolicy._apply_veto = original


def pack_window(snaps: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Stack a window of snapshots into flat arrays (ragged cells via offsets)."""

    def ragged(key: str) -> Tuple[np.ndarray, np.ndarray]:
        parts = [np.asarray(s[key], dtype=np.int16).reshape(-1, 2) for s in snaps]
        offsets = np.zeros(len(parts) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(p) for p in parts])
        cells = np.concatenate(parts) if parts else np.zeros((0, 2), np.int16)
        return cells.astype(np.int16), offsets

    out: Dict[str, np.ndarray] = {}
    for key in ("body", "others", "food"):
        out[f"{key}_cells"], out[f"{key}_offsets"] = ragged(key)
    for key in ("frame", "length", "direction", "boost_frames", "v2_need", "base", "final"):
        out[key] = np.asarray([int(s[key]) for s in snaps], dtype=np.int64)
    out["v2_counts"] = np.asarray([s["v2_counts"] for s in snaps], dtype=np.int64).reshape(-1, 3)
    out["q"] = np.asarray([s["q"] for s in snaps], dtype=np.float32).reshape(-1, NUM_ACTIONS)
    out["mask"] = np.asarray([s["mask"] for s in snaps], dtype=bool).reshape(-1, NUM_ACTIONS)
    return out


def unpack_window(packed: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Inverse of :func:`pack_window`."""
    n = len(packed["frame"])
    snaps: List[Dict[str, Any]] = []
    for t in range(n):
        snap: Dict[str, Any] = {}
        for key in ("body", "others", "food"):
            off = packed[f"{key}_offsets"]
            snap[key] = np.asarray(packed[f"{key}_cells"][off[t] : off[t + 1]])
        for key in ("frame", "length", "direction", "boost_frames", "v2_need", "base", "final"):
            snap[key] = int(packed[key][t])
        snap["v2_counts"] = np.asarray(packed["v2_counts"][t])
        snap["q"] = np.asarray(packed["q"][t])
        snap["mask"] = np.asarray(packed["mask"][t])
        snaps.append(snap)
    return snaps


def world_and_state(
    snap: Mapping[str, Any], meta: Mapping[str, Any]
) -> Tuple[StaticWorld, HeroState]:
    """The frozen world and exact hero state of one snapshot."""
    world = StaticWorld(
        width=int(meta["grid_width"]),
        height=int(meta["grid_height"]),
        blocked=frozenset((int(x), int(y)) for x, y in snap["others"]),
        food=frozenset((int(x), int(y)) for x, y in snap["food"]),
        min_boost_length=int(meta["min_boost_length"]),
        boost_cost_frames=int(meta["boost_cost_frames"]),
        trail_food=bool(meta["trail_food"]),
    )
    state = HeroState(
        body=tuple((int(x), int(y)) for x, y in snap["body"]),
        length=int(snap["length"]),
        direction=int(snap["direction"]),
        boost_frames=int(snap["boost_frames"]),
    )
    return world, state


# ---------------------------------------------------------------------------
# Offline analysis of one death
# ---------------------------------------------------------------------------
def model_replay_check(
    snaps: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]
) -> Dict[str, Any]:
    """Replay each taken action through :func:`step_hero` and compare to the next snapshot.

    Consecutive snapshots must be consecutive frames. The fatal decision's predicted
    outcome is reported separately (it should be ``self``).
    """
    mismatches: List[Dict[str, Any]] = []
    compared = 0
    for t in range(len(snaps) - 1):
        if snaps[t + 1]["frame"] != snaps[t]["frame"] + 1:
            mismatches.append({"index": t, "reason": "non-consecutive frames"})
            continue
        world, state = world_and_state(snaps[t], meta)
        nxt, outcome = step_hero(world, state, int(snaps[t]["final"]))
        _, actual = world_and_state(snaps[t + 1], meta)
        compared += 1
        if nxt is None:
            mismatches.append({"index": t, "reason": f"model predicts death ({outcome})"})
            continue
        fields = [
            name
            for name in ("body", "length", "direction", "boost_frames")
            if getattr(nxt, name) != getattr(actual, name)
        ]
        if fields:
            mismatches.append({"index": t, "reason": "state differs", "fields": fields})
    world, state = world_and_state(snaps[-1], meta)
    _, fatal_outcome = step_hero(world, state, int(snaps[-1]["final"]))
    return {
        "transitions_compared": compared,
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:5],
        "fatal_action_model_outcome": fatal_outcome,
    }


def _v2_view(snap: Mapping[str, Any]) -> Dict[str, Any]:
    counts = [int(c) for c in snap["v2_counts"]]
    need = int(snap["v2_need"])
    final = int(snap["final"])
    base = int(snap["base"])
    spacious = [c >= need for c in counts]
    if not any(spacious):
        outcome = "no_spacious"
    elif spacious[base % NUM_DIRECTIONS]:
        outcome = "kept"
    else:
        outcome = "vetoed"
    return {
        "counts": counts,
        "need": need,
        "spacious": spacious,
        "base_action": base,
        "final_action": final,
        "outcome": outcome,
        "taken_direction_count": counts[final % NUM_DIRECTIONS],
        "taken_direction_spacious": spacious[final % NUM_DIRECTIONS],
        "max_count": max(counts),
        "length": int(snap["length"]),
        "mask": [bool(x) for x in snap["mask"]],
    }


def combine_status(depth_only: str, count_only: str) -> str:
    """Count-or-depth verdict from the two one-sided searches (exact equivalence).

    A count-or-depth escape is a continuation that survives ``depth`` frames or reaches a
    spacious state, so it exists iff a depth-only or a count-only escape exists.
    """
    if ESCAPE in (depth_only, count_only):
        return ESCAPE
    if depth_only == NO_ESCAPE and count_only == NO_ESCAPE:
        return NO_ESCAPE
    return UNKNOWN


class FrameEvaluator:
    """Lazy, cached per-frame verdicts for one death window under each criterion.

    The depth-only and count-only searches run per frame (each with its own memo shared
    across the frame's six first actions); the primary count-or-depth verdict is
    :func:`combine_status` of the two.
    """

    def __init__(
        self,
        snaps: Sequence[Mapping[str, Any]],
        meta: Mapping[str, Any],
        depth: int = DEPTH,
        budget: int = NODE_BUDGET,
        slack: int = TAIL_SLACK,
    ) -> None:
        self.snaps = snaps
        self.meta = meta
        self.depth = int(depth)
        self.budget = int(budget)
        self.slack = int(slack)
        self.results: Dict[Tuple[int, str], Dict[int, ActionResult]] = {}
        self.seconds: Dict[Tuple[int, str], float] = {}

    def one_sided(self, t: int, criterion: str) -> Dict[int, ActionResult]:
        key = (int(t), criterion)
        if key not in self.results:
            world, state = world_and_state(self.snaps[t], self.meta)
            t0 = time.perf_counter()
            self.results[key] = analyze_frame(
                world,
                state,
                depth=self.depth,
                budget=self.budget,
                slack=self.slack,
                criterion=criterion,
            )
            self.seconds[key] = time.perf_counter() - t0
        return self.results[key]

    def statuses(self, t: int, criterion: str) -> Dict[int, str]:
        if criterion == COUNT_OR_DEPTH:
            d = self.one_sided(t, DEPTH_ONLY)
            c = self.one_sided(t, COUNT_ONLY)
            return {a: combine_status(d[a].status, c[a].status) for a in d}
        return {a: r.status for a, r in self.one_sided(t, criterion).items()}

    def frame_row(self, t: int) -> Dict[str, Any]:
        """Everything computed at index ``t`` (for the per-death JSON)."""
        row: Dict[str, Any] = {
            "index": int(t),
            "frame": int(self.snaps[t]["frame"]),
            "length": int(self.snaps[t]["length"]),
            "taken": int(self.snaps[t]["final"]),
        }
        for criterion in (DEPTH_ONLY, COUNT_ONLY):
            key = (int(t), criterion)
            if key in self.results:
                row[criterion] = {str(a): r.to_dict() for a, r in self.results[key].items()}
                row[f"seconds_{criterion}"] = self.seconds[key]
        if all((int(t), c) in self.results for c in (DEPTH_ONLY, COUNT_ONLY)):
            row[COUNT_OR_DEPTH] = {str(a): v for a, v in self.statuses(t, COUNT_OR_DEPTH).items()}
        return row


def taken_primary_status(
    world: StaticWorld,
    state: HeroState,
    action: int,
    depth: int = DEPTH,
    budget: int = NODE_BUDGET,
    slack: int = TAIL_SLACK,
) -> Tuple[str, float, int, float]:
    """Count-or-depth verdict of one action: ``(status, seconds, nodes, direct_seconds)``.

    Runs the direct :data:`COUNT_OR_DEPTH` search first (what an always-on shield would
    run; ``direct_seconds`` is its cost). Only if its budget runs out are the two
    one-sided searches tried and combined (:func:`combine_status`).
    """
    t0 = time.perf_counter()
    direct = EscapeSearch(world, depth, budget, slack, COUNT_OR_DEPTH).run(state, action)
    direct_seconds = time.perf_counter() - t0
    nodes = direct.nodes
    if direct.status != UNKNOWN:
        return direct.status, direct_seconds, nodes, direct_seconds
    d = EscapeSearch(world, depth, budget, slack, DEPTH_ONLY).run(state, action)
    nodes += d.nodes
    if d.status == ESCAPE:
        return ESCAPE, time.perf_counter() - t0, nodes, direct_seconds
    c = EscapeSearch(world, depth, budget, slack, COUNT_ONLY).run(state, action)
    nodes += c.nodes
    return combine_status(d.status, c.status), time.perf_counter() - t0, nodes, direct_seconds


def analyze_death(
    snaps: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    depth: int = DEPTH,
    budget: int = NODE_BUDGET,
    slack: int = TAIL_SLACK,
) -> Dict[str, Any]:
    """Walk back from the fatal decision (last snapshot) under all three criteria.

    Primary: :data:`COUNT_OR_DEPTH` (the diagnostic's definition). Sensitivities:
    :data:`DEPTH_ONLY` (survive ``depth`` frames) and :data:`COUNT_ONLY` (reach a state
    with a spacious tail-aware count within ``depth`` frames). A taken-action scan
    (primary criterion) covers the frames the primary walk did not evaluate.
    """
    started = time.perf_counter()
    ev = FrameEvaluator(snaps, meta, depth, budget, slack)
    taken = [int(s["final"]) for s in snaps]
    walks: Dict[str, Dict[str, Any]] = {}
    walk_seconds: Dict[str, float] = {}
    for criterion in (COUNT_OR_DEPTH, DEPTH_ONLY, COUNT_ONLY):
        t0 = time.perf_counter()
        walks[criterion] = walk_back(len(snaps), taken, lambda t, c=criterion: ev.statuses(t, c))
        walk_seconds[criterion] = time.perf_counter() - t0
    primary = walks[COUNT_OR_DEPTH]
    primary_indices = set(
        range(primary["fatal_index"] - primary["frames_evaluated"] + 1, len(snaps))
    )
    scan: List[Dict[str, Any]] = []
    for t in range(len(snaps)):
        if t in primary_indices:
            status = ev.statuses(t, COUNT_OR_DEPTH)[taken[t]]
            scan.append({"index": t, "status": status, "nodes": None, "seconds": None})
            continue
        world, state = world_and_state(snaps[t], meta)
        status, seconds, nodes, direct = taken_primary_status(
            world, state, taken[t], depth, budget, slack
        )
        scan.append(
            {
                "index": t,
                "status": status,
                "nodes": nodes,
                "seconds": seconds,
                "direct_seconds": direct,
            }
        )
    fatal = len(snaps) - 1

    def pnr_block(criterion: str) -> Dict[str, Any]:
        walk = walks[criterion]
        pnr = walk["pnr_index"]
        if pnr is None:
            return {"walk": walk, "pnr_frame": None, "pnr_v2": None, "pnr_row": None}
        return {
            "walk": walk,
            "pnr_frame": int(snaps[pnr]["frame"]),
            "pnr_v2": _v2_view(snaps[pnr]),
            "pnr_row": ev.frame_row(pnr),
        }

    blocks = {criterion: pnr_block(criterion) for criterion in CRITERIA}
    evaluated = sorted({t for t, _ in ev.results}, reverse=True)
    return {
        "window_frames": len(snaps),
        "fatal_frame": int(snaps[-1]["frame"]),
        "fatal_length": int(snaps[-1]["length"]),
        "walk": blocks[COUNT_OR_DEPTH]["walk"],
        "pnr_frame": blocks[COUNT_OR_DEPTH]["pnr_frame"],
        "pnr_v2": blocks[COUNT_OR_DEPTH]["pnr_v2"],
        "pnr_row": blocks[COUNT_OR_DEPTH]["pnr_row"],
        "sensitivity": {c: blocks[c] for c in (DEPTH_ONLY, COUNT_ONLY)},
        "fatal_v2": _v2_view(snaps[fatal]),
        "evaluated_frames": [ev.frame_row(t) for t in evaluated],
        "taken_scan": scan,
        "model_check": model_replay_check(snaps, meta),
        "seconds": {
            **{f"walk_{c}": v for c, v in walk_seconds.items()},
            "one_sided_frame_searches": {
                c: sum(v for (t, cc), v in ev.seconds.items() if cc == c)
                for c in (DEPTH_ONLY, COUNT_ONLY)
            },
            "total": time.perf_counter() - started,
        },
    }


def analyze_death_file(args: Tuple[str, Mapping[str, Any], int, int, int]) -> Dict[str, Any]:
    """Pool worker: load one saved window (npz + meta) and analyze it."""
    path, meta, depth, budget, slack = args
    with np.load(path) as data:
        packed = {k: data[k] for k in data.files}
    return analyze_death(unpack_window(packed), meta, depth, budget, slack)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def on_ac_power() -> bool:
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True)
    return "AC Power" in out.stdout


def _git() -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty_paths": run("status", "--porcelain")}


def write_new_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def world_seeds(domain: str, count: int) -> List[int]:
    from research.apex_safety_20260926 import dev_screen as ds

    return ds.screen_seeds(count, domain, NAMESPACE)


def disjointness(seeds: Sequence[int], domain: str) -> Dict[str, Any]:
    """Check the worlds against every earlier namespace we can recompute."""
    from research.apex_safety_20260926 import dev_screen as ds
    from research.apex_veto_v4_screen_20261001 import screen as v4

    earlier: Dict[str, Sequence[str]] = dict(v4.EARLIER_DOMAINS)
    earlier[v4.DOMAIN] = (v4.NAMESPACE,)  # the v4 screen itself (and smoke, already there)
    earlier[v4.SMOKE_DOMAIN] = (v4.NAMESPACE,)
    for other in (DOMAIN, SMOKE_DOMAIN):
        if other != domain:
            earlier[other] = (NAMESPACE,)
    extra = {
        f"{name}/{purpose}[0:{EXCLUSION_PREFIX}]": [
            ds.uint32_seed(name, purpose, i) for i in range(EXCLUSION_PREFIX)
        ]
        for name, purposes in earlier.items()
        for purpose in purposes
    }
    report = ds.disjointness_report(
        seeds,
        ds.DEFAULT_PILOT_OUTPUT if ds.DEFAULT_PILOT_OUTPUT.is_dir() else None,
        screen_domain=domain,
        extra_namespaces=extra,
    )
    report["simd_parity_h5000"] = "used apex-safety-screen-v1 worlds (checked above)"
    return report


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="smoke namespace, any horizon")
    parser.add_argument("--worlds-per-mix", type=int, default=WORLDS_PER_MIX)
    parser.add_argument("--frames", type=int, default=None, help="smoke only")
    parser.add_argument("--mixes", default="frozen,scripted,mixed")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--budget", type=int, default=NODE_BUDGET)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        print(f"refusing: {out} already exists (create-only)", file=sys.stderr)
        return 2
    if not args.smoke and (args.frames is not None or args.worlds_per_mix != WORLDS_PER_MIX):
        print("refusing: --frames/--worlds-per-mix overrides are smoke-only", file=sys.stderr)
        return 2
    wrong_env = {k: os.environ.get(k) for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v}
    if wrong_env:
        print(f"refusing: environment must have {REQUIRED_ENV}, got {wrong_env}", file=sys.stderr)
        return 2
    if not on_ac_power():
        print("refusing: `pmset -g batt` does not show 'AC Power'", file=sys.stderr)
        return 2
    if not SLOT_LOCK.is_file():
        print(f"refusing: slot lock {SLOT_LOCK} missing (never created here)", file=sys.stderr)
        return 2
    lock = SLOT_LOCK.open("r")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print(f"refusing: CPU slot 1 busy ({SLOT_LOCK})", file=sys.stderr)
        return 2
    try:
        return _run(args, out, argv)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _run(args: argparse.Namespace, out: Path, argv: Sequence[str] | None) -> int:
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    from research.apex_safety_20260926 import dev_screen as ds
    from src.core.config_loader import load_and_initialize_config
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts.tournament_eval import evaluation_profile_for_name
    from src.simd_env.eval_engine import run_simd_eval

    started_total = time.monotonic()
    domain = SMOKE_DOMAIN if args.smoke else DOMAIN
    horizon = int(args.frames) if (args.smoke and args.frames) else ds.HORIZON
    mixes = [m for m in args.mixes.split(",") if m]
    if any(m not in ds.MIXES for m in mixes):
        raise SystemExit(f"unknown mix in {mixes}")
    seeds = world_seeds(domain, int(args.worlds_per_mix))
    report = disjointness(seeds, domain)
    if not report["disjoint"]:
        raise SystemExit(f"world seeds overlap an earlier namespace: {report}")
    if ds.sha256_file(ds.DEFAULT_CONFIG) != ds.CONFIG_SHA256:
        raise SystemExit("deployment config bytes differ from dev_screen.CONFIG_SHA256")
    load_and_initialize_config(str(ds.DEFAULT_CONFIG))
    profile = evaluation_profile_for_name(ds.PROFILE_NAME, ds.HORIZON)
    if horizon != ds.HORIZON:  # smoke only: the same profile with a shorter horizon
        from dataclasses import replace

        profile = replace(profile, scored_horizon=horizon)
    elif profile.digest != ds.PROFILE_DIGEST:
        raise SystemExit("resolved profile digest differs from dev_screen.PROFILE_DIGEST")
    veto_sha = ds.sha256_file(REPO / "src" / "evaluation" / "safety_veto.py")

    rows_by_mix = {
        mix: [row for row in ds._design_rows(seeds) if row["mix"] == mix] for mix in mixes
    }
    out.mkdir(parents=True)
    (out / "deaths").mkdir()
    (out / "windows").mkdir()
    (out / "records").mkdir()
    snapshots = ds.snapshot_checkpoints(ds.DEFAULT_CHECKPOINT_DIR, out)
    lookup = ds.agent_lookup(snapshots)
    hero = lookup[ds.CHAMPION[1]]
    write_new_json(
        out / "intent.json",
        {
            "schema_version": SCHEMA,
            "authority": AUTHORITY,
            "argv": list(sys.argv if argv is None else argv),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git": _git(),
            "smoke": bool(args.smoke),
            "worlds": {
                "domain": domain,
                "namespace": NAMESPACE,
                "recipe": "uint32 big-endian prefix of sha256('<domain>|<namespace>|<i>')",
                "seeds": seeds,
                "shared_by_mixes": mixes,
                "disjointness": report,
            },
            "config": {"path": str(ds.DEFAULT_CONFIG), "sha256": ds.CONFIG_SHA256},
            "profile": {"name": ds.PROFILE_NAME, "digest": profile.digest, "horizon": horizon},
            "checkpoint_snapshots": snapshots,
            "hero": {"name": ds.CHAMPION[0], "sha256": ds.CHAMPION[1]},
            "veto": {
                "method": "free-space-veto/v2-speed-preserving (released)",
                "source": "src/evaluation/safety_veto.py",
                "source_sha256": veto_sha,
            },
            "engine": {"engine": "simd", "vector61": True, "vector61_forward": VECTOR61_FORWARD},
            "capture": {
                "window_decisions": int(args.window),
                "point": "Vector61SimdPolicy._apply_veto (read-only wrapper, original first)",
            },
            "search": {
                "depth": int(args.depth),
                "node_budget_per_action": int(args.budget),
                "tail_release_slack": TAIL_SLACK,
                "others": "static walls (all cells of other live snakes at that frame)",
                "own_moves": "exact BatchSim movement incl. boost and growth from that "
                "frame's food",
                "escape": "tail-aware count >= v2 need, or survive `depth` frames",
                "primary_criterion": COUNT_OR_DEPTH,
                "primary_computed_as": "combine(depth_only, count_only) (exact equivalence)",
                "sensitivity_criteria": [DEPTH_ONLY, COUNT_ONLY],
                "prune": "doomed(): sound region bound for length > 3 * remaining + 2",
                "budget_exhausted": "unknown (never guessed)",
            },
            "threads": {
                "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                "torch_intraop": torch.get_num_threads(),
                "torch_interop": torch.get_num_interop_threads(),
                "analysis_workers": int(args.workers),
            },
            "slot_lock": str(SLOT_LOCK),
            "ac_power_checked_at_start": True,
        },
    )

    jobs: List[Tuple[str, Dict[str, Any]]] = []  # (npz path, death info)
    sim_batches: List[Dict[str, Any]] = []
    all_deaths: List[Dict[str, Any]] = []
    meta: Optional[Dict[str, Any]] = None
    for mix in mixes:
        rows = rows_by_mix[mix]
        mix_seeds = [row["world_seed"] for row in rows]
        rosters = {
            row["world_seed"]: [lookup[s["member_sha256"]] for s in row["slots"]] for row in rows
        }
        identities = {row["world_seed"]: _expected_world_identity(row) for row in rows}
        recorder = DecisionRecorder(args.window)
        t0 = time.monotonic()
        with recording(recorder):
            records = run_simd_eval(
                hero,
                rosters[mix_seeds[0]],
                horizon,
                mix_seeds,
                profile=profile,
                opponent_specs_by_world=rosters,
                world_identities=identities,
                mix_id=mix,
                vector61=True,
                hero_safety_veto=True,
                vector61_forward=VECTOR61_FORWARD,
            )
        wall = time.monotonic() - t0
        meta = meta or recorder.world_meta
        deaths_here = 0
        for env, (row, record) in enumerate(zip(rows, records)):
            seed = int(row["world_seed"])
            counters = record["probes"]["safety_veto"]["counters"]
            cause = record["probes"].get("death_cause")
            entry = {
                "mix": mix,
                "world_index": env,
                "world_seed": seed,
                "deaths": record["deaths"],
                "death_cause": cause,
                "survival_fraction": record["survival_fraction"],
                "mass_integral": record["mass_integral"],
                "veto_decisions": counters["decisions"],
                "captured_decisions": recorder.decisions.get(env, 0),
            }
            write_new_json(out / "records" / f"{mix}-{seed}.json", {**entry, "record": record})
            if entry["captured_decisions"] != entry["veto_decisions"]:
                raise RuntimeError(f"capture count differs from veto decisions: {entry}")
            all_deaths.append(entry)
            if cause == "self" and float(record["deaths"]) >= 1:
                snaps = list(recorder.buffers[env])
                path = out / "windows" / f"{mix}-{seed}.npz"
                np.savez_compressed(path, **pack_window(snaps))
                jobs.append((str(path), entry))
                deaths_here += 1
        sim_batches.append(
            {
                "mix": mix,
                "episodes": len(mix_seeds),
                "wall_seconds": wall,
                "self_deaths": deaths_here,
            }
        )
        print(json.dumps({"mix": mix, "wall_s": round(wall, 1), "self_deaths": deaths_here}))
        del recorder
    sim_seconds = time.monotonic() - started_total

    assert meta is not None
    t_an = time.monotonic()
    work = [(p, meta, int(args.depth), int(args.budget), TAIL_SLACK) for p, _ in jobs]
    if int(args.workers) > 1 and len(work) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(int(args.workers)) as pool:
            results = pool.map(analyze_death_file, work, chunksize=1)
    else:
        results = [analyze_death_file(w) for w in work]
    analysis_seconds = time.monotonic() - t_an
    death_rows: List[Dict[str, Any]] = []
    for (path, entry), result in zip(jobs, results):
        row = {**entry, **result, "window_file": Path(path).name}
        write_new_json(out / "deaths" / f"{entry['mix']}-{entry['world_seed']}.json", row)
        death_rows.append(row)

    summary = build_summary(death_rows, all_deaths, sim_batches)
    summary["timing"] = {
        "total_wall_seconds": time.monotonic() - started_total,
        "simulation_wall_seconds": sim_seconds,
        "analysis_wall_seconds": analysis_seconds,
        "simulation_batches": sim_batches,
    }
    summary.update(
        {
            "schema_version": SCHEMA,
            "authority": AUTHORITY,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "smoke": bool(args.smoke),
            "search": {"depth": int(args.depth), "node_budget_per_action": int(args.budget)},
        }
    )
    for sha, path in snapshots.items():
        if ds.sha256_file(Path(path)) != sha:
            raise RuntimeError(f"checkpoint snapshot {path} changed during the run")
    write_new_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("pnr", "timing")}, indent=2, default=str))
    return 0


def _ms_stats(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float) * 1000.0
    return {
        "n": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }


def build_summary(
    death_rows: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    sim_batches: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summary over analyzed self deaths and all episodes."""
    causes: Dict[str, Dict[str, int]] = {}
    for ep in episodes:
        cell = causes.setdefault(ep["mix"], {})
        label = ep["death_cause"] if float(ep["deaths"]) >= 1 else "survived"
        cell[str(label)] = cell.get(str(label), 0) + 1
    pnr = summarize_walks(death_rows)
    sensitivity = {
        c: summarize_walks(
            [{"mix": d["mix"], "walk": d["sensitivity"][c]["walk"]} for d in death_rows]
        )
        for c in (DEPTH_ONLY, COUNT_ONLY)
    }
    found = [d for d in death_rows if d["walk"]["pnr_index"] is not None]
    v2_outcomes = _tally(d["pnr_v2"]["outcome"] for d in found)
    escape_kinds: Dict[str, int] = {}
    for d in found:
        row = d["pnr_row"]
        kinds = set()
        for criterion, kind in ((DEPTH_ONLY, "survive_depth"), (COUNT_ONLY, "spacious_count")):
            if any(r["status"] == ESCAPE for r in row.get(criterion, {}).values()):
                kinds.add(kind)
        label = "+".join(sorted(kinds)) or "none"
        escape_kinds[label] = escape_kinds.get(label, 0) + 1
    frame_seconds = {
        c: [
            f[f"seconds_{c}"]
            for d in death_rows
            for f in d["evaluated_frames"]
            if f"seconds_{c}" in f
        ]
        for c in (DEPTH_ONLY, COUNT_ONLY)
    }
    taken_seconds = [
        s["direct_seconds"]
        for d in death_rows
        for s in d["taken_scan"]
        if s.get("direct_seconds") is not None
    ]
    model = {
        "transitions_compared": sum(d["model_check"]["transitions_compared"] for d in death_rows),
        "mismatches": sum(d["model_check"]["mismatches"] for d in death_rows),
        "fatal_action_model_outcomes": _tally(
            d["model_check"]["fatal_action_model_outcome"] for d in death_rows
        ),
    }
    return {
        "episodes": len(episodes),
        "episode_outcomes_by_mix": dict(sorted(causes.items())),
        "pnr": pnr,
        "pnr_sensitivity": sensitivity,
        "pnr_v2_outcome": v2_outcomes,
        "pnr_v2_taken_direction_spacious": sum(
            bool(d["pnr_v2"]["taken_direction_spacious"]) for d in found
        ),
        "pnr_v2_any_direction_spacious": sum(any(d["pnr_v2"]["spacious"]) for d in found),
        "pnr_escape_kinds": dict(sorted(escape_kinds.items())),
        "fatal_lengths": sorted(int(d["fatal_length"]) for d in death_rows),
        "fatal_v2_outcome": _tally(d["fatal_v2"]["outcome"] for d in death_rows),
        "taken_scan_status": _tally(s["status"] for d in death_rows for s in d["taken_scan"]),
        "model_check": model,
        "search_cost": {
            "depth_only_all_six_actions_per_evaluated_frame": _ms_stats(frame_seconds[DEPTH_ONLY]),
            "count_only_all_six_actions_per_evaluated_frame": _ms_stats(frame_seconds[COUNT_ONLY]),
            "taken_action_direct_count_or_depth_outside_primary_walk": _ms_stats(taken_seconds),
            "note": "pre-death windows only (biased to hard states); pure Python, 1 core",
        },
    }


def _tally(values: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    raise SystemExit(main())
