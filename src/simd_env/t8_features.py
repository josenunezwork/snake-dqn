"""Arrival-race topology features for the T8 diagnostic.

The functions here are deliberately read-only helpers.  They model a
hypothetical hero action exactly as T5 does, then compare the hero's tail-aware
arrival time with a simultaneous, unit-speed flood seeded at every live
opponent head.  A compiled NumPy-array kernel is parity-tested against the
small Python reference.  This is a diagnostic feature extractor; it never
chooses an action or changes simulator state.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Set, Tuple

import numpy as np
from numba import njit

from src.simd_env.t5_features import _advance, _body, _log_area, _release_map

__all__ = [
    "FEATURE_NAMES",
    "compute_features",
    "compute_slot_features",
    "race_features",
]

FEATURE_NAMES = ("voronoi_area_log", "contested_fraction", "contested_margin")
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
Cell = Tuple[int, int]


def _inside(cell: Cell, width: int, height: int) -> bool:
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def _arrival_times(
    starts: Iterable[Cell],
    width: int,
    height: int,
    blocked: Set[Cell],
    release: Dict[Cell, int],
) -> Dict[Cell, int]:
    """Return first-arrival times through the T5 tail-aware static world.

    A start is admitted even when it is an opponent head in ``blocked``.  This
    lets heads seed the opponent flood while preventing either flood from
    traversing the bodies they currently occupy.  Hero tail cells use T5's
    conservative release distances for both floods.
    """
    times: Dict[Cell, int] = {}
    queue: deque[tuple[Cell, int]] = deque()
    for start in starts:
        if _inside(start, width, height) and start not in times:
            times[start] = 0
            queue.append((start, 0))
    while queue:
        cell, distance = queue.popleft()
        for dx, dy in _NEIGHBOURS:
            nxt = (cell[0] + dx, cell[1] + dy)
            next_distance = distance + 1
            if (
                nxt not in times
                and _inside(nxt, width, height)
                and nxt not in blocked
                and (nxt not in release or next_distance >= release[nxt])
            ):
                times[nxt] = next_distance
                queue.append((nxt, next_distance))
    return times


@njit(cache=True, nogil=True)
def _race_kernel(
    hero_x: int,
    hero_y: int,
    heads: np.ndarray,
    blocked: np.ndarray,
    release: np.ndarray,
) -> tuple[int, float, int]:
    """Compiled equivalent of the two tail-aware reference floods."""
    height, width = blocked.shape
    hero = np.full((height, width), -1, np.int32)
    opponent = np.full((height, width), -1, np.int32)
    qx = np.empty(height * width, np.int32)
    qy = np.empty(height * width, np.int32)
    head, tail = 0, 1
    hero[hero_y, hero_x] = 0
    qx[0], qy[0] = hero_x, hero_y
    while head < tail:
        x, y = qx[head], qy[head]
        head += 1
        distance = hero[y, x] + 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and hero[ny, nx] < 0
                and not blocked[ny, nx]
                and (release[ny, nx] == 0 or distance >= release[ny, nx])
            ):
                hero[ny, nx] = distance
                qx[tail], qy[tail] = nx, ny
                tail += 1
    head = 0
    tail = 0
    for index in range(heads.shape[0]):
        x, y = heads[index, 0], heads[index, 1]
        if 0 <= x < width and 0 <= y < height and opponent[y, x] < 0:
            opponent[y, x] = 0
            qx[tail], qy[tail] = x, y
            tail += 1
    while head < tail:
        x, y = qx[head], qy[head]
        head += 1
        distance = opponent[y, x] + 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and opponent[ny, nx] < 0
                and not blocked[ny, nx]
                and (release[ny, nx] == 0 or distance >= release[ny, nx])
            ):
                opponent[ny, nx] = distance
                qx[tail], qy[tail] = nx, ny
                tail += 1
    region, strict, contested, margin = 0, 0, 0, width * height + 1
    for y in range(height):
        for x in range(width):
            if hero[y, x] >= 0:
                region += 1
                if opponent[y, x] < 0 or hero[y, x] < opponent[y, x]:
                    strict += 1
                else:
                    contested += 1
                    if hero[y, x] < margin:
                        margin = hero[y, x]
    if region == 0:
        return 0, 0.0, 0
    return strict, contested / region, margin


def _race_features_compiled(
    hero_start: Cell,
    opponent_heads: Iterable[Cell],
    width: int,
    height: int,
    permanently_blocked: Set[Cell],
    release: Dict[Cell, int],
) -> tuple[int, float, int]:
    """Marshal sparse reference inputs for the compiled BFS kernel."""
    blocked = np.zeros((height, width), dtype=np.bool_)
    releases = np.zeros((height, width), dtype=np.int32)
    for x, y in permanently_blocked:
        if _inside((x, y), width, height):
            blocked[y, x] = True
    for (x, y), time in release.items():
        if _inside((x, y), width, height):
            releases[y, x] = time
    heads = np.asarray(list(opponent_heads), dtype=np.int32).reshape(-1, 2)
    return _race_kernel(hero_start[0], hero_start[1], heads, blocked, releases)


def race_features(
    hero_start: Cell,
    opponent_heads: Iterable[Cell],
    width: int,
    height: int,
    permanently_blocked: Set[Cell],
    release: Dict[Cell, int],
    *,
    compiled: bool = True,
) -> tuple[int, float, int]:
    """Return raw arrival-race measurements for one post-action state.

    The returned tuple is ``(strict_voronoi_area, contested_fraction,
    contested_margin)``.  The fraction's denominator is the hero's entire
    tail-aware reachable region.  A contested cell is any hero-reachable cell
    where an opponent arrives first or ties.  Margin is the minimum hero
    arrival time among those cells (the nearest contested frontier).  An
    uncontested valid region uses the finite sentinel ``width * height + 1``;
    this deliberately differs from both an immediate contest (zero) and an
    invalid action row (all feature values zero).
    """
    if width <= 0 or height <= 0 or not _inside(hero_start, width, height):
        return 0, 0.0, 0
    if compiled:
        return _race_features_compiled(
            hero_start, opponent_heads, width, height, permanently_blocked, release
        )
    hero = _arrival_times((hero_start,), width, height, permanently_blocked, release)
    if not hero:
        return 0, 0.0, 0
    opponent = _arrival_times(opponent_heads, width, height, permanently_blocked, release)
    contested = [cell for cell, time in hero.items() if opponent.get(cell, np.inf) <= time]
    strict = sum(opponent.get(cell, np.inf) > time for cell, time in hero.items())
    margin = min((hero[cell] for cell in contested), default=width * height + 1)
    return strict, float(len(contested) / len(hero)), margin


def compute_slot_features(sim, env: int, slot: int, *, raw_area: bool = False) -> np.ndarray:
    """Compute three arrival-race values for each action of one live slot.

    Output is action-major ``float32`` with shape ``(18,)``.  By default the
    first value for each action is T5's capped logarithmic representation of an
    uncapped strict Voronoi count.  ``raw_area=True`` exposes that count itself
    for the T1 LAD comparison; it leaves the other two columns unchanged.
    Invalid/dead action rows are all zero.
    """
    out = np.zeros((6, len(FEATURE_NAMES)), dtype=np.float32)
    if not bool(sim.alive[env, slot]):
        return out.reshape(-1)
    width, height = int(sim.grid_w), int(sim.grid_h)
    length = int(sim.length[env, slot])
    boost_frames = int(sim.boost_frames[env, slot])
    heading = int(sim.direction[env, slot])
    mine = _body(sim, env, slot)
    opponents: Set[Cell] = set()
    heads: list[Cell] = []
    for other in range(int(sim.S)):
        if other != slot and bool(sim.alive[env, other]):
            body = _body(sim, env, other)
            if body:
                heads.append(body[0])
                opponents.update(body)
    min_boost = int(sim.cfg.min_boost_length)
    boost_cost = int(sim.cfg.boost_length_cost_frames)
    for action in range(6):
        direction = (heading + action % 3 - 1) % 4
        moved = _advance(
            mine,
            length,
            boost_frames,
            direction,
            action >= 3,
            min_boost,
            boost_cost,
            width,
            height,
            opponents,
        )
        if moved is None:
            continue
        post_body, post_length, _ = moved
        release, _tail_cells = _release_map(post_body, post_length)
        area, contested, margin = race_features(
            post_body[0], heads, width, height, opponents, release
        )
        out[action] = (float(area) if raw_area else _log_area(area), contested, float(margin))
    return out.reshape(-1)


def compute_features(sim, *, raw_area: bool = False) -> np.ndarray:
    """Return arrival features as ``float32 (E, S, 18)`` without mutation."""
    out = np.zeros((int(sim.E), int(sim.S), 18), dtype=np.float32)
    for env in range(int(sim.E)):
        for slot in range(int(sim.S)):
            out[env, slot] = compute_slot_features(sim, env, slot, raw_area=raw_area)
    return out
