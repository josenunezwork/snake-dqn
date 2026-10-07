"""Content-validated per-snake geometry cache for the observation / action-mask hot paths.

Every AI snake, every frame, scans every OTHER snake's body several times: the safe-action
simulation (``ai_snake.simulate_relative_action_fatality``), the per-action danger features
and the free-space flood fill (``snake_state``). The same other-snake object (a pre-move
snapshot during action selection, the live snake during reward/next-state capture) is
scanned by up to five observers in the same phase. This module builds the derived geometry
of one snake once and hands it to every observer.

Exactness. An entry (keyed by the snake's game id) is reused only when the snake's
``segments`` are element-for-element equal to the cached tuple, its ``length`` is equal and
the cell size is equal. The derived data are pure functions of exactly those inputs, so a cache hit
returns what a rebuild would return. Nothing here changes any game rule or float
expression; callers keep their own exactness arguments (see ``exact_radius_sq``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, FrozenSet, List, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from src.game.snake import Snake

Point = Tuple[int, int]
Cell = Tuple[int, int]

_MAX_ENTRIES = 64
_cache: Dict[int, tuple] = {}


class SnakeGeometry:
    """Derived geometry of one snake at one cell size.

    Attributes:
        obstacle_cells: ``cell -> [points]`` for the snake's collision points as seen by
            another snake: its head, then ``segments[1:]`` minus the vacating tail when the
            snake is full (``len(segments) >= length`` and ``len(segments) > 1``). Cell is
            ``(x // ss, y // ss)``. Points keep body order within a bucket.
        has_obstacles: Whether ``obstacle_cells`` is non-empty.
        body_cells: ``frozenset`` of ``(x // ss, y // ss)`` over ALL segments (the
            free-space flood fill's walls).
        neck_to_tail_cells: ``frozenset`` of the cells of ``segments[1:]`` (the body a
            head can hit: ``GameLogic.check_body_collision``).
        bbox: ``(min_x, max_x, min_y, max_y)`` of the obstacle points (empty: max < min).
    """

    __slots__ = ("obstacle_cells", "has_obstacles", "body_cells", "neck_to_tail_cells", "bbox")

    def __init__(self, segments: Tuple[Point, ...], length: int, ss: int) -> None:
        cells: Dict[Cell, List[Point]] = {}
        stop = len(segments)
        if stop >= length and stop > 1:
            stop -= 1
        if segments:
            px, py = segments[0]
            cells[(px // ss, py // ss)] = [(px, py)]
            for index in range(1, stop):
                px, py = segments[index]
                key = (px // ss, py // ss)
                bucket = cells.get(key)
                if bucket is None:
                    cells[key] = [(px, py)]
                else:
                    bucket.append((px, py))
        self.obstacle_cells = cells
        self.has_obstacles = bool(cells)
        # Bounding box of the obstacle points: (min_x, max_x, min_y, max_y).
        if cells:
            xs = [segments[0][0]] + [segments[i][0] for i in range(1, stop)]
            ys = [segments[0][1]] + [segments[i][1] for i in range(1, stop)]
            self.bbox = (min(xs), max(xs), min(ys), max(ys))
        else:
            self.bbox = (0, -1, 0, -1)
        self.body_cells: FrozenSet[Cell] = frozenset([(x // ss, y // ss) for x, y in segments])
        self.neck_to_tail_cells: FrozenSet[Cell] = frozenset(
            [(x // ss, y // ss) for x, y in segments[1:]]
        )


def geometry_for(snake: "Snake", ss: int) -> SnakeGeometry:
    """The (cached) :class:`SnakeGeometry` of ``snake`` at cell size ``ss``.

    Entries are keyed by the snake's game ``id`` so a pre-move snapshot and the live snake
    it was copied from share one entry, and every hit is validated on content (segments,
    length, cell size), so a hit is exactly a rebuild.
    """
    segments = tuple(snake.segments)
    length = snake.length
    key = getattr(snake, "id", None)
    if not isinstance(key, int):
        key = ("object", id(snake))
    entry = _cache.get(key)
    if entry is not None and entry[1] == length and entry[2] == ss and entry[0] == segments:
        return entry[3]
    geometry = SnakeGeometry(segments, length, ss)
    if len(_cache) >= _MAX_ENTRIES and key not in _cache:
        _cache.clear()
    _cache[key] = (segments, length, ss, geometry)
    return geometry


def clear_geometry_cache() -> None:
    """Drop every cached entry (tests; never needed for correctness)."""
    _cache.clear()


def near_obstacle_min_d2(
    geometries: List[SnakeGeometry], qx: int, qy: int, ss: int, reach: int
) -> "int | None":
    """Minimum ``(qx - px) ** 2 + (qy - py) ** 2`` over obstacle points within ``reach`` cells.

    Scans the ``(2 * reach + 1)**2`` cell neighbourhood of ``(qx // ss, qy // ss)`` in
    every geometry whose bounding box comes within ``reach * ss`` pixels of the query
    (a point outside that box has ``|dx| > reach * ss`` or ``|dy| > reach * ss`` and so a
    cell offset above ``reach``). Returns ``None`` when no point is in range. Callers
    decide which distances can matter (they pick ``reach`` from their own threshold).
    """
    best = None
    span = reach * ss
    cx = qx // ss
    cy = qy // ss
    for geometry in geometries:
        min_x, max_x, min_y, max_y = geometry.bbox
        if qx < min_x - span or qx > max_x + span or qy < min_y - span or qy > max_y + span:
            continue
        cells = geometry.obstacle_cells
        for kx in range(cx - reach, cx + reach + 1):
            for ky in range(cy - reach, cy + reach + 1):
                bucket = cells.get((kx, ky))
                if bucket is None:
                    continue
                for px, py in bucket:
                    d2 = (qx - px) ** 2 + (qy - py) ** 2
                    if best is None or d2 < best:
                        best = d2
    return best
