"""Read-only topology measurements for the T9 fatal/surviving divergence study.

The study deliberately uses a *static* component world: only own cells that
vacate on the next transition (TTL <= 1) are open; every opponent cell remains
blocked.  Arrival races are a separate, optimistic time-aware diagnostic.  No
helper here mutates the supplied simulator.
"""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numba import njit

from research.t1_commitment_20261015.topology import _label_components, component_geometry
from src.simd_env.batch_sim import CARDINAL

Cell = Tuple[int, int]
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_HORIZONS = (16, 32, 64, 128, 256)

__all__ = ["measure"]


def _inside(cell: Cell, width: int, height: int) -> bool:
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def _bfs_distance(start: Cell, goal: Cell, blocked: np.ndarray) -> Optional[int]:
    """Return a static four-neighbor distance, or ``None`` when unreachable."""
    height, width = blocked.shape
    if not _inside(start, width, height) or not _inside(goal, width, height):
        return None
    if blocked[start[1], start[0]] or blocked[goal[1], goal[0]]:
        return None
    queue: deque[tuple[Cell, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        cell, distance = queue.popleft()
        if cell == goal:
            return distance
        for dx, dy in _NEIGHBORS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt not in seen and _inside(nxt, width, height) and not blocked[nxt[1], nxt[0]]:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def _food_cells(sim: Any, e: int) -> List[Cell]:
    """Read food through BatchSim's public-in-practice per-environment list."""
    return sorted((int(x), int(y)) for x, y in sim.food_cells[e])


def _arrival_reference(
    starts: Iterable[Cell],
    blocked: np.ndarray,
    release: np.ndarray,
    *,
    horizon: int = 256,
) -> np.ndarray:
    """Compute time-aware arrivals with ``max(previous + 1, ttl)`` waiting.

    Sources are admitted even if their cells are otherwise blocked.  This opens
    only the source cell for initialization, so a source head cannot turn its
    occupied body into traversable terrain.
    """
    height, width = blocked.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    heap: List[tuple[int, int, int]] = []
    for x, y in starts:
        if _inside((x, y), width, height) and dist[y, x] < 0:
            dist[y, x] = 0
            heappush(heap, (0, x, y))
    while heap:
        current, x, y = heappop(heap)
        if current != int(dist[y, x]) or current >= horizon:
            continue
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not _inside((nx, ny), width, height) or blocked[ny, nx]:
                continue
            candidate = max(current + 1, int(release[ny, nx]))
            if candidate <= horizon and (dist[ny, nx] < 0 or candidate < int(dist[ny, nx])):
                dist[ny, nx] = candidate
                heappush(heap, (candidate, nx, ny))
    return dist


@njit(cache=True, nogil=True)
def _arrival_bucket_kernel(
    starts: np.ndarray, blocked: np.ndarray, release: np.ndarray, horizon: int
) -> np.ndarray:
    """Bounded integer-time Dijkstra for ``max(arrival + 1, ttl)`` costs."""
    height, width = blocked.shape
    cells = height * width
    dist = np.full((height, width), -1, np.int32)
    buckets = np.empty((horizon + 1, cells), np.int32)
    counts = np.zeros(horizon + 1, np.int32)
    for index in range(starts.shape[0]):
        x, y = starts[index, 0], starts[index, 1]
        if 0 <= x < width and 0 <= y < height and dist[y, x] < 0:
            dist[y, x] = 0
            buckets[0, counts[0]] = y * width + x
            counts[0] += 1
    for current in range(horizon + 1):
        for index in range(counts[current]):
            cell = buckets[current, index]
            y, x = cell // width, cell % width
            if dist[y, x] != current or current >= horizon:
                continue
            for delta in range(4):
                if delta == 0:
                    nx, ny = x + 1, y
                elif delta == 1:
                    nx, ny = x - 1, y
                elif delta == 2:
                    nx, ny = x, y + 1
                else:
                    nx, ny = x, y - 1
                if nx < 0 or nx >= width or ny < 0 or ny >= height or blocked[ny, nx]:
                    continue
                candidate = current + 1
                if release[ny, nx] > candidate:
                    candidate = release[ny, nx]
                if candidate <= horizon and dist[ny, nx] < 0:
                    dist[ny, nx] = candidate
                    buckets[candidate, counts[candidate]] = ny * width + nx
                    counts[candidate] += 1
    return dist


def _arrival(
    starts: Iterable[Cell],
    blocked: np.ndarray,
    release: np.ndarray,
    *,
    horizon: int = 256,
) -> np.ndarray:
    """Compiled bounded equivalent of :func:`_arrival_reference`."""
    starts_array = np.asarray(list(starts), dtype=np.int32).reshape(-1, 2)
    return _arrival_bucket_kernel(starts_array, blocked, release, horizon)


def _race_counts(
    head: Cell,
    opponent_heads: List[Cell],
    owners: np.ndarray,
    ttl: np.ndarray,
) -> Dict[str, int]:
    """Count hero-first cells under the T9 full-board optimistic arrival race."""
    # There are no permanent body blockers in this optimistic race.  All
    # current segments, regardless of owner, release at their present TTL.
    # _arrival admits source heads at zero only for flood initialization.
    release = np.where(owners >= 0, ttl, 0).astype(np.int32, copy=False)
    open_board = np.zeros_like(owners, dtype=bool)
    hero = _arrival((head,), open_board, release)
    opponent = _arrival(opponent_heads, open_board, release)
    out: Dict[str, int] = {}
    for horizon in _HORIZONS:
        hero_here = (hero >= 0) & (hero <= horizon)
        out[f"race_{horizon}"] = int(
            np.count_nonzero(hero_here & ((opponent < 0) | (hero < opponent)))
        )
    return out


def _perimeter(
    label: int,
    labels: np.ndarray,
    owners: np.ndarray,
    head: Cell,
) -> Dict[str, float | int]:
    """Classify each exposed cardinal edge of a labelled head component."""
    height, width = labels.shape
    counts = {"own": 0, "opponent": 0, "wall": 0}
    outside15 = 0
    total = 0
    maximum = 0
    for y, x in np.argwhere(labels == label):
        x, y = int(x), int(y)
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if _inside((nx, ny), width, height) and int(labels[ny, nx]) == label:
                continue
            total += 1
            maximum = max(maximum, max(abs(nx - head[0]), abs(ny - head[1])))
            if max(abs(nx - head[0]), abs(ny - head[1])) > 15:
                outside15 += 1
            if not _inside((nx, ny), width, height):
                counts["wall"] += 1
            elif int(owners[ny, nx]) == 0:
                counts["own"] += 1
            elif int(owners[ny, nx]) >= 0:
                counts["opponent"] += 1
            else:  # Defensive malformed geometry: still account for the edge.
                counts["wall"] += 1
    divisor = float(total) if total else 1.0
    return {
        "boundary_own": counts["own"] / divisor,
        "boundary_opponent": counts["opponent"] / divisor,
        "boundary_wall": counts["wall"] / divisor,
        "boundary_max_chebyshev": int(maximum),
        "boundary_outside15_fraction": outside15 / divisor,
    }


def measure(sim: Any, action: int, e: int = 0) -> Dict[str, object]:
    """Return JSON-serializable T9 diagnostics for prepared live hero slot zero.

    ``action`` is used solely for the first-cell food-direction measure.  All
    other values describe the supplied decision state and therefore remain
    directly comparable between the recorded fatal and surviving continuations.
    """
    if not 0 <= action <= 5:
        raise ValueError(f"expected action in [0, 5], got {action}")
    if not bool(sim.alive[e, 0]):
        raise ValueError("T9 metrics require a live hero in slot zero")
    geometry = component_geometry(sim, e)
    width, height = int(geometry["width"]), int(geometry["height"])
    head = tuple(int(v) for v in geometry["head"])
    tail = tuple(int(v) for v in geometry["tail"])
    owners = geometry["owners"]
    ttl = geometry["ttl"]

    # T9's static rule: only a TTL-one own segment is free.  The current hero
    # head is then opened as a component/BFS seed, independent of its TTL.
    blocked = (owners >= 0) & ~((owners == 0) & (ttl <= 1))
    if _inside(head, width, height):
        blocked[head[1], head[0]] = False
    labels, sizes = _label_components(blocked)
    head_label = int(labels[head[1], head[0]]) if _inside(head, width, height) else -1
    head_region = int(sizes[head_label]) if head_label >= 0 else 0
    largest = int(sizes.max()) if sizes.size else 0

    foods = _food_cells(sim, e)
    nearest = (
        min(foods, key=lambda cell: (abs(cell[0] - head[0]) + abs(cell[1] - head[1]), cell))
        if foods
        else None
    )
    direction = int(sim.direction[e, 0])
    dx, dy = (int(v) for v in CARDINAL[(direction + action % 3 - 1) % 4])
    next_cell = (head[0] + dx, head[1] + dy)
    toward_food = bool(
        nearest is not None
        and _inside(next_cell, width, height)
        and abs(next_cell[0] - nearest[0]) + abs(next_cell[1] - nearest[1])
        < abs(head[0] - nearest[0]) + abs(head[1] - nearest[1])
    )
    if nearest is None:
        food_in_smaller: Optional[bool] = None
    else:
        food_label = int(labels[nearest[1], nearest[0]]) if _inside(nearest, width, height) else -1
        food_in_smaller = bool(food_label >= 0 and int(sizes[food_label]) < largest)

    opponent_heads = [
        tuple(map(int, geometry["heads"][slot]))
        for slot in range(1, len(geometry["heads"]))
        if bool(geometry["alive"][slot])
    ]
    own_cells = np.argwhere(owners == 0)
    opponent_cells = np.argwhere(owners > 0)
    tail_bfs = _bfs_distance(head, tail, blocked)
    result: Dict[str, object] = {
        "head_region": head_region,
        "largest_region": largest,
        "head_in_largest": bool(head_label >= 0 and head_region == largest),
        "tail_bfs": tail_bfs,
        "tail_reachable": tail_bfs is not None,
        "tail_manhattan": abs(head[0] - tail[0]) + abs(head[1] - tail[1]),
        "own_cells_r15": int(
            np.count_nonzero(
                np.maximum(abs(own_cells[:, 1] - head[0]), abs(own_cells[:, 0] - head[1])) <= 15
            )
        ),
        "own_cells_r30": int(
            np.count_nonzero(
                np.maximum(abs(own_cells[:, 1] - head[0]), abs(own_cells[:, 0] - head[1])) <= 30
            )
        ),
        "opponent_cells_r15": int(
            np.count_nonzero(
                np.maximum(abs(opponent_cells[:, 1] - head[0]), abs(opponent_cells[:, 0] - head[1]))
                <= 15
            )
        ),
        "opponent_cells_r30": int(
            np.count_nonzero(
                np.maximum(abs(opponent_cells[:, 1] - head[0]), abs(opponent_cells[:, 0] - head[1]))
                <= 30
            )
        ),
        "toward_food": toward_food,
        "food_in_smaller_component": food_in_smaller,
    }
    result.update(
        _perimeter(head_label, labels, owners, head)
        if head_label >= 0
        else _perimeter(-2, labels, owners, head)
    )
    result.update(_race_counts(head, opponent_heads, owners, ttl))
    return result
