"""Action-conditioned local geometry from Sandbox B's stored B8 observations.

The admitted F0 table preserves B8's heading-normalized local raster, rather
than a complete simulator snapshot.  These features deliberately use that
stored evidence only: they are deterministic, label-free, and never replay a
world.  They are therefore local counterfactual geometry, not a replacement
for GridBatchSim's exact closed-loop action transition.

Each action receives seven crop-derived values followed by B8's five stored v8
values in ``FEATURE_NAMES`` order.  Actions 3--5
use two cells only when the stored B8 boost-available scalar is true; otherwise
they intentionally share the normal action's geometry, matching v8's
unavailable-boost convention.  The time-aware own-TTL flood is a local
approximation, not a full-state lower bound or exact simulator transition.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numba import njit

__all__ = [
    "FEATURE_NAMES",
    "N_ACTION_FEATURES",
    "action_geometry",
    "geometry_features",
    "flatten_action_geometry",
]

FEATURE_NAMES = (
    "immediate_legal",
    "reachable_over_length",
    "reachable_local_fraction",
    "tail_reachable",
    "landing_open_degree",
    "landing_local_occupancy",
    "boost_effective",
    "v8_area_g",
    "v8_head_risky",
    "v8_landing_ok",
    "v8_v5_eligible",
    "v8_spacious",
)
N_ACTION_FEATURES = len(FEATURE_NAMES)
_CROP_FEATURES = 7

# B8 rasters are heading-normalized with the head at this pixel.  A relative
# left/straight/right action thus moves west/north/east in the raster frame.
_HEAD_ROW = 23
_HEAD_COL = 15
_DIRECTIONS = ((0, -1), (-1, 0), (0, 1))
_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@njit(cache=True, nogil=True)
def _passable_nb(own_ttl, enemy_ttl, wall, row, col, time):
    return (
        0 <= row < 31
        and 0 <= col < 31
        and wall[row, col] == 0
        and enemy_ttl[row, col] == 0
        and (own_ttl[row, col] == 0 or own_ttl[row, col] <= time)
    )


@njit(cache=True, nogil=True)
def _action_geometry_nb(local, scalars):
    """Compiled implementation; keep its control flow parallel to the public contract."""
    out = np.zeros((local.shape[0], 6, _CROP_FEATURES), dtype=np.float32)
    queue_r = np.empty(31 * 31, dtype=np.int64)
    queue_c = np.empty(31 * 31, dtype=np.int64)
    queue_t = np.empty(31 * 31, dtype=np.int64)
    for index in range(local.shape[0]):
        own_ttl, enemy_ttl, wall = local[index, 0], local[index, 1], local[index, 4]
        length = max(1.0, float(scalars[index, 1]) * 1000.0)
        boost_available = scalars[index, 3] >= 0.5
        for action in range(6):
            relative = action % 3
            dr = 0 if relative != 1 else -1
            dc = -1 if relative == 0 else (1 if relative == 2 else 0)
            boosted = action >= 3 and boost_available
            steps = 2 if boosted else 1
            row, col = _HEAD_ROW, _HEAD_COL
            legal = True
            for time in range(1, steps + 1):
                row += dr
                col += dc
                if not _passable_nb(own_ttl, enemy_ttl, wall, row, col, time):
                    legal = False
                    break
            if not legal:
                continue
            seen = np.zeros((31, 31), dtype=np.uint8)
            seen[row, col] = 1
            queue_r[0], queue_c[0], queue_t[0] = row, col, steps
            head, tail, area = 0, 1, 0
            reaches_tail, degree = own_ttl[row, col] == 1, 0
            while head < tail:
                current_r, current_c, current_t = queue_r[head], queue_c[head], queue_t[head]
                head += 1
                area += 1
                for direction in range(4):
                    nr = current_r + (-1 if direction == 0 else (1 if direction == 1 else 0))
                    nc = current_c + (-1 if direction == 2 else (1 if direction == 3 else 0))
                    if _passable_nb(own_ttl, enemy_ttl, wall, nr, nc, current_t + 1):
                        if current_r == row and current_c == col:
                            degree += 1
                        if seen[nr, nc] == 0:
                            seen[nr, nc] = 1
                            reaches_tail = reaches_tail or own_ttl[nr, nc] == 1
                            queue_r[tail], queue_c[tail], queue_t[tail] = nr, nc, current_t + 1
                            tail += 1
            out[index, action, 0] = 1.0
            out[index, action, 1] = min(area / length, 1.0)
            out[index, action, 2] = area / float(31 * 31)
            out[index, action, 3] = 1.0 if reaches_tail else 0.0
            out[index, action, 4] = degree / 4.0
            out[index, action, 5] = (4 - degree) / 4.0
            out[index, action, 6] = 1.0 if boosted else 0.0
    return out


def _validate(local: np.ndarray, scalars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(local)
    scalars = np.asarray(scalars)
    if local.ndim != 4 or local.shape[1:] != (7, 31, 31) or local.dtype != np.uint8:
        raise ValueError("local must be uint8 [N, 7, 31, 31] B8 observations")
    if scalars.ndim != 2 or scalars.shape[0] != len(local) or scalars.shape[1] < 60:
        raise ValueError("scalars must be [N, >=60] and aligned with local")
    if not np.isfinite(scalars).all():
        raise ValueError("scalars must be finite")
    return local, scalars


def _in_bounds(row: int, col: int) -> bool:
    return 0 <= row < 31 and 0 <= col < 31


def _passable(
    own_ttl: np.ndarray, enemy_ttl: np.ndarray, wall: np.ndarray, row: int, col: int, time: int
) -> bool:
    """Whether a local cell is usable at a future integer frame.

    Enemy occupancy is held fixed because the stored raster does not preserve
    every opponent's future action.  Own cells use B8's TTL semantics: a cell
    becomes passable when its remaining TTL is no greater than arrival time.
    """
    return (
        _in_bounds(row, col)
        and wall[row, col] == 0
        and enemy_ttl[row, col] == 0
        and (own_ttl[row, col] == 0 or own_ttl[row, col] <= time)
    )


def _flood(
    own_ttl: np.ndarray,
    enemy_ttl: np.ndarray,
    wall: np.ndarray,
    start: tuple[int, int],
    start_time: int,
) -> tuple[int, bool, int]:
    """Time-aware local BFS returning (area, reaches_tail, open_degree)."""
    if not _passable(own_ttl, enemy_ttl, wall, start[0], start[1], start_time):
        return 0, False, 0
    seen = {start}
    queue: deque[tuple[int, int, int]] = deque([(start[0], start[1], start_time)])
    tail = bool(own_ttl[start] == 1)
    degree = 0
    while queue:
        row, col, time = queue.popleft()
        for dr, dc in _NEIGHBOURS:
            nr, nc = row + dr, col + dc
            if _passable(own_ttl, enemy_ttl, wall, nr, nc, time + 1):
                if (row, col) == start:
                    degree += 1
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    tail |= bool(own_ttl[nr, nc] == 1)
                    queue.append((nr, nc, time + 1))
    return len(seen), tail, degree


def action_geometry(local: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    """Return label-free ``float32 [N, 6, 12]`` geometry for B8 actions.

    Scalar index 1 is B8's unbounded ``length_per_1000`` and index 3 is its
    boost-available flag.  The inputs are never mutated.
    """
    local, scalars = _validate(local, scalars)
    # bv8 scalars 30..59 are already action-major per-action v8 diagnostics,
    # generated from the original full state before the admitted F0 table was
    # sealed.  Preserve them directly instead of trying to reconstruct them
    # from the cropped raster.
    return np.concatenate(
        (_action_geometry_nb(local, scalars), scalars[:, 30:60].reshape(-1, 6, 5)), axis=2
    )


def geometry_features(table: dict[str, np.ndarray], progress=None) -> np.ndarray:
    """Return ``[N, 6, 12]`` geometry from an admitted Sandbox-B table.

    This is the B-3 runtime entry point.  It accepts only the admitted local
    B8 raster and scalars; outcome columns are neither accessed nor needed.
    ``progress`` is called once per completed source row so a long CPU fallback
    can emit outcome-blind telemetry.

    ``reachable_*`` is a 31x31-window local approximation.  ``tail_reachable``
    means a visible own TTL-one cell was reached by this approximation; it does
    not claim that the physical tail is in the crop or reachable in the full
    simulator state.
    """
    if not isinstance(table, dict):
        raise TypeError("table must be a dictionary containing local and scalars")
    local, scalars = _validate(table.get("local"), table.get("scalars"))
    if progress is None:
        return action_geometry(local, scalars)
    out = np.empty((len(local), 6, N_ACTION_FEATURES), dtype=np.float32)
    for start in range(0, len(local), 1024):
        stop = min(start + 1024, len(local))
        out[start:stop] = action_geometry(local[start:stop], scalars[start:stop])
        for _ in range(start, stop):
            progress()
    return out


def flatten_action_geometry(local: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    """Return action-major ``float32 [N, 72]`` geometry for concatenation with phi."""
    return action_geometry(local, scalars).reshape(len(local), 6 * N_ACTION_FEATURES)
