"""T33-local parallel writers for the frozen ego-raster numba kernels.

The upstream raster kernels are the geometry oracle.  These wrappers only
partition independent destination rows with ``prange`` and pass a one-row view
to the existing row kernel.  Each invocation owns its reach-flood workspaces,
which avoids the shared-scratch bug that invalidated earlier threaded kernels.
"""

from __future__ import annotations

import threading

import numba
import numpy as np
from numba import njit, prange

from src.simd_env.ego_raster_nb import _local_kernel

_LOCAL_SCRATCH = threading.local()


def _local_workspace(size: int, reach_steps: int, parts: int) -> tuple:
    key = (size, reach_steps, parts)
    cache = getattr(_LOCAL_SCRATCH, "cache", None)
    if cache is None:
        cache = _LOCAL_SCRATCH.cache = {}
    if key not in cache:
        n = size * size
        cache[key] = tuple(
            np.empty((parts, count), dtype=dtype)
            for count, dtype in (
                (n, np.int64),
                (n, np.bool_),
                (n, np.int64),
                (reach_steps + 2, np.int64),
                (4 * n + 1, np.int64),
                (4 * n + 1, np.int64),
                (n, np.int64),
            )
        )
    return cache[key]


def write_local_prange(
    owner: np.ndarray,
    slot: np.ndarray,
    food: np.ndarray,
    head_ptr: np.ndarray,
    length: np.ndarray,
    alive: np.ndarray,
    direction: np.ndarray,
    heads: np.ndarray,
    rows: np.ndarray,
    table: np.ndarray,
    size: int,
    head_row: int,
    head_col: int,
    pad: int,
    stride: int,
    cap: int,
    reach_steps: int,
    out: np.ndarray,
) -> None:
    """Write locals with persistent, disjoint per-partition flood scratch.

    Scratch belongs to this Python caller and is reused across builds. Every
    live row fully overwrites the TTL/wall inputs and reach resets its queues.
    """
    parts = min(len(rows), numba.get_num_threads())
    workspace = _local_workspace(size, reach_steps, max(1, parts))
    if parts <= 1:
        _local_kernel(
            owner,
            slot,
            food,
            head_ptr,
            length,
            alive,
            direction,
            heads,
            rows,
            table,
            size,
            head_row,
            head_col,
            pad,
            stride,
            cap,
            reach_steps,
            out,
            tuple(a[0] for a in workspace),
        )
        return
    _write_local_chunks(
        owner,
        slot,
        food,
        head_ptr,
        length,
        alive,
        direction,
        heads,
        rows,
        table,
        size,
        head_row,
        head_col,
        pad,
        stride,
        cap,
        reach_steps,
        out,
        parts,
        workspace,
    )


@njit(cache=True, nogil=True, parallel=True)
def _write_local_chunks(
    owner,
    slot,
    food,
    head_ptr,
    length,
    alive,
    direction,
    heads,
    rows,
    table,
    size,
    head_row,
    head_col,
    pad,
    stride,
    cap,
    reach_steps,
    out,
    parts,
    workspace,
):
    """Invoke the canonical local kernel over persistent private-scratch chunks."""
    total = rows.shape[0]
    for part in prange(parts):
        begin = part * total // parts
        end = (part + 1) * total // parts
        _local_kernel(
            owner,
            slot,
            food,
            head_ptr,
            length,
            alive,
            direction,
            heads,
            rows[begin:end],
            table,
            size,
            head_row,
            head_col,
            pad,
            stride,
            cap,
            reach_steps,
            out[begin:end],
            (
                workspace[0][part],
                workspace[1][part],
                workspace[2][part],
                workspace[3][part],
                workspace[4][part],
                workspace[5][part],
                workspace[6][part],
            ),
        )


@njit(cache=True, nogil=True, parallel=True)
def write_global_prange(
    counts: np.ndarray,
    food_counts: np.ndarray,
    alive: np.ndarray,
    direction: np.ndarray,
    heads: np.ndarray,
    rows: np.ndarray,
    drow: np.ndarray,
    dcol: np.ndarray,
    cell: int,
    out: np.ndarray,
) -> None:
    """Write four global planes without materializing an ``(E, Hc, Wc)`` total.

    The canonical global kernel accepts a pre-reduced per-env count grid.  T33
    only needs a handful of blocks per selected observer, so reducing all
    environments first is both an allocation and a bandwidth tax.  Summing the
    (small, fixed) snake axis at the addressed block is exact and keeps each
    destination row independent for ``prange``.
    """
    for index in prange(rows.shape[0]):
        env = rows[index, 0]
        observer = rows[index, 1]
        if not alive[env, observer]:
            continue
        hc = counts.shape[2]
        wc = counts.shape[3]
        hr = heads[env, observer, 1] // cell
        hcol = heads[env, observer, 0] // cell
        heading = direction[env, observer]
        for pixel in range(drow.shape[1]):
            br = hr + drow[heading, pixel]
            bc = hcol + dcol[heading, pixel]
            if br < 0 or br >= hc or bc < 0 or bc >= wc:
                out[index, 3, pixel] = 255
                continue
            own = counts[env, observer, br, bc]
            total = 0
            for snake in range(counts.shape[1]):
                total += counts[env, snake, br, bc]
            if own > 255:
                own = 255
            sub = own if own < total else total
            enemy = total - sub
            if enemy > 255:
                enemy = 255
            food = food_counts[env, br, bc]
            if food > 255:
                food = 255
            out[index, 0, pixel] = own
            out[index, 1, pixel] = enemy
            out[index, 2, pixel] = food


@njit(cache=True, nogil=True, parallel=True)
def write_enemy_next_prange(
    alive: np.ndarray,
    direction: np.ndarray,
    heads: np.ndarray,
    boosting: np.ndarray,
    rows: np.ndarray,
    grid_w: int,
    grid_h: int,
    head_row: int,
    head_col: int,
    out: np.ndarray,
) -> None:
    """Write Bv8's enemy-next channel directly into ``out``.

    This is the exact integer rotation and max-composition used by
    ``enemy_next_plane``.  Destination rows are independent, so no lane shares
    scratch or writes another lane's pixels.
    """
    for index in prange(rows.shape[0]):
        env = rows[index, 0]
        observer = rows[index, 1]
        if not alive[env, observer]:
            continue
        hx = heads[env, observer, 0]
        hy = heads[env, observer, 1]
        heading = direction[env, observer]
        if heading == 0:
            ax, ay = 0, -1
        elif heading == 1:
            ax, ay = 1, 0
        elif heading == 2:
            ax, ay = 0, 1
        else:
            ax, ay = -1, 0
        rx, ry = -ay, ax
        for enemy in range(alive.shape[1]):
            if enemy == observer or not alive[env, enemy]:
                continue
            enemy_heading = direction[env, enemy]
            if enemy_heading == 0:
                ex, ey = 0, -1
            elif enemy_heading == 1:
                ex, ey = 1, 0
            elif enemy_heading == 2:
                ex, ey = 0, 1
            else:
                ex, ey = -1, 0
            for step in range(1, 3):
                if boosting[env, enemy]:
                    value = 255
                elif step == 1:
                    value = 128
                else:
                    value = 0
                x = heads[env, enemy, 0] + ex * step
                y = heads[env, enemy, 1] + ey * step
                if value == 0 or x < 0 or x >= grid_w or y < 0 or y >= grid_h:
                    continue
                dx, dy = x - hx, y - hy
                row = head_row - (dx * ax + dy * ay)
                col = head_col + (dx * rx + dy * ry)
                if row >= 0 and row < 31 and col >= 0 and col < 31 and value > out[index, row, col]:
                    out[index, row, col] = value


NUMBA_VERSION = numba.__version__
