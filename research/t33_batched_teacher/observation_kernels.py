"""T33-local parallel writers for the frozen ego-raster numba kernels.

The upstream raster kernels are the geometry oracle.  These wrappers only
partition independent destination rows with ``prange`` and pass a one-row view
to the existing row kernel.  Each invocation owns its reach-flood workspaces,
which avoids the shared-scratch bug that invalidated earlier threaded kernels.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange

from src.simd_env.ego_raster_nb import _global_kernel, _local_kernel


@njit(cache=True, nogil=True, parallel=True)
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
    """Write six local planes into ``out``; every prange lane owns scratch."""
    for index in prange(rows.shape[0]):
        _local_kernel(
            owner,
            slot,
            food,
            head_ptr,
            length,
            alive,
            direction,
            heads,
            rows[index : index + 1],
            table,
            size,
            head_row,
            head_col,
            pad,
            stride,
            cap,
            reach_steps,
            out[index : index + 1],
        )


@njit(cache=True, nogil=True, parallel=True)
def write_global_prange(
    counts: np.ndarray,
    total: np.ndarray,
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
    """Write four global planes into ``out`` in independent destination rows."""
    for index in prange(rows.shape[0]):
        _global_kernel(
            counts,
            total,
            food_counts,
            alive,
            direction,
            heads,
            rows[index : index + 1],
            drow,
            dcol,
            cell,
            out[index : index + 1],
        )


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
