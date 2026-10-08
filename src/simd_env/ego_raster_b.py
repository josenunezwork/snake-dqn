"""``ego2s-b`` additions to the ego raster (redesign M2b; closes the M2 observation gap).

M2's student plateaued at ~0.80 agreement with its teacher. The teacher's 61-D input
carries enemy headings, a nearest-enemy approach trend, a kill-opportunity flag and
explicit nearest / second-nearest enemy geometry that ``ego2s-draft`` lacks, and the
student almost never boosted. ``ego2s-b`` = ``ego2s-draft`` plus:

Local channel 6, ``enemy_next`` (uint8, same ego crop):
    the cell each living enemy reaches next frame if it keeps its heading: 128, or 255
    (that cell AND the one after it) when the enemy boosted on its last move
    (``BatchSim.get_boosted_this_step`` / live ``Snake.is_boosting``). Cells outside the
    arena are skipped (the wall channel covers them). Values combine by ``max``.

Scalars 12..29 (float32), all in the observer's ego frame (``ahead``, ``right``):
    ====  =========================  =============================================
    12-13 nearest_enemy_ahead/right  head offset / 145 (0 when no enemy)
    14    nearest_enemy_size         ``clip(L_e / L, 0, 2) / 2`` (the teacher's [46])
    15-16 nearest_enemy_heading      enemy heading projected on ahead / right (-1/0/1)
    17    nearest_enemy_boosting     0 / 1
    18    nearest_enemy_approach     ``clip(d_next - d_now, -2, 2) / 2``: head distance
                                     after both heads advance (enemy 2 cells if boosting)
                                     minus now; negative = closing (the teacher's trend,
                                     made stateless)
    19    kill_opportunity           1 when the enemy's next cell is < 3 cells from my
                                     head (the teacher's [53])
    20-22 second_enemy_ahead/right/size
    23-25 region_frac_left/straight/right   ``min(R / L, 4) / 4``
    26-28 region_log_left/straight/right    ``log1p(R) / log1p(H * W)``
    29    enemies_alive_fraction     living enemies / (S - 1)
    ====  =========================  =============================================

    ``R`` = the size of the 4-connected free component that contains the action's next
    cell (0 if that cell is outside the arena or blocked), UNCAPPED (the teacher's
    free-space BFS stops at 160 cells), tail-aware: a cell is blocked when a living
    snake's segment there has TTL ``L - k > 1`` (a segment that vacates next frame counts
    as free). Nearest = smallest squared head distance, ties to the lower slot (the
    teacher's stable sort).

Every quantity is integer geometry from the :class:`EgoGridView` (plus ``boosting``), so
the live view and the sim view give identical bytes. Only the component labelling has two
implementations: a pure-Python BFS reference and a numba kernel (bitwise equal, tested).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from src.simd_env.batch_sim import CARDINAL

__all__ = ["B_LOCAL_EXTRA", "B_SCALAR_NAMES", "enemy_next_plane", "b_scalars", "region_sizes"]

B_LOCAL_EXTRA = ("enemy_next",)
B_SCALAR_NAMES = (
    "nearest_enemy_ahead",
    "nearest_enemy_right",
    "nearest_enemy_size",
    "nearest_enemy_heading_ahead",
    "nearest_enemy_heading_right",
    "nearest_enemy_boosting",
    "nearest_enemy_approach",
    "kill_opportunity",
    "second_enemy_ahead",
    "second_enemy_right",
    "second_enemy_size",
    "region_frac_left",
    "region_frac_straight",
    "region_frac_right",
    "region_log_left",
    "region_log_straight",
    "region_log_right",
    "enemies_alive_fraction",
)


def _boosting(view) -> np.ndarray:
    b = getattr(view, "boosting", None)
    if b is None:
        raise ValueError("ego2s-b needs EgoGridView.boosting")
    return np.asarray(b, dtype=bool) & np.asarray(view.alive, dtype=bool)


def _rows(view, rows: Optional[np.ndarray]) -> np.ndarray:
    if rows is not None:
        return rows
    return np.stack(np.meshgrid(np.arange(view.E), np.arange(view.S), indexing="ij"), -1).reshape(
        -1, 2
    )


def enemy_next_plane(view, cfg, rows: Optional[np.ndarray]) -> np.ndarray:
    """``(N, size, size)`` uint8 enemy next-cell plane for ``rows`` (all if None)."""
    rows = _rows(view, rows)
    N, S, size = len(rows), view.S, cfg.local_size
    out = np.zeros((N, size * size), dtype=np.uint8)
    if not N:
        return out.reshape(N, size, size)
    e, s = rows[:, 0], rows[:, 1]
    boosting = _boosting(view)
    head = view.heads[e, s]  # (N, 2)
    a = CARDINAL[view.direction[e, s]]  # (N, 2) ahead
    r = np.stack([-a[:, 1], a[:, 0]], axis=1)  # right
    for j in range(S):
        ok = view.alive[e, j] & (j != s) & view.alive[e, s]
        dj = CARDINAL[view.direction[e, j]]
        for step in (1, 2):
            cell = view.heads[e, j] + dj * step
            val = np.where(boosting[e, j], 255, 128 if step == 1 else 0).astype(np.uint8)
            inside = (
                (cell[:, 0] >= 0)
                & (cell[:, 0] < view.grid_w)
                & (cell[:, 1] >= 0)
                & (cell[:, 1] < view.grid_h)
            )
            delta = cell - head
            ahead = (delta * a).sum(axis=1)
            right = (delta * r).sum(axis=1)
            pr = cfg.local_head_row - ahead
            pc = cfg.local_head_col + right
            inwin = (pr >= 0) & (pr < size) & (pc >= 0) & (pc < size)
            m = ok & inside & inwin & (val > 0)
            idx = np.flatnonzero(m)
            if len(idx):
                flat = pr[idx] * size + pc[idx]
                out[idx, flat] = np.maximum(out[idx, flat], val[idx])
    return out.reshape(N, size, size)


def _blocked_grid(view, env: int) -> np.ndarray:
    """``(H, W)`` bool: cells holding a living segment with TTL > 1."""
    P, H, W = view.pad, view.grid_h, view.grid_w
    owner = view.owner_pad[env, P : P + H, P : P + W].astype(np.int64)
    slot = view.slot_pad[env, P : P + H, P : P + W].astype(np.int64)
    occ = owner >= 0
    o = np.where(occ, owner, 0)
    k = (view.head_ptr[env][o] - slot) % view.cap
    ttl = view.length[env][o] - k
    return occ & (ttl > 1)


def _components_python(blocked: np.ndarray):
    """Reference labelling: 4-connected free components (labels, sizes), plain BFS."""
    H, W = blocked.shape
    labels = np.full((H, W), -1, dtype=np.int64)
    sizes = []
    for r0 in range(H):
        for c0 in range(W):
            if blocked[r0, c0] or labels[r0, c0] >= 0:
                continue
            lab = len(sizes)
            labels[r0, c0] = lab
            q = deque([(r0, c0)])
            n = 0
            while q:
                r, c = q.popleft()
                n += 1
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and not blocked[rr, cc] and labels[rr, cc] < 0:
                        labels[rr, cc] = lab
                        q.append((rr, cc))
            sizes.append(n)
    return labels, np.asarray(sizes, dtype=np.int64)


def region_sizes(view, rows: Optional[np.ndarray], backend: str) -> np.ndarray:
    """``(N, 3)`` int64 free-component size at the left / straight / right next cell."""
    rows = _rows(view, rows)
    N = len(rows)
    out = np.zeros((N, 3), dtype=np.int64)
    if not N:
        return out
    if backend == "numba":
        from src.simd_env.ego_raster_nb import region_sizes_numba

        return region_sizes_numba(view, rows)
    for env in np.unique(rows[:, 0]):
        sel = np.flatnonzero((rows[:, 0] == env) & view.alive[env, rows[:, 1]])
        if not len(sel):
            continue
        labels, sizes = _components_python(_blocked_grid(view, int(env)))
        for i in sel:
            s = rows[i, 1]
            hx, hy = view.heads[env, s]
            for rel in range(3):
                d = CARDINAL[(view.direction[env, s] + rel - 1) % 4]
                x, y = hx + d[0], hy + d[1]
                if 0 <= x < view.grid_w and 0 <= y < view.grid_h and labels[y, x] >= 0:
                    out[i, rel] = sizes[labels[y, x]]
    return out


def b_scalars(view, rows: Optional[np.ndarray], backend: str) -> np.ndarray:
    """``(N, 18)`` float32 scalars 12..29 of ``ego2s-b`` (see the module table)."""
    rows = _rows(view, rows)
    N, S = len(rows), view.S
    out = np.zeros((N, len(B_SCALAR_NAMES)), dtype=np.float32)
    if not N:
        return out
    e, s = rows[:, 0], rows[:, 1]
    boosting = _boosting(view)
    alive_row = view.alive[e, s]
    head = view.heads[e, s].astype(np.int64)
    a = CARDINAL[view.direction[e, s]]
    r = np.stack([-a[:, 1], a[:, 0]], axis=1)
    L = np.maximum(view.length[e, s], 1).astype(np.float64)
    # Enemy table (N, S): squared head distance, inf for self / dead.
    eh = view.heads[e].astype(np.int64)  # (N, S, 2)
    delta = eh - head[:, None, :]
    d2 = (delta**2).sum(axis=2).astype(np.float64)
    valid = view.alive[e] & (np.arange(S)[None, :] != s[:, None])
    d2 = np.where(valid, d2, np.inf)
    order = np.argsort(d2, axis=1, kind="stable")
    n_enemies = valid.sum(axis=1)
    for rank, base in ((0, 0), (1, 8)):
        j = order[:, rank] if S > rank else np.zeros(N, dtype=np.int64)
        has = (n_enemies > rank) & alive_row
        dj = delta[np.arange(N), j]
        ahead = (dj * a).sum(axis=1)
        right = (dj * r).sum(axis=1)
        size = np.clip(view.length[e, j] / L, 0.0, 2.0) / 2.0
        if rank == 0:
            out[:, 0] = np.where(has, ahead / 145.0, 0.0)
            out[:, 1] = np.where(has, right / 145.0, 0.0)
            out[:, 2] = np.where(has, size, 0.0)
            hd = CARDINAL[view.direction[e, j]]
            out[:, 3] = np.where(has, (hd * a).sum(axis=1), 0.0)
            out[:, 4] = np.where(has, (hd * r).sum(axis=1), 0.0)
            bj = boosting[e, j]
            out[:, 5] = np.where(has, bj, 0.0)
            e_next = eh[np.arange(N), j] + hd * np.where(bj, 2, 1)[:, None]
            my_next = head + a
            d_now = np.sqrt(d2[np.arange(N), j])
            d_next = np.sqrt(((e_next - my_next) ** 2).sum(axis=1).astype(np.float64))
            approach = np.clip(np.where(has, d_next - d_now, 0.0), -2.0, 2.0) / 2.0
            out[:, 6] = approach
            k1 = eh[np.arange(N), j] + hd
            kd2 = ((k1 - head) ** 2).sum(axis=1)
            out[:, 7] = np.where(has & (kd2 < 9), 1.0, 0.0)
        else:
            out[:, 8] = np.where(has, ahead / 145.0, 0.0)
            out[:, 9] = np.where(has, right / 145.0, 0.0)
            out[:, 10] = np.where(has, size, 0.0)
    reg = region_sizes(view, rows, backend).astype(np.float64)
    out[:, 11:14] = np.minimum(reg / L[:, None], 4.0) / 4.0
    out[:, 14:17] = np.log1p(reg) / np.log1p(float(view.grid_w * view.grid_h))
    out[:, 17] = n_enemies / float(max(S - 1, 1))
    out[~alive_row] = 0.0
    return out
