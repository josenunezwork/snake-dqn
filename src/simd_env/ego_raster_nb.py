"""Compiled (numba) backend of the ``ego2s-draft`` featurizer.

Bitwise equal to the NumPy reference in :mod:`src.simd_env.ego_raster` (pinned by
``tests/test_ego_raster.py`` on played, big-body, paused and random worlds). Same
pixel maps (the reference's own offset tables are passed in), same integer
arithmetic, and the same float64 expression for the enemy-head size ratio.

The reach flood is computed per agent as a bucket-queue shortest path with the
"wait for the tail" relaxation ``arrival(m) = max(arrival(c) + 1, ttl(m))``, which
equals the reference's synchronous flood (see the :mod:`~src.simd_env.ego_raster`
docstring) at O(window) per agent instead of O(K * window).

numba is a pinned dependency of the redesign branch (``requirements.txt``). The
kernels run single-threaded (no ``parallel=True``) so the one-thread compute rules
hold; ``cache=True`` keeps the JIT cost to the first call per environment.
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

import numba
import numpy as np
from numba import njit

from src.simd_env.ego_raster import (
    GLOBAL_CHANNELS,
    LOCAL_CHANNELS,
    EgoGridView,
    EgoRasterConfig,
    _offset_table,
)
from src.simd_env.grid_sim import FOOD_AMBIENT, FOOD_CORPSE, FOOD_OUTSIDE

__all__ = ["build_planes_numba", "food_cells_words", "word_views", "NUMBA_VERSION"]

NUMBA_VERSION = numba.__version__
_NL = len(LOCAL_CHANNELS)
_NG = len(GLOBAL_CHANNELS)


@njit(cache=True, nogil=True)
def _reach_one(
    ttl: np.ndarray,
    wall: np.ndarray,
    size: int,
    head_row: int,
    head_col: int,
    K: int,
    arr: np.ndarray,
    bhead: np.ndarray,
    nnext: np.ndarray,
    ncell: np.ndarray,
    out: np.ndarray,
) -> None:
    """Bucket-queue flood for one agent; writes arrival 1..K (else 0) into ``out``."""
    n = size * size
    inf = K + 1
    for p in range(n):
        arr[p] = inf
    for t in range(K + 2):
        bhead[t] = -1
    head = head_row * size + head_col
    arr[head] = 0
    used = 0
    ncell[used] = head
    nnext[used] = bhead[0]
    bhead[0] = used
    used += 1
    for t in range(K + 1):
        node = bhead[t]
        while node != -1:
            c = ncell[node]
            node = nnext[node]
            if arr[c] != t:
                continue  # stale entry (relaxed later to a smaller time)
            r = c // size
            q = c - r * size
            for d in range(4):
                if d == 0:
                    if r == 0:
                        continue
                    m = c - size
                elif d == 1:
                    if r == size - 1:
                        continue
                    m = c + size
                elif d == 2:
                    if q == 0:
                        continue
                    m = c - 1
                else:
                    if q == size - 1:
                        continue
                    m = c + 1
                if wall[m]:
                    continue
                cand = t + 1
                tm = ttl[m]
                if tm > cand:
                    cand = tm
                if cand <= K and cand < arr[m]:
                    arr[m] = cand
                    ncell[used] = m
                    nnext[used] = bhead[cand]
                    bhead[cand] = used
                    used += 1
    for p in range(n):
        a = arr[p]
        out[p] = a if (a >= 1 and a <= K) else 0


@njit(cache=True, nogil=True)
def _local_kernel(
    owner: np.ndarray,  # (E, Hp*Wp) int16
    slot: np.ndarray,  # (E, Hp*Wp) int32
    food: np.ndarray,  # (E, Hp*Wp) int8
    head_ptr: np.ndarray,
    length: np.ndarray,
    alive: np.ndarray,
    direction: np.ndarray,
    heads: np.ndarray,  # (E, S, 2)
    rows: np.ndarray,  # (N, 2)
    table: np.ndarray,  # (4, n) flat padded offsets
    size: int,
    head_row: int,
    head_col: int,
    pad: int,
    stride: int,
    cap: int,
    K: int,
    out: np.ndarray,  # (N, 6, n) uint8, zeroed
) -> None:
    n = size * size
    ttlg = np.zeros(n, dtype=np.int64)
    wall = np.zeros(n, dtype=np.bool_)
    arr = np.empty(n, dtype=np.int64)
    bhead = np.empty(K + 2, dtype=np.int64)
    nnext = np.empty(4 * n + 1, dtype=np.int64)
    ncell = np.empty(4 * n + 1, dtype=np.int64)
    reach = np.empty(n, dtype=np.int64)
    for i in range(rows.shape[0]):
        e = rows[i, 0]
        s = rows[i, 1]
        if not alive[e, s]:
            continue
        base = (heads[e, s, 1] + pad) * stride + heads[e, s, 0] + pad
        h = direction[e, s]
        own_len = length[e, s]
        if own_len < 1:
            own_len = 1
        own_len_f = np.float64(own_len)
        for p in range(n):
            idx = base + table[h, p]
            o = np.int64(owner[e, idx])
            f = food[e, idx]
            ttl = 0
            if o >= 0:
                k = (head_ptr[e, o] - np.int64(slot[e, idx])) % cap
                olen = length[e, o]
                ttl = olen - k
                if ttl < 0:
                    ttl = 0
                elif ttl > 255:
                    ttl = 255
                if o == s:
                    out[i, 0, p] = ttl
                else:
                    out[i, 1, p] = ttl
                    if k == 0:
                        ratio = np.float64(olen) / own_len_f
                        if ratio < 0.0:
                            ratio = 0.0
                        elif ratio > 2.0:
                            ratio = 2.0
                        ratio = ratio / 2.0
                        out[i, 2, p] = np.uint8(np.uint8(ratio * 254.0) + np.uint8(1))
            ttlg[p] = ttl
            if f == FOOD_CORPSE:
                out[i, 3, p] = 255
            elif f == FOOD_AMBIENT:
                out[i, 3, p] = 128
            is_wall = f == FOOD_OUTSIDE
            wall[p] = is_wall
            if is_wall:
                out[i, 4, p] = 255
        if K > 0:
            _reach_one(ttlg, wall, size, head_row, head_col, K, arr, bhead, nnext, ncell, reach)
            for p in range(n):
                out[i, 5, p] = reach[p]


@njit(cache=True, nogil=True)
def _coarse_kernel(
    owner: np.ndarray,  # (E, Hp, Wp) int16
    food: np.ndarray,  # (E, Hp, Wp) int8
    env_needed: np.ndarray,  # (E,) bool
    pad: int,
    H: int,
    W: int,
    cell: int,
    counts: np.ndarray,  # (E, S, hc, wc) int64, zeroed
    total: np.ndarray,  # (E, hc, wc) int64, zeroed
    fcounts: np.ndarray,  # (E, hc, wc) int64, zeroed
) -> None:
    for e in range(owner.shape[0]):
        if not env_needed[e]:
            continue
        for r in range(H):
            br = r // cell
            for c in range(W):
                bc = c // cell
                o = owner[e, r + pad, c + pad]
                if o >= 0:
                    counts[e, o, br, bc] += 1
                    total[e, br, bc] += 1
                if food[e, r + pad, c + pad] > 0:
                    fcounts[e, br, bc] += 1


_FULL16 = np.uint64(0xFFFFFFFFFFFFFFFF)  # four empty owner cells (-1 = 0xFFFF each)
_ZERO = np.uint64(0)


@njit(cache=True, nogil=True)
def _coarse_words_kernel(
    owner_w: np.ndarray,  # (E, Hp, Wp // 4) uint64 view of the int16 owner grid
    food_w: np.ndarray,  # (E, Hp, Wp // 8) uint64 view of the int8 food grid
    env_needed: np.ndarray,
    pad: int,
    H: int,
    W: int,
    cell: int,
    counts: np.ndarray,
    total: np.ndarray,
    fcounts: np.ndarray,
) -> None:
    """:func:`_coarse_kernel` reading 4 owner / 8 food cells per 64-bit word.

    Little-endian only (checked by the caller): cell ``4w + j`` is bits ``16j..16j+15``
    of owner word ``w``. Words of empty cells (all ``-1`` owner, all ``0`` food) are
    skipped whole; partial words at the interior edges check the column bounds.
    """
    ow0 = pad // 4
    ow1 = (pad + W - 1) // 4 + 1
    fw0 = pad // 8
    fw1 = (pad + W - 1) // 8 + 1
    m16 = np.uint64(0xFFFF)
    m8 = np.uint64(0xFF)
    for e in range(owner_w.shape[0]):
        if not env_needed[e]:
            continue
        for r in range(H):
            br = r // cell
            rr = r + pad
            for w in range(ow0, ow1):
                word = owner_w[e, rr, w]
                if word == _FULL16:
                    continue
                for j in range(4):
                    v = np.int64((word >> np.uint64(16 * j)) & m16)
                    if v == 0xFFFF:
                        continue
                    col = 4 * w + j - pad
                    if col < 0 or col >= W:
                        continue
                    bc = col // cell
                    counts[e, v, br, bc] += 1
                    total[e, br, bc] += 1
            for w in range(fw0, fw1):
                word = food_w[e, rr, w]
                if word == _ZERO:
                    continue
                for j in range(8):
                    b = np.int64((word >> np.uint64(8 * j)) & m8)
                    if b == 0 or b >= 0x80:  # empty, or FOOD_OUTSIDE (-1)
                        continue
                    col = 8 * w + j - pad
                    if col < 0 or col >= W:
                        continue
                    fcounts[e, br, col // cell] += 1


@njit(cache=True, nogil=True)
def food_cells_words(
    food_w: np.ndarray, e: int, pad: int, H: int, W: int, xs: np.ndarray, ys: np.ndarray
) -> int:
    """Pellet cells of env ``e`` (row-major order) into ``xs``/``ys``; returns the count."""
    fw0 = pad // 8
    fw1 = (pad + W - 1) // 8 + 1
    m8 = np.uint64(0xFF)
    count = 0
    for r in range(H):
        rr = r + pad
        for w in range(fw0, fw1):
            word = food_w[e, rr, w]
            if word == _ZERO:
                continue
            for j in range(8):
                b = np.int64((word >> np.uint64(8 * j)) & m8)
                if b == 0 or b >= 0x80:
                    continue
                col = 8 * w + j - pad
                if col < 0 or col >= W:
                    continue
                xs[count] = col
                ys[count] = r
                count += 1
    return count


def word_views(owner_pad: np.ndarray, food_pad: np.ndarray):
    """``(owner_w, food_w)`` uint64 views when the row layout allows it, else ``None``."""
    if sys.byteorder != "little":
        return None
    if (
        owner_pad.dtype != np.int16
        or food_pad.dtype != np.int8
        or not owner_pad.flags.c_contiguous
        or not food_pad.flags.c_contiguous
        or owner_pad.shape[-1] % 8
    ):
        return None
    return owner_pad.view(np.uint64), food_pad.view(np.uint64)


@njit(cache=True, nogil=True)
def _global_kernel(
    counts: np.ndarray,
    total: np.ndarray,
    fcounts: np.ndarray,
    alive: np.ndarray,
    direction: np.ndarray,
    heads: np.ndarray,
    rows: np.ndarray,
    drow: np.ndarray,  # (4, n) block row offsets
    dcol: np.ndarray,  # (4, n) block col offsets
    cell: int,
    out: np.ndarray,  # (N, 4, n) uint8, zeroed
) -> None:
    hc = counts.shape[2]
    wc = counts.shape[3]
    n = drow.shape[1]
    for i in range(rows.shape[0]):
        e = rows[i, 0]
        s = rows[i, 1]
        if not alive[e, s]:
            continue
        hr = heads[e, s, 1] // cell
        hcol = heads[e, s, 0] // cell
        h = direction[e, s]
        for p in range(n):
            br = hr + drow[h, p]
            bc = hcol + dcol[h, p]
            if br < 0 or br >= hc or bc < 0 or bc >= wc:
                out[i, 3, p] = 255
                continue
            own = counts[e, s, br, bc]
            if own > 255:
                own = 255
            allc = total[e, br, bc]
            sub = own if own < allc else allc
            enemy = allc - sub
            if enemy > 255:
                enemy = 255
            fc = fcounts[e, br, bc]
            if fc > 255:
                fc = 255
            out[i, 0, p] = own
            out[i, 1, p] = enemy
            out[i, 2, p] = fc


_RC_CACHE: dict = {}


def _offset_rc(size: int, head_row: int, head_col: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-heading (row, col) pixel offsets, decoded from the reference's flat table."""
    key = (size, head_row, head_col)
    if key not in _RC_CACHE:
        stride = 4 * size + 1  # wide enough that (drow, dcol) decode uniquely
        flat = _offset_table(size, head_row, head_col, stride)
        drow = np.floor_divide(flat + 2 * size, stride)
        dcol = flat - drow * stride
        drow = np.ascontiguousarray(drow, dtype=np.int64)
        dcol = np.ascontiguousarray(dcol, dtype=np.int64)
        if not np.array_equal(drow * stride + dcol, flat) or np.abs(dcol).max() > 2 * size:
            raise AssertionError("offset table decode failed")
        _RC_CACHE[key] = (drow, dcol)
    return _RC_CACHE[key]


def _i64(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def build_planes_numba(
    view: EgoGridView, cfg: EgoRasterConfig, rows: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """``(local (N,6,s,s), global (N,4,g,g))`` uint8 for ``rows`` (all ``E*S`` if None)."""
    E, S = view.E, view.S
    if rows is None:
        rows = np.stack(np.meshgrid(np.arange(E), np.arange(S), indexing="ij"), -1).reshape(-1, 2)
    rows = _i64(rows)
    N = rows.shape[0]
    size = cfg.local_size
    n = size * size
    owner_p = np.ascontiguousarray(view.owner_pad)
    slot_p = np.ascontiguousarray(view.slot_pad)
    food_p = np.ascontiguousarray(view.food_pad)
    stride = owner_p.shape[2]
    table = _offset_table(size, cfg.local_head_row, cfg.local_head_col, stride)
    head_ptr = _i64(view.head_ptr)
    length = _i64(view.length)
    alive = np.ascontiguousarray(view.alive, dtype=np.bool_)
    direction = _i64(view.direction)
    heads = _i64(view.heads)
    local = np.zeros((N, _NL, n), dtype=np.uint8)
    _local_kernel(
        owner_p.reshape(E, -1),
        slot_p.reshape(E, -1),
        food_p.reshape(E, -1),
        head_ptr,
        length,
        alive,
        direction,
        heads,
        rows,
        table,
        size,
        cfg.local_head_row,
        cfg.local_head_col,
        int(view.pad),
        stride,
        int(view.cap),
        int(cfg.reach_steps),
        local,
    )
    gsize, cell = cfg.global_size, cfg.global_cell
    hc = -(-view.grid_h // cell)
    wc = -(-view.grid_w // cell)
    if (
        view.coarse_cell == cell
        and view.coarse_snake is not None
        and view.coarse_food is not None
        and view.coarse_snake.shape == (E, S, hc, wc)
    ):
        # Counts maintained by the sim (pinned equal to a recount by its invariants).
        counts = view.coarse_snake
        fcounts = view.coarse_food
        total = counts.sum(axis=1, dtype=np.int64)
    else:
        counts = np.zeros((E, S, hc, wc), dtype=np.int64)
        total = np.zeros((E, hc, wc), dtype=np.int64)
        fcounts = np.zeros((E, hc, wc), dtype=np.int64)
        needed = np.zeros(E, dtype=np.bool_)
        if N:
            # Not ``needed[e] = alive[e, s]``: with repeated envs the last row would win.
            needed[rows[alive[rows[:, 0], rows[:, 1]], 0]] = True
        words = word_views(owner_p, food_p)
        if words is not None:
            _coarse_words_kernel(
                words[0],
                words[1],
                needed,
                int(view.pad),
                int(view.grid_h),
                int(view.grid_w),
                cell,
                counts,
                total,
                fcounts,
            )
        else:
            _coarse_kernel(
                owner_p,
                food_p,
                needed,
                int(view.pad),
                int(view.grid_h),
                int(view.grid_w),
                cell,
                counts,
                total,
                fcounts,
            )
    drow, dcol = _offset_rc(gsize, gsize // 2, gsize // 2)
    glob = np.zeros((N, _NG, gsize * gsize), dtype=np.uint8)
    _global_kernel(counts, total, fcounts, alive, direction, heads, rows, drow, dcol, cell, glob)
    return local.reshape(N, _NL, size, size), glob.reshape(N, _NG, gsize, gsize)


@njit(cache=True, nogil=True)
def _region_kernel(
    owner_pad: np.ndarray,  # (E, Hp, Wp) int16
    slot_pad: np.ndarray,  # (E, Hp, Wp) int32
    pad: int,
    H: int,
    W: int,
    cap: int,
    head_ptr: np.ndarray,
    length: np.ndarray,
    alive: np.ndarray,
    direction: np.ndarray,
    heads: np.ndarray,
    rows: np.ndarray,
    dxs: np.ndarray,
    dys: np.ndarray,
    out: np.ndarray,  # (N, 3) int64, zeroed
) -> None:
    """``ego_raster_b.region_sizes`` (4-connected free components, TTL > 1 blocks)."""
    n = H * W
    labels = np.empty(n, dtype=np.int64)
    blocked = np.empty(n, dtype=np.bool_)
    sizes = np.empty(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    order = np.argsort(rows[:, 0], kind="mergesort")
    current = -1
    for jj in range(rows.shape[0]):
        i = order[jj]
        e = rows[i, 0]
        s = rows[i, 1]
        if not alive[e, s]:
            continue
        if e != current:
            current = e
            for y in range(H):
                for x in range(W):
                    o = np.int64(owner_pad[e, y + pad, x + pad])
                    b = False
                    if o >= 0:
                        k = (head_ptr[e, o] - np.int64(slot_pad[e, y + pad, x + pad])) % cap
                        b = length[e, o] - k > 1
                    blocked[y * W + x] = b
                    labels[y * W + x] = -1
            nlab = 0
            for start in range(n):
                if blocked[start] or labels[start] >= 0:
                    continue
                labels[start] = nlab
                head = 0
                tail = 0
                queue[tail] = start
                tail += 1
                while head < tail:
                    c = queue[head]
                    head += 1
                    y = c // W
                    x = c - y * W
                    for d in range(4):
                        if d == 0:
                            if y == 0:
                                continue
                            m = c - W
                        elif d == 1:
                            if y == H - 1:
                                continue
                            m = c + W
                        elif d == 2:
                            if x == 0:
                                continue
                            m = c - 1
                        else:
                            if x == W - 1:
                                continue
                            m = c + 1
                        if not blocked[m] and labels[m] < 0:
                            labels[m] = nlab
                            queue[tail] = m
                            tail += 1
                sizes[nlab] = tail
                nlab += 1
        hx = heads[e, s, 0]
        hy = heads[e, s, 1]
        for rel in range(3):
            d = (direction[e, s] + rel - 1) % 4
            x = hx + dxs[d]
            y = hy + dys[d]
            if x >= 0 and x < W and y >= 0 and y < H:
                lab = labels[y * W + x]
                if lab >= 0:
                    out[i, rel] = sizes[lab]


def region_sizes_numba(view: EgoGridView, rows: np.ndarray) -> np.ndarray:
    """Compiled :func:`src.simd_env.ego_raster_b.region_sizes` (bitwise equal)."""
    rows = _i64(rows)
    out = np.zeros((len(rows), 3), dtype=np.int64)
    if not len(rows):
        return out
    _region_kernel(
        np.ascontiguousarray(view.owner_pad),
        np.ascontiguousarray(view.slot_pad),
        int(view.pad),
        int(view.grid_h),
        int(view.grid_w),
        int(view.cap),
        _i64(view.head_ptr),
        _i64(view.length),
        np.ascontiguousarray(view.alive, dtype=np.bool_),
        _i64(view.direction),
        _i64(view.heads),
        rows,
        np.array([0, 1, 0, -1], dtype=np.int64),
        np.array([-1, 0, 1, 0], dtype=np.int64),
        out,
    )
    return out
