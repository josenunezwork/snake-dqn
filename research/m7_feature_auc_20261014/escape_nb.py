"""Candidate long-horizon "escape" features for one hero row of a GridBatchSim (M7 study A).

Development diagnostic only (offline AUC study); not an observation spec. Everything is
integer geometry read from the sim's ring buffers (``bodies``, ``head_ptr``, ``seg_count``,
``length``, ``alive``, ``direction``), so a live twin would only need the same body lists.

Time convention: decision time 0 is the prepared world the policy sees. A segment at
offset ``k`` (0 = head) of a snake of logical length ``L`` vacates at time ``ttl = L - k``
(the ``own_ttl`` / ``region_sizes`` convention; growth or boosting would shift it), so a
cell is free at time ``t`` iff ``t >= ttl`` (empty cells: ``ttl = 0``). Other snakes'
bodies release the same way (optimistic: their heads also advance, not modelled).

Families (per relative action ``a`` in left / straight / right, from the next head cell
``c1`` entered at time 1; an action whose ``c1`` is outside the arena or holds a segment
with ``ttl > 1`` gets the "blocked" values):

A. Time-aware flood (waiting allowed): ``arrival(c1) = 1``,
   ``arrival(n) = max(arrival(c) + 1, ttl(n))`` over 4-neighbours, horizon ``2L``.
   Because the relaxation is monotone in ``arrival(c)``, a bucket queue settles every cell
   on its first relaxation (label-setting, ``O(cells + 2L)``). ``N(t)`` = cells with
   ``arrival <= t``. A head that moves ``t <= L`` times visits ``t`` distinct cells, each no
   earlier than its arrival, so ``N(t) >= t`` is necessary to survive ``t`` steps.
   * ``A_nq4/A_nq2/A_nq1``: ``N(T) / T`` for ``T = L/4, L/2, L`` (clipped at 8);
   * ``A_tstar``: ``t* / L`` with ``t*`` the first ``t in [1, L]`` with ``N(t) < t``
     (``(L + 1) / L`` when none: no counting bound within ``L``);
   * ``A_own_first``: first arrival at a released own-body cell / ``L`` (2 if none);
   * ``A_own_frac``: fraction of own-body cells with ``arrival <= L``;
   * ``A_arr_largest``: first arrival inside the largest static component / ``L`` (2 if none).
B. Static-region frontier vacate times. Static region = the 4-connected component of
   ``c1`` with every ``ttl > 1`` cell blocked (exactly ``ego2s-b``'s ``region_sizes``
   ``R``); frontier = own-body cells 4-adjacent to it:
   * ``B_front_own``: frontier own cells / ``L``;
   * ``B_front_le_half`` / ``B_front_le_R``: fraction of them with ``ttl <= R/2`` / ``<= R``
     (they vacate before the head could exhaust the region);
   * ``B_min_ttl``: smallest frontier ttl / ``L`` (2 if none);
   * ``B_slack``: ``(R - min frontier ttl) / L`` clipped to ``[-2, 8]`` (8 if none);
   * ``B_own_share``: own / (own + enemy) frontier cells (0.5 if none).
C. Global region structure: ``C_in_largest`` per action, and per row ``C_largest_over_L``
   (clip 16), ``C_largest_frac`` (of free cells), ``C_ncomp32`` (components >= 32 cells),
   ``C_free_frac`` (free cells / arena), ``C_second_over_largest``.
D. Body shape: ``D_own_dens8`` / ``D_own_dens16`` (own cells within Chebyshev 8 / 16 of the
   head over the window area), ``D_bbox_ratio`` (L / own-body bounding-box area),
   ``D_fill`` (all snake cells / arena).
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
from numba import njit

__all__ = [
    "ACTION_FEATURES",
    "STATE_FEATURES",
    "escape_features",
    "escape_features_timed",
]

ACTION_FEATURES = (
    "A_nq4",
    "A_nq2",
    "A_nq1",
    "A_tstar",
    "A_own_first",
    "A_own_frac",
    "A_arr_largest",
    "B_front_own",
    "B_front_le_half",
    "B_front_le_R",
    "B_min_ttl",
    "B_slack",
    "B_own_share",
    "C_in_largest",
)
STATE_FEATURES = (
    "C_largest_over_L",
    "C_largest_frac",
    "C_ncomp32",
    "C_free_frac",
    "C_second_over_largest",
    "D_own_dens8",
    "D_own_dens16",
    "D_bbox_ratio",
    "D_fill",
)
NA = len(ACTION_FEATURES)
NS = len(STATE_FEATURES)

# batch_sim CARDINAL (up, right, down, left) as (dx, dy)
_DX = np.array([0, 1, 0, -1], dtype=np.int64)
_DY = np.array([-1, 0, 1, 0], dtype=np.int64)
_NX = np.array([1, -1, 0, 0], dtype=np.int64)
_NY = np.array([0, 0, 1, -1], dtype=np.int64)

# blocked-action values (c1 outside the arena / not free at time 1)
BLOCKED = np.array(
    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.5, 0.0], dtype=np.float64
)


# ----------------------------------------------------------------------------- kernels
@njit(cache=True, nogil=True)
def _fill_world(bodies, head_ptr, seg_count, length, alive, e, gw, gh, ttl, own, snake):
    """ttl (max over segments), own-body offset (-1: none) and owner (-1: none) per cell."""
    S = alive.shape[1]
    ring = bodies.shape[2]
    for i in range(gw * gh):
        ttl[i] = 0
        own[i] = -1
        snake[i] = -1
    total = 0
    for s in range(S):
        if not alive[e, s]:
            continue
        n = seg_count[e, s]
        L = length[e, s]
        hp = head_ptr[e, s]
        for k in range(n):
            j = (hp - k) % ring
            x = bodies[e, s, j, 0]
            y = bodies[e, s, j, 1]
            if x < 0 or x >= gw or y < 0 or y >= gh:
                continue
            idx = y * gw + x
            t = L - k
            if snake[idx] < 0:
                total += 1
            if t > ttl[idx]:
                ttl[idx] = t
            if s == 0 and (own[idx] < 0 or k < own[idx]):
                own[idx] = k
            if snake[idx] < 0 or s == 0:
                snake[idx] = s
    return total


@njit(cache=True, nogil=True)
def _label(ttl, gw, gh, labels, sizes, queue):
    """4-connected components of cells with ttl <= 1; returns the number of components."""
    ncell = gw * gh
    for i in range(ncell):
        labels[i] = -1
    nlab = 0
    for i0 in range(ncell):
        if ttl[i0] > 1 or labels[i0] >= 0:
            continue
        labels[i0] = nlab
        head = 0
        tail = 1
        queue[0] = i0
        while head < tail:
            c = queue[head]
            head += 1
            x = c % gw
            y = c // gw
            for k in range(4):
                nx = x + _NX[k]
                ny = y + _NY[k]
                if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                    continue
                n = ny * gw + nx
                if ttl[n] > 1 or labels[n] >= 0:
                    continue
                labels[n] = nlab
                queue[tail] = n
                tail += 1
        sizes[nlab] = tail
        nlab += 1
    return nlab


@njit(cache=True, nogil=True)
def _flood(ttl, own, labels, largest, gw, gh, sx, sy, L, arr, nxt, bucket, hist, out):
    """Family A for one action (``out`` gets A_* in ACTION_FEATURES order, slots 0..6)."""
    tmax = 2 * L
    ncell = gw * gh
    for i in range(ncell):
        arr[i] = -1
    for t in range(tmax + 2):
        bucket[t] = -1
        hist[t] = 0
    i0 = sy * gw + sx
    arr[i0] = 1
    nxt[i0] = -1
    bucket[1] = i0
    own_first = -1
    arr_largest = -1
    own_reached = 0
    for t in range(1, tmax + 1):
        c = bucket[t]
        while c != -1:
            hist[t] += 1
            if own[c] >= 0:
                if own_first < 0:
                    own_first = t
                if t <= L:
                    own_reached += 1
            if arr_largest < 0 and labels[c] == largest and largest >= 0:
                arr_largest = t
            x = c % gw
            y = c // gw
            for k in range(4):
                nx = x + _NX[k]
                ny = y + _NY[k]
                if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                    continue
                n = ny * gw + nx
                if arr[n] != -1:
                    continue
                cand = t + 1
                if ttl[n] > cand:
                    cand = ttl[n]
                if cand > tmax:
                    arr[n] = tmax + 1  # settled beyond the horizon (monotone relaxation)
                    continue
                arr[n] = cand
                nxt[n] = bucket[cand]
                bucket[cand] = n
            c = nxt[c]
    # N(t) and the counting bound
    q4 = max(1, L // 4)
    q2 = max(1, L // 2)
    q1 = max(1, L)
    cum = 0
    tstar = L + 1
    n4 = 0
    n2 = 0
    n1 = 0
    for t in range(1, tmax + 1):
        cum += hist[t]
        if t == q4:
            n4 = cum
        if t == q2:
            n2 = cum
        if t == q1:
            n1 = cum
        if t <= L and cum < t and tstar == L + 1:
            tstar = t
    out[0] = min(n4 / q4, 8.0)
    out[1] = min(n2 / q2, 8.0)
    out[2] = min(n1 / q1, 8.0)
    out[3] = tstar / q1
    out[4] = (own_first / q1) if own_first >= 0 else 2.0
    return own_reached, arr_largest


@njit(cache=True, nogil=True)
def _row(
    bodies, head_ptr, seg_count, length, alive, direction, e, gw, gh,
    ttl, own, snake, labels, sizes, queue, arr, nxt, bucket, hist,
    do_a, out_a, out_s,
):  # fmt: skip
    ncell = gw * gh
    total = _fill_world(bodies, head_ptr, seg_count, length, alive, e, gw, gh, ttl, own, snake)
    nlab = _label(ttl, gw, gh, labels, sizes, queue)
    L = length[e, 0]
    Lf = float(max(L, 1))
    free = 0
    largest = -1
    lsize = 0
    second = 0
    ncomp32 = 0
    for i in range(nlab):
        sz = sizes[i]
        free += sz
        if sz >= 32:
            ncomp32 += 1
        if sz > lsize:
            second = lsize
            lsize = sz
            largest = i
        elif sz > second:
            second = sz
    out_s[0] = min(lsize / Lf, 16.0)
    out_s[1] = lsize / max(free, 1)
    out_s[2] = ncomp32
    out_s[3] = free / ncell
    out_s[4] = second / max(lsize, 1)
    # D: body shape
    ring = bodies.shape[2]
    hp = head_ptr[e, 0]
    hx = bodies[e, 0, hp, 0]
    hy = bodies[e, 0, hp, 1]
    n0 = seg_count[e, 0]
    d8 = 0
    d16 = 0
    x0 = gw
    x1 = -1
    y0 = gh
    y1 = -1
    for k in range(n0):
        j = (hp - k) % ring
        x = bodies[e, 0, j, 0]
        y = bodies[e, 0, j, 1]
        dx = abs(x - hx)
        dy = abs(y - hy)
        if dx <= 8 and dy <= 8:
            d8 += 1
        if dx <= 16 and dy <= 16:
            d16 += 1
        x0 = min(x0, x)
        x1 = max(x1, x)
        y0 = min(y0, y)
        y1 = max(y1, y)
    out_s[5] = d8 / 289.0
    out_s[6] = d16 / 1089.0
    out_s[7] = Lf / max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    out_s[8] = total / ncell
    # per action
    heading = direction[e, 0]
    labs = np.full(3, -1, dtype=np.int64)
    for a in range(3):
        hd = (heading + a - 1) % 4
        sx = hx + _DX[hd]
        sy = hy + _DY[hd]
        if sx < 0 or sx >= gw or sy < 0 or sy >= gh or ttl[sy * gw + sx] > 1:
            for f in range(out_a.shape[1]):
                out_a[a, f] = BLOCKED[f]
            continue
        labs[a] = labels[sy * gw + sx]
        out_a[a, 13] = 1.0 if labs[a] == largest else 0.0
    # B: frontier stats per action component (own and enemy body cells adjacent to it)
    f_own = np.zeros(3, dtype=np.int64)
    f_en = np.zeros(3, dtype=np.int64)
    f_half = np.zeros(3, dtype=np.int64)
    f_full = np.zeros(3, dtype=np.int64)
    f_min = np.full(3, -1, dtype=np.int64)
    for c in range(ncell):
        if snake[c] < 0 or ttl[c] <= 1:
            continue
        x = c % gw
        y = c // gw
        for a in range(3):
            la = labs[a]
            if la < 0:
                continue
            dup = False
            for b in range(a):
                if labs[b] == la:
                    dup = True
            if dup:
                continue
            adj = False
            for k in range(4):
                nx = x + _NX[k]
                ny = y + _NY[k]
                if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                    continue
                if labels[ny * gw + nx] == la:
                    adj = True
                    break
            if not adj:
                continue
            if own[c] >= 0:
                tt = L - own[c]
                R = sizes[la]
                f_own[a] += 1
                if 2 * tt <= R:
                    f_half[a] += 1
                if tt <= R:
                    f_full[a] += 1
                if f_min[a] < 0 or tt < f_min[a]:
                    f_min[a] = tt
            else:
                f_en[a] += 1
    for a in range(3):
        la = labs[a]
        if la < 0:
            continue
        src = a
        for b in range(a):
            if labs[b] == la:
                src = b
                break
        R = sizes[la]
        fo = f_own[src]
        out_a[a, 7] = fo / Lf
        out_a[a, 8] = (f_half[src] / fo) if fo > 0 else 0.0
        out_a[a, 9] = (f_full[src] / fo) if fo > 0 else 0.0
        if f_min[src] >= 0:
            out_a[a, 10] = f_min[src] / Lf
            out_a[a, 11] = min(max((R - f_min[src]) / Lf, -2.0), 8.0)
        else:
            out_a[a, 10] = 2.0
            out_a[a, 11] = 8.0
        tot = fo + f_en[src]
        out_a[a, 12] = (fo / tot) if tot > 0 else 0.5
    # A: time-aware floods
    if do_a:
        n_own = 0
        for c in range(ncell):
            if own[c] >= 0:
                n_own += 1
        tmp = np.zeros(5, dtype=np.float64)
        for a in range(3):
            if labs[a] < 0:
                continue
            hd = (heading + a - 1) % 4
            sx = hx + _DX[hd]
            sy = hy + _DY[hd]
            reached, arr_l = _flood(
                ttl, own, labels, largest, gw, gh, sx, sy, L, arr, nxt, bucket, hist, tmp
            )
            for f in range(5):
                out_a[a, f] = tmp[f]
            out_a[a, 5] = reached / max(n_own, 1)
            out_a[a, 6] = (arr_l / Lf) if arr_l >= 0 else 2.0


@njit(cache=True, nogil=True)
def _rows(
    bodies, head_ptr, seg_count, length, alive, direction, envs, gw, gh, do_a, out_a, out_s
):  # fmt: skip
    ncell = gw * gh
    ttl = np.zeros(ncell, dtype=np.int64)
    own = np.zeros(ncell, dtype=np.int64)
    snake = np.zeros(ncell, dtype=np.int64)
    labels = np.zeros(ncell, dtype=np.int64)
    sizes = np.zeros(ncell + 1, dtype=np.int64)
    queue = np.zeros(ncell, dtype=np.int64)
    arr = np.zeros(ncell, dtype=np.int64)
    nxt = np.zeros(ncell, dtype=np.int64)
    Lmax = 1
    for r in range(envs.shape[0]):
        Lmax = max(Lmax, length[envs[r], 0])
    bucket = np.zeros(2 * Lmax + 2, dtype=np.int64)
    hist = np.zeros(2 * Lmax + 2, dtype=np.int64)
    for r in range(envs.shape[0]):
        _row(
            bodies, head_ptr, seg_count, length, alive, direction, envs[r], gw, gh,
            ttl, own, snake, labels, sizes, queue, arr, nxt, bucket, hist,
            do_a, out_a[r], out_s[r],
        )  # fmt: skip


def _i64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a), dtype=np.int64)


def escape_features(sim, envs, do_a: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """``((N, 3, NA) action features, (N, NS) state features)`` for hero rows ``(env, 0)``.

    ``envs``: the hero envs (alive heroes only). ``do_a=False`` skips the family-A floods
    (their slots then hold zeros), for timing the families separately.
    """
    envs = _i64(envs).reshape(-1)
    N = len(envs)
    out_a = np.zeros((N, 3, NA), dtype=np.float64)
    out_s = np.zeros((N, NS), dtype=np.float64)
    if N:
        _rows(
            _i64(sim.bodies), _i64(sim.head_ptr), _i64(sim.seg_count), _i64(sim.length),
            np.ascontiguousarray(sim.alive, dtype=np.bool_), _i64(sim.direction), envs,
            int(sim.grid_w), int(sim.grid_h), bool(do_a), out_a, out_s,
        )  # fmt: skip
    return out_a, out_s


def escape_features_timed(sim, envs) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """``escape_features`` plus the kernel seconds of the static part (B/C/D) and of A."""
    t0 = time.perf_counter()
    _, _ = escape_features(sim, envs, do_a=False)
    t1 = time.perf_counter()
    a, s = escape_features(sim, envs, do_a=True)
    t2 = time.perf_counter()
    return a, s, {"static_bcd": t1 - t0, "full": t2 - t1}
