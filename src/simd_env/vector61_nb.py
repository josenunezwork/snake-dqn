"""Compiled (numba) ``vector61`` featurizer: float32-identical to :class:`Vector61Featurizer`.

:class:`Vector61FeaturizerNB` is a drop-in for
:class:`src.simd_env.vector61_featurizer.Vector61Featurizer` (same constructor, same
``featurize(sim, rows, memory, update_memory)`` contract, same trend-memory updates).
It is opt-in: :class:`src.simd_env.vector61_policy.Vector61Runtime` uses it only with
``backend="numba"`` (or env ``SNAKE_V61_BACKEND=numba``); the default stays the NumPy
featurizer. Parity (bit-equal float32 rows and trend memory on GridBatchSim training
worlds, short and long bodies) is pinned by ``tests/test_vector61_nb.py``.

Why the results are identical, not just close: every column is the same IEEE-754 double
expression as the NumPy code, evaluated in the same operation order on the same exact
integer inputs (pixel deltas, squared distances), then cast to float32 once. The
differences are only in *which* inputs are visited:

- Order-independent reductions (sector ``max`` / ``count``, ``any``, ``min``) are computed
  per row over the same point set; body points are visited through a coarse bucket grid
  so a row only scans buckets within its radius. A point outside the radius cannot
  contribute: sector danger needs ``d2 <= maxd**2`` and the per-action obstacle term only
  changes the result when ``d2 <= max_check**2`` from a candidate cell one step from the
  head (a nearer obstacle than the ``max_check``-capped wall distance), both inside the
  scanned Chebyshev box.
- Order-dependent choices keep the NumPy tie rules: nearest food is the first minimum in
  live food-list order, enemies are the stable ``(distance, slot)`` order.
- Sector indices come from the same ``math.atan2`` lookup table.
- The capped flood fill is a plain BFS that stops at ``cap`` cells: it returns
  ``min(component size, cap)``, the quantity the NumPy certificates and the memoized
  fallback compute.

Rows whose sector-table index would fall outside the table (impossible for in-arena
bodies) make the call fall back to the NumPy featurizer instead of reading out of bounds.
"""

from __future__ import annotations

import math
from itertools import chain
from typing import Dict, Optional

import numpy as np
from numba import njit

from src.simd_env.batch_sim import CARDINAL, BatchSim
from src.simd_env.vector61_featurizer import (
    _CARDINAL_TUPLES,
    FREE_SPACE_BFS_CAP,
    FREE_SPACE_MIN_CAP,
    Vector61Featurizer,
    Vector61Memory,
    Vector61Params,
)

__all__ = ["BUCKET", "Vector61FeaturizerNB"]

#: Body-point bucket edge in cells.
BUCKET = 8
_I64_MAX_NB = np.iinfo(np.int64).max  # a compile-time constant inside the kernels


@njit(cache=True, nogil=True)
def _bfs_capped(blk, stamp, vis, vstamp, queue, gw, gh, x, y, cap):
    """``min(size of (x, y)'s 4-connected free component, cap)``; 0 if blocked / off-board."""
    if x < 0 or x >= gw or y < 0 or y >= gh:
        return 0
    c0 = x * gh + y
    if blk[c0] == stamp:
        return 0
    vis[c0] = vstamp
    queue[0] = c0
    head = 0
    tail = 1
    count = 1
    if count >= cap:
        return cap
    while head < tail:
        c = queue[head]
        head += 1
        cx = c // gh
        cy = c - cx * gh
        for k in range(4):
            if k == 0:
                if cx + 1 >= gw:
                    continue
                nb = c + gh
            elif k == 1:
                if cx <= 0:
                    continue
                nb = c - gh
            elif k == 2:
                if cy + 1 >= gh:
                    continue
                nb = c + 1
            else:
                if cy <= 0:
                    continue
                nb = c - 1
            if blk[nb] == stamp or vis[nb] == vstamp:
                continue
            vis[nb] = vstamp
            queue[tail] = nb
            tail += 1
            count += 1
            if count >= cap:
                return cap
    return count


@njit(cache=True, nogil=True)
def _featurize_kernel(
    order,
    rows_e,
    rows_s,
    out,
    alive,
    length,
    direction,
    seg_count,
    head_ptr,
    bodies,
    cap_ring,
    food_buf,
    food_n,
    lut,
    rx,
    ry,
    onehot,
    card,
    ss,
    W,
    H,
    gw,
    gh,
    max_length,
    nsec,
    maxd,
    mc,
    use_boundary,
    min_boost,
    expected_per_sector,
    max_dim,
    board_diag,
    use_free_space,
    fs_cap,
    fs_min_cap,
    use_mem,
    upd_mem,
    prev_dist,
    prev_id,
    bucket,
):
    """Fill ``out`` (float64, zero-initialised) for alive rows. Returns 0, or 1 when a
    sector-table index would be out of range (the caller then falls back)."""
    N = order.shape[0]
    S = alive.shape[1]
    lut_w = lut.shape[0]
    lut_h = lut.shape[1]
    nbx = gw // bucket + 3
    nby = gh // bucket + 3
    nb_all = nbx * nby
    bcount = np.zeros(nb_all + 1, dtype=np.int64)
    blk = np.zeros(gw * gh, dtype=np.int64)
    vis = np.zeros(gw * gh, dtype=np.int64)
    queue = np.empty(gw * gh + 1, dtype=np.int64)
    stamp = 0
    vstamp = 0
    maxd2 = maxd * maxd
    ss2 = ss * ss
    rd_cells = maxd // ss
    pa_cells = (mc + ss) // ss + 1  # per-action box radius (cells), with slack
    rwin = rd_cells if rd_cells > pa_cells else pa_cells
    pa_px = mc + ss
    dang = np.zeros(nsec, dtype=np.float64)
    dcnt = np.zeros(nsec, dtype=np.int64)
    fcnt = np.zeros(nsec, dtype=np.int64)
    hit = np.zeros(3, dtype=np.bool_)
    d2min = np.zeros(3, dtype=np.int64)
    cxa = np.zeros(3, dtype=np.int64)
    cya = np.zeros(3, dtype=np.int64)
    dd3 = np.zeros(3, dtype=np.int64)
    i = 0
    while i < N:
        e = rows_e[order[i]]
        j = i
        while j < N and rows_e[order[j]] == e:
            j += 1
        any_alive = False
        for q in range(i, j):
            if alive[e, rows_s[order[q]]]:
                any_alive = True
                break
        if not any_alive:
            i = j
            continue
        # ---- per-env body points (alive snakes, head-first), bucketed by cell
        total = 0
        for t in range(S):
            if alive[e, t]:
                total += seg_count[e, t]
        pxs = np.empty(total, dtype=np.int64)
        pys = np.empty(total, dtype=np.int64)
        pown = np.empty(total, dtype=np.int64)
        pself = np.empty(total, dtype=np.bool_)
        poth = np.empty(total, dtype=np.bool_)
        pb = np.empty(total, dtype=np.int64)
        stamp += 1
        for b in range(nb_all + 1):
            bcount[b] = 0
        p = 0
        for t in range(S):
            if not alive[e, t]:
                continue
            n = seg_count[e, t]
            L = length[e, t]
            hp = head_ptr[e, t]
            drop_other = 1 if (n > 1 and n >= L) else 0
            drop_self = 1 if (n >= L and n > 2) else 0
            for k in range(n):
                r = (hp - k) % cap_ring
                x = bodies[e, t, r, 0]
                y = bodies[e, t, r, 1]
                pxs[p] = x
                pys[p] = y
                pown[p] = t
                poth[p] = k < n - drop_other
                pself[p] = k >= 2 and k < n - drop_self
                bx = x // bucket + 1
                by = y // bucket + 1
                if bx < 0:
                    bx = 0
                elif bx >= nbx:
                    bx = nbx - 1
                if by < 0:
                    by = 0
                elif by >= nby:
                    by = nby - 1
                pb[p] = bx * nby + by
                bcount[pb[p] + 1] += 1
                if x >= 0 and x < gw and y >= 0 and y < gh:
                    blk[x * gh + y] = stamp
                p += 1
        for b in range(nb_all):
            bcount[b + 1] += bcount[b]
        bstart = bcount[:nb_all].copy()
        pidx = np.empty(total, dtype=np.int64)
        fill = bcount[:nb_all].copy()
        for q in range(total):
            pidx[fill[pb[q]]] = q
            fill[pb[q]] += 1
        bend = fill
        nf = food_n[e]
        # ---- rows of this env
        for q in range(i, j):
            row = order[q]
            s = rows_s[row]
            if not alive[e, s]:
                continue
            hr = head_ptr[e, s] % cap_ring
            hcx = bodies[e, s, hr, 0]
            hcy = bodies[e, s, hr, 1]
            hx = hcx * ss
            hy = hcy * ss
            L = length[e, s]
            logical = L if L > 1 else 1
            d = direction[e, s]
            out[row, onehot[d]] = 1.0
            v = logical / max_length
            out[row, 4] = v if v < 1.0 else 1.0
            # food (5..23)
            for k in range(nsec):
                fcnt[k] = 0
            best = _I64_MAX_NB
            bdx = 0
            bdy = 0
            for f in range(nf):
                dx = food_buf[e, f, 0] * ss - hx
                dy = food_buf[e, f, 1] * ss - hy
                d2 = dx * dx + dy * dy
                if d2 < best:
                    best = d2
                    bdx = dx
                    bdy = dy
                li = dx // ss + rx
                lj = dy // ss + ry
                if li < 0 or li >= lut_w or lj < 0 or lj >= lut_h:
                    return 1
                fcnt[lut[li, lj]] += 1
            if nf > 0:
                out[row, 5] = bdx / max_dim
                out[row, 6] = bdy / max_dim
                v = math.sqrt(float(best)) / board_diag
                out[row, 7] = v if v < 1.0 else 1.0
            else:
                out[row, 7] = 1.0
            for k in range(nsec):
                v = float(fcnt[k]) / expected_per_sector
                out[row, 8 + k] = v if v < 1.0 else 1.0
            # body points: sector danger (24..39) + per-action collision terms (54..56)
            for k in range(nsec):
                dang[k] = 0.0
                dcnt[k] = 0
            for a in range(3):
                dd = (d + 3 + a) % 4
                dd3[a] = dd
                cxa[a] = hx + card[dd, 0] * ss
                cya[a] = hy + card[dd, 1] * ss
                hit[a] = False
                d2min[a] = _I64_MAX_NB
            bx0 = (hcx - rwin) // bucket + 1
            bx1 = (hcx + rwin) // bucket + 1
            by0 = (hcy - rwin) // bucket + 1
            by1 = (hcy + rwin) // bucket + 1
            bx0 = min(max(bx0, 0), nbx - 1)
            bx1 = min(max(bx1, 0), nbx - 1)
            by0 = min(max(by0, 0), nby - 1)
            by1 = min(max(by1, 0), nby - 1)
            for bx in range(bx0, bx1 + 1):
                for by in range(by0, by1 + 1):
                    b = bx * nby + by
                    for z in range(bstart[b], bend[b]):
                        pt = pidx[z]
                        if pown[pt] == s:
                            if not pself[pt]:
                                continue
                        elif not poth[pt]:
                            continue
                        px = pxs[pt] * ss
                        py = pys[pt] * ss
                        dx = px - hx
                        dy = py - hy
                        d2 = dx * dx + dy * dy
                        if d2 <= maxd2:
                            dist = math.sqrt(float(d2))
                            v = 1.0 - dist / float(maxd)
                            if v < 0.0:
                                v = 0.0
                            v = v * 1.5
                            li = dx // ss + rx
                            lj = dy // ss + ry
                            if li < 0 or li >= lut_w or lj < 0 or lj >= lut_h:
                                return 1
                            sec = lut[li, lj]
                            if v > dang[sec]:
                                dang[sec] = v
                            if v > 0.0:
                                dcnt[sec] += 1
                        if dx <= pa_px and dx >= -pa_px and dy <= pa_px and dy >= -pa_px:
                            for a in range(3):
                                ex = cxa[a] - px
                                ey = cya[a] - py
                                e2 = ex * ex + ey * ey
                                if e2 < ss2:
                                    hit[a] = True
                                if e2 < d2min[a]:
                                    d2min[a] = e2
            if use_boundary:
                lim = maxd + ss
                for side in range(4):
                    if side == 0:
                        on = hx < lim
                    elif side == 1:
                        on = (W - hx) < lim
                    elif side == 2:
                        on = hy < lim
                    else:
                        on = (H - hy) < lim
                    if not on:
                        continue
                    o = -maxd
                    while o <= maxd:
                        if side == 0 or side == 1:
                            wy = hy + o
                            ok = wy >= 0 and wy < H
                            wx = -ss if side == 0 else W + ss
                        else:
                            wx = hx + o
                            ok = wx >= 0 and wx < W
                            wy = -ss if side == 2 else H + ss
                        if ok:
                            dx = wx - hx
                            dy = wy - hy
                            d2 = dx * dx + dy * dy
                            if d2 <= maxd2:
                                dist = math.sqrt(float(d2))
                                v = 1.0 - dist / float(maxd)
                                if v < 0.0:
                                    v = 0.0
                                v = v * 2.0
                                li = dx // ss + rx
                                lj = dy // ss + ry
                                if li < 0 or li >= lut_w or lj < 0 or lj >= lut_h:
                                    return 1
                                sec = lut[li, lj]
                                if v > dang[sec]:
                                    dang[sec] = v
                                if v > 0.0:
                                    dcnt[sec] += 1
                        o += ss
            for k in range(nsec):
                v = dang[k]
                c = dcnt[k]
                if c > 1:
                    bonus = c / 5.0
                    if bonus > 1.0:
                        bonus = 1.0
                    bonus = bonus * 0.2
                    v = v + bonus
                    if v > 2.0:
                        v = 2.0
                out[row, 24 + k] = v if v < 1.0 else 1.0
            # boundaries (40..43)
            out[row, 40] = hx / W
            out[row, 41] = (W - hx) / W
            out[row, 42] = hy / H
            out[row, 43] = (H - hy) / H
            # enemies (44..53): stable (distance, slot) order
            t1 = -1
            t2 = -1
            d1 = np.inf
            d2s = np.inf
            for t in range(S):
                if t == s or not alive[e, t]:
                    continue
                tr = head_ptr[e, t] % cap_ring
                dx = bodies[e, t, tr, 0] * ss - hx
                dy = bodies[e, t, tr, 1] * ss - hy
                dist = math.sqrt(float(dx * dx + dy * dy))
                if dist < d1:
                    t2 = t1
                    d2s = d1
                    t1 = t
                    d1 = dist
                elif dist < d2s:
                    t2 = t
                    d2s = dist
            if t1 >= 0:
                tr = head_ptr[e, t1] % cap_ring
                ehx = bodies[e, t1, tr, 0] * ss
                ehy = bodies[e, t1, tr, 1] * ss
                out[row, 44] = (ehx - hx) / max_dim
                out[row, 45] = (ehy - hy) / max_dim
                lt = length[e, t1]
                lt = lt if lt > 1 else 1
                v = lt / logical
                out[row, 46] = (v if v < 2.0 else 2.0) / 2.0
                hdx = card[direction[e, t1], 0]
                hdy = card[direction[e, t1], 1]
                out[row, 47] = float(hdx)
                out[row, 48] = float(hdy)
                if use_mem:
                    prev = prev_dist[e, s]
                    if (not np.isinf(prev)) and prev_id[e, s] == t1:
                        if d1 < prev:
                            out[row, 49] = 1.0
                        elif d1 > prev:
                            out[row, 49] = -1.0
                kx = hx - (ehx + hdx * ss)
                ky = hy - (ehy + hdy * ss)
                kd = math.sqrt(float(kx * kx + ky * ky))
                if kd < ss * 3:
                    out[row, 53] = 1.0
            if use_mem and upd_mem:
                if t1 >= 0:
                    prev_dist[e, s] = d1
                    prev_id[e, s] = t1
                else:
                    prev_dist[e, s] = np.inf
                    prev_id[e, s] = -1
            if t2 >= 0:
                tr = head_ptr[e, t2] % cap_ring
                out[row, 50] = (bodies[e, t2, tr, 0] * ss - hx) / max_dim
                out[row, 51] = (bodies[e, t2, tr, 1] * ss - hy) / max_dim
                lt = length[e, t2]
                lt = lt if lt > 1 else 1
                v = lt / logical
                out[row, 52] = (v if v < 2.0 else 2.0) / 2.0
            # per-action danger (54..56)
            for a in range(3):
                nx = cxa[a]
                ny = cya[a]
                wall_hit = nx < 0 or nx >= W or ny < 0 or ny >= H
                mw = min(min(nx, W - nx), min(ny, H - ny))
                nearest = float(mc if mc < mw else mw)
                if d2min[a] != _I64_MAX_NB:
                    ob = math.sqrt(float(d2min[a]))
                    if ob < nearest:
                        nearest = ob
                if wall_hit or hit[a]:
                    out[row, 54 + a] = 1.0
                elif nearest < mc:
                    v = 1.0 - nearest / mc
                    out[row, 54 + a] = v if v < 0.95 else 0.95
            # boost (57)
            if L >= min_boost:
                out[row, 57] = 1.0
            # free space (58..60)
            if use_free_space:
                capr = L * 2
                if capr < fs_min_cap:
                    capr = fs_min_cap
                if capr > fs_cap:
                    capr = fs_cap
                for a in range(3):
                    dd = dd3[a]
                    vstamp += 1
                    size = _bfs_capped(
                        blk,
                        stamp,
                        vis,
                        vstamp,
                        queue,
                        gw,
                        gh,
                        hcx + card[dd, 0],
                        hcy + card[dd, 1],
                        fs_cap,
                    )
                    cnt = size if size < capr else capr
                    v = cnt / capr
                    out[row, 58 + a] = v if v < 1.0 else 1.0
        i = j
    return 0


class Vector61FeaturizerNB(Vector61Featurizer):
    """:class:`Vector61Featurizer` with a compiled ``featurize`` (float32-identical)."""

    def __init__(self, params: Vector61Params) -> None:
        super().__init__(params)
        self._lut = np.ascontiguousarray(self.lut, dtype=np.int64)
        self._onehot = np.ascontiguousarray(self.onehot_index, dtype=np.int64)
        self._card = np.ascontiguousarray(CARDINAL, dtype=np.int64)
        assert tuple(map(tuple, self._card.tolist())) == _CARDINAL_TUPLES
        self._food_sim_id: Optional[int] = None
        self._food_lists: Dict[int, list] = {}
        self._food_buf = np.zeros((0, 0, 2), dtype=np.int64)
        self._food_n = np.zeros(0, dtype=np.int64)
        self.fallbacks = 0

    def _food_arrays(self, sim: BatchSim, envs: np.ndarray):
        """Per-env food cells in live list order, refreshed only for changed lists."""
        # GridBatchSim maintains this ordered mirror and version at every food
        # mutation.  It has the same tie-sensitive order as ``food_cells`` and
        # lets the hot carry path avoid an E-by-F Python list comparison and
        # rebuild.  Plain BatchSim intentionally keeps the legacy fallback.
        direct = getattr(sim, "_food_array", None)
        direct_n = getattr(sim, "_food_n", None)
        if (
            isinstance(direct, np.ndarray)
            and isinstance(direct_n, np.ndarray)
            and direct.ndim == 3
            and direct.shape[0] == int(sim.E)
            and direct.shape[2] == 2
            and direct_n.shape == (int(sim.E),)
        ):
            return direct, direct_n
        E = int(sim.E)
        if self._food_sim_id != id(sim) or self._food_buf.shape[0] != E:
            self._food_sim_id = id(sim)
            self._food_lists = {}
            self._food_buf = np.zeros((E, 8, 2), dtype=np.int64)
            self._food_n = np.zeros(E, dtype=np.int64)
        for e in envs.tolist():
            cur = sim.food_cells[e]
            old = self._food_lists.get(e)
            if old is not None and old == cur:
                continue
            n = len(cur)
            if n > self._food_buf.shape[1]:
                grown = np.zeros((E, max(n, 2 * self._food_buf.shape[1]), 2), dtype=np.int64)
                grown[:, : self._food_buf.shape[1]] = self._food_buf
                self._food_buf = grown
            if n:  # fromiter over the flattened tuples: ~3x faster than a list assignment
                flat = np.fromiter(chain.from_iterable(cur), dtype=np.int64, count=2 * n)
                self._food_buf[e, :n] = flat.reshape(n, 2)
            self._food_n[e] = n
            self._food_lists[e] = list(cur)
        return self._food_buf, self._food_n

    def featurize(
        self,
        sim: BatchSim,
        rows: Optional[np.ndarray] = None,
        memory: Optional[Vector61Memory] = None,
        update_memory: bool = True,
    ) -> np.ndarray:
        """Same contract as :meth:`Vector61Featurizer.featurize` (float32-identical)."""
        if sim.cfg.game_width != self.W or sim.cfg.game_height != self.H:
            raise ValueError("sim world size does not match the featurizer params")
        if sim.s != self.ss:
            raise ValueError("sim segment_size does not match the featurizer params")
        if rows is None:
            rows = np.argwhere(sim.alive)
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        out = np.zeros((len(rows), self.input_size), dtype=np.float64)
        if not len(rows):
            return out.astype(np.float32)
        if memory is not None and update_memory:
            if len(np.unique(rows[:, 0] * sim.S + rows[:, 1])) != len(rows):
                raise ValueError("rows must be unique when updating trend memory")
        rows_e = np.ascontiguousarray(rows[:, 0])
        rows_s = np.ascontiguousarray(rows[:, 1])
        order = np.argsort(rows_e, kind="stable")
        food_buf, food_n = self._food_arrays(sim, np.unique(rows_e))
        use_mem = memory is not None
        if use_mem:
            prev_dist, prev_id = memory.prev_dist, memory.prev_id
            saved = (prev_dist.copy(), prev_id.copy())
        else:
            prev_dist = np.zeros((1, 1), dtype=np.float64)
            prev_id = np.zeros((1, 1), dtype=np.int64)
        p = self.p
        status = _featurize_kernel(
            order,
            rows_e,
            rows_s,
            out,
            _i64(sim.alive, np.bool_),
            _i64(sim.length),
            _i64(sim.direction),
            _i64(sim.seg_count),
            _i64(sim.head_ptr),
            _i64(sim.bodies),
            int(sim.cap),
            food_buf,
            food_n,
            self._lut,
            int(self.rx),
            int(self.ry),
            self._onehot,
            self._card,
            int(self.ss),
            int(self.W),
            int(self.H),
            int(self.gw),
            int(self.gh),
            int(p.max_length),
            int(p.num_sectors),
            int(self.maxd),
            int(self.max_check),
            bool(p.use_boundary_as_danger),
            int(p.min_boost_length),
            float(self.expected_per_sector),
            int(self.max_dim),
            float(self.board_diagonal),
            bool(p.use_free_space),
            int(FREE_SPACE_BFS_CAP),
            int(FREE_SPACE_MIN_CAP),
            use_mem,
            bool(update_memory),
            prev_dist,
            prev_id,
            int(BUCKET),
        )
        if status:  # never expected: let the reference implementation decide (or raise)
            self.fallbacks += 1
            if use_mem:
                prev_dist[...] = saved[0]
                prev_id[...] = saved[1]
            return super().featurize(sim, rows, memory, update_memory)
        return out.astype(np.float32)


def _i64(a: np.ndarray, dtype=np.int64) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=dtype)
