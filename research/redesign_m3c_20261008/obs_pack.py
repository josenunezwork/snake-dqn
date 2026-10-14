"""Lossless packed storage of ``ego2s-b`` / ``ego2s-bv8`` observation planes (perf-ram).

The trainer's observation rings and demonstration buffer store every row's local planes
``(7, 31, 31)`` and global planes ``(4, 37, 37)`` as raw uint8: 12,203 bytes per row, most of
it redundant. ``--obs-pack`` stores them in :class:`PackLayout` form (3,064 bytes for the
training arena) and unpacks them on gather: the batch the network sees is byte-identical
(``tests/test_obs_pack.py``). The scalars and masks are stored unchanged.

Every packed field follows from the featurizer's definitions
(:mod:`src.simd_env.ego_raster`, :mod:`src.simd_env.ego_raster_nb`,
:mod:`src.simd_env.ego_raster_b`):

=================  ===========================================  ===========================
plane              values (why)                                 packed as
=================  ===========================================  ===========================
local 0 own_ttl    0..255                                       ``ttl`` byte + ``own`` bit:
local 1 enemy_ttl  0..255; a cell has ONE owner, so at most     ch0 = own ? ttl : 0,
                   one of ch0 / ch1 is non-zero per pixel       ch1 = own ? 0 : ttl
local 2 enemy_hd   non-zero only on an enemy's head cell        sparse: <= ``head_cap``
                   (``k == 0``): at most ``S - 1`` pixels       (pixel, value) entries
local 3 food       {0, 128, 255}                                2-bit code  } one byte with
local 4 wall       {0, 255}                                     1 bit       } the reach
local 5 reach      0..``reach_steps`` (24 <= 31)                5 bits      } time
local 6 enemy_nx   {0, 128, 255}                                2-bit code (+ own bit: a nibble)
global 3 outside   255 off the arena's coarse map, 0 on it;     the in-arena rectangle
                   the crop is a 90-degree rotation + shift     ``(r0, c0, h, w)``
                   of the zero-padded map, so the 0-cells form
                   an axis-aligned rectangle of ``hc x wc``
global 0-2 masses  0 off the map (zero padding)                 the rectangle's bytes only
dead rows          every plane 0                                flag 0
=================  ===========================================  ===========================

Packing VERIFIES each invariant per row and raises ``ValueError`` (naming the row and the
broken rule) instead of storing a row it cannot reproduce exactly: a packed row always
unpacks to the original bytes. Two implementations: NumPy (:func:`pack_ref`,
:func:`unpack_ref`; the readable definition) and numba kernels (:func:`pack`,
:func:`unpack_into`; what the trainer runs), pinned equal in the tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

LOCAL_SHAPE = (7, 31, 31)
GLOBAL_SHAPE = (4, 37, 37)
N_LOCAL_PX = LOCAL_SHAPE[1] * LOCAL_SHAPE[2]
N_GLOBAL_PX = GLOBAL_SHAPE[1] * GLOBAL_SHAPE[2]
RAW_OBS_BYTES = int(np.prod(LOCAL_SHAPE) + np.prod(GLOBAL_SHAPE))  # 12,203
REACH_MAX = 31  # 5 bits
CODES = np.array([0, 128, 255, 0], dtype=np.uint8)  # 2-bit code -> value (3: never written)

VIOLATIONS = {
    1: "own_ttl and enemy_ttl both non-zero on one pixel",
    2: "food plane value not in {0, 128, 255}",
    3: "wall plane value not in {0, 255}",
    4: f"reach_time above {REACH_MAX}",
    5: "enemy_next plane value not in {0, 128, 255}",
    6: "more enemy-head pixels than head_cap",
    7: "outside plane value not in {0, 255}",
    8: "in-arena (outside == 0) cells do not form a rectangle",
    9: "non-zero mass outside the in-arena rectangle",
    10: "in-arena rectangle larger than rect_cells",
}


@dataclass(frozen=True)
class PackLayout:
    """Byte layout of one packed row (``nbytes`` uint8).

    ``rect_cells``: capacity of the global in-arena rectangle (``hc * wc`` coarse blocks of
    the arena, see :func:`layout_for_arena`); ``head_cap``: enemy-head entries (>= S - 1).
    """

    rect_cells: int
    head_cap: int = 8

    def __post_init__(self) -> None:
        if not 0 < int(self.rect_cells) <= N_GLOBAL_PX:
            raise ValueError("rect_cells out of range")
        if not 0 < int(self.head_cap) <= 255:
            raise ValueError("head_cap out of range")

    # offsets (bytes) ------------------------------------------------------------------------
    @property
    def off_b(self) -> int:  # reach | food << 5 | wall << 7, one byte per local pixel
        return N_LOCAL_PX

    @property
    def off_c(self) -> int:  # own | enemy_next << 1, one nibble per local pixel
        return 2 * N_LOCAL_PX

    @property
    def off_h(self) -> int:  # [count] + head_cap x (pixel lo, pixel hi, value)
        return self.off_c + (N_LOCAL_PX + 1) // 2

    @property
    def off_g(self) -> int:  # [flag, r0, c0, h, w]
        return self.off_h + 1 + 3 * int(self.head_cap)

    @property
    def off_gv(self) -> int:  # 3 x rect_cells mass bytes (own, enemy, food), row-major
        return self.off_g + 5

    @property
    def nbytes(self) -> int:
        n = self.off_gv + 3 * int(self.rect_cells)
        return -(-n // 8) * 8  # 8-byte rows keep the arrays after it aligned

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["PackLayout"]:
        return None if d is None else cls(**d)

    def consts(self) -> Tuple[int, ...]:
        return (
            int(self.rect_cells),
            int(self.head_cap),
            self.off_b,
            self.off_c,
            self.off_h,
            self.off_g,
            self.off_gv,
        )


def layout_for_arena(grid_w: int, grid_h: int, num_snakes: int, cfg=None) -> PackLayout:
    """The layout for an arena of ``grid_w x grid_h`` cells with ``num_snakes`` slots.

    The global crop shows the arena's coarse map (``ceil(H / cell) x ceil(W / cell)``
    blocks), rotated by a multiple of 90 degrees, clipped by the ``global_size`` window.
    """
    from src.simd_env.ego_raster import EgoRasterConfig

    cfg = cfg or EgoRasterConfig(version="b")
    if (cfg.local_size, cfg.global_size) != (LOCAL_SHAPE[1], GLOBAL_SHAPE[1]):
        raise ValueError("packed layout is defined for the 31 / 37 ego2s windows")
    if cfg.reach_steps > REACH_MAX:
        raise ValueError(f"reach_steps {cfg.reach_steps} does not fit 5 bits")
    cell, gs = cfg.global_cell, cfg.global_size
    hc, wc = -(-int(grid_h) // cell), -(-int(grid_w) // cell)
    return PackLayout(rect_cells=min(hc, gs) * min(wc, gs), head_cap=max(8, int(num_snakes) - 1))


def raise_on_violation(status: np.ndarray) -> None:
    bad = np.flatnonzero(status)
    if len(bad):
        code = int(status[bad[0]])
        raise ValueError(
            f"observation row {int(bad[0])} cannot be packed losslessly: "
            f"{VIOLATIONS.get(code, code)} ({len(bad)} bad rows)"
        )


# ----------------------------------------------------------------------------- NumPy reference
def pack_ref(local: np.ndarray, glob: np.ndarray, layout: PackLayout) -> np.ndarray:
    """The packed rows ``(N, layout.nbytes)`` (NumPy reference; raises on a violation)."""
    N = len(local)
    L = np.asarray(local, dtype=np.uint8).reshape(N, LOCAL_SHAPE[0], N_LOCAL_PX)
    G = np.asarray(glob, dtype=np.uint8).reshape(N, GLOBAL_SHAPE[0], N_GLOBAL_PX)
    status = np.zeros(N, dtype=np.int64)

    def flag(code: int, bad: np.ndarray) -> None:
        status[(status == 0) & bad] = code

    def code_of(x: np.ndarray, code: int) -> np.ndarray:
        flag(code, ~np.isin(x, (0, 128, 255)).all(axis=1))
        return np.where(x == 255, 2, np.where(x == 128, 1, 0)).astype(np.uint8)

    flag(1, ((L[:, 0] != 0) & (L[:, 1] != 0)).any(axis=1))
    fc = code_of(L[:, 3], 2)
    flag(3, ~np.isin(L[:, 4], (0, 255)).all(axis=1))
    flag(4, (L[:, 5] > REACH_MAX).any(axis=1))
    ec = code_of(L[:, 6], 5)
    flag(6, (L[:, 2] != 0).sum(axis=1) > layout.head_cap)
    out = np.zeros((N, layout.nbytes), dtype=np.uint8)
    own = (L[:, 0] != 0).astype(np.uint8)
    out[:, :N_LOCAL_PX] = np.where(own == 1, L[:, 0], L[:, 1])
    wall = (L[:, 4] == 255).astype(np.uint8)
    out[:, layout.off_b : layout.off_b + N_LOCAL_PX] = (
        (L[:, 5] & 31) | (fc << 5) | (wall << 7)
    ).astype(np.uint8)
    nib = own | (ec << 1)
    nib = np.concatenate([nib, np.zeros((N, 1), np.uint8)], axis=1) if N_LOCAL_PX % 2 else nib
    out[:, layout.off_c : layout.off_h] = nib[:, 0::2] | (nib[:, 1::2] << 4)
    gs = GLOBAL_SHAPE[1]
    for i in range(N):
        heads = np.flatnonzero(L[i, 2])
        if len(heads) <= layout.head_cap:
            out[i, layout.off_h] = len(heads)
            for k, p in enumerate(heads):
                q = layout.off_h + 1 + 3 * k
                out[i, q : q + 3] = (p & 255, p >> 8, L[i, 2, p])
        if not G[i].any():
            continue  # flag 0: every global plane zero (a dead row)
        out_ = G[i, 3]
        if not np.isin(out_, (0, 255)).all():
            flag(7, np.arange(N) == i)
            continue
        inside = (out_ == 0).reshape(gs, gs)
        rr, cc = np.nonzero(inside)
        if len(rr):
            r0, c0 = int(rr.min()), int(cc.min())
            h, w = int(rr.max()) - r0 + 1, int(cc.max()) - c0 + 1
        else:
            r0 = c0 = h = w = 0
        if len(rr) != h * w:
            flag(8, np.arange(N) == i)
            continue
        box = np.zeros((gs, gs), dtype=bool)
        box[r0 : r0 + h, c0 : c0 + w] = True
        if G[i, :3].reshape(3, gs, gs)[:, ~box].any():
            flag(9, np.arange(N) == i)
            continue
        if h * w > layout.rect_cells:
            flag(10, np.arange(N) == i)
            continue
        out[i, layout.off_g : layout.off_g + 5] = (1, r0, c0, h, w)
        vals = G[i, :3].reshape(3, gs, gs)[:, r0 : r0 + h, c0 : c0 + w].reshape(3, -1)
        for c in range(3):
            o = layout.off_gv + c * layout.rect_cells
            out[i, o : o + h * w] = vals[c]
    raise_on_violation(status)
    return out


def unpack_ref(packed: np.ndarray, layout: PackLayout) -> Tuple[np.ndarray, np.ndarray]:
    """``(local (N, 7, 31, 31), global (N, 4, 37, 37))`` from packed rows (NumPy reference)."""
    P = np.asarray(packed, dtype=np.uint8)
    N = len(P)
    L = np.zeros((N, LOCAL_SHAPE[0], N_LOCAL_PX), dtype=np.uint8)
    ttl = P[:, :N_LOCAL_PX]
    b = P[:, layout.off_b : layout.off_b + N_LOCAL_PX]
    c = P[:, layout.off_c : layout.off_h]
    nib = np.stack([c & 15, c >> 4], axis=2).reshape(N, -1)[:, :N_LOCAL_PX]
    own = (nib & 1).astype(bool)
    L[:, 0] = np.where(own, ttl, 0)
    L[:, 1] = np.where(own, 0, ttl)
    L[:, 3] = CODES[(b >> 5) & 3]
    L[:, 4] = np.where(b & 128, 255, 0)
    L[:, 5] = b & 31
    L[:, 6] = CODES[(nib >> 1) & 3]
    G = np.zeros((N,) + GLOBAL_SHAPE, dtype=np.uint8)
    for i in range(N):
        for k in range(int(P[i, layout.off_h])):
            q = layout.off_h + 1 + 3 * k
            L[i, 2, int(P[i, q]) | (int(P[i, q + 1]) << 8)] = P[i, q + 2]
        flag, r0, c0, h, w = (int(x) for x in P[i, layout.off_g : layout.off_g + 5])
        if flag == 0:
            continue
        G[i, 3] = 255
        G[i, 3, r0 : r0 + h, c0 : c0 + w] = 0
        for ch in range(3):
            o = layout.off_gv + ch * layout.rect_cells
            G[i, ch, r0 : r0 + h, c0 : c0 + w] = P[i, o : o + h * w].reshape(h, w)
    return L.reshape((N,) + LOCAL_SHAPE), G


# ----------------------------------------------------------------------------- numba kernels
try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a pinned core dependency
    njit = None

if njit is not None:

    @njit(cache=True, nogil=True)
    def _pack_kernel(
        local, glob, out, status, rect_cells, head_cap, off_b, off_c, off_h, off_g, off_gv
    ):
        n_l = local.shape[2]
        gs = 37
        n_g = glob.shape[2]
        for i in range(local.shape[0]):
            st = 0
            nh = 0
            for p in range(n_l):
                a = np.int64(local[i, 0, p])
                b = np.int64(local[i, 1, p])
                if a != 0 and b != 0:
                    st = 1
                    break
                f = np.int64(local[i, 3, p])
                if f == 0:
                    fc = 0
                elif f == 128:
                    fc = 1
                elif f == 255:
                    fc = 2
                else:
                    st = 2
                    break
                w = np.int64(local[i, 4, p])
                if w == 0:
                    wb = 0
                elif w == 255:
                    wb = 1
                else:
                    st = 3
                    break
                r = np.int64(local[i, 5, p])
                if r > 31:
                    st = 4
                    break
                e = np.int64(local[i, 6, p])
                if e == 0:
                    ec = 0
                elif e == 128:
                    ec = 1
                elif e == 255:
                    ec = 2
                else:
                    st = 5
                    break
                own = 1 if a != 0 else 0
                out[i, p] = a if a != 0 else b
                out[i, off_b + p] = r | (fc << 5) | (wb << 7)
                nib = own | (ec << 1)
                q = off_c + (p >> 1)
                if p & 1:
                    out[i, q] = np.int64(out[i, q]) | (nib << 4)
                else:
                    out[i, q] = np.int64(out[i, q]) | nib
                hv = local[i, 2, p]
                if hv != 0:
                    if nh >= head_cap:
                        st = 6
                        break
                    q = off_h + 1 + 3 * nh
                    out[i, q] = p & 255
                    out[i, q + 1] = p >> 8
                    out[i, q + 2] = hv
                    nh += 1
            if st != 0:
                status[i] = st
                continue
            out[i, off_h] = nh
            # global: all-zero row -> flag 0
            anynz = False
            for c in range(4):
                for p in range(n_g):
                    if glob[i, c, p] != 0:
                        anynz = True
                        break
                if anynz:
                    break
            if not anynz:
                continue
            rmin = gs
            rmax = -1
            cmin = gs
            cmax = -1
            cnt = 0
            for p in range(n_g):
                o = glob[i, 3, p]
                if o == 0:
                    rr = p // gs
                    cc = p - rr * gs
                    if rr < rmin:
                        rmin = rr
                    if rr > rmax:
                        rmax = rr
                    if cc < cmin:
                        cmin = cc
                    if cc > cmax:
                        cmax = cc
                    cnt += 1
                elif o != 255:
                    st = 7
                    break
            if st != 0:
                status[i] = st
                continue
            if cnt == 0:
                r0 = 0
                c0 = 0
                h = 0
                w_ = 0
            else:
                r0 = rmin
                c0 = cmin
                h = rmax - rmin + 1
                w_ = cmax - cmin + 1
            if cnt != h * w_:
                status[i] = 8
                continue
            for c in range(3):
                for p in range(n_g):
                    if glob[i, c, p] != 0:
                        rr = p // gs
                        cc = p - rr * gs
                        if rr < r0 or rr >= r0 + h or cc < c0 or cc >= c0 + w_:
                            st = 9
                            break
                if st != 0:
                    break
            if st != 0:
                status[i] = st
                continue
            if h * w_ > rect_cells:
                status[i] = 10
                continue
            out[i, off_g] = 1
            out[i, off_g + 1] = r0
            out[i, off_g + 2] = c0
            out[i, off_g + 3] = h
            out[i, off_g + 4] = w_
            for c in range(3):
                base = off_gv + c * rect_cells
                k = 0
                for rr in range(h):
                    row = (r0 + rr) * gs + c0
                    for cc in range(w_):
                        out[i, base + k] = glob[i, c, row + cc]
                        k += 1

    @njit(cache=True, nogil=True)
    def _unpack_kernel(
        src,
        src_rows,
        out_l,
        out_g,
        out_rows,
        codes,
        rect_cells,
        head_cap,
        off_b,
        off_c,
        off_h,
        off_g,
        off_gv,
    ):
        n_l = out_l.shape[2]
        n_g = out_g.shape[2]
        gs = 37
        for i in range(src_rows.shape[0]):
            s = src_rows[i]
            o = out_rows[i]
            for p in range(n_l):
                t = src[s, p]
                b = np.int64(src[s, off_b + p])
                nb = np.int64(src[s, off_c + (p >> 1)])
                if p & 1:
                    nb = nb >> 4
                if nb & 1:
                    out_l[o, 0, p] = t
                    out_l[o, 1, p] = 0
                else:
                    out_l[o, 0, p] = 0
                    out_l[o, 1, p] = t
                out_l[o, 2, p] = 0
                out_l[o, 3, p] = codes[(b >> 5) & 3]
                out_l[o, 4, p] = 255 if b & 128 else 0
                out_l[o, 5, p] = b & 31
                out_l[o, 6, p] = codes[(nb >> 1) & 3]
            nh = np.int64(src[s, off_h])
            for k in range(nh):
                q = off_h + 1 + 3 * k
                p = np.int64(src[s, q]) | (np.int64(src[s, q + 1]) << 8)
                out_l[o, 2, p] = src[s, q + 2]
            if src[s, off_g] == 0:
                for c in range(4):
                    for p in range(n_g):
                        out_g[o, c, p] = 0
                continue
            r0 = np.int64(src[s, off_g + 1])
            c0 = np.int64(src[s, off_g + 2])
            h = np.int64(src[s, off_g + 3])
            w_ = np.int64(src[s, off_g + 4])
            for c in range(3):
                for p in range(n_g):
                    out_g[o, c, p] = 0
            for p in range(n_g):
                out_g[o, 3, p] = 255
            k = 0
            for rr in range(h):
                row = (r0 + rr) * gs + c0
                for cc in range(w_):
                    p = row + cc
                    out_g[o, 3, p] = 0
                    out_g[o, 0, p] = src[s, off_gv + k]
                    out_g[o, 1, p] = src[s, off_gv + rect_cells + k]
                    out_g[o, 2, p] = src[s, off_gv + 2 * rect_cells + k]
                    k += 1


def _rows3(a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """``(n, C, H*W)`` view of a C-contiguous ``(n, C, H, W)`` block (never a copy)."""
    if not a.flags.c_contiguous or a.shape[1:] != shape or a.dtype != np.uint8:
        raise ValueError(f"need a C-contiguous uint8 (n,) + {shape} block")
    return a.reshape(a.shape[0], shape[0], -1)


def pack(local: np.ndarray, glob: np.ndarray, layout: PackLayout) -> np.ndarray:
    """``(N, layout.nbytes)`` packed rows (numba; raises ``ValueError`` on a violation)."""
    N = len(local)
    L = _rows3(np.ascontiguousarray(local, dtype=np.uint8), LOCAL_SHAPE)
    G = _rows3(np.ascontiguousarray(glob, dtype=np.uint8), GLOBAL_SHAPE)
    out = np.zeros((N, layout.nbytes), dtype=np.uint8)
    status = np.zeros(N, dtype=np.int64)
    if N:
        _pack_kernel(L, G, out, status, *layout.consts())
    raise_on_violation(status)
    return out


def unpack_into(
    src: np.ndarray,
    src_rows: np.ndarray,
    out_local: np.ndarray,
    out_global: np.ndarray,
    out_rows: np.ndarray,
    layout: PackLayout,
) -> None:
    """``out_local[out_rows[i]], out_global[out_rows[i]] = unpack(src[src_rows[i]])``.

    ``src`` is any ``(M, nbytes)`` packed array (a ring, a buffer, a read-only memmap);
    ``out_*`` are C-contiguous ``(n, 7, 31, 31)`` / ``(n, 4, 37, 37)`` blocks (e.g. the
    learner's staging slices), written in place."""
    src_rows = np.ascontiguousarray(src_rows, dtype=np.int64).reshape(-1)
    out_rows = np.ascontiguousarray(out_rows, dtype=np.int64).reshape(-1)
    if len(src_rows) != len(out_rows):
        raise ValueError("src_rows / out_rows length mismatch")
    if not len(src_rows):
        return
    if src.ndim != 2 or src.shape[1] != layout.nbytes or src.dtype != np.uint8:
        raise ValueError("src is not a packed (M, nbytes) uint8 array")
    L = _rows3(out_local, LOCAL_SHAPE)
    G = _rows3(out_global, GLOBAL_SHAPE)
    if src_rows.min() < 0 or src_rows.max() >= len(src):
        raise IndexError("src_rows out of range")
    if out_rows.min() < 0 or out_rows.max() >= len(L) or len(G) != len(L):
        raise IndexError("out_rows out of range")
    _unpack_kernel(src, src_rows, L, G, out_rows, CODES, *layout.consts())


def unpack(src: np.ndarray, rows: np.ndarray, layout: PackLayout) -> Tuple[np.ndarray, np.ndarray]:
    """``(local, global)`` for ``src[rows]`` as new ``(n, 7, 31, 31)`` / ``(n, 4, 37, 37)``."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    n = len(rows)
    L = np.empty((n,) + LOCAL_SHAPE, dtype=np.uint8)
    G = np.empty((n,) + GLOBAL_SHAPE, dtype=np.uint8)
    unpack_into(src, rows, L, G, np.arange(n, dtype=np.int64), layout)
    return L, G


def train_layout() -> PackLayout:
    """The layout of the training arena (the pinned evaluation profile's world)."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    world = gid._context(1)["profile"].world
    seg = int(world.segment_size)
    return layout_for_arena(world.width // seg, world.height // seg, world.num_snakes)
