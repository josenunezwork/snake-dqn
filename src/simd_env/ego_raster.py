"""Dual-scale ego raster built straight from occupancy grids (redesign M1).

Observation ``ego2s-draft`` for the redesign scope
(``docs/research/redesign_scope_2026-10-07.md``). Unlike ``raster31v2`` (which
re-paints every snake and pellet from coordinate lists per agent), everything
here is a gather from persistent per-env grids, so its cost is independent of
snake length and pellet count.

The featurizer reads an :class:`EgoGridView`, never a simulator directly, so the
same function serves two producers that must agree byte for byte:

- :meth:`GridBatchSim.ego_view <src.simd_env.grid_sim.GridBatchSim.ego_view>`
  (training / teacher data: views of the sim's own arrays, no copies);
- :func:`src.simd_env.ego_live_adapter.game_state_to_ego_view` (serving: a live
  ``GameState`` rasterised into the same padded grids).

Two backends compute the planes: ``"numpy"`` (the vectorized reference) and
``"numba"`` (:mod:`src.simd_env.ego_raster_nb`, compiled kernels). Both are
pinned bitwise equal in ``tests/test_ego_raster.py``; ``"auto"`` picks numba
when it imports.

Every spatial plane is heading-rotated ("facing up"); with four cardinal headings
the rotation is an exact index permutation (no interpolation).

Local plane stack ``(..., 6, 31, 31)`` uint8, cell = 1 segment, head fixed at
``(row 23, col 15)``:

====  =====================  ==============================================
ch    name                   value
====  =====================  ==============================================
0     own_ttl                frames until the own segment vacates (``L - k``),
                             clamped to 255; 0 = no own segment
1     enemy_ttl              same for every other living snake
2     enemy_head             ``clamp(L_enemy / L_own, 0, 2) / 2 * 254 + 1``
3     food                   128 ambient pellet, 255 corpse/trail pellet
4     wall                   255 outside the arena
5     reach_time             first arrival time (1..K) of a time-aware flood
                             from the head in which a body cell becomes
                             passable once its TTL has elapsed; 0 = unreached
                             within ``K`` steps (``reach_steps=0`` disables)
====  =====================  ==============================================

The reach flood is defined per agent: ``arrival(c) = max(min_n arrival(n) + 1,
ttl(c), 1)`` over 4-neighbours ``n`` inside the window, with the head at 0, walls
impassable, and arrivals above ``K`` left at 0. (The first draft stopped the
synchronous flood for the WHOLE batch on the first step at which no agent gained
a cell, which made an agent's plane depend on the other agents in its batch and
cut off cells that a vacating tail would open later; fixed for M1.)

Global plane stack ``(..., 4, 37, 37)`` uint8, cell = 8 x 8 segments (the full
145 x 83 arena is visible from any head: the window spans +-144 cells):

====  ==================  ===============================================
0     own_mass            own segments in the block (count, clamped 255)
1     enemy_mass          other snakes' segments in the block
2     food_mass           pellets in the block
3     outside             255 when the block lies outside the arena
====  ==================  ===============================================

Scalars ``(..., 12)`` float32: ``log1p(L)/log1p(4096)``, ``L/1000`` (uncapped,
unlike vector61's ``min(L/150, 1)``), ``(L - seg_count)/16`` (pending growth),
boost-available, ``boost_frames/boost_length_cost_frames``, ``hunger/1000``,
wall distance ahead/right/behind/left in cells / 145, alive fraction.

Dead rows observe all zeros. This is a prototype for throughput and design
review; it is not yet an obs spec registered in :mod:`src.model.obs_spec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.simd_env.batch_sim import CARDINAL
from src.simd_env.grid_sim import (
    FOOD_AMBIENT,
    FOOD_CORPSE,
    FOOD_OUTSIDE,
    GRID_PAD,
    padded_width,
)

__all__ = [
    "EgoRasterConfig",
    "EgoGridView",
    "LOCAL_CHANNELS",
    "GLOBAL_CHANNELS",
    "SCALAR_NAMES",
    "BACKENDS",
    "build_ego_raster",
    "numba_available",
]

LOCAL_CHANNELS = ("own_ttl", "enemy_ttl", "enemy_head", "food", "wall", "reach_time")
GLOBAL_CHANNELS = ("own_mass", "enemy_mass", "food_mass", "outside")
SCALAR_NAMES = (
    "log_length",
    "length_per_1000",
    "pending_growth",
    "boost_available",
    "boost_phase",
    "hunger",
    "wall_ahead",
    "wall_right",
    "wall_behind",
    "wall_left",
    "alive_fraction",
    "reserved",
)
BACKENDS = ("auto", "numpy", "numba")


@dataclass(frozen=True)
class EgoRasterConfig:
    """Geometry knobs for :func:`build_ego_raster`."""

    local_size: int = 31
    local_head_row: int = 23
    local_head_col: int = 15
    global_size: int = 37
    global_cell: int = 8
    reach_steps: int = 24
    min_boost_length: int = 5
    #: ``"draft"`` (``ego2s-draft``: 6 local channels, 12 scalars) or ``"b"`` (``ego2s-b``:
    #: + the ``enemy_next`` channel and 18 enemy / region scalars, see ego_raster_b).
    version: str = "draft"


@dataclass(frozen=True)
class EgoGridView:
    """Everything the featurizer reads, for ``E`` envs of ``S`` snakes.

    The padded grids carry a ``pad``-cell border (food border = ``FOOD_OUTSIDE``).
    ``slot`` holds the ring slot of the segment in a cell, so its offset from its
    owner's head is ``(head_ptr - slot) % cap``. Only living snakes own cells.
    """

    owner_pad: np.ndarray  # (E, H + 2P, W + 2P) int16, -1 empty
    slot_pad: np.ndarray  # (E, H + 2P, W + 2P) int32
    food_pad: np.ndarray  # (E, H + 2P, W + 2P) int8
    head_ptr: np.ndarray  # (E, S) int64
    seg_count: np.ndarray  # (E, S) int64
    length: np.ndarray  # (E, S) int64 (logical length)
    alive: np.ndarray  # (E, S) bool
    direction: np.ndarray  # (E, S) int64 heading 0..3 (up, right, down, left)
    heads: np.ndarray  # (E, S, 2) int64 (col, row); ignored for dead rows
    boost_frames: np.ndarray  # (E, S) int64
    frames_since_food: np.ndarray  # (E, S) int64
    grid_w: int
    grid_h: int
    cap: int
    pad: int
    boost_length_cost_frames: int
    # Optional maintained coarse block counts (GridBatchSim keeps them; the live adapter
    # does not). Used by the numba backend when ``coarse_cell == cfg.global_cell``;
    # the NumPy reference always recounts from the grids.
    coarse_snake: Optional[np.ndarray] = None  # (E, S, hc, wc) int32
    coarse_food: Optional[np.ndarray] = None  # (E, hc, wc) int32
    coarse_cell: int = 0
    # Enemy boosted on its last move (BatchSim.get_boosted_this_step / live
    # Snake.is_boosting); read by the ego2s-b features only.
    boosting: Optional[np.ndarray] = None  # (E, S) bool

    @property
    def E(self) -> int:
        return int(self.head_ptr.shape[0])

    @property
    def S(self) -> int:
        return int(self.head_ptr.shape[1])


@lru_cache(maxsize=16)
def _offset_table(size: int, head_row: int, head_col: int, row_stride: int) -> np.ndarray:
    """Flat index offsets of every ego pixel for each heading, shape ``(4, size*size)``.

    Pixel ``(r, c)`` lies ``head_row - r`` cells ahead and ``c - head_col`` cells
    to the right of the head; ``row_stride`` is the padded grid width.
    """
    r = np.arange(size)
    ahead = (head_row - r)[:, None]
    lateral = (r - head_col)[None, :]
    table = np.empty((4, size * size), dtype=np.int64)
    for h, (dx, dy) in enumerate(CARDINAL.tolist()):
        rx, ry = -dy, dx  # right-hand vector in screen space
        dcol = ahead * dx + lateral * rx
        drow = ahead * dy + lateral * ry
        table[h] = (drow * row_stride + dcol).ravel()
    table.setflags(write=False)
    return table


def _crop_index(
    heads_rc: np.ndarray,
    alive: np.ndarray,
    direction: np.ndarray,
    pad: int,
    row_stride: int,
    table: np.ndarray,
    centre: Tuple[int, int],
) -> np.ndarray:
    """``(E, S, n)`` flat indices into a padded grid; dead rows crop at ``centre``."""
    col = np.where(alive, heads_rc[..., 0], centre[0]) + pad
    row = np.where(alive, heads_rc[..., 1], centre[1]) + pad
    return (row * row_stride + col)[..., None] + table[direction]


def _take(grid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-env flat gather: ``grid`` ``(E, ...)`` with ``idx`` ``(E, S, n)``."""
    E = grid.shape[0]
    flat = grid.reshape(E, -1)
    return np.take_along_axis(flat, idx.reshape(E, -1), axis=1).reshape(idx.shape)


def _check_local_geometry(view: EgoGridView, cfg: EgoRasterConfig) -> None:
    size = cfg.local_size
    reach = max(
        cfg.local_head_row,
        size - 1 - cfg.local_head_row,
        cfg.local_head_col,
        size - 1 - cfg.local_head_col,
    )
    if reach > view.pad:
        raise ValueError("local crop reaches beyond the grid padding")


# ---------------------------------------------------------------------------
# NumPy reference backend
# ---------------------------------------------------------------------------
def _local_planes(view: EgoGridView, cfg: EgoRasterConfig) -> np.ndarray:
    E, S, cap = view.E, view.S, view.cap
    size = cfg.local_size
    _check_local_geometry(view, cfg)
    stride = view.owner_pad.shape[2]
    table = _offset_table(size, cfg.local_head_row, cfg.local_head_col, stride)
    centre = (view.grid_w // 2, view.grid_h // 2)
    idx = _crop_index(view.heads, view.alive, view.direction, view.pad, stride, table, centre)
    owner = _take(view.owner_pad, idx).astype(np.int32)  # (E, S, n)
    slot = _take(view.slot_pad, idx)
    food = _take(view.food_pad, idx)
    occ = owner >= 0
    o = np.where(occ, owner, 0).reshape(E, -1)
    hp = np.take_along_axis(view.head_ptr, o, axis=1).reshape(owner.shape)
    olen = np.take_along_axis(view.length, o, axis=1).reshape(owner.shape)
    k = (hp - slot) % cap
    ttl = np.clip(olen - k, 0, 255)
    own = occ & (owner == np.arange(S, dtype=np.int32)[None, :, None])
    enemy = occ & ~own
    n = size * size
    out = np.zeros((E, S, len(LOCAL_CHANNELS), n), dtype=np.uint8)
    out[:, :, 0] = np.where(own, ttl, 0)
    out[:, :, 1] = np.where(enemy, ttl, 0)
    own_len = np.maximum(view.length, 1)[:, :, None].astype(np.float64)
    ratio = np.clip(olen / own_len, 0.0, 2.0) / 2.0
    out[:, :, 2] = np.where(enemy & (k == 0), (ratio * 254.0).astype(np.uint8) + 1, 0)
    out[:, :, 3] = np.where(food == FOOD_CORPSE, 255, np.where(food == FOOD_AMBIENT, 128, 0))
    wall = food == FOOD_OUTSIDE
    out[:, :, 4] = np.where(wall, 255, 0)
    out = out.reshape(E, S, len(LOCAL_CHANNELS), size, size)
    if cfg.reach_steps > 0:
        blocked_ttl = np.where(occ, ttl, 0).reshape(E, S, size, size)
        out[:, :, 5] = _reach_time(blocked_ttl, wall.reshape(E, S, size, size), cfg)
    # Dead rows observe nothing.
    out[~view.alive] = 0
    return out


def _reach_time(ttl: np.ndarray, wall: np.ndarray, cfg: EgoRasterConfig) -> np.ndarray:
    """Time-aware 4-neighbour flood from the head pixel within the local window.

    A cell is enterable at step ``t`` when it is inside the arena and either
    empty or its occupant's TTL is ``<= t`` (the segment has vacated by then).
    Returns the first arrival step (1..K) per pixel, 0 when unreached. The loop
    stops early only when no agent can ever gain a cell again (no new cell and
    no blocked frontier cell that opens within ``K``), so each agent's result is
    independent of the batch it is computed in.
    """
    E, S = ttl.shape[:2]
    N = E * S
    size = cfg.local_size
    K = cfg.reach_steps
    ttl = ttl.reshape(N, size, size)
    wall = wall.reshape(N, size, size)
    reach = np.zeros((N, size, size), dtype=bool)
    reach[:, cfg.local_head_row, cfg.local_head_col] = True
    arrival = np.zeros((N, size, size), dtype=np.uint8)
    for t in range(1, K + 1):
        grow = np.zeros_like(reach)
        grow[:, 1:, :] |= reach[:, :-1, :]
        grow[:, :-1, :] |= reach[:, 1:, :]
        grow[:, :, 1:] |= reach[:, :, :-1]
        grow[:, :, :-1] |= reach[:, :, 1:]
        cand = grow & ~reach & ~wall
        new = cand & (ttl <= t)
        if not new.any() and not (cand & (ttl <= K)).any():
            break
        arrival[new] = t
        reach |= new
    return arrival.reshape(E, S, size, size)


def _coarse_counts(view: EgoGridView, cell: int):
    """Per-snake and food block counts at ``cell`` x ``cell`` resolution."""
    E, S, P = view.E, view.S, view.pad
    H, W = view.grid_h, view.grid_w
    owner_g = view.owner_pad[:, P : P + H, P : P + W]
    food_g = view.food_pad[:, P : P + H, P : P + W]
    hc = -(-H // cell)
    wc = -(-W // cell)
    rows = np.arange(H) // cell
    cols = np.arange(W) // cell
    block = (rows[:, None] * wc + cols[None, :]).ravel()  # (H*W,)
    nb = hc * wc
    own = owner_g.reshape(E, -1).astype(np.int64)
    occ = own >= 0
    e_idx = np.broadcast_to(np.arange(E)[:, None], own.shape)
    idx = (e_idx * S + np.clip(own, 0, None)) * nb + block[None, :]
    snake_counts = np.bincount(idx[occ], minlength=E * S * nb).reshape(E, S, hc, wc)
    fidx = e_idx * nb + block[None, :]
    has_food = food_g.reshape(E, -1) > 0
    food_counts = np.bincount(fidx[has_food], minlength=E * nb).reshape(E, hc, wc)
    return snake_counts, food_counts, hc, wc


def _global_planes(view: EgoGridView, cfg: EgoRasterConfig) -> np.ndarray:
    E, S = view.E, view.S
    size, cell = cfg.global_size, cfg.global_cell
    half = size // 2
    snake_counts, food_counts, hc, wc = _coarse_counts(view, cell)
    pw = ((0, 0), (0, 0), (half, half), (half, half))
    own_p = np.pad(np.minimum(snake_counts, 255).astype(np.int32), pw)
    total_p = np.pad(snake_counts.sum(axis=1), pw[1:])
    food_p = np.pad(food_counts, pw[1:])
    outside_p = np.pad(np.zeros((1, hc, wc), dtype=np.uint8), pw[1:], constant_values=255)
    stride = wc + 2 * half
    table = _offset_table(size, half, half, stride)
    heads_c = view.heads // cell
    idx = _crop_index(heads_c, view.alive, view.direction, half, stride, table, (wc // 2, hc // 2))
    own = np.take_along_axis(own_p.reshape(E * S, -1), idx.reshape(E * S, -1), axis=1)
    own = own.reshape(idx.shape)
    allc = _take(total_p, idx)
    out = np.zeros((E, S, len(GLOBAL_CHANNELS), size * size), dtype=np.uint8)
    out[:, :, 0] = own
    out[:, :, 1] = np.clip(allc - np.minimum(own, allc), 0, 255)
    out[:, :, 2] = np.clip(_take(food_p, idx), 0, 255)
    out[:, :, 3] = outside_p.reshape(-1)[idx]
    out = out.reshape(E, S, len(GLOBAL_CHANNELS), size, size)
    out[~view.alive] = 0
    return out


def _scalars(
    view: EgoGridView, cfg: EgoRasterConfig, rows: Optional[np.ndarray] = None
) -> np.ndarray:
    """The 12 scalars (NumPy for both backends: float math stays in one library).

    Elementwise per row, so ``rows`` (``(N, 2)``) gives exactly ``full[rows]``.
    """
    S = view.S
    if rows is None:
        pick = (slice(None), slice(None))
        shape: Tuple[int, ...] = (view.E, S)
    else:
        pick = (rows[:, 0], rows[:, 1])
        shape = (len(rows),)
    length = view.length[pick]
    L = length.astype(np.float64)
    alive = view.alive[pick]
    direction = view.direction[pick]
    heads = view.heads[pick]
    out = np.zeros(shape + (len(SCALAR_NAMES),), dtype=np.float32)
    out[..., 0] = np.log1p(L) / np.log1p(4096.0)
    out[..., 1] = L / 1000.0
    out[..., 2] = (length - view.seg_count[pick]) / 16.0
    out[..., 3] = length >= cfg.min_boost_length
    out[..., 4] = view.boost_frames[pick] / float(view.boost_length_cost_frames)
    out[..., 5] = view.frames_since_food[pick] / 1000.0
    # Free cells to the arena edge along each cardinal (up, right, down, left).
    edge = np.stack(
        [
            heads[..., 1],
            view.grid_w - 1 - heads[..., 0],
            view.grid_h - 1 - heads[..., 1],
            heads[..., 0],
        ],
        axis=-1,
    )
    for j in range(4):  # ahead, right, behind, left in the heading frame
        d = (direction + j) % 4
        out[..., 6 + j] = np.take_along_axis(edge, d[..., None], axis=-1)[..., 0] / 145.0
    frac = (view.alive.sum(axis=1, keepdims=True) / S).astype(np.float32)  # (E, 1)
    out[..., 10] = np.broadcast_to(frac, (view.E, S))[pick]
    out[~alive] = 0.0
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def numba_available() -> bool:
    """Whether the compiled backend imports (numba is a pinned redesign dependency)."""
    try:
        import src.simd_env.ego_raster_nb  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "auto":
        return "numba" if numba_available() else "numpy"
    return backend


def _as_view(source: Any) -> EgoGridView:
    if isinstance(source, EgoGridView):
        return source
    ego_view = getattr(source, "ego_view", None)
    if not callable(ego_view):
        raise TypeError("build_ego_raster needs an EgoGridView or a GridBatchSim")
    return ego_view()


def _normalize_rows(view: EgoGridView, rows: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if rows is None:
        return None
    rows = np.ascontiguousarray(np.asarray(rows, dtype=np.int64).reshape(-1, 2))
    if rows.size and (
        rows[:, 0].min() < 0
        or rows[:, 0].max() >= view.E
        or rows[:, 1].min() < 0
        or rows[:, 1].max() >= view.S
    ):
        raise ValueError("rows index outside the (E, S) batch")
    return rows


def build_ego_raster(
    source: Union[EgoGridView, Any],
    cfg: EgoRasterConfig = EgoRasterConfig(),
    *,
    backend: str = "auto",
    rows: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build local + global ego planes and scalars.

    Args:
        source: An :class:`EgoGridView`, or a ``GridBatchSim`` (its ``ego_view()``;
            its grids are read, never written).
        cfg: Geometry knobs.
        backend: ``"auto"`` (numba when available), ``"numpy"`` or ``"numba"``.
            Bitwise identical outputs.
        rows: Optional ``(N, 2)`` ``(env, snake)`` rows to featurize (e.g. hero
            slots only). Outputs are then ``(N, ...)`` instead of ``(E, S, ...)``.

    Returns:
        ``{"local": (...,6,31,31) uint8, "global": (...,4,37,37) uint8,
        "scalars": (...,12) float32}`` plus ``"mask"`` ``(..., 6)`` bool (the
        resolved action mask) when ``source`` is a simulator.
    """
    view = _as_view(source)
    picked = _normalize_rows(view, rows)
    which = _resolve_backend(backend)
    scalars = _scalars(view, cfg, picked)
    if which == "numba":
        from src.simd_env.ego_raster_nb import build_planes_numba

        _check_local_geometry(view, cfg)
        local, glob = build_planes_numba(view, cfg, picked)
        if picked is None:
            local = local.reshape((view.E, view.S) + local.shape[1:])
            glob = glob.reshape((view.E, view.S) + glob.shape[1:])
    else:
        local = _local_planes(view, cfg)
        glob = _global_planes(view, cfg)
        if picked is not None:
            local = local[picked[:, 0], picked[:, 1]]
            glob = glob[picked[:, 0], picked[:, 1]]
    if cfg.version == "b":
        from src.simd_env import ego_raster_b as eb

        lead = local.shape[:-3]
        flat = local.reshape((-1,) + local.shape[-3:])
        extra = eb.enemy_next_plane(view, cfg, picked)
        local = np.concatenate([flat, extra[:, None]], axis=1).reshape(
            lead + (flat.shape[1] + 1,) + flat.shape[2:]
        )
        bs = eb.b_scalars(view, picked, which)
        scalars = np.concatenate([scalars.reshape(-1, scalars.shape[-1]), bs], axis=1).reshape(
            scalars.shape[:-1] + (scalars.shape[-1] + bs.shape[-1],)
        )
    elif cfg.version != "draft":
        raise ValueError(f"unknown ego raster version {cfg.version!r}")
    out = {
        "local": local,
        "global": glob,
        "scalars": scalars,
    }
    if not isinstance(source, EgoGridView):
        mask = source.get_resolved_action_mask()
        out["mask"] = mask if picked is None else mask[picked[:, 0], picked[:, 1]]
    return out


def empty_view_like(
    E: int, S: int, grid_w: int, grid_h: int, *, pad: int = GRID_PAD
) -> Dict[str, np.ndarray]:
    """Freshly allocated padded grids for a producer that fills them (live adapter)."""
    shape = (E, grid_h + 2 * pad, padded_width(grid_w, pad))
    owner = np.full(shape, -1, dtype=np.int16)
    slot = np.zeros(shape, dtype=np.int32)
    food = np.full(shape, FOOD_OUTSIDE, dtype=np.int8)
    food[:, pad : pad + grid_h, pad : pad + grid_w] = 0
    return {"owner_pad": owner, "slot_pad": slot, "food_pad": food}
