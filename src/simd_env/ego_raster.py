"""Dual-scale ego raster built straight from GridBatchSim grids (redesign M1 draft).

Draft observation ``ego2s-draft`` for the redesign scope
(``docs/research/redesign_scope_2026-10-07.md``). Unlike ``raster31v2`` (which
re-paints every snake and pellet from coordinate lists per agent), everything
here is a gather from the simulator's persistent per-env grids, so its cost is
independent of snake length and pellet count.

Every spatial plane is heading-rotated ("facing up"); with four cardinal headings
the rotation is an exact index permutation (no interpolation).

Local plane stack ``(E, S, 6, 31, 31)`` uint8, cell = 1 segment, head fixed at
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

Global plane stack ``(E, S, 4, 37, 37)`` uint8, cell = 8 x 8 segments (the full
145 x 83 arena is visible from any head: the window spans +-144 cells):

====  ==================  ===============================================
0     own_mass            own segments in the block (count, clamped 255)
1     enemy_mass          other snakes' segments in the block
2     food_mass           pellets in the block
3     outside             255 when the block lies outside the arena
====  ==================  ===============================================

Scalars ``(E, S, 12)`` float32: ``log1p(L)/log1p(4096)``, ``L/1000`` (uncapped,
unlike vector61's ``min(L/150, 1)``), ``(L - seg_count)/16`` (pending growth),
boost-available, ``boost_frames/3``, ``hunger/1000``, wall distance
ahead/right/behind/left in cells / 145, alive fraction.

This is a prototype for throughput and design review; it is not an obs spec
registered in :mod:`src.model.obs_spec` and nothing trains on it yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from src.simd_env.batch_sim import CARDINAL
from src.simd_env.grid_sim import (
    FOOD_AMBIENT,
    FOOD_CORPSE,
    FOOD_OUTSIDE,
    GRID_PAD,
    GridBatchSim,
)

__all__ = [
    "EgoRasterConfig",
    "LOCAL_CHANNELS",
    "GLOBAL_CHANNELS",
    "SCALAR_NAMES",
    "build_ego_raster",
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


def _local_planes(sim: GridBatchSim, cfg: EgoRasterConfig) -> np.ndarray:
    E, S, cap = sim.E, sim.S, sim.cap
    size = cfg.local_size
    if (
        max(
            cfg.local_head_row,
            size - 1 - cfg.local_head_row,
            cfg.local_head_col,
            size - 1 - cfg.local_head_col,
        )
        > GRID_PAD
    ):
        raise ValueError("local crop reaches beyond the grid padding")
    owner_p, slot_p, food_p = sim.get_padded_grids()
    stride = owner_p.shape[2]
    table = _offset_table(size, cfg.local_head_row, cfg.local_head_col, stride)
    centre = (sim.grid_w // 2, sim.grid_h // 2)
    idx = _crop_index(sim.heads(), sim.alive, sim.direction, GRID_PAD, stride, table, centre)
    owner = _take(owner_p, idx).astype(np.int32)  # (E, S, n)
    slot = _take(slot_p, idx)
    food = _take(food_p, idx)
    occ = owner >= 0
    o = np.where(occ, owner, 0).reshape(E, -1)
    hp = np.take_along_axis(sim.head_ptr, o, axis=1).reshape(owner.shape)
    olen = np.take_along_axis(sim.length, o, axis=1).reshape(owner.shape)
    k = (hp - slot) % cap
    ttl = np.clip(olen - k, 0, 255)
    own = occ & (owner == np.arange(S, dtype=np.int32)[None, :, None])
    enemy = occ & ~own
    n = size * size
    out = np.zeros((E, S, len(LOCAL_CHANNELS), n), dtype=np.uint8)
    out[:, :, 0] = np.where(own, ttl, 0)
    out[:, :, 1] = np.where(enemy, ttl, 0)
    own_len = np.maximum(sim.length, 1)[:, :, None].astype(np.float64)
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
    out[~sim.alive] = 0
    return out


def _reach_time(ttl: np.ndarray, wall: np.ndarray, cfg: EgoRasterConfig) -> np.ndarray:
    """Time-aware 4-neighbour flood from the head pixel within the local window.

    A cell is enterable at step ``t`` when it is inside the arena and either
    empty or its occupant's TTL is ``<= t`` (the segment has vacated by then).
    Returns the first arrival step (1..K) per pixel, 0 when unreached.
    """
    E, S = ttl.shape[:2]
    N = E * S
    size = cfg.local_size
    ttl = ttl.reshape(N, size, size)
    wall = wall.reshape(N, size, size)
    reach = np.zeros((N, size, size), dtype=bool)
    reach[:, cfg.local_head_row, cfg.local_head_col] = True
    arrival = np.zeros((N, size, size), dtype=np.uint8)
    for t in range(1, cfg.reach_steps + 1):
        grow = np.zeros_like(reach)
        grow[:, 1:, :] |= reach[:, :-1, :]
        grow[:, :-1, :] |= reach[:, 1:, :]
        grow[:, :, 1:] |= reach[:, :, :-1]
        grow[:, :, :-1] |= reach[:, :, 1:]
        new = grow & ~reach & ~wall & (ttl <= t)
        if not new.any():
            break
        arrival[new] = t
        reach |= new
    return arrival.reshape(E, S, size, size)


def _coarse_counts(sim: GridBatchSim, cell: int):
    """Per-snake and food block counts at ``cell`` x ``cell`` resolution."""
    E, S = sim.E, sim.S
    owner_g, _, food_g = sim.get_grids()
    hc = -(-sim.grid_h // cell)
    wc = -(-sim.grid_w // cell)
    rows = np.arange(sim.grid_h) // cell
    cols = np.arange(sim.grid_w) // cell
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


def _global_planes(sim: GridBatchSim, cfg: EgoRasterConfig) -> np.ndarray:
    E, S = sim.E, sim.S
    size, cell = cfg.global_size, cfg.global_cell
    half = size // 2
    snake_counts, food_counts, hc, wc = _coarse_counts(sim, cell)
    pw = ((0, 0), (0, 0), (half, half), (half, half))
    own_p = np.pad(np.minimum(snake_counts, 255).astype(np.int32), pw)
    total_p = np.pad(snake_counts.sum(axis=1), pw[1:])
    food_p = np.pad(food_counts, pw[1:])
    outside_p = np.pad(np.zeros((1, hc, wc), dtype=np.uint8), pw[1:], constant_values=255)
    stride = wc + 2 * half
    table = _offset_table(size, half, half, stride)
    heads_c = sim.heads() // cell
    idx = _crop_index(heads_c, sim.alive, sim.direction, half, stride, table, (wc // 2, hc // 2))
    own = np.take_along_axis(own_p.reshape(E * S, -1), idx.reshape(E * S, -1), axis=1)
    own = own.reshape(idx.shape)
    allc = _take(total_p, idx)
    out = np.zeros((E, S, len(GLOBAL_CHANNELS), size * size), dtype=np.uint8)
    out[:, :, 0] = own
    out[:, :, 1] = np.clip(allc - np.minimum(own, allc), 0, 255)
    out[:, :, 2] = np.clip(_take(food_p, idx), 0, 255)
    out[:, :, 3] = outside_p.reshape(-1)[idx]
    out = out.reshape(E, S, len(GLOBAL_CHANNELS), size, size)
    out[~sim.alive] = 0
    return out


def _scalars(sim: GridBatchSim, cfg: EgoRasterConfig) -> np.ndarray:
    E, S = sim.E, sim.S
    L = sim.length.astype(np.float64)
    out = np.zeros((E, S, len(SCALAR_NAMES)), dtype=np.float32)
    out[:, :, 0] = np.log1p(L) / np.log1p(4096.0)
    out[:, :, 1] = L / 1000.0
    out[:, :, 2] = (sim.length - sim.seg_count) / 16.0
    out[:, :, 3] = sim.length >= cfg.min_boost_length
    out[:, :, 4] = sim.boost_frames / float(sim.cfg.boost_length_cost_frames)
    out[:, :, 5] = sim.frames_since_food / 1000.0
    heads = sim.heads()
    # Distance in cells to the first out-of-arena cell along each ego direction.
    # Free cells to the arena edge along each cardinal (up, right, down, left).
    edge = np.stack(
        [
            heads[..., 1],
            sim.grid_w - 1 - heads[..., 0],
            sim.grid_h - 1 - heads[..., 1],
            heads[..., 0],
        ],
        axis=-1,
    )
    for j in range(4):  # ahead, right, behind, left in the heading frame
        d = (sim.direction + j) % 4
        out[:, :, 6 + j] = np.take_along_axis(edge, d[..., None], axis=-1)[..., 0] / 145.0
    out[:, :, 10] = (sim.alive.sum(axis=1, keepdims=True) / S).astype(np.float32)
    out[~sim.alive] = 0.0
    return out


def build_ego_raster(sim: GridBatchSim, cfg: EgoRasterConfig = EgoRasterConfig()) -> Dict:
    """Build local + global ego planes and scalars for every ``(env, snake)``.

    Args:
        sim: A :class:`GridBatchSim` (its grids are read, never written).
        cfg: Geometry knobs.

    Returns:
        ``{"local": (E,S,6,31,31) uint8, "global": (E,S,4,37,37) uint8,
        "scalars": (E,S,12) float32, "mask": (E,S,6) bool}``.
    """
    return {
        "local": _local_planes(sim, cfg),
        "global": _global_planes(sim, cfg),
        "scalars": _scalars(sim, cfg),
        "mask": sim.get_resolved_action_mask(),
    }
