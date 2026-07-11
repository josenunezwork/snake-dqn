"""Dual-scale ego-raster featurizer (blueprint §2, obs_spec ``raster31v2``).

One featurizer, two producers (blueprint §0.10c): the CORE functions here take a
:class:`ObsInputs` dataclass of plain NumPy arrays and know nothing about
:class:`~src.simd_env.batch_sim.BatchSim` internals or the live game. Both
:func:`obs_inputs_from_batch_sim` (below) and the live web path
(:mod:`src.simd_env.live_adapter`) fill an :class:`ObsInputs`, so serve-time
observations are train-time observations by construction.

Everything spatial is rendered in the snake's HEADING frame ("always facing
up"). With 4 cardinal headings this is an exact ``rot90`` — no interpolation.
The tactical head is fixed forward-biased at ``(row 23, col 15)``; the strategic
head is centered at ``(row 12, col 12)``.

Outputs (fully vectorized over ``E`` envs x ``S`` snakes):

- ``tactical_uint8``  ``(E, S, 2, 31, 31)`` uint8 — plane 0 = type code, plane 1
  = value byte. Expand to 9 float channels with :func:`expand_tactical`.
- ``strategic_uint8`` ``(E, S, 3, 25, 25)`` uint8 — three density channels.
- ``scalars``        ``(E, S, 26)`` float32 — ego-frame, Markov-completing.
- ``mask``           ``(E, S, 6)`` — per-agent safe-action mask passed through.

See :data:`TACTICAL_CHANNELS` / :data:`STRATEGIC_CHANNELS` for channel meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# --- Raster geometry (blueprint §2.1 / §2.2) ---------------------------------
TACTICAL_SIZE = 31
TACTICAL_HEAD_ROW = 23  # forward-biased: ~230px ahead, ~70px behind
TACTICAL_HEAD_COL = 15
STRATEGIC_SIZE = 25
STRATEGIC_CELL = 5  # segments per strategic cell (~50px)
STRATEGIC_HEAD_ROW = 12  # centered
STRATEGIC_HEAD_COL = 12

# --- Tactical type codes (overlap priority: heads > bodies > predicted >
#     corpse food > ambient food > empty). Higher code wins a contested cell. ---
CODE_EMPTY = 0
CODE_WALL = 1
CODE_AMBIENT_FOOD = 2
CODE_CORPSE_FOOD = 3
CODE_ENEMY_PRED = 4
CODE_OWN_BODY = 5
CODE_ENEMY_BODY = 6
CODE_ENEMY_HEAD = 7
CODE_OWN_HEAD = 8

# Channel layout for the expanded tactical raster (blueprint §2.1).
TACTICAL_CHANNELS = 9
TACTICAL_CHANNEL_NAMES = (
    "own_body",  # 0 value = TTL (tail low -> head high)
    "enemy_body",  # 1 value = TTL
    "enemy_head",  # 2 value = clamp(their_len/own_len, 0, 2)/2
    "enemy_pred",  # 3 predicted-next cells (2 when boosting)
    "ambient_food",  # 4 pellet mass
    "corpse_food",  # 5 corpse/boost-trail pellet mass
    "wall",  # 6 out-of-arena mask
    "own_head",  # 7 value = boost-engaged bit
    "reserved",  # 8 zeros (ablation slot)
)

# Map each type code -> tactical channel it expands into (CODE_EMPTY -> none).
_CODE_TO_CHANNEL = {
    CODE_OWN_BODY: 0,
    CODE_ENEMY_BODY: 1,
    CODE_ENEMY_HEAD: 2,
    CODE_ENEMY_PRED: 3,
    CODE_AMBIENT_FOOD: 4,
    CODE_CORPSE_FOOD: 5,
    CODE_WALL: 6,
    CODE_OWN_HEAD: 7,
}

STRATEGIC_CHANNELS = 3
STRATEGIC_CHANNEL_NAMES = (
    "enemy_mass_density",  # 0 bodies + heads, mass-weighted
    "food_mass_density",  # 1 ambient + corpse
    "own_body_density",  # 2
)

SCALARS_DIM = 26

# Wall-distance normalization (blueprint §2.3: "cells/64 capped").
_WALL_DIST_NORM = 64.0
# Food-value byte scale: pellet mass 1 -> full byte. Corpse pellets share it.
_FOOD_VALUE = 255


@dataclass
class ObsInputs:
    """Producer-agnostic world snapshot the featurizer consumes.

    All arrays are plain NumPy (no torch, no BatchSim / GameState references) so
    the core featurizer stays independent of either producer. ``E`` is the batch
    of environments, ``S`` the snakes per env, ``MAXLEN`` a padded body capacity.
    Coordinates are integer cell indices ``(col, row)`` on the shared lattice.

    Attributes:
        heads: ``(E, S, 2)`` head cell ``(col, row)``.
        bodies: ``(E, S, MAXLEN, 2)`` body cells head->tail; slots beyond
            ``body_len`` are ignored (may be any value).
        body_len: ``(E, S)`` number of valid body segments (>=1 for alive).
        lengths: ``(E, S)`` logical length.
        alive: ``(E, S)`` bool alive mask.
        heading: ``(E, S)`` cardinal heading index (0=up, 1=right, 2=down,
            3=left) matching :data:`src.simd_env.batch_sim.CARDINAL`.
        boost_frames: ``(E, S)`` boost burn-cadence counter.
        frames_since_food: ``(E, S)`` hunger counter.
        boosting: ``(E, S)`` bool — snake engaged boost this step (own-head bit
            and 2-cell enemy prediction).
        food_cells: ``(E, F, 2)`` padded food cell ``(col, row)`` per env.
        food_mass: ``(E, F)`` pellet mass (0 marks a padding slot).
        food_is_corpse: ``(E, F)`` bool corpse-class flag.
        grid_w: arena width in cells.
        grid_h: arena height in cells.
        max_snakes: nominal snakes-per-env for ``alive_count`` normalization.
        starvation_max: starvation frame cap for the hunger scalar.
        max_length: length cap for the length scalars.
        min_boost_length: minimum length to boost (``boost_available`` scalar).
        boost_cost_frames: boost burn cadence (``boost_cost_phase`` scalar).
        frame: ``(E,)`` current frame per env (episode-progress numerator).
        max_frames: episode length cap (episode-progress denominator).
        arena_type_flag: 0.0 rectangular, 1.0 circular (scalar flag).
    """

    heads: np.ndarray
    bodies: np.ndarray
    body_len: np.ndarray
    lengths: np.ndarray
    alive: np.ndarray
    heading: np.ndarray
    boost_frames: np.ndarray
    frames_since_food: np.ndarray
    boosting: np.ndarray
    food_cells: np.ndarray
    food_mass: np.ndarray
    food_is_corpse: np.ndarray
    grid_w: int
    grid_h: int
    max_snakes: int
    starvation_max: int
    max_length: int
    min_boost_length: int
    boost_cost_frames: int
    frame: np.ndarray
    max_frames: int
    arena_type_flag: float = 0.0

    @property
    def E(self) -> int:  # noqa: N802 - matches BatchSim naming
        """Number of environments."""
        return int(self.heads.shape[0])

    @property
    def S(self) -> int:  # noqa: N802 - matches BatchSim naming
        """Number of snakes per environment."""
        return int(self.heads.shape[1])


# ---------------------------------------------------------------------------
# Heading rotation
# ---------------------------------------------------------------------------
def _ego_offsets(
    dcol: np.ndarray, drow: np.ndarray, heading: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    """Rotate world offsets ``(dcol, drow)`` into the heading ("up") frame.

    Returns ``(ahead, lateral)``: ``ahead`` counts cells in the snake's forward
    direction (positive = ahead), ``lateral`` counts cells to the snake's right
    (positive = right). With four cardinal headings this is an exact rotation.

    ``heading`` broadcasts against ``dcol``/``drow``. Heading indices follow
    :data:`src.simd_env.batch_sim.CARDINAL` (0=up, 1=right, 2=down, 3=left);
    world +row is down, +col is right.

    Args:
        dcol: World column offset from the head.
        drow: World row offset from the head.
        heading: Cardinal heading index, broadcastable to ``dcol``/``drow``.

    Returns:
        ``(ahead, lateral)`` integer offset arrays in the heading frame.
    """
    h = heading
    # forward unit vector f = CARDINAL[h]; right = f rotated 90 deg clockwise.
    # ahead = -(offset . f_row_down) ... derive per-heading closed form:
    #   up    (0): forward=-row, right=+col  -> ahead=-drow, lateral=+dcol
    #   right (1): forward=+col, right=+row  -> ahead=+dcol, lateral=+drow
    #   down  (2): forward=+row, right=-col  -> ahead=+drow, lateral=-dcol
    #   left  (3): forward=-col, right=-row  -> ahead=-dcol, lateral=-drow
    ahead = np.select(
        [h == 0, h == 1, h == 2, h == 3],
        [-drow, dcol, drow, -dcol],
    )
    lateral = np.select(
        [h == 0, h == 1, h == 2, h == 3],
        [dcol, drow, -dcol, -drow],
    )
    return ahead, lateral


# ---------------------------------------------------------------------------
# Core build
# ---------------------------------------------------------------------------
def build_observations(inp: ObsInputs, mask: Optional[np.ndarray] = None) -> dict:
    """Build the full dual-scale ego observation for every ``(E, S)`` agent.

    Fully vectorized over the ``(E, S)`` agent grid. Everything spatial is
    rendered in the heading frame (exact ``rot90``). Channel semantics follow
    blueprint §2 exactly.

    Args:
        inp: Producer-filled :class:`ObsInputs` snapshot.
        mask: Optional ``(E, S, 6)`` safe-action mask to pass through. When None,
            an all-True mask is emitted (the featurizer does not compute masks).

    Returns:
        Dict with keys ``tactical_uint8`` ``(E, S, 2, 31, 31)`` uint8,
        ``strategic_uint8`` ``(E, S, 3, 25, 25)`` uint8, ``scalars``
        ``(E, S, 26)`` float32, and ``mask`` ``(E, S, 6)``.
    """
    E, S = inp.E, inp.S
    tactical = _build_tactical(inp)
    strategic = _build_strategic(inp)
    scalars = _build_scalars(inp)

    if mask is None:
        out_mask = np.ones((E, S, 6), dtype=bool)
    else:
        out_mask = np.asarray(mask)

    return {
        "tactical_uint8": tactical,
        "strategic_uint8": strategic,
        "scalars": scalars,
        "mask": out_mask,
    }


def _paint(
    code_plane: np.ndarray,
    val_plane: np.ndarray,
    ei: np.ndarray,
    si: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    code: np.ndarray,
    value: np.ndarray,
    size: int,
) -> None:
    """Priority-scatter ``(code, value)`` into ego rasters at ``(row, col)``.

    Cells inside the ``size x size`` raster whose incoming ``code`` beats the
    resident code overwrite both planes (higher code wins per the overlap
    priority). Sorting by code ascending and using last-write-wins scatter makes
    the higher-priority code land last for any contested cell.

    Args:
        code_plane: ``(E, S, size, size)`` uint16 type-code accumulator.
        val_plane: ``(E, S, size, size)`` uint16 value accumulator.
        ei, si: Flat env/snake indices for each point.
        row, col: Flat ego raster row/col for each point.
        code: Flat type code for each point.
        value: Flat value byte for each point.
        size: Raster edge length.
    """
    inb = (row >= 0) & (row < size) & (col >= 0) & (col < size)
    if not np.any(inb):
        return
    ei, si, row, col, code, value = (
        ei[inb],
        si[inb],
        row[inb],
        col[inb],
        code[inb],
        value[inb],
    )
    # Stable sort by code ascending so highest-priority code writes last.
    order = np.argsort(code, kind="stable")
    ei, si, row, col, code, value = (
        ei[order],
        si[order],
        row[order],
        col[order],
        code[order],
        value[order],
    )
    # Guard the value write: a cell contested by a lower code must not keep the
    # lower code's value. Because we sort ascending and scatter last-wins, the
    # final code and value written for a contested cell are the highest code's.
    code_plane[ei, si, row, col] = code
    val_plane[ei, si, row, col] = value


def _build_tactical(inp: ObsInputs) -> np.ndarray:
    """Render the 31x31 tactical rasters as 2 uint8 planes ``(E, S, 2, 31, 31)``."""
    E, S = inp.E, inp.S
    size = TACTICAL_SIZE
    # uint16 accumulators (code + value) then downcast; codes fit in a byte.
    code_plane = np.zeros((E, S, size, size), dtype=np.uint16)
    val_plane = np.zeros((E, S, size, size), dtype=np.uint16)

    heads = inp.heads  # (E, S, 2)
    heading = inp.heading  # (E, S)
    alive = inp.alive
    lengths = np.maximum(inp.lengths, 1)

    ei_grid = np.repeat(np.arange(E), S)
    si_grid = np.tile(np.arange(S), E)
    head_col = heads[..., 0].reshape(-1)
    head_row = heads[..., 1].reshape(-1)
    heading_flat = heading.reshape(-1)
    alive_flat = alive.reshape(-1)

    def to_raster(dcol, drow, head_idx):
        ahead, lateral = _ego_offsets(dcol, drow, heading_flat[head_idx])
        row = TACTICAL_HEAD_ROW - ahead
        col = TACTICAL_HEAD_COL + lateral
        return row, col

    # --- Wall / out-of-arena (ch 6): paint every ego cell whose world cell is
    #     outside the arena. Build the full ego grid of offsets once. ---
    rr, cc = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    # ego cell (r, c): ahead = HEAD_ROW - r, lateral = c - HEAD_COL.
    ego_ahead = (TACTICAL_HEAD_ROW - rr).reshape(-1)  # (size*size,)
    ego_lat = (cc - TACTICAL_HEAD_COL).reshape(-1)
    # Invert ego->world per heading: given (ahead, lateral) recover (dcol, drow).
    #   up:    dcol=lat,  drow=-ahead
    #   right: dcol=ahead,drow=lat
    #   down:  dcol=-lat, drow=ahead
    #   left:  dcol=-ahead,drow=-lat
    ncell = size * size
    n_agents = E * S
    # Broadcast: (n_agents, ncell)
    h_col = heading_flat[:, None]
    a = ego_ahead[None, :]
    lat = ego_lat[None, :]
    dcol_w = np.select(
        [h_col == 0, h_col == 1, h_col == 2, h_col == 3],
        [np.broadcast_to(lat, (n_agents, ncell)), a, -lat, -a],
    )
    drow_w = np.select(
        [h_col == 0, h_col == 1, h_col == 2, h_col == 3],
        [-a, lat, a, -lat],
    )
    world_col = head_col[:, None] + dcol_w
    world_row = head_row[:, None] + drow_w
    outside = (
        (world_col < 0) | (world_col >= inp.grid_w) | (world_row < 0) | (world_row >= inp.grid_h)
    )
    # Scatter wall code where outside AND agent alive.
    wall_pts = outside & alive_flat[:, None]
    if np.any(wall_pts):
        ai, ci = np.nonzero(wall_pts)
        _paint(
            code_plane,
            val_plane,
            ei_grid[ai],
            si_grid[ai],
            rr.reshape(-1)[ci],  # ego row
            cc.reshape(-1)[ci],  # ego col
            np.full(ai.shape, CODE_WALL, dtype=np.uint16),
            np.full(ai.shape, _FOOD_VALUE, dtype=np.uint16),
            size,
        )

    # --- Food (ambient ch4 / corpse ch5) ---
    _paint_food_tactical(inp, code_plane, val_plane, heads, heading, alive, size)

    # --- Snakes: own body / own head / enemy body / enemy head / enemy pred ---
    _paint_snakes_tactical(inp, code_plane, val_plane, heads, heading, alive, lengths, size)

    out = np.zeros((E, S, 2, size, size), dtype=np.uint8)
    out[:, :, 0] = code_plane.astype(np.uint8)
    out[:, :, 1] = np.minimum(val_plane, 255).astype(np.uint8)
    return out


def _paint_food_tactical(inp, code_plane, val_plane, heads, heading, alive, size):
    """Scatter ambient/corpse food pellets into the tactical rasters."""
    E, S = inp.E, inp.S
    food_cells = inp.food_cells  # (E, F, 2)
    food_mass = inp.food_mass  # (E, F)
    food_corpse = inp.food_is_corpse  # (E, F)
    F = food_cells.shape[1]
    if F == 0:
        return
    # For each agent (E,S), each pellet in its env: offset from that agent head.
    # Broadcast agent heads (E,S,1) vs env food (E,1,F).
    fc = food_cells[:, None, :, 0]  # (E,1,F) col
    fr = food_cells[:, None, :, 1]  # (E,1,F) row
    hc = heads[..., 0:1]  # (E,S,1)
    hr = heads[..., 1:2]
    dcol = fc - hc  # (E,S,F)
    drow = fr - hr
    head_idx_es = heading[..., None]  # (E,S,1)
    ahead, lateral = _ego_offsets(dcol, drow, head_idx_es)
    row = TACTICAL_HEAD_ROW - ahead
    col = TACTICAL_HEAD_COL + lateral
    mass = np.broadcast_to(food_mass[:, None, :], (E, S, F))
    corpse = np.broadcast_to(food_corpse[:, None, :], (E, S, F))
    valid = (mass > 0) & alive[..., None]
    ei = np.broadcast_to(np.arange(E)[:, None, None], (E, S, F))
    si = np.broadcast_to(np.arange(S)[None, :, None], (E, S, F))
    sel = valid
    if not np.any(sel):
        return
    code = np.where(corpse, CODE_CORPSE_FOOD, CODE_AMBIENT_FOOD).astype(np.uint16)
    value = np.minimum(mass.astype(np.uint16) * _FOOD_VALUE, 255).astype(np.uint16)
    _paint(
        code_plane,
        val_plane,
        ei[sel],
        si[sel],
        row[sel],
        col[sel],
        code[sel],
        value[sel],
        size,
    )


def _paint_snakes_tactical(inp, code_plane, val_plane, heads, heading, alive, lengths, size):
    """Scatter own/enemy bodies, heads and enemy predicted cells (tactical)."""
    E, S = inp.E, inp.S
    bodies = inp.bodies  # (E, S, MAXLEN, 2)
    body_len = inp.body_len  # (E, S)
    MAXLEN = bodies.shape[2]

    # For every observer agent o and every source snake s in the same env, paint
    # source s's segments into o's ego raster. Iterate over source snake index s
    # (S is small, e.g. 12) to keep memory bounded; within each we vectorize over
    # observers and segments.
    seg_k = np.arange(MAXLEN)  # offset from head (0 = head)

    for src in range(S):
        src_len = np.maximum(inp.lengths[:, src], 1)  # (E,)
        src_alive = alive[:, src]  # (E,)
        src_bodylen = body_len[:, src]  # (E,)
        src_heading = heading[:, src]  # (E,)
        src_boost = inp.boosting[:, src]  # (E,)

        # Segment cells for src: (E, MAXLEN, 2)
        seg = bodies[:, src]  # (E, MAXLEN, 2)
        seg_valid = (seg_k[None, :] < src_bodylen[:, None]) & src_alive[:, None]  # (E,MAXLEN)
        # TTL value: tail low -> head high. head (k=0) high, tail (k=len-1) low.
        # ttl = (bodylen - k) / bodylen, scaled to byte.
        with np.errstate(divide="ignore", invalid="ignore"):
            ttl = (src_bodylen[:, None] - seg_k[None, :]) / np.maximum(src_bodylen[:, None], 1)
        ttl_byte = np.clip(np.round(ttl * 255), 0, 255).astype(np.uint16)  # (E,MAXLEN)

        # Broadcast to observers: (E, S_obs, MAXLEN)
        segc = seg[:, None, :, 0]  # (E,1,MAXLEN)
        segr = seg[:, None, :, 1]
        hc = heads[..., 0:1]  # (E,S,1)
        hr = heads[..., 1:2]
        dcol = segc - hc  # (E,S,MAXLEN)
        drow = segr - hr
        ahead, lateral = _ego_offsets(dcol, drow, heading[..., None])
        row = TACTICAL_HEAD_ROW - ahead
        col = TACTICAL_HEAD_COL + lateral

        obs_alive = alive  # (E,S)
        seg_valid_b = np.broadcast_to(seg_valid[:, None, :], (E, S, MAXLEN))
        active = seg_valid_b & obs_alive[..., None]

        ei = np.broadcast_to(np.arange(E)[:, None, None], (E, S, MAXLEN))
        si = np.broadcast_to(np.arange(S)[None, :, None], (E, S, MAXLEN))
        obs_is_src = np.arange(S)[None, :, None] == src  # (1,S,1)
        obs_is_src = np.broadcast_to(obs_is_src, (E, S, MAXLEN))
        is_head = seg_k[None, None, :] == 0  # head segment

        # --- Own body / own head (observer == src) ---
        own = active & obs_is_src
        own_body = own & ~is_head
        own_head = own & is_head
        # Own body value = TTL.
        ttl_b = np.broadcast_to(ttl_byte[:, None, :], (E, S, MAXLEN))
        if np.any(own_body):
            _paint(
                code_plane,
                val_plane,
                ei[own_body],
                si[own_body],
                row[own_body],
                col[own_body],
                np.full(int(own_body.sum()), CODE_OWN_BODY, np.uint16),
                ttl_b[own_body],
                size,
            )
        if np.any(own_head):
            # Own head value = boost-engaged bit (255 if boosting else a low base).
            boost_val = np.where(src_boost, 255, 1).astype(np.uint16)  # (E,)
            hv = np.broadcast_to(boost_val[:, None, None], (E, S, MAXLEN))
            _paint(
                code_plane,
                val_plane,
                ei[own_head],
                si[own_head],
                row[own_head],
                col[own_head],
                np.full(int(own_head.sum()), CODE_OWN_HEAD, np.uint16),
                hv[own_head],
                size,
            )

        # --- Enemy body / enemy head (observer != src) ---
        enemy = active & ~obs_is_src
        enemy_body = enemy & ~is_head
        enemy_head = enemy & is_head
        if np.any(enemy_body):
            _paint(
                code_plane,
                val_plane,
                ei[enemy_body],
                si[enemy_body],
                row[enemy_body],
                col[enemy_body],
                np.full(int(enemy_body.sum()), CODE_ENEMY_BODY, np.uint16),
                ttl_b[enemy_body],
                size,
            )
        if np.any(enemy_head):
            # value = clamp(their_len/own_len, 0, 2)/2 -> byte.
            own_len = np.maximum(lengths, 1)  # (E,S)
            their_len = np.broadcast_to(src_len[:, None], (E, S))  # (E,S)
            ratio = np.clip(their_len / own_len, 0, 2) / 2.0  # (E,S)
            ratio_byte = np.clip(np.round(ratio * 255), 0, 255).astype(np.uint16)
            rb = np.broadcast_to(ratio_byte[..., None], (E, S, MAXLEN))
            _paint(
                code_plane,
                val_plane,
                ei[enemy_head],
                si[enemy_head],
                row[enemy_head],
                col[enemy_head],
                np.full(int(enemy_head.sum()), CODE_ENEMY_HEAD, np.uint16),
                rb[enemy_head],
                size,
            )

        # --- Enemy predicted-next cells (ch3): 1 step from heading (2 if boost) ---
        _paint_enemy_pred(
            inp,
            code_plane,
            val_plane,
            heads,
            heading,
            alive,
            size,
            src,
            seg,
            src_bodylen,
            src_alive,
            src_heading,
            src_boost,
        )


# Cardinal step vectors (up, right, down, left) as (dcol, drow) with +row down.
_CARDINAL_STEP = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int64)


def _paint_enemy_pred(
    inp,
    code_plane,
    val_plane,
    heads,
    heading,
    alive,
    size,
    src,
    seg,
    src_bodylen,
    src_alive,
    src_heading,
    src_boost,
):
    """Scatter enemy ``src``'s predicted-next cell(s) into observers' ch3."""
    E, S = inp.E, inp.S
    # src head cell = segment 0.
    src_head = seg[:, 0]  # (E, 2) col,row
    step = _CARDINAL_STEP[src_heading]  # (E, 2)
    pred1 = src_head + step  # (E, 2)
    pred2 = src_head + 2 * step  # (E, 2) only when boosting

    # Build a small list of (cell, valid) prediction points: pred1 always,
    # pred2 only when boosting.
    preds = [(pred1, src_alive), (pred2, src_alive & src_boost)]
    for pcell, pvalid in preds:
        # pcell is (E,2); expand to (E,1) then broadcast over observers (E,S).
        pc = pcell[:, 0][:, None]  # (E,1) col
        pr = pcell[:, 1][:, None]  # (E,1) row
        hc = heads[..., 0]  # (E,S)
        hr = heads[..., 1]
        dcol = pc - hc  # (E,S) via broadcast (E,1)-(E,S)
        drow = pr - hr
        ahead, lateral = _ego_offsets(dcol, drow, heading)
        row = TACTICAL_HEAD_ROW - ahead
        col = TACTICAL_HEAD_COL + lateral
        obs_is_src = np.arange(S)[None, :] == src  # (1,S)
        active = np.broadcast_to(pvalid[:, None], (E, S)) & alive & ~obs_is_src
        if not np.any(active):
            continue
        ei = np.broadcast_to(np.arange(E)[:, None], (E, S))
        si = np.broadcast_to(np.arange(S)[None, :], (E, S))
        _paint(
            code_plane,
            val_plane,
            ei[active],
            si[active],
            row[active],
            col[active],
            np.full(int(active.sum()), CODE_ENEMY_PRED, np.uint16),
            np.full(int(active.sum()), _FOOD_VALUE, np.uint16),
            size,
        )


def _build_strategic(inp: ObsInputs) -> np.ndarray:
    """Render the 25x25 strategic density rasters ``(E, S, 3, 25, 25)`` uint8.

    Three mass-weighted density channels at 5-segment resolution: enemy mass
    (bodies + heads), food mass (ambient + corpse), own body. Same ego rot90.
    Density is accumulated per coarse cell then scaled to a byte.
    """
    E, S = inp.E, inp.S
    size = STRATEGIC_SIZE
    acc = np.zeros((E, S, STRATEGIC_CHANNELS, size, size), dtype=np.float64)

    bodies = inp.bodies
    body_len = inp.body_len
    MAXLEN = bodies.shape[2]
    heads = inp.heads
    heading = inp.heading
    alive = inp.alive
    seg_k = np.arange(MAXLEN)

    def scatter(dcol, drow, chan, weight, valid, extra_es_mask=None):
        # dcol/drow: (E,S,K); chan int; weight (E,S,K) or scalar.
        ahead, lateral = _ego_offsets(dcol, drow, heading[..., None])
        # Coarse cell = round(offset / STRATEGIC_CELL) + head coarse index, so
        # the head (offset 0) sits exactly at (STRATEGIC_HEAD_ROW, _COL).
        crow = STRATEGIC_HEAD_ROW - np.round(ahead / STRATEGIC_CELL).astype(np.int64)
        ccol = STRATEGIC_HEAD_COL + np.round(lateral / STRATEGIC_CELL).astype(np.int64)
        inb = (crow >= 0) & (crow < size) & (ccol >= 0) & (ccol < size) & valid
        if extra_es_mask is not None:
            inb = inb & extra_es_mask
        if not np.any(inb):
            return
        ei = np.broadcast_to(np.arange(E)[:, None, None], dcol.shape)[inb]
        si = np.broadcast_to(np.arange(S)[None, :, None], dcol.shape)[inb]
        w = weight if np.ndim(weight) else np.full(dcol.shape, weight)
        w = np.broadcast_to(w, dcol.shape)[inb]
        np.add.at(acc, (ei, si, chan, crow[inb], ccol[inb]), w)

    # --- Snakes ---
    for src in range(S):
        seg = bodies[:, src]  # (E, MAXLEN, 2)
        src_bodylen = body_len[:, src]
        src_alive = alive[:, src]
        seg_valid = (seg_k[None, :] < src_bodylen[:, None]) & src_alive[:, None]  # (E,MAXLEN)
        segc = seg[:, None, :, 0]  # (E,1,MAXLEN)
        segr = seg[:, None, :, 1]
        dcol = segc - heads[..., 0:1]  # (E,S,MAXLEN)
        drow = segr - heads[..., 1:2]
        seg_valid_b = np.broadcast_to(seg_valid[:, None, :], (E, S, MAXLEN))
        active = seg_valid_b & alive[..., None]
        obs_is_src = np.arange(S)[None, :, None] == src
        obs_is_src = np.broadcast_to(obs_is_src, (E, S, MAXLEN))
        # Enemy mass density (ch0): observer != src.
        scatter(dcol, drow, 0, 1.0, active & ~obs_is_src)
        # Own body density (ch2): observer == src.
        scatter(dcol, drow, 2, 1.0, active & obs_is_src)

    # --- Food (ch1) ---
    food_cells = inp.food_cells
    food_mass = inp.food_mass
    F = food_cells.shape[1]
    if F > 0:
        fc = food_cells[:, None, :, 0]  # (E,1,F)
        fr = food_cells[:, None, :, 1]
        dcol = fc - heads[..., 0:1]  # (E,S,F)
        drow = fr - heads[..., 1:2]
        mass = np.broadcast_to(food_mass[:, None, :], (E, S, F))
        valid = (mass > 0) & alive[..., None]
        scatter(dcol, drow, 1, mass, valid)

    # Scale each channel to a byte. Density normalization: a coarse cell holds up
    # to STRATEGIC_CELL^2 = 25 segments; scale so a full cell ~ 255.
    norm = float(STRATEGIC_CELL * STRATEGIC_CELL)
    out = np.clip(np.round(acc / norm * 255.0), 0, 255).astype(np.uint8)
    return out


def _build_scalars(inp: ObsInputs) -> np.ndarray:
    """Build the 26 ego-frame Markov-completing scalars ``(E, S, 26)`` float32."""
    E, S = inp.E, inp.S
    lengths = np.maximum(inp.lengths, 1).astype(np.float64)
    alive = inp.alive
    heads = inp.heads
    heading = inp.heading

    out = np.zeros((E, S, SCALARS_DIM), dtype=np.float32)
    idx = 0

    def put(vals):
        nonlocal idx
        out[..., idx] = np.asarray(vals, dtype=np.float32)
        idx += 1

    max_len = max(1, inp.max_length)
    put(lengths / max_len)  # 0 length/max_length
    put(np.log1p(lengths) / np.log1p(max_len))  # 1 log-length
    put((lengths >= inp.min_boost_length).astype(np.float64))  # 2 boost_available
    cost_frames = max(1, inp.boost_cost_frames)
    put(inp.boost_frames / cost_frames)  # 3 boost_cost_phase
    starv = max(1, inp.starvation_max)
    put(np.clip(inp.frames_since_food / starv, 0, 1))  # 4 hunger
    max_frames = max(1, inp.max_frames)
    prog = np.clip(inp.frame / max_frames, 0, 1)  # (E,)
    put(np.broadcast_to(prog[:, None], (E, S)))  # 5 episode_progress
    alive_count = alive.sum(axis=1).astype(np.float64)  # (E,)
    put(np.broadcast_to((alive_count / max(1, inp.max_snakes))[:, None], (E, S)))  # 6

    # 7 mass_rank_percentile: fraction of snakes (in-env) with length <= this.
    lengths_masked = np.where(alive, lengths, -1.0)
    rank = (lengths_masked[:, :, None] >= lengths_masked[:, None, :]) & alive[:, None, :]
    rank_count = rank.sum(axis=2).astype(np.float64)  # (E,S)
    denom = np.maximum(alive_count[:, None], 1)
    percentile = np.where(alive, rank_count / denom, 0.0)
    put(percentile)  # 7

    # 8-11 wall distance ahead/right/behind/left (cells / 64 capped).
    hc = heads[..., 0].astype(np.float64)  # (E,S)
    hr = heads[..., 1].astype(np.float64)
    dist_left = hc  # cells to west wall (col 0)
    dist_right = (inp.grid_w - 1) - hc
    dist_up = hr
    dist_down = (inp.grid_h - 1) - hr
    # Map world wall distances into ego ahead/right/behind/left per heading.
    #   up:    ahead=up, right=right, behind=down, left=left
    #   right: ahead=right,right=down, behind=left, left=up
    #   down:  ahead=down, right=left, behind=up,   left=right
    #   left:  ahead=left, right=up,   behind=right,left=down
    h = heading
    ego_ahead = np.select(
        [h == 0, h == 1, h == 2, h == 3], [dist_up, dist_right, dist_down, dist_left]
    )
    ego_right = np.select(
        [h == 0, h == 1, h == 2, h == 3], [dist_right, dist_down, dist_left, dist_up]
    )
    ego_behind = np.select(
        [h == 0, h == 1, h == 2, h == 3], [dist_down, dist_left, dist_up, dist_right]
    )
    ego_left = np.select(
        [h == 0, h == 1, h == 2, h == 3], [dist_left, dist_up, dist_right, dist_down]
    )
    put(np.clip(ego_ahead / _WALL_DIST_NORM, 0, 1))  # 8
    put(np.clip(ego_right / _WALL_DIST_NORM, 0, 1))  # 9
    put(np.clip(ego_behind / _WALL_DIST_NORM, 0, 1))  # 10
    put(np.clip(ego_left / _WALL_DIST_NORM, 0, 1))  # 11

    # 12-14 nearest food BEYOND the tactical raster footprint (ego dx, dy, dist).
    fdx, fdy, fdist = _nearest_food_ego(inp)
    put(fdx)  # 12
    put(fdy)  # 13
    put(fdist)  # 14

    # 15-22 nearest-2-enemy-head summaries (ego dx, dy, size_ratio, is_boosting).
    e1, e2 = _nearest_enemy_summaries(inp)
    for arr in (e1, e2):
        put(arr[..., 0])  # dx
        put(arr[..., 1])  # dy
        put(arr[..., 2])  # size_ratio
        put(arr[..., 3])  # is_boosting

    # 23 world x, 24 world y (normalized to [0,1]).
    put(np.clip(hc / max(1, inp.grid_w - 1), 0, 1))  # 23
    put(np.clip(hr / max(1, inp.grid_h - 1), 0, 1))  # 24
    put(np.full((E, S), float(inp.arena_type_flag)))  # 25 arena_type flag

    assert idx == SCALARS_DIM, f"filled {idx} scalars, expected {SCALARS_DIM}"
    return out


# Tactical raster footprint in the ego (heading) frame, used to define which
# cells are "beyond the raster" for the nearest-food-beyond-raster scalar. A cell
# at ego (ahead, lateral) maps to raster (row = HEAD_ROW - ahead, col = HEAD_COL +
# lateral); it is inside iff both fall within [0, SIZE-1]. That yields the
# (asymmetric, forward-biased) ego bounds below.
_TACTICAL_AHEAD_MAX = TACTICAL_HEAD_ROW  # rows 0..HEAD_ROW ahead
_TACTICAL_AHEAD_MIN = TACTICAL_HEAD_ROW - (TACTICAL_SIZE - 1)  # negative == behind
_TACTICAL_LAT_MAX = TACTICAL_SIZE - 1 - TACTICAL_HEAD_COL
_TACTICAL_LAT_MIN = -TACTICAL_HEAD_COL


def _nearest_food_ego(inp: ObsInputs):
    """Ego (dx, dy, dist) to the nearest food pellet BEYOND the tactical raster.

    Pellets whose ego offset falls inside the 31x31 tactical footprint (blueprint
    §2.3) are excluded before the argmin, so this supplies the out-of-view Markov
    signal the spec intends rather than duplicating in-raster information. ``dx``/
    ``dy`` are the heading-frame lateral/ahead components normalized by the arena
    diagonal; ``dist`` is the normalized cell distance. Zeros when the env has no
    food beyond the raster or the agent is dead.
    """
    E, S = inp.E, inp.S
    food_cells = inp.food_cells
    food_mass = inp.food_mass
    F = food_cells.shape[1]
    diag = float(np.hypot(inp.grid_w, inp.grid_h)) or 1.0
    dx = np.zeros((E, S), dtype=np.float64)
    dy = np.zeros((E, S), dtype=np.float64)
    dist = np.zeros((E, S), dtype=np.float64)
    if F == 0:
        return dx, dy, dist
    fc = food_cells[:, None, :, 0]  # (E,1,F)
    fr = food_cells[:, None, :, 1]
    dcol = fc - inp.heads[..., 0:1]  # (E,S,F)
    drow = fr - inp.heads[..., 1:2]
    # Per-pellet ego offsets to test tactical-footprint containment.
    p_ahead, p_lat = _ego_offsets(dcol, drow, inp.heading[..., None])  # (E,S,F)
    in_raster = (
        (p_ahead >= _TACTICAL_AHEAD_MIN)
        & (p_ahead <= _TACTICAL_AHEAD_MAX)
        & (p_lat >= _TACTICAL_LAT_MIN)
        & (p_lat <= _TACTICAL_LAT_MAX)
    )
    d2 = dcol.astype(np.float64) ** 2 + drow.astype(np.float64) ** 2
    valid = (
        np.broadcast_to(food_mass[:, None, :] > 0, (E, S, F))
        & inp.alive[..., None]
        & ~in_raster  # only pellets BEYOND the tactical raster
    )
    d2 = np.where(valid, d2, np.inf)
    best = np.argmin(d2, axis=2)  # (E,S)
    any_valid = np.any(valid, axis=2)  # (E,S)
    ei = np.arange(E)[:, None]
    si = np.arange(S)[None, :]
    bcol = dcol[ei, si, best]
    brow = drow[ei, si, best]
    ahead, lateral = _ego_offsets(bcol, brow, inp.heading)
    dcell = np.sqrt(bcol.astype(np.float64) ** 2 + brow.astype(np.float64) ** 2)
    dx = np.where(any_valid, lateral / diag, 0.0)
    dy = np.where(any_valid, ahead / diag, 0.0)
    dist = np.where(any_valid, np.clip(dcell / diag, 0, 1), 0.0)
    return dx, dy, dist


def _nearest_enemy_summaries(inp: ObsInputs):
    """Ego summaries of the two nearest enemy heads per agent.

    Returns two ``(E, S, 4)`` arrays (nearest, 2nd nearest); columns are
    ``(ego_dx, ego_dy, size_ratio, is_boosting)``. Missing enemies are zeros.
    """
    E, S = inp.E, inp.S
    diag = float(np.hypot(inp.grid_w, inp.grid_h)) or 1.0
    heads = inp.heads  # (E,S,2)
    lengths = np.maximum(inp.lengths, 1).astype(np.float64)  # (E,S)
    alive = inp.alive
    boosting = inp.boosting

    # Pairwise: observer o vs source s (enemy heads). offset = enemy_head - o_head.
    ohc = heads[..., 0][:, :, None]  # (E,S,1)
    ohr = heads[..., 1][:, :, None]
    ehc = heads[..., 0][:, None, :]  # (E,1,S)
    ehr = heads[..., 1][:, None, :]
    dcol = ehc - ohc  # (E,S,S) observer x source
    drow = ehr - ohr
    d2 = dcol.astype(np.float64) ** 2 + drow.astype(np.float64) ** 2
    is_self = np.eye(S, dtype=bool)[None, :, :]
    src_alive = alive[:, None, :]  # (E,1,S)
    obs_alive = alive[:, :, None]
    valid = src_alive & obs_alive & ~is_self  # (E,S,S)
    d2 = np.where(valid, d2, np.inf)

    order = np.argsort(d2, axis=2)  # (E,S,S) sorted source indices by distance
    ei = np.arange(E)[:, None, None]
    oi = np.arange(S)[None, :, None]

    out = [np.zeros((E, S, 4), dtype=np.float64) for _ in range(2)]
    for rank in range(min(2, S)):
        src = order[:, :, rank]  # (E,S)
        d2_sel = np.take_along_axis(d2, src[..., None], axis=2)[..., 0]  # (E,S)
        present = np.isfinite(d2_sel)
        bcol = dcol[ei[..., 0], oi[..., 0], src]  # (E,S)
        brow = drow[ei[..., 0], oi[..., 0], src]
        ahead, lateral = _ego_offsets(bcol, brow, inp.heading)
        their_len = lengths[np.arange(E)[:, None], src]  # (E,S)
        ratio = np.clip(their_len / lengths, 0, 2) / 2.0
        their_boost = boosting[np.arange(E)[:, None], src].astype(np.float64)
        arr = out[rank]
        arr[..., 0] = np.where(present, lateral / diag, 0.0)
        arr[..., 1] = np.where(present, ahead / diag, 0.0)
        arr[..., 2] = np.where(present, ratio, 0.0)
        arr[..., 3] = np.where(present, their_boost, 0.0)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Expansion / network input
# ---------------------------------------------------------------------------
def expand_tactical(tactical_uint8: np.ndarray) -> np.ndarray:
    """Expand the 2-plane tactical uint8 to ``(E, S, 9, 31, 31)`` float32.

    The type-code plane routes each cell's value byte (``/255``) into its channel
    (blueprint §2.1). Channel 8 stays zero (reserved ablation slot).

    Args:
        tactical_uint8: ``(..., 2, 31, 31)`` uint8 (code plane, value plane).

    Returns:
        ``(..., 9, 31, 31)`` float32.
    """
    arr = np.asarray(tactical_uint8)
    code = arr[..., 0, :, :]  # (..., 31, 31)
    value = arr[..., 1, :, :].astype(np.float32) / 255.0
    lead = arr.shape[:-3]
    out = np.zeros(lead + (TACTICAL_CHANNELS, TACTICAL_SIZE, TACTICAL_SIZE), dtype=np.float32)
    for c, chan in _CODE_TO_CHANNEL.items():
        sel = code == c
        out[..., chan, :, :] = np.where(sel, value, out[..., chan, :, :])
    return out


def to_network_input(obs: dict):
    """Concatenate an observation dict into the tensors the CNN consumes.

    Args:
        obs: Dict from :func:`build_observations` (or a batched stack of them).

    Returns:
        Dict with float32 arrays ``tactical`` ``(..., 9, 31, 31)``,
        ``strategic`` ``(..., 3, 25, 25)`` and ``scalars`` ``(..., 26)``.
    """
    tactical = expand_tactical(obs["tactical_uint8"])
    strategic = np.asarray(obs["strategic_uint8"]).astype(np.float32) / 255.0
    scalars = np.asarray(obs["scalars"]).astype(np.float32)
    return {"tactical": tactical, "strategic": strategic, "scalars": scalars}


# ---------------------------------------------------------------------------
# BatchSim producer
# ---------------------------------------------------------------------------
def obs_inputs_from_batch_sim(sim, max_frames: int = 5000) -> ObsInputs:
    """Build an :class:`ObsInputs` from a :class:`BatchSim` via its accessors.

    Reads only the documented public/state arrays. Bodies are gathered into a
    padded ``(E, S, MAXLEN, 2)`` array in head->tail order; food is padded to the
    max per-env pellet count with mass 1 (ambient) and a corpse flag.

    Args:
        sim: A :class:`~src.simd_env.batch_sim.BatchSim` instance.
        max_frames: Episode-length cap for the episode-progress scalar.

    Returns:
        A filled :class:`ObsInputs` (E = sim.E, S = sim.S).
    """
    E, S = sim.E, sim.S
    heads = sim.get_heads().astype(np.int64)  # (E,S,2)
    lengths = sim.get_lengths().astype(np.int64)
    alive = sim.get_alive()
    heading = sim.get_directions().astype(np.int64)
    boost_frames = sim.get_boost_frames().astype(np.int64)
    frames_since_food = sim.get_frames_since_food().astype(np.int64)

    # Bodies -> padded array. seg_count is the number of live segments.
    seg_count = sim.seg_count.astype(np.int64)  # (E,S)
    maxlen = int(seg_count.max()) if seg_count.size else 1
    maxlen = max(maxlen, 1)
    bodies = np.zeros((E, S, maxlen, 2), dtype=np.int64)
    for e in range(E):
        for s in range(S):
            segs = sim.get_bodies(e, s)  # head->tail
            for k, cell in enumerate(segs):
                if k >= maxlen:
                    break
                bodies[e, s, k] = cell
    body_len = np.minimum(seg_count, maxlen)

    # Boosting flag: whether the snake moved a boost 2nd cell on the last step.
    # Read the persistent accessor (get_boosted_this_step) rather than the
    # internal _trav_valid[:, :, 1] mask, which is reset to False at the end of
    # every step() by _rebuild_traversed_from_heads and would otherwise be
    # permanently all-False at observation time.
    boosting = sim.get_boosted_this_step().astype(bool) & alive

    # Food: pad per-env lists.
    food_lists = [sim.get_food(e) for e in range(E)]
    corpse_sets = [set(sim.get_corpse_food(e)) for e in range(E)]
    F = max((len(f) for f in food_lists), default=0)
    F = max(F, 1)
    food_cells = np.zeros((E, F, 2), dtype=np.int64)
    food_mass = np.zeros((E, F), dtype=np.float64)
    food_is_corpse = np.zeros((E, F), dtype=bool)
    for e in range(E):
        for i, cell in enumerate(food_lists[e]):
            food_cells[e, i] = cell
            food_mass[e, i] = 1.0
            food_is_corpse[e, i] = cell in corpse_sets[e]

    return ObsInputs(
        heads=heads,
        bodies=bodies,
        body_len=body_len,
        lengths=lengths,
        alive=alive,
        heading=heading,
        boost_frames=boost_frames,
        frames_since_food=frames_since_food,
        boosting=boosting,
        food_cells=food_cells,
        food_mass=food_mass,
        food_is_corpse=food_is_corpse,
        grid_w=sim.grid_w,
        grid_h=sim.grid_h,
        max_snakes=sim.S,
        starvation_max=getattr(sim.cfg, "starvation_max", 500),
        max_length=int(getattr(sim.cfg, "max_capacity", 400)),
        min_boost_length=sim.cfg.min_boost_length,
        boost_cost_frames=sim.cfg.boost_length_cost_frames,
        frame=np.asarray(sim.frame, dtype=np.int64),
        max_frames=max_frames,
        arena_type_flag=1.0 if sim.cfg.arena_type == "circular" else 0.0,
    )
