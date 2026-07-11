"""GPU (torch) reimplementation of the dual-scale ego-raster featurizer.

This is a byte-for-byte-parity port of :mod:`src.simd_env.featurizer`
(:func:`build_observations` + :func:`expand_tactical`) that runs entirely on the
GPU. The CPU NumPy featurizer is the serve-time path and the ground truth; this
module exists purely to move the *training* featurize cost off the CPU (the PQN
trainer's dominant bottleneck) onto the otherwise-idle GPU while producing
observations that are indistinguishable from the NumPy path:

- ``tactical`` / ``strategic`` planes are byte-identical to
  ``expand_tactical(build_observations(...))`` (uint8-equivalent, then the same
  ``/255`` float expansion).
- ``scalars`` match within ``1e-4`` (float32 vs float64 accumulation only).

The parity test :mod:`tests` gate compares this module against the NumPy
featurizer over a stepped :class:`~src.simd_env.batch_sim.BatchSim`.

Design — replicating the NumPy overlap priority
------------------------------------------------
The NumPy tactical builder paints ``(code, value)`` points into two uint8 planes
with **unconditional last-write-wins** scatter, relying on the *order* of paint
operations (wall, then food, then per-source-snake own-body/own-head/enemy-body/
enemy-head/enemy-pred) to realize the overlap priority
``heads > bodies > predicted > corpse-food > ambient-food > wall``. Within a
single paint the only intra-op collision is a snake body self-overlap, resolved
by "largest segment offset ``k`` wins" (NumPy's C-order last-write).

CUDA ``index_put_(..., accumulate=False)`` has undefined behavior for duplicate
indices, so we cannot rely on scatter last-write. Instead each paint op:

1. reduces its candidate points to **one winner per cell** via a per-cell
   ``amax`` over a within-op rank (``k`` for bodies, ``0`` otherwise) — the
   winners then target pairwise-distinct cells, so the subsequent scatter has no
   duplicate indices and is deterministic; and
2. writes those winners **unconditionally**, in the exact NumPy op order, so a
   later op overwrites an earlier one at a shared cell — reproducing NumPy's
   sequential last-write-wins across ops.

Heading rotation reuses the same integer coefficient lookup as the NumPy
featurizer (exact ``rot90`` for the four cardinal headings). Everything is
vectorized over the ``E*S`` agent grid; only the ``S`` source-snake loop remains
(``S`` is small), matching the NumPy structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from src.simd_env.featurizer import (
    _CODE_TO_CHANNEL,
    _FOOD_VALUE,
    _TACTICAL_AHEAD_MAX,
    _TACTICAL_AHEAD_MIN,
    _TACTICAL_LAT_MAX,
    _TACTICAL_LAT_MIN,
    _WALL_DIST_NORM,
    CODE_AMBIENT_FOOD,
    CODE_CORPSE_FOOD,
    CODE_ENEMY_BODY,
    CODE_ENEMY_HEAD,
    CODE_ENEMY_PRED,
    CODE_OWN_BODY,
    CODE_OWN_HEAD,
    CODE_WALL,
    SCALARS_DIM,
    STRATEGIC_CELL,
    STRATEGIC_CHANNELS,
    STRATEGIC_HEAD_COL,
    STRATEGIC_HEAD_ROW,
    STRATEGIC_SIZE,
    TACTICAL_CHANNELS,
    TACTICAL_HEAD_COL,
    TACTICAL_HEAD_ROW,
    TACTICAL_SIZE,
)

__all__ = [
    "GpuObsState",
    "obs_inputs_to_torch",
    "build_observations_gpu",
    "raster_tensors_from_gpu_obs",
]

# Per-heading rotation coefficients (0=up, 1=right, 2=down, 3=left), matching
# featurizer._AHEAD_*/_LAT_* exactly.
_AHEAD_DCOL = (0, 1, 0, -1)
_AHEAD_DROW = (-1, 0, 1, 0)
_LAT_DCOL = (1, 0, -1, 0)
_LAT_DROW = (0, 1, 0, -1)

# Cardinal step vectors (dcol, drow) with +row down (matches _CARDINAL_STEP).
_CARDINAL_STEP = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass
class GpuObsState:
    """Compact, GPU-resident world snapshot the torch featurizer consumes.

    The torch analogue of :class:`~src.simd_env.featurizer.ObsInputs`. Only the
    small compact arrays are transferred to the device (never the finished
    rasters — that transfer is exactly what this path eliminates). ``E`` is the
    number of environments, ``S`` snakes per env, ``MAXLEN`` the padded body
    capacity, ``F`` the padded per-env pellet count.

    Attributes:
        heads: ``(E, S, 2)`` long head cell ``(col, row)``.
        bodies: ``(E, S, MAXLEN, 2)`` long body cells head->tail (slots beyond
            ``body_len`` are ignored).
        body_len: ``(E, S)`` long valid body-segment count.
        lengths: ``(E, S)`` long logical length.
        alive: ``(E, S)`` bool alive mask.
        heading: ``(E, S)`` long cardinal heading (0=up..3=left).
        boost_frames: ``(E, S)`` long boost burn-cadence counter.
        frames_since_food: ``(E, S)`` long hunger counter.
        boosting: ``(E, S)`` bool boost-engaged-this-step flag.
        food_cells: ``(E, F, 2)`` long pellet cell ``(col, row)``.
        food_mass: ``(E, F)`` float pellet mass (0 marks padding).
        food_is_corpse: ``(E, F)`` bool corpse-class flag.
        frame: ``(E,)`` long current frame per env.
        grid_w / grid_h: arena extent in cells.
        max_snakes / starvation_max / max_length / min_boost_length /
            boost_cost_frames / max_frames: scalar normalization constants
            (mirrors :class:`ObsInputs`).
        arena_type_flag: 0.0 rectangular, 1.0 circular.
        device: The device all tensors live on.
    """

    heads: torch.Tensor
    bodies: torch.Tensor
    body_len: torch.Tensor
    lengths: torch.Tensor
    alive: torch.Tensor
    heading: torch.Tensor
    boost_frames: torch.Tensor
    frames_since_food: torch.Tensor
    boosting: torch.Tensor
    food_cells: torch.Tensor
    food_mass: torch.Tensor
    food_is_corpse: torch.Tensor
    frame: torch.Tensor
    grid_w: int
    grid_h: int
    max_snakes: int
    starvation_max: int
    max_length: int
    min_boost_length: int
    boost_cost_frames: int
    max_frames: int
    arena_type_flag: float
    device: torch.device

    @property
    def E(self) -> int:  # noqa: N802 - matches BatchSim naming
        """Number of environments."""
        return int(self.heads.shape[0])

    @property
    def S(self) -> int:  # noqa: N802 - matches BatchSim naming
        """Number of snakes per environment."""
        return int(self.heads.shape[1])


# ---------------------------------------------------------------------------
# Sim -> GPU bridge
# ---------------------------------------------------------------------------
def obs_inputs_to_torch(sim, device: torch.device, max_frames: int = 5000) -> GpuObsState:
    """Extract the compact sim state and move it to ``device`` in one shot.

    Mirrors :func:`~src.simd_env.featurizer.obs_inputs_from_batch_sim` but keeps
    everything as small compact arrays (bodies gathered to ``(E, S, MAXLEN, 2)``
    via a vectorized ring-buffer read, food padded per-env) so the host->device
    transfer stays tiny. The finished rasters are built on the GPU afterwards.

    Args:
        sim: A :class:`~src.simd_env.batch_sim.BatchSim` instance.
        device: Target compute device.
        max_frames: Episode-length cap for the episode-progress scalar.

    Returns:
        A :class:`GpuObsState` with all arrays resident on ``device``.
    """
    E, S = sim.E, sim.S

    heads = sim.get_heads().astype(np.int64)  # (E, S, 2)
    lengths = sim.get_lengths().astype(np.int64)
    alive = sim.get_alive()
    heading = sim.get_directions().astype(np.int64)
    boost_frames = sim.get_boost_frames().astype(np.int64)
    frames_since_food = sim.get_frames_since_food().astype(np.int64)

    # Bodies -> padded (E, S, MAXLEN, 2) via a vectorized ring-buffer gather.
    # ring slot for segment offset k from head is (head_ptr - k) % cap; this is
    # the exact head->tail order obs_inputs_from_batch_sim builds cell-by-cell.
    seg_count = sim.seg_count.astype(np.int64)  # (E, S)
    maxlen = int(seg_count.max()) if seg_count.size else 1
    maxlen = max(maxlen, 1)
    cap = int(sim.cap)
    k = np.arange(maxlen, dtype=np.int64)
    ring = (sim.head_ptr[:, :, None] - k[None, None, :]) % cap  # (E, S, MAXLEN)
    ei = np.arange(E)[:, None, None]
    si = np.arange(S)[None, :, None]
    bodies = sim.bodies[ei, si, ring].astype(np.int64)  # (E, S, MAXLEN, 2)
    body_len = np.minimum(seg_count, maxlen)

    # Boost-engaged flag (persistent accessor); AND with alive like NumPy.
    boosting = sim.get_boosted_this_step().astype(bool) & alive

    # Food: padded (E, Fmax, 2) cells + mass + corpse flag, built in one batched
    # pass by the sim (no per-call list copies / element-wise Python loop).
    food_cells, food_mass, food_is_corpse = sim.get_food_batched()

    def t(arr, dtype):
        return torch.as_tensor(arr, dtype=dtype, device=device)

    return GpuObsState(
        heads=t(heads, torch.int64),
        bodies=t(bodies, torch.int64),
        body_len=t(body_len, torch.int64),
        lengths=t(lengths, torch.int64),
        alive=t(alive, torch.bool),
        heading=t(heading, torch.int64),
        boost_frames=t(boost_frames, torch.int64),
        frames_since_food=t(frames_since_food, torch.int64),
        boosting=t(boosting, torch.bool),
        food_cells=t(food_cells, torch.int64),
        food_mass=t(food_mass, torch.float64),
        food_is_corpse=t(food_is_corpse, torch.bool),
        frame=t(np.asarray(sim.frame, dtype=np.int64), torch.int64),
        grid_w=int(sim.grid_w),
        grid_h=int(sim.grid_h),
        max_snakes=int(sim.S),
        starvation_max=int(getattr(sim.cfg, "starvation_max", 500)),
        max_length=int(getattr(sim.cfg, "max_capacity", 400)),
        min_boost_length=int(sim.cfg.min_boost_length),
        boost_cost_frames=int(sim.cfg.boost_length_cost_frames),
        max_frames=int(max_frames),
        arena_type_flag=1.0 if sim.cfg.arena_type == "circular" else 0.0,
        device=device,
    )


# ---------------------------------------------------------------------------
# Heading rotation (torch)
# ---------------------------------------------------------------------------
def _ego_offsets(dcol: torch.Tensor, drow: torch.Tensor, heading: torch.Tensor):
    """Rotate world offsets ``(dcol, drow)`` into the heading ("up") frame.

    Torch port of :func:`~src.simd_env.featurizer._ego_offsets`; identical
    integer coefficient lookup (exact for the four cardinal headings).

    Args:
        dcol: World column offset from the head.
        drow: World row offset from the head.
        heading: Cardinal heading index, broadcastable to ``dcol``/``drow``.

    Returns:
        ``(ahead, lateral)`` long offset tensors in the heading frame.
    """
    dev = dcol.device
    ahead_dcol = torch.tensor(_AHEAD_DCOL, dtype=torch.int64, device=dev)
    ahead_drow = torch.tensor(_AHEAD_DROW, dtype=torch.int64, device=dev)
    lat_dcol = torch.tensor(_LAT_DCOL, dtype=torch.int64, device=dev)
    lat_drow = torch.tensor(_LAT_DROW, dtype=torch.int64, device=dev)
    h = heading
    ahead = ahead_dcol[h] * dcol + ahead_drow[h] * drow
    lateral = lat_dcol[h] * dcol + lat_drow[h] * drow
    return ahead, lateral


def _inbounds(row: torch.Tensor, col: torch.Tensor, size: int) -> torch.Tensor:
    """Boolean mask of raster cells inside ``[0, size)`` on both axes."""
    return (row >= 0) & (row < size) & (col >= 0) & (col < size)


# ---------------------------------------------------------------------------
# Tactical raster
# ---------------------------------------------------------------------------
def _paint(
    code_plane: torch.Tensor,
    val_plane: torch.Tensor,
    cell: torch.Tensor,
    code: torch.Tensor,
    value: torch.Tensor,
    krank: torch.Tensor,
) -> None:
    """Last-write-wins scatter of one paint op with intra-op ``amax`` tie-break.

    Reduces candidates to one winner per cell (highest ``krank`` — segment offset
    ``k`` for bodies, ``0`` otherwise) so the winning cells are pairwise distinct,
    then writes ``(code, value)`` unconditionally. Called in NumPy op order, so a
    later op overwrites an earlier one at a shared cell.

    Args:
        code_plane: ``(A*size*size,)`` flat int64 type-code plane.
        val_plane: ``(A*size*size,)`` flat int64 value plane.
        cell: ``(M,)`` flat global cell index per candidate (already in-bounds).
        code: ``(M,)`` long type code per candidate.
        value: ``(M,)`` long value byte per candidate.
        krank: ``(M,)`` long within-op rank (higher wins a contested cell).
    """
    if cell.numel() == 0:
        return
    ncells = code_plane.numel()
    best = torch.full((ncells,), -1, dtype=torch.int64, device=code_plane.device)
    best.scatter_reduce_(0, cell, krank, reduce="amax", include_self=True)
    winner = krank == best[cell]
    wc = cell[winner]
    code_plane[wc] = code[winner]
    val_plane[wc] = value[winner]


def _build_tactical(state: GpuObsState) -> torch.Tensor:
    """Render the 31x31 tactical rasters and expand to ``(E, S, 9, 31, 31)``.

    Args:
        state: The GPU world snapshot.

    Returns:
        ``(E, S, 9, 31, 31)`` float32 tensor (the expanded tactical raster).
    """
    dev = state.device
    E, S = state.E, state.S
    size = TACTICAL_SIZE
    ncell = size * size
    A = E * S

    code_plane = torch.zeros(A * ncell, dtype=torch.int64, device=dev)
    val_plane = torch.zeros(A * ncell, dtype=torch.int64, device=dev)

    heads = state.heads  # (E, S, 2)
    heading = state.heading  # (E, S)
    alive = state.alive  # (E, S)
    head_col = heads[..., 0]  # (E, S)
    head_row = heads[..., 1]

    # Flat agent index a = e*S + s for each (E, S) agent.
    agent_idx = torch.arange(E, device=dev)[:, None] * S + torch.arange(S, device=dev)[None, :]

    # --- Wall / out-of-arena (code 1): ego grid -> world per heading. ---
    rr, cc = torch.meshgrid(
        torch.arange(size, device=dev), torch.arange(size, device=dev), indexing="ij"
    )
    ego_ahead = (TACTICAL_HEAD_ROW - rr).reshape(-1)  # (ncell,)
    ego_lat = (cc - TACTICAL_HEAD_COL).reshape(-1)
    heading_flat = heading.reshape(-1)  # (A,)
    head_col_flat = head_col.reshape(-1)
    head_row_flat = head_row.reshape(-1)
    alive_flat = alive.reshape(-1)
    # Invert ego->world per heading (coefficient lookup; matches NumPy).
    dcol_a = torch.tensor([0, 1, 0, -1], dtype=torch.int64, device=dev)
    dcol_l = torch.tensor([1, 0, -1, 0], dtype=torch.int64, device=dev)
    drow_a = torch.tensor([-1, 0, 1, 0], dtype=torch.int64, device=dev)
    drow_l = torch.tensor([0, 1, 0, -1], dtype=torch.int64, device=dev)
    h_col = heading_flat[:, None]  # (A, 1)
    a = ego_ahead[None, :]  # (1, ncell)
    lat = ego_lat[None, :]
    dcol_w = dcol_a[h_col] * a + dcol_l[h_col] * lat  # (A, ncell)
    drow_w = drow_a[h_col] * a + drow_l[h_col] * lat
    world_col = head_col_flat[:, None] + dcol_w
    world_row = head_row_flat[:, None] + drow_w
    outside = (
        (world_col < 0)
        | (world_col >= state.grid_w)
        | (world_row < 0)
        | (world_row >= state.grid_h)
    )
    wall_pts = outside & alive_flat[:, None]  # (A, ncell)
    if bool(wall_pts.any()):
        agent_ar = torch.arange(A, device=dev)[:, None].expand(A, ncell)
        cellidx_ar = torch.arange(ncell, device=dev)[None, :].expand(A, ncell)
        sel = wall_pts
        cell = (agent_ar[sel] * ncell + cellidx_ar[sel]).reshape(-1)
        m = cell.numel()
        _paint(
            code_plane,
            val_plane,
            cell,
            torch.full((m,), CODE_WALL, dtype=torch.int64, device=dev),
            torch.full((m,), int(_FOOD_VALUE), dtype=torch.int64, device=dev),
            torch.zeros(m, dtype=torch.int64, device=dev),
        )

    # --- Food (ambient code 2 / corpse code 3), value = mass byte. ---
    _paint_food_tactical(state, code_plane, val_plane, agent_idx, size, ncell)

    # --- Snakes: own/enemy body + head + enemy predicted cells. ---
    _paint_snakes_tactical(state, code_plane, val_plane, agent_idx, size, ncell)

    # Expand to 9 float channels (byte-identical to expand_tactical).
    code_p = code_plane.reshape(E, S, size, size)
    value_f = (torch.clamp(val_plane, max=255).reshape(E, S, size, size).to(torch.float32)) / 255.0
    out = torch.zeros((E, S, TACTICAL_CHANNELS, size, size), dtype=torch.float32, device=dev)
    for c, chan in _CODE_TO_CHANNEL.items():
        sel = code_p == c
        out[:, :, chan] = torch.where(sel, value_f, out[:, :, chan])
    return out


def _paint_food_tactical(state, code_plane, val_plane, agent_idx, size, ncell) -> None:
    """Scatter ambient/corpse food pellets into the tactical rasters."""
    E, S = state.E, state.S
    F = state.food_cells.shape[1]
    heads = state.heads
    heading = state.heading
    alive = state.alive

    fc = state.food_cells[:, None, :, 0]  # (E, 1, F) col
    fr = state.food_cells[:, None, :, 1]
    dcol = fc - heads[..., 0:1]  # (E, S, F)
    drow = fr - heads[..., 1:2]
    ahead, lateral = _ego_offsets(dcol, drow, heading[..., None])
    row = TACTICAL_HEAD_ROW - ahead
    col = TACTICAL_HEAD_COL + lateral
    mass = state.food_mass[:, None, :].expand(E, S, F)  # (E, S, F)
    corpse = state.food_is_corpse[:, None, :].expand(E, S, F)
    valid = (mass > 0) & alive[..., None] & _inbounds(row, col, size)
    if not bool(valid.any()):
        return
    code = torch.where(
        corpse,
        torch.full_like(corpse, CODE_CORPSE_FOOD, dtype=torch.int64),
        torch.full_like(corpse, CODE_AMBIENT_FOOD, dtype=torch.int64),
    )
    value = torch.clamp(mass.to(torch.int64) * int(_FOOD_VALUE), max=255)
    cell = agent_idx[:, :, None] * ncell + row * size + col
    sel = valid
    # Rank food candidates by their type CODE so the per-cell ``amax`` in ``_paint``
    # picks the highest-priority pellet (corpse code 3 beats ambient code 2) at a
    # contested ego cell — reproducing the NumPy path's stable code-ascending sort
    # + last-write-wins. Passing a constant rank (e.g. zeros) would instead select
    # every colliding pellet as a "winner", leaving the survivor to torch's
    # duplicate-index scatter order (pellet array order; nondeterministic on CUDA),
    # which diverges from NumPy whenever a corpse precedes an ambient pellet.
    _paint(
        code_plane,
        val_plane,
        cell[sel],
        code[sel],
        value[sel],
        code[sel],
    )


def _paint_snakes_tactical(state, code_plane, val_plane, agent_idx, size, ncell) -> None:
    """Scatter own/enemy bodies, heads and enemy predicted cells (tactical)."""
    dev = state.device
    E, S = state.E, state.S
    bodies = state.bodies  # (E, S, MAXLEN, 2)
    body_len = state.body_len  # (E, S)
    MAXLEN = bodies.shape[2]
    heads = state.heads
    heading = state.heading
    alive = state.alive
    lengths = torch.clamp(state.lengths, min=1)  # (E, S) observer lengths

    seg_k = torch.arange(MAXLEN, device=dev)  # offset from head (0 = head)
    obs_range = torch.arange(S, device=dev)
    cardinal_step = torch.tensor(_CARDINAL_STEP, dtype=torch.int64, device=dev)  # (4, 2)

    for src in range(S):
        seg = bodies[:, src]  # (E, MAXLEN, 2)
        src_bodylen = body_len[:, src]  # (E,)
        src_alive = alive[:, src]  # (E,)
        src_len = torch.clamp(state.lengths[:, src], min=1)  # (E,)
        src_heading = heading[:, src]  # (E,)
        src_boost = state.boosting[:, src]  # (E,)

        seg_valid = (seg_k[None, :] < src_bodylen[:, None]) & src_alive[:, None]  # (E, MAXLEN)
        # TTL byte: head (k=0) high -> tail low. ttl = (bodylen - k) / bodylen.
        denom = torch.clamp(src_bodylen[:, None], min=1).to(torch.float64)
        ttl = (src_bodylen[:, None] - seg_k[None, :]).to(torch.float64) / denom
        ttl_byte = torch.clamp(torch.round(ttl * 255.0), 0, 255).to(torch.int64)  # (E, MAXLEN)

        segc = seg[:, None, :, 0]  # (E, 1, MAXLEN)
        segr = seg[:, None, :, 1]
        dcol = segc - heads[..., 0:1]  # (E, S, MAXLEN)
        drow = segr - heads[..., 1:2]
        ahead, lateral = _ego_offsets(dcol, drow, heading[..., None])
        row = TACTICAL_HEAD_ROW - ahead
        col = TACTICAL_HEAD_COL + lateral

        seg_valid_b = seg_valid[:, None, :].expand(E, S, MAXLEN)
        active = seg_valid_b & alive[..., None] & _inbounds(row, col, size)
        obs_is_src = (obs_range == src)[None, :, None].expand(E, S, MAXLEN)
        is_head = (seg_k == 0)[None, None, :].expand(E, S, MAXLEN)
        ttl_b = ttl_byte[:, None, :].expand(E, S, MAXLEN)
        cell = agent_idx[:, :, None] * ncell + row * size + col
        krank_k = seg_k[None, None, :].expand(E, S, MAXLEN)  # k tie-break for bodies

        # --- Own body / own head (observer == src) ---
        own = active & obs_is_src
        own_body = own & ~is_head
        own_head = own & is_head
        _paint(
            code_plane,
            val_plane,
            cell[own_body],
            torch.full((int(own_body.sum()),), CODE_OWN_BODY, dtype=torch.int64, device=dev),
            ttl_b[own_body],
            krank_k[own_body],
        )
        head_val = torch.where(
            src_boost,
            torch.full_like(src_boost, 255, dtype=torch.int64),
            torch.full_like(src_boost, 1, dtype=torch.int64),
        )  # (E,)
        head_val_b = head_val[:, None, None].expand(E, S, MAXLEN)
        _paint(
            code_plane,
            val_plane,
            cell[own_head],
            torch.full((int(own_head.sum()),), CODE_OWN_HEAD, dtype=torch.int64, device=dev),
            head_val_b[own_head],
            torch.zeros(int(own_head.sum()), dtype=torch.int64, device=dev),
        )

        # --- Enemy body / enemy head (observer != src) ---
        enemy = active & ~obs_is_src
        enemy_body = enemy & ~is_head
        enemy_head = enemy & is_head
        _paint(
            code_plane,
            val_plane,
            cell[enemy_body],
            torch.full((int(enemy_body.sum()),), CODE_ENEMY_BODY, dtype=torch.int64, device=dev),
            ttl_b[enemy_body],
            krank_k[enemy_body],
        )
        # Enemy head value = clamp(their_len / own_len, 0, 2) / 2 -> byte.
        their_len = src_len[:, None].expand(E, S).to(torch.float64)  # (E, S)
        ratio = torch.clamp(their_len / lengths.to(torch.float64), 0, 2) / 2.0
        ratio_byte = torch.clamp(torch.round(ratio * 255.0), 0, 255).to(torch.int64)  # (E, S)
        ratio_b = ratio_byte[..., None].expand(E, S, MAXLEN)
        _paint(
            code_plane,
            val_plane,
            cell[enemy_head],
            torch.full((int(enemy_head.sum()),), CODE_ENEMY_HEAD, dtype=torch.int64, device=dev),
            ratio_b[enemy_head],
            torch.zeros(int(enemy_head.sum()), dtype=torch.int64, device=dev),
        )

        # --- Enemy predicted-next cell(s) (code 4): 1 step, 2 when boosting. ---
        src_head = seg[:, 0]  # (E, 2) col,row
        step = cardinal_step[src_heading]  # (E, 2)
        pred1 = src_head + step
        pred2 = src_head + 2 * step
        preds = [(pred1, src_alive), (pred2, src_alive & src_boost)]
        for pcell, pvalid in preds:
            pc = pcell[:, 0][:, None]  # (E, 1) col
            pr = pcell[:, 1][:, None]
            dcol_p = pc - heads[..., 0]  # (E, S)
            drow_p = pr - heads[..., 1]
            ahead_p, lateral_p = _ego_offsets(dcol_p, drow_p, heading)
            row_p = TACTICAL_HEAD_ROW - ahead_p
            col_p = TACTICAL_HEAD_COL + lateral_p
            obs_is_src2 = (obs_range == src)[None, :].expand(E, S)
            active_p = pvalid[:, None] & alive & ~obs_is_src2 & _inbounds(row_p, col_p, size)
            if not bool(active_p.any()):
                continue
            cell_p = agent_idx * ncell + row_p * size + col_p
            m = int(active_p.sum())
            _paint(
                code_plane,
                val_plane,
                cell_p[active_p],
                torch.full((m,), CODE_ENEMY_PRED, dtype=torch.int64, device=dev),
                torch.full((m,), int(_FOOD_VALUE), dtype=torch.int64, device=dev),
                torch.zeros(m, dtype=torch.int64, device=dev),
            )


# ---------------------------------------------------------------------------
# Strategic raster
# ---------------------------------------------------------------------------
def _build_strategic(state: GpuObsState) -> torch.Tensor:
    """Render the 25x25 strategic density rasters ``(E, S, 3, 25, 25)`` float32.

    Three mass-weighted density channels (enemy mass, food mass, own body) at
    5-cell resolution. Accumulation is a pure sum (order-independent) so the
    uint8 result is byte-identical to NumPy; ``/255`` gives the float output.

    Args:
        state: The GPU world snapshot.

    Returns:
        ``(E, S, 3, 25, 25)`` float32 tensor.
    """
    dev = state.device
    E, S = state.E, state.S
    size = STRATEGIC_SIZE
    ncell = size * size
    A = E * S
    acc = torch.zeros(A * STRATEGIC_CHANNELS * ncell, dtype=torch.float64, device=dev)

    heads = state.heads
    heading = state.heading
    alive = state.alive
    agent_idx = torch.arange(E, device=dev)[:, None] * S + torch.arange(S, device=dev)[None, :]
    stride_agent = STRATEGIC_CHANNELS * ncell

    def scatter(dcol, drow, chan, weight, valid):
        ahead, lateral = _ego_offsets(dcol, drow, heading[..., None])
        crow = STRATEGIC_HEAD_ROW - torch.round(ahead.to(torch.float64) / STRATEGIC_CELL).to(
            torch.int64
        )
        ccol = STRATEGIC_HEAD_COL + torch.round(lateral.to(torch.float64) / STRATEGIC_CELL).to(
            torch.int64
        )
        inb = _inbounds(crow, ccol, size) & valid
        if not bool(inb.any()):
            return
        key = agent_idx[:, :, None] * stride_agent + chan * ncell + crow * size + ccol
        w = (
            weight
            if torch.is_tensor(weight)
            else torch.full_like(dcol, weight, dtype=torch.float64)
        )
        w = w.to(torch.float64).expand_as(dcol)
        acc.index_add_(0, key[inb].reshape(-1), w[inb].reshape(-1))

    bodies = state.bodies
    body_len = state.body_len
    MAXLEN = bodies.shape[2]
    seg_k = torch.arange(MAXLEN, device=dev)
    obs_range = torch.arange(S, device=dev)

    for src in range(S):
        seg = bodies[:, src]  # (E, MAXLEN, 2)
        src_bodylen = body_len[:, src]
        src_alive = alive[:, src]
        seg_valid = (seg_k[None, :] < src_bodylen[:, None]) & src_alive[:, None]  # (E, MAXLEN)
        segc = seg[:, None, :, 0]  # (E, 1, MAXLEN)
        segr = seg[:, None, :, 1]
        dcol = segc - heads[..., 0:1]  # (E, S, MAXLEN)
        drow = segr - heads[..., 1:2]
        seg_valid_b = seg_valid[:, None, :].expand(E, S, MAXLEN)
        active = seg_valid_b & alive[..., None]
        obs_is_src = (obs_range == src)[None, :, None].expand(E, S, MAXLEN)
        scatter(dcol, drow, 0, 1.0, active & ~obs_is_src)  # enemy mass density
        scatter(dcol, drow, 2, 1.0, active & obs_is_src)  # own body density

    F = state.food_cells.shape[1]
    fc = state.food_cells[:, None, :, 0]  # (E, 1, F)
    fr = state.food_cells[:, None, :, 1]
    dcol = fc - heads[..., 0:1]  # (E, S, F)
    drow = fr - heads[..., 1:2]
    mass = state.food_mass[:, None, :].expand(E, S, F)
    valid = (mass > 0) & alive[..., None]
    scatter(dcol, drow, 1, mass, valid)  # food mass density

    norm = float(STRATEGIC_CELL * STRATEGIC_CELL)
    acc = acc.reshape(E, S, STRATEGIC_CHANNELS, size, size)
    out_u8 = torch.clamp(torch.round(acc / norm * 255.0), 0, 255)
    return (out_u8.to(torch.float32)) / 255.0


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------
def _first_argmin(values: torch.Tensor, dim: int) -> torch.Tensor:
    """First-occurrence ``argmin`` along ``dim`` (matches ``np.argmin``).

    The inputs here are integer squared distances (distinct values differ by
    >= 1) padded with ``inf``. A tiny strictly-increasing index tie-break is
    added so equal-distance ties resolve to the lowest index — reproducing
    ``np.argmin``'s first-occurrence rule deterministically on any device.
    """
    n = values.shape[dim]
    tb = torch.arange(n, device=values.device, dtype=torch.float64) / (n + 1)
    shape = [1] * values.dim()
    shape[dim] = n
    return torch.argmin(values + tb.reshape(shape), dim=dim)


def _nearest_food_ego(state: GpuObsState):
    """Ego ``(dx, dy, dist)`` to the nearest food pellet BEYOND the tactical raster."""
    dev = state.device
    E, S = state.E, state.S
    diag = float(np.hypot(state.grid_w, state.grid_h)) or 1.0

    fc = state.food_cells[:, None, :, 0]  # (E, 1, F)
    fr = state.food_cells[:, None, :, 1]
    dcol = fc - state.heads[..., 0:1]  # (E, S, F)
    drow = fr - state.heads[..., 1:2]
    p_ahead, p_lat = _ego_offsets(dcol, drow, state.heading[..., None])
    in_raster = (
        (p_ahead >= _TACTICAL_AHEAD_MIN)
        & (p_ahead <= _TACTICAL_AHEAD_MAX)
        & (p_lat >= _TACTICAL_LAT_MIN)
        & (p_lat <= _TACTICAL_LAT_MAX)
    )
    d2 = dcol.to(torch.float64) ** 2 + drow.to(torch.float64) ** 2
    valid = (state.food_mass[:, None, :] > 0) & state.alive[..., None] & ~in_raster
    inf = torch.tensor(float("inf"), dtype=torch.float64, device=dev)
    d2 = torch.where(valid, d2, inf)
    best = _first_argmin(d2, dim=2)  # (E, S)
    any_valid = valid.any(dim=2)
    bcol = torch.gather(dcol, 2, best[..., None]).squeeze(-1)  # (E, S)
    brow = torch.gather(drow, 2, best[..., None]).squeeze(-1)
    ahead, lateral = _ego_offsets(bcol, brow, state.heading)
    dcell = torch.sqrt(bcol.to(torch.float64) ** 2 + brow.to(torch.float64) ** 2)
    z = torch.zeros((E, S), dtype=torch.float64, device=dev)
    dx = torch.where(any_valid, lateral.to(torch.float64) / diag, z)
    dy = torch.where(any_valid, ahead.to(torch.float64) / diag, z)
    dist = torch.where(any_valid, torch.clamp(dcell / diag, 0, 1), z)
    return dx, dy, dist


def _nearest_enemy_summaries(state: GpuObsState):
    """Ego summaries ``(dx, dy, size_ratio, is_boosting)`` of the 2 nearest enemies."""
    dev = state.device
    E, S = state.E, state.S
    diag = float(np.hypot(state.grid_w, state.grid_h)) or 1.0
    heads = state.heads
    lengths = torch.clamp(state.lengths, min=1).to(torch.float64)  # (E, S)
    alive = state.alive
    boosting = state.boosting

    ohc = heads[..., 0][:, :, None]  # (E, S, 1)
    ohr = heads[..., 1][:, :, None]
    ehc = heads[..., 0][:, None, :]  # (E, 1, S)
    ehr = heads[..., 1][:, None, :]
    dcol = ehc - ohc  # (E, S, S)  observer x source
    drow = ehr - ohr
    d2 = dcol.to(torch.float64) ** 2 + drow.to(torch.float64) ** 2
    is_self = torch.eye(S, dtype=torch.bool, device=dev)[None, :, :]
    valid = alive[:, None, :] & alive[:, :, None] & ~is_self  # (E, S, S)
    inf = torch.tensor(float("inf"), dtype=torch.float64, device=dev)
    d2 = torch.where(valid, d2, inf)

    # Stable order by distance with a lowest-index tie-break (matches np.argmin
    # for the first, deterministic for the rest).
    n = S
    tb = torch.arange(n, device=dev, dtype=torch.float64) / (n + 1)
    order = torch.argsort(d2 + tb[None, None, :], dim=2, stable=True)  # (E, S, S)

    out = [torch.zeros((E, S, 4), dtype=torch.float64, device=dev) for _ in range(2)]
    z = torch.zeros((E, S), dtype=torch.float64, device=dev)
    for rank in range(min(2, S)):
        src = order[:, :, rank]  # (E, S)
        d2_sel = torch.gather(d2, 2, src[..., None]).squeeze(-1)  # (E, S)
        present = torch.isfinite(d2_sel)
        bcol = torch.gather(dcol, 2, src[..., None]).squeeze(-1)
        brow = torch.gather(drow, 2, src[..., None]).squeeze(-1)
        ahead, lateral = _ego_offsets(bcol, brow, state.heading)
        their_len = torch.gather(lengths, 1, src)  # (E, S)
        ratio = torch.clamp(their_len / lengths, 0, 2) / 2.0
        their_boost = torch.gather(boosting, 1, src).to(torch.float64)
        arr = out[rank]
        arr[..., 0] = torch.where(present, lateral.to(torch.float64) / diag, z)
        arr[..., 1] = torch.where(present, ahead.to(torch.float64) / diag, z)
        arr[..., 2] = torch.where(present, ratio, z)
        arr[..., 3] = torch.where(present, their_boost, z)
    return out[0], out[1]


def _build_scalars(state: GpuObsState) -> torch.Tensor:
    """Build the 26 ego-frame Markov-completing scalars ``(E, S, 26)`` float32."""
    dev = state.device
    E, S = state.E, state.S
    lengths = torch.clamp(state.lengths, min=1).to(torch.float64)  # (E, S)
    alive = state.alive
    heading = state.heading
    heads = state.heads

    out = torch.zeros((E, S, SCALARS_DIM), dtype=torch.float64, device=dev)
    idx = 0

    def put(vals):
        nonlocal idx
        out[..., idx] = vals.to(torch.float64) if torch.is_tensor(vals) else vals
        idx += 1

    max_len = max(1, state.max_length)
    put(lengths / max_len)  # 0
    put(torch.log1p(lengths) / float(np.log1p(max_len)))  # 1
    put((lengths >= state.min_boost_length).to(torch.float64))  # 2
    cost_frames = max(1, state.boost_cost_frames)
    put(state.boost_frames.to(torch.float64) / cost_frames)  # 3
    starv = max(1, state.starvation_max)
    put(torch.clamp(state.frames_since_food.to(torch.float64) / starv, 0, 1))  # 4
    max_frames = max(1, state.max_frames)
    prog = torch.clamp(state.frame.to(torch.float64) / max_frames, 0, 1)  # (E,)
    put(prog[:, None].expand(E, S))  # 5
    alive_count = alive.sum(dim=1).to(torch.float64)  # (E,)
    put((alive_count / max(1, state.max_snakes))[:, None].expand(E, S))  # 6

    # 7 mass_rank_percentile.
    lengths_masked = torch.where(alive, lengths, torch.full_like(lengths, -1.0))
    rank = (lengths_masked[:, :, None] >= lengths_masked[:, None, :]) & alive[:, None, :]
    rank_count = rank.sum(dim=2).to(torch.float64)  # (E, S)
    denom = torch.clamp(alive_count[:, None], min=1)
    percentile = torch.where(alive, rank_count / denom, torch.zeros_like(rank_count))
    put(percentile)  # 7

    # 8-11 wall distances ahead/right/behind/left (cells / 64 capped).
    hc = heads[..., 0].to(torch.float64)  # (E, S)
    hr = heads[..., 1].to(torch.float64)
    dist_left = hc
    dist_right = (state.grid_w - 1) - hc
    dist_up = hr
    dist_down = (state.grid_h - 1) - hr
    wall_dists = torch.stack([dist_up, dist_right, dist_down, dist_left], dim=0)  # (4, E, S)
    h = heading
    ego_ahead = torch.gather(wall_dists, 0, (h % 4)[None]).squeeze(0)
    ego_right = torch.gather(wall_dists, 0, ((h + 1) % 4)[None]).squeeze(0)
    ego_behind = torch.gather(wall_dists, 0, ((h + 2) % 4)[None]).squeeze(0)
    ego_left = torch.gather(wall_dists, 0, ((h + 3) % 4)[None]).squeeze(0)
    put(torch.clamp(ego_ahead / _WALL_DIST_NORM, 0, 1))  # 8
    put(torch.clamp(ego_right / _WALL_DIST_NORM, 0, 1))  # 9
    put(torch.clamp(ego_behind / _WALL_DIST_NORM, 0, 1))  # 10
    put(torch.clamp(ego_left / _WALL_DIST_NORM, 0, 1))  # 11

    # 12-14 nearest food beyond the tactical raster footprint.
    fdx, fdy, fdist = _nearest_food_ego(state)
    put(fdx)  # 12
    put(fdy)  # 13
    put(fdist)  # 14

    # 15-22 nearest-2-enemy-head summaries.
    e1, e2 = _nearest_enemy_summaries(state)
    for arr in (e1, e2):
        put(arr[..., 0])
        put(arr[..., 1])
        put(arr[..., 2])
        put(arr[..., 3])

    # 23 world x, 24 world y, 25 arena-type flag.
    put(torch.clamp(hc / max(1, state.grid_w - 1), 0, 1))  # 23
    put(torch.clamp(hr / max(1, state.grid_h - 1), 0, 1))  # 24
    put(torch.full((E, S), float(state.arena_type_flag), dtype=torch.float64, device=dev))  # 25

    assert idx == SCALARS_DIM, f"filled {idx} scalars, expected {SCALARS_DIM}"
    return out.to(torch.float32)


# ---------------------------------------------------------------------------
# Public build
# ---------------------------------------------------------------------------
def build_observations_gpu(
    state: GpuObsState, mask: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Build the full dual-scale ego observation on the GPU for every agent.

    Semantically identical to ``expand(build_observations(ObsInputs))`` — the
    tactical/strategic planes are byte-identical (after the shared ``/255``
    expansion) and scalars match within ``1e-4``.

    Args:
        state: The GPU world snapshot from :func:`obs_inputs_to_torch`.
        mask: Optional ``(E, S, 6)`` bool safe-action mask to pass through. When
            None an all-True mask is emitted.

    Returns:
        Dict with ``tactical`` ``(E, S, 9, 31, 31)``, ``strategic``
        ``(E, S, 3, 25, 25)``, ``scalars`` ``(E, S, 26)`` float32 tensors and
        ``mask`` ``(E, S, 6)`` bool — all on ``state.device``.
    """
    E, S = state.E, state.S
    tactical = _build_tactical(state)
    strategic = _build_strategic(state)
    scalars = _build_scalars(state)
    if mask is None:
        out_mask = torch.ones((E, S, 6), dtype=torch.bool, device=state.device)
    else:
        out_mask = mask.to(device=state.device, dtype=torch.bool)
    return {
        "tactical": tactical,
        "strategic": strategic,
        "scalars": scalars,
        "mask": out_mask,
    }


def raster_tensors_from_gpu_obs(obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return the network-input triple from a :func:`build_observations_gpu` dict.

    The GPU featurizer already emits the expanded float tensors the network
    consumes, so this is a light passthrough mirroring
    :func:`~src.model.raster_network.raster_tensors_from_obs` for symmetry with
    the NumPy path.

    Args:
        obs: Dict from :func:`build_observations_gpu`.

    Returns:
        Dict with ``tactical`` / ``strategic`` / ``scalars`` float tensors.
    """
    return {
        "tactical": obs["tactical"],
        "strategic": obs["strategic"],
        "scalars": obs["scalars"],
    }
