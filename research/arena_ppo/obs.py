"""GPU-resident and CPU-reference full-arena + heading-aligned ego observation builders.

Observation contract (`obs.py`):
- Arena stream: ``(N, 13, 83, 145)`` float32 allocentric grid at 1 cell = 1 sim cell.
- Ego stream: ``(N, 13, 47, 47)`` float32 heading-aligned crop centered on the seat's
  head ``(row=23, col=23)``, with out-of-arena pixels marked in channel 10 (``wall=1.0``)
  and enemy heading ``(dx, dy)`` rotated into the seat's ego frame.
- Scalars: ``(N, 6)`` float32 ``[length / 2048, boost_available, heading_one_hot(4)]``.
- 2-frame history is carried on head channels only (`own_head_prev`, `enemy_head_prev`).
- Dead seats observe all zeros across all three tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.arena_ppo import ARENA_CHANNELS, ARENA_GRID_H, ARENA_GRID_W, EGO_CROP_SIZE, SCALAR_DIM
from research.arena_ppo.sim import TrainSim

NUM_OBS_CHANNELS = len(ARENA_CHANNELS)
EGO_HEAD_ROW = EGO_CROP_SIZE // 2
EGO_HEAD_COL = EGO_CROP_SIZE // 2
NORM_DIVISOR = 2048.0
INV_NORM_DIVISOR = 1.0 / 2048.0

CH_OWN_TTL = 0
CH_ENEMY_TTL = 1
CH_OWN_HEAD_CURR = 2
CH_OWN_HEAD_PREV = 3
CH_ENEMY_HEAD_CURR = 4
CH_ENEMY_HEAD_PREV = 5
CH_ENEMY_HEAD_DX = 6
CH_ENEMY_HEAD_DY = 7
CH_FOOD_AMBIENT = 8
CH_FOOD_CORPSE = 9
CH_WALL = 10
CH_ENEMY_LENGTH = 11
CH_BOOST_FLAGS = 12

_CARDINAL_NP = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int64)
_EGO_OFFSET_CACHE: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
    {}
)


@dataclass(frozen=True)
class ArenaEgoObs:
    """Batched multi-seat observation tuple `(arena, ego, scalars)`."""

    arena: torch.Tensor
    ego: torch.Tensor
    scalars: torch.Tensor


def _ego_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached `(dcol, drow, cardinal_f32)` tables on `device`.

    For heading ``h`` with forward unit vector ``(fx, fy) = CARDINAL[h]`` and
    rightward unit vector ``(rx, ry) = (-fy, fx)`` in screen coordinates (`+x` right,
    `+y` down), ego pixel ``(r, c)`` lies ``ahead = EGO_HEAD_ROW - r`` cells forward
    and ``lateral = c - EGO_HEAD_COL`` cells to the right.
    """
    key = (device.type, device.index)
    cached = _EGO_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached

    coords = np.arange(EGO_CROP_SIZE, dtype=np.int64)
    ahead = (EGO_HEAD_ROW - coords)[:, None]
    lateral = (coords - EGO_HEAD_COL)[None, :]
    dcol_np = np.empty((4, EGO_CROP_SIZE * EGO_CROP_SIZE), dtype=np.int64)
    drow_np = np.empty((4, EGO_CROP_SIZE * EGO_CROP_SIZE), dtype=np.int64)
    for h_idx, (fx, fy) in enumerate(_CARDINAL_NP.tolist()):
        rx, ry = -fy, fx
        dcol_np[h_idx] = (ahead * fx + lateral * rx).ravel()
        drow_np[h_idx] = (ahead * fy + lateral * ry).ravel()

    dcol = torch.as_tensor(dcol_np, dtype=torch.long, device=device)
    drow = torch.as_tensor(drow_np, dtype=torch.long, device=device)
    cardinal_f32 = torch.as_tensor(_CARDINAL_NP, dtype=torch.float32, device=device)
    _EGO_OFFSET_CACHE[key] = (dcol, drow, cardinal_f32)
    return dcol, drow, cardinal_f32


def _prefix_suffix_enemy_max_i32(per_snake: torch.Tensor) -> torch.Tensor:
    """Compute ``max_{j != s} per_snake[:, j]`` across ``S = 6`` without extra scatters."""
    slots = int(per_snake.shape[1])
    prefix: list[torch.Tensor] = [torch.zeros_like(per_snake[:, 0])]
    for idx in range(1, slots):
        prefix.append(torch.maximum(prefix[idx - 1], per_snake[:, idx - 1]))
    suffix: list[torch.Tensor] = [torch.zeros_like(per_snake[:, 0]) for _ in range(slots)]
    for idx in range(slots - 2, -1, -1):
        suffix[idx] = torch.maximum(suffix[idx + 1], per_snake[:, idx + 1])
    return torch.stack(
        [torch.maximum(prefix[idx], suffix[idx]) for idx in range(slots)],
        dim=1,
    )


def build_arena_ego_obs_from_tensors(
    *,
    ordered_bodies: torch.Tensor | None = None,
    ring_bodies: torch.Tensor | None = None,
    head_ptr: torch.Tensor | None = None,
    seg_count: torch.Tensor,
    length: torch.Tensor,
    alive: torch.Tensor,
    direction: torch.Tensor,
    boosted: torch.Tensor,
    prev_heads: torch.Tensor,
    prev_alive: torch.Tensor,
    food_cells: torch.Tensor,
    food_present: torch.Tensor,
    corpse_food: torch.Tensor,
    min_boost_length: int = 5,
    grid_h: int = ARENA_GRID_H,
    grid_w: int = ARENA_GRID_W,
    ring_chunk: int = 512,
    seat_mask: torch.Tensor | None = None,
) -> ArenaEgoObs:
    """Build exact `(arena, ego, scalars)` tensors on device with zero host syncs.

    Supports both ring-buffer state (`ring_bodies` + `head_ptr` from live ``TrainSim``)
    and contiguous ordered segments (`ordered_bodies` from rollout minibatch gathers).

    Args:
        ordered_bodies: Optional ``(E, S, K, 2)`` segments ordered ``k = 0 .. K-1`` (head..tail).
        ring_bodies: Optional ``(E, S, cap, 2)`` circular ring buffer.
        head_ptr: Ring head pointer ``(E, S)``, required when ``ring_bodies`` is used.
        seg_count: Active segment counts ``(E, S)``.
        length: Snake lengths ``(E, S)``.
        alive: Alive flags ``(E, S)``.
        direction: Heading indices ``(E, S)`` in ``0..3``.
        boosted: Boosted-on-last-step flags ``(E, S)``.
        prev_heads: Previous decision head cells ``(E, S, 2)``.
        prev_alive: Previous decision survival flags ``(E, S)``.
        food_cells: Food cell coordinates ``(E, F, 2)``.
        food_present: Live food mask ``(E, F)``.
        corpse_food: Corpse food class mask ``(E, F)``.
        min_boost_length: Minimum snake length required for legal boost (`5` in v2).
        grid_h: Arena height in cells (`83`).
        grid_w: Arena width in cells (`145`).
        ring_chunk: Segment chunk width to bound peak temporary memory at large ``E``.
        seat_mask: Optional boolean ``(E, S)`` or index selector; when None returns all ``E * S``.

    Returns:
        ``ArenaEgoObs`` with float32 tensors on ``alive.device``.
    """
    device = alive.device
    num_envs, num_snakes = int(alive.shape[0]), int(alive.shape[1])
    board_n = int(grid_h) * int(grid_w)
    dcol_table, drow_table, cardinal_f32 = _ego_tables(device)

    per_snake_ttl = torch.zeros((num_envs, num_snakes, board_n), dtype=torch.int32, device=device)
    seg_i32 = seg_count.to(torch.int32)
    len_i32 = length.to(torch.int32)

    if ordered_bodies is not None:
        max_k = int(ordered_bodies.shape[2])
        for k_start in range(0, max_k, max(1, int(ring_chunk))):
            k_end = min(max_k, k_start + max(1, int(ring_chunk)))
            k_vec = torch.arange(k_start, k_end, dtype=torch.int32, device=device)[None, None, :]
            cells = ordered_bodies[:, :, k_start:k_end, :]
            cx = cells[..., 0].to(torch.long)
            cy = cells[..., 1].to(torch.long)
            valid = (
                alive[..., None]
                & (k_vec < seg_i32[..., None])
                & (cx >= 0)
                & (cx < grid_w)
                & (cy >= 0)
                & (cy < grid_h)
            )
            ttl = (len_i32[..., None] - k_vec).clamp_min(0)
            flat_idx = cy.clamp(0, grid_h - 1) * grid_w + cx.clamp(0, grid_w - 1)
            per_snake_ttl.scatter_reduce_(
                2,
                flat_idx,
                torch.where(valid, ttl, torch.zeros_like(ttl)),
                reduce="amax",
                include_self=True,
            )
        curr_heads = ordered_bodies[:, :, 0, :].to(torch.long)
    else:
        if ring_bodies is None or head_ptr is None:
            raise ValueError("Either ordered_bodies or (ring_bodies, head_ptr) must be provided")
        cap = int(ring_bodies.shape[2])
        env_grid = torch.arange(num_envs, device=device)[:, None, None]
        snake_grid = torch.arange(num_snakes, device=device)[None, :, None]
        hp_long = head_ptr.to(torch.long)
        for k_start in range(0, cap, max(1, int(ring_chunk))):
            k_end = min(cap, k_start + max(1, int(ring_chunk)))
            k_long = torch.arange(k_start, k_end, dtype=torch.long, device=device)[None, None, :]
            k_i32 = k_long.to(torch.int32)
            ring = (hp_long[..., None] - k_long) % cap
            cells = ring_bodies[env_grid, snake_grid, ring]
            cx = cells[..., 0]
            cy = cells[..., 1]
            valid = (
                alive[..., None]
                & (k_i32 < seg_i32[..., None])
                & (cx >= 0)
                & (cx < grid_w)
                & (cy >= 0)
                & (cy < grid_h)
            )
            ttl = (len_i32[..., None] - k_i32).clamp_min(0)
            flat_idx = cy.clamp(0, grid_h - 1) * grid_w + cx.clamp(0, grid_w - 1)
            per_snake_ttl.scatter_reduce_(
                2,
                flat_idx,
                torch.where(valid, ttl, torch.zeros_like(ttl)),
                reduce="amax",
                include_self=True,
            )
        env_2d = torch.arange(num_envs, device=device)[:, None]
        snake_2d = torch.arange(num_snakes, device=device)[None, :]
        curr_heads = ring_bodies[env_2d, snake_2d, hp_long]

    enemy_ttl = _prefix_suffix_enemy_max_i32(per_snake_ttl)

    arena_flat = torch.zeros(
        (num_envs, num_snakes, NUM_OBS_CHANNELS, board_n), dtype=torch.float32, device=device
    )
    arena_flat[:, :, CH_OWN_TTL, :] = per_snake_ttl.to(torch.float32) * INV_NORM_DIVISOR
    arena_flat[:, :, CH_ENEMY_TTL, :] = enemy_ttl.to(torch.float32) * INV_NORM_DIVISOR

    # World-shared ambient and corpse food planes
    fx = food_cells[..., 0].to(torch.long)
    fy = food_cells[..., 1].to(torch.long)
    food_inb = food_present & (fx >= 0) & (fx < grid_w) & (fy >= 0) & (fy < grid_h)
    food_idx = fy.clamp(0, grid_h - 1) * grid_w + fx.clamp(0, grid_w - 1)
    ones_food = torch.ones_like(fx, dtype=torch.int32)
    zeros_food = torch.zeros_like(fx, dtype=torch.int32)

    amb_plane_i32 = torch.zeros((num_envs, board_n), dtype=torch.int32, device=device)
    amb_plane_i32.scatter_reduce_(
        1,
        food_idx,
        torch.where(food_inb & (~corpse_food), ones_food, zeros_food),
        reduce="amax",
        include_self=True,
    )
    corpse_plane_i32 = torch.zeros((num_envs, board_n), dtype=torch.int32, device=device)
    corpse_plane_i32.scatter_reduce_(
        1,
        food_idx,
        torch.where(food_inb & corpse_food, ones_food, zeros_food),
        reduce="amax",
        include_self=True,
    )
    arena_flat[:, :, CH_FOOD_AMBIENT, :] = amb_plane_i32.to(torch.float32)[:, None, :]
    arena_flat[:, :, CH_FOOD_CORPSE, :] = corpse_plane_i32.to(torch.float32)[:, None, :]

    # Per-snake head channels (`O(E * S)` single-cell updates, zero host syncs)
    env_idx = torch.arange(num_envs, device=device)
    dir_clamped = direction.to(torch.long).clamp(0, 3)
    head_dirs_f32 = cardinal_f32[dir_clamped]
    len_norm_f32 = length.to(torch.float32) * INV_NORM_DIVISOR
    one_vec = torch.ones((num_envs,), dtype=torch.float32, device=device)

    for j_slot in range(num_snakes):
        hx = curr_heads[:, j_slot, 0]
        hy = curr_heads[:, j_slot, 1]
        inb_c = (
            alive[:, j_slot]
            & (seg_count[:, j_slot] > 0)
            & (hx >= 0)
            & (hx < grid_w)
            & (hy >= 0)
            & (hy < grid_h)
        )
        flat_c = hy.clamp(0, grid_h - 1) * grid_w + hx.clamp(0, grid_w - 1)

        px = prev_heads[:, j_slot, 0].to(torch.long)
        py = prev_heads[:, j_slot, 1].to(torch.long)
        inb_p = (
            alive[:, j_slot]
            & prev_alive[:, j_slot]
            & (px >= 0)
            & (px < grid_w)
            & (py >= 0)
            & (py < grid_h)
        )
        flat_p = py.clamp(0, grid_h - 1) * grid_w + px.clamp(0, grid_w - 1)

        arena_flat[env_idx, j_slot, CH_OWN_HEAD_CURR, flat_c] = torch.where(
            inb_c, one_vec, arena_flat[env_idx, j_slot, CH_OWN_HEAD_CURR, flat_c]
        )
        arena_flat[env_idx, j_slot, CH_OWN_HEAD_PREV, flat_p] = torch.where(
            inb_p, one_vec, arena_flat[env_idx, j_slot, CH_OWN_HEAD_PREV, flat_p]
        )

        inb_boost = inb_c & boosted[:, j_slot]
        for s_slot in range(num_snakes):
            arena_flat[env_idx, s_slot, CH_BOOST_FLAGS, flat_c] = torch.where(
                inb_boost, one_vec, arena_flat[env_idx, s_slot, CH_BOOST_FLAGS, flat_c]
            )
            if s_slot == j_slot:
                continue
            arena_flat[env_idx, s_slot, CH_ENEMY_HEAD_CURR, flat_c] = torch.where(
                inb_c, one_vec, arena_flat[env_idx, s_slot, CH_ENEMY_HEAD_CURR, flat_c]
            )
            arena_flat[env_idx, s_slot, CH_ENEMY_HEAD_PREV, flat_p] = torch.where(
                inb_p, one_vec, arena_flat[env_idx, s_slot, CH_ENEMY_HEAD_PREV, flat_p]
            )
            arena_flat[env_idx, s_slot, CH_ENEMY_HEAD_DX, flat_c] = torch.where(
                inb_c,
                head_dirs_f32[:, j_slot, 0],
                arena_flat[env_idx, s_slot, CH_ENEMY_HEAD_DX, flat_c],
            )
            arena_flat[env_idx, s_slot, CH_ENEMY_HEAD_DY, flat_c] = torch.where(
                inb_c,
                head_dirs_f32[:, j_slot, 1],
                arena_flat[env_idx, s_slot, CH_ENEMY_HEAD_DY, flat_c],
            )
            arena_flat[env_idx, s_slot, CH_ENEMY_LENGTH, flat_c] = torch.where(
                inb_c, len_norm_f32[:, j_slot], arena_flat[env_idx, s_slot, CH_ENEMY_LENGTH, flat_c]
            )

    # Zero out dead seats across all channels
    arena_flat = torch.where(alive[:, :, None, None], arena_flat, torch.zeros_like(arena_flat))

    total_rows = num_envs * num_snakes
    arena_rows_flat = arena_flat.reshape(total_rows, NUM_OBS_CHANNELS, board_n)
    alive_rows = alive.reshape(total_rows)
    heads_rows = curr_heads.reshape(total_rows, 2)
    dirs_rows = dir_clamped.reshape(total_rows)
    length_rows = length.reshape(total_rows)

    if seat_mask is not None:
        arena_rows_flat = arena_rows_flat[seat_mask]
        alive_rows = alive_rows[seat_mask]
        heads_rows = heads_rows[seat_mask]
        dirs_rows = dirs_rows[seat_mask]
        length_rows = length_rows[seat_mask]

    n_out = int(alive_rows.shape[0])

    # Heading-aligned 47x47 ego crop via single gather from `arena_rows_flat`
    wx = heads_rows[:, 0:1] + dcol_table[dirs_rows]
    wy = heads_rows[:, 1:2] + drow_table[dirs_rows]
    ego_inb = (wx >= 0) & (wx < grid_w) & (wy >= 0) & (wy < grid_h)
    ego_idx = wy.clamp(0, grid_h - 1) * grid_w + wx.clamp(0, grid_w - 1)

    ego_flat = arena_rows_flat.gather(2, ego_idx[:, None, :].expand(-1, NUM_OBS_CHANNELS, -1))
    ego_flat = torch.where(ego_inb[:, None, :], ego_flat, torch.zeros_like(ego_flat))
    ego_flat[:, CH_WALL, :] = torch.where(
        alive_rows[:, None] & (~ego_inb),
        torch.ones_like(ego_flat[:, CH_WALL, :]),
        ego_flat[:, CH_WALL, :],
    )

    # Rotate allocentric enemy `(dx, dy)` into the heading-aligned ego screen frame:
    # forward = `(fx, fy)` maps to `(0, -1)` (up); right = `(-fy, fx)` maps to `(+1, 0)`.
    fwd = cardinal_f32[dirs_rows]
    fx_ego = fwd[:, 0:1]
    fy_ego = fwd[:, 1:2]
    raw_dx = ego_flat[:, CH_ENEMY_HEAD_DX, :]
    raw_dy = ego_flat[:, CH_ENEMY_HEAD_DY, :]
    ego_dx = -raw_dx * fy_ego + raw_dy * fx_ego
    ego_dy = -(raw_dx * fx_ego + raw_dy * fy_ego)
    # Normalize `-0.0` to `+0.0` for bit-exact parity across backends
    ego_flat[:, CH_ENEMY_HEAD_DX, :] = ego_dx + 0.0
    ego_flat[:, CH_ENEMY_HEAD_DY, :] = ego_dy + 0.0
    ego_flat = torch.where(alive_rows[:, None, None], ego_flat, torch.zeros_like(ego_flat))

    # Scalars `(N, 6)`: `[length / 2048, boost_available, heading_one_hot(4)]`
    len_scalar = (length_rows.to(torch.float32) * INV_NORM_DIVISOR).unsqueeze(1)
    boost_avail = (
        (alive_rows & (length_rows >= int(min_boost_length))).to(torch.float32).unsqueeze(1)
    )
    heading_oh = F.one_hot(dirs_rows, num_classes=4).to(torch.float32)
    scalars = torch.cat((len_scalar, boost_avail, heading_oh), dim=1)
    scalars = torch.where(alive_rows[:, None], scalars, torch.zeros_like(scalars))

    arena_out = arena_rows_flat.reshape(n_out, NUM_OBS_CHANNELS, grid_h, grid_w)
    ego_out = ego_flat.reshape(n_out, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE)
    assert scalars.shape == (n_out, SCALAR_DIM)
    return ArenaEgoObs(arena=arena_out, ego=ego_out, scalars=scalars)


def build_arena_ego_obs_gpu(
    train_sim: TrainSim,
    *,
    env_slice: slice | None = None,
    ring_chunk: int = 512,
) -> ArenaEgoObs:
    """Build `(arena, ego, scalars)` directly from a prepared ``TrainSim`` on device.

    Args:
        train_sim: Prepared ``TrainSim`` wrapper.
        env_slice: Optional slice along world axis ``0 .. E`` for chunked forward passes.
        ring_chunk: Ring buffer segment scan chunk size.

    Returns:
        ``ArenaEgoObs`` on ``train_sim.device``.
    """
    sim = train_sim.sim
    sl = slice(None) if env_slice is None else env_slice
    return build_arena_ego_obs_from_tensors(
        ring_bodies=sim.bodies[sl],
        head_ptr=sim.head_ptr[sl],
        seg_count=sim.seg_count[sl],
        length=sim.length[sl],
        alive=sim.alive[sl],
        direction=sim.direction[sl],
        boosted=sim._boosted_this_step[sl],
        prev_heads=train_sim.prev_heads[sl],
        prev_alive=train_sim.prev_alive[sl],
        food_cells=sim.food_cells[sl],
        food_present=sim.food_present[sl],
        corpse_food=sim.corpse_food[sl],
        min_boost_length=train_sim.min_boost_length,
        grid_h=train_sim.grid_h,
        grid_w=train_sim.grid_w,
        ring_chunk=ring_chunk,
    )


def build_arena_ego_obs_cpu(
    ref_sim: Any,
    prev_heads: np.ndarray,
    prev_alive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact NumPy CPU reference `(arena, ego, scalars)` from `GridBatchSim` or arrays.

    Computes every channel via explicit NumPy array logic independent of PyTorch
    ``scatter_reduce_`` / ``gather`` kernels so GPU/MPS parity tests validate both
    indexing semantics and IEEE-754 float32 values.

    Args:
        ref_sim: CPU ``GridBatchSim``, ``BatchSim``, or ``DeviceSim`` state.
        prev_heads: ``(E, S, 2)`` integer array of previous decision heads.
        prev_alive: ``(E, S)`` boolean array of previous survival flags.

    Returns:
        Tuple ``(arena, ego, scalars)`` of float32 NumPy arrays with shapes
        ``(E * S, 13, 83, 145)``, ``(E * S, 13, 47, 47)``, and ``(E * S, 6)``.
    """
    bodies = np.asarray(
        (
            ref_sim.bodies.detach().cpu().numpy()
            if isinstance(ref_sim.bodies, torch.Tensor)
            else ref_sim.bodies
        ),
        dtype=np.int64,
    )
    head_ptr = np.asarray(
        (
            ref_sim.head_ptr.detach().cpu().numpy()
            if isinstance(ref_sim.head_ptr, torch.Tensor)
            else ref_sim.head_ptr
        ),
        dtype=np.int64,
    )
    seg_count = np.asarray(
        (
            ref_sim.seg_count.detach().cpu().numpy()
            if isinstance(ref_sim.seg_count, torch.Tensor)
            else ref_sim.seg_count
        ),
        dtype=np.int64,
    )
    length = np.asarray(
        (
            ref_sim.length.detach().cpu().numpy()
            if isinstance(ref_sim.length, torch.Tensor)
            else ref_sim.length
        ),
        dtype=np.int64,
    )
    alive = np.asarray(
        (
            ref_sim.alive.detach().cpu().numpy()
            if isinstance(ref_sim.alive, torch.Tensor)
            else ref_sim.alive
        ),
        dtype=bool,
    )
    direction = np.asarray(
        (
            ref_sim.direction.detach().cpu().numpy()
            if isinstance(ref_sim.direction, torch.Tensor)
            else ref_sim.direction
        ),
        dtype=np.int64,
    )
    boosted = np.asarray(
        (
            ref_sim._boosted_this_step.detach().cpu().numpy()
            if isinstance(ref_sim._boosted_this_step, torch.Tensor)
            else ref_sim._boosted_this_step
        ),
        dtype=bool,
    )
    prev_heads_np = np.asarray(prev_heads, dtype=np.int64)
    prev_alive_np = np.asarray(prev_alive, dtype=bool)

    num_envs, num_snakes, cap, _ = bodies.shape
    grid_h = int(ref_sim.grid_h)
    grid_w = int(ref_sim.grid_w)
    min_boost_length = int(ref_sim.cfg.min_boost_length)

    arena = np.zeros((num_envs, num_snakes, NUM_OBS_CHANNELS, grid_h, grid_w), dtype=np.float32)
    ego = np.zeros(
        (num_envs, num_snakes, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE), dtype=np.float32
    )
    scalars = np.zeros((num_envs, num_snakes, SCALAR_DIM), dtype=np.float32)

    coords = np.arange(EGO_CROP_SIZE, dtype=np.int64)
    ahead = (EGO_HEAD_ROW - coords)[:, None]
    lateral = (coords - EGO_HEAD_COL)[None, :]

    has_device_food = isinstance(getattr(ref_sim, "food_present", None), torch.Tensor)
    if has_device_food:
        dev_food_cells = ref_sim.food_cells.detach().cpu().numpy()
        dev_food_present = ref_sim.food_present.detach().cpu().numpy()
        dev_corpse_food = ref_sim.corpse_food.detach().cpu().numpy()

    for env in range(num_envs):
        ttl_per_snake = np.zeros((num_snakes, grid_h, grid_w), dtype=np.int32)
        for j_slot in range(num_snakes):
            if not alive[env, j_slot]:
                continue
            n_seg = int(seg_count[env, j_slot])
            if n_seg <= 0:
                continue
            s_len = int(length[env, j_slot])
            hp = int(head_ptr[env, j_slot])
            k_arr = np.arange(n_seg, dtype=np.int64)
            rings = (hp - k_arr) % cap
            pts = bodies[env, j_slot, rings]
            cx_arr, cy_arr = pts[:, 0], pts[:, 1]
            inb_seg = (cx_arr >= 0) & (cx_arr < grid_w) & (cy_arr >= 0) & (cy_arr < grid_h)
            if np.any(inb_seg):
                ttl_vals = np.maximum(s_len - k_arr[inb_seg], 0).astype(np.int32)
                np.maximum.at(ttl_per_snake[j_slot], (cy_arr[inb_seg], cx_arr[inb_seg]), ttl_vals)

        amb_plane = np.zeros((grid_h, grid_w), dtype=np.float32)
        corpse_plane = np.zeros((grid_h, grid_w), dtype=np.float32)
        if has_device_food:
            f_pres = dev_food_present[env]
            if np.any(f_pres):
                f_pts = dev_food_cells[env, f_pres]
                f_corpse = dev_corpse_food[env, f_pres]
                fx_arr, fy_arr = f_pts[:, 0], f_pts[:, 1]
                f_inb = (fx_arr >= 0) & (fx_arr < grid_w) & (fy_arr >= 0) & (fy_arr < grid_h)
                amb_sel = f_inb & (~f_corpse)
                cor_sel = f_inb & f_corpse
                amb_plane[fy_arr[amb_sel], fx_arr[amb_sel]] = 1.0
                corpse_plane[fy_arr[cor_sel], fx_arr[cor_sel]] = 1.0
        else:
            corpse_set = ref_sim.corpse_cells[env]
            for cell in ref_sim.food_cells[env]:
                fx, fy = int(cell[0]), int(cell[1])
                if 0 <= fx < grid_w and 0 <= fy < grid_h:
                    if (fx, fy) in corpse_set:
                        corpse_plane[fy, fx] = 1.0
                    else:
                        amb_plane[fy, fx] = 1.0

        for s_slot in range(num_snakes):
            if not alive[env, s_slot]:
                continue
            arena[env, s_slot, CH_OWN_TTL] = ttl_per_snake[s_slot].astype(np.float32) * np.float32(
                INV_NORM_DIVISOR
            )
            enemy_ttl_max = np.zeros((grid_h, grid_w), dtype=np.int32)
            for j_slot in range(num_snakes):
                if j_slot != s_slot:
                    np.maximum(enemy_ttl_max, ttl_per_snake[j_slot], out=enemy_ttl_max)
            arena[env, s_slot, CH_ENEMY_TTL] = enemy_ttl_max.astype(np.float32) * np.float32(
                INV_NORM_DIVISOR
            )
            arena[env, s_slot, CH_FOOD_AMBIENT] = amb_plane
            arena[env, s_slot, CH_FOOD_CORPSE] = corpse_plane

            for j_slot in range(num_snakes):
                if not alive[env, j_slot]:
                    continue
                hp_j = int(head_ptr[env, j_slot])
                hx, hy = int(bodies[env, j_slot, hp_j, 0]), int(bodies[env, j_slot, hp_j, 1])
                inb_c = seg_count[env, j_slot] > 0 and 0 <= hx < grid_w and 0 <= hy < grid_h
                px, py = int(prev_heads_np[env, j_slot, 0]), int(prev_heads_np[env, j_slot, 1])
                inb_p = prev_alive_np[env, j_slot] and 0 <= px < grid_w and 0 <= py < grid_h

                if inb_c and boosted[env, j_slot]:
                    arena[env, s_slot, CH_BOOST_FLAGS, hy, hx] = 1.0

                if j_slot == s_slot:
                    if inb_c:
                        arena[env, s_slot, CH_OWN_HEAD_CURR, hy, hx] = 1.0
                    if inb_p:
                        arena[env, s_slot, CH_OWN_HEAD_PREV, py, px] = 1.0
                else:
                    if inb_c:
                        d_j = int(direction[env, j_slot]) & 3
                        dx_j, dy_j = _CARDINAL_NP[d_j]
                        arena[env, s_slot, CH_ENEMY_HEAD_CURR, hy, hx] = 1.0
                        arena[env, s_slot, CH_ENEMY_HEAD_DX, hy, hx] = np.float32(dx_j)
                        arena[env, s_slot, CH_ENEMY_HEAD_DY, hy, hx] = np.float32(dy_j)
                        arena[env, s_slot, CH_ENEMY_LENGTH, hy, hx] = np.float32(
                            length[env, j_slot]
                        ) * np.float32(INV_NORM_DIVISOR)
                    if inb_p:
                        arena[env, s_slot, CH_ENEMY_HEAD_PREV, py, px] = 1.0

            hp_s = int(head_ptr[env, s_slot])
            hx_s, hy_s = int(bodies[env, s_slot, hp_s, 0]), int(bodies[env, s_slot, hp_s, 1])
            d_s = int(direction[env, s_slot]) & 3
            fx_s, fy_s = int(_CARDINAL_NP[d_s, 0]), int(_CARDINAL_NP[d_s, 1])
            rx_s, ry_s = -fy_s, fx_s
            wx = hx_s + ahead * fx_s + lateral * rx_s
            wy = hy_s + ahead * fy_s + lateral * ry_s
            inb_ego = (wx >= 0) & (wx < grid_w) & (wy >= 0) & (wy < grid_h)
            wx_c = np.clip(wx, 0, grid_w - 1)
            wy_c = np.clip(wy, 0, grid_h - 1)

            cropped = arena[env, s_slot, :, wy_c, wx_c]
            # NumPy advanced indexing on axes `(1, 2)` yields `(47, 47, C)`; transpose to `(C, 47, 47)`
            if cropped.shape[0] != NUM_OBS_CHANNELS:
                cropped = np.transpose(cropped, (2, 0, 1))
            cropped = np.where(inb_ego[None, :, :], cropped, np.float32(0.0))
            cropped[CH_WALL] = np.where(~inb_ego, np.float32(1.0), np.float32(0.0))
            raw_dx = cropped[CH_ENEMY_HEAD_DX].copy()
            raw_dy = cropped[CH_ENEMY_HEAD_DY].copy()
            cropped[CH_ENEMY_HEAD_DX] = (
                -raw_dx * np.float32(fy_s) + raw_dy * np.float32(fx_s)
            ) + np.float32(0.0)
            cropped[CH_ENEMY_HEAD_DY] = (
                -(raw_dx * np.float32(fx_s) + raw_dy * np.float32(fy_s))
            ) + np.float32(0.0)
            ego[env, s_slot] = cropped

            scalars[env, s_slot, 0] = np.float32(length[env, s_slot]) * np.float32(INV_NORM_DIVISOR)
            scalars[env, s_slot, 1] = (
                np.float32(1.0) if length[env, s_slot] >= min_boost_length else np.float32(0.0)
            )
            scalars[env, s_slot, 2 + d_s] = np.float32(1.0)

    total = num_envs * num_snakes
    return (
        arena.reshape(total, NUM_OBS_CHANNELS, grid_h, grid_w),
        ego.reshape(total, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE),
        scalars.reshape(total, SCALAR_DIM),
    )
