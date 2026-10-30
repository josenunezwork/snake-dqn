"""Sparse full-arena observations for the v1 spatial-student pilot.

Slots are cyclically reordered so the hero is slot zero. Planes 0:6 are
per-snake occupancy (overlaps are preserved); 6:12 are remaining ordered-body
clearance ``(logical_length - head_offset) / 10400``, maximized on self-overlap.
This is body order plus current growth debt, NOT exact future vacancy: future
growth, boost and newly laid segments are unknown. Planes 12/13 are ambient and
corpse food, 14 marks the arena-edge cells, 15 is ``(x/(W-1)+y/(H-1))/2``.
The edge plane describes the boundary; edge cells themselves remain playable.

Six scalars per slot: logical length/10400, direction index/3 (U,R,D,L),
growth debt/10400, boost-clock/cadence, hunger-clock/5000, alive. No RNG state
is observed. Sparse head-first int16 cell lists are saved; dense float planes
exist only for the current inference/training microbatch, never a T256 buffer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np
import torch

OBSERVATION_VERSION = "spatial-student-v1-16p36s"
LENGTH_SCALE = 10400.0
N_SNAKES = 6
_META = ("grid_h", "grid_w", "boost_cost_frames", "version")
_FIELDS = ("body_count", "length", "alive", "direction", "boost_frames", "hunger")


def _tensor(value: Any, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.as_tensor(value, device=device)


def compact_from_sim(
    sim: Any,
    env_indices: Sequence[int] | torch.Tensor | None = None,
    hero: int = 0,
    *,
    padded: bool = False,
) -> dict[str, Any]:
    """Extract selected GridBatchSim/DeviceSim rows without copying GPU bodies to CPU."""
    if sim.S != N_SNAKES or not 0 <= hero < N_SNAKES:
        raise ValueError("the pilot requires six snakes and a valid hero slot")
    if sim.cfg.arena_type != "rectangular":
        raise ValueError("the pilot supports rectangular arenas only")
    source = _tensor(sim.bodies)
    device = source.device
    ids = (
        torch.arange(sim.E, device=device)
        if env_indices is None
        else _tensor(env_indices, device).long().reshape(-1)
    )
    if ids.numel() == 0:
        raise ValueError("an observation batch cannot be empty")
    slots = (torch.arange(N_SNAKES, device=device) + hero) % N_SNAKES
    counts = _tensor(sim.seg_count, device)[ids][:, slots].long()
    if padded:
        from research.gpu_sim_spike import r2_cache

        span = r2_cache.window(sim)
    else:
        span = max(1, int(counts.max().item()))
    if span > sim.cap or max(sim.grid_h, sim.grid_w) > 32767:
        raise ValueError("body count/cell coordinates exceed compact storage")
    hp = _tensor(sim.head_ptr, device)[ids][:, slots].long()
    offsets = torch.arange(span, device=device)
    rings = (hp[..., None] - offsets) % sim.cap
    bodies = source[ids[:, None, None], slots[None, :, None], rings]
    bodies = torch.where((offsets < counts[..., None])[..., None], bodies, 0)
    out = {
        "version": OBSERVATION_VERSION,
        "grid_h": int(sim.grid_h),
        "grid_w": int(sim.grid_w),
        "boost_cost_frames": int(sim.cfg.boost_length_cost_frames),
        "body_cells": bodies.to(torch.int16),
        "body_count": counts.to(torch.int32),
    }
    for field, attr in zip(
        _FIELDS[1:], ("length", "alive", "direction", "boost_frames", "frames_since_food")
    ):
        out[field] = _tensor(getattr(sim, attr), device)[ids][:, slots].clone()
    out["heads"] = bodies[:, 0, 0].to(torch.int16)
    out["frame"] = _tensor(sim.frame, device)[ids].clone()
    if isinstance(sim.food_cells, torch.Tensor):
        present = sim.food_present[ids]
        # Fixed simulator food buffers may contain holes; preserve them sparsely.
        used = torch.where(present, torch.arange(present.shape[1], device=device) + 1, 0)
        food_span = present.shape[1] if padded else max(1, int(used.max().item()))
        out["food_cells"] = sim.food_cells[ids, :food_span].to(torch.int16)
        out["food_kind"] = torch.where(
            present[:, :food_span], 1 + sim.corpse_food[ids, :food_span].to(torch.uint8), 0
        )
    else:
        rows = ids.cpu().tolist()
        food_span = max(1, max(len(sim.food_cells[e]) for e in rows))
        food = torch.zeros((len(rows), food_span, 2), dtype=torch.int16)
        kind = torch.zeros((len(rows), food_span), dtype=torch.uint8)
        for row, env in enumerate(rows):
            cells = sim.food_cells[env]
            if cells:
                food[row, : len(cells)] = torch.tensor(cells, dtype=torch.int16)
                kind[row, : len(cells)] = torch.tensor(
                    [2 if c in sim.corpse_cells[env] else 1 for c in cells], dtype=torch.uint8
                )
        out["food_cells"], out["food_kind"] = food, kind
    return out


def compact_from_live(game_state: Any, hero: int = 0) -> dict[str, Any]:
    """Independently extract real Snake/FoodManager state using cell coordinates.

    Like the existing ego live adapter, learned wrappers must advance each
    snake's hunger clock; scripted snakes do not do reward bookkeeping.
    """
    from src.core.mechanics_constants import cell_index
    from src.simd_env.live_adapter import _heading_index

    snakes = list(game_state.snakes)
    if len(snakes) != N_SNAKES or not 0 <= hero < N_SNAKES:
        raise ValueError("the pilot requires six snakes and a valid hero slot")
    snakes = snakes[hero:] + snakes[:hero]
    fm = game_state.food_manager
    segment_size = int(snakes[0].segment_size)
    width = int(getattr(game_state, "_game_width", fm.game_width))
    height = int(getattr(game_state, "_game_height", fm.game_height))
    if width % segment_size or height % segment_size:
        raise ValueError("arena dimensions must be multiples of segment size")
    cfg = getattr(game_state, "cfg", None)
    if cfg is None:
        from src.core.game_config import GameConfig

        cadence = int(GameConfig.BOOST_LENGTH_COST_FRAMES)
    else:
        if cfg.arena_type != "rectangular":
            raise ValueError("the pilot supports rectangular arenas only")
        cadence = int(cfg.boost_length_cost_frames)
    cells = [[cell_index(p, segment_size) for p in snake.segments] for snake in snakes]
    span = max(1, max(map(len, cells)))
    bodies = np.zeros((1, N_SNAKES, span, 2), dtype=np.int16)
    for slot, row in enumerate(cells):
        if row:
            arr = np.asarray(row, dtype=np.int64)
            if bool(snakes[slot].is_alive) and (
                (arr < 0).any()
                or (arr[:, 0] >= width // segment_size).any()
                or (arr[:, 1] >= height // segment_size).any()
            ):
                raise ValueError("living snake body lies outside the arena")
            bodies[0, slot, : len(row)] = arr
    food = [cell_index(p, segment_size) for p in fm.food]
    food_cells = np.zeros((1, max(1, len(food)), 2), dtype=np.int16)
    food_kind = np.zeros((1, max(1, len(food))), dtype=np.uint8)
    corpses = getattr(fm, "_corpse_positions", set())
    for i, (pos, cell) in enumerate(zip(fm.food, food)):
        if not (0 <= cell[0] < width // segment_size and 0 <= cell[1] < height // segment_size):
            raise ValueError("food lies outside the arena")
        food_cells[0, i] = cell
        food_kind[0, i] = 2 if pos in corpses else 1
    result = {
        "version": OBSERVATION_VERSION,
        "grid_h": height // segment_size,
        "grid_w": width // segment_size,
        "boost_cost_frames": cadence,
        "body_cells": torch.from_numpy(bodies),
        "body_count": torch.tensor([[len(row) for row in cells]], dtype=torch.int32),
        "length": torch.tensor([[int(s.length) for s in snakes]]),
        "alive": torch.tensor([[bool(s.is_alive) for s in snakes]]),
        "direction": torch.tensor([[_heading_index(s.direction) for s in snakes]]),
        "boost_frames": torch.tensor([[int(getattr(s, "boost_frames", 0)) for s in snakes]]),
        "hunger": torch.tensor([[int(getattr(s, "frames_since_food", 0)) for s in snakes]]),
        "frame": torch.tensor([int(game_state.frame)]),
        "heads": torch.from_numpy(bodies[:, 0, 0].copy()),
        "food_cells": torch.from_numpy(food_cells),
        "food_kind": torch.from_numpy(food_kind),
    }
    return result


def compact_take(compact: dict[str, Any], indices: Any) -> dict[str, Any]:
    """Select rows while preserving archive metadata and a batch dimension."""
    result = {key: compact[key] for key in _META}
    for key, value in compact.items():
        if key not in _META:
            idx = torch.as_tensor(indices, device=value.device, dtype=torch.long).reshape(-1)
            result[key] = value.index_select(0, idx)
    return result


def compact_cat(compacts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate sparse tick shards, padding only body/food integer lists."""
    rows = list(compacts)
    if not rows:
        raise ValueError("cannot concatenate no observations")
    if any(any(row[k] != rows[0][k] for k in _META) for row in rows):
        raise ValueError("incompatible compact observation metadata")
    result = {key: rows[0][key] for key in _META}
    for key in rows[0]:
        if key in _META:
            continue
        values = [row[key] for row in rows]
        if key in ("body_cells", "food_cells", "food_kind"):
            axis = 2 if key == "body_cells" else 1
            span = max(value.shape[axis] for value in values)
            padded = []
            for value in values:
                shape = list(value.shape)
                shape[axis] = span
                dest = value.new_zeros(shape)
                slices = tuple(slice(0, n) for n in value.shape)
                dest[slices] = value
                padded.append(dest)
            values = padded
        result[key] = torch.cat(values)
    return result


def build_observation(
    compact: dict[str, Any], device: str | torch.device = "cpu", *, async_checks: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-native scatter reconstruction of one sparse microbatch."""
    if compact["version"] != OBSERVATION_VERSION:
        raise ValueError("unknown spatial observation version")
    h, w = int(compact["grid_h"]), int(compact["grid_w"])
    body = compact["body_cells"].to(device=device, dtype=torch.long)
    n, snakes, span, _ = body.shape
    if snakes != N_SNAKES or min(h, w, int(compact["boost_cost_frames"])) < 1:
        raise ValueError("invalid observation geometry")
    count = compact["body_count"].to(device=device, dtype=torch.long)
    alive = compact["alive"].to(device=device, dtype=torch.bool)
    length = compact["length"].to(device=device, dtype=torch.float32)
    order = torch.arange(span, device=device)[None, None, :]
    valid = (order < count[..., None]) & alive[..., None]
    x, y = body[..., 0], body[..., 1]
    bad = valid & ((x < 0) | (x >= w) | (y < 0) | (y >= h))
    if async_checks:
        torch._assert_async(~bad.any(), "living snake body lies outside the arena")
    elif bool(bad.any()):
        raise ValueError("living snake body lies outside the arena")
    idx = (y * w + x).clamp(0, h * w - 1)
    occ = torch.zeros((n, N_SNAKES, h * w), device=device)
    occ.scatter_reduce_(2, idx, valid.float(), reduce="amax", include_self=True)
    clearance = torch.where(valid, (length[..., None] - order) / LENGTH_SCALE, 0)
    ordered = torch.zeros_like(occ)
    ordered.scatter_reduce_(2, idx, clearance, reduce="amax", include_self=True)
    food = compact["food_cells"].to(device=device, dtype=torch.long)
    kind = compact["food_kind"].to(device=device)
    fx, fy = food[..., 0], food[..., 1]
    bad_food = ((kind > 0) & ((fx < 0) | (fx >= w) | (fy < 0) | (fy >= h))).any()
    if async_checks:
        torch._assert_async(~bad_food, "food lies outside the arena")
    elif bool(bad_food):
        raise ValueError("food lies outside the arena")
    food_idx = (fy * w + fx).clamp(0, h * w - 1)
    food_planes = torch.zeros((n, 2, h * w), device=device)
    for k in range(2):
        food_planes[:, k].scatter_reduce_(
            1, food_idx, (kind == k + 1).float(), reduce="amax", include_self=True
        )
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    boundary = ((xx == 0) | (yy == 0) | (xx == w - 1) | (yy == h - 1)).float()
    geometry = (xx.float() / max(1, w - 1) + yy.float() / max(1, h - 1)) / 2
    fixed = torch.stack((boundary, geometry))[None].expand(n, -1, -1, -1)
    planes = torch.cat(
        (
            occ.reshape(n, 6, h, w),
            ordered.reshape(n, 6, h, w),
            food_planes.reshape(n, 2, h, w),
            fixed,
        ),
        dim=1,
    )
    scalar_fields = (
        length / LENGTH_SCALE,
        compact["direction"].to(device=device).float() / 3,
        (length - count).clamp_min(0) / LENGTH_SCALE,
        compact["boost_frames"].to(device=device).float() / compact["boost_cost_frames"],
        compact["hunger"].to(device=device).float() / 5000,
        alive.float(),
    )
    scalars = torch.stack(scalar_fields, dim=2).reshape(n, 36)
    heads = compact["heads"].to(device=device, dtype=torch.long)
    # Dead heroes may retain a fatal out-of-board head; they are not acted on.
    heads = torch.stack((heads[:, 0].clamp(0, w - 1), heads[:, 1].clamp(0, h - 1)), dim=1)
    return planes, scalars, heads


def build_observation_cpu_reference(
    compact: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent scalar NumPy oracle; does not call the GPU scatter builder."""
    h, w = int(compact["grid_h"]), int(compact["grid_w"])
    data = {k: v.detach().cpu().numpy() for k, v in compact.items() if k not in _META}
    n = len(data["length"])
    p = np.zeros((n, 16, h, w), dtype=np.float32)
    s = np.zeros((n, 6, 6), dtype=np.float32)
    for row in range(n):
        for slot in range(6):
            length, count = int(data["length"][row, slot]), int(data["body_count"][row, slot])
            alive = bool(data["alive"][row, slot])
            if alive:
                for offset in range(count):
                    x, y = map(int, data["body_cells"][row, slot, offset])
                    if not 0 <= x < w or not 0 <= y < h:
                        raise ValueError("living snake body lies outside the arena")
                    p[row, slot, y, x] = 1
                    ttl = np.float32(length - offset) / np.float32(LENGTH_SCALE)
                    p[row, slot + 6, y, x] = max(p[row, slot + 6, y, x], ttl)
            s[row, slot] = (
                np.float32(length) / np.float32(LENGTH_SCALE),
                np.float32(data["direction"][row, slot]) / np.float32(3),
                np.float32(max(0, length - count)) / np.float32(LENGTH_SCALE),
                np.float32(data["boost_frames"][row, slot])
                / np.float32(compact["boost_cost_frames"]),
                np.float32(data["hunger"][row, slot]) / np.float32(5000),
                np.float32(alive),
            )
        for (x, y), kind in zip(data["food_cells"][row], data["food_kind"][row]):
            if kind:
                if not 0 <= x < w or not 0 <= y < h:
                    raise ValueError("food lies outside the arena")
                p[row, 11 + kind, y, x] = 1
        for y in range(h):
            for x in range(w):
                p[row, 14, y, x] = float(x in (0, w - 1) or y in (0, h - 1))
                p[row, 15, y, x] = (
                    np.float32(x) / np.float32(max(1, w - 1))
                    + np.float32(y) / np.float32(max(1, h - 1))
                ) / np.float32(2)
    heads = data["heads"].astype(np.int64).copy()
    heads[:, 0] = heads[:, 0].clip(0, w - 1)
    heads[:, 1] = heads[:, 1].clip(0, h - 1)
    return p, s.reshape(n, 36), heads


@lru_cache(maxsize=8)
def _numpy_geometry(h: int, w: int) -> np.ndarray:
    """Cache only immutable arena geometry, never simulator-dependent state."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    boundary = ((xx == 0) | (yy == 0) | (xx == w - 1) | (yy == h - 1)).astype(np.float32)
    geometry = (
        xx.astype(np.float32) / np.float32(max(1, w - 1))
        + yy.astype(np.float32) / np.float32(max(1, h - 1))
    ) / np.float32(2)
    result = np.stack((boundary, geometry))
    result.flags.writeable = False
    return result


def build_observation_numpy(
    compact: dict[str, Any], device: str | torch.device = "cpu", *, async_checks: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast CPU live raster + one upload per input, avoiding tiny MPS scatters.

    Colab collection and fit use the GPU-native builder above. The Mac live
    adapter already owns host Snake objects, making this equivalent NumPy
    reconstruction cheaper than dispatching many small GPU scatter kernels.
    """
    if compact["version"] != OBSERVATION_VERSION:
        raise ValueError("unknown spatial observation version")
    data = {k: v.detach().cpu().numpy() for k, v in compact.items() if k not in _META}
    h, w = int(compact["grid_h"]), int(compact["grid_w"])
    if min(h, w, int(compact["boost_cost_frames"])) < 1:
        raise ValueError("invalid observation geometry")
    n = len(data["length"])
    planes = np.zeros((n, 16, h, w), dtype=np.float32)
    planes[:, 14:16] = _numpy_geometry(h, w)
    scalars = np.empty((n, 6, 6), dtype=np.float32)
    length = data["length"].astype(np.float32)
    scalars[..., 0] = length / np.float32(LENGTH_SCALE)
    scalars[..., 1] = data["direction"].astype(np.float32) / np.float32(3)
    scalars[..., 2] = np.maximum(length - data["body_count"], 0).astype(np.float32) / np.float32(
        LENGTH_SCALE
    )
    scalars[..., 3] = data["boost_frames"].astype(np.float32) / np.float32(
        compact["boost_cost_frames"]
    )
    scalars[..., 4] = data["hunger"].astype(np.float32) / np.float32(5000)
    scalars[..., 5] = data["alive"].astype(np.float32)
    for row in range(n):
        for slot in range(6):
            if not data["alive"][row, slot]:
                continue
            count = int(data["body_count"][row, slot])
            cells = data["body_cells"][row, slot, :count].astype(np.int64)
            x, y = cells[:, 0], cells[:, 1]
            if (x < 0).any() or (x >= w).any() or (y < 0).any() or (y >= h).any():
                raise ValueError("living snake body lies outside the arena")
            planes[row, slot, y, x] = 1
            clearance = (length[row, slot] - np.arange(count, dtype=np.float32)) / np.float32(
                LENGTH_SCALE
            )
            np.maximum.at(planes[row, slot + 6], (y, x), clearance)
        food = data["food_cells"][row].astype(np.int64)
        kind = data["food_kind"][row]
        for food_class in (1, 2):
            cells = food[kind == food_class]
            x, y = cells[:, 0], cells[:, 1]
            if (x < 0).any() or (x >= w).any() or (y < 0).any() or (y >= h).any():
                raise ValueError("food lies outside the arena")
            planes[row, 11 + food_class, y, x] = 1
    heads = data["heads"].astype(np.int64).copy()
    heads[:, 0] = heads[:, 0].clip(0, w - 1)
    heads[:, 1] = heads[:, 1].clip(0, h - 1)
    return tuple(torch.from_numpy(a).to(device) for a in (planes, scalars.reshape(n, 36), heads))
