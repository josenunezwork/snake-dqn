"""Exact rectangular advisory masks from device ring buffers.

Direct tensor translation of GridBatchSim._compute_action_masks_numpy. Owner
precedence is highest live slot; self-overlap keeps the newest body segment.
"""

from __future__ import annotations

from typing import Any

import torch


def advisory_mask(sim: Any) -> torch.Tensor:
    """Return the six exact advisory bits for all E*S rows, entirely on device."""
    if sim.cfg.arena_type != "rectangular":
        raise NotImplementedError("Grid reference advisory masks require rectangular arenas")
    ecount, slots, cap = sim.E, sim.S, sim.cap
    device, width, height = sim.device, sim.grid_w, sim.grid_h
    env = torch.arange(ecount, device=device)[:, None, None]
    snake = torch.arange(slots, device=device)[None, :, None]
    offset = torch.arange(cap, device=device)[None, None, :]
    ring = (sim.head_ptr[..., None] - offset) % cap
    body = sim.bodies[env, snake, ring]
    bx, by = body[..., 0], body[..., 1]
    active = sim.alive[..., None] & (offset < sim.seg_count[..., None])
    valid = active & (bx >= 0) & (bx < width) & (by >= 0) & (by < height)
    # Decode both owner and offset from one deterministic max reduction.
    rank = (snake + 1) * (cap + 1) + cap - offset
    values = torch.where(valid, rank.expand_as(bx), torch.zeros_like(bx))
    grid = torch.zeros((ecount, height * width), dtype=torch.long, device=device)
    index = by.clamp(0, height - 1) * width + bx.clamp(0, width - 1)
    grid.scatter_reduce_(1, index.reshape(ecount, -1), values.reshape(ecount, -1), "amax")
    direction = (sim.direction[..., None] + torch.tensor((-1, 0, 1), device=device)) % 4
    cardinal = torch.tensor(((0, -1), (1, 0), (0, 1), (-1, 0)), device=device)
    delta = cardinal[direction]
    first = sim.heads()[:, :, None] + delta
    cells = torch.stack((first, first + delta), dim=3)
    x, y = cells[..., 0], cells[..., 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    positions = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
    gathered = grid.gather(1, positions.reshape(ecount, -1)).reshape_as(x)
    owner = torch.div(gathered, cap + 1, rounding_mode="floor") - 1
    k = cap - gathered.remainder(cap + 1)
    safe_owner = owner.clamp_min(0)
    e4 = torch.arange(ecount, device=device)[:, None, None, None]
    owner_n, owner_l = sim.seg_count[e4, safe_owner], sim.length[e4, safe_owner]
    owner_alive = sim.alive[e4, safe_owner]
    own = owner == torch.arange(slots, device=device)[None, :, None, None]
    other = (owner >= 0) & ~own & owner_alive
    drop_tail = (owner_n >= 2) & (owner_n >= owner_l)
    other_last = torch.where(drop_tail, owner_n - 2, owner_n - 1)
    other_hit = other & (k <= other_last)
    length, n = sim.length[..., None], sim.seg_count[..., None]
    normal_last = torch.minimum(n + 1, length) - 2
    own_normal = own[..., 0] & (k[..., 0] >= 2) & (k[..., 0] <= normal_last)
    fatal_normal = ~inside[..., 0] | own_normal | other_hit[..., 0]
    l1 = torch.minimum(n + 1, length)
    l2 = torch.minimum(l1 + 1, length)
    burn = sim.boost_frames[..., None] + 1 >= sim.cfg.boost_length_cost_frames
    final = torch.where(burn, torch.minimum(l2, (length - 1).clamp_min(1)), l2)
    own_boost = own & (k >= 1) & (k <= (final - 3)[..., None])
    fatal_boost = (~inside | own_boost | other_hit).any(dim=3)
    normal = ~fatal_normal & sim.alive[..., None]
    boost = normal & (length >= sim.cfg.min_boost_length) & ~fatal_boost
    return torch.cat((normal, boost), dim=2)
