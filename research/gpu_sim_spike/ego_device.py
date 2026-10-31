"""Device-resident implementation of the ``ego2s-b`` observation contract.

This is deliberately a direct Torch port of ``ego_raster.py`` and
``ego_raster_b.py``.  It consumes only ``DeviceSim`` tensors and returns the
raw tensors expected by :class:`src.model.ego2s_network.Ego2sNet`: seven local
uint8 planes, four global uint8 planes, and the thirty ``ego2s-b`` float32
scalars.  Optional v8 features are owned by the v8 port and are simply
appended after the base thirty scalars here.

CUDA connectivity uses tiled union-find, boundary unions, pointer jumping and
integer size reduction once per world. CPU/shape checks retain the monotone
minimum-label oracle. Both gather the same three requested action starts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from research.gpu_sim_spike import flood_cuda, r2_cache

if TYPE_CHECKING:
    from research.gpu_sim_spike.device_sim import DeviceSim


_CARDINAL = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _host_scalar_safe(value: torch.Tensor) -> bool:
    """Whether ``value.item()`` is safe for the optional convergence check.

    Real CUDA tensors intentionally synchronize here only after a bulk round.
    Meta and FakeTensor shape tests retain the fixed round bound instead.
    """
    return value.device.type != "meta" and type(value).__name__ != "FakeTensor"


def _rows(sim: "DeviceSim", rows: torch.Tensor | None) -> torch.Tensor:
    if rows is None:
        env, snake = torch.meshgrid(
            torch.arange(sim.E, device=sim.device),
            torch.arange(sim.S, device=sim.device),
            indexing="ij",
        )
        return torch.stack((env.reshape(-1), snake.reshape(-1)), dim=1)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError("rows must have shape (N, 2)")
    if rows.device != sim.device:
        raise ValueError("rows must be on the DeviceSim device")
    return rows.to(dtype=torch.long)


def _cardinal(sim: "DeviceSim") -> torch.Tensor:
    """Cardinal table: pinned per-call host copy, or the R2 cached device copy."""
    if r2_cache.enabled(sim):
        return r2_cache.const(_CARDINAL, torch.long, sim.device)
    return torch.tensor(_CARDINAL, dtype=torch.long, device=sim.device)


def _float_constant(sim: "DeviceSim", value: float) -> torch.Tensor:
    if r2_cache.enabled(sim):
        return r2_cache.const(value, torch.float64, sim.device)
    return torch.tensor(value, dtype=torch.float64, device=sim.device)


def _ttl_window(sim: "DeviceSim", span: int) -> torch.Tensor:
    """R2 L1: the same TTL maps from the first ``span`` logical ring offsets.

    The pinned map visits physical slots; offset ``k`` of a slot is its logical
    age, so iterating ``k < span`` instead visits every live entry exactly once
    (``span`` covers ``seg_count``) and skips only zero-valued padding.
    """
    E, S, H, W = sim.E, sim.S, sim.grid_h, sim.grid_w
    cells, active = r2_cache.ring_view(sim, span)
    k = torch.arange(span, device=sim.device)[None, None, :]
    ttl = (sim.length[..., None] - k).clamp_min(0)
    x, y = cells[..., 0], cells[..., 1]
    valid = active & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    flat_index = y.clamp(0, H - 1) * W + x.clamp(0, W - 1)
    result = torch.zeros((E, S, H * W), dtype=torch.long, device=sim.device)
    result.scatter_reduce_(
        2,
        flat_index,
        torch.where(valid, ttl, torch.zeros_like(ttl)),
        reduce="amax",
        include_self=True,
    )
    return result.reshape(E, S, H, W)


def _ttl_by_owner(sim: "DeviceSim") -> torch.Tensor:
    """Return ``(E,S,H,W)`` TTL maps, following the reference ring convention."""
    if r2_cache.enabled(sim):
        return r2_cache.memo(
            sim,
            "ego_ttl",
            lambda: _ttl_window(sim, r2_cache.window(sim)),
            reference=lambda: _ttl_pinned(sim),
        )
    return _ttl_pinned(sim)


def _ttl_pinned(sim: "DeviceSim") -> torch.Tensor:
    """The pinned physical-slot TTL scan over the full ring capacity."""
    E, S, H, W = sim.E, sim.S, sim.grid_h, sim.grid_w
    slots = torch.arange(sim.cap, device=sim.device)[None, None, :]
    offset = (sim.head_ptr[..., None] - slots) % sim.cap
    ttl = (sim.length[..., None] - offset).clamp_min(0)
    # Ring storage is rotated after the first wrap.  Physical slot order is not
    # age order: ``offset == 0`` is the head, so that is the membership test.
    active = sim.alive[..., None] & (offset < sim.seg_count[..., None])
    xy = sim.bodies
    x, y = xy[..., 0], xy[..., 1]
    valid = active & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    flat_index = y.clamp(0, H - 1) * W + x.clamp(0, W - 1)
    result = torch.zeros((E, S, H * W), dtype=torch.long, device=sim.device)
    result.scatter_reduce_(
        2,
        flat_index,
        torch.where(valid, ttl, torch.zeros_like(ttl)),
        reduce="amax",
        include_self=True,
    )
    return result.reshape(E, S, H, W)


def _crop_coordinates(
    heads: torch.Tensor,
    direction: torch.Tensor,
    *,
    size: int,
    head_row: int,
    head_col: int,
    cardinal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rr, cc = torch.meshgrid(
        torch.arange(size, device=heads.device),
        torch.arange(size, device=heads.device),
        indexing="ij",
    )
    ahead, right = head_row - rr, cc - head_col
    if cardinal is None:
        cardinal = torch.tensor(_CARDINAL, dtype=torch.long, device=heads.device)
    forward = cardinal[direction]
    rightward = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
    x = (
        heads[:, 0, None, None]
        + ahead * forward[:, 0, None, None]
        + right * rightward[:, 0, None, None]
    )
    y = (
        heads[:, 1, None, None]
        + ahead * forward[:, 1, None, None]
        + right * rightward[:, 1, None, None]
    )
    return x, y


def _gather_grid(
    grid: torch.Tensor, env: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Gather a ``(E,H,W)`` grid at fixed-shape ``(N,A,B)`` coordinates."""
    H, W = grid.shape[-2:]
    flat = grid.reshape(grid.shape[0], H * W)
    index = (y.clamp(0, H - 1) * W + x.clamp(0, W - 1)).reshape(x.shape[0], -1)
    return flat[env].gather(1, index).reshape_as(x)


def _reach_time(ttl: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    """Reference synchronous tail-aware flood, with a static CUDA launch count."""
    N, H, W = ttl.shape
    reach = torch.zeros((N, H, W), dtype=torch.bool, device=ttl.device)
    reach[:, 23, 15] = True
    arrival = torch.zeros((N, H, W), dtype=torch.uint8, device=ttl.device)
    # The reference's early exit is only a performance shortcut.  Keeping every
    # round preserves exactly the same first-arrival values and is FakeTensor-safe.
    for tick in range(1, 25):
        grow = torch.zeros_like(reach)
        grow[:, 1:] |= reach[:, :-1]
        grow[:, :-1] |= reach[:, 1:]
        grow[:, :, 1:] |= reach[:, :, :-1]
        grow[:, :, :-1] |= reach[:, :, 1:]
        new = grow & ~reach & ~wall & (ttl <= tick)
        arrival = torch.where(new, torch.full_like(arrival, tick), arrival)
        reach |= new
    return arrival


def _component_sizes(sim: "DeviceSim", rows: torch.Tensor, ttl: torch.Tensor) -> torch.Tensor:
    """Exact tail-aware region sizes for left/straight/right, ``(N,3)``.

    Each world receives a monotone component label: every open cell begins with
    its flat index and blocked cells carry sentinel ``V``.  A bulk round hooks a
    label to the smallest cardinal-neighbour label and pointer-jumps through
    that candidate.  Labels only decrease, so a component converges to its
    minimum flat index.  Counts scatter once by component root, after which all
    three action starts gather their component root's count.

    The convergence check is a scalar synchronization on real tensors, sampled
    every eight rounds.  All bulk propagation remains on device; meta/FakeTensor
    paths use the static cardinal-path bound without reading a tensor value.
    """
    N, H, W = rows.shape[0], sim.grid_h, sim.grid_w
    # CUDA admission is explicit: a benchmark cannot accidentally report the
    # slow Torch label propagation as a native E1 kernel result.  CPU retains
    # this independently testable oracle/fallback; FakeTensor keeps shapes.
    if sim.device.type == "cuda" and type(rows).__name__ != "FakeTensor":
        return flood_cuda.components(
            (ttl.amax(dim=1) <= 1).contiguous(),
            sim.heads().contiguous(),
            sim.direction.contiguous(),
            sim.alive.contiguous(),
            rows.contiguous(),
        )
    env, snake = rows[:, 0], rows[:, 1]
    cardinal = _cardinal(sim)
    heads, direction = sim.heads()[env, snake], sim.direction[env, snake]
    rel = torch.arange(3, device=sim.device)[None, :] - 1
    move = cardinal[(direction[:, None] + rel) % 4]
    start = heads[:, None, :] + move
    sx, sy = start[..., 0], start[..., 1]
    inside = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    V = H * W
    # This occupancy is world-only: label once for each world, never for every
    # selected row/action.  ``V`` is also the blocked sentinel and cannot be a
    # valid component root.
    allowed = ttl.amax(dim=1) <= 1
    flat = torch.arange(V, device=sim.device, dtype=torch.long).expand(sim.E, -1)
    labels = torch.where(allowed.reshape(sim.E, V), flat, torch.full_like(flat, V))
    # Pointer jumping accelerates ordinary paths, but arbitrary graph label
    # order has no proven logarithmic convergence bound for this local hook.
    # V-1 is the rigorous cardinal-path bound; real inputs exit early below.
    rounds = max(1, V - 1)
    for round_index in range(rounds):
        old = labels.reshape(sim.E, H, W)
        candidate = old.clone()
        candidate[:, 1:] = torch.minimum(candidate[:, 1:], old[:, :-1])
        candidate[:, :-1] = torch.minimum(candidate[:, :-1], old[:, 1:])
        candidate[:, :, 1:] = torch.minimum(candidate[:, :, 1:], old[:, :, :-1])
        candidate[:, :, :-1] = torch.minimum(candidate[:, :, :-1], old[:, :, 1:])
        candidate = candidate.reshape(sim.E, V)
        # Every non-sentinel candidate names a valid flat cell.  Append a
        # sentinel entry so blocked cells can be gathered without a host branch.
        parents = torch.cat((labels, torch.full_like(labels[:, :1], V)), dim=1)
        jumped = parents.gather(1, candidate)
        updated = torch.minimum(candidate, jumped)
        updated = torch.where(allowed.reshape(sim.E, V), updated, torch.full_like(updated, V))
        changed = updated.ne(labels)
        labels = updated
        if (round_index + 1) % 8 == 0 and _host_scalar_safe(changed):
            if not bool(changed.any().item()):
                break
    counts = torch.zeros((sim.E, V + 1), dtype=torch.long, device=sim.device)
    counts.scatter_add_(1, labels, allowed.reshape(sim.E, V).long())
    root_grid = labels.reshape(sim.E, H, W)
    flat_start = sy.clamp(0, H - 1) * W + sx.clamp(0, W - 1)
    start_root = root_grid[env].reshape(N, V).gather(1, flat_start)
    sizes = counts[env].gather(1, start_root)
    sizes = torch.where(inside & (start_root != V), sizes, torch.zeros_like(sizes))
    return torch.where(sim.alive[env, snake, None], sizes, torch.zeros_like(sizes))


def build_observation(
    sim: "DeviceSim", rows: torch.Tensor | None = None, v8_features: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build true ``ego2s-b`` observations for selected ``(env, snake)`` rows.

    ``v8_features``, when present, must be ``(N,30)`` and is appended to the
    base 30 float scalars.  It is intentionally produced elsewhere because v8
    decision bookkeeping is a separate port.
    """
    rows = _rows(sim, rows)
    N = rows.shape[0]
    env, snake = rows[:, 0], rows[:, 1]
    H, W, S = sim.grid_h, sim.grid_w, sim.S
    alive = sim.alive[env, snake]
    heads, direction = sim.heads()[env, snake], sim.direction[env, snake]
    x, y = _crop_coordinates(
        heads,
        direction,
        size=31,
        head_row=23,
        head_col=15,
        cardinal=_cardinal(sim) if r2_cache.enabled(sim) else None,
    )
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    ttl = _ttl_by_owner(sim)
    owner_ttl = ttl[env]
    gathered = (
        owner_ttl.reshape(N, S, H * W)
        .gather(2, (y.clamp(0, H - 1) * W + x.clamp(0, W - 1)).reshape(N, 1, -1).expand(-1, S, -1))
        .reshape(N, S, 31, 31)
    )
    own = gathered[torch.arange(N, device=sim.device), snake]
    enemy = gathered.masked_fill(
        torch.nn.functional.one_hot(snake, S).to(torch.bool)[:, :, None, None], 0
    ).amax(dim=1)
    local = torch.zeros((N, 7, 31, 31), dtype=torch.uint8, device=sim.device)
    own = torch.where(inside, own, torch.zeros_like(own))
    enemy = torch.where(inside, enemy, torch.zeros_like(enemy))
    local[:, 0] = own.clamp(0, 255).to(torch.uint8)
    local[:, 1] = enemy.clamp(0, 255).to(torch.uint8)
    cardinal = _cardinal(sim)
    # Enemy-head channel exactly mirrors the reference's k == 0 / length ratio encoding.
    all_heads = sim.heads()[env]
    for other in range(S):
        is_enemy = (snake != other) & sim.alive[env, other] & alive
        hit = (x == all_heads[:, other, 0, None, None]) & (y == all_heads[:, other, 1, None, None])
        ratio = (
            sim.length[env, other].float() / sim.length[env, snake].clamp_min(1).float()
        ).clamp(0, 2) / 2
        value = (ratio * 254).to(torch.uint8) + 1
        local[:, 2] = torch.maximum(
            local[:, 2], (hit & is_enemy[:, None, None]).to(torch.uint8) * value[:, None, None]
        )
    food = torch.where(inside, _gather_grid(sim.food_grid(), env, x, y), torch.zeros_like(x))
    local[:, 3] = torch.where(food == 2, 255, torch.where(food == 1, 128, 0)).to(torch.uint8)
    local[:, 4] = (~inside).to(torch.uint8) * 255
    local[:, 5] = _reach_time(torch.maximum(own, enemy), ~inside)
    # Enemy next (including the second cell after a prior boost) is an exact ego crop scatter.
    forward = cardinal[sim.direction[env]]
    boosted = sim._boosted_this_step[env] & sim.alive[env] & (sim.length[env] > 1)
    for other in range(S):
        valid = (snake != other) & sim.alive[env, other] & alive
        for step in (1, 2):
            cell = all_heads[:, other] + forward[:, other] * step
            hit = (x == cell[:, 0, None, None]) & (y == cell[:, 1, None, None])
            cell_inside = (
                (cell[:, 0] >= 0) & (cell[:, 0] < W) & (cell[:, 1] >= 0) & (cell[:, 1] < H)
            )
            value = torch.where(boosted[:, other], 255, 128 if step == 1 else 0)
            local[:, 6] = torch.maximum(
                local[:, 6],
                (hit & valid[:, None, None] & cell_inside[:, None, None]).to(torch.uint8)
                * value[:, None, None].to(torch.uint8),
            )
    # Global coarse planes in the same heading-rotated block coordinate system.
    block_h, block_w = (H + 7) // 8, (W + 7) // 8
    yy = torch.arange(H, device=sim.device)[:, None].expand(H, W)
    xx = torch.arange(W, device=sim.device)[None, :].expand(H, W)
    block_index = (yy // 8 * block_w + xx // 8).reshape(1, 1, -1)
    coarse = torch.zeros((sim.E, S, block_h * block_w), dtype=torch.long, device=sim.device)
    coarse.scatter_add_(2, block_index.expand(sim.E, S, -1), ttl.reshape(sim.E, S, -1).gt(0).long())
    food_coarse = torch.zeros((sim.E, block_h * block_w), dtype=torch.long, device=sim.device)
    food_coarse.scatter_add_(
        1,
        block_index[:, 0].expand(sim.E, -1),
        sim.food_grid().gt(0).reshape(sim.E, -1).long(),
    )
    gr, gc = torch.meshgrid(
        torch.arange(37, device=sim.device), torch.arange(37, device=sim.device), indexing="ij"
    )
    ga, gright = 18 - gr, gc - 18
    head_block = heads // 8
    rightward = torch.stack((-cardinal[direction][:, 1], cardinal[direction][:, 0]), dim=1)
    bx = (
        head_block[:, 0, None, None]
        + ga * cardinal[direction][:, 0, None, None]
        + gright * rightward[:, 0, None, None]
    )
    by = (
        head_block[:, 1, None, None]
        + ga * cardinal[direction][:, 1, None, None]
        + gright * rightward[:, 1, None, None]
    )
    bin_inside = (bx >= 0) & (bx < block_w) & (by >= 0) & (by < block_h)
    bind = (by.clamp(0, block_h - 1) * block_w + bx.clamp(0, block_w - 1)).reshape(N, -1)
    own_coarse = coarse[env, snake].gather(1, bind).reshape(N, 37, 37)
    all_coarse = coarse[env].sum(dim=1).gather(1, bind).reshape(N, 37, 37)
    food_vals = food_coarse[env].gather(1, bind).reshape(N, 37, 37)
    own_coarse = torch.where(bin_inside, own_coarse, torch.zeros_like(own_coarse))
    all_coarse = torch.where(bin_inside, all_coarse, torch.zeros_like(all_coarse))
    food_vals = torch.where(bin_inside, food_vals, torch.zeros_like(food_vals))
    glob = (
        torch.stack(
            (
                own_coarse,
                (all_coarse - own_coarse).clamp_min(0),
                food_vals,
                (~bin_inside).long() * 255,
            ),
            dim=1,
        )
        .clamp(0, 255)
        .to(torch.uint8)
    )
    # Scalars 0..11 from ego_raster.py.
    scalar = torch.zeros((N, 30), dtype=torch.float32, device=sim.device)
    L = sim.length[env, snake].float()
    # The reference scalar producer evaluates transcendental terms in float64
    # and only then stores them as float32.
    L64 = sim.length[env, snake].double()
    scalar[:, 0] = torch.log1p(L64) / torch.log(_float_constant(sim, 4097.0))
    scalar[:, 1] = L / 1000.0
    scalar[:, 2] = (sim.length[env, snake] - sim.seg_count[env, snake]).float() / 16.0
    scalar[:, 3] = (sim.length[env, snake] >= sim.cfg.min_boost_length).float()
    scalar[:, 4] = sim.boost_frames[env, snake].float() / sim.cfg.boost_length_cost_frames
    scalar[:, 5] = sim.frames_since_food[env, snake].float() / 1000.0
    edge = torch.stack(
        (heads[:, 1], W - 1 - heads[:, 0], H - 1 - heads[:, 1], heads[:, 0]), dim=1
    ).float()
    for offset in range(4):
        scalar[:, 6 + offset] = (
            edge.gather(1, ((direction + offset) % 4)[:, None]).squeeze(1) / 145.0
        )
    scalar[:, 10] = sim.alive.float().sum(dim=1)[env] / S
    # b scalars 12..29 from ego_raster_b.py.
    delta = all_heads - heads[:, None]
    # ``ego_raster_b`` deliberately computes distance geometry in float64
    # before assigning the float32 feature.  Keep that contract on CPU/CUDA.
    d2 = delta.double().square().sum(dim=2)
    valid_enemy = sim.alive[env] & (torch.arange(S, device=sim.device)[None, :] != snake[:, None])
    d2 = d2.masked_fill(~valid_enemy, torch.inf)
    order = d2.argsort(dim=1, stable=True)
    n_enemy = valid_enemy.sum(dim=1)
    forward_self = cardinal[direction]
    right_self = torch.stack((-forward_self[:, 1], forward_self[:, 0]), dim=1)
    for rank, base in ((0, 12), (1, 20)):
        other = order[:, rank]
        has = (n_enemy > rank) & alive
        d = delta[torch.arange(N, device=sim.device), other]
        scalar[:, base] = torch.where(has, (d * forward_self).sum(dim=1).float() / 145.0, 0.0)
        scalar[:, base + 1] = torch.where(has, (d * right_self).sum(dim=1).float() / 145.0, 0.0)
        scalar[:, base + 2] = torch.where(
            has, (sim.length[env, other].float() / L.clamp_min(1)).clamp(0, 2) / 2, 0.0
        )
        if rank == 0:
            other_dir = cardinal[sim.direction[env, other]]
            scalar[:, 15] = torch.where(has, (other_dir * forward_self).sum(dim=1).float(), 0.0)
            scalar[:, 16] = torch.where(has, (other_dir * right_self).sum(dim=1).float(), 0.0)
            other_boost = boosted[torch.arange(N, device=sim.device), other]
            scalar[:, 17] = torch.where(has, other_boost.float(), 0.0)
            next_enemy = all_heads[
                torch.arange(N, device=sim.device), other
            ] + other_dir * torch.where(other_boost[:, None], 2, 1)
            next_self = heads + forward_self
            scalar[:, 18] = torch.where(
                has,
                (
                    torch.linalg.vector_norm((next_enemy - next_self).double(), dim=1)
                    - torch.sqrt(d2[torch.arange(N, device=sim.device), other])
                ).clamp(-2, 2)
                / 2,
                0.0,
            )
            candidate = all_heads[torch.arange(N, device=sim.device), other] + other_dir
            scalar[:, 19] = (has & ((candidate - heads).square().sum(dim=1) < 9)).float()
    region = _component_sizes(sim, rows, ttl).float()
    scalar[:, 23:26] = (region / L[:, None].clamp_min(1)).clamp(max=4) / 4
    scalar[:, 26:29] = torch.log1p(region.double()) / torch.log(
        _float_constant(sim, float(H * W + 1))
    )
    scalar[:, 29] = n_enemy.float() / max(S - 1, 1)
    local = torch.where(alive[:, None, None, None], local, torch.zeros_like(local))
    glob = torch.where(alive[:, None, None, None], glob, torch.zeros_like(glob))
    scalar = torch.where(alive[:, None], scalar, torch.zeros_like(scalar))
    if v8_features is not None:
        if v8_features.shape != (N, 30) or v8_features.device != sim.device:
            raise ValueError("v8_features must have shape (N, 30) on the DeviceSim device")
        scalar = torch.cat((scalar, v8_features.to(torch.float32)), dim=1)
    return local, glob, scalar
