"""Pure-tensor ``ego2s-b`` observation compiler boundary.

The resident simulator and its R2 cache are deliberately kept outside this
module.  Callers materialize the cache-dependent TTL, component-size, head,
and food tensors before entering ``torch.compile(fullgraph=True)``.
"""

from __future__ import annotations

import torch


def _crop_coordinates(
    heads: torch.Tensor,
    direction: torch.Tensor,
    *,
    size: int,
    head_row: int,
    head_col: int,
    cardinal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return heading-oriented crop coordinates without simulator access."""
    rr, cc = torch.meshgrid(
        torch.arange(size, device=heads.device),
        torch.arange(size, device=heads.device),
        indexing="ij",
    )
    ahead, right = head_row - rr, cc - head_col
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
    height, width = grid.shape[-2:]
    flat = grid.reshape(grid.shape[0], height * width)
    index = (y.clamp(0, height - 1) * width + x.clamp(0, width - 1)).reshape(x.shape[0], -1)
    return flat[env].gather(1, index).reshape_as(x)


def _reach_time(ttl: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    """Fixed-round, tail-aware reference flood used by the raster contract."""
    count, height, width = ttl.shape
    del height, width
    reach = torch.zeros_like(ttl, dtype=torch.bool)
    reach[:, 23, 15] = True
    arrival = torch.zeros_like(ttl, dtype=torch.uint8)
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


def build_observation_tensor(
    ttl: torch.Tensor,
    region: torch.Tensor,
    food_grid: torch.Tensor,
    heads: torch.Tensor,
    length: torch.Tensor,
    seg_count: torch.Tensor,
    alive: torch.Tensor,
    direction: torch.Tensor,
    boost_frames: torch.Tensor,
    frames_since_food: torch.Tensor,
    boosted_this_step: torch.Tensor,
    rows: torch.Tensor,
    v8_features: torch.Tensor,
    *,
    min_boost_length: int,
    boost_length_cost_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build observations from current tensors and precomputed cache-derived inputs.

    ``ttl`` and ``region`` preserve the existing R2/cache and exact component
    semantics, but are intentionally prepared by the mutable simulator boundary.
    All arguments here are tensors except the shape-stable frozen configuration.
    """
    env, snake = rows[:, 0], rows[:, 1]
    count = rows.shape[0]
    worlds, snakes, height, width = ttl.shape
    del worlds
    cardinal = torch.tensor(((0, -1), (1, 0), (0, 1), (-1, 0)), dtype=torch.long, device=ttl.device)
    alive_rows = alive[env, snake]
    selected_heads = heads[env, snake]
    selected_direction = direction[env, snake]
    x, y = _crop_coordinates(
        selected_heads,
        selected_direction,
        size=31,
        head_row=23,
        head_col=15,
        cardinal=cardinal,
    )
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    owner_ttl = ttl[env]
    gathered = (
        owner_ttl.reshape(count, snakes, height * width)
        .gather(
            2,
            (y.clamp(0, height - 1) * width + x.clamp(0, width - 1))
            .reshape(count, 1, -1)
            .expand(-1, snakes, -1),
        )
        .reshape(count, snakes, 31, 31)
    )
    own = gathered[torch.arange(count, device=ttl.device), snake]
    enemy = gathered.masked_fill(
        torch.nn.functional.one_hot(snake, snakes).to(torch.bool)[:, :, None, None], 0
    ).amax(dim=1)
    local = torch.zeros((count, 7, 31, 31), dtype=torch.uint8, device=ttl.device)
    own = torch.where(inside, own, torch.zeros_like(own))
    enemy = torch.where(inside, enemy, torch.zeros_like(enemy))
    local[:, 0] = own.clamp(0, 255).to(torch.uint8)
    local[:, 1] = enemy.clamp(0, 255).to(torch.uint8)
    all_heads = heads[env]
    for other in range(snakes):
        is_enemy = (snake != other) & alive[env, other] & alive_rows
        hit = (x == all_heads[:, other, 0, None, None]) & (y == all_heads[:, other, 1, None, None])
        ratio = (length[env, other].float() / length[env, snake].clamp_min(1).float()).clamp(
            0, 2
        ) / 2
        value = (ratio * 254).to(torch.uint8) + 1
        local[:, 2] = torch.maximum(
            local[:, 2], (hit & is_enemy[:, None, None]).to(torch.uint8) * value[:, None, None]
        )
    food = torch.where(inside, _gather_grid(food_grid, env, x, y), torch.zeros_like(x))
    local[:, 3] = torch.where(food == 2, 255, torch.where(food == 1, 128, 0)).to(torch.uint8)
    local[:, 4] = (~inside).to(torch.uint8) * 255
    local[:, 5] = _reach_time(torch.maximum(own, enemy), ~inside)
    forward = cardinal[direction[env]]
    boosted = boosted_this_step[env] & alive[env]
    for other in range(snakes):
        valid = (snake != other) & alive[env, other] & alive_rows
        for step in (1, 2):
            cell = all_heads[:, other] + forward[:, other] * step
            hit = (x == cell[:, 0, None, None]) & (y == cell[:, 1, None, None])
            cell_inside = (
                (cell[:, 0] >= 0) & (cell[:, 0] < width) & (cell[:, 1] >= 0) & (cell[:, 1] < height)
            )
            value = torch.where(boosted[:, other], 255, 128 if step == 1 else 0)
            local[:, 6] = torch.maximum(
                local[:, 6],
                (hit & valid[:, None, None] & cell_inside[:, None, None]).to(torch.uint8)
                * value[:, None, None].to(torch.uint8),
            )
    block_height, block_width = (height + 7) // 8, (width + 7) // 8
    yy = torch.arange(height, device=ttl.device)[:, None].expand(height, width)
    xx = torch.arange(width, device=ttl.device)[None, :].expand(height, width)
    block_index = (yy // 8 * block_width + xx // 8).reshape(1, 1, -1)
    coarse = torch.zeros(
        (ttl.shape[0], snakes, block_height * block_width), dtype=torch.long, device=ttl.device
    )
    coarse.scatter_add_(
        2,
        block_index.expand(ttl.shape[0], snakes, -1),
        ttl.reshape(ttl.shape[0], snakes, -1).gt(0).long(),
    )
    food_coarse = torch.zeros(
        (ttl.shape[0], block_height * block_width), dtype=torch.long, device=ttl.device
    )
    food_coarse.scatter_add_(
        1,
        block_index[:, 0].expand(ttl.shape[0], -1),
        food_grid.gt(0).reshape(ttl.shape[0], -1).long(),
    )
    gr, gc = torch.meshgrid(
        torch.arange(37, device=ttl.device), torch.arange(37, device=ttl.device), indexing="ij"
    )
    ahead, right = 18 - gr, gc - 18
    head_block = selected_heads // 8
    forward_self = cardinal[selected_direction]
    rightward = torch.stack((-forward_self[:, 1], forward_self[:, 0]), dim=1)
    bx = (
        head_block[:, 0, None, None]
        + ahead * forward_self[:, 0, None, None]
        + right * rightward[:, 0, None, None]
    )
    by = (
        head_block[:, 1, None, None]
        + ahead * forward_self[:, 1, None, None]
        + right * rightward[:, 1, None, None]
    )
    bin_inside = (bx >= 0) & (bx < block_width) & (by >= 0) & (by < block_height)
    bind = (by.clamp(0, block_height - 1) * block_width + bx.clamp(0, block_width - 1)).reshape(
        count, -1
    )
    own_coarse = coarse[env, snake].gather(1, bind).reshape(count, 37, 37)
    all_coarse = coarse[env].sum(dim=1).gather(1, bind).reshape(count, 37, 37)
    food_values = food_coarse[env].gather(1, bind).reshape(count, 37, 37)
    own_coarse = torch.where(bin_inside, own_coarse, torch.zeros_like(own_coarse))
    all_coarse = torch.where(bin_inside, all_coarse, torch.zeros_like(all_coarse))
    food_values = torch.where(bin_inside, food_values, torch.zeros_like(food_values))
    glob = (
        torch.stack(
            (
                own_coarse,
                (all_coarse - own_coarse).clamp_min(0),
                food_values,
                (~bin_inside).long() * 255,
            ),
            dim=1,
        )
        .clamp(0, 255)
        .to(torch.uint8)
    )
    scalar = torch.zeros((count, 30), dtype=torch.float32, device=ttl.device)
    selected_length = length[env, snake].float()
    scalar[:, 0] = torch.log1p(length[env, snake].double()) / torch.log(
        torch.tensor(4097.0, dtype=torch.float64, device=ttl.device)
    )
    scalar[:, 1] = selected_length / 1000.0
    scalar[:, 2] = (length[env, snake] - seg_count[env, snake]).float() / 16.0
    scalar[:, 3] = (length[env, snake] >= min_boost_length).float()
    scalar[:, 4] = boost_frames[env, snake].float() / boost_length_cost_frames
    scalar[:, 5] = frames_since_food[env, snake].float() / 1000.0
    edge = torch.stack(
        (
            selected_heads[:, 1],
            width - 1 - selected_heads[:, 0],
            height - 1 - selected_heads[:, 1],
            selected_heads[:, 0],
        ),
        dim=1,
    ).float()
    for offset in range(4):
        scalar[:, 6 + offset] = (
            edge.gather(1, ((selected_direction + offset) % 4)[:, None]).squeeze(1) / 145.0
        )
    scalar[:, 10] = alive.float().sum(dim=1)[env] / snakes
    delta = all_heads - selected_heads[:, None]
    d2 = delta.double().square().sum(dim=2)
    valid_enemy = alive[env] & (torch.arange(snakes, device=ttl.device)[None, :] != snake[:, None])
    d2 = d2.masked_fill(~valid_enemy, torch.inf)
    order = d2.argsort(dim=1, stable=True)
    n_enemy = valid_enemy.sum(dim=1)
    for rank, base in ((0, 12), (1, 20)):
        other = order[:, rank]
        has = (n_enemy > rank) & alive_rows
        d = delta[torch.arange(count, device=ttl.device), other]
        scalar[:, base] = torch.where(has, (d * forward_self).sum(dim=1).float() / 145.0, 0.0)
        scalar[:, base + 1] = torch.where(has, (d * rightward).sum(dim=1).float() / 145.0, 0.0)
        scalar[:, base + 2] = torch.where(
            has, (length[env, other].float() / selected_length.clamp_min(1)).clamp(0, 2) / 2, 0.0
        )
        if rank == 0:
            other_direction = cardinal[direction[env, other]]
            scalar[:, 15] = torch.where(
                has, (other_direction * forward_self).sum(dim=1).float(), 0.0
            )
            scalar[:, 16] = torch.where(has, (other_direction * rightward).sum(dim=1).float(), 0.0)
            other_boost = boosted[torch.arange(count, device=ttl.device), other]
            scalar[:, 17] = torch.where(has, other_boost.float(), 0.0)
            next_enemy = all_heads[
                torch.arange(count, device=ttl.device), other
            ] + other_direction * torch.where(other_boost[:, None], 2, 1)
            next_self = selected_heads + forward_self
            scalar[:, 18] = torch.where(
                has,
                (
                    torch.linalg.vector_norm((next_enemy - next_self).double(), dim=1)
                    - torch.sqrt(d2[torch.arange(count, device=ttl.device), other])
                ).clamp(-2, 2)
                / 2,
                0.0,
            )
            candidate = all_heads[torch.arange(count, device=ttl.device), other] + other_direction
            scalar[:, 19] = (has & ((candidate - selected_heads).square().sum(dim=1) < 9)).float()
    scalar[:, 23:26] = (region.float() / selected_length[:, None].clamp_min(1)).clamp(max=4) / 4
    scalar[:, 26:29] = torch.log1p(region.double()) / torch.log(
        torch.tensor(float(height * width + 1), dtype=torch.float64, device=ttl.device)
    )
    scalar[:, 29] = n_enemy.float() / max(snakes - 1, 1)
    local = torch.where(alive_rows[:, None, None, None], local, torch.zeros_like(local))
    glob = torch.where(alive_rows[:, None, None, None], glob, torch.zeros_like(glob))
    scalar = torch.where(alive_rows[:, None], scalar, torch.zeros_like(scalar))
    return local, glob, torch.cat((scalar, v8_features.to(torch.float32)), dim=1)
