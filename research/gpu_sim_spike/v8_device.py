"""Device-resident port of the frozen :mod:`src.simd_env.v8_nb` decision rule.

This is deliberately a direct decision-time port, rather than a cheaper danger
screen.  It retains v2 capped counts, v5's post-boost landing flood, v7's
tail-release landing area and re-rank, and v8's opponent-head layer.  The
CUDA uses one sparse-frontier block per world/slot/action, with queue storage
covering discovered cells and asynchronous watchdog checks. CPU/shape checks
retain the Torch wavefront oracle. Only capped cardinality is consumed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from research.gpu_sim_spike import flood_cuda
from src.core.mechanics_constants import HEADON_SIZE_RATIO

_DX = (0, 1, 0, -1)
_DY = (-1, 0, 1, 0)
_CAP_MIN, _CAP_MAX, _AREA_MAX = 32, 160, 4096
_LANDING_UNKNOWN, _LANDING_NO_BOOST = -1, -2
_KEPT, _VETOED, _NO_SPACIOUS = 0, 1, 2
_R_KEPT, _R_V2, _R_SAME_DIR, _R_LANDING_V2, _R_LANDING_NONE = range(5)


def _host_scalar_safe(value: torch.Tensor) -> bool:
    """Whether a convergence scalar can be read outside meta/FakeTensor tests."""
    return value.device.type != "meta" and type(value).__name__ != "FakeTensor"


@dataclass
class V8DeviceResult:
    """Torch counterpart of ``v8_nb.V8Result``; every member stays on ``q.device``."""

    action: torch.Tensor
    base: torch.Tensor
    v5_action: torch.Tensor
    v7_action: torch.Tensor
    outcome: torch.Tensor
    v5_reason: torch.Tensor
    counts: torch.Tensor
    cap: torch.Tensor
    need: torch.Tensor
    area_cap: torch.Tensor
    area: torch.Tensor
    landing: torch.Tensor
    head_risky: torch.Tensor
    head_waived: torch.Tensor
    v5_eligible: torch.Tensor

    @property
    def spacious(self) -> torch.Tensor:
        return self.counts >= self.need[:, None]

    @property
    def landing_ok(self) -> torch.Tensor:
        out = torch.ones((self.action.shape[0], 6), dtype=torch.bool, device=self.action.device)
        out[:, 3:] = (self.landing == _LANDING_NO_BOOST) | (self.landing >= self.need[:, None])
        return out

    @property
    def area_g(self) -> torch.Tensor:
        return self.area.clamp_min(0).minimum(self.area_cap[:, None]).to(
            torch.float32
        ) / self.area_cap[:, None].clamp_min(1)

    def feature_planes(self) -> torch.Tensor:
        """Return the B8-v8 diagnostic planes as ``(N, 6, 5)`` float32."""
        sp = torch.cat((self.spacious, self.spacious), dim=1)
        return torch.stack(
            (
                self.area_g,
                self.head_risky.float(),
                self.landing_ok.float(),
                self.v5_eligible.float(),
                sp.float(),
            ),
            dim=-1,
        )


def _heads(sim: Any, env: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    return sim.bodies[env, slot, sim.head_ptr[env, slot]]


def _body_cells(sim: Any, env: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cells ordered head-to-tail for all slots in selected environments."""
    n, s, cap = env.shape[0], sim.S, sim.cap
    slots = torch.arange(s, device=env.device)[None, :].expand(n, -1)
    hp = sim.head_ptr[env[:, None], slots]
    k = torch.arange(cap, device=env.device)[None, None, :]
    ring = (hp[..., None] - k) % cap
    cells = sim.bodies[env[:, None, None], slots[..., None], ring]
    valid = sim.alive[env[:, None], slots][..., None] & (
        k < sim.seg_count[env[:, None], slots][..., None]
    )
    return cells, valid


def _paint(cells: torch.Tensor, valid: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Paint an ``N,H,W`` union without host-side indexing or scalar extraction."""
    n = cells.shape[0]
    x, y = cells[..., 0], cells[..., 1]
    ok = valid & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    # CUDA scatter_reduce(amax) excludes Bool (CPU/meta accept it). Integer
    # 0/1 max is the same union, including duplicate and invalid clamped cells.
    flat = torch.zeros((n, h * w), dtype=torch.uint8, device=cells.device)
    index = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).reshape(n, -1)
    flat.scatter_reduce_(
        1, index, ok.reshape(n, -1).to(torch.uint8), reduce="amax", include_self=True
    )
    return flat.reshape(n, h, w) != 0


def _flood(
    start: torch.Tensor,
    blocked: torch.Tensor,
    cap: torch.Tensor,
    release: torch.Tensor | None = None,
    *,
    max_steps: int | None = None,
) -> torch.Tensor:
    """Capped 4-neighbour component sizes for a batch of start cells.

    ``release`` contains the minimum graph distance at which an otherwise own-body
    cell becomes passable.  It is the tensor form of v7's FIFO release map.
    """
    n, a = start.shape[:2]
    h, w = blocked.shape[-2:]
    if start.device.type == "cuda" and type(start).__name__ != "FakeTensor":
        # Native path owns one atomic-deduplicated FIFO per lane and terminates
        # on empty queue or cap. Storage covers min(V, 1 + 4*cap) QUEUED cells,
        # including the frontier beyond the CPU's popped-cell cap.
        return flood_cuda.flood(
            start.contiguous(),
            blocked.contiguous(),
            cap.contiguous(),
            None if release is None else release.contiguous(),
            min(h * w, max_steps if max_steps is not None else _AREA_MAX),
        )
    sx, sy = start[..., 0], start[..., 1]
    inside = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    index = sy.clamp(0, h - 1) * w + sx.clamp(0, w - 1)
    start_blocked = blocked.reshape(n, a, -1).gather(2, index[..., None]).squeeze(-1)
    if release is not None:
        r0 = release.reshape(n, a, -1).gather(2, index[..., None]).squeeze(-1)
        start_blocked |= r0 > 0
    seed = torch.zeros((n, a, h * w), dtype=torch.bool, device=start.device)
    seed.scatter_(2, index[..., None], (inside & ~start_blocked)[..., None])
    seen = seed.reshape(n, a, h, w)
    frontier = seen
    # A FIFO cannot consume more than ``cap`` cells.  v2/v5 cap at 160 and v7
    # cap at 4096, so those are the only useful wavefront bounds.  We sample the
    # all-lane completion predicate every eight rounds; reading that scalar is a
    # deliberate CUDA synchronization, while all frontier work remains device
    # resident.  Meta/FakeTensor paths keep the static bound for shape checks.
    limit = min(h * w - 1, max_steps if max_steps is not None else _AREA_MAX)
    for step in range(max(0, limit)):
        # The CPU kernel's _NBX/_NBY is cardinal-only.  A 3x3 max-pool would
        # silently add diagonal reachability and must never be used here.
        grow = torch.zeros_like(frontier)
        grow[..., 1:, :] |= frontier[..., :-1, :]
        grow[..., :-1, :] |= frontier[..., 1:, :]
        grow[..., :, 1:] |= frontier[..., :, :-1]
        grow[..., :, :-1] |= frontier[..., :, 1:]
        passable = ~blocked
        if release is not None:
            passable = passable & ((release <= (step + 1)) | (release == 0))
        nxt = grow & ~seen & passable
        # Once a component has reached its cap further cells cannot change the answer.
        count = seen.flatten(2).sum(-1)
        nxt &= count[..., None, None] < cap[..., None, None]
        seen |= nxt
        frontier = nxt
        if (step + 1) % 8 == 0 and _host_scalar_safe(frontier):
            active = frontier.flatten(2).any(-1) & (seen.flatten(2).sum(-1) < cap)
            # FIFO release has no waiting queue: an empty frontier stays empty
            # even if a later release threshold would make another cell passable.
            if not bool(active.any().item()):
                break
    return seen.flatten(2).sum(-1).minimum(cap)


def _post_body(
    sim: Any,
    env: torch.Tensor,
    slot: torch.Tensor,
    action: torch.Tensor,
    old_cells: torch.Tensor,
    old_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact v5 post-move ring view for each row/action, in head-to-tail order."""
    a, cap = action.shape[1], sim.cap
    length = sim.length[env, slot][:, None].expand(-1, a)
    count = sim.seg_count[env, slot][:, None].expand(-1, a)
    bf = sim.boost_frames[env, slot][:, None].expand(-1, a)
    heading = sim.direction[env, slot][:, None]
    rel = action.remainder(3)
    hd = (heading + rel - 1) % 4
    dx = torch.as_tensor(_DX, device=env.device)[hd]
    dy = torch.as_tensor(_DY, device=env.device)[hd]
    boosted = (action >= 3) & (length >= int(sim.cfg.min_boost_length))
    ln = torch.minimum(count + 1, length)
    ln = torch.where(boosted, torch.minimum(ln + 1, length), ln)
    lpost = torch.where(
        boosted & (bf + 1 >= int(sim.cfg.boost_length_cost_frames)) & (length > 1),
        length - 1,
        length,
    )
    ln = torch.minimum(ln, lpost)
    nh = torch.where(boosted, torch.full_like(length, 2), torch.ones_like(length))
    heads = _heads(sim, env, slot)[:, None, :]
    i = torch.arange(cap, device=env.device)[None, None, :]
    new_step = (nh[..., None] - i).clamp_min(0)
    new = heads[:, :, None, :] + torch.stack((dx, dy), -1)[:, :, None, :] * new_step[..., None]
    old_index = (i - nh[..., None]).clamp_min(0)
    old = (
        old_cells[:, None]
        .expand(-1, a, -1, -1)
        .gather(2, old_index[..., None].expand(-1, -1, -1, 2))
    )
    cells = torch.where((i < nh[..., None])[..., None], new, old)
    valid = i < ln[..., None]
    return cells, valid, lpost, boosted


def _choice(
    q: torch.Tensor, eligible: torch.Tensor, base: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen v2 same-speed-first replacement, including first-max tie order."""
    any_el = eligible.any(dim=1)
    same = eligible & ((torch.arange(6, device=q.device)[None] >= 3) == (base[:, None] >= 3))
    candidate = torch.where(same.any(1, keepdim=True), same, eligible)
    best = q.masked_fill(~candidate, -torch.inf).argmax(1)
    action = torch.where(
        any_el, torch.where(eligible.gather(1, base[:, None]).squeeze(1), base, best), base
    )
    outcome = torch.where(
        ~any_el,
        torch.full_like(base, _NO_SPACIOUS),
        torch.where(
            eligible.gather(1, base[:, None]).squeeze(1),
            torch.full_like(base, _KEPT),
            torch.full_like(base, _VETOED),
        ),
    )
    return action, outcome


def _rerank(
    q: torch.Tensor,
    candidates: torch.Tensor,
    anchor: torch.Tensor,
    area: torch.Tensor,
    cap: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Eager equivalent of v7's lazy ``space_preference_choice``."""
    # v8_nb copies float32 Q into a float64 qv/scratch before normalizing.
    # Preserve that precision and operation order because area-heavy scores can
    # differ close to a replacement boundary without being an argmax tie.
    q64 = q.to(torch.float64)
    q_c = q64.masked_fill(~candidates, -torch.inf)
    top, bot = q_c.max(1).values, q_c.masked_fill(~candidates, torch.inf).min(1).values
    qn = (q64 - top[:, None]) / (top - bot).abs().clamp_min(1e-6)[:, None]
    score = qn + float(lam) * area.clamp_min(0).minimum(cap[:, None]).to(torch.float64) / cap[
        :, None
    ].clamp_min(1).to(torch.float64)
    score = score.masked_fill(~candidates, -torch.inf)
    best_score, best = score.max(1)
    anchor_score = score.gather(1, anchor[:, None]).squeeze(1)
    # ``space_preference_choice`` only replaces on strictly greater score.
    return torch.where(anchor_score >= best_score, anchor, best)


def v8_decide(
    sim: Any,
    rows: torch.Tensor,
    q: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 8.0,
    features: bool = False,
) -> V8DeviceResult:
    """Apply the complete v8 decision rule using only device tensors.

    ``sim`` must be ``DeviceSim``-compatible.  ``rows`` is ``(N,2)`` on the
    same device, and all returned diagnostic tensors are in caller row order.
    """
    if q.ndim != 2 or q.shape[1] != 6 or mask.shape != q.shape or rows.shape != (q.shape[0], 2):
        raise ValueError("rows must be (N,2), and q/mask must be (N,6)")
    if not math.isfinite(float(lam)) or lam < 0:
        raise ValueError("lam must be finite and >= 0")
    if getattr(sim.cfg, "arena_type", "rectangular") == "circular":
        raise ValueError("v8 device port supports rectangular arenas only")
    # The frozen NumPy entry point coerces Q to float32 before masking/ranking.
    q = q.to(dtype=torch.float32)
    device, n = q.device, q.shape[0]
    rows = rows.to(device=device, dtype=torch.long)
    env, slot = rows[:, 0], rows[:, 1]
    if n == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        z6 = torch.zeros((0, 6), dtype=torch.bool, device=device)
        return V8DeviceResult(
            z,
            z,
            z,
            z,
            z,
            z,
            torch.zeros((0, 3), dtype=torch.long, device=device),
            z,
            z,
            z,
            torch.zeros((0, 6), dtype=torch.long, device=device),
            torch.zeros((0, 3), dtype=torch.long, device=device),
            z6,
            z6,
            z6,
        )
    h, w = int(sim.grid_h), int(sim.grid_w)
    old_cells_all, old_valid_all = _body_cells(sim, env)
    own_cells = old_cells_all[torch.arange(n, device=device), slot]
    own_valid = old_valid_all[torch.arange(n, device=device), slot]
    all_occ = _paint(old_cells_all, old_valid_all, h, w)
    # Other snakes are static walls for v5/v7 post-move paths.
    other_valid = old_valid_all & (
        torch.arange(sim.S, device=device)[None, :, None] != slot[:, None, None]
    )
    other_occ = _paint(old_cells_all, other_valid, h, w)
    heads = _heads(sim, env, slot)
    direction = sim.direction[env, slot]
    rel3 = torch.arange(3, device=device)[None]
    hd3 = (direction[:, None] + rel3 - 1) % 4
    delta3 = torch.stack(
        (torch.as_tensor(_DX, device=device)[hd3], torch.as_tensor(_DY, device=device)[hd3]), -1
    )
    starts = heads[:, None] + delta3
    length = sim.length[env, slot]
    cap = (2 * length).clamp(_CAP_MIN, _CAP_MAX)
    need = length.clamp_min(1).minimum(cap)
    counts = _flood(
        starts,
        all_occ[:, None].expand(-1, 3, -1, -1),
        cap[:, None].expand(-1, 3),
        max_steps=_CAP_MAX,
    )
    spacious = counts >= need[:, None]
    legal = mask.bool()
    legal = torch.where(legal.any(1, keepdim=True), legal, torch.arange(6, device=device)[None] < 3)
    base = q.masked_fill(~legal, -1e30).argmax(1)
    actions6 = torch.arange(6, device=device)[None].expand(n, -1)
    post_cells, post_valid, lpost, boosted = _post_body(
        sim, env, slot, actions6, own_cells, own_valid
    )
    # v5 boost landing: paint post-move indices 1..n, rather than painting the
    # whole body then clearing the head coordinate.  The distinction matters
    # when a new head overlaps a retained body cell: CPU ``_landing_count``
    # keeps that cell blocked and reports landing area zero.
    head_pos = post_cells[:, :, 0]
    post_tail_occ = _paint(
        post_cells[:, :, 1:].reshape(n * 6, sim.cap - 1, 2),
        post_valid[:, :, 1:].reshape(n * 6, sim.cap - 1),
        h,
        w,
    ).reshape(n, 6, h, w)
    landing_block = other_occ[:, None] | post_tail_occ
    landing6 = _flood(head_pos, landing_block, cap[:, None].expand(-1, 6), max_steps=_CAP_MAX)
    landing = torch.where(
        boosted[:, 3:], landing6[:, 3:], torch.full_like(landing6[:, 3:], _LANDING_NO_BOOST)
    )
    landing_ok3 = (landing == _LANDING_NO_BOOST) | (landing >= need[:, None])
    elig = legal & torch.cat((spacious, spacious & landing_ok3), dim=1)
    v5, outcome = _choice(q, elig, base)
    base_v2_eligible = legal.gather(1, base[:, None]).squeeze(1) & spacious.gather(
        1, base.remainder(3)[:, None]
    ).squeeze(1)
    base_boost_bad = (
        (base >= 3)
        & base_v2_eligible
        & ~landing_ok3.gather(1, (base - 3).clamp_min(0)[:, None]).squeeze(1)
    )
    same_normal = legal.gather(1, base.remainder(3)[:, None]).squeeze(1)
    v5 = torch.where(base_boost_bad & same_normal, base.remainder(3), v5)
    outcome = torch.where(base_boost_bad & same_normal, torch.full_like(outcome, _VETOED), outcome)
    reason = torch.where(
        base_boost_bad & same_normal,
        torch.full_like(base, _R_SAME_DIR),
        torch.where(
            base_boost_bad & (outcome == _NO_SPACIOUS),
            torch.full_like(base, _R_LANDING_NONE),
            torch.where(
                base_boost_bad,
                torch.full_like(base, _R_LANDING_V2),
                torch.where(
                    elig.gather(1, base[:, None]).squeeze(1),
                    torch.full_like(base, _R_KEPT),
                    torch.full_like(base, _R_V2),
                ),
            ),
        ),
    )
    # v7 tail-release area.  ``lpost-i+1`` matches the frozen release-map values;
    # a head is explicitly released for the flood source.
    i = torch.arange(sim.cap, device=device)[None, None, :]
    rel_values = (lpost[:, :, None] - i + 1).clamp_min(0)
    release = torch.zeros((n, 6, h * w), dtype=torch.long, device=device)
    px, py = post_cells[..., 0], post_cells[..., 1]
    good = post_valid & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    pind = py.clamp(0, h - 1) * w + px.clamp(0, w - 1)
    release.scatter_reduce_(
        2,
        pind,
        torch.where(good, rel_values, torch.zeros_like(rel_values)),
        reduce="amax",
        include_self=True,
    )
    release = release.reshape(n, 6, h, w)
    # New head is unblocked by release.pop(head), preserving opponent blockers.
    hi = head_pos[..., 1].clamp(0, h - 1) * w + head_pos[..., 0].clamp(0, w - 1)
    release.reshape(n, 6, -1).scatter_(
        2, hi[..., None], torch.zeros((n, 6, 1), dtype=torch.long, device=device)
    )
    covered = other_occ.flatten(1).sum(1)
    area_cap = (
        (2 * length)
        .clamp_min(_CAP_MIN)
        .minimum(torch.full_like(length, _AREA_MAX))
        .minimum((h * w - covered).clamp_min(1))
    )
    area_full = _flood(
        head_pos,
        other_occ[:, None].expand(-1, 6, -1, -1),
        area_cap[:, None].expand(-1, 6),
        release,
        max_steps=_AREA_MAX,
    )
    area = area_full if features else torch.full_like(area_full, _LANDING_UNKNOWN)
    # v7 only compares legal, spacious actions in v5's speed group and skips a
    # v5 same-direction replacement exactly as the CPU rule does.
    speed = v5 >= 3
    candidates = (
        legal
        & torch.cat((spacious, spacious & landing_ok3), 1)
        & ((actions6 >= 3) == speed[:, None])
    )
    candidates.scatter_(1, v5[:, None], True)
    reranked = _rerank(q, candidates, v5, area_full, area_cap, lam)
    do_v7 = (
        (outcome != _NO_SPACIOUS)
        & (reason != _R_SAME_DIR)
        & (float(lam) != 0.0)
        & (candidates.sum(1) >= 2)
    )
    v7 = torch.where(do_v7, reranked, v5)
    v7_outcome = torch.where(do_v7 & (v7 != v5), torch.full_like(outcome, _VETOED), outcome)
    # v8 head-risk: opponent current and relative one/two-step reaches.
    opp = torch.arange(sim.S, device=device)[None, :] != slot[:, None]
    all_slots = torch.arange(sim.S, device=device)[None].expand(n, -1)
    oh = sim.bodies[env[:, None], all_slots, sim.head_ptr[env[:, None], all_slots]]
    od = sim.direction[env]
    ol = sim.length[env]
    oa = sim.alive[env] & opp & (sim.seg_count[env] > 0)
    od3 = (od[:, :, None] + torch.arange(3, device=device)[None, None, :] - 1) % 4
    odelta = torch.stack(
        (torch.as_tensor(_DX, device=device)[od3], torch.as_tensor(_DY, device=device)[od3]), -1
    )
    reach1, reach2 = oh[:, :, None] + odelta, oh[:, :, None] + 2 * odelta
    reach = torch.cat((oh[:, :, None], reach1, reach2), 2)
    rvalid = torch.cat(
        (
            oa[:, :, None],
            oa[:, :, None].expand(-1, -1, 3),
            (oa & (ol >= int(sim.cfg.min_boost_length)))[:, :, None].expand(-1, -1, 3),
        ),
        2,
    )
    bound = (ol.clamp_min(1) + 1)[:, :, None].expand_as(rvalid)
    paths = post_cells[:, :, :2]
    path_valid = (
        torch.arange(2, device=device)[None, None, :] < torch.where(boosted, 2, 1)[:, :, None]
    )
    equal = (paths[:, :, :, None, None, :] == reach[:, None, None, :, :, :]).all(-1) & rvalid[
        :, None, None
    ]
    # Preserve v8's per-traversed-cell rule: a waivable collision on one
    # boost cell does not waive a risky collision on its other traversed cell.
    bmax_point = torch.where(
        equal, bound[:, None, None], torch.full_like(bound[:, None, None], -1)
    ).amax(dim=(-1, -2))
    waived_point = (int(getattr(sim.cfg, "mechanics_version", 1)) == 2) & (
        lpost[:, :, None] >= float(HEADON_SIZE_RATIO) * bmax_point
    )
    risky = ((bmax_point >= 0) & ~waived_point & path_valid).any(dim=2)
    waived = (waived_point & (bmax_point >= 0) & path_valid).any(dim=2) & ~risky
    safe = legal & torch.cat((spacious, spacious & landing_ok3), 1) & ~risky
    same_v7 = safe & ((actions6 >= 3) == (v7[:, None] >= 3)) & (actions6 != v7[:, None])
    alternate = torch.where(same_v7.any(1, keepdim=True), same_v7, safe & (actions6 != v7[:, None]))
    anchor = q.masked_fill(~alternate, -1e30).argmax(1)
    hcand = alternate & ((actions6 >= 3) == (anchor[:, None] >= 3))
    head_best = _rerank(q, hcand, anchor, area_full, area_cap, lam)
    replace = (
        (v7_outcome != _NO_SPACIOUS) & risky.gather(1, v7[:, None]).squeeze(1) & alternate.any(1)
    )
    final = torch.where(replace, head_best, v7)
    final_outcome = torch.where(replace, torch.full_like(outcome, _VETOED), v7_outcome)
    return V8DeviceResult(
        final,
        base,
        v5,
        v7,
        final_outcome,
        reason,
        counts,
        cap,
        need,
        area_cap,
        area,
        landing,
        risky,
        waived,
        elig,
    )
