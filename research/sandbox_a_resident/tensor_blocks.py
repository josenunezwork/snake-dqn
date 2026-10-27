"""Pure-tensor compiler boundaries for the resident A-3 collection path.

The certified simulator is intentionally mutable: stepping and reset invalidate
the R2 weak-side cache on every tick.  It must therefore never be an argument
to a ``fullgraph`` block.  This module exposes the v8 feature and ego-observation
calculations as functions whose dynamic inputs are tensors only.  Static arena
metadata is retained by :class:`TensorBlocks` and a short-lived view is created
inside the eager function for compatibility with the pinned tensor ports.

The view is deliberately not registered with ``r2_cache``.  Cache resolution
belongs to the mutable assay boundary; the compiled graph sees only the current
state tensors, so it cannot retain a stale generation or cache entry.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch

from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike.ego_device import _component_sizes, _ttl_by_owner
from research.gpu_sim_spike.v8_device import (
    _AREA_MAX,
    _CAP_MAX,
    _CAP_MIN,
    _LANDING_NO_BOOST,
    HEADON_SIZE_RATIO,
    _deltas,
    _own_cells,
    _paint,
    _post_body,
)

# State needed by both pinned tensor ports.  Keep this explicit: adding a
# simulator read to a compiled boundary requires adding its tensor here.
STATE_FIELDS = (
    "bodies",
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "food_cells",
    "food_present",
    "corpse_food",
    "_boosted_this_step",
)


@torch.library.custom_op("sandbox_a_resident::flood", mutates_args=())
def _flood_op(
    start: torch.Tensor,
    blocked: torch.Tensor,
    cap: torch.Tensor,
    release: torch.Tensor,
    has_release: bool,
    max_steps: int,
) -> torch.Tensor:
    """Opaque exact flood boundary: CPU oracle or CUDA native FIFO as appropriate."""
    from research.gpu_sim_spike.v8_device import _flood

    return _flood(start, blocked, cap, release if has_release else None, max_steps=max_steps)


@_flood_op.register_fake
def _flood_op_fake(
    start: torch.Tensor,
    blocked: torch.Tensor,
    cap: torch.Tensor,
    release: torch.Tensor,
    has_release: bool,
    max_steps: int,
) -> torch.Tensor:
    return start.new_empty(start.shape[:-1], dtype=torch.long)


class _TensorSim:
    """Read-only DeviceSim-compatible view constructed wholly from tensors."""

    def __init__(self, metadata: dict[str, Any], values: tuple[torch.Tensor, ...]) -> None:
        for name, value in zip(STATE_FIELDS, values, strict=True):
            setattr(self, name, value)
        self.cfg = metadata["cfg"]
        self.E = values[0].shape[0]
        self.S = metadata["snakes"]
        self.cap = metadata["cap"]
        self.grid_h = metadata["grid_h"]
        self.grid_w = metadata["grid_w"]
        self.food_capacity = metadata["food_capacity"]
        self.device = values[0].device

    def heads(self) -> torch.Tensor:
        env = torch.arange(self.E, device=self.device)[:, None]
        slot = torch.arange(self.S, device=self.device)[None, :]
        return self.bodies[env, slot, self.head_ptr]

    def food_grid(self) -> torch.Tensor:
        index = self.food_cells[..., 1] * self.grid_w + self.food_cells[..., 0]
        valid = (
            self.food_present
            & (self.food_cells[..., 0] >= 0)
            & (self.food_cells[..., 0] < self.grid_w)
        )
        valid &= (self.food_cells[..., 1] >= 0) & (self.food_cells[..., 1] < self.grid_h)
        flat = torch.zeros(
            (self.E, self.grid_h * self.grid_w), dtype=torch.long, device=self.device
        )
        # Ambient food is 1, corpse food is 2, matching DeviceSim.food_grid().
        value = torch.where(self.corpse_food, 2, 1)
        flat.scatter_reduce_(
            1,
            index.clamp(0, self.grid_h * self.grid_w - 1),
            torch.where(valid, value, torch.zeros_like(value)),
            reduce="amax",
            include_self=True,
        )
        return flat.reshape(self.E, self.grid_h, self.grid_w)


class TensorBlocks:
    """Static metadata plus pure-tensor v8 and ego-observation entrypoints."""

    def __init__(self, sim: Any) -> None:
        self._metadata = {
            "cfg": sim.cfg,
            "snakes": int(sim.S),
            "cap": int(sim.cap),
            "grid_h": int(sim.grid_h),
            "grid_w": int(sim.grid_w),
            "food_capacity": int(sim.food_capacity),
        }
        from research.sandbox_a_resident.observation_tensor import build_observation_tensor

        self.observation_tensor = partial(
            build_observation_tensor,
            min_boost_length=int(sim.cfg.min_boost_length),
            boost_length_cost_frames=int(sim.cfg.boost_length_cost_frames),
        )

    @staticmethod
    def state(sim: Any) -> tuple[torch.Tensor, ...]:
        """Resolve mutable simulator state outside a compiled graph."""
        return tuple(getattr(sim, name) for name in STATE_FIELDS)

    @staticmethod
    def v8_bucket_spans(sim: Any) -> tuple[int, ...]:
        """Return every static R2 live-window bucket representable by ``sim``."""
        cap, align = int(sim.cap), r2_cache.ALIGN
        if cap <= align:
            return (cap,)
        spans = list(range(align, cap + 1, align))
        if spans[-1] != cap:
            spans.append(cap)
        return tuple(spans)

    @staticmethod
    def v8_inputs(
        sim: Any, rows: torch.Tensor, *, span: int | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Resolve mutable R2 inputs before entering the pure graph.

        ``span`` is only for excluded compile warmup. It selects an existing
        aligned R2 bucket without changing live state or the runtime window.
        """
        env, slot = rows[:, 0], rows[:, 1]
        if span is None:
            span = r2_cache.window(sim)
        elif span not in TensorBlocks.v8_bucket_spans(sim):
            raise ValueError("v8 warmup span is not an R2 bucket")
        return (
            r2_cache.all_occupancy(sim)[env],
            r2_cache.other_occupancy(sim, env, slot),
            *_own_cells(sim, env, slot, span),
        )

    @staticmethod
    def observation_inputs(sim: Any, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Resolve R2/mutable observation intermediates before graph entry."""
        ttl = _ttl_by_owner(sim)
        return ttl, _component_sizes(sim, rows, ttl), sim.food_grid(), sim.heads()

    def _sim(self, values: tuple[torch.Tensor, ...]) -> _TensorSim:
        return _TensorSim(self._metadata, values)

    def v8_features(
        self,
        *values: torch.Tensor,
        rows: torch.Tensor,
        mask: torch.Tensor,
        all_occ: torch.Tensor,
        other_occ: torch.Tensor,
        own_cells: torch.Tensor,
        own_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the five v8 feature planes from pure current-state tensors.

        This is the subset used by B8.  Final action reranking is deliberately
        omitted: it cannot affect the observation feature planes.
        """
        sim = self._sim(values)
        env, slot = rows[:, 0], rows[:, 1]
        n, h, w = rows.shape[0], sim.grid_h, sim.grid_w
        span = own_cells.shape[1]
        heads = sim.heads()[env, slot]
        direction = sim.direction[env, slot]
        rel3 = torch.arange(3, device=rows.device)[None]
        table_x, table_y = _deltas(rows.device, False)
        hd3 = (direction[:, None] + rel3 - 1) % 4
        delta3 = torch.stack((table_x[hd3], table_y[hd3]), -1)
        starts = heads[:, None] + delta3
        length = sim.length[env, slot]
        cap = (2 * length).clamp(_CAP_MIN, _CAP_MAX)
        need = length.clamp_min(1).minimum(cap)
        empty_release = torch.empty((0,), dtype=torch.long, device=rows.device)
        counts = _flood_op(
            starts,
            all_occ[:, None].expand(-1, 3, -1, -1),
            cap[:, None].expand(-1, 3),
            empty_release,
            False,
            _CAP_MAX,
        )
        spacious = counts >= need[:, None]
        legal = mask.bool()
        legal = torch.where(
            legal.any(1, keepdim=True), legal, torch.arange(6, device=rows.device)[None] < 3
        )
        actions6 = torch.arange(6, device=rows.device)[None].expand(n, -1)
        post_cells, post_valid, lpost, boosted = _post_body(
            sim, env, slot, actions6, own_cells, own_valid, span, False
        )
        head_pos, first_pos = post_cells[:, :, 0], post_cells[:, :, 1]
        post_tail_occ = _paint(
            post_cells[:, :, 1:].reshape(n * 6, span - 1, 2),
            post_valid[:, :, 1:].reshape(n * 6, span - 1),
            h,
            w,
        ).reshape(n, 6, h, w)
        landing = _flood_op(
            head_pos,
            other_occ[:, None] | post_tail_occ,
            cap[:, None].expand(-1, 6),
            empty_release,
            False,
            _CAP_MAX,
        )
        landing = torch.where(
            boosted[:, 3:], landing[:, 3:], torch.full_like(landing[:, 3:], _LANDING_NO_BOOST)
        )
        landing_ok3 = (landing == _LANDING_NO_BOOST) | (landing >= need[:, None])
        v5_eligible = legal & torch.cat((spacious, spacious & landing_ok3), dim=1)
        i = torch.arange(span, device=rows.device)[None, None, :]
        rel_values = (lpost[:, :, None] - i + 1).clamp_min(0)
        release = torch.zeros((n, 6, h * w), dtype=torch.long, device=rows.device)
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
        hi = head_pos[..., 1].clamp(0, h - 1) * w + head_pos[..., 0].clamp(0, w - 1)
        release.reshape(n, 6, -1).scatter_(
            2, hi[..., None], torch.zeros((n, 6, 1), dtype=torch.long, device=rows.device)
        )
        covered = other_occ.flatten(1).sum(1)
        area_cap = (
            (2 * length)
            .clamp_min(_CAP_MIN)
            .minimum(torch.full_like(length, _AREA_MAX))
            .minimum((h * w - covered).clamp_min(1))
        )
        area = _flood_op(
            head_pos,
            other_occ[:, None].expand(-1, 6, -1, -1),
            area_cap[:, None].expand(-1, 6),
            release,
            True,
            _AREA_MAX,
        )
        opp = torch.arange(sim.S, device=rows.device)[None, :] != slot[:, None]
        all_slots = torch.arange(sim.S, device=rows.device)[None].expand(n, -1)
        oh = sim.bodies[env[:, None], all_slots, sim.head_ptr[env[:, None], all_slots]]
        od, ol = sim.direction[env], sim.length[env]
        oa = sim.alive[env] & opp & (sim.seg_count[env] > 0)
        od3 = (od[:, :, None] + torch.arange(3, device=rows.device)[None, None, :] - 1) % 4
        odelta = torch.stack((table_x[od3], table_y[od3]), -1)
        reach = torch.cat((oh[:, :, None], oh[:, :, None] + odelta, oh[:, :, None] + 2 * odelta), 2)
        rvalid = torch.cat(
            (
                oa[:, :, None],
                oa[:, :, None].expand(-1, -1, 3),
                (oa & (ol >= int(sim.cfg.min_boost_length)))[:, :, None].expand(-1, -1, 3),
            ),
            2,
        )
        bound = (ol.clamp_min(1) + 1)[:, :, None].expand_as(rvalid)
        paths = torch.stack((head_pos, first_pos), dim=2)
        path_valid = (
            torch.arange(2, device=rows.device)[None, None, :]
            < torch.where(boosted, 2, 1)[:, :, None]
        )
        equal = (paths[:, :, :, None, None, :] == reach[:, None, None, :, :, :]).all(-1) & rvalid[
            :, None, None
        ]
        bmax = torch.where(
            equal, bound[:, None, None], torch.full_like(bound[:, None, None], -1)
        ).amax(dim=(-1, -2))
        waived_point = (int(getattr(sim.cfg, "mechanics_version", 1)) == 2) & (
            lpost[:, :, None] >= float(HEADON_SIZE_RATIO) * bmax
        )
        risky = ((bmax >= 0) & ~waived_point & path_valid).any(dim=2)
        landing_ok = torch.cat((torch.ones_like(spacious), landing_ok3), dim=1)
        area_g = area.clamp_min(0).minimum(area_cap[:, None]).to(torch.float32) / area_cap[
            :, None
        ].clamp_min(1)
        return torch.stack(
            (
                area_g,
                risky.float(),
                landing_ok.float(),
                v5_eligible.float(),
                torch.cat((spacious, spacious), dim=1).float(),
            ),
            dim=-1,
        ).reshape(n, 30)

    def observation(
        self,
        *values: torch.Tensor,
        ttl: torch.Tensor,
        region: torch.Tensor,
        food_grid: torch.Tensor,
        heads: torch.Tensor,
        rows: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ego2s inputs from tensor-only current-state arguments."""
        (
            bodies,
            head_ptr,
            seg_count,
            length,
            alive,
            direction,
            boost_frames,
            frames_since_food,
            food_cells,
            food_present,
            corpse_food,
            boosted_this_step,
        ) = values
        # Keep only the raw tensors the observation contract reads.  The other
        # simulator tensors are accepted by the sibling v8 block, never turned
        # into a Python simulator view in this fullgraph entrypoint.
        del bodies, head_ptr, food_cells, food_present, corpse_food
        return self.observation_tensor(
            ttl,
            region,
            food_grid,
            heads,
            length,
            seg_count,
            alive,
            direction,
            boost_frames,
            frames_since_food,
            boosted_this_step,
            rows,
            features,
        )
