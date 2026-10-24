"""Torch-resident vector61 state and carry runtime for the GPU simulator spike.

This is deliberately separate from :mod:`src.simd_env.vector61_policy`: that
module is the frozen CPU evaluator contract.  The spike needs the same *state
layout* on a ``DeviceSim`` without a hidden host featurizer call. Both dense
and static flood backends use the CPU sector lookup and pixel-space
float64 intermediates, with one final float32 cast. CUDA feature arithmetic
uses explicit native round-to-nearest operations rather than torch division.
Neither path establishes cross-platform policy or whole-engine identity.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Optional

import numpy as np
import torch

from research.gpu_sim_spike import flood_cuda
from src.simd_env.vector61_featurizer import (
    FREE_SPACE_BFS_CAP,
    FREE_SPACE_MIN_CAP,
    Vector61Params,
    _sector_lut,
)

_CARDINAL = torch.tensor(((0, -1), (1, 0), (0, 1), (-1, 0)), dtype=torch.long)

# ``features`` materializes a long-coordinate distance tensor for every row,
# action, snake, and stored body cell.  At the A1 ruler maximum this is
# [6144, 3, 6, 10400, 2] int64: 17.14 GiB before its derived tensors exist.
# Keep the working set bounded independently of the resident world count.
FEATURE_ROW_CHUNK = 64


class DeviceVectorRuntime:
    """Device equivalent of the CPU vector61 selection/carry lifecycle.

    Call :meth:`bind`, then ``prepare -> selection -> sim.step ->
    observe_step`` for each rollout frame.  All mutable state, including the
    nearest-enemy trend baseline, remains on ``sim.device``.  The only host work
    is setup-time construction of :class:`Vector61Params`.
    """

    def __init__(self, params: Optional[Vector61Params] = None) -> None:
        self._params = params
        self.params: Optional[Vector61Params] = None
        self.device: Optional[torch.device] = None
        self.controlled: Optional[torch.Tensor] = None
        self.prev_dist: Optional[torch.Tensor] = None
        self.prev_id: Optional[torch.Tensor] = None
        self._carry_state: Optional[torch.Tensor] = None
        self._carry_mask: Optional[torch.Tensor] = None
        self._carry_valid: Optional[torch.Tensor] = None
        self._alive_post: Optional[torch.Tensor] = None
        self.sel_state: Optional[torch.Tensor] = None
        self.sel_mask: Optional[torch.Tensor] = None
        self.sel_ready: Optional[torch.Tensor] = None
        self.sel_fresh: Optional[torch.Tensor] = None
        self._cpu_sector_lut: Optional[torch.Tensor] = None
        self._cpu_sector_rx = 0
        self._cpu_sector_ry = 0
        self._free_ratio: Optional[torch.Tensor] = None
        # Device counters intentionally remain tensors: reading them per frame
        # would synchronize CUDA and make a host stage invisible in the spike.
        self.fresh_rows: int | torch.Tensor = 0
        self.carried_rows: int | torch.Tensor = 0

    def bind(self, sim: Any, controlled: Optional[torch.Tensor] = None) -> None:
        """Attach to a new fixed-shape world without copying state to host."""
        if getattr(sim.cfg, "arena_type", "rectangular") != "rectangular":
            raise NotImplementedError("DeviceVectorRuntime supports rectangular arenas only")
        requested = self._params or Vector61Params.from_game_config(sim.cfg)
        # This package feeds the frozen vector61 opponent checkpoint.  A
        # process-global evaluator config may currently select vector58, but it
        # must not silently shrink this resident policy input to 58 columns.
        self.params = (
            requested if requested.use_free_space else replace(requested, use_free_space=True)
        )
        self.device = torch.device(sim.device)
        e, s = int(sim.E), int(sim.S)
        if controlled is None:
            controlled = torch.ones((e, s), dtype=torch.bool, device=self.device)
        controlled = torch.as_tensor(controlled, device=self.device, dtype=torch.bool)
        if controlled.shape != (e, s):
            raise ValueError(f"controlled must have shape {(e, s)}, got {tuple(controlled.shape)}")
        self.controlled = controlled.clone()
        shape = (e, s)
        trend_dtype = torch.float64
        self.prev_dist = torch.full(shape, torch.inf, device=self.device, dtype=trend_dtype)
        self.prev_id = torch.full(shape, -1, dtype=torch.long, device=self.device)
        width = self.params.input_size
        self._carry_state = torch.zeros((*shape, width), dtype=torch.float32, device=self.device)
        self._carry_mask = torch.zeros((*shape, 6), dtype=torch.bool, device=self.device)
        self._carry_valid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._alive_post = sim.alive.to(self.device, dtype=torch.bool).clone()
        self.sel_state = torch.zeros((*shape, width), dtype=torch.float32, device=self.device)
        self.sel_mask = torch.zeros((*shape, 6), dtype=torch.bool, device=self.device)
        self.sel_ready = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.sel_fresh = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.fresh_rows = torch.zeros((), dtype=torch.long, device=self.device)
        self.carried_rows = torch.zeros((), dtype=torch.long, device=self.device)
        # Both repaired backends use the CPU setup-time integer sector LUT.
        lut, rx, ry = _sector_lut(
            sim.grid_w, sim.grid_h, int(self.params.segment_size), self.params.num_sectors
        )
        self._cpu_sector_lut = torch.as_tensor(lut, device=self.device, dtype=torch.long)
        self._cpu_sector_rx, self._cpu_sector_ry = int(rx), int(ry)
        # Finite integer count/cap domain: compute CPU float64 division once
        # and upload its final float32 bytes. CUDA does only integer indexing.
        counts = np.arange(FREE_SPACE_BFS_CAP + 1, dtype=np.int64)
        caps = np.maximum(counts, 1)
        ratios = np.minimum(counts[None, :] / caps[:, None], 1.0).astype(np.float32)
        self._free_ratio = torch.as_tensor(ratios, device=self.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.set_float32_matmul_precision("highest")

    @classmethod
    def from_reference(cls, runtime: Any, sim: Any) -> "DeviceVectorRuntime":
        """Clone a CPU ``Vector61Runtime`` carry/trend state at setup time.

        This is deliberately a one-time bridge used while a branch is forked
        from an already-running reference world.  It reads NumPy host arrays
        here, never in ``prepare``/``observe_step`` or policy evaluation.
        """
        params = getattr(runtime, "_params", None)
        if params is None and getattr(runtime, "featurizer", None) is not None:
            params = runtime.featurizer.p
        out = cls(params)
        controlled = getattr(runtime, "controlled", None)
        out.bind(sim, controlled)
        mapping = {
            "prev_dist": ("memory", "prev_dist"),
            "prev_id": ("memory", "prev_id"),
            "_carry_state": ("_carry_state",),
            "_carry_mask": ("_carry_mask",),
            "_carry_valid": ("_carry_valid",),
            "_alive_post": ("_alive_post",),
            "sel_state": ("sel_state",),
            "sel_mask": ("sel_mask",),
            "sel_ready": ("sel_ready",),
            "sel_fresh": ("sel_fresh",),
        }
        for target, path in mapping.items():
            source = runtime
            for attr in path:
                source = getattr(source, attr)
            if source is not None:
                setattr(out, target, torch.as_tensor(source, device=out.device).clone())
        return out

    def _check(self) -> None:
        if self.params is None or self.device is None or self.controlled is None:
            raise RuntimeError("DeviceVectorRuntime called before bind")

    def _body_view(self, sim: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ring cells, live entries, self-safe and other-safe masks."""
        e, s, cap = sim.bodies.shape[:3]
        k = torch.arange(cap, device=self.device)[None, None, :]
        ring = (sim.head_ptr.to(self.device)[..., None] - k) % cap
        cells = sim.bodies.to(self.device).gather(2, ring[..., None].expand(-1, -1, -1, 2))
        n = sim.seg_count.to(self.device)[..., None]
        length = sim.length.to(self.device)[..., None]
        live = sim.alive.to(self.device)[..., None] & (k < n)
        drop_other = (n > 1) & (n >= length)
        other_ok = live & (k < n - drop_other.to(n.dtype))
        drop_self = (n >= length) & (n > 2)
        self_ok = live & (k >= 2) & (k < n - drop_self.to(n.dtype))
        return cells, live, self_ok, other_ok

    def _static_sectors(self, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """CPU vector61's integer sector lookup for both repaired backends."""
        assert self._cpu_sector_lut is not None
        ix = (dx.to(torch.long) + self._cpu_sector_rx).clamp(0, self._cpu_sector_lut.shape[0] - 1)
        iy = (dy.to(torch.long) + self._cpu_sector_ry).clamp(0, self._cpu_sector_lut.shape[1] - 1)
        return self._cpu_sector_lut[ix, iy]

    def _free_space(
        self, sim: Any, heads: torch.Tensor, direction: torch.Tensor, occupied: torch.Tensor
    ) -> torch.Tensor:
        """Capped, fixed-iteration 4-connected flood for the three relative moves."""
        assert self.params is not None
        e, s = heads.shape[:2]
        h, w = int(sim.grid_h), int(sim.grid_w)
        dirs = (direction[..., None] + torch.tensor((-1, 0, 1), device=self.device)) % 4
        delta = _CARDINAL.to(self.device)[dirs]
        starts = heads[..., None, :] + delta
        sx, sy = starts[..., 0], starts[..., 1]
        inside = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        # CPU _BatchView marks every live full body, including the observer's
        # head, before it starts the candidate flood.  Do not clear that head:
        # doing so joins regions through a cell that remains occupied at the
        # decision instant.
        blocked = occupied[:, None].expand(-1, s, -1, -1).clone()
        frontier = torch.zeros((e, s, 3, h, w), dtype=torch.bool, device=self.device)
        flat = (sy.clamp(0, h - 1) * w + sx.clamp(0, w - 1)).reshape(e, s, 3, 1)
        frontier.flatten(-2).scatter_(3, flat, inside[..., None])
        frontier &= ~blocked[:, :, None]
        seen = frontier.clone()
        # A tensor loop has a static bound; CUDA graph / compile sees fixed shapes.
        for _ in range(FREE_SPACE_BFS_CAP - 1):
            north = torch.nn.functional.pad(frontier[..., :-1, :], (0, 0, 1, 0))
            south = torch.nn.functional.pad(frontier[..., 1:, :], (0, 0, 0, 1))
            west = torch.nn.functional.pad(frontier[..., :, :-1], (1, 0, 0, 0))
            east = torch.nn.functional.pad(frontier[..., :, 1:], (0, 1, 0, 0))
            frontier = (north | south | west | east) & ~seen & ~blocked[:, :, None]
            seen |= frontier
        count = seen.flatten(-2).sum(-1)
        cap = torch.minimum(
            torch.full_like(sim.length, FREE_SPACE_BFS_CAP),
            torch.clamp(sim.length.to(self.device) * 2, min=FREE_SPACE_MIN_CAP),
        ).to(torch.long)
        return self._normalize_free(count, cap)

    def _normalize_free(self, count: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
        """Lookup the CPU final cast for every possible capped integer ratio."""
        assert self._free_ratio is not None
        return self._free_ratio[cap[..., None], torch.minimum(count, cap[..., None])]

    def _free_space_all(self, sim: Any) -> torch.Tensor:
        """Compute the full resident flood result once for a feature call.

        Flood planes are world-shaped and modest compared with the row-local
        all-body distance tensor.  Building them here prevents row chunking
        from repeating the same E x S x 3 flood for every chunk.
        """
        cells, live_body, _self_ok, _other_ok = self._body_view(sim)
        e = int(sim.E)
        occupied = torch.zeros((e, sim.grid_h, sim.grid_w), dtype=torch.uint8, device=self.device)
        all_cells = cells.reshape(e, -1, 2)
        all_live = live_body.reshape(e, -1)
        ox, oy = all_cells[..., 0], all_cells[..., 1]
        inb = all_live & (ox >= 0) & (ox < sim.grid_w) & (oy >= 0) & (oy < sim.grid_h)
        occupied.flatten(1).scatter_reduce_(
            1,
            (oy.clamp(0, sim.grid_h - 1) * sim.grid_w + ox.clamp(0, sim.grid_w - 1)),
            inb.to(torch.uint8),
            "amax",
            include_self=True,
        )
        backend = os.environ.get("GPU_SIM_VECTOR_FLOOD", "dense").lower()
        if backend == "static":
            # The static vector61 contract is cardinal occupancy at the
            # decision instant, capped at 160.  Keep it separate from v8's
            # release-aware FIFO and ego's uncapped components.
            if self.device is not None and self.device.type == "cuda":
                heads = cells[..., 0, :]
                direction = sim.direction.to(self.device, dtype=torch.long)
                dirs = (direction[..., None] + torch.tensor((-1, 0, 1), device=self.device)) % 4
                starts = heads[..., None, :] + _CARDINAL.to(self.device)[dirs]
                cap = torch.minimum(
                    torch.full_like(sim.length, FREE_SPACE_BFS_CAP),
                    torch.clamp(sim.length.to(self.device) * 2, min=FREE_SPACE_MIN_CAP),
                ).to(torch.long)
                count = flood_cuda.static_capped(starts, occupied != 0, cap)
                return self._normalize_free(count, cap)
            # CPU is an oracle-only fallback, preserving import/test usability.
            return self._free_space(
                sim,
                cells[..., 0, :],
                sim.direction.to(self.device, dtype=torch.long),
                occupied != 0,
            )
        if backend != "dense":
            raise ValueError("GPU_SIM_VECTOR_FLOOD must be 'dense' or 'static'")
        return self._free_space(
            sim,
            cells[..., 0, :],
            sim.direction.to(self.device, dtype=torch.long),
            occupied != 0,
        )

    def features(
        self,
        sim: Any,
        rows: Optional[torch.Tensor] = None,
        *,
        update_memory: bool = True,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return vector61 features while bounding row-local scratch memory.

        The resident simulator remains full-shape.  Only the row-local
        geometry reductions are partitioned, in input order, so the returned
        rows and per-row trend updates have the same values as one unchunked
        call.  This is intentionally not a world/episode split.
        """
        self._check()
        assert self.params is not None
        e, s = int(sim.E), int(sim.S)
        caller_rows = rows is not None
        if rows is None:
            # Fixed E*S rows are essential for CUDA-graph shape checking and
            # avoid compaction/nonzero synchronization in the rollout loop.
            er0 = torch.arange(e, device=self.device)[:, None].expand(e, s).reshape(-1)
            sr0 = torch.arange(s, device=self.device)[None, :].expand(e, s).reshape(-1)
            rows = torch.stack((er0, sr0), dim=1)
        rows = torch.as_tensor(rows, device=self.device, dtype=torch.long).reshape(-1, 2)
        if memory_mask is None:
            memory_mask = torch.ones((len(rows),), dtype=torch.bool, device=self.device)
        memory_mask = torch.as_tensor(memory_mask, dtype=torch.bool, device=self.device).reshape(-1)
        if memory_mask.shape != (len(rows),):
            raise ValueError("memory_mask must have one entry per feature row")
        if not len(rows):
            return torch.zeros((0, self.params.input_size), device=self.device, dtype=torch.float32)
        # Duplicate indexed CUDA writes have no stable cross-kernel ordering.
        # The A1 full matrix uses generated, unique E*S rows.  Preserve the
        # legacy one-kernel path for an explicit duplicate caller instead of
        # splitting its writes across row blocks.
        if caller_rows and update_memory and len(rows) > FEATURE_ROW_CHUNK:
            row_ids = rows[:, 0] * s + rows[:, 1]
            if int(torch.unique(row_ids).numel()) != len(row_ids):
                return self._features_rows(
                    sim, rows, update_memory=update_memory, memory_mask=memory_mask
                )
        # Keep this world-shaped BFS result outside the row loop.  The chunked
        # geometry below is the allocation which caused the A100 OOM.
        free_space = self._free_space_all(sim) if self.params.use_free_space else None
        if len(rows) <= FEATURE_ROW_CHUNK:
            return self._features_rows(
                sim,
                rows,
                update_memory=update_memory,
                memory_mask=memory_mask,
                free_space=free_space,
            )
        parts = []
        for start in range(0, len(rows), FEATURE_ROW_CHUNK):
            stop = min(start + FEATURE_ROW_CHUNK, len(rows))
            parts.append(
                self._features_rows(
                    sim,
                    rows[start:stop],
                    update_memory=update_memory,
                    memory_mask=memory_mask[start:stop],
                    free_space=free_space,
                )
            )
        return torch.cat(parts, dim=0)

    def _features_rows(
        self,
        sim: Any,
        rows: Optional[torch.Tensor] = None,
        *,
        update_memory: bool = True,
        memory_mask: Optional[torch.Tensor] = None,
        free_space: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return every vector61 column for ``(env, slot)`` rows on device.

        Dead rows are zero, mirroring the CPU featurizer.  Duplicate rows are
        rejected when updating trend memory because the live call sequence has
        one observation per snake per frame.
        """
        self._check()
        assert self.params is not None and self.prev_dist is not None and self.prev_id is not None
        e, s = int(sim.E), int(sim.S)
        if rows is None:
            # Fixed E*S rows are essential for CUDA-graph shape checking and
            # avoid compaction/nonzero synchronization in the rollout loop.
            er0 = torch.arange(e, device=self.device)[:, None].expand(e, s).reshape(-1)
            sr0 = torch.arange(s, device=self.device)[None, :].expand(e, s).reshape(-1)
            rows = torch.stack((er0, sr0), dim=1)
        rows = torch.as_tensor(rows, device=self.device, dtype=torch.long).reshape(-1, 2)
        if memory_mask is None:
            memory_mask = torch.ones((len(rows),), dtype=torch.bool, device=self.device)
        memory_mask = torch.as_tensor(memory_mask, dtype=torch.bool, device=self.device).reshape(-1)
        if memory_mask.shape != (len(rows),):
            raise ValueError("memory_mask must have one entry per feature row")
        out = torch.zeros(
            (len(rows), self.params.input_size), device=self.device, dtype=torch.float32
        )
        if not len(rows):
            return out
        if self.device is not None and self.device.type == "cuda":
            from research.gpu_sim_spike.vector_geometry import exact_features_rows

            return exact_features_rows(
                self,
                sim,
                rows,
                update_memory=update_memory,
                memory_mask=memory_mask,
                free_space=free_space,
            )
        er, sr = rows[:, 0], rows[:, 1]
        math_dtype = torch.float64
        distance_scale = float(self.params.segment_size)
        alive_row = sim.alive.to(self.device)[er, sr]
        cells, live_body, self_ok, other_ok = self._body_view(sim)
        heads = cells[..., 0, :]
        direction = sim.direction.to(self.device, dtype=torch.long)
        length = sim.length.to(self.device, dtype=torch.long)
        hrow, drow, lrow = heads[er, sr], direction[er, sr], length[er, sr]
        ar = torch.arange(len(rows), device=self.device)
        valid = alive_row
        onehot = torch.zeros((len(rows), 4), dtype=torch.float32, device=self.device)
        onehot.scatter_(1, drow[:, None], 1.0)
        out[:, :4] = onehot * valid[:, None]
        logical_length = lrow.clamp_min(1)
        out[:, 4] = (logical_length.to(math_dtype) / float(self.params.max_length)).clamp(
            max=1
        ) * valid

        # Ordered food list: argmin's first-index tie rule matches the CPU list contract.
        food = sim.food_cells.to(self.device)
        present = sim.food_present.to(self.device)
        fd = food[er] - hrow[:, None]
        fd2 = fd.square().sum(-1).to(math_dtype).masked_fill(~present[er], torch.inf)
        nearest = fd2.argmin(-1)
        fdelta = fd[ar, nearest]
        has_food = present[er].any(-1)
        max_dim = float(
            max(self.params.game_width, self.params.game_height) // self.params.segment_size
        )
        diag = float((sim.grid_w**2 + sim.grid_h**2) ** 0.5)
        fd2 = fd2 * (distance_scale**2)
        diag = float((self.params.game_width**2 + self.params.game_height**2) ** 0.5)
        out[:, 5] = torch.where(
            has_food,
            fdelta[:, 0].to(math_dtype) / max_dim,
            torch.zeros_like(lrow, dtype=torch.float32),
        )
        out[:, 6] = torch.where(
            has_food,
            fdelta[:, 1].to(math_dtype) / max_dim,
            torch.zeros_like(lrow, dtype=torch.float32),
        )
        out[:, 7] = torch.where(
            has_food, fd2[ar, nearest].sqrt() / diag, torch.ones_like(lrow, dtype=torch.float32)
        ).clamp(max=1)
        fsector = self._static_sectors(fd[..., 0], fd[..., 1])
        fone = (
            torch.nn.functional.one_hot(fsector, self.params.num_sectors).to(torch.long)
            * present[er, :, None]
        )
        expected = max(float(self.params.food_capacity) / self.params.num_sectors, 1.0)
        out[:, 8:24] = (fone.sum(1).to(math_dtype) / expected).clamp(max=1) * valid[:, None]

        # Danger map includes every body segment selected by the CPU self/other tail rules.
        body = cells[er]
        owner = torch.arange(s, device=self.device)[None, :, None].expand(
            len(rows), -1, body.shape[2]
        )
        mask = (self_ok[er] & (owner == sr[:, None, None])) | (
            other_ok[er] & (owner != sr[:, None, None])
        )
        bdelta = body - hrow[:, None, None]
        bdist2 = bdelta.square().sum(-1).to(math_dtype)
        bdist = (bdist2 * distance_scale**2).sqrt()
        maxd = float(self.params.danger_max_distance) * distance_scale
        near = mask & (bdist <= maxd)
        sec = self._static_sectors(bdelta[..., 0], bdelta[..., 1])
        values = (1.0 - bdist / maxd).clamp(min=0) * 1.5
        danger = torch.zeros(
            (len(rows), self.params.num_sectors), device=self.device, dtype=math_dtype
        )
        danger.scatter_reduce_(
            1,
            sec.reshape(len(rows), -1),
            torch.where(near, values, torch.zeros_like(values)).reshape(len(rows), -1),
            "amax",
            include_self=True,
        )
        count = torch.zeros_like(danger).scatter_add_(
            1,
            sec.reshape(len(rows), -1),
            (near & (values > 0)).reshape(len(rows), -1).to(math_dtype),
        )
        if self.params.use_boundary_as_danger:
            offsets = torch.arange(
                -int(self.params.danger_max_distance),
                int(self.params.danger_max_distance) + 1,
                device=self.device,
            )
            wx = torch.cat(
                (
                    -torch.ones((len(rows), len(offsets)), device=self.device, dtype=torch.long),
                    torch.full(
                        (len(rows), len(offsets)),
                        int(sim.grid_w) + 1,
                        device=self.device,
                        dtype=torch.long,
                    ),
                    hrow[:, 0, None] + offsets,
                    hrow[:, 0, None] + offsets,
                ),
                dim=1,
            )
            wy = torch.cat(
                (
                    hrow[:, 1, None] + offsets,
                    hrow[:, 1, None] + offsets,
                    -torch.ones((len(rows), len(offsets)), device=self.device, dtype=torch.long),
                    torch.full(
                        (len(rows), len(offsets)),
                        int(sim.grid_h) + 1,
                        device=self.device,
                        dtype=torch.long,
                    ),
                ),
                dim=1,
            )
            offset_ok_y = ((hrow[:, 1, None] + offsets) >= 0) & (
                (hrow[:, 1, None] + offsets) < sim.grid_h
            )
            offset_ok_x = ((hrow[:, 0, None] + offsets) >= 0) & (
                (hrow[:, 0, None] + offsets) < sim.grid_w
            )
            wall_valid = torch.cat(
                (
                    (hrow[:, 0, None] < int(self.params.danger_max_distance) + 1) & offset_ok_y,
                    ((sim.grid_w - hrow[:, 0, None]) < int(self.params.danger_max_distance) + 1)
                    & offset_ok_y,
                    (hrow[:, 1, None] < int(self.params.danger_max_distance) + 1) & offset_ok_x,
                    ((sim.grid_h - hrow[:, 1, None]) < int(self.params.danger_max_distance) + 1)
                    & offset_ok_x,
                ),
                dim=1,
            )
            wall_delta = torch.stack((wx - hrow[:, 0, None], wy - hrow[:, 1, None]), -1)
            wall_dist2 = wall_delta.square().sum(-1).to(math_dtype)
            wall_dist = (wall_dist2 * distance_scale**2).sqrt()
            wall_near = wall_valid & (wall_dist <= maxd)
            wall_sec = self._static_sectors(wall_delta[..., 0], wall_delta[..., 1])
            wall_values = (1.0 - wall_dist / maxd).clamp(min=0) * 2.0
            danger.scatter_reduce_(
                1,
                wall_sec,
                torch.where(wall_near, wall_values, torch.zeros_like(wall_values)),
                "amax",
                include_self=True,
            )
            count.scatter_add_(1, wall_sec, (wall_near & (wall_values > 0)).to(math_dtype))
        danger = torch.where(
            count > 1, (danger + (count / 5).clamp(max=1) * 0.2).clamp(max=2), danger
        ).clamp(max=1)
        out[:, 24:40] = danger.to(torch.float32) * valid[:, None]
        x, y = hrow[:, 0].to(math_dtype), hrow[:, 1].to(math_dtype)
        out[:, 40] = x / float(sim.grid_w)
        out[:, 41] = (float(sim.grid_w) - x) / float(sim.grid_w)
        out[:, 42] = y / float(sim.grid_h)
        out[:, 43] = (float(sim.grid_h) - y) / float(sim.grid_h)

        # Stable nearest/second enemy: torch.argsort is stable and slots are input ordered.
        hd = heads[er] - hrow[:, None]
        enemy_dist2 = hd.square().sum(-1).to(math_dtype)
        enemy_dist = (enemy_dist2 * distance_scale**2).sqrt()
        candidate = sim.alive.to(self.device)[er] & (
            torch.arange(s, device=self.device)[None] != sr[:, None]
        )
        order = torch.argsort(enemy_dist.masked_fill(~candidate, torch.inf), dim=1, stable=True)
        first, second = order[:, 0], order[:, min(1, s - 1)]
        n_enemy = candidate.sum(1)
        has1, has2 = n_enemy >= 1, n_enemy >= 2
        first_delta, second_delta = hd[ar, first], hd[ar, second]
        own = lrow.clamp_min(1).to(math_dtype)
        lens = length[er].clamp_min(1).to(math_dtype)
        first_dir = _CARDINAL.to(self.device)[direction[er, first]]
        out[:, 44] = torch.where(has1, first_delta[:, 0].to(math_dtype) / max_dim, 0.0)
        out[:, 45] = torch.where(has1, first_delta[:, 1].to(math_dtype) / max_dim, 0.0)
        out[:, 46] = torch.where(has1, (lens[ar, first] / own).clamp(max=2) / 2, 0.0)
        out[:, 47:49] = first_dir.float() * has1[:, None]
        d1 = enemy_dist[ar, first]
        same = torch.isfinite(self.prev_dist[er, sr]) & (self.prev_id[er, sr] == first)
        trend = torch.where(
            d1 < self.prev_dist[er, sr], 1.0, torch.where(d1 > self.prev_dist[er, sr], -1.0, 0.0)
        )
        out[:, 49] = torch.where(has1 & same, trend, 0.0)
        out[:, 50] = torch.where(has2, second_delta[:, 0].to(math_dtype) / max_dim, 0.0)
        out[:, 51] = torch.where(has2, second_delta[:, 1].to(math_dtype) / max_dim, 0.0)
        out[:, 52] = torch.where(has2, (lens[ar, second] / own).clamp(max=2) / 2, 0.0)
        future = heads[er, first] + first_dir
        out[:, 53] = (has1 & ((hrow - future).square().sum(-1) < 9)).float()
        if update_memory:
            store = memory_mask & valid
            self.prev_dist[er, sr] = torch.where(
                store, torch.where(has1, d1, torch.inf), self.prev_dist[er, sr]
            )
            self.prev_id[er, sr] = torch.where(
                store, torch.where(has1, first, -torch.ones_like(first)), self.prev_id[er, sr]
            )

        dirs3 = (drow[:, None] + torch.tensor((-1, 0, 1), device=self.device)) % 4
        next_cells = hrow[:, None] + _CARDINAL.to(self.device)[dirs3]
        inside = (
            (next_cells[..., 0] >= 0)
            & (next_cells[..., 0] < sim.grid_w)
            & (next_cells[..., 1] >= 0)
            & (next_cells[..., 1] < sim.grid_h)
        )
        delta = next_cells[:, :, None, None] - body[:, None]
        ddist2 = delta.square().sum(-1).to(math_dtype)
        ddist = (ddist2 * distance_scale**2).sqrt()
        body_mask = mask[:, None]
        hit = (~inside) | ((ddist < distance_scale) & body_mask).any((-1, -2))
        nearest = (
            torch.minimum(
                torch.minimum(next_cells[..., 0], (sim.grid_w - next_cells[..., 0]).abs()),
                torch.minimum(next_cells[..., 1], (sim.grid_h - next_cells[..., 1]).abs()),
            )
            .to(math_dtype)
            .clamp(max=3)
        )
        nearest = nearest * distance_scale
        obstacle = ddist.masked_fill(~body_mask, torch.inf).amin((-1, -2))
        nearest = torch.minimum(nearest, obstacle)
        out[:, 54:57] = (
            torch.where(
                hit,
                torch.ones_like(nearest),
                torch.where(
                    nearest < 3 * distance_scale,
                    (1 - nearest / (3 * distance_scale)).clamp(max=0.95),
                    torch.zeros_like(nearest),
                ),
            )
            * valid[:, None]
        )
        out[:, 57] = (lrow >= int(self.params.min_boost_length)).float() * valid
        if self.params.use_free_space:
            if free_space is None:
                free_space = self._free_space_all(sim)
            out[:, 58:61] = free_space[er, sr] * valid[:, None]
        # The CPU oracle emits canonical positive zero in every dead row.
        out = torch.where(valid[:, None], out, torch.zeros_like(out))
        return out

    def _mask(self, sim: Any) -> torch.Tensor:
        """Six-action advisory mask built from the same device danger geometry."""
        provided = getattr(sim, "advisory_mask", None)
        if callable(provided):
            return provided().to(self.device, dtype=torch.bool)
        states = self.features(sim, update_memory=False)
        normal = states[:, 54:57] < 1.0
        boost = (
            normal & (sim.length.to(self.device) >= int(self.params.min_boost_length))[..., None]
        )
        return torch.cat((normal, boost), dim=-1)

    def prepare(self, sim: Any, rows_mask: Optional[torch.Tensor] = None) -> None:
        self._check()
        assert (
            self.controlled is not None
            and self._carry_valid is not None
            and self._alive_post is not None
        )
        assert (
            self.sel_state is not None
            and self.sel_mask is not None
            and self.sel_ready is not None
            and self.sel_fresh is not None
        )
        env_mask = (
            torch.ones(sim.E, dtype=torch.bool, device=self.device)
            if rows_mask is None
            else torch.as_tensor(rows_mask, device=self.device, dtype=torch.bool)
        )
        if env_mask.shape != (sim.E,):
            raise ValueError(f"rows_mask must have shape {(sim.E,)}, got {tuple(env_mask.shape)}")
        affected = self.controlled & env_mask[:, None]
        alive = sim.alive.to(self.device, dtype=torch.bool)
        respawned = (alive & ~self._alive_post).any(1)
        ready = affected & alive
        carried = ready & self._carry_valid & ~respawned[:, None]
        fresh = ready & ~carried
        # Keep every update full-shape.  Boolean advanced indexing lowers to a
        # dynamic nonzero/compaction operation and breaks CUDA fake/meta shape
        # checks even when the logical state update is otherwise resident.
        next_state = torch.where(
            affected[..., None], torch.zeros_like(self.sel_state), self.sel_state
        )
        next_mask = torch.where(affected[..., None], torch.zeros_like(self.sel_mask), self.sel_mask)
        next_state = torch.where(carried[..., None], self._carry_state, next_state)
        next_mask = torch.where(carried[..., None], self._carry_mask, next_mask)
        states = self.features(sim, update_memory=True, memory_mask=fresh.reshape(-1)).reshape_as(
            self.sel_state
        )
        masks = self._mask(sim)
        self.sel_state = torch.where(fresh[..., None], states, next_state)
        self.sel_mask = torch.where(fresh[..., None], masks, next_mask)
        self._carry_valid = torch.where(
            affected, torch.zeros_like(self._carry_valid), self._carry_valid
        )
        self.sel_ready = torch.where(affected, ready, self.sel_ready)
        self.sel_fresh = torch.where(affected, fresh, self.sel_fresh)
        self.fresh_rows = self.fresh_rows + fresh.sum()
        self.carried_rows = self.carried_rows + carried.sum()

    def observe_step(self, sim: Any, rows_mask: Optional[torch.Tensor] = None) -> None:
        self._check()
        assert (
            self.controlled is not None
            and self._carry_valid is not None
            and self._alive_post is not None
        )
        env_mask = (
            torch.ones(sim.E, dtype=torch.bool, device=self.device)
            if rows_mask is None
            else torch.as_tensor(rows_mask, device=self.device, dtype=torch.bool)
        )
        affected = self.controlled & env_mask[:, None]
        done = getattr(sim, "_last_done", torch.zeros_like(sim.alive)).to(
            self.device, dtype=torch.bool
        )
        reset = done & affected
        self.prev_dist = torch.where(
            reset, torch.full_like(self.prev_dist, torch.inf), self.prev_dist
        )
        self.prev_id = torch.where(reset, torch.full_like(self.prev_id, -1), self.prev_id)
        alive = sim.alive.to(self.device, dtype=torch.bool)
        capture = affected & alive
        states = self.features(sim, update_memory=True, memory_mask=capture.reshape(-1)).reshape_as(
            self._carry_state
        )
        masks = self._mask(sim)
        self._carry_state = torch.where(capture[..., None], states, self._carry_state)
        self._carry_mask = torch.where(capture[..., None], masks, self._carry_mask)
        self._carry_valid = torch.where(affected, capture, self._carry_valid)
        self._alive_post = torch.where(env_mask[:, None], alive, self._alive_post)

    def selection(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check()
        assert (
            self.sel_ready is not None and self.sel_state is not None and self.sel_mask is not None
        )
        slots = torch.as_tensor(slots, device=self.device, dtype=torch.long).reshape(-1, 2)
        return self.sel_state[slots[:, 0], slots[:, 1]], self.sel_mask[slots[:, 0], slots[:, 1]]

    def repeat_interleave(self, repeats: int) -> "DeviceVectorRuntime":
        if repeats < 1:
            raise ValueError("repeats must be positive")
        self._check()
        out = DeviceVectorRuntime(self.params)
        for key, value in self.__dict__.items():
            if key == "_cpu_sector_lut":
                # Setup-time immutable lookup has no environment axis.
                setattr(out, key, value)
                continue
            if (
                isinstance(value, torch.Tensor)
                and value.ndim
                and value.shape[0] == self.controlled.shape[0]
            ):
                setattr(out, key, value.repeat_interleave(repeats, dim=0))
            else:
                setattr(out, key, value)
        return out
