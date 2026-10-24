"""Indexed vector61 non-flood geometry adapter.

The dense :mod:`vector_device` implementation is the frozen feature contract.
It intentionally splits rows into small blocks, but each block used to rebuild
the same full-world ring-ordered body tensor.  This adapter makes precisely one
indexed ring snapshot for one public ``features`` invocation and presents it to
the unchanged feature arithmetic.  Consequently the numerical reductions,
row order, and vector61 flood columns continue to be owned by the dense/FLOOD
implementation while the repeated all-world gather disappears.

The adapter is deliberately opt-in.  Benchmark callers wrap an already-bound
runtime with :meth:`IndexedVectorRuntime.from_runtime`; no policy or simulator
module needs to import this experimental implementation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch

from research.gpu_sim_spike.vector_device import DeviceVectorRuntime

BodyView = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

_EXTENSION: object | None = None


def _native_body_view(sim: Any, device: torch.device) -> Optional[BodyView]:
    """Return a native ring view on CUDA, leaving CPU as a deterministic oracle.

    Compilation is opt-in because the benchmark's Mac preflight must never
    build CUDA sources.  A requested native CUDA path fails closed instead of
    silently presenting the Torch gather as an A100 native measurement.
    """
    if device.type != "cuda" or os.environ.get("GPU_SIM_VECTOR_GEOMETRY", "torch") != "native":
        return None
    global _EXTENSION
    if _EXTENSION is None:
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError("GPU_SIM_VECTOR_GEOMETRY=native requires CUDA PyTorch")
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        _EXTENSION = load(
            name="snake_dqn_vector_geometry_v1",
            sources=[str(root / "vector_geometry.cpp"), str(root / "vector_geometry.cu")],
            extra_cflags=["-O3"],
            # No fast-math or fmad contraction: this native part only moves
            # integer ring cells/masks, and feature arithmetic stays explicit.
            extra_cuda_cflags=["-O3", "--fmad=false"],
            verbose=os.environ.get("GPU_SIM_SPIKE_CUDA_VERBOSE") == "1",
        )
    cells, live, self_ok, other_ok = _EXTENSION.indexed_body_view(
        sim.bodies.contiguous(),
        sim.head_ptr.contiguous(),
        sim.seg_count.contiguous(),
        sim.length.contiguous(),
        sim.alive.contiguous(),
    )
    return cells, live, self_ok, other_ok


class IndexedVectorRuntime(DeviceVectorRuntime):
    """A ``DeviceVectorRuntime`` which reuses one all-world indexed body view.

    ``features`` may still process rows in bounded chunks.  All chunks observe
    the exact same immutable pre-feature snapshot, including the free-space
    builder, so updating nearest-enemy carry remains in the parent method's
    original input order.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._geometry_sim: Optional[Any] = None
        self._geometry_body_view: Optional[BodyView] = None

    @classmethod
    def from_runtime(cls, runtime: DeviceVectorRuntime) -> "IndexedVectorRuntime":
        """Clone a bound runtime's observable state into the indexed adapter.

        Tensor state is cloned so an A/B packet can retain its dense arm without
        aliasing carry, selection buffers, or counters.  Immutable parameter
        objects and the setup-time sector lookup remain shared by reference.
        """
        out = cls(getattr(runtime, "_params", None))
        for key, value in runtime.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(out, key, value.clone())
            else:
                setattr(out, key, value)
        out._geometry_sim = None
        out._geometry_body_view = None
        return out

    def _indexed_body_view(self, sim: Any) -> BodyView:
        """Materialize ring order once across every resident world and slot.

        The gather indexes each slot's ring directly from ``head_ptr``.  It is
        intentionally the same indexing and tail eligibility expression as the
        frozen parent implementation; only its lifetime differs.
        """
        if self.device is None:
            raise RuntimeError("IndexedVectorRuntime called before bind")
        native = _native_body_view(sim, self.device)
        if native is not None:
            return native
        _e, _s, cap = sim.bodies.shape[:3]
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

    def _body_view(self, sim: Any) -> BodyView:
        """Return the current call's indexed snapshot, or the parent fallback."""
        if self._geometry_sim is sim and self._geometry_body_view is not None:
            return self._geometry_body_view
        return super()._body_view(sim)

    def _features_rows(
        self, sim: Any, rows: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        """Use direct CUDA rows only for the frozen static, unique-row contract."""
        if (
            self.device is None
            or self.device.type != "cuda"
            or os.environ.get("GPU_SIM_VECTOR_GEOMETRY", "torch") != "native"
            or os.environ.get("GPU_SIM_VECTOR_FLOOD", "dense") != "static"
            or rows is None
        ):
            return super()._features_rows(sim, rows, **kwargs)
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device).reshape(-1, 2)
        ids = rows[:, 0] * int(sim.S) + rows[:, 1]
        if int(torch.unique(ids).numel()) != len(ids) and kwargs.get("update_memory", True):
            raise ValueError("rows must be unique when updating trend memory")
        assert self.params is not None and self.prev_dist is not None and self.prev_id is not None
        if (
            tuple(self.params.actions) != ((0, -1), (1, 0), (0, 1), (-1, 0))
            or self.params.num_sectors != 16
        ):
            raise ValueError("native geometry requires the frozen cardinal vector61 contract")
        torch._assert_async(
            (
                (rows[:, 0] >= 0) & (rows[:, 0] < sim.E) & (rows[:, 1] >= 0) & (rows[:, 1] < sim.S)
            ).all(),
            "native geometry row outside world",
        )
        torch._assert_async(
            (
                (sim.head_ptr >= 0)
                & (sim.head_ptr < sim.cap)
                & (sim.seg_count >= 0)
                & (sim.seg_count <= sim.cap)
                & (sim.direction >= 0)
                & (sim.direction < 4)
            ).all(),
            "native geometry invalid ring state",
        )
        memory_mask = kwargs.get("memory_mask")
        if memory_mask is None:
            memory_mask = torch.ones(len(rows), dtype=torch.bool, device=self.device)
        result, next_dist, next_id = _EXTENSION.indexed_features58(  # type: ignore[union-attr]
            sim.bodies.contiguous(),
            sim.head_ptr.contiguous(),
            sim.seg_count.contiguous(),
            sim.length.contiguous(),
            sim.alive.contiguous(),
            sim.direction.contiguous(),
            sim.food_cells.contiguous(),
            sim.food_present.contiguous(),
            rows.contiguous(),
            self.prev_dist.contiguous(),
            self.prev_id.contiguous(),
            self._cpu_sector_lut.contiguous(),
            self._cpu_sector_rx,
            self._cpu_sector_ry,
            int(sim.grid_w),
            int(sim.grid_h),
            int(self.params.segment_size),
            self.params.num_sectors,
            self.params.max_length,
            self.params.danger_max_distance,
            self.params.min_boost_length,
            self.params.food_capacity,
            self.params.use_boundary_as_danger,
        )
        er, sr = rows[:, 0], rows[:, 1]
        valid = sim.alive[er, sr]
        if kwargs.get("update_memory", True):
            store = torch.as_tensor(memory_mask, dtype=torch.bool, device=self.device) & valid
            self.prev_dist[er, sr] = torch.where(store, next_dist, self.prev_dist[er, sr])
            self.prev_id[er, sr] = torch.where(store, next_id, self.prev_id[er, sr])
        free_space = kwargs.get("free_space")
        if free_space is None:
            free_space = self._free_space_all(sim)
        result = torch.cat((result, free_space[er, sr] * valid[:, None]), dim=1)
        return torch.where(valid[:, None], result, torch.zeros_like(result))

    def features(
        self,
        sim: Any,
        rows: Optional[torch.Tensor] = None,
        *,
        update_memory: bool = True,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute frozen vector61 features using a single ring snapshot.

        ``super().features`` still invokes the unmodified flood path for
        columns 58:61.  ``finally`` guarantees callers cannot accidentally
        reuse stale geometry after the simulator moves or respawns a snake.
        """
        self._check()
        self._geometry_sim = sim
        self._geometry_body_view = self._indexed_body_view(sim)
        try:
            native = (
                self.device is not None
                and self.device.type == "cuda"
                and os.environ.get("GPU_SIM_VECTOR_GEOMETRY", "torch") == "native"
                and os.environ.get("GPU_SIM_VECTOR_FLOOD", "dense") == "static"
            )
            if native:
                if rows is None:
                    er = (
                        torch.arange(sim.E, device=self.device)[:, None]
                        .expand(sim.E, sim.S)
                        .reshape(-1)
                    )
                    sr = (
                        torch.arange(sim.S, device=self.device)[None, :]
                        .expand(sim.E, sim.S)
                        .reshape(-1)
                    )
                    rows = torch.stack((er, sr), dim=1)
                rows = torch.as_tensor(rows, dtype=torch.long, device=self.device).reshape(-1, 2)
                ids = rows[:, 0] * int(sim.S) + rows[:, 1]
                if int(torch.unique(ids).numel()) != len(ids) and update_memory:
                    # The canonical CPU featurizer rejects this invalid call;
                    # never silently fall back to CPU-inexact legacy arithmetic.
                    raise ValueError("rows must be unique when updating trend memory")
                if int(torch.unique(ids).numel()) == len(ids) or not update_memory:
                    if memory_mask is None:
                        memory_mask = torch.ones(len(rows), dtype=torch.bool, device=self.device)
                    free_space = (
                        self._free_space_all(sim)
                        if self.params and self.params.use_free_space
                        else None
                    )
                    return self._features_rows(
                        sim,
                        rows,
                        update_memory=update_memory,
                        memory_mask=memory_mask,
                        free_space=free_space,
                    )
            return super().features(sim, rows, update_memory=update_memory, memory_mask=memory_mask)
        finally:
            self._geometry_sim = None
            self._geometry_body_view = None
