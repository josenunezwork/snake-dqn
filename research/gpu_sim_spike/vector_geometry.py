"""Indexed body-gather adapter and shared CPU-order CUDA feature arithmetic.

Both flood backends use the same explicitly rounded feature58 kernel. This
adapter additionally hoists the immutable body view for one feature call; the
native geometry mode avoids row chunking without changing the CPU contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch

from research.gpu_sim_spike import r2_cache
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
    extension = _load_exact_extension()
    cells, live, self_ok, other_ok = extension.indexed_body_view(
        sim.bodies.contiguous(),
        sim.head_ptr.contiguous(),
        sim.seg_count.contiguous(),
        sim.length.contiguous(),
        sim.alive.contiguous(),
    )
    return cells, live, self_ok, other_ok


def _load_exact_extension() -> Any:
    """Load CPU-order RN arithmetic; CUDA never falls back to torch floats."""
    global _EXTENSION
    if _EXTENSION is None:
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise RuntimeError("GPU_SIM_VECTOR_GEOMETRY=native requires CUDA PyTorch")
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        _EXTENSION = load(
            name="snake_dqn_vector_geometry_exact_v2",
            sources=[str(root / "vector_geometry.cpp"), str(root / "vector_geometry.cu")],
            extra_cflags=["-O3"],
            # No fast-math or FMA contraction in integer views or feature arithmetic.
            extra_cuda_cflags=["-O3", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"],
            verbose=os.environ.get("GPU_SIM_SPIKE_CUDA_VERBOSE") == "1",
        )
    return _EXTENSION


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

    def features(
        self,
        sim: Any,
        rows: Optional[torch.Tensor] = None,
        *,
        update_memory: bool = True,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute CPU-exact vector61 features using a single ring snapshot.

        ``super().features`` still invokes the unmodified flood path for
        columns 58:61.  ``finally`` guarantees callers cannot accidentally
        reuse stale geometry after the simulator moves or respawns a snake.
        """
        self._check()
        native = (
            self.device is not None
            and self.device.type == "cuda"
            and os.environ.get("GPU_SIM_VECTOR_GEOMETRY", "torch") == "native"
        )
        self._geometry_sim = sim
        if r2_cache.enabled(sim):
            # R2: native features read the rings directly and free space uses
            # the shared per-state occupancy, so the full-capacity native view
            # is never read there.  Other routes take one live-window view.
            self._geometry_body_view = None if native else self._body_view(sim)
        else:
            self._geometry_body_view = self._indexed_body_view(sim)
        try:
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


def exact_features_rows(
    runtime: DeviceVectorRuntime, sim: Any, rows: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    """Shared exact CUDA feature58 arithmetic for dense and static flood backends."""
    rows = torch.as_tensor(rows, dtype=torch.long, device=runtime.device).reshape(-1, 2)
    ids = rows[:, 0] * int(sim.S) + rows[:, 1]
    if int(torch.unique(ids).numel()) != len(ids) and kwargs.get("update_memory", True):
        raise ValueError("rows must be unique when updating trend memory")
    assert (
        runtime.params is not None and runtime.prev_dist is not None and runtime.prev_id is not None
    )
    if (
        tuple(runtime.params.actions) != ((0, -1), (1, 0), (0, 1), (-1, 0))
        or runtime.params.num_sectors != 16
    ):
        raise ValueError("native geometry requires the frozen cardinal vector61 contract")
    torch._assert_async(
        ((rows[:, 0] >= 0) & (rows[:, 0] < sim.E) & (rows[:, 1] >= 0) & (rows[:, 1] < sim.S)).all(),
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
        memory_mask = torch.ones(len(rows), dtype=torch.bool, device=runtime.device)
    result, next_dist, next_id = _load_exact_extension().indexed_features58(
        sim.bodies.contiguous(),
        sim.head_ptr.contiguous(),
        sim.seg_count.contiguous(),
        sim.length.contiguous(),
        sim.alive.contiguous(),
        sim.direction.contiguous(),
        sim.food_cells.contiguous(),
        sim.food_present.contiguous(),
        rows.contiguous(),
        runtime.prev_dist.contiguous(),
        runtime.prev_id.contiguous(),
        runtime._cpu_sector_lut.contiguous(),
        runtime._cpu_sector_rx,
        runtime._cpu_sector_ry,
        int(sim.grid_w),
        int(sim.grid_h),
        int(runtime.params.segment_size),
        runtime.params.num_sectors,
        runtime.params.max_length,
        runtime.params.danger_max_distance,
        runtime.params.min_boost_length,
        runtime.params.food_capacity,
        runtime.params.use_boundary_as_danger,
    )
    er, sr = rows[:, 0], rows[:, 1]
    valid = sim.alive[er, sr]
    if kwargs.get("update_memory", True):
        store = torch.as_tensor(memory_mask, dtype=torch.bool, device=runtime.device) & valid
        runtime.prev_dist[er, sr] = torch.where(store, next_dist, runtime.prev_dist[er, sr])
        runtime.prev_id[er, sr] = torch.where(store, next_id, runtime.prev_id[er, sr])
    if runtime.params.use_free_space:
        free_space = kwargs.get("free_space")
        if free_space is None:
            free_space = runtime._free_space_all(sim)
        result = torch.cat((result, free_space[er, sr] * valid[:, None]), dim=1)
    return torch.where(valid[:, None], result, torch.zeros_like(result))
