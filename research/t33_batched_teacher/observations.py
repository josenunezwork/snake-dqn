"""Preallocated ``ego2s-bv8`` observations for the T33 lockstep teacher.

The teacher keeps a fixed number of rollout rows alive.  This module owns one
byte slab for those rows so a caller can retain stable NumPy and Torch views
while rebuilding observations each decision.  Raster pixels are written into
the slab by T33-local ``prange`` wrappers around the frozen geometry kernels.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from research.t33_batched_teacher.observation_kernels import (
    write_enemy_next_prange,
    write_global_prange,
    write_local_prange,
)
from src.simd_env.ego_raster import EgoRasterConfig, _offset_table, _scalars
from src.simd_env.ego_raster_b import b_scalars
from src.simd_env.ego_raster_bv8 import v8_scalars
from src.simd_env.ego_raster_nb import _offset_rc

LOCAL_SHAPE = (7, 31, 31)
GLOBAL_SHAPE = (4, 37, 37)
SCALAR_SHAPE = (60,)


class ObservationBuffer:
    """One aligned CPU slab for a fixed number of ``ego2s-bv8`` rows.

    ``pinned`` is intentionally opt-in: pinned allocations are only useful for
    the CUDA H2D path and can be unavailable in CPU-only test environments.
    The three arrays never own independent storage.
    """

    def __init__(self, rows: int, pinned: bool = False) -> None:
        if rows < 0:
            raise ValueError("rows must be non-negative")
        self.rows = int(rows)
        self.pinned = bool(pinned)
        local_bytes = self.rows * int(np.prod(LOCAL_SHAPE))
        global_bytes = self.rows * int(np.prod(GLOBAL_SHAPE))
        scalar_offset = self._align4(local_bytes + global_bytes)
        scalar_bytes = self.rows * int(np.prod(SCALAR_SHAPE)) * np.dtype(np.float32).itemsize
        self._local_offset = 0
        self._global_offset = local_bytes
        self._global_end = local_bytes + global_bytes
        self._scalar_offset = scalar_offset
        self.storage = torch.empty(
            scalar_offset + scalar_bytes, dtype=torch.uint8, pin_memory=pinned
        )
        raw = self.storage.numpy()
        self.local = raw[self._local_offset : self._global_offset].reshape(
            (self.rows,) + LOCAL_SHAPE
        )
        self.global_ = raw[self._global_offset : self._global_end].reshape(
            (self.rows,) + GLOBAL_SHAPE
        )
        self.scalars = raw[scalar_offset:].view(np.float32).reshape((self.rows,) + SCALAR_SHAPE)
        self.arrays: Dict[str, np.ndarray] = {
            "local": self.local,
            "global": self.global_,
            "scalars": self.scalars,
        }
        self._last_built = 0
        self.zero_()

    @staticmethod
    def _align4(value: int) -> int:
        return (value + 3) & ~3

    def zero_(self) -> None:
        """Clear all output rows, including any padding between array regions."""
        self.storage.zero_()

    def build(self, sim, rows: np.ndarray) -> Dict[str, np.ndarray]:
        """Build exact Bv8 observations for ``rows`` into the reusable slab.

        Destination row ``i`` corresponds to ``rows[i]``.  Only rows which can
        hold stale bytes are cleared, so a resident batch is not memset on every
        search tick.  Local and global planes are written
        directly into the slab by per-row ``prange`` wrappers; scalar helpers
        retain their canonical implementations because they are not rasters.
        """
        selected = np.ascontiguousarray(np.asarray(rows, dtype=np.int64).reshape(-1, 2))
        if len(selected) > self.rows:
            raise ValueError(f"received {len(selected)} rows for capacity {self.rows}")
        count = len(selected)
        # Geometry writers are sparse (outside pixels and dead sources remain
        # untouched).  Clear the selected prefix, then clear only a stale tail
        # when a smaller selection follows a larger one.
        if count:
            self.local[:count].fill(0)
            self.global_[:count].fill(0)
            self.scalars[:count].fill(0.0)
        if count < self._last_built:
            self.local[count : self._last_built].fill(0)
            self.global_[count : self._last_built].fill(0)
            self.scalars[count : self._last_built].fill(0.0)
        self._last_built = count
        if not len(selected):
            return self.arrays
        view = sim.ego_view()
        if (
            selected[:, 0].min() < 0
            or selected[:, 0].max() >= view.E
            or selected[:, 1].min() < 0
            or selected[:, 1].max() >= view.S
        ):
            raise ValueError("rows index outside the (E, S) batch")
        cfg = EgoRasterConfig(version="bv8")
        self._write_local(view, cfg, selected)
        self._write_global(view, cfg, selected)
        self._write_enemy_next(view, cfg, selected)
        base = _scalars(view, cfg, selected)
        b = b_scalars(view, selected, "numba")
        v8, _ = v8_scalars(sim, selected, membership="owner_pad")
        self.scalars[: len(selected)] = np.concatenate((base, b, v8), axis=1)
        return self.arrays

    def _write_local(self, view, cfg: EgoRasterConfig, rows: np.ndarray) -> None:
        """Write channels 0..5 directly into the local slab prefix."""
        owner = np.ascontiguousarray(view.owner_pad)
        slot = np.ascontiguousarray(view.slot_pad)
        food = np.ascontiguousarray(view.food_pad)
        write_local_prange(
            owner.reshape(view.E, -1),
            slot.reshape(view.E, -1),
            food.reshape(view.E, -1),
            np.ascontiguousarray(view.head_ptr, dtype=np.int64),
            np.ascontiguousarray(view.length, dtype=np.int64),
            np.ascontiguousarray(view.alive, dtype=np.bool_),
            np.ascontiguousarray(view.direction, dtype=np.int64),
            np.ascontiguousarray(view.heads, dtype=np.int64),
            rows,
            _offset_table(cfg.local_size, cfg.local_head_row, cfg.local_head_col, owner.shape[2]),
            cfg.local_size,
            cfg.local_head_row,
            cfg.local_head_col,
            int(view.pad),
            owner.shape[2],
            int(view.cap),
            cfg.reach_steps,
            self.local[: len(rows), :6].reshape(len(rows), 6, -1),
        )

    def _write_global(self, view, cfg: EgoRasterConfig, rows: np.ndarray) -> None:
        """Write global planes directly into the global slab prefix."""
        cell = cfg.global_cell
        hc = -(-view.grid_h // cell)
        wc = -(-view.grid_w // cell)
        if (
            view.coarse_cell != cell
            or view.coarse_snake is None
            or view.coarse_food is None
            or view.coarse_snake.shape != (view.E, view.S, hc, wc)
        ):
            raise ValueError("T33 ObservationBuffer requires GridBatchSim maintained coarse counts")
        counts = np.ascontiguousarray(view.coarse_snake)
        food_counts = np.ascontiguousarray(view.coarse_food)
        drow, dcol = _offset_rc(cfg.global_size, cfg.global_size // 2, cfg.global_size // 2)
        write_global_prange(
            counts,
            food_counts,
            np.ascontiguousarray(view.alive, dtype=np.bool_),
            np.ascontiguousarray(view.direction, dtype=np.int64),
            np.ascontiguousarray(view.heads, dtype=np.int64),
            rows,
            drow,
            dcol,
            cell,
            self.global_[: len(rows)].reshape(len(rows), 4, -1),
        )

    def _write_enemy_next(self, view, cfg: EgoRasterConfig, rows: np.ndarray) -> None:
        """Write Bv8 local channel six directly into the local slab prefix."""
        if view.boosting is None:
            raise ValueError("ego2s-b needs EgoGridView.boosting")
        write_enemy_next_prange(
            np.ascontiguousarray(view.alive, dtype=np.bool_),
            np.ascontiguousarray(view.direction, dtype=np.int64),
            np.ascontiguousarray(view.heads, dtype=np.int64),
            np.ascontiguousarray(view.boosting, dtype=np.bool_),
            rows,
            int(view.grid_w),
            int(view.grid_h),
            cfg.local_head_row,
            cfg.local_head_col,
            self.local[: len(rows), 6],
        )

    def tensor_views(self, storage: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Return zero-copy Torch views of this buffer or a compatible byte slab.

        ``storage`` is useful to consumers that retain the same layout in a
        shared-memory or CUDA staging arena.  It must be a contiguous ``uint8``
        tensor of exactly this buffer's byte length.
        """
        source = self.storage if storage is None else storage
        if source.dtype != torch.uint8 or not source.is_contiguous():
            raise TypeError("storage must be a contiguous torch.uint8 tensor")
        if source.numel() != self.storage.numel():
            raise ValueError("storage byte length does not match this ObservationBuffer")
        local = source[self._local_offset : self._global_offset].reshape((self.rows,) + LOCAL_SHAPE)
        glob = source[self._global_offset : self._global_end].reshape((self.rows,) + GLOBAL_SHAPE)
        scalars = (
            source[self._scalar_offset :].view(torch.float32).reshape((self.rows,) + SCALAR_SHAPE)
        )
        return {"local": local, "global": glob, "scalars": scalars}
