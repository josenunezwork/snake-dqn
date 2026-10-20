"""Parallel Vector61 carry runtime used by the T33 many-worlds teacher.

The public contract intentionally matches :class:`Vector61Runtime`.  It only
changes the expensive fresh/carry observation build: independent environments
are partitioned in stable environment order and the already-audited numba
kernel runs once per contiguous partition.  Each invocation owns its kernel
scratch and writes a disjoint output/trend-memory environment slice.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional

import numba
import numpy as np
from numba import njit, prange

from src.simd_env.batch_sim import BatchSim
from src.simd_env.vector61_featurizer import (
    FREE_SPACE_BFS_CAP,
    FREE_SPACE_MIN_CAP,
    Vector61Memory,
    Vector61Params,
)
from src.simd_env.vector61_nb import (
    BUCKET,
    Vector61FeaturizerNB,
    _featurize_kernel,
    _i64,
)
from src.simd_env.vector61_policy import Vector61Runtime

__all__ = ["BatchedVector61Runtime", "ParallelVector61Featurizer"]


@njit(cache=True, nogil=True, parallel=True)
def _featurize_by_environment(
    starts,
    ends,
    order,
    rows_e,
    rows_s,
    out,
    alive,
    length,
    direction,
    seg_count,
    head_ptr,
    bodies,
    cap_ring,
    food_buf,
    food_n,
    lut,
    rx,
    ry,
    onehot,
    card,
    ss,
    width,
    height,
    gw,
    gh,
    max_length,
    nsec,
    maxd,
    max_check,
    use_boundary,
    min_boost,
    expected_per_sector,
    max_dim,
    board_diag,
    use_free_space,
    fs_cap,
    fs_min_cap,
    use_mem,
    update_memory,
    prev_dist,
    prev_id,
    bucket,
):
    """Invoke the scalar kernel over contiguous, disjoint environment partitions."""
    statuses = np.zeros(len(starts), dtype=np.int64)
    for group in prange(len(starts)):
        statuses[group] = _featurize_kernel(
            order[starts[group] : ends[group]],
            rows_e,
            rows_s,
            out,
            alive,
            length,
            direction,
            seg_count,
            head_ptr,
            bodies,
            cap_ring,
            food_buf,
            food_n,
            lut,
            rx,
            ry,
            onehot,
            card,
            ss,
            width,
            height,
            gw,
            gh,
            max_length,
            nsec,
            maxd,
            max_check,
            use_boundary,
            min_boost,
            expected_per_sector,
            max_dim,
            board_diag,
            use_free_space,
            fs_cap,
            fs_min_cap,
            use_mem,
            update_memory,
            prev_dist,
            prev_id,
            bucket,
        )
    return statuses


class ParallelVector61Featurizer(Vector61FeaturizerNB):
    """Numba Vector61 featurizer partitioned by environment for T33 only.

    Food staging deliberately reuses ``Vector61FeaturizerNB._food_arrays``:
    the ordered food list is semantically significant for nearest-food ties.
    On any kernel fallback, trend memory is restored before invoking the
    canonical inherited implementation for the whole request.
    """

    def __init__(self, params: Vector61Params) -> None:
        super().__init__(params)
        self.timings = {"food_s": 0.0, "kernel_s": 0.0, "calls": 0}

    def featurize(
        self,
        sim: BatchSim,
        rows: Optional[np.ndarray] = None,
        memory: Optional[Vector61Memory] = None,
        update_memory: bool = True,
    ) -> np.ndarray:
        """Return canonical Vector61 rows, parallel only across environments."""
        if sim.cfg.game_width != self.W or sim.cfg.game_height != self.H:
            raise ValueError("sim world size does not match the featurizer params")
        if sim.s != self.ss:
            raise ValueError("sim segment_size does not match the featurizer params")
        if rows is None:
            rows = np.argwhere(sim.alive)
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        out = np.zeros((len(rows), self.input_size), dtype=np.float64)
        if not len(rows):
            return out.astype(np.float32)
        if memory is not None and update_memory:
            if len(np.unique(rows[:, 0] * sim.S + rows[:, 1])) != len(rows):
                raise ValueError("rows must be unique when updating trend memory")

        rows_e = np.ascontiguousarray(rows[:, 0])
        rows_s = np.ascontiguousarray(rows[:, 1])
        order = np.argsort(rows_e, kind="stable")
        sorted_envs = rows_e[order]
        env_starts = np.r_[0, np.flatnonzero(np.diff(sorted_envs)) + 1].astype(np.int64)
        partition_count = min(len(env_starts), numba.get_num_threads())
        partition_groups = np.linspace(0, len(env_starts), partition_count + 1, dtype=np.int64)
        starts = env_starts[partition_groups[:-1]]
        ends = np.empty(partition_count, dtype=np.int64)
        for index, group_end in enumerate(partition_groups[1:]):
            ends[index] = len(order) if group_end == len(env_starts) else env_starts[group_end]
        food_started = time.perf_counter()
        food_buf, food_n = self._food_arrays(sim, np.unique(rows_e))
        self.timings["food_s"] += time.perf_counter() - food_started

        use_mem = memory is not None
        if use_mem:
            prev_dist, prev_id = memory.prev_dist, memory.prev_id
            saved = (prev_dist.copy(), prev_id.copy())
        else:
            prev_dist = np.zeros((1, 1), dtype=np.float64)
            prev_id = np.zeros((1, 1), dtype=np.int64)
            saved = None
        p = self.p
        kernel_started = time.perf_counter()
        kernel_args = (
            rows_e,
            rows_s,
            out,
            _i64(sim.alive, np.bool_),
            _i64(sim.length),
            _i64(sim.direction),
            _i64(sim.seg_count),
            _i64(sim.head_ptr),
            _i64(sim.bodies),
            int(sim.cap),
            food_buf,
            food_n,
            self._lut,
            int(self.rx),
            int(self.ry),
            self._onehot,
            self._card,
            int(self.ss),
            int(self.W),
            int(self.H),
            int(self.gw),
            int(self.gh),
            int(p.max_length),
            int(p.num_sectors),
            int(self.maxd),
            int(self.max_check),
            bool(p.use_boundary_as_danger),
            int(p.min_boost_length),
            float(self.expected_per_sector),
            int(self.max_dim),
            float(self.board_diagonal),
            bool(p.use_free_space),
            int(FREE_SPACE_BFS_CAP),
            int(FREE_SPACE_MIN_CAP),
            use_mem,
            bool(update_memory),
            prev_dist,
            prev_id,
            int(BUCKET),
        )
        if partition_count == 1:
            # Keep the one-thread path on the exact canonical call shape; the
            # parallel wrapper adds scheduling overhead when no partition exists.
            statuses = np.asarray([_featurize_kernel(order, *kernel_args)], dtype=np.int64)
        else:
            statuses = _featurize_by_environment(starts, ends, order, *kernel_args)
        self.timings["kernel_s"] += time.perf_counter() - kernel_started
        self.timings["calls"] += 1
        if np.any(statuses):
            self.fallbacks += 1
            if saved is not None:
                prev_dist[...] = saved[0]
                prev_id[...] = saved[1]
            return super().featurize(sim, rows, memory, update_memory)
        return out.astype(np.float32)


class BatchedVector61Runtime(Vector61Runtime):
    """Vector61 carry runtime with T33's environment-parallel featurizer."""

    def __init__(self, params: Optional[Vector61Params] = None, trace=None) -> None:
        super().__init__(params=params, trace=trace, backend="numba")
        self.timings = {"prepare_s": 0.0, "observe_s": 0.0, "prepare_calls": 0, "observe_calls": 0}

    def bind(self, sim: BatchSim, controlled: np.ndarray) -> None:
        """Bind canonical carry storage then install isolated parallel featurizers."""
        super().bind(sim, controlled)
        assert self.featurizer is not None
        params = self.featurizer.p
        self.featurizer = ParallelVector61Featurizer(params)
        self.free_space_featurizer = (
            self.featurizer
            if params.use_free_space
            else ParallelVector61Featurizer(replace(params, use_free_space=True))
        )
        self.timings = {"prepare_s": 0.0, "observe_s": 0.0, "prepare_calls": 0, "observe_calls": 0}

    def prepare(self, sim: BatchSim, rows_mask: Optional[np.ndarray] = None) -> None:
        """Prepare selections and accumulate wall time for the T33 profile."""
        started = time.perf_counter()
        super().prepare(sim, rows_mask=rows_mask)
        self.timings["prepare_s"] += time.perf_counter() - started
        self.timings["prepare_calls"] += 1

    def observe_step(self, sim: BatchSim, rows_mask: Optional[np.ndarray] = None) -> None:
        """Capture post-step carry and accumulate wall time for the T33 profile."""
        started = time.perf_counter()
        super().observe_step(sim, rows_mask=rows_mask)
        self.timings["observe_s"] += time.perf_counter() - started
        self.timings["observe_calls"] += 1

    def profile(self) -> dict[str, float | int]:
        """Return parent wall time plus nested featurizer timing counters.

        ``prepare_s`` and ``observe_s`` include ``food_s`` and ``kernel_s``;
        consumers must not add parent and child counters together.
        """
        assert isinstance(self.featurizer, ParallelVector61Featurizer)
        return {**self.timings, **self.featurizer.timings}
