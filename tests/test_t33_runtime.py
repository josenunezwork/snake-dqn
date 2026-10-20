"""T33 carry-runtime parity against the canonical Vector61 runtime."""

from __future__ import annotations

import os

import numpy as np

# This suite is deliberately one native thread: it proves the parallel wrapper's
# semantics without consuming the Mac capacity reserved for the other T33 lanes.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

from research.t33_batched_teacher.runtime import (  # noqa: E402
    BatchedVector61Runtime,
    ParallelVector61Featurizer,
)
from src.simd_env.batch_sim import BatchSim, BatchSimConfig  # noqa: E402
from src.simd_env.vector61_featurizer import Vector61Params  # noqa: E402
from src.simd_env.vector61_nb import Vector61FeaturizerNB  # noqa: E402
from src.simd_env.vector61_policy import Vector61Runtime  # noqa: E402


def _sim() -> BatchSim:
    cfg = BatchSimConfig(
        num_envs=4,
        num_snakes=4,
        game_width=300,
        game_height=200,
        initial_food=12,
        max_food=16,
        max_capacity=80,
    )
    return BatchSim(cfg, seeds=[11, 12, 13, 14], train_mode=True, allow_respawn=True)


def _assert_runtime_bytes(reference: Vector61Runtime, candidate: BatchedVector61Runtime) -> None:
    for name in (
        "_carry_state",
        "_carry_mask",
        "_carry_valid",
        "_alive_post",
        "sel_state",
        "sel_mask",
        "sel_ready",
        "sel_fresh",
    ):
        left, right = getattr(reference, name), getattr(candidate, name)
        assert left.dtype == right.dtype and left.shape == right.shape, name
        assert left.tobytes() == right.tobytes(), name
    assert reference.memory is not None and candidate.memory is not None
    for name in ("prev_dist", "prev_id"):
        left, right = getattr(reference.memory, name), getattr(candidate.memory, name)
        assert left.dtype == right.dtype and left.shape == right.shape, name
        assert left.tobytes() == right.tobytes(), name


def test_batched_runtime_carry_matches_numba_reference_through_mutations_and_respawn() -> None:
    """Fresh/carry, food ordering, death reset, and respawn invalidation stay byte-identical."""
    reference_sim, candidate_sim = _sim(), _sim()
    controlled = np.ones((reference_sim.E, reference_sim.S), dtype=bool)
    reference = Vector61Runtime(backend="numba")
    candidate = BatchedVector61Runtime()
    reference.bind(reference_sim, controlled)
    candidate.bind(candidate_sim, controlled)
    assert isinstance(candidate.featurizer, ParallelVector61Featurizer)
    assert isinstance(candidate.free_space_featurizer, ParallelVector61Featurizer)

    for frame in range(7):
        reference.prepare(reference_sim)
        candidate.prepare(candidate_sim)
        _assert_runtime_bytes(reference, candidate)

        # The list ordering is load-bearing for nearest-food ties.  Reversing it
        # drives the private staging cache down its changed-list path.
        if frame == 1:
            reference_sim.food_cells[1].reverse()
            candidate_sim.food_cells[1].reverse()
        actions = np.full((reference_sim.E, reference_sim.S), frame % 3, dtype=np.int64)
        reference_sim.step(actions)
        candidate_sim.step(actions)
        reference.observe_step(reference_sim)
        candidate.observe_step(candidate_sim)
        _assert_runtime_bytes(reference, candidate)

        if frame == 3:
            # Simulate the post-step death observation and the next Watch seam
            # respawn.  This directly exercises the runtime's cache invalidation
            # contract without relying on a probabilistic collision schedule.
            reference_sim.alive[2, 1] = False
            candidate_sim.alive[2, 1] = False
            reference_sim._last_done[2, 1] = True
            candidate_sim._last_done[2, 1] = True
            reference.observe_step(reference_sim)
            candidate.observe_step(candidate_sim)
            _assert_runtime_bytes(reference, candidate)
            reference_sim.alive[2, 1] = True
            candidate_sim.alive[2, 1] = True

    profile = candidate.profile()
    assert profile["prepare_calls"] == 7
    assert profile["observe_calls"] == 8
    assert profile["calls"] > 0
    assert profile["food_s"] >= 0.0 and profile["kernel_s"] >= 0.0


def test_parallel_featurizer_preserves_unsorted_row_order_and_food_cache_scope() -> None:
    """Rows may be shuffled; a bind to another sim must not retain old food staging."""
    first, second = _sim(), _sim()
    candidate = ParallelVector61Featurizer(Vector61Params.from_game_config(first.cfg))
    reference = Vector61FeaturizerNB(candidate.p)
    rows = np.array([[3, 2], [0, 1], [2, 3], [0, 0], [1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(
        reference.featurize(first, rows, None, update_memory=False),
        candidate.featurize(first, rows, None, update_memory=False),
    )
    assert candidate._food_sim_id == id(first)
    second.food_cells[0].reverse()
    np.testing.assert_array_equal(
        reference.featurize(second, rows, None, update_memory=False),
        candidate.featurize(second, rows, None, update_memory=False),
    )
    assert candidate._food_sim_id == id(second)
