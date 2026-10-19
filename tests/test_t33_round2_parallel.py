"""Explicit two-thread private-scratch parity for the new feature kernels."""

import numba
import numpy as np
import pytest

from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.v8_nb import V8Result, v8_decide


def test_v8_serial_parallel_painted_owner_bytes_after_reuse() -> None:
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("run serialized with NUMBA_NUM_THREADS>=2")
    original = numba.get_num_threads()
    cfg = BatchSimConfig(
        num_envs=8,
        num_snakes=3,
        game_width=300,
        game_height=200,
        initial_food=12,
        max_food=16,
        max_capacity=100,
    )
    sim = GridBatchSim(cfg, seeds=list(range(8)), train_mode=True, allow_respawn=True)
    # Include duplicate and non-grouped rows to exercise output reordering and
    # independent scratch when several selected seats share one environment.
    rows = np.array([[7, 0], [1, 1], [0, 0], [7, 0], [1, 0], [3, 2], [2, 1], [5, 0]])
    q = np.zeros((len(rows), 6), dtype=np.float32)
    try:
        for _ in range(3):
            mask = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]
            numba.set_num_threads(1)
            reference = v8_decide(sim, rows, q, mask, features=True, membership="painted")
            for membership in ("painted", "owner_pad"):
                numba.set_num_threads(2)
                actual = v8_decide(sim, rows, q, mask, features=True, membership=membership)
                for name in V8Result.__dataclass_fields__:
                    assert (
                        getattr(actual, name).tobytes() == getattr(reference, name).tobytes()
                    ), name
            sim.step(np.ones((sim.E, sim.S), dtype=np.int64))
    finally:
        numba.set_num_threads(original)
