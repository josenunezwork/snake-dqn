"""Explicit two-thread private-scratch parity for the new feature kernels."""

import numba
import numpy as np
import pytest

from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.v8_nb import V8Result, v8_decide


def _assert_result_bytes(actual: V8Result, expected: V8Result) -> None:
    for name in V8Result.__dataclass_fields__:
        assert getattr(actual, name).tobytes() == getattr(expected, name).tobytes(), name


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
                _assert_result_bytes(actual, reference)
            sim.step(np.ones((sim.E, sim.S), dtype=np.int64))
    finally:
        numba.set_num_threads(original)


def test_v8_direct_owner_matches_painted_l800_fixture_with_duplicate_rows() -> None:
    """The direct accessor retains the painted oracle on the real long-body boundary."""
    import json

    from tests.test_t33_teacher_integration import B8_CHECKPOINT, WORLD_BANK, _prepared_world

    if not B8_CHECKPOINT.is_file():
        pytest.skip("the hash-pinned local B8 checkpoint inventory is unavailable")
    bank = json.loads(WORLD_BANK.read_text(encoding="utf-8"))
    row = next(item for item in bank["fixtures"]["timing"] if item["mix"] == "scripted")
    sim = _prepared_world(row, 800).sim
    alive = np.flatnonzero(sim.alive[0])
    rows = np.column_stack((np.zeros(len(alive) + 2, dtype=np.int64), np.r_[alive, alive[:2]]))
    q = np.arange(len(rows) * 6, dtype=np.float32).reshape(len(rows), 6) / 19.0
    mask = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]
    painted = v8_decide(sim, rows, q, mask, features=True, membership="painted")
    owner = v8_decide(sim, rows, q, mask, features=True, membership="owner_pad")
    _assert_result_bytes(owner, painted)
