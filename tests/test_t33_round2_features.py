"""Focused round-two contracts for T33's masked vector carry runtime."""

from __future__ import annotations

import numpy as np

from research.t33_batched_teacher.runtime import BatchedVector61Runtime
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.v8_nb import V8Result, v8_decide
from src.simd_env.vector61_featurizer import Vector61Params
from src.simd_env.vector61_nb import Vector61FeaturizerNB


def _runtime() -> tuple[BatchSim, BatchedVector61Runtime]:
    sim = BatchSim(
        BatchSimConfig(
            num_envs=3,
            num_snakes=3,
            game_width=300,
            game_height=200,
            initial_food=8,
            max_food=12,
            max_capacity=80,
        ),
        seeds=[71, 72, 73],
        train_mode=True,
        allow_respawn=True,
    )
    runtime = BatchedVector61Runtime()
    runtime.bind(sim, np.ones((sim.E, sim.S), dtype=bool))
    return sim, runtime


def _env_bytes(runtime: BatchedVector61Runtime, env: int) -> bytes:
    assert runtime.memory is not None
    arrays = (
        runtime._carry_state,
        runtime._carry_mask,
        runtime._carry_valid,
        runtime._alive_post,
        runtime.sel_state,
        runtime.sel_mask,
        runtime.sel_ready,
        runtime.sel_fresh,
        runtime.memory.prev_dist,
        runtime.memory.prev_id,
    )
    return b"".join(array[env].tobytes() for array in arrays)


def test_rows_mask_leaves_inactive_carry_and_trend_memory_byte_identical() -> None:
    """Leader-only prepare/observe does not mutate a dormant follower environment."""
    sim, runtime = _runtime()
    all_rows = np.ones(sim.E, dtype=bool)
    runtime.prepare(sim, rows_mask=all_rows)
    sim.step(np.ones((sim.E, sim.S), dtype=np.int64))
    runtime.observe_step(sim, rows_mask=all_rows)
    dormant_before = _env_bytes(runtime, 1)

    leaders = np.array([True, False, True])
    runtime.prepare(sim, rows_mask=leaders)
    sim.step(np.full((sim.E, sim.S), 2, dtype=np.int64))
    runtime.observe_step(sim, rows_mask=leaders)

    assert _env_bytes(runtime, 1) == dormant_before
    assert runtime.sel_ready[1].all()


def test_rows_mask_validates_environment_shape() -> None:
    sim, runtime = _runtime()
    with np.testing.assert_raises_regex(ValueError, "rows_mask must have shape"):
        runtime.prepare(sim, rows_mask=np.ones((sim.E, 1), dtype=bool))


def test_vector61_food_staging_uses_grid_ordered_array_without_list_copy() -> None:
    """The GridBatchSim mirror is the featurizer input and preserves list order."""
    sim, _ = _runtime()
    grid = GridBatchSim(sim.cfg, seeds=[71, 72, 73], train_mode=True, allow_respawn=True)
    featurizer = Vector61FeaturizerNB(Vector61Params.from_game_config(grid.cfg))
    food, count = featurizer._food_arrays(grid, np.array([0, 2], dtype=np.int64))
    assert food is grid._food_array
    assert count is grid._food_n
    for env in (0, 2):
        n = int(count[env])
        assert np.array_equal(food[env, :n], np.asarray(grid.food_cells[env], dtype=np.int64))


def test_v8_owner_pad_matches_painted_reference_on_grid_worlds() -> None:
    """The owner-pad fast route matches painted body membership on valid grid worlds."""
    sim, _ = _runtime()
    grid = GridBatchSim(sim.cfg, seeds=[71, 72, 73], train_mode=True, allow_respawn=True)
    rows = np.column_stack((np.arange(grid.E), np.zeros(grid.E, dtype=np.int64)))
    q = np.array([[0.2, 0.1, 0.3, -0.1, 0.0, -0.2]] * grid.E, dtype=np.float32)
    mask = grid.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]
    painted = v8_decide(grid, rows, q, mask, features=True, membership="painted")
    owner = v8_decide(grid, rows, q, mask, features=True, membership="owner_pad")
    for name in V8Result.__dataclass_fields__:
        assert getattr(owner, name).tobytes() == getattr(painted, name).tobytes(), name
