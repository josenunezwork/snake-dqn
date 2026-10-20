"""Round-2 contracts for T33's grid food mirror and reusable observation slab."""

from __future__ import annotations

import numpy as np

from research.t33_batched_teacher.observations import ObservationBuffer
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_sim import GridBatchSim


def _sim() -> GridBatchSim:
    sim = GridBatchSim(
        BatchSimConfig(num_envs=2, num_snakes=3, game_width=80, game_height=60, max_capacity=128),
        seeds=[901, 902],
        train_mode=True,
        allow_respawn=False,
    )
    sim.step(np.ones((sim.E, sim.S), dtype=np.int64))
    return sim


def test_ordered_food_mirror_is_zero_copy_ready_and_tracks_ordered_mutations():
    sim = _sim()
    for env in range(sim.E):
        count = int(sim._food_n[env])
        assert count == len(sim.food_cells[env])
        assert np.array_equal(sim._food_array[env, :count], np.asarray(sim.food_cells[env]))

    env = 0
    old_version = int(sim._food_version[env])
    cell = next(cell for cell in sim.food_cells[env] if cell not in sim.corpse_cells[env])
    sim._food[env, cell[1], cell[0]] = 1
    # Drive the canonical consume path with an intentionally selected traversed cell.
    sim._trav[env, 0, 0] = cell
    sim._trav_valid[env, 0, 0] = True
    sim._trav_valid[env, 0, 1] = False
    sim._active_envs = lambda: np.array([True, False])
    sim._consume_food()
    count = int(sim._food_n[env])
    assert int(sim._food_version[env]) > old_version
    assert np.array_equal(sim._food_array[env, :count], np.asarray(sim.food_cells[env]))
    sim.check_grid_invariants()


def test_observation_build_clears_only_reused_rows_and_keeps_reference_bytes():
    sim = _sim()
    rows = np.array([[0, 0], [1, 1]], dtype=np.int64)
    buf = ObservationBuffer(4)
    # A full slab clear is allowed at construction, never during a build.
    buf.zero_ = lambda: (_ for _ in ()).throw(AssertionError("whole slab clear"))
    got = buf.build(sim, rows)
    ref = build_ego_raster(sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
    for name in ("local", "global", "scalars"):
        assert got[name][: len(rows)].tobytes() == ref[name].tobytes()
    buf.build(sim, rows[:1])
    for value in buf.arrays.values():
        assert not value[1:].any()


def test_global_enemy_mass_clamps_to_uint8_without_narrowing_coarse_counts():
    sim = _sim()
    observer = 0
    head = sim.heads()[0, observer]
    block_r, block_c = int(head[1] // 8), int(head[0] // 8)
    # A coarse block can hold many segments from several snakes.  The backing
    # count stays int32; only the raster channel clamps its rendered mass.
    assert sim._csnake.dtype == np.int32
    sim._csnake[0, observer, block_r, block_c] = 64
    sim._csnake[0, 1, block_r, block_c] = 300
    sim._csnake[0, 2, block_r, block_c] = 100
    obs = ObservationBuffer(1).build(sim, np.array([[0, observer]], dtype=np.int64))
    center = obs["global"][0, :, 18, 18]
    assert center[0] == 64
    assert center[1] == 255


def test_numeric_corpse_queue_preserves_eviction_readd_order_and_rng():
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.grid_sim import GridBatchSim

    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=3,
        game_width=300,
        game_height=200,
        initial_food=0,
        max_food=2,
        max_capacity=100,
        mechanics_version=2,
    )
    reference = GridBatchSim(cfg, seeds=[21], train_mode=True, jit=False)
    candidate = GridBatchSim(cfg, seeds=[21], train_mode=True, jit=True)
    # Exceed corpse cap, revisit previously evicted cells, and include duplicates.
    cells = [(x, 3) for x in range(2, 18)] + [(2, 3), (3, 3), (2, 3), (18, 3)]
    before = candidate._rngs[0]._rng.getstate()
    reference._drop_corpse_v2(0, cells)
    candidate._drop_corpse_v2(0, cells)
    assert reference.food_cells == candidate.food_cells
    assert reference.food_set == candidate.food_set
    assert reference.corpse_cells == candidate.corpse_cells
    assert reference._food.tobytes() == candidate._food.tobytes()
    assert reference._cfood.tobytes() == candidate._cfood.tobytes()
    assert candidate._rngs[0]._rng.getstate() == before
    candidate.check_grid_invariants()


def test_persistent_geometry_scratch_reused_and_stale_values_fully_overwritten():
    import numba

    from research.t33_batched_teacher.observation_kernels import _local_workspace
    from src.simd_env.ego_raster_nb import _region_workspace

    sim = _sim()
    rows = np.array([[0, 0], [1, 1], [0, 2]], dtype=np.int64)
    buffer = ObservationBuffer(len(rows))
    expected = {key: value.copy() for key, value in buffer.build(sim, rows).items()}
    parts = min(len(rows), numba.get_num_threads())
    local = _local_workspace(31, EgoRasterConfig(version="bv8").reach_steps, parts)
    region = _region_workspace(sim.grid_h, sim.grid_w, parts)
    pointers = [a.__array_interface__["data"][0] for a in (*local, *region)]
    for array in (*local, *region):
        array.fill(7)
    actual = buffer.build(sim, rows)
    for key, value in expected.items():
        assert actual[key].tobytes() == value.tobytes(), key
    assert pointers == [a.__array_interface__["data"][0] for a in (*local, *region)]
