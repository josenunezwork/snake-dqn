"""T33 observation slab contracts against the canonical Bv8 featurizer."""

from __future__ import annotations

import os

import numba
import numpy as np
import pytest
import torch

from research.t33_batched_teacher.observations import ObservationBuffer
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.grid_sim import GridBatchSim


@pytest.fixture
def tiny_sim():
    sim = GridBatchSim(
        BatchSimConfig(num_envs=2, num_snakes=3, game_width=80, game_height=60, max_capacity=128),
        seeds=[101, 102],
        train_mode=True,
        allow_respawn=False,
    )
    # Advance once so every maintained grid and advisory mask is decision-ready.
    sim.step(np.ones((sim.E, sim.S), dtype=np.int64))
    return sim


def test_observation_slab_matches_bv8_reference_and_torch_views(tiny_sim):
    rows = np.array([[1, 2], [0, 0], [1, 0]], dtype=np.int64)
    buf = ObservationBuffer(5)
    out = buf.build(tiny_sim, rows)
    ref = build_ego_raster(tiny_sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
    for key in ("local", "global", "scalars"):
        assert out[key].shape[0] == 5
        assert out[key][: len(rows)].tobytes() == ref[key].tobytes(), key
        assert not out[key][len(rows) :].any(), key
    assert buf.storage.dtype == torch.uint8
    assert buf._scalar_offset % 4 == 0
    views = buf.tensor_views()
    assert torch.equal(views["local"], torch.from_numpy(out["local"]))
    assert torch.equal(views["global"], torch.from_numpy(out["global"]))
    assert torch.equal(views["scalars"], torch.from_numpy(out["scalars"]))


def test_observation_slab_clears_stale_and_dead_rows(tiny_sim):
    rows = np.array([[0, 0], [0, 1]], dtype=np.int64)
    buf = ObservationBuffer(3)
    first = buf.build(tiny_sim, rows)
    assert first["local"][:2].any()
    tiny_sim.alive[0, 0] = False
    second = buf.build(tiny_sim, rows[:1])
    ref = build_ego_raster(tiny_sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows[:1])
    for name, value in second.items():
        assert not value.any()
        assert value[:1].tobytes() == ref[name].tobytes()


def test_repeated_shuffled_builds_have_no_cross_row_state(tiny_sim):
    first_rows = np.array([[0, 2], [1, 1], [0, 0]], dtype=np.int64)
    second_rows = first_rows[::-1].copy()
    buf = ObservationBuffer(3)
    first = {key: value.copy() for key, value in buf.build(tiny_sim, first_rows).items()}
    second = buf.build(tiny_sim, second_rows)
    assert second["local"][0].tobytes() == first["local"][2].tobytes()
    assert second["global"][0].tobytes() == first["global"][2].tobytes()
    assert second["scalars"][0].tobytes() == first["scalars"][2].tobytes()


def test_capacity_and_external_slab_validation():
    buf = ObservationBuffer(1)
    with pytest.raises(ValueError, match="capacity"):
        buf.build(object(), np.array([[0, 0], [0, 1]], dtype=np.int64))
    with pytest.raises(ValueError, match="byte length"):
        buf.tensor_views(torch.empty(1, dtype=torch.uint8))


@pytest.mark.parametrize("heading", range(4))
def test_direct_bv8_bytes_cover_rotations_boosting_and_border(tiny_sim, heading):
    """The direct seventh channel shares the reference's integer geometry exactly."""
    tiny_sim.direction[0, 0] = heading
    tiny_sim.direction[0, 1] = (heading + 1) % 4
    tiny_sim._boosted_this_step[0, 1] = True
    # This small arena makes every 31x31 local crop include outside border cells.
    rows = np.array([[0, 0]], dtype=np.int64)
    got = ObservationBuffer(1).build(tiny_sim, rows)
    ref = build_ego_raster(tiny_sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
    for name in ("local", "global", "scalars"):
        assert got[name].tobytes() == ref[name].tobytes(), (heading, name)


@pytest.mark.parametrize("rows", [np.array([[-1, 0]]), np.array([[2, 0]]), np.array([[0, 3]])])
def test_selected_rows_are_bounds_checked_before_kernel_access(tiny_sim, rows):
    with pytest.raises(ValueError, match="outside"):
        ObservationBuffer(1).build(tiny_sim, rows)


@pytest.mark.skipif(
    os.environ.get("T33_OBS_THREAD_PARITY") != "1",
    reason="requires explicit two-thread authorization",
)
def test_local_and_global_prange_are_byte_equal_at_one_and_two_threads(tiny_sim):
    """Thread count must not affect slab bytes; opt-in to respect Mac capacity."""
    rows = np.array([[0, 0], [1, 2], [0, 1], [1, 0]], dtype=np.int64)
    original = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = {
            name: value.copy() for name, value in ObservationBuffer(4).build(tiny_sim, rows).items()
        }
        numba.set_num_threads(2)
        parallel = ObservationBuffer(4).build(tiny_sim, rows)
    finally:
        numba.set_num_threads(original)
    for name in serial:
        assert np.array_equal(serial[name], parallel[name]), name
