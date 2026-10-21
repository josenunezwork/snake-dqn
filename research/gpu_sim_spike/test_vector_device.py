"""Focused device-vector runtime checks (CPU-safe; CUDA is shape-tested elsewhere)."""

from __future__ import annotations

import numpy as np
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.vector61_featurizer import Vector61Featurizer, Vector61Memory, Vector61Params
from src.simd_env.vector61_policy import Vector61Runtime


def _world() -> DeviceSim:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=2,
        game_width=200,
        game_height=200,
        initial_food=0,
        max_food=2,
        max_capacity=8,
    )
    sim = DeviceSim(cfg, device="cpu", cap=8)
    # Heads are first ring entries; the remaining two cells exercise tail
    # masking, nearest-enemy ordering, food order, and the flood columns.
    # Deliberately rotate both rings: index zero is not a head in this fixture.
    sim.head_ptr[0] = torch.tensor((4, 6))
    sim.bodies[0, 0, (4, 3, 2)] = torch.tensor(((5, 5), (4, 5), (3, 5)))
    sim.bodies[0, 1, (6, 5, 4)] = torch.tensor(((7, 5), (7, 6), (7, 7)))
    sim.seg_count[:] = 3
    sim.length[:] = 6
    sim.direction[0] = torch.tensor((1, 2))
    sim.food_cells[0, :2] = torch.tensor(((6, 5), (12, 5)))
    sim.food_present[0, :2] = True
    sim.food_count[0] = 2
    return sim


def _reference(sim: DeviceSim) -> GridBatchSim:
    ref = GridBatchSim(sim.cfg, seeds=[41], train_mode=True, allow_respawn=True)
    sim.copy_env_to_reference(ref)
    return ref


def _params() -> Vector61Params:
    return Vector61Params(
        game_width=200,
        game_height=200,
        segment_size=10,
        max_length=150,
        num_sectors=16,
        danger_max_distance=3,
        use_boundary_as_danger=True,
        min_boost_length=5,
        food_capacity=2,
    )


def test_device_vector_emits_all_61_columns_and_carry_lifecycle() -> None:
    sim = _world()
    runtime = DeviceVectorRuntime(_params())
    runtime.bind(sim)
    runtime.prepare(sim)
    states, mask = runtime.selection(torch.tensor(((0, 0), (0, 1))))
    assert states.shape == (2, 61)
    assert mask.shape == (2, 6)
    assert torch.isfinite(states).all()
    assert states[0, 1] == 1  # east one-hot
    assert states[0, 5] > 0  # ordered nearest food is east of the head
    assert states[0, 8:24].sum() > 0  # food-sector density is populated
    assert states[0, 24:40].sum() > 0  # enemy is two cells away, inside radius three
    assert states[0, 44:54].abs().sum() > 0  # enemy block is populated
    assert states[0, 58:61].sum() > 0  # exact capped device flood is populated
    runtime.observe_step(sim)
    runtime.prepare(sim)
    assert not runtime.sel_fresh.any()
    assert runtime.carried_rows.item() == 2


def test_device_vector_branch_clone_keeps_device_runtime_state() -> None:
    sim = _world()
    runtime = DeviceVectorRuntime(_params())
    runtime.bind(sim)
    runtime.prepare(sim)
    branched = runtime.repeat_interleave(3)
    assert branched.sel_state.shape == (3, 2, 61)
    assert branched.prev_dist.shape == (3, 2)
    assert branched.sel_state.device.type == "cpu"


def test_device_vector_columns_and_trend_carry_match_cpu_oracle_on_rotated_rings() -> None:
    """Compare every state column to the frozen CPU featurizer on one fixture.

    The device port is allowed float32/libm variation, but this fixture catches
    column swaps, missing feature blocks, ordered-food errors, tail masking,
    and a carry update attached to the wrong ring slot.
    """
    sim = _world()
    params = _params()
    ref = _reference(sim)
    cpu_memory = Vector61Memory(1, 2)
    cpu = Vector61Featurizer(params)
    rows = np.array(((0, 0), (0, 1)), dtype=np.int64)
    expected_first = cpu.featurize(ref, rows, cpu_memory, update_memory=True)
    runtime = DeviceVectorRuntime(params)
    runtime.bind(sim)
    actual_first = runtime.features(sim, update_memory=True).detach().numpy()
    np.testing.assert_allclose(actual_first, expected_first, rtol=0.0, atol=1e-6)

    # Move the second head while keeping its ring rotated.  Both runtimes must
    # retain the prior nearest-enemy baseline and put the resulting trend in
    # column 49, rather than resetting it or using the wrong actor's memory.
    sim.bodies[0, 1, 6] = torch.tensor((8, 5))
    ref = _reference(sim)
    expected_second = cpu.featurize(ref, rows, cpu_memory, update_memory=True)
    actual_second = runtime.features(sim, update_memory=True).detach().numpy()
    np.testing.assert_allclose(actual_second, expected_second, rtol=0.0, atol=1e-6)
    assert actual_second[0, 49] == expected_second[0, 49] == -1.0


def test_from_reference_converts_cpu_pixel_trend_memory_to_device_cells() -> None:
    """A stationary enemy after a CPU carry must report trend zero, not closing."""
    sim = _world()
    ref = _reference(sim)
    cpu_runtime = Vector61Runtime(params=_params())
    cpu_runtime.bind(ref, np.ones((1, 2), dtype=bool))
    cpu_runtime.prepare(ref)
    cpu_runtime.observe_step(ref)
    # The CPU baseline is 20 pixels between (5,5) and (7,5); device rings use
    # two cells.  This setup bridge must convert only that runtime scalar.
    assert cpu_runtime.memory.prev_dist[0, 0] == 20.0
    runtime = DeviceVectorRuntime.from_reference(cpu_runtime, sim)
    assert runtime.prev_dist[0, 0] == 2.0
    state = runtime.features(sim, update_memory=True).detach().numpy()
    assert state[0, 49] == 0.0
