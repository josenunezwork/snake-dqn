"""Focused device-vector runtime checks (CPU-safe; CUDA is shape-tested elsewhere)."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch

from research.gpu_sim_spike import vector_device
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.vector61_featurizer import Vector61Featurizer, Vector61Memory, Vector61Params
from src.simd_env.vector61_policy import Vector61Runtime

_FROZEN_VECTOR = Path(__file__).with_name("fixtures") / "a1_memory" / "vector_device_pre_memory.py"
_FROZEN_VECTOR_SHA256 = "6d32148935b9cccd3d2d2e7bdf1093a2abf9a88349d3e46e48907bb738a4d69d"


def _frozen_runtime_type():
    """Load the pre-memory-repair implementation as an independent oracle."""
    assert hashlib.sha256(_FROZEN_VECTOR.read_bytes()).hexdigest() == _FROZEN_VECTOR_SHA256
    spec = importlib.util.spec_from_file_location("a1_vector_pre_memory", _FROZEN_VECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceVectorRuntime


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


def test_feature_row_chunks_match_one_block_and_flood_once(monkeypatch) -> None:
    """Chunking only bounds scratch tensors; it cannot change vector61 rows."""
    cfg = BatchSimConfig(
        num_envs=22,
        num_snakes=3,
        game_width=40,
        game_height=40,
        initial_food=0,
        max_food=1,
        max_capacity=8,
    )
    sim = DeviceSim(cfg, device="cpu", cap=8)
    sim.head_ptr[:] = 0
    sim.bodies[:, :, 0] = torch.tensor((1, 1))
    sim.seg_count[:] = 1
    sim.length[:] = 6
    params = Vector61Params.from_game_config(cfg)
    rows = torch.stack(
        (
            torch.arange(sim.E)[:, None].expand(sim.E, sim.S).reshape(-1),
            torch.arange(sim.S)[None, :].expand(sim.E, sim.S).reshape(-1),
        ),
        dim=1,
    )
    mask = torch.arange(len(rows)).remainder(2).bool()
    full = DeviceVectorRuntime(params)
    full.bind(sim)
    expected = full._features_rows(sim, rows, update_memory=True, memory_mask=mask)

    chunked = DeviceVectorRuntime(params)
    chunked.bind(sim)
    calls = 0
    original_flood = chunked._free_space_all

    def counted_flood(current: DeviceSim) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_flood(current)

    monkeypatch.setattr(vector_device, "FEATURE_ROW_CHUNK", 7)
    monkeypatch.setattr(chunked, "_free_space_all", counted_flood)
    actual = chunked.features(sim, rows, update_memory=True, memory_mask=mask)

    assert calls == 1
    assert torch.equal(actual, expected)
    assert torch.equal(chunked.prev_dist, full.prev_dist)
    assert torch.equal(chunked.prev_id, full.prev_id)


def test_feature_chunks_match_frozen_pre_memory_oracle_on_shuffled_rows(monkeypatch) -> None:
    """Oracle-check shuffled unique rows across three feature blocks."""
    cfg = BatchSimConfig(
        num_envs=3,
        num_snakes=3,
        game_width=100,
        game_height=80,
        initial_food=0,
        max_food=3,
        max_capacity=8,
    )
    sim = DeviceSim(cfg, device="cpu", cap=8)
    for env in range(sim.E):
        for slot in range(sim.S):
            x, y = 2 + env * 2, 2 + slot * 2
            sim.bodies[env, slot, :3] = torch.tensor(((x, y), (x - 1, y), (x - 2, y)))
    sim.seg_count[:] = 3
    sim.length[:] = torch.tensor((6, 7, 8))
    sim.direction[:] = torch.tensor((1, 2, 3))
    sim.food_cells[:, :3] = torch.tensor(((1, 1), (4, 3), (7, 5)))
    sim.food_present[:, :3] = True
    sim.food_count[:] = 3
    sim.alive[2, 2] = False
    # All nine rows occur once in non-grid order. The dead row makes the
    # zero-row path part of the same independent oracle comparison.
    rows = torch.tensor(((2, 1), (0, 2), (1, 0), (2, 2), (0, 0), (1, 2), (2, 0), (0, 1), (1, 1)))
    memory_mask = torch.tensor((True, False, True, True, True, True, False, False, True))
    params = Vector61Params.from_game_config(cfg)
    frozen = _frozen_runtime_type()(params)
    frozen.bind(sim)
    frozen.prev_dist[:] = 7.0
    frozen.prev_id[:] = 2
    expected = frozen.features(sim, rows, update_memory=True, memory_mask=memory_mask)

    candidate = DeviceVectorRuntime(params)
    candidate.bind(sim)
    candidate.prev_dist[:] = 7.0
    candidate.prev_id[:] = 2
    monkeypatch.setattr(vector_device, "FEATURE_ROW_CHUNK", 3)
    actual = candidate.features(sim, rows, update_memory=True, memory_mask=memory_mask)

    assert torch.equal(actual, expected)
    assert torch.equal(candidate.prev_dist, frozen.prev_dist)
    assert torch.equal(candidate.prev_id, frozen.prev_id)


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
