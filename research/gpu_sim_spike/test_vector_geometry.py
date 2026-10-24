"""Source-pinned byte oracle for indexed vector61 geometry."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.vector_geometry import IndexedVectorRuntime
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.vector61_featurizer import Vector61Params

_PIN = "313758255ee5b9f1422a0202c14a80cc32554857"
_PATH = "research/gpu_sim_spike/vector_device.py"
_SHA256 = "62f92ba1c3b104ba25db626ea92502097244be25e72f3b017c0dc7ea375de942"


def _frozen_runtime_type(tmp_path: Path):
    """Load the dense reference from the qualified transport pin, not the tree."""
    source = subprocess.check_output(("git", "show", f"{_PIN}:{_PATH}"))
    assert hashlib.sha256(source).hexdigest() == _SHA256
    frozen = tmp_path / "vector_device_31375825.py"
    frozen.write_bytes(source)
    spec = importlib.util.spec_from_file_location("vector_device_frozen_31375825", frozen)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceVectorRuntime


def _world(envs: int = 3) -> DeviceSim:
    cfg = BatchSimConfig(
        num_envs=envs,
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
            # Each ring starts at a distinct nonzero location so direct ring
            # indexing, self masking and enemy masking all participate.
            head = (env + slot + 2) % 8
            ring = torch.tensor(((head - 0) % 8, (head - 1) % 8, (head - 2) % 8))
            x, y = 2 + (env % 3) * 2, 2 + slot * 2
            sim.head_ptr[env, slot] = head
            sim.bodies[env, slot, ring] = torch.tensor(((x, y), (x - 1, y), (x - 2, y)))
    sim.seg_count[:] = 3
    sim.length[:] = torch.tensor((6, 7, 8))
    sim.direction[:] = torch.tensor((1, 2, 3))
    sim.food_cells[:, :3] = torch.tensor(((1, 1), (4, 3), (7, 5)))
    sim.food_present[:, :3] = True
    sim.food_count[:] = 3
    sim.alive[-1, 2] = False
    return sim


def test_indexed_geometry_matches_cpu_repair_and_preserves_unaffected_fields(
    tmp_path: Path, monkeypatch
) -> None:
    """CPU bytes repair numerical fields; independent dense keeps discrete invariants."""
    monkeypatch.delenv("GPU_SIM_VECTOR_FLOOD", raising=False)
    sim = _world()
    params = Vector61Params.from_game_config(sim.cfg)
    rows = torch.tensor(((2, 1), (0, 2), (1, 0), (2, 2), (0, 0), (1, 2), (2, 0), (0, 1), (1, 1)))
    memory_mask = torch.tensor((True, False, True, True, True, True, False, False, True))
    frozen = _frozen_runtime_type(tmp_path)(params)
    frozen.bind(sim)
    frozen.prev_dist[:] = 0.7
    frozen.prev_id[:] = 2
    expected = frozen.features(sim, rows, update_memory=True, memory_mask=memory_mask)

    indexed = IndexedVectorRuntime(params)
    indexed.bind(sim)
    indexed.prev_dist[:] = 7.0
    indexed.prev_id[:] = 2
    canonical, memory = _canonical(sim, indexed, rows.numpy())
    actual = indexed.features(sim, rows, update_memory=True, memory_mask=memory_mask)
    assert actual.numpy().tobytes() == canonical.tobytes()
    # Declared repair changes numerical columns and distance carry. Preserve
    # discrete live columns against the independent pre-repair device baseline.
    live = sim.alive[rows[:, 0], rows[:, 1]]
    discrete = [0, 1, 2, 3, 47, 48, 49, 53, 57]
    assert torch.equal(actual[live][:, discrete], expected[live][:, discrete])
    for index, (env, slot) in enumerate(rows.tolist()):
        if live[index] and memory_mask[index]:
            assert indexed.prev_dist[env, slot] == memory.prev_dist[env, slot]
        else:
            assert indexed.prev_dist[env, slot] == 7.0
    assert torch.equal(indexed.prev_id, frozen.prev_id)
    assert indexed._geometry_body_view is None


def test_from_runtime_clones_carry_without_aliasing() -> None:
    """Packet wrappers retain carry bytes but dense and indexed arms stay independent."""
    sim = _world()
    params = Vector61Params.from_game_config(sim.cfg)
    source = IndexedVectorRuntime(params)
    source.bind(sim)
    source.prev_dist[:] = 3.0
    wrapped = IndexedVectorRuntime.from_runtime(source)
    assert wrapped.prev_dist.cpu().numpy().tobytes() == source.prev_dist.cpu().numpy().tobytes()
    assert wrapped.prev_dist is not source.prev_dist


def test_indexed_geometry_multi_chunk_permuted_and_duplicate_rows(
    tmp_path: Path, monkeypatch
) -> None:
    """A >64-row call retains the dense duplicate order and byte-level carry."""
    monkeypatch.delenv("GPU_SIM_VECTOR_FLOOD", raising=False)
    sim = _world(envs=24)
    params = Vector61Params.from_game_config(sim.cfg)
    grid = torch.stack(
        (
            torch.arange(sim.E)[:, None].expand(sim.E, sim.S).reshape(-1),
            torch.arange(sim.S)[None, :].expand(sim.E, sim.S).reshape(-1),
        ),
        dim=1,
    )
    # First pass is unique and crosses the 64-row boundary.  The reverse pass
    # contains explicit duplicates and must take the parent's single-kernel
    # compatibility path without a stale indexed view.
    rows = grid.flip(0)
    frozen = _frozen_runtime_type(tmp_path)(params)
    frozen.bind(sim)
    indexed = IndexedVectorRuntime(params)
    indexed.bind(sim)
    expected, memory = _canonical(sim, indexed, rows.numpy())
    actual = indexed.features(sim, rows, update_memory=True)
    assert actual.numpy().tobytes() == expected.tobytes()
    assert indexed.prev_dist.numpy().tobytes() == memory.prev_dist.tobytes()

    duplicate = torch.cat((rows[:4], rows[:4], rows[4:]))
    expected_dup, _ = _canonical(sim, indexed, duplicate.numpy(), update_memory=False)
    actual_dup = indexed.features(sim, duplicate, update_memory=False)
    assert actual_dup.cpu().numpy().tobytes() == expected_dup.tobytes()


def test_indexed_geometry_drops_snapshot_after_feature_call(tmp_path: Path, monkeypatch) -> None:
    """A second call reads post-mutation ring cells, never retained geometry."""
    monkeypatch.delenv("GPU_SIM_VECTOR_FLOOD", raising=False)
    sim = _world()
    params = Vector61Params.from_game_config(sim.cfg)
    frozen = _frozen_runtime_type(tmp_path)(params)
    frozen.bind(sim)
    indexed = IndexedVectorRuntime(params)
    indexed.bind(sim)
    indexed.features(sim, update_memory=False)
    sim.bodies[0, 0, sim.head_ptr[0, 0]] = torch.tensor((8, 2))
    expected, _ = _canonical(sim, indexed, update_memory=False)
    actual = indexed.features(sim, update_memory=False)
    assert actual.cpu().numpy().tobytes() == expected.tobytes()
    assert indexed._geometry_body_view is None


def _canonical(sim, runtime, rows=None, update_memory=True):
    from types import SimpleNamespace

    from src.simd_env.vector61_featurizer import Vector61Featurizer, Vector61Memory

    snapshot = SimpleNamespace(cfg=sim.cfg, E=sim.E, S=sim.S, s=sim.cfg.segment_size, cap=sim.cap)
    for name in ("bodies", "head_ptr", "seg_count", "length", "alive", "direction"):
        setattr(snapshot, name, getattr(sim, name).cpu().numpy())
    snapshot.food_cells = [
        [
            tuple(map(int, cell))
            for cell in sim.food_cells[e, : int(sim.food_count[e])].cpu().numpy()
        ]
        for e in range(sim.E)
    ]
    memory = Vector61Memory(sim.E, sim.S)
    memory.prev_dist[:] = runtime.prev_dist.cpu().numpy()
    memory.prev_id[:] = runtime.prev_id.cpu().numpy()
    if rows is None:
        rows = np.stack((np.repeat(np.arange(sim.E), sim.S), np.tile(np.arange(sim.S), sim.E)), 1)
    result = Vector61Featurizer(runtime.params).featurize(snapshot, rows, memory, update_memory)
    return result, memory


@pytest.mark.parametrize("backend", ["dense", "static"])
@pytest.mark.parametrize("indexed", [False, True])
def test_cuda_indexed_geometry_bytes_match_canonical_cpu(monkeypatch, backend, indexed) -> None:
    """CUDA admission: independent canonical CPU bytes and masked trend carry."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA-only indexed geometry admission")
    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", backend)
    monkeypatch.setenv("GPU_SIM_VECTOR_GEOMETRY", "native")
    cpu = _world()
    params = Vector61Params.from_game_config(cpu.cfg)
    reference = IndexedVectorRuntime(params)
    reference.bind(cpu)
    expected, memory = _canonical(cpu, reference)
    cuda = DeviceSim(cpu.cfg, device="cuda", cap=8)
    for name in (
        "bodies",
        "head_ptr",
        "seg_count",
        "length",
        "alive",
        "direction",
        "food_cells",
        "food_present",
        "food_count",
    ):
        setattr(cuda, name, getattr(cpu, name).to("cuda"))
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime

    candidate = (IndexedVectorRuntime if indexed else DeviceVectorRuntime)(params)
    candidate.bind(cuda)
    actual = candidate.features(cuda, update_memory=True).cpu().numpy()
    assert actual.tobytes() == expected.tobytes()
    assert candidate.prev_dist.cpu().numpy().tobytes() == memory.prev_dist.tobytes()
    assert candidate.prev_id.cpu().numpy().tobytes() == memory.prev_id.tobytes()
    assert actual[-1].tobytes() == np.zeros_like(actual[-1]).tobytes()
    reference.prev_dist[:] = torch.as_tensor(memory.prev_dist)
    reference.prev_id[:] = torch.as_tensor(memory.prev_id)
    rows = torch.tensor(((0, 0), (0, 1), (0, 0)), device="cuda")
    expected_dup, _ = _canonical(cpu, reference, rows.cpu().numpy(), update_memory=False)
    actual_dup = candidate.features(cuda, rows, update_memory=False).cpu().numpy()
    assert actual_dup.tobytes() == expected_dup.tobytes()
    with pytest.raises(ValueError, match="unique"):
        candidate.features(cuda, rows, update_memory=True)

    # The native helper also respects an explicitly selected feature58 width.
    from dataclasses import replace

    candidate.params = replace(candidate.params, use_free_space=False)
    actual58 = candidate.features(cuda, update_memory=False).cpu().numpy()
    expected58, _ = _canonical(cpu, candidate, update_memory=False)
    assert actual58.shape[1] == 58
    assert actual58.tobytes() == expected58.tobytes()
