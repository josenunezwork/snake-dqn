"""Independent CUDA checks for indexed vector61 static capped traversal."""

from __future__ import annotations

from collections import deque

import pytest
import torch

from research.gpu_sim_spike import flood_cuda


@pytest.mark.parametrize("dead", [False, True])
def test_static_vector61_cpu_bytes_and_carry(monkeypatch, dead: bool) -> None:
    """Full output and trend memory use the independent production CPU oracle."""
    import numpy as np

    from research.gpu_sim_spike.test_vector_device import _params, _reference, _world
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
    from src.simd_env.vector61_featurizer import Vector61Featurizer, Vector61Memory

    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "static")
    sim = _world()
    if dead:
        sim.alive[0, 1] = False
    runtime = DeviceVectorRuntime(_params())
    runtime.bind(sim)
    memory = Vector61Memory(1, 2)
    oracle = Vector61Featurizer(_params())
    rows = np.array(((0, 0), (0, 1)), dtype=np.int64)
    for moved in (False, True):
        if moved:
            sim.bodies[0, 1, 6] = torch.tensor((8, 5))
        expected = oracle.featurize(_reference(sim), rows, memory, update_memory=True)
        actual = runtime.features(sim, update_memory=True).numpy()
        assert actual.tobytes() == expected.tobytes(), np.argwhere(
            actual.view(np.uint32) != expected.view(np.uint32)
        ).tolist()


def test_static_vector61_shuffled_duplicate_rows_cpu_bytes(monkeypatch) -> None:
    """Read-only selection preserves row ordering, duplicates and every float bit."""
    import numpy as np

    from research.gpu_sim_spike.test_vector_device import _params, _reference, _world
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
    from src.simd_env.vector61_featurizer import Vector61Featurizer

    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "static")
    sim = _world()
    runtime = DeviceVectorRuntime(_params())
    runtime.bind(sim)
    rows = np.array(((0, 1), (0, 0), (0, 1)), dtype=np.int64)
    expected = Vector61Featurizer(_params()).featurize(_reference(sim), rows, update_memory=False)
    actual = runtime.features(sim, torch.as_tensor(rows), update_memory=False).numpy()
    assert actual.tobytes() == expected.tobytes()


def _count(start: tuple[int, int], occupied: torch.Tensor, cap: int) -> int:
    """Standalone cardinal oracle: no candidate helper or candidate BFS."""
    h, w = occupied.shape
    x, y = start
    if not (0 <= x < w and 0 <= y < h) or bool(occupied[y, x]):
        return 0
    todo, seen = deque([(x, y)]), {(x, y)}
    while todo and len(seen) < cap:
        x, y = todo.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and not bool(occupied[ny, nx]) and (nx, ny) not in seen:
                seen.add((nx, ny))
                todo.append((nx, ny))
                if len(seen) == cap:
                    break
    return len(seen)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native static traversal requires CUDA")
def test_indexed_static_capped_matches_independent_fifo_at_boundaries() -> None:
    """One occupancy grid serves duplicate, disconnected, blocked and OOB lanes."""
    h, w = 19, 23
    occupied = torch.ones((1, h, w), dtype=torch.bool, device="cuda")
    occupied[0, 2, 1:22] = False
    occupied[0, 3:18, 21] = False
    occupied[0, 17, 2:22] = False
    starts = torch.tensor(
        [[[[1, 2], [1, 2], [-1, 2]], [[21, 17], [0, 0], [22, 18]]]], device="cuda"
    )
    caps = torch.tensor([[32, 160]], dtype=torch.long, device="cuda")
    got = flood_cuda.static_capped(starts, occupied, caps).cpu()
    grid = occupied.cpu()[0]
    expected = torch.tensor(
        [
            [
                [_count((1, 2), grid, 32), _count((1, 2), grid, 32), 0],
                [_count((21, 17), grid, 160), 0, 0],
            ]
        ]
    )
    assert torch.equal(got, expected)
