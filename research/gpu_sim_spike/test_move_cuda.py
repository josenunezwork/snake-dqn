"""MOVE optimization identity against frozen source, plus independent CPU cases."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.move_cuda import collisions
from src.simd_env.batch_sim import BatchSimConfig

FROZEN_SHA = "816ebd5fbd4016bb01eda5fe31ee7739f8cb7bf021b2885684f769ea63dc58ae"


def frozen_class() -> type:
    path = Path(__file__).parent / "fixtures/move/device_sim_pre_move.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_SHA
    spec = importlib.util.spec_from_file_location("_pre_move", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceSim


def assert_state(left: Any, right: Any) -> None:
    fields = {name for name, value in left.__dict__.items() if isinstance(value, torch.Tensor)}
    assert fields == {
        name for name, value in right.__dict__.items() if isinstance(value, torch.Tensor)
    }
    for name in sorted(fields):
        a, b = getattr(left, name), getattr(right, name)
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert a.cpu().numpy().tobytes() == b.cpu().numpy().tobytes(), name
    assert left._prepared == right._prepared


def seeded(
    *,
    version: int = 2,
    cap: int = 64,
    snakes: int = 6,
    width: int = 160,
    height: int = 120,
    device: str = "cpu",
) -> tuple[Any, Any]:
    cfg = BatchSimConfig(
        num_envs=2,
        num_snakes=snakes,
        game_width=width,
        game_height=height,
        max_capacity=cap,
        initial_food=0,
        max_food=0,
        mechanics_version=version,
    )
    left = frozen_class()(cfg, device=device)
    right = DeviceSim(cfg, device=device, move_backend="indexed", exact_mechanics=False)
    rng = np.random.default_rng(241)
    for name, value in left.__dict__.items():
        if isinstance(value, torch.Tensor):
            getattr(right, name).copy_(value)
    for sim in (left, right):
        # Deterministic overlapping owner cells, wrapped rings and stale padding.
        rng = np.random.default_rng(241)
        sim.bodies.copy_(
            torch.tensor(
                rng.integers(-1, min(width, height) // 10 + 1, size=tuple(sim.bodies.shape)),
                device=device,
            )
        )
        sim.head_ptr.copy_(torch.tensor(rng.integers(0, cap, size=(2, snakes)), device=device))
        n = rng.integers(1, min(cap, 27) + 1, size=(2, snakes))
        sim.seg_count.copy_(torch.tensor(n, device=device))
        sim.length.copy_(torch.tensor(n, device=device))
        sim.direction.copy_(torch.tensor(rng.integers(0, 4, size=(2, snakes)), device=device))
        sim.rng_key.copy_(torch.tensor([9223372036854775807, 127], device=device))
        sim.rng_counter.copy_(torch.tensor([9223372036854775799, 33], device=device))
    return left, right


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="native CUDA proof requires A100"
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["none", "one", "mixed", "duplicates", "crowded", "long"])
def test_respawn_complete_state_and_rng(device: str, case: str) -> None:
    kwargs: dict[str, Any] = {"device": device}
    if case == "duplicates":
        kwargs.update(width=10, height=10)
    if case == "crowded":
        kwargs.update(width=80, height=80, cap=64)
    if case == "long":
        kwargs.update(cap=1500)
    left, right = seeded(**kwargs)
    for sim in (left, right):
        if case == "one":
            sim.alive[0, 1] = False
            sim.respawn_timer[0, 1] = 1
        elif case != "none":
            sim.alive[:, 1:] = False
            sim.respawn_timer[:, 2] = 2
        if case == "crowded":
            # Every in-bounds cell occupied by one live snake (duplicates too).
            cells = [(x, y) for y in range(sim.grid_h) for x in range(sim.grid_w)]
            sim.bodies[0, 0, : min(len(cells), sim.cap)] = torch.tensor(
                cells[: sim.cap], device=device
            )
            sim.head_ptr[0, 0] = 0
            sim.seg_count[0, 0] = min(len(cells), sim.cap)
        if case == "long":
            sim.seg_count[:, 0] = 1500
    left._respawn()
    right._respawn()
    assert_state(left, right)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("snakes", [1, 3, 6])
def test_indexed_collision_flags_and_order(device: str, version: int, snakes: int) -> None:
    left, right = seeded(device=device, version=version, snakes=snakes)
    rng = np.random.default_rng(731)
    for _ in range(6):
        head0 = torch.tensor(rng.integers(-1, 13, size=(2, snakes, 2)), device=device)
        path = torch.tensor(rng.integers(-1, 13, size=(2, snakes, 2, 2)), device=device)
        valid = torch.tensor(rng.integers(0, 2, size=(2, snakes, 2)).astype(bool), device=device)
        arena = torch.tensor(rng.integers(0, 2, size=(2, snakes)).astype(bool), device=device)
        _, own, hp, bp = (
            left._collision_flags_reference(head0, path, valid)
            if hasattr(left, "_collision_flags_reference")
            else right._collision_flags_reference(head0, path, valid)
        )
        # Dense helpers are preserved verbatim; whole-step tests below use the
        # frozen module with no shared candidate collision helper at all.
        expected = right._resolve_collisions_reference(arena, own, hp, bp)
        actual = collisions(right, head0, path, valid, arena)
        for a, b in zip(expected, actual):
            assert a.cpu().numpy().tobytes() == b.cpu().numpy().tobytes()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("version", [1, 2])
def test_full_step_frozen_byte_identity(device: str, version: int) -> None:
    left, right = seeded(device=device, version=version)
    rng = np.random.default_rng(161)
    for _ in range(8):
        actions = torch.tensor(rng.integers(0, 6, size=(2, 6)), device=device)
        left.step(actions)
        right.step(actions)
        assert_state(left, right)


@pytest.mark.parametrize(
    "name",
    [
        "test_head_path_swap_is_a_head_collision",
        "test_three_way_equal_head_collision_keeps_last_cpu_pair_survivor",
        "test_boost_burn_drops_the_v2_tail_pellet",
        "test_food_resolution_rechecks_second_boost_cell_after_contest",
    ],
)
def test_independent_grid_cpu_cases(monkeypatch: Any, name: str) -> None:
    from research.gpu_sim_spike import test_device_sim

    class IndexedSim(DeviceSim):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs, move_backend="indexed")

    monkeypatch.setattr(test_device_sim, "DeviceSim", IndexedSim)
    getattr(test_device_sim, name)()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lengths", [(20, 23, 22), (40, 46, 45), (800, 1500, 1000)])
def test_size_ratio_boundaries_and_long_collision_reductions(
    device: str, lengths: tuple[int, ...]
) -> None:
    _, sim = seeded(device=device, snakes=3, cap=1500)
    sim.seg_count[:] = torch.tensor(lengths, device=device)
    sim.length[:] = torch.tensor(lengths, device=device)
    sim.alive[:] = True
    sim.bodies.fill_(7)
    # All pairs meet at one cell; stable resolver kills/credits order matters.
    path = torch.full((2, 3, 2, 2), 7, dtype=torch.long, device=device)
    h0 = torch.full((2, 3, 2), 6, dtype=torch.long, device=device)
    active = torch.ones((2, 3, 2), dtype=torch.bool, device=device)
    arena = torch.ones((2, 3), dtype=torch.bool, device=device)
    _, own, hp, bp = sim._collision_flags_reference(h0, path, active)
    expected = sim._resolve_collisions_reference(arena, own, hp, bp)
    actual = collisions(sim, h0, path, active, arena)
    for a, b in zip(expected, actual):
        assert a.cpu().numpy().tobytes() == b.cpu().numpy().tobytes()
    # Isolate head-pair size decisions from self/body collisions.
    sim.bodies.fill_(9)
    _, own, hp, bp = sim._collision_flags_reference(h0, path, active)
    expected = sim._resolve_collisions_reference(arena, own, hp, bp)
    actual = collisions(sim, h0, path, active, arena)
    for a, b in zip(expected, actual):
        assert a.cpu().numpy().tobytes() == b.cpu().numpy().tobytes()
