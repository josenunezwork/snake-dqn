"""Independent Codex-V oracle fixtures for the A0 device event path.

The reference arm is loaded from the exact pre-A0 source with ``git show`` and
is deliberately kept separate from the working-tree ``DeviceSim``. This file
tests event state and counter-RNG identity only; it does not qualify CUDA speed
or A1 throughput.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from src.simd_env.batch_sim import BatchSimConfig

_RECEIPT = Path(__file__).parent / "fixtures" / "a0_event_oracle" / "ORACLE_RECEIPT.json"
_BASELINE_REV = "01334e1ed04b49ebf61ab16a6df0cf6c7e4c2884"
_BASELINE_SHA256 = "7fbabf67b439c6e8ebf2cbb252abed5dadcc1ce08194a8d6177605ab4bbe8d88"


def _receipt() -> dict[str, Any]:
    return json.loads(_RECEIPT.read_text())


def _frozen_module() -> ModuleType:
    """Load the untouched baseline module without sharing candidate helpers."""
    path = Path(__file__).parent / "device_sim_frozen.py"
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == _BASELINE_SHA256
    spec = importlib.util.spec_from_loader("_a0_frozen_device_sim", loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, f"{_BASELINE_REV}:device_sim.py", "exec"), module.__dict__)
    return module


def _cfg(
    *,
    envs: int = 1,
    snakes: int = 1,
    width: int = 3000,
    height: int = 100,
    max_food: int = 0,
    cap: int = 64,
) -> BatchSimConfig:
    return BatchSimConfig(
        num_envs=envs,
        num_snakes=snakes,
        game_width=width,
        game_height=height,
        initial_food=max_food,
        max_food=max_food,
        max_capacity=cap,
        mechanics_version=2,
        frame_rate=1,
    )


def _food(sim: Any) -> tuple[bytes, bytes, bytes, int]:
    count = int(sim.food_count[0].item())
    return (
        sim.food_cells[0].detach().cpu().numpy().tobytes(),
        sim.food_present[0].detach().cpu().numpy().tobytes(),
        sim.corpse_food[0].detach().cpu().numpy().tobytes(),
        count,
    )


def _core(sim: Any) -> dict[str, bytes | int]:
    fields = (
        "bodies",
        "head_ptr",
        "seg_count",
        "length",
        "alive",
        "direction",
        "boost_frames",
        "frames_since_food",
        "respawn_timer",
        "_reward_prev_length",
        "frame",
        "rng_counter",
        "rng_key",
        "_last_done",
        "_last_death_cause",
        "_last_kills",
        "_last_food_ate",
        "_last_ambient_food_ate",
        "_last_corpse_food_ate",
        "_last_transition_valid",
    )
    result: dict[str, bytes | int] = {}
    for name in fields:
        value = getattr(sim, name)
        result[name] = value.detach().cpu().numpy().tobytes()
    result["food"] = _food(sim)[0]  # type: ignore[assignment]
    result["food_present"] = _food(sim)[1]  # type: ignore[assignment]
    result["corpse_food"] = _food(sim)[2]  # type: ignore[assignment]
    result["food_count"] = _food(sim)[3]
    return result


def _seed_food(
    sim: Any,
    cells: list[tuple[int, int]],
    corpse: list[bool] | None = None,
    env: int = 0,
) -> None:
    flags = corpse or [False] * len(cells)
    values = torch.tensor(cells, dtype=torch.long).reshape(1, len(cells), 2)
    sim.food_cells[env, : len(cells)] = values[0]
    sim.food_present[env, : len(cells)] = True
    sim.corpse_food[env, : len(cells)] = torch.tensor(flags, dtype=torch.bool)
    sim.food_count[env] = len(cells)


def _seed_body(
    sim: Any,
    slot: int,
    cells: list[tuple[int, int]],
    direction: int,
    env: int = 0,
) -> None:
    values = torch.tensor(cells, dtype=torch.long)
    # DeviceSim stores the newest head at head_ptr and walks tails backwards
    # around the ring.  Populate the ring in that representation so the next
    # push retains every actual segment instead of reading zero padding.
    sim.bodies[env, slot, 0] = values[0]
    for offset, cell in enumerate(values[1:], start=1):
        sim.bodies[env, slot, (-offset) % sim.cap] = cell
    sim.head_ptr[env, slot] = 0
    sim.seg_count[env, slot] = len(cells)
    sim.length[env, slot] = len(cells)
    sim.alive[env, slot] = True
    sim.direction[env, slot] = direction


def _new_pair(cfg: BatchSimConfig) -> tuple[Any, Any]:
    frozen = _frozen_module().DeviceSim(cfg, device="cpu", cap=cfg.max_capacity)
    candidate = DeviceSim(cfg, device="cpu", cap=cfg.max_capacity)
    return frozen, candidate


def _assert_core_equal(frozen: Any, candidate: Any) -> None:
    left, right = _core(frozen), _core(candidate)
    assert left.keys() == right.keys()
    for key in left:
        assert left[key] == right[key], key


def test_oracle_receipt_binds_production_capacity_and_fixtures() -> None:
    receipt = _receipt()
    assert receipt["baseline_revision"] == _BASELINE_REV
    assert receipt["baseline_fixture"] == "research/gpu_sim_spike/device_sim_frozen.py"
    assert receipt["baseline_device_sim_sha256"] == _BASELINE_SHA256
    assert receipt["production_body_capacity"] == 10400
    assert receipt["rng_fixture"]["ticks"] == 1000
    assert set(receipt["fixtures"]) == {
        "full-food-empty-event",
        "food-refill",
        "single-death",
        "simultaneous-deaths",
        "capacity-eviction",
    }


def test_empty_event_is_rejected_without_calling_corpse_append() -> None:
    sim = DeviceSim(_cfg(max_food=0), device="cpu", cap=10400)
    _seed_body(sim, 0, [(100, 5)], direction=1)
    calls: list[tuple[tuple[int, ...], tuple[int, ...], bool]] = []
    original = sim._append_many

    def spy(cells: torch.Tensor, mask: torch.Tensor, corpse: bool) -> None:
        calls.append((tuple(cells.shape), tuple(mask.shape), corpse))
        original(cells, mask, corpse)

    sim._append_many = spy  # type: ignore[method-assign]
    sim.step(torch.tensor([[1]], dtype=torch.long))
    assert calls == []
    assert sim.cap == 10400
    assert int(sim.food_count[0]) == 0


def test_full_food_maintenance_1000_ticks_matches_frozen_state_and_rng() -> None:
    cfg = _cfg(max_food=2, cap=10400)
    frozen, candidate = _new_pair(cfg)
    for sim in (frozen, candidate):
        _seed_body(sim, 0, [(100, 5)], direction=1)
        _seed_food(sim, [(5, 2), (7, 3)])

    for _ in range(1000):
        # Keep the 1,000-tick RNG proof isolated from the pre-A0 S*cap corpse
        # loop. A separate death fixture exercises the full event path with a
        # small ring; this loop is specifically food maintenance and stream
        # preservation.
        frozen._maintain_food()
        candidate._maintain_food()
        _assert_core_equal(frozen, candidate)


def test_food_refill_matches_frozen_rng_and_order() -> None:
    cfg = _cfg(max_food=3, cap=10400)
    frozen, candidate = _new_pair(cfg)
    for sim in (frozen, candidate):
        _seed_body(sim, 0, [(100, 5)], direction=1)
        _seed_food(sim, [(5, 2)])
        sim._maintain_food()
    _assert_core_equal(frozen, candidate)
    assert int(candidate.food_count[0]) == 3


def test_food_maintenance_active_subset_rows_at_production_capacity() -> None:
    """Compare full, refill, no-space, and exhausted rows in one RNG tick."""
    cfg = _cfg(envs=4, max_food=3, width=3000, height=100, cap=10400)
    frozen, candidate = _new_pair(cfg)
    blocked = [(x, y) for y in range(10) for x in range(300)]
    for sim in (frozen, candidate):
        # Row 0: full ambient food, so it is outside the active subset.
        _seed_body(sim, 0, [(100, 5)], direction=1, env=0)
        _seed_food(sim, [(5, 2), (7, 3), (9, 4)], env=0)
        # Row 1: refill two ambient slots.
        _seed_body(sim, 0, [(100, 5)], direction=1, env=1)
        _seed_food(sim, [(5, 2)], env=1)
        # Row 2: no free lattice space, but already full ambient food.
        _seed_body(sim, 0, blocked, direction=1, env=2)
        _seed_food(sim, [(5, 2), (7, 3), (9, 4)], env=2)
        # Row 3: no free lattice space and every refill trial is exhausted.
        _seed_body(sim, 0, blocked, direction=1, env=3)
        sim._maintain_food()

    _assert_core_equal(frozen, candidate)
    assert frozen.food_count.tolist() == [3, 3, 3, 0]
    assert candidate.rng_counter.tolist() == [3, 3, 3, 3]


@pytest.mark.parametrize("snakes", [1, 2])
def test_single_and_simultaneous_death_compact_actual_segments(snakes: int) -> None:
    cfg = _cfg(snakes=snakes, max_food=0, cap=10400)
    frozen, candidate = _new_pair(cfg)
    for sim in (frozen, candidate):
        _seed_body(sim, 0, [(0, 5), (1, 5), (2, 5), (3, 5)], direction=3)
        if snakes == 2:
            _seed_body(sim, 1, [(0, 7), (1, 7), (2, 7), (3, 7)], direction=3)

    actions = torch.ones((1, snakes), dtype=torch.long)
    frozen.step(actions)
    candidate.step(actions)
    _assert_core_equal(frozen, candidate)
    # The newly inserted wall-crossing head is outside the lattice and is not
    # a food cell; only the three in-arena dead segments are materialized.
    assert int(candidate.food_count[0]) == 3 * snakes
    assert int(candidate.corpse_food[0].sum()) == 3 * snakes
    assert int(candidate.food_present[0, candidate.food_count[0] :].sum()) == 0


@pytest.mark.parametrize("snakes", [1, 2])
def test_production_capacity_event_batch_uses_only_actual_dead_segments(snakes: int) -> None:
    """Exercise the production ring allocation without a baseline S*cap tick."""
    cfg = _cfg(snakes=snakes, max_food=20, width=3000, cap=10400)
    frozen, candidate = _new_pair(cfg)
    per_snake = [[(10 + offset, 2 + slot * 3) for offset in range(3)] for slot in range(snakes)]
    actual = [cell for row in per_snake for cell in row]
    cells = torch.tensor([actual], dtype=torch.long)
    mask = torch.ones((1, len(actual)), dtype=torch.bool)
    for sim in (frozen, candidate):
        sim._append_many(cells, mask, True)
        assert sim.cap == 10400
        assert sim.bodies.shape[2] == 10400
    _assert_core_equal(frozen, candidate)
    assert int(candidate.food_count[0]) == 3 * snakes
    assert int(candidate.corpse_food[0].sum()) == 3 * snakes


def test_capacity_eviction_at_10400_uses_actual_bounded_append() -> None:
    cfg = _cfg(max_food=20, width=3000, cap=10400)
    frozen, candidate = _new_pair(cfg)
    old = [(i, 2) for i in range(10)]
    incoming = [(10, 2), (11, 2), (0, 2), (12, 2)]
    mask = torch.ones((1, len(incoming)), dtype=torch.bool)
    cells = torch.tensor([incoming], dtype=torch.long)
    for sim in (frozen, candidate):
        _seed_food(sim, old, [True] * len(old))
        # Stale coordinates in non-present padding are not food and must not
        # reject an incoming candidate that happens to reuse that coordinate.
        sim.food_cells[0, 15] = torch.tensor((12, 2))
        sim._append_many(cells, mask, True)
        assert sim.cap == 10400
        assert sim.bodies.shape[2] == 10400
    _assert_core_equal(frozen, candidate)
    got = [tuple(cell) for cell in candidate.food_cells[0, : candidate.food_count[0]].tolist()]
    assert got == [(i, 2) for i in range(4, 10)] + [(10, 2), (11, 2), (0, 2), (12, 2)]


def test_stage_instrumentation_has_complete_nonnegative_timer_and_launch_schema() -> None:
    cfg = _cfg(max_food=0, cap=10400)
    sim = DeviceSim(cfg, device="cpu", cap=10400)
    if not hasattr(sim, "set_instrumentation"):
        pytest.fail("A0 requires DeviceSim.set_instrumentation")
    if not hasattr(sim, "consume_timing"):
        pytest.fail("A0 requires DeviceSim.consume_timing")
    sim.set_instrumentation(True)
    _seed_body(sim, 0, [(100, 5)], direction=1)
    sim.step(torch.tensor([[1]], dtype=torch.long))
    receipt = sim.consume_timing()
    assert set(receipt) >= {"stage_timers", "launch_counts"}
    stages = receipt["stage_timers"]
    launches = receipt["launch_counts"]
    expected = {
        "movement",
        "deaths_corpse",
        "food",
        "vector61",
        "v8_floods",
        "opponent_forwards",
        "b8_forward",
        "observation_paint",
        "collection",
        "reset",
    }
    assert set(stages) == expected
    assert set(launches) == expected
    assert all(isinstance(value, (int, float)) and value >= 0 for value in stages.values())
    assert all(isinstance(value, int) and value >= 0 for value in launches.values())
