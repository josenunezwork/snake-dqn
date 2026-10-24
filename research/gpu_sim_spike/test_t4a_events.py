"""Hermetic contract checks for the T4A exceptional-path evidence API."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.gpu_sim_spike import t4a_events
from research.gpu_sim_spike.device_sim import DeviceSim
from src.simd_env.batch_sim import BatchSimConfig


def test_event_result_has_only_frozen_coverage_keys_and_evidence(monkeypatch) -> None:
    def fake(event, checkpoint, roster, worlds, device, lane):
        return event != "stale_cache", {"event": event, "worlds": worlds}

    monkeypatch.setattr(t4a_events, "_run_one", fake)
    result = t4a_events.run_events(Path("checkpoint.pth"), "frozen", "cpu", "smoke", float("inf"))
    assert set(result) == {*t4a_events.EVENTS, "evidence"}
    assert result["ring_wrap"] is True
    assert result["stale_cache"] is False
    assert len(result["evidence"]) == len(t4a_events.EVENTS)
    assert all(row["worlds"] == 4 and row["ticks"] == 2 for row in result["evidence"])


def test_event_exception_fails_closed_with_actionable_record(monkeypatch) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError("fixture missing")

    monkeypatch.setattr(t4a_events, "_run_one", explode)
    result = t4a_events.run_events(Path("checkpoint.pth"), "mixed", "cpu", "smoke", float("inf"))
    assert not any(result[name] for name in t4a_events.EVENTS)
    assert all(
        row["status"] == "INCOMPLETE" and "fixture missing" in row["reason"]
        for row in result["evidence"]
    )


def test_event_timeout_is_not_coverage(monkeypatch) -> None:
    monkeypatch.setattr(t4a_events.time, "monotonic", lambda: 10.0)
    result = t4a_events.run_events(Path("checkpoint.pth"), "scripted", "cpu", "smoke", 9.0)
    assert not any(result[name] for name in t4a_events.EVENTS)
    assert all(row["status"] == "timeout" and row["ticks"] == 0 for row in result["evidence"])


def test_internal_restore_mismatch_is_terminal_self_consistency(monkeypatch) -> None:
    witness = {"field": "sim.rng_counter", "byte_offset": 0, "expected": 0, "actual": 1}
    monkeypatch.setattr(t4a_events, "_run_one", lambda *args: (False, {"mismatch": witness}))
    result = t4a_events.run_events(Path("checkpoint"), "frozen", "cpu", "smoke", float("inf"))
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["status"] == "INVALID_SELF_CONSISTENCY"
    assert result["evidence"][0]["witness"]["first_witness"] == witness


def test_invalid_capture_is_not_a_mechanical_completion(monkeypatch) -> None:
    def invalid(*args):
        raise ValueError("nonfinite Q")

    monkeypatch.setattr(t4a_events, "_run_one", invalid)
    with pytest.raises(ValueError, match="nonfinite Q"):
        t4a_events.run_events(Path("checkpoint"), "frozen", "cpu", "smoke", float("inf"))


def test_actual_device_sim_ring_wrap_and_ordered_contested_food() -> None:
    """Exercise the real DeviceSim mechanics used by the fixture predicates."""
    cfg = BatchSimConfig(
        num_envs=1, num_snakes=3, game_width=200, game_height=100, initial_food=0, max_food=1
    )
    sim = DeviceSim(cfg, train_mode=True, allow_respawn=False)
    sim.head_ptr[:, 0] = sim.cap - 1
    sim.seg_count[:, 0] = sim.length[:, 0] = 2
    sim.alive[:, 0] = True
    sim.bodies[0, 0, sim.cap - 1] = torch.tensor([4, 4])
    sim.bodies[0, 0, sim.cap - 2] = torch.tensor([3, 4])
    sim.direction[:, 0] = 1
    sim.step(torch.ones((1, 3), dtype=torch.long))
    assert sim.head_ptr[0, 0].item() == 0

    # Slots 1 and 2 enter the same ambient food cell; the fixed slot order
    # makes slot 1 the sole consumer before normal compaction/refill.
    for slot, x, direction in ((1, 16, 1), (2, 18, 3)):
        sim.head_ptr[:, slot] = 0
        sim.seg_count[:, slot] = sim.length[:, slot] = 1
        sim.alive[:, slot] = True
        sim.bodies[:, slot, 0] = torch.tensor([x, 5])
        sim.direction[:, slot] = direction
    sim.food_present.zero_()
    sim.corpse_food.zero_()
    sim.food_count.zero_()
    sim.food_cells[:, 0] = torch.tensor([17, 5])
    sim.food_present[:, 0] = True
    sim.food_count[:] = 1
    sim.step(torch.ones((1, 3), dtype=torch.long))
    assert sim._last_ambient_food_ate[0, 1].item()
    assert not sim._last_ambient_food_ate[0, 2].item()
