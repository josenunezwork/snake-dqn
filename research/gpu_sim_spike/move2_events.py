"""Executable MOVE2 event probes; CPU portability is never native acceptance."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

from research.gpu_sim_spike import move2_parity as parity
from research.gpu_sim_spike.device_sim import DeviceSim
from src.simd_env.batch_sim import BatchSimConfig

CASES = (
    "no_ready_rng_overflow",
    "ready_respawn",
    "crowded_respawn",
    "duplicate_respawn_candidates",
    "long_wrapped_rings",
    "duplicate_occupancy_dead_padding",
    "simultaneous_deaths_ordered_corpse_food",
    "boost_tail_release",
    "food_corpse_order",
    "lifecycle_ready_respawn",
)


def _frozen_class() -> type[Any]:
    path = parity.FROZEN_DEVICE_SIM
    if hashlib.sha256(path.read_bytes()).hexdigest() != parity.FROZEN_DEVICE_SIM_SHA256:
        raise ValueError("frozen MOVE oracle hash differs")
    spec = importlib.util.spec_from_file_location("_move2_event_oracle", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load frozen MOVE oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DeviceSim


def _make(device: str, case: str) -> tuple[Any, Any]:
    width, height = (80, 80) if case == "crowded_respawn" else (1600, 1200)
    if case == "duplicate_respawn_candidates":
        width = height = 10
    cfg = BatchSimConfig(
        num_envs=2,
        num_snakes=6,
        game_width=width,
        game_height=height,
        max_capacity=10400,
        initial_food=0,
        max_food=20,
        mechanics_version=2,
    )
    left, right = _frozen_class()(cfg, device=device), DeviceSim(
        cfg, device=device, move_backend="indexed"
    )
    for sim in (left, right):
        sim.bodies.fill_(-7)  # Stale padding must survive the optimization.
        sim.head_ptr.fill_(sim.cap - 1)
        sim.seg_count.fill_(12)
        sim.length.fill_(12)
        sim._reward_prev_length.fill_(12)
        sim.direction.fill_(1)
        sim.train_mode = False
        sim.allow_respawn = False
        sim._prepared = True
        for slot in range(6):
            rings = (sim.cap - 1 - torch.arange(12, device=device)) % sim.cap
            cells = torch.stack(
                (
                    30 - torch.arange(12, device=device),
                    torch.full((12,), 10 + slot * 17, device=device),
                ),
                dim=1,
            )
            sim.bodies[:, slot, rings] = cells
        sim.rng_key[:] = torch.tensor([9223372036854775807, 127], device=device)
        sim.rng_counter[:] = torch.tensor([9223372036854775799, 33], device=device)
        if case in ("ready_respawn", "lifecycle_ready_respawn"):
            sim.alive[:, 1] = False
            sim.respawn_timer[:, 1] = 1
        elif case == "crowded_respawn":
            sim.alive[:, 1:] = False
            cells = torch.tensor([(x, y) for y in range(8) for x in range(8)], device=device)
            rings = (sim.cap - 1 - torch.arange(64, device=device)) % sim.cap
            sim.bodies[:, 0, rings] = cells
            sim.seg_count[:, 0] = sim.length[:, 0] = 64
        elif case == "duplicate_respawn_candidates":
            sim.alive[:, 1:] = False
            sim.bodies[:, 0, :] = -7
            sim.seg_count[:, 0] = 0
        elif case == "long_wrapped_rings":
            for slot, length in enumerate((800, 900, 1000, 1100, 1300, 1500)):
                k = torch.arange(length, device=device)
                x = torch.where((k // 141) % 2 == 0, 5 + k % 141, 145 - k % 141)
                cells = torch.stack((x, 3 + slot * 18 + k // 141), dim=1)
                sim.bodies[:, slot, (sim.cap - 1 - k) % sim.cap] = cells
                sim.seg_count[:, slot] = sim.length[:, slot] = length
                sim._reward_prev_length[:, slot] = length
            sim.direction.fill_(3)
        elif case == "duplicate_occupancy_dead_padding":
            sim.alive[:, 4:] = False
            # Repeated live owner cells and a dead body's stale cells share space.
            sim.bodies[:, 1] = sim.bodies[:, 0]
            sim.bodies[:, 4] = sim.bodies[:, 0]
        elif case == "simultaneous_deaths_ordered_corpse_food":
            for slot, head, direction in ((0, (30, 30), 1), (1, (32, 30), 3), (2, (31, 29), 2)):
                sim.bodies[:, slot, sim.cap - 1] = torch.tensor(head, device=device)
                sim.direction[:, slot] = direction
            sim.length[:, :3] = 12
            sim.food_cells[:, :3] = torch.tensor([[80, 80], [81, 80], [82, 80]], device=device)
            sim.food_present[:, :3] = True
            sim.corpse_food[:, 1] = True
            sim.food_count.fill_(3)
        elif case == "boost_tail_release":
            sim.boost_frames.fill_(int(cfg.boost_length_cost_frames) - 1)
        elif case == "food_corpse_order":
            sim.food_cells[:, :4] = torch.tensor(
                [[31, 10], [31, 27], [90, 90], [91, 90]], device=device
            )
            sim.food_present[:, :4] = True
            sim.corpse_food[:, 1] = True
            sim.corpse_food[:, 3] = True
            sim.food_count.fill_(4)
        if case == "lifecycle_ready_respawn":
            sim._prepared = False
            sim.allow_respawn = True
    return left, right


def _exercise(sim: Any, case: str) -> None:
    if case in CASES[:4]:
        sim._respawn()
        return
    action = 4 if case == "boost_tail_release" else 1
    sim.step(torch.full((sim.E, sim.S), action, dtype=torch.long, device=sim.device))


def _witness(sim: Any, case: str, before: dict[str, Any]) -> dict[str, Any]:
    """Require each fixture to actually enter the event advertised in its name."""
    if case == "no_ready_rng_overflow":
        expected = torch.tensor([-9223372036854775785, 65], device=sim.device)
        assert torch.equal(sim.rng_counter, expected)
        assert bool(sim.alive.all())
        return {"counter_overflow": True, "ready_slots": 0}
    if case == "ready_respawn":
        assert bool(sim.alive[:, 1].all()) and bool((sim.length[:, 1] == 1).all())
        return {"respawned_slots": 2}
    if case == "crowded_respawn":
        assert not bool(sim.alive[:, 1:].any())
        return {"occupied_cells": 64, "unplaced_ready_slots": 10}
    if case == "duplicate_respawn_candidates":
        assert not bool(sim.alive[:, 1:].any())
        return {"candidate_cells_per_trial": 1, "earlier_ineligible_duplicate": True}
    if case == "long_wrapped_rings":
        assert bool((sim.head_ptr == 0).all())
        assert int(sim.seg_count.min()) >= 800
        return {"ring_wrapped": True, "capacity": sim.cap, "max_live_segments": 1500}
    if case == "duplicate_occupancy_dead_padding":
        assert not bool(sim._last_transition_valid[:, 4:].any())
        return {"duplicate_live_owners": [0, 1], "dead_rows": [4, 5], "padding_sentinel": -7}
    if case == "simultaneous_deaths_ordered_corpse_food":
        assert int(sim._last_done.sum()) >= 4
        assert bool(sim.corpse_food.any())
        return {"simultaneous_deaths": int(sim._last_done.sum()), "ordered_corpse_food": True}
    if case == "boost_tail_release":
        assert bool(sim._boosted_this_step.all())
        assert bool((sim.length == 11).all())
        assert bool((sim.boost_frames == 0).all())
        assert bool(sim.corpse_food.any())
        return {"boost_burn": True, "tail_pellets": int(sim.corpse_food.sum())}
    if case == "food_corpse_order":
        assert bool(sim._last_ambient_food_ate[:, 0].all())
        assert bool(sim._last_corpse_food_ate[:, 1].all())
        return {"ambient_ate": True, "corpse_ate": True, "ordered_survivors": True}
    assert case == "lifecycle_ready_respawn"
    assert before["non_tensors"]["_prepared"] is False and sim._prepared is False
    assert bool(sim.alive[:, 1].all())
    return {"prepare_step_lifecycle": True, "respawned_slots": 2}


def run(device: str) -> dict[str, Any]:
    """Execute every event probe; never count CPU or skipped CUDA as native PASS."""
    from research.gpu_sim_spike import move_cuda

    cuda = device == "cuda"
    if cuda and (not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0)):
        return {"status": "INVALID", "reason": "NATIVE_A100_REQUIRED", "cases": []}
    calls = {"respawn": 0, "collisions": 0}
    original = move_cuda._EXTENSION
    extension = None
    if cuda:
        try:
            extension = move_cuda._extension()
        except Exception as exc:
            return {
                "status": "INVALID",
                "reason": f"NATIVE_COMPILE:{type(exc).__name__}",
                "cases": [],
            }

        class ObservedExtension:
            def respawn(self, tensors: list[torch.Tensor], *args: Any) -> Any:
                assert all(value.is_cuda for value in tensors)
                calls["respawn"] += 1
                return extension.respawn(tensors, *args)

            def collisions(self, tensors: list[torch.Tensor], *args: Any) -> Any:
                assert all(value.is_cuda for value in tensors)
                calls["collisions"] += 1
                return extension.collisions(tensors, *args)

        move_cuda._EXTENSION = ObservedExtension()
    rows = []
    try:
        for case in CASES:
            left, right = _make(device, case)
            before = parity.state_descriptor(left)
            initial = parity.compare(left, right)
            if not initial["match"]:
                raise ValueError(f"{case}: unequal initial state: {initial}")
            calls_before = calls.copy()
            _exercise(left, case)
            _exercise(right, case)
            if cuda:
                torch.cuda.synchronize()
            proof = parity.compare(left, right)
            if not proof["match"]:
                raise ValueError(f"{case}: frozen identity failure: {proof}")
            observed = {name: calls[name] - calls_before[name] for name in calls}
            if cuda:
                required = "respawn" if case in CASES[:4] else "collisions"
                if observed[required] != 1:
                    raise ValueError(f"{case}: native {required} did not execute exactly once")
                if case == "lifecycle_ready_respawn" and observed["respawn"] != 1:
                    raise ValueError("lifecycle respawn kernel did not execute")
            rows.append(
                {
                    "case": case,
                    "configuration": before["non_tensors"]["cfg"],
                    "initial_state_sha256": parity.digest_bytes(
                        json.dumps(before, sort_keys=True).encode()
                    ),
                    "supplied_random_inputs": {
                        "rng_key": [9223372036854775807, 127],
                        "rng_counter": [9223372036854775799, 33],
                    },
                    "action": (
                        None if case in CASES[:4] else 4 if case == "boost_tail_release" else 1
                    ),
                    "native_calls": observed,
                    "identity": proof,
                    "witness": _witness(right, case, before),
                }
            )
    except Exception as exc:
        return {"status": "INVALID", "reason": str(exc), "cases": rows, "native_calls": calls}
    finally:
        if cuda:
            move_cuda._EXTENSION = extension if original is None else original
    if cuda and (not calls["respawn"] or not calls["collisions"]):
        return {
            "status": "INVALID",
            "reason": "NATIVE_KERNEL_NOT_EXECUTED",
            "native_calls": calls,
            "cases": rows,
        }
    extension_path = Path(extension.__file__) if extension is not None else None
    return {
        "schema": "gpu-sim-spike-move2-event-probes/v1",
        "status": "PASS" if cuda else "NOT_RUN",
        "portability_identity": "PASS",
        "native_calls": calls,
        "fallback": False if cuda else None,
        "extension_sha256": (
            hashlib.sha256(extension_path.read_bytes()).hexdigest() if extension_path else None
        ),
        "cases": rows,
    }
