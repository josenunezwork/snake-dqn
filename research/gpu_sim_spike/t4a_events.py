"""Executed exceptional-path witnesses for the frozen T4A premise test.

The matrix harness owns cross-batch comparison.  This module deliberately owns
only the five independently reset two-tick fixtures required by the frozen
protocol.  A fixture may arrange an exceptional starting state, but it always
uses :class:`DeviceSim`'s normal ``prepare``/``step`` lifecycle afterwards.
Missing observations are reported as false coverage; names and setup flags are
never treated as evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import integration_engine as engine
from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike import t4a_compare as compare
from research.gpu_sim_spike import t4a_fixtures as fixtures
from research.gpu_sim_spike.t4a_capture import Capture

EVENTS = (
    "ring_wrap",
    "simultaneous_death_respawn",
    "contested_food_refill",
    "clone_restore",
    "stale_cache",
)
WORLD_SIZES = (4, 64, 1024)
_TAPES: dict[int, list[dict[str, torch.Tensor]]] = {}


def _mode_flags(lane: str) -> tuple[bool, bool]:
    if lane == "smoke":
        return True, False
    if lane == "canary":
        return False, True
    if lane == "main":
        return False, False
    raise ValueError(f"unknown T4A lane {lane!r}")


def _snapshot(case: tuple) -> dict[str, torch.Tensor]:
    sim, policy = case[:2]
    result = compare.persistent_tensors(sim, "sim")
    result.update(compare.persistent_tensors(policy.vector_runtime, "runtime"))
    return result


def _mapped_snapshot(case: tuple) -> dict[str, torch.Tensor]:
    """Persistent fields for the four canonical logical event worlds."""
    sim = case[0]
    rows = torch.arange(4, device=sim.device)
    fields = compare.world_tensors(case[0], "sim", sim.E)
    fields.update(compare.world_tensors(case[1].vector_runtime, "runtime", sim.E))
    return {name: value.index_select(0, rows).contiguous() for name, value in fields.items()}


def _state_digest(case: tuple) -> str:
    compact = compare.compact_inventory(_snapshot(case))
    return hashlib.sha256(json.dumps(compact, sort_keys=True).encode()).hexdigest()


def _actions(sim: Any) -> torch.Tensor:
    """Frozen supplied actions for exceptional fixtures only."""
    return torch.ones((sim.E, sim.S), dtype=torch.long, device=sim.device)


def _tick(
    case: tuple,
    actions: torch.Tensor | None = None,
    after_prepare: Callable[[], None] | None = None,
) -> None:
    """Use the candidate's normal prepare, step, and vector carry paths."""
    sim, policy = case[:2]
    # The fixture can replace actions only after every ordinary learned forward
    # runs, retaining B8/vector61/v8 preparation and carry maintenance.
    with Capture(case) as cap:
        with engine._arm("I"), torch.inference_mode():
            cap.begin_tick()
            sim.prepare()
            if after_prepare is not None:
                after_prepare()
            _q, _legal, hero = policy.evaluate(sim)
            policy.actions(sim, hero)
            # Every fixture is an event-only supplied-action witness.
            # Normal policy actions are captured before their replacement.
            supplied = _actions(sim) if actions is None else actions
            sim.step(supplied)
            policy.after_step(sim)
        rows = (0, 1, 2, 3)
        captured = cap.fields(rows)
        supplied_mapped = supplied.index_select(
            0, torch.tensor(rows, device=sim.device, dtype=torch.long)
        ).contiguous()
    # Capture temporarily installs an instance method; inspect persistent
    # fields only after its restoration so the strict inventory sees no hook.
    fields = _mapped_snapshot(case)
    fields.update(captured)
    fields["event.supplied_actions"] = supplied_mapped
    _TAPES.setdefault(id(sim), []).append(fields)


def _set_head(sim: Any, slot: int, x: int, y: int, direction: int) -> None:
    rows = torch.arange(sim.E, device=sim.device)
    sim.head_ptr[:, slot] = 0
    sim.seg_count[:, slot] = 1
    sim.length[:, slot] = 1
    sim.alive[:, slot] = True
    sim.bodies[rows, slot, 0, 0] = x
    sim.bodies[rows, slot, 0, 1] = y
    sim.direction[:, slot] = direction


def _event_ring_wrap(case: tuple) -> tuple[bool, dict[str, Any]]:
    sim = case[0]
    # ``_push`` advances the circular storage pointer; this starts immediately
    # before index zero, while preserving a two-segment logical body.
    rows = torch.arange(sim.E, device=sim.device)
    sim.head_ptr[:, 0] = sim.cap - 1
    sim.seg_count[:, 0] = 2
    sim.length[:, 0] = 2
    sim.alive[:, 0] = True
    sim.bodies[rows, 0, sim.cap - 1] = torch.tensor([4, 4], device=sim.device)
    sim.bodies[rows, 0, sim.cap - 2] = torch.tensor([3, 4], device=sim.device)
    sim.direction[:, 0] = 1
    before = sim.head_ptr[:, 0].clone()
    _tick(case)
    after = sim.head_ptr[:, 0]
    wrapped = bool(torch.all(before == sim.cap - 1).item() and torch.all(after == 0).item())
    ordered = bool(torch.all(sim.seg_count[:, 0] >= 2).item())
    _tick(case)
    return wrapped and ordered, {
        "before": before.tolist(),
        "after": after.tolist(),
        "ordered": ordered,
    }


def _event_death_respawn(case: tuple) -> tuple[bool, dict[str, Any]]:
    sim = case[0]
    # Bent four-cell bodies make both victims self-collide in arena.  That
    # forces observed corpse production before their ordinary respawn.
    for slot, y in ((1, 2), (2, 5)):
        _set_head(sim, slot, 5, y, 1)
        sim.seg_count[:, slot] = 4
        sim.length[:, slot] = 4
        # ``head_ptr`` is zero before the move; after one push, old logical
        # offset two is read at physical ``cap - 2`` and becomes offset three.
        sim.bodies[:, slot, sim.cap - 1] = torch.tensor([4, y], device=sim.device)
        sim.bodies[:, slot, sim.cap - 2] = torch.tensor([6, y], device=sim.device)
        sim.bodies[:, slot, sim.cap - 3] = torch.tensor([4, y + 1], device=sim.device)
    _tick(case)
    died = sim._last_done[:, 1] & sim._last_done[:, 2]
    corpse = sim.corpse_food.any(dim=1)
    with torch.inference_mode():
        sim.respawn_timer[:, 1] = torch.where(
            died, torch.ones_like(sim.respawn_timer[:, 1]), sim.respawn_timer[:, 1]
        )
        sim.respawn_timer[:, 2] = torch.where(
            died, torch.ones_like(sim.respawn_timer[:, 2]), sim.respawn_timer[:, 2]
        )
    _tick(case)
    respawned = sim.alive[:, 1] & sim.alive[:, 2]
    ok = bool(torch.all(died).item() and torch.all(respawned).item() and torch.any(corpse).item())
    return ok, {"died": died.tolist(), "respawned": respawned.tolist(), "corpse": corpse.tolist()}


def _event_contested_food(case: tuple) -> tuple[bool, dict[str, Any]]:
    sim = case[0]
    # Two living opponents enter the same ordered ambient-food cell.  Slot 1
    # must own it because DeviceSim resolves food in increasing slot order.
    _set_head(sim, 1, 16, 5, 1)
    _set_head(sim, 2, 18, 5, 3)

    def install_contested_food() -> None:
        sim.food_present.zero_()
        sim.corpse_food.zero_()
        sim.food_count.zero_()
        sim.food_cells.zero_()
        sim.food_cells[:, 0] = torch.tensor([17, 5], device=sim.device)
        sim.food_present[:, 0] = True
        sim.corpse_food[:, 0] = False
        sim.food_count[:] = 1

    rng_before = sim.rng_counter.clone()
    _tick(case, after_prepare=install_contested_food)
    ate_1, ate_2 = sim._last_ambient_food_ate[:, 1], sim._last_ambient_food_ate[:, 2]
    # Train-mode step refills after consumption; a second ordinary lifecycle
    # witnesses the ordered food state and additional counter progression.
    food_after_first = sim.food_count.clone()
    refill_cells = sim.food_cells[:, 0].clone()
    _tick(case)
    rng_after = sim.rng_counter
    ordered = ate_1 & ~ate_2
    # The ordered target was consumed by slot 1 and the normal food lifecycle
    # left a live ordered entry.  Its coordinates may legitimately repeat when
    # the deterministic refill draw selects the vacated cell, so identity is
    # established by consumption plus the live post-refill list and RNG tape.
    refilled = food_after_first > 0
    rng_moved = rng_after > rng_before
    ok = bool(
        torch.all(ordered).item() and torch.all(refilled).item() and torch.all(rng_moved).item()
    )
    return ok, {
        "slot_1_ate": ate_1.tolist(),
        "slot_2_ate": ate_2.tolist(),
        "food_after_first": food_after_first.tolist(),
        "refill_cells": refill_cells.tolist(),
        "rng_before": rng_before.tolist(),
        "rng_after": rng_after.tolist(),
    }


def _clone_policy(policy: Any) -> Any:
    # DevicePolicy carries tensors and fixed-forward modules only.  Deep copy
    # keeps an independent vector runtime rather than sharing mutable carry.
    return copy.deepcopy(policy)


def _event_clone_restore(case: tuple) -> tuple[bool, dict[str, Any]]:
    sim, policy = case[:2]
    _tick(case)
    saved = (sim.clone(), _clone_policy(policy))
    # Continue the original branch and independently restore the complete
    # post-step seam (including vector runtime carry and RNG) into branch B.
    branch_a = case
    branch_b = (case[2].clone(), _clone_policy(case[3]), *case[2:])
    before_restore = _state_digest(branch_b)
    ruler._reset(branch_b[0], branch_b[1], saved[0], saved[1])
    restored = compare.compare_inventory(_snapshot(branch_b), _snapshot((saved[0], saved[1])))
    _tick(branch_a)
    _tick(branch_b)
    mismatch = restored or compare.compare_inventory(_snapshot(branch_a), _snapshot(branch_b))
    mismatch = mismatch or compare.compare_inventory(
        _TAPES[id(branch_a[0])][-1], _TAPES[id(branch_b[0])][-1]
    )
    _TAPES.pop(id(branch_b[0]), None)
    return mismatch is None and restored is None, {
        "before_restore_digest": before_restore,
        "restored_exact": restored is None,
        "saved_digest": _state_digest((saved[0], saved[1], *case[2:])),
        "branch_digest": _state_digest(branch_a),
        "mismatch": None if mismatch is None else mismatch.__dict__,
    }


def _event_stale_cache(
    case: tuple, checkpoint: Path, roster: str, device: str, lane: str
) -> tuple[bool, dict[str, Any]]:
    smoke, canary = _mode_flags(lane)
    sim, policy = case[:2]
    # Prime the warm branch through real prepare/geometry reads, then clone it.
    sim.prepare()
    sim.advisory_mask()
    warm_stats = r2_cache.stats(sim)
    clone = (sim.clone(), _clone_policy(policy), *case[2:])
    fresh, _wrappers = fixtures.build_case(
        checkpoint, roster, sim.E, device, smoke=smoke, canary=canary
    )
    fresh_sim, fresh_policy = fresh[:2]
    fresh_sim.prepare()
    fresh_sim.advisory_mask()
    # The warm original, cold clone, and independently fresh equivalent all
    # progress twice through ordinary lifecycle invalidation.
    _tick(case)
    _tick(clone)
    _tick(fresh)
    _tick(case)
    _tick(clone)
    _tick(fresh)
    mismatch = compare.compare_inventory(_snapshot(case), _snapshot(clone))
    mismatch = mismatch or compare.compare_inventory(_snapshot(clone), _snapshot(fresh))
    for index in range(2):
        mismatch = mismatch or compare.compare_inventory(
            _TAPES[id(sim)][index], _TAPES[id(clone[0])][index]
        )
        mismatch = mismatch or compare.compare_inventory(
            _TAPES[id(sim)][index], _TAPES[id(fresh[0])][index]
        )
    _TAPES.pop(id(clone[0]), None)
    _TAPES.pop(id(fresh[0]), None)
    clone_stats, fresh_stats = r2_cache.stats(clone[0]), r2_cache.stats(fresh_sim)
    lifecycle = warm_stats.get("misses", 0) > 0 and clone_stats.get("bumps", 0) > 0
    return mismatch is None and lifecycle, {
        "warm_stats": warm_stats,
        "clone_stats": clone_stats,
        "fresh_stats": fresh_stats,
        "mismatch": None if mismatch is None else mismatch.__dict__,
        "lifecycle": lifecycle,
    }


def _run_one(
    event: str, checkpoint: Path, roster: str, worlds: int, device: str, lane: str
) -> tuple[bool, dict[str, Any]]:
    smoke, canary = _mode_flags(lane)
    case, _wrappers = fixtures.build_case(
        checkpoint, roster, worlds, device, smoke=smoke, canary=canary
    )
    _TAPES[id(case[0])] = []
    table: dict[str, Callable[..., tuple[bool, dict[str, Any]]]] = {
        "ring_wrap": _event_ring_wrap,
        "simultaneous_death_respawn": _event_death_respawn,
        "contested_food_refill": _event_contested_food,
        "clone_restore": _event_clone_restore,
    }
    if event == "stale_cache":
        passed, witness = _event_stale_cache(case, checkpoint, roster, device, lane)
    else:
        passed, witness = table[event](case)
    tape = _TAPES.pop(id(case[0]), [])
    witness["ticks"] = len(tape)
    witness["mapped_tape"] = [compare.compact_inventory(item) for item in tape]
    witness["action_q_tape"] = [
        {
            "actions_hex": item["event.supplied_actions"].cpu().numpy().tobytes().hex(),
            "policy_actions_hex": item["output.actions"].cpu().numpy().tobytes().hex(),
            "q_hex": {
                name.removeprefix("output.q."): value.cpu().numpy().tobytes().hex()
                for name, value in item.items()
                if name.startswith("output.q.")
            },
        }
        for item in tape
    ]
    # The raw tensors are consumed by the immediate E comparator and never
    # written into the sealed evidence JSON.
    witness["_raw_mapped_tape"] = tape
    return passed, witness


def run_events(
    checkpoint: Path, roster: str, device: str, lane: str, deadline: float
) -> dict[str, Any]:
    """Run all frozen exceptional-path fixtures, returning only executed evidence.

    Failure and timeout are intentionally coverage false.  The caller can then
    classify an otherwise clean matrix as ``INCOMPLETE`` without confusing a
    fixture construction failure for an invariance mismatch.
    """
    result: dict[str, Any] = {name: False for name in EVENTS}
    evidence: list[dict[str, Any]] = []
    sizes = (4,) if lane == "smoke" else WORLD_SIZES
    references: dict[str, list[dict[str, torch.Tensor]]] = {}
    for worlds in sizes:
        for event in EVENTS:
            if time.monotonic() >= deadline:
                evidence.append({"event": event, "worlds": worlds, "ticks": 0, "status": "timeout"})
                continue
            try:
                passed, witness = _run_one(event, checkpoint, roster, worlds, device, lane)
                raw_tape = witness.pop("_raw_mapped_tape", [])
                mismatch: dict[str, Any] | None = witness.get("mismatch")
                if worlds == 4:
                    references[event] = raw_tape
                else:
                    reference = references.get(event, [])
                    if len(reference) != len(raw_tape):
                        mismatch = {
                            "field": "event_ticks",
                            "expected": len(reference),
                            "actual": len(raw_tape),
                        }
                    else:
                        for tick, (left, right) in enumerate(zip(reference, raw_tape)):
                            found = compare.compare_inventory(left, right)
                            if found is not None:
                                mismatch = {"tick": tick, **found.__dict__}
                                break
                if mismatch is not None:
                    passed = False
                    witness["first_witness"] = mismatch
                encoded = json.dumps(witness, sort_keys=True, default=str).encode()
                evidence.append(
                    {
                        "event": event,
                        "worlds": worlds,
                        "ticks": 2,
                        "status": (
                            "PASS"
                            if passed
                            else (
                                "INVALID_SELF_CONSISTENCY" if mismatch is not None else "INCOMPLETE"
                            )
                        ),
                        "witness_sha256": hashlib.sha256(encoded).hexdigest(),
                        "witness": witness,
                    }
                )
                result[event] = bool(result[event] or passed)
                if mismatch is not None:
                    result["evidence"] = evidence
                    return result
            except ValueError:
                # Invalid captures/field inventories/nonfinite Q are evidence
                # failures, never mechanical completion opportunities.
                raise
            except Exception as exc:  # Unsupported/unexercised fixtures fail closed.
                evidence.append(
                    {
                        "event": event,
                        "worlds": worlds,
                        "ticks": 0,
                        "status": "INCOMPLETE",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    result["evidence"] = evidence
    # A required E cell failing cannot be hidden by an earlier E pass.
    for name in EVENTS:
        rows = [row for row in evidence if row["event"] == name]
        result[name] = len(rows) == len(sizes) and all(row["status"] == "PASS" for row in rows)
    return result
