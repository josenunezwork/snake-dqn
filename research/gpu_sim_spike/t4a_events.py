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
from unittest.mock import patch

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


def _snapshot_copy(case: tuple) -> dict[str, torch.Tensor]:
    """Freeze persistent tensors for a before/after fixture assertion."""
    return {name: value.detach().clone() for name, value in _snapshot(case).items()}


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


def _tensor_hex(value: torch.Tensor) -> str:
    """Serialize bounded negative-control evidence with its type recorded separately."""
    return value.detach().contiguous().cpu().numpy().tobytes().hex()


def _tensor_commitment(value: torch.Tensor) -> dict[str, Any]:
    """Return the typed byte commitment used by the sealed event witness."""
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _mapped_event_fields(sim: Any, fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map an event-local E tensor through the canonical four witness rows."""
    rows = torch.arange(4, device=sim.device)
    return {name: value.index_select(0, rows).contiguous() for name, value in fields.items()}


def _actions(sim: Any) -> torch.Tensor:
    """Frozen supplied actions for exceptional fixtures only."""
    return torch.ones((sim.E, sim.S), dtype=torch.long, device=sim.device)


def _tick(
    case: tuple,
    actions: torch.Tensor | None = None,
    after_prepare: Callable[[], None] | None = None,
    event_fields: Callable[[], dict[str, torch.Tensor]] | None = None,
    observe_food_transition: bool = False,
) -> dict[str, torch.Tensor]:
    """Use the candidate's normal prepare, step, and vector carry paths."""
    sim, policy = case[:2]
    # The fixture can replace actions only after every ordinary learned forward
    # runs, retaining B8/vector61/v8 preparation and carry maintenance.
    transition: dict[str, torch.Tensor] = {}
    prepared_boundary = bool(sim._prepared and policy.vector_prepared)
    boundary_carry: dict[str, torch.Tensor] = {}
    with Capture(case) as cap:
        with engine._arm("I"), torch.inference_mode():
            cap.begin_tick()
            sim.prepare()
            if after_prepare is not None:
                after_prepare()
            _q, _legal, hero = policy.evaluate(sim)
            policy.actions(sim, hero)
            if prepared_boundary:
                boundary_carry = _prepared_carry(case)
            # Every fixture is an event-only supplied-action witness.
            # Normal policy actions are captured before their replacement.
            supplied = _actions(sim) if actions is None else actions
            if observe_food_transition:
                original_maintain = sim._maintain_food

                def maintain() -> None:
                    transition.update(
                        {
                            "event.contested_food.after_consume_cells": (
                                sim.food_cells.detach().clone()
                            ),
                            "event.contested_food.after_consume_present": (
                                sim.food_present.detach().clone()
                            ),
                            "event.contested_food.after_consume_count": (
                                sim.food_count.detach().clone()
                            ),
                        }
                    )
                    original_maintain()
                    transition.update(
                        {
                            "event.contested_food.after_refill_cells": (
                                sim.food_cells.detach().clone()
                            ),
                            "event.contested_food.after_refill_present": (
                                sim.food_present.detach().clone()
                            ),
                            "event.contested_food.after_refill_count": (
                                sim.food_count.detach().clone()
                            ),
                        }
                    )

                with patch.object(sim, "_maintain_food", maintain):
                    sim.step(supplied)
            else:
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
    fields.update(boundary_carry)
    fields.update(_mapped_event_fields(sim, transition))
    if event_fields is not None:
        fields.update(_mapped_event_fields(sim, event_fields()))
    fields["event.supplied_actions"] = supplied_mapped
    _TAPES.setdefault(id(sim), []).append(fields)
    return fields


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

    def ring_capture() -> dict[str, torch.Tensor]:
        after_ptr = sim.head_ptr[:, 0]
        offsets = torch.arange(2, device=sim.device)
        ring = (after_ptr[:, None] - offsets[None, :]) % sim.cap
        prefix = sim.bodies[:, 0].gather(1, ring[..., None].expand(-1, -1, 2)).contiguous()
        return {
            "event.ring_wrap.before_head_ptr": before,
            "event.ring_wrap.after_head_ptr": after_ptr.detach().clone(),
            "event.ring_wrap.prefix": prefix,
        }

    first_tick = _tick(case, event_fields=ring_capture)
    after = sim.head_ptr[:, 0]
    wrapped = bool(torch.all(before == sim.cap - 1).item() and torch.all(after == 0).item())
    # ``seg_count >= 2`` proves only occupancy.  Read the live circular prefix
    # in logical order so this witness establishes that the moved head precedes
    # the retained tail across the physical wrap.
    offsets = torch.arange(2, device=sim.device)
    ring = (after[:, None] - offsets[None, :]) % sim.cap
    logical_prefix = sim.bodies[:, 0].gather(1, ring[..., None].expand(-1, -1, 2)).contiguous()
    expected_head = torch.tensor([5, 4], device=sim.device)
    expected_tail = torch.tensor([4, 4], device=sim.device)
    expected = torch.stack((expected_head, expected_tail)).expand_as(logical_prefix)
    ordered = bool(torch.equal(logical_prefix, expected))
    _tick(case)
    return wrapped and ordered, {
        "before": before.tolist(),
        "after": after.tolist(),
        "logical_prefix": logical_prefix.tolist(),
        # The reducer derives capacity and body dtype/shape from the captured
        # ``sim.bodies`` descriptor, then checks this is the frozen geometry.
        # It never accepts producer-selected coordinates as an oracle.
        "ring_schema": {
            "field": "sim.bodies",
            "tick": 0,
            "capacity": sim.cap,
            "dtype": str(sim.bodies.dtype),
            "shape": list(first_tick["sim.bodies"].shape),
            "physical_worlds": sim.E,
        },
        "frozen_geometry": {
            "slot": 0,
            "direction": 1,
            "prefix_length": 2,
            "initial_head": [4, 4],
            "initial_tail": [3, 4],
            "moved_head": expected_head.tolist(),
            "retained_tail": expected_tail.tolist(),
        },
        "ring_prefix_capture": {
            "field": "event.ring_wrap.prefix",
            "tick": 0,
            **_tensor_commitment(first_tick["event.ring_wrap.prefix"]),
        },
        "ring_before_capture": {
            "field": "event.ring_wrap.before_head_ptr",
            "tick": 0,
            **_tensor_commitment(first_tick["event.ring_wrap.before_head_ptr"]),
        },
        "ring_after_capture": {
            "field": "event.ring_wrap.after_head_ptr",
            "tick": 0,
            **_tensor_commitment(first_tick["event.ring_wrap.after_head_ptr"]),
        },
        "wrapped": wrapped,
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
    death_tick = _tick(
        case,
        event_fields=lambda: {
            "event.death_respawn.died": sim._last_done[:, (1, 2)].all(dim=1),
            "event.death_respawn.corpse": sim.corpse_food.any(dim=1),
        },
    )
    died = sim._last_done[:, 1] & sim._last_done[:, 2]
    corpse = sim.corpse_food.any(dim=1)
    with torch.inference_mode():
        sim.respawn_timer[:, 1] = torch.where(
            died, torch.ones_like(sim.respawn_timer[:, 1]), sim.respawn_timer[:, 1]
        )
        sim.respawn_timer[:, 2] = torch.where(
            died, torch.ones_like(sim.respawn_timer[:, 2]), sim.respawn_timer[:, 2]
        )
    respawn_tick = _tick(
        case,
        event_fields=lambda: {"event.death_respawn.respawned": sim.alive[:, (1, 2)].all(dim=1)},
    )
    respawned = sim.alive[:, 1] & sim.alive[:, 2]
    ok = bool(torch.all(died).item() and torch.all(respawned).item() and torch.any(corpse).item())
    return ok, {
        "died": died.tolist(),
        "respawned": respawned.tolist(),
        "corpse": corpse.tolist(),
        "death_respawn_capture": {
            "died": {
                "field": "event.death_respawn.died",
                "tick": 0,
                **_tensor_commitment(death_tick["event.death_respawn.died"]),
            },
            "corpse": {
                "field": "event.death_respawn.corpse",
                "tick": 0,
                **_tensor_commitment(death_tick["event.death_respawn.corpse"]),
            },
            "respawned": {
                "field": "event.death_respawn.respawned",
                "tick": 1,
                **_tensor_commitment(respawn_tick["event.death_respawn.respawned"]),
            },
        },
    }


def _event_contested_food(case: tuple) -> tuple[bool, dict[str, Any]]:
    sim = case[0]
    # Two living opponents enter the same ordered ambient-food cell.  Slot 1
    # must own it because DeviceSim resolves food in increasing slot order.
    _set_head(sim, 1, 16, 5, 1)
    _set_head(sim, 2, 18, 5, 3)

    # This is fixture setup, before the ordinary prepare/policy/step lifecycle.
    # ``prepare`` retains this valid ambient-food entry; mutating it after
    # prepare would evade the protocol's setup-only constraint.
    sim.food_present.zero_()
    sim.corpse_food.zero_()
    sim.food_count.zero_()
    sim.food_cells.zero_()
    sim.food_cells[:, 0] = torch.tensor([17, 5], device=sim.device)
    sim.food_present[:, 0] = True
    sim.corpse_food[:, 0] = False
    sim.food_count[:] = 1

    rng_before = sim.rng_counter.clone()
    prepared_transition: dict[str, torch.Tensor] = {}

    def capture_pre_consume() -> None:
        # ``prepare`` may refill the setup list to max_food.  This is the
        # immediate ordered ambient state from which the contested target is
        # consumed, not the producer's pre-prepare fixture arrangement.
        prepared_transition.update(
            {
                "event.contested_food.before_cells": sim.food_cells.detach().clone(),
                "event.contested_food.before_present": sim.food_present.detach().clone(),
                "event.contested_food.before_count": sim.food_count.detach().clone(),
                "event.contested_food.before_corpse": sim.corpse_food.detach().clone(),
            }
        )

    _tick(
        case,
        after_prepare=capture_pre_consume,
        event_fields=lambda: {
            "event.contested_food.after_consume_cells": sim.food_cells.detach().clone(),
            "event.contested_food.after_consume_present": sim.food_present.detach().clone(),
            "event.contested_food.after_consume_count": sim.food_count.detach().clone(),
            "event.contested_food.after_consume_corpse": sim.corpse_food.detach().clone(),
        },
    )
    ate_1, ate_2 = sim._last_ambient_food_ate[:, 1], sim._last_ambient_food_ate[:, 2]
    # Eval-mode food maintenance happens at the next ordinary ``prepare``;
    # preserve its immediate post-maintenance state before tick 1 consumes.
    food_after_first = sim.food_count.clone()
    refill_transition: dict[str, torch.Tensor] = {}

    def capture_after_refill() -> None:
        refill_transition.update(
            {
                "event.contested_food.after_refill_cells": sim.food_cells.detach().clone(),
                "event.contested_food.after_refill_present": sim.food_present.detach().clone(),
                "event.contested_food.after_refill_count": sim.food_count.detach().clone(),
                "event.contested_food.after_refill_corpse": sim.corpse_food.detach().clone(),
            }
        )

    _tick(case, after_prepare=capture_after_refill)
    rng_after = sim.rng_counter
    ordered = ate_1 & ~ate_2
    refilled = torch.tensor(
        [
            int(refill_transition["event.contested_food.after_refill_count"][row].item())
            > int(food_after_first[row].item())
            for row in range(sim.E)
        ],
        dtype=torch.bool,
        device=sim.device,
    )
    rng_moved = rng_after > rng_before
    ok = bool(
        torch.all(ordered).item() and torch.all(refilled).item() and torch.all(rng_moved).item()
    )
    transition_fields = (
        "event.contested_food.before_cells",
        "event.contested_food.before_present",
        "event.contested_food.before_count",
        "event.contested_food.before_corpse",
        "event.contested_food.after_consume_cells",
        "event.contested_food.after_consume_present",
        "event.contested_food.after_consume_count",
        "event.contested_food.after_consume_corpse",
        "event.contested_food.after_refill_cells",
        "event.contested_food.after_refill_present",
        "event.contested_food.after_refill_count",
        "event.contested_food.after_refill_corpse",
    )
    # Bind the pre-step side of the ordered transition into the same capture
    # tape as the maintenance boundary; this makes refill evidence inspectable
    # without treating the fixture's summary booleans as proof.
    first_tape, second_tape = _TAPES[id(sim)]
    first_tape.update(
        _mapped_event_fields(
            sim,
            {
                **prepared_transition,
            },
        )
    )
    second_tape.update(_mapped_event_fields(sim, refill_transition))
    transition_values = {
        name.removeprefix("event.contested_food."): (
            first_tape[name].tolist() if "after_refill" not in name else second_tape[name].tolist()
        )
        for name in transition_fields
    }
    after_consume_count = first_tape["event.contested_food.after_consume_count"]
    after_refill_count = second_tape["event.contested_food.after_refill_count"]
    after_refill_cells = second_tape["event.contested_food.after_refill_cells"]
    # Food compaction preserves survivors in order and maintenance appends new
    # ambient cells. Export exactly that appended suffix rather than merely the
    # first post-refill survivor, which may predate the contested consumption.
    refill_appended_cells = [
        after_refill_cells[row, int(before.item()) : int(after.item())].tolist()
        for row, (before, after) in enumerate(zip(after_consume_count, after_refill_count))
    ]
    return ok, {
        "slot_1_ate": ate_1.tolist(),
        "slot_2_ate": ate_2.tolist(),
        "food_after_first": food_after_first.tolist(),
        "refill_cells": refill_appended_cells,
        "rng_before": rng_before.tolist(),
        "rng_after": rng_after.tolist(),
        "food_transition": transition_values,
        "food_transition_capture": {
            name.removeprefix("event.contested_food."): {
                "field": name,
                "tick": 1 if "after_refill" in name else 0,
                **_tensor_commitment(
                    second_tape[name] if "after_refill" in name else first_tape[name]
                ),
            }
            for name in transition_fields
        },
    }


def _clone_policy(policy: Any) -> Any:
    # DevicePolicy carries tensors and fixed-forward modules only.  Deep copy
    # keeps an independent vector runtime rather than sharing mutable carry.
    return copy.deepcopy(policy)


def _remap_world_tensors(case: tuple, rows: torch.Tensor) -> None:
    """Apply a physical world-row remap to the simulator and vector runtime."""
    sim, policy = case[:2]
    for owner, names in ((sim, compare.SIM_WORLD), (policy.vector_runtime, compare.RUNTIME_WORLD)):
        for name in names:
            value = getattr(owner, name)
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == sim.E:
                value.copy_(value.index_select(0, rows))
    r2_cache.bump(sim)


def _prepared_carry(case: tuple) -> dict[str, torch.Tensor]:
    """Freeze vector carry at the policy boundary, before after_step mutates it."""
    return {
        "prepared." + name: value.detach().clone()
        for name, value in _mapped_snapshot(case).items()
        if name in {"runtime._carry_state", "runtime._carry_mask", "runtime._carry_valid"}
    }


def _prepared_outputs(fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Select semantic outputs shared by pre-step and prepared-tick captures."""
    return {
        name: value
        for name, value in fields.items()
        if name.startswith(("output.q.", "output.obs.", "output.v8.", "prepared.runtime."))
        or name in {"output.actions", "configuration.sha256"}
    }


def _warm_output_capture(case: tuple) -> dict[str, torch.Tensor]:
    """Capture real Q, observation, v8 and advisory output at a warm seam."""
    sim, policy = case[:2]
    with Capture(case) as cap:
        with engine._arm("I"), torch.inference_mode():
            cap.begin_tick()
            sim.prepare()
            sim.advisory_mask()
            _q, _legal, hero = policy.evaluate(sim)
            policy.actions(sim, hero)
        # Source invokes vector ``prepare`` on this first warm action while a
        # cloned prepared runtime correctly skips it.  The prepare path adds
        # accounting and an extra indexed advisory instrumentation call. Keep
        # the stable semantic boundary: Q, observations, v8 (including masks),
        # selected actions, lifecycle, and configuration.
        fields = cap.fields((0, 1, 2, 3))
        fields.update(_prepared_carry(case))
        return {
            name: value
            for name, value in fields.items()
            if (
                name.startswith(("output.q.", "output.obs.", "output.v8.", "prepared.runtime."))
                or name in {"output.actions", "lifecycle.sim_prepared", "lifecycle.vector_prepared"}
                or name == "configuration.sha256"
            )
        }


def _warm_outputs_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Retain a structured harness check, never an engine mismatch witness."""
    difference = compare.compare_inventory(left, right)
    return {
        "equal": difference is None,
        "kind": type(difference).__name__ if difference is not None else None,
        "difference": difference.__dict__.copy() if difference is not None else None,
    }


def _schema_witness(value: Any) -> dict[str, Any] | None:
    """Serialize a harness schema failure without mislabelling it as engine evidence."""
    if isinstance(value, compare.SchemaIncompatibility):
        return value.__dict__.copy()
    return None


def _first_schema(*values: Any) -> dict[str, Any] | None:
    return next(
        (witness for value in values if (witness := _schema_witness(value)) is not None), None
    )


def _event_clone_restore(case: tuple) -> tuple[bool, dict[str, Any]]:
    sim, policy = case[:2]
    # Establish a deterministic ordinary food transition on the saved seam:
    # the first tick moves the controlled hero from [4,4] to [5,4], leaving
    # the ambient [6,4] entry for the resident restore target's next tick.
    # The target therefore proves body and food progression before reset while
    # reset still restores the exact, unedited saved snapshot.
    _set_head(sim, 0, 4, 4, 1)
    sim.food_present.zero_()
    sim.corpse_food.zero_()
    sim.food_count.zero_()
    sim.food_cells[:, 0] = torch.tensor([6, 4], device=sim.device)
    sim.food_present[:, 0] = True
    sim.food_count[:] = 1
    _tick(case)
    saved = (sim.clone(), _clone_policy(policy))
    # Continue the original branch and independently restore the complete
    # post-step seam (including vector runtime carry and RNG) into branch B.
    branch_a = case
    # A full-state snapshot defines tensor dtypes as well as values. The
    # benchmark's initial reset template can have pre-step reward dtypes, so
    # materialize the saved typed snapshot before testing in-place restore.
    branch_b = (saved[0].clone(), _clone_policy(saved[1]), *case[2:])
    branch_b[0].rng_counter += 7
    branch_b[0].frame += 2
    branch_b[1].vector_runtime._carry_valid.logical_not_()
    # Review-02 requires the resident restore target to traverse real body and
    # food mechanics before reset, rather than exercising only scalar cursors.
    # Keep the exact saved food inventory intact so this remains a production
    # reset-path test rather than a reset capability test for fixture edits.
    target_before = _snapshot_copy(branch_b)
    _tick(branch_b)
    target_after = _snapshot(branch_b)
    target_body_progressed = (
        compare.compare(target_before["sim.bodies"], target_after["sim.bodies"], "sim.bodies")
        is not None
    )
    target_food_progressed = any(
        compare.compare(target_before[name], target_after[name], name) is not None
        for name in ("sim.food_cells", "sim.food_present", "sim.food_count")
    )
    before_restore = _state_digest(branch_b)
    ruler._reset(branch_b[0], branch_b[1], saved[0], saved[1])
    restored_left = _snapshot_copy(branch_b)
    restored_right = _snapshot_copy((saved[0], saved[1]))
    restored = compare.compare_inventory(restored_left, restored_right)
    restored_digest = _state_digest(branch_b)
    _tick(branch_a)
    _tick(branch_b)
    branch_left, branch_right = _snapshot(branch_a), _snapshot(branch_b)
    branch_compare = compare.compare_inventory(branch_left, branch_right)
    replay_left, replay_right = _TAPES[id(branch_a[0])][-1], _TAPES[id(branch_b[0])][-1]
    replay_compare = compare.compare_inventory(replay_left, replay_right)
    schema_incompatibility = _first_schema(restored, branch_compare, replay_compare)
    mismatch = None
    for value, left, right in (
        (restored, restored_left, restored_right),
        (branch_compare, branch_left, branch_right),
        (replay_compare, replay_left, replay_right),
    ):
        if isinstance(value, compare.Mismatch):
            mismatch = _internal_byte_witness(value, left, right, sim.E)
            break
    _TAPES.pop(id(branch_b[0]), None)
    replay_digest = _state_digest(branch_b)
    return (
        mismatch is None
        and schema_incompatibility is None
        and restored is None
        and target_body_progressed
        and target_food_progressed
    ), {
        "before_restore_digest": before_restore,
        "restored_exact": restored is None,
        "saved_digest": _state_digest((saved[0], saved[1], *case[2:])),
        "restored_digest": restored_digest,
        "branch_digest": _state_digest(branch_a),
        "replay_digest": replay_digest,
        "mismatch": mismatch,
        "schema_incompatibility": schema_incompatibility,
        "restore_target_body_progressed": target_body_progressed,
        "restore_target_food_progressed": target_food_progressed,
    }


@torch.inference_mode()
def _event_stale_cache(
    case: tuple, checkpoint: Path, roster: str, device: str, lane: str
) -> tuple[bool, dict[str, Any]]:
    smoke, canary = _mode_flags(lane)
    sim, policy = case[:2]
    sealed_mismatches: list[dict[str, Any]] = []
    prepared_ticks: dict[str, dict[str, torch.Tensor]] = {}

    def compared(
        left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
    ) -> compare.Mismatch | compare.SchemaIncompatibility | None:
        result = compare.compare_inventory(left, right)
        if isinstance(result, compare.Mismatch):
            # Seal the canonical evidence at detection time. Subsequent event
            # ticks mutate persistent tensors in place, so retaining live
            # dictionaries would bind the wrong state to this byte witness.
            sealed_mismatches.append(_internal_byte_witness(result, left, right, sim.E))
        return result

    # Materialize the ordinary post-step tensor dtypes before creating a
    # warmed restore source.  Resetting a progressed target from the initial
    # pre-step template can otherwise be a dtype mismatch rather than a cache
    # boundary test.  Setup ticks are not event evidence.
    _tick(case)
    _TAPES[id(sim)] = []
    # Prime the actual vector/policy path before the clone/remap/restore seam.
    # Calling only ``prepare``/``advisory_mask`` leaves vector-runtime cache and
    # carry state unexercised, which is the boundary this fixture certifies.
    warm_outputs = _warm_output_capture(case)
    if not policy.vector_prepared:
        raise RuntimeError("stale-cache warmup did not prepare vector runtime")
    warm_stats = r2_cache.stats(sim)
    warm_generation = r2_cache.generation(sim)
    warm_digest = _state_digest(case)
    clone = (sim.clone(), _clone_policy(policy), *case[2:])
    clone_warm_outputs = _warm_output_capture(clone)
    source_clone_warm_exact = _warm_outputs_equal(warm_outputs, clone_warm_outputs)
    clone_vector_prepared = bool(clone[1].vector_prepared)
    clone_generation = r2_cache.generation(clone[0])
    # Exercise a real device-to-device remap and inverse remap of every
    # E-indexed simulator/runtime field, then compare its ordinary replay with
    # the unremapped source.  The reverse is its own inverse, so the second
    # application restores the logical row order without regenerating RNG or
    # vector carry state.
    remapped = (sim.clone(), _clone_policy(policy), *case[2:])
    reverse_rows = torch.arange(sim.E - 1, -1, -1, device=sim.device)
    _remap_world_tensors(remapped, reverse_rows)
    _remap_world_tensors(remapped, reverse_rows)
    remap_warm_outputs = _warm_output_capture(remapped)
    source_remap_warm_exact = _warm_outputs_equal(warm_outputs, remap_warm_outputs)
    remap_compare = compared(_snapshot(remapped), _snapshot(case))
    remap_exact = remap_compare is None
    remap_restored_digest = _state_digest(remapped)
    remap_replay = (sim.clone(), _clone_policy(policy), *case[2:])
    prepared_ticks["remap"] = _prepared_outputs(_tick(remapped))
    prepared_ticks["remap_replay"] = _prepared_outputs(_tick(remap_replay))
    remap_replay_compare = compared(_snapshot(remapped), _snapshot(remap_replay))
    remap_replay_exact = remap_replay_compare is None
    remap_branch_digest = _state_digest(remapped)
    remap_replay_digest = _state_digest(remap_replay)
    _TAPES.pop(id(remapped[0]), None)
    _TAPES.pop(id(remap_replay[0]), None)
    # Restore the warmed snapshot into a separately warmed resident target
    # using the production reset path, then replay one ordinary tick on both
    # branches.  This makes the restore boundary a positive witness rather
    # than leaving it only to the deliberately broken control below.
    restored, _wrappers = fixtures.build_case(
        checkpoint, roster, sim.E, device, smoke=smoke, canary=canary
    )
    # The restore resident must execute a body *and food* transition before it
    # is reset.  Seed one legal ambient cell immediately ahead of the hero;
    # after the ordinary step the occupied cell cannot be its refill result.
    cardinal = torch.tensor(((0, -1), (1, 0), (0, 1), (-1, 0)), device=sim.device)
    target_heads = restored[0].heads()[:, 0]
    target_next = target_heads + cardinal[restored[0].direction[:, 0]]
    restored[0].food_present.zero_()
    restored[0].corpse_food.zero_()
    restored[0].food_count.zero_()
    restored[0].food_cells[:, 0] = target_next
    restored[0].food_present[:, 0] = True
    restored[0].food_count[:] = 1
    target_before_progress = _snapshot_copy(restored)
    _tick(restored)
    restored_sim, restored_policy = restored[:2]
    target_after_progress = _snapshot(restored)
    target_body_progressed = (
        compare.compare(
            target_before_progress["sim.bodies"], target_after_progress["sim.bodies"], "sim.bodies"
        )
        is not None
    )
    target_food_progressed = any(
        compare.compare(target_before_progress[name], target_after_progress[name], name) is not None
        for name in ("sim.food_cells", "sim.food_present", "sim.food_count")
    )
    _warm_output_capture(restored)
    ruler._reset(restored_sim, restored_policy, sim, policy)
    restore_compare = compared(_snapshot(restored), _snapshot(case))
    restore_exact = restore_compare is None
    restored_digest = _state_digest(restored)
    restored_warm_outputs = _warm_output_capture(restored)
    source_restore_warm_exact = _warm_outputs_equal(warm_outputs, restored_warm_outputs)
    replay = (sim.clone(), _clone_policy(policy), *case[2:])
    prepared_ticks["restore"] = _prepared_outputs(_tick(restored))
    prepared_ticks["restore_replay"] = _prepared_outputs(_tick(replay))
    restore_replay_compare = compared(_snapshot(restored), _snapshot(replay))
    restore_replay_exact = restore_replay_compare is None
    restore_branch_digest = _state_digest(restored)
    restore_replay_digest = _state_digest(replay)
    _TAPES.pop(id(restored_sim), None)
    _TAPES.pop(id(replay[0]), None)
    equivalent, _wrappers = fixtures.build_case(
        checkpoint, roster, sim.E, device, smoke=smoke, canary=canary
    )
    equivalent_sim, equivalent_policy = equivalent[:2]
    _tick(equivalent)
    _TAPES[id(equivalent_sim)] = []
    fresh_warm_outputs = _warm_output_capture(equivalent)
    source_fresh_warm_exact = _warm_outputs_equal(warm_outputs, fresh_warm_outputs)
    if not equivalent_policy.vector_prepared:
        raise RuntimeError("stale-cache fresh warmup did not prepare vector runtime")
    fresh_vector_prepared = bool(equivalent_policy.vector_prepared)
    fresh_generation = r2_cache.generation(equivalent_sim)
    # A one-byte Q alteration at the warm boundary must be observable to the
    # same full-output comparator used above.  This control is intentionally
    # local to the retained capture, never a policy or engine mutation.
    corrupted_warm = {name: value.clone() for name, value in clone_warm_outputs.items()}
    q_name = next(name for name in corrupted_warm if name.startswith("output.q."))
    q_bytes = corrupted_warm[q_name].view(torch.uint8).reshape(-1)
    q_bytes[0] ^= 1
    warm_q_corruption_detected = not _warm_outputs_equal(warm_outputs, corrupted_warm)["equal"]
    # A disposable, already-warmed target crosses the same resident restore
    # seam with invalidation deliberately suppressed.  Compare its stale
    # cached advisory result with the result after restoring normal invalidation
    # to prove this fixture would detect the broken-cache control.
    control, _wrappers = fixtures.build_case(
        checkpoint, roster, sim.E, device, smoke=smoke, canary=canary
    )
    control_sim, control_policy = control[:2]
    _tick(control)
    # Force a deterministic geometry distinction before caching.  The source
    # will restore its original body to this warmed target; without the reset
    # invalidation its advisory occupancy remains the deliberately displaced
    # body, making the broken-control result observable on every platform.
    _set_head(control_sim, 0, 0, 0, 0)
    r2_cache.bump(control_sim)
    control_sim.prepare()
    _control_q, _control_legal, control_hero = control_policy.evaluate(control_sim)
    control_policy.actions(control_sim, control_hero)
    control_cache_warm = bool(getattr(r2_cache, "_entry")(control_sim).cache)
    with patch.object(r2_cache, "bump", lambda _target: None):
        ruler._reset(control_sim, control_policy, sim, policy)
    stale_mask = control_sim.advisory_mask().clone()
    r2_cache.bump(control_sim)
    repaired_mask = control_sim.advisory_mask()
    broken_invalidation_control = bool(
        control_cache_warm and not torch.equal(stale_mask, repaired_mask)
    )
    # The warm clone/remap/restore proofs above deliberately observe a
    # prepared vector seam.  Advance each actual comparison branch once using
    # its ordinary complete lifecycle, retain those boundary captures, then
    # collect the registered two event ticks from the canonical post-step
    # unprepared seam.  This keeps the warm-boundary evidence intact while
    # ensuring every published event tick includes advisory/accounting fields.
    prepared_ticks["source"] = _prepared_outputs(_tick(case))
    prepared_ticks["clone"] = _prepared_outputs(_tick(clone))
    prepared_ticks["fresh"] = _prepared_outputs(_tick(equivalent))
    prepared_checks = {
        name: _warm_outputs_equal(_prepared_outputs(warm_outputs), output)
        for name, output in prepared_ticks.items()
    }
    if sim._prepared or policy.vector_prepared:
        raise RuntimeError("stale-cache source setup did not return to post-step seam")
    if clone[0]._prepared or clone[1].vector_prepared:
        raise RuntimeError("stale-cache clone setup did not return to post-step seam")
    if equivalent_sim._prepared or equivalent_policy.vector_prepared:
        raise RuntimeError("stale-cache fresh setup did not return to post-step seam")
    _TAPES[id(sim)] = []
    _TAPES[id(clone[0])] = []
    _TAPES[id(equivalent[0])] = []
    # The warm original, cold clone, and independently fresh equivalent all
    # progress twice through ordinary lifecycle invalidation.
    _tick(case)
    _tick(clone)
    _tick(equivalent)
    _tick(case)
    _tick(clone)
    _tick(equivalent)
    source_clone_compare = compared(_snapshot(case), _snapshot(clone))
    clone_fresh_compare = compared(_snapshot(clone), _snapshot(equivalent))
    tick_compares: list[Any] = []
    for index in range(2):
        tick_compares.extend(
            (
                compared(_TAPES[id(sim)][index], _TAPES[id(clone[0])][index]),
                compared(_TAPES[id(sim)][index], _TAPES[id(equivalent[0])][index]),
            )
        )
    all_compares = (
        remap_compare,
        remap_replay_compare,
        restore_compare,
        restore_replay_compare,
        source_clone_compare,
        clone_fresh_compare,
        *tick_compares,
    )
    schema_incompatibility = _first_schema(*all_compares)
    mismatch_result = next(
        (value for value in all_compares if isinstance(value, compare.Mismatch)), None
    )
    mismatch = sealed_mismatches[0] if mismatch_result is not None else None
    _TAPES.pop(id(clone[0]), None)
    _TAPES.pop(id(equivalent[0]), None)
    _TAPES.pop(id(control_sim), None)
    clone_stats, fresh_stats = r2_cache.stats(clone[0]), r2_cache.stats(equivalent_sim)
    lifecycle = (
        warm_stats.get("misses", 0) > 0
        and clone_stats.get("bumps", 0) > 0
        and fresh_stats.get("misses", 0) > 0
    )
    return (
        mismatch is None
        and lifecycle
        and restore_exact
        and restore_replay_exact
        and remap_exact
        and remap_replay_exact
        and broken_invalidation_control
        and source_clone_warm_exact["equal"]
        and source_remap_warm_exact["equal"]
        and source_restore_warm_exact["equal"]
        and source_fresh_warm_exact["equal"]
        and all(check["equal"] for check in prepared_checks.values())
        and warm_q_corruption_detected
        and target_body_progressed
        and target_food_progressed
        and schema_incompatibility is None
    ), {
        "warm_stats": warm_stats,
        "clone_stats": clone_stats,
        "fresh_stats": fresh_stats,
        "warm_generation": warm_generation,
        "clone_generation": clone_generation,
        "fresh_generation": fresh_generation,
        "warm_digest": warm_digest,
        "warm_vector_prepared": True,
        "clone_vector_prepared": clone_vector_prepared,
        "fresh_vector_prepared": fresh_vector_prepared,
        "restore_exact": restore_exact,
        "restore_replay_exact": restore_replay_exact,
        "restored_digest": restored_digest,
        "restore_branch_digest": restore_branch_digest,
        "restore_replay_digest": restore_replay_digest,
        "remap_exact": remap_exact,
        "remap_replay_exact": remap_replay_exact,
        "remap_source_digest": warm_digest,
        "remap_restored_digest": remap_restored_digest,
        "remap_branch_digest": remap_branch_digest,
        "remap_replay_digest": remap_replay_digest,
        "mismatch": mismatch,
        "schema_incompatibility": schema_incompatibility,
        "lifecycle": lifecycle,
        "broken_invalidation_control": broken_invalidation_control,
        "warm_output_comparisons": {
            "source_clone": source_clone_warm_exact,
            "source_remap": source_remap_warm_exact,
            "source_restore": source_restore_warm_exact,
            "source_fresh": source_fresh_warm_exact,
        },
        "prepared_tick_captures": {
            name: compare.compact_inventory(output) for name, output in prepared_ticks.items()
        },
        "prepared_tick_comparisons": prepared_checks,
        "evidence_errors": [
            {"check": name, **check}
            for name, check in {
                "source_clone": source_clone_warm_exact,
                "source_remap": source_remap_warm_exact,
                "source_restore": source_restore_warm_exact,
                "source_fresh": source_fresh_warm_exact,
                **{"prepared_" + name: check for name, check in prepared_checks.items()},
            }.items()
            if not check["equal"]
        ],
        "warm_q_corruption_detected": warm_q_corruption_detected,
        "warm_q_corruption_control": {
            "field": q_name,
            "source_descriptor": compare.compact_inventory({q_name: warm_outputs[q_name]})[q_name],
            "corrupted_descriptor": compare.compact_inventory({q_name: corrupted_warm[q_name]})[
                q_name
            ],
        },
        "restore_target_body_progressed": target_body_progressed,
        "restore_target_food_progressed": target_food_progressed,
        "warm_output_captures": {
            "source": compare.compact_inventory(warm_outputs),
            "clone": compare.compact_inventory(clone_warm_outputs),
            "remap": compare.compact_inventory(remap_warm_outputs),
            "restore": compare.compact_inventory(restored_warm_outputs),
            "fresh": compare.compact_inventory(fresh_warm_outputs),
            "corrupted_q_control": compare.compact_inventory(corrupted_warm),
        },
        "mask_shape": list(stale_mask.shape),
        "mask_dtype": str(stale_mask.dtype),
        "stale_mask_hex": _tensor_hex(stale_mask),
        "repaired_mask_hex": _tensor_hex(repaired_mask),
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


def _valid_tensor_tape(value: Any) -> bool:
    """Return whether a capture tape is usable for a byte comparison.

    An absent E4 reference is an execution/evidence failure, not an engine
    mismatch.  Do not hand an empty or placeholder tape to the comparator:
    only two nonempty dictionaries of actual tensors can establish a
    self-consistency byte difference.
    """
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(tick, dict)
            and bool(tick)
            and all(isinstance(tensor, torch.Tensor) for tensor in tick.values())
            for tick in value
        )
    )


def _valid_tensor_prefix(value: Any) -> list[dict[str, torch.Tensor]]:
    """Return a nonempty valid tensor prefix, or no executable evidence."""
    if not isinstance(value, list):
        return []
    prefix: list[dict[str, torch.Tensor]] = []
    for tick in value:
        if (
            not isinstance(tick, dict)
            or not tick
            or not all(isinstance(tensor, torch.Tensor) for tensor in tick.values())
        ):
            break
        prefix.append(tick)
    return prefix


def _byte_comparable_tapes(
    left: list[dict[str, torch.Tensor]], right: list[dict[str, torch.Tensor]]
) -> bool:
    """Require identical tensor inventory before interpreting a byte delta."""
    return all(
        set(expected) == set(observed)
        and all(
            expected[name].dtype == observed[name].dtype
            and tuple(expected[name].shape) == tuple(observed[name].shape)
            for name in expected
        )
        for expected, observed in zip(left, right)
    )


def _reference_incomplete(
    witness: dict[str, Any], raw_tape: Any, reason: str
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Mark a cell incomplete without manufacturing a mismatch witness."""
    witness["reference_status"] = reason
    return False, None, "INCOMPLETE"


def _event_comparison_record(witness: dict[str, Any], worlds: int) -> dict[str, Any]:
    """Expose sealed event capture descriptors to the common witness validator."""
    return {
        "worlds": worlds,
        "layout": "event",
        "replay": 0,
        "field_digests": witness.get("mapped_tape"),
        "action_q_tape": witness.get("action_q_tape"),
    }


def _event_byte_witness(
    found: compare.Mismatch,
    tick: int,
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    reference_witness: dict[str, Any],
    candidate_witness: dict[str, Any],
    reference_worlds: int,
    candidate_worlds: int,
) -> dict[str, Any]:
    """Bind a genuine cross-E byte delta to both sealed event captures."""
    field = found.field
    return {
        "tick": tick,
        **found.__dict__,
        "expected_descriptor": compare.compact_inventory({field: reference[field]})[field],
        "actual_descriptor": compare.compact_inventory({field: candidate[field]})[field],
        "comparison": {
            "reference": {"worlds": reference_worlds, "layout": "event", "replay": 0, "tick": tick},
            "candidate": {"worlds": candidate_worlds, "layout": "event", "replay": 0, "tick": tick},
        },
        "comparison_records": [
            _event_comparison_record(reference_witness, reference_worlds),
            _event_comparison_record(candidate_witness, candidate_worlds),
        ],
    }


def _internal_byte_witness(
    found: compare.Mismatch,
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    physical_worlds: int,
) -> dict[str, Any]:
    """Bind an internal persistent-state delta to its exact snapshot pair."""
    field = found.field
    if left[field].ndim == 0 or left[field].shape[0] != physical_worlds:
        raise ValueError("internal mismatch has no physical world axis")
    row_bytes = left[field].numel() * left[field].element_size() // physical_worlds
    offending_row = found.byte_offset // row_bytes
    rows = [offending_row] + [row for row in range(physical_worlds) if row != offending_row][:3]
    index = torch.tensor(rows, dtype=torch.long, device=left[field].device)

    def canonical(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: (
                value.index_select(0, index).detach().clone()
                if value.ndim and value.shape[0] == physical_worlds
                else value.detach().clone()
            )
            for name, value in values.items()
        }

    canonical_left, canonical_right = canonical(left), canonical(right)
    canonical_found = compare.compare_inventory(canonical_left, canonical_right)
    if not isinstance(canonical_found, compare.Mismatch):
        raise ValueError("canonical internal mismatch did not retain the offending byte")
    field = canonical_found.field

    def raw_action_q(values: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        """Retain exact policy bytes when an internal replay compares outputs."""

        def raw(value: torch.Tensor) -> str:
            return value.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes().hex()

        if "output.actions" not in values and not any(
            name.startswith("output.q.") for name in values
        ):
            return []
        result = {
            "q_hex": {
                name.removeprefix("output.q."): raw(value)
                for name, value in values.items()
                if name.startswith("output.q.")
            }
        }
        if "output.actions" in values:
            result["actions_hex"] = raw(values["output.actions"])
        return [result]

    reference = {
        "worlds": 4,
        "layout": "event-internal",
        "replay": 0,
        "field_digests": [compare.compact_inventory(canonical_left)],
        "action_q_tape": raw_action_q(canonical_left),
    }
    candidate = {
        "worlds": 4,
        "layout": "event-internal",
        "replay": 1,
        "field_digests": [compare.compact_inventory(canonical_right)],
        "action_q_tape": raw_action_q(canonical_right),
    }
    return {
        "tick": 0,
        **canonical_found.__dict__,
        "expected_descriptor": reference["field_digests"][0][field],
        "actual_descriptor": candidate["field_digests"][0][field],
        "comparison": {
            "reference": {"worlds": 4, "layout": "event-internal", "replay": 0, "tick": 0},
            "candidate": {"worlds": 4, "layout": "event-internal", "replay": 1, "tick": 0},
        },
        "comparison_records": [reference, candidate],
        "physical_worlds": physical_worlds,
        "physical_rows": rows,
    }


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
    references: dict[str, tuple[list[dict[str, torch.Tensor]], dict[str, Any]]] = {}
    for worlds in sizes:
        for event in EVENTS:
            if time.monotonic() >= deadline:
                evidence.append({"event": event, "worlds": worlds, "ticks": 0, "status": "timeout"})
                continue
            try:
                passed, witness = _run_one(event, checkpoint, roster, worlds, device, lane)
                raw_tape = witness.pop("_raw_mapped_tape", [])
                mismatch: dict[str, Any] | None = witness.get("mismatch")
                internal_mismatch = mismatch
                fixture_contradiction = (
                    not passed and mismatch is None and _valid_tensor_tape(raw_tape)
                )
                forced_status: str | None = None
                if witness.get("schema_incompatibility") is not None:
                    mismatch = None
                    forced_status = "INVALID_EVIDENCE"
                if worlds == 4:
                    # E4 is the canonical reference.  An incomplete capture
                    # is retained as evidence but cannot become an empty
                    # reference that later falsely accuses E64/E1024.
                    reference_prefix = _valid_tensor_prefix(raw_tape)
                    if reference_prefix:
                        references[event] = (reference_prefix, witness)
                    if _valid_tensor_tape(raw_tape):
                        pass
                    else:
                        passed, mismatch, forced_status = _reference_incomplete(
                            witness, raw_tape, "canonical E4 tape absent or incomplete"
                        )
                else:
                    reference_entry = references.get(event)
                    reference, reference_witness = (
                        reference_entry if reference_entry is not None else ([], {})
                    )
                    reference_prefix = _valid_tensor_prefix(reference)
                    observed_prefix = _valid_tensor_prefix(raw_tape)
                    if not reference_prefix:
                        passed, mismatch, forced_status = _reference_incomplete(
                            witness, raw_tape, "canonical E4 reference absent or incomplete"
                        )
                    elif not observed_prefix:
                        passed, mismatch, forced_status = _reference_incomplete(
                            witness, raw_tape, "event tape absent or incomplete"
                        )
                    else:
                        prefix_len = min(len(reference_prefix), len(observed_prefix))
                        left_prefix, right_prefix = (
                            reference_prefix[:prefix_len],
                            observed_prefix[:prefix_len],
                        )
                        if not _byte_comparable_tapes(left_prefix, right_prefix):
                            passed, mismatch, forced_status = _reference_incomplete(
                                witness,
                                raw_tape,
                                "event tape inventory/dtype/shape differs from E4",
                            )
                        else:
                            for tick, (left, right) in enumerate(zip(left_prefix, right_prefix)):
                                found = compare.compare_inventory(left, right)
                                if isinstance(found, compare.SchemaIncompatibility):
                                    mismatch = None
                                    witness["schema_incompatibility"] = found.__dict__.copy()
                                    forced_status = "INVALID_EVIDENCE"
                                    break
                                if isinstance(found, compare.Mismatch):
                                    mismatch = _event_byte_witness(
                                        found,
                                        tick,
                                        left,
                                        right,
                                        reference_witness,
                                        witness,
                                        4,
                                        worlds,
                                    )
                                    break
                        if mismatch is None and (
                            not _valid_tensor_tape(reference) or not _valid_tensor_tape(raw_tape)
                        ):
                            passed, mismatch, forced_status = _reference_incomplete(
                                witness, raw_tape, "event tape absent or incomplete"
                            )
                # A sealed internal unequal-byte witness does not depend on
                # the availability of another E cell. Preserve its precedence.
                if internal_mismatch is not None and witness.get("schema_incompatibility") is None:
                    mismatch, forced_status = internal_mismatch, None
                if mismatch is None and (witness.get("evidence_errors") or fixture_contradiction):
                    forced_status = "INVALID_EVIDENCE"
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
                            forced_status
                            or (
                                "PASS"
                                if passed
                                else (
                                    "INVALID_SELF_CONSISTENCY"
                                    if mismatch is not None
                                    else "INCOMPLETE"
                                )
                            )
                        ),
                        "witness_sha256": hashlib.sha256(encoded).hexdigest(),
                        "witness": witness,
                    }
                )
                result[event] = bool(result[event] or passed)
                # Only a byte comparison between two valid tensor tapes may
                # terminate the event block as an engine mismatch.  Fixture
                # exceptions and incomplete E4 references remain coverage
                # failures and E64/E1024 still run to completion.
                if mismatch is not None and forced_status is None:
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
