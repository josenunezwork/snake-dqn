"""Frozen Mac-only T4A reducer; remote summaries cannot grant a premise PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import t4a_units as units

LAYOUTS = ("canonical", "reversed", "interspersed")
EVENTS = (
    "ring_wrap",
    "simultaneous_death_respawn",
    "contested_food_refill",
    "clone_restore",
    "stale_cache",
)
CONTROLS = ("q", "state", "cursor", "row_map")
SEEDS = {"main": 9402401, "canary": 9402411, "smoke": 9402421}
SIM_FIELDS = (
    "bodies head_ptr seg_count length alive direction boost_frames frames_since_food "
    "respawn_timer _reward_prev_length frame rng_counter rng_key food_cells food_present "
    "corpse_food food_count _last_reward _last_done _last_death_cause _last_kills "
    "_last_food_ate _last_ambient_food_ate _last_corpse_food_ate _last_transition_valid "
    "_boosted_this_step terminal_slots capacity_overflow"
).split()
RUNTIME_FIELDS = (
    "controlled prev_dist prev_id _carry_state _carry_mask _carry_valid _alive_post "
    "sel_state sel_mask sel_ready sel_fresh"
).split()
MANDATORY_FIELDS = frozenset(
    [f"sim.{f}" for f in SIM_FIELDS]
    + [f"runtime.{f}" for f in RUNTIME_FIELDS]
    + [f"collector.{f}" for f in ("length", "alive", "action", "qmax")]
)


def required_fields(mix: str) -> frozenset[str]:
    slots = {"scripted": (0,), "mixed": (0, 1, 3, 5), "frozen": tuple(range(6))}[mix]
    v8_fields = (
        "mask action base v5_action v7_action outcome v5_reason counts cap need "
        "area_cap area landing head_risky head_waived v5_eligible"
    ).split()
    return MANDATORY_FIELDS | frozenset(
        [
            "output.actions",
            "accounting.reconciled",
            "lifecycle.sim_prepared",
            "lifecycle.vector_prepared",
            "configuration.sha256",
            "output.obs.0.1",
            "output.obs.0.2",
        ]
        + [f"output.q.{s}" for s in slots]
        + [f"output.obs.{s}.0" for s in slots]
        + [f"output.v8.{i}.{name}" for i in range(len(slots)) for name in v8_fields]
        + [f"output.advisory.{i}" for i in range(3 + 6 - len(slots))]
    )


def _json(path: Path) -> dict[str, Any]:
    units._verify(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("T4A evidence must be an object")
    return value


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_bool_hex(value: Any, byte_count: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 2 * byte_count
        and all(value[index : index + 2] in {"00", "01"} for index in range(0, len(value), 2))
    )


def _valid_descriptor(field: str, descriptor: Any) -> bool:
    """Validate the exported descriptor rather than merely its field name.

    ``sim.bodies`` deliberately uses a compact device-side fingerprint.  Every
    other field remains a raw-byte SHA-256 commitment, which is required for
    the action/Q binding below.
    """
    valid_dtypes = {
        "torch.bool",
        "torch.float16",
        "torch.float32",
        "torch.float64",
        "torch.int8",
        "torch.int16",
        "torch.int32",
        "torch.int64",
        "torch.uint8",
    }
    if (
        not isinstance(descriptor, dict)
        or not isinstance(descriptor.get("shape"), list)
        or descriptor.get("dtype") not in valid_dtypes
        or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
            for dimension in descriptor["shape"]
        )
    ):
        return False
    shape, dtype = descriptor["shape"], descriptor["dtype"]
    if field in {"lifecycle.sim_prepared", "lifecycle.vector_prepared"}:
        return (
            dtype == "torch.bool"
            and shape == []
            and set(descriptor) == {"dtype", "shape", "sha256"}
            and _is_digest(descriptor.get("sha256"))
        )
    if field == "configuration.sha256":
        return dtype == "torch.uint8" and shape == [32] and _is_digest(descriptor.get("sha256"))
    if not shape or shape[0] != 4 or any(dimension == 0 for dimension in shape):
        return False
    if field != "sim.bodies" and set(descriptor) != {"dtype", "shape", "sha256"}:
        return False
    if field == "sim.bodies":
        return (
            len(descriptor["shape"]) == 4
            and descriptor["shape"][1] == 6
            and descriptor["shape"][2] > 0
            and descriptor["shape"][3] == 2
            and descriptor.get("dtype") in {"torch.int32", "torch.int64"}
            and descriptor.get("digest_algorithm") == "t4a-device-mix64x4-v1"
            and _is_digest(descriptor.get("digest"))
            and set(descriptor) == {"dtype", "shape", "digest_algorithm", "digest"}
        )
    sim_int_slots = {
        "head_ptr",
        "seg_count",
        "length",
        "direction",
        "boost_frames",
        "frames_since_food",
        "respawn_timer",
        "_reward_prev_length",
        "_last_death_cause",
        "_last_kills",
    }
    sim_bool_slots = {
        "alive",
        "terminal_slots",
        "capacity_overflow",
        "_last_done",
        "_last_food_ate",
        "_last_ambient_food_ate",
        "_last_corpse_food_ate",
        "_last_transition_valid",
        "_boosted_this_step",
    }
    if field.startswith("sim."):
        name = field.removeprefix("sim.")
        if name in sim_int_slots:
            return (
                dtype == "torch.int64" and shape == [4, 6] and _is_digest(descriptor.get("sha256"))
            )
        if name in sim_bool_slots:
            return (
                dtype == "torch.bool" and shape == [4, 6] and _is_digest(descriptor.get("sha256"))
            )
        if name in {"frame", "rng_counter", "rng_key", "food_count"}:
            return dtype == "torch.int64" and shape == [4] and _is_digest(descriptor.get("sha256"))
        if name == "_last_reward":
            return (
                dtype == "torch.float32"
                and shape == [4, 6]
                and _is_digest(descriptor.get("sha256"))
            )
        if name == "food_cells":
            return (
                dtype == "torch.int64"
                and len(shape) == 3
                and shape[2] == 2
                and _is_digest(descriptor.get("sha256"))
            )
        if name in {"food_present", "corpse_food"}:
            return (
                dtype == "torch.bool" and len(shape) == 2 and _is_digest(descriptor.get("sha256"))
            )
        return False
    if field.startswith("runtime."):
        name = field.removeprefix("runtime.")
        if name in {"controlled", "_carry_valid", "_alive_post", "sel_ready", "sel_fresh"}:
            return (
                dtype == "torch.bool" and shape == [4, 6] and _is_digest(descriptor.get("sha256"))
            )
        if name == "prev_dist":
            return (
                dtype == "torch.float64"
                and shape == [4, 6]
                and _is_digest(descriptor.get("sha256"))
            )
        if name == "prev_id":
            return (
                dtype == "torch.int64" and shape == [4, 6] and _is_digest(descriptor.get("sha256"))
            )
        if name in {"_carry_state", "sel_state"}:
            return (
                dtype == "torch.float32"
                and len(shape) == 3
                and shape[1] == 6
                and _is_digest(descriptor.get("sha256"))
            )
        if name in {"_carry_mask", "sel_mask"}:
            return (
                dtype == "torch.bool"
                and shape == [4, 6, 6]
                and _is_digest(descriptor.get("sha256"))
            )
        return False
    if field.startswith("collector."):
        return {
            "collector.length": ("torch.int64", [4]),
            "collector.alive": ("torch.bool", [4]),
            "collector.action": ("torch.int64", [4]),
            "collector.qmax": ("torch.float32", [4]),
        }.get(field) == (dtype, shape) and _is_digest(descriptor.get("sha256"))
    if field == "output.actions" or field == "event.supplied_actions":
        return dtype == "torch.int64" and shape == [4, 6] and _is_digest(descriptor.get("sha256"))
    if field.startswith("output.q."):
        return dtype == "torch.float32" and shape == [4, 6] and _is_digest(descriptor.get("sha256"))
    if field.startswith("output.obs."):
        _output, _obs, slot, index = field.split(".")
        expected = {
            ("0", "0"): ("torch.uint8", [4, 7, 31, 31]),
            ("0", "1"): ("torch.uint8", [4, 4, 37, 37]),
            ("0", "2"): ("torch.float32", [4, 60]),
        }.get((slot, index))
        if expected is not None:
            return (dtype, shape) == expected and _is_digest(descriptor.get("sha256"))
        return (dtype, shape) == ("torch.float32", [4, 61]) and _is_digest(descriptor.get("sha256"))
    if field.startswith("output.advisory."):
        return (dtype, shape) == ("torch.bool", [4, 6, 6]) and _is_digest(descriptor.get("sha256"))
    if field.startswith("output.v8."):
        name = field.rsplit(".", 1)[-1]
        v8_schema = {
            "mask": ("torch.bool", [4, 6]),
            "action": ("torch.int64", [4]),
            "base": ("torch.int64", [4]),
            "v5_action": ("torch.int64", [4]),
            "v7_action": ("torch.int64", [4]),
            "outcome": ("torch.int64", [4]),
            "v5_reason": ("torch.int64", [4]),
            "counts": ("torch.int64", [4, 3]),
            "cap": ("torch.int64", [4]),
            "need": ("torch.int64", [4]),
            "area_cap": ("torch.int64", [4]),
            "area": ("torch.int64", [4, 6]),
            "landing": ("torch.int64", [4, 3]),
            "head_risky": ("torch.bool", [4, 6]),
            "head_waived": ("torch.bool", [4, 6]),
            "v5_eligible": ("torch.bool", [4, 6]),
        }
        return (dtype, shape) == v8_schema.get(name) and _is_digest(descriptor.get("sha256"))
    return _is_digest(descriptor.get("sha256")) and set(descriptor) == {
        "dtype",
        "shape",
        "sha256",
    }


def _learned_slots(mix: str) -> set[str]:
    return {
        "scripted": {"0"},
        "mixed": {"0", "1", "3", "5"},
        "frozen": {str(slot) for slot in range(6)},
    }[mix]


def _expected_rows(worlds: int, layout: str) -> list[int]:
    if layout == "canonical":
        return [0, 1, 2, 3]
    if layout == "reversed":
        return [worlds - 1, worlds - 2, worlds - 3, worlds - 4]
    if layout == "interspersed":
        return [0, (worlds - 1) // 3, 2 * (worlds - 1) // 3, worlds - 1]
    raise ValueError("unknown frozen layout")


def _valid_map_proof(proof: Any, worlds: int, layout: str) -> bool:
    """Validate producer placement and extraction maps against an independent oracle."""
    expected = _expected_rows(worlds, layout)
    return (
        isinstance(proof, dict)
        and proof.get("layout") == layout
        and proof.get("worlds") == worlds
        and proof.get("logical_ids") == [0, 1, 2, 3]
        and proof.get("history_offsets") == [0, 3, 11, 23]
        and proof.get("insertion_rows") == expected
        and proof.get("expected_rows") == expected
        and proof.get("extraction_rows") == expected
        and proof.get("placement_valid") is True
        and proof.get("extraction_valid") is True
        and proof.get("placement_checks") == [True, True, True, True]
    )


def _event_predicate_error(event: str, witness: dict[str, Any], worlds: int) -> str | None:
    """Recompute the frozen fixture predicate from its exported values.

    Fixture code may report ``PASS`` only after these witnessed values prove the
    event.  This reducer intentionally does not trust a producer boolean.
    """
    values = witness
    if event == "ring_wrap":
        before, after = values.get("before"), values.get("after")
        expected_head, expected_tail = values.get("expected_head"), values.get("expected_tail")
        prefix = values.get("logical_prefix")
        if not (
            isinstance(before, list)
            and len(before) == worlds
            and all(isinstance(value, int) and value >= 1 for value in before)
            and len(set(before)) == 1
            and isinstance(after, list)
            and after == [0] * worlds
            and isinstance(expected_head, list)
            and len(expected_head) == 2
            and isinstance(expected_tail, list)
            and len(expected_tail) == 2
            and all(isinstance(value, int) for value in expected_head + expected_tail)
            and isinstance(prefix, list)
            and len(prefix) == worlds
            and all(item == [expected_head, expected_tail] for item in prefix)
            and values.get("wrapped") is True
            and values.get("ordered") is True
        ):
            return "ring_wrap witnessed values do not prove wrap/order"
    elif event == "simultaneous_death_respawn":
        if not all(
            isinstance(values.get(name), list)
            and len(values[name]) == worlds
            and all(value is True for value in values[name])
            for name in ("died", "respawned", "corpse")
        ):
            return "death/respawn witnessed values do not prove event"
    elif event == "contested_food_refill":
        first, second = values.get("slot_1_ate"), values.get("slot_2_ate")
        refilled, rng_before, rng_after = (
            values.get("food_after_first"),
            values.get("rng_before"),
            values.get("rng_after"),
        )
        if not (
            isinstance(first, list)
            and len(first) == worlds
            and all(value is True for value in first)
            and isinstance(second, list)
            and len(second) == worlds
            and all(value is False for value in second)
            and isinstance(refilled, list)
            and len(refilled) == worlds
            and all(isinstance(value, int) and value > 0 for value in refilled)
            and isinstance(rng_before, list)
            and isinstance(rng_after, list)
            and len(rng_before) == worlds == len(rng_after)
            and all(
                isinstance(left, int) and isinstance(right, int) and right > left
                for left, right in zip(rng_before, rng_after)
            )
        ):
            return "contested-food witnessed values do not prove ordered refill"
    elif event == "clone_restore":
        if not (
            values.get("restored_exact") is True
            and values.get("mismatch") is None
            and _is_digest(values.get("saved_digest"))
            and values.get("saved_digest") == values.get("restored_digest")
            and _is_digest(values.get("branch_digest"))
            and values.get("branch_digest") == values.get("replay_digest")
            and _is_digest(values.get("before_restore_digest"))
            and values["before_restore_digest"] != values["saved_digest"]
        ):
            return "clone/restore witnessed values do not prove exact restore"
    elif event == "stale_cache":
        stats = (values.get("warm_stats"), values.get("clone_stats"), values.get("fresh_stats"))
        digests = (
            "warm_digest",
            "restored_digest",
            "restore_branch_digest",
            "restore_replay_digest",
            "remap_source_digest",
            "remap_restored_digest",
            "remap_branch_digest",
            "remap_replay_digest",
        )
        mask_shape = values.get("mask_shape")
        stale_mask, repaired_mask = values.get("stale_mask_hex"), values.get("repaired_mask_hex")
        if not (
            all(
                isinstance(stats_map, dict)
                and bool(stats_map)
                and all(
                    isinstance(key, str) and isinstance(value, int) and value >= 0
                    for key, value in stats_map.items()
                )
                for stats_map in stats
            )
            and values.get("warm_vector_prepared") is True
            and values.get("clone_vector_prepared") is True
            and values.get("fresh_vector_prepared") is True
            and values.get("restore_exact") is True
            and values.get("restore_replay_exact") is True
            and values.get("remap_exact") is True
            and values.get("remap_replay_exact") is True
            and values.get("lifecycle") is True
            and values.get("broken_invalidation_control") is True
            and values.get("mismatch") is None
            and values["warm_stats"].get("misses", 0) > 0
            and values["clone_stats"].get("bumps", 0) > 0
            and values["fresh_stats"].get("misses", 0) > 0
            and all(
                isinstance(values.get(name), int) and values[name] >= 0
                for name in ("warm_generation", "clone_generation", "fresh_generation")
            )
            and all(_is_digest(values.get(name)) for name in digests)
            and values["restored_digest"] == values["warm_digest"]
            and values["restore_branch_digest"] == values["restore_replay_digest"]
            and values["remap_source_digest"]
            == values["remap_restored_digest"]
            == values["warm_digest"]
            and values["remap_branch_digest"] == values["remap_replay_digest"]
            and values.get("mask_dtype") == "torch.bool"
            and mask_shape == [worlds, 6, 6]
            and _is_bool_hex(stale_mask, worlds * 6 * 6)
            and _is_bool_hex(repaired_mask, worlds * 6 * 6)
            and stale_mask != repaired_mask
        ):
            return "stale-cache witnessed values do not prove lifecycle/invalidation"
    else:  # Defensive: EVENTS is frozen, so a new event needs a reducer schema.
        return "unknown event predicate schema"
    return None


def _event_tick_error(tick: Any, descriptors: Any, mix: str, event_fields: set[str]) -> str | None:
    if not isinstance(tick, dict) or not isinstance(descriptors, dict):
        return "raw tick or descriptor map malformed"
    if set(descriptors) != event_fields or any(
        not _valid_descriptor(name, descriptor) for name, descriptor in descriptors.items()
    ):
        return "typed event descriptor map malformed"
    try:
        supplied = bytes.fromhex(tick["actions_hex"])
        policy = bytes.fromhex(tick["policy_actions_hex"])
        if len(supplied) != 4 * 6 * 8 or len(policy) != 4 * 6 * 8:
            raise ValueError("action bytes have wrong length")
        if any(action not in range(6) for action in struct.unpack("<24q", supplied)):
            raise ValueError("supplied action outside action space")
        if any(action not in range(6) for action in struct.unpack("<24q", policy)):
            raise ValueError("policy action outside action space")
        for field, raw in (("event.supplied_actions", supplied), ("output.actions", policy)):
            descriptor = descriptors[field]
            if (
                descriptor.get("shape") != [4, 6]
                or descriptor.get("dtype") != "torch.int64"
                or descriptor.get("sha256") != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError(f"{field} raw bytes are not bound")
        q_hex = tick["q_hex"]
        if not isinstance(q_hex, dict) or set(q_hex) != _learned_slots(mix):
            raise ValueError("learned-seat Q inventory differs")
        for slot, encoded in q_hex.items():
            raw = bytes.fromhex(encoded)
            if len(raw) != 4 * 6 * 4 or not all(
                math.isfinite(value) for value in struct.unpack("<24f", raw)
            ):
                raise ValueError("Q bytes are malformed or nonfinite")
            descriptor = descriptors[f"output.q.{slot}"]
            if (
                descriptor.get("shape") != [4, 6]
                or descriptor.get("dtype") != "torch.float32"
                or descriptor.get("sha256") != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError("Q bytes are not bound")
    except (KeyError, TypeError, ValueError, struct.error) as exc:
        return str(exc)
    return None


def _content_errors(content: dict, mix: str, lane: str) -> tuple[list[str], list[str], list[Any]]:
    invalid: list[str] = []
    missing: list[str] = []
    mismatches: list[Any] = []
    horizon = {"main": 256, "canary": 2, "smoke": 1}[lane]
    if (
        content.get("roster"),
        content.get("mode"),
        content.get("horizon"),
        content.get("root_seed"),
    ) != (mix, lane, horizon, SEEDS[lane] + units.MIXES.index(mix)):
        invalid.append(f"{mix}: root/mode/horizon differs")
    controls = content.get("controls", {})
    if not isinstance(controls, dict) or any(controls.get(k) is not True for k in CONTROLS):
        invalid.append(f"{mix}: negative controls failed/missing")
    witnesses = content.get("control_witnesses", {})
    for key, field in {
        "q": "output.q.0",
        "state": "sim.bodies",
        "cursor": "sim.rng_counter",
        "row_map": "output.q.0",
    }.items():
        witness = witnesses.get(key, {}) if isinstance(witnesses, dict) else {}
        allowed_fields = required_fields(mix) if key == "row_map" else {field}
        if (
            witness.get("field") not in allowed_fields
            or not isinstance(witness.get("byte_offset"), int)
            or witness["byte_offset"] < 0
            or witness.get("expected") == witness.get("actual")
            or not all(
                isinstance(witness.get(n), int) and 0 <= witness[n] <= 255
                for n in ("expected", "actual")
            )
        ):
            invalid.append(f"{mix}: invalid comparison control witness {key}")
    fields = content.get("fields")
    if (
        not isinstance(fields, list)
        or not fields
        or len(set(fields)) != len(fields)
        or any(not isinstance(x, str) or not x for x in fields)
    ):
        invalid.append(f"{mix}: field inventory missing or duplicate")
    else:
        mandatory = required_fields(mix)
        if not mandatory.issubset(fields):
            invalid.append(f"{mix}: mandatory fields omitted: {sorted(mandatory - set(fields))}")
        digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
        if content.get("field_inventory_sha256") != digest:
            invalid.append(f"{mix}: field inventory binding differs")
    sizes = (4,) if lane == "smoke" else (4, 64, 1024)
    expected = {(e, layout, r) for e in sizes for layout in LAYOUTS for r in (0, 1)}
    seen: set[tuple] = set()
    tapes: list[list] = []
    digests: list[list] = []
    records = content.get("records")
    if not isinstance(records, list):
        return invalid + [f"{mix}: malformed replay records"], missing, mismatches
    for row in records:
        if not isinstance(row, dict):
            invalid.append(f"{mix}: malformed replay")
            continue
        key = (row.get("worlds"), row.get("layout"), row.get("replay"))
        if key not in expected or key in seen:
            invalid.append(f"{mix}: foreign/duplicate replay {key}")
            continue
        seen.add(key)
        if row.get("ticks") != horizon or row.get("matched_ticks") != horizon:
            missing.append(f"{mix}: incomplete replay {key}")
        expected_calls = horizon * {"frozen": 6, "mixed": 4, "scripted": 1}[mix]
        if (
            not isinstance(row.get("comparison_count"), int)
            or row["comparison_count"] != horizon * len(fields or [])
            or row.get("forward_rows") != 1024
            or not isinstance(row.get("forward_calls"), int)
            or row["forward_calls"] != expected_calls
            or row.get("field_inventory_sha256") != content.get("field_inventory_sha256")
        ):
            invalid.append(f"{mix}: comparison/forward/inventory omitted in {key}")
        if not _valid_map_proof(row.get("map_proof"), row["worlds"], row["layout"]):
            invalid.append(f"{mix}: independent insertion/extraction map proof invalid in {key}")
        elif row.get("extracted_rows") != _expected_rows(row["worlds"], row["layout"]):
            invalid.append(f"{mix}: actual extracted row order differs from frozen layout in {key}")
        shapes = row.get("forward_shapes")
        if not isinstance(shapes, list) or len(shapes) != expected_calls:
            invalid.append(f"{mix}: missing physical forward shape proof in {key}")
        else:
            for call in shapes:
                try:
                    padded, output = call["padded_shapes"], call["output_shape"]
                    if (
                        not padded
                        or any(s[0] != 1024 for s in padded)
                        or output != [1024, 6]
                        or call["real_rows"] != row["worlds"]
                    ):
                        raise ValueError("native forward shape differs")
                except (KeyError, TypeError, IndexError, ValueError):
                    invalid.append(f"{mix}: variable-shaped learned forward in {key}")
                    break
        tape = row.get("action_q_tape")
        field_digests = row.get("field_digests")
        if not isinstance(field_digests, list) or len(field_digests) != horizon:
            invalid.append(f"{mix}: per-field/tick evidence omitted in {key}")
        else:
            digests.append(field_digests)
            for tick_fields in field_digests:
                if not isinstance(tick_fields, dict) or set(tick_fields) != set(fields or []):
                    invalid.append(f"{mix}: per-tick field inventory differs in {key}")
                    break
                for field, descriptor in tick_fields.items():
                    # Matrix descriptors are raw-byte commitments except for
                    # the explicitly compacted body-ring fingerprint.
                    if not _valid_descriptor(field, descriptor):
                        invalid.append(f"{mix}: malformed typed field digest in {key}")
                        break
        if not isinstance(tape, list) or len(tape) != horizon:
            invalid.append(f"{mix}: missing exact action/Q tape in {key}")
        else:
            tapes.append(tape)
            for tick_index, tick in enumerate(tape):
                try:
                    if not isinstance(tick, dict) or not isinstance(tick.get("q_hex"), dict):
                        raise ValueError("tape tick is not an exact typed action/Q object")
                    actions = bytes.fromhex(tick["actions_hex"])
                    if len(actions) != 4 * 6 * 8:
                        raise ValueError("action tape must contain all four worlds and six seats")
                    if any(a not in range(6) for a in struct.unpack("<24q", actions)):
                        raise ValueError("action outside six-action space")
                    descriptors = (
                        field_digests[tick_index]
                        if isinstance(field_digests, list) and tick_index < len(field_digests)
                        else {}
                    )
                    if not isinstance(descriptors, dict):
                        raise ValueError("typed field tick container is not an object")
                    action_desc = descriptors.get("output.actions", {})
                    if (
                        not isinstance(action_desc, dict)
                        or action_desc.get("shape") != [4, 6]
                        or action_desc.get("dtype") != "torch.int64"
                        or action_desc.get("sha256") != hashlib.sha256(actions).hexdigest()
                    ):
                        raise ValueError("exact action tape is not bound to typed field digest")
                    slots = {
                        "scripted": {"0"},
                        "mixed": {"0", "1", "3", "5"},
                        "frozen": {str(s) for s in range(6)},
                    }[mix]
                    if set(tick["q_hex"]) != slots:
                        raise ValueError("learned-seat Q tape inventory differs")
                    if "0" not in tick["q_hex"]:
                        raise ValueError("hero Q tape omitted")
                    for slot, qhex in tick["q_hex"].items():
                        q = bytes.fromhex(qhex)
                        if len(q) != 4 * 6 * 4 or not all(
                            math.isfinite(v) for v in struct.unpack("<24f", q)
                        ):
                            raise ValueError("raw fp32 Q bytes malformed or nonfinite")
                        desc = descriptors.get(f"output.q.{slot}", {})
                        if (
                            not isinstance(desc, dict)
                            or desc.get("shape") != [4, 6]
                            or desc.get("dtype") != "torch.float32"
                            or desc.get("sha256") != hashlib.sha256(q).hexdigest()
                        ):
                            raise ValueError("exact Q tape is not bound to typed field digest")
                except (KeyError, ValueError, TypeError, struct.error) as exc:
                    invalid.append(f"{mix}: invalid exact action/Q tape in {key}: {exc}")
                    break
    if expected != seen:
        missing.append(f"{mix}: missing replays {sorted(expected - seen)}")
    if tapes and any(tape != tapes[0] for tape in tapes[1:]):
        mismatches.append(f"{mix}: exact action/Q tapes differ across configuration replays")
    if digests and any(tape != digests[0] for tape in digests[1:]):
        mismatches.append(f"{mix}: field/tick digests differ across configuration replays")
    coverage = content.get("coverage", {})
    if not isinstance(coverage, dict) or any(coverage.get(k) is not True for k in EVENTS):
        missing.append(f"{mix}: event coverage incomplete")
    event_rows = content.get("event_records")
    expected_events = {(event, e) for event in EVENTS for e in sizes}
    seen_events: set[tuple] = set()
    event_tapes: dict[str, Any] = {}
    if not isinstance(event_rows, list):
        missing.append(f"{mix}: executed event witnesses absent")
    else:
        for row in event_rows:
            if not isinstance(row, dict):
                invalid.append(f"{mix}: malformed event record")
                continue
            key = (row.get("event"), row.get("worlds"))
            if key not in expected_events or key in seen_events:
                invalid.append(f"{mix}: foreign/duplicate event record")
                continue
            seen_events.add(key)
            status = row.get("status")
            if status in {"INCOMPLETE", "timeout"} or row.get("ticks") != 2:
                missing.append(f"{mix}: unexercised event {key}")
            elif status not in {"PASS", "INVALID_SELF_CONSISTENCY"}:
                invalid.append(f"{mix}: invalid event status {key}")
            witness = row.get("witness")
            if not isinstance(witness, dict) or not witness:
                missing.append(f"{mix}: absent event predicate witness {key}")
            else:
                digest = hashlib.sha256(
                    json.dumps(witness, sort_keys=True, default=str).encode()
                ).hexdigest()
                if row.get("witness_sha256") != digest:
                    invalid.append(f"{mix}: event witness digest differs {key}")
                mapped = witness.get("mapped_tape")
                raw = witness.get("action_q_tape")
                event_fields = (
                    required_fields(mix)
                    - {f"collector.{n}" for n in ("length", "alive", "action", "qmax")}
                ) | {"event.supplied_actions"}
                if (
                    witness.get("ticks") != 2
                    or not isinstance(mapped, list)
                    or len(mapped) != 2
                    or not isinstance(raw, list)
                    or len(raw) != 2
                ):
                    missing.append(f"{mix}: event policy/state/tick proof omitted {key}")
                else:
                    # An incomplete producer may preserve a partial witness;
                    # it is absence of execution evidence, not contradictory
                    # evidence.  Only completed/mismatch rows are certified.
                    if status in {"INCOMPLETE", "timeout"}:
                        continue
                    predicate_error = _event_predicate_error(key[0], witness, key[1])
                    errors = [
                        _event_tick_error(raw_tick, descriptor_tick, mix, event_fields)
                        for raw_tick, descriptor_tick in zip(raw, mapped)
                    ]
                    if any(errors):
                        invalid.append(
                            f"{mix}: invalid event policy/state/tick proof {key}: "
                            f"{next(error for error in errors if error)}"
                        )
                    if predicate_error is not None:
                        invalid.append(f"{mix}: {predicate_error} {key}")
                    if row.get("status") == "INVALID_SELF_CONSISTENCY":
                        first = witness.get("first_witness")
                        if not isinstance(first, dict) or not first.get("field"):
                            invalid.append(f"{mix}: event mismatch witness absent {key}")
                        else:
                            mismatches.append(first)
                    data = {"mapped_tape": mapped, "action_q_tape": raw}
                    if key[0] in event_tapes and event_tapes[key[0]] != data:
                        mismatches.append(f"{mix}: event {key[0]} mapped tape differs across E")
                    event_tapes.setdefault(key[0], data)
        if seen_events != expected_events:
            missing.append(f"{mix}: required event/E cells missing")
    return invalid, missing, mismatches


def aggregate(
    exports: Path, commit: str, *, canary: bool = False, smoke: bool = False
) -> dict[str, Any]:
    """Reduce all verified units once, without trusting producer verdicts."""
    lane = units.mode(smoke=smoke, canary=canary)
    invalid: list[str] = []
    missing: list[str] = []
    mismatches: list[Any] = []
    result: dict[str, Any] = {
        "schema": "gpu-sim-spike-t4a-analysis/v1",
        "commit": commit,
        "mode": lane,
        "claim": "standalone dev batch-invariance premise only; no CPU/label/release authority",
        "invalid_evidence": invalid,
        "incomplete": missing,
        "mismatches": mismatches,
    }
    try:
        native_path = exports / "native-parity.json"
        native = _json(native_path) if native_path.is_file() else {}
        expected = "PREFLIGHT" if smoke else "PASS"
        if not native:
            missing.append("native proof absent")
        elif native.get("commit") != commit or native.get("status") != expected:
            invalid.append("current-pin native proof absent or failed")
        if native and not smoke and native.get("native_cuda_executed") is not True:
            invalid.append("native CUDA execution not proved")
        failure = exports / "t4a-failure.json"
        if failure.exists():
            fail = _json(failure)
            if fail.get("commit") != commit:
                invalid.append("foreign failure record")
            if fail.get("status") == "INVALID_SELF_CONSISTENCY":
                witness = fail.get("first_witness")
                if not isinstance(witness, dict) or not witness.get("field"):
                    invalid.append("behavioral failure omitted first witness")
                else:
                    mismatches.append(witness)
            elif fail.get("status") == "INVALID_EVIDENCE":
                invalid.append(str(fail.get("reason", "producer evidence failure")))
            else:
                missing.append(str(fail.get("reason", "producer interrupted")))
        try:
            if not (exports / "verified-ledger.json").is_file():
                raise FileNotFoundError("verified ledger absent")
            rows = units.load_ledger(
                exports / "verified-ledger.json",
                units.namespace(commit, smoke=smoke, canary=canary),
                units.allocation(),
                exports=exports,
            )
        except FileNotFoundError:
            rows = {}
            missing.append("verified ledger absent")
        expected_paths = {f"{u}.json" for u in rows}
        actual_paths = {
            p.name
            for p in (exports / units.UNIT_DIRECTORY).glob("*.json")
            if Path(str(p) + ".complete").exists()
        }
        if actual_paths != expected_paths:
            invalid.append("unaccounted completed export or duplicate source")
        for unit in units.allocation():
            if unit not in rows:
                missing.append(f"missing unit: {unit}")
                continue
            content = rows[unit]
            status = content.get("status")
            if status == "INVALID_EVIDENCE":
                invalid.append(f"{unit}: producer invalid evidence")
            elif status == "INVALID_SELF_CONSISTENCY":
                mismatches.append(content.get("first_witness", unit))
            elif status != "COMPLETE":
                missing.append(f"{unit}: {status}")
            errs, gaps, differences = _content_errors(content, units.parse(unit), lane)
            invalid.extend(errs)
            missing.extend(gaps)
            mismatches.extend(differences)
            if content.get("first_witness") is not None:
                mismatches.append(content["first_witness"])
    except FileNotFoundError as exc:
        missing.append(str(exc))
    except (ValueError, TypeError, KeyError) as exc:
        invalid.append(str(exc))
    result["verdict"] = (
        "INVALID_EVIDENCE"
        if invalid
        else (
            "INVALID_SELF_CONSISTENCY"
            if mismatches
            else (
                "INCOMPLETE"
                if missing
                else "PREFLIGHT" if smoke else "PASS_CANARY" if canary else "PASS_T4A_DEV"
            )
        )
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    result = aggregate(a.exports, a.commit, canary=a.canary, smoke=a.smoke)
    units.publish_json(a.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
