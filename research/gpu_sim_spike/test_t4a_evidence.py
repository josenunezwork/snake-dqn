"""T4A evidence admission, immutable resume and reducer precedence regressions."""

from __future__ import annotations

import hashlib
import json
import struct
from copy import deepcopy

import pytest

from research.gpu_sim_spike import t4a_analyze as analyze
from research.gpu_sim_spike import t4a_units as units

COMMIT = "c" * 40


def _content(mix: str, *, canary: bool = True) -> dict:
    fields = sorted(analyze.required_fields(mix))
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    horizon = 2 if canary else 256
    slots = {"scripted": (0,), "mixed": (0, 1, 3, 5), "frozen": tuple(range(6))}[mix]
    tape_tick = {
        "actions_hex": struct.pack("<24q", *([1] * 24)).hex(),
        "policy_actions_hex": struct.pack("<24q", *([1] * 24)).hex(),
        "q_hex": {str(s): struct.pack("<24f", *([1.0] * 24)).hex() for s in slots},
    }

    def descriptor(name: str) -> dict:
        dtype, shape = "torch.float32", [4, 6]
        if name in {"lifecycle.sim_prepared", "lifecycle.vector_prepared"}:
            dtype, shape = "torch.bool", []
        elif name == "configuration.sha256":
            dtype, shape = "torch.uint8", [32]
        elif name.startswith("sim."):
            part = name.removeprefix("sim.")
            if part in {"frame", "rng_counter", "rng_key", "food_count"}:
                dtype, shape = "torch.int64", [4]
            elif part == "food_cells":
                dtype, shape = "torch.int64", [4, 8, 2]
            elif part in {"food_present", "corpse_food"}:
                dtype, shape = "torch.bool", [4, 8]
            elif part in {
                "alive",
                "terminal_slots",
                "capacity_overflow",
                "_last_done",
                "_last_food_ate",
                "_last_ambient_food_ate",
                "_last_corpse_food_ate",
                "_last_transition_valid",
                "_boosted_this_step",
            }:
                dtype, shape = "torch.bool", [4, 6]
            elif part != "_last_reward":
                dtype, shape = "torch.int64", [4, 6]
        elif name.startswith("runtime."):
            part = name.removeprefix("runtime.")
            if part in {"controlled", "_carry_valid", "_alive_post", "sel_ready", "sel_fresh"}:
                dtype, shape = "torch.bool", [4, 6]
            elif part == "prev_dist":
                dtype, shape = "torch.float64", [4, 6]
            elif part == "prev_id":
                dtype, shape = "torch.int64", [4, 6]
            elif part in {"_carry_state", "sel_state"}:
                dtype, shape = "torch.float32", [4, 6, 61]
            elif part in {"_carry_mask", "sel_mask"}:
                dtype, shape = "torch.bool", [4, 6, 6]
        elif name.startswith("collector."):
            dtype, shape = {
                "collector.length": ("torch.int64", [4]),
                "collector.alive": ("torch.bool", [4]),
                "collector.action": ("torch.int64", [4]),
                "collector.qmax": ("torch.float32", [4]),
            }[name]
        elif name == "output.actions":
            dtype, shape = "torch.int64", [4, 6]
        elif name.startswith("output.obs."):
            _output, _obs, slot, index = name.split(".")
            if (slot, index) == ("0", "0"):
                dtype, shape = "torch.uint8", [4, 7, 31, 31]
            elif (slot, index) == ("0", "1"):
                dtype, shape = "torch.uint8", [4, 4, 37, 37]
            elif (slot, index) == ("0", "2"):
                dtype, shape = "torch.float32", [4, 60]
            else:
                dtype, shape = "torch.float32", [4, 61]
        elif name.startswith("output.advisory."):
            dtype, shape = "torch.bool", [4, 6, 6]
        elif name.startswith("output.v8."):
            part = name.rsplit(".", 1)[-1]
            dtype, shape = {
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
            }[part]
        elif name == "event.ring_wrap.before_head_ptr":
            dtype, shape = "torch.int64", [4]
        elif name == "event.ring_wrap.prefix":
            dtype, shape = "torch.int64", [4, 2, 2]
        elif name.startswith("event.contested_food."):
            suffix = name.removeprefix("event.contested_food.")
            if suffix.endswith("cells"):
                dtype, shape = "torch.int64", [4, 8, 2]
            elif suffix.endswith("present") or suffix.endswith("corpse"):
                dtype, shape = "torch.bool", [4, 8]
            else:
                dtype, shape = "torch.int64", [4]
        return {"dtype": dtype, "shape": shape, "sha256": "a" * 64}

    descriptors = {name: descriptor(name) for name in fields}
    descriptors["output.actions"]["sha256"] = hashlib.sha256(
        bytes.fromhex(tape_tick["policy_actions_hex"])
    ).hexdigest()
    descriptors["sim.bodies"] = {
        "dtype": "torch.int64",
        "shape": [4, 6, 8, 2],
        "digest_algorithm": "t4a-device-mix64x4-v1",
        "digest": "b" * 64,
    }
    for slot, raw in tape_tick["q_hex"].items():
        descriptors[f"output.q.{slot}"] = {
            "dtype": "torch.float32",
            "shape": [4, 6],
            "sha256": hashlib.sha256(bytes.fromhex(raw)).hexdigest(),
        }
    event_descriptors = {k: v for k, v in descriptors.items() if not k.startswith("collector.")}
    event_descriptors["event.supplied_actions"] = descriptors["output.actions"]
    base_event_fields = set(event_descriptors)
    event_witness_base = {
        "executed": True,
        "ticks": 2,
        "mapped_tape": [event_descriptors] * 2,
        "action_q_tape": [tape_tick] * 2,
    }
    predicates = {
        "ring_wrap": {
            "before": [7] * 4,
            "after": [0] * 4,
            "logical_prefix": [[[5, 4], [4, 4]]] * 4,
            "ring_schema": {
                "field": "sim.bodies",
                "tick": 0,
                "capacity": 8,
                "dtype": "torch.int64",
                "shape": [4, 6, 8, 2],
                "physical_worlds": 4,
            },
            "frozen_geometry": {
                "slot": 0,
                "direction": 1,
                "prefix_length": 2,
                "initial_head": [4, 4],
                "initial_tail": [3, 4],
                "moved_head": [5, 4],
                "retained_tail": [4, 4],
            },
            "wrapped": True,
            "ordered": True,
        },
        "simultaneous_death_respawn": {
            "died": [True] * 4,
            "respawned": [True] * 4,
            "corpse": [True] * 4,
        },
        "contested_food_refill": {
            "slot_1_ate": [True] * 4,
            "slot_2_ate": [False] * 4,
            "food_after_first": [1] * 4,
            "refill_cells": [[1, 1]] * 4,
            "rng_before": [1] * 4,
            "rng_after": [2] * 4,
        },
        "clone_restore": {
            "before_restore_digest": "c" * 64,
            "restored_exact": True,
            "saved_digest": "d" * 64,
            "restored_digest": "d" * 64,
            "branch_digest": "e" * 64,
            "replay_digest": "e" * 64,
            "restore_target_body_progressed": True,
            "restore_target_food_progressed": True,
            "mismatch": None,
        },
        "stale_cache": {
            "warm_stats": {"misses": 1},
            "clone_stats": {"bumps": 1},
            "fresh_stats": {"misses": 1},
            "warm_generation": 1,
            "clone_generation": 2,
            "fresh_generation": 1,
            "warm_digest": "f" * 64,
            "restored_digest": "f" * 64,
            "restore_branch_digest": "e" * 64,
            "restore_replay_digest": "e" * 64,
            "remap_source_digest": "f" * 64,
            "remap_restored_digest": "f" * 64,
            "remap_branch_digest": "d" * 64,
            "remap_replay_digest": "d" * 64,
            "warm_vector_prepared": True,
            "clone_vector_prepared": True,
            "fresh_vector_prepared": True,
            "restore_exact": True,
            "restore_replay_exact": True,
            "remap_exact": True,
            "remap_replay_exact": True,
            "lifecycle": True,
            "broken_invalidation_control": True,
            "mismatch": None,
        },
    }

    def event_witness(event: str, worlds: int) -> dict:
        def typed_digest(value, dtype: str) -> str:
            flattened = []

            def visit(item):
                if isinstance(item, list):
                    for child in item:
                        visit(child)
                else:
                    flattened.append(item)

            visit(value)
            fmt = {"torch.bool": "<?", "torch.int64": "<q"}[dtype]
            return hashlib.sha256(
                b"".join(struct.pack(fmt, item) for item in flattened)
            ).hexdigest()

        witness = {**event_witness_base, **deepcopy(predicates[event])}
        captures: dict[str, dict] = {}
        if event == "ring_wrap":
            witness["before"] = [7] * worlds
            witness["after"] = [0] * worlds
            witness["logical_prefix"] = [[[5, 4], [4, 4]]] * worlds
            witness["ring_schema"]["physical_worlds"] = worlds
            prefix_raw = struct.pack("<" + "q" * 16, *([5, 4, 4, 4] * 4))
            event_descriptors["event.ring_wrap.before_head_ptr"] = {
                "dtype": "torch.int64",
                "shape": [4],
                "sha256": hashlib.sha256(struct.pack("<4q", 7, 7, 7, 7)).hexdigest(),
            }
            event_descriptors["event.ring_wrap.after_head_ptr"] = {
                "dtype": "torch.int64",
                "shape": [4],
                "sha256": hashlib.sha256(struct.pack("<4q", 0, 0, 0, 0)).hexdigest(),
            }
            event_descriptors["event.ring_wrap.prefix"] = {
                "dtype": "torch.int64",
                "shape": [4, 2, 2],
                "sha256": hashlib.sha256(prefix_raw).hexdigest(),
            }
            captures = {
                "ring_prefix_capture": {
                    "field": "event.ring_wrap.prefix",
                    "tick": 0,
                    **event_descriptors["event.ring_wrap.prefix"],
                },
                "ring_before_capture": {
                    "field": "event.ring_wrap.before_head_ptr",
                    "tick": 0,
                    **event_descriptors["event.ring_wrap.before_head_ptr"],
                },
                "ring_after_capture": {
                    "field": "event.ring_wrap.after_head_ptr",
                    "tick": 0,
                    **event_descriptors["event.ring_wrap.after_head_ptr"],
                },
            }
            witness.update(captures)
        elif event == "simultaneous_death_respawn":
            for name in ("died", "respawned", "corpse"):
                witness[name] = [True] * worlds
            for name, tick in (("died", 0), ("corpse", 0), ("respawned", 1)):
                field = f"event.death_respawn.{name}"
                event_descriptors[field] = {
                    "dtype": "torch.bool",
                    "shape": [4],
                    "sha256": typed_digest([True] * 4, "torch.bool"),
                }
            witness["death_respawn_capture"] = {
                name: {
                    "field": f"event.death_respawn.{name}",
                    "tick": 1 if name == "respawned" else 0,
                    **event_descriptors[f"event.death_respawn.{name}"],
                }
                for name in ("died", "corpse", "respawned")
            }
        elif event == "contested_food_refill":
            witness["slot_1_ate"] = [True] * worlds
            witness["slot_2_ate"] = [False] * worlds
            witness["food_after_first"] = [1] * worlds
            witness["refill_cells"] = [[[1, 1]] for _ in range(4)]
            witness["rng_before"] = [1] * worlds
            witness["rng_after"] = [2] * worlds
            transition = {
                "before_cells": [[[17, 5]] + [[0, 0]] * 7 for _ in range(4)],
                "before_present": [[True] + [False] * 7 for _ in range(4)],
                "before_count": [1] * 4,
                "before_corpse": [[False] * 8 for _ in range(4)],
                "after_consume_cells": [[[0, 0]] * 8 for _ in range(4)],
                "after_consume_present": [[False] * 8 for _ in range(4)],
                "after_consume_count": [0] * 4,
                "after_consume_corpse": [[False] * 8 for _ in range(4)],
                "after_refill_cells": [[[1, 1]] + [[0, 0]] * 7 for _ in range(4)],
                "after_refill_present": [[True] + [False] * 7 for _ in range(4)],
                "after_refill_count": [1] * 4,
                "after_refill_corpse": [[False] * 8 for _ in range(4)],
            }
            witness["food_transition"] = transition
            witness["food_transition_capture"] = {}
            for name, value in transition.items():
                field = f"event.contested_food.{name}"
                event_descriptors[field] = descriptor(field)
                event_descriptors[field]["sha256"] = typed_digest(
                    value, event_descriptors[field]["dtype"]
                )
                witness["food_transition_capture"][name] = {
                    "field": field,
                    "tick": 1 if name.startswith("after_refill_") else 0,
                    **event_descriptors[field],
                }
            witness["refill_cells"] = [[[1, 1]] for _ in range(4)]
        elif event == "stale_cache":
            witness["mask_shape"] = [worlds, 6, 6]
            witness["mask_dtype"] = "torch.bool"
            witness["stale_mask_hex"] = "00" * (worlds * 6 * 6)
            witness["repaired_mask_hex"] = "01" * (worlds * 6 * 6)
            warm_fields = {
                field
                for field in base_event_fields
                if (
                    field == "configuration.sha256"
                    or field.startswith("lifecycle.")
                    or field == "output.actions"
                    or field.startswith("output.q.")
                    or field.startswith("output.obs.")
                    or field.startswith("output.v8.")
                )
            }
            warm = {field: deepcopy(event_descriptors[field]) for field in warm_fields}
            for name in ("_carry_state", "_carry_mask", "_carry_valid"):
                warm["prepared.runtime." + name] = deepcopy(event_descriptors["runtime." + name])
            witness["warm_output_captures"] = {
                name: deepcopy(warm) for name in ("source", "clone", "remap", "restore", "fresh")
            }
            corrupted = deepcopy(warm)
            corrupted["output.q.0"]["sha256"] = "f" * 64
            witness["warm_output_captures"]["corrupted_q_control"] = corrupted
            witness["warm_output_comparisons"] = {
                name: {"equal": True, "kind": None, "difference": None}
                for name in ("source_clone", "source_remap", "source_restore", "source_fresh")
            }
            names = (
                "source",
                "clone",
                "fresh",
                "remap",
                "remap_replay",
                "restore",
                "restore_replay",
            )
            witness["prepared_tick_captures"] = {
                name: {
                    field: deepcopy(value)
                    for field, value in warm.items()
                    if not field.startswith("lifecycle.")
                }
                for name in names
            }
            witness["prepared_tick_comparisons"] = {
                name: {"equal": True, "kind": None, "difference": None} for name in names
            }
            witness["evidence_errors"] = []
            witness["warm_q_corruption_detected"] = True
            witness["warm_q_corruption_control"] = {
                "field": "output.q.0",
                "source_descriptor": warm["output.q.0"],
                "corrupted_descriptor": corrupted["output.q.0"],
            }
            witness["restore_target_body_progressed"] = True
            witness["restore_target_food_progressed"] = True
        witness["mapped_tape"] = [
            {
                field: deepcopy(event_descriptors[field])
                for field in base_event_fields
                | analyze.EVENT_CAPTURE_TICK_FIELDS.get(event, (set(), set()))[tick]
            }
            for tick in range(2)
        ]
        return witness

    return {
        "status": "COMPLETE",
        "roster": mix,
        "mode": "canary" if canary else "main",
        "horizon": horizon,
        "root_seed": (9402411 if canary else 9402401) + units.MIXES.index(mix),
        "fields": fields,
        "field_types": {
            name: {key: value for key, value in descriptor.items() if key in {"dtype", "shape"}}
            for name, descriptor in descriptors.items()
        },
        "field_inventory_sha256": digest,
        "controls": dict.fromkeys(analyze.CONTROLS, True),
        "control_witnesses": {
            key: {"field": field, "byte_offset": 0, "expected": 0, "actual": 1}
            for key, field in {
                "q": "output.q.0",
                "state": "sim.bodies",
                "cursor": "sim.rng_counter",
                "row_map": "output.q.0",
            }.items()
        },
        "coverage": dict.fromkeys(analyze.EVENTS, True),
        "event_records": [
            {
                "event": event,
                "worlds": e,
                "ticks": 2,
                "status": "PASS",
                "witness": event_witness(event, e),
                "witness_sha256": hashlib.sha256(
                    json.dumps(event_witness(event, e), sort_keys=True).encode()
                ).hexdigest(),
            }
            for event in analyze.EVENTS
            for e in (4, 64, 1024)
        ],
        "first_witness": None,
        "records": [
            {
                "worlds": e,
                "layout": layout,
                "map_proof": {
                    "layout": layout,
                    "worlds": e,
                    "logical_ids": [0, 1, 2, 3],
                    "history_offsets": [0, 3, 11, 23],
                    "insertion_rows": analyze._expected_rows(e, layout),
                    "expected_rows": analyze._expected_rows(e, layout),
                    "extraction_rows": analyze._expected_rows(e, layout),
                    "placement_valid": True,
                    "extraction_valid": True,
                    "placement_checks": [True, True, True, True],
                },
                "extracted_rows": analyze._expected_rows(e, layout),
                "replay": repeat,
                "ticks": horizon,
                "matched_ticks": horizon,
                "comparison_count": horizon * len(fields),
                "forward_rows": 1024,
                "forward_calls": horizon * len(slots),
                "forward_shapes": [
                    {"padded_shapes": [[1024, 61]], "output_shape": [1024, 6], "real_rows": e}
                ]
                * (horizon * len(slots)),
                "field_inventory_sha256": digest,
                "field_digests": [descriptors] * horizon,
                "action_q_tape": [tape_tick] * horizon,
            }
            for e in (4, 64, 1024)
            for layout in analyze.LAYOUTS
            for repeat in (0, 1)
        ],
    }


def _tree(root, *, canary=True, mutate=None, drop=None, mutate_mix=None):
    ns, cells = units.namespace(COMMIT, canary=canary), units.allocation()
    for unit in cells:
        if unit == drop:
            continue
        content = _content(units.parse(unit), canary=canary)
        if mutate and (units.parse(unit) == mutate_mix if mutate_mix else unit == cells[0]):
            mutate(content)
        units.publish_unit(root, ns, cells, unit, content)
    units.build_ledger(root, root / "verified-ledger.json", commit=COMMIT, canary=canary)
    units.publish_json(
        root / "native-parity.json",
        {"status": "PASS", "commit": COMMIT, "native_cuda_executed": True},
    )
    return root


def reduce(root, commit=COMMIT):
    return analyze.aggregate(root, commit, canary=True)


def test_canary_is_not_a_main_certificate(tmp_path):
    assert reduce(_tree(tmp_path / "m"))["verdict"] == "PASS_CANARY"
    assert (
        analyze.aggregate(_tree(tmp_path / "c", canary=True), COMMIT, canary=True)["verdict"]
        == "PASS_CANARY"
    )
    assert reduce(tmp_path / "m", "d" * 40)["verdict"] == "INVALID_EVIDENCE"
    assert analyze.aggregate(tmp_path / "m", COMMIT)["verdict"] == "INVALID_EVIDENCE"


def test_invalid_controls_precede_valid_mismatch_and_incomplete(tmp_path):
    def change(c):
        c["controls"]["q"] = False
        c["first_witness"] = {"field": "q", "byte": 0}
        c["status"] = "INVALID_SELF_CONSISTENCY"

    assert (
        reduce(_tree(tmp_path, mutate=change, drop="roster-mixed"))["verdict"] == "INVALID_EVIDENCE"
    )


def test_mismatch_precedes_missing_units(tmp_path):
    def change(c):
        reference = c["records"][0]
        candidate = c["records"][1]
        raw = bytearray.fromhex(candidate["action_q_tape"][0]["q_hex"]["0"])
        raw[0] ^= 1
        candidate["action_q_tape"][0] = {
            **candidate["action_q_tape"][0],
            "q_hex": {**candidate["action_q_tape"][0]["q_hex"], "0": raw.hex()},
        }
        candidate["field_digests"][0] = {
            **candidate["field_digests"][0],
            "output.q.0": {
                **candidate["field_digests"][0]["output.q.0"],
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        }
        c["first_witness"] = {
            "field": "output.q.0",
            "byte_offset": 0,
            "expected": 0,
            "actual": raw[0],
            "expected_descriptor": reference["field_digests"][0]["output.q.0"],
            "actual_descriptor": candidate["field_digests"][0]["output.q.0"],
            "comparison": {
                "reference": {"worlds": 4, "layout": "canonical", "replay": 0, "tick": 0},
                "candidate": {"worlds": 4, "layout": "canonical", "replay": 1, "tick": 0},
            },
        }
        c["status"] = "INVALID_SELF_CONSISTENCY"

    result = reduce(_tree(tmp_path, mutate=change, drop="roster-mixed"))
    assert result["verdict"] == "INVALID_SELF_CONSISTENCY"
    assert result["incomplete"]


@pytest.mark.parametrize("gap", ["coverage", "ticks", "record"])
def test_no_subset_pass(tmp_path, gap):
    def change(c):
        if gap == "coverage":
            c["coverage"]["stale_cache"] = False
        elif gap == "ticks":
            c["records"][0]["matched_ticks"] -= 1
        else:
            c["records"].pop()

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] == "INCOMPLETE"


def test_duplicate_or_omitted_forward_is_invalid(tmp_path):
    def change(c):
        c["records"].append(c["records"][0])
        c["records"][0]["forward_rows"] = 4

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] == "INVALID_EVIDENCE"


def test_omitted_policy_output_cannot_self_certify_inventory(tmp_path):
    def change(c):
        c["fields"].remove("output.obs.0.0")
        digest = hashlib.sha256(json.dumps(c["fields"], sort_keys=True).encode()).hexdigest()
        c["field_inventory_sha256"] = digest
        for row in c["records"]:
            row["field_inventory_sha256"] = digest

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] == "INVALID_EVIDENCE"


def test_valid_raw_q_mismatch_is_self_consistency_failure(tmp_path):
    def change(c):
        row = c["records"][1]
        tick = dict(row["action_q_tape"][0])
        tick["q_hex"] = dict(tick["q_hex"])
        raw = bytearray.fromhex(tick["q_hex"]["0"])
        raw[0] ^= 1
        tick["q_hex"]["0"] = raw.hex()
        row["action_q_tape"] = [tick, *row["action_q_tape"][1:]]
        desc = {k: dict(v) for k, v in row["field_digests"][0].items()}
        desc["output.q.0"]["sha256"] = hashlib.sha256(raw).hexdigest()
        row["field_digests"] = [desc, *row["field_digests"][1:]]

    result = reduce(_tree(tmp_path, mutate=change))
    assert result["verdict"] == "INVALID_SELF_CONSISTENCY"


@pytest.mark.parametrize("container", [False, True])
def test_malformed_typed_descriptor_is_invalid_evidence(tmp_path, container):
    def change(c):
        if container:
            c["records"][0]["field_digests"][0] = 0
        else:
            c["records"][0]["field_digests"][0]["output.actions"] = 0

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize(
    "fault",
    ["predicate", "none_descriptor", "empty_raw", "nonfinite_q", "missing_seat", "digest"],
)
def test_event_placeholders_or_unbound_raw_values_cannot_pass(tmp_path, fault):
    """B1: a producer label and matching placeholders never self-certify an event."""

    def change(content):
        witness = content["event_records"][0]["witness"]
        if fault == "predicate":
            witness.pop("before")
        elif fault == "none_descriptor":
            witness["mapped_tape"][0]["sim.bodies"] = None
        elif fault == "empty_raw":
            witness["action_q_tape"][0] = {}
        elif fault == "nonfinite_q":
            witness["action_q_tape"][0]["q_hex"]["0"] = struct.pack(
                "<24f", *([float("nan")] * 24)
            ).hex()
        elif fault == "missing_seat":
            witness["action_q_tape"][0]["q_hex"].pop("0")
        else:
            witness["action_q_tape"][0]["actions_hex"] = struct.pack("<24q", *([2] * 24)).hex()
        content["event_records"][0]["witness_sha256"] = hashlib.sha256(
            json.dumps(witness, sort_keys=True).encode()
        ).hexdigest()

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] != "PASS_CANARY"


def test_declared_reversed_layout_cannot_self_certify_canonical_row_map(tmp_path):
    def change(content):
        row = next(row for row in content["records"] if row["layout"] == "reversed")
        row["map_proof"]["insertion_rows"] = [0, 1, 2, 3]
        row["map_proof"]["expected_rows"] = [0, 1, 2, 3]
        row["map_proof"]["extraction_rows"] = [0, 1, 2, 3]

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize("fault", ["extracted", "placement"])
def test_row_map_requires_actual_extraction_and_nonempty_placement_checks(tmp_path, fault):
    def change(content):
        row = next(
            row
            for row in content["records"]
            if row["layout"] == "interspersed" and row["worlds"] == 64
        )
        if fault == "extracted":
            row["extracted_rows"] = [0, 1, 2, 3]
        else:
            row["map_proof"]["placement_checks"] = []

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] == "INVALID_EVIDENCE"


def test_nonempty_but_wrong_typed_descriptor_cannot_self_certify_event(tmp_path):
    def change(content):
        witness = content["event_records"][0]["witness"]
        witness["mapped_tape"][0]["sim.rng_counter"] = {
            "dtype": "torch.float64",
            "shape": [4, 0],
            "sha256": "a" * 64,
        }
        content["event_records"][0]["witness_sha256"] = hashlib.sha256(
            json.dumps(witness, sort_keys=True).encode()
        ).hexdigest()

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] == "INVALID_EVIDENCE"


def test_immutable_verified_skip_and_foreign_allocation(tmp_path):
    ns, cells = units.namespace(COMMIT, canary=True), units.allocation()
    empty = tmp_path / "empty.json"
    units.publish_json(empty, units.empty_ledger(ns, cells))
    assert units.load_ledger(empty, ns, cells) == {}
    root = _tree(tmp_path / "source")
    target = tmp_path / "recovery/verified-ledger.json"
    ledger = units.build_ledger(root, target, commit=COMMIT, canary=True)
    assert len(ledger["units"]) == 3
    assert len(units.load_ledger(target, ns, cells)) == 3
    with pytest.raises(FileExistsError):
        units.publish_unit(root, ns, cells, cells[0], {})
    with pytest.raises(ValueError, match="foreign"):
        units.load_ledger(target, units.namespace(COMMIT), cells)
    payload = units.payload_path(target.parent, cells[0])
    payload.write_bytes(payload.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        units.load_ledger(target, ns, cells)


def test_incomplete_marker_not_eligible_for_skip(tmp_path):
    root = _tree(tmp_path / "source")
    payload = units.payload_path(root, "roster-frozen")
    payload.with_name(payload.name + ".complete").unlink()
    result = units.build_ledger(
        root, tmp_path / "recovery/verified-ledger.json", commit=COMMIT, canary=True
    )
    assert len(result["units"]) == 2


def test_unknown_persistent_field_cannot_extend_the_frozen_inventory(tmp_path):
    def change(content):
        content["fields"].append("runtime.unknown_state")
        digest = hashlib.sha256(json.dumps(content["fields"], sort_keys=True).encode()).hexdigest()
        content["field_inventory_sha256"] = digest
        for record in content["records"]:
            record["field_inventory_sha256"] = digest

    result = reduce(_tree(tmp_path, mutate=change))
    assert result["verdict"] == "INVALID_EVIDENCE"
    assert any("foreign" in error for error in result["invalid_evidence"])


@pytest.mark.parametrize("fault", ["before", "geometry", "refill", "redirect"])
def test_review02_b1_predicate_contradictions_are_invalid_evidence(tmp_path, fault):
    def change(content):
        event = (
            "contested_food_refill"
            if fault == "refill"
            else "simultaneous_death_respawn" if fault == "redirect" else "ring_wrap"
        )
        row = next(row for row in content["event_records"] if row["event"] == event)
        witness = row["witness"]
        if fault == "before":
            witness["before"] = [1] * 4
        elif fault == "geometry":
            witness["logical_prefix"] = [[[99, 99], [99, 99]]] * 4
        elif fault == "redirect":
            witness["death_respawn_capture"]["respawned"] = {
                **witness["death_respawn_capture"]["died"]
            }
        else:
            witness.pop("refill_cells")
        row["witness_sha256"] = hashlib.sha256(
            json.dumps(witness, sort_keys=True).encode()
        ).hexdigest()

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize("fault", ["remove", "corrupt", "target"])
def test_review02_s1_s2_warm_and_restore_proof_is_required(tmp_path, fault):
    def change(content):
        witness = next(
            row["witness"] for row in content["event_records"] if row["event"] == "stale_cache"
        )
        if fault == "remove":
            witness.pop("warm_output_captures")
        elif fault == "corrupt":
            witness["warm_output_captures"]["clone"]["output.q.0"]["sha256"] = "f" * 64
        else:
            witness.pop("restore_target_body_progressed")
        for row in content["event_records"]:
            if row["event"] == "stale_cache":
                row["witness_sha256"] = hashlib.sha256(
                    json.dumps(row["witness"], sort_keys=True).encode()
                ).hexdigest()

    assert reduce(_tree(tmp_path, mutate=change))["verdict"] == "INVALID_EVIDENCE"
