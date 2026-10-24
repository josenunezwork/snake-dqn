"""T4A evidence admission, immutable resume and reducer precedence regressions."""

from __future__ import annotations

import hashlib
import json
import struct

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
        "q_hex": {str(s): struct.pack("<24f", *([1.0] * 24)).hex() for s in slots},
    }
    descriptors = {
        name: {"dtype": "torch.int64", "shape": [4, 6], "sha256": "a" * 64} for name in fields
    }
    descriptors["output.actions"]["sha256"] = hashlib.sha256(
        bytes.fromhex(tape_tick["actions_hex"])
    ).hexdigest()
    for slot, raw in tape_tick["q_hex"].items():
        descriptors[f"output.q.{slot}"] = {
            "dtype": "torch.float32",
            "shape": [4, 6],
            "sha256": hashlib.sha256(bytes.fromhex(raw)).hexdigest(),
        }
    event_descriptors = {k: v for k, v in descriptors.items() if not k.startswith("collector.")}
    event_descriptors["event.supplied_actions"] = descriptors["output.actions"]
    event_witness = {
        "executed": True,
        "ticks": 2,
        "mapped_tape": [event_descriptors] * 2,
        "action_q_tape": [tape_tick] * 2,
    }
    return {
        "status": "COMPLETE",
        "roster": mix,
        "mode": "canary" if canary else "main",
        "horizon": horizon,
        "root_seed": (9402411 if canary else 9402401) + units.MIXES.index(mix),
        "fields": fields,
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
                "witness": event_witness,
                "witness_sha256": hashlib.sha256(
                    json.dumps(event_witness, sort_keys=True).encode()
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


def _tree(root, *, canary=True, mutate=None, drop=None):
    ns, cells = units.namespace(COMMIT, canary=canary), units.allocation()
    for unit in cells:
        if unit == drop:
            continue
        content = _content(units.parse(unit), canary=canary)
        if mutate and unit == cells[0]:
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
        c["first_witness"] = {"field": "sim.bodies", "byte": 1}
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
