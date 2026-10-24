"""T4A evidence admission, immutable resume and reducer precedence regressions."""

from __future__ import annotations

import hashlib
import json
import struct

import pytest

from research.gpu_sim_spike import t4a_analyze as analyze
from research.gpu_sim_spike import t4a_units as units

COMMIT = "c" * 40


def _content(mix: str, *, canary: bool = False) -> dict:
    fields = ["sim.bodies", "sim.rng_counter", "policy.q", "policy.actions"]
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    horizon = 2 if canary else 256
    slots = {"scripted": (0,), "mixed": (0, 1, 3, 5), "frozen": tuple(range(6))}[mix]
    tape_tick = {
        "actions_hex": struct.pack("<24q", *([1] * 24)).hex(),
        "q_hex": {str(s): struct.pack("<24f", *([1.0] * 24)).hex() for s in slots},
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
        "coverage": dict.fromkeys(analyze.EVENTS, True),
        "first_witness": None,
        "records": [
            {
                "worlds": e,
                "layout": layout,
                "replay": repeat,
                "ticks": horizon,
                "matched_ticks": horizon,
                "comparison_count": 100,
                "forward_rows": 1024,
                "forward_calls": horizon,
                "field_inventory_sha256": digest,
                "action_q_tape": [tape_tick] * horizon,
            }
            for e in (4, 64, 1024)
            for layout in analyze.LAYOUTS
            for repeat in (0, 1)
        ],
    }


def _tree(root, *, canary=False, mutate=None, drop=None):
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


def test_complete_and_canary_are_distinct(tmp_path):
    assert analyze.aggregate(_tree(tmp_path / "m"), COMMIT)["verdict"] == "PASS_T4A_DEV"
    assert (
        analyze.aggregate(_tree(tmp_path / "c", canary=True), COMMIT, canary=True)["verdict"]
        == "PASS_CANARY"
    )
    assert analyze.aggregate(tmp_path / "m", "d" * 40)["verdict"] == "INVALID_EVIDENCE"


def test_invalid_controls_precede_valid_mismatch_and_incomplete(tmp_path):
    def change(c):
        c["controls"]["q"] = False
        c["first_witness"] = {"field": "q", "byte": 0}
        c["status"] = "INVALID_SELF_CONSISTENCY"

    assert (
        analyze.aggregate(_tree(tmp_path, mutate=change, drop="roster-mixed"), COMMIT)["verdict"]
        == "INVALID_EVIDENCE"
    )


def test_mismatch_precedes_missing_units(tmp_path):
    def change(c):
        c["first_witness"] = {"field": "sim.bodies", "byte": 1}
        c["status"] = "INVALID_SELF_CONSISTENCY"

    result = analyze.aggregate(_tree(tmp_path, mutate=change, drop="roster-mixed"), COMMIT)
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

    assert analyze.aggregate(_tree(tmp_path, mutate=change), COMMIT)["verdict"] == "INCOMPLETE"


def test_duplicate_or_omitted_forward_is_invalid(tmp_path):
    def change(c):
        c["records"].append(c["records"][0])
        c["records"][0]["forward_rows"] = 4

    assert (
        analyze.aggregate(_tree(tmp_path, mutate=change), COMMIT)["verdict"] == "INVALID_EVIDENCE"
    )


def test_immutable_verified_skip_and_foreign_allocation(tmp_path):
    ns, cells = units.namespace(COMMIT), units.allocation()
    empty = tmp_path / "empty.json"
    units.publish_json(empty, units.empty_ledger(ns, cells))
    assert units.load_ledger(empty, ns, cells) == {}
    root = _tree(tmp_path / "source")
    target = tmp_path / "recovery/verified-ledger.json"
    ledger = units.build_ledger(root, target, commit=COMMIT)
    assert len(ledger["units"]) == 3
    assert len(units.load_ledger(target, ns, cells)) == 3
    with pytest.raises(FileExistsError):
        units.publish_unit(root, ns, cells, cells[0], {})
    with pytest.raises(ValueError, match="foreign"):
        units.load_ledger(target, units.namespace(COMMIT, canary=True), cells)
    payload = units.payload_path(target.parent, cells[0])
    payload.write_bytes(payload.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        units.load_ledger(target, ns, cells)


def test_incomplete_marker_not_eligible_for_skip(tmp_path):
    root = _tree(tmp_path / "source")
    payload = units.payload_path(root, "roster-frozen")
    payload.with_name(payload.name + ".complete").unlink()
    result = units.build_ledger(root, tmp_path / "recovery/verified-ledger.json", commit=COMMIT)
    assert len(result["units"]) == 2
