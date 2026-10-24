"""Offline I0 contract tests: no CUDA, allocation, or timing required."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.gpu_sim_spike import integration_analyze as analyze
from research.gpu_sim_spike import integration_engine as engine
from research.gpu_sim_spike import integration_units as units

PIN = "a" * 40


def cell(mix: str, worlds: int) -> dict:
    arms = {}
    for name, seconds in (("R", 2.0), ("I", worlds / 2500)):
        arms[name] = {
            "plain_seconds": [seconds] * 3,
            "digests": ["a" * 64] * 3,
            "stages": [
                {
                    "repeat": index,
                    "wall_seconds": seconds,
                    "exclusive_cuda_elapsed_seconds": {"unscoped": seconds},
                }
                for index in range(3)
            ],
        }
    continuous = {"status": "COMPLETE", "orders": analyze.ORDERS, "ticks": 32, "arms": {}}
    for name, rate in (("R", 1000.0), ("I", 2500.0)):
        wall = 32768 / rate
        continuous["arms"][name] = {
            "repeats": [
                {
                    "repeat": index,
                    "completed_world_steps": 32768,
                    "wall_seconds": wall,
                    "tick_seconds": [wall / 32] * 32,
                    "complete_world_steps_per_second": rate,
                    "final_digest": "b" * 64,
                    "collector_positions": list(range(32)),
                }
                for index in range(3)
            ]
        }
    return {
        "mix": mix,
        "worlds": worlds,
        "repeats": 3,
        "warmups": 3,
        "orders": analyze.ORDERS,
        "capacity": 10400,
        "lengths": [800, 900, 1000, 1100, 1300, 1500],
        "arms": arms,
        "continuous": continuous,
        "native_parity": "PASS",
        "frozen_device_identity": True,
        "cpu_geometry_carry": True,
        "combined_parity": {
            "frozen_device_identity": True,
            "cpu_geometry_carry": True,
            "frozen_move": {"R": {"match": True}, "I": {"match": True}},
            "feature_calls": [{"arm": "I", "cpu_match": True}],
            "state_inventory": {"R": {}, "I": {}},
            "action_tape": {"shape": [worlds, 6]},
        },
    }


def native() -> dict:
    from research.gpu_sim_spike.benchmark import source_hashes

    return {
        "commit": PIN,
        "source_sha256": source_hashes(),
        "status": "PASS",
        "native_cuda_executed": True,
        "fallback": False,
        "rows": [{"outcome": "passed"}] * 14,
        "geometry_native_calls": {"indexed_body_view": 2, "indexed_features58": 4},
        "move2_events": {"status": "PASS", "fallback": False, "native_calls": {"move": 10}},
        "compiled_binaries": dict.fromkeys(("geometry", "move", "events", "flood"), {}),
    }


def exports(tmp_path, *, canary=False, slow=False):
    root = tmp_path / "exports"
    cells, ns = units.allocation(canary=canary), units.namespace(PIN, canary=canary)
    units.publish_json(root / "native-parity.json", native())
    units.publish_json(
        root / "discrepancy-register.json",
        {"commit": PIN, "cpu_component_register": {"status": "DIAGNOSTIC"}},
    )
    for unit in cells:
        mix, worlds = unit.split("-")[1], int(unit.rsplit("e", 1)[1])
        value = cell(mix, worlds)
        if slow and worlds == 1024:
            for repeat in value["continuous"]["arms"]["I"]["repeats"]:
                repeat.update(
                    wall_seconds=32.768,
                    complete_world_steps_per_second=1000.0,
                    tick_seconds=[1.024] * 32,
                )
        if canary:
            value.update(repeats=1, warmups=1, orders=analyze.ORDERS[:1])
            for arm in value["arms"].values():
                for key in ("plain_seconds", "digests", "stages"):
                    arm[key] = arm[key][:1]
        units.publish_unit(root, ns, cells, unit, value)
    units.build_ledger(root, root / "verified-ledger.json", commit=PIN, canary=canary)
    return root


def test_primary_order_canary_and_historical_control_are_frozen() -> None:
    assert units.allocation()[:9] == tuple(
        f"cell-{mix}-e{worlds}"
        for worlds in (1024, 256, 64)
        for mix in ("frozen", "mixed", "scripted")
    )
    assert units.allocation()[-3:] == ("cell-frozen-e6", "cell-mixed-e6", "cell-scripted-e6")
    assert units.allocation(canary=True) == (
        "cell-frozen-e64",
        "cell-mixed-e64",
        "cell-scripted-e64",
    )
    assert engine.arm_spec("I").vector_flood == "static" and not engine.arm_spec("I").v8_indexed


def test_ledger_skips_only_complete_same_namespace_payloads(tmp_path: Path) -> None:
    cells, ns = units.allocation(smoke=True), units.namespace(PIN, smoke=True)
    units.publish_unit(tmp_path, ns, cells, cells[0], {"ok": True})
    ledger = tmp_path / "recovery" / "verified-ledger.json"
    units.build_ledger(tmp_path, ledger, commit=PIN, smoke=True)
    assert units.load_ledger(ledger, ns, cells) == {cells[0]: {"ok": True}}
    with pytest.raises(ValueError, match="foreign"):
        units.load_ledger(ledger, units.namespace("b" * 40, smoke=True), cells)


def test_reducer_requires_complete_native_and_canary_coverage(tmp_path):
    root = exports(tmp_path, canary=True)
    assert analyze.aggregate([root], PIN, canary=True)["verdict"] == "PASS_CANARY"
    assert analyze.aggregate([root, root], PIN, canary=True)["verdict"] == "INCOMPLETE"
    (root / "native-parity.json.sha256").unlink()
    assert analyze.aggregate([root], PIN, canary=True)["verdict"] == "INCOMPLETE"


def test_slow_baseline_allowed_and_continuous_speed_failure_is_stop(tmp_path):
    root = exports(tmp_path / "fast")
    assert analyze.aggregate([root], PIN)["verdict"] == "PASS_SPEED"
    root = exports(tmp_path / "slow", slow=True)
    report = analyze.aggregate([root], PIN)
    assert report["verdict"] == "STOP_SPEED"
    assert len(report["e1024_complete_world_steps_per_second"]) == 3


@pytest.mark.parametrize(
    "damage",
    [
        "empty_stage",
        "negative_stage",
        "missing_continuous",
        "unequal_digest",
        "false_proof",
        "wrong_order",
        "forged_rate",
    ],
)
def test_adversarial_incomplete_or_forged_cells_refused(damage):
    value = cell("frozen", 1024)
    if damage == "empty_stage":
        value["arms"]["I"]["stages"][0]["exclusive_cuda_elapsed_seconds"] = {}
    if damage == "negative_stage":
        value["arms"]["I"]["stages"][0]["exclusive_cuda_elapsed_seconds"] = {"unscoped": -1}
    if damage == "missing_continuous":
        value["continuous"] = {}
    if damage == "unequal_digest":
        value["arms"]["I"]["digests"][1] = "c" * 64
    if damage == "false_proof":
        value["combined_parity"]["frozen_move"]["I"]["match"] = False
    if damage == "wrong_order":
        value["orders"] = []
    if damage == "forged_rate":
        value["continuous"]["arms"]["I"]["repeats"][0]["complete_world_steps_per_second"] = float(
            "nan"
        )
    assert not analyze.valid_cell("cell-frozen-e1024", value)
