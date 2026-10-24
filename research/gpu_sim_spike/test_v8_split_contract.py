"""Fixed engineering selection, coverage precedence and exact immutable resumption."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.gpu_sim_spike import v8_split_analyze as analyze
from research.gpu_sim_spike import v8_split_bench as bench
from research.gpu_sim_spike import v8_split_units as units

PIN = "a" * 40


def row(mix: str = "frozen", worlds: int = 1024, canary: bool = False) -> dict:
    n = 1 if canary else 3
    result = {
        "mix": mix,
        "worlds": worlds,
        "repeats": n,
        "warmups": n,
        "verdict": "PASS_CELL",
        "arm_orders": [list(o) for o in bench.ORDERS[:n]],
        "feature_call_cpu_parity": True,
        "same_action_mechanics_byte_identical": True,
        "full_inventory_compared": True,
        "g_gv_byte_identical": True,
        "state_carry_byte_identical": True,
        "cpu_oracle": {
            a: {
                "vector61": {"match": True},
                "v8": [{"match": True, "feature_planes_match": True}] * 6,
            }
            for a in ("G", "GV")
        },
    }
    for arm, value in (("R", 6.0), ("G", 2.0), ("GV", 1.9)):
        result[arm] = {
            "plain_ticks": [value] * n,
            "digests": ["f" * 64] * n,
            "stages": [
                {
                    "repeat": i,
                    "wall_seconds": value,
                    "exclusive_cuda_elapsed_seconds": {
                        k: value
                        for k in (
                            "vector61_other",
                            "vector61_flood_fill",
                            "b8_v8_features",
                            "opponent_v8_bundle",
                            "movement_collision_events",
                            "policy_forward",
                        )
                    },
                }
                for i in range(n)
            ],
        }
    result["stage_seconds"] = bench.statistics_for({a: result[a] for a in bench.ARMS}, "cuda")
    return result


def matrix() -> list[dict]:
    return [row(unit.split("-")[1], int(unit.rsplit("e", 1)[1])) for unit in units.allocation()]


def export(tmp_path: Path, rows: list[dict], *, canary: bool = False) -> Path:
    target = tmp_path / "exports"
    units.publish_json(target / "native-parity.json", {"commit": PIN, "status": "PASS"})
    cells = units.allocation(canary=canary)
    for value in rows:
        unit = f"cell-{value['mix']}-e{value['worlds']}"
        units.publish_unit(target, units.namespace(PIN, canary=canary), cells, unit, value)
    return target


def test_allocation_exact_order_and_namespaces() -> None:
    assert units.allocation() == tuple(
        f"cell-{m}-e{w}" for w in (1024, 256, 64) for m in ("frozen", "mixed", "scripted")
    )
    assert (
        len(
            {
                units.namespace(PIN),
                units.namespace(PIN, canary=True),
                units.namespace(PIN, smoke=True),
            }
        )
        == 3
    )


def test_arm_switches_independent() -> None:
    import os

    for name, geometry, v8 in [("R", "torch", "0"), ("G", "native", "0"), ("GV", "native", "1")]:
        with bench._arm(name):
            assert os.environ["GPU_SIM_VECTOR_GEOMETRY"] == geometry
            assert os.environ["GPU_SIM_V8_INDEXED"] == v8
            assert os.environ["GPU_SIM_VECTOR_FLOOD"] == "static"
    assert bench.ORDERS == (("R", "G", "GV"), ("G", "GV", "R"), ("GV", "R", "G"))


def test_reducer_selects_gv_and_ties_inconsistency_select_g(tmp_path: Path) -> None:
    rows = matrix()
    target = export(tmp_path / "a", rows)
    assert analyze.aggregate([target], PIN)["verdict"] == "SELECT_GV"
    for index, repeat in ((0, 1), (4, 0)):
        altered = copy.deepcopy(rows)
        if index == 0:
            altered[index]["GV"]["plain_ticks"][repeat] = 2.0
        else:
            altered[index]["GV"]["plain_ticks"] = [2.1] * 3
        assert bench.select(altered) == "SELECT_G"
        target = export(tmp_path / str(index), altered)
        assert analyze.aggregate([target], PIN)["verdict"] == "SELECT_G"


def test_exact_threshold_and_stop_with_registered_prefix(tmp_path: Path) -> None:
    value = row()
    value["G"]["plain_ticks"] = [3.0] * 3
    assert bench.decide(value) == "PASS_CELL"
    value["G"]["plain_ticks"] = [3.0000001] * 3
    value["verdict"] = bench.decide(value)
    assert value["verdict"] == "STOP_SPLIT"
    target = export(tmp_path / "prefix", [value])
    assert analyze.aggregate([target], PIN)["verdict"] == "STOP_SPLIT"
    missing = row("mixed")
    missing["G"]["plain_ticks"] = [4.0] * 3
    missing["verdict"] = bench.decide(missing)
    target = export(tmp_path / "hole", [missing])
    assert analyze.aggregate([target], PIN)["verdict"] == "INCOMPLETE"


def test_invalid_precedes_stop_and_missing(tmp_path: Path) -> None:
    value = row()
    value["verdict"] = "INVALID"
    target = export(tmp_path, [value])
    assert analyze.aggregate([target], PIN)["verdict"] == "INVALID"


@pytest.mark.parametrize("fault", ["nan", "repeat", "arithmetic", "short"])
def test_malformed_timing_is_incomplete(tmp_path: Path, fault: str) -> None:
    value = row()
    if fault == "nan":
        value["G"]["plain_ticks"][0] = float("inf")
        with pytest.raises(ValueError):
            analyze.validate_row(value, "cell-frozen-e1024", canary=False)
        return
    if fault == "repeat":
        value["repeats"] = 4
    elif fault == "arithmetic":
        value["stage_seconds"]["vector61_non_flood"]["G"] = 0.0
    else:
        value["GV"]["plain_ticks"].pop()
    target = export(tmp_path, [value])
    assert analyze.aggregate([target], PIN)["verdict"] == "INCOMPLETE"


def test_canary_coverage_and_skip_exactly_verified(tmp_path: Path) -> None:
    rows = [row(m, 64, True) for m in ("scripted", "frozen", "mixed")]
    target = export(tmp_path / "run", rows, canary=True)
    assert analyze.aggregate([target], PIN, canary=True)["verdict"] == "PASS_CANARY"
    ledger = tmp_path / "ledger/verified-ledger.json"
    units.build_ledger(target, ledger, commit=PIN, canary=True)
    loaded = units.load_ledger(
        ledger, units.namespace(PIN, canary=True), units.allocation(canary=True)
    )
    assert len(loaded) == 3
    assert list(loaded) == sorted(loaded)
    with pytest.raises(ValueError, match="foreign"):
        units.load_ledger(ledger, units.namespace(PIN), units.allocation())
    with pytest.raises(ValueError, match="duplicate"):
        analyze.aggregate([target, target], PIN, canary=True)
    value = json.loads(ledger.read_text())
    payload = ledger.parent / value["units"][0]["path"]
    payload.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        units.load_ledger(ledger, units.namespace(PIN, canary=True), units.allocation(canary=True))


def test_hash_corruption_is_incomplete_and_native_invalid_precedes_missing(tmp_path: Path) -> None:
    target = export(tmp_path / "corrupt", [row()])
    payload = units.payload_path(target, units.allocation()[0])
    payload.write_text("{}")
    result = analyze.aggregate([target], PIN)
    assert result["verdict"] == "INCOMPLETE"
    assert "checksum" in result["ingestion_errors"][0]["error"]
    failed = tmp_path / "native-failed"
    units.publish_json(failed / "native-parity.json", {"commit": PIN, "status": "INVALID"})
    assert analyze.aggregate([failed], PIN)["verdict"] == "INVALID"
