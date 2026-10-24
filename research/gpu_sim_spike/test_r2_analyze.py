"""Synthetic-export checks of the pre-registered R2 reducer and ledger contract."""

from __future__ import annotations

import pytest

from research.gpu_sim_spike import r2_analyze, r2_units

COMMIT = "a" * 40


def _content(unit: str, i_tick: float, p_tick: float, rate: float = 3000.0) -> dict:
    kind, _mix, worlds = r2_units.parse(unit)
    arms = {
        "I": {"plain_seconds": [i_tick] * 3, "digests": ["d"] * 3, "stages": []},
        "I_prime": {"plain_seconds": [p_tick] * 3, "digests": ["d"] * 3, "stages": []},
    }
    continuous = {"status": "NOT_REQUIRED"}
    if kind == "cell" and worlds == 1024:
        block = {"final_digest": "f", "complete_world_steps_per_second": rate}
        continuous = {
            "status": "COMPLETE",
            "arms": {
                "I": {"repeats": [dict(block, complete_world_steps_per_second=1400.0)] * 3},
                "I_prime": {"repeats": [block] * 3},
            },
        }
    return {
        "identity_proof": {"status": "PASS", "candidate_debug": {"debug_checks": 9}},
        "pure_optimization_identity": True,
        "arms": arms,
        "continuous": continuous,
    }


def _tree(tmp_path, *, gain=0.5, exc=1.0, rate=3000.0, drop=(), mutate=None):
    root = tmp_path / "exports"
    cells = r2_units.allocation()
    ns = r2_units.namespace(COMMIT)
    for unit in cells:
        if unit in drop:
            continue
        i_tick = 0.7
        p_tick = i_tick * (1 - gain) if unit.startswith("cell") else i_tick * exc
        content = _content(unit, i_tick, p_tick, rate)
        if mutate and unit == mutate[0]:
            mutate[1](content)
        r2_units.publish_unit(root, ns, cells, unit, content)
    r2_units.build_ledger(root, root / "verified-ledger.json", commit=COMMIT)
    for name in ("native-parity.json", "r2-native-parity.json"):
        r2_units.publish_json(
            root / name,
            {"status": "PASS", "native_cuda_executed": True, "device": "cuda", "commit": COMMIT},
        )
    return root


def test_allocation_keeps_i0_order_then_exception_packet():
    main = r2_units.allocation()
    assert main[:3] == ("cell-frozen-e1024", "cell-mixed-e1024", "cell-scripted-e1024")
    assert main[-3:] == ("exc-frozen-e256", "exc-mixed-e256", "exc-scripted-e256")
    assert r2_units.allocation(canary=True)[-1] == "exc-scripted-e64"
    assert r2_units.parse("exc-mixed-e3") == ("exc", "mixed", 3)
    with pytest.raises(ValueError):
        r2_units.parse("cell-mixed-e2048")


def test_advance_and_q4_pass(tmp_path):
    value = r2_analyze.aggregate([_tree(tmp_path)], COMMIT)
    assert value["verdict"] == "R2_ADVANCE" and value["q4_speed"] == "PASS_SPEED"
    assert set(value["descriptive_e6_floor_seconds"]) and value["identity_errors"] == []


def test_gain_below_fifteen_percent_stops(tmp_path):
    value = r2_analyze.aggregate([_tree(tmp_path, gain=0.14)], COMMIT)
    assert value["verdict"] == "R2_STOP"


def test_exception_regression_above_ten_percent_stops(tmp_path):
    value = r2_analyze.aggregate([_tree(tmp_path, exc=1.11)], COMMIT)
    assert value["verdict"] == "R2_STOP" and value["exception_packet_ratio"] > 1.10


def test_q4_requires_every_block_at_2000(tmp_path):
    value = r2_analyze.aggregate([_tree(tmp_path, rate=1999.0)], COMMIT)
    assert value["verdict"] == "R2_ADVANCE" and value["q4_speed"] == "STOP_SPEED"


def test_missing_gating_cell_is_incomplete(tmp_path):
    value = r2_analyze.aggregate([_tree(tmp_path, drop=("exc-mixed-e256",))], COMMIT)
    assert value["verdict"] == "INCOMPLETE"


def test_any_identity_difference_is_invalid(tmp_path):
    def differ(content):
        content["arms"]["I_prime"]["digests"][1] = "other"

    value = r2_analyze.aggregate([_tree(tmp_path, mutate=("cell-e6", differ))], COMMIT)
    assert value["verdict"] == "R2_ADVANCE"  # mutate target absent: control
    root = _tree(tmp_path / "b", mutate=("cell-scripted-e6", differ))
    assert r2_analyze.aggregate([root], COMMIT)["verdict"] == "INVALID"


def test_producer_failure_and_native_failure_are_invalid(tmp_path):
    root = _tree(tmp_path)
    r2_units.publish_json(root / "r2-failure.json", {"status": "INVALID"})
    assert r2_analyze.aggregate([root], COMMIT)["verdict"] == "INVALID"
    other = _tree(tmp_path / "n")
    (other / "r2-native-parity.json").unlink()
    assert r2_analyze.aggregate([other], COMMIT)["verdict"] == "INVALID"
