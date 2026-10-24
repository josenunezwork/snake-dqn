"""Prospective V8 packet admission and per-arm routing checks."""

import copy
import os

from research.gpu_sim_spike.v8_bench import _arm, decide


def _row():
    return {
        "worlds": 1024,
        "same_action_mechanics_byte_identical": True,
        "feature_call_cpu_parity": True,
        "cpu_oracle": {"vector61": {"match": True}},
        "state_carry_byte_identical": False,
        "stage_seconds": {"vector61_non_flood": {"speedup": 2.0}, "v8_combined": {"speedup": 2.0}},
    }


def test_stage_bar_exact_boundary_and_nonfinite_refusal():
    row = _row()
    assert decide(row) == "PASS_CELL"
    for bad in (1.999999, float("nan"), float("inf"), None):
        candidate = copy.deepcopy(row)
        candidate["stage_seconds"]["v8_combined"]["speedup"] = bad
        assert decide(candidate) == "STOP_ROUTE"
    row["same_action_mechanics_byte_identical"] = False
    assert decide(row) == "INVALID"


def test_repairs_do_not_waive_cpu_or_common_action_proof():
    row = _row()
    row["observed_declared_cpu_repair_columns"] = [4, 46]
    assert decide(row) == "PASS_CELL"
    row["feature_call_cpu_parity"] = False
    assert decide(row) == "INVALID"
    assert decide({"verdict": "INVALID"}) == "INVALID"


def test_arm_switches_both_native_paths_and_restores_environment(monkeypatch):
    monkeypatch.setenv("GPU_SIM_V8_INDEXED", "old")
    monkeypatch.setenv("GPU_SIM_VECTOR_GEOMETRY", "old")
    with _arm(True):
        assert os.environ["GPU_SIM_V8_INDEXED"] == "1"
        assert os.environ["GPU_SIM_VECTOR_GEOMETRY"] == "native"
        with _arm(False):
            assert os.environ["GPU_SIM_V8_INDEXED"] == "0"
            assert os.environ["GPU_SIM_VECTOR_GEOMETRY"] == "torch"
        assert os.environ["GPU_SIM_VECTOR_GEOMETRY"] == "native"
    assert os.environ["GPU_SIM_VECTOR_GEOMETRY"] == "old"
