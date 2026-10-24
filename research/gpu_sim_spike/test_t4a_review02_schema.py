"""Review-02 regressions for evidence schema and byte-witness admission."""

from __future__ import annotations

import argparse

import pytest
import torch

from research.gpu_sim_spike import t4a_compare as compare


def test_dtype_difference_is_schema_incompatibility_not_byte_mismatch() -> None:
    result = compare.compare(
        torch.zeros((4, 2), dtype=torch.int64),
        torch.zeros((4, 2), dtype=torch.int32),
        "sim.bodies",
    )
    assert isinstance(result, compare.SchemaIncompatibility)
    assert result.reason == "dtype_or_shape"


def test_inventory_difference_is_schema_incompatibility_not_byte_mismatch() -> None:
    result = compare.compare_inventory(
        {"sim.frame": torch.zeros(4, dtype=torch.int64)},
        {"sim.rng_counter": torch.zeros(4, dtype=torch.int64)},
    )
    assert isinstance(result, compare.SchemaIncompatibility)
    assert result.reason == "field_inventory"


def test_later_schema_failure_precedes_earlier_byte_delta() -> None:
    result = compare.compare_inventory(
        {
            "a": torch.tensor([0], dtype=torch.uint8),
            "z": torch.tensor([0], dtype=torch.int64),
        },
        {
            "a": torch.tensor([1], dtype=torch.uint8),
            "z": torch.tensor([0], dtype=torch.int32),
        },
    )
    assert isinstance(result, compare.SchemaIncompatibility)
    assert result.field == "z"


@pytest.mark.parametrize("fault", ("dtype", "inventory"))
def test_completed_unit_schema_fault_is_invalid_evidence(tmp_path, fault) -> None:
    from research.gpu_sim_spike.test_t4a_evidence import _tree, reduce

    def mutate(content):
        captured = content["records"][1]["field_digests"][0]
        if fault == "dtype":
            captured["sim.bodies"]["dtype"] = "torch.int32"
        else:
            captured.pop("sim.rng_counter")

    result = reduce(_tree(tmp_path, mutate=mutate))
    assert result["verdict"] == "INVALID_EVIDENCE"
    assert result["invalid_evidence"]


@pytest.mark.parametrize("fault", ("dtype", "inventory", "ticks"))
def test_actual_failure_writer_classifies_schema_fault_as_harness_invalid(
    tmp_path, monkeypatch, fault
) -> None:
    from research.gpu_sim_spike import t4a_analyze, t4a_bench
    from research.gpu_sim_spike.test_t4a_evidence import COMMIT

    left = [{"sim.frame": torch.zeros(4, dtype=torch.int64)}]
    right = [{"sim.frame": torch.zeros(4, dtype=torch.int32)}]
    if fault == "inventory":
        right = [{"sim.rng_counter": torch.zeros(4, dtype=torch.int64)}]
    elif fault == "ticks":
        right = []

    def exercise_real_comparator(_args):
        t4a_bench._compare_tapes(
            left, right, roster="frozen", worlds=4, layout="canonical", replay=0
        )
        pytest.fail("incompatible captured evidence was admitted")

    monkeypatch.setattr(t4a_bench, "_run", exercise_real_comparator)
    monkeypatch.setattr(t4a_bench, "source_commit", lambda: COMMIT)
    monkeypatch.setattr(t4a_bench, "source_hashes", lambda: {})
    failure = t4a_bench.run(argparse.Namespace(output_dir=tmp_path, smoke=False, canary=False))
    assert failure["status"] == "INVALID_EVIDENCE"
    result = t4a_analyze.aggregate(tmp_path, COMMIT)
    assert result["verdict"] == "INVALID_EVIDENCE"
    assert not result["mismatches"]


@pytest.mark.parametrize("field", ("sim.frame", "output.q.0", "output.actions"))
def test_internal_event_failure_preserves_real_mismatch_over_missing_units(
    tmp_path, monkeypatch, field
) -> None:
    from research.gpu_sim_spike import t4a_analyze, t4a_bench, t4a_events
    from research.gpu_sim_spike.test_t4a_evidence import COMMIT

    dtype = torch.float32 if field.startswith("output.q.") else torch.int64
    shape = (64,) if field == "sim.frame" else (64, 6)
    left = {field: torch.zeros(shape, dtype=dtype)}
    right = {field: left[field].clone()}
    right[field].view(torch.uint8).reshape(64, -1)[63, 0] ^= 1
    found = compare.compare_inventory(left, right)
    witness = t4a_events._internal_byte_witness(found, left, right, 64)
    pair = witness["comparison_records"]
    descriptor = pair[0]["field_digests"][0][field]
    error = t4a_bench.T4AMismatch(
        witness,
        fields=[field],
        field_types={field: {"dtype": descriptor["dtype"], "shape": descriptor["shape"]}},
        comparison_records=pair,
    )

    def interrupted(_args):
        raise error

    monkeypatch.setattr(t4a_bench, "_run", interrupted)
    monkeypatch.setattr(t4a_bench, "source_commit", lambda: COMMIT)
    monkeypatch.setattr(t4a_bench, "source_hashes", lambda: {})
    failure = t4a_bench.run(argparse.Namespace(output_dir=tmp_path, smoke=False, canary=False))
    assert failure["status"] == "INVALID_SELF_CONSISTENCY"
    result = t4a_analyze.aggregate(tmp_path, COMMIT)
    assert result["verdict"] == "INVALID_SELF_CONSISTENCY", result
    assert result["incomplete"] and not result["invalid_evidence"]


def test_missing_internal_q_raw_tape_is_invalid_evidence() -> None:
    from research.gpu_sim_spike import t4a_analyze, t4a_events

    left = {"output.q.0": torch.zeros((4, 6), dtype=torch.float32)}
    right = {"output.q.0": torch.ones((4, 6), dtype=torch.float32)}
    witness = t4a_events._internal_byte_witness(
        compare.compare_inventory(left, right), left, right, 4
    )
    for record in witness["comparison_records"]:
        record["action_q_tape"] = []
    descriptor = witness["expected_descriptor"]
    assert (
        t4a_analyze._witness_error(
            witness,
            ["output.q.0"],
            {"output.q.0": {"dtype": descriptor["dtype"], "shape": descriptor["shape"]}},
            witness["comparison_records"],
        )
        == "witness action/Q bytes are absent or malformed"
    )


def test_tick1_event_failure_uses_referenced_capture_schema() -> None:
    from research.gpu_sim_spike import t4a_analyze, t4a_bench, t4a_events

    field = "event.death_respawn.respawned"
    left = {field: torch.zeros(4, dtype=torch.bool)}
    right = {field: torch.ones(4, dtype=torch.bool)}
    witness = t4a_events._internal_byte_witness(
        compare.compare_inventory(left, right), left, right, 4
    )
    for record, worlds in zip(witness["comparison_records"], (4, 64)):
        record.update(worlds=worlds, layout="event", replay=0)
        record["field_digests"].insert(0, {})
    for side, worlds in (("reference", 4), ("candidate", 64)):
        witness["comparison"][side].update(worlds=worlds, layout="event", replay=0, tick=1)
    witness["tick"] = 1
    error = t4a_bench._event_exception(witness, roster="frozen", event="simultaneous_death_respawn")
    assert error.fields == [field]
    assert (
        t4a_analyze._witness_error(
            error.witness, error.fields, error.field_types, error.comparison_records
        )
        is None
    )
    witness["comparison"]["candidate"]["tick"] = 0
    assert (
        t4a_analyze._witness_error(
            witness, error.fields, error.field_types, error.comparison_records
        )
        == "witness compares different execution ticks"
    )
