"""Offline contracts for the unsealed F5 long-body timing supplement draft."""

from __future__ import annotations

import copy
import types

import pytest

from research.bl_residual import f5_r10a_annex
from research.bl_residual import f5_timing_supplement as subject


def _fixtures() -> list[dict]:
    return [row for row in f5_r10a_annex.allocate(0) if row["timing_only"]]


def _present_record() -> dict:
    row = _fixtures()[3]
    record = {
        "schema": subject.SCHEMA,
        "component_id": subject.COMPONENT_ID,
        "component_key": f"{subject.COMPONENT_ID}/{row['allocation_index']}",
        "allocation_index": row["allocation_index"],
        "parent_id": row["parent_id"],
        "stream": "scripted",
        "branch_source_seam": "scripted-batched-one-action",
        "horizon": 5000,
        "long_body_minimum": 700,
        "fit_usable": False,
        "science_admission": False,
        "fixture_descriptor_sha256": "a" * 64,
        "source_provenance": {
            "source_sha256": "b" * 64,
            "environment_sha256": "c" * 64,
            "native_v6": True,
        },
        "kernel_provenance": {
            "assignment_raw_sha256": "d" * 64,
            "platform": "Linux-test",
            "release": "test-kernel",
        },
        "crossing": "PRESENT_FIRST_WITHIN_H5000",
        "crossing_decision": 17,
        "crossing_length": 700,
        "snapshot_restore_seconds": 0.1,
        "feature_seconds": 0.1,
        "feature_shape": [1, 7, 31, 31],
        "setup_seconds": 0.1,
        "root_length": 700,
        "root_length_before_setup": 700,
        "root_length_after_setup": 700,
        "legal_action_count": 2,
        "legal_actions": [1, 4],
        "branch_units": [
            {"action": 1, "seconds": 0.1, "frames": 300, "full_h300": True},
            {"action": 4, "seconds": 0.2, "frames": 299, "full_h300": False},
        ],
        "all_actions_seconds": 0.3,
        "all_actions_units": [
            {"action": 1, "frames": 300, "full_h300": True},
            {"action": 4, "frames": 299, "full_h300": False},
        ],
        "individual_actions_seconds": 0.3,
    }
    record["operational_payload_sha256"] = subject.operational_payload_sha256(record)
    return record


def _absent_record() -> dict:
    row = _fixtures()[0]
    record = {
        "schema": subject.SCHEMA,
        "component_id": subject.COMPONENT_ID,
        "component_key": f"{subject.COMPONENT_ID}/{row['allocation_index']}",
        "allocation_index": row["allocation_index"],
        "parent_id": row["parent_id"],
        "stream": "watch6",
        "branch_source_seam": "common-native-serial-one-action",
        "horizon": 5000,
        "long_body_minimum": 700,
        "fit_usable": False,
        "science_admission": False,
        "fixture_descriptor_sha256": "a" * 64,
        "source_provenance": {
            "source_sha256": "b" * 64,
            "environment_sha256": "c" * 64,
            "native_v6": True,
        },
        "kernel_provenance": {
            "assignment_raw_sha256": "d" * 64,
            "platform": "Linux-test",
            "release": "test-kernel",
        },
        "crossing": "ABSENT_WITHIN_H5000",
        "crossing_decision": None,
    }
    record["operational_payload_sha256"] = subject.operational_payload_sha256(record)
    return record


def test_draft_plan_is_finite_unsealed_and_retains_all_original_fixture_roles() -> None:
    plan = subject.draft_plan(_fixtures())
    assert plan["status"] == "DRAFT_UNSEALED_HOLD"
    assert plan["dispatch_authorized"] is False
    assert [row["allocation_index"] for row in plan["fixtures"]] == list(range(672, 684))
    assert plan["construction"]["horizon"] == 5000
    assert plan["source_seams"]["scripted"].startswith("existing scripted")


@pytest.mark.parametrize(
    "trace, expected",
    [
        ([{"decision": 0, "hero_length": 699}, {"decision": 1, "hero_length": 700}], 1),
        ([{"decision": 0, "hero_length": 22}], None),
    ],
)
def test_first_crossing_is_deterministic_and_never_uses_a_later_search(trace, expected) -> None:
    assert subject.crossing_from_trace(trace) == expected


@pytest.mark.parametrize(
    "trace",
    [
        [{"decision": 1, "hero_length": 700}],
        [{"decision": 5000, "hero_length": 700}],
        [{"decision": 0, "hero_length": "700"}],
    ],
)
def test_crossing_refuses_noncanonical_or_extended_natural_traces(trace) -> None:
    with pytest.raises(subject.SupplementHold):
        subject.crossing_from_trace(trace)


def test_fixture_identity_refuses_seed_substitution_or_partial_matrix() -> None:
    rows = _fixtures()
    rows[0]["seed"] += 1
    with pytest.raises(subject.SupplementHold):
        subject.frozen_fixtures(rows)
    with pytest.raises(subject.SupplementHold):
        subject.frozen_fixtures(_fixtures()[:-1])


def test_single_measurement_refuses_a_foreign_seed_before_any_root_is_built() -> None:
    row = _fixtures()[0]
    row["seed"] += 1
    with pytest.raises(subject.SupplementHold):
        subject.measure_fixture(
            row, checkpoint=object(), root_builder=lambda *_: (None, None, None)
        )


def test_operational_record_preserves_per_action_mapping_and_terminal_branch_status() -> None:
    record = _present_record()
    subject.validate_operational_record(record)
    assert record["branch_units"][1]["full_h300"] is False
    bad = copy.deepcopy(record)
    bad["branch_units"] = bad["branch_units"][:1]
    bad["operational_payload_sha256"] = subject.operational_payload_sha256(bad)
    with pytest.raises(subject.SupplementHold):
        subject.validate_operational_record(bad)


@pytest.mark.parametrize("field", ["loss", "label", "outcome", "strict_ledger"])
def test_operational_record_refuses_science_or_strict_ledger_fields(field: str) -> None:
    record = _present_record()
    record[field] = "forbidden"
    with pytest.raises(subject.SupplementHold):
        subject.validate_operational_record(record)


def test_absent_crossing_is_a_retained_operational_disposition() -> None:
    subject.validate_operational_record(_absent_record())


def test_crossing_root_captures_the_selector_prepared_state_before_the_natural_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.bl_residual import gameplay
    from research.t34_shadow import shadow

    class Runtime:
        def prepare(self, prepared: object) -> None:
            prepared.length[0, 0] = 700

    class World:
        def __init__(self) -> None:
            self.sim = types.SimpleNamespace(
                alive=__import__("numpy").array([[True]]), length=__import__("numpy").array([[699]])
            )
            self.runtime = Runtime()
            self.hero = object()

        def clone(self) -> "World":
            return copy.deepcopy(self)

    world = World()
    monkeypatch.setattr(
        subject.f5_qualification, "_source_root_observation", lambda *_: (world, {}, 0.0)
    )
    monkeypatch.setattr(shadow, "clone_world_exact", lambda value: copy.deepcopy(value))
    monkeypatch.setattr(shadow, "full_digest", lambda value: repr(value.sim.length.tolist()))

    observed: dict[str, int] = {}

    def step(value: World, decision: int) -> dict[str, int]:
        value.runtime.prepare(value.sim)
        length = int(value.sim.length[0, 0])
        value.sim.length[0, 0] = 1  # Natural dispatch would mutate after the capture seam.
        observed["post_dispatch_length"] = int(value.sim.length[0, 0])
        return {"hero_length": length}

    monkeypatch.setattr(gameplay, "_step_with_trace", step)
    captured, decision, length = subject._crossing_root(_fixtures()[0], object())
    assert decision == 0
    assert length == 700
    assert captured.sim.length[0, 0] == 700
    assert observed["post_dispatch_length"] == 1


@pytest.mark.parametrize(
    "field,value", [("allocation_index", 999), ("parent_id", "foreign"), ("horizon", 5001)]
)
def test_absent_record_still_requires_frozen_identity(field, value) -> None:
    record = _absent_record()
    record[field] = value
    record["operational_payload_sha256"] = subject.operational_payload_sha256(record)
    with pytest.raises(subject.SupplementHold):
        subject.validate_operational_record(record)


def test_individual_branch_cannot_hide_outcome_fields() -> None:
    record = _present_record()
    record["branch_units"][0]["score"] = 9
    record["operational_payload_sha256"] = subject.operational_payload_sha256(record)
    with pytest.raises(subject.SupplementHold, match="unknown fields"):
        subject.validate_operational_record(record)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(feature_seconds=0.0),
        lambda r: r.update(root_length_after_setup=699),
        lambda r: r.update(individual_actions_seconds=0.5),
        lambda r: r.update(feature_shape=[1, 6, 31, 31]),
    ],
)
def test_primitive_accounting_and_prepared_root_cannot_drift(mutate) -> None:
    record = _present_record()
    mutate(record)
    record["operational_payload_sha256"] = subject.operational_payload_sha256(record)
    with pytest.raises(subject.SupplementHold):
        subject.validate_operational_record(record)
