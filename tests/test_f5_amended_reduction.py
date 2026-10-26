"""Contracts for the prospective F5 reducer amendment."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from research.bl_residual import f5_amended_reduction as subject
from research.bl_residual import f5_timing


def _sha(letter: str) -> str:
    return letter * 64


def _legacy() -> dict:
    return subject.load_original_reduction(
        Path(
            "research/bl_residual/ledger/F5_POST_LONGRUN01_VERIFICATION_20261026/"
            "TIMING_REDUCTION.json"
        )
    )


def _coverage() -> dict:
    roles = {}
    for role in subject.REQUIRED_ROLES:
        fixtures = [675] if role == "common_natural_h5000" else [673]
        if role in ("h300_continuation", "watch_respawn_14400"):
            fixtures = [672]
        row = {"complete": True, "fixture_indices": fixtures, "source_review_sha256": _sha("c")}
        if role == "h300_continuation":
            row["horizon"] = 300
        if role == "watch_respawn_14400":
            row["frames"] = 14_400
        roles[role] = row
    return {"roles": roles}


def _supplement() -> dict:
    crossings = []
    canonical = {row["allocation_index"]: row for row in f5_timing._canonical_timing_parents()}
    legacy_kernel = _legacy()["kernel_strata"][0]
    for index in range(672, 684):
        fixture = canonical[index]
        common = {
            "schema": "bl-f5-targeted-long-body-supplement-draft/v1",
            "component_id": "f5-targeted-long-body-first-crossing-v1",
            "component_key": f"f5-targeted-long-body-first-crossing-v1/{index}",
            "allocation_index": index,
            "parent_id": fixture["parent_id"],
            "stream": fixture["group"].split("/", 1)[1],
            "branch_source_seam": (
                "scripted-batched-one-action"
                if fixture["group"] == "timing/scripted"
                else "common-native-serial-one-action"
            ),
            "horizon": 5000,
            "long_body_minimum": 700,
            "fit_usable": False,
            "science_admission": False,
            "fixture_descriptor_sha256": _sha("d"),
            "source_provenance": {
                "source_sha256": _sha("e"),
                "environment_sha256": _sha("f"),
                "native_v6": True,
            },
            "kernel_provenance": {
                "assignment_raw_sha256": _sha("0"),
                "platform": legacy_kernel["platform"],
                "release": legacy_kernel["release"],
            },
        }
        if index == 672:
            crossings.append(
                {
                    **common,
                    "crossing": "ABSENT_WITHIN_H5000",
                    "crossing_decision": None,
                }
            )
        else:
            crossings.append(
                {
                    **common,
                    "crossing": "PRESENT_FIRST_WITHIN_H5000",
                    "crossing_decision": 99,
                    "crossing_length": 700,
                    "snapshot_restore_seconds": 0.01,
                    "feature_seconds": 0.01,
                    "feature_shape": [1, 7, 31, 31],
                    "setup_seconds": 0.01,
                    "root_length": 700,
                    "root_length_before_setup": 700,
                    "root_length_after_setup": 700,
                    "all_actions_seconds": 0.03,
                    "legal_action_count": 3,
                    "legal_actions": [0, 1, 2],
                    "branch_units": [
                        {"action": action, "seconds": 0.01, "frames": 300, "full_h300": True}
                        for action in (0, 1, 2)
                    ],
                    "all_actions_units": [
                        {"action": action, "frames": 300, "full_h300": True} for action in (0, 1, 2)
                    ],
                    "individual_actions_seconds": 0.03,
                }
            )
    result = {
        "schema": "bl-f5-targeted-long-body-supplement-draft/v1",
        "records": crossings,
    }
    for record in crossings:
        record["operational_payload_sha256"] = subject._canonical_sha256(
            record, "operational_payload_sha256"
        )
    return result


def _schedule() -> dict:
    return {
        "l4_assignments_at_once": 1,
        "host_workers": 6,
        "math_threads": 1,
        "global_max_completion_rounds": 2,
        "runtime_constraints_status": "PENDING_QUALIFICATION",
        "shards": [
            {
                "ordinal": index,
                "natural": 8,
                "root": 920,
                "branch_action": 5520,
                "natural_range": [index * 8, index * 8 + 8],
                "root_range": [index * 920, index * 920 + 920],
                "branch_action_range": [index * 5520, index * 5520 + 5520],
                "base_hours": 6,
                "effective_hours": 12,
                "reserved_setup_export_imbalance_seconds": 1639.9866971273159,
            }
            for index in range(16)
        ],
    }


def _kwargs() -> dict:
    return {
        "coverage": _coverage(),
        "supplement": _supplement(),
        "forecast": {
            "branch_mapping": {"kind": "conservative_full_bundle_per_action", "divisor": 1},
            "forecast_charged_historical_seconds": {"current_series": 2239.787154037055},
            "separate_historical_exposure_seconds": {
                "prior_timing": 6811.840706833173,
                "current_timing": 4090.2527135000564,
                "canary": 3849.601884124102,
                "known_total": 14751.695304457331,
            },
            "new_overhead_seconds": {
                "setup": 10.0,
                "qualification": 10.0,
                "plan_cache": 10.0,
                "export_io": 10.0,
                "imbalance": 10.0,
            },
        },
        "schedule": _schedule(),
        "provenance": {
            "ruling_sha256": _sha("1"),
            "original_reduction_sha256": subject.PUBLISHED_TIMING_REDUCTION_SHA256,
            "amendment_sha256": _sha("3"),
            "expected_supplement_source_sha256": _sha("e"),
            "expected_supplement_environment_sha256": _sha("f"),
            "expected_fixture_descriptor_sha256": _sha("d"),
            "supplement_content_sha256": subject._canonical_sha256(_supplement()),
        },
    }


def test_amended_reduction_retains_old_hold_immutable_units_and_maxima() -> None:
    legacy = _legacy()
    original = copy.deepcopy(legacy)
    result = subject.reduce_amended(legacy, **_kwargs())
    assert legacy == original
    assert result["status"] == "HOLD_PENDING_INDEPENDENT_TARGETED_TIMING_ACCEPTANCE"
    assert result["original_reduction"] == original
    assert result["original_hold_retained"] is True
    assert result["scientific_recipe_or_loss_changed"] is False
    assert result["archive_forecast"]["base_guard_hours"] == 96
    assert result["archive_forecast"]["effective_guard_hours"] == 192


@pytest.mark.parametrize(
    "mutate",
    [
        lambda args: args["coverage"]["roles"]["watch_respawn_14400"].update(
            {"fixture_indices": [675]}
        ),
        lambda args: args["coverage"]["roles"]["common_natural_h5000"].update(
            {"fixture_indices": [672]}
        ),
        lambda args: args["supplement"]["records"].pop(),
        lambda args: args["supplement"]["records"][1]["branch_units"][0].update(
            {"full_h300": False}
        ),
        lambda args: args["forecast"]["branch_mapping"].update({"divisor": 6}),
        lambda args: args["schedule"].update({"l4_assignments_at_once": 2}),
        lambda args: args["schedule"].update({"global_max_completion_rounds": 3}),
        lambda args: args["provenance"].update({"supplement_content_sha256": _sha("f")}),
        lambda args: args["schedule"]["shards"][0].update({"root_range": [1, 921]}),
    ],
)
def test_refusal_boundaries_hold_for_substitution_unmeasured_seams_and_placement(mutate) -> None:
    args = _kwargs()
    mutate(args)
    with pytest.raises(subject.AmendedReductionHold):
        subject.reduce_amended(_legacy(), **args)


def test_conservative_branch_charge_and_complete_accounting_are_explicit() -> None:
    result = subject.reduce_amended(_legacy(), **_kwargs())
    forecast = result["archive_forecast"]
    assert forecast["workload"] == {"natural": 128, "root": 14720, "branch_action": 88320}
    assert forecast["workers"] == 6
    assert forecast["separate_historical_exposure_seconds"] == {
        "prior_timing": 6811.840706833173,
        "current_timing": 4090.2527135000564,
        "canary": 3849.601884124102,
        "known_total": 14751.695304457331,
    }
    assert subject.primitive_definitions()["t_branch_bundle"]["default_divisor"] == 1
    assert result["scheduled_execution"]["shards"][0]["effective_hours"] == 12


def test_actual_all_actions_bundle_can_raise_the_conservative_branch_bound() -> None:
    args = _kwargs()
    for record in args["supplement"]["records"]:
        if record["crossing"] == "PRESENT_FIRST_WITHIN_H5000":
            record["all_actions_seconds"] = 22.5
            record["operational_payload_sha256"] = subject._canonical_sha256(
                record, "operational_payload_sha256"
            )
    args["provenance"]["supplement_content_sha256"] = subject._canonical_sha256(args["supplement"])
    with pytest.raises(subject.AmendedReductionHold, match="shard maximum"):
        subject.reduce_amended(_legacy(), **args)


@pytest.mark.parametrize("field", ["max_seconds", "immutable_units"])
def test_published_old_reduction_content_and_annex_cannot_be_forged(field: str) -> None:
    legacy = _legacy()
    if field == "max_seconds":
        legacy[field]["t_branch"] += 1.0
    else:
        legacy[field][0]["sha256"] = _sha("f")
    with pytest.raises(subject.AmendedReductionHold):
        subject.reduce_amended(legacy, **_kwargs())


def test_rehashed_record_cannot_substitute_its_source_provenance() -> None:
    args = _kwargs()
    record = args["supplement"]["records"][1]
    record["source_provenance"]["source_sha256"] = _sha("9")
    record["operational_payload_sha256"] = subject._canonical_sha256(
        record, "operational_payload_sha256"
    )
    args["provenance"]["supplement_content_sha256"] = subject._canonical_sha256(args["supplement"])
    with pytest.raises(subject.AmendedReductionHold, match="source, environment, or fixture"):
        subject.reduce_amended(_legacy(), **args)


def _rehash_supplement(args: dict) -> None:
    for record in args["supplement"]["records"]:
        record["operational_payload_sha256"] = subject._canonical_sha256(
            record, "operational_payload_sha256"
        )
    args["provenance"]["supplement_content_sha256"] = subject._canonical_sha256(args["supplement"])


@pytest.mark.parametrize("field", ["environment_sha256", "source_sha256"])
def test_rehashed_foreign_identity_is_still_refused(field: str) -> None:
    args = _kwargs()
    args["supplement"]["records"][0]["source_provenance"][field] = _sha("9")
    _rehash_supplement(args)
    with pytest.raises(subject.AmendedReductionHold, match="source, environment, or fixture"):
        subject.reduce_amended(_legacy(), **args)


def test_role_cannot_claim_an_absent_crossing() -> None:
    args = _kwargs()
    args["coverage"]["roles"]["long_body_root"]["fixture_indices"] = [672]
    with pytest.raises(subject.AmendedReductionHold, match="absent"):
        subject.reduce_amended(_legacy(), **args)


def test_terminal_branches_are_completed_measurements_but_need_full_h300_per_path() -> None:
    args = _kwargs()
    for record in args["supplement"]["records"]:
        if record["crossing"] == "PRESENT_FIRST_WITHIN_H5000":
            record["branch_units"][0].update(frames=1, full_h300=False)
            record["all_actions_units"][0].update(frames=1, full_h300=False)
    _rehash_supplement(args)
    subject.reduce_amended(_legacy(), **args)
    for record in args["supplement"]["records"]:
        if record["crossing"] == "PRESENT_FIRST_WITHIN_H5000":
            for unit in record["branch_units"] + record["all_actions_units"]:
                unit.update(frames=1, full_h300=False)
    _rehash_supplement(args)
    with pytest.raises(subject.AmendedReductionHold, match="complete H300"):
        subject.reduce_amended(_legacy(), **args)


def test_one_new_unbounded_kernel_cannot_borrow_another_stratum() -> None:
    args = _kwargs()
    record = next(
        r for r in args["supplement"]["records"] if r["crossing"] == "PRESENT_FIRST_WITHIN_H5000"
    )
    record["kernel_provenance"]["release"] = "foreign-kernel"
    _rehash_supplement(args)
    with pytest.raises(subject.AmendedReductionHold, match="kernel stratum"):
        subject.reduce_amended(_legacy(), **args)


def test_whole_stage_carries_all_scheduled_overhead_reservations_once() -> None:
    args = _kwargs()
    for key in args["forecast"]["new_overhead_seconds"]:
        args["forecast"]["new_overhead_seconds"][key] = 0.0
    result = subject.reduce_amended(_legacy(), **args)["archive_forecast"]
    assert result["charged_new_overhead_seconds"] == pytest.approx(24000)
    assert result["total_seconds"] == pytest.approx(
        result["compute_seconds"] + 2239.787154037055 + 24000
    )
