"""Synthetic contract tests for the sealed F5 timing plan."""

from __future__ import annotations

import copy

import pytest

from research.bl_residual import f5_timing as timing


def _allocation():
    return timing._canonical_timing_parents()


def _proof():
    samples = []
    for row in _allocation():
        samples.append(
            {
                "stream": row["group"].split("/")[1],
                "t_root": 0.01,
                "t_branch": 0.01,
                "t_nat": 0.01,
                "full_h300": True,
                "long_body": True,
                "full_h5000_natural": True,
                "watch_fixture_frames": 14400,
                "paired_fit_seconds": 1,
                "probe_seconds": 1,
                "gameplay_seconds": 1,
                "product_seconds": 1,
                "allocation_index": row["allocation_index"],
                "parent_id": row["parent_id"],
                "timing_only": True,
            }
        )
    return {
        "schema": "bl-f5-timing-proof/v1",
        "status": "COMPLETE",
        "allocation_sha256": timing.ALLOCATION_SHA256,
        "source_identity": {"sha": "source"},
        "environment_identity": {"sha": "environment"},
        "batch_identity": {"sha": "batch"},
        "transport": {
            "upload_chunk_bytes": 2 * 1024 * 1024,
            "upload_workers": 2,
            "export_readers": 1,
            "max_call_seconds": 60,
            "transfer_seconds": 150,
            "staging_seconds": 300,
            "private_input_bytes": 100,
        },
        "input_inventory": [
            {"name": "selected-960.npz", "size": 100, "scope": "private"},
            {"name": "b8.pth", "size": 200, "scope": "public_b8", "release_qualified": True},
        ],
        "placement": {
            "l4_assignments": 1,
            "vcpu": 6,
            "cuda_processes": 1,
            "precision": "fp32",
            "cpu_label_workers": 6,
            "mac_workers": 2,
            "mac_nice": 15,
            "math_threads": 1,
        },
        "samples": samples,
        "forecast": {
            "setup_io_hours": 0.1,
            "max_parent_imbalance_hours": 0.1,
            "stage_setup_io_hours": {
                "paired_fit": 0.1,
                "probe": 0.1,
                "gameplay": 0.1,
                "product": 0.1,
            },
        },
    }


def _validate(proof):
    return timing.validate_operational_proof(
        proof,
        source_identity={"sha": "source"},
        environment_identity={"sha": "environment"},
        batch_identity={"sha": "batch"},
        allocation=_allocation(),
    )


def test_plan_seals_exact_twelve_parent_allocation_and_bounds():
    plan = timing.build_plan(_allocation())
    assert plan["status"] == "PREPARED_NOT_MEASURED"
    assert len(plan["timing_parents"]) == 12
    assert plan["transport"]["transfer_seconds_inclusive"] == 150
    assert plan["stage_envelopes_hours"]["paired_fit_including_canary"] == {
        "base": 2,
        "effective": 4,
    }
    assert plan["selection"]["cov_selected"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.__setitem__("status", "PREPARED_NOT_MEASURED"),
        lambda p: p["source_identity"].__setitem__("sha", "drift"),
        lambda p: p["transport"].__setitem__("transfer_seconds", 150.001),
        lambda p: p["placement"].__setitem__("cuda_processes", 2),
        lambda p: p["samples"].pop(),
        lambda p: p.__setitem__("loss", 0.0),
        lambda p: p["samples"][0].__setitem__("t_root", float("nan")),
        lambda p: p["forecast"]["stage_setup_io_hours"].__setitem__("product", 12.01),
    ],
)
def test_operational_validator_holds_malformed_drift_or_early_proofs(mutator):
    proof = _proof()
    mutator(proof)
    with pytest.raises(timing.TimingHold):
        _validate(proof)


def test_operational_validator_accepts_complete_redacted_synthetic_proof():
    result = _validate(copy.deepcopy(_proof()))
    assert result["status"] == "QUALIFIED"
    assert result["archive_base_hours"] <= 8


def test_plan_refuses_non_disjoint_or_wrong_stream_allocation():
    allocation = _allocation()
    allocation[-1]["parent_id"] = allocation[0]["parent_id"]
    with pytest.raises(timing.TimingHold):
        timing.build_plan(allocation)


def test_operational_validator_rejects_science_or_duplicate_fixture():
    proof = _proof()
    proof["samples"][0]["parent_id"] = "science/fixture"
    with pytest.raises(timing.TimingHold):
        _validate(proof)


def test_operational_validator_allows_smaller_repacked_private_payload():
    assert _validate(_proof())["status"] == "QUALIFIED"


def test_operational_validator_refuses_inventory_overflow_or_unqualified_public_b8():
    proof = _proof()
    proof["transport"]["private_input_bytes"] = 101
    with pytest.raises(timing.TimingHold):
        _validate(proof)


def test_stage_forecasts_expand_complete_fixture_maxima_to_registered_workload():
    proof = _proof()
    for sample in proof["samples"]:
        sample["gameplay_seconds"] = 20
    result = _validate(proof)
    assert result["stage_workload_hours"]["gameplay"] == pytest.approx(3.2)
    proof = _proof()
    for sample in proof["samples"]:
        sample["gameplay_seconds"] = 100
    with pytest.raises(timing.TimingHold):
        _validate(proof)


def test_watch_fixture_is_required_for_watch6_but_not_every_stream():
    proof = _proof()
    for sample in proof["samples"]:
        if sample["stream"] != "watch6":
            sample["watch_fixture_frames"] = 0
    assert _validate(proof)["status"] == "QUALIFIED"
    proof = _proof()
    proof["input_inventory"] = [
        {"name": "oversize.npz", "size": timing.SOURCE_BYTES + 1, "scope": "private"}
    ]
    proof["transport"]["private_input_bytes"] = timing.SOURCE_BYTES + 1
    with pytest.raises(timing.TimingHold):
        _validate(proof)
    proof = _proof()
    proof["input_inventory"][1]["release_qualified"] = False
    with pytest.raises(timing.TimingHold):
        _validate(proof)
