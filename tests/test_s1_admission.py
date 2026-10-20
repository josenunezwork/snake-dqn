"""Focused non-allocating tests for the S1 admission boundary."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from research.s1_mature_20261018 import admission, final_artifacts, run


def _final_transport(tmp_path: Path) -> dict:
    sources = final_artifacts.source_manifest(final_artifacts.SOURCES)
    receipts = {}
    for arm in ("n3", "n1"):
        receipt = tmp_path / f"{arm}-size.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": final_artifacts.SIZE_SCHEMA,
                    "status": "PASS",
                    "arm": arm,
                    "source_manifest": sources,
                    "production_contract_validated": True,
                    "export_contract": final_artifacts.PROTOCOL,
                    "measured_final_bundle_bytes": 13_000_000,
                    "final_bundle_bytes_upper_bound": 64 * 1024**2,
                    "telemetry_log_allowance_bytes": 1_000_000,
                    "artifact_files": sorted(final_artifacts.REQUIRED_ARTIFACT_FILES),
                }
            )
        )
        receipts[arm] = {"path": str(receipt), "sha256": admission.sha256(receipt)}
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "qualified": True,
                "s1_final_artifact_transfer": {
                    "status": "PASS",
                    "hardware": "L4",
                    "protocol": final_artifacts.PROTOCOL,
                    "verified_on_mac": True,
                    "transport_source_sha256": sources["research/s1_mature_20261018/train.py"],
                    "transport_source_manifest": final_artifacts.source_manifest(
                        final_artifacts.CLI_SOURCES
                    ),
                    "exported_bytes": 13_000_000,
                    "elapsed_seconds": 13,
                    "bytes_per_second_lower_bound": 1_000_000,
                    "final_artifact_size_receipts": receipts,
                },
            }
        )
    )
    return {
        "schema": final_artifacts.SCHEMA,
        "status": "PASS",
        "source_manifest": sources,
        "cli_acceptance_receipt": {"path": str(acceptance), "sha256": admission.sha256(acceptance)},
    }


def test_preregistration_requires_prospective_approval_line(tmp_path: Path) -> None:
    draft = tmp_path / "PREREGISTRATION.md"
    draft.write_text("**Status: DRAFT_PENDING_OWNER_REVIEW.**\n")
    with pytest.raises(admission.AdmissionRefused):
        admission.require_approved_preregistration(draft)
    # The old draft text is intentionally retained in the frozen history.
    draft.write_text(
        "**Status: APPROVED_S1_MATURE_20261018.**\nold DRAFT_PENDING_OWNER_REVIEW bytes\n"
    )
    assert admission.require_approved_preregistration(draft)["sha256"] == admission.sha256(draft)


def test_liabilities_reject_wrong_arm_timing() -> None:
    good = []
    for arm, minutes, timeout in (("n3", 30, 60), ("n1", 68, 136)):
        for seed in range(5):
            good.append(
                {
                    "job_id": f"{arm}-{seed}",
                    "env": {"S1_ARM": arm},
                    "requirements": {"est_minutes": minutes},
                    "timeout_minutes": timeout,
                }
            )
    bound = admission.liabilities(good)
    assert len(bound) == 10
    assert not set(bound).intersection({"retry-n3", "retry-n1"})
    assert bound["n3-0"]["gpu_allocation_hours"] == pytest.approx(65 / 60)
    assert bound["n1-0"]["units"] == pytest.approx(141 / 30)
    good[0]["timeout_minutes"] = 136
    with pytest.raises(admission.AdmissionRefused):
        admission.liabilities(good)


def test_final_export_recovery_rejects_path_escape(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("bad")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="../outside")
    with pytest.raises(admission.AdmissionRefused):
        run._safe_extract_final(archive, tmp_path / "extract")


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_pricing_rejects_nonfinite_bounds(tmp_path: Path, number: float) -> None:
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "status": "PASS",
                "counter_admission": "PASS",
                "identity": {
                    "source_manifest": {
                        relative: admission.sha256(admission.ROOT / relative)
                        for relative in (
                            "research/redesign_m3c_20261008/train_m3c.py",
                            "research/s1_mature_20261018/train.py",
                        )
                    }
                },
            }
        )
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schema": "s1-eval-timing/v1",
                "status": "completed",
                "episode_count": 10,
                "identity": {"source_manifest": {}},
            }
        )
    )
    receipt = tmp_path / "pricing.json"
    receipt.write_text(
        __import__("json").dumps(
            {
                "status": "PASS",
                "evidence": {
                    name: {"path": str(item), "sha256": admission.sha256(item)}
                    for name, item in (("training", training), ("evaluation", evaluation))
                },
                "complete_upper_bound": {"mac_worker_hours": number, "elapsed_hours": 1},
                "final_artifact_transport": _final_transport(tmp_path),
            }
        )
    )
    with pytest.raises(admission.AdmissionRefused):
        admission.priced_mac_bound(receipt)


def test_pricing_requires_current_counter_sources_and_bound_final_transfer(tmp_path: Path) -> None:
    sources = {
        relative: admission.sha256(admission.ROOT / relative)
        for relative in (
            "research/redesign_m3c_20261008/train_m3c.py",
            "research/s1_mature_20261018/train.py",
        )
    }
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"status": "PASS", "counter_admission": "PASS"}))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schema": "s1-eval-timing/v1",
                "status": "completed",
                "episode_count": 10,
                "identity": {"source_manifest": {}},
            }
        )
    )

    def priced() -> Path:
        path = tmp_path / "pricing.json"
        path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "evidence": {
                        name: {"path": str(item), "sha256": admission.sha256(item)}
                        for name, item in (("training", training), ("evaluation", evaluation))
                    },
                    "complete_upper_bound": {"mac_worker_hours": 1, "elapsed_hours": 1},
                    "final_artifact_transport": _final_transport(tmp_path),
                }
            )
        )
        return path

    with pytest.raises(admission.AdmissionRefused, match="source identity"):
        admission.priced_mac_bound(priced())
    training.write_text(
        json.dumps(
            {
                "status": "PASS",
                "counter_admission": "PASS",
                "identity": {"source_manifest": sources},
            }
        )
    )
    bound = admission.priced_mac_bound(priced())
    assert bound["per_arm_gpu_hours"]["n3"] > 30 / 60
    assert bound["per_arm_gpu_hours"]["n1"] > 68 / 60


def test_primary_forecast_fits_caps_without_pre_funding_optional_reruns(tmp_path: Path) -> None:
    from research.s1_mature_20261018 import bank, ledger, price

    # This is an accounting unit fixture, not fresh timing/qualification evidence.
    # Historical receipts must keep rejecting source changes after optimizations.
    relative = "src/simd_env/ego_raster_bv8.py"
    timing = {
        "status": "completed",
        "workers": 2,
        "episode_count": 10,
        "parallel_elapsed_seconds": [1.0] * 10,
        "identity": {
            "world_hashes": [
                world["world_semantic_sha256"] for world in bank.load_world_bank()["timing_worlds"]
            ],
            "source_manifest": {relative: admission.sha256(admission.ROOT / relative)},
        },
    }
    evaluation = tmp_path / "synthetic-evaluation.json"
    evaluation.write_text(json.dumps(timing))
    training = tmp_path / "synthetic-training.json"
    training.write_text(
        json.dumps(
            {
                "status": "PASS",
                "counter_admission": "PASS",
                "identity": {
                    "source_manifest": {
                        name: admission.sha256(admission.ROOT / name)
                        for name in (
                            "research/redesign_m3c_20261008/train_m3c.py",
                            "research/s1_mature_20261018/train.py",
                        )
                    }
                },
            }
        )
    )
    forecast = price.forecast(evaluation, training)
    hard = forecast["primary_hard_reservation"]
    assert hard["gpu_allocation_hours"] == pytest.approx(5 * (65 + 141) / 60 + 2)
    assert hard["units"] == pytest.approx(38.333333333333336)
    assert all(hard[key] <= cap for key, cap in ledger.USABLE_CAPS.items())
    assert forecast["conditional_reruns"]["pre_funded"] is False
    assert forecast["conditional_planning_forecast"]["known_base_plus_cleanup_gpu_hours"] == 12.8
    assert forecast["complete_upper_bound"] is None  # Missing live transfer qualification.
    assert forecast["status"] == "REFUSED"
    assert not any("exceeds usable" in reason for reason in forecast["refusal_reasons"])

    timing["identity"]["source_manifest"][relative] = "0" * 64
    evaluation.write_text(json.dumps(timing))
    with pytest.raises(ValueError, match="timing source changed"):
        price.forecast(evaluation, training)
