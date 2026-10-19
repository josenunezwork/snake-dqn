"""Focused non-allocating tests for the S1 admission boundary."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from research.s1_mature_20261018 import admission, run


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
    receipt = tmp_path / "pricing.json"
    receipt.write_text(
        __import__("json").dumps(
            {
                "status": "PASS",
                "complete_upper_bound": {"mac_worker_hours": number, "elapsed_hours": 1},
                "full_replay_transport": {
                    "status": "PASS",
                    "source_sha256": "a",
                    "timing_receipt_sha256": "b",
                    "snapshot_bytes": 4 * 1024**3,
                    "per_run_gpu_hours_upper_bound": 1,
                },
            }
        )
    )
    with pytest.raises(admission.AdmissionRefused):
        admission.priced_mac_bound(receipt)
