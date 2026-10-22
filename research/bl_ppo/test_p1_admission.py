"""Adversarial checks for independent review and shared Y3 debit discharge."""

from copy import deepcopy
from pathlib import Path

import pytest

from research.bl_ppo.exports import export_json
from research.bl_ppo.p1_admission import reconcile, verify_source_admission


def receipts(tmp_path: Path) -> list[Path]:
    result = []
    for name, seconds, before, after in (
        ("source-02", 2914.9896361669526, 1131.87, 1130.72),
        ("source-latebody-02", 467.8529948340147, 1130.32, 1130.04),
    ):
        path = tmp_path / name / "colab-output/receipt.json"
        path.parent.mkdir(parents=True)
        export_json(
            path,
            dict(
                status="INCOMPLETE",
                stop_verified=True,
                cleanup_error=None,
                elapsed_seconds=seconds,
                endpoint=name,
                usage_before={"balance": before},
                usage_after={"balance": after},
                accounting={"own_rate_units_per_hour": 2.0, "observed_rate_units_per_hour": 1.54},
            ),
        )
        result.append(path)
    return result


def test_reconciliation_charges_elapsed_ceiling_and_concurrent_gross_debit(tmp_path):
    debit = reconcile(receipts(tmp_path))
    assert [r["charged_units"] for r in debit["attempts"]] == [1.62, 0.28]
    assert debit["spent_units"] == 1.9 and debit["remaining_units"] == 6.1
    assert debit["source"]["unit_cap"] + debit["replacement"]["unit_cap"] == 6.1
    review = dict(
        schema="bl-p1-codex-v-review/v1",
        status="PASS",
        source_sha256={"x": "a"},
        unresolved_findings=[],
        no_new_cuda_path=True,
    )
    assert verify_source_admission(review, debit, {"x": "a"}) == debit["source"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "PENDING"),
        ("source_sha256", {"x": "stale"}),
        ("unresolved_findings", ["defect"]),
        ("no_new_cuda_path", False),
    ],
)
def test_source_rejects_stale_or_incomplete_review(tmp_path, field, value):
    debit = reconcile(receipts(tmp_path))
    review = dict(
        schema="bl-p1-codex-v-review/v1",
        status="PASS",
        source_sha256={"x": "a"},
        unresolved_findings=[],
        no_new_cuda_path=True,
    )
    review[field] = value
    with pytest.raises(ValueError):
        verify_source_admission(review, debit, {"x": "a"})


def test_source_recomputes_debit_instead_of_trusting_asserted_remainder(tmp_path):
    debit = reconcile(receipts(tmp_path))
    changed = deepcopy(debit)
    changed["remaining_units"] = 8
    review = dict(
        schema="bl-p1-codex-v-review/v1",
        status="PASS",
        source_sha256={},
        unresolved_findings=[],
        no_new_cuda_path=True,
    )
    with pytest.raises(ValueError, match="differs"):
        verify_source_admission(review, changed, {})
    with pytest.raises(ValueError, match="inventory changed"):
        reconcile([Path(debit["attempts"][0]["path"])] * 2)


def test_new_terminal_attempt_cannot_escape_shared_debit(tmp_path):
    paths = receipts(tmp_path)
    other = tmp_path / "unexpected-attempt/colab-output/receipt.json"
    other.parent.mkdir(parents=True)
    export_json(other, {"status": "INCOMPLETE"})
    with pytest.raises(ValueError, match="inventory changed"):
        reconcile(paths)


def test_raw_create_only_supervisor_receipts_are_hash_bound_without_modifying_them(tmp_path):
    paths = receipts(tmp_path)
    for path in paths:
        Path(str(path) + ".sha256").unlink()
        Path(str(path) + ".complete").unlink()
    debit = reconcile(paths)
    assert debit["spent_units"] == 1.9
    assert all(len(row["sha256"]) == 64 for row in debit["attempts"])
    assert not any(Path(str(path) + ".sha256").exists() for path in paths)
