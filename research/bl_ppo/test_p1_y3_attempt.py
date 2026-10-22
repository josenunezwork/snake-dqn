"""Prospective Y3-05 envelope and resume-contract checks."""

from pathlib import Path

import pytest

from research.bl_ppo.p1_admission import reconcile_y3_05


def _receipts(root: Path) -> list[Path]:
    rows = [
        (
            "source-02/colab-output/receipt.json",
            "INCOMPLETE",
            2914.9896361669526,
            2.0,
            1131.87,
            1130.72,
        ),
        (
            "source-latebody-02/colab-output/receipt.json",
            "INCOMPLETE",
            467.8529948340147,
            2.0,
            1130.32,
            1130.04,
        ),
        (
            "source-latebody-03-canary-01/canary-output/receipt.json",
            "PASS",
            580.9944156669662,
            1.54,
            1128.1,
            1127.86,
        ),
        (
            "source-latebody-03-canary-01/source-output/receipt.json",
            "INCOMPLETE",
            2973.10854645801,
            1.54,
            1127.6,
            1125.08,
        ),
    ]
    paths = []
    for relative, status, elapsed, rate, before, after in rows:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            __import__("json").dumps(
                {
                    "status": status,
                    "stop_verified": True,
                    "cleanup_error": None,
                    "elapsed_seconds": elapsed,
                    "usage_before": {"balance": before},
                    "usage_after": {"balance": after},
                    "accounting": {"own_rate_units_per_hour": rate},
                }
            )
        )
        paths.append(path)
    return paths


def test_y3_05_debit_preserves_four_real_receipts_and_frozen_effective_envelope(
    tmp_path: Path,
) -> None:
    debit = reconcile_y3_05(_receipts(tmp_path))
    assert debit["legacy_conservative_spend_units"] == 4.67
    assert debit["envelope_units"] == 25.5
    assert debit["source"]["forecast_minutes"] == 75
    assert debit["source"]["timeout_minutes"] == 300
    assert debit["source"]["effective_cap_units"] == 9.62
    assert debit["replacement"]["forecast_minutes"] == 30
    assert debit["replacement"]["timeout_minutes"] == 120
    assert debit["replacement"]["effective_cap_units"] == 3.86
    assert debit["canary"]["effective_cap_units"] == 2.0
    assert debit["retry"]["max_outcome_blind_retries"] == 2


def test_y3_05_reconciliation_refuses_changed_receipt_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity/order"):
        reconcile_y3_05(list(reversed(_receipts(tmp_path))))


def test_y3_05_reconciliation_requires_released_terminal_receipts(tmp_path: Path) -> None:
    receipts = _receipts(tmp_path)
    value = __import__("json").loads(receipts[0].read_text())
    value["stop_verified"] = False
    receipts[0].write_text(__import__("json").dumps(value))
    with pytest.raises(ValueError, match="released terminal"):
        reconcile_y3_05(receipts)
