"""Light offline checks for all-attempt source admission accounting."""

import json
from pathlib import Path

import pytest

from research.bl_ppo.e8_04_source_admission import reconcile


def _receipt(path: Path, *, debit: float, released: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "INCOMPLETE",
                "stop_verified": released,
                "cleanup_error": None,
                "elapsed_seconds": 3600,
                "accounting": {"own_rate_units_per_hour": 1.54},
                "usage_before": {"balance": 100},
                "usage_after": {"balance": 100 - debit},
            }
        )
    )
    return path


def test_charges_staging_and_current_canary_once(tmp_path: Path) -> None:
    _receipt(tmp_path / "old/colab-output/receipt.json", debit=2)
    _receipt(tmp_path / "staging/family-canary/source-output/receipt.json", debit=3)
    canary = _receipt(tmp_path / "new/final/canary-output/receipt.json", debit=0.1)
    value = reconcile(tmp_path, canary)
    assert len(value["attempts"]) == 3
    assert value["charged_units"] == 6.54
    assert value["reserved_total_units"] == 20.02
    assert value["warning"] == "WARNING_UNEXPLAINED_DEBIT"


def test_refuses_unverified_stop(tmp_path: Path) -> None:
    _receipt(tmp_path / "old/colab-output/receipt.json", debit=1, released=False)
    with pytest.raises(ValueError, match="release"):
        reconcile(tmp_path)


def test_refuses_full_invoice_over_envelope(tmp_path: Path) -> None:
    _receipt(tmp_path / "old/colab-output/receipt.json", debit=11)
    with pytest.raises(ValueError, match="25.50"):
        reconcile(tmp_path)
