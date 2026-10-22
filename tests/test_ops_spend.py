"""Contracts for tools/ops/spend.py without provider access."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEND_PATH = REPO / "tools/ops/spend.py"
SPEC = importlib.util.spec_from_file_location("ops_spend", SPEND_PATH)
assert SPEC and SPEC.loader
spend = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spend
SPEC.loader.exec_module(spend)


def _receipt(path: Path, *, elapsed: object = 1800, rate: object = 2, **extra: object) -> Path:
    payload = {
        "elapsed_seconds": elapsed,
        "accounting": {"own_rate_units_per_hour": rate},
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reports_each_verified_receipt_and_total(tmp_path: Path) -> None:
    one = _receipt(tmp_path / "one.json", elapsed=1800, rate=2, generation="g1", session="s1")
    two = _receipt(tmp_path / "two.json", elapsed=900, rate=4, generation="g2", session="s2")

    result = subprocess.run(
        [sys.executable, str(SPEND_PATH), str(one), str(two)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "receipt=" + str(one.resolve()) in result.stdout
    assert "units=1.000000000" in result.stdout
    assert "TOTAL receipts=2 units=2.000000000" in result.stdout
    assert "sha256=" in result.stdout


@pytest.mark.parametrize(
    "elapsed,rate", [(True, 1), (float("nan"), 1), (1, -1), (1, "2"), (1e308, 1e308)]
)
def test_malformed_or_nonfinite_accounting_fails_closed(
    tmp_path: Path, elapsed: object, rate: object
) -> None:
    receipt = _receipt(tmp_path / "bad.json", elapsed=elapsed, rate=rate)

    with pytest.raises(spend.ReceiptRefused):
        spend.verify_receipts([receipt])


def test_duplicate_path_and_identity_are_refused(tmp_path: Path) -> None:
    first = _receipt(tmp_path / "first.json", generation="generation", session="session")
    same_identity = _receipt(tmp_path / "second.json", generation="generation", session="session")

    with pytest.raises(spend.ReceiptRefused, match="duplicate receipt path"):
        spend.verify_receipts([first, first])
    with pytest.raises(spend.ReceiptRefused, match="duplicate receipt generation/session identity"):
        spend.verify_receipts([first, same_identity])


def test_missing_or_partial_identity_is_handled_conservatively(tmp_path: Path) -> None:
    anonymous = _receipt(tmp_path / "anonymous.json")
    partial = _receipt(tmp_path / "partial.json", generation="only-generation")

    assert spend.verify_receipts([anonymous])[0].identity is None
    with pytest.raises(spend.ReceiptRefused, match="incomplete generation/session"):
        spend.verify_receipts([partial])


def test_missing_receipt_is_reported_as_a_refusal(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SPEND_PATH), str(tmp_path / "missing.json")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "REFUSED: cannot resolve receipt" in result.stderr


def test_total_overflow_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    row = spend.SpendRow(tmp_path / "receipt.json", "digest", 1, 1, 1e308, None)
    monkeypatch.setattr(spend, "verify_receipts", lambda _: [row, row])

    assert spend.main([str(tmp_path / "receipt.json")]) == 2
    assert "total own-spend calculation is not finite" in capsys.readouterr().err
