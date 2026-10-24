"""Focused refusal coverage for the immutable V8 reconstruction boundary."""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools import v8_admission_budget as budget


@pytest.fixture
def boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Use only inert host paths; do not patch frozen acceptance functions."""
    source = Path(__file__).with_name("test_v8_admission_evidence.py")
    spec = importlib.util.spec_from_file_location("v8_boundary_fixture", source)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.accepted.__wrapped__(tmp_path, monkeypatch)
    record = json.loads(budget.RECONSTRUCTION.read_text())
    return record, fixture.bind


def test_working_templates_have_no_live_accounting_assertions() -> None:
    record = budget.reconstruction_working_template()
    assert record["accounting"]["B"] is None
    assert record["reconstructed_spendable_effective_limit_units"] is None
    assert budget.ledger_working_template()["complete_history"] is False


def test_freeze_refuses_any_production_path_substitution(tmp_path: Path) -> None:
    with pytest.raises(budget.Refused, match="path is fixed"):
        budget.freeze_reconstruction(tmp_path / "missing.json", tmp_path / "foreign.json")


def test_validate_reconstruction_refuses_foreign_binding(tmp_path: Path) -> None:
    with pytest.raises(budget.Refused, match="foreign reconstruction"):
        budget.validate_reconstruction(
            {"path": str(tmp_path / "RECONSTRUCTION.json"), "sha256": "0" * 64, "pointer": ""}
        )


@pytest.mark.parametrize(
    "mode",
    [
        "wrong_C",
        "wrong_base",
        "wrong_reserve",
        "later_clock",
        "recovered_seal",
        "missing_historical",
        "wrong_ruling",
        "wrong_policy",
        "wrong_raw_balance",
        "cached_spend",
        "unsupported_row_charge",
        "incomplete_history",
        "external_omission",
        "missing_liability",
    ],
)
def test_reconstruction_cannot_reset_or_fabricate_authority(
    boundary: tuple, tmp_path: Path, mode: str
) -> None:
    original, bind = boundary
    record = copy.deepcopy(original)
    assert budget._check_record(original) == original
    if mode == "wrong_C":
        record["reconstructed_spendable_effective_limit_units"] += 1
    elif mode == "wrong_base":
        record["base_limit_units"] += 1
    elif mode == "wrong_reserve":
        record["included_reserve_units"] += 1
    elif mode == "later_clock":
        record["original_t0"] = "2026-10-24T02:00:00Z"
    elif mode == "recovered_seal":
        record["historical_original_sealed_admission"] = "invented"
    elif mode == "missing_historical":
        del record["historical_original_t0"]
    elif mode == "wrong_ruling":
        record["authority"]["budget_ruling_sha256"] = "0" * 64
    elif mode == "wrong_policy":
        record["policy_version"] = "foreign"
    elif mode == "wrong_raw_balance":
        record["accounting"]["B"] = 999
        record["accounting_sources"]["B"] = bind(tmp_path / "mutant-B.json", 999)
    elif mode in {
        "cached_spend",
        "missing_liability",
        "unsupported_row_charge",
        "incomplete_history",
    }:
        ledger = budget.sourced(record["ledger"])
        if mode == "cached_spend":
            for row in ledger["rows"]:
                row["charged"] = {"wall_minutes": 0, "units": 0}
        elif mode == "missing_liability":
            del ledger["rows"][0]["remaining_liability"]
        elif mode == "incomplete_history":
            ledger["complete_history"] = False
        else:
            ledger["rows"][0]["charged"]["units"] = 100
            ledger["reconciliation"]["spent"]["global"]["units"] = 102.5
            record["accounting"]["S"] = 102.5
            record["accounting_sources"]["S"] = bind(tmp_path / "mutant-S.json", 102.5)
            record["reconstructed_spendable_effective_limit_units"] = 192.5
            record["sealed_spendable_effective_limit_units"] = 192.5
            record["base_limit_units"] = 192.5 / 2
            record["included_reserve_units"] = 192.5 / 6
        record["ledger"] = bind(tmp_path / "mutant-ledger.json", ledger)
    else:
        record["accounting"]["L_external"] = 0
        record["accounting_sources"]["L_external"] = bind(tmp_path / "external-zero.json", 0)
    with pytest.raises(budget.Refused):
        budget._check_record(record)


def test_exact_freeze_and_second_freeze_are_create_only(
    boundary: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, _ = boundary
    path = tmp_path / "new-frozen-boundary.json"
    monkeypatch.setattr(budget, "RECONSTRUCTION", path)
    input_path = tmp_path / "reconstruction-input.json"
    input_path.write_text(json.dumps(record))
    budget.freeze_reconstruction(input_path, path)
    expected = budget.sha256(path)
    assert budget._verified(path) == expected
    with pytest.raises(budget.Refused, match="create-only"):
        budget.freeze_reconstruction(input_path, path)
    assert budget.sha256(path) == expected


@pytest.mark.parametrize("minutes", [-1, 16])
def test_freeze_boundary_cannot_be_future_or_stale(
    boundary: tuple, tmp_path: Path, minutes: int
) -> None:
    record, _ = boundary
    record["accounting_as_of_utc"] = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes)
    ).isoformat()
    input_path = tmp_path / "invalid-time.json"
    input_path.write_text(json.dumps(record))
    with pytest.raises(budget.Refused, match="older|future"):
        budget.freeze_reconstruction(input_path, budget.RECONSTRUCTION)
