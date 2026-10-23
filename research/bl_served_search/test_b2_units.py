"""Focused offline tests for B2's outcome-blind completed-unit contract."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from research.bl_served_search import b2_units
from research.bl_served_search.exports import export_bytes

NAMESPACE = "bl-c15-b2-natural-roots/v1"
ALLOCATION = hashlib.sha256(b"frozen allocation").hexdigest()


def _rows() -> list[dict]:
    return [
        {
            "trajectory_index": index % 32,
            "life": index + 1,
            "frame": 800 + index,
            "prediction": {"escape": index % 2 == 0},
            "outcomes": {"would_change": index},
        }
        for index in range(9)
    ]


def _complete_unit(tmp_path: Path, index: int = 0) -> tuple[str, Path]:
    unit = b2_units.unit_id(NAMESPACE, index)
    path = b2_units.unit_payload_path(tmp_path, unit)
    export_bytes(path, b"this deliberately is not JSON outcome payload\n")
    return unit, path


def test_hash_selected_roots_are_order_independent_and_ignore_outcomes() -> None:
    rows = _rows()
    first = b2_units.select_roots(rows, NAMESPACE)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["prediction"] = {"escape": not row["prediction"]["escape"]}
        row["outcomes"] = {"different": object()}
    second = b2_units.select_roots(reversed(changed), NAMESPACE)
    assert first == second
    assert len(first) == 4
    assert [row["root_sha256"] for row in first] == sorted(row["root_sha256"] for row in first)


def test_root_selection_refuses_duplicate_or_malformed_coordinates() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate"):
        b2_units.select_roots([rows[0], copy.deepcopy(rows[0])], NAMESPACE)
    rows[0]["life"] = True
    with pytest.raises(ValueError, match="integer"):
        b2_units.select_roots(rows, NAMESPACE)


def test_empty_ledger_is_offline_and_has_no_units() -> None:
    ledger = b2_units.empty_ledger(NAMESPACE, ALLOCATION)
    assert ledger["units"] == []
    assert b2_units.validate_ledger(ledger, NAMESPACE, ALLOCATION) == set()
    assert b2_units.verify_ledger(ledger, Path("."), NAMESPACE, ALLOCATION) == set()


def test_ledger_uses_only_complete_export_identity_not_outcome_json(tmp_path: Path) -> None:
    unit, path = _complete_unit(tmp_path)
    ledger = b2_units.make_ledger(tmp_path, NAMESPACE, ALLOCATION, [unit])
    assert ledger["units"] == [
        {
            "unit_id": unit,
            "path": f"units/{path.name}",
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]
    assert b2_units.verify_ledger(ledger, tmp_path, NAMESPACE, ALLOCATION) == {unit}
    assert b2_units.validate_ledger(ledger, NAMESPACE, ALLOCATION) == {unit}


def test_runtime_ledger_validation_never_opens_a_prior_outcome_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit, path = _complete_unit(tmp_path)
    observed: list[Path] = []

    def identity_only(candidate: Path) -> str:
        observed.append(candidate)
        return "a" * 64

    monkeypatch.setattr(b2_units, "verify_export", identity_only)
    ledger = b2_units.make_ledger(tmp_path, NAMESPACE, ALLOCATION, [unit])
    assert b2_units.validate_ledger(ledger, NAMESPACE, ALLOCATION) == {unit}
    assert observed == [path, path]


@pytest.mark.parametrize(
    "mutation",
    ["foreign", "duplicate", "traversal", "bool_size", "malformed_unit", "tamper"],
)
def test_ledger_refuses_foreign_duplicates_traversal_malformed_and_tampered_units(
    tmp_path: Path, mutation: str
) -> None:
    unit, path = _complete_unit(tmp_path)
    ledger = b2_units.make_ledger(tmp_path, NAMESPACE, ALLOCATION, [unit])
    if mutation == "foreign":
        ledger["binding"]["namespace"] = "other"
    elif mutation == "duplicate":
        ledger["units"] *= 2
    elif mutation == "traversal":
        ledger["units"][0]["path"] = "units/../elsewhere.json"
    elif mutation == "bool_size":
        ledger["units"][0]["size"] = True
    elif mutation == "malformed_unit":
        ledger["units"][0]["unit_id"] = "b2-not-a-unit"
    else:
        path.write_bytes(b"modified payload")
    with pytest.raises(ValueError):
        b2_units.verify_ledger(ledger, tmp_path, NAMESPACE, ALLOCATION)


def test_ledger_export_is_create_only(tmp_path: Path) -> None:
    ledger = b2_units.empty_ledger(NAMESPACE, ALLOCATION)
    path = tmp_path / "verified-ledger.json"
    b2_units.export_ledger(path, ledger)
    with pytest.raises(FileExistsError):
        b2_units.export_ledger(path, ledger)
