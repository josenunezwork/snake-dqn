"""Admission tests use frozen bank data, never step the simulator."""

from __future__ import annotations

import copy
import json

import pytest

from research.t34_round0 import registration
from research.t34_round0.common import PACKAGE, read_json


def test_saved_bank_and_audit_cover_disjoint_campaign_and_preflight() -> None:
    bank = registration.check_audit(
        PACKAGE / "BANK_ROUND0.json", PACKAGE / "SEED_AUDIT_ROUND0.json"
    )
    rows = bank["rows"] + bank["excluded_preflight_rows"]
    assert len(rows) == len({r["world_seed"] for r in rows}) == 242
    assert len({r["namespace"] for r in bank["rows"]}) == 3


def test_registration_cannot_admit_draft_or_modified_bank(tmp_path) -> None:
    with pytest.raises(ValueError, match="not ratified"):
        registration.require_registration(
            {}, PACKAGE / "BANK_ROUND0.json", PACKAGE / "SEED_AUDIT_ROUND0.json"
        )
    bank = copy.deepcopy(read_json(PACKAGE / "BANK_ROUND0.json"))
    bank["rows"][0]["world_seed"] += 1
    changed = tmp_path / "bank.json"
    changed.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="deterministic"):
        registration.check_audit(changed, PACKAGE / "SEED_AUDIT_ROUND0.json")


def test_json_admission_rejects_ambiguous_or_nonfinite_bytes(tmp_path) -> None:
    path = tmp_path / "bad.json"
    for text in ('{"commit":1,"commit":2}', '{"estimate":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)


def test_tabular_audit_is_mandatory_and_bank_bound(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(registration, "PACKAGE", tmp_path)
    bad = read_json(PACKAGE / "TABULAR_SEED_AUDIT.json")
    bad["bank_sha256"] = "0" * 64
    (tmp_path / "TABULAR_SEED_AUDIT.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="CSV/TSV"):
        registration.check_audit(PACKAGE / "BANK_ROUND0.json", PACKAGE / "SEED_AUDIT_ROUND0.json")


def test_retention_growth_cannot_introduce_a_late_collision(tmp_path, monkeypatch) -> None:
    bank = read_json(PACKAGE / "BANK_ROUND0.json")
    path = tmp_path / "research/t34_pathb/RETENTION_BANK_V2.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "terminal_rows": [{"seed": bank["rows"][0]["world_seed"]}],
                "product_rows": [],
                "timing_row": {"seed": 1},
            }
        )
    )
    monkeypatch.setattr(registration, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="current retention bank"):
        registration.check_audit(PACKAGE / "BANK_ROUND0.json", PACKAGE / "SEED_AUDIT_ROUND0.json")
