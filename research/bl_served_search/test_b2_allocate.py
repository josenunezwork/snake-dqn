"""Mocked, offline allocation tests for the B2 natural-root library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_served_search import b2, b2_allocate, b2_units
from research.bl_served_search.exports import export_json, verify_export


def _ruling(path: Path, **changes: object) -> Path:
    value = {
        "status": b2_allocate.RULING_STATUS,
        "plan_sha256": b2.PLAN_SHA256,
        "trajectory_frames": 14400,
        "root_stride": 5,
        "escape_total_frames": 1440,
        "unit_timeout_seconds": 1800,
        "root_grid": "absolute-trajectory-decision",
    }
    value.update(changes)
    export_json(path, value)
    return path


def _clear_audit(marker: int = 1) -> dict:
    banks = (
        "research/t34_round0/BANK_ROUND0.json",
        "research/t34_round0d/BANK_ROUND0D.json",
        "research/t34_pathb/RETENTION_BANK_V2.json",
        "research/phase0_search_20261019/WORLD_BANK.json",
    )
    return {
        "schema": "bl-served-search-seed-audit/v1",
        "status": "CLEAR",
        "candidate_count": 33,
        "unique_count": 33,
        "collisions": [],
        "internal_collision": False,
        "explicit_banks": {
            name: {
                "positive_control": 100 + index,
                "collisions": [],
                "sha256": "a" * 64,
                "count": 1,
            }
            for index, name in enumerate(banks)
        },
        "history": {
            "roots": ["repository", "artifacts"],
            "hits": {"repository": list(range(100, 104))},
            "never_opened": b2_allocate.bank.SEALED,
        },
        "training_exclusions": [{"collisions": [], "sha256": "b" * 64, "count": 1}] * 7,
        "coverage": "all visible text banks",
        "marker": marker,
    }


def test_round_derivation_is_deterministic_uint32_and_whole_allocation() -> None:
    seeds, canary, rows = b2_allocate.derive(0)
    other, other_canary, other_rows = b2_allocate.derive(1)
    assert len(seeds) == 32
    assert len(rows) == 33
    assert len({*seeds, canary}) == 33
    assert all(type(seed) is int and 0 <= seed < 2**32 for seed in [*seeds, canary])
    assert (seeds, canary, rows) == b2_allocate.derive(0)
    assert not ({*seeds, canary} & {*other, other_canary})
    assert all(row["allocation"] == "production" for row in rows[:-1])
    assert rows[-1]["allocation"] == "excluded-canary"
    assert len(other_rows) == 33


def test_missing_or_foreign_ruling_refuses_before_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[dict]] = []
    monkeypatch.setattr(
        b2_allocate.bank, "audit", lambda rows: calls.append(rows) or _clear_audit()
    )
    ruling = _ruling(tmp_path / "ruling.json", root_stride=None)
    with pytest.raises(ValueError, match="missing positive root_stride"):
        b2_allocate.allocate(ruling, tmp_path / "out")
    assert calls == []
    bad_plan = _ruling(tmp_path / "bad-plan.json", plan_sha256="0" * 64)
    with pytest.raises(ValueError, match="authority"):
        b2_allocate.allocate(bad_plan, tmp_path / "out-two")
    assert calls == []
    bad_horizon = _ruling(tmp_path / "bad-horizon.json", escape_total_frames=300)
    with pytest.raises(ValueError, match="legal H1440"):
        b2_allocate.allocate(bad_horizon, tmp_path / "out-three")
    assert calls == []


def test_rejected_whole_round_is_retained_then_clear_round_freezes_design_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[dict]] = []

    def audit(rows: list[dict]) -> dict:
        calls.append(rows)
        if len(calls) == 1:
            return {"status": "REJECTED", "marker": 1}
        return _clear_audit(2)

    monkeypatch.setattr(b2_allocate.bank, "audit", audit)
    ruling = _ruling(tmp_path / "ruling.json")
    out = tmp_path / "allocation"
    design = b2_allocate.allocate(ruling, out)
    assert [len(rows) for rows in calls] == [33, 33]
    assert calls[0] == b2_allocate.derive(0)[2]
    assert calls[1] == b2_allocate.derive(1)[2]
    assert json.loads((out / "audit-round-00.json").read_text())["status"] == "REJECTED"
    assert verify_export(out / "audit-round-00.json")
    assert design["round_index"] == 1
    assert design["freshness"]["marker"] == 2
    assert design["freshness"]["seed_allocation_sha256"] == b2.digest_value(
        [*design["seeds"], design["canary_seed"]]
    )
    assert design["ruling"]["sha256"] == hashlib.sha256(ruling.read_bytes()).hexdigest()
    assert b2.validate_spec(json.loads((out / "b2-design.json").read_text())) == design
    freshness = json.loads((out / "freshness-audit.json").read_text())
    assert freshness["seeds"] == [*design["seeds"], design["canary_seed"]]
    assert design["freshness_report_sha256"] == verify_export(out / "freshness-audit.json")
    ledger = json.loads((out / "verified-ledger.json").read_text())
    assert (
        b2_units.validate_ledger(ledger, design["namespace"], design["allocation_digest"]) == set()
    )


def test_no_clear_round_retains_all_audits_and_never_writes_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(b2_allocate.bank, "audit", lambda rows: {"status": "REJECTED"})
    with pytest.raises(RuntimeError, match="no CLEAR"):
        b2_allocate.allocate(_ruling(tmp_path / "ruling.json"), tmp_path / "out", max_rounds=2)
    assert verify_export(tmp_path / "out/audit-round-00.json")
    assert verify_export(tmp_path / "out/audit-round-01.json")
    assert not (tmp_path / "out/b2-design.json").exists()


def test_self_reported_clear_without_full_bank_proof_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(b2_allocate.bank, "audit", lambda rows: {"status": "CLEAR"})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="bank.audit schema"):
        b2_allocate.allocate(_ruling(tmp_path / "ruling.json"), out)
    assert verify_export(out / "audit-round-00.json")
    assert not (out / "freshness-audit.json").exists()
    assert not (out / "b2-design.json").exists()


def test_output_is_create_only_and_auditor_is_not_reused_for_single_seed_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[dict]] = []
    monkeypatch.setattr(
        b2_allocate.bank,
        "audit",
        lambda rows: calls.append(rows) or _clear_audit(),
    )
    ruling = _ruling(tmp_path / "ruling.json")
    output = tmp_path / "out"
    b2_allocate.allocate(ruling, output)
    with pytest.raises(FileExistsError):
        b2_allocate.allocate(ruling, output)
    assert len(calls) == 1
    assert calls[0] == b2_allocate.derive(0)[2]
