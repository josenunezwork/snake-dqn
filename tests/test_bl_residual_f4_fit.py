"""Synthetic F4 reserve/funding fixtures; no factory labels are opened."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import research.bl_residual.heldout as heldout
from research.bl_residual.data import allocate_lives, world_fold
from research.bl_residual.fit import FitRecipe, _ordinary_growth_loss, _preference_loss
from research.bl_residual.funding import evaluate


def _look(dose: int, checkpoint: str, retention: float, *, harm: int = 0) -> dict:
    numerator = round(retention * 20)
    retention = numerator / 20
    lives = [
        {
            "life_id": f"life-{index}",
            "parent_world": f"parent-{index // 2}",
            "retention": retention,
            "decisive_rows": 20,
            "retained_rows": numerator,
        }
        for index in range(64)
    ]
    return {
        "schema": "bl-residual-heldout/v2",
        "dose": dose,
        "namespace": "synthetic-series",
        "recipe_sha256": "r" * 64,
        "allocation_sha256": "a" * 64,
        "reserve_allocation_sha256": "z" * 64,
        "checkpoint_sha256": checkpoint,
        "archive_receipt_sha256": "c" * 64,
        "natural_archive_complete": True,
        "metrics": {
            "life_retention": lives,
            "reserve_lives_with_zero_decisive_rows": 0,
            "harmful_override_count_natural": harm,
        },
    }


def test_joint_funding_uses_harm_only_not_d1024_retention_floor() -> None:
    result = evaluate(
        _look(256, "2" * 64, 0.50),
        _look(1024, "1" * 64, 0.60),
        {"disposition": "STRIKES_VOLUME"},
    )
    assert result["verdict"] == "PASS"
    assert result["mean_delta_life_retention"] == pytest.approx(0.10)
    assert result["harm_screen_d1024_pass"] is True


def test_unpublished_f2_uses_the_strikes_branch_and_harm_stops_funding() -> None:
    result = evaluate(
        _look(256, "2" * 64, 0.50),
        _look(1024, "1" * 64, 0.60, harm=1),
        {},
    )
    assert result["f2_disposition"] == "UNPUBLISHED_APPLY_STRIKES_BRANCH"
    assert result["verdict"] == "STOP_FUNDING"


def test_unpublished_nonstrike_text_cannot_bypass_the_strikes_branch() -> None:
    result = evaluate(
        _look(256, "2" * 64, 0.50),
        _look(1024, "1" * 64, 0.50),
        {"disposition": "DOES_NOT_STRIKE_VOLUME"},
    )
    assert result["f2_disposition"] == "UNPUBLISHED_APPLY_STRIKES_BRANCH"
    assert result["verdict"] == "STOP_FUNDING"


def test_sealed_published_f2_nonstrike_can_advance_without_delta_bar() -> None:
    result = evaluate(
        _look(256, "2" * 64, 0.50),
        _look(1024, "1" * 64, 0.50),
        {
            "schema": "bl-residual-f2-disposition/v1",
            "volume_struck": False,
            "published": True,
            "source_sha256": "a" * 64,
            "published_ref": "docs/research/f2-result.md",
        },
    )
    assert result["f2_disposition"] == "DOES_NOT_STRIKE_VOLUME"
    assert result["verdict"] == "PASS"


def test_exact_five_point_retention_gain_passes_registered_funding_boundary() -> None:
    result = evaluate(
        _look(256, "2" * 64, 0.55),
        _look(1024, "1" * 64, 0.60),
        {"schema": "bl-residual-f2-disposition/v1", "volume_struck": True},
    )
    assert result["mean_delta_life_retention"] == pytest.approx(0.05)
    assert result["verdict"] == "PASS"


def test_reserve_cap_four_applies_across_watch_and_other_selection() -> None:
    parents = [parent for parent in (f"p-{i}" for i in range(800)) if world_fold(parent) == 0]
    rows = [
        {
            "life_id": f"l-{index}",
            "parent_world": parents[index // 5],
            "stream": "S-Watch6" if index < 80 else "S-gate",
            "rescue_bearing": True,
            "allocation_index": index,
        }
        for index in range(500)
    ]
    value = allocate_lives(rows)
    heldout = value["doses"]["256"]["heldout_lives"]
    counts = {}
    for life in heldout:
        parent = value["parents"][life]
        counts[parent] = counts.get(parent, 0) + 1
    assert max(counts.values()) <= 4


def test_survival_ignores_bare_action_and_growth_excludes_dying_actions() -> None:
    q = torch.zeros(1, 6)
    data = {
        "status": torch.ones(1, 6, dtype=torch.bool),
        "legal": torch.ones(1, 6, dtype=torch.bool),
        "action_mask": torch.ones(1, 6, dtype=torch.bool),
        "died": torch.tensor([[False, True, True, True, True, True]]),
        "bare": torch.tensor([0]),
        "ordinary": torch.tensor([True]),
        "mass": torch.tensor([[1.0, 100.0, 0.0, 0.0, 0.0, 0.0]]),
    }
    assert _preference_loss(q, data, FitRecipe()).item() > 0
    # The only high-mass action dies, so survivor-only growth has no eligible pair.
    assert _ordinary_growth_loss(q, data, FitRecipe()).item() == 0


def test_missing_approval_refuses_before_any_npz_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened = False

    def refuse(*_args: object) -> None:
        raise ValueError("approval required")

    def npz_sentinel(*_args: object, **_kwargs: object) -> None:
        nonlocal opened
        opened = True
        raise AssertionError("outcome table must remain closed")

    monkeypatch.setattr(heldout, "require_judgment", refuse)
    monkeypatch.setattr(heldout.np, "load", npz_sentinel)
    with pytest.raises(ValueError, match="approval"):
        heldout.evaluate(
            tmp_path / "checkpoint",
            tmp_path / "b8",
            tmp_path / "outcomes.npz",
            allocation_path=tmp_path / "allocation",
            control_path=tmp_path / "control",
            approval_path=tmp_path / "missing-approval",
            archive_receipt=tmp_path / "census",
        )
    assert opened is False


def test_foreign_ra_claim_refuses_before_any_npz_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preflight = {
        "dose": 256,
        "checkpoint_sha256": "c" * 64,
        "control_sha256": "d" * 64,
        "allocation_sha256": "a" * 64,
        "archive_receipt_sha256": "r" * 64,
        "registration_sha256": "g" * 64,
    }
    monkeypatch.setattr(heldout, "preflight_look", lambda *_args: preflight)
    monkeypatch.setattr(
        heldout.np, "load", lambda *_args, **_kwargs: pytest.fail("opened outcomes")
    )
    claim = tmp_path / "foreign.lock.json"
    claim.write_text('{"schema":"bl-residual-r-a-lock/v1"}')
    with pytest.raises(ValueError, match="does not bind"):
        heldout.evaluate(
            tmp_path / "checkpoint",
            tmp_path / "b8",
            tmp_path / "outcomes.npz",
            allocation_path=tmp_path / "allocation",
            control_path=tmp_path / "control",
            approval_path=tmp_path / "approval",
            archive_receipt=tmp_path / "census",
            joint_claim_path=claim,
        )
