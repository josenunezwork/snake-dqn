"""Contracts for the F3 gameplay-development screen seam."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_residual import gameplay


def _row(mix: str = "scripted", seed: int = 7, parent: str = "parent-7") -> dict:
    return {
        "world_id": f"world-{seed}",
        "parent_world_id": parent,
        "world_seed": seed,
        "mix": mix,
        "plan": {"seed": seed, "mix": mix, "specs": []},
    }


def _fixture_bank(row: dict | None = None) -> dict:
    return {
        "namespace": "f3-fixture/v1",
        "collision_receipt": {"status": "CLEAR", "namespace": "f3-fixture/v1"},
        "rows": [_row() if row is None else row],
    }


def test_fixture_runs_paired_arms_and_never_emits_a_decision(monkeypatch, tmp_path: Path) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    candidates = {}
    for dose in gameplay.DOSES:
        path = tmp_path / f"{dose}.pth"
        path.write_bytes(dose.encode())
        candidates[dose] = path
    monkeypatch.setattr(
        gameplay, "sha256_file", lambda path: gameplay.B8_SHA256 if path == b8 else "a" * 64
    )
    monkeypatch.setattr(gameplay, "_validate_materialized_rosters", lambda rows: None)
    monkeypatch.setattr(gameplay, "_root_digest", lambda plan, **kwargs: "d" * 64)
    monkeypatch.setattr(
        gameplay.ceiling,
        "run_arm",
        lambda plan, **kwargs: {
            "mass_integral": float(kwargs["horizon"]),
            "alive_frame_fraction": 1.0,
            "no_death_episode": True,
        },
    )
    receipt = gameplay.run_screen(_fixture_bank(), b8_path=b8, candidates=candidates, horizon=3)
    assert receipt["fixture"] is True
    assert receipt["decision"]["label"] == "NOT_A_DECISION"
    assert list(receipt["observations"][0]["arms"]) == ["B8", *gameplay.DOSES]


def test_unratified_production_refuses_before_bank_execution(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gameplay, "_require_mac", lambda: None)
    with pytest.raises(RuntimeError, match="unratified"):
        gameplay.run_screen({}, b8_path=tmp_path / "none", candidates={}, horizon=gameplay.HORIZON)


@pytest.mark.parametrize(
    "row, message",
    [
        (_row(parent="train-parent"), "collides"),
        ({**_row(), "plan": {"seed": 8, "mix": "scripted"}}, "plan identity"),
    ],
)
def test_bank_rejects_parent_collision_and_plan_mismatch(row: dict, message: str) -> None:
    parents = {"train-parent"} if row["parent_world_id"] == "train-parent" else None
    with pytest.raises(ValueError, match=message):
        gameplay.validate_bank(_fixture_bank(row), training_parent_ids=parents, production=False)


def test_bank_requires_clear_collision_receipt() -> None:
    bank = _fixture_bank()
    bank["collision_receipt"]["status"] = "REJECTED"
    with pytest.raises(ValueError, match="CLEAR"):
        gameplay.validate_bank(bank, production=False)


def test_production_requires_parent_allocation_before_audit(monkeypatch) -> None:
    rows = []
    for mix, offset in zip(gameplay.MIXES, range(3)):
        rows.extend(
            _row(mix=mix, seed=10_000 * offset + index, parent=f"p-{mix}-{index}")
            for index in range(gameplay.COUNT_PER_MIX)
        )
    bank = {"namespace": "production/v1", "collision_receipt": {"status": "CLEAR"}, "rows": rows}
    with pytest.raises(ValueError, match="parent allocation"):
        gameplay.validate_bank(bank, production=True)


def test_production_evidence_must_be_hash_bound_successor() -> None:
    with pytest.raises(RuntimeError, match="draft"):
        gameplay._approved_successor(
            {
                "status": "RATIFIED",
                "registration_sha256": "a" * 64,
                "independent_review_sha256": "b" * 64,
                "registration_path": "research/bl_residual/F4_DRAFT.md",
            }
        )


def test_production_evidence_binds_review_files(tmp_path: Path) -> None:
    registration = tmp_path / "F4_RATIFIED.md"
    review = tmp_path / "F4_REVIEW.md"
    registration.write_text("ratified")
    review.write_text("independent review")
    evidence = {
        "status": "RATIFIED",
        "registration_path": str(registration),
        "registration_sha256": gameplay.sha256_file(registration),
        "independent_review_path": str(review),
        "independent_review_sha256": gameplay.sha256_file(review),
    }
    assert gameplay._approved_successor(evidence) == {
        "registration_sha256": evidence["registration_sha256"],
        "independent_review_sha256": evidence["independent_review_sha256"],
    }
