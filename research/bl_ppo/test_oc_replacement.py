"""Targeted prospective Z-4 statistical and isolation checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.bl_ppo import oc
from research.bl_ppo import oc_replacement as replacement
from research.bl_ppo.exports import export_json, verify_export


def inputs() -> dict:
    return {
        "mixes": {
            m: {
                "paired_mass_variance": 10000.0,
                "paired_alive_variance": 0.04,
                "paired_episode_variance": 0.16,
                "b8_alive": 0.8,
            }
            for m in oc.MIXES
        }
    }


def test_stage_pooling_matches_raw_worlds() -> None:
    rng = np.random.default_rng(21)
    a, b = rng.normal(size=(8, 3, 4, 16)), rng.normal(0.7, size=(8, 3, 4, 16))
    mean, variance = replacement.pool_stages(
        a.mean(axis=-1),
        a.var(axis=-1, ddof=1),
        b.mean(axis=-1),
        b.var(axis=-1, ddof=1),
        np.full((1, 3, 1), 16),
    )
    combined = np.concatenate((a, b), axis=-1)
    np.testing.assert_allclose(mean, combined.mean(axis=-1))
    np.testing.assert_allclose(variance, combined.var(axis=-1, ddof=1))


def test_futility_boundary_and_stopped_holm_slot() -> None:
    active = replacement.continuation_mask(
        np.array([[39.0, 40.0], [41.0, 100.0]]),
        np.zeros((2, 2)),
        np.full((2, 2), 100),
        0,
    )
    assert active.tolist() == [[False, True], [True, True]]
    # Stopped R cannot help F step down from .025 to .05.
    assert not oc._holm(np.where(active[:1], [[0.001, 0.03]], 1.0)).any()
    rng = np.random.default_rng(8)
    p = rng.uniform(0, 0.1, size=(1000, 2))
    active = rng.random((1000, 2)) > 0.5
    assert not (oc._holm(np.where(active, p, 1.0)) & ~oc._holm(p)).any()


def test_candidates_report_full_grid_and_partition_uncoupled_route() -> None:
    for design in replacement.DESIGNS:
        rows = replacement.simulate(inputs(), design, reps=1000, seed=1)
        assert len(rows) == 60
        for row in rows:
            assert sum(row["funding_routes_joint"].values()) == pytest.approx(1)
            assert row["family_funding_probability"] == pytest.approx(
                1 - row["funding_routes_joint"]["neither"]
            )
            if not design["e6_funds"]:
                assert row["family_funding_probability"] == row["E2_any_GO"]
                assert row["funding_routes_joint"]["E6_only"] == 0
            if "stage_counts" in design:
                assert 480 <= row["expected_screen_controller_episodes"] <= 960
                # Final outcomes partition all replicates, including futility.
                assert sum(row["E2_joint"]["joint_patterns"].values()) == pytest.approx(1)
        summary = replacement.summarize(rows)
        assert summary["minimum_joint_funding_at_40_nonnegative_survival"] <= 1


def test_budget_charges_two_times_liability_reserve_and_e6() -> None:
    invoice = replacement.screening_invoice(960, 1, e6=True)
    assert invoice["screening_units_including_reserve_and_E6"] == pytest.approx(
        (1860 * 4 / 3600) / 0.8 + 4
    )
    with pytest.raises(ValueError):
        replacement.screening_invoice(960, 0, e6=False)


def test_proposed_schema_cannot_be_consumed_by_live_admission(tmp_path: Path) -> None:
    report = {
        "schema": "bl-ppo-e9-replacement-oc/v1",
        "status": "PROPOSED_AWAITING_FUNDING_JUDGE",
        "selection": {"dispatch_admitted": False},
    }
    target = tmp_path / "report.json"
    export_json(target, report)
    assert len(verify_export(target)) == 64
    with pytest.raises(FileExistsError):
        export_json(target, report)
    with pytest.raises(ValueError, match="schema or frozen rule mismatch"):
        oc.validate_admission(report)
