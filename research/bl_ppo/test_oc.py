"""Targeted synthetic E9 family, prospective bank and immutable report tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from research.bl_ppo import bank, oc, rule
from research.bl_ppo.exports import export_json, verify_export


def test_stratified_variance_uses_unequal_mix_counts() -> None:
    assert oc.stratified_variance(
        dict.fromkeys(oc.MIXES, 200**2), dict.fromkeys(oc.MIXES, 96)
    ) == pytest.approx(200**2 / 288)
    assert oc.stratified_variance(dict.fromkeys(oc.MIXES, 200**2), oc.COUNTS[0]) == pytest.approx(
        324.074074074
    )


def test_holm_keeps_both_slots_and_steps_down() -> None:
    assert rule.holm_rejections({"R": 0.024, "F": 0.049}) == {"R": True, "F": True}
    assert rule.holm_rejections({"R": 0.026, "F": 0.04}) == {"R": False, "F": False}
    assert rule.holm_rejections({"R": 0.03, "F": 1})["R"] is False
    with pytest.raises(ValueError):
        rule.holm_rejections({"R": 0.01})
    with pytest.raises(ValueError):
        rule.holm_rejections({"R": float("nan"), "F": 1})
    p = np.array([[0.024, 0.049], [0.026, 0.04], [0.9, 0.001]])
    assert oc._holm(p).tolist() == [[True, True], [False, False], [False, True]]


def test_efficacy_zero_variance_and_unequal_mix_weights() -> None:
    stats = rule.paired_efficacy(
        {"scripted": [30.0] * 96, "frozen": [0.0] * 32, "mixed": [0.0] * 32}
    )
    assert stats["estimate"] == 10
    assert stats["pvalue"] == 0
    assert stats["lower"] == 10
    assert rule.mass_pvalue({"estimate": 0, "se": 0, "df": None}) == 1


def test_new_bank_preserves_legacy_bytes() -> None:
    legacy = bank.materialize(0, allocation_version=1)
    prior = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/screen-bank-v2/bank.json"
    )
    if prior.exists():
        import json

        assert legacy == json.loads(prior.read_text())
    assert legacy["counts"] == bank.LEGACY_COUNTS
    current = bank.materialize(0)
    assert current["counts"] == {"scripted": 128, "frozen": 48, "mixed": 48}
    assert not bank.all_seeds(current) & bank.all_seeds(legacy)
    bank.validate_bank(legacy)
    bank.validate_bank(current)


def test_report_export_sidecars_and_admission_fail_closed(tmp_path: Path) -> None:
    report = {
        "schema": "bl-ppo-e9-oc/v1",
        "rule": rule.RULE,
        "selection": {"dispatch_admitted": False, "reason": "power"},
    }
    path = tmp_path / "report.json"
    export_json(path, report)
    assert len(verify_export(path)) == 64
    with pytest.raises(ValueError, match="admission blocked"):
        oc.validate_admission(report)
    with pytest.raises(FileExistsError):
        export_json(path, report)
    path.write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        verify_export(path)


def test_admission_rejects_forged_probability_summary() -> None:
    report = {
        "schema": "bl-ppo-e9-oc/v1",
        "rule": rule.RULE,
        "selection": {"dispatch_admitted": True, "counts": oc.COUNTS[1]},
        "measured_inputs": {
            "sha256": "a" * 64,
            "mixes": dict.fromkeys(
                oc.MIXES,
                {
                    "paired_mass_variance": 1.0,
                    "paired_alive_variance": 1.0,
                    "paired_episode_variance": 1.0,
                },
            ),
        },
        "S1_inputs": {
            "sha256": "b" * 64,
            "mixes": dict.fromkeys(
                oc.MIXES,
                {
                    "paired_mass_variance": 1.0,
                    "paired_alive_variance": 1.0,
                    "paired_episode_variance": 1.0,
                },
            ),
        },
        "summaries": [{"counts": oc.COUNTS[1], "passes_probability_bars": True}],
        "scenarios": [
            {
                "counts": oc.COUNTS[1],
                "rho_shared_B8": r,
                "true_mass_and_early_dividend": g,
                "true_survival_delta": d,
                "family_funding_probability": 0.01 if g == 0 else 0.5,
            }
            for r in (0.0, 0.5, 0.9)
            for g in (0.0, 20.0, 40.0, 60.0, 80.0)
            for d in (-0.05, 0.0, 0.03, 0.06)
        ],
    }
    with pytest.raises(ValueError, match="probability bars fail"):
        oc.validate_admission(report)
    altered = copy.deepcopy(report)
    altered["rule"]["one_look"] = False
    with pytest.raises(ValueError, match="rule mismatch"):
        oc.validate_admission(altered)


def test_wilson_lower_is_one_sided_90() -> None:
    assert 0.15 < oc.wilson_lower(26, 128) < 0.17
    with pytest.raises(ValueError):
        oc.wilson_lower(2, 1)
