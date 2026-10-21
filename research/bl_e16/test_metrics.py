"""Synthetic E16 metric tests; no simulator, checkpoint, or bank data."""

from __future__ import annotations

import json

import pytest

from research.bl_e16.metrics import (
    MIXES,
    decide_arm,
    holm_one_sided_bounds,
    ledger_by_mix,
    paired_equal_mix_bounds,
    simultaneous_bonferroni_bounds,
    survival_band_v2,
    whole_life_ledger,
)


def _life(value: float, death: int | None = None, horizon: int = 10) -> list[float]:
    return [value if death is None or frame < death else 0.0 for frame in range(horizon)]


def test_three_term_ledger_identity_and_collateral_over_all_worlds() -> None:
    ledger = whole_life_ledger(_life(12, death=None), _life(10, death=5), horizon=10)
    assert ledger["total_delta"] == pytest.approx(
        ledger["common_alive"] + ledger["candidate_only"] - ledger["b8_only"]
    )
    assert ledger["common_alive"] == pytest.approx(1.0)
    assert ledger["candidate_only"] == pytest.approx(6.0)
    assert ledger["b8_only"] == 0.0
    assert ledger["productive_rescue"] is True
    assert ledger["tail_retention"] == pytest.approx(1.2)

    loss = whole_life_ledger(_life(7), _life(10), horizon=10)
    assert loss["collateral"] == pytest.approx(3.0)
    assert loss["induced_death"] is False


def test_productive_and_induced_death_have_the_frozen_definitions() -> None:
    productive = whole_life_ledger(_life(9), _life(10, death=4), horizon=10)
    delayed_but_not_productive = whole_life_ledger(_life(8), _life(10, death=4), horizon=10)
    induced = whole_life_ledger(_life(8, death=4), _life(10), horizon=10)
    assert productive["productive_rescue"] is True
    assert delayed_but_not_productive["productive_rescue"] is False
    assert induced["induced_death"] is True


def test_terminal_whole_life_trace_rejects_resurrection() -> None:
    with pytest.raises(ValueError, match="resurrects"):
        whole_life_ledger([10, 10, 0, 10], [10, 10, 10, 10], horizon=4)


def test_ledger_is_stratified_and_averages_collateral_over_natural_worlds() -> None:
    rows = []
    for mix in MIXES:
        rows.extend(
            [
                {
                    "mix": mix,
                    "candidate_lengths": _life(12),
                    "b8_lengths": _life(10, death=5),
                    "horizon": 10,
                },
                {"mix": mix, "candidate_lengths": _life(7), "b8_lengths": _life(10), "horizon": 10},
            ]
        )
    summary = ledger_by_mix(rows)
    assert summary["scripted"]["productive_rescues_per_100"] == 50.0
    assert summary["scripted"]["collateral"] == pytest.approx(1.5)
    strata = summary["scripted"]["b8_deathframe_x_last_length"]
    assert len(strata) == 20
    assert {row["b8_death_frame_bin"] for row in strata} == {
        "<2000",
        "2000-2999",
        "3000-3999",
        "4000-4999",
        "survivor",
    }
    observed = next(row for row in strata if row["n"] == 1 and row["productive_rescue_count"] == 1)
    assert observed["candidate_only"] is not None


def test_rescue_tail_retention_is_reported_separately_from_all_b8_deaths() -> None:
    rows = []
    for mix in MIXES:
        rows.extend(
            [
                {
                    "mix": mix,
                    "candidate_lengths": _life(10),
                    "b8_lengths": _life(10, death=5),
                    "horizon": 10,
                },
                {
                    "mix": mix,
                    "candidate_lengths": _life(10, death=6),
                    "b8_lengths": _life(10, death=5),
                    "horizon": 10,
                },
            ]
        )
    summary = ledger_by_mix(rows)["scripted"]
    assert summary["rescue_tail_retention"] == [1.0]
    assert len(summary["all_b8_death_tail_retention"]) == 2


def test_equal_mix_bound_uses_registered_variance_formula() -> None:
    groups = {mix: [0.0, 2.0, 4.0] for mix in MIXES}
    result = paired_equal_mix_bounds(groups)
    # s^2=4 and n=3 for every mix: sum_m 4/(9*3)=4/9.
    assert result["variance"] == pytest.approx(4.0 / 9.0)
    assert result["estimate"] == pytest.approx(2.0)


def test_zero_variance_bound_is_json_safe() -> None:
    result = paired_equal_mix_bounds({mix: [1.0, 1.0] for mix in MIXES})
    assert result["df"] is None
    assert result["lower"] == result["upper"] == 1.0
    json.dumps(result, allow_nan=False)


def test_bonferroni_companion_is_simultaneous_for_j_g_and_collateral() -> None:
    groups = {mix: [1.0, 2.0, 3.0] for mix in MIXES}
    result = simultaneous_bonferroni_bounds(
        {
            "A:J": {"groups": groups, "direction": "lower"},
            "A:G": {"groups": groups, "direction": "upper"},
            "A:collateral": {"groups": groups, "direction": "upper"},
        }
    )
    assert result["A:J"]["bonferroni_alpha_one_sided"] == pytest.approx(0.10 / 3.0)
    assert result["A:J"]["simultaneous_bound"] == result["A:J"]["lower"]
    assert result["A:G"]["simultaneous_bound"] == result["A:G"]["upper"]


def test_holm_is_simultaneous_and_not_per_arm_alpha() -> None:
    strong = {mix: [10.0, 11.0, 12.0, 13.0] for mix in MIXES}
    weak = {mix: [-3.0, -2.0, -1.0, 0.0] for mix in MIXES}
    result = holm_one_sided_bounds({"A:J": strong, "G:J": weak})
    assert result["A:J"]["family_alpha_one_sided"] == 0.10
    assert result["A:J"]["holm_step_alpha"] == pytest.approx(0.05)
    assert result["A:J"]["holm_rejects_nonpositive"] is True
    assert result["G:J"]["holm_rejects_nonpositive"] is False


def test_survival_guard_a_uses_pooled_mix_catastrophe_and_floor() -> None:
    candidate = {mix: [0.9, 0.9, 0.9] for mix in MIXES}
    b8 = {mix: [0.9, 0.9, 0.9] for mix in MIXES}
    assert survival_band_v2(candidate, b8)["pass"] is True
    candidate["scripted"] = [0.1, 0.1, 0.1]
    assert survival_band_v2(candidate, b8)["pass"] is False


def test_decision_precedence_unranked_invalid_then_kill() -> None:
    assert (
        decide_arm(
            available=False,
            valid=False,
            tripwires=["nonfinite"],
            j=None,
            g=None,
            ledger=None,
            collateral=None,
            survival=None,
        )["status"]
        == "UNRANKED"
    )
    assert (
        decide_arm(
            available=True,
            valid=False,
            tripwires=[],
            j=None,
            g=None,
            ledger=None,
            collateral=None,
            survival=None,
        )["status"]
        == "INCOMPLETE"
    )
    assert (
        decide_arm(
            available=True,
            valid=True,
            tripwires=["nonfinite"],
            j=None,
            g=None,
            ledger=None,
            collateral=None,
            survival=None,
        )["status"]
        == "KILL"
    )


def test_decision_admission_and_artifact_rejection() -> None:
    ledger = {
        "equal_mix": {"common_alive": 1.0},
        **{
            mix: {"productive_rescues_per_100": 5.0, "rescue_tail_retention": [0.9]}
            for mix in MIXES
        },
    }
    common = dict(
        available=True,
        valid=True,
        tripwires=[],
        ledger=ledger,
        collateral={"upper": 10.0},
        survival={"pass": True},
    )
    admitted = decide_arm(
        j={"lower": 0.1, "holm_rejects_nonpositive": True},
        g={"estimate": 10.0, "upper": 41.0},
        **common,
    )
    assert admitted["status"] == "ADMIT_100M"
    rejected = decide_arm(
        j={"lower": -1.0, "holm_rejects_nonpositive": False},
        g={"estimate": 9.0, "upper": 39.9},
        **common,
    )
    assert rejected["status"] == "INCONCLUSIVE"
    assert rejected["artifact_status"] == "ARTIFACT_REJECTED_AT_BAR"
