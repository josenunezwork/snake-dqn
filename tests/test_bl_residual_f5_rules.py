"""Synthetic-only R10c parity tests for :mod:`research.bl_residual.f5_rules`."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from research.bl_residual import f5_rules as rules


def _mix(value: float) -> dict[str, list[float]]:
    return {name: [value] * 128 for name in rules.MIXES}


def test_r10c_selection_harm_and_cov_boundaries() -> None:
    assert rules.select_candidate() == "HOLD_DIAGNOSTIC"
    assert rules.select_candidate(1, 0, 0, valid=False) == "INVALID"
    assert rules.select_candidate(0, 0, 0) == "HOLD_NO_TARGET"
    assert rules.select_candidate(2, 2, 0) == "COV"
    assert rules.select_candidate(4, 1, 2) == "P"
    assert rules.select_candidate(4, 1, 1) == "W"
    assert rules.select_candidate(1, 0, 1) == "P"
    assert rules.select_candidate(1, 0, 0) == "W"
    assert rules.select_candidate(1, 0, 2) == "INVALID"
    assert rules.select_candidate(True, 0, 0) == "INVALID"
    assert [rules.margin_bin(x) for x in (".009999999", ".01", ".1", "1")] == [0, 1, 2, 3]
    assert rules.conservative_harm(True, False, True, False, True)
    assert not rules.conservative_harm(True, True, True, False, True)
    assert rules.cov_status({1, 2}, [1]) == "INCOMPLETE"
    assert rules.cov_status({1, 2}, [1, 1]) == "INVALID"
    assert rules.cov_status(set(range(46_081)), []) == "HOLD_COVERAGE_IMPLEMENTATION"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        (".009999999", 0),
        (".01", 1),
        (".099999999", 1),
        (".1", 2),
        (".999999999", 2),
        ("1", 3),
        ("100", 3),
    ],
)
def test_r10c_margin_bin_exact_boundaries(value: str, expected: int) -> None:
    # The governing oracle freezes .01/.1/1, not a later .5 proposal.
    assert rules.margin_bin(value) == expected


@pytest.mark.parametrize("value", ["-.0001", "NaN", "Infinity"])
def test_r10c_margin_bin_refuses_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        rules.margin_bin(value)


def test_p7_boundaries_and_control_drift_close_matched_study() -> None:
    assert not rules.fit_drift([Fraction(1, 20)] * 10, Fraction(3, 2), 1, Fraction(7, 10))
    assert not rules.fit_drift([0.05] * 10, 1.5, 1.0, 0.70)
    assert rules.fit_drift([math.nextafter(0.05, math.inf)] * 10, 0, 0, 1)
    assert rules.fit_drift([0] * 10, math.nextafter(1.5, math.inf), 1, 1)
    assert rules.fit_drift([0] * 10, 0, 0, math.nextafter(0.70, -math.inf))
    out = rules.r6_adapter(t_drift=False, c_drift=True)
    assert out["status"] == "KILL_FIT" and out["affected_fits"] == ["C"]
    derived = rules.r6_adapter(t_drift=False, c_drift=False, t_retention=0.69)
    assert derived["status"] == "KILL_FIT" and derived["affected_fits"] == ["T"]
    assert (
        rules.support_status(rescue_parents=31, natural_rows=1000, scheduled_census_complete=True)
        == "INSUFFICIENT"
    )
    assert (
        rules.support_status(rescue_parents=32, natural_rows=999, scheduled_census_complete=True)
        == "INSUFFICIENT"
    )
    assert (
        rules.support_status(rescue_parents=32, natural_rows=1000, scheduled_census_complete=False)
        == "INCOMPLETE"
    )


def test_p7_rank_deciles_sort_ties_and_refuse_incomplete_or_invalid_rows() -> None:
    rows = [
        {"length": 1, "bank_index": index, "decision": index, "disagrees": index == 10}
        for index in range(20)
    ]
    deciles = rules.rank_decile_disagreements(rows)
    assert len(deciles) == 10 and deciles[5] == 0.5
    with pytest.raises(ValueError):
        rules.fit_drift([0] * 9, 0, 0, 1)
    with pytest.raises(ValueError):
        rules.rank_decile_disagreements(rows[:9])
    with pytest.raises(ValueError):
        rules.rank_decile_disagreements([{**rows[0], "length": -1}] + rows[1:])


def test_action_collapse_sliding_window_and_exact_boundaries() -> None:
    assert not rules.action_collapse_flag([0] * 99_999)
    assert not rules.action_collapse_flag([0] * 90_000 + [1] * 10_000)
    assert rules.action_collapse_flag([0] * 90_001 + [1] * 9_999)
    assert rules.action_collapse_flag([1] + [0] * 90_001 + [1] * 9_999)


def test_r7_welch_holm_guards_and_r9_precedence() -> None:
    good = rules.r7_adapter(
        top=_mix(40),
        change=_mix(20),
        t_survival=_mix(1),
        b8_survival=_mix(1),
        scripted_no_death_gain=0.03,
        delivered_boost_share=0.005,
    )
    assert good["all_h5000_bars_pass"] and good["status"] == "PASS_H5000_OPEN_S4"
    closed = rules.r7_adapter(
        top=_mix(-1),
        change=_mix(0),
        t_survival=_mix(1),
        b8_survival=_mix(1),
        scripted_no_death_gain=0.03,
        delivered_boost_share=0.005,
    )
    assert "KILL_CHANGE_CLOSED" in closed["flags"] and closed["status"] == "KILL_GAMEPLAY_GUARD"
    assert rules.r9_disposition({"INVALID", "KILL_FIT"}) == "INVALID"
    assert rules.r9_disposition({"INCOMPLETE", "KILL_FIT"}) == "INCOMPLETE"
    assert rules.r9_disposition({"HOLD_DIAGNOSTIC"}) == "HOLD_DIAGNOSTIC"
    assert (
        rules.r9_disposition(set(), ub_change=math.nextafter(20, -math.inf)) == "KILL_CHANGE_FUTILE"
    )
    assert rules.r9_disposition(set(), ub_change=20, all_pass=False) == "INCONCLUSIVE"
    assert rules.r9_disposition(set(), ub_change=Fraction(20), all_pass=True) == "GO_F5_DEV"
    assert rules.r9_disposition({"KILL_PRODUCT"}) == "KILL_PRODUCT"
    assert rules.r9_disposition({"KILL_CHANGE_CLOSED", "KILL_PRODUCT"}) == "KILL_CHANGE_CLOSED"
    assert rules.r9_disposition({"KILL_CHANGE_FUTILE", "KILL_PRODUCT"}) == "KILL_CHANGE_FUTILE"


def test_r7_nextafter_bar_boundaries_and_absolute_survival_floor() -> None:
    base = dict(
        t_survival=_mix(0.30),
        b8_survival=_mix(0.30),
        scripted_no_death_gain=0.03,
        delivered_boost_share=0.005,
    )
    assert rules.r7_adapter(top=_mix(40), change=_mix(20), **base)["status"] == "PASS_H5000_OPEN_S4"
    assert (
        rules.r7_adapter(top=_mix(math.nextafter(40, -math.inf)), change=_mix(20), **base)["status"]
        == "INCONCLUSIVE"
    )
    assert (
        rules.r7_adapter(top=_mix(40), change=_mix(math.nextafter(20, -math.inf)), **base)["status"]
        == "KILL_CHANGE_FUTILE"
    )
    low_floor = rules.r7_adapter(
        top=_mix(40),
        change=_mix(20),
        t_survival=_mix(math.nextafter(0.30, -math.inf)),
        b8_survival=_mix(0.30),
        scripted_no_death_gain=0.03,
        delivered_boost_share=0.005,
    )
    assert low_floor["status"] == "KILL_GAMEPLAY_GUARD"


def test_product_bootstrap_is_paired_frozen_and_refuses_undefined_baseline() -> None:
    rows = [
        {
            "t_deaths": 5,
            "t_decisions": 1000,
            "b8_deaths": 10,
            "b8_decisions": 1000,
            "t_mass": 100,
            "b8_mass": 100,
        }
        for _ in range(128)
    ]
    first, second = rules.product_bootstrap(rows), rules.product_bootstrap(rows)
    assert first == second and first["passes"]
    rows[0]["b8_deaths"] = 0
    for row in rows[1:]:
        row["b8_deaths"] = 0
    assert rules.product_bootstrap(rows)["status"] == "INSUFFICIENT"


def test_r10c_product_ratio_and_reason_sets_are_not_rounded() -> None:
    rows = [
        {
            "t_deaths": 10,
            "t_decisions": 1000,
            "b8_deaths": 10,
            "b8_decisions": 1000,
            "t_mass": 94,
            "b8_mass": 100,
        }
        for _ in range(128)
    ]
    outcome = rules.product_bootstrap(rows)
    assert outcome["status"] == "KILL_PRODUCT" and not outcome["passes"]
    assert outcome["interpretation"] == {
        "classification": "EXPLORATORY_SCREENING",
        "status": "KILL_PRODUCT",
        "reasons": ["product_bar_failed"],
        "confirmatory": False,
        "release": False,
        "confirmatory_efficacy_claim": False,
        "release_authorized": False,
    }


def test_r10c_adoption_phase_and_whole_round_oracles() -> None:
    assert rules.adoption_order(1, 2, 3, 4, 5, 6, 7)
    assert not rules.adoption_order(4, 2, 3, 4, 5, 6, 7)
    assert not rules.adoption_order(1, 2, 5, 4, 5, 6, 7)
    assert not rules.adoption_order(1, 2, 3, 4, 5, 6, 6)
    assert 0 <= rules.ordinary_phase("synthetic-parent") < 200
    freq = [0] * 5000
    for phase in range(200):
        for decision in range(phase, 5000, 200):
            freq[decision] += 1
    assert set(freq) == {1} and 128 * 25 == 3200 and 128 * 90 == 11520
    assert (3200 + 11520) * 6 * 300 == 26_496_000
    assert 46_080 * 300 == 13_824_000
    rows = rules.allocate_round(0)
    assert len(rows) == 684 and len({row[1] for row in rows}) == 684
    round_index, shifted = rules.first_clear_round({rows[0][2]}, set())
    assert round_index >= 1 and shifted == rules.allocate_round(round_index)
    assert rules.first_clear_round(set(), set(), audit_complete=False) == ("HOLD", None)


def test_exact_survival_oc_is_deterministic_and_uses_actual_pooled_implementation() -> None:
    first = rules.survival_operating_characteristic(effect=0.0, trials=8, worlds_per_mix=8)
    second = rules.survival_operating_characteristic(effect=0.0, trials=8, worlds_per_mix=8)
    assert first == second
    assert first["synthetic_only"] and not first["confirmatory"] and not first["release"]
    assert 0 <= first["pass_rate"] <= 1


def test_exploratory_screening_never_claims_confirmation_or_release() -> None:
    assert rules.exploratory_screening("GO_F5_DEV") == {
        "classification": "EXPLORATORY_SCREENING",
        "status": "GO_F5_DEV",
        "reasons": [],
        "confirmatory": False,
        "release": False,
        "confirmatory_efficacy_claim": False,
        "release_authorized": False,
    }


def test_only_full_lifecycle_can_issue_go_f5_dev() -> None:
    assert rules.r6_adapter(t_drift=False, c_drift=False)["status"] == "PASS_OFFLINE_OPEN_S3"
    product = rules.product_bootstrap(
        [
            {
                "t_deaths": 5,
                "t_decisions": 1000,
                "b8_deaths": 10,
                "b8_decisions": 1000,
                "t_mass": 100,
                "b8_mass": 100,
            }
            for _ in range(128)
        ]
    )
    assert product["status"] == "PASS_PRODUCT"
    assert (
        rules.final_lifecycle(
            s2_status="PASS_OFFLINE_OPEN_S3",
            s3_status="PASS_H5000_OPEN_S4",
            s4_status="PASS_PRODUCT",
        )["status"]
        == "GO_F5_DEV"
    )
    assert (
        rules.final_lifecycle(
            s2_status="PASS_OFFLINE_OPEN_S3",
            s3_status="PASS_H5000_OPEN_S4",
            s4_status="NOT_RUN_BY_RULE",
        )["status"]
        == "INCONCLUSIVE"
    )
