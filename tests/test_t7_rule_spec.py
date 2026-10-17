"""Synthetic fixed-N checks for the T7 pure decision rule (no gameplay)."""

from __future__ import annotations

import copy

import pytest

from research.redesign_t7_phase_r_20261016 import rule_spec_t7 as t7


def _banks():
    return {seed: list(range(seed * 10_000, seed * 10_000 + 192)) for seed in t7.SEEDS}


def _entries():
    rows = []
    for seed, worlds in _banks().items():
        for mix, count in t7.MIX_COUNTS.items():
            for index, world in enumerate(worlds[:count]):
                noise = ((world % 11) - 5) * 0.2
                base_mass = 100.0 + noise
                base_survival = 0.72 + (index % 5) * 0.01
                for hero in (t7.HERO, t7.INCUMBENT, t7.CHAMPION):
                    if hero == t7.HERO:
                        delta = 65.0 + ((index % 3) - 1) * 0.5
                        mi5, mi10, survival, deaths = (
                            base_mass + delta,
                            base_mass + delta + 5.0,
                            base_survival + 0.08,
                            0.0,
                        )
                    elif hero == t7.INCUMBENT:
                        mi5, mi10, survival, deaths = (
                            base_mass,
                            base_mass,
                            base_survival,
                            0.0 if index % 5 else 1.0,
                        )
                    else:
                        mi5, mi10, survival, deaths = (
                            base_mass + 4.0,
                            base_mass + 4.0,
                            base_survival,
                            1.0 if index % 3 else 0.0,
                        )
                    rows.append(
                        {
                            "hero": hero,
                            "seed": seed,
                            "mix": mix,
                            "world_seed": world,
                            "horizon": 10000,
                            "control": False,
                            "prefix_h5000": {
                                "mass_integral": mi5,
                                "survival_fraction": survival,
                                "deaths": deaths,
                            },
                            "record": {
                                "mass_integral": mi10,
                                "survival_fraction": survival,
                                "deaths": deaths,
                            },
                        }
                    )
    return rows


def _decision(rows=None, controls=True):
    return t7.decide(copy.deepcopy(_entries() if rows is None else rows), _banks(), controls)


def _hero(rows, *, seed=None, mix=None):
    return [
        r
        for r in rows
        if r["hero"] == t7.HERO
        and (seed is None or r["seed"] == seed)
        and (mix is None or r["mix"] == mix)
    ]


def test_rule_is_stable_and_base_data_strictly_passes():
    first, second = t7.rule(), t7.rule()
    assert first["rule_sha256"] == second["rule_sha256"]
    assert first["cells"]["guard"]["seeds"] == [0]
    decision = _decision()
    assert decision["valid"] and decision["status"] == "STRICT_PASS"
    assert decision["candidate_selection"]["seed"] in t7.SEEDS
    assert decision["tier2"]["pass"]
    assert decision["clauses"]["tier1_absolute_episode_floor"]["by_mix"]["scripted"]["n"] == 960
    assert decision["clauses"]["tier1_absolute_episode_floor"]["by_mix"]["frozen"]["n"] == 480
    assert decision["sensitivity"]["champion"]["scripted"]["episode"]["n"] == 960


@pytest.mark.parametrize(
    ("name", "change"),
    [
        (
            "c1_mass_point",
            lambda rows: [_set(r["prefix_h5000"], "mass_integral", 105.0) for r in _hero(rows)],
        ),
        (
            "c2_scripted_mass",
            lambda rows: [
                _set(r["prefix_h5000"], "mass_integral", 99.0) for r in _hero(rows, mix="scripted")
            ],
        ),
        (
            "c2_scripted_alive",
            lambda rows: [
                _set(r["prefix_h5000"], "survival_fraction", 0.60)
                for r in _hero(rows, mix="scripted")
            ],
        ),
        (
            "c2_frozen_alive",
            lambda rows: [
                _set(r["prefix_h5000"], "survival_fraction", 0.60)
                for r in _hero(rows, mix="frozen")
            ],
        ),
        (
            "c2_mixed_alive",
            lambda rows: [
                _set(r["prefix_h5000"], "survival_fraction", 0.60) for r in _hero(rows, mix="mixed")
            ],
        ),
        ("c5_mi10", lambda rows: [_set(r["record"], "mass_integral", 105.0) for r in _hero(rows)]),
        (
            "c6_guard_mass_seed0",
            lambda rows: [
                _set(r["prefix_h5000"], "mass_integral", 90.0) for r in _hero(rows, seed=0)
            ],
        ),
        (
            "tier1_per_seed_alive_floor",
            lambda rows: [
                _set(r["prefix_h5000"], "survival_fraction", 0.40)
                for r in _hero(rows, seed=0, mix="scripted")
            ],
        ),
        (
            "tier1_absolute_episode_floor",
            lambda rows: [
                _set(r["prefix_h5000"], "deaths", 1.0) for r in _hero(rows, mix="scripted")
            ],
        ),
    ],
)
def test_each_point_survival_and_guard_clause_can_fail(name, change):
    rows = _entries()
    change(rows)
    decision = _decision(rows)
    assert decision["status"] == "FAIL"
    assert not decision["clauses"][name]["pass"]


def _set(mapping, key, value):
    mapping[key] = value


def test_positive_seed_and_prefix_control_clauses_fail():
    rows = _entries()
    for row in _hero(rows):
        row["prefix_h5000"]["mass_integral"] = 99.0
    decision = _decision(rows)
    assert not decision["clauses"]["c4_positive_seeds"]["pass"]
    controls = _decision(controls=False)
    assert not controls["clauses"]["c5_prefix_controls"]["pass"]


def test_feasibility_and_full_look_futility_are_reported():
    rows = _entries()
    # Large within-mix variation with a modest positive effect makes c7's fixed-N projection fail.
    for row in _hero(rows):
        baseline = 100.0 + ((row["world_seed"] % 11) - 5) * 0.2
        row["prefix_h5000"]["mass_integral"] = (
            baseline + 25.0 + (600.0 if row["world_seed"] % 2 else -600.0)
        )
    decision = _decision(rows, controls=False)
    assert not decision["clauses"]["c7_feasibility"]["pass"]
    assert decision["status"] == "FAIL"
    rows = _entries()
    for row in _hero(rows):
        row["prefix_h5000"]["mass_integral"] = 125.0
        row["record"]["mass_integral"] = 125.0
    decision = _decision(rows, controls=False)
    assert decision["futility"]["triggered"]
    assert "NO_GO_EARLY: mass_futility" in decision["reasons"]


def test_tier2_is_diagnostic_and_candidate_excludes_floor_failures():
    rows = _entries()
    for row in _hero(rows, seed=0, mix="scripted"):
        row["prefix_h5000"]["survival_fraction"] = 0.40
    for row in rows:
        if row["hero"] == t7.HERO and row["mix"] == "scripted":
            row["prefix_h5000"]["deaths"] = 1.0
    decision = _decision(rows)
    assert 0 not in decision["candidate_eligible_seeds"]
    assert decision["candidate_selection"]["seed"] != 0
    assert not decision["tier2"]["pass"]
    assert decision["status"] == "FAIL"


def test_incomplete_duplicate_nonfinite_and_controls_are_handled_fail_closed():
    rows = _entries()
    assert _decision(rows + [dict(rows[0])])["status"] == "INVALID_ANALYSIS"
    assert _decision(rows[:-1])["status"] == "INVALID_ANALYSIS"
    rows = _entries()
    rows[0]["prefix_h5000"]["mass_integral"] = float("nan")
    assert _decision(rows)["status"] == "INVALID_ANALYSIS"
    control = dict(_entries()[0])
    control["control"] = True
    assert _decision(_entries() + [control])["status"] == "STRICT_PASS"


def test_bank_and_episode_contracts_fail_closed():
    banks = _banks()
    banks[1][0] = banks[0][0]
    assert t7.decide(_entries(), banks, True)["status"] == "INVALID_ANALYSIS"
    rows = _entries()
    rows[0]["prefix_h5000"]["deaths"] = 2.0
    assert _decision(rows)["status"] == "INVALID_ANALYSIS"
    rows = _entries()
    rows[0]["prefix_h5000"]["survival_fraction"] = 1.1
    assert _decision(rows)["status"] == "INVALID_ANALYSIS"


def _baseline_mass(row):
    return 100.0 + ((row["world_seed"] % 11) - 5) * 0.2


def _set_seed_effect(rows, seed, effect, spread=0.0):
    for row in _hero(rows, seed=seed):
        offset = ((row["world_seed"] % 3) - 1) * spread
        row["prefix_h5000"]["mass_integral"] = _baseline_mass(row) + effect + offset


def test_candidate_uses_lower_endpoint_not_highest_mean():
    rows = _entries()
    _set_seed_effect(rows, 0, 80.0, spread=500.0)  # Highest mean, weak lower endpoint.
    _set_seed_effect(rows, 1, 70.0, spread=0.25)
    decision = _decision(rows)
    ranking = decision["candidate_selection"]["ranking"]
    assert decision["candidate_selection"]["seed"] == 1
    assert (
        next(row for row in ranking if row["seed"] == 0)["effect"]
        > next(row for row in ranking if row["seed"] == 1)["effect"]
    )


def test_candidate_exact_tie_selects_lowest_seed():
    rows = _entries()
    _set_seed_effect(rows, 0, 80.0, spread=0.25)
    _set_seed_effect(rows, 1, 80.0, spread=0.25)
    decision = _decision(rows)
    assert decision["candidate_selection"]["seed"] == 0
    ranking = decision["candidate_selection"]["ranking"]
    assert ranking[0]["lower"] == pytest.approx(ranking[1]["lower"])


def test_candidate_is_null_when_no_seed_passes_its_floor():
    rows = _entries()
    for seed in t7.SEEDS:
        for row in _hero(rows, seed=seed, mix="scripted"):
            row["prefix_h5000"]["survival_fraction"] = 0.40
    decision = _decision(rows)
    assert decision["candidate_eligible_seeds"] == []
    assert decision["candidate_selection"] is None


def test_tier2_failure_is_nonblocking_when_tier1_passes():
    rows = _entries()
    for row in rows:
        if row["hero"] == t7.HERO and row["world_seed"] % 5 < 2:
            row["prefix_h5000"]["deaths"] = 1.0  # Hero episode survival remains 0.60.
        if row["hero"] == t7.CHAMPION:
            row["prefix_h5000"]["deaths"] = 0.0
    decision = _decision(rows)
    assert decision["clauses"]["tier1_absolute_episode_floor"]["pass"]
    assert not decision["tier2"]["pass"]
    assert decision["status"] == "STRICT_PASS"


def test_full_pass_precedes_mass_futility_at_ub_below_50():
    rows = _entries()
    for row in _hero(rows):
        offset = ((row["world_seed"] % 3) - 1) * 1.0
        row["prefix_h5000"]["mass_integral"] = _baseline_mass(row) + 30.0 + offset
        row["record"]["mass_integral"] = _baseline_mass(row) + 35.0 + offset
    decision = _decision(rows)
    assert decision["futility"]["upper_bound"] < 50.0
    assert decision["futility"]["triggered"]
    assert decision["status"] == "STRICT_PASS"
    assert "NO_GO_EARLY: mass_futility" not in decision["reasons"]
