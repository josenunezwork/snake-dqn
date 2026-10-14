"""M7 growth governor (research/m7_governor_20261014/governor.py): action rule and summary."""

from __future__ import annotations

import numpy as np
import pytest

from research.m7_governor_20261014 import governor as gov


def _adv(rows):
    return np.array(rows, dtype=bool)


def test_off_is_identity():
    act = np.array([0, 1, 5])
    adv = _adv([[1] * 6] * 3)
    out, rep, blk = gov.GovernorEgo2sPolicy.govern(act, adv, np.array([9000] * 3), None)
    assert out.tolist() == [0, 1, 5] and not rep.any() and not blk.any()


def test_twin_swaps_only_over_cap_safe_normal_actions():
    act = np.array([0, 1, 2, 4, 2])
    length = np.array([801, 801, 801, 801, 800])
    adv = _adv(
        [
            [1, 1, 1, 1, 1, 1],  # over, twin 3 safe -> 3
            [1, 1, 1, 1, 0, 1],  # over, twin 4 unsafe -> keep 1 (blocked)
            [1, 1, 1, 0, 0, 1],  # over, twin 5 safe -> 5
            [1, 1, 1, 1, 1, 1],  # student already boosts -> keep
            [1, 1, 1, 1, 1, 1],  # L == cap is not over -> keep
        ]
    )
    out, rep, blk = gov.GovernorEgo2sPolicy.govern(act, adv, length, 800)
    assert out.tolist() == [3, 1, 5, 4, 2]
    assert rep.tolist() == [True, False, True, False, False]
    assert blk.tolist() == [False, True, False, False, False]


def test_bestboost_picks_highest_q_safe_boost():
    act = np.array([1, 1])
    adv = _adv([[1, 1, 1, 1, 0, 1], [1, 1, 1, 0, 0, 0]])
    q = np.array([[0, 9, 0, 1, 7, 2], [0, 9, 0, 5, 5, 5]], dtype=np.float32)
    out, rep, blk = gov.GovernorEgo2sPolicy.govern(
        act, adv, np.array([900, 900]), 800, q, "bestboost"
    )
    assert out.tolist() == [5, 1] and rep.tolist() == [True, False]
    assert blk.tolist() == [False, True]
    with pytest.raises(ValueError):
        gov.GovernorEgo2sPolicy.govern(act, adv, np.array([900, 900]), 800, q, "nope")


def test_governed_patch_is_restored():
    from src.simd_env import ego2s_policy as ep

    orig = ep.Ego2sSimdPolicy
    with gov.governed(1000, "twin") as built:
        assert ep.Ego2sSimdPolicy is not orig and ep.Ego2sSimdPolicy.cap == 1000
        assert built == []
    assert ep.Ego2sSimdPolicy is orig


def test_arm_key_and_order():
    assert gov.arm_key({"cap": "off", "mode": "bestboost"}) == "off"
    assert gov.arm_key({"cap": "900"}) == "900"
    assert gov.arm_key({"cap": "900", "mode": "bestboost"}) == "900/bestboost"
    keys = ["900/bestboost", "1100", "off", "800"]
    assert sorted(keys, key=gov._arm_order) == ["off", "800", "1100", "900/bestboost"]


def _row(name, cap, mix, deaths, mass=100.0):
    pw = {
        str(i): {
            "decisions": 10,
            "over_cap": 0,
            "replaced": 0,
            "blocked": 0,
            "student_boost": 0,
            "ge800": 5,
            "ge1000": 0,
            "ge1200": 0,
            "max_len": 900,
            "len_at_death": 900 if d else None,
            "cause": "self" if d else None,
            "since_governor_boost": None,
        }
        for i, d in enumerate(deaths)
    }
    return {
        "name": name,
        "cap": cap,
        "mix": mix,
        "mass": {str(i): mass for i in range(len(deaths))},
        "deaths": {str(i): d for i, d in enumerate(deaths)},
        "per_world": pw,
    }


def test_verdict_rule():
    inc = 434.9
    arms = {
        "off": {"per_mix": {"scripted": {"survival": 0.53}}, "mass_stratified": 545.0},
        "900": {"per_mix": {"scripted": {"survival": 0.65}}, "mass_stratified": 500.0},
    }
    assert gov.verdict(arms, inc)["verdict"] == "GO"
    arms["900"]["mass_stratified"] = 490.0
    assert gov.verdict(arms, inc)["verdict"] == "INCONCLUSIVE"
    arms["900"]["per_mix"]["scripted"]["survival"] = 0.59
    assert gov.verdict(arms, inc)["verdict"] == "NO_GO"


def test_arm_summary_and_paired_delta():
    rows = [_row("B8_s0", "900", m, [0, 1, 0, 0]) for m in gov.MIXES]
    off = [_row("B8_s0", "off", m, [1, 1, 0, 0]) for m in gov.MIXES]
    a = gov.arm_summary(rows, ["B8_s0"])
    sc = a["per_mix"]["scripted"]
    assert sc["survival"] == 0.75 and sc["deaths_L_ge_800"] == 1
    assert sc["hazard_per_1000_L_ge_800"] == pytest.approx(1000 / 20)
    assert a["mass_stratified"] == 100.0
    d = gov.paired_scripted_delta(rows, off)
    assert d["pairs"] == 4 and d["delta_survival"] == 0.25 and d["rescued"] == 1


def test_twinspace_requires_space_ok():
    act = np.array([0, 1, 2])
    adv = _adv([[1] * 6] * 3)
    space = np.array([True, False, True])
    out, rep, blk = gov.GovernorEgo2sPolicy.govern(
        act, adv, np.array([900, 900, 700]), 800, None, "twinspace", space_ok=space
    )
    assert out.tolist() == [3, 1, 2]
    assert rep.tolist() == [True, False, False] and blk.tolist() == [False, True, False]
