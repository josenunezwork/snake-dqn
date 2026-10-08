"""Pin of the M3 sequential Phase R rule spec and plan (amendment condition 1)."""

from __future__ import annotations

from research.redesign_m3_20261008 import rule_spec
from src.evaluation.sequential_phase_r import _sha

RULE_SHA256 = "fd42a5d5ba898e41c2f4b251ac7833a461ed7773e5c91290f4d5015c00e04267"
PLAN_SHA256 = "5d1ee49d0ce8ce0f764f61909ad1803b542e825c70a08c01c12b83636651304e"


def test_rule_and_plan_are_pinned():
    assert _sha(rule_spec.m3_rule()) == RULE_SHA256
    assert rule_spec.m3_plan().sha256() == PLAN_SHA256


def test_plan_matches_the_preregistration():
    plan = rule_spec.m3_plan()
    assert plan.n_worlds == 32 and plan.look_sizes == (11, 22, 32)
    assert [round(p, 4) for p in plan.go_nominal_p] == [0.005, 0.0457, 0.086]
    rule = rule_spec.m3_rule()
    assert rule["seeds"] == [0, 1, 2, 3, 4]
    assert rule["no_go"]["all_of"][0]["upper_below"] == 50.0
    assert rule["kill"]["all_of"][0]["upper_below"] == 20.0
    assert rule["cells"]["guard"]["seeds"] == [0]
