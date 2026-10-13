"""M4 (DRAFT) Phase R rule specs, plans and world namespaces.

The pins are draft pins: ratification re-pins them (or confirms them) in the same commit.
"""

from __future__ import annotations

from research.apex_safety_20260926 import dev_screen
from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b
from research.redesign_m3_20261008 import student_record_parity
from research.redesign_m3c_20261008 import rule_spec_m3c
from research.redesign_m4_20261013 import rule_spec_m4 as rs
from research.redesign_scope_20261007 import grid_h5000_identity as gid
from research.sequential_phase_r.example_rules import frp_v4_like
from src.evaluation.sequential_phase_r import _sha

PRIMARY_RULE_SHA256 = "f614b80639e1b9b29f32a08ad38426e8b87644d5720cfe54a89d943a66f5e6b4"
SECONDARY_RULE_SHA256 = "e53691a6907e37294f9b720e7959d3e13650539f1501dd0a78d36bf92f865c46"
PRIMARY_PLAN_SHA256 = "72097708d03e81b4d4b56bc2ea013df4736b9b9f6c0c727ef2a1bb938f6f42d2"
SECONDARY_PLAN_SHA256 = "85d20b3a67de87ad24fbeca9cd4c2f3fe9888e14c5bb2e6fab30c6a53d22b5ea"
M3_TRAIN = "redesign-m3-train/v1"  # M3, M3-C and M4-A training worlds (keys seed0..)
M3_PHASE_R = "redesign-m3-phase-r/v1"
M4_SMOKE = "redesign-m4-phase-r-smoke/v1"
# Worlds used per distillation / probe round index (M3's list; M3-C / M4-A probe round 52 up
# to 32 worlds per mix).
USED = {0: 416, 1: 208, 2: 208, 9: 32, 10: 416, 11: 208, 12: 208, 50: 16, 51: 16, 52: 32}


def test_rules_and_plans_are_pinned():
    assert (rs.PRIMARY_RULE_SHA256, rs.SECONDARY_RULE_SHA256) == (
        PRIMARY_RULE_SHA256,
        SECONDARY_RULE_SHA256,
    )
    assert (rs.PRIMARY_PLAN_SHA256, rs.SECONDARY_PLAN_SHA256) == (
        PRIMARY_PLAN_SHA256,
        SECONDARY_PLAN_SHA256,
    )
    assert _sha(rs.primary_rule()) == PRIMARY_RULE_SHA256
    assert _sha(rs.secondary_rule()) == SECONDARY_RULE_SHA256
    assert rs.primary_plan().sha256() == PRIMARY_PLAN_SHA256
    assert rs.secondary_plan().sha256() == SECONDARY_PLAN_SHA256


def test_rules_match_the_draft():
    for rule, plan, hero in (
        (rs.primary_rule(), rs.primary_plan(), rs.HERO_NV),
        (rs.secondary_rule(), rs.secondary_plan(), rs.HERO_V8),
    ):
        assert plan.n_worlds == 32 and plan.look_sizes == (11, 22, 32)
        assert rule["seeds"] == [0, 1, 2, 3, 4]
        assert rule["no_go"]["all_of"][0]["upper_below"] == 50.0
        assert rule["kill"]["all_of"][0]["upper_below"] == 20.0
        assert rule["cells"]["primary"] == {
            "hero": hero,
            "baseline": "incumbent",
            "seeds": [0, 1, 2, 3, 4],
        }
        assert rule["cells"]["guard"] == {"hero": hero, "baseline": "champion", "seeds": [0]}
        names = {c["name"] for c in rule["go_clauses"]}
        assert {"c5_mi10", "c3_frozen_survival", "c2_scripted_survival"} <= names
        c4 = [c for c in rule["go_clauses"] if c["name"] == "c4_positive_seeds"]
        assert [c["min"] for c in c4] == [4]  # frp_v4_like's 4 of 5
        # no departure from frp_v4_like at its native five seeds
        base = frp_v4_like(
            seeds=(0, 1, 2, 3, 4), hero=hero, cap=600, no_go=50.0, study_id=rule["study_id"]
        )
        assert base == rule
    assert rs.primary_rule()["study_id"] != rs.secondary_rule()["study_id"]
    assert rs.PRIMARY_ID not in (rule_spec_m3c.PRIMARY_ID, rule_spec_m3c.SECONDARY_ID)


def test_phase_r_banks_are_fresh_and_disjoint():
    used = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds()) | set(gid.world_seeds(8))
    used |= set(dev_screen.screen_seeds(3, student_record_parity.DOMAIN, "worlds"))
    for r, n in USED.items():
        used |= set(ni_spec.distill_seeds(n, round_index=r))
    train, earlier_banks = set(), set()
    banks = {k: set() for k in rs.BANK_NS}
    for s in range(5):
        # M3 / M3-C / M4-A training (same namespace, keys seed0-4): at most
        # 20M / 64 / 5000 = 63 segments x 64 worlds.
        train |= set(dev_screen.screen_seeds(4100, M3_TRAIN, f"seed{s}"))
        earlier_banks |= set(dev_screen.screen_seeds(32, M3_PHASE_R, f"seed{s}"))
        for ns in rule_spec_m3c.BANK_NS.values():
            earlier_banks |= set(dev_screen.screen_seeds(32, ns, f"seed{s}"))
        earlier_banks |= set(
            dev_screen.screen_seeds(4, "redesign-m3c-phase-r-smoke/v1", f"seed{s}")
        )
    smoke = set()
    for s in rs.TRAINING_SEEDS:
        smoke |= set(dev_screen.screen_seeds(4, M4_SMOKE, f"seed{s}"))
        for study, ns in rs.BANK_NS.items():
            banks[study] |= set(dev_screen.screen_seeds(rs.N_WORLDS, ns, f"seed{s}"))
    nv, v8 = banks[rs.PRIMARY_ID], banks[rs.SECONDARY_ID]
    assert len(nv) == len(v8) == 160
    for bank in (nv, v8):
        assert not bank & used and not bank & train and not bank & earlier_banks
        assert not bank & smoke
    assert not nv & v8
