"""M3-C (DRAFT) Phase R rule specs, plans and world namespaces.

The pins are draft pins: ratification re-pins them (or confirms them) in the same commit.
"""

from __future__ import annotations

from research.apex_safety_20260926 import dev_screen
from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b
from research.redesign_m3_20261008 import student_record_parity
from research.redesign_m3c_20261008 import rule_spec_m3c as rs
from research.redesign_scope_20261007 import grid_h5000_identity as gid
from src.evaluation.sequential_phase_r import _sha

PRIMARY_RULE_SHA256 = "5103b4133f819e6dea097487ed5f0192aa2cc9ce3ae75f40d595afa63d84b6c0"
SECONDARY_RULE_SHA256 = "1f50018e803abd501ed0783890bbed9c74f1e850eb0387fbbf0b4eef63b97488"
PRIMARY_PLAN_SHA256 = "70e4b4d78c9b77b411f36479b28a3f183a8fc6cc438c0d1c8368fc65e92350f2"
SECONDARY_PLAN_SHA256 = "4a71565d16693c0ae599ace7591770226f2636594f7e394c1be1d4ae12b8a211"
M3_TRAIN = "redesign-m3-train/v1"
M3_PHASE_R = "redesign-m3-phase-r/v1"
# Worlds used per distillation / probe round index (M3's list; M3-C probes round 52 up to 32).
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
        assert rule["cells"]["guard"]["seeds"] == [0]
        names = {c["name"] for c in rule["go_clauses"]}
        assert {"c5_mi10", "c3_frozen_survival", "c2_scripted_survival"} <= names
    assert rs.primary_rule()["study_id"] != rs.secondary_rule()["study_id"]


def test_phase_r_banks_are_fresh_and_disjoint():
    used = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds()) | set(gid.world_seeds(8))
    used |= set(dev_screen.screen_seeds(3, student_record_parity.DOMAIN, "worlds"))
    for r, n in USED.items():
        used |= set(ni_spec.distill_seeds(n, round_index=r))
    train, m3_bank = set(), set()
    banks = {k: set() for k in rs.BANK_NS}
    for s in range(5):
        # M3 and M3-C training (same namespace, keys seed0-4): at most 20M / 64 / 5000 = 63
        # segments x 64 worlds.
        train |= set(dev_screen.screen_seeds(4100, M3_TRAIN, f"seed{s}"))
        m3_bank |= set(dev_screen.screen_seeds(32, M3_PHASE_R, f"seed{s}"))
        for study, ns in rs.BANK_NS.items():
            banks[study] |= set(dev_screen.screen_seeds(rs.N_WORLDS, ns, f"seed{s}"))
    nv, v8 = banks[rs.PRIMARY_ID], banks[rs.SECONDARY_ID]
    assert len(nv) == len(v8) == 160
    for bank in (nv, v8):
        assert not bank & used and not bank & train and not bank & m3_bank
    assert not nv & v8


def test_phase_r_harness_hero_calls(tmp_path):
    """The nv study plays the candidate and the control arm WITHOUT a veto (advisory mask);
    the incumbent / champion always carry v8; the v8 study vetoes everyone."""
    from types import SimpleNamespace

    from research.redesign_m3c_20261008 import phase_r_m3c as pr

    ck = tmp_path / "c.pth"
    ck.write_bytes(b"x")
    sha = pr.sha256_file(ck)
    ctx = {"lookup": SimpleNamespace(__getitem__=None)}
    lookup = {gid.FRP3_S12[1]: ("checkpoint", "frp3"), dev_screen.CHAMPION[1]: ("checkpoint", "a5")}
    ctx = {"lookup": lookup}
    for key, no_veto in (("nv", True), ("v8", False)):
        hero = pr.STUDIES[key]["hero"]
        study = {
            "study_id": pr.STUDIES[key]["id"],
            "checkpoints": {f"{hero}|s{s}": {"path": str(ck), "sha256": sha} for s in range(5)},
        }
        spec, extra = pr.hero_call(hero, 2, ctx, study)
        assert extra["hero_ego2s"] == str(ck) and extra["veto"] is (not no_veto)
        assert extra.get("ego2s_mask") == ("advisory" if no_veto else None)
        assert pr.hero_call("incumbent", 2, ctx, study)[1] == {"veto": True}
        assert pr.hero_call("champion", 0, ctx, study)[1] == {"veto": True}
        assert pr.control_arm(key) == ("M2b-student" if no_veto else "M2b-student+v8")
        assert [h["hero"] for h in pr.heroes(key)][0] == hero
        smoke = dict(study, study_id=study["study_id"] + "-SMOKE")
        assert pr.study_key(smoke) == key


def test_phase_r_refuses_unratified_or_unselected(tmp_path):
    import json

    import pytest

    from research.redesign_m3c_20261008 import phase_r_m3c as pr

    pre = tmp_path / "P.md"
    pre.write_text("Status: NOT RATIFIED\nwe will write **RATIFIED later\n")
    assert not pr.ratified(pre)
    pre.write_text("x\n**RATIFIED 2026-10-12 by the owner** at commit abc\n")
    assert pr.ratified(pre) and not pr.ratified(tmp_path / "missing.md")
    runs = tmp_path / "runs"
    paths = {}
    for s in range(5):
        d = runs / f"B_s{s}"
        d.mkdir(parents=True)
        (d / "final.pth").write_bytes(b"x")
        paths[str(s)] = str(d / "final.pth")
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"arm": "B", "paths": paths}))
    sel = tmp_path / "sel.json"
    good = {
        "primary": {"arm": "B", "enter": True},
        "secondary": {},
        "target_transitions": pr.TRAINING_TRANSITIONS,
        "scores": {"B": {"eligible": [0, 1, 2, 3, 4]}},
    }
    sel.write_text(json.dumps(good))
    assert pr.check_selection("nv", sel, cand)["gate"]["arm"] == "B"
    for bad in (
        dict(good, target_transitions=None),  # made without the eligibility check
        dict(good, scores={"B": {"eligible": [0, 1, 2, 4]}}),  # a seed missing
    ):
        sel.write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            pr.check_selection("nv", sel, cand)
    sel.write_text(json.dumps(good))
    with pytest.raises(SystemExit):
        pr.check_selection("v8", sel, cand)  # the secondary did not enter
    with pytest.raises(SystemExit):
        pr.check_selection("nv", None, cand)
    sel.write_text(json.dumps(dict(good, primary={"arm": "D", "enter": True})))
    with pytest.raises(SystemExit):
        pr.check_selection("nv", sel, cand)  # another arm was selected
    sel.write_text(json.dumps(good))
    paths["2"] = str(runs / "B_s2" / "ckpt_5000k.pth")
    cand.write_text(json.dumps({"arm": "B", "paths": paths}))
    with pytest.raises(SystemExit):
        pr.check_selection("nv", sel, cand)  # not the final checkpoint
