"""The pre-registered M2 non-inferiority rule (research/redesign_m2_20261008/ni_spec.py)."""

from __future__ import annotations

import pytest

from research.redesign_m2_20261008 import ni_spec


def test_seeds_are_pinned_and_disjoint():
    seeds = ni_spec.ni_seeds()
    assert len(seeds) == 48 and len(set(seeds)) == 48
    assert seeds[:3] == [2643297547, 2781215250, 3530758914]
    distill = set()
    for r in range(3):
        distill |= set(ni_spec.distill_seeds(3000, round_index=r))
    assert len(distill) == 9000 and not set(seeds) & distill


def test_rule_constants():
    assert ni_spec.MARGIN == -15.0 and ni_spec.CONFIDENCE == 0.90
    assert ni_spec.WORLDS_PER_MIX == 48 and ni_spec.MIXES == ("frozen", "scripted", "mixed")


def _arms(delta):
    seeds = ni_spec.ni_seeds()
    base = {m: {s: 100.0 + (i % 7) * 10 for i, s in enumerate(seeds)} for m in ni_spec.MIXES}
    stud = {
        m: {s: v + delta + ((i % 5) - 2) * 4 for i, (s, v) in enumerate(base[m].items())}
        for m in ni_spec.MIXES
    }
    return stud, base


def test_decide_pass_fail_incomplete():
    stud, base = _arms(0.0)
    out = ni_spec.decide(stud, base)
    assert out["verdict"] == "PASS" and out["pooled"]["n"] == 144
    stud, base = _arms(-20.0)
    assert ni_spec.decide(stud, base)["verdict"] == "FAIL"
    stud, base = _arms(0.0)
    del stud["mixed"][ni_spec.ni_seeds()[0]]
    assert ni_spec.decide(stud, base)["verdict"] == "INCOMPLETE"


def test_lower_bound_matches_t():
    out = ni_spec.lower_bound([1.0, 2.0, 3.0, 4.0])
    assert out["mean"] == 2.5
    assert out["lb"] == pytest.approx(2.5 - out["t"] * out["sd"] / 2.0)
    assert 1.6 < out["t"] < 1.7  # t_{0.90, 3} = 1.638


def _write_arms(out, sha, tamper=None):
    import json

    seeds = ni_spec.ni_seeds()
    for arm in ("student", "baseline"):
        for mix in ni_spec.MIXES:
            recs = []
            for s in seeds:
                r = {"seed": s, "mass_integral": 100.0, "mix_id": mix}
                r["evaluation_profile_digest"] = "d"
                if arm == "student":
                    r["ego2s_hero"] = {"sha256": sha, "veto": None}
                recs.append(r)
            payload = {
                "arm": arm,
                "mix": mix,
                "git": {"commit": "c", "dirty_paths": ""},
                "student_sha256": sha if arm == "student" else None,
                "records": recs,
            }
            if tamper:
                tamper(arm, mix, payload)
            (out / f"{arm}-{mix}.json").write_text(json.dumps(payload))


def test_ni_check_audit_blocks_bad_provenance(tmp_path):
    from research.redesign_m2_20261008 import ni_check

    intent = {"student": {"sha256": "abc"}}
    _write_arms(tmp_path, "abc")
    assert ni_check.audit(tmp_path, intent) == []

    def drop_student(arm, mix, p):
        if arm == "student" and mix == "mixed":
            for r in p["records"]:
                r.pop("ego2s_hero")

    _write_arms(tmp_path, "abc", drop_student)
    assert any("not played by the student" in x for x in ni_check.audit(tmp_path, intent))

    def lose_world(arm, mix, p):
        if arm == "baseline" and mix == "frozen":
            p["records"].pop()

    _write_arms(tmp_path, "abc", lose_world)
    assert any("seeds differ" in x for x in ni_check.audit(tmp_path, intent))
    (tmp_path / "student-scripted.json").unlink()
    assert any("missing" in x for x in ni_check.audit(tmp_path, intent))
