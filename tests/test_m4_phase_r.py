"""M4 Phase R harness: a rebound private copy of the pinned M3-C harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.apex_safety_20260926 import dev_screen
from research.redesign_m4_20261013 import phase_r_m4 as pr
from research.redesign_m4_20261013 import rule_spec_m4 as rs
from research.redesign_scope_20261007 import grid_h5000_identity as gid


def test_base_copy_is_rebound_and_m3c_is_untouched():
    from research.redesign_m3c_20261008 import phase_r_m3c as m3c

    assert pr.sha256_file(pr.BASE_PATH) == pr.BASE_SHA256
    b = pr.base
    assert b is not m3c and b.rs is rs and b.SCHEMA == "redesign-m4-phase-r-record/v1"
    assert b.SMOKE_NS == "redesign-m4-phase-r-smoke/v1" and b.PREREG == pr.PREREG
    assert Path(b.__file__).name == "phase_r_m4.py"
    assert b.STUDIES["nv"]["id"] == "redesign-m4-phase-r-nv"
    # the shared M3-C module keeps its own study
    assert m3c.rs.PRIMARY_ID == "redesign-m3c-phase-r-nv"
    assert m3c.SCHEMA == "redesign-m3c-phase-r-record/v1"
    assert Path(m3c.__file__).name == "phase_r_m3c.py"
    assert pr.heroes("nv") == [
        {"hero": rs.HERO_NV, "seeds": [0, 1, 2], "tier": 1},
        {"hero": "incumbent", "seeds": [0, 1, 2], "tier": 1},
        {"hero": "champion", "seeds": [0], "tier": 1},
        {"hero": "M2b-student", "seeds": [0], "tier": 2},
    ]
    assert pr.base.make_banks(4, pr.SMOKE_NS).keys() == {0, 1, 2}


def test_hero_calls(tmp_path):
    ck = tmp_path / "c.pth"
    ck.write_bytes(b"x")
    sha = pr.sha256_file(ck)
    lookup = {gid.FRP3_S12[1]: ("checkpoint", "frp3"), dev_screen.CHAMPION[1]: ("checkpoint", "a5")}
    ctx = {"lookup": lookup}
    for key, no_veto in (("nv", True), ("v8", False)):
        hero = pr.STUDIES[key]["hero"]
        study = {
            "study_id": pr.STUDIES[key]["id"],
            "checkpoints": {f"{hero}|s{s}": {"path": str(ck), "sha256": sha} for s in range(3)},
        }
        spec, extra = pr.hero_call(hero, 2, ctx, study)
        assert spec == ("checkpoint", "frp3")
        assert extra["hero_ego2s"] == str(ck) and extra["veto"] is (not no_veto)
        assert extra.get("ego2s_mask") == ("advisory" if no_veto else None)
        assert pr.hero_call("incumbent", 2, ctx, study)[1] == {"veto": True}
        assert pr.hero_call("champion", 0, ctx, study)[1] == {"veto": True}
        assert pr.control_arm(key) == ("M2b-student" if no_veto else "M2b-student+v8")
        assert pr.study_key(dict(study, study_id=study["study_id"] + "-SMOKE")) == key
    with pytest.raises(SystemExit):
        pr.study_key({"study_id": "redesign-m3c-phase-r-nv"})  # an M3-C plan is not M4's


def _selection(tmp_path, arm="C", eligible=(0, 1, 2)):
    runs = tmp_path / "runs"
    paths = {}
    for s in range(3):
        d = runs / f"{arm}_s{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "final.pth").write_bytes(f"{arm}{s}".encode())
        paths[str(s)] = str(d / "final.pth")
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"arm": arm, "paths": paths}))
    good = {
        "checkpoint": "final",
        "primary": {"arm": arm, "enter": True},
        "secondary": {"arm": None, "enter": False},
        "target_transitions": 20_000_000,
        "scores": {
            arm: {
                "eligible": list(eligible),
                "final_sha256": {
                    str(s): pr.sha256_file(runs / f"{arm}_s{s}" / "final.pth") for s in range(3)
                },
            }
        },
    }
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(good))
    return sel, cand, good


def test_selection_check_with_three_seeds(tmp_path):
    sel, cand, good = _selection(tmp_path, "B8")
    assert pr.check_selection("nv", sel, cand)["gate"]["arm"] == "B8"
    with pytest.raises(SystemExit):
        pr.check_selection("v8", sel, cand)  # the secondary did not enter
    sel.write_text(
        json.dumps(dict(good, scores={"B8": dict(good["scores"]["B8"], eligible=[0, 1])}))
    )
    with pytest.raises(SystemExit):
        pr.check_selection("nv", sel, cand)
    sel.write_text(json.dumps(dict(good, target_transitions=5_000_000)))
    with pytest.raises(SystemExit):
        pr.check_selection("nv", sel, cand)
    data = json.loads(cand.read_text())
    data["paths"]["3"] = data["paths"]["2"]  # M3-C's five seeds are not M4's three
    cand.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        pr.load_candidates(cand)


def test_ratification_line(tmp_path):
    p = tmp_path / "P.md"
    p.write_text("**Status: DRAFT -- NOT RATIFIED.** the owner adds **RATIFIED <date> later\n")
    assert not pr.ratified(p)
    p.write_text("x\n**RATIFIED 2026-10-14 by the owner**\n")
    assert pr.ratified(p)
    assert not pr.ratified(pr.PREREG)  # the committed pre-registration is a DRAFT


def test_obs_spec_binding_and_bv8_p2_rule(tmp_path):
    import torch

    paths = {}
    for s, spec in enumerate(("ego2s-bv8", "ego2s-bv8", "ego2s-bv8")):
        paths[s] = tmp_path / f"s{s}.pth"
        torch.save({"obs_spec": spec}, paths[s])
    assert pr.candidate_obs_spec(paths) == "ego2s-bv8"
    with pytest.raises(SystemExit, match="no live path"):
        pr.p2_for("nv", paths, "ego2s-bv8", bv8_simd_only=False)
    rec = pr.p2_for("v8", paths, "ego2s-bv8", bv8_simd_only=True)
    assert rec["status"] == "not_applicable"
    with pytest.raises(SystemExit):
        pr.p2_for("nv", paths, "ego2s-b", bv8_simd_only=True)
    torch.save({"obs_spec": "ego2s-b"}, paths[1])
    with pytest.raises(SystemExit, match="differ"):
        pr.candidate_obs_spec(paths)
    for s in paths:
        torch.save({"obs_spec": "ego2s-draft"}, paths[s])
    with pytest.raises(SystemExit, match="not one of"):
        pr.candidate_obs_spec(paths)


def test_look_units_run_this_harness(tmp_path, monkeypatch):
    """``cmd_look``'s unit subprocesses run ``phase_r_m4.py unit`` (not M3-C's file)."""
    seen = []

    class _Done:
        pid, returncode = 0, 0

        def poll(self):
            return 0

    def fake_popen(argv, **kwargs):
        seen.append(argv)
        return _Done()

    monkeypatch.setattr(pr.base.subprocess, "Popen", fake_popen)
    (tmp_path / "shards" / "look-0").mkdir(parents=True)
    failed = pr.base.run_units(tmp_path, 0, [{"unit_id": "L0|x"}], 1)
    assert failed == [] and len(seen) == 1
    argv = seen[0]
    assert argv[1].endswith("research/redesign_m3_20261008/slotrun.py")
    assert argv[4].endswith("research/redesign_m4_20261013/phase_r_m4.py") and argv[5] == "unit"


def test_smoke_plan_binds_m4(tmp_path):
    """``plan --smoke`` end to end on real checkpoints (skipped without the artifacts)."""
    ck = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4a-gpu-20261012/"
        "attempt00/out/C_s0/ckpt_5000k.pth"
    )
    if not ck.exists() or not pr.base.M2B.exists():
        pytest.skip("artifacts not on this machine")
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"arm": "C", "paths": {str(s): str(ck) for s in range(3)}}))
    root = tmp_path / "root"
    assert (
        pr.main(
            ["plan", "--study", "nv", "--root", str(root), "--candidates", str(cand), "--smoke"]
        )
        == 0
    )
    plan = json.loads((root / "plan.json").read_text())
    st = plan["study"]
    assert (
        st["study_id"] == "redesign-m4-phase-r-nv-SMOKE" and st["candidate_obs_spec"] == "ego2s-b"
    )
    assert st["namespace"] == pr.SMOKE_NS and sorted(plan["banks"]) == ["0", "1", "2"]
    assert st["base_harness"]["sha256"] == pr.BASE_SHA256
    assert {h["hero"] for h in st["heroes"]} == {rs.HERO_NV, "incumbent", "champion", "M2b-student"}
    units = pr.base.units_for(root, 0)
    assert {u["seed"] for u in units} == {0, 1, 2}
