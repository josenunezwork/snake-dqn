"""M4 selection rule and Phase R entry gates on synthetic probe / run files."""

from __future__ import annotations

import hashlib
import json

from research.redesign_m4_20261013 import m4_select as sel

N_WORLDS = 32


def _sha(seed_bytes: bytes) -> str:
    return hashlib.sha256(seed_bytes).hexdigest()


def _probe(path, base, sha, veto="none", deaths=0, mixes=sel.MIXES, worlds=N_WORLDS, **over):
    with path.open("w") as fh:
        for mix in mixes:
            row = {
                "arm": "student",
                "veto": veto,
                "round_index": 52,
                "ego2s_forward": "rowwise",
                "ego2s_mask": "advisory" if veto == "none" else None,
                "student_sha256": sha,
                "mix": mix,
                "mass": {str(w): base + w % 4 for w in range(worlds)},
                "deaths": {str(w): (1 if w < deaths else 0) for w in range(worlds)},
            }
            row.update(over)
            fh.write(json.dumps(row) + "\n")


def _incumbent(path, base=400, deaths=0, worlds=N_WORLDS, round_index=52):
    with path.open("w") as fh:
        for mix in sel.MIXES:
            row = {
                "arm": "baseline",
                "veto": "v8",
                "round_index": round_index,
                "student": None,
                "mix": mix,
                "mass": {str(w): base + w % 4 for w in range(worlds)},
                "deaths": {str(w): (1 if w < deaths else 0) for w in range(worlds)},
            }
            fh.write(json.dumps(row) + "\n")


def _layout(tmp_path, levels, *, v8=None, halt=None, split=("C",)):
    """Arms in ``split`` live in pod-1 dirs, the rest in pod-2 dirs (as M4-A does)."""
    dirs = {}
    for pod in ("p1", "p2"):
        dirs[pod] = (tmp_path / pod / "probes", tmp_path / pod / "out")
        for d in dirs[pod]:
            d.mkdir(parents=True)
    for arm, nv in levels.items():
        probes, out = dirs["p1" if arm in split else "p2"]
        for s in sel.SEEDS:
            run = out / f"{arm}_s{s}"
            run.mkdir()
            body = f"{arm}{s}".encode()
            (run / "final.pth").write_bytes(body)
            h = "action collapse" if halt and (arm, s) in halt else None
            (run / "summary.json").write_text(json.dumps({"halt": h, "transitions": 100}))
            _probe(probes / f"{arm}_s{s}__final__none.jsonl", nv + s, _sha(body))
            if v8 is not None:
                _probe(probes / f"{arm}_s{s}__final__v8.jsonl", v8[arm] + s, _sha(body), "v8")
    inc = tmp_path / "incumbent__dev__v8.jsonl"
    _incumbent(inc)
    return [dirs[p][0] for p in dirs], [dirs[p][1] for p in dirs], inc


def test_selection_across_two_pods(tmp_path):
    levels = {"C": 380, "D": 360, "S": 340, "B8": 390}
    probes, runs, inc = _layout(tmp_path, levels)
    out = sel.select(probes, list(levels), runs, 100, inc)
    assert out["world_check"]["ok"] and out["worlds_per_mix"] == {m: 32 for m in sel.MIXES}
    assert out["seeds_used"] == [0, 1, 2] and out["incumbent_v8"] == 401.5
    p = out["primary"]
    assert p["arm"] == "B8" and p["score"] == 392.5 and p["enter"] and p["all_seeds_eligible"]
    assert out["scores"]["C"]["run_dirs"][0].endswith("p1/out/C_s0")
    assert out["scores"]["D"]["run_dirs"][0].endswith("p2/out/D_s0")
    assert out["scores"]["B8"]["final_sha256"][1] == _sha(b"B81")
    # no v8 probes anywhere: the secondary is not evaluated
    assert out["secondary"] == {
        "arm": None,
        "enter": False,
        "reason": "no complete v8 final probes",
    }
    c = out["contrasts"]
    assert c["D-C"]["none"]["mean"] == -20.0 and c["B8-D"]["none"]["n"] == 3
    assert "v8" not in c["D-C"]


def test_secondary_with_v8_probes_and_the_score_bar(tmp_path):
    levels = {"C": 300, "D": 340}
    probes, runs, inc = _layout(tmp_path, levels, v8={"C": 440, "D": 420})
    out = sel.select(probes, ["C", "D"], runs, 100, inc)
    p, s = out["primary"], out["secondary"]
    assert p["arm"] == "D" and p["score"] == 342.5 and not p["score_ok"] and not p["enter"]
    assert s["arm"] == "C" and s["delta"] == 41.0 and s["enter"]


def test_halted_seed_drops_it_for_every_arm_and_blocks_entry(tmp_path):
    levels = {"C": 450, "D": 380}
    probes, runs, inc = _layout(tmp_path, levels, halt={("C", 2)})
    out = sel.select(probes, ["C", "D"], runs, 100, inc)
    assert out["scores"]["C"]["seeds"][2].startswith("halt")
    assert out["seeds_used"] == [0, 1]
    p = out["primary"]
    # C is best on the common seeds but lacks seed 2: it cannot enter Phase R
    assert p["arm"] == "C" and p["score_ok"] and not p["all_seeds_eligible"] and not p["enter"]
    two = sel.select(probes, ["C", "D"], runs, 1000, inc)  # nothing reached the target
    assert two["seeds_used"] == [] and two["primary"]["arm"] is None


def test_world_set_must_match_the_incumbent(tmp_path):
    probes, runs, inc = _layout(tmp_path, {"C": 450})
    _incumbent(inc, worlds=16)  # the 16-world mid-probe reference: wrong set
    out = sel.select(probes, ["C"], runs, 100, inc)
    assert not out["world_check"]["ok"] and len(out["world_check"]["files"]) == 9
    assert not out["primary"]["enter"] and "--incumbent" in out["primary"]["reason"]
    _incumbent(inc, round_index=51)
    out = sel.select(probes, ["C"], runs, 100, inc)
    assert "round 52" in out["primary"]["reason"]
    missing = sel.select(probes, ["C"], runs, 100, tmp_path / "nope.jsonl")
    assert not missing["primary"]["enter"] and missing["incumbent_file"]["sha256"] is None


def test_probe_validity_checks(tmp_path):
    probes, runs, inc = _layout(tmp_path, {"C": 450, "D": 420})
    p2 = probes[1]
    good = sel.select(probes, ["C", "D"], runs, 100, inc)
    assert good["seeds_used"] == [0, 1, 2]
    cases = (
        ({"student_sha256": "0" * 64}, "not made with this run's final.pth"),
        ({"ego2s_mask": "resolved"}, "hero modes"),
        ({"round_index": 51}, "round 51"),
    )
    for over, why in cases:
        f = p2 / "D_s1__final__none.jsonl"
        _probe(f, 421, _sha(b"D1"), **over)
        out = sel.select(probes, ["C", "D"], runs, 100, inc)
        assert why in out["scores"]["D"]["seeds"][1] and out["seeds_used"] == [0, 2]
    _probe(p2 / "D_s1__final__none.jsonl", 421, _sha(b"D1"), mixes=("frozen", "scripted"))
    out = sel.select(probes, ["C", "D"], runs, 100, inc)
    assert "incomplete" in out["scores"]["D"]["seeds"][1]


def test_ambiguous_locations_make_the_seed_ineligible(tmp_path):
    probes, runs, inc = _layout(tmp_path, {"C": 450, "D": 420})
    # a stale copy of D_s0 in pod 1's out dir (e.g. pod 1's slow D rerun reaching final)
    stale = runs[0] / "D_s0"
    stale.mkdir()
    (stale / "summary.json").write_text(json.dumps({"halt": None, "transitions": 100}))
    out = sel.select(probes, ["C", "D"], runs, 100, inc)
    assert out["scores"]["D"]["seeds"][0].startswith("run ambiguous")
    assert out["seeds_used"] == [1, 2]
    (stale / "summary.json").unlink()
    stale.rmdir()
    (probes[0] / "D_s2__final__none.jsonl").write_text(
        (probes[1] / "D_s2__final__none.jsonl").read_text()
    )
    out = sel.select(probes, ["C", "D"], runs, 100, inc)
    assert out["scores"]["D"]["seeds"][2].startswith("probe ambiguous")


def test_real_incumbent_reference_is_round52_32_worlds():
    """The default reference (M3-C's) has 32 round-52 worlds per mix (if it is on this Mac)."""
    import pytest

    if not sel.INCUMBENT.exists():
        pytest.skip("artifacts not on this machine")
    sets = sel.world_sets(sel.INCUMBENT)
    assert set(sets) == set(sel.MIXES)
    assert all(r == 52 and len(ws) == 32 for r, ws in sets.values())
