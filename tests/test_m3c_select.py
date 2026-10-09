"""M3-C selection rule and Phase R entry gates on synthetic probe files."""

from __future__ import annotations

import json

from research.redesign_m3c_20261008 import m3c_select as sel

N_WORLDS = 40


def _write(path, base, deaths=0, mixes=sel.MIXES):
    """deaths: number of worlds (the first ones) where the hero died."""
    with path.open("w") as fh:
        for mix in mixes:
            row = {
                "mix": mix,
                "mass": {str(w): base + w % 4 for w in range(N_WORLDS)},
                "deaths": {str(w): (1 if w < deaths else 0) for w in range(N_WORLDS)},
            }
            fh.write(json.dumps(row) + "\n")


def _runs(tmp_path, arms, bad=None, transitions=100):
    for arm in arms:
        for s in sel.SEEDS:
            d = tmp_path / "runs" / f"{arm}_s{s}"
            d.mkdir(parents=True)
            h = "action collapse" if bad and (arm, s) in bad else None
            (d / "summary.json").write_text(json.dumps({"halt": h, "transitions": transitions}))
            (d / "final.pth").write_bytes(b"x")


def _both(tmp_path, arm, s, nv, v8, deaths_nv=0, deaths_v8=0):
    _write(tmp_path / f"{arm}_s{s}__final__none.jsonl", nv, deaths_nv)
    _write(tmp_path / f"{arm}_s{s}__final__v8.jsonl", v8, deaths_v8)


def test_selection_gates_and_contrasts(tmp_path):
    levels = {"A": (300, 420), "B": (330, 410), "D": (310, 450), "L": (200, 300)}
    for arm, (nv, v8) in levels.items():
        for s in sel.SEEDS:
            _both(tmp_path, arm, s, nv + s, v8 + s, deaths_nv=30, deaths_v8=6)
    _write(tmp_path / "incumbent__dev__v8.jsonl", 400, deaths=6)
    _runs(tmp_path, levels)
    out = sel.select(tmp_path, ["A", "B", "D", "L"], tmp_path / "runs", 100)
    assert out["worlds_per_mix"] == {m: N_WORLDS for m in sel.MIXES}
    assert out["incumbent_v8"] == 401.5 and out["seeds_used"] == [0, 1, 2, 3, 4]
    p, s = out["primary"], out["secondary"]
    # B is best without the veto (333.5) but 333.5 < 351.5 and it dies in 30/40 worlds
    assert p["arm"] == "B" and p["score"] == 333.5 and not p["score_ok"]
    assert not p["survival_ok"] and not p["enter"]
    assert s["arm"] == "D" and s["delta"] == 52.0 and s["survival_ok"] and s["enter"]
    c = out["contrasts"]
    assert c["B-A"]["none"]["mean"] == 30.0 and c["D-B"]["v8"]["mean"] == 40.0
    assert c["B-L"]["none"]["n"] == 5 and c["B-A"]["none"]["se"] == 0.0


def test_survival_gate_is_noise_aware():
    inc = {m: {str(w): (0.0, float(w % 10 == 0)) for w in range(40)} for m in sel.MIXES}
    same = [inc] * 5  # exactly the incumbent's survival: must pass
    chk = sel.survival_check(same, inc, {m: [str(w) for w in range(40)] for m in sel.MIXES})
    assert all(v["ok"] and v["mean"] == 0.0 for v in chk.values())
    worse = [{m: {str(w): (0.0, float(w % 3 == 0)) for w in range(40)} for m in sel.MIXES}] * 5
    chk = sel.survival_check(worse, inc, {m: [str(w) for w in range(40)] for m in sel.MIXES})
    assert not any(v["ok"] for v in chk.values())


def test_short_queue_uses_common_seeds_or_selects_nothing(tmp_path):
    for arm, nv in (("B", 500), ("C", 360)):
        for s in sel.SEEDS:
            _both(tmp_path, arm, s, nv, nv)
    _write(tmp_path / "incumbent__dev__v8.jsonl", 400)
    _runs(tmp_path, ["B", "C"], bad={("B", 3)})
    out = sel.select(tmp_path, ["B", "C"], tmp_path / "runs", 100)
    assert out["scores"]["B"]["seeds"][3].startswith("halt")
    assert out["seeds_used"] == [0, 1, 2, 4]  # seed 3 dropped for every arm
    assert out["primary"]["arm"] == "B" and out["primary"]["enter"]
    assert out["contrasts"] == {}  # B-C is not a pre-registered contrast
    (tmp_path / "runs" / "C_s1" / "summary.json").write_text(
        json.dumps({"halt": None, "transitions": 50})
    )
    two = sel.select(tmp_path, ["B", "C"], tmp_path / "runs", 100)
    assert two["seeds_used"] == [] and two["primary"]["arm"] is None


def test_partial_probe_file_does_not_empty_a_mix(tmp_path):
    for s in sel.SEEDS:
        _both(tmp_path, "B", s, 380, 380)
        _write(tmp_path / f"C_s{s}__final__none.jsonl", 390, mixes=("frozen", "scripted"))
        _write(tmp_path / f"C_s{s}__final__v8.jsonl", 390)
    _write(tmp_path / "incumbent__dev__v8.jsonl", 400)
    _runs(tmp_path, ["B", "C"])
    out = sel.select(tmp_path, ["B"], tmp_path / "runs", 100)
    assert out["worlds_per_mix"]["mixed"] == N_WORLDS and out["primary"]["arm"] == "B"
    both = sel.select(tmp_path, ["B", "C"], tmp_path / "runs", 100)
    assert both["worlds_per_mix"]["mixed"] == N_WORLDS
    assert both["seeds_used"] == []  # C never has "mixed" without the veto: nothing selected
    missing = sel.select(tmp_path / "nowhere", ["B"], tmp_path / "runs", 100)
    assert not missing["primary"]["enter"] and "incumbent" in missing["primary"]["reason"]


def test_probe_of_another_checkpoint_makes_the_seed_ineligible(tmp_path):
    for arm in ("B", "C"):
        for s in sel.SEEDS:
            _both(tmp_path, arm, s, 380, 380)
    _write(tmp_path / "incumbent__dev__v8.jsonl", 400)
    _runs(tmp_path, ["B", "C"])
    f = tmp_path / "B_s2__final__none.jsonl"
    rows = [dict(json.loads(x), student_sha256="0" * 64) for x in f.read_text().splitlines()]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = sel.select(tmp_path, ["B", "C"], tmp_path / "runs", 100)
    assert out["scores"]["B"]["seeds"][2] == "probe none made with another checkpoint"
    assert out["seeds_used"] == [0, 1, 3, 4]
