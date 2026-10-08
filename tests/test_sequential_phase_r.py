"""Tests for the sequential Phase R module, hooks, receipts, audit and OC simulator."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research.sequential_phase_r import audit as audit_mod
from research.sequential_phase_r import example_rules, hooks, receipts
from research.sequential_phase_r import simulate as sim
from src.evaluation.sequential_phase_r import (
    METHOD,
    look_sizes_for,
    make_plan,
    n_fixed,
    required_inputs,
    sequential_phase_r_decision,
    spending,
    validate_rule,
)

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "research" / "sequential_phase_r"
MIXES = ("frozen", "mixed", "scripted")


# ----------------------------------------------------------------------------- plan
def test_look_sizes_and_fractions():
    assert look_sizes_for(32) == (11, 22, 32)
    assert look_sizes_for(33) == (11, 22, 33)
    assert look_sizes_for(6) == (2, 4, 6)
    with pytest.raises(ValueError):
        look_sizes_for(3)  # first look would be 1 world
    with pytest.raises(ValueError):
        look_sizes_for(32, (0.5, 0.4, 1.0))


def test_spending_functions():
    assert spending(1.0, 0.1, "obf") == pytest.approx(0.1)
    assert spending(1.0, 0.1, "pocock") == pytest.approx(0.1)
    assert spending(1.0, 0.1, "hsd:1") == pytest.approx(0.1)
    assert spending(0.5, 0.1, "pocock") == pytest.approx(0.1 * math.log(1 + (math.e - 1) / 2))
    assert spending(1 / 3, 0.1, "obf") < spending(1 / 3, 0.1, "pocock")
    with pytest.raises(ValueError):
        spending(0.5, 0.1, "linear")


def test_plan_nominal_levels_and_spent_alpha():
    plan = make_plan(32, example_rules.frp_v4_like())
    assert plan.look_sizes == (11, 22, 32)
    assert plan.go_spending == "obf" and plan.kill_spending == "pocock"
    assert plan.go_alpha_spent[-1] == pytest.approx(0.10, abs=2e-4)
    assert plan.kill_alpha_spent[-1] == pytest.approx(0.10, abs=2e-4)
    assert plan.go_nominal_p == pytest.approx((0.0050, 0.0457, 0.0860), abs=2e-4)
    assert plan.kill_nominal_p == pytest.approx((0.0464, 0.0508, 0.0528), abs=2e-4)
    assert plan.as_dict()["method"] == METHOD
    assert plan.sha256() == make_plan(32, example_rules.frp_v4_like()).sha256()
    with pytest.raises(ValueError):
        make_plan(32, example_rules.frp_v4_like(), go_spending="pocock")


def test_rule_validation_rejects_bad_specs():
    rule = example_rules.frp_v4_like()
    validate_rule(rule)
    bad = copy.deepcopy(rule)
    bad["efficacy"]["stat"] = {
        "kind": "mix_mean",
        "cell": "primary",
        "metric": "mi5",
        "mix": "frozen",
    }
    with pytest.raises(ValueError, match="Hartung-Knapp"):
        validate_rule(bad)
    bad = copy.deepcopy(rule)
    bad["final_outcomes"][0]["all_of"].append("nope")
    with pytest.raises(ValueError, match="unknown clauses"):
        validate_rule(bad)
    bad = copy.deepcopy(rule)
    bad["statuses"]["kill"] = "GO_R"
    with pytest.raises(ValueError, match="distinct"):
        validate_rule(bad)
    bad = copy.deepcopy(rule)
    bad["go_clauses"][0]["stat"]["metric"] = "mass"
    with pytest.raises(ValueError, match="metric"):
        validate_rule(bad)
    assert required_inputs(validate_rule(rule)) == {
        "guard": ["mi5", "surv5"],
        "primary": ["mi10", "mi5", "surv5"],
    }


def test_n_fixed_matches_linear_search():
    from src.scripts.eval_stats import student_t_isf

    def linear(sd, mde):
        n = 2
        while n < ((student_t_isf(0.05 / 3, n - 1) + student_t_isf(0.10, n - 1)) * sd / mde) ** 2:
            n += 1
        return n

    for sd, mde in ((311.0, 65.0), (290.0, 44.0), (200.0, 120.0), (300.0, 300.0)):
        assert n_fixed(sd, mde) == linear(sd, mde)
    assert n_fixed(300.0, 0.0) is None and n_fixed(300.0, -3.0) is None


# ----------------------------------------------------------------------------- decisions
def _deltas(
    rule, n, effect, sd=200.0, surv=0.0, seed=0, control_effect=0.0, guard_gap=75.0, surv_sd=0.3
):
    rng = np.random.default_rng(seed)
    out = {}
    for cell, spec in rule["cells"].items():
        eff = {"primary": effect, "control": control_effect, "guard": effect + guard_gap}[cell]
        out[cell] = {}
        for metric in ("mi5", "surv5", "mi10"):
            scale = {"mi5": sd, "surv5": surv_sd, "mi10": 2 * sd}[metric]
            shift = {"mi5": eff, "surv5": surv, "mi10": 1.5 * eff}[metric]
            out[cell][metric] = {
                s: {m: (rng.standard_normal(n) * scale + shift).tolist() for m in rule["mixes"]}
                for s in spec["seeds"]
            }
    return out


def _cut(deltas, n):
    return {
        c: {
            m: {s: {x: v[:n] for x, v in bm.items()} for s, bm in cm.items()} for m, cm in d.items()
        }
        for c, d in deltas.items()
    }


FLAGS = {"prefix_controls": True}


def _run(plan, deltas):
    for k, n in enumerate(plan.look_sizes):
        d = sequential_phase_r_decision(plan, k, _cut(deltas, n), FLAGS)
        if d["action"] != "CONTINUE":
            return d
    return d


def test_large_effect_stops_early_for_go():
    plan = make_plan(32, example_rules.frp_v4_like())
    d = _run(plan, _deltas(plan.rule, 32, effect=400.0, sd=60.0, surv=0.2))
    assert d["status"] == "GO_R" and d["action"] == "STOP" and d["look"] == 0 and d["valid"]
    assert d["candidate_selection"]["seed"] in plan.rule["seeds"]


def test_large_harm_stops_early_for_kill():
    plan = make_plan(32, example_rules.frp_v4_like())
    d = _run(plan, _deltas(plan.rule, 32, effect=-300.0, sd=60.0))
    assert d["status"] == "KILL_V4" and d["look"] == 0
    assert d["kill"]["parts"][0]["upper_bound"] < 20.0


def test_null_continues_and_ends_at_final_with_final_rule():
    plan = make_plan(32, example_rules.frp_v4_like())
    deltas = _deltas(plan.rule, 32, effect=0.0, sd=200.0, seed=3)
    d0 = sequential_phase_r_decision(plan, 0, _cut(deltas, 11), FLAGS)
    assert d0["status"] == "CONTINUE" and d0["action"] == "CONTINUE"
    d = sequential_phase_r_decision(plan, 2, deltas, FLAGS)
    assert d["final_look"] and d["action"] == "STOP"
    assert d["status"] in ("PARTIAL", "KILL_V4", "GO_R_UNGATEABLE")
    assert [h["look"] for h in d["history"]] == [0, 1, 2]


def test_replay_marks_looks_after_a_stop_invalid():
    plan = make_plan(32, example_rules.frp_v4_like())
    deltas = _deltas(plan.rule, 32, effect=-300.0, sd=60.0)
    d = sequential_phase_r_decision(plan, 1, _cut(deltas, 22), FLAGS)
    assert d["valid"] is False and "already ended" in d["invalid_reason"]


def test_data_beyond_the_look_is_invalid():
    plan = make_plan(32, example_rules.frp_v4_like())
    deltas = _deltas(plan.rule, 32, effect=0.0)
    d = sequential_phase_r_decision(plan, 0, deltas, FLAGS)  # 32 worlds at look 0
    assert d["valid"] is False and "prefix integrity" in d["invalid_reason"]


def test_missing_data_is_invalid_analysis_and_halts():
    plan = make_plan(32, example_rules.frp_v4_like())
    deltas = _cut(_deltas(plan.rule, 32, effect=0.0), 11)
    del deltas["guard"]
    d = sequential_phase_r_decision(plan, 0, deltas, FLAGS)
    assert d["status"] == "INVALID_ANALYSIS" and d["action"] == "HALT"


def test_failed_prefix_controls_block_go():
    plan = make_plan(32, example_rules.frp_v4_like())
    deltas = _cut(_deltas(plan.rule, 32, effect=400.0, sd=60.0, surv=0.2), 11)
    d = sequential_phase_r_decision(plan, 0, deltas, {"prefix_controls": False})
    assert d["status"] != "GO_R" and "c5_prefix_controls" in d["failed_go_clauses"]
    with pytest.raises(ValueError):
        sequential_phase_r_decision(plan, 0, deltas, [FLAGS, FLAGS])


def test_interim_margins_and_final_point_rule():
    plan = make_plan(32, example_rules.frp_v4_like())
    deltas = _deltas(plan.rule, 32, effect=0.0, seed=5)
    d0 = sequential_phase_r_decision(plan, 0, _cut(deltas, 11), FLAGS)
    t = 11 / 32
    row = d0["clauses"]["c3_frozen_survival"]  # protective: sqrt(1 - t) shape at z 1.645
    assert row["margin"] == pytest.approx(plan.protective_margin_z * row["se"] * math.sqrt(1 - t))
    row = d0["clauses"]["c1_point"]  # non-protective: (1 - sqrt t) at z 1.2816
    assert row["margin"] == pytest.approx(plan.margin_z * row["se"] * (1 - math.sqrt(t)))
    d2 = sequential_phase_r_decision(plan, 2, deltas, FLAGS)
    assert all(r["margin"] == 0.0 for r in d2["clauses"].values() if r["kind"] == "point") or d2[
        "status"
    ] in ("KILL_V4",)


def test_final_outcome_suppresses_interim_kill_and_wins_at_final():
    rule = example_rules.frp_v5_like(seeds=(1, 2, 3, 4, 5, 6, 7, 8))
    plan = make_plan(32, rule)
    # H ~ C ~ I on H5000 (KILL on the contrast is likely) but a large MI10 contrast holds RECIPE
    deltas = _deltas(
        plan.rule, 32, effect=8.0, sd=2.0, surv=0.01, surv_sd=0.01, seed=9, control_effect=5.0
    )
    for seed in plan.rule["cells"]["primary"]["seeds"]:
        for mix in MIXES:
            deltas["primary"]["mi10"][seed][mix] = [
                v + 30.0 for v in deltas["primary"]["mi10"][seed][mix]
            ]
    d0 = sequential_phase_r_decision(plan, 0, _cut(deltas, 11), FLAGS)
    assert d0["kill"]["all_below"] and d0["kill"]["suppressed_by"] == ["GO_H_RECIPE"]
    assert d0["status"] == "CONTINUE"
    d = sequential_phase_r_decision(plan, 2, deltas, FLAGS)
    assert d["status"] == "GO_H_RECIPE"


def test_no_go_stops_only_when_declared():
    deltas_rule = example_rules.frp_v4_like()
    plain = make_plan(32, deltas_rule)
    with_no_go = make_plan(32, example_rules.frp_v4_like(no_go=50.0))
    deltas = _deltas(plain.rule, 32, effect=0.0, sd=150.0, seed=21)
    a = sequential_phase_r_decision(plain, 0, _cut(deltas, 11), FLAGS)
    b = sequential_phase_r_decision(with_no_go, 0, _cut(deltas, 11), FLAGS)
    assert a["status"] in ("CONTINUE", "KILL_V4")
    assert b["status"] in ("NO_GO_EARLY", "KILL_V4")
    if a["status"] == "CONTINUE":
        assert b["status"] == "NO_GO_EARLY"


# ----------------------------------------------------------------------------- simulator
@pytest.mark.parametrize("family", ["v4", "v5"])
def test_vectorised_simulator_matches_pure_decision(family):
    maker = example_rules.frp_v4_like if family == "v4" else example_rules.frp_v5_like
    pool = sim.load_pool()
    for no_go, effect in ((None, 65.0), (50.0, 0.0), (None, -10.0)):
        plan = make_plan(32, maker(no_go=no_go))
        scenario = {"effect": effect, "tau2": 500.0, "control_effect": 7.0}
        data = sim.draw(pool, plan.rule, scenario, 32, 12, np.random.default_rng(4))
        result = sim.evaluate(sim.sequential_design(plan, "s"), data)
        for r in range(12):
            deltas = {
                cell: {
                    metric: {
                        s: {m: data[cell][r, i, :, j, idx].tolist() for j, m in enumerate(MIXES)}
                        for i, s in enumerate(spec["seeds"])
                    }
                    for metric, idx in sim.METRIC_INDEX.items()
                }
                for cell, spec in plan.rule["cells"].items()
            }
            d = _run(plan, deltas)
            assert d["status"] == result["status"][r]
            assert d["look"] == result["stop"][r]


def test_fixed_design_is_the_one_look_rule():
    plan = make_plan(32, example_rules.frp_v4_like(no_go=50.0))
    fixed = sim.fixed_plan(plan.rule, 32)
    assert fixed["rule"]["no_go"] is None and fixed["go_nominal_p"] == (0.10,)
    data = sim.draw(sim.load_pool(), plan.rule, {"effect": 0.0}, 32, 50, np.random.default_rng(1))
    result = sim.evaluate(fixed, data)
    assert set(result["stop"]) == {0} and np.all(result["fraction"] == 1.0)


def test_variance_pool_provenance():
    pool = json.loads((PKG / "variance_pool_20261007.json").read_text())
    assert pool["schema"] == "sequential-phase-r-variance-pool/v1"
    assert len(pool["blocks"]) == 480 and sorted(pool["sources"]) == [
        "frp_v2_phase2",
        "frp_v3_phaseR",
        "frp_v4_phaseR",
    ]
    blocks = np.asarray(pool["blocks"])
    assert blocks.shape == (480, 3, 3)
    assert np.allclose(blocks.reshape(3, 160, 3, 3).mean(axis=1), 0.0, atol=1e-3)


# ------------------------------------------------------------------ hooks / receipts / audit
SEEDS = (1, 2, 3, 4, 5)


def _study(tmp_path, effect, n_worlds=6, no_go=None):
    rule = example_rules.frp_v4_like(seeds=SEEDS, no_go=no_go)
    plan = make_plan(n_worlds, rule)
    banks = {s: [1000 * s + i for i in range(n_worlds)] for s in SEEDS}
    root = tmp_path / "seq"
    receipts.write_plan(root, plan.as_dict(), {str(k): v for k, v in banks.items()}, {"study": "t"})
    return plan, banks, root


def _write_records(shard: Path, plan, banks, lo, hi, effect, rng, broken_control=False):
    shard.mkdir(parents=True, exist_ok=True)
    (shard / "records").mkdir(exist_ok=True)
    heroes = [("incumbent", SEEDS), ("R4@60000", SEEDS), ("champion", SEEDS[:1])]
    for hero, seeds in heroes:
        for seed in seeds:
            for mix in MIXES:
                for world in banks[seed][lo:hi]:
                    shift = {"incumbent": 0.0, "R4@60000": effect, "champion": -75.0}[hero]
                    mi5 = float(300 + shift + rng.normal(0, 30))
                    rec = {
                        "hero": hero,
                        "seed": seed,
                        "mix": mix,
                        "world_seed": world,
                        "control": False,
                        "prefix_h5000": {
                            "mass_integral": mi5,
                            "survival_fraction": float(0.6 + rng.normal(0, 0.05)),
                        },
                        "record": {
                            "mass_integral": float(2 * mi5 + rng.normal(0, 30)),
                            "survival_fraction": 0.5,
                        },
                    }
                    name = f"{hero}-s{seed}-{mix}-{world}.json".replace("@", "_")
                    (shard / "records" / name).write_text(json.dumps(rec))
                    if lo == 0 and world == banks[seed][0] and hero != "champion":
                        control = dict(rec, control=True, record=dict(rec["prefix_h5000"]))
                        del control["prefix_h5000"]
                        if broken_control:
                            control["record"]["mass_integral"] += 1.0
                        (shard / "records" / f"control-{name}").write_text(json.dumps(control))


def _play(tmp_path, effect, no_go=None, stop_after=None):
    plan, banks, root = _study(tmp_path, effect, no_go=no_go)
    rng = np.random.default_rng(0)
    shards = []
    for k, (lo, hi) in enumerate(hooks.look_ranges(plan.look_sizes)):
        binding = receipts.look_gate(root, k)
        assert binding["world_index_range"] == [lo, hi]
        shard = tmp_path / f"shard-L{k}"
        _write_records(shard, plan, banks, lo, hi, effect, rng)
        shards.append(shard)
        entries = hooks.load_entries(shards)
        receipt = hooks.analyse_look(root, plan, entries, k, FLAGS)
        if receipt["action"] != "CONTINUE" or (stop_after is not None and k == stop_after):
            break
    return plan, root, shards, receipt


def test_look_units_schedule():
    plan = make_plan(32, example_rules.frp_v4_like())
    banks = {s: list(range(100 * s, 100 * s + 32)) for s in plan.rule["seeds"]}
    heroes = [
        {"hero": "incumbent", "seeds": plan.rule["seeds"]},
        {"hero": "R4@60000", "seeds": plan.rule["seeds"]},
    ]
    controls = [{"hero": "incumbent", "seeds": plan.rule["seeds"][:1]}]
    units0 = hooks.look_units(plan, 0, banks, heroes, controls)
    units1 = hooks.look_units(plan, 1, banks, heroes, controls)
    assert all(len(u["worlds"]) <= 8 for u in units0 + units1)
    worlds0 = {w for u in units0 if not u["control"] for w in u["worlds"]}
    assert worlds0 == {w for s in plan.rule["seeds"] for w in banks[s][:11]}
    assert not any(u["control"] for u in units1)
    assert sum(u["control"] for u in units0) == 3
    groups = {}
    for u in units0:
        groups.setdefault(u["group"], set()).add(u["hero"])
    assert all(len(h) >= 2 for g, h in groups.items())  # every world group has both arms


def test_end_to_end_receipts_and_audit_pass(tmp_path):
    plan, root, shards, receipt = _play(tmp_path, effect=0.0)
    assert receipt["action"] == "STOP"
    result = audit_mod.audit(root, shards)
    assert result["verdict"] == "PASS", result["problems"]
    assert result["final_status"] == receipt["decision"]["status"]
    # create-only: the last receipt cannot be rewritten, and no further look may start
    with pytest.raises(receipts.ReceiptError):
        receipts.write_once(receipts.receipt_path(root, receipt["look"]), {"x": 1})
    if receipt["look"] < len(plan.look_sizes) - 1:
        with pytest.raises(receipts.ReceiptError):
            receipts.look_gate(root, receipt["look"] + 1)


def test_audit_detects_tampering_and_beyond_look_records(tmp_path):
    plan, root, shards, receipt = _play(tmp_path, effect=-400.0)  # KILL at look 0
    assert receipt["look"] == 0 and receipt["decision"]["status"] == "KILL_V4"
    assert audit_mod.audit(root, shards)["verdict"] == "PASS"
    # a record of look 1 that should never have been played
    extra = tmp_path / "shard-extra"
    _write_records(
        extra,
        plan,
        {s: [1000 * s + i for i in range(6)] for s in SEEDS},
        2,
        3,
        0.0,
        np.random.default_rng(1),
    )
    bad = audit_mod.audit(root, shards + [extra])
    assert bad["verdict"] == "FAIL" and any(
        "beyond the stopping look" in p for p in bad["problems"]
    )
    # tamper with a used record
    used = receipt["records"][0]["path"]
    data = json.loads(Path(used).read_text())
    data["prefix_h5000"]["mass_integral"] += 1.0
    Path(used).write_text(json.dumps(data))
    bad = audit_mod.audit(root, shards)
    assert bad["verdict"] == "FAIL" and any("sha256 mismatch" in p for p in bad["problems"])


def test_audit_unclosed_and_gate_refusals(tmp_path):
    plan, root, shards, receipt = _play(tmp_path, effect=0.0, stop_after=0)
    if receipt["action"] == "CONTINUE":
        assert audit_mod.audit(root, shards)["verdict"] == "UNCLOSED"
        with pytest.raises(receipts.ReceiptError, match="already analysed"):
            receipts.look_gate(root, 0)
        with pytest.raises(receipts.ReceiptError, match="no receipt"):
            receipts.look_gate(root, 2)
    with pytest.raises(receipts.ReceiptError):
        receipts.write_plan(root, plan.as_dict(), {}, {})


def test_analyse_look_refuses_missing_records(tmp_path):
    plan, banks, root = _study(tmp_path, 0.0)
    shard = tmp_path / "shard-L0"
    _write_records(shard, plan, banks, 0, 1, 0.0, np.random.default_rng(0))  # 1 of 2 worlds
    with pytest.raises(hooks.LookDataError, match="missing record"):
        hooks.analyse_look(root, plan, hooks.load_entries([shard]), 0, FLAGS)
    assert not receipts.receipt_path(root, 0).exists()


def test_audit_is_standalone_stdlib(tmp_path):
    imports = [
        line.strip()
        for line in (PKG / "audit.py").read_text().splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert imports and not any(("src" in line or "research" in line) for line in imports)
    plan, root, shards, receipt = _play(tmp_path, effect=-400.0)
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(PKG / "audit.py"),
            "--root",
            str(root),
            "--record-dirs",
            *map(str, shards),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["verdict"] == "PASS"


def test_beyond_look_records_halt_and_audit_reports_halted(tmp_path):
    plan, banks, root = _study(tmp_path, 0.0)
    shard = tmp_path / "shard-L0"
    _write_records(shard, plan, banks, 0, 3, 0.0, np.random.default_rng(0))  # 3 worlds, look 0 = 2
    receipt = hooks.analyse_look(root, plan, hooks.load_entries([shard]), 0, FLAGS)
    assert receipt["action"] == "HALT"
    assert (
        receipt["decision"]["status"] == "INVALID_ANALYSIS"
        and "status_if_valid" in receipt["decision"]
    )
    with pytest.raises(receipts.ReceiptError):
        receipts.look_gate(root, 1)
    result = audit_mod.audit(root, [])
    assert result["verdict"] == "HALTED" and result["final_status"] == "INVALID_ANALYSIS"


def test_audit_recomputes_prefix_controls(tmp_path):
    plan, banks, root = _study(tmp_path, 0.0)
    shard = tmp_path / "shard-L0"
    _write_records(shard, plan, banks, 0, 2, -400.0, np.random.default_rng(0), broken_control=True)
    receipt = hooks.analyse_look(root, plan, hooks.load_entries([shard]), 0, FLAGS)
    assert receipt["action"] == "STOP"
    bad = audit_mod.audit(root, [shard])
    assert bad["verdict"] == "FAIL" and any("prefix_controls" in p for p in bad["problems"])


def test_validate_rule_rejects_pooled_stats_on_single_seed_cells():
    rule = example_rules.frp_v4_like()
    rule["go_clauses"].append(
        {
            "name": "bad",
            "kind": "point",
            "stat": {"kind": "hk", "cell": "guard", "metric": "mi5"},
            "op": ">",
            "threshold": 0.0,
        }
    )
    with pytest.raises(ValueError, match=">= 2 seeds"):
        validate_rule(rule)


def test_audit_boundaries_match_module():
    plan = make_plan(32, example_rules.frp_v4_like())
    for side in ("go", "kill"):
        mine = audit_mod.boundaries(
            plan.fractions, getattr(plan, f"{side}_alpha"), getattr(plan, f"{side}_spending")
        )
        assert mine == pytest.approx(getattr(plan, f"{side}_boundaries"), abs=2e-4)
