"""Survival band v2 of research/sequential_strict_template: ``band_policy="pooled_ni_continue"``.

Governance amendment survival band v2 (2026-10-06): pooled paired NI (mean over the mixes of the
per-world survival delta) plus a per-mix catastrophic NI, repeated confidence bounds (rci_obf),
judged at the qualifying look and every later look; a qualified run whose bands fail continues
(``CONTINUE_BANDS``).  Same guards as the v1/v2 tests (the autouse ``guards`` fixture): dry-run
intents, a fake episode function, the in-process executor, a fake skew probe, a fake per-study
check output and the in-process audit.
"""

from __future__ import annotations

import json
import random

import pytest

from research.sequential_strict_template import sequential_audit as A
from research.sequential_strict_template import sequential_runner as R
from src.evaluation.sequential_gate import pooled_band_check
from tests.test_sequential_strict_template import (  # noqa: F401 - guards is an autouse fixture
    MIXES,
    N_MAX,
    FakeWorld,
    execute,
    final_records,
    guards,
    intent_kwargs,
    make_spec,
    output_of,
    receipts,
    why,
)

OPTION_1 = dict(R.POOLED_BAND_OPTIONS["1"])
THETAS = (15.0, 20.0, 30.0, 45.0, 60.0)
POOLS = ("screen_a", "screen_b")
DEV_DELTA_NI = 5.8


class NoisySurvivalWorld(FakeWorld):
    """FakeWorld with realistic per-arm survival noise (between-checkpoint deaths not tied)
    and a candidate survival shift per mix."""

    def __init__(self, survival_noise=0.04, shift=None, **kwargs):
        super().__init__(**kwargs)
        self.survival_noise = survival_noise
        self.shift = shift or {}

    def __call__(self, episode, row, context):
        record = super().__call__(episode, row, context)
        arm = random.Random(f"{episode['world_seed']}|{episode['mix']}|{episode['arm']}|surv")
        record["survival_fraction"] = 0.6 + arm.gauss(0, self.survival_noise)
        if episode["arm"] == "candidate":
            record["survival_fraction"] += self.shift.get(episode["mix"], 0.0)
        return record


def plan_params(spec, band=None, **overrides):
    kwargs = dict(n_max=N_MAX, mde=30.0, paired_band=OPTION_1 if band is None else band)
    kwargs.update(overrides)
    return R.study_plan_parameters(spec, **kwargs)


def write_check(tmp_path, params, *, rate=0.045, reps=20000, thetas=THETAS, delta_ni=DEV_DELTA_NI):
    """A ``survival_band_v2 simulate.py --part study`` shaped output (no simulation)."""
    data = tmp_path / "screen_pool.json"
    if not data.exists():
        rng = random.Random(3)
        pool = {
            "world_seeds": list(range(30)),
            **{
                m: {
                    "incumbent_survival": [rng.random() for _ in range(30)],
                    "candidate_survival": [rng.random() for _ in range(30)],
                    "mass_delta": [rng.gauss(30, 100) for _ in range(30)],
                }
                for m in MIXES
            },
        }
        data.write_text(json.dumps({p: pool for p in POOLS}))
    scenarios = ["pooled_at_margin"] + [f"one_mix_at_margin@{m}" for m in MIXES]
    results, joint = {}, {}
    for pool in POOLS:
        for theta in thetas:
            results[f"{pool}|theta={theta:g}|no_change"] = {"p_pass": {"rate": 0.9}}
            for scenario in scenarios:
                key = f"{pool}|theta={theta:g}|{scenario}"
                joint[key] = rate
                results[key] = {"p_pass": {"rate": rate}}
    threshold = 1.2 * params["band_alpha"]
    report = {
        "schema": "survival-band-v2-study-check/v1",
        "config": {
            "plan_params": params,
            "delta_ni": delta_ni,
            "thetas": list(thetas),
            "data": str(data),
        },
        "data_sha256": R.sha256_file(data),
        "reps": reps,
        "results": results,
        "check": {
            "rule": "pooled_ni_continue",
            "threshold": threshold,
            "joint_rates": joint,
            "max_joint_rate": rate,
            "passes": rate <= threshold,
        },
        "cpu_seconds": 1.0,
    }
    path = tmp_path / "check.json"
    path.write_text(json.dumps(report))
    return path


def make_pooled_spec(runner, tmp_path, *, band=None, **check_kwargs):
    base = make_spec(runner, tmp_path)
    draft = R.StudySpec(
        **{
            **base.__dict__,
            "band_policy": "pooled_ni_continue",
            "bands": R.paired_survival_bands(MIXES),
            "paired_band_check_path": str(tmp_path / "check.json"),
            "paired_band_pool_path": str(tmp_path / "screen_pool.json"),
        }
    )
    write_check(tmp_path, plan_params(draft, band), **check_kwargs)
    return draft


def prepare_pooled(tmp_path, spec, band=None, **overrides):
    kwargs = intent_kwargs(
        tmp_path, paired_band=OPTION_1 if band is None else band, development_delta_ni=DEV_DELTA_NI
    )
    kwargs.update(overrides)
    return R.prepare(R.build_intent(spec, **kwargs))


# ---------------------------------------------------------------- end to end


def test_pooled_band_pass_at_first_qualifying_look_and_audit(tmp_path):
    spec = make_pooled_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_pooled(tmp_path, spec)
    intent = R.read_json(intent_path)
    assert intent["template_version"] == "sequential-strict-template/v2-pooled"
    assert intent["spec"]["template_version"] == "sequential-strict-template/v2-pooled"
    assert intent["plan"]["band_policy"] == "pooled_ni_continue"
    assert intent["plan"]["band_mix_margin"] == 0.075 and intent["plan"]["band_bound"] == "rci_obf"
    assert intent["band_rule"]["policy"] == "pooled_ni_continue"
    amendment = intent["band_amendment"]
    assert amendment["path"].endswith("governance_amendment_survival_band_v2_2026-10-06.md")
    assert amendment["sha256"] and amendment["ratified"] is True  # ratified option 1 (865ee9e)
    assert amendment["production_problems"] == []
    assert intent["paired_band_check"]["rule"] == "pooled_ni_continue"
    assert intent["paired_band_check"]["rows_judged"] == len(POOLS) * len(THETAS) * 4
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    assert closeout["stop_decision"] == "STOP_PASS" and closeout["stop_look"] == 0
    [receipt] = receipts(intent_path)
    assert receipt["band_policy"] == "pooled_ni_continue"
    assert receipt["band_judged_looks"] == [0] and receipt["band_judged_look"] == 0
    [row] = receipt["band_results"]
    assert row["policy"] == "pooled_ni_continue" and row["mixes"] == list(MIXES)
    assert row["n"] == 10 and row["nominal_p"] == intent["plan"]["band_nominal_p"][0]
    assert row["pooled"]["margin"] == 0.05 and row["passes"] is True
    assert all(row["per_mix"][m]["margin"] == 0.075 for m in MIXES)
    report = A.run_audit(intent_path.parent)
    assert report["status"] == "PASS", report["failures"]
    assert report["schema_version"] == "sequential-strict-audit/v2-pooled"
    rules = {r["rule"] for r in report["checks"]}
    assert {"looks.paired_bands", "preregistration.pooled_band_check"} <= rules
    assert {"intent.band_amendment", "plan.paired_band_nominal_p"} <= rules


def test_qualified_run_whose_bands_fail_continues_then_passes(tmp_path):
    world = NoisySurvivalWorld(effect=400.0, survival_noise=0.04, shift={m: -0.01 for m in MIXES})
    spec = make_pooled_spec(world, tmp_path)
    intent_path = prepare_pooled(tmp_path, spec)
    closeout = execute(intent_path, spec)
    looks = receipts(intent_path)
    decisions = [r["decision"] for r in looks]
    assert decisions[0] == "CONTINUE_BANDS" and looks[0]["action"] == "continue"
    assert decisions[-1] in ("STOP_PASS", "FINAL_PASS") and len(looks) >= 2
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    assert looks[-1]["band_judged_looks"] == list(range(len(looks)))
    # the records stop at the stopping look's prefix
    expected = 2 * 3 * R.read_json(intent_path)["plan"]["look_sizes"][len(looks) - 1]
    assert len(final_records(intent_path)) == expected
    report = A.run_audit(intent_path.parent)
    assert report["status"] == "PASS", report["failures"]


def test_survival_regression_continues_to_final_fail(tmp_path):
    world = NoisySurvivalWorld(effect=400.0, survival_noise=0.04, shift={m: -0.2 for m in MIXES})
    spec = make_pooled_spec(world, tmp_path)
    intent_path = prepare_pooled(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", why(closeout)
    looks = receipts(intent_path)
    assert [r["decision"] for r in looks] == ["CONTINUE_BANDS"] * 3 + ["FINAL_FAIL"]
    assert looks[-1]["band_results"][0]["passes_pooled"] is False
    report = A.run_audit(intent_path.parent)
    assert report["status"] == "PASS", report["failures"]


def test_one_mix_catastrophe_fails_while_pooled_passes(tmp_path):
    world = NoisySurvivalWorld(
        effect=400.0, survival_noise=0.01, shift={"frozen": 0.08, "scripted": -0.15, "mixed": 0.08}
    )
    spec = make_pooled_spec(world, tmp_path)
    intent_path = prepare_pooled(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", why(closeout)
    last = receipts(intent_path)[-1]["band_results"][0]
    assert last["passes_pooled"] is True and last["passes_mix"] is False
    assert last["per_mix"]["scripted"]["passes_ni"] is False


def test_audit_catches_a_forged_pooled_band_row(tmp_path):
    spec = make_pooled_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_pooled(tmp_path, spec)
    execute(intent_path, spec)
    path = output_of(intent_path) / "looks" / "look-0.json"
    receipt = R.read_json(path)
    receipt["bands_by_look"][0][0]["pooled"]["lower_bound"] += 0.01
    path.chmod(0o644)
    path.write_text(json.dumps(receipt))
    failed = {r["rule"] for r in A.run_audit(intent_path.parent)["failures"]}
    assert "looks.paired_bands" in failed


# ---------------------------------------------------------------- intent and ratification


def test_pooled_band_needs_five_explicit_values_and_rci(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    pooled = R.StudySpec(
        **{
            **spec.__dict__,
            "band_policy": "pooled_ni_continue",
            "bands": R.paired_survival_bands(MIXES),
        }
    )
    with pytest.raises(R.StrictRunError, match="band_mix_margin"):
        plan_params(pooled, dict(R.RATIFIED_PAIRED_BAND))
    with pytest.raises(ValueError, match="rci_obf"):
        R.plan_from_parameters(plan_params(pooled, {**OPTION_1, "band_bound": "pointwise"}))
    params = plan_params(pooled)
    assert params["band_mix_margin"] == 0.075 and params["band_policy"] == "pooled_ni_continue"
    assert R.parse_paired_band("0.05,0.05,rci_obf,0.30,0.075") == OPTION_1
    assert "band_mix_margin" not in R.parse_paired_band("0.05,0.05,rci_obf,0.30")
    with pytest.raises(R.StrictRunError):
        R.parse_paired_band("0.05,0.05,rci_obf")


def test_amendment_status_and_ratified_option(tmp_path):
    doc = tmp_path / "amendment.md"
    doc.write_text("## Ratification\n\n- Decision: ______\n- Option: ______\n")
    status = R.amendment_status(doc)
    assert status["ratified"] is False and status["option"] is None
    assert len(R.ratified_band_problems(status, OPTION_1)) == 2
    doc.write_text("## Ratification\n\n- Decision: ratified\n- Option: 1\n")
    status = R.amendment_status(doc)
    assert status["ratified"] is True and status["option"] == "1"
    assert R.ratified_band_problems(status, OPTION_1) == []
    assert R.ratified_band_problems(status, {**OPTION_1, "band_mix_margin": 0.10})
    doc.write_text("## Ratification\n\n- Decision: ratified with changes\n- Option: 1\n")
    assert R.amendment_status(doc)["ratified"] is False
    doc.write_text(
        "## Ratification\n\n- Decision: ratified\n- Option: 1\n- Changes, if any: margin 0.06\n"
    )
    changed = R.amendment_status(doc)
    assert changed["ratified"] is True and changed["changes"] == "margin 0.06"
    assert any("records changes" in p for p in R.ratified_band_problems(changed, OPTION_1))
    doc.write_text(
        "## Ratification\n\n- Decision: Ratified\n- Option: 1\n- Changes, if any: none\n"
    )
    assert R.ratified_band_problems(R.amendment_status(doc), OPTION_1) == []
    doc.write_text("## Ratification\n\n- Decision: ratified\n- Option: 2\n")
    assert R.ratified_band_problems(R.amendment_status(doc), R.POOLED_BAND_OPTIONS["2"]) == []
    repo_copy = R.amendment_status(R.REPO / R.SURVIVAL_BAND_V2_AMENDMENT)
    # ratified option 1 in the repo at 865ee9e (2026-10-06)
    assert repo_copy["sha256"] and repo_copy["ratified"] is True and repo_copy["option"] == "1"


def test_production_intent_refused_until_ratified(tmp_path, monkeypatch):
    spec = make_pooled_spec(FakeWorld(effect=400.0), tmp_path)
    real_closure = R.source_closure

    def clean_closure(roots, repo=R.REPO):
        out = real_closure(roots, repo)
        out["dirty"] = []
        return out

    monkeypatch.setattr(R, "source_closure", clean_closure)
    R.SLOT_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    kwargs = intent_kwargs(
        tmp_path,
        paired_band=OPTION_1,
        development_delta_ni=DEV_DELTA_NI,
        dry_run=False,
        allow_dirty=False,
        slot_lock_root=R.SLOT_LOCK_ROOT,
        ledger_path=R.LEDGER_PATH,
    )
    ratified = {
        "path": str(R.REPO / R.SURVIVAL_BAND_V2_AMENDMENT),
        "sha256": "x",
        "ratified": True,
        "option": "1",
    }
    # the repo copy is ratified since 865ee9e: simulate the Pending state it had before
    pending = {**ratified, "ratified": False, "option": None}
    monkeypatch.setattr(R, "amendment_status", lambda path: dict(pending))
    with pytest.raises(R.StrictRunError, match="not ratified"):
        R.build_intent(spec, **kwargs)
    monkeypatch.setattr(R, "amendment_status", lambda path: dict(ratified))
    intent = R.build_intent(spec, **kwargs)
    assert intent["band_amendment"]["production_problems"] == []
    monkeypatch.setattr(R, "amendment_status", lambda path: {**ratified, "option": "2"})
    with pytest.raises(R.StrictRunError, match="ratified option 2"):
        R.build_intent(spec, **kwargs)


def test_forged_ratification_is_refused_by_runner_and_audit(tmp_path):
    spec = make_pooled_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_pooled(tmp_path, spec)
    execute(intent_path, spec)
    intent = R.read_json(intent_path)
    forged = json.loads(json.dumps(intent))
    # flip the recorded ratification (the repo copy is ratified since 865ee9e)
    forged["band_amendment"]["ratified"] = not intent["band_amendment"]["ratified"]
    with pytest.raises(R.StrictRunError, match="amendment differs"):
        R.validate_intent(forged, spec)
    intent_path.chmod(0o644)
    intent_path.write_text(json.dumps(forged))
    failed = {r["rule"] for r in A.run_audit(intent_path.parent)["failures"]}
    assert "intent.band_amendment" in failed


def test_pooled_check_must_pass_and_cover_every_scenario(tmp_path):
    spec = make_pooled_spec(FakeWorld(effect=400.0), tmp_path, rate=0.07)
    with pytest.raises(R.StrictRunError, match="check fails"):
        prepare_pooled(tmp_path, spec)
    report = json.loads((tmp_path / "check.json").read_text())
    params = plan_params(spec)
    judged = R.judge_pooled_check(report, params, spec.bands, DEV_DELTA_NI)
    assert not judged["passes"] and any("above 0.06" in p for p in judged["problems"])
    write_check(tmp_path, params)
    report = json.loads((tmp_path / "check.json").read_text())
    assert R.judge_pooled_check(report, params, spec.bands, DEV_DELTA_NI)["passes"]
    dropped = json.loads(json.dumps(report))
    del dropped["check"]["joint_rates"]["screen_a|theta=30|one_mix_at_margin@scripted"]
    assert not R.judge_pooled_check(dropped, params, spec.bands, DEV_DELTA_NI)["passes"]
    assert not R.judge_pooled_check(report, params, spec.bands, 9.9)["passes"]
    paired = dict(report, schema="other")
    assert not R.judge_pooled_check(paired, params, spec.bands, DEV_DELTA_NI)["passes"]


def test_legacy_specs_and_intents_carry_no_pooled_fields(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    assert spec.template_version == "sequential-strict-template/v1" and not spec.pooled
    descriptor = spec.descriptor()
    assert "band_policy" not in descriptor
    intent = R.build_intent(spec, **intent_kwargs(tmp_path))
    assert "band_amendment" not in intent and "band_mix_margin" not in intent["plan"]
    paired = R.StudySpec(
        **{
            **spec.__dict__,
            "band_policy": "paired_ni_at_stop",
            "bands": R.paired_survival_bands(MIXES),
        }
    )
    assert paired.template_version == "sequential-strict-template/v2" and not paired.pooled
    assert paired.descriptor()["template_version"] == "sequential-strict-template/v2"


def test_pooled_band_rows_match_the_gate_function(tmp_path):
    spec = make_pooled_spec(
        NoisySurvivalWorld(effect=400.0, survival_noise=0.04, shift={m: -0.01 for m in MIXES}),
        tmp_path,
    )
    intent_path = prepare_pooled(tmp_path, spec)
    execute(intent_path, spec)
    intent = R.read_json(intent_path)
    entries = {
        p.stem: R.read_json(p) for p in (output_of(intent_path) / "final" / "records").glob("*")
    }
    for receipt in receipts(intent_path):
        k = receipt["look"]
        n = intent["plan"]["look_sizes"][k]
        cand = {
            m: [entries[R.episode_id("final", "candidate", m, w)]["record"]["survival_fraction"]
                for w in range(n)]
            for m in MIXES
        }  # fmt: skip
        inc = {
            m: [entries[R.episode_id("final", "incumbent", m, w)]["record"]["survival_fraction"]
                for w in range(n)]
            for m in MIXES
        }  # fmt: skip
        direct = pooled_band_check(
            cand, inc, pooled_margin=0.05, mix_margin=0.075,
            nominal_p=intent["plan"]["band_nominal_p"][k], floor=0.30,
        )  # fmt: skip
        row = receipt["bands_by_look"][k][0]
        assert row["passes"] == direct["passes"]
        assert row["pooled"]["lower_bound"] == pytest.approx(direct["pooled"]["lower_bound"])


def test_check_simulator_is_bound_by_sha256(tmp_path, monkeypatch):
    simulator = tmp_path / "simulate.py"
    simulator.write_text(R.POOLED_BAND_SIMULATOR.read_text())
    monkeypatch.setattr(R, "POOLED_BAND_SIMULATOR", simulator)
    spec = make_pooled_spec(FakeWorld(effect=400.0), tmp_path)
    intent = R.read_json(prepare_pooled(tmp_path, spec))
    frozen = intent["paired_band_check"]
    assert frozen["simulator_sha256"] == R.sha256_file(simulator)
    R.validate_intent(intent, spec)
    simulator.write_text(simulator.read_text() + "\n# edited after prepare\n")
    with pytest.raises(R.StrictRunError, match="simulator changed"):
        R.validate_intent(intent, spec)
