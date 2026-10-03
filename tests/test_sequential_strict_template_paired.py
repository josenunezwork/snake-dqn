"""Template v2 of research/sequential_strict_template: paired survival bands.

``band_policy="paired_ni_at_stop"`` (governance amendment paired bands, 2026-10-03, ratified
M 0.05, alpha 0.05 pointwise, floor 0.30).  Same guards as the v1 tests: every path to a real
rollout, probe or child process raises (the autouse ``guards`` fixture is imported from
``test_sequential_strict_template``); runs are dry-run intents with a fake episode function,
the in-process executor, a fake skew probe, a fake per-study paired-band check output and the
in-process audit.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from research.sequential_strict_template import example_spec
from research.sequential_strict_template import sequential_audit as A
from research.sequential_strict_template import sequential_runner as R
from src.evaluation.sequential_gate import paired_band_check, sequential_gate_plan
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

RATIFIED = dict(R.RATIFIED_PAIRED_BAND)
THETAS = (15.0, 20.0, 30.0, 45.0, 60.0)
POOLS = ("screen_a", "screen_b")


class PairedWorld(FakeWorld):
    """FakeWorld plus a survival shift applied to both arms (absolute level, not the delta)."""

    def __init__(self, survival_base_shift=0.0, **kwargs):
        super().__init__(**kwargs)
        self.survival_base_shift = survival_base_shift

    def __call__(self, episode, row, context):
        record = super().__call__(episode, row, context)
        record["survival_fraction"] += self.survival_base_shift
        return record


def plan_params(spec, paired_band=None, **overrides):
    kwargs = dict(
        n_max=N_MAX, mde=30.0, paired_band=RATIFIED if paired_band is None else paired_band
    )
    kwargs.update(overrides)
    return R.study_plan_parameters(spec, **kwargs)


def write_check(tmp_path, params, *, rate=0.045, reps=20000, thetas=THETAS, name="check.json"):
    """A ``simulate.py --part gate`` shaped output of the per-study check (no simulation)."""
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
    margin, alpha, bound = params["band_ni_margin"], params["band_alpha"], params["band_bound"]
    rule = R.paired_rule_name(margin, alpha, bound)
    results, joint = {}, {}
    for pool in POOLS:
        for theta in thetas:
            for mix in MIXES:
                key = f"{pool}|theta={theta:g}|one_mix_at_margin@{mix}"
                joint[key] = rate
                results[key] = {rule: {"p_regressed_band_and_qualify": rate}}
    threshold = 1.2 * alpha
    report = {
        "reps": reps,
        "delta_ni": 5.8,
        "results": results,
        "check": {
            "rule": rule,
            "threshold": threshold,
            "joint_rates": joint,
            "max_joint_rate": rate,
            "passes": rate <= threshold,
        },
        "data_sha256": R.sha256_file(data),
        "config": {
            "data": str(data),
            "plan_params": params,
            "check_rule": [margin, alpha, bound],
            "thetas": list(thetas),
            "delta_ni": 5.8,
            "n_max": params["n_max"],
            "mde": params["mde"],
        },
        "cpu_seconds": 1.0,
    }
    path = tmp_path / name
    path.write_text(json.dumps(report))
    return path


def make_paired_spec(runner, tmp_path, *, paired_band=None, plan_overrides=None, **check_kwargs):
    base = make_spec(runner, tmp_path)
    draft = R.StudySpec(
        **{
            **base.__dict__,
            "band_policy": "paired_ni_at_stop",
            "bands": R.paired_survival_bands(MIXES),
            "paired_band_check_path": str(tmp_path / "check.json"),
        }
    )
    params = plan_params(draft, paired_band, **(plan_overrides or {}))
    write_check(tmp_path, params, **check_kwargs)
    return draft


def prepare_paired(tmp_path, spec, paired_band=None, **overrides):
    kwargs = intent_kwargs(tmp_path, paired_band=RATIFIED if paired_band is None else paired_band)
    kwargs.update(overrides)
    return R.prepare(R.build_intent(spec, **kwargs))


# ---------------------------------------------------------------- end to end


def test_paired_band_pass_records_band_results_and_audits(tmp_path):
    spec = make_paired_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_paired(tmp_path, spec)
    intent = R.read_json(intent_path)
    assert intent["template_version"] == "sequential-strict-template/v2"
    assert intent["spec"]["band_policy"] == "paired_ni_at_stop"
    assert intent["plan"]["band_policy"] == "paired_ni_at_stop"
    assert intent["plan"]["band_nominal_p"] == [0.05] * 4 and intent["plan"]["band_floor"] == 0.3
    assert set(intent["preregistration"]) == {
        "protocol",
        "oc_report",
        "band_cost_report",
        "paired_band_check",
    }
    assert intent["paired_band_check"]["passes"] is True
    assert intent["paired_band_check"]["rows_judged"] == len(POOLS) * len(THETAS) * 3
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    assert closeout["stop_decision"] == "STOP_PASS" and closeout["stop_look"] == 0
    [receipt] = receipts(intent_path)
    assert receipt["band_policy"] == "paired_ni_at_stop" and receipt["band_judged_look"] == 0
    assert [b["mix"] for b in receipt["band_results"]] == list(MIXES)
    for band in receipt["band_results"]:
        assert band["n"] == 10 and band["df"] == 9 and band["margin"] == 0.05
        assert band["floor"] == 0.3 and band["nominal_p"] == 0.05
        assert band["lower_bound"] == pytest.approx(
            band["mean_delta"] - band["t_critical"] * band["standard_error"]
        )
        assert band["passes"] is band["passes_ni"] is band["passes_floor"] is True
    calibration = R.read_json(output_of(intent_path) / "calibration.json")
    assert all("lower" not in b and "incumbent_calibration_mean" in b for b in calibration["bands"])
    report = A.run_audit(intent_path.parent)
    assert report["status"] == "PASS", report["failures"]
    assert report["schema_version"] == "sequential-strict-audit/v2"
    rules = {row["rule"] for row in report["checks"]}
    assert {"looks.paired_bands", "preregistration.paired_band_check"} <= rules
    assert {"plan.paired_band_nominal_p", "plan.paired_band_parameters"} <= rules


def test_paired_band_regression_blocks_at_the_qualifying_look(tmp_path):
    spec = make_paired_spec(FakeWorld(effect=400.0, survival_shift=-0.2), tmp_path)
    intent_path = prepare_paired(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", why(closeout)
    assert closeout["stop_look"] == 0 and closeout["stop_decision"] == "STOP_FAIL_BANDS"
    [receipt] = receipts(intent_path)
    assert receipt["bands_pass_by_look"] == [False] and receipt["action"] == "stop"
    assert all(b["passes_ni"] is False and b["passes_floor"] for b in receipt["band_results"])
    assert len(final_records(intent_path)) == 60  # nothing beyond the look-0 prefix
    assert closeout["audit_passed"] is True


def test_floor_trips_when_both_arms_collapse(tmp_path):
    # paired delta ~ 0 (NI holds) but the candidate's absolute survival is ~0.1 < floor 0.30
    spec = make_paired_spec(PairedWorld(effect=400.0, survival_base_shift=-0.7), tmp_path)
    intent_path = prepare_paired(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", why(closeout)
    assert closeout["stop_decision"] == "STOP_FAIL_BANDS"
    [receipt] = receipts(intent_path)
    for band in receipt["band_results"]:
        assert band["passes_ni"] is True and band["passes_floor"] is False
        assert band["candidate_mean"] < 0.3
    assert closeout["audit_passed"] is True


def test_no_floor_lets_the_same_run_pass(tmp_path):
    no_floor = dict(RATIFIED, band_floor=None)
    spec = make_paired_spec(
        PairedWorld(effect=400.0, survival_base_shift=-0.7), tmp_path, paired_band=no_floor
    )
    intent_path = prepare_paired(tmp_path, spec, paired_band=no_floor)
    assert "band_floor" in R.read_json(intent_path)["plan"]
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    assert all(b["floor"] is None for b in receipts(intent_path)[0]["band_results"])


def test_paired_run_without_qualifying_has_no_judged_band(tmp_path):
    spec = make_paired_spec(
        FakeWorld(effect=0.0),
        tmp_path,
        plan_overrides={"mde": 2000.0},
        thetas=(1000, 1334, 2000, 3000, 4000),
    )
    intent_path = prepare_paired(tmp_path, spec, mde=2000.0)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL" and closeout["stop_look"] == 3, why(closeout)
    looks = receipts(intent_path)
    assert all(r["band_judged_look"] is None and r["band_results"] is None for r in looks)
    assert [len(r["bands_by_look"]) for r in looks] == [1, 2, 3, 4]
    assert A.run_audit(intent_path.parent)["status"] == "PASS"


# ---------------------------------------------------------------- audit catches tampering


def test_audit_catches_an_altered_survival_value(tmp_path):
    spec = make_paired_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_paired(tmp_path, spec)
    execute(intent_path, spec)
    path = output_of(intent_path) / "final" / "records" / "final-incumbent-mixed-w00003.json"
    entry = json.loads(path.read_text())
    entry["record"]["survival_fraction"] -= 0.05  # mass (efficacy deltas) untouched
    path.write_text(json.dumps(entry))
    report = A.run_audit(intent_path.parent)
    assert report["status"] == "FAIL"
    failed = {row["rule"]: row["detail"] for row in report["failures"]}
    assert "looks.paired_bands" in failed
    assert any("mixed" in row and "mean_delta" in row for row in failed["looks.paired_bands"])
    assert not any("deltas_digest" in row for row in failed["looks.replay"])  # bands only


def test_audit_catches_an_altered_band_receipt(tmp_path):
    spec = make_paired_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_paired(tmp_path, spec)
    execute(intent_path, spec)
    path = output_of(intent_path) / "looks" / "look-0.json"
    receipt = json.loads(path.read_text())
    receipt["bands_by_look"][0][1]["lower_bound"] -= 0.01
    receipt["band_results"][2]["floor"] = 0.1
    path.write_text(json.dumps(receipt))
    report = A.run_audit(intent_path.parent)
    rows = next(r["detail"] for r in report["failures"] if r["rule"] == "looks.paired_bands")
    assert any("scripted" in row and "lower_bound" in row for row in rows)
    assert any("band_results" in row for row in rows)


def test_audit_catches_an_edited_paired_check_and_floor_in_plan(tmp_path):
    spec = make_paired_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_paired(tmp_path, spec)
    execute(intent_path, spec)
    check = Path(spec.paired_band_check_path)
    report = json.loads(check.read_text())
    report["check"]["joint_rates"][next(iter(report["check"]["joint_rates"]))] = 0.08
    check.write_text(json.dumps(report))
    intent = json.loads(intent_path.read_text())
    intent["plan"]["band_floor"] = 0.0
    intent_path.write_text(json.dumps(intent))
    failed = {row["rule"] for row in A.run_audit(intent_path.parent)["failures"]}
    assert {
        "preregistration.paired_band_check",
        "intent.preregistration_documents",
        "plan.paired_band_parameters",
        "plan.sha256",
    } <= failed


def test_audit_paired_band_matches_sequential_gate():
    rng = random.Random(11)
    for n in (2, 10, 63, 249):
        inc = [rng.random() for _ in range(n)]
        cand = [min(1.0, max(0.0, v + rng.gauss(-0.02, 0.2))) for v in inc]
        for p, floor in ((0.05, 0.3), (0.0012, None), (0.1, 0.9)):
            mine = A.paired_band(cand, inc, 0.05, p, floor)
            theirs = paired_band_check(cand, inc, margin=0.05, nominal_p=p, floor=floor)
            for key in A.PAIRED_COMPARED:
                if key != "df":
                    assert mine[key] == pytest.approx(theirs[key], rel=1e-9, abs=1e-12), key
            assert all(mine[k] is theirs[k] for k in A.PAIRED_VERDICTS)
    flat = A.paired_band([0.5, 0.5, 0.5], [0.55, 0.55, 0.55], 0.05, 0.05, None)
    assert flat["lower_bound"] == pytest.approx(-0.05) and flat["passes"] is False  # strict >


def test_audit_recomputes_rci_obf_band_levels_and_rejects_altered_ones():
    plan = sequential_gate_plan(
        N_MAX,
        mde=30.0,
        band_policy="paired_ni_at_stop",
        band_ni_margin=0.05,
        band_alpha=0.10,
        band_bound="rci_obf",
        band_floor=None,
    ).as_dict()
    params = {k: plan[k] for k in A.PAIRED_PARAM_FIELDS}
    fractions = tuple(plan["fractions"])
    audit = A.Audit()
    A.audit_paired_plan(audit, plan, params, fractions)
    assert not audit.failures, audit.failures
    plan["band_nominal_p"][0] *= 1.5
    audit = A.Audit()
    A.audit_paired_plan(audit, plan, params, fractions)
    assert {r["rule"] for r in audit.failures} == {"plan.paired_band_nominal_p"}


# ---------------------------------------------------------------- pre-registration


def test_paired_check_is_required_and_judged_before_prepare(tmp_path):
    spec = make_paired_spec(FakeWorld(), tmp_path)
    with pytest.raises(R.StrictRunError, match="paired_band_check_path"):
        R.validate_spec(R.StudySpec(**{**spec.__dict__, "paired_band_check_path": ""}))
    with pytest.raises(R.StrictRunError, match="explicit"):
        R.build_intent(spec, **intent_kwargs(tmp_path))  # no paired_band settings
    write_check(tmp_path, plan_params(spec), rate=0.061)
    with pytest.raises(R.StrictRunError, match="joint rate above"):
        prepare_paired(tmp_path, spec)
    write_check(tmp_path, plan_params(spec), reps=10000)
    with pytest.raises(R.StrictRunError, match="reps"):
        prepare_paired(tmp_path, spec)
    write_check(tmp_path, plan_params(spec), thetas=(15.0, 30.0, 45.0, 60.0))
    with pytest.raises(R.StrictRunError, match="0.67"):
        prepare_paired(tmp_path, spec)
    # a check run with another plan (here: alpha 0.10) does not pre-register this one
    write_check(tmp_path, plan_params(spec, dict(RATIFIED, band_alpha=0.10)))
    with pytest.raises(R.StrictRunError, match="frozen plan parameters"):
        prepare_paired(tmp_path, spec)
    write_check(tmp_path, plan_params(spec))
    (tmp_path / "screen_pool.json").write_text("{}")
    with pytest.raises(R.StrictRunError, match="pool data changed"):
        prepare_paired(tmp_path, spec)


def test_paired_check_drift_after_prepare_is_refused(tmp_path):
    spec = make_paired_spec(FakeWorld(), tmp_path)
    intent = R.read_json(prepare_paired(tmp_path, spec))
    R.validate_intent(intent, spec)
    write_check(tmp_path, plan_params(spec), rate=0.03)  # still passing, but other bytes
    with pytest.raises(R.StrictRunError, match="paired_band_check drift"):
        R.validate_intent(intent, spec)


def test_paired_spec_shape_is_enforced(tmp_path):
    spec = make_paired_spec(FakeWorld(), tmp_path)
    with pytest.raises(R.StrictRunError, match="malformed paired band"):
        R.validate_spec(R.StudySpec(**{**spec.__dict__, "bands": R.survival_bands(MIXES)}))
    with pytest.raises(R.StrictRunError, match="malformed paired band"):
        R.validate_spec(
            R.StudySpec(
                **{**spec.__dict__, "bands": ({"metric": "mass_integral", "mix": "frozen"},)}
            )
        )
    with pytest.raises(R.StrictRunError, match="two paired bands"):
        R.validate_spec(
            R.StudySpec(**{**spec.__dict__, "bands": R.paired_survival_bands(("frozen", "frozen"))})
        )
    with pytest.raises(R.StrictRunError, match="band_policy"):
        R.validate_spec(R.StudySpec(**{**spec.__dict__, "band_policy": "paired"}))
    legacy = make_spec(FakeWorld(), tmp_path)
    with pytest.raises(R.StrictRunError, match="only used under"):
        R.validate_spec(R.StudySpec(**{**legacy.__dict__, "paired_band_check_path": "x.json"}))
    with pytest.raises(R.StrictRunError, match="band_policy paired"):
        R.build_intent(legacy, **intent_kwargs(tmp_path, paired_band=RATIFIED))


def test_cli_plan_document_is_the_simulator_input(tmp_path, monkeypatch):
    spec = make_paired_spec(FakeWorld(), tmp_path)
    monkeypatch.setattr(R, "resolve_spec", lambda ref: spec)
    out = tmp_path / "plan.json"
    code = R.main(
        [
            "plan",
            "--spec",
            "x:y",
            "--out",
            str(out),
            "--n-max",
            str(N_MAX),
            "--mde",
            "30",
            "--paired-band",
            "0.05,0.05,pointwise,0.30",
        ]
    )
    assert code == 0
    document = R.read_json(out)
    assert document["plan_parameters"] == plan_params(spec)
    assert document["plan"] == R.plan_from_parameters(plan_params(spec)).as_dict()
    intent = R.read_json(prepare_paired(tmp_path, spec))
    assert intent["plan_parameters"] == document["plan_parameters"]
    assert intent["plan"] == document["plan"]
    assert R.parse_paired_band("0.05,0.1,rci_obf,none") == {
        "band_ni_margin": 0.05,
        "band_alpha": 0.1,
        "band_bound": "rci_obf",
        "band_floor": None,
    }


# ---------------------------------------------------------------- legacy policy unchanged

# Computed on main (f3af8c2) before template v2 existed.
GOLDEN_V1_PARAMS_SHA = "244dd17d1f9ea58f9894c3219e9429f729c823c9f5d2945881d78cebac5e01a7"
GOLDEN_V1_PLAN_SHA = "0c869cb7e759c749ffc98e19e799b286d43ed0f8fe38ec4962a9c070329b1220"
GOLDEN_V1_EXAMPLE_DESCRIPTOR_SHA = (
    "5a13e24c5ba950c6af7a06ebbde5f6ba132a5a4769d0dd1a031a110252d0232b"
)
V1_INTENT_KEYS = {
    "schema_version",
    "template_version",
    "study_id",
    "authority",
    "dry_run",
    "created_utc",
    "authorization_quote",
    "deadline_utc",
    "spec_ref",
    "spec",
    "arms",
    "method",
    "plan_parameters",
    "plan",
    "plan_sha256",
    "futility_action",
    "interleaving",
    "banks",
    "banks_check",
    "calibration_rule",
    "skew_check",
    "preregistration",
    "sizing",
    "caps",
    "source_closure",
    "allow_dirty",
    "template_readme_sha256",
    "output_root",
    "repo",
    "python",
    "slot_lock_root",
    "ledger_path",
    "audit",
}
V1_RECEIPT_KEYS = {
    "schema_version",
    "study_id",
    "method",
    "look",
    "final_look",
    "n_per_mix",
    "units",
    "per_worker",
    "intent_sha256",
    "plan_sha256",
    "calibration_sha256",
    "skew_check_sha256",
    "prior_look_receipt_sha256",
    "segment_reports_sha256",
    "records_sha256",
    "deltas_digest",
    "bands_by_look",
    "bands_pass_by_look",
    "decision",
    "valid",
    "passes",
    "action",
    "futility_action",
    "sequential_decision",
}


def test_block_at_stop_hashes_and_intent_shape_are_unchanged(tmp_path):
    params = R.plan_parameters(n_max=N_MAX, mde=30.0, mixes=MIXES, scripted_mix="scripted")
    assert R.canonical_sha(params) == GOLDEN_V1_PARAMS_SHA
    assert R.canonical_sha(R.plan_from_parameters(params).as_dict()) == GOLDEN_V1_PLAN_SHA
    assert R.canonical_sha(example_spec.SPEC.descriptor()) == GOLDEN_V1_EXAMPLE_DESCRIPTOR_SHA
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    assert spec.band_policy == "block_at_stop" and spec.template_version == R.TEMPLATE_VERSION
    intent_path = R.prepare(R.build_intent(spec, **intent_kwargs(tmp_path)))
    intent = R.read_json(intent_path)
    assert set(intent) == V1_INTENT_KEYS
    assert intent["template_version"] == "sequential-strict-template/v1"
    assert intent["plan_parameters"] == params and intent["plan_sha256"] == GOLDEN_V1_PLAN_SHA
    assert "band_policy" not in intent["spec"] and "paired_band_check_path" not in intent["spec"]
    assert set(intent["preregistration"]) == {"protocol", "oc_report", "band_cost_report"}
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    [receipt] = receipts(intent_path)
    assert set(receipt) == V1_RECEIPT_KEYS
    calibration = R.read_json(output_of(intent_path) / "calibration.json")
    assert all({"lower", "upper", "reference_mean"} <= set(b) for b in calibration["bands"])
    report = A.run_audit(intent_path.parent)
    assert report["status"] == "PASS" and report["schema_version"] == "sequential-strict-audit/v1"
    assert "looks.paired_bands" not in {row["rule"] for row in report["checks"]}


def test_audit_rejects_paired_fields_smuggled_into_a_v1_receipt(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = R.prepare(R.build_intent(spec, **intent_kwargs(tmp_path)))
    execute(intent_path, spec)
    path = output_of(intent_path) / "looks" / "look-0.json"
    receipt = json.loads(path.read_text())
    receipt["band_policy"] = "paired_ni_at_stop"
    path.write_text(json.dumps(receipt))
    rows = next(
        r["detail"]
        for r in A.run_audit(intent_path.parent)["failures"]
        if r["rule"] == "looks.replay"
    )
    assert any("paired band fields on a block_at_stop receipt" in row for row in rows)
