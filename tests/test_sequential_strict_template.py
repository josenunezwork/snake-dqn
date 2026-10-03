"""Tests for research/sequential_strict_template (group-sequential strict runner + audit).

No test plays an episode or runs a numerical probe: ``tournament_eval.rollout``, the
subprocess executor, the subprocess skew probe and audit runners and ``supervise_children``
raise in every test.  Runs use a fake, deterministic episode function, the in-process
executor, a fake skew-probe output and the in-process audit.
"""

from __future__ import annotations

import ast
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.sequential_strict_template import example_spec
from research.sequential_strict_template import sequential_audit as A
from research.sequential_strict_template import sequential_runner as R
from src.evaluation.sequential_gate import (
    round_robin_plan,
    sequential_gate_plan,
    worker_look_counts,
)
from src.scripts.eval_stats import student_t_isf, student_t_sf

MIXES = ("frozen", "scripted", "mixed")
N_MAX = 40  # looks of 10, 20, 30, 40 worlds per mix


def _fatal(*args, **kwargs):
    raise AssertionError("tests must never launch a real rollout, probe or child process")


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    """Every path to real evaluation or a child process is fatal; AC power is faked."""
    from src.scripts import tournament_eval

    monkeypatch.setattr(tournament_eval, "rollout", _fatal)
    monkeypatch.setattr(R, "subprocess_skew_runner", _fatal)
    monkeypatch.setattr(R, "subprocess_audit_runner", _fatal)
    monkeypatch.setattr(R, "supervise_children", _fatal)
    monkeypatch.setattr(R.SubprocessExecutor, "run_segment", _fatal)
    monkeypatch.setattr(R, "on_ac_power", lambda: True)
    monkeypatch.setattr(R, "SLOT_LOCK_ROOT", tmp_path / "global-locks-must-not-be-used")


class SimulatedCrash(BaseException):
    """Stands in for the parent dying (SIGKILL / power loss): nothing catches it."""


class FakeWorld:
    """Deterministic episode function: per-world base mass, arm noise, candidate effect."""

    def __init__(self, effect=0.0, survival_shift=0.0, sd=40.0, crash_after=None):
        self.effect = effect if isinstance(effect, dict) else {m: effect for m in MIXES}
        self.survival_shift = survival_shift
        self.sd = sd
        self.crash_after = crash_after
        self.final_calls = 0

    def __call__(self, episode, row, context):
        assert row["world_seed"] == episode["world_seed"]
        if episode["phase"] == "final":
            self.final_calls += 1
            if self.crash_after is not None and self.final_calls > self.crash_after:
                raise SimulatedCrash(f"parent died at final episode {self.final_calls}")
        world = random.Random(f"{episode['world_seed']}|{episode['mix']}")
        arm = random.Random(f"{episode['world_seed']}|{episode['mix']}|{episode['arm']}")
        candidate = episode["arm"] == "candidate"
        mass = 1000.0 + world.gauss(0, 200) + arm.gauss(0, self.sd)
        survival = 0.8 + world.gauss(0, 0.004) + arm.gauss(0, 0.002)
        if candidate:
            mass += self.effect[episode["mix"]]
            survival += self.survival_shift
        return {"mass_integral": mass, "survival_fraction": survival}


def make_spec(runner) -> R.StudySpec:
    return R.StudySpec(
        study_id="test-sequential-strict",
        schema="test-sequential-strict/v1",
        namespaces={"calibration": "test-seq-dev-v1", "final": "test-seq-final-v1"},
        arm_identities=lambda: {"incumbent": {"method": "inc"}, "candidate": {"method": "cand"}},
        episode_runner=runner,
        mixes=MIXES,
        closure_roots=("research/sequential_strict_template",),
    )


def fake_skew(rates=None):
    """A skew runner that writes a probe-shaped output with the given any-look rates."""
    rates = rates or {}

    def runner(ctx, mix, deltas_path, out_path, wall_seconds):
        efficacy, ni = rates.get(mix, (0.015, 0.05))
        reps = ctx.intent["skew_check"]["reps"]
        payload = {
            "part": "resample",
            "cpu_seconds": 0.0,
            "result": {
                "n_saved": len(json.loads(Path(deltas_path).read_text())),
                "look_sizes": ctx.intent["plan"]["look_sizes"],
                "sample_skewness": 0.0,
                "efficacy_null": {"reps": reps, "any_look": efficacy},
                "ni_null": {"reps": reps, "any_look": ni},
                "passes": efficacy <= 0.02 and ni <= 0.06,
            },
        }
        Path(out_path).write_text(json.dumps(payload))
        return {
            "cause": None,
            "elapsed_seconds": 0.0,
            "wall_seconds": wall_seconds,
            "children": [
                {
                    "command": ["fake-skew-probe", mix],
                    "returncode": 0,
                    "termination": "natural_exit",
                    "confirmed_exit": True,
                    "peak_group_rss_bytes": 0,
                }
            ],
        }

    return runner


def prepare_run(tmp_path, spec, **overrides):
    skew_input = tmp_path / "pilot-deltas.json"
    rng = random.Random(7)
    skew_input.write_text(json.dumps({m: [rng.gauss(100, 140) for _ in range(30)] for m in MIXES}))
    (tmp_path / "locks").mkdir(exist_ok=True)
    kwargs = dict(
        spec_ref="tests.test_sequential_strict_template:unused",
        out_root=tmp_path / "run-v1",
        n_max=N_MAX,
        mde=30.0,
        n_calibration=4,
        skew_input=skew_input,
        deadline=datetime.now(timezone.utc) + timedelta(days=2),
        authorization_quote="test only",
        skew_reps=1000,
        slot_lock_root=tmp_path / "locks",
        allow_dirty=True,
    )
    kwargs.update(overrides)
    return R.prepare(R.build_intent(spec, **kwargs))


def execute(intent_path, spec, skew=None, resume=False):
    return R.run(
        intent_path,
        spec=spec,
        executor=R.InProcessExecutor(),
        skew_runner=skew or fake_skew(),
        audit_runner=R.in_process_audit_runner,
        resume=resume,
    )


def output_of(intent_path: Path) -> Path:
    return Path(intent_path).parent / "output"


def receipts(intent_path: Path):
    looks = output_of(intent_path) / "looks"
    return [R.read_json(looks / f"look-{k}.json") for k in range(len(list(looks.glob("*.json"))))]


def final_records(intent_path: Path):
    return sorted(p.stem for p in (output_of(intent_path) / "final" / "records").glob("*.json"))


# ---------------------------------------------------------------- scenarios


def test_large_effect_stops_at_look_one_with_audit_pass(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0))
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "STRICT_PASS", (
        closeout["failure"],
        closeout["deadline_stop"],
        closeout["audit_failures"],
    )
    assert closeout["stop_look"] == 0 and closeout["stop_decision"] == "STOP_PASS"
    assert closeout["worlds_per_mix_used"] == 10 and closeout["audit_passed"] is True
    [receipt] = receipts(intent_path)
    assert receipt["action"] == "stop" and receipt["per_worker"] == [15, 15]
    assert len(final_records(intent_path)) == 2 * 30  # both arms of the 30-unit prefix
    assert (output_of(intent_path) / "receipt.json").is_file()
    assert A.run_audit(intent_path.parent)["status"] == "PASS"  # post-hoc, with closeout


def test_futility_stop_is_followed(tmp_path):
    spec = make_spec(FakeWorld(effect=-150.0))
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "STRICT_FAIL", (
        closeout["failure"],
        closeout["deadline_stop"],
        closeout["audit_failures"],
    )
    assert closeout["stop_decision"] == "STOP_FUTILE"
    assert closeout["stop_look"] < 3
    assert receipts(intent_path)[-1]["action"] == "stop"
    assert not (output_of(intent_path) / "receipt.json").exists()


def test_runs_to_final_look_when_no_stop(tmp_path):
    # null effect, MDE so large that conditional power never signals futility
    spec = make_spec(FakeWorld(effect=0.0))
    intent_path = prepare_run(tmp_path, spec, mde=2000.0)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "STRICT_FAIL", (
        closeout["failure"],
        closeout["deadline_stop"],
        closeout["audit_failures"],
    )
    assert closeout["stop_look"] == 3 and closeout["stop_decision"] == "FINAL_FAIL"
    assert [r["decision"] for r in receipts(intent_path)][:3] == ["CONTINUE"] * 3
    assert len(final_records(intent_path)) == 2 * 3 * N_MAX


def test_overridable_futility_continue_is_disclosed_and_valid(tmp_path):
    spec = make_spec(FakeWorld(effect=-150.0))
    intent_path = prepare_run(
        tmp_path, spec, futility_policy="overridable", futility_action="continue"
    )
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "STRICT_FAIL" and closeout["stop_look"] == 3, (
        closeout["failure"],
        closeout["deadline_stop"],
        closeout["audit_failures"],
    )
    looks = receipts(intent_path)
    assert looks[0]["decision"] == "STOP_FUTILE" and looks[0]["action"] == "continue"
    assert looks[-1]["valid"] is True
    assert looks[-1]["sequential_decision"]["futility_overrides"]


def test_futility_override_requires_overridable_policy(tmp_path):
    spec = make_spec(FakeWorld())
    with pytest.raises(R.StrictRunError, match="overridable"):
        prepare_run(tmp_path, spec, futility_action="continue")


def test_band_failure_at_the_qualifying_look_stops_fail_bands(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0, survival_shift=-0.2))
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "STRICT_FAIL", (
        closeout["failure"],
        closeout["deadline_stop"],
        closeout["audit_failures"],
    )
    assert closeout["stop_look"] == 0 and closeout["stop_decision"] == "STOP_FAIL_BANDS"
    [receipt] = receipts(intent_path)
    assert receipt["bands_pass_by_look"] == [False]
    assert len(final_records(intent_path)) == 60


def test_skew_check_failure_plays_no_final_world(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0))
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec, skew=fake_skew({"frozen": (0.03, 0.05)}))
    assert closeout["outcome"] == "SKEW_CHECK_FAILED", (
        closeout["failure"],
        closeout["deadline_stop"],
        closeout["audit_failures"],
    )
    assert closeout["skew_check_passed"] is False and closeout["audit_passed"] is True
    skew = R.read_json(output_of(intent_path) / "skew_check.json")
    assert skew["per_mix"]["frozen"]["passes"] is False
    assert skew["remedy_on_fail"] == "stop_and_escalate"
    assert final_records(intent_path) == []
    with pytest.raises(R.StrictRunError, match="skew check failed"):
        R.prior_gate(R.read_json(intent_path), output_of(intent_path), "final", 0)


def test_scripted_ni_skew_rate_is_judged_only_on_the_scripted_mix(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0))
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec, skew=fake_skew({"frozen": (0.01, 0.2)}))
    assert closeout["outcome"] == "STRICT_PASS"
    tmp2 = tmp_path / "second"
    tmp2.mkdir()
    intent_path = prepare_run(tmp2, spec)
    closeout = execute(intent_path, spec, skew=fake_skew({"scripted": (0.01, 0.07)}))
    assert closeout["outcome"] == "SKEW_CHECK_FAILED"


def test_crash_and_resume_keeps_write_once_receipts(tmp_path):
    crashing = FakeWorld(effect=0.0, crash_after=70)  # look 0 = 60 episodes, crash in look 1
    spec = make_spec(crashing)
    intent_path = prepare_run(tmp_path, spec, mde=2000.0)
    with pytest.raises(SimulatedCrash):
        execute(intent_path, spec)
    output = output_of(intent_path)
    assert not (output / "closeout.json").exists()
    look0 = output / "looks" / "look-0.json"
    before = R.sha256_file(look0)
    assert len(final_records(intent_path)) == 70  # 60 + 10 of look 1, nothing beyond
    with pytest.raises(FileExistsError):
        R.write_once(look0, {"tampered": True})
    with pytest.raises(R.StrictRunError, match="create-only"):
        execute(intent_path, spec)  # a fresh run never reuses an output root

    resumed = make_spec(FakeWorld(effect=0.0))
    closeout = execute(intent_path, resumed, resume=True)
    assert closeout["outcome"] == "STRICT_FAIL", (
        closeout["failure"],
        closeout["deadline_stop"],
        closeout["audit_failures"],
    )
    assert closeout["resumes"] == 1 and closeout["stop_look"] == 3
    assert R.sha256_file(look0) == before
    attempts = R.attempt_dirs(output, "final", 1)
    assert len(attempts) == 2
    report = R.read_json(attempts[-1] / "shard-0" / "report.json")
    assert len(report["adopted_episode_ids"]) == 10
    assert A.run_audit(intent_path.parent)["status"] == "PASS"
    with pytest.raises(R.StrictRunError, match="closed out"):
        execute(intent_path, resumed, resume=True)


def test_resume_refused_when_not_pre_registered(tmp_path):
    spec = make_spec(FakeWorld(effect=0.0, crash_after=5))
    intent_path = prepare_run(tmp_path, spec, resume_authorized=False)
    with pytest.raises(SimulatedCrash):
        execute(intent_path, spec)
    with pytest.raises(R.StrictRunError, match="pre-registered"):
        execute(intent_path, make_spec(FakeWorld()), resume=True)


def test_resume_rejects_a_receipt_that_does_not_replay(tmp_path):
    spec = make_spec(FakeWorld(effect=0.0, crash_after=70))
    intent_path = prepare_run(tmp_path, spec, mde=2000.0)
    with pytest.raises(SimulatedCrash):
        execute(intent_path, spec)
    # alter a look-0 record: the resumed parent must refuse to build on it
    record = output_of(intent_path) / "final" / "records" / "final-candidate-frozen-w00000.json"
    entry = json.loads(record.read_text())
    entry["record"]["mass_integral"] += 5000.0
    record.write_text(json.dumps(entry))
    closeout = execute(intent_path, make_spec(FakeWorld(effect=0.0)), resume=True)
    assert closeout["outcome"] == "INVALID_STOP"
    assert "record bytes changed" in closeout["failure"]


def test_battery_refuses_before_any_record(tmp_path, monkeypatch):
    spec = make_spec(FakeWorld())
    intent_path = prepare_run(tmp_path, spec)
    monkeypatch.setattr(R, "on_ac_power", lambda: False)
    with pytest.raises(R.StrictRunError, match="battery"):
        execute(intent_path, spec)
    assert not output_of(intent_path).exists()


def test_worker_refuses_to_run_past_a_stop(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0))
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    with pytest.raises(R.StrictRunError, match="no unit beyond it may run"):
        R.prior_gate(R.read_json(intent_path), output_of(intent_path), "final", 1)


def test_validate_intent_rejects_a_plan_that_does_not_recompute(tmp_path):
    spec = make_spec(FakeWorld())
    intent = R.read_json(prepare_run(tmp_path, spec))
    intent["plan"]["efficacy_boundaries"][0] -= 0.5
    with pytest.raises(R.StrictRunError, match="frozen plan"):
        R.validate_intent(intent, spec)


def test_intent_freezes_plan_interleaving_and_skew_preregistration(tmp_path):
    spec = make_spec(FakeWorld())
    intent = R.read_json(prepare_run(tmp_path, spec))
    plan = sequential_gate_plan(N_MAX, mde=30.0)
    assert intent["plan"] == plan.as_dict()
    assert intent["plan_sha256"] == R.canonical_sha(plan.as_dict())
    for key in ("look_sizes", "spending", "futility_policy", "band_policy", "band_margin_z"):
        assert key in intent["plan"]
    assert intent["interleaving"]["worker_look_counts"] == worker_look_counts(plan.look_sizes, 3, 2)
    skew = intent["skew_check"]
    assert skew["limits"] == {
        "efficacy_any_look": pytest.approx(0.02),
        "ni_any_look_scripted": 0.06,
    }
    assert skew["remedy_on_fail"] == "stop_and_escalate" and skew["probe_part"] == "resample"


def test_segments_partition_the_round_robin_plan_at_each_look(tmp_path):
    spec = make_spec(FakeWorld())
    intent = R.read_json(prepare_run(tmp_path, spec))
    plan_units = round_robin_plan(N_MAX, MIXES, 2)
    counts = intent["interleaving"]["worker_look_counts"]
    for look, row in enumerate(counts):
        prefix = {u["unit"] for u in plan_units[: row["units"]]}
        ran = {
            u["unit"]
            for j in range(look + 1)
            for w in range(2)
            for u in R.segment_units(intent, "final", j, w)
        }
        assert ran == prefix
        for w in range(2):
            assert sum(len(R.segment_units(intent, "final", j, w)) for j in range(look + 1)) == (
                row["per_worker"][w]
            )


# ---------------------------------------------------------------- audit


def test_audit_detects_an_extra_record_past_the_stop(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0))
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    records = output_of(intent_path) / "final" / "records"
    entry = json.loads((records / "final-candidate-frozen-w00000.json").read_text())
    intent = R.read_json(intent_path)
    unit = R.phase_units(intent, "final")[30]  # first unit of look 1
    entry.update(
        {
            "episode_id": R.episode_id("final", "candidate", unit["mix"], unit["world_index"]),
            "mix": unit["mix"],
            "world_index": unit["world_index"],
            "world_seed": unit["world_seed"],
            "unit": unit["unit"],
            "worker": unit["worker"],
            "look": 1,
        }
    )
    (records / f"{entry['episode_id']}.json").write_text(json.dumps(entry))
    report = A.run_audit(intent_path.parent)
    failed = {row["rule"] for row in report["failures"]}
    assert report["status"] == "FAIL"
    assert "final.no_record_beyond_stop" in failed and "final.prefix_exact_at_stop" in failed


def test_audit_detects_an_altered_delta(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0))
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    record = output_of(intent_path) / "final" / "records" / "final-incumbent-mixed-w00003.json"
    entry = json.loads(record.read_text())
    entry["record"]["mass_integral"] -= 1.0
    record.write_text(json.dumps(entry))
    report = A.run_audit(intent_path.parent)
    assert report["status"] == "FAIL"
    details = next(r for r in report["failures"] if r["rule"] == "looks.replay")["detail"]
    assert any("deltas_digest" in row for row in details)
    assert any("record bytes" in row for row in details)


def test_audit_detects_an_altered_plan(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0))
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    intent = json.loads(intent_path.read_text())
    intent["plan"]["efficacy_boundaries"][0] -= 0.01
    intent_path.write_text(json.dumps(intent))
    report = A.run_audit(intent_path.parent)
    failed = {row["rule"] for row in report["failures"]}
    assert {"plan.efficacy_boundaries", "plan.sha256", "intent.sha256_bound"} <= failed


def test_audit_detects_a_relabelled_decision(tmp_path):
    spec = make_spec(FakeWorld(effect=-150.0))
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    claim_path = output_of(intent_path) / "producer-outcome.json"
    claim = json.loads(claim_path.read_text())
    claim["outcome"] = "STRICT_PASS"
    claim_path.write_text(json.dumps(claim))
    failed = {row["rule"] for row in A.run_audit(intent_path.parent)["failures"]}
    assert "outcome.producer_claim" in failed


def test_audit_boundaries_match_gsdesign_and_the_producer_plan():
    published = [4.3326, 2.9631, 2.3590, 2.0141]
    mine = A.obf_boundaries((0.25, 0.5, 0.75, 1.0), 0.025)
    assert all(abs(a - b) <= 2e-4 for a, b in zip(mine, published))
    plan = sequential_gate_plan(243, mde=30.0)
    for alpha, theirs in (
        (plan.efficacy_alpha_per_mix, plan.efficacy_boundaries),
        (plan.ni_alpha, plan.ni_boundaries),
    ):
        recomputed = A.obf_boundaries(tuple(plan.fractions), alpha)
        assert all(abs(a - b) <= 2e-4 for a, b in zip(recomputed, theirs))


def test_audit_t_distribution_matches_eval_stats():
    for t, df in ((0.3, 9), (2.1, 29), (8.0, 60), (25.0, 9), (-1.5, 15)):
        assert A.t_sf(t, df) == pytest.approx(student_t_sf(t, df), rel=1e-9, abs=1e-15)
    for p, df in ((1.77e-6, 60), (0.0149, 242), (0.04, 9)):
        assert A.t_isf(p, df) == pytest.approx(student_t_isf(p, df), rel=1e-9)


def test_audit_is_stdlib_only():
    tree = ast.parse(Path(A.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    imported.discard("__future__")
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)


def test_example_spec_never_plays_a_game():
    spec = R.resolve_spec("research.sequential_strict_template.example_spec:SPEC")
    assert spec is example_spec.SPEC
    with pytest.raises(NotImplementedError):
        spec.episode_runner({}, {}, None)


def test_prepare_refuses_a_plan_the_skew_probe_cannot_check(tmp_path):
    spec = make_spec(FakeWorld())
    with pytest.raises(R.StrictRunError, match="skew_probe.py assumes"):
        prepare_run(tmp_path, spec, family_alpha=0.025)
