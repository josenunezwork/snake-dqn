"""Tests for research/sequential_strict_template (group-sequential strict runner + audit).

No test plays an episode or runs a numerical probe: ``tournament_eval.rollout``, the
subprocess executor, the subprocess skew-probe and audit runners and ``supervise_children``
raise in every test (one test drives the real supervisor over a ``sleep`` child to check the
battery poll).  Runs are ``dry_run`` intents with a fake, deterministic episode function,
the in-process executor, a fake skew-probe output and the in-process audit.
"""

from __future__ import annotations

import ast
import json
import os
import random
import signal
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
REAL_SUPERVISE = R.supervise_children
REAL_SLOT_LOCK_ROOT = R.SLOT_LOCK_ROOT  # captured before the guard fixture patches it


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
    monkeypatch.setattr(R, "LEDGER_PATH", tmp_path / "global-ledger-must-not-be-used.jsonl")


class SimulatedCrash(BaseException):
    """Stands in for the parent dying (SIGKILL / power loss): nothing catches it."""


class FakeWorld:
    """Deterministic episode function: per-world base mass, arm noise, candidate effect."""

    def __init__(self, effect=0.0, survival_shift=0.0, sd=40.0, crash_after=None, raise_at=None):
        self.effect = effect if isinstance(effect, dict) else {m: effect for m in MIXES}
        self.survival_shift = survival_shift
        self.sd = sd
        self.crash_after = crash_after
        self.raise_at = raise_at  # (final episode number, callable raising/signalling)
        self.final_calls = 0

    def __call__(self, episode, row, context):
        assert row["world_seed"] == episode["world_seed"]
        if episode["phase"] == "final":
            self.final_calls += 1
            if self.crash_after is not None and self.final_calls > self.crash_after:
                raise SimulatedCrash(f"parent died at final episode {self.final_calls}")
            if self.raise_at is not None and self.final_calls == self.raise_at[0]:
                self.raise_at[1]()
        world = random.Random(f"{episode['world_seed']}|{episode['mix']}")
        arm = random.Random(f"{episode['world_seed']}|{episode['mix']}|{episode['arm']}")
        candidate = episode["arm"] == "candidate"
        mass = 1000.0 + world.gauss(0, 200) + arm.gauss(0, self.sd)
        survival = 0.8 + world.gauss(0, 0.004) + arm.gauss(0, 0.002)
        if candidate:
            mass += self.effect[episode["mix"]]
            survival += self.survival_shift
        return {"mass_integral": mass, "survival_fraction": survival}


def make_spec(runner, tmp_path) -> R.StudySpec:
    docs = tmp_path / "prereg"
    docs.mkdir(exist_ok=True)
    for name in ("protocol.md", "oc.json", "band_cost.json"):
        (docs / name).write_text(f"{name}: test pre-registration document\n")
    return R.StudySpec(
        study_id="test-sequential-strict",
        schema="test-sequential-strict/v1",
        namespaces={"calibration": "test-seq-dev-v1", "final": "test-seq-final-v1"},
        arm_identities=lambda: {"incumbent": {"method": "inc"}, "candidate": {"method": "cand"}},
        episode_runner=runner,
        excluded_seeds=lambda: {"earlier-screen": R.seed_bank("test-seq-screen-v1", 100)},
        protocol_path=str(docs / "protocol.md"),
        oc_report_path=str(docs / "oc.json"),
        band_cost_report_path=str(docs / "band_cost.json"),
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


def intent_kwargs(tmp_path, **overrides):
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
        ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        allow_dirty=True,
    )
    kwargs.update(overrides)
    return kwargs


def prepare_run(tmp_path, spec, **overrides):
    return R.prepare(R.build_intent(spec, **intent_kwargs(tmp_path, **overrides)))


def execute(intent_path, spec, skew=None, executor=None):
    return R.run(
        intent_path,
        spec=spec,
        executor=executor or R.InProcessExecutor(),
        skew_runner=skew or fake_skew(),
        audit_runner=R.in_process_audit_runner,
    )


def why(closeout):
    return closeout["failure"], closeout["deadline_stop"], closeout["audit_failures"]


def output_of(intent_path: Path) -> Path:
    return Path(intent_path).parent / "output"


def receipts(intent_path: Path):
    looks = output_of(intent_path) / "looks"
    return [R.read_json(looks / f"look-{k}.json") for k in range(len(list(looks.glob("*.json"))))]


def final_records(intent_path: Path):
    return sorted(p.stem for p in (output_of(intent_path) / "final" / "records").glob("*.json"))


# ---------------------------------------------------------------- scenarios


def test_large_effect_stops_at_look_one_dry_run_never_passes_strict(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    assert closeout["stop_look"] == 0 and closeout["stop_decision"] == "STOP_PASS"
    assert closeout["worlds_per_mix_used"] == 10 and closeout["audit_passed"] is True
    assert closeout["provenance"]["executor"] == "in-process"
    assert closeout["provenance"]["skew_runner"].startswith("injected:")
    started = R.read_json(output_of(intent_path) / "started.json")
    assert started["provenance"]["dry_run"] is True and started["provenance"]["allow_dirty"]
    [receipt] = receipts(intent_path)
    assert receipt["action"] == "stop" and receipt["per_worker"] == [15, 15]
    assert len(final_records(intent_path)) == 2 * 30  # both arms of the 30-unit prefix
    assert not (output_of(intent_path) / "receipt.json").exists()
    assert A.run_audit(intent_path.parent)["status"] == "PASS"  # post-hoc, with closeout


def test_futility_stop_is_followed(tmp_path):
    spec = make_spec(FakeWorld(effect=-150.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", why(closeout)
    assert closeout["stop_decision"] == "STOP_FUTILE" and closeout["stop_look"] < 3
    assert receipts(intent_path)[-1]["action"] == "stop"


def test_runs_to_final_look_when_no_stop(tmp_path):
    # null effect, MDE so large that conditional power never signals futility
    spec = make_spec(FakeWorld(effect=0.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec, mde=2000.0)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", why(closeout)
    assert closeout["stop_look"] == 3 and closeout["stop_decision"] == "FINAL_FAIL"
    assert [r["decision"] for r in receipts(intent_path)][:3] == ["CONTINUE"] * 3
    assert len(final_records(intent_path)) == 2 * 3 * N_MAX
    assert A.run_audit(intent_path.parent)["status"] == "PASS"


def test_overridable_futility_continue_is_disclosed_and_valid(tmp_path):
    spec = make_spec(FakeWorld(effect=-150.0), tmp_path)
    intent_path = prepare_run(
        tmp_path, spec, futility_policy="overridable", futility_action="continue"
    )
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL" and closeout["stop_look"] == 3, why(closeout)
    looks = receipts(intent_path)
    assert looks[0]["decision"] == "STOP_FUTILE" and looks[0]["action"] == "continue"
    assert looks[-1]["valid"] is True
    assert looks[-1]["sequential_decision"]["futility_overrides"]


def test_futility_override_requires_overridable_policy(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    with pytest.raises(R.StrictRunError, match="overridable"):
        prepare_run(tmp_path, spec, futility_action="continue")


def test_band_failure_at_the_qualifying_look_stops_fail_bands(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0, survival_shift=-0.2), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", why(closeout)
    assert closeout["stop_look"] == 0 and closeout["stop_decision"] == "STOP_FAIL_BANDS"
    [receipt] = receipts(intent_path)
    assert receipt["bands_pass_by_look"] == [False]
    assert len(final_records(intent_path)) == 60


def test_skew_check_failure_plays_no_final_world(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec, skew=fake_skew({"frozen": (0.03, 0.05)}))
    assert closeout["outcome"] == "DRY_RUN_SKEW_CHECK_FAILED", why(closeout)
    assert closeout["skew_check_passed"] is False and closeout["audit_passed"] is True
    skew = R.read_json(output_of(intent_path) / "skew_check.json")
    assert skew["per_mix"]["frozen"]["passes"] is False
    assert skew["remedy_on_fail"] == "stop_and_escalate"
    assert final_records(intent_path) == []
    with pytest.raises(R.StrictRunError, match="skew check failed"):
        R.prior_gate(R.read_json(intent_path), output_of(intent_path), "final", 0)


def test_scripted_ni_skew_rate_is_judged_only_on_the_scripted_mix(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec, skew=fake_skew({"frozen": (0.01, 0.2)}))
    assert closeout["outcome"] == "DRY_RUN_PASS"
    second = tmp_path / "second"
    second.mkdir()
    intent_path = prepare_run(second, spec)
    closeout = execute(intent_path, spec, skew=fake_skew({"scripted": (0.01, 0.07)}))
    assert closeout["outcome"] == "DRY_RUN_SKEW_CHECK_FAILED"


# ---------------------------------------------------------------- no resume, signals, guards


def test_crashed_run_is_abandoned_and_can_never_be_resumed(tmp_path):
    spec = make_spec(FakeWorld(effect=0.0, crash_after=70), tmp_path)
    intent_path = prepare_run(tmp_path, spec, mde=2000.0)
    with pytest.raises(SimulatedCrash):
        execute(intent_path, spec)
    output = output_of(intent_path)
    assert not (output / "closeout.json").exists()
    assert len(final_records(intent_path)) == 70  # 60 + 10 of look 1, nothing beyond
    assert R.run_status(intent_path.parent)["state"] == "ABANDONED"
    assert R.main(["status", "--intent", str(intent_path)]) == 0
    with pytest.raises(R.StrictRunError, match="ABANDONED"):
        execute(intent_path, make_spec(FakeWorld(effect=0.0), tmp_path))
    with pytest.raises(R.StrictRunError, match="already started a run"):
        R.build_intent(spec, **intent_kwargs(tmp_path, mde=2000.0))
    assert A.run_audit(intent_path.parent)["status"] == "UNCLOSED"
    [entry] = R.ledger_status(tmp_path / "ledger.jsonl")
    assert entry["state"] == "ABANDONED" and entry["final_namespace"] == "test-seq-final-v1"
    with pytest.raises(FileExistsError):
        R.write_once(output / "looks" / "look-0.json", {"tampered": True})
    assert not hasattr(R, "resume") and "resume" not in R.parse_args.__code__.co_consts


def test_closed_run_is_never_rerun(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    assert R.run_status(intent_path.parent) == {"state": "CLOSED", "outcome": "DRY_RUN_PASS"}
    with pytest.raises(R.StrictRunError, match="CLOSED"):
        execute(intent_path, spec)


def test_keyboard_interrupt_inside_a_segment_closes_out_invalid(tmp_path):
    def interrupt():
        raise KeyboardInterrupt

    spec = make_spec(FakeWorld(effect=0.0, raise_at=(5, interrupt)), tmp_path)
    intent_path = prepare_run(tmp_path, spec, mde=2000.0)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "INVALID_STOP" and "KeyboardInterrupt" in closeout["failure"]


def test_sigint_is_handled_like_sigterm(tmp_path):
    spec = make_spec(
        FakeWorld(effect=0.0, raise_at=(5, lambda: os.kill(os.getpid(), signal.SIGINT))),
        tmp_path,
    )
    intent_path = prepare_run(tmp_path, spec, mde=2000.0)
    previous = signal.getsignal(signal.SIGINT)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "INVALID_STOP"
    assert "Interrupted: SIGINT" in closeout["failure"]
    assert signal.getsignal(signal.SIGINT) is previous  # restored after closeout


def test_keyboard_interrupt_outside_segments_closes_out_invalid(tmp_path):
    def interrupting_skew(ctx, mix, deltas_path, out_path, wall_seconds):
        raise KeyboardInterrupt

    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec, skew=interrupting_skew)
    assert closeout["outcome"] == "INVALID_STOP" and "KeyboardInterrupt" in closeout["failure"]
    assert final_records(intent_path) == []


def test_worker_exits_without_writing_when_its_parent_is_gone(tmp_path):
    class OrphanedFinal(R.InProcessExecutor):
        def run_segment(self, ctx, phase, look, wall_seconds, worker_deadline):
            if phase != "final":
                return super().run_segment(ctx, phase, look, wall_seconds, worker_deadline)
            code = R.worker_body(
                ctx.intent, ctx.intent_sha, ctx.spec, phase, look, 0, worker_deadline, -1
            )
            child = {
                "command": ["orphan"],
                "returncode": code,
                "termination": "natural_exit",
                "confirmed_exit": True,
                "peak_group_rss_bytes": 0,
            }
            return {"cause": None, "elapsed_seconds": 0.0, "children": [child]}

    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    closeout = execute(intent_path, spec, executor=OrphanedFinal())
    assert closeout["outcome"] == "INVALID_STOP"
    assert final_records(intent_path) == []
    shard = output_of(intent_path) / "final" / "segments" / "look-0" / "shard-0"
    assert not (shard / "started.json").exists() and not (shard / "report.json").exists()


def test_supervisor_polls_ac_power_and_stops_on_battery(tmp_path):
    result = REAL_SUPERVISE(
        [[sys.executable, "-c", "import time; time.sleep(30)"]],
        cwd=tmp_path,
        env=dict(os.environ),
        logs=[tmp_path / "child.log"],
        heartbeats=[None],
        wall_seconds=60,
        poll_seconds=0.05,
        grace_seconds=2,
        available_floor_bytes=0,
        power_check=lambda: False,
        power_poll_seconds=0.0,
    )
    assert result["cause"] == "on_battery"
    assert result["children"][0]["termination"] in ("terminated", "killed")
    assert not R.clean_supervision(result)


def test_battery_refuses_before_any_record(tmp_path, monkeypatch):
    spec = make_spec(FakeWorld(), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    monkeypatch.setattr(R, "on_ac_power", lambda: False)
    with pytest.raises(R.StrictRunError, match="battery"):
        execute(intent_path, spec)
    assert not output_of(intent_path).exists()


def test_battery_between_segments_stops_admitting(tmp_path, monkeypatch):
    calls = {"n": 0}

    def power():
        calls["n"] += 1
        return calls["n"] <= 3  # admission, calibration, skew check; then on battery

    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    monkeypatch.setattr(R, "on_ac_power", power)
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "INVALID_STOP" and "battery" in closeout["failure"]
    assert final_records(intent_path) == []


def test_final_stage_cap_is_charged_across_segments_and_cannot_reset(tmp_path):
    class Clocked(R.InProcessExecutor):
        def run_segment(self, ctx, phase, look, wall_seconds, worker_deadline):
            result = super().run_segment(ctx, phase, look, wall_seconds, worker_deadline)
            if phase == "final":  # pretend the segment and its analysis took 2000 s
                ctx.state["stage_started_mono"]["final"] -= 2000.0
            return result

    caps = dict(R.DEFAULT_CAPS, final=2000)
    spec = make_spec(FakeWorld(effect=0.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec, mde=2000.0, caps=caps)
    closeout = execute(intent_path, spec, executor=Clocked())
    assert closeout["outcome"] == "INCOMPLETE"
    assert "final stage cap exhausted" in closeout["deadline_stop"]
    assert len(receipts(intent_path)) == 1
    segment = R.read_json(output_of(intent_path) / "final" / "segments" / "look-0" / "segment.json")
    assert segment["stage_cap_seconds"] == 2000


@pytest.mark.parametrize("caps", [{"final": 0}, {"final": 45001}, {"audit": 1.5}])
def test_caps_must_be_positive_and_within_defaults(tmp_path, caps):
    spec = make_spec(FakeWorld(), tmp_path)
    with pytest.raises(R.StrictRunError, match="cap"):
        prepare_run(tmp_path, spec, caps=dict(R.DEFAULT_CAPS, **caps))


# ---------------------------------------------------------------- production vs dry run


def test_production_intent_requires_global_lock_root_and_clean_sources(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    with pytest.raises(R.StrictRunError, match="global slot lock root"):
        prepare_run(tmp_path, spec, dry_run=False, allow_dirty=False)
    with pytest.raises(R.StrictRunError, match="allow_dirty"):
        prepare_run(tmp_path, spec, dry_run=False, slot_lock_root=R.SLOT_LOCK_ROOT)


def test_production_intent_refuses_injected_executor_and_runners(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    intent = R.build_intent(spec, **intent_kwargs(tmp_path))
    intent.update(
        dry_run=False,
        allow_dirty=False,
        slot_lock_root=str(R.SLOT_LOCK_ROOT),
        ledger_path=str(R.LEDGER_PATH),
    )
    with pytest.raises(R.StrictRunError, match="global slot lock root"):
        R.validate_intent(dict(intent, slot_lock_root=str(tmp_path / "locks")), spec)
    intent_path = R.prepare(intent)
    with pytest.raises(R.StrictRunError, match="SubprocessExecutor only"):
        execute(intent_path, spec)
    with pytest.raises(R.StrictRunError, match="non-production run needs dry_run"):
        R.run(intent_path, spec=spec, skew_runner=fake_skew())
    assert not output_of(intent_path).exists()
    assert not R.LEDGER_PATH.exists()


def test_audit_fails_non_dry_run_provenance():
    production = {
        "dry_run": False,
        "allow_dirty": False,
        "source_closure": {"dirty": []},
        "slot_lock_root": A.SLOT_LOCK_ROOT,
        "ledger_path": A.LEDGER_PATH,
    }
    in_process = {
        "provenance": {
            "executor": "in-process",
            "skew_runner": "injected:tests.fake",
            "audit_runner": "injected:tests.fake",
            "allow_dirty": True,
            "dry_run": False,
        }
    }
    problems = A.provenance_problems(
        production, in_process, [{"executor": "in-process"}], None, None
    )
    assert len(problems) >= 4
    clean = {
        "provenance": {
            "executor": "subprocess",
            "skew_runner": "subprocess_skew_runner",
            "audit_runner": "subprocess_audit_runner",
            "allow_dirty": False,
            "dry_run": False,
        }
    }
    worker = {"children": [{"command": ["python", "runner.py", "worker"]}]}
    assert A.provenance_problems(production, clean, [worker], None, None) == []
    assert A.provenance_problems(dict(production, dry_run=True), in_process, [], None, None) == []


def test_validate_intent_rejects_a_plan_that_does_not_recompute(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    intent = R.read_json(prepare_run(tmp_path, spec))
    intent["plan"]["efficacy_boundaries"][0] -= 0.5
    with pytest.raises(R.StrictRunError, match="frozen plan"):
        R.validate_intent(intent, spec)


def test_preregistration_documents_are_required_and_frozen(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    with pytest.raises(R.StrictRunError, match="oc_report_path"):
        R.validate_spec(R.StudySpec(**{**spec.__dict__, "oc_report_path": ""}))
    intent = R.read_json(prepare_run(tmp_path, spec))
    assert set(intent["preregistration"]) == {"protocol", "oc_report", "band_cost_report"}
    Path(spec.band_cost_report_path).write_text("edited after prepare\n")
    with pytest.raises(R.StrictRunError, match="band_cost_report drift"):
        R.validate_intent(intent, spec)


def test_intent_freezes_plan_interleaving_and_skew_preregistration(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
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
    assert len(skew["deviations_from_amendment"]) == 2
    assert intent["caps"]["resume_authorized"] is False and "resume" not in intent


def test_prepare_refuses_a_plan_the_skew_probe_cannot_check(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    with pytest.raises(R.StrictRunError, match="skew_probe.py assumes"):
        prepare_run(tmp_path, spec, family_alpha=0.025)


def test_worker_refuses_to_run_past_a_stop(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    with pytest.raises(R.StrictRunError, match="no unit beyond it may run"):
        R.prior_gate(R.read_json(intent_path), output_of(intent_path), "final", 1)


def test_segments_partition_the_round_robin_plan_at_each_look(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    intent = R.read_json(prepare_run(tmp_path, spec))
    plan_units = round_robin_plan(N_MAX, MIXES, 2)
    for look, row in enumerate(intent["interleaving"]["worker_look_counts"]):
        prefix = {u["unit"] for u in plan_units[: row["units"]]}
        ran = {
            u["unit"]
            for j in range(look + 1)
            for w in range(2)
            for u in R.segment_units(intent, "final", j, w)
        }
        assert ran == prefix
        for w in range(2):
            count = sum(len(R.segment_units(intent, "final", j, w)) for j in range(look + 1))
            assert count == row["per_worker"][w]


# ---------------------------------------------------------------- audit


def test_audit_detects_an_extra_record_past_the_stop(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    records = output_of(intent_path) / "final" / "records"
    entry = json.loads((records / "final-candidate-frozen-w00000.json").read_text())
    unit = R.phase_units(R.read_json(intent_path), "final")[30]  # first unit of look 1
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
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
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
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    intent = json.loads(intent_path.read_text())
    intent["plan"]["efficacy_boundaries"][0] -= 0.01
    intent_path.write_text(json.dumps(intent))
    failed = {row["rule"] for row in A.run_audit(intent_path.parent)["failures"]}
    assert {"plan.efficacy_boundaries", "plan.sha256", "intent.sha256_bound"} <= failed


def test_audit_detects_a_relabelled_decision_and_a_dry_run_receipt(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    output = output_of(intent_path)
    claim = json.loads((output / "producer-outcome.json").read_text())
    claim["outcome"] = "STRICT_PASS"
    (output / "producer-outcome.json").write_text(json.dumps(claim))
    (output / "receipt.json").write_text("{}")
    failed = {row["rule"] for row in A.run_audit(intent_path.parent)["failures"]}
    assert {"outcome.producer_claim", "outcome.dry_run_has_no_receipt"} <= failed


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


def test_example_spec_never_plays_a_game_and_needs_excluded_seeds():
    spec = R.resolve_spec("research.sequential_strict_template.example_spec:SPEC")
    assert spec is example_spec.SPEC
    with pytest.raises(NotImplementedError):
        spec.episode_runner({}, {}, None)
    with pytest.raises(NotImplementedError):
        spec.excluded_seeds()


def test_dry_run_must_use_the_in_process_executor_and_its_own_slots(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    for executor in (None, R.SubprocessExecutor([])):
        with pytest.raises(R.StrictRunError, match="in-process executor"):
            R.run(
                intent_path,
                spec=spec,
                executor=executor,
                skew_runner=fake_skew(),
                audit_runner=R.in_process_audit_runner,
            )
    assert not output_of(intent_path).exists()
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(R.StrictRunError, match="global CPU slots"):
        prepare_run(other, spec, slot_lock_root=R.SLOT_LOCK_ROOT)
    with pytest.raises(R.StrictRunError, match="own ledger"):
        prepare_run(other, spec, ledger_path=R.LEDGER_PATH)


def test_ledger_refuses_a_second_run_of_a_final_namespace(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    ledger = tmp_path / "ledger.jsonl"
    [entry] = R.read_ledger(ledger)
    assert entry["study_id"] == "test-sequential-strict"
    assert entry["intent_sha256"] == R.sha256_file(intent_path)
    assert R.ledger_status(ledger)[0]["state"] == "CLOSED"
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(R.StrictRunError, match="already started a run"):
        R.build_intent(
            spec, **dict(intent_kwargs(other), ledger_path=ledger, out_root=other / "run-v2")
        )


def test_audit_never_passes_an_unclosed_root(tmp_path):
    spec = make_spec(FakeWorld(effect=400.0), tmp_path)
    intent_path = prepare_run(tmp_path, spec)
    execute(intent_path, spec)
    assert A.run_audit(intent_path.parent)["status"] == "PASS"
    (output_of(intent_path) / "closeout.json").unlink()
    assert A.run_audit(intent_path.parent)["status"] == "UNCLOSED"
    assert A.main(["--root", str(intent_path.parent), "--out", str(tmp_path / "audit")]) == 1


def test_audit_requires_the_global_slot_root_and_ledger_for_production():
    production = {
        "dry_run": False,
        "allow_dirty": False,
        "source_closure": {"dirty": []},
        "slot_lock_root": "/tmp/elsewhere",
        "ledger_path": "/tmp/elsewhere/ledger.jsonl",
    }
    clean = {
        "provenance": {
            "executor": "subprocess",
            "skew_runner": "subprocess_skew_runner",
            "audit_runner": "subprocess_audit_runner",
            "allow_dirty": False,
            "dry_run": False,
        }
    }
    problems = A.provenance_problems(production, clean, [], None, None)
    assert any("slot lock root" in p for p in problems) and any("ledger" in p for p in problems)
    production.update(slot_lock_root=A.SLOT_LOCK_ROOT, ledger_path=A.LEDGER_PATH)
    assert A.provenance_problems(production, clean, [], None, None) == []
    assert A.SLOT_LOCK_ROOT == str(REAL_SLOT_LOCK_ROOT)
