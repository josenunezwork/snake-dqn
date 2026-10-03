"""Tests for research/apex_veto_v8_strict_20261003 (the v8 vs v7 group-sequential strict spec).

No test plays an episode: ``tournament_eval.rollout``, ``dev_screen.run_episode``, the
subprocess executor, the subprocess skew/audit runners and ``supervise_children`` raise in
every test unless a test replaces ``rollout`` with a fake that returns a saved screen record.
Dry runs use the in-process executor, fake episodes, a fake skew probe and the in-process audit,
with their own slot lock root and ledger (the global ones are patched to unusable paths).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import random
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_v8_strict_20261003 import dry_run as D
from research.apex_veto_v8_strict_20261003 import spec as S
from research.sequential_strict_template import sequential_runner as R

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "research" / "apex_veto_v8_strict_20261003"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "apex_veto_v8_strict"
PYTHON = Path("/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python")
ARTIFACTS_AVAILABLE = (
    S.V7_STRICT_RECEIPT.is_file() and (S.CHECKPOINT_DIR / S.CHAMPION_NAME).is_file()
)
needs_artifacts = pytest.mark.skipif(
    not ARTIFACTS_AVAILABLE, reason="needs the local artifacts and checkpoints"
)


def _fatal(*args, **kwargs):
    raise AssertionError("tests must never launch a real rollout, probe or child process")


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    from src.scripts import tournament_eval

    monkeypatch.setattr(tournament_eval, "rollout", _fatal)
    monkeypatch.setattr(dev_screen, "run_episode", _fatal)
    monkeypatch.setattr(R, "subprocess_skew_runner", _fatal)
    monkeypatch.setattr(R, "subprocess_audit_runner", _fatal)
    monkeypatch.setattr(R, "supervise_children", _fatal)
    monkeypatch.setattr(R.SubprocessExecutor, "run_segment", _fatal)
    monkeypatch.setattr(R, "on_ac_power", lambda: True)
    monkeypatch.setattr(R, "SLOT_LOCK_ROOT", tmp_path / "global-locks-must-not-be-used")
    monkeypatch.setattr(R, "LEDGER_PATH", tmp_path / "global-ledger-must-not-be-used.jsonl")


def fake_skew(rates=None):
    rates = rates or {}

    def runner(ctx, mix, deltas_path, out_path, wall_seconds):
        efficacy, ni = rates.get(mix, (0.015, 0.05))
        reps = ctx.intent["skew_check"]["reps"]
        payload = {
            "part": "resample",
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
        child = {
            "command": ["fake-skew", mix],
            "returncode": 0,
            "termination": "natural_exit",
            "confirmed_exit": True,
            "peak_group_rss_bytes": 0,
        }
        return {
            "cause": None,
            "elapsed_seconds": 0.0,
            "wall_seconds": wall_seconds,
            "children": [child],
        }

    return runner


def dry_intent(tmp_path, spec, **overrides):
    (tmp_path / "locks").mkdir(exist_ok=True)
    kwargs = dict(
        spec_ref=D.FAKE_SPEC_REF,
        out_root=tmp_path / "run-v1",
        n_max=249,
        mde=30.0,
        n_calibration=S.N_CALIBRATION,
        skew_input=PACKAGE / "screen_deltas.json",
        deadline=datetime.now(timezone.utc) + timedelta(hours=16),
        authorization_quote="test only",
        skew_reps=1000,
        slot_lock_root=tmp_path / "locks",
        ledger_path=tmp_path / "ledger.jsonl",
        dry_run=True,
        allow_dirty=True,
    )
    kwargs.update(overrides)
    return R.build_intent(spec, **kwargs)


def execute(path, spec, skew=None):
    return R.run(
        path,
        spec=spec,
        executor=R.InProcessExecutor(),
        skew_runner=skew or fake_skew(),
        audit_runner=R.in_process_audit_runner,
    )


# ---------------------------------------------------------------- spec wiring


def test_spec_wiring_matches_the_v7_strict_design():
    from research.apex_veto_v7_strict_20261002 import strict_run as v7

    spec = S.SPEC
    R.validate_spec(spec)
    assert spec.namespaces == {
        "calibration": "apex-veto-v8-strict-dev-v1",
        "final": "apex-veto-v8-strict-final-v1",
    }
    assert spec.mixes == tuple(v7.MIXES) == ("frozen", "scripted", "mixed")
    assert spec.scripted_mix == "scripted" and spec.primary_metric == "mass_integral"
    assert spec.ni_fraction == v7.NI_FRACTION == 0.03
    assert spec.bands == R.survival_bands(spec.mixes, below=v7.BAND_BELOW, above=v7.BAND_ABOVE)
    assert S.HORIZON == v7.HORIZON == 5000 and S.PROFILE_NAME == v7.PROFILE_NAME
    assert S.N_CALIBRATION == v7.NAMESPACES["dev"][1] == 16
    assert set(R.DEFAULT_CLOSURE_ROOTS) <= set(spec.closure_roots)
    assert "research/apex_safety_20260926" in spec.closure_roots
    assert "research/apex_veto_v8_strict_20261003" in spec.closure_roots
    for name in R.PREREGISTRATION_DOCS:
        assert (REPO / getattr(spec, name)).is_file()
    assert spec.worker_setup is S.worker_setup and spec.episode_runner is S.episode_runner


def test_preregistration_reports_agree_with_the_plan_prepare_will_freeze():
    oc = json.loads((PACKAGE / "operating_characteristics.json").read_text())
    params = R.plan_parameters(n_max=249, mde=30.0, mixes=S.MIXES, scripted_mix=S.SCRIPTED_MIX)
    plan = R.plan_from_parameters(params).as_dict()
    assert oc["sizing"]["chosen"] == {"mde": 30.0, "n_fixed_90": 244, "n_max": 249}
    assert oc["plan"] == json.loads(json.dumps(plan))
    assert plan["look_sizes"] == [63, 125, 187, 249]
    assert oc["sizing"]["runtime_projection"]["within_limit"] is True
    assert oc["sizing"]["screen_effects_used"] is False
    infeasible = [row for row in oc["sizing"]["table"] if row["mde"] < 30.0]
    assert infeasible and not any(row["feasible"] for row in infeasible)
    assert oc["screen"]["screen_deltas_sha256"] == R.sha256_file(PACKAGE / "screen_deltas.json")
    assert oc["joint_bootstrap"]["crosscheck_vs_run_sequential_gate"]["mismatches"] == []
    bands = json.loads((PACKAGE / "look1_band_cost.json").read_text())
    assert bands["look_sizes"] == plan["look_sizes"]
    deltas = json.loads((PACKAGE / "screen_deltas.json").read_text())
    assert set(deltas) == set(S.MIXES) and all(len(v) == 60 for v in deltas.values())


@needs_artifacts
def test_screen_deltas_are_the_screen_summary_deltas():
    from research.apex_veto_v8_strict_20261003 import preregistration as P

    summary = json.loads((P.SCREEN_RUN / "merged" / "summary.json").read_text())
    deltas = json.loads((PACKAGE / "screen_deltas.json").read_text())
    for mix in S.MIXES:
        assert deltas[mix] == summary["per_mix"][mix]["mass_integral"]["deltas_B_minus_A"]
    assert R.sha256_file(P.SCREEN_RUN / "merged" / "summary.json") == P.SCREEN_SUMMARY_SHA256


# ---------------------------------------------------------------- identities


@needs_artifacts
def test_incumbent_is_the_released_v7_bound_to_its_strict_receipt():
    from web.backend import safety_veto_serving as serving

    ids = S.arm_identities()
    receipt = json.loads(S.V7_STRICT_RECEIPT.read_text())["candidate"]
    inc = ids["incumbent"]
    assert inc["method"] == receipt["wrapper"] == serving.V7_STRICT_RECEIPT_METHOD
    assert inc["descriptor"] == receipt["wrapper_identity"]["descriptor"]
    assert inc["source_sha256s"] == receipt["wrapper_identity"]["source_sha256s"]
    assert inc["source_sha256s"] == serving.V7_STRICT_RECEIPT_SOURCE_SHA256S
    assert inc["checkpoint_sha256"] == receipt["checkpoint_sha256"] == S.CHAMPION_SHA256
    assert S.V7_STRICT_RECEIPT_SHA256 == serving.V7_STRICT_RECEIPT_SHA256
    assert inc["release_binding"]["v7_strict_receipt_sha256"] == S.V7_STRICT_RECEIPT_SHA256
    assert inc["descriptor"]["space_preference_lambda"] == 4.0


@needs_artifacts
def test_candidate_is_v8_lambda_8_bound_to_the_screen_commit_bytes():
    ids = S.arm_identities()
    cand = ids["candidate"]
    assert cand["method"] == "free-space-veto/v8-space-and-head(lambda=8.0)"
    assert cand["descriptor"]["space_preference_lambda"] == 8.0
    assert cand["descriptor"]["head_avoidance"] is True
    assert set(cand["source_sha256s"]) == set(S.ARM_SOURCES["candidate"])
    for rel, expected in cand["source_sha256s"].items():
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{S.SCREEN_COMMIT}:{rel}"],
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == expected == S.CANDIDATE_SOURCE_SHA256S[rel]


def test_v8_imports_exactly_the_bound_veto_modules():
    import ast

    def veto_imports(rel):
        tree = ast.parse((REPO / rel).read_text())
        return {
            node.module.replace(".", "/") + ".py"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("src.evaluation.safety_veto")
        }

    closure, todo = set(), [S.V8]
    while todo:
        rel = todo.pop()
        if rel not in closure:
            closure.add(rel)
            todo.extend(veto_imports(rel))
    assert closure == set(S.ARM_SOURCES["candidate"])
    inc, todo = set(), [S.V7]
    while todo:
        rel = todo.pop()
        if rel not in inc:
            inc.add(rel)
            todo.extend(veto_imports(rel))
    assert inc == set(S.ARM_SOURCES["incumbent"])


@needs_artifacts
@pytest.mark.parametrize(
    "attribute, value",
    [
        ("V7_STRICT_RECEIPT_SHA256", "0" * 64),
        ("CANDIDATE_SOURCE_SHA256S", {**S.CANDIDATE_SOURCE_SHA256S, S.V8: "1" * 64}),
        ("INCUMBENT_SOURCE_SHA256S", {**S.INCUMBENT_SOURCE_SHA256S, S.V7: "2" * 64}),
        ("INCUMBENT_LAMBDA", 5.0),
        ("CANDIDATE_LAMBDA", 4.0),
        ("CHAMPION_SHA256", "3" * 64),
    ],
)
def test_identity_drift_is_refused(monkeypatch, attribute, value):
    monkeypatch.setattr(S, attribute, value)
    with pytest.raises(R.StrictRunError):
        S.arm_identities()


@needs_artifacts
def test_missing_v7_receipt_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "V7_STRICT_RECEIPT", tmp_path / "missing.json")
    with pytest.raises(R.StrictRunError, match="receipt"):
        S.arm_identities()


# ---------------------------------------------------------------- worlds


@needs_artifacts
def test_banks_are_disjoint_from_every_earlier_namespace():
    excluded = S.excluded_seeds()
    banks = {
        "calibration": R.seed_bank(S.NAMESPACES["calibration"], 16),
        "final": R.seed_bank(S.NAMESPACES["final"], R.N_MAX_CAP),
    }
    report = R.bank_report(banks, excluded)
    assert report["passes"], report["problems"]
    labels = set(excluded)
    for domain in (
        "apex-veto-v8-screen-v1",
        "apex-veto-v8-dev-v1",
        "apex-veto-v7-strict-final-v1",
        "apex-veto-v7-strict-dev-v1",
        "apex-veto-v7-screen-v1",
        "apex-veto-v7-dev-v1",
        "apex-veto-v7-web-serving-v1",
    ):
        assert any(label.startswith(f"{domain}/") for label in labels), domain
    assert "small-integer-seeds-0-999" in labels
    assert any(label.startswith("challenger:") for label in labels)
    assert any(label.startswith("strict-pilot-observed:") for label in labels)
    screen_summary = str(S.ARTIFACTS / "apex-veto-v8-screen-20261002/run-v1/merged/summary.json")
    observed = excluded[f"observed:{screen_summary}"]
    summary = json.loads(Path(screen_summary).read_text())
    assert set(summary["per_mix"]["frozen"]["paired_seeds"]) <= set(observed)
    assert set(S.NAMESPACES.values()).isdisjoint(S.EARLIER_DOMAINS)


def test_earlier_domains_cover_the_v8_screen_preflight():
    from research.apex_veto_v8_screen_20261002 import screen

    for domain, purposes in screen.EARLIER_DOMAINS.items():
        assert set(purposes) <= set(S.EARLIER_DOMAINS.get(domain, ())), domain
    assert {str(p) for p in screen.OBSERVED_SEED_SOURCES} <= {
        str(p) for p in S.OBSERVED_SEED_SOURCES
    }


def test_a_missing_observed_seed_source_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "OBSERVED_SEED_SOURCES", (tmp_path / "gone.json",))
    with pytest.raises(R.StrictRunError, match="missing"):
        S.observed_seeds(S.OBSERVED_SEED_SOURCES)


def test_overlapping_bank_is_refused_by_prepare(tmp_path):
    final = R.seed_bank(S.NAMESPACES["final"], 249)
    spec = dataclasses.replace(D.FAKE_SPEC, excluded_seeds=lambda: {"collides": [final[100]]})
    with pytest.raises(R.StrictRunError, match="world banks"):
        dry_intent(tmp_path, spec)


def test_rosters_are_bank_prefix_invariant():
    for phase, count in (("calibration", 16), ("final", 249)):
        seeds = R.seed_bank(S.NAMESPACES[phase], count)
        rows = dev_screen._design_rows(seeds)
        index = {seed: i for i, seed in enumerate(seeds)}
        for row in rows:
            built = S.build_row(phase, row["mix"], index[row["world_seed"]], row["world_seed"])
            assert built == {**row, "world_index": index[row["world_seed"]]}
    with pytest.raises(R.StrictRunError):
        S.build_row("final", "frozen", 0, 12345)


# ---------------------------------------------------------------- episodes


def _fixture(arm):
    return json.loads((FIXTURES / f"screen_{arm}_scripted_w0.json").read_text())


def test_fixtures_match_their_provenance():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())["files"]
    for name, row in provenance.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == row["sha256"]


@contextlib.contextmanager
def _fake_installer(install):
    class Hero:
        safety_veto = None

    yield [install(Hero())]


@needs_artifacts
@pytest.mark.parametrize("arm, screen_arm", [("incumbent", "A"), ("candidate", "B")])
def test_episode_runner_validates_real_screen_record_shapes(monkeypatch, arm, screen_arm):
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts import tournament_eval

    entry = _fixture(screen_arm)
    record = entry["record"]
    row = {
        "mix": "scripted",
        "world_seed": entry["world_seed"],
        "slots": [
            r for r in dev_screen._design_rows([entry["world_seed"]]) if r["mix"] == "scripted"
        ][0]["slots"],
        "world_index": 0,
    }
    profile = object()
    calls = []

    def fake_rollout(hero, opponents, horizon, seed, **kwargs):
        calls.append((hero, opponents, horizon, seed, kwargs))
        return json.loads(json.dumps(record))

    monkeypatch.setattr(tournament_eval, "rollout", fake_rollout)
    monkeypatch.setattr(dev_screen, "hero_veto_installer", _fake_installer)
    ids = S.arm_identities()
    context = {
        "profile": profile,
        "profile_ref": {
            "descriptor": record["evaluation_profile"],
            "digest": record["evaluation_profile_digest"],
        },
        "lookup": dev_screen.agent_lookup(S.verify_pool()),
        "arms": ids,
    }
    assert record["evaluation_profile_digest"] == S.PROFILE_DIGEST
    episode = {
        "arm": arm,
        "mix": "scripted",
        "world_seed": entry["world_seed"],
        "episode_id": f"final-{arm}-scripted-0",
        "phase": "final",
        "world_index": 0,
    }
    out = S.episode_runner(episode, row, context)
    [(hero, opponents, horizon, seed, kwargs)] = calls
    assert hero == ("checkpoint", str(S.CHECKPOINT_DIR / S.CHAMPION_NAME))
    assert horizon == 5000 and seed == entry["world_seed"]
    assert kwargs == {
        "profile": profile,
        "world_identity": _expected_world_identity(row),
        "mix_id": "scripted",
        "hero_safety_veto": True,
    }
    assert out["mass_integral"] == record["mass_integral"]
    assert S.DIAGNOSTICS_MARKER[arm] in out["veto_diagnostics"]
    envelope = {"arm": arm, "record": out}
    assert S.validate_record(envelope, row) == []
    # the other arm's descriptor must be refused by the strict validator
    other = "candidate" if arm == "incumbent" else "incumbent"
    context["arms"] = {arm: ids[other]}
    with pytest.raises(R.StrictRunError, match="strict record shape"):
        S.episode_runner(episode, row, context)


def test_episode_runner_reaches_rollout_only_through_the_guard(monkeypatch):
    monkeypatch.setattr(S, "verify_pool", lambda: {})
    row = {"mix": "scripted", "world_seed": 7, "slots": [], "world_index": 0}
    episode = {"arm": "candidate", "mix": "scripted", "world_seed": 7, "episode_id": "x"}
    context = {"lookup": {S.CHAMPION_SHA256: ("checkpoint", "x")}, "profile": None}
    with pytest.raises(AssertionError, match="never launch a real rollout"):
        S.episode_runner(episode, row, context)


def test_validate_record_rejects_wrong_method_and_diagnostics():
    row = {"mix": "frozen", "world_seed": 5}
    good = {
        "seed": 5,
        "world_identity": {"seed": 5, "mix_id": "frozen"},
        "probes": {"safety_veto": {"method": S.CANDIDATE_METHOD}},
        "veto_diagnostics": {"head_checks": 1, "v7": {"rerank_changes": 0}},
    }
    assert S.validate_record({"arm": "candidate", "record": good}, row) == []
    assert S.validate_record({"arm": "incumbent", "record": good}, row)  # v8 on incumbent
    bad = {**good, "veto_diagnostics": {"rerank_changes": 0}}
    assert S.validate_record({"arm": "candidate", "record": bad}, row)
    moved = {**good, "world_identity": {"seed": 6, "mix_id": "frozen"}}
    assert S.validate_record({"arm": "candidate", "record": moved}, row)
    assert S.validate_record({"arm": "other", "record": good}, row)


@needs_artifacts
def test_worker_setup_loads_the_pinned_config_and_profile_in_a_child():
    code = (
        "import json;"
        "from research.apex_veto_v8_strict_20261003 import spec as S;"
        "ctx = S.worker_setup({'arms': S.arm_identities()});"
        "print(json.dumps({'digest': ctx['profile'].digest,"
        " 'lookup': sorted(ctx['lookup'])}))"
    )
    proc = subprocess.run(
        [str(PYTHON), "-I", "-B", "-c", f"import sys; sys.path.insert(0, {str(REPO)!r}); {code}"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={"SNAKE_DQN_DEVICE": "cpu", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["digest"] == S.PROFILE_DIGEST
    assert set(sha for _, sha in S.POOL) <= set(out["lookup"])


# ---------------------------------------------------------------- refusal paths


@needs_artifacts
def test_production_intent_refuses_non_global_lock_root_and_dirty_sources(tmp_path):
    with pytest.raises(R.StrictRunError, match="global slot lock root"):
        dry_intent(tmp_path, S.SPEC, dry_run=False, allow_dirty=False)
    with pytest.raises(R.StrictRunError, match="allow_dirty"):
        dry_intent(
            tmp_path,
            S.SPEC,
            dry_run=False,
            slot_lock_root=R.SLOT_LOCK_ROOT,
            ledger_path=R.LEDGER_PATH,
            allow_dirty=True,
        )


@needs_artifacts
def test_production_run_refuses_an_injected_executor(tmp_path, monkeypatch):
    intent = dry_intent(tmp_path, D.FAKE_SPEC)
    intent["dry_run"] = False
    intent["slot_lock_root"] = str(R.SLOT_LOCK_ROOT)
    intent["ledger_path"] = str(R.LEDGER_PATH)
    intent["allow_dirty"] = False
    monkeypatch.setattr(R, "closure_drift", lambda closure: [])
    path = R.prepare(intent)
    with pytest.raises(R.StrictRunError, match="SubprocessExecutor"):
        R.run(path, spec=D.FAKE_SPEC, executor=R.InProcessExecutor())
    assert not (tmp_path / "run-v1" / "output").exists()


@needs_artifacts
def test_dry_intent_freezes_identities_banks_and_preregistration(tmp_path):
    intent = dry_intent(tmp_path, D.FAKE_SPEC)
    assert intent["arms"] == R.json_safe(S.arm_identities())
    assert intent["plan"]["n_max"] == 249 and intent["plan"]["mde"] == 30.0
    assert intent["plan"]["futility_policy"] == "followed"
    assert intent["plan"]["band_policy"] == "block_at_stop"
    assert intent["futility_action"] == "stop"
    assert intent["caps"]["workers"] == 2 and intent["caps"]["stage_seconds"] == R.DEFAULT_CAPS
    assert intent["banks"]["final"] == R.seed_bank("apex-veto-v8-strict-final-v1", 249)
    assert intent["banks_check"]["passes"] is True
    assert intent["skew_check"]["input_sha256"] == R.sha256_file(PACKAGE / "screen_deltas.json")
    assert intent["skew_check"]["input_counts"] == {m: 60 for m in S.MIXES}
    for name, row in intent["preregistration"].items():
        assert R.sha256_file(Path(row["path"])) == row["sha256"]
        assert Path(row["path"]).parent == PACKAGE


# ---------------------------------------------------------------- dry runs (fake episodes)


def _fake_spec(effect, survival_shift=0.0):
    def runner(episode, row, context):
        seed, mix, arm = int(episode["world_seed"]), episode["mix"], episode["arm"]
        world = random.Random(f"{seed}|{mix}")
        noise = random.Random(f"{seed}|{mix}|{arm}")
        mass = 300.0 + world.gauss(0, 120) + noise.gauss(0, 40)
        survival = 0.8 + world.gauss(0, 0.004) + noise.gauss(0, 0.002)
        if arm == "candidate":
            mass += effect
            survival += survival_shift
        return {
            "seed": seed,
            "mass_integral": mass,
            "survival_fraction": survival,
            "world_identity": {"seed": seed, "mix_id": mix},
            "probes": {"safety_veto": {"method": S.ARM_METHODS[arm]}},
            "veto_diagnostics": {S.DIAGNOSTICS_MARKER[arm]: 0},
        }

    return dataclasses.replace(D.FAKE_SPEC, episode_runner=runner)


@needs_artifacts
def test_dry_run_large_effect_stops_at_look_one_and_never_passes_strict(tmp_path):
    spec = _fake_spec(effect=150.0)
    path = R.prepare(dry_intent(tmp_path, spec))
    closeout = execute(path, spec)
    assert closeout["outcome"] == "DRY_RUN_PASS", closeout
    assert closeout["stop_look"] == 0 and closeout["worlds_per_mix_used"] == 63
    assert closeout["audit_passed"] is True
    assert not (path.parent / "output" / "receipt.json").exists()
    records = list((path.parent / "output" / "final" / "records").glob("*.json"))
    assert len(records) == 2 * 3 * 63


@needs_artifacts
def test_dry_run_null_effect_fails(tmp_path):
    spec = _fake_spec(effect=0.0)
    path = R.prepare(dry_intent(tmp_path, spec))
    closeout = execute(path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", closeout
    assert closeout["stop_decision"] in ("STOP_FUTILE", "FINAL_FAIL")


@needs_artifacts
def test_dry_run_band_violation_stops_fail_bands(tmp_path):
    spec = _fake_spec(effect=150.0, survival_shift=-0.1)
    path = R.prepare(dry_intent(tmp_path, spec))
    closeout = execute(path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", closeout
    assert closeout["stop_decision"] == "STOP_FAIL_BANDS"


@needs_artifacts
def test_dry_run_skew_failure_plays_no_final_world(tmp_path):
    spec = _fake_spec(effect=150.0)
    path = R.prepare(dry_intent(tmp_path, spec))
    closeout = execute(path, spec, skew=fake_skew({"mixed": (0.03, 0.05)}))
    assert closeout["outcome"] == "DRY_RUN_SKEW_CHECK_FAILED", closeout
    assert not (path.parent / "output" / "final").exists()


# ---------------------------------------------------------------- template CLI fix


def test_cli_resolves_a_study_spec_when_the_runner_is_main(tmp_path):
    """``prepare`` run as a script must accept a spec that imports the runner module."""
    proc = subprocess.run(
        [
            str(PYTHON),
            "-I",
            "-B",
            str(REPO / "research/sequential_strict_template/sequential_runner.py"),
            "prepare",
            "--spec",
            "research.sequential_strict_template.example_spec:SPEC",
            "--out-root",
            str(tmp_path / "x"),
            "--n-max",
            "40",
            "--mde",
            "30",
            "--n-calibration",
            "4",
            "--skew-input",
            str(tmp_path / "none.json"),
            "--deadline-utc",
            "2030-01-01T00:00:00+00:00",
            "--authorization-quote",
            "test",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode != 0
    assert "spec must be a StudySpec" not in proc.stderr
    assert "example_spec: list every earlier namespace" in proc.stderr
    assert not (tmp_path / "x").exists()
