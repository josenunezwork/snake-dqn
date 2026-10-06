"""Tests for research/frp3_strict_20261005 (FRP-v3 seed-12 M3@60000 + v8 vs champion + v8).

No test plays an episode: ``tournament_eval.rollout``, the subprocess executor, the subprocess
skew/audit runners and ``supervise_children`` raise in every test unless a test replaces
``rollout`` with a fake that returns a saved v8 record. Dry runs use the in-process executor,
fake episodes, a fake skew probe and the in-process audit, with their own slot lock root and
ledger (the global ones are patched to unusable paths). The LH-1 receipt is synthetic
everywhere (the real one is the gate's precondition, never faked in production code).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import random
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.apex_safety_20260926 import dev_screen
from research.frp3_strict_20261005 import preflight as P
from research.frp3_strict_20261005 import spec as S
from research.runpod_fanout import jobspec, serverless
from research.sequential_strict_template import remote_backend as RB
from research.sequential_strict_template import sequential_runner as R

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "research" / "frp3_strict_20261005"
V8_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "apex_veto_v8_strict"
PYTHON = Path("/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python")
ARTIFACTS_AVAILABLE = (
    S.V8_STRICT_RECEIPT.is_file()
    and S.PHASE_R_SUMMARY.is_file()
    and S.CANDIDATE_PATH.is_file()
    and (S.CHECKPOINT_DIR / S.CHAMPION_NAME).is_file()
)
needs_artifacts = pytest.mark.skipif(
    not ARTIFACTS_AVAILABLE, reason="needs the local artifacts and checkpoints"
)
PREREGISTERED = (PACKAGE / "operating_characteristics.json").is_file() and (
    PACKAGE / "paired_band_check.json"
).is_file()
needs_preregistration = pytest.mark.skipif(
    not PREREGISTERED, reason="pre-registration outputs not generated yet"
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


# ---------------------------------------------------------------- synthetic LH-1 receipts


def lh1_receipt(tmp_path: Path, **over) -> Path:
    """A receipt shaped like research/longh_screen analyze + lh1.py analyze_main output."""
    outcome = over.pop("outcome", "CLEAR")
    receipt = {
        "kind": "screen",
        "job_id": S.LH1_JOB_ID,
        "repo_commit": S.LH1_JOB_COMMIT,
        "authority": "tier-1 screen (non-authoritative)",
        "label": "screen (non-authoritative)",
        "missing": [],
        "problems": [],
        "platform": {"single_platform": True},
        "counts": {},
        "files": {},
        "descriptives": {},
        "prefix_controls": {"verified": True},
        "rules": {},
        "contrasts": [
            {"name": "C1", "candidate": "C1", "reference": "I", "decision": {"outcome": outcome}}
        ],
        "outcomes": {"C1": outcome},
        "outcome": outcome,
        "job_path": S.LH1_JOB_PATH,
        "job_sha256": S.LH1_JOB_SHA256,
        "intent_sha256": "1" * 64,
        "analysis_commit": "2" * 40,
        "decision_code_drift": [],
        "finished_utc": "2026-10-05T23:00:00+00:00",
    }
    receipt.update(over)
    path = tmp_path / "lh1" / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt))
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_lh1_clear_receipt_binds_by_sha256(tmp_path):
    path = lh1_receipt(tmp_path)
    binding = S.lh1_binding(path, sha(path))
    assert binding["outcome"] == "CLEAR"
    assert binding["receipt_sha256"] == sha(path)
    assert binding["job_sha256"] == S.LH1_JOB_SHA256


def test_missing_lh1_receipt_refuses(tmp_path):
    with pytest.raises(R.StrictRunError, match="refuses to start until the LH-1 screen"):
        S.lh1_binding(tmp_path / "nope" / "receipt.json")


def test_lh1_receipt_is_pinned_and_twins_refuse(tmp_path):
    path = lh1_receipt(tmp_path)
    with pytest.raises(R.StrictRunError, match="is not the pinned"):
        S.lh1_binding(path)  # the module pin is the real receipt's sha256
    for twin in ("receipt.1759700000.json", "receipt.partial.json"):
        (path.parent / twin).write_text("{}")
        with pytest.raises(R.StrictRunError, match="twins"):
            S.lh1_binding(path, sha(path))
        (path.parent / twin).unlink()
    assert S.lh1_binding(path, sha(path))["outcome"] == "CLEAR"


@pytest.mark.skipif(not S.LH1_RECEIPT.is_file(), reason="needs the real LH-1 receipt")
def test_the_real_lh1_receipt_is_clear_and_pinned():
    binding = S.lh1_binding()
    assert binding["outcome"] == "CLEAR" and binding["receipt_sha256"] == S.LH1_RECEIPT_SHA256


@pytest.mark.parametrize(
    "over, why",
    [
        ({"outcome": "REASON_REQUIRED"}, "outcome is CLEAR"),
        ({"outcome": "BLOCK"}, "outcome is CLEAR"),
        ({"outcome": "INCOMPLETE"}, "outcome is CLEAR"),
        ({"outcome": "INVALID", "problems": ["x"]}, "no problems"),
        ({"missing": ["I-frozen-1"]}, "no missing episodes"),
        ({"job_id": "lh1-other"}, "job_id is"),
        ({"job_sha256": "0" * 64}, "job_sha256 is the pinned"),
        ({"repo_commit": "0" * 40}, "repo_commit is the job's"),
        ({"kind": "calibration"}, "kind is screen"),
        ({"prefix_controls": {"verified": False}}, "prefix controls verified"),
        ({"decision_code_drift": ["research/longh_screen/decide.py"]}, "did not drift"),
        ({"decision_code_drift": None}, "did not drift"),
        ({"outcomes": {"C1": "CLEAR", "C2": "CLEAR"}}, "one contrast"),
    ],
)
def test_lh1_receipt_that_is_not_clear_refuses(tmp_path, over, why):
    path = lh1_receipt(tmp_path, **over)
    with pytest.raises(R.StrictRunError, match=why):
        S.lh1_binding(path, sha(path))


def test_lh1_pins_match_the_lh1_branch_when_present():
    """The pinned job file and the receipt keys the binding reads come from branch lh1-screen."""
    show = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{S.LH1_JOB_FILE_COMMIT}:{S.LH1_JOB_PATH}"],
        capture_output=True,
    )
    if show.returncode != 0:
        pytest.skip("branch lh1-screen (commit c9adbc3) not in this clone")
    assert hashlib.sha256(show.stdout).hexdigest() == S.LH1_JOB_SHA256
    job = json.loads(show.stdout)
    assert job["job_id"] == S.LH1_JOB_ID and job["kind"] == "screen"
    assert job["repo_commit"] == S.LH1_JOB_COMMIT
    assert job["arms"]["C1"]["checkpoint_sha256"] == S.CANDIDATE_SHA256
    assert job["arms"]["I"]["checkpoint_sha256"] == S.CHAMPION_SHA256
    for arm in ("C1", "I"):
        assert job["arms"][arm]["veto"]["variant"] == "v8"
        assert job["arms"][arm]["veto"]["lambda"] == S.VETO_LAMBDA
    assert job["contrasts"] == [{"candidate": "C1", "name": "C1", "reference": "I"}]
    assert job["domain"] in P.NEWER_DOMAINS
    sources = {}
    for rel in ("research/longh_screen/analyze.py", "research/longh_screen/lh1.py"):
        sources[rel] = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{S.LH1_JOB_FILE_COMMIT}:{rel}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    analyze = sources["research/longh_screen/analyze.py"]
    for key in ('"missing"', '"problems"', 'out["prefix_controls"]', 'out["outcomes"]'):
        assert key in analyze
    assert 'name = "summary.json" if job["kind"] == "calibration" else "receipt.json"' in (
        sources["research/longh_screen/lh1.py"]
    )
    assert '"decision_code_drift": drift' in sources["research/longh_screen/lh1.py"]


# ---------------------------------------------------------------- spec wiring


def test_spec_wiring_is_survival_band_v2_with_remote_hooks():
    spec = S.SPEC
    R.validate_spec(spec)
    assert spec.template_version == R.TEMPLATE_VERSION_POOLED
    assert spec.band_policy == "pooled_ni_continue"
    assert spec.bands == R.paired_survival_bands(("frozen", "scripted", "mixed"))
    assert spec.namespaces == {
        "calibration": "apex-frp3-strict-dev-v1",
        "final": "apex-frp3-strict-final-v1",
    }
    assert spec.primary_metric == "mass_integral" and spec.ni_fraction == 0.03
    assert set(R.DEFAULT_CLOSURE_ROOTS) <= set(spec.closure_roots)
    assert {"research/apex_safety_20260926", S.PACKAGE} <= set(spec.closure_roots)
    assert callable(spec.remote_worker_setup) and callable(spec.remote_checkpoints)
    # survival band v2 (amendment 2026-10-06) option 1
    assert S.PAIRED_BAND == R.POOLED_BAND_OPTIONS["1"]
    assert R.parse_paired_band(S.PAIRED_BAND_ARG) == S.PAIRED_BAND
    # the superseded template-v2 band artifacts are kept, never used
    old = PACKAGE / "superseded_v2_band"
    failed = json.loads((old / "paired_band_check_pointwise_failed.json").read_text())
    assert failed["check"]["rule"] == "paired_M0.05_a0.05_pointwise"
    assert failed["check"]["passes"] is False
    assert not (PACKAGE / "paired_band_check_pointwise_failed.json").exists()


def test_remote_checkpoints_are_pool_plus_candidate_and_allow_listed():
    shas = S.remote_checkpoints()
    assert shas == sorted({sha for _, sha in S.POOL} | {S.CANDIDATE_SHA256})
    allow = jobspec.load_allowlist()
    assert all(sha in allow for sha in shas)
    row = allow[S.CANDIDATE_SHA256]
    assert row["root"] == "artifacts_root"
    assert jobspec.load_policy()["artifacts_root"] + "/" + row["path"] == str(S.CANDIDATE_PATH)


def test_allowlist_is_exactly_the_pool_plus_this_candidate():
    allow = jobspec.load_allowlist()
    assert sorted(allow) == S.remote_checkpoints()
    row = allow[S.CANDIDATE_SHA256]
    assert row["path"] == "frp-v3-20261005/train/arm-M3/seed-12/checkpoints/apex_mark_u60000.pth"
    assert row["approved"].startswith("user decision 2026-10-05")
    assert "Tier-2 sequential strict gate frp3-m3s12-strict-20261005" in row["strict_gate_use"]
    assert row["pinned_from"]["train_commit"] == "41a160840231ff059965a6ba47e24f2382a46fc4"


def test_allowlist_root_resolution_and_refusal(tmp_path):
    policy = {"checkpoint_root": str(tmp_path / "repo"), "artifacts_root": str(tmp_path / "art")}
    (tmp_path / "art" / "x").mkdir(parents=True)
    payload = b"candidate bytes"
    (tmp_path / "art" / "x" / "c.pth").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    allow = {sha: {"path": "x/c.pth", "root": "artifacts_root", "sha256": sha}}
    assert jobspec.resolve_checkpoints([sha], policy, allow) == {sha: tmp_path / "art/x/c.pth"}
    with pytest.raises(jobspec.JobError, match="not allowed"):
        jobspec.resolve_checkpoints([sha], policy, {sha: {**allow[sha], "root": "home"}})
    bad = {
        "schema": "runpod-fanout-checkpoint-allowlist/v1",
        "checkpoints": [{"path": "x/c.pth", "sha256": sha, "root": "home"}],
    }
    (tmp_path / "allow.json").write_text(json.dumps(bad))
    with pytest.raises(jobspec.JobError, match="bad allow-list root"):
        jobspec.load_allowlist(tmp_path / "allow.json")


def test_veto_sources_are_the_v8_strict_receipt_bytes():
    from research.apex_veto_v8_strict_20261003 import spec as v8

    assert S.VETO_SOURCE_SHA256S == v8.CANDIDATE_SOURCE_SHA256S
    assert {rel: S.sha256_file(REPO / rel) for rel in S.VETO_SOURCES} == S.VETO_SOURCE_SHA256S


def test_served_default_pins_the_same_v8_receipt():
    from web.backend import safety_veto_serving as serving

    assert serving.V8_STRICT_RECEIPT_SHA256 == S.V8_STRICT_RECEIPT_SHA256
    assert serving.V8_STRICT_RECEIPT_CHECKPOINT_SHA256 == S.CHAMPION_SHA256
    assert serving.V8_STRICT_RECEIPT_SOURCE_SHA256S == S.VETO_SOURCE_SHA256S


@needs_artifacts
def test_arm_identities_differ_only_in_the_hero_checkpoint(tmp_path):
    clear = lh1_receipt(tmp_path)
    ids = S.arm_identities(lh1_receipt=clear, lh1_receipt_sha256=sha(clear))
    inc, cand = ids["incumbent"], ids["candidate"]
    assert inc["checkpoint_sha256"] == S.CHAMPION_SHA256
    assert cand["checkpoint_sha256"] == S.CANDIDATE_SHA256
    for key in ("method", "descriptor", "source_sha256", "source_sha256s", "install"):
        assert inc[key] == cand[key]
    assert inc["method"] == S.VETO_METHOD
    assert inc["descriptor"]["space_preference_lambda"] == 8.0
    assert inc["release_binding"]["v8_strict_receipt_sha256"] == S.V8_STRICT_RECEIPT_SHA256
    assert cand["phase_r_binding"]["status"] == "GO_R"
    assert cand["phase_r_binding"]["summary_sha256"] == S.PHASE_R_SUMMARY_SHA256
    assert cand["screen_binding"]["outcome"] == "CLEAR"


@needs_artifacts
def test_arm_identities_refuse_without_a_clear_lh1_receipt(tmp_path):
    with pytest.raises(R.StrictRunError, match="LH-1"):
        S.arm_identities(lh1_receipt=tmp_path / "missing.json")
    with pytest.raises(R.StrictRunError, match="LH-1"):
        bad = lh1_receipt(tmp_path, outcome="REASON_REQUIRED")
        S.arm_identities(lh1_receipt=bad, lh1_receipt_sha256=sha(bad))


@needs_artifacts
def test_arm_identities_refuse_drifted_bindings(tmp_path):
    clear = lh1_receipt(tmp_path)
    pin = sha(clear)
    fake = tmp_path / "ckpt.pth"
    fake.write_bytes(b"not the candidate")
    with pytest.raises(R.StrictRunError, match="candidate checkpoint sha256"):
        S.arm_identities(lh1_receipt=clear, lh1_receipt_sha256=pin, candidate_path=fake)
    receipt = tmp_path / "v8receipt.json"
    receipt.write_bytes(S.V8_STRICT_RECEIPT.read_bytes() + b" ")
    with pytest.raises(R.StrictRunError, match="v8 strict receipt sha256"):
        S.arm_identities(lh1_receipt=clear, lh1_receipt_sha256=pin, v8_receipt=receipt)
    summary = tmp_path / "summary.json"
    summary.write_bytes(S.PHASE_R_SUMMARY.read_bytes() + b" ")
    with pytest.raises(R.StrictRunError, match="Phase R summary sha256"):
        S.arm_identities(lh1_receipt=clear, lh1_receipt_sha256=pin, phase_r_summary=summary)


# ---------------------------------------------------------------- episodes


def _v8_record():
    return json.loads((V8_FIXTURES / "screen_B_scripted_w0.json").read_text())


@contextlib.contextmanager
def _fake_installer(install):
    class Hero:
        safety_veto = None

    class Veto:
        def diagnostics_record(self):
            return {"head_checks": 1, "v7": {"rerank_changes": 0}}

    install(Hero())
    yield [Veto()]


def _context(ids, record, paths):
    return {
        "profile": object(),
        "profile_ref": {
            "descriptor": record["evaluation_profile"],
            "digest": record["evaluation_profile_digest"],
        },
        "lookup": dev_screen.agent_lookup(paths),
        "paths": paths,
        "arms": ids,
    }


def _ids():
    descriptor = {"method": S.VETO_METHOD}
    return {
        arm: {"checkpoint_sha256": S.HERO_SHA256[arm], "descriptor": descriptor} for arm in S.ARMS
    }


@pytest.mark.parametrize("arm", ["incumbent", "candidate"])
def test_episode_runner_plays_the_arms_checkpoint_with_v8(monkeypatch, arm):
    from src.evaluation import safety_veto_v8
    from src.scripts import tournament_eval

    entry = _v8_record()
    record = entry["record"]
    row = {
        "mix": "scripted",
        "world_seed": entry["world_seed"],
        "slots": [
            r for r in dev_screen._design_rows([entry["world_seed"]]) if r["mix"] == "scripted"
        ][0]["slots"],
        "world_index": 0,
    }
    paths = {sha: f"/ckpt/{sha}.pth" for sha in S.remote_checkpoints()}
    calls, installs = [], []

    def fake_rollout(hero, opponents, horizon, seed, **kwargs):
        calls.append((hero, opponents, horizon, seed, kwargs))
        return json.loads(json.dumps(record))

    monkeypatch.setattr(tournament_eval, "rollout", fake_rollout)
    monkeypatch.setattr(dev_screen, "hero_veto_installer", _fake_installer)
    monkeypatch.setattr(
        safety_veto_v8, "install_space_and_head_veto", lambda h, lam: installs.append(lam)
    )
    monkeypatch.setattr(S, "verify_checkpoints", lambda p: dict(p))
    from src.evaluation import strict_promotion

    monkeypatch.setattr(strict_promotion, "validate_strict_world_record", lambda *a, **k: None)
    episode = {
        "arm": arm,
        "mix": "scripted",
        "world_seed": entry["world_seed"],
        "episode_id": f"final-{arm}-scripted-w00000",
    }
    out = S.episode_runner(episode, row, _context(_ids(), record, paths))
    [(hero, opponents, horizon, seed, kwargs)] = calls
    assert hero == ("checkpoint", paths[S.HERO_SHA256[arm]])
    assert installs == [8.0]
    assert horizon == 5000 and seed == entry["world_seed"]
    assert kwargs["hero_safety_veto"] is True and kwargs["mix_id"] == "scripted"
    assert out["hero_checkpoint_sha256"] == S.HERO_SHA256[arm]
    assert out["mass_integral"] == record["mass_integral"]
    assert S.validate_record({"arm": arm, "record": out}, row) == []


@needs_artifacts
def test_episode_runner_validates_a_real_v8_record_shape(monkeypatch):
    """The strict validator with the v8 descriptor accepts a real v8 H5000 record (both arms
    carry the same wrapper, so the candidate arm uses the same descriptor)."""
    from src.scripts import tournament_eval

    entry = _v8_record()
    record = entry["record"]
    row = {
        "mix": "scripted",
        "world_seed": entry["world_seed"],
        "slots": [
            r for r in dev_screen._design_rows([entry["world_seed"]]) if r["mix"] == "scripted"
        ][0]["slots"],
        "world_index": 0,
    }
    monkeypatch.setattr(tournament_eval, "rollout", lambda *a, **k: json.loads(json.dumps(record)))
    monkeypatch.setattr(dev_screen, "hero_veto_installer", _fake_installer)
    paths = S.checkpoint_paths()
    ids = {
        arm: {
            "checkpoint_sha256": S.HERO_SHA256[arm],
            "descriptor": S._veto_identity(REPO)["descriptor"],
        }
        for arm in S.ARMS
    }
    episode = {"arm": "candidate", "mix": "scripted", "world_seed": entry["world_seed"]}
    episode["episode_id"] = "final-candidate-scripted-w00000"
    out = S.episode_runner(episode, row, _context(ids, record, paths))
    assert out["hero_checkpoint_sha256"] == S.CANDIDATE_SHA256
    bad = {**record, "probes": {**record["probes"], "safety_veto": {"method": "other"}}}
    monkeypatch.setattr(tournament_eval, "rollout", lambda *a, **k: json.loads(json.dumps(bad)))
    with pytest.raises(R.StrictRunError, match="strict record shape"):
        S.episode_runner(episode, row, _context(ids, record, paths))


def test_episode_runner_refuses_an_arm_bound_to_another_checkpoint(monkeypatch):
    monkeypatch.setattr(S, "verify_checkpoints", lambda p: dict(p))
    paths = {sha: f"/ckpt/{sha}.pth" for sha in S.remote_checkpoints()}
    ids = _ids()
    ids["candidate"] = dict(ids["incumbent"])  # the candidate arm naming the champion
    episode = {"arm": "candidate", "mix": "scripted", "world_seed": 7, "episode_id": "x"}
    row = {"mix": "scripted", "world_seed": 7, "slots": [], "world_index": 0}
    with pytest.raises(R.StrictRunError, match="identity names checkpoint"):
        S.episode_runner(episode, row, _context(ids, _v8_record()["record"], paths))


def test_episode_runner_reaches_rollout_only_through_the_guard(monkeypatch):
    monkeypatch.setattr(S, "verify_checkpoints", lambda p: dict(p))
    paths = {sha: f"/ckpt/{sha}.pth" for sha in S.remote_checkpoints()}
    row = {"mix": "scripted", "world_seed": 7, "slots": [], "world_index": 0}
    episode = {"arm": "candidate", "mix": "scripted", "world_seed": 7, "episode_id": "x"}
    with pytest.raises(AssertionError, match="never launch a real rollout"):
        S.episode_runner(episode, row, _context(_ids(), _v8_record()["record"], paths))


def test_validate_record_rejects_wrong_hero_method_and_diagnostics():
    row = {"mix": "frozen", "world_seed": 5}
    good = {
        "seed": 5,
        "world_identity": {"seed": 5, "mix_id": "frozen"},
        "probes": {"safety_veto": {"method": S.VETO_METHOD}},
        "veto_diagnostics": {"head_checks": 1},
        "hero_checkpoint_sha256": S.CANDIDATE_SHA256,
    }
    assert S.validate_record({"arm": "candidate", "record": good}, row) == []
    assert S.validate_record({"arm": "incumbent", "record": good}, row)  # wrong hero
    for bad in (
        {**good, "probes": {"safety_veto": {"method": "free-space-veto/v7"}}},
        {**good, "veto_diagnostics": {"rerank_changes": 0}},
        {**good, "world_identity": {"seed": 6, "mix_id": "frozen"}},
        {**good, "seed": 6},
    ):
        assert S.validate_record({"arm": "candidate", "record": bad}, row)
    assert S.validate_record({"arm": "other", "record": good}, row)


def test_checkpoint_maps_and_verification(tmp_path):
    paths = {sha: str(tmp_path / f"{sha}.pth") for sha in S.remote_checkpoints()}
    with pytest.raises(FileNotFoundError):
        S.verify_checkpoints(paths)
    with pytest.raises(R.StrictRunError, match="pool \\+ candidate"):
        S.verify_checkpoints({S.CANDIDATE_SHA256: "x"})
    mac = S.checkpoint_paths()
    assert mac[S.CANDIDATE_SHA256] == str(S.CANDIDATE_PATH)
    assert mac[S.CHAMPION_SHA256] == str(S.CHECKPOINT_DIR / S.CHAMPION_NAME)
    assert set(mac) == set(S.remote_checkpoints())


@needs_artifacts
def test_worker_setup_loads_config_profile_and_both_heroes_in_a_child(tmp_path):
    code = (
        "import json;"
        "from research.frp3_strict_20261005 import spec as S;"
        "ctx = S.worker_setup({'arms': {}});"
        "print(json.dumps({'digest': ctx['profile'].digest, 'lookup': sorted(ctx['lookup']),"
        " 'cand': ctx['lookup'][S.CANDIDATE_SHA256]}))"
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
    assert set(S.remote_checkpoints()) <= set(out["lookup"])
    assert out["cand"] == ["checkpoint", str(S.CANDIDATE_PATH)]


@needs_artifacts
def test_remote_worker_setup_maps_ckpt_dir_shas_in_a_child(tmp_path):
    """The RunPod-side setup in an isolated child: checkpoints come only from ckpt_dir as
    <sha256>.pth (symlinks to the real files here), every one re-hashed; a missing one fails."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for sha256, path in S.checkpoint_paths().items():
        (ckpt / f"{sha256}.pth").symlink_to(path)
    code = (
        "import json, sys;"
        "from pathlib import Path;"
        "from research.frp3_strict_20261005 import spec as S;"
        "ctx = S.remote_worker_setup({'arms': {}}, Path(sys.argv[1]));"
        "print(json.dumps({'digest': ctx['profile'].digest, 'paths': ctx['paths'],"
        " 'cand': ctx['lookup'][S.CANDIDATE_SHA256]}))"
    )

    def child():
        return subprocess.run(
            [
                str(PYTHON),
                "-I",
                "-B",
                "-c",
                f"import sys; sys.path.insert(0, {str(REPO)!r}); {code}",
                str(ckpt),
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
            env={"SNAKE_DQN_DEVICE": "cpu", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
        )

    proc = child()
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["digest"] == S.PROFILE_DIGEST
    assert out["paths"] == {s: str(ckpt / f"{s}.pth") for s in S.remote_checkpoints()}
    assert out["cand"] == ["checkpoint", str(ckpt / f"{S.CANDIDATE_SHA256}.pth")]
    (ckpt / f"{S.CANDIDATE_SHA256}.pth").unlink()
    proc = child()
    assert proc.returncode != 0 and S.CANDIDATE_SHA256 in proc.stderr


# ---------------------------------------------------------------- worlds


def test_banks_are_the_template_recipe_and_prefix_invariant():
    banks = P.gate_banks(300)
    assert banks["calibration"] == R.seed_bank(S.NAMESPACES["calibration"], 16)
    assert banks["final"] == R.seed_bank(S.NAMESPACES["final"], 300)
    assert S.build_row("final", "mixed", 5, banks["final"][5])["world_index"] == 5
    full = S._bank_rows(S.NAMESPACES["final"])
    prefix = {(r["mix"], i): r for i, r in enumerate(dev_screen._design_rows(banks["final"][:20]))}
    assert prefix  # dev_screen rows of a prefix bank (one row per mix and world)
    for i, seed in enumerate(banks["final"][:20]):
        for mix in S.MIXES:
            assert full[(mix, i)]["world_seed"] == seed


@functools.lru_cache(maxsize=1)
def _light_exclusions():
    return P.excluded_seeds(scan=False)


def test_banks_are_disjoint_from_every_enumerable_namespace():
    report = P.report(scan=False)
    assert report["disjoint"], report["problems"]
    excluded = _light_exclusions()
    names = " ".join(excluded)
    for domain in (
        "apex-frp-v3-eval-v1/seed12",
        "apex-frp-v3-train-v1/seed12/actor0",
        "apex-frp-v2-train-v1/seed0/ext/actor4",
        "apex-lh1-screen-frp3-s12-v1/worlds",
        "apex-veto-v8-strict-final-v1/worlds",
        "frp-v3 training_exclusions.json",
    ):
        assert domain in names, domain
    assert all(not name.startswith("apex-frp3-strict") for name in excluded)


def test_frozen_frp_constants_match_their_branches_when_present():
    for branch, module, frozen in (
        ("frp-v3", "research/frp_v3_20261005/spec.py", P.FRP_V3),
        ("frp-v2", "research/frp_v2_20261004/spec.py", P.FRP_V2),
    ):
        show = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{branch}:{module}"], capture_output=True, text=True
        )
        if show.returncode != 0:
            pytest.skip(f"branch {branch} not in this clone")
        text = show.stdout
        for key, value in frozen.items():
            if key.endswith("_domain"):
                assert f'"{value}"' in text, (branch, key)
        assert f"SEEDS = {tuple(frozen['seeds'])}" in text
        assert f"NUM_ACTORS = {frozen['actors']}" in text


def test_artifact_scan_reads_seed_keys_and_falls_back_to_a_regex(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    (root / "study-a").mkdir(parents=True)
    (root / "study-a" / "r.json").write_text(
        json.dumps({"record": {"seed": 11, "x": 5}, "world_seeds": [12, 13], "n": 99})
    )
    (root / "study-a" / "broken.json").write_text('{"world_seed": 21, "trunc')
    (root / "study-a" / "events.jsonl").write_text(
        json.dumps({"event": "x", "world_seed": 41}) + "\n" + '{"seed": 42, "cut' + "\n\n"
    )
    (root / P.STUDY_ROOT_NAME).mkdir()
    (root / P.STUDY_ROOT_NAME / "own.json").write_text(json.dumps({"seed": 31}))
    monkeypatch.setattr(P, "SCAN_MAX_BYTES", 1 << 20)
    out = P.scan_artifacts(root)
    assert out == {
        P.STUDY_ROOT_NAME: {"files": 1, "regex_files": 0, "seeds": [31]},  # own root scanned
        "study-a": {"files": 3, "regex_files": 2, "seeds": [11, 12, 13, 21, 41, 42]},
    }


def test_an_own_namespace_literal_on_any_branch_refuses(monkeypatch):
    class Done:
        def __init__(self, stdout):
            self.stdout, self.returncode, self.stderr = stdout, 0, ""

    def fake_run(cmd, **kw):
        if "for-each-ref" in cmd:
            return Done("refs/heads/other\n")
        return Done('EVAL_DOMAIN = "apex-frp3-strict-final-v1"\nX_DOMAIN = "apex-x-v1"\n')

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="is a domain literal on refs/heads/other"):
        P.branch_domain_literals()


def test_preflight_overlap_is_reported(monkeypatch):
    bank = P.gate_banks(300)
    monkeypatch.setattr(P, "excluded_seeds", lambda scan=True: {"x": [bank["final"][299]]})
    report = P.report(scan=False)
    assert report["disjoint"] is False and report["overlaps"] == {"x": [bank["final"][299]]}


# ---------------------------------------------------------------- pre-registration documents


@needs_preregistration
def test_preregistered_plan_is_the_one_prepare_freezes():
    from research.frp3_strict_20261005 import preregistration as PR

    oc = json.loads((PACKAGE / "operating_characteristics.json").read_text())
    chosen = oc["sizing"]["chosen"]
    assert (chosen["mde"], chosen["n_max"]) == (65.0, 275)
    params = PR.plan_parameters(275, 65.0)
    frozen = json.loads((PACKAGE / "frozen_plan.json").read_text())
    assert frozen == R.frozen_plan_document(params)
    assert oc["plan"] == R.plan_from_parameters(params).as_dict()
    assert oc["plan"]["look_sizes"] == [69, 138, 207, 275]
    assert oc["plan"]["band_policy"] == "pooled_ni_continue"
    cross = oc["joint_bootstrap"]["crosscheck_vs_run_sequential_gate"]
    assert cross["mismatches"] == [] and cross["replicates"] == 600
    assert cross["batches"]["at_null"]["outcome_counts"]["STOP_FUTILE"] > 0
    counts = cross["outcome_counts"]
    assert counts["STOP_FAIL_BANDS"] == 0 and counts["STOP_PASS"] and counts["FINAL_PASS"]
    dni = oc["development_delta_ni"]["value"]
    record = R.paired_check_record(S.SPEC, params, REPO, dni)
    assert record["rule"] == "pooled_ni_continue"
    assert record["passes"] and record["max_joint_rate"] <= 1.2 * 0.05
    assert record["rows_judged"] == 2 * 6 * 4
    assert record["pools"] == ["frp3_all_seeds", "frp3_s12"]
    assert oc["development_data"]["skew_input_sha256"] == R.sha256_file(
        PACKAGE / "phase_r_deltas.json"
    )


@needs_preregistration
@needs_artifacts
def test_skew_input_and_pool_are_the_candidates_phase_r_worlds():
    from research.frp3_strict_20261005 import preregistration as PR

    phase_r = PR.load_phase_r()
    own = phase_r["seeds"][12]
    deltas = json.loads((PACKAGE / "phase_r_deltas.json").read_text())
    assert deltas == {m: own[m]["deltas"] for m in S.MIXES}
    assert all(len(v) == 32 for v in deltas.values())
    pool = json.loads((PACKAGE / "paired_pool.json").read_text())
    assert pool == PR.pool_document(phase_r)
    assert pool["frp3_s12"]["scripted"]["mass_delta"] == deltas["scripted"]
    assert len(pool["frp3_all_seeds"]["world_seeds"]) == 160
    summary = json.loads(S.PHASE_R_SUMMARY.read_text())
    per_seed = {
        r["seed"]: r for r in summary["analysis"]["cells"]["M3@60000"]["h5000_prefix"]["per_seed"]
    }
    for mix in S.MIXES:
        mean = sum(deltas[mix]) / len(deltas[mix])
        assert mean == pytest.approx(per_seed[12]["per_mix_mean"][mix], abs=1e-9)


def test_protocol_names_the_platform_and_release_rollback():
    text = (PACKAGE / "protocol.md").read_text()
    assert "Execution platform: runpod-serverless" in text.splitlines()
    for phrase in ("checkpoint swap", "v8 unchanged", "Rollback", "serving qualification"):
        assert phrase in text, phrase


def test_remote_config_loads_and_meets_the_4x_rule():
    rpol, fp, sp = RB.policies()
    cfg = RB.load_remote_config(PACKAGE / "remote_config.json", rpol, sp)
    assert cfg["engine"] == "live" and cfg["horizon"] == 5000
    assert cfg["identity_worlds_per_mix"] == 3  # the amendment's default sample
    record = _v8_record()["record"]
    assert RB.get_path(record, cfg["horizon_path"]) == 5000
    for path, value in cfg["record_pins"].items():
        assert RB.get_path(record, path) == value, path
    plan = RB.plan_remote(
        mixes=S.MIXES,
        n_calibration=16,
        look_sizes=[69, 138, 207, 275],
        cfg=cfg,
        rpol=rpol,
        fp=fp,
        sp=sp,
        seeding=None,  # worst case: the strict runtime still has to be built (30 min)
    )
    assert plan["min_speedup"] == 4.0 and plan["meets_min_speedup"] is True
    choice = plan["choice"]
    assert choice["speedup"] >= 4.0 and choice["worst_gate_usd"] <= cfg["budget_usd"]
    assert choice["worst_identity_usd"] <= cfg["identity_budget_usd"]
    assert choice["reserved_usd_per_hr"] <= sp["max_endpoint_hourly_usd"]


# ---------------------------------------------------------------- dry runs (fake episodes)


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


def fake_spec(tmp_path, effect, survival_shift=0.0):
    clear = lh1_receipt(tmp_path)

    def runner(episode, row, context):
        seed, mix, arm = int(episode["world_seed"]), episode["mix"], episode["arm"]
        world = random.Random(f"{seed}|{mix}")
        noise = random.Random(f"{seed}|{mix}|{arm}")
        mass = 400.0 + world.gauss(0, 150) + noise.gauss(0, 40)
        survival = min(1.0, 0.8 + world.gauss(0, 0.02) + noise.gauss(0, 0.005))
        if arm == "candidate":
            mass += effect
            survival += survival_shift
        return {
            "seed": seed,
            "mass_integral": mass,
            "survival_fraction": survival,
            "world_identity": {"seed": seed, "mix_id": mix},
            "probes": {"safety_veto": {"method": S.VETO_METHOD}},
            "veto_diagnostics": {"head_checks": 0},
            "hero_checkpoint_sha256": S.HERO_SHA256[arm],
        }

    return dataclasses.replace(
        S.SPEC,
        episode_runner=runner,
        worker_setup=lambda intent: None,
        arm_identities=functools.partial(
            S.arm_identities, lh1_receipt=clear, lh1_receipt_sha256=sha(clear)
        ),
        excluded_seeds=_light_exclusions,
    )


def dry_intent(tmp_path, spec, **overrides):
    oc = json.loads((PACKAGE / "operating_characteristics.json").read_text())
    (tmp_path / "locks").mkdir(exist_ok=True)
    kwargs = dict(
        spec_ref="research.frp3_strict_20261005.spec:SPEC",
        out_root=tmp_path / "run-v1",
        n_max=275,
        mde=65.0,
        n_calibration=S.N_CALIBRATION,
        skew_input=PACKAGE / "phase_r_deltas.json",
        deadline=datetime.now(timezone.utc) + timedelta(hours=16),
        authorization_quote="test only",
        paired_band=dict(S.PAIRED_BAND),
        development_delta_ni=oc["development_delta_ni"]["value"],
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


@needs_preregistration
@needs_artifacts
def test_dry_intent_freezes_the_survival_band_v2_plan_and_bindings(tmp_path):
    spec = fake_spec(tmp_path, effect=0.0)
    intent = dry_intent(tmp_path, spec)
    assert intent["template_version"] == R.TEMPLATE_VERSION_POOLED
    assert intent["plan"]["n_max"] == 275 and intent["plan"]["mde"] == 65.0
    assert intent["plan"]["band_policy"] == "pooled_ni_continue"
    assert intent["plan"]["band_ni_margin"] == 0.05 and intent["plan"]["band_floor"] == 0.30
    assert intent["plan"]["band_bound"] == "rci_obf" and intent["plan"]["band_mix_margin"] == 0.075
    assert intent["paired_band_check"]["passes"] is True
    amendment = intent["band_amendment"]
    assert amendment["path"].endswith("governance_amendment_survival_band_v2_2026-10-06.md")
    # ratified option 1 at 865ee9e (before the strict intent): no production problems
    assert amendment["ratified"] is True and amendment["option"] == "1"
    assert amendment["production_problems"] == []
    assert intent["banks"]["final"] == R.seed_bank("apex-frp3-strict-final-v1", 275)
    assert intent["banks_check"]["passes"] is True
    assert intent["arms"]["candidate"]["screen_binding"]["outcome"] == "CLEAR"
    assert intent["skew_check"]["input_counts"] == {m: 32 for m in S.MIXES}
    for name, row in intent["preregistration"].items():
        assert R.sha256_file(Path(row["path"])) == row["sha256"]
        assert Path(row["path"]).parent == PACKAGE
    R.validate_intent(intent, spec)


@needs_preregistration
@needs_artifacts
def test_dry_v3_intent_freezes_runpod_execution_with_mac_fallback(tmp_path):
    spec = fake_spec(tmp_path, effect=0.0)
    seeding = {
        "runtime_id": "strict-test",
        "repo_sha256": "0" * 64,
        "template_id": "tpl",
        "ready": True,
        "needs": None,
        "problems": [],
    }
    intent = dry_intent(
        tmp_path,
        spec,
        remote_config=PACKAGE / "remote_config.json",
        remote_seeding=seeding,
    )
    block = intent["execution"]
    assert intent["template_version"] == R.TEMPLATE_VERSION_REMOTE
    assert block["band_template_version"] == R.TEMPLATE_VERSION_POOLED
    assert block["platform"] == "runpod-serverless"
    assert block["checkpoints"] == S.remote_checkpoints()
    assert block["speedup"] >= 4.0 and block["min_speedup"] == 4.0
    assert block["forced_below_min"] is False
    assert block["fallback"]["slots"] == 2 and block["fallback"]["backend"] == RB.FALLBACK_BACKEND
    sample = block["identity_check"]["sample"]
    assert len(sample) == 9 and all(s["world_index"] < 69 for s in sample)
    assert block["handler"]["runtime_id"] == serverless.runtime_id(
        jobspec.load_policy(), serverless.load_sls_policy(), RB.STRICT_HANDLER
    )
    assert block["serving_qualification"] == RB.SERVING_QUALIFICATION


@needs_preregistration
@needs_artifacts
def test_dry_run_large_effect_passes_dry_and_never_writes_a_receipt(tmp_path):
    spec = fake_spec(tmp_path, effect=200.0)
    path = R.prepare(dry_intent(tmp_path, spec))
    closeout = execute(path, spec)
    assert closeout["outcome"] == "DRY_RUN_PASS", closeout
    assert closeout["stop_look"] == 0 and closeout["worlds_per_mix_used"] == 69
    assert closeout["audit_passed"] is True
    assert not (path.parent / "output" / "receipt.json").exists()


@needs_preregistration
@needs_artifacts
def test_dry_run_survival_regression_continues_and_fails_survival_band_v2(tmp_path):
    spec = fake_spec(tmp_path, effect=200.0, survival_shift=-0.1)
    path = R.prepare(dry_intent(tmp_path, spec))
    closeout = execute(path, spec)
    assert closeout["outcome"] == "DRY_RUN_FAIL", closeout
    assert closeout["stop_decision"] == "FINAL_FAIL" and closeout["stop_look"] == 3
    looks = [
        json.loads((path.parent / "output" / "looks" / f"look-{k}.json").read_text())
        for k in range(4)
    ]
    assert [r["decision"] for r in looks] == ["CONTINUE_BANDS"] * 3 + ["FINAL_FAIL"]
    assert closeout["audit_passed"] is True


@needs_preregistration
@needs_artifacts
def test_dry_run_skew_failure_plays_no_final_world(tmp_path):
    spec = fake_spec(tmp_path, effect=200.0)
    path = R.prepare(dry_intent(tmp_path, spec))
    closeout = execute(path, spec, skew=fake_skew({"scripted": (0.03, 0.05)}))
    assert closeout["outcome"] == "DRY_RUN_SKEW_CHECK_FAILED", closeout
    assert not (path.parent / "output" / "final").exists()


def test_cli_resolves_the_real_spec_when_the_runner_is_main(tmp_path):
    """Script-mode ``prepare`` of this study's SPEC resolves it (no duplicate-module refusal)
    and stops at the deliberately omitted paired-band settings, before any scan or write."""
    proc = subprocess.run(
        [
            str(PYTHON),
            "-I",
            "-B",
            str(REPO / "research/sequential_strict_template/sequential_runner.py"),
            "prepare",
            "--spec",
            "research.frp3_strict_20261005.spec:SPEC",
            "--out-root",
            str(tmp_path / "x"),
            "--n-max",
            "275",
            "--mde",
            "65",
            "--n-calibration",
            "16",
            "--skew-input",
            str(PACKAGE / "phase_r_deltas.json"),
            "--deadline-utc",
            "2030-01-01T00:00:00+00:00",
            "--authorization-quote",
            "test",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={"SNAKE_DQN_DEVICE": "cpu", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode != 0
    assert "spec must be a StudySpec" not in proc.stderr
    # Since the real gate started (frp3-m3s12-strict-20261005/run-v1) the shared ledger
    # refuses its final namespace first; that is also past spec resolution.
    reused = "apex-frp3-strict-final-v1 already started a run" in proc.stderr
    assert "pooled_ni_continue needs explicit" in proc.stderr or reused, proc.stderr[-2000:]
    assert not (tmp_path / "x").exists()
