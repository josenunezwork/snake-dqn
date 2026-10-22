"""Offline ops qualification fences and reviewed-plan execution."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

OPS = Path(__file__).resolve().parents[1] / "tools/ops"
sys.path.insert(0, str(OPS))
import qualification  # noqa: E402
import requalify  # noqa: E402


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def broker(tmp_path):
    repo = tmp_path / "broker"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Ops Tests")
    git(repo, "config", "user.email", "ops@example.invalid")
    transport = repo / "research/colab_cli"
    transport.mkdir(parents=True)
    (repo / ".gitignore").write_text("__pycache__/\n")
    (transport / "launch.sh").write_text("#!/bin/bash\nexit 0\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "fixture")
    return repo


def registry(repo, tmp_path):
    path = tmp_path / "qualified.json"
    qualification.save(
        path,
        {
            "schema": "snake-dqn-colab-cli-qualified/v1",
            "tree_hash": qualification.tree(repo),
        },
    )
    return path


def test_tree_identity_survives_other_commits_and_refuses_dirty(broker, tmp_path):
    record = registry(broker, tmp_path)
    first = qualification.check(broker, record)
    (broker / "unrelated").write_text("new")
    git(broker, "add", ".")
    git(broker, "commit", "-qm", "unrelated commit")
    assert qualification.check(broker, record) == first
    path = broker / "research/colab_cli/launch.sh"
    path.write_text("changed")
    with pytest.raises(ValueError, match="working-tree"):
        qualification.check(broker, record)
    git(broker, "add", ".")
    with pytest.raises(ValueError, match="working-tree"):
        qualification.check(broker, record)
    git(broker, "commit", "-qm", "transport changed")
    with pytest.raises(ValueError, match="UNQUALIFIED"):
        qualification.check(broker, record)


def test_untracked_transport_refused_and_remote_ref_checked(broker, tmp_path):
    record = registry(broker, tmp_path)
    git(broker, "update-ref", "refs/remotes/origin/main", "HEAD")
    (broker / "research/colab_cli/new.py").write_text("x=1")
    with pytest.raises(ValueError, match="working-tree"):
        qualification.check(broker, record)
    assert qualification.check(broker, record, "origin/main")


def test_watcher_reports_remote_tree_mismatch_once(broker, tmp_path):
    ops = tmp_path / "ops"
    ops.mkdir()
    job_log = tmp_path / "log path with spaces.log"
    job_log.write_text("EXIT=7\n")
    (ops / "watch-logs.txt").write_text(f"{job_log} mac-fixture\n")
    qualification.save(
        ops / "colab-cli-qualified.json",
        {
            "schema": "snake-dqn-colab-cli-qualified/v1",
            "tree_hash": "0" * 40,
        },
    )
    git(broker, "update-ref", "refs/remotes/origin/main", "HEAD")
    home = tmp_path / "home"
    binaries = home / ".local/bin"
    binaries.mkdir(parents=True)
    colab = binaries / "colab"
    colab.write_text('#!/bin/sh\nprintf mock-sessions >> "$HOME/mock-colab-used"\nexit 0\n')
    colab.chmod(0o755)
    env = dict(
        os.environ,
        HOME=str(home),
        OPS_DIR=str(ops),
        OPS_REPO=str(broker),
        OPS_WATCH_ONCE="1",
        OPS_WATCH_FETCH="0",
        PATH=f'{binaries}:{os.environ["PATH"]}',
    )
    subprocess.run(["bash", str(OPS / "watchd.sh")], env=env, check=True, timeout=10)
    assert "COLAB_CLI_UNQUALIFIED" in (ops / "events.log").read_text()
    assert (home / "mock-colab-used").read_text() == "mock-sessions"
    assert "JOB_ENDED mac-fixture EXIT=7" in (ops / "events.log").read_text()


def test_real_offline_plans_are_fresh_in_same_second(broker, tmp_path):
    # Copy only the transport; planning never executes Colab or touches study data.
    shutil.copytree(
        OPS.parents[1] / "research/colab_cli",
        broker / "research/colab_cli",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    git(broker, "add", ".")
    git(broker, "commit", "-qm", "planning transport")
    account = tmp_path / "account.json"
    account.write_text("{}")

    def make_plan():
        result = subprocess.check_output(
            [
                sys.executable,
                str(OPS / "requalify.py"),
                "plan",
                "--broker",
                str(broker),
                "--state-parent",
                str(tmp_path / "plans"),
                "--other-assignments",
                str(account),
                "--colab",
                "/never/run/colab",
            ],
            text=True,
        )
        return Path(result.strip())

    first = make_plan()
    second = make_plan()
    assert first != second
    for path in (first, second):
        review = json.loads(path.read_text())
        assert review["plan"]["profile"] == "final-only"
        assert review["plan"]["limits"]["max_units_including_reserve"] == 4.0
        assert all(
            not Path(json.loads(Path(j["manifest"]).read_text())["output_dir"]).exists()
            for j in review["plan"]["jobs"]
        )


def execution_fixture(broker, tmp_path, monkeypatch, *, passes=True):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "commit": git(broker, "rev-parse", "HEAD"),
                "output_dir": str(tmp_path / "fresh-output"),
            }
        )
    )
    saved = tmp_path / "ops-review.json"
    review = {
        "repo": str(broker),
        "commit": git(broker, "rev-parse", "HEAD"),
        "tree_hash": qualification.tree(broker),
        "state_dir": str(tmp_path / "state"),
        "colab": "/fake/colab",
        "plan": {
            "profile": "final-only",
            "commit": git(broker, "rev-parse", "HEAD"),
            "jobs": [{"manifest": str(manifest)}],
        },
        "manifest_hashes": {str(manifest): hashlib.sha256(manifest.read_bytes()).hexdigest()},
    }
    saved.write_text(json.dumps(review))
    calls = []

    def execute(plan, **kwargs):
        calls.append("execute")
        (tmp_path / "acceptance-result.json").write_text(json.dumps({"final_plan": plan}))
        return 0 if passes else 1

    def require(result, repo):
        calls.append("require_final_only")

    acceptance = SimpleNamespace(
        REVIEW_TOKEN="COLAB_CLI_ACCEPTANCE_REVIEWED",
        execute=execute,
        verify_plan_sources=lambda p, r: calls.append("verify_sources"),
    )
    monkeypatch.setattr(
        requalify,
        "load_transport",
        lambda repo: (acceptance, SimpleNamespace(require_final_only=require)),
    )
    return saved, manifest, calls


def test_execute_updates_only_after_verified_pass_and_cannot_repeat(broker, tmp_path, monkeypatch):
    saved, _, calls = execution_fixture(broker, tmp_path, monkeypatch)
    target = tmp_path / "qualified.json"
    requalify.execute(saved, target, "COLAB_CLI_ACCEPTANCE_REVIEWED", "/fake/python")
    assert qualification.check(broker, target)
    assert calls == ["verify_sources", "execute", "require_final_only", "verify_sources"]
    with pytest.raises(FileExistsError):
        requalify.execute(saved, target, "COLAB_CLI_ACCEPTANCE_REVIEWED", "/fake/python")
    assert calls.count("execute") == 1


def test_failed_acceptance_preserves_registry(broker, tmp_path, monkeypatch):
    saved, _, _ = execution_fixture(broker, tmp_path, monkeypatch, passes=False)
    target = tmp_path / "qualified.json"
    target.write_text("previous")
    with pytest.raises(ValueError, match="acceptance failed"):
        requalify.execute(saved, target, "COLAB_CLI_ACCEPTANCE_REVIEWED", "/fake/python")
    assert target.read_text() == "previous"


def test_changed_manifest_and_wrong_token_refused_before_execution(broker, tmp_path, monkeypatch):
    saved, manifest, calls = execution_fixture(broker, tmp_path, monkeypatch)
    target = tmp_path / "qualified.json"
    with pytest.raises(ValueError, match="REVIEWED"):
        requalify.execute(saved, target, "wrong", "/fake/python")
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        requalify.execute(saved, target, "COLAB_CLI_ACCEPTANCE_REVIEWED", "/fake/python")
    assert "execute" not in calls
    assert not target.exists()


def test_failed_source_bound_admission_preserves_registry(broker, tmp_path, monkeypatch):
    saved, _, _ = execution_fixture(broker, tmp_path, monkeypatch)
    acceptance, admission = requalify.load_transport(broker)

    def refuse(result, repo):
        raise ValueError("incomplete release evidence")

    admission.require_final_only = refuse
    monkeypatch.setattr(requalify, "load_transport", lambda repo: (acceptance, admission))
    target = tmp_path / "qualified.json"
    target.write_text("previous")
    with pytest.raises(ValueError, match="release evidence"):
        requalify.execute(saved, target, "COLAB_CLI_ACCEPTANCE_REVIEWED", "/fake/python")
    assert target.read_text() == "previous"
