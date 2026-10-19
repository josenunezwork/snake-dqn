"""Offline contracts for the bounded, opt-in CLI acceptance harness."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.colab_cli import acceptance, acceptance_probe, supervisor


def _repo_with_probe(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    target = repo / "research" / "colab_cli"
    target.mkdir(parents=True)
    source = Path(__file__).parents[1] / "research" / "colab_cli"
    for name in ("acceptance_probe.py", "acceptance_probe.sh"):
        shutil.copy2(source / name, target / name)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "-qm",
            "probe",
        ],
        cwd=repo,
        check=True,
    )
    return repo


def test_plan_requires_known_cpu_price_and_keeps_job_output_fresh(tmp_path: Path) -> None:
    repo = _repo_with_probe(tmp_path)
    with pytest.raises(ValueError, match="CPU price is unknown"):
        acceptance.build_plan(repo, tmp_path / "state", None)

    record = acceptance.build_plan(repo, tmp_path / "state", 1.0)
    plan = record["plan"]
    assert plan["status"] == "NOT_RUN"
    assert plan["limits"]["usable_units"] == 3.2
    assert len(plan["gates"]) == 10
    for job in plan["jobs"]:
        payload = json.loads(Path(job["manifest"]).read_text())
        assert payload["requirements"]["gpu"] in {"cpu", "L4"}
        assert payload["timeout_minutes"] >= 2 * payload["requirements"]["est_minutes"]
        assert not Path(payload["output_dir"]).exists()
        supervisor.validate(payload, repo)


def test_manifest_rejects_tree_sha_in_place_of_pinned_commit(tmp_path: Path) -> None:
    repo = _repo_with_probe(tmp_path)
    plan = acceptance.build_plan(repo, tmp_path / "state", 1.0)["plan"]
    payload = json.loads(Path(plan["jobs"][0]["manifest"]).read_text())
    payload["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
    ).strip()
    with pytest.raises(supervisor.Refused, match="Git commit object"):
        supervisor.validate(payload, repo)


def test_probe_writes_verified_exports_and_toy_resume_is_not_trainer_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acceptance_probe.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(acceptance_probe, "benchmark", lambda _iterations: 0.01)
    monkeypatch.setattr(acceptance_probe, "await_control", lambda _root, _name: None)
    output = tmp_path / "output"
    acceptance_probe.suite(output, silence_seconds=61, snapshot_bytes=1024 * 1024)

    for artifact in output.glob("*"):
        if artifact.suffix not in {".sha256", ".complete"}:
            assert artifact.with_name(artifact.name + ".sha256").is_file()
            assert artifact.with_name(artifact.name + ".complete").is_file()
    resume = json.loads((output / "toy-resume.json").read_text())
    assert resume["exact"] is True
    assert "not evidence for a trainer" in resume["scope"]
    assert (output / "intentional-child-failure.json").read_text().find("23") >= 0
    assert (output / "checkpoint-overhead.json").is_file()


def test_probe_accepts_only_empty_runtime_created_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acceptance_probe.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(acceptance_probe, "benchmark", lambda _iterations: 0.01)
    monkeypatch.setattr(acceptance_probe, "await_control", lambda _root, _name: None)
    output = tmp_path / "runtime-created"
    output.mkdir()
    acceptance_probe.suite(output, silence_seconds=61, snapshot_bytes=1)
    with pytest.raises(FileExistsError):
        acceptance_probe.suite(output, silence_seconds=61, snapshot_bytes=1)


def test_resume_entrypoint_reloads_exported_toy_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acceptance_probe.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(acceptance_probe, "benchmark", lambda _iterations: 0.01)
    monkeypatch.setattr(acceptance_probe, "await_control", lambda _root, _name: None)
    output = tmp_path / "output"
    acceptance_probe.suite(output, silence_seconds=61, snapshot_bytes=1)
    resumed = tmp_path / "resumed.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(acceptance_probe.__file__)),
            "--resume-input",
            str(output / "toy-checkpoint.pkl"),
            "--resume-output",
            str(resumed),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    expected = json.loads((output / "toy-resume.json").read_text())
    actual = json.loads(resumed.read_text())
    assert actual["resumed_sha256"] == expected["uninterrupted_sha256"]


def test_cpu_failure_writes_final_result_and_does_not_spend_l4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_probe(tmp_path)
    record = acceptance.build_plan(repo, tmp_path / "state", 1.0)
    calls: list[str] = []

    class Observer:
        records: dict[str, dict[str, object]] = {}

        def __init__(self, _oauth_python: str) -> None:
            self.records = {}

        def write(self, _output: Path) -> None:
            return None

    class BrokenBroker:
        def __init__(self, _state: Path, _binary: str, observer: object) -> None:
            self.observer = observer

        def run(self, payload: dict[str, object]) -> None:
            calls.append(str(payload["requirements"]["gpu"]))
            raise supervisor.Refused("synthetic CPU admission failure")

    monkeypatch.setattr(acceptance, "AcceptanceObserver", Observer)
    monkeypatch.setattr(acceptance.supervisor, "Broker", BrokenBroker)
    rc = acceptance.execute(
        record["plan"],
        state_dir=tmp_path / "lanes",
        colab="fake-colab",
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python="fake-python",
    )
    assert rc == 1
    assert calls == ["cpu"]
    result_path = Path(record["run_dir"]) / "acceptance-result.json"
    assert json.loads(result_path.read_text())["final_plan"]["status"] == "OBSERVED_NOT_QUALIFIED"
