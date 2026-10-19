"""Offline contracts for the bounded, opt-in CLI acceptance harness."""

from __future__ import annotations

import hashlib
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
    assert len(plan["gates"]) == 11
    for job in plan["jobs"]:
        payload = json.loads(Path(job["manifest"]).read_text())
        assert payload["requirements"]["gpu"] in {"cpu", "L4"}
        assert payload["timeout_minutes"] >= 2 * payload["requirements"]["est_minutes"]
        assert not Path(payload["output_dir"]).exists()
        supervisor.validate(payload, repo)


def test_plan_carries_reviewed_other_assignment_rates(tmp_path: Path) -> None:
    repo = _repo_with_probe(tmp_path)
    mapping = {"notebook-endpoint": 1.5, "other-worker": 7.0}
    plan = acceptance.build_plan(repo, tmp_path / "state", 1.0, mapping)["plan"]
    for job in plan["jobs"]:
        payload = json.loads(Path(job["manifest"]).read_text())
        assert payload["budget"]["other_assignments"] == mapping


def test_other_assignment_file_rejects_nonpositive_or_nonobject_rates(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('["not-a-map"]')
    with pytest.raises(ValueError, match="JSON object"):
        acceptance.load_other_assignments(invalid)
    invalid.write_text('{"endpoint": 0}')
    with pytest.raises(ValueError, match="positive"):
        acceptance.load_other_assignments(invalid)


def test_release_receipt_rejects_cleanup_error_even_with_backend_stop(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stop_verified": True,
                "cleanup_error": "stop transport failed",
            }
        )
    )
    with pytest.raises(supervisor.Refused, match="cleanup error"):
        acceptance._clean_release_receipt(tmp_path)


def test_launch_check_routes_acceptance_through_guarded_launcher() -> None:
    launcher = Path(__file__).parents[1] / "research" / "colab_cli" / "launch.sh"
    result = subprocess.run(
        [str(launcher), "acceptance", "--check", "--state-dir", "/tmp/acceptance-state"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "research.colab_cli.acceptance" in result.stdout


def test_l4_watchdog_requires_backend_disappearance_within_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The L4 fault has to see its own session vanish before cleanup can pass."""
    from research.colab_cli.acceptance_faults import AcceptanceObserver

    checkpoint = b"acceptance-checkpoint"
    digest = hashlib.sha256(checkpoint).hexdigest()

    class CLI:
        session = "acceptance-session"

        class accounting:
            endpoint = "endpoint"

        def __init__(self) -> None:
            self.polls = 0

        def call(self, command: str, *_args: str, timeout: int) -> str:
            assert command == "download"
            destination = Path(_args[-1])
            if _args[-2].endswith(".sha256"):
                destination.write_text(f"{digest}  lease-checkpoint.pkl\n")
            else:
                destination.write_bytes(checkpoint)
            return ""

        def sessions(self, *, timeout: int) -> dict[str, str]:
            self.polls += 1
            return {"endpoint": self.session} if self.polls == 1 else {}

    observer = AcceptanceObserver("unused")
    cli = CLI()
    monkeypatch.setattr(observer, "_state", lambda _context: {})
    now = [0.0]
    monkeypatch.setattr("research.colab_cli.acceptance_faults.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "research.colab_cli.acceptance_faults.time.sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    output = tmp_path / "output"
    output.mkdir()
    observer._watchdog(
        {
            "cli": cli,
            "manifest": {"lease_seconds": 180},
            "output": output,
            "root": "/content/acceptance",
        }
    )
    record = observer.records["mac-loss-watchdog-unassign"]
    assert record["assignment_disappeared_within_bound"] is True
    assert record["release_bound_seconds"] == 300


def test_l4_watchdog_rejects_question_mark_session_when_endpoint_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend endpoint remains paid even when its CLI session label is `?`."""
    from research.colab_cli.acceptance_faults import AcceptanceObserver

    checkpoint = b"acceptance-checkpoint"
    digest = hashlib.sha256(checkpoint).hexdigest()

    class CLI:
        session = "acceptance-session"

        class accounting:
            endpoint = "endpoint"

        @staticmethod
        def call(command: str, *_args: str, timeout: int) -> str:
            assert command == "download"
            destination = Path(_args[-1])
            if _args[-2].endswith(".sha256"):
                destination.write_text(f"{digest}  lease-checkpoint.pkl\n")
            else:
                destination.write_bytes(checkpoint)
            return ""

        @staticmethod
        def sessions(*, timeout: int) -> dict[str, str]:
            return {"endpoint": "?"}

    now = [0.0]
    observer = AcceptanceObserver("unused")
    monkeypatch.setattr(observer, "_state", lambda _context: {})
    monkeypatch.setattr(
        "research.colab_cli.acceptance_faults.time.sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    monkeypatch.setattr("research.colab_cli.acceptance_faults.time.monotonic", lambda: now[0])
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(supervisor.Refused, match="did not disappear"):
        observer._watchdog(
            {
                "cli": CLI(),
                "manifest": {"lease_seconds": 180},
                "output": output,
                "root": "/content/acceptance",
            }
        )


def test_accounting_gate_aggregates_cpu_and_l4_admission_once() -> None:
    """A sequential acceptance must not reject L4 as a duplicate gate."""
    from research.colab_cli.acceptance_faults import AcceptanceObserver

    class Tracker:
        @staticmethod
        def accounted_assignments() -> dict[str, object]:
            return {
                "assignments": [{"endpoint": "notebook", "rate_units_per_hour": 1.5}],
                "unaccounted_endpoints": [],
            }

    class CLI:
        accounting = Tracker()

    observer = AcceptanceObserver("unused")
    observer._accounted_assignments(CLI(), "cpu")
    observer._accounted_assignments(CLI(), "L4")
    assert [entry["arm"] for entry in observer.records["account-assignment-spend"]["arms"]] == [
        "cpu",
        "L4",
    ]


def test_state_reconciles_session_spend_not_raw_shared_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Other lanes can debit far more than this arm's cap without stopping it."""
    from research.colab_cli.acceptance_faults import AcceptanceObserver

    class Tracker:
        called = False

        def check(self, usage: dict[str, float], **kwargs: object) -> dict[str, float]:
            self.called = True
            assert usage["balance"] == 1.0  # 99 units of other-work debit.
            assert kwargs["sessions"] == {"ours": "acceptance"}
            return {"own_elapsed_units": 0.1}

    class CLI:
        session = "acceptance"
        accounting = Tracker()

        @staticmethod
        def rpc(*_args: str) -> dict[str, object]:
            return {"runtime": {}, "probe": {}}

        @staticmethod
        def usage() -> dict[str, float]:
            return {"balance": 1.0, "rate": 100.0, "assignments": 1.0}

        @staticmethod
        def sessions() -> dict[str, str]:
            return {"ours": "acceptance"}

    monkeypatch.setattr(supervisor, "ACCOUNT_ROOT", tmp_path / "account")
    result = AcceptanceObserver._state(
        {
            "cli": CLI(),
            "root": "/content/acceptance",
            "generation": "generation",
            "manifest": {"budget": {"unit_cap": 1.0, "other_assignments": {}}},
        }
    )
    assert result["probe"] == {}
    assert CLI.accounting.called is True


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
