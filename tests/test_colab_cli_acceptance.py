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
    for path in source.iterdir():
        if path.is_file() and path.suffix in {".py", ".sh"}:
            shutil.copy2(path, target / path.name)
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


def test_plan_defaults_to_conservative_cpu_price_and_keeps_job_output_fresh(tmp_path: Path) -> None:
    repo = _repo_with_probe(tmp_path)
    record = acceptance.build_plan(repo, tmp_path / "state", None)
    plan = record["plan"]
    assert plan["status"] == "NOT_RUN"
    assert plan["limits"]["usable_units"] == 3.2
    assert len(plan["gates"]) == 14
    assert plan["profile"] == "final-only"
    assert {
        g["id"] for g in plan["gates"] if g["status"] == "NOT_REQUIRED"
    } == acceptance.FULL_STATE_GATES
    for job in plan["jobs"]:
        payload = json.loads(Path(job["manifest"]).read_text())
        assert payload["requirements"]["gpu"] in {"cpu", "L4"}
        assert payload["timeout_minutes"] >= 2 * payload["requirements"]["est_minutes"]
        assert not Path(payload["output_dir"]).exists()
        supervisor.validate(payload, repo)
    cpu = json.loads(Path(plan["jobs"][0]["manifest"]).read_text())
    l4 = json.loads(Path(plan["jobs"][1]["manifest"]).read_text())
    assert cpu["budget"]["rate_units_per_hour"] == 0.5
    assert l4["budget"]["rate_units_per_hour"] == 2.0


def test_plan_accepts_overridable_conservative_rate_ceilings(tmp_path: Path) -> None:
    repo = _repo_with_probe(tmp_path)
    plan = acceptance.build_plan(repo, tmp_path / "state", 0.75, l4_rate=2.5)["plan"]
    assert plan["jobs"][0]["priced_rate_units_per_hour"] == 0.75
    assert plan["jobs"][1]["priced_rate_units_per_hour"] == 2.5
    assert plan["expected_costs"]["cpu_20_minutes_at_priced_rate"] == 0.25
    assert plan["expected_costs"]["l4_30_minutes_at_priced_rate"] == 1.25


def test_calibration_evidence_requires_a_complete_provider_measurement() -> None:
    incomplete = acceptance._calibration_evidence({"accounting": {}})
    assert incomplete["status"] == "INCOMPLETE"

    complete = acceptance._calibration_evidence(
        {
            "accounting": {
                "calibration": {
                    "status": "PASS",
                    "priced_rate_units_per_hour": 2.0,
                    "observed_rate_units_per_hour": 1.54,
                    "samples": [1.54, 1.54, 1.54],
                }
            }
        }
    )
    assert complete["status"] == "PASS"
    assert complete["observed_rate_units_per_hour"] == 1.54


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


def test_cold_kernel_recovery_gate_requires_both_arms_and_first_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The observer clears only local metadata, then uses the real guard boundary."""
    from research.colab_cli import acceptance_faults, cli_guard
    from research.colab_cli.acceptance_faults import AcceptanceObserver

    calls: list[list[str]] = []
    guard_calls: list[tuple[object, ...]] = []
    metadata = {
        "kernel_id": "known-kernel",
        "endpoint": "assigned-endpoint",
        "url": "https://runtime.example/",
        "token_sha256": "a" * 64,
    }

    def run(argv: list[str], **_kwargs: object) -> object:
        calls.append(argv)
        marker = argv[2]
        if "SNAKE_ACCEPTANCE_KERNEL_METADATA_CLEAR" in marker:
            return type("Result", (), {"returncode": 0, "stdout": json.dumps(metadata)})()
        assert "SNAKE_ACCEPTANCE_KERNEL_METADATA_SNAPSHOT" in marker
        return type("Result", (), {"returncode": 0, "stdout": json.dumps(metadata)})()

    def idle(*args: object, **_kwargs: object) -> bool:
        guard_calls.append(args)
        return True

    class CLI:
        config = tmp_path / "sessions.json"
        session = "acceptance-session"
        contents_transfers = [
            {
                "direction": "upload",
                "remote": "/content/release-credential.json",
                "size_bytes": 128,
                "elapsed_seconds": 0.5,
                "status": "completed",
            }
        ]

        @staticmethod
        def remaining_seconds(allowance: float) -> float:
            assert allowance == 150
            return 47

    monkeypatch.setattr(acceptance_faults.subprocess, "run", run)
    monkeypatch.setattr(cli_guard, "kernel_idle", idle)
    observer = AcceptanceObserver("fake-oauth-python")
    for arm in ("cpu", "L4"):
        context = {"cli": CLI(), "manifest": {"requirements": {"gpu": arm}}, "output": tmp_path}
        observer("runtime_ready", context)
        assert observer.records["cold-kernel-metadata-recovery"]["status"] == "INCOMPLETE"
        observer("first_upload", context)
    record = observer.records["cold-kernel-metadata-recovery"]
    assert record["status"] == "PASS"
    assert [arm["arm"] for arm in record["arms"]] == ["cpu", "L4"]
    assert all(arm["metadata_recovery"]["same_kernel_id"] for arm in record["arms"])
    assert len(calls) == 4
    assert len(guard_calls) == 2
    assert all(call[0] == "fake-oauth-python" and call[1] == "-c" for call in calls)
    assert all(
        args[0] == "fake-oauth-python" and args[2] == "acceptance-session" for args in guard_calls
    )


def test_cold_kernel_recovery_refuses_metadata_identity_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli import acceptance_faults, cli_guard
    from research.colab_cli.acceptance_faults import AcceptanceObserver

    responses = iter(
        [
            {
                "kernel_id": "known",
                "endpoint": "endpoint",
                "url": "https://one/",
                "token_sha256": "a" * 64,
            },
            {
                "kernel_id": "known",
                "endpoint": "endpoint",
                "url": "https://two/",
                "token_sha256": "a" * 64,
            },
        ]
    )

    def run(_argv: list[str], **_kwargs: object) -> object:
        return type("Result", (), {"returncode": 0, "stdout": json.dumps(next(responses))})()

    class CLI:
        config = tmp_path / "sessions.json"
        session = "acceptance-session"

        @staticmethod
        def remaining_seconds(_allowance: float) -> float:
            return 47

    monkeypatch.setattr(acceptance_faults.subprocess, "run", run)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *_args, **_kwargs: True)
    with pytest.raises(supervisor.Refused, match="changed tracked identity"):
        AcceptanceObserver("fake-oauth-python")._recover_cleared_kernel_metadata(CLI(), "cpu")


def test_first_upload_accepts_only_retry_attempts_for_its_one_remote(tmp_path: Path) -> None:
    from research.colab_cli.acceptance_faults import AcceptanceObserver

    observer = AcceptanceObserver("unused")
    observer.records["cold-kernel-metadata-recovery"] = {
        "status": "INCOMPLETE",
        "arms": [{"arm": "cpu", "metadata_recovery": {}}],
    }

    class CLI:
        contents_transfers = [
            {
                "direction": "upload",
                "remote": "/content/release-credential.json",
                "size_bytes": None,
                "elapsed_seconds": 0.1,
                "status": "failed",
            },
            {
                "direction": "upload",
                "remote": "/content/release-credential.json",
                "size_bytes": 128,
                "elapsed_seconds": 0.5,
                "status": "completed",
            },
        ]

    observer._verify_first_upload(CLI(), "cpu")
    evidence = observer.records["cold-kernel-metadata-recovery"]["arms"][0]["first_upload_transfer"]
    assert evidence["attempts"] == 2

    observer = AcceptanceObserver("unused")
    observer.records["cold-kernel-metadata-recovery"] = {
        "status": "INCOMPLETE",
        "arms": [{"arm": "cpu", "metadata_recovery": {}}],
    }
    CLI.contents_transfers[0]["remote"] = "/content/another-upload.json"
    with pytest.raises(supervisor.Refused, match="first upload ledger is incomplete"):
        observer._verify_first_upload(CLI(), "cpu")
    CLI.contents_transfers[0]["remote"] = CLI.contents_transfers[-1]["remote"]
    CLI.contents_transfers = [CLI.contents_transfers[0]] * 3 + [CLI.contents_transfers[-1]]
    with pytest.raises(supervisor.Refused, match="contents transfer ledger"):
        observer._verify_first_upload(CLI(), "cpu")


def test_l4_watchdog_requires_backend_disappearance_within_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The L4 fault has to see its own session vanish before cleanup can pass."""
    from research.colab_cli.acceptance_faults import AcceptanceObserver

    checkpoint = b"acceptance-checkpoint"
    digest = hashlib.sha256(checkpoint).hexdigest()

    class CLI:
        session = "acceptance-session"

        @staticmethod
        def stop_lease(*, wait: bool = False) -> None:
            assert wait

        @staticmethod
        def poll_rpc(code: str) -> bool:
            return True

        class accounting:
            endpoint = "endpoint"

        def __init__(self) -> None:
            self.polls = 0

        def call(self, command: str, *_args: str, timeout: int) -> str:
            now[0] = max(now[0], 180)
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
            "generation": "g",
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

        @staticmethod
        def stop_lease(*, wait: bool = False) -> None:
            assert wait

        @staticmethod
        def poll_rpc(code: str) -> bool:
            return True

        class accounting:
            endpoint = "endpoint"

        @staticmethod
        def call(command: str, *_args: str, timeout: int) -> str:
            now[0] = max(now[0], 180)
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
                "generation": "g",
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
        def poll_rpc(*_args: str) -> dict[str, object]:
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
    acceptance_probe.suite(
        output, silence_seconds=61, snapshot_bytes=1024 * 1024, profile="full-state"
    )

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
    acceptance_probe.suite(output, silence_seconds=61, snapshot_bytes=1, profile="full-state")
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
    monkeypatch.setattr(supervisor, "REPO", repo)
    calls: list[str] = []

    class Observer:
        records: dict[str, dict[str, object]] = {}

        def __init__(self, _oauth_python: str, profile: str = "final-only") -> None:
            self.records = {}

        def _record(self, name: str, **evidence: object) -> None:
            self.records[name] = dict(evidence)

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


@pytest.mark.parametrize("include_calibration", [True, False])
def test_expected_l4_watchdog_refusal_records_or_rejects_provider_rate_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_calibration: bool
) -> None:
    """The expected watchdog exception still needs its own measured rate evidence."""
    repo = _repo_with_probe(tmp_path)
    record = acceptance.build_plan(repo, tmp_path / "state", 0.5)
    monkeypatch.setattr(supervisor, "REPO", repo)

    class Observer:
        def __init__(self, _oauth_python: str, profile: str = "final-only") -> None:
            self.records: dict[str, dict[str, object]] = {}

        def _record(self, name: str, **evidence: object) -> None:
            self.records[name] = dict(evidence)

        def write(self, _output: Path) -> None:
            return None

    class WatchdogBroker:
        def __init__(self, _state: Path, _binary: str, observer: Observer) -> None:
            self.observer = observer

        def run(self, payload: dict[str, object]) -> None:
            output = Path(str(payload["output_dir"]))
            output.mkdir()
            calibration: dict[str, object] = (
                {
                    "status": "PASS",
                    "priced_rate_units_per_hour": 2.0,
                    "observed_rate_units_per_hour": 1.54,
                    "samples": [1.54, 1.54, 1.54],
                }
                if include_calibration
                or payload["requirements"] == {"gpu": "cpu", "min_vcpu": 2, "est_minutes": 10.0}
                else {}
            )
            (output / "receipt.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "stop_verified": True,
                        "accounting": {"calibration": calibration} if calibration else {},
                    }
                )
            )
            if payload["requirements"] == {"gpu": "L4", "min_vcpu": 2, "est_minutes": 15.0}:
                self.observer.records["mac-loss-watchdog-unassign"] = {
                    "assignment_disappeared_within_bound": True
                }
                raise supervisor.Refused("expected watchdog renewal refusal")

    monkeypatch.setattr(acceptance, "AcceptanceObserver", Observer)
    monkeypatch.setattr(acceptance.supervisor, "Broker", WatchdogBroker)
    rc = acceptance.execute(
        record["plan"],
        state_dir=tmp_path / "lanes",
        colab="fake-colab",
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python="fake-python",
    )
    result = json.loads((Path(record["run_dir"]) / "acceptance-result.json").read_text())
    l4 = result["final_plan"]["execution"][1]
    assert l4["priced_rate_units_per_hour"] == (2.0 if include_calibration else None)
    assert l4["measured_rate_units_per_hour"] == (1.54 if include_calibration else None)
    assert l4["status"] == ("PASS" if include_calibration else "INCOMPLETE")
    # This focused double does not fabricate every acceptance gate, so the
    # overall qualification stays incomplete even when the L4 arm passes.
    assert rc == 1


@pytest.mark.parametrize("profile", ["final-only", "full-state"])
def test_profile_inventory_and_plan_binding(tmp_path: Path, profile: str) -> None:
    repo = _repo_with_probe(tmp_path)
    plan = acceptance.build_plan(repo, tmp_path / "state", profile=profile)["plan"]
    acceptance.verify_plan_sources(plan, repo)
    omitted = {gate["id"] for gate in plan["gates"] if not gate["required"]}
    assert omitted == (
        acceptance.FULL_STATE_GATES if profile == "final-only" else {"final-artifact-download"}
    )
    for row in plan["jobs"]:
        env = json.loads(Path(row["manifest"]).read_text())["env"]
        assert env["COLAB_ACCEPTANCE_PROFILE"] == profile
    (repo / "research/colab_cli/acceptance_probe.py").write_text("# changed\n")
    with pytest.raises(supervisor.Refused, match="source changed"):
        acceptance.verify_plan_sources(plan, repo)
    plan["source_manifest"] = acceptance.source_manifest(repo)
    with pytest.raises(supervisor.Refused, match="differs from pinned commit"):
        acceptance.verify_plan_sources(plan, repo)


@pytest.mark.parametrize("arm,expected_bytes", [("cpu", 4096), ("L4", 64 * 1024 * 1024)])
def test_final_only_probe_has_no_full_state_or_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arm: str, expected_bytes: int
) -> None:
    monkeypatch.setenv("COLAB_ACCEPTANCE_ARM", arm)
    monkeypatch.setattr(acceptance_probe.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(acceptance_probe, "await_control", lambda _root, _name: None)
    monkeypatch.setattr(
        acceptance_probe, "benchmark", lambda _iterations: pytest.fail("full-state benchmark ran")
    )
    output = tmp_path / "output"
    acceptance_probe.suite(output, silence_seconds=61, snapshot_bytes=600 * 1024 * 1024)
    name = "final-artifact.bin" if arm == "L4" else "client-job.bin"
    assert (output / name).stat().st_size == expected_bytes
    assert not (output / "busy-snapshot.bin").exists()
    assert not (output / "toy-checkpoint.pkl").exists()
    assert not (output / "checkpoint-overhead.json").exists()


def test_wait_file_timeout_names_remote_phase_and_elapsed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli.acceptance_faults import AcceptanceObserver

    now = [10.0]
    monkeypatch.setattr("research.colab_cli.acceptance_faults.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "research.colab_cli.acceptance_faults.time.sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    cli = supervisor.CLI("unused", tmp_path / "config", "session")
    cli.deadline, cli.priced_deadline = 30, 20

    def missing(*_args: object, **_kwargs: object) -> None:
        raise supervisor.Refused("404 not found")

    monkeypatch.setattr(cli, "call", missing)
    with pytest.raises(
        supervisor.Refused, match="client-loss completion.*client-finished.json.*10.000s"
    ):
        AcceptanceObserver._wait_file(
            cli,
            "/content/client-finished.json",
            tmp_path / "marker",
            50,
            phase="client-loss completion",
        )
    assert now[0] == 20
