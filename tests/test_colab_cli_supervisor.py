"""Subprocess integration tests for the Mac-side Colab CLI broker."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from research.colab_cli import supervisor

REPO = Path(__file__).resolve().parents[1]
FAKE = REPO / "tests" / "fixtures" / "fake_colab.py"


def _fake_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, mode: str = "normal", assignments: int = 0
) -> str:
    state_path = tmp_path / "fake-state.json"
    log_path = tmp_path / "fake-calls.jsonl"
    remote_path = tmp_path / "remote"
    active = {}
    for index in range(assignments):
        active[f"gpu-notebook-{index + 1}"] = {
            "name": f"owner-notebook-{index + 1}",
            "gpu": "A100" if index == 0 else "L4",
        }
    state_path.write_text(
        json.dumps(
            {
                "balance": 1178.30,
                "rate": 6.84 if assignments else 0,
                "sessions": active,
                "exec_count": {},
            }
        )
    )
    log_path.touch()
    remote_path.mkdir()
    stub = tmp_path / "cli-stub" / "colab_cli"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("")
    (stub / "client.py").write_text(
        "class Client:\n    def assign(self,*a,**k): pass\n"
        "    def _issue_request(self,*a,**k): pass\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(stub.parent))
    monkeypatch.setenv("FAKE_COLAB_STATE", str(state_path))
    monkeypatch.setenv("FAKE_COLAB_LOG", str(log_path))
    monkeypatch.setenv("FAKE_COLAB_REMOTE", str(remote_path))
    monkeypatch.setenv("FAKE_COLAB_MODE", mode)
    monkeypatch.setenv(
        "SNAKE_COLAB_OAUTH_PYTHON", str(REPO / "tests/fixtures/fake_colab_helper.py")
    )
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IXUSR)
    return str(FAKE)


def _events(tmp_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()]


def _assert_transport(tmp_path: Path) -> None:
    events = _events(tmp_path)
    assert events
    for event in events:
        assert event["auth"] == "oauth2"
        assert event["assign_retries"] is None
        assert event["request_timeout"] is None
        argv = event["argv"]
        assert isinstance(argv, list)
        assert argv[:4] == ["--auth", "oauth2", "--config", event["config"]]
        assert event["config"]


def _manifest(
    tmp_path: Path,
    *,
    output: Path | None = None,
    job_id: str = "fake-job",
    gpu: str = "cpu",
    rate: float = 6.84,
) -> dict[str, object]:
    commit = os.environ.get("SNAKE_TEST_PINNED_COMMIT")
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    os.environ["FAKE_COLAB_OWN_RATE"] = str(rate)
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"pinned-input")
    return {
        "job_id": job_id,
        "commit": commit,
        "script": "research/colab_cli/acceptance_probe.sh",
        "requirements": {"gpu": gpu, "min_vcpu": 1, "est_minutes": 1.0},
        "timeout_minutes": 3.0,
        "budget": {"rate_units_per_hour": rate, "unit_cap": 1.0, "reserve_units": 0.0},
        "output_dir": str(output or tmp_path / "job-output"),
        "env": {"FAKE_JOB": "1"},
        "inputs": [
            {
                "path": str(input_path),
                "dest": "input.bin",
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            }
        ],
        "lease_seconds": 180,
        "max_export_bytes": 256 * 1024 * 1024,
    }


@pytest.fixture(autouse=True)
def _isolate_account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(supervisor, "ACCOUNT_ROOT", tmp_path / "account")
    # The implementation deliberately obtains runtime/bootstrap from the
    # manifest commit.  During this uncommitted test checkout, make an
    # unreachable temporary commit object containing those two files so the
    # integration tests exercise the same pinned-bundle path.
    index = tmp_path / "git-index"
    git_env = os.environ.copy()
    git_env["GIT_INDEX_FILE"] = str(index)
    subprocess.run(["git", "read-tree", "HEAD"], cwd=REPO, env=git_env, check=True)
    subprocess.run(
        [
            "git",
            "add",
            "research/colab_cli/runtime.py",
            "research/colab_cli/bootstrap.sh",
            "research/colab_cli/release.py",
            "research/colab_cli/source.py",
            "research/colab_cli/acceptance_probe.sh",
        ],
        cwd=REPO,
        env=git_env,
        check=True,
    )
    subprocess.run(
        ["git", "update-index", "--chmod=+x", "research/colab_cli/acceptance_probe.sh"],
        cwd=REPO,
        env=git_env,
        check=True,
    )
    tree = subprocess.check_output(["git", "write-tree"], cwd=REPO, env=git_env, text=True).strip()
    commit = subprocess.check_output(
        [
            "git",
            "commit-tree",
            tree,
            "-p",
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        ],
        cwd=REPO,
        env=git_env,
        input="fake pinned test bundle\n",
        text=True,
    ).strip()
    monkeypatch.setenv("GIT_INDEX_FILE", str(index))
    monkeypatch.setenv("SNAKE_TEST_PINNED_COMMIT", commit)


def test_allocation_cap_counts_existing_notebook_assignments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, assignments=3)
    manifest = _manifest(tmp_path)

    with pytest.raises(supervisor.Refused, match="account concurrency cap 3|assignment cap 3"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    events = _events(tmp_path)
    assert not any(event["argv"][4] == "new" for event in events)
    _assert_transport(tmp_path)


def test_normal_job_stops_and_verifies_backend_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)

    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    assert receipt["endpoint"]
    assert supervisor.CLI(binary, tmp_path / "status.json").sessions() == {}
    assert any(event["argv"][4] == "stop" for event in _events(tmp_path))
    _assert_transport(tmp_path)


def test_renewal_after_initial_staging_deadline_uses_fresh_contents_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A credential rotation after five minutes must not inherit staging's cap.

    This exercises the real broker through the subprocess-backed fake binary.
    ``ready`` is emitted only after the bounded initial input staging phase, so
    advancing the monotonic clock here reproduces an otherwise healthy runtime
    that needs a later credential renewal.
    """
    binary = _fake_env(monkeypatch, tmp_path, mode="export_success")
    monotonic = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: monotonic[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 10.0
    manifest["budget"]["unit_cap"] = 2.0
    renewed: list[bool] = []

    def observer(event: str, context: dict[str, object]) -> None:
        if event != "ready":
            return
        cli = context["cli"]
        # Stage uploads had their 300-second collective deadline at t=0. A
        # now-expired credential forces the later direct ContentsClient PUT.
        monotonic[0] = 301.0
        cli.release_credential["expires_at"] = 0
        renewed.append(True)

    supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)

    assert renewed == [True]
    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    credential_puts = [
        row for row in receipt["contents_transfers"] if "-release.json" in row["remote"]
    ]
    assert len(credential_puts) >= 2  # initial credential plus the post-staging renewal
    assert credential_puts[-1]["status"] == "completed"
    assert credential_puts[-1]["bytes_per_second"] > 0
    logical_download = next(row for row in receipt["transfers"] if row["direction"] == "download")
    raw_download = next(
        row for row in receipt["contents_transfers"] if row["direction"] == "download"
    )
    assert logical_download["status"] == "verified"
    assert logical_download["bytes_per_second"] > 0
    assert raw_download["status"] == "completed"
    assert raw_download["bytes_per_second"] > 0
    assert all("total upload time cap" not in str(row) for row in receipt["transfers"])


@pytest.mark.parametrize("baseline", [False, True])
def test_l4_observed_rate_below_priced_ceiling_calibrates_and_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, baseline: bool
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, assignments=int(baseline))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path, gpu="L4", rate=2.0)
    if baseline:
        manifest["budget"]["other_assignments"] = {"gpu-notebook-1": 6.84}
    monkeypatch.setenv("FAKE_COLAB_OWN_RATE", "1.54")
    # Account-wide samples lag before settling at baseline + own real rate.
    other = 6.84 if baseline else 0
    monkeypatch.setenv("FAKE_COLAB_USAGE_RATES", json.dumps([other, other + 1.54]))
    from research.colab_cli.accounting import AccountTracker

    original_read = AccountTracker.calibration_read

    def locked_read(self: AccountTracker, *args: object) -> float | None:
        # A second lane cannot read the temporary uncalibrated active record.
        with pytest.raises(supervisor.Refused, match="already supervised"):
            with supervisor.locked(supervisor.ACCOUNT_ROOT / "broker.lock", blocking=False):
                pass
        return original_read(self, *args)

    monkeypatch.setattr(AccountTracker, "calibration_read", locked_read)
    checked = []

    def observer(event: str, context: dict[str, object]) -> None:
        if event == "runtime_started":
            with supervisor.locked(supervisor.ACCOUNT_ROOT / "broker.lock", blocking=False):
                active = next(supervisor.ACCOUNT_ROOT.glob("active-*.json"))
                assert json.loads(active.read_text())[
                    "observed_rate_units_per_hour"
                ] == pytest.approx(1.54)
                checked.append(True)

    supervisor.Broker(tmp_path / "lanes", binary, observer=observer).run(manifest)
    assert checked == [True]
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    accounting = receipt["accounting"]
    assert accounting["priced_rate_units_per_hour"] == 2.0
    assert accounting["observed_rate_units_per_hour"] == pytest.approx(1.54)
    assert accounting["calibration"]["status"] == "PASS"
    assert len(accounting["calibration"]["samples"]) == 4
    assert (
        accounting["last_crosscheck"]["own_elapsed_units"]
        >= accounting["last_crosscheck"]["observed_elapsed_units"]
    )


@pytest.mark.parametrize(
    ("rates", "error"),
    [
        ([2.06], "exceeds priced upper bound"),
        ([0.0], "did not stabilize"),
        ([1.54, 1.7] * 13, "did not stabilize"),
        ([1.54, 1.54, 1.54, 1.7], "rate drift"),
    ],
)
def test_rate_calibration_and_later_drift_fail_closed_with_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rates: list[float], error: str
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path, gpu="L4", rate=2.0)
    monkeypatch.setenv("FAKE_COLAB_OWN_RATE", "1.54")
    monkeypatch.setenv("FAKE_COLAB_USAGE_RATES", json.dumps(rates))
    original_lock = supervisor.account_locked

    @contextlib.contextmanager
    def checked_lock(cli: supervisor.CLI, *, cleanup: bool = False):
        with original_lock(cli, cleanup=cleanup):
            try:
                yield
            finally:
                # Even on calibration failure, cleanup removes the uncalibrated
                # record before another lane is allowed to poll its account.
                for active in supervisor.ACCOUNT_ROOT.glob("active-*.json"):
                    assert (
                        json.loads(active.read_text())["observed_rate_units_per_hour"] is not None
                    )

    monkeypatch.setattr(supervisor, "account_locked", checked_lock)
    with pytest.raises(supervisor.Refused, match=error):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["stop_verified"] is True
    assert not json.loads((tmp_path / "fake-state.json").read_text())["sessions"]
    events = _events(tmp_path)
    assert sum(event["argv"][4] == "new" for event in events) == 1
    assert any(event["argv"][4] == "stop" for event in events)
    calibration = receipt["accounting"]["calibration"]
    assert calibration["status"] == ("PASS" if error == "rate drift" else "INCOMPLETE")
    assert calibration["samples"]
    _assert_transport(tmp_path)


def test_provider_calibration_timeout_is_a_real_elapsed_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    manifest = _manifest(tmp_path, gpu="L4", rate=2.0)
    monkeypatch.setenv("FAKE_COLAB_USAGE_RATES", "[0]")
    now = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        supervisor.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    with pytest.raises(supervisor.Refused, match="did not stabilize"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    assert receipt["stop_verified"] is True
    assert receipt["accounting"]["calibration"]["status"] == "INCOMPLETE"
    assert receipt["elapsed_seconds"] == 121  # One watchdog-start second, then 120 calibration.


def test_stop_success_exit_is_rejected_when_backend_keeps_assignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="stop_lie")
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)

    with pytest.raises(supervisor.Refused, match="STOP_UNVERIFIED|INCOMPLETE"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["stop_verified"] is False
    assert supervisor.CLI(binary, tmp_path / "status.json").sessions()
    _assert_transport(tmp_path)


def test_kernel_error_with_zero_cli_exit_is_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="kernel_error")
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "fake-session")

    with pytest.raises(supervisor.Refused, match="kernel error despite CLI exit zero"):
        cli.rpc("result=True")

    _assert_transport(tmp_path)


def test_busy_kernel_is_never_resubmitted_after_uncertain_exec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="busy_exec")
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "fake-session")

    with pytest.raises(supervisor.Refused, match="kernel completion not acknowledged"):
        cli.rpc("result=True")
    with pytest.raises(supervisor.Refused, match="uncertain/busy"):
        cli.rpc("result=True")

    events = _events(tmp_path)
    assert sum(event["argv"][4] == "exec" for event in events) == 1
    _assert_transport(tmp_path)


def test_corrupt_download_is_rejected_before_atomic_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="corrupt_download")
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "fake-session")
    cli.call("new", "-s", "fake-session")

    generation_root = "/content/snake-cli/fake-generation"
    remote_file = (
        Path(os.environ["FAKE_COLAB_REMOTE"])
        / "fake-session"
        / generation_root.lstrip("/")
        / "output"
    )
    payload = b"GOOD"
    remote_file.mkdir(parents=True)
    (remote_file / "checkpoint.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (remote_file / "checkpoint.bin.sha256").write_text(digest + "\n")
    (remote_file / "checkpoint.bin.complete").write_text("complete\n")
    entry = {"path": "checkpoint.bin", "size": len(payload), "sha256": digest}

    with pytest.raises(supervisor.Refused, match="download hash mismatch"):
        supervisor.export_file(cli, generation_root, entry, tmp_path / "published", 1024)

    assert not (tmp_path / "published" / "exports" / "checkpoint.bin").exists()
    assert (
        not list((tmp_path / "published").rglob("*.complete"))
        if (tmp_path / "published").exists()
        else True
    )
    _assert_transport(tmp_path)


def test_existing_output_directory_is_refused_as_nonfresh(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(supervisor.Refused, match="fresh Mac directory"):
        supervisor.validate(_manifest(tmp_path, output=output))


def test_h100_chain_allocates_once_stops_once_and_writes_release_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    first_output = tmp_path / "h100-first"
    second_output = tmp_path / "h100-second"
    manifests = [
        _manifest(tmp_path, output=first_output, job_id="h100-first", gpu="H100"),
        _manifest(tmp_path, output=second_output, job_id="h100-second", gpu="H100"),
    ]

    supervisor.Broker(tmp_path / "lanes", binary).chain(
        manifests, retention_minutes=7, retention_units=1
    )

    events = _events(tmp_path)
    assert sum(event["argv"][4] == "new" for event in events) == 1
    assert sum(event["argv"][4] == "stop" for event in events) == 1
    first_receipt = json.loads((first_output / "receipt.json").read_text())
    second_receipt = json.loads((second_output / "receipt.json").read_text())
    assert first_receipt["status"] == "COMPLETED_SESSION_RETAINED"
    assert second_receipt["status"] == "COMPLETED_SESSION_RETAINED"
    assert first_receipt["session"] == second_receipt["session"]
    for output in (first_output, second_output):
        release = list(output.glob("session-release.json"))
        assert len(release) == 1
        evidence = json.loads(release[0].read_text())
        assert evidence["stop_verified"] is True
        assert evidence["error"] is None
    assert supervisor.CLI(binary, tmp_path / "status.json").sessions() == {}
    _assert_transport(tmp_path)


@pytest.mark.parametrize("include_endpoint", [True, False])
def test_recover_pending_and_active_records_stops_owned_orphan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, include_endpoint: bool
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    session = "orphan-session"
    config = tmp_path / "orphan-sessions.json"
    cli = supervisor.CLI(binary, config, session)
    cli.call("new", "-s", session)
    endpoint = next(iter(cli.sessions()))
    output = tmp_path / "recovered-output"
    output.mkdir()
    supervisor.ACCOUNT_ROOT.mkdir(parents=True, exist_ok=True)
    pending = supervisor.ACCOUNT_ROOT / f"pending-{session}.json"
    pending.write_text(
        json.dumps(
            {
                "session": session,
                "config": str(config),
                "endpoint": endpoint if include_endpoint else None,
                "output": str(output),
            }
        )
    )
    active = supervisor.ACCOUNT_ROOT / f"active-{session}.json"
    active.write_text(json.dumps({"session": session, "endpoint": endpoint}))

    supervisor.Broker(tmp_path / "lanes", binary).recover(pending)

    assert not pending.exists()
    assert not active.exists()
    evidence = list(output.glob("recovery-*.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text())["stop_verified"] is True
    assert supervisor.CLI(binary, config, session).sessions() == {}
    _assert_transport(tmp_path)


def test_partial_local_export_without_completion_marker_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "unused")
    published = tmp_path / "published" / "exports"
    target = published / "checkpoint.bin"
    target.parent.mkdir(parents=True)
    payload = b"already-downloaded"
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    Path(str(target) + ".sha256").write_text(digest + "\n")

    with pytest.raises(
        supervisor.Refused, match="immutable export changed|local export publication incomplete"
    ):
        supervisor.export_file(
            cli,
            "/content/snake-cli/generation",
            {"path": "checkpoint.bin", "size": len(payload), "sha256": digest},
            tmp_path / "published",
            1024,
        )

    assert not Path(str(target) + ".complete").exists()
    assert not _events(tmp_path)


def test_cli_error_redacts_token_bearing_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="error_url")
    cli = supervisor.CLI(binary, tmp_path / "lane-sessions.json", "unused")

    with pytest.raises(supervisor.Refused) as error:
        cli.call("version")

    message = str(error.value)
    assert "secret-token" not in message
    assert "refresh-secret" not in message
    _assert_transport(tmp_path)


def test_module_entrypoint_validate_status_and_run_with_fake_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    manifest = _manifest(tmp_path, output=tmp_path / "entry-output")
    manifest_path = tmp_path / "entry-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    state_dir = tmp_path / "entry-state"
    prefix = [
        str(Path(sys.executable)),
        "-m",
        "research.colab_cli.supervisor",
        "--state-dir",
        str(state_dir),
        "--colab",
        binary,
    ]
    env = os.environ.copy()

    validated = subprocess.run(
        [*prefix, "validate", str(manifest_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validated.returncode == 0, validated.stderr
    assert not _events(tmp_path)

    status = subprocess.run(
        [*prefix, "status"], cwd=REPO, env=env, capture_output=True, text=True, check=False
    )
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["sessions"] == {}

    ran = subprocess.run(
        [*prefix, "run", str(manifest_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    receipt = json.loads((tmp_path / "entry-output" / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    _assert_transport(tmp_path)


def test_shared_reserved_budget_refuses_before_new_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    supervisor.ACCOUNT_ROOT.mkdir(parents=True, exist_ok=True)
    (supervisor.ACCOUNT_ROOT / "active-other-lane.json").write_text(
        json.dumps({"unit_cap": 1178.3, "reserve_units": 0.0})
    )
    manifest = _manifest(tmp_path)

    with pytest.raises(supervisor.Refused, match="insufficient balance"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    events = _events(tmp_path)
    assert not any(event["argv"][4] == "new" for event in events)
    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert "insufficient balance" in receipt["error"]
    _assert_transport(tmp_path)


def test_cli_call_caps_subprocess_timeout_at_session_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake-colab", tmp_path / "lane-sessions.json", "session")
    cli.deadline = 100.0
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 99.5)
    seen: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["args"] = args
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="Colab CLI 0.7.4\n", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    assert cli.call("version", timeout=45) == "Colab CLI 0.7.4\n"
    assert seen["timeout"] == pytest.approx(0.5)
    assert seen["args"] == [
        "fake-colab",
        "--auth",
        "oauth2",
        "--config",
        str(tmp_path / "lane-sessions.json"),
        "version",
    ]
    assert seen["env"]["SNAKE_COLAB_NOTEBOOK_HASH"]


def test_cli_call_caps_subprocess_timeout_at_priced_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake-colab", tmp_path / "lane-sessions.json", "session")
    cli.deadline = 1000.0
    cli.priced_deadline = 101.25
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 100.0)
    seen: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="Colab CLI 0.7.4\n", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    assert cli.call("usage", timeout=45) == "Colab CLI 0.7.4\n"
    assert seen["timeout"] == pytest.approx(1.25)


def test_upload_phase_caps_outer_rpc_grace_and_restores_session_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake-colab", tmp_path / "lane-sessions.json", "session")
    cli.deadline = 1000.0
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 100.0)
    seen = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, stdout="no kernel acknowledgment", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    with pytest.raises(supervisor.Refused, match="not acknowledged"):
        with cli.limit_deadline(103.0):
            cli.rpc("result=True", timeout=60)
    assert seen == [3.0]
    assert cli.deadline == 1000.0


def test_backup_is_invoked_only_after_verified_remote_stop_and_receipt_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)
    manifest["backup_dir"] = str(tmp_path / "backup")
    order: list[str] = []

    def backup(output: Path, backup_dir: str, job: str, generation: str) -> None:
        assert supervisor.CLI(binary, tmp_path / "post-stop.json").sessions() == {}
        order.append("backup-after-stop")
        destination = Path(backup_dir) / job / generation
        destination.mkdir(parents=True)
        (destination / "snapshot.bin").write_bytes(b"verified")
        digest = hashlib.sha256(b"verified").hexdigest()
        (destination / "snapshot.bin.sha256").write_text(digest + "\n")
        (destination / "snapshot.bin.complete").write_text("complete\n")
        (output / "backup.json").write_text(
            json.dumps({"destination": str(destination), "hash_verified": True})
        )

    monkeypatch.setattr(supervisor.Broker, "backup", staticmethod(backup))
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)

    assert order == ["backup-after-stop"]
    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    backup_record = json.loads((tmp_path / "job-output" / "backup.json").read_text())
    assert backup_record["hash_verified"] is True
    _assert_transport(tmp_path)


def test_stop_reissues_after_transient_lookup_error_with_long_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="stop_transient")
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    cli = supervisor.CLI(binary, tmp_path / "session.json", "owned-stop")
    cli.call("new", "-s", cli.session)
    endpoint = next(iter(cli.sessions()))
    timeouts = []
    original = cli.call

    def observe(*args: str, timeout: float = 45) -> str:
        if args[0] == "stop":
            timeouts.append(timeout)
        return original(*args, timeout=timeout)

    monkeypatch.setattr(cli, "call", observe)
    supervisor.stop_verified(cli, endpoint)
    assert len(timeouts) >= 2
    assert all(timeout >= 60 for timeout in timeouts)
    assert not cli.sessions()


def test_idempotent_poll_recovers_transport_hiccup_only_after_idle_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="poll_hiccup")
    cli = supervisor.CLI(binary, tmp_path / "session.json", "telemetry")
    assert cli.poll_rpc("result=True") is True
    assert sum(event["argv"][4] == "exec" for event in _events(tmp_path)) == 2


def test_other_notebook_spend_is_attributed_and_missing_rates_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, assignments=1)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    manifest = _manifest(tmp_path)
    with pytest.raises(supervisor.Refused, match="unaccounted"):
        supervisor.Broker(tmp_path / "missing-lane", binary).run(manifest)
    assert not any(event["argv"][4] == "new" for event in _events(tmp_path))
    manifest = _manifest(tmp_path, job_id="accounted", output=tmp_path / "accounted-output")
    manifest["budget"]["other_assignments"] = {"gpu-notebook-1": 6.84}
    supervisor.Broker(tmp_path / "accounted-lane", binary).run(manifest)
    receipt = json.loads((tmp_path / "accounted-output/receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["accounting"]["baseline_rate_units_per_hour"] == 6.84
    assert list(supervisor.CLI(binary, tmp_path / "status.json").sessions().values()) == [
        "owner-notebook-1"
    ]


def test_recover_unknown_assignment_by_durable_notebook_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    cli = supervisor.CLI(binary, tmp_path / "session.json", "interrupted")
    cli.call("new", "-s", cli.session)
    state = json.loads((tmp_path / "fake-state.json").read_text())
    endpoint = next(iter(state["sessions"]))
    state["sessions"][endpoint]["name"] = "?"
    (tmp_path / "fake-state.json").write_text(json.dumps(state))
    cli.config.unlink()
    output = tmp_path / "recovery-output"
    output.mkdir()
    supervisor.ACCOUNT_ROOT.mkdir(exist_ok=True)
    record = supervisor.ACCOUNT_ROOT / "pending-interrupted.json"
    record.write_text(
        json.dumps(
            {
                "session": cli.session,
                "config": str(cli.config),
                "output": str(output),
                "before": [],
                "notebook_hash": cli.notebook_hash,
            }
        )
    )
    supervisor.Broker(tmp_path / "recovery-state", binary).recover(record)
    assert cli.sessions() == {}
    assert not record.exists()
    assert sum(event["argv"][4] == "new" for event in _events(tmp_path)) == 1


def test_600_mib_snapshot_roundtrip_uses_bounded_fake_cli_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli import transfers

    binary = _fake_env(monkeypatch, tmp_path)
    cli = supervisor.CLI(binary, tmp_path / "large-session.json", "large-transfer")
    # Initial staging completes at t=0.  The later snapshot transfer starts
    # after the old five-minute staging cap and takes another ten simulated
    # minutes; its own size-scaled deadline must govern it.
    clock = [0.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    staging = tmp_path / "initial-staging.bin"
    staging.write_bytes(b"staged")
    transfers.upload_file(
        cli, staging, "/content/snake-cli/transfer-proof/staged.bin", lambda: None
    )
    clock[0] = 301.0
    real_call = cli.call

    def slow_upload(*args: str, **kwargs: object) -> str:
        result = real_call(*args, **kwargs)
        if args[0] == "upload":
            clock[0] += 2.0
        return result

    cli.call = slow_upload  # type: ignore[method-assign]
    source = tmp_path / "600mib.snapshot"
    size = 600 * 1024 * 1024
    with source.open("wb") as stream:
        stream.truncate(size)
    remote = "/content/snake-cli/transfer-proof/checkpoint.bin"
    digest = transfers.upload_file(cli, source, remote, renew=lambda: None)
    target = tmp_path / "downloaded.snapshot"
    transfers.download_file(cli, remote, target, size, expected_sha256=digest)
    assert target.stat().st_size == size
    assert supervisor.sha256(target) == digest
    later_upload = cli.transfer_records[-2]
    assert later_upload["direction"] == "upload"
    assert later_upload["elapsed_seconds"] == pytest.approx(600.0)
    assert later_upload["bytes_per_second"] > 0
    downloaded = cli.transfer_records[-1]
    assert downloaded["direction"] == "download"
    assert downloaded["bytes_per_second"] > 0
    assert any(
        record["direction"] == "download" and record["bytes_per_second"] > 0
        for record in cli.contents_transfers
    )
    uploads = [e for e in _events(tmp_path) if e["argv"][4] == "upload"]
    downloads = [e for e in _events(tmp_path) if e["argv"][4] == "download"]
    assert len(uploads) == 301  # One staged control file, then 300 snapshot chunks.
    assert len(downloads) == 75
    assert max(e["upload_bytes"] for e in uploads) <= 2 * 1024 * 1024
    assert not list((tmp_path / "remote").rglob("*.part"))


def test_chunk_retry_overwrites_partial_put_through_fake_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research.colab_cli import transfers

    binary = _fake_env(monkeypatch, tmp_path, mode="upload_transient")
    cli = supervisor.CLI(binary, tmp_path / "session.json", "retry-transfer")
    source = tmp_path / "input.payload"
    source.write_bytes(b"immutable-input" * 1000)
    monkeypatch.setattr(transfers, "UPLOAD_CHUNK_BYTES", 2048)
    digest = transfers.upload_file(cli, source, "/content/input.bin", renew=lambda: None)
    remote = tmp_path / "remote" / cli.session / "content/input.bin"
    assert remote.read_bytes() == source.read_bytes()
    assert supervisor.sha256(remote) == digest
    uploads = [event for event in _events(tmp_path) if event["argv"][4] == "upload"]
    assert uploads[0]["argv"][-1] == uploads[1]["argv"][-1]
    assert len(uploads) == 9
    assert max(event["upload_bytes"] for event in uploads) <= 2048


@pytest.mark.parametrize(
    "mode,transport", [("normal", "github"), ("github_unreachable", "fallback")]
)
def test_runtime_source_delivery_with_fake_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, transport: str
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode=mode)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    manifest = _manifest(tmp_path)
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    output = Path(manifest["output_dir"])
    delivery = json.loads((output / "source-delivery.json").read_text())
    assert delivery["transport"] == transport
    assert delivery["commit"] == manifest["commit"]
    assert (
        delivery["tree"]
        == subprocess.check_output(
            ["git", "rev-parse", f"{manifest['commit']}^{{tree}}"], text=True
        ).strip()
    )
    state = json.loads((tmp_path / "fake-state.json").read_text())
    assert len(state["source_fetches"]) == 1
    assert state["source_fetches"][0][-1] == manifest["commit"]
    assert state["source_fetches"][0][-2] == "https://github.com/josenunezwork/snake-dqn.git"
    archives = [
        event
        for event in _events(tmp_path)
        if event["argv"][4] == "upload" and event["argv"][-1].endswith("source.tar.gz")
    ]
    assert len(archives) == (1 if transport == "fallback" else 0)
    assert (output / "staging/source.tar.gz").stat().st_size < 2 * 1024 * 1024
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["stop_verified"] is True
    _assert_transport(tmp_path)


def test_github_identity_mismatch_refuses_without_uploading_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = _fake_env(monkeypatch, tmp_path, mode="github_identity_mismatch")
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    manifest = _manifest(tmp_path)
    with pytest.raises(supervisor.Refused, match="identity"):
        supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((Path(manifest["output_dir"]) / "receipt.json").read_text())
    assert receipt["stop_verified"] is True
    assert not any(
        event["argv"][4] == "upload" and event["argv"][-1].endswith("source.tar.gz")
        for event in _events(tmp_path)
    )


def test_waiting_for_account_lock_renews_active_runtime_lease(tmp_path: Path) -> None:
    import threading

    supervisor.ACCOUNT_ROOT.mkdir(parents=True, exist_ok=True)
    cli = supervisor.CLI("unused", tmp_path / "session.json")
    renewed = threading.Event()
    acquired = threading.Event()
    cli.renew = renewed.set

    def wait_for_broker() -> None:
        with supervisor.account_locked(cli):
            acquired.set()

    with supervisor.locked(supervisor.ACCOUNT_ROOT / "broker.lock"):
        waiter = threading.Thread(target=wait_for_broker)
        waiter.start()
        assert renewed.wait(timeout=5)
        assert not acquired.is_set()
    waiter.join(timeout=5)
    assert acquired.is_set()
    assert not waiter.is_alive()


def test_waiting_for_calibration_lock_still_enforces_priced_unit_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("unused", tmp_path / "session.json")
    cli.deadline = 2000
    cli.priced_deadline = 1800
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 1800)
    with supervisor.locked(supervisor.ACCOUNT_ROOT / "broker.lock"):
        with pytest.raises(supervisor.Refused, match="priced session unit cap"):
            with supervisor.account_locked(cli):
                pytest.fail("contended broker lock was unexpectedly acquired")


def test_account_lock_uses_new_run_cap_instead_of_previous_retained_tracker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    cli = supervisor.CLI("unused", tmp_path / "session.json")
    cli.deadline = 2100
    cli.priced_deadline = 2000
    cli.accounting = SimpleNamespace(started_monotonic=0, own_rate_units_per_hour=2, unit_cap=0.1)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 1800)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    calls = []

    def contended_once(handle: object, mode: int) -> None:
        calls.append(mode)
        if len(calls) == 1:
            raise BlockingIOError

    monkeypatch.setattr(supervisor.fcntl, "flock", contended_once)
    with supervisor.account_locked(cli):
        pass
    assert len(calls) == 3


def test_cleanup_lock_wait_ignores_expired_execution_and_uncertain_renewal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("unused", tmp_path / "session.json")
    cli.deadline = -1

    def unsafe_renew() -> None:
        raise AssertionError("cleanup must not renew an uncertain kernel")

    cli.renew = unsafe_renew
    calls = []

    def contended_once(handle: object, mode: int) -> None:
        calls.append(mode)
        if len(calls) == 1:
            raise BlockingIOError

    monkeypatch.setattr(supervisor.fcntl, "flock", contended_once)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    with supervisor.account_locked(cli, cleanup=True):
        pass
    assert len(calls) == 3


def test_disconnect_acceptance_observes_fake_backend_endpoint_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from research.colab_cli.acceptance_faults import AcceptanceObserver

    binary = _fake_env(monkeypatch, tmp_path)
    cli = supervisor.CLI(binary, tmp_path / "session.json", "disconnect")
    cli.call("new", "-s", cli.session)
    endpoint = next(iter(cli.sessions()))
    cli.accounting = SimpleNamespace(endpoint=endpoint)
    root = "/content/snake-cli/disconnect"
    remote = Path(os.environ["FAKE_COLAB_REMOTE"]) / cli.session / root.lstrip("/") / "output"
    remote.mkdir(parents=True)
    payload = b"checkpoint-on-lease-loss"
    (remote / "lease-checkpoint.pkl").write_bytes(payload)
    (remote / "lease-checkpoint.pkl.sha256").write_text(hashlib.sha256(payload).hexdigest())
    clock = [0.0]

    def disconnect_sleep(seconds: float) -> None:
        clock[0] += seconds
        state_path = tmp_path / "fake-state.json"
        state = json.loads(state_path.read_text())
        state["sessions"].pop(endpoint, None)
        state_path.write_text(json.dumps(state))

    monkeypatch.setattr(supervisor.time, "sleep", disconnect_sleep)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    observer = AcceptanceObserver("unused")
    monkeypatch.setattr(observer, "_state", lambda context: {})
    output = tmp_path / "watchdog-output"
    output.mkdir()
    observer._watchdog(
        {"cli": cli, "root": root, "output": output, "manifest": {"lease_seconds": 180}}
    )
    assert observer.records["mac-loss-watchdog-unassign"]["endpoint"] == endpoint
    assert not any(event["argv"][4] == "stop" for event in _events(tmp_path))
