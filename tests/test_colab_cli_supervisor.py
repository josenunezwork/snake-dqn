"""Subprocess integration tests for the Mac-side Colab CLI broker."""

from __future__ import annotations

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
                "rate": 6.84,
                "sessions": active,
                "exec_count": {},
            }
        )
    )
    log_path.touch()
    remote_path.mkdir()
    monkeypatch.setenv("FAKE_COLAB_STATE", str(state_path))
    monkeypatch.setenv("FAKE_COLAB_LOG", str(log_path))
    monkeypatch.setenv("FAKE_COLAB_REMOTE", str(remote_path))
    monkeypatch.setenv("FAKE_COLAB_MODE", mode)
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IXUSR)
    return str(FAKE)


def _events(tmp_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()]


def _assert_transport(tmp_path: Path) -> None:
    events = _events(tmp_path)
    assert events
    for event in events:
        assert event["auth"] == "oauth2"
        assert event["assign_retries"] == "0"
        assert event["request_timeout"] == "30"
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
    assert seen["env"]["COLAB_CLI_ASSIGN_RETRIES"] == "0"


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
