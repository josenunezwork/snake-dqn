"""Round08 F5 W transport regressions, virtual time and synthetic bytes only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import acceptance, cli_guard, policy, supervisor, transfers


def _reply(argv: list[str], value: Any) -> subprocess.CompletedProcess[str]:
    code = Path(argv[argv.index("-f") + 1]).read_text()
    nonce = "SNAKE_RPC_" + code.split("SNAKE_RPC_")[1][:32]
    return subprocess.CompletedProcess(
        argv, 0, nonce + json.dumps({"ok": True, "value": value}), ""
    )


def test_verify_recovers_exec_and_idle_helper_timeouts_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real verify/poll/helper paths: 30s exec + 30s idle timeout isn't fatal."""
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = [0.0]
    events = []
    digest = hashlib.sha256(b"synthetic").hexdigest()

    def invoke(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 30
        assert not cli.exec_uncertain
        if not events:
            events.append("exec-timeout")
            now[0] += kwargs["timeout"]
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        assert events[-1] == "idle-proved"
        events.append("verify-replay")
        now[0] += 1
        return _reply(argv, {"sha256": digest, "size": 9})

    def helper(*_args: Any, timeout: float) -> dict[str, Any]:
        assert timeout == 30
        assert cli.exec_uncertain
        if events[-1] == "exec-timeout":
            events.append("idle-timeout")
            now[0] += timeout
            raise cli_guard.GuardError("helper failed: TimeoutExpired")
        events.append("idle-proved")
        now[0] += 1
        return {"idle": True}

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda n: now.__setitem__(0, now[0] + n))
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "_run_helper", helper)
    with cli.staging_controls(300):
        transfers._verify_remote_file(cli, "/content/synthetic", 9, digest, deadline=300)
    assert events == ["exec-timeout", "idle-timeout", "idle-proved", "verify-replay"]
    assert now[0] == 66
    assert not cli.exec_uncertain
    assert [r["status"] for r in cli.operation_records if r["operation"] == "recovery-idle"] == [
        "failed",
        "completed",
    ]


@pytest.mark.parametrize("cap", ["recovery", "staging", "deadline"])
def test_repeated_exec_and_idle_timeouts_keep_first_cap_and_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: str
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = [0.0]
    bound = policy.STAGING_RECOVERY_SECONDS if cap == "recovery" else 65.0
    if cap in {"deadline", "priced_deadline"}:
        setattr(cli, cap, bound)

    def invoke(argv: list[str], **kwargs: Any) -> None:
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    def helper(*_args: Any, timeout: float) -> dict[str, Any]:
        now[0] += timeout
        raise cli_guard.GuardError("helper failed: TimeoutExpired")

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda n: now.__setitem__(0, now[0] + n))
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "_run_helper", helper)
    with pytest.raises(supervisor.Refused, match="deadline|busy/unknown|cap"):
        with cli.staging_controls(bound if cap == "staging" else 300):
            cli.poll_rpc("result=True")
    assert now[0] <= bound
    assert cli.exec_uncertain
    assert len(cli.call_records) == 1  # Never replay without idle proof.


def test_helper_auth_refusal_is_permanent_and_never_clears_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    calls = []

    def invoke(argv: list[str], **kwargs: Any) -> None:
        calls.append("exec")
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    def helper(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("idle")
        raise cli_guard.GuardError("authentication denied")

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "_run_helper", helper)
    with pytest.raises(cli_guard.GuardError, match="authentication denied"):
        with cli.staging_controls(supervisor.time.monotonic() + 300):
            cli.poll_rpc("result=True")
    assert calls == ["exec", "idle"]
    assert cli.exec_uncertain


def test_slow_foreground_heartbeat_preserves_successful_put_and_verified_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real upload/renew path accepts a 37s heartbeat, with no data resubmission."""
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    source = tmp_path / "synthetic.bin"
    source.write_bytes(b"synthetic")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    now = [0.0]
    calls = []

    def invoke(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "upload" in argv:
            remote = argv[-1]
            heartbeat = remote.endswith("lease-heartbeat.json")
            calls.append("heartbeat" if heartbeat else "PUT")
            latency = 37.022 if heartbeat else 23.798
            assert kwargs["timeout"] >= latency
            if heartbeat:
                assert kwargs["timeout"] <= 60
            now[0] += latency
            return subprocess.CompletedProcess(argv, 0, "", "")
        calls.append("verify")
        now[0] += 1
        return _reply(argv, {"sha256": digest, "size": source.stat().st_size})

    def renew() -> None:
        # Match the production 30s throttle: a due post-PUT renewal only.
        if now[0] >= 23 and "heartbeat" not in calls:
            cli.renew_runtime_lease("/content/job", "generation")

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    with cli.staging_controls(300):
        assert (
            transfers.upload_file(cli, source, "/content/synthetic", renew, deadline=300) == digest
        )
    assert calls == ["PUT", "heartbeat", "verify"]
    assert now[0] == pytest.approx(61.82)
    assert cli.transfer_records[-1]["status"] == "verified"


@pytest.mark.parametrize("cap", ["staging", "deadline"])
def test_foreground_heartbeat_never_borrows_past_real_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: str
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = [0.0]
    if cap != "staging":
        setattr(cli, cap, 17)
    calls = []

    def invoke(argv: list[str], **kwargs: Any) -> None:
        calls.append(kwargs["timeout"])
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    with pytest.raises(subprocess.TimeoutExpired):
        with cli.staging_controls(17 if cap == "staging" else 300):
            cli.renew_runtime_lease("/content/job", "generation")
    assert calls == [17]
    assert now[0] == 17


def test_transport_repair_keeps_acceptance_bars() -> None:
    assert acceptance.D1024_TRANSFER_BAR_SECONDS == 150
    assert transfers.UPLOAD_TOTAL_TIMEOUT_SECONDS == 300
    assert policy.STAGING_CONTROL_SECONDS == 30
    assert transfers.CONTENTS_ATTEMPTS == 3
