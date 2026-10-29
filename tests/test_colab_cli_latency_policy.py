"""Deadline and clock-skew regressions from the live acceptance failures."""

from __future__ import annotations

import hashlib
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.colab_cli import cli_guard, supervisor
from research.colab_cli.acceptance_faults import AcceptanceObserver


@pytest.mark.parametrize("cap", ["deadline", "explicit"])
def test_telemetry_recovery_cannot_extend_real_or_explicit_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: str
) -> None:
    now = [0.0]
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    if cap != "explicit":
        setattr(cli, cap, 150.0)
    calls = []
    probes = []

    def invoke(argv: list[str], **kwargs: object) -> None:
        timeout = float(kwargs["timeout"])
        calls.append(timeout)
        now[0] += timeout
        raise subprocess.TimeoutExpired(argv, timeout)

    def idle(*args: object, **kwargs: object) -> bool:
        probes.append(kwargs["timeout"])
        assert cli.exec_uncertain
        return True

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        supervisor.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "kernel_idle", idle)
    with pytest.raises(supervisor.Refused, match="deadline reached during telemetry recovery"):
        cli.poll_rpc("result=True", **({"timeout": 150} if cap == "explicit" else {}))
    assert calls == [120, 28]
    assert probes == [30]
    assert now[0] == 150
    assert cli.exec_uncertain  # Exhaustion never clears uncertain kernel state.


def test_telemetry_attempt_does_not_shorten_independent_heartbeat_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At t=110 an in-flight 120s poll must still allow a 60s lease PUT."""
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    cli.deadline = 1000
    cli.priced_deadline = 900
    now = [0.0]
    allowances = []

    def invoke(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 120
        now[0] = 110
        worker = threading.Thread(target=lambda: allowances.append(cli.remaining_seconds(60)))
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        wrapped = Path(argv[argv.index("-f") + 1]).read_text()
        nonce = "SNAKE_RPC_" + wrapped.split("SNAKE_RPC_")[1][:32]
        return subprocess.CompletedProcess(argv, 0, nonce + '{"ok":true,"value":true}', "")

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    assert cli.poll_rpc("result=True") is True
    assert allowances == [60]
    assert cli.deadline == 1000 and cli.priced_deadline == 900


def test_telemetry_timeout_does_not_resubmit_without_kernel_idle_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    calls = []

    def invoke(argv: list[str], **kwargs: object) -> None:
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, float(kwargs["timeout"]))

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: False)
    with pytest.raises(supervisor.Refused, match="kernel remains busy/unknown"):
        cli.poll_rpc("result=True")
    assert len(calls) == 1
    assert cli.exec_uncertain


def test_watchdog_collects_during_expiry_interval_and_recovers_status_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow final renewal must not skip the short remote checkpoint window."""
    now = [0.0]
    payload = b"checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    reads = []
    calls = []

    class CLI:
        session = "owned"
        accounting = SimpleNamespace(endpoint="owned-endpoint")

        def stop_lease(self, *, wait: bool = False) -> None:
            assert wait
            calls.append("pause")

        def poll_rpc(self, code: str) -> bool:
            assert "renew=True" in code
            calls.append("renew")
            now[0] += 30  # Acknowledgement latency; actual expiry is t=210.
            return True

        def call(self, *args: str, timeout: float) -> str:
            assert calls == ["pause", "renew"]
            assert 0 < timeout <= 120
            reads.append(now[0])
            now[0] += 6
            if now[0] < 210:
                raise supervisor.Refused("404 not found")
            if now[0] > 225:
                raise AssertionError("collector skipped the runtime's checkpoint grace")
            Path(args[-1]).write_bytes(
                (digest + "\n").encode() if args[-2].endswith("sha256") else payload
            )
            return "ok"

        def sessions(self, *, timeout: float) -> dict[str, str]:
            assert timeout <= 120
            if "status-reset" not in calls:
                calls.append("status-reset")
                raise supervisor.Refused("HTTP 503 temporarily unavailable")
            return {}

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    observer = AcceptanceObserver("fake")
    monkeypatch.setattr(observer, "_state", lambda context: {})
    observer._watchdog(
        {
            "cli": CLI(),
            "manifest": {"lease_seconds": 180},
            "root": "/content/snake-cli/g",
            "generation": "g",
            "output": tmp_path,
        }
    )
    evidence = observer.records["mac-loss-watchdog-unassign"]
    assert evidence["assignment_disappeared_within_bound"] is True
    assert evidence["lease_expiry_uncertainty_seconds"] == 30
    assert evidence["observed_after_lease_seconds"] < 300
    assert reads[0] == 30  # Poll before expiry, never sleep past the remote grace.
    assert (tmp_path / "watchdog-checkpoint.pkl").read_bytes() == payload


@pytest.mark.parametrize("failure", ["timeout", "missing"])
@pytest.mark.parametrize("response_seconds", [1, 3])
def test_watchdog_slow_pre_expiry_read_cannot_consume_checkpoint_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, response_seconds: int
) -> None:
    """Replay warmup2's t=168 read spanning the entire t=180..195 grace."""
    now = [0.0]
    payload = b"checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    reads = []

    class CLI:
        session = "owned"
        accounting = SimpleNamespace(endpoint="owned-endpoint")

        def stop_lease(self, *, wait: bool = False) -> None:
            assert wait

        def poll_rpc(self, code: str) -> bool:
            assert "disable_contents=True" in code
            return True

        def call(self, *args: str, timeout: float) -> str:
            assert timeout > 0
            # Earlier missing reads have consumed 168s. A provider request
            # begun before publication stalls longer than the remaining grace.
            reads.append(now[0])
            if now[0] < 168:
                now[0] += min(1, timeout)
                raise supervisor.Refused("404 not found")
            if now[0] < 180:
                now[0] += min(40, timeout)
                if failure == "timeout":
                    raise subprocess.TimeoutExpired("download", timeout)
                raise supervisor.Refused("404 not found")
            now[0] += min(response_seconds, timeout)
            if response_seconds > timeout:
                raise subprocess.TimeoutExpired("download", timeout)
            if now[0] >= 195:
                raise supervisor.Refused("404 not found: runtime released")
            Path(args[-1]).write_bytes(
                (digest + "\n").encode() if args[-2].endswith("sha256") else payload
            )
            return "ok"

        def sessions(self, *, timeout: float) -> dict[str, str]:
            assert timeout > 0
            return {} if now[0] >= 195 else {"owned-endpoint": self.session}

    observer = AcceptanceObserver("unused")
    monkeypatch.setattr(observer, "_state", lambda _context: {})
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    observer._watchdog(
        {
            "cli": CLI(),
            "manifest": {"lease_seconds": 180},
            "root": "/content/snake-cli/g",
            "generation": "g",
            "output": tmp_path,
        }
    )
    evidence = observer.records["mac-loss-watchdog-unassign"]
    assert evidence["checkpoint_exported_during_grace"] is True
    assert evidence["assignment_disappeared_within_bound"] is True
    assert any(180 <= started < 195 for started in reads)
    assert (tmp_path / "watchdog-checkpoint.pkl").read_bytes() == payload
