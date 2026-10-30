"""Offline regressions for the T34 pre-ready source-fetch hiccups."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from research.colab_cli import cli_guard, policy, supervisor
from tests import test_colab_cli_fault_properties as harness
from tests.test_colab_cli_supervisor import _manifest

_isolate_account = harness._isolate_account


@pytest.mark.parametrize("seconds", [45, 90, 140, 160, 170])
def test_source_fetch_gets_its_own_allowance_without_relaunch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolate_account: harness._TransportClock,
    seconds: int,
) -> None:
    """Real Broker/CLI subprocess path; slow source cannot release ready early."""
    binary = harness._fake_env(
        monkeypatch,
        tmp_path,
        {
            "max_latency_seconds": 170,
            "schedule": [
                {
                    "call": "exec",
                    "contains": "SNAKE_SOURCE_DELIVERY",
                    "fault": "latency",
                    "latency_seconds": seconds,
                }
            ],
        },
    )
    monkeypatch.setenv("FAKE_COLAB_MODE", "success")
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 10
    manifest["budget"]["unit_cap"] = 10
    ready = []
    broker = supervisor.Broker(
        tmp_path / "lanes", binary, observer=lambda event, context: ready.append(event)
    )
    if seconds <= policy.SOURCE_CONTROL_SECONDS:
        broker.run(manifest)
    else:
        with pytest.raises(supervisor.Refused, match="deadline reached during telemetry recovery"):
            broker.run(manifest)
    output = tmp_path / "job-output"
    receipt = json.loads((output / "receipt.json").read_text())
    measurement = json.loads((output / "upload-measurement.json").read_text())
    state = json.loads((tmp_path / "fake-state.json").read_text())
    calls = [c for c in receipt["cli_calls"] if c["operation"] == "source-fetch"]
    assert len(calls) == 1
    assert calls[0]["timeout_seconds"] == pytest.approx(policy.SOURCE_CONTROL_SECONDS)
    assert calls[0]["status"] == (
        "completed" if seconds <= policy.SOURCE_CONTROL_SECONDS else "failed"
    )
    assert measurement["source_delivery_seconds"] >= min(seconds, policy.SOURCE_CONTROL_SECONDS)
    assert measurement["elapsed_seconds"] <= 300
    assert receipt["status"] == (
        "PASS" if seconds <= policy.SOURCE_CONTROL_SECONDS else "INCOMPLETE"
    )
    assert receipt["stop_verified"] is True and receipt["cleanup_error"] is None
    assert state["allocation_count"] == 1 and state["sessions"] == {}
    assert list(state["job_launches"].values()) == [1]  # Watchdog, not scientific entrypoint.
    assert ("ready" in ready) == (seconds <= policy.SOURCE_CONTROL_SECONDS)
    assert len(state.get("source_fetches", [])) == (
        1 if seconds <= policy.SOURCE_CONTROL_SECONDS else 0
    )
    assert any(
        c["operation"] == "renewal-heartbeat"
        and c["status"] == "completed"
        and calls[0]["started_monotonic"] <= c["started_monotonic"] < calls[0]["ended_monotonic"]
        for c in receipt["cli_calls"]
    )
    assert all(
        c["timeout_seconds"] <= 30
        for c in receipt["cli_calls"]
        if c["command"] == "exec" and c["operation"] in {"mkdir", "verify", "assemble"}
    )


@pytest.mark.parametrize("cap", ["staging", "session", "explicit"])
def test_source_rpc_still_obeys_earliest_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cap: str
) -> None:
    from tests.test_colab_cli_supervisor import _patch_virtual_clock

    now = [100.0]
    _patch_virtual_clock(monkeypatch, now, advance_sleep=True)
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    cli.deadline = 120 if cap == "session" else 1000
    allowances = []

    def rpc(code: str, timeout: float, **kwargs: object) -> bool:
        allowances.append(timeout)
        now[0] += timeout
        cli.exec_uncertain = True
        raise TimeoutError("uncertain source exec timed out")

    monkeypatch.setattr(cli, "rpc", rpc)
    with cli.staging_controls(120 if cap == "staging" else 400):
        assert cli.remaining_seconds(200) == (20 if cap in {"session", "staging"} else 30)
        with pytest.raises(supervisor.Refused, match="cap reached|deadline reached"):
            with cli.staging_controls(400, source_delivery=True):
                cli.poll_rpc("source", timeout=20 if cap == "explicit" else None)
        assert cli.staging_control_seconds() == 30
    assert allowances == [20]
    assert now[0] == 120 and cli.exec_uncertain


def test_source_scope_is_thread_local_and_restores_short_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 100.0)
    allowances = []
    with cli.staging_controls(400):
        assert cli.remaining_seconds(200) == 30
        with cli.staging_controls(400, source_delivery=True):
            assert cli.remaining_seconds(200) == policy.SOURCE_CONTROL_SECONDS
            worker = threading.Thread(target=lambda: allowances.append(cli.remaining_seconds(60)))
            worker.start()
            worker.join(timeout=2)
            assert not worker.is_alive()
        assert cli.remaining_seconds(200) == 30
    assert cli.remaining_seconds(200) == 200 and allowances == [60]


@pytest.mark.parametrize("fallback", [False, True])
def test_source_recovers_early_failure_and_bounds_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _isolate_account: harness._TransportClock,
    fallback: bool,
) -> None:
    """Idle proof permits recovery; fallback shares the source-specific bound."""
    idle_proofs = []
    original_idle = cli_guard.kernel_idle

    def idle(*args: object, **kwargs: object) -> bool:
        result = original_idle(*args, **kwargs)
        idle_proofs.append(result)
        return result

    monkeypatch.setattr(cli_guard, "kernel_idle", idle)
    schedule = [
        {"call": "exec", "contains": "SNAKE_SOURCE_DELIVERY", "fault": "reset"},
        {
            "call": "exec",
            "contains": "mod.install_fallback" if fallback else "mod.fetch",
            "fault": "latency",
            "latency_seconds": 140,
        },
    ]
    binary = harness._fake_env(
        monkeypatch, tmp_path, {"max_latency_seconds": 140, "schedule": schedule}
    )
    monkeypatch.setenv("FAKE_COLAB_MODE", "github_unreachable" if fallback else "success")
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 10
    manifest["budget"]["unit_cap"] = 10
    supervisor.Broker(tmp_path / "lanes", binary).run(manifest)
    receipt = json.loads((tmp_path / "job-output" / "receipt.json").read_text())
    measurement = json.loads((tmp_path / "job-output" / "upload-measurement.json").read_text())
    state = json.loads((tmp_path / "fake-state.json").read_text())
    calls = [c for c in receipt["cli_calls"] if c["operation"].startswith("source-")]
    assert calls[0]["status"] == "failed"
    assert calls[-1]["status"] == "completed"
    assert all(c["timeout_seconds"] <= policy.SOURCE_CONTROL_SECONDS for c in calls)
    assert idle_proofs == [True]
    assert any(c["operation"] == "source-install_fallback" for c in calls) == fallback
    assert receipt["status"] == "PASS" and receipt["stop_verified"] is True
    assert receipt["cleanup_error"] is None
    assert measurement["source_delivery_seconds"] >= 140
    assert measurement["elapsed_seconds"] <= 300
    assert state["allocation_count"] == 1 and state["sessions"] == {}
    assert len(state["source_fetches"]) == (3 if fallback else 1)
    assert list(state["job_launches"].values()) == [1]
