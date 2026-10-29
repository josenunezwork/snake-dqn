"""Offline multi-minute transport outages on one live, owned fake assignment."""

from __future__ import annotations

import json
import uuid

import pytest

from research.colab_cli import cli_guard, policy, supervisor
from tests import test_colab_cli_fault_properties as harness

_isolate_account = harness._isolate_account


def owned_cli(monkeypatch, tmp_path, failures):
    binary = harness._fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": 29,
            "max_latency_seconds": 120,
            "schedule": [
                {
                    "call": "exec",
                    "contains": "TRANSPORT_OUTAGE_READ",
                    "fault": "timeout",
                    "phase": "pre",
                }
            ]
            * failures,
        },
    )
    cli = supervisor.CLI(binary, tmp_path / "config.json", "owned")
    cli.deadline = 3600
    cli.call("new", "-s", cli.session)
    cli.poll_recovery_endpoint = json.loads(cli.config.read_text())[cli.session]["endpoint"]
    cli.start_lease("/content/outage", "generation", 180)
    return cli


@pytest.mark.parametrize("failure_count", [4, 8, 50])
def test_live_owned_exec_outage_recovers_or_expires(
    monkeypatch, tmp_path, _isolate_account, failure_count
):
    clock = _isolate_account
    cli = owned_cli(monkeypatch, tmp_path, failure_count)
    before = clock.now
    try:
        if failure_count < 50:
            assert cli.poll_rpc("# TRANSPORT_OUTAGE_READ\nresult=True") is True
            assert clock.now - before > 300
            assert not cli.exec_uncertain
        else:
            with pytest.raises(
                supervisor.Refused, match="deadline reached during telemetry recovery"
            ):
                cli.poll_rpc("# TRANSPORT_OUTAGE_READ\nresult=True")
            assert clock.now - before == pytest.approx(1200, abs=1)
        events = [
            json.loads(line) for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()
        ]
        state = json.loads((tmp_path / "fake-state.json").read_text())
        assert state["allocation_count"] == 1
        assert len(state["sessions"]) == 1
        assert len([e for e in events if e["argv"][4] == "upload"]) >= 10
        execs = [e for e in events if e["argv"][4] == "exec"]
        assert all(0 < float(e["argv"][-1]) <= policy.RPC_SECONDS for e in execs)
    finally:
        cli.stop_lease()
        supervisor.stop_verified(cli, cli.poll_recovery_endpoint)
    assert not cli.sessions()


@pytest.mark.parametrize(
    "failure", ["lost", "renamed", "record", "busy", "transient-probe", "wall"]
)
def test_outage_preserves_identity_idle_and_session_caps(
    monkeypatch, tmp_path, _isolate_account, failure
):
    clock = _isolate_account
    cli = owned_cli(monkeypatch, tmp_path, 50 if failure == "wall" else 1)
    original_idle = cli_guard.kernel_idle
    original_sessions = cli.sessions
    probes = []

    def sessions(*args, **kwargs):
        if failure == "lost":
            return {}
        if failure == "renamed":
            return {cli.poll_recovery_endpoint: "foreign"}
        return original_sessions(*args, **kwargs)

    monkeypatch.setattr(cli, "sessions", sessions)
    if failure == "record":
        cli.assignment_record.write_text(json.dumps({"endpoint": "foreign"}))
    if failure == "wall":
        cli.deadline = clock.now + 200

    def idle(*args, **kwargs):
        probes.append(clock.now)
        if failure == "busy":
            return False
        if failure == "transient-probe" and clock.now < 600:
            raise TimeoutError("probe timed out")
        return original_idle(*args, **kwargs)

    monkeypatch.setattr(cli_guard, "kernel_idle", idle)
    try:
        if failure == "transient-probe":
            assert cli.poll_rpc("# TRANSPORT_OUTAGE_READ\nresult=True") is True
            assert clock.now >= 600
        else:
            with pytest.raises(
                supervisor.Refused, match="disappeared|identity changed|deadline reached"
            ):
                cli.poll_rpc("# TRANSPORT_OUTAGE_READ\nresult=True")
        events = [
            json.loads(line) for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()
        ]
        execs = [e for e in events if e["argv"][4] == "exec"]
        if failure not in {"transient-probe", "wall"}:
            assert len(execs) == 1
        if failure == "wall":
            assert clock.now <= cli.deadline + 1
        assert cli.deadline <= 3600
    finally:
        cli.stop_lease()
        monkeypatch.setattr(cli, "sessions", original_sessions)
        supervisor.stop_verified(cli, cli.poll_recovery_endpoint)


def test_manifest_recovery_window_is_finite_bounded_and_optional(tmp_path, monkeypatch):
    from tests.test_colab_cli_supervisor import _manifest

    manifest = _manifest(tmp_path)
    assert "exec_poll_recovery_seconds" not in supervisor.validate(manifest)
    for invalid in (True, 0, -1, float("inf"), float("nan"), 1201):
        with pytest.raises(supervisor.Refused, match="exec_poll_recovery_seconds"):
            supervisor.validate({**manifest, "exec_poll_recovery_seconds": invalid})
    assert (
        supervisor.validate({**manifest, "exec_poll_recovery_seconds": 900})[
            "exec_poll_recovery_seconds"
        ]
        == 900
    )


@pytest.mark.parametrize("recovers", [True, False])
def test_supervisor_outage_receipt_and_verified_release(
    monkeypatch, tmp_path, _isolate_account, recovers
):
    """Use the real Broker, fake subprocess provider and real terminal receipt."""
    from tests.test_colab_cli_supervisor import _manifest

    binary = harness._fake_env(monkeypatch, tmp_path, {"schedule": []})
    monkeypatch.setenv("FAKE_COLAB_MODE", "success")
    manifest = _manifest(tmp_path)
    manifest["timeout_minutes"] = 60

    def observer(event, context):
        if event == "ready":
            monkeypatch.setenv(
                "FAKE_COLAB_FAULT_PLAN",
                json.dumps(
                    {
                        "seed": 29,
                        "max_latency_seconds": 120,
                        "schedule": [
                            {
                                "call": "exec",
                                "contains": "read_state",
                                "fault": "timeout",
                                "phase": "pre",
                            }
                        ]
                        * (4 if recovers else 50),
                    }
                ),
            )

    broker = supervisor.Broker(tmp_path / "lanes", binary, observer=observer)
    if recovers:
        broker.run(manifest)
    else:
        with pytest.raises(supervisor.Refused, match="telemetry recovery"):
            broker.run(manifest)
    receipt = json.loads((tmp_path / "job-output/receipt.json").read_text())
    state = json.loads((tmp_path / "fake-state.json").read_text())
    assert receipt["status"] == ("PASS" if recovers else "INCOMPLETE")
    assert receipt["failure_class"] == (None if recovers else "RPC_EXEC_POLL_TIMEOUT")
    assert receipt["stop_verified"] is True and receipt["cleanup_error"] is None
    assert receipt["elapsed_seconds"] > 300
    assert receipt["exec_poll_recovery_seconds"] == 1200
    assert state["allocation_count"] == 1
    assert list(state["job_launches"].values()) == [1]
    assert state["sessions"] == {}
    assert (
        len([c for c in receipt["cli_calls"] if c["command"] == "exec" and c["status"] == "failed"])
        >= 4
    )


@pytest.mark.parametrize("form", ["hex", "canonical", "foreign", "malformed", "missing"])
def test_recovery_uuid_matches_real_guard_record_format(
    monkeypatch, tmp_path, _isolate_account, form
):
    cli = owned_cli(monkeypatch, tmp_path, 0)
    record = json.loads(cli.assignment_record.read_text())
    # The fake uses the real guard's persisted canonical representation.
    assert record["notebook_hash"] == str(uuid.UUID(cli.notebook_hash))
    if form == "hex":
        record["notebook_hash"] = cli.notebook_hash
    elif form == "foreign":
        record["notebook_hash"] = str(uuid.uuid4())
    elif form == "malformed":
        record["notebook_hash"] = "not-a-uuid"
    elif form == "missing":
        record.pop("notebook_hash")
    cli.assignment_record.write_text(json.dumps(record))
    try:
        if form in {"hex", "canonical"}:
            cli.recovery_identity(30)
        else:
            with pytest.raises(supervisor.Refused, match="identity changed"):
                cli.recovery_identity(30)
    finally:
        cli.stop_lease()
        supervisor.stop_verified(cli, cli.poll_recovery_endpoint)
