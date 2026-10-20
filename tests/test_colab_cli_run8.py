"""Offline replay of the L4 run-8 pre-download transport sequence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import acceptance, cli_guard, supervisor
from tests import test_colab_cli_fault_properties as harness

_isolate_account = harness._isolate_account
pinned_repo = harness.pinned_repo


@pytest.mark.parametrize("exec_timeouts", [1, 3])
def test_run8_l4_pre_download_exec_stall_recovers_and_finishes_final_only_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pinned_repo: Path,
    _isolate_account: harness._TransportClock,
    exec_timeouts: int,
) -> None:
    """Replay the receipt sequence without masking the real broker transport.

    The selectors apply only to the L4 session: a 42-second release upload,
    failed heartbeat, 19.75-second bootstrap upload, a 39.4-second heartbeat,
    then a completed-but-unacknowledged VERIFY_FILE exec.  The latter must
    prove idle before its retry, retain the single allocation, and reach the
    final artifact and watchdog gates.
    """
    binary = harness._fake_env(
        monkeypatch,
        tmp_path,
        {
            "seed": 8,
            "max_latency_seconds": 42,
            "schedule": [
                {
                    "call": "upload",
                    "session_contains": "-l4-",
                    "contains": "-release.json",
                    "fault": "latency",
                    "latency_seconds": 42,
                },
                {
                    "call": "upload",
                    "session_contains": "-l4-",
                    "contains": "lease-heartbeat.json",
                    "fault": "reset",
                    "phase": "pre",
                },
                {
                    "call": "upload",
                    "session_contains": "-l4-",
                    "contains": "/bootstrap.sh",
                    "fault": "latency",
                    "latency_seconds": 19.75,
                },
                {
                    "call": "upload",
                    "session_contains": "-l4-",
                    "contains": "lease-heartbeat.json",
                    "fault": "latency",
                    "latency_seconds": 39.4,
                },
                *[
                    {
                        "call": "exec",
                        "session_contains": "-l4-",
                        "contains": "SNAKE_TRANSFER_VERIFY_FILE",
                        "fault": "timeout",
                        "phase": "post",
                    }
                ]
                * exec_timeouts,
            ],
        },
    )
    plan = harness._pinned_plan(tmp_path, pinned_repo, profile="final-only")
    order = []
    original_idle = cli_guard.kernel_idle
    original_rpc = supervisor.CLI.rpc

    def idle(*args: Any, **kwargs: Any) -> bool:
        value = original_idle(*args, **kwargs)
        order.append("idle" if value else "busy")
        return value

    def rpc(cli: supervisor.CLI, code: str, *args: Any, **kwargs: Any) -> Any:
        if "-l4-" in cli.session and "SNAKE_TRANSFER_VERIFY_FILE" in code:
            order.append("exec")
        return original_rpc(cli, code, *args, **kwargs)

    monkeypatch.setattr(cli_guard, "kernel_idle", idle)
    monkeypatch.setattr(supervisor.CLI, "rpc", rpc)
    rc = acceptance.execute(
        plan,
        state_dir=tmp_path / "state",
        colab=binary,
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python=str(harness.HELPER),
    )

    harness._assert_safe(tmp_path, plan)
    run_dir = Path(plan["jobs"][0]["manifest"]).parent.parent
    result = json.loads((run_dir / "acceptance-result.json").read_text())
    receipt = json.loads((run_dir / "outputs/l4/receipt.json").read_text())
    state = json.loads((tmp_path / "fake-state.json").read_text())
    events: list[dict[str, Any]] = [
        json.loads(line) for line in (tmp_path / "fake-calls.jsonl").read_text().splitlines()
    ]
    faults = [event["fault"] for event in events if event.get("fault")]

    assert [fault["fault"] for fault in faults[:4]] == ["latency", "reset", "latency", "latency"]
    assert all(fault["fault"] == "timeout" for fault in faults[4:])
    assert 1 <= len(faults[4:]) <= exec_timeouts
    assert [fault["virtual_latency_seconds"] for fault in faults[:5]] == pytest.approx(
        [42, 0, 19.75, 39.4, 120]
    )
    assert state["allocation_count"] == 2
    assert receipt["stop_verified"] is True
    assert receipt["cleanup_error"] is None
    assert order[:3] == ["exec", "idle", "exec"]
    if exec_timeouts > 1:
        assert rc == 1
        assert receipt["status"] == "INCOMPLETE"
        assert "deadline reached during telemetry recovery" in receipt["error"]
        # Earlier uploads/renewals consume the same transfer clock, so the cap
        # may prevent even dispatching the third scheduled timeout. Every
        # dispatched retry must still have an independent idle prerequisite.
        assert len(faults[4:]) >= 2
        assert all(event == "exec" for event in order[::2])
        assert all(event == "idle" for event in order[1::2])
        assert receipt["transfers"][-1]["elapsed_seconds"] == pytest.approx(300)
        assert not state.get("watchdog_released")
        assert not (run_dir / "outputs/l4/exports/final-artifact.bin").exists()
        return

    assert rc == 0
    assert state["watchdog_released"] == [receipt["session"]]

    gates = {gate["id"]: gate for gate in result["final_plan"]["gates"]}
    download = gates["final-artifact-download"]
    assert download["status"] == "OBSERVED"
    assert download["evidence_record"]["exported_bytes"] == 64 * 1024 * 1024
    assert download["evidence_record"]["verified_on_mac"] is True
    assert gates["mac-loss-watchdog-unassign"]["status"] == "OBSERVED"
    assert all(
        call["timeout_seconds"] <= 120
        for call in receipt["cli_calls"]
        if call["command"] == "exec" and call["status"] == "failed"
    )
