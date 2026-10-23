"""Replay acceptance-1792762502 without inventing a cumulative read quota."""

from __future__ import annotations

import copy
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import cli_guard, policy, supervisor
from research.rpc_qualification.checker import _raw_faults

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/colab_control_faults_acceptance_1792762502.json").read_text()
)


def test_receipt_census_preserves_recovered_and_terminal_l4_timeouts() -> None:
    """Both real exec failures count even with a generic terminal helper error."""
    calls = FIXTURE["cli_calls"]
    assert len(calls) == 122
    assert Counter(row["command"] for row in calls) == {
        "version": 1,
        "new": 1,
        "sessions": 19,
        "usage": 19,
        "upload": 19,
        "exec": 62,
        "stop": 1,
    }
    assert sum(FIXTURE["cpu_cli_counts"].values()) == 136
    assert FIXTURE["cpu_cli_counts"]["exec"] == 73
    receipt = {**FIXTURE["receipt"], "cli_calls": calls}
    faults = _raw_faults(receipt)
    # The pre-runtime upload failure is preserved too; this test grants no
    # unregistered carve-out. The helper refusal counts as a terminal fault.
    assert [(row["source"], row.get("index")) for row in faults] == [
        ("cli_calls", 7),
        ("cli_calls", 58),
        ("cli_calls", 117),
        ("terminal", None),
    ]
    for index in (58, 117):
        assert calls[index]["failure_class"] == "CONTROL_RPC_TIMEOUT"
        assert calls[index]["elapsed_seconds"] > calls[index]["timeout_seconds"]
    assert calls[58]["request_id"] == calls[59]["request_id"]
    assert calls[59]["status"] == "completed"  # Recovery never deletes the fault.
    assert receipt["stop_verified"] is True  # Cleanup never deletes it either.


@pytest.mark.parametrize("count", [62, 122, 136])
def test_fast_read_helpers_at_observed_counts_have_independent_deadlines(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    """More marker/read calls do not consume another helper's elapsed budget."""
    now = [0.0]
    calls = []

    def read(*_args: Any, timeout: float) -> dict[str, bool]:
        calls.append(timeout)
        now[0] += 0.25
        return {"idle": True}

    monkeypatch.setattr(cli_guard.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli_guard, "_run_helper", read)
    for _ in range(count):
        assert cli_guard.kernel_idle("fake", "config", "owned", timeout=30) is True
    assert calls == [30] * count
    assert now[0] == count * 0.25


@pytest.mark.parametrize("cap,cap_seconds", [("deadline", 1755.0), ("priced_deadline", 1800.0)])
def test_l4_exec_timeout_then_guard_elapsed_budget_exhaustion_keeps_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: str, cap_seconds: float
) -> None:
    """Replay call 117 followed by a stalled GET, with no cap or retry change.

    The receipt omits helper timing; a full 30s helper timeout is a deterministic
    reproduction of the code path, not a measured historical helper duration.
    """
    row = FIXTURE["cli_calls"][117]
    now = [0.0]
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    cli.call_records = copy.deepcopy(FIXTURE["cli_calls"][:117])
    # Historical wall cap reserves 45s for cleanup; the priced cap is 1,800s.
    setattr(cli, cap, cap_seconds)
    exec_calls = []
    helper_calls = []

    def execute(*_args: str, timeout: float) -> str:
        exec_calls.append(timeout)
        assert timeout <= policy.RPC_SECONDS
        now[0] += row["elapsed_seconds"]
        raise subprocess.TimeoutExpired("exec", timeout)

    def read(*_args: Any, timeout: float) -> dict[str, Any]:
        helper_calls.append(timeout)
        assert cli.exec_uncertain
        now[0] += timeout
        raise cli_guard.GuardError("guard helper failed: TimeoutExpired")

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(cli, "_call_once", execute)
    monkeypatch.setattr(cli_guard, "_run_helper", read)
    with pytest.raises(
        cli_guard.GuardError, match="read helper operation budget exhausted"
    ) as raised:
        cli.poll_rpc("result=True")
    assert "TimeoutExpired" in str(raised.value.__cause__)
    assert exec_calls == [30]
    assert helper_calls == [30]
    assert now[0] == pytest.approx(row["elapsed_seconds"] + 30)
    assert getattr(cli, cap) == cap_seconds
    assert cli.exec_uncertain
    assert cli.call_records[-1]["failure_class"] == "CONTROL_RPC_TIMEOUT"
    with pytest.raises(supervisor.Refused, match="previous exec uncertain/busy"):
        cli.rpc("result=True")
    assert len(exec_calls) == 1  # No mutation replay without independent idle proof.
