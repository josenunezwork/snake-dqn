"""Regression coverage for coordinator-side Ape-X child supervision."""

from __future__ import annotations

import subprocess
import sys
from select import select
from pathlib import Path

import pytest

from src.training.apex_runtime import (
    ApexRunBudgets,
    ApexRuntimeFailure,
    ApexRuntimeSnapshot,
    ApexRuntimeSupervisor,
    stop_processes,
)


class FakeChild:
    """Minimal process-shaped object for deterministic supervision tests."""

    def __init__(self, alive: bool = True, exitcode: int | None = None) -> None:
        self.alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self.alive


class PopenChild:
    """Adapt ``subprocess.Popen`` to the multiprocessing-shaped cleanup contract."""

    def __init__(self, child: subprocess.Popen[str]) -> None:
        self.child = child

    @property
    def exitcode(self) -> int | None:
        return self.child.poll()

    def is_alive(self) -> bool:
        return self.child.poll() is None

    def join(self, timeout: float) -> None:
        try:
            self.child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self) -> None:
        self.child.terminate()

    def kill(self) -> None:
        self.child.kill()


def make_supervisor(*, clock=lambda: 100.0) -> ApexRuntimeSupervisor:
    """Build a supervisor with two healthy fake actors."""
    return ApexRuntimeSupervisor(
        actors=[FakeChild(), FakeChild()],
        buffer_process=FakeChild(),
        budgets=ApexRunBudgets(max_learner_updates=4, max_environment_transitions=10),
        heartbeat_timeout_seconds=5.0,
        clock=clock,
    )


def test_supervisor_rejects_actor_exit_before_warmup():
    """A dead actor is a failure, even before replay is sampleable."""
    supervisor = make_supervisor()
    supervisor.actors[1].alive = False
    supervisor.actors[1].exitcode = 23

    with pytest.raises(ApexRuntimeFailure, match="actor 1 exited.*23"):
        supervisor.check_children({})


@pytest.mark.parametrize("actor_stats", [{}, {0: {"heartbeat_monotonic": 90.0}}])
def test_supervisor_rejects_buffer_exit_before_and_after_ready(actor_stats):
    """The buffer's liveness does not depend on warmup state or actor telemetry."""
    supervisor = make_supervisor()
    supervisor.buffer_process.alive = False
    supervisor.buffer_process.exitcode = 31

    with pytest.raises(ApexRuntimeFailure, match="buffer process exited.*31"):
        supervisor.check_children(actor_stats)


def test_supervisor_rejects_stale_heartbeat_from_live_actor():
    """A wedged process cannot pass only because its PID still exists."""
    supervisor = make_supervisor()

    with pytest.raises(ApexRuntimeFailure, match="heartbeat stale"):
        supervisor.check_children({0: {"heartbeat_monotonic": 94.0}})


def test_supervisor_rejects_actor_that_never_reports_initial_heartbeat():
    """Startup gets the same bounded liveness deadline as steady-state work."""
    supervisor = make_supervisor(clock=lambda: 106.0)
    supervisor.started_at = 100.0

    with pytest.raises(ApexRuntimeFailure, match="actor 0 heartbeat stale"):
        supervisor.check_children({})


def test_budget_precedence_and_snapshot_fields_are_explicit():
    """The declared first ceiling explains why a controlled run stopped."""
    supervisor = make_supervisor()
    snapshot = ApexRuntimeSnapshot(
        learner_updates=4,
        environment_transitions=10,
        replay_rows_emitted=9,
        learner_samples=8,
        policy_version=3,
        elapsed_seconds=1.0,
        actor_heartbeat_ages={0: 0.1, 1: None},
    )

    assert supervisor.stop_cause(snapshot) == "learner_update_budget"


def test_failure_fixture_exits_before_hard_subprocess_deadline():
    """A child crash is observable without letting the regression hang pytest."""
    fixture = Path(__file__).with_name("helpers") / "apex_failure_children.py"
    completed = subprocess.run(
        [sys.executable, str(fixture), "exit"],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )

    assert completed.returncode == 17


def test_ignored_shutdown_fixture_is_force_killed_with_hard_deadline():
    """Tests model the final terminate path without leaving a child behind."""
    fixture = Path(__file__).with_name("helpers") / "apex_failure_children.py"
    raw_child = subprocess.Popen(
        [sys.executable, str(fixture), "ignore-term"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        readable, _, _ = select([raw_child.stdout], [], [], 2.0)
        assert readable
        assert raw_child.stdout.readline().strip() == "ready"
        child = PopenChild(raw_child)
        stop_processes([child], timeout_seconds=0.2)
        assert not child.is_alive()
    finally:
        if raw_child.poll() is None:
            raw_child.kill()
        assert raw_child.wait(timeout=2.0) < 0
