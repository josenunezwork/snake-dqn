"""Synthetic-only coverage for the F4 post-mortem host runtime fence."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from research import mac_lease
from research.bl_residual import f4_postmortem_runtime as runtime


class _CleanGuard:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def check(self):
        return {"ok": self.ok, "reasons": [] if self.ok else ["synthetic_guard_failure"]}


class _Signals:
    SIGALRM = 14
    SIGINT = 2
    SIGTERM = 15
    ITIMER_REAL = 0

    def __init__(self) -> None:
        self.handlers = {
            self.SIGALRM: "old-alarm",
            self.SIGINT: "old-int",
            self.SIGTERM: "old-term",
        }
        self.timer = (0.0, 0.0)
        self.timer_history: list[tuple[float, float]] = []

    def getsignal(self, signum):
        return self.handlers[signum]

    def signal(self, signum, handler):
        self.handlers[signum] = handler

    def setitimer(self, _kind, seconds, interval=0.0):
        old, self.timer = self.timer, (seconds, interval)
        self.timer_history.append(self.timer)
        return old

    def getitimer(self, _kind):
        return self.timer


def test_deadline_checks_synthetic_host_and_restores_signal_state() -> None:
    now = [10.0]
    signals = _Signals()
    deadline = runtime.ClaimDeadline(
        seconds=60.0,
        guard=_CleanGuard(),
        lid_reader=lambda: True,
        clock=lambda: now[0],
        signal_module=signals,
    )

    with deadline as active:
        assert active.guard_callback() is not None
        active.guard_callback()({"synthetic": "ignored"})
        now[0] = 41.0
        with pytest.raises(runtime.PostmortemRuntimeDeadline):
            active.checkpoint()
    assert signals.handlers == {14: "old-alarm", 2: "old-int", 15: "old-term"}
    assert signals.timer == (0.0, 0.0)


def test_deadline_arms_hard_cap_before_and_after_blocking_host_probes() -> None:
    now = [0.0]
    signals = _Signals()

    def probe_lid() -> bool:
        now[0] += 10.0
        return True

    class AdvancingGuard(_CleanGuard):
        def check(self):
            now[0] += 10.0
            return super().check()

    deadline = runtime.ClaimDeadline(
        seconds=60.0,
        guard=AdvancingGuard(),
        lid_reader=probe_lid,
        clock=lambda: now[0],
        signal_module=signals,
    )
    with deadline as active:
        # Entry arms the compute timer before its two host probes, whose
        # synthetic 20 seconds leave only ten seconds of compute time.
        assert signals.timer == (30.0, 0.0)
        now[0] = 30.0
        with pytest.raises(runtime.PostmortemRuntimeDeadline, match="finalization"):
            active.checkpoint()
        assert signals.timer == (30.0, 0.0)


def test_three_ten_second_host_probes_cannot_consume_the_finalization_reserve() -> None:
    now = [0.0]
    signals = _Signals()

    def advance() -> None:
        now[0] += 10.0

    class AdvancingGuard(_CleanGuard):
        def check(self):
            advance()
            return super().check()

    with pytest.raises(runtime.PostmortemRuntimeDeadline, match="finalization"):
        with runtime.ClaimDeadline(
            seconds=60.0,
            guard=AdvancingGuard(),
            lid_reader=lambda: (advance() or True),
            runtime_check=advance,
            clock=lambda: now[0],
            signal_module=signals,
        ):
            pass
    # Initial soft timer is armed before the probes.  The post-probe check
    # converts it to a remaining absolute hard timer before propagating.
    assert signals.timer_history == [(30.0, 0.0), (30.0, 0.0), (0.0, 0.0)]


def test_periodic_alarm_rechecks_host_and_rearms_before_the_probe() -> None:
    now = [0.0]
    signals = _Signals()
    progress: list[dict] = []

    class CountingGuard(_CleanGuard):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def check(self):
            self.calls += 1
            return super().check()

    guard = CountingGuard()
    with runtime.ClaimDeadline(
        seconds=60.0,
        guard=guard,
        lid_reader=lambda: True,
        clock=lambda: now[0],
        signal_module=signals,
        progress=progress.append,
    ) as active:
        assert guard.calls == 1
        now[0] = 1.0
        active._alarm(signals.SIGALRM, None)
        assert guard.calls == 2
        assert signals.timer_history[-1] == (29.0, 0.0)
        assert progress == [
            {
                "event": "f4_postmortem_runtime_guard",
                "elapsed_seconds": 1.0,
                "seconds_left": 59.0,
            }
        ]
        active._checking_host = True
        now[0] = 2.0
        active._alarm(signals.SIGALRM, None)
        assert guard.calls == 2
        assert signals.timer_history[-1] == (28.0, 0.0)
        assert progress[-1]["event"] == "f4_postmortem_runtime_guard"
        active._checking_host = False


def test_hard_alarm_forces_exit_without_real_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    signals = _Signals()
    exits: list[int] = []

    class SyntheticHardExit(BaseException):
        pass

    def fake_exit(code: int) -> None:
        exits.append(code)
        raise SyntheticHardExit

    monkeypatch.setattr(runtime.os, "_exit", fake_exit)
    deadline = runtime.ClaimDeadline(
        seconds=60.0,
        guard=_CleanGuard(),
        lid_reader=lambda: True,
        clock=lambda: now[0],
        signal_module=signals,
    )
    with deadline as active:
        now[0] = 30.0
        with pytest.raises(runtime.PostmortemRuntimeDeadline):
            active._alarm(signals.SIGALRM, None)
        assert exits == []
        assert signals.timer == (30.0, 0.0)
        now[0] = 60.0
        with pytest.raises(SyntheticHardExit):
            active._alarm(signals.SIGALRM, None)
        assert exits == [124]


def test_deadline_fails_closed_for_lid_or_thermal_changes() -> None:
    signals = _Signals()
    with pytest.raises(runtime.PostmortemRuntimeError, match="lid"):
        with runtime.ClaimDeadline(
            guard=_CleanGuard(), lid_reader=lambda: False, signal_module=signals
        ):
            pass


def test_deadline_refuses_to_nest_inside_an_existing_timer() -> None:
    signals = _Signals()
    signals.timer = (3.0, 0.0)
    with pytest.raises(runtime.PostmortemRuntimeError, match="preexisting SIGALRM"):
        with runtime.ClaimDeadline(
            guard=_CleanGuard(), lid_reader=lambda: True, signal_module=signals
        ):
            pass
    with pytest.raises(runtime.PostmortemRuntimeError, match="synthetic_guard_failure"):
        with runtime.ClaimDeadline(
            guard=_CleanGuard(False), lid_reader=lambda: True, signal_module=_Signals()
        ):
            pass


def test_wait_for_lease_waits_without_preempting_then_uses_first_available() -> None:
    sleeps: list[float] = []
    calls = [0]

    @contextmanager
    def unavailable():
        raise mac_lease.LeaseHeld("synthetic owner")
        yield  # pragma: no cover

    @contextmanager
    def available():
        yield SimpleNamespace(receipt={"id": "synthetic"})

    def factory():
        calls[0] += 1
        return unavailable() if calls[0] == 1 else available()

    held = runtime._wait_for_lease(factory, sleep=sleeps.append, poll_seconds=30.0)
    assert sleeps == [30.0]
    assert held.lease.receipt == {"id": "synthetic"}
    assert held.__exit__(None, None, None) is False


def test_nice_refuses_inherited_value_that_cannot_be_lowered(monkeypatch) -> None:
    monkeypatch.setattr(runtime.os, "nice", lambda _increment: 16)
    with pytest.raises(runtime.PostmortemRuntimeError, match="cannot lower"):
        runtime._set_exact_nice()


def test_thread_fence_refuses_inherited_wrong_environment_before_torch_import(monkeypatch) -> None:
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "mps")
    for name in runtime.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    with pytest.raises(runtime.PostmortemRuntimeError, match="SNAKE_DQN_DEVICE"):
        runtime._configure_one_thread_torch()


def test_exclusive_runtime_is_synthetic_single_process_cpu_wake_and_lease(monkeypatch) -> None:
    events: list[object] = []
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    monkeypatch.setattr(runtime, "_set_exact_nice", lambda: events.append("nice") or 15)
    monkeypatch.setattr(runtime, "_configure_one_thread_torch", lambda: events.append("threads"))
    monkeypatch.setattr(runtime, "_on_ac_power", lambda _runner: True)
    monkeypatch.setattr(runtime, "_lid_open", lambda _runner: True)
    monkeypatch.setattr(runtime, "ThermalGuard", lambda **_kwargs: _CleanGuard())

    @contextmanager
    def synthetic_held(*_args, **_kwargs):
        events.append("lease-enter")
        yield SimpleNamespace(receipt={"id": "lease"})
        events.append("lease-exit")

    monkeypatch.setattr(runtime.mac_lease, "held", synthetic_held)

    class Wake:
        def __init__(self):
            self.alive = True

        def poll(self):
            return None if self.alive else 0

        def terminate(self):
            events.append("wake-term")
            self.alive = False

        def wait(self, timeout):
            events.append(("wake-wait", timeout))
            return 0

        def kill(self):
            raise AssertionError("healthy wake helper must not be killed")

    def popen(argv):
        events.append(tuple(argv))
        return Wake()

    with runtime.exclusive_runtime(popen=popen) as evidence:
        assert evidence.nice == 15 and evidence.threads == 1
        assert evidence.lease == {"id": "lease"}
    assert events[:3] == ["nice", "threads", "lease-enter"]
    assert events[3][:3] == ("/usr/bin/caffeinate", "-is", "-w")
    assert "wake-term" in events and "lease-exit" in events


def test_runtime_rejects_any_nonfixed_ceiling_before_host_or_payload_access(monkeypatch) -> None:
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="fixed at 60 minutes"):
        with runtime.exclusive_runtime(seconds=59):
            pass
