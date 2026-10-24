"""State transitions and ownership failures; all process tables are synthetic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research.compute import factory_companion as companion
from research.compute.test_factory_companion import Clock, ready, row, running, supervisor


def test_missing_registry_monitor_never_manufactures_health(tmp_path) -> None:
    subject = supervisor(tmp_path, Clock(), {})
    assert subject.assessment(policy_sha=companion.POLICY_SHA256).healthy is False


def test_wait4_adds_only_late_cpu_and_preserves_resource_limit(tmp_path) -> None:
    clock = Clock()
    clock.value = 2000
    subject = supervisor(tmp_path, clock, {})
    state = subject._empty()
    running(state)
    state["registrations"]["t"].update(cpu=84.0, status="resource_limit")
    subject._write(state)
    code = subject._finalize_wait4("t", 10, 0, SimpleNamespace(ru_utime=85.0, ru_stime=1.0))
    after = subject._read()
    assert code == 124
    assert after["registrations"]["t"]["status"] == "resource_limit"
    assert after["registrations"]["t"]["descendants_gone"] is True
    assert sum(s["total_seconds"] for s in after["samples"]) == 2
    assert after["duty"] == [{"at": 2000, "cpu": 86}]


def test_reaped_root_with_live_descendant_retains_reservation_and_slot(
    tmp_path, monkeypatch
) -> None:
    clock = Clock()
    rows = {11: row(11, pgid=10)}
    subject = supervisor(tmp_path, clock, rows)
    state = subject._empty()
    running(state)
    subject._write(state)
    released = []
    monkeypatch.setattr(subject, "_release_test_slot", lambda: released.append(True))
    code = subject._finalize_wait4("t", 10, 0, SimpleNamespace(ru_utime=1, ru_stime=0))
    assert code != 0 and released == []
    record = subject._read()["registrations"]["t"]
    assert record["status"] == "failed_cleanup" and record["descendants_gone"] is False


def test_pid_reuse_cleanup_never_signals_foreign_group(tmp_path, monkeypatch) -> None:
    clock = Clock()
    subject = supervisor(tmp_path, clock, {10: row(10, start="reused"), 99: row(99, pgid=99)})
    state = subject._empty()
    running(state)
    subject._write(state)
    signals = []
    monkeypatch.setattr(companion.os, "killpg", lambda *args: signals.append(args))
    subject.terminate("t")
    assert signals == []
    assert subject._read()["registrations"]["t"]["status"] == "failed_cleanup"


def test_protected_period_keeps_light_tools_open_but_drains_test(tmp_path, monkeypatch) -> None:
    clock = Clock()
    subject = supervisor(tmp_path, clock, {})
    state = subject._empty()
    ready(state)
    running(state)
    subject._write(state)
    terminated = []
    monkeypatch.setattr(subject, "terminate", terminated.append)
    subject.update_factory_state(identity="a", protected=True, streams={})
    assert terminated == ["t"]
    assert not subject._read().get("admission_closed")


def test_exclusive_arm_closes_every_local_lane_and_does_not_signal_daemons(
    tmp_path, monkeypatch
) -> None:
    clock = Clock()
    subject = supervisor(tmp_path, clock, {99: row(99, pgid=99)})
    state = subject._empty()
    ready(state)
    running(state)
    for token, klass, pid in (("client", "client", 11), ("tool", "light", 12)):
        state["registrations"][token] = {
            **state["registrations"]["t"],
            "token": token,
            "class": klass,
            "pid": pid,
            "pgid": pid,
        }
    subject._write(state)
    terminated = []
    monkeypatch.setattr(subject, "terminate", terminated.append)
    subject.prepare_exclusive(attempt_identity="a", policy_sha=companion.POLICY_SHA256)
    assert set(terminated) == {"t", "client", "tool"}
    assert subject._read()["admission_closed"] is True


def test_failed_native_measurement_is_not_a_zero_cpu_sample(tmp_path) -> None:
    clock = Clock()
    subject = supervisor(tmp_path, clock, {})
    state = subject._empty()
    running(state)
    process = row(10)
    process["native_error"] = True
    assert subject._group_usage(state["registrations"]["t"], {10: process})[2] is False


def test_registry_restart_preserves_duty_and_initial_missing_history_hold(tmp_path) -> None:
    clock = Clock()
    subject = supervisor(tmp_path, clock, {})
    state = subject._empty()
    assert state["history_hold_until"] == 1800
    state["duty"] = [{"at": 0, "cpu": 200}]
    subject._write(state)
    restarted = supervisor(tmp_path, clock, {})
    assert restarted._read()["duty"] == [{"at": 0, "cpu": 200}]
    assert restarted._read()["history_hold_until"] == 1800


def test_os_advisory_slot_remains_global_across_supervisor_objects(tmp_path) -> None:
    clock = Clock()
    first = supervisor(tmp_path, clock, {})
    second = supervisor(tmp_path, clock, {})
    state = first._empty()
    ready(state)
    first._admit(state, "test", companion.POLICY_SHA256, "a")
    try:
        with pytest.raises(companion.AdmissionError, match="global"):
            second._admit(state, "test", companion.POLICY_SHA256, "a")
    finally:
        first._release_test_slot()


@pytest.mark.parametrize(
    ("duration", "warning", "hard_trip"),
    [
        (115.0, False, False),
        (115.000001, True, False),
        (130.0, True, False),
        (130.000001, True, True),
    ],
)
def test_warning_and_hard_slowdown_have_independent_exact_boundaries(
    duration: float, warning: bool, hard_trip: bool
) -> None:
    from research.bl_labels.slowdown_guard import SlowdownGuard
    from research.bl_labels.test_slowdown_guard import _frozen, _lives

    guard = SlowdownGuard("watch6", 6, 0, persisted=_frozen())
    original = guard.snapshot(0)["baselines"]
    events, now = _lives(guard, [duration] * 24)
    assert bool(guard.companion_warning()) is warning
    assert bool(events) is hard_trip
    assert guard.snapshot(now)["baselines"] == original
