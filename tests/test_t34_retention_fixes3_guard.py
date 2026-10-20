"""FIX-FIRST #3 guard rules, using synthetic clocks and telemetry only."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from research.t34_pathb import host_guard_v2

POLICY = json.loads(Path("research/t34_pathb/RETENTION_PROFILE_V2.json").read_text())["load_policy"]


@pytest.mark.parametrize("cap", [40 * 3600, 8 * 3600])
def test_five_second_sample_never_stops_an_early_row(cap):
    now = [0.0]
    sample = {"contended": True, "reasons": ["load_1m_excess"]}
    guard = host_guard_v2.HostGuard(
        POLICY, phase_cap_seconds=cap, clock=lambda: now[0], sample=lambda: dict(sample)
    )
    try:
        guard.start()
        for _ in range(10):
            guard.check()
            guard.between_rows()
        assert guard.summary()["contended_seconds"] == 5
        assert guard.summary()["contended_fraction"] == 5 / cap
        now[0] = 100
        assert guard.summary()["contended_seconds"] == 5
    finally:
        guard.close()


def test_budget_is_checked_at_phase_end_only_and_equality_passes():
    guard = host_guard_v2.HostGuard(
        POLICY, phase_cap_seconds=250, sample=lambda: {"contended": True, "reasons": ["load"]}
    )
    try:
        guard.start()
        assert guard.summary()["status"] == "PASS"  # 5s == 2% of 250s
        guard._capture()
        guard.between_rows()  # 10s > 2%; row boundaries still do not judge the phase.
        with pytest.raises(host_guard_v2.HostContentionError):
            guard.summary()
    finally:
        guard.close()


def test_admission_requires_entire_quiet_window_and_does_not_charge_it():
    now = [0.0]
    guard = host_guard_v2.HostGuard(
        POLICY,
        clock=lambda: now[0],
        sample=lambda: {"contended": False, "quiet": True, "reasons": []},
        wait=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    guard.admit()
    assert now[0] == 120
    assert len(guard.admission_records) == 25
    assert guard.records == []
    assert guard.summary()["contended_seconds"] == 0
    guard.admit()  # Already-admitted precheck seam does not repeat admission.
    assert now[0] == 120


@pytest.mark.parametrize("busy_at", [0, 115, 120])
def test_admission_refuses_even_transient_raw_contention(busy_at):
    now = [0.0]
    guard = host_guard_v2.HostGuard(
        POLICY,
        clock=lambda: now[0],
        sample=lambda: {"contended": False, "quiet": now[0] < busy_at, "reasons": []},
        wait=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    with pytest.raises(host_guard_v2.HostAdmissionError):
        guard.admit()
    assert guard._thread is None
    assert guard.records == []


def test_load_requires_over_sixty_continuous_seconds_above_measured_baseline(monkeypatch):
    now, load = [0.0], [7.8]
    monkeypatch.setattr(host_guard_v2.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(host_guard_v2, "_power_thermal", lambda: (True, True, True))
    monkeypatch.setattr(host_guard_v2, "_own_pids", lambda: set())
    monkeypatch.setattr(host_guard_v2, "_processes", lambda: [])
    monkeypatch.setattr(host_guard_v2.os, "getloadavg", lambda: (load[0], 0, 0))
    guard = host_guard_v2.HostGuard(POLICY, clock=lambda: now[0])
    baseline = guard._measure()
    assert baseline["idle_load_1m"] == 7.8
    assert baseline["load_1m_threshold"] == 10.8
    load[0] = 11.0
    now[0] = 5
    assert guard._measure()["contended"] is False
    now[0] = 65
    assert guard._measure()["contended"] is False
    now[0] = 70
    assert guard._measure()["reasons"] == ["load_1m_excess"]
    load[0], now[0] = 7.8, 75
    assert guard._measure()["contended"] is False
    load[0], now[0] = 11, 80
    assert guard._measure()["contended"] is False
    now[0] = 140
    assert guard._measure()["contended"] is False
    now[0] = 145
    assert guard._measure()["contended"] is True


def test_process_cpu_persistence_resets_and_blocks_quiet_admission(monkeypatch):
    now, cpu = [0.0], [51.0]
    monkeypatch.setattr(host_guard_v2.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(host_guard_v2, "_power_thermal", lambda: (True, True, True))
    monkeypatch.setattr(host_guard_v2, "_own_pids", lambda: set())
    monkeypatch.setattr(host_guard_v2.os, "getloadavg", lambda: (7.8, 0, 0))
    monkeypatch.setattr(
        host_guard_v2,
        "_processes",
        lambda: [{"pid": 123, "cpu_percent": cpu[0], "command": "research/runner"}],
    )
    guard = host_guard_v2.HostGuard(POLICY, clock=lambda: now[0])
    assert guard._measure()["quiet"] is False
    now[0] = 60
    assert guard._measure()["contended"] is False
    now[0] = 65
    assert "known_heavy_process" in guard._measure()["reasons"]
    cpu[0], now[0] = 50, 70
    assert guard._measure()["quiet"] is True
    cpu[0], now[0] = 51, 75
    assert guard._measure()["contended"] is False


@pytest.mark.parametrize("late_failure", [False, True])
def test_close_fences_a_sample_already_in_flight(late_failure):
    entered, release = threading.Event(), threading.Event()

    def slow_sample():
        entered.set()
        assert release.wait(1)
        if late_failure:
            raise RuntimeError("late sampling failure")
        return {"contended": True, "reasons": ["load"]}

    guard = host_guard_v2.HostGuard(POLICY, sample=slow_sample)
    capture = threading.Thread(target=guard._capture)
    capture.start()
    assert entered.wait(1)
    guard.close()  # No thread join is needed to fence publication atomically.
    before = guard.summary(validate=False)
    release.set()
    capture.join(timeout=1)
    assert not capture.is_alive()
    assert guard.summary(validate=False) == before
    assert guard.records == []
    assert guard._sample_error is None
