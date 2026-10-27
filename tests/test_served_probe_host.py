"""Synthetic host admission tests; never acquire a real Mac lease."""

import os
from unittest.mock import Mock

import pytest

from research.bl_served_search import probe_host


def test_one_thread_cpu_nice_required(monkeypatch):
    for key in probe_host.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    monkeypatch.setattr(os, "getpriority", lambda *args: 15)
    assert probe_host.environment()["nice"] == 15
    monkeypatch.setenv("NUMBA_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="thread caps"):
        probe_host.environment()


def test_foreign_load_keeps_daemons_and_exempts_owned_children(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "Process", lambda _: Mock(children=lambda **kw: [Mock(pid=12)]))
    screen = {
        "processes": "PID PPID %CPU COMM\n10 1 90 python\n12 10 90 python\n"
        "99 1 21 daemon\n101 1 80 /System/Library/analyticsd\n20 10 26 /bin/ps\n"
    }
    rows = probe_host.foreign_load(screen, 10)
    assert [row["pid"] for row in rows] == [99]
    assert screen["probe_exempt_pids"] == [10, 12]
    assert screen["ordinary_system_background"][0]["pid"] == 101


def test_preflight_is_read_only_and_does_not_admit_heavy_host(monkeypatch):
    monkeypatch.setattr(probe_host, "environment", lambda: {"device": "cpu"})
    monkeypatch.setattr(probe_host.host, "snapshot", lambda **kw: {"quiet": False})
    assert probe_host.preflight()["admitted"] is False


def test_monitor_check_refuses_failed_or_stale_heartbeat(tmp_path):
    monitor = probe_host.HostMonitor(tmp_path)
    monitor.failed.set()
    with pytest.raises(RuntimeError, match="guard failed"):
        monitor.check()
    monitor.failed.clear()
    with pytest.raises(RuntimeError, match="unavailable"):
        monitor.check()
