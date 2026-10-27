"""Synthetic host admission tests; never acquire a real Mac lease."""

import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from research.bl_paced_identity.record import read_export
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


@pytest.mark.parametrize("error", ["AccessDenied", "NoSuchProcess", "ZombieProcess"])
def test_foreign_load_retains_child_enumeration_churn_without_aborting(monkeypatch, error):
    import psutil

    exception = getattr(psutil, error)(pid=10)
    monkeypatch.setattr(psutil, "Process", lambda _: Mock(children=Mock(side_effect=exception)))
    screen = {"processes": "PID PPID %CPU COMM\n10 1 90 python\n12 10 90 python\n"}

    assert probe_host.foreign_load(screen, 10) == []
    assert screen["probe_exempt_pids"] == [10]
    assert error in screen["probe_exempt_child_enumeration"]


@pytest.mark.parametrize("error", ["AccessDenied", "NoSuchProcess", "ZombieProcess"])
def test_owned_inventory_retains_process_churn_without_aborting(monkeypatch, error):
    import psutil

    exception = getattr(psutil, error)(pid=10)
    monkeypatch.setattr(psutil, "Process", lambda _: Mock(children=Mock(side_effect=exception)))

    assert probe_host._owned_inventory(10) == [
        {"pid": 10, "status": f"unavailable_during_sample:{error}"}
    ]


@pytest.mark.parametrize("error", ["AccessDenied", "NoSuchProcess", "ZombieProcess"])
def test_owned_inventory_retains_child_attribute_churn_without_aborting(monkeypatch, error):
    import psutil

    owner = Mock(pid=10)
    owner.memory_info.return_value = Mock(rss=123)
    owner.num_threads.return_value = 1
    owner.nice.return_value = 15
    owner.cmdline.return_value = ["owner"]
    child = Mock(pid=11)
    child.memory_info.side_effect = getattr(psutil, error)(pid=11)
    owner.children.return_value = [child]
    monkeypatch.setattr(psutil, "Process", lambda _: owner)

    rows = probe_host._owned_inventory(10)

    assert rows[0]["pid"] == 10
    assert rows[1] == {"pid": 11, "status": f"unavailable_during_sample:{error}"}


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


def test_monitor_exports_invalid_evidence_when_sampling_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        probe_host.host, "snapshot", Mock(side_effect=RuntimeError("sample failed"))
    )
    ready, stop, failed = threading.Event(), threading.Event(), threading.Event()
    heartbeat = SimpleNamespace(value=0.0)

    probe_host._monitor(10, str(tmp_path), 1.0, ready, stop, failed, heartbeat)

    evidence = read_export(tmp_path / "host.json")
    assert ready.is_set()
    assert failed.is_set()
    assert evidence["status"] == "INVALID"
    assert evidence["reason"] == "RuntimeError: sample failed"


def test_monitor_exit_preserves_final_failure_for_parent_report(tmp_path):
    monitor = probe_host.HostMonitor(tmp_path)
    monitor.process = Mock(is_alive=Mock(return_value=False), exitcode=1)
    monitor.failed.set()

    monitor.__exit__(None, None, None)

    assert monitor.receipt["host_path"] == str(tmp_path / "host.json")
    assert monitor.receipt["monitor_exitcode"] == 1
    assert monitor.receipt["monitor_status"] == "INVALID"
    evidence = read_export(tmp_path / "host.json")
    assert evidence["status"] == "INVALID"
    assert evidence["reason"] == "host monitor exited with code 1"
