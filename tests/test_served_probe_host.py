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


def test_classified_screen_retains_renderer_but_rejects_python_and_unknown(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "Process", lambda _: Mock(children=lambda **kw: []))
    screen = {
        "processes": (
            "PID PPID %CPU COMM\n10 1 90 python\n11 10 90 python\n12 11 80 /bin/ps\n"
            "20 1 200 /Applications/Codex.app/Contents/Frameworks/Codex Helper (Renderer)\n"
            "21 1 240 /System/Library/corespotlightd\n"
            "22 1 60 /usr/libexec/fileproviderd\n23 1 20 /usr/local/bin/python\n"
            "24 1 30 /bin/ps\n"
        )
    }
    heavy = probe_host.foreign_load(screen, 10, classified=True)
    assert [row["pid"] for row in heavy] == [23, 24]
    assert screen["probe_exempt_pids"] == [10, 11, 12]
    assert [row["pid"] for row in screen["ordinary_system_background"]] == [20, 21, 22]


def test_desktop_point_screen_enforces_workload_precedence_and_retains_admission(monkeypatch):
    import psutil

    class Classifier:
        def admit_process(self, row):
            return self.classify(row)

        def classify(self, row):
            if row["command"].endswith("Microsoft Outlook") and row.get("uid") == 501:
                return {"kind": "desktop_app", "reason": "enrolled", "role": "main_ui"}
            return {"kind": "unknown", "reason": "unenrolled"}

    monkeypatch.setattr(psutil, "Process", lambda _: Mock(children=lambda **kw: []))
    monkeypatch.setattr(
        probe_host,
        "_desktop_point_inventory",
        lambda: [
            {
                "pid": 20,
                "ppid": 1,
                "uid": 501,
                "birth": "outlook",
                "cpu_percent": 80,
                "command": "/Applications/Microsoft Outlook.app/Contents/MacOS/Microsoft Outlook",
            },
            {
                "pid": 21,
                "ppid": 1,
                "uid": 501,
                "birth": "python",
                "cpu_percent": 20,
                "command": "/Applications/Microsoft Outlook.app/Contents/MacOS/python",
            },
        ],
    )
    screen = {
        "processes": (
            "PID PPID %CPU COMM\n20 1 80 /Applications/Microsoft Outlook.app/Contents/MacOS/"
            "Microsoft Outlook\n21 1 20 /Applications/Microsoft Outlook.app/Contents/MacOS/python\n"
        )
    }
    heavy = probe_host.foreign_load(
        screen, 10, classified=True, desktop_classifier=Classifier(), admit_desktop=True
    )
    assert [row["pid"] for row in screen["ordinary_system_background"]] == [20]
    assert [row["pid"] for row in heavy] == [21]
    assert screen["classified_processes"][0]["admission"]["role"] == "main_ui"
    assert screen["load_policy"] == "classified-desktop-v1"


def test_desktop_point_does_not_join_raw_cpu_to_later_pid_identity(monkeypatch):
    import psutil

    class Classifier:
        def admit_process(self, row):
            return self.classify(row)

        def classify(self, row):
            return {"kind": "desktop_app", "reason": "enrolled"}

    monkeypatch.setattr(psutil, "Process", lambda _: Mock(children=lambda **kw: []))
    monkeypatch.setattr(
        probe_host,
        "_desktop_point_inventory",
        lambda: [
            {
                "pid": 20,
                "ppid": 1,
                "uid": 501,
                "birth": "new",
                "cpu_percent": 0,
                "command": "/Applications/Mail.app/Contents/MacOS/Mail",
            }
        ],
    )
    screen = {
        "processes": "PID PPID %CPU COMM\n20 1 80 /Applications/Mail.app/Contents/MacOS/Mail\n"
    }
    assert (
        probe_host.foreign_load(screen, 10, classified=True, desktop_classifier=Classifier()) == []
    )
    assert screen["classified_processes"][0]["cpu_percent"] == 0
    assert screen["processes_identity_stale_unjoined"] is True


def test_desktop_point_identity_change_aborts_even_when_final_kind_is_service(monkeypatch):
    import psutil

    class Classifier:
        def admit_process(self, row):
            return {
                "kind": "unknown",
                "reason": "executable_replaced",
                "identity_binding_changed": True,
            }

        def classify(self, row):
            return {
                "kind": "unknown",
                "reason": "executable_replaced",
                "identity_binding_changed": True,
            }

    monkeypatch.setattr(psutil, "Process", lambda _: Mock(children=lambda **kw: []))
    monkeypatch.setattr(
        probe_host,
        "_desktop_point_inventory",
        lambda: [
            {
                "pid": 20,
                "ppid": 1,
                "uid": 501,
                "birth": "new",
                "cpu_percent": 0,
                "command": "/usr/libexec/cloudd",
            }
        ],
    )
    screen = {"processes": "PID PPID %CPU COMM\n20 1 0 /usr/libexec/cloudd\n"}
    rows = probe_host.foreign_load(screen, 10, classified=True, desktop_classifier=Classifier())
    assert rows[0]["kind"] == "macos_service"
    assert rows[0]["classification"]["identity_binding_changed"] is True


def test_point_identity_parser_keeps_executables_with_and_without_spaces(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "  PID   UID STARTED COMMAND\n"
                "   20   501 Tue Oct 27 08:00:00 2026 /Applications/Microsoft Outlook.app/"
                "Contents/MacOS/Microsoft Outlook\n"
                "   21   501 Tue Oct 27 08:01:00 2026 /Applications/Claude.app/Contents/"
                "MacOS/Claude\n"
            )
        ),
    )
    identities = probe_host._point_identities()
    assert identities[20]["command"].endswith("Microsoft Outlook")
    assert identities[21]["command"].endswith("/Claude")


def test_desktop_point_inventory_binds_cpu_uid_and_birth_in_one_capture(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "PID PPID UID STARTED %CPU COMMAND\n"
                "20 1 501 Tue Oct 27 08:00:00 2026 80.0 /Applications/Microsoft Outlook.app/"
                "Contents/MacOS/Microsoft Outlook\n"
            )
        ),
    )
    row = probe_host._desktop_point_inventory()[0]
    assert row == {
        "pid": 20,
        "ppid": 1,
        "uid": 501,
        "birth": "Tue Oct 27 08:00:00 2026",
        "cpu_percent": 80.0,
        "command": "/Applications/Microsoft Outlook.app/Contents/MacOS/Microsoft Outlook",
    }


def test_classified_monitor_exports_final_interval_and_policy(monkeypatch, tmp_path):
    from research.bl_served_search import host_load

    monkeypatch.setattr(probe_host.host, "snapshot", lambda **kw: {"quiet": True})
    monkeypatch.setattr(probe_host, "_owned_inventory", lambda _: [])
    ready, stop, failed = threading.Event(), threading.Event(), threading.Event()
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds
        stop.set()

    def capture():
        at = int(clock[0] * 1e9)
        return {"time_ns": at, "capture_start_ns": at, "capture_end_ns": at, "processes": []}

    monkeypatch.setattr(probe_host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probe_host.time, "sleep", sleep)
    monkeypatch.setattr(host_load, "capture_cpu", capture)
    probe_host._monitor(10, str(tmp_path), 120, ready, stop, failed, SimpleNamespace(value=0), True)
    evidence = read_export(tmp_path / "host.json")
    assert evidence["status"] == "COMPLETE"
    assert evidence["load_policy"] == "classified-v1"
    assert evidence["cpu_samples"][0]["end_ns"] == 1_000_000_000
    assert not failed.is_set()


@pytest.mark.parametrize("stop_at", ["ready", "capture-0", "capture-1", "guard"])
def test_classified_monitor_captures_after_stop_and_always_exports_samples(
    monkeypatch, tmp_path, stop_at
):
    from research.bl_served_search import host_load

    monkeypatch.setattr(probe_host.host, "snapshot", lambda **kw: {"quiet": True})
    monkeypatch.setattr(probe_host, "_owned_inventory", lambda _: [])
    ready, stop, failed = threading.Event(), threading.Event(), threading.Event()
    clock, captures, stop_times = [0.0], [], []

    def finish():
        if not stop.is_set():
            stop_times.append(int(clock[0] * 1e9))
            stop.set()

    def capture():
        start = int(clock[0] * 1e9)
        clock[0] += 0.1
        midpoint = int(clock[0] * 1e9)
        clock[0] += 0.1
        if stop_at == f"capture-{len(captures)}":
            finish()  # Stop after this capture's midpoint, before returning it.
        result = {
            "time_ns": midpoint,
            "capture_start_ns": start,
            "capture_end_ns": int(clock[0] * 1e9),
            "processes": [],
        }
        captures.append(result)
        return result

    def sleep(seconds):
        clock[0] += seconds

    original_check = host_load.LoadGuard.check

    def check(guard, sample):
        original_check(guard, sample)
        if stop_at == "guard":
            finish()

    if stop_at == "ready":
        monkeypatch.setattr(ready, "set", finish)
    monkeypatch.setattr(probe_host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probe_host.time, "sleep", sleep)
    monkeypatch.setattr(host_load, "capture_cpu", capture)
    monkeypatch.setattr(host_load.LoadGuard, "check", check)
    probe_host._monitor(10, str(tmp_path), 120, ready, stop, failed, SimpleNamespace(value=0), True)
    evidence = read_export(tmp_path / "host.json")
    assert evidence["status"] == "COMPLETE"
    assert not failed.is_set()
    assert evidence["cpu_samples"]
    assert captures[-1]["capture_start_ns"] >= stop_times[0]
    assert evidence["cpu_samples"][-1]["end_ns"] > stop_times[0]
