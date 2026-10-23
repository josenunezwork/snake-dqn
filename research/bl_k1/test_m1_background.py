"""Exact identity exclusions and sampled CPU disclosures; no real timing work."""

from types import SimpleNamespace

import pytest

from research.bl_k1 import m1
from research.bl_k1 import m1_background as bg
from research.bl_mac_latency import host


def screen(command: str, cpu: float = 99.0, identity: dict | None = None) -> dict:
    return {
        "processes": f"PID PPID %CPU COMM\n10 1 1 coordinator\n20 1 {cpu} {command}\n",
        "other_heavy_processes": (
            [{"pid": 20, "ppid": 1, "cpu_percent": cpu, "command": command}] if cpu >= 50 else []
        ),
        "process_identity": {"20": identity or {}},
    }


@pytest.mark.parametrize("command", sorted(bg.SYSTEM_DAEMONS))
def test_exact_system_daemons_never_foreign(command: str) -> None:
    assert command.startswith(("/System/", "/usr/libexec/"))
    value = screen(command, 250.0)
    assert m1.foreign_load(value, 10) == []
    assert bg.allowed_background(value)[0]["cpu_percent"] == 250.0
    assert bg.allowed_background(screen(command, 4.0))[0]["cpu_percent"] == 4.0


@pytest.mark.parametrize(
    "command",
    [
        "fileproviderd",
        "/tmp/fileproviderd",
        "/System/fake/fileproviderd",
        "/usr/libexec/nsurlsessiond-unknown",
        "/usr/libexec/../libexec/nsurlsessiond",
        "/Applications/Spotlight.app/Contents/MacOS/Spotlight",
        "codex",
        "unknown",
        "python3",
    ],
)
def test_names_and_prefixes_do_not_authorize_foreign_load(command: str) -> None:
    value = screen(command)
    assert m1.foreign_load(value, 10) == value["other_heavy_processes"]


@pytest.mark.parametrize(
    "argv,cwd,allowed",
    [
        (
            ["python3", "-m", "research.colab_cli.supervisor", "run", "manifest.json"],
            "/Users/josenunez/Projects/ml/snake-dqn-a3",
            True,
        ),
        (["python3", "-m", "research.colab_cli"], "/Users/josenunez/Projects/ml/snake-dqn", True),
        (
            ["python3", "/Users/josenunez/Projects/ml/snake-dqn-a3/tools/ops/dispatch.py"],
            "/tmp",
            True,
        ),
        (["python3", "tools/ops/dispatch.py"], "/Users/josenunez/Projects/ml/snake-dqn-a3", True),
        (
            ["python3", "-c", "research.colab_cli.supervisor"],
            "/Users/josenunez/Projects/ml/snake-dqn",
            False,
        ),
        (
            ["python3", "other.py", "-m", "research.colab_cli.supervisor"],
            "/Users/josenunez/Projects/ml/snake-dqn",
            False,
        ),
        (
            ["python3", "-m", "research.colab_cli_fake"],
            "/Users/josenunez/Projects/ml/snake-dqn",
            False,
        ),
        (["python3", "-m", "research.colab_cli.supervisor"], "/tmp/snake-dqn-a3", False),
        (
            ["python3", "-m", "research.colab_cli.supervisor"],
            "/Users/josenunez/Projects/ml/snake-dqn-artifacts",
            False,
        ),
        (["python3", "-m", "research.colab_cli.supervisor"], "", False),
        (
            ["python3", "-m", "research.colab_cli.supervisor"],
            "/Users/josenunez/Projects/ml/snake-dqn/../evil",
            False,
        ),
        (["python3", "/tmp/snake-dqn/tools/ops/dispatch.py"], "/tmp", False),
    ],
)
def test_supervisor_requires_python_entrypoint_and_our_checkout(argv, cwd, allowed) -> None:
    value = screen("/usr/local/bin/python3", identity={"argv": argv, "cwd": cwd})
    assert (m1.foreign_load(value, 10) == []) is allowed


def test_monitor_keeps_daemon_cpu_and_stops_on_other_python() -> None:
    samples = iter(
        [
            {"monotonic_ns": 0, "process_screen": screen("/usr/libexec/nsurlsessiond")},
            {"monotonic_ns": 2_000_000_000, "process_screen": screen("python3")},
        ]
    )
    monitor = m1.HostMonitor(lambda: next(samples))
    monitor.stop.wait = lambda _: False
    monitor._loop()
    assert len(monitor.rows) == 2
    assert monitor.rows[0]["foreign_heavy_processes"] == []
    assert monitor.rows[0]["allowed_background_processes"][0]["cpu_percent"] == 99
    assert monitor.error == m1.RESTART_POLICY["trigger"]


def test_quiet_gate_accepts_authorized_load_above_frozen_threshold() -> None:
    result = m1.wait_quiet(
        lambda: {"monotonic_ns": 0, "process_screen": screen("/usr/libexec/nsurlsessiond", 200)},
        interval_seconds=0,
    )
    assert result["quiet"] is True
    assert len(result["samples"]) == 3
    assert all(row["allowed_background_processes"] for row in result["samples"])


def test_summary_weights_samples_absence_multiple_pids_and_discard_gaps() -> None:
    command = "/usr/libexec/nsurlsessiond"

    def row(t, cpus):
        return {
            "monotonic_ns": t * 1_000_000_000,
            "allowed_background_processes": [
                {"command": command, "cpu_percent": cpu} for cpu in cpus
            ],
        }

    result = bg.background_summary(
        [
            ([row(0, [40, 20]), row(2, [10]), row(6, [])], 8_000_000_000),
            ([row(20, [100])], 22_000_000_000),
        ]
    )
    assert result["observed_seconds"] == 10
    assert result["sample_count"] == 4
    assert result["total"] == {
        "max_cpu_percent": 100,
        "mean_cpu_percent": 36,
        "time_at_or_above_50_seconds": 4,
    }
    assert result["by_command"][command] == result["total"]
    assert bg.background_summary([])["total"]["mean_cpu_percent"] is None


def test_host_collects_os_identity_but_fails_closed_on_unreadable(monkeypatch) -> None:
    import psutil

    monkeypatch.setattr(host, "_read", lambda _: "PID PPID %CPU COMM\n20 1 99 /usr/bin/python3\n")
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda _: SimpleNamespace(
            cmdline=lambda: ["python3", "-m", "research.colab_cli.supervisor"],
            cwd=lambda: "/Users/josenunez/Projects/ml/snake-dqn-a3",
        ),
    )
    assert m1.foreign_load(host.process_screen(include_identity=True), 10) == []

    def denied(_):
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil, "Process", denied)
    assert m1.foreign_load(host.process_screen(include_identity=True), 10)
    assert "process_identity" not in host.process_screen()


def test_snapshot_policy_is_m1_only(monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(host, "check_guard", lambda: {})
    monkeypatch.setattr(host, "_read", lambda _: "")
    monkeypatch.setattr(host, "process_screen", lambda **kw: screen("/usr/libexec/nsurlsessiond"))
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="", returncode=0)
    )
    assert host.snapshot()["quiet"] is False
    assert host.snapshot(process_policy=lambda s: m1.foreign_load(s, 10))["quiet"] is True
