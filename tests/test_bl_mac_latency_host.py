"""Targeted guard failures and quantile population checks."""

import pytest

from research.bl_mac_latency import host
from research.bl_mac_latency.ruler_run import distribution


@pytest.mark.parametrize(
    "power,lid,thermal",
    [
        ("Battery Power", '"AppleClamshellState" = No', ""),
        ("AC Power", '"AppleClamshellState" = Yes', ""),
        ("AC Power", '"AppleClamshellState" = No', "unparsed"),
    ],
)
def test_host_refuses_unknown_or_unsafe(monkeypatch, power, lid, thermal):
    monkeypatch.setattr(host.platform, "system", lambda: "Darwin")
    responses = iter([power, lid, thermal])
    monkeypatch.setattr(host, "_read", lambda argv: next(responses))
    with pytest.raises(RuntimeError, match="host guard"):
        host.check_guard()


def test_host_accepts_nominal_ac_open(monkeypatch):
    monkeypatch.setattr(host.platform, "system", lambda: "Darwin")
    responses = iter(
        [
            "AC Power",
            '"AppleClamshellState" = No',
            "Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded\n"
            "Note: No CPU power status has been recorded\n",
        ]
    )
    monkeypatch.setattr(host, "_read", lambda argv: next(responses))
    assert host.check_guard()["reasons"] == []


def test_nearest_rank_keeps_slow_tail():
    value = distribution(list(range(1, 101)))
    assert value == {"n": 100, "p50": 50, "p95": 95, "p99": 99, "max": 100}
    assert distribution([])["p99"] is None


def test_process_screen_keeps_foreign_load_and_excludes_measurement(monkeypatch):
    monkeypatch.setattr(host.os, "getpid", lambda: 42)
    monkeypatch.setattr(
        host,
        "_read",
        lambda argv: "PID PPID %CPU COMM\n42 1 100 own\n43 1 90 foreign\n44 1 4 background\n",
    )
    screen = host.process_screen()
    assert [row["pid"] for row in screen["other_heavy_processes"]] == [43]
    assert "42 1 100 own" in screen["processes"]


def test_quiet_admission_stops_before_operator_math(monkeypatch, tmp_path):
    import time

    from research.bl_mac_latency import ruler_run

    monkeypatch.setattr(ruler_run, "check_guard", lambda: {"reasons": []})
    monkeypatch.setattr(
        ruler_run, "process_screen", lambda: {"other_heavy_processes": [{"pid": 43}]}
    )
    operator = ruler_run.GuardedOperator(tmp_path, time.monotonic() + 60, require_quiet=True)
    with pytest.raises(host.GuardError) as caught:
        operator.batch({"fixture_id": "unexecuted"})
    assert caught.value.observation["reasons"] == ["foreign_cpu_load_not_quiet"]
    assert operator.completed == 0
