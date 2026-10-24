"""Mechanical ps acceptance regressions; synthetic metadata, no scientific work."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from research.bl_labels import factory_persistence as fp
from research.bl_labels import mac_driver as md


def native(pid):
    return {
        "birth": str(pid * 100),
        "birth_precision": {"unit": "native tick", "seconds": 1e-9},
        "executable": "/usr/bin/python3",
    }


@pytest.mark.parametrize("nice", range(-20, 21))
def test_full_darwin_nice_range_at_both_validation_layers(monkeypatch, nice):
    monkeypatch.setattr(fp, "_native_identity", native)
    rows = fp.process_rows(lambda _: f"18774 18737 {nice} python3 -m research.fixture\n", {})
    tracker = fp.PersistenceTracker("3cc0134838a94e12810f297348c71286")
    tracker.observe(rows, 0, set())
    assert not tracker.errors and tracker.latched is None
    assert bool(tracker.pending) == (nice < 15)
    assert rows[0]["pid"] == 18774 and rows[0]["ppid"] == 18737
    assert rows[0]["nice"] == nice
    if nice >= 15:
        assert tracker.records[0]["decision"] == "clear_priority"


def test_incident_nice20_clears_prior_streak_through_real_guard(monkeypatch):
    """Exact reported PID/PPID/nice and attempt ID, with synthetic birth/argv."""
    monkeypatch.setattr(fp, "_native_identity", native)
    clock = [0.0]
    monkeypatch.setattr(fp.time, "monotonic", lambda: clock[0])
    nice = [14]
    monkeypatch.setattr(
        md, "_command_output", lambda _: f"18774 18737 {nice[0]} codex exec fixture\n"
    )
    tracker = fp.PersistenceTracker("3cc0134838a94e12810f297348c71286")
    for value in range(0, 30, 5):
        clock[0] = value
        md._exclusive_factory_guard(set(), tracker)
    nice[0], clock[0] = 20, 30
    md._exclusive_factory_guard(set(), tracker)
    assert not tracker.pending and not tracker.errors and tracker.latched is None
    clear = next(r for r in tracker.records if r["decision"] == "clear_priority")
    assert (clear["pid"], clear["ppid"], clear["nice"]) == (18774, 18737, 20)


@pytest.mark.parametrize(
    "args", ["", "   ", "<defunct>", "[kernel_task]", "<exiting>", "/usr/sbin/netbiosd"]
)
def test_system_and_other_user_rows_need_no_native_access(monkeypatch, args):
    monkeypatch.setattr(fp, "_native_identity", lambda _: pytest.fail("unrelated OS process"))
    rows = fp.process_rows(lambda _: f"550 1 20 {args}\n", {})
    tracker = fp.PersistenceTracker("system")
    tracker.observe(rows, 0, set())
    assert len(rows) == 1 and not rows[0]["relevant"] and not tracker.errors


def test_empty_args_relevant_descendant_keeps_identity_and_priority(monkeypatch):
    monkeypatch.setattr(fp, "_native_identity", native)
    rows = fp.process_rows(lambda _: "123 1 15 codex exec task\n124 123 20\n", {})
    assert {r["pid"] for r in rows if r["relevant"]} == {123, 124}
    assert rows[1]["argv"] == "" and rows[1]["relevance_reason"] == "relevant_descendant"
    known = {(124, rows[1]["birth"]): rows[1]}
    reparented = fp.process_rows(lambda _: "124 1 20\n", known)
    assert reparented[0]["relevant"] and reparented[0]["birth"] == rows[1]["birth"]


@pytest.mark.parametrize("suffix", [" café_日本語", " a\u2028b\u0085c\u2029d", " \udcff"])
def test_long_and_nonascii_args_are_one_untruncated_row(monkeypatch, suffix):
    monkeypatch.setattr(fp, "_native_identity", native)
    args = "python3 " + "x " * 100_000 + "-m research.fixture" + suffix
    calls = []

    def read(command):
        calls.append(command)
        return "123 1 20 " + args + "\n"

    rows = fp.process_rows(read, {})
    assert len(rows) == 1 and rows[0]["argv"] == args and rows[0]["relevant"]
    assert "-ww" in calls[0]
    tracker = fp.PersistenceTracker("unicode")
    tracker.observe(rows, 0, set())
    assert not tracker.errors and not tracker.pending


@pytest.mark.parametrize("state", ["Z", "Z+", "Zs"])
def test_native_lookup_of_known_zombie_is_valid_clear(monkeypatch, state):
    monkeypatch.setattr(fp, "_confirmed_absent", lambda _: False)

    def gone(_):
        raise ProcessLookupError("native zombie metadata unavailable")

    monkeypatch.setattr(fp, "_native_identity", gone)
    known = {(123, "birth"): {"relevance_reason": "agent"}}
    calls = []

    def read(command):
        calls.append(command)
        return "123 1 20 <defunct>\n" if "-A" in command else f"123 {state}\n"

    rows = fp.process_rows(read, known)
    assert not rows[0]["relevant"]
    assert calls[1] == ["/bin/ps", "-p", "123", "-o", "pid=,stat="]


def test_zombie_root_does_not_hide_its_live_high_priority_child(monkeypatch):
    monkeypatch.setattr(fp, "_confirmed_absent", lambda _: False)

    def identity(pid):
        if pid == 18774:
            raise ProcessLookupError("unreaped exit")
        return native(pid)

    monkeypatch.setattr(fp, "_native_identity", identity)

    def read(command):
        if "-A" in command:
            return "18774 18737 20 <defunct>\n18775 18774 14 opaque-child\n"
        return "18774 Z\n"

    known = {(18774, "old-birth"): {"relevance_reason": "agent"}}
    rows = fp.process_rows(read, known)
    tracker = fp.PersistenceTracker("3cc0134838a94e12810f297348c71286")
    tracker.observe(rows, 0, set())
    assert not tracker.errors
    assert {pid for pid, _ in tracker.pending} == {18775}


@pytest.mark.parametrize("failure", ["live", "denied", "malformed", "ps-failure"])
def test_defunct_argv_or_other_user_access_never_excuses_live_sensor_fault(monkeypatch, failure):
    monkeypatch.setattr(fp, "_confirmed_absent", lambda _: False)

    def denied(_):
        raise PermissionError("unreadable live metadata")

    monkeypatch.setattr(fp, "_native_identity", denied)

    def read(command):
        if "-A" in command:
            return "18774 18737 20 codex exec <defunct>\n"
        if failure == "denied":
            raise PermissionError("other user")
        if failure == "ps-failure":
            raise subprocess.CalledProcessError(2, command)
        return "bad" if failure == "malformed" else "18774 S\n"

    tracker = fp.PersistenceTracker("live-fault")
    with pytest.raises(fp.ProcessMonitorStop, match="monitor fault"):
        tracker.sample(read, set())
    assert tracker.records[-1]["sensor_reason"] == "unreadable live process birth or executable"
    assert tracker.records[-1]["nice"] == 20
    assert "argv" not in tracker.records[-1]


def test_vanish_between_native_lookup_and_state_ps_is_valid_absence(monkeypatch):
    alive_checks = iter([False, True])
    monkeypatch.setattr(fp, "_confirmed_absent", lambda _: next(alive_checks))

    def gone(_):
        raise ProcessLookupError("gone")

    monkeypatch.setattr(fp, "_native_identity", gone)

    def read(command):
        if "-A" in command:
            return "123 1 20 codex exec task\n"
        raise subprocess.CalledProcessError(1, command)

    rows = fp.process_rows(read, {})
    assert not rows[0]["relevant"]


def test_new_pid_on_next_full_table_is_valid_observation(monkeypatch):
    monkeypatch.setattr(fp, "_native_identity", native)
    tracker = fp.PersistenceTracker("churn")
    tables = iter(["123 1 14 codex exec first\n", "124 1 20 codex exec next\n"])
    for now in [0, 5]:
        tracker.observe(fp.process_rows(lambda _: next(tables), tracker.known), now, set())
    assert not tracker.pending and not tracker.errors
    assert any(r["decision"] == "clear_absent" for r in tracker.records)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "broken",
        "123 1 ? task",
        "123 1 21 task",
        "123 1 -21 task",
        "123 1 20 task\n123 1 20 task",
    ],
)
def test_genuinely_malformed_full_tables_still_fault(bad):
    tracker = fp.PersistenceTracker("bad-table")
    with pytest.raises(fp.ProcessMonitorStop, match="monitor fault"):
        tracker.sample(lambda _: bad, set())


@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_genuine_ps_failures_still_fault(failure):
    def read(command):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(fp.ProcessMonitorStop, match="monitor fault"):
        fp.PersistenceTracker("ps-failure").sample(read, set())


def test_real_reader_decodes_non_utf8_os_output_losslessly():
    raw = md._command_output(["/usr/bin/printf", "\\377caf\\303\\251"])
    assert raw.encode("utf-8", "surrogateescape") == b"\xffcaf\xc3\xa9"


def test_reader_keeps_timeout_and_checked_exit(monkeypatch):
    def run(command, **kwargs):
        assert kwargs["check"] and kwargs["timeout"] <= 10
        assert kwargs["errors"] == "surrogateescape"
        return SimpleNamespace(stdout="123 1 20\n")

    monkeypatch.setattr(md.subprocess, "run", run)
    assert md._command_output(["/bin/ps"]) == "123 1 20\n"
