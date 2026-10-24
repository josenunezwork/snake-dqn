"""Small synthetic OS/clock checks for ALLOW_PERSISTENCE_30S, no numerical work."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from research.bl_labels import factory_persistence as fp
from research.bl_labels import mac_driver as md
from research.bl_labels import slowdown_resume as sr
from research.bl_labels.exports import publish_json

ROOT = Path(__file__).resolve().parents[2]
BASE = "387b1dde8241c9c24ea352eb92de802b66fc798f"
SCIENCE = "61900af28932e47e22e332840fc4406559adb062"


def row(pid=123, birth="100", nice=14, parent=1, command="codex exec secret-token"):
    return {
        "pid": pid,
        "birth": birth,
        "birth_precision": {"unit": "native tick", "seconds": 1e-9},
        "ppid": parent,
        "nice": nice,
        "argv": command,
        "executable": "/opt/bin/codex",
        "relevant": True,
        "relevance_reason": "agent",
    }


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def sample(tracker, times, rows=None):
    for now in times:
        tracker.observe([row()] if rows is None else rows, now, set())


def test_one_six_seven_and_latched_priority_trip(tmp_path):
    tracker = fp.PersistenceTracker("attempt", tmp_path / "events.jsonl")
    sample(tracker, [0])
    assert tracker.pending[(123, "100")]["count"] == 1
    sample(tracker, [5, 10, 15, 20, 25])
    assert tracker.latched is None
    with pytest.raises(fp.ProcessMonitorStop, match="persistent concurrent compute pid 123"):
        sample(tracker, [30])
    with pytest.raises(fp.ProcessMonitorStop):
        tracker.observe([], 35, set())
    summary = tracker.summary()
    assert summary["sightings"][0]["maximum_streak"] == 7
    assert summary["sightings"][0]["confirmed_trips"] == 1
    payload = tracker.path.read_bytes()
    assert b"secret-token" not in payload
    assert b"preempt_persistent" in payload
    assert hashlib.sha256(payload).hexdigest() == summary["telemetry_sha256"]


@pytest.mark.parametrize("clear", [[], [row(nice=15)], [row(nice=19)]])
def test_disappearance_or_priority_clears_without_accumulation(clear):
    tracker = fp.PersistenceTracker("clear")
    sample(tracker, [0, 5, 10, 15, 20, 25])
    tracker.observe(clear, 30, set())
    assert not tracker.pending
    assert tracker.latched is None
    sample(tracker, [35])
    assert tracker.pending[(123, "100")]["count"] == 1
    assert tracker.summary()["sightings"][0]["cleared_transients"] == 1


def test_early_seventh_waits_for_elapsed_span():
    tracker = fp.PersistenceTracker("early")
    sample(tracker, [0, 4.9, 9.8, 14.7, 19.6, 24.5, 29.4])
    assert tracker.pending[(123, "100")]["count"] == 7
    with pytest.raises(fp.ProcessMonitorStop):
        sample(tracker, [34.4])


@pytest.mark.parametrize("mode", ["reuse", "alternating_pid", "alternating_birth"])
def test_different_identities_never_pool(mode):
    tracker = fp.PersistenceTracker(mode)
    for index in range(10):
        pid = 123 + (index % 2 if mode == "alternating_pid" else 0)
        birth = str(index // 6 if mode == "reuse" else index % 2)
        tracker.observe([row(pid=pid, birth=birth)], index * 5, set())
    assert tracker.latched is None
    assert max(r["count"] for r in tracker.pending.values()) <= 4


def test_late_candidate_does_not_get_earlier_observations():
    tracker = fp.PersistenceTracker("late")
    for index in range(7):
        tracker.observe([] if index < 5 else [row()], index * 5, set())
    assert tracker.pending[(123, "100")]["count"] == 2
    assert tracker.pending[(123, "100")]["first_seen"] == 25


def native(monkeypatch):
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda pid: {
            "birth": str(pid * 100),
            "executable": "/bin/bash" if pid == 123 else "/usr/bin/python3",
            "birth_precision": {"unit": "Mach tick", "seconds": 1e-9},
        },
    )


def test_reparent_relevance_and_low_priority_parent_high_priority_research_child(monkeypatch):
    native(monkeypatch)
    tracker = fp.PersistenceTracker("tree")
    for index in range(7):
        command = "123 1 15 codex exec task\n124 123 14 python -m research.x\n"
        if index > 1:
            command = "123 1 15 /bin/sh\n124 1 14 python research/x.py\n"
        rows = fp.process_rows(lambda _: command, tracker.known)
        if index < 6:
            tracker.observe(rows, index * 5, set())
        else:
            with pytest.raises(fp.ProcessMonitorStop, match="pid 124"):
                tracker.observe(rows, 30, set())
    assert tracker.summary()["sightings"][1]["maximum_streak"] == 7


def test_pid_reuse_generic_argv_drops_old_relevance(monkeypatch):
    native(monkeypatch)
    tracker = fp.PersistenceTracker("reuse-generic")
    rows = fp.process_rows(lambda _: "123 1 14 codex exec task\n", {})
    tracker.observe(rows, 0, set())
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda pid: {
            "birth": "new",
            "executable": "/usr/bin/opaque",
            "birth_precision": "native tick",
        },
    )
    rows = fp.process_rows(lambda _: "123 1 14 /usr/bin/opaque\n", tracker.known)
    tracker.observe(rows, 5, set())
    assert not tracker.pending


def test_verified_tmux_server_never_seeds_unrelated_panes(monkeypatch):
    native(monkeypatch)
    original = fp._native_identity
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda pid: {
            **original(pid),
            "executable": "/opt/homebrew/bin/tmux" if pid == 200 else "/bin/bash",
        },
    )
    table = (
        "200 1 0 tmux new-session codex-handoff.sh retained\n"
        "201 200 0 /bin/bash unrelated\n"
        "123 200 0 /bin/bash codex-handoff.sh real\n"
    )

    def read(command):
        if command[0] == "/usr/sbin/lsof":
            return "p200\nn/private/tmp/tmux/custom\n"
        if "display-message" in command:
            assert command[1:3] == ["-S", "/private/tmp/tmux/custom"]
            return "200\n"
        return table

    rows = fp.process_rows(read, {})
    assert {r["pid"] for r in rows if r["relevant"]} == {123}
    tracker = fp.PersistenceTracker("pane")
    for index in range(6):
        tracker.observe(rows, index * 5, set())
    with pytest.raises(fp.ProcessMonitorStop, match="pid 123"):
        tracker.observe(rows, 30, set())


@pytest.mark.parametrize("nice", [0, 15, 19])
def test_supervisor_has_no_name_exemption(monkeypatch, nice):
    native(monkeypatch)
    original = fp._native_identity
    monkeypatch.setattr(
        fp, "_native_identity", lambda pid: {**original(pid), "executable": "/usr/bin/python3"}
    )
    tracker = fp.PersistenceTracker("supervisor")
    table = f"123 1 {nice} python -m research.colab_cli.supervisor run job.json\n"
    rows = fp.process_rows(lambda _: table, {})
    for index in range(6):
        tracker.observe(rows, index * 5, set())
    if nice < 15:
        with pytest.raises(fp.ProcessMonitorStop):
            tracker.observe(rows, 30, set())
    else:
        tracker.observe(rows, 30, set())
        assert not tracker.pending


def test_foreign_tmux_name_is_never_executed_or_exempted(tmp_path, monkeypatch):
    foreign = tmp_path / "tmux"
    foreign.write_text("#!/bin/sh\nexit 99\n")
    foreign.chmod(0o700)
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda pid: {
            "birth": "foreign-tmux",
            "executable": str(foreign),
            "birth_precision": "tick",
        },
    )
    calls = []

    def read(command):
        calls.append(command)
        assert command[0] == "/bin/ps"
        return f"123 1 0 {foreign} codex-handoff.sh fake\n"

    rows = fp.process_rows(read, {})
    assert rows[0]["relevant"]
    assert len(calls) == 1


@pytest.mark.parametrize(
    "command",
    [
        "/Applications/ChatGPT.app/Contents/MacOS/codex exec task",
        str(Path.home() / ".codex/computer-use/codex-handoff.sh"),
        str(Path.home() / "Library/Application Support/Codex/codex") + " exec task",
    ],
)
def test_desktop_helpers_stay_excluded(command):
    assert fp._seed(command) is None


def test_owned_tree_exclusion_is_not_foreign_adoption(monkeypatch):
    tracker = fp.PersistenceTracker("owned")
    own = row(pid=123, parent=os.getpid())
    child = row(pid=124, parent=123)
    for index in range(8):
        tracker.observe([own, child, row(pid=125, nice=15)], index * 5, set())
    assert not tracker.pending
    assert set(tracker.known) == {(125, "100")}


def test_foreign_identity_cannot_be_adopted_after_reparenting():
    tracker = fp.PersistenceTracker("never-adopt")
    sample(tracker, [0])
    for now in [5, 10, 15, 20, 25]:
        tracker.observe([row(parent=os.getpid())], now, {123})
    with pytest.raises(fp.ProcessMonitorStop):
        tracker.observe([row(parent=os.getpid())], 30, {123})


@pytest.mark.parametrize(
    "field,value", [("birth", None), ("nice", None), ("nice", 21), ("executable", "")]
)
def test_unknown_required_metadata_is_immediate_fault(field, value):
    tracker = fp.PersistenceTracker("bad")
    with pytest.raises(fp.ProcessMonitorStop, match="monitor fault"):
        tracker.observe([{**row(), field: value}], 0, set())
    assert tracker.summary()["observation_errors"]


@pytest.mark.parametrize("gap", [15.001, 100])
def test_stale_observation_stops_without_new_grace(gap):
    tracker = fp.PersistenceTracker("stale")
    sample(tracker, [0])
    with pytest.raises(fp.ProcessMonitorStop, match="monitor fault"):
        tracker.observe([], gap, set())


@pytest.mark.parametrize("failure", ["timeout", "table", "nice", "access", "live-lookup"])
def test_broken_sensor_stops_immediately(monkeypatch, failure):
    native(monkeypatch)
    monkeypatch.setattr(fp, "_confirmed_absent", lambda _: False)
    if failure in {"access", "live-lookup"}:

        def broken(_):
            raise PermissionError("access denied")

        monkeypatch.setattr(fp, "_native_identity", broken)

    def read(_):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("ps", 10)
        return {"table": "malformed", "nice": "123 1 ? codex exec task"}.get(
            failure, "123 1 14 codex exec task"
        )

    tracker = fp.PersistenceTracker("sensor")
    with pytest.raises(fp.ProcessMonitorStop, match="monitor fault"):
        tracker.sample(read, set())
    assert len(tracker.errors) == 1


def test_confirmed_lookup_disappearance_is_absence(monkeypatch):
    tracker = fp.PersistenceTracker("race")
    sample(tracker, [0])

    def vanished(_):
        raise ProcessLookupError("gone")

    monkeypatch.setattr(fp, "_native_identity", vanished)
    monkeypatch.setattr(fp, "_confirmed_absent", lambda _: True)
    clock = Clock()
    clock.now = 5
    monkeypatch.setattr(fp, "time", clock)
    tracker.sample(lambda _: "123 1 14 codex exec task\n", set())
    assert not tracker.pending and not tracker.errors
    assert any(r["decision"] == "clear_absent" for r in tracker.records)


def test_vanished_root_does_not_hide_live_research_child(monkeypatch):
    calls = []

    def identity(pid):
        calls.append(pid)
        if pid == 123:
            raise ProcessLookupError("gone")
        return {
            "birth": "child",
            "executable": "/usr/bin/python3",
            "birth_precision": "native tick",
        }

    monkeypatch.setattr(fp, "_native_identity", identity)
    monkeypatch.setattr(fp, "_confirmed_absent", lambda pid: pid == 123)
    table = "123 1 15 codex exec task\n124 123 14 python -m research.x\n"
    rows = fp.process_rows(lambda _: table, {})
    assert calls == [123, 124]
    assert {r["pid"] for r in rows if r["relevant"]} == {124}


def test_real_native_birth_identity_keeps_os_precision():
    first = fp._native_identity(os.getpid())
    second = fp._native_identity(os.getpid())
    assert first == second
    assert int(first["birth"]) > 0
    assert first["birth_precision"]["seconds"] > 0
    assert Path(first["executable"]).is_file()


def test_aggregate_observation_deadline_bounds_every_command(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(fp, "time", clock)
    tracker = fp.PersistenceTracker("bounded-sensor")
    tracker.observe([], 0, set())
    clock.now = 5
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda _: {
            "birth": "server",
            "executable": str(fp.TRUSTED_TMUX),
            "birth_precision": "tick",
        },
    )
    calls = []

    def read(command):
        remaining = fp.observation_timeout()
        calls.append((command, remaining))
        if command[0] == "/bin/ps":
            clock.now += 6
            return "123 1 0 tmux codex-handoff.sh stale\n"
        assert remaining == 4
        clock.now += remaining
        raise subprocess.TimeoutExpired(command, remaining)

    with pytest.raises(fp.ProcessMonitorStop, match="monitor fault"):
        tracker.sample(read, set())
    assert clock.now == 15
    assert len(calls) == 2
    assert fp.observation_timeout() == 10  # Deadline context never leaks.


def test_actual_command_reader_consumes_remaining_observation_budget(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(fp, "time", clock)
    seen = []

    def run(command, **kwargs):
        seen.append(kwargs["timeout"])
        clock.now += 6
        from types import SimpleNamespace

        return SimpleNamespace(stdout="1 0 0 /sbin/launchd\n")

    monkeypatch.setattr(md.subprocess, "run", run)
    token = fp._OBSERVATION_DEADLINE.set(10)
    try:
        md._command_output(["/bin/ps"])
        md._command_output(["/bin/ps"])
        with pytest.raises(TimeoutError):
            md._command_output(["/bin/ps"])
    finally:
        fp._OBSERVATION_DEADLINE.reset(token)
    assert seen == [10, 4]


@pytest.mark.parametrize(
    "mode,expected", [("transient", "PASS"), ("late", "HOLD"), ("persistent", "TRIP")]
)
@pytest.mark.parametrize("recovery", [False, True])
def test_startup_and_recovery_use_same_bounded_window(
    tmp_path, monkeypatch, mode, expected, recovery
):
    native(monkeypatch)
    clock = Clock()
    monkeypatch.setattr(fp, "time", clock)
    health = {"ac_power": True, "lid_open": True, "thermal": "CPU_Speed_Limit = 100"}
    monkeypatch.setattr(md, "_guard_snapshot", lambda _: health)
    monkeypatch.setattr(md, "_timing_armed", lambda _: False)

    def table(_):
        present = (
            mode == "persistent"
            or (mode == "transient" and clock.now < 10)
            or (mode == "late" and clock.now >= 25)
        )
        return "123 1 14 codex exec task\n" if present else "1 0 0 /sbin/launchd\n"

    monkeypatch.setattr(md, "_command_output", table)
    monkeypatch.setattr(sr, "_operational_pin", lambda: "a" * 40)
    monkeypatch.setattr(sr, "escalation", lambda _: {"w6_epoch": 100})
    monkeypatch.setattr(sr.subprocess, "check_output", lambda *args, **kwargs: "123 0.0 python\n")
    from research.compute import factory_interim_launcher

    monkeypatch.setattr(factory_interim_launcher, "TEST_SLOT_PATH", tmp_path / "test.lock")
    tracker = fp.PersistenceTracker("startup", tmp_path / "startup.jsonl")

    def check():
        if recovery:
            path = sr.record_host_check(tmp_path, checked_epoch=101)
            receipt = sr._read_triplet(path)
            assert sr._host_check_after(tmp_path, 100)
            return receipt["process_window"]
        return md._factory_admission_window(tracker, tmp_path, tmp_path / "unarmed")

    if expected == "PASS":
        summary = check()
        assert summary["observation_count"] == 7
        assert summary["window_end_monotonic"] - summary["window_start_monotonic"] == 30
        assert not summary["final_pending"]
        assert summary["sightings"][0]["cleared_transients"] == 1
    else:
        with pytest.raises(
            (md.MacGuardError, ValueError),
            match="HOLD_PENDING" if expected == "HOLD" else "persistent concurrent",
        ):
            check()
        assert not list(tmp_path.glob("slowdown-host-check-101.json"))


@pytest.mark.parametrize("failure", ["timing", "health"])
def test_immediate_startup_guards_bypass_grace(tmp_path, monkeypatch, failure):
    tracker = fp.PersistenceTracker("immediate")
    monkeypatch.setattr(md, "_timing_armed", lambda _: failure == "timing")

    def health(_):
        raise md.MacGuardError("health fault")

    monkeypatch.setattr(md, "_guard_snapshot", health)
    with pytest.raises(md.MacGuardError):
        md._factory_admission_window(tracker, tmp_path, tmp_path / "arm")
    assert tracker.sequence == 0


def valid_receipt(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(fp, "time", clock)
    tracker = fp.PersistenceTracker("valid-host", tmp_path / "telemetry.jsonl")
    fp.admission_window(tracker, lambda: tracker.observe([], clock.now, set()), lambda: None)
    monkeypatch.setattr(sr, "_operational_pin", lambda: "a" * 40)
    return {
        "schema": sr.HOST_CHECK_SCHEMA,
        "policy": sr.PERSISTENCE_POLICY,
        "policy_id": fp.POLICY_ID,
        "ruling_sha256": fp.RULING_SHA256,
        "interim_ruling_sha256": fp.INTERIM_RULING_SHA256,
        "operational_pin": "a" * 40,
        "for_trip_epoch": 100,
        "checked_epoch": 101,
        "no_disallowed_competing_compute": True,
        "serial_test_lane_stopped": True,
        "host_health": {"ac_power": True, "lid_open": True},
        "pmset_therm": "CPU_Speed_Limit = 100",
        "top_cpu_consumers": [],
        "process_window": tracker.summary(),
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "bl-labels-f0-slowdown-host-check-nice-interim/v2"),
        ("policy_id", "foreign"),
        ("policy", "ALLOW_NICE15_INTERIM"),
        ("ruling_sha256", "0" * 64),
        ("interim_ruling_sha256", "0" * 64),
        ("operational_pin", "0" * 40),
        ("for_trip_epoch", 99),
        ("no_sol_codex", True),
        ("process_window", {}),
    ],
)
def test_foreign_or_v2_receipt_cannot_prove_v3(tmp_path, monkeypatch, field, value):
    receipt = valid_receipt(tmp_path, monkeypatch)
    receipt[field] = value
    path = tmp_path / "slowdown-host-check-101.json"
    publish_json(path, receipt)
    before = path.read_bytes()
    assert not sr._host_check_after(tmp_path, 100)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_pending", [row()]),
        ("observation_count", 6),
        ("telemetry_sha256", "0" * 64),
        ("window_end_monotonic", 25),
    ],
)
def test_incomplete_or_pending_window_is_never_clean_v3(tmp_path, monkeypatch, field, value):
    receipt = valid_receipt(tmp_path, monkeypatch)
    receipt["process_window"][field] = value
    publish_json(tmp_path / "slowdown-host-check-101.json", receipt)
    assert not sr._host_check_after(tmp_path, 100)


def test_clean_v3_receipt_is_create_only_and_honest(tmp_path, monkeypatch):
    receipt = valid_receipt(tmp_path, monkeypatch)
    path = tmp_path / "slowdown-host-check-101.json"
    publish_json(path, receipt)
    assert sr._host_check_after(tmp_path, 100)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        publish_json(path, {**receipt, "operational_pin": "b" * 40})
    assert path.read_bytes() == before


def test_science_and_accounting_closure_unchanged():
    def old(path):
        return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)

    for name in [
        "slowdown_guard.py",
        "slowdown_state.py",
        "d256_resume.py",
        "frozen_factory.py",
        "d256_w12.py",
        "w12_runtime.py",
        "exports.py",
        "RUN_D256_RESUME.sh",
    ]:
        path = "research/bl_labels/" + name
        assert (ROOT / path).read_bytes() == old(path), path
    science = ROOT.parent / "snake-dqn-d256-resume-science.noindex"
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=science, text=True).strip()
        == SCIENCE
    )
    assert (
        subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=science, capture_output=True
        ).returncode
        != 0
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=science, text=True
    ).strip()
    assert (
        subprocess.run(
            ["git", "diff", "--exit-code", SCIENCE, "--"], cwd=science, capture_output=True
        ).returncode
        == 0
    )
    import ast

    def assignments(source):
        tree = ast.parse(source)
        return {
            n.value: ast.dump(v)
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for n, v in zip(node.keys, node.values)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and any(word in n.value for word in ("worker_seconds", "elapsed_seconds", "wall_cap"))
        }

    assert assignments((ROOT / "research/bl_labels/mac_driver.py").read_text()) == assignments(
        old("research/bl_labels/mac_driver.py")
    )
    ruling = (
        ROOT.parent / "snake-dqn-artifacts/big-lever/exp/JUDGMENT_FACTORY_PERSISTENCE_20261023.md"
    )
    assert hashlib.sha256(ruling.read_bytes()).hexdigest() == fp.RULING_SHA256
