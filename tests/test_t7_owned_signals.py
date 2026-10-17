"""Owned-worker lifecycle and scheduler tests; no gameplay is executed."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from research.redesign_t7_phase_r_20261016 import phase_r_t7 as r


class FakeChild:
    def __init__(self, pid: int, events: list, timeout_once: bool = False) -> None:
        self.pid = pid
        self.returncode = None
        self.events = events
        self.calls = []
        self.timeout_once = timeout_once
        self.vanish_next_signal = False

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: signal.Signals) -> None:
        if self.vanish_next_signal:
            self.returncode = 0
            self.vanish_next_signal = False
            raise ProcessLookupError("owned child exited between poll and signal")
        self.calls.append(sig)
        self.events.append(("signal", self.pid, sig))
        if sig in (signal.SIGTERM, signal.SIGKILL) and not self.timeout_once:
            self.returncode = -int(sig)

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(("wait", self.pid, timeout))
        if self.timeout_once and timeout is not None:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("owned fake child", timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


@pytest.fixture(autouse=True)
def forbid_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("process-group signaling is forbidden")

    monkeypatch.setattr(r.os, "killpg", forbidden)


def test_real_owned_child_lifecycle() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    owner = r.OwnedWorkerSignals()
    try:
        owner.send(child, signal.SIGCONT)
        assert child.pid not in owner.paused_pids
        owner.send(child, signal.SIGSTOP)
        assert child.pid in owner.paused_pids
        owner.send(child, signal.SIGCONT)
        assert child.pid not in owner.paused_pids
        owner.send(child, signal.SIGTERM)
        owner.reap(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_exit_between_poll_and_signal_clears_pause_tracking() -> None:
    child, owner = FakeChild(10, []), r.OwnedWorkerSignals()
    owner.send(child, signal.SIGSTOP)
    child.vanish_next_signal = True
    owner.send(child, signal.SIGCONT)
    assert child.calls == [signal.SIGSTOP]
    assert child.pid not in owner.paused_pids
    owner.send(child, signal.SIGTERM)
    assert child.calls == [signal.SIGSTOP]


def test_unpaused_continue_and_exited_child_receive_no_signals() -> None:
    child, owner = FakeChild(10, []), r.OwnedWorkerSignals()
    owner.send(child, signal.SIGCONT)
    assert child.calls == []
    owner.paused_pids.add(child.pid)
    child.returncode = 0
    owner.send(child, signal.SIGTERM)
    assert child.calls == []
    assert child.pid not in owner.paused_pids


def test_timeout_escalates_to_direct_kill_then_wait() -> None:
    events = []
    child, owner = FakeChild(10, events, timeout_once=True), r.OwnedWorkerSignals()
    owner.send(child, signal.SIGTERM)
    owner.reap(child, timeout=0.01)
    assert events == [
        ("signal", 10, signal.SIGTERM),
        ("wait", 10, 0.01),
        ("signal", 10, signal.SIGKILL),
        ("wait", 10, None),
    ]
    assert child.poll() is not None


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    plan = {"commit": "synthetic-no-gameplay"}
    units = [{"unit_id": f"u{i}", "worlds": [i]} for i in range(7)]
    r.receipts.write_once(tmp_path / "plan.json", plan)
    monkeypatch.setattr(r, "load_plan", lambda _root: plan)
    monkeypatch.setattr(r, "_verify_plan_runtime", lambda _plan: None)
    monkeypatch.setattr(r, "units_for", lambda _plan: units)
    monkeypatch.setattr(r, "record_name", lambda unit, world: f"{unit['unit_id']}-{world}.json")
    monkeypatch.setattr(r.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(r, "decide", lambda *_args: {"synthetic": True})
    state = {"tick": 0, "peak": 0, "children": [], "events": [], "root": tmp_path}
    state["unowned"] = FakeChild(99999, state["events"])

    def popen(argv: list[str], **_kwargs: Any) -> FakeChild:
        child = FakeChild(100 + len(state["children"]), state["events"])
        state["children"].append(child)
        state["events"].append(("spawn", child.pid, state["tick"]))
        state["peak"] = max(state["peak"], sum(c.poll() is None for c in state["children"]))
        assert argv[-2] == "--unit-id"
        return child

    monkeypatch.setattr(r.subprocess, "Popen", popen)
    return state


def test_guard_blocks_launch_and_scheduler_preserves_cap(
    monkeypatch: pytest.MonkeyPatch, scheduler: dict[str, Any]
) -> None:
    state = scheduler
    monkeypatch.setattr(r, "guard_ok", lambda: ["guard"] if state["tick"] == 0 else [])

    def sleep(_seconds: float) -> None:
        state["tick"] += 1
        if state["tick"] >= 2:
            for child in state["children"]:
                child.returncode = 0

    monkeypatch.setattr(r.time, "sleep", sleep)
    assert r.run(state["root"], 6) == 0
    assert len(state["children"]) == 7
    assert state["peak"] == 6
    assert all(e[2] >= 1 for e in state["events"] if e[0] == "spawn")
    assert state["unowned"].calls == []


def test_interrupted_guard_signals_all_paused_children_before_wait(
    monkeypatch: pytest.MonkeyPatch, scheduler: dict[str, Any]
) -> None:
    state = scheduler
    monkeypatch.setattr(r, "guard_ok", lambda: ["guard"] if state["tick"] else [])

    def sleep(_seconds: float) -> None:
        if state["tick"]:
            raise KeyboardInterrupt("synthetic interruption")
        state["tick"] += 1

    monkeypatch.setattr(r.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        r.run(state["root"], 6)
    assert len(state["children"]) == 6
    for child in state["children"]:
        assert child.calls == [signal.SIGSTOP, signal.SIGCONT, signal.SIGTERM]
    first_wait = next(i for i, event in enumerate(state["events"]) if event[0] == "wait")
    terms = [
        i
        for i, event in enumerate(state["events"])
        if event[0] == "signal" and event[2] == signal.SIGTERM
    ]
    assert len(terms) == 6 and max(terms) < first_wait
    assert state["unowned"].calls == []


def test_failed_unit_cleans_every_remaining_child_before_wait(
    monkeypatch: pytest.MonkeyPatch, scheduler: dict[str, Any]
) -> None:
    state = scheduler
    monkeypatch.setattr(r, "guard_ok", lambda: [])

    def sleep(_seconds: float) -> None:
        state["tick"] += 1
        state["children"][2].returncode = 1

    monkeypatch.setattr(r.time, "sleep", sleep)
    with pytest.raises(SystemExit, match="unit failed: u2"):
        r.run(state["root"], 6)
    assert len(state["children"]) == 6
    assert state["peak"] == 6
    assert state["children"][2].calls == []
    for index, child in enumerate(state["children"]):
        if index != 2:
            assert child.calls == [signal.SIGTERM]
    first_wait = next(i for i, event in enumerate(state["events"]) if event[0] == "wait")
    terms = [
        i
        for i, event in enumerate(state["events"])
        if event[0] == "signal" and event[2] == signal.SIGTERM
    ]
    assert len(terms) == 5 and max(terms) < first_wait
    assert {e[1] for e in state["events"] if e[0] == "wait"} == {c.pid for c in state["children"]}
    assert state["unowned"].calls == []
