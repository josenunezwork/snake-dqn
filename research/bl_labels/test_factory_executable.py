"""Executable relevance regressions; synthetic tables/clocks, no label collection."""

from __future__ import annotations

import pytest

from research.bl_labels import factory_persistence as fp
from research.bl_labels import mac_driver as md
from research.bl_labels import slowdown_resume as sr
from research.bl_labels.test_factory_persistence import Clock


@pytest.mark.parametrize("tool", ["tail", "cat", "grep", "less", "rg", "git", "bash", "zsh"])
@pytest.mark.parametrize("parent", [1, 44331])
def test_generic_tools_with_research_paths_never_inherit_relevance(monkeypatch, tool, parent):
    table = (
        "44331 1 15 python -m research.x\n"
        f"44334 {parent} 0 /usr/bin/{tool} research/bl_labels/progress.log\n"
    )
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda pid: {
            "birth": str(pid),
            "birth_precision": {"unit": "fixture"},
            "executable": "/usr/bin/python3" if pid == 44331 else f"/usr/bin/{tool}",
        },
    )
    tracker = fp.PersistenceTracker("generic")
    for index in range(7):
        rows = fp.process_rows(lambda _: table, tracker.known)
        assert {r["pid"] for r in rows if r["relevant"]} == {44331}
        tracker.observe(rows, index * 5, set())
    assert not tracker.pending and tracker.latched is None


@pytest.mark.parametrize("recovery", [False, True])
def test_exact_tail_identity_does_not_preempt_guard_or_recovery(tmp_path, monkeypatch, recovery):
    """PID 44334: nice 0, non-owned tail, seven sightings spanning 33.814936 s."""
    birth = "12418367951205"
    table = (
        "44331 1 15 /bin/bash codex-handoff.sh research/bl_labels/task\n"
        "44334 44331 0 /usr/bin/tail -f research/bl_labels/progress.log\n"
    )
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda pid: {
            "birth": birth if pid == 44334 else "parent-birth",
            "executable": "/usr/bin/tail" if pid == 44334 else "/bin/bash",
            "birth_precision": {
                "seconds": 4.166666666666666e-8,
                "unit": "raw Darwin Mach absolute tick",
            },
        },
    )
    clock = Clock()
    monkeypatch.setattr(fp, "time", clock)
    monkeypatch.setattr(md, "_command_output", lambda _: table)
    if recovery:
        from research.compute import factory_interim_launcher

        monkeypatch.setattr(factory_interim_launcher, "TEST_SLOT_PATH", tmp_path / "test.lock")
        monkeypatch.setattr(sr, "escalation", lambda _: {"w6_epoch": 100})
        monkeypatch.setattr(sr, "_operational_pin", lambda: "a" * 40)
        monkeypatch.setattr(md, "_timing_armed", lambda _: False)
        monkeypatch.setattr(
            md, "_guard_snapshot", lambda _: {"ac_power": True, "lid_open": True, "thermal": "safe"}
        )
        monkeypatch.setattr(sr.subprocess, "check_output", lambda *a, **kw: "44334 0.0 tail\n")
        receipt = sr._read_triplet(sr.record_host_check(tmp_path, checked_epoch=101))
        assert not receipt["process_window"]["final_pending"]
        assert sr._host_check_after(tmp_path, 100)
    else:
        tracker = fp.PersistenceTracker("dd949d8d50bd473dbe2c54ff604eea79")
        # Even an earlier inherited classification cannot make a tool relevant.
        tracker.known[(44334, birth)] = {"relevance_reason": "relevant_descendant"}
        for elapsed in [
            0,
            5.57126625,
            11.177757083,
            16.942666208,
            22.796263583,
            28.295439583,
            33.814936125,
        ]:
            clock.now = 517432.480715375 + elapsed
            md._exclusive_factory_guard(set(), tracker)
        assert not tracker.pending and tracker.latched is None
        assert all(r.get("pid") != 44334 for r in tracker.records)


@pytest.mark.parametrize(
    "executable", ["python", "Python", "python3", "python3.12", "pytest", "pytest-3"]
)
@pytest.mark.parametrize(
    "command", ["python -m research.x", "python research/x.py", "pytest research/test_x.py"]
)
def test_only_research_interpreters_are_relevant(executable, command):
    expected = None if executable == "pytest-3" else "research_python_or_pytest"
    assert fp._seed(command, f"/usr/bin/{executable}") == expected


@pytest.mark.parametrize(
    "executable", ["tail", "bash", "python-helper", "pythonic", "pytest-helper"]
)
def test_native_executable_overrides_interpreter_text(executable):
    assert fp._seed("python -m research.x", f"/usr/bin/{executable}") is None


@pytest.mark.parametrize(
    "command", ["codex-handoff.sh task", "codex exec task", "CODEX_MODEL=gpt-6.1-sol task"]
)
def test_existing_agent_matcher_is_preserved_for_launch_shells(command):
    assert fp._seed(f"/bin/bash {command}", "/bin/bash") == "agent"


def test_exec_to_generic_tool_clears_existing_identity(monkeypatch):
    tracker = fp.PersistenceTracker("exec")
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda _: {"birth": "same", "birth_precision": "fixture", "executable": "/usr/bin/python3"},
    )
    for index in range(6):
        tracker.observe(
            fp.process_rows(lambda _: "123 1 0 python -m research.x\n", tracker.known),
            index * 5,
            set(),
        )
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda _: {"birth": "same", "birth_precision": "fixture", "executable": "/usr/bin/tail"},
    )
    tracker.observe(
        fp.process_rows(lambda _: "123 1 0 tail research/x.log\n", tracker.known), 30, set()
    )
    assert not tracker.pending and tracker.latched is None
    assert tracker.records[-2]["decision"] == "clear_absent"
