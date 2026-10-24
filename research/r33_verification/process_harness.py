"""Exercise actual factory callers against one synthetic process table and clock."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TABLE = Path(__file__).with_name("process_fixture_table.json")


def table() -> dict[str, Any]:
    value = json.loads(TABLE.read_text())
    if value.get("consumers") != ["startup", "runtime", "recovery"]:
        raise ValueError("caller coverage incomplete")
    return value


def _ps(rows: list[list[Any]]) -> str:
    return "\n".join(f"{pid} {parent} {nice} {argv}" for pid, parent, nice, argv in rows) + "\n"


def exercise(case: dict[str, Any], monkeypatch: Any, state: Path) -> None:
    """Use current persistence callers; native sensors remain explicitly pending."""
    from research.bl_labels import factory_persistence as fp
    from research.bl_labels import mac_driver as md
    from research.bl_labels import slowdown_resume as sr

    clock = [100.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(fp, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    text = _ps(case["rows"])
    monkeypatch.setattr(md, "_command_output", lambda _command: text)
    monkeypatch.setattr(md, "_timing_armed", lambda _path: False)
    monkeypatch.setattr(md, "_guard_snapshot", lambda _state: {"thermal": "CPU_Speed_Limit = 100"})
    by_pid = {row[0]: row for row in case["rows"]}
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda pid: {
            "birth": str(pid),
            "birth_precision": {"unit": "synthetic tick", "seconds": 1e-9},
            "executable": (
                "/opt/bin/python"
                if "python" in by_pid[pid][3] or "pytest" in by_pid[pid][3]
                else "/opt/bin/codex"
            ),
        },
    )
    observed = []
    for caller in ("startup", "runtime"):
        tracker = fp.PersistenceTracker(caller)
        try:
            if caller == "startup":
                md._factory_admission_window(tracker, state, state / "synthetic-timing-arm")
            else:
                for _index in range(7):
                    md._exclusive_factory_guard(set(), tracker)
                    sleep(5)
        except md.MacGuardError:
            observed.append(True)
        else:
            observed.append(False)
    monkeypatch.setattr(sr, "escalation", lambda _state: {"w6_epoch": 1000.0})
    monkeypatch.setattr(sr, "_operational_pin", lambda: "a" * 40)
    monkeypatch.setattr(sr.subprocess, "check_output", lambda *_a, **_k: " 1 0.1 /usr/bin/idle\n")
    try:
        sr._record_host_check_locked(state, checked_epoch=1001.0)
    except (ValueError, fp.ProcessMonitorStop):
        observed.append(True)
    else:
        observed.append(False)
    expected = [case["expected"] == "BLOCK"] * 3
    if observed != expected:
        raise AssertionError(f"{case['id']} observed={observed} expected={expected}")
