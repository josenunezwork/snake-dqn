"""Exercise current factory startup/runtime/recovery process guards from one table."""

from __future__ import annotations

import json
from pathlib import Path
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
    """Invoke actual startup/runtime vetoes plus the recovery host-check path."""
    from research.bl_labels import mac_driver, slowdown_resume

    text = _ps(case["rows"])
    monkeypatch.setattr(mac_driver, "_command_output", lambda _command: text)
    observed: list[bool] = []
    for _caller in ("startup", "runtime"):
        try:
            mac_driver._exclusive_factory_guard(set())
        except mac_driver.MacGuardError:
            observed.append(True)
        else:
            observed.append(False)
    monkeypatch.setattr(slowdown_resume, "escalation", lambda _state: {"w6_epoch": 1000.0})
    monkeypatch.setattr(slowdown_resume, "_operational_pin", lambda: "a" * 40)
    monkeypatch.setattr(
        mac_driver, "_guard_snapshot", lambda _state: {"thermal": "CPU_Speed_Limit = 100"}
    )
    monkeypatch.setattr(
        slowdown_resume.subprocess, "check_output", lambda *_a, **_k: " 1 0.1 /usr/bin/idle\n"
    )
    try:
        slowdown_resume._record_host_check_locked(state, checked_epoch=1001.0)
    except Exception:
        observed.append(True)
    else:
        observed.append(False)
    expected = [case["expected"] == "BLOCK"] * 3
    if observed != expected:
        raise AssertionError(f"{case['id']} observed={observed} expected={expected}")
