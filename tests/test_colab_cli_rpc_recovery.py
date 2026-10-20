"""Exercise the real mutation fence and verified-idle recovery without a runtime."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import cli_guard, supervisor


@pytest.mark.parametrize("phase", ["before", "after", "partial"])
def test_mutation_hang_recovers_only_absent_or_completed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = [0.0]
    events = []
    namespace: dict[str, Any] = {}
    target = tmp_path / "mutation"
    operation = (
        "from pathlib import Path\n"
        f"with Path({str(target)!r}).open('x') as handle: handle.write('once')\n"
        + ("raise RuntimeError('interrupted operation')\n" if phase == "partial" else "")
        + "result={'committed':True}"
    )

    def invoke(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == 120
        assert not cli.exec_uncertain
        code = Path(argv[argv.index("-f") + 1]).read_text()
        code = code.replace("/content/.snake-rpc-", str(tmp_path / ".snake-rpc-"))
        first = not events
        events.append("exec")
        output = io.StringIO()
        if not first or phase != "before":
            with contextlib.redirect_stdout(output):
                exec(compile(code, "<fake-kernel>", "exec"), namespace)
        if first:
            now[0] += 120
            raise subprocess.TimeoutExpired(argv, 120)
        return subprocess.CompletedProcess(argv, 0, output.getvalue(), "")

    def idle(*args: Any, **kwargs: Any) -> bool:
        assert cli.exec_uncertain
        assert events == ["exec"]
        events.append("idle")
        return True

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "kernel_idle", idle)
    if phase == "partial":
        with pytest.raises(supervisor.Refused, match="started without durable completion"):
            cli.recoverable_rpc(operation)
    else:
        assert cli.recoverable_rpc(operation) == {"committed": True}
    assert target.read_text() == "once"
    assert events == ["exec", "idle", "exec"]
    assert now[0] == 122
    records = list(tmp_path.glob(".snake-rpc-*.json"))
    assert len(records) == 1  # One operation identity across both dispatches.
    assert json.loads(records[0].read_text())["status"] == (
        "started" if phase == "partial" else "completed"
    )


@pytest.mark.parametrize("cap", ["deadline", "priced_deadline", "explicit"])
def test_mutation_recovery_keeps_first_cap_and_busy_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: str
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = [0.0]
    if cap != "explicit":
        setattr(cli, cap, 150)
    calls = []

    def invoke(argv: list[str], **kwargs: Any) -> None:
        calls.append(kwargs["timeout"])
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *args, **kwargs: True)
    with pytest.raises(supervisor.Refused, match="deadline reached during telemetry recovery"):
        cli.recoverable_rpc("result=True", **({"timeout": 150} if cap == "explicit" else {}))
    assert calls == [120, 28]
    assert now[0] == 150
    assert cli.exec_uncertain
