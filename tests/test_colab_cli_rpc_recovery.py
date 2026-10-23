"""Replay F1 control stalls against real request claims, not a result-only mock."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import cli_guard, failures, policy, rpc, supervisor

FIXTURE = json.loads((Path(__file__).parent / "fixtures/colab_control_faults_f1.json").read_text())


def _kernel(code: str, tmp_path: Path, *, synchronous: bool = True) -> str:
    code = code.replace("/content/.snake-rpc-", str(tmp_path / ".snake-rpc-"))
    if synchronous:
        code = code.replace("threading.Thread(target=worker, daemon=True).start()", "worker()")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "<fake-kernel>", "exec"), {})
    return output.getvalue()


@pytest.mark.parametrize("row", [row for row in FIXTURE["faults"] if row["elapsed_seconds"] > 119])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_observed_120s_stalls_recover_one_command_within_short_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: dict[str, Any], phase: str
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = [0.0]
    requests = []
    events = []
    target = tmp_path / "mutation"
    operation = (
        "from pathlib import Path\n"
        f"with Path({str(target)!r}).open('x') as handle: handle.write('once')\n"
        "result={'committed':True}"
    )

    def invoke(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert 0 < kwargs["timeout"] <= 30
        assert not cli.exec_uncertain
        code = Path(argv[argv.index("-f") + 1]).read_text()
        requests.append(re.search(r"# SNAKE_REQUEST_ID (\w+)", code).group(1))
        first = len(requests) == 1
        events.append("exec")
        output = _kernel(code, tmp_path) if not first or phase == "after" else ""
        if first:
            # Replay the old 120s outage, with the new client cutting it at 30s.
            now[0] += min(row["elapsed_seconds"], kwargs["timeout"])
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, output, "")

    def idle(*args: Any, **kwargs: Any) -> bool:
        assert cli.exec_uncertain
        assert 0 < kwargs["timeout"] <= 30
        events.append("idle")
        return True

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(policy, "poll_delay", lambda _: 2)
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "kernel_idle", idle)
    assert cli.recoverable_rpc(operation) == {"committed": True}
    assert target.read_text() == "once"
    assert requests[0] == requests[1]
    assert events == ["exec", "idle", "exec"]
    assert now[0] == 32
    assert len(list(tmp_path.glob(".snake-rpc-*/*.started"))) == 1
    assert len(list(tmp_path.glob(".snake-rpc-*/*.done"))) == 1
    # Recovery does not erase, discount or re-label the observed fault.
    faults = [r for r in cli.call_records if r["status"] == "failed"]
    assert len(faults) == 1
    assert faults[0]["failure_class"] == "CONTROL_RPC_TIMEOUT"


@pytest.mark.parametrize("fault", ["503", "403", "kernel"])
def test_provider_errors_retry_only_transient_channel_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    requests = []
    target = tmp_path / "mutation"
    operation = f"from pathlib import Path\nPath({str(target)!r}).write_text('once')\nresult=True"

    def invoke(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        code = Path(argv[argv.index("-f") + 1]).read_text()
        requests.append(code)
        if len(requests) == 1 and fault != "kernel":
            return subprocess.CompletedProcess(argv, 1, "", f"HTTP {fault} provider response")
        if fault == "kernel":
            code = code.replace("result=True", "raise RuntimeError('command failed')")
        return subprocess.CompletedProcess(argv, 0, _kernel(code, tmp_path), "")

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *a, **k: True)
    if fault == "503":
        assert cli.recoverable_rpc(operation) is True
        assert requests[0] == requests[1]
        assert cli.call_records[0]["failure_class"] == "CONTROL_RPC_ERROR"
    else:
        with pytest.raises(supervisor.Refused, match="403|command failed"):
            cli.recoverable_rpc(operation)
        assert len(requests) == 1


def test_duplicate_request_during_running_worker_never_relaunches(tmp_path: Path) -> None:
    release = tmp_path / "release"
    target = tmp_path / "mutation"
    code = (
        "import time\nfrom pathlib import Path\n"
        f"with Path({str(target)!r}).open('x') as handle: handle.write('once')\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(.001)\n"
        "result=42"
    )
    wrapped = rpc.request_code(code, "a" * 32, "b" * 32, "SNAKE_RPC_" + "a" * 32)
    # Real thread stays alive across independent kernel namespaces/requests.
    try:
        for _ in range(3):
            assert '"pending": true' in _kernel(wrapped, tmp_path, synchronous=False)
        assert len(list(tmp_path.glob(".snake-rpc-*/*.started"))) == 1
    finally:
        release.touch()
    # Join only our worker; no waits on unrelated runtime threads.
    for thread in threading.enumerate():
        if thread.name.endswith("(worker)"):
            thread.join(timeout=2)
    result = _kernel(wrapped, tmp_path, synchronous=False)
    assert '"value": 42' in result
    assert target.read_text() == "once"


def test_started_without_completion_and_changed_code_refuse_replay(tmp_path: Path) -> None:
    request_id, namespace = "a" * 32, "b" * 32
    root = tmp_path / (".snake-rpc-" + namespace)
    root.mkdir()
    code = "result=True"
    identity = {"request_id": request_id, "sha256": hashlib.sha256(code.encode()).hexdigest()}
    (root / (request_id + ".started")).write_text(json.dumps(identity, sort_keys=True))
    wrapped = rpc.request_code(code, request_id, namespace, "SNAKE_RPC_" + request_id)
    assert '"pending": true' in _kernel(wrapped, tmp_path)
    assert not (root / (request_id + ".done")).exists()
    changed = rpc.request_code("result=False", request_id, namespace, "SNAKE_RPC_" + request_id)
    assert "identity mismatch" in _kernel(changed, tmp_path)
    assert not (root / (request_id + ".done")).exists()


@pytest.mark.parametrize("cap", ["deadline", "priced_deadline", "explicit"])
def test_mutation_recovery_keeps_first_cap_and_busy_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: str
) -> None:
    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = [0.0]
    if cap != "explicit":
        setattr(cli, cap, 31)
    calls = []

    def invoke(argv: list[str], **kwargs: Any) -> None:
        calls.append(kwargs["timeout"])
        now[0] += kwargs["timeout"]
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(policy, "poll_delay", lambda _: 2)
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(cli_guard, "kernel_idle", lambda *a, **k: True)
    with pytest.raises(
        failures.ClassifiedFailure, match="deadline reached during telemetry recovery"
    ):
        cli.recoverable_rpc("result=True", **({"timeout": 31} if cap == "explicit" else {}))
    assert calls == [30]
    assert now[0] == 31
    assert cli.exec_uncertain


def test_control_stall_class_survives_data_transfer_wrapper() -> None:
    from research.colab_cli.transfers import TransferError

    try:
        try:
            raise subprocess.TimeoutExpired(["colab", "exec"], 30)
        except subprocess.TimeoutExpired as exc:
            raise TransferError("upload verify failed") from exc
    except TransferError as exc:
        assert failures.classify_exception(exc) == failures.FailureClass.CONTROL_RPC_TIMEOUT
    assert failures.classify_exception(RuntimeError("colab exec failed: HTTP 503")) == (
        failures.FailureClass.CONTROL_RPC_ERROR
    )


def test_pending_command_deadline_counts_fault_and_fences_new_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.bl_labels.f1_analyze import transport_faults

    cli = supervisor.CLI("fake", tmp_path / "config", "owned")
    now = [0.0]

    def invoke(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        code = Path(argv[argv.index("-f") + 1]).read_text()
        nonce = re.search(r"SNAKE_RPC_[0-9a-f]{32}", code).group()
        return subprocess.CompletedProcess(argv, 0, nonce + '{"ok":true,"pending":true}', "")

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(policy, "poll_delay", lambda _: 2)
    with pytest.raises(failures.ClassifiedFailure) as caught:
        cli.poll_rpc("result=True", timeout=3)
    assert caught.value.failure_class == failures.FailureClass.CONTROL_RPC_TIMEOUT
    assert now[0] == 3
    assert cli.exec_uncertain
    assert len(transport_faults({"cli_calls": cli.call_records})) == 1
    with pytest.raises(supervisor.Refused, match="uncertain/busy"):
        cli.rpc("result=False")


def test_historical_non_timeout_fault_stays_named_and_counted() -> None:
    from research.bl_labels.f1_analyze import transport_faults

    faults = transport_faults({"cli_calls": FIXTURE["faults"]})
    assert len(faults) == 10  # Nine timeouts plus one unresolved non-timeout exec failure.
    other = [f for f in faults if f["elapsed_seconds"] < 119]
    assert len(other) == 1
    assert other[0]["elapsed_seconds"] == 36.73803158401279
    assert failures.classify_exception(supervisor.Refused("colab exec failed: unresolved")) == (
        failures.FailureClass.CONTROL_RPC_ERROR
    )


def test_provider_traceback_source_timeout_is_not_a_stall() -> None:
    error = supervisor.Refused(
        "colab exec failed (1): │ timeout=timeout │\n" "GuardError: HTTP 503 service unavailable"
    )
    assert failures.classify_exception(error) == failures.FailureClass.CONTROL_RPC_ERROR


def test_explicit_duplicate_request_survives_client_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "mutation"
    code = (
        "from pathlib import Path\n"
        f"with Path({str(target)!r}).open('x') as handle: handle.write('once')\n"
        "result=True"
    )

    def invoke(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        wrapped = Path(argv[argv.index("-f") + 1]).read_text()
        return subprocess.CompletedProcess(argv, 0, _kernel(wrapped, tmp_path), "")

    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    for _ in range(2):
        cli = supervisor.CLI("fake", tmp_path / "config", "same-unique-session")
        assert cli.rpc(code, request_id="a" * 32) is True
    assert target.read_text() == "once"
    assert len(list(tmp_path.glob(".snake-rpc-*/*.started"))) == 1
    assert len(list(tmp_path.glob(".snake-rpc-*/*.done"))) == 1
