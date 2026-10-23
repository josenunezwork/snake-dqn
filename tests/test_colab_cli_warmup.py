"""Offline startup probes, including cold kernels and first-cap-wins cleanup."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.colab_cli import cli_guard, supervisor


def _clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [0.0]
    monkeypatch.setattr(
        supervisor,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)
        ),
    )
    return now


def _ack(args: tuple[str, ...]) -> str:
    code = Path(args[args.index("-f") + 1]).read_text()
    return re.search(r"SNAKE_RPC_[0-9a-f]+", code).group(0) + json.dumps(
        {"ok": True, "value": {"ready": True}}
    )


def test_slow_first_exec_recovers_only_after_independent_idle_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    now = _clock(monkeypatch)
    cli = supervisor.CLI("never-invoked", tmp_path / "session.json", "cold")
    cli.deadline = 600
    calls = []
    idle_reads = []
    monkeypatch.setattr(cli_guard, "runtime_ready", lambda *_args, **_kwargs: True)

    def idle(*_args: object, **_kwargs: object) -> bool:
        idle_reads.append(now[0])
        return now[0] >= 134.1

    def call(*args: str, timeout: float) -> str:
        calls.append(now[0])
        assert timeout <= 30
        if len(calls) == 1:
            now[0] += timeout
            raise subprocess.TimeoutExpired("fake first exec", timeout)
        return _ack(args)

    monkeypatch.setattr(cli_guard, "kernel_idle", idle)
    monkeypatch.setattr(cli, "_call_once", call)
    cli.wait_runtime_ready(tmp_path)
    assert calls == [0, 135]
    assert idle_reads and min(idle_reads) >= 30
    assert cli.warmup_seconds == 135
    assert not cli.exec_uncertain
    evidence = json.loads((tmp_path / "warmup-measurement.json").read_text())
    assert evidence["status"] == "PASS"
    assert evidence["elapsed_seconds"] == 135


@pytest.mark.parametrize(
    ("wall", "priced", "bound"), [(600, 600, 300), (40, 600, 40), (600, 42, 42)]
)
def test_unready_runtime_is_bounded_by_warmup_wall_and_unit_deadlines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wall: float, priced: float, bound: float
) -> None:
    now = _clock(monkeypatch)
    cli = supervisor.CLI("never-invoked", tmp_path / "session.json", "cold")
    cli.deadline, cli.priced_deadline = wall, priced
    timeouts = []

    def unready(*_args: object, timeout: float) -> bool:
        timeouts.append(timeout)
        now[0] += timeout
        return False

    monkeypatch.setattr(cli_guard, "runtime_ready", unready)
    with pytest.raises(supervisor.Refused, match="readiness warmup deadline"):
        cli.wait_runtime_ready(tmp_path)
    assert cli.warmup_seconds == bound
    assert max(timeouts) <= 30
    assert (cli.deadline, cli.priced_deadline) == (wall, priced)
    assert json.loads((tmp_path / "warmup-measurement.json").read_text())["status"] == "INCOMPLETE"


def test_auth_error_after_uncertain_exec_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clock(monkeypatch)
    cli = supervisor.CLI("never-invoked", tmp_path / "session.json", "cold")
    cli.exec_uncertain = True
    calls = []

    def forbidden(*_args: object, **_kwargs: object) -> bool:
        calls.append(1)
        raise cli_guard.GuardError("HTTP 403 forbidden")

    monkeypatch.setattr(cli_guard, "runtime_ready", forbidden)
    with pytest.raises(cli_guard.GuardError, match="403"):
        cli.wait_runtime_ready(tmp_path)
    assert calls == [1]
    assert cli.exec_uncertain


def test_unknown_kernel_after_uncertain_probe_is_never_resubmitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    now = _clock(monkeypatch)
    cli = supervisor.CLI("never-invoked", tmp_path / "session.json", "cold")
    calls = []
    monkeypatch.setattr(cli_guard, "runtime_ready", lambda *_args, **_kwargs: True)

    def call(*args: str, timeout: float) -> str:
        calls.append(args)
        now[0] += timeout
        raise subprocess.TimeoutExpired("first exec", timeout)

    def unknown(*_args: object, **_kwargs: object) -> bool:
        raise cli_guard.GuardError("tracked session lacks kernel probe metadata")

    monkeypatch.setattr(cli, "_call_once", call)
    monkeypatch.setattr(cli_guard, "kernel_idle", unknown)
    with pytest.raises(cli_guard.GuardError, match="lacks kernel probe metadata"):
        cli.wait_runtime_ready(tmp_path)
    assert len(calls) == 1
    assert cli.exec_uncertain


def test_contents_transient_failure_is_polled_before_first_exec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    now = _clock(monkeypatch)
    cli = supervisor.CLI("never-invoked", tmp_path / "session.json", "cold")
    probes = []

    def contents(*_args: object, **_kwargs: object) -> bool:
        probes.append(now[0])
        if len(probes) == 1:
            raise cli_guard.GuardError("HTTP 503 unavailable")
        return len(probes) > 2

    monkeypatch.setattr(cli_guard, "runtime_ready", contents)
    monkeypatch.setattr(cli, "_call_once", lambda *args, **_kwargs: _ack(args))
    cli.wait_runtime_ready(tmp_path)
    assert probes == [0, 5, 10]
    assert cli.warmup_seconds == 10


@pytest.mark.parametrize("error", ["RuntimeError('bad state')", "TimeoutError('kernel failed')"])
def test_acknowledged_kernel_error_fails_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: str
) -> None:
    _clock(monkeypatch)
    cli = supervisor.CLI("never-invoked", tmp_path / "session.json", "cold")
    monkeypatch.setattr(cli_guard, "runtime_ready", lambda *_args, **_kwargs: True)

    def call(*args: str, **_kwargs: object) -> str:
        nonce = _ack(args).split("{", 1)[0]
        return nonce + json.dumps({"ok": False, "error": error})

    monkeypatch.setattr(cli, "_call_once", call)
    with pytest.raises(supervisor.Refused, match="kernel error"):
        cli.wait_runtime_ready(tmp_path)
    assert len(cli.call_records) == 1
    assert not cli.exec_uncertain


def test_permanent_exec_failure_is_not_retried_despite_uncertainty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clock(monkeypatch)
    cli = supervisor.CLI("never-invoked", tmp_path / "session.json", "cold")
    monkeypatch.setattr(cli_guard, "runtime_ready", lambda *_args, **_kwargs: True)

    def forbidden(*_args: str, **_kwargs: object) -> str:
        raise supervisor.Refused("colab exec failed (1): HTTP 403 forbidden")

    monkeypatch.setattr(cli, "_call_once", forbidden)
    with pytest.raises(supervisor.Refused, match="403"):
        cli.wait_runtime_ready(tmp_path)
    assert len(cli.call_records) == 1
    assert cli.exec_uncertain


def test_requalification_identity_refusal_fails_fast_from_exact_receipt_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Replay acceptance-1792727428: one fast exec failure, then empty GETs.

    Rich includes a neighboring reconnect-required raise in its source excerpt.
    That text is not the exception; it must not turn a pre-POST guard refusal
    into 50 probes and a misleading 300-second busy/unknown timeout.
    """
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/colab_kernel_identity_refusal.json").read_text()
    )
    receipt, warmup = fixture["receipt"], fixture["warmup"]
    exec_call = next(call for call in receipt["cli_calls"] if call["command"] == "exec")
    assert warmup["attempts"] == 50 and 300 <= warmup["elapsed_seconds"] < 301
    assert receipt["contents_transfers"] == receipt["exports"] == []
    assert receipt["terminal"] is None and receipt["stop_verified"] is True
    error = ast.literal_eval(warmup["last_probe_error"][len("Refused(") : -1])
    stderr = error.split("colab exec failed (1): ", 1)[1]
    assert 'raise GuardError("tracked kernel reconnect required' in stderr
    assert stderr.endswith("GuardError: kernel creation does not match tracked runtime identity\n")
    now = _clock(monkeypatch)
    cli = supervisor.CLI("never-invoked", tmp_path / "session.json", receipt["session"])
    probes = []
    idle_reads = []

    def ready(*_args: object, **_kwargs: object) -> bool:
        probes.append(now[0])
        return True

    def empty_inventory(*_args: object, **_kwargs: object) -> bool:
        idle_reads.append(now[0])
        return False

    def invoke(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        now[0] += exec_call["elapsed_seconds"]
        return subprocess.CompletedProcess(argv, 1, "", stderr)

    monkeypatch.setattr(cli_guard, "runtime_ready", ready)
    monkeypatch.setattr(cli_guard, "kernel_idle", empty_inventory)
    monkeypatch.setattr(supervisor.subprocess, "run", invoke)
    with pytest.raises(supervisor.Refused, match="kernel creation does not match"):
        cli.wait_runtime_ready(tmp_path)
    assert probes == [0] and idle_reads == []
    assert len(cli.call_records) == 1
    assert cli.exec_uncertain
    measured = json.loads((tmp_path / "warmup-measurement.json").read_text())
    assert measured["attempts"] == 1
    assert measured["elapsed_seconds"] == exec_call["elapsed_seconds"]


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        ("kernel creation does not match tracked runtime identity", False),
        ("kernel creation returned malformed kernel identity", False),
        ("kernel creation already attempted; refusing uncertain retry", False),
        ("tracked kernel reconnect required; kernel already exists", True),
    ],
)
def test_rich_source_excerpt_does_not_classify_kernel_guard_exception(
    error: str, transient: bool
) -> None:
    from research.colab_cli import policy

    traceback = (
        "colab exec failed (1): │ timeout=timeout │\n"
        '│ raise GuardError("kernel creation does not match tracked runtime identity") │\n'
        '│ raise GuardError("tracked kernel reconnect required; kernel already exists") │\n'
        f"GuardError: {error}\n"
    )
    assert policy.transient_error(supervisor.Refused(traceback)) is transient
