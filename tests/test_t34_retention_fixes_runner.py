"""FIX-FIRST runner and host-guard contracts for T34 retention.

These tests use only injected samples/adapters.  They never start a service,
tmux session, efficacy bank, or real timing probe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.t34_pathb import host_guard_v2, retention_v2

REPO = Path(__file__).resolve().parents[1]
PROFILE = json.loads((REPO / "research/t34_pathb/RETENTION_PROFILE_V2.json").read_text())
POLICY = {**PROFILE["load_policy"], "max_contended_fraction": 0.02}


def _probe(fps: int, *, update_p95: float = 1.0, overruns: int = 0) -> dict:
    return {
        "fps": fps,
        "ticks": 10,
        "warmup": 2,
        "contended": False,
        "update_ms": {"n": 10, "p95": update_p95, "p99": update_p95, "max": update_p95},
        "full_tick_ms": {"n": 10, "p95": 2.0, "p99": 2.0, "max": 2.0},
        "achieved_fps": float(fps),
        "overruns": overruns,
    }


def _timing(*, update_12: float = 1.0, update_24: float = 1.0) -> dict:
    return {
        "probes": {
            "bare-12": _probe(12),
            "bare-24": _probe(24),
            "paced-12": _probe(12, update_p95=update_12),
            "paced-24": _probe(24, update_p95=update_24),
        }
    }


def test_host_guard_checks_are_cached_and_only_row_boundaries_pause():
    """A tick check is cached; a transient fault is handled between rows."""
    now = [0.0]
    samples = iter(
        [
            {"contended": True, "reasons": ["ac_power_lost"]},
            {"contended": False, "reasons": []},
            {"contended": False, "reasons": []},
        ]
    )
    sleeps: list[float] = []

    guard = host_guard_v2.HostGuard(
        POLICY,
        clock=lambda: now[0],
        sample=lambda: next(samples),
    )

    guard.start()
    first = guard.check()
    assert first["contended"] is True
    assert guard.check()["contended"] is True
    assert len(guard.records) == 1
    assert sleeps == []

    guard._capture()
    guard.between_rows()
    assert guard.records[-1]["contended"] is False
    assert len(guard.records) == 2
    guard.close()


def test_host_guard_persistent_contention_becomes_inconclusive_between_rows():
    clock_values = iter([1.0, 1.0, 121.0])

    guard = host_guard_v2.HostGuard(
        POLICY,
        clock=lambda: next(clock_values),
        sample=lambda: {"contended": True, "reasons": ["thermal_not_ok"]},
    )

    guard.start()
    with pytest.raises(host_guard_v2.HostContentionError, match="thermal_not_ok"):
        guard.between_rows()
    assert len(guard.records) == 1
    guard.close()


@pytest.mark.parametrize(
    ("power", "expected"),
    [
        ((False, True, True), "ac_power_lost"),
        ((True, False, True), "lid_closed"),
        ((True, True, False), "thermal_not_ok"),
    ],
)
def test_host_guard_measures_power_lid_and_thermal_changes(monkeypatch, power, expected):
    monkeypatch.setattr(host_guard_v2, "_power_thermal", lambda: power)
    monkeypatch.setattr(host_guard_v2, "_own_pids", lambda: {os.getpid()})
    monkeypatch.setattr(host_guard_v2, "_processes", lambda: [])
    monkeypatch.setattr(host_guard_v2.os, "getloadavg", lambda: (0.0, 0.0, 0.0))

    record = host_guard_v2.HostGuard(POLICY)._measure()

    assert record["contended"] is True
    assert expected in record["reasons"]


def test_host_guard_requires_sustained_other_process_over_fifty_percent(monkeypatch):
    """A brief >50% process is diagnostic; persistence makes it contention."""
    clock_values = iter([0.0, 61.0])
    row = {"pid": 4242, "ppid": 1, "cpu_percent": 51.0, "command": "python worker.py"}
    monkeypatch.setattr(host_guard_v2, "_power_thermal", lambda: (True, True, True))
    monkeypatch.setattr(host_guard_v2, "_own_pids", lambda: {os.getpid()})
    monkeypatch.setattr(host_guard_v2, "_processes", lambda: [row])
    monkeypatch.setattr(host_guard_v2.os, "getloadavg", lambda: (2.0, 2.0, 2.0))
    guard = host_guard_v2.HostGuard(POLICY, clock=lambda: next(clock_values))

    first = guard._measure()
    second = guard._measure()
    assert first["other_processes_over_cpu_limit"] == []
    assert second["other_processes_over_cpu_limit"] == [row]
    assert "other_cpu_sustained" in second["reasons"]


def test_host_guard_excludes_ancestors_and_ignores_low_cpu_research_siblings(monkeypatch):
    monkeypatch.setattr(host_guard_v2.os, "getpid", lambda: 10)
    monkeypatch.setattr(
        host_guard_v2.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "10 9\n9 8\n11 10\n100 1\n",
    )
    owned = host_guard_v2._own_pids()
    assert {8, 9, 10, 11}.issubset(owned)
    assert 100 not in owned

    ancestor = {
        "pid": 9,
        "ppid": 8,
        "cpu_percent": 80.0,
        "command": "python research/phase0_runner.py",
    }
    sibling = {
        "pid": 100,
        "ppid": 1,
        "cpu_percent": 1.0,
        "command": "python research/phase0_runner.py",
    }
    monkeypatch.setattr(host_guard_v2, "_own_pids", lambda: {8, 9, 10, 11})
    monkeypatch.setattr(host_guard_v2, "_processes", lambda: [ancestor, sibling])
    monkeypatch.setattr(host_guard_v2, "_power_thermal", lambda: (True, True, True))
    monkeypatch.setattr(host_guard_v2.os, "getloadavg", lambda: (2.0, 2.0, 2.0))
    record = host_guard_v2.HostGuard(POLICY)._measure()

    assert record["known_heavy_processes"] == []
    assert "known_heavy_process" not in record["reasons"]
    assert all(item["pid"] != 9 for item in record["known_heavy_processes"])


def test_host_guard_counts_a_research_sibling_only_after_sustained_high_cpu(monkeypatch):
    clock_values = iter([0.0, 61.0])
    sibling = {
        "pid": 100,
        "ppid": 1,
        "cpu_percent": 51.0,
        "command": "python research/phase0_runner.py",
    }
    monkeypatch.setattr(host_guard_v2, "_power_thermal", lambda: (True, True, True))
    monkeypatch.setattr(host_guard_v2, "_own_pids", lambda: {os.getpid()})
    monkeypatch.setattr(host_guard_v2, "_processes", lambda: [sibling])
    monkeypatch.setattr(host_guard_v2.os, "getloadavg", lambda: (2.0, 2.0, 2.0))
    guard = host_guard_v2.HostGuard(POLICY, clock=lambda: next(clock_values))

    assert guard._measure()["known_heavy_processes"] == []
    record = guard._measure()
    assert record["known_heavy_processes"] == [sibling]
    assert "known_heavy_process" in record["reasons"]


def test_timing_12fps_p95_is_reported_but_24fps_p95_is_a_gate():
    allowed = retention_v2._timing_passes(_timing(update_12=12.0, update_24=11.0))
    assert allowed == (True, "PASS")
    failed = retention_v2._timing_passes(_timing(update_12=1.0, update_24=12.0))
    assert failed[0] is False
    assert failed[1].startswith("KILL-timing")


@pytest.mark.parametrize("elapsed", [None, -1.0, float("nan"), float("inf")])
def test_phase_debit_rejects_missing_negative_or_nonfinite_caps(elapsed):
    with pytest.raises(retention_v2.RegistrationError, match="debits"):
        retention_v2._phase_debit({"elapsed_seconds": elapsed, "cpu_seconds": 0.0})


def test_timing_failure_prevents_gameplay_even_with_injected_adapter(monkeypatch):
    opened: list[tuple] = []
    adapter = SimpleNamespace(
        open_terminal=lambda *args: opened.append(args),
        open_watch=lambda *args: opened.append(args),
    )
    monkeypatch.setattr(retention_v2, "_execution_guard", lambda: None)
    monkeypatch.setattr(retention_v2, "require_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_require_committed_pushed_registration", lambda *_args: None)

    with pytest.raises(retention_v2.RegistrationError, match="timing gate did not pass"):
        retention_v2.run_precheck(
            [],
            open_terminal=adapter.open_terminal,
            timing_gate=_timing(update_24=12.0),
            registration={},
            registration_path=Path(__file__),
        )
    assert opened == []


def test_live_phases_require_published_registration_even_with_mock_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(retention_v2, "_execution_guard", lambda: None)
    monkeypatch.setattr(retention_v2, "require_registration", lambda *_args: None)
    adapter = SimpleNamespace(open_watch=lambda *_args: pytest.fail("gameplay started"))

    with pytest.raises(retention_v2.RegistrationError, match="registration|timing"):
        retention_v2.run_timing_gate(tmp_path / "timing", {}, adapter=adapter)
    with pytest.raises(retention_v2.RegistrationError, match="registration|timing"):
        retention_v2.run_registered(tmp_path / "run", {}, {}, adapter=adapter)
    with pytest.raises(retention_v2.RegistrationError, match="registration|timing"):
        retention_v2.run_precheck([], open_terminal=adapter.open_watch)


def test_committed_registration_guard_requires_exact_head_bytes_and_published_tip(
    monkeypatch,
    tmp_path,
):
    authoring = tmp_path / "authoring-worktree"
    authoring.mkdir()
    path = authoring / "registration.json"
    registration = {"ratified": True, "source_commit": "source"}
    path.write_text(json.dumps(registration, sort_keys=True) + "\n")
    try:
        head_sha = "a" * 40
        head_bytes = path.read_bytes()

        def git(args, **_kwargs):
            if args[:3] == ["git", "rev-parse", "--show-toplevel"]:
                return str(authoring) + "\n"
            if args[:2] == ["git", "show"]:
                return head_bytes
            if args[:2] == ["git", "ls-remote"]:
                return f"{head_sha}\trefs/heads/main\n"
            raise AssertionError(args)

        monkeypatch.setattr(retention_v2.subprocess, "check_output", git)
        retention_v2._require_committed_pushed_registration(path, registration)

        path.write_text(json.dumps({**registration, "source_commit": "changed"}, sort_keys=True))
        with pytest.raises(retention_v2.RegistrationError, match="differ from HEAD"):
            retention_v2._require_committed_pushed_registration(path, registration)
        path.write_bytes(head_bytes)
        with pytest.raises(retention_v2.RegistrationError, match="supplied registration object"):
            retention_v2._require_committed_pushed_registration(
                path, {**registration, "source_commit": "different"}
            )
    finally:
        path.unlink(missing_ok=True)


def test_committed_registration_guard_rejects_unpublished_head(monkeypatch):
    path = REPO / ".t34-registration-guard-test.json"
    registration = {"ratified": True}
    path.write_text(json.dumps(registration, sort_keys=True) + "\n")
    try:

        def git(args, **_kwargs):
            if args[:3] == ["git", "rev-parse", "--show-toplevel"]:
                return str(REPO) + "\n"
            if args[:2] == ["git", "show"]:
                if args[2].startswith("b" * 40):
                    raise subprocess.CalledProcessError(1, args)
                return path.read_bytes()
            if args[:2] == ["git", "ls-remote"]:
                return f"{'b' * 40}\trefs/heads/main\n"
            raise AssertionError(args)

        monkeypatch.setattr(retention_v2.subprocess, "check_output", git)
        with pytest.raises(retention_v2.RegistrationError, match="not pushed"):
            retention_v2._require_committed_pushed_registration(path, registration)
    finally:
        path.unlink(missing_ok=True)


def test_analyze_cli_is_descriptive_and_preserves_create_only_decision(
    monkeypatch, tmp_path, capsys
):
    official = {
        "schema": "t34-pathb-v2-decision/v1",
        "verdict": "KILL",
        "reason": "official timing result",
        "release_authorized": False,
    }
    (tmp_path / "decision.json").write_text(json.dumps(official))

    def descriptive(_root, *, unpaced_shards):
        assert unpaced_shards == []
        return {**official, "verdict": "GO", "reason": "descriptive result", "mass": 12.0}

    monkeypatch.setattr(retention_v2, "analyze_path", descriptive)
    monkeypatch.setattr(
        sys,
        "argv",
        ["retention_v2", "analyze", "--out", str(tmp_path)],
    )

    assert retention_v2.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["analysis_kind"] == "descriptive_reanalysis"
    assert output["official_decision"] == official
    assert output["descriptive_statistics"] == {
        "mass": 12.0,
        "release_authorized": False,
        "schema": official["schema"],
    }
    assert json.loads((tmp_path / "decision.json").read_text()) == official


def test_timing_gate_generic_fault_writes_one_inconclusive_decision(monkeypatch, tmp_path):
    monkeypatch.setattr(retention_v2, "require_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_execution_guard", lambda: None)
    monkeypatch.setattr(retention_v2, "_require_committed_pushed_registration", lambda *_args: None)
    fake_guard = SimpleNamespace(
        records=[],
        check=lambda **_kwargs: {"contended": False},
        admit=lambda: None,
        start=lambda: None,
        between_rows=lambda **_kwargs: None,
        summary=lambda **_kwargs: {},
        close=lambda: None,
    )
    monkeypatch.setattr(retention_v2, "HostGuard", lambda _policy, **_kwargs: fake_guard)

    def fail_open(*_args):
        raise RuntimeError("missing timing session")

    registration = {
        "load_profile": POLICY,
        "timing_row": {},
        "timing_ticks": 1,
        "timing_warmup": 0,
    }
    adapter = SimpleNamespace(open_watch=fail_open)
    out = tmp_path / "timing"
    with pytest.raises(RuntimeError, match="missing timing session"):
        retention_v2.run_timing_gate(
            out,
            registration,
            adapter=adapter,
            registration_path=Path(__file__),
        )

    decision = out / "decision.json"
    payload = json.loads(decision.read_text())
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["release_authorized"] is False
    with pytest.raises(FileExistsError):
        retention_v2._write_new(decision, {**payload, "reason": "overwrite"})


def test_partial_unconditional_fault_writes_one_create_only_decision(monkeypatch, tmp_path):
    monkeypatch.setattr(retention_v2, "_execution_guard", lambda: None)
    monkeypatch.setattr(retention_v2, "require_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "require_timing_gate", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_require_committed_pushed_registration", lambda *_args: None)
    fake_guard = SimpleNamespace(
        records=[],
        check=lambda **_kwargs: {"contended": False},
        admit=lambda: None,
        start=lambda: None,
        between_rows=lambda **_kwargs: None,
        summary=lambda **_kwargs: {},
        close=lambda: None,
    )
    monkeypatch.setattr(retention_v2, "HostGuard", lambda _policy, **_kwargs: fake_guard)
    monkeypatch.setattr(retention_v2, "TERMINAL_PER_MIX", 1)
    monkeypatch.setattr(retention_v2, "PRODUCT_SESSIONS", 1)

    def fail_open(*_args):
        raise retention_v2.UnconditionalFault("nonfinite_q")

    adapter = SimpleNamespace(open_terminal=fail_open, open_watch=fail_open)
    timing_gate = {"elapsed_seconds": 0.0, "cpu_seconds": 0.0}
    registration = {
        "load_profile": POLICY,
        "terminal_rows": [
            {"seed": 1, "mix": "scripted"},
            {"seed": 2, "mix": "frozen"},
            {"seed": 3, "mix": "mixed"},
        ],
        "product_rows": [{"seed": 4, "mix": "mixed"}],
    }
    registration_path = REPO / "tests/test_t34_retention_fixes_runner.py"
    with pytest.raises(retention_v2.UnconditionalFault, match="nonfinite_q"):
        retention_v2.run_registered(
            tmp_path / "run",
            registration,
            {
                "timing_gate_sha256": retention_v2._canonical_sha(timing_gate),
                "registration_sha256": retention_v2._canonical_sha(registration),
                "elapsed_seconds": 0.0,
                "cpu_seconds": 0.0,
            },
            adapter=adapter,
            timing_gate=timing_gate,
            registration_path=registration_path,
        )

    decision = tmp_path / "run" / "decision.json"
    before = decision.read_bytes()
    payload = json.loads(before)
    assert payload["verdict"] == "KILL"
    assert payload["release_authorized"] is False
    with pytest.raises(FileExistsError):
        retention_v2._write_new(decision, {**payload, "reason": "overwrite"})
    assert decision.read_bytes() == before


def test_precheck_admission_refuses_without_terminal_before_any_world(monkeypatch, tmp_path):
    monkeypatch.setattr(retention_v2, "require_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "require_timing_gate", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_require_committed_pushed_registration", lambda *_args: None)
    monkeypatch.setattr(retention_v2, "_execution_guard", lambda: None)
    observed = [{"contended": True, "reasons": ["ac_power_lost"]}]

    def refuse(**_kwargs):
        raise host_guard_v2.HostAdmissionError("ac_power_lost")

    guard = SimpleNamespace(
        records=observed,
        check=refuse,
        admit=refuse,
        start=lambda: None,
        between_rows=refuse,
        summary=lambda **_kwargs: {},
        close=lambda: None,
    )
    made = []
    monkeypatch.setattr(
        retention_v2,
        "HostGuard",
        lambda policy, **_kwargs: made.append(policy) or guard,
    )
    out = tmp_path / "precheck"
    with pytest.raises(host_guard_v2.HostAdmissionError):
        retention_v2.run_precheck(
            [],
            open_terminal=lambda *_args: pytest.fail("world constructed"),
            registration={"load_profile": POLICY},
            registration_path=Path(__file__),
            timing_gate={"elapsed_seconds": 10.0, "cpu_seconds": 2.0},
            out=out,
        )
    assert made == [POLICY]
    assert not out.exists()
    with pytest.raises(host_guard_v2.HostAdmissionError):
        retention_v2.run_precheck(
            [],
            registration={"load_profile": POLICY},
            registration_path=Path(__file__),
            timing_gate={"elapsed_seconds": 10.0, "cpu_seconds": 2.0},
            out=out,
        )


def test_precheck_cannot_reuse_a_different_timing_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(retention_v2, "require_timing_gate", lambda *_args: None)
    with pytest.raises(retention_v2.RegistrationError, match="does not bind"):
        retention_v2.run_registered(
            tmp_path / "run",
            {},
            {"timing_gate_sha256": "wrong"},
            timing_gate={"elapsed_seconds": 0.0, "cpu_seconds": 0.0},
            registration_path=Path(__file__),
            adapter=SimpleNamespace(open_terminal=lambda *_args: pytest.fail("world constructed")),
        )
