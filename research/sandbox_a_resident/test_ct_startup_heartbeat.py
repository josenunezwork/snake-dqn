"""Startup status cadence, closed admission, and independent work-deadline proof."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research.sandbox_a_resident import ct_liveness, ct_progress, ct_resume


def _row(phase: str = "imports") -> dict:
    return {
        "schema": "t34-cf-heartbeat/v1",
        "study": "t34-cf-1028",
        "phase": phase,
        "elapsed_sec": 15.0,
    }


def test_ast_heartbeat_inventory_covers_all_emitters_and_literal_phases() -> None:
    keys, phases = ct_progress.heartbeat_inventory()
    assert keys == frozenset(_row())
    assert {"imports", "forced-state-checks", "forced-state-cpu-world", "fit"} <= phases
    seen = {"imports"}
    for name in ct_progress.HEARTBEAT_SOURCES:
        tree = ast.parse((ct_progress.HERE / name).read_text())
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "set_phase"
            ):
                assert isinstance(call.args[0], ast.Constant)
                seen.add(call.args[0].value)
    assert phases == seen
    # Admission remains torch-free and cannot execute the numerical producer.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from research.sandbox_a_resident.ct_progress import "
            "heartbeat_inventory; import sys; heartbeat_inventory(); "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
    )
    assert result.returncode == 0


@pytest.mark.parametrize("phase", sorted(ct_progress.heartbeat_inventory()[1]))
def test_each_ast_phase_healthy_and_fault_twins(phase: str, tmp_path: Path) -> None:
    line = "CT_HEARTBEAT " + json.dumps(_row(phase), sort_keys=True)
    ct_resume._scan_job_tail(line, label="healthy")
    ct_resume._scan_json_faults({"poll": {"snapshot": {"job_log_tail": line}}}, label="poll")
    (tmp_path / "progress-history.log").write_text(line + "\n")
    ct_resume._check_attempt_base_fault_evidence(tmp_path)
    for bad in (line + " LookupError: injected", line + "\nLookupError: injected"):
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_job_tail(bad, label="fault")
        with pytest.raises(ct_resume.CTResumeRefused):
            ct_resume._scan_json_faults({"poll": {"snapshot": {"job_log_tail": bad}}}, label="poll")
    (tmp_path / "progress-history.log").write_text(line + "\nLookupError: injected\n")
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._check_attempt_base_fault_evidence(tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "foreign"),
        ("study", "foreign"),
        ("phase", "unknown"),
        ("phase", "imports LookupError: injected"),
        ("phase", []),
        ("elapsed_sec", -1),
        ("elapsed_sec", True),
        ("elapsed_sec", float("nan")),
        ("elapsed_sec", float("inf")),
        ("error", "LookupError: injected"),
    ],
)
def test_heartbeat_drift_and_embedded_fault_refuse(key: str, value: object) -> None:
    row = {**_row(), key: value}
    with pytest.raises(ct_resume.CTResumeRefused):
        ct_resume._scan_job_tail("CT_HEARTBEAT " + json.dumps(row), label="fault")


def test_periodic_status_and_phase_markers_leave_rng_and_watchdog_bytes_unchanged(
    tmp_path: Path,
) -> None:
    import random

    import numpy as np
    import torch

    from research.sandbox_a_resident.stall_watchdog import StallWatchdog

    output = tmp_path / "out"
    output.mkdir()
    watchdog = StallWatchdog(output / "stall-watchdog", 900)
    watchdog.start("startup")
    before = (watchdog.diagnostics_dir / "stall-watchdog.json").read_bytes()
    py_state, np_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    status = ct_liveness.StatusHeartbeat(output, interval=0.02)
    try:
        status.start()
        status.set_phase("forced-state-checks")
        time.sleep(0.12)
    finally:
        status.close()
        watchdog.close()
    assert (watchdog.diagnostics_dir / "stall-watchdog.json").read_bytes() == before
    assert random.getstate() == py_state
    actual = np.random.get_state()
    assert actual[0] == np_state[0] and actual[2:] == np_state[2:]
    assert actual[1].tobytes() == np_state[1].tobytes()
    assert torch.equal(torch.get_rng_state(), torch_state)
    lines = (output / "progress-history.log").read_text().splitlines()
    assert len(lines) >= 4
    for line in lines:
        ct_resume._scan_job_tail(line, label="real-status")


def test_live_status_does_not_prevent_native_work_watchdog_expiry(tmp_path: Path) -> None:
    # Real subprocess death proves status cannot rearm the watchdog even while
    # many healthy heartbeat lines keep reaching stdout.
    script = """
import time
from pathlib import Path
from research.sandbox_a_resident.ct_liveness import StatusHeartbeat
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
out = Path(__import__('sys').argv[1]); out.mkdir()
watchdog = StallWatchdog(out / 'stall-watchdog', 0.6); watchdog.start('startup')
status = StatusHeartbeat(out, interval=0.05); status.start()
time.sleep(10)
raise SystemExit('watchdog did not fire')
"""
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "out")],
        capture_output=True,
        text=True,
        timeout=5,
    )
    # Both exact hard-stop outcomes are required: native _exit(1), or the
    # independent guard's SIGKILL if CPython's best-effort dump cannot return.
    assert result.returncode in (1, -signal.SIGKILL)
    assert time.monotonic() - started < 5
    assert result.stdout.count("CT_HEARTBEAT ") >= 3
    dump = (tmp_path / "out/stall-watchdog/faulthandler-stacks.log").read_text()
    assert "Timeout" in dump and "time.sleep" not in result.stdout
    record = json.loads((tmp_path / "out/stall-watchdog/stall-watchdog.json").read_text())
    assert record["stage"] == "startup" and record["timeout_seconds"] == 0.6


def test_status_thread_stops_before_history_sealing(tmp_path: Path) -> None:
    output = tmp_path / "out"
    heartbeat = ct_liveness.start_status(output)
    assert ct_liveness.owns_output(output)
    ct_liveness.set_phase("imports")
    cadence = ct_liveness.finish_status()
    assert cadence["line_count"] == 2 and cadence["max_gap_seconds"] < 1
    assert not heartbeat._thread.is_alive()
    history = output / "progress-history.log"
    before = hashlib.sha256(history.read_bytes()).hexdigest()
    ct_liveness.set_phase("fit")
    assert hashlib.sha256(history.read_bytes()).hexdigest() == before
    assert ct_liveness.finish_status() is None


@pytest.mark.parametrize("interval", [0, -1, 30.1, float("inf"), float("nan")])
def test_status_interval_cannot_exceed_thirty_seconds(tmp_path: Path, interval: float) -> None:
    with pytest.raises(ValueError):
        ct_liveness.StatusHeartbeat(tmp_path, interval=interval)


def test_entrypoint_emits_before_numerical_imports_and_stops_on_import_failure(
    tmp_path: Path,
) -> None:
    # The entrypoint starts the real heartbeat; intercept only runpy's import
    # boundary to verify it is already live before the first numerical import.
    script = """
import runpy, sys, threading
from research.sandbox_a_resident import ct_entrypoint, ct_liveness
def boundary(*args, **kwargs):
    assert 'torch' not in sys.modules
    assert ct_liveness._CURRENT._thread.is_alive()
    raise ImportError('fixture boundary')
runpy.run_module = boundary
sys.argv = ['entry', 'runner', '--output', sys.argv[1]]
try: ct_entrypoint.main()
except ImportError: pass
assert ct_liveness._CURRENT is None
assert not any(t.name == 'ct-status' for t in threading.enumerate())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "out")],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    ct_resume._scan_job_tail(result.stdout, label="early-import")
    assert json.loads(result.stdout.partition(" ")[2])["phase"] == "imports"


@pytest.mark.parametrize(
    "fault",
    [None, "gap", "heartbeat_gap", "hash", "missing", "backwards", "nan", "claim", "exit"],
)
def test_cadence_healthy_and_fault_twins(fault: str | None) -> None:
    line = "CT_HEARTBEAT " + json.dumps(_row()) + "\n"
    times = [0.1, 15.1, 30.1, 45.1]
    record = {
        "clock": "parent-monotonic-receive-including-start-and-exit",
        "limit_seconds": 60,
        "heartbeat_limit_seconds": 30,
        "exit_seconds": 46,
        "line_count": 4,
        "max_gap_seconds": 15.0,
        "heartbeat_max_gap_seconds": 15.0,
        "observations": [
            {"received_seconds": t, "line_sha256": hashlib.sha256(line.encode()).hexdigest()}
            for t in times
        ],
    }
    if fault == "gap":
        record["exit_seconds"] = 106
        record["max_gap_seconds"] = record["heartbeat_max_gap_seconds"] = 60.9
    elif fault == "heartbeat_gap":
        record["exit_seconds"] = 76
        record["max_gap_seconds"] = record["heartbeat_max_gap_seconds"] = 30.9
    elif fault == "hash":
        record["observations"][1]["line_sha256"] = "0" * 64
    elif fault == "missing":
        record["observations"].pop()
    elif fault == "backwards":
        record["observations"][2]["received_seconds"] = 1
    elif fault == "nan":
        record["observations"][2]["received_seconds"] = float("nan")
    elif fault == "claim":
        record["max_gap_seconds"] = 1
    elif fault == "exit":
        record["exit_seconds"] = 40
    if fault is None:
        ct_progress.verify_cadence(line * 4, record)
    else:
        with pytest.raises(ValueError):
            ct_progress.verify_cadence(line * 4, record)
