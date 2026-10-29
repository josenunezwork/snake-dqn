"""Outcome-blind admission, process-tree and detached allocation regressions."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.ops import dispatch, processes
from tools.sandbox import admission
from tools.sandbox.supervisor import sandbox_admission_route


def test_slow_child_and_grandchild_are_killed_and_reaped(tmp_path: Path) -> None:
    child = tmp_path / "child.py"
    ids = tmp_path / "ids.json"
    child.write_text(
        "import os, signal, subprocess, sys, time, json\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        "def stop(*a):\n p.terminate(); p.wait(); raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "open(sys.argv[1],'w').write(json.dumps([os.getpid(),p.pid]))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        processes.run([sys.executable, str(child), str(ids)], timeout=1)
    for pid in json.loads(ids.read_text()):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_outer_timeout_cleans_nested_owned_group(tmp_path: Path) -> None:
    ids = tmp_path / "pid"
    code = (
        "import sys; from tools.ops.processes import run; "
        "run([sys.executable,'-c',"
        + repr(f"import os,time; open({str(ids)!r},'w').write(str(os.getpid())); time.sleep(60)")
        + "],timeout=60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        processes.run([sys.executable, "-c", code], timeout=1)
    with pytest.raises(ProcessLookupError):
        os.kill(int(ids.read_text()), 0)


def test_command_has_independent_validation_and_allocation_windows(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(dispatch.time, "monotonic", lambda: clock[0])
    timeouts = []

    def run(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        clock[0] += 262

    monkeypatch.setattr(dispatch, "run_process_tree", run)
    dispatch.command(["validation"], timeout_seconds=admission.VALIDATION_SECONDS)
    deadline = clock[0] + 600
    dispatch.command(["allocation"], deadline=deadline)
    assert timeouts == [1200, 60]
    assert deadline == 962


def test_missing_or_changed_test_receipt_fails_closed(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tests.json"
    monkeypatch.setenv(admission.TEST_RECEIPT_ENV, str(path))
    monkeypatch.setenv(admission.REQUIRE_RECEIPT_ENV, "1")
    with pytest.raises(ValueError, match="required"):
        admission.cached_tests("a")
    admission.publish_tests("a")
    assert admission.cached_tests("a")
    with pytest.raises(ValueError, match="differs"):
        admission.cached_tests("b")


def test_identity_binds_code_inputs_and_environment(tmp_path, monkeypatch) -> None:
    def git(*args):
        processes.run(["git", *args], cwd=tmp_path, timeout=10, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    code = tmp_path / "policy.py"
    code.write_text("pass\n")
    manifest = tmp_path / "sandbox.json"
    manifest.write_text("{}")
    git("add", ".")
    git("commit", "-qm", "fixture")
    external = tmp_path.parent / "input"
    external.write_bytes(b"original")
    original = {"inputs": [{"path": str(external)}]}
    first = admission.test_identity(tmp_path, original, manifest)
    external.write_bytes(b"changed")
    assert admission.test_identity(tmp_path, original, manifest) != first
    external.write_bytes(b"original")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert admission.test_identity(tmp_path, original, manifest) != first
    code.write_text("raise ValueError\n")
    with pytest.raises(ValueError, match="clean"):
        admission.test_identity(tmp_path, original, manifest)


def test_child_owns_serialization_after_parent_releases(tmp_path) -> None:
    lock = tmp_path / "alloc.lock"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    lease_path = tmp_path / "lease.json"
    with admission.parent_lease(lock, lease_path, manifest, 600):
        child = admission.AllocationLease(lease_path, manifest)
        child.acquire()
    # Parent revocation prevents another late detached runner from acquiring.
    with pytest.raises(FileNotFoundError):
        admission.AllocationLease(lease_path, manifest).acquire()
    with lock.open("a") as competitor:
        with pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child.close()
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("failure", ["expired", "changed-manifest", "parent-dead", "revoked"])
def test_native_run_cannot_allocate_late(tmp_path, monkeypatch, failure) -> None:
    from research.colab_cli import supervisor

    lock = tmp_path / "alloc.lock"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    lease_path = tmp_path / "lease.json"
    calls = []

    def native_run(self, value):
        supervisor.CLI._call_once(self, "new", timeout=60)

    monkeypatch.setattr(supervisor.Broker, "run", native_run)
    monkeypatch.setattr(supervisor.CLI, "_call_once", lambda *a, **k: calls.append("new"))
    monkeypatch.setenv(admission.LEASE_ENV, str(lease_path))
    monkeypatch.setattr(sys, "argv", ["supervisor", "run", str(manifest)])
    with admission.parent_lease(lock, lease_path, manifest, 600):
        value = json.loads(lease_path.read_text())
        if failure == "expired":
            value["deadline"] = time.monotonic() - 1
        elif failure == "changed-manifest":
            manifest.write_text('{"foreign":true}')
        elif failure == "parent-dead":
            value["parent_pid"] = 99999999
        if failure == "revoked":
            lease_path.unlink()
        else:
            lease_path.write_text(json.dumps(value))
        with (
            sandbox_admission_route(),
            pytest.raises((ValueError, ProcessLookupError, FileNotFoundError)),
        ):
            supervisor.Broker.run(object(), {"sandbox_admission": {}})
    assert calls == []


def test_live_dispatch_262_second_admission_and_native_run(tmp_path, monkeypatch) -> None:
    """Exercise live dispatch -> refreshed validation -> native new -> calibration.

    Only provider/host inventory and test execution are fixtures. Native adapter
    and both real flock owners enforce serialization; virtual time models 262s.
    """
    from research.colab_cli import supervisor

    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    ops = tmp_path / "ops"
    ops.mkdir()
    monkeypatch.setenv("OPS_DIR", str(ops))
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "output"
    packet = {
        "job_id": "c2",
        "commit": "a" * 40,
        "output_dir": str(output),
        "sandbox_admission": {},
        "budget": {"other_assignments": {}},
        "resume": {
            "unit_kind": "snapshot",
            "ledger_path": "inputs/ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
    }
    manifest.write_text(json.dumps(packet))
    events = []
    monkeypatch.setattr(dispatch, "qualified", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "require_scientific_or_f5_plan_admission", lambda *a: None)
    monkeypatch.setattr(dispatch, "ensure_watcher", lambda *a: None)

    def validation(broker, cwd, state, path, log, deadline=None, *, test_receipt=None):
        if deadline is None:
            with (ops / "alloc.lock").open("a") as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            clock[0] += 262
            test_receipt.write_text("completed suites")
            events.append("full-validation")
        else:
            assert test_receipt.read_text() == "completed suites"
            assert deadline == 962
            clock[0] += 33
            events.append("live-checks")

    def refresh(path, evidence, deadline):
        dispatch.write_create_only_json(evidence, {"manifest_sha256": admission.digest(path)})
        return path

    def native_run(self, value):
        # Model the native validator's two checks-only passes, then allocation.
        clock[0] += 66
        supervisor.CLI._call_once(self, "new", timeout=300)
        supervisor.Broker.calibrate_rate(self)

    def new(self, *args, timeout):
        with (ops / "alloc.lock").open("a") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert clock[0] < 962
        events.append("new-under-lock")
        clock[0] += 30

    def calibrate(self):
        clock[0] += 60
        output.mkdir()
        (output / "rate-calibration.json").write_text('{"status":"PASS"}')
        events.append("calibrated")

    def start(broker, cwd, state, path, log, name, deadline, **kwargs):
        log.touch()
        monkeypatch.setenv(admission.LEASE_ENV, str(kwargs["sandbox_lease"]))
        monkeypatch.setattr(sys, "argv", ["supervisor", "run", str(path)])
        fake_cli = argparse.Namespace(cleanup_mode=False)
        with sandbox_admission_route():
            supervisor.Broker.run(fake_cli, packet)

    monkeypatch.setattr(dispatch, "validate", validation)
    monkeypatch.setattr(dispatch, "refresh_assignments", refresh)
    monkeypatch.setattr(dispatch, "start", start)
    monkeypatch.setattr(supervisor.Broker, "run", native_run)
    monkeypatch.setattr(supervisor.CLI, "_call_once", new)
    monkeypatch.setattr(supervisor.Broker, "calibrate_rate", calibrate)
    args = argparse.Namespace(
        name="c2",
        manifest=manifest,
        broker=tmp_path,
        cwd=tmp_path,
        state_dir=None,
        dispatch_txt=None,
        dry_run=False,
    )
    assert "allocation=success" in dispatch.dispatch(args)
    assert events == ["full-validation", "live-checks", "new-under-lock", "calibrated"]
    assert not list((ops / "dispatch/c2").glob("*.allocation-lease.json"))
