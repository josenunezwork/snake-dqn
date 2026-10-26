"""Data-free hard-cutoff tests: pytest parent plus exactly one owned worker."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from pathlib import Path

import pytest

from research.bl_ppo.exports import export_json, verify_export
from research.bl_residual import f4_postmortem as entry
from research.bl_residual import f4_postmortem_watchdog as watchdog


@pytest.mark.parametrize("phase", ["compute", "finalization"])
def test_native_stall_killed_independently_with_spent_claim_and_durable_record(
    tmp_path: Path, phase: str
) -> None:
    output = tmp_path / "attempt"
    receipt = watchdog.kill_receipt_path(output)
    deferred = tmp_path / "signal-handler-ran"
    complete = tmp_path / "finished"

    def worker(arm):
        arm()
        entry.claim_bundle(
            output,
            {"implementation_commit": "a" * 40, "manifest_sha256": "b" * 64},
            {"id": "synthetic"},
        )

        def alarm(*_args):
            deferred.write_text("unexpected Python callback\n")

        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, 0.02)

        def native_stall():
            (tmp_path / "phase").write_text(phase)
            # This native call defers Python SIGALRM dispatch. No data or model
            # is involved, and only SIGKILL from the parent can bound this call.
            hashlib.pbkdf2_hmac("sha256", b"synthetic", b"fixture", 2_000_000_000)
            complete.write_text("unexpected completion\n")

        if phase == "finalization":
            # A sealed diagnostic must not survive a subsequent finalization
            # stall as a successful terminal result. Its bytes are never parsed.
            export_json(output / "diagnostic.json", {"synthetic_opaque": True})
            try:
                raise RuntimeError("synthetic soft deadline")
            finally:
                native_stall()
        else:
            native_stall()

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="watchdog killed.*no retry"):
        watchdog.supervise(
            worker,
            kill_receipt=receipt,
            seconds=0.4,
            on_kill=lambda record: entry._record_watchdog_failure(
                "a" * 40, {"metadata": {"parents": {"synthetic": True}}}, output, record
            ),
        )
    elapsed = time.monotonic() - started
    assert elapsed < 3.0
    assert (tmp_path / "phase").read_text() == phase
    assert not complete.exists() and not deferred.exists()
    assert verify_export(output / "claim.json")
    assert verify_export(receipt)
    record = json.loads(receipt.read_bytes())
    assert record["reason"] == "hard_ceiling"
    assert record["signal"] == "SIGKILL"
    assert os.WIFSIGNALED(record["wait_status"])
    assert os.WTERMSIG(record["wait_status"]) == signal.SIGKILL
    assert 0.4 <= record["elapsed_seconds"] < 3.0
    assert record["retry_authorized"] is False
    assert verify_export(output / "failure.json")
    failure = json.loads((output / "failure.json").read_bytes())
    assert failure["status"] == "INCOMPLETE"
    assert failure["failure_class"] == "IndependentWatchdogKill"
    assert failure["claim_sha256"] == verify_export(output / "claim.json")
    assert verify_export(output / "development-parents.json")
    with pytest.raises(FileExistsError):
        entry.claim_bundle(output, {}, {})
    with pytest.raises(FileExistsError, match="no retry"):
        watchdog.supervise(lambda arm: pytest.fail("retry"), kill_receipt=receipt, seconds=0.4)


def test_supervisor_allows_one_normal_completion_and_no_kill_receipt(tmp_path: Path) -> None:
    marker = tmp_path / "done"
    receipt = tmp_path / "kill.json"

    def worker(arm):
        arm()
        marker.write_text("done\n")

    watchdog.supervise(worker, kill_receipt=receipt, seconds=1.0)
    assert marker.read_text() == "done\n"
    assert not receipt.exists()


def test_ceiling_starts_at_ack_after_lease_wait_before_claim(tmp_path: Path) -> None:
    marker = tmp_path / "done"

    def worker(arm):
        time.sleep(0.2)  # synthetic pre-claim lease wait is outside the ceiling
        arm()
        marker.write_text("claimed only after ACK\n")

    watchdog.supervise(worker, kill_receipt=tmp_path / "kill.json", seconds=0.1)
    assert marker.exists()


@pytest.mark.parametrize("mode", ["error", "unarmed"])
def test_abnormal_exit_never_retries(tmp_path: Path, mode: str) -> None:
    def worker(arm):
        if mode == "error":
            arm()
            raise ValueError("synthetic private error")

    with pytest.raises(RuntimeError, match="incomplete|handshake"):
        watchdog.supervise(worker, kill_receipt=tmp_path / "kill.json", seconds=1.0)


def test_supervisor_interrupt_kills_and_records_only_owned_child(tmp_path, monkeypatch) -> None:
    receipt = tmp_path / "kill.json"

    def worker(arm):
        arm()
        time.sleep(10)

    def interrupted(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(watchdog.select, "select", interrupted)
    with pytest.raises(KeyboardInterrupt):
        watchdog.supervise(worker, kill_receipt=receipt, seconds=1.0)
    assert verify_export(receipt)
    assert json.loads(receipt.read_bytes())["reason"] == "supervisor_interrupted"


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_supervisor_terminal_signal_kills_records_and_restores_handler(
    tmp_path, monkeypatch, signum
) -> None:
    receipt = tmp_path / "kill.json"
    prior = signal.getsignal(signum)

    def worker(arm):
        arm()
        time.sleep(10)

    def interrupt(*_args):
        os.kill(os.getpid(), signum)

    monkeypatch.setattr(watchdog.select, "select", interrupt)
    with pytest.raises(RuntimeError, match="supervisor interrupted"):
        watchdog.supervise(worker, kill_receipt=receipt, seconds=1.0)
    assert signal.getsignal(signum) is prior
    assert verify_export(receipt)
    record = json.loads(receipt.read_bytes())
    assert record["reason"] == "supervisor_interrupted"
    assert os.WTERMSIG(record["wait_status"]) == signal.SIGKILL


def test_repeated_terminal_signals_cannot_interrupt_durable_kill_record(
    tmp_path, monkeypatch
) -> None:
    receipt = tmp_path / "kill.json"
    real_export = watchdog.export_json

    def worker(arm):
        arm()
        time.sleep(10)

    def interrupted_export(path, value):
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGHUP)
        real_export(path, value)

    monkeypatch.setattr(watchdog, "export_json", interrupted_export)
    with pytest.raises(RuntimeError, match="watchdog killed"):
        watchdog.supervise(worker, kill_receipt=receipt, seconds=0.1)
    assert verify_export(receipt)
    assert json.loads(receipt.read_bytes())["reason"] == "hard_ceiling"
