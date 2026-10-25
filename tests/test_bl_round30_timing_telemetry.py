"""Hermetic failure, timeout and hard-kill evidence; no admitted scientific inputs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time

import pytest

from research.bl_ppo.exports import export_json
from research.bl_round30 import timing, timing_forecast
from research.bl_round30.timing_telemetry import (
    MAX_BYTES,
    Interrupted,
    Telemetry,
    WorkerDeadline,
    validate_event,
    validate_inventory,
)


@pytest.mark.parametrize(
    "phase,error,code,kind",
    [
        ("plan_coefficients", RuntimeError("SECRET 12345.6789"), "OTHER_EXCEPTION", "RuntimeError"),
        ("certificate", TimeoutError("SECRET 12345.6789"), "CERTIFICATE_DEADLINE", "TimeoutError"),
        ("certificate", WorkerDeadline("SECRET 12345.6789"), "WORKER_DEADLINE", "TimeoutError"),
        ("adam_0_10", Interrupted("SECRET 12345.6789"), "INTERRUPTED", "BaseException"),
        (
            "closure_0",
            type("SECRET12345", (Exception,), {})("SECRET"),
            "OTHER_EXCEPTION",
            "OtherException",
        ),
    ],
)
def test_failure_is_durable_redacted_and_never_a_completion(
    tmp_path, monkeypatch, capfd, phase, error, code, kind
):
    def faulty(input_dir, output, *, config_sha256, synthetic, timer):
        timer.telemetry.end()  # preflight
        timer.call("feature_cache", lambda: None)

        def fail():
            print("SECRET 12345.6789")
            os.write(2, b"SECRET_NATIVE 12345.6789\n")
            raise error

        timer.call(phase, fail)

    monkeypatch.setattr(timing, "_run", faulty)
    output = tmp_path / "output"
    with pytest.raises(type(error)), timing.silence():
        timing.run(tmp_path, output, config_sha256="a" * 64, synthetic=True)
    assert capfd.readouterr() == ("", "")
    records = validate_inventory(output)
    record = records[-1]
    assert record["event"] == "failure" and record["status"] == "failed"
    assert record["phase"] == phase and record["diagnostic_code"] == code
    assert record["exception_type"] == kind
    assert "feature_cache" in record["completed_phase_seconds"]
    assert phase not in record["completed_phase_seconds"]
    assert record["phase_elapsed_seconds"] >= 0
    assert "SECRET" not in json.dumps(records) and "12345.6789" not in json.dumps(records)
    assert not (output / "timing.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        validate_inventory(output, complete=True)
    with pytest.raises(ValueError):
        timing_forecast.forecast(record, {})
    from research.bl_round30.resume import build

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    export_json(inputs / "control.json", {"namespace": "synthetic", "commit": "a" * 40})
    with pytest.raises(ValueError):
        build(inputs, [output], tmp_path / "ledger.json")
    assert not (tmp_path / "ledger.json").exists()


def test_actual_alarm_has_worker_category_even_in_certificate(tmp_path, monkeypatch):
    def faulty(input_dir, output, *, config_sha256, synthetic, timer):
        timer.telemetry.end()
        with timing.deadline(0.02):
            timer.call("certificate", lambda: time.sleep(1))

    monkeypatch.setattr(timing, "_run", faulty)
    with pytest.raises(WorkerDeadline):
        timing.run(tmp_path, tmp_path / "output", config_sha256="a" * 64, synthetic=True)
    records = validate_inventory(tmp_path / "output")
    assert records[-1]["diagnostic_code"] == "WORKER_DEADLINE"
    assert "certificate" not in records[-1]["completed_phase_seconds"]


@pytest.mark.parametrize("stage", [".json", ".json.sha256", ".json.complete"])
def test_deadline_during_boundary_publication_does_not_erase_cause(tmp_path, monkeypatch, stage):
    from research.bl_ppo import exports

    publish = exports._publish
    sent = False

    def interrupted_publish(path, payload):
        nonlocal sent
        publish(path, payload)
        # Interrupt each component of the feature-cache END transaction.
        if path.name == "000005" + stage and not sent:
            sent = True
            os.kill(os.getpid(), signal.SIGALRM)

    def faulty(input_dir, output, *, config_sha256, synthetic, timer):
        timer.telemetry.end()
        with timing.deadline(5):
            timer.call("feature_cache", lambda: None)

    monkeypatch.setattr(exports, "_publish", interrupted_publish)
    monkeypatch.setattr(timing, "_run", faulty)
    with pytest.raises(WorkerDeadline):
        timing.run(tmp_path, tmp_path / "output", config_sha256="a" * 64, synthetic=True)
    assert sent
    records = validate_inventory(tmp_path / "output")
    assert records[-1]["diagnostic_code"] == "WORKER_DEADLINE"
    assert records[-1]["phase"] == "worker"
    assert "feature_cache" in records[-1]["completed_phase_seconds"]


@pytest.mark.parametrize("termination", [signal.SIGTERM, signal.SIGKILL])
def test_native_hang_retains_exportable_start_through_termination(tmp_path, termination):
    """Use real immutable scanner without constructing any Colab client."""
    from research.colab_cli.runtime import _exports

    output = tmp_path / "output"
    script = """
import sys, time
from pathlib import Path
from research.bl_round30 import timing
def hang(input_dir, output, *, config_sha256, synthetic, timer):
    timer.telemetry.end()
    timer.call('feature_cache', lambda: None)
    timer.call('plan_coefficients', lambda: time.sleep(120))
timing._run = hang
with timing.silence():
    try:
        timing.run(
        Path(sys.argv[1]), Path(sys.argv[1]) / 'output',
        config_sha256='a'*64, synthetic=True,
    )
    except BaseException:
        sys.exit(1)
"""
    child = subprocess.Popen(
        [str(timing.ROOT / "venv/bin/python"), "-c", script, str(tmp_path)],
        cwd=timing.ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if (output / "telemetry/000006.json.complete").exists():
                break
            if child.poll() is not None:
                pytest.fail("synthetic worker exited before phase start")
            time.sleep(0.01)
        else:
            pytest.fail("synthetic worker did not publish phase start")
        # Independently verify before and after death, as qualified polling does.
        before = _exports(tmp_path, MAX_BYTES)
        child.send_signal(termination)
        stdout, stderr = child.communicate(timeout=5)
        assert stdout == stderr == b""
        after = _exports(tmp_path, MAX_BYTES)
        assert all(entry in after for entry in before)
        records = validate_inventory(output)
        assert records[-1]["phase"] == "plan_coefficients"
        assert "plan_coefficients" not in records[-1]["completed_phase_seconds"]
        if termination == signal.SIGTERM:
            assert records[-1]["diagnostic_code"] == "INTERRUPTED"
        else:
            assert records[-1]["event"] == "start"
        assert not (output / "timing.json").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_telemetry_rejects_arbitrary_keys_and_nested_science(tmp_path):
    telemetry = Telemetry(tmp_path)
    telemetry.start("feature_cache")
    event = validate_inventory(tmp_path)[0]
    for key, bad in [
        ("loss", 12345.6789),
        ("exception_type", "SECRET12345"),
        ("diagnostic_code", "SECRET"),
    ]:
        value = {**event, key: bad}
        with pytest.raises(ValueError):
            validate_event(value)
    event["completed_phase_seconds"]["loss"] = 12345.6789
    with pytest.raises(ValueError):
        validate_event(event)


def test_no_telemetry_publication_can_exceed_total_output_allowance(tmp_path):
    telemetry = Telemetry(tmp_path)
    (tmp_path / "padding").write_bytes(bytes(MAX_BYTES))
    with pytest.raises(ValueError, match="allowance"):
        telemetry.start("feature_cache")
    assert list((tmp_path / "telemetry").iterdir()) == []


@pytest.mark.parametrize("stage", [".json", ".json.sha256", ".json.complete"])
def test_transient_end_export_failure_keeps_original_type_and_abandons_slot(
    tmp_path, monkeypatch, stage
):
    from research.bl_ppo import exports

    publish = exports._publish
    failed = False

    def fail_once(path, payload):
        nonlocal failed
        if path.name == "000005" + stage and not failed:
            failed = True
            raise OSError("SECRET 12345.6789")
        publish(path, payload)

    def faulty(input_dir, output, *, config_sha256, synthetic, timer):
        timer.telemetry.end()
        timer.call("feature_cache", lambda: None)

    monkeypatch.setattr(exports, "_publish", fail_once)
    monkeypatch.setattr(timing, "_run", faulty)
    output = tmp_path / "output"
    with pytest.raises(OSError):
        timing.run(tmp_path, output, config_sha256="a" * 64, synthetic=True)
    records = validate_inventory(output)
    assert records[-1]["exception_type"] == "OSError"
    assert records[-1]["diagnostic_code"] == "OTHER_EXCEPTION"
    assert records[-1]["phase"] == "feature_cache"
    assert "feature_cache" not in records[-1]["completed_phase_seconds"]
    assert "SECRET" not in json.dumps(records)
    with pytest.raises(ValueError):
        validate_inventory(output, complete=True)
