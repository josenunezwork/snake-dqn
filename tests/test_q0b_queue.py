"""Queue prelude tests without CUDA, Drive writes, or a learner launch."""

import json
import tarfile
from types import SimpleNamespace

import pytest

from research.q0q1_20261017 import prepare_q0b_inputs, q0b_queue
from research.q0q1_20261017.run_queue_job import sha256_file


def test_observed_pricing_refuses_unknown_or_zero_rates():
    value = {
        "schema": "q0q1-observed-pricing/v1",
        "observed_at": "2026-10-18",
        "evidence": "observed dashboard",
        "rates": {"L4": 2.0, "A100": 5.3, "cpu": 0.5},
    }
    assert q0b_queue.observed_pricing(value, "L4") == (2.0, 0.5)
    value["rates"]["cpu"] = 0
    with pytest.raises(RuntimeError, match="positive"):
        q0b_queue.observed_pricing(value, "A100")
    value["rates"].pop("cpu")
    with pytest.raises(RuntimeError, match="missing"):
        q0b_queue.observed_pricing(value, "L4")


def test_bound_input_refuses_changed_bytes_and_escape(tmp_path):
    target = tmp_path / "pricing.json"
    target.write_text('{"schema":"test"}')
    target.with_name(target.name + ".sha256").write_text(sha256_file(target))
    assert q0b_queue.bound_input(tmp_path, "pricing.json")[1] == {"schema": "test"}
    target.write_text("{}")
    with pytest.raises(RuntimeError, match="changed receipt"):
        q0b_queue.bound_input(tmp_path, "pricing.json")
    with pytest.raises(RuntimeError, match="escapes"):
        q0b_queue.bound_input(tmp_path, "../outside.json")


def test_prepare_inputs_preserves_observed_values_and_binds_mac_reference(tmp_path):
    reference = tmp_path / "mac.json"
    reference.write_text(json.dumps({"schema": "q1-parity/v1", "runtime": {"system": "Darwin"}}))
    pricing = tmp_path / "observed-pricing.json"
    argv = [
        "--l4-units-per-hour",
        "2",
        "--a100-units-per-hour",
        "5.3",
        "--cpu-units-per-hour",
        "0.5",
        "--evidence",
        "actual debit reading",
        "--mac-reference",
        str(reference),
        "--out",
        str(pricing),
    ]
    assert prepare_q0b_inputs.main(argv) == 0
    assert q0b_queue.bound_input(tmp_path, pricing.name)[1]["rates"] == {
        "L4": 2,
        "A100": 5.3,
        "cpu": 0.5,
    }
    assert q0b_queue.bound_input(tmp_path, reference.name)[1]["runtime"]["system"] == "Darwin"
    with pytest.raises(SystemExit):
        prepare_q0b_inputs.main(argv)


def test_bounded_prelude_rejects_an_exhausted_packet_before_launch():
    with pytest.raises(RuntimeError, match="exhausted"):
        q0b_queue.run_bounded(["must-not-run"], seconds=0, env={})


def test_slow_export_cannot_leave_a_qualified_packet(monkeypatch, tmp_path):
    from research.q0q1_20261017 import run_queue_job as runner

    output = tmp_path / "output"
    output.mkdir()

    def fake_child(command, **kwargs):
        local = Path(command[command.index("--out") + 1])
        (local / "receipt.json").write_text(json.dumps({"qualification": {"status": "QUALIFIED"}}))
        return SimpleNamespace(returncode=0)

    from pathlib import Path

    monkeypatch.setattr(runner.subprocess, "run", fake_child)
    monkeypatch.setattr(runner, "_scientific_done", lambda *args: None)
    monkeypatch.setattr(q0b_queue, "attempt_age", lambda env, now: 3601.0)
    env = {
        "Q0Q1_REGISTRATION_STATUS": "APPROVED_Q0Q1_20261017",
        "Q0Q1_OWNER_RELEASE": "APPROVED_Q0Q1_20261017",
        "Q0Q1_JOB_KIND": "q0b_measure",
        "Q0Q1_HARDWARE": "L4",
        "Q0Q1_INPUT_ROOT": "/inputs",
        "QUEUE_OUTPUT_DIR": str(output),
    }
    with pytest.raises(RuntimeError, match="export exceeded"):
        runner.run(env)
    assert (output / "over-budget-result.tar.gz").is_file()
    archive = output / "result.tar.gz"
    assert archive.with_name(archive.name + ".sha256").read_text().split()[0] == sha256_file(
        archive
    )
    with tarfile.open(archive) as tar:
        value = json.load(tar.extractfile("attempt/receipt.json"))
        attempt = json.load(tar.extractfile("attempt/queue-attempt-receipt.json"))
    assert value["qualification"]["status"] == attempt["status"] == "INCOMPLETE"
    assert value["packet_seconds"] == 3601
