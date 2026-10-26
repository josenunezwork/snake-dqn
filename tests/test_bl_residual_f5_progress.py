"""Offline operational telemetry contracts for F5 qualification."""

from __future__ import annotations

import json
import random
import threading
import time

import numpy as np
import pytest
import torch

from research.bl_residual.f5_progress import QualificationProgress


def _records(captured) -> list[dict]:
    return [json.loads(line) for line in captured.readouterr().out.splitlines() if line]


def test_heartbeat_reports_a_blocked_phase_without_outcome_content(capsys) -> None:
    progress = QualificationProgress(interval_seconds=0.01)
    progress.start()
    progress.phase("source-h5000", completed=3)
    time.sleep(0.03)
    progress.close(success=False)
    records = _records(capsys)
    assert any(record["status"] == "HEARTBEAT" for record in records)
    heartbeat = next(record for record in records if record["status"] == "HEARTBEAT")
    assert heartbeat["phase"] == "source-h5000"
    assert heartbeat["fixtures_completed"] == 3
    assert heartbeat["fixtures_total"] == 12
    rendered = json.dumps(records).lower()
    assert "action" not in rendered
    assert "loss" not in rendered
    assert "q_" not in rendered


def test_stop_prevents_later_heartbeat_and_failure_is_redacted(capsys) -> None:
    progress = QualificationProgress(interval_seconds=0.01)
    progress.start()
    progress.phase("admission")
    progress.failure(RuntimeError("secret checkpoint payload and Q values"))
    progress.close(success=False)
    assert progress._thread is not None and not progress._thread.is_alive()
    time.sleep(0.03)
    records = _records(capsys)
    assert records[-1]["status"] == "STOPPED"
    failure = next(record for record in records if record["status"] == "FAILED")
    assert failure["failure_class"] == "RuntimeError"
    assert "secret" not in json.dumps(records)


def test_progress_thread_preserves_python_numpy_and_torch_rng_state() -> None:
    random.seed(104)
    np.random.seed(105)
    torch.manual_seed(106)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    progress = QualificationProgress(interval_seconds=0.01)
    progress.start()
    progress.phase("cuda-parity")
    time.sleep(0.02)
    progress.close(success=True)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_concurrent_phase_and_failure_output_remains_parseable_jsonl(capsys) -> None:
    progress = QualificationProgress(interval_seconds=0.001)
    progress.start()
    worker = threading.Thread(target=lambda: progress.failure(RuntimeError("redacted")))
    worker.start()
    for _ in range(10):
        progress.phase("source-root")
    worker.join()
    progress.close(success=False)
    records = _records(capsys)
    assert any(record["status"] == "FAILED" for record in records)
    assert all(record["schema"] == "bl-f5-qualification-progress/v1" for record in records)
    failure_index = next(
        index for index, record in enumerate(records) if record["status"] == "FAILED"
    )
    assert all(record["status"] != "HEARTBEAT" for record in records[failure_index + 1 :])


def test_heartbeat_and_fixture_completion_publish_monotonic_counters(capsys) -> None:
    progress = QualificationProgress(interval_seconds=0.001)
    progress.start()
    try:
        for completed in range(13):
            progress.phase("fixture-complete", completed=completed)
            time.sleep(0.002)
    finally:
        progress.close(success=True)
    records = _records(capsys)
    assert any(record["status"] == "HEARTBEAT" for record in records)
    counts = [record["fixtures_completed"] for record in records]
    assert counts == sorted(counts)
    assert counts[-1] == 12
    elapsed = [record["elapsed_seconds"] for record in records]
    assert elapsed == sorted(elapsed)


def test_cli_refusal_emits_terminal_stop_and_closes_heartbeat(
    monkeypatch, capsys, tmp_path
) -> None:
    from research.bl_residual import f5_qualification

    def refuse(*args, **kwargs):
        raise f5_qualification.QualificationRefusal("deliberately redacted")

    monkeypatch.setattr(f5_qualification, "run", refuse)
    monkeypatch.setattr(
        "sys.argv",
        [
            "f5_qualification.py",
            "--inputs",
            str(tmp_path / "inputs"),
            "--fixtures",
            str(tmp_path / "fixtures"),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(f5_qualification.QualificationRefusal):
        f5_qualification.main()
    records = _records(capsys)
    assert records[-1]["status"] == "STOPPED"
    assert records[-2]["failure_class"] == "QualificationRefusal"
