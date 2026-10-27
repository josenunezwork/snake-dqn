"""Future-only sandbox child CPU accounting primitives."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sandbox import supervision_accounting as subject


def _journal(tmp_path: Path) -> subject.ImmutableJournal:
    return subject.ImmutableJournal(tmp_path / "journal")


_THREAD_CAP_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _command(code: str) -> list[str]:
    return ["nice", "-n", "15", sys.executable, "-c", code]


def _launch(journal: subject.ImmutableJournal, operation_id: str, code: str):
    return subject.launch_operation(
        journal,
        operation_id,
        _command(code),
        payload=subject.operation_payload("status", b""),
        parent_identity="900@100",
        env=_THREAD_CAP_ENV,
    )


def test_payload_binding_is_digest_only_and_has_exact_64_kib_cap() -> None:
    payload = subject.operation_payload("read_state", b"a" * 65535, b"b")
    assert payload["application_bytes"] == 65536
    assert "a" * 10 not in json.dumps(payload)
    with pytest.raises(subject.AccountingError, match="65536"):
        subject.operation_payload("read_state", b"a" * 65536, b"b")


def test_wait4_records_a_child_that_exits_between_external_samples(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    child, tracked = subject.launch_operation(
        journal,
        "op-read-01",
        _command(
            "import os; "
            "assert os.environ['OMP_NUM_THREADS'] == '1'; "
            "assert os.environ['OPENBLAS_NUM_THREADS'] == '1'; "
            "assert os.environ['MKL_NUM_THREADS'] == '1'; "
            "assert os.environ['NUMBA_NUM_THREADS'] == '1'; "
            "assert os.environ['VECLIB_MAXIMUM_THREADS'] == '1'; "
            "sum(i * i for i in range(900000))"
        ),
        payload=subject.operation_payload("read_state", b"request"),
        parent_identity="900@100",
        env=_THREAD_CAP_ENV,
    )
    result = subject.wait4_exit(journal, child, tracked)
    assert result["returncode"] == 0
    assert result["final_cpu"] > 0
    assert result["accounting_kind"] == "wait4-direct-child-self"
    assert result["final_cpu_scope"] == "descendant_inclusive"
    events = sorted((tmp_path / "journal/events").glob("*.json"))
    launch, exit_event = [json.loads(path.read_text()) for path in events]
    assert launch["kind"] == "launch"
    assert exit_event["kind"] == "exit"
    assert exit_event["previous_sha256"] == launch["sha256"]
    assert exit_event["payload"]["pid"] == tracked.pid


def test_wait4_rejects_already_reaped_child_without_aggregate_fallback(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    child, tracked = _launch(journal, "op-read-02", "pass")
    os.waitpid(child.pid, 0)
    with pytest.raises(subject.AccountingError, match="already reaped"):
        subject.wait4_exit(journal, child, tracked)


def test_poll_path_keeps_wait4_rusage_and_operation_lifecycle(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    payload = subject.operation_payload("sessions", b"request", b"reply")
    child, tracked = subject.launch_operation(
        journal,
        "op-read-03",
        _command("pass"),
        payload=payload,
        parent_identity="900@100",
        env=_THREAD_CAP_ENV,
    )
    subject.record_exec(journal, tracked)
    subject.record_operation_start(journal, tracked, payload)
    while child.poll() is None:
        pass
    subject.record_operation_end(journal, tracked.operation_id)
    result = subject.wait4_exit(journal, child, tracked)
    assert result["final_cpu"] >= 0
    kinds = [
        json.loads(path.read_text())["kind"]
        for path in sorted((tmp_path / "journal/events").glob("*.json"))
    ]
    assert kinds == ["launch", "exec", "operation_start", "operation_end", "exit"]


def test_create_only_journal_chain_and_counter_sample(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    one = journal.append("root_identity", {"pid": 10}, monotonic_ns=100)
    sample = subject.record_counter_sample(
        journal, process_id=10, counter_seconds=0.25, sample_kind="independent-ps", monotonic_ns=200
    )
    sealed = journal.seal()
    assert sample["counter_seconds"] == 0.25
    assert sealed["terminal_event_sha256"] is not None
    with pytest.raises(FileExistsError):
        subject.ImmutableJournal(tmp_path / "journal")
    assert one["sha256"] != sealed["sha256"]
    assert subject.CAPABILITY == "NOT_ADMISSIBLE"
    assert (
        tmp_path / "journal/events/00000001-root_identity.json.complete"
    ).read_text() == "complete\n"


def test_journal_serializes_concurrent_events_and_refuses_postseal_or_time_regression(
    tmp_path: Path,
) -> None:
    journal = _journal(tmp_path)
    failures: list[BaseException] = []

    def write(index: int) -> None:
        try:
            journal.append("counter_sample", {"index": index})
        except BaseException as exc:  # pragma: no cover - assertion below
            failures.append(exc)

    workers = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not failures
    events = [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "journal/events").glob("*.json"))
    ]
    assert [event["sequence"] for event in events] == list(range(1, 9))
    assert all(a["monotonic_ns"] < b["monotonic_ns"] for a, b in zip(events, events[1:]))
    journal.seal()
    with pytest.raises(subject.AccountingError, match="after journal seal"):
        journal.append("late", {})

    failed = _journal(tmp_path / "failed")
    failed.append("one", {}, monotonic_ns=10)
    with pytest.raises(subject.AccountingError, match="strictly increase"):
        failed.append("two", {}, monotonic_ns=10)
    with pytest.raises(subject.AccountingError, match="after journal failure"):
        failed.append("three", {}, monotonic_ns=11)


def test_post_spawn_journal_failure_leaves_cleanup_to_future_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    journal = _journal(tmp_path)

    def fail(*args, **kwargs):
        raise subject.AccountingError("disk failure")

    monkeypatch.setattr(journal, "append", fail)
    with pytest.raises(subject.LaunchAccountingError) as raised:
        subject.launch_operation(
            journal,
            "op-cleanup-01",
            _command("import time; time.sleep(10)"),
            payload=subject.operation_payload("status", b""),
            parent_identity="900@100",
            env=_THREAD_CAP_ENV,
        )
    assert raised.value.child.poll() is None
    raised.value.child.terminate()
    assert raised.value.child.wait() is not None


def test_malformed_postspawn_payload_and_nonfinite_counter_refuse(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with pytest.raises(subject.LaunchAccountingError) as raised:
        subject.launch_operation(
            journal,
            "op-cleanup-02",
            _command("import time; time.sleep(10)"),
            payload={"nonfinite": float("nan")},
            parent_identity="900@100",
            env=_THREAD_CAP_ENV,
        )
    raised.value.child.terminate()
    assert raised.value.child.wait() is not None
    other = _journal(tmp_path / "counter")
    for value in (float("nan"), float("inf"), -0.1, True):
        with pytest.raises(subject.AccountingError):
            subject.record_counter_sample(
                other, process_id=1, counter_seconds=value, sample_kind="independent-ps"
            )


def test_nonfinite_final_wait4_counter_refuses(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    child, tracked = _launch(journal, "op-final-counter", "pass")
    child.wait()
    child.final_rusage = SimpleNamespace(ru_utime=float("inf"), ru_stime=0.0)
    with pytest.raises(subject.AccountingError, match="finite"):
        subject.wait4_exit(journal, child, tracked)
