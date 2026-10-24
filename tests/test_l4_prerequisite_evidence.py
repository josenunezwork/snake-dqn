"""Regression evidence for the refused L4 v2 prerequisite (Astra 2026-10-24)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.bl_labels import l4_faults_v2 as v2
from research.colab_cli import transfers
from research.colab_cli.acceptance_faults import AcceptanceObserver, DirectObserverDownloadEvidence

FIXTURES = Path(__file__).parent / "fixtures" / "l4_prerequisite_refusal"
CPU_RECEIPT_SHA = "ff4546c1166617ddad4ef177c25c18fab74222dcdeeb131a0f73f0d926214685"
L4_RECEIPT_SHA = "ab95e746a8a235f9bcc2965fa4c9d62c68f3338d8ae4936f700a98a49b1add33"


class _CLI:
    session = "evidence-session"

    def __init__(self) -> None:
        self.call_records: list[dict] = []
        self.transfer_records: list[dict] = []


def _row(remote: str, status: str, ordinal: int, **extra: object) -> dict:
    return {
        "command": "download",
        "remote": remote,
        "status": status,
        "attempt_id": f"call-{ordinal}",
        "operation_id": f"operation-{ordinal}",
        "started_monotonic": float(ordinal),
        "finished_monotonic": float(ordinal) + 0.25,
        "elapsed_seconds": 0.25,
        "timeout_seconds": 3.0,
        **extra,
    }


def test_refused_fixtures_are_exact_receipts_and_observers() -> None:
    assert (
        hashlib.sha256((FIXTURES / "cpu-receipt.json").read_bytes()).hexdigest() == CPU_RECEIPT_SHA
    )
    assert hashlib.sha256((FIXTURES / "l4-receipt.json").read_bytes()).hexdigest() == L4_RECEIPT_SHA
    for arm in ("cpu", "l4"):
        assert json.loads((FIXTURES / f"{arm}-acceptance-observer.json").read_text())
        assert json.loads((FIXTURES / f"{arm}.injection.json").read_text())[
            "written_before_dispatch"
        ]


def test_refused_arms_remain_evidence_missing_with_no_counted_faults() -> None:
    for arm in ("cpu", "l4"):
        result = v2.classify_session(
            json.loads((FIXTURES / f"{arm}-receipt.json").read_text()),
            json.loads((FIXTURES / f"{arm}.injection.json").read_text()),
            acceptance_observer=json.loads(
                (FIXTURES / f"{arm}-acceptance-observer.json").read_text()
            ),
        )
        assert result["verdict"] == "EVIDENCE_MISSING"
        assert result["counted_by_class"] == {
            name: 0 for name in ("C1", "C2", "C3", "C4", "C5", "C6")
        }


def test_real_cli_missing_diagnostic_requires_declared_selector() -> None:
    receipt = json.loads((FIXTURES / "cpu-receipt.json").read_text())
    row = copy.deepcopy(receipt["cli_calls"][92])
    cli = _CLI()
    cli.call_records.append(row)
    selectors = {"/client-start.json": "client-loss-start-absent"}
    AcceptanceObserver._tag_observed_absence(cli, row["remote"], declared_selectors=selectors)
    assert row["failure_type"] == "NOT_FOUND"
    assert row["realized_selector_id"] == "client-loss-start-absent"

    unproved = copy.deepcopy(receipt["cli_calls"][92])
    cli = _CLI()
    cli.call_records.append(unproved)
    AcceptanceObserver._tag_observed_absence(cli, unproved["remote"], declared_selectors={})
    assert "failure_type" not in unproved
    assert "realized_selector_id" not in unproved


def test_real_timeout_is_not_absence_and_never_gets_a_future_digest() -> None:
    receipt = json.loads((FIXTURES / "l4-receipt.json").read_text())
    timeout = copy.deepcopy(receipt["cli_calls"][166])
    cli = _CLI()
    cli.call_records.append(timeout)
    AcceptanceObserver._tag_observed_absence(
        cli,
        timeout["remote"],
        declared_selectors={
            "/output/lease-checkpoint.pkl.sha256": "watchdog-checkpoint-marker-absent"
        },
    )
    assert timeout["failure_class"] == "RPC_EXEC_POLL_TIMEOUT"
    assert "failure_type" not in timeout
    assert "verified_sha256" not in timeout


def test_absence_parser_rejects_wrong_remote_timeout_503_and_traceback() -> None:
    remote = "/content/root/client-start.json"
    declared = {"/client-start.json": "client-loss-start-absent"}
    exact = "[colab] Download failed: File or directory not found: "
    cases = [
        exact + "/content/root/other.json",
        "HTTP 503 Service Unavailable\n" + exact + remote,
        "Traceback (most recent call last):\n" + exact + remote,
        "path mentions HTTP 404 but is not a terminal provider status",
    ]
    for error in cases:
        cli = _CLI()
        row = _row(remote, "failed", 1, error=error)
        cli.call_records.append(row)
        AcceptanceObserver._tag_observed_absence(cli, remote, declared_selectors=declared)
        assert "failure_type" not in row
        assert "realized_selector_id" not in row


def test_direct_observer_keeps_raw_ordinals_and_binds_only_completed_row() -> None:
    remote = "/content/root/output/lease-checkpoint.pkl.sha256"
    digest = "a" * 64
    cli = _CLI()
    evidence = DirectObserverDownloadEvidence(cli, {remote})
    for ordinal in range(1, 5):
        cli.call_records.append(_row(remote, "failed", ordinal, error="missing"))
        evidence.observe_since(len(cli.call_records) - 1)
    successful = _row(remote, "completed", 5)
    cli.call_records.append(successful)
    evidence.observe_since(len(cli.call_records) - 1)
    assert [row["transfer_attempt"] for row in cli.call_records] == [1, 2, 3, 4, 5]
    assert len({row["parent_transfer_id"] for row in cli.call_records}) == 1
    assert len({row["transfer_id"] for row in cli.call_records}) == 1
    assert evidence.bind_verified(remote, digest) is True
    assert [row.get("verified_sha256") for row in cli.call_records] == [
        None,
        None,
        None,
        None,
        digest,
    ]
    assert cli.transfer_records[0]["children"][0]["attempts"] == 5


def test_census_verification_preserves_observer_identity_and_failed_row() -> None:
    remote = "/content/root/client-start.json"
    payload = b"marker"
    digest = hashlib.sha256(payload).hexdigest()
    cli = _CLI()
    evidence = DirectObserverDownloadEvidence(cli, {remote})
    failed = _row(remote, "failed", 1, error="missing")
    completed = _row(remote, "completed", 2)
    cli.call_records.extend((failed, completed))
    evidence.observe_since(0)
    assert transfers.record_census_verified_download(
        cli,
        first_call=1,
        remote=remote,
        local_size=len(payload),
        local_sha256=digest,
        census={"client_files": {remote: {"sha256": digest, "size": len(payload)}}},
    )
    assert failed["transfer_attempt"] == 1 and "verified_sha256" not in failed
    assert completed["transfer_attempt"] == 2 and completed["verified_sha256"] == digest
    assert len(cli.transfer_records) == 1


def test_missing_census_stays_evidence_missing_but_digest_mismatch_stays_c6() -> None:
    remote = "/content/root/client-start.json"
    payload = b"marker"
    digest = hashlib.sha256(payload).hexdigest()
    cli = _CLI()
    evidence = DirectObserverDownloadEvidence(cli, {remote}, verification_kind="census-file-digest")
    cli.call_records.append(_row(remote, "completed", 1))
    evidence.observe_since(0)
    assert not transfers.record_census_verified_download(
        cli, first_call=0, remote=remote, local_size=len(payload), local_sha256=digest, census={}
    )
    base = {
        "job_id": "job-1",
        "status": "PASS",
        "stop_verified": True,
        "cli_calls": cli.call_records,
        "transfers": cli.transfer_records,
        "contents_transfers": [],
        "marker_census": {},
    }
    assert v2.classify_session(base)["verdict"] == "EVIDENCE_MISSING"

    mismatch = copy.deepcopy(cli)
    assert not transfers.record_census_verified_download(
        mismatch,
        first_call=0,
        remote=remote,
        local_size=len(payload),
        local_sha256=digest,
        census={"client_files": {remote: {"sha256": "0" * 64, "size": len(payload)}}},
    )
    result = v2.classify_session({**base, "transfers": mismatch.transfer_records})
    assert result["counted_by_class"]["C6"] == 1


def _prospective_receipt(cli: _CLI, **extra: object) -> dict:
    """A new offline operation, never a reclassification of an old receipt."""
    return {
        "job_id": "prospective-counterexample",
        "status": "PASS",
        "stop_verified": True,
        "cli_calls": cli.call_records,
        "transfers": cli.transfer_records,
        "contents_transfers": [],
        "marker_census": {},
        **extra,
    }


def _unbound_real_row(index: int) -> dict:
    receipt = json.loads((FIXTURES / "l4-receipt.json").read_text())
    row = copy.deepcopy(receipt["cli_calls"][index])
    for key in ("parent_transfer_id", "transfer_id", "transfer_attempt", "verified_sha256"):
        row.pop(key, None)
    return row


def _verified_real_timeout_operation() -> tuple[_CLI, str]:
    """New two-attempt operation with the real timeout shape and verified bytes."""
    cli = _CLI()
    timeout, completed = _unbound_real_row(166), _unbound_real_row(204)
    remote = timeout["remote"]
    evidence = DirectObserverDownloadEvidence(cli, {remote})
    for row in (timeout, completed):
        cli.call_records.append(row)
        evidence.observe_since(len(cli.call_records) - 1)
    # Read-only hash verification, never unpickle the real checkpoint.
    digest = hashlib.sha256((FIXTURES / "watchdog-checkpoint.pkl").read_bytes()).hexdigest()
    assert (FIXTURES / "watchdog-checkpoint.pkl.sha256").read_text().split()[0] == digest
    assert evidence.bind_verified(remote, digest)
    return cli, remote


def test_real_timeout_shape_requires_fully_correlated_prospective_recovery() -> None:
    cli, _remote = _verified_real_timeout_operation()
    failed, completed = cli.call_records
    assert failed["failure_class"] == "RPC_EXEC_POLL_TIMEOUT"
    assert failed["elapsed_seconds"] == 3.019524540985003
    assert failed["timeout_seconds"] == 2.9999962500296533
    assert "verified_sha256" not in failed
    assert [failed["transfer_attempt"], completed["transfer_attempt"]] == [1, 2]
    result = v2.classify_session(_prospective_receipt(cli))
    assert result["verdict"] == "PASS", result
    assert result["recovered"][0]["attempts"] == 2
    assert result["recovered"][0]["seconds"] == 79.20817104203161

    unverified = copy.deepcopy(cli)
    unverified.call_records[1].pop("verified_sha256")
    result = v2.classify_session(_prospective_receipt(unverified))
    assert result["counted_by_class"]["C6"] >= 1


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_remote",
        "wrong_parent",
        "wrong_child",
        "changed_hash",
        "excess_retries",
        "excess_bound",
        "503",
    ],
)
def test_real_timeout_counterexamples_stay_counted(defect: str) -> None:
    cli, remote = _verified_real_timeout_operation()
    failed, completed = cli.call_records
    if defect == "wrong_remote":
        completed["remote"] = remote + ".foreign"
    elif defect == "wrong_parent":
        completed["parent_transfer_id"] += "-foreign"
    elif defect == "wrong_child":
        cli.transfer_records[0]["children"][0]["remote"] = remote + ".foreign"
    elif defect == "changed_hash":
        completed["verified_sha256"] = "0" * 64
    elif defect == "excess_retries":
        completed["transfer_attempt"] = 4
    elif defect == "excess_bound":
        failed["elapsed_seconds"] = failed["timeout_seconds"] + 2.01
    elif defect == "503":
        failed.update(failure_class="SUPERVISOR", error="HTTP 503 Service Unavailable")
        # A 503 without later verified recovery must remain a real transfer fault.
        cli.call_records.pop()
    result = v2.classify_session(_prospective_receipt(cli))
    assert result["verdict"] == "FAIL", result
    assert result["counted_by_class"]["C1" if defect == "excess_bound" else "C6"] >= 1


def test_stale_earlier_verification_never_recovers_a_later_timeout() -> None:
    cli, _remote = _verified_real_timeout_operation()
    failed, completed = cli.call_records
    completed.update(started_monotonic=0.0, finished_monotonic=1.0, transfer_attempt=1)
    failed["transfer_attempt"] = 2
    cli.call_records[:] = [completed, failed]
    result = v2.classify_session(_prospective_receipt(cli))
    assert result["counted_by_class"]["C6"] == 1


def test_real_refused_poll_sequence_exposes_prospective_raw_retry_conflict() -> None:
    """Diagnostic of a future same-shape operation, not amended historical evidence."""
    cli = _CLI()
    rows = [_unbound_real_row(i) for i in range(123, 206)]
    remotes = {row["remote"] for row in rows}
    evidence = DirectObserverDownloadEvidence(cli, remotes)
    injection = json.loads((FIXTURES / "l4.injection.json").read_text())
    observer = AcceptanceObserver("unused", injections={injection["session_job_id"]: injection})
    selectors = observer._declared_absence_selectors(injection["session_job_id"])
    for row in rows:
        cli.call_records.append(row)
        evidence.observe_since(len(cli.call_records) - 1)
        observer._tag_observed_absence(cli, row["remote"], declared_selectors=selectors)
    digest = hashlib.sha256((FIXTURES / "watchdog-checkpoint.pkl").read_bytes()).hexdigest()
    for remote in remotes:
        assert evidence.bind_verified(remote, digest)
    timeout = cli.call_records[166 - 123]
    assert timeout["transfer_attempt"] == 44
    assert cli.call_records[204 - 123]["transfer_attempt"] == 82
    assert "failure_type" not in timeout and "verified_sha256" not in timeout
    result = v2.classify_session(
        _prospective_receipt(cli, job_id=injection["session_job_id"]), injection
    )
    assert result["counted_by_class"] == {**{f"C{i}": 0 for i in range(1, 7)}, "C6": 1}
    assert result["counted"][0]["index"] == 166 - 123
    assert result["verdict"] == "FAIL"


def test_repeated_successful_sidecars_bind_only_the_final_verified_read() -> None:
    remote = "/content/root/output/lease-checkpoint.pkl.sha256"
    cli = _CLI()
    evidence = DirectObserverDownloadEvidence(cli, {remote})
    for ordinal in (1, 2):
        cli.call_records.append(_row(remote, "completed", ordinal))
        evidence.observe_since(len(cli.call_records) - 1)
    assert evidence.bind_verified(remote, "a" * 64)
    assert "verified_sha256" not in cli.call_records[0]
    assert cli.call_records[1]["verified_sha256"] == "a" * 64
    # The final pair is proved, but an earlier unverified consumed read is not forgiven.
    result = v2.classify_session(_prospective_receipt(cli))
    assert result["counted_by_class"]["C6"] == 1


@pytest.mark.parametrize(
    "error",
    [
        "Traceback (most recent call last):\nHTTP 404 Not Found: /content/root/other.json",
        "HTTP 404 Not Found: /content/root/other.json",
        "HTTPError 503 Server Error\nHTTP 404 Not Found",
        "download timed out\nHTTP 404 Not Found",
        "not found in local cache",
        "HTTP 404 Not Found: /content/root/CLIENT-START.json",
    ],
)
def test_legacy_status_text_never_hides_wrong_remote_or_real_failure(error: str) -> None:
    remote = "/content/root/client-start.json"
    cli = _CLI()
    row = _row(remote, "failed", 1, error=error)
    cli.call_records.append(row)
    AcceptanceObserver._tag_observed_absence(
        cli, remote, declared_selectors={"/client-start.json": "client-loss-start-absent"}
    )
    assert "failure_type" not in row
