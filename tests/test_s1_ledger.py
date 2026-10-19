"""Focused admission tests; no Colab command is invoked here."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from research.s1_mature_20261018.ledger import Ledger, Refused


def dump(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transport_sources() -> list[dict[str, str]]:
    root = Path(__file__).parents[1] / "research/colab_cli"
    return [
        {"path": str(root / name), "sha256": digest(root / name)}
        for name in (
            "supervisor.py",
            "runtime.py",
            "accounting.py",
            "release.py",
            "transfers.py",
            "cli_guard.py",
            "bootstrap.sh",
            "launch.sh",
        )
    ]


def qualified_result() -> dict:
    return {
        "qualified": True,
        "final_plan": {
            "execution": [{"job": "cpu", "status": "PASS"}, {"job": "l4", "status": "PASS"}]
        },
    }


def fault_evidence(root: Path) -> dict:
    fault = dump(
        root / "l4-fault-receipt.json",
        {
            "status": "INCOMPLETE",
            "cleanup_error": None,
            "stop_verified": True,
            "endpoint": "l4-endpoint",
        },
    )
    observer = dump(
        root / "l4-observer.json",
        {
            "gates": {
                "mac-loss-watchdog-unassign": {
                    "endpoint": "l4-endpoint",
                    "assignment_disappeared_within_bound": True,
                    "renewal_suppressed": True,
                    "checkpoint_exported_during_grace": True,
                    "release_bound_seconds": 300,
                    "observed_after_lease_seconds": 299,
                }
            }
        },
    )
    return {
        "receipt": {"path": str(fault), "sha256": digest(fault)},
        "observer": {"path": str(observer), "sha256": digest(observer)},
    }


def bound(units=8.0, gpu=4.0, mac=0.0, elapsed=4.0):
    return {
        "units": units,
        "gpu_allocation_hours": gpu,
        "mac_worker_hours": mac,
        "elapsed_hours": elapsed,
    }


@pytest.fixture
def prepared(tmp_path: Path):
    now = 1_000_000.0
    supervisor = Path(__file__).parents[1] / "research/colab_cli/supervisor.py"
    result = dump(tmp_path / "acceptance-result.json", qualified_result())
    receipt = dump(tmp_path / "acceptance-receipt.json", {"status": "PASS", "stop_verified": True})
    review = dump(
        tmp_path / "review.json",
        {
            "schema": "s1-cli-acceptance-review/v1",
            "status": "PASS",
            "supervisor_sha256": digest(supervisor),
            "acceptance_result": {"path": str(result), "sha256": digest(result)},
            "receipts": [{"path": str(receipt), "sha256": digest(receipt)}],
            "transport_sources": transport_sources(),
            "fault_receipt": fault_evidence(tmp_path),
            "watchdog_disappearance_seconds": 299,
            "study_started_at": now,
            "acceptance_charge": bound(2, 1, 0.2, 1),
        },
    )
    snap = dump(
        tmp_path / "account.json",
        {
            "captured_at": now,
            "balance_units": 1174.37,
            "usage_rate_units_per_hour": 0,
            "active_assignments": 0,
            "assignments_known": True,
        },
    )
    history = {
        name: dump(tmp_path / f"{name}.json", {"identity": name}) for name in ("q1", "a2", "a3")
    }
    ledger = Ledger(tmp_path / "ledger.json", clock=lambda: now)
    ledger.initialize(
        acceptance_review_path=review,
        account_snapshot_path=snap,
        history_refs=history,
        planned_liabilities={
            "train-a": bound(),
            "train-b": bound(),
            "retry-n3": bound(),
        },
    )
    return ledger, tmp_path, snap


def manifest(path: Path, job_id="train-a") -> Path:
    return dump(
        path,
        {
            "job_id": job_id,
            "timeout_minutes": 120,
            "requirements": {"gpu": "L4"},
            "budget": {"rate_units_per_hour": 2},
        },
    )


def test_reservation_is_atomic_and_one_active_job(prepared):
    ledger, root, _ = prepared
    target = manifest(root / "manifest.json")

    def reserve():
        try:
            ledger.reserve("train-a", bound(), target)
            return "ok"
        except Refused:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: reserve(), range(2)))
    assert outcomes.count("ok") == 1
    assert ledger.status()["outstanding_jobs"] == ["train-a"]


def test_cap_boundary_refuses_complete_plan(tmp_path: Path):
    # Acceptance itself consumes 2 units, so a 48-unit remaining plan is inadmissible.
    now = 4.0
    supervisor = Path(__file__).parents[1] / "research/colab_cli/supervisor.py"
    result, receipt = dump(tmp_path / "r", qualified_result()), dump(
        tmp_path / "x", {"status": "PASS", "stop_verified": True}
    )
    review = dump(
        tmp_path / "review",
        {
            "schema": "s1-cli-acceptance-review/v1",
            "status": "PASS",
            "supervisor_sha256": digest(supervisor),
            "acceptance_result": {"path": str(result), "sha256": digest(result)},
            "receipts": [{"path": str(receipt), "sha256": digest(receipt)}],
            "transport_sources": transport_sources(),
            "fault_receipt": fault_evidence(tmp_path),
            "watchdog_disappearance_seconds": 1,
            "study_started_at": now,
            "acceptance_charge": bound(2, 0, 0, 0),
        },
    )
    snap = dump(
        tmp_path / "s",
        {
            "captured_at": now,
            "balance": 100,
            "rate": 0,
            "assignments": 0,
            "assignments_known": True,
        },
    )
    refs = {x: dump(tmp_path / x, {}) for x in ("q1", "a2", "a3")}
    with pytest.raises(Refused, match="usable units"):
        Ledger(tmp_path / "l", clock=lambda: now).initialize(
            acceptance_review_path=review,
            account_snapshot_path=snap,
            history_refs=refs,
            planned_liabilities={"x": bound(48, 0, 0, 0)},
        )


def test_failed_or_missing_receipt_cannot_release_reservation(prepared):
    ledger, root, _ = prepared
    target = manifest(root / "manifest.json")
    ledger.reserve("train-a", bound(), target)
    with pytest.raises((FileNotFoundError, Refused)):
        ledger.settle("train-a", root / "absent.json")
    bad = dump(root / "bad.json", {"status": "INCOMPLETE"})
    with pytest.raises(Refused):
        ledger.settle("train-a", bad)
    assert ledger.status()["outstanding_jobs"] == ["train-a"]


def test_tampered_evidence_and_stale_snapshot_refuse(prepared):
    ledger, root, snapshot = prepared
    # A fresh ledger also refuses an old, screen-observed account reading.
    stale = dump(
        root / "stale.json",
        {"captured_at": 0, "balance": 1174, "rate": 0, "assignments": 0, "assignments_known": True},
    )
    with pytest.raises(Refused, match="stale"):
        ledger.record_account_snapshot(stale)
    snapshot.write_text("{}")
    with pytest.raises(Refused, match="snapshot evidence changed"):
        ledger.status()


def test_cli_receipt_charges_full_request_even_if_provider_debit_lags(prepared):
    ledger, root, _ = prepared
    target = manifest(root / "manifest.json")
    ledger.reserve("train-a", bound(), target)
    receipt = dump(
        root / "receipt.json",
        {
            "status": "PASS",
            "stop_verified": True,
            "error": None,
            "cleanup_error": None,
            "endpoint": "e",
            "terminal": {"done": True},
            "manifest_sha256": digest(target),
            "usage_before": {"balance": 100},
            "usage_after": {"balance": 100},
            "elapsed_seconds": 1,
        },
    )
    ledger.settle("train-a", receipt)
    state = json.loads(ledger.path.read_text())
    assert state["settled"]["train-a"]["actual"]["units"] == 8
    summary = ledger.status()
    assert summary["spent"]["units"] == 10  # 2-unit acceptance + full 8-unit request.
    assert summary["outstanding"] == bound(0, 0, 0, 0)
    assert summary["remaining"]["units"] == 16


def test_failed_release_and_unknown_interval_keep_reservation_outstanding(prepared):
    ledger, root, _ = prepared
    target = manifest(root / "manifest.json")
    ledger.reserve("train-a", bound(), target)
    common = {
        "status": "PASS",
        "stop_verified": True,
        "error": None,
        "endpoint": "e",
        "terminal": {"done": True},
        "manifest_sha256": digest(target),
        "usage_before": {"balance": 10},
        "usage_after": {"balance": 9},
    }
    failed_release = dump(
        root / "failed-release.json",
        {**common, "cleanup_error": "STOP_UNVERIFIED", "elapsed_seconds": 1},
    )
    with pytest.raises(Refused, match="execution or cleanup"):
        ledger.settle("train-a", failed_release)
    unknown_interval = dump(root / "unknown-interval.json", {**common, "cleanup_error": None})
    with pytest.raises(Refused, match="elapsed_seconds"):
        ledger.settle("train-a", unknown_interval)
    assert ledger.status()["outstanding_jobs"] == ["train-a"]


def test_canonical_manifest_must_preserve_reserved_identity_and_budget(prepared):
    ledger, root, _ = prepared
    target = manifest(root / "source.json")
    ledger.reserve("train-a", bound(), target)
    canonical = dump(
        root / "manifest.json",
        {**json.loads(target.read_text()), "budget": {"rate_units_per_hour": 1}},
    )
    receipt = dump(
        root / "receipt.json",
        {
            "status": "PASS",
            "stop_verified": True,
            "error": None,
            "cleanup_error": None,
            "endpoint": "e",
            "terminal": {"done": True},
            "manifest_sha256": digest(canonical),
            "usage_before": {"balance": 10},
            "usage_after": {"balance": 9},
            "elapsed_seconds": 1,
        },
    )
    with pytest.raises(Refused, match="differs"):
        ledger.settle("train-a", receipt)


def test_manual_operational_retry_is_one_per_arm_and_semantically_identical(prepared):
    worker_ledger, root, _ = prepared
    source = dump(
        root / "source-n3.json",
        {
            "job_id": "train-a",
            "commit": "a" * 40,
            "script": "research/s1_mature_20261018/train.sh",
            "env": {"S1_ARM": "n3", "S1_SEED": "7", "S1_TRAINING_NAMESPACE": "ns"},
            "requirements": {"gpu": "L4"},
            "timeout_minutes": 120,
            "output_dir": str(root / "original-out"),
            "inputs": [{"dest": "seed-bank.json", "sha256": "b" * 64}],
            "budget": {"rate_units_per_hour": 2},
        },
    )
    worker_ledger.reserve("train-a", bound(), source)
    receipt = dump(
        root / "lost.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "endpoint": "exact-endpoint",
            "manifest_sha256": digest(source),
            "usage_before": {"balance": 100},
            "usage_after": {"balance": 99},
            "elapsed_seconds": 10,
            "verified_final_export": False,
        },
    )
    review = dump(
        root / "review-operational.json",
        {
            "schema": "s1-operational-failure-review/v1",
            "status": "PASS",
            "job_id": "train-a",
            "manifest_sha256": digest(source),
            "receipt_sha256": digest(receipt),
            "failure_classification": "transfer_failure",
            "no_efficacy_look": True,
            "mac_final_evaluation_planned": True,
            "verified_final_export": False,
            "tripwire": False,
            "integrity_failure": False,
            "parity_failure": False,
            "hash_failure": False,
            "counter_failure": False,
        },
    )
    worker_ledger.reconcile_operational_failure("train-a", receipt, review)
    retry = json.loads(source.read_text())
    retry.update(
        {
            "job_id": "retry-n3",
            "output_dir": str(root / "fresh-out"),
            "attempt": {"kind": "manual-operational-rerun", "of": "train-a"},
        }
    )
    retry_path = dump(root / "retry.json", retry)
    worker_ledger.reserve_manual_retry("train-a", "retry-n3", retry_path)
    assert worker_ledger.status()["outstanding_jobs"] == ["retry-n3"]
    with pytest.raises(Refused, match="unavailable"):
        worker_ledger.reserve_manual_retry("train-a", "retry-n3", retry_path)


def test_operational_retry_refuses_integrity_failure_or_verified_final(prepared):
    worker_ledger, root, _ = prepared
    target = manifest(root / "manifest.json")
    worker_ledger.reserve("train-a", bound(), target)
    receipt = dump(
        root / "lost.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "endpoint": "e",
            "manifest_sha256": digest(target),
            "usage_before": {"balance": 1},
            "usage_after": {"balance": 1},
            "elapsed_seconds": 1,
            "verified_final_export": True,
        },
    )
    review = dump(
        root / "operational-review.json",
        {"schema": "s1-operational-failure-review/v1", "status": "PASS"},
    )
    with pytest.raises(Refused, match="verified final"):
        worker_ledger.reconcile_operational_failure("train-a", receipt, review)


@pytest.mark.parametrize(
    "dimension", ("units", "gpu_allocation_hours", "mac_worker_hours", "elapsed_hours")
)
def test_every_usable_cap_boundary_is_enforced(prepared, dimension):
    ledger, _, _ = prepared
    state = json.loads(ledger.path.read_text())
    state["planned"]["train-b"][dimension] = state["usable_caps"][dimension] + 1
    ledger.path.write_text(json.dumps(state))
    with pytest.raises(Refused, match=f"usable {dimension}"):
        ledger.status()


def test_nested_acceptance_and_supervisor_bindings_are_rechecked(prepared):
    ledger, root, _ = prepared
    state = json.loads(ledger.path.read_text())
    review_path = Path(state["acceptance_review"]["path"])
    review = json.loads(review_path.read_text())
    nested = Path(review["receipts"][0]["path"])
    nested.write_text("tampered")
    with pytest.raises(Refused, match="hash mismatch"):
        ledger.status()
    nested.write_text(json.dumps({"status": "PASS", "stop_verified": True}))
    review["receipts"][0]["sha256"] = digest(nested)
    review["supervisor_sha256"] = "0" * 64
    review_path.write_text(json.dumps(review))
    with pytest.raises(Refused, match="current supervisor"):
        ledger.status()


def test_zero_assignments_with_nonzero_rate_refuses_new_snapshot(prepared):
    ledger, root, _ = prepared
    snapshot = dump(
        root / "unknown-rate.json",
        {
            "captured_at": 1_000_000,
            "balance": 100,
            "rate": 0.01,
            "assignments": 0,
            "assignments_known": True,
        },
    )
    with pytest.raises(Refused, match="unknown debit"):
        ledger.record_account_snapshot(snapshot)


def test_elapsed_gap_blocks_remaining_study_even_without_a_job(tmp_path: Path):
    now = [1_000_000.0]
    supervisor = Path(__file__).parents[1] / "research/colab_cli/supervisor.py"
    result, receipt = dump(tmp_path / "r", qualified_result()), dump(
        tmp_path / "x", {"status": "PASS", "stop_verified": True}
    )
    review = dump(
        tmp_path / "review",
        {
            "schema": "s1-cli-acceptance-review/v1",
            "status": "PASS",
            "supervisor_sha256": digest(supervisor),
            "acceptance_result": {"path": str(result), "sha256": digest(result)},
            "receipts": [{"path": str(receipt), "sha256": digest(receipt)}],
            "transport_sources": transport_sources(),
            "fault_receipt": fault_evidence(tmp_path),
            "watchdog_disappearance_seconds": 1,
            "study_started_at": now[0],
            "acceptance_charge": bound(0, 0, 0, 0),
        },
    )
    snapshot = dump(
        tmp_path / "snapshot",
        {
            "captured_at": now[0],
            "balance": 100,
            "rate": 0,
            "assignments": 0,
            "assignments_known": True,
        },
    )
    refs = {name: dump(tmp_path / name, {}) for name in ("q1", "a2", "a3")}
    ledger = Ledger(tmp_path / "ledger", clock=lambda: now[0])
    ledger.initialize(
        acceptance_review_path=review,
        account_snapshot_path=snapshot,
        history_refs=refs,
        planned_liabilities={"train": bound(1, 1, 0, 1)},
    )
    now[0] += 38 * 3600
    with pytest.raises(Refused, match="usable elapsed_hours"):
        ledger.status()


def test_reviewed_start_is_required_and_cannot_reset_elapsed_window(prepared):
    ledger, root, _ = prepared
    state = json.loads(ledger.path.read_text())
    review_path = Path(state["acceptance_review"]["path"])
    review = json.loads(review_path.read_text())
    review.pop("study_started_at")
    review_path.write_text(json.dumps(review))
    with pytest.raises(Refused, match="study_started_at"):
        Ledger(root / "missing-start-ledger", clock=lambda: 1_000_000).initialize(
            acceptance_review_path=review_path,
            account_snapshot_path=state["account_snapshot"]["path"],
            history_refs={name: item["path"] for name, item in state["history_refs"].items()},
            planned_liabilities={"x": bound(1, 1, 0, 1)},
        )
    review["study_started_at"] = 1_000_000 - 39 * 3600
    review_path.write_text(json.dumps(review))
    with pytest.raises(Refused, match="usable elapsed_hours"):
        Ledger(root / "gap-start-ledger", clock=lambda: 1_000_000).initialize(
            acceptance_review_path=review_path,
            account_snapshot_path=state["account_snapshot"]["path"],
            history_refs={name: item["path"] for name, item in state["history_refs"].items()},
            planned_liabilities={"x": bound(1, 1, 0, 1)},
        )


def test_mac_collective_evaluation_reserves_and_settles_worker_time(prepared):
    ledger, root, _ = prepared
    state = json.loads(ledger.path.read_text())
    state["planned"] = {"eval-final": bound(0, 0, 2, 3)}
    state["outstanding"] = {}
    ledger.path.write_text(json.dumps(state))
    mac_manifest = dump(
        root / "eval-manifest.json",
        {"kind": "mac", "job_id": "eval-final", "workers": 2, "timeout_minutes": 120},
    )
    ledger.reserve("eval-final", bound(0, 0, 2, 3), mac_manifest)
    receipt = dump(
        root / "eval-receipt.json",
        {"status": "PASS", "job_id": "eval-final", "actual": bound(0, 0, 1.5, 2)},
    )
    summary = ledger.settle_mac("eval-final", receipt)
    assert summary["spent"]["mac_worker_hours"] == pytest.approx(1.7)
    assert summary["spent"]["units"] == 2


def test_mac_manifest_refuses_more_than_two_workers(prepared):
    ledger, root, _ = prepared
    state = json.loads(ledger.path.read_text())
    state["planned"] = {"eval-final": bound(0, 0, 2, 3)}
    ledger.path.write_text(json.dumps(state))
    mac_manifest = dump(
        root / "eval-manifest.json",
        {"kind": "mac", "job_id": "eval-final", "workers": 3, "timeout_minutes": 120},
    )
    with pytest.raises(Refused, match="worker/timeout"):
        ledger.reserve("eval-final", bound(0, 0, 2, 3), mac_manifest)


def test_initialize_is_create_once_under_concurrent_attempts(tmp_path: Path):
    now = 1_000_000.0
    supervisor = Path(__file__).parents[1] / "research/colab_cli/supervisor.py"
    result, receipt = dump(tmp_path / "r", qualified_result()), dump(
        tmp_path / "x", {"status": "PASS", "stop_verified": True}
    )
    review = dump(
        tmp_path / "review",
        {
            "schema": "s1-cli-acceptance-review/v1",
            "status": "PASS",
            "supervisor_sha256": digest(supervisor),
            "acceptance_result": {"path": str(result), "sha256": digest(result)},
            "receipts": [{"path": str(receipt), "sha256": digest(receipt)}],
            "transport_sources": transport_sources(),
            "fault_receipt": fault_evidence(tmp_path),
            "watchdog_disappearance_seconds": 1,
            "study_started_at": now,
            "acceptance_charge": bound(0, 0, 0, 0),
        },
    )
    snapshot = dump(
        tmp_path / "snapshot",
        {
            "captured_at": now,
            "balance": 100,
            "rate": 0,
            "assignments": 0,
            "assignments_known": True,
        },
    )
    refs = {name: dump(tmp_path / name, {}) for name in ("q1", "a2", "a3")}

    def initialize():
        try:
            Ledger(tmp_path / "ledger", clock=lambda: now).initialize(
                acceptance_review_path=review,
                account_snapshot_path=snapshot,
                history_refs=refs,
                planned_liabilities={"train": bound(1, 1, 0, 1)},
            )
            return "ok"
        except Refused:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: initialize(), range(2)))
    assert outcomes.count("ok") == 1
