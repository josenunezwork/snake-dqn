"""Focused admission tests; no Colab command is invoked here."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from research.s1_mature_20261018 import manifests
from research.s1_mature_20261018.ledger import (
    Ledger,
    Refused,
    parallel_elapsed_schedule,
    validate_staging_continuation,
)


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
            "inputs.py",
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
            "train-b": bound(16, 8, 0, 8),
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


def parallel_amendment(root: Path) -> Path:
    path = root / "parallel-amendment.md"
    path.write_text("# S1 parallel lanes\n\n**Status: APPROVED_S1_PARALLEL_20261019.**\n")
    return path


def parallel_snapshot(root: Path, *, now: float, records: list[dict]) -> Path:
    return dump(
        root / f"parallel-account-{len(records)}.json",
        {
            "captured_at": now,
            "balance_units": 1174.37,
            "usage_rate_units_per_hour": sum(row["rate_units_per_hour"] for row in records),
            "active_assignments": len(records),
            "assignments_known": True,
            "active_records": records,
            "sessions": {row["endpoint"]: row["session"] for row in records},
        },
    )


def parallel_manifest(path: Path, job_id: str) -> Path:
    value = json.loads(manifest(path, job_id).read_text())
    value["output_dir"] = str(path.parent / f"output-{job_id}")
    value["env"] = {"S1_ARM": "n3" if job_id.endswith("a") else "n1"}
    return dump(path, value)


def test_parallel_mode_binds_amendment_and_requires_broker_known_lanes(prepared):
    worker_ledger, root, _ = prepared
    amendment = parallel_amendment(root)
    worker_ledger.configure_parallel(amendment, lanes=2)
    first = parallel_manifest(root / "parallel-a.json", "train-a")
    worker_ledger.record_parallel_account_snapshot(
        parallel_snapshot(root, now=1_000_000.0, records=[])
    )
    worker_ledger.reserve_parallel("train-a", bound(), first)

    unknown = parallel_snapshot(
        root,
        now=1_000_000.0,
        records=[
            {
                "endpoint": "foreign-endpoint",
                "session": "foreign",
                "output": "/unknown/output",
                "rate_units_per_hour": 2,
            }
        ],
    )
    with pytest.raises(Refused, match="unknown assignment|misses a live S1 lane"):
        worker_ledger.record_parallel_account_snapshot(unknown)

    known = parallel_snapshot(
        root,
        now=1_000_000.0,
        records=[
            {
                "endpoint": "endpoint-a",
                "session": "snake-a",
                "output": json.loads(first.read_text())["output_dir"],
                "rate_units_per_hour": 2,
            }
        ],
    )
    worker_ledger.record_parallel_account_snapshot(known)
    second = parallel_manifest(root / "parallel-b.json", "train-b")
    worker_ledger.reserve_parallel("train-b", bound(16, 8, 0, 8), second)
    assert worker_ledger.status()["outstanding_jobs"] == ["train-a", "train-b"]


def test_parallel_schedule_uses_dynamic_list_scheduling_upper_bound():
    schedule = parallel_elapsed_schedule(
        {"a": bound(elapsed=8), "b": bound(elapsed=4), "c": bound(elapsed=4)}, 2
    )
    assert schedule["scheduled_makespan_hours"] == 8
    assert schedule["makespan_hours"] == 12
    with pytest.raises(Refused, match="lanes"):
        parallel_elapsed_schedule({}, 4)


def test_parallel_handoff_recomputes_elapsed_without_resetting_start(prepared):
    worker, root, _ = prepared
    worker.clock = lambda: 1_000_000.0 + 27 * 3600
    with pytest.raises(Refused, match="elapsed_hours"):
        worker.status()
    before = worker.path.read_bytes()
    amendment = parallel_amendment(root)
    preview = worker.preview_parallel(amendment, lanes=3)
    assert preview["projected_total"]["elapsed_hours"] == pytest.approx(27 + 12 / 3 + 2 * 8 / 3)
    assert worker.path.read_bytes() == before
    configured = worker.configure_parallel(amendment, lanes=3)
    assert configured["study_started_at"] == 1_000_000.0
    assert configured["projected_total"]["units"] == preview["projected_total"]["units"]


def test_parallel_claim_is_atomic_and_never_double_picks(prepared):
    worker_ledger, root, _ = prepared
    worker_ledger.configure_parallel(parallel_amendment(root), lanes=2)
    first = parallel_manifest(root / "claim-a.json", "train-a")
    worker_ledger.record_parallel_account_snapshot(
        parallel_snapshot(root, now=1_000_000.0, records=[])
    )
    worker_ledger.reserve_parallel("train-a", bound(), first)
    snapshot = parallel_snapshot(
        root,
        now=1_000_000.0,
        records=[
            {
                "endpoint": "endpoint-a",
                "session": "snake-a",
                "output": json.loads(first.read_text())["output_dir"],
                "rate_units_per_hour": 2,
            }
        ],
    )
    second = parallel_manifest(root / "claim-b.json", "train-b")

    def claim():
        try:
            return worker_ledger.claim_next(
                ["train-b"],
                {"train-b": bound(16, 8, 0, 8)},
                {"train-b": second},
                account_snapshot_path=snapshot,
            )
        except Refused:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: claim(), range(2)))
    assert outcomes.count("train-b") == 1
    assert worker_ledger.status()["outstanding_jobs"] == ["train-a", "train-b"]


def test_parallel_lane_cap_refuses_a_fourth_live_reservation(prepared):
    worker_ledger, root, _ = prepared
    # Add two registered primary liabilities before the amendment is bound; this
    # models a four-job immutable plan without changing any active reservation.
    state = json.loads(worker_ledger.path.read_text())
    state["planned"].update({"train-c": bound(4, 2, 0, 2), "train-d": bound(4, 2, 0, 2)})
    dump(worker_ledger.path, state)
    worker_ledger.configure_parallel(parallel_amendment(root), lanes=3)
    records = []
    for job_id in ("train-a", "train-b", "train-c"):
        manifest_path = parallel_manifest(root / f"{job_id}.json", job_id)
        worker_ledger.record_parallel_account_snapshot(
            parallel_snapshot(root, now=1_000_000.0, records=records)
        )
        worker_ledger.reserve_parallel(job_id, state["planned"][job_id], manifest_path)
        records.append(
            {
                "endpoint": f"endpoint-{job_id}",
                "session": f"snake-{job_id}",
                "output": json.loads(manifest_path.read_text())["output_dir"],
                "rate_units_per_hour": 2,
            }
        )
    worker_ledger.record_parallel_account_snapshot(
        parallel_snapshot(root, now=1_000_000.0, records=records)
    )
    fourth = parallel_manifest(root / "train-d.json", "train-d")
    with pytest.raises(Refused, match="lane count"):
        worker_ledger.reserve_parallel("train-d", bound(4, 2, 0, 2), fourth)
    assert worker_ledger.status()["outstanding_jobs"] == ["train-a", "train-b", "train-c"]


def test_parallel_settlement_uses_lane_accounting_not_raw_account_debit(prepared):
    worker_ledger, root, _ = prepared
    worker_ledger.configure_parallel(parallel_amendment(root), lanes=2)
    source = parallel_manifest(root / "settle-a.json", "train-a")
    worker_ledger.record_parallel_account_snapshot(
        parallel_snapshot(root, now=1_000_000.0, records=[])
    )
    worker_ledger.reserve_parallel("train-a", bound(), source)
    receipt = dump(
        root / "parallel-receipt.json",
        {
            "status": "PASS",
            "stop_verified": True,
            "error": None,
            "cleanup_error": None,
            "endpoint": "endpoint-a",
            "terminal": {"done": True},
            "manifest_sha256": digest(source),
            # Includes another lane's debit and would exceed this lane bound.
            "usage_before": {"balance": 100},
            "usage_after": {"balance": 1},
            "elapsed_seconds": 1,
            "accounting": {
                "allocated_endpoint": "endpoint-a",
                "unaccounted_endpoints": [],
                "assignments": [],
                "baseline_rate_units_per_hour": 0,
                "own_rate_units_per_hour": 2,
                "observed_rate_units_per_hour": 2,
                "unit_cap": 8,
                "calibration": {"status": "PASS"},
                "integrated_accounted_other_units": 50,
                "last_crosscheck": {"attributable_account_debit_units": 1},
            },
        },
    )
    worker_ledger.settle("train-a", receipt)
    assert worker_ledger.status()["spent"]["units"] == 10


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


@pytest.mark.parametrize("insufficient", [None, *bound().keys(), "elapsed_gap", "balance"])
def test_manual_operational_retry_is_one_per_arm_and_semantically_identical(prepared, insufficient):
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
    assert "retry-n3" not in worker_ledger.status()["planned_jobs"]
    if insufficient:
        state = json.loads(worker_ledger.path.read_text())
        if insufficient in bound():
            spent = worker_ledger.status()["spent"][insufficient]
            state["planned"]["train-b"][insufficient] = (
                state["usable_caps"][insufficient]
                - spent
                + (0.01 if insufficient == "mac_worker_hours" else -0.01)
            )
        elif insufficient == "elapsed_gap":
            worker_ledger.clock = lambda: 1_000_000 + 39 * 3600
            snapshot = json.loads((root / "account.json").read_text())
            snapshot["captured_at"] = worker_ledger.clock()
            fresh = dump(root / "fresh-account.json", snapshot)
        else:
            snapshot = json.loads((root / "account.json").read_text())
            snapshot["balance_units"] = 16  # Remaining primary fits; rerun does not.
            fresh = dump(root / "fresh-account.json", snapshot)
        worker_ledger.path.write_text(json.dumps(state))
        options = (
            {"account_snapshot_path": fresh} if insufficient in {"balance", "elapsed_gap"} else {}
        )
        with pytest.raises(Refused, match="N3 INCOMPLETE"):
            worker_ledger.reserve_manual_retry("train-a", "retry-n3", retry_path, **options)
        refused = json.loads(worker_ledger.path.read_text())
        assert refused["incomplete_arms"]["n3"]["status"] == "INCOMPLETE"
        if insufficient == "elapsed_gap":
            assert "usable elapsed_hours" in refused["incomplete_arms"]["n3"]["reason"]
        assert refused["outstanding"] == {}
        assert refused["planned"] == state["planned"]
        assert refused["settled"] == state["settled"]
        return
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


def test_parallel_operational_failure_uses_lane_accounting_and_refuses_missing_accounting(prepared):
    worker_ledger, root, _ = prepared
    worker_ledger.configure_parallel(parallel_amendment(root), lanes=2)
    source = parallel_manifest(root / "parallel-lost-a.json", "train-a")
    worker_ledger.record_parallel_account_snapshot(
        parallel_snapshot(root, now=1_000_000.0, records=[])
    )
    worker_ledger.reserve_parallel("train-a", bound(), source)
    receipt = dump(
        root / "parallel-lost-a-receipt.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "endpoint": "endpoint-a",
            "manifest_sha256": digest(source),
            # This is account-wide debit, including a healthy sibling lane.
            "usage_before": {"balance": 100, "assignments": 2, "rate": 4},
            "usage_after": {"balance": 1, "assignments": 1, "rate": 2},
            "elapsed_seconds": 1,
            "verified_final_export": False,
            "accounting": {
                "allocated_endpoint": "endpoint-a",
                "unaccounted_endpoints": [],
                "assignments": [],
                "baseline_rate_units_per_hour": 0,
                "own_rate_units_per_hour": 2,
                "observed_rate_units_per_hour": 2,
                "unit_cap": 8,
                "calibration": {"status": "PASS"},
                "integrated_accounted_other_units": 50,
                "last_crosscheck": {"attributable_account_debit_units": 1},
            },
        },
    )
    review = dump(
        root / "parallel-lost-a-review.json",
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
    summary = worker_ledger.reconcile_operational_failure("train-a", receipt, review)
    assert summary["spent"]["units"] == 10

    next_source = parallel_manifest(root / "parallel-lost-b.json", "train-b")
    worker_ledger.record_parallel_account_snapshot(
        parallel_snapshot(root, now=1_000_000.0, records=[])
    )
    worker_ledger.reserve_parallel("train-b", bound(16, 8, 0, 8), next_source)
    missing = dict(json.loads(receipt.read_text()))
    missing.update({"endpoint": "endpoint-b", "manifest_sha256": digest(next_source)})
    missing.pop("accounting")
    missing_receipt = dump(root / "parallel-lost-b-receipt.json", missing)
    missing_review = dict(json.loads(review.read_text()))
    missing_review.update(
        {
            "job_id": "train-b",
            "manifest_sha256": digest(next_source),
            "receipt_sha256": digest(missing_receipt),
        }
    )
    missing_review_path = dump(root / "parallel-lost-b-review.json", missing_review)
    with pytest.raises(Refused, match="reconciled lane accounting"):
        worker_ledger.reconcile_operational_failure("train-b", missing_receipt, missing_review_path)
    assert worker_ledger.status()["outstanding_jobs"] == ["train-b"]


def test_parallel_staging_and_qualification_charge_reserved_lane_units(prepared):
    worker_ledger, root, _ = prepared
    source = manifest(root / "parallel-staging-manifest.json")
    entry = {"bound": bound(), "manifest": worker_ledger._evidence(source), "parallel": True}
    receipt = {
        "endpoint": "endpoint-a",
        "usage_before": {"balance": 100, "assignments": 2, "rate": 4},
        "usage_after": {"balance": 1, "assignments": 1, "rate": 2},
        "elapsed_seconds": 10,
        "accounting": {
            "allocated_endpoint": "endpoint-a",
            "unaccounted_endpoints": [],
            "assignments": [],
            "baseline_rate_units_per_hour": 0,
            "own_rate_units_per_hour": 2,
            "observed_rate_units_per_hour": 2,
            "unit_cap": 8,
            "calibration": {"status": "PASS"},
            "integrated_accounted_other_units": 50,
            "last_crosscheck": None,
        },
    }
    actual = worker_ledger._staging_actual(receipt, entry)
    assert actual["units"] == 8
    assert actual["gpu_allocation_hours"] == pytest.approx(10 / 3600)
    missing = dict(receipt)
    missing.pop("accounting")
    with pytest.raises(Refused, match="reconciled lane accounting"):
        worker_ledger._staging_actual(missing, entry)


def test_preallocation_refusal_restores_same_job_without_consuming_retry(prepared):
    worker_ledger, root, snapshot = prepared
    source = manifest(root / "source.json")
    worker_ledger.reserve("train-a", bound(), source)
    # The admission snapshot is historical evidence by recovery time; only the
    # fresh post-refusal snapshot is subject to the normal freshness window.
    worker_ledger.clock = lambda: 1_001_000
    state_dir = root / "supervisor-empty"
    state_dir.mkdir()
    log = root / "dispatch.log"
    log.write_text("symlink prohibited in fallback source: research/s1/S1_SEED_AUDIT.json\n")
    fresh = dump(
        root / "fresh-account.json",
        {
            "captured_at": 1_001_000,
            "balance_units": 1174.37,
            "usage_rate_units_per_hour": 0,
            "active_assignments": 0,
            "assignments_known": True,
        },
    )
    review = dump(
        root / "preallocation-review.json",
        {
            "schema": "s1-preallocation-refusal/v1",
            "status": "REFUSED_BEFORE_ALLOCATION",
            "job_id": "train-a",
            "manifest_sha256": digest(source),
            "failure_classification": "source_preflight",
            "colab_new_called": False,
            "assignment_created": False,
            "before_snapshot_sha256": digest(snapshot),
            "fresh_snapshot_sha256": digest(fresh),
            "supervisor_state_dir": str(state_dir.resolve()),
            "dispatch_log": {"path": str(log), "sha256": digest(log)},
        },
    )
    summary = worker_ledger.reconcile_preallocation_refusal(
        "train-a", review, snapshot, fresh, state_dir
    )
    assert summary["planned_jobs"] == ["train-a", "train-b"]
    assert summary["outstanding_jobs"] == []
    assert "retry-n3" not in summary["settled_jobs"]
    assert summary["spent"]["units"] == 2
    assert summary["remaining"]["units"] == 24
    saved = json.loads(worker_ledger.path.read_text())
    event = saved["events"][-1]
    assert event["kind"] == "preallocation_refusal"
    assert event["failure_classification"] == "source_preflight"
    assert saved["settled"].get("train-a") is None
    refreshed = json.loads(source.read_text())
    refreshed.update({"commit": "c" * 40, "output_dir": str(root / "refreshed-output")})
    refreshed_path = dump(root / "refreshed.json", refreshed)
    assert worker_ledger.validate_preallocation_continuation("train-a", refreshed_path)[
        "sha256"
    ] == digest(refreshed_path)
    (state_dir / "unexpected-state").write_text("state")
    with pytest.raises(Refused, match="supervisor state directory is not empty"):
        worker_ledger.status()
    (state_dir / "unexpected-state").unlink()
    log.write_text("tampered\n")
    with pytest.raises(Refused, match="preallocation dispatch log hash mismatch"):
        worker_ledger.status()


@pytest.mark.parametrize(
    "defect",
    (
        "missing_log",
        "stale_fresh",
        "reused_snapshot",
        "changed_balance",
        "nonempty_state",
        "changed_review",
    ),
)
def test_preallocation_refusal_requires_fresh_bound_zero_allocation_evidence(prepared, defect):
    worker_ledger, root, snapshot = prepared
    source = manifest(root / "source.json")
    worker_ledger.reserve("train-a", bound(), source)
    state_dir = root / "supervisor-empty"
    state_dir.mkdir()
    log = root / "dispatch.log"
    log.write_text("symlink prohibited in fallback source: research/s1/S1_SEED_AUDIT.json\n")
    fresh_data = {
        "captured_at": 1_000_000,
        "balance_units": 1174.37,
        "usage_rate_units_per_hour": 0,
        "active_assignments": 0,
        "assignments_known": True,
    }
    if defect == "stale_fresh":
        fresh_data["captured_at"] = 0
    elif defect == "changed_balance":
        fresh_data["balance_units"] = 1173.37
    fresh = dump(root / "fresh-account.json", fresh_data)
    if defect == "reused_snapshot":
        fresh = snapshot
    review = dump(
        root / "preallocation-review.json",
        {
            "schema": "s1-preallocation-refusal/v1",
            "status": "REFUSED_BEFORE_ALLOCATION",
            "job_id": "train-a",
            "manifest_sha256": digest(source),
            "failure_classification": "source_preflight",
            "colab_new_called": False,
            "assignment_created": False,
            "before_snapshot_sha256": digest(snapshot),
            "fresh_snapshot_sha256": digest(fresh),
            "supervisor_state_dir": str(state_dir.resolve()),
            "dispatch_log": {"path": str(log), "sha256": digest(log)},
        },
    )
    if defect == "missing_log":
        log.unlink()
    elif defect == "nonempty_state":
        (state_dir / "unexpected").write_text("state")
    elif defect == "changed_review":
        value = json.loads(review.read_text())
        value["assignment_created"] = True
        review.write_text(json.dumps(value))
    with pytest.raises((FileNotFoundError, Refused)):
        worker_ledger.reconcile_preallocation_refusal("train-a", review, snapshot, fresh, state_dir)
    assert worker_ledger.status()["outstanding_jobs"] == ["train-a"]


def test_preallocation_continuation_only_allows_commit_change(prepared):
    worker_ledger, root, _ = prepared
    source = dump(
        root / "source.json",
        {
            "job_id": "train-a",
            "commit": "a" * 40,
            "script": "research/s1_mature_20261018/train.sh",
            "env": {"S1_ARM": "n3", "S1_SEED": "7"},
            "requirements": {"gpu": "L4"},
            "timeout_minutes": 120,
            "output_dir": str(root / "new-output"),
            "inputs": [{"dest": "seed-bank.json", "sha256": "b" * 64}],
            "budget": {"rate_units_per_hour": 2},
        },
    )
    worker_ledger.reserve("train-a", bound(), source)
    refreshed = json.loads(source.read_text())
    refreshed["commit"] = "c" * 40
    refreshed_path = dump(root / "refreshed.json", refreshed)
    assert worker_ledger.validate_preallocation_continuation("train-a", refreshed_path)[
        "sha256"
    ] == digest(refreshed_path)
    refreshed["env"]["S1_SEED"] = "8"
    changed = dump(root / "changed.json", refreshed)
    with pytest.raises(Refused, match="frozen scientific"):
        worker_ledger.validate_preallocation_continuation("train-a", changed)
    refreshed["env"]["S1_SEED"] = "7"
    refreshed["output_dir"] = str(root / "changed-output")
    changed = dump(root / "changed-output.json", refreshed)
    assert worker_ledger.validate_preallocation_continuation("train-a", changed)[
        "sha256"
    ] == digest(changed)
    Path(refreshed["output_dir"]).mkdir(parents=True)
    with pytest.raises(Refused, match="fresh absolute output"):
        worker_ledger.validate_preallocation_continuation("train-a", changed)
    refreshed["commit"] = "A" * 40
    invalid_commit = dump(root / "invalid-commit.json", refreshed)
    with pytest.raises(Refused, match="full SHA"):
        worker_ledger.validate_preallocation_continuation("train-a", invalid_commit)


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
    with pytest.raises(Refused, match="evidence identity changed"):
        ledger.status()


@pytest.mark.parametrize(
    "seconds, debit, expected",
    [(10, 0.2, 0.2), (339.0773167080042, 0.04, 339.0773167080042 * 2 / 3600)],
)
def test_pretraining_staging_failure_charges_actual_and_restores_same_job(
    prepared, seconds, debit, expected
):
    worker_ledger, root, _ = prepared
    source = manifest(root / "staging-manifest.json")
    worker_ledger.reserve("train-a", bound(), source)
    receipt = dump(
        root / "staging-receipt.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "endpoint": "l4-endpoint",
            "terminal": None,
            "exports": [],
            "error": "upload /content/snake/inputs/league.tar.gz timed out",
            "manifest_sha256": digest(source),
            "usage_before": {"balance": 100, "assignments": 0, "rate": 0},
            "usage_after": {"balance": 100 - debit, "assignments": 0, "rate": 0},
            "elapsed_seconds": seconds,
        },
    )
    review = dump(
        root / "staging-review.json",
        {
            "schema": "s1-pretraining-staging-failure-review/v1",
            "status": "PASS",
            "job_id": "train-a",
            "manifest_sha256": digest(source),
            "receipt_sha256": digest(receipt),
            "failure_classification": "pretraining_staging_failure",
            "owner_approved": True,
            "trainer_process_started": False,
            "exports_present": False,
        },
    )
    summary = worker_ledger.settle_pretraining_staging_failure("train-a", receipt, review)
    assert summary["outstanding_jobs"] == []
    assert "train-a" in summary["planned_jobs"]
    assert worker_ledger.status()["outstanding_jobs"] == []
    assert summary["spent"]["units"] == pytest.approx(2 + expected)
    state = json.loads(worker_ledger.path.read_text())
    settled = state["settled"]["staging-failure:train-a"]
    assert settled["actual"]["gpu_allocation_hours"] == pytest.approx(seconds / 3600)
    assert settled["original_bound"] == bound()
    event = state["events"][-1]
    assert event["kind"] == "pretraining_staging_failure"
    assert event["manifest"]["sha256"] == digest(source)


@pytest.mark.parametrize("field, value", [("terminal", {"started": True}), ("exports", ["x"])])
def test_pretraining_staging_failure_refuses_trainer_or_exports(prepared, field, value):
    worker_ledger, root, _ = prepared
    source = manifest(root / "staging-manifest.json")
    worker_ledger.reserve("train-a", bound(), source)
    receipt_value = {
        "status": "INCOMPLETE",
        "stop_verified": True,
        "cleanup_error": None,
        "endpoint": "l4-endpoint",
        "terminal": None,
        "exports": [],
        "error": "upload /content/snake/inputs/league.tar.gz timed out",
        "manifest_sha256": digest(source),
        "usage_before": {"balance": 100, "assignments": 0, "rate": 0},
        "usage_after": {"balance": 100, "assignments": 0, "rate": 0},
        "elapsed_seconds": 1,
    }
    receipt_value[field] = value
    receipt = dump(root / "staging-receipt.json", receipt_value)
    review = dump(root / "staging-review.json", {})
    with pytest.raises(Refused, match="trainer start|exports exist"):
        worker_ledger.settle_pretraining_staging_failure("train-a", receipt, review)
    assert worker_ledger.status()["outstanding_jobs"] == ["train-a"]


@pytest.mark.parametrize(
    "seconds, debit, expected",
    [(10, 0.2, 0.2), (339.0773167080042, 0.04, 339.0773167080042 * 2 / 3600)],
)
def test_pretraining_qualification_failure_charges_actual_and_restores_same_job(
    prepared, seconds, debit, expected
):
    worker_ledger, root, _ = prepared
    source = manifest(root / "qualification-manifest.json")
    worker_ledger.reserve("train-a", bound(), source)
    terminal = dump(
        root / "terminal.json",
        {
            "status": "failed",
            "exports": [],
            "generation": "g",
            "job_id": "train-a",
            "returncode": 2,
        },
    )
    poll = dump(
        root / "poll-000001.json",
        {
            "job_log_tail": "q0b_cuda_fixtures returned non-zero exit status 2",
            "terminal": json.loads(terminal.read_text()),
        },
    )
    receipt = dump(
        root / "qualification-receipt.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "endpoint": "l4-endpoint",
            "generation": "g",
            "exports": [],
            "error": "Refused('remote terminal failed')",
            "manifest_sha256": digest(source),
            "usage_before": {"balance": 100, "assignments": 0, "rate": 0},
            "usage_after": {"balance": 100 - debit, "assignments": 0, "rate": 0},
            "elapsed_seconds": seconds,
        },
    )
    review = dump(
        root / "qualification-review.json",
        {
            "schema": "s1-pretraining-qualification-failure-review/v1",
            "status": "PASS",
            "job_id": "train-a",
            "manifest_sha256": digest(source),
            "receipt_sha256": digest(receipt),
            "failure_classification": "pretraining_qualification_failure",
            "owner_approved": True,
            "trainer_process_started": True,
            "qualification_passed": False,
            "training_metrics_written": False,
            "training_exports_present": False,
            "no_efficacy_look": True,
            "terminal_evidence": {"path": str(terminal), "sha256": digest(terminal)},
            "poll_evidence": {"path": str(poll), "sha256": digest(poll)},
        },
    )
    summary = worker_ledger.settle_pretraining_qualification_failure("train-a", receipt, review)
    assert summary["outstanding_jobs"] == []
    assert "train-a" in summary["planned_jobs"]
    state = json.loads(worker_ledger.path.read_text())
    settled = state["settled"]["qualification-failure:train-a"]
    assert settled["actual"]["units"] == pytest.approx(expected)
    assert settled["actual"]["gpu_allocation_hours"] == pytest.approx(seconds / 3600)
    assert settled["actual"]["elapsed_hours"] == pytest.approx(seconds / 3600)
    assert state["events"][-1]["kind"] == "pretraining_qualification_failure"


@pytest.mark.parametrize(
    "field", ["qualification_passed", "training_metrics_written", "training_exports_present"]
)
def test_pretraining_qualification_failure_refuses_any_training_evidence(prepared, field):
    worker_ledger, root, _ = prepared
    source = manifest(root / "qualification-manifest.json")
    worker_ledger.reserve("train-a", bound(), source)
    terminal = dump(
        root / "terminal.json",
        {
            "status": "failed",
            "exports": [],
            "generation": "g",
            "job_id": "train-a",
            "returncode": 2,
        },
    )
    poll = dump(
        root / "poll-000001.json",
        {
            "job_log_tail": "q0b_cuda_fixtures returned non-zero exit status 2",
            "terminal": json.loads(terminal.read_text()),
        },
    )
    receipt = dump(
        root / "qualification-receipt.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "endpoint": "e",
            "generation": "g",
            "exports": [],
            "error": "failed",
            "manifest_sha256": digest(source),
            "usage_before": {"balance": 1, "assignments": 0, "rate": 0},
            "usage_after": {"balance": 1, "assignments": 0, "rate": 0},
            "elapsed_seconds": 1,
        },
    )
    review_value = {
        "schema": "s1-pretraining-qualification-failure-review/v1",
        "status": "PASS",
        "job_id": "train-a",
        "manifest_sha256": digest(source),
        "receipt_sha256": digest(receipt),
        "failure_classification": "pretraining_qualification_failure",
        "owner_approved": True,
        "trainer_process_started": True,
        "qualification_passed": False,
        "training_metrics_written": False,
        "training_exports_present": False,
        "no_efficacy_look": True,
        "terminal_evidence": {"path": str(terminal), "sha256": digest(terminal)},
        "poll_evidence": {"path": str(poll), "sha256": digest(poll)},
    }
    review_value[field] = True
    review = dump(root / "qualification-review.json", review_value)
    with pytest.raises(Refused, match="no-training evidence"):
        worker_ledger.settle_pretraining_qualification_failure("train-a", receipt, review)
    assert worker_ledger.status()["outstanding_jobs"] == ["train-a"]


def test_pretraining_qualification_failure_refuses_training_progress_in_bound_poll(prepared):
    worker_ledger, root, _ = prepared
    source = manifest(root / "qualification-manifest.json")
    worker_ledger.reserve("train-a", bound(), source)
    terminal = dump(
        root / "terminal.json",
        {
            "status": "failed",
            "exports": [],
            "generation": "g",
            "job_id": "train-a",
            "returncode": 2,
        },
    )
    poll = dump(
        root / "poll-000001.json",
        {
            "job_log_tail": "q0b_cuda_fixtures returned non-zero exit status 2",
            "terminal": json.loads(terminal.read_text()),
            "progress": {"phase": "training", "done": 1},
        },
    )
    receipt = dump(
        root / "qualification-receipt.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "endpoint": "e",
            "generation": "g",
            "exports": [],
            "error": "failed",
            "manifest_sha256": digest(source),
            "usage_before": {"balance": 1, "assignments": 0, "rate": 0},
            "usage_after": {"balance": 1, "assignments": 0, "rate": 0},
            "elapsed_seconds": 1,
        },
    )
    review = dump(
        root / "qualification-review.json",
        {
            "schema": "s1-pretraining-qualification-failure-review/v1",
            "status": "PASS",
            "job_id": "train-a",
            "manifest_sha256": digest(source),
            "receipt_sha256": digest(receipt),
            "failure_classification": "pretraining_qualification_failure",
            "owner_approved": True,
            "trainer_process_started": True,
            "qualification_passed": False,
            "training_metrics_written": False,
            "training_exports_present": False,
            "no_efficacy_look": True,
            "terminal_evidence": {"path": str(terminal), "sha256": digest(terminal)},
            "poll_evidence": {"path": str(poll), "sha256": digest(poll)},
        },
    )
    with pytest.raises(Refused, match="training progress"):
        worker_ledger.settle_pretraining_qualification_failure("train-a", receipt, review)


def test_pretraining_qualification_failure_refuses_export_file_absent_from_receipt(prepared):
    worker_ledger, root, _ = prepared
    source = manifest(root / "qualification-manifest.json")
    worker_ledger.reserve("train-a", bound(), source)
    terminal = dump(
        root / "terminal.json",
        {
            "status": "failed",
            "exports": [],
            "generation": "g",
            "job_id": "train-a",
            "returncode": 2,
        },
    )
    poll = dump(
        root / "poll-000001.json",
        {
            "job_log_tail": "q0b_cuda_fixtures returned non-zero exit status 2",
            "terminal": json.loads(terminal.read_text()),
        },
    )
    (root / "exports").mkdir()
    (root / "exports" / "s1-final.part-000").write_bytes(b"partial training export")
    receipt = dump(
        root / "qualification-receipt.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "endpoint": "e",
            "generation": "g",
            "exports": [],
            "error": "failed",
            "manifest_sha256": digest(source),
            "usage_before": {"balance": 1, "assignments": 0, "rate": 0},
            "usage_after": {"balance": 1, "assignments": 0, "rate": 0},
            "elapsed_seconds": 1,
        },
    )
    review = dump(
        root / "qualification-review.json",
        {
            "schema": "s1-pretraining-qualification-failure-review/v1",
            "status": "PASS",
            "job_id": "train-a",
            "manifest_sha256": digest(source),
            "receipt_sha256": digest(receipt),
            "failure_classification": "pretraining_qualification_failure",
            "owner_approved": True,
            "trainer_process_started": True,
            "qualification_passed": False,
            "training_metrics_written": False,
            "training_exports_present": False,
            "no_efficacy_look": True,
            "terminal_evidence": {"path": str(terminal), "sha256": digest(terminal)},
            "poll_evidence": {"path": str(poll), "sha256": digest(poll)},
        },
    )
    with pytest.raises(Refused, match="training export exists"):
        worker_ledger.settle_pretraining_qualification_failure("train-a", receipt, review)


def test_staging_continuation_preserves_input_science_and_binds_fresh_acceptance(tmp_path: Path):
    bank = tmp_path / "bank.bin"
    bank.write_bytes(b"bank")
    old_acceptance = tmp_path / "old-acceptance.json"
    old_acceptance.write_text("old")
    fresh_acceptance = tmp_path / "fresh-acceptance.json"
    fresh_acceptance.write_text("fresh")
    old = {
        "job_id": "s1-mature-n3-7",
        "commit": "a" * 40,
        "output_dir": "/old/out",
        "inputs": [
            {"dest": "bank.bin", "path": str(bank), "sha256": digest(bank)},
            {
                "dest": "cli-acceptance-result.json",
                "path": str(old_acceptance),
                "sha256": digest(old_acceptance),
            },
        ],
        "env": {"S1_ARM": "n3"},
    }
    refreshed = {
        **old,
        "commit": "b" * 40,
        "output_dir": "/fresh/out",
        "inputs": [
            {
                "dest": "bank.bin",
                "url": "https://example.invalid/bank.bin",
                "sha256": digest(bank),
                "size_bytes": bank.stat().st_size,
            },
            {
                "dest": "cli-acceptance-result.json",
                "path": str(fresh_acceptance),
                "sha256": digest(fresh_acceptance),
                "size_bytes": fresh_acceptance.stat().st_size,
            },
        ],
        "env": {"S1_ARM": "n3", "COLAB_CLI_REQUIREMENTS": manifests.RUNTIME_REQUIREMENTS},
        "source_paths": list(manifests.RUNTIME_SOURCE_PATHS),
    }
    with pytest.raises(Refused, match="reviewed binding"):
        validate_staging_continuation(old, refreshed)
    validate_staging_continuation(
        old,
        refreshed,
        acceptance_result={"path": str(fresh_acceptance), "sha256": digest(fresh_acceptance)},
    )
    refreshed["inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(Refused, match="frozen input identity"):
        validate_staging_continuation(
            old,
            refreshed,
            acceptance_result={"path": str(fresh_acceptance), "sha256": digest(fresh_acceptance)},
        )


def test_requalify_acceptance_is_idempotent_and_charges_new_review(prepared):
    worker_ledger, root, _ = prepared
    state = json.loads(worker_ledger.path.read_text())
    review = Path(state["acceptance_review"]["path"])
    first = worker_ledger.requalify_acceptance(review)
    second = worker_ledger.requalify_acceptance(review)
    assert first == second
    state = json.loads(worker_ledger.path.read_text())
    assert "cli-acceptance-requalification" in state["settled"]


def test_zero_assignments_with_nonzero_rate_refuses_new_snapshot(prepared):
    ledger, root, _ = prepared
    snapshot = dump(
        root / "unknown-rate.json",
        {
            "captured_at": 1_000_001,
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


@pytest.fixture
def legacy_staging(prepared):
    worker, root, _ = prepared
    test_pretraining_staging_failure_charges_actual_and_restores_same_job(
        prepared, 496.2047262920096, 0.19, 496.2047262920096 * 2 / 3600
    )
    state = json.loads(worker.path.read_text())
    sid = "staging-failure:train-a"
    item = state["settled"][sid]
    corrected = dict(item["actual"])
    item["actual"]["units"] = 100 - (100 - 0.19)
    worker.path.write_text(json.dumps(state))
    review = {
        "schema": "s1-staging-charge-correction/v1",
        "status": "PASS",
        "settlement_id": sid,
        "settlement_sha256": hashlib.sha256(
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "receipt": item["receipt"],
        "source_manifest": item["source_manifest"],
        "staging_failure_review": item["staging_failure_review"],
        "corrected_actual": corrected,
        "additional_charge": {key: corrected[key] - item["actual"][key] for key in corrected},
    }
    path = dump(root / "correction-review.json", review)
    return worker, sid, path, state


def test_staging_correction_is_append_only_counted_once_and_allows_outstanding(legacy_staging):
    worker, sid, review, before = legacy_staging
    # A later attempt can remain outstanding while the old debit is corrected.
    worker.reserve("train-a", bound(), manifest(review.parent / "next-manifest.json"))
    before = json.loads(worker.path.read_text())
    old_spent = worker.status()["spent"]
    result = worker.correct_staging_charge(sid, review)
    after = json.loads(worker.path.read_text())
    assert after["settled"] == before["settled"]
    assert after["events"][:-1] == before["events"]
    assert after["planned"] == before["planned"]
    assert after["outstanding"] == before["outstanding"]
    assert result["spent"]["units"] - old_spent["units"] == pytest.approx(0.0856692923844)
    for key in ("gpu_allocation_hours", "elapsed_hours", "mac_worker_hours"):
        assert result["spent"][key] == old_spent[key]
    assert worker.status()["spent"] == result["spent"]
    saved = worker.path.read_bytes()
    with pytest.raises(Refused, match="already applied"):
        worker.correct_staging_charge(sid, review)
    assert worker.path.read_bytes() == saved


@pytest.mark.parametrize("field", ["receipt", "settlement_sha256", "additional_charge"])
def test_staging_correction_refuses_changed_review_without_mutation(legacy_staging, field):
    worker, sid, path, _ = legacy_staging
    review = json.loads(path.read_text())
    review[field] = "wrong"
    dump(path, review)
    saved = worker.path.read_bytes()
    with pytest.raises(Refused, match="correction review"):
        worker.correct_staging_charge(sid, path)
    assert worker.path.read_bytes() == saved


def test_staging_correction_rechecks_evidence_and_event_amount(legacy_staging):
    worker, sid, path, _ = legacy_staging
    worker.correct_staging_charge(sid, path)
    state = json.loads(worker.path.read_text())
    state["events"][-1]["actual"]["units"] = 0
    dump(worker.path, state)
    with pytest.raises(Refused, match="event charge"):
        worker.status()
    state["events"][-1]["actual"]["units"] = json.loads(path.read_text())["additional_charge"][
        "units"
    ]
    dump(worker.path, state)
    path.write_text("{}")
    with pytest.raises(Refused, match="hash mismatch"):
        worker.status()


def test_staging_correction_obeys_caps_atomically(legacy_staging):
    worker, sid, path, _ = legacy_staging
    state = json.loads(worker.path.read_text())
    remaining = sum(x["units"] for x in state["planned"].values())
    spent = worker.status()["spent"]["units"]
    state["planned"]["train-b"]["units"] += 48 - remaining - spent - 0.01
    dump(worker.path, state)
    saved = worker.path.read_bytes()
    with pytest.raises(Refused, match="usable units"):
        worker.correct_staging_charge(sid, path)
    assert worker.path.read_bytes() == saved


@pytest.mark.parametrize("rate", [None, 0, -1, float("nan"), True])
def test_failure_charge_refuses_unknown_or_invalid_priced_ceiling(prepared, rate):
    worker, root, _ = prepared
    source = manifest(root / "bad-rate.json")
    value = json.loads(source.read_text())
    value["budget"]["rate_units_per_hour"] = rate
    dump(source, value)
    receipt = {
        "usage_before": {"balance": 100, "assignments": 0, "rate": 0},
        "usage_after": {"balance": 100, "assignments": 0, "rate": 0},
        "elapsed_seconds": 339,
    }
    with pytest.raises(Refused, match="rate"):
        worker._staging_actual(receipt, {"manifest": worker._evidence(source), "bound": bound()})


def test_failure_priced_charge_cannot_exceed_reserved_units(prepared):
    worker, root, _ = prepared
    source = manifest(root / "manifest.json")
    receipt = {
        "usage_before": {"balance": 100, "assignments": 0, "rate": 0},
        "usage_after": {"balance": 100, "assignments": 0, "rate": 0},
        "elapsed_seconds": 3600,
    }
    with pytest.raises(Refused, match="exceeds reserved"):
        worker._staging_actual(
            receipt, {"manifest": worker._evidence(source), "bound": bound(units=1)}
        )
