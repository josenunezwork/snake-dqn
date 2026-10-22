"""Synthetic native STOP receipt integrity; no game worlds or Mac timing."""

import copy

import pytest

from research.bl_served_search import native_ruler, ruler
from research.bl_served_search.test_ruler import _fixtures


def paired_receipt(label: str = "mac", *, canary: bool = False) -> dict:
    fixtures = {**_fixtures(), "source_identity": ruler.source_identity()}
    repeats = 1 if canary else 30
    cells = {}
    for mode in native_ruler.MODES:
        records = []
        for fixture in fixtures["fixtures"]:
            order = [0, 1, 2] if mode == "all" else [0]
            outcomes = {
                str(action): (
                    {"survives": True, "frames_executed": 300}
                    if action in order
                    else {"status": "NOT_EVALUATED"}
                )
                for action in fixture["legal_actions"]
            }
            batch = {
                "semantic": {
                    "fired": True,
                    "chosen": 0,
                    "bare": 0,
                    "search_mode": mode,
                    "outcomes": outcomes,
                    "evaluation_order": order,
                    "branches_evaluated": len(order),
                    "branches_skipped": 3 - len(order),
                    "rollout_branch_frames": 300 * len(order),
                },
                "native_v8": {
                    "mode": "kernel",
                    "kernel_rows": 0 if fixture["mix"] == "scripted" else 5,
                    "stock_rows": 0,
                    "noncandidate_rows": 0,
                    "fallback_rows": 0,
                    "calls": 0 if fixture["mix"] == "scripted" else 1,
                },
                "stage_seconds": dict.fromkeys(
                    (*native_ruler.STAGES, "submit_visible", "submit_consumed"), 0.0
                ),
                "worker_wall_seconds": 1.0,
                "worker_threads": {
                    "environment": dict.fromkeys(native_ruler.THREAD_ENV, "1"),
                    "torch": 1,
                    "numba": 1,
                },
            }
            records.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "batches": [copy.deepcopy(batch) for _ in range(repeats)],
                }
            )
        cells[mode] = {
            "source_identity": ruler.source_identity(),
            "fixture_sha256": ruler.sha256_bytes(ruler.canonical(fixtures).encode()),
            "platform": {"label": label, "system": "Darwin" if label == "mac" else "Linux"},
            "min_batches_per_fixture": repeats,
            "workers": 1 if label == "mac" else 12,
            "completed_batches": repeats * 9,
            "records": records,
            "elapsed_seconds": 10.0,
            "aggregate_batches_per_second": repeats * 9 / 10,
        }
    return {
        "schema": native_ruler.SCHEMA,
        "elapsed_seconds": 20.0,
        "status": "CANARY_PASS" if canary else "PAIRED_PASS",
        "native_mode": "kernel",
        "source_identity": native_ruler.source_identity(),
        "fixtures": fixtures,
        "platform": cells["all"]["platform"],
        "cells": cells,
        "hardware": {
            "reported": {"hardware": {"kind": "L4"}},
            "vcpu_count": 12,
            "sha256": "a" * 64,
        },
        "mac_lease": {"id": "synthetic-only"},
        "k1_sha256": "b" * 64,
        "canary_sha256": None if canary else "c" * 64,
        "decision_identity": {"status": "PASS", "compared_batches": repeats * 9, "mismatches": []},
    }


def test_full_and_canary_receipt_contract() -> None:
    native_ruler.validate_paired_receipt(paired_receipt())
    native_ruler.validate_paired_receipt(paired_receipt("l4"))
    native_ruler.validate_paired_receipt(paired_receipt(canary=True), canary=True)
    with pytest.raises(ValueError, match="INVALID paired"):
        native_ruler.validate_paired_receipt(paired_receipt(canary=True))


def test_one_chosen_action_mismatch_fails_even_if_attestation_says_pass() -> None:
    value = paired_receipt()
    value["cells"]["all"]["records"][0]["batches"][0]["semantic"]["chosen"] = 1
    with pytest.raises(ValueError, match="chosen-action mismatch"):
        native_ruler.validate_paired_receipt(value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("kernel_rows", 0),
        ("stock_rows", 1),
        ("fallback_rows", 1),
        ("noncandidate_rows", 1),
        ("calls", 0),
    ],
)
def test_native_claim_requires_executed_rows_without_fallback(key: str, value: int) -> None:
    receipt = paired_receipt()
    batches = next(
        row["batches"]
        for row in receipt["cells"]["all"]["records"]
        if row["fixture_id"].startswith("frozen")
    )
    batches[0]["native_v8"][key] = value
    with pytest.raises(ValueError, match="native"):
        native_ruler.validate_paired_receipt(receipt)


def test_skipped_branch_cannot_claim_survival_and_counters_reconcile() -> None:
    receipt = paired_receipt()
    batch = receipt["cells"]["exact_stop"]["records"][0]["batches"][0]
    batch["semantic"]["outcomes"]["1"] = {"status": "NOT_EVALUATED", "survives": True}
    with pytest.raises(ValueError, match="NOT_EVALUATED"):
        native_ruler.validate_paired_receipt(receipt)
    batch["semantic"]["outcomes"]["1"] = {"status": "NOT_EVALUATED"}
    batch["semantic"]["branches_skipped"] = 1
    with pytest.raises(ValueError, match="branch execution"):
        native_ruler.validate_paired_receipt(receipt)


def test_missing_or_forged_batch_coverage_refuses() -> None:
    receipt = paired_receipt()
    receipt["cells"]["all"]["records"][0]["batches"].pop()
    with pytest.raises(ValueError, match="INCOMPLETE"):
        native_ruler.validate_paired_receipt(receipt)


def test_fixture_compatibility_is_an_exact_allowlist() -> None:
    source = ruler.source_identity()
    for name, hashes in native_ruler.LEGACY.items():
        source[name] = sorted(hashes)[0]
    native_ruler.validate_source(source)
    source["research/t3_rollout_labels_20261015/engine.py"] = "f" * 64
    with pytest.raises(ValueError, match="outside prospective"):
        native_ruler.validate_source(source)


@pytest.mark.parametrize("fail_on,expected_submits", [(1, 0), (2, 1)])
def test_lease_loss_blocks_admission_and_terminates_only_owned_children(
    monkeypatch: pytest.MonkeyPatch, tmp_path, fail_on: int, expected_submits: int
) -> None:
    """The real scheduler must check leases before submit and stop its own pool."""
    import concurrent.futures

    class Process:
        terminated = False

        def terminate(self) -> None:
            self.terminated = True

    class Pool:
        def __init__(self, **kwargs) -> None:
            self._processes = {1: Process()}
            self.submits = 0
            self.closed = False

        def submit(self, *args):
            self.submits += 1
            result = concurrent.futures.Future()
            result.set_result(None)
            return result

        def shutdown(self, **kwargs) -> None:
            self.closed = True

    pool = Pool()
    monkeypatch.setattr(native_ruler.concurrent.futures, "ProcessPoolExecutor", lambda **kw: pool)
    calls = 0

    def guard() -> None:
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise RuntimeError("foreign lease")

    receipt = paired_receipt(canary=True)
    with pytest.raises(RuntimeError, match="foreign lease"):
        native_ruler.run_pairs(
            receipt["fixtures"],
            label="mac",
            workers=1,
            repeats=1,
            guard=guard,
            out=tmp_path,
            max_seconds=60,
            canary=True,
        )
    assert pool.submits == expected_submits
    assert pool.closed
    assert all(process.terminated for process in pool._processes.values())


def test_survivor_must_execute_full_h300() -> None:
    receipt = paired_receipt()
    batch = receipt["cells"]["all"]["records"][0]["batches"][0]
    batch["semantic"]["outcomes"]["0"]["frames_executed"] = 1
    batch["semantic"]["rollout_branch_frames"] -= 299
    with pytest.raises(ValueError, match="survivor did not execute H300"):
        native_ruler.validate_paired_receipt(receipt)


def test_wall_cap_is_checked_before_any_worker_submission(monkeypatch, tmp_path) -> None:
    class Pool:
        _processes = {}
        submitted = False
        closed = False

        def submit(self, *args):
            self.submitted = True
            raise AssertionError("must refuse before submit")

        def shutdown(self, **kwargs) -> None:
            self.closed = True

    receipt = paired_receipt(canary=True)
    pool = Pool()
    ticks = iter([0.0, 61.0])
    monkeypatch.setattr(native_ruler.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(native_ruler.concurrent.futures, "ProcessPoolExecutor", lambda **kw: pool)
    with pytest.raises(RuntimeError, match="wall cap before submit"):
        native_ruler.run_pairs(
            receipt["fixtures"],
            label="mac",
            workers=1,
            repeats=1,
            guard=lambda: None,
            out=tmp_path,
            max_seconds=60,
            canary=True,
        )
    assert not pool.submitted
    assert pool.closed
