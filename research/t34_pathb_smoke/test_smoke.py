"""Targeted tests for excluded smoke accounting and six-thread transport."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from research.t34_pathb_smoke.parallel import BranchPool
from research.t34_pathb_smoke.run import summarize


def test_fixed_denominator_and_half_threshold() -> None:
    rows = [
        {
            "workers": workers,
            "group": "T3",
            "delivered_before_its_frame": i < n,
            "applied": {"intervention_kind": "rescue" if i % 2 else "delay"},
        }
        for workers, n in ((1, 37), (6, 38))
        for i in range(75)
    ]
    result = summarize(rows)
    assert result["workers"]["1"]["fraction"] == 37 / 75
    assert result["workers"]["6"]["fraction"] == 38 / 75
    assert result["stopping_rule_verdict"] == "STOP_OPERATOR02_AMEND_SECTION4_AND_RETIME"
    assert summarize(rows[:-1])["stopping_rule_verdict"] == "INCOMPLETE"


def test_first_winner_is_latched_and_helpers_cooperatively_drain() -> None:
    calls = []
    lock = threading.Lock()

    def evaluator(decisions, horizons, *, cancel_event, on_complete):
        for _ in range(decisions[0].delay):
            time.sleep(0.001)
            if cancel_event.is_set():
                raise TimeoutError("cooperative cancel")
        with lock:
            calls.append(decisions[0].frame)
        on_complete(0, {0: decisions[0].frame})
        return [{0: d.frame} for d in decisions]

    pool = BranchPool(6, evaluator)
    winners = []
    try:
        results = pool.evaluate(
            [SimpleNamespace(frame=i, delay=1 if i == 3 else 100) for i in range(6)],
            [1] * 6,
            on_complete=lambda i, outcomes: winners.append(i) or True,
        )
        assert winners == [3]
        assert calls == [3]
        assert results[3] == {0: 3}
    finally:
        pool.close()


def test_real_native_batch_serial_parallel_identity() -> None:
    from research.t3_rollout_labels_20261015.bank_engine import build_world
    from research.t34_pathb.batch_v2 import evaluate_decisions
    from research.t34_pathb.engine_v2 import _capture_and_step
    from research.t34_pathb.precheck_v2 import materialize_t3

    row = materialize_t3()[0]
    world = build_world(row["t3_source_arm"], row)
    decisions = [_capture_and_step(world) for _ in range(8)]
    expected = evaluate_decisions(decisions, [4] * 8)
    pool = BranchPool(6, evaluate_decisions)
    try:
        observed = pool.evaluate(decisions, [4] * 8)
        assert observed == expected
    finally:
        pool.close()


def test_invalid_worker_count() -> None:
    with pytest.raises(ValueError):
        BranchPool(7, lambda: None)


def test_later_on_time_execution_is_not_hidden_by_first_late_result() -> None:
    from research.t34_pathb_smoke.run import delivered_search

    first = {"status": "IMPROVING", "target_frame": 10, "action": 0, "finish_wall": 101.0}
    second = {"status": "IMPROVING", "target_frame": 48, "action": 2, "finish_wall": 101.5}
    applied = {"frame": 48, "executed_action": 2}
    assert delivered_search([first, second], applied, 0, 100.0) == second
    assert delivered_search([first], {"frame": 10, "executed_action": 0}, 0, 100.0) is None
