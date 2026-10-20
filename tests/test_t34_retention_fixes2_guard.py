"""Second FIX-FIRST host-guard contracts using only synthetic samples."""

from __future__ import annotations

import threading

import pytest

from research.t34_pathb.host_guard_v2 import HostContentionError, HostGuard

POLICY = {
    "max_load_1m_excess": 3.0,
    "other_process_cpu_percent": 50.0,
    "other_process_cpu_sustained_seconds": 60,
    "persistence_seconds": 120,
    "check_interval_seconds": 30,
    "own_cpu_worker_allowance": 2,
    "known_heavy_process_patterns": ["research/", "runner"],
    "max_contended_fraction": 0.02,
}


def _sample(*, contended: bool = False, reasons: list[str] | None = None) -> dict:
    return {"contended": contended, "reasons": reasons or []}


def test_tick_checks_use_cached_sample_without_sampling_or_waiting() -> None:
    now = [0.0]
    calls = [0]

    def sample() -> dict:
        calls[0] += 1
        return _sample()

    guard = HostGuard(POLICY, clock=lambda: now[0], sample=sample)
    try:
        guard.start()
        assert calls == [1]
        for tick in range(1, 6):
            now[0] = float(tick)
            assert guard.check()["contended"] is False
        assert calls == [1]
        assert guard.summary(validate=False)["tick_count"] == 5
    finally:
        guard.close()


def test_contention_fraction_is_accounted_at_ticks_and_reportable_on_failure() -> None:
    now = [0.0]
    guard = HostGuard(
        POLICY, clock=lambda: now[0], sample=lambda: _sample(contended=True, reasons=["load"])
    )
    try:
        guard.start()
        guard.check()
        now[0] = 10.0
        guard.check()
        diagnostic = guard.summary(validate=False)
        assert diagnostic["contended_seconds"] == 10.0
        assert diagnostic["contended_fraction"] == 1.0
        assert diagnostic["status"] == "INCONCLUSIVE"
        with pytest.raises(HostContentionError) as raised:
            guard.summary()
        assert raised.value.summary == diagnostic
    finally:
        guard.close()


def test_prior_timing_accounting_is_included_in_final_budget() -> None:
    now = [0.0]
    guard = HostGuard(
        POLICY,
        clock=lambda: now[0],
        sample=lambda: _sample(),
        prior_elapsed_seconds=100.0,
        prior_contended_seconds=2.0,
    )
    try:
        guard.start()
        assert guard.summary()["contended_fraction"] == 0.02
    finally:
        guard.close()


def test_power_pause_happens_between_rows_and_discards_assists_once() -> None:
    now = [0.0]
    samples = iter(
        [
            _sample(contended=True, reasons=["ac_power_lost"]),
            _sample(),
        ]
    )
    guard = HostGuard(
        POLICY,
        clock=lambda: now[0],
        sample=lambda: next(samples),
        prior_elapsed_seconds=1_000.0,
    )
    callback_called = threading.Event()

    def publish_recovery() -> None:
        callback_called.wait(1.0)
        now[0] = 10.0
        guard._capture()  # The sampler's publication path; no gameplay tick waits for it.

    try:
        guard.start()
        publisher = threading.Thread(target=publish_recovery)
        publisher.start()
        record = guard.between_rows(on_pause=callback_called.set)
        publisher.join(timeout=1.0)
        assert record["contended"] is False
        assert callback_called.is_set()
        summary = guard.summary(validate=False)
        assert summary["tick_count"] == 0
        assert summary["contended_seconds"] == 10.0
    finally:
        guard.close()
