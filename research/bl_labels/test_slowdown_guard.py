"""Timestamp-only D256 replay and frozen operational-guard contract tests."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from research.bl_labels.slowdown_guard import SCHEMA, SlowdownGuard

FIXTURE = Path(__file__).parent / "fixtures/d256_slowdown_timestamps.json"


def _attempts() -> list[dict]:
    return json.loads(FIXTURE.read_text())["attempts"]


def _frozen(median: float = 100.0, **extra: dict) -> dict:
    return {
        "schema": SCHEMA,
        "stream": "watch6",
        "width_wall_seconds": {"6": 3600.0},
        "baselines": {"6": {"median": median, "sample_count": 136, "frozen": True}, **extra},
        "candidates": {},
    }


def _duration_sequence(attempt: dict) -> list[float]:
    starts, durations = {}, []
    for kind, parent, now in attempt["events"]:
        if kind == "admit":
            starts[parent] = now
        elif kind == "life_boundary":
            durations.append(now - starts[parent])
            starts[parent] = now
    return durations


def _lives(guard, durations, start=0.0):
    now, events = start, []
    guard.admit("parent", False, now)
    for index, duration in enumerate(durations, 1):
        now += duration
        event = guard.boundary("parent", now)
        if event is not None:
            events.append((index, event))
    return events, now


def test_actual_attempt_one_timeline_no_trip_and_armed_ratio_below_104() -> None:
    attempt = _attempts()[0]
    guard = SlowdownGuard("watch6", 6, attempt["events"][0][2])
    ratios, trips = [], []
    for kind, parent, now in attempt["events"]:
        if kind == "admit":
            guard.admit(parent, False, now)
        elif kind == "life_boundary":
            event = guard.boundary(parent, now)
            if event:
                trips.append(event)
            baseline = guard.snapshot(now)["baselines"].get("6")
            if baseline and len(guard._window) == 24:
                ratios.append(statistics.median(guard._window) / baseline["median"])
        else:
            guard.complete(parent, now)
    assert trips == []
    assert ratios and max(ratios) <= 1.04
    baseline = guard.snapshot(now)["baselines"]["6"]
    assert baseline["sample_count"] == 136
    assert baseline["median"] == pytest.approx(
        statistics.median(attempt["first_hour_life_seconds"]), abs=0.0001
    )


def test_actual_attempt_two_prior_state_excludes_every_replay_first_life() -> None:
    first, attempt = _attempts()
    baseline = statistics.median(first["first_hour_life_seconds"])
    guard = SlowdownGuard("watch6", 6, attempt["events"][0][2], persisted=_frozen(baseline))
    prior, first_boundary, excluded, trips = set(attempt["prior_life_parents"]), set(), set(), []
    eligible = 0
    for kind, parent, now in attempt["events"]:
        if kind == "admit":
            guard.admit(parent, parent in prior, now)
        elif kind == "life_boundary":
            if parent in prior and parent not in first_boundary:
                excluded.add(parent)
            else:
                eligible += 1
            first_boundary.add(parent)
            event = guard.boundary(parent, now)
            if event:
                trips.append(event)
            assert len(guard._window) == min(24, eligible)
        else:
            guard.complete(parent, now)
    assert excluded == {p for k, p, _ in attempt["events"] if k == "life_boundary" and p in prior}
    assert len(excluded) == 4
    assert eligible == 5 < 24 and trips == []


def test_actual_attempt_one_lives_scaled_135_trip_within_k_120_never_trips() -> None:
    attempt = _attempts()[0]
    baseline = statistics.median(attempt["first_hour_life_seconds"])
    durations = _duration_sequence(attempt)
    # Freeze the original baseline; onset is the sixth K-boundary block.
    onset, k = 6 * 24, 24
    for scale in (1.35, 1.20):
        guard = SlowdownGuard("watch6", 6, 0, persisted=_frozen(baseline))
        values = durations[:onset] + [scale * d for d in durations[onset:]]
        events, _now = _lives(guard, values)
        if scale == 1.35:
            assert events and onset < events[0][0] <= onset + k
            assert events[0][1]["statistic"] == "window_median"
        else:
            assert events == []


def test_width_switches_clear_window_exclude_inflight_and_select_width_baseline() -> None:
    saved = _frozen(**{"12": {"median": 150.0, "sample_count": 60, "frozen": True}})
    guard = SlowdownGuard("watch6", 6, 0, persisted=saved)
    events, now = _lives(guard, [129.0] * 23)
    assert events == []
    guard.switch(12, now + 1)
    assert guard.boundary("parent", now + 200) is None
    assert len(guard._window) == 0
    for i in range(48):
        now += 200
        event = guard.boundary("parent", now + 200)
        assert (event is not None) == (i == 47)
    assert event["baseline_median"] == 150 and event["window_k"] == 48
    guard.switch(6, now + 201)
    assert guard.boundary("parent", now + 300) is None
    assert len(guard._window) == 0
    for i in range(24):
        event = guard.boundary("parent", now + 300 + (i + 1) * 131)
        assert (event is not None) == (i == 23)
    assert event["baseline_median"] == 100 and event["window_k"] == 24


def test_baseline_sanity_1911_bound_and_existing_width_reference() -> None:
    guard = SlowdownGuard("watch6", 12, 0, persisted=_frozen())
    events, _ = _lives(guard, [191.10001] * 60)
    assert events[-1][1]["stop_reason"] == "slowdown baseline sanity"
    assert "12" not in guard.snapshot(12000)["baselines"]
    # Both rules apply at W12, including an existing same-pin W12 reference.
    guard = SlowdownGuard("watch6", 12, 0, persisted=_frozen(), reference_baselines={12: 100})
    events, _ = _lives(guard, [140.0] * 60)
    assert events[-1][1]["stop_reason"] == "slowdown baseline sanity"


def test_resume_frozen_baseline_window_fresh_and_queued_stopped_lives_excluded() -> None:
    guard = SlowdownGuard("watch6", 6, 0, persisted=_frozen())
    events, now = _lives(guard, [129.0] * 24)
    assert events == []
    saved = guard.snapshot(now)
    resumed = SlowdownGuard("watch6", 6, 0, persisted=saved)
    events, now = _lives(resumed, [135.0] * 23)
    assert events == [] and len(resumed._window) == 23
    assert resumed.boundary("parent", now + 135)["window_k"] == 24
    # Stop drains all already-queued lives without observing any of them.
    assert resumed.boundary("parent", now + 270, stopped=True) is None
    assert resumed.boundary("parent", now + 405, stopped=True) is None
    assert len(resumed._window) == 24
    assert resumed.snapshot(now + 405)["baselines"] == saved["baselines"]


def test_minimum_sixty_short_lives_and_paused_stream_wall_clock() -> None:
    guard = SlowdownGuard("watch6", 6, 0)
    guard.tick(10000)  # Waiting for this stream is not stream execution time.
    assert guard.snapshot(10000)["width_wall_seconds"]["6"] == 0
    guard.admit("parent", False, 10000)
    for i in range(59):
        assert guard.boundary("parent", 10000 + i * 0.01) is None
    assert guard.tick(13600) is None
    assert "6" not in guard.snapshot(13600)["baselines"]
    assert guard.boundary("parent", 13600.01) is None
    baseline = guard.snapshot(13600.01)["baselines"]["6"]
    assert baseline["sample_count"] == 60
    guard.complete("parent", 13600.02)
    assert guard.snapshot(20000)["width_wall_seconds"]["6"] == pytest.approx(3600.02)


def test_no_single_life_trip_and_threshold_equality_is_allowed() -> None:
    guard = SlowdownGuard("watch6", 6, 0, persisted=_frozen())
    events, _ = _lives(guard, [100] * 23 + [1_000_000])
    assert events == []
    guard = SlowdownGuard("watch6", 6, 0, persisted=_frozen())
    assert _lives(guard, [130] * 48)[0] == []
