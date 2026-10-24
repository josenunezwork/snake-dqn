"""Synthetic clocks/state for every resource/protection admission boundary."""

from __future__ import annotations

import copy

import pytest

from research.compute.factory_companion_policy import (
    POLICY_SHA256,
    check_admission,
    update_factory_state,
)


def ready() -> dict:
    state = {
        "monitor": {"pid": 1, "start": "kernel-identity"},
        "heartbeat": 2000.0,
        "observed_since": 1900.0,
        "light_only_since": 1900.0,
        "history_hold_until": 1800.0,
        "samples": [],
        "duty": [],
        "registrations": {},
    }
    update_factory_state(
        state,
        "lease/attempt",
        False,
        {
            "watch6": {
                "baseline_ready": True,
                "fresh_window_ready": True,
                "window_full": True,
                "window_started_at": 1100.0,
                "median_ratio": 1.0,
            }
        },
        2000,
    )
    return state


def admit(state: dict, klass: str = "test", now: float = 2000) -> None:
    check_admission(state, klass, now, POLICY_SHA256, "lease/attempt")


def test_exact_thresholds_are_admitted() -> None:
    state = ready()
    state["duty"] = [{"at": 2000, "cpu": 210}]
    state["samples"] = [{"at": 2000, "total_seconds": 75, "light_seconds": 15, "rss": 8 * 1024**3}]
    state["light_only_since"] = state["observed_since"] = 1940
    admit(state)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("admission_closed", True),
        ("exclusive_arm", "strict"),
        ("heartbeat", 1989.999),
        ("heartbeat", 2001),
        ("heartbeat", float("nan")),
        ("monitor", None),
        ("unmanaged", {"123": "unknown"}),
        ("observation_failure", "failed-native-probe"),
        ("suspended_until", 2000.001),
        ("history_hold_until", 2000.001),
        ("light_only_since", 1940.001),
        ("observed_since", 1940.001),
    ],
)
def test_invalid_or_incomplete_observation_closes_admission(key: str, value: object) -> None:
    state = ready()
    state[key] = value
    with pytest.raises(ValueError):
        admit(state)


@pytest.mark.parametrize("key", ["history_hold_until", "light_only_since", "factory", "heartbeat"])
def test_missing_state_never_resets_budget_or_creates_admission(key: str) -> None:
    state = ready()
    del state[key]
    with pytest.raises(ValueError):
        admit(state)


@pytest.mark.parametrize(
    "change",
    [
        {"protected": True},
        {"identity": "another-attempt"},
        {"policy_sha256": "foreign"},
        {"streams": {}},
        {"streams": {"new-stream": {"baseline_ready": False, "fresh_window_ready": False}}},
        {"streams": {"watch6": {"baseline_ready": True, "fresh_window_ready": False}}},
    ],
)
def test_first_hour_resume_width_new_stream_and_w12_protect_test_lane(change: dict) -> None:
    state = ready()
    state["factory"].update(change)
    with pytest.raises(ValueError):
        admit(state)
    if "identity" not in change and "policy_sha256" not in change:
        admit(state, "light")


@pytest.mark.parametrize(
    "sample",
    [
        {"total_seconds": 75.000001, "light_seconds": 0, "rss": 0},
        {"total_seconds": 15.000001, "light_seconds": 15.000001, "rss": 0},
        {"total_seconds": 0, "light_seconds": 0, "rss": 8 * 1024**3 + 1},
    ],
)
def test_over_ceiling_closes_all_new_work(sample: dict) -> None:
    state = ready()
    state["samples"] = [{"at": 2000, **sample}]
    with pytest.raises(ValueError):
        admit(state, "light")


def test_rolling_duty_and_all_live_reservations_are_shared() -> None:
    state = ready()
    state["duty"] = [{"at": 2000, "cpu": 120.001}]
    state["registrations"] = {"other-worktree": {"class": "test", "status": "starting"}}
    with pytest.raises(ValueError, match="duty"):
        admit(state)
    state["duty"][0]["cpu"] = 120
    admit(state)  # Exactly 120 + 90 + 90 == 300, slot lock separately refuses overlap.
    state["duty"] = [{"at": 199.999, "cpu": 999}]
    state["registrations"] = {}
    admit(state)


def test_reservation_survives_stopped_or_dead_pid_until_reap_and_cleanup_proof() -> None:
    state = ready()
    state["duty"] = [{"at": 2000, "cpu": 121}]
    record = {"class": "test", "status": "stopped", "wait4_reaped": True}
    state["registrations"] = {"retained": record}
    with pytest.raises(ValueError, match="duty"):
        admit(state)
    record["descendants_gone"] = True
    admit(state)


def test_warning_requires_fifteen_minutes_and_fresh_entire_clean_window() -> None:
    state = ready()
    state["warning"] = {"revoked_at": 1100}
    admit(state)
    original = copy.deepcopy(state)
    for patch in (
        {"window_started_at": 1099.999},
        {"median_ratio": 1.150000001},
        {"window_full": False},
    ):
        state = copy.deepcopy(original)
        state["factory"]["streams"]["watch6"].update(patch)
        with pytest.raises(ValueError, match="fresh full clean"):
            admit(state)
        admit(state, "light")
    state = copy.deepcopy(original)
    state["warning"]["revoked_at"] = 1100.001
    with pytest.raises(ValueError, match="suspension"):
        admit(state)


def test_factory_control_update_does_not_rewrite_baseline_or_accounting() -> None:
    state = ready()
    state["baseline"] = {"median": 20}
    state["worker_seconds_denominator"] = 57600 * 12
    original = copy.deepcopy(state)
    update_factory_state(state, "lease/new-attempt", True, {}, 2001)
    assert state["baseline"] == original["baseline"]
    assert state["worker_seconds_denominator"] == original["worker_seconds_denominator"]
    assert state["duty"] == original["duty"]
