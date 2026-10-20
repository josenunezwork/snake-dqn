"""Bounded provider-lag rules for broker-owned Colab assignment transitions."""

from __future__ import annotations

import pytest

from research.colab_cli import accounting


def _usage(rate: float, assignments: int) -> dict[str, float]:
    return {"balance": 100.0, "rate": rate, "assignments": assignments}


def _tracker(*, other: bool = False) -> accounting.AccountTracker:
    sessions = {"other": "other-lane"} if other else {}
    tracker = accounting.admit_account(
        sessions,
        _usage(1.0 if other else 0.0, 1 if other else 0),
        {"rate_units_per_hour": 2.0, "unit_cap": 5.0},
        active_records=[{"endpoint": "other", "rate_units_per_hour": 1.0}] if other else [],
        now=0.0,
    )
    confirmed_sessions = {**sessions, "ours": "our-lane"}
    tracker.confirm_own_assignment(
        confirmed_sessions,
        _usage(3.0 if other else 2.0, 2 if other else 1),
        "ours",
    )
    return tracker


def test_delayed_broker_departure_warns_twice_then_fails_closed() -> None:
    tracker = _tracker(other=True)
    for now in (1.0, 2.0):
        spend = tracker.check(
            _usage(3.0, 2),
            now=now,
            sessions={"ours": "our-lane"},
            active_records=[],
        )
        assert spend["provider_transition_status"] == "WARNING_BROKER_TRANSITION_LAG"
        assert spend["provider_transition_mismatch_polls"] == int(now)
    with pytest.raises(accounting.AccountingError, match="count disagrees"):
        tracker.check(
            _usage(3.0, 2),
            now=3.0,
            sessions={"ours": "our-lane"},
            active_records=[],
        )


def test_delayed_departure_preserves_broker_provenance_when_manifest_also_prices_it() -> None:
    tracker = accounting.admit_account(
        {"other": "other-lane"},
        _usage(1.0, 1),
        {
            "rate_units_per_hour": 2.0,
            "unit_cap": 5.0,
            "other_assignments": {"other": 1.0},
        },
        active_records=[{"endpoint": "other", "rate_units_per_hour": 1.0}],
        now=0.0,
    )
    tracker.confirm_own_assignment(
        {"other": "other-lane", "ours": "our-lane"}, _usage(3.0, 2), "ours"
    )
    spend = tracker.check(
        _usage(3.0, 2),
        now=1.0,
        sessions={"ours": "our-lane"},
        active_records=[],
        budget_other_assignments={"other": 1.0},
    )
    assert spend["provider_transition_status"] == "WARNING_BROKER_TRANSITION_LAG"


def test_delayed_broker_arrival_warns_then_resets_when_provider_settles() -> None:
    tracker = _tracker()
    sessions = {"ours": "our-lane", "other": "other-lane"}
    active = [{"endpoint": "other", "rate_units_per_hour": 1.0}]
    delayed = tracker.check(_usage(2.0, 1), now=1.0, sessions=sessions, active_records=active)
    assert delayed["provider_transition_mismatch_polls"] == 1
    settled = tracker.check(_usage(3.0, 2), now=2.0, sessions=sessions, active_records=active)
    assert settled["provider_transition_status"] == "OK"
    assert settled["provider_transition_mismatch_polls"] == 0


def test_provider_usage_can_lead_a_stale_session_read_for_broker_membership() -> None:
    arrival = _tracker()
    spend = arrival.check(
        _usage(3.0, 2),
        now=1.0,
        sessions={"ours": "our-lane"},
        active_records=[{"endpoint": "other", "rate_units_per_hour": 1.0}],
    )
    assert spend["provider_transition_status"] == "WARNING_BROKER_TRANSITION_LAG"

    departure = _tracker(other=True)
    spend = departure.check(
        _usage(2.0, 1),
        now=1.0,
        sessions={"ours": "our-lane", "other": "other-lane"},
        active_records=[],
    )
    assert spend["provider_transition_status"] == "WARNING_BROKER_TRANSITION_LAG"


def test_unknown_endpoint_is_never_graced_as_a_transition() -> None:
    tracker = _tracker()
    with pytest.raises(accounting.AccountingError, match="unaccounted active assignment"):
        tracker.check(
            _usage(3.0, 2),
            now=1.0,
            sessions={"ours": "our-lane", "unknown": "foreign"},
            active_records=[],
        )


def test_cap_and_rate_envelope_violations_refuse_without_transition_grace() -> None:
    tracker = _tracker()
    with pytest.raises(accounting.AccountingError, match="cap exceeded"):
        tracker.check(_usage(2.0, 4), now=1.0, sessions={"ours": "our-lane"}, active_records=[])

    tracker = _tracker()
    with pytest.raises(accounting.AccountingError, match="rate drift"):
        tracker.check(_usage(4.0, 1), now=1.0, sessions={"ours": "our-lane"}, active_records=[])


def test_stable_broker_endpoint_rate_change_is_not_hidden_by_an_arrival() -> None:
    tracker = _tracker(other=True)
    with pytest.raises(accounting.AccountingError, match="rate drift"):
        tracker.check(
            _usage(4.0, 3),
            now=1.0,
            sessions={"ours": "our-lane", "other": "other-lane", "new": "new-lane"},
            active_records=[
                {"endpoint": "other", "rate_units_per_hour": 2.0},
                {"endpoint": "new", "rate_units_per_hour": 1.0},
            ],
        )
