"""Bounded provider-lag rules for broker-owned Colab assignment transitions."""

from __future__ import annotations

import contextlib
import io

import pytest

from research.colab_cli import accounting


@contextlib.contextmanager
def _warns(code: str):
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
        yield
    assert '"WARNING":' in captured.getvalue()
    assert code in captured.getvalue()


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


@pytest.mark.parametrize("manifest_prices_other", [False, True])
@pytest.mark.parametrize("provider_departed", [False, True])
def test_broker_departed_but_provider_settled_endpoint_is_bounded(
    manifest_prices_other: bool,
    provider_departed: bool,
) -> None:
    tracker = _tracker(other=True)
    declared = {"other": 1.0} if manifest_prices_other else None
    usage = _usage(2.0 if provider_departed else 3.0, 1 if provider_departed else 2)
    for poll in (1, 2):
        spend = tracker.check(
            usage,
            now=float(poll),
            sessions={"ours": "our-lane", "other": "other-lane"},
            active_records=[],
            budget_other_assignments=declared,
        )
        assert spend["provider_transition_status"] == "WARNING_BROKER_TRANSITION_LAG"
        assert spend["provider_transition_mismatch_polls"] == poll
        assert tracker.integrated_other_units == pytest.approx(poll / 3600.0)
        assert tracker.current_other_rates == {"other": (1.0, "broker.active")}
    with pytest.raises(accounting.AccountingError, match="unaccounted active assignment.*other"):
        tracker.check(
            usage,
            now=3.0,
            sessions={"ours": "our-lane", "other": "other-lane"},
            active_records=[],
            budget_other_assignments=declared,
        )


@pytest.mark.parametrize("warning_polls", [1, 2])
@pytest.mark.parametrize("relisted", [False, True])
def test_broker_carryover_clears_when_endpoint_disappears_or_is_relisted(
    warning_polls: int, relisted: bool
) -> None:
    tracker = _tracker(other=True)
    sessions = {"ours": "our-lane", "other": "other-lane"}
    for poll in range(1, warning_polls + 1):
        spend = tracker.check(_usage(3.0, 2), now=float(poll), sessions=sessions)
        assert spend["provider_transition_mismatch_polls"] == poll
    if not relisted:
        del sessions["other"]
    for poll in range(warning_polls + 1, warning_polls + 4):
        spend = tracker.check(
            _usage(3.0 if relisted else 2.0, 2 if relisted else 1),
            now=float(poll),
            sessions=sessions,
            active_records=[{"endpoint": "other", "rate_units_per_hour": 1.0}] if relisted else [],
        )
        assert spend["provider_transition_status"] == "OK"
        assert spend["provider_transition_mismatch_polls"] == 0
    if relisted:
        spend = tracker.check(_usage(3.0, 2), now=float(poll + 1), sessions=sessions)
        assert spend["provider_transition_mismatch_polls"] == 1


@pytest.mark.parametrize("declared", [None, {"other": 2.0}])
def test_broker_carryover_does_not_grace_unexplained_rate_drift(
    declared: dict[str, float] | None,
) -> None:
    tracker = _tracker(other=True)
    with _warns("PROVIDER_RATE_DRIFT"):
        tracker.check(
            _usage(4.0, 2),
            now=1.0,
            sessions={"ours": "our-lane", "other": "other-lane"},
            active_records=[],
            budget_other_assignments=declared,
        )


def test_broker_carryover_and_provider_lag_share_one_warning_window() -> None:
    tracker = _tracker(other=True)
    spend = tracker.check(
        _usage(3.0, 2), now=1.0, sessions={"ours": "our-lane", "other": "other-lane"}
    )
    assert spend["provider_transition_mismatch_polls"] == 1
    spend = tracker.check(_usage(3.0, 2), now=2.0, sessions={"ours": "our-lane"})
    assert spend["provider_transition_mismatch_polls"] == 2
    with pytest.raises(accounting.AccountingError, match="count disagrees"):
        tracker.check(_usage(3.0, 2), now=3.0, sessions={"ours": "our-lane"})


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
        tracker.check(
            _usage(2.0, accounting.ACCOUNT_ASSIGNMENT_CAP + 1),
            now=1.0,
            sessions={"ours": "our-lane"},
            active_records=[],
        )

    tracker = _tracker()
    with _warns("PROVIDER_RATE_DRIFT"):
        tracker.check(_usage(4.0, 1), now=1.0, sessions={"ours": "our-lane"}, active_records=[])


def test_stable_broker_endpoint_rate_change_is_not_hidden_by_an_arrival() -> None:
    tracker = _tracker(other=True)
    with _warns("PROVIDER_RATE_DRIFT"):
        tracker.check(
            _usage(4.0, 3),
            now=1.0,
            sessions={"ours": "our-lane", "other": "other-lane", "new": "new-lane"},
            active_records=[
                {"endpoint": "other", "rate_units_per_hour": 2.0},
                {"endpoint": "new", "rate_units_per_hour": 1.0},
            ],
        )


@pytest.mark.parametrize("rate", [0.0, 7.0, 999.0])
@pytest.mark.parametrize("settles", [False, True])
def test_count_lag_grace_never_depends_on_rate(rate, settles, capsys):
    tracker = _tracker()
    other = [{"endpoint": "arriving", "rate_units_per_hour": 1.0}]
    sessions = {"ours": "our-lane", "arriving": "other-lane"}
    for now in (1.0, 2.0):
        spend = tracker.check(_usage(rate, 1), now=now, sessions=sessions, active_records=other)
        assert spend["provider_transition_mismatch_polls"] == int(now)
    if settles:
        spend = tracker.check(_usage(rate, 2), now=3, sessions=sessions, active_records=other)
        assert spend["provider_transition_status"] == "OK"
    else:
        with pytest.raises(accounting.AccountingError, match="count disagrees"):
            tracker.check(_usage(rate, 1), now=3, sessions=sessions, active_records=other)
    assert '"WARNING": "PROVIDER_RATE_DRIFT"' in capsys.readouterr().err


def test_failed_rate_measurement_without_price_keeps_numeric_receipt():
    tracker = accounting.admit_account({}, _usage(0, 0), {}, now=0)
    tracker.confirm_own_assignment({"ours": "our-lane"}, _usage(0, 1), "ours", rate_confirmed=False)
    assert tracker.receipt()["own_rate_units_per_hour"] == 0.0
    assert tracker.receipt()["rate_measurement_unavailable"] is True
