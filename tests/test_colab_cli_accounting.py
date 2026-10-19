"""Pure per-session accounting contracts for the Colab broker."""

from __future__ import annotations

import pytest

from research.colab_cli import accounting


def _budget(**extra: object) -> dict[str, object]:
    return {
        "rate_units_per_hour": 2.0,
        "unit_cap": 1.0,
        **extra,
    }


def _usage(*, balance: float = 100.0, rate: float = 3.0, assignments: int = 2) -> dict[str, float]:
    return {"balance": balance, "rate": rate, "assignments": assignments}


def test_admission_requires_every_existing_endpoint_to_be_priced() -> None:
    sessions = {"notebook-endpoint": "q1", "other-endpoint": "other"}
    with pytest.raises(accounting.AccountingError, match="unaccounted active assignment"):
        accounting.admit_account(
            sessions,
            _usage(),
            _budget(other_assignments={"notebook-endpoint": 1.5}),
            now=10,
        )


def test_admission_crosschecks_provider_baseline_against_priced_assignments() -> None:
    with pytest.raises(accounting.AccountingError, match="baseline rate"):
        accounting.admit_account(
            {"notebook-endpoint": "q1", "other-endpoint": "other"},
            _usage(rate=3.0),
            _budget(other_assignments={"notebook-endpoint": 1.0, "other-endpoint": 1.0}),
            now=10,
        )


def test_broker_records_account_for_concurrent_lanes_and_are_evidenced() -> None:
    tracker = accounting.admit_account(
        {"notebook-endpoint": "q1", "broker-endpoint": "other-lane"},
        _usage(rate=3.0),
        _budget(other_assignments={"notebook-endpoint": {"rate_units_per_hour": 1.5}}),
        active_records=[{"endpoint": "broker-endpoint", "rate_units_per_hour": 1.5}],
        now=10,
    )

    evidence = tracker.accounted_assignments()
    assert evidence["baseline_rate_units_per_hour"] == 3.0
    assert evidence["unaccounted_endpoints"] == []
    assert evidence["assignments"] == [
        {
            "endpoint": "broker-endpoint",
            "session": "other-lane",
            "rate_units_per_hour": 1.5,
            "source": "broker.active",
        },
        {
            "endpoint": "notebook-endpoint",
            "session": "q1",
            "rate_units_per_hour": 1.5,
            "source": "manifest.other_assignments",
        },
    ]


def test_confirm_requires_one_known_new_endpoint_and_expected_rate() -> None:
    tracker = accounting.admit_account(
        {"notebook-endpoint": "q1"},
        _usage(rate=1.0, assignments=1),
        _budget(other_assignments={"notebook-endpoint": 1.0}),
        now=10,
    )
    evidence = tracker.confirm_own_assignment(
        {"notebook-endpoint": "q1", "our-endpoint": "snake-lane"},
        _usage(rate=3.0, assignments=2),
        "our-endpoint",
    )
    assert evidence["allocated_endpoint"] == "our-endpoint"
    assert evidence["allocated_rate_units_per_hour"] == 2.0

    unconfirmed = accounting.admit_account(
        {"notebook-endpoint": "q1"},
        _usage(rate=1.0, assignments=1),
        _budget(other_assignments={"notebook-endpoint": 1.0}),
        now=10,
    )
    with pytest.raises(accounting.AccountingError, match="does not match"):
        unconfirmed.confirm_own_assignment(
            {"notebook-endpoint": "q1", "our-endpoint": "snake-lane"},
            _usage(rate=2.5, assignments=2),
            "our-endpoint",
        )


def test_balance_crosscheck_subtracts_measured_baseline_before_attribution() -> None:
    tracker = accounting.admit_account(
        {"notebook-endpoint": "q1"},
        _usage(balance=100.0, rate=3.0, assignments=1),
        _budget(unit_cap=5.0, other_assignments={"notebook-endpoint": 3.0}),
        now=0,
    )
    tracker.confirm_own_assignment(
        {"notebook-endpoint": "q1", "our-endpoint": "snake-lane"},
        _usage(balance=100.0, rate=5.0, assignments=2),
        "our-endpoint",
    )

    # 3 account units vanished over an hour.  The 3-unit notebook baseline
    # explains all of it, so the new session has no attributable debit.
    crosscheck = tracker.check(_usage(balance=97.0, rate=5.0, assignments=2), now=3600)
    assert crosscheck["own_elapsed_units"] == pytest.approx(2.0)
    assert crosscheck["attributable_account_debit_units"] == pytest.approx(0.0)


def test_own_wall_clock_cap_remains_authoritative_with_other_spend() -> None:
    tracker = accounting.admit_account(
        {"notebook-endpoint": "q1"},
        _usage(balance=100.0, rate=7.0, assignments=1),
        _budget(other_assignments={"notebook-endpoint": 7.0}),
        now=0,
    )
    tracker.confirm_own_assignment(
        {"notebook-endpoint": "q1", "our-endpoint": "snake-lane"},
        _usage(balance=100.0, rate=9.0, assignments=2),
        "our-endpoint",
    )
    with pytest.raises(accounting.AccountingError, match="priced session unit cap"):
        tracker.check(_usage(balance=91.0, rate=9.0, assignments=2), now=1800)


def test_reconcile_accounts_for_another_broker_lane_starting_mid_session() -> None:
    tracker = accounting.admit_account(
        {"notebook-endpoint": "q1"},
        _usage(balance=100.0, rate=1.0, assignments=1),
        _budget(unit_cap=5.0, other_assignments={"notebook-endpoint": 1.0}),
        now=0,
    )
    tracker.confirm_own_assignment(
        {"notebook-endpoint": "q1", "our-endpoint": "snake-lane"},
        _usage(balance=100.0, rate=3.0, assignments=2),
        "our-endpoint",
    )
    # The second lane was absent during the first half-hour, then appears at
    # this measured poll boundary and is charged only in later intervals.
    tracker.check(
        _usage(balance=98.5, rate=4.5, assignments=3),
        now=1800,
        sessions={
            "notebook-endpoint": "q1",
            "our-endpoint": "snake-lane",
            "broker-endpoint": "other-lane",
        },
        budget_other_assignments={"notebook-endpoint": 1.0},
        active_records=[{"endpoint": "broker-endpoint", "rate_units_per_hour": 1.5}],
    )
    crosscheck = tracker.check(
        _usage(balance=96.25, rate=4.5, assignments=3),
        now=3600,
        sessions={
            "notebook-endpoint": "q1",
            "our-endpoint": "snake-lane",
            "broker-endpoint": "other-lane",
        },
        budget_other_assignments={"notebook-endpoint": 1.0},
        active_records=[{"endpoint": "broker-endpoint", "rate_units_per_hour": 1.5}],
    )
    assert crosscheck["baseline_integrated_units"] == pytest.approx(1.75)
    assert crosscheck["attributable_account_debit_units"] == pytest.approx(2.0)


def test_reconcile_refuses_a_new_unaccounted_endpoint() -> None:
    tracker = accounting.admit_account(
        {}, _usage(rate=0.0, assignments=0), _budget(unit_cap=5.0), now=0
    )
    tracker.confirm_own_assignment(
        {"our-endpoint": "snake-lane"}, _usage(rate=2.0, assignments=1), "our-endpoint"
    )
    with pytest.raises(accounting.AccountingError, match="unaccounted active assignment"):
        tracker.check(
            _usage(rate=3.0, assignments=2),
            now=1,
            sessions={"our-endpoint": "snake-lane", "unknown": "new-lane"},
        )


def test_poll_refuses_provider_rate_drift_not_explained_by_priced_assignments() -> None:
    tracker = accounting.admit_account(
        {}, _usage(rate=0.0, assignments=0), _budget(unit_cap=5.0), now=0
    )
    tracker.confirm_own_assignment(
        {"our-endpoint": "snake-lane"}, _usage(rate=2.0, assignments=1), "our-endpoint"
    )
    with pytest.raises(accounting.AccountingError, match="rate drift"):
        tracker.check(
            _usage(rate=20.0, assignments=1),
            now=60,
            sessions={"our-endpoint": "snake-lane"},
        )


def test_batched_balance_debit_becomes_a_loud_crosscheck_warning_not_raw_cap() -> None:
    tracker = accounting.admit_account(
        {}, _usage(balance=100.0, rate=0.0, assignments=0), _budget(unit_cap=5.0), now=0
    )
    tracker.confirm_own_assignment(
        {"our-endpoint": "snake-lane"},
        _usage(balance=100.0, rate=2.0, assignments=1),
        "our-endpoint",
    )
    crosscheck = tracker.check(
        _usage(balance=95.0, rate=2.0, assignments=1),
        now=60,
        sessions={"our-endpoint": "snake-lane"},
    )
    assert crosscheck["own_elapsed_units"] == pytest.approx(2.0 / 60.0)
    assert crosscheck["balance_crosscheck_status"] == "WARNING_UNEXPLAINED_DEBIT"
    assert (
        crosscheck["balance_discrepancy_units"] > crosscheck["balance_discrepancy_tolerance_units"]
    )
