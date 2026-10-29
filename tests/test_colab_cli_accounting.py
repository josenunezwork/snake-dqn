"""Pure per-session accounting contracts for the Colab broker."""

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
    with _warns("BASELINE_RATE_MISMATCH"):
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
    with _warns("OBSERVED_RATE_ABOVE_PRICED_BOUND"):
        unconfirmed.confirm_own_assignment(
            {"notebook-endpoint": "q1", "our-endpoint": "snake-lane"},
            _usage(rate=3.5, assignments=2),
            "our-endpoint",
        )


def test_observed_rate_and_unit_thresholds_warn_and_retain_measurements() -> None:
    tracker = accounting.admit_account({}, _usage(rate=0, assignments=0), _budget(), now=0)
    tracker.confirm_own_assignment({"own": "lane"}, _usage(rate=1.54, assignments=1), "own")
    result = tracker.check(_usage(rate=1.54, assignments=1), now=60, sessions={"own": "lane"})
    assert result["own_elapsed_units"] == pytest.approx(2 / 60)
    assert result["observed_elapsed_units"] == pytest.approx(1.54 / 60)
    assert tracker.receipt()["priced_rate_units_per_hour"] == 2
    assert tracker.receipt()["observed_rate_units_per_hour"] == 1.54
    with _warns("PROVIDER_RATE_DRIFT"):
        tracker.check(_usage(rate=2, assignments=1), now=61, sessions={"own": "lane"})
    with _warns("SESSION_UNIT_CAP_REACHED"):
        tracker.check(_usage(rate=1.54, assignments=1), now=1800)


def test_other_broker_lane_uses_observed_rate_instead_of_its_priced_ceiling() -> None:
    tracker = accounting.admit_account(
        {"other": "lane"},
        _usage(rate=1.54, assignments=1),
        _budget(),
        active_records=[
            {
                "endpoint": "other",
                "rate_units_per_hour": 2,
                "priced_rate_units_per_hour": 2,
                "observed_rate_units_per_hour": 1.54,
            }
        ],
    )
    assert tracker.accounted_assignments()["assignments"][0]["rate_units_per_hour"] == 1.54
    with _warns("RATE_UNAVAILABLE"):
        accounting.admit_account(
            {"other": "lane"},
            _usage(rate=2, assignments=1),
            _budget(),
            active_records=[{"endpoint": "other", "priced_rate_units_per_hour": 2}],
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


def test_unit_threshold_warns_with_other_spend() -> None:
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
    with _warns("SESSION_UNIT_CAP_REACHED"):
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


def test_poll_warns_provider_rate_drift_and_continues() -> None:
    tracker = accounting.admit_account(
        {}, _usage(rate=0.0, assignments=0), _budget(unit_cap=5.0), now=0
    )
    tracker.confirm_own_assignment(
        {"our-endpoint": "snake-lane"}, _usage(rate=2.0, assignments=1), "our-endpoint"
    )
    with _warns("PROVIDER_RATE_DRIFT"):
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


def test_manifest_can_require_exact_priced_ceiling_but_cannot_loosen_tolerance():
    from research.colab_cli.accounting import admit_account

    budget = {
        "rate_units_per_hour": 1.54,
        "unit_cap": 4.81,
        "other_assignments": {},
        "rate_tolerance_units_per_hour": 0.0,
    }
    tracker = admit_account({}, {"assignments": 0, "balance": 100, "rate": 0}, budget)
    assert (
        tracker.calibration_read(
            {"new": "test"}, {"assignments": 1, "balance": 100, "rate": 1.54}, "new"
        )
        == 1.54
    )
    with _warns("OBSERVED_RATE_ABOVE_PRICED_BOUND"):
        tracker.calibration_read(
            {"new": "test"}, {"assignments": 1, "balance": 100, "rate": 1.55}, "new"
        )
    with _warns("RATE_TOLERANCE_RELAXED"):
        admit_account(
            {},
            {"assignments": 0, "balance": 100, "rate": 0},
            budget | {"rate_tolerance_units_per_hour": 0.06},
        )


def test_hundreds_of_drift_alerts_never_stop_polling(capsys):
    tracker = accounting.admit_account(
        {}, _usage(rate=0, assignments=0), _budget(unit_cap=0), now=0
    )
    tracker.confirm_own_assignment({"own": "lane"}, _usage(rate=2, assignments=1), "own")
    for poll in range(1, 401):
        spend = tracker.check(
            _usage(balance=-1, rate=20, assignments=1), now=poll, sessions={"own": "lane"}
        )
        assert spend["rate_crosscheck_status"] == "WARNING_RATE_DRIFT"
        assert spend["unit_cap_status"] == "WARNING_UNIT_CAP_REACHED"
    assert tracker.receipt()["last_crosscheck"]["elapsed_seconds"] == 400
    warnings = capsys.readouterr().err
    assert warnings.count('"WARNING": "PROVIDER_RATE_DRIFT"') == 400
    assert warnings.count('"WARNING": "UNEXPLAINED_ACCOUNT_DEBIT"') == 400


@pytest.mark.parametrize(
    "budget", [{}, {"unit_cap": None}, {"unit_cap": -1, "rate_units_per_hour": None}]
)
def test_optional_or_invalid_budget_measurements_do_not_stop_accounting(budget):
    tracker = accounting.admit_account({}, _usage(rate=0, assignments=0), budget, now=0)
    tracker.confirm_own_assignment({"own": "lane"}, _usage(rate=2, assignments=1), "own")
    assert tracker.check(_usage(rate=2, assignments=1), now=60, sessions={"own": "lane"})[
        "own_elapsed_units"
    ] == pytest.approx(2 / 60)
