"""Per-session Colab unit accounting.

The provider reports an account-wide balance and rate, while a broker owns one
session.  This module makes that distinction explicit: every pre-existing
assignment must have a declared price, a session is capped by its own
pre-priced wall-clock charge, and the account balance is only a reconciled
cross-check after the measured baseline charge has been removed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping


class AccountingError(ValueError):
    """The account state cannot be safely attributed to one supervised session."""


# `usage` is rendered to two decimal places and provider debits can arrive late.
# Permit five minutes of the fully accounted rate plus one rendered cent before
# labelling a balance observation unexplained. This is evidence/alert only:
# the session's priced wall clock remains its spending stop.
BALANCE_ROUNDING_TOLERANCE_UNITS = 0.01
BALANCE_LAG_TOLERANCE_SECONDS = 300.0
TRANSITION_MISMATCH_POLLS = 2


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AccountingError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise AccountingError(f"{label} must be finite and {qualifier}")
    return result


def _usage(value: Mapping[str, Any]) -> dict[str, float]:
    try:
        assignments = value["assignments"]
        if isinstance(assignments, bool) or not isinstance(assignments, (int, float)):
            raise TypeError
        if int(assignments) != assignments or assignments < 0:
            raise ValueError
        return {
            "balance": _number(value["balance"], "usage.balance"),
            "rate": _number(value["rate"], "usage.rate"),
            "assignments": float(assignments),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise AccountingError(
            "usage must contain nonnegative balance, rate, and integer assignments"
        ) from exc


def _rate(value: Any, label: str) -> float:
    if isinstance(value, Mapping):
        value = value.get("rate_units_per_hour")
    return _number(value, label, positive=True)


def _active_rates(active_records: Iterable[Mapping[str, Any]]) -> dict[str, tuple[float, str]]:
    """Read broker-owned active records without accepting ambiguous identities."""
    result: dict[str, tuple[float, str]] = {}
    for record in active_records:
        endpoint = record.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise AccountingError("active broker record lacks endpoint")
        # New broker records reserve a priced ceiling before calibration. That
        # ceiling is never evidence of what another assignment actually bills.
        field = (
            "observed_rate_units_per_hour"
            if "priced_rate_units_per_hour" in record
            else "rate_units_per_hour"
        )
        rate = _rate(record.get(field), f"active record {endpoint} {field}")
        existing = result.get(endpoint)
        if existing and existing[0] != rate:
            raise AccountingError(f"conflicting active broker rates for {endpoint}")
        result[endpoint] = (rate, "broker.active")
    return result


def _manifest_rates(value: Any) -> dict[str, tuple[float, str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AccountingError("budget.other_assignments must map endpoint to rate_units_per_hour")
    result: dict[str, tuple[float, str]] = {}
    for endpoint, declared in value.items():
        if not isinstance(endpoint, str) or not endpoint:
            raise AccountingError("budget.other_assignments endpoint must be a nonempty string")
        result[endpoint] = (
            _rate(declared, f"other assignment {endpoint} rate_units_per_hour"),
            "manifest.other_assignments",
        )
    return result


def _merge_rates(
    manifest: dict[str, tuple[float, str]], active: dict[str, tuple[float, str]]
) -> dict[str, tuple[float, str]]:
    result = dict(active)
    for endpoint, declaration in manifest.items():
        existing = result.get(endpoint)
        if existing and not math.isclose(existing[0], declaration[0], rel_tol=0, abs_tol=1e-9):
            raise AccountingError(f"conflicting manifest and active broker rates for {endpoint}")
        # Keep broker provenance when both sources price the same endpoint.
        # The manifest proves its rate, while broker ownership is required to
        # distinguish a bounded membership transition from generic rate drift.
        if existing is None:
            result[endpoint] = declaration
    return result


@dataclass
class AccountTracker:
    """Admission snapshot and later attributable-spend checks for one session."""

    baseline_balance_units: float
    baseline_rate_units_per_hour: float
    baseline_endpoints: dict[str, str]
    known_rates: dict[str, tuple[float, str]]
    own_rate_units_per_hour: float
    unit_cap: float
    started_monotonic: float
    rate_tolerance_units_per_hour: float
    endpoint: str | None = None
    last_reconciled_monotonic: float | None = None
    integrated_other_units: float = 0.0
    current_other_rates: dict[str, tuple[float, str]] | None = None
    last_crosscheck: dict[str, float] | None = None
    observed_rate_units_per_hour: float | None = None
    calibration: dict[str, Any] | None = None
    transition_mismatch_polls: int = 0
    transition_from_other_rates: dict[str, tuple[float, str]] | None = None
    transition_to_other_rates: dict[str, tuple[float, str]] | None = None
    transition_broker_active_rates: dict[str, tuple[float, str]] | None = None

    def accounted_assignments(self) -> dict[str, Any]:
        """Stable evidence format suitable for admission and terminal receipts."""
        entries = []
        for endpoint, session in sorted(self.baseline_endpoints.items()):
            rate, source = self.known_rates[endpoint]
            entries.append(
                {
                    "endpoint": endpoint,
                    "session": session,
                    "rate_units_per_hour": rate,
                    "source": source,
                }
            )
        return {
            "assignments": entries,
            "baseline_rate_units_per_hour": self.baseline_rate_units_per_hour,
            "baseline_balance_units": self.baseline_balance_units,
            "declared_other_rate_units_per_hour": sum(
                rate for rate, _ in self.known_rates.values()
            ),
            "unaccounted_endpoints": [],
        }

    def receipt(self) -> dict[str, Any]:
        """Return durable accounting evidence without claiming exact provider billing."""
        result = self.accounted_assignments()
        result.update(
            {
                "allocated_endpoint": self.endpoint,
                "own_rate_units_per_hour": self.own_rate_units_per_hour,
                "priced_rate_units_per_hour": self.own_rate_units_per_hour,
                "observed_rate_units_per_hour": self.observed_rate_units_per_hour,
                "calibration": self.calibration,
                "unit_cap": self.unit_cap,
                "integrated_accounted_other_units": self.integrated_other_units,
                "last_crosscheck": self.last_crosscheck,
            }
        )
        return result

    def calibration_read(
        self,
        sessions: Mapping[str, str],
        usage: Mapping[str, Any],
        endpoint: str,
    ) -> float | None:
        """Validate one provider sample; None means a still-lagging observation.

        The broker requires repeated stable positive samples within its bounded
        calibration window before calling confirm_own_assignment. Unknown
        endpoints and an above-ceiling rate always fail immediately.
        """
        if not isinstance(endpoint, str) or not endpoint:
            raise AccountingError("allocated endpoint must be a nonempty string")
        now_sessions = dict(sessions)
        expected = set(self.baseline_endpoints) | {endpoint}
        if set(now_sessions) - expected or any(
            now_sessions.get(e) != name for e, name in self.baseline_endpoints.items()
        ):
            raise AccountingError("assignment endpoints changed during allocation")
        current = _usage(usage)
        if int(current["assignments"]) > len(expected):
            raise AccountingError(
                "provider assignment count disagrees with sessions during allocation"
            )
        observed = current["rate"] - self.baseline_rate_units_per_hour
        if observed > self.own_rate_units_per_hour + self.rate_tolerance_units_per_hour:
            raise AccountingError("observed session rate exceeds priced upper bound plus tolerance")
        if (
            set(now_sessions) != expected
            or int(current["assignments"]) != len(expected)
            or observed <= 0
        ):
            return None
        return observed

    def confirm_own_assignment(
        self,
        sessions: Mapping[str, str],
        usage: Mapping[str, Any],
        endpoint: str,
    ) -> dict[str, Any]:
        """Record a stable provider rate after broker-side bounded calibration."""
        observed = self.calibration_read(sessions, usage, endpoint)
        if observed is None:
            raise AccountingError("provider session rate is unknown or not stabilized")
        self.observed_rate_units_per_hour = observed
        self.endpoint = endpoint
        self.last_reconciled_monotonic = self.started_monotonic
        self.current_other_rates = dict(self.known_rates)
        result = self.accounted_assignments()
        result.update(
            {
                "allocated_endpoint": endpoint,
                "allocated_rate_units_per_hour": self.own_rate_units_per_hour,
                "priced_rate_units_per_hour": self.own_rate_units_per_hour,
                "observed_rate_units_per_hour": observed,
                "observed_account_rate_units_per_hour": _usage(usage)["rate"],
            }
        )
        return result

    def reconcile(
        self,
        sessions: Mapping[str, str],
        *,
        active_records: Iterable[Mapping[str, Any]] = (),
        budget_other_assignments: Any = None,
        previous_other_rates: Mapping[str, tuple[float, str]] | None = None,
    ) -> dict[str, tuple[float, str]]:
        """Price the currently active *other* assignments at this poll boundary.

        Lanes may start or stop while this session is alive.  A fresh endpoint
        must be declared by the manifest or a broker-owned active record before
        it can be subtracted from the account balance.  Departed endpoints are
        intentionally allowed: their old price is integrated only through the
        last observation.
        """
        if self.endpoint is None or self.observed_rate_units_per_hour is None:
            raise AccountingError("allocation was not rate-confirmed")
        current = dict(sessions)
        if self.endpoint not in current:
            raise AccountingError("supervised assignment disappeared from sessions")
        other_endpoints = set(current) - {self.endpoint}
        known = _merge_rates(
            _manifest_rates(budget_other_assignments), _active_rates(active_records)
        )
        previous = previous_other_rates or {}
        missing = sorted(other_endpoints - set(known))
        historical_broker = {
            endpoint
            for endpoint in missing
            if endpoint in previous and previous[endpoint][1] == "broker.active"
        }
        missing = sorted(set(missing) - historical_broker)
        if missing:
            raise AccountingError(f"unaccounted active assignment endpoints: {', '.join(missing)}")
        result: dict[str, tuple[float, str]] = {}
        for endpoint in other_endpoints:
            current = known.get(endpoint)
            prior = previous.get(endpoint)
            # A same-priced manifest declaration does not erase a broker
            # identity already observed for this endpoint. The latter remains
            # needed to explain a bounded provider departure lag.
            if (
                current is not None
                and prior is not None
                and prior[1] == "broker.active"
                and math.isclose(current[0], prior[0], rel_tol=0, abs_tol=1e-9)
            ):
                result[endpoint] = prior
            else:
                result[endpoint] = current if current is not None else previous[endpoint]
        return result

    def _transition_match(
        self,
        usage: Mapping[str, float],
        previous: Mapping[str, tuple[float, str]],
        current: Mapping[str, tuple[float, str]],
        broker_active: Mapping[str, tuple[float, str]],
    ) -> bool:
        """Allow a bounded provider lag only for broker-owned membership changes.

        A departing broker lane can disappear from its durable active record
        before the provider read loses it.  Conversely, a newly durable
        broker record can appear before the provider read includes it.  Test
        count and rate together against every intermediate membership state;
        a bare rate discrepancy never earns this grace period.
        """
        previous_endpoints = set(previous)
        current_endpoints = set(current)
        arrival_rates = {
            endpoint: current[endpoint] if endpoint in current else broker_active[endpoint]
            for endpoint in (current_endpoints - previous_endpoints)
            | (set(broker_active) - previous_endpoints)
            if (current.get(endpoint) or broker_active.get(endpoint))[1] == "broker.active"
        }
        departure_rates = {
            endpoint: previous[endpoint]
            for endpoint in (previous_endpoints - current_endpoints)
            | (previous_endpoints - set(broker_active))
            if previous[endpoint][1] == "broker.active"
        }
        arrivals = sorted(arrival_rates)
        departures = sorted(departure_rates)
        changed = arrivals + departures
        fixed = (previous_endpoints & current_endpoints) - set(departures)
        if any(
            not math.isclose(previous[endpoint][0], current[endpoint][0], rel_tol=0, abs_tol=1e-9)
            for endpoint in fixed
        ):
            return False
        # A manifest-only membership change is declared, but is not a broker
        # transition whose provider lag can be trusted.
        if not changed:
            return False
        base_count = 1 + len(fixed) + len(departures)
        base_rate = self.observed_rate_units_per_hour + sum(previous[e][0] for e in fixed)
        base_rate += sum(previous[e][0] for e in departures)
        settled_count = 1 + len(current_endpoints)
        settled_rate = self.observed_rate_units_per_hour + sum(rate for rate, _ in current.values())
        for applied in product((False, True), repeat=len(changed)):
            count = base_count
            rate = base_rate
            for endpoint, is_applied in zip(changed, applied):
                if not is_applied:
                    continue
                if endpoint in arrivals:
                    count += 1
                    rate += arrival_rates[endpoint][0]
                else:
                    count -= 1
                    rate -= departure_rates[endpoint][0]
            if count == settled_count and math.isclose(
                rate, settled_rate, rel_tol=0, abs_tol=self.rate_tolerance_units_per_hour
            ):
                continue
            if int(usage["assignments"]) == count and math.isclose(
                usage["rate"], rate, rel_tol=0, abs_tol=self.rate_tolerance_units_per_hour
            ):
                return True
        return False

    def check(
        self,
        usage: Mapping[str, Any],
        *,
        now: float | None = None,
        sessions: Mapping[str, str] | None = None,
        active_records: Iterable[Mapping[str, Any]] = (),
        budget_other_assignments: Any = None,
    ) -> dict[str, Any]:
        """Enforce the session's own cap and reconcile account debit separately.

        Account balance never directly trips the cap because it also includes
        declared notebook and other broker sessions.  The attributable debit is
        retained as conservative evidence after subtracting the measured
        pre-allocation provider rate integrated over this session's lifetime.
        """
        if self.endpoint is None:
            raise AccountingError("allocation was not rate-confirmed")
        current = _usage(usage)
        now = time.monotonic() if now is None else _number(now, "now")
        elapsed = now - self.started_monotonic
        if elapsed < 0:
            raise AccountingError("monotonic clock moved backward")
        own_elapsed_units = elapsed * self.own_rate_units_per_hour / 3600.0
        if own_elapsed_units >= self.unit_cap:
            raise AccountingError("priced session unit cap reached")
        if self.last_reconciled_monotonic is None or self.current_other_rates is None:
            raise AccountingError("allocation was not rate-confirmed")
        interval = now - self.last_reconciled_monotonic
        if interval < 0:
            raise AccountingError("monotonic clock moved backward")
        self.integrated_other_units += (
            interval * sum(rate for rate, _ in self.current_other_rates.values()) / 3600.0
        )
        self.last_reconciled_monotonic = now
        if sessions is not None:
            current_sessions = dict(sessions)
            if max(int(current["assignments"]), len(current_sessions)) > 3:
                raise AccountingError("account-wide assignment cap exceeded during polling")
            previous_other_rates = dict(self.current_other_rates)
            active_records = list(active_records)
            broker_active_rates = _active_rates(active_records)
            current_other_rates = self.reconcile(
                sessions,
                active_records=active_records,
                budget_other_assignments=budget_other_assignments,
                previous_other_rates=previous_other_rates,
            )
            broker_departed = sorted(
                endpoint
                for endpoint in current_other_rates
                if endpoint in previous_other_rates
                and previous_other_rates[endpoint][1] == "broker.active"
                and endpoint not in broker_active_rates
            )
            if any(
                not math.isclose(
                    previous_other_rates[endpoint][0],
                    current_other_rates[endpoint][0],
                    rel_tol=0,
                    abs_tol=1e-9,
                )
                for endpoint in broker_departed
            ):
                raise AccountingError(
                    "provider usage rate drift is not explained by accounted active assignments"
                )
            expected_rate = self.observed_rate_units_per_hour + sum(
                rate for rate, _ in current_other_rates.values()
            )
            provider_settled = int(current["assignments"]) == len(
                current_sessions
            ) and math.isclose(
                current["rate"],
                expected_rate,
                rel_tol=0,
                abs_tol=self.rate_tolerance_units_per_hour,
            )
            # Matching provider reads do not renew broker ownership. Keep the
            # known price, but share the existing bounded transition window
            # until sessions drops the endpoint or the broker re-lists it.
            settled = provider_settled and not broker_departed
            if settled:
                self.transition_mismatch_polls = 0
                self.transition_from_other_rates = None
                self.transition_to_other_rates = None
                self.transition_broker_active_rates = None
            else:
                transition_from = previous_other_rates
                transition_to = current_other_rates
                transition_broker_active = broker_active_rates
                if (
                    self.transition_from_other_rates is not None
                    and self.transition_to_other_rates == current_other_rates
                ):
                    transition_from = self.transition_from_other_rates
                    transition_to = self.transition_to_other_rates
                    transition_broker_active = dict(self.transition_broker_active_rates or {})
                    transition_broker_active.update(broker_active_rates)
                transition_matches = (provider_settled and bool(broker_departed)) or (
                    self._transition_match(
                        current, transition_from, transition_to, transition_broker_active
                    )
                )
            if not settled and transition_matches:
                self.transition_mismatch_polls += 1
                if self.transition_mismatch_polls > TRANSITION_MISMATCH_POLLS:
                    if broker_departed:
                        raise AccountingError(
                            "unaccounted active assignment endpoints: " + ", ".join(broker_departed)
                        )
                    if int(current["assignments"]) != len(current_sessions):
                        raise AccountingError(
                            "provider assignment count disagrees with sessions during polling"
                        )
                    raise AccountingError(
                        "provider usage rate drift is not explained by accounted active assignments"
                    )
                self.transition_from_other_rates = dict(transition_from)
                self.transition_to_other_rates = dict(transition_to)
                self.transition_broker_active_rates = dict(transition_broker_active)
            elif not settled and int(current["assignments"]) != len(current_sessions):
                raise AccountingError(
                    "provider assignment count disagrees with sessions during polling"
                )
            elif not settled:
                raise AccountingError(
                    "provider usage rate drift is not explained by accounted active assignments"
                )
            self.current_other_rates = current_other_rates
        observed_account_debit = self.baseline_balance_units - current["balance"]
        attributable_debit = max(0.0, observed_account_debit - self.integrated_other_units)
        observed_elapsed_units = elapsed * self.observed_rate_units_per_hour / 3600.0
        expected_account_debit = self.integrated_other_units + observed_elapsed_units
        balance_discrepancy = observed_account_debit - expected_account_debit
        rate_for_lag = self.observed_rate_units_per_hour + sum(
            rate for rate, _ in self.current_other_rates.values()
        )
        balance_tolerance = (
            BALANCE_ROUNDING_TOLERANCE_UNITS + rate_for_lag * BALANCE_LAG_TOLERANCE_SECONDS / 3600.0
        )
        # Provider billing can lag or batch.  The result is evidence, never a
        # raw-account cap; the wall-clock charge above is authoritative.
        result = {
            "elapsed_seconds": elapsed,
            "own_elapsed_units": own_elapsed_units,
            "observed_elapsed_units": observed_elapsed_units,
            "observed_account_debit_units": observed_account_debit,
            "baseline_integrated_units": self.integrated_other_units,
            "attributable_account_debit_units": attributable_debit,
            "observed_rate_units_per_hour": current["rate"],
            "expected_account_debit_units": expected_account_debit,
            "balance_discrepancy_units": balance_discrepancy,
            "balance_discrepancy_tolerance_units": balance_tolerance,
            "balance_crosscheck_status": (
                "WARNING_UNEXPLAINED_DEBIT"
                if abs(balance_discrepancy) > balance_tolerance
                else "OK"
            ),
            "provider_transition_status": (
                "WARNING_BROKER_TRANSITION_LAG" if self.transition_mismatch_polls else "OK"
            ),
            "provider_transition_mismatch_polls": self.transition_mismatch_polls,
        }
        self.last_crosscheck = result
        return result


def admit_account(
    sessions: Mapping[str, str],
    usage: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    active_records: Iterable[Mapping[str, Any]] = (),
    now: float | None = None,
    rate_tolerance_units_per_hour: float = 0.05,
) -> AccountTracker:
    """Require full attribution of pre-existing assignments before allocation."""
    endpoint_sessions = dict(sessions)
    if any(not isinstance(endpoint, str) or not endpoint for endpoint in endpoint_sessions):
        raise AccountingError("session endpoint must be a nonempty string")
    baseline = _usage(usage)
    if int(baseline["assignments"]) != len(endpoint_sessions):
        raise AccountingError("provider assignment count disagrees with sessions at admission")
    tolerance = _number(rate_tolerance_units_per_hour, "rate tolerance")
    if not isinstance(budget, Mapping):
        raise AccountingError("budget must be a mapping")
    own_rate = _rate(budget.get("rate_units_per_hour"), "budget.rate_units_per_hour")
    unit_cap = _number(budget.get("unit_cap"), "budget.unit_cap", positive=True)
    known = _merge_rates(
        _manifest_rates(budget.get("other_assignments")), _active_rates(active_records)
    )
    missing = sorted(set(endpoint_sessions) - set(known))
    stale = sorted(set(known) - set(endpoint_sessions))
    if missing:
        raise AccountingError(f"unaccounted active assignment endpoints: {', '.join(missing)}")
    if stale:
        raise AccountingError(
            f"accounted assignment endpoints are no longer active: {', '.join(stale)}"
        )
    declared_rate = sum(rate for rate, _ in known.values())
    if abs(baseline["rate"] - declared_rate) > tolerance:
        raise AccountingError("provider baseline rate does not match accounted active assignments")
    started = time.monotonic() if now is None else _number(now, "now")
    return AccountTracker(
        baseline_balance_units=baseline["balance"],
        baseline_rate_units_per_hour=baseline["rate"],
        baseline_endpoints=endpoint_sessions,
        known_rates=known,
        own_rate_units_per_hour=own_rate,
        unit_cap=unit_cap,
        started_monotonic=started,
        rate_tolerance_units_per_hour=tolerance,
    )
