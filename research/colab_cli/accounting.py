"""Per-session Colab unit accounting.

Provider prices, unit caps, balances and drift are measurement and warning
thresholds only. Assignment identity, reconciliation and concurrency remain
safety boundaries. The supervisor enforces whole-session wall time separately.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping


class AccountingError(ValueError):
    """The account state cannot be safely attributed to one supervised session."""


# `usage` is rendered to two decimal places and provider debits can arrive late.
# Permit five minutes of the fully accounted rate plus one rendered cent before
# labelling a balance observation unexplained. This is evidence/alert only:
# the session's wall timeout remains its runaway stop.
BALANCE_ROUNDING_TOLERANCE_UNITS = 0.01
BALANCE_LAG_TOLERANCE_SECONDS = 300.0
TRANSITION_MISMATCH_POLLS = 2


def budget_warning(code: str, **evidence: Any) -> None:
    """Emit a structured, non-fatal accounting warning without losing evidence."""
    print(json.dumps({"WARNING": code, **evidence}), file=sys.stderr, flush=True)


def measurement_rate(value: Any, label: str) -> float:
    """Keep a usable measurement even when a legacy price is unavailable."""
    try:
        return _number(value, label)
    except AccountingError:
        budget_warning("RATE_UNAVAILABLE", field=label, value=value)
        return 0.0


def warning_threshold(value: Any, label: str, default: float | None = None) -> float | None:
    """Ignore invalid legacy thresholds rather than making them admission gates."""
    if value is None:
        return default
    try:
        return _number(value, label)
    except AccountingError:
        budget_warning("INVALID_LEGACY_THRESHOLD", field=label, value=value)
        return default


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
        if isinstance(value["balance"], bool) or not math.isfinite(float(value["balance"])):
            raise ValueError
        return {
            "balance": float(value["balance"]),
            "rate": measurement_rate(value["rate"], "usage.rate"),
            "assignments": float(assignments),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise AccountingError(
            "usage must contain finite balance, a rate field, and integer assignments"
        ) from exc


def _rate(value: Any, label: str) -> float:
    if isinstance(value, Mapping):
        value = value.get("rate_units_per_hour")
    return measurement_rate(value, label)


def _active_rates(active_records: Iterable[Mapping[str, Any]]) -> dict[str, tuple[float, str]]:
    """Read broker-owned active records without accepting ambiguous identities."""
    result: dict[str, tuple[float, str]] = {}
    for record in active_records:
        endpoint = record.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise AccountingError("active broker record lacks endpoint")
        # A priced fallback is informational when bounded calibration cannot
        # establish a settled observed rate. The active record marks that gap.
        field = (
            "observed_rate_units_per_hour"
            if "priced_rate_units_per_hour" in record
            else "rate_units_per_hour"
        )
        if record.get(field) is None:
            budget_warning(
                "RATE_UNAVAILABLE",
                endpoint=endpoint,
                informational_rate=record.get("priced_rate_units_per_hour"),
            )
        rate = _rate(
            (
                record.get(field)
                if record.get(field) is not None
                else record.get("priced_rate_units_per_hour")
            ),
            f"active record {endpoint} {field}",
        )
        existing = result.get(endpoint)
        if existing and existing[0] != rate:
            budget_warning(
                "CONFLICTING_ACTIVE_RATES", endpoint=endpoint, previous=existing[0], rate=rate
            )
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
            budget_warning(
                "CONFLICTING_DECLARED_RATE",
                endpoint=endpoint,
                declared=declaration[0],
                observed=existing[0],
            )
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
    unit_cap: float | None
    started_monotonic: float
    rate_tolerance_units_per_hour: float
    endpoint: str | None = None
    last_reconciled_monotonic: float | None = None
    integrated_other_units: float = 0.0
    current_other_rates: dict[str, tuple[float, str]] | None = None
    last_crosscheck: dict[str, Any] | None = None
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

    @property
    def effective_rate(self) -> float:
        """Use the priced informational estimate until a rate is settled."""
        return self.observed_rate_units_per_hour or self.own_rate_units_per_hour or 0.0

    def receipt(self) -> dict[str, Any]:
        """Return durable accounting evidence without claiming exact provider billing."""
        result = self.accounted_assignments()
        result.update(
            {
                "allocated_endpoint": self.endpoint,
                "own_rate_units_per_hour": self.own_rate_units_per_hour or self.effective_rate,
                "rate_measurement_unavailable": self.observed_rate_units_per_hour is None,
                "accounting_rate_basis": (
                    "observed"
                    if self.observed_rate_units_per_hour is not None
                    else "priced_informational"
                ),
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

        The broker requires repeated identity samples within its bounded
        calibration window before calling confirm_own_assignment. Unknown
        endpoints fail immediately; rates produce warnings only.
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
            budget_warning(
                "OBSERVED_RATE_ABOVE_PRICED_BOUND",
                observed=observed,
                priced=self.own_rate_units_per_hour,
            )
        if set(now_sessions) != expected or int(current["assignments"]) != len(expected):
            return None
        if observed <= 0:
            budget_warning("NONPOSITIVE_OBSERVED_RATE", observed=observed)
        return max(0.0, observed)

    def confirm_own_assignment(
        self,
        sessions: Mapping[str, str],
        usage: Mapping[str, Any],
        endpoint: str,
        *,
        rate_confirmed: bool = True,
    ) -> dict[str, Any]:
        """Record a stable provider rate after broker-side bounded calibration."""
        observed = self.calibration_read(sessions, usage, endpoint)
        if observed is None:
            raise AccountingError("allocated assignment identity/count not confirmed")
        self.observed_rate_units_per_hour = observed if rate_confirmed and observed > 0 else None
        self.endpoint = endpoint
        self.last_reconciled_monotonic = self.started_monotonic
        self.current_other_rates = dict(self.known_rates)
        result = self.accounted_assignments()
        result.update(
            {
                "allocated_endpoint": endpoint,
                "allocated_rate_units_per_hour": self.own_rate_units_per_hour,
                "priced_rate_units_per_hour": self.own_rate_units_per_hour,
                "observed_rate_units_per_hour": self.observed_rate_units_per_hour,
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
        if self.endpoint is None:
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
        count alone against every intermediate membership state;
        rate differences remain informational and never decide count grace.
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
        # A manifest-only membership change is declared, but is not a broker
        # transition whose provider lag can be trusted.
        if not changed:
            return False
        base_count = 1 + len(fixed) + len(departures)
        settled_count = 1 + len(current_endpoints)
        for applied in product((False, True), repeat=len(changed)):
            count = base_count
            for endpoint, is_applied in zip(changed, applied):
                if not is_applied:
                    continue
                if endpoint in arrivals:
                    count += 1
                else:
                    count -= 1
            if count == settled_count:
                continue
            if int(usage["assignments"]) == count:
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
        """Measure session usage and reconcile account debit without budget stops.

        Balance includes notebook and other broker sessions. Retain attributable
        debit after subtracting their measured rates, without stopping work.
        """
        if self.endpoint is None:
            raise AccountingError("allocation was not rate-confirmed")
        current = _usage(usage)
        if int(current["assignments"]) > 3:
            raise AccountingError("account-wide assignment cap exceeded during polling")
        now = time.monotonic() if now is None else _number(now, "now")
        elapsed = now - self.started_monotonic
        if elapsed < 0:
            raise AccountingError("monotonic clock moved backward")
        own_elapsed_units = (
            elapsed
            * (self.own_rate_units_per_hour or self.observed_rate_units_per_hour or 0)
            / 3600.0
        )
        if self.unit_cap is not None and own_elapsed_units >= self.unit_cap:
            budget_warning(
                "SESSION_UNIT_CAP_REACHED", units=own_elapsed_units, unit_cap=self.unit_cap
            )
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
                budget_warning(
                    "PROVIDER_RATE_DRIFT", observed=current["rate"], endpoints=broker_departed
                )
            expected_rate = self.effective_rate + sum(
                rate for rate, _ in current_other_rates.values()
            )
            provider_settled = int(current["assignments"]) == len(current_sessions)
            if abs(current["rate"] - expected_rate) > self.rate_tolerance_units_per_hour:
                budget_warning(
                    "PROVIDER_RATE_DRIFT", observed=current["rate"], expected=expected_rate
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
                    budget_warning(
                        "PROVIDER_RATE_DRIFT", observed=current["rate"], expected=expected_rate
                    )
                self.transition_from_other_rates = dict(transition_from)
                self.transition_to_other_rates = dict(transition_to)
                self.transition_broker_active_rates = dict(transition_broker_active)
            elif not settled and int(current["assignments"]) != len(current_sessions):
                raise AccountingError(
                    "provider assignment count disagrees with sessions during polling"
                )
            elif not settled:
                budget_warning(
                    "PROVIDER_RATE_DRIFT", observed=current["rate"], endpoints=broker_departed
                )
            self.current_other_rates = current_other_rates
        else:
            expected_rate = self.effective_rate + sum(
                rate for rate, _ in self.current_other_rates.values()
            )
            if abs(current["rate"] - expected_rate) > self.rate_tolerance_units_per_hour:
                budget_warning(
                    "PROVIDER_RATE_DRIFT", observed=current["rate"], expected=expected_rate
                )
        observed_account_debit = self.baseline_balance_units - current["balance"]
        attributable_debit = max(0.0, observed_account_debit - self.integrated_other_units)
        observed_elapsed_units = elapsed * self.effective_rate / 3600.0
        expected_account_debit = self.integrated_other_units + observed_elapsed_units
        balance_discrepancy = observed_account_debit - expected_account_debit
        rate_for_lag = self.effective_rate + sum(
            rate for rate, _ in self.current_other_rates.values()
        )
        balance_tolerance = (
            BALANCE_ROUNDING_TOLERANCE_UNITS + rate_for_lag * BALANCE_LAG_TOLERANCE_SECONDS / 3600.0
        )
        # Provider billing can lag or batch. All prices and debits are evidence;
        # only the supervisor's wall-clock timeout bounds session lifetime.
        result = {
            "elapsed_seconds": elapsed,
            "unit_cap_status": (
                "WARNING_UNIT_CAP_REACHED"
                if self.unit_cap is not None and own_elapsed_units >= self.unit_cap
                else "OK"
            ),
            "rate_crosscheck_status": (
                "WARNING_RATE_DRIFT"
                if abs(current["rate"] - expected_rate) > self.rate_tolerance_units_per_hour
                else "OK"
            ),
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
        if result["balance_crosscheck_status"] != "OK":
            budget_warning("UNEXPLAINED_ACCOUNT_DEBIT", spend=result)
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
    """Require known endpoints and record available price measurements."""
    endpoint_sessions = dict(sessions)
    if any(not isinstance(endpoint, str) or not endpoint for endpoint in endpoint_sessions):
        raise AccountingError("session endpoint must be a nonempty string")
    baseline = _usage(usage)
    if max(int(baseline["assignments"]), len(endpoint_sessions)) > 3:
        raise AccountingError("account-wide assignment cap exceeded at admission")
    if int(baseline["assignments"]) != len(endpoint_sessions):
        raise AccountingError("provider assignment count disagrees with sessions at admission")
    tolerance = _number(rate_tolerance_units_per_hour, "rate tolerance")
    if not isinstance(budget, Mapping):
        raise AccountingError("budget must be a mapping")
    if "rate_tolerance_units_per_hour" in budget:
        requested = warning_threshold(
            budget["rate_tolerance_units_per_hour"], "manifest rate tolerance", tolerance
        )
        if requested > tolerance:
            budget_warning("RATE_TOLERANCE_RELAXED", requested=requested, previous=tolerance)
        tolerance = requested
    own_rate = _rate(budget.get("rate_units_per_hour", 0), "budget.rate_units_per_hour")
    unit_cap = warning_threshold(budget.get("unit_cap"), "budget.unit_cap")
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
        budget_warning("BASELINE_RATE_MISMATCH", observed=baseline["rate"], declared=declared_rate)
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
