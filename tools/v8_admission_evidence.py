"""Local source and arithmetic checks for V8 admission; no external operations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tools import v8_admission_budget as budget

CAPS = {
    "packet": (150, 25),
    "v8": (216, 36),
    "q2": (864, 144),
    "q4": (1152, 192),
    "global": (7200, 1100),
}


def finite(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"INCOMPLETE_ADMISSION: invalid nonnegative {name}")
    return float(value)


def sourced(binding: dict[str, Any]) -> Any:
    """Read the exact field of a hash-bound local source, never a naked assertion."""
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("INCOMPLETE_ADMISSION: missing source binding")
    path = Path(binding["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("INCOMPLETE_ADMISSION: invalid source path")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != binding.get("sha256"):
        raise ValueError("INCOMPLETE_ADMISSION: changed source hash")
    pointer = binding.get("pointer")
    if not isinstance(pointer, str):
        raise ValueError("INCOMPLETE_ADMISSION: missing RFC6901 source pointer")
    if binding.get("format") == "text":
        if pointer:
            raise ValueError("INCOMPLETE_ADMISSION: text source cannot have field pointer")
        return data.decode()
    result = json.loads(data)
    if pointer:
        if not pointer.startswith("/"):
            raise ValueError("INCOMPLETE_ADMISSION: invalid RFC6901 source pointer")
        for part in pointer[1:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            result = result[int(key)] if isinstance(result, list) else result[key]
    return result


def utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("INCOMPLETE_ADMISSION: invalid UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("INCOMPLETE_ADMISSION: timestamp must have UTC offset")
    return parsed


def pair(value: dict[str, Any], name: str) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise ValueError(f"INCOMPLETE_ADMISSION: missing {name}")
    return finite(value.get("wall_minutes"), name), finite(value.get("units"), name)


def close(left: float, right: float, name: str) -> None:
    if not math.isclose(left, right, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"INCOMPLETE_ADMISSION: unreconciled {name}")


def validate_evidence(
    body: dict[str, Any],
    *,
    now: datetime,
    receipt_hashes: dict[str, str],
    reconstruction_path: Path | None = None,
) -> None:
    """Verify sourced accounting, parent balances, original clock and live boundary.

    Codex-V still owns provenance judgment and exact-body acceptance. Mechanical
    checks never promote a newly written JSON statement into original authority.
    """
    evidence = body.get("accounting_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("INCOMPLETE_ADMISSION: missing accounting evidence")
    reconstruction = budget.validate_reconstruction(
        evidence.get("reconstruction"), path=reconstruction_path or budget.RECONSTRUCTION
    )
    frozen_clock = reconstruction["original_t0"]
    original_t0 = frozen_clock
    sealed = finite(
        reconstruction["reconstructed_spendable_effective_limit_units"], "frozen spendable limit"
    )
    tightening = evidence.get("later_tightening", {})
    if tightening is not None:
        if not isinstance(tightening, dict):
            raise ValueError("INCOMPLETE_ADMISSION: invalid later tightening")
        earlier = tightening.get("earlier_authenticated_clock_source")
        if earlier is not None:
            candidate = sourced(earlier)
            if utc(candidate) > utc(frozen_clock):
                raise ValueError("INCOMPLETE_ADMISSION: later clock cannot extend reconstruction")
            original_t0 = candidate
        smaller = tightening.get("smaller_historical_limit_source")
        if smaller is not None:
            candidate_limit = finite(sourced(smaller), "recovered historical limit")
            if candidate_limit > sealed:
                raise ValueError("INCOMPLETE_ADMISSION: recovered ceiling cannot increase frozen C")
            sealed = candidate_limit
    if body.get("original_t0") != original_t0:
        raise ValueError("INCOMPLETE_ADMISSION: original T0 differs from source")
    clock = utc(original_t0)
    if clock > now:
        raise ValueError("INCOMPLETE_ADMISSION: original T0 is in future")
    if (
        body.get("reconstructed_spendable_effective_limit_units")
        != reconstruction["reconstructed_spendable_effective_limit_units"]
    ):
        raise ValueError("INCOMPLETE_ADMISSION: frozen C differs from reconstruction")
    if body.get("effective_spendable_limit_units") != sealed:
        raise ValueError("INCOMPLETE_ADMISSION: effective C does not include later tightening")
    ledger = sourced(evidence.get("ledger"))
    if not isinstance(ledger, dict) or ledger.get("schema") != "v8-admission-ledger/v1":
        raise ValueError("INCOMPLETE_ADMISSION: missing conservative ledger")
    if ledger.get("history_basis") != "INVENTORIED_CONSERVATIVE_BOUNDS" or any(
        ledger.get(key) is not True for key in ("complete_history", "complete_liabilities")
    ):
        raise ValueError("INCOMPLETE_ADMISSION: incomplete conservative ledger")
    rows = ledger.get("rows")
    holds = ledger.get("protected_holds")
    if not isinstance(rows, list) or not isinstance(holds, list):
        raise ValueError("INCOMPLETE_ADMISSION: ledger rows/holds missing")
    identities = set()
    endpoint_intervals: dict[str, list[tuple[datetime, datetime]]] = {}
    endpoint_without_interval = set()
    recomputed_spent = {scope: [0.0, 0.0] for scope in CAPS}
    recomputed_liabilities = {scope: [0.0, 0.0] for scope in CAPS}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("job_id"), str) or not row["job_id"]:
            raise ValueError("INCOMPLETE_ADMISSION: unbounded ledger identity")
        if row["job_id"] in identities:
            raise ValueError("INCOMPLETE_ADMISSION: duplicate ledger identity")
        identities.add(row["job_id"])
        interval = row.get("charged_interval")
        endpoint = row.get("endpoint")
        if endpoint is not None:
            if not isinstance(endpoint, str) or not endpoint:
                raise ValueError("INCOMPLETE_ADMISSION: endpoint interval missing")
            # A unique released endpoint may be bounded by trustworthy broker
            # elapsed time without inventing a historical absolute timestamp.
            # Reused endpoints require disjoint source-bound intervals.
            if interval is None:
                if endpoint in endpoint_intervals or endpoint in endpoint_without_interval:
                    raise ValueError("INCOMPLETE_ADMISSION: reused endpoint lacks interval union")
                endpoint_without_interval.add(endpoint)
            elif not isinstance(interval, dict) or endpoint in endpoint_without_interval:
                raise ValueError("INCOMPLETE_ADMISSION: reused endpoint lacks interval union")
        if endpoint is not None and interval is not None:
            if not all(
                isinstance(value, str) and value
                for value in (endpoint, interval.get("start_utc"), interval.get("end_utc"))
            ):
                raise ValueError("INCOMPLETE_ADMISSION: invalid endpoint interval")
            start, end = utc(interval["start_utc"]), utc(interval["end_utc"])
            if end <= start or any(
                start < previous_end and previous_start < end
                for previous_start, previous_end in endpoint_intervals.setdefault(endpoint, [])
            ):
                raise ValueError("INCOMPLETE_ADMISSION: overlapping endpoint interval")
            endpoint_intervals[endpoint].append((start, end))
        if row.get("charge_basis") not in {
            "CONSERVATIVE_BOUND",
            "MEASURED_PROVIDER_DEBIT",
            "OVERLAP_GUARD",
        }:
            raise ValueError("INCOMPLETE_ADMISSION: unbounded ledger charge")
        charged = pair(row.get("charged"), "ledger charge")
        if (
            not isinstance(row.get("charge_source"), dict)
            or sourced(row["charge_source"]) != row["charged"]
        ):
            raise ValueError("INCOMPLETE_ADMISSION: unsourced ledger charge")
        for key in ("authority_sha256", "manifest_sha256", "terminal_status", "release_status"):
            if not row.get(key):
                raise ValueError("INCOMPLETE_ADMISSION: incomplete ledger provenance")
        if not isinstance(row.get("evidence"), list) or not row["evidence"]:
            raise ValueError("INCOMPLETE_ADMISSION: missing ledger evidence")
        for descriptor in row["evidence"]:
            sourced(descriptor)
        if (
            not isinstance(row.get("scope_memberships"), list)
            or not row["scope_memberships"]
            or not set(row["scope_memberships"]).issubset(CAPS)
            or len(row["scope_memberships"]) != len(set(row["scope_memberships"]))
        ):
            raise ValueError("INCOMPLETE_ADMISSION: ledger scope memberships missing")
        if "global" not in row["scope_memberships"]:
            raise ValueError("INCOMPLETE_ADMISSION: ledger row does not debit global once")
        if row["charge_basis"] == "OVERLAP_GUARD" and not {"q2", "q4"}.issubset(
            row["scope_memberships"]
        ):
            raise ValueError("INCOMPLETE_ADMISSION: overlap guard must protect q2 and q4")
        liability = pair(row.get("remaining_liability"), "ledger liability")
        for scope in row["scope_memberships"]:
            recomputed_spent[scope] = [a + b for a, b in zip(recomputed_spent[scope], charged)]
            recomputed_liabilities[scope] = [
                a + b for a, b in zip(recomputed_liabilities[scope], liability)
            ]
    protected = {"I0", "Q3", "Q5", "Q6", "O1"}
    held = set()
    for hold in holds:
        if not isinstance(hold, dict) or hold.get("name") not in protected:
            raise ValueError("INCOMPLETE_ADMISSION: missing protected required-work hold")
        if hold["name"] in held:
            raise ValueError("INCOMPLETE_ADMISSION: duplicate protected required-work hold")
        held.add(hold["name"])
        charged = pair(hold.get("charged"), "protected hold")
        if not isinstance(hold.get("source"), dict) or sourced(hold["source"]) != hold["charged"]:
            raise ValueError("INCOMPLETE_ADMISSION: unsourced protected hold")
        if not set(hold.get("scope_memberships", [])) or not set(
            hold["scope_memberships"]
        ).issubset(CAPS):
            raise ValueError("INCOMPLETE_ADMISSION: protected hold scope missing")
        if "global" not in hold["scope_memberships"]:
            raise ValueError("INCOMPLETE_ADMISSION: protected hold does not debit global")
        if hold["name"] == "I0" and "q4" not in hold["scope_memberships"]:
            raise ValueError("INCOMPLETE_ADMISSION: I0 hold must protect q4")
        if len(hold["scope_memberships"]) != len(set(hold["scope_memberships"])):
            raise ValueError("INCOMPLETE_ADMISSION: duplicate protected hold scope")
        if hold["name"] == "I0" and any(charged) and "q4" not in hold["scope_memberships"]:
            raise ValueError("INCOMPLETE_ADMISSION: I0 integration hold must protect Q4")
        ceilings = {
            "I0": (144, 24),
            "Q3": (576, 96),
            "Q5": (1440, 240),
            "Q6": (2304, 384),
            "O1": (288, 48),
        }
        ceiling = ceilings[hold["name"]]
        if charged[0] + 1e-8 < ceiling[0] or charged[1] + 1e-8 < ceiling[1]:
            release = hold.get("released_or_completed_source")
            released = sourced(release) if isinstance(release, dict) else None
            if (
                not isinstance(released, dict)
                or released.get("status")
                not in {"COMPLETED", "RELEASED", "NARROWER_ACCEPTED_RESERVATION"}
                or released.get("charged") != hold["charged"]
            ):
                raise ValueError(
                    "INCOMPLETE_ADMISSION: reduced protected hold lacks sourced authority"
                )
        for scope in hold["scope_memberships"]:
            recomputed_liabilities[scope] = [
                a + b for a, b in zip(recomputed_liabilities[scope], charged)
            ]
    if held != protected:
        raise ValueError("INCOMPLETE_ADMISSION: missing protected required-work hold")
    reconciliation = ledger.get("reconciliation")
    if not isinstance(reconciliation, dict) or any(
        reconciliation.get(key) is not True for key in ("complete_history", "complete_liabilities")
    ):
        raise ValueError("INCOMPLETE_ADMISSION: incomplete original accounting history/liabilities")
    if reconciliation.get("original_t0") != original_t0:
        raise ValueError("INCOMPLETE_ADMISSION: reconciliation uses foreign T0")
    if (
        reconciliation.get("spent", {}).get("global") is None
        or reconciliation.get("liabilities", {}).get("global") is None
    ):
        raise ValueError("INCOMPLETE_ADMISSION: reconstruction ledger lacks global totals")
    deadline = clock + timedelta(hours=48)
    if "candidate_freeze" in evidence:
        freeze = sourced(evidence["candidate_freeze"])
        if (
            not clock <= utc(freeze["frozen_at_utc"]) <= min(now, deadline)
            or freeze.get("runtime_commit") != "669405e2bef629f7f69bc951cf49486d8b0a87bf"
            or freeze.get("manifest_sha256") != body.get("manifest_sha256")
        ):
            raise ValueError("INCOMPLETE_ADMISSION: exact candidate was not frozen by T0+48")
        integrated = finite(
            sourced(evidence.get("integrated_time_reservation_minutes")),
            "remaining integrated measurement reservation",
        )
        if integrated <= 0:
            raise ValueError("INCOMPLETE_ADMISSION: integrated measurement needs reserved time")
        deadline = clock + timedelta(hours=72) - timedelta(minutes=integrated)
    if utc(body["calendar_deadline_utc"]) != deadline or now + timedelta(minutes=120) > deadline:
        raise ValueError("INCOMPLETE_ADMISSION: original applicable deadline cannot fit")
    if reconciliation.get("v8_receipt_sha256") != receipt_hashes:
        raise ValueError("INCOMPLETE_ADMISSION: V8 receipt set is missing, revised or foreign")
    # Every reconciled input has its own source field. An independent reviewer
    # must establish completeness and the ownership of overlapping account costs.
    for category in ("spent", "liabilities"):
        for scope in CAPS:
            value = reconciliation.get(category, {}).get(scope)
            actual = sourced(reconciliation.get("sources", {}).get(category, {}).get(scope))
            if actual != value:
                raise ValueError(f"INCOMPLETE_ADMISSION: unsourced {scope} {category}")
            pair(value, f"{scope} {category}")
            expected = (
                recomputed_spent[scope] if category == "spent" else recomputed_liabilities[scope]
            )
            for observed, derived in zip(pair(value, f"{scope} {category}"), expected):
                close(observed, derived, f"{scope} {category} ledger total")
    if body.get("liabilities_and_reservations") != reconciliation["liabilities"]:
        raise ValueError("INCOMPLETE_ADMISSION: liabilities differ from reviewed accounting")
    balances = {}
    for scope, (minutes_cap, units_cap) in CAPS.items():
        if scope == "global":
            units_cap = min(units_cap, sealed)
        spent = pair(reconciliation["spent"][scope], scope + " spent")
        liability = pair(reconciliation["liabilities"][scope], scope + " liabilities")
        remaining = pair(reconciliation.get("remaining", {}).get(scope), scope + " remaining")
        close(remaining[0], minutes_cap - spent[0] - liability[0], scope + " wall balance")
        close(remaining[1], units_cap - spent[1] - liability[1], scope + " unit balance")
        if remaining[0] < 120 or remaining[1] < 20:
            raise ValueError(f"INCOMPLETE_ADMISSION: {scope} cannot reserve full 120/20")
        balances[scope] = remaining
    # Nested scopes need not be summed, but cannot have less historical spend
    # than their children. The main reservation is made once within every scope.
    for outer, inner in (("v8", "packet"), ("global", "v8")):
        for bigger, smaller in zip(
            pair(reconciliation["spent"][outer], outer),
            pair(reconciliation["spent"][inner], inner),
        ):
            if bigger + 1e-8 < smaller:
                raise ValueError("INCOMPLETE_ADMISSION: nested expenditure is inconsistent")
    expected = {
        "split_packet_remaining": balances["packet"],
        "v8_series_remaining": balances["v8"],
        "global_remaining": balances["global"],
        "q2_q4_remaining": tuple(min(a, b) for a, b in zip(balances["q2"], balances["q4"])),
    }
    for name, values in expected.items():
        actual = pair(body.get(name), name)
        for left, right in zip(actual, values):
            close(left, right, name)
    # Native provider-debit reconciliation must explicitly include every V8
    # attempt once, even if billing is below the conservative broker bound.
    charges = sourced(evidence.get("v8_charge_reconciliation"))
    if not isinstance(charges, list) or len(charges) != len(receipt_hashes):
        raise ValueError("INCOMPLETE_ADMISSION: incomplete V8 charge reconciliation")
    seen = set()
    totals = {"v8": [0.0, 0.0], "packet": [0.0, 0.0]}
    for charge in charges:
        job = charge.get("job_id")
        if job in seen or charge.get("receipt_sha256") != receipt_hashes.get(job) or job is None:
            raise ValueError("INCOMPLETE_ADMISSION: duplicate/foreign charged attempt")
        seen.add(job)
        values = pair(charge.get("charged"), "charged attempt")
        if sourced(charge.get("source")) != charge["charged"]:
            raise ValueError("INCOMPLETE_ADMISSION: unsourced reconciled charge")
        for scope in ("v8", "packet"):
            if scope == "packet" and "split" not in job:
                continue
            totals[scope] = [a + b for a, b in zip(totals[scope], values)]
        ledger_row = next((row for row in rows if row.get("job_id") == job), None)
        if (
            ledger_row is None
            or ledger_row.get("charged") != charge["charged"]
            or not {"v8", "global"}.issubset(ledger_row.get("scope_memberships", []))
            or ("split" in job and "packet" not in ledger_row.get("scope_memberships", []))
        ):
            raise ValueError("INCOMPLETE_ADMISSION: historical charge lacks exact ledger identity")
    if seen != set(receipt_hashes):
        raise ValueError("INCOMPLETE_ADMISSION: missing charged attempt")
    for scope in totals:
        for actual, total in zip(pair(reconciliation["spent"][scope], scope), totals[scope]):
            close(actual, total, scope + " reconciled charge sum")
    if totals["packet"][0] > 30 or totals["packet"][1] > 5:
        raise ValueError("INCOMPLETE_ADMISSION: historical canary portion is exhausted")
    usage_raw = sourced(evidence.get("usage"))
    sessions_raw = sourced(evidence.get("sessions"))
    if not isinstance(usage_raw, str) or not isinstance(sessions_raw, str):
        raise ValueError("INCOMPLETE_ADMISSION: live CLI evidence must be captured raw text")
    usage = {}
    for key, label in (
        ("balance", "Current balance"),
        ("rate", "Usage rate"),
        ("assignments", "Active assignments"),
    ):
        match = re.search(rf"{label}:\s*([0-9]+(?:\.[0-9]+)?)", usage_raw)
        if not match:
            raise ValueError("INCOMPLETE_ADMISSION: unrecognized live usage output")
        usage[key] = float(match[1])
    sessions = re.findall(r"^\[[^\]]+\]\s+(\S+)\s+\| Hardware:\s*(\S+)", sessions_raw, re.M)
    if not sessions and "No active sessions found on server." not in sessions_raw:
        raise ValueError("INCOMPLETE_ADMISSION: unrecognized live sessions output")
    endpoints = {endpoint for endpoint, _ in sessions}
    if (
        len(endpoints) != len(sessions)
        or len(sessions) > 2
        or usage["assignments"] != len(sessions)
    ):
        raise ValueError("INCOMPLETE_ADMISSION: live account assignment cap/count mismatch")
    if sessions and "No active sessions found on server." in sessions_raw:
        raise ValueError("INCOMPLETE_ADMISSION: contradictory sessions evidence")
    if any(hardware.upper() == "A100" for _, hardware in sessions):
        raise ValueError("INCOMPLETE_ADMISSION: previous A100 assignment remains")
    rates = body.get("current_assignments_and_rates")
    if not isinstance(rates, dict) or set(rates) != endpoints:
        raise ValueError("INCOMPLETE_ADMISSION: missing/unexplained live endpoint rates")
    for rate in rates.values():
        if finite(rate, "current endpoint rate") <= 0:
            raise ValueError("INCOMPLETE_ADMISSION: current endpoint rate must be positive")
    close(sum(rates.values()), usage["rate"], "provider account rate")
    current_rate = finite(body.get("verified_main_rate_units_per_hour"), "verified main rate")
    if not 0 < current_rate <= 10:
        raise ValueError("INCOMPLETE_ADMISSION: main rate outside A100 ceiling")
    close(
        current_rate, finite(sourced(evidence.get("main_rate")), "main price source"), "main price"
    )
    account_liability = finite(
        reconciliation.get("account_outstanding_units"), "account liabilities"
    )
    external = finite(
        reconciliation.get("external_liabilities_units"), "current external liabilities"
    )
    if sourced(reconciliation.get("sources", {}).get("external_liabilities_units")) != external:
        raise ValueError("INCOMPLETE_ADMISSION: unsourced current external liabilities")
    close(
        account_liability,
        external + pair(reconciliation["liabilities"]["global"], "global liabilities")[1],
        "current account liabilities",
    )
    if (
        sourced(reconciliation.get("sources", {}).get("account_outstanding_units"))
        != account_liability
    ):
        raise ValueError("INCOMPLETE_ADMISSION: unsourced account liabilities")
    close(
        finite(body.get("live_unencumbered_units"), "live units"),
        usage["balance"] - account_liability,
        "unencumbered account units",
    )
    if body["live_unencumbered_units"] < 20:
        raise ValueError("INCOMPLETE_ADMISSION: account cannot reserve the full main")
    active_endpoints = set()
    for descriptor in evidence.get("active_records", []):
        active = sourced(descriptor)
        endpoint = active.get("endpoint")
        session = active.get("session")
        if (
            endpoint not in endpoints
            or endpoint in active_endpoints
            or not isinstance(session, str)
            or not session
            or Path(descriptor["path"]).name != f"active-{session}.json"
        ):
            raise ValueError("INCOMPLETE_ADMISSION: foreign broker-owned active record")
        active_endpoints.add(endpoint)
        field = (
            "observed_rate_units_per_hour"
            if "priced_rate_units_per_hour" in active
            else "rate_units_per_hour"
        )
        close(finite(active.get(field), "calibrated endpoint rate"), rates[endpoint], "broker rate")
        hardware = dict(sessions)[endpoint].upper()
        registered = {"L4": 1.54, "CPU": 0.08}.get(hardware)
        if registered is None:
            raise ValueError("INCOMPLETE_ADMISSION: unregistered concurrent hardware")
        close(rates[endpoint], registered, "registered endpoint rate")
    if active_endpoints != endpoints:
        raise ValueError("INCOMPLETE_ADMISSION: live endpoints lack broker ownership evidence")
    release = sourced(evidence.get("previous_endpoint_release"))
    if (
        not isinstance(release, dict)
        or release.get("stop_verified") is not True
        or release.get("cleanup_error")
        or not release.get("endpoint")
        or not release.get("job_id")
        or release.get("endpoint") != body.get("previous_endpoint")
        or release["endpoint"] in endpoints
    ):
        raise ValueError("INCOMPLETE_ADMISSION: previous endpoint lacks verified release")
