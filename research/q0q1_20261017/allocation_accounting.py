"""Create a sibling allocation-lifetime upper bound without rewriting receipts.

Missing assignment starts and releases must be supplied as explicit conservative
bounds, never inferred from a claim or a stale/idle heartbeat. The original A3
unit envelope is frozen: correcting debit cannot increase any credit limit.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping

from research.q0q1_20261017 import q0_accounting as old

SCHEMA = "q0q1-allocation-upper-bound/v1"
FROZEN_Q0_UNITS = 68.63035524611341


def validate_dispatch_horizon(value: Mapping[str, Any], *, now: dt.datetime | None = None) -> None:
    """Refuse generation after the assumed first-dispatch time has elapsed."""
    allocation = value.get("allocation_accounting", {})
    dispatch = old._timestamp(allocation.get("dispatch_at"), "dispatch_at")
    observed = now if now is not None else dt.datetime.now(dt.timezone.utc)
    if observed.tzinfo is None or observed > dispatch:
        raise old.AccountingError("allocation dispatch assumption expired; reconciliation required")


def _equal(actual: Any, expected: float, label: str) -> None:
    if not old._same_number(actual, expected):
        raise old.AccountingError(f"allocation accounting invalid {label}")


def validate_accounting(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute lifetime charges and return the unchanged A3 credit envelope."""
    allocation = value.get("allocation_accounting")
    if not isinstance(allocation, Mapping) or allocation.get("schema") != SCHEMA:
        raise old.AccountingError("A5 requires allocation-inclusive upper-bound accounting")
    if allocation.get("status") != "COMPLETE_WITH_CONSERVATIVE_BOUNDS":
        raise old.AccountingError("allocation upper bound is incomplete")
    envelope = old.validate_a3_envelope(
        {"schema": old.SCHEMA, "q0_units": FROZEN_Q0_UNITS, "a3_envelope": value.get("a3_envelope")}
    )
    dispatch = old._timestamp(allocation.get("dispatch_at"), "dispatch_at")
    through = old._timestamp(allocation.get("idle_through"), "idle_through")
    if through != dispatch + dt.timedelta(minutes=30):
        raise old.AccountingError("L4 idle bound must be dispatch time plus 30 minutes")
    rows = allocation.get("runtimes")
    if not isinstance(rows, list) or not rows:
        raise old.AccountingError("allocation upper bound lacks runtime inventory")
    ids: set[str] = set()
    hardware_by_worker: dict[str, str] = {}
    units = gpu = cpu = 0.0
    rates = allocation.get("rates_units_per_hour", {})
    retained = 0
    covered: dict[tuple[str, int, str], tuple[str, Mapping[str, Any]]] = {}
    for row in rows:
        worker = row.get("worker_id")
        if not isinstance(worker, str) or worker in ids:
            raise old.AccountingError("duplicate or invalid allocation worker")
        ids.add(worker)
        hardware = row.get("hardware")
        if hardware not in {"A100", "L4", "cpu"}:
            raise old.AccountingError("unknown allocation hardware")
        hardware_by_worker[worker] = hardware
        start = old._timestamp(row.get("start_bound"), "start_bound")
        end = old._timestamp(row.get("end_bound"), "end_bound")
        hours = (end - start).total_seconds() / 3600
        if hours <= 0 or not row.get("assumptions") or not row.get("evidence"):
            raise old.AccountingError("runtime needs positive lifetime, assumptions and evidence")
        rate = old._finite(rates.get(hardware), "allocation rate", positive=True)
        _equal(row.get("hours_upper_bound"), hours, "runtime hours")
        _equal(row.get("charged_units_upper_bound"), hours * rate, "runtime units")
        for claim in row.get("claims", []):
            key = (claim.get("id"), claim.get("generation"), claim.get("token"))
            if (
                not isinstance(key[0], str)
                or not isinstance(key[1], int)
                or not isinstance(key[2], str)
                or key in covered
            ):
                raise old.AccountingError("invalid or duplicate allocation claim identity")
            covered[key] = (worker, claim)
            claim_start = old._timestamp(claim.get("created_at"), "claim.created_at")
            claim_end = old._timestamp(claim.get("end_at"), "claim.end_at")
            if claim_start < start or claim_end > end or claim_end < claim_start:
                raise old.AccountingError("allocation lifetime does not contain its attempt")
        if row.get("retained_l4") is True:
            retained += 1
            if hardware != "L4" or end != through:
                raise old.AccountingError(
                    "retained L4 must charge through dispatch plus 30 minutes"
                )
        units += hours * rate
        if hardware == "cpu":
            cpu += hours
        else:
            gpu += hours
    if retained != 1:
        raise old.AccountingError("allocation inventory must name exactly one retained L4")
    historical = allocation.get("historical_attempts")
    if not isinstance(historical, list) or not historical:
        raise old.AccountingError("allocation inventory lacks historical attempt coverage")
    for attempt in historical:
        key = (attempt.get("id"), attempt.get("generation"), attempt.get("token"))
        if key not in covered or covered[key][0] != attempt.get("worker_id"):
            raise old.AccountingError("allocation inventory omits a historical worker or claim")
        if hardware_by_worker[attempt["worker_id"]] != attempt.get("hardware"):
            raise old.AccountingError("allocation hardware differs from historical attempt")
        _, claim = covered[key]
        if claim.get("sha256") != attempt.get("claim_sha256"):
            raise old.AccountingError("allocation claim differs from historical evidence")
        historical_start = old._timestamp(attempt.get("claim_started_at"), "historical claim")
        historical_end = historical_start + dt.timedelta(
            seconds=old._finite(attempt.get("elapsed_seconds"), "historical elapsed", positive=True)
        )
        if (
            old._timestamp(claim.get("created_at"), "claim start") != historical_start
            or old._timestamp(claim.get("end_at"), "claim end") != historical_end
        ):
            raise old.AccountingError("allocation claim timing differs from historical attempt")
    if "attempts" in value:

        def identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
            return (row.get("id"), row.get("generation"), row.get("token"), row.get("claim_sha256"))

        if {identity(row) for row in value["attempts"]} != {identity(row) for row in historical}:
            raise old.AccountingError("allocation historical attempts differ from original receipt")
    for name, total in (("q0_units", units), ("q0_gpu_hours", gpu), ("q0_cpu_hours", cpu)):
        _equal(value.get(name), total, name)
    return envelope


def build_accounting(*, baseline_path: Path, bounds_path: Path, drive_root: Path) -> dict[str, Any]:
    """Bind every attempt, heartbeat snapshot, and explicit lifetime bound."""
    baseline = old._read_object(baseline_path, "historical accounting")
    old.validate_a3_envelope(baseline)
    _equal(baseline.get("q0_units"), FROZEN_Q0_UNITS, "frozen historical Q0 debit")
    bounds = old._read_object(bounds_path, "allocation bounds")
    rows = copy.deepcopy(bounds["runtimes"])
    for row in rows:
        for evidence in row.get("evidence", []):
            path = Path(evidence["path"])
            if old.sha256_file(path) != evidence["sha256"]:
                raise old.AccountingError(f"allocation evidence changed: {path}")
    by_worker = {row["worker_id"]: row for row in rows}
    if len(by_worker) != len(rows):
        raise old.AccountingError("duplicate runtime bounds")
    for row in rows:
        row["claims"] = []
    # All claims (including host contracts) on a Q0 runtime are charged once
    # inside that runtime's lifetime; there is no claim-duration double debit.
    old_attempts = {
        (row["id"], row["generation"], row["token"]): row for row in baseline["attempts"]
    }
    seen = set()
    historical_attempts = []
    for path in sorted((drive_root / "queue/claimed").glob("*/*.json")):
        claim = old._read_object(path, "queue claim")
        worker = claim.get("worker_id")
        if worker not in by_worker:
            if str(claim.get("id", "")).startswith(("q0-", "q0b-")):
                raise old.AccountingError(f"unbounded Q0 worker: {worker}")
            continue
        row = by_worker[worker]
        if claim.get("capabilities", {}).get("gpu") != row["hardware"]:
            raise old.AccountingError("runtime hardware differs from claim")
        key = (claim["id"], claim["generation"], claim["token"])
        seen.add(key)
        prior = old_attempts.get(key)
        if prior:
            if old.sha256_file(path) != prior["claim_sha256"]:
                raise old.AccountingError("historical claim evidence changed")
            historical_attempts.append({**prior, "worker_id": worker})
        start = old._timestamp(claim["created_at"], "claim start")
        end = start + dt.timedelta(seconds=prior["elapsed_seconds"]) if prior else start
        row["claims"].append(
            {
                "id": claim["id"],
                "generation": claim["generation"],
                "token": claim["token"],
                "created_at": claim["created_at"],
                "end_at": end.isoformat(),
                "path": str(path),
                "sha256": old.sha256_file(path),
            }
        )
    if set(old_attempts) - seen:
        raise old.AccountingError("historical attempts missing from allocation inventory")
    units = gpu = cpu = 0.0
    rates = baseline["pricing"]["rates_units_per_hour"]
    for row in rows:
        hours = (
            old._timestamp(row["end_bound"], "end") - old._timestamp(row["start_bound"], "start")
        ).total_seconds() / 3600
        row["hours_upper_bound"] = hours
        row["charged_units_upper_bound"] = hours * rates[row["hardware"]]
        row["historical_claim_hours"] = sum(
            (
                old._timestamp(c["end_at"], "end") - old._timestamp(c["created_at"], "start")
            ).total_seconds()
            / 3600
            for c in row["claims"]
        )
        if row["hardware"] == "cpu":
            cpu += hours
        else:
            gpu += hours
        units += row["charged_units_upper_bound"]
    result = copy.deepcopy(baseline)
    result.update(q0_units=units, q0_gpu_hours=gpu, q0_cpu_hours=cpu)
    result["allocation_accounting"] = {
        "schema": SCHEMA,
        "status": "COMPLETE_WITH_CONSERVATIVE_BOUNDS",
        "dispatch_at": bounds["dispatch_at"],
        "idle_through": bounds["idle_through"],
        "study_window_start": bounds["study_window_start"],
        "baseline_path": str(baseline_path),
        "baseline_sha256": old.sha256_file(baseline_path),
        "bounds_path": str(bounds_path),
        "bounds_sha256": old.sha256_file(bounds_path),
        "rates_units_per_hour": rates,
        "runtimes": rows,
        "historical_claim_totals": {
            name: baseline[name] for name in ("q0_units", "q0_gpu_hours", "q0_cpu_hours")
        },
        "historical_attempts": historical_attempts,
        "limitation": bounds["limitation"],
    }
    result["derivation"] = (
        "Sum full per-runtime allocation bounds, including idle/setup/failure/copy/cleanup. "
        "Claims remain provenance only and are not added a second time. Missing assignment "
        "records use disclosed study-window bounds, which may overcharge overlapping uncertainty. "
        "A3 unit caps remain frozen; no retroactive validation of historical breaches."
    )
    validate_accounting(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--bounds", required=True, type=Path)
    parser.add_argument("--drive-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite allocation accounting: {args.out}")
    result = build_accounting(
        baseline_path=args.baseline, bounds_path=args.bounds, drive_root=args.drive_root
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: result[name] for name in ("q0_units", "q0_gpu_hours", "q0_cpu_hours")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
