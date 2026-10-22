"""Prospective L2' placement forecast from paired native ruler receipts.

This is deliberately separate from :mod:`pricing_v2`: the legacy pricing path
does not know about the paired ``all``/``exact_stop`` ruler contract.  This
module admits no dispatch.  It produces a Fable-judge input after the Friday
target-platform rulers have attested the exact-stop decision identity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_served_search.chunks import CHUNK_COUNT, CHUNK_MIX_COUNTS
from research.bl_served_search.exports import digest, export_json, verify_export
from research.bl_served_search.native_provenance import compatible
from research.bl_served_search.ruler import canonical

MODES = ("all", "exact_stop")
MIXES = tuple(CHUNK_MIX_COUNTS)
BATCHES_PER_WORLD = 578
L4_WORKERS = 12
MAC_WORKERS = 2
L4_RATE_FLOOR = 2.18
L4_CHUNK_CAP_MINUTES = 720.0
UNIT_RATE_PER_HOUR = 2.0
UNIT_CAP = 200.0
RESERVE_UNITS = 40.0
SETUP_MINUTES_PER_CHUNK = 30.0
MAC_RESERVE_FRACTION = 0.20
MAC_WALL_CAP_HOURS = 60.0


def _finite_positive(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {field}")
    return float(value)


def _paired_validator(receipt: dict[str, Any]) -> None:
    """Use the native-ruler boundary instead of duplicating its attestations."""
    from research.bl_served_search.native_ruler import validate_paired_receipt

    validate_paired_receipt(receipt)


def _fixture_mixes(receipt: dict[str, Any]) -> dict[str, str]:
    fixtures = receipt.get("fixtures", {}).get("fixtures", [])
    mapping = {row.get("fixture_id"): row.get("mix") for row in fixtures if isinstance(row, dict)}
    if len(mapping) != 9 or set(mapping.values()) != set(MIXES):
        raise ValueError("incomplete paired fixture mix identity")
    return mapping


def _mode_measurement(receipt: dict[str, Any], mode: str) -> dict[str, Any]:
    """Count actual worker seconds by mix from executed branch-cell records."""
    if mode not in MODES:
        raise ValueError("unknown native ruler mode")
    fixture_mixes = _fixture_mixes(receipt)
    cell = receipt["cells"].get(mode)
    if not isinstance(cell, dict):
        raise ValueError("missing paired ruler cell")
    records = cell.get("records")
    if not isinstance(records, list) or len(records) != len(fixture_mixes):
        raise ValueError("incomplete paired cell records")
    by_mix = {mix: {"batches": 0, "worker_seconds": 0.0} for mix in MIXES}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("malformed paired cell record")
        fixture_id = record.get("fixture_id")
        if fixture_id not in fixture_mixes or fixture_id in seen:
            raise ValueError("paired cell fixture identity mismatch")
        seen.add(fixture_id)
        batches = record.get("batches")
        if not isinstance(batches, list) or not batches:
            raise ValueError("paired cell has no executed batches")
        mix = fixture_mixes[fixture_id]
        for batch in batches:
            if not isinstance(batch, dict):
                raise ValueError("malformed paired batch")
            by_mix[mix]["batches"] += 1
            by_mix[mix]["worker_seconds"] += _finite_positive(
                batch.get("worker_wall_seconds"), "worker wall seconds"
            )
    if seen != set(fixture_mixes):
        raise ValueError("paired cell fixture coverage mismatch")
    completed = sum(int(value["batches"]) for value in by_mix.values())
    if completed != cell.get("completed_batches"):
        raise ValueError("paired cell completed-batch counter disagrees")
    if completed != 270:
        raise ValueError("full 270-batch paired ruler required")
    aggregate = _finite_positive(
        cell.get("aggregate_batches_per_second"), "cell aggregate batches per second"
    )
    per_mix = {}
    for mix, value in by_mix.items():
        batches, seconds = value["batches"], value["worker_seconds"]
        if batches != 90:
            raise ValueError("paired ruler requires 90 executed batches per mix")
        per_mix[mix] = {
            "actual_batches": batches,
            "actual_worker_seconds": seconds,
            "actual_batches_per_worker_second": batches / seconds,
        }
    return {
        "mode": mode,
        "actual_completed_batches": completed,
        "per_mix": per_mix,
        "aggregate_actual_batches_per_worker_second": completed
        / sum(value["actual_worker_seconds"] for value in per_mix.values()),
        # This is the native L4 admission R.  It is measured over the cell's
        # own homogeneous slab wall interval, never inferred from Mac timing
        # or reconstructed worker-seconds.
        "cell_aggregate_batches_per_second": aggregate,
    }


def _measure(receipt: dict[str, Any], label: str) -> dict[str, Any]:
    """Validate a complete paired receipt before any forecast arithmetic."""
    _paired_validator(receipt)
    if receipt.get("decision_identity") != {
        "status": "PASS",
        "compared_batches": 270,
        "mismatches": [],
    }:
        raise ValueError("chosen-action identity must pass for all 270 paired batches")
    platform = receipt.get("platform", {})
    if platform.get("label") != label:
        raise ValueError(f"expected {label} paired receipt")
    return {
        "platform": platform,
        "fixture_sha256": {mode: receipt["cells"][mode].get("fixture_sha256") for mode in MODES},
        "source_identity": {mode: receipt["cells"][mode].get("source_identity") for mode in MODES},
        "decision_identity": receipt["decision_identity"],
        "modes": {mode: _mode_measurement(receipt, mode) for mode in MODES},
    }


def _interleaved_lanes(
    per_mix: dict[str, Any], workers: int, counts: dict[str, int]
) -> list[float]:
    """Model deterministic frozen/mixed/scripted interleaving on fixed lanes."""
    sequence: list[str] = []
    for offset in range(counts["frozen"]):
        sequence.extend(("frozen", "mixed"))
        if offset < counts["scripted"]:
            sequence.append("scripted")
    if {mix: sequence.count(mix) for mix in MIXES} != counts:
        raise ValueError("invalid registered mix allocation")
    world_seconds = {
        mix: BATCHES_PER_WORLD / per_mix[mix]["actual_batches_per_worker_second"] for mix in MIXES
    }
    lanes = [0.0] * workers
    for mix in sequence:
        lane = min(range(workers), key=lambda index: (lanes[index], index))
        lanes[lane] += world_seconds[mix]
    return lanes


def _l4_forecast(measurement: dict[str, Any]) -> dict[str, Any]:
    lanes = _interleaved_lanes(measurement["per_mix"], L4_WORKERS, CHUNK_MIX_COUNTS)
    estimate_minutes = SETUP_MINUTES_PER_CHUNK + max(lanes) / 60.0
    two_times_minutes = 2.0 * estimate_minutes
    liability = CHUNK_COUNT * two_times_minutes / 60.0 * UNIT_RATE_PER_HOUR
    rate = measurement["cell_aggregate_batches_per_second"]
    lawful = (
        rate >= L4_RATE_FLOOR
        and two_times_minutes <= L4_CHUNK_CAP_MINUTES
        and liability + RESERVE_UNITS <= UNIT_CAP
    )
    return {
        "measured_native_cell_batches_per_second": rate,
        "l4_rate_floor": L4_RATE_FLOOR,
        "chunks": CHUNK_COUNT,
        "workers_per_chunk": L4_WORKERS,
        "chunk_mix_counts": CHUNK_MIX_COUNTS,
        "batches_per_world_assumption": BATCHES_PER_WORLD,
        "per_mix": measurement["per_mix"],
        "projected_chunk_worker_seconds": lanes,
        "setup_minutes_per_chunk": SETUP_MINUTES_PER_CHUNK,
        "conservative_chunk_minutes": estimate_minutes,
        "two_times_chunk_minutes": two_times_minutes,
        "chunk_cap_minutes": L4_CHUNK_CAP_MINUTES,
        "two_times_liability_units": liability,
        "reserve_units": RESERVE_UNITS,
        "liability_including_reserve_units": liability + RESERVE_UNITS,
        "unit_cap": UNIT_CAP,
        "lawful": lawful,
    }


def _mac_forecast(measurement: dict[str, Any]) -> dict[str, Any]:
    counts = {"frozen": 64, "mixed": 64, "scripted": 32}
    lanes = _interleaved_lanes(measurement["per_mix"], MAC_WORKERS, counts)
    setup_seconds = CHUNK_COUNT * SETUP_MINUTES_PER_CHUNK * 60.0
    base_seconds = max(lanes) + setup_seconds
    reserved_seconds = base_seconds * (1.0 + MAC_RESERVE_FRACTION)
    return {
        "workers": MAC_WORKERS,
        "whole_workload_mix_counts": counts,
        "batches_per_world_assumption": BATCHES_PER_WORLD,
        "per_mix": measurement["per_mix"],
        "projected_worker_seconds": lanes,
        "setup_minutes_per_chunk": SETUP_MINUTES_PER_CHUNK,
        "setup_chunks_assumption": CHUNK_COUNT,
        "setup_seconds": setup_seconds,
        "base_wall_hours": base_seconds / 3600.0,
        "reserve_fraction": MAC_RESERVE_FRACTION,
        "wall_hours_including_reserve": reserved_seconds / 3600.0,
        "wall_cap_hours": MAC_WALL_CAP_HOURS,
        "eligible": reserved_seconds / 3600.0 <= MAC_WALL_CAP_HOURS,
        "paced_displacement_requires_judge_input": True,
    }


def _cross_platform_semantics(mac_receipt: dict[str, Any], l4_receipt: dict[str, Any]) -> None:
    """Fail closed unless every full/stop batch has the same exact semantic result."""
    for mode in MODES:
        mac_records = {
            row["fixture_id"]: row["batches"] for row in mac_receipt["cells"][mode]["records"]
        }
        l4_records = {
            row["fixture_id"]: row["batches"] for row in l4_receipt["cells"][mode]["records"]
        }
        if set(mac_records) != set(l4_records):
            raise ValueError("Mac/L4 paired semantic fixture coverage differs")
        for fixture_id in sorted(mac_records):
            left = [canonical(row.get("semantic")) for row in mac_records[fixture_id]]
            right = [canonical(row.get("semantic")) for row in l4_records[fixture_id]]
            if left != right:
                raise ValueError(f"Mac/L4 paired semantic mismatch: {mode}/{fixture_id}")


def forecast(
    mac_receipt: dict[str, Any], l4_receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Produce separate all/stop forecasts and a recommendation without dispatch authority."""
    mac = _measure(mac_receipt, "mac")
    l4 = _measure(l4_receipt, "l4") if l4_receipt is not None else None
    if l4 is not None and (
        mac["fixture_sha256"] != l4["fixture_sha256"]
        or any(
            not compatible(mac["source_identity"][mode], l4["source_identity"][mode])
            for mode in MODES
        )
    ):
        raise ValueError("Mac/L4 paired receipt provenance differs")
    if l4 is not None:
        _cross_platform_semantics(mac_receipt, l4_receipt)
    modes = {}
    for mode in MODES:
        item = {"mac": _mac_forecast(mac["modes"][mode])}
        if l4 is not None:
            item["l4"] = _l4_forecast(l4["modes"][mode])
        modes[mode] = item
    # The all-branch cell is the identity oracle.  Placement selects the
    # decision-identical exact-stop cell, while preserving both forecasts.
    l4_lawful = l4 is not None and modes["exact_stop"]["l4"]["lawful"]
    mac_eligible = modes["exact_stop"]["mac"]["eligible"]
    recommendation = (
        "L4_LAWFUL" if l4_lawful else "MAC_AMENDMENT_ELIGIBLE" if mac_eligible else "NOT_READY"
    )
    return {
        "schema": "bl-served-search-placement-stop/v1",
        "status": recommendation,
        "recommendation": recommendation,
        "authority": "prospective Fable judge input only; no automatic dispatch or Mac start",
        "decision_identity": mac["decision_identity"],
        "mac_provenance": {
            "platform": mac["platform"],
            "fixture_sha256": mac["fixture_sha256"],
            "source_identity": mac["source_identity"],
            "native_source_identity": mac_receipt["source_identity"],
        },
        "l4_provenance": (
            None
            if l4 is None
            else {
                "platform": l4["platform"],
                "fixture_sha256": l4["fixture_sha256"],
                "source_identity": l4["source_identity"],
                "native_source_identity": l4_receipt["source_identity"],
            }
        ),
        "l4_receipt_present": l4 is not None,
        "source_compatibility": (
            "native_provenance: exact pre-lease ruler and pre-compatibility native hashes only; "
            "original receipt source identities retained"
        ),
        "timing_transfer": "FORBIDDEN: Mac timings never price L4",
        "modes": modes,
        "mac_amendment_requires": [
            "Fable judge ruling",
            "same bank/analyzer/one-look prospective amendment",
            "no paced-work displacement confirmed by judge",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-receipt", type=Path, required=True)
    parser.add_argument("--l4-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    verify_export(args.mac_receipt)
    mac = json.loads(args.mac_receipt.read_text())
    if args.l4_receipt:
        verify_export(args.l4_receipt)
    l4 = json.loads(args.l4_receipt.read_text()) if args.l4_receipt else None
    value = forecast(mac, l4)
    published = {
        **value,
        "input_receipt_sha256": {
            "mac": digest(args.mac_receipt),
            "l4": digest(args.l4_receipt) if args.l4_receipt else None,
        },
    }
    if args.out:
        export_json(args.out, published)
    print(json.dumps(published, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
