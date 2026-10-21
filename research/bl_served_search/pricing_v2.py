"""PLAN_v2 sizing only: per-mix rates, process waves and bounded liabilities."""

from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any

from research.bl_served_search import ruler
from research.bl_served_search.chunks import CHUNK_COUNT, CHUNK_MIX_COUNTS
from research.bl_served_search.provenance_v2 import compatible_source

FALLBACK_NOT_BEFORE = datetime(2026, 10, 22, 19, tzinfo=timezone.utc)
UNIT_CAP = 200.0
RESERVE = 40.0
RATE = 2.0
WORKERS = 12
BATCHES_PER_WORLD = 578
OVERHEAD_MINUTES = 30.0  # bootstrap + assignment parity + export, per assignment


def rates(receipt: dict[str, Any], fixtures: dict[str, Any], label: str) -> dict[str, float]:
    """Reconstruct worker goodput from all 270 real Y1 batch records."""
    if (
        receipt.get("schema") != ruler.RECEIPT_SCHEMA
        or receipt.get("status") not in {"RULER_PASS", "NOT_DISPATCHABLE"}
        or receipt.get("platform", {}).get("label") != label
        or receipt.get("min_batches_per_fixture", 0) < 30
        or receipt.get("fixture_sha256") != ruler.sha256_bytes(ruler.canonical(fixtures).encode())
        or not compatible_source(receipt.get("source_identity", {}), ruler.source_identity())
        or receipt.get("operator_spec")
        != "research.bl_served_search.ruler_operator:create_operator"
    ):
        raise ValueError("full source/fixture-bound Y1 receipt required")
    identities = {row["fixture_id"]: row for row in ruler.validate_fixtures(fixtures)}
    records = receipt.get("records", [])
    if len(records) != 9 or {row["fixture_id"] for row in records} != set(identities):
        raise ValueError("nine unique Y1 fixture records required")
    sums = {mix: [0, 0.0] for mix in CHUNK_MIX_COUNTS}
    for row in records:
        batches = row.get("batches", [])
        if len(batches) < 30:
            raise ValueError("30 actual batches per fixture required")
        mix = identities[row["fixture_id"]]["mix"]
        for batch in batches:
            seconds = batch.get("worker_wall_seconds")
            if (
                type(seconds) not in (int, float)
                or not math.isfinite(seconds)
                or seconds <= 0
                or batch.get("semantic", {}).get("fired") is not True
            ):
                raise ValueError("invalid actual Y1 worker timing")
            sums[mix][0] += 1
            sums[mix][1] += seconds
    if receipt.get("completed_batches") != sum(item[0] for item in sums.values()):
        raise ValueError("Y1 completed batch count disagrees")
    result = {mix: count / seconds for mix, (count, seconds) in sums.items()}
    if label == "mac" and receipt.get("platform", {}).get("system") != "Darwin":
        raise ValueError("Mac sizing requires Mac receipt")
    return result


def forecast(mac: dict, fixtures: dict, l4: dict | None = None) -> dict[str, Any]:
    """Never clip: refuse slow receipts whose two-wave price exceeds the cap."""
    mac_rates = rates(mac, fixtures, "mac")
    if l4 is None:
        effective = {mix: value / 2 for mix, value in mac_rates.items()}
        basis = "MAC_Y1_X2_SIZING_ONLY"
    else:
        # Reconcile semantics/hardware with the existing ruler validator. Only
        # the explicitly permitted old partition identity is normalized in-memory.
        values = []
        for receipt in (mac, l4):
            if not compatible_source(receipt.get("source_identity", {}), ruler.source_identity()):
                raise ValueError("ruler semantic source drift")
            value = copy.deepcopy(receipt)
            value["source_identity"] = ruler.source_identity()
            values.append(value)
        checked = ruler.verify_qualification(*values)
        if checked["status"] == "INVALID":
            raise ValueError("Y1-L4 semantic mismatch")
        effective = rates(l4, fixtures, "l4")
        basis = "Y1_L4_MEASURED"
    world_seconds = {mix: BATCHES_PER_WORLD / value for mix, value in effective.items()}
    # Model the exact interleaved submission order on twelve worker lanes.
    # This includes tail imbalance, unlike total worker-seconds / 12.
    sequence = []
    for offset in range(CHUNK_MIX_COUNTS["frozen"]):
        sequence.extend(("frozen", "mixed"))
        if offset < CHUNK_MIX_COUNTS["scripted"]:
            sequence.append("scripted")
    lanes = [0.0] * WORKERS
    for mix in sequence:
        lane = min(range(WORKERS), key=lambda index: (lanes[index], index))
        lanes[lane] += world_seconds[mix]
    estimate = OVERHEAD_MINUTES + max(lanes) / 60
    timeout = 2 * estimate
    liability = CHUNK_COUNT * timeout / 60 * RATE
    fits = timeout <= 720 and liability <= UNIT_CAP - RESERVE
    return {
        "schema": "bl-served-search-forecast/v3",
        "status": "FITS" if fits else "NOT_DISPATCHABLE",
        "basis": basis,
        "fallback_not_before_utc": FALLBACK_NOT_BEFORE.isoformat(),
        "timing_evidence": "sizing only; no Mac/Colab timing qualification transfer",
        "worlds": 160,
        "chunks": CHUNK_COUNT,
        "worlds_per_chunk": 20,
        "chunk_mix_counts": CHUNK_MIX_COUNTS,
        "workers": WORKERS,
        "batches_per_world_assumption": BATCHES_PER_WORLD,
        "total_batches": 92480,
        "per_mix_effective_batches_per_worker_second": effective,
        "per_mix_world_seconds": world_seconds,
        "projected_worker_seconds": lanes,
        "chunk_setup_parity_export_minutes": OVERHEAD_MINUTES,
        "conservative_allocation_minutes": estimate,
        "timeout_minutes": timeout,
        "two_times_conservative_minutes": timeout,
        "total_estimated_allocation_minutes": CHUNK_COUNT * estimate,
        "total_two_times_minutes": CHUNK_COUNT * timeout,
        "two_times_liability_units": liability,
        "unit_cap": UNIT_CAP,
        "reserve_units": RESERVE,
        "chunk_unit_cap": UNIT_CAP / CHUNK_COUNT,
        "chunk_reserve_units": RESERVE / CHUNK_COUNT,
        "rate_units_per_hour": RATE,
        "liability_plus_reserve_units": liability + RESERVE,
        "absolute_wall_cap_minutes": 720,
    }


def fallback_open(value: dict, now: datetime | None = None) -> bool:
    """The absence of Y1-L4 opens the fallback only at the judge's deadline."""
    return (
        value["basis"] != "MAC_Y1_X2_SIZING_ONLY"
        or (now or datetime.now(timezone.utc)) >= FALLBACK_NOT_BEFORE
    )
