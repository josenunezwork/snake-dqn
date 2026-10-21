"""A5-D receipt -> frozen rate-to-action table, wave-based chunk sizing, ceilings, dense R1 yield.

Round 0-D rules (PREREGISTRATION_ROUND0D.md "Sizing"), frozen pre-data:

* r (dense teacher worlds / lane-hour, K=12) = min(r_rows, r_completed), where
  r_rows = 12*3600 / (rho * (R + 5000)); rho = measured teacher process-seconds per
  (rollout row-step + decision) over every completed A5-D world, R = max(T3's
  859,420 rows/world, the 4:3:3 mix-weighted mean over completed throughput
  worlds); r_completed = 12*3600 / mean teacher process-seconds per completed
  throughput world.
* Wave model (deliberate difference 4; Round 0's perfect-packing ``size_round`` is not
  used): the runner gives each pair to one of 12 processes, so a chunk of n pairs runs
  ceil(n/12) whole waves. One wave = 12*60/r + (bare+metric)/60 minutes. Chunks are
  multiples of 12 pairs, the largest with setup + waves*wave <= 180 minutes. Each
  chunk's timeout >= max(estimate+25, 2*estimate, setup + waves*max(slowest, wave) + 25)
  where ``slowest`` is A5-D's slowest measured world (completed or censored at stop).
* Setup minutes are frozen: max(15, ceil(observed A5-D setup/export/cleanup) + 5).
* Straggler schedule (FIX-FIRST 2, N1): every chunk is also charged at its slow-world
  makespan, setup + ceil(n/12)*max(slowest, wave) minutes at 2.0 u/h. The replay cap is
  reserved inside the 120 u total, so chunks are admitted against 120 - replay cap, each
  new chunk at its full unit cap (the supervisor's hard stop), with the other lane's
  attempt in flight at its full cap. Simulating that 2-lane admission on the straggler
  charges must admit every chunk, and straggler total + replay cap must be <= 120.
  The wall forecast uses the same straggler makespans.
* Action: HOLD if fidelity is not PASS/EXPLAINED_TIE/EXPLAINED_PLATFORM or nothing was
  measured; KILL if r < 5; KILL if expected > 120 u, derived ceiling > 120 u, the
  straggler schedule strands under 120 u, or straggler wall > 24 h on the 2 registered
  lanes; otherwise RATIFY. Derived ceiling
  C = 1.2 * expected + (lanes - 1) * max_chunk(cap - expected) must be <= 120; the
  operative study ceiling is then exactly 120 u (JUDGMENT lane L7 total), a hard total.
* Dense R1 (only after a Round 0-D GO): label-yield sizing, <= 200 u, <= 24 h on 3 lanes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.t34_round0.make_manifest import (
    EXPORT_BYTES_PER_WORLD,
    EXPORT_SIZING_FACTOR,
    MAX_ESTIMATE_MINUTES,
    RATE_UNITS_PER_HOUR,
    RESERVE_FRACTION,
    TIMEOUT_GRACE_MINUTES,
)
from research.t34_round0d.common import B8_SHA, read_json, sha256, source_hashes, write_once

SCHEMA = "t34-round0d-a5d/v3"
KILL_WORLDS_PER_HOUR = 5.0
STUDY_UNIT_CAP = 120.0
WALL_HOURS_CAP = 24.0
LANES = 2
OVERRUN_FRACTION = 0.20
T3_ROWS_PER_WORLD = 859_420  # docs/research/t3_ceiling_test_2026-10-15.md:129 (E8)
DECISIONS_PER_WORLD = 5000
MIX_WEIGHTS = {"scripted": 0.4, "frozen": 0.3, "mixed": 0.3}
CPU_SLOWDOWN_FACTOR = 1.0  # bare/metric are measured on the L4 lane itself
EXPORT_BYTES_PER_PAIR_LIMIT = 9 * 1024**2  # Round 0's registered allowance
DISPATCH_GAP_MINUTES = 10  # operator ledger-publication barrier between serial chunks
PROCESSES = 12
MIN_SETUP_MINUTES = 15
SETUP_MARGIN_MINUTES = 5
R1_MAX_FRESH_WORLDS = 560
R1_SEALED_WORLDS = 240
R1_LANES = 3
R1_UNIT_CAP = 200.0
R1_HELDOUT_DECISIVE_ROWS = 300  # L2 held-out minimum (JUDGMENT lane L2)
R1_HELDOUT_DECISIVE_WORLDS = 60
R1_NAMESPACE = "t34-round0d-dense-r1-20261021/v1"
FIDELITY_OK = ("PASS", "EXPLAINED_TIE", "EXPLAINED_PLATFORM")


class SizingRefused(ValueError):
    """Measured evidence does not admit a Round 0-D plan."""


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SizingRefused(f"{name} must be finite")
    if value <= 0:
        raise SizingRefused(f"{name} must be positive")
    return float(value)


def rate_estimates(worlds: list[dict[str, Any]], processes: int = 12) -> dict[str, Any]:
    """Frozen estimator from completed A5-D worlds (never a completion-window count)."""
    done = [w for w in worlds if w.get("teacher_seconds", 0) > 0]
    if not done:
        raise SizingRefused("A5-D completed no dense world: HOLD, nothing measured")
    seconds = sum(w["teacher_seconds"] for w in done)
    steps = sum(w["rollout_row_steps"] + w["decisions"] for w in done)
    rho = seconds / steps
    throughput = [w for w in done if w["kind"] == "throughput"]
    by_mix = {
        mix: [w["rollout_row_steps"] for w in throughput if w["mix"] == mix] for mix in MIX_WEIGHTS
    }
    measured_rows = (
        sum(MIX_WEIGHTS[m] * sum(v) / len(v) for m, v in by_mix.items() if v)
        / sum(MIX_WEIGHTS[m] for m, v in by_mix.items() if v)
        if throughput
        else 0.0
    )
    rows = max(float(T3_ROWS_PER_WORLD), measured_rows)
    r_rows = processes * 3600.0 / (rho * (rows + DECISIONS_PER_WORLD))
    r_completed = (
        processes * 3600.0 / (sum(w["teacher_seconds"] for w in throughput) / len(throughput))
        if throughput
        else None
    )
    rate = min(r_rows, r_completed) if r_completed is not None else r_rows
    return {
        "seconds_per_step": rho,
        "rows_per_world_used": rows,
        "rows_per_world_measured_mix_weighted": measured_rows,
        "rows_per_world_t3": T3_ROWS_PER_WORLD,
        "r_rows": r_rows,
        "r_completed": r_completed,
        "teacher_worlds_per_hour": rate,
        "completed_worlds": len(done),
        "completed_throughput_worlds": len(throughput),
    }


def frozen_setup_minutes(observed_minutes: float) -> int:
    """Setup allowance frozen from A5-D's observed setup/export/cleanup (not an operator choice)."""
    return max(MIN_SETUP_MINUTES, math.ceil(observed_minutes) + SETUP_MARGIN_MINUTES)


def wave_minutes(receipt: dict[str, Any]) -> float:
    """One whole 12-process wave: one dense world plus its bare arm and metric replay."""
    rate = float(receipt["teacher_worlds_per_hour"])
    extra = float(receipt["bare_seconds_per_world"]) + float(receipt["metric_seconds_per_pair"])
    return PROCESSES * 60.0 / rate + extra * float(receipt["cpu_slowdown_factor"]) / 60.0


def chunk_record(start: int, count: int, receipt: dict[str, Any]) -> dict[str, Any]:
    """Wave-quantized chunk forecast, straggler-safe timeout, priced cap and export bound."""
    setup = float(receipt["setup_minutes"])
    wave = wave_minutes(receipt)
    waves = math.ceil(count / PROCESSES)
    workload = setup + waves * wave
    estimate = math.ceil(workload)
    straggler = setup + waves * max(float(receipt["slowest_world_minutes"]), wave)
    timeout = max(
        math.ceil(workload + TIMEOUT_GRACE_MINUTES),
        2 * estimate,
        math.ceil(straggler + TIMEOUT_GRACE_MINUTES),
    )
    unit_cap = timeout * RATE_UNITS_PER_HOUR / 60.0
    return {
        "start": start,
        "count": count,
        "waves": waves,
        "setup_minutes": setup,
        "wave_minutes": wave,
        "workload_minutes": workload,
        "estimate_minutes": estimate,
        "straggler_minutes": straggler,
        "timeout_minutes": timeout,
        "unit_cap": unit_cap,
        "reserve_units": unit_cap * RESERVE_FRACTION,
        "export_bytes": math.ceil(count * EXPORT_BYTES_PER_WORLD * EXPORT_SIZING_FACTOR),
    }


def plan(receipt: dict[str, Any], pairs: int = 240, lanes: int = LANES) -> dict[str, Any]:
    """Wave-based partition + the frozen rate-to-action decision and the 120 u ceiling."""
    validate(receipt)
    rate = float(receipt["teacher_worlds_per_hour"])
    base: dict[str, Any] = {"teacher_worlds_per_hour": rate, "lanes": lanes}
    if rate < KILL_WORLDS_PER_HOUR:
        return dict(base, action="KILL", reason="r < 5 dense worlds per lane-hour")
    wave = wave_minutes(receipt)
    waves_max = math.floor((MAX_ESTIMATE_MINUTES - float(receipt["setup_minutes"])) / wave)
    waves_max = min(
        waves_max,
        math.floor((2 * 1024**3 - 1) / (PROCESSES * EXPORT_BYTES_PER_WORLD * EXPORT_SIZING_FACTOR)),
    )
    if waves_max < 1:
        return dict(base, action="KILL", reason="one 12-pair wave exceeds the 180-min chunk bound")
    chunk_pairs = PROCESSES * waves_max
    chunks = [
        chunk_record(start, min(chunk_pairs, pairs - start), receipt)
        for start in range(0, pairs, chunk_pairs)
    ]
    expected_chunk = [c["workload_minutes"] / 60.0 * RATE_UNITS_PER_HOUR for c in chunks]
    replay = replay_sizing(receipt, chunks)
    expected = sum(expected_chunk) + replay["expected_units"]
    headroom = max(c["unit_cap"] - e for c, e in zip(chunks, expected_chunk))
    derived = (1 + OVERRUN_FRACTION) * expected + (lanes - 1) * headroom
    chunk_ceiling = STUDY_UNIT_CAP - replay["unit_cap"]
    straggler_chunk = [c["straggler_minutes"] / 60.0 * RATE_UNITS_PER_HOUR for c in chunks]
    straggler_total = sum(straggler_chunk) + replay["unit_cap"]
    stranded = admission_schedule(chunks, straggler_chunk, chunk_ceiling, lanes)
    per_lane = [0.0] * lanes
    for index, chunk in enumerate(chunks):
        per_lane[index % lanes] += chunk["straggler_minutes"] + DISPATCH_GAP_MINUTES
    wall_hours = (max(per_lane) + replay["estimate_minutes"] + DISPATCH_GAP_MINUTES) / 60.0
    sizing = dict(
        base,
        chunk_pairs=chunk_pairs,
        chunks=chunks,
        unit_cap=sum(c["unit_cap"] for c in chunks),
        reserve_units=sum(c["reserve_units"] for c in chunks),
        expected_units=expected,
        expected_chunk_units=expected_chunk,
        replay=replay,
        derived_ceiling=derived,
        straggler_chunk_units=straggler_chunk,
        straggler_total_units=straggler_total,
        first_stranded_chunk=stranded,
        chunk_admission_ceiling=chunk_ceiling,
        study_unit_ceiling=STUDY_UNIT_CAP,
        study_unit_cap=STUDY_UNIT_CAP,
        wall_hours_forecast=wall_hours,
        wall_hours_cap=WALL_HOURS_CAP,
        lane_hours_expected=expected / RATE_UNITS_PER_HOUR,
        max_in_flight=lanes,
    )
    if expected > STUDY_UNIT_CAP:
        sizing.update(action="KILL", reason="forecast expected units > 120")
    elif derived > STUDY_UNIT_CAP:
        sizing.update(action="KILL", reason="derived ceiling > 120 u")
    elif stranded is not None or straggler_total > STUDY_UNIT_CAP:
        sizing.update(action="KILL", reason="measured straggler schedule strands under 120 u")
    elif wall_hours > WALL_HOURS_CAP:
        sizing.update(
            action="KILL", reason="straggler wall forecast > 24 h on the registered lanes"
        )
    else:
        sizing.update(
            action="RATIFY",
            reason=(
                "r >= 5, expected and derived ceiling <= 120 u, straggler schedule admits "
                "every chunk within 120 u incl. the replay cap, straggler wall <= 24 h"
            ),
        )
    return sizing


def admission_schedule(
    chunks: list[dict[str, Any]], charges: list[float], ceiling: float, lanes: int = LANES
) -> int | None:
    """First chunk the 2-lane dispatcher would refuse, else None.

    Chunk k is admitted (``admission_spend``) while the previous ``lanes - 1`` chunks are
    still in flight at their full caps and every earlier chunk has settled at its
    ``charges`` entry (the measured slow-world makespan at 2.0 u/h).
    """
    for k, chunk in enumerate(chunks):
        first_live = max(k - (lanes - 1), 0)
        prior: list[dict[str, Any]] = [
            {"elapsed_seconds": charge / RATE_UNITS_PER_HOUR * 3600.0}
            for charge in charges[:first_live]
        ]
        prior += [
            {"elapsed_seconds": None, "unit_cap": c["unit_cap"]} for c in chunks[first_live:k]
        ]
        if not admission_spend(prior, chunk, ceiling)["allowed"]:
            return k
    return None


def replay_sizing(receipt: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """One second-L4 replay job: Round 0's ceil(10%) per chunk, K=12, measured L4 metric cost."""
    sample = sum(math.ceil(c["count"] / 10) for c in chunks)
    per_pair = float(receipt["metric_seconds_per_pair"]) * float(receipt["cpu_slowdown_factor"])
    workload = float(receipt["setup_minutes"]) + math.ceil(sample / PROCESSES) * per_pair / 60
    estimate = math.ceil(workload)
    timeout = max(math.ceil(workload + TIMEOUT_GRACE_MINUTES), 2 * estimate, 60)
    cap = timeout * RATE_UNITS_PER_HOUR / 60.0
    return {
        "sample_pairs": sample,
        "estimate_minutes": estimate,
        "timeout_minutes": timeout,
        "unit_cap": cap,
        "reserve_units": RESERVE_FRACTION * cap,
        "expected_units": workload / 60.0 * RATE_UNITS_PER_HOUR,
    }


def validate(receipt: dict[str, Any]) -> dict[str, Any]:
    """Refuse a missing, held, or out-of-contract A5-D receipt."""
    if receipt.get("schema") != SCHEMA:
        raise SizingRefused("A5-D receipt schema differs")
    if receipt.get("fidelity_status") not in FIDELITY_OK:
        raise SizingRefused("A5-D fidelity is not PASS/EXPLAINED_TIE/EXPLAINED_PLATFORM: HOLD")
    _positive(receipt.get("teacher_worlds_per_hour"), "teacher_worlds_per_hour")
    for key in (
        "bare_seconds_per_world",
        "metric_seconds_per_pair",
        "setup_minutes",
        "slowest_world_minutes",
    ):
        _positive(receipt.get(key), key)
    if receipt.get("cpu_slowdown_factor") != CPU_SLOWDOWN_FACTOR:
        raise SizingRefused("A5-D receipt must use measured L4 costs (factor 1.0)")
    exported = _positive(receipt.get("measured_export_bytes_per_pair"), "export bytes")
    if exported > EXPORT_BYTES_PER_PAIR_LIMIT:
        raise SizingRefused("measured exports exceed the 9 MiB/pair allowance; re-size first")
    digest = hashlib.sha256(json.dumps(receipt.get("evidence"), sort_keys=True).encode())
    if digest.hexdigest() != receipt.get("evidence_sha256"):
        raise SizingRefused("A5-D evidence digest differs")
    return receipt


def admission_spend(
    prior: list[dict[str, Any]], chunk: dict[str, Any], ceiling: float
) -> dict[str, Any]:
    """Refuse a chunk whose capped spend would cross the chunk admission ceiling.

    ``ceiling`` is the registered ``chunk_admission_ceiling`` (120 u minus the reserved
    replay cap). Settled attempts are charged their whole-session elapsed time at
    2.0 u/h; an in-flight or receipt-less attempt, and the chunk being admitted, are
    charged their full unit caps (the supervisor's hard stop), so 120 u is a hard total.
    ``infeasible_on_settled_charges`` reports whether the refusal stands without any
    in-flight charge (-> INCONCLUSIVE).
    """
    settled = unsettled = 0.0
    for item in prior:
        elapsed = item.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0:
            settled += float(elapsed) / 3600.0 * RATE_UNITS_PER_HOUR
        else:
            unsettled += float(item.get("unit_cap", ceiling))
    expected = float(chunk["workload_minutes"]) / 60.0 * RATE_UNITS_PER_HOUR
    cap = float(chunk["unit_cap"])
    return {
        "charged_units": settled + unsettled,
        "settled_units": settled,
        "chunk_expected_units": expected,
        "chunk_charged_units": cap,
        "ceiling": ceiling,
        "allowed": settled + unsettled + cap <= ceiling,
        "infeasible_on_settled_charges": settled + cap > ceiling,
    }


def dense_r1_plan(
    teacher_worlds_per_hour: float, decisive_rows_per_world: float, decisive_world_fraction: float
) -> dict[str, Any]:
    """Label-yield dense R1 sizing; executes only after a published Round 0-D GO.

    Yield (decisive rows/world) and the decisive-bearing world fraction are measured
    on the 240 sealed teacher worlds (labels.py, after GO). N_fresh is the smallest
    whole-world count whose expectation meets the L2 held-out minimum (>= 300
    decisive rows and >= 60 decisive-bearing worlds), capped at 560; funded only
    within 200 u and 24 h wall on 3 lanes. Fitting additionally requires the
    ratified Claim A loss amendment and the PB-disposition ordering clause.
    """
    rate = _positive(teacher_worlds_per_hour, "teacher_worlds_per_hour")
    if decisive_rows_per_world <= 0 or decisive_world_fraction <= 0:
        return {"fundable": False, "reason": "zero measured decisive yield", "fresh_worlds": 0}
    needed = max(
        math.ceil(R1_HELDOUT_DECISIVE_ROWS / decisive_rows_per_world),
        math.ceil(R1_HELDOUT_DECISIVE_WORLDS / decisive_world_fraction),
    )
    fresh = min(R1_MAX_FRESH_WORLDS, needed)
    lane_hours = fresh / rate
    units = lane_hours * RATE_UNITS_PER_HOUR
    wall = lane_hours / R1_LANES
    reachable = needed <= R1_MAX_FRESH_WORLDS
    fundable = reachable and units <= R1_UNIT_CAP and wall <= WALL_HOURS_CAP
    return {
        "namespace": R1_NAMESPACE,
        "fresh_worlds": fresh,
        "fresh_worlds_needed_for_heldout_minimum": needed,
        "sealed_round0d_teacher_worlds": R1_SEALED_WORLDS,
        "lane_hours": lane_hours,
        "wall_hours_on_3_lanes": wall,
        "expected_units": units,
        "unit_cap": R1_UNIT_CAP,
        "fundable": fundable,
        "reason": (
            "fundable"
            if fundable
            else (
                "held-out minimum unreachable within 560 fresh worlds"
                if not reachable
                else "exceeds 200 u or 24 h on 3 lanes"
            )
        ),
        "fit_preconditions": [
            "ratified Claim A loss amendment (JUDGMENT §7a) shown to move decisive argmaxes",
            "dense R1 ordering clause: labels only until the Path B disposition is published",
        ],
    }


def slowest_world_minutes(data: dict[str, Any]) -> float:
    """Slowest A5-D world in minutes: completed full pairs, fidelity worlds, and censored
    in-progress items at stop (their elapsed time is a lower bound and still counts)."""
    values = [
        (w.get("teacher_seconds", 0) + w.get("bare_seconds", 0) + w.get("metric_seconds", 0)) / 60.0
        for w in data["worlds"]
    ]
    values += [item["elapsed_seconds"] / 60.0 for item in data.get("in_progress_at_stop", [])]
    if not values:
        raise SizingRefused("A5-D measured no world: HOLD")
    return max(values)


def resolved_fidelity(
    data: dict[str, Any], resolutions: list[Path], receipt: dict[str, Any] | None = None
) -> str:
    """Overall fidelity after the pre-registered Mac re-run resolution of HOLD worlds.

    A resolution record counts only if it is bound to this A5-D run: its
    ``l4_export_sha256`` equals the supervisor-verified receipt export
    ``fidelity-<index>.json`` for that world, its ``checkpoint_sha256`` is B8 s3, the Mac
    reproduced T3 exactly (``mac_reproduces_t3`` with no differences), and it used the
    registered epsilon. Anything else leaves the world HOLD.
    """
    from research.t34_round0d.a5d import RESOLUTION_SCHEMA, TIE_EPSILON

    status = data["fidelity"]["status"]
    if status != "HOLD":
        return status
    exports = (receipt or {}).get("exports", [])
    records = [read_json(path) for path in resolutions]
    for index, world in enumerate(data["fidelity"]["worlds"]):
        if world["status"] in ("PASS", "EXPLAINED_TIE"):
            continue
        name = f"fidelity-{index:02d}.json"
        bound = {
            e.get("sha256")
            for e in exports
            if str(e.get("path", "")).rsplit("/", 1)[-1] == name and e.get("sha256")
        }
        if not any(
            record.get("schema") == RESOLUTION_SCHEMA
            and record.get("world_id") == world["world_id"]
            and record.get("status") == "EXPLAINED_PLATFORM"
            and record.get("l4_export_sha256") in bound
            and record.get("checkpoint_sha256") == B8_SHA
            and record.get("mac_reproduces_t3") is True
            and not record.get("mac_differences")
            and record.get("tie_epsilon") == TIE_EPSILON
            for record in records
        ):
            return "HOLD"
    return "EXPLAINED_PLATFORM"


def build(a5d_benchmark: Path, preflight: Path, resolutions: list[Path] | None = None) -> dict:
    """Bind the supervisor-verified A5-D export (L4 costs) and the source-current A6."""
    resolutions = list(resolutions or [])
    data = read_json(a5d_benchmark)
    job = a5d_benchmark.parents[2]  # <job>/exports/a5d/benchmark.json
    receipt = read_json(job / "receipt.json")
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or not any(
            entry.get("path", "").endswith("benchmark.json")
            and entry.get("sha256") == sha256(a5d_benchmark)
            for entry in receipt.get("exports", [])
        )
        or data.get("schema") != "t34-round0d-a5d-benchmark/v3"
        or data.get("processes") != PROCESSES
    ):
        raise SizingRefused("A5-D requires a verified L4 PASS benchmark with K=12 processes")
    observed_setup = max(0.0, (receipt["elapsed_seconds"] - data["elapsed_seconds"]) / 60)
    setup_minutes = frozen_setup_minutes(observed_setup)
    chunk_path = preflight / "chunk.json"
    chunk = read_json(chunk_path)
    if chunk.get("preflight") is not True or chunk.get("status") != "complete":
        raise SizingRefused("A6 must be a complete excluded Mac preflight")
    rates = rate_estimates(data["worlds"])
    paired = [w for w in data["worlds"] if w["kind"] == "throughput" and w.get("bare_seconds")]
    if not paired:
        raise SizingRefused("A5-D measured no paired bare/metric cost: HOLD")
    bare = sum(w["bare_seconds"] for w in paired) / len(paired)
    metric = sum(w["metric_seconds"] for w in paired) / len(paired)
    evidence = {
        "a5d_benchmark": {"path": str(a5d_benchmark.resolve()), "sha256": sha256(a5d_benchmark)},
        "a5d_receipt_sha256": sha256(job / "receipt.json"),
        "a5d_rates": rates,
        "a5d_fidelity": data["fidelity"],
        "fidelity_resolutions": [{"path": str(p), "sha256": sha256(p)} for p in resolutions],
        "a6_chunk_sha256": sha256(chunk_path),
        "observed_setup_export_cleanup_minutes": observed_setup,
        "setup_minutes_rule": "max(15, ceil(observed) + 5)",
    }
    payload = {
        "schema": SCHEMA,
        "teacher_worlds_per_hour": rates["teacher_worlds_per_hour"],
        "fidelity_status": resolved_fidelity(data, resolutions, receipt),
        "bare_seconds_per_world": bare,
        "metric_seconds_per_pair": metric,
        "slowest_world_minutes": slowest_world_minutes(data),
        "cpu_slowdown_factor": CPU_SLOWDOWN_FACTOR,
        "setup_minutes": setup_minutes,
        "measured_export_bytes_per_pair": float(data["export_bytes_per_pair_upper"]),
        "a6_measurement_sha256": sha256(chunk_path),
        "a6_source_sha256": hashlib.sha256(
            json.dumps(chunk["run_identity"]["source_hashes"], sort_keys=True).encode()
        ).hexdigest(),
        "current_source_sha256": hashlib.sha256(
            json.dumps(source_hashes(), sort_keys=True).encode()
        ).hexdigest(),
        "evidence": evidence,
    }
    payload["evidence_sha256"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True).encode()
    ).hexdigest()
    payload["plan"] = plan(payload)
    return payload


def estimated_receipt(teacher_worlds_per_hour: float, setup_minutes: int = 15) -> dict[str, Any]:
    """ESTIMATED placeholder receipt; never admits dispatch.

    Placeholder L4 costs: Round 0-D's own A6 (Mac, H500x10: bare 20.2 s/world, metric
    5.4 s/pair) times the ops panel's 6-9x L4 per-process penalty (9x), rounded up; the
    slowest world is 1.3x the mean wave (the reviewer's max-of-n straggler factor).
    """
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "fidelity_status": "PASS",
        "teacher_worlds_per_hour": teacher_worlds_per_hour,
        "bare_seconds_per_world": 190.0,
        "metric_seconds_per_pair": 50.0,
        "cpu_slowdown_factor": CPU_SLOWDOWN_FACTOR,
        "setup_minutes": setup_minutes,
        "measured_export_bytes_per_pair": 4 * 1024**2,
        "evidence": {"estimated": True},
    }
    receipt["slowest_world_minutes"] = 1.3 * wave_minutes(receipt)
    receipt["evidence_sha256"] = hashlib.sha256(
        json.dumps(receipt["evidence"], sort_keys=True).encode()
    ).hexdigest()
    return receipt


def estimated_plan(teacher_worlds_per_hour: float, setup_minutes: int = 15) -> dict:
    """ESTIMATED draft table row; never admits dispatch."""
    receipt = estimated_receipt(teacher_worlds_per_hour, setup_minutes)
    return {"status": "ESTIMATED_NOT_DISPATCHABLE", "plan": plan(receipt)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5d-benchmark", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--fidelity-resolution", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.a5d_benchmark, args.preflight_dir, args.fidelity_resolution)
    write_once(args.out, payload)
    print(json.dumps({k: payload["plan"].get(k) for k in ("action", "reason")}))


if __name__ == "__main__":
    main()
