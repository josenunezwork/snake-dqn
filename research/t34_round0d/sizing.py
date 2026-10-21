"""A5-D receipt -> frozen rate-to-action table, Round 0 chunk sizing, ceilings, dense R1 yield.

Round 0's chunk formula (``research.t34_round0.make_manifest.size_round``) is
imported unchanged: chunk n is the largest multiple of ten with
setup + n*(3600/r + f*(bare+metric)/12)/60 <= 180 minutes, timeout >=
max(estimate+25, 2*estimate), priced at 2.0 u/h with a held 20% reserve.

Round 0-D rules (PREREGISTRATION_ROUND0D.md "Rate-to-action table"), frozen pre-data:

* r (dense teacher worlds / lane-hour, K=12) = min(r_rows, r_completed), where
  r_rows = 12*3600 / (rho * (R + 5000)); rho = measured teacher process-seconds per
  (rollout row-step + decision) over every completed A5-D world, R = max(T3's
  859,420 rows/world, the 4:3:3 mix-weighted mean over completed throughput
  worlds); r_completed = 12*3600 / mean teacher process-seconds per completed
  throughput world (reported; biased fast because early deaths finish first).
* Bare and metric costs are the measured L4 per-process means (factor 1.0).
* Action: KILL if r < 5; KILL if forecast expected units > 120 or forecast wall
  > 24 h on the 2 registered lanes; HOLD if fidelity is not PASS/EXPLAINED_TIE or
  nothing was measured; otherwise RATIFY with the study charged-unit ceiling
  C = 1.2 * expected + (lanes - 1) * max_chunk(cap - expected), derived from
  the forecast so that the last chunk is never stranded by in-flight charging.
* Dense R1 (only after a Round 0-D GO): label-yield sizing, <= 200 u, <= 24 h on 3 lanes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.t34_round0.make_manifest import RATE_UNITS_PER_HOUR, size_round
from research.t34_round0d.common import read_json, sha256, source_hashes, write_once

SCHEMA = "t34-round0d-a5d/v2"
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
REPLAY_SAMPLE_PAIRS = 24
R1_MAX_FRESH_WORLDS = 560
R1_SEALED_WORLDS = 240
R1_LANES = 3
R1_UNIT_CAP = 200.0
R1_HELDOUT_DECISIVE_ROWS = 300  # L2 held-out minimum (JUDGMENT lane L2)
R1_HELDOUT_DECISIVE_WORLDS = 60
R1_NAMESPACE = "t34-round0d-dense-r1-20261021/v1"
FIDELITY_OK = ("PASS", "EXPLAINED_TIE")


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


def plan(receipt: dict[str, Any], pairs: int = 240, lanes: int = LANES) -> dict[str, Any]:
    """Round 0 chunk partition + the frozen rate-to-action decision and derived ceiling."""
    validate(receipt)
    rate = float(receipt["teacher_worlds_per_hour"])
    base: dict[str, Any] = {"teacher_worlds_per_hour": rate, "lanes": lanes}
    if rate < KILL_WORLDS_PER_HOUR:
        return dict(base, action="KILL", reason="r < 5 dense worlds per lane-hour")
    sizing = size_round(
        pairs,
        teacher_worlds_per_hour=rate,
        bare_seconds_per_world=receipt["bare_seconds_per_world"],
        metric_seconds_per_pair=receipt["metric_seconds_per_pair"],
        cpu_slowdown_factor=receipt["cpu_slowdown_factor"],
        setup_minutes=receipt["setup_minutes"],
    )
    chunks = sizing["chunks"]
    expected_chunk = [float(c["workload_minutes"]) / 60.0 * RATE_UNITS_PER_HOUR for c in chunks]
    replay = replay_sizing(receipt)
    expected = sum(expected_chunk) + replay["expected_units"]
    headroom = max(float(c["unit_cap"]) - e for c, e in zip(chunks, expected_chunk))
    ceiling = (1 + OVERRUN_FRACTION) * expected + (lanes - 1) * headroom
    per_lane = [0.0] * lanes
    for index, chunk in enumerate(chunks):
        per_lane[index % lanes] += float(chunk["workload_minutes"]) + DISPATCH_GAP_MINUTES
    wall_hours = (max(per_lane) + replay["estimate_minutes"] + DISPATCH_GAP_MINUTES) / 60.0
    sizing.update(
        base,
        expected_units=expected,
        expected_chunk_units=expected_chunk,
        replay=replay,
        study_unit_ceiling=ceiling,
        study_unit_cap=STUDY_UNIT_CAP,
        wall_hours_forecast=wall_hours,
        wall_hours_cap=WALL_HOURS_CAP,
        lane_hours_expected=expected / RATE_UNITS_PER_HOUR,
        max_in_flight=lanes,
    )
    if expected > STUDY_UNIT_CAP:
        sizing.update(action="KILL", reason="forecast expected units > 120")
    elif wall_hours > WALL_HOURS_CAP:
        sizing.update(action="KILL", reason="forecast wall > 24 h on the registered lanes")
    else:
        sizing.update(action="RATIFY", reason="r >= 5, expected <= 120 u, wall <= 24 h")
    return sizing


def replay_sizing(receipt: dict[str, Any]) -> dict[str, Any]:
    """One second-L4 replay job sized from A5-D's measured L4 metric cost (K=12)."""
    per_pair = float(receipt["metric_seconds_per_pair"]) * float(receipt["cpu_slowdown_factor"])
    workload = float(receipt["setup_minutes"]) + math.ceil(REPLAY_SAMPLE_PAIRS / 12) * per_pair / 60
    estimate = math.ceil(workload)
    timeout = max(math.ceil(workload + 25), 2 * estimate, 60)
    cap = timeout * RATE_UNITS_PER_HOUR / 60.0
    return {
        "sample_pairs": REPLAY_SAMPLE_PAIRS,
        "estimate_minutes": estimate,
        "timeout_minutes": timeout,
        "unit_cap": cap,
        "reserve_units": 0.2 * cap,
        "expected_units": workload / 60.0 * RATE_UNITS_PER_HOUR,
    }


def validate(receipt: dict[str, Any]) -> dict[str, Any]:
    """Refuse a missing, held, or out-of-contract A5-D receipt."""
    if receipt.get("schema") != SCHEMA:
        raise SizingRefused("A5-D receipt schema differs")
    if receipt.get("fidelity_status") not in FIDELITY_OK:
        raise SizingRefused("A5-D fidelity is not PASS/EXPLAINED_TIE: HOLD")
    _positive(receipt.get("teacher_worlds_per_hour"), "teacher_worlds_per_hour")
    for key in ("bare_seconds_per_world", "metric_seconds_per_pair", "setup_minutes"):
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
    """Refuse a chunk whose expected spend would cross the derived study ceiling.

    Settled attempts are charged their whole-session elapsed time at 2.0 u/h; an
    in-flight or receipt-less attempt is charged its full unit cap. ``settled_only``
    reports whether the refusal stands without any in-flight charge (-> INCONCLUSIVE).
    """
    settled = unsettled = 0.0
    for item in prior:
        elapsed = item.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0:
            settled += float(elapsed) / 3600.0 * RATE_UNITS_PER_HOUR
        else:
            unsettled += float(item.get("unit_cap", ceiling))
    expected = float(chunk["workload_minutes"]) / 60.0 * RATE_UNITS_PER_HOUR
    return {
        "charged_units": settled + unsettled,
        "settled_units": settled,
        "chunk_expected_units": expected,
        "ceiling": ceiling,
        "allowed": settled + unsettled + expected <= ceiling,
        "infeasible_on_settled_charges": settled + expected > ceiling,
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


def build(a5d_benchmark: Path, preflight: Path, setup_minutes: float) -> dict[str, Any]:
    """Bind the supervisor-verified A5-D export (L4 costs) and the source-current A6."""
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
        or data.get("schema") != "t34-round0d-a5d-benchmark/v2"
        or data.get("processes") != 12
    ):
        raise SizingRefused("A5-D requires a verified L4 PASS benchmark with K=12 processes")
    observed_setup = max(0.0, (receipt["elapsed_seconds"] - data["elapsed_seconds"]) / 60)
    if setup_minutes < observed_setup:
        raise SizingRefused("setup allowance is below measured A5-D setup/export/cleanup")
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
        "a6_chunk_sha256": sha256(chunk_path),
        "observed_setup_export_cleanup_minutes": observed_setup,
        "setup_minutes": setup_minutes,
    }
    payload = {
        "schema": SCHEMA,
        "teacher_worlds_per_hour": rates["teacher_worlds_per_hour"],
        "fidelity_status": data["fidelity"]["status"],
        "bare_seconds_per_world": bare,
        "metric_seconds_per_pair": metric,
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


def estimated_plan(teacher_worlds_per_hour: float, setup_minutes: float = 15.0) -> dict:
    """ESTIMATED draft table row; never admits dispatch.

    Placeholder L4 costs: Round 0-D's own A6 (Mac, H500x10: bare 20.2 s/world, metric
    5.4 s/pair) times the ops panel's 6-9x L4 per-process penalty (9x), rounded up.
    """
    receipt = {
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
    receipt["evidence_sha256"] = hashlib.sha256(
        json.dumps(receipt["evidence"], sort_keys=True).encode()
    ).hexdigest()
    return {"status": "ESTIMATED_NOT_DISPATCHABLE", "plan": plan(receipt)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5d-benchmark", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--setup-minutes", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.a5d_benchmark, args.preflight_dir, args.setup_minutes)
    write_once(args.out, payload)
    print(json.dumps({k: payload["plan"].get(k) for k in ("action", "reason")}))


if __name__ == "__main__":
    main()
