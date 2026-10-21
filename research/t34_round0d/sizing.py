"""Bind the A5-D receipt and the excluded Mac A6 into Round 0-D chunk sizing.

Round 0's chunk formula (``research.t34_round0.make_manifest.size_round``) is
imported unchanged: chunk n is the largest multiple of ten with
setup + n*(3600/r + 3*(bare+metric)/12)/60 <= 180 minutes, timeout >=
max(estimate+25, 2*estimate), priced at 2.0 u/h with a held 20% reserve.

Round 0-D-specific rules (PREREGISTRATION_ROUND0D.md "Sizing and caps"):

* r is the A5-D dense-teacher completion rate on one L4 lane (K=12), measured
  over a 1,200 s steady window after a 1,200 s cold fill. KILL if r < 8.
* A5-D fidelity must be PASS (all 8 T3 census worlds reproduce); any
  difference HOLDS sizing until explained by the operator's review.
* Ratification requires a forecast expected spend <= 40 u (the plan cap). The study
  charged-unit ceiling is 48 u = 40 u + the held 20% reserve (``admission_spend``).
* Dense R1 (only after a Round 0-D GO) is sized here from the same r.
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

SCHEMA = "t34-round0d-a5d/v1"
KILL_WORLDS_PER_HOUR = 8.0
STEADY_WINDOW_SECONDS = 1200
COLD_FILL_SECONDS = 1200
EXPECTED_UNIT_CEILING = 40.0
STUDY_UNIT_CEILING = 48.0
H500_TO_H5000 = 10
CPU_SLOWDOWN_FACTOR = 3.0
EXPORT_BYTES_PER_PAIR_LIMIT = 9 * 1024**2  # Round 0's registered allowance
R1_FRESH_WORLDS = 560
R1_SEALED_WORLDS = 240
R1_LANES = 3
R1_UNIT_CAP = 30.0
R1_NAMESPACE = "t34-round0d-dense-r1-20261021/v1"
R1_COUNTS = {"scripted": 224, "frozen": 168, "mixed": 168}


class SizingRefused(ValueError):
    """Measured evidence does not admit a Round 0-D plan."""


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SizingRefused(f"{name} must be finite")
    if value <= 0:
        raise SizingRefused(f"{name} must be positive")
    return float(value)


def plan(receipt: dict[str, Any], pairs: int = 240) -> dict[str, Any]:
    """Return Round 0's frozen chunk partition plus the Round 0-D spend forecast."""
    validate(receipt)
    sizing = size_round(
        pairs,
        teacher_worlds_per_hour=receipt["teacher_worlds_per_hour"],
        bare_seconds_per_world=receipt["bare_seconds_per_world"],
        metric_seconds_per_pair=receipt["metric_seconds_per_pair"],
        cpu_slowdown_factor=receipt["cpu_slowdown_factor"],
        setup_minutes=receipt["setup_minutes"],
    )
    expected = sum(
        float(chunk["workload_minutes"]) / 60.0 * RATE_UNITS_PER_HOUR for chunk in sizing["chunks"]
    )
    largest_headroom = max(
        float(chunk["unit_cap"]) - float(chunk["workload_minutes"]) / 60.0 * RATE_UNITS_PER_HOUR
        for chunk in sizing["chunks"]
    )
    sizing.update(
        expected_units=expected,
        expected_unit_ceiling=EXPECTED_UNIT_CEILING,
        study_unit_ceiling=STUDY_UNIT_CEILING,
        worst_case_charged_units=STUDY_UNIT_CEILING + largest_headroom,
        lane_hours_expected=expected / RATE_UNITS_PER_HOUR,
        initial_liability_units=sizing["unit_cap"] + sizing["reserve_units"],
    )
    return sizing


def validate(receipt: dict[str, Any]) -> dict[str, Any]:
    """Refuse a missing, failed, or out-of-contract A5-D receipt."""
    if receipt.get("schema") != SCHEMA:
        raise SizingRefused("A5-D receipt schema differs")
    if receipt.get("fidelity_status") != "PASS":
        raise SizingRefused("A5-D fidelity is not PASS: HOLD until each difference is explained")
    rate = _positive(receipt.get("teacher_worlds_per_hour"), "teacher_worlds_per_hour")
    if rate < KILL_WORLDS_PER_HOUR:
        raise SizingRefused("A5-D KILL: fewer than 8 dense worlds/h/lane; re-plan on Mac capacity")
    for key in ("bare_seconds_per_world", "metric_seconds_per_pair", "setup_minutes"):
        _positive(receipt.get(key), key)
    if receipt.get("cpu_slowdown_factor") != CPU_SLOWDOWN_FACTOR:
        raise SizingRefused("A5-D receipt must use the registered 3x CPU planning factor")
    if receipt.get("steady_window_seconds") != STEADY_WINDOW_SECONDS:
        raise SizingRefused("A5-D receipt must use the registered 1,200 s steady window")
    exported = _positive(receipt.get("measured_export_bytes_per_pair"), "export bytes")
    if exported > EXPORT_BYTES_PER_PAIR_LIMIT:
        raise SizingRefused("measured exports exceed the 9 MiB/pair allowance; re-size first")
    digest = hashlib.sha256(json.dumps(receipt.get("evidence"), sort_keys=True).encode())
    if digest.hexdigest() != receipt.get("evidence_sha256"):
        raise SizingRefused("A5-D evidence digest differs")
    return receipt


def admission_spend(prior_receipts: list[dict[str, Any]], chunk: dict[str, Any]) -> dict:
    """Study ceiling: refuse a chunk whose expected spend would cross 48 charged units.

    Prior attempts are charged at the priced rate over their recorded whole-session
    elapsed time; an attempt without a terminal elapsed time is charged its full cap.
    """
    charged = 0.0
    for receipt in prior_receipts:
        elapsed = receipt.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed >= 0:
            charged += float(elapsed) / 3600.0 * RATE_UNITS_PER_HOUR
        else:
            charged += float(receipt.get("unit_cap", STUDY_UNIT_CEILING))
    expected = float(chunk["workload_minutes"]) / 60.0 * RATE_UNITS_PER_HOUR
    allowed = charged + expected <= STUDY_UNIT_CEILING
    return {"charged_units": charged, "chunk_expected_units": expected, "allowed": allowed}


def dense_r1_plan(teacher_worlds_per_hour: float) -> dict[str, Any]:
    """Conditional dense R1 sizing; executes only after a published Round 0-D GO."""
    rate = _positive(teacher_worlds_per_hour, "teacher_worlds_per_hour")
    lane_hours = R1_FRESH_WORLDS / rate
    wall_hours = lane_hours / R1_LANES
    expected = lane_hours * RATE_UNITS_PER_HOUR
    return {
        "namespace": R1_NAMESPACE,
        "fresh_worlds": R1_FRESH_WORLDS,
        "fresh_counts": R1_COUNTS,
        "sealed_round0d_teacher_worlds": R1_SEALED_WORLDS,
        "total_label_worlds": R1_FRESH_WORLDS + R1_SEALED_WORLDS,
        "lanes": R1_LANES,
        "lane_hours": lane_hours,
        "wall_hours": wall_hours,
        "expected_units": expected,
        "unit_cap": R1_UNIT_CAP,
        "fundable": expected <= R1_UNIT_CAP,
        "condition": "published Round 0-D analysis/decision.json == GO",
    }


def build(a5d_benchmark: Path, preflight: Path, setup_minutes: float) -> dict[str, Any]:
    """Bind the supervisor-verified A5-D export and the excluded A6 Mac costs."""
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
        or data.get("schema") != "t34-round0d-a5d-benchmark/v1"
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
    pairs = [read_json(preflight / item["path"]) for item in chunk["pairs"]]
    costs = [
        {
            arm: {
                k: p["arms"][arm][k] for k in ("rollout_elapsed_seconds", "metric_elapsed_seconds")
            }
            for arm in ("bare", "teacher")
        }
        for p in pairs
    ]
    bare = max(c["bare"]["rollout_elapsed_seconds"] for c in costs) * H500_TO_H5000
    metric = (
        max(sum(c[a]["metric_elapsed_seconds"] for a in ("bare", "teacher")) for c in costs)
        * H500_TO_H5000
    )
    evidence = {
        "a5d_benchmark": {"path": str(a5d_benchmark.resolve()), "sha256": sha256(a5d_benchmark)},
        "a5d_receipt_sha256": sha256(job / "receipt.json"),
        "a5d_window": data["window"],
        "a5d_fidelity": data["fidelity"],
        "a6_chunk_sha256": sha256(chunk_path),
        "a6_pair_costs": costs,
        "h500_to_h5000_factor": H500_TO_H5000,
        "cpu_slowdown_factor": CPU_SLOWDOWN_FACTOR,
        "observed_setup_export_cleanup_minutes": observed_setup,
        "setup_minutes": setup_minutes,
    }
    payload = {
        "schema": SCHEMA,
        "teacher_worlds_per_hour": float(data["window"]["completed_worlds_per_hour"]),
        "fidelity_status": data["fidelity"]["status"],
        "bare_seconds_per_world": bare,
        "metric_seconds_per_pair": metric,
        "cpu_slowdown_factor": CPU_SLOWDOWN_FACTOR,
        "setup_minutes": setup_minutes,
        "steady_window_seconds": STEADY_WINDOW_SECONDS,
        "cold_fill_seconds": COLD_FILL_SECONDS,
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
    payload["dense_r1_plan"] = dense_r1_plan(payload["teacher_worlds_per_hour"])
    return payload


def estimated_plan(teacher_worlds_per_hour: float, setup_minutes: float = 15.0) -> dict:
    """ESTIMATED draft sizing from the plan's E8 bracket; never admits dispatch."""
    receipt = {
        "schema": SCHEMA,
        "fidelity_status": "PASS",
        "teacher_worlds_per_hour": teacher_worlds_per_hour,
        # Placeholders only: Round 0's own measured A6 costs (bare 91.9 s/world incl.
        # warm-up, metric 3.9 s/pair, H500x10), rounded up. Ratification uses the
        # source-current Round 0-D A6 instead (``build``).
        "bare_seconds_per_world": 92.0,
        "metric_seconds_per_pair": 8.0,
        "cpu_slowdown_factor": CPU_SLOWDOWN_FACTOR,
        "setup_minutes": setup_minutes,
        "steady_window_seconds": STEADY_WINDOW_SECONDS,
        "measured_export_bytes_per_pair": 4 * 1024**2,
        "evidence": {"estimated": True},
    }
    receipt["evidence_sha256"] = hashlib.sha256(
        json.dumps(receipt["evidence"], sort_keys=True).encode()
    ).hexdigest()
    return {
        "status": "ESTIMATED_NOT_DISPATCHABLE",
        "assumed_teacher_worlds_per_hour": teacher_worlds_per_hour,
        "plan": plan(receipt),
        "dense_r1_plan": dense_r1_plan(teacher_worlds_per_hour),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5d-benchmark", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--setup-minutes", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.a5d_benchmark, args.preflight_dir, args.setup_minutes)
    if payload["plan"]["expected_units"] > EXPECTED_UNIT_CEILING:
        raise SizingRefused("forecast expected spend exceeds 40 u: re-plan, do not ratify")
    write_once(args.out, payload)


if __name__ == "__main__":
    main()
