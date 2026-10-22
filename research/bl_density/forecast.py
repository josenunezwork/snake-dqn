"""Host-sensitive engineering forecast from tiny immutable preflight exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.bl_density.runner import DENSITY2_ALLOCATION, DEPTH, HORIZON
from research.bl_ppo.exports import export_json, verify_export

PRODUCTION_EPISODES = sum(sum(by_mix.values()) for by_mix in DENSITY2_ALLOCATION.values())
PRODUCTION_WORKERS = 10
PRODUCTION_WORK_WALL_SECONDS = 10_800
PRODUCTION_SESSION_MINUTES = 180
PRODUCTION_UNIT_CAP = 5.0
PRODUCTION_RESERVE_UNITS = 1.0
PRODUCTION_RATE_UNITS_PER_HOUR = 1.54

# Immutable, attributed per-episode CPU timings from the completed sizing pilot.
# They size the bounded replication only; no pilot outcome enters this forecast.
PILOT_CPU_SECONDS_PER_EPISODE = {
    "A": {"frozen": 67.7, "mixed": 48.6, "scripted": 62.5},
    "B70m": {"frozen": 42.6, "mixed": 44.2, "scripted": 33.8},
    "B70p": {"frozen": 43.1, "mixed": 56.9, "scripted": 20.8},
    "C-P1": {"frozen": 50.0, "mixed": 57.6, "scripted": 48.9},
}


def pilot_cpu_projection() -> dict[str, float]:
    """Return the pre-data pilot-timing projection for the frozen allocation."""
    by_arm = {
        arm: sum(
            DENSITY2_ALLOCATION[arm][mix] * PILOT_CPU_SECONDS_PER_EPISODE[arm][mix]
            for mix in DENSITY2_ALLOCATION[arm]
        )
        for arm in DENSITY2_ALLOCATION
    }
    return {**by_arm, "total": sum(by_arm.values())}


def forecast(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Price fixed work ceilings; do not turn tiny timings into a throughput claim."""
    branch_rows = [row for record in records for row in record["sampled"] if row.get("prefilter")]
    timed = [
        (
            float(row["h300_cpu_seconds"]),
            int(row.get("continuation_branch_decisions", row["h300_frames_executed"])),
        )
        for row in branch_rows
        if row.get("continuation_branch_decisions", row.get("h300_frames_executed", 0)) > 0
    ]
    if len(timed) < 3:
        raise ValueError("forecast needs at least three real timed continuation units")
    # Max observed per-frame cost is deliberately conservative about mix variation.
    branch_seconds_per_frame = max(seconds / frames for seconds, frames in timed)
    live_rates = [
        float(record["steady_live_step_cpu_seconds"]) / int(record["steady_live_decisions"])
        for record in records
        if record.get("steady_live_decisions", 0) > 0
    ]
    seconds_per_frame = max([branch_seconds_per_frame, *live_rates])
    setup_seconds = max(
        float(record.get("world_setup_cpu_seconds", 0))
        + float(record.get("first_step_cpu_seconds", 0))
        for record in records
    )
    episodes, workers, roots, horizon, depth = (
        PRODUCTION_EPISODES,
        PRODUCTION_WORKERS,
        24,
        HORIZON,
        DEPTH,
    )
    decision_bound = episodes * (horizon + roots * depth + 4 * 5 * depth)
    cpu_bound = seconds_per_frame * decision_bound + episodes * setup_seconds
    pilot_projection = pilot_cpu_projection()
    cases = []
    for multiplier in (2, 3, 5):
        minutes = 10 + 5 + cpu_bound * multiplier / workers / 60
        cases.append(
            {
                "cloud_cpu_slowdown_assumption": multiplier,
                "session_minutes": minutes,
                "units_at_2_per_hour": minutes * 2 / 60,
                "within_180_minute_cap": minutes <= PRODUCTION_SESSION_MINUTES,
                "within_10800_second_work_cap": minutes * 60 <= PRODUCTION_WORK_WALL_SECONDS,
            }
        )
    return {
        "schema": "bl-density-engineering-forecast/v2",
        "development_only": True,
        "status": "TINY_MAC_SENSITIVITY_NOT_L4_MEASUREMENT",
        "timed_continuations": len(timed),
        "measured_continuation_cpu_seconds": [seconds for seconds, _ in timed],
        "measured_continuation_decisions": [frames for _, frames in timed],
        "max_observed_cpu_seconds_per_world_decision": seconds_per_frame,
        "max_observed_setup_cpu_seconds": setup_seconds,
        "production_episodes": episodes,
        "workers": workers,
        "max_sampled_roots_per_episode": roots,
        "max_avoidability_roots_per_episode": 4,
        "max_alternative_actions_per_root": 5,
        "world_decision_upper_bound": decision_bound,
        "mac_cpu_seconds_projected_at_max_observed_rate": cpu_bound,
        "pilot_timing_projection_cpu_seconds": pilot_projection,
        "pilot_timing_projection_cpu_hours": pilot_projection["total"] / 3600,
        "pilot_six_worker_effective_wall_minutes": pilot_projection["total"] / 4.1 / 60,
        "ten_worker_wall_minutes_expected_range": [70, 80],
        "bootstrap_public_download_export_allowance_minutes": 10,
        "analysis_allowance_minutes": 5,
        "sensitivity": cases,
        "manifest_work_wall_seconds": PRODUCTION_WORK_WALL_SECONDS,
        "whole_session_wall_cap_minutes": PRODUCTION_SESSION_MINUTES,
        "unit_cap": PRODUCTION_UNIT_CAP,
        "reserve_units": PRODUCTION_RESERVE_UNITS,
        "rate_units_per_hour_planning_ceiling": PRODUCTION_RATE_UNITS_PER_HOUR,
        "worst_case_units_at_session_cap": (
            PRODUCTION_SESSION_MINUTES / 60 * PRODUCTION_RATE_UNITS_PER_HOUR
        ),
        "limits": [
            "tiny warm/compile/cache state differs from fresh Colab",
            "no measured L4 throughput, hardware availability or rate verification",
            "timed injected lengths and mixes only; future natural long states may cost more",
            "screen pass rate and early deaths can reduce work below the decision ceiling",
            "ten-worker scaling is unmeasured; the 180-minute cap absorbs a 1.7x miss",
            "supervisor first wall/unit cap wins; incomplete collection is not completion",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted((args.preflight / "episodes").glob("*.json")):
        verify_export(path)
        records.append(json.loads(path.read_text()))
    export_json(args.out, forecast(records))


if __name__ == "__main__":
    main()
