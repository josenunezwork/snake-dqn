"""Host-sensitive engineering forecast from tiny immutable preflight exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_json, verify_export


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
    episodes, workers, roots, horizon, depth = 90, 6, 24, 5000, 300
    decision_bound = episodes * (horizon + roots * depth + 4 * 5 * depth)
    cpu_bound = seconds_per_frame * decision_bound + episodes * setup_seconds
    cases = []
    for multiplier in (2, 3, 5):
        minutes = 10 + 5 + cpu_bound * multiplier / workers / 60
        cases.append(
            {
                "cloud_cpu_slowdown_assumption": multiplier,
                "session_minutes": minutes,
                "units_at_2_per_hour": minutes * 2 / 60,
                "within_60_minute_plan": minutes <= 60,
                "within_120_minute_priced_wall": minutes <= 120,
            }
        )
    return {
        "schema": "bl-density-engineering-forecast/v1",
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
        "bootstrap_public_download_export_allowance_minutes": 10,
        "analysis_allowance_minutes": 5,
        "sensitivity": cases,
        "manifest_session_plan_minutes": 60,
        "whole_session_wall_cap_minutes": 120,
        "registered_outer_wall_cap_minutes": 150,
        "unit_cap": 5,
        "reserve_units": 1.0,
        "rate_units_per_hour_planning_ceiling": 2,
        "limits": [
            "tiny warm/compile/cache state differs from fresh Colab",
            "no measured L4 throughput, hardware availability or rate verification",
            "timed injected lengths and mixes only; future natural long states may cost more",
            "screen pass rate and early deaths can reduce work below the decision ceiling",
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
