"""Recompute conditional resource forecasts from disjoint teacher speed receipts.

No platform is contacted. Historical rental/unit rates are scenario inputs,
not current quotations. The L4 calculation deliberately exposes its unmeasured
GPU service-time assumption and CPU/GPU overlap bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

MIX_COUNTS = {"scripted": 96, "frozen": 32, "mixed": 32}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def long_times(receipt: dict[str, Any]) -> dict[str, float]:
    """Median searched-decision time over the L800/L1500 preflight fixtures."""
    return {
        mix: statistics.median(
            row["seconds"]
            for row in receipt["rows"]
            if row["mix"] == mix and row["length"] in (800, 1500)
        )
        for mix in MIX_COUNTS
    }


def forecast(
    timing: dict[str, Any],
    mps_timing: dict[str, Any],
    scaling_one: dict[str, Any],
    scaling_six: dict[str, Any],
    calibration: dict[str, Any],
    mps_profile: dict[str, Any],
) -> dict[str, Any]:
    """Apply the original preflight extrapolation, with explicit remote scenarios."""
    cpu = long_times(timing)
    mps = long_times(mps_timing)
    searched = {
        mix: MIX_COUNTS[mix]
        * next(
            w["mandatory_length_decisions"]
            for w in calibration["worlds"]
            if f"-{mix}-" in w["world_id"]
        )
        for mix in MIX_COUNTS
    }
    proposal_counts = {
        mix: MIX_COUNTS[mix]
        * next(
            w["proposal_length800_decisions"]
            for w in calibration["worlds"]
            if f"-{mix}-" in w["world_id"]
        )
        for mix in MIX_COUNTS
    }
    # Same allowance as PREFLIGHT.json: twice bare calibration cost per mix.
    bare_seconds = 2 * sum(
        MIX_COUNTS[mix]
        * next(w["seconds"] for w in calibration["worlds"] if f"-{mix}-" in w["world_id"])
        for mix in MIX_COUNTS
    )
    scale = scaling_one["batch_wall_seconds"] / scaling_six["batch_wall_seconds"]
    phase_core_h = (sum(searched[mix] * cpu[mix] for mix in MIX_COUNTS) + bare_seconds) / 3600
    distill_core_h = statistics.mean(cpu.values()) * 1_000_000 / 3600
    gpu_service = {
        mix: statistics.median(
            row["categories_seconds"]["b8_forward"]
            for row in mps_profile["measurements"]
            if row["mix"] == mix
        )
        for mix in MIX_COUNTS
    }
    host = {mix: max(0.0, mps[mix] - gpu_service[mix]) for mix in MIX_COUNTS}
    l4 = {}
    for name, counts, bare in (
        ("phase0", searched, bare_seconds),
        ("distillation_1m_equal_mix", {mix: 1_000_000 / 3 for mix in MIX_COUNTS}, 0),
    ):
        # 12 logical vCPUs / 6 physical cores, historical 3.7x host slowdown.
        host_h = (sum(counts[mix] * host[mix] for mix in MIX_COUNTS) + bare) * 3.7 / 6 / 3600
        l4[name] = {}
        for factor in (0.5, 1.0, 2.0):
            gpu_h = sum(counts[mix] * gpu_service[mix] for mix in MIX_COUNTS) * factor / 3600
            bounds = [max(host_h, gpu_h), host_h + gpu_h]
            l4[name][str(factor)] = {
                "host_hours": host_h,
                "gpu_service_hours": gpu_h,
                "wall_hours_overlap_to_serial": bounds,
                "units_at_historical_2_per_hour": [2 * value for value in bounds],
            }
    return {
        "schema": "phase0-teacher-speed-forecast/v1",
        "search_config": timing["config"],
        "searched_decisions_by_mix": searched,
        "searched_decisions_total": sum(searched.values()),
        "long_fixture_seconds_cpu": cpu,
        "long_fixture_seconds_mps": mps,
        "six_task_measured_scaling_speedup": scale,
        "phase0": {
            "mac_single_core_hours": phase_core_h,
            "mac_six_worker_wall_hours": phase_core_h / scale,
            "mac_wall_hours_with_20pct_reserve": phase_core_h / scale * 1.2,
            "mac_cloud_rental_usd": 0,
            "runpod_32_vcpu_hours_2_4x_core_slowdown": phase_core_h * 2.4 / 32,
            "runpod_usd_at_historical_0_89_to_0_95_per_hour": [
                phase_core_h * 2.4 / 32 * rate for rate in (0.89, 0.95)
            ],
            "runpod_speedup_vs_mac_scenario": 32 / (2.4 * scale),
        },
        "distillation_1m_equal_mix": {
            "mac_single_core_hours": distill_core_h,
            "mac_six_worker_wall_hours": distill_core_h / scale,
            "runpod_32_vcpu_hours_2_4x_core_slowdown": distill_core_h * 2.4 / 32,
            "runpod_usd_at_historical_0_89_to_0_95_per_hour": [
                distill_core_h * 2.4 / 32 * rate for rate in (0.89, 0.95)
            ],
        },
        "l4_12_vcpu_scenarios": l4,
        "serving_planner_descriptive_only": {
            device: {
                "forced_search_ms_equal_mix": statistics.mean(times.values()) * 1000,
                "times_41ms_tick": statistics.mean(times.values()) / 0.041,
                "amortized_search_ms_at_old_calibration_fraction": statistics.mean(times.values())
                * calibration["searched_fraction"]
                * 1000,
            }
            for device, times in (("cpu", cpu), ("mps", mps))
        },
        "proposals_not_adopted": {
            "N6_H96": {
                "branch_frame_ratio": 6 * 96 / (8 * 128),
                "linear_work_mac_wall_hours_phase0": (
                    (phase_core_h * 3600 - bare_seconds) * 0.5625 + bare_seconds
                )
                / 3600
                / scale,
                "linear_work_mac_wall_hours_distillation": distill_core_h * 0.5625 / scale,
            },
            "length_gate800": {
                "calibration_fraction": calibration["proposal_length800_fraction"],
                "searched_decisions_total": sum(proposal_counts.values()),
                "mac_wall_hours_phase0_scenario": (
                    sum(proposal_counts[mix] * cpu[mix] for mix in MIX_COUNTS) + bare_seconds
                )
                / 3600
                / scale,
            },
        },
        "limitations": [
            "Synthetic timing bodies; search changes visitation/survival. "
            "One bare calibration world per mix.",
            "Mac scaling includes startup for six tasks, not steady-state whole-bank scaling.",
            "Remote hardware/rates/setup/transfers not measured; no remote action performed.",
            "L4 service time is 0.5/1/2 times measured Mac MPS forward time: "
            "sensitivity only, not an L4 benchmark.",
            "Remote ranges exclude setup/export; add 20 percent reserve "
            "before any authorized scheduling.",
            "Distillation assumes one million already-selected search decisions "
            "in equal mix proportions; excludes collection/training/I/O.",
            "N/H and length-gate alternatives change the policy and need "
            "prospective owner approval and new evidence.",
            "CPU/MPS fixture equality does not qualify universal trajectory identity or serving.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "timing",
        "mps-timing",
        "scaling-one",
        "scaling-six",
        "calibration",
        "mps-profile",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    names = ("timing", "mps_timing", "scaling_one", "scaling_six", "calibration", "mps_profile")
    paths = {name: getattr(args, name) for name in names}
    result = forecast(**{name: _load(path) for name, path in paths.items()})
    result["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()
    }
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
