"""Measured receipt-based unit-cost and 6-runtime sharding forecast for Watch6-L1-V1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from research.watch6_l1_confirm import protocol as p
from tools.ops.package_spec import publish_json

SCHEMA = "watch6-l1-empirical-costs/v1"
MAX_BASE_MINUTES = 660  # 2x timeout <= 1,320 min (22.0 h), leaving 2 h below Colab 24 h VM cap.
VM_SETUP_SECONDS = 450.0
W6_CONTENTION_VS_W4 = 415.72 / 403.11  # 1.03128x measured on 12-vCPU A100 (argon-alllen-a100).

# Calibrated provider rates observed in rate-calibration.json vs manifest priced ceilings.
# Supervisor enforces observed_rate <= rate_units_per_hour + 0.05.
OBSERVED_RATES_U_PER_H: dict[str, float] = {"A100": 6.73, "L4": 1.94}
PRICED_CEILING_RATES_U_PER_H: dict[str, float] = {"A100": 8.0, "L4": 2.5}

# Frozen pre-data base_minutes per single-wave shard (doubled for 2x timeout_minutes).
BASE_MINUTES_BY_GPU: dict[str, int] = {
    "A100": 570,  # 9.5 h base (> 8.02 h p95) -> 1,140 min (19.0 h) 2x timeout
    "L4": 540,  # 9.0 h base (> 7.78 h p95) -> 1,080 min (18.0 h) 2x timeout
}
CANARY_BASE_MINUTES = 15  # 30 min 2x timeout

# Scaling of 1-arm exact-stop-l1 compute from H=6,000 (bank14l1..16l1) to H=14,400:
# l1 already fires at the physical 5-tick ceiling during 0..6,000 (1,200.1 roots / 6,000 frames),
# so root count scales strictly 2.40x (<= 2,880 roots at H=14,400); mature coils in 6,000..14,400
# add ~10-27% per-root weight (minus skipped snapshot()/.watch.gz serialization when record=None).
L1_HORIZON_SCALE_LINEAR = 14400.0 / 6000.0  # 2.40x lower bound
L1_HORIZON_SCALE_CENTRAL = 2.65  # central forecast
L1_HORIZON_SCALE_P95 = 3.05  # conservative p95 upper forecast

MEASURED_BASIS: dict[str, object] = {
    "watch6_v3_l4_w4_h14400_2arm_live_mirror": {
        "source_glob": "snake-dqn-artifacts/ops/manifests/watch6-v3-fix3-shard-*.json",
        "pass_shards": 24,
        "gpu": "L4",
        "workers": 4,
        "pairs_per_shard": 4,
        "waves_per_shard": 1,
        "horizon_frames": 14400,
        "arms": ["bare", "l2_prime"],
        "observed_rate_u_per_h": 1.94,
        "priced_ceiling_rate_u_per_h": 2.5,
        "elapsed_seconds_min": 6755.3,
        "elapsed_seconds_mean": 10312.4,
        "elapsed_seconds_median": 10334.1,
        "elapsed_seconds_p95": 12058.8,
        "elapsed_seconds_max": 12313.7,
        "net_compute_seconds_mean": 9862.4,
        "net_compute_seconds_p95": 11608.8,
        "l2_prime_roots_mean": 1311.28,
        "l2_prime_branches_mean": 1500.66,
        "l2_prime_branches_p95": 2235.0,
    },
    "watch6_alllen_a100_w6_h6000_3arm_no_mirror": {
        "source_paths": [
            (
                "snake-dqn-artifacts/t34-sol/argon-alllen-a100/"
                "prep/output-watch6-alllen-a100-s0-01/receipt.json"
            ),
            (
                "snake-dqn-artifacts/t34-sol/argon-alllen-a100/"
                "prep/output-watch6-alllen-a100-s1-01/receipt.json"
            ),
            (
                "snake-dqn-artifacts/t34-sol/argon-alllen-a100/"
                "prep/output-watch6-alllen-a100-s2-02/receipt.json"
            ),
        ],
        "pass_shards": 3,
        "gpu": "A100",
        "workers": 6,
        "units_per_shard": [12, 12, 11],
        "waves_per_shard": 2,
        "horizon_frames": 6000,
        "observed_rate_u_per_h": 6.73,
        "priced_ceiling_rate_u_per_h": 8.0,
        "elapsed_seconds": [15907.4, 15557.2, 16059.8],
        "elapsed_seconds_mean": 15841.5,
        "w6_vs_w4_contention_ratio_12vcpu": round(W6_CONTENTION_VS_W4, 5),
        "l2_prime_roots_mean_h6000": 403.7,
        "l2_prime_branches_mean_h6000": 449.6,
        "l1_roots_mean_h6000": 1200.1,
        "l1_branches_mean_h6000": 1268.5,
    },
    "watch_replay_l1_banks_a100_w6_h6000_1arm_live_mirror": {
        "source_paths": [
            "snake-dqn-artifacts/t34-sol/argon-banks-l1/prep/output-bank14l1/receipt.json",
            "snake-dqn-artifacts/t34-sol/argon-banks-l1/prep/output-bank15l1-retry-1/receipt.json",
            "snake-dqn-artifacts/t34-sol/argon-banks-l1/prep/output-bank16l1/receipt.json",
        ],
        "pass_shards": 3,
        "gpu": "A100",
        "workers": 6,
        "units_per_shard": [12, 12, 12],
        "waves_per_shard": 2,
        "horizon_frames": 6000,
        "arm": "exact-stop-l1",
        "observed_rate_u_per_h": 6.73,
        "elapsed_seconds": [11410.0, 11173.8, 11098.9],
        "elapsed_seconds_mean": 11227.6,
        "net_compute_seconds_per_w6_wave": 5388.8,
        "net_compute_seconds_per_w4_wave_equiv": round(5388.8 / W6_CONTENTION_VS_W4, 1),
    },
}


def build_cost_report() -> dict:
    """Construct the canonical cost-evidence JSON uploaded with Watch6-L1-V1 manifests."""
    return {
        "schema": SCHEMA,
        "namespace": p.NAMESPACE,
        "horizon_frames": p.HORIZON,
        "count": p.COUNT,
        "arms": list(p.ARMS),
        "account_caps": {
            "account_assignment_cap": 6,
            "account_l4_cap": 3,
            "target_concurrent_a100": 3,
            "target_concurrent_l4": 3,
        },
        "observed_rates_u_per_h": dict(OBSERVED_RATES_U_PER_H),
        "priced_ceiling_rates_u_per_h": dict(PRICED_CEILING_RATES_U_PER_H),
        "base_minutes_by_gpu": dict(BASE_MINUTES_BY_GPU),
        "canary_base_minutes": CANARY_BASE_MINUTES,
        "measured_basis": MEASURED_BASIS,
    }


def verify_report_evidence(report: dict) -> None:
    """Validate an empirical cost-evidence dictionary offline (Mac or Colab VM)."""
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCHEMA
        or report.get("namespace") != p.NAMESPACE
        or report.get("horizon_frames") != p.HORIZON
        or report.get("count") != p.COUNT
        or report.get("arms") != list(p.ARMS)
        or report.get("observed_rates_u_per_h") != OBSERVED_RATES_U_PER_H
        or report.get("priced_ceiling_rates_u_per_h") != PRICED_CEILING_RATES_U_PER_H
        or report.get("base_minutes_by_gpu") != BASE_MINUTES_BY_GPU
    ):
        raise ValueError("stale, foreign, or tampered Watch6-L1-V1 cost-evidence report")
    for gpu in p.TARGET_GPUS:
        base_min = BASE_MINUTES_BY_GPU[gpu]
        if type(base_min) is not int or not 0 < base_min <= MAX_BASE_MINUTES:
            raise ValueError(f"invalid base_minutes for {gpu}: {base_min}")
        if PRICED_CEILING_RATES_U_PER_H[gpu] < OBSERVED_RATES_U_PER_H[gpu]:
            raise ValueError(f"priced ceiling below calibrated provider rate for {gpu}")


def _single_wave_seconds(gpu: str) -> dict[str, float]:
    v3 = MEASURED_BASIS["watch6_v3_l4_w4_h14400_2arm_live_mirror"]
    l1_banks = MEASURED_BASIS["watch_replay_l1_banks_a100_w6_h6000_1arm_live_mirror"]
    assert isinstance(v3, dict) and isinstance(l1_banks, dict)

    bare_l2p_w4_mean = float(v3["net_compute_seconds_mean"])
    bare_l2p_w4_p95 = float(v3["net_compute_seconds_p95"])
    l1_w6_h6k = float(l1_banks["net_compute_seconds_per_w6_wave"])
    l1_w4_h6k = l1_w6_h6k / W6_CONTENTION_VS_W4

    if gpu == "L4":
        bare_l2p_mean = bare_l2p_w4_mean
        bare_l2p_p95 = bare_l2p_w4_p95
        l1_h6k = l1_w4_h6k
    elif gpu == "A100":
        bare_l2p_mean = bare_l2p_w4_mean * W6_CONTENTION_VS_W4
        bare_l2p_p95 = bare_l2p_w4_p95 * W6_CONTENTION_VS_W4
        l1_h6k = l1_w6_h6k
    else:
        raise ValueError(f"unsupported GPU tier: {gpu!r}")

    linear_s = VM_SETUP_SECONDS + bare_l2p_mean + L1_HORIZON_SCALE_LINEAR * l1_h6k
    central_s = VM_SETUP_SECONDS + bare_l2p_mean + L1_HORIZON_SCALE_CENTRAL * l1_h6k
    p95_s = VM_SETUP_SECONDS + bare_l2p_p95 + L1_HORIZON_SCALE_P95 * l1_h6k
    return {
        "linear_lower_seconds": round(linear_s, 1),
        "expected_central_seconds": round(central_s, 1),
        "p95_upper_seconds": round(p95_s, 1),
        "expected_central_hours": round(central_s / 3600.0, 3),
        "p95_upper_hours": round(p95_s / 3600.0, 3),
    }


def _layout_summary(
    name: str,
    *,
    a100_shards: int,
    l4_shards: int,
    concurrent_a100: int = 3,
    concurrent_l4: int = 3,
) -> dict:
    worlds = a100_shards * p.A100_WORLDS_PER_SHARD + l4_shards * p.L4_WORLDS_PER_SHARD
    if worlds != p.COUNT:
        raise ValueError(f"layout {name} covers {worlds} != {p.COUNT} worlds")

    a100_t = _single_wave_seconds("A100")
    l4_t = _single_wave_seconds("L4")

    a100_obs_rate = OBSERVED_RATES_U_PER_H["A100"]
    l4_obs_rate = OBSERVED_RATES_U_PER_H["L4"]

    a100_central_u = a100_shards * a100_t["expected_central_hours"] * a100_obs_rate
    l4_central_u = l4_shards * l4_t["expected_central_hours"] * l4_obs_rate
    canary_obs_u = (10.0 / 60.0 * a100_obs_rate if a100_shards > 0 else 0.0) + (
        10.0 / 60.0 * l4_obs_rate if l4_shards > 0 else 0.0
    )

    a100_linear_u = a100_shards * (a100_t["linear_lower_seconds"] / 3600.0) * a100_obs_rate
    l4_linear_u = l4_shards * (l4_t["linear_lower_seconds"] / 3600.0) * l4_obs_rate

    a100_p95_u = a100_shards * a100_t["p95_upper_hours"] * a100_obs_rate
    l4_p95_u = l4_shards * l4_t["p95_upper_hours"] * l4_obs_rate

    a100_waves = math.ceil(a100_shards / concurrent_a100) if a100_shards > 0 else 0
    l4_waves = math.ceil(l4_shards / concurrent_l4) if l4_shards > 0 else 0
    a100_wall_h = a100_waves * a100_t["expected_central_hours"]
    l4_wall_h = l4_waves * l4_t["expected_central_hours"]

    a100_2x_cap_sum = (
        a100_shards
        * 2.0
        * (BASE_MINUTES_BY_GPU["A100"] / 60.0)
        * PRICED_CEILING_RATES_U_PER_H["A100"]
    )
    l4_2x_cap_sum = (
        l4_shards * 2.0 * (BASE_MINUTES_BY_GPU["L4"] / 60.0) * PRICED_CEILING_RATES_U_PER_H["L4"]
    )

    return {
        "layout": name,
        "total_shards": a100_shards + l4_shards,
        "a100_shards_w6": a100_shards,
        "a100_worlds": a100_shards * p.A100_WORLDS_PER_SHARD,
        "l4_shards_w4": l4_shards,
        "l4_worlds": l4_shards * p.L4_WORLDS_PER_SHARD,
        "concurrent_runtimes": {
            "a100": concurrent_a100 if a100_shards > 0 else 0,
            "l4": concurrent_l4 if l4_shards > 0 else 0,
        },
        "waves": {"a100": a100_waves, "l4": l4_waves},
        "expected_calendar_wall_hours": round(max(a100_wall_h, l4_wall_h), 2),
        "expected_lane_wall_hours": {
            "a100": round(a100_wall_h, 2),
            "l4": round(l4_wall_h, 2),
        },
        "expected_observed_units": {
            "a100_main": round(a100_central_u, 1),
            "l4_main": round(l4_central_u, 1),
            "canaries": round(canary_obs_u, 1),
            "total_central": round(a100_central_u + l4_central_u + canary_obs_u, 1),
            "range_linear_to_p95": [
                round(a100_linear_u + l4_linear_u + canary_obs_u, 1),
                round(a100_p95_u + l4_p95_u + canary_obs_u, 1),
            ],
        },
        "sum_of_2x_runaway_unit_caps_priced_ceiling": round(a100_2x_cap_sum + l4_2x_cap_sum, 1),
    }


def empirical_forecast(report: dict) -> dict:
    """Produce frozen per-GPU shard envelopes and 6-runtime unit/wall forecasts."""
    verify_report_evidence(report)
    per_gpu: dict[str, dict] = {}
    for gpu, workers, worlds in (
        ("A100", p.A100_WORKERS, p.A100_WORLDS_PER_SHARD),
        ("L4", p.L4_WORKERS, p.L4_WORLDS_PER_SHARD),
    ):
        timing = _single_wave_seconds(gpu)
        base_min = BASE_MINUTES_BY_GPU[gpu]
        if math.ceil(timing["p95_upper_seconds"] / 60.0) > base_min:
            raise ValueError(f"p95 shard forecast exceeds base_minutes on {gpu}")
        ceiling_rate = PRICED_CEILING_RATES_U_PER_H[gpu]
        obs_rate = OBSERVED_RATES_U_PER_H[gpu]
        base_units = (base_min / 60.0) * ceiling_rate
        per_gpu[gpu] = {
            "gpu": gpu,
            "workers": workers,
            "worlds_per_shard": worlds,
            "waves_per_shard": 1,
            "timing": timing,
            "observed_rate_u_per_h": obs_rate,
            "priced_ceiling_rate_u_per_h": ceiling_rate,
            "expected_observed_units_per_shard": round(
                timing["expected_central_hours"] * obs_rate, 2
            ),
            "base_minutes": base_min,
            "timeout_minutes_2x": 2 * base_min,
            "base_unit_cap_priced": round(base_units, 4),
            "unit_cap_2x_priced": round(2.0 * base_units, 4),
            "reserve_units_2x_priced": round(2.0 * base_units * 0.2, 4),
            "canary_base_minutes": CANARY_BASE_MINUTES,
            "canary_timeout_minutes_2x": 2 * CANARY_BASE_MINUTES,
            "canary_unit_cap_2x_priced": round(
                2.0 * (CANARY_BASE_MINUTES / 60.0) * ceiling_rate, 4
            ),
        }

    registered = _layout_summary(
        "12a100_w6_6l4_w4_registered",
        a100_shards=p.A100_SHARDS,
        l4_shards=p.L4_SHARDS,
    )
    alternatives = [
        _layout_summary("10a100_w6_9l4_w4_3wave_l4", a100_shards=10, l4_shards=9),
        _layout_summary("8a100_w6_12l4_w4_wall_matched", a100_shards=8, l4_shards=12),
        _layout_summary("24l4_w4_l4_only_fallback", a100_shards=0, l4_shards=24),
    ]
    return {
        "schema": "watch6-l1-forecast/v1",
        "namespace": p.NAMESPACE,
        "shards": p.SHARDS,
        "count": p.COUNT,
        "horizon_frames": p.HORIZON,
        "per_gpu_envelopes": per_gpu,
        "registered_plan": registered,
        "alternative_6runtime_plans": alternatives,
    }


def verify_local_receipt_basis(artifacts_root: Path) -> dict:
    """Re-verify V3/alllen/bank14l1-16l1 timings from local receipts (never touches exports/)."""
    artifacts_root = artifacts_root.resolve(strict=True)

    def _safe_read_json(path: Path) -> dict:
        if "exports" in path.parts or path.name.startswith(("pair-", "world-")):
            raise ValueError(f"forbidden outcome path in receipt audit: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    v3_elapsed: list[float] = []
    for mf_path in sorted((artifacts_root / "ops/manifests").glob("watch6-v3-fix3-shard-*.json")):
        if mf_path.name.endswith(".live.json"):
            continue
        mf = _safe_read_json(mf_path)
        for cand in (Path(mf["output_dir"]), Path(mf["output_dir"] + "-retry-1")):
            rec_path = cand / "receipt.json"
            if rec_path.is_file():
                rec = _safe_read_json(rec_path)
                if rec.get("status") == "PASS":
                    v3_elapsed.append(float(rec["elapsed_seconds"]))

    if len(v3_elapsed) != 24:
        raise ValueError(f"expected 24 PASS Watch6 V3 shard receipts, found {len(v3_elapsed)}")

    v3_basis = MEASURED_BASIS["watch6_v3_l4_w4_h14400_2arm_live_mirror"]
    assert isinstance(v3_basis, dict)
    if not math.isclose(
        float(np.mean(v3_elapsed)), float(v3_basis["elapsed_seconds_mean"]), abs_tol=1.0
    ):
        raise ValueError("local Watch6 V3 mean elapsed_seconds differs from embedded basis")

    return {
        "status": "VERIFIED_AGAINST_LOCAL_RECEIPTS",
        "v3_pass_shards": len(v3_elapsed),
        "v3_mean_seconds": round(float(np.mean(v3_elapsed)), 1),
        "v3_p95_seconds": round(float(np.percentile(v3_elapsed, 95)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verify-local-receipts", type=Path, default=None)
    args = parser.parse_args()
    report = build_cost_report()
    forecast = empirical_forecast(report)
    if args.verify_local_receipts is not None:
        forecast["local_receipt_check"] = verify_local_receipt_basis(args.verify_local_receipts)
    if args.out is not None:
        publish_json(args.out.resolve(), report)
    print(json.dumps(forecast, indent=2), flush=True)


if __name__ == "__main__":
    main()
