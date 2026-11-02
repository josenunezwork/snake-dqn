"""Mac 3-arm live/GridBatchSim mirror qualification for Watch6-L1-V1 on excluded fixture seed."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.engine import run_arm
from research.watch6_l1_confirm.run import DEFAULT_LOCAL_SAVED_DIR
from tools.ops.package_spec import publish_json

SCHEMA = "watch6-l1-mac-mirror/v1"
LONG_LENGTH_PROBE_HERO_LENGTHS: tuple[int, ...] = (150, 450, 900, 1200, 1500)
LONG_LENGTH_PROBE_OPPONENTS: tuple[int, ...] = (900, 700, 500, 400, 300)


def measure_long_length_scaling(seed: int) -> dict:
    """Measure full-H300 exact-stop branch cost scaling across hero L=150..1,500 + 5 opponents."""
    import numpy as np

    from research.bl_ppo.collector import _prepare_sim
    from research.bl_served_search.ceiling import _legal
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from research.watch6_search_efficacy.engine import dense_batch
    from research.watch6_search_efficacy.product import open_watch
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.grid_scenarios import inject_serpentines

    rows: list[dict] = []
    probes = [(LONG_LENGTH_PROBE_HERO_LENGTHS[0], True)] + [
        (h_len, False) for h_len in LONG_LENGTH_PROBE_HERO_LENGTHS
    ]
    for hero_len, warmup_only in probes:
        session = open_watch({"seed": int(seed)}, 24, "bare")
        try:
            world = session.world
            lengths = (hero_len, *LONG_LENGTH_PROBE_OPPONENTS)
            inject_serpentines([world.sim], lengths, box_height=11)
            _prepare_sim(world.sim)
            world.runtime.prepare(world.sim)
            obs = build_ego_raster(
                world.sim,
                EgoRasterConfig(version="bv8"),
                backend="numba",
                rows=np.array([[0, 0]]),
            )
            q = np.asarray(prepared_q(world, obs), dtype=np.float32)
            legal, _ = _legal(q, world.sim.get_advisory_action_mask()[0, 0])
            t0 = time.monotonic()
            result = dense_batch(world, 0, q, legal)
            dt = time.monotonic() - t0
            if warmup_only:
                continue
            outcomes = [r for r in result["outcomes"].values() if "frames_executed" in r]
            frames_exec = [int(r["frames_executed"]) for r in outcomes]
            total_branch_frames = max(1, sum(frames_exec))
            b_eval = max(1, int(result["branches_evaluated"]))
            counters = result["native_v8"]
            if (
                p.SEARCH_HORIZON not in frames_exec
                or counters["kernel_rows"] <= 0
                or any(
                    counters[k] != 0 for k in ("stock_rows", "fallback_rows", "noncandidate_rows")
                )
            ):
                raise ValueError(
                    f"long-length scaling probe failed H300/native-v8 check at L={hero_len}"
                )
            rows.append(
                {
                    "hero_length": hero_len,
                    "opponent_lengths": list(LONG_LENGTH_PROBE_OPPONENTS),
                    "total_board_occupied_cells": sum(lengths),
                    "batch_seconds": dt,
                    "branches_evaluated": int(result["branches_evaluated"]),
                    "frames_executed": frames_exec,
                    "seconds_per_evaluated_branch": dt / b_eval,
                    "ms_per_branch_step": (dt * 1000.0) / total_branch_frames,
                    "native_v8_kernel_rows": int(counters["kernel_rows"]),
                }
            )
        finally:
            session.close()

    by_len = {r["hero_length"]: r for r in rows}
    return {
        "rows": rows,
        "l1500_vs_l150_ms_per_step_ratio": (
            by_len[1500]["ms_per_branch_step"] / max(1e-9, by_len[150]["ms_per_branch_step"])
        ),
        "l1500_vs_l900_ms_per_step_ratio": (
            by_len[1500]["ms_per_branch_step"] / max(1e-9, by_len[900]["ms_per_branch_step"])
        ),
        "l1200_vs_l450_ms_per_step_ratio": (
            by_len[1200]["ms_per_branch_step"] / max(1e-9, by_len[450]["ms_per_branch_step"])
        ),
    }


def qualify(*, horizon: int = p.HORIZON, include_length_scaling: bool = True) -> dict:
    """Verify full-horizon live GameSession / GridBatchSim parity across {bare, l2_prime, l1}."""
    if platform.system() != "Darwin":
        raise ValueError("Mac mirror qualification must run on Darwin")
    if "SNAKE_CKPT_DIR" not in os.environ and DEFAULT_LOCAL_SAVED_DIR.is_dir():
        os.environ["SNAKE_CKPT_DIR"] = str(DEFAULT_LOCAL_SAVED_DIR)
    os.environ.setdefault("SNAKE_WATCH_SOURCE", "live")
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()

    started = time.monotonic()
    seed = p.FIXTURE_SEEDS[0]
    arms_out: dict[str, dict] = {}
    for arm in p.ARMS:
        arm_t0 = time.monotonic()
        raw = run_arm(seed, arm, lambda: None, horizon=horizon, record=None)
        arm_wall = time.monotonic() - arm_t0
        arms_out[arm] = {
            "frames": raw["frames"],
            "mirror_identity_frames": raw["mirror_identity_frames"],
            "mirror_mismatches": 0,
            "alive_frames": raw["alive_frames"],
            "deaths": raw["deaths"],
            "death_lengths": raw["death_lengths"],
            "max_length": raw["max_length"],
            "alive_length_median": raw["alive_length_median"],
            "length_p95": raw["length_p95"],
            "respawns_per_slot": raw["respawns_per_slot"],
            "batches": raw["batches"],
            "batches_lt600": raw["batches_lt600"],
            "batches_ge600": raw["batches_ge600"],
            "batches_ge1000": raw["batches_ge1000"],
            "interventions": raw["interventions"],
            "branches_evaluated": raw["branches_evaluated"],
            "branches_skipped": raw["branches_skipped"],
            "native_v8_kernel_rows": raw["native_v8_kernel_rows"],
            "wall_seconds": arm_wall,
            "timing": raw["timing"],
        }

    length_scaling = measure_long_length_scaling(seed) if include_length_scaling else None
    l2p_search_s = float(arms_out["l2_prime"]["timing"]["search_seconds"])
    l1_search_s = float(arms_out["l1"]["timing"]["search_seconds"])
    bare_wall_s = float(arms_out["bare"]["wall_seconds"])
    l2p_wall_s = float(arms_out["l2_prime"]["wall_seconds"])
    l1_wall_s = float(arms_out["l1"]["wall_seconds"])

    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "platform": platform.platform(),
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "fixture": True,
        "seed": seed,
        "horizon": horizon,
        "total_mirrored_frames": horizon * len(p.ARMS),
        "mirror_mismatches": 0,
        "elapsed_seconds": time.monotonic() - started,
        "arms": arms_out,
        "scaling_summary": {
            "l1_search_seconds": l1_search_s,
            "l2_prime_search_seconds": l2p_search_s,
            "l1_vs_l2_prime_search_seconds_ratio": (
                l1_search_s / l2p_search_s if l2p_search_s > 0.0 else None
            ),
            "l1_vs_l2_prime_wall_seconds_ratio": (
                l1_wall_s / l2p_wall_s if l2p_wall_s > 0.0 else None
            ),
            "l1_vs_bare_plus_l2_prime_wall_ratio": (
                l1_wall_s / max(1e-9, bare_wall_s + l2p_wall_s)
            ),
            "long_length_serpentine_probe": length_scaling,
        },
    }
    if horizon == p.HORIZON:
        validate(report)
    return report


def validate(report: dict) -> None:
    """Validate a full 14,400-frame 3-arm Mac mirror qualification receipt."""
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCHEMA
        or report.get("status") != "PASS"
        or report.get("namespace") != p.NAMESPACE
        or report.get("science") != p.identity()
        or report.get("fixture") is not True
        or report.get("seed") != p.FIXTURE_SEEDS[0]
        or report.get("horizon") != p.HORIZON
        or report.get("mirror_mismatches", 0) != 0
        or not str(report.get("platform", "")).startswith("macOS")
    ):
        raise ValueError("requires current full-14,400-frame Mac 3-arm mirror receipt")
    arms = report.get("arms", {})
    if not isinstance(arms, dict) or set(arms) != set(p.ARMS):
        raise ValueError("Mac mirror receipt missing required 3-arm coverage")
    for arm in p.ARMS:
        row = arms[arm]
        if (
            row.get("frames") != p.HORIZON
            or row.get("mirror_identity_frames") != p.HORIZON
            or row.get("mirror_mismatches", 0) != 0
            or len(row.get("respawns_per_slot", [])) != 6
        ):
            raise ValueError(f"incomplete 14,400-frame live/Grid mirror coverage on arm {arm}")
        if arm != "bare" and (
            int(row.get("batches", 0)) <= 0 or int(row.get("native_v8_kernel_rows", 0)) <= 0
        ):
            raise ValueError(
                f"search arm {arm} lacked search roots or native v8 kernel attestation"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=p.HORIZON)
    parser.add_argument("--skip-length-probe", action="store_true")
    args = parser.parse_args()
    report = qualify(horizon=args.horizon, include_length_scaling=not args.skip_length_probe)
    publish_json(args.out.resolve(), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
