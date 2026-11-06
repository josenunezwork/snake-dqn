"""Mac canary for exact-stop-l1-os vs exact-stop-l1 on the Bank-v2 DEV namespace.

Dev-only (namespace ``BANK_V2_DEV_NAMESPACE``): never training, tuning or release
evidence, never the screen worlds.  For each short world it runs both controllers on
the same seed and reports

  (i)   frame identity: every recorded Watch frame of l1-os before its first
        action-changing open-stop override / scaled-horizon override must be
        byte-identical to l1 (open-stop silent and L < 800 => 100% identical);
        optionally (--silent-check) a third run with open-stop forced silent must be
        byte-identical to l1 over the whole world while L < 800;
  (ii)  open-stop roots and fire (override) count;
  (iii) per-decision wall cost p50/p95 per controller (search calls and whole frames);

plus long-body serpentine timings of the exact search at H=300 and the l1-os scaled H,
and of one open-stop root, for screen sizing.

Usage (foreground, Mac, one thread):
  OMP_NUM_THREADS=1 SNAKE_CKPT_DIR=.../saved_snakes python -m research.watch_replay.canary_l1_os \\
      --out <dir>/canary.json --worlds 3 --horizon 600
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import numpy as np

from research.watch_replay import engine, engine_l1_os
from research.watch_replay.seed_audit import (
    BANK_V2_DEV_COUNT,
    BANK_V2_DEV_NAMESPACE,
    bank_v2_disjointness,
    seed_for,
)
from web.backend.watch_replay import canonical

L1 = "exact-stop-l1"
L1_OS = engine_l1_os.L1_OS_CONTROLLER


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_ms": None if not values else 1000 * _pct(values, 50),
        "p95_ms": None if not values else 1000 * _pct(values, 95),
        "mean_ms": None if not values else 1000 * float(np.mean(values)),
        "total_s": float(np.sum(values)) if values else 0.0,
    }


def _timed(fn: Callable, sink: list[float]) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            sink.append(time.perf_counter() - started)

    return wrapper


def run_controller(
    seed: int, controller: str, horizon: int, *, silent_open_stop: bool = False
) -> dict[str, Any]:
    """One run_arm session; returns canonical frame bytes, raw result and timings."""
    lines: list[bytes] = []
    frame_walls: list[float] = []
    exact_walls: list[float] = []
    open_walls: list[float] = []
    last = [time.perf_counter()]

    def record(snapshot: dict) -> None:
        # Same normalisation as research/watch_replay/run.py (drop v8 wall diagnostics).
        veto = snapshot["session"].get("safety_veto")
        if isinstance(veto, dict):
            veto.pop("diagnostics", None)
        lines.append(canonical(snapshot))
        now = time.perf_counter()
        frame_walls.append(now - last[0])
        last[0] = now

    def silent_open_stop_batch(world: Any, decision: int, q: np.ndarray, legal: np.ndarray) -> dict:
        bare = int(np.where(legal, q.astype(np.float64), -np.inf).argmax())
        return {"chosen": bare, "bare": bare, "silenced": True}

    open_fn = silent_open_stop_batch if silent_open_stop else engine_l1_os.open_stop_batch

    started = time.perf_counter()
    with (
        patch.object(engine, "dense_batch", _timed(engine.dense_batch, exact_walls)),
        patch.object(
            engine_l1_os, "dense_batch_h", _timed(engine_l1_os.dense_batch_h, exact_walls)
        ),
        patch.object(engine_l1_os, "open_stop_batch", _timed(open_fn, open_walls)),
    ):
        last[0] = time.perf_counter()
        raw = engine_l1_os.run_controller(
            seed, controller, lambda: None, horizon=horizon, record=record
        )
    return {
        "lines": lines,
        "raw": raw,
        "wall_s": time.perf_counter() - started,
        "frame_walls": frame_walls,
        "exact_walls": exact_walls,
        "open_walls": open_walls,
    }


def first_divergence_cause(raw: dict) -> int | None:
    """First frame at which l1-os could legitimately differ from l1."""
    frames = [b["frame"] for b in raw.get("open_stop_batches", []) if b["chosen"] != b["bare"]]
    frames += [b["frame"] for b in raw["batches"] if int(b.get("exact_horizon", 300)) > 300]
    return min(frames) if frames else None


def compare(l1: dict, os_run: dict) -> dict[str, Any]:
    cause = first_divergence_cause(os_run["raw"])
    limit = len(l1["lines"]) if cause is None else cause - 1
    identical_prefix = 0
    for a, b in zip(l1["lines"][:limit], os_run["lines"][:limit]):
        if a != b:
            break
        identical_prefix += 1
    first_diff = next(
        (i + 1 for i, (a, b) in enumerate(zip(l1["lines"], os_run["lines"])) if a != b),
        None,
    )
    return {
        "first_override_frame": cause,
        "frames_required_identical": limit,
        "frames_identical_in_required": identical_prefix,
        "identity_pct": 100.0 if limit == 0 else 100.0 * identical_prefix / limit,
        "first_differing_frame": first_diff,
        "pass": identical_prefix == limit,
    }


def controller_stats(run: dict) -> dict[str, Any]:
    raw = run["raw"]
    decisions = raw["decisions"]
    os_batches = raw.get("open_stop_batches", [])
    n_dec = len(decisions)
    search = sum(run["exact_walls"]) + sum(run["open_walls"])
    return {
        "wall_s": run["wall_s"],
        "decisions": n_dec,
        "max_hero_length": max((d["length"] for d in decisions), default=0),
        "hero_deaths": sum(f["death"] for f in raw["frames"]),
        "mass_mean": sum(f["mass"] for f in raw["frames"]) / max(1, len(raw["frames"])),
        "exact_roots": len(raw["batches"]),
        "exact_overrides": sum(b["chosen"] != b["bare"] for b in raw["batches"]),
        "exact_branches": sum(int(b["branches_evaluated"]) for b in raw["batches"]),
        "scaled_horizon_roots": sum(int(b.get("exact_horizon", 300)) > 300 for b in raw["batches"]),
        "open_stop_roots": len(os_batches),
        "open_stop_fires": sum(b["chosen"] != b["bare"] for b in os_batches),
        "open_stop_branches": sum(int(b.get("branches_evaluated", 0)) for b in os_batches),
        "exact_root_cost": _summary(run["exact_walls"]),
        "open_stop_root_cost": _summary(run["open_walls"]),
        "frame_cost": _summary(run["frame_walls"]),
        "search_s_per_decision_mean_ms": 1000 * search / max(1, n_dec),
    }


def long_body_timings(seed: int, lengths: tuple[int, ...]) -> list[dict[str, Any]]:
    """Exact search at H300 and scaled H, plus one open-stop root, on serpentine bodies."""
    from research.bl_ppo.collector import _prepare_sim
    from research.bl_served_search.ceiling import _legal
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from research.watch_replay.product import open_watch
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.grid_scenarios import inject_serpentines

    rows = []
    for hero_length in lengths:
        session = open_watch({"seed": seed}, 24, "bare")
        try:
            world = session.world
            # Equal-height bands: the hero band must hold L cells (6 bands fit the arena
            # up to L ~= 1,550 at the product grid size).
            box = max(8, -(-int(hero_length) // (int(world.sim.grid_w) - 4)))
            inject_serpentines([world.sim], (hero_length, 600, 400, 300, 150, 150), box_height=box)
            _prepare_sim(world.sim)
            world.runtime.prepare(world.sim)
            obs = build_ego_raster(
                world.sim,
                EgoRasterConfig(version="bv8"),
                backend="numba",
                rows=np.array([[0, 0]]),
            )
            q = prepared_q(world, obs)
            legal, _ = _legal(q, world.sim.get_advisory_action_mask()[0, 0])
            length = int(world.sim.length[0, 0])
            row: dict[str, Any] = {"hero_length": length}
            for label, h in (
                ("h300", 300),
                ("scaled", engine_l1_os.exact_horizon_for(L1_OS, length)),
            ):
                started = time.perf_counter()
                result = (
                    engine.dense_batch(world, 0, q, legal)
                    if h == 300
                    else engine_l1_os.dense_batch_h(world, 0, q, legal, h)
                )
                seconds = time.perf_counter() - started
                frames = sum(
                    int(o.get("frames_executed", 0))
                    for o in result["outcomes"].values()
                    if isinstance(o, dict) and "frames_executed" in o
                )
                row[label] = {
                    "horizon": h,
                    "seconds": seconds,
                    "branches": result["branches_evaluated"],
                    "branch_frames": frames,
                    "ms_per_branch_frame": 1000 * seconds / max(1, frames),
                }
            started = time.perf_counter()
            ostop = engine_l1_os.open_stop_batch(world, 0, q, legal)
            row["open_stop"] = {
                "seconds": time.perf_counter() - started,
                "branches": ostop["branches_evaluated"],
                "fired": ostop["chosen"] != ostop["bare"],
            }
            rows.append(row)
        finally:
            session.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--silent-check", type=int, default=1, help="worlds to re-run silenced")
    parser.add_argument("--long-lengths", type=int, nargs="*", default=[800, 1000, 1200, 1500])
    args = parser.parse_args()
    if not 1 <= args.worlds <= BANK_V2_DEV_COUNT:
        raise SystemExit(f"--worlds must be 1..{BANK_V2_DEV_COUNT} (dev namespace size)")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("run with OMP_NUM_THREADS=1")
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    audit = bank_v2_disjointness()
    report: dict[str, Any] = {
        "schema": "t34-bank-v2-canary/v1",
        "scope": "dev-only Mac canary; not screen, training, tuning or release evidence",
        "namespace": BANK_V2_DEV_NAMESPACE,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "load_avg_start": list(os.getloadavg()),
        "horizon": args.horizon,
        "seed_audit": audit["status"],
        "worlds": [],
    }
    for index in range(args.worlds):
        seed = seed_for(BANK_V2_DEV_NAMESPACE, index)
        print(f"world {index} seed {seed}: {L1}", flush=True)
        l1 = run_controller(seed, L1, args.horizon)
        print(f"world {index} seed {seed}: {L1_OS}", flush=True)
        os_run = run_controller(seed, L1_OS, args.horizon)
        world = {
            "index": index,
            "seed": seed,
            L1: controller_stats(l1),
            L1_OS: controller_stats(os_run),
            "identity": compare(l1, os_run),
        }
        if index < args.silent_check:
            print(f"world {index} seed {seed}: {L1_OS} (open-stop silenced)", flush=True)
            silent = run_controller(seed, L1_OS, args.horizon, silent_open_stop=True)
            max_len = controller_stats(silent)["max_hero_length"]
            same = sum(a == b for a, b in zip(l1["lines"], silent["lines"]))
            world["silenced_identity"] = {
                "max_hero_length": max_len,
                "frames": len(l1["lines"]),
                "identical_frames": same,
                "pass": max_len < engine_l1_os.L1_OS_SCALE_MIN_LENGTH
                and same == len(l1["lines"]) == len(silent["lines"]),
            }
        report["worlds"].append(world)
        print(json.dumps({k: world[k] for k in ("index", "identity")}), flush=True)
    if args.long_lengths:
        print("long-body timings", flush=True)
        report["long_body"] = long_body_timings(
            seed_for(BANK_V2_DEV_NAMESPACE, BANK_V2_DEV_COUNT - 1), tuple(args.long_lengths)
        )
    report["load_avg_end"] = list(os.getloadavg())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
