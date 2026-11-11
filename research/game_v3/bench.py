"""Throughput and stage-split benchmark for Game V3 (sim + obs)."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch

from research.game_v3.config import GameV3Config
from research.game_v3.obs import ObservationBatch, allocate_obs_batch, build_obs
from research.game_v3.rng import PURPOSE_SCRIPTED, rng_int_torch
from research.game_v3.sim import GameV3Sim

STAGE_ORDER: tuple[str, ...] = ("move", "rasterise", "collide", "food", "respawn", "obs")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _release_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _tensor_bytes_of_sim_and_obs(sim: GameV3Sim, obs: ObservationBatch | None) -> int:
    total = 0
    seen: set[int] = set()
    for val in sim.__dict__.values():
        if isinstance(val, torch.Tensor) and val.data_ptr() not in seen:
            seen.add(val.data_ptr())
            total += val.numel() * val.element_size()
    if obs is not None:
        for t in (obs.raster_s1, obs.raster_s4, obs.raster_s16, obs.scalars, obs.action_mask):
            if t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                total += t.numel() * t.element_size()
    return total


def _sample_memory_mib(
    device: torch.device, sim: GameV3Sim, obs: ObservationBatch | None
) -> dict[str, float]:
    footprint_mib = _tensor_bytes_of_sim_and_obs(sim, obs) / (1024.0 * 1024.0)
    rss_mib = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    if device.type == "cuda":
        alloc_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        reserved_mib = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        return {
            "peak_memory_mib": round(max(alloc_mib, footprint_mib), 2),
            "device_reserved_mib": round(reserved_mib, 2),
            "tensor_footprint_mib": round(footprint_mib, 2),
            "host_rss_mib": round(rss_mib, 2),
        }
    if device.type == "mps":
        alloc_mib = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
        driver_mib = torch.mps.driver_allocated_memory() / (1024.0 * 1024.0)
        return {
            "peak_memory_mib": round(max(driver_mib, alloc_mib, footprint_mib), 2),
            "mps_allocated_mib": round(alloc_mib, 2),
            "mps_driver_mib": round(driver_mib, 2),
            "tensor_footprint_mib": round(footprint_mib, 2),
            "host_rss_mib": round(rss_mib, 2),
        }
    return {
        "peak_memory_mib": round(max(rss_mib, footprint_mib), 2),
        "tensor_footprint_mib": round(footprint_mib, 2),
        "host_rss_mib": round(rss_mib, 2),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def benchmark_single_grid(
    cfg: GameV3Config,
    *,
    num_envs: int,
    device: torch.device,
    warmup_steps: int,
    bench_steps: int,
    stage_steps: int,
) -> dict[str, Any]:
    """Benchmark a single (E, N) configuration and return throughput + stage breakdown."""
    label = f"grid_E{num_envs}_N{cfg.num_snakes}_{device.type}"
    t_case_start = time.perf_counter()
    print(f"START {label} {_iso_now()}", flush=True)

    _release_device_cache(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    seeds = [10_000 + e for e in range(num_envs)]
    sim = GameV3Sim(cfg, num_envs=num_envs, seeds=seeds, device=device)
    obs_buf = allocate_obs_batch(sim)

    total_tape = max(warmup_steps + bench_steps + stage_steps, 16)
    slots = torch.arange(cfg.num_snakes, dtype=torch.int64, device=device)[None, :]
    action_tape = [
        rng_int_torch(sim.seeds[:, None], t + 1, PURPOSE_SCRIPTED, slots, 6).to(torch.int64)
        for t in range(total_tape)
    ]

    # 1. Warmup pass (sim + obs).
    for t in range(warmup_steps):
        sim.step(action_tape[t])
        build_obs(sim, out=obs_buf)
    _sync_device(device)
    mem_after_warmup = _sample_memory_mib(device, sim, obs_buf)

    # 2. Uninstrumented pipelined end-to-end loop (sim + obs, zero inner syncs).
    _sync_device(device)
    t0 = time.perf_counter()
    for t in range(bench_steps):
        sim.step(action_tape[warmup_steps + t])
        build_obs(sim, out=obs_buf)
    _sync_device(device)
    e2e_elapsed = max(1e-9, time.perf_counter() - t0)
    mem_after_e2e = _sample_memory_mib(device, sim, obs_buf)

    # 3. Uninstrumented sim-only loop (zero inner syncs).
    _sync_device(device)
    t_sim0 = time.perf_counter()
    for t in range(bench_steps):
        sim.step(action_tape[t % len(action_tape)])
    _sync_device(device)
    sim_only_elapsed = max(1e-9, time.perf_counter() - t_sim0)

    # 4. Synchronized per-stage split (move, rasterise, collide, food, respawn, obs).
    sim.set_stage_timing(True)
    obs_seconds = 0.0
    _sync_device(device)
    for t in range(stage_steps):
        sim.step(action_tape[(warmup_steps + bench_steps + t) % len(action_tape)])
        _sync_device(device)
        t_obs0 = time.perf_counter()
        build_obs(sim, out=obs_buf)
        _sync_device(device)
        obs_seconds += time.perf_counter() - t_obs0

    stage_sec = sim.consume_stage_timing()
    sim.set_stage_timing(False)
    stage_sec["obs"] = obs_seconds

    total_stage_sec = max(1e-9, sum(stage_sec[k] for k in STAGE_ORDER))
    stage_ms_per_tick = {
        k: round((stage_sec[k] / max(1, stage_steps)) * 1000.0, 4) for k in STAGE_ORDER
    }
    stage_pct = {k: round((stage_sec[k] / total_stage_sec) * 100.0, 2) for k in STAGE_ORDER}

    agent_steps = num_envs * cfg.num_snakes * bench_steps
    agent_steps_per_sec = agent_steps / e2e_elapsed
    sim_only_agent_steps_per_sec = agent_steps / sim_only_elapsed
    peak_mem_mib = max(mem_after_warmup["peak_memory_mib"], mem_after_e2e["peak_memory_mib"])
    mem_report = dict(mem_after_e2e)
    mem_report["peak_memory_mib"] = round(peak_mem_mib, 2)

    case_elapsed = time.perf_counter() - t_case_start
    print(
        f"END {label} {_iso_now()} elapsed_s={case_elapsed:.3f} "
        f"agent_steps_per_s={agent_steps_per_sec:,.1f} "
        f"sim_only_agent_steps_per_s={sim_only_agent_steps_per_sec:,.1f} "
        f"peak_mem_mib={peak_mem_mib:.1f}",
        flush=True,
    )

    del action_tape, obs_buf, sim
    _release_device_cache(device)

    return {
        "device": str(device),
        "num_envs": num_envs,
        "num_snakes": cfg.num_snakes,
        "agents_total": num_envs * cfg.num_snakes,
        "warmup_steps": warmup_steps,
        "bench_steps": bench_steps,
        "stage_steps": stage_steps,
        "e2e_wall_seconds": round(e2e_elapsed, 6),
        "sim_only_wall_seconds": round(sim_only_elapsed, 6),
        "agent_steps_per_sec": round(agent_steps_per_sec, 2),
        "sim_only_agent_steps_per_sec": round(sim_only_agent_steps_per_sec, 2),
        "world_steps_per_sec": round((num_envs * bench_steps) / e2e_elapsed, 2),
        "e2e_ms_per_tick": round((e2e_elapsed / max(1, bench_steps)) * 1000.0, 4),
        "sim_only_ms_per_tick": round((sim_only_elapsed / max(1, bench_steps)) * 1000.0, 4),
        "stage_ms_per_tick": stage_ms_per_tick,
        "stage_share_pct": stage_pct,
        "memory": mem_report,
    }


def run_benchmark(
    *,
    device: str | torch.device = "cpu",
    envs_list: Sequence[int] = (64, 256, 1024),
    num_snakes: int = 24,
    arena_radius: float = 600.0,
    warmup_steps: int = 5,
    bench_steps: int = 25,
    stage_steps: int = 15,
    json_output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the Game V3 benchmark across ``envs_list`` and optionally write JSON output."""
    req_dev = torch.device(device)
    dev = (
        torch.device(req_dev.type, 0)
        if req_dev.type in ("cuda", "mps") and req_dev.index is None
        else req_dev
    )
    if dev.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(dev)

    cfg = GameV3Config(arena_radius=arena_radius, num_snakes=num_snakes)
    t_all_start = time.perf_counter()
    start_iso = _iso_now()
    print(f"START game_v3_bench {start_iso} device={dev}", flush=True)

    cases: list[dict[str, Any]] = []
    for num_envs in envs_list:
        cases.append(
            benchmark_single_grid(
                cfg,
                num_envs=int(num_envs),
                device=dev,
                warmup_steps=warmup_steps,
                bench_steps=bench_steps,
                stage_steps=stage_steps,
            )
        )

    end_iso = _iso_now()
    total_elapsed = time.perf_counter() - t_all_start
    print(f"END game_v3_bench {end_iso} elapsed_s={total_elapsed:.3f}", flush=True)

    result: dict[str, Any] = {
        "schema": "game-v3-bench/v1",
        "started_at": start_iso,
        "ended_at": end_iso,
        "elapsed_seconds": round(total_elapsed, 4),
        "torch_version": torch.__version__,
        "device": str(dev),
        "config": cfg.to_dict(),
        "cases": cases,
    }
    if json_output is not None:
        _atomic_write_json(Path(json_output), result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Game V3 batched sim + obs.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu, mps, cuda).")
    parser.add_argument(
        "--envs",
        type=str,
        default="64,256,1024",
        help="Comma-separated list of world counts E.",
    )
    parser.add_argument("--snakes", type=int, default=24, help="Snakes per world N.")
    parser.add_argument("--arena-radius", type=float, default=600.0, help="Arena radius R.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup ticks per grid.")
    parser.add_argument("--bench-steps", type=int, default=25, help="Timed pipelined ticks.")
    parser.add_argument("--stage-steps", type=int, default=15, help="Stage-split timed ticks.")
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to write JSON benchmark summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for Game V3 benchmarking."""
    args = _parse_args(argv)
    envs_list = [int(tok.strip()) for tok in args.envs.split(",") if tok.strip()]
    run_benchmark(
        device=args.device,
        envs_list=envs_list,
        num_snakes=args.snakes,
        arena_radius=args.arena_radius,
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
        stage_steps=args.stage_steps,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    main()
