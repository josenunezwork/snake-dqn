"""CUDA bitwise parity (vs CPU) and throughput/memory check writing one JSON result."""

from __future__ import annotations

import argparse
import os
import platform
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.game_v3.bench import _atomic_write_json, run_benchmark
from research.game_v3.config import NUM_ACTIONS, GameV3Config
from research.game_v3.obs import LearnerIndex, allocate_obs_batch, build_obs
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import GameV3Sim

_PARITY_TENSOR_ATTRS: tuple[str, ...] = (
    "seeds",
    "world_ticks",
    "head_pos",
    "theta",
    "alive",
    "boosting",
    "length",
    "seg_count",
    "head_ptr",
    "oldest_active_subtick",
    "dist_accum_fp",
    "boost_ticks",
    "boost_drop_fp_accum",
    "respawn_timer",
    "life_age",
    "life_start",
    "truncated_pending",
    "prev_alive",
    "prev_boosting",
    "prev_head_pos",
    "prev_theta",
    "prev_length",
    "occ_hi",
    "occ_lo",
    "occ_subtick_4x",
    "occ_subtick_16x",
    "ambient_grid",
    "ambient_grid_4x",
    "ambient_grid_16x",
    "corpse_grid",
    "corpse_grid_4x",
    "corpse_grid_16x",
    "ambient_count",
    "total_spawned_snake_mass",
    "total_spawned_ambient_mass",
    "total_corpse_decay_sink",
    "total_boost_decay_sink",
    "total_length_cap_sink",
    "total_truncation_sink",
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def assert_device_equals_cpu_bitwise(
    sim_dev: GameV3Sim,
    sim_cpu: GameV3Sim,
    tick: int,
    *,
    check_rings: bool = False,
    check_obs: bool = True,
    idx_dev: LearnerIndex | None = None,
    idx_cpu: LearnerIndex | None = None,
    out_dev: Any = None,
    out_cpu: Any = None,
) -> None:
    """Assert every state, pyramid, active ring, and observation tensor matches CPU bitwise."""
    for attr in _PARITY_TENSOR_ATTRS:
        gd = getattr(sim_dev, attr).detach().cpu()
        gc = getattr(sim_cpu, attr).detach().cpu()
        if not torch.equal(gd, gc):
            raise AssertionError(f"{sim_dev.device} vs cpu tensor {attr} diverged at tick={tick}")

    occ_d = sim_dev.active_occupancy_grid().detach().cpu()
    occ_c = sim_cpu.active_occupancy_grid().detach().cpu()
    if not torch.equal(occ_d, occ_c):
        raise AssertionError(
            f"{sim_dev.device} vs cpu active_occupancy_grid diverged at tick={tick}"
        )

    if check_rings:
        max_len = int(sim_cpu.cfg.max_length)
        alive_c = sim_cpu.alive
        seg_c = sim_cpu.seg_count
        ptr_c = sim_cpu.head_ptr
        rp_d = sim_dev.ring_pos.detach().cpu()
        rc_d = sim_dev.ring_cell.detach().cpu()
        rr_d = sim_dev.ring_r_sq.detach().cpu()
        rs_d = sim_dev.ring_subtick.detach().cpu()
        for e in range(sim_cpu.E):
            for i in range(sim_cpu.N):
                cnt = int(seg_c[e, i].item())
                if not bool(alive_c[e, i].item()) or cnt <= 0:
                    continue
                idxs = (int(ptr_c[e, i].item()) - torch.arange(cnt, dtype=torch.int64)) % max_len
                if not torch.equal(rp_d[e, i, idxs], sim_cpu.ring_pos[e, i, idxs]):
                    raise AssertionError(f"ring_pos diverged at tick={tick} env={e} slot={i}")
                if not torch.equal(rc_d[e, i, idxs], sim_cpu.ring_cell[e, i, idxs]):
                    raise AssertionError(f"ring_cell diverged at tick={tick} env={e} slot={i}")
                if not torch.equal(rr_d[e, i, idxs], sim_cpu.ring_r_sq[e, i, idxs]):
                    raise AssertionError(f"ring_r_sq diverged at tick={tick} env={e} slot={i}")
                if not torch.equal(rs_d[e, i, idxs], sim_cpu.ring_subtick[e, i, idxs]):
                    raise AssertionError(f"ring_subtick diverged at tick={tick} env={e} slot={i}")

    if check_obs:
        obs_d = build_obs(sim_dev)
        obs_c = build_obs(sim_cpu)
        for field_name in ("raster_s1", "raster_s4", "raster_s16", "action_mask"):
            if not torch.equal(getattr(obs_d, field_name).cpu(), getattr(obs_c, field_name)):
                raise AssertionError(f"full obs {field_name} diverged at tick={tick}")
        if not torch.equal(
            obs_d.scalars.cpu().view(torch.uint32), obs_c.scalars.view(torch.uint32)
        ):
            raise AssertionError(f"full obs scalars diverged bitwise at tick={tick}")

        if idx_dev is not None and idx_cpu is not None:
            sub_d = build_obs(sim_dev, idx_dev, out=out_dev)
            sub_c = build_obs(sim_cpu, idx_cpu, out=out_cpu)
            for field_name in ("raster_s1", "raster_s4", "raster_s16", "action_mask"):
                if not torch.equal(getattr(sub_d, field_name).cpu(), getattr(sub_c, field_name)):
                    raise AssertionError(f"masked obs {field_name} diverged at tick={tick}")
            if not torch.equal(
                sub_d.scalars.cpu().view(torch.uint32), sub_c.scalars.view(torch.uint32)
            ):
                raise AssertionError(f"masked obs scalars diverged bitwise at tick={tick}")


def run_device_vs_cpu_parity(
    *,
    device: str | torch.device = "cuda",
    num_envs: int = 32,
    num_ticks: int = 1000,
    num_snakes: int = 24,
    learner_slots: int = 8,
    arena_radius: float = 180.0,
    check_every: int = 100,
) -> dict[str, Any]:
    """Run exact bitwise trajectory parity on ``device`` vs CPU across ``num_envs x num_ticks``."""
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for parity check, but CUDA is not available")

    t0 = time.perf_counter()
    print(
        f"START game_v3_parity device={dev} envs={num_envs} ticks={num_ticks} "
        f"snakes={num_snakes} R={arena_radius} {_iso_now()}",
        flush=True,
    )

    cfg = GameV3Config(
        arena_radius=arena_radius,
        num_snakes=num_snakes,
        respawn_delay=15,
        ambient_food_density=0.006,
    )
    seeds = [30_000 + e * 19 for e in range(num_envs)]
    sim_dev = GameV3Sim(cfg, num_envs=num_envs, device=dev, seeds=seeds)
    sim_cpu = GameV3Sim(cfg, num_envs=num_envs, device="cpu", seeds=seeds)

    k_learn = min(max(1, int(learner_slots)), num_snakes)
    mask_cpu = torch.tensor(
        [[j < k_learn for j in range(num_snakes)] for _ in range(num_envs)], dtype=torch.bool
    )
    mask_dev = mask_cpu.to(dev)
    idx_cpu = LearnerIndex(sim_cpu, mask_cpu)
    idx_dev = LearnerIndex(sim_dev, mask_dev)
    out_cpu = allocate_obs_batch(sim_cpu, idx_cpu)
    out_dev = allocate_obs_batch(sim_dev, idx_dev)

    assert_device_equals_cpu_bitwise(
        sim_dev,
        sim_cpu,
        0,
        check_rings=True,
        check_obs=True,
        idx_dev=idx_dev,
        idx_cpu=idx_cpu,
        out_dev=out_dev,
        out_cpu=out_cpu,
    )

    rng = np.random.default_rng(20261111)
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    total_deaths = 0
    total_kills = 0
    total_rollout_truncs = 0

    for t in range(1, num_ticks + 1):
        mode = t % 4
        if mode == 0:
            act_dev = food_seeking_actions(sim_dev)
            act_cpu = food_seeking_actions(sim_cpu)
            if not torch.equal(act_dev.cpu(), act_cpu):
                raise AssertionError(f"food_seeking_actions diverged at tick={t}")
        elif mode == 1:
            act_dev = random_safe_actions(sim_dev)
            act_cpu = random_safe_actions(sim_cpu)
            if not torch.equal(act_dev.cpu(), act_cpu):
                raise AssertionError(f"random_safe_actions diverged at tick={t}")
        elif mode == 2:
            act_np = rng.integers(0, NUM_ACTIONS, size=(num_envs, num_snakes), dtype=np.int64)
            act_cpu = torch.from_numpy(act_np)
            act_dev = act_cpu.to(dev)
        else:
            act_dev = food_seeking_actions(sim_dev).clone()
            act_cpu = food_seeking_actions(sim_cpu).clone()
            tail_n = max(1, num_snakes // 3)
            rand_tail = torch.from_numpy(
                rng.integers(0, NUM_ACTIONS, size=(num_envs, tail_n), dtype=np.int64)
            )
            act_cpu[:, -tail_n:] = rand_tail
            act_dev[:, -tail_n:] = rand_tail.to(dev)
            if not torch.equal(act_dev.cpu(), act_cpu):
                raise AssertionError(f"mixed actions diverged at tick={t}")

        np.add.at(action_counts, act_cpu.numpy().ravel(), 1)
        do_trunc = (t % 250) == 0
        info_dev = sim_dev.step(act_dev, truncate=do_trunc)
        info_cpu = sim_cpu.step(act_cpu, truncate=do_trunc)

        for field_name in info_cpu.__dataclass_fields__:
            if not torch.equal(getattr(info_dev, field_name).cpu(), getattr(info_cpu, field_name)):
                raise AssertionError(f"StepInfo.{field_name} diverged at tick={t}")

        total_deaths += int(info_cpu.died.sum().item())
        total_kills += int(info_cpu.kill_credit.sum().item())
        total_rollout_truncs += int(info_cpu.truncated.sum().item())

        if t == 1 or t == num_ticks or (check_every > 0 and t % check_every == 0):
            assert_device_equals_cpu_bitwise(
                sim_dev,
                sim_cpu,
                t,
                check_rings=(t == num_ticks),
                check_obs=True,
                idx_dev=idx_dev,
                idx_cpu=idx_cpu,
                out_dev=out_dev,
                out_cpu=out_cpu,
            )

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    print(
        f"END game_v3_parity {_iso_now()} elapsed_s={elapsed:.3f} "
        f"deaths={total_deaths} kills={total_kills} truncs={total_rollout_truncs}",
        flush=True,
    )

    return {
        "passed": True,
        "device": str(dev),
        "num_envs": num_envs,
        "num_ticks": num_ticks,
        "num_snakes": num_snakes,
        "learner_slots": k_learn,
        "arena_radius": arena_radius,
        "elapsed_seconds": round(elapsed, 4),
        "total_deaths": total_deaths,
        "total_kills": total_kills,
        "total_rollout_truncations": total_rollout_truncs,
        "action_counts": [int(c) for c in action_counts.tolist()],
    }


def run_cuda_check(
    *,
    device: str | torch.device = "cuda",
    parity_envs: int = 32,
    parity_ticks: int = 1000,
    parity_snakes: int = 24,
    parity_arena_radius: float = 180.0,
    bench_envs: Sequence[int] = (64, 256),
    bench_snakes_list: Sequence[int] = (24, 64),
    learner_slots: int = 8,
    bench_arena_radius: float = 600.0,
    warmup_steps: int = 10,
    bench_steps: int = 50,
    stage_steps: int = 20,
    json_output: Path | str | None = None,
) -> dict[str, Any]:
    """Run bitwise parity vs CPU plus N-list benchmarks and write a single JSON result."""
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested for cuda_check.py, but torch.cuda.is_available() is False"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=(dev.type == "mps"))

    started_at = _iso_now()
    t_start = time.perf_counter()

    parity_result = run_device_vs_cpu_parity(
        device=dev,
        num_envs=parity_envs,
        num_ticks=parity_ticks,
        num_snakes=parity_snakes,
        learner_slots=learner_slots,
        arena_radius=parity_arena_radius,
    )

    bench_results: list[dict[str, Any]] = []
    for n_snakes in bench_snakes_list:
        bench_results.append(
            run_benchmark(
                device=dev,
                envs_list=bench_envs,
                num_snakes=int(n_snakes),
                learner_slots=learner_slots,
                arena_radius=bench_arena_radius,
                warmup_steps=warmup_steps,
                bench_steps=bench_steps,
                stage_steps=stage_steps,
                json_output=None,
            )
        )

    total_elapsed = time.perf_counter() - t_start
    device_name = (
        torch.cuda.get_device_name(dev)
        if dev.type == "cuda"
        else f"{platform.processor() or platform.machine()} ({dev.type})"
    )
    payload: dict[str, Any] = {
        "schema": "game-v3-cuda-check/v1",
        "started_at": started_at,
        "finished_at": _iso_now(),
        "elapsed_seconds": round(total_elapsed, 4),
        "device": str(dev),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "parity": parity_result,
        "benchmarks": bench_results,
    }
    if json_output is not None:
        out_path = Path(json_output)
        _atomic_write_json(out_path, payload)
        print(f"WROTE {out_path} ({out_path.stat().st_size} bytes)", flush=True)
    return payload


def _parse_int_list(spec: str) -> list[int]:
    vals = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected non-empty comma-separated integer list")
    return vals


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Game V3.1 CUDA vs CPU bitwise parity + benchmark JSON runner."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target Torch device to check against CPU (default: cuda).",
    )
    parser.add_argument("--parity-envs", type=int, default=32)
    parser.add_argument("--parity-ticks", type=int, default=1000)
    parser.add_argument("--parity-snakes", type=int, default=24)
    parser.add_argument("--parity-arena-radius", type=float, default=180.0)
    parser.add_argument("--bench-envs", type=_parse_int_list, default=[64, 256])
    parser.add_argument("--bench-snakes", type=_parse_int_list, default=[24, 64])
    parser.add_argument("--learners", type=int, default=8)
    parser.add_argument("--bench-arena-radius", type=float, default=600.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--bench-steps", type=int, default=50)
    parser.add_argument("--stage-steps", type=int, default=20)
    parser.add_argument(
        "--json-out",
        type=Path,
        required=True,
        help="Output path for the single combined JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_cuda_check(
        device=args.device,
        parity_envs=args.parity_envs,
        parity_ticks=args.parity_ticks,
        parity_snakes=args.parity_snakes,
        parity_arena_radius=args.parity_arena_radius,
        bench_envs=args.bench_envs,
        bench_snakes_list=args.bench_snakes,
        learner_slots=args.learners,
        bench_arena_radius=args.bench_arena_radius,
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
        stage_steps=args.stage_steps,
        json_output=args.json_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
