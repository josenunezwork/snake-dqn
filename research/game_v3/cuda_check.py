"""CUDA bitwise parity (normal + deterministic kernels) and OOM-safe benchmark JSON runner."""

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

from research.arena_ppo.runtime import SignalStopFlag, write_create_only_json
from research.game_v3.bench import (
    _atomic_write_json,
    _release_device_cache,
    _sync_device,
    _tensor_bytes_of_sim_and_obs,
    benchmark_single_grid,
)
from research.game_v3.config import NUM_ACTIONS, OCC_OWNER_BITS, GameV3Config
from research.game_v3.obs import (
    NUM_RASTER_CHANNELS,
    NUM_SCALAR_FEATURES,
    LearnerIndex,
    allocate_obs_batch,
    build_obs,
)
from research.game_v3.rng import PURPOSE_SCRIPTED, rng_int_torch
from research.game_v3.scripted import (
    cardinal6_safe_actions,
    food_seeking_actions,
    random_safe_actions,
)
from research.game_v3.sim import AUTO_REBASE_SUBTICK_THRESHOLD, STATE_FIELDS, GameV3Sim

CUDA_CHECK_EXPORT_FILENAME: str = "cuda_check_result.json"
CUDA_CHECK_REQUIRED_EXPORTS: tuple[str, ...] = (CUDA_CHECK_EXPORT_FILENAME,)
EARLY_GAME_MEMORY_NOTE: str = (
    "Short benchmark cells (~80-200 ticks) measure early-game memory (snakes near "
    "start_length=10 with sparse body/corpse occupancy; persistent ring buffers and dense "
    "grids are preallocated at max_length, while active-segment and food-conflict temporaries "
    "reflect early-game density)."
)

_PARITY_TENSOR_ATTRS: tuple[str, ...] = tuple(
    field.name for field in STATE_FIELDS if not field.is_ring
)

_OutOfMemoryError: tuple[type[BaseException], ...] = (
    (torch.OutOfMemoryError, MemoryError) if hasattr(torch, "OutOfMemoryError") else (MemoryError,)
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _is_oom_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a CUDA/MPS/CPU out-of-memory error."""
    if isinstance(exc, _OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _estimate_world_mib(cfg: GameV3Config) -> tuple[float, float]:
    """Return analytical ``(sim_mib_per_world, all_obs_mib_per_world)`` for ``cfg``."""
    g = int(cfg.grid_size)
    g4 = g // 4
    g16 = g // 16
    n = int(cfg.num_snakes)
    max_len = int(cfg.max_length)
    sim_bytes = (
        2 * g * g * 4  # occ_hi, occ_lo
        + g4 * g4 * n * 4  # occ_subtick_4x
        + g16 * g16 * n * 4  # occ_subtick_16x
        + g * g * 3  # ambient_grid (u8) + corpse_grid (i16)
        + 2 * g4 * g4 * 4  # ambient_grid_4x + corpse_grid_4x
        + 2 * g16 * g16 * 4  # ambient_grid_16x + corpse_grid_16x
        + n * max_len * 18  # ring_pos(8) + ring_cell(4) + ring_r_sq(2) + ring_subtick(4)
        + n * 96
        + 128
    )
    f_sz = int(cfg.obs_fine_size)
    c_sz = int(cfg.obs_coarse_size)
    obs_per_slot = (
        NUM_RASTER_CHANNELS * (f_sz * f_sz + 2 * c_sz * c_sz)
        + NUM_SCALAR_FEATURES * 4
        + NUM_ACTIONS
    )
    obs_bytes = n * obs_per_slot
    mib = 1024.0 * 1024.0
    return sim_bytes / mib, obs_bytes / mib


def build_cuda_check_plan(
    *,
    device: torch.device,
    parity_envs: int,
    parity_ticks: int,
    det_parity_ticks: int,
    parity_snakes: int,
    parity_arena_radius: float,
    rebase_threshold: int | None,
    bench_envs: Sequence[int],
    bench_snakes_list: Sequence[int],
    learner_slots: int,
    bench_arena_radius: float,
    warmup_steps: int,
    bench_steps: int,
    stage_steps: int,
    rebase_reset_envs: int = 256,
    rebase_reset_snakes: int = 24,
    rebase_reset_arena_radius: float | None = None,
) -> dict[str, Any]:
    """Compute the execution plan with L4 wall-time and peak-VRAM estimates."""
    p_cfg = GameV3Config(arena_radius=parity_arena_radius, num_snakes=parity_snakes)
    p_sim_mib, p_obs_mib = _estimate_world_mib(p_cfg)
    mib_to_gb = (1024.0 * 1024.0) / 1e9
    p_state_mib = (p_sim_mib + p_obs_mib) * parity_envs
    p_peak_gb = (p_state_mib * 1.5 + 320.0) * mib_to_gb
    # Parity is dominated by CPU reference side (~0.055 s/tick at E=32, N=24 on L4 vCPU).
    p_norm_sec = max(2.0, parity_ticks * (parity_envs / 32.0) * (parity_snakes / 24.0) * 0.06)
    p_det_sec = max(1.5, det_parity_ticks * (parity_envs / 32.0) * (parity_snakes / 24.0) * 0.065)

    phases: list[dict[str, Any]] = [
        {
            "phase": "parity_normal",
            "deterministic_algorithms": False,
            "num_envs": int(parity_envs),
            "num_snakes": int(parity_snakes),
            "num_ticks": int(parity_ticks),
            "arena_radius": float(parity_arena_radius),
            "rebase_threshold": rebase_threshold,
            "exercises_reset_worlds": True,
            "est_peak_vram_gb": round(p_peak_gb, 2),
            "est_l4_seconds": round(p_norm_sec, 1),
        },
        {
            "phase": "parity_deterministic",
            "deterministic_algorithms": True,
            "num_envs": int(parity_envs),
            "num_snakes": int(parity_snakes),
            "num_ticks": int(det_parity_ticks),
            "arena_radius": float(parity_arena_radius),
            "rebase_threshold": rebase_threshold,
            "exercises_reset_worlds": True,
            "est_peak_vram_gb": round(p_peak_gb, 2),
            "est_l4_seconds": round(p_det_sec, 1),
        },
    ]

    total_ticks_per_cell = int(warmup_steps) + 3 * int(bench_steps) + 2 * int(stage_steps)
    for n_snakes in bench_snakes_list:
        b_cfg = GameV3Config(arena_radius=bench_arena_radius, num_snakes=int(n_snakes))
        sim_mib, obs_mib = _estimate_world_mib(b_cfg)
        for num_envs in bench_envs:
            e_int = int(num_envs)
            n_int = int(n_snakes)
            state_obs_mib = (sim_mib + obs_mib) * e_int
            # Chunked OBS_LEARNER_CHUNK=512 workspace + step temps (~2.2-3.2 GiB at E=256, N=64).
            workspace_mib = min(3400.0, max(384.0, 8.5 * e_int * (n_int / 24.0)))
            est_peak_gb = (state_obs_mib + workspace_mib) * mib_to_gb
            est_sec = max(
                2.0,
                total_ticks_per_cell * (e_int / 256.0) * (n_int / 24.0) * 0.14 + 3.0,
            )
            phases.append(
                {
                    "phase": f"bench_N{n_int}_E{e_int}",
                    "deterministic_algorithms": False,
                    "num_envs": e_int,
                    "num_snakes": n_int,
                    "learner_slots": min(max(1, int(learner_slots)), n_int),
                    "arena_radius": float(bench_arena_radius),
                    "total_bench_ticks": total_ticks_per_cell,
                    "state_plus_obs_mib": round(state_obs_mib, 1),
                    "est_peak_vram_gb": round(est_peak_gb, 2),
                    "est_l4_seconds": round(est_sec, 1),
                }
            )

    rr_envs = int(rebase_reset_envs)
    if rr_envs > 0:
        rr_snakes = int(rebase_reset_snakes)
        rr_radius = (
            float(bench_arena_radius)
            if rebase_reset_arena_radius is None
            else float(rebase_reset_arena_radius)
        )
        rr_cfg = GameV3Config(arena_radius=rr_radius, num_snakes=rr_snakes)
        rr_sim_mib, _ = _estimate_world_mib(rr_cfg)
        rr_state_mib = rr_sim_mib * rr_envs
        rr_ws_mib = max(256.0, 4.0 * rr_envs * (rr_snakes / 24.0))
        rr_peak_gb = (rr_state_mib + rr_ws_mib) * mib_to_gb
        rr_warm = max(2, int(warmup_steps))
        rr_sec = max(2.0, rr_warm * (rr_envs / 256.0) * (rr_snakes / 24.0) * 0.08 + 4.0)
        phases.append(
            {
                "phase": f"bench_rebase_reset_N{rr_snakes}_E{rr_envs}",
                "deterministic_algorithms": False,
                "num_envs": rr_envs,
                "num_snakes": rr_snakes,
                "arena_radius": rr_radius,
                "warmup_ticks": rr_warm,
                "exercises_rebase_subticks": True,
                "exercises_reset_worlds": True,
                "sim_state_mib": round(rr_state_mib, 1),
                "est_peak_vram_gb": round(rr_peak_gb, 2),
                "est_l4_seconds": round(rr_sec, 1),
            }
        )

    est_total_sec = sum(float(p["est_l4_seconds"]) for p in phases)
    est_max_vram_gb = max((float(p["est_peak_vram_gb"]) for p in phases), default=0.0)
    return {
        "device": str(device),
        "target_wall_minutes_l4": 15.0,
        "target_peak_vram_gb": 20.0,
        "estimated_l4_wall_seconds": round(est_total_sec, 1),
        "estimated_l4_wall_minutes": round(est_total_sec / 60.0, 2),
        "estimated_peak_vram_gb": round(est_max_vram_gb, 2),
        "within_l4_budget": bool(est_total_sec <= 900.0 and est_max_vram_gb < 20.0),
        "incremental_atomic_json_writes": True,
        "memory_measurement_note": EARLY_GAME_MEMORY_NOTE,
        "phases": phases,
    }


def _print_cuda_check_plan(plan: dict[str, Any], json_output: Path | str | None) -> None:
    """Print the execution plan before running parity and benchmark cells."""
    out_desc = str(json_output) if json_output is not None else "<in-memory>"
    print(
        f"PLAN game_v3_cuda_check device={plan['device']} "
        f"est_l4_min={plan['estimated_l4_wall_minutes']:.2f}/15.00 "
        f"est_peak_vram_gb={plan['estimated_peak_vram_gb']:.2f}/20.00 "
        f"json_out={out_desc}",
        flush=True,
    )
    for idx, ph in enumerate(plan["phases"], start=1):
        print(
            f"  [{idx}/{len(plan['phases'])}] {ph['phase']}: "
            f"det={ph['deterministic_algorithms']} "
            f"E={ph['num_envs']} N={ph['num_snakes']} R={ph['arena_radius']} "
            f"est_vram_gb={ph['est_peak_vram_gb']:.2f} est_l4_s={ph['est_l4_seconds']:.1f}",
            flush=True,
        )


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
    if sim_dev._subtick_offset != sim_cpu._subtick_offset:
        raise AssertionError(
            f"{sim_dev.device} vs cpu _subtick_offset diverged at tick={tick}: "
            f"{sim_dev._subtick_offset} != {sim_cpu._subtick_offset}"
        )
    if sim_dev.clean_step_streak != sim_cpu.clean_step_streak:
        raise AssertionError(
            f"{sim_dev.device} vs cpu clean_step_streak diverged at tick={tick}: "
            f"{sim_dev.clean_step_streak} != {sim_cpu.clean_step_streak}"
        )
    for attr in _PARITY_TENSOR_ATTRS:
        raw_d = getattr(sim_dev, attr)
        if raw_d.device != sim_dev.device:
            raise AssertionError(
                f"{attr}.device is {raw_d.device}, expected sim device {sim_dev.device}"
            )
        gd = raw_d.detach().cpu()
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
    rebase_threshold: int | None = 400,
    exercise_resets: bool = True,
    deterministic_algorithms: bool | None = None,
) -> dict[str, Any]:
    """Run exact bitwise trajectory parity on ``device`` vs CPU across ``num_envs x num_ticks``.

    Exercises automatic ``rebase_subticks`` (via a lowered ``auto_rebase_threshold``),
    explicit ``rebase_subticks()``, ``reset_worlds()`` (including one triggering auto-rebase),
    and ``bool`` / ``(E,)`` / ``(E, N)`` ``truncate=`` masks.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for parity check, but CUDA is not available")

    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    eff_rebase_thresh: int | None = None
    if rebase_threshold is not None:
        eff_rebase_thresh = (
            min(int(rebase_threshold), max(48, num_ticks))
            if num_ticks >= 20
            else int(rebase_threshold)
        )
    sim_rebase_thresh = (
        eff_rebase_thresh if eff_rebase_thresh is not None else AUTO_REBASE_SUBTICK_THRESHOLD
    )

    try:
        if deterministic_algorithms is not None:
            if deterministic_algorithms:
                if dev.type == "cuda":
                    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                torch.use_deterministic_algorithms(True, warn_only=(dev.type == "mps"))
            else:
                torch.use_deterministic_algorithms(False)

        det_active = bool(torch.are_deterministic_algorithms_enabled())
        t0 = time.perf_counter()
        print(
            f"START game_v3_parity device={dev} det={det_active} envs={num_envs} "
            f"ticks={num_ticks} snakes={num_snakes} R={arena_radius} "
            f"rebase_thresh={eff_rebase_thresh} {_iso_now()}",
            flush=True,
        )

        cfg = GameV3Config(
            arena_radius=arena_radius,
            num_snakes=num_snakes,
            respawn_delay=15,
            ambient_food_density=0.006,
        )
        seeds = [30_000 + e * 19 for e in range(num_envs)]
        sim_dev = GameV3Sim(
            cfg,
            num_envs=num_envs,
            device=dev,
            seeds=seeds,
            auto_rebase_threshold=sim_rebase_thresh,
        )
        sim_cpu = GameV3Sim(
            cfg,
            num_envs=num_envs,
            device="cpu",
            seeds=seeds,
            auto_rebase_threshold=sim_rebase_thresh,
        )

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
        rebase_events = 0
        world_reset_events = 0

        reset_tick_1: int | None = None
        reset_tick_2: int | None = None
        manual_rebase_tick: int | None = None
        if exercise_resets and num_ticks >= 20:
            reset_tick_1 = max(12, num_ticks // 3)
            reset_tick_2 = max(reset_tick_1 + 4, (2 * num_ticks) // 3)
            manual_rebase_tick = max(14, num_ticks // 2)
        elif exercise_resets and num_ticks >= 4:
            reset_tick_1 = max(2, num_ticks // 2)

        trunc_world_tick = max(3, num_ticks // 4) if num_ticks >= 8 else -1
        trunc_slot_tick = max(6, num_ticks // 2) if num_ticks >= 12 else -1

        for t in range(1, num_ticks + 1):
            mode = 4 if (num_ticks < 5 and t == num_ticks) else (t % 5)
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
            elif mode == 3:
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
            else:
                act_dev = cardinal6_safe_actions(sim_dev)
                act_cpu = cardinal6_safe_actions(sim_cpu)
                if not torch.equal(act_dev.cpu(), act_cpu):
                    raise AssertionError(f"cardinal6_safe_actions diverged at tick={t}")

            np.add.at(action_counts, act_cpu.numpy().ravel(), 1)

            trunc_cpu: bool | torch.Tensor
            trunc_dev: bool | torch.Tensor
            if t == trunc_world_tick:
                trunc_cpu = torch.tensor([(e % 2) == 0 for e in range(num_envs)], dtype=torch.bool)
                trunc_dev = trunc_cpu.to(dev)
            elif t == trunc_slot_tick:
                trunc_cpu = torch.tensor(
                    [[(e + j) % 3 == 0 for j in range(num_snakes)] for e in range(num_envs)],
                    dtype=torch.bool,
                )
                trunc_dev = trunc_cpu.to(dev)
            else:
                do_trunc = (t % 250) == 0 or (num_ticks < 250 and t == num_ticks)
                trunc_cpu = do_trunc
                trunc_dev = do_trunc

            off_before = sim_cpu._subtick_offset
            info_dev = sim_dev.step(act_dev, truncate=trunc_dev)
            info_cpu = sim_cpu.step(act_cpu, truncate=trunc_cpu)
            if sim_dev._subtick_offset != sim_cpu._subtick_offset:
                raise AssertionError(
                    f"subtick_offset diverged at tick={t}: "
                    f"{sim_dev._subtick_offset} != {sim_cpu._subtick_offset}"
                )
            if sim_cpu._subtick_offset > off_before:
                rebase_events += 1

            for field_name in info_cpu.__dataclass_fields__:
                if not torch.equal(
                    getattr(info_dev, field_name).cpu(), getattr(info_cpu, field_name)
                ):
                    raise AssertionError(f"StepInfo.{field_name} diverged at tick={t}")

            total_deaths += int(info_cpu.died.sum().item())
            total_kills += int(info_cpu.kill_credit.sum().item())
            total_rollout_truncs += int(info_cpu.truncated.sum().item())

            if manual_rebase_tick is not None and t == manual_rebase_tick:
                sh_dev = sim_dev.rebase_subticks()
                sh_cpu = sim_cpu.rebase_subticks()
                if sh_dev != sh_cpu:
                    raise AssertionError(
                        f"explicit rebase_subticks shift diverged at tick={t}: "
                        f"{sh_dev} != {sh_cpu}"
                    )
                if sh_cpu > 0:
                    rebase_events += 1
                assert_device_equals_cpu_bitwise(
                    sim_dev,
                    sim_cpu,
                    t,
                    check_rings=True,
                    check_obs=True,
                    idx_dev=idx_dev,
                    idx_cpu=idx_cpu,
                    out_dev=out_dev,
                    out_cpu=out_cpu,
                )

            if reset_tick_1 is not None and t == reset_tick_1:
                wmask_cpu = torch.tensor([(e % 4) == 0 for e in range(num_envs)], dtype=torch.bool)
                wmask_cpu[0] = True
                full_seeds = [90_000 + t * 31 + e for e in range(num_envs)]
                off_before = sim_cpu._subtick_offset
                sim_dev.reset_worlds(wmask_cpu.to(dev), seeds=full_seeds)
                sim_cpu.reset_worlds(wmask_cpu, seeds=full_seeds)
                if sim_cpu._subtick_offset > off_before:
                    rebase_events += 1
                world_reset_events += 1
                assert_device_equals_cpu_bitwise(
                    sim_dev,
                    sim_cpu,
                    t,
                    check_rings=True,
                    check_obs=True,
                    idx_dev=idx_dev,
                    idx_cpu=idx_cpu,
                    out_dev=out_dev,
                    out_cpu=out_cpu,
                )

            if reset_tick_2 is not None and t == reset_tick_2:
                wmask_cpu = torch.tensor([(e % 3) == 1 for e in range(num_envs)], dtype=torch.bool)
                if not bool(wmask_cpu.any()):
                    wmask_cpu[-1] = True
                m_sel = int(wmask_cpu.sum().item())
                sub_seeds = [120_000 + t * 17 + k for k in range(m_sel)]
                off_before = sim_cpu._subtick_offset
                saved_dev_thresh = sim_dev.auto_rebase_threshold
                saved_cpu_thresh = sim_cpu.auto_rebase_threshold
                try:
                    if eff_rebase_thresh is not None:
                        init_st = max(1, sim_cpu.max_pts * sim_cpu.tick + 1 - off_before)
                        forced_thresh = max(4, init_st)
                        sim_dev.auto_rebase_threshold = forced_thresh
                        sim_cpu.auto_rebase_threshold = forced_thresh
                    sim_dev.reset_worlds(wmask_cpu.to(dev), seeds=sub_seeds)
                    sim_cpu.reset_worlds(wmask_cpu, seeds=sub_seeds)
                finally:
                    sim_dev.auto_rebase_threshold = saved_dev_thresh
                    sim_cpu.auto_rebase_threshold = saved_cpu_thresh
                if sim_cpu._subtick_offset > off_before:
                    rebase_events += 1
                world_reset_events += 1
                assert_device_equals_cpu_bitwise(
                    sim_dev,
                    sim_cpu,
                    t,
                    check_rings=True,
                    check_obs=True,
                    idx_dev=idx_dev,
                    idx_cpu=idx_cpu,
                    out_dev=out_dev,
                    out_cpu=out_cpu,
                )

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

        final_offset = int(sim_cpu._subtick_offset)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elif dev.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
        print(
            f"END game_v3_parity det={det_active} {_iso_now()} elapsed_s={elapsed:.3f} "
            f"deaths={total_deaths} kills={total_kills} truncs={total_rollout_truncs} "
            f"rebases={rebase_events} resets={world_reset_events}",
            flush=True,
        )

        del out_dev, out_cpu, idx_dev, idx_cpu, mask_dev, mask_cpu, sim_dev, sim_cpu
        _release_device_cache(dev)

        return {
            "passed": True,
            "device": str(dev),
            "deterministic_algorithms": det_active,
            "num_envs": num_envs,
            "num_ticks": num_ticks,
            "num_snakes": num_snakes,
            "learner_slots": k_learn,
            "arena_radius": arena_radius,
            "rebase_threshold": eff_rebase_thresh,
            "rebases": rebase_events,
            "subtick_offset": final_offset,
            "world_resets": world_reset_events,
            "elapsed_seconds": round(elapsed, 4),
            "total_deaths": total_deaths,
            "total_kills": total_kills,
            "total_rollout_truncations": total_rollout_truncs,
            "action_counts": [int(c) for c in action_counts.tolist()],
        }
    finally:
        if deterministic_algorithms is not None:
            torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


def _advance_subticks_chunked_for_rebase(sim: GameV3Sim, delta: int = 4096) -> None:
    """Advance stored positive subticks in-place over 8-world chunks so rebase shift > 0."""
    chunk = 8
    d_int = int(delta)
    shift_6 = d_int << OCC_OWNER_BITS
    sim.tick += d_int
    sim.oldest_active_subtick.add_(d_int)
    for e0 in range(0, sim.E, chunk):
        e1 = min(sim.E, e0 + chunk)
        rs = sim.ring_subtick[e0:e1]
        m_zero = rs <= 0
        rs.masked_fill_(m_zero, -d_int)
        del m_zero
        rs.add_(d_int)

        for grid_1x in (sim.occ_hi[e0:e1], sim.occ_lo[e0:e1]):
            m_empty = grid_1x < 0
            grid_1x.masked_fill_(m_empty, -1 - shift_6)
            del m_empty
            grid_1x.add_(shift_6)

        sub_16x = sim.occ_subtick_16x[e0:e1]
        m_16 = sub_16x < 0
        sub_16x.masked_fill_(m_16, -1 - d_int)
        del m_16
        sub_16x.add_(d_int)

    for blk in sim._occ_4x_blocks:
        b_len = int(blk.shape[0])
        for b0 in range(0, b_len, chunk):
            sub_4x = blk[b0 : min(b_len, b0 + chunk)]
            m_4 = sub_4x < 0
            sub_4x.masked_fill_(m_4, -1 - d_int)
            del m_4
            sub_4x.add_(d_int)


def _device_memory_snapshot_bytes(device: torch.device, sim: GameV3Sim) -> tuple[int, int]:
    """Return ``(current_allocated_bytes, max_memory_allocated_bytes)`` on ``device``."""
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device)), int(
            torch.cuda.max_memory_allocated(device)
        )
    if device.type == "mps":
        cur_b = int(torch.mps.current_allocated_memory())
        drv_b = int(torch.mps.driver_allocated_memory())
        return cur_b, max(cur_b, drv_b)
    sim_b = _tensor_bytes_of_sim_and_obs(sim, None)
    return sim_b, sim_b


def benchmark_rebase_and_reset_memory(
    cfg: GameV3Config,
    *,
    num_envs: int = 256,
    device: torch.device,
    warmup_steps: int = 10,
) -> dict[str, Any]:
    """Run forced ``rebase_subticks`` and ``reset_worlds`` on ``device`` and record memory peaks.

    At ``num_envs=256, arena_radius=600`` (default), records ``max_memory_allocated`` before and
    after both forced ``rebase_subticks()`` and partial ``reset_worlds(mask)`` on CUDA (resetting
    peak stats before each operation so ``after - before`` captures that operation's exact peak
    temporary allocation alongside the cumulative peak).
    """
    label = f"rebase_reset_E{num_envs}_N{cfg.num_snakes}_R{int(cfg.arena_radius)}_{device.type}"
    t_stage0 = time.perf_counter()
    print(f"START {label} {_iso_now()}", flush=True)

    _release_device_cache(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    seeds = [50_000 + e for e in range(num_envs)]
    sim = GameV3Sim(cfg, num_envs=num_envs, seeds=seeds, device=device)

    n_warm = max(2, int(warmup_steps))
    slots = torch.arange(cfg.num_snakes, dtype=torch.int64, device=device)[None, :]
    for t in range(n_warm):
        act = rng_int_torch(sim.seeds[:, None], t + 1, PURPOSE_SCRIPTED, slots, NUM_ACTIONS).to(
            torch.int64
        )
        sim.step(act)

    frontier_st = max(2, sim.max_pts * sim.tick + 2 - sim._subtick_offset + sim.max_pts + 1)
    min_active = int(
        torch.where(sim.alive, sim.oldest_active_subtick.to(torch.int64), frontier_st).amin().item()
    )
    if min_active <= 64:
        _advance_subticks_chunked_for_rebase(sim, delta=4096)

    mib = 1024.0 * 1024.0
    sim_state_bytes = _tensor_bytes_of_sim_and_obs(sim, None)
    cuda_stats = bool(device.type == "cuda")

    # 1. Forced rebase_subticks() peak measurement.
    _sync_device(device)
    _release_device_cache(device)
    alloc_before_reb, cum_max_before_reb = _device_memory_snapshot_bytes(device, sim)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _, reb_max_before = _device_memory_snapshot_bytes(device, sim)

    t_reb0 = time.perf_counter()
    shift_applied = sim.rebase_subticks()
    _sync_device(device)
    reb_elapsed_ms = (time.perf_counter() - t_reb0) * 1000.0
    if shift_applied <= 0:
        raise AssertionError(f"Expected positive forced rebase shift, got {shift_applied}")

    alloc_after_reb, reb_max_after = _device_memory_snapshot_bytes(device, sim)
    cum_max_after_reb = max(cum_max_before_reb, reb_max_after)

    # 2. Partial reset_worlds(mask) peak measurement across half the worlds.
    reset_mask = (torch.arange(num_envs, device=device) % 2) == 0
    reset_mask[0] = True
    worlds_reset = int(reset_mask.sum().item())
    reset_seeds = [80_000 + e for e in range(num_envs)]

    _sync_device(device)
    _release_device_cache(device)
    alloc_before_rst, snap_max_before_rst = _device_memory_snapshot_bytes(device, sim)
    cum_max_before_rst = max(cum_max_after_reb, snap_max_before_rst)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _, rst_max_before = _device_memory_snapshot_bytes(device, sim)

    t_rst0 = time.perf_counter()
    sim.reset_worlds(reset_mask, seeds=reset_seeds)
    _sync_device(device)
    rst_elapsed_ms = (time.perf_counter() - t_rst0) * 1000.0

    alloc_after_rst, rst_max_after = _device_memory_snapshot_bytes(device, sim)
    cum_max_after_rst = max(cum_max_before_rst, rst_max_after)

    stage_elapsed = time.perf_counter() - t_stage0
    print(
        f"END {label} {_iso_now()} elapsed_s={stage_elapsed:.3f} "
        f"rebase_shift={shift_applied} rebase_max_mib={reb_max_before / mib:.1f}->"
        f"{reb_max_after / mib:.1f} reset_worlds={worlds_reset} "
        f"reset_max_mib={rst_max_before / mib:.1f}->{rst_max_after / mib:.1f}",
        flush=True,
    )

    del reset_mask, slots, sim
    _release_device_cache(device)

    return {
        "status": "ok",
        "oom": False,
        "deterministic_algorithms": False,
        "device": str(device),
        "num_envs": int(num_envs),
        "num_snakes": int(cfg.num_snakes),
        "arena_radius": float(cfg.arena_radius),
        "warmup_steps": n_warm,
        "sim_state_mib": round(sim_state_bytes / mib, 2),
        "sim_mib_per_world": round((sim_state_bytes / mib) / max(1, num_envs), 4),
        "cuda_memory_stats_available": cuda_stats,
        "memory_measurement_note": EARLY_GAME_MEMORY_NOTE,
        "elapsed_seconds": round(stage_elapsed, 4),
        "rebase_subticks": {
            "shift_applied": int(shift_applied),
            "elapsed_ms": round(reb_elapsed_ms, 4),
            "allocated_before_bytes": alloc_before_reb,
            "allocated_after_bytes": alloc_after_reb,
            "max_memory_allocated_before_bytes": reb_max_before,
            "max_memory_allocated_after_bytes": reb_max_after,
            "max_memory_allocated_before_mib": round(reb_max_before / mib, 2),
            "max_memory_allocated_after_mib": round(reb_max_after / mib, 2),
            "peak_temp_overhead_mib": round(max(0, reb_max_after - reb_max_before) / mib, 2),
            "cumulative_max_memory_allocated_before_mib": round(cum_max_before_reb / mib, 2),
            "cumulative_max_memory_allocated_after_mib": round(cum_max_after_reb / mib, 2),
        },
        "reset_worlds": {
            "worlds_reset": worlds_reset,
            "elapsed_ms": round(rst_elapsed_ms, 4),
            "allocated_before_bytes": alloc_before_rst,
            "allocated_after_bytes": alloc_after_rst,
            "max_memory_allocated_before_bytes": rst_max_before,
            "max_memory_allocated_after_bytes": rst_max_after,
            "max_memory_allocated_before_mib": round(rst_max_before / mib, 2),
            "max_memory_allocated_after_mib": round(rst_max_after / mib, 2),
            "peak_temp_overhead_mib": round(max(0, rst_max_after - rst_max_before) / mib, 2),
            "cumulative_max_memory_allocated_before_mib": round(cum_max_before_rst / mib, 2),
            "cumulative_max_memory_allocated_after_mib": round(cum_max_after_rst / mib, 2),
        },
    }


def _flush_check_payload(
    out_path: Path | None,
    payload: dict[str, Any],
    t_start: float,
    stage_label: str,
) -> None:
    """Refresh timestamps and write ``payload`` via atomic create-then-rename."""
    payload["finished_at"] = _iso_now()
    payload["elapsed_seconds"] = round(time.perf_counter() - t_start, 4)
    if out_path is not None:
        _atomic_write_json(out_path, payload)
        print(
            f"WROTE {out_path} after {stage_label} ({out_path.stat().st_size} bytes)",
            flush=True,
        )


def _check_stop_budget(
    t_start: float,
    max_wall_seconds: float | None,
    stop_flag: SignalStopFlag,
) -> tuple[str, str] | None:
    """Return ``(status, reason)`` if a cooperative signal or wall guard has triggered."""
    if stop_flag.stopped:
        return ("stopped_signal", f"signal:{stop_flag.signal_name}")
    if max_wall_seconds is not None and (time.perf_counter() - t_start) >= float(max_wall_seconds):
        return ("stopped_wall_guard", f"max_wall_seconds:{float(max_wall_seconds):.1f}")
    return None


def run_cuda_check(
    *,
    device: str | torch.device = "cuda",
    parity_envs: int = 32,
    parity_ticks: int = 1000,
    det_parity_ticks: int | None = None,
    parity_snakes: int = 24,
    parity_arena_radius: float = 180.0,
    rebase_threshold: int | None = 400,
    bench_envs: Sequence[int] = (64, 256),
    bench_snakes_list: Sequence[int] = (24, 64),
    learner_slots: int = 8,
    bench_arena_radius: float = 600.0,
    warmup_steps: int = 10,
    bench_steps: int = 50,
    stage_steps: int = 20,
    rebase_reset_envs: int = 256,
    rebase_reset_snakes: int = 24,
    rebase_reset_arena_radius: float | None = None,
    json_output: Path | str | None = None,
    export_dir: Path | str | None = None,
    max_wall_seconds: float | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Run normal + deterministic parity and OOM-isolated benchmark cells with atomic JSON saves."""
    req_dev = torch.device(device)
    dev = (
        torch.device(req_dev.type, 0)
        if req_dev.type in ("cuda", "mps") and req_dev.index is None
        else req_dev
    )
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested for cuda_check.py, but torch.cuda.is_available() is False"
            )
        torch.cuda.init()
        torch.cuda.set_device(dev)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    eff_det_ticks = (
        min(int(parity_ticks), 250) if det_parity_ticks is None else int(det_parity_ticks)
    )
    plan = build_cuda_check_plan(
        device=dev,
        parity_envs=parity_envs,
        parity_ticks=parity_ticks,
        det_parity_ticks=eff_det_ticks,
        parity_snakes=parity_snakes,
        parity_arena_radius=parity_arena_radius,
        rebase_threshold=rebase_threshold,
        bench_envs=bench_envs,
        bench_snakes_list=bench_snakes_list,
        learner_slots=learner_slots,
        bench_arena_radius=bench_arena_radius,
        warmup_steps=warmup_steps,
        bench_steps=bench_steps,
        stage_steps=stage_steps,
        rebase_reset_envs=rebase_reset_envs,
        rebase_reset_snakes=rebase_reset_snakes,
        rebase_reset_arena_radius=rebase_reset_arena_radius,
    )
    _print_cuda_check_plan(plan, json_output)

    out_path = Path(json_output) if json_output is not None else None
    export_root = Path(export_dir) if export_dir is not None else None
    if export_root is not None:
        export_root.mkdir(parents=True, exist_ok=True)
    device_name = (
        torch.cuda.get_device_name(dev)
        if dev.type == "cuda"
        else f"{platform.processor() or platform.machine()} ({dev.type})"
    )

    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    started_at = _iso_now()
    t_start = time.perf_counter()
    bench_results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema": "game-v3-cuda-check/v1",
        "status": "in_progress",
        "started_at": started_at,
        "finished_at": started_at,
        "elapsed_seconds": 0.0,
        "device": str(dev),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "max_wall_seconds": (float(max_wall_seconds) if max_wall_seconds is not None else None),
        "memory_measurement_note": EARLY_GAME_MEMORY_NOTE,
        "plan": plan,
        "parity": None,
        "parity_deterministic": None,
        "benchmarks": bench_results,
        "rebase_reset_benchmark": None,
        "completed_cells": 0,
        "oom_cells": 0,
    }

    stage_seq = 0
    published_final = False

    def _publish_stage(stage_label: str, stage_data: dict[str, Any]) -> None:
        nonlocal stage_seq
        if export_root is None:
            return
        stage_seq += 1
        stage_path = export_root / "stages" / f"{stage_seq:02d}_{stage_label}.json"
        write_create_only_json(
            stage_path,
            {
                "schema": "game-v3-cuda-check-stage/v1",
                "stage": stage_label,
                "finished_at": payload["finished_at"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "result": stage_data,
            },
        )

    def _publish_final_once() -> None:
        nonlocal published_final
        if export_root is None or published_final:
            return
        final_export_path = export_root / CUDA_CHECK_EXPORT_FILENAME
        write_create_only_json(final_export_path, payload)
        published_final = True
        print(
            f"PUBLISHED {final_export_path} (.sha256 + .complete) status={payload['status']}",
            flush=True,
        )

    current_stage = "init"
    early_stop_exc: RuntimeError | None = None

    with SignalStopFlag() as stop_flag:
        try:
            # 1. Normal-kernel bitwise parity (deterministic_algorithms=False).
            current_stage = "parity_normal"
            parity_result = run_device_vs_cpu_parity(
                device=dev,
                num_envs=parity_envs,
                num_ticks=parity_ticks,
                num_snakes=parity_snakes,
                learner_slots=learner_slots,
                arena_radius=parity_arena_radius,
                rebase_threshold=rebase_threshold,
                exercise_resets=True,
                deterministic_algorithms=False,
            )
            payload["parity"] = parity_result
            _flush_check_payload(out_path, payload, t_start, "parity_normal")
            _publish_stage("parity_normal", parity_result)

            stop_hit = _check_stop_budget(t_start, max_wall_seconds, stop_flag)
            if stop_hit is None:
                # 2. Separate deterministic-kernel bitwise parity (deterministic_algorithms=True).
                current_stage = "parity_deterministic"
                parity_det_result = run_device_vs_cpu_parity(
                    device=dev,
                    num_envs=parity_envs,
                    num_ticks=eff_det_ticks,
                    num_snakes=parity_snakes,
                    learner_slots=learner_slots,
                    arena_radius=parity_arena_radius,
                    rebase_threshold=rebase_threshold,
                    exercise_resets=True,
                    deterministic_algorithms=True,
                )
                payload["parity_deterministic"] = parity_det_result
                _flush_check_payload(out_path, payload, t_start, "parity_deterministic")
                _publish_stage("parity_deterministic", parity_det_result)
                stop_hit = _check_stop_budget(t_start, max_wall_seconds, stop_flag)

            # 3. Throughput & VRAM benchmarks with normal kernels (deterministic_algorithms=False).
            torch.use_deterministic_algorithms(False)
            if stop_hit is None:
                for n_snakes in bench_snakes_list:
                    stop_hit = _check_stop_budget(t_start, max_wall_seconds, stop_flag)
                    if stop_hit is not None:
                        break
                    n_int = int(n_snakes)
                    cfg = GameV3Config(arena_radius=bench_arena_radius, num_snakes=n_int)
                    group_start_iso = _iso_now()
                    t_group0 = time.perf_counter()
                    cases: list[dict[str, Any]] = []
                    group_entry: dict[str, Any] = {
                        "schema": "game-v3-bench/v2",
                        "started_at": group_start_iso,
                        "ended_at": group_start_iso,
                        "elapsed_seconds": 0.0,
                        "torch_version": torch.__version__,
                        "device": str(dev),
                        "deterministic_algorithms": False,
                        "memory_measurement_note": EARLY_GAME_MEMORY_NOTE,
                        "config": cfg.to_dict(),
                        "cases": cases,
                    }
                    bench_results.append(group_entry)

                    for num_envs in bench_envs:
                        stop_hit = _check_stop_budget(t_start, max_wall_seconds, stop_flag)
                        if stop_hit is not None:
                            break
                        e_int = int(num_envs)
                        cell_label = f"bench_N{n_int}_E{e_int}"
                        current_stage = cell_label
                        t_cell0 = time.perf_counter()
                        try:
                            case = benchmark_single_grid(
                                cfg,
                                num_envs=e_int,
                                device=dev,
                                warmup_steps=warmup_steps,
                                bench_steps=bench_steps,
                                stage_steps=stage_steps,
                                learner_slots=learner_slots,
                            )
                            case["status"] = "ok"
                            case["oom"] = False
                            case["deterministic_algorithms"] = False
                            case["memory_measurement_note"] = EARLY_GAME_MEMORY_NOTE
                            cases.append(case)
                            payload["completed_cells"] = int(payload["completed_cells"]) + 1
                        except BaseException as exc:
                            if not _is_oom_error(exc):
                                raise
                            _release_device_cache(dev)
                            cell_elapsed = time.perf_counter() - t_cell0
                            print(
                                f"OOM {cell_label} {_iso_now()} elapsed_s={cell_elapsed:.3f}: "
                                f"{exc}",
                                flush=True,
                            )
                            case = {
                                "status": "oom",
                                "oom": True,
                                "deterministic_algorithms": False,
                                "device": str(dev),
                                "num_envs": e_int,
                                "num_snakes": n_int,
                                "arena_radius": float(bench_arena_radius),
                                "learner_slots_per_env": min(max(1, int(learner_slots)), n_int),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "elapsed_seconds": round(cell_elapsed, 4),
                            }
                            cases.append(case)
                            payload["oom_cells"] = int(payload["oom_cells"]) + 1

                        group_entry["ended_at"] = _iso_now()
                        group_entry["elapsed_seconds"] = round(time.perf_counter() - t_group0, 4)
                        _flush_check_payload(out_path, payload, t_start, cell_label)
                        _publish_stage(cell_label, case)

            # 4. Forced rebase_subticks + reset_worlds memory stage (default: 256 worlds, R=600).
            if stop_hit is None and int(rebase_reset_envs) > 0:
                stop_hit = _check_stop_budget(t_start, max_wall_seconds, stop_flag)
                if stop_hit is None:
                    rr_envs = int(rebase_reset_envs)
                    rr_snakes = int(rebase_reset_snakes)
                    rr_radius = (
                        float(bench_arena_radius)
                        if rebase_reset_arena_radius is None
                        else float(rebase_reset_arena_radius)
                    )
                    rr_label = f"bench_rebase_reset_N{rr_snakes}_E{rr_envs}"
                    current_stage = rr_label
                    rr_cfg = GameV3Config(arena_radius=rr_radius, num_snakes=rr_snakes)
                    t_rr0 = time.perf_counter()
                    try:
                        rr_res = benchmark_rebase_and_reset_memory(
                            rr_cfg,
                            num_envs=rr_envs,
                            device=dev,
                            warmup_steps=warmup_steps,
                        )
                        payload["rebase_reset_benchmark"] = rr_res
                        payload["completed_cells"] = int(payload["completed_cells"]) + 1
                    except BaseException as exc:
                        if not _is_oom_error(exc):
                            raise
                        _release_device_cache(dev)
                        rr_elapsed = time.perf_counter() - t_rr0
                        print(
                            f"OOM {rr_label} {_iso_now()} elapsed_s={rr_elapsed:.3f}: {exc}",
                            flush=True,
                        )
                        rr_res = {
                            "status": "oom",
                            "oom": True,
                            "deterministic_algorithms": False,
                            "device": str(dev),
                            "num_envs": rr_envs,
                            "num_snakes": rr_snakes,
                            "arena_radius": rr_radius,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "elapsed_seconds": round(rr_elapsed, 4),
                        }
                        payload["rebase_reset_benchmark"] = rr_res
                        payload["oom_cells"] = int(payload["oom_cells"]) + 1
                    _flush_check_payload(out_path, payload, t_start, rr_label)
                    _publish_stage(rr_label, rr_res)

            if stop_hit is not None:
                stop_status, stop_reason = stop_hit
                payload["status"] = stop_status
                payload["stop_reason"] = stop_reason
                _flush_check_payload(out_path, payload, t_start, stop_status)
                _publish_final_once()
                if require_complete:
                    early_stop_exc = RuntimeError(
                        f"cuda_check stopped before completing all stages ({stop_reason})"
                    )
                    raise early_stop_exc
                return payload

            payload["status"] = (
                "completed_with_oom" if int(payload["oom_cells"]) > 0 else "completed"
            )
            _flush_check_payload(out_path, payload, t_start, "final")
            _publish_final_once()
            return payload
        except BaseException as exc:
            if exc is early_stop_exc:
                raise
            payload["status"] = "oom" if _is_oom_error(exc) else "failed"
            payload["failed_stage"] = current_stage
            payload["error_type"] = type(exc).__name__
            payload["error"] = str(exc)
            _flush_check_payload(out_path, payload, t_start, f"failed_{current_stage}")
            _publish_final_once()
            raise
        finally:
            torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


def _parse_int_list(spec: str) -> list[int]:
    vals = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected non-empty comma-separated integer list")
    return vals


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Game V3.1 CUDA vs CPU bitwise parity + OOM-safe benchmark JSON runner."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target Torch device to check against CPU (default: cuda).",
    )
    parser.add_argument("--parity-envs", type=int, default=32)
    parser.add_argument("--parity-ticks", type=int, default=1000)
    parser.add_argument(
        "--det-parity-ticks",
        type=int,
        default=None,
        help=(
            "Ticks for the separate deterministic-algorithms parity pass "
            "(default: min(ticks, 250))."
        ),
    )
    parser.add_argument("--parity-snakes", type=int, default=24)
    parser.add_argument("--parity-arena-radius", type=float, default=180.0)
    parser.add_argument(
        "--rebase-threshold",
        type=int,
        default=400,
        help="Simulator auto_rebase_threshold exercised during parity (default: 400).",
    )
    parser.add_argument("--bench-envs", type=_parse_int_list, default=[64, 256])
    parser.add_argument("--bench-snakes", type=_parse_int_list, default=[24, 64])
    parser.add_argument("--learners", type=int, default=8)
    parser.add_argument("--bench-arena-radius", type=float, default=600.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--bench-steps", type=int, default=50)
    parser.add_argument("--stage-steps", type=int, default=20)
    parser.add_argument(
        "--rebase-reset-envs",
        type=int,
        default=256,
        help=(
            "World count for forced rebase_subticks + reset_worlds memory stage at R=600 "
            "(default: 256; 0 disables)."
        ),
    )
    parser.add_argument(
        "--rebase-reset-snakes",
        type=int,
        default=24,
        help="Snake count for forced rebase_subticks + reset_worlds memory stage (default: 24).",
    )
    parser.add_argument(
        "--rebase-reset-arena-radius",
        type=float,
        default=None,
        help="Arena radius for rebase/reset memory stage (default: --bench-arena-radius = 600.0).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Incremental atomic JSON output path (live updates after every stage).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional create-only Colab export directory (.sha256 + .complete sidecars).",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Cooperative Python wall-clock guard in seconds; saves partial results on expiry.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero if stopped early by wall guard or signal (used in canary mode).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.json_out is None and args.export_dir is None:
        parser.error("at least one of --json-out or --export-dir is required")
    run_cuda_check(
        device=args.device,
        parity_envs=args.parity_envs,
        parity_ticks=args.parity_ticks,
        det_parity_ticks=args.det_parity_ticks,
        parity_snakes=args.parity_snakes,
        parity_arena_radius=args.parity_arena_radius,
        rebase_threshold=args.rebase_threshold,
        bench_envs=args.bench_envs,
        bench_snakes_list=args.bench_snakes,
        learner_slots=args.learners,
        bench_arena_radius=args.bench_arena_radius,
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
        stage_steps=args.stage_steps,
        rebase_reset_envs=args.rebase_reset_envs,
        rebase_reset_snakes=args.rebase_reset_snakes,
        rebase_reset_arena_radius=args.rebase_reset_arena_radius,
        json_output=args.json_out,
        export_dir=args.export_dir,
        max_wall_seconds=args.max_wall_seconds,
        require_complete=bool(args.require_complete),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
