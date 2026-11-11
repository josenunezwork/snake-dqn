"""Throughput and VRAM ruler (`ruler.py`) and family canary suite for Stage 1 Arena PPO."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from research.arena_ppo import EGO_CROP_SIZE, SCALAR_DIM
from research.arena_ppo.net import (
    VALID_NET_ARCHS,
    ArenaNet,
    NetArch,
    NetSize,
    build_arena_model,
    default_actor_chunk_envs,
    default_minibatch_size,
    larger_ruler_minibatch_size,
    measure_model_flops_per_row,
)
from research.arena_ppo.obs import (
    NUM_OBS_CHANNELS,
    NUM_WORLD_CHANNELS,
    SharedWorldObs,
    build_arena_ego_obs_gpu,
    build_shared_world_obs_gpu,
)
from research.arena_ppo.ppo import (
    ArenaPPOConfig,
    ArenaPPOTrainer,
    StopRequestedError,
    estimate_rollout_buffer_bytes,
    run_training_job,
    verify_staged_checkpoint,
)
from research.arena_ppo.runtime import (
    SignalStopFlag,
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    preload_native_sim_extensions,
    sha256_file,
    synchronize_device,
    verify_completed_artifact,
    write_create_only_json,
)
from research.arena_ppo.sim import (
    DEFAULT_GRID_H,
    DEFAULT_GRID_W,
    DEFAULT_NUM_SNAKES,
    TrainSim,
    verify_indexed_vs_reference_parity,
)

DEFAULT_RULER_ARCHS: tuple[NetArch, ...] = ("ego_only", "shared_trunk", "per_seat")
DEFAULT_RULER_ENVS: tuple[int, ...] = (1024, 2048, 4096, 8192)
DEFAULT_RULER_NETS: tuple[NetSize, ...] = ("S", "M")
VARIANT_COST_WEIGHTS: dict[tuple[NetArch, NetSize], int] = {
    ("ego_only", "S"): 1,
    ("shared_trunk", "S"): 2,
    ("ego_only", "M"): 4,
    ("shared_trunk", "M"): 6,
    ("per_seat", "S"): 16,
    ("per_seat", "M"): 48,
}
DEFAULT_BURNIN_TICKS: int = 200
DEFAULT_TRAIN_ROLLOUT_LEN: int = 256
COMPILE_ESTIMATE_SECONDS: float = 180.0
MIN_ACTOR_CHUNK_ENVS: int = 64
MIN_MINIBATCH_ROWS: int = 192

RULER_REQUIRED_EXPORTS: tuple[str, ...] = ("ruler_result.json",)
TRAIN_REQUIRED_EXPORTS: tuple[str, ...] = (
    "checkpoints/final.pt",
    "arena_ppo_weights.pth",
    "train_result.json",
)
CANARY_REQUIRED_EXPORTS: tuple[str, ...] = (
    "indexed_parity.json",
    "ruler_result.json",
    "checkpoints/final.pt",
    "arena_ppo_weights.pth",
    "train_result.json",
)


def _is_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or "mps backend out of memory" in msg
    )


def _format_exception_payload(exc: BaseException) -> dict[str, str]:
    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": tb_text[-2000:],
    }


def _cell_succeeded(cell: dict[str, Any]) -> bool:
    if cell.get("oom", False) or "error" in cell:
        return False
    integ = cell.get("integrated")
    return (
        isinstance(integ, dict)
        and not integ.get("oom", True)
        and "error" not in integ
        and "agent_steps_per_sec" in integ
    )


def _cleanup_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _peak_vram_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    return None


def _reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        ensure_cuda_initialized(device)
        torch.cuda.reset_peak_memory_stats(device)


class StageSplitTimer:
    """Asynchronous CUDA-event timer on CUDA, or synchronized wall timer on MPS/CPU.

    Also tracks peak allocator VRAM bytes reached within each stage without forcing
    CUDA stream synchronization inside timed ticks.
    """

    STAGES: tuple[str, ...] = (
        "sim_step",
        "rollout_record",
        "obs_build",
        "policy_forward",
        "gae",
        "learner",
    )

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.use_cuda_events = device.type == "cuda"
        self._cuda_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            name: [] for name in self.STAGES
        }
        self._wall_totals: dict[str, float] = {name: 0.0 for name in self.STAGES}
        self._stage_peak_vram: dict[str, int | None] = {name: None for name in self.STAGES}
        self._active_stage: str | None = None
        self._active_start_ev: torch.cuda.Event | None = None
        self._active_start_wall: float = 0.0

    def start(self, stage: str) -> None:
        """Begin timing and peak-memory tracking for `stage`."""
        if self._active_stage is not None:
            raise RuntimeError(
                f"Nested StageSplitTimer.start({stage!r}) inside {self._active_stage!r}"
            )
        if stage not in self._wall_totals:
            raise KeyError(f"Unknown stage {stage!r}; expected one of {self.STAGES}")
        self._active_stage = stage
        _reset_peak_vram(self.device)
        if self.use_cuda_events:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._active_start_ev = ev
        else:
            synchronize_device(self.device)
            self._active_start_wall = time.perf_counter()

    def stop(self, stage: str) -> None:
        """Finish timing and record allocator peak VRAM for `stage`."""
        if self._active_stage != stage:
            raise RuntimeError(
                f"StageSplitTimer.stop({stage!r}) mismatch with active {self._active_stage!r}"
            )
        if self.use_cuda_events:
            assert self._active_start_ev is not None
            end_ev = torch.cuda.Event(enable_timing=True)
            end_ev.record()
            self._cuda_pairs[stage].append((self._active_start_ev, end_ev))
            self._active_start_ev = None
        else:
            synchronize_device(self.device)
            self._wall_totals[stage] += time.perf_counter() - self._active_start_wall

        stage_peak = _peak_vram_bytes(self.device)
        if stage_peak is not None:
            prev = self._stage_peak_vram[stage]
            self._stage_peak_vram[stage] = stage_peak if prev is None else max(prev, stage_peak)
        self._active_stage = None

    def summarize(self) -> tuple[dict[str, float], dict[str, int | None]]:
        """Synchronize device once and return `(stage_seconds, stage_peak_vram_bytes)`."""
        synchronize_device(self.device)
        if not self.use_cuda_events:
            secs = {name: float(self._wall_totals[name]) for name in self.STAGES}
        else:
            secs = {}
            for name in self.STAGES:
                ms_sum = sum(s_ev.elapsed_time(e_ev) for s_ev, e_ev in self._cuda_pairs[name])
                secs[name] = float(ms_sum) / 1000.0
        peaks = {name: self._stage_peak_vram[name] for name in self.STAGES}
        return secs, peaks


def _max_peak_vram(
    baseline: int | None,
    stage_peaks: dict[str, int | None],
    post: int | None = None,
) -> int | None:
    vals = [v for v in (baseline, post, *stage_peaks.values()) if v is not None]
    return max(vals) if vals else None


def _summarize_stage_timing(
    wall_sec: float,
    stage_sec: dict[str, float],
) -> tuple[float, float, dict[str, float], float]:
    sum_stages = float(sum(stage_sec.values()))
    untracked = float(wall_sec - sum_stages)
    denom = max(1e-9, float(wall_sec))
    share_pct = {k: round(100.0 * float(v) / denom, 2) for k, v in stage_sec.items()}
    untracked_pct = round(100.0 * untracked / denom, 2)
    return sum_stages, untracked, share_pct, untracked_pct


class _WorldMaturityTracker:
    """Accumulate snake length, death, and corpse-food maturity statistics on device."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.length_sum = torch.zeros((), dtype=torch.int64, device=device)
        self.alive_count = torch.zeros((), dtype=torch.int64, device=device)
        self.deaths_total = torch.zeros((), dtype=torch.int64, device=device)
        self.corpse_cells_sum = torch.zeros((), dtype=torch.int64, device=device)
        self.corpse_world_max = torch.zeros((), dtype=torch.int64, device=device)
        self.worlds_recorded: int = 0
        self.ticks_recorded: int = 0
        self._length_samples: list[torch.Tensor] = []

    @torch.no_grad()
    def record_tick(
        self,
        train_sim: TrainSim,
        died: torch.Tensor,
        *,
        sample_stride: int = 4,
    ) -> None:
        sim = train_sim.sim
        alive = sim.alive
        alive_i64 = alive.to(torch.int64)
        self.length_sum += (sim.length.to(torch.int64) * alive_i64).sum()
        self.alive_count += alive_i64.sum()
        self.deaths_total += died.to(torch.int64).sum()

        corpse_per_world = (sim.food_present & sim.corpse_food).to(torch.int64).sum(dim=1)
        self.corpse_cells_sum += corpse_per_world.sum()
        self.worlds_recorded += int(corpse_per_world.numel())
        if corpse_per_world.numel() > 0:
            self.corpse_world_max = torch.maximum(self.corpse_world_max, corpse_per_world.max())

        if self.ticks_recorded % max(1, sample_stride) == 0:
            # Subsample up to 4096 seats on device with NaN masking (zero host stream sync)
            flat_len = sim.length.reshape(-1)
            flat_alive = alive.reshape(-1)
            step = max(1, flat_len.numel() // 4096)
            sub_len = flat_len[::step].to(torch.float32)
            sub_alive = flat_alive[::step]
            nan_fill = torch.full_like(sub_len, float("nan"))
            self._length_samples.append(torch.where(sub_alive, sub_len, nan_fill))
        self.ticks_recorded += 1

    @torch.no_grad()
    def summarize(self) -> dict[str, Any]:
        alive_n = int(self.alive_count.detach().cpu().item())
        mean_len = float(self.length_sum.detach().cpu().item()) / float(max(1, alive_n))
        p95_len = mean_len
        if self._length_samples:
            cat = torch.cat(self._length_samples, dim=0)
            if cat.numel() > 0:
                q_val = float(torch.nanquantile(cat, 0.95).detach().cpu().item())
                if not math.isnan(q_val):
                    p95_len = q_val
        mean_corpse = float(self.corpse_cells_sum.detach().cpu().item()) / float(
            max(1, self.worlds_recorded)
        )
        return {
            "mean_length": round(mean_len, 3),
            "p95_length": round(p95_len, 3),
            "deaths_count": int(self.deaths_total.detach().cpu().item()),
            "mean_corpse_count": round(mean_corpse, 3),
            "max_corpse_count": int(self.corpse_world_max.detach().cpu().item()),
        }


@torch.no_grad()
def _build_mature_sim_state(
    *,
    device: torch.device,
    num_envs: int,
    grid_h: int,
    grid_w: int,
    num_snakes: int,
    max_capacity: int,
    burnin_ticks: int,
    seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run untimed `scripted_greedy_food_actions` burn-in and snapshot mature CPU sim state."""
    _cleanup_device_memory(device)
    sim = TrainSim.create(
        num_envs,
        device=device,
        seed=seed,
        grid_h=grid_h,
        grid_w=grid_w,
        num_snakes=num_snakes,
        max_capacity=max_capacity,
    )
    for _ in range(max(0, int(burnin_ticks))):
        if should_stop is not None and should_stop():
            raise StopRequestedError("Signal stop requested during untimed ruler burn-in")
        sim.prepare()
        actions = sim.scripted_greedy_food_actions()
        sim.step(actions)
    sim.assert_no_capacity_overflow()
    synchronize_device(device)
    state = sim.state_dict()
    del sim
    _cleanup_device_memory(device)
    return state


@torch.no_grad()
def measure_batch6_forward_ms(
    device: torch.device,
    net_size: NetSize,
    *,
    net_arch: NetArch = "per_seat",
    num_snakes: int = DEFAULT_NUM_SNAKES,
    grid_h: int = DEFAULT_GRID_H,
    grid_w: int = DEFAULT_GRID_W,
    ego_crop_size: int = EGO_CROP_SIZE,
    warmup_iters: int = 10,
    bench_iters: int = 40,
    compile_model: bool = False,
) -> dict[str, Any]:
    """Measure single-world (`batch = num_snakes`) policy `act` latency and exact ATen FLOPs."""
    flops_info = measure_model_flops_per_row(
        net_arch,
        net_size,
        num_snakes=num_snakes,
        grid_h=grid_h,
        grid_w=grid_w,
        ego_crop_size=ego_crop_size,
    )
    net = (
        build_arena_model(
            net_arch,
            net_size,
            grid_h=grid_h,
            grid_w=grid_w,
            compile_model=compile_model,
        )
        .to(device)
        .eval()
    )
    params = net.parameter_count()
    b_rows = int(num_snakes)
    ego = torch.randn((b_rows, NUM_OBS_CHANNELS, ego_crop_size, ego_crop_size), device=device)
    scalars = torch.randn((b_rows, SCALAR_DIM), device=device)
    mask = torch.ones((b_rows, 6), dtype=torch.bool, device=device)
    mask[: min(2, b_rows), 3:] = False

    if isinstance(net, ArenaNet):
        arena = torch.randn((b_rows, NUM_OBS_CHANNELS, grid_h, grid_w), device=device)
        for _ in range(max(1, warmup_iters)):
            _ = net.act(arena, ego, scalars, legal_mask=mask, greedy=False)
        synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(max(1, bench_iters)):
            _ = net.act(arena, ego, scalars, legal_mask=mask, greedy=False)
        synchronize_device(device)
        forward_ms = ((time.perf_counter() - t0) * 1000.0) / float(max(1, bench_iters))
        del arena
    else:
        world_arena = torch.randn((1, NUM_WORLD_CHANNELS, grid_h, grid_w), device=device)
        head_xy = torch.zeros((1, b_rows, 2), dtype=torch.long, device=device)
        head_xy[..., 0] = min(grid_w - 1, grid_w // 2)
        head_xy[..., 1] = min(grid_h - 1, grid_h // 2)
        direction = torch.arange(b_rows, dtype=torch.long, device=device).remainder(4).unsqueeze(0)
        alive = torch.ones((1, b_rows), dtype=torch.bool, device=device)
        shared_obs = SharedWorldObs(
            world_arena=world_arena,
            ego=ego,
            scalars=scalars,
            head_xy=head_xy,
            direction=direction,
            alive=alive,
        )
        for _ in range(max(1, warmup_iters)):
            _ = net.act(shared_obs, legal_mask=mask, greedy=False)
        synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(max(1, bench_iters)):
            _ = net.act(shared_obs, legal_mask=mask, greedy=False)
        synchronize_device(device)
        forward_ms = ((time.perf_counter() - t0) * 1000.0) / float(max(1, bench_iters))
        del shared_obs, world_arena, head_xy, direction, alive

    token_grid = [int(net.cfg.token_grid_h), int(net.cfg.token_grid_w)]
    del net, ego, scalars, mask
    _cleanup_device_memory(device)
    return {
        "net_arch": net_arch,
        "net_size": net_size,
        "parameter_count": int(params),
        "forward_flops_per_row": int(flops_info["forward_flops_per_row"]),
        "learner_flops_per_row": int(flops_info["learner_flops_per_row"]),
        "token_grid": token_grid,
        "batch_size": b_rows,
        "warmup_iters": int(warmup_iters),
        "bench_iters": int(bench_iters),
        "forward_ms": round(forward_ms, 4),
    }


@torch.no_grad()
def _measure_collection_only_once(
    *,
    device: torch.device,
    mature_sim_state: dict[str, Any],
    num_envs: int,
    grid_h: int,
    grid_w: int,
    num_snakes: int,
    net_arch: NetArch = "shared_trunk",
    net_size: NetSize,
    warmup_ticks: int,
    collect_ticks: int,
    actor_chunk_envs: int,
    max_capacity: int,
    compile_model: bool,
    seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    _cleanup_device_memory(device)
    _reset_peak_vram(device)

    sim = TrainSim.create(
        num_envs,
        device=device,
        seed=seed,
        grid_h=grid_h,
        grid_w=grid_w,
        num_snakes=num_snakes,
        max_capacity=max_capacity,
    )
    sim.load_state_dict(mature_sim_state)
    net = (
        build_arena_model(
            net_arch,
            net_size,
            grid_h=grid_h,
            grid_w=grid_w,
            compile_model=compile_model,
        )
        .to(device)
        .eval()
    )
    chunk = max(1, min(num_envs, int(actor_chunk_envs)))
    ring_chunk = min(512, max_capacity)
    is_per_seat = isinstance(net, ArenaNet)

    def _one_tick(
        cur_sim: TrainSim,
        cur_net: Any,
        timer: StageSplitTimer | None,
        maturity: _WorldMaturityTracker | None,
    ) -> None:
        if should_stop is not None and should_stop():
            raise StopRequestedError("Signal stop requested during collection_only ruler tick")
        if timer is not None:
            timer.start("sim_step")
        cur_sim.prepare()
        legal_mask = cur_sim.legal_action_mask()
        if timer is not None:
            timer.stop("sim_step")

        if chunk == num_envs:
            if timer is not None:
                timer.start("obs_build")
            if is_per_seat:
                obs = build_arena_ego_obs_gpu(cur_sim, ring_chunk=ring_chunk)
            else:
                shared_obs = build_shared_world_obs_gpu(cur_sim, ring_chunk=ring_chunk)
            if timer is not None:
                timer.stop("obs_build")
                timer.start("policy_forward")
            if is_per_seat:
                act_out = cur_net.act(
                    obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask, greedy=False
                )
            else:
                act_out = cur_net.act(shared_obs, legal_mask=legal_mask, greedy=False)
            actions = act_out.actions.reshape(num_envs, cur_sim.S)
            if timer is not None:
                timer.stop("policy_forward")
        else:
            actions = torch.empty((num_envs, cur_sim.S), dtype=torch.long, device=device)
            for start in range(0, num_envs, chunk):
                end = min(num_envs, start + chunk)
                sl = slice(start, end)
                if timer is not None:
                    timer.start("obs_build")
                if is_per_seat:
                    obs = build_arena_ego_obs_gpu(cur_sim, env_slice=sl, ring_chunk=ring_chunk)
                else:
                    shared_obs = build_shared_world_obs_gpu(
                        cur_sim, env_slice=sl, ring_chunk=ring_chunk
                    )
                if timer is not None:
                    timer.stop("obs_build")
                    timer.start("policy_forward")
                if is_per_seat:
                    act_out = cur_net.act(
                        obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask[sl], greedy=False
                    )
                else:
                    act_out = cur_net.act(shared_obs, legal_mask=legal_mask[sl], greedy=False)
                actions[sl] = act_out.actions.reshape(end - start, cur_sim.S)
                if timer is not None:
                    timer.stop("policy_forward")

        if timer is not None:
            timer.start("sim_step")
        info = cur_sim.step(actions)
        if timer is not None:
            timer.stop("sim_step")
        if maturity is not None:
            maturity.record_tick(cur_sim, info.died)

    for _ in range(max(1, warmup_ticks)):
        _one_tick(sim, net, None, None)
    synchronize_device(device)
    _reset_peak_vram(device)
    baseline_vram = _peak_vram_bytes(device)

    stage_timer = StageSplitTimer(device)
    maturity_tracker = _WorldMaturityTracker(device)
    t0 = time.perf_counter()
    for _ in range(max(1, collect_ticks)):
        _one_tick(sim, net, stage_timer, maturity_tracker)
    sim.assert_no_capacity_overflow()
    stage_sec, stage_peaks = stage_timer.summarize()
    wall_sec = max(1e-9, time.perf_counter() - t0)
    peak_vram = _max_peak_vram(baseline_vram, stage_peaks, _peak_vram_bytes(device))
    maturity = maturity_tracker.summarize()

    agent_steps = int(collect_ticks) * int(num_envs) * int(sim.S)
    world_steps = int(collect_ticks) * int(num_envs)
    sum_stages, untracked, stage_pct, untracked_pct = _summarize_stage_timing(wall_sec, stage_sec)

    del sim, net
    _cleanup_device_memory(device)
    return {
        "oom": False,
        "ticks": int(collect_ticks),
        "actor_chunk_envs_used": int(chunk),
        "agent_steps": agent_steps,
        "world_steps": world_steps,
        "wall_seconds": round(wall_sec, 4),
        "sum_stage_seconds": round(sum_stages, 5),
        "untracked_seconds": round(untracked, 5),
        "agent_steps_per_sec": round(agent_steps / wall_sec, 2),
        "world_steps_per_sec": round(world_steps / wall_sec, 2),
        "stage_seconds": {k: round(v, 5) for k, v in stage_sec.items()},
        "stage_share_pct": stage_pct,
        "untracked_share_pct": untracked_pct,
        "stage_peak_vram_bytes": stage_peaks,
        "peak_vram_bytes": peak_vram,
        "world_maturity": maturity,
    }


def _measure_collection_only_cell(
    *,
    device: torch.device,
    mature_sim_state: dict[str, Any],
    num_envs: int,
    grid_h: int,
    grid_w: int,
    num_snakes: int,
    net_size: NetSize,
    net_arch: NetArch = "shared_trunk",
    warmup_ticks: int,
    collect_ticks: int,
    actor_chunk_envs: int,
    max_capacity: int,
    compile_model: bool,
    seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Measure collection-only throughput with OOM halving down to `MIN_ACTOR_CHUNK_ENVS`."""
    cur_chunk = max(1, min(num_envs, int(actor_chunk_envs)))
    floor_chunk = min(num_envs, MIN_ACTOR_CHUNK_ENVS)
    oom_retries = 0
    while True:
        try:
            res = _measure_collection_only_once(
                device=device,
                mature_sim_state=mature_sim_state,
                num_envs=num_envs,
                grid_h=grid_h,
                grid_w=grid_w,
                num_snakes=num_snakes,
                net_size=net_size,
                net_arch=net_arch,
                warmup_ticks=warmup_ticks,
                collect_ticks=collect_ticks,
                actor_chunk_envs=cur_chunk,
                max_capacity=max_capacity,
                compile_model=compile_model,
                seed=seed,
                should_stop=should_stop,
            )
            res["oom_retries"] = oom_retries
            return res
        except BaseException as exc:
            if not _is_oom_exception(exc) or cur_chunk <= floor_chunk:
                raise
            oom_retries += 1
            next_chunk = max(floor_chunk, cur_chunk // 2)
            if next_chunk >= cur_chunk:
                raise
            cur_chunk = next_chunk
            _cleanup_device_memory(device)
            phase_stamp(
                f"E{num_envs}_{net_arch}_{net_size}_collection_oom_retry",
                "RETRY",
                oom_retries=oom_retries,
                actor_chunk_envs=cur_chunk,
            )


def _measure_integrated_once(
    *,
    device: torch.device,
    mature_sim_state: dict[str, Any],
    num_envs: int,
    grid_h: int,
    grid_w: int,
    num_snakes: int,
    net_size: NetSize,
    net_arch: NetArch = "shared_trunk",
    warmup_ticks: int,
    rollout_len: int,
    ppo_epochs: int,
    minibatch_size: int,
    actor_chunk_envs: int,
    max_capacity: int,
    compile_model: bool,
    measure_larger_minibatch: bool,
    seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    _cleanup_device_memory(device)
    _reset_peak_vram(device)

    total_rows = int(rollout_len) * int(num_envs) * int(num_snakes)
    eff_mb = max(num_snakes, min(total_rows, int(minibatch_size)))
    ppo_cfg = ArenaPPOConfig(
        net_arch=net_arch,
        net_size=net_size,
        num_envs=num_envs,
        num_snakes=num_snakes,
        grid_h=grid_h,
        grid_w=grid_w,
        rollout_len=rollout_len,
        ppo_epochs=ppo_epochs,
        minibatch_size=eff_mb,
        actor_chunk_envs=min(num_envs, int(actor_chunk_envs)),
        ring_chunk=min(512, max_capacity),
        max_capacity=max_capacity,
        seed=seed,
        deterministic=False,
        compile_model=compile_model,
    )
    trainer = ArenaPPOTrainer(ppo_cfg, device=device)
    trainer.sim.load_state_dict(mature_sim_state)
    is_per_seat = isinstance(trainer.model, ArenaNet)

    # Untimed warmup of forward + 1 minibatch backward so cuDNN/Inductor kernels are warm
    trainer.model.eval()
    with torch.no_grad():
        for w_tick in range(min(max(1, warmup_ticks), rollout_len)):
            if should_stop is not None and should_stop():
                raise StopRequestedError("Signal stop requested during integrated ruler warmup")
            trainer.sim.prepare()
            trainer.buffer.record_pre_step(w_tick, trainer.sim)
            lmask = trainer.sim.legal_action_mask()
            acts, lp, vals = trainer._act_all_seats_chunked(trainer.model, lmask, greedy=False)
            info = trainer.sim.step(acts)
            trainer.buffer.actions[w_tick] = acts.to(torch.uint8)
            trainer.buffer.log_probs[w_tick] = lp
            trainer.buffer.values[w_tick] = vals
            trainer.buffer.rewards[w_tick] = info.reward
            trainer.buffer.died[w_tick] = info.died
            trainer.buffer.acted[w_tick] = info.acted
            trainer.buffer.learner_mask[w_tick] = info.learner_mask

    trainer.model.train()
    w_t = torch.zeros(
        (min(num_envs, max(1, eff_mb // num_snakes)),), dtype=torch.long, device=device
    )
    w_e = torch.arange(w_t.shape[0], dtype=torch.long, device=device)
    w_act = trainer.buffer.actions[w_t, w_e].reshape(-1).to(torch.long)
    if is_per_seat:
        w_arena, w_ego, w_scal, w_mask = trainer.buffer.reconstruct_world_ticks(
            w_t, w_e, ring_chunk=ppo_cfg.ring_chunk
        )
        w_lp, w_ent, w_vl, w_al = trainer.model.evaluate_actions(
            w_arena, w_ego, w_scal, w_mask, w_act
        )
        del w_arena, w_ego, w_scal, w_mask
    else:
        w_obs, w_mask = trainer.buffer.reconstruct_shared_world_ticks(
            w_t, w_e, ring_chunk=ppo_cfg.ring_chunk
        )
        w_lp, w_ent, w_vl, w_al = trainer.model.evaluate_actions(w_obs, w_mask, w_act)
        del w_obs, w_mask
    w_loss = -w_lp.mean() - 0.01 * w_ent.mean() + 0.0 * w_vl.mean() + 0.0 * w_al.mean()
    trainer.optimizer.zero_grad(set_to_none=True)
    w_loss.backward()
    trainer.optimizer.step()
    del w_act, w_lp, w_ent, w_vl, w_al, w_loss

    # Reset world state back to exact post-burnin mature state before timed window
    trainer.sim.load_state_dict(mature_sim_state)
    synchronize_device(device)
    _reset_peak_vram(device)
    baseline_vram = _peak_vram_bytes(device)

    stage_timer = StageSplitTimer(device)
    maturity_tracker = _WorldMaturityTracker(device)
    assert ppo_cfg.actor_chunk_envs is not None
    chunk = max(1, min(num_envs, int(ppo_cfg.actor_chunk_envs)))
    trainer.model.eval()
    t0 = time.perf_counter()

    with torch.no_grad():
        for t in range(rollout_len):
            if should_stop is not None and should_stop():
                raise StopRequestedError("Signal stop requested during integrated ruler rollout")
            stage_timer.start("sim_step")
            trainer.sim.prepare()
            legal_mask = trainer.sim.legal_action_mask()
            stage_timer.stop("sim_step")

            stage_timer.start("rollout_record")
            trainer.buffer.record_pre_step(t, trainer.sim)
            stage_timer.stop("rollout_record")

            if chunk == num_envs:
                if is_per_seat:
                    stage_timer.start("obs_build")
                    obs = build_arena_ego_obs_gpu(trainer.sim, ring_chunk=ppo_cfg.ring_chunk)
                    stage_timer.stop("obs_build")

                    stage_timer.start("policy_forward")
                    act_out = trainer.model.act(
                        obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask, greedy=False
                    )
                    stage_timer.stop("policy_forward")
                else:
                    stage_timer.start("obs_build")
                    obs_sh: SharedWorldObs = build_shared_world_obs_gpu(
                        trainer.sim, ring_chunk=ppo_cfg.ring_chunk
                    )
                    stage_timer.stop("obs_build")

                    stage_timer.start("policy_forward")
                    act_out = trainer.model.act(obs_sh, legal_mask=legal_mask, greedy=False)
                    stage_timer.stop("policy_forward")
                self_actions = act_out.actions.reshape(num_envs, trainer.sim.S)
                log_probs = act_out.log_probs.reshape(num_envs, trainer.sim.S)
                values = act_out.values.reshape(num_envs, trainer.sim.S)
            else:
                self_actions = torch.empty(
                    (num_envs, trainer.sim.S), dtype=torch.long, device=device
                )
                log_probs = torch.empty(
                    (num_envs, trainer.sim.S), dtype=torch.float32, device=device
                )
                values = torch.empty((num_envs, trainer.sim.S), dtype=torch.float32, device=device)
                for start in range(0, num_envs, chunk):
                    end = min(num_envs, start + chunk)
                    sl = slice(start, end)
                    if is_per_seat:
                        stage_timer.start("obs_build")
                        obs = build_arena_ego_obs_gpu(
                            trainer.sim, env_slice=sl, ring_chunk=ppo_cfg.ring_chunk
                        )
                        stage_timer.stop("obs_build")

                        stage_timer.start("policy_forward")
                        act_out = trainer.model.act(
                            obs.arena,
                            obs.ego,
                            obs.scalars,
                            legal_mask=legal_mask[sl],
                            greedy=False,
                        )
                        stage_timer.stop("policy_forward")
                    else:
                        stage_timer.start("obs_build")
                        obs_sh = build_shared_world_obs_gpu(
                            trainer.sim, env_slice=sl, ring_chunk=ppo_cfg.ring_chunk
                        )
                        stage_timer.stop("obs_build")

                        stage_timer.start("policy_forward")
                        act_out = trainer.model.act(
                            obs_sh,
                            legal_mask=legal_mask[sl],
                            greedy=False,
                        )
                        stage_timer.stop("policy_forward")
                    self_actions[sl] = act_out.actions.reshape(end - start, trainer.sim.S)
                    log_probs[sl] = act_out.log_probs.reshape(end - start, trainer.sim.S)
                    values[sl] = act_out.values.reshape(end - start, trainer.sim.S)

            stage_timer.start("sim_step")
            info = trainer.sim.step(self_actions)
            trainer._maybe_apply_world_time_limits(t, info.survived)
            stage_timer.stop("sim_step")

            stage_timer.start("rollout_record")
            trainer.buffer.actions[t] = self_actions.to(torch.uint8)
            trainer.buffer.log_probs[t] = log_probs
            trainer.buffer.values[t] = values
            trainer.buffer.rewards[t] = info.reward
            trainer.buffer.died[t] = info.died
            trainer.buffer.acted[t] = info.acted
            trainer.buffer.learner_mask[t] = info.learner_mask
            stage_timer.stop("rollout_record")
            maturity_tracker.record_tick(trainer.sim, info.died)

        trainer.sim.assert_no_capacity_overflow()

        # Bootstrap at state T + GAE
        stage_timer.start("sim_step")
        trainer.sim.prepare()
        bootstrap_mask = trainer.sim.legal_action_mask()
        stage_timer.stop("sim_step")

        last_values = torch.empty((num_envs, trainer.sim.S), dtype=torch.float32, device=device)
        for start in range(0, num_envs, chunk):
            end = min(num_envs, start + chunk)
            sl = slice(start, end)
            if is_per_seat:
                stage_timer.start("obs_build")
                obs_b = build_arena_ego_obs_gpu(
                    trainer.sim, env_slice=sl, ring_chunk=ppo_cfg.ring_chunk
                )
                stage_timer.stop("obs_build")
                stage_timer.start("policy_forward")
                out_b = trainer.model.forward(
                    obs_b.arena, obs_b.ego, obs_b.scalars, legal_mask=bootstrap_mask[sl]
                )
                stage_timer.stop("policy_forward")
            else:
                stage_timer.start("obs_build")
                obs_b_sh = build_shared_world_obs_gpu(
                    trainer.sim, env_slice=sl, ring_chunk=ppo_cfg.ring_chunk
                )
                stage_timer.stop("obs_build")
                stage_timer.start("policy_forward")
                out_b = trainer.model.forward(obs_b_sh, legal_mask=bootstrap_mask[sl])
                stage_timer.stop("policy_forward")
            last_values[sl] = out_b.value.reshape(end - start, trainer.sim.S)

        stage_timer.start("gae")
        trainer.buffer.compute_gae_and_aux(
            last_values=last_values,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )
        stage_timer.stop("gae")

    # Snapshot pre-learner weights on CPU only if we will also probe the 2x larger minibatch
    larger_mb_target = min(total_rows, larger_ruler_minibatch_size(net_size, net_arch=net_arch))
    do_larger_probe = bool(measure_larger_minibatch and larger_mb_target > eff_mb)
    saved_model_cpu: dict[str, torch.Tensor] | None = None
    if do_larger_probe:
        saved_model_cpu = {
            k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()
        }

    stage_timer.start("learner")
    ppo_stats, learn_wall_default = trainer.ppo_update(should_stop=should_stop)
    stage_timer.stop("learner")

    stage_sec, stage_peaks = stage_timer.summarize()
    wall_sec = max(1e-9, time.perf_counter() - t0)
    peak_vram = _max_peak_vram(baseline_vram, stage_peaks, _peak_vram_bytes(device))
    maturity = maturity_tracker.summarize()

    larger_mb_report: dict[str, Any] | None = None
    if do_larger_probe and saved_model_cpu is not None:
        if should_stop is not None and should_stop():
            raise StopRequestedError("Signal stop requested before larger-minibatch probe")
        trainer.model.load_state_dict(saved_model_cpu)
        trainer.cfg = ArenaPPOConfig(**{**ppo_cfg.to_dict(), "minibatch_size": larger_mb_target})
        _cleanup_device_memory(device)
        _reset_peak_vram(device)
        try:
            larger_stats, larger_sec = trainer.ppo_update(should_stop=should_stop)
            larger_peak = _peak_vram_bytes(device)
            larger_rows = int(ppo_epochs) * total_rows
            larger_mb_report = {
                "oom": False,
                "minibatch_size": int(larger_mb_target),
                "learn_seconds": round(larger_sec, 5),
                "learner_rows_per_sec": round(larger_rows / max(1e-9, larger_sec), 2),
                "peak_vram_bytes": larger_peak,
                "epoch0_approx_kl": larger_stats["epoch0_approx_kl"],
            }
        except StopRequestedError:
            raise
        except BaseException as exc:
            if not _is_oom_exception(exc):
                raise
            larger_peak = _peak_vram_bytes(device)
            _cleanup_device_memory(device)
            larger_mb_report = {
                "oom": True,
                "minibatch_size": int(larger_mb_target),
                "peak_vram_bytes": larger_peak,
                "error": str(exc),
            }
        del saved_model_cpu

    agent_steps = int(rollout_len) * int(num_envs) * int(trainer.sim.S)
    world_steps = int(rollout_len) * int(num_envs)
    sum_stages, untracked, stage_pct, untracked_pct = _summarize_stage_timing(wall_sec, stage_sec)
    rollout_buffer_bytes = int(trainer.buffer.nbytes())
    train_rollout_len = max(int(rollout_len), DEFAULT_TRAIN_ROLLOUT_LEN)
    train_rollout_buffer_bytes = int(
        estimate_rollout_buffer_bytes(
            rollout_len=train_rollout_len,
            num_envs=num_envs,
            num_snakes=num_snakes,
            max_capacity=max_capacity,
            grid_h=grid_h,
            grid_w=grid_w,
            food_capacity=trainer.sim.sim.food_capacity,
        )
    )
    projected_train_peak_vram_bytes = (
        int(peak_vram) + max(0, train_rollout_buffer_bytes - rollout_buffer_bytes)
        if peak_vram is not None
        else None
    )
    world_ticks_per_mb = max(1, int(eff_mb) // int(num_snakes))
    learner_settings = {
        "net_arch": net_arch,
        "net_size": net_size,
        "ppo_epochs": int(ppo_epochs),
        "rollout_len": int(rollout_len),
        "minibatch_size": int(eff_mb),
        "world_ticks_per_minibatch": int(world_ticks_per_mb),
        "total_rows_per_epoch": int(total_rows),
        "total_learner_rows": int(ppo_epochs) * int(total_rows),
    }

    del trainer
    _cleanup_device_memory(device)
    return {
        "oom": False,
        "net_arch": net_arch,
        "rollout_len": int(rollout_len),
        "ppo_epochs": int(ppo_epochs),
        "actor_chunk_envs_used": int(chunk),
        "minibatch_size_used": int(eff_mb),
        "learner_settings": learner_settings,
        "agent_steps": agent_steps,
        "world_steps": world_steps,
        "wall_seconds": round(wall_sec, 4),
        "sum_stage_seconds": round(sum_stages, 5),
        "untracked_seconds": round(untracked, 5),
        "agent_steps_per_sec": round(agent_steps / wall_sec, 2),
        "world_steps_per_sec": round(world_steps / wall_sec, 2),
        "stage_seconds": {k: round(v, 5) for k, v in stage_sec.items()},
        "stage_share_pct": stage_pct,
        "untracked_share_pct": untracked_pct,
        "stage_peak_vram_bytes": stage_peaks,
        "peak_vram_bytes": peak_vram,
        "rollout_buffer_bytes": rollout_buffer_bytes,
        "train_rollout_len": train_rollout_len,
        "train_rollout_buffer_bytes": train_rollout_buffer_bytes,
        "projected_train_peak_vram_bytes": projected_train_peak_vram_bytes,
        "world_maturity": maturity,
        "ppo_stats": ppo_stats,
        "epoch0_approx_kl": ppo_stats["epoch0_approx_kl"],
        "learner_default_minibatch": {
            "minibatch_size": int(eff_mb),
            "world_ticks_per_minibatch": int(world_ticks_per_mb),
            "ppo_epochs": int(ppo_epochs),
            "learn_seconds": round(learn_wall_default, 5),
            "learner_rows_per_sec": round(
                (int(ppo_epochs) * total_rows) / max(1e-9, learn_wall_default), 2
            ),
            "peak_vram_bytes": stage_peaks["learner"],
            "epoch0_approx_kl": ppo_stats["epoch0_approx_kl"],
        },
        "learner_larger_minibatch": larger_mb_report,
    }


def _measure_integrated_cell(
    *,
    device: torch.device,
    mature_sim_state: dict[str, Any],
    num_envs: int,
    grid_h: int,
    grid_w: int,
    num_snakes: int,
    net_size: NetSize,
    net_arch: NetArch = "shared_trunk",
    warmup_ticks: int,
    rollout_len: int,
    ppo_epochs: int,
    minibatch_size: int,
    actor_chunk_envs: int,
    max_capacity: int,
    compile_model: bool,
    measure_larger_minibatch: bool,
    seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Measure integrated throughput with OOM halving down to floors."""
    total_rows = int(rollout_len) * int(num_envs) * int(num_snakes)
    cur_chunk = max(1, min(num_envs, int(actor_chunk_envs)))
    floor_chunk = min(num_envs, MIN_ACTOR_CHUNK_ENVS)
    cur_mb = max(num_snakes, min(total_rows, int(minibatch_size)))
    floor_mb = min(total_rows, max(num_snakes, MIN_MINIBATCH_ROWS))
    oom_retries = 0

    while True:
        try:
            res = _measure_integrated_once(
                device=device,
                mature_sim_state=mature_sim_state,
                num_envs=num_envs,
                grid_h=grid_h,
                grid_w=grid_w,
                num_snakes=num_snakes,
                net_size=net_size,
                net_arch=net_arch,
                warmup_ticks=warmup_ticks,
                rollout_len=rollout_len,
                ppo_epochs=ppo_epochs,
                minibatch_size=cur_mb,
                actor_chunk_envs=cur_chunk,
                max_capacity=max_capacity,
                compile_model=compile_model,
                measure_larger_minibatch=measure_larger_minibatch,
                seed=seed,
                should_stop=should_stop,
            )
            res["oom_retries"] = oom_retries
            return res
        except BaseException as exc:
            if not _is_oom_exception(exc) or (cur_chunk <= floor_chunk and cur_mb <= floor_mb):
                raise
            oom_retries += 1
            next_chunk = max(floor_chunk, cur_chunk // 2) if cur_chunk > floor_chunk else cur_chunk
            if cur_mb > floor_mb:
                halved = max(floor_mb, cur_mb // 2)
                next_mb = max(num_snakes, (halved // num_snakes) * num_snakes)
            else:
                next_mb = cur_mb
            if next_chunk >= cur_chunk and next_mb >= cur_mb:
                raise
            cur_chunk = next_chunk
            cur_mb = next_mb
            _cleanup_device_memory(device)
            phase_stamp(
                f"E{num_envs}_{net_arch}_{net_size}_integrated_oom_retry",
                "RETRY",
                oom_retries=oom_retries,
                actor_chunk_envs=cur_chunk,
                minibatch_size=cur_mb,
            )


def _cell_cost_key(item: tuple[int, NetArch, NetSize]) -> tuple[int, int, int]:
    num_envs, net_arch, net_sz = item
    mult = VARIANT_COST_WEIGHTS[(net_arch, net_sz)]
    return (int(num_envs) * mult, int(num_envs), mult)


def _best_completed_env_for_variant(
    cells: list[dict[str, Any]], net_arch: NetArch, net_sz: NetSize
) -> int | None:
    best_e: int | None = None
    best_sps = -1.0
    for c in cells:
        if (
            "error" in c
            or c.get("net_arch", "per_seat") != net_arch
            or c.get("net_size") != net_sz
            or bool(c.get("compile_model"))
            or bool(c.get("is_one_epoch_probe"))
        ):
            continue
        integ = c.get("integrated")
        coll = c.get("collection_only")
        sps = -1.0
        if (
            isinstance(integ, dict)
            and not integ.get("oom", True)
            and "agent_steps_per_sec" in integ
        ):
            sps = float(integ["agent_steps_per_sec"])
        elif isinstance(coll, dict) and not coll.get("oom", True) and "agent_steps_per_sec" in coll:
            sps = float(coll["agent_steps_per_sec"])
        if sps > best_sps:
            best_sps = sps
            best_e = int(c["num_envs"])
    return best_e


def _best_completed_env_for_net(cells: list[dict[str, Any]], net_sz: NetSize) -> int | None:
    return _best_completed_env_for_variant(cells, "per_seat", net_sz)


def _project_cell_seconds(
    *,
    num_envs: int,
    net_sz: NetSize,
    is_compile: bool,
    device: torch.device,
    sec_per_env_by_net: dict[str, float] | None = None,
    net_arch: NetArch = "per_seat",
    sec_per_env_by_variant: dict[tuple[NetArch, NetSize], float] | None = None,
    ppo_epochs_scale: float = 1.0,
    measured_compile_overhead_sec: float = 0.0,
) -> float:
    base = 0.0
    if sec_per_env_by_variant:
        key = (net_arch, net_sz)
        if key in sec_per_env_by_variant:
            base = sec_per_env_by_variant[key] * float(num_envs)
        else:
            target_w = float(VARIANT_COST_WEIGHTS[key])
            # Prefer a completed variant of the same architecture, else closest cost weight
            ref_key = min(
                sec_per_env_by_variant.keys(),
                key=lambda k: (
                    0 if k[0] == net_arch else 1,
                    abs(VARIANT_COST_WEIGHTS[k] - target_w),
                ),
            )
            ref_w = float(max(1, VARIANT_COST_WEIGHTS[ref_key]))
            base = sec_per_env_by_variant[ref_key] * (target_w / ref_w) * float(num_envs)
    elif sec_per_env_by_net:
        if net_sz in sec_per_env_by_net:
            base = sec_per_env_by_net[net_sz] * float(num_envs)
        elif net_sz == "M" and "S" in sec_per_env_by_net:
            base = 3.0 * sec_per_env_by_net["S"] * float(num_envs)
        elif net_sz == "S" and "M" in sec_per_env_by_net:
            base = (sec_per_env_by_net["M"] / 3.0) * float(num_envs)

    base = base * float(ppo_epochs_scale)
    compile_extra = (
        max(COMPILE_ESTIMATE_SECONDS, float(measured_compile_overhead_sec)) if is_compile else 0.0
    )
    return base * 1.15 + compile_extra


def _format_cell_id(
    *,
    num_envs: int,
    net_arch: NetArch,
    net_sz: NetSize,
    multi_arch_naming: bool,
    is_extra_compile: bool,
    is_one_epoch_probe: bool,
) -> str:
    stem = f"E{num_envs}_{net_arch}_{net_sz}" if multi_arch_naming else f"E{num_envs}_{net_sz}"
    if is_one_epoch_probe:
        stem = f"{stem}_ep1"
    if is_extra_compile:
        stem = f"{stem}_compile"
    return stem


def run_ruler(
    *,
    output_dir: Path,
    device: str | torch.device = "cuda",
    input_dir: Path | None = None,
    envs_list: list[int] | tuple[int, ...] = DEFAULT_RULER_ENVS,
    net_sizes: list[NetSize] | tuple[NetSize, ...] = DEFAULT_RULER_NETS,
    net_archs: Sequence[NetArch] | None = None,
    grid_h: int = DEFAULT_GRID_H,
    grid_w: int = DEFAULT_GRID_W,
    num_snakes: int = DEFAULT_NUM_SNAKES,
    burnin_ticks: int = DEFAULT_BURNIN_TICKS,
    warmup_ticks: int = 6,
    collect_ticks: int = 32,
    rollout_len: int = 64,
    ppo_epochs: int = 2,
    include_one_epoch_cell: bool = True,
    minibatch_size: int | None = None,
    actor_chunk_envs: int | None = None,
    max_capacity: int = 1600,
    compile_model: bool = False,
    include_best_compile: bool = True,
    measure_larger_minibatch: bool = True,
    max_wall_seconds: float | None = None,
    seed: int = 20261110,
    step_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute signal-safe throughput & VRAM sweep and write per-cell + summary JSONs."""
    output_dir = Path(output_dir)
    result_path = output_dir / "ruler_result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {result_path}")

    multi_arch_naming = net_archs is not None
    resolved_archs: list[NetArch] = (
        [a for a in net_archs] if net_archs is not None else ["per_seat"]
    )
    if not resolved_archs:
        raise ValueError("net_archs must not be empty")
    for arch in resolved_archs:
        if arch not in VALID_NET_ARCHS:
            raise ValueError(f"Unknown net_arch {arch!r}; expected one of {VALID_NET_ARCHS}")

    wall_start = time.monotonic()
    resolved_dev = normalize_device(device)
    phase_stamp(
        "ruler_setup",
        "START",
        device=str(resolved_dev),
        net_archs=resolved_archs,
        net_sizes=list(net_sizes),
    )
    ensure_cuda_initialized(resolved_dev)
    configure_determinism(False)
    ext_info = preload_native_sim_extensions(resolved_dev)

    total_vram_bytes: int | None = None
    device_name = resolved_dev.type
    if resolved_dev.type == "cuda":
        props = torch.cuda.get_device_properties(resolved_dev)
        total_vram_bytes = int(props.total_memory)
        device_name = str(props.name)

    phase_stamp(
        "ruler_setup",
        "END",
        device_name=device_name,
        total_vram_bytes=total_vram_bytes,
        native_extensions=ext_info,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    batch6_bench: dict[str, Any] = {}
    failed_benchmarks: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    skipped_cells: list[dict[str, Any]] = []
    reused_cells: list[str] = []
    mature_cache: dict[int, dict[str, Any]] = {}
    sec_per_env_by_variant: dict[tuple[NetArch, NetSize], float] = {}
    measured_compile_overhead_sec: float = 0.0
    stopped_reason = "completed"
    partial = False

    export_reserve_sec = (
        min(30.0, max(2.0, float(max_wall_seconds) * 0.05)) if max_wall_seconds is not None else 0.0
    )

    def _try_load_existing_cell(cid: str) -> dict[str, Any] | None:
        local_cell = cells_dir / f"{cid}.json"
        if (
            local_cell.is_file()
            and Path(f"{local_cell}.complete").is_file()
            and Path(f"{local_cell}.sha256").is_file()
        ):
            verify_completed_artifact(local_cell)
            return json.loads(local_cell.read_text(encoding="utf-8"))
        if input_dir is not None:
            staged_cell = Path(input_dir) / "cells" / f"{cid}.json"
            if staged_cell.is_file():
                data = json.loads(staged_cell.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "error" in data:
                    return None
                write_create_only_json(local_cell, data)
                return data
        return None

    with SignalStopFlag() as stop_flag:
        should_stop = lambda: stop_flag.stopped  # noqa: E731
        try:
            for net_arch in resolved_archs:
                for net_sz in net_sizes:
                    if should_stop():
                        raise StopRequestedError("Signal stop requested during batch6 benchmark")
                    bkey = f"{net_arch}_{net_sz}" if multi_arch_naming else net_sz
                    phase_stamp(
                        f"batch6_forward_{bkey}",
                        "START",
                        net_arch=net_arch,
                        net_size=net_sz,
                    )
                    try:
                        bench_entry = measure_batch6_forward_ms(
                            resolved_dev,
                            net_sz,
                            net_arch=net_arch,
                            num_snakes=num_snakes,
                            grid_h=grid_h,
                            grid_w=grid_w,
                            compile_model=compile_model,
                        )
                        batch6_bench[bkey] = bench_entry
                        if len(resolved_archs) == 1 and net_sz not in batch6_bench:
                            batch6_bench[net_sz] = bench_entry
                        phase_stamp(
                            f"batch6_forward_{bkey}",
                            "END",
                            forward_ms=bench_entry["forward_ms"],
                            params=bench_entry["parameter_count"],
                            forward_flops_per_row=bench_entry["forward_flops_per_row"],
                        )
                    except (StopRequestedError, KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:
                        _cleanup_device_memory(resolved_dev)
                        err_payload = _format_exception_payload(exc)
                        err_entry = {
                            "benchmark_key": bkey,
                            "net_arch": net_arch,
                            "net_size": net_sz,
                            **err_payload,
                        }
                        batch6_bench[bkey] = err_entry
                        if len(resolved_archs) == 1 and net_sz not in batch6_bench:
                            batch6_bench[net_sz] = err_entry
                        failed_benchmarks.append(err_entry)
                        phase_stamp(
                            f"batch6_forward_{bkey}",
                            "ERROR",
                            error=err_payload["error"],
                        )

            ordered_base = sorted(
                [
                    (int(e), net_arch, net_sz)
                    for e in envs_list
                    for net_arch in resolved_archs
                    for net_sz in net_sizes
                ],
                key=_cell_cost_key,
            )
            variants_list = [(arch, sz) for arch in resolved_archs for sz in net_sizes]
            total_base_by_variant: dict[tuple[NetArch, NetSize], int] = {
                v: sum(1 for _, a, nz in ordered_base if (a, nz) == v) for v in variants_list
            }
            seen_base_by_variant: dict[tuple[NetArch, NetSize], int] = {v: 0 for v in variants_list}

            # Choose the 1-PPO-epoch probe cell (prefer smallest-E shared_trunk S, else first S,
            # else cheapest cell) and schedule it right after its standard-epoch counterpart so
            # learner vs collection share is directly visible at identical E/arch/size.
            ep1_target: tuple[int, NetArch, NetSize] | None = None
            if include_one_epoch_cell and int(ppo_epochs) != 1 and ordered_base:
                st_s = [
                    item for item in ordered_base if item[1] == "shared_trunk" and item[2] == "S"
                ]
                any_s = [item for item in ordered_base if item[2] == "S"]
                ep1_target = st_s[0] if st_s else (any_s[0] if any_s else ordered_base[0])

            # Pending tuple: (num_envs, net_arch, net_sz, compile_model, cell_ppo_epochs, is_ep1)
            pending: list[tuple[int, NetArch, NetSize, bool, int, bool]] = []
            for e, arch, nz in ordered_base:
                pending.append((e, arch, nz, bool(compile_model), int(ppo_epochs), False))
                if ep1_target is not None and (e, arch, nz) == ep1_target:
                    pending.append((e, arch, nz, bool(compile_model), 1, True))

            idx = 0
            while idx < len(pending):
                num_envs, net_arch, net_sz, cell_compile, cell_epochs, is_ep1 = pending[idx]
                idx += 1
                is_extra_compile = bool(cell_compile and not compile_model)
                is_base_cell = not is_extra_compile and not is_ep1
                cell_id = _format_cell_id(
                    num_envs=num_envs,
                    net_arch=net_arch,
                    net_sz=net_sz,
                    multi_arch_naming=multi_arch_naming,
                    is_extra_compile=is_extra_compile,
                    is_one_epoch_probe=is_ep1,
                )
                bkey = f"{net_arch}_{net_sz}" if multi_arch_naming else net_sz

                if should_stop():
                    raise StopRequestedError("Signal stop requested between ruler cells")

                existing = _try_load_existing_cell(cell_id)
                if existing is not None:
                    cells.append(existing)
                    reused_cells.append(cell_id)
                    phase_stamp(cell_id, "REUSE_EXISTING")
                else:
                    elapsed_now = time.monotonic() - wall_start
                    ep_scale = (
                        (0.35 + 0.65 * (float(cell_epochs) / float(max(1, ppo_epochs))))
                        if is_ep1
                        else 1.0
                    )
                    projected_sec = _project_cell_seconds(
                        num_envs=num_envs,
                        net_sz=net_sz,
                        net_arch=net_arch,
                        is_compile=cell_compile,
                        device=resolved_dev,
                        sec_per_env_by_variant=sec_per_env_by_variant,
                        ppo_epochs_scale=ep_scale,
                        measured_compile_overhead_sec=measured_compile_overhead_sec,
                    )
                    if max_wall_seconds is not None and (
                        elapsed_now >= (float(max_wall_seconds) - export_reserve_sec)
                        or (elapsed_now + projected_sec + export_reserve_sec)
                        > float(max_wall_seconds)
                    ):
                        partial = True
                        if stopped_reason == "completed":
                            stopped_reason = "wall_projection_guard"
                        skipped_cells.append(
                            {
                                "cell_id": cell_id,
                                "num_envs": int(num_envs),
                                "net_arch": net_arch,
                                "net_size": net_sz,
                                "ppo_epochs": int(cell_epochs),
                                "is_one_epoch_probe": bool(is_ep1),
                                "compile_model": bool(cell_compile),
                                "reason": "projected_wall_budget_skip",
                                "elapsed_wall_seconds": round(elapsed_now, 4),
                                "projected_cell_seconds": round(projected_sec, 4),
                            }
                        )
                        phase_stamp(
                            cell_id,
                            "SKIP",
                            reason="projected_wall_budget_skip",
                            projected_sec=round(projected_sec, 2),
                        )
                        if is_base_cell:
                            vkey = (net_arch, net_sz)
                            seen_base_by_variant[vkey] += 1
                            if (
                                include_best_compile
                                and not compile_model
                                and seen_base_by_variant[vkey] == total_base_by_variant[vkey]
                            ):
                                best_e = _best_completed_env_for_variant(cells, net_arch, net_sz)
                                if best_e is not None:
                                    pending.insert(
                                        idx,
                                        (best_e, net_arch, net_sz, True, int(ppo_epochs), False),
                                    )
                        continue

                    cell_t0 = time.monotonic()
                    phase_stamp(
                        cell_id,
                        "START",
                        num_envs=int(num_envs),
                        net_arch=net_arch,
                        net_size=net_sz,
                        ppo_epochs=int(cell_epochs),
                        compile=bool(cell_compile),
                    )
                    eff_actor_chunk = (
                        int(actor_chunk_envs)
                        if actor_chunk_envs is not None
                        else default_actor_chunk_envs(net_sz, net_arch=net_arch)
                    )
                    eff_mb_init = (
                        int(minibatch_size)
                        if minibatch_size is not None
                        else default_minibatch_size(net_sz, net_arch=net_arch)
                    )
                    bench_meta = batch6_bench.get(bkey, {})
                    cell_record: dict[str, Any] = {
                        "cell_id": cell_id,
                        "num_envs": int(num_envs),
                        "num_snakes": int(num_snakes),
                        "grid_h": int(grid_h),
                        "grid_w": int(grid_w),
                        "net_arch": net_arch,
                        "net_size": net_sz,
                        "ppo_epochs": int(cell_epochs),
                        "is_one_epoch_probe": bool(is_ep1),
                        "compile_model": bool(cell_compile),
                        "burnin_ticks": int(burnin_ticks),
                        "parameter_count": bench_meta.get("parameter_count"),
                        "forward_flops_per_row": bench_meta.get("forward_flops_per_row"),
                        "learner_flops_per_row": bench_meta.get("learner_flops_per_row"),
                        "token_grid": (
                            list(bench_meta["token_grid"]) if "token_grid" in bench_meta else None
                        ),
                        "learner_settings": {
                            "ppo_epochs": int(cell_epochs),
                            "rollout_len": int(rollout_len),
                            "requested_minibatch_size": int(eff_mb_init),
                            "requested_actor_chunk_envs": int(eff_actor_chunk),
                        },
                        "oom": False,
                        "collection_only": None,
                        "integrated": None,
                    }

                    try:
                        if num_envs not in mature_cache:
                            phase_stamp(
                                f"burnin_E{num_envs}",
                                "START",
                                burnin_ticks=int(burnin_ticks),
                            )
                            mature_cache[num_envs] = _build_mature_sim_state(
                                device=resolved_dev,
                                num_envs=int(num_envs),
                                grid_h=grid_h,
                                grid_w=grid_w,
                                num_snakes=num_snakes,
                                max_capacity=max_capacity,
                                burnin_ticks=burnin_ticks,
                                seed=seed,
                                should_stop=should_stop,
                            )
                            phase_stamp(f"burnin_E{num_envs}", "END")

                        mature_state = mature_cache[num_envs]

                        # 1. Collection-only
                        phase_stamp(f"{cell_id}_collection_only", "START")
                        coll_chunk_used = eff_actor_chunk
                        try:
                            coll = _measure_collection_only_cell(
                                device=resolved_dev,
                                mature_sim_state=mature_state,
                                num_envs=int(num_envs),
                                grid_h=grid_h,
                                grid_w=grid_w,
                                num_snakes=num_snakes,
                                net_size=net_sz,
                                net_arch=net_arch,
                                warmup_ticks=warmup_ticks,
                                collect_ticks=collect_ticks,
                                actor_chunk_envs=eff_actor_chunk,
                                max_capacity=max_capacity,
                                compile_model=bool(cell_compile),
                                seed=seed,
                                should_stop=should_stop,
                            )
                            cell_record["collection_only"] = coll
                            coll_chunk_used = int(coll["actor_chunk_envs_used"])
                            phase_stamp(
                                f"{cell_id}_collection_only",
                                "END",
                                agent_steps_per_sec=coll["agent_steps_per_sec"],
                                peak_vram_bytes=coll["peak_vram_bytes"],
                            )
                        except (StopRequestedError, KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as exc:
                            if _is_oom_exception(exc):
                                peak_b = _peak_vram_bytes(resolved_dev)
                                _cleanup_device_memory(resolved_dev)
                                cell_record["oom"] = True
                                cell_record["collection_only"] = {
                                    "oom": True,
                                    "error": str(exc),
                                    "peak_vram_bytes": peak_b,
                                }
                                phase_stamp(f"{cell_id}_collection_only", "END", oom=True)
                            else:
                                _cleanup_device_memory(resolved_dev)
                                err_payload = _format_exception_payload(exc)
                                cell_record.update(err_payload)
                                cell_record["collection_only"] = dict(err_payload)
                                phase_stamp(
                                    f"{cell_id}_collection_only",
                                    "ERROR",
                                    error=err_payload["error"],
                                )

                        # 2. Integrated (collect + learn)
                        if cell_record["oom"]:
                            cell_record["integrated"] = {
                                "oom": True,
                                "skipped_after_collection_oom": True,
                            }
                        elif "error" in cell_record:
                            cell_record["integrated"] = {
                                "skipped_after_collection_error": True,
                            }
                        else:
                            phase_stamp(f"{cell_id}_integrated", "START")
                            try:
                                integ = _measure_integrated_cell(
                                    device=resolved_dev,
                                    mature_sim_state=mature_state,
                                    num_envs=int(num_envs),
                                    grid_h=grid_h,
                                    grid_w=grid_w,
                                    num_snakes=num_snakes,
                                    net_size=net_sz,
                                    net_arch=net_arch,
                                    warmup_ticks=warmup_ticks,
                                    rollout_len=rollout_len,
                                    ppo_epochs=int(cell_epochs),
                                    minibatch_size=eff_mb_init,
                                    actor_chunk_envs=coll_chunk_used,
                                    max_capacity=max_capacity,
                                    compile_model=bool(cell_compile),
                                    measure_larger_minibatch=bool(
                                        measure_larger_minibatch and not is_ep1
                                    ),
                                    seed=seed,
                                    should_stop=should_stop,
                                )
                                cell_record["integrated"] = integ
                                cell_record["learner_settings"] = integ["learner_settings"]
                                cell_record["rollout_buffer_bytes"] = integ["rollout_buffer_bytes"]
                                cell_record["train_rollout_len"] = integ["train_rollout_len"]
                                cell_record["train_rollout_buffer_bytes"] = integ[
                                    "train_rollout_buffer_bytes"
                                ]
                                cell_record["projected_train_peak_vram_bytes"] = integ[
                                    "projected_train_peak_vram_bytes"
                                ]
                                phase_stamp(
                                    f"{cell_id}_integrated",
                                    "END",
                                    agent_steps_per_sec=integ["agent_steps_per_sec"],
                                    peak_vram_bytes=integ["peak_vram_bytes"],
                                    projected_train_peak_vram_bytes=integ[
                                        "projected_train_peak_vram_bytes"
                                    ],
                                )
                            except (StopRequestedError, KeyboardInterrupt, SystemExit):
                                raise
                            except Exception as exc:
                                if _is_oom_exception(exc):
                                    peak_b = _peak_vram_bytes(resolved_dev)
                                    _cleanup_device_memory(resolved_dev)
                                    cell_record["oom"] = True
                                    cell_record["integrated"] = {
                                        "oom": True,
                                        "error": str(exc),
                                        "peak_vram_bytes": peak_b,
                                    }
                                    phase_stamp(f"{cell_id}_integrated", "END", oom=True)
                                else:
                                    _cleanup_device_memory(resolved_dev)
                                    err_payload = _format_exception_payload(exc)
                                    cell_record.update(err_payload)
                                    cell_record["integrated"] = dict(err_payload)
                                    phase_stamp(
                                        f"{cell_id}_integrated",
                                        "ERROR",
                                        error=err_payload["error"],
                                    )
                    except (StopRequestedError, KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:
                        _cleanup_device_memory(resolved_dev)
                        err_payload = _format_exception_payload(exc)
                        cell_record.update(err_payload)
                        phase_stamp(cell_id, "ERROR", error=err_payload["error"])

                    cell_wall = max(1e-6, time.monotonic() - cell_t0)
                    cell_record["cell_wall_seconds"] = round(cell_wall, 4)
                    vkey = (net_arch, net_sz)
                    if (
                        not cell_record["oom"]
                        and "error" not in cell_record
                        and not cell_compile
                        and not is_ep1
                    ):
                        sec_per_env_by_variant[vkey] = cell_wall / float(max(1, num_envs))
                    elif (
                        not cell_record["oom"]
                        and "error" not in cell_record
                        and cell_compile
                        and vkey in sec_per_env_by_variant
                    ):
                        base_est = sec_per_env_by_variant[vkey] * float(num_envs)
                        measured_compile_overhead_sec = max(
                            measured_compile_overhead_sec, max(0.0, cell_wall - base_est)
                        )

                    # Persist cell immediately as create-only artifact so timeouts never lose work
                    write_create_only_json(cells_dir / f"{cell_id}.json", cell_record)
                    cells.append(cell_record)
                    phase_stamp(
                        cell_id,
                        "END",
                        oom=cell_record["oom"],
                        failed=("error" in cell_record),
                    )
                    if step_hook is not None:
                        step_hook(cell_id)

                if is_base_cell:
                    vkey = (net_arch, net_sz)
                    seen_base_by_variant[vkey] += 1
                    if (
                        include_best_compile
                        and not compile_model
                        and seen_base_by_variant[vkey] == total_base_by_variant[vkey]
                    ):
                        best_e = _best_completed_env_for_variant(cells, net_arch, net_sz)
                        if best_e is not None:
                            pending.insert(
                                idx, (best_e, net_arch, net_sz, True, int(ppo_epochs), False)
                            )

        except StopRequestedError:
            partial = True
            stopped_reason = f"signal:{stop_flag.signal_name or 'INTERRUPTED'}"
            phase_stamp("ruler_interrupt", "STOP", reason=stopped_reason)
        except BaseException as exc:
            partial = True
            stopped_reason = f"exception:{type(exc).__name__}"
            raise
        finally:
            if not result_path.exists():
                succeeded_cell_ids = [str(c["cell_id"]) for c in cells if _cell_succeeded(c)]
                failed_cell_ids = [str(c["cell_id"]) for c in cells if "error" in c]
                oom_cell_ids = [str(c["cell_id"]) for c in cells if c.get("oom", False)]
                failed_cells = [
                    {
                        "cell_id": str(c["cell_id"]),
                        "num_envs": int(c["num_envs"]),
                        "net_arch": c.get("net_arch"),
                        "net_size": c.get("net_size"),
                        "error": str(c["error"]),
                        "traceback": str(c.get("traceback", "")),
                    }
                    for c in cells
                    if "error" in c
                ]
                if failed_cells or failed_benchmarks:
                    partial = True
                    if stopped_reason == "completed":
                        stopped_reason = (
                            "completed_with_cell_errors"
                            if succeeded_cell_ids
                            else "all_cells_failed"
                        )

                failed_cell_desc = [f"{fc['cell_id']} ({fc['error']})" for fc in failed_cells]
                failed_bench_desc = [
                    f"{fb['benchmark_key']} ({fb['error']})" for fb in failed_benchmarks
                ]
                if not failed_cells and not failed_benchmarks and succeeded_cell_ids:
                    status_summary = (
                        f"PASS: {len(succeeded_cell_ids)}/{len(cells)} cells succeeded; "
                        "failed cells: none."
                    )
                elif succeeded_cell_ids:
                    status_summary = (
                        f"PARTIAL_PASS: {len(succeeded_cell_ids)}/{len(cells)} cells succeeded "
                        f"({', '.join(succeeded_cell_ids)}); "
                        f"failed cells ({len(failed_cells)}): "
                        f"{', '.join(failed_cell_desc) if failed_cell_desc else 'none'}"
                    )
                    if failed_bench_desc:
                        status_summary += (
                            f"; failed benchmarks ({len(failed_benchmarks)}): "
                            f"{', '.join(failed_bench_desc)}"
                        )
                    status_summary += "."
                else:
                    status_summary = (
                        f"FAIL: 0/{len(cells)} cells succeeded; "
                        f"failed cells ({len(failed_cells)}): "
                        f"{', '.join(failed_cell_desc) if failed_cell_desc else 'none'}"
                    )
                    if failed_bench_desc:
                        status_summary += (
                            f"; failed benchmarks ({len(failed_benchmarks)}): "
                            f"{', '.join(failed_bench_desc)}"
                        )
                    status_summary += "."

                phase_stamp("ruler_export", "START", partial=partial, reason=stopped_reason)
                best_integrated_sps = max(
                    (
                        float(c["integrated"]["agent_steps_per_sec"])
                        for c in cells
                        if not c.get("is_one_epoch_probe", False) and _cell_succeeded(c)
                    ),
                    default=0.0,
                )
                best_collection_sps = max(
                    (
                        float(c["collection_only"]["agent_steps_per_sec"])
                        for c in cells
                        if "error" not in c
                        and isinstance(c.get("collection_only"), dict)
                        and not c["collection_only"].get("oom", True)
                        and "error" not in c["collection_only"]
                        and "agent_steps_per_sec" in c["collection_only"]
                    ),
                    default=0.0,
                )
                best_by_net_arch: dict[str, dict[str, float]] = {}
                for arch in resolved_archs:
                    arch_int = max(
                        (
                            float(c["integrated"]["agent_steps_per_sec"])
                            for c in cells
                            if c.get("net_arch", "per_seat") == arch
                            and not c.get("is_one_epoch_probe", False)
                            and _cell_succeeded(c)
                        ),
                        default=0.0,
                    )
                    arch_coll = max(
                        (
                            float(c["collection_only"]["agent_steps_per_sec"])
                            for c in cells
                            if c.get("net_arch", "per_seat") == arch
                            and "error" not in c
                            and isinstance(c.get("collection_only"), dict)
                            and not c["collection_only"].get("oom", True)
                            and "error" not in c["collection_only"]
                            and "agent_steps_per_sec" in c["collection_only"]
                        ),
                        default=0.0,
                    )
                    best_by_net_arch[arch] = {
                        "best_collection_agent_steps_per_sec": arch_coll,
                        "best_integrated_agent_steps_per_sec": arch_int,
                    }

                one_epoch_comparison: dict[str, Any] | None = None
                ep1_cells = [c for c in cells if c.get("is_one_epoch_probe") and _cell_succeeded(c)]
                if ep1_cells:
                    ep1_c = ep1_cells[0]
                    base_matches = [
                        c
                        for c in cells
                        if not c.get("is_one_epoch_probe")
                        and not c.get("compile_model")
                        and _cell_succeeded(c)
                        and c.get("num_envs") == ep1_c.get("num_envs")
                        and c.get("net_arch") == ep1_c.get("net_arch")
                        and c.get("net_size") == ep1_c.get("net_size")
                    ]
                    if base_matches:
                        base_c = base_matches[0]
                        one_epoch_comparison = {
                            "base_cell_id": base_c["cell_id"],
                            "one_epoch_cell_id": ep1_c["cell_id"],
                            "base_ppo_epochs": base_c["ppo_epochs"],
                            "base_integrated_agent_steps_per_sec": base_c["integrated"][
                                "agent_steps_per_sec"
                            ],
                            "one_epoch_integrated_agent_steps_per_sec": ep1_c["integrated"][
                                "agent_steps_per_sec"
                            ],
                            "base_learner_share_pct": base_c["integrated"]["stage_share_pct"].get(
                                "learner", 0.0
                            ),
                            "one_epoch_learner_share_pct": ep1_c["integrated"][
                                "stage_share_pct"
                            ].get("learner", 0.0),
                        }

                learner_defaults = {
                    f"{arch}_{sz}": {
                        "actor_chunk_envs": (
                            int(actor_chunk_envs)
                            if actor_chunk_envs is not None
                            else default_actor_chunk_envs(sz, net_arch=arch)
                        ),
                        "minibatch_size": (
                            int(minibatch_size)
                            if minibatch_size is not None
                            else default_minibatch_size(sz, net_arch=arch)
                        ),
                        "larger_minibatch_size": larger_ruler_minibatch_size(sz, net_arch=arch),
                    }
                    for arch in resolved_archs
                    for sz in net_sizes
                }

                result = {
                    "schema": "arena-ppo-ruler-result/v1",
                    "partial": bool(partial),
                    "stopped_reason": stopped_reason,
                    "status_summary": status_summary,
                    "succeeded_cell_ids": succeeded_cell_ids,
                    "failed_cell_ids": failed_cell_ids,
                    "oom_cell_ids": oom_cell_ids,
                    "failed_cells": failed_cells,
                    "failed_benchmarks": failed_benchmarks,
                    "device": str(resolved_dev),
                    "device_name": device_name,
                    "total_vram_bytes": total_vram_bytes,
                    "native_extensions": ext_info,
                    "compile_model": bool(compile_model),
                    "include_best_compile": bool(include_best_compile),
                    "include_one_epoch_cell": bool(include_one_epoch_cell),
                    "requested_envs": [int(e) for e in envs_list],
                    "requested_net_archs": list(resolved_archs),
                    "requested_net_sizes": list(net_sizes),
                    "grid_h": int(grid_h),
                    "grid_w": int(grid_w),
                    "num_snakes": int(num_snakes),
                    "burnin_ticks": int(burnin_ticks),
                    "warmup_ticks": int(warmup_ticks),
                    "collect_ticks": int(collect_ticks),
                    "rollout_len": int(rollout_len),
                    "ppo_epochs": int(ppo_epochs),
                    "minibatch_size_override": minibatch_size,
                    "actor_chunk_envs_override": actor_chunk_envs,
                    "learner_settings": {
                        "rollout_len": int(rollout_len),
                        "ppo_epochs": int(ppo_epochs),
                        "include_one_epoch_cell": bool(include_one_epoch_cell),
                        "measure_larger_minibatch": bool(measure_larger_minibatch),
                        "per_variant_defaults": learner_defaults,
                    },
                    "one_epoch_comparison": one_epoch_comparison,
                    "max_capacity": int(max_capacity),
                    "max_wall_seconds": max_wall_seconds,
                    "batch6_forward": batch6_bench,
                    "cells": cells,
                    "reused_cells": reused_cells,
                    "skipped_cells": skipped_cells,
                    "best_by_net_arch": best_by_net_arch,
                    "best_collection_agent_steps_per_sec": best_collection_sps,
                    "best_integrated_agent_steps_per_sec": best_integrated_sps,
                    "meets_50k_target": bool(best_integrated_sps >= 50_000.0),
                    "wall_seconds": round(time.monotonic() - wall_start, 4),
                }
                write_create_only_json(result_path, result)
                phase_stamp(
                    "ruler_export",
                    "END",
                    result_path=str(result_path),
                    partial=partial,
                    completed_cells=len(cells),
                    succeeded_cells=len(succeeded_cell_ids),
                    failed_cells=len(failed_cells),
                    skipped_cells=len(skipped_cells),
                )

    out_result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        not out_result.get("succeeded_cell_ids")
        and (out_result.get("failed_cells") or out_result.get("failed_benchmarks"))
        and not str(out_result.get("stopped_reason", "")).startswith("signal:")
    ):
        raise RuntimeError(str(out_result["status_summary"]))
    return out_result


def run_canary_suite(
    *,
    output_dir: Path,
    device: str | torch.device = "cuda",
    input_dir: Path | None = None,
    ledger_path: Path | None = None,
    max_wall_seconds: float | None = None,
    num_envs: int = 32,
    rollout_len: int = 16,
    seed: int = 20261110,
) -> dict[str, Any]:
    """Run instrumented family canary: indexed parity + tiny real ruler + tiny real train."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved_dev = normalize_device(device)
    wall_t0 = time.monotonic()

    phase_stamp("canary_suite", "START", device=str(resolved_dev), num_envs=num_envs)
    preload_native_sim_extensions(resolved_dev)

    # 1. Indexed vs reference 4-world x 100-tick parity spot check (MINOR 8)
    phase_stamp("canary_indexed_parity", "START")
    parity_res = verify_indexed_vs_reference_parity(
        resolved_dev,
        num_envs=4,
        ticks=100,
        burnin_ticks=40,
        seed=seed,
    )
    write_create_only_json(out / "indexed_parity.json", parity_res)
    phase_stamp("canary_indexed_parity", "END", deaths=parity_res["total_deaths"])

    # 2. Verify any staged continuation inputs (checkpoint or partial ruler cells) up front.
    # For a staged checkpoint we instantiate its config and verify `load_checkpoint` (0 updates),
    # keeping the tiny 2-update canary train below independent of production world count/size.
    staged_verification: dict[str, Any] | None = None
    if ledger_path is not None and ledger_path.is_file() and input_dir is not None:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        resume_entry = ledger.get("resume_checkpoint")
        if isinstance(resume_entry, dict):
            staged_ckpt = Path(input_dir) / str(resume_entry["relpath"])
            if not staged_ckpt.is_file():
                raise FileNotFoundError(f"Missing staged resume checkpoint at {staged_ckpt}")
            actual_sha = sha256_file(staged_ckpt)
            if actual_sha != resume_entry.get("sha256"):
                raise ValueError(
                    f"Staged resume checkpoint SHA mismatch at {staged_ckpt}: "
                    f"{actual_sha!r} != {resume_entry.get('sha256')!r}"
                )
            verified_ckpt = verify_staged_checkpoint(
                staged_ckpt,
                device=resolved_dev,
                verify_markers=False,
            )
            staged_verification = {
                "kind": "checkpoint",
                "relpath": str(resume_entry["relpath"]),
                "sha256": actual_sha,
                "updates_completed": int(verified_ckpt["updates_completed"]),
            }
            _cleanup_device_memory(resolved_dev)
        else:
            cell_units = [
                u
                for u in ledger.get("verified_units", [])
                if isinstance(u, dict) and str(u.get("unit_id", "")).startswith("cell:")
            ]
            if cell_units:
                verified_cell_ids: list[str] = []
                for unit in cell_units:
                    staged_cell = Path(input_dir) / str(unit["relpath"])
                    if not staged_cell.is_file():
                        raise FileNotFoundError(f"Missing staged ruler cell at {staged_cell}")
                    actual_sha = sha256_file(staged_cell)
                    if actual_sha != unit.get("sha256"):
                        raise ValueError(
                            f"Staged ruler cell SHA mismatch at {staged_cell}: "
                            f"{actual_sha!r} != {unit.get('sha256')!r}"
                        )
                    parsed_cell = json.loads(staged_cell.read_text(encoding="utf-8"))
                    if not isinstance(parsed_cell, dict) or "cell_id" not in parsed_cell:
                        raise ValueError(f"Invalid staged ruler cell JSON at {staged_cell}")
                    verified_cell_ids.append(str(parsed_cell["cell_id"]))
                staged_verification = {
                    "kind": "ruler_cells",
                    "verified_cells": verified_cell_ids,
                }

    # 3. Tiny REAL ruler probing ONE cell of EACH arch (ego_only, shared_trunk, per_seat;
    # smallest E, a few ticks, single learner minibatch) so any CUDA-only bug surfaces in canary.
    canary_ruler_envs = max(1, min(int(num_envs), 4))
    canary_ruler_rollout = max(2, min(int(rollout_len), 4))
    canary_ruler_rows = canary_ruler_rollout * canary_ruler_envs * DEFAULT_NUM_SNAKES
    ruler_budget = (
        max(30.0, (float(max_wall_seconds) - (time.monotonic() - wall_t0)) * 0.55)
        if max_wall_seconds is not None
        else None
    )
    ruler_res = run_ruler(
        output_dir=out,
        device=resolved_dev,
        input_dir=None,
        envs_list=[canary_ruler_envs],
        net_sizes=["S"],
        net_archs=list(DEFAULT_RULER_ARCHS),
        burnin_ticks=12,
        warmup_ticks=1,
        collect_ticks=4,
        rollout_len=canary_ruler_rollout,
        ppo_epochs=1,
        include_one_epoch_cell=False,
        minibatch_size=canary_ruler_rows,
        actor_chunk_envs=canary_ruler_envs,
        max_capacity=256,
        compile_model=False,
        include_best_compile=False,
        measure_larger_minibatch=False,
        max_wall_seconds=ruler_budget,
        seed=seed,
    )
    succeeded_archs = {
        str(c.get("net_arch"))
        for c in ruler_res.get("cells", [])
        if str(c.get("cell_id")) in set(ruler_res.get("succeeded_cell_ids", []))
    }
    if (
        ruler_res.get("failed_cells")
        or ruler_res.get("failed_benchmarks")
        or ruler_res.get("oom_cell_ids")
        or succeeded_archs != set(DEFAULT_RULER_ARCHS)
    ):
        raise RuntimeError(
            "Canary multi-arch ruler probe failed: "
            f"{ruler_res.get('status_summary', ruler_res.get('failed_cell_ids'))}"
        )

    # 4. Tiny REAL 2-update run_training_job (own tiny config; exercises periodic + final exports)
    train_cfg = ArenaPPOConfig(
        net_arch="shared_trunk",
        net_size="S",
        num_envs=int(num_envs),
        rollout_len=int(rollout_len),
        ppo_epochs=1,
        minibatch_size=min(192, int(num_envs) * DEFAULT_NUM_SNAKES * 2),
        actor_chunk_envs=int(num_envs),
        max_capacity=256,
        total_updates=2,
        checkpoint_every_updates=1,
        output_publish_interval_sec=1.0e-6,
        seed=seed,
    )

    remaining_budget = (
        max(30.0, float(max_wall_seconds) - (time.monotonic() - wall_t0))
        if max_wall_seconds is not None
        else None
    )
    train_res = run_training_job(
        cfg=train_cfg,
        output_dir=out,
        device=resolved_dev,
        input_dir=None,
        ledger_path=None,
        max_wall_seconds=remaining_budget,
    )

    # Verify real final.pt loads cleanly (including CUDA RNG + int16 live bodies + optimizer)
    final_ckpt = out / "checkpoints" / "final.pt"
    verify_completed_artifact(final_ckpt)
    reload_trainer = ArenaPPOTrainer(train_cfg, device=resolved_dev)
    reload_trainer.load_checkpoint(final_ckpt)
    del reload_trainer
    _cleanup_device_memory(resolved_dev)

    # Ensure any staged inputs that need roundtrip verification exist and all exports pass
    for rel in CANARY_REQUIRED_EXPORTS:
        verify_completed_artifact(out / rel)

    phase_stamp("canary_suite", "END", wall_seconds=round(time.monotonic() - wall_t0, 4))
    return {
        "indexed_parity": parity_res,
        "staged_input_verification": staged_verification,
        "ruler_result": ruler_res,
        "train_result": train_res,
    }


def _parse_envs_csv(raw: str) -> list[int]:
    vals = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"Invalid --envs list: {raw!r}")
    return vals


def _parse_nets_csv(raw: str) -> list[NetSize]:
    vals: list[NetSize] = []
    for tok in raw.split(","):
        t = tok.strip().upper()
        if not t:
            continue
        if t not in ("S", "M"):
            raise ValueError(f"Invalid net size {t!r}; expected S or M")
        vals.append("S" if t == "S" else "M")
    if not vals:
        raise ValueError("Empty --nets list")
    return vals


def _parse_archs_csv(raw: str) -> list[NetArch]:
    vals: list[NetArch] = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in VALID_NET_ARCHS:
            raise ValueError(f"Invalid --net-arch entry {t!r}; expected one of {VALID_NET_ARCHS}")
        arch: NetArch = (
            "shared_trunk"
            if t == "shared_trunk"
            else ("ego_only" if t == "ego_only" else "per_seat")
        )
        if arch not in vals:
            vals.append(arch)
    if not vals:
        raise ValueError("Empty --net-arch list")
    return vals


def main() -> int:
    """CLI entrypoint for `python -m research.arena_ppo.ruler`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=os.environ.get("SNAKE_DQN_DEVICE", "cuda"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--canary-suite", action="store_true")
    parser.add_argument("--canary-envs", type=int, default=32)
    parser.add_argument("--canary-rollout-len", type=int, default=16)
    parser.add_argument(
        "--net-arch",
        type=str,
        default=",".join(DEFAULT_RULER_ARCHS),
        help="Comma-separated architecture list (default: ego_only,shared_trunk,per_seat)",
    )
    parser.add_argument("--envs", type=str, default="1024,2048,4096,8192")
    parser.add_argument("--nets", type=str, default="S,M")
    parser.add_argument("--grid-h", type=int, default=DEFAULT_GRID_H)
    parser.add_argument("--grid-w", type=int, default=DEFAULT_GRID_W)
    parser.add_argument("--num-snakes", type=int, default=DEFAULT_NUM_SNAKES)
    parser.add_argument("--burnin-ticks", type=int, default=DEFAULT_BURNIN_TICKS)
    parser.add_argument("--warmup-ticks", type=int, default=6)
    parser.add_argument("--collect-ticks", type=int, default=32)
    parser.add_argument("--rollout-len", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--no-one-epoch-cell", action="store_true")
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--actor-chunk-envs", type=int, default=None)
    parser.add_argument("--max-capacity", type=int, default=1600)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-best-compile", action="store_true")
    parser.add_argument("--no-larger-minibatch", action="store_true")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=(
            float(os.environ["ARENA_PPO_MAX_WALL_SECONDS"])
            if os.environ.get("ARENA_PPO_MAX_WALL_SECONDS")
            else None
        ),
    )
    args = parser.parse_args()

    if args.canary_suite:
        run_canary_suite(
            output_dir=args.output_dir.resolve(),
            device=args.device,
            input_dir=args.input_dir.resolve() if args.input_dir is not None else None,
            ledger_path=args.ledger.resolve() if args.ledger is not None else None,
            max_wall_seconds=args.max_wall_seconds,
            num_envs=args.canary_envs,
            rollout_len=args.canary_rollout_len,
            seed=args.seed,
        )
        return 0

    run_ruler(
        output_dir=args.output_dir.resolve(),
        device=args.device,
        input_dir=args.input_dir.resolve() if args.input_dir is not None else None,
        envs_list=_parse_envs_csv(args.envs),
        net_sizes=_parse_nets_csv(args.nets),
        net_archs=_parse_archs_csv(args.net_arch),
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        num_snakes=args.num_snakes,
        burnin_ticks=args.burnin_ticks,
        warmup_ticks=args.warmup_ticks,
        collect_ticks=args.collect_ticks,
        rollout_len=args.rollout_len,
        ppo_epochs=args.ppo_epochs,
        include_one_epoch_cell=not bool(args.no_one_epoch_cell),
        minibatch_size=args.minibatch_size,
        actor_chunk_envs=args.actor_chunk_envs,
        max_capacity=args.max_capacity,
        compile_model=bool(args.compile),
        include_best_compile=not bool(args.no_best_compile),
        measure_larger_minibatch=not bool(args.no_larger_minibatch),
        max_wall_seconds=args.max_wall_seconds,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
