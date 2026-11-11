"""Throughput and VRAM ruler (`ruler.py`) and family canary suite for Stage 1 Arena PPO."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch

from research.arena_ppo import EGO_CROP_SIZE, SCALAR_DIM
from research.arena_ppo.net import (
    ArenaNet,
    ArenaNetConfig,
    NetSize,
    default_actor_chunk_envs,
    default_minibatch_size,
    larger_ruler_minibatch_size,
)
from research.arena_ppo.obs import NUM_OBS_CHANNELS, build_arena_ego_obs_gpu
from research.arena_ppo.ppo import (
    ArenaPPOConfig,
    ArenaPPOTrainer,
    StopRequestedError,
    run_training_job,
)
from research.arena_ppo.runtime import (
    SignalStopFlag,
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    preload_native_sim_extensions,
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

DEFAULT_RULER_ENVS: tuple[int, ...] = (1024, 2048, 4096, 8192)
DEFAULT_RULER_NETS: tuple[NetSize, ...] = ("S", "M")
DEFAULT_BURNIN_TICKS: int = 200
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
            # Subsample up to 4096 alive snake lengths on device for accurate p95 quantile
            flat_len = sim.length.reshape(-1)
            flat_alive = alive.reshape(-1)
            step = max(1, flat_len.numel() // 4096)
            sub_len = flat_len[::step]
            sub_alive = flat_alive[::step]
            self._length_samples.append(sub_len[sub_alive].to(torch.float32))
        self.ticks_recorded += 1

    @torch.no_grad()
    def summarize(self) -> dict[str, Any]:
        alive_n = int(self.alive_count.detach().cpu().item())
        mean_len = float(self.length_sum.detach().cpu().item()) / float(max(1, alive_n))
        p95_len = mean_len
        if self._length_samples:
            cat = torch.cat([s for s in self._length_samples if s.numel() > 0], dim=0)
            if cat.numel() > 0:
                p95_len = float(torch.quantile(cat.detach().cpu(), 0.95).item())
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
    num_snakes: int = DEFAULT_NUM_SNAKES,
    grid_h: int = DEFAULT_GRID_H,
    grid_w: int = DEFAULT_GRID_W,
    ego_crop_size: int = EGO_CROP_SIZE,
    warmup_iters: int = 10,
    bench_iters: int = 40,
    compile_model: bool = False,
) -> dict[str, Any]:
    """Measure single-world (`batch = num_snakes`) policy `act` latency in milliseconds."""
    cfg = ArenaNetConfig.from_size(net_size)
    net = ArenaNet(cfg, compile_model=compile_model).to(device).eval()
    params = net.parameter_count()
    b_rows = int(num_snakes)
    arena = torch.randn((b_rows, NUM_OBS_CHANNELS, grid_h, grid_w), device=device)
    ego = torch.randn((b_rows, NUM_OBS_CHANNELS, ego_crop_size, ego_crop_size), device=device)
    scalars = torch.randn((b_rows, SCALAR_DIM), device=device)
    mask = torch.ones((b_rows, 6), dtype=torch.bool, device=device)
    mask[: min(2, b_rows), 3:] = False

    for _ in range(max(1, warmup_iters)):
        _ = net.act(arena, ego, scalars, legal_mask=mask, greedy=False)
    synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(max(1, bench_iters)):
        _ = net.act(arena, ego, scalars, legal_mask=mask, greedy=False)
    synchronize_device(device)
    forward_ms = ((time.perf_counter() - t0) * 1000.0) / float(max(1, bench_iters))
    del net, arena, ego, scalars, mask
    _cleanup_device_memory(device)
    return {
        "net_size": net_size,
        "parameter_count": int(params),
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
        ArenaNet(ArenaNetConfig.from_size(net_size), compile_model=compile_model).to(device).eval()
    )
    chunk = max(1, min(num_envs, int(actor_chunk_envs)))
    ring_chunk = min(512, max_capacity)

    def _one_tick(
        cur_sim: TrainSim,
        cur_net: ArenaNet,
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
            obs = build_arena_ego_obs_gpu(cur_sim, ring_chunk=ring_chunk)
            if timer is not None:
                timer.stop("obs_build")
                timer.start("policy_forward")
            act_out = cur_net.act(
                obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask, greedy=False
            )
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
                obs = build_arena_ego_obs_gpu(cur_sim, env_slice=sl, ring_chunk=ring_chunk)
                if timer is not None:
                    timer.stop("obs_build")
                    timer.start("policy_forward")
                act_out = cur_net.act(
                    obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask[sl], greedy=False
                )
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
                f"E{num_envs}_{net_size}_collection_oom_retry",
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
    w_arena, w_ego, w_scal, w_mask = trainer.buffer.reconstruct_world_ticks(
        w_t, w_e, ring_chunk=ppo_cfg.ring_chunk
    )
    w_act = trainer.buffer.actions[w_t, w_e].reshape(-1).to(torch.long)
    w_lp, w_ent, w_vl, w_al = trainer.model.evaluate_actions(w_arena, w_ego, w_scal, w_mask, w_act)
    w_loss = -w_lp.mean() - 0.01 * w_ent.mean() + 0.0 * w_vl.mean() + 0.0 * w_al.mean()
    trainer.optimizer.zero_grad(set_to_none=True)
    w_loss.backward()
    trainer.optimizer.step()
    del w_arena, w_ego, w_scal, w_mask, w_act, w_lp, w_ent, w_vl, w_al, w_loss

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
                stage_timer.start("obs_build")
                obs = build_arena_ego_obs_gpu(trainer.sim, ring_chunk=ppo_cfg.ring_chunk)
                stage_timer.stop("obs_build")

                stage_timer.start("policy_forward")
                act_out = trainer.model.act(
                    obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask, greedy=False
                )
                self_actions = act_out.actions.reshape(num_envs, trainer.sim.S)
                log_probs = act_out.log_probs.reshape(num_envs, trainer.sim.S)
                values = act_out.values.reshape(num_envs, trainer.sim.S)
                stage_timer.stop("policy_forward")
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
                    self_actions[sl] = act_out.actions.reshape(end - start, trainer.sim.S)
                    log_probs[sl] = act_out.log_probs.reshape(end - start, trainer.sim.S)
                    values[sl] = act_out.values.reshape(end - start, trainer.sim.S)
                    stage_timer.stop("policy_forward")

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
            stage_timer.start("obs_build")
            obs_b = build_arena_ego_obs_gpu(
                trainer.sim, env_slice=sl, ring_chunk=ppo_cfg.ring_chunk
            )
            stage_timer.stop("obs_build")
            stage_timer.start("policy_forward")
            out_b = trainer.model.forward(
                obs_b.arena, obs_b.ego, obs_b.scalars, legal_mask=bootstrap_mask[sl]
            )
            last_values[sl] = out_b.value.reshape(end - start, trainer.sim.S)
            stage_timer.stop("policy_forward")

        stage_timer.start("gae")
        trainer.buffer.compute_gae_and_aux(
            last_values=last_values,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )
        stage_timer.stop("gae")

    # Snapshot pre-learner weights on CPU only if we will also probe the 2x larger minibatch
    larger_mb_target = min(total_rows, larger_ruler_minibatch_size(net_size))
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

    del trainer
    _cleanup_device_memory(device)
    return {
        "oom": False,
        "rollout_len": int(rollout_len),
        "ppo_epochs": int(ppo_epochs),
        "actor_chunk_envs_used": int(chunk),
        "minibatch_size_used": int(eff_mb),
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
        "ppo_stats": ppo_stats,
        "epoch0_approx_kl": ppo_stats["epoch0_approx_kl"],
        "learner_default_minibatch": {
            "minibatch_size": int(eff_mb),
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
                f"E{num_envs}_{net_size}_integrated_oom_retry",
                "RETRY",
                oom_retries=oom_retries,
                actor_chunk_envs=cur_chunk,
                minibatch_size=cur_mb,
            )


def _cell_cost_key(item: tuple[int, NetSize]) -> tuple[int, int, int]:
    num_envs, net_sz = item
    mult = 1 if net_sz == "S" else 3
    return (int(num_envs) * mult, 0 if net_sz == "S" else 1, int(num_envs))


def _best_completed_env_for_net(cells: list[dict[str, Any]], net_sz: NetSize) -> int | None:
    best_e: int | None = None
    best_sps = -1.0
    for c in cells:
        if c.get("net_size") != net_sz or bool(c.get("compile_model")):
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


def _project_cell_seconds(
    *,
    num_envs: int,
    net_sz: NetSize,
    is_compile: bool,
    device: torch.device,
    sec_per_env_by_net: dict[str, float],
) -> float:
    if net_sz in sec_per_env_by_net:
        base = sec_per_env_by_net[net_sz] * float(num_envs)
    elif net_sz == "M" and "S" in sec_per_env_by_net:
        base = 3.0 * sec_per_env_by_net["S"] * float(num_envs)
    elif net_sz == "S" and "M" in sec_per_env_by_net:
        base = (sec_per_env_by_net["M"] / 3.0) * float(num_envs)
    else:
        base = 0.0
    compile_extra = (45.0 if device.type == "cuda" else 5.0) if is_compile else 0.0
    return base * 1.15 + compile_extra


def run_ruler(
    *,
    output_dir: Path,
    device: str | torch.device = "cuda",
    input_dir: Path | None = None,
    envs_list: list[int] | tuple[int, ...] = DEFAULT_RULER_ENVS,
    net_sizes: list[NetSize] | tuple[NetSize, ...] = DEFAULT_RULER_NETS,
    grid_h: int = DEFAULT_GRID_H,
    grid_w: int = DEFAULT_GRID_W,
    num_snakes: int = DEFAULT_NUM_SNAKES,
    burnin_ticks: int = DEFAULT_BURNIN_TICKS,
    warmup_ticks: int = 6,
    collect_ticks: int = 32,
    rollout_len: int = 64,
    ppo_epochs: int = 2,
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

    wall_start = time.monotonic()
    resolved_dev = normalize_device(device)
    phase_stamp("ruler_setup", "START", device=str(resolved_dev))
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
    cells: list[dict[str, Any]] = []
    skipped_cells: list[dict[str, Any]] = []
    reused_cells: list[str] = []
    mature_cache: dict[int, dict[str, Any]] = {}
    sec_per_env_by_net: dict[str, float] = {}
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
                write_create_only_json(local_cell, data)
                return data
        return None

    with SignalStopFlag() as stop_flag:
        should_stop = lambda: stop_flag.stopped  # noqa: E731
        try:
            for net_sz in net_sizes:
                if should_stop():
                    raise StopRequestedError("Signal stop requested during batch6 benchmark")
                phase_stamp(f"batch6_forward_{net_sz}", "START", net_size=net_sz)
                batch6_bench[net_sz] = measure_batch6_forward_ms(
                    resolved_dev,
                    net_sz,
                    num_snakes=num_snakes,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    compile_model=compile_model,
                )
                phase_stamp(
                    f"batch6_forward_{net_sz}",
                    "END",
                    forward_ms=batch6_bench[net_sz]["forward_ms"],
                    params=batch6_bench[net_sz]["parameter_count"],
                )

            ordered_base = sorted(
                [(int(e), net_sz) for e in envs_list for net_sz in net_sizes],
                key=_cell_cost_key,
            )
            total_base_by_net: dict[str, int] = {
                n: sum(1 for _, nz in ordered_base if nz == n) for n in net_sizes
            }
            seen_base_by_net: dict[str, int] = {n: 0 for n in net_sizes}

            # Build dynamic schedule: cheapest-first base cells, inserting E{best}_S_compile
            # immediately after all S base cells complete (before expensive M 4096/8192 cells),
            # and E{best}_M_compile immediately after all M base cells complete.
            pending: list[tuple[int, NetSize, bool]] = [
                (e, nz, bool(compile_model)) for e, nz in ordered_base
            ]
            idx = 0
            while idx < len(pending):
                num_envs, net_sz, cell_compile = pending[idx]
                idx += 1
                is_extra_compile = bool(cell_compile and not compile_model)
                cell_id = (
                    f"E{num_envs}_{net_sz}_compile" if is_extra_compile else f"E{num_envs}_{net_sz}"
                )

                if should_stop():
                    raise StopRequestedError("Signal stop requested between ruler cells")

                existing = _try_load_existing_cell(cell_id)
                if existing is not None:
                    cells.append(existing)
                    reused_cells.append(cell_id)
                    phase_stamp(cell_id, "REUSE_EXISTING")
                else:
                    elapsed_now = time.monotonic() - wall_start
                    projected_sec = _project_cell_seconds(
                        num_envs=num_envs,
                        net_sz=net_sz,
                        is_compile=cell_compile,
                        device=resolved_dev,
                        sec_per_env_by_net=sec_per_env_by_net,
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
                                "net_size": net_sz,
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
                        if not is_extra_compile:
                            seen_base_by_net[net_sz] += 1
                            if (
                                include_best_compile
                                and not compile_model
                                and seen_base_by_net[net_sz] == total_base_by_net[net_sz]
                            ):
                                best_e = _best_completed_env_for_net(cells, net_sz)
                                if best_e is not None:
                                    pending.insert(idx, (best_e, net_sz, True))
                        continue

                    cell_t0 = time.monotonic()
                    phase_stamp(
                        cell_id,
                        "START",
                        num_envs=int(num_envs),
                        net_size=net_sz,
                        compile=bool(cell_compile),
                    )
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
                    eff_actor_chunk = (
                        int(actor_chunk_envs)
                        if actor_chunk_envs is not None
                        else default_actor_chunk_envs(net_sz)
                    )
                    eff_mb_init = (
                        int(minibatch_size)
                        if minibatch_size is not None
                        else default_minibatch_size(net_sz)
                    )
                    cell_record: dict[str, Any] = {
                        "cell_id": cell_id,
                        "num_envs": int(num_envs),
                        "num_snakes": int(num_snakes),
                        "grid_h": int(grid_h),
                        "grid_w": int(grid_w),
                        "net_size": net_sz,
                        "compile_model": bool(cell_compile),
                        "burnin_ticks": int(burnin_ticks),
                        "parameter_count": batch6_bench[net_sz]["parameter_count"],
                        "oom": False,
                        "collection_only": None,
                        "integrated": None,
                    }

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
                    except StopRequestedError:
                        raise
                    except BaseException as exc:
                        if not _is_oom_exception(exc):
                            raise
                        peak_b = _peak_vram_bytes(resolved_dev)
                        _cleanup_device_memory(resolved_dev)
                        cell_record["oom"] = True
                        cell_record["collection_only"] = {
                            "oom": True,
                            "error": str(exc),
                            "peak_vram_bytes": peak_b,
                        }
                        phase_stamp(f"{cell_id}_collection_only", "END", oom=True)

                    # 2. Integrated (collect + learn)
                    if cell_record["oom"]:
                        cell_record["integrated"] = {
                            "oom": True,
                            "skipped_after_collection_oom": True,
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
                                warmup_ticks=warmup_ticks,
                                rollout_len=rollout_len,
                                ppo_epochs=ppo_epochs,
                                minibatch_size=eff_mb_init,
                                actor_chunk_envs=coll_chunk_used,
                                max_capacity=max_capacity,
                                compile_model=bool(cell_compile),
                                measure_larger_minibatch=measure_larger_minibatch,
                                seed=seed,
                                should_stop=should_stop,
                            )
                            cell_record["integrated"] = integ
                            phase_stamp(
                                f"{cell_id}_integrated",
                                "END",
                                agent_steps_per_sec=integ["agent_steps_per_sec"],
                                peak_vram_bytes=integ["peak_vram_bytes"],
                            )
                        except StopRequestedError:
                            raise
                        except BaseException as exc:
                            if not _is_oom_exception(exc):
                                raise
                            peak_b = _peak_vram_bytes(resolved_dev)
                            _cleanup_device_memory(resolved_dev)
                            cell_record["oom"] = True
                            cell_record["integrated"] = {
                                "oom": True,
                                "error": str(exc),
                                "peak_vram_bytes": peak_b,
                            }
                            phase_stamp(f"{cell_id}_integrated", "END", oom=True)

                    cell_wall = max(1e-6, time.monotonic() - cell_t0)
                    cell_record["cell_wall_seconds"] = round(cell_wall, 4)
                    if not cell_record["oom"] and not cell_compile:
                        sec_per_env_by_net[net_sz] = cell_wall / float(max(1, num_envs))

                    # Persist cell immediately as create-only artifact so timeouts never lose work
                    write_create_only_json(cells_dir / f"{cell_id}.json", cell_record)
                    cells.append(cell_record)
                    phase_stamp(cell_id, "END", oom=cell_record["oom"])
                    if step_hook is not None:
                        step_hook(cell_id)

                if not is_extra_compile:
                    seen_base_by_net[net_sz] += 1
                    if (
                        include_best_compile
                        and not compile_model
                        and seen_base_by_net[net_sz] == total_base_by_net[net_sz]
                    ):
                        best_e = _best_completed_env_for_net(cells, net_sz)
                        if best_e is not None:
                            pending.insert(idx, (best_e, net_sz, True))

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
                phase_stamp("ruler_export", "START", partial=partial, reason=stopped_reason)
                best_integrated_sps = max(
                    (
                        float(c["integrated"]["agent_steps_per_sec"])
                        for c in cells
                        if isinstance(c.get("integrated"), dict)
                        and not c["integrated"].get("oom", True)
                        and "agent_steps_per_sec" in c["integrated"]
                    ),
                    default=0.0,
                )
                best_collection_sps = max(
                    (
                        float(c["collection_only"]["agent_steps_per_sec"])
                        for c in cells
                        if isinstance(c.get("collection_only"), dict)
                        and not c["collection_only"].get("oom", True)
                        and "agent_steps_per_sec" in c["collection_only"]
                    ),
                    default=0.0,
                )
                result = {
                    "schema": "arena-ppo-ruler-result/v1",
                    "partial": bool(partial),
                    "stopped_reason": stopped_reason,
                    "device": str(resolved_dev),
                    "device_name": device_name,
                    "total_vram_bytes": total_vram_bytes,
                    "native_extensions": ext_info,
                    "compile_model": bool(compile_model),
                    "include_best_compile": bool(include_best_compile),
                    "requested_envs": [int(e) for e in envs_list],
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
                    "max_capacity": int(max_capacity),
                    "max_wall_seconds": max_wall_seconds,
                    "batch6_forward": batch6_bench,
                    "cells": cells,
                    "reused_cells": reused_cells,
                    "skipped_cells": skipped_cells,
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
                    skipped_cells=len(skipped_cells),
                )

    return json.loads(result_path.read_text(encoding="utf-8"))


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

    # 2. Tiny REAL ruler cell (writes real cells/E<num_envs>_S.json + ruler_result.json)
    ruler_budget = (
        max(30.0, (float(max_wall_seconds) - (time.monotonic() - wall_t0)) * 0.45)
        if max_wall_seconds is not None
        else None
    )
    ruler_res = run_ruler(
        output_dir=out,
        device=resolved_dev,
        input_dir=input_dir,
        envs_list=[int(num_envs)],
        net_sizes=["S"],
        burnin_ticks=20,
        warmup_ticks=2,
        collect_ticks=8,
        rollout_len=int(rollout_len),
        ppo_epochs=1,
        minibatch_size=min(192, int(num_envs) * DEFAULT_NUM_SNAKES * 2),
        actor_chunk_envs=int(num_envs),
        max_capacity=256,
        compile_model=False,
        include_best_compile=False,
        measure_larger_minibatch=False,
        max_wall_seconds=ruler_budget,
        seed=seed,
    )

    # 3. Tiny REAL run_training_job (real periodic + final.pt + arena_ppo_weights.pth + reload)
    train_cfg = ArenaPPOConfig(
        net_size="S",
        num_envs=int(num_envs),
        rollout_len=int(rollout_len),
        ppo_epochs=1,
        minibatch_size=min(192, int(num_envs) * DEFAULT_NUM_SNAKES * 2),
        actor_chunk_envs=int(num_envs),
        max_capacity=256,
        total_updates=2,
        checkpoint_every_updates=1,
        seed=seed,
    )
    if ledger_path is not None and ledger_path.is_file() and input_dir is not None:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        resume_entry = ledger.get("resume_checkpoint")
        if isinstance(resume_entry, dict):
            staged_ckpt = Path(input_dir) / str(resume_entry["relpath"])
            if staged_ckpt.is_file():
                ckpt_obj = torch.load(staged_ckpt, map_location="cpu", weights_only=False)
                prior_u = int((ckpt_obj.get("counters") or {}).get("updates_completed", 0))
                train_cfg = ArenaPPOConfig(
                    **{
                        **ckpt_obj["config"],
                        "total_updates": prior_u + 1,
                        "checkpoint_every_updates": 1,
                    }
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
        input_dir=input_dir,
        ledger_path=ledger_path,
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
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        num_snakes=args.num_snakes,
        burnin_ticks=args.burnin_ticks,
        warmup_ticks=args.warmup_ticks,
        collect_ticks=args.collect_ticks,
        rollout_len=args.rollout_len,
        ppo_epochs=args.ppo_epochs,
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
