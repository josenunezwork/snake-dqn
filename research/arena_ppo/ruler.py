"""Throughput and VRAM ruler (`ruler.py`) for Stage 1 GPU-resident Arena PPO."""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Any

import torch

from research.arena_ppo import ARENA_GRID_H, ARENA_GRID_W, EGO_CROP_SIZE, SCALAR_DIM
from research.arena_ppo.net import ArenaNet, ArenaNetConfig, NetSize
from research.arena_ppo.obs import NUM_OBS_CHANNELS, build_arena_ego_obs_gpu
from research.arena_ppo.ppo import ArenaPPOConfig, ArenaPPOTrainer
from research.arena_ppo.runtime import (
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    preload_native_sim_extensions,
    synchronize_device,
    write_create_only_artifact,
    write_create_only_json,
)
from research.arena_ppo.sim import TrainSim

DEFAULT_RULER_ENVS: tuple[int, ...] = (1024, 2048, 4096, 8192)
DEFAULT_RULER_NETS: tuple[NetSize, ...] = ("S", "M")
TRAIN_REQUIRED_EXPORTS: tuple[str, ...] = (
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
    """Asynchronous CUDA-event timer on CUDA, or synchronized wall timer on MPS/CPU."""

    STAGES: tuple[str, ...] = ("sim_step", "obs_build", "policy_forward", "learner")

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.use_cuda_events = device.type == "cuda"
        self._cuda_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            name: [] for name in self.STAGES
        }
        self._wall_totals: dict[str, float] = {name: 0.0 for name in self.STAGES}
        self._active_stage: str | None = None
        self._active_start_ev: torch.cuda.Event | None = None
        self._active_start_wall: float = 0.0

    def start(self, stage: str) -> None:
        """Begin timing `stage`."""
        if self._active_stage is not None:
            raise RuntimeError(
                f"Nested StageSplitTimer.start({stage!r}) inside {self._active_stage!r}"
            )
        self._active_stage = stage
        if self.use_cuda_events:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._active_start_ev = ev
        else:
            synchronize_device(self.device)
            self._active_start_wall = time.perf_counter()

    def stop(self, stage: str) -> None:
        """Finish timing `stage`."""
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
        self._active_stage = None

    def summarize(self) -> dict[str, float]:
        """Synchronize device once and return total seconds per stage."""
        synchronize_device(self.device)
        if not self.use_cuda_events:
            return {name: float(self._wall_totals[name]) for name in self.STAGES}
        totals: dict[str, float] = {}
        for name in self.STAGES:
            ms_sum = sum(s_ev.elapsed_time(e_ev) for s_ev, e_ev in self._cuda_pairs[name])
            totals[name] = float(ms_sum) / 1000.0
        return totals


@torch.no_grad()
def measure_batch6_forward_ms(
    device: torch.device,
    net_size: NetSize,
    *,
    warmup_iters: int = 10,
    bench_iters: int = 40,
    compile_model: bool = False,
) -> dict[str, Any]:
    """Measure single-world (`batch = 6`) policy `act` latency in milliseconds."""
    cfg = ArenaNetConfig.from_size(net_size)
    net = ArenaNet(cfg, compile_model=compile_model).to(device).eval()
    params = net.parameter_count()
    arena = torch.randn((6, NUM_OBS_CHANNELS, ARENA_GRID_H, ARENA_GRID_W), device=device)
    ego = torch.randn((6, NUM_OBS_CHANNELS, EGO_CROP_SIZE, EGO_CROP_SIZE), device=device)
    scalars = torch.randn((6, SCALAR_DIM), device=device)
    mask = torch.ones((6, 6), dtype=torch.bool, device=device)
    mask[:2, 3:] = False

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
        "batch_size": 6,
        "forward_ms": round(forward_ms, 4),
    }


@torch.no_grad()
def _measure_collection_only_cell(
    *,
    device: torch.device,
    num_envs: int,
    net_size: NetSize,
    warmup_ticks: int,
    collect_ticks: int,
    actor_chunk_envs: int,
    max_capacity: int,
    compile_model: bool,
    seed: int,
) -> dict[str, Any]:
    """Measure collection-only throughput and stage split for `(num_envs, net_size)`."""
    _cleanup_device_memory(device)
    _reset_peak_vram(device)

    sim = TrainSim.create(num_envs, device=device, seed=seed, max_capacity=max_capacity)
    net = (
        ArenaNet(ArenaNetConfig.from_size(net_size), compile_model=compile_model).to(device).eval()
    )
    chunk = max(1, min(num_envs, int(actor_chunk_envs)))
    ring_chunk = min(512, max_capacity)

    def _one_tick(timer: StageSplitTimer | None) -> None:
        if timer is not None:
            timer.start("sim_step")
        sim.prepare()
        legal_mask = sim.legal_action_mask()
        if timer is not None:
            timer.stop("sim_step")

        if chunk == num_envs:
            if timer is not None:
                timer.start("obs_build")
            obs = build_arena_ego_obs_gpu(sim, ring_chunk=ring_chunk)
            if timer is not None:
                timer.stop("obs_build")
                timer.start("policy_forward")
            act_out = net.act(obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask, greedy=False)
            actions = act_out.actions.reshape(num_envs, sim.S)
            if timer is not None:
                timer.stop("policy_forward")
        else:
            actions = torch.empty((num_envs, sim.S), dtype=torch.long, device=device)
            for start in range(0, num_envs, chunk):
                end = min(num_envs, start + chunk)
                sl = slice(start, end)
                if timer is not None:
                    timer.start("obs_build")
                obs = build_arena_ego_obs_gpu(sim, env_slice=sl, ring_chunk=ring_chunk)
                if timer is not None:
                    timer.stop("obs_build")
                    timer.start("policy_forward")
                act_out = net.act(
                    obs.arena, obs.ego, obs.scalars, legal_mask=legal_mask[sl], greedy=False
                )
                actions[sl] = act_out.actions.reshape(end - start, sim.S)
                if timer is not None:
                    timer.stop("policy_forward")

        if timer is not None:
            timer.start("sim_step")
        sim.step(actions)
        if timer is not None:
            timer.stop("sim_step")

    for _ in range(max(1, warmup_ticks)):
        _one_tick(None)
    synchronize_device(device)
    _reset_peak_vram(device)

    stage_timer = StageSplitTimer(device)
    t0 = time.perf_counter()
    for _ in range(max(1, collect_ticks)):
        _one_tick(stage_timer)
    stage_sec = stage_timer.summarize()
    wall_sec = max(1e-9, time.perf_counter() - t0)
    peak_vram = _peak_vram_bytes(device)

    agent_steps = int(collect_ticks) * int(num_envs) * int(sim.S)
    world_steps = int(collect_ticks) * int(num_envs)
    stage_total = max(1e-9, sum(stage_sec.values()))
    stage_pct = {k: round(100.0 * v / stage_total, 2) for k, v in stage_sec.items()}

    del sim, net
    _cleanup_device_memory(device)
    return {
        "oom": False,
        "ticks": int(collect_ticks),
        "agent_steps": agent_steps,
        "world_steps": world_steps,
        "wall_seconds": round(wall_sec, 4),
        "agent_steps_per_sec": round(agent_steps / wall_sec, 2),
        "world_steps_per_sec": round(world_steps / wall_sec, 2),
        "stage_seconds": {k: round(v, 5) for k, v in stage_sec.items()},
        "stage_share_pct": stage_pct,
        "peak_vram_bytes": peak_vram,
    }


def _measure_integrated_cell(
    *,
    device: torch.device,
    num_envs: int,
    net_size: NetSize,
    warmup_ticks: int,
    rollout_len: int,
    ppo_epochs: int,
    minibatch_size: int,
    actor_chunk_envs: int,
    max_capacity: int,
    compile_model: bool,
    seed: int,
) -> dict[str, Any]:
    """Measure integrated (`collect + learn`) throughput and stage split for `(num_envs, net_size)`."""
    _cleanup_device_memory(device)
    _reset_peak_vram(device)

    eff_mb = max(6, min(rollout_len * num_envs * 6, int(minibatch_size)))
    ppo_cfg = ArenaPPOConfig(
        net_size=net_size,
        num_envs=num_envs,
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

    # Untimed warmup of forward + 1 minibatch backward so cuDNN/Inductor kernels are warm
    trainer.model.eval()
    with torch.no_grad():
        for w_tick in range(min(max(1, warmup_ticks), rollout_len)):
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
    w_t = torch.zeros((min(num_envs, max(1, eff_mb // 6)),), dtype=torch.long, device=device)
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

    synchronize_device(device)
    _reset_peak_vram(device)

    # Timed integrated rollout + PPO learner update with per-stage split
    stage_timer = StageSplitTimer(device)
    chunk = max(1, min(num_envs, int(ppo_cfg.actor_chunk_envs)))
    trainer.model.eval()
    t0 = time.perf_counter()

    with torch.no_grad():
        for t in range(rollout_len):
            stage_timer.start("sim_step")
            trainer.sim.prepare()
            trainer.buffer.record_pre_step(t, trainer.sim)
            legal_mask = trainer.sim.legal_action_mask()
            stage_timer.stop("sim_step")

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
            trainer.buffer.actions[t] = self_actions.to(torch.uint8)
            trainer.buffer.log_probs[t] = log_probs
            trainer.buffer.values[t] = values
            trainer.buffer.rewards[t] = info.reward
            trainer.buffer.died[t] = info.died
            trainer.buffer.acted[t] = info.acted
            trainer.buffer.learner_mask[t] = info.learner_mask
            trainer._maybe_apply_world_time_limits(t, info.survived)
            stage_timer.stop("sim_step")

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

        stage_timer.start("sim_step")
        trainer.buffer.compute_gae_and_aux(
            last_values=last_values,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )
        stage_timer.stop("sim_step")

    stage_timer.start("learner")
    ppo_stats, _ = trainer.ppo_update()
    stage_timer.stop("learner")

    stage_sec = stage_timer.summarize()
    wall_sec = max(1e-9, time.perf_counter() - t0)
    peak_vram = _peak_vram_bytes(device)

    agent_steps = int(rollout_len) * int(num_envs) * int(trainer.sim.S)
    world_steps = int(rollout_len) * int(num_envs)
    stage_total = max(1e-9, sum(stage_sec.values()))
    stage_pct = {k: round(100.0 * v / stage_total, 2) for k, v in stage_sec.items()}

    del trainer
    _cleanup_device_memory(device)
    return {
        "oom": False,
        "rollout_len": int(rollout_len),
        "ppo_epochs": int(ppo_epochs),
        "minibatch_size": int(eff_mb),
        "agent_steps": agent_steps,
        "world_steps": world_steps,
        "wall_seconds": round(wall_sec, 4),
        "agent_steps_per_sec": round(agent_steps / wall_sec, 2),
        "world_steps_per_sec": round(world_steps / wall_sec, 2),
        "stage_seconds": {k: round(v, 5) for k, v in stage_sec.items()},
        "stage_share_pct": stage_pct,
        "peak_vram_bytes": peak_vram,
        "ppo_stats": ppo_stats,
    }


def write_canary_export_probes(output_dir: Path, *, source_job: str = "ruler") -> list[str]:
    """Write tiny verified `canary_export_probe` files for all `train` required exports.

    Allows a `ruler` receipt on a given GPU type to satisfy `research.colab_cli.canary.check`
    (`expected_exports <= set(proof["exports"])`) for subsequent `train` runs on that GPU,
    while marking every probe payload with `"canary_export_probe": True` so resume logic
    refuses to treat them as real training checkpoints.
    """
    written: list[str] = []
    ckpt_probe = output_dir / "checkpoints" / "final.pt"
    if not ckpt_probe.exists():
        probe_obj = {
            "canary_export_probe": True,
            "not_evidence": True,
            "source_job": source_job,
        }
        write_create_only_artifact(ckpt_probe, lambda tmp: torch.save(probe_obj, tmp))
        written.append("checkpoints/final.pt")

    weights_probe = output_dir / "arena_ppo_weights.pth"
    if not weights_probe.exists():
        w_obj = {
            "canary_export_probe": True,
            "not_evidence": True,
            "source_job": source_job,
        }
        write_create_only_artifact(weights_probe, lambda tmp: torch.save(w_obj, tmp))
        written.append("arena_ppo_weights.pth")

    train_res_probe = output_dir / "train_result.json"
    if not train_res_probe.exists():
        write_create_only_json(
            train_res_probe,
            {
                "schema": "arena-ppo-train-result-canary-probe/v1",
                "canary_export_probe": True,
                "not_evidence": True,
                "source_job": source_job,
            },
        )
        written.append("train_result.json")
    return written


def run_ruler(
    *,
    output_dir: Path,
    device: str | torch.device = "cuda",
    envs_list: list[int] | tuple[int, ...] = DEFAULT_RULER_ENVS,
    net_sizes: list[NetSize] | tuple[NetSize, ...] = DEFAULT_RULER_NETS,
    warmup_ticks: int = 6,
    collect_ticks: int = 32,
    rollout_len: int = 64,
    ppo_epochs: int = 2,
    minibatch_size: int = 3072,
    actor_chunk_envs: int = 1024,
    max_capacity: int = 1600,
    compile_model: bool = False,
    max_wall_seconds: float | None = None,
    write_canary_probes: bool = True,
    seed: int = 20261110,
) -> dict[str, Any]:
    """Execute throughput & VRAM sweep across `envs_list x net_sizes` and publish `ruler_result.json`."""
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

    # Measure batch=6 single-world forward latency for requested net sizes
    batch6_Bench: dict[str, Any] = {}
    for net_sz in net_sizes:
        phase_stamp(f"batch6_forward_{net_sz}", "START", net_size=net_sz)
        batch6_Bench[net_sz] = measure_batch6_forward_ms(
            resolved_dev, net_sz, compile_model=compile_model
        )
        phase_stamp(
            f"batch6_forward_{net_sz}",
            "END",
            forward_ms=batch6_Bench[net_sz]["forward_ms"],
            params=batch6_Bench[net_sz]["parameter_count"],
        )

    export_reserve_sec = (
        min(30.0, max(2.0, float(max_wall_seconds) * 0.1)) if max_wall_seconds is not None else 0.0
    )

    def _wall_exceeded() -> bool:
        if max_wall_seconds is None:
            return False
        return (time.monotonic() - wall_start) >= (float(max_wall_seconds) - export_reserve_sec)

    cells: list[dict[str, Any]] = []
    skipped_cells: list[dict[str, Any]] = []

    for num_envs in envs_list:
        for net_sz in net_sizes:
            cell_id = f"E{num_envs}_{net_sz}"
            if _wall_exceeded():
                skipped_cells.append(
                    {
                        "cell_id": cell_id,
                        "num_envs": int(num_envs),
                        "net_size": net_sz,
                        "reason": "max_wall_seconds_budget_reached",
                        "elapsed_wall_seconds": round(time.monotonic() - wall_start, 4),
                    }
                )
                phase_stamp(cell_id, "SKIP", reason="max_wall_seconds_budget_reached")
                continue

            phase_stamp(cell_id, "START", num_envs=int(num_envs), net_size=net_sz)
            cell_record: dict[str, Any] = {
                "cell_id": cell_id,
                "num_envs": int(num_envs),
                "net_size": net_sz,
                "parameter_count": batch6_Bench[net_sz]["parameter_count"],
                "oom": False,
                "collection_only": None,
                "integrated": None,
            }

            # 1. Collection-only
            phase_stamp(f"{cell_id}_collection_only", "START")
            try:
                coll = _measure_collection_only_cell(
                    device=resolved_dev,
                    num_envs=int(num_envs),
                    net_size=net_sz,
                    warmup_ticks=warmup_ticks,
                    collect_ticks=collect_ticks,
                    actor_chunk_envs=actor_chunk_envs,
                    max_capacity=max_capacity,
                    compile_model=compile_model,
                    seed=seed,
                )
                cell_record["collection_only"] = coll
                phase_stamp(
                    f"{cell_id}_collection_only",
                    "END",
                    agent_steps_per_sec=coll["agent_steps_per_sec"],
                    peak_vram_bytes=coll["peak_vram_bytes"],
                )
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

            # 2. Integrated (collect + learn), skipped if collection_only already OOM'd or wall hit
            if cell_record["oom"]:
                cell_record["integrated"] = {
                    "oom": True,
                    "skipped_after_collection_oom": True,
                }
            elif _wall_exceeded():
                skipped_cells.append(
                    {
                        "cell_id": f"{cell_id}:integrated",
                        "num_envs": int(num_envs),
                        "net_size": net_sz,
                        "reason": "max_wall_seconds_budget_reached_before_integrated",
                        "elapsed_wall_seconds": round(time.monotonic() - wall_start, 4),
                    }
                )
                phase_stamp(
                    f"{cell_id}_integrated",
                    "SKIP",
                    reason="max_wall_seconds_budget_reached",
                )
            else:
                phase_stamp(f"{cell_id}_integrated", "START")
                try:
                    integ = _measure_integrated_cell(
                        device=resolved_dev,
                        num_envs=int(num_envs),
                        net_size=net_sz,
                        warmup_ticks=warmup_ticks,
                        rollout_len=rollout_len,
                        ppo_epochs=ppo_epochs,
                        minibatch_size=minibatch_size,
                        actor_chunk_envs=actor_chunk_envs,
                        max_capacity=max_capacity,
                        compile_model=compile_model,
                        seed=seed,
                    )
                    cell_record["integrated"] = integ
                    phase_stamp(
                        f"{cell_id}_integrated",
                        "END",
                        agent_steps_per_sec=integ["agent_steps_per_sec"],
                        peak_vram_bytes=integ["peak_vram_bytes"],
                    )
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

            cells.append(cell_record)
            phase_stamp(cell_id, "END", oom=cell_record["oom"])

    phase_stamp("ruler_export", "START")
    output_dir.mkdir(parents=True, exist_ok=True)
    canary_probes_written: list[str] = []
    if write_canary_probes:
        canary_probes_written = write_canary_export_probes(output_dir, source_job="ruler")

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
        "device": str(resolved_dev),
        "device_name": device_name,
        "total_vram_bytes": total_vram_bytes,
        "native_extensions": ext_info,
        "compile_model": bool(compile_model),
        "requested_envs": [int(e) for e in envs_list],
        "requested_net_sizes": list(net_sizes),
        "warmup_ticks": int(warmup_ticks),
        "collect_ticks": int(collect_ticks),
        "rollout_len": int(rollout_len),
        "ppo_epochs": int(ppo_epochs),
        "minibatch_size": int(minibatch_size),
        "max_capacity": int(max_capacity),
        "max_wall_seconds": max_wall_seconds,
        "batch6_forward": batch6_Bench,
        "cells": cells,
        "skipped_cells": skipped_cells,
        "best_collection_agent_steps_per_sec": best_collection_sps,
        "best_integrated_agent_steps_per_sec": best_integrated_sps,
        "meets_50k_target": bool(best_integrated_sps >= 50_000.0),
        "canary_probes_written": canary_probes_written,
        "wall_seconds": round(time.monotonic() - wall_start, 4),
    }
    result_path = output_dir / "ruler_result.json"
    write_create_only_json(result_path, result)
    phase_stamp(
        "ruler_export",
        "END",
        result_path=str(result_path),
        completed_cells=len(cells),
        skipped_cells=len(skipped_cells),
    )
    return result


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
    parser.add_argument("--envs", type=str, default="1024,2048,4096,8192")
    parser.add_argument("--nets", type=str, default="S,M")
    parser.add_argument("--warmup-ticks", type=int, default=6)
    parser.add_argument("--collect-ticks", type=int, default=32)
    parser.add_argument("--rollout-len", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=3072)
    parser.add_argument("--actor-chunk-envs", type=int, default=1024)
    parser.add_argument("--max-capacity", type=int, default=1600)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-canary-probes", action="store_true")
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

    run_ruler(
        output_dir=args.output_dir.resolve(),
        device=args.device,
        envs_list=_parse_envs_csv(args.envs),
        net_sizes=_parse_nets_csv(args.nets),
        warmup_ticks=args.warmup_ticks,
        collect_ticks=args.collect_ticks,
        rollout_len=args.rollout_len,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        actor_chunk_envs=args.actor_chunk_envs,
        max_capacity=args.max_capacity,
        compile_model=bool(args.compile),
        max_wall_seconds=args.max_wall_seconds,
        write_canary_probes=not bool(args.no_canary_probes),
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
