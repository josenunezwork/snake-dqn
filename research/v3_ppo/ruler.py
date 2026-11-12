"""Throughput, stage-split, and peak-memory ruler + family canary for Game V3.1 PPO."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil
import torch

from research.arena_ppo.runtime import (
    SignalStopFlag,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    synchronize_device,
    verify_completed_artifact,
)
from research.game_v3.config import GameV3Config
from research.game_v3.obs import LearnerIndex, allocate_obs_batch, build_obs
from research.game_v3.scripted import food_seeking_actions
from research.game_v3.sim import GameV3Sim
from research.v3_ppo.checkpoint_io import (
    pack_sim_state_dict,
    unpack_sim_state_dict,
    verify_full_checkpoint_set,
)
from research.v3_ppo.config import (
    GPU_DEFAULT_TRAIN_ENVS,
    V3_PPO_NAMESPACE,
    VALID_NET_SIZES,
    NetSize,
    V3PPOConfig,
    default_actor_chunk_learners,
    estimate_sim_memory_bytes,
    estimate_training_memory,
    get_net_preset,
    obs_layout_from_game_config,
    parse_env_bool,
)
from research.v3_ppo.eval import derive_world_seeds, evaluate_policy
from research.v3_ppo.exports import (
    artifact_or_sidecar_exists,
    exported_payload_bytes,
    write_create_only_artifact,
    write_create_only_json,
)
from research.v3_ppo.job import run_training_job
from research.v3_ppo.net import build_ego_multiscale_net
from research.v3_ppo.resume import (
    resolve_resume_checkpoint,
    validate_staged_ledger_resume,
    verify_staged_checkpoint,
    write_latest_checkpoint_manifest,
)
from research.v3_ppo.trainer import StopRequestedError, V3PPOTrainer

V3_PPO_RULER_SCHEMA = "v3-ppo-ruler-result/v1"
V3_PPO_CANARY_SCHEMA = "v3-ppo-canary-checks/v1"

DEFAULT_RULER_ENVS: tuple[int, ...] = (128, 256, 512)
DEFAULT_RULER_NETS: tuple[NetSize, ...] = VALID_NET_SIZES
DEFAULT_RULER_LEARNERS: tuple[int, ...] = (24, 8)
DEFAULT_RULER_ROLLOUT_LEN: int = 32
DEFAULT_RULER_WARMUP_UPDATES: int = 1
DEFAULT_RULER_TIMED_UPDATES: int = 2

RULER_REQUIRED_EXPORTS: tuple[str, ...] = ("ruler_result.json",)
TRAIN_REQUIRED_EXPORTS: tuple[str, ...] = (
    "config.json",
    "train_result.json",
    "checkpoints/latest_manifest.json",
)
CANARY_REQUIRED_EXPORTS: tuple[str, ...] = (
    "canary_checks.json",
    "ruler_result.json",
    "config.json",
    "train_result.json",
    "checkpoints/latest_manifest.json",
)


def _is_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cudaerrormemoryallocation" in msg
            or "cublas_status_alloc_failed" in msg
        )
    return False


def _cleanup_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _read_peak_memory_bytes(device: torch.device, *, fallback_bytes: int = 0) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return max(int(fallback_bytes), int(torch.mps.current_allocated_memory()))
    try:
        rss = int(psutil.Process().memory_info().rss)
        return max(int(fallback_bytes), rss)
    except Exception:
        return int(fallback_bytes)


def _build_packed_mature_sim_state(
    *,
    game_cfg: GameV3Config,
    num_envs: int,
    burnin_ticks: int,
    device: torch.device,
    seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Advance ``num_envs`` worlds for ``burnin_ticks`` and return compressed CPU state dict.

    Packing via :func:`pack_sim_state_dict` shrinks ``E=512`` host RAM from `~14 GB` raw grids to
    `< 0.5 GB`, and ``run_ruler`` retains at most one ``num_envs`` slot at a time.
    """
    seeds = derive_world_seeds(seed, num_envs, stream_id=500)
    sim = GameV3Sim(game_cfg, num_envs=num_envs, device=device, seeds=seeds)
    for _ in range(int(burnin_ticks)):
        if should_stop is not None and should_stop():
            raise StopRequestedError("Stop requested during ruler world burn-in")
        acts = food_seeking_actions(sim)
        sim.step(acts)
    synchronize_device(device)
    raw_state = sim.state_dict()
    del sim
    _cleanup_device_memory(device)
    packed_state = pack_sim_state_dict(raw_state)
    del raw_state
    gc.collect()
    return packed_state


@torch.no_grad()
def _measure_collection_only_cell(
    *,
    game_cfg: GameV3Config,
    packed_mature_sim_state: dict[str, Any],
    num_envs: int,
    learner_slots: int,
    net_size: NetSize,
    amp: bool,
    use_channels_last: bool,
    compile_model: bool,
    fast_learner_ops: bool = True,
    warmup_ticks: int,
    collect_ticks: int,
    device: torch.device,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    sim = GameV3Sim(game_cfg, num_envs=num_envs, device=device, seeds=[1] * int(num_envs))
    unpacked_state = unpack_sim_state_dict(packed_mature_sim_state)
    sim.load_state_dict(unpacked_state)
    del unpacked_state

    k_slots = min(max(1, int(learner_slots)), sim.N)
    mask = torch.zeros((sim.E, sim.N), dtype=torch.bool, device=device)
    mask[:, :k_slots] = True
    learner_idx = LearnerIndex(sim, mask)
    obs_buf = allocate_obs_batch(sim, learner_idx)

    raw_model = build_ego_multiscale_net(
        net_size,
        use_channels_last=use_channels_last,
        use_bf16_autocast=amp,
        fast_preprocess=bool(fast_learner_ops),
    ).to(device)
    raw_model.eval()
    if compile_model and hasattr(torch, "compile"):
        raw_model.compile_execution_paths()
    model = raw_model
    chunk_learners = default_actor_chunk_learners(net_size)

    def _step_once(
        sim_ref: GameV3Sim,
        obs_ref: Any,
        model_ref: Any,
        *,
        timed: bool,
    ) -> tuple[float, float, float]:
        if timed:
            synchronize_device(device)
            t0 = time.perf_counter()
        build_obs(sim_ref, learner_index=learner_idx, out=obs_ref)
        if timed:
            synchronize_device(device)
            t1 = time.perf_counter()

        m_cnt = learner_idx.count
        if m_cnt <= chunk_learners:
            l_acts = model_ref.act(
                obs_ref.raster_s1,
                obs_ref.raster_s4,
                obs_ref.raster_s16,
                obs_ref.scalars,
                action_mask=obs_ref.action_mask,
            ).actions
        else:
            l_acts = torch.empty((m_cnt,), dtype=torch.long, device=device)
            for s0 in range(0, m_cnt, chunk_learners):
                s1 = min(m_cnt, s0 + chunk_learners)
                l_acts[s0:s1] = model_ref.act(
                    obs_ref.raster_s1[s0:s1],
                    obs_ref.raster_s4[s0:s1],
                    obs_ref.raster_s16[s0:s1],
                    obs_ref.scalars[s0:s1],
                    action_mask=obs_ref.action_mask[s0:s1],
                ).actions

        if k_slots == sim_ref.N:
            w_acts = l_acts.view(sim_ref.E, sim_ref.N)
        else:
            w_acts = food_seeking_actions(sim_ref)
            w_acts[:, :k_slots] = l_acts.view(sim_ref.E, k_slots)
        if timed:
            synchronize_device(device)
            t2 = time.perf_counter()

        sim_ref.step(w_acts)
        if timed:
            synchronize_device(device)
            t3 = time.perf_counter()
            return (t3 - t2), (t1 - t0), (t2 - t1)
        return 0.0, 0.0, 0.0

    for _ in range(int(warmup_ticks)):
        if should_stop is not None and should_stop():
            raise StopRequestedError("Stop requested during collection warmup")
        _step_once(sim, obs_buf, model, timed=False)
    synchronize_device(device)

    # Pass 1 (headline throughput): per-stage timing syncs OFF (`timed=False`).
    _reset_peak_memory(device)
    wall_t0 = time.perf_counter()
    for _ in range(int(collect_ticks)):
        if should_stop is not None and should_stop():
            raise StopRequestedError("Stop requested during collection measurement")
        _step_once(sim, obs_buf, model, timed=False)
    synchronize_device(device)
    wall_sec = max(1.0e-9, time.perf_counter() - wall_t0)

    sim_est = estimate_sim_memory_bytes(game_cfg, num_envs)
    obs_row_bytes = obs_layout_from_game_config(game_cfg).obs_batch_bytes_per_row
    fallback_mem = (
        sim_est["sim_persistent_bytes"]
        + learner_idx.count * obs_row_bytes
        + raw_model.parameter_count() * 4
    )
    peak_bytes = _read_peak_memory_bytes(device, fallback_bytes=fallback_mem)

    # Pass 2 (separate synced stage-split pass): per-stage timing syncs ON (`timed=True`).
    split_ticks = min(max(1, int(collect_ticks)), 8)
    raw_sim_sec = 0.0
    raw_obs_sec = 0.0
    raw_fwd_sec = 0.0
    for _ in range(split_ticks):
        if should_stop is not None and should_stop():
            raise StopRequestedError("Stop requested during collection stage-split pass")
        ds, do, df = _step_once(sim, obs_buf, model, timed=True)
        raw_sim_sec += ds
        raw_obs_sec += do
        raw_fwd_sec += df

    total_agent_steps = int(collect_ticks) * sim.E * sim.N
    total_learner_steps = int(collect_ticks) * learner_idx.count
    raw_split_total = max(1.0e-9, raw_sim_sec + raw_obs_sec + raw_fwd_sec)
    sim_sec = wall_sec * (raw_sim_sec / raw_split_total)
    obs_sec = wall_sec * (raw_obs_sec / raw_split_total)
    fwd_sec = wall_sec * (raw_fwd_sec / raw_split_total)

    del sim, obs_buf, model, raw_model
    _cleanup_device_memory(device)

    return {
        "oom": False,
        "ticks": int(collect_ticks),
        "headline_stage_timing_syncs": False,
        "wall_seconds": round(wall_sec, 6),
        "collect_seconds": round(wall_sec, 6),
        "learn_seconds": 0.0,
        "agent_steps": total_agent_steps,
        "learner_steps": total_learner_steps,
        "agent_steps_per_sec": round(float(total_agent_steps) / wall_sec, 2),
        "learner_steps_per_sec": round(float(total_learner_steps) / wall_sec, 2),
        "synced_stage_split_seconds": {
            "ticks": int(split_ticks),
            "sim": round(raw_sim_sec, 6),
            "obs": round(raw_obs_sec, 6),
            "forward": round(raw_fwd_sec, 6),
        },
        "time_split_seconds": {
            "sim": round(sim_sec, 6),
            "obs": round(obs_sec, 6),
            "forward": round(fwd_sec, 6),
            "learner": 0.0,
        },
        "time_split_pct": {
            "sim": round(100.0 * raw_sim_sec / raw_split_total, 2),
            "obs": round(100.0 * raw_obs_sec / raw_split_total, 2),
            "forward": round(100.0 * raw_fwd_sec / raw_split_total, 2),
            "learner": 0.0,
        },
        "peak_memory_bytes": int(peak_bytes),
        "peak_memory_mib": round(peak_bytes / float(1024 * 1024), 2),
    }


def _measure_integrated_cell(
    *,
    game_cfg: GameV3Config,
    packed_mature_sim_state: dict[str, Any],
    num_envs: int,
    learner_slots: int,
    net_size: NetSize,
    rollout_len: int,
    ppo_epochs: int,
    minibatch_size: int | None,
    amp: bool,
    use_channels_last: bool,
    compile_model: bool,
    fast_learner_ops: bool = True,
    warmup_updates: int,
    timed_updates: int,
    seed: int,
    device: torch.device,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if int(timed_updates) < 1:
        raise ValueError(f"timed_updates must be >= 1, got {timed_updates}")
    if int(warmup_updates) < 0:
        raise ValueError(f"warmup_updates must be >= 0, got {warmup_updates}")

    cfg = V3PPOConfig(
        game_config=game_cfg,
        num_envs=num_envs,
        learner_slots=learner_slots,
        opponent_policy="food_seeking",
        net_size=net_size,
        amp=amp,
        use_channels_last=use_channels_last,
        compile_model=compile_model,
        fast_learner_ops=bool(fast_learner_ops),
        rollout_len=rollout_len,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        eval_every_minutes=0.0,
        eval_every_updates=0,
        seed=seed,
    )
    unpacked_state = unpack_sim_state_dict(packed_mature_sim_state)
    trainer = V3PPOTrainer(cfg, device=device, mature_sim_state=unpacked_state)
    del unpacked_state

    # Untimed warm-up updates populate `_bootstrap_obs` (1 `build_obs`/tick steady state),
    # initialize AdamW moments, and run cuDNN kernel autotuning before steady-state measurement.
    for _ in range(int(warmup_updates)):
        if should_stop is not None and should_stop():
            raise StopRequestedError("Stop requested during integrated warmup")
        trainer.train_step(should_stop=should_stop, stage_timing=False)
    synchronize_device(device)

    # Pass 1 (headline integrated throughput): per-stage timing syncs OFF (`stage_timing=False`).
    _reset_peak_memory(device)
    collect_sec = 0.0
    lrn_sec = 0.0
    total_agent_steps = 0
    total_learner_steps = 0
    last_step_out: dict[str, Any] = {}

    t0 = time.perf_counter()
    for _ in range(int(timed_updates)):
        if should_stop is not None and should_stop():
            raise StopRequestedError("Stop requested during integrated measurement")
        step_out = trainer.train_step(should_stop=should_stop, stage_timing=False)
        collect_sec += float(step_out["collect_seconds"])
        lrn_sec += float(step_out["learn_seconds"])
        total_agent_steps += int(step_out["agent_steps"])
        total_learner_steps += int(step_out["learner_steps"])
        last_step_out = step_out
    synchronize_device(device)
    wall_sec = max(1.0e-9, time.perf_counter() - t0)

    sim_est = estimate_sim_memory_bytes(game_cfg, num_envs)
    buf_bytes = trainer.buffer.nbytes_total()
    fallback_mem = (
        sim_est["sim_persistent_bytes"]
        + buf_bytes
        + trainer.M * trainer.obs_layout.obs_batch_bytes_per_row
        + trainer.raw_model.parameter_count() * 16
    )
    peak_bytes = _read_peak_memory_bytes(device, fallback_bytes=fallback_mem)

    # Pass 2 (separate synced pass): 1 synced rollout (`stage_timing=True`) for stage split only.
    if should_stop is not None and should_stop():
        raise StopRequestedError("Stop requested before integrated stage-split pass")
    synced_rollout = trainer.collect_rollout(should_stop=should_stop, stage_timing=True)
    raw_sim_sec = float(synced_rollout["sim_seconds"])
    raw_obs_sec = float(synced_rollout["obs_seconds"])
    raw_fwd_sec = float(synced_rollout["forward_seconds"])
    raw_coll_total = max(1.0e-9, raw_sim_sec + raw_obs_sec + raw_fwd_sec)

    sim_sec = collect_sec * (raw_sim_sec / raw_coll_total)
    obs_sec = collect_sec * (raw_obs_sec / raw_coll_total)
    fwd_sec = collect_sec * (raw_fwd_sec / raw_coll_total)
    split_total = max(1.0e-9, collect_sec + lrn_sec)
    param_count = trainer.raw_model.parameter_count()

    del trainer
    _cleanup_device_memory(device)

    return {
        "oom": False,
        "rollout_len": int(rollout_len),
        "ppo_epochs": int(ppo_epochs),
        "warmup_updates": int(warmup_updates),
        "timed_updates": int(timed_updates),
        "headline_stage_timing_syncs": False,
        "parameter_count": int(param_count),
        "wall_seconds": round(wall_sec, 6),
        "collect_seconds": round(collect_sec, 6),
        "learn_seconds": round(lrn_sec, 6),
        "collect_pct": round(100.0 * collect_sec / split_total, 2),
        "learn_pct": round(100.0 * lrn_sec / split_total, 2),
        "agent_steps": total_agent_steps,
        "learner_steps": total_learner_steps,
        "agent_steps_per_sec": round(float(total_agent_steps) / wall_sec, 2),
        "learner_steps_per_sec": round(float(total_learner_steps) / wall_sec, 2),
        "synced_rollout_split_seconds": {
            "sim": round(raw_sim_sec, 6),
            "obs": round(raw_obs_sec, 6),
            "forward": round(raw_fwd_sec, 6),
        },
        "time_split_seconds": {
            "sim": round(sim_sec, 6),
            "obs": round(obs_sec, 6),
            "forward": round(fwd_sec, 6),
            "learner": round(lrn_sec, 6),
        },
        "time_split_pct": {
            "sim": round(100.0 * sim_sec / split_total, 2),
            "obs": round(100.0 * obs_sec / split_total, 2),
            "forward": round(100.0 * fwd_sec / split_total, 2),
            "learner": round(100.0 * lrn_sec / split_total, 2),
        },
        "rollout_buffer_bytes": int(buf_bytes),
        "sim_persistent_bytes": int(sim_est["sim_persistent_bytes"]),
        "peak_memory_bytes": int(peak_bytes),
        "peak_memory_mib": round(peak_bytes / float(1024 * 1024), 2),
        "ppo_stats": {
            "policy_loss": last_step_out["policy_loss"],
            "value_loss": last_step_out["value_loss"],
            "entropy": last_step_out["entropy"],
            "epoch0_approx_kl": last_step_out["epoch0_approx_kl"],
            "grad_norm": last_step_out["grad_norm"],
        },
    }


def _format_cell_id(
    *,
    num_envs: int,
    learner_slots: int,
    net_size: NetSize,
    fp32_probe: bool = False,
) -> str:
    base = f"E{int(num_envs)}_L{int(learner_slots)}_{net_size}"
    return f"{base}_fp32" if fp32_probe else base


def _ruler_build_ordered_grid(
    *,
    envs_list: Sequence[int],
    net_sizes: Sequence[NetSize],
    learners_list: Sequence[int],
    num_snakes: int,
    amp: bool,
    include_fp32: bool,
    resolved_dev: torch.device,
) -> tuple[list[int], list[tuple[int, int, NetSize, bool, bool]]]:
    """Clamp learner counts and order cells cheapest-first so equal ``num_envs`` run together."""
    eff_learners: list[int] = []
    for k in learners_list:
        k_clamped = min(max(1, int(k)), int(num_snakes))
        if k_clamped not in eff_learners:
            eff_learners.append(k_clamped)

    ordered_nets = sorted(net_sizes, key=lambda s: get_net_preset(s).ruler_order)
    ordered_grid: list[tuple[int, int, NetSize, bool, bool]] = []
    for e in sorted(int(x) for x in envs_list):
        for k_slots in sorted(eff_learners):
            for n_sz in ordered_nets:
                ordered_grid.append((e, k_slots, n_sz, bool(amp), False))
                if include_fp32 and resolved_dev.type == "cuda" and bool(amp):
                    ordered_grid.append((e, k_slots, n_sz, False, True))
    return eff_learners, ordered_grid


def _ruler_load_staged_or_local_cell(
    cid: str,
    *,
    cells_dir: Path,
    input_dir: Path | None,
) -> dict[str, Any] | None:
    """Return an already-completed cell result from ``cells_dir`` or staged ``input_dir``."""
    local_path = cells_dir / f"{cid}.json"
    if artifact_or_sidecar_exists(local_path):
        verify_completed_artifact(local_path)
        return json.loads(local_path.read_text(encoding="utf-8"))
    if input_dir is not None:
        staged = Path(input_dir) / "cells" / f"{cid}.json"
        if staged.is_file():
            parsed = json.loads(staged.read_text(encoding="utf-8"))
            if isinstance(parsed, dict) and "error" not in parsed:
                write_create_only_json(local_path, parsed)
                return parsed
    return None


def _ruler_execute_single_cell(
    *,
    cid: str,
    num_envs: int,
    k_slots: int,
    net_sz: NetSize,
    cell_amp: bool,
    is_fp32_probe: bool,
    game_cfg: GameV3Config,
    arena_radius: float,
    num_snakes: int,
    burnin_ticks: int,
    warmup_ticks: int,
    collect_ticks: int,
    warmup_updates: int,
    timed_updates: int,
    rollout_len: int,
    ppo_epochs: int,
    minibatch_size: int | None,
    use_channels_last: bool,
    compile_model: bool,
    fast_learner_ops: bool,
    eff_vram_limit: int | None,
    seed: int,
    resolved_dev: torch.device,
    cells_dir: Path,
    cached_mature_envs: int | None,
    cached_packed_mature_state: dict[str, Any] | None,
    should_stop: Callable[[], bool],
) -> tuple[dict[str, Any], int | None, dict[str, Any] | None]:
    """Run VRAM precheck + collection/integrated measurements for one ruler cell."""
    cell_t0 = time.monotonic()
    phase_stamp(
        cid,
        "START",
        num_envs=num_envs,
        learner_slots=k_slots,
        net_size=net_sz,
        amp=cell_amp,
    )

    cell_cfg = V3PPOConfig(
        game_config=game_cfg,
        num_envs=num_envs,
        learner_slots=k_slots,
        net_size=net_sz,
        amp=cell_amp,
        use_channels_last=use_channels_last,
        compile_model=compile_model,
        fast_learner_ops=bool(fast_learner_ops),
        rollout_len=rollout_len,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        eval_every_minutes=0.0,
        eval_every_updates=0,
    )
    mem_est = estimate_training_memory(cell_cfg)
    est_peak_bytes = int(mem_est["estimated_peak_bytes"])

    cell_record: dict[str, Any] = {
        "namespace": V3_PPO_NAMESPACE,
        "cell_id": cid,
        "num_envs": int(num_envs),
        "num_snakes": int(num_snakes),
        "learner_slots": int(k_slots),
        "arena_radius": float(arena_radius),
        "net_size": net_sz,
        "amp": bool(cell_amp),
        "is_fp32_probe": bool(is_fp32_probe),
        "use_channels_last": bool(use_channels_last),
        "compile_model": bool(compile_model),
        "fast_learner_ops": bool(fast_learner_ops),
        "estimated_peak_bytes": est_peak_bytes,
        "estimated_peak_mib": round(est_peak_bytes / float(1024 * 1024), 2),
        "precheck_oom": False,
        "oom": False,
        "collection_only": None,
        "integrated": None,
    }

    if eff_vram_limit is not None and est_peak_bytes > eff_vram_limit:
        cell_record["oom"] = True
        cell_record["precheck_oom"] = True
        cell_record["oom_error"] = (
            f"Memory precheck refused cell {cid}: estimated_peak_bytes="
            f"{est_peak_bytes} > vram_limit_bytes={eff_vram_limit}"
        )
        cell_record["cell_wall_seconds"] = round(time.monotonic() - cell_t0, 4)
        write_create_only_json(cells_dir / f"{cid}.json", cell_record)
        print(
            f"V3_PPO_RULER_CELL cell={cid} status=PRECHECK_OOM "
            f"est_mib={cell_record['estimated_peak_mib']:.1f} "
            f"limit_mib={eff_vram_limit / float(1024 * 1024):.1f}",
            flush=True,
        )
        return cell_record, cached_mature_envs, cached_packed_mature_state

    try:
        if cached_mature_envs != num_envs or cached_packed_mature_state is None:
            cached_packed_mature_state = None
            cached_mature_envs = None
            gc.collect()
            cached_packed_mature_state = _build_packed_mature_sim_state(
                game_cfg=game_cfg,
                num_envs=num_envs,
                burnin_ticks=burnin_ticks,
                device=resolved_dev,
                seed=seed,
                should_stop=should_stop,
            )
            cached_mature_envs = num_envs

        coll = _measure_collection_only_cell(
            game_cfg=game_cfg,
            packed_mature_sim_state=cached_packed_mature_state,
            num_envs=num_envs,
            learner_slots=k_slots,
            net_size=net_sz,
            amp=cell_amp,
            use_channels_last=use_channels_last,
            compile_model=compile_model,
            fast_learner_ops=bool(fast_learner_ops),
            warmup_ticks=warmup_ticks,
            collect_ticks=collect_ticks,
            device=resolved_dev,
            should_stop=should_stop,
        )
        cell_record["collection_only"] = coll

        integ = _measure_integrated_cell(
            game_cfg=game_cfg,
            packed_mature_sim_state=cached_packed_mature_state,
            num_envs=num_envs,
            learner_slots=k_slots,
            net_size=net_sz,
            rollout_len=rollout_len,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            amp=cell_amp,
            use_channels_last=use_channels_last,
            compile_model=compile_model,
            fast_learner_ops=bool(fast_learner_ops),
            warmup_updates=warmup_updates,
            timed_updates=timed_updates,
            seed=seed,
            device=resolved_dev,
            should_stop=should_stop,
        )
        cell_record["integrated"] = integ
        cell_record["parameter_count"] = integ["parameter_count"]
        cell_record["collection_agent_steps_per_sec"] = coll["agent_steps_per_sec"]
        cell_record["integrated_agent_steps_per_sec"] = integ["agent_steps_per_sec"]
        cell_record["collect_seconds"] = integ["collect_seconds"]
        cell_record["learn_seconds"] = integ["learn_seconds"]
        cell_record["time_split_pct"] = integ["time_split_pct"]
        cell_record["peak_memory_bytes"] = max(
            int(coll["peak_memory_bytes"]), int(integ["peak_memory_bytes"])
        )
        cell_record["peak_memory_mib"] = round(
            cell_record["peak_memory_bytes"] / float(1024 * 1024), 2
        )
    except (StopRequestedError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _cleanup_device_memory(resolved_dev)
        if _is_oom_exception(exc):
            cell_record["oom"] = True
            cell_record["oom_error"] = str(exc)
        else:
            cell_record["error"] = f"{type(exc).__name__}: {exc}"
            cell_record["traceback"] = traceback.format_exc()

    cell_record["cell_wall_seconds"] = round(time.monotonic() - cell_t0, 4)
    write_create_only_json(cells_dir / f"{cid}.json", cell_record)

    if cell_record["oom"]:
        print(f"V3_PPO_RULER_CELL cell={cid} status=OOM", flush=True)
    elif "error" in cell_record:
        print(
            f"V3_PPO_RULER_CELL cell={cid} status=ERROR error={cell_record['error']}",
            flush=True,
        )
    else:
        sp = cell_record["time_split_pct"]
        print(
            f"V3_PPO_RULER_CELL cell={cid} status=OK "
            f"coll_sps={cell_record['collection_agent_steps_per_sec']:.1f} "
            f"int_sps={cell_record['integrated_agent_steps_per_sec']:.1f} "
            f"collect_s={cell_record['collect_seconds']:.3f} "
            f"learn_s={cell_record['learn_seconds']:.3f} "
            f"sim_pct={sp['sim']:.1f} obs_pct={sp['obs']:.1f} "
            f"fwd_pct={sp['forward']:.1f} lrn_pct={sp['learner']:.1f} "
            f"peak_mib={cell_record['peak_memory_mib']:.1f}",
            flush=True,
        )
    return cell_record, cached_mature_envs, cached_packed_mature_state


def _ruler_summarize_and_write_result(
    *,
    out: Path,
    cells: list[dict[str, Any]],
    reused_cell_ids: list[str],
    skipped_cells: list[dict[str, Any]],
    stopped_reason: str,
    resolved_dev: torch.device,
    device_name: str,
    total_vram_bytes: int | None,
    eff_vram_limit: int | None,
    envs_list: Sequence[int],
    net_sizes: Sequence[NetSize],
    eff_learners: list[int],
    arena_radius: float,
    num_snakes: int,
    rollout_len: int,
    warmup_updates: int,
    timed_updates: int,
    ppo_epochs: int,
    amp: bool,
    use_channels_last: bool,
    compile_model: bool,
    fast_learner_ops: bool,
    wall_t0: float,
) -> dict[str, Any]:
    """Summarize completed/OOM/failed cells and write create-only ``ruler_result.json``."""
    succeeded_ids = [
        str(c["cell_id"])
        for c in cells
        if not c.get("oom", False) and "error" not in c and isinstance(c.get("integrated"), dict)
    ]
    oom_ids = [str(c["cell_id"]) for c in cells if c.get("oom", False)]
    failed_ids = [str(c["cell_id"]) for c in cells if "error" in c]

    if succeeded_ids and not failed_ids and not skipped_cells and stopped_reason == "completed":
        status = "PASS"
    elif succeeded_ids and not failed_ids and (skipped_cells or stopped_reason != "completed"):
        status = "PARTIAL"
    else:
        status = "FAIL"

    best_coll_sps = max(
        (
            float(c["collection_agent_steps_per_sec"])
            for c in cells
            if c["cell_id"] in succeeded_ids
        ),
        default=0.0,
    )
    best_int_sps = max(
        (
            float(c["integrated_agent_steps_per_sec"])
            for c in cells
            if c["cell_id"] in succeeded_ids
        ),
        default=0.0,
    )

    result_payload = {
        "schema": V3_PPO_RULER_SCHEMA,
        "namespace": V3_PPO_NAMESPACE,
        "status": status,
        "stopped_reason": stopped_reason,
        "device": str(resolved_dev),
        "device_name": device_name,
        "total_vram_bytes": total_vram_bytes,
        "vram_limit_bytes": eff_vram_limit,
        "requested_envs": [int(x) for x in envs_list],
        "requested_net_sizes": list(net_sizes),
        "requested_learners": eff_learners,
        "arena_radius": float(arena_radius),
        "num_snakes": int(num_snakes),
        "rollout_len": int(rollout_len),
        "warmup_updates": int(warmup_updates),
        "timed_updates": int(timed_updates),
        "ppo_epochs": int(ppo_epochs),
        "amp": bool(amp),
        "use_channels_last": bool(use_channels_last),
        "compile_model": bool(compile_model),
        "fast_learner_ops": bool(fast_learner_ops),
        "headline_stage_timing_syncs": False,
        "succeeded_cell_ids": succeeded_ids,
        "oom_cell_ids": oom_ids,
        "failed_cell_ids": failed_ids,
        "reused_cell_ids": reused_cell_ids,
        "skipped_cells": skipped_cells,
        "best_collection_agent_steps_per_sec": best_coll_sps,
        "best_integrated_agent_steps_per_sec": best_int_sps,
        "cells": cells,
        "wall_seconds": round(time.monotonic() - wall_t0, 4),
    }
    result_path = out / "ruler_result.json"
    if not artifact_or_sidecar_exists(result_path):
        write_create_only_json(result_path, result_payload)
    return json.loads(result_path.read_text(encoding="utf-8"))


def run_ruler(
    *,
    output_dir: Path,
    device: str | torch.device = "cuda",
    input_dir: Path | None = None,
    envs_list: Sequence[int] = DEFAULT_RULER_ENVS,
    net_sizes: Sequence[NetSize] = DEFAULT_RULER_NETS,
    learners_list: Sequence[int] = DEFAULT_RULER_LEARNERS,
    arena_radius: float = 600.0,
    num_snakes: int = 24,
    burnin_ticks: int = 60,
    warmup_ticks: int = 4,
    collect_ticks: int = 24,
    warmup_updates: int = DEFAULT_RULER_WARMUP_UPDATES,
    timed_updates: int = DEFAULT_RULER_TIMED_UPDATES,
    rollout_len: int = DEFAULT_RULER_ROLLOUT_LEN,
    ppo_epochs: int = 2,
    minibatch_size: int | None = None,
    amp: bool = True,
    use_channels_last: bool = True,
    compile_model: bool = False,
    fast_learner_ops: bool = True,
    include_fp32: bool = False,
    vram_budget_bytes: int | None = None,
    max_wall_seconds: float | None = None,
    seed: int = 20261111,
) -> dict[str, Any]:
    """Sweep ``worlds x net_size x learners`` and export create-only ``ruler_result.json``.

    Pre-checks estimated VRAM before allocating each cell (`precheck_oom=True` if exceeding
    ``vram_budget_bytes`` or CUDA physical VRAM) and caches at most one compressed mature
    simulator state in host RAM across consecutive cells sharing ``num_envs``.
    """
    if int(timed_updates) < 1:
        raise ValueError(f"timed_updates must be >= 1, got {timed_updates}")
    if int(warmup_updates) < 0:
        raise ValueError(f"warmup_updates must be >= 0, got {warmup_updates}")

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cells_dir = out / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    resolved_dev = normalize_device(device)
    ensure_cuda_initialized(resolved_dev)
    device_name = (
        torch.cuda.get_device_name(resolved_dev)
        if resolved_dev.type == "cuda"
        else str(resolved_dev)
    )
    total_vram_bytes = (
        int(torch.cuda.get_device_properties(resolved_dev).total_memory)
        if resolved_dev.type == "cuda"
        else None
    )
    eff_vram_limit = int(vram_budget_bytes) if vram_budget_bytes is not None else total_vram_bytes

    game_cfg = GameV3Config(arena_radius=float(arena_radius), num_snakes=int(num_snakes))
    eff_learners, ordered_grid = _ruler_build_ordered_grid(
        envs_list=envs_list,
        net_sizes=net_sizes,
        learners_list=learners_list,
        num_snakes=int(num_snakes),
        amp=bool(amp),
        include_fp32=bool(include_fp32),
        resolved_dev=resolved_dev,
    )

    wall_t0 = time.monotonic()
    export_reserve_sec = (
        min(30.0, max(2.0, float(max_wall_seconds) * 0.05)) if max_wall_seconds is not None else 0.0
    )

    cached_mature_envs: int | None = None
    cached_packed_mature_state: dict[str, Any] | None = None

    cells: list[dict[str, Any]] = []
    reused_cell_ids: list[str] = []
    skipped_cells: list[dict[str, Any]] = []
    stopped_reason = "completed"

    with SignalStopFlag() as stop_flag:
        should_stop = lambda: bool(stop_flag.stopped)  # noqa: E731
        try:
            for num_envs, k_slots, net_sz, cell_amp, is_fp32_probe in ordered_grid:
                if should_stop():
                    raise StopRequestedError("Signal stop requested between ruler cells")
                cid = _format_cell_id(
                    num_envs=num_envs,
                    learner_slots=k_slots,
                    net_size=net_sz,
                    fp32_probe=is_fp32_probe,
                )

                staged_rec = _ruler_load_staged_or_local_cell(
                    cid, cells_dir=cells_dir, input_dir=input_dir
                )
                if staged_rec is not None:
                    cells.append(staged_rec)
                    reused_cell_ids.append(cid)
                    continue

                elapsed_now = time.monotonic() - wall_t0
                if max_wall_seconds is not None and elapsed_now >= (
                    float(max_wall_seconds) - export_reserve_sec
                ):
                    stopped_reason = "wall_budget_guard"
                    skipped_cells.append(
                        {
                            "cell_id": cid,
                            "num_envs": num_envs,
                            "learner_slots": k_slots,
                            "net_size": net_sz,
                            "reason": "elapsed_wall_budget_skip",
                        }
                    )
                    continue

                cell_record, cached_mature_envs, cached_packed_mature_state = (
                    _ruler_execute_single_cell(
                        cid=cid,
                        num_envs=num_envs,
                        k_slots=k_slots,
                        net_sz=net_sz,
                        cell_amp=cell_amp,
                        is_fp32_probe=is_fp32_probe,
                        game_cfg=game_cfg,
                        arena_radius=arena_radius,
                        num_snakes=num_snakes,
                        burnin_ticks=burnin_ticks,
                        warmup_ticks=warmup_ticks,
                        collect_ticks=collect_ticks,
                        warmup_updates=warmup_updates,
                        timed_updates=timed_updates,
                        rollout_len=rollout_len,
                        ppo_epochs=ppo_epochs,
                        minibatch_size=minibatch_size,
                        use_channels_last=use_channels_last,
                        compile_model=compile_model,
                        fast_learner_ops=fast_learner_ops,
                        eff_vram_limit=eff_vram_limit,
                        seed=seed,
                        resolved_dev=resolved_dev,
                        cells_dir=cells_dir,
                        cached_mature_envs=cached_mature_envs,
                        cached_packed_mature_state=cached_packed_mature_state,
                        should_stop=should_stop,
                    )
                )
                cells.append(cell_record)

        except StopRequestedError:
            stopped_reason = f"signal:{stop_flag.signal_name or 'STOP'}"
        finally:
            cached_packed_mature_state = None
            cached_mature_envs = None
            gc.collect()

    return _ruler_summarize_and_write_result(
        out=out,
        cells=cells,
        reused_cell_ids=reused_cell_ids,
        skipped_cells=skipped_cells,
        stopped_reason=stopped_reason,
        resolved_dev=resolved_dev,
        device_name=device_name,
        total_vram_bytes=total_vram_bytes,
        eff_vram_limit=eff_vram_limit,
        envs_list=envs_list,
        net_sizes=net_sizes,
        eff_learners=eff_learners,
        arena_radius=arena_radius,
        num_snakes=num_snakes,
        rollout_len=rollout_len,
        warmup_updates=warmup_updates,
        timed_updates=timed_updates,
        ppo_epochs=ppo_epochs,
        amp=amp,
        use_channels_last=use_channels_last,
        compile_model=compile_model,
        fast_learner_ops=fast_learner_ops,
        wall_t0=wall_t0,
    )


@torch.no_grad()
def _verify_cardinal6_canary_decisions(
    device: torch.device,
    game_cfg: GameV3Config,
    *,
    num_envs: int,
    seed: int,
) -> int:
    """Run real 6-cardinal `{L,S,R}x{normal,boost}` policy decisions on `device` for canary proof.

    `research.colab_cli.canary` validates CUDA action decisions via `Tensor.argmax` over the
    6-cardinal action axis (`shape[-1] == 6`) in `SNAKE_COLAB_CANARY_PACKAGE` (`research/v3_ppo`).
    """
    seeds = derive_world_seeds(seed, num_envs, stream_id=777)
    sim = GameV3Sim(game_cfg, num_envs=num_envs, device=device, seeds=seeds)
    learner_idx = LearnerIndex(sim, torch.ones((sim.E, sim.N), dtype=torch.bool, device=device))
    obs = build_obs(sim, learner_index=learner_idx)
    net = build_ego_multiscale_net("S").to(device)
    net.eval()
    fwd = net(
        obs.raster_s1, obs.raster_s4, obs.raster_s16, obs.scalars, action_mask=obs.action_mask
    )
    cardinal6_idx = torch.tensor([0, 2, 4, 5, 7, 9], dtype=torch.long, device=device)
    logits6 = fwd.policy_logits.view(sim.E, sim.N, -1).index_select(-1, cardinal6_idx)
    choice6 = logits6.argmax(dim=-1)
    mapped_actions = cardinal6_idx[choice6]
    sim.step(mapped_actions)
    synchronize_device(device)
    n_dec = int(choice6.numel())
    del sim, learner_idx, obs, net, fwd, logits6, choice6, mapped_actions
    _cleanup_device_memory(device)
    return n_dec


def _copy_verified_export_single_inode(src: Path, dst: Path) -> str:
    """Copy a verified scratch file into ``output/`` as a single-inode (`st_nlink == 1`) export."""
    verify_completed_artifact(src)
    return write_create_only_artifact(dst, lambda tmp, s=src: shutil.copyfile(s, tmp))


def _canary_stage_verify_staged_input(
    *,
    input_dir: Path | None,
    ledger_path: Path | None,
    resolved_dev: torch.device,
) -> dict[str, Any] | None:
    """Stage 1 of canary suite: validate optional staged continuation checkpoint."""
    staged_meta = validate_staged_ledger_resume(input_dir=input_dir, ledger_path=ledger_path)
    if staged_meta is None:
        return None
    verified_meta = verify_staged_checkpoint(
        Path(staged_meta["path"]),
        device=resolved_dev,
        verify_markers=False,
    )
    staged_verification = {
        "kind": str(verified_meta["checkpoint_kind"]),
        "relpath": str(staged_meta["relpath"]),
        "sha256": str(staged_meta["sha256"]),
        "num_shards": int(verified_meta["num_shards"]),
        "updates_completed": int(verified_meta["updates_completed"]),
    }
    _cleanup_device_memory(resolved_dev)
    return staged_verification


def _canary_stage_ruler_sweep(
    *,
    out: Path,
    resolved_dev: torch.device,
    canary_game_cfg: GameV3Config,
    canary_envs: int,
    canary_arena_radius: float,
    n_snakes: int,
    subset_learners: int,
    canary_rollout_len: int,
    compile_model: bool,
    fast_learner_ops: bool,
    max_wall_seconds: float | None,
    wall_t0: float,
    seed: int,
) -> tuple[int, dict[str, Any]]:
    """Stage 2 of canary suite: 6-cardinal decision probe + tiny multi-preset ruler sweep."""
    cardinal6_decisions = _verify_cardinal6_canary_decisions(
        resolved_dev,
        canary_game_cfg,
        num_envs=max(2, int(canary_envs)),
        seed=seed,
    )
    ruler_budget = (
        max(30.0, (float(max_wall_seconds) - (time.monotonic() - wall_t0)) * 0.45)
        if max_wall_seconds is not None
        else None
    )
    ruler_res = run_ruler(
        output_dir=out,
        device=resolved_dev,
        input_dir=None,
        envs_list=[max(2, int(canary_envs))],
        net_sizes=list(VALID_NET_SIZES),
        learners_list=[n_snakes, subset_learners],
        arena_radius=float(canary_arena_radius),
        num_snakes=n_snakes,
        burnin_ticks=8,
        warmup_ticks=1,
        collect_ticks=4,
        warmup_updates=1,
        timed_updates=2,
        rollout_len=max(2, int(canary_rollout_len)),
        ppo_epochs=1,
        minibatch_size=max(8, int(canary_envs) * subset_learners),
        amp=True,
        compile_model=bool(compile_model),
        fast_learner_ops=bool(fast_learner_ops),
        include_fp32=(resolved_dev.type == "cuda"),
        max_wall_seconds=ruler_budget,
        seed=seed,
    )
    if (
        ruler_res.get("status") != "PASS"
        or ruler_res.get("oom_cell_ids")
        or ruler_res.get("failed_cell_ids")
        or ruler_res.get("skipped_cells")
    ):
        raise RuntimeError(
            f"Canary ruler sweep did not complete cleanly: status={ruler_res.get('status')}, "
            f"oom={ruler_res.get('oom_cell_ids')}, failed={ruler_res.get('failed_cell_ids')}"
        )
    return cardinal6_decisions, ruler_res


def _canary_stage_train_and_resume(
    *,
    out: Path,
    resolved_dev: torch.device,
    canary_envs: int,
    canary_arena_radius: float,
    n_snakes: int,
    subset_learners: int,
    canary_rollout_len: int,
    max_wall_seconds: float | None,
    wall_t0: float,
    seed: int,
) -> dict[str, Any]:
    """Stage 3 of canary suite: 2-update train + sharded FULL checkpoint + staged ledger resume."""
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{out.name}_canary_", dir=str(out.parent)))
    try:
        game_cfg = GameV3Config(arena_radius=float(canary_arena_radius), num_snakes=n_snakes)
        cfg_u1 = V3PPOConfig(
            game_config=game_cfg,
            num_envs=max(2, int(canary_envs)),
            learner_slots=None,
            net_size="S",
            rollout_len=max(2, int(canary_rollout_len)),
            ppo_epochs=1,
            minibatch_size=max(8, int(canary_envs) * n_snakes),
            total_updates=1,
            full_checkpoint_every_updates=1,
            eval_every_minutes=0.0,
            eval_every_updates=1,
            eval_envs=2,
            eval_burnin_ticks=4,
            eval_ticks=12,
            eval_learner_slots=subset_learners,
            seed=seed,
        )
        u1_out = tmp_dir / "u1_out"
        u1_work = tmp_dir / "u1_work"
        rem_u1 = (
            max(20.0, float(max_wall_seconds) - (time.monotonic() - wall_t0))
            if max_wall_seconds is not None
            else None
        )
        res_u1 = run_training_job(
            cfg=cfg_u1,
            output_dir=u1_out,
            work_dir=u1_work,
            device=resolved_dev,
            max_wall_seconds=rem_u1,
        )
        if res_u1.get("status") != "PASS":
            raise RuntimeError(f"Canary update-1 train failed: {res_u1}")

        u1_full_idx = u1_out / "checkpoints" / "full_u0000001.index.json"
        full_meta = verify_full_checkpoint_set(u1_full_idx, require_markers=True)
        verify_staged_checkpoint(u1_full_idx, device=resolved_dev, verify_markers=True)

        staged_in = tmp_dir / "resume_in"
        staged_in.mkdir(parents=True, exist_ok=True)
        staged_idx_dst = staged_in / u1_full_idx.name
        u1_idx_sha = write_create_only_artifact(
            staged_idx_dst, lambda tmp, src=u1_full_idx: shutil.copyfile(src, tmp)
        )
        companion_entries: list[dict[str, Any]] = []
        for comp_src in [full_meta["light_path"], *full_meta["shard_paths"]]:
            comp_dst = staged_in / comp_src.name
            comp_sha = write_create_only_artifact(
                comp_dst, lambda tmp, src=comp_src: shutil.copyfile(src, tmp)
            )
            companion_entries.append(
                {
                    "relpath": comp_src.name,
                    "sha256": comp_sha,
                    "size_bytes": int(comp_dst.stat().st_size),
                }
            )

        staged_ledger = staged_in / "verified-ledger.json"
        write_create_only_json(
            staged_ledger,
            {
                "schema": "v3-ppo-verified-ledger/v1",
                "namespace": V3_PPO_NAMESPACE,
                "unit_kind": "checkpoint",
                "skip_verified": True,
                "resume_checkpoint": {
                    "relpath": u1_full_idx.name,
                    "checkpoint_kind": "full",
                    "sha256": u1_idx_sha,
                    "size_bytes": int(staged_idx_dst.stat().st_size),
                    "companion_files": companion_entries,
                },
                "verified_units": [],
            },
        )

        cfg_u2 = V3PPOConfig.from_dict({**cfg_u1.to_dict(), "total_updates": 2})
        u2_out = tmp_dir / "u2_out"
        u2_work = tmp_dir / "u2_work"
        rem_u2 = (
            max(20.0, float(max_wall_seconds) - (time.monotonic() - wall_t0))
            if max_wall_seconds is not None
            else None
        )
        res_u2 = run_training_job(
            cfg=cfg_u2,
            output_dir=u2_out,
            work_dir=u2_work,
            device=resolved_dev,
            input_dir=staged_in,
            ledger_path=staged_ledger,
            max_wall_seconds=rem_u2,
        )
        if (
            res_u2.get("status") != "PASS"
            or res_u2.get("resumed_from") is None
            or int(res_u2.get("updates_completed", 0)) != 2
        ):
            raise RuntimeError(f"Canary resumed update-2 train failed: {res_u2}")
        u2_resolved = resolve_resume_checkpoint(u2_out / "checkpoints", require_markers=True)
        if (
            u2_resolved is None
            or str(u2_resolved.get("kind")) != "full"
            or int(u2_resolved["updates_completed"]) != 2
        ):
            raise RuntimeError(f"Canary scratch missing verified u2 FULL checkpoint: {u2_resolved}")

        _copy_verified_export_single_inode(u2_out / "config.json", out / "config.json")

        u2_full_idx = Path(u2_resolved["path"])
        u2_full_meta = verify_full_checkpoint_set(u2_full_idx, require_markers=True)
        out_ckpts = out / "checkpoints"
        out_ckpts.mkdir(parents=True, exist_ok=True)
        for ckpt_src in [u2_full_meta["light_path"], *u2_full_meta["shard_paths"], u2_full_idx]:
            _copy_verified_export_single_inode(ckpt_src, out_ckpts / ckpt_src.name)
            ckpt_meta_src = ckpt_src.with_suffix(".meta.json")
            if Path(f"{ckpt_meta_src}.complete").is_file():
                _copy_verified_export_single_inode(ckpt_meta_src, out_ckpts / ckpt_meta_src.name)

        write_latest_checkpoint_manifest(
            out_ckpts,
            reason="completed",
            counters={
                "updates_completed": int(res_u2["updates_completed"]),
                "agent_steps_collected": int(res_u2["agent_steps_collected"]),
                "world_steps_collected": int(res_u2["world_steps_collected"]),
            },
        )

        for sub_name in ("metrics", "eval"):
            src_sub = u2_out / sub_name
            if src_sub.is_dir():
                for seg_src in sorted(src_sub.glob("*.jsonl")):
                    if Path(f"{seg_src}.complete").is_file():
                        _copy_verified_export_single_inode(seg_src, out / sub_name / seg_src.name)

        verify_completed_artifact(u2_out / "train_result.json")
        u2_result_payload = json.loads((u2_out / "train_result.json").read_text(encoding="utf-8"))
        u2_result_payload["session_export_bytes_published"] = int(exported_payload_bytes(out))
        write_create_only_json(out / "train_result.json", u2_result_payload)

        final_resolved = resolve_resume_checkpoint(out_ckpts, require_markers=True)
        if final_resolved is None or int(final_resolved["updates_completed"]) != 2:
            raise RuntimeError(f"Canary output missing verified u2 checkpoint: {final_resolved}")
        return res_u2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _canary_stage_default_eval_benchmark(
    *,
    resolved_dev: torch.device,
    canary_game_cfg: GameV3Config,
    canary_envs: int,
    canary_rollout_len: int,
    canary_train_envs: int | None,
    compile_model: bool,
    fast_learner_ops: bool,
    canary_eval_cfg_override: V3PPOConfig | None,
    max_wall_seconds: float | None,
    wall_t0: float,
    seed: int,
) -> tuple[dict[str, Any], int, float, int]:
    """Stage 4 of canary suite: default-config evaluation with resident trainer memory."""
    default_eval_cfg = (
        canary_eval_cfg_override
        if canary_eval_cfg_override is not None
        else V3PPOConfig(
            compile_model=bool(compile_model),
            fast_learner_ops=bool(fast_learner_ops),
            seed=seed,
        )
    )
    env_train_envs = (
        int(os.environ["V3_PPO_CANARY_TRAIN_ENVS"])
        if os.environ.get("V3_PPO_CANARY_TRAIN_ENVS")
        else None
    )
    explicit_train_envs = (
        int(canary_train_envs) if canary_train_envs is not None else env_train_envs
    )
    if explicit_train_envs is not None:
        if explicit_train_envs < 1:
            raise ValueError(f"canary_train_envs must be >= 1, got {explicit_train_envs}")
        eff_resident_train_envs = explicit_train_envs
        eff_resident_game_cfg = default_eval_cfg.game_config
        eff_resident_rollout_len = int(default_eval_cfg.rollout_len)
    elif resolved_dev.type == "cuda":
        vram_total = int(torch.cuda.get_device_properties(resolved_dev).total_memory)
        if vram_total >= 60_000_000_000:
            eff_resident_train_envs = int(GPU_DEFAULT_TRAIN_ENVS["H100"])
        elif vram_total >= 32_000_000_000:
            eff_resident_train_envs = int(GPU_DEFAULT_TRAIN_ENVS["A100"])
        else:
            eff_resident_train_envs = int(GPU_DEFAULT_TRAIN_ENVS["L4"])
        eff_resident_game_cfg = default_eval_cfg.game_config
        eff_resident_rollout_len = int(default_eval_cfg.rollout_len)
    else:
        eff_resident_train_envs = max(2, int(canary_envs))
        eff_resident_game_cfg = (
            canary_eval_cfg_override.game_config
            if canary_eval_cfg_override is not None
            else canary_game_cfg
        )
        eff_resident_rollout_len = max(2, int(canary_rollout_len))

    with SignalStopFlag() as eval_stop_flag:
        eval_guard_reserve = 10.0

        def _canary_eval_should_stop() -> bool:
            if bool(eval_stop_flag.stopped):
                return True
            if max_wall_seconds is not None:
                return (time.monotonic() - wall_t0) >= max(
                    5.0, float(max_wall_seconds) - eval_guard_reserve
                )
            return False

        _cleanup_device_memory(resolved_dev)
        resident_train_cfg = V3PPOConfig(
            game_config=eff_resident_game_cfg,
            num_envs=eff_resident_train_envs,
            learner_slots=default_eval_cfg.learner_slots,
            opponent_policy=default_eval_cfg.opponent_policy,
            net_size=default_eval_cfg.net_size,
            amp=bool(default_eval_cfg.amp),
            use_channels_last=bool(default_eval_cfg.use_channels_last),
            compile_model=bool(default_eval_cfg.compile_model),
            fast_learner_ops=bool(default_eval_cfg.fast_learner_ops),
            rollout_len=eff_resident_rollout_len,
            eval_every_minutes=0.0,
            eval_every_updates=0,
            seed=seed,
        )
        resident_trainer = V3PPOTrainer(resident_train_cfg, device=resolved_dev)
        try:
            synchronize_device(resolved_dev)
            _reset_peak_memory(resolved_dev)
            default_eval_res = evaluate_policy(
                resident_trainer.model,
                default_eval_cfg,
                device=resolved_dev,
                should_stop=_canary_eval_should_stop,
            )
            resident_sim_bytes = estimate_sim_memory_bytes(
                eff_resident_game_cfg, eff_resident_train_envs
            )["sim_persistent_bytes"]
            eval_sim_bytes = estimate_sim_memory_bytes(
                default_eval_cfg.game_config, int(default_eval_res["eval_envs"])
            )["sim_persistent_bytes"]
            fallback_combined = (
                int(resident_sim_bytes)
                + int(resident_trainer.buffer.nbytes_total())
                + int(eval_sim_bytes)
                + int(resident_trainer.raw_model.parameter_count()) * 16
            )
            eval_peak_bytes = max(
                int(default_eval_res.get("peak_memory_bytes", 0)),
                _read_peak_memory_bytes(resolved_dev, fallback_bytes=fallback_combined),
            )
        finally:
            del resident_trainer
            _cleanup_device_memory(resolved_dev)

    if bool(default_eval_res.get("aborted", False)):
        raise RuntimeError(
            "Canary default-config evaluation aborted due to stop/wall guard: "
            f"{default_eval_res}"
        )

    eval_peak_mib = round(int(eval_peak_bytes) / float(1024 * 1024), 2)
    default_eval_benchmark = {
        "wall_seconds": float(default_eval_res["wall_seconds"]),
        "peak_memory_bytes": int(eval_peak_bytes),
        "peak_memory_mib": eval_peak_mib,
        "resident_train_envs": int(eff_resident_train_envs),
        "resident_rollout_len": int(eff_resident_rollout_len),
        "arena_radius": float(default_eval_cfg.game_config.arena_radius),
        "num_snakes": int(default_eval_cfg.num_snakes),
        "net_size": default_eval_cfg.net_size,
        "eval_envs": int(default_eval_res["eval_envs"]),
        "matchup_envs": default_eval_res["matchup_envs"],
        "eval_burnin_ticks": int(default_eval_res["eval_burnin_ticks"]),
        "eval_ticks": int(default_eval_res["eval_ticks"]),
        "eval_learner_slots": int(default_eval_res["eval_learner_slots"]),
        "vs_food_seeking_completed_lives": int(
            default_eval_res["matchups"]["vs_food_seeking"]["completed_lives"]
        ),
        "vs_random_safe_completed_lives": int(
            default_eval_res["matchups"]["vs_random_safe"]["completed_lives"]
        ),
        "aborted": False,
    }
    print(
        f"V3_PPO_CANARY_DEFAULT_EVAL wall_s={default_eval_benchmark['wall_seconds']:.3f} "
        f"peak_mib={eval_peak_mib:.1f} "
        f"resident_train_envs={eff_resident_train_envs} "
        f"eval_envs={default_eval_benchmark['eval_envs']} "
        f"burnin={default_eval_benchmark['eval_burnin_ticks']} "
        f"ticks={default_eval_benchmark['eval_ticks']} "
        f"rs_completed_lives={default_eval_benchmark['vs_random_safe_completed_lives']}",
        flush=True,
    )
    return (
        default_eval_benchmark,
        int(eval_peak_bytes),
        eval_peak_mib,
        int(eff_resident_train_envs),
    )


def run_canary_suite(
    *,
    output_dir: Path,
    device: str | torch.device = "cuda",
    input_dir: Path | None = None,
    ledger_path: Path | None = None,
    max_wall_seconds: float | None = None,
    canary_envs: int = 8,
    canary_arena_radius: float = 120.0,
    canary_num_snakes: int = 12,
    canary_rollout_len: int = 4,
    canary_train_envs: int | None = None,
    compile_model: bool = False,
    fast_learner_ops: bool = True,
    canary_eval_cfg_override: V3PPOConfig | None = None,
    seed: int = 20261111,
) -> dict[str, Any]:
    """Run instrumented family canary exercising ruler, resume, & 1 default-config evaluation."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    resolved_dev = normalize_device(device)
    wall_t0 = time.monotonic()

    phase_stamp("v3_ppo_canary", "START", device=str(resolved_dev))

    staged_verification = _canary_stage_verify_staged_input(
        input_dir=input_dir,
        ledger_path=ledger_path,
        resolved_dev=resolved_dev,
    )

    n_snakes = max(6, int(canary_num_snakes))
    subset_learners = min(4, max(1, n_snakes // 2))
    canary_game_cfg = GameV3Config(arena_radius=float(canary_arena_radius), num_snakes=n_snakes)
    cardinal6_decisions, ruler_res = _canary_stage_ruler_sweep(
        out=out,
        resolved_dev=resolved_dev,
        canary_game_cfg=canary_game_cfg,
        canary_envs=canary_envs,
        canary_arena_radius=canary_arena_radius,
        n_snakes=n_snakes,
        subset_learners=subset_learners,
        canary_rollout_len=canary_rollout_len,
        compile_model=compile_model,
        fast_learner_ops=fast_learner_ops,
        max_wall_seconds=max_wall_seconds,
        wall_t0=wall_t0,
        seed=seed,
    )

    res_u2 = _canary_stage_train_and_resume(
        out=out,
        resolved_dev=resolved_dev,
        canary_envs=canary_envs,
        canary_arena_radius=canary_arena_radius,
        n_snakes=n_snakes,
        subset_learners=subset_learners,
        canary_rollout_len=canary_rollout_len,
        max_wall_seconds=max_wall_seconds,
        wall_t0=wall_t0,
        seed=seed,
    )

    (
        default_eval_benchmark,
        eval_peak_bytes,
        eval_peak_mib,
        eff_resident_train_envs,
    ) = _canary_stage_default_eval_benchmark(
        resolved_dev=resolved_dev,
        canary_game_cfg=canary_game_cfg,
        canary_envs=canary_envs,
        canary_rollout_len=canary_rollout_len,
        canary_train_envs=canary_train_envs,
        compile_model=compile_model,
        fast_learner_ops=fast_learner_ops,
        canary_eval_cfg_override=canary_eval_cfg_override,
        max_wall_seconds=max_wall_seconds,
        wall_t0=wall_t0,
        seed=seed,
    )

    canary_payload = {
        "schema": V3_PPO_CANARY_SCHEMA,
        "namespace": V3_PPO_NAMESPACE,
        "status": "PASS",
        "device": str(resolved_dev),
        "staged_input_verification": staged_verification,
        "cardinal6_decisions": cardinal6_decisions,
        "ruler_succeeded_cells": ruler_res["succeeded_cell_ids"],
        "train_updates_completed": int(res_u2["updates_completed"]),
        "last_eval": res_u2.get("last_eval"),
        "default_eval_benchmark": default_eval_benchmark,
        "default_eval_wall_seconds": float(default_eval_benchmark["wall_seconds"]),
        "default_eval_peak_memory_bytes": int(eval_peak_bytes),
        "default_eval_peak_memory_mib": eval_peak_mib,
        "default_eval_resident_train_envs": int(eff_resident_train_envs),
        "wall_seconds": round(time.monotonic() - wall_t0, 4),
    }
    write_create_only_json(out / "canary_checks.json", canary_payload)
    for rel in CANARY_REQUIRED_EXPORTS:
        verify_completed_artifact(out / rel)

    phase_stamp("v3_ppo_canary", "END", status="PASS", wall_seconds=canary_payload["wall_seconds"])
    return canary_payload


def _parse_int_csv(raw: str) -> list[int]:
    vals = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"Expected comma-separated positive integers, got {raw!r}")
    return vals


def _parse_nets_csv(raw: str) -> list[NetSize]:
    out: list[NetSize] = []
    for tok in raw.split(","):
        t = tok.strip().upper()
        if not t:
            continue
        sz = get_net_preset(t).name
        if sz not in out:
            out.append(sz)
    if not out:
        raise ValueError("Empty --nets list")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m research.v3_ppo.ruler``.

    Optional overrides default to ``None`` so :func:`resolve_effective_ruler_config` is the
    single source of precedence across CLI flags, environment variables, and defaults.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)

    parser.add_argument("--canary-suite", action="store_true")
    parser.add_argument("--canary-envs", type=int, default=8)
    parser.add_argument("--canary-arena-radius", type=float, default=120.0)
    parser.add_argument("--canary-num-snakes", type=int, default=12)
    parser.add_argument("--canary-rollout-len", type=int, default=4)
    parser.add_argument("--canary-train-envs", type=int, default=None)

    parser.add_argument("--envs", type=str, default=None)
    parser.add_argument("--nets", type=str, default=None)
    parser.add_argument("--learners", type=str, default=None)
    parser.add_argument("--arena-radius", type=float, default=None)
    parser.add_argument("--num-snakes", type=int, default=None)
    parser.add_argument("--burnin-ticks", type=int, default=None)
    parser.add_argument("--warmup-ticks", type=int, default=4)
    parser.add_argument("--collect-ticks", type=int, default=24)
    parser.add_argument("--warmup-updates", type=int, default=None)
    parser.add_argument("--timed-updates", type=int, default=None)
    parser.add_argument("--rollout-len", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--vram-budget-bytes", type=int, default=None)

    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", dest="compile", action="store_true", default=None)
    compile_group.add_argument("--no-compile", dest="compile", action="store_false")

    fp32_group = parser.add_mutually_exclusive_group()
    fp32_group.add_argument(
        "--include-fp32", dest="include_fp32", action="store_true", default=None
    )
    fp32_group.add_argument("--no-include-fp32", dest="include_fp32", action="store_false")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no-amp", "--fp32", dest="amp", action="store_false")

    cl_group = parser.add_mutually_exclusive_group()
    cl_group.add_argument(
        "--channels-last", dest="use_channels_last", action="store_true", default=None
    )
    cl_group.add_argument("--no-channels-last", dest="use_channels_last", action="store_false")

    fast_group = parser.add_mutually_exclusive_group()
    fast_group.add_argument(
        "--fast-learner-ops", dest="fast_learner_ops", action="store_true", default=None
    )
    fast_group.add_argument("--no-fast-learner-ops", dest="fast_learner_ops", action="store_false")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    return parser


def resolve_effective_ruler_config(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve effective canary/ruler configuration (`explicit CLI > env vars > defaults`)."""
    env = os.environ if environ is None else environ
    default_game = GameV3Config()
    default_ppo = V3PPOConfig()

    def _pick_bool(cli_val: bool | None, env_key: str, default_val: bool) -> bool:
        if cli_val is not None:
            return bool(cli_val)
        env_val = parse_env_bool(env_key, env.get(env_key))
        return default_val if env_val is None else env_val

    eff_device = (
        str(args.device) if args.device is not None else str(env.get("SNAKE_DQN_DEVICE") or "cuda")
    )
    eff_max_wall = (
        float(args.max_wall_seconds)
        if args.max_wall_seconds is not None
        else (float(env["V3_PPO_MAX_WALL_SECONDS"]) if env.get("V3_PPO_MAX_WALL_SECONDS") else None)
    )
    eff_canary_train_envs = (
        int(args.canary_train_envs)
        if args.canary_train_envs is not None
        else (int(env["V3_PPO_CANARY_TRAIN_ENVS"]) if env.get("V3_PPO_CANARY_TRAIN_ENVS") else None)
    )
    eff_envs_csv = (
        args.envs
        if args.envs is not None
        else (env.get("V3_PPO_RULER_ENVS") or ",".join(str(x) for x in DEFAULT_RULER_ENVS))
    )
    eff_nets_csv = (
        args.nets
        if args.nets is not None
        else (env.get("V3_PPO_RULER_NETS") or ",".join(DEFAULT_RULER_NETS))
    )
    eff_learners_csv = (
        args.learners
        if args.learners is not None
        else (env.get("V3_PPO_RULER_LEARNERS") or ",".join(str(x) for x in DEFAULT_RULER_LEARNERS))
    )
    eff_arena_radius = (
        float(args.arena_radius)
        if args.arena_radius is not None
        else (
            float(env["V3_PPO_ARENA_RADIUS"])
            if env.get("V3_PPO_ARENA_RADIUS")
            else float(default_game.arena_radius)
        )
    )
    eff_num_snakes = (
        int(args.num_snakes)
        if args.num_snakes is not None
        else (
            int(env["V3_PPO_NUM_SNAKES"])
            if env.get("V3_PPO_NUM_SNAKES")
            else int(default_game.num_snakes)
        )
    )
    eff_burnin_ticks = (
        int(args.burnin_ticks)
        if args.burnin_ticks is not None
        else (int(env["V3_PPO_BURNIN_TICKS"]) if env.get("V3_PPO_BURNIN_TICKS") else 60)
    )
    eff_warmup_updates = (
        int(args.warmup_updates)
        if args.warmup_updates is not None
        else (
            int(env["V3_PPO_RULER_WARMUP_UPDATES"])
            if env.get("V3_PPO_RULER_WARMUP_UPDATES")
            else DEFAULT_RULER_WARMUP_UPDATES
        )
    )
    eff_timed_updates = (
        int(args.timed_updates)
        if args.timed_updates is not None
        else (
            int(env["V3_PPO_RULER_TIMED_UPDATES"])
            if env.get("V3_PPO_RULER_TIMED_UPDATES")
            else DEFAULT_RULER_TIMED_UPDATES
        )
    )
    eff_rollout_len = (
        int(args.rollout_len)
        if args.rollout_len is not None
        else (
            int(env["V3_PPO_RULER_ROLLOUT_LEN"])
            if env.get("V3_PPO_RULER_ROLLOUT_LEN")
            else DEFAULT_RULER_ROLLOUT_LEN
        )
    )
    eff_ppo_epochs = (
        int(args.ppo_epochs) if args.ppo_epochs is not None else int(default_ppo.ppo_epochs)
    )
    eff_seed = int(args.seed) if args.seed is not None else int(default_ppo.seed)

    eff_compile = _pick_bool(args.compile, "V3_PPO_COMPILE", bool(default_ppo.compile_model))
    eff_fast_ops = _pick_bool(
        args.fast_learner_ops, "V3_PPO_FAST_LEARNER_OPS", bool(default_ppo.fast_learner_ops)
    )
    eff_amp = _pick_bool(args.amp, "V3_PPO_AMP", bool(default_ppo.amp))
    eff_channels_last = _pick_bool(
        args.use_channels_last, "V3_PPO_CHANNELS_LAST", bool(default_ppo.use_channels_last)
    )
    eff_include_fp32 = _pick_bool(args.include_fp32, "V3_PPO_INCLUDE_FP32", False)

    if bool(args.canary_suite):
        return {
            "mode": "canary",
            "output_dir": Path(args.output_dir),
            "device": eff_device,
            "input_dir": Path(args.input_dir) if args.input_dir is not None else None,
            "ledger_path": Path(args.ledger) if args.ledger is not None else None,
            "max_wall_seconds": eff_max_wall,
            "canary_envs": int(args.canary_envs),
            "canary_arena_radius": float(args.canary_arena_radius),
            "canary_num_snakes": int(args.canary_num_snakes),
            "canary_rollout_len": int(args.canary_rollout_len),
            "canary_train_envs": eff_canary_train_envs,
            "compile_model": eff_compile,
            "fast_learner_ops": eff_fast_ops,
            "seed": eff_seed,
        }

    return {
        "mode": "ruler",
        "output_dir": Path(args.output_dir),
        "device": eff_device,
        "input_dir": Path(args.input_dir) if args.input_dir is not None else None,
        "envs_list": _parse_int_csv(eff_envs_csv),
        "net_sizes": _parse_nets_csv(eff_nets_csv),
        "learners_list": _parse_int_csv(eff_learners_csv),
        "arena_radius": eff_arena_radius,
        "num_snakes": eff_num_snakes,
        "burnin_ticks": eff_burnin_ticks,
        "warmup_ticks": int(args.warmup_ticks),
        "collect_ticks": int(args.collect_ticks),
        "warmup_updates": eff_warmup_updates,
        "timed_updates": eff_timed_updates,
        "rollout_len": eff_rollout_len,
        "ppo_epochs": eff_ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "amp": eff_amp,
        "use_channels_last": eff_channels_last,
        "compile_model": eff_compile,
        "fast_learner_ops": eff_fast_ops,
        "include_fp32": eff_include_fp32,
        "vram_budget_bytes": args.vram_budget_bytes,
        "max_wall_seconds": eff_max_wall,
        "seed": eff_seed,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; exits non-zero whenever terminal ``status != 'PASS'``."""
    args = build_arg_parser().parse_args(argv)
    eff = resolve_effective_ruler_config(args)
    mode = eff.pop("mode")
    if mode == "canary":
        res = run_canary_suite(**eff)
        return 0 if res.get("status") == "PASS" else 1

    res = run_ruler(**eff)
    return 0 if res.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
