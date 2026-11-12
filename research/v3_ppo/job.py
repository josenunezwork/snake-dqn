"""Training job loop, pure checkpoint/eval schedulers, recovery stages, and CLI config resolver."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

from research.arena_ppo.runtime import SignalStopFlag, phase_stamp, verify_completed_artifact
from research.game_v3.config import GameV3Config
from research.v3_ppo.checkpoint_io import (
    INIT_ANCHOR_LIGHT_NAME,
    ckpt_tag,
    prune_old_checkpoints,
    prune_work_checkpoints_after_rewind,
    publish_checkpoint_to_output,
    verify_full_checkpoint_set,
)
from research.v3_ppo.config import (
    DEFAULT_CHECKPOINT_SHARD_BYTES,
    GPU_MAX_TOTAL_EXPORT_BYTES,
    NONE_MODE_MAX_TOTAL_EXPORT_BYTES,
    RECIPE_OVERRIDE_KEYS,
    TRAINER_CHECKPOINT_RESERVE_SECONDS,
    V3_PPO_NAMESPACE,
    VALID_FULL_EXPORT_MODES,
    VALID_NET_SIZES,
    VALID_OPPONENT_POLICIES,
    FullExportMode,
    V3PPOConfig,
    assert_resume_config_compatible,
    estimate_training_memory,
    extract_recipe_dict,
    format_recipe_overrides,
    get_net_preset,
    parse_env_bool,
    parse_recipe_overrides,
    recipe_overrides_from_default,
    resolve_full_export_settings,
)
from research.v3_ppo.exports import (
    RotatingCreateOnlyJSONLWriter,
    artifact_or_sidecar_exists,
    exported_payload_bytes,
    write_create_only_json,
)
from research.v3_ppo.resume import (
    resolve_resume_checkpoint,
    select_best_resume_candidate,
    validate_staged_ledger_resume,
    write_latest_checkpoint_manifest,
)
from research.v3_ppo.trainer import StopRequestedError, V3PPOTrainer

V3_PPO_TRAIN_RESULT_SCHEMA = "v3-ppo-train-result/v1"


# ---------------------------------------------------------------------------
# Pure evaluation and checkpoint scheduling helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalScheduleDecision:
    """Pure scheduling decision for periodic evaluation after an update completes."""

    due_by_updates: bool
    due_by_minutes: bool
    due: bool
    has_wall_budget: bool
    should_run: bool
    should_skip_for_wall: bool
    remaining_wall_seconds: float
    required_wall_seconds: float


def check_eval_wall_budget(
    *,
    elapsed_wall_seconds: float,
    max_wall_seconds: float | None,
    checkpoint_reserve_seconds: float,
    eval_estimated_seconds: float,
    last_eval_wall_seconds: float | None,
) -> tuple[bool, float, float]:
    """Return ``(can_eval, remaining_wall_seconds, required_wall_seconds)``."""
    est_s = (
        float(last_eval_wall_seconds)
        if last_eval_wall_seconds is not None
        else float(eval_estimated_seconds)
    )
    required_s = 2.0 * est_s
    if max_wall_seconds is None:
        return True, float("inf"), required_s
    rem_s = (float(max_wall_seconds) - float(checkpoint_reserve_seconds)) - float(
        elapsed_wall_seconds
    )
    return bool(rem_s >= required_s), rem_s, required_s


def decide_eval_action(
    *,
    updates_completed: int,
    total_updates: int,
    eval_every_updates: int,
    eval_every_minutes: float,
    now_mono: float,
    last_eval_mono: float,
    wall_t0: float,
    max_wall_seconds: float | None,
    checkpoint_reserve_seconds: float,
    eval_estimated_seconds: float,
    last_eval_wall_seconds: float | None,
) -> EvalScheduleDecision:
    """Decide whether periodic evaluation is due and whether sufficient wall budget remains."""
    is_target_end = int(updates_completed) >= int(total_updates)
    due_updates = int(eval_every_updates) > 0 and (
        (int(updates_completed) % int(eval_every_updates) == 0) or is_target_end
    )
    due_minutes = float(eval_every_minutes) > 0.0 and (
        (float(now_mono) - float(last_eval_mono)) >= float(eval_every_minutes) * 60.0
    )
    due = bool(due_updates or due_minutes)
    can_eval, rem_s, req_s = check_eval_wall_budget(
        elapsed_wall_seconds=float(now_mono) - float(wall_t0),
        max_wall_seconds=max_wall_seconds,
        checkpoint_reserve_seconds=checkpoint_reserve_seconds,
        eval_estimated_seconds=eval_estimated_seconds,
        last_eval_wall_seconds=last_eval_wall_seconds,
    )
    return EvalScheduleDecision(
        due_by_updates=due_updates,
        due_by_minutes=due_minutes,
        due=due,
        has_wall_budget=can_eval,
        should_run=bool(due and can_eval),
        should_skip_for_wall=bool(due and not can_eval),
        remaining_wall_seconds=rem_s,
        required_wall_seconds=req_s,
    )


@dataclass(frozen=True)
class CheckpointScheduleDecision:
    """Pure scheduling decision for work-dir checkpointing and periodic output exports."""

    due_light_updates: bool
    due_full_updates: bool
    due_light_work_time: bool
    due_light_export_time: bool
    due_full_time: bool
    save_work_light: bool
    save_work_full: bool
    want_periodic_full_export: bool
    want_periodic_light_export: bool


def decide_checkpoint_actions(
    *,
    updates_completed: int,
    checkpoint_every_updates: int,
    full_checkpoint_every_updates: int,
    light_checkpoint_interval_sec: float,
    full_checkpoint_interval_sec: float,
    now_mono: float,
    last_light_work_mono: float,
    last_light_export_mono: float,
    last_full_work_mono: float,
    published_periodic_fulls: int,
    max_periodic_full_exports: int,
    full_exports: FullExportMode = "periodic",
) -> CheckpointScheduleDecision:
    """Decide which LIGHT/FULL checkpoints to write in ``work/`` and publish to ``output/``.

    ``due_light_export_time`` compares against ``last_light_export_mono`` (not
    ``last_light_work_mono``) so capped ``work/`` FULL saves never stretch periodic LIGHT exports.
    ``save_work_full`` runs regardless of ``full_exports`` so VM ``work/`` FULL checkpoints remain
    available for in-session restart even when ``full_exports == "none"``.
    """
    due_light_u = int(checkpoint_every_updates) > 0 and (
        int(updates_completed) % int(checkpoint_every_updates) == 0
    )
    due_full_u = int(full_checkpoint_every_updates) > 0 and (
        int(updates_completed) % int(full_checkpoint_every_updates) == 0
    )
    due_light_work_t = (float(now_mono) - float(last_light_work_mono)) >= float(
        light_checkpoint_interval_sec
    )
    due_light_export_t = (float(now_mono) - float(last_light_export_mono)) >= float(
        light_checkpoint_interval_sec
    )
    due_full_t = (float(now_mono) - float(last_full_work_mono)) >= float(
        full_checkpoint_interval_sec
    )
    save_work_light = bool(
        due_light_u or due_light_work_t or due_light_export_t or due_full_u or due_full_t
    )
    save_work_full = bool(due_full_u or due_full_t)
    eff_periodic_cap = max(0, int(max_periodic_full_exports)) if full_exports == "periodic" else 0
    want_periodic_full = bool(save_work_full and int(published_periodic_fulls) < eff_periodic_cap)
    want_periodic_light = bool(due_light_export_t or due_light_u)
    return CheckpointScheduleDecision(
        due_light_updates=due_light_u,
        due_full_updates=due_full_u,
        due_light_work_time=due_light_work_t,
        due_light_export_time=due_light_export_t,
        due_full_time=due_full_t,
        save_work_light=save_work_light,
        save_work_full=save_work_full,
        want_periodic_full_export=want_periodic_full,
        want_periodic_light_export=want_periodic_light,
    )


def check_periodic_export_budget(
    *,
    current_exported_bytes: int,
    new_files_bytes: int,
    max_session_export_bytes: int,
    last_measured_full_bytes: int | None,
    est_light_bytes: int,
    num_envs: int,
    include_final_full_reserve: bool = True,
) -> tuple[bool, int]:
    """Return ``(within_budget, projected_bytes)`` reserving room for final boundary export(s)."""
    if int(new_files_bytes) <= 0:
        return True, int(current_exported_bytes)
    if not include_final_full_reserve:
        stop_reserve = min(int(max_session_export_bytes) // 4, int(est_light_bytes))
    else:
        stop_reserve = (
            int(last_measured_full_bytes)
            if last_measured_full_bytes is not None
            else min(
                int(max_session_export_bytes) // 4,
                int(est_light_bytes) + int(num_envs) * 16 * 1024 * 1024,
            )
        )
    projected = int(current_exported_bytes) + int(new_files_bytes) + stop_reserve
    return bool(projected <= int(max_session_export_bytes)), projected


# ---------------------------------------------------------------------------
# Progress formatting and named training-job stages
# ---------------------------------------------------------------------------


def _format_progress_line(
    step_metrics: dict[str, Any],
    *,
    total_updates: int,
    wall_elapsed: float,
    eval_wall_total: float = 0.0,
    eval_metrics: dict[str, Any] | None = None,
) -> str:
    """Format one ``V3_PPO_PROGRESS`` telemetry line for a completed training update.

    Note: ``ep_ret`` (``step_metrics['mean_ep_return']``, also recorded in ``metrics/updates_*.jsonl``
    and as ``last_rollout_mean_ep_return`` in ``train_result.json`` / checkpoints) is
    **recipe-dependent** because it sums ``cfg``'s ``reward_*`` terms over completed learner lives;
    compare policies across recipes using the recipe-invariant ``eval_*`` fields instead.
    """
    eval_share_pct = 100.0 * float(eval_wall_total) / max(1.0e-9, float(wall_elapsed))
    line = (
        f"V3_PPO_PROGRESS "
        f"update={step_metrics['update']}/{total_updates} "
        f"agent_steps={step_metrics['total_agent_steps']} "
        f"sps={step_metrics['agent_steps_per_sec']:.1f} "
        f"sps_scope=train_excl_eval "
        f"ep_ret={step_metrics['mean_ep_return']:.4f} "
        f"p_loss={step_metrics['policy_loss']:.4f} "
        f"v_loss={step_metrics['value_loss']:.4f} "
        f"ent={step_metrics['entropy']:.4f} "
        f"kl={step_metrics['approx_kl']:.5f} "
        f"wall_s={wall_elapsed:.2f} "
        f"eval_wall_s={eval_wall_total:.2f} "
        f"eval_wall_pct={eval_share_pct:.2f}"
    )
    if eval_metrics is not None and not bool(eval_metrics.get("aborted", False)):
        comb = eval_metrics["combined"]
        line += (
            f" eval_run_s={eval_metrics['wall_seconds']:.2f}"
            f" eval_tavg_len={comb['time_averaged_length']:.2f}"
            f" eval_dlen_mean={comb['mean_death_length']:.2f}"
            f" eval_dlen_p95={comb['p95_death_length']:.2f}"
            f" eval_deaths={comb['total_deaths']}"
            f" eval_deaths_1k={comb['deaths_per_1k_agent_steps']:.2f}"
            f" eval_mass_1k={comb['mass_gained_per_1k_agent_steps']:.2f}"
            f" eval_kills_1k={comb['kills_per_1k_agent_steps']:.2f}"
        )
    return line


def _snapshot_clean_state(t_ref: V3PPOTrainer) -> dict[str, Any]:
    return {
        "device": str(t_ref.device),
        "parameter_count": int(t_ref.raw_model.parameter_count()),
        "updates_completed": int(t_ref.updates_completed),
        "world_steps_collected": int(t_ref.world_steps_collected),
        "agent_steps_collected": int(t_ref.agent_steps_collected),
        "learner_steps_collected": int(t_ref.learner_steps_collected),
        "episodes_completed": int(t_ref.episodes_completed),
        "last_rollout_mean_ep_return": float(t_ref.last_rollout_mean_ep_return),
        "last_eval": t_ref.last_eval_metrics,
    }


def _counters_dict(t_ref: V3PPOTrainer) -> dict[str, Any]:
    return {
        "updates_completed": int(t_ref.updates_completed),
        "agent_steps_collected": int(t_ref.agent_steps_collected),
        "world_steps_collected": int(t_ref.world_steps_collected),
    }


@dataclass
class _JobRunState:
    """Mutable session state shared across the named stages of ``run_training_job``."""

    cfg: V3PPOConfig
    out: Path
    wrk: Path
    work_ckpts: Path
    out_ckpts: Path
    eff_max_export_bytes: int
    max_wall_seconds: float | None
    checkpoint_reserve_sec: float
    wall_t0: float
    staged_meta: dict[str, Any] | None
    trainer: V3PPOTrainer
    resumed_from: dict[str, Any] | None
    clean_summary: dict[str, Any]
    metrics_writer: RotatingCreateOnlyJSONLWriter
    eval_writer: RotatingCreateOnlyJSONLWriter
    last_light_work_mono: float
    last_light_export_mono: float
    last_full_work_mono: float
    last_eval_mono: float
    est_light_bytes: int
    updates_this_session: int = 0
    session_agent_steps: int = 0
    train_wall_seconds_total: float = 0.0
    eval_wall_seconds_total: float = 0.0
    eval_runs_completed: int = 0
    eval_runs_aborted: int = 0
    eval_runs_skipped: int = 0
    last_eval_wall_sec: float | None = None
    in_session_restarts: int = 0
    in_session_restart_errors: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = "completed"
    error_class: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None
    trainer_state_clean: bool = True
    aborted_mid_update: bool = False
    published_periodic_lights: int = 0
    published_periodic_fulls: int = 0
    last_measured_full_bytes: int | None = None
    allow_recipe_change: bool = False

    def wall_expired(self) -> bool:
        if self.max_wall_seconds is None:
            return False
        return (time.monotonic() - self.wall_t0) >= (
            float(self.max_wall_seconds) - self.checkpoint_reserve_sec
        )


def _stage_init_job_dirs_and_config(
    *,
    cfg: V3PPOConfig,
    output_dir: Path,
    work_dir: Path | None,
    max_session_export_bytes: int | None,
    allow_recipe_change: bool = False,
) -> tuple[Path, Path, Path, Path, int]:
    """Create output/work checkpoint directories, resolve byte cap, and precheck ``config.json``."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wrk = Path(work_dir).resolve() if work_dir is not None else (out.parent / f".{out.name}_work")
    wrk.mkdir(parents=True, exist_ok=True)
    work_ckpts = wrk / "checkpoints"
    work_ckpts.mkdir(parents=True, exist_ok=True)
    out_ckpts = out / "checkpoints"
    out_ckpts.mkdir(parents=True, exist_ok=True)

    if max_session_export_bytes is not None:
        eff_max_export_bytes = int(max_session_export_bytes)
    elif os.environ.get("V3_PPO_MAX_SESSION_EXPORT_BYTES"):
        eff_max_export_bytes = int(os.environ["V3_PPO_MAX_SESSION_EXPORT_BYTES"])
    elif cfg.full_exports == "none":
        eff_max_export_bytes = int(NONE_MODE_MAX_TOTAL_EXPORT_BYTES)
    else:
        eff_max_export_bytes = int(GPU_MAX_TOTAL_EXPORT_BYTES["L4"])

    cfg_json_path = out / "config.json"
    if artifact_or_sidecar_exists(cfg_json_path):
        verify_completed_artifact(cfg_json_path)
        existing_cfg = V3PPOConfig.from_dict(json.loads(cfg_json_path.read_text(encoding="utf-8")))
        assert_resume_config_compatible(
            existing_cfg,
            cfg,
            allow_recipe_change=allow_recipe_change,
            context="Existing config.json mismatch on immutable key",
            reference_label="existing",
            candidate_label="current",
        )

    return out, wrk, work_ckpts, out_ckpts, eff_max_export_bytes


def _stage_init_trainer_and_resume(
    *,
    cfg: V3PPOConfig,
    device: str | torch.device,
    input_dir: Path | None,
    ledger_path: Path | None,
    work_ckpts: Path,
    out_ckpts: Path,
    allow_recipe_change: bool = False,
) -> tuple[V3PPOTrainer, dict[str, Any] | None, dict[str, Any] | None]:
    """Construct trainer, load best resume checkpoint if present, or write ``u0000000`` anchor."""
    staged_meta = validate_staged_ledger_resume(input_dir=input_dir, ledger_path=ledger_path)
    trainer = V3PPOTrainer(cfg, device=device)
    resumed_from: dict[str, Any] | None = None

    initial_cand = select_best_resume_candidate(
        work_ckpts=work_ckpts,
        out_ckpts=out_ckpts,
        staged_meta=staged_meta,
        include_init_anchor=False,
        light_resume_margin_minutes=float(cfg.light_resume_margin_minutes),
        log_choice=True,
        context="initial_resume",
    )
    if initial_cand is not None:
        load_info = trainer.load_checkpoint(
            Path(initial_cand["path"]),
            verify_markers=bool(initial_cand["require_markers"]),
            restore_rng=True,
            allow_recipe_change=allow_recipe_change,
        )
        rc_rec = load_info.get("recipe_change")
        if rc_rec is not None:
            old_fields = {k: v["old"] for k, v in sorted(rc_rec["changed_fields"].items())}
            new_fields = {k: v["new"] for k, v in sorted(rc_rec["changed_fields"].items())}
            print(
                f"V3_PPO_RECIPE_CHANGE old={json.dumps(old_fields, sort_keys=True)} "
                f"new={json.dumps(new_fields, sort_keys=True)} update={rc_rec['update']}",
                flush=True,
            )
        resumed_from = {
            "relpath": str(initial_cand["relpath"]),
            "source": str(initial_cand["source"]),
            "checkpoint_kind": str(load_info["checkpoint_kind"]),
            "worlds_reinitialized": bool(load_info["worlds_reinitialized"]),
            "sha256": str(initial_cand["sha256"]),
            "updates_completed_at_resume": int(trainer.updates_completed),
        }
        if rc_rec is not None:
            resumed_from["recipe_change"] = rc_rec
        phase_stamp(
            "train_resume",
            "OK",
            relpath=resumed_from["relpath"],
            kind=resumed_from["checkpoint_kind"],
            source=resumed_from["source"],
            updates_completed=trainer.updates_completed,
        )
    else:
        init_anchor = work_ckpts / INIT_ANCHOR_LIGHT_NAME
        if not artifact_or_sidecar_exists(init_anchor):
            trainer.save_light_checkpoint(init_anchor, restart_generation=0)

    return trainer, staged_meta, resumed_from


def _stage_recover_trainer_in_session(state: _JobRunState, exc: RuntimeError) -> None:
    """Log restart traceback, rebuild trainer from newest clean checkpoint, and advance perm RNG."""
    state.in_session_restarts += 1
    exc_class = type(exc).__name__
    exc_tb = traceback.format_exc()
    state.in_session_restart_errors.append(
        {
            "attempt": int(state.in_session_restarts),
            "failed_update": int(state.trainer.updates_completed) + 1,
            "error_class": exc_class,
            "error": str(exc),
            "traceback": exc_tb,
        }
    )
    print(
        f"V3_PPO_IN_SESSION_RESTART attempt={state.in_session_restarts}/"
        f"{state.cfg.max_in_session_restarts} error_class={exc_class} "
        f"error={exc_class}: {exc}",
        flush=True,
    )
    print(exc_tb.rstrip("\n"), flush=True)
    phase_stamp(
        "in_session_restart",
        "RECOVER",
        attempt=state.in_session_restarts,
        error_class=exc_class,
        error=f"{exc_class}: {exc}",
    )
    rec_dev = state.trainer.device
    del state.trainer
    gc.collect()
    if rec_dev.type == "cuda":
        torch.cuda.empty_cache()

    state.trainer = V3PPOTrainer(state.cfg, device=rec_dev)
    rec_cand = select_best_resume_candidate(
        work_ckpts=state.work_ckpts,
        out_ckpts=state.out_ckpts,
        staged_meta=state.staged_meta,
        include_init_anchor=True,
        light_resume_margin_minutes=float(state.cfg.light_resume_margin_minutes),
        log_choice=True,
        context=f"in_session_restart_{state.in_session_restarts}",
    )
    if rec_cand is not None:
        state.trainer.load_checkpoint(
            Path(rec_cand["path"]),
            verify_markers=bool(rec_cand["require_markers"]),
            restore_rng=True,
            allow_recipe_change=state.allow_recipe_change,
        )
    prune_work_checkpoints_after_rewind(
        state.work_ckpts,
        rewind_updates=int(state.trainer.updates_completed),
    )
    _ = torch.randperm(8, generator=state.trainer.perm_gen)
    state.trainer_state_clean = True
    state.clean_summary = _snapshot_clean_state(state.trainer)


def _stage_maybe_run_periodic_eval(
    state: _JobRunState,
    *,
    now_mono: float,
    eval_should_stop: Callable[[], bool],
) -> dict[str, Any] | None:
    """Run or skip periodic evaluation according to :func:`decide_eval_action`."""
    decision = decide_eval_action(
        updates_completed=state.trainer.updates_completed,
        total_updates=int(state.cfg.total_updates),
        eval_every_updates=int(state.cfg.eval_every_updates),
        eval_every_minutes=float(state.cfg.eval_every_minutes),
        now_mono=now_mono,
        last_eval_mono=state.last_eval_mono,
        wall_t0=state.wall_t0,
        max_wall_seconds=state.max_wall_seconds,
        checkpoint_reserve_seconds=state.checkpoint_reserve_sec,
        eval_estimated_seconds=float(state.cfg.eval_estimated_seconds),
        last_eval_wall_seconds=state.last_eval_wall_sec,
    )
    if not decision.due:
        return None
    if decision.should_skip_for_wall:
        state.eval_runs_skipped += 1
        state.last_eval_mono = time.monotonic()
        print(
            f"V3_PPO_EVAL_SKIPPED update={state.trainer.updates_completed} "
            f"remaining_wall_s={decision.remaining_wall_seconds:.2f} "
            f"threshold_s={decision.required_wall_seconds:.2f} "
            "reason=insufficient_remaining_wall",
            flush=True,
        )
        return None

    raw_eval = state.trainer.run_evaluation(should_stop=eval_should_stop)
    ev_wall = float(raw_eval.get("wall_seconds", 0.0))
    state.eval_wall_seconds_total += ev_wall
    if bool(raw_eval.get("aborted", False)):
        state.eval_runs_aborted += 1
        print(
            f"V3_PPO_EVAL_ABORTED update={state.trainer.updates_completed} "
            f"ticks_completed={raw_eval.get('ticks_completed', 0)} "
            f"wall_s={ev_wall:.2f}",
            flush=True,
        )
        return None

    state.eval_runs_completed += 1
    state.last_eval_wall_sec = ev_wall
    state.clean_summary = _snapshot_clean_state(state.trainer)
    state.eval_writer.append(
        {
            "update": state.trainer.updates_completed,
            "total_agent_steps": state.trainer.agent_steps_collected,
            **raw_eval,
        }
    )
    state.last_eval_mono = time.monotonic()
    return raw_eval


def _can_export_periodic_files(
    state: _JobRunState,
    files_to_publish: Sequence[Path],
    *,
    kind: str,
) -> bool:
    new_bytes = sum(
        int(f.stat().st_size)
        for f in files_to_publish
        if f.is_file() and not (state.out_ckpts / f.name).exists()
    )
    within_budget, projected = check_periodic_export_budget(
        current_exported_bytes=exported_payload_bytes(state.out),
        new_files_bytes=new_bytes,
        max_session_export_bytes=state.eff_max_export_bytes,
        last_measured_full_bytes=state.last_measured_full_bytes,
        est_light_bytes=state.est_light_bytes,
        num_envs=int(state.cfg.num_envs),
        include_final_full_reserve=bool(state.cfg.exports_final_full),
    )
    if not within_budget:
        print(
            f"V3_PPO_EXPORT_BUDGET_REACHED kind={kind} new_bytes={new_bytes} "
            f"projected_bytes={projected} cap_bytes={state.eff_max_export_bytes}",
            flush=True,
        )
    return within_budget


def _stage_save_and_publish_checkpoints(
    state: _JobRunState,
    *,
    now_mono: float,
    final_boundary: bool = False,
    is_stopping: Callable[[], bool] | None = None,
) -> None:
    """Save due ``work/`` checkpoints, prune ``work/``, and publish periodic ``output/`` exports.

    When ``final_boundary`` is ``True`` (the loop is about to exit on ``completed``,
    ``wall_clock_guard``, or ``stop_flag``), any coincidental periodic FULL build is deferred to
    ``_stage_finalize_training_job`` so the final ``LIGHT`` checkpoint is always exported and
    ``.complete`` before the final ``FULL`` shards start.
    """
    tag = ckpt_tag(state.trainer.updates_completed, state.in_session_restarts)
    sched = decide_checkpoint_actions(
        updates_completed=state.trainer.updates_completed,
        checkpoint_every_updates=int(state.cfg.checkpoint_every_updates),
        full_checkpoint_every_updates=int(state.cfg.full_checkpoint_every_updates),
        light_checkpoint_interval_sec=float(state.cfg.light_checkpoint_interval_sec),
        full_checkpoint_interval_sec=float(state.cfg.full_checkpoint_interval_sec),
        now_mono=now_mono,
        last_light_work_mono=state.last_light_work_mono,
        last_light_export_mono=state.last_light_export_mono,
        last_full_work_mono=state.last_full_work_mono,
        published_periodic_fulls=state.published_periodic_fulls,
        max_periodic_full_exports=int(state.cfg.max_periodic_full_exports),
        full_exports=state.cfg.full_exports,
    )

    light_work_path = state.work_ckpts / f"light_{tag}.pt"
    if sched.save_work_light:
        if not artifact_or_sidecar_exists(light_work_path):
            state.trainer.save_light_checkpoint(
                light_work_path,
                restart_generation=state.in_session_restarts,
            )
        state.last_light_work_mono = now_mono

    eff_final_boundary = bool(final_boundary or (is_stopping is not None and is_stopping()))
    full_work_index: Path | None = None
    full_work_meta: dict[str, Any] | None = None
    if sched.save_work_full and not eff_final_boundary:
        cand_idx = state.work_ckpts / f"full_{tag}.index.json"
        if not artifact_or_sidecar_exists(cand_idx):
            work_shard_cap = (
                max(int(state.cfg.checkpoint_shard_bytes), DEFAULT_CHECKPOINT_SHARD_BYTES)
                if state.cfg.full_exports == "none"
                else int(state.cfg.checkpoint_shard_bytes)
            )
            full_work_index = state.trainer.save_full_checkpoint(
                state.work_ckpts,
                tag=tag,
                light_path=light_work_path,
                max_shard_bytes=work_shard_cap,
                restart_generation=state.in_session_restarts,
            )
        else:
            full_work_index = cand_idx
        full_work_meta = verify_full_checkpoint_set(full_work_index)
        state.last_measured_full_bytes = int(full_work_meta["size_bytes"])
        state.last_full_work_mono = now_mono

    if sched.save_work_light:
        prune_old_checkpoints(
            state.work_ckpts,
            keep_full=int(state.cfg.keep_full_checkpoints),
            keep_light=int(state.cfg.keep_light_checkpoints),
        )

    published_full_this_step = False
    if (
        full_work_index is not None
        and full_work_meta is not None
        and sched.want_periodic_full_export
        and _can_export_periodic_files(state, full_work_meta["files"], kind="periodic_full")
    ):
        publish_checkpoint_to_output(
            full_work_index,
            state.out_ckpts,
            reason="periodic_full",
            counters=_counters_dict(state.trainer),
        )
        state.published_periodic_fulls += 1
        published_full_this_step = True
        state.last_light_export_mono = now_mono
        state.metrics_writer.seal_current_segment()
        state.eval_writer.seal_current_segment()

    if (
        not published_full_this_step
        and light_work_path.exists()
        and sched.want_periodic_light_export
        and _can_export_periodic_files(state, [light_work_path], kind="periodic_light")
    ):
        publish_checkpoint_to_output(
            light_work_path,
            state.out_ckpts,
            reason="periodic_light",
            counters=_counters_dict(state.trainer),
        )
        state.published_periodic_lights += 1
        state.last_light_export_mono = now_mono
        state.metrics_writer.seal_current_segment()
        state.eval_writer.seal_current_segment()


def _stage_finalize_training_job(
    state: _JobRunState,
    *,
    stop_flag: SignalStopFlag,
    eval_should_stop: Callable[[], bool],
) -> None:
    """Close JSONL writers, run final eval/checkpoint/rollback, and write ``train_result.json``."""

    def _log_finally_warn(stage: str, warn_exc: BaseException) -> None:
        warn_cls = type(warn_exc).__name__
        print(
            f"V3_PPO_FINALLY_WARN stage={stage} error={warn_cls}: {warn_exc}",
            flush=True,
        )
        print(traceback.format_exc().rstrip("\n"), flush=True)

    for writer_obj in (state.metrics_writer, state.eval_writer):
        try:
            writer_obj.close()
        except Exception as w_exc:
            _log_finally_warn("close_writer", w_exc)

    is_boundary_stop = (
        state.trainer_state_clean and not state.aborted_mid_update and state.error_message is None
    )
    is_clean_stop = is_boundary_stop and state.stopped_reason in {
        "completed",
        "wall_clock_guard",
    }

    eval_configured = (int(state.cfg.eval_every_updates) > 0) or (
        float(state.cfg.eval_every_minutes) > 0.0
    )
    if (
        is_clean_stop
        and eval_configured
        and state.trainer.last_eval_metrics is None
        and state.trainer.updates_completed > 0
        and state.eval_runs_aborted == 0
        and not stop_flag.stopped
    ):
        can_ev_f, rem_ev_f, thresh_ev_f = check_eval_wall_budget(
            elapsed_wall_seconds=time.monotonic() - state.wall_t0,
            max_wall_seconds=state.max_wall_seconds,
            checkpoint_reserve_seconds=state.checkpoint_reserve_sec,
            eval_estimated_seconds=float(state.cfg.eval_estimated_seconds),
            last_eval_wall_seconds=state.last_eval_wall_sec,
        )
        if not can_ev_f:
            state.eval_runs_skipped += 1
            print(
                f"V3_PPO_EVAL_SKIPPED update={state.trainer.updates_completed} "
                f"remaining_wall_s={rem_ev_f:.2f} threshold_s={thresh_ev_f:.2f} "
                "reason=insufficient_remaining_wall_at_stop",
                flush=True,
            )
        else:
            try:
                ev_final = state.trainer.run_evaluation(should_stop=eval_should_stop)
                ev_f_wall = float(ev_final.get("wall_seconds", 0.0))
                state.eval_wall_seconds_total += ev_f_wall
                if bool(ev_final.get("aborted", False)):
                    state.eval_runs_aborted += 1
                else:
                    state.eval_runs_completed += 1
                    state.clean_summary = _snapshot_clean_state(state.trainer)
                    eval_writer_final = RotatingCreateOnlyJSONLWriter(
                        state.out / "eval", prefix="eval_final"
                    )
                    eval_writer_final.append(
                        {
                            "update": state.trainer.updates_completed,
                            "total_agent_steps": state.trainer.agent_steps_collected,
                            **ev_final,
                        }
                    )
                    eval_writer_final.close()
            except Exception as ev_exc:
                _log_finally_warn("final_eval", ev_exc)

    final_summary = dict(state.clean_summary)
    if is_boundary_stop:
        stop_tag = ckpt_tag(state.trainer.updates_completed, state.in_session_restarts)
        stop_light = state.work_ckpts / f"light_{stop_tag}.pt"
        try:
            if not artifact_or_sidecar_exists(stop_light):
                state.trainer.save_light_checkpoint(
                    stop_light,
                    restart_generation=state.in_session_restarts,
                )
            publish_checkpoint_to_output(
                stop_light,
                state.out_ckpts,
                reason=state.stopped_reason,
                counters=_counters_dict(state.trainer),
            )
            if state.cfg.exports_final_full:
                stop_full = state.work_ckpts / f"full_{stop_tag}.index.json"
                if not artifact_or_sidecar_exists(stop_full):
                    stop_full = state.trainer.save_full_checkpoint(
                        state.work_ckpts,
                        tag=stop_tag,
                        light_path=stop_light,
                        restart_generation=state.in_session_restarts,
                    )
                publish_checkpoint_to_output(
                    stop_full,
                    state.out_ckpts,
                    reason=state.stopped_reason,
                    counters=_counters_dict(state.trainer),
                )
            prune_old_checkpoints(
                state.work_ckpts,
                keep_full=int(state.cfg.keep_full_checkpoints),
                keep_light=int(state.cfg.keep_light_checkpoints),
            )
        except Exception as ck_exc:
            _log_finally_warn("boundary_checkpoint", ck_exc)
        final_summary = _snapshot_clean_state(state.trainer)
    else:

        def _rollback_export_target(cand: dict[str, Any]) -> Path:
            if not state.cfg.exports_final_full and cand.get("kind") == "full":
                return Path(cand["light_path"])
            return Path(cand["path"])

        try:
            work_clean = resolve_resume_checkpoint(
                state.work_ckpts,
                require_markers=True,
                include_init_anchor=(state.staged_meta is None),
                light_resume_margin_minutes=float(state.cfg.light_resume_margin_minutes),
            )
            if work_clean is not None:
                wb_export_path = _rollback_export_target(work_clean)
                if wb_export_path.name != INIT_ANCHOR_LIGHT_NAME:
                    publish_checkpoint_to_output(
                        wb_export_path,
                        state.out_ckpts,
                        reason=f"rollback_{state.stopped_reason}",
                        counters={
                            "updates_completed": int(work_clean["updates_completed"]),
                            "agent_steps_collected": int(
                                work_clean.get("agent_steps_collected", 0)
                            ),
                            "world_steps_collected": int(
                                work_clean.get("world_steps_collected", 0)
                            ),
                        },
                    )
        except Exception as rb_exc1:
            _log_finally_warn("rollback_publish_work", rb_exc1)

        try:
            clean_fallback = select_best_resume_candidate(
                work_ckpts=state.work_ckpts,
                out_ckpts=state.out_ckpts,
                staged_meta=state.staged_meta,
                include_init_anchor=True,
                light_resume_margin_minutes=float(state.cfg.light_resume_margin_minutes),
                log_choice=True,
                context="finally_rollback",
            )
            if clean_fallback is not None:
                fb_path = Path(clean_fallback["path"])
                fb_export_path = _rollback_export_target(clean_fallback)
                if (
                    bool(clean_fallback.get("require_markers", False))
                    and fb_export_path.resolve().parent != state.out_ckpts.resolve()
                    and fb_export_path.name != INIT_ANCHOR_LIGHT_NAME
                ):
                    publish_checkpoint_to_output(
                        fb_export_path,
                        state.out_ckpts,
                        reason=f"rollback_{state.stopped_reason}",
                        counters={
                            "updates_completed": int(clean_fallback["updates_completed"]),
                            "agent_steps_collected": int(
                                clean_fallback.get("agent_steps_collected", 0)
                            ),
                            "world_steps_collected": int(
                                clean_fallback.get("world_steps_collected", 0)
                            ),
                        },
                        copy_only=(fb_export_path.resolve().parent != state.work_ckpts.resolve()),
                    )
                final_summary["updates_completed"] = int(clean_fallback["updates_completed"])
                final_summary["agent_steps_collected"] = int(
                    clean_fallback.get(
                        "agent_steps_collected", final_summary["agent_steps_collected"]
                    )
                )
                final_summary["world_steps_collected"] = int(
                    clean_fallback.get(
                        "world_steps_collected", final_summary["world_steps_collected"]
                    )
                )
                try:
                    state.trainer.load_checkpoint(
                        fb_path,
                        verify_markers=bool(clean_fallback.get("require_markers", False)),
                        restore_rng=False,
                        allow_recipe_change=state.allow_recipe_change,
                    )
                    final_summary = _snapshot_clean_state(state.trainer)
                except Exception as load_exc:
                    _log_finally_warn("rollback_load_checkpoint", load_exc)
        except Exception as rb_exc2:
            _log_finally_warn("rollback_select", rb_exc2)

    final_counters = {
        "updates_completed": int(final_summary["updates_completed"]),
        "agent_steps_collected": int(final_summary["agent_steps_collected"]),
        "world_steps_collected": int(final_summary["world_steps_collected"]),
    }
    try:
        write_latest_checkpoint_manifest(
            state.out_ckpts,
            reason=state.stopped_reason,
            counters=final_counters,
            light_resume_margin_minutes=float(state.cfg.light_resume_margin_minutes),
        )
    except Exception as mf_exc:
        _log_finally_warn("latest_manifest", mf_exc)

    passed = (
        state.error_message is None
        and not state.aborted_mid_update
        and state.stopped_reason in {"completed", "wall_clock_guard"}
        and (int(state.cfg.total_updates) == 0 or int(final_summary["updates_completed"]) >= 1)
        and (
            state.stopped_reason == "completed"
            or (state.max_wall_seconds is not None and state.updates_this_session >= 1)
        )
    )
    status = "PASS" if passed else "FAIL"

    total_wall_sec = max(1.0e-9, time.monotonic() - state.wall_t0)
    train_sps_excl_eval = (
        round(float(state.session_agent_steps) / max(1.0e-9, state.train_wall_seconds_total), 2)
        if state.updates_this_session > 0
        else 0.0
    )
    result_path = state.out / "train_result.json"
    if not artifact_or_sidecar_exists(result_path):
        result_payload = {
            "schema": V3_PPO_TRAIN_RESULT_SCHEMA,
            "namespace": V3_PPO_NAMESPACE,
            "status": status,
            "stopped_reason": state.stopped_reason,
            "aborted_mid_update": bool(state.aborted_mid_update),
            "in_session_restarts": int(state.in_session_restarts),
            "in_session_restart_errors": state.in_session_restart_errors,
            "error": state.error_message,
            "error_class": state.error_class,
            "error_traceback": state.error_traceback,
            "device": str(final_summary["device"]),
            "parameter_count": int(final_summary["parameter_count"]),
            "updates_completed": int(final_summary["updates_completed"]),
            "updates_this_session": int(state.updates_this_session),
            "world_steps_collected": int(final_summary["world_steps_collected"]),
            "agent_steps_collected": int(final_summary["agent_steps_collected"]),
            "learner_steps_collected": int(final_summary["learner_steps_collected"]),
            "episodes_completed": int(final_summary["episodes_completed"]),
            "last_rollout_mean_ep_return": round(
                float(final_summary["last_rollout_mean_ep_return"]), 6
            ),
            "last_eval": final_summary["last_eval"],
            "agent_steps_per_sec_scope": "train_excl_eval",
            "train_wall_seconds_excl_eval": round(state.train_wall_seconds_total, 4),
            "train_agent_steps_per_sec_excl_eval": train_sps_excl_eval,
            "eval_wall_seconds_total": round(state.eval_wall_seconds_total, 4),
            "eval_wall_share": round(state.eval_wall_seconds_total / total_wall_sec, 6),
            "eval_runs_completed": int(state.eval_runs_completed),
            "eval_runs_aborted": int(state.eval_runs_aborted),
            "eval_runs_skipped": int(state.eval_runs_skipped),
            "periodic_light_exports_published": int(state.published_periodic_lights),
            "periodic_full_exports_published": int(state.published_periodic_fulls),
            "session_export_bytes_published": int(exported_payload_bytes(state.out)),
            "max_session_export_bytes": int(state.eff_max_export_bytes),
            "resumed_from": state.resumed_from,
            "memory_estimate": estimate_training_memory(state.cfg),
            "config": state.cfg.to_dict(),
            "wall_seconds": round(total_wall_sec, 4),
        }
        write_create_only_json(result_path, result_payload)
    phase_stamp(
        "train_loop",
        "END",
        status=status,
        stopped_reason=state.stopped_reason,
        updates_completed=int(final_summary["updates_completed"]),
        in_session_restarts=state.in_session_restarts,
    )


def run_training_job(
    *,
    cfg: V3PPOConfig,
    output_dir: Path,
    work_dir: Path | None = None,
    device: str | torch.device = "cpu",
    input_dir: Path | None = None,
    ledger_path: Path | None = None,
    max_wall_seconds: float | None = None,
    max_session_export_bytes: int | None = None,
    allow_recipe_change: bool = False,
    step_hook: Callable[[V3PPOTrainer, int], None] | None = None,
) -> dict[str, Any]:
    """Execute training loop with wall-clock eval, LIGHT/FULL checkpoints, and crash recovery.

    Guarantees:
      - Every LIGHT checkpoint is exported at its real ``light_checkpoint_interval_sec`` (`30` min
        default) with no per-session count cap and no interval stretching.
      - Periodic sharded FULL sets are exported every ``full_checkpoint_interval_sec`` (`2` h
        default) up to ``cfg.max_periodic_full_exports`` per session, plus ``1`` FULL set on any
        clean boundary stop (`completed`, `wall_clock_guard`, or boundary cooperative signal).
      - ``keep_full_checkpoints`` / ``keep_light_checkpoints`` prune ``work/`` only; completed
        exports in ``output/`` are never rewritten or deleted.
      - Cooperative stop mid-update discards half-updated trainer state and rolls back cleanly,
        with protected ``finally`` fallbacks guaranteeing ``latest_manifest.json`` and
        ``train_result.json`` even if the device context is dead.
    """
    eff_allow_recipe_change = bool(allow_recipe_change)
    out, wrk, work_ckpts, out_ckpts, eff_max_export_bytes = _stage_init_job_dirs_and_config(
        cfg=cfg,
        output_dir=output_dir,
        work_dir=work_dir,
        max_session_export_bytes=max_session_export_bytes,
        allow_recipe_change=eff_allow_recipe_change,
    )
    wall_t0 = time.monotonic()
    trainer, staged_meta, resumed_from = _stage_init_trainer_and_resume(
        cfg=cfg,
        device=device,
        input_dir=input_dir,
        ledger_path=ledger_path,
        work_ckpts=work_ckpts,
        out_ckpts=out_ckpts,
        allow_recipe_change=eff_allow_recipe_change,
    )
    cfg = trainer.cfg
    cfg_json_path = out / "config.json"
    if not artifact_or_sidecar_exists(cfg_json_path):
        write_create_only_json(cfg_json_path, cfg.to_dict())
    now_init = time.monotonic()
    # Reserve `TRAINER_CHECKPOINT_RESERVE_SECONDS` (90 s; clamped to 8% on tiny sessions >= 5 s)
    # inside `max_wall_seconds` so an in-progress update completes before final eval/export starts.
    checkpoint_reserve_sec = (
        min(
            float(TRAINER_CHECKPOINT_RESERVE_SECONDS),
            max(5.0, float(max_wall_seconds) * 0.08),
        )
        if max_wall_seconds is not None
        else 0.0
    )
    state = _JobRunState(
        cfg=cfg,
        out=out,
        wrk=wrk,
        work_ckpts=work_ckpts,
        out_ckpts=out_ckpts,
        eff_max_export_bytes=eff_max_export_bytes,
        max_wall_seconds=max_wall_seconds,
        checkpoint_reserve_sec=checkpoint_reserve_sec,
        wall_t0=wall_t0,
        staged_meta=staged_meta,
        trainer=trainer,
        resumed_from=resumed_from,
        clean_summary=_snapshot_clean_state(trainer),
        metrics_writer=RotatingCreateOnlyJSONLWriter(out / "metrics", prefix="updates"),
        eval_writer=RotatingCreateOnlyJSONLWriter(out / "eval", prefix="eval"),
        last_light_work_mono=now_init,
        last_light_export_mono=now_init,
        last_full_work_mono=now_init,
        last_eval_mono=now_init,
        est_light_bytes=get_net_preset(cfg.net_size).est_light_checkpoint_bytes,
        allow_recipe_change=eff_allow_recipe_change,
    )

    with SignalStopFlag() as stop_flag:

        def _should_stop() -> bool:
            return bool(stop_flag.stopped)

        def _eval_should_stop() -> bool:
            return bool(stop_flag.stopped or state.wall_expired())

        try:
            active_recipe = extract_recipe_dict(cfg)
            recipe_ov_str = format_recipe_overrides(recipe_overrides_from_default(cfg))
            print(
                f"V3_PPO_RECIPE overrides={json.dumps(recipe_ov_str)} "
                f"recipe={json.dumps(active_recipe, sort_keys=True)}",
                flush=True,
            )
            phase_stamp(
                "train_loop",
                "START",
                device=str(state.trainer.device),
                num_envs=cfg.num_envs,
                learner_slots=cfg.resolved_learner_slots,
                net_size=cfg.net_size,
                start_update=state.trainer.updates_completed,
                target_updates=cfg.total_updates,
            )
            if stop_flag.stopped:
                state.stopped_reason = f"signal:{stop_flag.signal_name or 'STOP'}"
            elif state.wall_expired():
                state.stopped_reason = "wall_clock_guard"
            else:
                while state.trainer.updates_completed < int(cfg.total_updates):
                    if stop_flag.stopped:
                        state.stopped_reason = f"signal:{stop_flag.signal_name or 'STOP'}"
                        break
                    if state.wall_expired():
                        state.stopped_reason = "wall_clock_guard"
                        break

                    try:
                        state.trainer_state_clean = False
                        u_t0 = time.monotonic()
                        if step_hook is not None:
                            step_hook(state.trainer, int(state.trainer.updates_completed) + 1)
                        step_metrics = state.trainer.train_step(should_stop=_should_stop)
                        u_elapsed = time.monotonic() - u_t0
                        state.trainer_state_clean = True
                        state.updates_this_session += 1
                        state.session_agent_steps += int(step_metrics["agent_steps"])
                        state.train_wall_seconds_total += u_elapsed
                        state.clean_summary = _snapshot_clean_state(state.trainer)

                        if cfg.max_update_seconds is not None and u_elapsed > float(
                            cfg.max_update_seconds
                        ):
                            raise RuntimeError(
                                f"Update {state.trainer.updates_completed} took {u_elapsed:.2f}s > "
                                f"max_update_seconds={cfg.max_update_seconds:.2f}s"
                            )
                    except StopRequestedError:
                        state.aborted_mid_update = True
                        state.stopped_reason = f"signal:{stop_flag.signal_name or 'STOP'}"
                        break
                    except RuntimeError as exc:
                        if (
                            state.in_session_restarts >= int(cfg.max_in_session_restarts)
                            or stop_flag.stopped
                            or state.wall_expired()
                        ):
                            raise
                        _stage_recover_trainer_in_session(state, exc)
                        continue

                    eval_res = _stage_maybe_run_periodic_eval(
                        state,
                        now_mono=time.monotonic(),
                        eval_should_stop=_eval_should_stop,
                    )
                    state.metrics_writer.append(step_metrics)
                    now_mono = time.monotonic()
                    print(
                        _format_progress_line(
                            step_metrics,
                            total_updates=int(cfg.total_updates),
                            wall_elapsed=now_mono - state.wall_t0,
                            eval_wall_total=state.eval_wall_seconds_total,
                            eval_metrics=eval_res,
                        ),
                        flush=True,
                    )
                    # Compute boundary stop decision ONCE per update and reuse it for both
                    # `final_boundary` and loop exit so a due periodic FULL can never publish
                    # before the final LIGHT if `wall_expired()` trips during checkpointing.
                    if stop_flag.stopped:
                        post_stop_reason: str | None = f"signal:{stop_flag.signal_name or 'STOP'}"
                    elif state.wall_expired():
                        post_stop_reason = "wall_clock_guard"
                    elif state.trainer.updates_completed >= int(cfg.total_updates):
                        post_stop_reason = "completed"
                    else:
                        post_stop_reason = None

                    _stage_save_and_publish_checkpoints(
                        state,
                        now_mono=now_mono,
                        final_boundary=(post_stop_reason is not None),
                        is_stopping=lambda: bool(stop_flag.stopped or state.wall_expired()),
                    )
                    if post_stop_reason is not None:
                        state.stopped_reason = post_stop_reason
                        break

        except Exception as exc:
            state.error_class = type(exc).__name__
            state.error_message = str(exc)
            state.error_traceback = traceback.format_exc()
            state.stopped_reason = f"exception:{state.error_class}"
            print(
                f"V3_PPO_FINAL_FAILURE error_class={state.error_class} "
                f"error={state.error_class}: {exc}",
                flush=True,
            )
            print(state.error_traceback.rstrip("\n"), flush=True)
            raise
        finally:
            _stage_finalize_training_job(
                state,
                stop_flag=stop_flag,
                eval_should_stop=_eval_should_stop,
            )

    return json.loads((out / "train_result.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI argument parser and effective configuration precedence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectiveTrainJobConfig:
    """Resolved runtime arguments and ``V3PPOConfig`` for ``python -m research.v3_ppo.ppo``."""

    cfg: V3PPOConfig
    output_dir: Path
    work_dir: Path | None
    device: str
    input_dir: Path | None
    ledger_path: Path | None
    max_wall_seconds: float | None
    max_session_export_bytes: int | None
    recipe_overrides: dict[str, Any] = field(default_factory=dict)
    allow_recipe_change: bool = False


def _merge_config_json_overlay(
    cfg_dict: dict[str, Any],
    raw_json_str: str,
    *,
    source_label: str,
) -> set[str]:
    """Overlay parsed ``raw_json_str`` onto ``cfg_dict`` in-place and return explicit top keys."""
    parsed = json.loads(raw_json_str)
    if not isinstance(parsed, dict):
        raise ValueError(f"{source_label} must decode to a JSON object, got {type(parsed)!r}")
    valid_top_keys = {f.name for f in fields(V3PPOConfig)} | {"schema"}
    unknown_top = sorted(set(parsed.keys()) - valid_top_keys)
    if unknown_top:
        raise ValueError(f"Unknown keys in {source_label}: {unknown_top}")

    prev_net_size = cfg_dict["net_size"]
    valid_gc_keys = {f.name for f in fields(GameV3Config)}
    for key, val in parsed.items():
        if key == "schema":
            continue
        if key == "game_config":
            if not isinstance(val, dict):
                raise ValueError(
                    f"{source_label}.game_config must be a JSON object, got {type(val)!r}"
                )
            unknown_gc = sorted(set(val.keys()) - valid_gc_keys)
            if unknown_gc:
                raise ValueError(f"Unknown keys in {source_label}.game_config: {unknown_gc}")
            merged_gc = dict(cfg_dict["game_config"])
            merged_gc.update(val)
            cfg_dict["game_config"] = merged_gc
        else:
            cfg_dict[key] = val
    if "net_size" in parsed and parsed["net_size"] != prev_net_size:
        if "minibatch_size" not in parsed:
            cfg_dict["minibatch_size"] = None
        if "actor_chunk_learners" not in parsed:
            cfg_dict["actor_chunk_learners"] = None
    return set(parsed.keys())


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m research.v3_ppo.ppo``.

    Optional overrides default to ``None`` so :func:`resolve_effective_train_config` is the
    single source of precedence across CLI flags, ``--config-json``, environment variables,
    and :class:`V3PPOConfig` defaults.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument(
        "--recipe-json",
        type=str,
        default=None,
        help="Inline JSON object or path to JSON file overriding allow-listed recipe fields.",
    )
    parser.add_argument(
        "--allow-recipe-change",
        dest="allow_recipe_change",
        action="store_true",
        default=None,
        help="Permit allow-listed recipe field changes when resuming from a checkpoint.",
    )

    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-snakes", type=int, default=None)
    parser.add_argument("--arena-radius", type=float, default=None)
    parser.add_argument("--learner-slots", type=int, default=None)
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default=None,
        choices=VALID_OPPONENT_POLICIES,
    )
    parser.add_argument("--net-size", type=str, default=None, choices=VALID_NET_SIZES)
    parser.add_argument("--rollout-len", type=int, default=None)
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--light-checkpoint-interval-sec",
        "--checkpoint-interval-sec",
        dest="light_checkpoint_interval_sec",
        type=float,
        default=None,
    )
    parser.add_argument("--full-checkpoint-interval-sec", type=float, default=None)
    parser.add_argument(
        "--full-exports",
        type=str,
        default=None,
        choices=VALID_FULL_EXPORT_MODES,
        help="Simulator FULL checkpoint export mode: none (default), final, or periodic.",
    )
    parser.add_argument("--max-periodic-full-exports", type=int, default=None)
    parser.add_argument("--light-resume-margin-minutes", type=float, default=None)
    parser.add_argument("--checkpoint-shard-bytes", type=int, default=None)
    parser.add_argument("--checkpoint-every-updates", type=int, default=None)
    parser.add_argument("--full-checkpoint-every-updates", type=int, default=None)
    parser.add_argument("--eval-every-minutes", type=float, default=None)
    parser.add_argument("--eval-every-updates", type=int, default=None)
    parser.add_argument("--eval-envs", type=int, default=None)
    parser.add_argument("--eval-burnin-ticks", type=int, default=None)
    parser.add_argument("--eval-ticks", type=int, default=None)
    parser.add_argument("--eval-estimated-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", dest="compile", action="store_true", default=None)
    compile_group.add_argument("--no-compile", dest="compile", action="store_false")

    fast_group = parser.add_mutually_exclusive_group()
    fast_group.add_argument(
        "--fast-learner-ops", dest="fast_learner_ops", action="store_true", default=None
    )
    fast_group.add_argument("--no-fast-learner-ops", dest="fast_learner_ops", action="store_false")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no-amp", "--fp32", dest="amp", action="store_false")

    cl_group = parser.add_mutually_exclusive_group()
    cl_group.add_argument(
        "--channels-last", dest="use_channels_last", action="store_true", default=None
    )
    cl_group.add_argument("--no-channels-last", dest="use_channels_last", action="store_false")

    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument("--max-session-export-bytes", type=int, default=None)
    return parser


def resolve_effective_train_config(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
    base_cfg: V3PPOConfig | None = None,
    recipe_overrides: Mapping[str, Any] | None = None,
    default_full_exports: FullExportMode | None = None,
    forbid_recipe_scalar_conflict: bool = False,
) -> EffectiveTrainJobConfig:
    """Resolve effective training configuration from CLI args, ``--recipe-json``, JSON, and env.

    Single effective-config builder shared by ``prepare --mode train`` and ``ppo``.
    Precedence (highest to lowest):
      1. Explicit individual CLI flags (``--num-envs``, ``--lr``, ``--amp``/``--no-amp``, etc.)
      2. Explicit ``--recipe-json`` / ``recipe_overrides`` (allow-listed ``RECIPE_OVERRIDE_KEYS``)
      3. Explicit ``--config-json`` CLI flag (keys present in the JSON object)
      4. Environment variables (``V3_PPO_CONFIG_JSON``, overlaid by scalar ``V3_PPO_*`` and
         ``SNAKE_DQN_DEVICE``)
      5. ``base_cfg`` (when provided) or :class:`V3PPOConfig` / :class:`GameV3Config` defaults.
    """
    env = os.environ if environ is None else environ

    seed_cfg = base_cfg if base_cfg is not None else V3PPOConfig()
    cfg_dict = seed_cfg.to_dict()
    cfg_dict.pop("schema", None)
    if default_full_exports is not None:
        def_mode, def_mp = resolve_full_export_settings(
            default_full_exports,
            None,
            default_full_exports=default_full_exports,
            default_max_periodic=int(seed_cfg.max_periodic_full_exports),
        )
        cfg_dict["full_exports"] = def_mode
        cfg_dict["max_periodic_full_exports"] = def_mp

    env_cfg_json = env.get("V3_PPO_CONFIG_JSON")
    if env_cfg_json:
        _merge_config_json_overlay(
            cfg_dict,
            env_cfg_json,
            source_label="env:V3_PPO_CONFIG_JSON",
        )
    env_arena_radius = env.get("V3_PPO_ARENA_RADIUS")
    if env_arena_radius:
        cfg_dict["game_config"]["arena_radius"] = float(env_arena_radius)
    env_num_snakes = env.get("V3_PPO_NUM_SNAKES")
    if env_num_snakes:
        cfg_dict["game_config"]["num_snakes"] = int(env_num_snakes)

    for env_key, cfg_key in (
        ("V3_PPO_COMPILE", "compile_model"),
        ("V3_PPO_FAST_LEARNER_OPS", "fast_learner_ops"),
        ("V3_PPO_AMP", "amp"),
        ("V3_PPO_CHANNELS_LAST", "use_channels_last"),
    ):
        parsed_b = parse_env_bool(env_key, env.get(env_key))
        if parsed_b is not None:
            cfg_dict[cfg_key] = parsed_b

    if args.config_json is not None:
        _merge_config_json_overlay(
            cfg_dict,
            args.config_json,
            source_label="--config-json",
        )

    if args.arena_radius is not None:
        cfg_dict["game_config"]["arena_radius"] = float(args.arena_radius)
    if args.num_snakes is not None:
        cfg_dict["game_config"]["num_snakes"] = int(args.num_snakes)
    if args.net_size is not None:
        if args.net_size != cfg_dict["net_size"]:
            if args.minibatch_size is None:
                cfg_dict["minibatch_size"] = None
            cfg_dict["actor_chunk_learners"] = None
        cfg_dict["net_size"] = args.net_size

    merged_recipe: dict[str, Any] = {}
    if recipe_overrides is not None:
        merged_recipe.update(
            parse_recipe_overrides(recipe_overrides, source_label="recipe_overrides")
        )
    arg_recipe_json = getattr(args, "recipe_json", None)
    if arg_recipe_json is not None:
        merged_recipe.update(parse_recipe_overrides(arg_recipe_json, source_label="--recipe-json"))
    if merged_recipe:
        cfg_dict.update(merged_recipe)

    cli_scalar_overrides: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
        ("num_envs", "num_envs", int),
        ("learner_slots", "learner_slots", int),
        ("opponent_policy", "opponent_policy", str),
        ("rollout_len", "rollout_len", int),
        ("total_updates", "total_updates", int),
        ("ppo_epochs", "ppo_epochs", int),
        ("minibatch_size", "minibatch_size", int),
        ("lr", "lr", float),
        ("light_checkpoint_interval_sec", "light_checkpoint_interval_sec", float),
        ("full_checkpoint_interval_sec", "full_checkpoint_interval_sec", float),
        ("light_resume_margin_minutes", "light_resume_margin_minutes", float),
        ("checkpoint_shard_bytes", "checkpoint_shard_bytes", int),
        ("checkpoint_every_updates", "checkpoint_every_updates", int),
        ("full_checkpoint_every_updates", "full_checkpoint_every_updates", int),
        ("eval_every_minutes", "eval_every_minutes", float),
        ("eval_every_updates", "eval_every_updates", int),
        ("eval_envs", "eval_envs", int),
        ("eval_burnin_ticks", "eval_burnin_ticks", int),
        ("eval_ticks", "eval_ticks", int),
        ("eval_estimated_seconds", "eval_estimated_seconds", float),
        ("seed", "seed", int),
        ("compile", "compile_model", bool),
        ("fast_learner_ops", "fast_learner_ops", bool),
        ("amp", "amp", bool),
        ("use_channels_last", "use_channels_last", bool),
    )
    for arg_attr, cfg_key, caster in cli_scalar_overrides:
        val = getattr(args, arg_attr, None)
        if val is not None:
            cast_val = caster(val)
            if (
                forbid_recipe_scalar_conflict
                and cfg_key in merged_recipe
                and merged_recipe[cfg_key] != cast_val
            ):
                raise ValueError(
                    f"Conflicting explicit override for {cfg_key!r}: CLI flag={cast_val!r} vs "
                    f"recipe override={merged_recipe[cfg_key]!r}"
                )
            cfg_dict[cfg_key] = cast_val

    cli_full_exports = getattr(args, "full_exports", None)
    cli_max_periodic = getattr(args, "max_periodic_full_exports", None)
    if cli_full_exports is not None or cli_max_periodic is not None:
        eff_full_mode, eff_max_periodic = resolve_full_export_settings(
            cli_full_exports,
            cli_max_periodic,
            default_full_exports=cfg_dict["full_exports"],
            default_max_periodic=int(cfg_dict["max_periodic_full_exports"]),
        )
        cfg_dict["full_exports"] = eff_full_mode
        cfg_dict["max_periodic_full_exports"] = eff_max_periodic

    eff_device = (
        str(args.device) if args.device is not None else str(env.get("SNAKE_DQN_DEVICE") or "cpu")
    )
    eff_max_wall_seconds = (
        float(args.max_wall_seconds)
        if args.max_wall_seconds is not None
        else (float(env["V3_PPO_MAX_WALL_SECONDS"]) if env.get("V3_PPO_MAX_WALL_SECONDS") else None)
    )
    eff_max_export_bytes = (
        int(args.max_session_export_bytes)
        if args.max_session_export_bytes is not None
        else (
            int(env["V3_PPO_MAX_SESSION_EXPORT_BYTES"])
            if env.get("V3_PPO_MAX_SESSION_EXPORT_BYTES")
            else None
        )
    )

    env_allow_recipe = parse_env_bool(
        "V3_PPO_ALLOW_RECIPE_CHANGE", env.get("V3_PPO_ALLOW_RECIPE_CHANGE")
    )
    cli_allow_recipe = getattr(args, "allow_recipe_change", None)
    eff_allow_recipe_change = (
        bool(cli_allow_recipe)
        if cli_allow_recipe is not None
        else bool(env_allow_recipe if env_allow_recipe is not None else False)
    )

    resolved_cfg = V3PPOConfig.from_dict(cfg_dict)
    resolved_recipe_overrides = recipe_overrides_from_default(resolved_cfg)

    return EffectiveTrainJobConfig(
        cfg=resolved_cfg,
        output_dir=Path(args.output_dir),
        work_dir=Path(args.work_dir) if args.work_dir is not None else None,
        device=eff_device,
        input_dir=Path(args.input_dir) if args.input_dir is not None else None,
        ledger_path=Path(args.ledger) if args.ledger is not None else None,
        max_wall_seconds=eff_max_wall_seconds,
        max_session_export_bytes=eff_max_export_bytes,
        recipe_overrides=resolved_recipe_overrides,
        allow_recipe_change=eff_allow_recipe_change,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; exits non-zero whenever terminal ``status != 'PASS'``."""
    args = build_arg_parser().parse_args(argv)
    eff = resolve_effective_train_config(args)
    res = run_training_job(
        cfg=eff.cfg,
        output_dir=eff.output_dir,
        work_dir=eff.work_dir,
        device=eff.device,
        input_dir=eff.input_dir,
        ledger_path=eff.ledger_path,
        max_wall_seconds=eff.max_wall_seconds,
        max_session_export_bytes=eff.max_session_export_bytes,
        allow_recipe_change=eff.allow_recipe_change,
    )
    if res.get("status") != "PASS":
        return 1
    return 0
