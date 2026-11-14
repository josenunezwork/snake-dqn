"""Prepare Game V3.1 PPO Colab CLI packets (`canary`, `ruler`, `train`, and `eval` modes)."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

# Shared Colab packet/artifact helpers imported from `research.arena_ppo` (kept in one place
# until a shared cross-package packet module is extracted in a subsequent refactor round).
from research.arena_ppo.prepare import (
    CANARY_BASE_MINUTES,
    CANARY_TIMEOUT_MINUTES,
    CANARY_WALL_GUARD_SECONDS,
    GPU_FORECAST_RATES_UPH,
    RULER_BASE_MINUTES,
    RULER_TIMEOUT_MINUTES,
    RULER_WALL_GUARD_SECONDS,
    SUPPORTED_GPUS,
    _build_canary_receipt_binding,
    _write_json_create_only,
    re_safe_attempt,
)
from research.arena_ppo.runtime import sha256_file, verify_completed_artifact
from research.colab_cli import canary as colab_canary
from research.colab_cli import policy as colab_policy
from research.colab_cli import transfer_policy as colab_transfer_policy
from research.colab_cli import transfers as colab_transfers
from research.game_v3.config import GameV3Config
from research.v3_ppo.anchor_eval import (
    DEFAULT_ANCHOR_BURNIN_TICKS,
    DEFAULT_ANCHOR_LEARNER_SLOTS,
    DEFAULT_ANCHOR_NUM_ENVS,
    DEFAULT_ANCHOR_SCRIPTED_OPPONENTS,
    DEFAULT_ANCHOR_WINDOW_TICKS,
    build_anchor_candidate_reuse_key,
    candidate_unit_id,
    parse_anchor_scripted_opponents,
    validate_anchor_candidate_unit_reuse,
)
from research.v3_ppo.arena_eval import (
    DEFAULT_EVAL_SEED,
    DEFAULT_H2H_BURNIN_TICKS,
    DEFAULT_H2H_MASS_BIN_TICKS,
    DEFAULT_H2H_NUM_ENVS,
    DEFAULT_H2H_WINDOW_TICKS,
    DEFAULT_LONG_GAME_BURNIN_TICKS,
    DEFAULT_LONG_GAME_NUM_ENVS,
    DEFAULT_LONG_GAME_WINDOW_TICKS,
    DEFAULT_LONG_GAME_WORLD_AGE_BINS,
    EVAL_MODES,
    EVAL_REQUIRED_EXPORTS,
    V3_EVAL_POLICY_BUNDLE_SCHEMA,
    _load_checkpoint_payload,
    build_pair_reuse_key,
    discover_eval_checkpoints,
    discover_round_robin_checkpoints,
    estimate_eval_memory,
    extract_eval_policy_bundle_record,
    pack_eval_policy_bundle_records,
    resolve_unique_labels,
    select_pairs_per_batch,
    validate_pair_unit_reuse,
    write_eval_policy_bundle,
    write_weights_only_eval_bundle,
)
from research.v3_ppo.config import (
    ASSUMED_STAGED_INPUT_UPLOAD_RATE_BYTES_PER_SEC,
    ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS,
    CANARY_MAX_ESTIMATED_UPLOAD_SECONDS,
    DEFAULT_TRAIN_EVAL_OPPONENTS,
    EVAL_UPLOAD_CHUNK_RATE_BYTES_PER_SEC,
    EVAL_UPLOAD_CONTROL_HANG_PROB,
    GPU_VRAM_BUDGET_BYTES,
    MAX_SUPERVISOR_EXPORT_BYTES,
    OBS_BATCH_BYTES_PER_ROW,
    RECIPE_OVERRIDE_KEYS,
    V3_PPO_NAMESPACE,
    VALID_FULL_EXPORT_MODES,
    VALID_NET_SIZES,
    VALID_OPPONENT_POLICIES,
    FullExportMode,
    NetSize,
    OpponentPolicy,
    V3PPOConfig,
    assert_resume_config_compatible,
    build_recipe_change_record,
    check_minibatch_divisibility,
    default_train_envs_for_gpu,
    estimate_eval_upload_time,
    estimate_sim_memory_bytes,
    estimate_train_wall_headroom,
    estimate_train_wall_headroom_by_mode,
    estimate_training_memory,
    format_minibatch_remainder_warning,
    format_recipe_overrides,
    get_net_preset,
    merge_and_append_recipe_history,
    obs_layout_from_game_config,
    parse_opponent_policies,
    parse_recipe_overrides,
    resolve_full_export_settings,
    validate_checkpoint_export_caps,
    validate_memory_budget,
    validate_train_wall_headroom,
)
from research.v3_ppo.exports import write_create_only_artifact
from research.v3_ppo.job import resolve_effective_train_config
from research.v3_ppo.resume import resolve_resume_checkpoint
from research.v3_ppo.ruler import (
    CANARY_REQUIRED_EXPORTS,
    DEFAULT_CANARY_EVAL_TIMED_TICKS,
    DEFAULT_RULER_ENVS,
    DEFAULT_RULER_LEARNERS,
    DEFAULT_RULER_NETS,
    DEFAULT_RULER_ROLLOUT_LEN,
    DEFAULT_RULER_TIMED_UPDATES,
    DEFAULT_RULER_WARMUP_UPDATES,
    RULER_REQUIRED_EXPORTS,
    TRAIN_REQUIRED_EXPORTS,
    _parse_int_csv,
    _parse_nets_csv,
)

ROOT = Path(__file__).resolve().parents[2]
FAMILY_SCRIPT = "research/v3_ppo/job.sh"
FAMILY_PACKAGE = "research/v3_ppo"
REQUIREMENTS_RELPATH = "research/v3_ppo/requirements.txt"

PREPARE_MODES: tuple[str, ...] = ("canary", "ruler", "train", "eval")
UNLIMITED_HOURS_TOTAL_UPDATES: int = 1_000_000_000
EVAL_DEFAULT_SESSION_MINUTES: float = 60.0
EVAL_TIMING_SAFETY_FACTOR: float = 1.5
EVAL_RAW_UPLOAD_AUTO_STRIP_THRESHOLD_BYTES: int = 150 * 1024 * 1024


@dataclass(frozen=True)
class PreparePacketOptions:
    """Single options specification driving all stages of ``prepare_v3_ppo_packet``."""

    packet_dir: Path
    commit: str = "HEAD"
    mode: str = "canary"
    gpu: str = "L4"
    hours: float = 2.0
    expected_steps_per_sec: float | None = None
    net_size: NetSize | None = None
    num_envs: int | None = None
    num_snakes: int | None = None
    arena_radius: float | None = None
    learner_slots: int | None = None
    opponent_policy: OpponentPolicy | None = None
    rollout_len: int | None = None
    ruler_envs: str | None = None
    ruler_nets: str | None = None
    ruler_learners: str | None = None
    ruler_rollout_len: int = DEFAULT_RULER_ROLLOUT_LEN
    ruler_warmup_updates: int = DEFAULT_RULER_WARMUP_UPDATES
    ruler_timed_updates: int = DEFAULT_RULER_TIMED_UPDATES
    ruler_base_minutes: float | None = None
    amp: bool | None = None
    use_channels_last: bool | None = None
    fast_learner_ops: bool | None = None
    include_fp32: bool = False
    compile_model: bool = False
    full_exports: FullExportMode | None = None
    max_periodic_full_exports: int | None = None
    light_resume_margin_minutes: float | None = None
    eval_estimated_seconds: float | None = None
    eval_every_minutes: float | None = None
    eval_every_updates: int | None = None
    eval_envs: int | None = None
    eval_ticks: int | None = None
    eval_opponents: str | None = None
    eval_mode: str | None = None
    eval_policies: tuple[Path, ...] = ()
    eval_round_robin: tuple[Path, ...] = ()
    eval_candidates: tuple[Path, ...] = ()
    eval_references: tuple[Path, ...] = ()
    eval_labels: tuple[str, ...] | None = None
    eval_reference_labels: tuple[str, ...] | None = None
    anchor_scripted: str | None = None
    eval_burnin_ticks: int | None = None
    eval_window_ticks: int | None = None
    eval_mass_bin_ticks: int = DEFAULT_H2H_MASS_BIN_TICKS
    eval_world_age_bins: int = DEFAULT_LONG_GAME_WORLD_AGE_BINS
    eval_seed: int = DEFAULT_EVAL_SEED
    eval_greedy: bool = False
    eval_sequential_pairs: bool = False
    eval_max_pairs_per_batch: int | None = None
    assume_ms_per_tick: float | None = None
    eval_session_minutes: float = EVAL_DEFAULT_SESSION_MINUTES
    eval_weights_only: str = "true"  # "true" | "auto" | "false"
    output_dir: Path | None = None
    prior_output_dir: Path | None = None
    prior_kind: str = "auto"
    canary_receipt: Path | None = None
    allow_missing_canary_receipt: bool = False
    ppo_config: V3PPOConfig | None = None
    recipe_json: str | Path | None = None
    recipe_overrides: Mapping[str, Any] | None = None
    allow_recipe_change: bool = False
    explicit_total_updates: bool = False
    explicit_eval_estimated_seconds: bool = False
    measured_packed_bytes_per_world: int | None = None
    measured_checkpoint_bytes: int | None = None
    attempt: str = "attempt-01"
    allow_unmerged_commit: bool = False


@dataclass(frozen=True)
class _PriorSpec:
    """Resolved prior continuation artifact (checkpoint, ruler cells, eval pairs, or anchor candidates)."""

    kind: str  # "none" | "checkpoint" | "ruler_cells" | "eval_pairs" | "eval_candidates"
    prior_dir: Path | None = None
    chosen_checkpoint: dict[str, Any] | None = None
    staged_base_cfg: V3PPOConfig | None = None
    ruler_cell_paths: tuple[Path, ...] = ()
    eval_pair_paths: tuple[Path, ...] = ()
    eval_candidate_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _ValidatedPacketContext:
    """Validated packet metadata produced by ``_stage_validate_options``."""

    pkt_dir: Path
    out_path: Path
    gpu_norm: str
    gpu_slug: str
    resolved_commit: str
    source_paths: list[str]
    prior: _PriorSpec
    canary_block: dict[str, Any] | None
    canary_eval_metrics: dict[str, Any] | None
    canary_eval_smoke: dict[str, Any] | None = None


@dataclass(frozen=True)
class _SizedModePlan:
    """Mode-specific configuration, timing, and validated VRAM budget."""

    job_id: str
    est_minutes: float
    timeout_minutes: float
    base_sec: float
    eff_sec: float
    max_wall_sec_int: int
    extra_env: dict[str, str]
    required_exports: list[str]
    resume_fields: dict[str, Any]
    canary_block: dict[str, Any] | None
    train_cfg: V3PPOConfig | None = None
    training_plan: dict[str, Any] | None = None
    memory_estimate: dict[str, Any] | None = None
    wall_headroom: dict[str, Any] | None = None
    wall_headroom_by_mode: dict[FullExportMode, dict[str, Any]] | None = None
    recipe_overrides: dict[str, Any] | None = None
    recipe_change: dict[str, Any] | None = None
    minibatch_remainder: dict[str, Any] | None = None
    eval_plan: dict[str, Any] | None = None
    eval_policy_specs: tuple[dict[str, Any], ...] = ()
    anchor_ledger_spec: dict[str, Any] | None = None
    anchor_capacity: dict[str, Any] | None = None


@dataclass(frozen=True)
class _CappedModePlan:
    """Mode plan augmented with validated per-file and session checkpoint export caps."""

    sized: _SizedModePlan
    extra_env: dict[str, str]
    max_export_bytes: int
    checkpoint_export_caps: dict[str, Any] | None = None


@dataclass(frozen=True)
class _StagedInputs:
    """Staged input files, ledger, and post-staging per-file export cap."""

    inputs_list: list[dict[str, str]]
    ledger: dict[str, Any]
    max_export_bytes: int


def runtime_source_paths(commit: str) -> list[str]:
    """Return minimal hash-bound source inventory shipped to the Colab CLI runtime.

    Ships ONLY ``research/arena_ppo/{__init__,runtime}.py``, ``research/game_v3``, and
    ``research/v3_ppo`` — zero modules from the legacy grid simulator or observation layout.
    """
    arena_runtime_names = {
        "__init__.py",
        "runtime.py",
    }
    game_v3_names = {
        "__init__.py",
        "config.py",
        "rng.py",
        "sim.py",
        "reference.py",
        "obs.py",
        "scripted.py",
    }
    v3_ppo_names = {
        "__init__.py",
        "anchor_eval.py",
        "arena_eval.py",
        "buffer.py",
        "checkpoint_io.py",
        "config.py",
        "eval.py",
        "exports.py",
        "job.py",
        "job.sh",
        "net.py",
        "ppo.py",
        "requirements.txt",
        "resume.py",
        "ruler.py",
        "scripted_opponents.py",
        "trainer.py",
    }
    tracked = sorted(
        set(
            subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", commit, "research"],
                cwd=ROOT,
                text=True,
            ).splitlines()
        )
    )
    selected = [
        name
        for name in tracked
        if (
            name == "research/__init__.py"
            or (
                Path(name).parent.as_posix() == "research/arena_ppo"
                and Path(name).name in arena_runtime_names
            )
            or (
                Path(name).parent.as_posix() == "research/game_v3"
                and Path(name).name in game_v3_names
            )
            or (
                Path(name).parent.as_posix() == "research/v3_ppo"
                and Path(name).name in v3_ppo_names
            )
        )
    ]
    core_v3_ppo_names = v3_ppo_names - {
        "anchor_eval.py",
        "arena_eval.py",
        "scripted_opponents.py",
    }
    required_relpaths = (
        "research/arena_ppo/__init__.py",
        "research/arena_ppo/runtime.py",
        *(f"research/game_v3/{req}" for req in sorted(game_v3_names)),
        *(f"research/v3_ppo/{req}" for req in sorted(core_v3_ppo_names)),
    )
    missing_core = sorted(req for req in required_relpaths if req not in selected)
    if missing_core:
        raise ValueError(
            f"Pinned commit {commit} is missing required v3_ppo runtime files: {missing_core}"
        )
    return selected


def build_training_plan(
    cfg: V3PPOConfig,
    *,
    hours: float,
    expected_steps_per_sec: float,
    max_wall_seconds: float,
) -> dict[str, Any]:
    """Compute forecasted update rate and horizon for a ``train`` session."""
    agent_steps_per_update = int(cfg.rollout_len) * int(cfg.num_envs) * int(cfg.num_snakes)
    learner_steps_per_update = int(cfg.rollout_len) * int(cfg.total_learners)
    sec_per_update = float(agent_steps_per_update) / float(expected_steps_per_sec)
    expected_updates = int(max(1.0, math.floor(float(max_wall_seconds) / sec_per_update)))
    return {
        "hours": float(hours),
        "max_wall_seconds": float(max_wall_seconds),
        "expected_agent_steps_per_sec": float(expected_steps_per_sec),
        "agent_steps_per_update": agent_steps_per_update,
        "learner_steps_per_update": learner_steps_per_update,
        "expected_seconds_per_update": round(sec_per_update, 3),
        "expected_updates_in_session": expected_updates,
        "expected_agent_steps_in_session": expected_updates * agent_steps_per_update,
    }


def _check_foreign_checkpoints(ckpt_sub: Path) -> None:
    """Raise ``ValueError`` if completed checkpoints in ``ckpt_sub`` have a foreign namespace."""
    for idx_cand in sorted(ckpt_sub.glob("*.index.json")):
        if Path(f"{idx_cand}.complete").is_file() and Path(f"{idx_cand}.sha256").is_file():
            verify_completed_artifact(idx_cand)
            obj = json.loads(idx_cand.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj.get("namespace") != V3_PPO_NAMESPACE:
                raise ValueError(f"Foreign checkpoint namespace at {idx_cand}")
    for pt_cand in sorted(ckpt_sub.glob("*.pt")):
        if Path(f"{pt_cand}.complete").is_file() and Path(f"{pt_cand}.sha256").is_file():
            verify_completed_artifact(pt_cand)
            obj = torch.load(pt_cand, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and obj.get("namespace") != V3_PPO_NAMESPACE:
                raise ValueError(f"Foreign checkpoint schema/namespace at {pt_cand}")


def _resolve_verified_prior_checkpoint(
    prior_dir: Path,
    *,
    light_resume_margin_minutes: float = 45.0,
) -> dict[str, Any] | None:
    """Locate newest verified V3 PPO checkpoint set (preferring LIGHT only past margin)."""
    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

    for root_cand in search_roots:
        ckpt_sub = root_cand / "checkpoints"
        if not ckpt_sub.is_dir():
            continue
        _check_foreign_checkpoints(ckpt_sub)
        resolved = resolve_resume_checkpoint(
            ckpt_sub,
            require_markers=True,
            light_resume_margin_minutes=light_resume_margin_minutes,
        )
        if resolved is not None:
            return resolved
    return None


def _find_verified_prior_cells(prior_dir: Path) -> list[Path]:
    """Locate verified ``cells/*.json`` ruler artifacts under ``prior_dir``."""
    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

    for root_cand in search_roots:
        cells_sub = root_cand / "cells"
        if not cells_sub.is_dir():
            continue
        found: list[Path] = []
        for cell_json in sorted(cells_sub.glob("*.json")):
            if not (
                Path(f"{cell_json}.complete").is_file() and Path(f"{cell_json}.sha256").is_file()
            ):
                continue
            verify_completed_artifact(cell_json)
            parsed = json.loads(cell_json.read_text(encoding="utf-8"))
            if isinstance(parsed, dict) and parsed.get("namespace") not in (None, V3_PPO_NAMESPACE):
                raise ValueError(f"Foreign ruler cell namespace at {cell_json}")
            if isinstance(parsed, dict) and "error" in parsed:
                continue
            found.append(cell_json)
        if found:
            return found
    return []


def _find_verified_prior_eval_pairs(prior_dir: Path) -> list[Path]:
    """Locate verified ``pairs/pair_*.json`` artifacts under ``prior_dir``."""
    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

    for root_cand in search_roots:
        pairs_sub = root_cand / "pairs"
        if not pairs_sub.is_dir():
            continue
        found: list[Path] = []
        for pair_json in sorted(pairs_sub.glob("pair_*.json")):
            if not (
                Path(f"{pair_json}.complete").is_file() and Path(f"{pair_json}.sha256").is_file()
            ):
                continue
            verify_completed_artifact(pair_json)
            parsed = json.loads(pair_json.read_text(encoding="utf-8"))
            match_ns = (
                parsed.get("match", {}).get("namespace") if isinstance(parsed, dict) else None
            )
            if match_ns not in (None, V3_PPO_NAMESPACE):
                raise ValueError(f"Foreign eval pair namespace at {pair_json}")
            found.append(pair_json)
        if found:
            return found
    return []


def _find_verified_prior_eval_candidates(prior_dir: Path) -> list[Path]:
    """Locate verified ``candidates/cand_*.json`` anchor artifacts under ``prior_dir``."""
    search_roots = [prior_dir]
    if (prior_dir / "exports").is_dir():
        search_roots.append(prior_dir / "exports")

    for root_cand in search_roots:
        cands_sub = root_cand / "candidates"
        if not cands_sub.is_dir():
            continue
        found: list[Path] = []
        for cand_json in sorted(cands_sub.glob("cand_*.json")):
            if not (
                Path(f"{cand_json}.complete").is_file() and Path(f"{cand_json}.sha256").is_file()
            ):
                continue
            verify_completed_artifact(cand_json)
            parsed = json.loads(cand_json.read_text(encoding="utf-8"))
            cand_ns = parsed.get("namespace") if isinstance(parsed, dict) else None
            if cand_ns not in (None, V3_PPO_NAMESPACE):
                raise ValueError(f"Foreign anchor candidate namespace at {cand_json}")
            found.append(cand_json)
        if found:
            return found
    return []


def _detect_prior_kind(prior_output_dir: Path) -> str:
    prior_dir = Path(prior_output_dir).resolve()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_output_dir does not exist: {prior_dir}")
    has_ckpt = _resolve_verified_prior_checkpoint(prior_dir) is not None
    has_cells = bool(_find_verified_prior_cells(prior_dir))
    has_pairs = bool(_find_verified_prior_eval_pairs(prior_dir))
    has_cands = bool(_find_verified_prior_eval_candidates(prior_dir))
    if sum(bool(x) for x in (has_ckpt, has_cells, has_pairs, has_cands)) > 1:
        raise ValueError(
            f"Ambiguous prior_output_dir {prior_dir} contains multiple artifact kinds; "
            "pass explicit --prior-kind checkpoint, ruler_cells, eval_pairs, or eval_candidates."
        )
    if has_ckpt:
        return "checkpoint"
    if has_cells:
        return "ruler_cells"
    if has_pairs:
        return "eval_pairs"
    if has_cands:
        return "eval_candidates"
    raise FileNotFoundError(
        f"No verified V3 PPO checkpoints, ruler cells/*.json, eval pairs/*.json, or "
        f"candidates/*.json found under {prior_dir}"
    )


def _resolve_prior_spec(opts: PreparePacketOptions) -> _PriorSpec:
    """Resolve and verify ``prior_output_dir`` artifacts before staging or sizing."""
    if opts.prior_output_dir is None:
        return _PriorSpec(kind="none")

    if opts.prior_kind not in (
        "auto",
        "checkpoint",
        "ruler_cells",
        "eval_pairs",
        "eval_candidates",
    ):
        raise ValueError(f"Unsupported prior_kind={opts.prior_kind!r}")
    if opts.mode == "train" and opts.prior_kind not in ("auto", "checkpoint"):
        raise ValueError("mode='train' requires prior_kind in {'auto', 'checkpoint'}")
    if opts.mode == "ruler" and opts.prior_kind not in ("auto", "ruler_cells"):
        raise ValueError("mode='ruler' requires prior_kind in {'auto', 'ruler_cells'}")
    if opts.mode == "eval" and opts.prior_kind not in ("auto", "eval_pairs", "eval_candidates"):
        raise ValueError(
            "mode='eval' requires prior_kind in {'auto', 'eval_pairs', 'eval_candidates'}"
        )

    prior_dir = Path(opts.prior_output_dir).resolve()
    if not prior_dir.is_dir():
        raise FileNotFoundError(f"prior_output_dir does not exist: {prior_dir}")

    eff_prior_kind = opts.prior_kind
    if eff_prior_kind == "auto":
        if opts.mode == "train":
            eff_prior_kind = "checkpoint"
        elif opts.mode == "ruler":
            eff_prior_kind = "ruler_cells"
        elif opts.mode == "eval":
            if opts.eval_mode == "anchor" or bool(opts.eval_candidates):
                eff_prior_kind = "eval_candidates"
            else:
                eff_prior_kind = "eval_pairs"
        else:
            eff_prior_kind = _detect_prior_kind(prior_dir)

    if eff_prior_kind == "eval_candidates":
        found_cands = _find_verified_prior_eval_candidates(prior_dir)
        if not found_cands:
            raise FileNotFoundError(
                f"No verified V3 PPO anchor candidates/cand_*.json found under {prior_dir}"
            )
        return _PriorSpec(
            kind="eval_candidates",
            prior_dir=prior_dir,
            eval_candidate_paths=tuple(found_cands),
        )

    if eff_prior_kind == "eval_pairs":
        found_pairs = _find_verified_prior_eval_pairs(prior_dir)
        if not found_pairs:
            raise FileNotFoundError(f"No verified V3 PPO eval pairs/*.json found under {prior_dir}")
        return _PriorSpec(
            kind="eval_pairs",
            prior_dir=prior_dir,
            eval_pair_paths=tuple(found_pairs),
        )

    if eff_prior_kind == "ruler_cells":
        found_cells = _find_verified_prior_cells(prior_dir)
        if not found_cells:
            raise FileNotFoundError(
                f"No verified V3 PPO ruler cells/*.json found under {prior_dir}"
            )
        return _PriorSpec(
            kind="ruler_cells",
            prior_dir=prior_dir,
            ruler_cell_paths=tuple(found_cells),
        )

    eff_stage_margin = (
        float(opts.light_resume_margin_minutes)
        if opts.light_resume_margin_minutes is not None
        else (
            float(opts.ppo_config.light_resume_margin_minutes)
            if opts.ppo_config is not None
            else 45.0
        )
    )
    chosen = _resolve_verified_prior_checkpoint(
        prior_dir,
        light_resume_margin_minutes=eff_stage_margin,
    )
    if chosen is None:
        raise FileNotFoundError(f"No verified V3 PPO checkpoint found under {prior_dir}")
    staged_base_cfg = (
        V3PPOConfig.from_dict(chosen["config"]) if isinstance(chosen.get("config"), dict) else None
    )
    return _PriorSpec(
        kind="checkpoint",
        prior_dir=prior_dir,
        chosen_checkpoint=chosen,
        staged_base_cfg=staged_base_cfg,
    )


def _extract_canary_eval_metrics(canary_receipt: Path | str) -> dict[str, Any] | None:
    """Read measured default-eval wall seconds and combined peak bytes from a canary receipt."""
    rcpt_path = Path(canary_receipt).expanduser().resolve()
    if not rcpt_path.is_file():
        return None
    rcpt_obj = json.loads(rcpt_path.read_text(encoding="utf-8"))
    if not isinstance(rcpt_obj, dict):
        return None

    expected_export_shas: dict[str, str] = {}
    exports_list = rcpt_obj.get("exports")
    if isinstance(exports_list, list):
        for row in exports_list:
            if isinstance(row, dict) and isinstance(row.get("path"), str) and row.get("sha256"):
                expected_export_shas[row["path"]] = str(row["sha256"])

    checks_obj: dict[str, Any] | None = None
    for cand in (
        rcpt_path.parent / "exports" / "canary_checks.json",
        rcpt_path.parent / "canary_checks.json",
    ):
        if cand.is_file():
            exp_sha = expected_export_shas.get("canary_checks.json")
            if exp_sha is not None and sha256_file(cand) != exp_sha:
                raise ValueError(f"Canary export hash mismatch for {cand}: expected {exp_sha}")
            loaded = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                checks_obj = loaded
                break

    bench = (
        checks_obj.get("default_eval_benchmark")
        if isinstance(checks_obj, dict)
        and isinstance(checks_obj.get("default_eval_benchmark"), dict)
        else {}
    )
    raw_wall = (
        checks_obj.get("default_eval_wall_seconds")
        if isinstance(checks_obj, dict) and checks_obj.get("default_eval_wall_seconds") is not None
        else (
            bench.get("wall_seconds")
            if bench.get("wall_seconds") is not None
            else rcpt_obj.get("default_eval_wall_seconds")
        )
    )
    if (
        isinstance(raw_wall, bool)
        or not isinstance(raw_wall, (int, float))
        or not math.isfinite(float(raw_wall))
        or float(raw_wall) <= 0.0
    ):
        return None

    raw_peak = (
        checks_obj.get("default_eval_peak_memory_bytes")
        if isinstance(checks_obj, dict)
        and checks_obj.get("default_eval_peak_memory_bytes") is not None
        else (
            bench.get("peak_memory_bytes")
            if bench.get("peak_memory_bytes") is not None
            else rcpt_obj.get("default_eval_peak_memory_bytes")
        )
    )
    raw_res_envs = (
        checks_obj.get("default_eval_resident_train_envs")
        if isinstance(checks_obj, dict)
        and checks_obj.get("default_eval_resident_train_envs") is not None
        else bench.get("resident_train_envs")
    )
    out_metrics: dict[str, Any] = {"wall_seconds": float(raw_wall)}
    if not isinstance(raw_peak, bool) and isinstance(raw_peak, (int, float)) and int(raw_peak) > 0:
        out_metrics["peak_memory_bytes"] = int(raw_peak)
    if (
        not isinstance(raw_res_envs, bool)
        and isinstance(raw_res_envs, (int, float))
        and int(raw_res_envs) > 0
    ):
        out_metrics["resident_train_envs"] = int(raw_res_envs)
    return out_metrics


def _extract_canary_eval_smoke_metrics(canary_receipt: Path | str) -> dict[str, Any] | None:
    """Read hash-verified ``eval_canary_smoke`` timing/world metrics from a canary receipt."""
    rcpt_path = Path(canary_receipt).expanduser().resolve()
    if not rcpt_path.is_file():
        return None
    rcpt_obj = json.loads(rcpt_path.read_text(encoding="utf-8"))
    if not isinstance(rcpt_obj, dict):
        return None

    expected_export_shas: dict[str, str] = {}
    exports_list = rcpt_obj.get("exports")
    if isinstance(exports_list, list):
        for row in exports_list:
            if isinstance(row, dict) and isinstance(row.get("path"), str) and row.get("sha256"):
                expected_export_shas[row["path"]] = str(row["sha256"])

    checks_obj: dict[str, Any] | None = None
    for cand in (
        rcpt_path.parent / "exports" / "canary_checks.json",
        rcpt_path.parent / "canary_checks.json",
    ):
        if cand.is_file():
            exp_sha = expected_export_shas.get("canary_checks.json")
            if exp_sha is not None and sha256_file(cand) != exp_sha:
                raise ValueError(f"Canary export hash mismatch for {cand}: expected {exp_sha}")
            loaded = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                checks_obj = loaded
                break

    if not isinstance(checks_obj, dict):
        return None
    smoke = checks_obj.get("eval_canary_smoke")
    if not isinstance(smoke, dict):
        return None
    raw_ms = smoke.get("ms_per_tick")
    if (
        isinstance(raw_ms, bool)
        or not isinstance(raw_ms, (int, float))
        or not math.isfinite(float(raw_ms))
        or float(raw_ms) <= 0.0
    ):
        return None
    return dict(smoke)


def _stage_validate_options(opts: PreparePacketOptions) -> _ValidatedPacketContext:
    """Stage 1: Validate mode/GPU/commit/paths/canary-receipt and resolve prior artifacts."""
    if opts.mode not in PREPARE_MODES:
        raise ValueError(f"mode must be one of {PREPARE_MODES}, got {opts.mode!r}")
    gpu_norm = str(opts.gpu).strip().upper()
    if gpu_norm not in SUPPORTED_GPUS:
        raise ValueError(f"gpu must be one of {SUPPORTED_GPUS} (T4 unsupported), got {opts.gpu!r}")
    if not re_safe_attempt(opts.attempt):
        raise ValueError(f"Invalid attempt slug {opts.attempt!r}")
    if opts.net_size is not None:
        get_net_preset(opts.net_size)
    if opts.ruler_nets is not None:
        _parse_nets_csv(opts.ruler_nets)
    if opts.full_exports is not None or opts.max_periodic_full_exports is not None:
        resolve_full_export_settings(opts.full_exports, opts.max_periodic_full_exports)
    if (opts.recipe_json is not None or opts.recipe_overrides is not None) and opts.mode != "train":
        raise ValueError(f"--recipe-json is only supported in --mode train, got --mode {opts.mode}")
    if opts.allow_recipe_change and opts.mode != "train":
        raise ValueError(
            f"--allow-recipe-change is only supported in --mode train, got --mode {opts.mode}"
        )
    if opts.recipe_overrides is not None:
        parse_recipe_overrides(opts.recipe_overrides, source_label="recipe_overrides")
    if opts.recipe_json is not None:
        parse_recipe_overrides(opts.recipe_json, source_label="--recipe-json")
    if opts.mode == "canary" and opts.canary_receipt is not None:
        raise ValueError("canary mode produces a canary receipt and cannot bind --canary-receipt")
    if (
        opts.mode in ("ruler", "train", "eval")
        and opts.canary_receipt is None
        and not opts.allow_missing_canary_receipt
    ):
        raise ValueError(
            f"--mode {opts.mode} requires canary_receipt <path/to/receipt.json> "
            "(pass allow_missing_canary_receipt=True only for offline schema inspection)"
        )

    resolved_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{opts.commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    allowed_refs = ("HEAD", "origin/main") if opts.allow_unmerged_commit else ("origin/main",)
    if not any(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", resolved_commit, ref],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).returncode
        == 0
        for ref in allowed_refs
    ):
        raise ValueError(
            f"Refusing to prepare packet from unmerged commit {resolved_commit}: "
            f"commit is not an ancestor of {' or '.join(allowed_refs)}"
        )

    source_paths = runtime_source_paths(resolved_commit)
    has_eval_inputs = bool(opts.eval_policies or opts.eval_round_robin or opts.eval_candidates)
    if (
        opts.mode == "eval" or (opts.mode == "canary" and has_eval_inputs)
    ) and "research/v3_ppo/arena_eval.py" not in source_paths:
        raise ValueError(
            f"Pinned commit {resolved_commit} is missing research/v3_ppo/arena_eval.py "
            "required for --mode eval"
        )
    is_anchor_req = opts.eval_mode == "anchor" or bool(opts.eval_candidates)
    if is_anchor_req and "research/v3_ppo/anchor_eval.py" not in source_paths:
        raise ValueError(
            f"Pinned commit {resolved_commit} is missing research/v3_ppo/anchor_eval.py "
            "required for --eval-mode anchor"
        )
    wants_harder_scripted = (
        opts.opponent_policy in ("head_evader", "cut_off")
        or (
            opts.eval_opponents is not None
            and any(
                op in ("head_evader", "cut_off")
                for op in parse_opponent_policies(opts.eval_opponents)
            )
        )
        or (
            is_anchor_req
            and any(
                op in ("head_evader", "cut_off")
                for op in parse_anchor_scripted_opponents(opts.anchor_scripted)
            )
        )
    )
    if wants_harder_scripted and "research/v3_ppo/scripted_opponents.py" not in source_paths:
        raise ValueError(
            f"Pinned commit {resolved_commit} is missing research/v3_ppo/scripted_opponents.py "
            "required for head_evader / cut_off scripted opponents"
        )

    gpu_slug = gpu_norm.lower()
    raw_out_dir = opts.output_dir
    if raw_out_dir is None:
        raw_out_dir = (
            ROOT.parent
            / "snake-dqn-artifacts"
            / "big-lever"
            / "exp"
            / "argon-20261026"
            / f"v3-ppo-{opts.mode}-{gpu_slug}-{opts.attempt}"
            / "output"
        )
    out_path = Path(raw_out_dir).resolve()
    if out_path.exists():
        raise FileExistsError(f"output_dir must be a fresh non-existent directory: {out_path}")

    pkt_dir = Path(opts.packet_dir).resolve()
    if pkt_dir.exists():
        raise FileExistsError(f"packet_dir must be a fresh non-existent directory: {pkt_dir}")

    prior = _resolve_prior_spec(opts)
    if opts.allow_recipe_change and prior.kind != "checkpoint":
        raise ValueError(
            "--allow-recipe-change requires a prior checkpoint continuation "
            "(--prior-output-dir containing a verified checkpoint)"
        )

    canary_block: dict[str, Any] | None = None
    canary_eval_metrics: dict[str, Any] | None = None
    canary_eval_smoke: dict[str, Any] | None = None
    if opts.mode == "canary":
        canary_block = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": FAMILY_SCRIPT,
        }
    elif opts.canary_receipt is not None:
        canary_block = _build_canary_receipt_binding(opts.canary_receipt, gpu_norm)
        if opts.mode == "train":
            canary_eval_metrics = _extract_canary_eval_metrics(opts.canary_receipt)
        elif opts.mode == "eval":
            canary_eval_smoke = _extract_canary_eval_smoke_metrics(opts.canary_receipt)

    return _ValidatedPacketContext(
        pkt_dir=pkt_dir,
        out_path=out_path,
        gpu_norm=gpu_norm,
        gpu_slug=gpu_slug,
        resolved_commit=resolved_commit,
        source_paths=source_paths,
        prior=prior,
        canary_block=canary_block,
        canary_eval_metrics=canary_eval_metrics,
        canary_eval_smoke=canary_eval_smoke,
    )


def _size_canary_mode(
    opts: PreparePacketOptions,
    ctx: _ValidatedPacketContext,
) -> _SizedModePlan:
    est_minutes = CANARY_BASE_MINUTES
    timeout_minutes = CANARY_TIMEOUT_MINUTES
    base_sec = est_minutes * 60.0
    eff_sec = timeout_minutes * 60.0
    max_wall_sec_int = int(CANARY_WALL_GUARD_SECONDS)
    extra_env: dict[str, str] = {
        "V3_PPO_MAX_WALL_SECONDS": str(max_wall_sec_int),
        "V3_PPO_CANARY_TRAIN_ENVS": str(default_train_envs_for_gpu(ctx.gpu_norm)),
        "SNAKE_COLAB_CANARY": "1",
        "SNAKE_COLAB_CANARY_PACKAGE": FAMILY_PACKAGE,
    }
    if opts.compile_model:
        extra_env["V3_PPO_COMPILE"] = "1"
    if opts.fast_learner_ops is False:
        extra_env["V3_PPO_FAST_LEARNER_OPS"] = "0"

    required_exports = list(CANARY_REQUIRED_EXPORTS)
    eval_policy_specs: tuple[dict[str, Any], ...] = ()
    memory_estimate: dict[str, Any] | None = None
    anchor_ledger_spec: dict[str, Any] | None = None
    anchor_capacity: dict[str, Any] | None = None
    if opts.eval_policies or opts.eval_round_robin or opts.eval_candidates:
        layout = _resolve_and_validate_eval_layout(opts, ctx)
        policy_specs = layout["policy_specs"]
        use_weights_only = bool(layout["use_weights_only"])
        eff_eval_mode = str(layout["eff_eval_mode"])
        eff_amp = True if opts.amp is None else bool(opts.amp)
        eff_cl = True if opts.use_channels_last is None else bool(opts.use_channels_last)
        extra_env["V3_PPO_EVAL_MODE"] = eff_eval_mode
        extra_env["V3_PPO_EVAL_NUM_ENVS"] = str(int(layout["eff_envs"]))
        extra_env["V3_PPO_CANARY_EVAL_TICKS"] = str(int(DEFAULT_CANARY_EVAL_TIMED_TICKS))
        extra_env["V3_PPO_AMP"] = "1" if eff_amp else "0"
        extra_env["V3_PPO_CHANNELS_LAST"] = "1" if eff_cl else "0"
        if opts.learner_slots is not None:
            extra_env["V3_PPO_EVAL_LEARNER_SLOTS"] = str(int(opts.learner_slots))
        elif eff_eval_mode == "anchor":
            extra_env["V3_PPO_EVAL_LEARNER_SLOTS"] = str(int(layout["eff_learner_slots"]))
        if opts.opponent_policy is not None:
            extra_env["V3_PPO_EVAL_OPPONENT_POLICY"] = str(opts.opponent_policy)
        if opts.eval_greedy:
            extra_env["V3_PPO_EVAL_GREEDY"] = "1"
        if eff_eval_mode == "round-robin":
            if opts.eval_sequential_pairs:
                extra_env["V3_PPO_EVAL_SEQUENTIAL_PAIRS"] = "1"
            else:
                extra_env["V3_PPO_EVAL_MAX_PAIRS_PER_BATCH"] = str(
                    int(layout["eff_pairs_per_batch"])
                )
        elif eff_eval_mode == "anchor":
            anchor_ledger_spec = dict(layout["anchor_ledger_spec"])
            scripted_ops = anchor_ledger_spec["scripted_opponents"]
            extra_env["V3_PPO_ANCHOR_SCRIPTED"] = ",".join(scripted_ops) if scripted_ops else "none"
            anchor_capacity = _compute_anchor_capacity_summary(
                gpu_norm=ctx.gpu_norm,
                candidate_net_sizes=layout["candidate_net_sizes"],
                reference_net_sizes=layout["reference_net_sizes"],
                scripted_opponents=tuple(scripted_ops),
                eff_envs=int(layout["eff_envs"]),
                eff_learner_slots=int(layout["eff_learner_slots"]),
                eff_burnin=int(layout["eff_burnin"]),
                eff_window=int(layout["eff_window"]),
                n_snakes=int(layout["n_snakes"]),
                use_weights_only=use_weights_only,
                session_cap_min=float(opts.eval_session_minutes),
                measured_or_assumed_ms_per_tick=None,
                prior_unit_sizes=[int(p.stat().st_size) for p in ctx.prior.eval_candidate_paths],
                is_canary=True,
            )
            _print_anchor_capacity_stderr(anchor_capacity)
        required_exports = [*CANARY_REQUIRED_EXPORTS, *EVAL_REQUIRED_EXPORTS]
        eval_policy_specs = tuple({**sp, "weights_only": use_weights_only} for sp in policy_specs)
        memory_estimate = dict(layout["memory_estimate"])

    return _SizedModePlan(
        job_id=f"v3-ppo-canary-{ctx.gpu_slug}-{opts.attempt}",
        est_minutes=est_minutes,
        timeout_minutes=timeout_minutes,
        base_sec=base_sec,
        eff_sec=eff_sec,
        max_wall_sec_int=max_wall_sec_int,
        extra_env=extra_env,
        required_exports=required_exports,
        resume_fields={
            "resume_exemption": {
                "kind": "short-canary",
                "reason": (
                    f"Game V3.1 PPO {ctx.gpu_norm} family canary probe (15 min base / 30 min "
                    "timeout): tiny S/M ruler + 2-update FULL-resume + 1 default eval"
                ),
            }
        },
        canary_block=ctx.canary_block,
        memory_estimate=memory_estimate,
        eval_policy_specs=eval_policy_specs,
        anchor_ledger_spec=anchor_ledger_spec,
        anchor_capacity=anchor_capacity,
    )


def _size_ruler_mode(
    opts: PreparePacketOptions,
    ctx: _ValidatedPacketContext,
) -> _SizedModePlan:
    if opts.ruler_base_minutes is not None:
        if not math.isfinite(opts.ruler_base_minutes) or not (
            3.0 <= float(opts.ruler_base_minutes) <= 30.0
        ):
            raise ValueError(
                f"--ruler-base-minutes must be in [3.0, 30.0], got {opts.ruler_base_minutes}"
            )
        est_minutes = round(float(opts.ruler_base_minutes), 2)
        timeout_minutes = round(2.0 * est_minutes, 2)
        headroom_min = min(8.0, max(2.0, 0.2 * timeout_minutes))
        wall_guard_sec = int(max(60.0, (timeout_minutes - headroom_min) * 60.0))
    else:
        est_minutes = RULER_BASE_MINUTES
        timeout_minutes = RULER_TIMEOUT_MINUTES
        wall_guard_sec = int(RULER_WALL_GUARD_SECONDS)

    parsed_envs = (
        list(DEFAULT_RULER_ENVS) if opts.ruler_envs is None else _parse_int_csv(opts.ruler_envs)
    )
    parsed_nets = (
        list(DEFAULT_RULER_NETS) if opts.ruler_nets is None else _parse_nets_csv(opts.ruler_nets)
    )
    parsed_learners = (
        list(DEFAULT_RULER_LEARNERS)
        if opts.ruler_learners is None
        else _parse_int_csv(opts.ruler_learners)
    )
    eff_r_snakes = 24 if opts.num_snakes is None else int(opts.num_snakes)
    eff_r_radius = 600.0 if opts.arena_radius is None else float(opts.arena_radius)
    eff_r_rollout = int(opts.ruler_rollout_len)
    eff_r_warmup = int(opts.ruler_warmup_updates)
    eff_r_timed = int(opts.ruler_timed_updates)
    if eff_r_rollout < 1:
        raise ValueError(f"--ruler-rollout-len must be >= 1, got {eff_r_rollout}")
    if eff_r_warmup < 0:
        raise ValueError(f"--ruler-warmup-updates must be >= 0, got {eff_r_warmup}")
    if eff_r_timed < 1:
        raise ValueError(f"--ruler-timed-updates must be >= 1, got {eff_r_timed}")

    eff_r_amp = True if opts.amp is None else bool(opts.amp)
    eff_r_cl = True if opts.use_channels_last is None else bool(opts.use_channels_last)
    eff_r_fast = True if opts.fast_learner_ops is None else bool(opts.fast_learner_ops)

    if any(k < 1 or k > eff_r_snakes for k in parsed_learners):
        raise ValueError(f"ruler learners {parsed_learners} must be within [1, {eff_r_snakes}]")

    ruler_game_cfg = GameV3Config(
        arena_radius=eff_r_radius,
        num_snakes=eff_r_snakes,
        max_life_ticks=0,
    )
    budget_bytes = GPU_VRAM_BUDGET_BYTES[ctx.gpu_norm]
    fitting_cells: list[str] = []
    smallest_est_bytes: int | None = None
    for e_val in parsed_envs:
        for n_sz in parsed_nets:
            for k_val in parsed_learners:
                cell_cfg = V3PPOConfig(
                    game_config=ruler_game_cfg,
                    num_envs=int(e_val),
                    learner_slots=int(k_val),
                    net_size=n_sz,
                    amp=eff_r_amp,
                    use_channels_last=eff_r_cl,
                    fast_learner_ops=eff_r_fast,
                    rollout_len=eff_r_rollout,
                    eval_every_minutes=0.0,
                    eval_every_updates=0,
                )
                est = estimate_training_memory(cell_cfg)
                tot_b = int(est["estimated_peak_bytes"])
                if smallest_est_bytes is None or tot_b < smallest_est_bytes:
                    smallest_est_bytes = tot_b
                if tot_b <= budget_bytes:
                    fitting_cells.append(f"E{e_val}_{n_sz}_L{k_val}")

    if not fitting_cells:
        raise ValueError(
            f"Refusing ruler packet on {ctx.gpu_norm}: zero requested ruler cells fit within "
            f"{budget_bytes / 1e9:.2f} GB budget (smallest cell estimated at "
            f"{(smallest_est_bytes or 0) / 1e9:.2f} GB)."
        )
    memory_estimate = {
        "gpu": ctx.gpu_norm,
        "budget_bytes": budget_bytes,
        "fitting_cells": fitting_cells,
        "total_requested_cells": len(parsed_envs) * len(parsed_nets) * len(parsed_learners),
    }

    extra_env: dict[str, str] = {"V3_PPO_MAX_WALL_SECONDS": str(wall_guard_sec)}
    if opts.ruler_envs is not None:
        extra_env["V3_PPO_RULER_ENVS"] = ",".join(str(v) for v in parsed_envs)
    if opts.ruler_nets is not None:
        extra_env["V3_PPO_RULER_NETS"] = ",".join(parsed_nets)
    if opts.ruler_learners is not None:
        extra_env["V3_PPO_RULER_LEARNERS"] = ",".join(str(v) for v in parsed_learners)
    if opts.arena_radius is not None:
        extra_env["V3_PPO_ARENA_RADIUS"] = str(eff_r_radius)
    if opts.num_snakes is not None:
        extra_env["V3_PPO_NUM_SNAKES"] = str(eff_r_snakes)
    if eff_r_rollout != DEFAULT_RULER_ROLLOUT_LEN:
        extra_env["V3_PPO_RULER_ROLLOUT_LEN"] = str(eff_r_rollout)
    if eff_r_warmup != DEFAULT_RULER_WARMUP_UPDATES:
        extra_env["V3_PPO_RULER_WARMUP_UPDATES"] = str(eff_r_warmup)
    if eff_r_timed != DEFAULT_RULER_TIMED_UPDATES:
        extra_env["V3_PPO_RULER_TIMED_UPDATES"] = str(eff_r_timed)
    extra_env["V3_PPO_AMP"] = "1" if eff_r_amp else "0"
    extra_env["V3_PPO_CHANNELS_LAST"] = "1" if eff_r_cl else "0"
    if opts.include_fp32:
        extra_env["V3_PPO_INCLUDE_FP32"] = "1"
    if opts.compile_model:
        extra_env["V3_PPO_COMPILE"] = "1"
    if not eff_r_fast:
        extra_env["V3_PPO_FAST_LEARNER_OPS"] = "0"

    return _SizedModePlan(
        job_id=f"v3-ppo-ruler-{ctx.gpu_slug}-{opts.attempt}",
        est_minutes=est_minutes,
        timeout_minutes=timeout_minutes,
        base_sec=est_minutes * 60.0,
        eff_sec=timeout_minutes * 60.0,
        max_wall_sec_int=wall_guard_sec,
        extra_env=extra_env,
        required_exports=list(RULER_REQUIRED_EXPORTS),
        resume_fields={
            "resume_exemption": {
                "kind": "acceptance",
                "reason": (
                    f"Game V3.1 PPO {ctx.gpu_norm} throughput, stage-split & VRAM ruler "
                    "(<=60m runaway guard; per-cell create-only JSONs + signal-safe result)"
                ),
            }
        },
        canary_block=ctx.canary_block,
        memory_estimate=memory_estimate,
    )


def _resolve_measured_world_bytes(
    opts: PreparePacketOptions,
    cfg_obj: V3PPOConfig,
) -> int | None:
    """Resolve optional per-world packed checkpoint byte measurement from ``opts``."""
    eff_world_bytes = opts.measured_packed_bytes_per_world
    if eff_world_bytes is None and opts.measured_checkpoint_bytes is not None:
        if int(opts.measured_checkpoint_bytes) < 1:
            raise ValueError(
                f"measured_checkpoint_bytes must be >= 1, got {opts.measured_checkpoint_bytes}"
            )
        eff_world_bytes = max(
            1, int(math.ceil(int(opts.measured_checkpoint_bytes) / cfg_obj.num_envs))
        )
    return eff_world_bytes


def _train_eval_cost_scale_vs_default(cfg_obj: V3PPOConfig) -> float:
    """Return exact 1.0 for default train eval settings, else proportional wall-time multiplier."""
    if (
        int(cfg_obj.eval_envs) == 192
        and int(cfg_obj.eval_burnin_ticks) == 250
        and int(cfg_obj.eval_ticks) == 2000
        and tuple(cfg_obj.eval_opponents) == DEFAULT_TRAIN_EVAL_OPPONENTS
    ):
        return 1.0
    env_ratio = float(cfg_obj.eval_envs) / 192.0
    tick_ratio = float(int(cfg_obj.eval_burnin_ticks) + int(cfg_obj.eval_ticks)) / 2250.0
    opp_weights = [
        1.15 if op in ("cut_off", "head_evader") else 1.0 for op in cfg_obj.eval_opponents
    ]
    opp_ratio = float(sum(opp_weights)) / 2.0
    return max(0.05, env_ratio * tick_ratio * opp_ratio)


def _scale_train_canary_eval_wall_seconds(
    cfg_obj: V3PPOConfig,
    raw_canary_eval_wall: float | None,
) -> float | None:
    """Scale measured default-eval canary wall time when train eval settings are customized."""
    if raw_canary_eval_wall is None:
        return None
    scale = _train_eval_cost_scale_vs_default(cfg_obj)
    if scale == 1.0:
        return float(raw_canary_eval_wall)
    return round(float(raw_canary_eval_wall) * scale, 4)


def _size_train_mode(
    opts: PreparePacketOptions,
    ctx: _ValidatedPacketContext,
) -> _SizedModePlan:
    if not math.isfinite(opts.hours) or opts.hours <= 0:
        raise ValueError(f"--hours must be finite and positive, got {opts.hours}")
    timeout_minutes = round(float(opts.hours) * 60.0, 2)
    if timeout_minutes < 6.0:
        raise ValueError(f"--hours results in timeout_minutes={timeout_minutes} < 6.0 min")
    est_minutes = round(timeout_minutes / 2.0, 2)
    base_sec = est_minutes * 60.0
    eff_sec = timeout_minutes * 60.0

    staged_base_cfg = ctx.prior.staged_base_cfg
    base_cfg = opts.ppo_config or staged_base_cfg or V3PPOConfig()
    # Only an explicit `--full-exports` or `--max-periodic-full-exports` flag selects a non-`none`
    # export mode; config dumps (`--ppo-config-json` / prior checkpoint config) never silently
    # opt into `periodic` or `final`.
    default_full_mode: FullExportMode = "none"

    default_eval_est = float(V3PPOConfig().eval_estimated_seconds)
    caller_overrode_eval_est = (
        opts.eval_estimated_seconds is not None
        or bool(opts.explicit_eval_estimated_seconds)
        or (
            opts.ppo_config is not None
            and float(opts.ppo_config.eval_estimated_seconds) != default_eval_est
        )
    )
    eff_eval_est: float | None = None
    if opts.eval_estimated_seconds is not None:
        eff_eval_est = float(opts.eval_estimated_seconds)
    elif not caller_overrode_eval_est and ctx.canary_eval_metrics is not None:
        eff_eval_est = float(ctx.canary_eval_metrics["wall_seconds"])

    eff_train_eval_ticks = (
        opts.eval_ticks if opts.eval_ticks is not None else opts.eval_window_ticks
    )

    eff_train = resolve_effective_train_config(
        argparse.Namespace(
            output_dir=ctx.out_path,
            work_dir=None,
            device="cuda",
            input_dir=None,
            ledger=None,
            config_json=None,
            recipe_json=opts.recipe_json,
            num_envs=opts.num_envs,
            num_snakes=opts.num_snakes,
            arena_radius=opts.arena_radius,
            learner_slots=opts.learner_slots,
            opponent_policy=opts.opponent_policy,
            net_size=opts.net_size,
            rollout_len=opts.rollout_len,
            total_updates=None,
            ppo_epochs=None,
            minibatch_size=None,
            lr=None,
            light_checkpoint_interval_sec=None,
            full_checkpoint_interval_sec=None,
            full_exports=opts.full_exports,
            max_periodic_full_exports=opts.max_periodic_full_exports,
            light_resume_margin_minutes=opts.light_resume_margin_minutes,
            checkpoint_shard_bytes=None,
            checkpoint_every_updates=None,
            full_checkpoint_every_updates=None,
            eval_every_minutes=opts.eval_every_minutes,
            eval_every_updates=opts.eval_every_updates,
            eval_envs=opts.eval_envs,
            eval_burnin_ticks=opts.eval_burnin_ticks,
            eval_ticks=eff_train_eval_ticks,
            eval_opponents=opts.eval_opponents,
            eval_estimated_seconds=eff_eval_est,
            seed=None,
            compile=(True if opts.compile_model else None),
            fast_learner_ops=opts.fast_learner_ops,
            amp=opts.amp,
            use_channels_last=opts.use_channels_last,
            max_wall_seconds=None,
            max_session_export_bytes=None,
        ),
        environ={},
        base_cfg=base_cfg,
        recipe_overrides=opts.recipe_overrides,
        default_full_exports=default_full_mode,
        forbid_recipe_scalar_conflict=True,
    )
    cfg_obj = eff_train.cfg
    eval_scale = _train_eval_cost_scale_vs_default(cfg_obj)
    if eval_scale != 1.0 and not caller_overrode_eval_est:
        cfg_obj = replace(
            cfg_obj,
            eval_estimated_seconds=round(float(cfg_obj.eval_estimated_seconds) * eval_scale, 2),
        )

    recipe_change_rec: dict[str, Any] | None = None
    if staged_base_cfg is not None:
        assert_resume_config_compatible(
            staged_base_cfg,
            cfg_obj,
            allow_recipe_change=bool(opts.allow_recipe_change),
            context="Continuation checkpoint config mismatch",
            reference_label="ckpt",
            candidate_label="requested",
        )
        chosen_u = int((ctx.prior.chosen_checkpoint or {}).get("updates_completed", 0))
        recipe_change_rec = (
            build_recipe_change_record(
                update=chosen_u,
                reference=staged_base_cfg,
                candidate=cfg_obj,
            )
            if opts.allow_recipe_change
            else None
        )
        if staged_base_cfg.recipe_change_history or recipe_change_rec is not None:
            cfg_obj = merge_and_append_recipe_history(
                prior_history=staged_base_cfg.recipe_change_history,
                current_cfg=cfg_obj,
                new_record=recipe_change_rec,
            )

    memory_estimate = validate_memory_budget(cfg_obj, ctx.gpu_norm)
    minibatch_remainder = check_minibatch_divisibility(cfg_obj)

    eff_world_bytes = _resolve_measured_world_bytes(opts, cfg_obj)
    canary_eval_wall_sec = _scale_train_canary_eval_wall_seconds(
        cfg_obj,
        (
            float(ctx.canary_eval_metrics["wall_seconds"])
            if ctx.canary_eval_metrics is not None
            else None
        ),
    )
    staged_input_bytes = (
        int(ctx.prior.chosen_checkpoint["size_bytes"])
        if ctx.prior.kind == "checkpoint" and ctx.prior.chosen_checkpoint is not None
        else 0
    )
    wall_headroom = estimate_train_wall_headroom(
        cfg_obj,
        gpu=ctx.gpu_norm,
        session_seconds=eff_sec,
        canary_eval_wall_seconds=canary_eval_wall_sec,
        measured_packed_bytes_per_world=eff_world_bytes,
        staged_input_bytes=staged_input_bytes,
    )
    max_wall_sec_int = max(1, int(wall_headroom["max_wall_seconds"]))

    training_plan: dict[str, Any] | None = None
    if opts.expected_steps_per_sec is not None:
        if (
            not math.isfinite(opts.expected_steps_per_sec)
            or float(opts.expected_steps_per_sec) <= 0.0
        ):
            raise ValueError(
                "expected_steps_per_sec must be finite and > 0, "
                f"got {opts.expected_steps_per_sec}"
            )
        steps_per_update = (
            int(cfg_obj.rollout_len) * int(cfg_obj.num_envs) * int(cfg_obj.num_snakes)
        )
        expected_sec_per_update = float(steps_per_update) / float(opts.expected_steps_per_sec)
        if expected_sec_per_update > 0.5 * eff_sec:
            raise ValueError(
                f"Refusing train packet whose single update "
                f"({expected_sec_per_update:.1f} s at {opts.expected_steps_per_sec:.1f} "
                f"steps/s) exceeds half the {opts.hours:g} h session ({0.5 * eff_sec:.1f} s)."
            )
        training_plan = build_training_plan(
            cfg_obj,
            hours=float(opts.hours),
            expected_steps_per_sec=float(opts.expected_steps_per_sec),
            max_wall_seconds=float(max_wall_sec_int),
        )

    return _SizedModePlan(
        job_id=f"v3-ppo-train-{ctx.gpu_slug}-{opts.attempt}",
        est_minutes=est_minutes,
        timeout_minutes=timeout_minutes,
        base_sec=base_sec,
        eff_sec=eff_sec,
        max_wall_sec_int=max_wall_sec_int,
        extra_env={"V3_PPO_MAX_WALL_SECONDS": str(max_wall_sec_int)},
        required_exports=list(TRAIN_REQUIRED_EXPORTS),
        resume_fields={
            "resume": {
                "unit_kind": "checkpoint",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            }
        },
        canary_block=ctx.canary_block,
        train_cfg=cfg_obj,
        training_plan=training_plan,
        memory_estimate=memory_estimate,
        wall_headroom=wall_headroom,
        recipe_overrides=eff_train.recipe_overrides,
        recipe_change=recipe_change_rec,
        minibatch_remainder=minibatch_remainder,
    )


def _load_and_validate_checkpoint_records(
    ckpt_paths: Sequence[Path],
    *,
    ref_game_dict: dict[str, Any] | None = None,
    ref_layout_dict: dict[str, Any] | None = None,
    ref_first_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Load and structurally verify checkpoint records on Mac while checking game/obs compatibility."""
    records: list[dict[str, Any]] = []
    cur_game_dict = ref_game_dict
    cur_layout_dict = ref_layout_dict
    first_p = ref_first_path
    for c_path in ckpt_paths:
        payload, c_sha, c_size = _load_checkpoint_payload(c_path)
        c_cfg = V3PPOConfig.from_dict(payload["config"])
        c_layout = obs_layout_from_game_config(c_cfg.game_config).to_dict()
        if payload.get("obs_layout") is not None and payload["obs_layout"] != c_layout:
            raise ValueError(f"Checkpoint {c_path} obs_layout does not match its game_config")
        g_dict = c_cfg.game_config.to_dict()
        if cur_game_dict is None:
            cur_game_dict = g_dict
            cur_layout_dict = c_layout
            first_p = c_path
        elif g_dict != cur_game_dict or c_layout != cur_layout_dict:
            raise ValueError(f"Incompatible game_config/obs_layout in {c_path} vs {first_p}")
        orig_path = str(payload.get("source_checkpoint_path") or c_path)
        orig_sha = str(payload.get("source_checkpoint_sha256") or c_sha)
        orig_size = int(payload.get("source_checkpoint_size_bytes") or c_size)
        records.append(
            {
                "input_path": str(c_path),
                "src_path": orig_path,
                "src_sha256": orig_sha,
                "raw_size_bytes": orig_size,
                "net_size": c_cfg.net_size,
                "game_config": g_dict,
            }
        )
    return records, cur_game_dict, cur_layout_dict


def _resolve_eval_policy_specs(
    opts: PreparePacketOptions,
) -> tuple[str, list[dict[str, Any]], GameV3Config, int, bool, dict[str, Any] | None]:
    """Validate and resolve eval mode, checkpoint metadata, game_config, and weights-only flag."""
    if opts.eval_weights_only not in ("auto", "true", "false"):
        raise ValueError(
            f"eval_weights_only must be 'auto', 'true', or 'false', got {opts.eval_weights_only!r}"
        )

    eff_eval_mode = opts.eval_mode
    if eff_eval_mode is not None and eff_eval_mode not in EVAL_MODES:
        raise ValueError(f"eval_mode must be one of {EVAL_MODES}, got {eff_eval_mode!r}")

    if eff_eval_mode is None:
        if opts.eval_candidates:
            eff_eval_mode = "anchor"
        elif opts.eval_round_robin:
            eff_eval_mode = "round-robin"
        elif len(opts.eval_policies) == 1:
            eff_eval_mode = "long-game"
        elif len(opts.eval_policies) >= 2:
            eff_eval_mode = "head-to-head"
        else:
            raise ValueError(
                "--mode eval requires --eval-policies (1 for long-game, 2..6 for head-to-head), "
                "--eval-round-robin DIR_OR_LIST, or --candidates DIR_OR_LIST"
            )

    use_weights_only = opts.eval_weights_only != "false"

    if eff_eval_mode == "anchor":
        cand_sources = opts.eval_candidates if opts.eval_candidates else opts.eval_policies
        if not cand_sources:
            raise ValueError("--eval-mode anchor requires --candidates DIR_OR_LIST")
        cand_paths = discover_eval_checkpoints(cand_sources, min_checkpoints=1)
        ref_paths = (
            discover_eval_checkpoints(opts.eval_references, min_checkpoints=0)
            if opts.eval_references
            else []
        )
        scripted_ops = parse_anchor_scripted_opponents(opts.anchor_scripted)
        if opts.eval_labels is not None:
            resolve_unique_labels(cand_paths, opts.eval_labels)
        if opts.eval_reference_labels is not None and ref_paths:
            resolve_unique_labels(ref_paths, opts.eval_reference_labels)

        cand_records, ref_game_dict, ref_layout_dict = _load_and_validate_checkpoint_records(
            cand_paths
        )
        ref_records, ref_game_dict, ref_layout_dict = _load_and_validate_checkpoint_records(
            ref_paths,
            ref_game_dict=ref_game_dict,
            ref_layout_dict=ref_layout_dict,
            ref_first_path=cand_paths[0],
        )
        cand_labels = resolve_unique_labels(
            [Path(rec["src_path"]) for rec in cand_records],
            opts.eval_labels,
        )
        ref_labels = (
            resolve_unique_labels(
                [Path(rec["src_path"]) for rec in ref_records],
                opts.eval_reference_labels,
            )
            if ref_records
            else []
        )

        # Deduplicate bundle entries by original checkpoint SHA-256 (candidates first, then refs).
        bundle_idx_by_sha: dict[str, int] = {}
        unique_records: list[dict[str, Any]] = []
        unique_default_labels: list[str] = []
        cand_bundle_indices: list[int] = []
        ref_bundle_indices: list[int] = []
        for rec, lbl in zip(cand_records, cand_labels):
            sha = str(rec["src_sha256"])
            if sha not in bundle_idx_by_sha:
                bundle_idx_by_sha[sha] = len(unique_records)
                unique_records.append(rec)
                unique_default_labels.append(lbl)
            cand_bundle_indices.append(bundle_idx_by_sha[sha])
        for rec, lbl in zip(ref_records, ref_labels):
            sha = str(rec["src_sha256"])
            if sha not in bundle_idx_by_sha:
                bundle_idx_by_sha[sha] = len(unique_records)
                unique_records.append(rec)
                unique_default_labels.append(lbl)
            ref_bundle_indices.append(bundle_idx_by_sha[sha])

        bundle_labels = resolve_unique_labels(
            [Path(rec["src_path"]) for rec in unique_records],
            None,
        )
        policy_specs: list[dict[str, Any]] = []
        raw_total_bytes = 0
        for idx, (rec, lbl) in enumerate(zip(unique_records, bundle_labels)):
            raw_total_bytes += int(rec["raw_size_bytes"])
            safe_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", lbl).strip("_") or f"p{idx:02d}"
            policy_specs.append(
                {
                    "index": idx,
                    "label": lbl,
                    "safe_slug": safe_slug[:40],
                    **rec,
                }
            )
        assert ref_game_dict is not None
        eval_game_cfg = GameV3Config.from_dict({**ref_game_dict, "max_life_ticks": 0})
        anchor_meta = {
            "candidate_records": cand_records,
            "reference_records": ref_records,
            "candidate_labels": cand_labels,
            "reference_labels": ref_labels,
            "candidate_bundle_indices": cand_bundle_indices,
            "reference_bundle_indices": ref_bundle_indices,
            "scripted_opponents": list(scripted_ops),
        }
        return (
            eff_eval_mode,
            policy_specs,
            eval_game_cfg,
            raw_total_bytes,
            use_weights_only,
            anchor_meta,
        )

    if eff_eval_mode == "round-robin":
        sources = opts.eval_round_robin if opts.eval_round_robin else opts.eval_policies
        if not sources:
            raise ValueError(
                "--eval-mode round-robin requires --eval-round-robin or --eval-policies"
            )
        ckpt_paths = discover_round_robin_checkpoints(sources)
    elif eff_eval_mode == "long-game":
        if len(opts.eval_policies) != 1:
            raise ValueError(
                f"--eval-mode long-game requires exactly 1 checkpoint via --eval-policies, "
                f"got {len(opts.eval_policies)}"
            )
        ckpt_paths = [Path(opts.eval_policies[0]).resolve()]
    else:
        if not (2 <= len(opts.eval_policies) <= 6):
            raise ValueError(
                f"--eval-mode head-to-head requires 2 to 6 checkpoints via --eval-policies, "
                f"got {len(opts.eval_policies)}"
            )
        ckpt_paths = [Path(p).resolve() for p in opts.eval_policies]

    if opts.eval_labels is not None:
        resolve_unique_labels(ckpt_paths, opts.eval_labels)

    loaded_records, ref_game_dict, _ = _load_and_validate_checkpoint_records(ckpt_paths)
    raw_total_bytes = sum(int(rec["raw_size_bytes"]) for rec in loaded_records)
    labels = resolve_unique_labels(
        [Path(rec["src_path"]) for rec in loaded_records],
        opts.eval_labels,
    )
    policy_specs = []
    for idx, (rec, lbl) in enumerate(zip(loaded_records, labels)):
        safe_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", lbl).strip("_") or f"p{idx:02d}"
        policy_specs.append(
            {
                "index": idx,
                "label": lbl,
                "safe_slug": safe_slug[:40],
                **rec,
            }
        )

    assert ref_game_dict is not None
    eval_game_cfg = GameV3Config.from_dict({**ref_game_dict, "max_life_ticks": 0})
    return eff_eval_mode, policy_specs, eval_game_cfg, raw_total_bytes, use_weights_only, None


def _estimate_anchor_ms_per_tick(
    gpu_norm: str,
    *,
    num_envs: int,
    num_snakes: int,
    learner_slots: int,
    cand_net_size: NetSize,
    ref_net_sizes: Sequence[NetSize],
    scripted_opponents: Sequence[OpponentPolicy],
) -> float:
    """Calibrated GPU per-tick estimate for anchor evaluation (anchored to L4/A100 canary logs)."""
    gpu_scale_map = {"L4": 1.00, "A100": 0.48, "H100": 0.30}
    gpu_scale = float(gpu_scale_map.get(str(gpu_norm).strip().upper(), 1.00))
    row_ms_map: dict[str, float] = {"S": 0.040, "M": 0.088, "L": 0.165}

    e_cnt = max(1, int(num_envs))
    n_cnt = max(1, int(num_snakes))
    ls_cnt = max(1, min(n_cnt, int(learner_slots)))
    opp_seats = max(0, n_cnt - ls_cnt)
    num_opp_types = len(ref_net_sizes) + len(scripted_opponents)

    cand_rows = float(e_cnt * ls_cnt)
    cand_ms = cand_rows * row_ms_map[str(cand_net_size)]

    ref_ms = 0.0
    if opp_seats > 0 and num_opp_types > 0 and ref_net_sizes:
        seats_per_opp = float(opp_seats) / float(num_opp_types)
        for r_net in ref_net_sizes:
            ref_ms += float(e_cnt) * seats_per_opp * row_ms_map[str(r_net)]

    has_hard_scripted = any(op in ("cut_off", "head_evader") for op in scripted_opponents)
    world_sim_ms = float(e_cnt) * (0.45 if has_hard_scripted else 0.32)
    launch_ms = 6.0
    return round(gpu_scale * (launch_ms + world_sim_ms + cand_ms + ref_ms), 4)


def _est_compressed_policy_bytes(net_size: NetSize, *, use_weights_only: bool) -> int:
    """Conservative compressed bundle byte estimate for one policy of ``net_size``."""
    return int(
        math.ceil(
            get_net_preset(net_size).exact_param_count * 4 * (0.85 if use_weights_only else 0.90)
        )
    )


def _count_fitting_anchor_candidates(
    *,
    cand_net_sizes_seq: Sequence[NetSize],
    fallback_cand_net: NetSize,
    ref_net_sizes: Sequence[NetSize],
    use_weights_only: bool,
    ms_per_tick: float,
    ticks_per_cand: int,
    safety_factor: float,
    session_cap_min: float,
    transfer_cap_sec: float,
    prior_unit_sizes: Sequence[int] = (),
    max_probe: int = 128,
) -> dict[str, Any]:
    """Compute how many candidates fit ``session_cap_min`` and ``transfer_cap_sec``."""
    ref_bundle_b = sum(
        _est_compressed_policy_bytes(rn, use_weights_only=use_weights_only) for rn in ref_net_sizes
    )
    cand_sec = float(ticks_per_cand) * (float(ms_per_tick) / 1000.0) * float(safety_factor)
    max_session_sec = float(session_cap_min) * 60.0

    fit_session_k = 0
    fit_upload_k = 0
    fit_both_k = 0
    cand_bundle_b = 0
    for k in range(1, max_probe + 1):
        c_net = (
            cand_net_sizes_seq[k - 1] if (k - 1) < len(cand_net_sizes_seq) else fallback_cand_net
        )
        cand_bundle_b += _est_compressed_policy_bytes(c_net, use_weights_only=use_weights_only)
        bundle_b = 4096 + ref_bundle_b + cand_bundle_b
        ledger_b = 4096 + 320 * (len(ref_net_sizes) + k)
        up_est = estimate_eval_upload_time(
            [bundle_b, ledger_b, *prior_unit_sizes],
            transfer_cap_seconds=float(transfer_cap_sec),
            upload_chunk_bytes=int(colab_transfers.UPLOAD_CHUNK_BYTES),
            staging_control_seconds=float(colab_policy.STAGING_CONTROL_SECONDS),
            staging_recovery_seconds=float(colab_policy.STAGING_RECOVERY_SECONDS),
            backoff_first_seconds=float(colab_policy.BACKOFF_SECONDS[0]),
            control_hang_prob=float(EVAL_UPLOAD_CONTROL_HANG_PROB),
            chunk_rate_bytes_per_sec=float(EVAL_UPLOAD_CHUNK_RATE_BYTES_PER_SEC),
        )
        up_sec = float(up_est["estimated_upload_seconds"])
        tot_sec = 210.0 + up_sec + float(k) * cand_sec
        fits_sess = tot_sec <= max_session_sec
        fits_up = bool(up_est["fits_transfer_cap"])
        if fits_sess:
            fit_session_k = k
        if fits_up:
            fit_upload_k = k
        if fits_sess and fits_up:
            fit_both_k = k
        if not fits_sess and not fits_up:
            break

    return {
        "ms_per_tick": round(float(ms_per_tick), 4),
        "seconds_per_candidate_with_margin": round(cand_sec, 2),
        "fit_session_cap": int(fit_session_k),
        "fit_upload_cap": int(fit_upload_k),
        "fit_effective": int(fit_both_k),
    }


def _compute_anchor_capacity_summary(
    *,
    gpu_norm: str,
    candidate_net_sizes: Sequence[NetSize],
    reference_net_sizes: Sequence[NetSize],
    scripted_opponents: Sequence[OpponentPolicy],
    eff_envs: int,
    eff_learner_slots: int,
    eff_burnin: int,
    eff_window: int,
    n_snakes: int,
    use_weights_only: bool,
    session_cap_min: float,
    measured_or_assumed_ms_per_tick: float | None,
    prior_unit_sizes: Sequence[int],
    is_canary: bool,
) -> dict[str, Any]:
    """Compute candidate capacity for the packet AND for all-small / all-medium lists at defaults."""
    transfer_cap = (
        float(colab_canary.MAX_UPLOAD_SECONDS)
        if is_canary
        else float(colab_transfer_policy.DEFAULT_TRANSFER_SECONDS)
    )
    safety_factor = float(EVAL_TIMING_SAFETY_FACTOR)
    worst_cand_net: NetSize = (
        max(candidate_net_sizes, key=lambda sz: get_net_preset(sz).exact_param_count)
        if candidate_net_sizes
        else "S"
    )
    pkt_ms = (
        float(measured_or_assumed_ms_per_tick)
        if measured_or_assumed_ms_per_tick is not None
        else _estimate_anchor_ms_per_tick(
            gpu_norm,
            num_envs=eff_envs,
            num_snakes=n_snakes,
            learner_slots=eff_learner_slots,
            cand_net_size=worst_cand_net,
            ref_net_sizes=reference_net_sizes,
            scripted_opponents=scripted_opponents,
        )
    )
    packet_cap = _count_fitting_anchor_candidates(
        cand_net_sizes_seq=candidate_net_sizes,
        fallback_cand_net=worst_cand_net,
        ref_net_sizes=reference_net_sizes,
        use_weights_only=use_weights_only,
        ms_per_tick=pkt_ms,
        ticks_per_cand=int(eff_burnin + eff_window),
        safety_factor=safety_factor,
        session_cap_min=session_cap_min,
        transfer_cap_sec=transfer_cap,
        prior_unit_sizes=prior_unit_sizes,
    )

    def_ticks = int(DEFAULT_ANCHOR_BURNIN_TICKS + DEFAULT_ANCHOR_WINDOW_TICKS)
    def_envs = int(DEFAULT_ANCHOR_NUM_ENVS)
    def_ls = int(DEFAULT_ANCHOR_LEARNER_SLOTS)
    def_session_min = float(EVAL_DEFAULT_SESSION_MINUTES)
    def_scripted = DEFAULT_ANCHOR_SCRIPTED_OPPONENTS

    # All-small & all-medium at defaults against current references AND homogeneous 2-ref pool.
    s_with_cur_refs_ms = _estimate_anchor_ms_per_tick(
        gpu_norm,
        num_envs=def_envs,
        num_snakes=n_snakes,
        learner_slots=def_ls,
        cand_net_size="S",
        ref_net_sizes=reference_net_sizes,
        scripted_opponents=scripted_opponents,
    )
    m_with_cur_refs_ms = _estimate_anchor_ms_per_tick(
        gpu_norm,
        num_envs=def_envs,
        num_snakes=n_snakes,
        learner_slots=def_ls,
        cand_net_size="M",
        ref_net_sizes=reference_net_sizes,
        scripted_opponents=scripted_opponents,
    )
    # If caller/canary provided a measured ms/tick at default envs/ls, anchor the S/M delta to it!
    if (
        measured_or_assumed_ms_per_tick is not None
        and eff_envs == def_envs
        and eff_learner_slots == def_ls
    ):
        model_base = _estimate_anchor_ms_per_tick(
            gpu_norm,
            num_envs=def_envs,
            num_snakes=n_snakes,
            learner_slots=def_ls,
            cand_net_size=worst_cand_net,
            ref_net_sizes=reference_net_sizes,
            scripted_opponents=scripted_opponents,
        )
        delta_from_model = float(measured_or_assumed_ms_per_tick) - model_base
        s_with_cur_refs_ms = max(1.0, round(s_with_cur_refs_ms + delta_from_model, 4))
        m_with_cur_refs_ms = max(1.0, round(m_with_cur_refs_ms + delta_from_model, 4))

    all_small_with_refs = _count_fitting_anchor_candidates(
        cand_net_sizes_seq=(),
        fallback_cand_net="S",
        ref_net_sizes=reference_net_sizes,
        use_weights_only=True,
        ms_per_tick=s_with_cur_refs_ms,
        ticks_per_cand=def_ticks,
        safety_factor=safety_factor,
        session_cap_min=def_session_min,
        transfer_cap_sec=float(colab_transfer_policy.DEFAULT_TRANSFER_SECONDS),
    )
    all_medium_with_refs = _count_fitting_anchor_candidates(
        cand_net_sizes_seq=(),
        fallback_cand_net="M",
        ref_net_sizes=reference_net_sizes,
        use_weights_only=True,
        ms_per_tick=m_with_cur_refs_ms,
        ticks_per_cand=def_ticks,
        safety_factor=safety_factor,
        session_cap_min=def_session_min,
        transfer_cap_sec=float(colab_transfer_policy.DEFAULT_TRANSFER_SECONDS),
    )

    pure_s_ms = _estimate_anchor_ms_per_tick(
        gpu_norm,
        num_envs=def_envs,
        num_snakes=n_snakes,
        learner_slots=def_ls,
        cand_net_size="S",
        ref_net_sizes=("S", "S"),
        scripted_opponents=def_scripted,
    )
    pure_m_ms = _estimate_anchor_ms_per_tick(
        gpu_norm,
        num_envs=def_envs,
        num_snakes=n_snakes,
        learner_slots=def_ls,
        cand_net_size="M",
        ref_net_sizes=("M", "M"),
        scripted_opponents=def_scripted,
    )
    all_small_homogeneous = _count_fitting_anchor_candidates(
        cand_net_sizes_seq=(),
        fallback_cand_net="S",
        ref_net_sizes=("S", "S"),
        use_weights_only=True,
        ms_per_tick=pure_s_ms,
        ticks_per_cand=def_ticks,
        safety_factor=safety_factor,
        session_cap_min=def_session_min,
        transfer_cap_sec=float(colab_transfer_policy.DEFAULT_TRANSFER_SECONDS),
    )
    all_medium_homogeneous = _count_fitting_anchor_candidates(
        cand_net_sizes_seq=(),
        fallback_cand_net="M",
        ref_net_sizes=("M", "M"),
        use_weights_only=True,
        ms_per_tick=pure_m_ms,
        ticks_per_cand=def_ticks,
        safety_factor=safety_factor,
        session_cap_min=def_session_min,
        transfer_cap_sec=float(colab_transfer_policy.DEFAULT_TRANSFER_SECONDS),
    )

    return {
        "gpu": gpu_norm,
        "defaults": {
            "num_envs": def_envs,
            "learner_slots": def_ls,
            "burnin_ticks": int(DEFAULT_ANCHOR_BURNIN_TICKS),
            "window_ticks": int(DEFAULT_ANCHOR_WINDOW_TICKS),
            "ticks_per_candidate": def_ticks,
            "session_cap_minutes": def_session_min,
            "transfer_cap_seconds": float(colab_transfer_policy.DEFAULT_TRANSFER_SECONDS),
        },
        "packet": packet_cap,
        "all_small_at_defaults_with_current_references": all_small_with_refs,
        "all_medium_at_defaults_with_current_references": all_medium_with_refs,
        "all_small_at_defaults_with_2S_references": all_small_homogeneous,
        "all_medium_at_defaults_with_2M_references": all_medium_homogeneous,
    }


def _print_anchor_capacity_stderr(cap: Mapping[str, Any]) -> None:
    """Print mandatory all-small and all-medium default anchor capacity report to stderr."""
    defs = cap["defaults"]
    pkt = cap["packet"]
    s_cur = cap["all_small_at_defaults_with_current_references"]
    m_cur = cap["all_medium_at_defaults_with_current_references"]
    s_hom = cap["all_small_at_defaults_with_2S_references"]
    m_hom = cap["all_medium_at_defaults_with_2M_references"]
    print(
        f"V3_PPO_ANCHOR_CAPACITY gpu={cap['gpu']} "
        f"defaults(envs={defs['num_envs']},burnin={defs['burnin_ticks']},"
        f"window={defs['window_ticks']},session={defs['session_cap_minutes']:.1f}m,"
        f"upload_cap={defs['transfer_cap_seconds']:.1f}s): "
        f"packet_fits={pkt['fit_effective']} "
        f"(60m_session_cap={pkt['fit_session_cap']}, upload_cap={pkt['fit_upload_cap']}); "
        f"all_small_at_defaults={s_cur['fit_effective']} with_current_refs "
        f"(60m={s_cur['fit_session_cap']}, upload={s_cur['fit_upload_cap']}) / "
        f"{s_hom['fit_effective']} with_2S_refs "
        f"(60m={s_hom['fit_session_cap']}, upload={s_hom['fit_upload_cap']}); "
        f"all_medium_at_defaults={m_cur['fit_effective']} with_current_refs "
        f"(60m={m_cur['fit_session_cap']}, upload={m_cur['fit_upload_cap']}) / "
        f"{m_hom['fit_effective']} with_2M_refs "
        f"(60m={m_hom['fit_session_cap']}, upload={m_hom['fit_upload_cap']})",
        file=sys.stderr,
        flush=True,
    )


def _resolve_and_validate_eval_layout(
    opts: PreparePacketOptions,
    ctx: _ValidatedPacketContext,
) -> dict[str, Any]:
    """Validate eval divisibility, prior unit reuse keys, VRAM budget, and batch layout."""
    (
        eff_eval_mode,
        policy_specs,
        eval_game_cfg,
        raw_total_bytes,
        use_weights_only,
        anchor_meta,
    ) = _resolve_eval_policy_specs(opts)
    n_snakes = int(eval_game_cfg.num_snakes)

    if eff_eval_mode == "anchor":
        eff_envs = DEFAULT_ANCHOR_NUM_ENVS if opts.num_envs is None else int(opts.num_envs)
        eff_burnin = (
            DEFAULT_ANCHOR_BURNIN_TICKS
            if opts.eval_burnin_ticks is None
            else int(opts.eval_burnin_ticks)
        )
        eff_window = (
            DEFAULT_ANCHOR_WINDOW_TICKS
            if opts.eval_window_ticks is None
            else int(opts.eval_window_ticks)
        )
        eff_learner_slots = (
            DEFAULT_ANCHOR_LEARNER_SLOTS if opts.learner_slots is None else int(opts.learner_slots)
        )
    elif eff_eval_mode == "long-game":
        eff_envs = DEFAULT_LONG_GAME_NUM_ENVS if opts.num_envs is None else int(opts.num_envs)
        eff_burnin = (
            DEFAULT_LONG_GAME_BURNIN_TICKS
            if opts.eval_burnin_ticks is None
            else int(opts.eval_burnin_ticks)
        )
        eff_window = (
            DEFAULT_LONG_GAME_WINDOW_TICKS
            if opts.eval_window_ticks is None
            else int(opts.eval_window_ticks)
        )
        eff_learner_slots = n_snakes if opts.learner_slots is None else int(opts.learner_slots)
    else:
        eff_envs = DEFAULT_H2H_NUM_ENVS if opts.num_envs is None else int(opts.num_envs)
        eff_burnin = (
            DEFAULT_H2H_BURNIN_TICKS
            if opts.eval_burnin_ticks is None
            else int(opts.eval_burnin_ticks)
        )
        eff_window = (
            DEFAULT_H2H_WINDOW_TICKS
            if opts.eval_window_ticks is None
            else int(opts.eval_window_ticks)
        )
        eff_learner_slots = n_snakes if opts.learner_slots is None else int(opts.learner_slots)

    if eff_envs < 1:
        raise ValueError(f"eval num_envs must be >= 1, got {eff_envs}")
    if eff_burnin < 0:
        raise ValueError(f"eval burnin_ticks must be >= 0, got {eff_burnin}")
    if eff_window < 6:
        raise ValueError(f"eval window_ticks must be >= 6, got {eff_window}")

    if eff_eval_mode == "anchor":
        assert anchor_meta is not None
        if not (1 <= eff_learner_slots <= n_snakes):
            raise ValueError(
                f"anchor learner_slots must be in [1, {n_snakes}], got {eff_learner_slots}"
            )
        ref_records = anchor_meta["reference_records"]
        scripted_ops = anchor_meta["scripted_opponents"]
        if eff_learner_slots < n_snakes and (len(ref_records) + len(scripted_ops)) < 1:
            raise ValueError(
                f"anchor mode with learner_slots={eff_learner_slots} < {n_snakes} requires "
                "at least one reference checkpoint or scripted opponent"
            )

    if eff_eval_mode == "head-to-head" and eff_envs % len(policy_specs) != 0:
        raise ValueError(
            f"head-to-head requires num_envs ({eff_envs}) to be a multiple of the number of "
            f"policies sharing a world ({len(policy_specs)}) so rotated seats cancel slot bias."
        )
    if eff_eval_mode == "round-robin" and eff_envs % 2 != 0:
        raise ValueError(
            f"round-robin requires num_envs ({eff_envs}) per pair to be a multiple of 2 "
            "policies sharing a world so rotated seats cancel slot bias."
        )

    if ctx.prior.kind == "eval_pairs":
        if eff_eval_mode != "round-robin":
            raise ValueError(
                f"prior eval_pairs continuation is only supported in round-robin mode, "
                f"got {eff_eval_mode!r}"
            )
        valid_pairs = {
            f"pair_{i:02d}_{j:02d}": (i, j)
            for i in range(len(policy_specs))
            for j in range(i + 1, len(policy_specs))
        }
        for pair_json in ctx.prior.eval_pair_paths:
            uid = pair_json.stem
            if uid not in valid_pairs:
                raise ValueError(
                    f"Prior pair artifact {pair_json} ({uid!r}) does not belong to "
                    f"{len(policy_specs)}-policy round-robin allocation"
                )
            idx_a, idx_b = valid_pairs[uid]
            expected_rk = build_pair_reuse_key(
                num_envs=eff_envs,
                burnin_ticks=eff_burnin,
                window_ticks=eff_window,
                greedy=bool(opts.eval_greedy),
                seed=int(opts.eval_seed),
                policy_a_sha256=str(policy_specs[idx_a]["src_sha256"]),
                policy_b_sha256=str(policy_specs[idx_b]["src_sha256"]),
            )
            parsed_pair = json.loads(pair_json.read_text(encoding="utf-8"))
            validate_pair_unit_reuse(uid, parsed_pair, expected_rk)

    if ctx.prior.kind == "eval_candidates":
        if eff_eval_mode != "anchor":
            raise ValueError(
                f"prior eval_candidates continuation is only supported in anchor mode, "
                f"got {eff_eval_mode!r}"
            )
        assert anchor_meta is not None
        cand_records = anchor_meta["candidate_records"]
        ref_records = anchor_meta["reference_records"]
        scripted_ops = anchor_meta["scripted_opponents"]
        valid_cands = {
            candidate_unit_id(str(rec["src_sha256"])): str(rec["src_sha256"])
            for rec in cand_records
        }
        ref_shas = [str(rec["src_sha256"]) for rec in ref_records]
        for cand_json in ctx.prior.eval_candidate_paths:
            uid = cand_json.stem
            if uid not in valid_cands:
                raise ValueError(
                    f"Prior anchor candidate artifact {cand_json} ({uid!r}) does not belong to "
                    f"{len(cand_records)}-candidate anchor allocation"
                )
            expected_rk = build_anchor_candidate_reuse_key(
                arena_radius=float(eval_game_cfg.arena_radius),
                num_envs=eff_envs,
                num_snakes=n_snakes,
                learner_slots=eff_learner_slots,
                burnin_ticks=eff_burnin,
                window_ticks=eff_window,
                greedy=bool(opts.eval_greedy),
                seed=int(opts.eval_seed),
                candidate_sha256=valid_cands[uid],
                reference_sha256s=ref_shas,
                scripted_opponents=scripted_ops,
            )
            parsed_cand = json.loads(cand_json.read_text(encoding="utf-8"))
            validate_anchor_candidate_unit_reuse(uid, parsed_cand, expected_rk)

    budget_bytes = int(GPU_VRAM_BUDGET_BYTES[ctx.gpu_norm])
    net_sizes: list[NetSize] = [sp["net_size"] for sp in policy_specs]
    eff_amp = True if opts.amp is None else bool(opts.amp)
    vram_max_ppb: int | None = None
    anchor_ledger_spec: dict[str, Any] | None = None
    cand_net_sizes: list[NetSize] = []
    ref_net_sizes: list[NetSize] = []

    if eff_eval_mode == "round-robin":
        reused_pair_ids = {p.stem for p in ctx.prior.eval_pair_paths}
        total_pairs = (len(policy_specs) * (len(policy_specs) - 1)) // 2
        remaining_pairs = total_pairs - len(reused_pair_ids)
        sizing_pairs = max(1, remaining_pairs)
        if opts.eval_sequential_pairs:
            eff_pairs_per_batch = 1
            num_batches = remaining_pairs
            batch_worlds = eff_envs
            mem_est = estimate_eval_memory(
                game_cfg=eval_game_cfg,
                net_sizes=net_sizes,
                eval_mode="round-robin",
                num_envs_per_match=eff_envs,
                pairs_per_batch=1,
                amp=eff_amp,
                gpu=ctx.gpu_norm,
            )
            if int(mem_est["estimated_peak_bytes"]) > budget_bytes:
                raise ValueError(
                    f"Refusing eval packet on {ctx.gpu_norm}: sequential single-pair peak VRAM "
                    f"{int(mem_est['estimated_peak_bytes']) / 1e9:.2f} GB exceeds "
                    f"{budget_bytes / 1e9:.2f} GB budget."
                )
        else:
            vram_max_ppb, _ = select_pairs_per_batch(
                game_cfg=eval_game_cfg,
                net_sizes=net_sizes,
                num_envs_per_pair=eff_envs,
                total_pairs_to_run=sizing_pairs,
                vram_budget_bytes=budget_bytes,
                amp=eff_amp,
                max_pairs_override=None,
                gpu_label=ctx.gpu_norm,
            )
            eff_pairs_per_batch, mem_est = select_pairs_per_batch(
                game_cfg=eval_game_cfg,
                net_sizes=net_sizes,
                num_envs_per_pair=eff_envs,
                total_pairs_to_run=sizing_pairs,
                vram_budget_bytes=budget_bytes,
                amp=eff_amp,
                max_pairs_override=opts.eval_max_pairs_per_batch,
                gpu_label=ctx.gpu_norm,
            )
            num_batches = (
                int(math.ceil(remaining_pairs / eff_pairs_per_batch)) if remaining_pairs > 0 else 0
            )
            batch_worlds = eff_pairs_per_batch * eff_envs
    elif eff_eval_mode == "anchor":
        assert anchor_meta is not None
        cand_records = anchor_meta["candidate_records"]
        ref_records = anchor_meta["reference_records"]
        cand_net_sizes = [rec["net_size"] for rec in cand_records]
        ref_net_sizes = [rec["net_size"] for rec in ref_records]
        reused_cand_ids = {p.stem for p in ctx.prior.eval_candidate_paths}
        total_pairs = len(cand_records)
        remaining_pairs = sum(
            1
            for rec in cand_records
            if candidate_unit_id(str(rec["src_sha256"])) not in reused_cand_ids
        )
        eff_pairs_per_batch = 1
        num_batches = remaining_pairs
        batch_worlds = eff_envs
        mem_est = estimate_eval_memory(
            game_cfg=eval_game_cfg,
            net_sizes=net_sizes,
            eval_mode="anchor",
            num_envs_per_match=eff_envs,
            pairs_per_batch=1,
            learner_slots=eff_learner_slots,
            num_reference_policies=len(ref_records),
            num_scripted_opponents=len(anchor_meta["scripted_opponents"]),
            amp=eff_amp,
            gpu=ctx.gpu_norm,
        )
        if int(mem_est["estimated_peak_bytes"]) > budget_bytes:
            raise ValueError(
                f"Refusing eval packet on {ctx.gpu_norm}: estimated peak VRAM "
                f"{int(mem_est['estimated_peak_bytes']) / 1e9:.2f} GB exceeds "
                f"{budget_bytes / 1e9:.2f} GB budget."
            )
        anchor_ledger_spec = {
            "candidate_bundle_indices": list(anchor_meta["candidate_bundle_indices"]),
            "reference_bundle_indices": list(anchor_meta["reference_bundle_indices"]),
            "candidate_labels": list(anchor_meta["candidate_labels"]),
            "reference_labels": list(anchor_meta["reference_labels"]),
            "candidate_sha256s": [str(r["src_sha256"]) for r in cand_records],
            "reference_sha256s": [str(r["src_sha256"]) for r in ref_records],
            "scripted_opponents": list(anchor_meta["scripted_opponents"]),
            "learner_slots": int(eff_learner_slots),
        }
    else:
        total_pairs = 1
        remaining_pairs = 1
        eff_pairs_per_batch = 1
        num_batches = 1
        batch_worlds = eff_envs
        mem_est = estimate_eval_memory(
            game_cfg=eval_game_cfg,
            net_sizes=net_sizes,
            eval_mode=eff_eval_mode,
            num_envs_per_match=eff_envs,
            pairs_per_batch=1,
            learner_slots=opts.learner_slots,
            amp=eff_amp,
            gpu=ctx.gpu_norm,
        )
        if int(mem_est["estimated_peak_bytes"]) > budget_bytes:
            raise ValueError(
                f"Refusing eval packet on {ctx.gpu_norm}: estimated peak VRAM "
                f"{int(mem_est['estimated_peak_bytes']) / 1e9:.2f} GB exceeds "
                f"{budget_bytes / 1e9:.2f} GB budget."
            )

    memory_estimate: dict[str, Any] = {
        "gpu": ctx.gpu_norm,
        "budget_bytes": budget_bytes,
        "fits_vram_budget": True,
        **mem_est,
    }
    if vram_max_ppb is not None:
        memory_estimate["vram_max_pairs_per_batch"] = vram_max_ppb

    return {
        "eff_eval_mode": eff_eval_mode,
        "policy_specs": policy_specs,
        "eval_game_cfg": eval_game_cfg,
        "n_snakes": n_snakes,
        "raw_total_bytes": raw_total_bytes,
        "use_weights_only": use_weights_only,
        "eff_envs": eff_envs,
        "eff_learner_slots": eff_learner_slots,
        "eff_burnin": eff_burnin,
        "eff_window": eff_window,
        "total_pairs": total_pairs,
        "remaining_pairs": remaining_pairs,
        "eff_pairs_per_batch": eff_pairs_per_batch,
        "num_batches": num_batches,
        "batch_worlds": batch_worlds,
        "memory_estimate": memory_estimate,
        "anchor_ledger_spec": anchor_ledger_spec,
        "candidate_net_sizes": cand_net_sizes,
        "reference_net_sizes": ref_net_sizes,
    }


def _size_eval_mode(
    opts: PreparePacketOptions,
    ctx: _ValidatedPacketContext,
) -> _SizedModePlan:
    """Resolve and validate standalone policy evaluation packet (`round-robin`, `anchor`, etc.)."""
    layout = _resolve_and_validate_eval_layout(opts, ctx)
    eff_eval_mode = str(layout["eff_eval_mode"])
    policy_specs: list[dict[str, Any]] = layout["policy_specs"]
    n_snakes = int(layout["n_snakes"])
    raw_total_bytes = int(layout["raw_total_bytes"])
    use_weights_only = bool(layout["use_weights_only"])
    eff_envs = int(layout["eff_envs"])
    eff_learner_slots = int(layout["eff_learner_slots"])
    eff_burnin = int(layout["eff_burnin"])
    eff_window = int(layout["eff_window"])
    total_pairs = int(layout["total_pairs"])
    remaining_pairs = int(layout["remaining_pairs"])
    eff_pairs_per_batch = int(layout["eff_pairs_per_batch"])
    num_batches = int(layout["num_batches"])
    batch_worlds = int(layout["batch_worlds"])
    memory_estimate: dict[str, Any] = layout["memory_estimate"]
    anchor_ledger_spec: dict[str, Any] | None = layout["anchor_ledger_spec"]

    est_staged_bytes = (
        sum(
            int(
                math.ceil(
                    get_net_preset(sp["net_size"]).exact_param_count
                    * 4
                    * (0.85 if use_weights_only else 0.90)
                )
            )
            for sp in policy_specs
        )
        + 4096
    )
    prior_unit_sizes = [
        int(p.stat().st_size) for p in (*ctx.prior.eval_pair_paths, *ctx.prior.eval_candidate_paths)
    ]
    stage2_upload_est = estimate_eval_upload_time(
        [est_staged_bytes, 4096, *prior_unit_sizes],
        transfer_cap_seconds=float(colab_transfer_policy.DEFAULT_TRANSFER_SECONDS),
        upload_chunk_bytes=int(colab_transfers.UPLOAD_CHUNK_BYTES),
        staging_control_seconds=float(colab_policy.STAGING_CONTROL_SECONDS),
        staging_recovery_seconds=float(colab_policy.STAGING_RECOVERY_SECONDS),
        backoff_first_seconds=float(colab_policy.BACKOFF_SECONDS[0]),
        control_hang_prob=float(EVAL_UPLOAD_CONTROL_HANG_PROB),
        chunk_rate_bytes_per_sec=float(EVAL_UPLOAD_CHUNK_RATE_BYTES_PER_SEC),
    )

    if opts.assume_ms_per_tick is not None:
        if (
            not math.isfinite(float(opts.assume_ms_per_tick))
            or float(opts.assume_ms_per_tick) <= 0.0
        ):
            raise ValueError(
                f"--assume-ms-per-tick must be finite and > 0, got {opts.assume_ms_per_tick}"
            )
        base_ms_per_tick = float(opts.assume_ms_per_tick)
        timing_source = "assumed_override"
    elif ctx.canary_eval_smoke is not None:
        smoke = ctx.canary_eval_smoke
        smoke_mode = str(smoke.get("smoke_mode", ""))
        if smoke_mode != eff_eval_mode:
            raise ValueError(
                f"Bound canary eval_canary_smoke mode={smoke_mode!r} does not match "
                f"eval_mode={eff_eval_mode!r}; run an eval canary for this mode or pass "
                "--assume-ms-per-tick."
            )
        if eff_eval_mode == "round-robin":
            want_batched = not bool(opts.eval_sequential_pairs)
            if bool(smoke.get("batched_pairs", False)) != want_batched:
                raise ValueError(
                    f"Bound canary eval_canary_smoke batched_pairs={smoke.get('batched_pairs')} "
                    f"does not match batched_pairs={want_batched}; re-run eval canary or pass "
                    "--assume-ms-per-tick."
                )
        smoke_worlds = int(smoke.get("timed_worlds", 0))
        if remaining_pairs > 0 and smoke_worlds < batch_worlds:
            raise ValueError(
                f"Bound canary eval_canary_smoke timed_worlds={smoke_worlds} is smaller than "
                f"packet batch_worlds={batch_worlds}; re-run eval canary at target world count "
                "or pass --assume-ms-per-tick."
            )
        base_ms_per_tick = float(smoke["ms_per_tick"])
        timing_source = "canary_measured"
    elif eff_eval_mode == "anchor" and opts.allow_missing_canary_receipt:
        assert anchor_ledger_spec is not None
        cand_nets: list[NetSize] = layout["candidate_net_sizes"]
        worst_c_net: NetSize = (
            max(cand_nets, key=lambda sz: get_net_preset(sz).exact_param_count)
            if cand_nets
            else "S"
        )
        base_ms_per_tick = _estimate_anchor_ms_per_tick(
            ctx.gpu_norm,
            num_envs=eff_envs,
            num_snakes=n_snakes,
            learner_slots=eff_learner_slots,
            cand_net_size=worst_c_net,
            ref_net_sizes=layout["reference_net_sizes"],
            scripted_opponents=anchor_ledger_spec["scripted_opponents"],
        )
        timing_source = "calibrated_gpu_model"
    else:
        raise ValueError(
            "--mode eval requires a bound canary receipt containing eval_canary_smoke.ms_per_tick "
            "(prepared with --mode canary and the target eval policies/world count), or explicit "
            "--assume-ms-per-tick <MS>."
        )

    session_cap_min = float(opts.eval_session_minutes)
    if session_cap_min <= 0 or not math.isfinite(session_cap_min):
        raise ValueError(f"eval_session_minutes must be finite and > 0, got {session_cap_min}")

    anchor_capacity: dict[str, Any] | None = None
    if eff_eval_mode == "anchor":
        assert anchor_ledger_spec is not None
        anchor_capacity = _compute_anchor_capacity_summary(
            gpu_norm=ctx.gpu_norm,
            candidate_net_sizes=layout["candidate_net_sizes"],
            reference_net_sizes=layout["reference_net_sizes"],
            scripted_opponents=tuple(anchor_ledger_spec["scripted_opponents"]),
            eff_envs=eff_envs,
            eff_learner_slots=eff_learner_slots,
            eff_burnin=eff_burnin,
            eff_window=eff_window,
            n_snakes=n_snakes,
            use_weights_only=use_weights_only,
            session_cap_min=session_cap_min,
            measured_or_assumed_ms_per_tick=base_ms_per_tick,
            prior_unit_sizes=prior_unit_sizes,
            is_canary=False,
        )
        _print_anchor_capacity_stderr(anchor_capacity)

    ticks_per_match = int(eff_burnin + eff_window)
    ticks_per_batch = ticks_per_match
    steps_per_match = int(eff_envs) * n_snakes * ticks_per_match
    total_sim_ticks = int(num_batches * ticks_per_batch)
    safety_factor = float(EVAL_TIMING_SAFETY_FACTOR)
    effective_ms_per_tick = base_ms_per_tick * safety_factor
    raw_compute_sec = float(total_sim_ticks) * (base_ms_per_tick / 1000.0)
    est_compute_sec = raw_compute_sec * safety_factor

    upload_allowance_sec = float(stage2_upload_est["estimated_upload_seconds"])
    setup_and_export_sec = 210.0 + upload_allowance_sec
    max_session_sec = session_cap_min * 60.0
    available_compute_sec = max_session_sec - setup_and_export_sec
    sizing_arithmetic = (
        f"{num_batches} batch(es) * {ticks_per_batch} ticks/batch = {total_sim_ticks} sim ticks; "
        f"{total_sim_ticks} ticks * {base_ms_per_tick:.3f} ms/tick ({timing_source}) * "
        f"{safety_factor:.2f}x margin = {est_compute_sec:.2f} s ({est_compute_sec / 60.0:.2f} min) "
        f"compute + {setup_and_export_sec:.2f} s ({setup_and_export_sec / 60.0:.2f} min) "
        f"staging/teardown = {(est_compute_sec + setup_and_export_sec) / 60.0:.2f} min vs "
        f"{session_cap_min:.2f} min session cap"
    )
    if est_compute_sec > available_compute_sec:
        if eff_eval_mode == "anchor" and anchor_capacity is not None:
            pkt_cap = anchor_capacity["packet"]
            s_cap = anchor_capacity["all_small_at_defaults_with_current_references"]
            m_cap = anchor_capacity["all_medium_at_defaults_with_current_references"]
            raise ValueError(
                f"Refusing anchor eval packet ({total_pairs} candidates, "
                f"{remaining_pairs} pending, worlds={batch_worlds}, {sizing_arithmetic}): "
                f"estimated {ctx.gpu_norm} compute {est_compute_sec / 60.0:.1f} min + "
                f"staging/teardown {setup_and_export_sec / 60.0:.1f} min exceeds "
                f"{session_cap_min:.1f} min session cap; at most {pkt_cap['fit_session_cap']} "
                f"candidates fit {session_cap_min:.1f} min ({pkt_cap['fit_effective']} also "
                f"fitting the {colab_transfer_policy.DEFAULT_TRANSFER_SECONDS:.1f} s upload cap; "
                f"all-small at defaults fits {s_cap['fit_effective']}, "
                f"all-medium at defaults fits {m_cap['fit_effective']})."
            )
        raise ValueError(
            f"Refusing eval packet ({eff_eval_mode}, {len(policy_specs)} policies, "
            f"batch_worlds={batch_worlds}, {sizing_arithmetic}): estimated {ctx.gpu_norm} compute "
            f"{est_compute_sec / 60.0:.1f} min + staging/teardown {setup_and_export_sec / 60.0:.1f} min "
            f"exceeds {session_cap_min:.1f} min session cap. Lower --num-envs / ticks, pass fewer "
            "policies, or raise --eval-session-minutes."
        )

    base_total_min = max(4.0, (est_compute_sec + setup_and_export_sec) / 60.0)
    est_minutes = round(base_total_min, 2)
    # supervisor.validate() enforces timeout_minutes >= 2 * est_minutes (Owner's 2x rule)
    timeout_minutes = round(max(15.0, 2.0 * est_minutes), 2)
    headroom_sec = min(360.0, max(120.0, setup_and_export_sec))
    wall_guard_sec = int(max(60.0, timeout_minutes * 60.0 - headroom_sec))

    eff_amp = True if opts.amp is None else bool(opts.amp)
    eff_cl = True if opts.use_channels_last is None else bool(opts.use_channels_last)
    eff_fast = True if opts.fast_learner_ops is None else bool(opts.fast_learner_ops)
    batched_pairs_flag = eff_eval_mode == "round-robin" and not bool(opts.eval_sequential_pairs)

    extra_env: dict[str, str] = {
        "V3_PPO_EVAL_MODE": eff_eval_mode,
        "V3_PPO_EVAL_NUM_ENVS": str(eff_envs),
        "V3_PPO_EVAL_BURNIN_TICKS": str(eff_burnin),
        "V3_PPO_EVAL_WINDOW_TICKS": str(eff_window),
        "V3_PPO_EVAL_MASS_BIN_TICKS": str(int(opts.eval_mass_bin_ticks)),
        "V3_PPO_EVAL_WORLD_AGE_BINS": str(int(opts.eval_world_age_bins)),
        "V3_PPO_EVAL_SEED": str(int(opts.eval_seed)),
        "V3_PPO_AMP": "1" if eff_amp else "0",
        "V3_PPO_CHANNELS_LAST": "1" if eff_cl else "0",
        "V3_PPO_MAX_WALL_SECONDS": str(wall_guard_sec),
    }
    if opts.learner_slots is not None:
        extra_env["V3_PPO_EVAL_LEARNER_SLOTS"] = str(int(opts.learner_slots))
    elif eff_eval_mode == "anchor":
        extra_env["V3_PPO_EVAL_LEARNER_SLOTS"] = str(eff_learner_slots)
    if opts.opponent_policy is not None:
        extra_env["V3_PPO_EVAL_OPPONENT_POLICY"] = str(opts.opponent_policy)
    if opts.eval_greedy:
        extra_env["V3_PPO_EVAL_GREEDY"] = "1"
    if eff_eval_mode == "round-robin":
        if opts.eval_sequential_pairs:
            extra_env["V3_PPO_EVAL_SEQUENTIAL_PAIRS"] = "1"
        else:
            extra_env["V3_PPO_EVAL_MAX_PAIRS_PER_BATCH"] = str(eff_pairs_per_batch)
    elif eff_eval_mode == "anchor":
        assert anchor_ledger_spec is not None
        scripted_ops = anchor_ledger_spec["scripted_opponents"]
        extra_env["V3_PPO_ANCHOR_SCRIPTED"] = ",".join(scripted_ops) if scripted_ops else "none"
    if opts.compile_model:
        extra_env["V3_PPO_COMPILE"] = "1"
    if not eff_fast:
        extra_env["V3_PPO_FAST_LEARNER_OPS"] = "0"

    eval_plan = {
        "eval_mode": eff_eval_mode,
        "batched_pairs": batched_pairs_flag,
        "num_policies": len(policy_specs),
        "total_pairs": total_pairs,
        "reused_pairs": total_pairs - remaining_pairs,
        "remaining_pairs": remaining_pairs,
        "num_envs": eff_envs,
        "pairs_per_batch": eff_pairs_per_batch,
        "batch_worlds": batch_worlds,
        "num_batches": num_batches,
        "burnin_ticks": eff_burnin,
        "window_ticks": eff_window,
        "ticks_per_match": ticks_per_match,
        "ticks_per_batch": ticks_per_batch,
        "total_sim_ticks": total_sim_ticks,
        "agent_steps_per_match": steps_per_match,
        "total_agent_steps": remaining_pairs * steps_per_match,
        "timing_source": timing_source,
        "measured_or_assumed_ms_per_tick": round(base_ms_per_tick, 4),
        "timing_safety_factor": safety_factor,
        "effective_ms_per_tick": round(effective_ms_per_tick, 4),
        "raw_compute_seconds": round(raw_compute_sec, 2),
        "estimated_compute_seconds": round(est_compute_sec, 2),
        "estimated_compute_minutes": round(est_compute_sec / 60.0, 2),
        "estimated_staging_and_teardown_seconds": round(setup_and_export_sec, 2),
        "sizing_arithmetic": sizing_arithmetic,
        "raw_checkpoints_total_bytes": raw_total_bytes,
        "estimated_staged_inputs_bytes": est_staged_bytes,
        "weights_only_staged": use_weights_only,
        "session_cap_minutes": session_cap_min,
    }
    if anchor_ledger_spec is not None:
        eval_plan["num_candidates"] = len(anchor_ledger_spec["candidate_bundle_indices"])
        eval_plan["reused_candidates"] = total_pairs - remaining_pairs
        eval_plan["remaining_candidates"] = remaining_pairs
        eval_plan["num_reference_policies"] = len(anchor_ledger_spec["reference_bundle_indices"])
        eval_plan["scripted_opponents"] = list(anchor_ledger_spec["scripted_opponents"])
        eval_plan["learner_slots"] = eff_learner_slots
    if anchor_capacity is not None:
        eval_plan["anchor_capacity"] = anchor_capacity
    if ctx.canary_eval_smoke is not None:
        eval_plan["canary_eval_smoke"] = dict(ctx.canary_eval_smoke)

    return _SizedModePlan(
        job_id=f"v3-ppo-eval-{ctx.gpu_slug}-{opts.attempt}",
        est_minutes=est_minutes,
        timeout_minutes=timeout_minutes,
        base_sec=est_minutes * 60.0,
        eff_sec=timeout_minutes * 60.0,
        max_wall_sec_int=wall_guard_sec,
        extra_env=extra_env,
        required_exports=list(EVAL_REQUIRED_EXPORTS),
        resume_fields={
            "resume": {
                "unit_kind": "eval_unit",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            }
        },
        canary_block=ctx.canary_block,
        memory_estimate=memory_estimate,
        eval_plan=eval_plan,
        eval_policy_specs=tuple({**sp, "weights_only": use_weights_only} for sp in policy_specs),
        anchor_ledger_spec=anchor_ledger_spec,
        anchor_capacity=anchor_capacity,
    )


def _stage_size_and_memory_checks(
    opts: PreparePacketOptions,
    ctx: _ValidatedPacketContext,
) -> _SizedModePlan:
    """Stage 2: Resolve mode configuration, forecasted horizon, and GPU VRAM budget."""
    if opts.mode == "canary":
        return _size_canary_mode(opts, ctx)
    if opts.mode == "ruler":
        return _size_ruler_mode(opts, ctx)
    if opts.mode == "eval":
        return _size_eval_mode(opts, ctx)
    return _size_train_mode(opts, ctx)


def _stage_export_caps(
    opts: PreparePacketOptions,
    ctx: _ValidatedPacketContext,
    sized: _SizedModePlan,
) -> _CappedModePlan:
    """Stage 3: Validate per-file and total-session export byte caps."""
    if opts.mode == "canary":
        return _CappedModePlan(
            sized=sized,
            extra_env=dict(sized.extra_env),
            max_export_bytes=320 * 1024 * 1024,
        )
    if opts.mode == "ruler":
        return _CappedModePlan(
            sized=sized,
            extra_env=dict(sized.extra_env),
            max_export_bytes=128 * 1024 * 1024,
        )
    if opts.mode == "eval":
        return _CappedModePlan(
            sized=sized,
            extra_env=dict(sized.extra_env),
            max_export_bytes=128 * 1024 * 1024,
        )

    assert sized.train_cfg is not None
    cfg_obj = sized.train_cfg
    eff_world_bytes = _resolve_measured_world_bytes(opts, cfg_obj)

    if opts.ppo_config is None or not opts.explicit_total_updates:
        cfg_obj = replace(
            cfg_obj,
            total_updates=max(int(cfg_obj.total_updates), UNLIMITED_HOURS_TOTAL_UPDATES),
        )

    expected_updates_for_cap: int | None = None
    if sized.training_plan is not None:
        expected_updates_for_cap = int(sized.training_plan["expected_updates_in_session"])
        if opts.explicit_total_updates and int(cfg_obj.total_updates) > 0:
            expected_updates_for_cap = min(expected_updates_for_cap, int(cfg_obj.total_updates))
    elif opts.explicit_total_updates and int(cfg_obj.total_updates) > 0:
        expected_updates_for_cap = int(cfg_obj.total_updates)
    elif (
        int(cfg_obj.checkpoint_every_updates) > 0 or int(cfg_obj.full_checkpoint_every_updates) > 0
    ):
        steps_per_upd = int(cfg_obj.rollout_len) * int(cfg_obj.num_envs) * int(cfg_obj.num_snakes)
        fast_sec_per_upd = max(1.0, float(steps_per_upd) / 50_000.0)
        expected_updates_for_cap = max(
            1, int(math.ceil(float(sized.max_wall_sec_int) / fast_sec_per_upd))
        )

    checkpoint_export_caps = validate_checkpoint_export_caps(
        cfg_obj,
        ctx.gpu_norm,
        max_wall_seconds=sized.eff_sec,
        expected_updates=expected_updates_for_cap,
        measured_packed_bytes_per_world=eff_world_bytes,
    )

    canary_eval_wall_sec = _scale_train_canary_eval_wall_seconds(
        cfg_obj,
        (
            float(ctx.canary_eval_metrics["wall_seconds"])
            if ctx.canary_eval_metrics is not None
            else None
        ),
    )
    staged_input_bytes = (
        int(ctx.prior.chosen_checkpoint["size_bytes"])
        if ctx.prior.kind == "checkpoint" and ctx.prior.chosen_checkpoint is not None
        else 0
    )
    wall_headroom = validate_train_wall_headroom(
        cfg_obj,
        ctx.gpu_norm,
        session_seconds=sized.eff_sec,
        canary_eval_wall_seconds=canary_eval_wall_sec,
        measured_packed_bytes_per_world=eff_world_bytes,
        staged_input_bytes=staged_input_bytes,
    )
    wall_headroom_by_mode = estimate_train_wall_headroom_by_mode(
        cfg_obj,
        gpu=ctx.gpu_norm,
        session_seconds=sized.eff_sec,
        canary_eval_wall_seconds=canary_eval_wall_sec,
        measured_packed_bytes_per_world=eff_world_bytes,
        staged_input_bytes=staged_input_bytes,
    )

    if cfg_obj.full_exports == "periodic":
        print(
            "WARNING [V3_PPO_PERIODIC_FULL_EXPORTS]: --full-exports periodic is enabled! "
            "Periodic simulator FULL exports have KILLED 8-hour L4 and A100 runs mid-training "
            "on this Colab CLI transport by exhausting the Mac supervisor transfer-time cap "
            "(UPLOAD_CAP), losing all training progress after the last LIGHT checkpoint. "
            "Prefer the default --full-exports none (LIGHT checkpoints only; resumes cleanly "
            "with fresh worlds) or --full-exports final.",
            file=sys.stderr,
            flush=True,
        )
    if sized.minibatch_remainder is not None:
        print(
            format_minibatch_remainder_warning(sized.minibatch_remainder),
            file=sys.stderr,
            flush=True,
        )
    print(
        f"recipe overrides: {format_recipe_overrides(sized.recipe_overrides)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"V3_PPO_HEADROOM {json.dumps(wall_headroom, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "V3_PPO_HEADROOM_BY_MODE "
        + json.dumps(
            {
                m: {
                    "headroom_seconds": round(float(d["headroom_seconds"]), 2),
                    "headroom_minutes": d["headroom_minutes"],
                    "max_wall_seconds": d["max_wall_seconds"],
                    "final_export_bytes": d["final_export_bytes"],
                    "fits_max_headroom_fraction": d["fits_max_headroom_fraction"],
                }
                for m, d in wall_headroom_by_mode.items()
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    if sized.training_plan is not None:
        print(
            f"V3_PPO_PLAN {json.dumps(sized.training_plan, sort_keys=True)}",
            file=sys.stderr,
            flush=True,
        )

    extra_env = dict(sized.extra_env)
    extra_env["V3_PPO_AMP"] = "1" if bool(cfg_obj.amp) else "0"
    extra_env["V3_PPO_CHANNELS_LAST"] = "1" if bool(cfg_obj.use_channels_last) else "0"
    if bool(cfg_obj.compile_model):
        extra_env["V3_PPO_COMPILE"] = "1"
    if not bool(cfg_obj.fast_learner_ops):
        extra_env["V3_PPO_FAST_LEARNER_OPS"] = "0"
    if bool(opts.allow_recipe_change):
        extra_env["V3_PPO_ALLOW_RECIPE_CHANGE"] = "1"
    extra_env["V3_PPO_MAX_SESSION_EXPORT_BYTES"] = str(
        int(checkpoint_export_caps["session_total_export_cap_bytes"])
    )
    staged_cfg_dict = cfg_obj.to_dict()
    if cfg_obj.full_exports == "none":
        staged_cfg_dict.pop("max_periodic_full_exports", None)
    extra_env["V3_PPO_CONFIG_JSON"] = json.dumps(staged_cfg_dict, sort_keys=True)

    return _CappedModePlan(
        sized=replace(
            sized,
            train_cfg=cfg_obj,
            wall_headroom=wall_headroom,
            wall_headroom_by_mode=wall_headroom_by_mode,
        ),
        extra_env=extra_env,
        max_export_bytes=int(checkpoint_export_caps["per_file_max_export_bytes"]),
        checkpoint_export_caps=checkpoint_export_caps,
    )


def _stage_inputs(
    ctx: _ValidatedPacketContext,
    capped: _CappedModePlan,
    pkt_dir: Path,
) -> _StagedInputs:
    """Stage 4: Stage prior continuation artifacts, write verified ledger, and check input sizes."""
    pkt_inputs_dir = pkt_dir / "inputs"
    pkt_inputs_dir.mkdir(parents=True, exist_ok=True)
    unit_kind = (
        "eval_unit"
        if capped.sized.eval_policy_specs
        else (
            str(capped.sized.resume_fields["resume"]["unit_kind"])
            if "resume" in capped.sized.resume_fields
            else "checkpoint"
        )
    )
    ledger: dict[str, Any] = {
        "schema": "v3-ppo-verified-ledger/v1",
        "namespace": V3_PPO_NAMESPACE,
        "unit_kind": unit_kind,
        "skip_verified": True,
        "verified_units": [],
    }
    inputs_list: list[dict[str, str]] = []

    if capped.sized.eval_policy_specs:
        bundle_rel_dest = "policies_bundle.eval.pt"
        dst_bundle = pkt_inputs_dir / bundle_rel_dest
        src_ckpt_paths = [
            Path(sp.get("input_path") or sp["src_path"]) for sp in capped.sized.eval_policy_specs
        ]
        bundle_weights_only = all(bool(sp["weights_only"]) for sp in capped.sized.eval_policy_specs)
        bundle_summary = write_eval_policy_bundle(
            src_ckpt_paths,
            dst_bundle,
            weights_only=bundle_weights_only,
        )
        bundle_sha = str(bundle_summary["sha256"])
        bundle_size = int(bundle_summary["size_bytes"])
        inputs_list.append({"path": str(dst_bundle), "dest": bundle_rel_dest, "sha256": bundle_sha})
        ledger["eval_policy_bundle"] = {
            "schema": V3_EVAL_POLICY_BUNDLE_SCHEMA,
            "relpath": bundle_rel_dest,
            "sha256": bundle_sha,
            "size_bytes": bundle_size,
            "num_policies": len(capped.sized.eval_policy_specs),
            "weights_only": bundle_weights_only,
        }
        ledger["verified_units"].append(
            {
                "unit_id": "policy_bundle",
                "relpath": bundle_rel_dest,
                "sha256": bundle_sha,
                "size_bytes": bundle_size,
            }
        )
        eval_entries: list[dict[str, Any]] = []
        for sp, pol_meta in zip(capped.sized.eval_policy_specs, bundle_summary["policies"]):
            eval_entries.append(
                {
                    "index": int(sp["index"]),
                    "label": str(sp["label"]),
                    "relpath": bundle_rel_dest,
                    "bundle_policy_index": int(pol_meta["policy_index"]),
                    "sha256": bundle_sha,
                    "policy_sha256": str(pol_meta["policy_sha256"]),
                    "source_checkpoint_path": str(sp["src_path"]),
                    "source_checkpoint_sha256": str(sp["src_sha256"]),
                    "raw_checkpoint_size_bytes": int(sp["raw_size_bytes"]),
                    "weights_only": bool(sp["weights_only"]),
                }
            )
        ledger["eval_policies"] = eval_entries
        if capped.sized.anchor_ledger_spec is not None:
            ledger["anchor_eval"] = dict(capped.sized.anchor_ledger_spec)

    if ctx.prior.kind == "eval_candidates":
        for cand_json in ctx.prior.eval_candidate_paths:
            rel_dest = f"candidates/{cand_json.name}"
            dst_file = pkt_inputs_dir / "candidates" / cand_json.name
            c_sha = write_create_only_artifact(
                dst_file, lambda tmp, src=cand_json: shutil.copyfile(src, tmp)
            )
            inputs_list.append({"path": str(dst_file), "dest": rel_dest, "sha256": c_sha})
            ledger["verified_units"].append(
                {
                    "unit_id": f"cand:{cand_json.stem}",
                    "relpath": rel_dest,
                    "sha256": c_sha,
                    "size_bytes": int(dst_file.stat().st_size),
                }
            )
    elif ctx.prior.kind == "eval_pairs":
        for pair_json in ctx.prior.eval_pair_paths:
            rel_dest = f"pairs/{pair_json.name}"
            dst_file = pkt_inputs_dir / "pairs" / pair_json.name
            p_sha = write_create_only_artifact(
                dst_file, lambda tmp, src=pair_json: shutil.copyfile(src, tmp)
            )
            inputs_list.append({"path": str(dst_file), "dest": rel_dest, "sha256": p_sha})
            ledger["verified_units"].append(
                {
                    "unit_id": f"pair:{pair_json.stem}",
                    "relpath": rel_dest,
                    "sha256": p_sha,
                    "size_bytes": int(dst_file.stat().st_size),
                }
            )
    elif ctx.prior.kind == "ruler_cells":
        for cell_json in ctx.prior.ruler_cell_paths:
            rel_dest = f"cells/{cell_json.name}"
            dst_file = pkt_inputs_dir / "cells" / cell_json.name
            c_sha = write_create_only_artifact(
                dst_file, lambda tmp, src=cell_json: shutil.copyfile(src, tmp)
            )
            inputs_list.append({"path": str(dst_file), "dest": rel_dest, "sha256": c_sha})
            ledger["verified_units"].append(
                {
                    "unit_id": f"cell:{cell_json.stem}",
                    "relpath": rel_dest,
                    "sha256": c_sha,
                    "size_bytes": int(dst_file.stat().st_size),
                }
            )
    elif ctx.prior.kind == "checkpoint":
        assert ctx.prior.chosen_checkpoint is not None
        chosen = ctx.prior.chosen_checkpoint
        primary_src = Path(chosen["path"])
        dst_primary = pkt_inputs_dir / primary_src.name
        primary_sha = write_create_only_artifact(
            dst_primary, lambda tmp, src=primary_src: shutil.copyfile(src, tmp)
        )
        inputs_list.append(
            {"path": str(dst_primary), "dest": primary_src.name, "sha256": primary_sha}
        )

        companion_entries: list[dict[str, Any]] = []
        for comp_src in chosen.get("files", [])[1:]:
            c_src = Path(comp_src)
            dst_comp = pkt_inputs_dir / c_src.name
            c_sha = write_create_only_artifact(
                dst_comp, lambda tmp, src=c_src: shutil.copyfile(src, tmp)
            )
            c_size = int(dst_comp.stat().st_size)
            inputs_list.append({"path": str(dst_comp), "dest": c_src.name, "sha256": c_sha})
            comp_rec = {"relpath": c_src.name, "sha256": c_sha, "size_bytes": c_size}
            companion_entries.append(comp_rec)
            ledger["verified_units"].append({"unit_id": f"staged:{c_src.stem}", **comp_rec})

        entry: dict[str, Any] = {
            "relpath": primary_src.name,
            "checkpoint_kind": str(chosen["kind"]),
            "sha256": primary_sha,
            "size_bytes": int(dst_primary.stat().st_size),
        }
        if companion_entries:
            entry["companion_files"] = companion_entries

        ledger["resume_checkpoint"] = entry
        ledger["verified_units"].append(
            {
                "unit_id": f"staged:{primary_src.stem}",
                "relpath": primary_src.name,
                "sha256": primary_sha,
                "size_bytes": int(dst_primary.stat().st_size),
            }
        )

    ledger_path = pkt_inputs_dir / "verified-ledger.json"
    ledger_bytes = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ledger_sha = write_create_only_artifact(ledger_path, ledger_bytes)
    _write_json_create_only(pkt_dir / "verified-ledger.json", ledger)
    inputs_list.append(
        {"path": str(ledger_path), "dest": "verified-ledger.json", "sha256": ledger_sha}
    )

    max_export_bytes = int(capped.max_export_bytes)
    largest_input_bytes = max(
        (int(Path(inp["path"]).stat().st_size) for inp in inputs_list),
        default=0,
    )
    if largest_input_bytes > MAX_SUPERVISOR_EXPORT_BYTES:
        raise ValueError(
            f"Staged input file ({largest_input_bytes / 2**30:.2f} GiB) exceeds supervisor "
            f"2.00 GiB per-file limit."
        )
    if largest_input_bytes > max_export_bytes:
        max_export_bytes = min(
            MAX_SUPERVISOR_EXPORT_BYTES, int(math.ceil(largest_input_bytes * 1.10))
        )

    return _StagedInputs(
        inputs_list=inputs_list,
        ledger=ledger,
        max_export_bytes=max_export_bytes,
    )


def _estimate_eval_packet_upload(
    file_sizes_bytes: Sequence[int],
    *,
    is_canary: bool,
) -> dict[str, Any]:
    """Estimate eval/canary upload time from staged file sizes & count via `colab_cli` caps."""
    transfer_cap = (
        float(colab_canary.MAX_UPLOAD_SECONDS)
        if is_canary
        else float(colab_transfer_policy.DEFAULT_TRANSFER_SECONDS)
    )
    return estimate_eval_upload_time(
        file_sizes_bytes,
        transfer_cap_seconds=transfer_cap,
        upload_chunk_bytes=int(colab_transfers.UPLOAD_CHUNK_BYTES),
        staging_control_seconds=float(colab_policy.STAGING_CONTROL_SECONDS),
        staging_recovery_seconds=float(colab_policy.STAGING_RECOVERY_SECONDS),
        backoff_first_seconds=float(colab_policy.BACKOFF_SECONDS[0]),
        control_hang_prob=float(EVAL_UPLOAD_CONTROL_HANG_PROB),
        chunk_rate_bytes_per_sec=float(EVAL_UPLOAD_CHUNK_RATE_BYTES_PER_SEC),
    )


def _compute_eval_bundle_fitting_policies(
    ledger: Mapping[str, Any],
    eval_policy_specs: Sequence[Mapping[str, Any]],
    eval_entries: Sequence[Mapping[str, Any]],
    *,
    other_file_sizes: Sequence[int],
    is_canary: bool,
) -> tuple[int, dict[str, Any], int, int]:
    """Return ``(fit_prefix_k, fit_prefix_est, fit_worst_k, max_policy_compressed_bytes)``."""

    def _ledger_bytes_for_subset(
        chosen_entries: Sequence[Mapping[str, Any]],
        bundle_size_bytes: int,
    ) -> int:
        non_bundle_units = [
            u
            for u in ledger.get("verified_units", [])
            if str(u.get("unit_id", "")) != "policy_bundle"
            and not str(u.get("unit_id", "")).startswith("policy:")
        ]
        if not chosen_entries:
            sub_ledger: dict[str, Any] = {
                "schema": ledger["schema"],
                "namespace": ledger["namespace"],
                "unit_kind": ledger["unit_kind"],
                "skip_verified": True,
                "verified_units": non_bundle_units,
            }
        else:
            dummy_sha = "0" * 64
            sub_bundle = {
                "schema": V3_EVAL_POLICY_BUNDLE_SCHEMA,
                "relpath": "policies_bundle.eval.pt",
                "sha256": dummy_sha,
                "size_bytes": int(bundle_size_bytes),
                "num_policies": len(chosen_entries),
                "weights_only": all(bool(e.get("weights_only", True)) for e in chosen_entries),
            }
            sub_ledger = {
                **ledger,
                "eval_policy_bundle": sub_bundle,
                "eval_policies": [
                    {**dict(e), "bundle_policy_index": idx, "sha256": dummy_sha}
                    for idx, e in enumerate(chosen_entries)
                ],
                "verified_units": [
                    {
                        "unit_id": "policy_bundle",
                        "relpath": "policies_bundle.eval.pt",
                        "sha256": dummy_sha,
                        "size_bytes": int(bundle_size_bytes),
                    },
                    *non_bundle_units,
                ],
            }
        return len((json.dumps(sub_ledger, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    zero_ledger_b = _ledger_bytes_for_subset((), 0)
    fit_prefix_k = 0
    fit_prefix_est = _estimate_eval_packet_upload(
        [zero_ledger_b, *other_file_sizes], is_canary=is_canary
    )
    if not eval_policy_specs:
        return 0, fit_prefix_est, 0, 0

    record_cache: dict[tuple[str, bool], Any] = {}
    records: list[Any] = []
    for sp in eval_policy_specs:
        in_p = str(Path(sp.get("input_path") or sp["src_path"]).resolve())
        wo = bool(sp["weights_only"])
        key = (in_p, wo)
        if key not in record_cache:
            record_cache[key] = extract_eval_policy_bundle_record(Path(in_p), weights_only=wo)
        records.append(record_cache[key])

    lo, hi = 1, len(records)
    while lo <= hi:
        mid = (lo + hi) // 2
        sub_bytes, _ = pack_eval_policy_bundle_records(records[:mid])
        sub_ledger_b = _ledger_bytes_for_subset(eval_entries[:mid], len(sub_bytes))
        est_mid = _estimate_eval_packet_upload(
            [len(sub_bytes), sub_ledger_b, *other_file_sizes], is_canary=is_canary
        )
        if bool(est_mid["fits_transfer_cap"]):
            fit_prefix_k = mid
            fit_prefix_est = est_mid
            lo = mid + 1
        else:
            hi = mid - 1

    standalone_by_pol_sha: dict[str, tuple[int, int, int]] = {}
    for rec in records:
        if rec.policy_sha256 not in standalone_by_pol_sha:
            one_bytes, one_sum = pack_eval_policy_bundle_records([rec])
            standalone_by_pol_sha[rec.policy_sha256] = (
                int(one_sum["compressed_nbytes"]),
                int(one_sum["header_nbytes"]),
                len(one_bytes),
            )
    max_comp_b, max_hdr_b, max_single_bundle_b = max(
        standalone_by_pol_sha.values(), key=lambda item: item[0]
    )

    fit_worst_k = 0
    for k in range(1, len(records) + 1):
        worst_bundle_b = 16 + k * max_hdr_b + k * max_comp_b
        worst_ledger_b = _ledger_bytes_for_subset(eval_entries[:k], worst_bundle_b)
        est_w = _estimate_eval_packet_upload(
            [worst_bundle_b, worst_ledger_b, *other_file_sizes], is_canary=is_canary
        )
        if bool(est_w["fits_transfer_cap"]):
            fit_worst_k = k
        else:
            break

    return fit_prefix_k, fit_prefix_est, fit_worst_k, max_single_bundle_b


def _stage_write_manifest(
    opts: PreparePacketOptions,
    ctx: _ValidatedPacketContext,
    capped: _CappedModePlan,
    staged: _StagedInputs,
    pkt_dir: Path,
) -> dict[str, Any]:
    """Stage 5: Build technical envelope, validate canary rules, write manifest, and summarize."""
    sized = capped.sized
    staged_file_sizes = [int(Path(inp["path"]).stat().st_size) for inp in staged.inputs_list]
    staged_upload_file_count = len(staged_file_sizes)
    staged_upload_bytes = int(sum(staged_file_sizes))
    is_eval_upload_packet = bool(
        opts.mode == "eval" or (opts.mode == "canary" and sized.eval_policy_specs)
    )
    eval_upload_est: dict[str, Any] | None = None

    if is_eval_upload_packet:
        eval_upload_est = _estimate_eval_packet_upload(
            staged_file_sizes,
            is_canary=(opts.mode == "canary"),
        )
        est_upload_seconds = float(eval_upload_est["estimated_upload_seconds"])
        print(
            f"V3_PPO_STAGED_UPLOAD bytes={staged_upload_bytes} "
            f"mb={staged_upload_bytes / 1_000_000.0:.2f} "
            f"files={staged_upload_file_count} "
            f"small_files={eval_upload_est['small_file_count']} "
            f"chunked_files={eval_upload_est['chunked_file_count']} "
            f"control_calls={eval_upload_est['staging_control_calls']} "
            f"clean_seconds={eval_upload_est['clean_transfer_seconds']:.2f} "
            f"hang_allowance_seconds={eval_upload_est['hang_allowance_seconds']:.2f} "
            f"est_upload_seconds={est_upload_seconds:.2f} "
            f"cap_seconds={eval_upload_est['transfer_cap_seconds']:.1f} "
            f"chunk_rate_mbps={eval_upload_est['chunk_rate_mbps']:.2f}",
            file=sys.stderr,
            flush=True,
        )
        if not bool(eval_upload_est["fits_transfer_cap"]):
            eval_entries = list(staged.ledger.get("eval_policies", []))
            other_sizes = [
                int(Path(inp["path"]).stat().st_size)
                for inp in staged.inputs_list
                if inp.get("dest") not in {"policies_bundle.eval.pt", "verified-ledger.json"}
            ]
            fit_prefix_k, fit_prefix_est, fit_worst_k, max_pol_b = (
                _compute_eval_bundle_fitting_policies(
                    staged.ledger,
                    sized.eval_policy_specs,
                    eval_entries,
                    other_file_sizes=other_sizes,
                    is_canary=(opts.mode == "canary"),
                )
            )
            cap_sec = float(eval_upload_est["transfer_cap_seconds"])
            fit_b = int(fit_prefix_est["total_bytes"])
            fit_f = int(fit_prefix_est["file_count"])
            fit_s = float(fit_prefix_est["estimated_upload_seconds"])
            if eval_entries:
                if fit_prefix_k == fit_worst_k:
                    fit_clause = (
                        f"at most {fit_prefix_k} of the {len(eval_entries)} policies would fit "
                        f"({fit_b:,} bytes across {fit_f} files, {fit_s:.2f} s <= {cap_sec:.1f} s)"
                    )
                else:
                    fit_clause = (
                        f"at most {fit_prefix_k} of the {len(eval_entries)} supplied policies in "
                        f"order would fit ({fit_b:,} bytes across {fit_f} files, "
                        f"{fit_s:.2f} s <= {cap_sec:.1f} s; {fit_worst_k} policies at the largest "
                        f"policy compressed size of {max_pol_b:,} bytes)"
                    )
            else:
                fit_clause = "0 policies would fit"
            pkt_label = "canary" if opts.mode == "canary" else "eval"
            raise ValueError(
                f"Refusing {pkt_label} packet: staged upload is {staged_upload_bytes:,} bytes "
                f"({staged_upload_bytes / 1_000_000.0:.2f} MB across {staged_upload_file_count} "
                f"files: {eval_upload_est['small_file_count']} <=2MiB + "
                f"{eval_upload_est['chunked_file_count']} >2MiB, "
                f"{eval_upload_est['staging_control_calls']} staging control calls), "
                f"estimated at {est_upload_seconds:.2f} s "
                f"(clean {eval_upload_est['clean_transfer_seconds']:.2f} s at "
                f"{eval_upload_est['chunk_rate_mbps']:.2f} MB/s + "
                f"hang allowance {eval_upload_est['hang_allowance_seconds']:.2f} s at "
                f"p_hang={eval_upload_est['control_hang_prob']:.3f}), "
                f"exceeding the {cap_sec:.1f} s Colab CLI transfer cap; "
                f"{fit_clause}."
            )
    else:
        est_upload_seconds = float(staged_upload_bytes) / float(
            ASSUMED_STAGED_INPUT_UPLOAD_RATE_BYTES_PER_SEC
        )
        print(
            f"V3_PPO_STAGED_UPLOAD bytes={staged_upload_bytes} "
            f"mb={staged_upload_bytes / 1_000_000.0:.2f} "
            f"files={staged_upload_file_count} "
            f"est_upload_seconds={est_upload_seconds:.2f} "
            f"assumed_rate_mbps={ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS:.1f}",
            file=sys.stderr,
            flush=True,
        )
        if opts.mode == "canary" and est_upload_seconds > CANARY_MAX_ESTIMATED_UPLOAD_SECONDS:
            raise ValueError(
                f"Refusing canary packet: staged upload is {staged_upload_bytes:,} bytes "
                f"({staged_upload_bytes / 1_000_000.0:.2f} MB across {staged_upload_file_count} files), "
                f"estimated at {est_upload_seconds:.2f} s at "
                f"ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS={ASSUMED_STAGED_INPUT_UPLOAD_RATE_MBPS:.1f} MB/s, "
                f"exceeding the {CANARY_MAX_ESTIMATED_UPLOAD_SECONDS:.1f} s canary upload cap "
                "(leaving margin under the 150 s transport canary cap for one hung call)."
            )
    rate_uph = GPU_FORECAST_RATES_UPH[ctx.gpu_norm]
    unit_cap = math.ceil(((sized.timeout_minutes * rate_uph) / 60.0 - 1e-12) * 10000.0) / 10000.0
    tech_envelope: dict[str, Any] = {
        "base_seconds": sized.base_sec,
        "effective_seconds": sized.eff_sec,
        "forecast_unit_cap": unit_cap,
        "mode": opts.mode,
        "gpu": ctx.gpu_norm,
    }
    if capped.checkpoint_export_caps is not None:
        tech_envelope["session_total_export_cap_bytes"] = int(
            capped.checkpoint_export_caps["session_total_export_cap_bytes"]
        )
        tech_envelope["worst_case_session_export_bytes"] = int(
            capped.checkpoint_export_caps["worst_case_session_export_bytes"]
        )
    if ctx.canary_eval_metrics is not None:
        tech_envelope["canary_eval_wall_seconds"] = float(ctx.canary_eval_metrics["wall_seconds"])
        if "peak_memory_bytes" in ctx.canary_eval_metrics:
            tech_envelope["canary_eval_peak_memory_bytes"] = int(
                ctx.canary_eval_metrics["peak_memory_bytes"]
            )
    if sized.wall_headroom is not None:
        tech_envelope["wall_headroom"] = sized.wall_headroom

    manifest: dict[str, Any] = {
        "job_id": sized.job_id[:64],
        "commit": ctx.resolved_commit,
        "script": FAMILY_SCRIPT,
        "device": "cuda",
        "source_paths": ctx.source_paths,
        "requirements": {
            "gpu": ctx.gpu_norm,
            "allowed_gpus": [ctx.gpu_norm],
            "min_vcpu": 2,
            "est_minutes": sized.est_minutes,
        },
        "timeout_minutes": sized.timeout_minutes,
        "output_dir": str(ctx.out_path),
        "max_export_bytes": int(staged.max_export_bytes),
        "inputs": staged.inputs_list,
        "env": {
            "COLAB_CLI_REQUIREMENTS": REQUIREMENTS_RELPATH,
            "V3_PPO_SOURCE_COMMIT": ctx.resolved_commit,
            "V3_PPO_MODE": opts.mode,
            "SNAKE_DQN_DEVICE": "cuda",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "MAX_JOBS": "4",
            **capped.extra_env,
        },
        "required_exports": sized.required_exports,
        "budget": {
            "rate_units_per_hour": rate_uph,
            "unit_cap": unit_cap,
            "reserve_units": 0,
            "other_assignments": {},
        },
        **sized.resume_fields,
        "requirements_file": REQUIREMENTS_RELPATH,
        "technical_envelope": tech_envelope,
    }
    if opts.allow_unmerged_commit:
        manifest["unmerged_dev_packet"] = True
    if sized.canary_block is not None:
        manifest["canary"] = sized.canary_block

    if opts.mode == "canary" or opts.canary_receipt is not None:
        colab_canary.check(manifest, ROOT)

    manifest_path = pkt_dir / "dispatch-manifest.json"
    _write_json_create_only(manifest_path, manifest)

    summary: dict[str, Any] = {
        "packet_dir": str(pkt_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "mode": opts.mode,
        "job_id": manifest["job_id"],
        "commit": ctx.resolved_commit,
        "gpu": ctx.gpu_norm,
        "timeout_minutes": sized.timeout_minutes,
        "est_minutes": sized.est_minutes,
        "max_wall_seconds": int(capped.extra_env["V3_PPO_MAX_WALL_SECONDS"]),
        "max_export_bytes": int(staged.max_export_bytes),
        "unit_cap": unit_cap,
    }
    if sized.memory_estimate is not None:
        summary["memory_estimate"] = sized.memory_estimate
    if capped.checkpoint_export_caps is not None:
        summary["checkpoint_export_caps"] = capped.checkpoint_export_caps
    if sized.wall_headroom is not None:
        summary["wall_headroom"] = sized.wall_headroom
    if sized.wall_headroom_by_mode is not None:
        summary["wall_headroom_by_mode"] = sized.wall_headroom_by_mode
    if sized.training_plan is not None:
        summary["training_plan"] = sized.training_plan
    if ctx.canary_eval_metrics is not None:
        summary["canary_eval_metrics"] = ctx.canary_eval_metrics
    if sized.recipe_overrides:
        summary["recipe_overrides"] = dict(sized.recipe_overrides)
    if sized.minibatch_remainder is not None:
        summary["minibatch_remainder"] = dict(sized.minibatch_remainder)
    if opts.allow_recipe_change:
        summary["allow_recipe_change"] = True
        summary["recipe_change"] = sized.recipe_change
    if sized.eval_plan is not None:
        assert eval_upload_est is not None
        bundle_info = staged.ledger.get("eval_policy_bundle") or {}
        staged_policy_bytes = int(bundle_info.get("size_bytes", 0))
        summary["eval_plan"] = {
            **sized.eval_plan,
            "staged_inputs_total_bytes": staged_upload_bytes,
            "staged_inputs_file_count": staged_upload_file_count,
            "staged_policies_bytes": staged_policy_bytes,
            "small_file_count": int(eval_upload_est["small_file_count"]),
            "chunked_file_count": int(eval_upload_est["chunked_file_count"]),
            "staging_control_calls": int(eval_upload_est["staging_control_calls"]),
            "clean_transfer_seconds": round(float(eval_upload_est["clean_transfer_seconds"]), 2),
            "hang_allowance_seconds": round(float(eval_upload_est["hang_allowance_seconds"]), 2),
            "estimated_upload_seconds": round(est_upload_seconds, 2),
            "transfer_cap_seconds": float(eval_upload_est["transfer_cap_seconds"]),
            "chunk_rate_mbps": float(eval_upload_est["chunk_rate_mbps"]),
            "control_hang_prob": float(eval_upload_est["control_hang_prob"]),
        }
    elif opts.mode == "canary" and sized.eval_policy_specs:
        assert eval_upload_est is not None
        bundle_info = staged.ledger.get("eval_policy_bundle") or {}
        staged_policy_bytes = int(bundle_info.get("size_bytes", 0))
        summary["eval_canary_upload"] = {
            "staged_inputs_total_bytes": staged_upload_bytes,
            "staged_inputs_file_count": staged_upload_file_count,
            "staged_policies_bytes": staged_policy_bytes,
            "small_file_count": int(eval_upload_est["small_file_count"]),
            "chunked_file_count": int(eval_upload_est["chunked_file_count"]),
            "staging_control_calls": int(eval_upload_est["staging_control_calls"]),
            "clean_transfer_seconds": round(float(eval_upload_est["clean_transfer_seconds"]), 2),
            "hang_allowance_seconds": round(float(eval_upload_est["hang_allowance_seconds"]), 2),
            "estimated_upload_seconds": round(est_upload_seconds, 2),
            "transfer_cap_seconds": float(eval_upload_est["transfer_cap_seconds"]),
            "chunk_rate_mbps": float(eval_upload_est["chunk_rate_mbps"]),
            "control_hang_prob": float(eval_upload_est["control_hang_prob"]),
        }
        if sized.anchor_capacity is not None:
            summary["eval_canary_upload"]["anchor_capacity"] = sized.anchor_capacity
    if sized.anchor_capacity is not None:
        summary["anchor_capacity"] = sized.anchor_capacity
    return summary


def prepare_packet_from_options(opts: PreparePacketOptions) -> dict[str, Any]:
    """Execute all five packet preparation stages on a ``PreparePacketOptions`` instance."""
    ctx = _stage_validate_options(opts)
    sized = _stage_size_and_memory_checks(opts, ctx)
    capped = _stage_export_caps(opts, ctx, sized)

    pkt_dir = ctx.pkt_dir
    pkt_dir.mkdir(parents=True, exist_ok=False)
    try:
        staged = _stage_inputs(ctx, capped, pkt_dir)
        return _stage_write_manifest(opts, ctx, capped, staged, pkt_dir)
    except BaseException:
        shutil.rmtree(pkt_dir, ignore_errors=True)
        raise


def prepare_v3_ppo_packet(
    packet_dir: Path | str,
    *,
    commit: str = "HEAD",
    mode: str = "canary",
    gpu: str = "L4",
    hours: float = 2.0,
    expected_steps_per_sec: float | None = None,
    net_size: NetSize | None = None,
    num_envs: int | None = None,
    num_snakes: int | None = None,
    arena_radius: float | None = None,
    learner_slots: int | None = None,
    opponent_policy: OpponentPolicy | None = None,
    rollout_len: int | None = None,
    ruler_envs: str | None = None,
    ruler_nets: str | None = None,
    ruler_learners: str | None = None,
    ruler_rollout_len: int = DEFAULT_RULER_ROLLOUT_LEN,
    ruler_warmup_updates: int = DEFAULT_RULER_WARMUP_UPDATES,
    ruler_timed_updates: int = DEFAULT_RULER_TIMED_UPDATES,
    ruler_base_minutes: float | None = None,
    amp: bool | None = None,
    use_channels_last: bool | None = None,
    fast_learner_ops: bool | None = None,
    include_fp32: bool = False,
    compile_model: bool = False,
    full_exports: FullExportMode | None = None,
    max_periodic_full_exports: int | None = None,
    light_resume_margin_minutes: float | None = None,
    eval_estimated_seconds: float | None = None,
    eval_every_minutes: float | None = None,
    eval_every_updates: int | None = None,
    eval_envs: int | None = None,
    eval_ticks: int | None = None,
    eval_opponents: str | None = None,
    eval_mode: str | None = None,
    eval_policies: Sequence[Path | str] = (),
    eval_round_robin: Sequence[Path | str] = (),
    eval_candidates: Sequence[Path | str] = (),
    eval_references: Sequence[Path | str] = (),
    eval_labels: Sequence[str] | None = None,
    eval_reference_labels: Sequence[str] | None = None,
    anchor_scripted: str | None = None,
    eval_burnin_ticks: int | None = None,
    eval_window_ticks: int | None = None,
    eval_mass_bin_ticks: int = DEFAULT_H2H_MASS_BIN_TICKS,
    eval_world_age_bins: int = DEFAULT_LONG_GAME_WORLD_AGE_BINS,
    eval_seed: int = DEFAULT_EVAL_SEED,
    eval_greedy: bool = False,
    eval_sequential_pairs: bool = False,
    eval_max_pairs_per_batch: int | None = None,
    assume_ms_per_tick: float | None = None,
    eval_session_minutes: float = EVAL_DEFAULT_SESSION_MINUTES,
    eval_weights_only: str = "true",
    output_dir: Path | str | None = None,
    prior_output_dir: Path | str | None = None,
    prior_kind: str = "auto",
    canary_receipt: Path | str | None = None,
    allow_missing_canary_receipt: bool = False,
    ppo_config: V3PPOConfig | None = None,
    recipe_json: str | Path | None = None,
    recipe_overrides: Mapping[str, Any] | None = None,
    allow_recipe_change: bool = False,
    explicit_total_updates: bool = False,
    explicit_eval_estimated_seconds: bool = False,
    measured_packed_bytes_per_world: int | None = None,
    measured_checkpoint_bytes: int | None = None,
    attempt: str = "attempt-01",
    allow_unmerged_commit: bool = False,
) -> dict[str, Any]:
    """Prepare a Colab CLI packet for Game V3.1 PPO (`canary`, `ruler`, or `train`).

    Refuses configurations whose estimated VRAM exceeds the target GPU memory budget (`22 GB` L4,
    `40 GB` A100, `80 GB` H100) OR whose single-world/per-file/total-session checkpoint exports
    exceed calibrated shard (`256 MiB` default), supervisor (`2 GiB`/file), or session export caps
    (`12 GiB` L4, `24 GiB` A100, `40 GiB` H100).
    """
    opts = PreparePacketOptions(
        packet_dir=Path(packet_dir),
        commit=commit,
        mode=mode,
        gpu=gpu,
        hours=hours,
        expected_steps_per_sec=expected_steps_per_sec,
        net_size=net_size,
        num_envs=num_envs,
        num_snakes=num_snakes,
        arena_radius=arena_radius,
        learner_slots=learner_slots,
        opponent_policy=opponent_policy,
        rollout_len=rollout_len,
        ruler_envs=ruler_envs,
        ruler_nets=ruler_nets,
        ruler_learners=ruler_learners,
        ruler_rollout_len=ruler_rollout_len,
        ruler_warmup_updates=ruler_warmup_updates,
        ruler_timed_updates=ruler_timed_updates,
        ruler_base_minutes=ruler_base_minutes,
        amp=amp,
        use_channels_last=use_channels_last,
        fast_learner_ops=fast_learner_ops,
        include_fp32=include_fp32,
        compile_model=compile_model,
        full_exports=full_exports,
        max_periodic_full_exports=max_periodic_full_exports,
        light_resume_margin_minutes=light_resume_margin_minutes,
        eval_estimated_seconds=eval_estimated_seconds,
        eval_every_minutes=eval_every_minutes,
        eval_every_updates=eval_every_updates,
        eval_envs=eval_envs,
        eval_ticks=eval_ticks,
        eval_opponents=eval_opponents,
        eval_mode=eval_mode,
        eval_policies=tuple(Path(p) for p in eval_policies),
        eval_round_robin=tuple(Path(p) for p in eval_round_robin),
        eval_candidates=tuple(Path(p) for p in eval_candidates),
        eval_references=tuple(Path(p) for p in eval_references),
        eval_labels=tuple(eval_labels) if eval_labels is not None else None,
        eval_reference_labels=(
            tuple(eval_reference_labels) if eval_reference_labels is not None else None
        ),
        anchor_scripted=anchor_scripted,
        eval_burnin_ticks=eval_burnin_ticks,
        eval_window_ticks=eval_window_ticks,
        eval_mass_bin_ticks=eval_mass_bin_ticks,
        eval_world_age_bins=eval_world_age_bins,
        eval_seed=eval_seed,
        eval_greedy=eval_greedy,
        eval_sequential_pairs=eval_sequential_pairs,
        eval_max_pairs_per_batch=eval_max_pairs_per_batch,
        assume_ms_per_tick=assume_ms_per_tick,
        eval_session_minutes=eval_session_minutes,
        eval_weights_only=eval_weights_only,
        output_dir=Path(output_dir) if output_dir is not None else None,
        prior_output_dir=Path(prior_output_dir) if prior_output_dir is not None else None,
        prior_kind=prior_kind,
        canary_receipt=Path(canary_receipt) if canary_receipt is not None else None,
        allow_missing_canary_receipt=allow_missing_canary_receipt,
        ppo_config=ppo_config,
        recipe_json=recipe_json,
        recipe_overrides=recipe_overrides,
        allow_recipe_change=allow_recipe_change,
        explicit_total_updates=explicit_total_updates,
        explicit_eval_estimated_seconds=explicit_eval_estimated_seconds,
        measured_packed_bytes_per_world=measured_packed_bytes_per_world,
        measured_checkpoint_bytes=measured_checkpoint_bytes,
        attempt=attempt,
        allow_unmerged_commit=allow_unmerged_commit,
    )
    return prepare_packet_from_options(opts)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m research.v3_ppo.prepare``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        "--packet-dir",
        dest="packet",
        type=Path,
        required=True,
        help="Fresh target packet directory.",
    )
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--mode", choices=PREPARE_MODES, default="canary")
    parser.add_argument("--gpu", choices=SUPPORTED_GPUS, default="L4")
    parser.add_argument(
        "--hours",
        type=float,
        default=2.0,
        help="Whole-session timeout in hours for --mode train (default: 2.0).",
    )
    parser.add_argument("--expected-steps-per-sec", type=float, default=None)
    parser.add_argument("--net-size", choices=VALID_NET_SIZES, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-snakes", type=int, default=None)
    parser.add_argument("--arena-radius", type=float, default=None)
    parser.add_argument("--learner-slots", type=int, default=None)
    parser.add_argument("--opponent-policy", choices=VALID_OPPONENT_POLICIES, default=None)
    parser.add_argument("--rollout-len", type=int, default=None)

    parser.add_argument("--ruler-envs", type=str, default=None)
    parser.add_argument("--ruler-nets", type=str, default=None)
    parser.add_argument("--ruler-learners", type=str, default=None)
    parser.add_argument("--ruler-rollout-len", type=int, default=DEFAULT_RULER_ROLLOUT_LEN)
    parser.add_argument("--ruler-warmup-updates", type=int, default=DEFAULT_RULER_WARMUP_UPDATES)
    parser.add_argument("--ruler-timed-updates", type=int, default=DEFAULT_RULER_TIMED_UPDATES)
    parser.add_argument("--ruler-base-minutes", type=float, default=None)

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

    parser.add_argument("--include-fp32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--full-exports",
        choices=VALID_FULL_EXPORT_MODES,
        default=None,
        help="Simulator FULL export policy for --mode train: none (default), final, or periodic.",
    )
    parser.add_argument("--max-periodic-full-exports", type=int, default=None)
    parser.add_argument("--light-resume-margin-minutes", type=float, default=None)
    parser.add_argument("--eval-estimated-seconds", type=float, default=None)
    parser.add_argument(
        "--eval-every-minutes",
        type=float,
        default=None,
        help="Wall-clock interval in minutes between in-training evaluations for --mode train.",
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=None,
        help="Optional PPO update cadence for in-training evaluations in --mode train.",
    )
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=None,
        help="Parallel world count for in-training evaluations in --mode train (default: 192).",
    )
    parser.add_argument(
        "--eval-ticks",
        type=int,
        default=None,
        help="Measurement window ticks per opponent in --mode train (default: 2000).",
    )
    parser.add_argument(
        "--eval-opponents",
        type=str,
        default=None,
        help=(
            "Comma-separated scripted opponent policies for --mode train in-training eval "
            "(default unchanged: food_seeking,random_safe)."
        ),
    )
    parser.add_argument("--eval-mode", choices=EVAL_MODES, default=None)
    parser.add_argument(
        "--eval-policies",
        "--policies",
        dest="eval_policies",
        nargs="+",
        type=Path,
        default=None,
        help="Checkpoints to stage for --mode eval (1 for long-game, 2..6 for head-to-head).",
    )
    parser.add_argument(
        "--eval-round-robin",
        "--round-robin",
        dest="eval_round_robin",
        nargs="+",
        type=Path,
        default=None,
        help="Directory, .txt/.json list, or checkpoints for --mode eval round-robin.",
    )
    parser.add_argument(
        "--candidates",
        "--eval-candidates",
        dest="eval_candidates",
        nargs="+",
        type=Path,
        default=None,
        help="Candidate checkpoint list file, directory, or checkpoints for --eval-mode anchor.",
    )
    parser.add_argument(
        "--references",
        "--eval-references",
        dest="eval_references",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Frozen reference checkpoint list file, directory, or checkpoints "
            "for --eval-mode anchor."
        ),
    )
    parser.add_argument(
        "--eval-labels", "--labels", dest="eval_labels", nargs="+", type=str, default=None
    )
    parser.add_argument(
        "--reference-labels",
        "--eval-reference-labels",
        dest="eval_reference_labels",
        nargs="+",
        type=str,
        default=None,
        help="Optional unique display labels matching --references in anchor mode.",
    )
    parser.add_argument(
        "--anchor-scripted",
        "--eval-scripted-opponents",
        "--scripted-opponents",
        dest="anchor_scripted",
        type=str,
        default=None,
        help=(
            "Comma-separated scripted opponent policies for --eval-mode anchor "
            "(default: cut_off,head_evader; pass 'none' for neural-only reference pool)."
        ),
    )
    parser.add_argument(
        "--eval-burnin-ticks", "--burnin-ticks", dest="eval_burnin_ticks", type=int, default=None
    )
    parser.add_argument(
        "--eval-window-ticks", "--window-ticks", dest="eval_window_ticks", type=int, default=None
    )
    parser.add_argument("--eval-mass-bin-ticks", type=int, default=DEFAULT_H2H_MASS_BIN_TICKS)
    parser.add_argument("--eval-world-age-bins", type=int, default=DEFAULT_LONG_GAME_WORLD_AGE_BINS)
    parser.add_argument(
        "--eval-seed", "--seed", dest="eval_seed", type=int, default=DEFAULT_EVAL_SEED
    )
    parser.add_argument("--eval-greedy", "--greedy", dest="eval_greedy", action="store_true")
    eval_batch_group = parser.add_mutually_exclusive_group()
    eval_batch_group.add_argument(
        "--eval-batched-pairs",
        "--batched-pairs",
        dest="eval_sequential_pairs",
        action="store_false",
        default=False,
        help="Run round-robin pairs side-by-side in one batched simulator (default).",
    )
    eval_batch_group.add_argument(
        "--eval-sequential-pairs",
        "--sequential-pairs",
        dest="eval_sequential_pairs",
        action="store_true",
        help="Fallback: run round-robin pairs sequentially one pair at a time.",
    )
    parser.add_argument(
        "--eval-max-pairs-per-batch",
        "--max-pairs-per-batch",
        dest="eval_max_pairs_per_batch",
        type=int,
        default=None,
        help="Optional cap on pairs per batched simulator run for --mode eval round-robin.",
    )
    parser.add_argument(
        "--assume-ms-per-tick",
        type=float,
        default=None,
        help=(
            "Explicit ms/tick assumption for --mode eval when no bound canary "
            "eval_canary_smoke.ms_per_tick measurement is available."
        ),
    )
    parser.add_argument("--eval-session-minutes", type=float, default=EVAL_DEFAULT_SESSION_MINUTES)
    parser.add_argument(
        "--eval-weights-only",
        choices=("auto", "true", "false"),
        default="true",
        help="Stage policy-only eval files (weights + eval/recipe config + provenance): true (default), auto, or false.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prior-output-dir", type=Path, default=None)
    parser.add_argument(
        "--prior-kind",
        choices=("auto", "checkpoint", "ruler_cells", "eval_pairs", "eval_candidates"),
        default="auto",
    )
    parser.add_argument("--canary-receipt", type=Path, default=None)
    parser.add_argument("--allow-missing-canary-receipt", action="store_true")
    parser.add_argument("--ppo-config-json", type=str, default=None)
    parser.add_argument(
        "--recipe-json",
        type=str,
        default=None,
        help=(
            "Inline JSON object or path to JSON file overriding allow-listed recipe fields "
            f"({', '.join(RECIPE_OVERRIDE_KEYS)}) for --mode train."
        ),
    )
    parser.add_argument(
        "--allow-recipe-change",
        action="store_true",
        help=(
            "Permit changing ONLY allow-listed recipe fields on a checkpoint continuation, "
            "logging V3_PPO_RECIPE_CHANGE at resume and recording recipe_change_history."
        ),
    )
    parser.add_argument("--measured-packed-bytes-per-world", type=int, default=None)
    parser.add_argument("--measured-checkpoint-bytes", type=int, default=None)
    parser.add_argument("--attempt", type=str, default="attempt-01")
    parser.add_argument("--allow-unmerged-commit", action="store_true")
    return parser


def options_from_cli_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> PreparePacketOptions:
    """Construct ``PreparePacketOptions`` from parsed CLI ``args``."""
    if args.mode in ("ruler", "train", "eval") and args.canary_receipt is None:
        if not args.allow_missing_canary_receipt:
            msg = (
                f"--mode {args.mode} requires --canary-receipt <path/to/receipt.json> "
                "(pass --allow-missing-canary-receipt only for offline schema inspection)"
            )
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)

    ppo_cfg: V3PPOConfig | None = None
    explicit_total_updates = False
    explicit_eval_estimated_seconds = False
    if args.ppo_config_json is not None:
        raw_source = args.ppo_config_json
        if raw_source.lstrip().startswith("{"):
            raw_dict = json.loads(raw_source)
        else:
            raw_dict = json.loads(Path(raw_source).read_text(encoding="utf-8"))
        if not isinstance(raw_dict, dict):
            raise ValueError("--ppo-config-json root must be a JSON object")
        if "full_exports" in raw_dict:
            raise ValueError(
                "--ppo-config-json cannot set 'full_exports'; simulator FULL export mode is "
                "selected only on the command line via --full-exports {none,final,periodic} "
                "(or --max-periodic-full-exports)."
            )
        explicit_total_updates = "total_updates" in raw_dict
        explicit_eval_estimated_seconds = "eval_estimated_seconds" in raw_dict
        ppo_cfg = V3PPOConfig.from_dict(raw_dict)

    return PreparePacketOptions(
        packet_dir=Path(args.packet),
        commit=args.commit,
        mode=args.mode,
        gpu=args.gpu,
        hours=args.hours,
        expected_steps_per_sec=args.expected_steps_per_sec,
        net_size=args.net_size,
        num_envs=args.num_envs,
        num_snakes=args.num_snakes,
        arena_radius=args.arena_radius,
        learner_slots=args.learner_slots,
        opponent_policy=args.opponent_policy,
        rollout_len=args.rollout_len,
        ruler_envs=args.ruler_envs,
        ruler_nets=args.ruler_nets,
        ruler_learners=args.ruler_learners,
        ruler_rollout_len=args.ruler_rollout_len,
        ruler_warmup_updates=args.ruler_warmup_updates,
        ruler_timed_updates=args.ruler_timed_updates,
        ruler_base_minutes=args.ruler_base_minutes,
        amp=args.amp,
        use_channels_last=args.use_channels_last,
        fast_learner_ops=args.fast_learner_ops,
        include_fp32=bool(args.include_fp32),
        compile_model=bool(args.compile),
        full_exports=args.full_exports,
        max_periodic_full_exports=args.max_periodic_full_exports,
        light_resume_margin_minutes=args.light_resume_margin_minutes,
        eval_estimated_seconds=args.eval_estimated_seconds,
        eval_every_minutes=args.eval_every_minutes,
        eval_every_updates=args.eval_every_updates,
        eval_envs=args.eval_envs,
        eval_ticks=args.eval_ticks,
        eval_opponents=args.eval_opponents,
        eval_mode=args.eval_mode,
        eval_policies=tuple(args.eval_policies or ()),
        eval_round_robin=tuple(args.eval_round_robin or ()),
        eval_candidates=tuple(args.eval_candidates or ()),
        eval_references=tuple(args.eval_references or ()),
        eval_labels=tuple(args.eval_labels) if args.eval_labels is not None else None,
        eval_reference_labels=(
            tuple(args.eval_reference_labels) if args.eval_reference_labels is not None else None
        ),
        anchor_scripted=args.anchor_scripted,
        eval_burnin_ticks=args.eval_burnin_ticks,
        eval_window_ticks=args.eval_window_ticks,
        eval_mass_bin_ticks=args.eval_mass_bin_ticks,
        eval_world_age_bins=args.eval_world_age_bins,
        eval_seed=args.eval_seed,
        eval_greedy=bool(args.eval_greedy),
        eval_sequential_pairs=bool(args.eval_sequential_pairs),
        eval_max_pairs_per_batch=args.eval_max_pairs_per_batch,
        assume_ms_per_tick=args.assume_ms_per_tick,
        eval_session_minutes=args.eval_session_minutes,
        eval_weights_only=args.eval_weights_only,
        output_dir=args.output_dir,
        prior_output_dir=args.prior_output_dir,
        prior_kind=args.prior_kind,
        canary_receipt=args.canary_receipt,
        allow_missing_canary_receipt=bool(args.allow_missing_canary_receipt),
        ppo_config=ppo_cfg,
        recipe_json=args.recipe_json,
        allow_recipe_change=bool(args.allow_recipe_change),
        explicit_total_updates=explicit_total_updates,
        explicit_eval_estimated_seconds=explicit_eval_estimated_seconds,
        measured_packed_bytes_per_world=args.measured_packed_bytes_per_world,
        measured_checkpoint_bytes=args.measured_checkpoint_bytes,
        attempt=args.attempt,
        allow_unmerged_commit=bool(args.allow_unmerged_commit),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m research.v3_ppo.prepare``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    opts = options_from_cli_args(args, parser)
    info = prepare_packet_from_options(opts)
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
