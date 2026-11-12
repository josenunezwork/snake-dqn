"""Standalone multi-policy head-to-head, round-robin, and mature long-game evaluator for Game V3.1.

Reuses the exact observation builder (:func:`research.game_v3.obs.build_obs`), action mask, and
policy architecture (:class:`research.v3_ppo.net.EgoMultiScaleNet`) from training so every loaded
policy observes and acts identically to :class:`research.v3_ppo.trainer.V3PPOTrainer`.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.arena_ppo.runtime import (
    SignalStopFlag,
    configure_determinism,
    ensure_cuda_initialized,
    normalize_device,
    phase_stamp,
    sha256_file,
    synchronize_device,
    verify_completed_artifact,
)
from research.game_v3.config import NUM_ACTIONS, NUM_TURN_LEVELS, GameV3Config
from research.game_v3.obs import (
    LearnerIndex,
    ObservationBatch,
    allocate_obs_batch,
    build_obs,
)
from research.game_v3.sim import FP_SHIFT, GameV3Sim, step_delta_fp_torch
from research.v3_ppo.buffer import assert_obs_batch_matches_layout
from research.v3_ppo.checkpoint_io import V3_PPO_LIGHT_CHECKPOINT_SCHEMA, verify_light_checkpoint
from research.v3_ppo.config import (
    _CUDA_RUNTIME_RESERVE_BYTES,
    _OBS_BUILD_TEMP_BYTES,
    GPU_VRAM_BUDGET_BYTES,
    V3_PPO_NAMESPACE,
    VALID_OPPONENT_POLICIES,
    NetSize,
    ObsLayout,
    OpponentPolicy,
    V3PPOConfig,
    default_actor_chunk_learners,
    estimate_actor_chunk_temp_bytes,
    estimate_sim_memory_bytes,
    extract_recipe_dict,
    get_net_preset,
    obs_layout_from_game_config,
    recipe_overrides_from_default,
)
from research.v3_ppo.eval import derive_world_seeds, preserve_rng_state, scripted_opponent_actions
from research.v3_ppo.exports import (
    artifact_or_sidecar_exists,
    write_create_only_artifact,
    write_create_only_json,
)
from research.v3_ppo.net import EgoMultiScaleNet, EgoMultiScaleNetConfig
from research.v3_ppo.trainer import StopRequestedError, seed_everything

ROOT = Path(__file__).resolve().parents[2]

V3_ARENA_EVAL_SCHEMA = "v3-ppo-arena-eval/v1"
V3_ARENA_EVAL_PAIR_SCHEMA = "v3-ppo-arena-eval-pair/v1"
V3_WEIGHTS_ONLY_EVAL_BUNDLE_SCHEMA = "v3-ppo-eval-weights-bundle/v1"

EVAL_MODES: tuple[str, ...] = ("head-to-head", "round-robin", "long-game")
EVAL_REQUIRED_EXPORTS: tuple[str, ...] = ("eval_result.json", "summary.md")
PAIR_REUSE_KEY_FIELDS: tuple[str, ...] = (
    "num_envs",
    "burnin_ticks",
    "window_ticks",
    "greedy",
    "seed",
    "policy_a_sha256",
    "policy_b_sha256",
)

# Horizon defaults:
# - Head-to-head / round-robin: 24 worlds * (2,000 burn-in + 2,500 window = 4,500 ticks) =
#   2.592M agent-steps/pair (24 is divisible by 2, 3, 4, 6 so cyclic seat rotation cancels
#   spawn-slot bias). Round-robin batches pairs side by side in a single simulator tick loop
#   (sized to fit GPU VRAM) and `prepare --mode eval` gates session wall time from a measured
#   eval-canary ms/tick with a 1.5x margin. World-to-world spread and 95% CI half-widths are
#   reported directly from each run's measured data.
# - Long-game: 32 worlds * (6,500 burn-in + 6,000 window = 12,500 ticks), setting burn-in at
#   6,500 ticks (>= 3,000) where strong policies (rwa-01 u1472) reach crowding equilibrium
#   (~350-385 time-averaged length).
DEFAULT_H2H_NUM_ENVS: int = 24
DEFAULT_H2H_BURNIN_TICKS: int = 2000
DEFAULT_H2H_WINDOW_TICKS: int = 2500
DEFAULT_H2H_MASS_BIN_TICKS: int = 250

DEFAULT_LONG_GAME_NUM_ENVS: int = 32
DEFAULT_LONG_GAME_BURNIN_TICKS: int = 6500
DEFAULT_LONG_GAME_WINDOW_TICKS: int = 6000
DEFAULT_LONG_GAME_WORLD_AGE_BINS: int = 6

DEFAULT_LIFE_AGE_EDGES: tuple[int, ...] = (100, 250, 500, 1000, 2000, 4000)
DEFAULT_VICTIM_LENGTH_EDGES: tuple[int, ...] = (15, 30, 60, 120, 250)

DEFAULT_EVAL_SEED: int = 20261112
DEFAULT_BOOTSTRAP_SAMPLES: int = 2000
DEFAULT_BT_PRIOR_WINS: float = 0.25
STUCK_MIN_LIFE_AGE_TICKS: int = 500
STUCK_SPAWN_LENGTH_TOLERANCE: int = 2

TURN_LEVEL_NAMES: tuple[str, ...] = (
    "hard_left",
    "soft_left",
    "straight",
    "soft_right",
    "hard_right",
)
DEATH_CAUSE_NAMES: tuple[str, ...] = (
    "head_into_body",
    "wall",
    "self",
    "head_on",
)


@dataclass(frozen=True)
class LoadedEvalPolicy:
    """Verified policy model and provenance metadata loaded from a checkpoint."""

    label: str
    checkpoint_path: Path
    source_checkpoint_path: str
    checkpoint_sha256: str
    source_checkpoint_sha256: str
    checkpoint_size_bytes: int
    checkpoint_schema: str
    net_size: NetSize
    parameter_count: int
    updates_completed: int
    agent_steps_collected: int
    learner_steps_collected: int
    actor_chunk_learners: int
    obs_layout: ObsLayout
    ckpt_config: V3PPOConfig
    training_config_dict: dict[str, Any]
    training_recipe: dict[str, Any]
    recipe_overrides: dict[str, Any]
    model: EgoMultiScaleNet

    def provenance_dict(self, *, policy_index: int) -> dict[str, Any]:
        """Serialize immutable policy provenance for result JSON exports."""
        return {
            "policy_index": int(policy_index),
            "label": self.label,
            "checkpoint_path": self.source_checkpoint_path,
            "loaded_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.source_checkpoint_sha256,
            "loaded_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": int(self.checkpoint_size_bytes),
            "checkpoint_schema": self.checkpoint_schema,
            "net_size": self.net_size,
            "parameter_count": int(self.parameter_count),
            "updates_completed": int(self.updates_completed),
            "agent_steps_collected": int(self.agent_steps_collected),
            "learner_steps_collected": int(self.learner_steps_collected),
            "obs_layout": self.obs_layout.to_dict(),
            "training_config": dict(self.training_config_dict),
            "training_recipe": dict(self.training_recipe),
            "recipe_overrides_from_default": dict(self.recipe_overrides),
            "recipe_change_history": [dict(x) for x in self.ckpt_config.recipe_change_history],
        }


def resolve_code_commit() -> str:
    """Return pinned Colab source commit if present, otherwise current git HEAD."""
    env_commit = os.environ.get("V3_PPO_SOURCE_COMMIT", "").strip()
    if env_commit:
        return env_commit
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return "unknown"


def _default_policy_label(path: Path) -> str:
    """Build a compact human-readable label from a checkpoint path."""
    stem = path.stem
    m_upd = re.search(r"u0*(\d+)", stem)
    upd_tag = f"u{m_upd.group(1)}" if m_upd else stem
    parts = path.parts
    for idx in range(len(parts) - 1, -1, -1):
        seg = parts[idx]
        if seg.startswith("v3-ppo-train-"):
            short = seg[len("v3-ppo-train-") :]
            short = re.sub(r"^(a100|l4|h100|t4|g4|cpu)-", "", short)
            return f"{short}:{upd_tag}"
    if path.parent.name and path.parent.name not in {
        "checkpoints",
        "exports",
        "inputs",
        "policies",
    }:
        return f"{path.parent.name}:{upd_tag}"
    return stem


def resolve_unique_labels(
    paths: Sequence[Path],
    explicit_labels: Sequence[str] | None = None,
) -> list[str]:
    """Resolve non-empty unique labels for ``paths``."""
    if explicit_labels is not None:
        if len(explicit_labels) != len(paths):
            raise ValueError(
                f"--labels length ({len(explicit_labels)}) must match number of policies ({len(paths)})"
            )
        cleaned = [str(lbl).strip() for lbl in explicit_labels]
        if any(not lbl for lbl in cleaned):
            raise ValueError("Policy labels must be non-empty strings")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"Policy labels must be unique, got {cleaned}")
        return cleaned

    raw = [_default_policy_label(Path(p)) for p in paths]
    seen: dict[str, int] = {}
    unique: list[str] = []
    counts: dict[str, int] = {}
    for lbl in raw:
        counts[lbl] = counts.get(lbl, 0) + 1
    for lbl in raw:
        if counts[lbl] == 1:
            unique.append(lbl)
        else:
            idx = seen.get(lbl, 0)
            seen[lbl] = idx + 1
            unique.append(f"{lbl}#{idx + 1}")
    return unique


def _load_checkpoint_payload(path: Path) -> tuple[dict[str, Any], str, int]:
    """Load and structurally validate a LIGHT checkpoint or weights-only eval bundle."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {resolved}")
    sidecar = Path(f"{resolved}.sha256")
    complete = Path(f"{resolved}.complete")
    if sidecar.is_file() and complete.is_file():
        digest = verify_completed_artifact(resolved)
    else:
        digest = sha256_file(resolved)
    size_bytes = int(resolved.stat().st_size)

    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {resolved} root payload must be dict, got {type(payload)}")

    schema = str(payload.get("schema", ""))
    if schema == V3_PPO_LIGHT_CHECKPOINT_SCHEMA:
        if sidecar.is_file() and complete.is_file():
            verify_light_checkpoint(resolved)
        else:
            for req in ("config", "model_state_dict"):
                if req not in payload:
                    raise ValueError(f"LIGHT checkpoint {resolved} missing key {req!r}")
    elif schema == V3_WEIGHTS_ONLY_EVAL_BUNDLE_SCHEMA:
        for req in ("config", "model_state_dict"):
            if req not in payload:
                raise ValueError(f"Eval bundle {resolved} missing key {req!r}")
    else:
        raise ValueError(
            f"Unsupported checkpoint schema {schema!r} in {resolved}; expected "
            f"{V3_PPO_LIGHT_CHECKPOINT_SCHEMA!r} or {V3_WEIGHTS_ONLY_EVAL_BUNDLE_SCHEMA!r}"
        )
    return payload, digest, size_bytes


def write_weights_only_eval_bundle(src_light_ckpt: Path, dst_path: Path) -> str:
    """Create a compact weights-only eval bundle preserving source provenance and sha256."""
    payload, src_sha, src_size = _load_checkpoint_payload(src_light_ckpt)
    raw_cfg = payload["config"]
    ckpt_cfg = V3PPOConfig.from_dict(raw_cfg)
    obs_layout_dict = (
        payload["obs_layout"]
        if payload.get("obs_layout") is not None
        else obs_layout_from_game_config(ckpt_cfg.game_config).to_dict()
    )
    counters = payload.get("counters") if isinstance(payload.get("counters"), Mapping) else {}
    bundle: dict[str, Any] = {
        "schema": V3_WEIGHTS_ONLY_EVAL_BUNDLE_SCHEMA,
        "namespace": payload.get("namespace", V3_PPO_NAMESPACE),
        "source_checkpoint_path": str(Path(src_light_ckpt).resolve()),
        "source_checkpoint_sha256": src_sha,
        "source_checkpoint_size_bytes": int(src_size),
        "source_checkpoint_schema": str(payload.get("schema", V3_PPO_LIGHT_CHECKPOINT_SCHEMA)),
        "config": raw_cfg,
        "obs_layout": obs_layout_dict,
        "net_size": str(ckpt_cfg.net_size),
        "counters": dict(counters),
        "updates_completed": int(
            counters.get("updates_completed", payload.get("updates_completed", 0))
        ),
        "world_steps_collected": int(
            counters.get("world_steps_collected", payload.get("world_steps_collected", 0))
        ),
        "agent_steps_collected": int(
            counters.get("agent_steps_collected", payload.get("agent_steps_collected", 0))
        ),
        "learner_steps_collected": int(
            counters.get("learner_steps_collected", payload.get("learner_steps_collected", 0))
        ),
        "model_state_dict": payload["model_state_dict"],
    }
    if "recipe_change_history" in payload:
        bundle["recipe_change_history"] = payload["recipe_change_history"]
    buf = io.BytesIO()
    torch.save(bundle, buf)
    return write_create_only_artifact(dst_path, buf.getvalue())


def load_eval_policy(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    label: str | None = None,
    amp: bool | None = None,
    use_channels_last: bool | None = None,
    fast_learner_ops: bool | None = None,
    compile_model: bool = False,
) -> LoadedEvalPolicy:
    """Load a LIGHT checkpoint using the exact trainer network configuration and weights."""
    dev = normalize_device(device)
    ensure_cuda_initialized(dev)
    path = Path(checkpoint_path).resolve()
    payload, loaded_sha, size_bytes = _load_checkpoint_payload(path)

    raw_cfg = payload["config"]
    if not isinstance(raw_cfg, Mapping):
        raise ValueError(f"Checkpoint {path} 'config' must be a mapping")
    ckpt_cfg = V3PPOConfig.from_dict(raw_cfg)
    expected_layout = obs_layout_from_game_config(ckpt_cfg.game_config)
    if payload.get("obs_layout") is not None:
        obs_layout = ObsLayout.from_dict(payload["obs_layout"])
        if obs_layout != expected_layout:
            raise ValueError(
                f"Checkpoint {path} obs_layout {obs_layout.to_dict()} disagrees with "
                f"game_config layout {expected_layout.to_dict()}"
            )
    else:
        obs_layout = expected_layout

    eff_amp = (bool(ckpt_cfg.amp) if amp is None else bool(amp)) and dev.type == "cuda"
    eff_cl = (
        bool(ckpt_cfg.use_channels_last) if use_channels_last is None else bool(use_channels_last)
    )
    eff_fast = (
        bool(ckpt_cfg.fast_learner_ops) if fast_learner_ops is None else bool(fast_learner_ops)
    )

    net_cfg = EgoMultiScaleNetConfig.from_size(
        ckpt_cfg.net_size,
        use_channels_last=eff_cl,
        use_bf16_autocast=eff_amp,
        fast_preprocess=eff_fast,
    )
    model = EgoMultiScaleNet(net_cfg).to(dev)
    model.validate_obs_layout(obs_layout)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if compile_model and hasattr(torch, "compile"):
        model.compile_execution_paths()

    actor_chunk = (
        int(ckpt_cfg.actor_chunk_learners)
        if ckpt_cfg.actor_chunk_learners is not None
        else default_actor_chunk_learners(ckpt_cfg.net_size)
    )
    src_path = str(payload.get("source_checkpoint_path") or path)
    src_sha = str(payload.get("source_checkpoint_sha256") or loaded_sha)
    src_size = int(payload.get("source_checkpoint_size_bytes") or size_bytes)
    resolved_label = (label.strip() if label is not None else "") or _default_policy_label(
        Path(src_path)
    )

    counters = payload.get("counters") if isinstance(payload.get("counters"), Mapping) else {}
    updates_completed = int(counters.get("updates_completed", payload.get("updates_completed", 0)))
    agent_steps_collected = int(
        counters.get("agent_steps_collected", payload.get("agent_steps_collected", 0))
    )
    learner_steps_collected = int(
        counters.get("learner_steps_collected", payload.get("learner_steps_collected", 0))
    )

    return LoadedEvalPolicy(
        label=resolved_label,
        checkpoint_path=path,
        source_checkpoint_path=src_path,
        checkpoint_sha256=loaded_sha,
        source_checkpoint_sha256=src_sha,
        checkpoint_size_bytes=src_size,
        checkpoint_schema=str(payload.get("schema", "")),
        net_size=ckpt_cfg.net_size,
        parameter_count=int(model.parameter_count()),
        updates_completed=updates_completed,
        agent_steps_collected=agent_steps_collected,
        learner_steps_collected=learner_steps_collected,
        actor_chunk_learners=int(actor_chunk),
        obs_layout=obs_layout,
        ckpt_config=ckpt_cfg,
        training_config_dict=dict(raw_cfg),
        training_recipe=extract_recipe_dict(ckpt_cfg),
        recipe_overrides=recipe_overrides_from_default(ckpt_cfg),
        model=model,
    )


def validate_compatible_policies(policies: Sequence[LoadedEvalPolicy]) -> GameV3Config:
    """Verify all loaded policies share identical GameV3Config and ObsLayout."""
    if not policies:
        raise ValueError("At least one policy is required")
    ref_game = policies[0].ckpt_config.game_config
    ref_layout = policies[0].obs_layout
    ref_dict = ref_game.to_dict()
    for pol in policies[1:]:
        if pol.obs_layout != ref_layout:
            raise ValueError(
                f"Policy {pol.label!r} obs_layout {pol.obs_layout.to_dict()} != "
                f"{policies[0].label!r} obs_layout {ref_layout.to_dict()}"
            )
        cur_dict = pol.ckpt_config.game_config.to_dict()
        if cur_dict != ref_dict:
            raise ValueError(
                f"Policy {pol.label!r} game_config differs from {policies[0].label!r}: "
                f"{cur_dict} vs {ref_dict}"
            )
    return GameV3Config.from_dict({**ref_dict, "max_life_ticks": 0})


def estimate_eval_memory(
    *,
    game_cfg: GameV3Config,
    net_sizes: Sequence[NetSize],
    eval_mode: str,
    num_envs_per_match: int,
    pairs_per_batch: int = 1,
    learner_slots: int | None = None,
    amp: bool = True,
    gpu: str | None = None,
) -> dict[str, Any]:
    """Estimate GPU VRAM footprint for standalone arena evaluation (including batched pairs)."""
    if eval_mode not in EVAL_MODES:
        raise ValueError(f"eval_mode must be one of {EVAL_MODES}, got {eval_mode!r}")
    if not net_sizes:
        raise ValueError("net_sizes must be non-empty")
    w_per_match = int(num_envs_per_match)
    if w_per_match < 1:
        raise ValueError(f"num_envs_per_match must be >= 1, got {w_per_match}")
    b_pairs = max(1, int(pairs_per_batch)) if eval_mode == "round-robin" else 1
    total_worlds = w_per_match * b_pairs
    n_snakes = int(game_cfg.num_snakes)
    layout = obs_layout_from_game_config(game_cfg)

    sim_mem = estimate_sim_memory_bytes(game_cfg, total_worlds)
    obs_rows_per_world = (
        n_snakes
        if (eval_mode != "long-game" or learner_slots is None)
        else max(1, min(n_snakes, int(learner_slots)))
    )
    total_obs_rows = total_worlds * obs_rows_per_world
    obs_batch_bytes = total_obs_rows * int(layout.obs_batch_bytes_per_row)

    policy_weights_bytes = sum(
        int(get_net_preset(sz).memory_model_param_count) * 4 for sz in net_sizes
    )

    if eval_mode == "long-game":
        max_pol_rows = total_obs_rows
    elif eval_mode == "head-to-head":
        max_pol_rows = int(math.ceil(total_obs_rows / float(max(1, len(net_sizes)))))
    else:
        max_pairs_for_one_policy = min(b_pairs, max(1, len(net_sizes) - 1))
        seats_per_pair_side = (w_per_match * n_snakes + 1) // 2
        max_pol_rows = max_pairs_for_one_policy * seats_per_pair_side

    fwd_workspace_bytes = 0
    for sz in set(net_sizes):
        preset = get_net_preset(sz)
        chunk_rows = min(max(1, max_pol_rows), int(preset.default_actor_chunk_learners))
        chunk_temp = estimate_actor_chunk_temp_bytes(
            actor_chunk_rows=chunk_rows,
            net_size=sz,
            amp=bool(amp),
            game_cfg=game_cfg,
        )
        gather_slice_bytes = max_pol_rows * int(layout.obs_batch_bytes_per_row)
        fwd_workspace_bytes = max(fwd_workspace_bytes, int(chunk_temp + gather_slice_bytes))

    accumulators_bytes = total_worlds * 4096
    obs_build_temp_bytes = int(_OBS_BUILD_TEMP_BYTES)
    cuda_runtime_reserve_bytes = int(_CUDA_RUNTIME_RESERVE_BYTES)

    estimated_peak_bytes = int(
        sim_mem["sim_peak_bytes"]
        + obs_batch_bytes
        + obs_build_temp_bytes
        + policy_weights_bytes
        + fwd_workspace_bytes
        + accumulators_bytes
        + cuda_runtime_reserve_bytes
    )
    out: dict[str, Any] = {
        "eval_mode": eval_mode,
        "num_policies": len(net_sizes),
        "net_sizes": [str(sz) for sz in net_sizes],
        "num_envs_per_match": w_per_match,
        "pairs_per_batch": b_pairs,
        "total_worlds": total_worlds,
        "total_obs_rows": total_obs_rows,
        "max_policy_rows_per_tick": max_pol_rows,
        "persistent_per_world_bytes": int(sim_mem["persistent_per_world_bytes"]),
        "peak_per_world_bytes": int(sim_mem["peak_per_world_bytes"]),
        "sim_persistent_bytes": int(sim_mem["sim_persistent_bytes"]),
        "sim_peak_bytes": int(sim_mem["sim_peak_bytes"]),
        "obs_batch_bytes": int(obs_batch_bytes),
        "obs_build_temp_bytes": obs_build_temp_bytes,
        "policy_weights_bytes": int(policy_weights_bytes),
        "fwd_workspace_bytes": int(fwd_workspace_bytes),
        "accumulators_bytes": int(accumulators_bytes),
        "cuda_runtime_reserve_bytes": cuda_runtime_reserve_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
    }
    if gpu is not None:
        gpu_key = str(gpu).strip().upper()
        if gpu_key in GPU_VRAM_BUDGET_BYTES:
            budget = int(GPU_VRAM_BUDGET_BYTES[gpu_key])
            out["gpu"] = gpu_key
            out["budget_bytes"] = budget
            out["fits_budget"] = bool(estimated_peak_bytes <= budget)
            out["headroom_bytes"] = int(budget - estimated_peak_bytes)
    return out


def select_pairs_per_batch(
    *,
    game_cfg: GameV3Config,
    net_sizes: Sequence[NetSize],
    num_envs_per_pair: int,
    total_pairs_to_run: int,
    vram_budget_bytes: int,
    amp: bool = True,
    max_pairs_override: int | None = None,
    gpu_label: str = "GPU",
) -> tuple[int, dict[str, Any]]:
    """Choose the largest number of round-robin pairs per batch fitting ``vram_budget_bytes``."""
    if max_pairs_override is not None and int(max_pairs_override) < 1:
        raise ValueError(f"max_pairs_per_batch must be >= 1, got {max_pairs_override}")
    budget = int(vram_budget_bytes)
    single_mem = estimate_eval_memory(
        game_cfg=game_cfg,
        net_sizes=net_sizes,
        eval_mode="round-robin",
        num_envs_per_match=int(num_envs_per_pair),
        pairs_per_batch=1,
        amp=amp,
        gpu=gpu_label if gpu_label in GPU_VRAM_BUDGET_BYTES else None,
    )
    if int(single_mem["estimated_peak_bytes"]) > budget:
        raise ValueError(
            f"Refusing round-robin configuration on {gpu_label}: even 1 pair "
            f"({num_envs_per_pair} worlds) requires {single_mem['estimated_peak_bytes'] / 1e9:.2f} GB "
            f"peak VRAM, exceeding {budget / 1e9:.2f} GB budget."
        )

    upper = max(1, int(total_pairs_to_run))
    if max_pairs_override is not None:
        upper = min(upper, int(max_pairs_override))

    lo, hi = 1, upper
    best_b = 1
    best_mem = single_mem
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_mem = estimate_eval_memory(
            game_cfg=game_cfg,
            net_sizes=net_sizes,
            eval_mode="round-robin",
            num_envs_per_match=int(num_envs_per_pair),
            pairs_per_batch=mid,
            amp=amp,
            gpu=gpu_label if gpu_label in GPU_VRAM_BUDGET_BYTES else None,
        )
        if int(cand_mem["estimated_peak_bytes"]) <= budget:
            best_b = mid
            best_mem = cand_mem
            lo = mid + 1
        else:
            hi = mid - 1

    best_mem = dict(best_mem)
    best_mem["budget_bytes"] = budget
    best_mem["fits_budget"] = True
    best_mem["headroom_bytes"] = int(budget - int(best_mem["estimated_peak_bytes"]))
    return best_b, best_mem


def build_pair_reuse_key(
    *,
    num_envs: int,
    burnin_ticks: int,
    window_ticks: int,
    greedy: bool,
    seed: int,
    policy_a_sha256: str,
    policy_b_sha256: str,
) -> dict[str, Any]:
    """Build the complete reuse key governing round-robin pair artifact skipping."""
    return {
        "num_envs": int(num_envs),
        "burnin_ticks": int(burnin_ticks),
        "window_ticks": int(window_ticks),
        "greedy": bool(greedy),
        "seed": int(seed),
        "policy_a_sha256": str(policy_a_sha256),
        "policy_b_sha256": str(policy_b_sha256),
    }


def extract_pair_reuse_key(unit_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the full pair reuse key from a completed ``pairs/pair_XX_YY.json`` payload."""
    if not isinstance(unit_payload, Mapping):
        raise ValueError(f"Pair unit payload must be a dict, got {type(unit_payload)!r}")
    rk = unit_payload.get("reuse_key")
    if isinstance(rk, Mapping):
        missing = [k for k in PAIR_REUSE_KEY_FIELDS if k not in rk]
        if missing:
            raise ValueError(f"Pair unit reuse_key missing required fields: {missing}")
        if not isinstance(rk["greedy"], bool):
            raise ValueError(f"Pair unit reuse_key.greedy must be bool, got {rk['greedy']!r}")
        return build_pair_reuse_key(
            num_envs=int(rk["num_envs"]),
            burnin_ticks=int(rk["burnin_ticks"]),
            window_ticks=int(rk["window_ticks"]),
            greedy=bool(rk["greedy"]),
            seed=int(rk["seed"]),
            policy_a_sha256=str(rk["policy_a_sha256"]),
            policy_b_sha256=str(rk["policy_b_sha256"]),
        )

    match_obj = unit_payload.get("match")
    if not isinstance(match_obj, Mapping):
        raise ValueError("Pair unit payload missing 'match' mapping")
    ecfg = match_obj.get("eval_config")
    if not isinstance(ecfg, Mapping):
        raise ValueError("Pair unit match missing 'eval_config' mapping")
    for req_k in ("num_envs", "burnin_ticks", "window_ticks"):
        if req_k not in ecfg:
            raise ValueError(f"Pair unit eval_config missing required field {req_k!r}")
    if "greedy" not in match_obj or not isinstance(match_obj["greedy"], bool):
        raise ValueError("Pair unit match missing boolean 'greedy' field")
    if "seed" not in unit_payload:
        raise ValueError("Pair unit payload missing 'seed' field")
    pols = unit_payload.get("policies")
    if not isinstance(pols, Sequence) or len(pols) != 2:
        raise ValueError("Pair unit payload must record exactly 2 policies")
    return build_pair_reuse_key(
        num_envs=int(ecfg["num_envs"]),
        burnin_ticks=int(ecfg["burnin_ticks"]),
        window_ticks=int(ecfg["window_ticks"]),
        greedy=bool(match_obj["greedy"]),
        seed=int(unit_payload["seed"]),
        policy_a_sha256=str(pols[0]["checkpoint_sha256"]),
        policy_b_sha256=str(pols[1]["checkpoint_sha256"]),
    )


def validate_pair_unit_reuse(
    unit_id: str,
    unit_payload: Mapping[str, Any],
    expected_key: Mapping[str, Any],
) -> None:
    """Raise ``ValueError`` if ``unit_payload`` disagrees with ``expected_key`` on any reuse field."""
    actual = extract_pair_reuse_key(unit_payload)
    mismatches = {
        k: (actual.get(k), expected_key.get(k))
        for k in PAIR_REUSE_KEY_FIELDS
        if actual.get(k) != expected_key.get(k)
    }
    if mismatches:
        detail = ", ".join(
            f"{k}={got!r} != expected {exp!r}" for k, (got, exp) in mismatches.items()
        )
        raise ValueError(f"Staged/existing pair unit {unit_id} reuse-key mismatch: {detail}")


@torch.no_grad()
def act_policy_chunked(
    policy: LoadedEvalPolicy,
    obs_s1: torch.Tensor,
    obs_s4: torch.Tensor,
    obs_s16: torch.Tensor,
    obs_scalars: torch.Tensor,
    obs_mask: torch.Tensor,
    *,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select actions using the exact chunked ``EgoMultiScaleNet.act`` path from training."""
    m_total = int(obs_s1.shape[0])
    if m_total == 0:
        return torch.empty((0,), dtype=torch.long, device=obs_s1.device)
    chunk = max(1, min(m_total, int(policy.actor_chunk_learners)))
    if chunk == m_total:
        return policy.model.act(
            obs_s1,
            obs_s4,
            obs_s16,
            obs_scalars,
            action_mask=obs_mask,
            greedy=greedy,
            generator=generator,
            compute_values=False,
        ).actions

    actions = torch.empty((m_total,), dtype=torch.long, device=obs_s1.device)
    for start in range(0, m_total, chunk):
        end = min(m_total, start + chunk)
        actions[start:end] = policy.model.act(
            obs_s1[start:end],
            obs_s4[start:end],
            obs_s16[start:end],
            obs_scalars[start:end],
            action_mask=obs_mask[start:end],
            greedy=greedy,
            generator=generator,
            compute_values=False,
        ).actions
    return actions


@torch.no_grad()
def _batched_policy_logits_and_probs(
    policy: LoadedEvalPolicy,
    obs_s1: torch.Tensor,
    obs_s4: torch.Tensor,
    obs_s16: torch.Tensor,
    obs_scalars: torch.Tensor,
    obs_mask: torch.Tensor,
    *,
    greedy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run one chunked forward for ``policy`` across all seats it plays in across a pair batch."""
    m_total = int(obs_s1.shape[0])
    dev = obs_s1.device
    if m_total == 0:
        empty = torch.empty((0, NUM_ACTIONS), dtype=torch.float32, device=dev)
        return empty, (None if greedy else empty)

    chunk = max(1, min(m_total, int(policy.actor_chunk_learners)))
    model = policy.model
    masked_logits_out = torch.empty((m_total, NUM_ACTIONS), dtype=torch.float32, device=dev)
    probs_out = (
        None if greedy else torch.empty((m_total, NUM_ACTIONS), dtype=torch.float32, device=dev)
    )
    for start in range(0, m_total, chunk):
        end = min(m_total, start + chunk)
        with model._autocast_ctx(dev):
            trunk, _ = model.encode_trunk(
                obs_s1[start:end],
                obs_s4[start:end],
                obs_s16[start:end],
                obs_scalars[start:end],
            )
        if dev.type == "cuda":
            with torch.amp.autocast(device_type="cuda", enabled=False):
                raw_logits = model.policy_head(trunk.float())
                m_logits, safe_mask = model.mask_policy_logits(raw_logits, obs_mask[start:end])
                masked_logits_out[start:end] = m_logits
                if probs_out is not None:
                    p_chunk = F.softmax(m_logits, dim=-1).masked_fill(~safe_mask, 0.0)
                    probs_out[start:end] = p_chunk / p_chunk.sum(dim=-1, keepdim=True).clamp_min(
                        1e-12
                    )
        else:
            raw_logits = model.policy_head(trunk.float())
            m_logits, safe_mask = model.mask_policy_logits(raw_logits, obs_mask[start:end])
            masked_logits_out[start:end] = m_logits
            if probs_out is not None:
                p_chunk = F.softmax(m_logits, dim=-1).masked_fill(~safe_mask, 0.0)
                probs_out[start:end] = p_chunk / p_chunk.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return masked_logits_out, probs_out


@torch.no_grad()
def _sample_actions_from_probs_chunked(
    probs: torch.Tensor,
    *,
    actor_chunk_learners: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample actions from ``probs`` using the exact per-chunk ``torch.multinomial`` sequence."""
    m_side = int(probs.shape[0])
    dev = probs.device
    if m_side == 0:
        return torch.empty((0,), dtype=torch.long, device=dev)
    chunk = max(1, min(m_side, int(actor_chunk_learners)))
    out = torch.empty((m_side,), dtype=torch.long, device=dev)
    for start in range(0, m_side, chunk):
        end = min(m_side, start + chunk)
        p_slice = probs[start:end]
        if generator is not None and generator.device.type == "cpu" and dev.type != "cpu":
            act_cpu = torch.multinomial(p_slice.detach().cpu(), num_samples=1, generator=generator)
            out[start:end] = act_cpu.squeeze(-1).to(device=dev, dtype=torch.long)
        else:
            out[start:end] = torch.multinomial(p_slice, num_samples=1, generator=generator).squeeze(
                -1
            )
    return out


def build_rotated_seat_assignment(
    num_envs: int,
    num_snakes: int,
    num_policies: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Assign each ``(world, slot)`` to a policy index ``0..P-1`` with cyclic world rotation.

    Uses ``seat_policy[w, s] = (s + w) % P``, which:
    1. Splits the ``N`` seats in every world as evenly as possible (exact ``N // P`` when ``P | N``,
       e.g., ``12, 8, 6, 4`` seats for ``P = 2, 3, 4, 6`` on ``N = 24``; ``5`` or ``4`` for ``P = 5``).
    2. Interleaves policies across slot indices within every world (avoiding slot-index tie bias).
    3. Rotates spawn slot assignments across worlds with period ``P`` so every policy occupies
       every spawn slot ``s`` equally often when ``num_envs`` is a multiple of ``P``.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    if num_snakes < 2:
        raise ValueError(f"num_snakes must be >= 2, got {num_snakes}")
    if not (2 <= num_policies <= 6):
        raise ValueError(f"head-to-head requires 2 to 6 policies, got {num_policies}")
    if num_policies > num_snakes:
        raise ValueError(f"num_policies ({num_policies}) cannot exceed num_snakes ({num_snakes})")
    if int(num_envs) % int(num_policies) != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be a multiple of num_policies ({num_policies}) "
            "so cyclic seat rotation (slot + world) % P cancels spawn-slot bias"
        )
    dev = normalize_device(device)
    w_idx = torch.arange(int(num_envs), dtype=torch.int64, device=dev)[:, None]
    s_idx = torch.arange(int(num_snakes), dtype=torch.int64, device=dev)[None, :]
    return (s_idx + w_idx).remainder(int(num_policies))


def build_rotated_learner_mask(
    num_envs: int,
    num_snakes: int,
    learner_slots: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build ``(E, N)`` boolean mask for ``K`` learner seats rotated evenly across worlds."""
    if not (1 <= int(learner_slots) <= int(num_snakes)):
        raise ValueError(f"learner_slots must be in [1, {num_snakes}], got {learner_slots}")
    dev = normalize_device(device)
    if int(learner_slots) == int(num_snakes):
        return torch.ones((int(num_envs), int(num_snakes)), dtype=torch.bool, device=dev)
    w_idx = torch.arange(int(num_envs), dtype=torch.int64, device=dev)[:, None]
    s_idx = torch.arange(int(num_snakes), dtype=torch.int64, device=dev)[None, :]
    return ((s_idx + w_idx).remainder(int(num_snakes))) < int(learner_slots)


def classify_step_deaths(
    sim: GameV3Sim,
    *,
    acted_es: torch.Tensor,
    world_actions: torch.Tensor,
    pre_hx: torch.Tensor,
    pre_hy: torch.Tensor,
    pre_theta: torch.Tensor,
    pre_len: torch.Tensor,
    pre_accum: torch.Tensor,
    pre_boost_ticks: torch.Tensor,
    died: torch.Tensor,
    killer_id: torch.Tensor,
) -> torch.Tensor:
    """Classify every ``(E, N)`` slot's death cause for the just-completed ``sim.step``.

    Returns an ``int64`` tensor of shape ``(E, N)`` with:
    - ``-1``: did not die on this tick
    - ``0``: ``head_into_body`` (killed by enemy retained body or earlier boost sub-step)
    - ``1``: ``wall`` (arena boundary collision without enemy collision)
    - ``2``: ``self`` (always ``0`` occurrences in Game V3.1 because own-body crossings are legal)
    - ``3``: ``head_on`` (simultaneous sub-step head-to-head collision ``s_i == s_k``)
    """
    cause = torch.full(died.shape, -1, dtype=torch.int64, device=sim.device)
    actual_died = acted_es & died
    wall_died = actual_died & (killer_id < 0)
    enemy_died = actual_died & (killer_id >= 0)

    # Reconstruct exact sub-step points (mirrors GameV3Sim._phase_move + _phase_collide)
    act = world_actions.to(torch.int32).clamp(0, NUM_ACTIONS - 1)
    steer = act.remainder(NUM_TURN_LEVELS).to(torch.int64)
    want_boost = act >= NUM_TURN_LEVELS

    max_len = int(sim.cfg.max_length)
    cur_len_i64 = pre_len.clamp(0, max_len).to(torch.int64)
    dtheta = sim._turn_deltas_table[cur_len_i64].gather(2, steer[:, :, None]).squeeze(2)
    angle_mask = int(sim.cfg.angle_steps - 1)
    new_theta = torch.where(acted_es, (pre_theta + dtheta) & angle_mask, pre_theta)

    boosting = acted_es & want_boost & (pre_len > int(sim.cfg.min_boost_length))
    base_spd_fp = sim._speed_fp_table[cur_len_i64]
    boost_spd_fp = sim._boost_speed_fp_table[cur_len_i64]
    step_fp = torch.where(boosting, boost_spd_fp, base_spd_fp)

    th_i64 = new_theta.to(torch.int64)
    sin_t = sim._sin_table[th_i64]
    cos_t = sim._cos_table[th_i64]

    tot_dist = pre_accum + step_fp
    num_pts = torch.where(
        acted_es,
        torch.div(tot_dist, base_spd_fp, rounding_mode="floor"),
        torch.zeros_like(tot_dist),
    )
    d_pts = ((sim._pts_idx + 1) * base_spd_fp[:, :, None] - pre_accum[:, :, None]).clamp_min(0)
    pt_x = pre_hx[:, :, None] + step_delta_fp_torch(d_pts, cos_t[:, :, None])
    pt_y = pre_hy[:, :, None] + step_delta_fp_torch(d_pts, sin_t[:, :, None])
    pt_gx = (pt_x >> FP_SHIFT) + sim.half_grid
    pt_gy = (pt_y >> FP_SHIFT) + sim.half_grid

    inc_boost = torch.where(boosting, pre_boost_ticks + 1, pre_boost_ticks)
    shed = boosting & (inc_boost >= int(sim.cfg.boost_shed_ticks))
    len_post_shed = torch.where(
        acted_es & shed,
        pre_len - int(sim.cfg.boost_shed_mass),
        pre_len,
    )
    r_tier = sim._r_tier_table[len_post_shed.clamp(0, max_len).to(torch.int64)]

    k_idx = killer_id.clamp(0, sim.N - 1).to(torch.int64)
    num_pts_k = num_pts.gather(1, k_idx)
    r_tier_k = r_tier.gather(1, k_idx)
    pt_gx_k = pt_gx.gather(1, k_idx[:, :, None].expand(-1, -1, sim.max_pts))
    pt_gy_k = pt_gy.gather(1, k_idx[:, :, None].expand(-1, -1, sim.max_pts))

    simultaneous_head = torch.zeros_like(enemy_died)
    max_diam = int(sim._max_diam)
    for sub_s in range(sim.max_pts):
        ki = sub_s - (sim.max_pts - num_pts)
        kk = sub_s - (sim.max_pts - num_pts_k)
        vi = acted_es & (ki >= 0) & (ki < num_pts)
        vk = (kk >= 0) & (kk < num_pts_k)
        ki_c = ki.clamp(0, sim.max_pts - 1).to(torch.int64)
        kk_c = kk.clamp(0, sim.max_pts - 1).to(torch.int64)
        gxi = pt_gx.gather(2, ki_c[:, :, None]).squeeze(2)
        gyi = pt_gy.gather(2, ki_c[:, :, None]).squeeze(2)
        gxk = pt_gx_k.gather(2, kk_c[:, :, None]).squeeze(2)
        gyk = pt_gy_k.gather(2, kk_c[:, :, None]).squeeze(2)
        dgx = gxk - gxi
        dgy = gyk - gyi
        near = (
            vi
            & vk
            & (dgx >= -max_diam)
            & (dgx <= max_diam)
            & (dgy >= -max_diam)
            & (dgy <= max_diam)
        )
        dgx_idx = (dgx + max_diam).clamp(0, 2 * max_diam).to(torch.int64)
        dgy_idx = (dgy + max_diam).clamp(0, 2 * max_diam).to(torch.int64)
        ov = sim._overlap_lut[r_tier, r_tier_k, dgx_idx, dgy_idx]
        simultaneous_head = simultaneous_head | (near & ov)

    cause = torch.where(wall_died, torch.ones_like(cause), cause)
    cause = torch.where(
        enemy_died & simultaneous_head,
        torch.full_like(cause, 3),
        cause,
    )
    cause = torch.where(
        enemy_died & (~simultaneous_head),
        torch.zeros_like(cause),
        cause,
    )
    return cause


def _make_action_generator(device: torch.device, seed_val: int) -> torch.Generator | None:
    """Create a deterministic PyTorch RNG generator for ``device``."""
    masked = int(seed_val) & 0x7FFF_FFFF_FFFF_FFFF
    if device.type == "cuda":
        gen = torch.Generator(device=device)
        gen.manual_seed(masked)
        return gen
    if device.type == "cpu":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(masked)
        return gen
    return None


def _bootstrap_ratio_and_diff_ci(
    num_a: np.ndarray,
    num_b: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Compute point estimates and 95% world-bootstrap CIs for ``A/(A+B)`` and ``A - B``."""
    e_cnt = int(num_a.shape[0])
    tot_a = float(num_a.sum())
    tot_b = float(num_b.sum())
    denom = tot_a + tot_b
    share_pt = tot_a / denom if denom > 0.0 else 0.5
    diff_per_world_pt = float((num_a - num_b).mean()) if e_cnt > 0 else 0.0

    if e_cnt <= 1 or n_bootstrap <= 0:
        return {
            "share_a": round(share_pt, 6),
            "share_a_ci95_low": round(share_pt, 6),
            "share_a_ci95_high": round(share_pt, 6),
            "diff_a_minus_b_total": round(tot_a - tot_b, 4),
            "diff_a_minus_b_per_world": round(diff_per_world_pt, 4),
            "diff_per_world_ci95_low": round(diff_per_world_pt, 4),
            "diff_per_world_ci95_high": round(diff_per_world_pt, 4),
        }

    rng = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFF_FFFF_FFFF_FFFF))
    idx = rng.integers(0, e_cnt, size=(int(n_bootstrap), e_cnt))
    bs_a = num_a[idx].sum(axis=1)
    bs_b = num_b[idx].sum(axis=1)
    bs_denom = bs_a + bs_b
    bs_share = np.where(bs_denom > 0.0, bs_a / np.maximum(bs_denom, 1e-12), 0.5)
    bs_diff_pw = (bs_a - bs_b) / float(e_cnt)

    s_lo, s_hi = np.percentile(bs_share, [2.5, 97.5])
    d_lo, d_hi = np.percentile(bs_diff_pw, [2.5, 97.5])
    return {
        "share_a": round(share_pt, 6),
        "share_a_ci95_low": round(float(s_lo), 6),
        "share_a_ci95_high": round(float(s_hi), 6),
        "diff_a_minus_b_total": round(tot_a - tot_b, 4),
        "diff_a_minus_b_per_world": round(diff_per_world_pt, 4),
        "diff_per_world_ci95_low": round(float(d_lo), 4),
        "diff_per_world_ci95_high": round(float(d_hi), 4),
    }


def fit_bradley_terry_elo(
    wins_matrix: np.ndarray,
    games_matrix: np.ndarray,
    *,
    prior_wins: float = DEFAULT_BT_PRIOR_WINS,
    anchor_mean_elo: float = 1500.0,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Fit regularized Bradley-Terry ratings on Elo scale (``400 * log10(pi)``, mean ``1500``).

    Uses Hunter's minorization-maximization (MM) fixed-point iteration with symmetric ``prior_wins``
    pseudo-counts between every played pair so ratings remain finite and unique even under complete
    separation (e.g., ``32-0`` world sweeps).
    """
    m_cnt = int(wins_matrix.shape[0])
    if m_cnt == 0:
        return np.zeros((0,), dtype=np.float64)
    if m_cnt == 1:
        return np.array([anchor_mean_elo], dtype=np.float64)

    played = (games_matrix > 0) & (~np.eye(m_cnt, dtype=bool))
    w_reg = wins_matrix.astype(np.float64).copy()
    n_reg = games_matrix.astype(np.float64).copy()
    if prior_wins > 0.0:
        w_reg = np.where(played, w_reg + float(prior_wins), 0.0)
        n_reg = np.where(played, n_reg + 2.0 * float(prior_wins), 0.0)

    pi = np.ones((m_cnt,), dtype=np.float64)
    win_totals = w_reg.sum(axis=1)
    for _ in range(int(max_iter)):
        denom_mat = pi[:, None] + pi[None, :]
        inv_sum = np.where(n_reg > 0.0, n_reg / np.maximum(denom_mat, 1e-30), 0.0).sum(axis=1)
        next_pi = np.where(inv_sum > 0.0, np.maximum(win_totals, 1e-30) / inv_sum, pi)
        # Geometric mean normalization keeps log-ratings centered without drift
        log_next = np.log(np.maximum(next_pi, 1e-30))
        log_next -= log_next.mean()
        next_pi = np.exp(np.clip(log_next, -50.0, 50.0))
        if float(np.max(np.abs(np.log(next_pi) - np.log(pi)))) < tol:
            pi = next_pi
            break
        pi = next_pi

    elo = anchor_mean_elo + 400.0 * (np.log10(pi) - np.mean(np.log10(pi)))
    return elo


@torch.no_grad()
def evaluate_head_to_head(
    policies: Sequence[LoadedEvalPolicy],
    *,
    num_envs: int = DEFAULT_H2H_NUM_ENVS,
    burnin_ticks: int = DEFAULT_H2H_BURNIN_TICKS,
    window_ticks: int = DEFAULT_H2H_WINDOW_TICKS,
    mass_bin_ticks: int = DEFAULT_H2H_MASS_BIN_TICKS,
    greedy: bool = False,
    seed: int = DEFAULT_EVAL_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    deterministic: bool = True,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run a 2-to-6 policy head-to-head match with rotated spawn seats across ``num_envs`` worlds."""
    p_cnt = len(policies)
    if not (2 <= p_cnt <= 6):
        raise ValueError(f"Head-to-head requires 2 to 6 policies, got {p_cnt}")
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    if burnin_ticks < 0:
        raise ValueError(f"burnin_ticks must be >= 0, got {burnin_ticks}")
    if window_ticks < 1:
        raise ValueError(f"window_ticks must be >= 1, got {window_ticks}")
    if mass_bin_ticks < 1:
        raise ValueError(f"mass_bin_ticks must be >= 1, got {mass_bin_ticks}")

    game_cfg = validate_compatible_policies(policies)
    dev = next(policies[0].model.parameters()).device
    for pol in policies:
        pol.model.eval()

    t_wall0 = time.perf_counter()
    total_ticks = int(burnin_ticks) + int(window_ticks)
    num_ts_bins = max(1, int(math.ceil(total_ticks / float(mass_bin_ticks))))

    with preserve_rng_state():
        if dev.type == "cpu" and deterministic:
            configure_determinism(True)
        seed_everything(int(seed))

        world_seeds = derive_world_seeds(int(seed), int(num_envs), stream_id=701)
        sim = GameV3Sim(game_cfg, num_envs=int(num_envs), device=dev, seeds=world_seeds)
        e_cnt, n_cnt = sim.E, sim.N

        seat_policy = build_rotated_seat_assignment(e_cnt, n_cnt, p_cnt, device=dev)
        seat_policy_flat = seat_policy.reshape(-1)
        flat_indices_by_p: list[torch.Tensor] = [
            torch.nonzero(seat_policy_flat == p_idx, as_tuple=False).squeeze(1)
            for p_idx in range(p_cnt)
        ]
        seats_per_world_np = (
            torch.stack([(seat_policy == p_idx).sum(dim=1) for p_idx in range(p_cnt)], dim=1)
            .cpu()
            .numpy()
        )
        gens = [
            _make_action_generator(dev, int(seed) ^ (0x9E37_79B9 * (p_idx + 1)))
            for p_idx in range(p_cnt)
        ]

        full_obs: ObservationBatch = allocate_obs_batch(sim)
        assert_obs_batch_matches_layout(
            ObservationBatch(
                raster_s1=full_obs.raster_s1.flatten(0, 1),
                raster_s4=full_obs.raster_s4.flatten(0, 1),
                raster_s16=full_obs.raster_s16.flatten(0, 1),
                scalars=full_obs.scalars.flatten(0, 1),
                action_mask=full_obs.action_mask.flatten(0, 1),
            ),
            policies[0].obs_layout,
            batch_rows=e_cnt * n_cnt,
            context="arena_eval.head_to_head",
        )

        # Device accumulators (zero host syncs during simulation loop)
        ts_mass_ep = torch.zeros((num_ts_bins, p_cnt), dtype=torch.int64, device=dev)
        ts_ticks = np.zeros((num_ts_bins,), dtype=np.int64)

        win_mass_ep = torch.zeros((e_cnt, p_cnt), dtype=torch.int64, device=dev)
        win_steps_ep = torch.zeros((e_cnt, p_cnt), dtype=torch.int64, device=dev)
        win_kills_ep = torch.zeros((e_cnt, p_cnt), dtype=torch.int64, device=dev)
        win_deaths_ep = torch.zeros((e_cnt, p_cnt), dtype=torch.int64, device=dev)
        win_boost_ep = torch.zeros((e_cnt, p_cnt), dtype=torch.int64, device=dev)
        win_eaten_ep = torch.zeros((e_cnt, p_cnt), dtype=torch.int64, device=dev)
        win_shed_ep = torch.zeros((e_cnt, p_cnt), dtype=torch.int64, device=dev)

        # Per-world victim-by-(killer_policy | wall | self) matrix: shape (E, P, P + 2)
        kill_cols = p_cnt + 2
        world_kill_mat_1d = torch.zeros((e_cnt * p_cnt * kill_cols,), dtype=torch.int64, device=dev)
        world_ids_en = torch.arange(e_cnt, dtype=torch.int64, device=dev)[:, None].expand(
            e_cnt, n_cnt
        )
        death_causes_pc = torch.zeros((p_cnt * 4,), dtype=torch.int64, device=dev)

        world_actions_flat = torch.empty((e_cnt * n_cnt,), dtype=torch.long, device=dev)
        last_progress_mono = time.perf_counter()

        for tick in range(total_ticks):
            if should_stop is not None and should_stop():
                raise StopRequestedError(f"Stop requested during head-to-head at tick {tick}")
            now_mono = time.perf_counter()
            if (now_mono - last_progress_mono) >= 90.0:
                phase_stamp(
                    "arena_eval_h2h",
                    "PROGRESS",
                    tick=tick,
                    total_ticks=total_ticks,
                    elapsed_sec=round(now_mono - t_wall0, 2),
                )
                last_progress_mono = now_mono

            build_obs(sim, out=full_obs)
            r1 = full_obs.raster_s1.flatten(0, 1)
            r4 = full_obs.raster_s4.flatten(0, 1)
            r16 = full_obs.raster_s16.flatten(0, 1)
            sc = full_obs.scalars.flatten(0, 1)
            am = full_obs.action_mask.flatten(0, 1)

            for p_idx in range(p_cnt):
                f_idx = flat_indices_by_p[p_idx]
                acts_p = act_policy_chunked(
                    policies[p_idx],
                    r1.index_select(0, f_idx),
                    r4.index_select(0, f_idx),
                    r16.index_select(0, f_idx),
                    sc.index_select(0, f_idx),
                    am.index_select(0, f_idx),
                    greedy=greedy,
                    generator=gens[p_idx],
                )
                world_actions_flat.index_copy_(0, f_idx, acts_p)

            world_actions = world_actions_flat.view(e_cnt, n_cnt)
            acted_es = sim.alive & (~sim.truncated_pending)
            pre_len = sim.length.clone()
            pre_boost_ticks = sim.boost_ticks.clone()

            in_window = tick >= int(burnin_ticks)
            if in_window:
                pre_hx = sim.head_pos[:, :, 0].clone()
                pre_hy = sim.head_pos[:, :, 1].clone()
                pre_theta = sim.theta.clone()
                pre_accum = sim.dist_accum_fp.clone()

            live_len_i64 = torch.where(
                acted_es, pre_len.to(torch.int64), torch.zeros_like(seat_policy)
            )
            wp_mass_tick = torch.zeros((e_cnt, p_cnt), dtype=torch.int64, device=dev)
            wp_mass_tick.scatter_add_(1, seat_policy, live_len_i64)

            ts_idx = min(num_ts_bins - 1, tick // int(mass_bin_ticks))
            ts_mass_ep[ts_idx] += wp_mass_tick.sum(dim=0)
            ts_ticks[ts_idx] += 1

            info = sim.step(world_actions)

            if in_window:
                win_mass_ep += wp_mass_tick
                win_steps_ep.scatter_add_(1, seat_policy, acted_es.to(torch.int64))
                died_es = acted_es & info.died
                win_deaths_ep.scatter_add_(1, seat_policy, died_es.to(torch.int64))
                win_kills_ep.scatter_add_(1, seat_policy, info.kill_credit.to(torch.int64))

                boosting_es = (
                    acted_es
                    & (world_actions >= NUM_TURN_LEVELS)
                    & (pre_len > int(sim.cfg.min_boost_length))
                )
                inc_bt = torch.where(boosting_es, pre_boost_ticks + 1, pre_boost_ticks)
                shed_es = boosting_es & (inc_bt >= int(sim.cfg.boost_shed_ticks))
                win_boost_ep.scatter_add_(1, seat_policy, boosting_es.to(torch.int64))
                win_shed_ep.scatter_add_(
                    1, seat_policy, shed_es.to(torch.int64) * int(sim.cfg.boost_shed_mass)
                )
                win_eaten_ep.scatter_add_(1, seat_policy, info.mass_eaten.to(torch.int64))

                ks = info.killer_id
                killer_policy = seat_policy.gather(1, ks.clamp(0, n_cnt - 1).to(torch.int64))
                killer_col = torch.where(
                    ks >= 0,
                    killer_policy,
                    torch.full_like(seat_policy, p_cnt),
                )
                flat_km_key = (
                    world_ids_en * (p_cnt * kill_cols) + seat_policy * kill_cols + killer_col
                ).reshape(-1)
                world_kill_mat_1d.index_put_(
                    (flat_km_key,),
                    died_es.reshape(-1).to(torch.int64),
                    accumulate=True,
                )

                causes = classify_step_deaths(
                    sim,
                    acted_es=acted_es,
                    world_actions=world_actions,
                    pre_hx=pre_hx,
                    pre_hy=pre_hy,
                    pre_theta=pre_theta,
                    pre_len=pre_len,
                    pre_accum=pre_accum,
                    pre_boost_ticks=pre_boost_ticks,
                    died=info.died,
                    killer_id=info.killer_id,
                )
                valid_c = causes >= 0
                c_key = (seat_policy * 4 + causes.clamp_min(0)).reshape(-1)
                death_causes_pc.index_put_(
                    (c_key,), valid_c.reshape(-1).to(torch.int64), accumulate=True
                )

        synchronize_device(dev)
        wall_seconds = time.perf_counter() - t_wall0

    # Transfer compact summary arrays to CPU once
    ts_mass_np = ts_mass_ep.cpu().numpy()
    win_mass_np = win_mass_ep.cpu().numpy()
    win_steps_np = win_steps_ep.cpu().numpy()
    win_kills_np = win_kills_ep.cpu().numpy()
    win_deaths_np = win_deaths_ep.cpu().numpy()
    win_boost_np = win_boost_ep.cpu().numpy()
    win_eaten_np = win_eaten_ep.cpu().numpy()
    win_shed_np = win_shed_ep.cpu().numpy()
    world_kill_mat_np = world_kill_mat_1d.view(e_cnt, p_cnt, kill_cols).cpu().numpy()
    death_causes_np = death_causes_pc.view(p_cnt, 4).cpu().numpy()

    return _finalize_head_to_head_result(
        policies=policies,
        game_cfg=game_cfg,
        dev=dev,
        num_envs=int(num_envs),
        burnin_ticks=int(burnin_ticks),
        window_ticks=int(window_ticks),
        mass_bin_ticks=int(mass_bin_ticks),
        greedy=bool(greedy),
        seed=int(seed),
        bootstrap_samples=int(bootstrap_samples),
        deterministic=bool(deterministic),
        wall_seconds=wall_seconds,
        seats_per_world_np=seats_per_world_np,
        ts_mass_np=ts_mass_np,
        ts_ticks=ts_ticks,
        win_mass_np=win_mass_np,
        win_steps_np=win_steps_np,
        win_kills_np=win_kills_np,
        win_deaths_np=win_deaths_np,
        win_boost_np=win_boost_np,
        win_eaten_np=win_eaten_np,
        win_shed_np=win_shed_np,
        world_kill_mat_np=world_kill_mat_np,
        death_causes_np=death_causes_np,
    )


def _finalize_head_to_head_result(
    *,
    policies: Sequence[LoadedEvalPolicy],
    game_cfg: GameV3Config,
    dev: torch.device,
    num_envs: int,
    burnin_ticks: int,
    window_ticks: int,
    mass_bin_ticks: int,
    greedy: bool,
    seed: int,
    bootstrap_samples: int,
    deterministic: bool,
    wall_seconds: float,
    seats_per_world_np: np.ndarray,
    ts_mass_np: np.ndarray,
    ts_ticks: np.ndarray,
    win_mass_np: np.ndarray,
    win_steps_np: np.ndarray,
    win_kills_np: np.ndarray,
    win_deaths_np: np.ndarray,
    win_boost_np: np.ndarray,
    win_eaten_np: np.ndarray,
    win_shed_np: np.ndarray,
    world_kill_mat_np: np.ndarray,
    death_causes_np: np.ndarray,
) -> dict[str, Any]:
    """Build the canonical ``head-to-head`` JSON payload from per-world numpy accumulators."""
    p_cnt = len(policies)
    e_cnt = int(num_envs)
    n_cnt = int(game_cfg.num_snakes)
    total_ticks = int(burnin_ticks) + int(window_ticks)
    num_ts_bins = int(ts_mass_np.shape[0])
    kill_cols = p_cnt + 2

    total_arena_mass = float(win_mass_np.sum())
    labels = [pol.label for pol in policies]

    per_policy_summary: list[dict[str, Any]] = []
    for p_idx, pol in enumerate(policies):
        steps = int(win_steps_np[:, p_idx].sum())
        mass_sum = int(win_mass_np[:, p_idx].sum())
        deaths = int(win_deaths_np[:, p_idx].sum())
        kills = int(win_kills_np[:, p_idx].sum())
        boosts = int(win_boost_np[:, p_idx].sum())
        eaten = int(win_eaten_np[:, p_idx].sum())
        shed = int(win_shed_np[:, p_idx].sum())
        denom_1k = max(1, steps) / 1000.0
        cause_row = {DEATH_CAUSE_NAMES[c_i]: int(death_causes_np[p_idx, c_i]) for c_i in range(4)}
        per_policy_summary.append(
            {
                "policy_index": p_idx,
                "label": pol.label,
                "net_size": pol.net_size,
                "assigned_seats_total": int(seats_per_world_np[:, p_idx].sum()),
                "acted_steps": steps,
                "time_averaged_length": round(mass_sum / max(1, steps), 4),
                "arena_mass_share": round(
                    mass_sum / total_arena_mass if total_arena_mass > 0.0 else 1.0 / p_cnt, 6
                ),
                "deaths": deaths,
                "deaths_per_1k_agent_steps": round(deaths / denom_1k, 4),
                "kills": kills,
                "kills_per_1k_agent_steps": round(kills / denom_1k, 4),
                "boost_share": round(boosts / max(1, steps), 6),
                "mass_eaten_per_1k_agent_steps": round(eaten / denom_1k, 4),
                "mass_shed_boost_per_1k_agent_steps": round(shed / denom_1k, 4),
                "death_causes": cause_row,
            }
        )

    # Kill matrix: victim rows (0..P-1), killer columns (0..P-1 + wall + self), plus attacker view
    agg_kill_mat = world_kill_mat_np.sum(axis=0)  # (P, P + 2)
    col_names = [*labels, "wall", "self"]
    victim_by_killer: dict[str, dict[str, int]] = {}
    for v_idx, v_lbl in enumerate(labels):
        row_dict = {col_names[c_i]: int(agg_kill_mat[v_idx, c_i]) for c_i in range(kill_cols)}
        row_dict["total_deaths"] = int(agg_kill_mat[v_idx].sum())
        victim_by_killer[v_lbl] = row_dict

    killer_by_victim: dict[str, dict[str, int]] = {}
    for k_idx, k_lbl in enumerate(labels):
        row_dict = {labels[v_i]: int(agg_kill_mat[v_i, k_idx]) for v_i in range(p_cnt)}
        row_dict["enemy_policy_kills"] = int(
            sum(agg_kill_mat[v_i, k_idx] for v_i in range(p_cnt) if v_i != k_idx)
        )
        row_dict["friendly_seat_kills"] = int(agg_kill_mat[k_idx, k_idx])
        row_dict["total_kills"] = int(agg_kill_mat[:, k_idx].sum())
        killer_by_victim[k_lbl] = row_dict

    # Mass share over time bins
    mass_over_time: list[dict[str, Any]] = []
    for b_idx in range(num_ts_bins):
        t0 = b_idx * int(mass_bin_ticks)
        t1 = min(total_ticks, (b_idx + 1) * int(mass_bin_ticks))
        b_tot = float(ts_mass_np[b_idx].sum())
        b_ticks = int(ts_ticks[b_idx])
        shares = {
            labels[p_i]: round(
                float(ts_mass_np[b_idx, p_i]) / b_tot if b_tot > 0.0 else 1.0 / p_cnt, 6
            )
            for p_i in range(p_cnt)
        }
        mean_lens = {
            labels[p_i]: round(
                float(ts_mass_np[b_idx, p_i])
                / max(1, b_ticks * int(seats_per_world_np[:, p_i].sum())),
                4,
            )
            for p_i in range(p_cnt)
        }
        mass_over_time.append(
            {
                "tick_start": t0,
                "tick_end": t1,
                "phase": (
                    "burnin"
                    if t1 <= int(burnin_ticks)
                    else ("window" if t0 >= int(burnin_ticks) else "mixed")
                ),
                "ticks_in_bin": b_ticks,
                "mass_shares": shares,
                "mean_lengths": mean_lens,
            }
        )

    # Pairwise summaries with deterministic bootstrap 95% CIs over worlds
    pairwise_summaries: list[dict[str, Any]] = []
    for i in range(p_cnt):
        for j in range(i + 1, p_cnt):
            # Per-seat normalized mass per world handles P=5 (where one policy has 4 seats in a world)
            # and equals raw mass share up to a constant factor whenever seats are equal (P in 2,3,4,6).
            seats_i = np.maximum(1.0, seats_per_world_np[:, i].astype(np.float64))
            seats_j = np.maximum(1.0, seats_per_world_np[:, j].astype(np.float64))
            norm_mass_i = win_mass_np[:, i].astype(np.float64) / seats_i
            norm_mass_j = win_mass_np[:, j].astype(np.float64) / seats_j

            mass_ci = _bootstrap_ratio_and_diff_ci(
                norm_mass_i,
                norm_mass_j,
                n_bootstrap=int(bootstrap_samples),
                seed=int(seed) ^ (0x517C_C1B7 * (i * 16 + j + 1)),
            )

            kills_i_on_j_w = world_kill_mat_np[:, j, i].astype(np.float64)
            kills_j_on_i_w = world_kill_mat_np[:, i, j].astype(np.float64)
            kill_ci = _bootstrap_ratio_and_diff_ci(
                kills_i_on_j_w,
                kills_j_on_i_w,
                n_bootstrap=int(bootstrap_samples),
                seed=int(seed) ^ (0x85EB_CA6B * (i * 16 + j + 1)),
            )

            world_wins_i = float(
                np.sum(norm_mass_i > norm_mass_j) + 0.5 * np.sum(norm_mass_i == norm_mass_j)
            )
            world_shares_i = np.where(
                (norm_mass_i + norm_mass_j) > 0.0,
                norm_mass_i / np.maximum(norm_mass_i + norm_mass_j, 1e-12),
                0.5,
            )
            world_share_std = float(np.std(world_shares_i, ddof=1)) if e_cnt > 1 else 0.0
            share_half_width = 0.5 * (
                float(mass_ci["share_a_ci95_high"]) - float(mass_ci["share_a_ci95_low"])
            )

            k_ij_tot = int(kills_i_on_j_w.sum())
            k_ji_tot = int(kills_j_on_i_w.sum())
            pairwise_summaries.append(
                {
                    "pair": f"{labels[i]} vs {labels[j]}",
                    "policy_a": labels[i],
                    "policy_b": labels[j],
                    "num_worlds": e_cnt,
                    "world_mass_wins_a": world_wins_i,
                    "world_mass_win_rate_a": round(world_wins_i / max(1, e_cnt), 6),
                    "per_world_mass_share_std": round(world_share_std, 6),
                    "mass_share_ci95_half_width": round(share_half_width, 6),
                    "mass_share_diff_ci95_half_width": round(2.0 * share_half_width, 6),
                    "mass_share_a": mass_ci["share_a"],
                    "mass_share_b": round(1.0 - mass_ci["share_a"], 6),
                    "mass_share_a_ci95": [
                        mass_ci["share_a_ci95_low"],
                        mass_ci["share_a_ci95_high"],
                    ],
                    "mass_share_diff_a_minus_b": round(2.0 * mass_ci["share_a"] - 1.0, 6),
                    "mass_share_diff_ci95": [
                        round(2.0 * mass_ci["share_a_ci95_low"] - 1.0, 6),
                        round(2.0 * mass_ci["share_a_ci95_high"] - 1.0, 6),
                    ],
                    "kills_a_on_b": k_ij_tot,
                    "kills_b_on_a": k_ji_tot,
                    "kill_balance_a_minus_b": k_ij_tot - k_ji_tot,
                    "kill_balance_per_world": kill_ci["diff_a_minus_b_per_world"],
                    "kill_balance_per_world_ci95": [
                        kill_ci["diff_per_world_ci95_low"],
                        kill_ci["diff_per_world_ci95_high"],
                    ],
                    "kill_share_a": kill_ci["share_a"],
                    "kill_share_a_ci95": [
                        kill_ci["share_a_ci95_low"],
                        kill_ci["share_a_ci95_high"],
                    ],
                    "per_world_norm_mass_a": [round(float(x), 6) for x in norm_mass_i.tolist()],
                    "per_world_norm_mass_b": [round(float(x), 6) for x in norm_mass_j.tolist()],
                }
            )

    total_agent_steps = int(e_cnt * n_cnt * total_ticks)
    return {
        "schema": V3_ARENA_EVAL_SCHEMA,
        "namespace": V3_PPO_NAMESPACE,
        "mode": "head-to-head",
        "code_commit": resolve_code_commit(),
        "seed": int(seed),
        "greedy": bool(greedy),
        "eval_config": {
            "mode": "head-to-head",
            "num_policies": p_cnt,
            "num_envs": int(num_envs),
            "num_snakes": int(n_cnt),
            "arena_radius": float(game_cfg.arena_radius),
            "burnin_ticks": int(burnin_ticks),
            "window_ticks": int(window_ticks),
            "total_ticks": int(total_ticks),
            "mass_bin_ticks": int(mass_bin_ticks),
            "bootstrap_samples": int(bootstrap_samples),
            "deterministic": bool(deterministic),
            "device": str(dev),
            "seat_rotation": "(slot + world_idx) % num_policies",
        },
        "policies": [pol.provenance_dict(policy_index=i) for i, pol in enumerate(policies)],
        "per_policy": per_policy_summary,
        "kill_matrix": {
            "columns": col_names,
            "victim_by_killer": victim_by_killer,
            "killer_by_victim": killer_by_victim,
        },
        "mass_share_over_time": mass_over_time,
        "pairwise": pairwise_summaries,
        "throughput": {
            "wall_seconds": round(wall_seconds, 4),
            "total_world_ticks": int(e_cnt * total_ticks),
            "total_agent_steps": total_agent_steps,
            "world_ticks_per_sec": round((e_cnt * total_ticks) / max(1e-9, wall_seconds), 2),
            "agent_steps_per_sec": round(total_agent_steps / max(1e-9, wall_seconds), 2),
        },
    }


@torch.no_grad()
def _evaluate_pairs_batched(
    policies: Sequence[LoadedEvalPolicy],
    batch_pairs: Sequence[tuple[int, int]],
    *,
    num_envs: int,
    burnin_ticks: int,
    window_ticks: int,
    mass_bin_ticks: int,
    greedy: bool,
    seed: int,
    bootstrap_samples: int,
    deterministic: bool,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Run ``len(batch_pairs)`` 2-policy matches simultaneously in one batched ``GameV3Sim``.

    Pair ``b`` owns worlds ``[b * num_envs, (b + 1) * num_envs)`` with identical per-pair world
    seeds and independent per-(pair, side) RNG streams, while batching each unique policy's
    network forward across every world it plays in during that tick.
    """
    b_cnt = len(batch_pairs)
    if b_cnt < 1:
        return [], 0.0
    w_cnt = int(num_envs)
    if w_cnt < 2 or w_cnt % 2 != 0:
        raise ValueError(
            f"Round-robin num_envs per pair ({w_cnt}) must be an even integer >= 2 "
            "so cyclic seat rotation (slot + world) % 2 cancels spawn-slot bias"
        )
    if burnin_ticks < 0:
        raise ValueError(f"burnin_ticks must be >= 0, got {burnin_ticks}")
    if window_ticks < 1:
        raise ValueError(f"window_ticks must be >= 1, got {window_ticks}")
    if mass_bin_ticks < 1:
        raise ValueError(f"mass_bin_ticks must be >= 1, got {mass_bin_ticks}")

    game_cfg = validate_compatible_policies(policies)
    dev = next(policies[0].model.parameters()).device
    for pol in policies:
        pol.model.eval()

    total_ticks = int(burnin_ticks) + int(window_ticks)
    num_ts_bins = max(1, int(math.ceil(total_ticks / float(mass_bin_ticks))))
    e_total = b_cnt * w_cnt

    t_wall0 = time.perf_counter()
    with preserve_rng_state():
        if dev.type == "cpu" and deterministic:
            configure_determinism(True)
        seed_everything(int(seed))

        pair_world_seeds = derive_world_seeds(int(seed), w_cnt, stream_id=701)
        all_world_seeds = pair_world_seeds * b_cnt
        sim = GameV3Sim(game_cfg, num_envs=e_total, device=dev, seeds=all_world_seeds)
        n_cnt = sim.N

        single_seat_side = build_rotated_seat_assignment(w_cnt, n_cnt, 2, device=dev)
        seats_per_world_single_np = (
            torch.stack([(single_seat_side == side).sum(dim=1) for side in range(2)], dim=1)
            .cpu()
            .numpy()
        )
        single_flat = single_seat_side.reshape(-1)
        local_f0 = torch.nonzero(single_flat == 0, as_tuple=False).squeeze(1)
        local_f1 = torch.nonzero(single_flat == 1, as_tuple=False).squeeze(1)
        local_seat_side = single_seat_side.repeat(b_cnt, 1)

        occurrences_by_pol: dict[int, list[tuple[torch.Tensor, torch.Generator | None]]] = {}
        for b_idx, (pol_i, pol_j) in enumerate(batch_pairs):
            base_flat = b_idx * w_cnt * n_cnt
            gen_0 = _make_action_generator(dev, int(seed) ^ (0x9E37_79B9 * 1))
            gen_1 = _make_action_generator(dev, int(seed) ^ (0x9E37_79B9 * 2))
            occurrences_by_pol.setdefault(pol_i, []).append((base_flat + local_f0, gen_0))
            occurrences_by_pol.setdefault(pol_j, []).append((base_flat + local_f1, gen_1))

        active_pol_plans: list[
            tuple[int, torch.Tensor, list[tuple[int, int, torch.Generator | None]]]
        ] = []
        for pol_idx in sorted(occurrences_by_pol.keys()):
            occ_list = occurrences_by_pol[pol_idx]
            cat_idx = torch.cat([f_idx for f_idx, _ in occ_list], dim=0)
            slices: list[tuple[int, int, torch.Generator | None]] = []
            cursor = 0
            for f_idx, gen_bs in occ_list:
                nxt = cursor + int(f_idx.shape[0])
                slices.append((cursor, nxt, gen_bs))
                cursor = nxt
            active_pol_plans.append((pol_idx, cat_idx, slices))

        full_obs: ObservationBatch = allocate_obs_batch(sim)
        assert_obs_batch_matches_layout(
            ObservationBatch(
                raster_s1=full_obs.raster_s1.flatten(0, 1),
                raster_s4=full_obs.raster_s4.flatten(0, 1),
                raster_s16=full_obs.raster_s16.flatten(0, 1),
                scalars=full_obs.scalars.flatten(0, 1),
                action_mask=full_obs.action_mask.flatten(0, 1),
            ),
            policies[0].obs_layout,
            batch_rows=e_total * n_cnt,
            context="arena_eval.round_robin_batched",
        )

        # Device accumulators across all B * W worlds (zero host syncs inside loop)
        ts_mass_bp = torch.zeros((b_cnt, num_ts_bins, 2), dtype=torch.int64, device=dev)
        ts_ticks = np.zeros((num_ts_bins,), dtype=np.int64)

        win_mass_ep = torch.zeros((e_total, 2), dtype=torch.int64, device=dev)
        win_steps_ep = torch.zeros((e_total, 2), dtype=torch.int64, device=dev)
        win_kills_ep = torch.zeros((e_total, 2), dtype=torch.int64, device=dev)
        win_deaths_ep = torch.zeros((e_total, 2), dtype=torch.int64, device=dev)
        win_boost_ep = torch.zeros((e_total, 2), dtype=torch.int64, device=dev)
        win_eaten_ep = torch.zeros((e_total, 2), dtype=torch.int64, device=dev)
        win_shed_ep = torch.zeros((e_total, 2), dtype=torch.int64, device=dev)

        kill_cols = 4
        world_kill_mat_1d = torch.zeros((e_total * 2 * kill_cols,), dtype=torch.int64, device=dev)
        world_ids_en = torch.arange(e_total, dtype=torch.int64, device=dev)[:, None].expand(
            e_total, n_cnt
        )
        pair_ids_en = torch.div(world_ids_en, w_cnt, rounding_mode="floor")
        death_causes_bpc_1d = torch.zeros((b_cnt * 2 * 4,), dtype=torch.int64, device=dev)

        world_actions_flat = torch.empty((e_total * n_cnt,), dtype=torch.long, device=dev)
        last_progress_mono = time.perf_counter()

        for tick in range(total_ticks):
            if should_stop is not None and should_stop():
                raise StopRequestedError(
                    f"Stop requested during batched round-robin ({b_cnt} pairs) at tick {tick}"
                )
            now_mono = time.perf_counter()
            if (now_mono - last_progress_mono) >= 90.0:
                phase_stamp(
                    "arena_eval_rr_batch",
                    "PROGRESS",
                    pairs_in_batch=b_cnt,
                    worlds=e_total,
                    tick=tick,
                    total_ticks=total_ticks,
                    elapsed_sec=round(now_mono - t_wall0, 2),
                )
                last_progress_mono = now_mono

            build_obs(sim, out=full_obs)
            r1 = full_obs.raster_s1.flatten(0, 1)
            r4 = full_obs.raster_s4.flatten(0, 1)
            r16 = full_obs.raster_s16.flatten(0, 1)
            sc = full_obs.scalars.flatten(0, 1)
            am = full_obs.action_mask.flatten(0, 1)

            for pol_idx, pol_flat_idx, pol_slices in active_pol_plans:
                pol = policies[pol_idx]
                m_logits, probs = _batched_policy_logits_and_probs(
                    pol,
                    r1.index_select(0, pol_flat_idx),
                    r4.index_select(0, pol_flat_idx),
                    r16.index_select(0, pol_flat_idx),
                    sc.index_select(0, pol_flat_idx),
                    am.index_select(0, pol_flat_idx),
                    greedy=bool(greedy),
                )
                if greedy:
                    acts_pol = m_logits.argmax(dim=-1)
                else:
                    assert probs is not None
                    acts_pol = torch.empty(
                        (int(pol_flat_idx.shape[0]),), dtype=torch.long, device=dev
                    )
                    for sl_start, sl_end, gen_bs in pol_slices:
                        acts_pol[sl_start:sl_end] = _sample_actions_from_probs_chunked(
                            probs[sl_start:sl_end],
                            actor_chunk_learners=pol.actor_chunk_learners,
                            generator=gen_bs,
                        )
                world_actions_flat.index_copy_(0, pol_flat_idx, acts_pol)

            world_actions = world_actions_flat.view(e_total, n_cnt)
            acted_es = sim.alive & (~sim.truncated_pending)
            pre_len = sim.length.clone()
            pre_boost_ticks = sim.boost_ticks.clone()

            in_window = tick >= int(burnin_ticks)
            if in_window:
                pre_hx = sim.head_pos[:, :, 0].clone()
                pre_hy = sim.head_pos[:, :, 1].clone()
                pre_theta = sim.theta.clone()
                pre_accum = sim.dist_accum_fp.clone()

            live_len_i64 = torch.where(
                acted_es, pre_len.to(torch.int64), torch.zeros_like(local_seat_side)
            )
            wp_mass_tick = torch.zeros((e_total, 2), dtype=torch.int64, device=dev)
            wp_mass_tick.scatter_add_(1, local_seat_side, live_len_i64)

            ts_idx = min(num_ts_bins - 1, tick // int(mass_bin_ticks))
            ts_mass_bp[:, ts_idx, :] += wp_mass_tick.view(b_cnt, w_cnt, 2).sum(dim=1)
            ts_ticks[ts_idx] += 1

            info = sim.step(world_actions)

            if in_window:
                win_mass_ep += wp_mass_tick
                win_steps_ep.scatter_add_(1, local_seat_side, acted_es.to(torch.int64))
                died_es = acted_es & info.died
                win_deaths_ep.scatter_add_(1, local_seat_side, died_es.to(torch.int64))
                win_kills_ep.scatter_add_(1, local_seat_side, info.kill_credit.to(torch.int64))

                boosting_es = (
                    acted_es
                    & (world_actions >= NUM_TURN_LEVELS)
                    & (pre_len > int(sim.cfg.min_boost_length))
                )
                inc_bt = torch.where(boosting_es, pre_boost_ticks + 1, pre_boost_ticks)
                shed_es = boosting_es & (inc_bt >= int(sim.cfg.boost_shed_ticks))
                win_boost_ep.scatter_add_(1, local_seat_side, boosting_es.to(torch.int64))
                win_shed_ep.scatter_add_(
                    1, local_seat_side, shed_es.to(torch.int64) * int(sim.cfg.boost_shed_mass)
                )
                win_eaten_ep.scatter_add_(1, local_seat_side, info.mass_eaten.to(torch.int64))

                ks = info.killer_id
                killer_side = local_seat_side.gather(1, ks.clamp(0, n_cnt - 1).to(torch.int64))
                killer_col = torch.where(
                    ks >= 0,
                    killer_side,
                    torch.full_like(local_seat_side, 2),
                )
                flat_km_key = (
                    world_ids_en * (2 * kill_cols) + local_seat_side * kill_cols + killer_col
                ).reshape(-1)
                world_kill_mat_1d.index_put_(
                    (flat_km_key,),
                    died_es.reshape(-1).to(torch.int64),
                    accumulate=True,
                )

                causes = classify_step_deaths(
                    sim,
                    acted_es=acted_es,
                    world_actions=world_actions,
                    pre_hx=pre_hx,
                    pre_hy=pre_hy,
                    pre_theta=pre_theta,
                    pre_len=pre_len,
                    pre_accum=pre_accum,
                    pre_boost_ticks=pre_boost_ticks,
                    died=info.died,
                    killer_id=info.killer_id,
                )
                valid_c = causes >= 0
                c_key = (pair_ids_en * 8 + local_seat_side * 4 + causes.clamp_min(0)).reshape(-1)
                death_causes_bpc_1d.index_put_(
                    (c_key,), valid_c.reshape(-1).to(torch.int64), accumulate=True
                )

        synchronize_device(dev)
        batch_wall_seconds = time.perf_counter() - t_wall0

    ts_mass_bp_np = ts_mass_bp.cpu().numpy()
    win_mass_all = win_mass_ep.view(b_cnt, w_cnt, 2).cpu().numpy()
    win_steps_all = win_steps_ep.view(b_cnt, w_cnt, 2).cpu().numpy()
    win_kills_all = win_kills_ep.view(b_cnt, w_cnt, 2).cpu().numpy()
    win_deaths_all = win_deaths_ep.view(b_cnt, w_cnt, 2).cpu().numpy()
    win_boost_all = win_boost_ep.view(b_cnt, w_cnt, 2).cpu().numpy()
    win_eaten_all = win_eaten_ep.view(b_cnt, w_cnt, 2).cpu().numpy()
    win_shed_all = win_shed_ep.view(b_cnt, w_cnt, 2).cpu().numpy()
    world_kill_all = world_kill_mat_1d.view(b_cnt, w_cnt, 2, kill_cols).cpu().numpy()
    death_causes_all = death_causes_bpc_1d.view(b_cnt, 2, 4).cpu().numpy()

    per_pair_wall = batch_wall_seconds / float(max(1, b_cnt))
    h2h_list: list[dict[str, Any]] = []
    for b_idx, (pol_i, pol_j) in enumerate(batch_pairs):
        h2h_list.append(
            _finalize_head_to_head_result(
                policies=[policies[pol_i], policies[pol_j]],
                game_cfg=game_cfg,
                dev=dev,
                num_envs=w_cnt,
                burnin_ticks=int(burnin_ticks),
                window_ticks=int(window_ticks),
                mass_bin_ticks=int(mass_bin_ticks),
                greedy=bool(greedy),
                seed=int(seed),
                bootstrap_samples=int(bootstrap_samples),
                deterministic=bool(deterministic),
                wall_seconds=per_pair_wall,
                seats_per_world_np=seats_per_world_single_np,
                ts_mass_np=ts_mass_bp_np[b_idx],
                ts_ticks=ts_ticks,
                win_mass_np=win_mass_all[b_idx],
                win_steps_np=win_steps_all[b_idx],
                win_kills_np=win_kills_all[b_idx],
                win_deaths_np=win_deaths_all[b_idx],
                win_boost_np=win_boost_all[b_idx],
                win_eaten_np=win_eaten_all[b_idx],
                win_shed_np=win_shed_all[b_idx],
                world_kill_mat_np=world_kill_all[b_idx],
                death_causes_np=death_causes_all[b_idx],
            )
        )
    return h2h_list, batch_wall_seconds


def discover_round_robin_checkpoints(
    sources: Sequence[str | Path],
    *,
    latest_per_run: bool = False,
) -> list[Path]:
    """Resolve checkpoint paths from files, directories, or `.txt`/`.json` manifests."""
    discovered: list[Path] = []
    for raw in sources:
        p = Path(raw).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Round-robin source does not exist: {p}")
        if p.is_file():
            if p.suffix == ".pt":
                discovered.append(p.resolve())
            elif p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            discovered.append(Path(item).expanduser().resolve())
                        elif isinstance(item, Mapping) and "path" in item:
                            discovered.append(Path(str(item["path"])).expanduser().resolve())
                        else:
                            raise ValueError(f"Unsupported JSON entry in {p}: {item!r}")
                else:
                    raise ValueError(f"Round-robin JSON list {p} must be a JSON array")
            else:
                for line in p.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        discovered.append(Path(stripped).expanduser().resolve())
        elif p.is_dir():
            direct_lights = sorted(
                c.resolve()
                for c in p.glob("light_*.pt")
                if ".sim_shard_" not in c.name and c.is_file()
            )
            if direct_lights:
                if latest_per_run:
                    discovered.append(direct_lights[-1])
                else:
                    discovered.extend(direct_lights)
                continue

            # Check standard run layout: <dir>/<run>/output/exports/checkpoints/light_*.pt
            nested_runs = sorted(d for d in p.iterdir() if d.is_dir())
            found_nested = False
            for sub in nested_runs:
                for ckpt_dir in (
                    sub / "output" / "exports" / "checkpoints",
                    sub / "exports" / "checkpoints",
                    sub / "checkpoints",
                    sub,
                ):
                    if ckpt_dir.is_dir():
                        lights = sorted(
                            c.resolve()
                            for c in ckpt_dir.glob("light_*.pt")
                            if ".sim_shard_" not in c.name and c.is_file()
                        )
                        if lights:
                            found_nested = True
                            discovered.append(lights[-1])
                            break
            if not found_nested:
                any_pt = sorted(
                    c.resolve()
                    for c in p.glob("*.pt")
                    if ".sim_shard_" not in c.name
                    and not c.name.startswith("full_")
                    and c.is_file()
                )
                if latest_per_run and any_pt:
                    discovered.append(any_pt[-1])
                else:
                    discovered.extend(any_pt)

    # Deduplicate preserving order
    deduped: list[Path] = []
    seen: set[Path] = set()
    for c in discovered:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    if len(deduped) < 2:
        raise ValueError(
            f"Round-robin requires at least 2 checkpoints, discovered {len(deduped)} from {sources}"
        )
    return deduped


def evaluate_round_robin(
    policies: Sequence[LoadedEvalPolicy],
    *,
    num_envs: int = DEFAULT_H2H_NUM_ENVS,
    burnin_ticks: int = DEFAULT_H2H_BURNIN_TICKS,
    window_ticks: int = DEFAULT_H2H_WINDOW_TICKS,
    mass_bin_ticks: int = DEFAULT_H2H_MASS_BIN_TICKS,
    greedy: bool = False,
    seed: int = DEFAULT_EVAL_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bt_prior_wins: float = DEFAULT_BT_PRIOR_WINS,
    deterministic: bool = True,
    batched: bool = True,
    max_pairs_per_batch: int | None = None,
    vram_budget_bytes: int | None = None,
    output_dir: Path | None = None,
    verified_pair_paths: Mapping[str, Path] | None = None,
    max_wall_seconds: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Play every unordered pair of ``policies`` (batched in one simulator by default).

    Ranks policies primarily by mean pairwise mass share (with synchronous paired world-bootstrap
    95% CIs) and reports regularized Bradley-Terry Elo on per-world wins as a secondary column.
    """
    m_cnt = len(policies)
    if m_cnt < 2:
        raise ValueError(f"Round-robin requires >= 2 policies, got {m_cnt}")
    if int(num_envs) < 2 or int(num_envs) % 2 != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be a multiple of 2 in round-robin "
            "so cyclic seat rotation (slot + world) % 2 cancels spawn-slot bias"
        )
    if max_pairs_per_batch is not None and int(max_pairs_per_batch) < 1:
        raise ValueError(f"max_pairs_per_batch must be >= 1, got {max_pairs_per_batch}")
    game_cfg = validate_compatible_policies(policies)
    dev = next(policies[0].model.parameters()).device
    labels = [p.label for p in policies]

    t_start = time.perf_counter()
    pairs_dir = (Path(output_dir) / "pairs") if output_dir is not None else None
    if pairs_dir is not None:
        pairs_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: list[tuple[int, int]] = [(i, j) for i in range(m_cnt) for j in range(i + 1, m_cnt)]
    total_pairs = len(all_pairs)
    reused_pairs: list[str] = []
    h2h_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    pending_pairs: list[tuple[int, int]] = []

    for i, j in all_pairs:
        unit_id = f"pair_{i:02d}_{j:02d}"
        expected_key = build_pair_reuse_key(
            num_envs=int(num_envs),
            burnin_ticks=int(burnin_ticks),
            window_ticks=int(window_ticks),
            greedy=bool(greedy),
            seed=int(seed),
            policy_a_sha256=policies[i].source_checkpoint_sha256,
            policy_b_sha256=policies[j].source_checkpoint_sha256,
        )
        loaded_unit: dict[str, Any] | None = None
        if verified_pair_paths and unit_id in verified_pair_paths:
            vp = Path(verified_pair_paths[unit_id])
            verify_completed_artifact(vp)
            loaded_unit = json.loads(vp.read_text(encoding="utf-8"))
        elif (
            pairs_dir is not None
            and (pairs_dir / f"{unit_id}.json").is_file()
            and (pairs_dir / f"{unit_id}.json.complete").is_file()
        ):
            vp = pairs_dir / f"{unit_id}.json"
            verify_completed_artifact(vp)
            loaded_unit = json.loads(vp.read_text(encoding="utf-8"))

        if loaded_unit is not None:
            validate_pair_unit_reuse(unit_id, loaded_unit, expected_key)
            reused_pairs.append(unit_id)
            h2h_by_pair[(i, j)] = loaded_unit["match"]
        else:
            pending_pairs.append((i, j))

    def _coop_stop() -> bool:
        if should_stop is not None and should_stop():
            return True
        if max_wall_seconds is not None:
            return (time.perf_counter() - t_start) >= float(max_wall_seconds)
        return False

    def _export_pair_unit(i: int, j: int, h2h_payload: dict[str, Any]) -> None:
        if pairs_dir is None:
            return
        unit_id = f"pair_{i:02d}_{j:02d}"
        unit_file = pairs_dir / f"{unit_id}.json"
        if not artifact_or_sidecar_exists(unit_file):
            write_create_only_json(
                unit_file,
                {
                    "schema": V3_ARENA_EVAL_PAIR_SCHEMA,
                    "unit_id": unit_id,
                    "pair_indices": [i, j],
                    "seed": int(seed),
                    "reuse_key": build_pair_reuse_key(
                        num_envs=int(num_envs),
                        burnin_ticks=int(burnin_ticks),
                        window_ticks=int(window_ticks),
                        greedy=bool(greedy),
                        seed=int(seed),
                        policy_a_sha256=policies[i].source_checkpoint_sha256,
                        policy_b_sha256=policies[j].source_checkpoint_sha256,
                    ),
                    "policies": h2h_payload["policies"],
                    "match": h2h_payload,
                },
            )

    net_sizes_all: list[NetSize] = [p.net_size for p in policies]
    eff_amp = any(bool(p.model.cfg.use_bf16_autocast) for p in policies) and dev.type == "cuda"
    if not batched:
        eff_pairs_per_batch = 1
    elif pending_pairs:
        if vram_budget_bytes is not None or dev.type == "cuda":
            if vram_budget_bytes is not None:
                eff_budget = int(vram_budget_bytes)
            else:
                total_vram = int(torch.cuda.get_device_properties(dev).total_memory)
                eff_budget = int(total_vram * 0.92)
            eff_pairs_per_batch, _ = select_pairs_per_batch(
                game_cfg=game_cfg,
                net_sizes=net_sizes_all,
                num_envs_per_pair=int(num_envs),
                total_pairs_to_run=len(pending_pairs),
                vram_budget_bytes=eff_budget,
                amp=eff_amp,
                max_pairs_override=max_pairs_per_batch,
                gpu_label=str(dev),
            )
        elif max_pairs_per_batch is not None:
            eff_pairs_per_batch = min(len(pending_pairs), int(max_pairs_per_batch))
        else:
            eff_pairs_per_batch = len(pending_pairs)
    else:
        eff_pairs_per_batch = (
            int(max_pairs_per_batch) if max_pairs_per_batch is not None else max(1, total_pairs)
        )

    batches_executed = 0
    sim_compute_wall_sec = 0.0
    if pending_pairs:
        if batched:
            total_batches = int(math.ceil(len(pending_pairs) / float(eff_pairs_per_batch)))
            for b_ord, start_idx in enumerate(
                range(0, len(pending_pairs), eff_pairs_per_batch), start=1
            ):
                batch_slice = pending_pairs[start_idx : start_idx + eff_pairs_per_batch]
                first_uid = f"pair_{batch_slice[0][0]:02d}_{batch_slice[0][1]:02d}"
                if should_stop is not None and should_stop():
                    raise StopRequestedError(
                        f"Stop requested before round-robin batch {b_ord}/{total_batches} ({first_uid})"
                    )
                if (
                    max_wall_seconds is not None
                    and (time.perf_counter() - t_start) >= max_wall_seconds
                ):
                    raise StopRequestedError(
                        f"Wall guard ({max_wall_seconds:.1f}s) reached before batch {b_ord}/{total_batches} "
                        f"({len(h2h_by_pair)}/{total_pairs} pairs completed)"
                    )
                phase_stamp(
                    "arena_eval_round_robin",
                    "BATCH_START",
                    batch_ordinal=b_ord,
                    total_batches=total_batches,
                    pairs_in_batch=len(batch_slice),
                    worlds_in_batch=len(batch_slice) * int(num_envs),
                )
                batch_h2h, batch_wall = _evaluate_pairs_batched(
                    policies,
                    batch_slice,
                    num_envs=num_envs,
                    burnin_ticks=burnin_ticks,
                    window_ticks=window_ticks,
                    mass_bin_ticks=mass_bin_ticks,
                    greedy=greedy,
                    seed=seed,
                    bootstrap_samples=bootstrap_samples,
                    deterministic=deterministic,
                    should_stop=_coop_stop,
                )
                batches_executed += 1
                sim_compute_wall_sec += batch_wall
                for (i, j), h2h_item in zip(batch_slice, batch_h2h):
                    h2h_by_pair[(i, j)] = h2h_item
                    _export_pair_unit(i, j, h2h_item)
        else:
            for p_ord, (i, j) in enumerate(pending_pairs, start=1):
                unit_id = f"pair_{i:02d}_{j:02d}"
                if should_stop is not None and should_stop():
                    raise StopRequestedError(f"Stop requested before round-robin {unit_id}")
                if (
                    max_wall_seconds is not None
                    and (time.perf_counter() - t_start) >= max_wall_seconds
                ):
                    raise StopRequestedError(
                        f"Wall guard ({max_wall_seconds:.1f}s) reached before {unit_id} "
                        f"({len(h2h_by_pair)}/{total_pairs} pairs completed)"
                    )
                phase_stamp(
                    "arena_eval_round_robin",
                    "PAIR_START",
                    pair_ordinal=p_ord,
                    pending_pairs=len(pending_pairs),
                    pair=f"{labels[i]} vs {labels[j]}",
                )
                h2h_item = evaluate_head_to_head(
                    [policies[i], policies[j]],
                    num_envs=num_envs,
                    burnin_ticks=burnin_ticks,
                    window_ticks=window_ticks,
                    mass_bin_ticks=mass_bin_ticks,
                    greedy=greedy,
                    seed=seed,
                    bootstrap_samples=bootstrap_samples,
                    deterministic=deterministic,
                    should_stop=_coop_stop,
                )
                batches_executed += 1
                sim_compute_wall_sec += float(h2h_item["throughput"]["wall_seconds"])
                h2h_by_pair[(i, j)] = h2h_item
                _export_pair_unit(i, j, h2h_item)

    pair_results: list[dict[str, Any]] = []
    wins_mat = np.zeros((m_cnt, m_cnt), dtype=np.float64)
    games_mat = np.zeros((m_cnt, m_cnt), dtype=np.float64)
    mass_share_mat = np.full((m_cnt, m_cnt), np.nan, dtype=np.float64)
    per_world_wins_dict: dict[tuple[int, int], np.ndarray] = {}
    per_world_norm_mass_dict: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    agg_steps = np.zeros((m_cnt,), dtype=np.int64)
    agg_mass_sum = np.zeros((m_cnt,), dtype=np.float64)
    agg_kills = np.zeros((m_cnt,), dtype=np.int64)
    agg_deaths = np.zeros((m_cnt,), dtype=np.int64)
    agg_net_kill_bal = np.zeros((m_cnt,), dtype=np.int64)
    pair_shares_list: list[list[float]] = [[] for _ in range(m_cnt)]

    for i, j in all_pairs:
        unit_id = f"pair_{i:02d}_{j:02d}"
        h2h = h2h_by_pair[(i, j)]
        pw = dict(h2h["pairwise"][0])
        if "mass_share_ci95_half_width" not in pw:
            ci_s = pw["mass_share_a_ci95"]
            hw = 0.5 * (float(ci_s[1]) - float(ci_s[0]))
            pw["mass_share_ci95_half_width"] = round(hw, 6)
            pw["mass_share_diff_ci95_half_width"] = round(2.0 * hw, 6)
        pair_results.append(
            {
                "unit_id": unit_id,
                "i": i,
                "j": j,
                **pw,
                "wall_seconds": h2h["throughput"]["wall_seconds"],
            }
        )

        e_w = int(pw["num_worlds"])
        w_ij = float(pw["world_mass_wins_a"])
        wins_mat[i, j] = w_ij
        wins_mat[j, i] = float(e_w) - w_ij
        games_mat[i, j] = float(e_w)
        games_mat[j, i] = float(e_w)

        sh_i = float(pw["mass_share_a"])
        mass_share_mat[i, j] = sh_i
        mass_share_mat[j, i] = 1.0 - sh_i
        pair_shares_list[i].append(sh_i)
        pair_shares_list[j].append(1.0 - sh_i)

        ma = np.asarray(pw["per_world_norm_mass_a"], dtype=np.float64)
        mb = np.asarray(pw["per_world_norm_mass_b"], dtype=np.float64)
        per_world_norm_mass_dict[(i, j)] = (ma, mb)
        per_world_wins_dict[(i, j)] = np.where(ma > mb, 1.0, np.where(ma == mb, 0.5, 0.0))

        p0, p1 = h2h["per_policy"][0], h2h["per_policy"][1]
        agg_steps[i] += int(p0["acted_steps"])
        agg_steps[j] += int(p1["acted_steps"])
        agg_mass_sum[i] += float(p0["time_averaged_length"]) * int(p0["acted_steps"])
        agg_mass_sum[j] += float(p1["time_averaged_length"]) * int(p1["acted_steps"])
        agg_kills[i] += int(p0["kills"])
        agg_kills[j] += int(p1["kills"])
        agg_deaths[i] += int(p0["deaths"])
        agg_deaths[j] += int(p1["deaths"])
        bal = int(pw["kill_balance_a_minus_b"])
        agg_net_kill_bal[i] += bal
        agg_net_kill_bal[j] -= bal

    mean_share_pt = np.array(
        [float(np.mean(pair_shares_list[p_i])) for p_i in range(m_cnt)], dtype=np.float64
    )
    elo_pt = fit_bradley_terry_elo(wins_mat, games_mat, prior_wins=bt_prior_wins)

    # Synchronous paired world-bootstrap 95% CIs for BOTH headline Mean Pairwise Mass Share and BT Elo
    e_worlds = int(num_envs)
    if e_worlds > 1 and int(bootstrap_samples) > 0:
        rng = np.random.Generator(
            np.random.PCG64((int(seed) ^ 0xA5A5_5A5A) & 0xFFFF_FFFF_FFFF_FFFF)
        )
        bs_mean_shares = np.zeros((int(bootstrap_samples), m_cnt), dtype=np.float64)
        bs_elos = np.empty((int(bootstrap_samples), m_cnt), dtype=np.float64)
        denom_opp = float(max(1, m_cnt - 1))
        for b_idx in range(int(bootstrap_samples)):
            w_sample = rng.integers(0, e_worlds, size=(e_worlds,))
            bs_w = np.zeros((m_cnt, m_cnt), dtype=np.float64)
            for (i, j), w_vec in per_world_wins_dict.items():
                s_ij = float(w_vec[w_sample].sum())
                bs_w[i, j] = s_ij
                bs_w[j, i] = float(e_worlds) - s_ij
                ma, mb = per_world_norm_mass_dict[(i, j)]
                sa = float(ma[w_sample].sum())
                sb = float(mb[w_sample].sum())
                sh_ij = sa / (sa + sb) if (sa + sb) > 0.0 else 0.5
                bs_mean_shares[b_idx, i] += sh_ij
                bs_mean_shares[b_idx, j] += 1.0 - sh_ij
            bs_mean_shares[b_idx] /= denom_opp
            bs_elos[b_idx] = fit_bradley_terry_elo(bs_w, games_mat, prior_wins=bt_prior_wins)
        share_lo = np.percentile(bs_mean_shares, 2.5, axis=0)
        share_hi = np.percentile(bs_mean_shares, 97.5, axis=0)
        elo_lo = np.percentile(bs_elos, 2.5, axis=0)
        elo_hi = np.percentile(bs_elos, 97.5, axis=0)
    else:
        share_lo = mean_share_pt.copy()
        share_hi = mean_share_pt.copy()
        elo_lo = elo_pt.copy()
        elo_hi = elo_pt.copy()

    # Headline ranking by Mean Pairwise Mass Share descending (tie-break by BT Elo, then policy_index)
    order = sorted(
        range(m_cnt),
        key=lambda p_i: (-round(float(mean_share_pt[p_i]), 9), -float(elo_pt[p_i]), p_i),
    )
    standings: list[dict[str, Any]] = []
    for rank_idx, p_i in enumerate(order, start=1):
        tot_w = float(wins_mat[p_i].sum())
        tot_g = float(games_mat[p_i].sum())
        steps_i = int(agg_steps[p_i])
        d1k = max(1, steps_i) / 1000.0
        sh_lo_i = round(float(share_lo[p_i]), 6)
        sh_hi_i = round(float(share_hi[p_i]), 6)
        standings.append(
            {
                "rank": rank_idx,
                "policy_index": p_i,
                "label": labels[p_i],
                "net_size": policies[p_i].net_size,
                "mean_pairwise_mass_share": round(float(mean_share_pt[p_i]), 6),
                "mean_pairwise_mass_share_ci95": [sh_lo_i, sh_hi_i],
                "mean_pairwise_mass_share_ci95_half_width": round(0.5 * (sh_hi_i - sh_lo_i), 6),
                "elo": round(float(elo_pt[p_i]), 2),
                "elo_ci95": [round(float(elo_lo[p_i]), 2), round(float(elo_hi[p_i]), 2)],
                "world_wins": round(tot_w, 1),
                "world_games": int(tot_g),
                "world_win_rate": round(tot_w / max(1.0, tot_g), 6),
                "time_averaged_length": round(agg_mass_sum[p_i] / max(1, steps_i), 4),
                "kills_per_1k_agent_steps": round(int(agg_kills[p_i]) / d1k, 4),
                "deaths_per_1k_agent_steps": round(int(agg_deaths[p_i]) / d1k, 4),
                "net_kill_balance": int(agg_net_kill_bal[p_i]),
            }
        )

    wall_total = time.perf_counter() - t_start
    ticks_per_pair = int(burnin_ticks) + int(window_ticks)
    total_sim_ticks_executed = int(batches_executed * ticks_per_pair)
    total_steps_all = int(total_pairs * num_envs * game_cfg.num_snakes * ticks_per_pair)
    pairwise_mass_grid = {
        labels[i]: {
            labels[j]: (None if i == j else round(float(mass_share_mat[i, j]), 6))
            for j in range(m_cnt)
        }
        for i in range(m_cnt)
    }

    return {
        "schema": V3_ARENA_EVAL_SCHEMA,
        "namespace": V3_PPO_NAMESPACE,
        "mode": "round-robin",
        "code_commit": resolve_code_commit(),
        "seed": int(seed),
        "greedy": bool(greedy),
        "primary_ranking_metric": "mean_pairwise_mass_share",
        "rating_model": (
            "Headline ranking: mean pairwise mass share with synchronous paired world-bootstrap "
            f"95% CIs. Secondary column: regularized Bradley-Terry Elo on per-world mass-share wins "
            f"(prior_wins={bt_prior_wins}; secondary only because per-world win counts saturate "
            "under sweeps when world-to-world mass-share spread is ~0.5%)."
        ),
        "eval_config": {
            "mode": "round-robin",
            "num_policies": m_cnt,
            "num_pairs": total_pairs,
            "num_envs": int(num_envs),
            "num_snakes": int(game_cfg.num_snakes),
            "arena_radius": float(game_cfg.arena_radius),
            "burnin_ticks": int(burnin_ticks),
            "window_ticks": int(window_ticks),
            "mass_bin_ticks": int(mass_bin_ticks),
            "bootstrap_samples": int(bootstrap_samples),
            "bt_prior_wins": float(bt_prior_wins),
            "deterministic": bool(deterministic),
            "batched_pairs": bool(batched),
            "pairs_per_batch": int(eff_pairs_per_batch),
            "device": str(dev),
        },
        "policies": [pol.provenance_dict(policy_index=i) for i, pol in enumerate(policies)],
        "standings": standings,
        "pairwise_mass_share_matrix": pairwise_mass_grid,
        "pairs": pair_results,
        "reused_pair_units": reused_pairs,
        "throughput": {
            "wall_seconds": round(wall_total, 4),
            "sim_compute_wall_seconds": round(sim_compute_wall_sec, 4),
            "total_pairs": total_pairs,
            "reused_pairs": len(reused_pairs),
            "computed_pairs": total_pairs - len(reused_pairs),
            "batches_executed": batches_executed,
            "total_sim_ticks_executed": total_sim_ticks_executed,
            "ms_per_sim_tick": (
                round(1000.0 * sim_compute_wall_sec / float(total_sim_ticks_executed), 4)
                if total_sim_ticks_executed > 0
                else 0.0
            ),
            "nominal_total_agent_steps": total_steps_all,
        },
    }


def _format_bin_Row(
    *,
    bin_label: str,
    acted_steps: int,
    alive_length_sum: int,
    net_alive_growth: int,
    deaths: int,
    kills: int,
    boost_ticks: int,
    mass_shed_boost: int,
    mass_eaten: int,
    turn_counts: Sequence[int],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    denom_steps = max(1, int(acted_steps))
    denom_1k = denom_steps / 1000.0
    t_counts = [int(x) for x in turn_counts]
    turn_shares = {TURN_LEVEL_NAMES[idx]: round(t_counts[idx] / denom_steps, 6) for idx in range(5)}
    row: dict[str, Any] = {
        "bin": bin_label,
        "acted_steps": int(acted_steps),
        "deaths": int(deaths),
        "kills": int(kills),
        "boost_ticks": int(boost_ticks),
        "mass_eaten": int(mass_eaten),
        "mass_shed_boost": int(mass_shed_boost),
        "net_alive_length_gain": int(net_alive_growth),
        "time_averaged_length": round(float(alive_length_sum) / denom_steps, 4),
        "growth_per_1k_ticks": round(float(net_alive_growth) / denom_1k, 4),
        "deaths_per_1k_ticks": round(float(deaths) / denom_1k, 4),
        "kills_per_1k_ticks": round(float(kills) / denom_1k, 4),
        "boost_share": round(float(boost_ticks) / denom_steps, 6),
        "mass_shed_boost_per_1k_ticks": round(float(mass_shed_boost) / denom_1k, 4),
        "mass_eaten_per_1k_ticks": round(float(mass_eaten) / denom_1k, 4),
        "turn_counts": {TURN_LEVEL_NAMES[idx]: t_counts[idx] for idx in range(5)},
        "turn_level_shares": turn_shares,
        "turn_direction_shares": {
            "left": round((t_counts[0] + t_counts[1]) / denom_steps, 6),
            "straight": round(t_counts[2] / denom_steps, 6),
            "right": round((t_counts[3] + t_counts[4]) / denom_steps, 6),
        },
    }
    if extra:
        row.update(extra)
    return row


@torch.no_grad()
def evaluate_long_game(
    policy: LoadedEvalPolicy,
    *,
    num_envs: int = DEFAULT_LONG_GAME_NUM_ENVS,
    burnin_ticks: int = DEFAULT_LONG_GAME_BURNIN_TICKS,
    window_ticks: int = DEFAULT_LONG_GAME_WINDOW_TICKS,
    learner_slots: int | None = None,
    opponent_policy: OpponentPolicy = "food_seeking",
    num_world_age_bins: int = DEFAULT_LONG_GAME_WORLD_AGE_BINS,
    life_age_edges: Sequence[int] = DEFAULT_LIFE_AGE_EDGES,
    victim_length_edges: Sequence[int] = DEFAULT_VICTIM_LENGTH_EDGES,
    greedy: bool = False,
    seed: int = DEFAULT_EVAL_SEED,
    deterministic: bool = True,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate a single policy over a mature long-game horizon binned by world age and life age."""
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    if burnin_ticks < 0:
        raise ValueError(f"burnin_ticks must be >= 0, got {burnin_ticks}")
    if window_ticks < 6:
        raise ValueError(f"window_ticks must be >= 6 (for >=6 world-age bins), got {window_ticks}")
    if num_world_age_bins < 6:
        raise ValueError(f"num_world_age_bins must be >= 6, got {num_world_age_bins}")
    if opponent_policy not in VALID_OPPONENT_POLICIES:
        raise ValueError(
            f"opponent_policy must be one of {VALID_OPPONENT_POLICIES}, got {opponent_policy!r}"
        )

    l_edges = tuple(int(x) for x in life_age_edges)
    if len(l_edges) < 5 or any(x <= 0 for x in l_edges) or list(l_edges) != sorted(set(l_edges)):
        raise ValueError(
            f"life_age_edges must be >=5 strictly increasing positive ints (giving >=6 bins), got {l_edges}"
        )
    v_edges = tuple(int(x) for x in victim_length_edges)
    if len(v_edges) < 1 or any(x <= 0 for x in v_edges) or list(v_edges) != sorted(set(v_edges)):
        raise ValueError(
            f"victim_length_edges must be strictly increasing positive ints, got {v_edges}"
        )

    game_cfg = validate_compatible_policies([policy])
    n_snakes = int(game_cfg.num_snakes)
    k_slots = n_snakes if learner_slots is None else int(learner_slots)
    if not (1 <= k_slots <= n_snakes):
        raise ValueError(f"learner_slots must be in [1, {n_snakes}], got {k_slots}")

    dev = next(policy.model.parameters()).device
    policy.model.eval()
    t_wall0 = time.perf_counter()

    # Build burn-in world-age bins + >=6 post-burn-in window world-age bins
    win_bin_width = float(window_ticks) / float(num_world_age_bins)
    burn_bins = (
        max(1, int(math.ceil(float(burnin_ticks) / max(1.0, win_bin_width))))
        if burnin_ticks > 0
        else 0
    )
    world_bin_specs: list[tuple[int, int, str]] = []
    for b_i in range(burn_bins):
        t0 = int(round(b_i * float(burnin_ticks) / burn_bins))
        t1 = int(round((b_i + 1) * float(burnin_ticks) / burn_bins))
        if t1 > t0:
            world_bin_specs.append((t0, t1, "burnin"))
    for w_i in range(int(num_world_age_bins)):
        t0 = int(burnin_ticks) + int(round(w_i * float(window_ticks) / int(num_world_age_bins)))
        t1 = int(burnin_ticks) + int(
            round((w_i + 1) * float(window_ticks) / int(num_world_age_bins))
        )
        world_bin_specs.append((t0, t1, "window"))

    total_ticks = int(burnin_ticks) + int(window_ticks)
    tick_to_wbin = np.zeros((total_ticks,), dtype=np.int64)
    for b_idx, (t0, t1, _) in enumerate(world_bin_specs):
        tick_to_wbin[t0:t1] = b_idx
    num_wbins = len(world_bin_specs)
    num_lbins = len(l_edges) + 1
    num_vbins = len(v_edges) + 1

    with preserve_rng_state():
        if dev.type == "cpu" and deterministic:
            configure_determinism(True)
        seed_everything(int(seed))

        world_seeds = derive_world_seeds(int(seed), int(num_envs), stream_id=702)
        sim = GameV3Sim(game_cfg, num_envs=int(num_envs), device=dev, seeds=world_seeds)
        e_cnt = sim.E

        learner_mask = build_rotated_learner_mask(e_cnt, n_snakes, k_slots, device=dev)
        learner_idx = LearnerIndex(sim, learner_mask)
        obs_buf = allocate_obs_batch(sim, learner_idx)
        assert_obs_batch_matches_layout(
            obs_buf,
            policy.obs_layout,
            batch_rows=learner_idx.count,
            context="arena_eval.long_game",
        )
        gen = _make_action_generator(dev, int(seed) ^ 0xC2B2_AE35)

        l_edges_t = torch.tensor(l_edges, dtype=torch.int64, device=dev)
        v_edges_t = torch.tensor(v_edges, dtype=torch.int64, device=dev)

        # Accumulator columns:
        # 0:acted, 1:len_sum, 2:net_alive_growth, 3:deaths, 4:kills, 5:boost, 6:shed, 7:eaten,
        # 8..12:turn_0..4, 13:older_500_ticks, 14:stuck_ticks
        n_stat_cols = 15
        wbin_stats = torch.zeros((num_wbins, n_stat_cols), dtype=torch.int64, device=dev)
        lbin_stats = torch.zeros((num_lbins, n_stat_cols), dtype=torch.int64, device=dev)

        # World max snake length per tick (small: total_ticks * E int32s -> quantiles per world-age bin)
        world_max_len_te = torch.zeros((total_ticks, e_cnt), dtype=torch.int32, device=dev)
        learner_max_len_te = torch.zeros((total_ticks, e_cnt), dtype=torch.int32, device=dev)

        # Victim-length bin x 4 death causes in post-burn-in window
        victim_death_causes_1d = torch.zeros((num_vbins * 4,), dtype=torch.int64, device=dev)

        start_len = int(game_cfg.start_length)
        env_idx = learner_idx.env_idx
        slot_idx = learner_idx.slot_idx
        all_learners = k_slots == n_snakes
        last_progress_mono = time.perf_counter()

        for tick in range(total_ticks):
            if should_stop is not None and should_stop():
                raise StopRequestedError(f"Stop requested during long-game at tick {tick}")
            now_mono = time.perf_counter()
            if (now_mono - last_progress_mono) >= 90.0:
                phase_stamp(
                    "arena_eval_long_game",
                    "PROGRESS",
                    tick=tick,
                    total_ticks=total_ticks,
                    elapsed_sec=round(now_mono - t_wall0, 2),
                )
                last_progress_mono = now_mono

            build_obs(sim, learner_index=learner_idx, out=obs_buf)
            l_acts = act_policy_chunked(
                policy,
                obs_buf.raster_s1,
                obs_buf.raster_s4,
                obs_buf.raster_s16,
                obs_buf.scalars,
                obs_buf.action_mask,
                greedy=greedy,
                generator=gen,
            )
            if all_learners:
                world_actions = l_acts.view(e_cnt, n_snakes)
            else:
                world_actions = scripted_opponent_actions(sim, opponent_policy)
                world_actions[env_idx, slot_idx] = l_acts

            acted_es = sim.alive & (~sim.truncated_pending)
            pre_len = sim.length.clone()
            pre_life_age = sim.life_age.clone()
            pre_boost_ticks = sim.boost_ticks.clone()

            in_window = tick >= int(burnin_ticks)
            if in_window:
                pre_hx = sim.head_pos[:, :, 0].clone()
                pre_hy = sim.head_pos[:, :, 1].clone()
                pre_theta = sim.theta.clone()
                pre_accum = sim.dist_accum_fp.clone()

            alive_lens_all = torch.where(sim.alive, pre_len, torch.zeros_like(pre_len))
            world_max_len_te[tick] = alive_lens_all.amax(dim=1)
            learner_lens_all = torch.where(
                sim.alive & learner_mask, pre_len, torch.zeros_like(pre_len)
            )
            learner_max_len_te[tick] = learner_lens_all.amax(dim=1)

            info = sim.step(world_actions)

            # Extract learner-slot vectors of length M = E * K
            acted_m = acted_es[env_idx, slot_idx]
            acted_i64 = acted_m.to(torch.int64)
            pre_len_m = pre_len[env_idx, slot_idx].to(torch.int64)
            post_len_m = sim.length[env_idx, slot_idx].to(torch.int64)
            life_age_m = pre_life_age[env_idx, slot_idx].to(torch.int64)
            died_m = acted_m & info.died[env_idx, slot_idx]
            survived_m = acted_m & (~info.died[env_idx, slot_idx])
            kills_m = torch.where(acted_m, info.kill_credit[env_idx, slot_idx].to(torch.int64), 0)
            eaten_m = torch.where(acted_m, info.mass_eaten[env_idx, slot_idx].to(torch.int64), 0)

            act_m = l_acts.clamp(0, NUM_ACTIONS - 1)
            steer_m = act_m.remainder(NUM_TURN_LEVELS)
            boost_m = (
                acted_m & (act_m >= NUM_TURN_LEVELS) & (pre_len_m > int(sim.cfg.min_boost_length))
            )
            pre_bt_m = pre_boost_ticks[env_idx, slot_idx]
            inc_bt_m = torch.where(boost_m, pre_bt_m + 1, pre_bt_m)
            shed_m = (boost_m & (inc_bt_m >= int(sim.cfg.boost_shed_ticks))).to(torch.int64) * int(
                sim.cfg.boost_shed_mass
            )
            net_growth_m = torch.where(survived_m, post_len_m - pre_len_m, 0)

            older_500_m = acted_m & (life_age_m > STUCK_MIN_LIFE_AGE_TICKS)
            stuck_m = older_500_m & (
                torch.abs(pre_len_m - start_len) <= STUCK_SPAWN_LENGTH_TOLERANCE
            )

            wb = int(tick_to_wbin[tick])
            t0_m = acted_m & (steer_m == 0)
            t1_m = acted_m & (steer_m == 1)
            t2_m = acted_m & (steer_m == 2)
            t3_m = acted_m & (steer_m == 3)
            t4_m = acted_m & (steer_m == 4)

            step_vec = torch.stack(
                (
                    acted_i64.sum(),
                    torch.where(acted_m, pre_len_m, 0).sum(),
                    net_growth_m.sum(),
                    died_m.to(torch.int64).sum(),
                    kills_m.sum(),
                    boost_m.to(torch.int64).sum(),
                    shed_m.sum(),
                    eaten_m.sum(),
                    t0_m.to(torch.int64).sum(),
                    t1_m.to(torch.int64).sum(),
                    t2_m.to(torch.int64).sum(),
                    t3_m.to(torch.int64).sum(),
                    t4_m.to(torch.int64).sum(),
                    older_500_m.to(torch.int64).sum(),
                    stuck_m.to(torch.int64).sum(),
                )
            )
            wbin_stats[wb] += step_vec

            if in_window:
                lbin_m = torch.bucketize(life_age_m, l_edges_t, right=True)
                per_row = torch.stack(
                    (
                        acted_i64,
                        torch.where(acted_m, pre_len_m, 0),
                        net_growth_m,
                        died_m.to(torch.int64),
                        kills_m,
                        boost_m.to(torch.int64),
                        shed_m,
                        eaten_m,
                        t0_m.to(torch.int64),
                        t1_m.to(torch.int64),
                        t2_m.to(torch.int64),
                        t3_m.to(torch.int64),
                        t4_m.to(torch.int64),
                        older_500_m.to(torch.int64),
                        stuck_m.to(torch.int64),
                    ),
                    dim=1,
                )
                lbin_stats.index_add_(0, lbin_m, per_row)

                causes_es = classify_step_deaths(
                    sim,
                    acted_es=acted_es,
                    world_actions=world_actions,
                    pre_hx=pre_hx,
                    pre_hy=pre_hy,
                    pre_theta=pre_theta,
                    pre_len=pre_len,
                    pre_accum=pre_accum,
                    pre_boost_ticks=pre_boost_ticks,
                    died=info.died,
                    killer_id=info.killer_id,
                )
                causes_m = causes_es[env_idx, slot_idx]
                valid_d = died_m & (causes_m >= 0)
                vbin_m = torch.bucketize(pre_len_m, v_edges_t, right=True)
                vd_key = vbin_m * 4 + causes_m.clamp_min(0)
                victim_death_causes_1d.index_put_(
                    (vd_key,), valid_d.to(torch.int64), accumulate=True
                )

        synchronize_device(dev)
        wall_seconds = time.perf_counter() - t_wall0

    wbin_np = wbin_stats.cpu().numpy()
    lbin_np = lbin_stats.cpu().numpy()
    wmax_np = world_max_len_te.cpu().numpy()
    lmax_np = learner_max_len_te.cpu().numpy()
    vdc_np = victim_death_causes_1d.view(num_vbins, 4).cpu().numpy()

    # Format world-age bins (burn-in + >=6 post-burn-in window bins) & largest snake per world
    world_age_bins: list[dict[str, Any]] = []
    largest_snake_over_time: list[dict[str, Any]] = []
    for b_idx, (t0, t1, phase) in enumerate(world_bin_specs):
        r = wbin_np[b_idx]
        older_cnt = int(r[13])
        stuck_cnt = int(r[14])
        slice_wmax = wmax_np[t0:t1].reshape(-1)
        slice_lmax = lmax_np[t0:t1].reshape(-1)
        max_stats = {
            "mean_world_max_length": round(float(slice_wmax.mean()), 4),
            "median_world_max_length": round(float(np.median(slice_wmax)), 4),
            "p90_world_max_length": round(float(np.percentile(slice_wmax, 90)), 4),
            "peak_world_max_length": int(slice_wmax.max()),
            "mean_learner_max_length": round(float(slice_lmax.mean()), 4),
        }
        lbl = f"[{t0}, {t1})"
        world_age_bins.append(
            _format_bin_Row(
                bin_label=lbl,
                acted_steps=int(r[0]),
                alive_length_sum=int(r[1]),
                net_alive_growth=int(r[2]),
                deaths=int(r[3]),
                kills=int(r[4]),
                boost_ticks=int(r[5]),
                mass_shed_boost=int(r[6]),
                mass_eaten=int(r[7]),
                turn_counts=r[8:13].tolist(),
                extra={
                    "tick_start": t0,
                    "tick_end": t1,
                    "phase": phase,
                    "older_than_500_snake_ticks": older_cnt,
                    "stuck_snake_ticks": stuck_cnt,
                    "stuck_share": (
                        round(stuck_cnt / max(1, older_cnt), 6) if older_cnt > 0 else 0.0
                    ),
                    **max_stats,
                },
            )
        )
        largest_snake_over_time.append(
            {
                "bin": lbl,
                "tick_start": t0,
                "tick_end": t1,
                "phase": phase,
                **max_stats,
            }
        )

    # Format life-age bins (post-burn-in window)
    life_bounds = [0, *l_edges, None]
    life_age_bins: list[dict[str, Any]] = []
    for lb_i in range(num_lbins):
        lo = life_bounds[lb_i]
        hi = life_bounds[lb_i + 1]
        lbl = f"[{lo}, +inf)" if hi is None else f"[{lo}, {hi})"
        r = lbin_np[lb_i]
        life_age_bins.append(
            _format_bin_Row(
                bin_label=lbl,
                acted_steps=int(r[0]),
                alive_length_sum=int(r[1]),
                net_alive_growth=int(r[2]),
                deaths=int(r[3]),
                kills=int(r[4]),
                boost_ticks=int(r[5]),
                mass_shed_boost=int(r[6]),
                mass_eaten=int(r[7]),
                turn_counts=r[8:13].tolist(),
                extra={
                    "life_age_min": int(lo),
                    "life_age_max": None if hi is None else int(hi),
                },
            )
        )

    # Overall post-burn-in window summary (sum over lbin_np, which partitions all window steps)
    win_tot = lbin_np.sum(axis=0)
    older_tot = int(win_tot[13])
    stuck_tot = int(win_tot[14])
    win_wmax_slice = wmax_np[int(burnin_ticks) : total_ticks].reshape(-1)
    window_summary = _format_bin_Row(
        bin_label=f"[{burnin_ticks}, {total_ticks})",
        acted_steps=int(win_tot[0]),
        alive_length_sum=int(win_tot[1]),
        net_alive_growth=int(win_tot[2]),
        deaths=int(win_tot[3]),
        kills=int(win_tot[4]),
        boost_ticks=int(win_tot[5]),
        mass_shed_boost=int(win_tot[6]),
        mass_eaten=int(win_tot[7]),
        turn_counts=win_tot[8:13].tolist(),
        extra={
            "older_than_500_snake_ticks": older_tot,
            "stuck_snake_ticks": stuck_tot,
            "stuck_share": round(stuck_tot / max(1, older_tot), 6) if older_tot > 0 else 0.0,
            "mean_world_max_length": round(float(win_wmax_slice.mean()), 4),
            "median_world_max_length": round(float(np.median(win_wmax_slice)), 4),
            "p90_world_max_length": round(float(np.percentile(win_wmax_slice, 90)), 4),
            "peak_world_max_length": int(win_wmax_slice.max()),
        },
    )

    # Death causes by victim length bin
    v_bounds = [0, *v_edges, None]
    death_causes_by_length: list[dict[str, Any]] = []
    for vb_i in range(num_vbins):
        lo = v_bounds[vb_i]
        hi = v_bounds[vb_i + 1]
        lbl = f"[{lo}, +inf)" if hi is None else f"[{lo}, {hi})"
        counts_c = {DEATH_CAUSE_NAMES[c_i]: int(vdc_np[vb_i, c_i]) for c_i in range(4)}
        tot_d = int(vdc_np[vb_i].sum())
        shares_c = {
            DEATH_CAUSE_NAMES[c_i]: (round(int(vdc_np[vb_i, c_i]) / tot_d, 6) if tot_d > 0 else 0.0)
            for c_i in range(4)
        }
        death_causes_by_length.append(
            {
                "victim_length_bin": lbl,
                "length_min": int(lo),
                "length_max": None if hi is None else int(hi),
                "total_deaths": tot_d,
                "counts": counts_c,
                "shares": shares_c,
            }
        )
    all_c_np = vdc_np.sum(axis=0)
    all_d_tot = int(all_c_np.sum())
    death_causes_total = {
        "total_deaths": all_d_tot,
        "counts": {DEATH_CAUSE_NAMES[c_i]: int(all_c_np[c_i]) for c_i in range(4)},
        "shares": {
            DEATH_CAUSE_NAMES[c_i]: (
                round(int(all_c_np[c_i]) / all_d_tot, 6) if all_d_tot > 0 else 0.0
            )
            for c_i in range(4)
        },
    }

    total_world_agent_steps = int(e_cnt * n_snakes * total_ticks)
    return {
        "schema": V3_ARENA_EVAL_SCHEMA,
        "namespace": V3_PPO_NAMESPACE,
        "mode": "long-game",
        "code_commit": resolve_code_commit(),
        "seed": int(seed),
        "greedy": bool(greedy),
        "metric_definitions": {
            "growth_per_1k_ticks": (
                "Net length gained on surviving learner ticks ((post_len - pre_len) * 1000 / acted_steps); "
                "equals mass_eaten_per_1k_ticks - mass_shed_boost_per_1k_ticks up to max_length clamping."
            ),
            "stuck_share": (
                f"Share of live learner snake-ticks with life_age > {STUCK_MIN_LIFE_AGE_TICKS} "
                f"whose |length - start_length({start_len})| <= {STUCK_SPAWN_LENGTH_TOLERANCE}."
            ),
            "self_deaths": "Always 0 in Game V3.1 (crossing own body is legal by game design).",
        },
        "eval_config": {
            "mode": "long-game",
            "num_envs": int(num_envs),
            "num_snakes": n_snakes,
            "learner_slots": k_slots,
            "opponent_policy": None if all_learners else opponent_policy,
            "arena_radius": float(game_cfg.arena_radius),
            "start_length": start_len,
            "burnin_ticks": int(burnin_ticks),
            "window_ticks": int(window_ticks),
            "total_ticks": total_ticks,
            "num_world_age_bins": int(num_world_age_bins),
            "life_age_edges": list(l_edges),
            "victim_length_edges": list(v_edges),
            "deterministic": bool(deterministic),
            "device": str(dev),
        },
        "policies": [policy.provenance_dict(policy_index=0)],
        "window_summary": window_summary,
        "stuck_metric": {
            "life_age_threshold_ticks": STUCK_MIN_LIFE_AGE_TICKS,
            "spawn_length": start_len,
            "length_tolerance": STUCK_SPAWN_LENGTH_TOLERANCE,
            "older_than_500_snake_ticks": older_tot,
            "stuck_snake_ticks": stuck_tot,
            "stuck_share": window_summary["stuck_share"],
        },
        "world_age_bins": world_age_bins,
        "life_age_bins": life_age_bins,
        "largest_snake_length_over_time": largest_snake_over_time,
        "death_causes": {
            "overall": death_causes_total,
            "by_victim_length_bin": death_causes_by_length,
        },
        "throughput": {
            "wall_seconds": round(wall_seconds, 4),
            "total_world_ticks": int(e_cnt * total_ticks),
            "total_world_agent_steps": total_world_agent_steps,
            "learner_agent_steps": int(e_cnt * k_slots * total_ticks),
            "world_ticks_per_sec": round((e_cnt * total_ticks) / max(1e-9, wall_seconds), 2),
            "agent_steps_per_sec": round(total_world_agent_steps / max(1e-9, wall_seconds), 2),
        },
    }


def _format_measured_spread_bullet(pairs: Sequence[Mapping[str, Any]]) -> str | None:
    """Summarize actually measured per-world mass-share spread and 95% CI half-width."""
    if not pairs:
        return None
    sds = [float(pw.get("per_world_mass_share_std", 0.0)) for pw in pairs]
    hws = [
        float(
            pw.get(
                "mass_share_ci95_half_width",
                0.5 * (float(pw["mass_share_a_ci95"][1]) - float(pw["mass_share_a_ci95"][0])),
            )
        )
        for pw in pairs
    ]
    med_sd_pct = 100.0 * float(np.median(sds))
    max_sd_pct = 100.0 * float(np.max(sds))
    max_hw_pct = 100.0 * float(np.max(hws))
    max_gap_pp = 2.0 * max_hw_pct
    w_cnt = int(pairs[0].get("num_worlds", 0))
    sep_clause = (
        "separates a 5.00 pp (52.5% vs 47.5%) A-B mass-share gap at the measured spread"
        if max_gap_pp <= 5.0
        else f"separates A-B mass-share gaps >= {max_gap_pp:.2f} pp (does not separate a 5.00 pp gap at this measured spread)"
    )
    return (
        f"- **Measured World-to-World Spread (`{w_cnt}` worlds):** "
        f"per-world mass-share SD median `{med_sd_pct:.2f}%`, max `{max_sd_pct:.2f}%` -> "
        f"worst-case 95% CI half-width `±{max_hw_pct:.2f}%` on mass share "
        f"(`±{max_gap_pp:.2f} pp` on `A - B` gap; {sep_clause})"
    )


def format_eval_markdown(result: Mapping[str, Any]) -> str:
    """Format an ``arena_eval`` result dictionary into human-readable GitHub Markdown tables."""
    mode = str(result.get("mode", ""))
    cfg = result.get("eval_config", {})
    tp = result.get("throughput", {})
    lines: list[str] = [
        f"# Game V3.1 Arena Evaluation (`{mode}`)",
        "",
        f"- **Commit:** `{result.get('code_commit')}` | **Seed:** `{result.get('seed')}` | "
        f"**Action mode:** `{'greedy' if result.get('greedy') else 'sampled'}` | "
        f"**Device:** `{cfg.get('device')}` | **Wall:** `{tp.get('wall_seconds')}s`",
    ]

    if mode == "head-to-head":
        lines.append(
            f"- **Horizon:** `{cfg.get('num_envs')}` worlds × "
            f"(`{cfg.get('burnin_ticks')}` burn-in + `{cfg.get('window_ticks')}` window ticks), "
            f"`{cfg.get('num_snakes')}` seats rotated via `(slot + world) % P`"
        )
        spread_line = _format_measured_spread_bullet(result.get("pairwise", []))
        if spread_line is not None:
            lines.append(spread_line)
        lines.extend(
            [
                "",
                "## Per-Policy Window Summary",
                "",
                "| Policy | Net | Acted Steps | Time-Avg Len | Arena Mass Share | Kills / 1k | Deaths / 1k | Boost % | Eaten / 1k | Shed / 1k |",
                "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for p in result["per_policy"]:
            lines.append(
                f"| `{p['label']}` | {p['net_size']} | {p['acted_steps']:,} | "
                f"{p['time_averaged_length']:.2f} | {100.0 * p['arena_mass_share']:.2f}% | "
                f"{p['kills_per_1k_agent_steps']:.3f} | {p['deaths_per_1k_agent_steps']:.3f} | "
                f"{100.0 * p['boost_share']:.2f}% | {p['mass_eaten_per_1k_agent_steps']:.2f} | "
                f"{p['mass_shed_boost_per_1k_agent_steps']:.2f} |"
            )

        lines.extend(
            [
                "",
                "## Pairwise Comparisons (Bootstrap 95% CI & Measured World Spread)",
                "",
                "| Matchup (A vs B) | Worlds | A Mass Share [95% CI] | World Spread SD | ±CI95 Half-Width | Mass Diff (A-B) [95% CI] | Kills (A on B / B on A) | Kill Balance/World [95% CI] |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for pw in result["pairwise"]:
            ci_s = pw["mass_share_a_ci95"]
            ci_d = pw["mass_share_diff_ci95"]
            ci_k = pw["kill_balance_per_world_ci95"]
            sd_pct = 100.0 * float(pw.get("per_world_mass_share_std", 0.0))
            hw_pct = 100.0 * float(
                pw.get("mass_share_ci95_half_width", 0.5 * (float(ci_s[1]) - float(ci_s[0])))
            )
            lines.append(
                f"| `{pw['pair']}` | {pw['num_worlds']} | "
                f"{100.0 * pw['mass_share_a']:.2f}% [{100.0 * ci_s[0]:.2f}%, {100.0 * ci_s[1]:.2f}%] | "
                f"{sd_pct:.2f}% | ±{hw_pct:.2f}% | "
                f"{100.0 * pw['mass_share_diff_a_minus_b']:+.2f}% "
                f"[{100.0 * ci_d[0]:+.2f}%, {100.0 * ci_d[1]:+.2f}%] | "
                f"{pw['kills_a_on_b']} / {pw['kills_b_on_a']} ({pw['kill_balance_a_minus_b']:+d}) | "
                f"{pw['kill_balance_per_world']:+.2f} [{ci_k[0]:+.2f}, {ci_k[1]:+.2f}] |"
            )

        km = result["kill_matrix"]
        cols = km["columns"]
        lines.extend(
            [
                "",
                "## Kill Matrix (Rows = Victim, Columns = Killer / Cause)",
                "",
                "| Victim \\ Killer | " + " | ".join(f"`{c}`" for c in cols) + " | Total Deaths |",
                "| :--- | " + " | ".join("---:" for _ in cols) + " | ---: |",
            ]
        )
        for v_lbl, r_dict in km["victim_by_killer"].items():
            vals = " | ".join(str(r_dict[c]) for c in cols)
            lines.append(f"| `{v_lbl}` | {vals} | {r_dict['total_deaths']} |")

        p_labels = [p["label"] for p in result["per_policy"]]
        lines.extend(
            [
                "",
                "## Share of Total Arena Mass Over Time",
                "",
                "| Tick Window | Phase | "
                + " | ".join(f"`{l}` Share (Len)" for l in p_labels)
                + " |",
                "| :--- | :---: | " + " | ".join("---:" for _ in p_labels) + " |",
            ]
        )
        for row in result["mass_share_over_time"]:
            cells = " | ".join(
                f"{100.0 * row['mass_shares'][l]:.1f}% ({row['mean_lengths'][l]:.1f})"
                for l in p_labels
            )
            lines.append(
                f"| `[{row['tick_start']}, {row['tick_end']})` | {row['phase']} | {cells} |"
            )

    elif mode == "round-robin":
        exec_mode = (
            f"batched (`<= {cfg.get('pairs_per_batch')}` pairs/batch)"
            if cfg.get("batched_pairs", True)
            else "sequential (`1` pair/sim)"
        )
        lines.append(
            f"- **Tournament:** `{cfg.get('num_policies')}` policies (`{cfg.get('num_pairs')}` pairs, {exec_mode}), "
            f"`{cfg.get('num_envs')}` worlds × (`{cfg.get('burnin_ticks')}` + `{cfg.get('window_ticks')}` ticks)"
        )
        lines.append(f"- **Ranking & Ratings:** {result.get('rating_model')}")
        spread_line = _format_measured_spread_bullet(result.get("pairs", []))
        if spread_line is not None:
            lines.append(spread_line)
        lines.extend(
            [
                "",
                "## Round-Robin Standings (Ranked by Mean Pairwise Mass Share)",
                "",
                "| Rank | Policy | Net | Mean Mass Share [95% CI] | World Wins | BT Elo [95% CI] (Secondary) | Time-Avg Len | Kills / 1k | Deaths / 1k | Net Kills |",
                "| ---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for s in result["standings"]:
            eci = s["elo_ci95"]
            m_sh = float(s["mean_pairwise_mass_share"])
            m_ci = s.get("mean_pairwise_mass_share_ci95", [m_sh, m_sh])
            lines.append(
                f"| {s['rank']} | `{s['label']}` | {s['net_size']} | "
                f"{100.0 * m_sh:.2f}% [{100.0 * float(m_ci[0]):.2f}%, {100.0 * float(m_ci[1]):.2f}%] | "
                f"{s['world_wins']:.1f}/{s['world_games']} ({100.0 * s['world_win_rate']:.1f}%) | "
                f"{s['elo']:.1f} [{eci[0]:.1f}, {eci[1]:.1f}] | "
                f"{s['time_averaged_length']:.2f} | {s['kills_per_1k_agent_steps']:.3f} | "
                f"{s['deaths_per_1k_agent_steps']:.3f} | {s['net_kill_balance']:+d} |"
            )

        if result.get("pairs"):
            lines.extend(
                [
                    "",
                    "## Pairwise Matchups (Bootstrap 95% CI & Measured World Spread)",
                    "",
                    "| Pair | Matchup (A vs B) | Worlds | A Mass Share [95% CI] | World Spread SD | ±CI95 Half-Width | Mass Diff (A-B) [95% CI] | Kills (A on B / B on A) |",
                    "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for pw in result["pairs"]:
                ci_s = pw["mass_share_a_ci95"]
                ci_d = pw["mass_share_diff_ci95"]
                sd_pct = 100.0 * float(pw.get("per_world_mass_share_std", 0.0))
                hw_pct = 100.0 * float(
                    pw.get("mass_share_ci95_half_width", 0.5 * (float(ci_s[1]) - float(ci_s[0])))
                )
                lines.append(
                    f"| `{pw['unit_id']}` | `{pw['pair']}` | {pw['num_worlds']} | "
                    f"{100.0 * pw['mass_share_a']:.2f}% [{100.0 * ci_s[0]:.2f}%, {100.0 * ci_s[1]:.2f}%] | "
                    f"{sd_pct:.2f}% | ±{hw_pct:.2f}% | "
                    f"{100.0 * pw['mass_share_diff_a_minus_b']:+.2f}% "
                    f"[{100.0 * ci_d[0]:+.2f}%, {100.0 * ci_d[1]:+.2f}%] | "
                    f"{pw['kills_a_on_b']} / {pw['kills_b_on_a']} ({pw['kill_balance_a_minus_b']:+d}) |"
                )

    elif mode == "long-game":
        pol0 = result["policies"][0]
        ws = result["window_summary"]
        opp_str = (
            "all 24 self-play seats"
            if cfg.get("opponent_policy") is None
            else f"{cfg.get('learner_slots')} seats vs {cfg.get('opponent_policy')}"
        )
        lines.append(
            f"- **Policy:** `{pol0['label']}` (`{pol0['net_size']}`, `{opp_str}`) | "
            f"**Horizon:** `{cfg.get('num_envs')}` worlds × "
            f"(`{cfg.get('burnin_ticks')}` burn-in + `{cfg.get('window_ticks')}` window ticks)"
        )
        lines.append(
            f"- **Post-Burn-in Headline:** Time-Avg Len `{ws['time_averaged_length']:.2f}`, "
            f"World Max Len `{ws['mean_world_max_length']:.1f}` (peak `{ws['peak_world_max_length']}`), "
            f"Stuck (`age>500, |L-{cfg.get('start_length')}|<=2`) "
            f"`{100.0 * ws['stuck_share']:.2f}%` (`{ws['stuck_snake_ticks']:,}/{ws['older_than_500_snake_ticks']:,}`)"
        )

        lines.extend(
            [
                "",
                "## Metrics Binned by World Age",
                "",
                "| World Age | Phase | Acted Steps | Deaths | Kills | Len | Max Snake (Mean/Peak) | Growth/1k | Deaths/1k | Kills/1k | Boost % | Eaten/Shed /1k | Turns (HL/SL/St/SR/HR %) | Stuck % |",
                "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for b in result["world_age_bins"]:
            ts = b["turn_level_shares"]
            turn_s = (
                f"{100*ts['hard_left']:.1f}/{100*ts['soft_left']:.1f}/"
                f"{100*ts['straight']:.1f}/{100*ts['soft_right']:.1f}/{100*ts['hard_right']:.1f}"
            )
            lines.append(
                f"| `{b['bin']}` | {b['phase']} | {b['acted_steps']:,} | {b['deaths']} | {b['kills']} | "
                f"{b['time_averaged_length']:.2f} | {b['mean_world_max_length']:.1f}/{b['peak_world_max_length']} | "
                f"{b['growth_per_1k_ticks']:+.2f} | {b['deaths_per_1k_ticks']:.3f} | "
                f"{b['kills_per_1k_ticks']:.3f} | {100.0 * b['boost_share']:.2f}% | "
                f"{b['mass_eaten_per_1k_ticks']:.1f}/{b['mass_shed_boost_per_1k_ticks']:.1f} | "
                f"{turn_s} | {100.0 * b['stuck_share']:.2f}% |"
            )

        lines.extend(
            [
                "",
                "## Metrics Binned by Snake Life Age (Post-Burn-in Window)",
                "",
                "| Life Age (ticks) | Acted Steps | Deaths | Kills | Len | Growth/1k | Deaths/1k | Kills/1k | Boost % | Eaten/Shed /1k | Turns (HL/SL/St/SR/HR %) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for b in result["life_age_bins"]:
            ts = b["turn_level_shares"]
            turn_s = (
                f"{100*ts['hard_left']:.1f}/{100*ts['soft_left']:.1f}/"
                f"{100*ts['straight']:.1f}/{100*ts['soft_right']:.1f}/{100*ts['hard_right']:.1f}"
            )
            lines.append(
                f"| `{b['bin']}` | {b['acted_steps']:,} | {b['deaths']} | {b['kills']} | "
                f"{b['time_averaged_length']:.2f} | {b['growth_per_1k_ticks']:+.2f} | "
                f"{b['deaths_per_1k_ticks']:.3f} | {b['kills_per_1k_ticks']:.3f} | "
                f"{100.0 * b['boost_share']:.2f}% | "
                f"{b['mass_eaten_per_1k_ticks']:.1f}/{b['mass_shed_boost_per_1k_ticks']:.1f} | "
                f"{turn_s} |"
            )

        lines.extend(
            [
                "",
                "## Death Causes by Victim Length Bin (Post-Burn-in Window)",
                "",
                "| Victim Length | Total Deaths | Head-into-Body | Wall / Edge | Head-On | Self |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["death_causes"]["by_victim_length_bin"]:
            c = row["counts"]
            s = row["shares"]
            lines.append(
                f"| `{row['victim_length_bin']}` | {row['total_deaths']} | "
                f"{c['head_into_body']} ({100*s['head_into_body']:.1f}%) | "
                f"{c['wall']} ({100*s['wall']:.1f}%) | "
                f"{c['head_on']} ({100*s['head_on']:.1f}%) | "
                f"{c['self']} ({100*s['self']:.1f}%) |"
            )
        ov = result["death_causes"]["overall"]
        oc, os_ = ov["counts"], ov["shares"]
        lines.append(
            f"| **All Lengths** | **{ov['total_deaths']}** | "
            f"{oc['head_into_body']} ({100*os_['head_into_body']:.1f}%) | "
            f"{oc['wall']} ({100*os_['wall']:.1f}%) | "
            f"{oc['head_on']} ({100*os_['head_on']:.1f}%) | "
            f"{oc['self']} ({100*os_['self']:.1f}%) |"
        )

    return "\n".join(lines) + "\n"


def _parse_positive_int_csv(raw: str, flag_name: str) -> tuple[int, ...]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError(f"{flag_name} must be a non-empty comma-separated list of integers")
    vals = tuple(int(p) for p in parts)
    if any(v <= 0 for v in vals) or list(vals) != sorted(set(vals)):
        raise ValueError(f"{flag_name} must contain strictly increasing positive ints, got {vals}")
    return vals


def resolve_staged_ledger_inputs(
    input_dir: Path,
    ledger_path: Path,
) -> tuple[list[Path], list[str] | None, dict[str, Path]]:
    """Verify staged policies and prior completed pair units from a Colab CLI verified ledger."""
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    if ledger.get("namespace") != V3_PPO_NAMESPACE:
        raise ValueError(
            f"Foreign verified ledger namespace {ledger.get('namespace')!r} != {V3_PPO_NAMESPACE!r}"
        )
    policy_paths: list[Path] = []
    policy_labels: list[str] = []
    verified_pairs: dict[str, Path] = {}

    for entry in ledger.get("eval_policies", []):
        rel = str(entry["relpath"])
        p_file = Path(input_dir) / rel
        if not p_file.is_file():
            raise FileNotFoundError(f"Staged eval policy missing at {p_file}")
        actual_sha = sha256_file(p_file)
        if actual_sha != str(entry["sha256"]):
            raise ValueError(
                f"Staged eval policy sha256 mismatch for {rel}: {actual_sha} != {entry['sha256']}"
            )
        policy_paths.append(p_file)
        if entry.get("label"):
            policy_labels.append(str(entry["label"]))

    for unit in ledger.get("verified_units", []):
        uid = str(unit.get("unit_id", ""))
        if uid.startswith("pair:"):
            pair_name = uid[len("pair:") :]
            u_file = Path(input_dir) / str(unit["relpath"])
            if not u_file.is_file():
                raise FileNotFoundError(f"Verified pair unit missing at {u_file}")
            actual_sha = sha256_file(u_file)
            if actual_sha != str(unit["sha256"]):
                raise ValueError(f"Verified pair {uid} sha256 mismatch")
            verified_pairs[pair_name] = u_file

    resolved_labels = (
        policy_labels if len(policy_labels) == len(policy_paths) and policy_labels else None
    )
    return policy_paths, resolved_labels, verified_pairs


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for ``python -m research.v3_ppo.arena_eval``."""
    parser = argparse.ArgumentParser(
        prog="python -m research.v3_ppo.arena_eval",
        description=(
            "Standalone Game V3.1 policy evaluation: 2..6 policy head-to-head, "
            "all-pairs round-robin with Bradley-Terry Elo, and mature long-game diagnostics."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=EVAL_MODES,
        default=None,
        help="Evaluation mode (auto-inferred from --policies / --round-robin / --long-game if omitted).",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        type=Path,
        default=None,
        help="2 to 6 LIGHT checkpoints for head-to-head (or >=2 for round-robin, 1 for long-game).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Single LIGHT checkpoint for --long-game.",
    )
    parser.add_argument(
        "--round-robin",
        nargs="+",
        type=Path,
        default=None,
        metavar="DIR_OR_LIST",
        help="Directory, .txt/.json checkpoint list, or checkpoint paths for all-pairs round-robin.",
    )
    parser.add_argument(
        "--long-game",
        action="store_true",
        help="Run single-policy mature long-game evaluation.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=None,
        help="Optional unique display labels matching the supplied policies.",
    )
    parser.add_argument(
        "--latest-per-run",
        action="store_true",
        help="When --round-robin scans a checkpoint folder, pick only the latest light_u*.pt.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help=(
            f"Number of parallel worlds (default: {DEFAULT_H2H_NUM_ENVS} for head-to-head/round-robin, "
            f"{DEFAULT_LONG_GAME_NUM_ENVS} for long-game)."
        ),
    )
    parser.add_argument(
        "--burnin-ticks",
        type=int,
        default=None,
        help=(
            f"World maturation burn-in ticks before measurement window "
            f"(default: {DEFAULT_H2H_BURNIN_TICKS} for head-to-head/round-robin, "
            f"{DEFAULT_LONG_GAME_BURNIN_TICKS} for long-game)."
        ),
    )
    parser.add_argument(
        "--window-ticks",
        type=int,
        default=None,
        help=(
            f"Post-burn-in measurement window ticks "
            f"(default: {DEFAULT_H2H_WINDOW_TICKS} for head-to-head/round-robin, "
            f"{DEFAULT_LONG_GAME_WINDOW_TICKS} for long-game)."
        ),
    )
    parser.add_argument(
        "--mass-bin-ticks",
        type=int,
        default=DEFAULT_H2H_MASS_BIN_TICKS,
        help=f"Tick bin width for arena mass share over time in head-to-head (default: {DEFAULT_H2H_MASS_BIN_TICKS}).",
    )
    parser.add_argument(
        "--learner-slots",
        type=int,
        default=None,
        help="Number of policy seats (1..24) in long-game; defaults to all 24 seats.",
    )
    parser.add_argument(
        "--opponent-policy",
        choices=VALID_OPPONENT_POLICIES,
        default="food_seeking",
        help="Scripted opponent when --learner-slots < 24 in long-game (default: food_seeking).",
    )
    parser.add_argument(
        "--num-world-age-bins",
        type=int,
        default=DEFAULT_LONG_GAME_WORLD_AGE_BINS,
        help=f"Number of post-burn-in world-age bins in long-game, >= 6 (default: {DEFAULT_LONG_GAME_WORLD_AGE_BINS}).",
    )
    parser.add_argument(
        "--life-age-edges",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LIFE_AGE_EDGES),
        help="Comma-separated interior life-age bin edges in ticks (default: 100,250,500,1000,2000,4000).",
    )
    parser.add_argument(
        "--victim-length-edges",
        type=str,
        default=",".join(str(x) for x in DEFAULT_VICTIM_LENGTH_EDGES),
        help="Comma-separated interior victim-length bin edges (default: 15,30,60,120,250).",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic argmax actions instead of default categorical policy sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=f"Deterministic evaluation seed (default: {DEFAULT_EVAL_SEED}).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"Bootstrap resamples over worlds for 95%% CIs (default: {DEFAULT_BOOTSTRAP_SAMPLES}).",
    )
    parser.add_argument(
        "--bt-prior",
        type=float,
        default=DEFAULT_BT_PRIOR_WINS,
        help=f"Symmetric pseudo-win prior per pair for Bradley-Terry Elo (default: {DEFAULT_BT_PRIOR_WINS}).",
    )
    rr_batch_group = parser.add_mutually_exclusive_group()
    rr_batch_group.add_argument(
        "--batched-pairs",
        dest="batched_pairs",
        action="store_true",
        default=True,
        help="Run round-robin pairs side-by-side in one batched simulator (default).",
    )
    rr_batch_group.add_argument(
        "--sequential-pairs",
        dest="batched_pairs",
        action="store_false",
        help="Fallback: run round-robin pairs sequentially one pair at a time.",
    )
    parser.add_argument(
        "--max-pairs-per-batch",
        type=int,
        default=None,
        help="Optional upper bound on pairs per batched round-robin simulator run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device: cpu (default), cuda, cuda:0, mps, or auto.",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")

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

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile on loaded policy networks.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Colab CLI staged input directory (used with --ledger).",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Colab CLI verified-ledger.json binding staged checkpoints and completed pair units.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Cooperative wall-clock runaway guard in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Create-only directory for eval_result.json, summary.md (and pairs/*.json).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional create-only single JSON output file path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m research.v3_ppo.arena_eval``."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    verified_pairs: dict[str, Path] = {}
    ledger_paths: list[Path] = []
    ledger_labels: list[str] | None = None
    if args.ledger is not None:
        if args.input_dir is None:
            parser.error("--ledger requires --input-dir")
        ledger_paths, ledger_labels, verified_pairs = resolve_staged_ledger_inputs(
            args.input_dir, args.ledger
        )

    # Infer mode and checkpoint paths cleanly
    mode = args.mode
    if mode is None:
        if args.round_robin is not None:
            mode = "round-robin"
        elif args.long_game or args.policy is not None:
            mode = "long-game"
        elif args.policies is not None:
            mode = "long-game" if len(args.policies) == 1 else "head-to-head"
        elif ledger_paths:
            if len(ledger_paths) == 1:
                mode = "long-game"
            elif len(ledger_paths) > 6 or bool(verified_pairs):
                mode = "round-robin"
            else:
                mode = "head-to-head"
        else:
            parser.error(
                "Specify --policies A.pt B.pt [...], --round-robin DIR_OR_LIST, or --long-game --policy A.pt"
            )

    if mode == "round-robin":
        if args.round_robin is not None:
            ckpt_paths = discover_round_robin_checkpoints(
                args.round_robin, latest_per_run=bool(args.latest_per_run)
            )
        elif args.policies is not None:
            ckpt_paths = discover_round_robin_checkpoints(
                args.policies, latest_per_run=bool(args.latest_per_run)
            )
        elif ledger_paths:
            ckpt_paths = ledger_paths
        else:
            parser.error("--mode round-robin requires --round-robin or --policies")
    elif mode == "long-game":
        if args.policy is not None and args.policies is not None:
            parser.error("Pass either --policy or --policies (not both) for long-game")
        if args.policy is not None:
            ckpt_paths = [Path(args.policy).resolve()]
        elif args.policies is not None:
            if len(args.policies) != 1:
                parser.error(f"long-game expects exactly 1 policy, got {len(args.policies)}")
            ckpt_paths = [Path(args.policies[0]).resolve()]
        elif len(ledger_paths) == 1:
            ckpt_paths = ledger_paths
        else:
            parser.error("long-game requires --policy PATH.pt")
    else:
        raw_pols = args.policies if args.policies is not None else ledger_paths
        if not raw_pols or not (2 <= len(raw_pols) <= 6):
            parser.error(
                f"head-to-head requires 2 to 6 checkpoints via --policies, got {0 if not raw_pols else len(raw_pols)}"
            )
        ckpt_paths = [Path(p).resolve() for p in raw_pols]

    eff_labels = resolve_unique_labels(
        ckpt_paths, args.labels if args.labels is not None else ledger_labels
    )
    dev = normalize_device(args.device)
    ensure_cuda_initialized(dev)

    loaded_policies = [
        load_eval_policy(
            p,
            device=dev,
            label=lbl,
            amp=args.amp,
            use_channels_last=args.use_channels_last,
            fast_learner_ops=args.fast_learner_ops,
            compile_model=bool(args.compile),
        )
        for p, lbl in zip(ckpt_paths, eff_labels)
    ]

    main_started = time.perf_counter()
    with SignalStopFlag() as stop_flag:

        def _coop_stop() -> bool:
            if stop_flag.stopped:
                return True
            if args.max_wall_seconds is not None:
                return (time.perf_counter() - main_started) >= float(args.max_wall_seconds)
            return False

        if mode == "head-to-head":
            result = evaluate_head_to_head(
                loaded_policies,
                num_envs=DEFAULT_H2H_NUM_ENVS if args.num_envs is None else int(args.num_envs),
                burnin_ticks=(
                    DEFAULT_H2H_BURNIN_TICKS
                    if args.burnin_ticks is None
                    else int(args.burnin_ticks)
                ),
                window_ticks=(
                    DEFAULT_H2H_WINDOW_TICKS
                    if args.window_ticks is None
                    else int(args.window_ticks)
                ),
                mass_bin_ticks=int(args.mass_bin_ticks),
                greedy=bool(args.greedy),
                seed=int(args.seed),
                bootstrap_samples=int(args.bootstrap_samples),
                should_stop=_coop_stop,
            )
        elif mode == "round-robin":
            result = evaluate_round_robin(
                loaded_policies,
                num_envs=DEFAULT_H2H_NUM_ENVS if args.num_envs is None else int(args.num_envs),
                burnin_ticks=(
                    DEFAULT_H2H_BURNIN_TICKS
                    if args.burnin_ticks is None
                    else int(args.burnin_ticks)
                ),
                window_ticks=(
                    DEFAULT_H2H_WINDOW_TICKS
                    if args.window_ticks is None
                    else int(args.window_ticks)
                ),
                mass_bin_ticks=int(args.mass_bin_ticks),
                greedy=bool(args.greedy),
                seed=int(args.seed),
                bootstrap_samples=int(args.bootstrap_samples),
                bt_prior_wins=float(args.bt_prior),
                batched=bool(args.batched_pairs),
                max_pairs_per_batch=args.max_pairs_per_batch,
                output_dir=args.output_dir,
                verified_pair_paths=verified_pairs,
                max_wall_seconds=args.max_wall_seconds,
                should_stop=lambda: stop_flag.stopped,
            )
        else:
            l_edges = _parse_positive_int_csv(args.life_age_edges, "--life-age-edges")
            v_edges = _parse_positive_int_csv(args.victim_length_edges, "--victim-length-edges")
            result = evaluate_long_game(
                loaded_policies[0],
                num_envs=(
                    DEFAULT_LONG_GAME_NUM_ENVS if args.num_envs is None else int(args.num_envs)
                ),
                burnin_ticks=(
                    DEFAULT_LONG_GAME_BURNIN_TICKS
                    if args.burnin_ticks is None
                    else int(args.burnin_ticks)
                ),
                window_ticks=(
                    DEFAULT_LONG_GAME_WINDOW_TICKS
                    if args.window_ticks is None
                    else int(args.window_ticks)
                ),
                learner_slots=args.learner_slots,
                opponent_policy=args.opponent_policy,
                num_world_age_bins=int(args.num_world_age_bins),
                life_age_edges=l_edges,
                victim_length_edges=v_edges,
                greedy=bool(args.greedy),
                seed=int(args.seed),
                should_stop=_coop_stop,
            )

    md_text = format_eval_markdown(result)
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_create_only_json(out_dir / "eval_result.json", result)
        write_create_only_artifact(out_dir / "summary.md", md_text.encode("utf-8"))
    if args.json_out is not None:
        write_create_only_json(Path(args.json_out), result)

    print(md_text, end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
