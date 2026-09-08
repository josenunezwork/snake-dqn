#!/usr/bin/env python3
"""
Ape-X DQN Distributed Training Coordinator.

Wires together the modular components for distributed Ape-X DQN training:
- BufferProcess: Separate process hosting SumTree-backed prioritized replay
- ApexActor(s): Multiple processes generating experiences with diverse epsilon
- ApexLearner: GPU-accelerated centralized learner

Architecture:
                    ┌─────────────┐
                    │ BufferProcess│ (separate process, SumTree)
                    └──────┬──────┘
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │ Actor 0 │      │ Actor 1 │ ...  │Actor N-1│  (mp.Process each)
    │ ε=high  │      │ ε=med   │      │ ε=low   │
    └─────────┘      └─────────┘      └─────────┘
                    ┌──────▼──────┐
                    │   Learner   │  (main process, GPU)
                    └─────────────┘

Usage:
    # Small local test (Mac)
    python src/scripts/apex_train.py --num-actors 4 --total-steps 100000

    # Full distributed (H100 server)
    python src/scripts/apex_train.py --num-actors 64 --total-steps 10000000

    # Resume from checkpoint
    python src/scripts/apex_train.py --resume saved_snakes/apex_checkpoint.pth

    # With YAML config
    python src/scripts/apex_train.py --config configs/production.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import signal
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, TypeVar

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import load_and_initialize_config  # noqa: E402
from src.core.game_config import GameConfig  # noqa: E402
from src.core.reward_contract import current_reward_contract  # noqa: E402
from src.training.apex_runtime import ApexRunBudgets  # noqa: E402
from src.training.apex_runtime import (  # noqa: E402
    ApexControlledStop,
    ApexRuntimeSnapshot,
    ApexRuntimeSupervisor,
    SharedActorProgress,
    SharedEnvironmentFrameBudget,
    stop_processes,
)
from src.training.checkpoint_contract import validate_checkpoint_contract  # noqa: E402

if TYPE_CHECKING:
    import torch
    import torch.multiprocessing as mp

T = TypeVar("T")


def _resolve_configurable(
    value: Optional[T],
    configured_value: T,
    fallback: T,
    use_config: bool,
) -> T:
    """Use explicit CLI value, then config value, then legacy local default."""
    if value is not None:
        return value
    if use_config:
        return configured_value
    return fallback


def _mean_or_zero(values: Sequence[float]) -> float:
    """Return the arithmetic mean for logs without requiring NumPy."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def resolve_actor_replay_quality_fraction(
    value: Optional[float],
    name: str,
) -> float:
    """Resolve an optional actor replay-quality gate fraction in [0, 1]."""
    if value is None:
        return 0.0
    fraction = float(value)
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return fraction


def build_actor_replay_quality_gates(min_terminal_fraction: Optional[float] = 0.0) -> dict:
    """Return configured actor replay-quality gates for checkpoints."""
    return {
        "min_actor_terminal_fraction": resolve_actor_replay_quality_fraction(
            min_terminal_fraction,
            "min_actor_terminal_fraction",
        )
    }


def resolve_apex_min_buffer_size(
    batch_size: int,
    buffer_capacity: int,
    configured_min_buffer_size: int,
) -> int:
    """Resolve a warmup size that can actually support learner sampling."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if buffer_capacity <= 0:
        raise ValueError("buffer_capacity must be positive")
    if configured_min_buffer_size <= 0:
        raise ValueError("configured_min_buffer_size must be positive")
    if buffer_capacity < batch_size:
        raise ValueError("buffer_capacity must be at least batch_size")

    capped_target = min(configured_min_buffer_size, max(batch_size, buffer_capacity // 2))
    return max(batch_size, min(capped_target, buffer_capacity))


def resolve_apex_run_budgets(
    total_steps: int,
    max_learner_updates: Optional[int] = None,
    max_environment_transitions: Optional[int] = None,
    max_wall_time_seconds: Optional[float] = None,
) -> ApexRunBudgets:
    """Resolve explicit Ape-X work budgets.

    ``--total-steps`` remains the compatibility spelling for the learner-update
    ceiling.  Supplying both spellings with different values is ambiguous and
    therefore rejected before child processes are created.
    """
    if max_learner_updates is not None and int(max_learner_updates) != int(total_steps):
        raise ValueError("total_steps and max_learner_updates must match when both are set")
    return ApexRunBudgets(
        max_learner_updates=int(
            total_steps if max_learner_updates is None else max_learner_updates
        ),
        max_environment_transitions=(
            None if max_environment_transitions is None else int(max_environment_transitions)
        ),
        max_wall_time_seconds=(
            None if max_wall_time_seconds is None else float(max_wall_time_seconds)
        ),
    )


def build_apex_runtime_snapshot(
    *,
    learner_updates: int,
    actor_stats: Sequence[dict],
    buffer_replay_health: dict,
    elapsed_seconds: float,
    actor_heartbeat_ages: dict[int, float | None],
    shared_actor_progress: Sequence[SharedActorProgress] = (),
    environment_frame_budget: Optional[SharedEnvironmentFrameBudget] = None,
) -> ApexRuntimeSnapshot:
    """Build one reconciled runtime receipt from coordinator-owned counters."""
    shared_snapshots = [progress.snapshot() for progress in shared_actor_progress]
    return ApexRuntimeSnapshot(
        learner_updates=int(learner_updates),
        # This legacy-named field is explicitly world-frame count, not the
        # number of individual snake action transitions below.
        environment_transitions=(
            sum(int(snapshot["environment_frames"]) for snapshot in shared_snapshots)
            if shared_snapshots
            else sum(
                int(stats.get("environment_transitions", stats.get("total_steps", 0)))
                for stats in actor_stats
            )
        ),
        agent_transitions=(
            sum(int(snapshot["agent_transitions"]) for snapshot in shared_snapshots)
            if shared_snapshots
            else sum(int(stats.get("agent_transitions", 0)) for stats in actor_stats)
        ),
        replay_rows_emitted=(
            sum(int(snapshot["replay_rows_emitted"]) for snapshot in shared_snapshots)
            if shared_snapshots
            else sum(
                int(stats.get("replay_rows_emitted", stats.get("sent_experience_count", 0)))
                for stats in actor_stats
            )
        ),
        learner_samples=int(buffer_replay_health.get("total_sampled", 0)),
        policy_version=(
            max((int(snapshot["policy_version"]) for snapshot in shared_snapshots), default=0)
            if shared_snapshots
            else max((int(stats.get("policy_version", 0)) for stats in actor_stats), default=0)
        ),
        elapsed_seconds=float(elapsed_seconds),
        actor_heartbeat_ages=dict(actor_heartbeat_ages),
        reserved_environment_frames=(
            0 if environment_frame_budget is None else environment_frame_budget.reserved_frames()
        ),
    )


def attach_runtime_metadata(
    state: dict,
    snapshot: ApexRuntimeSnapshot,
    exit_cause: str,
    budgets: ApexRunBudgets | None = None,
    actor_policy_versions: Optional[dict[int, int]] = None,
) -> dict:
    """Attach bounded-work telemetry with an explicit terminal cause."""
    versions = actor_policy_versions or {}
    ages = {
        str(actor_id): max(0, snapshot.learner_updates - int(version))
        for actor_id, version in versions.items()
    }
    sorted_ages = sorted(ages.values())
    state["apex_runtime"] = {
        "exit_cause": exit_cause,
        "learner_updates": snapshot.learner_updates,
        "environment_transitions": snapshot.environment_transitions,
        "environment_frames": snapshot.environment_transitions,
        "agent_transitions": snapshot.agent_transitions,
        "replay_rows_emitted": snapshot.replay_rows_emitted,
        "learner_samples": snapshot.learner_samples,
        "policy_version": snapshot.policy_version,
        "actor_policy_version_age": ages,
        "actor_policy_version_age_max": max(sorted_ages, default=0),
        "actor_policy_version_age_p95": (
            sorted_ages[math.ceil(0.95 * len(sorted_ages)) - 1] if sorted_ages else 0
        ),
        "elapsed_seconds": snapshot.elapsed_seconds,
        "actor_heartbeat_ages": dict(snapshot.actor_heartbeat_ages),
        "reserved_environment_frames": snapshot.reserved_environment_frames,
        "resolved_budgets": None if budgets is None else asdict(budgets),
        "wall_time_enforcement": "cooperative_between_steps_including_startup",
        "wall_time_overshoot_seconds": (
            None
            if budgets is None or budgets.max_wall_time_seconds is None
            else max(0.0, snapshot.elapsed_seconds - budgets.max_wall_time_seconds)
        ),
    }
    return state


def validate_apex_training_config(
    *,
    num_actors: int,
    total_steps: int,
    batch_size: int,
    buffer_capacity: int,
    n_step: int,
    min_buffer_size: int,
    weight_broadcast_interval: int,
    checkpoint_interval: int,
    log_interval: int,
    stagger_delay: float,
    actor_env_num_snakes: int,
    actor_board_scale: float,
    actor_food_multiplier: float,
    actor_boost_exploration_rate: float,
    actor_danger_exploration_rate: float,
) -> None:
    """Reject distributed training configs that cannot produce learner updates."""
    positive_values = {
        "num_actors": num_actors,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "buffer_capacity": buffer_capacity,
        "n_step": n_step,
        "min_buffer_size": min_buffer_size,
        "weight_broadcast_interval": weight_broadcast_interval,
        "checkpoint_interval": checkpoint_interval,
        "log_interval": log_interval,
        "actor_env_num_snakes": actor_env_num_snakes,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    for name, value in {
        "actor_board_scale": actor_board_scale,
        "actor_food_multiplier": actor_food_multiplier,
    }.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    actor_width = int(GameConfig.WIDTH * actor_board_scale)
    actor_height = int(GameConfig.HEIGHT * actor_board_scale)
    min_actor_dimension = GameConfig.WALL_THICKNESS * 2 + GameConfig.SEGMENT_SIZE
    if actor_width < min_actor_dimension or actor_height < min_actor_dimension:
        raise ValueError(
            f"actor_board_scale produces an unusable actor arena ({actor_width}x{actor_height})"
        )

    if int(GameConfig.INITIAL_FOOD * actor_food_multiplier) <= 0:
        raise ValueError("actor_food_multiplier produces zero initial actor food")
    if int(GameConfig.MAX_FOOD * actor_food_multiplier) <= 0:
        raise ValueError("actor_food_multiplier produces zero max actor food")

    if stagger_delay < 0:
        raise ValueError("stagger_delay must be non-negative")
    for name, value in {
        "actor_boost_exploration_rate": actor_boost_exploration_rate,
        "actor_danger_exploration_rate": actor_danger_exploration_rate,
    }.items():
        if not math.isfinite(float(value)) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if buffer_capacity < batch_size:
        raise ValueError("buffer_capacity must be at least batch_size")
    if min_buffer_size < batch_size:
        raise ValueError("min_buffer_size must be at least batch_size")
    if min_buffer_size > buffer_capacity:
        raise ValueError("min_buffer_size must not exceed buffer_capacity")


def build_apex_checkpoint_config(
    *,
    num_actors: int,
    total_steps: int,
    batch_size: int,
    buffer_capacity: int,
    n_step: int,
    min_buffer_size: int,
    learning_rate: float,
    gamma: float,
    target_update_freq: int,
    weight_broadcast_interval: int,
    priority_alpha: float,
    priority_beta_start: float,
    priority_beta_end: float,
    priority_beta_frames: int,
    priority_epsilon: float,
    grad_clip_norm: float,
    log_interval: int,
    checkpoint_interval: int,
    actor_env_num_snakes: int,
    actor_board_scale: float,
    actor_food_multiplier: float,
    actor_boost_exploration_rate: float,
    actor_danger_exploration_rate: float,
    input_size: int,
    hidden_size: int,
    output_size: int,
    reward_death: float,
    reward_food_base: float,
) -> Dict[str, Any]:
    """Return the resolved distributed Apex training contract for checkpoints."""
    return {
        "actor_env_num_snakes": int(actor_env_num_snakes),
        "actor_board_scale": float(actor_board_scale),
        "actor_food_multiplier": float(actor_food_multiplier),
        "actor_boost_exploration_rate": float(actor_boost_exploration_rate),
        "actor_danger_exploration_rate": float(actor_danger_exploration_rate),
        "batch_size": int(batch_size),
        "buffer_size": int(buffer_capacity),
        "checkpoint_interval": int(checkpoint_interval),
        "gamma": float(gamma),
        "grad_clip_norm": float(grad_clip_norm),
        "hidden_size": int(hidden_size),
        "input_size": int(input_size),
        "learning_rate": float(learning_rate),
        "log_interval": int(log_interval),
        "min_replay_size": int(min_buffer_size),
        "n_step": int(n_step),
        "num_actors": int(num_actors),
        "output_size": int(output_size),
        "priority_alpha": float(priority_alpha),
        "priority_beta_end": float(priority_beta_end),
        "priority_beta_frames": int(priority_beta_frames),
        "priority_beta_start": float(priority_beta_start),
        "priority_epsilon": float(priority_epsilon),
        "reward_contract": current_reward_contract(),
        "reward_death": float(reward_death),
        "reward_food_base": float(reward_food_base),
        "target_update_freq": int(target_update_freq),
        "total_steps": int(total_steps),
        "weight_broadcast_interval": int(weight_broadcast_interval),
    }


def format_apex_checkpoint_provenance(checkpoint: dict) -> list[str]:
    """Return compact Apex checkpoint provenance lines for resume logs."""
    apex_config = checkpoint.get("apex_config") or checkpoint.get("config") or {}
    if not apex_config:
        return []

    batch_size = apex_config.get("batch_size")
    buffer_size = apex_config.get("buffer_size")
    min_replay_size = apex_config.get("min_replay_size", apex_config.get("min_buffer_size"))
    n_step = apex_config.get("n_step")
    gamma = apex_config.get("gamma")
    target_update_freq = apex_config.get("target_update_freq")
    priority_alpha = apex_config.get("priority_alpha")
    priority_beta_start = apex_config.get("priority_beta_start")
    priority_beta_end = apex_config.get("priority_beta_end")
    priority_epsilon = apex_config.get("priority_epsilon", apex_config.get("priority_eps"))
    num_actors = apex_config.get("num_actors")
    actor_env_num_snakes = apex_config.get("actor_env_num_snakes")
    actor_board_scale = apex_config.get("actor_board_scale")
    actor_food_multiplier = apex_config.get("actor_food_multiplier")
    actor_boost_exploration_rate = apex_config.get("actor_boost_exploration_rate")
    actor_danger_exploration_rate = apex_config.get("actor_danger_exploration_rate")
    reward_death = apex_config.get("reward_death")
    reward_food_base = apex_config.get("reward_food_base")

    lines = [
        "Checkpoint Apex config: "
        f"actors={num_actors} | actor_snakes={actor_env_num_snakes} | "
        f"actor_board={actor_board_scale} | actor_food={actor_food_multiplier} | "
        f"actor_boost={actor_boost_exploration_rate} | "
        f"actor_danger={actor_danger_exploration_rate} | "
        f"batch={batch_size} | buffer={buffer_size} | warmup={min_replay_size} | "
        f"n_step={n_step} | gamma={gamma} | target_sync={target_update_freq} | "
        f"reward death={reward_death}, food={reward_food_base} | "
        f"PER alpha={priority_alpha}, beta={priority_beta_start}->{priority_beta_end}, "
        f"eps={priority_epsilon}"
    ]
    return lines


def validate_apex_resume_checkpoint_config(
    checkpoint: dict,
    expected_config: dict,
    checkpoint_path: str = "checkpoint",
    *,
    override_reward_contract: bool = False,
) -> None:
    """Reject resume checkpoints with known-incompatible Apex training contracts.

    Args:
        override_reward_contract: When True, reward-economics mismatches
            (reward_contract and reward_* fields) log a loud warning instead of
            aborting — for a deliberate reward-migration fine-tune such as the P1
            3-arm reward-v2 experiment on champion_a5. Non-reward contract
            violations (shapes, gamma, n_step, board scale) still abort.
    """
    validate_checkpoint_contract(
        checkpoint,
        expected_config,
        checkpoint_path=checkpoint_path,
        integer_keys=(
            "input_size",
            "hidden_size",
            "output_size",
            "n_step",
            "actor_env_num_snakes",
        ),
        float_keys=(
            "gamma",
            "actor_board_scale",
            "actor_food_multiplier",
            "reward_death",
            "reward_food_base",
        ),
        mapping_keys=("reward_contract",),
        required_keys=("reward_contract", "reward_death", "reward_food_base"),
        error_type=ValueError,
        override_reward_contract=override_reward_contract,
    )


def load_validated_apex_resume_checkpoint(
    resume_checkpoint: Optional[str],
    expected_config: dict,
    map_location: Any = None,
    *,
    override_reward_contract: bool = False,
    resume_mode: str = "weights-only",
) -> Optional[dict]:
    """Load a requested resume checkpoint or fail before runtime processes start."""
    if not resume_checkpoint:
        return None

    import torch

    checkpoint_path = Path(resume_checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint payload must be a dict, got {type(checkpoint).__name__}")
        if resume_mode not in {"weights-only", "continuation"}:
            raise ValueError(f"Unsupported Apex resume mode {resume_mode!r}")
        if resume_mode == "continuation":
            validate_apex_resume_checkpoint_config(
                checkpoint,
                expected_config,
                checkpoint_path=str(checkpoint_path),
                override_reward_contract=override_reward_contract,
            )
        required = (
            ("dqn_state_dict",)
            if resume_mode == "weights-only"
            else ("dqn_state_dict", "target_dqn_state_dict", "optimizer_state_dict")
        )
        for key in required:
            if key not in checkpoint:
                raise KeyError(key)
    except (OSError, RuntimeError, EOFError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to load resume checkpoint {checkpoint_path}: {e}") from e

    return checkpoint


def broadcast_weights(
    payload: tuple[int, Dict[str, torch.Tensor]],
    weight_queues: list,
) -> None:
    """Broadcast learner weights to all actor weight queues.

    Clears stale weights before pushing new ones to ensure actors
    always receive the most recent parameters.

    Args:
        payload: Learner update version and CPU state dict.
        weight_queues: List of mp.Queue, one per actor
    """
    for q in weight_queues:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            continue

        try:
            q.put_nowait(payload)
        except queue.Full:
            pass
        except Exception:
            pass


def learner_weight_payload(learner: Any) -> tuple[int, Dict[str, torch.Tensor]]:
    """Return the versioned actor payload, retaining test-double compatibility."""
    get_payload = getattr(learner, "get_weight_payload", None)
    if get_payload is not None:
        return get_payload()
    # Legacy in-process fakes predate versioned transport. Runtime learners
    # always expose get_weight_payload; this is intentionally not a wire API.
    get_weights = getattr(learner, "get_weights", None)
    if get_weights is None:
        return int(getattr(learner, "update_version", 0)), {}
    return int(getattr(learner, "update_version", 0)), get_weights()


def collect_actor_stats(stats_queue: mp.Queue, max_items: int = 100) -> list:
    """Drain actor statistics from the stats queue.

    Args:
        stats_queue: Shared queue where actors push stats dicts
        max_items: Maximum items to drain per call

    Returns:
        List of stats dictionaries
    """
    stats = []
    for _ in range(max_items):
        try:
            stats.append(stats_queue.get_nowait())
        except Exception:
            break
    return stats


def update_latest_actor_stats(
    actor_stats_by_id: dict,
    stats_batch: list,
    episode_rewards: Optional[deque] = None,
) -> None:
    """Keep the latest cumulative stats per actor and optional reward history."""
    for stats in stats_batch:
        if episode_rewards is not None:
            episode_rewards.append(stats.get("avg_reward", 0))
        actor_id = stats.get("actor_id")
        if actor_id is None:
            continue
        actor_stats_by_id[int(actor_id)] = stats


def summarize_actor_replay_coverage(actor_stats: Sequence[dict]) -> dict:
    """Aggregate latest actor replay coverage counters for logging/checkpoints."""
    total_sent = sum(int(stats.get("sent_experience_count", 0)) for stats in actor_stats)
    action_counts = [0 for _ in range(GameConfig.OUTPUT_SIZE)]
    weighted_boost = 0.0
    weighted_exact_masks = 0.0
    weighted_terminals = 0.0
    terminal_count = 0
    total_nonterminal = 0
    weighted_nonterminal_exact_masks = 0.0
    weighted_nonterminal_trapped_next = 0.0
    weighted_positive_rewards = 0.0
    weighted_zero_rewards = 0.0
    weighted_negative_rewards = 0.0
    weighted_multistep = 0.0
    invalid_current_action_count = 0
    invalid_current_normal_action_count = 0
    invalid_current_boost_action_count = 0
    buffer_queued_message_count = 0
    buffer_dropped_message_count = 0
    buffer_dropped_experience_count = 0
    buffer_last_drop_error = None

    for stats in actor_stats:
        sent_count = int(stats.get("sent_experience_count", 0))
        for action, count in enumerate(stats.get("sent_action_counts", [])):
            if 0 <= action < len(action_counts):
                action_counts[action] += int(count)
        weighted_boost += float(stats.get("sent_boost_action_fraction", 0.0)) * sent_count
        weighted_exact_masks += float(stats.get("sent_exact_mask_fraction", 0.0)) * sent_count
        weighted_terminals += float(stats.get("sent_terminal_fraction", 0.0)) * sent_count
        terminal_count += int(
            stats.get(
                "sent_terminal_count",
                round(float(stats.get("sent_terminal_fraction", 0.0)) * sent_count),
            )
        )
        nonterminal_count = int(
            stats.get(
                "sent_nonterminal_count",
                max(
                    sent_count
                    - int(round(float(stats.get("sent_terminal_fraction", 0.0)) * sent_count)),
                    0,
                ),
            )
        )
        total_nonterminal += nonterminal_count
        weighted_nonterminal_exact_masks += (
            float(stats.get("sent_nonterminal_exact_mask_fraction", 0.0)) * nonterminal_count
        )
        weighted_nonterminal_trapped_next += (
            float(stats.get("sent_nonterminal_trapped_next_fraction", 0.0)) * nonterminal_count
        )
        weighted_positive_rewards += (
            float(stats.get("sent_positive_reward_fraction", 0.0)) * sent_count
        )
        weighted_zero_rewards += float(stats.get("sent_zero_reward_fraction", 0.0)) * sent_count
        weighted_negative_rewards += (
            float(stats.get("sent_negative_reward_fraction", 0.0)) * sent_count
        )
        weighted_multistep += float(stats.get("sent_multistep_fraction", 0.0)) * sent_count
        invalid_current_action_count += int(
            stats.get(
                "sent_invalid_current_action_count",
                round(float(stats.get("sent_invalid_current_action_fraction", 0.0)) * sent_count),
            )
        )
        invalid_current_normal_action_count += int(
            stats.get("sent_invalid_current_normal_action_count", 0)
        )
        invalid_current_boost_action_count += int(
            stats.get("sent_invalid_current_boost_action_count", 0)
        )
        buffer_queued_message_count += int(stats.get("buffer_queued_message_count", 0))
        buffer_dropped_message_count += int(stats.get("buffer_dropped_message_count", 0))
        buffer_dropped_experience_count += int(stats.get("buffer_dropped_experience_count", 0))
        last_drop_error = stats.get("buffer_last_drop_error")
        if last_drop_error:
            buffer_last_drop_error = last_drop_error

    if total_sent <= 0:
        return {
            "sent_experience_count": 0,
            "sent_action_counts": action_counts,
            "sent_active_action_count": 0,
            "sent_boost_action_fraction": 0.0,
            "sent_exact_mask_fraction": 0.0,
            "sent_terminal_count": 0,
            "sent_terminal_fraction": 0.0,
            "sent_nonterminal_count": 0,
            "sent_nonterminal_exact_mask_fraction": 0.0,
            "sent_nonterminal_trapped_next_fraction": 0.0,
            "sent_positive_reward_fraction": 0.0,
            "sent_zero_reward_fraction": 0.0,
            "sent_negative_reward_fraction": 0.0,
            "sent_multistep_fraction": 0.0,
            "sent_invalid_current_action_count": 0,
            "sent_invalid_current_action_fraction": 0.0,
            "sent_invalid_current_normal_action_count": 0,
            "sent_invalid_current_boost_action_count": 0,
            "buffer_queued_message_count": 0,
            "buffer_dropped_message_count": 0,
            "buffer_dropped_experience_count": 0,
            "buffer_dropped_experience_fraction": 0.0,
            "buffer_last_drop_error": None,
        }

    return {
        "sent_experience_count": total_sent,
        "sent_action_counts": action_counts,
        "sent_active_action_count": sum(1 for count in action_counts if count > 0),
        "sent_boost_action_fraction": weighted_boost / total_sent,
        "sent_exact_mask_fraction": weighted_exact_masks / total_sent,
        "sent_terminal_count": terminal_count,
        "sent_terminal_fraction": weighted_terminals / total_sent,
        "sent_nonterminal_count": total_nonterminal,
        "sent_nonterminal_exact_mask_fraction": (
            weighted_nonterminal_exact_masks / total_nonterminal if total_nonterminal > 0 else 0.0
        ),
        "sent_nonterminal_trapped_next_fraction": (
            weighted_nonterminal_trapped_next / total_nonterminal if total_nonterminal > 0 else 0.0
        ),
        "sent_positive_reward_fraction": weighted_positive_rewards / total_sent,
        "sent_zero_reward_fraction": weighted_zero_rewards / total_sent,
        "sent_negative_reward_fraction": weighted_negative_rewards / total_sent,
        "sent_multistep_fraction": weighted_multistep / total_sent,
        "sent_invalid_current_action_count": invalid_current_action_count,
        "sent_invalid_current_action_fraction": invalid_current_action_count / total_sent,
        "sent_invalid_current_normal_action_count": invalid_current_normal_action_count,
        "sent_invalid_current_boost_action_count": invalid_current_boost_action_count,
        "buffer_queued_message_count": buffer_queued_message_count,
        "buffer_dropped_message_count": buffer_dropped_message_count,
        "buffer_dropped_experience_count": buffer_dropped_experience_count,
        "buffer_dropped_experience_fraction": buffer_dropped_experience_count / total_sent,
        "buffer_last_drop_error": buffer_last_drop_error,
    }


def format_actor_replay_warnings(actor_replay: dict, indent: str = "  ") -> list[str]:
    """Return warnings for actor replay streams that may train poorly."""
    sent_count = int(actor_replay.get("sent_experience_count", 0))
    if sent_count <= 0:
        return []

    warnings = []
    active_action_count = int(actor_replay.get("sent_active_action_count", 0))
    positive_reward_fraction = float(actor_replay.get("sent_positive_reward_fraction", 0.0))
    negative_reward_fraction = float(actor_replay.get("sent_negative_reward_fraction", 0.0))
    terminal_fraction = float(actor_replay.get("sent_terminal_fraction", 0.0))
    terminal_count = int(
        actor_replay.get("sent_terminal_count", round(terminal_fraction * sent_count))
    )
    exact_mask_fraction = float(actor_replay.get("sent_exact_mask_fraction", 0.0))
    nonterminal_exact_mask_fraction = float(
        actor_replay.get("sent_nonterminal_exact_mask_fraction", exact_mask_fraction)
    )
    nonterminal_trapped_next_fraction = float(
        actor_replay.get("sent_nonterminal_trapped_next_fraction", 0.0)
    )
    multistep_fraction = float(actor_replay.get("sent_multistep_fraction", 0.0))
    invalid_current_action_fraction = float(
        actor_replay.get("sent_invalid_current_action_fraction", 0.0)
    )
    invalid_current_action_count = int(actor_replay.get("sent_invalid_current_action_count", 0))
    buffer_dropped_experience_count = int(actor_replay.get("buffer_dropped_experience_count", 0))
    buffer_dropped_experience_fraction = float(
        actor_replay.get("buffer_dropped_experience_fraction", 0.0)
    )

    if sent_count >= 128 and active_action_count <= 2:
        warnings.append(
            f"{indent}Only {active_action_count}/{GameConfig.OUTPUT_SIZE} actions are being sent; "
            "actor exploration may be too narrow"
        )
    if sent_count >= 128 and positive_reward_fraction <= 0.0:
        warnings.append(
            f"{indent}No positive rewards in {sent_count:,} sent actor transitions; "
            "food/kill reward learning may be absent"
        )
    if sent_count >= 128 and negative_reward_fraction <= 0.0:
        warnings.append(
            f"{indent}No negative rewards in {sent_count:,} sent actor transitions; "
            "death/danger avoidance learning may be weak"
        )
    if sent_count >= 128 and terminal_count <= 0:
        warnings.append(
            f"{indent}No terminal rows in {sent_count:,} sent actor transitions; "
            "collision learning may be weak"
        )
    elif sent_count >= 512 and terminal_fraction < 0.005:
        warnings.append(
            f"{indent}Only {terminal_count:,}/{sent_count:,} sent actor transitions "
            f"({terminal_fraction:.2%}) are terminal; collision learning may be weak"
        )
    if sent_count >= 128 and exact_mask_fraction < 0.5:
        warnings.append(
            f"{indent}Only {exact_mask_fraction:.1%} of sent actor transitions have exact "
            "next-action masks; target masking may fall back often"
        )
    if sent_count >= 128 and nonterminal_exact_mask_fraction < 0.8:
        warnings.append(
            f"{indent}Only {nonterminal_exact_mask_fraction:.1%} of nonterminal actor "
            "transitions have exact next-action masks; bootstrapped target masking may be weak"
        )
    if sent_count >= 128 and nonterminal_trapped_next_fraction > 0.2:
        warnings.append(
            f"{indent}{nonterminal_trapped_next_fraction:.1%} of nonterminal actor targets "
            "have no valid next actions; state/action masking may be over-trapping replay"
        )
    if sent_count >= 128 and multistep_fraction <= 0.0:
        warnings.append(
            f"{indent}All sent actor replay rows use bootstrap_steps=1; "
            "n-step return signal may be absent"
        )
    if sent_count >= 128 and invalid_current_action_fraction > 0.02:
        warnings.append(
            f"{indent}{invalid_current_action_count:,}/{sent_count:,} sent actor transitions "
            f"({invalid_current_action_fraction:.1%}) are invalid under current-state "
            "danger/boost features; tune danger exploration or check action/state alignment"
        )
    if sent_count >= 128 and buffer_dropped_experience_fraction > 0.05:
        latest = actor_replay.get("buffer_last_drop_error") or "unknown reason"
        warnings.append(
            f"{indent}{buffer_dropped_experience_count:,}/{sent_count:,} actor transitions "
            f"({buffer_dropped_experience_fraction:.1%}) were dropped before reaching the "
            f"buffer; latest: {latest}"
        )

    return warnings


def validate_actor_replay_quality_gates(
    actor_replay: dict,
    *,
    min_terminal_fraction: float = 0.0,
) -> None:
    """Fail when distributed actor replay misses requested learning-signal floors."""
    min_terminal_fraction = resolve_actor_replay_quality_fraction(
        min_terminal_fraction,
        "min_actor_terminal_fraction",
    )
    sent_count = int(actor_replay.get("sent_experience_count", 0))
    terminal_count = int(actor_replay.get("sent_terminal_count", 0))
    terminal_fraction = float(actor_replay.get("sent_terminal_fraction", 0.0))
    if min_terminal_fraction > 0.0 and (
        sent_count <= 0 or terminal_fraction < min_terminal_fraction
    ):
        raise RuntimeError(
            "Actor replay terminal fraction "
            f"{terminal_fraction:.2%} ({terminal_count:,}/{sent_count:,}) is below the "
            f"requested minimum {min_terminal_fraction:.2%}"
        )


def format_actor_replay_summary(actor_replay: dict, indent: str = "  ") -> str:
    """Return a compact actor replay coverage line for coordinator logs."""
    action_counts = list(actor_replay.get("sent_action_counts", []))
    sent_count = int(actor_replay.get("sent_experience_count", 0))
    active_action_count = int(
        actor_replay.get(
            "sent_active_action_count",
            sum(1 for count in action_counts if int(count) > 0),
        )
    )
    nonterminal_trapped_fraction = float(
        actor_replay.get("sent_nonterminal_trapped_next_fraction", 0.0)
    )
    invalid_current_action_fraction = float(
        actor_replay.get("sent_invalid_current_action_fraction", 0.0)
    )
    terminal_count = int(
        actor_replay.get(
            "sent_terminal_count",
            round(float(actor_replay.get("sent_terminal_fraction", 0.0)) * sent_count),
        )
    )
    terminal_fraction = float(actor_replay.get("sent_terminal_fraction", 0.0))
    return (
        f"{indent}Actor replay: "
        f"sent={sent_count:,} | "
        f"actions={active_action_count}/{GameConfig.OUTPUT_SIZE} | "
        f"boost={float(actor_replay.get('sent_boost_action_fraction', 0.0)):.1%} | "
        f"masks={float(actor_replay.get('sent_exact_mask_fraction', 0.0)):.1%} | "
        f"nt_masks={float(actor_replay.get('sent_nonterminal_exact_mask_fraction', 0.0)):.1%} | "
        f"nt_trapped={nonterminal_trapped_fraction:.1%} | "
        f"terminal={terminal_count:,}/{sent_count:,} ({terminal_fraction:.2%}) | "
        f"reward+={float(actor_replay.get('sent_positive_reward_fraction', 0.0)):.1%} | "
        f"nstep={float(actor_replay.get('sent_multistep_fraction', 0.0)):.1%} | "
        f"invalid_actions={invalid_current_action_fraction:.1%} | "
        f"dropped={float(actor_replay.get('buffer_dropped_experience_fraction', 0.0)):.1%}"
    )


def attach_replay_health_metadata(
    state: dict,
    *,
    actor_replay: dict,
    buffer_replay_health: dict,
    actor_replay_gates: dict | None = None,
) -> dict:
    """Attach replay health diagnostics and warnings to a checkpoint state."""
    state["actor_replay_coverage"] = dict(actor_replay)
    state["actor_replay_warnings"] = format_actor_replay_warnings(actor_replay)
    if actor_replay_gates is not None:
        state["actor_replay_gates"] = dict(actor_replay_gates)
    state["buffer_replay_health"] = dict(buffer_replay_health)
    state["buffer_replay_warnings"] = format_buffer_replay_warnings(buffer_replay_health)
    return state


def log_actor_replay_coverage(tb_logger, actor_replay: dict, step: int) -> bool:
    """Log actor-side replay coverage to TensorBoard when stats are available."""
    if tb_logger is None:
        return False

    sent_count = int(actor_replay.get("sent_experience_count", 0))
    if sent_count <= 0:
        return False

    tb_logger.log_scalar("actor_replay/sent_experience_count", sent_count, step)
    tb_logger.log_scalar(
        "actor_replay/sent_active_action_count",
        int(actor_replay.get("sent_active_action_count", 0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_boost_action_fraction",
        float(actor_replay.get("sent_boost_action_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_exact_mask_fraction",
        float(actor_replay.get("sent_exact_mask_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_terminal_fraction",
        float(actor_replay.get("sent_terminal_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_nonterminal_exact_mask_fraction",
        float(actor_replay.get("sent_nonterminal_exact_mask_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_nonterminal_trapped_next_fraction",
        float(actor_replay.get("sent_nonterminal_trapped_next_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_positive_reward_fraction",
        float(actor_replay.get("sent_positive_reward_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_zero_reward_fraction",
        float(actor_replay.get("sent_zero_reward_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_negative_reward_fraction",
        float(actor_replay.get("sent_negative_reward_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_multistep_fraction",
        float(actor_replay.get("sent_multistep_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_invalid_current_action_fraction",
        float(actor_replay.get("sent_invalid_current_action_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/sent_invalid_current_action_count",
        int(actor_replay.get("sent_invalid_current_action_count", 0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/buffer_dropped_experience_count",
        int(actor_replay.get("buffer_dropped_experience_count", 0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/buffer_dropped_experience_fraction",
        float(actor_replay.get("buffer_dropped_experience_fraction", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/buffer_dropped_message_count",
        int(actor_replay.get("buffer_dropped_message_count", 0)),
        step,
    )
    tb_logger.log_scalar(
        "actor_replay/warning_count",
        len(format_actor_replay_warnings(actor_replay)),
        step,
    )

    action_counts = list(actor_replay.get("sent_action_counts", []))
    for action in range(GameConfig.OUTPUT_SIZE):
        count = int(action_counts[action]) if action < len(action_counts) else 0
        tb_logger.log_scalar(f"actor_replay/action_{action}_count", count, step)
        tb_logger.log_scalar(f"actor_replay/action_{action}_fraction", count / sent_count, step)

    return True


def collect_buffer_replay_health(buffer_client) -> dict:
    """Return replay insertion health stats from a local or distributed buffer client."""
    get_stats = getattr(buffer_client, "get_stats", None)
    if get_stats is None:
        return {}

    try:
        stats = get_stats(timeout=0.1)
    except TypeError:
        stats = get_stats()
    except Exception:
        return {}
    return dict(stats) if stats else {}


def format_buffer_replay_warnings(buffer_stats: dict, indent: str = "  ") -> list[str]:
    """Return warnings for replay rows rejected at the buffer boundary."""
    warnings = []
    actor_rejected_count = int(buffer_stats.get("total_rejected_actor_messages", 0))
    if actor_rejected_count > 0:
        last_message = str(buffer_stats.get("last_rejected_actor_message") or "unknown reason")
        warnings.append(
            f"{indent}{actor_rejected_count:,} actor replay message(s) were rejected by the "
            f"buffer; latest: {last_message}"
        )

    priority_rejected_count = int(buffer_stats.get("total_rejected_priority_updates", 0))
    if priority_rejected_count > 0:
        last_priority_update = str(
            buffer_stats.get("last_rejected_priority_update") or "unknown reason"
        )
        warnings.append(
            f"{indent}{priority_rejected_count:,} learner priority update(s) were rejected by "
            f"the buffer; latest: {last_priority_update}"
        )
    return warnings


def should_report_buffer_replay_warnings(
    buffer_stats: dict,
    last_rejected_actor_messages: int,
    last_rejected_priority_updates: int = 0,
) -> bool:
    """Return whether buffer-side replay warnings should be emitted again."""
    actor_rejected_count = int(buffer_stats.get("total_rejected_actor_messages", 0))
    priority_rejected_count = int(buffer_stats.get("total_rejected_priority_updates", 0))
    return actor_rejected_count > int(last_rejected_actor_messages) or (
        priority_rejected_count > int(last_rejected_priority_updates)
    )


def log_buffer_replay_health(tb_logger, buffer_stats: dict, step: int) -> bool:
    """Log buffer-side replay insertion health to TensorBoard when available."""
    if tb_logger is None or not buffer_stats:
        return False

    tb_logger.log_scalar("buffer/replay_size", int(buffer_stats.get("size", 0)), step)
    tb_logger.log_scalar(
        "buffer/total_added",
        int(buffer_stats.get("total_added", 0)),
        step,
    )
    tb_logger.log_scalar(
        "buffer/total_sampled",
        int(buffer_stats.get("total_sampled", 0)),
        step,
    )
    tb_logger.log_scalar(
        "buffer/fill_ratio",
        float(buffer_stats.get("fill_ratio", 0.0)),
        step,
    )
    tb_logger.log_scalar(
        "buffer/total_rejected_actor_messages",
        int(buffer_stats.get("total_rejected_actor_messages", 0)),
        step,
    )
    tb_logger.log_scalar(
        "buffer/total_rejected_priority_updates",
        int(buffer_stats.get("total_rejected_priority_updates", 0)),
        step,
    )
    tb_logger.log_scalar(
        "buffer/replay_warning_count",
        len(format_buffer_replay_warnings(buffer_stats)),
        step,
    )
    return True


def format_learner_sample_warnings(metrics: dict, indent: str = "  ") -> list[str]:
    """Return warnings for learner sampling failures after replay warmup."""
    sample_error_count = int(metrics.get("sample_error_count", 0))
    if sample_error_count <= 0:
        return []

    last_error = str(metrics.get("last_sample_error") or "unknown reason")
    return [
        f"{indent}{sample_error_count:,} learner sample error(s) while replay was "
        f"ready; latest: {last_error}"
    ]


def should_report_learner_sample_warnings(
    metrics: dict,
    last_sample_error_count: int,
) -> bool:
    """Return whether learner sample warnings should be emitted again."""
    sample_error_count = int(metrics.get("sample_error_count", 0))
    return sample_error_count > int(last_sample_error_count)


def log_learner_sample_health(tb_logger, metrics: dict, step: int) -> bool:
    """Log learner sampling health to TensorBoard when failures are visible."""
    if tb_logger is None or int(metrics.get("sample_error_count", 0)) <= 0:
        return False

    tb_logger.log_scalar(
        "learner/sample_error_count",
        int(metrics.get("sample_error_count", 0)),
        step,
    )
    tb_logger.log_scalar(
        "learner/sample_warning_count",
        len(format_learner_sample_warnings(metrics)),
        step,
    )
    return True


def train_apex(
    num_actors: Optional[int] = None,
    total_steps: int = 1_000_000,
    batch_size: Optional[int] = None,
    buffer_capacity: Optional[int] = None,
    n_step: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    config_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    weight_broadcast_interval: Optional[int] = None,
    checkpoint_interval: Optional[int] = None,
    log_interval: Optional[int] = None,
    actor_env_num_snakes: Optional[int] = None,
    actor_board_scale: Optional[float] = None,
    actor_food_multiplier: Optional[float] = None,
    actor_boost_exploration_rate: Optional[float] = None,
    actor_danger_exploration_rate: Optional[float] = None,
    actor_priority_mode: Optional[str] = None,
    opponent_pool_dir: Optional[str] = None,
    pool_latest_fraction: Optional[float] = None,
    min_actor_terminal_fraction: Optional[float] = None,
    override_reward_contract: bool = False,
    stagger_delay: float = 0.5,
    max_learner_updates: Optional[int] = None,
    max_environment_transitions: Optional[int] = None,
    max_wall_time_seconds: Optional[float] = None,
    heartbeat_timeout_seconds: float = 30.0,
    seed: Optional[int] = None,
    resume_mode: str = "weights-only",
) -> None:
    """Run distributed Ape-X DQN training.

    Args:
        num_actors: Number of parallel actor processes
        total_steps: Total learner update steps to run
        batch_size: Learner batch size
        buffer_capacity: Replay buffer capacity
        n_step: Actor n-step return horizon
        checkpoint_dir: Directory for saving checkpoints
        resume_checkpoint: Path to checkpoint to resume from
        config_path: Optional YAML config path
        log_dir: TensorBoard log directory
        weight_broadcast_interval: Learner steps between weight broadcasts
        checkpoint_interval: Learner steps between checkpoints
        log_interval: Learner steps between log prints
        actor_env_num_snakes: Number of snakes per actor environment.
        actor_board_scale: Actor arena width/height multiplier.
        actor_food_multiplier: Actor food count multiplier.
        actor_boost_exploration_rate: Probability that actor random exploration
            samples a safe boost action when one is available.
        actor_danger_exploration_rate: Probability that actor random exploration
            samples a known-unsafe legal action when one is available.
        actor_priority_mode: Insert priority for new actor transitions: "max"
            (buffer max priority, no actor-side TD forwards) or "td" (legacy
            local TD-error priorities). None uses config
            apex.actor_priority_mode (default "max").
        opponent_pool_dir: Directory of frozen opponent checkpoints for actor
            pool self-play (blueprint §3.3). None uses config
            apex.opponent_pool_dir (default None = pure mirror self-play).
        pool_latest_fraction: Per-episode probability that each non-hero actor
            snake slot runs the latest policy instead of a frozen pool
            checkpoint. None uses config apex.pool_latest_fraction (default 0.8).
        min_actor_terminal_fraction: Optional final actor replay-quality gate.
            Fails after cleanup when terminal actor replay is below this fraction.
        stagger_delay: Seconds between starting each actor
        max_learner_updates: Explicit learner-update budget. When omitted,
            ``total_steps`` is its backward-compatible alias.
        max_environment_transitions: Optional aggregate actor-frame budget.
        max_wall_time_seconds: Optional coordinator wall-clock budget.
        heartbeat_timeout_seconds: Maximum age of a reported actor heartbeat.
    """
    if resume_mode not in {"weights-only", "continuation"}:
        raise ValueError(f"Unsupported Apex resume mode {resume_mode!r}")
    # This is deliberately the first stochastic operation in the coordinator.
    # Child actors receive named streams from this one resolved identity.
    from src.core.seeding import initialize_run_seed

    seed_context = initialize_run_seed(seed)
    seed_manifest = {
        "requested_seed": seed_context.requested_seed,
        "effective_seed": seed_context.effective_seed,
        "namespace": "apex/distributed",
        "actor_namespace": "apex/actor/{actor_id}",
        "buffer_namespace": "apex/buffer",
    }
    print(
        f"Run seed: requested={seed_manifest['requested_seed']}, effective={seed_context.effective_seed}"
    )
    print("=" * 70)
    print("APE-X DQN DISTRIBUTED TRAINING")
    print("=" * 70)

    budgets = resolve_apex_run_budgets(
        total_steps,
        max_learner_updates=max_learner_updates,
        max_environment_transitions=max_environment_transitions,
        max_wall_time_seconds=max_wall_time_seconds,
    )
    total_steps = budgets.max_learner_updates
    if heartbeat_timeout_seconds <= 0:
        raise ValueError("heartbeat_timeout_seconds must be positive")

    use_config = config_path is not None
    if config_path:
        load_and_initialize_config(config_path)
        print(f"Loaded config: {config_path}")

    # Distributed actors do not implement curriculum coordination. Fail before
    # importing/constructing the buffer, network, queues, or child processes.
    if GameConfig.CURRICULUM_ENABLED:
        raise ValueError(
            "Distributed Ape-X does not support curriculum; use src/main.py --headless"
        )

    num_actors = int(_resolve_configurable(num_actors, GameConfig.APEX_NUM_ACTORS, 4, use_config))
    batch_size = int(_resolve_configurable(batch_size, GameConfig.APEX_BATCH_SIZE, 512, use_config))
    buffer_capacity = int(
        _resolve_configurable(
            buffer_capacity,
            GameConfig.APEX_BUFFER_SIZE,
            100_000,
            use_config,
        )
    )
    n_step = int(_resolve_configurable(n_step, GameConfig.APEX_N_STEP, 3, use_config))
    checkpoint_dir = str(
        _resolve_configurable(
            checkpoint_dir,
            GameConfig.CHECKPOINT_DIR,
            "saved_snakes/",
            use_config,
        )
    )
    log_dir = str(
        _resolve_configurable(
            log_dir,
            "logs/tensorboard/apex",
            "logs/tensorboard/apex",
            use_config,
        )
    )
    weight_broadcast_interval = int(
        _resolve_configurable(
            weight_broadcast_interval,
            GameConfig.APEX_ACTOR_UPDATE_FREQ,
            400,
            use_config,
        )
    )
    checkpoint_interval = int(
        _resolve_configurable(
            checkpoint_interval,
            GameConfig.CHECKPOINT_FREQUENCY,
            50_000,
            use_config,
        )
    )
    log_interval = int(
        _resolve_configurable(log_interval, GameConfig.LOG_INTERVAL, 1_000, use_config)
    )
    actor_replay_gates = build_actor_replay_quality_gates(
        min_terminal_fraction=min_actor_terminal_fraction
    )
    min_buffer_size = resolve_apex_min_buffer_size(
        batch_size,
        buffer_capacity,
        GameConfig.APEX_MIN_BUFFER_SIZE,
    )
    import torch
    import torch.multiprocessing as mp

    from src.core.device_manager import DeviceManager
    from src.model.apex_network import ApexNetwork
    from src.training.apex_actor import (
        DEFAULT_ACTOR_BOOST_EXPLORATION_RATE,
        DEFAULT_ACTOR_DANGER_EXPLORATION_RATE,
        spawn_actors,
        start_actors,
    )
    from src.training.apex_buffer import BufferProcess
    from src.training.apex_learner import create_apex_learner

    actor_boost_exploration_rate = float(
        _resolve_configurable(
            actor_boost_exploration_rate,
            DEFAULT_ACTOR_BOOST_EXPLORATION_RATE,
            DEFAULT_ACTOR_BOOST_EXPLORATION_RATE,
            use_config,
        )
    )
    actor_danger_exploration_rate = float(
        _resolve_configurable(
            actor_danger_exploration_rate,
            DEFAULT_ACTOR_DANGER_EXPLORATION_RATE,
            DEFAULT_ACTOR_DANGER_EXPLORATION_RATE,
            use_config,
        )
    )
    actor_env_num_snakes = int(
        _resolve_configurable(
            actor_env_num_snakes,
            GameConfig.APEX_ACTOR_ENV_NUM_SNAKES,
            GameConfig.APEX_ACTOR_ENV_NUM_SNAKES,
            use_config,
        )
    )
    actor_board_scale = float(
        _resolve_configurable(
            actor_board_scale,
            GameConfig.APEX_ACTOR_BOARD_SCALE,
            GameConfig.APEX_ACTOR_BOARD_SCALE,
            use_config,
        )
    )
    actor_food_multiplier = float(
        _resolve_configurable(
            actor_food_multiplier,
            GameConfig.APEX_ACTOR_FOOD_MULTIPLIER,
            GameConfig.APEX_ACTOR_FOOD_MULTIPLIER,
            use_config,
        )
    )
    actor_priority_mode = str(
        _resolve_configurable(
            actor_priority_mode,
            GameConfig.APEX_ACTOR_PRIORITY_MODE,
            GameConfig.APEX_ACTOR_PRIORITY_MODE,
            use_config,
        )
    )
    opponent_pool_dir = _resolve_configurable(
        opponent_pool_dir,
        GameConfig.APEX_OPPONENT_POOL_DIR,
        GameConfig.APEX_OPPONENT_POOL_DIR,
        use_config,
    )
    pool_latest_fraction = float(
        _resolve_configurable(
            pool_latest_fraction,
            GameConfig.APEX_POOL_LATEST_FRACTION,
            GameConfig.APEX_POOL_LATEST_FRACTION,
            use_config,
        )
    )
    validate_apex_training_config(
        num_actors=num_actors,
        total_steps=total_steps,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        n_step=n_step,
        min_buffer_size=min_buffer_size,
        weight_broadcast_interval=weight_broadcast_interval,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        stagger_delay=stagger_delay,
        actor_env_num_snakes=actor_env_num_snakes,
        actor_board_scale=actor_board_scale,
        actor_food_multiplier=actor_food_multiplier,
        actor_boost_exploration_rate=actor_boost_exploration_rate,
        actor_danger_exploration_rate=actor_danger_exploration_rate,
    )

    # ── Device setup ──────────────────────────────────────────────────
    device = DeviceManager.get_device()
    print(f"\nLearner device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ── Configuration ─────────────────────────────────────────────────
    input_size = GameConfig.INPUT_SIZE
    hidden_size = GameConfig.HIDDEN_SIZE
    output_size = GameConfig.OUTPUT_SIZE
    gamma = GameConfig.APEX_GAMMA
    learning_rate = GameConfig.APEX_LEARNING_RATE
    target_update_freq = GameConfig.APEX_TARGET_UPDATE_FREQ
    apex_checkpoint_config = build_apex_checkpoint_config(
        num_actors=num_actors,
        total_steps=total_steps,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        n_step=n_step,
        min_buffer_size=min_buffer_size,
        learning_rate=learning_rate,
        gamma=gamma,
        target_update_freq=target_update_freq,
        weight_broadcast_interval=weight_broadcast_interval,
        priority_alpha=GameConfig.APEX_PRIORITY_ALPHA,
        priority_beta_start=GameConfig.APEX_PRIORITY_BETA_START,
        priority_beta_end=GameConfig.APEX_PRIORITY_BETA_END,
        priority_beta_frames=total_steps,
        priority_epsilon=GameConfig.APEX_PRIORITY_EPSILON,
        grad_clip_norm=GameConfig.GRAD_CLIP_NORM,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        actor_env_num_snakes=actor_env_num_snakes,
        actor_board_scale=actor_board_scale,
        actor_food_multiplier=actor_food_multiplier,
        actor_boost_exploration_rate=actor_boost_exploration_rate,
        actor_danger_exploration_rate=actor_danger_exploration_rate,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        reward_death=GameConfig.REWARD_DEATH,
        reward_food_base=GameConfig.REWARD_FOOD_BASE,
    )
    print("\nConfiguration:")
    if config_path:
        print(f"  Config path:      {config_path}")
    print(f"  Actors:          {num_actors}")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  Update budget:   {budgets.max_learner_updates:,}")
    print(f"  Env budget:      {budgets.max_environment_transitions or '(unbounded)'}")
    print(f"  Wall budget:     {budgets.max_wall_time_seconds or '(unbounded)'} seconds")
    print(f"  Batch size:      {batch_size}")
    print(f"  Buffer capacity: {buffer_capacity:,}")
    print(f"  Min buffer size: {min_buffer_size:,}")
    print(f"  N-step returns:  {n_step}")
    print(f"  State dim:       {input_size}")
    print(f"  Hidden dim:      {hidden_size}")
    print(f"  Actions:         {output_size}")
    print(f"  Actor env snakes: {actor_env_num_snakes}")
    print(f"  Actor board scale: {actor_board_scale:.2f}")
    print(f"  Actor food multiplier: {actor_food_multiplier:.2f}")
    print(f"  Actor boost exploration:  {actor_boost_exploration_rate:.2f}")
    print(f"  Actor danger exploration: {actor_danger_exploration_rate:.2f}")
    print(f"  Actor priority mode: {actor_priority_mode}")
    print(f"  Opponent pool dir: {opponent_pool_dir or '(disabled: pure mirror self-play)'}")
    print(f"  Pool latest fraction: {pool_latest_fraction:.2f}")
    print(f"  Gamma:           {gamma}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Target update:   {target_update_freq}")
    print()

    resume_checkpoint_state = load_validated_apex_resume_checkpoint(
        resume_checkpoint,
        apex_checkpoint_config,
        map_location=device,
        override_reward_contract=override_reward_contract,
        resume_mode=resume_mode,
    )

    # Resumed learner step, used both to start the training loop and to seed the
    # buffer's beta-annealing clock below. The BufferProcess is created fresh each
    # launch and its beta clock is NOT persisted, so without this seed a resumed
    # run would re-anneal IS-weight beta from beta_start over a full fresh window.
    resume_start_step = (
        int(resume_checkpoint_state.get("step_count", 0))
        if resume_checkpoint_state is not None and resume_mode != "weights-only"
        else 0
    )

    # ── Multiprocessing setup ─────────────────────────────────────────
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    # Limit actor threads to prevent CPU contention
    torch.set_num_threads(1)

    # ── Create BufferProcess ──────────────────────────────────────────
    print("Starting BufferProcess...")
    buffer_process = BufferProcess(
        capacity=buffer_capacity,
        alpha=GameConfig.APEX_PRIORITY_ALPHA,
        beta_start=GameConfig.APEX_PRIORITY_BETA_START,
        beta_end=GameConfig.APEX_PRIORITY_BETA_END,
        beta_frames=total_steps,
        state_size=input_size,
        # Seed beta annealing from the resumed step (0 for fresh runs). beta_frames
        # stays absolute (total_steps); do not also subtract start_step or the
        # offset would be double-counted.
        initial_frame_count=resume_start_step,
        base_seed=seed_context.stream_seed("apex/buffer"),
    )
    # ── Create shared network (CPU) for initial actor weight sync ─────
    shared_network = ApexNetwork(input_size, hidden_size, output_size)
    shared_network.eval()
    shared_network.share_memory()

    # ── Create queues ─────────────────────────────────────────────────
    weight_queues = [mp.Queue(maxsize=2) for _ in range(num_actors)]
    stats_queue = mp.Queue(maxsize=1000)
    stop_event = mp.Event()

    # ── Create learner ────────────────────────────────────────────────
    print("Initializing ApexLearner...")
    learner_client = buffer_process.get_learner_client()
    learner = create_apex_learner(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        n_step=n_step,
        target_update_freq=target_update_freq,
        buffer_client=learner_client,
        log_dir=log_dir,
        min_buffer_size=min_buffer_size,
        weight_broadcast_interval=weight_broadcast_interval,
        priority_alpha=GameConfig.APEX_PRIORITY_ALPHA,
        priority_eps=GameConfig.APEX_PRIORITY_EPSILON,
        grad_clip_norm=GameConfig.GRAD_CLIP_NORM,
        log_interval=log_interval,
    )

    from src.core.game_config import get_config
    from src.training.apex_recipe import ApexRecipe, validate_recipe_continuation

    learner_optimizer = getattr(learner, "optimizer", None)
    requested_recipe = (
        ApexRecipe.distributed(
            optimizer=learner_optimizer,
            input_size=input_size,
            output_size=output_size,
            gamma=gamma,
            n_step=n_step,
            priority_alpha=GameConfig.APEX_PRIORITY_ALPHA,
            priority_eps=GameConfig.APEX_PRIORITY_EPSILON,
            world=asdict(get_config().game),
            runtime={
                "mode": "distributed_apex",
                "training": True,
                "distributed": True,
                "curriculum": False,
                "replay_restored": False,
                "inflight_actor_state_restored": False,
                "rng_state_restored": False,
            },
            actor_scaling={
                "num_actors": num_actors,
                "actor_env_num_snakes": actor_env_num_snakes,
                "actor_board_scale": actor_board_scale,
                "actor_food_multiplier": actor_food_multiplier,
            },
            seed_identity={**seed_manifest, "actor_namespace": "apex/actor/{actor_id}"},
            target_clip=100.0,
            grad_clip_norm=learner.config.grad_clip_norm,
            batch_size=batch_size,
            replay_capacity=buffer_capacity,
            min_replay_size=min_buffer_size,
            beta_frames=total_steps,
            initial_beta_clock=resume_start_step,
        )
        if learner_optimizer is not None
        else None
    )
    if requested_recipe is not None and hasattr(learner, "config"):
        learner.config.apex_recipe = requested_recipe
    if resume_checkpoint_state is not None and resume_mode == "continuation":
        # All recipe and optimizer checks happen before weights, actors, or
        # BufferProcess are mutated or started.
        validate_recipe_continuation(
            resume_checkpoint_state,
            requested_recipe,
            weights_only=False,
            optimizer=learner_optimizer,
        )

    try:
        # Sync shared network weights from learner
        shared_network.load_state_dict({k: v.cpu() for k, v in learner.dqn.state_dict().items()})

        # ── Resume from checkpoint ────────────────────────────────────────
        start_step = 0
        if resume_checkpoint_state is not None:
            print(f"Resuming from: {resume_checkpoint}")
            learner.load_state_dict(
                resume_checkpoint_state,
                resume_mode=resume_mode,
                requested_recipe=requested_recipe,
            )
            start_step = resume_start_step
            # Re-sync shared network
            shared_network.load_state_dict(
                {k: v.cpu() for k, v in learner.dqn.state_dict().items()}
            )
            print(f"  Resumed at step {start_step:,}")
            for line in format_apex_checkpoint_provenance(resume_checkpoint_state):
                print(f"  {line}")

        # The initial actor message carries the learner's real update version,
        # including a resumed continuation. Actors must not infer it from queue
        # receipt count.
        broadcast_weights(learner_weight_payload(learner), weight_queues)

        # ── Spawn actors ──────────────────────────────────────────────────
        print(f"\nSpawning {num_actors} actors...")
        shared_actor_progress = [SharedActorProgress(mp) for _ in range(num_actors)]
        # The unbounded default must retain the original actor hot path: only a
        # declared frame ceiling installs the cross-process reservation lock.
        environment_frame_budget = (
            None
            if budgets.max_environment_transitions is None
            else SharedEnvironmentFrameBudget(mp, budgets.max_environment_transitions)
        )
        actors = spawn_actors(
            num_actors=num_actors,
            shared_network=shared_network,
            buffer_process=buffer_process,
            weight_queues=weight_queues,
            stats_queue=stats_queue,
            stop_event=stop_event,
            shared_progress=shared_actor_progress,
            environment_frame_budget=environment_frame_budget,
            gamma=gamma,
            n_step=n_step,
            alpha=GameConfig.APEX_PRIORITY_ALPHA,
            priority_eps=GameConfig.APEX_PRIORITY_EPSILON,
            base_epsilon=GameConfig.APEX_EPSILON_BASE,
            epsilon_alpha=GameConfig.APEX_EPSILON_ALPHA,
            weight_sync_interval=weight_broadcast_interval,
            env_num_snakes=actor_env_num_snakes,
            env_board_scale=actor_board_scale,
            env_food_multiplier=actor_food_multiplier,
            boost_exploration_rate=actor_boost_exploration_rate,
            danger_exploration_rate=actor_danger_exploration_rate,
            actor_priority_mode=actor_priority_mode,
            opponent_pool_dir=opponent_pool_dir,
            pool_latest_fraction=pool_latest_fraction,
            config_path=config_path,
            base_seed=seed_context.effective_seed,
        )

        # ── Checkpoint manager ────────────────────────────────────────────
        os.makedirs(checkpoint_dir, exist_ok=True)

        # ── Signal handler for graceful shutdown ───────────────────────────
        shutdown_requested = [False]

        def signal_handler(signum, frame):
            print("\n[Coordinator] Shutdown requested...")
            shutdown_requested[0] = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # ── Start actors with staggered delay ─────────────────────────────
        print("Starting BufferProcess and actors...")
        start_time = time.time()
        runtime_supervisor = ApexRuntimeSupervisor(
            actors=[],
            buffer_process=buffer_process,
            budgets=budgets,
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        )

        def shared_health() -> dict[int, dict]:
            return {
                index: progress.snapshot() for index, progress in enumerate(shared_actor_progress)
            }

        def current_runtime_cause() -> str | None:
            reserved_frames = (
                0
                if environment_frame_budget is None
                else environment_frame_budget.reserved_frames()
            )
            return runtime_supervisor.current_budget_cause(
                learner_updates=learner.step_count,
                reserved_environment_frames=reserved_frames,
            )

        def environment_budget_reached() -> bool:
            return (
                environment_frame_budget is not None
                and environment_frame_budget.reserved_frames()
                >= budgets.max_environment_transitions
            )

        def supervise_started_actor(actor) -> None:
            runtime_supervisor.register_actor(actor)
            cause = current_runtime_cause()
            runtime_supervisor.check_children(
                shared_health(),
                expected_stop_cause=cause,
                environment_budget_reached=environment_budget_reached(),
            )
            if cause:
                raise ApexControlledStop(cause)

        startup_exit_cause: str | None = None

        def supervise_startup_wait() -> None:
            cause = current_runtime_cause()
            runtime_supervisor.check_children(
                shared_health(),
                expected_stop_cause=cause,
                environment_budget_reached=environment_budget_reached(),
            )
            if cause:
                raise ApexControlledStop(cause)

        # ── Main training loop ────────────────────────────────────────────
        print("Waiting for buffer to fill...")
        episode_rewards: deque = deque(maxlen=100)
        actor_stats_by_id = {}
        last_log_step = start_step
        last_save_step = start_step
        last_reported_buffer_rejection_count = 0
        last_reported_buffer_priority_rejection_count = 0
        last_reported_learner_sample_error_count = 0
        step = start_step
        exit_cause = startup_exit_cause or "signal"
        run_failure: BaseException | None = None
        latest_snapshot: ApexRuntimeSnapshot | None = None

        def stop_buffer_and_learner() -> list[BaseException]:
            """Always release the remaining child and learner, preserving all failures."""
            cleanup_errors: list[BaseException] = []
            try:
                buffer_process.shutdown(timeout=5.0)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                buffer_child = getattr(buffer_process, "_process", None)
                if buffer_child is not None:
                    stop_processes([buffer_child], timeout_seconds=5.0)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                learner.cleanup()
            except BaseException as error:
                cleanup_errors.append(error)
            return cleanup_errors

    except BaseException as error:
        try:
            learner.cleanup()
        except BaseException as cleanup_error:
            error.add_note(f"Pre-start learner cleanup also failed: {cleanup_error!r}")
        raise

    try:
        buffer_process.start()
        try:
            supervise_startup_wait()
            start_actors(
                actors,
                stagger_delay=stagger_delay,
                on_started=supervise_started_actor,
                on_wait=supervise_startup_wait,
            )
        except ApexControlledStop as stop:
            startup_exit_cause = stop.cause
            exit_cause = stop.cause
        print(f"  {len(runtime_supervisor.actors)} actors started.\n")
        while not shutdown_requested[0] and startup_exit_cause is None:
            update_latest_actor_stats(
                actor_stats_by_id,
                collect_actor_stats(stats_queue),
                episode_rewards=episode_rewards,
            )
            actor_stats = list(actor_stats_by_id.values())
            precheck_cause = current_runtime_cause()
            heartbeat_ages = runtime_supervisor.check_children(
                shared_health(),
                expected_stop_cause=precheck_cause,
                environment_budget_reached=environment_budget_reached(),
            )
            buffer_replay_health = collect_buffer_replay_health(learner.buffer_client)
            latest_snapshot = build_apex_runtime_snapshot(
                learner_updates=learner.step_count,
                actor_stats=actor_stats,
                buffer_replay_health=buffer_replay_health,
                elapsed_seconds=runtime_supervisor.elapsed_seconds(),
                actor_heartbeat_ages=heartbeat_ages,
                shared_actor_progress=shared_actor_progress,
                environment_frame_budget=environment_frame_budget,
            )
            exit_cause = precheck_cause or runtime_supervisor.stop_cause(latest_snapshot) or ""
            if exit_cause:
                break

            # ── Learner training step ─────────────────────────────────
            metrics = learner.train_step()

            # If buffer not ready yet, collect stats and wait
            if metrics.get("status") == "waiting":
                update_latest_actor_stats(
                    actor_stats_by_id,
                    collect_actor_stats(stats_queue),
                    episode_rewards=episode_rewards,
                )
                buffer_replay_health = collect_buffer_replay_health(learner.buffer_client)
                if should_report_buffer_replay_warnings(
                    buffer_replay_health,
                    last_reported_buffer_rejection_count,
                    last_reported_buffer_priority_rejection_count,
                ):
                    print("  Buffer replay warnings while waiting for fill:")
                    for warning in format_buffer_replay_warnings(buffer_replay_health):
                        print(warning)
                    log_buffer_replay_health(
                        getattr(learner, "tb_logger", None),
                        buffer_replay_health,
                        step,
                    )
                    last_reported_buffer_rejection_count = int(
                        buffer_replay_health.get("total_rejected_actor_messages", 0)
                    )
                    last_reported_buffer_priority_rejection_count = int(
                        buffer_replay_health.get("total_rejected_priority_updates", 0)
                    )
                if should_report_learner_sample_warnings(
                    metrics,
                    last_reported_learner_sample_error_count,
                ):
                    print("  Learner sample warnings while waiting for fill:")
                    for warning in format_learner_sample_warnings(metrics):
                        print(warning)
                    log_learner_sample_health(
                        getattr(learner, "tb_logger", None),
                        metrics,
                        step,
                    )
                    last_reported_learner_sample_error_count = int(
                        metrics.get("sample_error_count", 0)
                    )
                time.sleep(0.1)
                continue

            step = learner.step_count

            # ── Broadcast weights to actors ───────────────────────────
            if learner.should_broadcast_weights():
                broadcast_weights(learner_weight_payload(learner), weight_queues)

            # ── Collect actor stats ───────────────────────────────────
            update_latest_actor_stats(
                actor_stats_by_id,
                collect_actor_stats(stats_queue),
                episode_rewards=episode_rewards,
            )
            precheck_cause = current_runtime_cause()
            heartbeat_ages = runtime_supervisor.check_children(
                shared_health(),
                expected_stop_cause=precheck_cause,
                environment_budget_reached=environment_budget_reached(),
            )
            latest_snapshot = build_apex_runtime_snapshot(
                learner_updates=learner.step_count,
                actor_stats=list(actor_stats_by_id.values()),
                buffer_replay_health=collect_buffer_replay_health(learner.buffer_client),
                elapsed_seconds=runtime_supervisor.elapsed_seconds(),
                actor_heartbeat_ages=heartbeat_ages,
                shared_actor_progress=shared_actor_progress,
                environment_frame_budget=environment_frame_budget,
            )
            exit_cause = precheck_cause or runtime_supervisor.stop_cause(latest_snapshot) or ""
            if exit_cause:
                break

            # ── Logging ───────────────────────────────────────────────
            if step - last_log_step >= log_interval:
                elapsed = time.time() - start_time
                sps = step / max(elapsed, 1e-6)
                avg_reward = _mean_or_zero(episode_rewards)
                actor_replay = summarize_actor_replay_coverage(list(actor_stats_by_id.values()))
                buffer_replay_health = collect_buffer_replay_health(learner.buffer_client)
                active_actors = sum(1 for a in actors if a.is_alive())
                loss = metrics.get("loss", 0)
                mean_q = metrics.get("mean_q_value", 0)

                print(
                    f"Step {step:,}/{total_steps:,} | "
                    f"SPS: {sps:.0f} | "
                    f"Loss: {loss:.4f} | "
                    f"Q: {mean_q:.2f} | "
                    f"Reward: {avg_reward:.2f} | "
                    f"Actors: {active_actors}/{num_actors}"
                )
                if actor_replay["sent_experience_count"]:
                    actor_replay_warnings = format_actor_replay_warnings(actor_replay)
                    print(format_actor_replay_summary(actor_replay))
                    if actor_replay_warnings:
                        print("  Actor replay warnings:")
                        for warning in actor_replay_warnings:
                            print(warning)
                    log_actor_replay_coverage(
                        getattr(learner, "tb_logger", None),
                        actor_replay,
                        step,
                    )
                buffer_replay_warnings = format_buffer_replay_warnings(buffer_replay_health)
                if buffer_replay_warnings:
                    print("  Buffer replay warnings:")
                    for warning in buffer_replay_warnings:
                        print(warning)
                log_buffer_replay_health(
                    getattr(learner, "tb_logger", None),
                    buffer_replay_health,
                    step,
                )
                last_log_step = step

            # ── Periodic checkpoint ───────────────────────────────────
            if step - last_save_step >= checkpoint_interval:
                ckpt_path = os.path.join(checkpoint_dir, f"apex_checkpoint_{step}.pth")
                state = learner.get_state_dict()
                state["apex_config"] = dict(apex_checkpoint_config)
                state["run_seed_manifest"] = dict(seed_manifest)
                state["resume_mode"] = (
                    resume_mode if resume_checkpoint_state is not None else "fresh"
                )
                state["avg_reward"] = _mean_or_zero(episode_rewards)
                attach_runtime_metadata(
                    state,
                    latest_snapshot,
                    "periodic_checkpoint",
                    budgets,
                    {
                        index: progress.snapshot()["policy_version"]
                        for index, progress in enumerate(shared_actor_progress)
                    },
                )
                attach_replay_health_metadata(
                    state,
                    actor_replay=summarize_actor_replay_coverage(list(actor_stats_by_id.values())),
                    buffer_replay_health=collect_buffer_replay_health(learner.buffer_client),
                    actor_replay_gates=actor_replay_gates,
                )
                torch.save(state, ckpt_path)

                # Also save as latest_apex.pth: a rolling pointer to the most
                # recent snapshot, written unconditionally with NO metric gate.
                # This is deliberately NOT named best_apex.pth — that file is the
                # CI-promoted champion living in the default checkpoint_dir
                # (saved_snakes/), and overwriting it from here would silently
                # clobber the curated model with a possibly-regressed late
                # snapshot. Champion promotion is owned by the tournament_eval
                # workflow, gated on a fixed benchmark.
                latest_path = os.path.join(checkpoint_dir, "latest_apex.pth")
                torch.save(state, latest_path)

                print(f"  Checkpoint saved: {ckpt_path}")
                last_save_step = step

    except BaseException as error:
        run_failure = error
        exit_cause = "failed"
        print(f"\n[Coordinator] Error: {error}")
        import traceback

        traceback.print_exc()

    finally:
        # Capture state while the learner is available, release every runtime
        # owner independently, and only then publish a terminal artifact.
        print("\n[Coordinator] Shutting down...")
        cleanup_errors: list[BaseException] = []
        final_state = None
        final_snapshot = None
        final_step = learner.step_count
        completed = False

        def attempt_cleanup(operation) -> None:
            try:
                operation()
            except BaseException as error:
                cleanup_errors.append(error)

        attempt_cleanup(stop_event.set)
        attempt_cleanup(lambda: stop_processes(actors, timeout_seconds=5.0))
        try:
            update_latest_actor_stats(
                actor_stats_by_id,
                collect_actor_stats(stats_queue),
                episode_rewards=episode_rewards,
            )
            final_actor_replay = summarize_actor_replay_coverage(list(actor_stats_by_id.values()))
            completed = (
                run_failure is None
                and not shutdown_requested[0]
                and exit_cause
                in {"learner_update_budget", "environment_transition_budget", "wall_time_budget"}
            )
            if completed:
                validate_actor_replay_quality_gates(
                    final_actor_replay,
                    min_terminal_fraction=actor_replay_gates["min_actor_terminal_fraction"],
                )
            captured_state = learner.get_state_dict()
            captured_state["apex_config"] = dict(apex_checkpoint_config)
            captured_state["run_seed_manifest"] = dict(seed_manifest)
            captured_state["resume_mode"] = (
                resume_mode if resume_checkpoint_state is not None else "fresh"
            )
            captured_state["avg_reward"] = _mean_or_zero(episode_rewards)
            final_buffer_replay_health = collect_buffer_replay_health(learner.buffer_client)
            attach_replay_health_metadata(
                captured_state,
                actor_replay=final_actor_replay,
                buffer_replay_health=final_buffer_replay_health,
                actor_replay_gates=actor_replay_gates,
            )
            final_snapshot = build_apex_runtime_snapshot(
                learner_updates=final_step,
                actor_stats=list(actor_stats_by_id.values()),
                buffer_replay_health=final_buffer_replay_health,
                elapsed_seconds=runtime_supervisor.elapsed_seconds(),
                actor_heartbeat_ages=runtime_supervisor.heartbeat_ages(shared_health()),
                shared_actor_progress=shared_actor_progress,
                environment_frame_budget=environment_frame_budget,
            )
            final_state = captured_state
        except BaseException as error:
            completed = False
            exit_cause = "failed_finalization"
            if run_failure is None:
                run_failure = error
            else:
                run_failure.add_note(f"Finalization also failed: {error!r}")
        finally:
            cleanup_errors.extend(stop_buffer_and_learner())

        if cleanup_errors:
            completed = False
            exit_cause = "failed_cleanup"
            if run_failure is None:
                run_failure = cleanup_errors[0]
            for error in cleanup_errors:
                if error is not run_failure:
                    run_failure.add_note(f"Cleanup also failed: {error!r}")

        if final_state is not None and final_snapshot is not None:
            final_snapshot = replace(
                final_snapshot, elapsed_seconds=runtime_supervisor.elapsed_seconds()
            )
            attach_runtime_metadata(
                final_state,
                final_snapshot,
                exit_cause or "interrupted",
                budgets,
                {
                    index: progress.snapshot()["policy_version"]
                    for index, progress in enumerate(shared_actor_progress)
                },
            )
            final_name = (
                "apex_final.pth"
                if completed
                else f"apex_{exit_cause or 'interrupted'}_{final_step}.pth"
            )
            final_path = os.path.join(checkpoint_dir, final_name)
            temporary_path = None
            try:
                descriptor, temporary_path = tempfile.mkstemp(
                    prefix=".apex-terminal-", suffix=".pth", dir=checkpoint_dir
                )
                os.close(descriptor)
                torch.save(final_state, temporary_path)
                os.replace(temporary_path, final_path)
                print(f"  {'Final' if completed else 'Recovery'} checkpoint: {final_path}")
            except BaseException as error:
                if run_failure is None:
                    run_failure = error
                else:
                    run_failure.add_note(f"Checkpoint publication also failed: {error!r}")
            finally:
                if temporary_path is not None:
                    try:
                        os.unlink(temporary_path)
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        if run_failure is None:
                            run_failure = error
                        else:
                            run_failure.add_note(f"Temporary checkpoint cleanup failed: {error!r}")

        print(f"  Learner updates: {final_step:,}")
        print(f"  Elapsed seconds: {runtime_supervisor.elapsed_seconds():.2f}")
        print(f"  Exit cause: {exit_cause or 'interrupted'}")
        if run_failure is not None:
            raise run_failure
        if shutdown_requested[0]:
            raise RuntimeError("Ape-X run interrupted before a declared budget was reached")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Ape-X DQN Distributed Training Coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Small local test (Mac, 4 actors)
  python src/scripts/apex_train.py --num-actors 4 --total-steps 100000

  # Full distributed (H100, 64 actors)
  python src/scripts/apex_train.py --num-actors 64 --total-steps 10000000 --batch-size 512

  # Resume from checkpoint
  python src/scripts/apex_train.py --resume saved_snakes/apex_checkpoint.pth

  # With config file
  python src/scripts/apex_train.py --config configs/production.yaml
        """,
    )

    parser.add_argument(
        "--num-actors",
        type=int,
        default=None,
        help="Number of parallel actor processes (default: 4, or config apex.num_actors)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Deprecated compatibility alias for --max-learner-updates (default: 1,000,000)",
    )
    parser.add_argument(
        "--max-learner-updates",
        type=int,
        default=None,
        help="Hard ceiling on learner updates; must match --total-steps when both are supplied",
    )
    parser.add_argument(
        "--max-environment-transitions",
        type=int,
        default=None,
        help="Optional hard ceiling on aggregate actor environment transitions",
    )
    parser.add_argument(
        "--max-wall-time-seconds",
        type=float,
        default=None,
        help=(
            "Cooperative wall-time budget checked during startup and between learner steps; "
            "shutdown or a slow step can overshoot"
        ),
    )
    parser.add_argument(
        "--heartbeat-timeout-seconds",
        type=float,
        default=30.0,
        help="Fail when a live actor heartbeat is older than this many seconds",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Learner batch size (default: 512, or config apex.batch_size)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Replay buffer capacity (default: 100,000, or config apex.buffer_size)",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=None,
        help="Actor n-step return horizon (default: 3, or config apex.n_step)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for checkpoints (default: saved_snakes/, or config checkpoint dir)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("weights-only", "continuation"),
        default="weights-only",
        help=(
            "Checkpoint restore policy. weights-only creates fresh optimizer, odometer, "
            "replay, actor and RNG runtime; continuation requires a verified matching "
            "recipe. Legacy checkpoints can be loaded only as weights-only."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional unsigned 64-bit run seed; omitted uses recorded entropy.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Steps between checkpoints (default: 50,000, or config training checkpoint_frequency)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Steps between log prints (default: 1,000, or config training.log_interval)",
    )
    parser.add_argument(
        "--actor-boost-exploration-rate",
        type=float,
        default=None,
        help=(
            "Actor random-exploration probability of choosing a safe boost action when "
            "available (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--actor-danger-exploration-rate",
        type=float,
        default=None,
        help=(
            "Actor random-exploration probability of choosing a known-unsafe legal action "
            "when available. Keep at 0.0 for executable-action replay; raise only for "
            "diagnostic terminal probing (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--actor-env-num-snakes",
        type=int,
        default=None,
        help=(
            "Snakes per actor environment for terminal-rich replay "
            "(default: config apex.actor_env_num_snakes)"
        ),
    )
    parser.add_argument(
        "--actor-board-scale",
        type=float,
        default=None,
        help=(
            "Actor arena width/height multiplier for collision-dense executable replay "
            "(default: config apex.actor_board_scale)"
        ),
    )
    parser.add_argument(
        "--actor-food-multiplier",
        type=float,
        default=None,
        help=(
            "Actor food-count multiplier for dense replay "
            "(default: config apex.actor_food_multiplier)"
        ),
    )
    parser.add_argument(
        "--actor-priority-mode",
        type=str,
        choices=("max", "td"),
        default=None,
        help=(
            "Insert priority for new actor transitions: 'max' (buffer max priority, no "
            "actor-side TD forwards) or 'td' (legacy local TD-error priorities) "
            "(default: config apex.actor_priority_mode)"
        ),
    )
    parser.add_argument(
        "--opponent-pool-dir",
        type=str,
        default=None,
        help=(
            "Directory of frozen opponent .pth checkpoints for actor pool self-play. "
            "Each episode, non-hero actor snake slots run a uniformly sampled frozen "
            "pool checkpoint with probability 1 - pool_latest_fraction "
            "(default: config apex.opponent_pool_dir; unset = pure mirror self-play)"
        ),
    )
    parser.add_argument(
        "--pool-latest-fraction",
        type=float,
        default=None,
        help=(
            "Per-episode probability that each non-hero actor snake slot runs the latest "
            "policy instead of a frozen pool checkpoint, in [0, 1] "
            "(default: config apex.pool_latest_fraction, 0.8)"
        ),
    )
    parser.add_argument(
        "--min-actor-terminal-fraction",
        type=float,
        default=None,
        help=(
            "Optional final actor replay-quality gate. Fail the run when terminal actor "
            "replay is below this fraction, e.g. 0.005 for 0.5%%."
        ),
    )
    parser.add_argument(
        "--override-reward-contract",
        action="store_true",
        help=(
            "Allow resuming a checkpoint whose reward economics differ from the current "
            "config (logs a loud warning instead of aborting). For deliberate reward "
            "migrations only — e.g. the P1 3-arm reward-v2 fine-tune of champion_a5. "
            "Non-reward mismatches (shapes, gamma, n_step, board scale) still abort."
        ),
    )

    args = parser.parse_args()

    if args.total_steps is not None and args.max_learner_updates is not None:
        requested_total_steps = args.total_steps
    elif args.max_learner_updates is not None:
        requested_total_steps = args.max_learner_updates
    else:
        requested_total_steps = args.total_steps if args.total_steps is not None else 1_000_000

    train_apex(
        num_actors=args.num_actors,
        total_steps=requested_total_steps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        n_step=args.n_step,
        checkpoint_dir=args.checkpoint_dir,
        resume_checkpoint=args.resume,
        config_path=args.config,
        log_dir=args.log_dir,
        checkpoint_interval=args.save_interval,
        log_interval=args.log_interval,
        actor_env_num_snakes=args.actor_env_num_snakes,
        actor_board_scale=args.actor_board_scale,
        actor_food_multiplier=args.actor_food_multiplier,
        actor_boost_exploration_rate=args.actor_boost_exploration_rate,
        actor_danger_exploration_rate=args.actor_danger_exploration_rate,
        actor_priority_mode=args.actor_priority_mode,
        opponent_pool_dir=args.opponent_pool_dir,
        pool_latest_fraction=args.pool_latest_fraction,
        min_actor_terminal_fraction=args.min_actor_terminal_fraction,
        override_reward_contract=args.override_reward_contract,
        max_learner_updates=args.max_learner_updates,
        max_environment_transitions=args.max_environment_transitions,
        max_wall_time_seconds=args.max_wall_time_seconds,
        heartbeat_timeout_seconds=args.heartbeat_timeout_seconds,
        seed=args.seed,
        resume_mode=args.resume_mode,
    )


if __name__ == "__main__":
    main()
