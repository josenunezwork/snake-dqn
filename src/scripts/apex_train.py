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
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, TypeVar

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import load_and_initialize_config  # noqa: E402
from src.core.game_config import GameConfig  # noqa: E402
from src.core.reward_contract import current_reward_contract  # noqa: E402
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
        "use_gru": False,
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
) -> None:
    """Reject resume checkpoints with known-incompatible Apex training contracts."""
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
    )


def load_validated_apex_resume_checkpoint(
    resume_checkpoint: Optional[str],
    expected_config: dict,
    map_location: Any = None,
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
        validate_apex_resume_checkpoint_config(
            checkpoint,
            expected_config,
            checkpoint_path=str(checkpoint_path),
        )
        for key in ("dqn_state_dict", "target_dqn_state_dict", "optimizer_state_dict"):
            if key not in checkpoint:
                raise KeyError(key)
    except (OSError, RuntimeError, EOFError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to load resume checkpoint {checkpoint_path}: {e}") from e

    return checkpoint


def broadcast_weights(
    weights: Dict[str, torch.Tensor],
    weight_queues: list,
) -> None:
    """Broadcast learner weights to all actor weight queues.

    Clears stale weights before pushing new ones to ensure actors
    always receive the most recent parameters.

    Args:
        weights: State dict from learner (already on CPU)
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
            q.put_nowait(weights)
        except queue.Full:
            pass
        except Exception:
            pass


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
    min_actor_terminal_fraction: Optional[float] = None,
    stagger_delay: float = 0.5,
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
        min_actor_terminal_fraction: Optional final actor replay-quality gate.
            Fails after cleanup when terminal actor replay is below this fraction.
        stagger_delay: Seconds between starting each actor
    """
    print("=" * 70)
    print("APE-X DQN DISTRIBUTED TRAINING")
    print("=" * 70)

    use_config = config_path is not None
    if config_path:
        load_and_initialize_config(config_path)
        print(f"Loaded config: {config_path}")

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
        stop_actors,
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
    print(f"  Gamma:           {gamma}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Target update:   {target_update_freq}")
    print()

    resume_checkpoint_state = load_validated_apex_resume_checkpoint(
        resume_checkpoint,
        apex_checkpoint_config,
        map_location=device,
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
    )
    buffer_process.start()
    print("  BufferProcess running.")

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

    # Sync shared network weights from learner
    shared_network.load_state_dict({k: v.cpu() for k, v in learner.dqn.state_dict().items()})

    # ── Resume from checkpoint ────────────────────────────────────────
    start_step = 0
    if resume_checkpoint_state is not None:
        print(f"Resuming from: {resume_checkpoint}")
        learner.load_state_dict(resume_checkpoint_state)
        start_step = resume_checkpoint_state.get("step_count", 0)
        # Re-sync shared network
        shared_network.load_state_dict({k: v.cpu() for k, v in learner.dqn.state_dict().items()})
        print(f"  Resumed at step {start_step:,}")
        for line in format_apex_checkpoint_provenance(resume_checkpoint_state):
            print(f"  {line}")

    # ── Spawn actors ──────────────────────────────────────────────────
    print(f"\nSpawning {num_actors} actors...")
    actors = spawn_actors(
        num_actors=num_actors,
        shared_network=shared_network,
        buffer_process=buffer_process,
        weight_queues=weight_queues,
        stats_queue=stats_queue,
        stop_event=stop_event,
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
    print("Starting actors...")
    start_actors(actors, stagger_delay=stagger_delay)
    print(f"  All {num_actors} actors started.\n")

    # ── Main training loop ────────────────────────────────────────────
    print("Waiting for buffer to fill...")
    start_time = time.time()
    episode_rewards: deque = deque(maxlen=100)
    actor_stats_by_id = {}
    last_log_step = start_step
    last_save_step = start_step
    last_reported_buffer_rejection_count = 0
    last_reported_buffer_priority_rejection_count = 0
    last_reported_learner_sample_error_count = 0
    step = start_step

    try:
        while step < total_steps and not shutdown_requested[0]:
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
                weights = learner.get_weights()
                broadcast_weights(weights, weight_queues)

            # ── Collect actor stats ───────────────────────────────────
            update_latest_actor_stats(
                actor_stats_by_id,
                collect_actor_stats(stats_queue),
                episode_rewards=episode_rewards,
            )

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
                state["avg_reward"] = _mean_or_zero(episode_rewards)
                attach_replay_health_metadata(
                    state,
                    actor_replay=summarize_actor_replay_coverage(list(actor_stats_by_id.values())),
                    buffer_replay_health=collect_buffer_replay_health(learner.buffer_client),
                    actor_replay_gates=actor_replay_gates,
                )
                torch.save(state, ckpt_path)

                # Also save as best_apex.pth
                best_path = os.path.join(checkpoint_dir, "best_apex.pth")
                torch.save(state, best_path)

                print(f"  Checkpoint saved: {ckpt_path}")
                last_save_step = step

    except Exception as e:
        print(f"\n[Coordinator] Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # ── Cleanup ───────────────────────────────────────────────────
        print("\n[Coordinator] Shutting down...")

        # Signal actors to stop
        stop_event.set()
        stop_actors(actors, timeout=5.0)
        print("  Actors stopped.")
        update_latest_actor_stats(
            actor_stats_by_id,
            collect_actor_stats(stats_queue),
            episode_rewards=episode_rewards,
        )
        final_actor_replay = summarize_actor_replay_coverage(list(actor_stats_by_id.values()))

        # Save final checkpoint
        final_path = os.path.join(checkpoint_dir, "apex_final.pth")
        final_state = learner.get_state_dict()
        final_state["apex_config"] = dict(apex_checkpoint_config)
        final_state["avg_reward"] = _mean_or_zero(episode_rewards)
        final_buffer_replay_health = collect_buffer_replay_health(learner.buffer_client)
        final_actor_replay_warnings = format_actor_replay_warnings(final_actor_replay)
        attach_replay_health_metadata(
            final_state,
            actor_replay=final_actor_replay,
            buffer_replay_health=final_buffer_replay_health,
            actor_replay_gates=actor_replay_gates,
        )
        torch.save(final_state, final_path)
        print(f"  Final checkpoint: {final_path}")

        # Shutdown buffer process
        buffer_process.shutdown(timeout=5.0)
        print("  BufferProcess stopped.")

        # Cleanup learner
        learner.cleanup()

        # ── Training summary ──────────────────────────────────────────
        elapsed = time.time() - start_time
        final_step = learner.step_count
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"  Total Steps:      {final_step:,}")
        print(f"  Training Time:    {elapsed / 3600:.2f} hours")
        print(f"  Steps/Second:     {final_step / max(elapsed, 1):.0f}")
        avg_r = _mean_or_zero(episode_rewards)
        print(f"  Final Avg Reward: {avg_r:.2f}")
        if final_actor_replay["sent_experience_count"]:
            print(format_actor_replay_summary(final_actor_replay))
            if final_actor_replay_warnings:
                print("  Actor replay warnings:")
                for warning in final_actor_replay_warnings:
                    print(warning)
        print(f"  Final Checkpoint: {final_path}")
        print("=" * 70)
        validate_actor_replay_quality_gates(
            final_actor_replay,
            min_terminal_fraction=actor_replay_gates["min_actor_terminal_fraction"],
        )


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
        default=1_000_000,
        help="Total learner update steps (default: 1,000,000)",
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
        "--min-actor-terminal-fraction",
        type=float,
        default=None,
        help=(
            "Optional final actor replay-quality gate. Fail the run when terminal actor "
            "replay is below this fraction, e.g. 0.005 for 0.5%%."
        ),
    )

    args = parser.parse_args()

    train_apex(
        num_actors=args.num_actors,
        total_steps=args.total_steps,
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
        min_actor_terminal_fraction=args.min_actor_terminal_fraction,
    )


if __name__ == "__main__":
    main()
