#!/usr/bin/env python3
"""Train Apex DQN from generated SQLite replay memories.

This script bridges the generated-data workflow:

1. src/scripts/generate_experiences.py writes transitions to snake_memories.db.
2. This script loads generated rows into local prioritized replay.
3. ApexPolicy.train_step() performs bounded offline gradient updates.
4. The trained checkpoint is saved in the normal Apex checkpoint format.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

# Add project root to path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import (  # noqa: E402
    apply_config_to_game_config,
    load_config,
)
from src.core.game_config import GameConfig, get_config, initialize_config  # noqa: E402
from src.core.reward_contract import current_reward_contract  # noqa: E402
from src.data.memory_db_handler import (  # noqa: E402
    REPLAY_QUALITY_GATE_ORDER,
    REPLAY_QUALITY_GATE_PRESETS,
    resolve_min_row_count,
    resolve_replay_quality_gate_values,
    validate_min_row_count,
    validate_replay_metadata_contract,
    validate_replay_quality_gates,
)
from src.training.checkpoint_contract import validate_checkpoint_contract  # noqa: E402

TARGET_ACTION_METRIC_KEYS = (
    "valid_next_action_fraction",
    "trapped_next_state_fraction",
    "exact_next_action_mask_fraction",
)
DEFAULT_OFFLINE_REPLAY_QUALITY_PRESET = "training"

if TYPE_CHECKING:
    import torch

    from src.training.apex_policy import ApexPolicy


def configure_device(device_name: str) -> torch.device:
    """Configure the project device override for offline training."""
    import torch

    from src.core.device_manager import DeviceManager

    if device_name == "auto":
        return DeviceManager.get_device()

    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    DeviceManager.override_device(requested)
    return requested


def resolve_offline_batch_size(batch_size: Optional[int] = None) -> int:
    """Resolve the batch size used by local Apex offline updates."""
    if batch_size is None:
        return GameConfig.APEX_BATCH_SIZE
    if isinstance(batch_size, bool):
        raise ValueError("batch-size must be a positive integer")
    resolved = int(batch_size)
    if resolved <= 0:
        raise ValueError("batch-size must be a positive integer")
    return resolved


def apply_offline_batch_size_override(batch_size: Optional[int] = None) -> int:
    """Apply the offline Apex batch size to the active immutable config."""
    resolved_batch_size = resolve_offline_batch_size(batch_size)

    config = get_config()
    initialize_config(
        replace(
            config,
            apex=replace(config.apex, batch_size=resolved_batch_size),
        )
    )
    return GameConfig.APEX_BATCH_SIZE


def resolve_checkpoint_path(checkpoint_path: str) -> Optional[Path]:
    """Resolve checkpoint input against cwd and configured checkpoint dirs."""
    candidate = Path(checkpoint_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    configured_candidate = Path(GameConfig.CHECKPOINT_DIR).expanduser() / checkpoint_path
    if configured_candidate.exists():
        return configured_candidate.resolve()

    saved_candidate = Path("saved_snakes") / checkpoint_path
    if saved_candidate != configured_candidate and saved_candidate.exists():
        return saved_candidate.resolve()

    return None


def resolve_output_checkpoint_dir(checkpoint_dir: Optional[str]) -> str:
    """Resolve offline output checkpoint directory from CLI or active config."""
    return checkpoint_dir if checkpoint_dir is not None else GameConfig.CHECKPOINT_DIR


def resolve_replay_load_limit(limit: Optional[int], capacity: int) -> int:
    """Resolve how many replay rows to load from SQLite into memory."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    return capacity if limit is None else min(limit, capacity)


def resolve_offline_replay_quality_gates(
    preset: str = DEFAULT_OFFLINE_REPLAY_QUALITY_PRESET,
    overrides: dict[str, object] | None = None,
) -> dict[str, float]:
    """Resolve offline replay quality gates from a preset and CLI overrides."""
    return resolve_replay_quality_gate_values(preset=preset, overrides=overrides)


def format_optional_percent(value: Optional[float], digits: int = 1) -> str:
    """Format an optional fraction as a human-readable percentage."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}%}"


def get_policy_target_action_metrics(policy: ApexPolicy) -> dict[str, float]:
    """Return target-action diagnostics from the most recent local train step."""
    raw_metrics = getattr(policy, "_last_train_metrics", None)
    if not raw_metrics:
        return {}

    metrics = {}
    for key in TARGET_ACTION_METRIC_KEYS:
        if key in raw_metrics:
            metrics[key] = float(raw_metrics[key])
    return metrics


def get_policy_offline_replay_quality(policy: ApexPolicy) -> dict:
    """Return quality diagnostics for the replay loaded into offline training."""
    replay_quality = getattr(policy, "_offline_replay_quality", None)
    return dict(replay_quality) if replay_quality else {}


def get_policy_offline_replay_gates(policy: ApexPolicy) -> dict[str, float]:
    """Return replay-quality gates applied before offline training started."""
    replay_gates = getattr(policy, "_offline_replay_gates", None)
    return dict(replay_gates) if replay_gates else {}


def get_policy_offline_replay_load(policy: ApexPolicy) -> dict:
    """Return replay loading settings used for offline training."""
    replay_load = getattr(policy, "_offline_replay_load", None)
    return dict(replay_load) if replay_load else {}


def get_policy_offline_replay_metadata(policy: ApexPolicy) -> dict:
    """Return durable source replay metadata captured before offline training."""
    replay_metadata = getattr(policy, "_offline_replay_metadata", None)
    return dict(replay_metadata) if replay_metadata else {}


def get_policy_offline_replay_warnings(policy: ApexPolicy) -> list[str]:
    """Return replay-quality warnings observed before offline training started."""
    replay_warnings = getattr(policy, "_offline_replay_warnings", None)
    return list(replay_warnings) if replay_warnings else []


def get_policy_min_replay_size(policy: ApexPolicy) -> Optional[int]:
    """Return the policy warmup requirement when the policy exposes one."""
    min_replay_size_fn = getattr(policy, "_min_replay_size", None)
    if not callable(min_replay_size_fn):
        return None
    min_replay_size = int(min_replay_size_fn())
    if min_replay_size <= 0:
        raise RuntimeError("Policy warmup requirement must be positive")
    return min_replay_size


def validate_replay_warmup_size(policy: ApexPolicy, loaded_rows: int) -> Optional[int]:
    """Fail before mutating memory when loaded replay cannot start training."""
    min_replay_size = get_policy_min_replay_size(policy)
    if min_replay_size is None:
        return None
    if loaded_rows < min_replay_size:
        raise RuntimeError(
            f"Replay database loaded {loaded_rows:,} rows but policy needs at least "
            f"{min_replay_size:,} before offline training can start"
        )
    return min_replay_size


def format_target_action_metrics(metrics: dict[str, float]) -> str:
    """Format target-action diagnostics for offline training progress output."""
    return (
        "Target actions: "
        f"valid={format_optional_percent(metrics.get('valid_next_action_fraction'))}, "
        f"trapped={format_optional_percent(metrics.get('trapped_next_state_fraction'))}, "
        f"exact_masks={format_optional_percent(metrics.get('exact_next_action_mask_fraction'))}"
    )


def format_checkpoint_replay_provenance(checkpoint: dict) -> list[str]:
    """Return compact replay provenance lines for an offline-trained checkpoint."""
    apex_config = checkpoint.get("apex_config") or {}
    replay_metadata = checkpoint.get("offline_replay_metadata") or {}
    replay_quality = checkpoint.get("offline_replay_quality") or {}
    replay_gates = checkpoint.get("offline_replay_gates") or {}
    replay_load = checkpoint.get("offline_replay_load") or {}
    replay_warnings = checkpoint.get("offline_replay_warnings") or []
    train_metrics = checkpoint.get("offline_train_metrics") or {}
    if (
        not apex_config
        and not replay_metadata
        and not replay_quality
        and not replay_gates
        and not replay_load
        and not replay_warnings
        and not train_metrics
    ):
        return []

    lines = []
    if replay_load or replay_quality:
        row_count = replay_load.get("loaded_rows", replay_quality.get("count"))
        row_count_text = f"{int(row_count):,}" if row_count is not None else "n/a"
        min_replay_size = replay_load.get("min_replay_size")
        warmup_text = f" | warmup={int(min_replay_size):,}" if min_replay_size is not None else ""
        batch_size = replay_load.get("batch_size")
        batch_text = f" | batch={int(batch_size):,}" if batch_size is not None else ""
        min_row_count = replay_load.get("min_row_count")
        row_gate_text = f" | row_gate={int(min_row_count):,}" if min_row_count is not None else ""
        replay_order = replay_load.get("replay_order", "unknown")
        source_db = replay_load.get("db_path", checkpoint.get("source_db", "unknown"))
        lines.append(
            "Checkpoint replay: "
            f"rows={row_count_text}{warmup_text}{batch_text}{row_gate_text} | "
            f"order={replay_order} | source={source_db}"
        )

    if replay_metadata:
        lines.append(
            "Checkpoint generated replay: "
            f"mode={replay_metadata.get('generation.mode')} | "
            f"episodes={replay_metadata.get('generation.episodes')} | "
            f"frame_limit={replay_metadata.get('generation.frame_limit')} | "
            f"snakes={replay_metadata.get('generation.num_snakes')} | "
            f"board={replay_metadata.get('generation.board_width')}x"
            f"{replay_metadata.get('generation.board_height')} | "
            f"state={replay_metadata.get('generation.state_size')} | "
            f"actions={replay_metadata.get('generation.action_size')} | "
            f"gamma={replay_metadata.get('generation.gamma')} | "
            f"n_step={replay_metadata.get('generation.apex_n_step')}"
        )

    if apex_config:
        batch_size = apex_config.get("batch_size")
        buffer_size = apex_config.get("buffer_size")
        min_replay_size = apex_config.get("min_replay_size")
        n_step = apex_config.get("n_step")
        gamma = apex_config.get("gamma")
        target_update_freq = apex_config.get("target_update_freq")
        priority_alpha = apex_config.get("priority_alpha")
        priority_beta_start = apex_config.get("priority_beta_start")
        priority_beta_current = apex_config.get("priority_beta_current")
        priority_beta_end = apex_config.get("priority_beta_end")
        priority_epsilon = apex_config.get("priority_epsilon")
        reward_death = apex_config.get("reward_death")
        reward_food_base = apex_config.get("reward_food_base")
        lines.append(
            "Checkpoint Apex config: "
            f"batch={batch_size} | buffer={buffer_size} | warmup={min_replay_size} | "
            f"n_step={n_step} | gamma={gamma} | target_sync={target_update_freq} | "
            f"reward death={reward_death}, food={reward_food_base} | "
            f"PER alpha={priority_alpha}, beta={priority_beta_start}->{priority_beta_end} "
            f"(current={priority_beta_current}), eps={priority_epsilon}"
        )

    if replay_quality:
        active_actions = replay_quality.get("active_action_count")
        dominant_action = replay_quality.get("dominant_action")
        dominant_fraction = replay_quality.get("dominant_action_fraction")
        exact_mask_fraction = replay_quality.get("nonterminal_mask_fraction")
        terminal_fraction = replay_quality.get("terminal_fraction")
        lines.append(
            "Checkpoint replay quality: "
            f"actions={active_actions}/{GameConfig.OUTPUT_SIZE} | "
            f"dominant={dominant_action} "
            f"({format_optional_percent(dominant_fraction)}) | "
            f"exact_masks={format_optional_percent(exact_mask_fraction)} | "
            f"terminal={format_optional_percent(terminal_fraction)}"
        )

    if replay_gates:
        terminal_gate = format_optional_percent(replay_gates.get("min_terminal_fraction"))
        immediate_terminal_gate = format_optional_percent(
            replay_gates.get("min_immediate_terminal_fraction")
        )
        exact_mask_gate = format_optional_percent(replay_gates.get("min_exact_mask_fraction"))
        boost_mask_gate = format_optional_percent(replay_gates.get("min_boost_mask_fraction"))
        action_coverage_gate = format_optional_percent(
            replay_gates.get("min_action_coverage_fraction")
        )
        positive_reward_gate = format_optional_percent(
            replay_gates.get("min_positive_reward_fraction")
        )
        negative_reward_gate = format_optional_percent(
            replay_gates.get("min_negative_reward_fraction")
        )
        multistep_gate = format_optional_percent(replay_gates.get("min_multistep_fraction"))
        dominant_gate = format_optional_percent(replay_gates.get("max_dominant_action_fraction"))
        invalid_current_action_gate = format_optional_percent(
            replay_gates.get("max_invalid_current_action_fraction")
        )
        trapped_next_gate = format_optional_percent(
            replay_gates.get("max_nonterminal_trapped_next_fraction")
        )
        mask_state_mismatch_gate = format_optional_percent(
            replay_gates.get("max_exact_mask_state_mismatch_fraction")
        )
        malformed_state_gate = format_optional_percent(
            replay_gates.get("max_malformed_state_feature_fraction")
        )
        lines.append(
            "Checkpoint replay gates: "
            f"terminal>={terminal_gate}, "
            f"immediate_terminal>={immediate_terminal_gate}, "
            f"exact_masks>={exact_mask_gate}, "
            f"boost_masks>={boost_mask_gate}, "
            f"action_coverage>={action_coverage_gate}, "
            f"positive_reward>={positive_reward_gate}, "
            f"negative_reward>={negative_reward_gate}, "
            f"multistep>={multistep_gate}, "
            f"dominant<={dominant_gate}, "
            f"invalid_current_action<={invalid_current_action_gate}, "
            f"trapped_next<={trapped_next_gate}, "
            f"mask_state_mismatch<={mask_state_mismatch_gate}, "
            f"malformed_state<={malformed_state_gate}"
        )

    if replay_warnings:
        lines.append("Checkpoint replay warnings:")
        lines.extend(f"  - {str(warning).strip()}" for warning in replay_warnings)

    if train_metrics:
        lines.append("Checkpoint " + format_target_action_metrics(train_metrics))

    return lines


def _validate_state_values(row_idx: int, field_name: str, values: Sequence[float]) -> None:
    """Validate a replay observation before constructing learner tensors."""
    if len(values) != GameConfig.INPUT_SIZE:
        raise RuntimeError(
            f"Replay row {row_idx} {field_name} size does not match "
            f"config INPUT_SIZE={GameConfig.INPUT_SIZE}"
        )

    for value_idx, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Replay row {row_idx} {field_name}[{value_idx}] must be numeric"
            ) from exc
        if not math.isfinite(number):
            raise RuntimeError(f"Replay row {row_idx} {field_name}[{value_idx}] must be finite")


def _validate_integral_range(
    row_idx: int,
    field_name: str,
    value: object,
    minimum: int,
    maximum_exclusive: Optional[int] = None,
) -> int:
    """Validate integral replay metadata without silently truncating floats."""
    if isinstance(value, (bool, str, bytes, bytearray, memoryview)):
        raise RuntimeError(f"Replay row {row_idx} {field_name} must be an integer")
    try:
        integer = int(value)
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Replay row {row_idx} {field_name} must be an integer") from exc
    if not math.isfinite(number) or number != integer:
        raise RuntimeError(f"Replay row {row_idx} {field_name} must be an integer")
    if integer < minimum:
        raise RuntimeError(f"Replay row {row_idx} {field_name} must be at least {minimum}")
    if maximum_exclusive is not None and integer >= maximum_exclusive:
        raise RuntimeError(
            f"Replay row {row_idx} {field_name} {integer} is outside "
            f"[{minimum}, {maximum_exclusive})"
        )
    return integer


def _validate_finite_float(
    row_idx: int,
    field_name: str,
    value: object,
    *,
    positive: bool = False,
) -> float:
    """Validate scalar replay values that feed TD targets or sampling weights."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Replay row {row_idx} {field_name} must be finite") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"Replay row {row_idx} {field_name} must be finite")
    if positive and number <= 0.0:
        raise RuntimeError(f"Replay row {row_idx} {field_name} must be positive")
    return number


def _validate_done_flag(row_idx: int, value: object) -> bool:
    """Validate replay terminal metadata without accepting broad truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise RuntimeError(f"Replay row {row_idx} done must be bool/0/1")
    try:
        integer = int(value)
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Replay row {row_idx} done must be bool/0/1") from exc
    if not math.isfinite(number) or number != integer or integer not in (0, 1):
        raise RuntimeError(f"Replay row {row_idx} done must be bool/0/1")
    return bool(integer)


def _validate_action_mask(
    row_idx: int,
    mask: Optional[Sequence[bool]],
) -> None:
    """Validate optional exact action masks loaded from generated replay."""
    if mask is None:
        return
    if len(mask) != GameConfig.OUTPUT_SIZE:
        raise RuntimeError(
            f"Replay row {row_idx} next_action_mask size does not match "
            f"OUTPUT_SIZE={GameConfig.OUTPUT_SIZE}"
        )

    for action_idx, value in enumerate(mask):
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            raise RuntimeError(
                f"Replay row {row_idx} next_action_mask[{action_idx}] must be bool/0/1"
            )
        if isinstance(value, bool):
            continue
        else:
            try:
                numeric = float(value)
                integer = int(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Replay row {row_idx} next_action_mask[{action_idx}] must be bool/0/1"
                ) from exc
            if not math.isfinite(numeric) or numeric != integer or integer not in (0, 1):
                raise RuntimeError(
                    f"Replay row {row_idx} next_action_mask[{action_idx}] must be bool/0/1"
                )


def validate_loaded_replay_rows(
    states: Sequence[Sequence[float]],
    actions: Sequence[int],
    rewards: Sequence[float],
    next_states: Sequence[Sequence[float]],
    dones: Sequence[bool],
    priorities: Sequence[float],
    bootstrap_steps: Sequence[int],
    db_path: str,
    next_action_masks: Optional[Sequence[Optional[Sequence[bool]]]] = None,
    snake_ids: Optional[Sequence[int]] = None,
) -> None:
    """Validate replay rows before loading them into the learner buffer."""
    row_count = len(states)
    if row_count == 0:
        raise RuntimeError(f"No Apex replay rows found in {db_path}")

    field_lengths = {
        "actions": len(actions),
        "rewards": len(rewards),
        "next_states": len(next_states),
        "dones": len(dones),
        "priorities": len(priorities),
        "bootstrap_steps": len(bootstrap_steps),
    }
    if next_action_masks is not None:
        field_lengths["next_action_masks"] = len(next_action_masks)
    if snake_ids is not None:
        field_lengths["snake_ids"] = len(snake_ids)
    mismatched = {name: count for name, count in field_lengths.items() if count != row_count}
    if mismatched:
        details = ", ".join(f"{name}={count}" for name, count in sorted(mismatched.items()))
        raise RuntimeError(f"Replay row fields are misaligned: states={row_count}, {details}")

    for row_idx, (state, next_state, action, reward, done, priority, steps) in enumerate(
        zip(states, next_states, actions, rewards, dones, priorities, bootstrap_steps)
    ):
        _validate_state_values(row_idx, "state", state)
        _validate_state_values(row_idx, "next_state", next_state)
        _validate_integral_range(row_idx, "action", action, 0, GameConfig.OUTPUT_SIZE)
        _validate_finite_float(row_idx, "reward", reward)
        _validate_done_flag(row_idx, done)
        _validate_finite_float(row_idx, "priority", priority, positive=True)
        _validate_integral_range(row_idx, "bootstrap_steps", steps, 1)
        if next_action_masks is not None:
            _validate_action_mask(row_idx, next_action_masks[row_idx])
        if snake_ids is not None:
            _validate_integral_range(row_idx, "snake_id", snake_ids[row_idx], 0)


def create_policy() -> ApexPolicy:
    """Create the feedforward Apex policy used by generated transition replay."""
    from src.training.apex_policy import ApexPolicy

    if GameConfig.USE_GRU:
        raise RuntimeError(
            "SQLite offline training expects flat transition replay, but use_gru is enabled. "
            "Use a feedforward config for generated replay or add sequence export first."
        )

    return ApexPolicy(
        input_size=GameConfig.INPUT_SIZE,
        hidden_size=GameConfig.HIDDEN_SIZE,
        output_size=GameConfig.OUTPUT_SIZE,
        use_gru=False,
        training=True,
    )


def validate_offline_resume_checkpoint_config(
    checkpoint: dict,
    policy: ApexPolicy,
    checkpoint_path: str = "checkpoint",
) -> None:
    """Reject offline resume checkpoints with incompatible target semantics."""
    validate_checkpoint_contract(
        checkpoint,
        {
            "input_size": int(getattr(policy, "input_size", GameConfig.INPUT_SIZE)),
            "hidden_size": int(getattr(policy, "hidden_size", GameConfig.HIDDEN_SIZE)),
            "output_size": int(getattr(policy, "output_size", GameConfig.OUTPUT_SIZE)),
            "n_step": int(getattr(policy, "n_step", GameConfig.APEX_N_STEP)),
            "gamma": float(getattr(policy, "gamma", GameConfig.APEX_GAMMA)),
            "reward_contract": current_reward_contract(),
            "reward_death": float(GameConfig.REWARD_DEATH),
            "reward_food_base": float(GameConfig.REWARD_FOOD_BASE),
            "use_gru": bool(getattr(policy, "use_gru", False)),
        },
        checkpoint_path=checkpoint_path,
        float_keys=("gamma", "reward_death", "reward_food_base"),
        mapping_keys=("reward_contract",),
        required_keys=("reward_contract", "reward_death", "reward_food_base"),
    )


def load_checkpoint(policy: ApexPolicy, checkpoint_path: Optional[str]) -> bool:
    """Optionally resume policy weights and optimizer state."""
    import torch

    if not checkpoint_path:
        return False

    resolved_path = resolve_checkpoint_path(checkpoint_path)
    if resolved_path is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(resolved_path, map_location=policy.device, weights_only=False)
    validate_offline_resume_checkpoint_config(
        checkpoint,
        policy,
        checkpoint_path=str(resolved_path),
    )
    policy.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {resolved_path}")
    for line in format_checkpoint_replay_provenance(checkpoint):
        print(line)
    return True


def load_replay_database(
    policy: ApexPolicy,
    db_path: str,
    limit: Optional[int],
    replay_order: str = "id_uniform",
    min_terminal_fraction: float = 0.0,
    min_immediate_terminal_fraction: float = 0.0,
    min_exact_mask_fraction: float = 0.0,
    min_boost_mask_fraction: float = 0.0,
    min_action_coverage_fraction: float = 0.0,
    min_positive_reward_fraction: float = 0.0,
    min_negative_reward_fraction: float = 0.0,
    min_multistep_fraction: float = 0.0,
    max_dominant_action_fraction: float = 1.0,
    max_invalid_current_action_fraction: float = 1.0,
    max_nonterminal_trapped_next_fraction: float = 1.0,
    max_exact_mask_state_mismatch_fraction: float = 1.0,
    max_malformed_state_feature_fraction: float = 1.0,
    min_row_count: int = 0,
) -> int:
    """Load generated replay rows into the policy's prioritized replay buffer."""
    from src.data.memory_db_handler import (
        MemoryDBHandler,
        build_replay_quality_stats,
        format_replay_quality_stats,
        format_replay_quality_warnings,
    )

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Replay database not found: {db_path}")
    if policy.memory is None:
        raise RuntimeError("Policy has no replay buffer")

    capacity = getattr(policy.memory, "capacity", GameConfig.MEMORY_SIZE)
    effective_limit = resolve_replay_load_limit(limit, capacity)

    db_handler = MemoryDBHandler(db_name=db_path)
    try:
        replay_metadata = db_handler.get_metadata()
        validate_replay_metadata_contract(
            replay_metadata,
            db_path,
            policy_type="apex",
            expected_state_size=GameConfig.INPUT_SIZE,
            expected_action_size=GameConfig.OUTPUT_SIZE,
            expected_gamma=GameConfig.APEX_GAMMA,
            expected_n_step=GameConfig.APEX_N_STEP,
            expected_reward_contract=current_reward_contract(),
            expected_reward_death=GameConfig.REWARD_DEATH,
            expected_reward_food_base=GameConfig.REWARD_FOOD_BASE,
            state_size_name="INPUT_SIZE",
            action_size_name="OUTPUT_SIZE",
            gamma_name="APEX_GAMMA",
            n_step_name="APEX_N_STEP",
            reward_death_name="REWARD_DEATH",
            reward_food_base_name="REWARD_FOOD_BASE",
        )
        loaded_rows = db_handler.load_memories_for_policy(
            policy_type="apex",
            limit=effective_limit,
            order_by=replay_order,
            include_action_masks=True,
            include_snake_ids=True,
        )
    finally:
        db_handler.close()
    if len(loaded_rows) == 9:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            priorities,
            bootstrap_steps,
            next_action_masks,
            snake_ids,
        ) = loaded_rows
    elif len(loaded_rows) == 8:
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            priorities,
            bootstrap_steps,
            next_action_masks,
        ) = loaded_rows
        snake_ids = None
    else:
        states, actions, rewards, next_states, dones, priorities, bootstrap_steps = loaded_rows
        next_action_masks = None
        snake_ids = None

    replay_quality = build_replay_quality_stats(
        actions,
        rewards,
        dones,
        priorities,
        bootstrap_steps,
        next_action_masks,
        states=states,
        next_states=next_states,
        snake_ids=snake_ids,
    )

    validate_loaded_replay_rows(
        states,
        actions,
        rewards,
        next_states,
        dones,
        priorities,
        bootstrap_steps,
        db_path,
        next_action_masks=next_action_masks,
        snake_ids=snake_ids,
    )

    print("\nReplay quality:")
    for line in format_replay_quality_stats(replay_quality):
        print(line)
    warnings = format_replay_quality_warnings(replay_quality)
    if warnings:
        print("Replay quality warnings:")
        for line in warnings:
            print(line)

    min_row_count = resolve_min_row_count(min_row_count)
    validate_min_row_count(replay_quality, min_row_count=min_row_count, context="Loaded replay")
    validate_replay_quality_gates(
        replay_quality,
        min_terminal_fraction=min_terminal_fraction,
        min_immediate_terminal_fraction=min_immediate_terminal_fraction,
        min_exact_mask_fraction=min_exact_mask_fraction,
        min_boost_mask_fraction=min_boost_mask_fraction,
        min_action_coverage_fraction=min_action_coverage_fraction,
        min_positive_reward_fraction=min_positive_reward_fraction,
        min_negative_reward_fraction=min_negative_reward_fraction,
        min_multistep_fraction=min_multistep_fraction,
        max_dominant_action_fraction=max_dominant_action_fraction,
        max_invalid_current_action_fraction=max_invalid_current_action_fraction,
        max_nonterminal_trapped_next_fraction=max_nonterminal_trapped_next_fraction,
        max_exact_mask_state_mismatch_fraction=max_exact_mask_state_mismatch_fraction,
        max_malformed_state_feature_fraction=max_malformed_state_feature_fraction,
        context="Loaded replay",
    )
    min_replay_size = validate_replay_warmup_size(policy, len(states))

    policy.memory.clear()
    masks_to_load = (
        next_action_masks
        if next_action_masks and any(mask is not None for mask in next_action_masks)
        else None
    )
    policy.memory.add_bulk(
        states,
        actions,
        rewards,
        next_states,
        dones,
        priorities,
        bootstrap_steps=bootstrap_steps,
        next_action_masks=masks_to_load,
        stream_ids=snake_ids,
    )
    policy._offline_replay_quality = dict(replay_quality)
    policy._offline_replay_warnings = list(warnings)
    policy._offline_replay_metadata = dict(replay_metadata)
    policy._offline_replay_gates = {
        "min_terminal_fraction": float(min_terminal_fraction),
        "min_immediate_terminal_fraction": float(min_immediate_terminal_fraction),
        "min_exact_mask_fraction": float(min_exact_mask_fraction),
        "min_boost_mask_fraction": float(min_boost_mask_fraction),
        "min_action_coverage_fraction": float(min_action_coverage_fraction),
        "min_positive_reward_fraction": float(min_positive_reward_fraction),
        "min_negative_reward_fraction": float(min_negative_reward_fraction),
        "min_multistep_fraction": float(min_multistep_fraction),
        "max_dominant_action_fraction": float(max_dominant_action_fraction),
        "max_invalid_current_action_fraction": float(max_invalid_current_action_fraction),
        "max_nonterminal_trapped_next_fraction": float(max_nonterminal_trapped_next_fraction),
        "max_exact_mask_state_mismatch_fraction": float(max_exact_mask_state_mismatch_fraction),
        "max_malformed_state_feature_fraction": float(max_malformed_state_feature_fraction),
    }
    policy._offline_replay_load = {
        "db_path": db_path,
        "effective_limit": int(effective_limit),
        "loaded_rows": len(states),
        "replay_order": replay_order,
        "requested_limit": limit,
        "batch_size": GameConfig.APEX_BATCH_SIZE,
    }
    if min_row_count > 0:
        policy._offline_replay_load["min_row_count"] = min_row_count
    if min_replay_size is not None:
        policy._offline_replay_load["min_replay_size"] = min_replay_size
    return len(states)


def save_policy_checkpoint(
    policy: ApexPolicy,
    checkpoint_dir: str,
    filename: str,
    db_path: str,
    iterations_done: int,
    avg_loss: Optional[float],
    elapsed_seconds: float,
) -> str:
    """Save an offline-trained Apex checkpoint."""
    from src.model.checkpoint_manager import CheckpointManager

    checkpoint = policy.get_state_dict()
    checkpoint.update(
        {
            "avg_loss": avg_loss if avg_loss is not None else 0.0,
            "iteration": policy.update_counter,
            "offline_iterations": iterations_done,
            "offline_replay_gates": get_policy_offline_replay_gates(policy),
            "offline_replay_load": get_policy_offline_replay_load(policy),
            "offline_replay_metadata": get_policy_offline_replay_metadata(policy),
            "offline_replay_quality": get_policy_offline_replay_quality(policy),
            "offline_replay_warnings": get_policy_offline_replay_warnings(policy),
            "offline_train_metrics": get_policy_target_action_metrics(policy),
            "replay_size": len(policy.memory) if policy.memory is not None else 0,
            "source_db": db_path,
            "training_time_seconds": elapsed_seconds,
        }
    )

    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    return manager.save_checkpoint_dict(checkpoint, filename)


def train_offline(
    policy: ApexPolicy,
    iterations: int,
    log_interval: int,
    checkpoint_interval: int,
    checkpoint_dir: str,
    checkpoint_filename: str,
    db_path: str,
) -> tuple[int, Optional[float]]:
    """Run offline gradient updates from already-loaded replay."""
    replay_size = len(policy.memory) if policy.memory is not None else 0
    min_replay_size = policy._min_replay_size()
    if replay_size < min_replay_size:
        raise RuntimeError(
            f"Replay has {replay_size:,} rows but needs at least {min_replay_size:,} "
            "before training can start"
        )

    losses = deque(maxlen=100)
    last_loss = None
    start_time = time.time()

    for local_iter in range(1, iterations + 1):
        loss, epsilon = policy.train_step()
        if loss is None:
            raise RuntimeError("Training stopped because the replay buffer was not ready")

        last_loss = float(loss)
        train_metrics = get_policy_target_action_metrics(policy)
        losses.append(last_loss)

        if local_iter == 1 or local_iter % log_interval == 0 or local_iter == iterations:
            avg_loss = sum(losses) / len(losses)
            elapsed = time.time() - start_time
            steps_per_sec = local_iter / elapsed if elapsed > 0 else 0.0
            print(
                f"Iter {local_iter:,}/{iterations:,} | "
                f"Updates: {policy.update_counter:,} | "
                f"Loss: {last_loss:.5f} | Avg100: {avg_loss:.5f} | "
                f"Epsilon: {epsilon:.4f} | {format_target_action_metrics(train_metrics)} | "
                f"Replay: {replay_size:,} | "
                f"{steps_per_sec:.1f} updates/s"
            )

        if checkpoint_interval > 0 and local_iter % checkpoint_interval == 0:
            avg_loss = sum(losses) / len(losses)
            save_policy_checkpoint(
                policy=policy,
                checkpoint_dir=checkpoint_dir,
                filename=checkpoint_filename,
                db_path=db_path,
                iterations_done=local_iter,
                avg_loss=avg_loss,
                elapsed_seconds=time.time() - start_time,
            )

    return iterations, last_loss


def main() -> None:
    """CLI entry point for generated-replay offline training."""
    parser = argparse.ArgumentParser(
        description="Train Apex DQN from generated SQLite replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data, then train from it
  python src/scripts/generate_experiences.py --episodes 5000 --parallel
  python src/scripts/offline_train.py --iterations 20000

  # Resume a checkpoint and use a custom database
  python src/scripts/offline_train.py --db snake_memories.db --resume best_apex.pth
        """,
    )
    parser.add_argument("--db", default="snake_memories.db", help="SQLite replay database path")
    parser.add_argument("--iterations", type=int, default=20000, help="Gradient updates to run")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional offline training batch-size override. Defaults to apex.batch_size "
            "from config; use a smaller value for quick local smokes with small replay DBs."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max replay rows to load; defaults to replay capacity",
    )
    parser.add_argument(
        "--replay-order",
        choices=("id_uniform", "id", "priority"),
        default="id_uniform",
        help=(
            "Replay load order when --limit is smaller than the DB. "
            "'id_uniform' spreads capped loads across insertion order; "
            "'id' loads the oldest rows exactly; 'priority' loads highest-priority rows."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path or saved_snakes/ filename to resume",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for output checkpoints",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="best_apex.pth",
        help="Output checkpoint filename",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device override for offline training",
    )
    parser.add_argument(
        "--replay-quality-preset",
        choices=tuple(REPLAY_QUALITY_GATE_PRESETS),
        default=DEFAULT_OFFLINE_REPLAY_QUALITY_PRESET,
        help=(
            "Named replay quality gate bundle. Defaults to 'training' so offline "
            "updates require terminal, mask, action, reward, n-step, and "
            "trapped-next-state coverage. Use 'none' for warning-only diagnostics."
        ),
    )
    parser.add_argument(
        "--min-row-count",
        type=int,
        default=0,
        help="Optional absolute replay row-count gate before offline training",
    )
    parser.add_argument(
        "--min-terminal-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if loaded terminal rows are "
            "below this fraction, e.g. 0.005 for at least 0.5%% terminal replay."
        ),
    )
    parser.add_argument(
        "--min-immediate-terminal-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if one-step terminal rows "
            "are below this fraction of loaded replay rows."
        ),
    )
    parser.add_argument(
        "--min-exact-mask-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if nonterminal rows with exact "
            "next-action masks are below this fraction."
        ),
    )
    parser.add_argument(
        "--min-boost-mask-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if fewer than this fraction "
            "of nonterminal rows have exact next-action masks allowing boost."
        ),
    )
    parser.add_argument(
        "--min-action-coverage-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if fewer than this fraction "
            "of the 6 actions appear in loaded replay."
        ),
    )
    parser.add_argument(
        "--min-multistep-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if fewer than this "
            "fraction of loaded replay rows have bootstrap_steps > 1."
        ),
    )
    parser.add_argument(
        "--min-positive-reward-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if fewer than this "
            "fraction of loaded replay rows have positive rewards."
        ),
    )
    parser.add_argument(
        "--min-negative-reward-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if fewer than this "
            "fraction of loaded replay rows have negative rewards."
        ),
    )
    parser.add_argument(
        "--max-dominant-action-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if one action accounts for "
            "more than this fraction of loaded replay rows."
        ),
    )
    parser.add_argument(
        "--max-invalid-current-action-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if stored actions that are "
            "invalid under current-state danger/boost features exceed this fraction."
        ),
    )
    parser.add_argument(
        "--max-nonterminal-trapped-next-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if nonterminal next-state "
            "targets with no valid actions exceed this fraction."
        ),
    )
    parser.add_argument(
        "--max-exact-mask-state-mismatch-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if exact next-action masks "
            "disagree with next-state per-action danger features above this fraction."
        ),
    )
    parser.add_argument(
        "--max-malformed-state-feature-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Refuse to train if decoded current/next states "
            "with malformed semantic features exceed this fraction."
        ),
    )
    parser.add_argument("--log-interval", type=int, default=100, help="Print every N updates")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Save an intermediate checkpoint every N updates; 0 disables",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    resolve_offline_batch_size(args.batch_size)
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive")
    if args.save_interval < 0:
        raise ValueError("save-interval must be non-negative")
    replay_gates = resolve_offline_replay_quality_gates(
        preset=args.replay_quality_preset,
        overrides={name: getattr(args, name) for name in REPLAY_QUALITY_GATE_ORDER},
    )
    min_row_count = resolve_min_row_count(args.min_row_count)

    if args.config:
        config_obj = load_config(args.config)
        apply_config_to_game_config(config_obj)
        print(f"Loaded config: {args.config}")
    batch_size = apply_offline_batch_size_override(args.batch_size)

    device = configure_device(args.device)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    checkpoint_dir = resolve_output_checkpoint_dir(args.checkpoint_dir)

    policy = create_policy()
    load_checkpoint(policy, args.resume)

    replay_count = load_replay_database(
        policy,
        args.db,
        args.limit,
        args.replay_order,
        min_row_count=min_row_count,
        **replay_gates,
    )
    print(f"Loaded replay rows: {replay_count:,}")
    print(f"Replay load order: {args.replay_order}")
    print(f"Training warmup requirement: {policy._min_replay_size():,}")

    start_time = time.time()
    iterations_done, last_loss = train_offline(
        policy=policy,
        iterations=args.iterations,
        log_interval=args.log_interval,
        checkpoint_interval=args.save_interval,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=args.output,
        db_path=args.db,
    )
    elapsed = time.time() - start_time

    output_path = save_policy_checkpoint(
        policy=policy,
        checkpoint_dir=checkpoint_dir,
        filename=args.output,
        db_path=args.db,
        iterations_done=iterations_done,
        avg_loss=last_loss,
        elapsed_seconds=elapsed,
    )
    print(f"Saved offline-trained checkpoint: {output_path}")


if __name__ == "__main__":
    main()
