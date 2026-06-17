import argparse
import os
import shutil
import sys
import time
import traceback
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import psutil
import torch
import torch.multiprocessing as mp
from PyQt5.QtWidgets import QApplication

# Add project root to sys.path to allow imports from src when run as a script.
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config_loader import (  # noqa: E402
    apply_config_to_game_config,
    get_config_summary,
    load_config,
)
from src.core.device_manager import DeviceManager  # noqa: E402
from src.core.game_config import GameConfig, get_config, initialize_config  # noqa: E402
from src.core.reward_contract import current_reward_contract  # noqa: E402
from src.data.memory_db_handler import (  # noqa: E402
    REPLAY_QUALITY_GATE_ORDER,
    REPLAY_QUALITY_GATE_PRESETS,
    MemoryDBHandler,
    build_replay_quality_stats,
    format_replay_quality_stats,
    format_replay_quality_warnings,
)
from src.data.memory_db_handler import (  # noqa: E402
    resolve_replay_quality_fraction as _resolve_replay_quality_fraction,
)
from src.data.memory_db_handler import (  # noqa: E402
    resolve_replay_quality_gate_values,
    validate_replay_metadata_contract,
    validate_replay_quality_gates,
)
from src.game.ai_snake import AISnake  # noqa: E402
from src.game.game_state import GameState  # noqa: E402
from src.training.checkpoint_contract import validate_checkpoint_contract  # noqa: E402
from src.training.replay_buffer import restore_replay_memories  # noqa: E402
from src.training.tensorboard_logger import TensorBoardLogger  # noqa: E402
from src.ui.slitherio import SlitherIOGame  # noqa: E402


def get_optimal_env_count():
    """Determine optimal number of environments based on hardware."""
    cpu_count = psutil.cpu_count(logical=False)  # Physical CPU cores
    memory = psutil.virtual_memory()
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()

    # M1-specific optimization
    if torch.backends.mps.is_available():
        # M1 has unified memory and neural engine - can handle more
        # 16GB M1 can comfortably run 4-6 environments
        optimal_envs = min(6, memory.total // (2 * 1024**3))
        print(f"🚀 M1 Detected: Optimizing for {optimal_envs} parallel environments")
    elif gpu_available:
        # With GPU, we can use more environments as computation is offloaded
        optimal_envs = min(cpu_count * 2, memory.total // (2 * 1024**3))  # 2GB per env as safety
    else:
        # CPU only, be more conservative
        optimal_envs = min(cpu_count - 1, memory.total // (3 * 1024**3))  # 3GB per env as safety

    # Ensure at least 1 environment and no more than 8 (to prevent memory issues)
    return max(1, min(8, optimal_envs))


def split_episodes_across_envs(episodes: int, num_envs: int) -> list[int]:
    """Split requested episodes across workers without dropping remainders."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if episodes < 0:
        raise ValueError("episodes must be non-negative")

    base = episodes // num_envs
    remainder = episodes % num_envs
    return [base + (1 if env_id < remainder else 0) for env_id in range(num_envs)]


def resolve_training_batch_size(batch_size: Optional[int] = None) -> Optional[int]:
    """Resolve an optional training batch-size override."""
    if batch_size is None:
        return None
    if isinstance(batch_size, bool):
        raise ValueError("batch-size must be a positive integer")
    resolved = int(batch_size)
    if resolved <= 0:
        raise ValueError("batch-size must be a positive integer")
    return resolved


def apply_training_batch_size_override(batch_size: Optional[int] = None) -> int:
    """Apply an optional batch-size override to the active immutable config."""
    resolved_batch_size = resolve_training_batch_size(batch_size)
    if resolved_batch_size is None:
        return GameConfig.BATCH_SIZE

    config = get_config()
    initialize_config(
        replace(
            config,
            training=replace(config.training, batch_size=resolved_batch_size),
            apex=replace(config.apex, batch_size=resolved_batch_size),
        )
    )
    return GameConfig.APEX_BATCH_SIZE


def resolve_replay_quality_fraction(value=None, field_name: str = "quality_fraction") -> float:
    """Resolve a replay-quality fraction while preserving the src.main import path."""
    return _resolve_replay_quality_fraction(value, field_name)


def resolve_prefill_replay_quality_gates(
    preset: str = "none",
    overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Resolve replay-quality gates for headless replay prefill."""
    return resolve_replay_quality_gate_values(preset=preset, overrides=overrides)


def resolve_legacy_replay_quality_gates(
    min_terminal_fraction: float = 0.0,
    min_exact_mask_fraction: float = 0.0,
) -> Dict[str, float]:
    """Resolve replay-quality gates from the original two prefill knobs."""
    return resolve_prefill_replay_quality_gates(
        overrides={
            "min_terminal_fraction": min_terminal_fraction,
            "min_exact_mask_fraction": min_exact_mask_fraction,
        }
    )


def format_enabled_replay_quality_gates(replay_quality_gates: Dict[str, float]) -> list[str]:
    """Return compact descriptions for replay-quality gates that are active."""
    lines = []
    for name in REPLAY_QUALITY_GATE_ORDER:
        value = float(replay_quality_gates[name])
        if name.startswith("min_") and value > 0.0:
            label = name.removeprefix("min_").removesuffix("_fraction").replace("_", " ")
            lines.append(f"Replay prefill min {label} fraction: {value:.2%}")
        elif name.startswith("max_") and value < 1.0:
            label = name.removeprefix("max_").removesuffix("_fraction").replace("_", " ")
            lines.append(f"Replay prefill max {label} fraction: {value:.2%}")
    return lines


def get_checkpoint_dir() -> Path:
    """Return the configured checkpoint directory for headless training."""
    return Path(GameConfig.CHECKPOINT_DIR).expanduser()


def get_checkpoint_path(filename: str) -> Path:
    """Return a checkpoint path inside the configured checkpoint directory."""
    return get_checkpoint_dir() / filename


def get_env_curriculum_checkpoint_path(env_id: int) -> Path:
    """Return the curriculum state path for one headless training worker."""
    return get_checkpoint_path(f"env_{env_id}_curriculum.pth")


def get_shared_apex_policy(game_state: GameState):
    """Return the shared Apex policy backing AI snakes in a GameState."""
    policy = getattr(game_state, "_shared_policy", None)
    if policy is not None:
        return policy

    for snake in game_state.snakes:
        if isinstance(snake, AISnake):
            return snake.policy
    return None


def validate_headless_checkpoint_contract(
    checkpoint: dict,
    policy,
    checkpoint_path: str = "checkpoint",
) -> None:
    """Reject headless-training checkpoints with incompatible target semantics."""
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


def save_training_checkpoint(
    game_state: GameState,
    filename: str,
    env_id: Optional[int] = None,
    curriculum=None,
) -> Optional[Path]:
    """Save the strongest current trainable snake policy to the checkpoint directory."""
    saveable_snakes = [snake for snake in game_state.snakes if hasattr(snake, "save_state")]
    if not saveable_snakes:
        return None

    checkpoint_path = get_checkpoint_path(filename)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    current_best = max(
        saveable_snakes,
        key=lambda snake: float(getattr(snake, "total_reward", 0.0)),
    )
    current_best.save_state(str(checkpoint_path))

    if curriculum is not None and env_id is not None:
        curriculum_path = get_env_curriculum_checkpoint_path(env_id)
        curriculum_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(curriculum.get_state(), curriculum_path)

    return checkpoint_path


def save_final_training_checkpoints(
    game_state: GameState,
    env_id: int,
    curriculum=None,
    best_checkpoint_path: Optional[Path] = None,
) -> tuple[Optional[Path], Optional[Path]]:
    """Save final env checkpoint and provide a best-checkpoint fallback for short runs."""
    final_checkpoint_path = save_training_checkpoint(
        game_state,
        f"env_{env_id}_final_snake.pth",
        env_id=env_id,
        curriculum=curriculum,
    )
    if final_checkpoint_path is None:
        return None, None

    return final_checkpoint_path, best_checkpoint_path or final_checkpoint_path


def get_best_env_model_path(env_id: int, env_stats: Dict[str, Any]) -> Path:
    """Return the checkpoint path the parent process should copy for one env."""
    reported_best_path = env_stats.get("best_checkpoint")
    if reported_best_path:
        return Path(reported_best_path)
    return get_checkpoint_path(f"env_{env_id}_best_snake.pth")


def resolve_checkpoint_path(checkpoint_path: str) -> Optional[Path]:
    """Resolve user checkpoint input against cwd and configured checkpoint dirs."""
    candidate = Path(checkpoint_path).expanduser()
    if candidate.exists():
        return candidate

    configured_candidate = get_checkpoint_path(checkpoint_path)
    if configured_candidate.exists():
        return configured_candidate

    legacy_candidate = Path("saved_snakes") / checkpoint_path
    if legacy_candidate != configured_candidate and legacy_candidate.exists():
        return legacy_candidate

    return None


def load_checkpoint_into_game_state(
    game_state: GameState,
    checkpoint_path: str,
    strict_training_contract: bool = True,
) -> bool:
    """Load a checkpoint into the shared headless policy.

    Args:
        game_state: Game state whose shared Apex policy receives the checkpoint.
        checkpoint_path: Checkpoint path or filename.
        strict_training_contract: If True, reject gamma/n-step/reward contract
            mismatches before loading. Evaluation/inference callers may set this
            False because those values define TD targets, not network inference.
    """
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    if resolved_path is None:
        print(f"Checkpoint not found: {checkpoint_path}")
        return False

    policy = get_shared_apex_policy(game_state)
    if policy is None:
        print(f"No Apex policy available for checkpoint load: {resolved_path}")
        return False

    device = getattr(policy, "device", torch.device("cpu"))
    try:
        checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
        if strict_training_contract:
            validate_headless_checkpoint_contract(
                checkpoint,
                policy,
                checkpoint_path=str(resolved_path),
            )
        policy.load_state_dict(checkpoint)
    except (OSError, RuntimeError, KeyError, ValueError) as e:
        print(f"Could not load checkpoint {resolved_path}: {e}")
        return False

    memories = checkpoint.get("memories", [])
    memory = getattr(policy, "memory", None)
    if memories:
        if memory is None:
            print(f"Checkpoint replay cannot be restored without a replay buffer: {resolved_path}")
            return False
        try:
            restore_replay_memories(memory, memories, device, clear=True)
        except (ValueError, RuntimeError) as e:
            print(f"Could not restore checkpoint replay from {resolved_path}: {e}")
            return False

    for snake in game_state.snakes:
        if isinstance(snake, AISnake):
            snake.current_epsilon = policy.epsilon

    print(f"Loaded checkpoint into headless policy: {resolved_path}")
    return True


def load_replay_db_into_game_state(
    game_state: GameState,
    db_path: str,
    limit: Optional[int] = None,
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
) -> int:
    """Prefill the shared policy replay buffer from an experience SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Replay database not found: {db_path}")

    policy = get_shared_apex_policy(game_state)
    memory = getattr(policy, "memory", None) if policy is not None else None
    if memory is None:
        raise RuntimeError(f"No replay buffer available for database load: {db_path}")
    if getattr(policy, "use_gru", False):
        raise RuntimeError("SQLite replay prefill is incompatible with GRU sequence policies")

    if limit is None:
        limit = getattr(memory, "capacity", GameConfig.MEMORY_SIZE)
    if limit <= 0:
        raise ValueError("replay prefill limit must be positive")

    db_handler = MemoryDBHandler(db_name=db_path)
    try:
        validate_replay_metadata_contract(
            db_handler.get_metadata(),
            db_path,
            policy_type="apex",
            expected_state_size=GameConfig.INPUT_SIZE,
            expected_action_size=GameConfig.OUTPUT_SIZE,
            expected_gamma=float(getattr(policy, "gamma", GameConfig.APEX_GAMMA)),
            expected_n_step=int(getattr(policy, "n_step", GameConfig.APEX_N_STEP)),
            state_size_name="INPUT_SIZE",
            action_size_name="OUTPUT_SIZE",
            gamma_name="policy.gamma",
            n_step_name="policy.n_step",
        )
        loaded_rows = load_prefill_replay_rows(
            db_handler,
            limit=limit,
            replay_order=replay_order,
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
    print("\nReplay prefill quality:")
    for line in format_replay_quality_stats(replay_quality):
        print(line)
    warnings = format_replay_quality_warnings(replay_quality)
    if warnings:
        print("Replay prefill quality warnings:")
        for line in warnings:
            print(line)

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
        context="Replay prefill",
    )

    if not states:
        raise RuntimeError(f"No Apex replay rows found in {db_path}")

    masks_to_load = (
        next_action_masks
        if next_action_masks and any(mask is not None for mask in next_action_masks)
        else None
    )
    if hasattr(memory, "add_bulk"):
        memory.add_bulk(
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
    else:
        if next_action_masks is None:
            next_action_masks = [None] * len(states)
        if snake_ids is None:
            snake_ids = [None] * len(states)
        for (
            state,
            action,
            reward,
            next_state,
            done,
            priority,
            steps,
            next_action_mask,
            stream_id,
        ) in zip(
            states,
            actions,
            rewards,
            next_states,
            dones,
            priorities,
            bootstrap_steps,
            next_action_masks,
            snake_ids,
        ):
            try:
                memory.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    priority,
                    bootstrap_steps=steps,
                    next_action_mask=next_action_mask,
                    stream_id=stream_id,
                )
            except TypeError:
                try:
                    memory.add(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        priority,
                        bootstrap_steps=steps,
                    )
                except TypeError:
                    memory.add(state, action, reward, next_state, done, priority)

    print(f"Loaded {len(states):,} replay memories from {db_path}")
    return len(states)


def load_prefill_replay_rows(db_handler, limit: int, replay_order: str = "id_uniform"):
    """Load generated replay rows for headless prefill in a representative order."""
    try:
        return db_handler.load_memories_for_policy(
            policy_type="apex",
            limit=limit,
            order_by=replay_order,
            include_action_masks=True,
            include_snake_ids=True,
        )
    except TypeError:
        try:
            return db_handler.load_memories_for_policy(
                policy_type="apex",
                limit=limit,
                order_by=replay_order,
                include_action_masks=True,
            )
        except TypeError:
            return db_handler.load_memories_for_policy(
                policy_type="apex",
                limit=limit,
                order_by=replay_order,
            )


def get_policy_stats(game_state: GameState) -> Dict[str, Any]:
    """Return compact training stats for the shared policy."""
    policy = getattr(game_state, "_shared_policy", None)
    if policy is None:
        for snake in game_state.snakes:
            policy = getattr(snake, "policy", None)
            if policy is not None:
                break

    memory = getattr(policy, "memory", None) if policy is not None else None
    replay_size = 0
    if memory is not None:
        try:
            replay_size = len(memory)
        except TypeError:
            replay_size = 0

    min_replay_size = 0
    if policy is not None:
        min_replay_fn = getattr(policy, "_min_replay_size", None)
        if callable(min_replay_fn):
            min_replay_size = int(min_replay_fn())
        elif memory is not None:
            min_replay_size = int(GameConfig.BATCH_SIZE)

    losses = getattr(policy, "_losses", None) if policy is not None else None
    last_loss: Optional[float] = None
    if losses:
        last_loss = float(losses[-1])

    stats = {
        "epsilon": float(getattr(policy, "epsilon", 0.0) or 0.0),
        "last_loss": last_loss,
        "min_replay_size": min_replay_size,
        "replay_size": replay_size,
        "training": bool(getattr(policy, "training", False)),
        "updates": int(getattr(policy, "update_counter", 0) or 0),
    }
    train_metrics = getattr(policy, "_last_train_metrics", None) if policy is not None else None
    if train_metrics:
        for key in (
            "valid_next_action_fraction",
            "trapped_next_state_fraction",
            "exact_next_action_mask_fraction",
        ):
            if key in train_metrics:
                stats[key] = float(train_metrics[key])

    return stats


def get_episode_stats(game_state: GameState, episode_length: int) -> Dict[str, Any]:
    """Snapshot episode metrics before curriculum changes can rebuild the environment."""
    current_reward = getattr(
        game_state,
        "episode_current_reward",
        getattr(game_state, "episode_best_reward", 0.0),
    )
    return {
        "best_length": int(game_state.episode_best_length),
        "best_reward": float(game_state.episode_best_reward),
        "collision_counts": dict(game_state.episode_collision_counts),
        "deaths": int(game_state.episode_deaths),
        "food_eaten": int(game_state.episode_food_eaten),
        "kills": int(game_state.episode_kills),
        "length": int(episode_length),
        "reward": float(current_reward),
    }


def get_training_game_settings(curriculum=None) -> Dict[str, Any]:
    """Resolve the game settings for the current training phase."""
    if curriculum is None:
        return {
            "num_snakes": GameConfig.NUM_SNAKES,
            "food_multiplier": 1.0,
            "board_scale": 1.0,
        }
    return curriculum.get_game_settings()


def configure_eval_game_state(game_state: GameState) -> None:
    """Make a headless GameState greedy and inference-only for evaluation."""
    policies = []
    shared_policy = getattr(game_state, "_shared_policy", None)
    if shared_policy is not None:
        policies.append(shared_policy)

    for snake in getattr(game_state, "snakes", []):
        policy = getattr(snake, "policy", None)
        if policy is not None and policy not in policies:
            policies.append(policy)
        if hasattr(snake, "actor_epsilon"):
            snake.actor_epsilon = 0.0
        if hasattr(snake, "current_epsilon"):
            snake.current_epsilon = 0.0

    for policy in policies:
        if hasattr(policy, "epsilon"):
            policy.epsilon = 0.0
        if hasattr(policy, "training"):
            policy.training = False
        network = getattr(policy, "dqn", None)
        if hasattr(network, "eval"):
            network.eval()
        target_network = getattr(policy, "target_dqn", None)
        if hasattr(target_network, "eval"):
            target_network.eval()


def create_training_game_state(
    curriculum=None, shared_policy=None, eval_mode: bool = False
) -> GameState:
    """Create a headless training GameState using the active curriculum settings."""
    settings = get_training_game_settings(curriculum)
    num_snakes = int(settings["num_snakes"])

    game_state = GameState(
        headless=True,
        snake_policies=["apex"] * num_snakes,
        num_snakes=num_snakes,
        shared_policy=shared_policy,
        food_multiplier=float(settings["food_multiplier"]),
        board_scale=float(settings["board_scale"]),
    )
    if eval_mode:
        configure_eval_game_state(game_state)
    return game_state


def collect_training_worker_failures(process_entries, active_envs, return_dict) -> list[str]:
    """Collect headless training worker failures before summarizing results."""
    failures = []

    for env_id, process in process_entries:
        if process.exitcode is None:
            failures.append(f"env {env_id} did not finish")
        elif process.exitcode != 0:
            failures.append(f"env {env_id} exited with code {process.exitcode}")

    for env_id, _ in active_envs:
        result = return_dict.get(env_id)
        if result is None:
            failures.append(f"env {env_id} did not report training stats")
            continue

        error = result.get("error")
        if error:
            failures.append(f"env {env_id} failed: {error}")

    return failures


def format_collision_counts(counts: Dict[str, int]) -> str:
    """Format collision counters for compact console progress logs."""
    return ", ".join(f"{key}:{counts.get(key, 0)}" for key in ("wall", "self", "head", "body"))


def format_optional_float(value: Optional[float], digits: int = 4) -> str:
    """Format an optional float without pretending missing metrics are zero."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_optional_percent(value: Optional[float], digits: int = 1) -> str:
    """Format an optional fraction as a percentage."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}%}"


def run_learning_health_smoke(
    max_frames: int = 200,
    checkpoint_filename: Optional[str] = "health_smoke_snake.pth",
    checkpoint_path: Optional[str] = None,
    replay_db_path: Optional[str] = None,
    replay_order: str = "id_uniform",
    min_terminal_fraction: float = 0.0,
    min_exact_mask_fraction: float = 0.0,
    replay_quality_gates: Optional[Dict[str, float]] = None,
    eval_mode: bool = False,
    game_state_factory=None,
    checkpoint_loader=None,
    replay_loader=None,
) -> Dict[str, Any]:
    """Run a bounded in-process training smoke and return learning-health metrics."""
    if max_frames <= 0:
        raise ValueError("health smoke frames must be positive")
    if checkpoint_filename is not None and not str(checkpoint_filename).strip():
        raise ValueError("health smoke checkpoint filename must not be empty")

    factory = game_state_factory or create_training_game_state
    loader = checkpoint_loader or load_checkpoint_into_game_state
    replay_prefill_loader = replay_loader or load_replay_db_into_game_state
    active_replay_gates = (
        resolve_prefill_replay_quality_gates(overrides=replay_quality_gates)
        if replay_quality_gates is not None
        else resolve_legacy_replay_quality_gates(
            min_terminal_fraction=min_terminal_fraction,
            min_exact_mask_fraction=min_exact_mask_fraction,
        )
    )
    game_state = None
    start_time = time.time()
    try:
        game_state = factory()
        checkpoint_loaded = False
        if checkpoint_path:
            if checkpoint_loader is None:
                checkpoint_loaded = bool(
                    loader(
                        game_state,
                        checkpoint_path,
                        strict_training_contract=not eval_mode,
                    )
                )
            else:
                checkpoint_loaded = bool(loader(game_state, checkpoint_path))
            if not checkpoint_loaded:
                raise RuntimeError(f"Could not load health smoke checkpoint: {checkpoint_path}")
        replay_rows_loaded = 0
        if replay_db_path:
            replay_rows_loaded = int(
                replay_prefill_loader(
                    game_state,
                    replay_db_path,
                    replay_order=replay_order,
                    **active_replay_gates,
                )
            )
        if eval_mode:
            configure_eval_game_state(game_state)

        initial_policy_stats = get_policy_stats(game_state)
        initial_updates = int(initial_policy_stats["updates"])
        frame = 0
        while frame < max_frames and game_state.alive_snakes > 0:
            if eval_mode:
                game_state.update(train_mode=True, learn=False)
            else:
                game_state.update(train_mode=True)
            frame += 1

        game_state.flush_episode_experience()
        episode_stats = get_episode_stats(game_state, frame)
        policy_stats = get_policy_stats(game_state)
        saved_checkpoint_path = (
            save_training_checkpoint(game_state, checkpoint_filename)
            if checkpoint_filename is not None
            else None
        )

        min_replay_size = int(policy_stats["min_replay_size"])
        replay_size = int(policy_stats["replay_size"])
        update_delta = int(policy_stats["updates"]) - initial_updates
        return {
            "checkpoint": str(saved_checkpoint_path) if saved_checkpoint_path else None,
            "checkpoint_loaded": checkpoint_loaded,
            "elapsed_seconds": time.time() - start_time,
            "episode": episode_stats,
            "eval_mode": eval_mode,
            "frames": frame,
            "loss_available": policy_stats["last_loss"] is not None,
            "loaded_checkpoint": checkpoint_path,
            "max_frames": int(max_frames),
            "policy": policy_stats,
            "replay_ready": min_replay_size > 0 and replay_size >= min_replay_size,
            "replay_db": replay_db_path,
            "replay_rows_loaded": replay_rows_loaded,
            "terminated": game_state.alive_snakes <= 0,
            "update_delta": update_delta,
            "updates_ran": update_delta > 0,
        }
    finally:
        if game_state is not None and hasattr(game_state, "full_cleanup"):
            game_state.full_cleanup()


def format_learning_health_smoke_report(stats: Dict[str, Any]) -> str:
    """Format learning-health smoke metrics for a compact console report."""
    policy = stats["policy"]
    episode = stats["episode"]
    replay_status = "ready" if stats["replay_ready"] else "warming"
    update_status = "yes" if stats["updates_ran"] else "no"
    update_delta = int(stats.get("update_delta", policy["updates"] if stats["updates_ran"] else 0))
    loss_status = format_optional_float(policy["last_loss"])
    termination = "all snakes died" if stats["terminated"] else "frame cap reached"
    exact_mask_status = format_optional_percent(policy.get("exact_next_action_mask_fraction"))
    valid_next_status = format_optional_percent(policy.get("valid_next_action_fraction"))
    trapped_next_status = format_optional_percent(policy.get("trapped_next_state_fraction"))
    loaded_checkpoint = stats.get("loaded_checkpoint")
    if loaded_checkpoint:
        loaded_state = "loaded" if stats.get("checkpoint_loaded") else "not loaded"
        loaded_status = f"{loaded_checkpoint} ({loaded_state})"
    else:
        loaded_status = "none"
    replay_db = stats.get("replay_db")
    replay_prefill_status = (
        f"{replay_db} ({int(stats.get('replay_rows_loaded', 0)):,} rows)" if replay_db else "none"
    )

    return "\n".join(
        [
            "Learning Health Smoke:",
            f"  Loaded checkpoint: {loaded_status}",
            f"  Replay prefill: {replay_prefill_status}",
            f"  Frames: {stats['frames']}/{stats['max_frames']} ({termination})",
            f"  Replay: {policy['replay_size']}/{policy['min_replay_size']} ({replay_status})",
            f"  Updates: {policy['updates']} (ran: {update_status}, delta: {update_delta})",
            f"  Loss: {loss_status}",
            (
                "  Target actions: "
                f"valid={valid_next_status}, trapped={trapped_next_status}, "
                f"exact_masks={exact_mask_status}"
            ),
            f"  Epsilon: {policy['epsilon']:.3f}",
            (
                "  Episode: "
                f"reward={episode['reward']:.2f}, length={episode['length']}, "
                f"food={episode['food_eaten']}, deaths={episode['deaths']}, "
                f"kills={episode['kills']}"
            ),
            f"  Checkpoint: {stats['checkpoint'] or 'not saved'}",
            f"  Elapsed: {stats['elapsed_seconds']:.2f}s",
        ]
    )


def validate_learning_health_smoke(stats: Dict[str, Any]) -> None:
    """Fail when a learning-health smoke did not exercise the training path."""
    if bool(stats.get("updates_ran")):
        return

    policy = stats.get("policy", {})
    replay_size = int(policy.get("replay_size", 0))
    min_replay_size = int(policy.get("min_replay_size", 0))
    update_delta = int(stats.get("update_delta", 0))
    raise RuntimeError(
        "Learning health smoke did not run any training updates "
        f"(update_delta={update_delta}, replay={replay_size}/{min_replay_size}). "
        "Increase --health-smoke-frames, lower warmup for a smoke config, or prefill replay."
    )


def train_environment(
    env_id,
    episodes,
    save_interval,
    return_dict,
    use_tensorboard=True,
    checkpoint_path: Optional[str] = None,
    replay_db_path: Optional[str] = None,
    replay_order: str = "id_uniform",
    min_terminal_fraction: float = 0.0,
    min_exact_mask_fraction: float = 0.0,
    replay_quality_gates: Optional[Dict[str, float]] = None,
    eval_mode: bool = False,
):
    try:
        # Spawned workers don't inherit the parent's initialized config (mp
        # 'spawn' re-imports the module but never runs main()). Re-initialize from
        # the config path the parent stashed in the environment, so overrides like
        # network.use_gru reach this worker's policy.
        _worker_config_path = os.environ.get("SNAKE_DQN_CONFIG")
        if _worker_config_path:
            initialize_config(load_config(_worker_config_path))

        # Set different seeds for each environment
        torch.manual_seed(env_id)
        np.random.seed(env_id)

        rewards_history = deque(maxlen=100)
        start_time = time.time()
        best_reward = float("-inf")
        best_saved_reward = float("-inf")
        avg_reward = 0.0
        frames_processed = 0
        last_save_time = time.time()
        last_print_time = time.time()
        best_checkpoint_path = None
        active_replay_gates = (
            resolve_prefill_replay_quality_gates(overrides=replay_quality_gates)
            if replay_quality_gates is not None
            else resolve_legacy_replay_quality_gates(
                min_terminal_fraction=min_terminal_fraction,
                min_exact_mask_fraction=min_exact_mask_fraction,
            )
        )

        # Curriculum manager for progressive difficulty
        curriculum = None
        if GameConfig.CURRICULUM_ENABLED:
            from src.training.curriculum import CurriculumManager

            curriculum = CurriculumManager(window_size=GameConfig.CURRICULUM_WINDOW_SIZE)
            # Try to load curriculum state from previous run
            curriculum_path = get_env_curriculum_checkpoint_path(env_id)
            if curriculum_path.exists():
                try:
                    curriculum_state = torch.load(curriculum_path, weights_only=False)
                    curriculum.load_state(curriculum_state)
                    print(
                        f"Env {env_id} | Curriculum resumed at phase: "
                        f"{curriculum.phase_name} (episode {curriculum.total_episodes})"
                    )
                except (RuntimeError, EOFError) as e:
                    print(
                        f"Env {env_id} | Could not load curriculum state: {e}, " f"starting fresh"
                    )
            else:
                print(
                    f"Env {env_id} | Curriculum enabled, starting phase: "
                    f"{curriculum.phase_name}"
                )

        game_state = create_training_game_state(curriculum, eval_mode=eval_mode)
        if checkpoint_path:
            checkpoint_loaded = load_checkpoint_into_game_state(
                game_state,
                checkpoint_path,
                strict_training_contract=not eval_mode,
            )
            if not checkpoint_loaded:
                raise RuntimeError(
                    f"Could not load headless training checkpoint: {checkpoint_path}"
                )
            if eval_mode:
                configure_eval_game_state(game_state)
        if replay_db_path:
            load_replay_db_into_game_state(
                game_state,
                replay_db_path,
                replay_order=replay_order,
                **active_replay_gates,
            )
        if eval_mode:
            configure_eval_game_state(game_state)

        # TensorBoard logger for this environment
        tb_logger = None
        if use_tensorboard:
            tb_logger = TensorBoardLogger(
                log_dir=f"logs/tensorboard/env_{env_id}", comment=f"_env{env_id}_apex"
            )

        print(f"Environment {env_id} initialized on device: {game_state.snakes[0].ai.device}")

        for episode in range(episodes):
            game_state.reset()
            episode_reward = 0.0
            frame = 0

            while frame < GameConfig.MAX_FRAMES and game_state.alive_snakes > 0:
                game_state.update(train_mode=True, learn=not eval_mode)
                frame += 1
                frames_processed += 1
                episode_reward = float(
                    getattr(game_state, "episode_current_reward", game_state.episode_best_reward)
                )
                if episode_reward > best_reward:
                    best_reward = episode_reward

                # Save periodically (but not too frequently)
                current_time = time.time()
                if (
                    frame % save_interval == 0 and current_time - last_save_time >= 60
                ):  # At least 1 minute between saves
                    if episode_reward > best_saved_reward:
                        best_saved_reward = episode_reward
                        best_model_name = f"env_{env_id}_best_snake.pth"
                        saved_checkpoint_path = save_training_checkpoint(
                            game_state,
                            best_model_name,
                            env_id=env_id,
                            curriculum=curriculum,
                        )
                        if saved_checkpoint_path is not None:
                            best_checkpoint_path = saved_checkpoint_path
                        last_save_time = current_time
                        print(
                            f"Env {env_id} | New best reward: {best_reward:.2f} | "
                            f"Saved model to {saved_checkpoint_path or best_model_name}"
                        )

            game_state.flush_episode_experience()
            episode_stats = get_episode_stats(game_state, frame)
            policy_stats = get_policy_stats(game_state)
            episode_reward = episode_stats["reward"]
            rewards_history.append(float(episode_reward))
            avg_reward = sum(rewards_history) / len(rewards_history) if rewards_history else 0

            # Curriculum: record episode and check for promotion
            if curriculum is not None:
                curriculum.record_episode(
                    length=episode_stats["best_length"],
                    kills=episode_stats["kills"],
                    deaths=episode_stats["deaths"],
                )
                promoted, phase = curriculum.check_and_promote()
                if promoted:
                    settings = curriculum.get_game_settings()
                    new_num = settings["num_snakes"]
                    food_mult = settings["food_multiplier"]
                    b_scale = settings["board_scale"]
                    print(
                        f"Env {env_id} | Curriculum promoted to phase "
                        f"'{phase.name}' (num_snakes={new_num}, "
                        f"food_mult={food_mult}, board_scale={b_scale}) "
                        f"at episode {episode}"
                    )
                    # Preserve learned policy (weights + replay buffer)
                    old_policy = game_state._shared_policy
                    game_state.release_snakes_keep_policy()
                    # Rebuild GameState with new settings, reusing the policy
                    game_state = create_training_game_state(
                        curriculum=curriculum,
                        shared_policy=old_policy,
                        eval_mode=eval_mode,
                    )
                    # Save curriculum state on promotion
                    curriculum_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        curriculum.get_state(),
                        curriculum_path,
                    )

            # TensorBoard episode logging
            if tb_logger:
                episode_log_kwargs = {
                    "deaths": episode_stats["deaths"],
                    "epsilon": policy_stats["epsilon"],
                    "kills": episode_stats["kills"],
                    "replay_min_ready": policy_stats["min_replay_size"],
                    "replay_size": policy_stats["replay_size"],
                    "updates": policy_stats["updates"],
                }
                if policy_stats["last_loss"] is not None:
                    episode_log_kwargs["loss"] = policy_stats["last_loss"]
                for metric_name in (
                    "valid_next_action_fraction",
                    "trapped_next_state_fraction",
                    "exact_next_action_mask_fraction",
                ):
                    if metric_name in policy_stats:
                        episode_log_kwargs[metric_name] = policy_stats[metric_name]
                for collision_type, count in episode_stats["collision_counts"].items():
                    episode_log_kwargs[f"collisions_{collision_type}"] = count

                tb_logger.log_episode_metrics(
                    episode=episode,
                    total_reward=episode_reward,
                    episode_length=episode_stats["length"],
                    snake_length=episode_stats["best_length"],
                    food_eaten=episode_stats["food_eaten"],
                    **episode_log_kwargs,
                )

            # Log progress every 10 episodes or at least every 30 seconds
            current_time = time.time()
            if episode % 10 == 0 or current_time - last_print_time >= 30:
                elapsed_time = current_time - start_time
                fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
                memory = psutil.Process().memory_info().rss / 1024**2  # Memory in MB
                phase_str = f" | Phase: {curriculum.phase_name}" if curriculum else ""
                collision_counts = format_collision_counts(episode_stats["collision_counts"])
                loss = format_optional_float(policy_stats["last_loss"])

                print(
                    f"Env {env_id} | Episode {episode}/{episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | Best: {best_reward:.2f} | "
                    f"Food: {episode_stats['food_eaten']} | "
                    f"Deaths: {episode_stats['deaths']} ({collision_counts}) | "
                    f"Kills: {episode_stats['kills']} | "
                    f"Replay: {policy_stats['replay_size']}/{policy_stats['min_replay_size']} | "
                    f"Updates: {policy_stats['updates']} | Loss: {loss} | "
                    f"FPS: {fps:.1f} | Epsilon: {policy_stats['epsilon']:.3f} | "
                    f"Memory: {memory:.1f}MB{phase_str}"
                )
                last_print_time = current_time

        if eval_mode:
            final_checkpoint_path = None
        else:
            final_checkpoint_path, best_checkpoint_path = save_final_training_checkpoints(
                game_state,
                env_id,
                curriculum=curriculum,
                best_checkpoint_path=best_checkpoint_path,
            )
            if final_checkpoint_path is not None:
                print(f"Env {env_id} | Saved final model to {final_checkpoint_path}")
                print(f"Env {env_id} | Best checkpoint for this run: {best_checkpoint_path}")

        final_policy_stats = get_policy_stats(game_state)

        # Release GPU memory and game resources on shutdown
        game_state.full_cleanup()

        # Close TensorBoard logger
        if tb_logger:
            tb_logger.close()

        return_dict[env_id] = {
            "best_reward": float(best_reward),
            "final_avg_reward": float(avg_reward),
            "final_replay_size": int(final_policy_stats["replay_size"]),
            "final_updates": int(final_policy_stats["updates"]),
            "total_frames": frames_processed,
            "training_time": time.time() - start_time,
            "final_checkpoint": str(final_checkpoint_path) if final_checkpoint_path else None,
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path else None,
            "eval_mode": eval_mode,
            "error": None,
        }
    except Exception as e:
        print(f"Error in environment {env_id}:")
        traceback.print_exc()
        return_dict[env_id] = {
            "best_reward": float("-inf"),
            "final_avg_reward": float("-inf"),
            "final_replay_size": 0,
            "final_updates": 0,
            "total_frames": 0,
            "training_time": 0,
            "error": str(e),
        }


def train_headless(
    episodes=1000000,
    save_interval=1000,
    num_envs=None,
    use_tensorboard=True,
    checkpoint_path: Optional[str] = None,
    replay_db_path: Optional[str] = None,
    replay_order: str = "id_uniform",
    min_terminal_fraction: float = 0.0,
    min_exact_mask_fraction: float = 0.0,
    replay_quality_gates: Optional[Dict[str, float]] = None,
    eval_mode: bool = False,
):
    # Print system info
    print("\nSystem Information:")
    print(
        f"CPU Cores: {psutil.cpu_count(logical=False)} (Physical), {psutil.cpu_count()} (Logical)"
    )
    memory = psutil.virtual_memory()
    print(
        f"Memory: {memory.total / 1024**3:.1f}GB Total, "
        f"{memory.available / 1024**3:.1f}GB Available"
    )
    selected_device = DeviceManager.get_device()
    gpu_name = selected_device.type.upper() if selected_device.type != "cpu" else "None (CPU)"
    print(f"GPU: {gpu_name}")
    print(f"TensorBoard: {'Enabled' if use_tensorboard else 'Disabled'}")

    # Automatically determine optimal number of environments if not specified
    if num_envs is None:
        num_envs = get_optimal_env_count()

    # Set up device and optimize threads
    if torch.cuda.is_available():
        torch.set_num_threads(1)  # Optimize CPU usage when using GPU
    elif torch.backends.mps.is_available():
        # MPS (Apple M1) specific optimizations
        torch.set_num_threads(4)  # M1 works best with 4 threads
        print("💡 M1 Optimizations: Larger batches, more parallel envs")

    manager = mp.Manager()
    return_dict = manager.dict()
    process_entries = []

    episode_counts = split_episodes_across_envs(episodes, num_envs)
    active_envs = [(env_id, count) for env_id, count in enumerate(episode_counts) if count > 0]
    active_replay_gates = (
        resolve_prefill_replay_quality_gates(overrides=replay_quality_gates)
        if replay_quality_gates is not None
        else resolve_legacy_replay_quality_gates(
            min_terminal_fraction=min_terminal_fraction,
            min_exact_mask_fraction=min_exact_mask_fraction,
        )
    )

    print("\nTraining Configuration:")
    print(f"Number of environments: {num_envs}")
    print(f"Active environments: {len(active_envs)}")
    print(f"Episodes per environment: {episode_counts}")
    print(f"Mode: {'evaluation' if eval_mode else 'training'}")
    print(f"Save interval: {save_interval}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    if replay_db_path:
        print(f"Replay prefill DB: {replay_db_path}")
        print(f"Replay prefill order: {replay_order}")
        for gate_line in format_enabled_replay_quality_gates(active_replay_gates):
            print(gate_line)
    print(f"Device: {DeviceManager.get_device()}")

    if not active_envs:
        print("\nNo episodes requested; nothing to train.")
        return

    start_time = time.time()
    try:
        for env_id, env_episodes in active_envs:
            p = mp.Process(
                target=train_environment,
                args=(
                    env_id,
                    env_episodes,
                    save_interval,
                    return_dict,
                    use_tensorboard,
                    checkpoint_path,
                    replay_db_path,
                    replay_order,
                    min_terminal_fraction,
                    min_exact_mask_fraction,
                    active_replay_gates,
                    eval_mode,
                ),
            )
            process_entries.append((env_id, p))
            p.start()
            time.sleep(1)  # Stagger start to prevent memory spikes

        for _, p in process_entries:
            p.join()

        failures = collect_training_worker_failures(process_entries, active_envs, return_dict)
        if failures:
            raise RuntimeError("Headless training failed:\n  - " + "\n  - ".join(failures))

        if not return_dict:
            print("\nNo environments completed successfully.")
            return

        # Summarize results from successful environments
        successful_envs = {k: v for k, v in return_dict.items() if v["best_reward"] > float("-inf")}
        if successful_envs:
            best_env = max(successful_envs.items(), key=lambda x: x[1]["best_reward"])
            total_time = time.time() - start_time
            total_frames = sum(env["total_frames"] for env in successful_envs.values())

            print("\nTraining Summary:")
            print(f"Total training time: {total_time/3600:.1f} hours")
            print(f"Total frames processed: {total_frames:,}")
            print(f"Average FPS across all envs: {total_frames/total_time:.1f}")
            print(f"\nBest performing environment: {best_env[0]}")
            print(f"Best reward achieved: {best_env[1]['best_reward']:.2f}")
            print(f"Final average reward: {best_env[1]['final_avg_reward']:.2f}")

            if not eval_mode:
                # Copy the best environment's model to the main best_snake.pth
                best_env_id = best_env[0]
                best_model_path = get_best_env_model_path(best_env_id, best_env[1])
                target_model_path = get_checkpoint_path(GameConfig.BEST_MODEL_NAME)
                if best_model_path.exists():
                    target_model_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(best_model_path, target_model_path)
                    print(f"\nCopied best model from env_{best_env_id} to {target_model_path}")
        else:
            print("\nNo environments completed successfully with valid rewards.")

    except Exception:
        print("Error in training process:")
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        for _, p in process_entries:
            if p.is_alive():
                p.terminate()
                p.join()


def main():
    parser = argparse.ArgumentParser(
        description="Snake RL Game - Apex DQN Training Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                             # Run UI with Apex DQN
  python src/main.py --headless --episodes 10000 # Headless training
  python src/main.py --human                     # Human play mode
  python src/main.py --load saved_snakes/best.pth  # Load trained model
        """,
    )

    # Mode selection
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help=(
            "Select compute device for training. On Apple Silicon, CPU is often faster for "
            "this small model."
        ),
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode for faster training"
    )
    parser.add_argument(
        "--human", action="store_true", help="Enable human control mode with arrow keys"
    )

    # Training settings
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000000,
        help="Number of episodes for headless training (default: 1000000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional training batch-size override. Useful for quick health smokes "
            "or headless runs with smaller replay prefills."
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Interval to save model checkpoints (default: 1000)",
    )
    parser.add_argument(
        "--num-envs", type=int, help="Number of parallel environments (default: auto-detect)"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,
        help="Enable TensorBoard logging (default: enabled)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_false",
        dest="tensorboard",
        help="Disable TensorBoard logging",
    )

    # Model loading
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to checkpoint file to load (e.g., saved_snakes/snake_apex_4.pth)",
    )
    parser.add_argument(
        "--load-memory-db",
        nargs="?",
        const="snake_memories.db",
        default=None,
        metavar="PATH",
        help=(
            "Prefill headless replay from a generated SQLite database "
            "(default path when flag is present: snake_memories.db)"
        ),
    )
    parser.add_argument(
        "--load-memory-order",
        choices=("id_uniform", "id", "priority"),
        default="id_uniform",
        help=(
            "Ordering for --load-memory-db when the replay DB is larger than memory capacity. "
            "'id_uniform' spreads capped loads across insertion order; "
            "'id' loads the oldest rows exactly; 'priority' loads highest-priority rows."
        ),
    )
    parser.add_argument(
        "--replay-quality-preset",
        choices=tuple(REPLAY_QUALITY_GATE_PRESETS),
        default="none",
        help=(
            "Named replay quality gate bundle for --load-memory-db. Use 'training' to "
            "require terminal, mask, action, reward, n-step, and target-quality coverage."
        ),
    )
    parser.add_argument(
        "--min-terminal-fraction",
        type=float,
        default=None,
        help=(
            "Minimum terminal-row fraction required for --load-memory-db; "
            "overrides the selected replay quality preset when provided."
        ),
    )
    parser.add_argument(
        "--min-immediate-terminal-fraction",
        type=float,
        default=None,
        help=(
            "Minimum one-step terminal-row fraction required for --load-memory-db; "
            "overrides the selected replay quality preset when provided."
        ),
    )
    parser.add_argument(
        "--min-exact-mask-fraction",
        type=float,
        default=None,
        help=(
            "Minimum exact next-action-mask fraction among nonterminal rows required for "
            "--load-memory-db; overrides the selected replay quality preset when provided."
        ),
    )
    parser.add_argument(
        "--min-boost-mask-fraction",
        type=float,
        default=None,
        help=(
            "Minimum fraction of nonterminal loaded replay rows whose exact next-action "
            "mask allows boost; overrides the selected replay quality preset when provided."
        ),
    )
    parser.add_argument(
        "--min-action-coverage-fraction",
        type=float,
        default=None,
        help=("Minimum fraction of the 6 actions that must appear in --load-memory-db replay."),
    )
    parser.add_argument(
        "--min-positive-reward-fraction",
        type=float,
        default=None,
        help="Minimum fraction of loaded replay rows that must have positive rewards.",
    )
    parser.add_argument(
        "--min-negative-reward-fraction",
        type=float,
        default=None,
        help="Minimum fraction of loaded replay rows that must have negative rewards.",
    )
    parser.add_argument(
        "--min-multistep-fraction",
        type=float,
        default=None,
        help="Minimum fraction of loaded replay rows that must use bootstrap_steps > 1.",
    )
    parser.add_argument(
        "--max-dominant-action-fraction",
        type=float,
        default=None,
        help="Maximum fraction of loaded replay rows that any single action may occupy.",
    )
    parser.add_argument(
        "--max-invalid-current-action-fraction",
        type=float,
        default=None,
        help=(
            "Maximum fraction of loaded replay rows whose stored action is invalid under "
            "current-state danger/boost features."
        ),
    )
    parser.add_argument(
        "--max-nonterminal-trapped-next-fraction",
        type=float,
        default=None,
        help=(
            "Maximum fraction of nonterminal loaded replay targets whose exact next-action "
            "mask has no legal actions."
        ),
    )
    parser.add_argument(
        "--max-exact-mask-state-mismatch-fraction",
        type=float,
        default=None,
        help=(
            "Maximum fraction of exact next-action masks that may disagree with next-state "
            "per-action danger features."
        ),
    )
    parser.add_argument(
        "--max-malformed-state-feature-fraction",
        type=float,
        default=None,
        help=(
            "Maximum fraction of decoded current/next states with malformed semantic " "features."
        ),
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluation mode: run with epsilon=0 (greedy, no exploration)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/default.yaml)",
    )
    parser.add_argument(
        "--show-config", action="store_true", help="Show configuration summary and exit"
    )
    parser.add_argument(
        "--health-smoke",
        action="store_true",
        help="Run a bounded in-process learning-health smoke and exit",
    )
    parser.add_argument(
        "--health-smoke-frames",
        type=int,
        default=200,
        help="Maximum frames for --health-smoke (default: 200)",
    )
    parser.add_argument(
        "--health-smoke-checkpoint",
        type=str,
        default="health_smoke_snake.pth",
        help="Checkpoint filename written by --health-smoke",
    )

    args = parser.parse_args()
    if args.device != "auto":
        os.environ["SNAKE_DQN_DEVICE"] = args.device

    try:
        resolve_training_batch_size(args.batch_size)
        replay_quality_gates = resolve_prefill_replay_quality_gates(
            preset=args.replay_quality_preset,
            overrides={name: getattr(args, name) for name in REPLAY_QUALITY_GATE_ORDER},
        )
    except ValueError as e:
        parser.error(str(e))

    # Load YAML configuration if specified
    if args.config:
        # Propagate the config path to spawned headless workers (mp 'spawn' does
        # NOT re-run main(), so workers would otherwise fall back to default
        # config and ignore overrides like network.use_gru). Workers read this
        # env var and re-initialize config themselves.
        os.environ["SNAKE_DQN_CONFIG"] = args.config
        try:
            config = load_config(args.config)
            apply_config_to_game_config(config)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(
                "   Available configs: configs/default.yaml, "
                "configs/production.yaml, configs/training_fast.yaml"
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    batch_size = apply_training_batch_size_override(args.batch_size)

    # Show config summary and exit if requested
    if args.show_config:
        print(get_config_summary(get_config()))
        sys.exit(0)

    # All snakes use Apex DQN
    snake_policies = ["apex"] * GameConfig.NUM_SNAKES

    # Print startup info
    print("\n" + "=" * 50)
    print("SNAKE RL - Apex DQN Training")
    print("=" * 50)
    if args.health_smoke:
        mode_str = "Health Smoke"
    elif args.headless:
        mode_str = "Headless"
    elif args.human:
        mode_str = "Human Control"
    else:
        mode_str = "AI (UI)"
    if args.eval:
        mode_str += " [EVAL: epsilon=0]"
    print(f"Mode: {mode_str}")
    print("Policy: APEX DQN")
    print(f"Batch size: {batch_size}")
    print("=" * 50 + "\n")

    if args.health_smoke:
        print("Running bounded learning-health smoke...")
        stats = run_learning_health_smoke(
            max_frames=args.health_smoke_frames,
            checkpoint_filename=args.health_smoke_checkpoint,
            checkpoint_path=args.load,
            replay_db_path=args.load_memory_db,
            replay_order=args.load_memory_order,
            replay_quality_gates=replay_quality_gates,
            eval_mode=args.eval,
        )
        print(format_learning_health_smoke_report(stats))
        if not args.eval:
            try:
                validate_learning_health_smoke(stats)
            except RuntimeError as e:
                print(f"Learning health smoke failed: {e}")
                sys.exit(1)
    elif args.headless:
        # Critical for macOS + PyQt5: must use 'spawn' start method
        mp.set_start_method("spawn", force=True)
        print("Starting optimized headless training...")
        if args.tensorboard:
            print("TensorBoard: Run 'tensorboard --logdir=logs/tensorboard' to view")
        train_headless(
            args.episodes,
            args.save_interval,
            args.num_envs,
            args.tensorboard,
            args.load,
            args.load_memory_db,
            args.load_memory_order,
            replay_quality_gates=replay_quality_gates,
            eval_mode=args.eval,
        )
    else:
        mode = "Human control" if args.human else "AI"
        print(f"Starting UI mode ({mode})...")
        if args.tensorboard:
            print("TensorBoard: Run 'tensorboard --logdir=logs/tensorboard' to view")
        app = QApplication(sys.argv)
        app.game = SlitherIOGame(
            human_mode=args.human,
            snake_policies=snake_policies,
            use_tensorboard=args.tensorboard,
            load_model_path=args.load,
            eval_mode=args.eval,
        )
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
