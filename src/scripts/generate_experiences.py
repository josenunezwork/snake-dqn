#!/usr/bin/env python3
"""
Fast Experience Generation for Apex-DQN

Generate training experiences quickly in headless mode and save to database.
Much faster than UI mode! Supports parallel environments.

This script uses Apex-DQN policy exclusively for experience generation.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import shlex
import sys
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

# Add parent directory to path (root of project)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import (  # noqa: E402
    apply_config_to_game_config,
    load_config,
)
from src.core.game_config import GameConfig  # noqa: E402
from src.core.reward_contract import current_reward_contract  # noqa: E402
from src.data.memory_db_handler import (  # noqa: E402
    REPLAY_QUALITY_GATE_ORDER,
    REPLAY_QUALITY_GATE_PRESETS,
    current_action_invalid_from_state,
    resolve_min_row_count,
    resolve_replay_quality_fraction,
    resolve_replay_quality_gate_values,
    validate_min_row_count,
)
from src.data.memory_db_handler import (  # noqa: E402
    validate_replay_quality_gates as validate_shared_replay_quality_gates,
)
from src.scripts.audit_replay import format_reusable_gate_args  # noqa: E402
from src.training.checkpoint_contract import validate_checkpoint_contract  # noqa: E402

if TYPE_CHECKING:
    from src.data.memory_db_handler import MemoryDBHandler

DEFAULT_REPLAY_DB = "snake_memories.db"
DEFAULT_GENERATION_ENV_PRESET = "collision_dense"
DEFAULT_GENERATION_REPLAY_QUALITY_PRESET = "training"
DEFAULT_BOOST_EXPLORATION_RATE = 0.25
DEFAULT_DANGER_EXPLORATION_RATE = 0.02
GENERATION_APPEND_CONTRACT_KEYS = (
    "generation.policy_type",
    "generation.state_size",
    "generation.action_size",
    "generation.gamma",
    "generation.apex_n_step",
    "generation.reward_contract",
    "generation.reward_death",
    "generation.reward_food_base",
    "generation.num_snakes",
    "generation.board_scale",
    "generation.board_width",
    "generation.board_height",
    "generation.food_multiplier",
    "generation.initial_food",
    "generation.max_food",
)
GENERATION_ENVIRONMENT_PRESETS = {
    "default": {
        "num_snakes": None,
        "board_scale": 1.0,
        "food_multiplier": 1.0,
        "exploration_epsilon": None,
        "boost_exploration_rate": DEFAULT_BOOST_EXPLORATION_RATE,
        "danger_exploration_rate": DEFAULT_DANGER_EXPLORATION_RATE,
    },
    "collision_dense": {
        "num_snakes": 6,
        "board_scale": 0.20,
        "food_multiplier": 0.5,
        "exploration_epsilon": None,
        "boost_exploration_rate": 0.25,
        "danger_exploration_rate": 0.0,
    },
}


def clamp_epsilon(value: float, field_name: str = "epsilon") -> float:
    """Return a finite epsilon clamped to [0, 1]."""
    try:
        epsilon = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite") from exc
    if not math.isfinite(epsilon):
        raise ValueError(f"{field_name} must be finite")
    return max(0.0, min(1.0, epsilon))


def get_optimal_env_count():
    """Determine optimal number of parallel environments based on system resources."""
    import psutil

    cpu_count = psutil.cpu_count(logical=False) or 4
    memory_gb = psutil.virtual_memory().total / (1024**3)

    # Each env uses ~1-2GB RAM, leave headroom
    memory_based = max(1, int((memory_gb - 4) / 1.5))
    cpu_based = max(1, cpu_count - 1)

    optimal = min(memory_based, cpu_based, 8)  # Cap at 8
    return max(1, optimal)


def configure_optional_torch_runtime() -> bool:
    """Apply optional Torch runtime tweaks without blocking CLI validation paths."""
    try:
        import torch
    except (ImportError, OSError):
        return False

    if torch.backends.mps.is_available():
        print("🚀 M1 Detected - Using Metal Performance Shaders")
        print("💡 Optimized for M1's unified memory architecture")
        torch.set_num_threads(4)
        return True
    return False


def get_shared_apex_policy(game_state):
    """Return the shared Apex policy used by the AI snakes."""
    policy = getattr(game_state, "_shared_policy", None)
    if policy is not None:
        return policy
    for snake in game_state.snakes:
        if hasattr(snake, "policy"):
            return snake.policy
    return None


def split_episodes_across_envs(episodes: int, num_envs: int) -> list[int]:
    """Split requested episodes across workers without dropping remainders."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    base = episodes // num_envs
    remainder = episodes % num_envs
    return [base + (1 if env_id < remainder else 0) for env_id in range(num_envs)]


def validate_database_path(db_path: str) -> str:
    """Resolve a replay database path and reject empty values."""
    if not str(db_path).strip():
        raise ValueError("db path must not be empty")
    return str(Path(db_path).expanduser())


def ensure_database_parent(db_path: str) -> str:
    """Create the parent directory for a replay database path if needed."""
    resolved_path = validate_database_path(db_path)
    parent = Path(resolved_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def get_env_database_path(db_path: str, env_id: int) -> str:
    """Return the per-worker temporary database path for a parallel run."""
    if env_id < 0:
        raise ValueError("env_id must be non-negative")

    path = Path(validate_database_path(db_path))
    suffix = path.suffix or ".db"
    return str(path.with_name(f"{path.stem}_env{env_id}{suffix}"))


def remove_sqlite_files(db_path: str) -> None:
    """Remove a SQLite database and its WAL/SHM sidecar files if present."""
    for suffix in ("", "-wal", "-shm"):
        path = f"{db_path}{suffix}"
        if os.path.exists(path):
            os.remove(path)


def prepare_generation_output_database(db_path: str, append: bool = False) -> str:
    """Prepare the main generated replay database path."""
    output_db_path = ensure_database_parent(db_path)
    if not append:
        remove_sqlite_files(output_db_path)
    return output_db_path


def resolve_generation_frame_limit(max_frames=None) -> int:
    """Resolve and validate the per-episode frame cap for replay generation."""
    frame_limit = GameConfig.MAX_FRAMES if max_frames is None else int(max_frames)
    if frame_limit <= 0:
        raise ValueError("max_frames must be positive")
    return frame_limit


def resolve_generation_num_snakes(num_snakes=None) -> int:
    """Resolve and validate the number of snakes for replay generation."""
    resolved = GameConfig.NUM_SNAKES if num_snakes is None else int(num_snakes)
    if resolved <= 0:
        raise ValueError("num_snakes must be positive")
    if resolved > len(GameConfig.SNAKE_COLORS):
        raise ValueError("num_snakes exceeds available snake colors")
    return resolved


def resolve_generation_scale(value, field_name: str) -> float:
    """Resolve a positive finite generation environment scale."""
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and positive") from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return scale


def resolve_generation_environment_settings(
    num_snakes=None,
    board_scale=1.0,
    food_multiplier=1.0,
) -> dict:
    """Return validated environment-shaping settings for generated replay."""
    resolved_num_snakes = resolve_generation_num_snakes(num_snakes)
    resolved_board_scale = resolve_generation_scale(board_scale, "board_scale")
    resolved_food_multiplier = resolve_generation_scale(food_multiplier, "food_multiplier")

    width = int(GameConfig.WIDTH * resolved_board_scale)
    height = int(GameConfig.HEIGHT * resolved_board_scale)
    min_dimension = GameConfig.WALL_THICKNESS * 2 + GameConfig.SEGMENT_SIZE
    if width <= min_dimension or height <= min_dimension:
        raise ValueError("board_scale makes the generated replay arena too small")

    initial_food = int(GameConfig.INITIAL_FOOD * resolved_food_multiplier)
    max_food = int(GameConfig.MAX_FOOD * resolved_food_multiplier)
    if initial_food <= 0 or max_food <= 0:
        raise ValueError("food_multiplier must leave at least one generated food item")

    return {
        "num_snakes": resolved_num_snakes,
        "board_scale": resolved_board_scale,
        "food_multiplier": resolved_food_multiplier,
    }


def resolve_generation_environment_preset(
    preset: str = DEFAULT_GENERATION_ENV_PRESET,
) -> dict:
    """Return replay-generation environment defaults for a named data profile."""
    if preset not in GENERATION_ENVIRONMENT_PRESETS:
        choices = ", ".join(sorted(GENERATION_ENVIRONMENT_PRESETS))
        raise ValueError(f"replay environment preset must be one of: {choices}")
    return dict(GENERATION_ENVIRONMENT_PRESETS[preset])


def resolve_generation_quality_fraction(value=None, field_name: str = "quality_fraction") -> float:
    """Resolve an optional generated replay-quality gate in [0, 1]."""
    return resolve_replay_quality_fraction(value, field_name)


def resolve_generation_min_terminal_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for terminal coverage."""
    return resolve_generation_quality_fraction(value, "min_terminal_fraction")


def resolve_generation_min_immediate_terminal_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for one-step terminal coverage."""
    return resolve_generation_quality_fraction(value, "min_immediate_terminal_fraction")


def resolve_generation_min_exact_mask_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for exact next-action-mask coverage."""
    return resolve_generation_quality_fraction(value, "min_exact_mask_fraction")


def resolve_generation_min_boost_mask_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for boost-valid next-action masks."""
    return resolve_generation_quality_fraction(value, "min_boost_mask_fraction")


def resolve_generation_min_action_coverage_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for action-space coverage."""
    return resolve_generation_quality_fraction(value, "min_action_coverage_fraction")


def resolve_generation_min_positive_reward_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for successful reward coverage."""
    return resolve_generation_quality_fraction(value, "min_positive_reward_fraction")


def resolve_generation_min_negative_reward_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for danger/death reward coverage."""
    return resolve_generation_quality_fraction(value, "min_negative_reward_fraction")


def resolve_generation_min_multistep_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for n-step return coverage."""
    return resolve_generation_quality_fraction(value, "min_multistep_fraction")


def resolve_generation_max_dominant_action_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for action concentration."""
    if value is None:
        return 1.0
    return resolve_generation_quality_fraction(value, "max_dominant_action_fraction")


def resolve_generation_max_invalid_current_action_fraction(value=None) -> float:
    """Resolve an optional gate for current-state-invalid stored actions."""
    if value is None:
        return 1.0
    return resolve_generation_quality_fraction(value, "max_invalid_current_action_fraction")


def resolve_generation_max_nonterminal_trapped_next_fraction(value=None) -> float:
    """Resolve an optional replay-quality gate for trapped next-state targets."""
    if value is None:
        return 1.0
    return resolve_generation_quality_fraction(
        value,
        "max_nonterminal_trapped_next_fraction",
    )


def resolve_generation_replay_quality_gates(
    preset: str = DEFAULT_GENERATION_REPLAY_QUALITY_PRESET,
    overrides: dict[str, object] | None = None,
) -> dict[str, float]:
    """Resolve generated-replay quality gates from a preset and CLI overrides."""
    return resolve_replay_quality_gate_values(preset=preset, overrides=overrides)


def build_generation_replay_quality_gates(
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
) -> dict[str, float]:
    """Return resolved generated-replay gates keyed by shared gate name."""
    return {
        "min_terminal_fraction": min_terminal_fraction,
        "min_immediate_terminal_fraction": min_immediate_terminal_fraction,
        "min_exact_mask_fraction": min_exact_mask_fraction,
        "min_boost_mask_fraction": min_boost_mask_fraction,
        "min_action_coverage_fraction": min_action_coverage_fraction,
        "min_positive_reward_fraction": min_positive_reward_fraction,
        "min_negative_reward_fraction": min_negative_reward_fraction,
        "min_multistep_fraction": min_multistep_fraction,
        "max_dominant_action_fraction": max_dominant_action_fraction,
        "max_invalid_current_action_fraction": max_invalid_current_action_fraction,
        "max_nonterminal_trapped_next_fraction": max_nonterminal_trapped_next_fraction,
        "max_exact_mask_state_mismatch_fraction": max_exact_mask_state_mismatch_fraction,
        "max_malformed_state_feature_fraction": max_malformed_state_feature_fraction,
    }


def build_generation_replay_contract(env_settings: dict) -> dict:
    """Return the replay semantics that must match when generated DBs are appended."""
    board_scale = float(env_settings["board_scale"])
    food_multiplier = float(env_settings["food_multiplier"])
    reward_contract = current_reward_contract()
    return {
        "generation.action_size": int(GameConfig.OUTPUT_SIZE),
        "generation.apex_n_step": int(GameConfig.APEX_N_STEP),
        "generation.board_height": int(GameConfig.HEIGHT * board_scale),
        "generation.board_scale": board_scale,
        "generation.board_width": int(GameConfig.WIDTH * board_scale),
        "generation.food_multiplier": food_multiplier,
        "generation.gamma": float(GameConfig.APEX_GAMMA),
        "generation.initial_food": int(GameConfig.INITIAL_FOOD * food_multiplier),
        "generation.max_food": int(GameConfig.MAX_FOOD * food_multiplier),
        "generation.num_snakes": int(env_settings["num_snakes"]),
        "generation.policy_type": "apex",
        "generation.reward_contract": reward_contract,
        "generation.reward_death": float(reward_contract["death"]),
        "generation.reward_food_base": float(reward_contract["food_base"]),
        "generation.state_size": int(GameConfig.INPUT_SIZE),
    }


def _metadata_value_label(value) -> str:
    """Return a concise metadata value for append-contract errors."""
    if isinstance(value, float):
        return f"{value:g}"
    return repr(value)


def _raise_append_metadata_mismatch(
    db_path: str,
    metadata_key: str,
    existing,
    expected,
) -> None:
    raise RuntimeError(
        f"Cannot append to {db_path}: existing replay metadata "
        f"{metadata_key}={_metadata_value_label(existing)} does not match current "
        f"{metadata_key}={_metadata_value_label(expected)}. Use overwrite mode or a new --db path."
    )


def _validate_append_metadata_value(
    db_path: str,
    metadata_key: str,
    existing,
    expected,
) -> None:
    """Validate one existing metadata value against the intended append contract."""
    if isinstance(expected, Mapping):
        if not isinstance(existing, Mapping):
            _raise_append_metadata_mismatch(db_path, metadata_key, existing, expected)
        for child_key, child_expected in expected.items():
            child_metadata_key = f"{metadata_key}.{child_key}"
            if child_key not in existing:
                raise RuntimeError(
                    f"Cannot append to {db_path}: existing replay metadata is missing "
                    f"{child_metadata_key}. Use overwrite mode or a new --db path."
                )
            _validate_append_metadata_value(
                db_path,
                child_metadata_key,
                existing[child_key],
                child_expected,
            )
        return

    if isinstance(expected, bool):
        if existing is not expected:
            _raise_append_metadata_mismatch(db_path, metadata_key, existing, expected)
        return

    if isinstance(expected, (int, float)):
        if isinstance(existing, (bool, str, bytes, bytearray, memoryview)):
            _raise_append_metadata_mismatch(db_path, metadata_key, existing, expected)
        try:
            existing_number = float(existing)
            expected_number = float(expected)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot append to {db_path}: existing replay metadata "
                f"{metadata_key} must be finite"
            ) from exc
        if not math.isfinite(existing_number) or not math.isfinite(expected_number):
            raise RuntimeError(
                f"Cannot append to {db_path}: existing replay metadata "
                f"{metadata_key} must be finite"
            )
        if not math.isclose(
            existing_number,
            expected_number,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            _raise_append_metadata_mismatch(
                db_path,
                metadata_key,
                existing_number,
                expected_number,
            )
        return

    if existing != expected:
        _raise_append_metadata_mismatch(db_path, metadata_key, existing, expected)


def validate_append_replay_contract(
    db_handler: "MemoryDBHandler",
    intended_metadata: dict,
    db_path: str,
    *,
    append: bool,
    policy_type: str = "apex",
) -> None:
    """Reject append mode when existing rows were generated under a different contract."""
    if not append:
        return

    existing_quality = db_handler.get_replay_quality_stats(policy_type=policy_type)
    existing_rows = int(existing_quality.get("count", 0))
    if existing_rows <= 0:
        return

    existing_metadata = db_handler.get_metadata()
    if not existing_metadata or not any(key.startswith("generation.") for key in existing_metadata):
        raise RuntimeError(
            f"Cannot append to {db_path}: found {existing_rows:,} existing replay rows but "
            "no generation metadata to prove they match the current run. Use overwrite mode "
            "or a new --db path."
        )

    for metadata_key in GENERATION_APPEND_CONTRACT_KEYS:
        expected = intended_metadata[metadata_key]
        if metadata_key not in existing_metadata:
            raise RuntimeError(
                f"Cannot append to {db_path}: existing replay metadata is missing "
                f"{metadata_key}. Use overwrite mode or a new --db path."
            )
        _validate_append_metadata_value(
            db_path,
            metadata_key,
            existing_metadata[metadata_key],
            expected,
        )


def build_generation_metadata(
    *,
    mode: str,
    episodes: int,
    save_interval: int,
    frame_limit: int,
    env_settings: dict,
    load_model: bool,
    model_loaded: bool | None,
    checkpoint_path: str | None,
    resolved_checkpoint_path: str | None,
    exploration_epsilon: float,
    exploration_min_epsilon: float,
    epsilon_min: float | None,
    epsilon_max: float | None,
    boost_exploration_rate: float,
    danger_exploration_rate: float,
    replay_quality_preset: str,
    replay_gates: dict[str, float],
    min_row_count: int,
    append: bool,
    config_path: str | None = None,
    num_envs: int | None = None,
) -> dict:
    """Return durable replay-generation metadata for later audits/training."""
    return {
        **build_generation_replay_contract(env_settings),
        "generation.append": bool(append),
        "generation.boost_exploration_rate": float(boost_exploration_rate),
        "generation.checkpoint_path": checkpoint_path,
        "generation.config_path": config_path,
        "generation.danger_exploration_rate": float(danger_exploration_rate),
        "generation.episodes": int(episodes),
        "generation.exploration_epsilon": float(exploration_epsilon),
        "generation.exploration_min_epsilon": float(exploration_min_epsilon),
        "generation.frame_limit": int(frame_limit),
        "generation.load_model": bool(load_model),
        "generation.min_row_count": int(resolve_min_row_count(min_row_count)),
        "generation.mode": mode,
        "generation.model_loaded": model_loaded,
        "generation.num_envs": None if num_envs is None else int(num_envs),
        "generation.quality_gates": dict(replay_gates),
        "generation.replay_quality_preset": replay_quality_preset,
        "generation.resolved_checkpoint_path": resolved_checkpoint_path,
        "generation.save_interval": int(save_interval),
    } | (
        {}
        if epsilon_min is None or epsilon_max is None
        else {
            "generation.epsilon_max": float(epsilon_max),
            "generation.epsilon_min": float(epsilon_min),
        }
    )


def build_generation_quality_metadata(quality: dict) -> dict:
    """Return durable generated-replay quality stats for later diagnosis."""
    action_counts = quality.get("action_counts", {})
    replay_quality = {
        "action_counts": [
            int(action_counts.get(action, 0)) for action in range(GameConfig.OUTPUT_SIZE)
        ],
        "active_action_count": int(quality.get("active_action_count", 0)),
        "boost_mask_fraction": float(quality.get("boost_mask_fraction", 0.0)),
        "count": int(quality.get("count", 0)),
        "dominant_action": quality.get("dominant_action"),
        "dominant_action_fraction": float(quality.get("dominant_action_fraction", 0.0)),
        "exact_mask_state_mismatch_fraction": float(
            quality.get("exact_mask_state_mismatch_fraction", 0.0)
        ),
        "invalid_current_action_fraction": float(
            quality.get("invalid_current_action_fraction", 0.0)
        ),
        "malformed_state_feature_fraction": float(
            quality.get("malformed_state_feature_fraction", 0.0)
        ),
        "multistep_fraction": float(quality.get("multistep_fraction", 0.0)),
        "nonterminal_mask_fraction": float(quality.get("nonterminal_mask_fraction", 0.0)),
        "nonterminal_trapped_next_state_fraction": float(
            quality.get("nonterminal_trapped_next_state_fraction", 0.0)
        ),
        "normalized_action_entropy": float(quality.get("normalized_action_entropy", 0.0)),
        "reward_negative_count": int(quality.get("reward_negative_count", 0)),
        "reward_positive_count": int(quality.get("reward_positive_count", 0)),
        "reward_zero_count": int(quality.get("reward_zero_count", 0)),
        "terminal_immediate_nonnegative_reward_count": int(
            quality.get("terminal_immediate_nonnegative_reward_count", 0)
        ),
        "terminal_immediate_nonnegative_reward_fraction": float(
            quality.get("terminal_immediate_nonnegative_reward_fraction", 0.0)
        ),
        "terminal_multistep_nonnegative_reward_count": int(
            quality.get("terminal_multistep_nonnegative_reward_count", 0)
        ),
        "terminal_multistep_nonnegative_reward_fraction": float(
            quality.get("terminal_multistep_nonnegative_reward_fraction", 0.0)
        ),
        "terminal_nonnegative_reward_count": int(
            quality.get("terminal_nonnegative_reward_count", 0)
        ),
        "terminal_nonnegative_reward_fraction": float(
            quality.get("terminal_nonnegative_reward_fraction", 0.0)
        ),
        "terminal_fraction": float(quality.get("terminal_fraction", 0.0)),
    }
    return {"generation.replay_quality": replay_quality}


def format_audit_replay_command(
    db_path: str,
    replay_quality_preset: str = DEFAULT_GENERATION_REPLAY_QUALITY_PRESET,
    gates: dict[str, float] | None = None,
    min_row_count: int = 0,
    expected_gamma: float | None = None,
    expected_n_step: int | None = None,
    config_path: str | None = None,
) -> str:
    """Return the audit command that should follow a generated replay run."""
    resolved_gates = dict(
        REPLAY_QUALITY_GATE_PRESETS[replay_quality_preset] if gates is None else gates
    )
    gate_args = format_reusable_gate_args(replay_quality_preset, resolved_gates)
    if gate_args.startswith("--replay-quality-preset "):
        gate_args = "--preset " + gate_args.removeprefix("--replay-quality-preset ")
    command = [
        "python",
        "src/scripts/audit_replay.py",
        "--db",
        shlex.quote(db_path),
    ]
    if config_path:
        command.extend(["--config", shlex.quote(config_path)])
    min_row_count = resolve_min_row_count(min_row_count)
    if min_row_count > 0:
        command.extend(["--min-row-count", str(min_row_count)])
    if expected_gamma is not None:
        command.extend(["--expected-gamma", f"{float(expected_gamma):g}"])
    if expected_n_step is not None:
        command.extend(["--expected-n-step", str(int(expected_n_step))])
    if gate_args:
        command.append(gate_args)
    command.append("--print-gate-args")
    return " ".join(command)


def print_generation_next_steps(
    db_path: str,
    replay_quality_preset: str,
    gates: dict[str, float],
    min_row_count: int = 0,
    config_path: str | None = None,
) -> None:
    """Print the next replay audit command after successful generation."""
    print("\nNext replay audit:")
    print(
        "   "
        + format_audit_replay_command(
            db_path,
            replay_quality_preset,
            gates,
            min_row_count,
            expected_gamma=GameConfig.APEX_GAMMA,
            expected_n_step=GameConfig.APEX_N_STEP,
            config_path=config_path,
        )
    )


def validate_replay_quality_gates(
    quality: dict,
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
) -> None:
    """Fail fast when generated replay misses requested learning-signal floors."""
    validate_min_row_count(quality, min_row_count=min_row_count, context="Generated replay")
    validate_shared_replay_quality_gates(
        quality,
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
        context="Generated replay",
    )


def validate_replay_terminal_fraction(quality: dict, min_terminal_fraction: float) -> None:
    """Fail fast when generated replay does not meet a requested terminal floor."""
    validate_replay_quality_gates(
        quality,
        min_terminal_fraction=min_terminal_fraction,
    )


def apply_generation_config(config_path, env_id=None) -> None:
    """Load YAML config inside a generator process when one was provided."""
    if not config_path:
        return

    config_obj = load_config(config_path)
    apply_config_to_game_config(config_obj)
    prefix = f"   [Env {env_id}] " if env_id is not None else ""
    print(f"{prefix}Loaded config from {config_path}")


def compute_generation_actor_epsilons(
    num_actors: int,
    base_epsilon: float,
    alpha: float,
    min_epsilon: float = 0.0,
) -> list[float]:
    """Return Ape-X-style actor epsilons with an optional exploration floor."""
    if num_actors <= 0:
        return []

    base_epsilon = clamp_epsilon(base_epsilon, "base_epsilon")
    min_epsilon = clamp_epsilon(min_epsilon, "min_epsilon")
    alpha = float(alpha)
    if num_actors == 1:
        return [max(base_epsilon, min_epsilon)]

    epsilons = []
    for actor_id in range(num_actors):
        exponent = 1 + actor_id / max(num_actors - 1, 1) * alpha
        epsilons.append(max(base_epsilon**exponent, min_epsilon))
    return epsilons


def resolve_generation_min_epsilon(
    model_loaded: bool,
    min_epsilon: float | None = None,
) -> float:
    """Choose the generated-replay exploration floor for this run."""
    if min_epsilon is not None:
        return clamp_epsilon(min_epsilon, "min_epsilon")
    if model_loaded:
        return 0.0
    # A fresh randomly initialized network has no meaningful greedy policy.
    # Keep generated replay action-diverse until a checkpoint is available.
    return 1.0


def configure_generation_exploration(
    game_state,
    base_epsilon: float,
    min_epsilon: float = 0.0,
    boost_exploration_rate: float = DEFAULT_BOOST_EXPLORATION_RATE,
    danger_exploration_rate: float = DEFAULT_DANGER_EXPLORATION_RATE,
) -> tuple[float, float]:
    """Set deterministic data-generation exploration for shared-policy snakes.

    Training is disabled during data generation, so the policy epsilon will not
    naturally decay. This keeps fresh-model collection from staying at epsilon=1
    forever and preserves Ape-X-style epsilon diversity across snakes.
    """
    base_epsilon = clamp_epsilon(base_epsilon, "base_epsilon")
    min_epsilon = clamp_epsilon(min_epsilon, "min_epsilon")
    boost_exploration_rate = clamp_epsilon(
        boost_exploration_rate,
        "boost_exploration_rate",
    )
    danger_exploration_rate = clamp_epsilon(
        danger_exploration_rate,
        "danger_exploration_rate",
    )
    policy = get_shared_apex_policy(game_state)
    if policy is not None and hasattr(policy, "epsilon"):
        policy.epsilon = base_epsilon

    ai_snakes = [snake for snake in game_state.snakes if hasattr(snake, "policy")]
    if len(ai_snakes) <= 1:
        epsilon = max(base_epsilon, min_epsilon)
        for snake in ai_snakes:
            snake.actor_epsilon = None
            snake.current_epsilon = epsilon
            snake.boost_exploration_rate = boost_exploration_rate
            snake.danger_exploration_rate = danger_exploration_rate
        return epsilon, epsilon

    epsilons = compute_generation_actor_epsilons(
        len(ai_snakes),
        base_epsilon,
        GameConfig.APEX_EPSILON_ALPHA,
        min_epsilon=min_epsilon,
    )
    for actor_id, (snake, epsilon) in enumerate(zip(ai_snakes, epsilons)):
        snake.actor_id = actor_id
        snake.num_actors = len(ai_snakes)
        snake.actor_epsilon = epsilon
        snake.current_epsilon = epsilon
        snake.boost_exploration_rate = boost_exploration_rate
        snake.danger_exploration_rate = danger_exploration_rate

    return min(epsilons), max(epsilons)


def apply_generated_priority_fallback(memories: list[dict]) -> None:
    """Assign useful export priorities when generated replay has flat priorities.

    Pure data generation runs with learn=False, so replay priorities often never
    receive TD-error updates. If all priorities are equal, rank memories by
    reward magnitude and terminal events before saving to SQLite.
    """
    if not memories:
        return

    priorities = [float(memory.get("priority", 0.0)) for memory in memories]
    if max(priorities) - min(priorities) > 1e-9:
        return

    alpha = GameConfig.APEX_PRIORITY_ALPHA
    eps = GameConfig.APEX_PRIORITY_EPSILON
    for memory in memories:
        reward_signal = abs(float(memory.get("reward", 0.0)))
        terminal_bonus = 1.0 if memory.get("done", False) else 0.0
        memory["priority"] = (reward_signal + terminal_bonus + eps) ** alpha


def is_untrainable_nonterminal_action_memory(memory: dict) -> bool:
    """Return whether one generated row contradicts its current-state action features."""
    if bool(memory.get("done", False)):
        return False
    try:
        invalid_action, _, _ = current_action_invalid_from_state(
            memory.get("action"),
            memory.get("state"),
        )
    except (TypeError, ValueError, OverflowError):
        # Leave malformed rows for the replay-quality gates to reject explicitly.
        return False
    return invalid_action


def filter_untrainable_generated_memories(memories: list[dict]) -> list[dict]:
    """Drop generated rows whose chosen action contradicts current-state features.

    A nonterminal row can legitimately have an exact next-action mask with no
    safe actions: the DQN target code treats that as a no-bootstrap target.
    Keeping those near-death rows gives replay more collision-funnel signal
    while quality gates still cap datasets dominated by trapped targets.
    """
    return [memory for memory in memories if not is_untrainable_nonterminal_action_memory(memory)]


def get_generation_checkpoint_candidates(checkpoint_path: str | None = None) -> list[Path]:
    """Return checkpoint paths that generated replay should try in order."""
    if checkpoint_path is not None:
        checkpoint_value = str(checkpoint_path).strip()
        if not checkpoint_value:
            raise ValueError("checkpoint path must not be empty")
        candidate = Path(checkpoint_value).expanduser()
        configured_candidate = Path(GameConfig.CHECKPOINT_DIR).expanduser() / checkpoint_value
        legacy_candidate = Path("saved_snakes") / checkpoint_value
        candidates = [candidate, configured_candidate, legacy_candidate]
    else:
        names = ["best_apex.pth", GameConfig.BEST_MODEL_NAME, "best_snake.pth"]
        candidates = []
        for checkpoint_dir in (Path(GameConfig.CHECKPOINT_DIR).expanduser(), Path("saved_snakes")):
            candidates.extend(checkpoint_dir / name for name in names)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates


def resolve_generation_checkpoint_path(checkpoint_path: str | None = None) -> Path | None:
    """Resolve generated-replay model input against configured checkpoint locations."""
    for candidate in get_generation_checkpoint_candidates(checkpoint_path):
        if candidate.exists():
            return candidate.resolve()
    return None


def validate_generation_checkpoint_contract(
    checkpoint: dict,
    policy,
    checkpoint_path: str = "checkpoint",
) -> None:
    """Reject generated-replay model inputs with stale target/reward semantics."""
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


def load_shared_apex_model(game_state, env_id=None, checkpoint_path: str | None = None) -> bool:
    """Load the best available Apex checkpoint into the shared policy once."""
    import torch

    from src.game.ai_snake import AISnake

    prefix = f"   [Env {env_id}] " if env_id is not None else ""
    resolved_checkpoint = resolve_generation_checkpoint_path(checkpoint_path)
    if resolved_checkpoint is None:
        tried = ", ".join(
            str(path) for path in get_generation_checkpoint_candidates(checkpoint_path)
        )
        if checkpoint_path is not None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}; tried {tried}")
        print(f"{prefix}⚠️  No checkpoint for apex, using fresh model (tried {tried})")
        return False

    for snake in game_state.snakes:
        if isinstance(snake, AISnake):
            try:
                checkpoint = torch.load(
                    resolved_checkpoint,
                    map_location=getattr(snake.policy, "device", snake.device),
                    weights_only=False,
                )
                validate_generation_checkpoint_contract(
                    checkpoint,
                    snake.policy,
                    checkpoint_path=str(resolved_checkpoint),
                )
                loaded = snake.load_state(str(resolved_checkpoint))
            except Exception as e:
                raise RuntimeError(
                    f"Could not load Apex model from {resolved_checkpoint}: {e}"
                ) from e
            if not loaded:
                raise RuntimeError(f"Could not load Apex model from {resolved_checkpoint}")
            print(f"{prefix}✅ Loaded apex model from {resolved_checkpoint}")
            return True
    raise RuntimeError(f"No Apex snake available to load model from {resolved_checkpoint}")


def save_shared_policy_memories(
    game_state, db_handler: MemoryDBHandler, snake_id: int, clear_after_save: bool = True
) -> int:
    """Persist newly collected shared-policy replay memories once."""
    policy = get_shared_apex_policy(game_state)
    if policy is None or not hasattr(policy, "prepare_memories_for_saving"):
        return 0

    memories = policy.prepare_memories_for_saving()
    if not memories:
        return 0

    filtered_memories = filter_untrainable_generated_memories(memories)
    dropped_count = len(memories) - len(filtered_memories)
    if dropped_count:
        print(
            f"   Dropped {dropped_count:,} generated replay rows with nonterminal "
            "current-action/state contradictions"
        )
    memories = filtered_memories
    if not memories:
        if clear_after_save and getattr(policy, "memory", None) is not None:
            policy.memory.clear()
        return 0

    apply_generated_priority_fallback(memories)
    save_memories_by_snake_id(db_handler, memories, default_snake_id=snake_id, policy_type="apex")

    if clear_after_save and getattr(policy, "memory", None) is not None:
        policy.memory.clear()

    return len(memories)


def get_memory_snake_id(memory: dict, default_snake_id: int) -> int:
    """Return the producer snake id for one generated memory."""
    for candidate in (
        memory.get("snake_id"),
        memory.get("stream_id"),
        default_snake_id,
    ):
        if isinstance(candidate, bool):
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return int(default_snake_id)


def get_parallel_memory_snake_id(env_id: int, local_snake_id: int, snakes_per_env: int) -> int:
    """Return a globally unique producer id for a parallel generated replay row."""
    return int(env_id) * int(snakes_per_env) + int(local_snake_id)


def group_memories_by_snake_id(memories: list, default_snake_id: int) -> dict[int, list]:
    """Group generated memories by their producer snake id."""
    grouped: dict[int, list] = {}
    for memory in memories:
        producer_snake_id = get_memory_snake_id(memory, default_snake_id)
        grouped.setdefault(producer_snake_id, []).append(memory)
    return grouped


def save_memories_by_snake_id(
    db_handler: MemoryDBHandler, memories: list, default_snake_id: int, policy_type: str = "apex"
) -> None:
    """Persist generated replay rows under their producer snake ids when available."""
    for producer_snake_id, producer_memories in group_memories_by_snake_id(
        memories,
        default_snake_id,
    ).items():
        db_handler.save_memories(producer_snake_id, producer_memories, policy_type=policy_type)


def update_generation_environment(game_state) -> None:
    """Advance replay generation with headless-training environment rules."""
    game_state.update(train_mode=True, learn=False, allow_respawn=False)


def load_generated_memories_for_merge(db_handler: MemoryDBHandler, policy_type: str):
    """Load all generated replay rows in insertion order for parallel merge."""
    try:
        return db_handler.load_memories_for_policy(
            policy_type,
            limit=None,
            order_by="id",
            include_action_masks=True,
            include_snake_ids=True,
        )
    except TypeError:
        try:
            return db_handler.load_memories_for_policy(
                policy_type,
                limit=None,
                order_by="id",
                include_action_masks=True,
            )
        except TypeError:
            return db_handler.load_memories_for_policy(
                policy_type,
                limit=None,
                order_by="id",
            )


def print_replay_quality_summary(db_handler: MemoryDBHandler, policy_type: str = "apex") -> dict:
    """Print compact replay diagnostics for generated/offline datasets."""
    from src.data.memory_db_handler import (
        format_replay_quality_stats,
        format_replay_quality_warnings,
    )

    quality = db_handler.get_replay_quality_stats(policy_type=policy_type)
    if not quality["count"]:
        return quality

    print("\n📈 Replay Quality:")
    for line in format_replay_quality_stats(quality):
        print(line)
    warnings = format_replay_quality_warnings(quality)
    if warnings:
        print("⚠️  Replay Quality Warnings:")
        for line in warnings:
            print(line)
    return quality


def collect_parallel_worker_failures(process_entries, active_envs, return_dict) -> list[str]:
    """Collect worker process failures before merging generated replay files."""
    failures = []
    failed_process_envs = set()

    for env_id, process in process_entries:
        if process.exitcode is None:
            failures.append(f"env {env_id} did not finish")
            failed_process_envs.add(env_id)
        elif process.exitcode != 0:
            failures.append(f"env {env_id} exited with code {process.exitcode}")
            failed_process_envs.add(env_id)

    for env_id, _ in active_envs:
        result = return_dict.get(env_id)
        if result is None:
            failures.append(f"env {env_id} did not report generation stats")
            continue
        if env_id in failed_process_envs:
            continue

        error = result.get("error")
        if error:
            failures.append(f"env {env_id} failed: {error}")
            continue

        raw_experiences = result.get("experiences", 0)
        if isinstance(raw_experiences, (bool, str, bytes, bytearray, memoryview)):
            failures.append(f"env {env_id} reported invalid experience count")
            continue
        try:
            experiences = int(raw_experiences)
            numeric_experiences = float(raw_experiences)
        except (TypeError, ValueError):
            failures.append(f"env {env_id} reported invalid experience count")
            continue
        if numeric_experiences != experiences:
            failures.append(f"env {env_id} reported invalid experience count")
            continue
        if experiences <= 0:
            failures.append(f"env {env_id} produced no replay memories")

    return failures


def collect_parallel_merge_failures(active_envs, output_db_path: str) -> list[str]:
    """Collect missing per-worker replay DBs before the merge step."""
    failures = []

    for env_id, _ in active_envs:
        env_db_path = get_env_database_path(output_db_path, env_id)
        if not os.path.exists(env_db_path):
            failures.append(f"env {env_id} database missing: {env_db_path}")

    return failures


def validate_generated_experience_count(total_experiences: int) -> None:
    """Fail fast when a generation run produced no trainable replay rows."""
    if total_experiences <= 0:
        raise RuntimeError("Experience generation produced no replay memories")


def validate_parallel_merge_counts(reported_experiences: int, merged_experiences: int) -> None:
    """Ensure the merged replay DB matches what workers reported saving."""
    validate_generated_experience_count(reported_experiences)
    if merged_experiences != reported_experiences:
        raise RuntimeError(
            "Parallel generation merge wrote "
            f"{merged_experiences:,} memories but workers reported "
            f"{reported_experiences:,}"
        )


def generate_single_env(
    env_id,
    episodes,
    save_interval,
    load_model,
    return_dict,
    exploration_epsilon,
    exploration_min_epsilon=None,
    config_path=None,
    max_frames=None,
    db_path=DEFAULT_REPLAY_DB,
    checkpoint_path=None,
    boost_exploration_rate=DEFAULT_BOOST_EXPLORATION_RATE,
    danger_exploration_rate=DEFAULT_DANGER_EXPLORATION_RATE,
    num_snakes=None,
    board_scale=1.0,
    food_multiplier=1.0,
):
    """Generate Apex-DQN experiences in a single environment (for parallel execution)."""
    db_handler = None
    try:
        import numpy as np

        from src.data.memory_db_handler import MemoryDBHandler
        from src.game.game_state import GameState

        apply_generation_config(config_path, env_id=env_id)
        frame_limit = resolve_generation_frame_limit(max_frames)
        env_settings = resolve_generation_environment_settings(
            num_snakes=num_snakes,
            board_scale=board_scale,
            food_multiplier=food_multiplier,
        )
        boost_exploration_rate = clamp_epsilon(
            boost_exploration_rate,
            "boost_exploration_rate",
        )
        danger_exploration_rate = clamp_epsilon(
            danger_exploration_rate,
            "danger_exploration_rate",
        )
        env_db_path = ensure_database_parent(get_env_database_path(db_path, env_id))
        remove_sqlite_files(env_db_path)
        snake_policies = ["apex"] * env_settings["num_snakes"]
        game_state = GameState(
            headless=True,
            snake_policies=snake_policies,
            num_snakes=env_settings["num_snakes"],
            board_scale=env_settings["board_scale"],
            food_multiplier=env_settings["food_multiplier"],
        )
        # Use separate database per environment to avoid SQLite concurrent write issues
        db_handler = MemoryDBHandler(db_name=env_db_path)

        # Load Apex model if requested
        model_loaded = False
        if load_model:
            model_loaded = load_shared_apex_model(
                game_state,
                env_id=env_id,
                checkpoint_path=checkpoint_path,
            )

        min_epsilon = resolve_generation_min_epsilon(model_loaded, exploration_min_epsilon)
        epsilon_min, epsilon_max = configure_generation_exploration(
            game_state,
            exploration_epsilon,
            min_epsilon=min_epsilon,
            boost_exploration_rate=boost_exploration_rate,
            danger_exploration_rate=danger_exploration_rate,
        )
        print(
            f"   [Env {env_id}] Exploration epsilon range: " f"{epsilon_min:.4f}-{epsilon_max:.4f}"
        )
        print(f"   [Env {env_id}] Boost exploration rate: {boost_exploration_rate:.2f}")
        print(f"   [Env {env_id}] Danger exploration rate: {danger_exploration_rate:.2f}")
        print(
            f"   [Env {env_id}] Environment: snakes={env_settings['num_snakes']}, "
            f"board_scale={env_settings['board_scale']:.2f}, "
            f"food_multiplier={env_settings['food_multiplier']:.2f}"
        )
        print(f"   [Env {env_id}] Frame limit: {frame_limit:,}")

        total_experiences = 0
        rewards_history = deque(maxlen=100)
        start_time = time.time()

        for episode in range(episodes):
            episode_length = 0
            last_progress_time = time.time()

            while episode_length < frame_limit and game_state.alive_snakes > 0:
                update_generation_environment(game_state)
                episode_length += 1

                # Progress indicator every 15 seconds during long episodes
                if time.time() - last_progress_time >= 15:
                    pct = 100 * episode_length / frame_limit
                    print(
                        f"   [Env {env_id}] Ep {episode+1}: "
                        f"frame {episode_length}/{frame_limit} "
                        f"({pct:.0f}%)"
                    )
                    last_progress_time = time.time()

            game_state.flush_episode_experience()
            rewards_history.append(
                float(
                    getattr(
                        game_state,
                        "episode_current_reward",
                        game_state.episode_best_reward,
                    )
                )
            )

            # Save once from the shared policy, then clear to avoid duplicates.
            saved = save_shared_policy_memories(
                game_state, db_handler, snake_id=env_id, clear_after_save=True
            )
            total_experiences += saved

            # Print progress periodically
            if (episode + 1) % save_interval == 0:
                elapsed = time.time() - start_time
                fps = total_experiences / elapsed if elapsed > 0 else 0
                avg_reward = np.mean(rewards_history) if rewards_history else 0
                print(
                    f"   [Env {env_id}] Episode {episode+1}/{episodes} | "
                    f"{total_experiences:,} exp | {fps:.0f} FPS | "
                    f"Reward: {avg_reward:.1f}"
                )

            # NOW reset for next episode (memories are already saved)
            game_state.reset()

        # Final save
        total_experiences += save_shared_policy_memories(
            game_state, db_handler, snake_id=env_id, clear_after_save=True
        )

        db_handler.close()
        return_dict[env_id] = {
            "experiences": total_experiences,
            "avg_reward": float(np.mean(rewards_history)) if rewards_history else 0,
            "time": time.time() - start_time,
            "error": None,
        }
    except Exception as e:
        if db_handler is not None:
            db_handler.close()
        print(f"   [Env {env_id}] Error: {e}")
        import traceback

        traceback.print_exc()
        return_dict[env_id] = {
            "experiences": 0,
            "avg_reward": 0,
            "time": 0,
            "error": str(e),
        }


def generate_experiences_parallel(
    episodes,
    save_interval,
    load_model,
    num_envs,
    exploration_epsilon=None,
    exploration_min_epsilon=None,
    config_path=None,
    max_frames=None,
    db_path=DEFAULT_REPLAY_DB,
    checkpoint_path=None,
    append=False,
    boost_exploration_rate=DEFAULT_BOOST_EXPLORATION_RATE,
    danger_exploration_rate=DEFAULT_DANGER_EXPLORATION_RATE,
    num_snakes=None,
    board_scale=1.0,
    food_multiplier=1.0,
    min_terminal_fraction=0.0,
    min_immediate_terminal_fraction=0.0,
    min_exact_mask_fraction=0.0,
    min_boost_mask_fraction=0.0,
    min_action_coverage_fraction=0.0,
    min_positive_reward_fraction=0.0,
    min_negative_reward_fraction=0.0,
    min_multistep_fraction=0.0,
    max_dominant_action_fraction=1.0,
    max_invalid_current_action_fraction=1.0,
    max_nonterminal_trapped_next_fraction=1.0,
    max_exact_mask_state_mismatch_fraction=1.0,
    max_malformed_state_feature_fraction=1.0,
    replay_quality_preset="none",
    min_row_count=0,
):
    """Generate Apex-DQN experiences using multiple parallel environments."""
    from src.data.memory_db_handler import MemoryDBHandler

    if exploration_epsilon is None:
        exploration_epsilon = GameConfig.APEX_EPSILON_BASE
    env_settings = resolve_generation_environment_settings(
        num_snakes=num_snakes,
        board_scale=board_scale,
        food_multiplier=food_multiplier,
    )
    boost_exploration_rate = clamp_epsilon(
        boost_exploration_rate,
        "boost_exploration_rate",
    )
    danger_exploration_rate = clamp_epsilon(
        danger_exploration_rate,
        "danger_exploration_rate",
    )
    frame_limit = resolve_generation_frame_limit(max_frames)
    output_db_path = prepare_generation_output_database(db_path, append=append)
    if append:
        append_db = MemoryDBHandler(db_name=output_db_path)
        try:
            validate_append_replay_contract(
                append_db,
                build_generation_replay_contract(env_settings),
                output_db_path,
                append=append,
            )
        finally:
            append_db.close()
    resolved_checkpoint = (
        resolve_generation_checkpoint_path(checkpoint_path) if load_model else None
    )

    episode_counts = split_episodes_across_envs(episodes, num_envs)
    active_envs = [(env_id, count) for env_id, count in enumerate(episode_counts) if count > 0]

    print(f"\n🚀 PARALLEL MODE: {num_envs} environments")
    print(f"   Active environments: {len(active_envs)}")
    print(f"   Episodes per env: {episode_counts}")
    print(f"   Total snakes: {len(active_envs) * env_settings['num_snakes']}")
    print(f"   Frame limit: {frame_limit:,}")
    print(
        "   Environment: "
        f"snakes={env_settings['num_snakes']}, "
        f"board_scale={env_settings['board_scale']:.2f}, "
        f"food_multiplier={env_settings['food_multiplier']:.2f}"
    )
    print(f"   Boost exploration rate: {boost_exploration_rate:.2f}")
    print(f"   Danger exploration rate: {danger_exploration_rate:.2f}")
    if load_model and resolved_checkpoint is not None:
        print(f"   Model checkpoint: {resolved_checkpoint}")
    print(f"   Output DB: {output_db_path}")
    print(f"   Output mode: {'append' if append else 'overwrite'}")

    manager = mp.Manager()
    return_dict = manager.dict()
    process_entries = []

    start_time = time.time()

    for env_id, env_episodes in active_envs:
        p = mp.Process(
            target=generate_single_env,
            args=(
                env_id,
                env_episodes,
                save_interval,
                load_model,
                return_dict,
                exploration_epsilon,
                exploration_min_epsilon,
                config_path,
                max_frames,
                output_db_path,
                checkpoint_path,
                boost_exploration_rate,
                danger_exploration_rate,
                env_settings["num_snakes"],
                env_settings["board_scale"],
                env_settings["food_multiplier"],
            ),
        )
        process_entries.append((env_id, p))
        p.start()
        time.sleep(0.5)  # Stagger to prevent memory spikes

    for _, p in process_entries:
        p.join()

    failures = collect_parallel_worker_failures(process_entries, active_envs, return_dict)
    if failures:
        for env_id, _ in active_envs:
            remove_sqlite_files(get_env_database_path(output_db_path, env_id))
        raise RuntimeError("Parallel generation failed:\n  - " + "\n  - ".join(failures))

    merge_failures = collect_parallel_merge_failures(active_envs, output_db_path)
    if merge_failures:
        for env_id, _ in active_envs:
            remove_sqlite_files(get_env_database_path(output_db_path, env_id))
        raise RuntimeError(
            "Parallel generation merge failed:\n  - " + "\n  - ".join(merge_failures)
        )

    # Merge all environment databases into main database
    import numpy as np

    print(f"\n💾 Merging databases from {len(active_envs)} active environments...")
    main_db = MemoryDBHandler(db_name=output_db_path)
    replay_gates = build_generation_replay_quality_gates(
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
    )
    main_db.update_metadata(
        build_generation_metadata(
            mode="parallel",
            episodes=episodes,
            save_interval=save_interval,
            frame_limit=frame_limit,
            env_settings=env_settings,
            load_model=load_model,
            model_loaded=None,
            checkpoint_path=checkpoint_path,
            resolved_checkpoint_path=str(resolved_checkpoint) if resolved_checkpoint else None,
            exploration_epsilon=exploration_epsilon,
            exploration_min_epsilon=resolve_generation_min_epsilon(
                resolved_checkpoint is not None,
                exploration_min_epsilon,
            ),
            epsilon_min=None,
            epsilon_max=None,
            boost_exploration_rate=boost_exploration_rate,
            danger_exploration_rate=danger_exploration_rate,
            replay_quality_preset=replay_quality_preset,
            replay_gates=replay_gates,
            min_row_count=min_row_count,
            append=append,
            config_path=config_path,
            num_envs=num_envs,
        )
    )
    total_merged = 0
    policy_type = "apex"
    merge_errors = []

    for env_id, _ in active_envs:
        env_db_path = get_env_database_path(output_db_path, env_id)
        if os.path.exists(env_db_path):
            env_db = None
            try:
                env_db = MemoryDBHandler(db_name=env_db_path)

                # Load and merge Apex memories. Use limit=None so large
                # generation runs do not silently drop rows during merge, and
                # order_by=id preserves each worker's generated data sequence.
                loaded_rows = load_generated_memories_for_merge(env_db, policy_type)
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
                    states, actions, rewards, next_states, dones, priorities, bootstrap_steps = (
                        loaded_rows
                    )
                    next_action_masks = None
                    snake_ids = None
                if len(states) > 0:
                    memories = []
                    snakes_per_env = env_settings["num_snakes"]
                    for i in range(len(states)):
                        memory = {
                            "state": states[i],
                            "action": actions[i],
                            "reward": rewards[i],
                            "next_state": next_states[i],
                            "done": dones[i],
                            "priority": priorities[i],
                            "bootstrap_steps": bootstrap_steps[i],
                        }
                        if next_action_masks is not None and next_action_masks[i] is not None:
                            memory["next_action_mask"] = next_action_masks[i]
                        if snake_ids is not None:
                            memory["snake_id"] = get_parallel_memory_snake_id(
                                env_id,
                                snake_ids[i],
                                snakes_per_env,
                            )
                        memories.append(memory)
                    save_memories_by_snake_id(
                        main_db,
                        memories,
                        default_snake_id=env_id,
                        policy_type=policy_type,
                    )
                    total_merged += len(memories)
                    print(f"   Merged {len(memories):,} apex memories from env {env_id}")

                remove_sqlite_files(env_db_path)
            except Exception as e:
                merge_errors.append(f"env {env_id}: {e}")
            finally:
                if env_db is not None:
                    env_db.close()

    quality = {
        "count": 0,
        "terminal_fraction": 0.0,
        "done_count": 0,
        "nonterminal_count": 0,
        "nonterminal_mask_count": 0,
        "nonterminal_mask_fraction": 0.0,
    }
    if not merge_errors:
        quality = print_replay_quality_summary(main_db, policy_type=policy_type)
        main_db.update_metadata(build_generation_quality_metadata(quality))

    main_db.close()
    if merge_errors:
        raise RuntimeError("Parallel generation merge failed:\n  - " + "\n  - ".join(merge_errors))
    validate_replay_quality_gates(
        quality,
        min_row_count=min_row_count,
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
    )
    total_exp = sum(r["experiences"] for r in return_dict.values())
    validate_parallel_merge_counts(total_exp, total_merged)

    print(f"   ✅ Total merged: {total_merged:,} memories")

    # Summarize results
    total_time = time.time() - start_time
    avg_reward = np.mean([r["avg_reward"] for r in return_dict.values()]) if return_dict else 0

    print("\n✅ Parallel generation complete!")
    print(f"   Total experiences: {total_exp:,}")
    print(f"   Combined FPS: {total_exp / total_time:.0f}")
    print(f"   Avg reward: {avg_reward:.2f}")
    print(f"   Time: {total_time/60:.1f} minutes")
    print_generation_next_steps(
        output_db_path,
        replay_quality_preset,
        replay_gates,
        min_row_count=min_row_count,
        config_path=config_path,
    )


def generate_experiences(
    episodes,
    save_interval,
    load_model=True,
    exploration_epsilon=None,
    exploration_min_epsilon=None,
    max_frames=None,
    db_path=DEFAULT_REPLAY_DB,
    checkpoint_path=None,
    append=False,
    boost_exploration_rate=DEFAULT_BOOST_EXPLORATION_RATE,
    danger_exploration_rate=DEFAULT_DANGER_EXPLORATION_RATE,
    num_snakes=None,
    board_scale=1.0,
    food_multiplier=1.0,
    min_terminal_fraction=0.0,
    min_immediate_terminal_fraction=0.0,
    min_exact_mask_fraction=0.0,
    min_boost_mask_fraction=0.0,
    min_action_coverage_fraction=0.0,
    min_positive_reward_fraction=0.0,
    min_negative_reward_fraction=0.0,
    min_multistep_fraction=0.0,
    max_dominant_action_fraction=1.0,
    max_invalid_current_action_fraction=1.0,
    max_nonterminal_trapped_next_fraction=1.0,
    max_exact_mask_state_mismatch_fraction=1.0,
    max_malformed_state_feature_fraction=1.0,
    replay_quality_preset="none",
    min_row_count=0,
    config_path=None,
):
    """Generate Apex-DQN experiences in headless mode."""
    import numpy as np

    from src.data.memory_db_handler import MemoryDBHandler
    from src.game.game_state import GameState

    frame_limit = resolve_generation_frame_limit(max_frames)
    env_settings = resolve_generation_environment_settings(
        num_snakes=num_snakes,
        board_scale=board_scale,
        food_multiplier=food_multiplier,
    )
    output_db_path = prepare_generation_output_database(db_path, append=append)
    if append:
        append_db = MemoryDBHandler(db_name=output_db_path)
        try:
            validate_append_replay_contract(
                append_db,
                build_generation_replay_contract(env_settings),
                output_db_path,
                append=append,
            )
        finally:
            append_db.close()
    snake_policies = ["apex"] * env_settings["num_snakes"]
    print(f"⚡ Generating experiences for {episodes} episodes...")
    print("   Headless mode: FAST!")
    print(f"   Frame limit: {frame_limit:,}")
    print(
        "   Environment: "
        f"snakes={env_settings['num_snakes']}, "
        f"board_scale={env_settings['board_scale']:.2f}, "
        f"food_multiplier={env_settings['food_multiplier']:.2f}"
    )
    print(f"   Output DB: {output_db_path}")
    print(f"   Output mode: {'append' if append else 'overwrite'}")

    # Initialize game with policy assignments
    game_state = GameState(
        headless=True,
        snake_policies=snake_policies,
        num_snakes=env_settings["num_snakes"],
        board_scale=env_settings["board_scale"],
        food_multiplier=env_settings["food_multiplier"],
    )
    db_handler = MemoryDBHandler(db_name=output_db_path)

    # Load best Apex model if available
    model_loaded = False
    resolved_checkpoint = (
        resolve_generation_checkpoint_path(checkpoint_path) if load_model else None
    )
    if load_model and len(game_state.snakes) > 0:
        model_loaded = load_shared_apex_model(game_state, checkpoint_path=checkpoint_path)

    if exploration_epsilon is None:
        exploration_epsilon = GameConfig.APEX_EPSILON_BASE
    boost_exploration_rate = clamp_epsilon(
        boost_exploration_rate,
        "boost_exploration_rate",
    )
    danger_exploration_rate = clamp_epsilon(
        danger_exploration_rate,
        "danger_exploration_rate",
    )
    min_epsilon = resolve_generation_min_epsilon(model_loaded, exploration_min_epsilon)
    epsilon_min, epsilon_max = configure_generation_exploration(
        game_state,
        exploration_epsilon,
        min_epsilon=min_epsilon,
        boost_exploration_rate=boost_exploration_rate,
        danger_exploration_rate=danger_exploration_rate,
    )
    print(f"   Exploration epsilon range: {epsilon_min:.4f}-{epsilon_max:.4f}")
    print(f"   Boost exploration rate: {boost_exploration_rate:.2f}")
    print(f"   Danger exploration rate: {danger_exploration_rate:.2f}")
    replay_gates = build_generation_replay_quality_gates(
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
    )
    db_handler.update_metadata(
        build_generation_metadata(
            mode="single",
            episodes=episodes,
            save_interval=save_interval,
            frame_limit=frame_limit,
            env_settings=env_settings,
            load_model=load_model,
            model_loaded=model_loaded,
            checkpoint_path=checkpoint_path,
            resolved_checkpoint_path=str(resolved_checkpoint) if resolved_checkpoint else None,
            exploration_epsilon=exploration_epsilon,
            exploration_min_epsilon=min_epsilon,
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
            boost_exploration_rate=boost_exploration_rate,
            danger_exploration_rate=danger_exploration_rate,
            replay_quality_preset=replay_quality_preset,
            replay_gates=replay_gates,
            min_row_count=min_row_count,
            append=append,
        )
    )

    # Statistics
    total_experiences = 0
    rewards_history = deque(maxlen=100)
    lengths_history = deque(maxlen=100)
    start_time = time.time()
    last_save_time = time.time()

    for episode in range(episodes):
        episode_length = 0

        # Run episode
        last_progress_time = time.time()
        while episode_length < frame_limit and game_state.alive_snakes > 0:
            update_generation_environment(game_state)
            episode_length += 1

            # Progress indicator every 10 seconds during long episodes
            if time.time() - last_progress_time >= 10:
                print(
                    f"   ⏳ Episode {episode+1}: "
                    f"frame {episode_length}/{frame_limit} "
                    f"({100 * episode_length / frame_limit:.0f}%)",
                    end="\r",
                )
                last_progress_time = time.time()

        # Convert to float to handle MPS tensors
        game_state.flush_episode_experience()
        rewards_history.append(
            float(
                getattr(
                    game_state,
                    "episode_current_reward",
                    game_state.episode_best_reward,
                )
            )
        )
        lengths_history.append(int(game_state.episode_best_length))

        # Save every episode from the shared policy and clear to prevent duplicates.
        saved_count = save_shared_policy_memories(
            game_state, db_handler, snake_id=0, clear_after_save=True
        )
        total_experiences += saved_count

        # Print progress periodically
        current_time = time.time()
        if (episode + 1) % save_interval == 0 or current_time - last_save_time >= 60:
            elapsed = current_time - start_time
            fps = total_experiences / elapsed if elapsed > 0 else 0
            avg_reward = np.mean(rewards_history) if rewards_history else 0
            avg_length = np.mean(lengths_history) if lengths_history else 0

            print(f"\n📊 Episode {episode+1}/{episodes}")
            print(f"   💾 Saved {saved_count:,} memories")
            print(f"   Experiences: {total_experiences:,}")
            print(f"   Avg reward: {avg_reward:.2f}")
            print(f"   Avg length: {avg_length:.0f}")
            print(f"   Speed: {fps:.0f} FPS")
            print(f"   Time: {elapsed/60:.1f} min")
            last_save_time = current_time

        # NOW reset for next episode (memories are already saved)
        game_state.reset()

    # Final save with buffer clear
    print("\n💾 Final save...")
    total_saved = save_shared_policy_memories(
        game_state, db_handler, snake_id=0, clear_after_save=True
    )
    total_experiences += total_saved

    if total_saved:
        print(f"✅ Saved {total_saved:,} final memories")
    else:
        print("✅ No additional final memories to save")

    # Show memory stats
    stats = db_handler.get_memory_stats()
    if stats:
        print("\n📊 Memory Statistics:")
        for policy, info in stats.items():
            if info["type"] == "sequence":
                total_transitions = info["total_transitions"]
                print(
                    f"   {policy}: {info['count']:,} sequences "
                    f"({total_transitions:,} transitions)"
                )
            else:
                print(f"   {policy}: {info['count']:,} memories")

    quality = print_replay_quality_summary(db_handler)
    db_handler.update_metadata(build_generation_quality_metadata(quality))

    total_time = time.time() - start_time
    db_handler.close()
    validate_generated_experience_count(total_experiences)
    validate_replay_quality_gates(
        quality,
        min_row_count=min_row_count,
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
    )
    print("\n✅ Experience generation complete!")
    print(f"   Total experiences: {total_experiences:,}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Avg speed: {total_experiences/total_time:.0f} FPS")
    print(f"\nGenerated replay data is ready in {output_db_path}")
    print_generation_next_steps(
        output_db_path,
        replay_quality_preset,
        replay_gates,
        min_row_count=min_row_count,
        config_path=config_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Fast Experience Generation for Apex-DQN")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_REPLAY_DB,
        help=f"Output SQLite replay database path (default: {DEFAULT_REPLAY_DB})",
    )
    parser.add_argument(
        "--save-interval", type=int, default=20, help="Save every N episodes (default: 20)"
    )
    parser.add_argument("--fresh", action="store_true", help="Start with fresh model")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing replay DB instead of replacing it",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint to load before generating replay. Relative names are resolved "
            "against the configured checkpoint dir and saved_snakes/."
        ),
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Use parallel environments for faster generation"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (auto-detected if not specified)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/default.yaml)",
    )
    parser.add_argument(
        "--exploration-epsilon",
        type=float,
        default=None,
        help=(
            "Base epsilon for generated replay. Defaults to the configured "
            "Ape-X epsilon base instead of staying at 1.0 when learning is disabled."
        ),
    )
    parser.add_argument(
        "--exploration-min-epsilon",
        type=float,
        default=None,
        help=(
            "Minimum per-snake epsilon during generated replay. Defaults to epsilon_end "
            "when no checkpoint loads and 0.0 when trained weights are loaded."
        ),
    )
    parser.add_argument(
        "--boost-exploration-rate",
        type=float,
        default=None,
        help=(
            "During generated replay exploration, probability of choosing a simulator-safe "
            "boost action when one is available (default: "
            f"{DEFAULT_BOOST_EXPLORATION_RATE})."
        ),
    )
    parser.add_argument(
        "--danger-exploration-rate",
        type=float,
        default=None,
        help=(
            "During generated replay exploration, probability of choosing a legal "
            "but simulator-unsafe action when one is available (default: "
            f"{DEFAULT_DANGER_EXPLORATION_RATE})."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Per-episode frame cap for generation. Defaults to game.max_frames from config; "
            "use a smaller value for quick bounded datasets."
        ),
    )
    parser.add_argument(
        "--num-snakes",
        type=int,
        default=None,
        help=(
            "Number of snakes in each generated replay environment. Defaults to "
            "game.num_snakes from config."
        ),
    )
    parser.add_argument(
        "--board-scale",
        type=float,
        default=None,
        help=(
            "Multiplier for generated replay arena width/height. Use values below 1.0 "
            "for smaller curriculum-style or collision-denser datasets."
        ),
    )
    parser.add_argument(
        "--food-multiplier",
        type=float,
        default=None,
        help=(
            "Multiplier for generated replay initial/max food counts. Use values below "
            "1.0 for sparser food datasets or above 1.0 for early curriculum data."
        ),
    )
    parser.add_argument(
        "--replay-env-preset",
        choices=tuple(GENERATION_ENVIRONMENT_PRESETS),
        default=DEFAULT_GENERATION_ENV_PRESET,
        help=(
            "Named generated-replay environment profile. Defaults to 'collision_dense', which "
            "uses a smaller arena with more snakes for faster terminal coverage from executable "
            "actions. Use 'default' for the full configured game shape."
        ),
    )
    parser.add_argument(
        "--replay-quality-preset",
        choices=tuple(REPLAY_QUALITY_GATE_PRESETS),
        default=DEFAULT_GENERATION_REPLAY_QUALITY_PRESET,
        help=(
            "Named replay quality gate bundle. Defaults to 'training' so generated replay "
            "must contain terminal, mask, action, reward, n-step, and trapped-next-state "
            "coverage. Use 'none' for warning-only diagnostics."
        ),
    )
    parser.add_argument(
        "--min-row-count",
        type=int,
        default=0,
        help="Optional absolute replay row-count gate for generated replay",
    )
    parser.add_argument(
        "--min-terminal-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when terminal rows are below "
            "this fraction, e.g. 0.005 for at least 0.5%% terminal replay."
        ),
    )
    parser.add_argument(
        "--min-immediate-terminal-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when one-step terminal rows "
            "are below this fraction of replay rows."
        ),
    )
    parser.add_argument(
        "--min-exact-mask-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when nonterminal rows with exact "
            "next-action masks are below this fraction."
        ),
    )
    parser.add_argument(
        "--min-boost-mask-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when fewer than this fraction "
            "of nonterminal rows have exact next-action masks allowing boost."
        ),
    )
    parser.add_argument(
        "--min-action-coverage-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when fewer than this fraction "
            "of the 6 actions appear in generated replay."
        ),
    )
    parser.add_argument(
        "--min-multistep-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when fewer than this "
            "fraction of replay rows have bootstrap_steps > 1."
        ),
    )
    parser.add_argument(
        "--min-positive-reward-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when fewer than this "
            "fraction of replay rows have positive rewards."
        ),
    )
    parser.add_argument(
        "--min-negative-reward-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when fewer than this "
            "fraction of replay rows have negative rewards."
        ),
    )
    parser.add_argument(
        "--max-dominant-action-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when one action accounts for "
            "more than this fraction of replay rows."
        ),
    )
    parser.add_argument(
        "--max-invalid-current-action-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when stored actions that are "
            "invalid under current-state danger/boost features exceed this fraction."
        ),
    )
    parser.add_argument(
        "--max-nonterminal-trapped-next-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when nonterminal next-state "
            "targets with no valid actions exceed this fraction."
        ),
    )
    parser.add_argument(
        "--max-exact-mask-state-mismatch-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when exact next-action masks "
            "disagree with next-state per-action danger features above this fraction."
        ),
    )
    parser.add_argument(
        "--max-malformed-state-feature-fraction",
        type=float,
        default=None,
        help=(
            "Optional replay quality gate. Fail generation when decoded current/next states "
            "with malformed semantic features exceed this fraction."
        ),
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.save_interval <= 0:
        raise ValueError("save-interval must be positive")
    if args.num_envs is not None and args.num_envs <= 0:
        raise ValueError("num-envs must be positive")
    if args.fresh and args.checkpoint:
        raise ValueError("--fresh cannot be combined with --checkpoint")
    validate_database_path(args.db)
    resolve_generation_frame_limit(args.max_frames)
    replay_gates = resolve_generation_replay_quality_gates(
        preset=args.replay_quality_preset,
        overrides={name: getattr(args, name) for name in REPLAY_QUALITY_GATE_ORDER},
    )
    min_row_count = resolve_min_row_count(args.min_row_count)

    worker_config_path = None

    # Load YAML configuration if specified
    if args.config:
        config_obj = load_config(args.config)
        apply_config_to_game_config(config_obj)
        worker_config_path = args.config
        print(f"📁 Loaded config from {args.config}")

    replay_env_preset = resolve_generation_environment_preset(args.replay_env_preset)
    exploration_epsilon = (
        args.exploration_epsilon
        if args.exploration_epsilon is not None
        else replay_env_preset["exploration_epsilon"]
    )
    boost_exploration_rate = clamp_epsilon(
        (
            args.boost_exploration_rate
            if args.boost_exploration_rate is not None
            else replay_env_preset["boost_exploration_rate"]
        ),
        "boost_exploration_rate",
    )
    danger_exploration_rate = clamp_epsilon(
        (
            args.danger_exploration_rate
            if args.danger_exploration_rate is not None
            else replay_env_preset["danger_exploration_rate"]
        ),
        "danger_exploration_rate",
    )
    env_settings = resolve_generation_environment_settings(
        num_snakes=(
            args.num_snakes if args.num_snakes is not None else replay_env_preset["num_snakes"]
        ),
        board_scale=(
            args.board_scale if args.board_scale is not None else replay_env_preset["board_scale"]
        ),
        food_multiplier=(
            args.food_multiplier
            if args.food_multiplier is not None
            else replay_env_preset["food_multiplier"]
        ),
    )

    print("Using policy: apex")
    if args.replay_env_preset != "default":
        print(f"Replay environment preset: {args.replay_env_preset}")

    configure_optional_torch_runtime()

    # Run in parallel or single mode
    if args.parallel:
        import psutil

        num_envs = args.num_envs or get_optimal_env_count()
        print(
            f"\n📊 System: {psutil.cpu_count(logical=False)} cores, "
            f"{psutil.virtual_memory().total / 1024**3:.1f}GB RAM"
        )
        generate_experiences_parallel(
            args.episodes,
            args.save_interval,
            load_model=not args.fresh,
            num_envs=num_envs,
            exploration_epsilon=exploration_epsilon,
            exploration_min_epsilon=args.exploration_min_epsilon,
            config_path=worker_config_path,
            max_frames=args.max_frames,
            db_path=args.db,
            checkpoint_path=args.checkpoint,
            append=args.append,
            boost_exploration_rate=boost_exploration_rate,
            danger_exploration_rate=danger_exploration_rate,
            num_snakes=env_settings["num_snakes"],
            board_scale=env_settings["board_scale"],
            food_multiplier=env_settings["food_multiplier"],
            replay_quality_preset=args.replay_quality_preset,
            min_row_count=min_row_count,
            **replay_gates,
        )
    else:
        generate_experiences(
            args.episodes,
            args.save_interval,
            load_model=not args.fresh,
            exploration_epsilon=exploration_epsilon,
            exploration_min_epsilon=args.exploration_min_epsilon,
            max_frames=args.max_frames,
            db_path=args.db,
            checkpoint_path=args.checkpoint,
            append=args.append,
            boost_exploration_rate=boost_exploration_rate,
            danger_exploration_rate=danger_exploration_rate,
            num_snakes=env_settings["num_snakes"],
            board_scale=env_settings["board_scale"],
            food_multiplier=env_settings["food_multiplier"],
            replay_quality_preset=args.replay_quality_preset,
            min_row_count=min_row_count,
            config_path=worker_config_path,
            **replay_gates,
        )


if __name__ == "__main__":
    main()
