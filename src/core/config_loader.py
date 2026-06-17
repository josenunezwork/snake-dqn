"""Configuration loader for YAML config files.

This module provides utilities for loading and saving configuration from YAML files.
Uses Pydantic for validation and AppConfig for immutable configuration.

Usage:
    # Load config and initialize globally
    from src.core.config_loader import load_and_initialize_config
    config = load_and_initialize_config('config.yaml')

    # Or load without initializing globally
    from src.core.config_loader import load_config
    config = load_config('config.yaml')
"""

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from src.core.game_config import (
    ApexSettings,
    AppConfig,
    CheckpointSettings,
    CurriculumSettings,
    GameSettings,
    NetworkSettings,
    RewardSettings,
    TrainingSettings,
    initialize_config,
)

_TRAINING_TO_APEX_RECONCILIATION = (
    ("learning_rate", "learning_rate"),
    ("gamma", "gamma"),
    ("batch_size", "batch_size"),
    ("memory_size", "buffer_size"),
    ("target_update_frequency", "target_update_freq"),
    ("priority_alpha", "priority_alpha"),
    ("priority_beta_start", "priority_beta_start"),
)

# =============================================================================
# PYDANTIC SCHEMAS FOR YAML VALIDATION
# =============================================================================


class GameSettingsSchema(BaseModel):
    """Game configuration schema for YAML loading with validation."""

    width: int = Field(default=1450, ge=100)
    height: int = Field(default=830, ge=100)
    num_snakes: int = Field(default=4, ge=1, le=8)
    segment_size: int = Field(default=10, ge=5)
    wall_thickness: int = Field(default=10, ge=1)
    initial_food: int = Field(default=250, ge=10)
    max_food: int = Field(default=300, ge=10)
    max_frames: int = Field(default=5000, ge=100)
    frame_rate: int = Field(default=100, ge=1)
    max_length: int = Field(default=100, ge=10)
    num_sectors: int = Field(default=16, ge=4)
    min_boost_length: int = Field(default=5, ge=2)
    boost_length_cost_frames: int = Field(default=3, ge=1)
    arena_type: Literal["rectangular", "circular"] = Field(default="rectangular")
    arena_radius: int = Field(default=400, ge=50)
    arena_center_x: int = Field(default=725, ge=0)
    arena_center_y: int = Field(default=415, ge=0)


class NetworkSettingsSchema(BaseModel):
    """Neural network configuration with validation."""

    input_size: int = Field(default=58, ge=1)
    hidden_size: int = Field(default=512, ge=64)
    output_size: int = Field(default=6, ge=2)
    danger_max_distance: int = Field(default=30, ge=1)
    use_boundary_as_danger: bool = Field(default=True)
    vision_cone_radius: int = Field(default=80, ge=10)
    vision_cone_opacity: int = Field(default=100, ge=0, le=255)
    use_gru: bool = Field(default=False)
    gru_hidden_size: int = Field(default=256, ge=32)
    sequence_length: int = Field(default=20, ge=1)
    burn_in_length: int = Field(default=5, ge=0)


class TrainingSettingsSchema(BaseModel):
    """Training hyperparameters with validation."""

    batch_size: int = Field(default=128, ge=1)
    memory_size: int = Field(default=100000, ge=1000)
    learning_rate: float = Field(default=0.005, gt=0, lt=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    epsilon_start: float = Field(default=1.0, ge=0, le=1)
    epsilon_end: float = Field(default=0.02, ge=0, le=1)
    epsilon_decay: float = Field(default=0.99995, gt=0, lt=1)
    epsilon_eval: float = Field(default=0.0, ge=0, le=1)
    target_update_frequency: int = Field(default=2500, ge=1)
    train_frequency: int = Field(default=8, ge=1)
    checkpoint_frequency: int = Field(default=1000, ge=1)
    grad_clip_norm: float = Field(default=10.0, gt=0)
    default_iterations: int = Field(default=10000, ge=1)
    save_interval: int = Field(default=1000, ge=1)
    log_interval: int = Field(default=100, ge=1)
    gameplay_epsilon: float = Field(default=0.02, ge=0, le=1)
    priority_alpha: float = Field(default=0.6, ge=0, le=1)
    priority_beta_start: float = Field(default=0.4, ge=0, le=1)
    priority_beta_increment: float = Field(default=0.000001, ge=0)


class RewardSettingsSchema(BaseModel):
    """Reward configuration with validation."""

    death: float = Field(default=-11.0)
    food_base: float = Field(default=3.0)
    toward_food: float = Field(default=0.1)
    away_food: float = Field(default=-0.1)
    survival: float = Field(default=0.01)
    wall_danger: float = Field(default=-0.15)
    wall_danger_threshold: float = Field(default=0.02, ge=0, le=1)
    wall_awareness_threshold: float = Field(default=0.15, ge=0, le=1)
    danger_critical: float = Field(default=-0.5)
    danger_high: float = Field(default=-0.2)
    danger_medium: float = Field(default=-0.05)
    danger_critical_threshold: float = Field(default=0.9, ge=0, le=1)
    danger_high_threshold: float = Field(default=0.7, ge=0, le=1)
    danger_medium_threshold: float = Field(default=0.5, ge=0, le=1)
    starvation_start_frame: int = Field(default=100, ge=0)
    starvation_max_frames: int = Field(default=500, ge=1)
    starvation_max_penalty: float = Field(default=0.1, ge=0)
    kill_base: float = Field(default=1.0)
    kill_length_scale: float = Field(default=0.05, ge=0)
    kill_max: float = Field(default=5.0)
    boost_segment: float = Field(default=0.0, ge=0)
    death_length_scale: float = Field(default=0.0, ge=0)
    reward_max: float = Field(default=5.0)
    reward_min: float = Field(default=-12.0)


class CheckpointSettingsSchema(BaseModel):
    """Checkpoint configuration."""

    checkpoint_dir: str = Field(default="saved_snakes")
    best_model_name: str = Field(default="best_snake.pth")


class ApexSettingsSchema(BaseModel):
    """Ape-X DQN configuration with validation."""

    num_actors: int = Field(default=64, ge=1)
    actor_update_freq: int = Field(default=400, ge=1)
    epsilon_base: float = Field(default=0.4, ge=0, le=1)
    epsilon_alpha: float = Field(default=7.0, gt=0)
    actor_env_num_snakes: int = Field(default=6, ge=1)
    actor_board_scale: float = Field(default=0.2, gt=0)
    actor_food_multiplier: float = Field(default=0.5, gt=0)
    batch_size: int = Field(default=512, ge=1)
    buffer_size: int = Field(default=1_000_000, ge=1000)
    min_buffer_size: int = Field(default=50000, ge=1)
    target_update_freq: int = Field(default=2500, ge=1)
    learning_rate: float = Field(default=0.00025, gt=0, lt=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    n_step: int = Field(default=3, ge=1)
    priority_alpha: float = Field(default=0.6, ge=0, le=1)
    priority_beta_start: float = Field(default=0.4, ge=0, le=1)
    priority_beta_end: float = Field(default=1.0, ge=0, le=1)
    priority_epsilon: float = Field(default=1e-6, ge=0)
    use_compile: bool = Field(default=True)
    pin_memory: bool = Field(default=True)


class CurriculumSettingsSchema(BaseModel):
    """Curriculum learning configuration with validation."""

    enabled: bool = Field(default=False)
    window_size: int = Field(default=50, ge=1)
    phase1_threshold: float = Field(default=200.0, gt=0)
    phase2_threshold: float = Field(default=500.0, gt=0)
    phase3_threshold: float = Field(default=300.0, gt=0)
    phase4_threshold: float = Field(default=0.5, gt=0)


class PolicyConfig(BaseModel):
    """Policy/Algorithm configuration (Apex-only)."""

    default: str = Field(default="apex")
    snake_policies: List[str] = Field(default_factory=list)

    def get_snake_policies(self, num_snakes: int) -> List[str]:
        """Get policy list for all snakes (always Apex)."""
        if self.snake_policies and len(self.snake_policies) >= num_snakes:
            return self.snake_policies[:num_snakes]
        return [self.default] * num_snakes


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")
    tensorboard_dir: str = Field(default="logs/tensorboard")


class HardwareConfig(BaseModel):
    """Hardware configuration."""

    device: str = Field(default="auto")
    num_threads: int = Field(default=4, ge=1)
    num_parallel_envs: int = Field(default=4, ge=1)


class ConfigSchema(BaseModel):
    """Complete configuration schema with validation."""

    game: GameSettingsSchema = Field(default_factory=GameSettingsSchema)
    network: NetworkSettingsSchema = Field(default_factory=NetworkSettingsSchema)
    training: TrainingSettingsSchema = Field(default_factory=TrainingSettingsSchema)
    rewards: RewardSettingsSchema = Field(default_factory=RewardSettingsSchema)
    checkpoint: CheckpointSettingsSchema = Field(default_factory=CheckpointSettingsSchema)
    apex: ApexSettingsSchema = Field(default_factory=ApexSettingsSchema)
    curriculum: CurriculumSettingsSchema = Field(default_factory=CurriculumSettingsSchema)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)


def validate_config_invariants(schema: ConfigSchema) -> None:
    """Validate cross-field invariants that individual schema fields cannot express."""
    if schema.network.input_size != 58:
        raise ValueError("network.input_size must remain 58 for the current state contract")
    if schema.network.output_size != 6:
        raise ValueError("network.output_size must remain 6 for the relative action space")
    if schema.game.max_food < schema.game.initial_food:
        raise ValueError("game.max_food must be greater than or equal to game.initial_food")
    if schema.training.memory_size < schema.training.batch_size:
        raise ValueError(
            "training.memory_size must be greater than or equal to training.batch_size"
        )
    if schema.training.epsilon_end > schema.training.epsilon_start:
        raise ValueError("training.epsilon_end must not exceed training.epsilon_start")
    if schema.network.use_gru and schema.network.burn_in_length >= schema.network.sequence_length:
        raise ValueError("network.burn_in_length must be smaller than network.sequence_length")
    if schema.rewards.reward_max < schema.rewards.food_base:
        raise ValueError("rewards.reward_max must be at least rewards.food_base")
    if schema.rewards.reward_max < schema.rewards.kill_max:
        raise ValueError("rewards.reward_max must be at least rewards.kill_max")
    if schema.rewards.reward_min > schema.rewards.death:
        raise ValueError("rewards.reward_min must be less than or equal to rewards.death")
    max_positive_then_death_return = sum(
        (schema.apex.gamma**step) * schema.rewards.reward_max
        for step in range(max(schema.apex.n_step - 1, 0))
    )
    max_positive_then_death_return += (schema.apex.gamma ** max(schema.apex.n_step - 1, 0)) * (
        schema.rewards.death
    )
    if max_positive_then_death_return >= 0.0:
        raise ValueError(
            "rewards.death must make max-positive-then-death n-step returns negative "
            "for apex.gamma, apex.n_step, and rewards.reward_max"
        )
    if schema.apex.batch_size > schema.apex.buffer_size:
        raise ValueError("apex.batch_size must not exceed apex.buffer_size")
    if schema.apex.min_buffer_size < schema.apex.batch_size:
        raise ValueError("apex.min_buffer_size must be at least apex.batch_size")
    if schema.apex.min_buffer_size > schema.apex.buffer_size:
        raise ValueError("apex.min_buffer_size must not exceed apex.buffer_size")
    if schema.apex.priority_beta_start > schema.apex.priority_beta_end:
        raise ValueError("apex.priority_beta_start must not exceed apex.priority_beta_end")


def _reconcile_training_overrides_into_apex(schema: ConfigSchema) -> ConfigSchema:
    """Copy explicit training learner knobs into Apex unless Apex explicitly set them."""
    apex_updates = {}
    training_fields_set = schema.training.model_fields_set
    apex_fields_set = schema.apex.model_fields_set

    for training_key, apex_key in _TRAINING_TO_APEX_RECONCILIATION:
        if training_key in training_fields_set and apex_key not in apex_fields_set:
            apex_updates[apex_key] = getattr(schema.training, training_key)

    if not apex_updates:
        return schema

    return schema.model_copy(
        update={
            "apex": schema.apex.model_copy(update=apex_updates),
        }
    )


# =============================================================================
# CONFIGURATION LOADING FUNCTIONS
# =============================================================================


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        Validated AppConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    if config_path is None:
        return AppConfig.from_defaults()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        yaml_config = yaml.safe_load(f) or {}

    # Validate with pydantic schema
    validated = ConfigSchema(**yaml_config)
    validated = _reconcile_training_overrides_into_apex(validated)
    validate_config_invariants(validated)

    # Convert to AppConfig dataclasses
    return _schema_to_appconfig(validated)


def load_and_initialize_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration and initialize it globally.

    This is the recommended way to load configuration at application startup.
    It loads the config from YAML, validates it, and initializes the global
    configuration state (including updating legacy GameConfig).

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        The initialized AppConfig instance
    """
    config = load_config(config_path)
    return initialize_config(config)


def _schema_to_appconfig(schema: ConfigSchema) -> AppConfig:
    """Convert validated pydantic schema to AppConfig dataclasses."""
    return AppConfig(
        game=GameSettings(
            width=schema.game.width,
            height=schema.game.height,
            num_snakes=schema.game.num_snakes,
            segment_size=schema.game.segment_size,
            wall_thickness=schema.game.wall_thickness,
            initial_food=schema.game.initial_food,
            max_food=schema.game.max_food,
            max_frames=schema.game.max_frames,
            frame_rate=schema.game.frame_rate,
            max_length=schema.game.max_length,
            num_sectors=schema.game.num_sectors,
            min_boost_length=schema.game.min_boost_length,
            boost_length_cost_frames=schema.game.boost_length_cost_frames,
            arena_type=schema.game.arena_type,
            arena_radius=schema.game.arena_radius,
            arena_center_x=schema.game.arena_center_x,
            arena_center_y=schema.game.arena_center_y,
        ),
        network=NetworkSettings(
            input_size=schema.network.input_size,
            hidden_size=schema.network.hidden_size,
            output_size=schema.network.output_size,
            danger_max_distance=schema.network.danger_max_distance,
            use_boundary_as_danger=schema.network.use_boundary_as_danger,
            vision_cone_radius=schema.network.vision_cone_radius,
            vision_cone_opacity=schema.network.vision_cone_opacity,
            use_gru=schema.network.use_gru,
            gru_hidden_size=schema.network.gru_hidden_size,
            sequence_length=schema.network.sequence_length,
            burn_in_length=schema.network.burn_in_length,
        ),
        training=TrainingSettings(
            batch_size=schema.training.batch_size,
            memory_size=schema.training.memory_size,
            learning_rate=schema.training.learning_rate,
            gamma=schema.training.gamma,
            epsilon_start=schema.training.epsilon_start,
            epsilon_end=schema.training.epsilon_end,
            epsilon_decay=schema.training.epsilon_decay,
            epsilon_eval=schema.training.epsilon_eval,
            target_update_frequency=schema.training.target_update_frequency,
            train_frequency=schema.training.train_frequency,
            checkpoint_frequency=schema.training.checkpoint_frequency,
            grad_clip_norm=schema.training.grad_clip_norm,
            default_iterations=schema.training.default_iterations,
            save_interval=schema.training.save_interval,
            log_interval=schema.training.log_interval,
            gameplay_epsilon=schema.training.gameplay_epsilon,
            priority_alpha=schema.training.priority_alpha,
            priority_beta_start=schema.training.priority_beta_start,
            priority_beta_increment=schema.training.priority_beta_increment,
        ),
        rewards=RewardSettings(
            death=schema.rewards.death,
            food_base=schema.rewards.food_base,
            toward_food=schema.rewards.toward_food,
            away_food=schema.rewards.away_food,
            survival=schema.rewards.survival,
            wall_danger=schema.rewards.wall_danger,
            wall_danger_threshold=schema.rewards.wall_danger_threshold,
            wall_awareness_threshold=schema.rewards.wall_awareness_threshold,
            danger_critical=schema.rewards.danger_critical,
            danger_high=schema.rewards.danger_high,
            danger_medium=schema.rewards.danger_medium,
            danger_critical_threshold=schema.rewards.danger_critical_threshold,
            danger_high_threshold=schema.rewards.danger_high_threshold,
            danger_medium_threshold=schema.rewards.danger_medium_threshold,
            starvation_start_frame=schema.rewards.starvation_start_frame,
            starvation_max_frames=schema.rewards.starvation_max_frames,
            starvation_max_penalty=schema.rewards.starvation_max_penalty,
            kill_base=schema.rewards.kill_base,
            kill_length_scale=schema.rewards.kill_length_scale,
            kill_max=schema.rewards.kill_max,
            boost_segment=schema.rewards.boost_segment,
            death_length_scale=schema.rewards.death_length_scale,
            reward_max=schema.rewards.reward_max,
            reward_min=schema.rewards.reward_min,
        ),
        checkpoint=CheckpointSettings(
            checkpoint_dir=schema.checkpoint.checkpoint_dir,
            best_model_name=schema.checkpoint.best_model_name,
        ),
        apex=ApexSettings(
            num_actors=schema.apex.num_actors,
            actor_update_freq=schema.apex.actor_update_freq,
            epsilon_base=schema.apex.epsilon_base,
            epsilon_alpha=schema.apex.epsilon_alpha,
            actor_env_num_snakes=schema.apex.actor_env_num_snakes,
            actor_board_scale=schema.apex.actor_board_scale,
            actor_food_multiplier=schema.apex.actor_food_multiplier,
            batch_size=schema.apex.batch_size,
            buffer_size=schema.apex.buffer_size,
            min_buffer_size=schema.apex.min_buffer_size,
            target_update_freq=schema.apex.target_update_freq,
            learning_rate=schema.apex.learning_rate,
            gamma=schema.apex.gamma,
            n_step=schema.apex.n_step,
            priority_alpha=schema.apex.priority_alpha,
            priority_beta_start=schema.apex.priority_beta_start,
            priority_beta_end=schema.apex.priority_beta_end,
            priority_epsilon=schema.apex.priority_epsilon,
            use_compile=schema.apex.use_compile,
            pin_memory=schema.apex.pin_memory,
        ),
        curriculum=CurriculumSettings(
            enabled=schema.curriculum.enabled,
            window_size=schema.curriculum.window_size,
            phase1_threshold=schema.curriculum.phase1_threshold,
            phase2_threshold=schema.curriculum.phase2_threshold,
            phase3_threshold=schema.curriculum.phase3_threshold,
            phase4_threshold=schema.curriculum.phase4_threshold,
        ),
    )


def apply_config_to_game_config(config: AppConfig) -> None:
    """Apply configuration to GameConfig (no-op for backward compatibility).

    With the new GameConfig implementation that dynamically reads from AppConfig
    via get_config(), this function is no longer needed. It's kept for backward
    compatibility with code that still calls it.

    Args:
        config: AppConfig instance (will be used via initialize_config instead)
    """
    # GameConfig now dynamically reads from get_config(), so no explicit
    # application is needed. Just ensure the config is initialized.
    initialize_config(config)


def get_config_summary(config: AppConfig) -> str:
    """Get a human-readable summary of the configuration.

    Args:
        config: AppConfig object

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 50,
        "Configuration Summary",
        "=" * 50,
        "",
        "Game:",
        f"  Dimensions: {config.game.width}x{config.game.height}",
        f"  Snakes: {config.game.num_snakes}",
        f"  Food: {config.game.initial_food} initial, {config.game.max_food} max",
        "",
        "Network:",
        (
            f"  Architecture: {config.network.input_size}->"
            f"{config.network.hidden_size}->{config.network.output_size}"
        ),
        "",
        "Local Training Loop:",
        f"  Batch Size: {config.training.batch_size}",
        f"  Memory Size: {config.training.memory_size:,}",
        f"  Train Frequency: every {config.training.train_frequency} frame(s)",
        f"  Target Update: every {config.training.target_update_frequency} update(s)",
        f"  Epsilon: {config.training.epsilon_start}→{config.training.epsilon_end}",
        "",
        "Apex DQN:",
        f"  Learning Rate: {config.apex.learning_rate}",
        f"  Gamma: {config.apex.gamma}",
        f"  N-step Returns: {config.apex.n_step}",
        f"  PER Alpha/Beta: {config.apex.priority_alpha}/{config.apex.priority_beta_start}",
        f"  Priority Epsilon: {config.apex.priority_epsilon}",
        f"  Actor Epsilon: base={config.apex.epsilon_base}, alpha={config.apex.epsilon_alpha}",
        (
            "  Actor Env: "
            f"snakes={config.apex.actor_env_num_snakes}, "
            f"board_scale={config.apex.actor_board_scale}, "
            f"food_multiplier={config.apex.actor_food_multiplier}"
        ),
        f"  Distributed Batch: {config.apex.batch_size}",
        f"  Distributed Min Buffer: {config.apex.min_buffer_size:,}",
        "",
        "Rewards:",
        f"  Death: {config.rewards.death}",
        f"  Food Base: {config.rewards.food_base}",
        "",
        "=" * 50,
    ]
    return "\n".join(lines)
