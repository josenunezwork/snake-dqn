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
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.game_config import (
    ApexSettings,
    AppConfig,
    CheckpointSettings,
    CurriculumSettings,
    GameSettings,
    NetworkSettings,
    RewardSettings,
    TrainingSettings,
    assert_reward_return_invariant,
    initialize_config,
    validate_network_contract,
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


class _StrictModel(BaseModel):
    """Base schema that rejects unknown keys.

    pydantic v2 defaults to ``extra='ignore'``, which silently drops misspelled
    keys (e.g. ``apex.gamaa``) and whole misspelled sections (``trainning:``) at
    load time — the run then proceeds on default hyperparameters with no error.
    ``extra='forbid'`` surfaces typos immediately instead. This is the same class
    of silent-dead-config bug previously fixed for the training.* vs apex.* split.
    """

    model_config = ConfigDict(extra="forbid")


class GameSettingsSchema(_StrictModel):
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
    mechanics_version: int = Field(default=1, ge=1, le=2)


class NetworkSettingsSchema(_StrictModel):
    """Neural network configuration with validation."""

    input_size: int = Field(default=58, ge=1)
    hidden_size: int = Field(default=512, ge=64)
    output_size: int = Field(default=6, ge=2)
    danger_max_distance: int = Field(default=30, ge=1)
    use_boundary_as_danger: bool = Field(default=True)
    vision_cone_radius: int = Field(default=80, ge=10)
    vision_cone_opacity: int = Field(default=100, ge=0, le=255)
    use_free_space: bool = Field(default=False)

    @model_validator(mode="after")
    def _check_state_mode(self) -> "NetworkSettingsSchema":
        """Validate input_size against the active state mode.

        Delegates to the authoritative game_config.validate_network_contract so the
        state-mode <-> input_size rule has a single source of truth. use_free_space
        appends 3 features (61-D); otherwise the 58-D base contract applies.
        """
        validate_network_contract(
            NetworkSettings(
                input_size=self.input_size,
                use_free_space=self.use_free_space,
            )
        )
        return self


class TrainingSettingsSchema(_StrictModel):
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


class RewardSettingsSchema(_StrictModel):
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
    version: int = Field(default=1, ge=1, le=2)


class CheckpointSettingsSchema(_StrictModel):
    """Checkpoint configuration."""

    checkpoint_dir: str = Field(default="saved_snakes")
    best_model_name: str = Field(default="best_snake.pth")


class ApexSettingsSchema(_StrictModel):
    """Ape-X DQN configuration with validation."""

    num_actors: int = Field(default=64, ge=1)
    actor_update_freq: int = Field(default=400, ge=1)
    epsilon_base: float = Field(default=0.4, ge=0, le=1)
    epsilon_alpha: float = Field(default=7.0, gt=0)
    actor_env_num_snakes: int = Field(default=6, ge=1)
    actor_board_scale: float = Field(default=0.2, gt=0)
    actor_food_multiplier: float = Field(default=0.5, gt=0)
    actor_priority_mode: Literal["max", "td"] = Field(default="max")
    opponent_pool_dir: Optional[str] = Field(default=None)
    pool_latest_fraction: float = Field(default=0.8, ge=0, le=1)
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


class CurriculumSettingsSchema(_StrictModel):
    """Curriculum learning configuration with validation."""

    enabled: bool = Field(default=False)
    window_size: int = Field(default=50, ge=1)
    phase1_threshold: float = Field(default=200.0, gt=0)
    phase2_threshold: float = Field(default=500.0, gt=0)
    phase3_threshold: float = Field(default=300.0, gt=0)
    phase4_threshold: float = Field(default=0.5, gt=0)


class PQNSettingsSchema(_StrictModel):
    """PQN trainer knobs (``src/scripts/train_pqn.py``).

    ``train_pqn.py`` reads this block itself via raw ``yaml.safe_load`` and maps it
    onto ``PQNConfig``; nothing here reaches :class:`AppConfig`. The section exists
    on this schema so that one mechanics-v2 config can drive both the vector
    pipeline (``load_config`` -> tournament_eval / main / apex_train) and PQN
    without ``extra='forbid'`` rejecting the file.

    Fields mirror ``PQNConfig`` by name so a typo is still rejected, but every one
    is optional and defaults to ``None`` ("not set here"): ``PQNConfig`` owns the
    defaults, and duplicating them would let the two drift apart silently.
    Field-name parity with ``PQNConfig`` is locked by test_config_pqn_block.
    """

    num_envs: Optional[int] = Field(default=None, ge=1)
    num_snakes: Optional[int] = Field(default=None, ge=1)
    rollout_len: Optional[int] = Field(default=None, ge=1)
    gamma: Optional[float] = Field(default=None, gt=0, le=1)
    lambda_: Optional[float] = Field(default=None, ge=0, le=1)
    lr: Optional[float] = Field(default=None, gt=0)
    adam_eps: Optional[float] = Field(default=None, gt=0)
    grad_clip: Optional[float] = Field(default=None, gt=0)
    minibatches: Optional[int] = Field(default=None, ge=1)
    minibatch_size: Optional[int] = Field(default=None, ge=1)
    eps_start: Optional[float] = Field(default=None, ge=0, le=1)
    eps_end: Optional[float] = Field(default=None, ge=0, le=1)
    eps_decay_steps: Optional[int] = Field(default=None, ge=1)
    hero_frac: Optional[float] = Field(default=None, ge=0, le=1)
    pool_capacity: Optional[int] = Field(default=None, ge=0)
    pool_add_interval: Optional[int] = Field(default=None, ge=1)
    death_value: Optional[float] = Field(default=None)
    kill_scale: Optional[float] = Field(default=None)
    flip_augment: Optional[bool] = Field(default=None)
    max_frames: Optional[int] = Field(default=None, ge=1)
    max_abs_q_alarm: Optional[float] = Field(default=None, gt=0)
    seed: Optional[int] = Field(default=None)
    arena_type: Optional[Literal["rectangular", "circular"]] = Field(default=None)
    mechanics_version: Optional[int] = Field(default=None, ge=1, le=2)
    reward_version: Optional[int] = Field(default=None, ge=1, le=2)
    profile: Optional[bool] = Field(default=None)

    @model_validator(mode="after")
    def _check_epsilon_order(self) -> "PQNSettingsSchema":
        """Reject an ε schedule that decays upward."""
        if (
            self.eps_start is not None
            and self.eps_end is not None
            and self.eps_end > self.eps_start
        ):
            raise ValueError("pqn.eps_end must not exceed pqn.eps_start")
        return self


class PolicyConfig(_StrictModel):
    """Policy/Algorithm configuration (Apex-only)."""

    default: str = Field(default="apex")
    snake_policies: List[str] = Field(default_factory=list)

    def get_snake_policies(self, num_snakes: int) -> List[str]:
        """Get policy list for all snakes (always Apex)."""
        if self.snake_policies and len(self.snake_policies) >= num_snakes:
            return self.snake_policies[:num_snakes]
        return [self.default] * num_snakes


class LoggingConfig(_StrictModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")
    tensorboard_dir: str = Field(default="logs/tensorboard")


class HardwareConfig(_StrictModel):
    """Hardware configuration."""

    device: str = Field(default="auto")
    num_threads: int = Field(default=4, ge=1)
    num_parallel_envs: int = Field(default=4, ge=1)


class ConfigSchema(_StrictModel):
    """Complete configuration schema with validation."""

    game: GameSettingsSchema = Field(default_factory=GameSettingsSchema)
    network: NetworkSettingsSchema = Field(default_factory=NetworkSettingsSchema)
    training: TrainingSettingsSchema = Field(default_factory=TrainingSettingsSchema)
    rewards: RewardSettingsSchema = Field(default_factory=RewardSettingsSchema)
    checkpoint: CheckpointSettingsSchema = Field(default_factory=CheckpointSettingsSchema)
    apex: ApexSettingsSchema = Field(default_factory=ApexSettingsSchema)
    curriculum: CurriculumSettingsSchema = Field(default_factory=CurriculumSettingsSchema)
    pqn: PQNSettingsSchema = Field(default_factory=PQNSettingsSchema)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)


def validate_config_invariants(schema: ConfigSchema) -> None:
    """Validate cross-field invariants that individual schema fields cannot express."""
    # State-mode <-> input_size contract: 58 = base; 61 = base + 3 free-space.
    # Delegated to the single authoritative rule in game_config so all load
    # paths stay consistent.
    validate_network_contract(
        NetworkSettings(
            input_size=schema.network.input_size,
            use_free_space=schema.network.use_free_space,
        )
    )
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
    # Reward bounds + n-step-return safety contract (shared with AppConfig validation).
    assert_reward_return_invariant(
        reward_max=schema.rewards.reward_max,
        food_base=schema.rewards.food_base,
        kill_max=schema.rewards.kill_max,
        reward_min=schema.rewards.reward_min,
        death=schema.rewards.death,
        gamma=schema.apex.gamma,
        n_step=schema.apex.n_step,
    )
    # Danger thresholds are consumed in a fixed tier order (critical > high > medium);
    # out-of-order thresholds would make higher tiers unreachable.
    if not (
        schema.rewards.danger_critical_threshold
        >= schema.rewards.danger_high_threshold
        >= schema.rewards.danger_medium_threshold
    ):
        raise ValueError(
            "rewards.danger_critical_threshold >= danger_high_threshold >= "
            "danger_medium_threshold is required (thresholds are consumed in that tier order); "
            f"got critical={schema.rewards.danger_critical_threshold}, "
            f"high={schema.rewards.danger_high_threshold}, "
            f"medium={schema.rewards.danger_medium_threshold}"
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


def resolve_yaml_device(config_path: Optional[str]) -> Optional[str]:
    """Return the ``hardware.device`` requested by a YAML config, or ``None``.

    Lets a config's ``hardware.device`` drive device selection by feeding the
    ``SNAKE_DQN_DEVICE`` env var that ``DeviceManager`` reads, instead of being a
    silent no-op. Returns ``None`` when no config is given, the ``hardware`` section
    is absent, or the value is ``"auto"`` (fall back to auto-detection). The caller
    is expected to apply this only when the CLI ``--device`` flag is unset, so the
    precedence stays CLI > YAML > auto-detect.

    Args:
        config_path: Path to a YAML config file, or None.

    Returns:
        A concrete device string (e.g. ``"cuda"``) or ``None`` to mean auto.
    """
    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    hardware = HardwareConfig(**(data.get("hardware") or {}))
    return hardware.device if hardware.device != "auto" else None


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
    """Convert the validated pydantic schema to AppConfig dataclasses.

    Each *Schema's field names match its dataclass 1:1 (locked by
    test_config_schema_dataclass_field_parity), so ``model_dump()`` maps straight
    into the dataclass constructor. A name drift raises TypeError loudly at load
    time instead of silently dropping a YAML value — the failure mode this
    project has been burned by before (dead training.* / apex.* knobs).
    """
    return AppConfig(
        game=GameSettings(**schema.game.model_dump()),
        network=NetworkSettings(**schema.network.model_dump()),
        training=TrainingSettings(**schema.training.model_dump()),
        rewards=RewardSettings(**schema.rewards.model_dump()),
        checkpoint=CheckpointSettings(**schema.checkpoint.model_dump()),
        apex=ApexSettings(**schema.apex.model_dump()),
        curriculum=CurriculumSettings(**schema.curriculum.model_dump()),
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
