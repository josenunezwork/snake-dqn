"""Game configuration module with immutable dataclass-based configuration.

This module provides the AppConfig system - immutable frozen dataclasses
for type-safe, validated configuration.

Usage:
    from src.core.game_config import AppConfig, initialize_config, get_config

    # Initialize from YAML
    config = initialize_config('config.yaml')

    # Or initialize with defaults
    config = initialize_config()

    # Access anywhere
    config = get_config()
    print(config.game.width)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml

# =============================================================================
# IMMUTABLE CONFIGURATION DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class GameSettings:
    """Game dimensions and basic settings."""

    width: int = 1450
    height: int = 830
    num_snakes: int = 4
    segment_size: int = 10
    wall_thickness: int = 10
    initial_food: int = 250
    max_food: int = 300
    max_frames: int = 5000
    frame_rate: int = 100
    max_length: int = 100
    num_sectors: int = 16
    min_boost_length: int = 5
    boost_length_cost_frames: int = 3  # Lose 1 segment every N boost frames
    arena_type: str = "rectangular"  # "rectangular" or "circular"
    arena_radius: int = 400
    arena_center_x: int = 725  # WIDTH // 2
    arena_center_y: int = 415  # HEIGHT // 2
    # Game-mechanics version. 1 = legacy mechanics (bit-identical to the
    # pre-blueprint behavior); 2 = blueprint §1 mechanics (boost trail pellets,
    # full cap-exempt corpse drops, size-resolved head-ons, training population
    # floor). See src/core/mechanics_constants.py for the v2 constants.
    mechanics_version: int = 1


@dataclass(frozen=True)
class NetworkSettings:
    """Neural network configuration."""

    input_size: int = 58  # Was: 57 (added 1D boost availability)
    hidden_size: int = 512
    output_size: int = 6  # 6 actions: 3 relative dirs × 2 speed modes (normal/boost)
    danger_max_distance: int = 30
    use_boundary_as_danger: bool = True
    vision_cone_radius: int = 80
    vision_cone_opacity: int = 100
    # Append 3 free-space ("don't trap yourself") features. Opt-in for backward
    # compatibility: False keeps the 58-D state; True makes get_state emit 61-D
    # (set input_size: 61 to match). See StateIndices.FREE_SPACE_*.
    use_free_space: bool = False


@dataclass(frozen=True)
class TrainingSettings:
    """Training hyperparameters."""

    batch_size: int = 128
    memory_size: int = 100000
    learning_rate: float = 0.005
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.99995
    epsilon_eval: float = 0.0
    target_update_frequency: int = 2500
    train_frequency: int = 8
    checkpoint_frequency: int = 1000
    grad_clip_norm: float = 10.0
    default_iterations: int = 10000
    save_interval: int = 1000
    log_interval: int = 100
    gameplay_epsilon: float = 0.02
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_increment: float = 0.000001


@dataclass(frozen=True)
class RewardSettings:
    """Reward configuration for easy tuning.

    Uses a unified reward scale where death must outweigh any clamped positive
    rewards collected inside the Ape-X n-step horizon, while shaping rewards
    stay proportionally smaller.
    """

    # Terminal death penalty. Keep magnitude above short max-reward streaks so
    # "score, then immediately die" does not become a positive n-step target.
    death: float = -11.0
    food_base: float = 3.0

    # Shaping rewards (scaled down for unified system)
    toward_food: float = 0.1
    away_food: float = -0.1

    # Survival reward per step
    survival: float = 0.01

    # Wall proximity penalty (continuous gradient from threshold to wall)
    wall_danger: float = -0.15
    wall_danger_threshold: float = 0.02
    wall_awareness_threshold: float = 0.15

    # Danger proximity penalties
    danger_critical: float = -0.5
    danger_high: float = -0.2
    danger_medium: float = -0.05

    # Danger thresholds
    danger_critical_threshold: float = 0.9
    danger_high_threshold: float = 0.7
    danger_medium_threshold: float = 0.5

    # Starvation mechanic
    starvation_start_frame: int = 100
    starvation_max_frames: int = 500
    starvation_max_penalty: float = 0.1

    # Kill attribution rewards (scaled by victim size)
    kill_base: float = 1.0
    kill_length_scale: float = 0.05
    kill_max: float = 5.0

    # Boost cost: per-segment reward penalty when a boost action burns a body
    # segment. Sized ~food_base so "eat then boost it away" nets ~zero, removing
    # the reward-free boost-spam exploit. One-sided negative (does not affect the
    # death-outweighs-positive-streak invariant). 0.0 = disabled (legacy default).
    boost_segment: float = 0.0

    # Mass-proportional death penalty. Death penalty is multiplied by
    # (1 + death_length_scale * normalized_length), so dying while large costs more
    # and the agent learns to protect accumulated mass. 0.0 = flat death (legacy).
    death_length_scale: float = 0.0

    # Reward clamping. Keep this at least as high as food_base/kill_max so
    # configured event rewards are not silently flattened by calculate_reward().
    reward_max: float = 5.0
    reward_min: float = -12.0

    # Reward-function version (blueprint §3.2). 1 = legacy reward; 2 = event-based
    # reward v2 (consumed by the reward implementation, not by config plumbing).
    version: int = 1


@dataclass(frozen=True)
class CheckpointSettings:
    """Checkpoint and save settings."""

    checkpoint_dir: str = "saved_snakes"
    best_model_name: str = "best_snake.pth"


@dataclass(frozen=True)
class ApexSettings:
    """Ape-X DQN configuration parameters for distributed training."""

    # Ape-X Architecture
    num_actors: int = 64  # Number of actor processes (scale with CPU cores)
    buffer_size: int = 1_000_000  # Replay buffer capacity
    batch_size: int = 512  # Larger batch for GPU efficiency

    # Actor parameters
    actor_update_freq: int = 400  # Steps between weight syncs
    epsilon_base: float = 0.4  # Base epsilon for exploration
    epsilon_alpha: float = 7.0  # Controls epsilon distribution across actors
    actor_env_num_snakes: int = 6  # Snakes per actor environment for terminal-rich replay
    actor_board_scale: float = 0.2  # Actor arena scale for collision-dense replay
    actor_food_multiplier: float = 0.5  # Actor food density multiplier
    actor_priority_mode: str = "max"  # Insert priority for new transitions: "max" or "td"

    # Opponent-pool self-play (blueprint §3.3). When opponent_pool_dir is set and
    # contains checkpoints, each actor episode assigns every non-hero snake slot
    # to the shared latest policy with p=pool_latest_fraction, otherwise to a
    # frozen policy sampled uniformly from the pool. None = pure mirror self-play.
    opponent_pool_dir: Optional[str] = None  # Directory of frozen opponent .pth checkpoints
    pool_latest_fraction: float = 0.8  # P(non-hero slot runs the latest policy)

    # Learner parameters
    learning_rate: float = 0.00025
    gamma: float = 0.99
    n_step: int = 3
    target_update_freq: int = 2500
    min_buffer_size: int = 50000  # Min experiences before learning starts

    # Priority parameters
    priority_alpha: float = 0.6  # Priority exponent
    priority_beta_start: float = 0.4  # Initial importance sampling
    priority_beta_end: float = 1.0  # Final importance sampling
    priority_epsilon: float = 1e-6  # Small constant for priorities

    # H100 optimizations
    use_compile: bool = True  # Use torch.compile
    pin_memory: bool = True  # Pin memory for faster GPU transfer


@dataclass(frozen=True)
class CurriculumSettings:
    """Curriculum learning settings for progressive difficulty."""

    enabled: bool = False
    window_size: int = 50
    phase1_threshold: float = 200.0  # avg episode length
    phase2_threshold: float = 500.0
    phase3_threshold: float = 300.0
    phase4_threshold: float = 0.5  # kill/death ratio


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration (immutable).

    This is the recommended way to manage configuration.
    Create once at startup and pass to components that need it.
    """

    game: GameSettings = field(default_factory=GameSettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    rewards: RewardSettings = field(default_factory=RewardSettings)
    checkpoint: CheckpointSettings = field(default_factory=CheckpointSettings)
    apex: ApexSettings = field(default_factory=ApexSettings)
    curriculum: CurriculumSettings = field(default_factory=CurriculumSettings)

    # Actions (immutable tuple)
    actions: Tuple[Tuple[int, int], ...] = (
        (0, -1),  # Up
        (1, 0),  # Right
        (0, 1),  # Down
        (-1, 0),  # Left
    )

    # Snake colors (immutable tuple)
    snake_colors: Tuple[Tuple[int, int, int], ...] = (
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    )

    @classmethod
    def from_defaults(cls) -> "AppConfig":
        """Create configuration with all default values."""
        config = cls()
        _validate_reward_return_contract(config)
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from YAML file through the validated loader.

        Delegates to ``config_loader.load_config`` so that file-based loading shares
        a single validated pipeline with the rest of the app: pydantic range/type
        checks, unknown-key rejection (``extra='forbid'``), training.* -> apex.*
        reconciliation, and all cross-field invariants.

        The import is local to avoid a config_loader <-> game_config import cycle.

        Args:
            path: Path to YAML configuration file

        Returns:
            AppConfig instance with values from YAML merged with defaults

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If validation or a cross-field invariant fails
        """
        from src.core.config_loader import load_config

        return load_config(path)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        from dataclasses import asdict

        return {
            "game": asdict(self.game),
            "network": asdict(self.network),
            "training": asdict(self.training),
            "rewards": asdict(self.rewards),
            "checkpoint": asdict(self.checkpoint),
            "apex": asdict(self.apex),
            "curriculum": asdict(self.curriculum),
        }

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def assert_reward_return_invariant(
    *,
    reward_max: float,
    food_base: float,
    kill_max: float,
    reward_min: float,
    death: float,
    gamma: float,
    n_step: int,
) -> None:
    """Reject reward settings that can make terminal n-step targets non-negative.

    Pure (plain values) so both config-load paths — AppConfig validation and the
    pydantic-schema validation in config_loader — share one source of truth and
    can never disagree about what's a valid reward contract.
    """
    if reward_max < food_base:
        raise ValueError("rewards.reward_max must be at least rewards.food_base")
    if reward_max < kill_max:
        raise ValueError("rewards.reward_max must be at least rewards.kill_max")
    if reward_min > death:
        raise ValueError("rewards.reward_min must be less than or equal to rewards.death")

    max_positive_then_death_return = sum(
        (gamma**step) * reward_max for step in range(max(n_step - 1, 0))
    )
    max_positive_then_death_return += (gamma ** max(n_step - 1, 0)) * death
    if max_positive_then_death_return >= 0.0:
        raise ValueError(
            "rewards.death must make max-positive-then-death n-step returns negative "
            "for apex.gamma, apex.n_step, and rewards.reward_max"
        )


def _validate_reward_return_contract(config: AppConfig) -> None:
    """Reject reward settings that can make terminal n-step targets non-negative."""
    assert_reward_return_invariant(
        reward_max=config.rewards.reward_max,
        food_base=config.rewards.food_base,
        kill_max=config.rewards.kill_max,
        reward_min=config.rewards.reward_min,
        death=config.rewards.death,
        gamma=config.apex.gamma,
        n_step=config.apex.n_step,
    )


# =============================================================================
# GLOBAL CONFIGURATION STATE
# =============================================================================

_current_config: Optional[AppConfig] = None


def initialize_config(config: Optional[AppConfig] = None) -> AppConfig:
    """Initialize the global configuration.

    This should be called once at application startup. If called multiple times,
    subsequent calls will override the previous configuration.

    Args:
        config: AppConfig instance to use. If None, uses defaults.

    Returns:
        The initialized AppConfig instance
    """
    global _current_config
    _current_config = config or AppConfig.from_defaults()
    return _current_config


def get_config() -> AppConfig:
    """Get the current global configuration.

    Returns:
        The current AppConfig instance, or defaults if not initialized
    """
    global _current_config
    if _current_config is None:
        _current_config = AppConfig.from_defaults()
    return _current_config


class StateIndices:
    """Named indices for state vector to avoid magic numbers."""

    # Direction (one-hot encoded)
    DIRECTION_START = 0
    DIRECTION_END = 4

    # Snake stats
    LENGTH_NORMALIZED = 4

    # Food features
    FOOD_REL_X = 5
    FOOD_REL_Y = 6
    FOOD_DISTANCE = 7
    FOOD_DENSITY_START = 8
    FOOD_DENSITY_END = 24  # 16 sectors

    # Danger map
    DANGER_MAP_START = 24
    DANGER_MAP_END = 40  # 16 sectors

    # Boundary distances
    BOUNDARY_LEFT = 40  # Was: 41
    BOUNDARY_RIGHT = 41  # Was: 42
    BOUNDARY_TOP = 42  # Was: 43
    BOUNDARY_BOTTOM = 43  # Was: 44

    # Enemy features (expanded from 3D to 10D)
    ENEMY_REL_X = 44
    ENEMY_REL_Y = 45
    ENEMY_REL_SIZE = 46
    ENEMY_HEADING_DX = 47
    ENEMY_HEADING_DY = 48
    ENEMY_DISTANCE_TREND = 49
    ENEMY2_REL_X = 50
    ENEMY2_REL_Y = 51
    ENEMY2_REL_SIZE = 52
    KILL_OPPORTUNITY = 53

    # Per-action danger signals
    PER_ACTION_DANGER_START = 54
    PER_ACTION_DANGER_END = 57  # 3 values: left, straight, right

    # Boost availability
    BOOST_AVAILABLE = 57

    # Free-space ("don't trap yourself") features — reachable open area per
    # relative action (left/straight/right). Appended at the END so indices
    # 0-57 are unchanged and 58-D checkpoints stay column-aligned for widening.
    # Only emitted when network.use_free_space is enabled (input_size 61).
    FREE_SPACE_START = 58
    FREE_SPACE_LEFT = 58
    FREE_SPACE_STRAIGHT = 59
    FREE_SPACE_RIGHT = 60
    FREE_SPACE_END = 61  # 3 values


def expected_input_size(network: NetworkSettings) -> int:
    """Authoritative input_size for the active state mode.

    Free-space adds 3 features to the 58-D base (61); otherwise the 58-D base
    contract applies.
    """
    return 61 if network.use_free_space else 58


def validate_network_contract(network: NetworkSettings) -> None:
    """Enforce the state-mode <-> input_size contract (single source of truth).

    input_size must equal the size the active state mode emits: 61 for
    free-space, else 58.

    Raises:
        ValueError: If the invariant is violated.
    """
    expected = expected_input_size(network)
    if network.input_size != expected:
        raise ValueError(
            f"network.input_size must be {expected} "
            f"(use_free_space={network.use_free_space}); "
            f"got {network.input_size}"
        )


# =============================================================================
# GAMECONFIG COMPATIBILITY LAYER
# =============================================================================
# This class provides backward compatibility with code that uses GameConfig.ATTRIBUTE
# syntax instead of get_config().section.attribute


class GameConfig:
    """Backward-compatible static configuration accessor.

    Provides GameConfig.ATTRIBUTE syntax that maps to AppConfig values.
    This allows existing code to continue working without modification.

    Usage:
        from src.core.game_config import GameConfig

        width = GameConfig.WIDTH  # Same as get_config().game.width
        batch_size = GameConfig.BATCH_SIZE  # Same as get_config().training.batch_size
    """

    # Game settings
    @property
    def WIDTH(self) -> int:
        return get_config().game.width

    @property
    def HEIGHT(self) -> int:
        return get_config().game.height

    @property
    def NUM_SNAKES(self) -> int:
        return get_config().game.num_snakes

    @property
    def SEGMENT_SIZE(self) -> int:
        return get_config().game.segment_size

    @property
    def WALL_THICKNESS(self) -> int:
        return get_config().game.wall_thickness

    @property
    def INITIAL_FOOD(self) -> int:
        return get_config().game.initial_food

    @property
    def MAX_FOOD(self) -> int:
        return get_config().game.max_food

    @property
    def MAX_FRAMES(self) -> int:
        return get_config().game.max_frames

    @property
    def FRAME_RATE(self) -> int:
        return get_config().game.frame_rate

    @property
    def MAX_LENGTH(self) -> int:
        return get_config().game.max_length

    @property
    def NUM_SECTORS(self) -> int:
        return get_config().game.num_sectors

    @property
    def MIN_BOOST_LENGTH(self) -> int:
        return get_config().game.min_boost_length

    @property
    def BOOST_LENGTH_COST_FRAMES(self) -> int:
        return get_config().game.boost_length_cost_frames

    @property
    def ARENA_TYPE(self) -> str:
        return get_config().game.arena_type

    @property
    def ARENA_RADIUS(self) -> int:
        return get_config().game.arena_radius

    @property
    def ARENA_CENTER_X(self) -> int:
        return get_config().game.arena_center_x

    @property
    def ARENA_CENTER_Y(self) -> int:
        return get_config().game.arena_center_y

    @property
    def MECHANICS_VERSION(self) -> int:
        return get_config().game.mechanics_version

    # Network settings
    @property
    def INPUT_SIZE(self) -> int:
        return get_config().network.input_size

    @property
    def USE_FREE_SPACE(self) -> bool:
        return get_config().network.use_free_space

    @property
    def HIDDEN_SIZE(self) -> int:
        return get_config().network.hidden_size

    @property
    def OUTPUT_SIZE(self) -> int:
        return get_config().network.output_size

    @property
    def DANGER_MAX_DISTANCE(self) -> int:
        return get_config().network.danger_max_distance

    @property
    def USE_BOUNDARY_AS_DANGER(self) -> bool:
        return get_config().network.use_boundary_as_danger

    @property
    def VISION_CONE_RADIUS(self) -> int:
        return get_config().network.vision_cone_radius

    @property
    def VISION_CONE_OPACITY(self) -> int:
        return get_config().network.vision_cone_opacity

    # Training settings
    @property
    def BATCH_SIZE(self) -> int:
        return get_config().training.batch_size

    @property
    def MEMORY_SIZE(self) -> int:
        return get_config().training.memory_size

    @property
    def LEARNING_RATE(self) -> float:
        return get_config().training.learning_rate

    @property
    def GAMMA(self) -> float:
        return get_config().training.gamma

    @property
    def EPSILON_START(self) -> float:
        return get_config().training.epsilon_start

    @property
    def EPSILON_END(self) -> float:
        return get_config().training.epsilon_end

    @property
    def EPSILON_DECAY(self) -> float:
        return get_config().training.epsilon_decay

    @property
    def EPSILON_EVAL(self) -> float:
        return get_config().training.epsilon_eval

    @property
    def TARGET_UPDATE_FREQUENCY(self) -> int:
        return get_config().training.target_update_frequency

    @property
    def TRAIN_FREQUENCY(self) -> int:
        return get_config().training.train_frequency

    @property
    def CHECKPOINT_FREQUENCY(self) -> int:
        return get_config().training.checkpoint_frequency

    @property
    def GRAD_CLIP_NORM(self) -> float:
        return get_config().training.grad_clip_norm

    @property
    def DEFAULT_ITERATIONS(self) -> int:
        return get_config().training.default_iterations

    @property
    def SAVE_INTERVAL(self) -> int:
        return get_config().training.save_interval

    @property
    def LOG_INTERVAL(self) -> int:
        return get_config().training.log_interval

    @property
    def GAMEPLAY_EPSILON(self) -> float:
        return get_config().training.gameplay_epsilon

    @property
    def PRIORITY_ALPHA(self) -> float:
        return get_config().training.priority_alpha

    @property
    def PRIORITY_BETA_START(self) -> float:
        return get_config().training.priority_beta_start

    @property
    def PRIORITY_BETA_INCREMENT(self) -> float:
        return get_config().training.priority_beta_increment

    # Reward settings
    @property
    def REWARD_DEATH(self) -> float:
        return get_config().rewards.death

    @property
    def REWARD_FOOD_BASE(self) -> float:
        return get_config().rewards.food_base

    @property
    def REWARD_TOWARD_FOOD(self) -> float:
        return get_config().rewards.toward_food

    @property
    def REWARD_AWAY_FOOD(self) -> float:
        return get_config().rewards.away_food

    @property
    def REWARD_SURVIVAL(self) -> float:
        return get_config().rewards.survival

    @property
    def REWARD_WALL_DANGER(self) -> float:
        return get_config().rewards.wall_danger

    @property
    def WALL_DANGER_THRESHOLD(self) -> float:
        return get_config().rewards.wall_danger_threshold

    @property
    def WALL_AWARENESS_THRESHOLD(self) -> float:
        return get_config().rewards.wall_awareness_threshold

    @property
    def REWARD_DANGER_CRITICAL(self) -> float:
        return get_config().rewards.danger_critical

    @property
    def REWARD_DANGER_HIGH(self) -> float:
        return get_config().rewards.danger_high

    @property
    def REWARD_DANGER_MEDIUM(self) -> float:
        return get_config().rewards.danger_medium

    @property
    def DANGER_CRITICAL_THRESHOLD(self) -> float:
        return get_config().rewards.danger_critical_threshold

    @property
    def DANGER_HIGH_THRESHOLD(self) -> float:
        return get_config().rewards.danger_high_threshold

    @property
    def DANGER_MEDIUM_THRESHOLD(self) -> float:
        return get_config().rewards.danger_medium_threshold

    @property
    def STARVATION_START_FRAME(self) -> int:
        return get_config().rewards.starvation_start_frame

    @property
    def STARVATION_MAX_FRAMES(self) -> int:
        return get_config().rewards.starvation_max_frames

    @property
    def STARVATION_MAX_PENALTY(self) -> float:
        return get_config().rewards.starvation_max_penalty

    # Kill attribution rewards
    @property
    def REWARD_KILL_BASE(self) -> float:
        return get_config().rewards.kill_base

    @property
    def REWARD_KILL_LENGTH_SCALE(self) -> float:
        return get_config().rewards.kill_length_scale

    @property
    def REWARD_BOOST_SEGMENT(self) -> float:
        return get_config().rewards.boost_segment

    @property
    def REWARD_DEATH_LENGTH_SCALE(self) -> float:
        return get_config().rewards.death_length_scale

    @property
    def REWARD_KILL_MAX(self) -> float:
        return get_config().rewards.kill_max

    @property
    def REWARD_MAX(self) -> float:
        return get_config().rewards.reward_max

    @property
    def REWARD_MIN(self) -> float:
        return get_config().rewards.reward_min

    @property
    def REWARD_VERSION(self) -> int:
        return get_config().rewards.version

    # Checkpoint settings
    @property
    def CHECKPOINT_DIR(self) -> str:
        return get_config().checkpoint.checkpoint_dir

    @property
    def BEST_MODEL_NAME(self) -> str:
        return get_config().checkpoint.best_model_name

    # Ape-X DQN settings
    @property
    def APEX_NUM_ACTORS(self) -> int:
        return get_config().apex.num_actors

    @property
    def APEX_BUFFER_SIZE(self) -> int:
        return get_config().apex.buffer_size

    @property
    def APEX_BATCH_SIZE(self) -> int:
        return get_config().apex.batch_size

    @property
    def APEX_ACTOR_UPDATE_FREQ(self) -> int:
        return get_config().apex.actor_update_freq

    @property
    def APEX_EPSILON_BASE(self) -> float:
        return get_config().apex.epsilon_base

    @property
    def APEX_EPSILON_ALPHA(self) -> float:
        return get_config().apex.epsilon_alpha

    @property
    def APEX_ACTOR_ENV_NUM_SNAKES(self) -> int:
        return get_config().apex.actor_env_num_snakes

    @property
    def APEX_ACTOR_BOARD_SCALE(self) -> float:
        return get_config().apex.actor_board_scale

    @property
    def APEX_ACTOR_FOOD_MULTIPLIER(self) -> float:
        return get_config().apex.actor_food_multiplier

    @property
    def APEX_ACTOR_PRIORITY_MODE(self) -> str:
        return get_config().apex.actor_priority_mode

    @property
    def APEX_OPPONENT_POOL_DIR(self) -> Optional[str]:
        return get_config().apex.opponent_pool_dir

    @property
    def APEX_POOL_LATEST_FRACTION(self) -> float:
        return get_config().apex.pool_latest_fraction

    @property
    def APEX_LEARNING_RATE(self) -> float:
        return get_config().apex.learning_rate

    @property
    def APEX_GAMMA(self) -> float:
        return get_config().apex.gamma

    @property
    def APEX_N_STEP(self) -> int:
        return get_config().apex.n_step

    @property
    def APEX_TARGET_UPDATE_FREQ(self) -> int:
        return get_config().apex.target_update_freq

    @property
    def APEX_MIN_BUFFER_SIZE(self) -> int:
        return get_config().apex.min_buffer_size

    @property
    def APEX_PRIORITY_ALPHA(self) -> float:
        return get_config().apex.priority_alpha

    @property
    def APEX_PRIORITY_BETA_START(self) -> float:
        return get_config().apex.priority_beta_start

    @property
    def APEX_PRIORITY_BETA_END(self) -> float:
        return get_config().apex.priority_beta_end

    @property
    def APEX_PRIORITY_EPSILON(self) -> float:
        return get_config().apex.priority_epsilon

    @property
    def APEX_USE_COMPILE(self) -> bool:
        return get_config().apex.use_compile

    @property
    def APEX_PIN_MEMORY(self) -> bool:
        return get_config().apex.pin_memory

    # Curriculum settings
    @property
    def CURRICULUM_ENABLED(self) -> bool:
        return get_config().curriculum.enabled

    @property
    def CURRICULUM_WINDOW_SIZE(self) -> int:
        return get_config().curriculum.window_size

    @property
    def CURRICULUM_PHASE1_THRESHOLD(self) -> float:
        return get_config().curriculum.phase1_threshold

    @property
    def CURRICULUM_PHASE2_THRESHOLD(self) -> float:
        return get_config().curriculum.phase2_threshold

    @property
    def CURRICULUM_PHASE3_THRESHOLD(self) -> float:
        return get_config().curriculum.phase3_threshold

    @property
    def CURRICULUM_PHASE4_THRESHOLD(self) -> float:
        return get_config().curriculum.phase4_threshold

    # Actions and colors
    @property
    def ACTIONS(self) -> Tuple[Tuple[int, int], ...]:
        return get_config().actions

    @property
    def SNAKE_COLORS(self) -> Tuple[Tuple[int, int, int], ...]:
        return get_config().snake_colors


# Create singleton instance for static-style access
GameConfig = GameConfig()
