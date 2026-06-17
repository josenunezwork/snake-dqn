"""Tests for game_config module."""

import pytest

from src.core.config_loader import get_config_summary, load_config
from src.core.game_config import (
    ApexSettings,
    AppConfig,
    GameConfig,
    StateIndices,
    TrainingSettings,
)


class TestGameConfig:
    """Test suite for GameConfig class."""

    def test_game_dimensions(self):
        """Test game dimension constants."""
        assert GameConfig.WIDTH > 0
        assert GameConfig.HEIGHT > 0
        assert GameConfig.SEGMENT_SIZE > 0

    def test_food_settings(self):
        """Test food-related settings."""
        assert GameConfig.INITIAL_FOOD > 0
        assert GameConfig.MAX_FOOD >= GameConfig.INITIAL_FOOD

    def test_snake_settings(self):
        """Test snake-related settings."""
        assert GameConfig.NUM_SNAKES > 0
        assert GameConfig.NUM_SNAKES <= len(GameConfig.SNAKE_COLORS)

    def test_neural_network_dimensions(self):
        """Test neural network dimensions are valid."""
        assert GameConfig.INPUT_SIZE == 58
        assert GameConfig.HIDDEN_SIZE > 0
        assert GameConfig.OUTPUT_SIZE == 6  # 3 relative dirs × 2 speed modes

    def test_actions_length(self):
        """Test that ACTIONS has 4 cardinal directions (used for state encoding)."""
        assert len(GameConfig.ACTIONS) == 4  # Cardinal dirs for state encoding
        assert GameConfig.OUTPUT_SIZE == 6  # 3 relative dirs × 2 speed modes

    def test_actions_are_valid(self):
        """Test that all actions are valid direction tuples."""
        for action in GameConfig.ACTIONS:
            assert isinstance(action, tuple)
            assert len(action) == 2
            assert action in [(0, -1), (1, 0), (0, 1), (-1, 0)]

    def test_learning_parameters(self):
        """Test learning parameters are in valid ranges."""
        assert 0 < GameConfig.LEARNING_RATE < 1
        assert 0 < GameConfig.APEX_LEARNING_RATE < 1
        assert 0 < GameConfig.GAMMA <= 1
        assert GameConfig.APEX_N_STEP >= 1
        assert 0 <= GameConfig.EPSILON_END < GameConfig.EPSILON_START <= 1
        assert 0 < GameConfig.EPSILON_DECAY < 1

    def test_reward_cap_preserves_positive_event_rewards(self):
        """Reward cap should not flatten configured food or kill rewards."""
        assert GameConfig.REWARD_MAX >= GameConfig.REWARD_FOOD_BASE
        assert GameConfig.REWARD_MAX >= GameConfig.REWARD_KILL_MAX

    def test_default_death_penalty_dominates_n_step_max_reward_streak(self):
        """Max positive rewards right before death should not make terminals positive."""
        preterminal_positive_return = sum(
            (GameConfig.APEX_GAMMA**step) * GameConfig.REWARD_MAX
            for step in range(max(GameConfig.APEX_N_STEP - 1, 0))
        )
        terminal_return = (
            preterminal_positive_return
            + (GameConfig.APEX_GAMMA ** max(GameConfig.APEX_N_STEP - 1, 0))
            * GameConfig.REWARD_DEATH
        )

        assert terminal_return < 0.0

    def test_direct_yaml_loader_rejects_weak_terminal_reward_contract(self, tmp_path):
        """AppConfig.from_yaml should not bypass reward invariants used by training."""
        config_path = tmp_path / "weak_terminal_reward.yaml"
        config_path.write_text(
            "rewards:\n  death: -7.0\n  reward_max: 5.0\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="max-positive-then-death"):
            AppConfig.from_yaml(str(config_path))

    def test_training_fast_apex_config_is_local_sized(self):
        """Fast config should not inherit production-scale distributed defaults."""
        config = load_config("configs/training_fast.yaml")

        assert config.apex.num_actors <= 4
        assert config.apex.batch_size <= 128
        assert config.apex.min_buffer_size < config.apex.buffer_size
        assert config.apex.gamma == config.training.gamma
        assert config.apex.n_step >= 1

    def test_config_summary_shows_active_apex_learning_knobs(self):
        """Configuration summaries should not hide the Apex knobs local training uses."""
        config = AppConfig(
            training=TrainingSettings(
                batch_size=64,
                memory_size=50000,
                learning_rate=0.01,
                target_update_frequency=200,
                train_frequency=1,
            ),
            apex=ApexSettings(
                learning_rate=0.0005,
                gamma=0.95,
                n_step=3,
                batch_size=128,
                min_buffer_size=2048,
                epsilon_base=0.4,
                epsilon_alpha=7.0,
                actor_env_num_snakes=6,
                actor_board_scale=0.2,
                actor_food_multiplier=0.5,
            ),
        )

        summary = get_config_summary(config)

        assert "Local Training Loop:" in summary
        assert "Apex DQN:" in summary
        assert "  Learning Rate: 0.0005" in summary
        assert "  Gamma: 0.95" in summary
        assert "  N-step Returns: 3" in summary
        assert "  Actor Env: snakes=6, board_scale=0.2, food_multiplier=0.5" in summary
        assert "  Distributed Min Buffer: 2,048" in summary
        assert "  Learning Rate: 0.01" not in summary

    def test_apex_epsilon_base_above_one_is_rejected(self, tmp_path):
        """Actor epsilon base must stay in probability range."""
        config_path = tmp_path / "bad_apex_epsilon.yaml"
        config_path.write_text("apex:\n  epsilon_base: 1.5\n", encoding="utf-8")

        with pytest.raises(Exception, match="epsilon_base"):
            load_config(str(config_path))

    def test_unknown_arena_type_is_rejected(self, tmp_path):
        """Arena typos should not silently train with rectangular mechanics."""
        config_path = tmp_path / "bad_arena.yaml"
        config_path.write_text("game:\n  arena_type: circle\n", encoding="utf-8")

        with pytest.raises(Exception, match="arena_type"):
            load_config(str(config_path))

    @pytest.mark.parametrize(
        ("yaml_text", "match"),
        [
            ("network:\n  input_size: 57\n", "network.input_size"),
            ("network:\n  output_size: 5\n", "network.output_size"),
            ("game:\n  initial_food: 200\n  max_food: 100\n", "game.max_food"),
            ("training:\n  batch_size: 2000\n  memory_size: 1000\n", "training.memory_size"),
            ("training:\n  epsilon_start: 0.1\n  epsilon_end: 0.2\n", "training.epsilon_end"),
            (
                "network:\n  use_gru: true\n  sequence_length: 5\n  burn_in_length: 5\n",
                "network.burn_in_length",
            ),
            ("rewards:\n  food_base: 3.0\n  reward_max: 2.0\n", "rewards.reward_max"),
            ("rewards:\n  kill_max: 5.0\n  reward_max: 4.0\n", "rewards.reward_max"),
            ("rewards:\n  death: -3.0\n  reward_min: -2.0\n", "rewards.reward_min"),
            (
                "rewards:\n  death: -7.0\n  reward_max: 5.0\n",
                "rewards.death",
            ),
            ("apex:\n  batch_size: 2000\n  buffer_size: 1000\n", "apex.batch_size"),
            ("apex:\n  batch_size: 512\n  min_buffer_size: 128\n", "apex.min_buffer_size"),
            ("apex:\n  buffer_size: 1000\n  min_buffer_size: 2000\n", "apex.min_buffer_size"),
            (
                "apex:\n  priority_beta_start: 0.8\n  priority_beta_end: 0.4\n",
                "apex.priority_beta_start",
            ),
        ],
    )
    def test_config_cross_field_invariants_are_rejected(self, tmp_path, yaml_text, match):
        """Config loading should fail before impossible training settings run."""
        config_path = tmp_path / "bad_invariant.yaml"
        config_path.write_text(yaml_text, encoding="utf-8")

        with pytest.raises(ValueError, match=match):
            load_config(str(config_path))

    def test_batch_size_positive(self):
        """Test batch size is positive."""
        assert GameConfig.BATCH_SIZE > 0

    def test_memory_size_positive(self):
        """Test memory size is positive."""
        assert GameConfig.MEMORY_SIZE > GameConfig.BATCH_SIZE

    def test_training_frequency_positive(self):
        """Test training frequency is positive."""
        assert GameConfig.TRAIN_FREQUENCY > 0

    def test_target_update_frequency_positive(self):
        """Test target update frequency is positive."""
        assert GameConfig.TARGET_UPDATE_FREQUENCY > 0
        assert GameConfig.APEX_TARGET_UPDATE_FREQ > 0

    def test_snake_colors_count(self):
        """Test sufficient snake colors are defined."""
        assert len(GameConfig.SNAKE_COLORS) >= GameConfig.NUM_SNAKES

    def test_snake_colors_are_rgb_tuples(self):
        """Test that all snake colors are valid RGB tuples."""
        for color in GameConfig.SNAKE_COLORS:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


class TestStateIndices:
    """Test suite for StateIndices class."""

    def test_direction_indices(self):
        """Test direction indices are valid."""
        assert StateIndices.DIRECTION_START == 0
        assert StateIndices.DIRECTION_END == 4
        assert StateIndices.DIRECTION_END - StateIndices.DIRECTION_START == 4

    def test_food_indices(self):
        """Test food indices are valid."""
        assert StateIndices.FOOD_REL_X < StateIndices.FOOD_REL_Y
        assert StateIndices.FOOD_REL_Y < StateIndices.FOOD_DISTANCE
        assert StateIndices.FOOD_DENSITY_START < StateIndices.FOOD_DENSITY_END

    def test_danger_indices(self):
        """Test danger map indices are valid."""
        assert StateIndices.DANGER_MAP_START < StateIndices.DANGER_MAP_END
        assert StateIndices.DANGER_MAP_END - StateIndices.DANGER_MAP_START == 16

    def test_boundary_indices(self):
        """Test boundary indices are valid."""
        assert StateIndices.BOUNDARY_LEFT < StateIndices.BOUNDARY_RIGHT
        assert StateIndices.BOUNDARY_RIGHT < StateIndices.BOUNDARY_TOP
        assert StateIndices.BOUNDARY_TOP < StateIndices.BOUNDARY_BOTTOM
        assert (
            StateIndices.BOUNDARY_BOTTOM == GameConfig.INPUT_SIZE - 15
        )  # 10 enemy + 3 per-action danger + 1 boost

    def test_all_indices_within_input_size(self):
        """Test that all state indices are within input size."""
        assert StateIndices.DIRECTION_END <= GameConfig.INPUT_SIZE
        assert StateIndices.FOOD_DENSITY_END <= GameConfig.INPUT_SIZE
        assert StateIndices.DANGER_MAP_END <= GameConfig.INPUT_SIZE
        assert StateIndices.BOUNDARY_BOTTOM < GameConfig.INPUT_SIZE
