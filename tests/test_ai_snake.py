"""Tests for AISnake with policy system."""

import pytest
import torch

from src.core.game_config import (
    ApexSettings,
    AppConfig,
    CheckpointSettings,
    GameConfig,
    get_config,
    initialize_config,
)
from src.game.ai_snake import AISnake
from src.game.snake import Snake
from src.training.apex_policy import ApexPolicy


@pytest.fixture
def policy():
    """Create an Apex policy for testing."""
    return ApexPolicy(GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE)


class TestAISnake:
    """Test suite for AISnake with policies."""

    def test_ai_snake_initialization_default_policy(self, policy):
        """Test AISnake initializes with default Apex policy."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )

        assert snake.policy_type == "apex"
        assert snake.policy.get_policy_name() == "apex"

    def test_ai_snake_uses_configured_checkpoint_dir(self, policy, tmp_path):
        """AI snake checkpoints should follow the active config directory."""
        original_config = get_config()
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(tmp_path)))

        try:
            initialize_config(config)
            snake = AISnake(
                id=0,
                color=(255, 0, 0),
                start_pos=(100, 100),
                segment_size=10,
                game_width=800,
                game_height=600,
                policy=policy,
            )

            assert snake.checkpoint_manager.checkpoint_dir == tmp_path
        finally:
            initialize_config(original_config)

    def test_ai_snake_initialization_explicit_policy(self, policy):
        """Test AISnake initializes with explicit policy type."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
            policy_type="apex",
        )

        assert snake.policy_type == "apex"

    def test_backward_compatibility_ai_alias(self, policy):
        """Test that snake.ai alias works for backward compatibility."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )

        assert hasattr(snake, "ai")
        assert snake.ai is snake.policy
        assert hasattr(snake.ai, "epsilon")

    def test_multi_snake_epsilon_tracks_shared_policy_decay(self, policy):
        """Factory-style local actors should not freeze exploration forever."""
        original_config = get_config()
        config = AppConfig(apex=ApexSettings(epsilon_alpha=2.0))
        try:
            initialize_config(config)
            policy.epsilon = 0.5
            snake = AISnake(
                id=1,
                color=(255, 0, 0),
                start_pos=(100, 100),
                segment_size=10,
                game_width=800,
                game_height=600,
                policy=policy,
                actor_id=1,
                num_actors=3,
            )

            assert snake.actor_epsilon is None
            assert snake.actor_epsilon_exponent == pytest.approx(2.0)
            assert snake._get_effective_epsilon() == pytest.approx(0.25)

            policy.epsilon = 0.25

            assert snake._get_effective_epsilon() == pytest.approx(0.0625)
        finally:
            initialize_config(original_config)

    def test_fixed_actor_epsilon_override_wins_over_policy_decay(self, policy):
        """Distributed actors and replay generation can still pin epsilon."""
        policy.epsilon = 0.5
        snake = AISnake(
            id=1,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
            actor_id=1,
            num_actors=3,
        )
        snake.actor_epsilon = 0.123

        assert snake._get_effective_epsilon() == pytest.approx(0.123)

    def test_get_state_returns_tensor(self, policy):
        """Test get_state returns correct shape."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )

        state = snake.get_state([], [(200, 200)])

        assert isinstance(state, torch.Tensor)
        assert state.shape == (GameConfig.INPUT_SIZE,)

    def test_epsilon_exploration_uses_only_safe_masked_actions(self, monkeypatch):
        """Random exploration should not deliberately choose known-unsafe actions."""

        class DqnPolicyStub:
            epsilon = 1.0
            use_gru = False

            def dqn(self, state):
                return torch.tensor([[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]])

            def select_action(self, state):
                raise AssertionError("masked DQN branch should handle action selection")

        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(400, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=DqnPolicyStub(),
        )
        snake.direction = (1, 0)
        enemy = Snake(1, (0, 255, 0), (410, 300), 10, 800, 600)
        enemy.is_alive = True
        choice_inputs = []

        monkeypatch.setattr("src.game.ai_snake.random.random", lambda: 0.0)

        def choose_first(options):
            choice_inputs.append(list(options))
            assert 1 not in options
            return options[0]

        monkeypatch.setattr("src.game.ai_snake.random.choice", choose_first)

        snake.update([snake, enemy], [(100, 100)])

        assert choice_inputs
        assert snake._pre_collision_action in choice_inputs[0]
        assert snake._pre_collision_action != 1

    def test_danger_exploration_can_sample_known_unsafe_actions(self, monkeypatch):
        """Generated replay can intentionally sample unsafe actions for terminal examples."""

        class DqnPolicyStub:
            epsilon = 1.0
            use_gru = False

            def dqn(self, state):
                return torch.zeros((1, GameConfig.OUTPUT_SIZE))

            def select_action(self, state):
                raise AssertionError("danger random branch should choose directly")

        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(400, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=DqnPolicyStub(),
        )
        snake.direction = (1, 0)
        snake.danger_exploration_rate = 1.0
        enemy = Snake(1, (0, 255, 0), (410, 300), 10, 800, 600)
        enemy.is_alive = True
        choice_inputs = []
        random_values = iter([0.0, 0.0])

        monkeypatch.setattr("src.game.ai_snake.random.random", lambda: next(random_values))

        def choose_first(options):
            choice_inputs.append(list(options))
            return options[0]

        monkeypatch.setattr("src.game.ai_snake.random.choice", choose_first)

        snake.update([snake, enemy], [(100, 100)], record_q_values=False)

        assert choice_inputs == [[1]]
        assert snake._pre_collision_action == 1

    def test_boost_exploration_bias_prefers_safe_boost_actions(self, monkeypatch):
        """Generated replay can bias random exploration toward simulator-safe boost moves."""

        class DqnPolicyStub:
            epsilon = 1.0
            use_gru = False

            def dqn(self, state):
                return torch.zeros((1, GameConfig.OUTPUT_SIZE))

            def select_action(self, state):
                raise AssertionError("safe random branch should choose directly")

        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(400, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=DqnPolicyStub(),
        )
        snake.direction = (1, 0)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(400 - 10 * idx, 300) for idx in range(GameConfig.MIN_BOOST_LENGTH)]
        snake.boost_exploration_rate = 1.0
        random_values = iter([0.0, 0.0, 0.0])
        choice_inputs = []

        monkeypatch.setattr("src.game.ai_snake.random.random", lambda: next(random_values))

        def choose_first(options):
            choice_inputs.append(list(options))
            return options[0]

        monkeypatch.setattr("src.game.ai_snake.random.choice", choose_first)

        snake.update([snake], [(100, 100)], record_q_values=False)

        assert choice_inputs
        assert all(action >= 3 for action in choice_inputs[0])
        assert snake._pre_collision_action >= 3

    def test_greedy_selection_hard_masks_unsafe_q_values(self):
        """Greedy action selection should not execute invalid actions with high Q-values."""

        class DqnPolicyStub:
            epsilon = 0.0
            use_gru = False

            def dqn(self, state):
                return torch.tensor([[1.0, 10_000.0, 2.0, 3_000.0, 4_000.0, 5_000.0]])

            def select_action(self, state):
                raise AssertionError("masked DQN branch should handle action selection")

        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(400, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=DqnPolicyStub(),
        )
        snake.direction = (1, 0)
        enemy = Snake(1, (0, 255, 0), (410, 300), 10, 800, 600)
        enemy.is_alive = True

        snake.update([snake, enemy], [(100, 100)])

        assert snake._pre_collision_action == 2

    def test_all_safe_greedy_selection_reuses_computed_q_values(self):
        """Open-space greedy selection should not call policy.select_action again."""

        class DqnPolicyStub:
            epsilon = 0.0
            use_gru = False

            def __init__(self):
                self.dqn_calls = 0

            def dqn(self, state):
                self.dqn_calls += 1
                return torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])

            def select_action(self, state):
                raise AssertionError("AISnake should reuse the already-computed Q-values")

        policy_stub = DqnPolicyStub()
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(400, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy_stub,
        )
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(400, 300), (390, 300), (380, 300), (370, 300), (360, 300)]

        snake.update([snake], [(100, 100)])

        assert policy_stub.dqn_calls == 1
        assert snake._pre_collision_action == 5

    def test_headless_exploration_can_skip_display_only_q_values(self, monkeypatch):
        """Headless random exploration should not run a discarded feedforward pass."""

        class DqnPolicyStub:
            epsilon = 1.0
            use_gru = False

            def dqn(self, state):
                raise AssertionError("random headless exploration should skip DQN inference")

            def select_action(self, state):
                raise AssertionError("safe random branch should choose directly")

        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(400, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=DqnPolicyStub(),
        )
        snake.direction = (1, 0)
        snake.last_q_values = [1.0, 2.0, 3.0]

        monkeypatch.setattr("src.game.ai_snake.random.random", lambda: 0.0)
        monkeypatch.setattr("src.game.ai_snake.random.choice", lambda options: options[0])

        snake.update([snake], [(100, 100)], record_q_values=False)

        assert snake._pre_collision_action == 0
        assert snake.last_q_values is None

    def test_policy_fallback_receives_exact_safe_action_mask(self):
        """AISnake fallback policy selection should keep simulator-safe boost options."""

        class PolicyStub:
            epsilon = 0.0
            use_gru = False

            def __init__(self):
                self.seen_action_mask = None

            def select_action(self, state, action_mask=None):
                self.seen_action_mask = action_mask
                return 4

        policy_stub = PolicyStub()
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(400, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy_stub,
        )
        snake.direction = (1, 0)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(400 - i * 10, 300) for i in range(snake.length)]

        snake.update([snake], [(100, 100)])

        assert policy_stub.seen_action_mask is not None
        assert bool(policy_stub.seen_action_mask[4]) is True
        assert snake._pre_collision_action == 4

    def test_reward_step_records_exact_next_action_mask(self):
        """Replay should capture simulator-safe next actions, including boost distance."""

        class MemoryStub:
            def __init__(self):
                self.add_calls = []

            def add(self, *args, **kwargs):
                self.add_calls.append((args, kwargs))

        class PolicyStub:
            epsilon = 0.0
            use_gru = False
            training = True
            device = torch.device("cpu")

            def __init__(self):
                self.memory = MemoryStub()
                self.total_reward = 0.0

        policy_stub = PolicyStub()
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(785, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy_stub,
        )
        snake.direction = (1, 0)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(785 - i * 10, 300) for i in range(snake.length)]
        snake._pre_collision_state = snake.get_state([snake], [(100, 100)])
        snake._pre_collision_action = 1

        snake.compute_reward_and_train([snake], [(100, 100)], collided=False)

        assert snake.last_next_action_mask is not None
        assert bool(snake.last_next_action_mask[1]) is True
        assert bool(snake.last_next_action_mask[4]) is False
        _, kwargs = policy_stub.memory.add_calls[0]
        assert torch.equal(kwargs["next_action_mask"], snake.last_next_action_mask)

    def test_reward_step_records_empty_exact_next_action_mask_when_trapped(self):
        """Replay should store exact empty masks instead of falling back to approximation."""

        class MemoryStub:
            def __init__(self):
                self.add_calls = []

            def add(self, *args, **kwargs):
                self.add_calls.append((args, kwargs))

        class PolicyStub:
            epsilon = 0.0
            use_gru = False
            training = True
            device = torch.device("cpu")

            def __init__(self):
                self.memory = MemoryStub()
                self.total_reward = 0.0

        policy_stub = PolicyStub()
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(400, 300),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy_stub,
        )
        snake.direction = (1, 0)
        blockers = [
            Snake(1, (0, 255, 0), (400, 290), 10, 800, 600),
            Snake(2, (0, 0, 255), (410, 300), 10, 800, 600),
            Snake(3, (255, 255, 0), (400, 310), 10, 800, 600),
        ]
        other_snakes = [snake, *blockers]
        snake._pre_collision_state = snake.get_state(other_snakes, [(100, 100)])
        snake._pre_collision_action = 1

        assert snake._get_safe_actions(other_snakes, allow_fallback=False) == []
        assert snake._get_safe_actions(other_snakes) == [0, 1, 2]

        snake.compute_reward_and_train(other_snakes, [(100, 100)], collided=False)

        assert snake.last_next_action_mask is not None
        assert snake.last_next_action_mask.tolist() == [False, False, False, False, False, False]
        _, kwargs = policy_stub.memory.add_calls[0]
        assert torch.equal(kwargs["next_action_mask"], snake.last_next_action_mask)

    def test_gru_replay_buffer_keeps_exact_next_action_mask(self):
        """GRU snake replay should keep simulator masks in the episode buffer."""

        class MemoryStub:
            def add_episode(self, episode):
                raise AssertionError("nonterminal transition should stay pending")

        class PolicyStub:
            epsilon = 0.0
            use_gru = True
            training = True
            device = torch.device("cpu")

            def __init__(self):
                self.memory = MemoryStub()
                self._episode_buffers = {}
                self.total_reward = 0.0

            def reset_hidden(self, snake_id):
                pass

        policy_stub = PolicyStub()
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy_stub,
        )
        state = snake.get_state([snake], [(200, 200)])
        next_state = snake.get_state([snake], [(210, 200)])
        next_action_mask = torch.tensor([False, True, False, False, True, False])

        snake._add_experience(
            state,
            action=1,
            reward=0.5,
            next_state=next_state,
            done=False,
            next_action_mask=next_action_mask,
        )

        stored_transition = policy_stub._episode_buffers[0][0]
        assert len(stored_transition) == 6
        assert torch.equal(stored_transition[5], next_action_mask)

    def test_calculate_reward_death(self, policy):
        """Test death returns large negative reward."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )

        state = snake.get_state([], [(200, 200)])
        reward = snake.calculate_reward(
            ate_food=False,
            collided=True,
            old_state=state,
            new_state=None,
            other_snakes=[],
            food=[(200, 200)],
        )

        assert reward == GameConfig.REWARD_DEATH

    def test_calculate_reward_eating(self, policy):
        """Test eating returns positive reward."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )

        state = snake.get_state([], [(200, 200)])
        reward = snake.calculate_reward(
            ate_food=True,
            collided=False,
            old_state=state,
            new_state=state,
            other_snakes=[],
            food=[(200, 200)],
        )

        assert reward == GameConfig.REWARD_FOOD_BASE  # Flat food reward, no length scaling

    def test_total_reward_property(self, policy):
        """Test total_reward property."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )

        assert snake.total_reward == 0
        snake._total_reward = 100.0
        assert snake.total_reward == 100.0

    def test_soft_reset_clears_enemy_trend_tracker(self, policy):
        """Episode resets should not carry enemy trend memory across episodes."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )
        snake._prev_nearest_enemy_dist = 50.0
        snake._prev_nearest_enemy_id = 7

        snake.soft_reset((200, 200))

        assert snake._prev_nearest_enemy_dist == float("inf")
        assert snake._prev_nearest_enemy_id is None

    def test_policy_state_dict_serialization(self, policy):
        """Test policy state can be serialized and loaded."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )

        # Get state dict
        state_dict = snake.policy.get_state_dict()

        assert "policy_type" in state_dict
        assert state_dict["policy_type"] == "apex"
        assert "dqn_state_dict" in state_dict
        assert "target_dqn_state_dict" in state_dict
        assert "epsilon" in state_dict

    def test_color_to_name(self, policy):
        """Test color name mapping."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )

        assert snake.color_to_name((255, 0, 0)) == "Red"
        assert snake.color_to_name((0, 255, 0)) == "Green"
        assert snake.color_to_name((0, 0, 255)) == "Blue"


class TestPolicyCheckpoints:
    """Test checkpoint saving/loading with policies."""

    def test_checkpoint_manager_saves_policy_type(self):
        """Test CheckpointManager includes policy_type."""
        import tempfile

        from src.model.checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(tmpdir, verbose=False)

            # Create policy and get state dict
            policy = ApexPolicy(
                GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE
            )
            state_dict = policy.get_state_dict()

            # Add metadata
            checkpoint_data = {**state_dict, "total_reward": 100.0, "iteration": 1000}

            # Save
            cm.save_checkpoint_dict(checkpoint_data, "test.pth")

            # Load and verify
            loaded = cm.load_checkpoint(torch.device("cpu"), "test.pth", strict=False)
            assert loaded is not None
            assert loaded["policy_type"] == "apex"
            assert loaded["total_reward"] == 100.0

    def test_checkpoint_backward_compatibility(self):
        """Test loading old checkpoints without policy_type."""
        import tempfile

        from src.model.checkpoint_manager import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(tmpdir, verbose=False)

            # Create legacy checkpoint (no policy_type)
            policy = ApexPolicy(
                GameConfig.INPUT_SIZE, GameConfig.HIDDEN_SIZE, GameConfig.OUTPUT_SIZE
            )
            legacy_checkpoint = {
                "dqn_state_dict": policy.dqn.state_dict(),
                "target_dqn_state_dict": policy.target_dqn.state_dict(),
                "optimizer_state_dict": policy.optimizer.state_dict(),
                "epsilon": 0.5,
            }

            # Save using old format
            import torch

            torch.save(legacy_checkpoint, f"{tmpdir}/legacy.pth")

            # Load
            loaded = cm.load_checkpoint(torch.device("cpu"), "legacy.pth", strict=False)
            assert loaded is not None
            # Should default to 'apex' if not specified
            policy_type = loaded.get("policy_type", "apex")
            assert policy_type == "apex"

    def test_ai_snake_save_state_honors_explicit_path(self, policy, tmp_path):
        """save_state should write to the provided directory, not just the filename."""
        snake = AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=policy,
        )
        checkpoint_path = tmp_path / "nested" / "snake.pth"

        snake.save_state(str(checkpoint_path))

        assert checkpoint_path.exists()


class TestMultiPolicyGame:
    """Test multi-policy gameplay scenarios."""

    def test_mixed_policy_game_state(self):
        """Test game with different policies per snake."""
        from src.game.game_state import GameState

        # When more policies are available, test mixed
        # For now, test all Apex
        policies = ["apex"] * 4
        game = GameState(headless=True, snake_policies=policies)

        assert len(game.snakes) == 4
        for i, snake in enumerate(game.snakes):
            assert snake.policy_type == policies[i]

    def test_game_update_with_policies(self):
        """Test game update works with policy system."""
        from src.game.game_state import GameState

        game = GameState(headless=True, snake_policies=["apex", "apex"])
        initial_frame = game.frame

        # Update game
        game.update(train_mode=False)

        assert game.frame == initial_frame + 1
        # Snakes should still be alive (no collisions yet)
        assert game.alive_snakes > 0
