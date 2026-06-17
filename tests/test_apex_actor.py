"""Tests for Ape-X Actor component."""

import queue
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from src.core.game_config import GameConfig, StateIndices
from src.training.apex_actor import (
    ApexActor,
    Experience,
    compute_actor_epsilon,
    spawn_actors,
    stop_actors,
)
from src.training.apex_buffer import ActorBufferClient, BufferProcess


class FixedQ(torch.nn.Module):
    """Return deterministic Q-values for target-selection tests."""

    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.to(states.device).expand(states.shape[0], -1)


def _state_with_no_boost() -> torch.Tensor:
    """Create a 58D state where normal actions are safe but boost is unavailable."""
    state = torch.zeros(58, dtype=torch.float32)
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    state[danger_start:danger_end] = 0.0
    state[StateIndices.BOOST_AVAILABLE] = 0.0
    return state


def _state_with_boost_available() -> torch.Tensor:
    """Create a 58D state whose approximate features allow every action."""
    state = _state_with_no_boost()
    state[StateIndices.BOOST_AVAILABLE] = 1.0
    return state


def _trapped_state() -> torch.Tensor:
    """Create a 58D state where all normal directions are immediate collisions."""
    state = _state_with_no_boost()
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    state[danger_start:danger_end] = 1.0
    return state


# ============================================================================
# Epsilon Calculation
# ============================================================================


class TestComputeActorEpsilon:
    """Tests for the Ape-X actor epsilon formula."""

    def test_single_actor_returns_base(self):
        eps = compute_actor_epsilon(0, num_actors=1, base_epsilon=0.4)
        assert eps == pytest.approx(0.4)

    def test_first_actor_has_highest_epsilon(self):
        eps_0 = compute_actor_epsilon(0, num_actors=4, base_epsilon=0.4)
        eps_1 = compute_actor_epsilon(1, num_actors=4, base_epsilon=0.4)
        eps_3 = compute_actor_epsilon(3, num_actors=4, base_epsilon=0.4)
        assert eps_0 > eps_1 > eps_3

    def test_last_actor_has_lowest_epsilon(self):
        eps = compute_actor_epsilon(3, num_actors=4, base_epsilon=0.4, alpha=7.0)
        # epsilon^(1+7) = 0.4^8 ≈ 0.000655
        assert eps == pytest.approx(0.4**8, rel=1e-5)

    def test_first_actor_exponent_is_one(self):
        eps = compute_actor_epsilon(0, num_actors=4, base_epsilon=0.4, alpha=7.0)
        # exponent = 1 + 0/(4-1)*7 = 1.0
        assert eps == pytest.approx(0.4, rel=1e-5)

    def test_middle_actor(self):
        eps = compute_actor_epsilon(1, num_actors=3, base_epsilon=0.4, alpha=7.0)
        # exponent = 1 + 1/(3-1)*7 = 1+3.5 = 4.5
        assert eps == pytest.approx(0.4**4.5, rel=1e-5)

    def test_all_epsilons_positive(self):
        for i in range(8):
            eps = compute_actor_epsilon(i, num_actors=8)
            assert eps > 0

    def test_all_epsilons_at_most_base(self):
        for i in range(8):
            eps = compute_actor_epsilon(i, num_actors=8, base_epsilon=0.4)
            assert eps <= 0.4 + 1e-10


# ============================================================================
# Actor Construction
# ============================================================================


class TestApexActorConstruction:
    """Tests for ApexActor initialization."""

    def _make_mock_network(self):
        net = MagicMock()
        net.state_dict.return_value = {}
        return net

    def _make_actor(self, actor_id=0, num_actors=4, **kwargs):
        net = self._make_mock_network()
        client = MagicMock(spec=ActorBufferClient)
        weight_q = MagicMock(spec=mp.Queue)
        stats_q = queue.Queue()
        stop = SimpleNamespace(is_set=MagicMock(return_value=False))
        return ApexActor(
            actor_id=actor_id,
            num_actors=num_actors,
            shared_network=net,
            buffer_client=client,
            weight_queue=weight_q,
            stats_queue=stats_q,
            stop_event=stop,
            **kwargs,
        )

    def test_actor_stores_buffer_client(self):
        actor = self._make_actor()
        assert hasattr(actor, "buffer_client")
        assert isinstance(actor.buffer_client, MagicMock)

    def test_actor_computes_epsilon(self):
        actor = self._make_actor(actor_id=0, num_actors=4)
        assert actor.epsilon == pytest.approx(0.4)

    def test_actor_stores_hyperparams(self):
        actor = self._make_actor(gamma=0.95, n_step=5, alpha=0.7)
        assert actor.gamma == 0.95
        assert actor.n_step == 5
        assert actor.alpha == 0.7

    def test_actor_default_danger_exploration_is_disabled(self):
        actor = self._make_actor()

        assert actor.danger_exploration_rate == pytest.approx(0.0)

    def test_actor_defaults_to_configured_env_snake_count(self):
        actor = self._make_actor()

        assert actor.env_num_snakes == GameConfig.APEX_ACTOR_ENV_NUM_SNAKES

    def test_actor_defaults_to_configured_terminal_rich_environment_shape(self):
        actor = self._make_actor()

        assert actor.env_board_scale == pytest.approx(GameConfig.APEX_ACTOR_BOARD_SCALE)
        assert actor.env_food_multiplier == pytest.approx(GameConfig.APEX_ACTOR_FOOD_MULTIPLIER)

    def test_actor_accepts_single_snake_env_override(self):
        actor = self._make_actor(env_num_snakes=1)

        assert actor.env_num_snakes == 1

    def test_actor_rejects_invalid_exploration_rates(self):
        with pytest.raises(ValueError, match="danger_exploration_rate"):
            self._make_actor(danger_exploration_rate=1.1)
        with pytest.raises(ValueError, match="boost_exploration_rate"):
            self._make_actor(boost_exploration_rate=-0.1)

    def test_actor_rejects_invalid_environment_shape(self):
        with pytest.raises(ValueError, match="env_num_snakes"):
            self._make_actor(env_num_snakes=0)
        with pytest.raises(ValueError, match="env_board_scale"):
            self._make_actor(env_board_scale=0.0)
        with pytest.raises(ValueError, match="env_food_multiplier"):
            self._make_actor(env_food_multiplier=float("nan"))

    def test_actor_disables_local_snake_policy_training(self):
        actor = self._make_actor(actor_id=0, num_actors=4)
        memory = MagicMock()
        policy = MagicMock()
        policy.training = True
        policy.memory = memory
        snake = MagicMock()
        snake.actor_epsilon = None
        snake.current_epsilon = 0.0
        snake.policy = policy
        actor.env = MagicMock(snakes=[snake])

        actor._override_snake_epsilon()

        assert snake.actor_epsilon == pytest.approx(actor.epsilon)
        assert snake.current_epsilon == pytest.approx(actor.epsilon)
        assert policy.training is False
        memory.clear.assert_called_once()

    def test_actor_configures_snake_exploration_rates(self):
        actor = self._make_actor(
            actor_id=0,
            num_actors=4,
            boost_exploration_rate=0.33,
            danger_exploration_rate=0.07,
        )
        memory = MagicMock()
        policy = MagicMock()
        policy.training = True
        policy.memory = memory
        snake = SimpleNamespace(
            actor_epsilon=None,
            current_epsilon=0.0,
            boost_exploration_rate=0.0,
            danger_exploration_rate=0.0,
            policy=policy,
            last_state=None,
            last_action=None,
        )
        actor.env = SimpleNamespace(snakes=[snake])

        actor._override_snake_epsilon()

        assert snake.boost_exploration_rate == pytest.approx(0.33)
        assert snake.danger_exploration_rate == pytest.approx(0.07)

    def test_actor_configures_all_replay_snakes_once_per_policy(self):
        actor = self._make_actor(actor_id=1, num_actors=4)
        memory = MagicMock()
        shared_policy = SimpleNamespace(training=True, memory=memory)
        snakes = [
            SimpleNamespace(
                id=i,
                actor_epsilon=None,
                current_epsilon=0.0,
                policy=shared_policy,
                last_state=None,
                last_action=None,
            )
            for i in range(3)
        ]
        actor.env = SimpleNamespace(snakes=snakes)

        actor._override_snake_epsilon()

        for snake in snakes:
            assert snake.actor_epsilon == pytest.approx(actor.epsilon)
            assert snake.current_epsilon == pytest.approx(actor.epsilon)
        assert shared_policy.training is False
        memory.clear.assert_called_once()


# ============================================================================
# Experience Sending
# ============================================================================


class TestSendExperienceBatch:
    """Tests for _send_experience_batch using buffer client."""

    def _make_actor_with_mock_client(self):
        net = MagicMock()
        net.state_dict.return_value = {}
        client = MagicMock(spec=ActorBufferClient)
        weight_q = MagicMock(spec=mp.Queue)
        stats_q = queue.Queue()
        stop = SimpleNamespace(is_set=MagicMock(return_value=False))
        actor = ApexActor(
            actor_id=0,
            num_actors=1,
            shared_network=net,
            buffer_client=client,
            weight_queue=weight_q,
            stats_queue=stats_q,
            stop_event=stop,
        )
        return actor, client

    def test_send_empty_batch_does_nothing(self):
        actor, client = self._make_actor_with_mock_client()
        actor._send_experience_batch([])
        client.add_batch.assert_not_called()

    def test_send_batch_calls_client(self):
        actor, client = self._make_actor_with_mock_client()
        experiences = [
            Experience(
                state=np.zeros(58, dtype=np.float32),
                action=1,
                reward=0.5,
                next_state=np.zeros(58, dtype=np.float32),
                done=False,
                bootstrap_steps=3,
                td_error=0.1,
            )
        ]
        actor._send_experience_batch(experiences)
        client.add_batch.assert_called_once()
        assert client.add_batch.call_args[1]["bootstrap_steps"] == [3]

    def test_send_batch_passes_numpy_states(self):
        actor, client = self._make_actor_with_mock_client()
        state = np.random.randn(58).astype(np.float32)
        next_state = np.random.randn(58).astype(np.float32)
        experiences = [
            Experience(
                state=state,
                action=2,
                reward=1.0,
                next_state=next_state,
                done=True,
                bootstrap_steps=1,
                td_error=0.5,
            )
        ]
        actor._send_experience_batch(experiences)
        call_kwargs = client.add_batch.call_args[1]
        # States should be numpy arrays
        assert isinstance(call_kwargs["states"][0], np.ndarray)
        assert isinstance(call_kwargs["next_states"][0], np.ndarray)

    def test_send_batch_priorities_from_td_errors(self):
        actor, client = self._make_actor_with_mock_client()
        experiences = [
            Experience(
                state=np.zeros(58, dtype=np.float32),
                action=0,
                reward=0.0,
                next_state=np.zeros(58, dtype=np.float32),
                done=False,
                bootstrap_steps=2,
                td_error=2.0,
            ),
            Experience(
                state=np.zeros(58, dtype=np.float32),
                action=1,
                reward=1.0,
                next_state=np.zeros(58, dtype=np.float32),
                done=True,
                bootstrap_steps=1,
                td_error=0.5,
            ),
        ]
        actor._send_experience_batch(experiences)
        call_kwargs = client.add_batch.call_args[1]
        priorities = call_kwargs["priorities"]
        assert len(priorities) == 2
        # Higher TD error → higher priority
        assert priorities[0] > priorities[1]

    def test_send_batch_passes_next_action_masks(self):
        actor, client = self._make_actor_with_mock_client()
        mask = np.array([False, True, False, False, False, False], dtype=np.bool_)
        experiences = [
            Experience(
                state=np.zeros(58, dtype=np.float32),
                action=0,
                reward=0.0,
                next_state=np.zeros(58, dtype=np.float32),
                done=False,
                bootstrap_steps=1,
                td_error=1.0,
                next_action_mask=mask,
            )
        ]

        actor._send_experience_batch(experiences)

        call_kwargs = client.add_batch.call_args[1]
        assert call_kwargs["next_action_masks"][0].tolist() == mask.tolist()

    def test_send_batch_updates_replay_coverage_counters(self):
        actor, _client = self._make_actor_with_mock_client()
        mask = np.array([False, True, False, False, False, False], dtype=np.bool_)
        experiences = [
            Experience(
                state=np.zeros(58, dtype=np.float32),
                action=4,
                reward=0.5,
                next_state=np.zeros(58, dtype=np.float32),
                done=False,
                bootstrap_steps=3,
                td_error=1.0,
                next_action_mask=mask,
            ),
            Experience(
                state=np.zeros(58, dtype=np.float32),
                action=1,
                reward=-1.0,
                next_state=np.zeros(58, dtype=np.float32),
                done=True,
                bootstrap_steps=1,
                td_error=1.0,
            ),
        ]

        actor._send_experience_batch(experiences)

        assert actor.sent_experience_count == 2
        assert actor.sent_action_counts == [0, 1, 0, 0, 1, 0]
        assert actor.sent_boost_action_count == 1
        assert actor.sent_exact_mask_count == 1
        assert actor.sent_terminal_count == 1
        assert actor.sent_nonterminal_count == 1
        assert actor.sent_nonterminal_exact_mask_count == 1
        assert actor.sent_nonterminal_trapped_next_count == 0
        assert actor.sent_positive_reward_count == 1
        assert actor.sent_zero_reward_count == 0
        assert actor.sent_negative_reward_count == 1
        assert actor.sent_multistep_count == 1
        assert actor.sent_invalid_current_action_count == 1
        assert actor.sent_invalid_current_normal_action_count == 0
        assert actor.sent_invalid_current_boost_action_count == 1

    def test_send_stats_includes_replay_coverage(self):
        actor, _client = self._make_actor_with_mock_client()
        actor.buffer_client.get_stats.return_value = {
            "queued_message_count": 3,
            "dropped_message_count": 2,
            "dropped_experience_count": 1,
            "last_drop_error": "queue full",
        }
        actor.sent_experience_count = 4
        actor.sent_action_counts = [1, 0, 0, 2, 1, 0]
        actor.sent_boost_action_count = 3
        actor.sent_exact_mask_count = 2
        actor.sent_terminal_count = 1
        actor.sent_nonterminal_count = 3
        actor.sent_nonterminal_exact_mask_count = 2
        actor.sent_nonterminal_trapped_next_count = 1
        actor.sent_positive_reward_count = 2
        actor.sent_zero_reward_count = 1
        actor.sent_negative_reward_count = 1
        actor.sent_multistep_count = 3
        actor.sent_invalid_current_action_count = 2
        actor.sent_invalid_current_normal_action_count = 1
        actor.sent_invalid_current_boost_action_count = 1
        actor.dropped_missing_next_state_count = 1

        actor._send_stats(episode=3, avg_reward=1.25, total_steps=40)

        stats = actor.stats_queue.get_nowait()
        assert stats["sent_experience_count"] == 4
        assert stats["sent_action_counts"] == [1, 0, 0, 2, 1, 0]
        assert stats["sent_active_action_count"] == 3
        assert stats["sent_boost_action_fraction"] == pytest.approx(0.75)
        assert stats["sent_exact_mask_fraction"] == pytest.approx(0.5)
        assert stats["sent_terminal_fraction"] == pytest.approx(0.25)
        assert stats["sent_nonterminal_count"] == 3
        assert stats["sent_nonterminal_exact_mask_fraction"] == pytest.approx(2 / 3)
        assert stats["sent_nonterminal_trapped_next_fraction"] == pytest.approx(1 / 3)
        assert stats["dropped_missing_next_state_count"] == 1
        assert stats["dropped_missing_next_state_fraction"] == pytest.approx(1 / 5)
        assert stats["sent_positive_reward_fraction"] == pytest.approx(0.5)
        assert stats["sent_zero_reward_fraction"] == pytest.approx(0.25)
        assert stats["sent_negative_reward_fraction"] == pytest.approx(0.25)
        assert stats["sent_multistep_fraction"] == pytest.approx(0.75)
        assert stats["sent_invalid_current_action_count"] == 2
        assert stats["sent_invalid_current_action_fraction"] == pytest.approx(0.5)
        assert stats["sent_invalid_current_normal_action_count"] == 1
        assert stats["sent_invalid_current_boost_action_count"] == 1
        assert stats["buffer_queued_message_count"] == 3
        assert stats["buffer_dropped_message_count"] == 2
        assert stats["buffer_dropped_experience_count"] == 1
        assert stats["buffer_dropped_experience_fraction"] == pytest.approx(0.25)
        assert stats["buffer_last_drop_error"] == "queue full"

    def test_shutdown_stats_reports_final_reward_average(self):
        actor, _client = self._make_actor_with_mock_client()
        actor._send_stats = MagicMock()

        actor._send_shutdown_stats(
            episode=5,
            rewards_history=deque([1.0, 3.0, -1.0], maxlen=100),
            total_steps=42,
        )

        actor._send_stats.assert_called_once_with(
            5,
            pytest.approx(1.0),
            42,
        )


# ============================================================================
# Weight Sync
# ============================================================================


class TestWeightSync:
    """Tests for weight synchronization."""

    def test_sync_from_shared(self):
        net = MagicMock()
        net.state_dict.return_value = {"layer.weight": torch.zeros(10)}
        client = MagicMock(spec=ActorBufferClient)
        weight_q = MagicMock(spec=mp.Queue)
        stats_q = MagicMock(spec=mp.Queue)
        stop = SimpleNamespace(is_set=MagicMock(return_value=False))

        actor = ApexActor(
            actor_id=0,
            num_actors=1,
            shared_network=net,
            buffer_client=client,
            weight_queue=weight_q,
            stats_queue=stats_q,
            stop_event=stop,
        )
        # Simulate initialized local/target networks
        actor.local_network = MagicMock()
        actor.target_network = MagicMock()

        actor._sync_weights_from_shared()
        actor.local_network.load_state_dict.assert_called_once()
        actor.target_network.load_state_dict.assert_called_once()

    def test_check_weight_queue_does_not_trust_empty(self):
        """Actors should apply queued weights even when Queue.empty() is unreliable."""

        class UnreliableEmptyQueue:
            def __init__(self, items):
                self.items = list(items)

            def empty(self):
                return True

            def get_nowait(self):
                if not self.items:
                    raise queue.Empty
                return self.items.pop(0)

        weights = {"layer.weight": torch.ones(3)}
        actor = ApexActor(
            actor_id=0,
            num_actors=1,
            shared_network=MagicMock(),
            buffer_client=MagicMock(spec=ActorBufferClient),
            weight_queue=UnreliableEmptyQueue([weights]),
            stats_queue=MagicMock(spec=mp.Queue),
            stop_event=MagicMock(spec=mp.Event),
        )
        actor.local_network = MagicMock()
        actor.target_network = MagicMock()
        actor._sync_snake_policy_weights = MagicMock()

        applied = actor._check_weight_queue()

        assert applied is True
        actor.local_network.load_state_dict.assert_called_once_with(weights)
        actor.target_network.load_state_dict.assert_called_once_with(weights)
        actor._sync_snake_policy_weights.assert_called_once()

    def test_check_weight_queue_returns_false_without_updates(self):
        """Empty weight queues should not trigger snake policy syncs."""

        class EmptyQueue:
            def get_nowait(self):
                raise queue.Empty

        actor = ApexActor(
            actor_id=0,
            num_actors=1,
            shared_network=MagicMock(),
            buffer_client=MagicMock(spec=ActorBufferClient),
            weight_queue=EmptyQueue(),
            stats_queue=MagicMock(spec=mp.Queue),
            stop_event=MagicMock(spec=mp.Event),
        )
        actor.local_network = MagicMock()
        actor.target_network = MagicMock()
        actor._sync_snake_policy_weights = MagicMock()

        applied = actor._check_weight_queue()

        assert applied is False
        actor.local_network.load_state_dict.assert_not_called()
        actor.target_network.load_state_dict.assert_not_called()
        actor._sync_snake_policy_weights.assert_not_called()


# ============================================================================
# N-step Return
# ============================================================================


class TestNStepReturn:
    """Tests for n-step return computation."""

    def _make_actor(self, gamma=0.99, n_step=3):
        from src.core.game_config import GameConfig

        net = MagicMock()
        net.state_dict.return_value = {}
        client = MagicMock(spec=ActorBufferClient)
        weight_q = MagicMock(spec=mp.Queue)
        stats_q = MagicMock(spec=mp.Queue)
        stop = SimpleNamespace(is_set=MagicMock(return_value=False))
        actor = ApexActor(
            actor_id=0,
            num_actors=1,
            shared_network=net,
            buffer_client=client,
            weight_queue=weight_q,
            stats_queue=stats_q,
            stop_event=stop,
            gamma=gamma,
            n_step=n_step,
        )
        actor.device = torch.device("cpu")
        # Create simple mock networks that return fixed Q-values
        mock_net = MagicMock()
        mock_net.return_value = torch.tensor([[1.0] * GameConfig.OUTPUT_SIZE])
        actor.local_network = mock_net
        actor.target_network = mock_net
        return actor

    def test_single_step_terminal(self):
        from collections import deque

        actor = self._make_actor(gamma=0.99)
        buf = deque(
            [
                {
                    "state": torch.zeros(58),
                    "action": 0,
                    "reward": 1.0,
                    "next_state": None,
                    "done": True,
                }
            ]
        )
        exp = actor._compute_n_step_experience(buf)
        assert exp is not None
        assert exp.reward == pytest.approx(1.0)
        assert exp.done is True
        assert exp.bootstrap_steps == 1

    def test_multi_step_discounted_return(self):
        from collections import deque

        gamma = 0.99
        actor = self._make_actor(gamma=gamma, n_step=3)
        buf = deque(
            [
                {
                    "state": torch.zeros(58),
                    "action": 0,
                    "reward": 1.0,
                    "next_state": torch.ones(58),
                    "done": False,
                },
                {
                    "state": torch.ones(58),
                    "action": 1,
                    "reward": 2.0,
                    "next_state": torch.ones(58),
                    "done": False,
                },
                {
                    "state": torch.ones(58),
                    "action": 2,
                    "reward": 3.0,
                    "next_state": None,
                    "done": True,
                },
            ]
        )
        exp = actor._compute_n_step_experience(buf)
        expected = 1.0 + gamma * 2.0 + gamma**2 * 3.0
        assert exp.reward == pytest.approx(expected, rel=1e-5)
        assert exp.bootstrap_steps == 3

    def test_non_terminal_n_step_reward_excludes_bootstrap_q(self):
        from collections import deque

        gamma = 0.9
        actor = self._make_actor(gamma=gamma, n_step=2)
        buf = deque(
            [
                {
                    "state": torch.zeros(58),
                    "action": 0,
                    "reward": 1.0,
                    "next_state": _state_with_no_boost(),
                    "done": False,
                },
                {
                    "state": _state_with_no_boost(),
                    "action": 1,
                    "reward": 2.0,
                    "next_state": _state_with_no_boost(),
                    "done": False,
                },
            ]
        )

        exp = actor._compute_n_step_experience(buf)

        assert exp is not None
        assert exp.reward == pytest.approx(1.0 + gamma * 2.0)
        assert exp.bootstrap_steps == 2
        assert exp.done is False
        # Mock networks return Q=1, so priority target includes gamma^2 * 1
        # while stored reward remains reward-only for learner bootstrapping.
        assert exp.td_error == pytest.approx(abs((1.0 + gamma * 2.0 + gamma**2) - 1.0))

    def test_priority_bootstrap_masks_invalid_next_actions(self):
        """Actor-side priorities should not bootstrap from invalid high-Q actions."""
        from collections import deque

        gamma = 0.5
        actor = self._make_actor(gamma=gamma, n_step=1)
        actor.local_network = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        actor.target_network = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])
        buf = deque(
            [
                {
                    "state": torch.zeros(58),
                    "action": 0,
                    "reward": 0.0,
                    "next_state": _state_with_no_boost(),
                    "done": False,
                }
            ]
        )

        exp = actor._compute_n_step_experience(buf)

        assert exp is not None
        assert exp.reward == pytest.approx(0.0)
        assert exp.td_error == pytest.approx(2.5)

    def test_priority_bootstrap_prefers_exact_next_action_mask(self):
        """Actor priorities should use exact simulator masks when replay has them."""
        from collections import deque

        gamma = 0.5
        actor = self._make_actor(gamma=gamma, n_step=1)
        actor.local_network = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        actor.target_network = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])
        exact_mask = torch.tensor([False, True, False, False, False, False])
        buf = deque(
            [
                {
                    "state": torch.zeros(58),
                    "action": 0,
                    "reward": 0.0,
                    "next_state": _state_with_boost_available(),
                    "next_action_mask": exact_mask,
                    "done": False,
                }
            ]
        )

        exp = actor._compute_n_step_experience(buf)

        assert exp is not None
        assert exp.td_error == pytest.approx(2.5)
        assert exp.next_action_mask is not None
        assert exp.next_action_mask.tolist() == exact_mask.numpy().tolist()

    def test_priority_bootstrap_skips_trapped_next_state(self):
        """Actor-side priorities should be reward-only when no next action is valid."""
        from collections import deque

        gamma = 0.5
        actor = self._make_actor(gamma=gamma, n_step=1)
        actor.local_network = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        actor.target_network = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])
        buf = deque(
            [
                {
                    "state": torch.zeros(58),
                    "action": 0,
                    "reward": 1.25,
                    "next_state": _trapped_state(),
                    "done": False,
                }
            ]
        )

        exp = actor._compute_n_step_experience(buf)

        assert exp is not None
        assert exp.reward == pytest.approx(1.25)
        assert exp.td_error == pytest.approx(1.25)

    def test_empty_buffer_returns_none(self):
        from collections import deque

        actor = self._make_actor()
        exp = actor._compute_n_step_experience(deque())
        assert exp is None

    def test_truncated_tail_stays_nonterminal(self):
        from collections import deque

        actor = self._make_actor(gamma=0.99, n_step=3)
        buf = deque(
            [
                {
                    "state": torch.zeros(58),
                    "action": 0,
                    "reward": 1.0,
                    "next_state": torch.ones(58),
                    "done": False,
                }
            ]
        )

        exp = actor._compute_n_step_experience(buf)

        assert exp is not None
        assert exp.done is False
        assert exp.bootstrap_steps == 1
        assert exp.reward == pytest.approx(1.0)
        assert exp.next_state.sum() == pytest.approx(58.0)

    def test_nonterminal_missing_next_state_is_dropped(self):
        from collections import deque

        actor = self._make_actor(gamma=0.99, n_step=1)
        buf = deque(
            [
                {
                    "state": torch.arange(58, dtype=torch.float32),
                    "action": 0,
                    "reward": 0.5,
                    "next_state": None,
                    "done": False,
                }
            ]
        )

        exp = actor._compute_n_step_experience(buf)

        assert exp is None
        assert actor.dropped_missing_next_state_count == 1


# ============================================================================
# Episode Collection
# ============================================================================


class TestRunEpisodeCollection:
    """Tests for actor-side replay collection from multi-snake environments."""

    def _make_actor(self, n_step=1):
        net = MagicMock()
        net.state_dict.return_value = {}
        client = MagicMock(spec=ActorBufferClient)
        weight_q = MagicMock(spec=mp.Queue)
        stats_q = MagicMock(spec=mp.Queue)
        stop = SimpleNamespace(is_set=MagicMock(return_value=False))
        actor = ApexActor(
            actor_id=0,
            num_actors=1,
            shared_network=net,
            buffer_client=client,
            weight_queue=weight_q,
            stats_queue=stats_q,
            stop_event=stop,
            n_step=n_step,
        )
        actor.device = torch.device("cpu")
        return actor

    def _make_snake(self, snake_id, reward):
        return SimpleNamespace(
            id=snake_id,
            is_alive=True,
            actor_epsilon=None,
            current_epsilon=0.0,
            policy=SimpleNamespace(training=True, memory=None),
            last_state=torch.full((58,), float(snake_id)),
            last_action=snake_id % 6,
            last_reward=reward,
            last_next_state=torch.full((58,), float(snake_id + 1)),
            last_done=False,
            last_transition_frame=None,
        )

    def _make_experience(self, action=0, reward=0.0, done=False, bootstrap_steps=1):
        return Experience(
            state=np.zeros(58, dtype=np.float32),
            action=action,
            reward=reward,
            next_state=np.zeros(58, dtype=np.float32),
            done=done,
            bootstrap_steps=bootstrap_steps,
            td_error=abs(reward),
        )

    def test_run_episode_collects_fresh_transition_from_each_snake(self):
        actor = self._make_actor(n_step=1)
        snakes = [self._make_snake(0, 1.0), self._make_snake(1, 2.0)]
        env = SimpleNamespace(frame=0, snakes=snakes, reset=MagicMock())

        def update(train_mode=True, learn=False):
            env.frame += 1
            for snake in snakes:
                snake.last_transition_frame = env.frame
                snake.is_alive = False

        env.update = MagicMock(side_effect=update)
        actor.env = env
        experiences = [
            self._make_experience(action=0, reward=1.0),
            self._make_experience(action=1, reward=2.0),
        ]
        actor._compute_n_step_experience = MagicMock(side_effect=experiences)

        episode_reward, episode_steps, collected = actor._run_episode()

        assert episode_steps == 1
        assert episode_reward == pytest.approx(3.0)
        assert collected == experiences
        assert actor._compute_n_step_experience.call_count == 2

    def test_run_episode_ignores_stale_snake_transition(self):
        actor = self._make_actor(n_step=1)
        fresh_snake = self._make_snake(0, 1.0)
        stale_snake = self._make_snake(1, 2.0)
        env = SimpleNamespace(frame=0, snakes=[fresh_snake, stale_snake], reset=MagicMock())

        def update(train_mode=True, learn=False):
            env.frame += 1
            fresh_snake.last_transition_frame = env.frame
            stale_snake.last_transition_frame = env.frame - 1
            fresh_snake.is_alive = False
            stale_snake.is_alive = False

        env.update = MagicMock(side_effect=update)
        actor.env = env
        experience = self._make_experience(action=0, reward=1.0)
        actor._compute_n_step_experience = MagicMock(return_value=experience)

        episode_reward, episode_steps, collected = actor._run_episode()

        assert episode_steps == 1
        assert episode_reward == pytest.approx(1.0)
        assert collected == [experience]
        actor._compute_n_step_experience.assert_called_once()

    def test_run_episode_polls_weights_during_long_episode(self):
        actor = self._make_actor(n_step=1)
        actor.weight_sync_interval = 2
        snake = self._make_snake(0, 1.0)
        env = SimpleNamespace(frame=0, snakes=[snake], reset=MagicMock())

        def update(train_mode=True, learn=False):
            env.frame += 1
            snake.last_transition_frame = env.frame
            snake.last_reward = 1.0
            snake.last_done = env.frame == 5
            snake.is_alive = env.frame < 5

        env.update = MagicMock(side_effect=update)
        actor.env = env
        actor._check_weight_queue = MagicMock(return_value=False)
        actor._compute_n_step_experience = MagicMock(
            side_effect=lambda buffer: self._make_experience(
                action=buffer[0]["action"],
                reward=sum(float(transition["reward"]) for transition in buffer),
                done=bool(buffer[-1]["done"]),
                bootstrap_steps=len(buffer),
            )
        )

        _, episode_steps, collected = actor._run_episode()

        assert episode_steps == 5
        assert len(collected) == 5
        assert actor._check_weight_queue.call_count == 2

    def test_run_episode_streams_full_batches_before_episode_end(self):
        actor = self._make_actor(n_step=1)
        actor.batch_send_size = 2
        snake = self._make_snake(0, 1.0)
        env = SimpleNamespace(frame=0, snakes=[snake], reset=MagicMock())
        sent_batches_before_third_update = []

        def update(train_mode=True, learn=False):
            env.frame += 1
            if env.frame == 3:
                sent_batches_before_third_update.append(actor.buffer_client.add_batch.call_count)
            snake.last_transition_frame = env.frame
            snake.last_reward = float(env.frame)
            snake.last_done = env.frame == 5
            snake.is_alive = env.frame < 5

        env.update = MagicMock(side_effect=update)
        actor.env = env
        actor._compute_n_step_experience = MagicMock(
            side_effect=lambda buffer: self._make_experience(
                action=buffer[0]["action"],
                reward=sum(float(transition["reward"]) for transition in buffer),
                done=bool(buffer[-1]["done"]),
                bootstrap_steps=len(buffer),
            )
        )

        _, episode_steps, collected = actor._run_episode(stream_to_buffer=True)

        assert episode_steps == 5
        assert sent_batches_before_third_update == [1]
        assert actor.buffer_client.add_batch.call_count == 2
        assert [
            len(call.kwargs["states"]) for call in actor.buffer_client.add_batch.call_args_list
        ] == [
            2,
            2,
        ]
        assert len(collected) == 1
        assert collected[0].reward == pytest.approx(5.0)

    def test_run_episode_keeps_n_step_buffers_separate_by_snake(self):
        actor = self._make_actor(n_step=2)
        snake_a = self._make_snake(0, 0.0)
        snake_b = self._make_snake(1, 0.0)
        env = SimpleNamespace(frame=0, snakes=[snake_a, snake_b], reset=MagicMock())

        frame_rewards = {
            1: {0: 1.0, 1: 10.0},
            2: {0: 2.0, 1: 20.0},
        }

        def update(train_mode=True, learn=False):
            env.frame += 1
            for snake in env.snakes:
                snake.last_reward = frame_rewards[env.frame][snake.id]
                snake.last_transition_frame = env.frame
                snake.last_done = env.frame == 2
                snake.is_alive = env.frame < 2

        def compute_experience(buffer):
            return self._make_experience(
                action=buffer[0]["action"],
                reward=sum(float(transition["reward"]) for transition in buffer),
                done=bool(buffer[-1]["done"]),
                bootstrap_steps=len(buffer),
            )

        env.update = MagicMock(side_effect=update)
        actor.env = env
        actor._compute_n_step_experience = MagicMock(side_effect=compute_experience)

        _, _, collected = actor._run_episode()

        assert [exp.reward for exp in collected] == pytest.approx([3.0, 2.0, 30.0, 20.0])
        assert [exp.bootstrap_steps for exp in collected] == [2, 1, 2, 1]


# ============================================================================
# spawn_actors
# ============================================================================


class TestSpawnActors:
    """Tests for the spawn_actors factory function."""

    def test_spawn_creates_correct_count(self):
        net = MagicMock()
        net.state_dict.return_value = {}
        buf_proc = MagicMock(spec=BufferProcess)
        buf_proc.get_actor_client.return_value = MagicMock(spec=ActorBufferClient)
        weight_qs = [MagicMock(spec=mp.Queue) for _ in range(4)]
        stats_q = MagicMock(spec=mp.Queue)
        stop = SimpleNamespace(is_set=MagicMock(return_value=False))

        actors = spawn_actors(
            num_actors=4,
            shared_network=net,
            buffer_process=buf_proc,
            weight_queues=weight_qs,
            stats_queue=stats_q,
            stop_event=stop,
        )
        assert len(actors) == 4
        assert buf_proc.get_actor_client.call_count == 4

    def test_spawn_assigns_unique_ids(self):
        net = MagicMock()
        net.state_dict.return_value = {}
        buf_proc = MagicMock(spec=BufferProcess)
        buf_proc.get_actor_client.return_value = MagicMock(spec=ActorBufferClient)
        weight_qs = [MagicMock(spec=mp.Queue) for _ in range(3)]
        stats_q = MagicMock(spec=mp.Queue)
        stop = SimpleNamespace(is_set=MagicMock(return_value=False))

        actors = spawn_actors(
            num_actors=3,
            shared_network=net,
            buffer_process=buf_proc,
            weight_queues=weight_qs,
            stats_queue=stats_q,
            stop_event=stop,
        )
        ids = [a.actor_id for a in actors]
        assert ids == [0, 1, 2]

    def test_spawn_each_actor_gets_unique_client(self):
        net = MagicMock()
        net.state_dict.return_value = {}
        buf_proc = MagicMock(spec=BufferProcess)
        # Return unique mock for each call
        clients = [MagicMock(spec=ActorBufferClient) for _ in range(3)]
        buf_proc.get_actor_client.side_effect = clients
        weight_qs = [MagicMock(spec=mp.Queue) for _ in range(3)]
        stats_q = MagicMock(spec=mp.Queue)
        stop = SimpleNamespace(is_set=MagicMock(return_value=False))

        actors = spawn_actors(
            num_actors=3,
            shared_network=net,
            buffer_process=buf_proc,
            weight_queues=weight_qs,
            stats_queue=stats_q,
            stop_event=stop,
        )
        for i, actor in enumerate(actors):
            assert actor.buffer_client is clients[i]

    def test_spawn_passes_actor_environment_and_exploration_rates(self):
        net = MagicMock()
        net.state_dict.return_value = {}
        buf_proc = MagicMock(spec=BufferProcess)
        buf_proc.get_actor_client.return_value = MagicMock(spec=ActorBufferClient)
        weight_qs = [MagicMock(spec=mp.Queue)]
        stats_q = MagicMock(spec=mp.Queue)
        stop = SimpleNamespace(is_set=MagicMock(return_value=False))

        actors = spawn_actors(
            num_actors=1,
            shared_network=net,
            buffer_process=buf_proc,
            weight_queues=weight_qs,
            stats_queue=stats_q,
            stop_event=stop,
            env_num_snakes=5,
            env_board_scale=0.3,
            env_food_multiplier=0.7,
            boost_exploration_rate=0.31,
            danger_exploration_rate=0.09,
        )

        assert actors[0].env_num_snakes == 5
        assert actors[0].env_board_scale == pytest.approx(0.3)
        assert actors[0].env_food_multiplier == pytest.approx(0.7)
        assert actors[0].boost_exploration_rate == pytest.approx(0.31)
        assert actors[0].danger_exploration_rate == pytest.approx(0.09)


class TestStopActors:
    """Tests for graceful actor shutdown."""

    class FakeActor:
        def __init__(self, exits_on_join=True):
            self.exits_on_join = exits_on_join
            self.alive = True
            self.calls = []

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            self.calls.append(("join", timeout))
            if self.exits_on_join:
                self.alive = False

        def terminate(self):
            self.calls.append(("terminate", None))
            self.alive = False

    def test_stop_actors_joins_graceful_actor_without_terminating(self):
        actor = self.FakeActor(exits_on_join=True)

        stop_actors([actor], timeout=0.25)

        assert actor.calls == [("join", 0.25)]

    def test_stop_actors_terminates_actor_that_ignores_join_timeout(self):
        actor = self.FakeActor(exits_on_join=False)

        stop_actors([actor], timeout=0.25)

        assert actor.calls == [
            ("join", 0.25),
            ("terminate", None),
            ("join", 0.25),
        ]
