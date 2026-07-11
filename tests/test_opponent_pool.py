"""Tests for opponent-pool self-play in the Apex actor path (blueprint §3.3).

Covers: per-episode latest/frozen slot sampling (hero always latest), frozen
slots being excluded from replay collection and weight syncs, the realized
opponent-exposure telemetry, the mechanics-v2 population-floor episode break,
and the YAML + CLI round-trip of the two pool knobs.
"""

import queue
import sys
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import yaml

from src.core.config_loader import load_config
from src.core.game_config import GameConfig
from src.model.apex_network import ApexNetwork
from src.training.apex_actor import (
    ApexActor,
    FrozenOpponentPolicy,
    _resolve_opponent_pool_dir,
    _resolve_pool_latest_fraction,
)
from src.training.apex_buffer import ActorBufferClient


def _make_actor(**kwargs) -> ApexActor:
    """Create an ApexActor with mocked IPC plumbing (never started)."""
    net = MagicMock()
    net.state_dict.return_value = {}
    return ApexActor(
        actor_id=0,
        num_actors=1,
        shared_network=net,
        buffer_client=MagicMock(spec=ActorBufferClient),
        weight_queue=MagicMock(spec=mp.Queue),
        stats_queue=queue.Queue(),
        stop_event=SimpleNamespace(is_set=MagicMock(return_value=False)),
        **kwargs,
    )


def _make_snake(snake_id: int, policy) -> SimpleNamespace:
    """Create a minimal replay-snake stand-in for actor episode tests."""
    return SimpleNamespace(
        id=snake_id,
        is_alive=True,
        actor_epsilon=None,
        current_epsilon=0.0,
        policy=policy,
        last_state=torch.zeros(58),
        last_action=snake_id % 6,
        last_reward=0.0,
        last_next_state=torch.zeros(58),
        last_done=False,
        last_transition_frame=None,
    )


def _fake_frozen_policy(checkpoint_path: str) -> SimpleNamespace:
    """Frozen-policy stand-in matching the FrozenOpponentPolicy surface."""
    return SimpleNamespace(
        is_frozen_opponent=True,
        checkpoint_path=checkpoint_path,
        dqn=MagicMock(),
        device=torch.device("cpu"),
        epsilon=0.0,
        training=False,
        memory=None,
    )


# ============================================================================
# Knob resolution
# ============================================================================


class TestPoolKnobResolution:
    """Constructor/config resolution of the two pool knobs."""

    def test_explicit_values_stored(self):
        actor = _make_actor(opponent_pool_dir="saved_snakes/pool", pool_latest_fraction=0.5)
        assert actor.opponent_pool_dir == "saved_snakes/pool"
        assert actor.pool_latest_fraction == pytest.approx(0.5)

    def test_blank_pool_dir_resolves_to_none(self):
        actor = _make_actor(opponent_pool_dir="   ")
        assert actor.opponent_pool_dir is None

    @pytest.mark.parametrize("fraction", [-0.1, 1.5, float("nan"), float("inf")])
    def test_invalid_fraction_rejected(self, fraction):
        with pytest.raises(ValueError, match="pool_latest_fraction"):
            _make_actor(pool_latest_fraction=fraction)

    def test_none_falls_back_to_config_defaults(self, setup_config):
        assert _resolve_opponent_pool_dir(None) is None
        assert _resolve_pool_latest_fraction(None) == pytest.approx(0.8)


# ============================================================================
# Pool scanning and frozen-policy loading
# ============================================================================


class TestPoolScanAndLoad:
    """Checkpoint discovery and InferenceAgent-backed frozen policies."""

    def test_scan_disabled_pool_returns_empty(self):
        actor = _make_actor()
        assert actor._scan_opponent_pool() == []

    def test_scan_missing_dir_returns_empty(self, tmp_path):
        actor = _make_actor(opponent_pool_dir=str(tmp_path / "missing"))
        assert actor._scan_opponent_pool() == []

    def test_scan_returns_sorted_pth_files_only(self, tmp_path):
        (tmp_path / "b.pth").write_bytes(b"x")
        (tmp_path / "a.pth").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("not a checkpoint")
        actor = _make_actor(opponent_pool_dir=str(tmp_path))
        assert actor._scan_opponent_pool() == [str(tmp_path / "a.pth"), str(tmp_path / "b.pth")]

    def test_get_frozen_policy_loads_once_and_caches(self, tmp_path, setup_config):
        input_size = GameConfig.INPUT_SIZE
        network = ApexNetwork(input_size, 32, 6)
        checkpoint_path = tmp_path / "opponent.pth"
        torch.save(
            {
                "dqn_state_dict": network.state_dict(),
                "input_size": input_size,
                "hidden_size": 32,
                "output_size": 6,
            },
            checkpoint_path,
        )
        actor = _make_actor()

        first = actor._get_frozen_policy(str(checkpoint_path))
        second = actor._get_frozen_policy(str(checkpoint_path))

        assert first is second
        assert isinstance(first, FrozenOpponentPolicy)
        assert first.is_frozen_opponent is True
        assert first.training is False
        assert first.memory is None
        action = first.select_action(torch.zeros(input_size))
        assert 0 <= int(action) < 6

    def test_get_frozen_policy_rejects_input_size_mismatch(self, tmp_path, setup_config):
        wrong_size = GameConfig.INPUT_SIZE + 3
        network = ApexNetwork(wrong_size, 32, 6)
        checkpoint_path = tmp_path / "mismatch.pth"
        torch.save(
            {
                "dqn_state_dict": network.state_dict(),
                "input_size": wrong_size,
                "hidden_size": 32,
                "output_size": 6,
            },
            checkpoint_path,
        )
        actor = _make_actor()

        with pytest.raises(ValueError, match="input size"):
            actor._get_frozen_policy(str(checkpoint_path))


# ============================================================================
# Per-episode slot assignment
# ============================================================================


class TestEpisodePolicyAssignment:
    """Latest/frozen slot sampling per episode."""

    def _pool_actor(self, num_snakes=6, pool_latest_fraction=0.8):
        actor = _make_actor(pool_latest_fraction=pool_latest_fraction)
        shared_policy = SimpleNamespace(training=False, memory=None, dqn=MagicMock())
        snakes = [_make_snake(i, shared_policy) for i in range(num_snakes)]
        actor.env = SimpleNamespace(snakes=snakes)
        actor._pool_checkpoint_paths = ["pool/a.pth", "pool/b.pth", "pool/c.pth"]
        frozen_cache = {}
        actor._get_frozen_policy = lambda path: frozen_cache.setdefault(
            path, _fake_frozen_policy(path)
        )
        return actor, shared_policy, snakes

    def test_no_pool_is_a_noop(self):
        actor = _make_actor()
        shared_policy = SimpleNamespace(training=False, memory=None)
        snakes = [_make_snake(i, shared_policy) for i in range(3)]
        actor.env = SimpleNamespace(snakes=snakes)

        actor._assign_episode_policies()

        assert all(snake.policy is shared_policy for snake in snakes)
        assert actor._episode_frozen_opponent_count == 0

    def test_sampling_distribution_over_100_episodes(self):
        """~20% of non-hero slots run frozen policies at pool_latest_fraction=0.8."""
        actor, shared_policy, snakes = self._pool_actor(num_snakes=6, pool_latest_fraction=0.8)
        np.random.seed(1234)

        frozen_slots = 0
        checkpoint_counts = Counter()
        for _ in range(100):
            actor._assign_episode_policies()
            # Hero slot is always the latest policy.
            assert snakes[0].policy is shared_policy
            assert snakes[0].actor_epsilon == pytest.approx(actor.epsilon)
            for snake in snakes[1:]:
                if getattr(snake.policy, "is_frozen_opponent", False):
                    frozen_slots += 1
                    checkpoint_counts[snake.policy.checkpoint_path] += 1
                    assert snake.actor_epsilon == pytest.approx(0.0)
                else:
                    assert snake.policy is shared_policy
                    assert snake.actor_epsilon == pytest.approx(actor.epsilon)

        total_opponent_slots = 100 * 5
        frozen_fraction = frozen_slots / total_opponent_slots
        assert frozen_fraction == pytest.approx(0.2, abs=0.05)
        # Frozen checkpoints are sampled uniformly: every pool member appears.
        assert set(checkpoint_counts) == set(actor._pool_checkpoint_paths)
        assert actor.pool_assigned_frozen_slot_count == frozen_slots
        assert actor.pool_assigned_latest_slot_count == total_opponent_slots - frozen_slots

    def test_all_frozen_at_zero_latest_fraction(self):
        actor, shared_policy, snakes = self._pool_actor(num_snakes=4, pool_latest_fraction=0.0)
        np.random.seed(7)

        actor._assign_episode_policies()

        assert snakes[0].policy is shared_policy
        assert all(snake.policy.is_frozen_opponent for snake in snakes[1:])
        assert actor._episode_frozen_opponent_count == 3

    def test_frozen_slot_can_return_to_latest_next_episode(self):
        actor, shared_policy, snakes = self._pool_actor(num_snakes=2)
        actor.pool_latest_fraction = 0.0
        np.random.seed(3)
        actor._assign_episode_policies()
        assert snakes[1].policy.is_frozen_opponent

        actor.pool_latest_fraction = 1.0
        actor._assign_episode_policies()

        assert snakes[1].policy is shared_policy
        assert actor._episode_frozen_opponent_count == 0

    def test_weight_sync_skips_frozen_policies(self):
        actor, shared_policy, snakes = self._pool_actor(num_snakes=2, pool_latest_fraction=0.0)
        actor.local_network = MagicMock()
        actor.local_network.state_dict.return_value = {"w": 1}
        actor.target_network = MagicMock()
        actor.target_network.state_dict.return_value = {"w": 1}
        np.random.seed(5)
        actor._assign_episode_policies()
        frozen_policy = snakes[1].policy
        assert frozen_policy.is_frozen_opponent

        actor._sync_snake_policy_weights()

        shared_policy.dqn.load_state_dict.assert_called_once_with({"w": 1})
        frozen_policy.dqn.load_state_dict.assert_not_called()


# ============================================================================
# Replay collection gating and telemetry
# ============================================================================


class TestFrozenReplayExclusion:
    """Frozen-driven snakes must never produce replay transitions."""

    def _episode_actor(self, num_snakes=2, death_frame=3, **kwargs):
        actor = _make_actor(n_step=1, **kwargs)
        actor.device = torch.device("cpu")
        shared_policy = SimpleNamespace(training=False, memory=None)
        snakes = [_make_snake(i, shared_policy) for i in range(num_snakes)]
        env = SimpleNamespace(frame=0, snakes=snakes, reset=MagicMock())

        def update(train_mode=True, learn=False):
            env.frame += 1
            for snake in snakes:
                snake.last_transition_frame = env.frame
                snake.last_reward = 1.0
                snake.last_done = env.frame == death_frame
                snake.is_alive = env.frame < death_frame

        env.update = MagicMock(side_effect=update)
        actor.env = env
        return actor, shared_policy, snakes, env

    def test_frozen_slot_transitions_are_not_collected(self):
        actor, shared_policy, snakes, _ = self._episode_actor(
            num_snakes=2, pool_latest_fraction=0.0
        )
        actor._pool_checkpoint_paths = ["pool/only.pth"]
        actor._get_frozen_policy = lambda path: _fake_frozen_policy(path)
        np.random.seed(11)

        episode_reward, episode_steps, collected = actor._run_episode()

        assert snakes[0].policy is shared_policy
        assert snakes[1].policy.is_frozen_opponent
        assert episode_steps == 3
        # Hero-only: 3 transitions and 3 reward, not 6 from both snakes.
        assert len(collected) == 3
        assert episode_reward == pytest.approx(3.0)

    def test_frozen_slot_never_reaches_buffer_client(self):
        actor, _, _, _ = self._episode_actor(num_snakes=2, pool_latest_fraction=0.0)
        actor.batch_send_size = 1
        actor._pool_checkpoint_paths = ["pool/only.pth"]
        actor._get_frozen_policy = lambda path: _fake_frozen_policy(path)
        np.random.seed(11)

        _, _, collected = actor._run_episode(stream_to_buffer=True)

        assert collected == []
        # One push per hero transition; the frozen snake contributes none.
        assert actor.buffer_client.add_batch.call_count == 3

    def test_without_pool_both_snakes_are_collected(self):
        actor, _, _, _ = self._episode_actor(num_snakes=2)

        episode_reward, episode_steps, collected = actor._run_episode()

        assert episode_steps == 3
        assert len(collected) == 6
        assert episode_reward == pytest.approx(6.0)

    def test_exposure_mix_reports_transition_fractions(self):
        actor, _, _, _ = self._episode_actor(num_snakes=2, pool_latest_fraction=0.0)
        actor._pool_checkpoint_paths = ["pool/only.pth"]
        actor._get_frozen_policy = lambda path: _fake_frozen_policy(path)
        np.random.seed(11)
        actor._run_episode()

        actor._send_stats(episode=1, avg_reward=0.0, total_steps=3)
        stats = actor.stats_queue.get_nowait()

        assert stats["opponent_pool_size"] == 1
        assert stats["pool_latest_fraction"] == pytest.approx(0.0)
        assert stats["pool_assigned_frozen_slot_count"] == 1
        assert stats["pool_assigned_frozen_slot_fraction"] == pytest.approx(1.0)
        # All collected transitions came from an episode with 1 frozen opponent.
        assert stats["pool_exposure_mix"] == {"1": pytest.approx(1.0)}


# ============================================================================
# Population-floor episode break
# ============================================================================


class TestPopulationFloorBreak:
    """Actor episodes end when GameState.population_floor_reached is True."""

    def _floor_actor(self, floor_frame=None, floor_value=True, death_frame=5):
        actor = _make_actor(n_step=1)
        snake = _make_snake(0, SimpleNamespace(training=False, memory=None))
        env = SimpleNamespace(
            frame=0, snakes=[snake], reset=MagicMock(), population_floor_reached=False
        )

        def update(train_mode=True, learn=False):
            env.frame += 1
            snake.last_transition_frame = env.frame
            snake.last_reward = 1.0
            snake.last_done = env.frame == death_frame
            snake.is_alive = env.frame < death_frame
            if floor_frame is not None and env.frame >= floor_frame:
                env.population_floor_reached = floor_value

        env.update = MagicMock(side_effect=update)
        actor.env = env
        return actor

    def test_breaks_when_floor_reached(self):
        actor = self._floor_actor(floor_frame=2)

        _, episode_steps, collected = actor._run_episode()

        assert episode_steps == 2
        assert len(collected) == 2

    def test_ignores_truthy_non_bool_values(self):
        """Guard is `is True`: mock-ish truthy values must not end episodes."""
        actor = self._floor_actor(floor_frame=2, floor_value=1)

        _, episode_steps, _ = actor._run_episode()

        assert episode_steps == 5

    def test_env_without_property_uses_legacy_break(self):
        actor = self._floor_actor()
        del actor.env.population_floor_reached

        _, episode_steps, _ = actor._run_episode()

        assert episode_steps == 5


# ============================================================================
# Config knob round-trips (YAML + CLI)
# ============================================================================


class TestKnobRoundTrip:
    """apex.opponent_pool_dir / apex.pool_latest_fraction / actor_priority_mode."""

    def test_yaml_round_trip(self, tmp_path):
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        config_path = tmp_path / "pool.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "apex": {
                        "opponent_pool_dir": str(pool_dir),
                        "pool_latest_fraction": 0.5,
                        "actor_priority_mode": "td",
                    }
                }
            )
        )

        config = load_config(str(config_path))

        assert config.apex.opponent_pool_dir == str(pool_dir)
        assert config.apex.pool_latest_fraction == pytest.approx(0.5)
        assert config.apex.actor_priority_mode == "td"

    def test_yaml_defaults(self):
        config = load_config(None)
        assert config.apex.opponent_pool_dir is None
        assert config.apex.pool_latest_fraction == pytest.approx(0.8)
        assert config.apex.actor_priority_mode == "max"

    def test_yaml_rejects_out_of_range_fraction(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(yaml.safe_dump({"apex": {"pool_latest_fraction": 1.5}}))

        with pytest.raises(ValueError):
            load_config(str(config_path))

    def test_cli_round_trip(self, monkeypatch):
        import src.scripts.apex_train as apex_train

        captured = {}
        monkeypatch.setattr(apex_train, "train_apex", lambda **kwargs: captured.update(kwargs))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "apex_train.py",
                "--opponent-pool-dir",
                "saved_snakes/pool",
                "--pool-latest-fraction",
                "0.6",
                "--actor-priority-mode",
                "td",
            ],
        )

        apex_train.main()

        assert captured["opponent_pool_dir"] == "saved_snakes/pool"
        assert captured["pool_latest_fraction"] == pytest.approx(0.6)
        assert captured["actor_priority_mode"] == "td"

    def test_cli_defaults_to_none_for_config_fallback(self, monkeypatch):
        import src.scripts.apex_train as apex_train

        captured = {}
        monkeypatch.setattr(apex_train, "train_apex", lambda **kwargs: captured.update(kwargs))
        monkeypatch.setattr(sys, "argv", ["apex_train.py"])

        apex_train.main()

        assert captured["opponent_pool_dir"] is None
        assert captured["pool_latest_fraction"] is None
        assert captured["actor_priority_mode"] is None
