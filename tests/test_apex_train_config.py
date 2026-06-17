"""Tests for Ape-X training coordinator configuration behavior."""

import queue

import pytest

from src.core.game_config import GameConfig
from src.core.reward_contract import current_reward_contract
from src.scripts.apex_train import (
    _mean_or_zero,
    _resolve_configurable,
    attach_replay_health_metadata,
    broadcast_weights,
    build_actor_replay_quality_gates,
    build_apex_checkpoint_config,
    collect_buffer_replay_health,
    format_actor_replay_summary,
    format_actor_replay_warnings,
    format_apex_checkpoint_provenance,
    format_buffer_replay_warnings,
    format_learner_sample_warnings,
    log_actor_replay_coverage,
    log_buffer_replay_health,
    log_learner_sample_health,
    resolve_actor_replay_quality_fraction,
    resolve_apex_min_buffer_size,
    should_report_buffer_replay_warnings,
    should_report_learner_sample_warnings,
    summarize_actor_replay_coverage,
    train_apex,
    update_latest_actor_stats,
    validate_actor_replay_quality_gates,
    validate_apex_resume_checkpoint_config,
    validate_apex_training_config,
)


def test_resolve_configurable_prefers_explicit_value():
    """Explicit CLI values should override config and fallback defaults."""
    assert _resolve_configurable(8, configured_value=64, fallback=4, use_config=True) == 8


def test_resolve_configurable_uses_config_when_requested():
    """Omitted CLI values should use YAML-derived config values when loaded."""
    assert _resolve_configurable(None, configured_value=64, fallback=4, use_config=True) == 64


def test_resolve_configurable_uses_legacy_default_without_config():
    """No-config launches keep the previous local-friendly defaults."""
    assert _resolve_configurable(None, configured_value=64, fallback=4, use_config=False) == 4


def test_resolve_actor_replay_quality_fraction_defaults_to_disabled_gate():
    """Omitting the CLI gate should keep distributed training warning-only."""
    assert resolve_actor_replay_quality_fraction(None, "min_actor_terminal_fraction") == 0.0


def test_resolve_actor_replay_quality_fraction_accepts_valid_fraction():
    """Explicit actor replay gates should preserve the requested fraction."""
    assert resolve_actor_replay_quality_fraction(
        0.005, "min_actor_terminal_fraction"
    ) == pytest.approx(0.005)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_resolve_actor_replay_quality_fraction_rejects_invalid_fraction(value):
    """Replay quality gates should fail before the expensive runtime starts."""
    with pytest.raises(ValueError, match="min_actor_terminal_fraction"):
        resolve_actor_replay_quality_fraction(value, "min_actor_terminal_fraction")


def test_build_actor_replay_quality_gates_records_terminal_floor():
    """Checkpoint metadata should carry the configured terminal replay floor."""
    assert build_actor_replay_quality_gates(0.005) == {
        "min_actor_terminal_fraction": pytest.approx(0.005)
    }


def test_build_apex_checkpoint_config_records_distributed_training_contract():
    """Distributed checkpoints should preserve the resolved run contract."""
    config = build_apex_checkpoint_config(
        num_actors=4,
        total_steps=10_000,
        batch_size=128,
        buffer_capacity=50_000,
        n_step=5,
        min_buffer_size=512,
        learning_rate=0.00025,
        gamma=0.97,
        target_update_freq=2000,
        weight_broadcast_interval=300,
        priority_alpha=0.55,
        priority_beta_start=0.2,
        priority_beta_end=0.9,
        priority_beta_frames=10_000,
        priority_epsilon=1e-5,
        grad_clip_norm=7.5,
        log_interval=100,
        checkpoint_interval=1000,
        actor_env_num_snakes=6,
        actor_board_scale=0.2,
        actor_food_multiplier=0.5,
        actor_boost_exploration_rate=0.25,
        actor_danger_exploration_rate=0.02,
        input_size=58,
        hidden_size=512,
        output_size=6,
        reward_death=-7.0,
        reward_food_base=3.0,
    )

    assert config == {
        "actor_env_num_snakes": 6,
        "actor_board_scale": 0.2,
        "actor_food_multiplier": 0.5,
        "actor_boost_exploration_rate": 0.25,
        "actor_danger_exploration_rate": 0.02,
        "batch_size": 128,
        "buffer_size": 50_000,
        "checkpoint_interval": 1000,
        "gamma": 0.97,
        "grad_clip_norm": 7.5,
        "hidden_size": 512,
        "input_size": 58,
        "learning_rate": 0.00025,
        "log_interval": 100,
        "min_replay_size": 512,
        "n_step": 5,
        "num_actors": 4,
        "output_size": 6,
        "priority_alpha": 0.55,
        "priority_beta_end": 0.9,
        "priority_beta_frames": 10_000,
        "priority_beta_start": 0.2,
        "priority_epsilon": 1e-5,
        "reward_contract": current_reward_contract(),
        "reward_death": -7.0,
        "reward_food_base": 3.0,
        "target_update_freq": 2000,
        "total_steps": 10_000,
        "use_gru": False,
        "weight_broadcast_interval": 300,
    }


def test_format_apex_checkpoint_provenance_returns_compact_line():
    """Resume logs should expose the training contract saved in checkpoints."""
    lines = format_apex_checkpoint_provenance(
        {
            "apex_config": {
                "actor_env_num_snakes": 6,
                "actor_board_scale": 0.2,
                "actor_food_multiplier": 0.5,
                "actor_boost_exploration_rate": 0.25,
                "actor_danger_exploration_rate": 0.02,
                "batch_size": 128,
                "buffer_size": 50_000,
                "gamma": 0.97,
                "min_replay_size": 512,
                "n_step": 5,
                "num_actors": 4,
                "priority_alpha": 0.55,
                "priority_beta_end": 0.9,
                "priority_beta_start": 0.2,
                "priority_epsilon": 1e-5,
                "reward_death": -7.0,
                "reward_food_base": 3.0,
                "target_update_freq": 2000,
            }
        }
    )

    assert lines == [
        (
            "Checkpoint Apex config: actors=4 | actor_snakes=6 | actor_board=0.2 | "
            "actor_food=0.5 | actor_boost=0.25 | actor_danger=0.02 | batch=128 | "
            "buffer=50000 | warmup=512 | n_step=5 | "
            "gamma=0.97 | target_sync=2000 | reward death=-7.0, food=3.0 | "
            "PER alpha=0.55, beta=0.2->0.9, eps=1e-05"
        )
    ]


def test_format_apex_checkpoint_provenance_supports_legacy_learner_config():
    """Older learner checkpoints should still print useful resume metadata."""
    lines = format_apex_checkpoint_provenance(
        {
            "config": {
                "batch_size": 64,
                "gamma": 0.99,
                "min_buffer_size": 256,
                "priority_alpha": 0.6,
                "priority_eps": 1e-6,
                "target_update_freq": 500,
            }
        }
    )

    assert lines == [
        (
            "Checkpoint Apex config: actors=None | actor_snakes=None | actor_board=None | "
            "actor_food=None | actor_boost=None | actor_danger=None | batch=64 | "
            "buffer=None | warmup=256 | n_step=None | "
            "gamma=0.99 | target_sync=500 | reward death=None, food=None | "
            "PER alpha=0.6, beta=None->None, eps=1e-06"
        )
    ]


def test_validate_apex_resume_checkpoint_config_accepts_matching_contract():
    """Distributed resume should accept checkpoints from the same training contract."""
    expected_config = build_apex_checkpoint_config(
        num_actors=4,
        total_steps=10_000,
        batch_size=128,
        buffer_capacity=50_000,
        n_step=5,
        min_buffer_size=512,
        learning_rate=0.00025,
        gamma=0.97,
        target_update_freq=2000,
        weight_broadcast_interval=300,
        priority_alpha=0.55,
        priority_beta_start=0.2,
        priority_beta_end=0.9,
        priority_beta_frames=10_000,
        priority_epsilon=1e-5,
        grad_clip_norm=7.5,
        log_interval=100,
        checkpoint_interval=1000,
        actor_env_num_snakes=6,
        actor_board_scale=0.2,
        actor_food_multiplier=0.5,
        actor_boost_exploration_rate=0.25,
        actor_danger_exploration_rate=0.02,
        input_size=58,
        hidden_size=512,
        output_size=6,
        reward_death=-7.0,
        reward_food_base=3.0,
    )

    validate_apex_resume_checkpoint_config(
        {"apex_config": dict(expected_config)},
        expected_config,
        checkpoint_path="apex_checkpoint.pth",
    )


def test_validate_apex_resume_checkpoint_config_allows_exploration_rate_changes():
    """Resume fine-tunes may change future actor exploration without changing TD semantics."""
    expected_config = build_apex_checkpoint_config(
        num_actors=4,
        total_steps=10_000,
        batch_size=128,
        buffer_capacity=50_000,
        n_step=3,
        min_buffer_size=512,
        learning_rate=0.00025,
        gamma=0.99,
        target_update_freq=2000,
        weight_broadcast_interval=300,
        priority_alpha=0.55,
        priority_beta_start=0.2,
        priority_beta_end=0.9,
        priority_beta_frames=10_000,
        priority_epsilon=1e-5,
        grad_clip_norm=7.5,
        log_interval=100,
        checkpoint_interval=1000,
        actor_env_num_snakes=6,
        actor_board_scale=0.2,
        actor_food_multiplier=0.5,
        actor_boost_exploration_rate=0.0,
        actor_danger_exploration_rate=0.0,
        input_size=58,
        hidden_size=512,
        output_size=6,
        reward_death=GameConfig.REWARD_DEATH,
        reward_food_base=GameConfig.REWARD_FOOD_BASE,
    )
    checkpoint_config = dict(expected_config)
    checkpoint_config["actor_boost_exploration_rate"] = 0.25
    checkpoint_config["actor_danger_exploration_rate"] = 0.02

    validate_apex_resume_checkpoint_config(
        {"apex_config": checkpoint_config},
        expected_config,
        checkpoint_path="apex_checkpoint.pth",
    )


def test_validate_apex_resume_checkpoint_config_rejects_missing_reward_contract():
    """Distributed resume should not accept legacy checkpoints with unknown reward scale."""
    expected_config = {
        "input_size": 58,
        "hidden_size": 512,
        "output_size": 6,
        "n_step": 3,
        "gamma": 0.99,
        "reward_contract": current_reward_contract(),
        "reward_death": GameConfig.REWARD_DEATH,
        "reward_food_base": 3.0,
    }
    checkpoint = {
        "apex_config": {
            key: value for key, value in expected_config.items() if key != "reward_contract"
        }
    }

    with pytest.raises(ValueError, match="missing required reward_contract"):
        validate_apex_resume_checkpoint_config(
            checkpoint,
            expected_config,
            checkpoint_path="legacy_apex_checkpoint.pth",
        )


def test_validate_apex_resume_checkpoint_config_rejects_reward_mismatch():
    """Distributed resume should not silently switch terminal reward scale."""
    expected_config = {
        "input_size": 58,
        "hidden_size": 512,
        "output_size": 6,
        "n_step": 3,
        "gamma": 0.99,
        "reward_contract": current_reward_contract(),
        "reward_death": GameConfig.REWARD_DEATH,
        "reward_food_base": 3.0,
    }
    checkpoint = {"apex_config": {**expected_config, "reward_death": -3.0}}

    with pytest.raises(ValueError, match="reward_death=-3"):
        validate_apex_resume_checkpoint_config(
            checkpoint,
            expected_config,
            checkpoint_path="stale_apex_checkpoint.pth",
        )


def test_validate_apex_resume_checkpoint_config_rejects_full_reward_mismatch():
    """Distributed resume should not silently switch shaping rewards."""
    expected_config = {
        "input_size": 58,
        "hidden_size": 512,
        "output_size": 6,
        "n_step": 3,
        "gamma": 0.99,
        "reward_contract": current_reward_contract(),
        "reward_death": GameConfig.REWARD_DEATH,
        "reward_food_base": 3.0,
    }
    stale_contract = dict(expected_config["reward_contract"])
    stale_contract["survival"] = float(stale_contract["survival"]) + 1.0
    checkpoint = {"apex_config": {**expected_config, "reward_contract": stale_contract}}

    with pytest.raises(ValueError, match="reward_contract.survival"):
        validate_apex_resume_checkpoint_config(
            checkpoint,
            expected_config,
            checkpoint_path="stale_apex_checkpoint.pth",
        )


def test_validate_apex_resume_checkpoint_config_rejects_n_step_mismatch():
    """Resume should not silently switch TD-target horizon from the checkpoint."""
    expected_config = {"input_size": 58, "hidden_size": 512, "output_size": 6, "n_step": 3}
    checkpoint = {"apex_config": {**expected_config, "n_step": 5}}

    with pytest.raises(ValueError, match="n_step"):
        validate_apex_resume_checkpoint_config(
            checkpoint,
            expected_config,
            checkpoint_path="apex_checkpoint.pth",
        )


def test_validate_apex_resume_checkpoint_config_rejects_gamma_mismatch():
    """Resume should not silently change the discount used for learner targets."""
    expected_config = {"input_size": 58, "hidden_size": 512, "output_size": 6, "gamma": 0.99}
    checkpoint = {"apex_config": {**expected_config, "gamma": 0.95}}

    with pytest.raises(ValueError, match="gamma"):
        validate_apex_resume_checkpoint_config(
            checkpoint,
            expected_config,
            checkpoint_path="apex_checkpoint.pth",
        )


def test_validate_apex_resume_checkpoint_config_rejects_actor_environment_mismatch():
    """Resume should not silently change the actor replay distribution."""
    expected_config = {
        "input_size": 58,
        "hidden_size": 512,
        "output_size": 6,
        "actor_env_num_snakes": 6,
        "actor_board_scale": 0.2,
        "actor_food_multiplier": 0.5,
    }
    checkpoint = {
        "apex_config": {
            **expected_config,
            "actor_board_scale": 1.0,
        }
    }

    with pytest.raises(ValueError, match="actor_board_scale"):
        validate_apex_resume_checkpoint_config(
            checkpoint,
            expected_config,
            checkpoint_path="apex_checkpoint.pth",
        )


def test_validate_apex_resume_checkpoint_config_checks_legacy_learner_config_gamma():
    """Legacy learner checkpoints still carry target semantics through config.gamma."""
    expected_config = {"input_size": 58, "hidden_size": 512, "output_size": 6, "gamma": 0.99}
    checkpoint = {"config": {"input_size": 58, "hidden_size": 512, "output_size": 6, "gamma": 0.95}}

    with pytest.raises(ValueError, match="gamma"):
        validate_apex_resume_checkpoint_config(
            checkpoint,
            expected_config,
            checkpoint_path="legacy_checkpoint.pth",
        )


def test_mean_or_zero_returns_zero_for_empty_values():
    """Empty reward windows should log as zero without NumPy."""
    assert _mean_or_zero([]) == 0.0


def test_mean_or_zero_returns_arithmetic_mean():
    """Reward logging should use the arithmetic mean."""
    assert _mean_or_zero([1.0, 2.0, 6.0]) == 3.0


def test_broadcast_weights_does_not_trust_empty_when_dropping_stale_weights():
    """Actor queues should carry the newest weights even when Queue.empty() lies."""

    class FullButEmptyLiesQueue:
        def __init__(self):
            self.items = [{"version": "stale"}]

        def empty(self):
            return True

        def get_nowait(self):
            if not self.items:
                raise queue.Empty
            return self.items.pop(0)

        def put_nowait(self, item):
            if self.items:
                raise queue.Full
            self.items.append(item)

    weights = {"version": "fresh"}
    weight_queue = FullButEmptyLiesQueue()

    broadcast_weights(weights, [weight_queue])

    assert weight_queue.items == [weights]


def test_broadcast_weights_keeps_sending_after_one_queue_fails():
    """A closed actor queue should not prevent other actors from receiving weights."""

    class ClosedQueue:
        def get_nowait(self):
            raise ValueError("queue is closed")

        def put_nowait(self, item):
            raise AssertionError("closed queues should not receive new weights")

    class OpenQueue:
        def __init__(self):
            self.items = []

        def get_nowait(self):
            raise queue.Empty

        def put_nowait(self, item):
            self.items.append(item)

    weights = {"version": "fresh"}
    open_queue = OpenQueue()

    broadcast_weights(weights, [ClosedQueue(), open_queue])

    assert open_queue.items == [weights]


def test_update_latest_actor_stats_keeps_one_snapshot_per_actor():
    """Actor stats are cumulative, so the coordinator should keep latest per actor."""
    latest = {}
    rewards = []

    update_latest_actor_stats(
        latest,
        [
            {"actor_id": 0, "avg_reward": 1.0, "sent_experience_count": 2},
            {"actor_id": 1, "avg_reward": 2.0, "sent_experience_count": 3},
            {"actor_id": 0, "avg_reward": 4.0, "sent_experience_count": 5},
        ],
        episode_rewards=rewards,
    )

    assert rewards == [1.0, 2.0, 4.0]
    assert latest[0]["sent_experience_count"] == 5
    assert latest[1]["sent_experience_count"] == 3


def test_summarize_actor_replay_coverage_weights_latest_actor_stats():
    """Coordinator logs should expose actor-side action/mask/terminal coverage."""
    summary = summarize_actor_replay_coverage(
        [
            {
                "sent_experience_count": 2,
                "sent_action_counts": [1, 0, 0, 1, 0, 0],
                "sent_boost_action_fraction": 0.5,
                "sent_exact_mask_fraction": 1.0,
                "sent_terminal_count": 0,
                "sent_terminal_fraction": 0.0,
                "sent_nonterminal_count": 2,
                "sent_nonterminal_exact_mask_fraction": 1.0,
                "sent_nonterminal_trapped_next_fraction": 0.0,
                "sent_positive_reward_fraction": 0.5,
                "sent_zero_reward_fraction": 0.5,
                "sent_negative_reward_fraction": 0.0,
                "sent_multistep_fraction": 1.0,
                "sent_invalid_current_action_count": 1,
                "sent_invalid_current_normal_action_count": 0,
                "sent_invalid_current_boost_action_count": 1,
                "buffer_queued_message_count": 2,
                "buffer_dropped_message_count": 1,
                "buffer_dropped_experience_count": 1,
                "buffer_last_drop_error": "queue full",
            },
            {
                "sent_experience_count": 3,
                "sent_action_counts": [0, 2, 1, 0, 0, 0],
                "sent_boost_action_fraction": 0.0,
                "sent_exact_mask_fraction": 1 / 3,
                "sent_terminal_count": 2,
                "sent_terminal_fraction": 2 / 3,
                "sent_nonterminal_count": 1,
                "sent_nonterminal_exact_mask_fraction": 1.0,
                "sent_nonterminal_trapped_next_fraction": 1.0,
                "sent_positive_reward_fraction": 1 / 3,
                "sent_zero_reward_fraction": 1 / 3,
                "sent_negative_reward_fraction": 1 / 3,
                "sent_multistep_fraction": 1 / 3,
                "sent_invalid_current_action_count": 1,
                "sent_invalid_current_normal_action_count": 1,
                "sent_invalid_current_boost_action_count": 0,
                "buffer_queued_message_count": 3,
                "buffer_dropped_message_count": 1,
                "buffer_dropped_experience_count": 2,
                "buffer_last_drop_error": "queue still full",
            },
        ]
    )

    assert summary["sent_experience_count"] == 5
    assert summary["sent_action_counts"] == [1, 2, 1, 1, 0, 0]
    assert summary["sent_active_action_count"] == 4
    assert summary["sent_boost_action_fraction"] == pytest.approx(0.2)
    assert summary["sent_exact_mask_fraction"] == pytest.approx(0.6)
    assert summary["sent_terminal_count"] == 2
    assert summary["sent_terminal_fraction"] == pytest.approx(0.4)
    assert summary["sent_nonterminal_count"] == 3
    assert summary["sent_nonterminal_exact_mask_fraction"] == pytest.approx(1.0)
    assert summary["sent_nonterminal_trapped_next_fraction"] == pytest.approx(1 / 3)
    assert summary["sent_positive_reward_fraction"] == pytest.approx(0.4)
    assert summary["sent_zero_reward_fraction"] == pytest.approx(0.4)
    assert summary["sent_negative_reward_fraction"] == pytest.approx(0.2)
    assert summary["sent_multistep_fraction"] == pytest.approx(0.6)
    assert summary["sent_invalid_current_action_count"] == 2
    assert summary["sent_invalid_current_action_fraction"] == pytest.approx(0.4)
    assert summary["sent_invalid_current_normal_action_count"] == 1
    assert summary["sent_invalid_current_boost_action_count"] == 1
    assert summary["buffer_queued_message_count"] == 5
    assert summary["buffer_dropped_message_count"] == 2
    assert summary["buffer_dropped_experience_count"] == 3
    assert summary["buffer_dropped_experience_fraction"] == pytest.approx(0.6)
    assert summary["buffer_last_drop_error"] == "queue still full"


def test_format_actor_replay_warnings_flags_weak_learning_signal():
    """Coordinator warnings should name actor replay streams that are structurally weak."""
    warnings = format_actor_replay_warnings(
        {
            "sent_experience_count": 256,
            "sent_active_action_count": 2,
            "sent_positive_reward_fraction": 0.0,
            "sent_negative_reward_fraction": 0.0,
            "sent_terminal_count": 0,
            "sent_terminal_fraction": 0.0,
            "sent_exact_mask_fraction": 0.25,
            "sent_nonterminal_exact_mask_fraction": 0.25,
            "sent_nonterminal_trapped_next_fraction": 0.5,
            "sent_multistep_fraction": 0.0,
            "sent_invalid_current_action_count": 64,
            "sent_invalid_current_action_fraction": 0.25,
            "buffer_dropped_experience_count": 32,
            "buffer_dropped_experience_fraction": 0.125,
            "buffer_last_drop_error": "queue full",
        }
    )

    assert any("Only 2/6 actions" in warning for warning in warnings)
    assert any("No positive rewards" in warning for warning in warnings)
    assert any("No negative rewards" in warning for warning in warnings)
    assert any("No terminal rows" in warning for warning in warnings)
    assert any("exact next-action masks" in warning for warning in warnings)
    assert any("nonterminal actor transitions" in warning for warning in warnings)
    assert any("no valid next actions" in warning for warning in warnings)
    assert any("bootstrap_steps=1" in warning for warning in warnings)
    assert any("invalid under current-state" in warning for warning in warnings)
    assert any("dropped before reaching the buffer" in warning for warning in warnings)


def test_format_actor_replay_warnings_flags_sparse_terminal_signal():
    """Tiny terminal fractions should still warn even when not exactly zero."""
    warnings = format_actor_replay_warnings(
        {
            "sent_experience_count": 10_000,
            "sent_active_action_count": 6,
            "sent_positive_reward_fraction": 0.2,
            "sent_negative_reward_fraction": 0.2,
            "sent_terminal_count": 3,
            "sent_terminal_fraction": 0.0003,
            "sent_exact_mask_fraction": 1.0,
            "sent_nonterminal_exact_mask_fraction": 1.0,
            "sent_nonterminal_trapped_next_fraction": 0.0,
            "sent_multistep_fraction": 1.0,
            "sent_invalid_current_action_count": 0,
            "sent_invalid_current_action_fraction": 0.0,
            "buffer_dropped_experience_count": 0,
            "buffer_dropped_experience_fraction": 0.0,
        }
    )

    assert any("Only 3/10,000" in warning for warning in warnings)
    assert any("0.03%" in warning for warning in warnings)


def test_format_actor_replay_warnings_flags_low_invalid_action_drift():
    """Actor health should flag invalid current actions above the training replay budget."""
    warnings = format_actor_replay_warnings(
        {
            "sent_experience_count": 1_000,
            "sent_active_action_count": 6,
            "sent_positive_reward_fraction": 0.2,
            "sent_negative_reward_fraction": 0.2,
            "sent_terminal_count": 20,
            "sent_terminal_fraction": 0.02,
            "sent_exact_mask_fraction": 1.0,
            "sent_nonterminal_exact_mask_fraction": 1.0,
            "sent_nonterminal_trapped_next_fraction": 0.0,
            "sent_multistep_fraction": 1.0,
            "sent_invalid_current_action_count": 30,
            "sent_invalid_current_action_fraction": 0.03,
            "buffer_dropped_experience_count": 0,
            "buffer_dropped_experience_fraction": 0.0,
        }
    )

    assert any("30/1,000 sent actor transitions" in warning for warning in warnings)
    assert any("(3.0%) are invalid under current-state" in warning for warning in warnings)


def test_validate_actor_replay_quality_gates_accepts_terminal_fraction_floor():
    """Configured final gates should pass when actor replay has enough terminal signal."""
    validate_actor_replay_quality_gates(
        {
            "sent_experience_count": 1_000,
            "sent_terminal_count": 10,
            "sent_terminal_fraction": 0.01,
        },
        min_terminal_fraction=0.005,
    )


def test_validate_actor_replay_quality_gates_rejects_sparse_terminal_fraction():
    """Configured final gates should fail noisy runs with too little collision signal."""
    with pytest.raises(RuntimeError, match="terminal fraction .*0.30%.*3/1,000.*0.50%"):
        validate_actor_replay_quality_gates(
            {
                "sent_experience_count": 1_000,
                "sent_terminal_count": 3,
                "sent_terminal_fraction": 0.003,
            },
            min_terminal_fraction=0.005,
        )


def test_validate_actor_replay_quality_gates_stays_disabled_by_default():
    """Default training should still finish with warnings instead of hard gates."""
    validate_actor_replay_quality_gates(
        {
            "sent_experience_count": 1_000,
            "sent_terminal_count": 0,
            "sent_terminal_fraction": 0.0,
        }
    )


def test_format_actor_replay_summary_includes_learning_signals():
    """Actor replay summaries should expose the signals needed to debug learning."""
    line = format_actor_replay_summary(
        {
            "sent_experience_count": 128,
            "sent_action_counts": [64, 32, 32, 0, 0, 0],
            "sent_active_action_count": 3,
            "sent_boost_action_fraction": 0.0,
            "sent_exact_mask_fraction": 0.9,
            "sent_nonterminal_exact_mask_fraction": 0.95,
            "sent_nonterminal_trapped_next_fraction": 0.05,
            "sent_terminal_count": 13,
            "sent_terminal_fraction": 0.1,
            "sent_positive_reward_fraction": 0.2,
            "sent_multistep_fraction": 0.8,
            "sent_invalid_current_action_fraction": 0.03,
            "buffer_dropped_experience_fraction": 0.01,
        }
    )

    assert "sent=128" in line
    assert "actions=3/6" in line
    assert "nt_masks=95.0%" in line
    assert "terminal=13/128 (10.00%)" in line
    assert "reward+=20.0%" in line
    assert "invalid_actions=3.0%" in line


def test_attach_replay_health_metadata_persists_warnings():
    """Checkpoints should preserve replay warnings for post-run diagnosis."""
    state = {"weights": "placeholder"}
    actor_replay = {
        "sent_experience_count": 256,
        "sent_active_action_count": 6,
        "sent_positive_reward_fraction": 0.2,
        "sent_negative_reward_fraction": 0.2,
        "sent_terminal_fraction": 0.0,
        "sent_exact_mask_fraction": 1.0,
        "sent_nonterminal_exact_mask_fraction": 1.0,
        "sent_nonterminal_trapped_next_fraction": 0.0,
        "sent_multistep_fraction": 1.0,
        "sent_invalid_current_action_count": 0,
        "sent_invalid_current_action_fraction": 0.0,
        "buffer_dropped_experience_count": 0,
        "buffer_dropped_experience_fraction": 0.0,
    }
    buffer_replay_health = {
        "total_rejected_actor_messages": 1,
        "last_rejected_actor_message": "next_action_mask shape mismatch",
        "total_rejected_priority_updates": 0,
    }

    result = attach_replay_health_metadata(
        state,
        actor_replay=actor_replay,
        buffer_replay_health=buffer_replay_health,
        actor_replay_gates={"min_actor_terminal_fraction": 0.005},
    )

    assert result is state
    assert result["actor_replay_coverage"] == actor_replay
    assert any("No terminal rows" in warning for warning in result["actor_replay_warnings"])
    assert result["actor_replay_gates"] == {"min_actor_terminal_fraction": 0.005}
    assert result["buffer_replay_health"] == buffer_replay_health
    assert any("actor replay message" in warning for warning in result["buffer_replay_warnings"])


def test_format_actor_replay_warnings_ignores_small_startup_samples():
    """Small startup actor samples should not produce noisy warnings."""
    warnings = format_actor_replay_warnings(
        {
            "sent_experience_count": 16,
            "sent_active_action_count": 1,
            "sent_positive_reward_fraction": 0.0,
            "sent_negative_reward_fraction": 0.0,
            "sent_terminal_fraction": 0.0,
            "sent_exact_mask_fraction": 0.0,
            "sent_nonterminal_exact_mask_fraction": 0.0,
            "sent_nonterminal_trapped_next_fraction": 0.0,
            "sent_multistep_fraction": 0.0,
        }
    )

    assert warnings == []


def test_log_actor_replay_coverage_writes_tensorboard_scalars():
    """Actor replay diagnostics should reach TensorBoard during long runs."""

    class FakeLogger:
        def __init__(self):
            self.scalars = []

        def log_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

    logger = FakeLogger()
    coverage = {
        "sent_experience_count": 10,
        "sent_action_counts": [2, 3, 0, 5, 0, 0],
        "sent_active_action_count": 3,
        "sent_boost_action_fraction": 0.5,
        "sent_exact_mask_fraction": 0.8,
        "sent_terminal_fraction": 0.2,
        "sent_nonterminal_exact_mask_fraction": 1.0,
        "sent_nonterminal_trapped_next_fraction": 0.1,
        "sent_positive_reward_fraction": 0.3,
        "sent_zero_reward_fraction": 0.4,
        "sent_negative_reward_fraction": 0.3,
        "sent_multistep_fraction": 0.7,
        "sent_invalid_current_action_count": 1,
        "sent_invalid_current_action_fraction": 0.1,
        "buffer_dropped_message_count": 2,
        "buffer_dropped_experience_count": 1,
        "buffer_dropped_experience_fraction": 0.1,
    }

    logged = log_actor_replay_coverage(logger, coverage, step=42)

    assert logged is True
    assert ("actor_replay/sent_experience_count", 10, 42) in logger.scalars
    assert ("actor_replay/sent_active_action_count", 3, 42) in logger.scalars
    assert ("actor_replay/sent_boost_action_fraction", 0.5, 42) in logger.scalars
    assert ("actor_replay/sent_exact_mask_fraction", 0.8, 42) in logger.scalars
    assert ("actor_replay/sent_terminal_fraction", 0.2, 42) in logger.scalars
    assert ("actor_replay/sent_nonterminal_exact_mask_fraction", 1.0, 42) in logger.scalars
    assert ("actor_replay/sent_nonterminal_trapped_next_fraction", 0.1, 42) in logger.scalars
    assert ("actor_replay/sent_positive_reward_fraction", 0.3, 42) in logger.scalars
    assert ("actor_replay/sent_zero_reward_fraction", 0.4, 42) in logger.scalars
    assert ("actor_replay/sent_negative_reward_fraction", 0.3, 42) in logger.scalars
    assert ("actor_replay/sent_multistep_fraction", 0.7, 42) in logger.scalars
    assert ("actor_replay/sent_invalid_current_action_fraction", 0.1, 42) in logger.scalars
    assert ("actor_replay/sent_invalid_current_action_count", 1, 42) in logger.scalars
    assert ("actor_replay/buffer_dropped_experience_count", 1, 42) in logger.scalars
    assert ("actor_replay/buffer_dropped_experience_fraction", 0.1, 42) in logger.scalars
    assert ("actor_replay/buffer_dropped_message_count", 2, 42) in logger.scalars
    assert ("actor_replay/warning_count", 0, 42) in logger.scalars
    assert ("actor_replay/action_0_count", 2, 42) in logger.scalars
    assert ("actor_replay/action_0_fraction", 0.2, 42) in logger.scalars
    assert ("actor_replay/action_3_count", 5, 42) in logger.scalars
    assert ("actor_replay/action_3_fraction", 0.5, 42) in logger.scalars


def test_log_actor_replay_coverage_skips_missing_stats():
    """Empty coverage should not write misleading zero-only TensorBoard rows."""

    class FakeLogger:
        def __init__(self):
            self.scalars = []

        def log_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

    logger = FakeLogger()

    assert log_actor_replay_coverage(None, {"sent_experience_count": 10}, step=7) is False
    assert log_actor_replay_coverage(logger, {"sent_experience_count": 0}, step=7) is False
    assert logger.scalars == []


def test_collect_buffer_replay_health_reads_timeout_aware_buffer_stats():
    """Coordinator should read distributed buffer stats without assuming local APIs."""

    class FakeBufferClient:
        def __init__(self):
            self.timeout = None

        def get_stats(self, timeout):
            self.timeout = timeout
            return {
                "size": 9,
                "total_added": 10,
                "total_rejected_actor_messages": 1,
                "last_rejected_actor_message": "actions=1",
            }

    buffer_client = FakeBufferClient()

    stats = collect_buffer_replay_health(buffer_client)

    assert buffer_client.timeout == pytest.approx(0.1)
    assert stats["size"] == 9
    assert stats["total_rejected_actor_messages"] == 1
    assert stats["last_rejected_actor_message"] == "actions=1"


def test_collect_buffer_replay_health_reads_local_buffer_stats_without_timeout():
    """Coordinator should also support LocalApexBuffer-style get_stats()."""

    class FakeLocalBuffer:
        def get_stats(self):
            return {"size": 3, "total_added": 4}

    assert collect_buffer_replay_health(FakeLocalBuffer()) == {"size": 3, "total_added": 4}


def test_format_buffer_replay_warnings_reports_rejected_actor_messages():
    """Rejected actor inserts are a learning-data health warning."""
    warnings = format_buffer_replay_warnings(
        {
            "total_rejected_actor_messages": 2,
            "last_rejected_actor_message": "Replay batch fields are misaligned: actions=1",
        }
    )

    assert len(warnings) == 1
    assert "2 actor replay message" in warnings[0]
    assert "actions=1" in warnings[0]


def test_format_buffer_replay_warnings_reports_rejected_priority_updates():
    """Rejected learner priority updates are a replay-feedback health warning."""
    warnings = format_buffer_replay_warnings(
        {
            "total_rejected_priority_updates": 3,
            "last_rejected_priority_update": "td_errors must be finite",
        }
    )

    assert len(warnings) == 1
    assert "3 learner priority update" in warnings[0]
    assert "td_errors" in warnings[0]


def test_should_report_buffer_replay_warnings_only_for_new_rejections():
    """Waiting loop should report buffer warnings when rejection count advances."""
    assert should_report_buffer_replay_warnings({}, last_rejected_actor_messages=0) is False
    assert (
        should_report_buffer_replay_warnings(
            {"total_rejected_actor_messages": 1},
            last_rejected_actor_messages=0,
        )
        is True
    )
    assert (
        should_report_buffer_replay_warnings(
            {"total_rejected_actor_messages": 1},
            last_rejected_actor_messages=1,
        )
        is False
    )
    assert (
        should_report_buffer_replay_warnings(
            {"total_rejected_priority_updates": 1},
            last_rejected_actor_messages=0,
            last_rejected_priority_updates=0,
        )
        is True
    )
    assert (
        should_report_buffer_replay_warnings(
            {"total_rejected_priority_updates": 1},
            last_rejected_actor_messages=0,
            last_rejected_priority_updates=1,
        )
        is False
    )


def test_log_buffer_replay_health_writes_tensorboard_scalars():
    """Buffer insertion health should reach TensorBoard during long runs."""

    class FakeLogger:
        def __init__(self):
            self.scalars = []

        def log_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

    logger = FakeLogger()
    buffer_stats = {
        "size": 7,
        "total_added": 11,
        "total_sampled": 5,
        "fill_ratio": 0.7,
        "total_rejected_actor_messages": 1,
        "last_rejected_actor_message": "actions=1",
        "total_rejected_priority_updates": 2,
        "last_rejected_priority_update": "td_errors must be finite",
    }

    logged = log_buffer_replay_health(logger, buffer_stats, step=42)

    assert logged is True
    assert ("buffer/replay_size", 7, 42) in logger.scalars
    assert ("buffer/total_added", 11, 42) in logger.scalars
    assert ("buffer/total_sampled", 5, 42) in logger.scalars
    assert ("buffer/fill_ratio", 0.7, 42) in logger.scalars
    assert ("buffer/total_rejected_actor_messages", 1, 42) in logger.scalars
    assert ("buffer/total_rejected_priority_updates", 2, 42) in logger.scalars
    assert ("buffer/replay_warning_count", 2, 42) in logger.scalars


def test_log_buffer_replay_health_skips_missing_stats():
    """Missing buffer stats should not produce misleading zero-only TensorBoard rows."""

    class FakeLogger:
        def __init__(self):
            self.scalars = []

        def log_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

    logger = FakeLogger()

    assert log_buffer_replay_health(None, {"size": 1}, step=7) is False
    assert log_buffer_replay_health(logger, {}, step=7) is False
    assert logger.scalars == []


def test_format_learner_sample_warnings_reports_sample_errors():
    """Learner sample failures after warmup are learning-data health warnings."""
    warnings = format_learner_sample_warnings(
        {
            "sample_error_count": 2,
            "last_sample_error": "bad replay batch",
        }
    )

    assert len(warnings) == 1
    assert "2 learner sample error" in warnings[0]
    assert "bad replay batch" in warnings[0]


def test_should_report_learner_sample_warnings_only_for_new_errors():
    """Waiting loop should report learner sample warnings when the count advances."""
    assert should_report_learner_sample_warnings({}, last_sample_error_count=0) is False
    assert (
        should_report_learner_sample_warnings(
            {"sample_error_count": 1},
            last_sample_error_count=0,
        )
        is True
    )
    assert (
        should_report_learner_sample_warnings(
            {"sample_error_count": 1},
            last_sample_error_count=1,
        )
        is False
    )


def test_log_learner_sample_health_writes_tensorboard_scalars():
    """Learner sample health should reach TensorBoard during waiting loops."""

    class FakeLogger:
        def __init__(self):
            self.scalars = []

        def log_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

    logger = FakeLogger()

    logged = log_learner_sample_health(
        logger,
        {"sample_error_count": 1, "last_sample_error": "bad replay batch"},
        step=42,
    )

    assert logged is True
    assert ("learner/sample_error_count", 1, 42) in logger.scalars
    assert ("learner/sample_warning_count", 1, 42) in logger.scalars


def test_log_learner_sample_health_skips_missing_errors():
    """No sample errors should not produce misleading zero-only TensorBoard rows."""

    class FakeLogger:
        def __init__(self):
            self.scalars = []

        def log_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

    logger = FakeLogger()

    assert log_learner_sample_health(None, {"sample_error_count": 1}, step=7) is False
    assert log_learner_sample_health(logger, {}, step=7) is False
    assert logger.scalars == []


def test_resolve_apex_min_buffer_uses_configured_warmup_when_feasible():
    """Normal configs keep the requested warmup below capacity."""
    assert (
        resolve_apex_min_buffer_size(
            batch_size=512,
            buffer_capacity=100_000,
            configured_min_buffer_size=50_000,
        )
        == 50_000
    )


def test_resolve_apex_min_buffer_never_drops_below_batch_size():
    """Warmup must not let learner sample before a full batch exists."""
    assert (
        resolve_apex_min_buffer_size(
            batch_size=512,
            buffer_capacity=10_000,
            configured_min_buffer_size=128,
        )
        == 512
    )


def test_resolve_apex_min_buffer_caps_to_small_capacity():
    """Small local buffers should still get a feasible warmup size."""
    assert (
        resolve_apex_min_buffer_size(
            batch_size=512,
            buffer_capacity=1_000,
            configured_min_buffer_size=50_000,
        )
        == 512
    )


def test_resolve_apex_min_buffer_rejects_buffer_smaller_than_batch():
    """A learner can never sample a batch larger than replay capacity."""
    with pytest.raises(ValueError, match="buffer_capacity"):
        resolve_apex_min_buffer_size(
            batch_size=512,
            buffer_capacity=128,
            configured_min_buffer_size=50_000,
        )


def valid_apex_config(**overrides):
    """Build a minimal valid distributed training config for validation tests."""
    config = {
        "num_actors": 4,
        "total_steps": 1000,
        "batch_size": 128,
        "buffer_capacity": 10_000,
        "n_step": 3,
        "min_buffer_size": 512,
        "weight_broadcast_interval": 400,
        "checkpoint_interval": 1000,
        "log_interval": 100,
        "stagger_delay": 0.5,
        "actor_env_num_snakes": 6,
        "actor_board_scale": 0.2,
        "actor_food_multiplier": 0.5,
        "actor_boost_exploration_rate": 0.25,
        "actor_danger_exploration_rate": 0.02,
    }
    config.update(overrides)
    return config


def test_validate_apex_training_config_accepts_valid_config():
    """Valid coordinator config should pass without touching Torch."""
    validate_apex_training_config(**valid_apex_config())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_actors", 0),
        ("total_steps", 0),
        ("batch_size", 0),
        ("buffer_capacity", 0),
        ("n_step", 0),
        ("min_buffer_size", 0),
        ("weight_broadcast_interval", 0),
        ("checkpoint_interval", 0),
        ("log_interval", 0),
        ("actor_env_num_snakes", 0),
    ],
)
def test_validate_apex_training_config_rejects_non_positive_values(field, value):
    """Coordinator should fail fast instead of starting impossible runs."""
    with pytest.raises(ValueError, match=field):
        validate_apex_training_config(**valid_apex_config(**{field: value}))


def test_validate_apex_training_config_rejects_negative_stagger_delay():
    """Actor startup delay must not be negative."""
    with pytest.raises(ValueError, match="stagger_delay"):
        validate_apex_training_config(**valid_apex_config(stagger_delay=-0.1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor_board_scale", 0.0),
        ("actor_board_scale", float("nan")),
        ("actor_food_multiplier", 0.0),
        ("actor_food_multiplier", float("inf")),
    ],
)
def test_validate_apex_training_config_rejects_invalid_actor_environment_shape(
    field,
    value,
):
    """Actor environment shaping values must be usable before processes spawn."""
    with pytest.raises(ValueError, match=field):
        validate_apex_training_config(**valid_apex_config(**{field: value}))


def test_validate_apex_training_config_rejects_actor_board_that_rounds_too_small():
    """Actor board scale should not create an arena that cannot spawn useful replay."""
    with pytest.raises(ValueError, match="actor_board_scale"):
        validate_apex_training_config(**valid_apex_config(actor_board_scale=0.01))


def test_validate_apex_training_config_rejects_actor_food_that_rounds_to_zero():
    """Actor food multiplier should not silently create foodless replay."""
    with pytest.raises(ValueError, match="actor_food_multiplier"):
        validate_apex_training_config(**valid_apex_config(actor_food_multiplier=0.001))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor_boost_exploration_rate", -0.1),
        ("actor_boost_exploration_rate", 1.1),
        ("actor_boost_exploration_rate", float("nan")),
        ("actor_danger_exploration_rate", -0.1),
        ("actor_danger_exploration_rate", 1.1),
        ("actor_danger_exploration_rate", float("inf")),
    ],
)
def test_validate_apex_training_config_rejects_invalid_actor_exploration_rates(
    field,
    value,
):
    """Actor exploration probabilities must stay bounded."""
    with pytest.raises(ValueError, match=field):
        validate_apex_training_config(**valid_apex_config(**{field: value}))


def test_validate_apex_training_config_rejects_capacity_below_batch():
    """A replay buffer smaller than the learner batch can never train."""
    with pytest.raises(ValueError, match="buffer_capacity"):
        validate_apex_training_config(
            **valid_apex_config(batch_size=512, buffer_capacity=128, min_buffer_size=512)
        )


def test_validate_apex_training_config_rejects_warmup_below_batch():
    """Warmup below batch size makes sampling fail after readiness."""
    with pytest.raises(ValueError, match="min_buffer_size"):
        validate_apex_training_config(**valid_apex_config(batch_size=512, min_buffer_size=128))


def test_validate_apex_training_config_rejects_warmup_above_capacity():
    """Warmup above capacity means learner can wait forever."""
    with pytest.raises(ValueError, match="min_buffer_size"):
        validate_apex_training_config(
            **valid_apex_config(buffer_capacity=1_000, min_buffer_size=2_000)
        )


def test_train_apex_invalid_config_fails_before_runtime_setup():
    """Invalid distributed config should fail before Torch/process startup."""
    with pytest.raises(ValueError, match="num_actors"):
        train_apex(num_actors=0, total_steps=100, batch_size=32, buffer_capacity=64)


def test_train_apex_invalid_actor_replay_gate_fails_before_runtime_setup():
    """Invalid replay-quality gates should fail before Torch/process startup."""
    with pytest.raises(ValueError, match="min_actor_terminal_fraction"):
        train_apex(
            num_actors=1,
            total_steps=100,
            batch_size=32,
            buffer_capacity=64,
            min_actor_terminal_fraction=1.1,
        )


def test_train_apex_missing_resume_checkpoint_fails_before_runtime_setup(tmp_path):
    """A typoed distributed resume path should not silently start from random weights."""
    missing_checkpoint = tmp_path / "missing_apex_resume.pth"

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        train_apex(
            num_actors=1,
            total_steps=1,
            batch_size=2,
            buffer_capacity=4,
            n_step=3,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            resume_checkpoint=str(missing_checkpoint),
            log_dir=str(tmp_path / "logs"),
            checkpoint_interval=10,
            log_interval=1,
            stagger_delay=0.0,
        )

    assert not (tmp_path / "checkpoints" / "apex_final.pth").exists()


def test_train_apex_incompatible_resume_checkpoint_fails_before_runtime_setup(tmp_path):
    """Distributed resume should fail rather than train with a mismatched TD horizon."""
    import torch

    checkpoint_path = tmp_path / "bad_n_step_resume.pth"
    torch.save(
        {
            "apex_config": {
                "input_size": 58,
                "hidden_size": 512,
                "output_size": 6,
                "gamma": 0.99,
                "n_step": 5,
                "reward_contract": current_reward_contract(),
                "reward_death": GameConfig.REWARD_DEATH,
                "reward_food_base": 3.0,
                "use_gru": False,
            },
            "dqn_state_dict": {},
            "target_dqn_state_dict": {},
            "optimizer_state_dict": {},
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="Failed to load resume checkpoint.*n_step"):
        train_apex(
            num_actors=1,
            total_steps=1,
            batch_size=2,
            buffer_capacity=4,
            n_step=3,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            resume_checkpoint=str(checkpoint_path),
            log_dir=str(tmp_path / "logs"),
            checkpoint_interval=10,
            log_interval=1,
            stagger_delay=0.0,
        )

    assert not (tmp_path / "checkpoints" / "apex_final.pth").exists()
