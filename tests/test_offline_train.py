"""Tests for offline generated-replay training helpers."""

import sys
import types

import pytest

from src.core.game_config import (
    ApexSettings,
    AppConfig,
    CheckpointSettings,
    GameConfig,
    TrainingSettings,
    get_config,
    initialize_config,
)
from src.core.reward_contract import current_reward_contract
from src.data.memory_db_handler import (
    MemoryDBHandler,
    resolve_min_row_count,
    resolve_replay_quality_fraction,
    validate_min_row_count,
    validate_replay_metadata_contract,
    validate_replay_quality_gates,
)
from src.scripts.offline_train import (
    DEFAULT_OFFLINE_REPLAY_QUALITY_PRESET,
    apply_offline_batch_size_override,
    format_checkpoint_replay_provenance,
    format_target_action_metrics,
    get_policy_min_replay_size,
    get_policy_target_action_metrics,
    load_checkpoint,
    load_replay_database,
    resolve_checkpoint_path,
    resolve_offline_batch_size,
    resolve_offline_replay_quality_gates,
    resolve_output_checkpoint_dir,
    resolve_replay_load_limit,
    save_policy_checkpoint,
    train_offline,
    validate_loaded_replay_rows,
    validate_offline_resume_checkpoint_config,
    validate_replay_warmup_size,
)


class TestCheckpointPathResolution:
    """Tests for resolving resume checkpoint paths."""

    def test_existing_direct_path_is_returned(self, tmp_path):
        checkpoint = tmp_path / "model.pth"
        checkpoint.write_bytes(b"checkpoint")

        assert resolve_checkpoint_path(str(checkpoint)) == checkpoint

    def test_configured_checkpoint_dir_fallback_is_used(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "best.pth"
        checkpoint.write_bytes(b"checkpoint")
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))

        try:
            initialize_config(config)

            assert resolve_checkpoint_path("best.pth") == checkpoint
        finally:
            initialize_config(original_config)

    def test_saved_snakes_fallback_is_used(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        checkpoint_dir = tmp_path / "saved_snakes"
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "best_apex.pth"
        checkpoint.write_bytes(b"checkpoint")

        assert resolve_checkpoint_path("best_apex.pth") == checkpoint

    def test_missing_checkpoint_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        assert resolve_checkpoint_path("missing.pth") is None

    def test_output_checkpoint_dir_defaults_to_configured_directory(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "configured_checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))

        try:
            initialize_config(config)

            assert resolve_output_checkpoint_dir(None) == str(checkpoint_dir)
        finally:
            initialize_config(original_config)

    def test_output_checkpoint_dir_uses_cli_override(self, tmp_path):
        override_dir = tmp_path / "cli_checkpoints"

        assert resolve_output_checkpoint_dir(str(override_dir)) == str(override_dir)


class TestReplayLoadLimit:
    """Tests for replay load limit resolution."""

    def test_default_uses_capacity(self):
        assert resolve_replay_load_limit(None, capacity=100) == 100

    def test_explicit_limit_is_clamped_to_capacity(self):
        assert resolve_replay_load_limit(250, capacity=100) == 100

    def test_explicit_limit_below_capacity_is_used(self):
        assert resolve_replay_load_limit(25, capacity=100) == 25

    def test_non_positive_limit_raises(self):
        with pytest.raises(ValueError, match="limit"):
            resolve_replay_load_limit(0, capacity=100)

    def test_non_positive_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            resolve_replay_load_limit(None, capacity=0)


class TestOfflineBatchSizeOverride:
    """Tests for optional offline-training batch-size overrides."""

    def test_missing_batch_size_defaults_to_apex_batch_size(self):
        original_config = get_config()
        config = AppConfig(
            training=TrainingSettings(batch_size=8, memory_size=1000),
            apex=ApexSettings(batch_size=32, min_buffer_size=32, buffer_size=1000),
        )

        try:
            initialize_config(config)

            assert resolve_offline_batch_size(None) == 32
            assert apply_offline_batch_size_override(None) == 32
            assert GameConfig.APEX_BATCH_SIZE == 32
            assert get_config().apex.batch_size == 32
            assert GameConfig.BATCH_SIZE == 8
            assert get_config().training.batch_size == 8
        finally:
            initialize_config(original_config)

    def test_batch_size_override_updates_apex_config(self):
        original_config = get_config()
        try:
            applied = apply_offline_batch_size_override(32)

            assert applied == 32
            assert GameConfig.APEX_BATCH_SIZE == 32
            assert get_config().apex.batch_size == 32
        finally:
            initialize_config(original_config)

    @pytest.mark.parametrize("batch_size", [0, -1, True])
    def test_invalid_batch_size_raises(self, batch_size):
        with pytest.raises(ValueError, match="batch-size"):
            resolve_offline_batch_size(batch_size)


class TestReplayWarmupSize:
    """Tests for offline replay warmup validation."""

    class PolicyWithoutWarmup:
        pass

    class PolicyWithWarmup:
        def __init__(self, min_replay_size=4):
            self.min_replay_size = min_replay_size

        def _min_replay_size(self):
            return self.min_replay_size

    def test_missing_policy_warmup_returns_none(self):
        assert get_policy_min_replay_size(self.PolicyWithoutWarmup()) is None

    def test_policy_warmup_is_returned(self):
        assert get_policy_min_replay_size(self.PolicyWithWarmup(7)) == 7

    def test_invalid_policy_warmup_raises(self):
        with pytest.raises(RuntimeError, match="warmup requirement"):
            get_policy_min_replay_size(self.PolicyWithWarmup(0))

    def test_replay_warmup_validation_raises_when_loaded_replay_is_too_small(self):
        with pytest.raises(RuntimeError, match="loaded 3 rows"):
            validate_replay_warmup_size(self.PolicyWithWarmup(4), loaded_rows=3)

    def test_replay_warmup_validation_returns_requirement(self):
        assert validate_replay_warmup_size(self.PolicyWithWarmup(4), loaded_rows=4) == 4


class TestReplayRowCountValidation:
    """Tests for absolute replay row-count validation."""

    def test_resolve_min_row_count_rejects_negative_values(self):
        assert resolve_min_row_count(None) == 0
        assert resolve_min_row_count(0) == 0
        assert resolve_min_row_count(128) == 128
        with pytest.raises(ValueError, match="min-row-count"):
            resolve_min_row_count(-1)

    def test_validate_min_row_count_raises_when_loaded_replay_is_too_small(self):
        with pytest.raises(RuntimeError, match="has 4 rows.*at least 8"):
            validate_min_row_count({"count": 4}, min_row_count=8, context="Loaded replay")


class TestOfflineTargetActionMetrics:
    """Tests for offline target-action diagnostic helpers."""

    class PolicyStub:
        def __init__(self, metrics=None):
            self._last_train_metrics = metrics or {}

    def test_get_policy_target_action_metrics_extracts_known_keys(self):
        policy = self.PolicyStub(
            {
                "valid_next_action_fraction": 0.75,
                "trapped_next_state_fraction": 0.25,
                "exact_next_action_mask_fraction": 0.5,
                "ignored": 1.0,
            }
        )

        assert get_policy_target_action_metrics(policy) == {
            "valid_next_action_fraction": 0.75,
            "trapped_next_state_fraction": 0.25,
            "exact_next_action_mask_fraction": 0.5,
        }

    def test_format_target_action_metrics_uses_percentages(self):
        report = format_target_action_metrics(
            {
                "valid_next_action_fraction": 0.75,
                "trapped_next_state_fraction": 0.25,
                "exact_next_action_mask_fraction": 0.5,
            }
        )

        assert report == "Target actions: valid=75.0%, trapped=25.0%, exact_masks=50.0%"

    def test_format_target_action_metrics_marks_missing_values(self):
        assert (
            format_target_action_metrics({})
            == "Target actions: valid=n/a, trapped=n/a, exact_masks=n/a"
        )


class TestCheckpointReplayProvenance:
    """Tests for offline checkpoint replay provenance summaries."""

    def _checkpoint(self):
        return {
            "source_db": "fallback.db",
            "apex_config": {
                "batch_size": 8,
                "buffer_size": 1000,
                "gamma": 0.99,
                "min_replay_size": 32,
                "n_step": 3,
                "priority_alpha": 0.6,
                "priority_beta_current": 0.4,
                "priority_beta_end": 1.0,
                "priority_beta_start": 0.4,
                "priority_epsilon": 1e-6,
                "reward_contract": current_reward_contract(),
                "reward_death": GameConfig.REWARD_DEATH,
                "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                "target_update_freq": 2500,
            },
            "offline_replay_quality": {
                "count": 42,
                "active_action_count": 6,
                "dominant_action": 0,
                "dominant_action_fraction": 0.25,
                "nonterminal_mask_fraction": 0.9,
                "terminal_fraction": 0.05,
            },
            "offline_replay_gates": {
                "min_terminal_fraction": 0.005,
                "min_immediate_terminal_fraction": 0.001,
                "min_exact_mask_fraction": 0.8,
                "min_boost_mask_fraction": 0.05,
                "min_action_coverage_fraction": 1.0,
                "min_positive_reward_fraction": 0.1,
                "min_negative_reward_fraction": 0.2,
                "min_multistep_fraction": 0.4,
                "max_dominant_action_fraction": 0.75,
                "max_invalid_current_action_fraction": 0.0,
                "max_nonterminal_trapped_next_fraction": 0.0,
                "max_exact_mask_state_mismatch_fraction": 0.0,
                "max_malformed_state_feature_fraction": 0.0,
            },
            "offline_replay_load": {
                "db_path": "snake_memories.db",
                "loaded_rows": 42,
                "min_replay_size": 32,
                "batch_size": 8,
                "replay_order": "id_uniform",
            },
            "offline_replay_metadata": {
                "generation.mode": "single",
                "generation.episodes": 12,
                "generation.frame_limit": 500,
                "generation.num_snakes": 4,
                "generation.board_width": 725,
                "generation.board_height": 415,
                "generation.state_size": 58,
                "generation.action_size": 6,
                "generation.gamma": 0.99,
                "generation.apex_n_step": 3,
            },
            "offline_replay_warnings": [
                "   Action 0 accounts for 80.0% of replay rows; "
                "policy updates may overfit one behavior",
                "   Replay priorities are flat at 1.000000",
            ],
            "offline_train_metrics": {
                "valid_next_action_fraction": 0.75,
                "trapped_next_state_fraction": 0.25,
                "exact_next_action_mask_fraction": 0.5,
            },
        }

    def test_format_checkpoint_replay_provenance_returns_compact_lines(self):
        lines = format_checkpoint_replay_provenance(self._checkpoint())

        assert lines == [
            (
                "Checkpoint replay: rows=42 | warmup=32 | batch=8 | order=id_uniform | "
                "source=snake_memories.db"
            ),
            (
                "Checkpoint generated replay: mode=single | episodes=12 | frame_limit=500 | "
                "snakes=4 | board=725x415 | state=58 | actions=6 | gamma=0.99 | n_step=3"
            ),
            (
                "Checkpoint Apex config: batch=8 | buffer=1000 | warmup=32 | n_step=3 | "
                "gamma=0.99 | target_sync=2500 | reward death=-11.0, food=3.0 | "
                "PER alpha=0.6, beta=0.4->1.0 (current=0.4), eps=1e-06"
            ),
            (
                "Checkpoint replay quality: actions=6/6 | dominant=0 (25.0%) | "
                "exact_masks=90.0% | terminal=5.0%"
            ),
            (
                "Checkpoint replay gates: terminal>=0.5%, immediate_terminal>=0.1%, "
                "exact_masks>=80.0%, "
                "boost_masks>=5.0%, action_coverage>=100.0%, positive_reward>=10.0%, "
                "negative_reward>=20.0%, "
                "multistep>=40.0%, dominant<=75.0%, "
                "invalid_current_action<=0.0%, trapped_next<=0.0%, "
                "mask_state_mismatch<=0.0%, malformed_state<=0.0%"
            ),
            "Checkpoint replay warnings:",
            (
                "  - Action 0 accounts for 80.0% of replay rows; "
                "policy updates may overfit one behavior"
            ),
            "  - Replay priorities are flat at 1.000000",
            "Checkpoint Target actions: valid=75.0%, trapped=25.0%, exact_masks=50.0%",
        ]

    def test_format_checkpoint_replay_provenance_ignores_plain_checkpoints(self):
        assert format_checkpoint_replay_provenance({"iteration": 3}) == []

    def test_load_checkpoint_prints_replay_provenance(self, tmp_path, capsys):
        import torch

        class PolicyStub:
            device = "cpu"

            def __init__(self):
                self.loaded_checkpoint = None

            def load_state_dict(self, checkpoint):
                self.loaded_checkpoint = checkpoint

        checkpoint_path = tmp_path / "offline.pth"
        torch.save(self._checkpoint(), checkpoint_path)
        policy = PolicyStub()

        assert load_checkpoint(policy, str(checkpoint_path)) is True
        output = capsys.readouterr().out

        assert policy.loaded_checkpoint is not None
        assert f"Loaded checkpoint: {checkpoint_path}" in output
        assert "Checkpoint replay: rows=42 | warmup=32 | batch=8 | order=id_uniform" in output
        assert "Checkpoint generated replay: mode=single | episodes=12" in output
        assert "Checkpoint replay warnings:" in output
        assert "Replay priorities are flat at 1.000000" in output
        assert "Checkpoint Target actions: valid=75.0%, trapped=25.0%" in output

    def test_validate_offline_resume_checkpoint_config_accepts_matching_contract(self):
        class PolicyStub:
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

        validate_offline_resume_checkpoint_config(
            self._checkpoint(),
            PolicyStub(),
            checkpoint_path="offline.pth",
        )

    def test_validate_offline_resume_checkpoint_config_rejects_missing_reward_contract(self):
        class PolicyStub:
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

        checkpoint = self._checkpoint()
        checkpoint["apex_config"].pop("reward_contract")

        with pytest.raises(RuntimeError, match="missing required reward_contract"):
            validate_offline_resume_checkpoint_config(
                checkpoint,
                PolicyStub(),
                checkpoint_path="legacy_offline.pth",
            )

    def test_validate_offline_resume_checkpoint_config_rejects_reward_mismatch(self):
        class PolicyStub:
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

        checkpoint = self._checkpoint()
        checkpoint["apex_config"]["reward_death"] = -3.0

        with pytest.raises(RuntimeError, match="reward_death=-3"):
            validate_offline_resume_checkpoint_config(
                checkpoint,
                PolicyStub(),
                checkpoint_path="stale_offline.pth",
            )

    def test_validate_offline_resume_checkpoint_config_rejects_full_reward_mismatch(self):
        class PolicyStub:
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

        checkpoint = self._checkpoint()
        stale_contract = dict(checkpoint["apex_config"]["reward_contract"])
        stale_contract["survival"] = float(stale_contract["survival"]) + 1.0
        checkpoint["apex_config"]["reward_contract"] = stale_contract

        with pytest.raises(RuntimeError, match="reward_contract.survival"):
            validate_offline_resume_checkpoint_config(
                checkpoint,
                PolicyStub(),
                checkpoint_path="stale_offline.pth",
            )

    def test_validate_offline_resume_checkpoint_config_rejects_n_step_mismatch(self):
        class PolicyStub:
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

        checkpoint = self._checkpoint()
        checkpoint["apex_config"]["n_step"] = GameConfig.APEX_N_STEP + 1

        with pytest.raises(RuntimeError, match="n_step"):
            validate_offline_resume_checkpoint_config(
                checkpoint,
                PolicyStub(),
                checkpoint_path="offline.pth",
            )

    def test_load_checkpoint_gamma_mismatch_raises_before_policy_load(self, tmp_path):
        import torch

        class PolicyStub:
            device = "cpu"
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

            def __init__(self):
                self.loaded_checkpoint = None

            def load_state_dict(self, checkpoint):
                self.loaded_checkpoint = checkpoint

        checkpoint = self._checkpoint()
        checkpoint["apex_config"]["gamma"] = GameConfig.APEX_GAMMA - 0.01
        checkpoint_path = tmp_path / "offline.pth"
        torch.save(checkpoint, checkpoint_path)
        policy = PolicyStub()

        with pytest.raises(RuntimeError, match="gamma"):
            load_checkpoint(policy, str(checkpoint_path))

        assert policy.loaded_checkpoint is None


class TestReplayQualityGates:
    """Tests for optional offline replay quality gates."""

    def test_default_quality_fraction_is_disabled(self):
        assert resolve_replay_quality_fraction(None, "min_terminal_fraction") == pytest.approx(0.0)

    def test_explicit_quality_fraction_is_preserved(self):
        assert resolve_replay_quality_fraction(0.25, "min_terminal_fraction") == pytest.approx(0.25)

    def test_replay_quality_preset_resolves_with_overrides(self):
        gates = resolve_offline_replay_quality_gates(
            preset="training",
            overrides={
                "min_exact_mask_fraction": 0.9,
                "min_positive_reward_fraction": 0.2,
            },
        )

        assert gates["min_terminal_fraction"] == pytest.approx(0.005)
        assert gates["min_immediate_terminal_fraction"] == pytest.approx(0.001)
        assert gates["min_exact_mask_fraction"] == pytest.approx(0.9)
        assert gates["min_boost_mask_fraction"] == pytest.approx(0.05)
        assert gates["min_action_coverage_fraction"] == pytest.approx(1.0)
        assert gates["min_positive_reward_fraction"] == pytest.approx(0.2)
        assert gates["min_negative_reward_fraction"] == pytest.approx(0.005)
        assert gates["min_multistep_fraction"] == pytest.approx(0.5)
        assert gates["max_dominant_action_fraction"] == pytest.approx(0.75)
        assert gates["max_invalid_current_action_fraction"] == pytest.approx(0.0)
        assert gates["max_exact_mask_state_mismatch_fraction"] == pytest.approx(0.0)
        assert gates["max_malformed_state_feature_fraction"] == pytest.approx(0.0)

    def test_default_replay_quality_preset_requires_training_signal(self):
        gates = resolve_offline_replay_quality_gates()

        assert DEFAULT_OFFLINE_REPLAY_QUALITY_PRESET == "training"
        assert gates["min_terminal_fraction"] == pytest.approx(0.005)
        assert gates["min_immediate_terminal_fraction"] == pytest.approx(0.001)
        assert gates["min_exact_mask_fraction"] == pytest.approx(0.8)
        assert gates["min_boost_mask_fraction"] == pytest.approx(0.05)
        assert gates["min_action_coverage_fraction"] == pytest.approx(1.0)
        assert gates["min_positive_reward_fraction"] == pytest.approx(0.005)
        assert gates["min_negative_reward_fraction"] == pytest.approx(0.005)
        assert gates["min_multistep_fraction"] == pytest.approx(0.5)
        assert gates["max_dominant_action_fraction"] == pytest.approx(0.75)
        assert gates["max_invalid_current_action_fraction"] == pytest.approx(0.0)
        assert gates["max_nonterminal_trapped_next_fraction"] == pytest.approx(0.05)
        assert gates["max_exact_mask_state_mismatch_fraction"] == pytest.approx(0.0)
        assert gates["max_malformed_state_feature_fraction"] == pytest.approx(0.0)

    def test_explicit_none_replay_quality_preset_keeps_diagnostic_escape_hatch(self):
        gates = resolve_offline_replay_quality_gates(preset="none")

        assert gates["min_terminal_fraction"] == pytest.approx(0.0)
        assert gates["min_immediate_terminal_fraction"] == pytest.approx(0.0)
        assert gates["min_exact_mask_fraction"] == pytest.approx(0.0)
        assert gates["min_boost_mask_fraction"] == pytest.approx(0.0)
        assert gates["max_dominant_action_fraction"] == pytest.approx(1.0)
        assert gates["max_exact_mask_state_mismatch_fraction"] == pytest.approx(1.0)
        assert gates["max_malformed_state_feature_fraction"] == pytest.approx(1.0)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
    def test_invalid_quality_fraction_raises(self, fraction):
        with pytest.raises(ValueError, match="min_exact_mask_fraction"):
            resolve_replay_quality_fraction(fraction, "min_exact_mask_fraction")

    def test_quality_gates_pass_when_disabled(self):
        stats = {
            "count": 1000,
            "done_count": 0,
            "terminal_fraction": 0.0,
            "nonterminal_count": 1000,
            "nonterminal_mask_count": 0,
            "nonterminal_mask_fraction": 0.0,
        }

        validate_replay_quality_gates(stats)

    def test_terminal_quality_gate_raises_when_replay_is_too_sparse(self):
        stats = {
            "count": 1000,
            "done_count": 2,
            "terminal_fraction": 0.002,
            "nonterminal_count": 998,
            "nonterminal_mask_count": 998,
            "nonterminal_mask_fraction": 1.0,
        }

        with pytest.raises(RuntimeError, match="terminal fraction"):
            validate_replay_quality_gates(stats, min_terminal_fraction=0.005)

    def test_exact_mask_quality_gate_raises_when_masks_are_too_sparse(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 100,
            "nonterminal_mask_fraction": 100 / 990,
        }

        with pytest.raises(RuntimeError, match="exact-mask fraction"):
            validate_replay_quality_gates(stats, min_exact_mask_fraction=0.5)

    def test_action_coverage_quality_gate_raises_when_actions_are_too_sparse(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "action_counts": {0: 500, 1: 500},
        }

        with pytest.raises(RuntimeError, match="action coverage"):
            validate_replay_quality_gates(stats, min_action_coverage_fraction=0.5)

    def test_boost_mask_quality_gate_raises_when_targets_are_too_sparse(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "boost_mask_count": 10,
            "boost_mask_fraction": 10 / 990,
        }

        with pytest.raises(RuntimeError, match="boost-mask fraction"):
            validate_replay_quality_gates(stats, min_boost_mask_fraction=0.05)

    def test_positive_reward_quality_gate_raises_when_success_signal_is_sparse(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "reward_positive_count": 10,
        }

        with pytest.raises(RuntimeError, match="positive-reward fraction"):
            validate_replay_quality_gates(stats, min_positive_reward_fraction=0.05)

    def test_negative_reward_quality_gate_raises_when_danger_signal_is_sparse(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "reward_negative_count": 10,
        }

        with pytest.raises(RuntimeError, match="negative-reward fraction"):
            validate_replay_quality_gates(stats, min_negative_reward_fraction=0.05)

    def test_dominant_action_quality_gate_raises_when_one_action_dominates(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "action_counts": {0: 850, 1: 50, 2: 50, 3: 25, 4: 15, 5: 10},
        }

        with pytest.raises(RuntimeError, match="dominant-action fraction"):
            validate_replay_quality_gates(stats, max_dominant_action_fraction=0.8)

    def test_trapped_next_quality_gate_raises_when_dead_ends_dominate(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "action_counts": {0: 334, 1: 333, 2: 333},
            "nonterminal_trapped_next_state_count": 100,
            "nonterminal_trapped_next_state_fraction": 100 / 990,
        }

        with pytest.raises(RuntimeError, match="trapped-next-state fraction"):
            validate_replay_quality_gates(
                stats,
                max_nonterminal_trapped_next_fraction=0.05,
            )


class TestLoadedReplayValidation:
    """Tests for generated replay rows before adding them to policy memory."""

    def _valid_rows(self):
        state = [0.0] * GameConfig.INPUT_SIZE
        next_state = [1.0] * GameConfig.INPUT_SIZE
        return {
            "states": [state],
            "actions": [0],
            "rewards": [1.0],
            "next_states": [next_state],
            "dones": [False],
            "priorities": [1.0],
            "bootstrap_steps": [1],
            "db_path": "replay.db",
        }

    def test_valid_rows_pass(self):
        validate_loaded_replay_rows(**self._valid_rows())

    def test_valid_rows_with_action_mask_pass(self):
        rows = self._valid_rows()
        rows["next_action_masks"] = [[False, True, False, False, False, False]]

        validate_loaded_replay_rows(**rows)

    def test_valid_rows_with_empty_exact_action_mask_pass(self):
        rows = self._valid_rows()
        rows["next_action_masks"] = [[False, False, False, False, False, False]]

        validate_loaded_replay_rows(**rows)

    def test_valid_rows_with_snake_id_pass(self):
        rows = self._valid_rows()
        rows["snake_ids"] = [3]

        validate_loaded_replay_rows(**rows)

    def test_empty_rows_raise(self):
        rows = self._valid_rows()
        for key in (
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
            "priorities",
            "bootstrap_steps",
        ):
            rows[key] = []

        with pytest.raises(RuntimeError, match="No Apex replay rows"):
            validate_loaded_replay_rows(**rows)

    def test_misaligned_fields_raise(self):
        rows = self._valid_rows()
        rows["actions"] = []

        with pytest.raises(RuntimeError, match="misaligned"):
            validate_loaded_replay_rows(**rows)

    def test_misaligned_action_masks_raise(self):
        rows = self._valid_rows()
        rows["next_action_masks"] = []

        with pytest.raises(RuntimeError, match="misaligned"):
            validate_loaded_replay_rows(**rows)

    def test_misaligned_snake_ids_raise(self):
        rows = self._valid_rows()
        rows["snake_ids"] = []

        with pytest.raises(RuntimeError, match="misaligned"):
            validate_loaded_replay_rows(**rows)

    def test_wrong_state_size_raises(self):
        rows = self._valid_rows()
        rows["states"] = [[0.0] * (GameConfig.INPUT_SIZE - 1)]

        with pytest.raises(RuntimeError, match="state size"):
            validate_loaded_replay_rows(**rows)

    def test_invalid_action_raises(self):
        rows = self._valid_rows()
        rows["actions"] = [GameConfig.OUTPUT_SIZE]

        with pytest.raises(RuntimeError, match="outside"):
            validate_loaded_replay_rows(**rows)

    @pytest.mark.parametrize("action", [1.5, True, "1"])
    def test_non_integral_action_raises(self, action):
        rows = self._valid_rows()
        rows["actions"] = [action]

        with pytest.raises(RuntimeError, match="action.*integer"):
            validate_loaded_replay_rows(**rows)

    @pytest.mark.parametrize(
        ("field", "replacement"),
        [
            ("states", [[float("nan")] * GameConfig.INPUT_SIZE]),
            ("next_states", [[float("inf")] * GameConfig.INPUT_SIZE]),
        ],
    )
    def test_non_finite_state_values_raise(self, field, replacement):
        rows = self._valid_rows()
        rows[field] = replacement

        with pytest.raises(RuntimeError, match="must be finite"):
            validate_loaded_replay_rows(**rows)

    @pytest.mark.parametrize("reward", [float("nan"), float("inf")])
    def test_non_finite_reward_raises(self, reward):
        rows = self._valid_rows()
        rows["rewards"] = [reward]

        with pytest.raises(RuntimeError, match="reward.*finite"):
            validate_loaded_replay_rows(**rows)

    @pytest.mark.parametrize("done", [2, -1, 0.5, "False", float("nan")])
    def test_invalid_done_raises(self, done):
        rows = self._valid_rows()
        rows["dones"] = [done]

        with pytest.raises(RuntimeError, match="done"):
            validate_loaded_replay_rows(**rows)

    @pytest.mark.parametrize("priority", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_priority_raises(self, priority):
        rows = self._valid_rows()
        rows["priorities"] = [priority]

        with pytest.raises(RuntimeError, match="priority"):
            validate_loaded_replay_rows(**rows)

    def test_invalid_bootstrap_steps_raises(self):
        rows = self._valid_rows()
        rows["bootstrap_steps"] = [0]

        with pytest.raises(RuntimeError, match="bootstrap_steps"):
            validate_loaded_replay_rows(**rows)

    @pytest.mark.parametrize("bootstrap_steps", [1.5, True, "2"])
    def test_non_integral_bootstrap_steps_raise(self, bootstrap_steps):
        rows = self._valid_rows()
        rows["bootstrap_steps"] = [bootstrap_steps]

        with pytest.raises(RuntimeError, match="bootstrap_steps.*integer"):
            validate_loaded_replay_rows(**rows)

    @pytest.mark.parametrize(
        "next_action_mask",
        [
            [False] * (GameConfig.OUTPUT_SIZE - 1),
            [False, True, False, False, False, "yes"],
        ],
    )
    def test_invalid_action_mask_raises(self, next_action_mask):
        rows = self._valid_rows()
        rows["next_action_masks"] = [next_action_mask]

        with pytest.raises(RuntimeError, match="next_action_mask"):
            validate_loaded_replay_rows(**rows)

    @pytest.mark.parametrize("snake_id", [-1, 1.5, True, "1"])
    def test_invalid_snake_id_raises(self, snake_id):
        rows = self._valid_rows()
        rows["snake_ids"] = [snake_id]

        with pytest.raises(RuntimeError, match="snake_id"):
            validate_loaded_replay_rows(**rows)


class TestReplayMetadataContract:
    """Tests for generated replay metadata compatibility."""

    def test_empty_metadata_passes_for_legacy_replay(self):
        validate_replay_metadata_contract({}, "replay.db")

    def test_current_generation_contract_passes(self):
        validate_replay_metadata_contract(
            {
                "generation.policy_type": "apex",
                "generation.state_size": GameConfig.INPUT_SIZE,
                "generation.action_size": GameConfig.OUTPUT_SIZE,
                "generation.gamma": GameConfig.APEX_GAMMA,
                "generation.apex_n_step": GameConfig.APEX_N_STEP,
                "generation.reward_contract": current_reward_contract(),
                "generation.reward_death": GameConfig.REWARD_DEATH,
                "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
            },
            "replay.db",
            expected_gamma=GameConfig.APEX_GAMMA,
            expected_n_step=GameConfig.APEX_N_STEP,
            expected_reward_contract=current_reward_contract(),
            expected_reward_death=GameConfig.REWARD_DEATH,
            expected_reward_food_base=GameConfig.REWARD_FOOD_BASE,
        )

    def test_state_size_mismatch_raises(self):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE - 1,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
        }

        with pytest.raises(RuntimeError, match="generation.state_size=.*STATE_SIZE"):
            validate_replay_metadata_contract(metadata, "replay.db")

    def test_action_size_mismatch_raises(self):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE - 1,
        }

        with pytest.raises(RuntimeError, match="generation.action_size=.*ACTION_SIZE"):
            validate_replay_metadata_contract(metadata, "replay.db")

    def test_policy_type_mismatch_raises(self):
        metadata = {
            "generation.policy_type": "legacy",
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
        }

        with pytest.raises(RuntimeError, match="policy_type"):
            validate_replay_metadata_contract(metadata, "replay.db")

    @pytest.mark.parametrize("value", [True, "58", 58.5])
    def test_malformed_state_size_raises(self, value):
        metadata = {
            "generation.state_size": value,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
        }

        with pytest.raises(RuntimeError, match="generation.state_size.*integer"):
            validate_replay_metadata_contract(metadata, "replay.db")

    def test_gamma_mismatch_raises_when_expected_gamma_is_supplied(self):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.gamma": 0.95,
        }

        with pytest.raises(RuntimeError, match="generation.gamma=.*APEX_GAMMA"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_gamma=GameConfig.APEX_GAMMA,
                gamma_name="APEX_GAMMA",
            )

    @pytest.mark.parametrize("value", [True, "0.99", float("nan")])
    def test_malformed_gamma_raises_when_expected_gamma_is_supplied(self, value):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.gamma": value,
        }

        with pytest.raises(RuntimeError, match="generation.gamma.*finite"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_gamma=GameConfig.APEX_GAMMA,
            )

    def test_n_step_mismatch_raises_when_expected_n_step_is_supplied(self):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.apex_n_step": GameConfig.APEX_N_STEP + 1,
        }

        with pytest.raises(RuntimeError, match="generation.apex_n_step=.*APEX_N_STEP"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_n_step=GameConfig.APEX_N_STEP,
                n_step_name="APEX_N_STEP",
            )

    def test_reward_metadata_missing_raises_when_expected_reward_is_supplied(self):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.gamma": GameConfig.APEX_GAMMA,
            "generation.apex_n_step": GameConfig.APEX_N_STEP,
        }

        with pytest.raises(RuntimeError, match="missing required generation.reward_death"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_reward_death=GameConfig.REWARD_DEATH,
            )

    def test_full_reward_contract_missing_raises_when_expected_contract_is_supplied(self):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.gamma": GameConfig.APEX_GAMMA,
            "generation.apex_n_step": GameConfig.APEX_N_STEP,
            "generation.reward_death": GameConfig.REWARD_DEATH,
            "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
        }

        with pytest.raises(RuntimeError, match="missing required generation.reward_contract"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_reward_contract=current_reward_contract(),
            )

    def test_full_reward_contract_mismatch_raises_when_shaping_reward_changed(self):
        contract = current_reward_contract()
        stale_contract = dict(contract)
        stale_contract["survival"] = contract["survival"] + 0.25
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.reward_contract": stale_contract,
            "generation.reward_death": GameConfig.REWARD_DEATH,
            "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
        }

        with pytest.raises(RuntimeError, match="generation.reward_contract.survival"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_reward_contract=contract,
            )

    @pytest.mark.parametrize("value", [True, "0.01", float("nan")])
    def test_full_reward_contract_malformed_value_raises(self, value):
        contract = current_reward_contract()
        malformed_contract = dict(contract)
        malformed_contract["survival"] = value
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.reward_contract": malformed_contract,
        }

        with pytest.raises(RuntimeError, match="generation.reward_contract.survival.*finite"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_reward_contract=contract,
            )

    def test_reward_metadata_mismatch_raises_when_expected_reward_is_supplied(self):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.reward_death": GameConfig.REWARD_DEATH + 1.0,
            "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
        }

        with pytest.raises(RuntimeError, match="generation.reward_death=.*REWARD_DEATH"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_reward_death=GameConfig.REWARD_DEATH,
                expected_reward_food_base=GameConfig.REWARD_FOOD_BASE,
                reward_death_name="REWARD_DEATH",
                reward_food_base_name="REWARD_FOOD_BASE",
            )

    @pytest.mark.parametrize("value", [True, "3.0", float("nan")])
    def test_malformed_reward_metadata_raises_when_expected_reward_is_supplied(self, value):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.reward_death": GameConfig.REWARD_DEATH,
            "generation.reward_food_base": value,
        }

        with pytest.raises(RuntimeError, match="generation.reward_food_base.*finite"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_reward_death=GameConfig.REWARD_DEATH,
                expected_reward_food_base=GameConfig.REWARD_FOOD_BASE,
            )

    @pytest.mark.parametrize("value", [True, "3", 3.5])
    def test_malformed_n_step_raises_when_expected_n_step_is_supplied(self, value):
        metadata = {
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.apex_n_step": value,
        }

        with pytest.raises(RuntimeError, match="generation.apex_n_step.*integer"):
            validate_replay_metadata_contract(
                metadata,
                "replay.db",
                expected_n_step=GameConfig.APEX_N_STEP,
            )


class TestLoadReplayDatabase:
    """Tests for loading SQLite replay into local policy memory."""

    class MemoryStub:
        capacity = 10

        def __init__(self):
            self.cleared = False
            self.add_bulk_kwargs = None
            self.existing_rows = ["keep-me"]

        def clear(self):
            self.cleared = True

        def add_bulk(self, *args, **kwargs):
            self.add_bulk_kwargs = kwargs

    class PolicyStub:
        def __init__(self):
            self.memory = TestLoadReplayDatabase.MemoryStub()

    class WarmupPolicyStub(PolicyStub):
        def _min_replay_size(self):
            return 2

    def test_load_replay_database_passes_next_action_masks_to_memory(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        mask = [False, True, False, False, False, False]
        try:
            handler.save_memories(
                snake_id=6,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 1.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 2,
                        "next_action_mask": mask,
                    }
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        loaded = load_replay_database(policy, str(db_path), limit=None)

        assert loaded == 1
        assert policy.memory.cleared is True
        assert policy.memory.add_bulk_kwargs["next_action_masks"] == [tuple(mask)]
        assert policy.memory.add_bulk_kwargs["stream_ids"] == [6]

    def test_load_replay_database_fails_before_clearing_when_below_warmup(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 1.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                ],
            )
        finally:
            handler.close()
        policy = self.WarmupPolicyStub()

        with pytest.raises(RuntimeError, match="loaded 1 rows.*needs at least 2"):
            load_replay_database(policy, str(db_path), limit=None)

        assert policy.memory.cleared is False
        assert policy.memory.add_bulk_kwargs is None
        assert policy.memory.existing_rows == ["keep-me"]

    def test_load_replay_database_fails_before_clearing_when_below_row_gate(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 1.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        with pytest.raises(RuntimeError, match="has 1 rows.*at least 2"):
            load_replay_database(policy, str(db_path), limit=None, min_row_count=2)

        assert policy.memory.cleared is False
        assert policy.memory.add_bulk_kwargs is None
        assert policy.memory.existing_rows == ["keep-me"]

    def test_load_replay_database_metadata_mismatch_does_not_mutate_memory(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.update_metadata(
                {
                    "generation.state_size": GameConfig.INPUT_SIZE - 1,
                    "generation.action_size": GameConfig.OUTPUT_SIZE,
                }
            )
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 1.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        with pytest.raises(RuntimeError, match="generation.state_size"):
            load_replay_database(policy, str(db_path), limit=None)

        assert policy.memory.cleared is False
        assert policy.memory.add_bulk_kwargs is None
        assert policy.memory.existing_rows == ["keep-me"]

    def test_load_replay_database_gamma_mismatch_does_not_mutate_memory(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.update_metadata(
                {
                    "generation.state_size": GameConfig.INPUT_SIZE,
                    "generation.action_size": GameConfig.OUTPUT_SIZE,
                    "generation.gamma": GameConfig.APEX_GAMMA - 0.01,
                }
            )
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 1.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        with pytest.raises(RuntimeError, match="generation.gamma"):
            load_replay_database(policy, str(db_path), limit=None)

        assert policy.memory.cleared is False
        assert policy.memory.add_bulk_kwargs is None
        assert policy.memory.existing_rows == ["keep-me"]

    def test_load_replay_database_n_step_mismatch_does_not_mutate_memory(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.update_metadata(
                {
                    "generation.state_size": GameConfig.INPUT_SIZE,
                    "generation.action_size": GameConfig.OUTPUT_SIZE,
                    "generation.gamma": GameConfig.APEX_GAMMA,
                    "generation.apex_n_step": GameConfig.APEX_N_STEP + 1,
                }
            )
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 1.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        with pytest.raises(RuntimeError, match="generation.apex_n_step"):
            load_replay_database(policy, str(db_path), limit=None)

        assert policy.memory.cleared is False
        assert policy.memory.add_bulk_kwargs is None
        assert policy.memory.existing_rows == ["keep-me"]

    def test_load_replay_database_missing_reward_metadata_does_not_mutate_memory(
        self,
        tmp_path,
    ):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.update_metadata(
                {
                    "generation.state_size": GameConfig.INPUT_SIZE,
                    "generation.action_size": GameConfig.OUTPUT_SIZE,
                    "generation.gamma": GameConfig.APEX_GAMMA,
                    "generation.apex_n_step": GameConfig.APEX_N_STEP,
                }
            )
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 1.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        with pytest.raises(RuntimeError, match="missing required generation.reward_contract"):
            load_replay_database(policy, str(db_path), limit=None)

        assert policy.memory.cleared is False
        assert policy.memory.add_bulk_kwargs is None
        assert policy.memory.existing_rows == ["keep-me"]

    def test_load_replay_database_passes_empty_exact_next_action_masks_to_memory(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        mask = [False, False, False, False, False, False]
        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 1.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 2,
                        "next_action_mask": mask,
                    }
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        loaded = load_replay_database(policy, str(db_path), limit=None)

        assert loaded == 1
        assert policy.memory.cleared is True
        assert policy.memory.add_bulk_kwargs["next_action_masks"] == [tuple(mask)]

    def test_load_replay_database_reports_loaded_subset_quality(self, tmp_path, capsys):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [float(i)] * GameConfig.INPUT_SIZE,
                        "action": i,
                        "reward": float(i),
                        "next_state": [float(i + 1)] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 2,
                        "next_action_mask": [idx == i for idx in range(GameConfig.OUTPUT_SIZE)],
                    }
                    for i in range(3)
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        loaded = load_replay_database(policy, str(db_path), limit=2, replay_order="id")
        output = capsys.readouterr().out

        assert loaded == 2
        assert "Rows: 2" in output
        assert "Nonterminal exact masks: 2/2 (100.0%)" in output
        assert "Actions: 0:1, 1:1, 2:0" in output
        assert "Reward signs neg/zero/pos: 0/1/1" in output
        assert "Boost available states: 1 (50.0%)" in output
        assert "Exact masks allowing boost: 0 (0.0%)" in output
        assert "Rows per snake_id min/avg/max: 2/2.00/2" in output
        assert "Replay quality warnings:" in output
        assert "normal action(s) 2" in output
        assert any("normal action(s) 2" in warning for warning in policy._offline_replay_warnings)

    def test_load_replay_database_records_replay_provenance_on_policy(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.update_metadata(
                {
                    "generation.mode": "parallel",
                    "generation.episodes": 6,
                    "generation.state_size": GameConfig.INPUT_SIZE,
                    "generation.action_size": GameConfig.OUTPUT_SIZE,
                    "generation.reward_contract": current_reward_contract(),
                    "generation.reward_death": GameConfig.REWARD_DEATH,
                    "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
                }
            )
            handler.save_memories(
                snake_id=3,
                memories=[
                    {
                        "state": [float(idx)] * GameConfig.INPUT_SIZE,
                        "action": idx % GameConfig.OUTPUT_SIZE,
                        "reward": -1.0 if idx == 5 else float(idx + 1),
                        "next_state": [float(idx + 1)] * GameConfig.INPUT_SIZE,
                        "done": idx == 5,
                        "priority": 1.0 + idx,
                        "bootstrap_steps": 2 if idx < 3 else 1,
                        "next_action_mask": (
                            [mask_idx == idx for mask_idx in range(GameConfig.OUTPUT_SIZE)]
                            if idx < 5
                            else None
                        ),
                    }
                    for idx in range(6)
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        loaded = load_replay_database(
            policy,
            str(db_path),
            limit=None,
            replay_order="id",
            min_terminal_fraction=1 / 6,
            min_immediate_terminal_fraction=1 / 6,
            min_exact_mask_fraction=1.0,
            min_boost_mask_fraction=0.0,
            min_action_coverage_fraction=1.0,
            min_positive_reward_fraction=5 / 6,
            min_negative_reward_fraction=1 / 6,
            min_multistep_fraction=0.5,
            max_dominant_action_fraction=0.5,
            max_invalid_current_action_fraction=1.0,
            max_nonterminal_trapped_next_fraction=1.0,
            max_exact_mask_state_mismatch_fraction=1.0,
            max_malformed_state_feature_fraction=1.0,
            min_row_count=6,
        )

        assert loaded == 6
        assert policy._offline_replay_quality["count"] == 6
        assert policy._offline_replay_quality["active_action_count"] == 6
        assert policy._offline_replay_quality["dominant_action_fraction"] == pytest.approx(1 / 6)
        assert policy._offline_replay_gates == {
            "min_terminal_fraction": pytest.approx(1 / 6),
            "min_immediate_terminal_fraction": pytest.approx(1 / 6),
            "min_exact_mask_fraction": 1.0,
            "min_boost_mask_fraction": 0.0,
            "min_action_coverage_fraction": 1.0,
            "min_positive_reward_fraction": pytest.approx(5 / 6),
            "min_negative_reward_fraction": pytest.approx(1 / 6),
            "min_multistep_fraction": 0.5,
            "max_dominant_action_fraction": 0.5,
            "max_invalid_current_action_fraction": 1.0,
            "max_nonterminal_trapped_next_fraction": 1.0,
            "max_exact_mask_state_mismatch_fraction": 1.0,
            "max_malformed_state_feature_fraction": 1.0,
        }
        assert policy._offline_replay_load == {
            "db_path": str(db_path),
            "effective_limit": policy.memory.capacity,
            "loaded_rows": 6,
            "replay_order": "id",
            "requested_limit": None,
            "batch_size": GameConfig.APEX_BATCH_SIZE,
            "min_row_count": 6,
        }
        assert policy._offline_replay_metadata == {
            "generation.action_size": GameConfig.OUTPUT_SIZE,
            "generation.episodes": 6,
            "generation.mode": "parallel",
            "generation.reward_contract": current_reward_contract(),
            "generation.reward_death": GameConfig.REWARD_DEATH,
            "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
            "generation.state_size": GameConfig.INPUT_SIZE,
        }

    def test_load_replay_database_reports_loaded_snake_id_distribution(self, tmp_path, capsys):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            for snake_id, row_count in ((0, 1), (1, 3)):
                handler.save_memories(
                    snake_id=snake_id,
                    memories=[
                        {
                            "state": [float(idx)] * GameConfig.INPUT_SIZE,
                            "action": idx % GameConfig.OUTPUT_SIZE,
                            "reward": 1.0,
                            "next_state": [float(idx + 1)] * GameConfig.INPUT_SIZE,
                            "done": False,
                            "priority": 1.0,
                            "bootstrap_steps": 1,
                        }
                        for idx in range(row_count)
                    ],
                )
        finally:
            handler.close()
        policy = self.PolicyStub()

        loaded = load_replay_database(policy, str(db_path), limit=None, replay_order="id")
        output = capsys.readouterr().out

        assert loaded == 4
        assert "Rows per snake_id min/avg/max: 1/2.00/3" in output

    def test_load_replay_database_quality_gate_does_not_mutate_memory(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 1.0,
                        "next_state": [0.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        with pytest.raises(RuntimeError, match="terminal fraction"):
            load_replay_database(
                policy,
                str(db_path),
                limit=None,
                min_terminal_fraction=0.01,
            )

        assert policy.memory.cleared is False
        assert policy.memory.add_bulk_kwargs is None
        assert policy.memory.existing_rows == ["keep-me"]

    def test_load_replay_database_action_gate_does_not_mutate_memory(self, tmp_path):
        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [float(idx)] * GameConfig.INPUT_SIZE,
                        "action": idx % 2,
                        "reward": 1.0,
                        "next_state": [float(idx + 1)] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                    for idx in range(4)
                ],
            )
        finally:
            handler.close()
        policy = self.PolicyStub()

        with pytest.raises(RuntimeError, match="action coverage"):
            load_replay_database(
                policy,
                str(db_path),
                limit=None,
                min_action_coverage_fraction=0.5,
            )

        assert policy.memory.cleared is False
        assert policy.memory.add_bulk_kwargs is None
        assert policy.memory.existing_rows == ["keep-me"]


class TestTrainOffline:
    """Tests for offline gradient update loop diagnostics."""

    class PolicyStub:
        def __init__(self):
            self.memory = [object()]
            self.update_counter = 0
            self._last_train_metrics = {}

        def _min_replay_size(self):
            return 1

        def train_step(self):
            self.update_counter += 1
            self._last_train_metrics = {
                "valid_next_action_fraction": 0.75,
                "trapped_next_state_fraction": 0.25,
                "exact_next_action_mask_fraction": 0.5,
            }
            self._offline_replay_quality = {
                "count": 42,
                "active_action_count": 6,
                "dominant_action": 0,
                "dominant_action_fraction": 0.25,
            }
            self._offline_replay_gates = {
                "min_terminal_fraction": 0.005,
                "min_immediate_terminal_fraction": 0.001,
                "min_exact_mask_fraction": 0.9,
                "min_boost_mask_fraction": 0.05,
                "min_action_coverage_fraction": 1.0,
                "min_positive_reward_fraction": 0.1,
                "min_negative_reward_fraction": 0.2,
                "min_multistep_fraction": 0.4,
                "max_dominant_action_fraction": 0.8,
                "max_invalid_current_action_fraction": 0.3,
                "max_nonterminal_trapped_next_fraction": 0.0,
            }
            self._offline_replay_load = {
                "db_path": "replay.db",
                "effective_limit": 42,
                "loaded_rows": 42,
                "replay_order": "id_uniform",
                "requested_limit": None,
                "batch_size": GameConfig.BATCH_SIZE,
            }
            self._offline_replay_metadata = {
                "generation.mode": "single",
                "generation.episodes": 12,
                "generation.state_size": GameConfig.INPUT_SIZE,
                "generation.gamma": GameConfig.APEX_GAMMA,
                "generation.apex_n_step": GameConfig.APEX_N_STEP,
                "generation.reward_contract": current_reward_contract(),
                "generation.reward_death": GameConfig.REWARD_DEATH,
                "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
            }
            self._offline_replay_warnings = [
                "   Action 0 accounts for 80.0% of replay rows; "
                "policy updates may overfit one behavior",
                "   Replay priorities are flat at 1.000000",
            ]
            return 0.25, 0.5

    def test_train_offline_logs_target_action_metrics(self, tmp_path, capsys):
        policy = self.PolicyStub()

        iterations_done, last_loss = train_offline(
            policy=policy,
            iterations=1,
            log_interval=1,
            checkpoint_interval=0,
            checkpoint_dir=str(tmp_path),
            checkpoint_filename="offline.pth",
            db_path="replay.db",
        )
        output = capsys.readouterr().out

        assert iterations_done == 1
        assert last_loss == pytest.approx(0.25)
        assert "Target actions: valid=75.0%, trapped=25.0%, exact_masks=50.0%" in output

    def test_train_offline_raises_when_replay_is_too_small(self, tmp_path):
        policy = self.PolicyStub()
        policy.memory = []

        with pytest.raises(RuntimeError, match="Replay has"):
            train_offline(
                policy=policy,
                iterations=1,
                log_interval=1,
                checkpoint_interval=0,
                checkpoint_dir=str(tmp_path),
                checkpoint_filename="offline.pth",
                db_path="replay.db",
            )


class TestSavePolicyCheckpoint:
    """Tests for offline checkpoint metadata."""

    class MemoryStub:
        def __len__(self):
            return 42

    class PolicyStub:
        def __init__(self):
            self.memory = TestSavePolicyCheckpoint.MemoryStub()
            self.update_counter = 7
            self._last_train_metrics = {
                "valid_next_action_fraction": 0.75,
                "trapped_next_state_fraction": 0.25,
                "exact_next_action_mask_fraction": 0.5,
            }
            self._offline_replay_quality = {
                "count": 42,
                "active_action_count": 6,
                "dominant_action": 0,
                "dominant_action_fraction": 0.25,
            }
            self._offline_replay_gates = {
                "min_terminal_fraction": 0.005,
                "min_immediate_terminal_fraction": 0.001,
                "min_exact_mask_fraction": 0.9,
                "min_boost_mask_fraction": 0.05,
                "min_action_coverage_fraction": 1.0,
                "min_positive_reward_fraction": 0.1,
                "min_negative_reward_fraction": 0.2,
                "min_multistep_fraction": 0.4,
                "max_dominant_action_fraction": 0.8,
                "max_invalid_current_action_fraction": 0.3,
                "max_nonterminal_trapped_next_fraction": 0.0,
            }
            self._offline_replay_load = {
                "db_path": "replay.db",
                "effective_limit": 42,
                "loaded_rows": 42,
                "replay_order": "id_uniform",
                "requested_limit": None,
                "batch_size": GameConfig.BATCH_SIZE,
            }
            self._offline_replay_warnings = [
                "   Action 0 accounts for 80.0% of replay rows; "
                "policy updates may overfit one behavior",
                "   Replay priorities are flat at 1.000000",
            ]
            self._offline_replay_metadata = {
                "generation.mode": "single",
                "generation.episodes": 12,
                "generation.state_size": GameConfig.INPUT_SIZE,
                "generation.gamma": GameConfig.APEX_GAMMA,
                "generation.apex_n_step": GameConfig.APEX_N_STEP,
                "generation.reward_contract": current_reward_contract(),
                "generation.reward_death": GameConfig.REWARD_DEATH,
                "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
            }

        def get_state_dict(self):
            return {"weights": "stub"}

    def test_save_policy_checkpoint_includes_offline_train_metrics(self, tmp_path, monkeypatch):
        saved = {}

        class CheckpointManagerStub:
            def __init__(self, checkpoint_dir):
                self.checkpoint_dir = checkpoint_dir

            def save_checkpoint_dict(self, checkpoint, filename):
                saved["checkpoint_dir"] = self.checkpoint_dir
                saved["checkpoint"] = checkpoint
                saved["filename"] = filename
                return f"{self.checkpoint_dir}/{filename}"

        checkpoint_module = types.ModuleType("src.model.checkpoint_manager")
        checkpoint_module.CheckpointManager = CheckpointManagerStub
        monkeypatch.setitem(sys.modules, "src.model.checkpoint_manager", checkpoint_module)

        output_path = save_policy_checkpoint(
            policy=self.PolicyStub(),
            checkpoint_dir=str(tmp_path),
            filename="offline.pth",
            db_path="replay.db",
            iterations_done=3,
            avg_loss=0.125,
            elapsed_seconds=12.5,
        )

        assert output_path == f"{tmp_path}/offline.pth"
        assert saved["checkpoint_dir"] == str(tmp_path)
        assert saved["filename"] == "offline.pth"
        assert saved["checkpoint"]["offline_train_metrics"] == {
            "valid_next_action_fraction": 0.75,
            "trapped_next_state_fraction": 0.25,
            "exact_next_action_mask_fraction": 0.5,
        }
        assert saved["checkpoint"]["offline_replay_quality"] == {
            "count": 42,
            "active_action_count": 6,
            "dominant_action": 0,
            "dominant_action_fraction": 0.25,
        }
        assert saved["checkpoint"]["offline_replay_gates"] == {
            "min_terminal_fraction": 0.005,
            "min_immediate_terminal_fraction": 0.001,
            "min_exact_mask_fraction": 0.9,
            "min_boost_mask_fraction": 0.05,
            "min_action_coverage_fraction": 1.0,
            "min_positive_reward_fraction": 0.1,
            "min_negative_reward_fraction": 0.2,
            "min_multistep_fraction": 0.4,
            "max_dominant_action_fraction": 0.8,
            "max_invalid_current_action_fraction": 0.3,
            "max_nonterminal_trapped_next_fraction": 0.0,
        }
        assert saved["checkpoint"]["offline_replay_load"] == {
            "db_path": "replay.db",
            "effective_limit": 42,
            "loaded_rows": 42,
            "replay_order": "id_uniform",
            "requested_limit": None,
            "batch_size": GameConfig.BATCH_SIZE,
        }
        assert saved["checkpoint"]["offline_replay_metadata"] == {
            "generation.mode": "single",
            "generation.episodes": 12,
            "generation.state_size": GameConfig.INPUT_SIZE,
            "generation.gamma": GameConfig.APEX_GAMMA,
            "generation.apex_n_step": GameConfig.APEX_N_STEP,
            "generation.reward_contract": current_reward_contract(),
            "generation.reward_death": GameConfig.REWARD_DEATH,
            "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
        }
        assert saved["checkpoint"]["offline_replay_warnings"] == [
            "   Action 0 accounts for 80.0% of replay rows; "
            "policy updates may overfit one behavior",
            "   Replay priorities are flat at 1.000000",
        ]
        assert saved["checkpoint"]["replay_size"] == 42
