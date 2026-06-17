"""Tests for experience generation helpers."""

from dataclasses import replace

import pytest

from src.core.game_config import GameConfig, get_config, initialize_config
from src.core.reward_contract import current_reward_contract
from src.data.memory_db_handler import MemoryDBHandler
from src.scripts.generate_experiences import (
    DEFAULT_BOOST_EXPLORATION_RATE,
    DEFAULT_DANGER_EXPLORATION_RATE,
    DEFAULT_GENERATION_ENV_PRESET,
    DEFAULT_GENERATION_REPLAY_QUALITY_PRESET,
    apply_generated_priority_fallback,
    build_generation_metadata,
    build_generation_quality_metadata,
    build_generation_replay_contract,
    build_generation_replay_quality_gates,
    collect_parallel_merge_failures,
    collect_parallel_worker_failures,
    compute_generation_actor_epsilons,
    configure_generation_exploration,
    filter_untrainable_generated_memories,
    format_audit_replay_command,
    get_env_database_path,
    get_generation_checkpoint_candidates,
    get_parallel_memory_snake_id,
    group_memories_by_snake_id,
    load_generated_memories_for_merge,
    load_shared_apex_model,
    main,
    prepare_generation_output_database,
    remove_sqlite_files,
    resolve_generation_checkpoint_path,
    resolve_generation_environment_preset,
    resolve_generation_environment_settings,
    resolve_generation_frame_limit,
    resolve_generation_max_dominant_action_fraction,
    resolve_generation_max_invalid_current_action_fraction,
    resolve_generation_max_nonterminal_trapped_next_fraction,
    resolve_generation_min_action_coverage_fraction,
    resolve_generation_min_boost_mask_fraction,
    resolve_generation_min_epsilon,
    resolve_generation_min_exact_mask_fraction,
    resolve_generation_min_immediate_terminal_fraction,
    resolve_generation_min_multistep_fraction,
    resolve_generation_min_negative_reward_fraction,
    resolve_generation_min_positive_reward_fraction,
    resolve_generation_min_terminal_fraction,
    resolve_generation_num_snakes,
    resolve_generation_quality_fraction,
    resolve_generation_replay_quality_gates,
    resolve_generation_scale,
    save_memories_by_snake_id,
    update_generation_environment,
    validate_append_replay_contract,
    validate_database_path,
    validate_generated_experience_count,
    validate_generation_checkpoint_contract,
    validate_parallel_merge_counts,
    validate_replay_quality_gates,
    validate_replay_terminal_fraction,
)


def _valid_replay_state(offset: float = 0.0) -> list[float]:
    state = [0.0] * GameConfig.INPUT_SIZE
    state[1] = 1.0
    state[5] = max(min(offset, 1.0), -1.0)
    state[57] = 0.0
    return state


def _one_replay_memory() -> dict:
    state = _valid_replay_state()
    return {
        "state": state,
        "action": 1,
        "reward": 0.25,
        "next_state": _valid_replay_state(0.1),
        "done": False,
        "priority": 1.0,
        "bootstrap_steps": 1,
        "next_action_mask": [True, True, True, False, False, False],
    }


class TestGenerationDatabasePaths:
    """Tests for generated replay database path handling."""

    def test_parallel_env_db_path_keeps_parent_and_suffix(self):
        assert get_env_database_path("runs/replay.db", 2) == "runs/replay_env2.db"

    def test_parallel_env_db_path_adds_db_suffix_when_missing(self):
        assert get_env_database_path("runs/replay", 1) == "runs/replay_env1.db"

    def test_empty_db_path_raises(self):
        with pytest.raises(ValueError, match="db path"):
            validate_database_path("  ")

    def test_negative_env_id_raises(self):
        with pytest.raises(ValueError, match="env_id"):
            get_env_database_path("replay.db", -1)

    def test_remove_sqlite_files_removes_database_and_sidecars(self, tmp_path):
        db_path = tmp_path / "replay_env0.db"
        for suffix in ("", "-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).write_text("x")

        remove_sqlite_files(str(db_path))

        for suffix in ("", "-wal", "-shm"):
            assert not db_path.with_name(db_path.name + suffix).exists()

    def test_prepare_output_database_removes_stale_rows_by_default(self, tmp_path):
        db_path = tmp_path / "replay.db"
        for suffix in ("", "-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).write_text("stale")

        output_path = prepare_generation_output_database(str(db_path))

        assert output_path == str(db_path)
        for suffix in ("", "-wal", "-shm"):
            assert not db_path.with_name(db_path.name + suffix).exists()

    def test_prepare_output_database_preserves_existing_db_in_append_mode(self, tmp_path):
        db_path = tmp_path / "replay.db"
        db_path.write_text("existing")

        output_path = prepare_generation_output_database(str(db_path), append=True)

        assert output_path == str(db_path)
        assert db_path.read_text() == "existing"

    def test_append_contract_allows_empty_database_without_metadata(self, tmp_path):
        db_path = tmp_path / "empty_replay.db"
        handler = MemoryDBHandler(db_name=str(db_path))
        try:
            validate_append_replay_contract(
                handler,
                build_generation_replay_contract(resolve_generation_environment_settings()),
                str(db_path),
                append=True,
            )
        finally:
            handler.close()

    def test_append_contract_rejects_rows_without_generation_metadata(self, tmp_path):
        db_path = tmp_path / "legacy_replay.db"
        handler = MemoryDBHandler(db_name=str(db_path))
        try:
            handler.save_memories(0, [_one_replay_memory()])

            with pytest.raises(RuntimeError, match="no generation metadata"):
                validate_append_replay_contract(
                    handler,
                    build_generation_replay_contract(resolve_generation_environment_settings()),
                    str(db_path),
                    append=True,
                )
        finally:
            handler.close()

    def test_append_contract_rejects_stale_reward_contract_rows(self, tmp_path):
        db_path = tmp_path / "stale_reward_replay.db"
        env_settings = resolve_generation_environment_settings()
        stale_contract = build_generation_replay_contract(env_settings)
        stale_reward_contract = dict(stale_contract["generation.reward_contract"])
        stale_reward_contract["survival"] = float(stale_reward_contract["survival"]) + 1.0
        stale_contract["generation.reward_contract"] = stale_reward_contract

        handler = MemoryDBHandler(db_name=str(db_path))
        try:
            handler.update_metadata(stale_contract)
            handler.save_memories(0, [_one_replay_memory()])

            with pytest.raises(RuntimeError, match="generation.reward_contract.survival"):
                validate_append_replay_contract(
                    handler,
                    build_generation_replay_contract(env_settings),
                    str(db_path),
                    append=True,
                )
        finally:
            handler.close()

    def test_append_contract_accepts_matching_existing_replay_rows(self, tmp_path):
        db_path = tmp_path / "matching_replay.db"
        env_settings = resolve_generation_environment_settings(num_snakes=2)
        handler = MemoryDBHandler(db_name=str(db_path))
        try:
            handler.update_metadata(build_generation_replay_contract(env_settings))
            handler.save_memories(0, [_one_replay_memory()])

            validate_append_replay_contract(
                handler,
                build_generation_replay_contract(env_settings),
                str(db_path),
                append=True,
            )
        finally:
            handler.close()


class TestGenerationNextSteps:
    """Tests for generated replay follow-up command formatting."""

    def test_audit_command_defaults_to_training_replay_contract(self):
        command = format_audit_replay_command("replay.db")

        assert (
            command == "python src/scripts/audit_replay.py --db replay.db "
            "--preset training --print-gate-args"
        )

    def test_audit_command_quotes_db_path_and_preserves_preset_overrides(self):
        gates = resolve_generation_replay_quality_gates(
            preset="training",
            overrides={
                "min_action_coverage_fraction": 0.5,
                "max_dominant_action_fraction": 1.0,
                "max_invalid_current_action_fraction": 0.2,
            },
        )

        command = format_audit_replay_command(
            "/tmp/replay db.sqlite",
            replay_quality_preset="training",
            gates=gates,
            min_row_count=128,
            expected_gamma=0.99,
            expected_n_step=3,
            config_path="configs/training_fast.yaml",
        )

        assert (
            command == "python src/scripts/audit_replay.py --db '/tmp/replay db.sqlite' "
            "--config configs/training_fast.yaml "
            "--min-row-count 128 "
            "--expected-gamma 0.99 --expected-n-step 3 "
            "--preset training --min-action-coverage-fraction 0.5 "
            "--max-dominant-action-fraction 1 "
            "--max-invalid-current-action-fraction 0.2 --print-gate-args"
        )

    def test_audit_command_preserves_generated_current_action_gate(self):
        gates = build_generation_replay_quality_gates(
            max_invalid_current_action_fraction=0.2,
        )

        command = format_audit_replay_command(
            "replay.db",
            replay_quality_preset="none",
            gates=gates,
        )

        assert "--preset none" in command
        assert "--max-invalid-current-action-fraction 0.2" in command


class TestGenerationMetadata:
    """Tests for durable generated-replay metadata."""

    def test_generation_metadata_records_effective_replay_contract(self):
        metadata = build_generation_metadata(
            mode="single",
            episodes=7,
            save_interval=2,
            frame_limit=123,
            env_settings={
                "num_snakes": 3,
                "board_scale": 0.5,
                "food_multiplier": 0.25,
            },
            load_model=True,
            model_loaded=False,
            checkpoint_path="best.pth",
            resolved_checkpoint_path=None,
            exploration_epsilon=0.8,
            exploration_min_epsilon=0.1,
            epsilon_min=0.1,
            epsilon_max=0.8,
            boost_exploration_rate=0.3,
            danger_exploration_rate=0.2,
            replay_quality_preset="training",
            replay_gates={"min_terminal_fraction": 0.005},
            min_row_count=64,
            append=False,
            config_path="configs/training_fast.yaml",
            num_envs=None,
        )

        assert metadata["generation.policy_type"] == "apex"
        assert metadata["generation.mode"] == "single"
        assert metadata["generation.state_size"] == get_config().network.input_size
        assert metadata["generation.action_size"] == get_config().network.output_size
        assert metadata["generation.num_snakes"] == 3
        assert metadata["generation.board_width"] == int(get_config().game.width * 0.5)
        assert metadata["generation.board_height"] == int(get_config().game.height * 0.5)
        assert metadata["generation.initial_food"] == int(get_config().game.initial_food * 0.25)
        assert metadata["generation.frame_limit"] == 123
        assert metadata["generation.load_model"] is True
        assert metadata["generation.model_loaded"] is False
        assert metadata["generation.checkpoint_path"] == "best.pth"
        assert metadata["generation.resolved_checkpoint_path"] is None
        assert metadata["generation.epsilon_min"] == 0.1
        assert metadata["generation.epsilon_max"] == 0.8
        assert metadata["generation.gamma"] == get_config().apex.gamma
        assert metadata["generation.apex_n_step"] == get_config().apex.n_step
        assert metadata["generation.reward_contract"] == current_reward_contract()
        assert metadata["generation.reward_death"] == get_config().rewards.death
        assert metadata["generation.reward_food_base"] == get_config().rewards.food_base
        assert metadata["generation.quality_gates"] == {"min_terminal_fraction": 0.005}
        assert metadata["generation.min_row_count"] == 64

    def test_generation_quality_metadata_records_observed_replay_contract(self):
        metadata = build_generation_quality_metadata(
            {
                "action_counts": {0: 7, 3: 2},
                "active_action_count": 2,
                "boost_mask_fraction": 0.25,
                "count": 10,
                "dominant_action": 0,
                "dominant_action_fraction": 0.7,
                "exact_mask_state_mismatch_fraction": 0.0,
                "invalid_current_action_fraction": 0.1,
                "malformed_state_feature_fraction": 0.0,
                "multistep_fraction": 0.8,
                "nonterminal_mask_fraction": 0.9,
                "nonterminal_trapped_next_state_fraction": 0.05,
                "normalized_action_entropy": 0.5,
                "reward_negative_count": 1,
                "reward_positive_count": 3,
                "reward_zero_count": 6,
                "terminal_immediate_nonnegative_reward_count": 0,
                "terminal_immediate_nonnegative_reward_fraction": 0.0,
                "terminal_multistep_nonnegative_reward_count": 1,
                "terminal_multistep_nonnegative_reward_fraction": 0.5,
                "terminal_nonnegative_reward_count": 1,
                "terminal_nonnegative_reward_fraction": 0.5,
                "terminal_fraction": 0.2,
            }
        )

        quality = metadata["generation.replay_quality"]
        assert quality["count"] == 10
        assert quality["terminal_fraction"] == pytest.approx(0.2)
        assert quality["nonterminal_mask_fraction"] == pytest.approx(0.9)
        assert quality["action_counts"] == [7, 0, 0, 2, 0, 0]
        assert quality["reward_negative_count"] == 1
        assert quality["reward_zero_count"] == 6
        assert quality["reward_positive_count"] == 3
        assert quality["terminal_immediate_nonnegative_reward_count"] == 0
        assert quality["terminal_immediate_nonnegative_reward_fraction"] == pytest.approx(0.0)
        assert quality["terminal_multistep_nonnegative_reward_count"] == 1
        assert quality["terminal_multistep_nonnegative_reward_fraction"] == pytest.approx(0.5)
        assert quality["terminal_nonnegative_reward_count"] == 1
        assert quality["terminal_nonnegative_reward_fraction"] == pytest.approx(0.5)
        assert quality["invalid_current_action_fraction"] == pytest.approx(0.1)


class TestGenerationCheckpointResolution:
    """Tests for loading trained weights before generated replay collection."""

    def test_auto_resolver_prefers_apex_checkpoint_in_configured_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        apex_checkpoint = checkpoint_dir / "best_apex.pth"
        configured_checkpoint = checkpoint_dir / "winner.pth"
        apex_checkpoint.write_bytes(b"apex")
        configured_checkpoint.write_bytes(b"configured")
        custom_config = replace(
            original_config,
            checkpoint=replace(
                original_config.checkpoint,
                checkpoint_dir=str(checkpoint_dir),
                best_model_name="winner.pth",
            ),
        )

        try:
            initialize_config(custom_config)

            assert resolve_generation_checkpoint_path() == apex_checkpoint
        finally:
            initialize_config(original_config)

    def test_auto_resolver_falls_back_to_configured_best_model_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        configured_checkpoint = checkpoint_dir / "winner.pth"
        configured_checkpoint.write_bytes(b"configured")
        custom_config = replace(
            original_config,
            checkpoint=replace(
                original_config.checkpoint,
                checkpoint_dir=str(checkpoint_dir),
                best_model_name="winner.pth",
            ),
        )

        try:
            initialize_config(custom_config)

            assert resolve_generation_checkpoint_path() == configured_checkpoint
        finally:
            initialize_config(original_config)

    def test_auto_resolver_falls_back_to_legacy_saved_snakes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        original_config = get_config()
        checkpoint_dir = tmp_path / "empty_checkpoints"
        checkpoint_dir.mkdir()
        legacy_dir = tmp_path / "saved_snakes"
        legacy_dir.mkdir()
        legacy_checkpoint = legacy_dir / "best_snake.pth"
        legacy_checkpoint.write_bytes(b"legacy")
        custom_config = replace(
            original_config,
            checkpoint=replace(original_config.checkpoint, checkpoint_dir=str(checkpoint_dir)),
        )

        try:
            initialize_config(custom_config)

            assert resolve_generation_checkpoint_path() == legacy_checkpoint
        finally:
            initialize_config(original_config)

    def test_explicit_checkpoint_resolves_configured_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "custom.pth"
        checkpoint.write_bytes(b"custom")
        custom_config = replace(
            original_config,
            checkpoint=replace(original_config.checkpoint, checkpoint_dir=str(checkpoint_dir)),
        )

        try:
            initialize_config(custom_config)

            assert resolve_generation_checkpoint_path("custom.pth") == checkpoint
        finally:
            initialize_config(original_config)

    def test_empty_explicit_checkpoint_raises(self):
        with pytest.raises(ValueError, match="checkpoint path"):
            get_generation_checkpoint_candidates("  ")

    def test_generation_checkpoint_contract_accepts_matching_reward_scale(self):
        class PolicyStub:
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

        validate_generation_checkpoint_contract(
            {
                "apex_config": {
                    "input_size": GameConfig.INPUT_SIZE,
                    "hidden_size": GameConfig.HIDDEN_SIZE,
                    "output_size": GameConfig.OUTPUT_SIZE,
                    "n_step": GameConfig.APEX_N_STEP,
                    "gamma": GameConfig.APEX_GAMMA,
                    "reward_contract": current_reward_contract(),
                    "reward_death": GameConfig.REWARD_DEATH,
                    "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                    "use_gru": False,
                }
            },
            PolicyStub(),
            checkpoint_path="fresh_reward_checkpoint.pth",
        )

    def test_generation_checkpoint_contract_rejects_missing_reward_scale(self):
        class PolicyStub:
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

        with pytest.raises(RuntimeError, match="missing required reward_contract"):
            validate_generation_checkpoint_contract(
                {
                    "apex_config": {
                        "input_size": GameConfig.INPUT_SIZE,
                        "hidden_size": GameConfig.HIDDEN_SIZE,
                        "output_size": GameConfig.OUTPUT_SIZE,
                        "n_step": GameConfig.APEX_N_STEP,
                        "gamma": GameConfig.APEX_GAMMA,
                        "reward_death": GameConfig.REWARD_DEATH,
                        "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                        "use_gru": False,
                    }
                },
                PolicyStub(),
                checkpoint_path="legacy_reward_checkpoint.pth",
            )

    def test_generation_checkpoint_contract_rejects_reward_scale_mismatch(self):
        class PolicyStub:
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

        with pytest.raises(RuntimeError, match="reward_death=-3"):
            validate_generation_checkpoint_contract(
                {
                    "apex_config": {
                        "input_size": GameConfig.INPUT_SIZE,
                        "hidden_size": GameConfig.HIDDEN_SIZE,
                        "output_size": GameConfig.OUTPUT_SIZE,
                        "n_step": GameConfig.APEX_N_STEP,
                        "gamma": GameConfig.APEX_GAMMA,
                        "reward_contract": current_reward_contract(),
                        "reward_death": -3.0,
                        "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                        "use_gru": False,
                    }
                },
                PolicyStub(),
                checkpoint_path="stale_reward_checkpoint.pth",
            )

    def test_generation_checkpoint_contract_rejects_full_reward_contract_mismatch(self):
        class PolicyStub:
            input_size = GameConfig.INPUT_SIZE
            hidden_size = GameConfig.HIDDEN_SIZE
            output_size = GameConfig.OUTPUT_SIZE
            n_step = GameConfig.APEX_N_STEP
            gamma = GameConfig.APEX_GAMMA
            use_gru = False

        stale_contract = current_reward_contract()
        stale_contract["survival"] = float(stale_contract["survival"]) + 1.0

        with pytest.raises(RuntimeError, match="reward_contract.survival"):
            validate_generation_checkpoint_contract(
                {
                    "apex_config": {
                        "input_size": GameConfig.INPUT_SIZE,
                        "hidden_size": GameConfig.HIDDEN_SIZE,
                        "output_size": GameConfig.OUTPUT_SIZE,
                        "n_step": GameConfig.APEX_N_STEP,
                        "gamma": GameConfig.APEX_GAMMA,
                        "reward_contract": stale_contract,
                        "reward_death": GameConfig.REWARD_DEATH,
                        "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                        "use_gru": False,
                    }
                },
                PolicyStub(),
                checkpoint_path="stale_reward_checkpoint.pth",
            )

    def test_existing_checkpoint_load_failure_raises_instead_of_fresh_generation(self, tmp_path):
        from src.game.ai_snake import AISnake

        class UnloadableAISnake(AISnake):
            def load_state(self, filepath):
                return False

        class GameStateStub:
            snakes = [UnloadableAISnake.__new__(UnloadableAISnake)]

        checkpoint = tmp_path / "bad_checkpoint.pth"
        checkpoint.write_bytes(b"not a usable checkpoint")

        with pytest.raises(RuntimeError, match="Could not load Apex model"):
            load_shared_apex_model(GameStateStub(), checkpoint_path=str(checkpoint))


class TestParallelMergeOrdering:
    """Tests for preserving generated replay order during parallel DB merge."""

    def test_merge_loader_uses_insertion_order(self):
        class FakeDB:
            def __init__(self):
                self.calls = []

            def load_memories_for_policy(self, policy_type, limit, order_by):
                self.calls.append((policy_type, limit, order_by))
                return ([], [], [], [], [], [], [])

        db = FakeDB()

        result = load_generated_memories_for_merge(db, "apex")

        assert result == ([], [], [], [], [], [], [])
        assert db.calls[-1] == ("apex", None, "id")

    def test_merge_loader_requests_action_masks_and_snake_ids_when_supported(self):
        class FakeDB:
            def __init__(self):
                self.calls = []

            def load_memories_for_policy(
                self,
                policy_type,
                limit,
                order_by,
                include_action_masks=False,
                include_snake_ids=False,
            ):
                self.calls.append(
                    (policy_type, limit, order_by, include_action_masks, include_snake_ids)
                )
                return ([], [], [], [], [], [], [], [], [])

        db = FakeDB()

        result = load_generated_memories_for_merge(db, "apex")

        assert result == ([], [], [], [], [], [], [], [], [])
        assert db.calls == [("apex", None, "id", True, True)]

    def test_group_memories_uses_producer_snake_id_with_fallback(self):
        memories = [
            {"action": 0, "snake_id": 3},
            {"action": 1, "stream_id": 4},
            {"action": 2},
            {"action": 3, "snake_id": "bad", "stream_id": 5},
            {"action": 4, "snake_id": "bad"},
        ]

        grouped = group_memories_by_snake_id(memories, default_snake_id=9)

        assert [memory["action"] for memory in grouped[3]] == [0]
        assert [memory["action"] for memory in grouped[4]] == [1]
        assert [memory["action"] for memory in grouped[5]] == [3]
        assert [memory["action"] for memory in grouped[9]] == [2, 4]

    def test_parallel_memory_snake_id_is_unique_across_envs(self):
        assert get_parallel_memory_snake_id(env_id=0, local_snake_id=3, snakes_per_env=4) == 3
        assert get_parallel_memory_snake_id(env_id=1, local_snake_id=0, snakes_per_env=4) == 4
        assert get_parallel_memory_snake_id(env_id=1, local_snake_id=3, snakes_per_env=4) == 7

    def test_save_memories_by_snake_id_groups_db_writes(self):
        class FakeDB:
            def __init__(self):
                self.calls = []

            def save_memories(self, snake_id, memories, policy_type="apex"):
                self.calls.append(
                    (snake_id, [memory["action"] for memory in memories], policy_type)
                )

        db = FakeDB()
        memories = [
            {"action": 0, "snake_id": 3},
            {"action": 1, "snake_id": 4},
            {"action": 2, "snake_id": 3},
        ]

        save_memories_by_snake_id(db, memories, default_snake_id=9, policy_type="apex")

        assert db.calls == [(3, [0, 2], "apex"), (4, [1], "apex")]


class TestParallelWorkerFailures:
    """Tests for detecting failed generator workers before DB merge."""

    class FakeProcess:
        """Minimal process stand-in exposing the exitcode contract."""

        def __init__(self, exitcode):
            self.exitcode = exitcode

    def test_no_failures_when_all_workers_finish_and_report_success(self):
        failures = collect_parallel_worker_failures(
            [(0, self.FakeProcess(0)), (1, self.FakeProcess(0))],
            [(0, 3), (1, 3)],
            {
                0: {"experiences": 10, "avg_reward": 1.0, "time": 0.1, "error": None},
                1: {"experiences": 12, "avg_reward": 2.0, "time": 0.1, "error": None},
            },
        )

        assert failures == []

    def test_reports_nonzero_process_exitcode(self):
        failures = collect_parallel_worker_failures(
            [(0, self.FakeProcess(2))],
            [(0, 1)],
            {0: {"experiences": 0, "avg_reward": 0.0, "time": 0.0, "error": None}},
        )

        assert failures == ["env 0 exited with code 2"]

    def test_reports_missing_worker_stats(self):
        failures = collect_parallel_worker_failures(
            [(1, self.FakeProcess(0))],
            [(1, 1)],
            {},
        )

        assert failures == ["env 1 did not report generation stats"]

    def test_reports_worker_exception_message(self):
        failures = collect_parallel_worker_failures(
            [(2, self.FakeProcess(0))],
            [(2, 1)],
            {2: {"experiences": 0, "avg_reward": 0.0, "time": 0.0, "error": "boom"}},
        )

        assert failures == ["env 2 failed: boom"]

    def test_reports_clean_worker_with_no_replay(self):
        failures = collect_parallel_worker_failures(
            [(4, self.FakeProcess(0))],
            [(4, 1)],
            {4: {"experiences": 0, "avg_reward": 0.0, "time": 0.1, "error": None}},
        )

        assert failures == ["env 4 produced no replay memories"]

    @pytest.mark.parametrize("experiences", ["many", "1", 1.5, True])
    def test_reports_clean_worker_with_invalid_replay_count(self, experiences):
        failures = collect_parallel_worker_failures(
            [(5, self.FakeProcess(0))],
            [(5, 1)],
            {5: {"experiences": experiences, "avg_reward": 0.0, "time": 0.1, "error": None}},
        )

        assert failures == ["env 5 reported invalid experience count"]

    def test_reports_unfinished_process(self):
        failures = collect_parallel_worker_failures(
            [(3, self.FakeProcess(None))],
            [(3, 1)],
            {3: {"experiences": 0, "avg_reward": 0.0, "time": 0.0, "error": None}},
        )

        assert failures == ["env 3 did not finish"]


class TestParallelMergeFailures:
    """Tests for detecting missing worker replay databases before merge."""

    def test_no_failures_when_worker_database_exists(self, tmp_path):
        output_db = tmp_path / "replay.db"
        env_db = tmp_path / "replay_env0.db"
        env_db.write_bytes(b"sqlite placeholder")

        failures = collect_parallel_merge_failures([(0, 1)], str(output_db))

        assert failures == []

    def test_reports_missing_worker_database(self, tmp_path):
        output_db = tmp_path / "replay.db"

        failures = collect_parallel_merge_failures([(0, 1), (1, 1)], str(output_db))

        assert failures == [
            f"env 0 database missing: {tmp_path / 'replay_env0.db'}",
            f"env 1 database missing: {tmp_path / 'replay_env1.db'}",
        ]


class TestGeneratedReplayCountValidation:
    """Tests for rejecting empty or partially merged replay datasets."""

    def test_positive_generation_count_is_valid(self):
        validate_generated_experience_count(1)

    def test_zero_generation_count_raises(self):
        with pytest.raises(RuntimeError, match="no replay memories"):
            validate_generated_experience_count(0)

    def test_parallel_merge_counts_match(self):
        validate_parallel_merge_counts(reported_experiences=10, merged_experiences=10)

    def test_parallel_merge_count_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="workers reported"):
            validate_parallel_merge_counts(reported_experiences=10, merged_experiences=8)


class TestGeneratedReplayQualityGates:
    """Tests for optional generated replay quality gates."""

    def test_default_quality_fraction_is_disabled(self):
        assert resolve_generation_quality_fraction(
            None, "min_exact_mask_fraction"
        ) == pytest.approx(0.0)

    def test_explicit_quality_fraction_is_preserved(self):
        assert resolve_generation_quality_fraction(
            0.25, "min_exact_mask_fraction"
        ) == pytest.approx(0.25)

    def test_replay_quality_preset_resolves_with_overrides(self):
        gates = resolve_generation_replay_quality_gates(
            preset="training",
            overrides={
                "min_action_coverage_fraction": 0.5,
                "max_dominant_action_fraction": 0.9,
            },
        )

        assert gates["min_terminal_fraction"] == pytest.approx(0.005)
        assert gates["min_immediate_terminal_fraction"] == pytest.approx(0.001)
        assert gates["min_boost_mask_fraction"] == pytest.approx(0.05)
        assert gates["min_action_coverage_fraction"] == pytest.approx(0.5)
        assert gates["min_positive_reward_fraction"] == pytest.approx(0.005)
        assert gates["min_negative_reward_fraction"] == pytest.approx(0.005)
        assert gates["max_dominant_action_fraction"] == pytest.approx(0.9)
        assert gates["max_invalid_current_action_fraction"] == pytest.approx(0.0)
        assert gates["max_exact_mask_state_mismatch_fraction"] == pytest.approx(0.0)
        assert gates["max_malformed_state_feature_fraction"] == pytest.approx(0.0)

    def test_default_replay_quality_preset_requires_training_signal(self):
        gates = resolve_generation_replay_quality_gates()

        assert DEFAULT_GENERATION_REPLAY_QUALITY_PRESET == "training"
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
        gates = resolve_generation_replay_quality_gates(preset="none")

        assert gates["min_terminal_fraction"] == pytest.approx(0.0)
        assert gates["min_immediate_terminal_fraction"] == pytest.approx(0.0)
        assert gates["min_exact_mask_fraction"] == pytest.approx(0.0)
        assert gates["min_boost_mask_fraction"] == pytest.approx(0.0)
        assert gates["max_dominant_action_fraction"] == pytest.approx(1.0)
        assert gates["max_exact_mask_state_mismatch_fraction"] == pytest.approx(1.0)
        assert gates["max_malformed_state_feature_fraction"] == pytest.approx(1.0)

    def test_default_min_terminal_fraction_is_disabled(self):
        assert resolve_generation_min_terminal_fraction(None) == pytest.approx(0.0)

    def test_default_min_immediate_terminal_fraction_is_disabled(self):
        assert resolve_generation_min_immediate_terminal_fraction(None) == pytest.approx(0.0)

    def test_default_min_exact_mask_fraction_is_disabled(self):
        assert resolve_generation_min_exact_mask_fraction(None) == pytest.approx(0.0)

    def test_default_min_boost_mask_fraction_is_disabled(self):
        assert resolve_generation_min_boost_mask_fraction(None) == pytest.approx(0.0)

    def test_default_min_action_coverage_fraction_is_disabled(self):
        assert resolve_generation_min_action_coverage_fraction(None) == pytest.approx(0.0)

    def test_default_min_multistep_fraction_is_disabled(self):
        assert resolve_generation_min_multistep_fraction(None) == pytest.approx(0.0)

    def test_default_min_positive_reward_fraction_is_disabled(self):
        assert resolve_generation_min_positive_reward_fraction(None) == pytest.approx(0.0)

    def test_default_min_negative_reward_fraction_is_disabled(self):
        assert resolve_generation_min_negative_reward_fraction(None) == pytest.approx(0.0)

    def test_default_max_dominant_action_fraction_allows_all(self):
        assert resolve_generation_max_dominant_action_fraction(None) == pytest.approx(1.0)

    def test_default_max_invalid_current_action_fraction_allows_all(self):
        assert resolve_generation_max_invalid_current_action_fraction(None) == pytest.approx(1.0)

    def test_default_max_nonterminal_trapped_next_fraction_allows_all(self):
        assert resolve_generation_max_nonterminal_trapped_next_fraction(None) == pytest.approx(1.0)

    def test_explicit_min_terminal_fraction_is_preserved(self):
        assert resolve_generation_min_terminal_fraction(0.005) == pytest.approx(0.005)

    def test_explicit_min_immediate_terminal_fraction_is_preserved(self):
        assert resolve_generation_min_immediate_terminal_fraction(0.001) == pytest.approx(0.001)

    def test_explicit_min_exact_mask_fraction_is_preserved(self):
        assert resolve_generation_min_exact_mask_fraction(0.5) == pytest.approx(0.5)

    def test_explicit_min_boost_mask_fraction_is_preserved(self):
        assert resolve_generation_min_boost_mask_fraction(0.05) == pytest.approx(0.05)

    def test_explicit_action_diversity_fractions_are_preserved(self):
        assert resolve_generation_min_action_coverage_fraction(0.5) == pytest.approx(0.5)
        assert resolve_generation_min_positive_reward_fraction(0.1) == pytest.approx(0.1)
        assert resolve_generation_min_negative_reward_fraction(0.2) == pytest.approx(0.2)
        assert resolve_generation_min_multistep_fraction(0.4) == pytest.approx(0.4)
        assert resolve_generation_max_dominant_action_fraction(0.8) == pytest.approx(0.8)
        assert resolve_generation_max_invalid_current_action_fraction(0.3) == pytest.approx(0.3)
        assert resolve_generation_max_nonterminal_trapped_next_fraction(0.25) == pytest.approx(0.25)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
    def test_invalid_min_terminal_fraction_raises(self, fraction):
        with pytest.raises(ValueError, match="min_terminal_fraction"):
            resolve_generation_min_terminal_fraction(fraction)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
    def test_invalid_min_immediate_terminal_fraction_raises(self, fraction):
        with pytest.raises(ValueError, match="min_immediate_terminal_fraction"):
            resolve_generation_min_immediate_terminal_fraction(fraction)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
    def test_invalid_min_exact_mask_fraction_raises(self, fraction):
        with pytest.raises(ValueError, match="min_exact_mask_fraction"):
            resolve_generation_min_exact_mask_fraction(fraction)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
    def test_invalid_min_boost_mask_fraction_raises(self, fraction):
        with pytest.raises(ValueError, match="min_boost_mask_fraction"):
            resolve_generation_min_boost_mask_fraction(fraction)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
    def test_invalid_action_diversity_fractions_raise(self, fraction):
        with pytest.raises(ValueError, match="min_action_coverage_fraction"):
            resolve_generation_min_action_coverage_fraction(fraction)
        with pytest.raises(ValueError, match="min_positive_reward_fraction"):
            resolve_generation_min_positive_reward_fraction(fraction)
        with pytest.raises(ValueError, match="min_negative_reward_fraction"):
            resolve_generation_min_negative_reward_fraction(fraction)
        with pytest.raises(ValueError, match="min_multistep_fraction"):
            resolve_generation_min_multistep_fraction(fraction)
        with pytest.raises(ValueError, match="max_dominant_action_fraction"):
            resolve_generation_max_dominant_action_fraction(fraction)
        with pytest.raises(ValueError, match="max_invalid_current_action_fraction"):
            resolve_generation_max_invalid_current_action_fraction(fraction)
        with pytest.raises(ValueError, match="max_nonterminal_trapped_next_fraction"):
            resolve_generation_max_nonterminal_trapped_next_fraction(fraction)

    def test_terminal_fraction_gate_passes_when_disabled(self):
        quality = {"count": 1000, "done_count": 0, "terminal_fraction": 0.0}

        validate_replay_terminal_fraction(quality, min_terminal_fraction=0.0)

    def test_terminal_fraction_gate_passes_when_requirement_is_met(self):
        quality = {"count": 1000, "done_count": 8, "terminal_fraction": 0.008}

        validate_replay_terminal_fraction(quality, min_terminal_fraction=0.005)

    def test_terminal_fraction_gate_raises_when_requirement_is_missed(self):
        quality = {"count": 1000, "done_count": 2, "terminal_fraction": 0.002}

        with pytest.raises(RuntimeError, match="below the requested minimum"):
            validate_replay_terminal_fraction(quality, min_terminal_fraction=0.005)

    def test_quality_gates_pass_when_disabled(self):
        quality = {
            "count": 1000,
            "done_count": 0,
            "terminal_fraction": 0.0,
            "nonterminal_count": 1000,
            "nonterminal_mask_count": 0,
            "nonterminal_mask_fraction": 0.0,
        }

        validate_replay_quality_gates(quality)

    def test_min_row_count_gate_raises_when_requirement_is_missed(self):
        quality = {
            "count": 4,
            "done_count": 1,
            "terminal_fraction": 0.25,
        }

        with pytest.raises(RuntimeError, match="has 4 rows.*at least 8"):
            validate_replay_quality_gates(quality, min_row_count=8)

    def test_exact_mask_fraction_gate_raises_when_requirement_is_missed(self):
        quality = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 100,
            "nonterminal_mask_fraction": 100 / 990,
        }

        with pytest.raises(RuntimeError, match="exact-mask fraction"):
            validate_replay_quality_gates(quality, min_exact_mask_fraction=0.5)

    def test_boost_mask_fraction_gate_raises_when_requirement_is_missed(self):
        quality = {
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
            validate_replay_quality_gates(quality, min_boost_mask_fraction=0.05)

    def test_action_diversity_gates_raise_when_requirements_are_missed(self):
        quality = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "action_counts": {0: 850, 1: 150},
            "reward_positive_count": 10,
            "reward_negative_count": 10,
        }

        with pytest.raises(RuntimeError, match="action coverage"):
            validate_replay_quality_gates(quality, min_action_coverage_fraction=0.5)
        with pytest.raises(RuntimeError, match="positive-reward fraction"):
            validate_replay_quality_gates(quality, min_positive_reward_fraction=0.05)
        with pytest.raises(RuntimeError, match="negative-reward fraction"):
            validate_replay_quality_gates(quality, min_negative_reward_fraction=0.05)
        with pytest.raises(RuntimeError, match="multi-step fraction"):
            validate_replay_quality_gates(quality, min_multistep_fraction=0.5)
        with pytest.raises(RuntimeError, match="dominant-action fraction"):
            validate_replay_quality_gates(quality, max_dominant_action_fraction=0.8)

    def test_trapped_next_fraction_gate_raises_when_requirement_is_missed(self):
        quality = {
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
                quality,
                max_nonterminal_trapped_next_fraction=0.05,
            )


class TestGenerationEnvironmentStep:
    """Tests for replay-generation environment semantics."""

    def test_generation_update_uses_training_rules_without_learning_or_respawn(self):
        class FakeGameState:
            def __init__(self):
                self.calls = []

            def update(self, **kwargs):
                self.calls.append(kwargs)

        game_state = FakeGameState()

        update_generation_environment(game_state)

        assert game_state.calls == [
            {
                "train_mode": True,
                "learn": False,
                "allow_respawn": False,
            }
        ]


class TestGenerationCliConfig:
    """Tests for command-line config handling."""

    def test_missing_config_fails_before_generation(self, monkeypatch, tmp_path):
        missing_config = tmp_path / "missing.yaml"

        def fail_generation(*args, **kwargs):
            raise AssertionError("generation should not start with a missing config")

        monkeypatch.setattr(
            "src.scripts.generate_experiences.generate_experiences",
            fail_generation,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_experiences.py",
                "--episodes",
                "1",
                "--config",
                str(missing_config),
            ],
        )

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            main()

    def test_action_diversity_quality_args_are_passed_to_generation(self, monkeypatch, tmp_path):
        captured = {}

        def fake_generation(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(
            "src.scripts.generate_experiences.generate_experiences",
            fake_generation,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_experiences.py",
                "--episodes",
                "1",
                "--fresh",
                "--db",
                str(tmp_path / "replay.db"),
                "--min-action-coverage-fraction",
                "0.5",
                "--min-positive-reward-fraction",
                "0.1",
                "--min-negative-reward-fraction",
                "0.2",
                "--min-immediate-terminal-fraction",
                "0.003",
                "--min-multistep-fraction",
                "0.4",
                "--max-dominant-action-fraction",
                "0.8",
                "--max-invalid-current-action-fraction",
                "0.3",
                "--max-nonterminal-trapped-next-fraction",
                "0.25",
                "--max-exact-mask-state-mismatch-fraction",
                "0.15",
                "--max-malformed-state-feature-fraction",
                "0.05",
                "--min-row-count",
                "64",
            ],
        )

        main()

        assert captured["kwargs"]["min_action_coverage_fraction"] == pytest.approx(0.5)
        assert captured["kwargs"]["min_positive_reward_fraction"] == pytest.approx(0.1)
        assert captured["kwargs"]["min_negative_reward_fraction"] == pytest.approx(0.2)
        assert captured["kwargs"]["min_immediate_terminal_fraction"] == pytest.approx(0.003)
        assert captured["kwargs"]["min_multistep_fraction"] == pytest.approx(0.4)
        assert captured["kwargs"]["max_dominant_action_fraction"] == pytest.approx(0.8)
        assert captured["kwargs"]["max_invalid_current_action_fraction"] == pytest.approx(0.3)
        assert captured["kwargs"]["max_nonterminal_trapped_next_fraction"] == pytest.approx(0.25)
        assert captured["kwargs"]["max_exact_mask_state_mismatch_fraction"] == pytest.approx(0.15)
        assert captured["kwargs"]["max_malformed_state_feature_fraction"] == pytest.approx(0.05)
        assert captured["kwargs"]["min_row_count"] == 64

    def test_min_row_count_is_passed_to_parallel_generation(self, monkeypatch, tmp_path):
        captured = {}

        def fake_parallel_generation(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(
            "src.scripts.generate_experiences.generate_experiences_parallel",
            fake_parallel_generation,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_experiences.py",
                "--episodes",
                "2",
                "--fresh",
                "--parallel",
                "--num-envs",
                "2",
                "--db",
                str(tmp_path / "replay.db"),
                "--min-row-count",
                "128",
            ],
        )

        main()

        assert captured["kwargs"]["min_row_count"] == 128

    def test_replay_quality_preset_is_passed_to_generation(self, monkeypatch, tmp_path):
        captured = {}

        def fake_generation(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(
            "src.scripts.generate_experiences.generate_experiences",
            fake_generation,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_experiences.py",
                "--episodes",
                "1",
                "--fresh",
                "--db",
                str(tmp_path / "replay.db"),
                "--replay-quality-preset",
                "training",
                "--min-action-coverage-fraction",
                "0.5",
            ],
        )

        main()

        assert captured["kwargs"]["min_terminal_fraction"] == pytest.approx(0.005)
        assert captured["kwargs"]["min_immediate_terminal_fraction"] == pytest.approx(0.001)
        assert captured["kwargs"]["min_exact_mask_fraction"] == pytest.approx(0.8)
        assert captured["kwargs"]["min_boost_mask_fraction"] == pytest.approx(0.05)
        assert captured["kwargs"]["min_action_coverage_fraction"] == pytest.approx(0.5)
        assert captured["kwargs"]["min_positive_reward_fraction"] == pytest.approx(0.005)
        assert captured["kwargs"]["min_negative_reward_fraction"] == pytest.approx(0.005)
        assert captured["kwargs"]["min_multistep_fraction"] == pytest.approx(0.5)
        assert captured["kwargs"]["max_dominant_action_fraction"] == pytest.approx(0.75)
        assert captured["kwargs"]["max_invalid_current_action_fraction"] == pytest.approx(0.0)
        assert captured["kwargs"]["max_nonterminal_trapped_next_fraction"] == pytest.approx(0.05)
        assert captured["kwargs"]["max_exact_mask_state_mismatch_fraction"] == pytest.approx(0.0)
        assert captured["kwargs"]["max_malformed_state_feature_fraction"] == pytest.approx(0.0)
        assert captured["kwargs"]["replay_quality_preset"] == "training"

    def test_generation_defaults_to_trainable_replay_profile(self, monkeypatch, tmp_path):
        captured = {}

        def fake_generation(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(
            "src.scripts.generate_experiences.generate_experiences",
            fake_generation,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_experiences.py",
                "--episodes",
                "1",
                "--fresh",
                "--db",
                str(tmp_path / "replay.db"),
            ],
        )

        main()

        assert captured["kwargs"]["num_snakes"] == 6
        assert captured["kwargs"]["board_scale"] == pytest.approx(0.20)
        assert captured["kwargs"]["food_multiplier"] == pytest.approx(0.5)
        assert captured["kwargs"]["exploration_epsilon"] is None
        assert captured["kwargs"]["boost_exploration_rate"] == pytest.approx(0.25)
        assert captured["kwargs"]["danger_exploration_rate"] == pytest.approx(0.0)
        assert captured["kwargs"]["replay_quality_preset"] == "training"
        assert captured["kwargs"]["min_terminal_fraction"] == pytest.approx(0.005)
        assert captured["kwargs"]["min_immediate_terminal_fraction"] == pytest.approx(0.001)
        assert captured["kwargs"]["min_exact_mask_fraction"] == pytest.approx(0.8)
        assert captured["kwargs"]["min_boost_mask_fraction"] == pytest.approx(0.05)
        assert captured["kwargs"]["min_action_coverage_fraction"] == pytest.approx(1.0)
        assert captured["kwargs"]["min_positive_reward_fraction"] == pytest.approx(0.005)
        assert captured["kwargs"]["min_negative_reward_fraction"] == pytest.approx(0.005)
        assert captured["kwargs"]["min_multistep_fraction"] == pytest.approx(0.5)
        assert captured["kwargs"]["max_dominant_action_fraction"] == pytest.approx(0.75)
        assert captured["kwargs"]["max_invalid_current_action_fraction"] == pytest.approx(0.0)
        assert captured["kwargs"]["max_nonterminal_trapped_next_fraction"] == pytest.approx(0.05)
        assert captured["kwargs"]["max_exact_mask_state_mismatch_fraction"] == pytest.approx(0.0)
        assert captured["kwargs"]["max_malformed_state_feature_fraction"] == pytest.approx(0.0)

    def test_replay_environment_preset_is_passed_to_generation(self, monkeypatch, tmp_path):
        captured = {}

        def fake_generation(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(
            "src.scripts.generate_experiences.generate_experiences",
            fake_generation,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_experiences.py",
                "--episodes",
                "1",
                "--fresh",
                "--db",
                str(tmp_path / "replay.db"),
                "--replay-env-preset",
                "collision_dense",
            ],
        )

        main()

        assert captured["kwargs"]["num_snakes"] == 6
        assert captured["kwargs"]["board_scale"] == pytest.approx(0.20)
        assert captured["kwargs"]["food_multiplier"] == pytest.approx(0.5)
        assert captured["kwargs"]["exploration_epsilon"] is None
        assert captured["kwargs"]["boost_exploration_rate"] == pytest.approx(0.25)
        assert captured["kwargs"]["danger_exploration_rate"] == pytest.approx(0.0)

    def test_loaded_checkpoint_uses_model_guided_exploration_default(
        self,
        monkeypatch,
        tmp_path,
    ):
        captured = {}

        def fake_generation(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        checkpoint_path = tmp_path / "checkpoint.pth"
        monkeypatch.setattr(
            "src.scripts.generate_experiences.generate_experiences",
            fake_generation,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_experiences.py",
                "--episodes",
                "1",
                "--checkpoint",
                str(checkpoint_path),
                "--db",
                str(tmp_path / "replay.db"),
            ],
        )

        main()

        assert captured["kwargs"]["load_model"] is True
        assert captured["kwargs"]["checkpoint_path"] == str(checkpoint_path)
        assert captured["kwargs"]["exploration_epsilon"] is None
        assert captured["kwargs"]["exploration_min_epsilon"] is None

    def test_replay_environment_preset_allows_explicit_overrides(self, monkeypatch, tmp_path):
        captured = {}

        def fake_generation(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(
            "src.scripts.generate_experiences.generate_experiences",
            fake_generation,
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_experiences.py",
                "--episodes",
                "1",
                "--fresh",
                "--db",
                str(tmp_path / "replay.db"),
                "--replay-env-preset",
                "collision_dense",
                "--num-snakes",
                "2",
                "--board-scale",
                "0.5",
                "--food-multiplier",
                "1.25",
                "--danger-exploration-rate",
                "0.9",
                "--exploration-epsilon",
                "0.7",
            ],
        )

        main()

        assert captured["kwargs"]["num_snakes"] == 2
        assert captured["kwargs"]["board_scale"] == pytest.approx(0.5)
        assert captured["kwargs"]["food_multiplier"] == pytest.approx(1.25)
        assert captured["kwargs"]["exploration_epsilon"] == pytest.approx(0.7)
        assert captured["kwargs"]["boost_exploration_rate"] == pytest.approx(0.25)
        assert captured["kwargs"]["danger_exploration_rate"] == pytest.approx(0.9)


class TestGenerationFrameLimit:
    """Tests for bounded replay-generation episode lengths."""

    def test_default_frame_limit_uses_game_config(self):
        assert resolve_generation_frame_limit() == get_config().game.max_frames

    def test_explicit_frame_limit_overrides_config(self):
        assert resolve_generation_frame_limit(123) == 123

    def test_non_positive_frame_limit_raises(self):
        with pytest.raises(ValueError, match="max_frames"):
            resolve_generation_frame_limit(0)


class TestGenerationEnvironmentSettings:
    """Tests for generated replay environment-shaping controls."""

    def test_default_environment_preset_is_terminal_rich_shape(self):
        preset = resolve_generation_environment_preset()

        assert DEFAULT_GENERATION_ENV_PRESET == "collision_dense"
        assert preset == {
            "num_snakes": 6,
            "board_scale": pytest.approx(0.20),
            "food_multiplier": pytest.approx(0.5),
            "exploration_epsilon": None,
            "boost_exploration_rate": pytest.approx(0.25),
            "danger_exploration_rate": pytest.approx(0.0),
        }

    def test_named_default_environment_preset_preserves_configured_game_shape(self):
        preset = resolve_generation_environment_preset("default")

        assert preset == {
            "num_snakes": None,
            "board_scale": pytest.approx(1.0),
            "food_multiplier": pytest.approx(1.0),
            "exploration_epsilon": None,
            "boost_exploration_rate": pytest.approx(DEFAULT_BOOST_EXPLORATION_RATE),
            "danger_exploration_rate": pytest.approx(DEFAULT_DANGER_EXPLORATION_RATE),
        }

    def test_collision_dense_environment_preset_is_terminal_rich_shape(self):
        preset = resolve_generation_environment_preset("collision_dense")

        assert preset == {
            "num_snakes": 6,
            "board_scale": pytest.approx(0.20),
            "food_multiplier": pytest.approx(0.5),
            "exploration_epsilon": None,
            "boost_exploration_rate": pytest.approx(0.25),
            "danger_exploration_rate": pytest.approx(0.0),
        }

    def test_unknown_environment_preset_raises(self):
        with pytest.raises(ValueError, match="replay environment preset"):
            resolve_generation_environment_preset("missing")

    def test_default_environment_settings_follow_config(self):
        settings = resolve_generation_environment_settings()

        assert settings == {
            "num_snakes": get_config().game.num_snakes,
            "board_scale": pytest.approx(1.0),
            "food_multiplier": pytest.approx(1.0),
        }

    def test_explicit_environment_settings_are_preserved(self):
        settings = resolve_generation_environment_settings(
            num_snakes=2,
            board_scale=0.5,
            food_multiplier=0.25,
        )

        assert settings == {
            "num_snakes": 2,
            "board_scale": pytest.approx(0.5),
            "food_multiplier": pytest.approx(0.25),
        }

    def test_configured_num_snakes_is_used_when_not_overridden(self):
        original_config = get_config()
        custom_config = replace(
            original_config,
            game=replace(original_config.game, num_snakes=2),
        )

        try:
            initialize_config(custom_config)
            assert resolve_generation_num_snakes() == 2
        finally:
            initialize_config(original_config)

    @pytest.mark.parametrize("num_snakes", [0, -1])
    def test_invalid_num_snakes_raises(self, num_snakes):
        with pytest.raises(ValueError, match="num_snakes"):
            resolve_generation_num_snakes(num_snakes)

    def test_too_many_num_snakes_raises(self):
        with pytest.raises(ValueError, match="snake colors"):
            resolve_generation_num_snakes(999)

    @pytest.mark.parametrize("scale", [0, -0.1, float("nan")])
    def test_invalid_environment_scale_raises(self, scale):
        with pytest.raises(ValueError, match="board_scale"):
            resolve_generation_scale(scale, "board_scale")

    def test_board_scale_must_leave_playable_arena(self):
        with pytest.raises(ValueError, match="arena too small"):
            resolve_generation_environment_settings(board_scale=0.001)

    def test_food_multiplier_must_leave_food_available(self):
        with pytest.raises(ValueError, match="food_multiplier"):
            resolve_generation_environment_settings(food_multiplier=0.001)


class TestGenerationExploration:
    """Tests for generated replay exploration scheduling."""

    def test_actor_epsilons_use_apex_formula_without_floor(self):
        epsilons = compute_generation_actor_epsilons(
            num_actors=4,
            base_epsilon=0.4,
            alpha=7.0,
        )

        assert epsilons[0] == pytest.approx(0.4)
        assert epsilons[-1] == pytest.approx(0.4**8)

    def test_actor_epsilons_apply_minimum_floor(self):
        epsilons = compute_generation_actor_epsilons(
            num_actors=4,
            base_epsilon=0.4,
            alpha=7.0,
            min_epsilon=0.05,
        )

        assert min(epsilons) == pytest.approx(0.05)
        assert epsilons[0] == pytest.approx(0.4)

    def test_fresh_generation_defaults_min_epsilon_to_full_exploration(self):
        original_config = get_config()
        custom_config = replace(
            original_config,
            training=replace(original_config.training, epsilon_end=0.07),
        )

        try:
            initialize_config(custom_config)
            assert resolve_generation_min_epsilon(model_loaded=False) == pytest.approx(1.0)
        finally:
            initialize_config(original_config)

    def test_loaded_generation_defaults_min_epsilon_to_zero(self):
        assert resolve_generation_min_epsilon(model_loaded=True) == pytest.approx(0.0)

    def test_explicit_min_epsilon_overrides_model_status(self):
        assert resolve_generation_min_epsilon(model_loaded=True, min_epsilon=0.2) == pytest.approx(
            0.2
        )

    def test_non_finite_min_epsilon_raises(self):
        with pytest.raises(ValueError, match="min_epsilon"):
            resolve_generation_min_epsilon(model_loaded=False, min_epsilon=float("nan"))

    def test_generation_exploration_sets_boost_bias_on_snakes(self):
        from types import SimpleNamespace

        class PolicyStub:
            epsilon = 1.0
            use_gru = False

        snake = SimpleNamespace(
            actor_epsilon=None,
            current_epsilon=None,
            boost_exploration_rate=0.0,
            danger_exploration_rate=0.0,
            policy=PolicyStub(),
        )
        game_state = SimpleNamespace(snakes=[snake], _shared_policy=snake.policy)

        epsilon_min, epsilon_max = configure_generation_exploration(
            game_state,
            base_epsilon=0.4,
            min_epsilon=0.1,
            boost_exploration_rate=0.75,
            danger_exploration_rate=0.25,
        )

        assert epsilon_min == pytest.approx(0.4)
        assert epsilon_max == pytest.approx(0.4)
        assert snake.boost_exploration_rate == pytest.approx(0.75)
        assert snake.danger_exploration_rate == pytest.approx(0.25)

    def test_default_boost_exploration_rate_is_enabled_for_replay_generation(self):
        assert DEFAULT_BOOST_EXPLORATION_RATE == pytest.approx(0.25)

    def test_default_danger_exploration_rate_is_enabled_for_replay_generation(self):
        assert DEFAULT_DANGER_EXPLORATION_RATE == pytest.approx(0.02)


class TestGeneratedPriorityFallback:
    """Tests for untrained generated-replay priority enrichment."""

    def test_filter_keeps_trapped_nonterminal_targets_for_no_bootstrap_learning(self):
        def state_with_danger(danger_values, boost_available=1.0):
            state = [0.0] * 58
            state[1] = 1.0
            state[54:57] = [float(value) for value in danger_values]
            state[57] = float(boost_available)
            return state

        memories = [
            {
                "state": state_with_danger((0.0, 1.0, 1.0)),
                "action": 0,
                "next_action_mask": [True, False, False, False, False, False],
                "reward": 1.0,
                "done": False,
            },
            {
                "state": state_with_danger((1.0, 1.0, 1.0)),
                "action": 0,
                "next_action_mask": [True, False, False, False, False, False],
                "reward": 3.0,
                "done": False,
            },
            {
                "state": state_with_danger((0.0, 1.0, 1.0)),
                "action": 0,
                "next_action_mask": [False, False, False, False, False, False],
                "reward": 2.0,
                "done": False,
            },
            {
                "state": state_with_danger((1.0, 1.0, 1.0)),
                "action": 0,
                "next_action_mask": [False, False, False, False, False, False],
                "reward": -3.0,
                "done": True,
            },
            {
                "state": state_with_danger((0.0, 1.0, 1.0)),
                "action": 0,
                "reward": 0.5,
                "done": False,
            },
        ]

        filtered = filter_untrainable_generated_memories(memories)

        assert filtered == [memories[0], memories[2], memories[3], memories[4]]

    def test_flat_priorities_use_reward_and_terminal_signal(self):
        memories = [
            {"reward": 0.0, "done": False, "priority": 1.0},
            {"reward": 3.0, "done": False, "priority": 1.0},
            {"reward": -3.0, "done": True, "priority": 1.0},
        ]

        apply_generated_priority_fallback(memories)

        assert memories[2]["priority"] > memories[1]["priority"] > memories[0]["priority"]

    def test_existing_priority_spread_is_preserved(self):
        memories = [
            {"reward": 10.0, "done": True, "priority": 0.25},
            {"reward": 0.0, "done": False, "priority": 0.75},
        ]

        apply_generated_priority_fallback(memories)

        assert memories[0]["priority"] == pytest.approx(0.25)
        assert memories[1]["priority"] == pytest.approx(0.75)

    def test_flat_priorities_use_apex_priority_alpha(self):
        original_config = get_config()
        custom_config = replace(
            original_config,
            training=replace(original_config.training, priority_alpha=1.0),
            apex=replace(original_config.apex, priority_alpha=0.5, priority_epsilon=0.0),
        )
        memories = [{"reward": 3.0, "done": False, "priority": 1.0}]

        try:
            initialize_config(custom_config)
            apply_generated_priority_fallback(memories)
        finally:
            initialize_config(original_config)

        assert memories[0]["priority"] == pytest.approx(3.0**0.5)
