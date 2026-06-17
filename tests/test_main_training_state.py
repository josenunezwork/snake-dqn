"""Tests for headless training GameState setup helpers."""

import pytest

import src.main as main_module
from src.core.game_config import (
    AppConfig,
    CheckpointSettings,
    GameConfig,
    GameSettings,
    get_config,
    initialize_config,
)
from src.core.reward_contract import current_reward_contract
from src.data.memory_db_handler import MemoryDBHandler
from src.main import (
    apply_training_batch_size_override,
    collect_training_worker_failures,
    create_training_game_state,
    format_learning_health_smoke_report,
    get_best_env_model_path,
    get_checkpoint_path,
    get_episode_stats,
    get_training_game_settings,
    load_checkpoint_into_game_state,
    load_prefill_replay_rows,
    load_replay_db_into_game_state,
    resolve_checkpoint_path,
    resolve_prefill_replay_quality_gates,
    resolve_replay_quality_fraction,
    resolve_training_batch_size,
    run_learning_health_smoke,
    save_final_training_checkpoints,
    save_training_checkpoint,
    train_environment,
    validate_headless_checkpoint_contract,
    validate_learning_health_smoke,
)
from src.training.curriculum import CurriculumManager


class FakePolicy:
    """Small stand-in for shared-policy construction tests."""

    def __init__(self):
        self.epsilon = 0.75
        self.training = True


class TestTrainingGameStateSetup:
    """Tests for applying curriculum settings before training starts."""

    def test_curriculum_phase_zero_settings_create_solo_easy_game_state(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=800,
                height=600,
                num_snakes=4,
                initial_food=2,
                max_food=10,
            )
        )

        try:
            initialize_config(config)
            curriculum = CurriculumManager()

            game_state = create_training_game_state(
                curriculum=curriculum,
                shared_policy=FakePolicy(),
            )

            assert game_state.num_snakes == 1
            assert len(game_state.snakes) == 1
            assert game_state._game_width == 400
            assert game_state._game_height == 300
            assert game_state.food_manager.max_food == 20
            assert game_state._effective_initial_food == 4
            assert game_state.snakes[0].game_width == 400
            assert game_state.snakes[0].game_height == 300
        finally:
            initialize_config(original_config)

    def test_resumed_curriculum_phase_settings_are_used(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=800,
                height=600,
                num_snakes=4,
                initial_food=2,
                max_food=10,
            )
        )

        try:
            initialize_config(config)
            curriculum = CurriculumManager()
            curriculum.current_phase_idx = 2

            game_state = create_training_game_state(
                curriculum=curriculum,
                shared_policy=FakePolicy(),
            )

            assert game_state.num_snakes == 2
            assert len(game_state.snakes) == 2
            assert game_state._game_width == 800
            assert game_state._game_height == 600
            assert game_state.snakes[0].policy is game_state.snakes[1].policy
        finally:
            initialize_config(original_config)

    def test_without_curriculum_settings_use_game_config_defaults(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=800,
                height=600,
                num_snakes=3,
                initial_food=2,
                max_food=10,
            )
        )

        try:
            initialize_config(config)
            settings = get_training_game_settings()
            game_state = create_training_game_state(shared_policy=FakePolicy())

            assert settings["num_snakes"] == 3
            assert settings["food_multiplier"] == 1.0
            assert settings["board_scale"] == 1.0
            assert game_state.num_snakes == 3
            assert len(game_state.snakes) == 3
            assert game_state._game_width == 800
            assert game_state._game_height == 600
        finally:
            initialize_config(original_config)

    def test_eval_game_state_forces_shared_policy_to_greedy_inference(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=800,
                height=600,
                num_snakes=2,
                initial_food=2,
            )
        )
        policy = FakePolicy()

        try:
            initialize_config(config)

            game_state = create_training_game_state(
                shared_policy=policy,
                eval_mode=True,
            )

            assert policy.epsilon == pytest.approx(0.0)
            assert policy.training is False
            assert all(getattr(snake, "actor_epsilon", None) == 0.0 for snake in game_state.snakes)
            assert all(
                getattr(snake, "current_epsilon", None) == 0.0 for snake in game_state.snakes
            )
        finally:
            initialize_config(original_config)


class TestTrainingBatchSizeOverride:
    """Tests for optional main training batch-size overrides."""

    def test_missing_batch_size_leaves_active_config_unchanged(self):
        assert resolve_training_batch_size(None) is None
        assert apply_training_batch_size_override(None) == GameConfig.BATCH_SIZE

    def test_batch_size_override_updates_training_config(self):
        original_config = get_config()
        try:
            applied = apply_training_batch_size_override(32)

            assert applied == 32
            assert GameConfig.BATCH_SIZE == 32
            assert get_config().training.batch_size == 32
        finally:
            initialize_config(original_config)

    @pytest.mark.parametrize("batch_size", [0, -1, True])
    def test_invalid_batch_size_raises(self, batch_size):
        with pytest.raises(ValueError, match="batch-size"):
            resolve_training_batch_size(batch_size)


class TestLearningHealthSmoke:
    """Tests for bounded in-process learning-health smoke helpers."""

    class MemoryStub:
        def __init__(self):
            self.size = 0

        def __len__(self):
            return self.size

    class PolicyStub:
        epsilon = 0.5
        training = True

        def __init__(self):
            self.memory = TestLearningHealthSmoke.MemoryStub()
            self._losses = []
            self._last_train_metrics = {}
            self.update_counter = 0

        def _min_replay_size(self):
            return 4

    class SnakeStub:
        total_reward = 1.5

        def __init__(self):
            self.saved_paths = []

        def save_state(self, path):
            self.saved_paths.append(path)
            with open(path, "w", encoding="utf-8") as checkpoint:
                checkpoint.write("smoke checkpoint")

    class GameStateStub:
        def __init__(self):
            self._shared_policy = TestLearningHealthSmoke.PolicyStub()
            self.snakes = [TestLearningHealthSmoke.SnakeStub()]
            self.alive_snakes = 1
            self.episode_best_length = 1
            self.episode_collision_counts = {"wall": 1, "self": 0, "head": 0, "body": 0}
            self.episode_deaths = 0
            self.episode_food_eaten = 0
            self.episode_kills = 0
            self.episode_best_reward = 0.0
            self.flush_called = False
            self.full_cleanup_called = False
            self.update_calls = 0
            self.learn_calls = []

        def update(self, train_mode=True, learn=True):
            assert train_mode is True
            self.learn_calls.append(learn)
            self.update_calls += 1
            self._shared_policy.memory.size += 2
            if self.update_calls >= 2:
                self.alive_snakes = 0
                self.episode_deaths = 1
                self.episode_food_eaten = 1
                self.episode_best_reward = 1.5
                self._shared_policy.update_counter = 1 if learn else 0
                self._shared_policy._losses = [0.25]
                self._shared_policy._last_train_metrics = {
                    "valid_next_action_fraction": 1.0,
                    "trapped_next_state_fraction": 0.0,
                    "exact_next_action_mask_fraction": 0.5,
                }

        def flush_episode_experience(self):
            self.flush_called = True

        def full_cleanup(self):
            self.full_cleanup_called = True

    def test_learning_health_smoke_reports_replay_updates_and_checkpoint(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))
        game_state = self.GameStateStub()

        try:
            initialize_config(config)

            stats = run_learning_health_smoke(
                max_frames=5,
                checkpoint_filename="smoke.pth",
                game_state_factory=lambda: game_state,
            )

            assert stats["frames"] == 2
            assert stats["terminated"] is True
            assert stats["replay_ready"] is True
            assert stats["updates_ran"] is True
            assert stats["update_delta"] == 1
            assert stats["loss_available"] is True
            assert stats["policy"]["valid_next_action_fraction"] == pytest.approx(1.0)
            assert stats["policy"]["trapped_next_state_fraction"] == pytest.approx(0.0)
            assert stats["policy"]["exact_next_action_mask_fraction"] == pytest.approx(0.5)
            assert stats["checkpoint"] == str(checkpoint_dir / "smoke.pth")
            assert (checkpoint_dir / "smoke.pth").read_text(encoding="utf-8") == (
                "smoke checkpoint"
            )
            assert game_state.flush_called is True
            assert game_state.full_cleanup_called is True
        finally:
            initialize_config(original_config)

    def test_learning_health_smoke_eval_mode_is_greedy_and_does_not_learn(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))
        game_state = self.GameStateStub()

        try:
            initialize_config(config)

            stats = run_learning_health_smoke(
                max_frames=5,
                checkpoint_filename="eval_smoke.pth",
                eval_mode=True,
                game_state_factory=lambda: game_state,
            )

            assert game_state.learn_calls == [False, False]
            assert game_state._shared_policy.epsilon == pytest.approx(0.0)
            assert game_state._shared_policy.training is False
            assert stats["eval_mode"] is True
            assert stats["updates_ran"] is False
            assert stats["update_delta"] == 0
        finally:
            initialize_config(original_config)

    def test_learning_health_smoke_loads_checkpoint_before_running(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))
        game_state = self.GameStateStub()
        loaded_paths = []

        def checkpoint_loader(loaded_game_state, checkpoint_path):
            assert loaded_game_state is game_state
            loaded_paths.append(checkpoint_path)
            return True

        try:
            initialize_config(config)

            stats = run_learning_health_smoke(
                max_frames=5,
                checkpoint_filename="smoke.pth",
                checkpoint_path="/tmp/offline_smoke.pth",
                game_state_factory=lambda: game_state,
                checkpoint_loader=checkpoint_loader,
            )

            assert loaded_paths == ["/tmp/offline_smoke.pth"]
            assert stats["checkpoint_loaded"] is True
            assert stats["loaded_checkpoint"] == "/tmp/offline_smoke.pth"
            assert stats["update_delta"] == 1
            assert game_state.update_calls == 2
            assert game_state.full_cleanup_called is True
        finally:
            initialize_config(original_config)

    def test_learning_health_smoke_eval_loads_checkpoint_without_training_contract(
        self, tmp_path, monkeypatch
    ):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))
        game_state = self.GameStateStub()
        load_calls = []

        def checkpoint_loader(loaded_game_state, checkpoint_path, strict_training_contract=True):
            assert loaded_game_state is game_state
            load_calls.append((checkpoint_path, strict_training_contract))
            return True

        try:
            initialize_config(config)
            monkeypatch.setattr(main_module, "load_checkpoint_into_game_state", checkpoint_loader)

            stats = run_learning_health_smoke(
                max_frames=5,
                checkpoint_filename="eval_smoke.pth",
                checkpoint_path="/tmp/inference_only.pth",
                eval_mode=True,
                game_state_factory=lambda: game_state,
            )

            assert load_calls == [("/tmp/inference_only.pth", False)]
            assert stats["checkpoint_loaded"] is True
            assert stats["updates_ran"] is False
        finally:
            initialize_config(original_config)

    def test_learning_health_smoke_prefills_replay_before_running(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))
        game_state = self.GameStateStub()
        replay_calls = []

        def replay_loader(loaded_game_state, replay_db_path, **kwargs):
            assert loaded_game_state is game_state
            replay_calls.append((replay_db_path, kwargs))
            game_state._shared_policy.memory.size = 8
            return 8

        try:
            initialize_config(config)

            stats = run_learning_health_smoke(
                max_frames=5,
                checkpoint_filename="smoke.pth",
                replay_db_path="/tmp/replay.db",
                replay_order="id",
                min_terminal_fraction=0.1,
                min_exact_mask_fraction=0.9,
                game_state_factory=lambda: game_state,
                replay_loader=replay_loader,
            )

            expected_gates = resolve_prefill_replay_quality_gates(
                overrides={
                    "min_terminal_fraction": 0.1,
                    "min_exact_mask_fraction": 0.9,
                }
            )
            assert replay_calls == [
                (
                    "/tmp/replay.db",
                    {
                        "replay_order": "id",
                        **expected_gates,
                    },
                )
            ]
            assert stats["replay_db"] == "/tmp/replay.db"
            assert stats["replay_rows_loaded"] == 8
            assert stats["replay_ready"] is True
            assert game_state.full_cleanup_called is True
        finally:
            initialize_config(original_config)

    def test_learning_health_smoke_does_not_count_loaded_checkpoint_updates_as_new(self):
        class PolicyStub:
            epsilon = 0.25
            training = True

            def __init__(self):
                self.memory = TestLearningHealthSmoke.MemoryStub()
                self._losses = []
                self._last_train_metrics = {}
                self.update_counter = 8

            def _min_replay_size(self):
                return 4

        class GameStateStub:
            episode_best_length = 1
            episode_collision_counts = {"wall": 0, "self": 0, "head": 0, "body": 0}
            episode_deaths = 0
            episode_food_eaten = 0
            episode_kills = 0
            episode_best_reward = 0.0

            def __init__(self):
                self._shared_policy = PolicyStub()
                self.snakes = [TestLearningHealthSmoke.SnakeStub()]
                self.alive_snakes = 1
                self.full_cleanup_called = False

            def update(self, train_mode=True):
                assert train_mode is True
                self._shared_policy.memory.size += 1
                self.alive_snakes = 0

            def flush_episode_experience(self):
                pass

            def full_cleanup(self):
                self.full_cleanup_called = True

        game_state = GameStateStub()

        stats = run_learning_health_smoke(
            max_frames=5,
            checkpoint_filename="/tmp/loaded_update_delta_smoke.pth",
            checkpoint_path="/tmp/offline_smoke.pth",
            game_state_factory=lambda: game_state,
            checkpoint_loader=lambda _game_state, _checkpoint_path: True,
        )

        assert stats["policy"]["updates"] == 8
        assert stats["update_delta"] == 0
        assert stats["updates_ran"] is False
        assert game_state.full_cleanup_called is True

    def test_learning_health_smoke_raises_when_checkpoint_load_fails(self):
        game_state = self.GameStateStub()

        with pytest.raises(RuntimeError, match="Could not load health smoke checkpoint"):
            run_learning_health_smoke(
                max_frames=5,
                checkpoint_path="/tmp/missing.pth",
                game_state_factory=lambda: game_state,
                checkpoint_loader=lambda _game_state, _checkpoint_path: False,
            )

        assert game_state.update_calls == 0
        assert game_state.full_cleanup_called is True

    def test_episode_stats_report_current_reward_not_best_moment(self):
        """Episode reward summaries should expose final negative outcomes."""

        class GameStateStatsStub:
            episode_best_length = 3
            episode_collision_counts = {"wall": 1}
            episode_deaths = 1
            episode_food_eaten = 0
            episode_kills = 0
            episode_current_reward = -2.5
            episode_best_reward = 0.25

        stats = get_episode_stats(GameStateStatsStub(), episode_length=12)

        assert stats["reward"] == -2.5
        assert stats["best_reward"] == 0.25

    def test_learning_health_smoke_rejects_invalid_frame_count(self):
        with pytest.raises(ValueError, match="frames must be positive"):
            run_learning_health_smoke(max_frames=0)

    def test_learning_health_smoke_report_formats_key_metrics(self):
        stats = {
            "checkpoint": "/tmp/smoke.pth",
            "elapsed_seconds": 0.5,
            "episode": {
                "deaths": 1,
                "food_eaten": 2,
                "kills": 0,
                "length": 5,
                "reward": 1.25,
            },
            "frames": 5,
            "loss_available": True,
            "max_frames": 10,
            "policy": {
                "epsilon": 0.5,
                "last_loss": 0.25,
                "min_replay_size": 4,
                "replay_size": 8,
                "updates": 2,
                "valid_next_action_fraction": 0.75,
                "trapped_next_state_fraction": 0.25,
                "exact_next_action_mask_fraction": 0.5,
            },
            "replay_ready": True,
            "replay_db": None,
            "replay_rows_loaded": 0,
            "terminated": False,
            "update_delta": 2,
            "updates_ran": True,
        }

        report = format_learning_health_smoke_report(stats)

        assert "Loaded checkpoint: none" in report
        assert "Replay prefill: none" in report
        assert "Frames: 5/10 (frame cap reached)" in report
        assert "Replay: 8/4 (ready)" in report
        assert "Updates: 2 (ran: yes, delta: 2)" in report
        assert "Loss: 0.2500" in report
        assert "Target actions: valid=75.0%, trapped=25.0%, exact_masks=50.0%" in report
        assert "Checkpoint: /tmp/smoke.pth" in report

    def test_learning_health_smoke_report_includes_loaded_checkpoint(self):
        stats = {
            "checkpoint": "/tmp/out.pth",
            "checkpoint_loaded": True,
            "elapsed_seconds": 0.5,
            "episode": {
                "deaths": 0,
                "food_eaten": 0,
                "kills": 0,
                "length": 3,
                "reward": 0.0,
            },
            "frames": 3,
            "loaded_checkpoint": "/tmp/in.pth",
            "max_frames": 3,
            "policy": {
                "epsilon": 0.0,
                "last_loss": None,
                "min_replay_size": 4,
                "replay_size": 2,
                "updates": 0,
            },
            "replay_ready": False,
            "replay_db": "/tmp/replay.db",
            "replay_rows_loaded": 12,
            "terminated": False,
            "update_delta": 0,
            "updates_ran": False,
        }

        report = format_learning_health_smoke_report(stats)

        assert "Loaded checkpoint: /tmp/in.pth (loaded)" in report
        assert "Replay prefill: /tmp/replay.db (12 rows)" in report
        assert "Updates: 0 (ran: no, delta: 0)" in report

    def test_learning_health_smoke_validation_requires_training_updates(self):
        stats = {
            "policy": {
                "min_replay_size": 4,
                "replay_size": 2,
            },
            "update_delta": 0,
            "updates_ran": False,
        }

        with pytest.raises(RuntimeError, match="did not run any training updates"):
            validate_learning_health_smoke(stats)

    def test_learning_health_smoke_validation_allows_training_updates(self):
        validate_learning_health_smoke({"updates_ran": True})


class TestTrainingWorkerFailures:
    """Tests for detecting failed headless training workers."""

    class FakeProcess:
        """Minimal process stand-in exposing the exitcode contract."""

        def __init__(self, exitcode):
            self.exitcode = exitcode

    def test_no_failures_when_all_workers_finish_and_report_success(self):
        failures = collect_training_worker_failures(
            [(0, self.FakeProcess(0)), (1, self.FakeProcess(0))],
            [(0, 5), (1, 5)],
            {
                0: {"best_reward": 1.0, "total_frames": 10, "error": None},
                1: {"best_reward": 2.0, "total_frames": 10, "error": None},
            },
        )

        assert failures == []

    def test_reports_nonzero_exitcode(self):
        failures = collect_training_worker_failures(
            [(0, self.FakeProcess(9))],
            [(0, 5)],
            {0: {"best_reward": 0.0, "total_frames": 0, "error": None}},
        )

        assert failures == ["env 0 exited with code 9"]

    def test_reports_missing_stats(self):
        failures = collect_training_worker_failures(
            [(1, self.FakeProcess(0))],
            [(1, 5)],
            {},
        )

        assert failures == ["env 1 did not report training stats"]

    def test_reports_worker_error_message(self):
        failures = collect_training_worker_failures(
            [(2, self.FakeProcess(0))],
            [(2, 5)],
            {2: {"best_reward": float("-inf"), "total_frames": 0, "error": "boom"}},
        )

        assert failures == ["env 2 failed: boom"]

    def test_reports_unfinished_process(self):
        failures = collect_training_worker_failures(
            [(3, self.FakeProcess(None))],
            [(3, 5)],
            {3: {"best_reward": 0.0, "total_frames": 0, "error": None}},
        )

        assert failures == ["env 3 did not finish"]


class TestHeadlessCheckpointLoad:
    """Tests for headless checkpoint compatibility checks."""

    class PolicyStub:
        device = "cpu"
        input_size = GameConfig.INPUT_SIZE
        hidden_size = GameConfig.HIDDEN_SIZE
        output_size = GameConfig.OUTPUT_SIZE
        n_step = GameConfig.APEX_N_STEP
        gamma = GameConfig.APEX_GAMMA
        use_gru = False
        epsilon = 0.1

        def __init__(self):
            self.loaded_checkpoint = None

        def load_state_dict(self, checkpoint):
            self.loaded_checkpoint = checkpoint

    class MemoryStub:
        n_step = GameConfig.APEX_N_STEP

        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    class GameStateStub:
        def __init__(self):
            self._shared_policy = TestHeadlessCheckpointLoad.PolicyStub()
            self.snakes = []

    def test_headless_checkpoint_contract_accepts_matching_config(self):
        checkpoint = {
            "apex_config": {
                "input_size": GameConfig.INPUT_SIZE,
                "hidden_size": GameConfig.HIDDEN_SIZE,
                "output_size": GameConfig.OUTPUT_SIZE,
                "gamma": GameConfig.APEX_GAMMA,
                "n_step": GameConfig.APEX_N_STEP,
                "reward_contract": current_reward_contract(),
                "reward_death": GameConfig.REWARD_DEATH,
                "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                "use_gru": False,
            }
        }

        validate_headless_checkpoint_contract(
            checkpoint,
            self.PolicyStub(),
            checkpoint_path="checkpoint.pth",
        )

    def test_checkpoint_load_rejects_n_step_mismatch_before_policy_mutation(self, tmp_path):
        import torch

        checkpoint = {
            "apex_config": {
                "input_size": GameConfig.INPUT_SIZE,
                "hidden_size": GameConfig.HIDDEN_SIZE,
                "output_size": GameConfig.OUTPUT_SIZE,
                "gamma": GameConfig.APEX_GAMMA,
                "n_step": GameConfig.APEX_N_STEP + 1,
                "reward_contract": current_reward_contract(),
                "reward_death": GameConfig.REWARD_DEATH,
                "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                "use_gru": False,
            }
        }
        checkpoint_path = tmp_path / "bad_n_step.pth"
        torch.save(checkpoint, checkpoint_path)
        game_state = self.GameStateStub()

        assert load_checkpoint_into_game_state(game_state, str(checkpoint_path)) is False
        assert game_state._shared_policy.loaded_checkpoint is None

    def test_inference_checkpoint_load_allows_td_contract_mismatch(self, tmp_path):
        import torch

        checkpoint = {
            "apex_config": {
                "input_size": GameConfig.INPUT_SIZE,
                "hidden_size": GameConfig.HIDDEN_SIZE,
                "output_size": GameConfig.OUTPUT_SIZE,
                "gamma": GameConfig.APEX_GAMMA + 0.01,
                "n_step": GameConfig.APEX_N_STEP + 1,
                "reward_contract": current_reward_contract(),
                "reward_death": GameConfig.REWARD_DEATH,
                "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                "use_gru": False,
            }
        }
        checkpoint_path = tmp_path / "inference_contract_drift.pth"
        torch.save(checkpoint, checkpoint_path)
        game_state = self.GameStateStub()

        assert (
            load_checkpoint_into_game_state(
                game_state,
                str(checkpoint_path),
                strict_training_contract=False,
            )
            is True
        )
        assert game_state._shared_policy.loaded_checkpoint["apex_config"]["n_step"] == (
            GameConfig.APEX_N_STEP + 1
        )

    def test_checkpoint_load_rejects_missing_reward_contract_before_policy_mutation(self, tmp_path):
        import torch

        checkpoint = {
            "apex_config": {
                "input_size": GameConfig.INPUT_SIZE,
                "hidden_size": GameConfig.HIDDEN_SIZE,
                "output_size": GameConfig.OUTPUT_SIZE,
                "gamma": GameConfig.APEX_GAMMA,
                "n_step": GameConfig.APEX_N_STEP,
                "reward_death": GameConfig.REWARD_DEATH,
                "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                "use_gru": False,
            }
        }
        checkpoint_path = tmp_path / "legacy_reward.pth"
        torch.save(checkpoint, checkpoint_path)
        game_state = self.GameStateStub()

        assert load_checkpoint_into_game_state(game_state, str(checkpoint_path)) is False
        assert game_state._shared_policy.loaded_checkpoint is None

    def test_checkpoint_load_rejects_reward_mismatch_before_policy_mutation(self, tmp_path):
        import torch

        checkpoint = {
            "apex_config": {
                "input_size": GameConfig.INPUT_SIZE,
                "hidden_size": GameConfig.HIDDEN_SIZE,
                "output_size": GameConfig.OUTPUT_SIZE,
                "gamma": GameConfig.APEX_GAMMA,
                "n_step": GameConfig.APEX_N_STEP,
                "reward_contract": current_reward_contract(),
                "reward_death": -3.0,
                "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                "use_gru": False,
            }
        }
        checkpoint_path = tmp_path / "stale_reward.pth"
        torch.save(checkpoint, checkpoint_path)
        game_state = self.GameStateStub()

        assert load_checkpoint_into_game_state(game_state, str(checkpoint_path)) is False
        assert game_state._shared_policy.loaded_checkpoint is None

    def test_checkpoint_load_rejects_full_reward_mismatch_before_policy_mutation(self, tmp_path):
        import torch

        stale_contract = current_reward_contract()
        stale_contract["survival"] = float(stale_contract["survival"]) + 1.0
        checkpoint = {
            "apex_config": {
                "input_size": GameConfig.INPUT_SIZE,
                "hidden_size": GameConfig.HIDDEN_SIZE,
                "output_size": GameConfig.OUTPUT_SIZE,
                "gamma": GameConfig.APEX_GAMMA,
                "n_step": GameConfig.APEX_N_STEP,
                "reward_contract": stale_contract,
                "reward_death": GameConfig.REWARD_DEATH,
                "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                "use_gru": False,
            }
        }
        checkpoint_path = tmp_path / "stale_reward_contract.pth"
        torch.save(checkpoint, checkpoint_path)
        game_state = self.GameStateStub()

        assert load_checkpoint_into_game_state(game_state, str(checkpoint_path)) is False
        assert game_state._shared_policy.loaded_checkpoint is None

    def test_checkpoint_load_rejects_malformed_replay_restore(self, tmp_path):
        import torch

        checkpoint = {
            "apex_config": {
                "input_size": GameConfig.INPUT_SIZE,
                "hidden_size": GameConfig.HIDDEN_SIZE,
                "output_size": GameConfig.OUTPUT_SIZE,
                "gamma": GameConfig.APEX_GAMMA,
                "n_step": GameConfig.APEX_N_STEP,
                "reward_contract": current_reward_contract(),
                "reward_death": GameConfig.REWARD_DEATH,
                "reward_food_base": GameConfig.REWARD_FOOD_BASE,
                "use_gru": False,
            },
            "memories": [("too", "short")],
        }
        checkpoint_path = tmp_path / "bad_replay.pth"
        torch.save(checkpoint, checkpoint_path)
        game_state = self.GameStateStub()
        game_state._shared_policy.memory = self.MemoryStub()

        assert load_checkpoint_into_game_state(game_state, str(checkpoint_path)) is False
        assert game_state._shared_policy.loaded_checkpoint is not None
        assert game_state._shared_policy.memory.cleared is False

    class WorkerPolicyStub:
        epsilon = 0.0
        memory = []
        training = True
        update_counter = 0
        _losses = []

        def _min_replay_size(self):
            return 1

    class WorkerSnakeStub:
        total_reward = 0.0

        def __init__(self):
            self.ai = type("AIStub", (), {"device": "cpu"})()
            self.saved_paths = []

        def save_state(self, path):
            self.saved_paths.append(path)
            with open(path, "w", encoding="utf-8") as checkpoint:
                checkpoint.write("unexpected random restart checkpoint")

    class WorkerGameStateStub:
        def __init__(self):
            self._shared_policy = TestHeadlessCheckpointLoad.WorkerPolicyStub()
            self.snakes = [TestHeadlessCheckpointLoad.WorkerSnakeStub()]
            self.cleaned_up = False

        def full_cleanup(self):
            self.cleaned_up = True

    def test_training_worker_fails_fast_when_checkpoint_load_returns_false(
        self, tmp_path, monkeypatch
    ):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))
        game_state = self.WorkerGameStateStub()
        requested_checkpoint = tmp_path / "resume.pth"
        return_dict = {}

        monkeypatch.setattr(
            main_module,
            "create_training_game_state",
            lambda curriculum, eval_mode=False: game_state,
        )
        monkeypatch.setattr(
            main_module,
            "load_checkpoint_into_game_state",
            lambda gs, path, strict_training_contract=True: False,
        )

        try:
            initialize_config(config)

            train_environment(
                env_id=0,
                episodes=0,
                save_interval=100,
                return_dict=return_dict,
                use_tensorboard=False,
                checkpoint_path=str(requested_checkpoint),
            )
        finally:
            initialize_config(original_config)

        assert "Could not load headless training checkpoint" in return_dict[0]["error"]
        assert return_dict[0]["best_reward"] == float("-inf")
        assert game_state.snakes[0].saved_paths == []
        assert not (checkpoint_dir / "env_0_final_snake.pth").exists()


class TestReplayPrefill:
    """Tests for generated replay prefill loading."""

    class FakeDB:
        def __init__(self):
            self.calls = []

        def load_memories_for_policy(self, policy_type, limit, order_by):
            self.calls.append((policy_type, limit, order_by))
            return ([], [], [], [], [], [], [])

    def test_prefill_loader_uses_uniform_id_by_default(self):
        db = self.FakeDB()

        result = load_prefill_replay_rows(db, limit=123)

        assert result == ([], [], [], [], [], [], [])
        assert db.calls == [("apex", 123, "id_uniform")]

    def test_prefill_loader_allows_priority_order(self):
        db = self.FakeDB()

        load_prefill_replay_rows(db, limit=456, replay_order="priority")

        assert db.calls == [("apex", 456, "priority")]

    def test_prefill_loader_requests_action_masks_when_supported(self):
        class FakeDB:
            def __init__(self):
                self.calls = []

            def load_memories_for_policy(
                self, policy_type, limit, order_by, include_action_masks=False
            ):
                self.calls.append((policy_type, limit, order_by, include_action_masks))
                return ([], [], [], [], [], [], [], [])

        db = FakeDB()

        result = load_prefill_replay_rows(db, limit=789)

        assert result == ([], [], [], [], [], [], [], [])
        assert db.calls == [("apex", 789, "id_uniform", True)]

    def test_prefill_loader_requests_snake_ids_when_supported(self):
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
                    (
                        policy_type,
                        limit,
                        order_by,
                        include_action_masks,
                        include_snake_ids,
                    )
                )
                return ([], [], [], [], [], [], [], [], [])

        db = FakeDB()

        result = load_prefill_replay_rows(db, limit=789)

        assert result == ([], [], [], [], [], [], [], [], [])
        assert db.calls == [("apex", 789, "id_uniform", True, True)]

    def test_replay_db_prefill_passes_next_action_masks_to_memory(self, tmp_path):
        class MemoryStub:
            capacity = 10

            def __init__(self):
                self.add_bulk_kwargs = None

            def add_bulk(self, *args, **kwargs):
                self.add_bulk_kwargs = kwargs

        class PolicyStub:
            use_gru = False

            def __init__(self):
                self.memory = MemoryStub()

        class GameStateStub:
            def __init__(self):
                self._shared_policy = PolicyStub()
                self.snakes = []

        db_path = tmp_path / "replay.db"
        mask = [False, True, False, False, False, False]
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.save_memories(
                snake_id=4,
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
        game_state = GameStateStub()

        loaded = load_replay_db_into_game_state(game_state, str(db_path), limit=None)

        assert loaded == 1
        memory = game_state._shared_policy.memory
        assert memory.add_bulk_kwargs["next_action_masks"] == [tuple(mask)]
        assert memory.add_bulk_kwargs["stream_ids"] == [4]

    def test_replay_db_prefill_validates_against_policy_n_step(self, tmp_path):
        class MemoryStub:
            capacity = 10

            def __init__(self):
                self.add_bulk_kwargs = None

            def add_bulk(self, *args, **kwargs):
                self.add_bulk_kwargs = kwargs

        class PolicyStub:
            use_gru = False
            gamma = GameConfig.APEX_GAMMA
            n_step = GameConfig.APEX_N_STEP + 1

            def __init__(self):
                self.memory = MemoryStub()

        class GameStateStub:
            def __init__(self):
                self._shared_policy = PolicyStub()
                self.snakes = []

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
                        "bootstrap_steps": 2,
                    }
                ],
            )
        finally:
            handler.close()
        game_state = GameStateStub()

        loaded = load_replay_db_into_game_state(game_state, str(db_path), limit=None)

        assert loaded == 1
        assert game_state._shared_policy.memory.add_bulk_kwargs is not None

    def test_replay_db_prefill_reports_quality(self, tmp_path, capsys):
        class MemoryStub:
            capacity = 10

            def add_bulk(self, *args, **kwargs):
                pass

        class PolicyStub:
            use_gru = False

            def __init__(self):
                self.memory = MemoryStub()

        class GameStateStub:
            def __init__(self):
                self._shared_policy = PolicyStub()
                self.snakes = []

        db_path = tmp_path / "replay.db"
        mask = [False, True, False, False, False, False]
        handler = MemoryDBHandler(str(db_path))
        try:
            for snake_id, row_count in ((0, 1), (1, 2)):
                handler.save_memories(
                    snake_id=snake_id,
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
                        for _ in range(row_count)
                    ],
                )
        finally:
            handler.close()

        load_replay_db_into_game_state(GameStateStub(), str(db_path), limit=None)

        output = capsys.readouterr().out
        assert "Replay prefill quality:" in output
        assert "Nonterminal exact masks: 3/3" in output
        assert "Rows per snake_id min/avg/max: 1/1.50/2" in output

    def test_replay_db_prefill_empty_apex_rows_raise_before_memory_mutation(self, tmp_path):
        class MemoryStub:
            capacity = 10

            def __init__(self):
                self.add_bulk_kwargs = None

            def add_bulk(self, *args, **kwargs):
                self.add_bulk_kwargs = kwargs

        class PolicyStub:
            use_gru = False

            def __init__(self):
                self.memory = MemoryStub()

        class GameStateStub:
            def __init__(self):
                self._shared_policy = PolicyStub()
                self.snakes = []

        db_path = tmp_path / "empty_replay.db"
        handler = MemoryDBHandler(str(db_path))
        handler.close()
        game_state = GameStateStub()

        with pytest.raises(RuntimeError, match="No Apex replay rows"):
            load_replay_db_into_game_state(game_state, str(db_path), limit=None)

        assert game_state._shared_policy.memory.add_bulk_kwargs is None

    def test_replay_db_prefill_terminal_gate_raises_before_memory_mutation(self, tmp_path):
        class MemoryStub:
            capacity = 10

            def __init__(self):
                self.add_bulk_kwargs = None

            def add_bulk(self, *args, **kwargs):
                self.add_bulk_kwargs = kwargs

        class PolicyStub:
            use_gru = False

            def __init__(self):
                self.memory = MemoryStub()

        class GameStateStub:
            def __init__(self):
                self._shared_policy = PolicyStub()
                self.snakes = []

        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 0.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                        "next_action_mask": [False, True, False, False, False, False],
                    }
                ],
            )
        finally:
            handler.close()
        game_state = GameStateStub()

        with pytest.raises(RuntimeError, match="terminal fraction"):
            load_replay_db_into_game_state(
                game_state,
                str(db_path),
                limit=None,
                min_terminal_fraction=0.01,
            )

        assert game_state._shared_policy.memory.add_bulk_kwargs is None

    def test_replay_db_prefill_exact_mask_gate_raises_before_memory_mutation(self, tmp_path):
        class MemoryStub:
            capacity = 10

            def __init__(self):
                self.add_bulk_kwargs = None

            def add_bulk(self, *args, **kwargs):
                self.add_bulk_kwargs = kwargs

        class PolicyStub:
            use_gru = False

            def __init__(self):
                self.memory = MemoryStub()

        class GameStateStub:
            def __init__(self):
                self._shared_policy = PolicyStub()
                self.snakes = []

        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 0.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                ],
            )
        finally:
            handler.close()
        game_state = GameStateStub()

        with pytest.raises(RuntimeError, match="exact-mask fraction"):
            load_replay_db_into_game_state(
                game_state,
                str(db_path),
                limit=None,
                min_exact_mask_fraction=1.0,
            )

        assert game_state._shared_policy.memory.add_bulk_kwargs is None

    def test_replay_db_prefill_action_coverage_gate_raises_before_memory_mutation(self, tmp_path):
        class MemoryStub:
            capacity = 10

            def __init__(self):
                self.add_bulk_kwargs = None

            def add_bulk(self, *args, **kwargs):
                self.add_bulk_kwargs = kwargs

        class PolicyStub:
            use_gru = False

            def __init__(self):
                self.memory = MemoryStub()

        class GameStateStub:
            def __init__(self):
                self._shared_policy = PolicyStub()
                self.snakes = []

        db_path = tmp_path / "replay.db"
        handler = MemoryDBHandler(str(db_path))
        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    {
                        "state": [0.0] * GameConfig.INPUT_SIZE,
                        "action": 1,
                        "reward": 0.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                        "next_action_mask": [False, True, False, False, False, False],
                    }
                    for _ in range(3)
                ],
            )
        finally:
            handler.close()
        game_state = GameStateStub()

        with pytest.raises(RuntimeError, match="action coverage"):
            load_replay_db_into_game_state(
                game_state,
                str(db_path),
                limit=None,
                min_action_coverage_fraction=1.0,
            )

        assert game_state._shared_policy.memory.add_bulk_kwargs is None

    def test_replay_db_prefill_n_step_mismatch_raises_before_memory_mutation(self, tmp_path):
        class MemoryStub:
            capacity = 10

            def __init__(self):
                self.add_bulk_kwargs = None

            def add_bulk(self, *args, **kwargs):
                self.add_bulk_kwargs = kwargs

        class PolicyStub:
            use_gru = False

            def __init__(self):
                self.memory = MemoryStub()

        class GameStateStub:
            def __init__(self):
                self._shared_policy = PolicyStub()
                self.snakes = []

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
                        "reward": 0.0,
                        "next_state": [1.0] * GameConfig.INPUT_SIZE,
                        "done": False,
                        "priority": 1.0,
                        "bootstrap_steps": 1,
                    }
                ],
            )
        finally:
            handler.close()
        game_state = GameStateStub()

        with pytest.raises(RuntimeError, match="generation.apex_n_step"):
            load_replay_db_into_game_state(game_state, str(db_path), limit=None)

        assert game_state._shared_policy.memory.add_bulk_kwargs is None

    @pytest.mark.parametrize("value", [None, 0.0, 0.5, 1.0])
    def test_resolve_replay_quality_fraction_accepts_valid_values(self, value):
        expected = 0.0 if value is None else value

        assert resolve_replay_quality_fraction(value, "gate") == expected

    def test_resolve_prefill_replay_quality_gates_applies_preset_and_overrides(self):
        gates = resolve_prefill_replay_quality_gates(
            preset="training",
            overrides={
                "min_positive_reward_fraction": 0.2,
                "max_dominant_action_fraction": 0.9,
            },
        )

        assert gates["min_terminal_fraction"] == pytest.approx(0.005)
        assert gates["min_immediate_terminal_fraction"] == pytest.approx(0.001)
        assert gates["min_exact_mask_fraction"] == pytest.approx(0.8)
        assert gates["min_boost_mask_fraction"] == pytest.approx(0.05)
        assert gates["min_positive_reward_fraction"] == pytest.approx(0.2)
        assert gates["max_dominant_action_fraction"] == pytest.approx(0.9)
        assert gates["max_invalid_current_action_fraction"] == pytest.approx(0.0)

    @pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), "bad"])
    def test_resolve_replay_quality_fraction_rejects_invalid_values(self, value):
        with pytest.raises(ValueError, match=r"gate must be finite and in \[0, 1\]"):
            resolve_replay_quality_fraction(value, "gate")

    def test_missing_replay_db_raises(self, tmp_path):
        missing_db = tmp_path / "missing.db"

        with pytest.raises(FileNotFoundError, match="Replay database not found"):
            load_replay_db_into_game_state(object(), str(missing_db))


class TestConfiguredCheckpoints:
    """Tests for using configured checkpoint paths in headless helpers."""

    class SaveableSnake:
        def __init__(self, total_reward):
            self.total_reward = total_reward
            self.saved_paths = []

        def save_state(self, path):
            self.saved_paths.append(path)
            with open(path, "w", encoding="utf-8") as checkpoint:
                checkpoint.write(f"reward={self.total_reward}")

    class GameStateStub:
        def __init__(self, snakes):
            self.snakes = snakes

    def test_checkpoint_path_uses_configured_directory(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))

        try:
            initialize_config(config)

            assert get_checkpoint_path("best.pth") == checkpoint_dir / "best.pth"
        finally:
            initialize_config(original_config)

    def test_resolve_checkpoint_path_uses_configured_directory(self, tmp_path):
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

    def test_save_training_checkpoint_uses_current_best_snake(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))
        weaker = self.SaveableSnake(total_reward=1.0)
        stronger = self.SaveableSnake(total_reward=5.0)
        game_state = self.GameStateStub([weaker, stronger])

        try:
            initialize_config(config)

            checkpoint_path = save_training_checkpoint(game_state, "env_0_best_snake.pth")

            assert checkpoint_path == checkpoint_dir / "env_0_best_snake.pth"
            assert checkpoint_path.read_text(encoding="utf-8") == "reward=5.0"
            assert weaker.saved_paths == []
            assert stronger.saved_paths == [str(checkpoint_path)]
        finally:
            initialize_config(original_config)

    def test_final_checkpoint_is_best_fallback_for_short_runs(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))
        snake = self.SaveableSnake(total_reward=2.0)
        game_state = self.GameStateStub([snake])

        try:
            initialize_config(config)

            final_path, best_path = save_final_training_checkpoints(game_state, env_id=3)

            assert final_path == checkpoint_dir / "env_3_final_snake.pth"
            assert best_path == final_path
            assert final_path.read_text(encoding="utf-8") == "reward=2.0"
            assert not (checkpoint_dir / "env_3_best_snake.pth").exists()
        finally:
            initialize_config(original_config)

    def test_final_checkpoint_keeps_current_run_best_when_provided(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        best_path = checkpoint_dir / "env_2_best_snake.pth"
        best_path.write_text("current-run-best", encoding="utf-8")
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))
        snake = self.SaveableSnake(total_reward=2.0)
        game_state = self.GameStateStub([snake])

        try:
            initialize_config(config)

            final_path, returned_best_path = save_final_training_checkpoints(
                game_state,
                env_id=2,
                best_checkpoint_path=best_path,
            )

            assert final_path == checkpoint_dir / "env_2_final_snake.pth"
            assert returned_best_path == best_path
            assert final_path.read_text(encoding="utf-8") == "reward=2.0"
            assert best_path.read_text(encoding="utf-8") == "current-run-best"
        finally:
            initialize_config(original_config)

    def test_best_env_model_path_prefers_reported_run_checkpoint(self, tmp_path):
        reported_path = tmp_path / "env_1_final_snake.pth"

        assert get_best_env_model_path(1, {"best_checkpoint": str(reported_path)}) == reported_path

    def test_best_env_model_path_falls_back_to_env_best_name(self, tmp_path):
        original_config = get_config()
        checkpoint_dir = tmp_path / "checkpoints"
        config = AppConfig(checkpoint=CheckpointSettings(checkpoint_dir=str(checkpoint_dir)))

        try:
            initialize_config(config)

            assert get_best_env_model_path(4, {}) == checkpoint_dir / "env_4_best_snake.pth"
        finally:
            initialize_config(original_config)
