"""Tests for CurriculumManager (§2.4)."""

from src.training.curriculum import CurriculumManager, CurriculumPhase


def test_training_package_lazy_exports_curriculum_manager():
    """Package-level curriculum export should not import the whole training stack."""
    from src.training import CurriculumManager as ExportedCurriculumManager

    assert ExportedCurriculumManager is CurriculumManager


class TestCurriculumInitialState:
    """Test initial state of CurriculumManager."""

    def test_starts_at_phase_zero(self):
        cm = CurriculumManager()
        assert cm.current_phase_idx == 0

    def test_initial_phase_is_solo_easy(self):
        cm = CurriculumManager()
        assert cm.phase_name == "solo_easy"

    def test_no_episodes_recorded(self):
        cm = CurriculumManager()
        assert cm.total_episodes == 0
        assert cm.phase_episodes == 0
        assert len(cm.episode_lengths) == 0

    def test_five_phases_defined(self):
        assert len(CurriculumManager.PHASES) == 5

    def test_phases_use_curriculum_phase_records(self):
        assert all(isinstance(phase, CurriculumPhase) for phase in CurriculumManager.PHASES)


class TestPhaseProgression:
    """Test phase progression and promotion criteria."""

    def test_not_promoted_with_insufficient_episodes(self):
        """Don't promote if min_episodes not reached."""
        cm = CurriculumManager(window_size=50)
        # Record only 10 episodes (need 50)
        for _ in range(10):
            cm.record_episode(length=999)
        assert not cm.should_promote()

    def test_not_promoted_below_threshold(self):
        """Don't promote if metric is below threshold."""
        cm = CurriculumManager(window_size=50)
        # Phase 0 needs avg_length >= 200
        for _ in range(50):
            cm.record_episode(length=100)
        assert not cm.should_promote()

    def test_promoted_at_threshold(self):
        """Promote when metric meets threshold."""
        cm = CurriculumManager(window_size=50)
        # Phase 0: avg_length >= 200 over 50 episodes
        for _ in range(50):
            cm.record_episode(length=200)
        assert cm.should_promote()

    def test_promote_advances_phase(self):
        """promote() moves to next phase and resets phase_episodes."""
        cm = CurriculumManager(window_size=50)
        for _ in range(50):
            cm.record_episode(length=200)
        old_phase = cm.current_phase_idx
        cm.promote()
        assert cm.current_phase_idx == old_phase + 1
        assert cm.phase_episodes == 0

    def test_check_and_promote_auto_advances(self):
        """check_and_promote() returns (True, new_phase) when ready."""
        cm = CurriculumManager(window_size=50)
        for _ in range(50):
            cm.record_episode(length=200)
        promoted, phase = cm.check_and_promote()
        assert promoted is True
        assert phase.name == "solo_full"

    def test_check_and_promote_returns_false_when_not_ready(self):
        cm = CurriculumManager(window_size=50)
        for _ in range(10):
            cm.record_episode(length=50)
        promoted, phase = cm.check_and_promote()
        assert promoted is False
        assert phase.name == "solo_easy"

    def test_cannot_promote_past_last_phase(self):
        """Already at max phase: don't promote."""
        cm = CurriculumManager(window_size=50)
        cm.current_phase_idx = len(CurriculumManager.PHASES) - 1  # last phase
        for _ in range(50):
            cm.record_episode(length=9999)
        assert not cm.should_promote()

    def test_phase1_to_phase2_requires_avg_500(self):
        """Phase 1 (solo_full) promotes at avg_length >= 500."""
        cm = CurriculumManager(window_size=50)
        cm.current_phase_idx = 1
        for _ in range(50):
            cm.record_episode(length=499)
        assert not cm.should_promote()
        # Add one more that pushes avg over 500
        cm.record_episode(length=550)
        # Window is 50, so oldest 499 dropped, now 49x499 + 1x550
        # Need to fill entirely above 500
        cm2 = CurriculumManager(window_size=50)
        cm2.current_phase_idx = 1
        for _ in range(50):
            cm2.record_episode(length=500)
        assert cm2.should_promote()


class TestGameSettings:
    """Test get_game_settings() for each phase."""

    def test_phase0_settings(self):
        cm = CurriculumManager()
        settings = cm.get_game_settings()
        assert settings["num_snakes"] == 1
        assert settings["board_scale"] == 0.5
        assert settings["food_multiplier"] == 2.0
        assert settings["phase_name"] == "solo_easy"
        assert settings["phase_idx"] == 0

    def test_phase3_settings(self):
        cm = CurriculumManager()
        cm.current_phase_idx = 3  # competitive
        settings = cm.get_game_settings()
        assert settings["num_snakes"] == 4
        assert settings["board_scale"] == 1.0
        assert settings["food_multiplier"] == 1.0
        assert settings["phase_name"] == "competitive"

    def test_phase4_settings(self):
        cm = CurriculumManager()
        cm.current_phase_idx = 4  # advanced
        settings = cm.get_game_settings()
        assert settings["num_snakes"] == 4
        assert settings["board_scale"] == 1.0
        assert settings["food_multiplier"] == 1.0
        assert settings["phase_name"] == "advanced"


class TestStateSerialization:
    """Test get_state() / load_state() round-trip."""

    def test_get_state_returns_dict(self):
        cm = CurriculumManager()
        for _ in range(10):
            cm.record_episode(length=50, kills=2, deaths=1)
        state = cm.get_state()
        assert isinstance(state, dict)
        assert "current_phase_idx" in state
        assert "total_episodes" in state
        assert "phase_episodes" in state
        assert "episode_lengths" in state
        assert "episode_kills" in state
        assert "episode_deaths" in state

    def test_round_trip_preserves_all_fields(self):
        cm = CurriculumManager(window_size=50)
        cm.current_phase_idx = 2
        for i in range(25):
            cm.record_episode(length=100 + i, kills=i % 3, deaths=i % 2)
        state = cm.get_state()

        cm2 = CurriculumManager(window_size=50)
        cm2.load_state(state)

        assert cm2.current_phase_idx == cm.current_phase_idx
        assert cm2.total_episodes == cm.total_episodes
        assert cm2.phase_episodes == cm.phase_episodes
        assert list(cm2.episode_lengths) == list(cm.episode_lengths)
        assert list(cm2.episode_kills) == list(cm.episode_kills)
        assert list(cm2.episode_deaths) == list(cm.episode_deaths)

    def test_load_state_with_empty_dict(self):
        """Load from empty dict uses defaults."""
        cm = CurriculumManager()
        cm.load_state({})
        assert cm.current_phase_idx == 0
        assert cm.total_episodes == 0


class TestKillDeathRatioMetric:
    """Test kill/death ratio promotion in competitive phase."""

    def test_phase3_uses_kill_death_ratio(self):
        assert CurriculumManager.PHASES[3].promotion_metric == "kill_death_ratio"
        assert CurriculumManager.PHASES[3].promotion_threshold == 0.5

    def test_kd_ratio_promotes_when_met(self):
        """Phase 3 promotes when kill_death_ratio >= 0.5."""
        cm = CurriculumManager(window_size=50)
        cm.current_phase_idx = 3
        # Record 50 episodes: 1 kill per episode, 1 death per episode → ratio 1.0
        for _ in range(50):
            cm.record_episode(length=100, kills=1, deaths=1)
        assert cm.should_promote()

    def test_kd_ratio_does_not_promote_below_threshold(self):
        """Phase 3 doesn't promote when kill_death_ratio < 0.5."""
        cm = CurriculumManager(window_size=50)
        cm.current_phase_idx = 3
        # 0 kills, 2 deaths per episode → ratio 0.0
        for _ in range(50):
            cm.record_episode(length=100, kills=0, deaths=2)
        assert not cm.should_promote()

    def test_kd_ratio_with_zero_deaths(self):
        """kill_death_ratio with 0 deaths uses max(total_deaths, 1)."""
        cm = CurriculumManager(window_size=50)
        cm.current_phase_idx = 3
        for _ in range(50):
            cm.record_episode(length=100, kills=1, deaths=0)
        # ratio = 50 / max(0, 1) = 50.0 >= 0.5
        assert cm.should_promote()

    def test_total_episodes_tracked_correctly(self):
        cm = CurriculumManager()
        for i in range(10):
            cm.record_episode(length=50)
        assert cm.total_episodes == 10
        assert cm.phase_episodes == 10
