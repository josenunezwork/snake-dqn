"""Tests for per-term reward accounting and behavioral probes (blueprint §5.3)."""

from types import SimpleNamespace

import pytest
import torch

from src.core.game_config import (
    AppConfig,
    GameConfig,
    GameSettings,
    RewardSettings,
    StateIndices,
    get_config,
    initialize_config,
)
from src.game.game_state import GameState
from src.game.snake_reward import REWARD_TERM_KEYS
from src.training.behavior_probes import BehaviorProbes
from tests.conftest import make_test_snake

pytestmark = pytest.mark.usefixtures("setup_config")


class FakePolicy:
    """Minimal policy stand-in for GameState construction."""

    epsilon = 0.0


def _zero_state() -> torch.Tensor:
    return torch.zeros(GameConfig.INPUT_SIZE)


def _assert_breakdown_sums_to(snake, reward: float) -> None:
    """Assert the stored breakdown has all keys and sums exactly to the reward."""
    breakdown = snake.last_reward_breakdown
    assert breakdown is not None
    assert tuple(breakdown.keys()) == REWARD_TERM_KEYS
    assert sum(breakdown.values()) == reward


class TestRewardBreakdown:
    """Per-term reward accounting: breakdown sums exactly to the returned float."""

    def test_death_breakdown(self):
        snake = make_test_snake(0, (400, 300))
        reward = snake.calculate_reward(False, True, _zero_state(), None, [], [])

        _assert_breakdown_sums_to(snake, reward)
        assert snake.last_reward_breakdown["death"] == GameConfig.REWARD_DEATH
        assert reward == GameConfig.REWARD_DEATH

    def test_death_breakdown_with_length_scale_floor(self):
        """The REWARD_MIN floor on scaled death shows up as clamp_delta."""
        original_config = get_config()
        try:
            initialize_config(AppConfig(rewards=RewardSettings(death_length_scale=0.5)))
            snake = make_test_snake(0, (400, 300))
            snake.length = 10 * GameConfig.MAX_LENGTH  # norm_len caps at 2.0

            reward = snake.calculate_reward(False, True, _zero_state(), None, [], [])

            _assert_breakdown_sums_to(snake, reward)
            assert reward == GameConfig.REWARD_MIN
            assert snake.last_reward_breakdown["death"] < GameConfig.REWARD_MIN
            assert snake.last_reward_breakdown["clamp_delta"] > 0.0
        finally:
            initialize_config(original_config)

    def test_food_breakdown(self):
        snake = make_test_snake(0, (400, 300))
        state = _zero_state()
        reward = snake.calculate_reward(True, False, state, state, [], [])

        _assert_breakdown_sums_to(snake, reward)
        assert snake.last_reward_breakdown["food"] == GameConfig.REWARD_FOOD_BASE
        assert snake.last_reward_breakdown["survival"] == 0.0

    def test_starvation_breakdown(self):
        snake = make_test_snake(0, (400, 300))
        snake.frames_since_food = GameConfig.STARVATION_START_FRAME + 250
        state = _zero_state()
        # Keep boundary distances high so the wall term stays zero.
        state[StateIndices.BOUNDARY_LEFT : StateIndices.BOUNDARY_BOTTOM + 1] = 1.0

        reward = snake.calculate_reward(False, False, state, state, [], [])

        _assert_breakdown_sums_to(snake, reward)
        assert snake.last_reward_breakdown["starvation"] < 0.0
        assert snake.last_reward_breakdown["survival"] == GameConfig.REWARD_SURVIVAL
        assert snake.last_reward_breakdown["wall"] == 0.0

    def test_boost_burn_breakdown(self):
        original_config = get_config()
        try:
            initialize_config(AppConfig(rewards=RewardSettings(boost_segment=0.5)))
            snake = make_test_snake(0, (400, 300))
            snake.length = 6
            snake._pre_collision_action = 4  # boosted straight
            snake._reward_prev_length = 7  # burned one segment this frame
            state = _zero_state()
            state[StateIndices.BOUNDARY_LEFT : StateIndices.BOUNDARY_BOTTOM + 1] = 1.0

            reward = snake.calculate_reward(False, False, state, state, [], [])

            _assert_breakdown_sums_to(snake, reward)
            assert snake.last_reward_breakdown["boost"] == -0.5
        finally:
            initialize_config(original_config)

    def test_kill_breakdown(self):
        killer = make_test_snake(0, (400, 300))
        victim = make_test_snake(1, (500, 300), segments=[(500, 300), (490, 300), (480, 300)])
        victim.die()
        state = _zero_state()
        state[StateIndices.BOUNDARY_LEFT : StateIndices.BOUNDARY_BOTTOM + 1] = 1.0

        reward = killer.calculate_reward(
            False, False, state, state, [killer, victim], [], frame_kills={0: [1]}
        )

        _assert_breakdown_sums_to(killer, reward)
        assert killer.last_reward_breakdown["kill"] > 0.0

    def test_clamped_total_accounted_by_clamp_delta(self):
        """Food + max kill overshoots REWARD_MAX; clamp_delta absorbs it exactly."""
        killer = make_test_snake(0, (400, 300))
        long_victim = make_test_snake(
            1, (500, 300), segments=[(500 + 10 * i, 300) for i in range(120)]
        )
        long_victim.die()
        state = _zero_state()

        reward = killer.calculate_reward(
            True, False, state, state, [killer, long_victim], [], frame_kills={0: [1]}
        )

        _assert_breakdown_sums_to(killer, reward)
        assert reward == GameConfig.REWARD_MAX
        assert killer.last_reward_breakdown["clamp_delta"] < 0.0


def _fake_snake(snake_id: int, head, segments, length=None, alive=True) -> SimpleNamespace:
    return SimpleNamespace(
        id=snake_id,
        is_alive=alive,
        head=head,
        segments=list(segments),
        length=length if length is not None else len(segments),
        segment_size=10,
        game_width=200,
        game_height=200,
    )


def _fake_game_state(snakes, frame=0) -> SimpleNamespace:
    return SimpleNamespace(
        frame=frame,
        snakes=snakes,
        frame_collisions={},
        frame_kills={},
        frame_death_causes={},
        _game_width=200,
        _game_height=200,
    )


class TestBehaviorProbesCounting:
    """Boost fraction, food, kill, and kill-opportunity accounting."""

    def test_boost_fraction_and_reward_terms(self):
        snake = _fake_snake(0, (100.0, 100.0), [(100, 100), (90, 100)])
        game_state = _fake_game_state([snake])
        probes = BehaviorProbes()

        breakdown_food = {key: 0.0 for key in REWARD_TERM_KEYS}
        breakdown_food["food"] = 3.0
        breakdown_plain = {key: 0.0 for key in REWARD_TERM_KEYS}
        breakdown_plain["survival"] = 0.01

        kill_opp_state = torch.zeros(GameConfig.INPUT_SIZE)
        kill_opp_state[StateIndices.KILL_OPPORTUNITY] = 0.9

        actions = [3, 1, 5, 0]
        for i, action in enumerate(actions):
            game_state.frame = i + 1
            snake.last_transition_frame = game_state.frame
            snake.last_action = action
            snake.last_reward_breakdown = breakdown_food if i == 0 else breakdown_plain
            snake.last_state = kill_opp_state if i < 2 else None
            game_state.frame_kills = {0: [1]} if i == 1 else {}
            probes.observe(game_state)

        records = probes.finalize_episode()

        assert len(records) == 1
        record = records[0]
        assert record["boost_frame_fraction"] == pytest.approx(0.5)  # actions 3 and 5 of 4
        assert record["food_eaten"] == 1
        assert record["kills"] == 1
        assert record["kill_opportunity_count"] == 2
        assert record["survival_frames"] == 4
        assert record["death_cause"] is None
        assert record["reward_terms"]["food"] == pytest.approx(3.0)
        assert record["reward_terms"]["survival"] == pytest.approx(0.03)

    def test_non_learning_snake_food_and_boost_from_game_state_maps(self):
        """Scripted/human snakes (no last_transition_frame) still get real
        food_eaten / boost_frame_fraction via GameState.frame_ate_food and
        their is_boosting flag instead of silent zeros."""
        snake = _fake_snake(0, (100.0, 100.0), [(100, 100), (90, 100)])
        game_state = _fake_game_state([snake])
        probes = BehaviorProbes()

        frames = [(True, False), (False, True), (True, False), (False, False)]
        for i, (ate, boosting) in enumerate(frames):
            game_state.frame = i + 1
            game_state.frame_ate_food = {0: ate}
            snake.is_boosting = boosting
            probes.observe(game_state)

        record = probes.finalize_episode()[0]

        assert record["food_eaten"] == 2
        assert record["boost_frame_fraction"] == pytest.approx(0.25)
        assert record["survival_frames"] == 4
        # No reward breakdown / state vector exists for non-learning snakes.
        assert record["kill_opportunity_count"] == 0

    def test_frame_ate_food_map_supersedes_breakdown_food(self):
        """When GameState exposes frame_ate_food, the breakdown food term must
        not double count the same consumption for learning snakes."""
        snake = _fake_snake(0, (100.0, 100.0), [(100, 100), (90, 100)])
        game_state = _fake_game_state([snake], frame=1)
        game_state.frame_ate_food = {0: True}
        breakdown_food = {key: 0.0 for key in REWARD_TERM_KEYS}
        breakdown_food["food"] = 3.0
        snake.last_transition_frame = 1
        snake.last_action = 1
        snake.last_reward_breakdown = breakdown_food
        snake.last_state = None
        probes = BehaviorProbes()

        probes.observe(game_state)
        record = probes.finalize_episode()[0]

        assert record["food_eaten"] == 1
        assert record["reward_terms"]["food"] == pytest.approx(3.0)

    def test_summarize_aggregates(self):
        snake = _fake_snake(0, (100.0, 100.0), [(100, 100)])
        game_state = _fake_game_state([snake], frame=1)
        probes = BehaviorProbes()
        probes.observe(game_state)
        probes.finalize_episode()

        summary = probes.summarize()

        assert summary["episodes"] == 1
        assert summary["snake_episodes"] == 1
        assert summary["total_kills"] == 0
        assert summary["entrapment_events"] == 0
        assert set(summary["reward_terms_total"]) == set(REWARD_TERM_KEYS)


class TestDeathCauseTagging:
    """Death causes are tagged in a real GameState collision resolution."""

    def test_wall_death_is_tagged_and_probed(self):
        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=200,
                height=200,
                num_snakes=1,
                initial_food=0,
                max_food=0,
                frame_rate=1,
            )
        )
        try:
            initialize_config(config)
            game_state = GameState(headless=True, num_snakes=1, shared_policy=FakePolicy())
            snake = game_state.snakes[0]
            snake.segments = [(50, 50)]
            snake.length = 1
            snake.direction = (-1, 0)

            def drive_left(*args, **kwargs):
                snake.move()

            snake.update = drive_left

            probes = BehaviorProbes()
            for _ in range(100):
                game_state.update(train_mode=True, learn=False)
                probes.observe(game_state)
                if not snake.is_alive:
                    break

            assert not snake.is_alive
            assert snake.last_death_cause == "wall"
            assert game_state.frame_death_causes == {snake.id: "wall"}
            assert game_state.frame_collisions == {snake.id: "wall"}

            records = probes.finalize_episode()
            assert len(records) == 1
            assert records[0]["death_cause"] == "wall"
            assert records[0]["death_frame"] == game_state.frame
            assert records[0]["entrapment_event"] is False
        finally:
            initialize_config(original_config)


class TestEntrapmentDetector:
    """Encircle-detector v0 on hand-built occupancy scenarios."""

    def test_enemy_encirclement_is_detected(self):
        """Enemy ring seals the victim in a 1-cell pocket → entrapment event."""
        # Grid cells (segment_size=10): victim head (10,10), body (9,10).
        victim = _fake_snake(0, (100, 100), [(100, 100), (90, 100)], alive=False)
        ring_cells = [
            (8, 10),
            (12, 10),
            (9, 9),
            (10, 9),
            (11, 9),
            (9, 11),
            (10, 11),
            (11, 11),
        ]
        enemy = _fake_snake(1, (80, 100), [(gx * 10, gy * 10) for gx, gy in ring_cells])
        game_state = _fake_game_state([victim, enemy], frame=5)
        game_state.frame_collisions = {0: "body"}
        game_state.frame_kills = {1: [0]}
        game_state.frame_death_causes = {0: "enemy_body"}

        probes = BehaviorProbes()
        probes.observe(game_state)
        records = probes.finalize_episode()

        victim_record = next(r for r in records if r["snake_id"] == 0)
        assert victim_record["death_cause"] == "enemy_body"
        assert victim_record["entrapment_event"] is True
        entrapment = victim_record["entrapment"]
        # Only the cell at (11, 10) is reachable: below the 2x-length threshold.
        assert entrapment["free_space_cells"] == 1
        assert entrapment["free_space_threshold"] == 4
        assert entrapment["dominant_enemy_id"] == 1
        assert entrapment["dominant_fraction"] >= 0.6
        assert entrapment["head_history"] == [(100.0, 100.0)]

        enemy_record = next(r for r in records if r["snake_id"] == 1)
        assert enemy_record["kills"] == 1
        assert probes.summarize()["entrapment_events"] == 1

    def test_self_trap_is_not_an_entrapment_event(self):
        """A snake sealed in by its OWN body has no dominant enemy blocker."""
        own_ring = [
            (10, 10),  # head
            (10, 9),
            (11, 9),
            (12, 9),
            (12, 10),
            (12, 11),
            (11, 11),
            (10, 11),
            (9, 11),
            (9, 10),
            (9, 9),
        ]
        victim = _fake_snake(
            0, (100, 100), [(gx * 10, gy * 10) for gx, gy in own_ring], alive=False
        )
        game_state = _fake_game_state([victim], frame=3)
        game_state.frame_collisions = {0: "self"}
        game_state.frame_death_causes = {0: "self"}

        probes = BehaviorProbes()
        probes.observe(game_state)
        records = probes.finalize_episode()

        record = records[0]
        entrapment = record["entrapment"]
        assert record["death_cause"] == "self"
        assert record["entrapment_event"] is False
        # Trapped (tiny free space) but no enemy owns the blockers.
        assert entrapment["free_space_cells"] < entrapment["free_space_threshold"]
        assert entrapment["dominant_enemy_id"] is None
