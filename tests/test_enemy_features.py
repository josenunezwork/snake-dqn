"""Tests for enriched enemy features (§2.6) — 10D output."""

import math

import pytest

from src.core.game_config import GameConfig, StateIndices
from tests.conftest import make_test_snake as _make_snake

pytestmark = pytest.mark.usefixtures("setup_config")


class TestNearestEnemyFeatures:
    """Test Snake._get_nearest_enemy_features() returns correct 10D output."""

    def test_no_enemies_returns_zeros(self):
        """With no enemies, all 10 features should be 0."""
        snake = _make_snake(0, (400, 300))
        features = snake._get_nearest_enemy_features([])
        assert len(features) == 10
        assert all(f == 0.0 for f in features)

    def test_no_enemies_with_self_in_list(self):
        """Passing self in other_snakes list should still return zeros."""
        snake = _make_snake(0, (400, 300))
        features = snake._get_nearest_enemy_features([snake])
        assert len(features) == 10
        assert all(f == 0.0 for f in features)

    def test_dead_enemy_ignored(self):
        """Dead enemies should be ignored."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (410, 300))
        enemy.is_alive = False
        features = snake._get_nearest_enemy_features([snake, enemy])
        assert all(f == 0.0 for f in features)

    def test_one_enemy_basic_position(self):
        """With 1 enemy, nearest position features should be populated."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(1, 0))

        features = snake._get_nearest_enemy_features([snake, enemy])

        max_dim = max(800, 600)
        dx = 500 - 400  # 100
        expected_rel_x = dx / max_dim  # 100/800 = 0.125
        assert features[0] == pytest.approx(expected_rel_x)
        assert features[1] == pytest.approx(0.0)  # same y

    def test_one_enemy_relative_size(self):
        """rel_size = min(enemy_len / self_len, 2.0) / 2.0."""
        snake = _make_snake(0, (400, 300), segments=[(400, 300), (390, 300)])
        enemy = _make_snake(
            1,
            (500, 300),
            direction=(1, 0),
            segments=[(500, 300), (490, 300), (480, 300), (470, 300)],
        )

        features = snake._get_nearest_enemy_features([snake, enemy])

        # enemy has 4 segments, self has 2 → ratio = 4/2 = 2.0, clamped 2.0, /2 = 1.0
        assert features[2] == pytest.approx(1.0)

    def test_enemy_relative_size_uses_logical_lengths_while_bodies_fill_in(self):
        """Enemy size should reflect game-rule length, not delayed segment fill-in."""
        snake = _make_snake(0, (400, 300), segments=[(400, 300)])
        snake.length = 5
        enemy = _make_snake(1, (500, 300), direction=(1, 0), segments=[(500, 300)])
        enemy.length = 10

        features = snake._get_nearest_enemy_features([snake, enemy])

        assert features[2] == pytest.approx(1.0)

    def test_one_enemy_heading(self):
        """Heading features should match enemy's direction tuple."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(0, -1))  # facing up

        features = snake._get_nearest_enemy_features([snake, enemy])

        assert features[3] == pytest.approx(0.0)  # heading dx
        assert features[4] == pytest.approx(-1.0)  # heading dy

    def test_distance_trend_closing(self):
        """Distance trend = +1 when enemy is closer than previous call."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(1, 0))

        # First call establishes the baseline; no trend is known yet.
        features1 = snake._get_nearest_enemy_features([snake, enemy])
        assert features1[5] == pytest.approx(0.0)

        # Move enemy closer
        enemy.segments = [(450, 300)]
        features2 = snake._get_nearest_enemy_features([snake, enemy])
        assert features2[5] == pytest.approx(1.0)  # still closing

    def test_distance_trend_separating(self):
        """Distance trend = -1 when enemy is farther than previous call."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (450, 300), direction=(1, 0))

        # First call establishes the baseline.
        snake._get_nearest_enemy_features([snake, enemy])

        # Move enemy farther away
        enemy.segments = [(600, 300)]
        features = snake._get_nearest_enemy_features([snake, enemy])
        assert features[5] == pytest.approx(-1.0)  # separating

    def test_distance_trend_neutral_when_nearest_enemy_changes(self):
        """Trend should not compare distances from different enemy identities."""
        snake = _make_snake(0, (400, 300))
        enemy1 = _make_snake(1, (450, 300), direction=(1, 0))
        enemy2 = _make_snake(2, (430, 300), direction=(-1, 0))

        first = snake._get_nearest_enemy_features([snake, enemy1])
        assert first[5] == pytest.approx(0.0)
        assert snake._prev_nearest_enemy_id == 1

        switched = snake._get_nearest_enemy_features([snake, enemy1, enemy2])
        assert switched[5] == pytest.approx(0.0)
        assert snake._prev_nearest_enemy_id == 2

        enemy2.segments = [(420, 300)]
        closing = snake._get_nearest_enemy_features([snake, enemy1, enemy2])
        assert closing[5] == pytest.approx(1.0)

    def test_replay_next_state_does_not_consume_enemy_trend(self):
        """Replay next-state capture should not hide trend from the next action state."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(-1, 0))

        # The action state establishes the trend baseline at distance 100.
        first_state = snake.get_state([snake, enemy], [(100, 100)])
        assert float(first_state[StateIndices.ENEMY_DISTANCE_TREND]) == pytest.approx(0.0)
        assert snake._prev_nearest_enemy_dist == pytest.approx(100.0)

        # The replay next_state sees the enemy closing, but must not advance
        # the baseline. Otherwise the next real action state would read neutral.
        enemy.segments = [(450, 300)]
        replay_next_state = snake.get_state(
            [snake, enemy],
            [(100, 100)],
            update_enemy_memory=False,
        )
        assert float(replay_next_state[StateIndices.ENEMY_DISTANCE_TREND]) == pytest.approx(1.0)
        assert snake._prev_nearest_enemy_dist == pytest.approx(100.0)

        next_action_state = snake.get_state([snake, enemy], [(100, 100)])
        assert float(next_action_state[StateIndices.ENEMY_DISTANCE_TREND]) == pytest.approx(1.0)
        assert snake._prev_nearest_enemy_dist == pytest.approx(50.0)

    def test_second_enemy_populated(self):
        """With 2+ enemies, 2nd nearest features are populated (indices 6-8)."""
        snake = _make_snake(0, (400, 300))
        enemy1 = _make_snake(1, (450, 300), direction=(1, 0))
        enemy2 = _make_snake(2, (600, 300), direction=(-1, 0))

        features = snake._get_nearest_enemy_features([snake, enemy1, enemy2])

        max_dim = 800
        # 2nd nearest is enemy2 at (600, 300)
        dx2 = 600 - 400
        expected_rel_x2 = dx2 / max_dim
        assert features[6] == pytest.approx(expected_rel_x2)
        assert features[7] == pytest.approx(0.0)  # same y

    def test_second_enemy_zeros_when_only_one(self):
        """With only 1 enemy, 2nd nearest features (6-8) should be 0."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(1, 0))

        features = snake._get_nearest_enemy_features([snake, enemy])

        assert features[6] == 0.0
        assert features[7] == 0.0
        assert features[8] == 0.0

    def test_kill_opportunity_adjacent(self):
        """Kill opportunity = 1.0 when close to enemy's projected path."""
        snake = _make_snake(0, (400, 300))
        enemy2 = _make_snake(1, (415, 300), direction=(1, 0))
        features2 = snake._get_nearest_enemy_features([snake, enemy2])
        # projected = (425, 300), dist = sqrt(25^2) = 25 < 30 → 1.0
        assert features2[9] == pytest.approx(1.0)

    def test_kill_opportunity_far(self):
        """Kill opportunity = 0.0 when far from enemy's projected path."""
        snake = _make_snake(0, (400, 300))
        # Enemy far away
        enemy = _make_snake(1, (600, 300), direction=(1, 0))

        features = snake._get_nearest_enemy_features([snake, enemy])

        assert features[9] == pytest.approx(0.0)


class TestEnemyFeaturesInState:
    """Test enemy features are correctly embedded in the full state vector."""

    def test_state_enemy_indices(self):
        """State vector enemy slice starts at index 44 and spans 10 values."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(0, 1))

        state = snake.get_state([snake, enemy], [(100, 100)])

        # Enemy features span indices 44-53
        assert StateIndices.ENEMY_REL_X == 44
        assert StateIndices.KILL_OPPORTUNITY == 53

        # Verify rel_x is non-zero (enemy is to the right)
        assert float(state[StateIndices.ENEMY_REL_X]) > 0.0

        # Verify heading matches enemy direction (0, 1) = facing down
        assert float(state[StateIndices.ENEMY_HEADING_DX]) == pytest.approx(0.0)
        assert float(state[StateIndices.ENEMY_HEADING_DY]) == pytest.approx(1.0)

    def test_state_size_with_enemies(self):
        """State vector has correct total size."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(1, 0))
        state = snake.get_state([snake, enemy], [(100, 100)])
        assert state.shape == (GameConfig.INPUT_SIZE,)


class TestPrevEnemyDistTracking:
    """Test _prev_nearest_enemy_dist lifecycle."""

    def test_initialized_to_inf(self):
        """Should be infinity on construction."""
        snake = _make_snake(0, (400, 300))
        assert snake._prev_nearest_enemy_dist == float("inf")
        assert snake._prev_nearest_enemy_id is None

    def test_updated_after_get_state(self):
        """Should be updated to actual distance after get_state call."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(1, 0))

        snake.get_state([snake, enemy], [(100, 100)])

        expected_dist = math.sqrt((500 - 400) ** 2 + (300 - 300) ** 2)
        assert snake._prev_nearest_enemy_dist == pytest.approx(expected_dist)
        assert snake._prev_nearest_enemy_id == 1

    def test_no_enemies_resets_tracker(self):
        """Losing enemy visibility should make the next sighting a fresh baseline."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(1, 0))

        snake.get_state([snake, enemy], [(100, 100)])
        assert snake._prev_nearest_enemy_id == 1

        features = snake._get_nearest_enemy_features([snake])

        assert all(f == 0.0 for f in features)
        assert snake._prev_nearest_enemy_dist == float("inf")
        assert snake._prev_nearest_enemy_id is None

    def test_reset_on_respawn(self):
        """Should reset to infinity on respawn."""
        snake = _make_snake(0, (400, 300))
        enemy = _make_snake(1, (500, 300), direction=(1, 0))

        snake.get_state([snake, enemy], [(100, 100)])
        assert snake._prev_nearest_enemy_dist != float("inf")
        assert snake._prev_nearest_enemy_id == 1

        snake.respawn((200, 200))
        assert snake._prev_nearest_enemy_dist == float("inf")
        assert snake._prev_nearest_enemy_id is None

        features = snake._get_nearest_enemy_features([snake, enemy])
        assert features[5] == pytest.approx(0.0)
