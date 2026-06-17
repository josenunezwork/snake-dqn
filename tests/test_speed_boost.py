"""Tests for speed boost mechanic (§2.3)."""

import pytest

from src.core.game_config import GameConfig, StateIndices
from src.game.ai_snake import AISnake
from src.game.snake import Snake

pytestmark = pytest.mark.usefixtures("setup_config")


def _make_long_snake(length: int = 10) -> Snake:
    """Create a snake with the given length, heading right, in the center."""
    snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
    snake.direction = (1, 0)
    # Grow the snake to desired length
    snake.length = length
    # Build segment list: head at (400,300), body trailing left
    snake.segments = [(400 - i * 10, 300) for i in range(length)]
    return snake


class _PolicyStub:
    """Minimal policy object for safe-action helper tests."""

    epsilon = 0.0


class TestBoostMovement:
    """Test that boosting moves 2 cells per frame."""

    def test_normal_move_one_cell(self):
        """Non-boosting snake moves 1 cell per frame."""
        snake = _make_long_snake(5)
        snake.is_boosting = False
        old_head = snake.head
        snake.move()
        new_head = snake.head
        dx = new_head[0] - old_head[0]
        assert dx == 10  # One segment_size step

    def test_boost_move_two_cells(self):
        """Boosting snake moves 2 cells per frame."""
        snake = _make_long_snake(10)
        snake.is_boosting = True
        old_head = snake.head
        snake.move()
        new_head = snake.head
        dx = new_head[0] - old_head[0]
        assert dx == 20  # Two segment_size steps


class TestBoostMinLength:
    """Test boost requires minimum length."""

    def test_boost_requires_min_length(self):
        """Cannot effectively boost with length < MIN_BOOST_LENGTH."""
        snake = _make_long_snake(3)  # Below min_boost_length (5)
        snake.is_boosting = True
        old_head = snake.head
        snake.move()
        new_head = snake.head
        dx = new_head[0] - old_head[0]
        # With length < MIN_BOOST_LENGTH, only moves 1 cell
        assert dx == 10

    def test_boost_works_at_min_length(self):
        """Boost works at exactly MIN_BOOST_LENGTH."""
        min_len = GameConfig.MIN_BOOST_LENGTH  # 5
        snake = _make_long_snake(min_len)
        snake.is_boosting = True
        old_head = snake.head
        snake.move()
        new_head = snake.head
        dx = new_head[0] - old_head[0]
        assert dx == 20  # Two cells

    def test_boost_uses_length_budget_while_body_is_still_growing(self):
        """Boost availability follows length, not the currently filled segment count."""
        snake = _make_long_snake(3)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.is_boosting = True
        old_head = snake.head

        snake.move()

        new_head = snake.head
        dx = new_head[0] - old_head[0]
        assert dx == 20


class TestBoostLengthCost:
    """Test boost length cost mechanic."""

    def test_boost_loses_segment_after_n_frames(self):
        """Boosting loses 1 segment every BOOST_LENGTH_COST_FRAMES."""
        cost_frames = GameConfig.BOOST_LENGTH_COST_FRAMES  # 3
        snake = _make_long_snake(10)
        snake.is_boosting = True
        initial_length = snake.length

        # Move cost_frames times → should lose 1 length
        for _ in range(cost_frames):
            snake.move()

        assert snake.length == initial_length - 1

    def test_boost_frames_counter_resets(self):
        """boost_frames resets to 0 after paying length cost."""
        cost_frames = GameConfig.BOOST_LENGTH_COST_FRAMES
        snake = _make_long_snake(10)
        snake.is_boosting = True

        for _ in range(cost_frames):
            snake.move()

        assert snake.boost_frames == 0

    def test_length_never_goes_below_one(self):
        """Length can't drop below 1 from boosting."""
        snake = _make_long_snake(2)
        snake.is_boosting = True
        # Move many times to try to drain length
        for _ in range(20):
            snake.move()
        assert snake.length >= 1

    def test_no_length_cost_when_not_boosting(self):
        """Normal movement doesn't cost length."""
        snake = _make_long_snake(10)
        snake.is_boosting = False
        initial_length = snake.length

        for _ in range(20):
            snake.move()

        assert snake.length == initial_length


class TestSixActionSpace:
    """Test 6-action output space (3 dirs x 2 speed modes)."""

    # NOTE: test_output_size_is_six removed -- covered by TestConstants in
    # test_relative_actions.py.

    def test_actions_0_to_2_are_normal(self):
        """Actions 0-2 map to left/straight/right without boost."""
        # This is a design test — the action space is:
        # 0=normal left, 1=normal straight, 2=normal right
        # 3=boost left, 4=boost straight, 5=boost right
        # Verify via OUTPUT_SIZE and the split logic
        assert GameConfig.OUTPUT_SIZE == 6
        # Normal actions: action % 3 gives relative direction
        for action in range(3):
            assert action % 3 == action  # 0→0, 1→1, 2→2
        # Boost actions: action % 3 also gives relative direction
        for action in range(3, 6):
            assert action % 3 == action - 3  # 3→0, 4→1, 5→2


class TestBoostSafeActions:
    """Test boost-aware action masking."""

    def _make_ai_snake(self, start_pos=(400, 300)) -> AISnake:
        """Create an AI snake with a stub policy for action-mask tests."""
        return AISnake(
            id=0,
            color=(255, 0, 0),
            start_pos=start_pos,
            segment_size=10,
            game_width=800,
            game_height=600,
            policy=_PolicyStub(),
        )

    def test_boost_straight_unsafe_when_second_step_hits_wall(self):
        """Boost actions should check their two-step destination."""
        snake = self._make_ai_snake(start_pos=(785, 300))
        snake.direction = (1, 0)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(785 - i * 10, 300) for i in range(snake.length)]

        safe_actions = snake._get_safe_actions()

        assert 1 in safe_actions  # normal straight ends at x=795, still in bounds
        assert 4 not in safe_actions  # boosted straight ends at x=805, out of bounds

    def test_normal_action_into_enemy_head_is_masked(self):
        """Safe-action masking should include immediate snake collisions."""
        snake = self._make_ai_snake()
        snake.direction = (1, 0)
        enemy = Snake(1, (0, 255, 0), (410, 300), 10, 800, 600)
        enemy.is_alive = True

        safe_actions = snake._get_safe_actions([snake, enemy])

        assert 1 not in safe_actions

    def test_moving_into_tail_remains_safe_when_tail_will_move(self):
        """The action mask should not block moves into a tail that will be popped."""
        snake = self._make_ai_snake()
        snake.direction = (1, 0)
        snake.length = 3
        snake.segments = [(400, 300), (390, 300), (400, 290)]

        safe_actions = snake._get_safe_actions([snake])

        assert 0 in safe_actions

    def test_moving_into_persistent_self_body_is_masked(self):
        """The action mask should block moves into body segments that persist."""
        snake = self._make_ai_snake()
        snake.direction = (1, 0)
        snake.length = 4
        snake.segments = [(400, 300), (390, 300), (400, 290), (380, 300)]

        safe_actions = snake._get_safe_actions([snake])

        assert 0 not in safe_actions

    def test_boost_action_into_enemy_second_step_is_masked(self):
        """Boost action masking should check the final two-step destination."""
        snake = self._make_ai_snake()
        snake.direction = (1, 0)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(400 - i * 10, 300) for i in range(snake.length)]
        enemy = Snake(1, (0, 255, 0), (420, 300), 10, 800, 600)
        enemy.is_alive = True

        safe_actions = snake._get_safe_actions([snake, enemy])

        assert 1 in safe_actions
        assert 4 not in safe_actions

    def test_boost_actions_use_length_budget_while_body_is_still_growing(self):
        """Safe-action masks should agree with the length-based boost state feature."""
        snake = self._make_ai_snake()
        snake.direction = (1, 0)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(400, 300), (390, 300), (380, 300)]

        safe_actions = snake._get_safe_actions([snake])

        assert any(action >= 3 for action in safe_actions)

    def test_boost_action_crossing_enemy_head_is_masked(self):
        """Boost action masking should check the first traversed head position."""
        snake = self._make_ai_snake()
        snake.direction = (1, 0)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(400 - i * 10, 300) for i in range(snake.length)]
        enemy = Snake(1, (0, 255, 0), (410, 300), 10, 800, 600)
        enemy.is_alive = True

        safe_actions = snake._get_safe_actions([snake, enemy])

        assert 1 not in safe_actions
        assert 4 not in safe_actions

    def test_boost_action_crossing_enemy_body_is_masked(self):
        """Boost action masking should match collision checks for traversed body hits."""
        snake = self._make_ai_snake()
        snake.direction = (1, 0)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        snake.segments = [(400 - i * 10, 300) for i in range(snake.length)]
        enemy = Snake(1, (0, 255, 0), (500, 300), 10, 800, 600)
        enemy.length = 3
        enemy.segments = [(500, 300), (410, 300), (490, 300)]
        enemy.is_alive = True

        safe_actions = snake._get_safe_actions([snake, enemy])

        assert 1 not in safe_actions
        assert 4 not in safe_actions

    def test_moving_into_enemy_tail_remains_safe_when_tail_will_move(self):
        """The action mask should allow a non-growing enemy's vacating tail."""
        snake = self._make_ai_snake()
        snake.direction = (1, 0)
        enemy = Snake(1, (0, 255, 0), (430, 300), 10, 800, 600)
        enemy.direction = (1, 0)
        enemy.length = 3
        enemy.segments = [(430, 300), (420, 300), (410, 300)]

        safe_actions = snake._get_safe_actions([snake, enemy])

        assert 1 in safe_actions

    def test_moving_into_growing_enemy_tail_is_masked(self):
        """A growing enemy's current tail persists and should stay blocked."""
        snake = self._make_ai_snake()
        snake.direction = (1, 0)
        enemy = Snake(1, (0, 255, 0), (430, 300), 10, 800, 600)
        enemy.direction = (1, 0)
        enemy.length = 4
        enemy.segments = [(430, 300), (420, 300), (410, 300)]

        safe_actions = snake._get_safe_actions([snake, enemy])

        assert 1 not in safe_actions


class TestBoostState:
    """Test boost availability in state vector."""

    def test_boost_state_index_is_57(self):
        assert StateIndices.BOOST_AVAILABLE == 57

    def test_boost_available_when_long_enough(self):
        """[57] = 1.0 when length >= min_boost_length."""
        snake = _make_long_snake(GameConfig.MIN_BOOST_LENGTH)
        food = [(500, 300)]
        state = snake.get_state([], food)
        assert float(state[57]) == 1.0

    def test_boost_available_uses_length_when_body_is_still_growing(self):
        """[57] should reflect the configured length budget, not body fill-in lag."""
        snake = _make_long_snake(3)
        snake.length = GameConfig.MIN_BOOST_LENGTH
        state = snake.get_state([], [(500, 300)])

        assert float(state[57]) == 1.0

    def test_boost_unavailable_when_too_short(self):
        """[57] = 0.0 when length < min_boost_length."""
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        snake.direction = (1, 0)
        # Default length = 1, below min_boost_length (5)
        food = [(500, 300)]
        state = snake.get_state([], food)
        assert float(state[57]) == 0.0

    def test_boost_state_is_last_feature(self):
        """Boost state is the last element of the state vector (index 57)."""
        snake = _make_long_snake(10)
        state = snake.get_state([], [(500, 300)])
        assert state.shape == (58,)
        # Index 57 is the last element
        assert float(state[57]) == 1.0  # length 10 >= min 5


class TestBoostRespawn:
    """Test boost state resets on respawn."""

    def test_is_boosting_false_after_respawn(self):
        snake = _make_long_snake(10)
        snake.is_boosting = True
        snake.boost_frames = 2
        snake.respawn((100, 100))
        assert snake.is_boosting is False

    def test_boost_frames_zero_after_respawn(self):
        snake = _make_long_snake(10)
        snake.is_boosting = True
        snake.boost_frames = 2
        snake.respawn((100, 100))
        assert snake.boost_frames == 0

    def test_respawn_resets_length_to_one(self):
        """After respawn, snake has length 1 (can't boost)."""
        snake = _make_long_snake(10)
        snake.respawn((100, 100))
        assert snake.length == 1
        # With length 1, boost state should be 0.0
        state = snake.get_state([], [(200, 200)])
        assert float(state[57]) == 0.0


class TestBoostInit:
    """Test initial boost state on snake creation."""

    def test_initial_is_boosting_false(self):
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        assert snake.is_boosting is False

    def test_initial_boost_frames_zero(self):
        snake = Snake(0, (255, 0, 0), (400, 300), 10, 800, 600)
        assert snake.boost_frames == 0
