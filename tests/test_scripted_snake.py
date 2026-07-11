"""Tests for scripted anchor snakes (greedy_food / random_safe evaluation baselines)."""

import pytest

from src.game.game_logic import TURN_LEFT, TURN_RIGHT, TURN_STRAIGHT
from src.game.scripted_snake import SCRIPTED_KINDS, ScriptedSnake
from src.game.snake import Snake
from src.game.snake_factory import SnakeFactory

BOARD_W = 800
BOARD_H = 600
SEG = 10


class FakePolicy:
    """Stand-in policy so GameState can build its default AI roster."""

    epsilon = 0.0


def make_scripted(kind, pos, direction=(1, 0), seed=None, segments=None):
    """Create a ScriptedSnake at a known position on the standard test board."""
    snake = ScriptedSnake(0, (255, 0, 0), pos, SEG, BOARD_W, BOARD_H, kind=kind, seed=seed)
    snake.direction = direction
    if segments is not None:
        snake.segments = list(segments)
        snake.length = len(segments)
    return snake


def make_enemy(sid, segments):
    """Create a plain Snake whose body acts as an obstacle wall.

    The first segment is the head and the last is a dummy tail (the collision
    simulation drops the last body segment), so wrap real wall cells with two
    far-away positions.
    """
    snake = Snake(sid, (0, 255, 0), segments[0], SEG, BOARD_W, BOARD_H)
    snake.segments = list(segments)
    snake.length = len(segments)
    return snake


def boxed_in_world():
    """Hero at (100, 100) facing east with all three moves blocked by enemy body."""
    hero_pos = (100, 100)
    enemy = make_enemy(
        1,
        [(500, 500), (110, 100), (100, 90), (100, 110), (500, 510)],
    )
    return hero_pos, [enemy]


class TestGreedyFood:
    """Unit tests for the greedy_food behavior on constructed mini-scenarios."""

    def test_moves_straight_toward_food_ahead(self, setup_config):
        snake = make_scripted("greedy_food", (200, 200))
        snake.update([], [(300, 200)])
        assert snake.last_action == TURN_STRAIGHT
        assert snake.direction == (1, 0)
        assert snake.head == (210, 200)

    def test_fatal_wall_ahead_is_vetoed_despite_food_ahead(self, setup_config):
        # Straight would leave the arena; food above pulls the tie toward left.
        snake = make_scripted("greedy_food", (790, 300))
        snake.update([], [(790, 200)])
        assert snake.last_action == TURN_LEFT
        assert snake.direction == (0, -1)
        assert snake.head == (790, 290)

    def test_fatal_enemy_body_ahead_is_vetoed(self, setup_config):
        snake = make_scripted("greedy_food", (100, 100))
        enemy = make_enemy(1, [(500, 500), (110, 100), (500, 510)])
        snake.update([enemy], [(300, 100)])
        # Straight is fatal; left and right tie on food distance -> left wins.
        assert snake.last_action == TURN_LEFT
        assert snake.direction == (0, -1)

    def test_tiebreak_prefers_left_over_right(self, setup_config):
        # Food behind the snake: left and right candidate heads are equidistant.
        snake = make_scripted("greedy_food", (200, 200))
        snake.update([], [(100, 200)])
        assert snake.last_action == TURN_LEFT

    def test_no_food_goes_straight(self, setup_config):
        snake = make_scripted("greedy_food", (200, 200))
        snake.update([], [])
        assert snake.last_action == TURN_STRAIGHT
        assert snake.head == (210, 200)

    def test_all_fatal_still_picks_an_action_and_moves(self, setup_config):
        hero_pos, others = boxed_in_world()
        snake = make_scripted("greedy_food", hero_pos)
        snake.update(others, [(300, 100)])
        assert snake.last_action in (TURN_LEFT, TURN_STRAIGHT, TURN_RIGHT)
        assert snake.head != hero_pos

    def test_flood_fill_veto_rejects_small_pocket(self, setup_config):
        # Pocket: top-left arena corner (cells x,y in 0..40 -> 25 cells) sealed
        # by an enemy wall along x=50 and y=50 with a single entrance at
        # (20, 50). The hero (length 30) sits below the entrance facing up;
        # food inside the pocket makes straight the greedy-by-distance pick,
        # but reachable space (26 cells) < length (30), so it must turn away.
        wall = [(50, y) for y in range(0, 60, 10)]
        wall += [(x, 50) for x in (0, 10, 30, 40)]
        enemy = make_enemy(1, [(500, 500)] + wall + [(500, 510)])

        hero_body = [(20, 60 + 10 * i) for i in range(30)]
        snake = make_scripted("greedy_food", hero_body[0], direction=(0, -1), segments=hero_body)

        snake.update([enemy], [(20, 20)])
        assert snake.last_action == TURN_LEFT  # left/right tie on distance -> left
        assert snake.direction == (-1, 0)

    def test_flood_fill_veto_skipped_when_no_safer_alternative(self, setup_config):
        # Same pocket, but left and right are made fatal by extending the
        # enemy wall, leaving the (vetoable) pocket entrance as the only safe
        # move: the veto must not fire when no alternative exists.
        wall = [(50, y) for y in range(0, 60, 10)]
        wall += [(x, 50) for x in (0, 10, 30, 40)]
        wall += [(10, 60), (30, 60)]  # block left and right candidates
        enemy = make_enemy(1, [(500, 500)] + wall + [(500, 510)])

        hero_body = [(20, 60 + 10 * i) for i in range(30)]
        snake = make_scripted("greedy_food", hero_body[0], direction=(0, -1), segments=hero_body)

        snake.update([enemy], [(20, 20)])
        assert snake.last_action == TURN_STRAIGHT
        assert snake.head == (20, 50)


class TestRandomSafe:
    """Unit tests for the random_safe behavior."""

    def test_seeded_determinism(self, setup_config):
        snake_a = make_scripted("random_safe", (400, 300), seed=123)
        snake_b = make_scripted("random_safe", (400, 300), seed=123)
        trajectory_a, trajectory_b = [], []
        for _ in range(80):
            snake_a.update([], [])
            snake_b.update([], [])
            trajectory_a.append((snake_a.head, snake_a.direction))
            trajectory_b.append((snake_b.head, snake_b.direction))
        assert trajectory_a == trajectory_b

    def test_different_seeds_diverge(self, setup_config):
        snake_a = make_scripted("random_safe", (400, 300), seed=123)
        snake_b = make_scripted("random_safe", (400, 300), seed=7)
        trajectory_a, trajectory_b = [], []
        for _ in range(80):
            snake_a.update([], [])
            snake_b.update([], [])
            trajectory_a.append(snake_a.head)
            trajectory_b.append(snake_b.head)
        assert trajectory_a != trajectory_b

    def test_never_selects_fatal_action(self, setup_config):
        # Wall directly ahead: straight is fatal, so it must never be chosen.
        snake = make_scripted("random_safe", (790, 300), seed=42)
        for _ in range(30):
            assert snake._choose_action([], []) in (TURN_LEFT, TURN_RIGHT)

    def test_straight_when_nothing_is_safe(self, setup_config):
        hero_pos, others = boxed_in_world()
        snake = make_scripted("random_safe", hero_pos, seed=1)
        snake.update(others, [])
        assert snake.last_action == TURN_STRAIGHT
        assert snake.direction == (1, 0)

    def test_never_boosts(self, setup_config):
        snake = make_scripted("random_safe", (400, 300), seed=5)
        snake.length = 20  # Well above MIN_BOOST_LENGTH
        for _ in range(50):
            snake.update([], [])
            assert snake.is_boosting is False
            assert snake.last_action in (TURN_LEFT, TURN_STRAIGHT, TURN_RIGHT)


class TestFactoryAndContract:
    """Factory signature, learning exclusion, and package export."""

    @pytest.mark.parametrize("kind", SCRIPTED_KINDS)
    def test_factory_creates_scripted_snake(self, setup_config, kind):
        snake = SnakeFactory.create_scripted_snake(
            snake_id=3,
            color=(255, 165, 0),
            start_pos=(120, 140),
            kind=kind,
            game_width=BOARD_W,
            game_height=BOARD_H,
            seed=11,
        )
        assert isinstance(snake, ScriptedSnake)
        assert snake.kind == kind
        assert snake.id == 3
        assert snake.game_width == BOARD_W
        assert snake.game_height == BOARD_H
        # Standard snake surface used by the game loop and eval scripts.
        assert snake.is_alive
        assert snake.length == 1
        assert snake.segments == [(120, 140)]

    def test_factory_rejects_unknown_kind(self, setup_config):
        with pytest.raises(ValueError):
            SnakeFactory.create_scripted_snake(
                snake_id=0,
                color=(255, 0, 0),
                start_pos=(100, 100),
                kind="tree_search",
                game_width=BOARD_W,
                game_height=BOARD_H,
            )

    def test_excluded_from_learning_paths(self, setup_config):
        snake = SnakeFactory.create_scripted_snake(
            snake_id=0,
            color=(255, 0, 0),
            start_pos=(100, 100),
            kind="greedy_food",
            game_width=BOARD_W,
            game_height=BOARD_H,
        )
        assert snake.policy is None
        # GameState's reward/learning pass is hasattr-gated on this method.
        assert not hasattr(snake, "compute_reward_and_train")
        assert snake.total_reward == 0.0

    def test_package_export(self):
        from src.game import ScriptedSnake as exported

        assert exported is ScriptedSnake


class TestGameStateIntegration:
    """200-frame smoke test inside a real GameState."""

    def test_two_scripted_snakes_run_and_greedy_eats(self, setup_config):
        from src.core.game_config import (
            AppConfig,
            GameSettings,
            get_config,
            initialize_config,
        )
        from src.game.game_state import GameState

        original_config = get_config()
        config = AppConfig(
            game=GameSettings(
                width=400,
                height=300,
                num_snakes=2,
                initial_food=40,
                max_food=40,
            )
        )
        try:
            initialize_config(config)
            game_state = GameState(headless=True, num_snakes=2, shared_policy=FakePolicy())

            greedy = SnakeFactory.create_scripted_snake(
                snake_id=0,
                color=(255, 0, 0),
                start_pos=(100, 150),
                kind="greedy_food",
                game_width=400,
                game_height=300,
            )
            random_safe = SnakeFactory.create_scripted_snake(
                snake_id=1,
                color=(0, 255, 0),
                start_pos=(300, 150),
                kind="random_safe",
                game_width=400,
                game_height=300,
                seed=99,
            )
            game_state.snakes = [greedy, random_safe]
            game_state._shared_policy = None

            start_heads = (greedy.head, random_safe.head)
            for _ in range(200):
                game_state.update(train_mode=True, learn=False)

            assert game_state.frame == 200
            assert greedy.head != start_heads[0]
            assert random_safe.head != start_heads[1]
            assert greedy.length > 1  # greedy ate at least once
            assert game_state.episode_food_eaten >= 1
        finally:
            initialize_config(original_config)
