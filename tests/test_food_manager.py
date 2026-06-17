"""Tests for food spawning and consumption management."""

import pytest

from src.game.food_manager import FoodManager
from src.game.game_logic import GameLogic
from src.game.snake import Snake

pytestmark = pytest.mark.usefixtures("setup_config")


class TestFoodSpawnPositions:
    """Tests for food placement quality."""

    def test_add_food_rejects_overlapping_food(self):
        food_manager = FoodManager(
            game_width=200,
            game_height=200,
            max_food=2,
            initial_food=0,
            segment_size=10,
            wall_thickness=10,
        )

        assert food_manager.add_food((50, 50)) is True
        assert food_manager.add_food((55, 50)) is False
        assert food_manager.add_food((70, 50)) is True
        assert food_manager.food == [(50, 50), (70, 50)]

    def test_initial_food_avoids_duplicate_positions(self, monkeypatch):
        positions = iter([(50, 50), (50, 50), (70, 50)])

        def next_position(food_manager):
            return next(positions)

        monkeypatch.setattr(FoodManager, "_get_random_position", next_position)

        food_manager = FoodManager(
            game_width=200,
            game_height=200,
            max_food=2,
            initial_food=2,
            segment_size=10,
            wall_thickness=10,
        )

        assert food_manager.food == [(50, 50), (70, 50)]

    def test_spawn_avoids_existing_food_when_avoiding_snakes(self, monkeypatch):
        positions = iter([(50, 50), (50, 50), (80, 50)])
        snake = Snake(0, (255, 0, 0), (120, 120), 10, 200, 200)

        def find_empty_position(width, height, snakes):
            assert snakes == [snake]
            return next(positions)

        monkeypatch.setattr(
            GameLogic,
            "find_empty_position",
            staticmethod(find_empty_position),
        )

        food_manager = FoodManager(
            game_width=200,
            game_height=200,
            max_food=2,
            initial_food=0,
            segment_size=10,
            wall_thickness=10,
        )
        food_manager.food = [(50, 50)]

        spawned = food_manager.spawn(1, [snake])

        assert spawned == 1
        assert food_manager.food == [(50, 50), (80, 50)]
