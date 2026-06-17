"""Lightweight package exports for snake simulation and logic."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    # Snake entities
    "Snake",
    "AISnake",
    "HumanSnake",
    # Game logic and state
    "GameLogic",
    "GameState",
    "FoodManager",
    "SnakeFactory",
    # Action constants
    "TURN_LEFT",
    "TURN_STRAIGHT",
    "TURN_RIGHT",
]

_LAZY_EXPORTS = {
    "Snake": ("src.game.snake", "Snake"),
    "AISnake": ("src.game.ai_snake", "AISnake"),
    "HumanSnake": ("src.game.human_snake", "HumanSnake"),
    "GameLogic": ("src.game.game_logic", "GameLogic"),
    "GameState": ("src.game.game_state", "GameState"),
    "FoodManager": ("src.game.food_manager", "FoodManager"),
    "SnakeFactory": ("src.game.snake_factory", "SnakeFactory"),
    "TURN_LEFT": ("src.game.game_logic", "TURN_LEFT"),
    "TURN_STRAIGHT": ("src.game.game_logic", "TURN_STRAIGHT"),
    "TURN_RIGHT": ("src.game.game_logic", "TURN_RIGHT"),
}

if TYPE_CHECKING:
    from src.game.ai_snake import AISnake as AISnake
    from src.game.food_manager import FoodManager as FoodManager
    from src.game.game_logic import TURN_LEFT as TURN_LEFT
    from src.game.game_logic import TURN_RIGHT as TURN_RIGHT
    from src.game.game_logic import TURN_STRAIGHT as TURN_STRAIGHT
    from src.game.game_logic import GameLogic as GameLogic
    from src.game.game_state import GameState as GameState
    from src.game.human_snake import HumanSnake as HumanSnake
    from src.game.snake import Snake as Snake
    from src.game.snake_factory import SnakeFactory as SnakeFactory


def __getattr__(name: str) -> object:
    """Load package-level exports only when callers ask for them."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
