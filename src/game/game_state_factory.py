"""Factory helpers for building headless training / eval GameStates.

Extracted from main.py so the eval scripts (tournament_eval, ensemble_eval) and
the training loop share one home for game-state construction, instead of
importing library code from the CLI entry point (an inverted dependency).
"""

from typing import Any, Dict

from src.core.game_config import GameConfig
from src.game.game_state import GameState


def get_training_game_settings(curriculum=None) -> Dict[str, Any]:
    """Resolve the game settings for the current training phase."""
    if curriculum is None:
        return {
            "num_snakes": GameConfig.NUM_SNAKES,
            "food_multiplier": 1.0,
            "board_scale": 1.0,
        }
    return curriculum.get_game_settings()


def configure_eval_game_state(game_state: GameState) -> None:
    """Make a headless GameState greedy and inference-only for evaluation."""
    policies = []
    shared_policy = getattr(game_state, "_shared_policy", None)
    if shared_policy is not None:
        policies.append(shared_policy)

    for snake in getattr(game_state, "snakes", []):
        policy = getattr(snake, "policy", None)
        if policy is not None and policy not in policies:
            policies.append(policy)
        if hasattr(snake, "actor_epsilon"):
            snake.actor_epsilon = 0.0
        if hasattr(snake, "current_epsilon"):
            snake.current_epsilon = 0.0

    for policy in policies:
        if hasattr(policy, "epsilon"):
            policy.epsilon = 0.0
        if hasattr(policy, "training"):
            policy.training = False
        network = getattr(policy, "dqn", None)
        if hasattr(network, "eval"):
            network.eval()
        target_network = getattr(policy, "target_dqn", None)
        if hasattr(target_network, "eval"):
            target_network.eval()


def create_training_game_state(
    curriculum=None, shared_policy=None, eval_mode: bool = False
) -> GameState:
    """Create a headless training GameState using the active curriculum settings."""
    settings = get_training_game_settings(curriculum)
    num_snakes = int(settings["num_snakes"])

    game_state = GameState(
        headless=True,
        snake_policies=["apex"] * num_snakes,
        num_snakes=num_snakes,
        shared_policy=shared_policy,
        food_multiplier=float(settings["food_multiplier"]),
        board_scale=float(settings["board_scale"]),
    )
    if eval_mode:
        configure_eval_game_state(game_state)
    return game_state
