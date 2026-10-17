"""Serving-side frame-rate ceilings for expensive live policies.

The ceiling is deliberately a scheduling control.  It never changes policy
inputs, inference batching, action selection, or game mechanics.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.game.ai_snake import AISnake
from src.model.obs_spec import EGO2S_OBS_SPECS

DEFAULT_SERVER_SPEED_CEILING = 120.0


def driven_ai_count(snakes: Iterable[object], policy: object) -> int:
    """Count the persistent roster driven by ``policy``.

    Dead AI snakes remain in the roster because Watch and Play respawn them.
    A ``HumanSnake`` is excluded by the ``AISnake`` type check.
    """
    return sum(
        isinstance(snake, AISnake) and getattr(snake, "policy", None) is policy for snake in snakes
    )


def speed_ceiling(obs_spec: str, driven_count: int) -> float:
    """Return the server-side frames-per-second ceiling for this served roster."""
    if obs_spec not in EGO2S_OBS_SPECS:
        return DEFAULT_SERVER_SPEED_CEILING
    if driven_count <= 6:
        return 24.0
    if driven_count <= 12:
        return 12.0
    return max(1.0, float(144 // driven_count))
