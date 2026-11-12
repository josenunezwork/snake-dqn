"""Game V3.1: GPU-first deterministic fixed-point slither-like simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from research.game_v3.config import (
    MAX_SNAKES,
    NUM_ACTIONS,
    NUM_TURN_LEVELS,
    OCC_OWNER_BITS,
    OCC_OWNER_MASK,
    GameV3Config,
)
from research.game_v3.obs import (
    NUM_RASTER_CHANNELS,
    NUM_SCALAR_FEATURES,
    LearnerIndex,
    ObservationBatch,
    ReferenceObservation,
    allocate_obs_batch,
    build_obs,
    build_obs_reference,
)
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import STATE_FIELDS, GameV3Sim, StateField, StepInfo

if TYPE_CHECKING:
    from research.game_v3.reference import ReferenceStepInfo, ReferenceWorld

__all__ = [
    "GameV3Config",
    "GameV3Sim",
    "LearnerIndex",
    "MAX_SNAKES",
    "NUM_ACTIONS",
    "NUM_RASTER_CHANNELS",
    "NUM_SCALAR_FEATURES",
    "NUM_TURN_LEVELS",
    "OCC_OWNER_BITS",
    "OCC_OWNER_MASK",
    "ObservationBatch",
    "ReferenceObservation",
    "ReferenceStepInfo",
    "ReferenceWorld",
    "STATE_FIELDS",
    "StateField",
    "StepInfo",
    "allocate_obs_batch",
    "build_obs",
    "build_obs_reference",
    "food_seeking_actions",
    "random_safe_actions",
]


def __getattr__(name: str) -> Any:
    if name in ("ReferenceStepInfo", "ReferenceWorld"):
        from research.game_v3 import reference

        return getattr(reference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
