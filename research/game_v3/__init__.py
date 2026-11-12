"""Game V3.1: GPU-first deterministic fixed-point slither-like simulator."""

from research.game_v3.config import NUM_ACTIONS, NUM_TURN_LEVELS, GameV3Config
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
from research.game_v3.reference import ReferenceStepInfo, ReferenceWorld
from research.game_v3.scripted import food_seeking_actions, random_safe_actions
from research.game_v3.sim import STATE_FIELDS, GameV3Sim, StateField, StepInfo

__all__ = [
    "GameV3Config",
    "GameV3Sim",
    "LearnerIndex",
    "NUM_ACTIONS",
    "NUM_RASTER_CHANNELS",
    "NUM_SCALAR_FEATURES",
    "NUM_TURN_LEVELS",
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
