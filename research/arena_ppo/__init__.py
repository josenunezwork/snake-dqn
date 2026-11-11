"""From-scratch GPU-resident full-arena PPO self-play package (Stage 1)."""

from __future__ import annotations

__all__ = [
    "ARENA_CHANNELS",
    "ARENA_GRID_H",
    "ARENA_GRID_W",
    "EGO_CROP_SIZE",
    "SCALAR_DIM",
    "WORLD_ARENA_CHANNELS",
]

ARENA_GRID_H = 83
ARENA_GRID_W = 145
EGO_CROP_SIZE = 47
SCALAR_DIM = 6
ARENA_CHANNELS = (
    "own_ttl",
    "enemy_ttl",
    "own_head_curr",
    "own_head_prev",
    "enemy_head_curr",
    "enemy_head_prev",
    "enemy_head_dx",
    "enemy_head_dy",
    "food_ambient",
    "food_corpse",
    "wall",
    "enemy_length",
    "boost_flags",
)
WORLD_ARENA_CHANNELS = (
    "body_ttl",
    "head_curr",
    "head_dx",
    "head_dy",
    "head_length",
    "boost_flags",
    "food_ambient",
    "food_corpse",
    "wall",
)
