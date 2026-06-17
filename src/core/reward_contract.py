"""Helpers for recording the reward configuration that produced replay targets."""

from __future__ import annotations

from dataclasses import fields

from src.core.game_config import RewardSettings, get_config

REWARD_CONTRACT_METADATA_KEY = "generation.reward_contract"


def current_reward_contract() -> dict[str, float | int]:
    """Return the current reward settings that affect persisted replay rewards."""
    rewards = get_config().rewards
    return {field.name: getattr(rewards, field.name) for field in fields(RewardSettings)}
