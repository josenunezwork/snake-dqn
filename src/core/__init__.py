"""Core infrastructure modules."""

from .config_loader import (
    apply_config_to_game_config,
    get_config_summary,
    load_and_initialize_config,
    load_config,
)
from .game_config import (
    ApexSettings,
    AppConfig,
    CheckpointSettings,
    CurriculumSettings,
    GameConfig,
    GameSettings,
    NetworkSettings,
    RewardSettings,
    StateIndices,
    TrainingSettings,
    get_config,
    initialize_config,
)


def __getattr__(name: str):
    if name == "DeviceManager":
        from .device_manager import DeviceManager

        return DeviceManager

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Device management
    "DeviceManager",
    # Game configuration
    "AppConfig",
    "GameConfig",
    "GameSettings",
    "NetworkSettings",
    "TrainingSettings",
    "RewardSettings",
    "CheckpointSettings",
    "ApexSettings",
    "CurriculumSettings",
    "StateIndices",
    "initialize_config",
    "get_config",
    # Config loading utilities
    "load_config",
    "load_and_initialize_config",
    "apply_config_to_game_config",
    "get_config_summary",
]
