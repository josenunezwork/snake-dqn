#!/usr/bin/env python3
"""Reproduce evaluator metric, respawn-timing, and serving-config evidence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.core.config_loader import load_and_initialize_config, load_config  # noqa: E402
from src.core.game_config import GameConfig  # noqa: E402
from src.game.snake import Snake  # noqa: E402
from src.scripts.tournament_eval import DEFAULT_CONFIG  # noqa: E402
from src.simd_env.eval_engine import _config_from_game_config  # noqa: E402
from web.backend.session import _config_for  # noqa: E402


OUT = Path(__file__).resolve().parent


def write_json(name: str, value: object) -> None:
    """Write stable, human-readable JSON beside this script."""
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def config_snapshot(path: str) -> dict[str, object]:
    """Return the environment fields that affect gate/serve comparability."""
    config = load_config(path)
    return {
        "path": path,
        "mechanics_version": int(config.game.mechanics_version),
        "frame_rate": int(config.game.frame_rate),
        "num_snakes": int(config.game.num_snakes),
        "max_frames": int(config.game.max_frames),
        "reward_version": int(config.rewards.version),
    }


def main() -> None:
    """Generate the two configuration and metric evidence files."""
    os.chdir(ROOT)
    load_and_initialize_config(DEFAULT_CONFIG)
    simd_config = _config_from_game_config(num_snakes=6, gamma=0.99)

    snake = Snake(
        0,
        (255, 0, 0),
        (100, 100),
        int(GameConfig.SEGMENT_SIZE),
        int(GameConfig.WIDTH),
        int(GameConfig.HEIGHT),
    )
    before = {"logical_length": int(snake.length), "segment_count": len(snake.segments)}
    snake.grow()
    after = {"logical_length": int(snake.length), "segment_count": len(snake.segments)}

    write_json(
        "eval_contract_repro.json",
        {
            "config": DEFAULT_CONFIG,
            "active_live": {
                "frame_rate": int(GameConfig.FRAME_RATE),
                "mechanics_version": int(GameConfig.MECHANICS_VERSION),
                "arena_type": str(GameConfig.ARENA_TYPE),
            },
            "derived_simd": {
                "frame_rate": int(simd_config.frame_rate),
                "mechanics_version": int(simd_config.mechanics_version),
                "arena_type": str(simd_config.arena_type),
            },
            "growth_mass_units": {
                "before": before,
                "immediately_after_grow": after,
                "live_tournament_uses": "segment_count",
                "simd_tournament_uses": "logical_length",
            },
        },
    )

    # GameSession treats an absent flag as mechanics-v2 enabled. Make this probe
    # independent of the invoking shell's optional override.
    os.environ.pop("SNAKE_MECHANICS_V2", None)
    serving_config = _config_for(61)
    write_json(
        "config_authority_repro.json",
        {
            "environment_flag": "SNAKE_MECHANICS_V2 unset => enabled",
            "tournament_default": config_snapshot(DEFAULT_CONFIG),
            "web_default_for_61_or_raster": config_snapshot(serving_config),
        },
    )


if __name__ == "__main__":
    main()
