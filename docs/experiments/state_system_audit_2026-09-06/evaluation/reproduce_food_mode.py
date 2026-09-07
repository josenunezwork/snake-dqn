#!/usr/bin/env python3
"""Reproduce the train/eval food-replacement branch difference without a game."""

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

from src.simd_env.batch_sim import BatchSim, BatchSimConfig  # noqa: E402


def run_branch(train_mode: bool) -> dict[str, object]:
    """Consume one corpse pellet while ambient food is already at its cap."""
    config = BatchSimConfig(
        num_envs=1,
        num_snakes=1,
        game_width=100,
        game_height=100,
        segment_size=10,
        wall_thickness=10,
        initial_food=0,
        max_food=1,
        mechanics_version=2,
        frame_rate=1,
    )
    sim = BatchSim(config, seeds=[17], train_mode=train_mode, allow_respawn=True)
    head = tuple(map(int, sim.heads()[0, 0]))
    ambient = (5, 5) if head != (5, 5) else (6, 5)

    sim.food_cells[0] = [ambient, head]
    sim.food_set[0] = {ambient, head}
    sim.corpse_cells[0] = {head}
    sim._rebuild_traversed_from_heads()
    ate = bool(sim._consume_food()[0, 0])

    return {
        "ate_corpse": ate,
        "food_after": [list(cell) for cell in sim.food_cells[0]],
        "corpse_after": [list(cell) for cell in sorted(sim.corpse_cells[0])],
        "ambient_count_after": int(sim._ambient_count(0)),
        "total_food_after": len(sim.food_cells[0]),
    }


def main() -> None:
    """Write the controlled branch comparison beside this script."""
    data = {
        "scenario": "ambient already at max=1; hero consumes cap-exempt corpse pellet",
        "train_mode_true": run_branch(True),
        "train_mode_false": run_branch(False),
    }
    output = Path(__file__).with_name("food_mode_repro.json")
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
