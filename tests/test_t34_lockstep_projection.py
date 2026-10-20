"""Adversarial cache projection across unequal world mirror capacities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.t34_exit_b8.lockstep import _copy_environment, _merge_prepared_worlds
from research.t34_exit_b8.trajectory import build_world, make_world_rows, prepare_world
from research.t34_shadow.shadow import clone_world_exact, full_digest


def test_food_mirror_keeps_independent_growth_and_intermediate_tail_writes():
    inventory = json.loads(Path("research/t34_exit_b8/INPUTS_REQUIRED.json").read_text())
    checkpoint = next(
        Path(row["local_path"]) for row in inventory["inputs"] if row["name"] == "B8_s3_final.pth"
    )
    if not checkpoint.is_file():
        pytest.skip("pinned checkpoint inventory unavailable")
    rows = make_world_rows(
        {"namespace": "t34-shadow-mirror-tail/v1", "mixes": ("scripted", "frozen")}, 2
    )
    worlds = [prepare_world(build_world(row, checkpoint)) for row in rows]
    # Give the second world a larger historical capacity and distinctive unused
    # bytes. Padding a batch must preserve this history without spreading it.
    old = worlds[1].sim._food_array
    wider = np.zeros((1, old.shape[1] * 2 + 17, 2), dtype=old.dtype)
    wider[:, : old.shape[1]] = old
    wider[:, old.shape[1] :] = 731
    worlds[1].sim._food_array = wider
    serial = [clone_world_exact(world) for world in worlds]
    merged = _merge_prepared_worlds(worlds)

    def remove_twice(sim, env):
        # Two within-frame corpse-pellet removals leave the first refresh's
        # shifted data in the now-unused tail after the second refresh.
        for _ in range(2):
            sim.food_cells[env].pop(0)
            sim._sync_food_array(env)

    for env, world in enumerate(serial):
        remove_twice(world.sim, 0)
        remove_twice(merged.sim, env)
    # Exercise a per-row reservation which fits the merged width but grows the
    # smaller serial world's cache, even though no final pellet count grows.
    requested = serial[0].sim._food_array.shape[1] + 1
    serial[0].sim._spawn(0, requested - len(serial[0].sim.food_cells[0]))
    merged.sim._spawn(0, requested - len(merged.sim.food_cells[0]))
    for env, world in enumerate(worlds):
        _copy_environment(merged, env, world)
        assert world.sim._food_array.tobytes() == serial[env].sim._food_array.tobytes()
        assert world.sim._food_array.shape == serial[env].sim._food_array.shape
        assert full_digest(world) == full_digest(serial[env])
