"""B8 death census helpers (research/b8_death_census_20261014/census.py): pure analysis
functions and the deep-copied-world clone the one-frame counterfactual relies on."""

from __future__ import annotations

import numpy as np

from research.b8_death_census_20261014 import census
from research.redesign_scope_20261007 import grid_h5000_identity as gid
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim


def test_trailing_run():
    assert census.trailing_run([]) == 0
    assert census.trailing_run([True, False, True, True]) == 2
    assert census.trailing_run([False]) == 0
    assert census.trailing_run([True, True]) == 2


def test_within_k_ends_at_the_fatal_decision():
    got = census.within_k([True, False, False, False], k_max=4)
    assert got == {1: False, 2: False, 3: False, 4: True}
    assert census.within_k([False, False, True], k_max=2) == {1: True, 2: True}


def test_length_hazard_buckets():
    length = np.array([[50, 150, 150, 0], [450, 450, 1300, 0]])
    alive = np.array([[True, True, True, False], [True, True, True, False]])
    hz = census.length_hazard(length, alive, [150, 1300])
    assert hz["0-100"] == {"decisions": 1, "deaths": 0, "per_1000": 0.0}
    assert hz["100-200"]["decisions"] == 2 and hz["100-200"]["deaths"] == 1
    assert hz["1200-inf"] == {"decisions": 1, "deaths": 1, "per_1000": 1000.0}
    assert hz["200-400"]["per_1000"] is None


def test_clone_steps_identically_and_independently():
    cfg = BatchSimConfig(
        num_envs=3,
        num_snakes=6,
        game_width=400,
        game_height=300,
        initial_food=60,
        max_food=80,
        mechanics_version=2,
        max_capacity=1200,
        frame_rate=3,
    )
    sim = GridBatchSim(cfg, seeds=[11, 12, 13], train_mode=False, allow_respawn=True)
    rng = np.random.default_rng(0)
    deaths = 0
    for _ in range(300):
        actions = rng.integers(0, 6, size=(sim.E, sim.S))
        c = census.clone_sim(sim)
        assert c._owner.base is c._owner_pad and c._food.base is c._food_pad
        c.step(actions)
        sim.step(actions)
        deaths += int((sim.get_death_cause() != 0).sum())
        for e in range(sim.E):
            assert gid.env_digest(c, e) == gid.env_digest(sim, e)
    assert deaths > 0
    # an alternative action on the clone never touches the original
    before = [gid.env_digest(sim, e) for e in range(sim.E)]
    c = census.clone_sim(sim)
    c.step(np.zeros((sim.E, sim.S), dtype=np.int64))
    assert [gid.env_digest(sim, e) for e in range(sim.E)] == before
