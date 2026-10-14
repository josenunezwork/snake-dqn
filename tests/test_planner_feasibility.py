"""Planner feasibility helpers (research/planner_feasibility_20261014/planner.py): the
single-env world fork the lookahead rollouts rely on."""

from __future__ import annotations

import numpy as np
import pytest

from research.planner_feasibility_20261014 import planner
from research.redesign_scope_20261007 import grid_h5000_identity as gid
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim


def _sim() -> GridBatchSim:
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
    return GridBatchSim(cfg, seeds=[11, 12, 13], train_mode=False, allow_respawn=True)


def _grids_equal(a: GridBatchSim, i: int, b: GridBatchSim, e: int) -> bool:
    names = ("_owner_pad", "_slot_pad", "_food_pad", "_csnake", "_cfood", "_amb_count")
    return all(np.array_equal(getattr(a, n)[i], getattr(b, n)[e]) for n in names)


@pytest.mark.parametrize("k", [1, 4])
def test_fork_env_steps_like_its_source_env(k: int) -> None:
    sim = _sim()
    rng = np.random.default_rng(0)
    for _ in range(40):  # warm up some bodies and food
        sim.step(rng.integers(0, 6, size=(sim.E, sim.S)))
    e = 1
    fork = planner.fork_env(sim, e, k)
    assert fork.E == k and fork.cfg.num_envs == k
    assert fork._owner.base is fork._owner_pad
    deaths = 0
    for _ in range(250):
        actions = rng.integers(0, 6, size=(sim.E, sim.S))
        sim.step(actions)
        fork.step(np.repeat(actions[e : e + 1], k, axis=0))
        deaths += int((sim.get_death_cause()[e] != 0).sum())
        for i in range(k):
            assert gid.env_digest(fork, i) == gid.env_digest(sim, e)
            assert _grids_equal(fork, i, sim, e)
    fork.check_grid_invariants()
    assert deaths > 0


def test_fork_env_is_independent_of_its_source() -> None:
    sim = _sim()
    before = [gid.env_digest(sim, e) for e in range(sim.E)]
    fork = planner.fork_env(sim, 2, 3)
    for _ in range(30):
        fork.step(np.zeros((3, sim.S), dtype=np.int64))
    assert [gid.env_digest(sim, e) for e in range(sim.E)] == before
    # copies of one env diverge independently under different actions
    fork = planner.fork_env(sim, 0, 2)
    fork.step(np.array([[0] * sim.S, [2] * sim.S], dtype=np.int64))
    assert gid.env_digest(fork, 0) != gid.env_digest(fork, 1)


def test_fork_env_rejects_unknown_state() -> None:
    sim = _sim()
    sim.some_new_cache = object()
    with pytest.raises(TypeError):
        planner.fork_env(sim, 0, 1)


def test_reseed_private_changes_only_the_fork_rng() -> None:
    sim = _sim()
    state = sim._rngs[0]._rng.getstate()
    fork = planner.fork_env(sim, 0, 2)
    planner.reseed_private(fork, "x")
    assert sim._rngs[0]._rng.getstate() == state
    assert fork._rngs[0]._rng.getstate() != state
    assert fork._rngs[0]._rng.getstate() != fork._rngs[1]._rng.getstate()
