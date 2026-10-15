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


def _episode(worlds: int, base_survival: object) -> dict:
    return {
        "worlds": worlds,
        "depth": 32 if base_survival is not None else 16,
        "survival": 0.75,
        "base_survival": base_survival,
        "mass": 700.0,
        "base_mass": 600.0,
        "paired_mass_delta": 100.0,
        "deaths_avoided": 8,
        "deaths_new": 0,
        "decisions": 1000,
        "fires": 10,
        "switches": 2,
        "planner_ms_per_decision": 1.5,
        "base_policy_ms_per_decision": 0.5,
    }


def _cl_row(death: int, survived: bool) -> dict:
    return {
        "kind": "closed_loop",
        "depth": 32,
        "death": death,
        "trap_run": 10 * death,
        "length_at_death": 900,
        "survived_past_T": survived,
        "censored": survived,
        "extra_decisions": 200 if survived else 0,
        "fires": 1,
        "switches": 1,
        "decisions": 10,
        "planner_ms_per_decision": 5.0,
    }


def test_summary_pools_episodes_and_matches_closed_loop() -> None:
    eps = {"smoke": _episode(11, None), "a": _episode(32, 0.5), "b": _episode(32, 0.5)}
    pooled = planner.episode_pooled(eps)
    assert list(pooled) == ["d32"]  # the smoke run is excluded
    assert pooled["d32"]["worlds"] == 2 * planner.WORLDS
    assert pooled["d32"]["deaths_avoided"] == 16 and pooled["d32"]["fire_rate"] == 0.01
    probes = {
        "oracle_oracle": [{"arm": "b8_s0", "results": [_cl_row(1, True), _cl_row(5, True)]}],
        "student_private": [{"arm": "b8_s0", "results": [_cl_row(1, True), _cl_row(5, False)]}],
    }
    m = planner.closed_loop_matched(probes)["d32"]
    assert m["all"]["oracle"]["survived_past_T"] == 2
    assert m["all"]["honest"]["survived_past_T"] == 1
    assert m["trap_run>30"]["honest"]["n"] == 1
    lat = planner.latency_vs_frame_budget({"episode_pooled": pooled})
    assert lat["frame_period_ms_default"] == 1e3 / planner.LIVE_DEFAULT_FPS
    assert all(v["fits_default_frame"] for v in lat["per_decision_ms"].values())
