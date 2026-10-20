"""Focused contract tests for T34's reusable shadow state machinery."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest

from research.t34_shadow.identity import _value_digest, game_state_digest
from research.t34_shadow.shadow import (
    ShadowDeadlineExceeded,
    ShadowTrajectory,
    clone_world_exact,
    full_digest,
)


class _World:
    def __init__(self) -> None:
        self.sim = SimpleNamespace(
            E=1,
            S=1,
            cap=4,
            seg_count=np.array([[2]]),
            head_ptr=np.array([[1]]),
            bodies=np.array([[[[4, 4], [3, 4], [0, 0], [0, 0]]]]),
            food_cells=[[(2, 2), (1, 1)]],
            corpse_cells=[set()],
            _rngs=[SimpleNamespace(_rng=random.Random(17))],
            frame=np.array([0]),
            alive=np.array([[True]]),
            length=np.array([[2]]),
            _policy_phase_prepared=True,
            _policy_step_in_progress=False,
        )
        self.runtime = SimpleNamespace(memory={"v8": np.array([1, 2])})
        self.hero = SimpleNamespace(runtime=self.runtime)
        self.opponents = [SimpleNamespace(runtime=self.runtime, carry=np.array([3]))]

    def clone(self):
        return copy.deepcopy(self)


def test_digest_ignores_immutable_alias_topology_and_nested_elapsed_timers():
    shared = SimpleNamespace(lam=8.0, reference_seconds_total=1.0)
    independent = [copy.deepcopy(shared), copy.deepcopy(shared)]
    independent[0].reference_seconds_total = 2.0
    independent[1].reference_seconds_total = 3.0
    assert _value_digest([shared, shared]) == _value_digest(independent)
    independent[1].lam = 9.0
    assert _value_digest([shared, shared]) != _value_digest(independent)


def _step(world):
    world.sim.frame += 1
    world.sim.bodies[0, 0, 1] += 1
    return np.array([[1]], dtype=np.int64), np.ones((1, 1, 6), dtype=bool), None


def test_full_digest_includes_ordered_food_rng_runtime_and_global_random():
    world = _World()
    baseline = full_digest(world, checkpoint="abc")
    world.sim.food_cells[0].reverse()
    assert full_digest(world, checkpoint="abc") != baseline
    world.sim.food_cells[0].reverse()
    world.runtime.memory["v8"][0] = 99
    assert full_digest(world, checkpoint="abc") != baseline
    world.runtime.memory["v8"][0] = 1
    world.sim._rngs[0]._rng.random()
    assert full_digest(world, checkpoint="abc") != baseline


def test_catch_up_uses_its_isolated_global_random_stream():
    def global_step(world):
        random.random()
        return _step(world)

    shadow = ShadowTrajectory(depth=1, stepper=global_step)
    source = _World()
    random.seed(456)
    ambient = random.getstate()
    shadow.fork(source)
    shadow.catch_up()
    assert random.getstate() == ambient


def test_clone_normalizes_post_step_lifecycle_to_source():
    source = _World()
    delattr(source.sim, "_policy_phase_prepared")
    source.sim._policy_step_in_progress = False
    clone = clone_world_exact(source)
    assert not hasattr(clone.sim, "_policy_phase_prepared")
    assert clone.sim._policy_step_in_progress is False


def test_clone_isolation_avoids_nb_identity_cache_and_policy_carry_aliases():
    source = _World()
    source.runtime.featurizer = SimpleNamespace(
        _food_sim_id=id(source.sim),
        _food_lists={0: [(2, 2)]},
        _food_buf=np.ones((1, 1, 2), dtype=np.int64),
        _food_n=np.ones(1, dtype=np.int64),
    )
    source.opponents[0].veto_counters = {0: {"called": 1}}
    source.runtime.free_space_featurizer = source.runtime.featurizer
    clone = clone_world_exact(source)
    assert full_digest(source) == full_digest(clone)
    assert clone.runtime.featurizer._food_sim_id is None
    assert clone.runtime.free_space_featurizer is clone.runtime.featurizer
    clone.opponents[0].veto_counters[0]["called"] = 2
    assert source.opponents[0].veto_counters[0]["called"] == 1


def test_shadow_confirms_then_invalidates_on_digest_mismatch():
    source = _World()
    shadow = ShadowTrajectory(depth=2, checkpoint="abc", stepper=_step)
    shadow.fork(source)
    assert shadow.catch_up() == 2
    live = clone_world_exact(source)
    _step(live)
    assert shadow.confirm(live)
    live.sim.frame += 99
    assert not shadow.confirm(live)
    assert shadow.invalid_reason == "full_digest_mismatch"
    assert not shadow.confirmed


def test_shadow_hard_deadline_is_checked_inside_step_loop():
    shadow = ShadowTrajectory(depth=2, stepper=_step)
    shadow.fork(_World())
    with pytest.raises(ShadowDeadlineExceeded):
        shadow.catch_up(deadline=0.0)


def test_live_game_digest_tracks_segments_and_tuple_heading():
    snake = SimpleNamespace(
        id=0,
        is_alive=True,
        segments=[(1, 2), (1, 3)],
        length=2,
        direction=(1, 0),
        boost_frames=0,
        frames_since_food=0,
    )
    game = SimpleNamespace(
        frame=1,
        snakes=[snake],
        food_manager=SimpleNamespace(food=[(4, 5)]),
        frame_death_causes={},
    )
    baseline = game_state_digest(game)
    snake.segments[0] = (2, 2)
    assert game_state_digest(game) != baseline
