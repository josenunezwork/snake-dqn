"""Focused checks for O1 food-provenance side-state instrumentation."""

from __future__ import annotations

import copy
import random

import numpy as np

from research.t34_o1_harvest.provenance import BOOST_PELLET, DEATH_DROP, ProvenanceObserver
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim


def _sim(seed: int = 17) -> GridBatchSim:
    return GridBatchSim(
        BatchSimConfig(
            num_envs=1,
            num_snakes=2,
            game_width=200,
            game_height=120,
            initial_food=0,
            max_food=0,
            mechanics_version=2,
            min_boost_length=3,
            boost_length_cost_frames=1,
        ),
        seeds=[seed],
        train_mode=False,
        allow_respawn=False,
        jit=False,
    )


def test_provenance_tracks_trail_and_mass_balance() -> None:
    sim = _sim()
    observer = ProvenanceObserver(sim)
    # Grow from the native one-segment spawn to the legal boost threshold, then
    # a one-frame boost pays one segment and creates a v2 trail.  All food goes
    # through the simulator's normal mutation and consumption paths.
    sim.direction[0, 0] = 1
    for _ in range(2):
        head = sim.heads()[0, 0]
        sim._add_food(0, (int(head[0] + 1), int(head[1])), corpse=False)
        sim.step(np.array([[1, 1]], dtype=np.int64))
    head = sim.heads()[0, 0]
    sim._add_food(0, (int(head[0] + 1), int(head[1])), corpse=False)
    sim.step(np.array([[4, 1]], dtype=np.int64))
    observer.assert_accounting()
    ledger = observer.ledger()
    assert ledger["sources"]["ambient"] == 3
    assert ledger["boost_cost"] == 1
    assert any(source == BOOST_PELLET for source in observer._sources.values())


def test_bulk_death_drop_is_tagged_and_clone_requires_detach() -> None:
    sim = _sim()
    observer = ProvenanceObserver(sim)
    cells = [(1, 1), (2, 1), (3, 1)]
    sim._drop_corpse_v2(0, cells)
    assert all(observer._sources[(0, cell)] == DEATH_DROP for cell in cells)
    observer.assert_food_accounting()

    with observer.detached():
        clone_sim = copy.deepcopy(sim)
    clone = observer.clone_for(clone_sim)
    assert clone.sim is clone_sim
    assert clone._sources == observer._sources
    clone.assert_food_accounting()


def test_bulk_fifo_reinsertion_retags_old_trail_as_death_drop() -> None:
    sim = _sim()
    observer = ProvenanceObserver(sim)
    # max_food=0 still has the mechanics minimum cap of ten corpse pellets.
    # Seed ten boost-origin cells, then evict the first and reinsert it as the
    # last cell of one bulk corpse call.  A final-set-only observer gets this
    # wrong because the coordinate exists both before and after the call.
    trail = np.full((1, 2, 3), -1, dtype=np.int64)
    for index in range(10):
        trail[0, 0] = (index + 1, 1, 1)
        sim._drop_trail_pellets(trail)
    assert observer._sources[(0, (1, 1))] == BOOST_PELLET
    incoming = [(index, 2) for index in range(11, 20)] + [(1, 1)]
    sim._drop_corpse_v2(0, incoming)
    observer.assert_food_accounting()
    assert observer._sources[(0, (1, 1))] == DEATH_DROP


def test_weighted_death_delta_uses_cumulative_intake_area() -> None:
    alt = {"weighted_cumulative_intake": {DEATH_DROP: 20}}
    bare = {"weighted_cumulative_intake": {DEATH_DROP: 5}}
    assert ProvenanceObserver.death_attributed_delta(alt, bare, 5) == 3.0


def test_detach_clone_ledger_isolation_and_analyzer_accepts_zero_tick() -> None:
    """Detached cloning restores descriptors; a clone owns its ledger and tags."""
    from research.t34_o1_harvest.analyze import validate_ledger

    sim = _sim()
    original_keys = set(vars(sim))
    observer = ProvenanceObserver(sim)
    with observer.detached():
        assert set(vars(sim)) == original_keys
        for name in observer._METHODS:
            assert name not in vars(sim)
            assert getattr(sim, name).__func__ is getattr(type(sim), name)
        clone_sim = copy.deepcopy(sim)
    clone = observer.clone_for(clone_sim)
    clone.begin()
    clone_sim.step(np.array([[1, 1]], dtype=np.int64))
    clone_ledger = clone.ledger()
    assert len(clone_ledger["events"]) == 1
    assert clone_ledger["events"][0]["intake"] == {
        "ambient": 0,
        "death_drop": 0,
        "boost_pellet": 0,
    }
    validate_ledger(clone_ledger, rows=1)

    clone_sim._drop_corpse_v2(0, [(5, 5)])
    assert (0, (5, 5)) in clone._sources
    assert (0, (5, 5)) not in observer._sources
    assert observer.ledger()["events"] == []
    clone.uninstall()
    observer.uninstall()


class _Runtime:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.observed = 0

    def prepare(self, _sim: GridBatchSim) -> None:
        self.prepare_calls += 1

    def observe_step(self, _sim: GridBatchSim) -> None:
        self.observed += 1


class _PolicyCarry:
    """Copyable policy-shaped state accepted by CapturedWorld.clone."""

    def __init__(self, runtime: _Runtime, **values: object) -> None:
        self.runtime = runtime
        for name, value in values.items():
            setattr(self, name, value)


class _ReactingPolicy:
    """Synthetic policy whose second opponent action reacts to first hero action."""

    second_actions: list[tuple[int, int]] = []

    def evaluate(self, _world: object):
        from research.t34_exit_b8.policy import PolicyEvaluation

        return PolicyEvaluation(
            q=np.array([0, 2, 1, -1, -1, -1], dtype=np.float32),
            legal=np.array([False, True, True, False, False, False]),
            select=np.array([False, True, True, False, False, False]),
        )

    def action_grid(self, world: object, action: int) -> np.ndarray:
        hero = world.hero
        calls = int(getattr(hero, "calls", 0))
        if calls:
            opponent_action = int(hero.first_action)
            self.second_actions.append((int(hero.first_action), opponent_action))
        else:
            hero.first_action = int(action)
            opponent_action = 1
        hero.calls = calls + 1
        return np.array([[int(action), opponent_action]], dtype=np.int64)


def test_evaluate_root_keeps_source_digest_and_clones_reacting_carry() -> None:
    """Bounded no-model CapturedWorld fixture for O1 branch isolation."""
    from research.t3_rollout_labels_20261015.bank_engine import CapturedWorld
    from research.t34_o1_harvest.runner import evaluate_root

    sim = _sim(seed=23)
    runtime = _Runtime()
    hero = _PolicyCarry(runtime, calls=0, first_action=-1)
    opponent = _PolicyCarry(runtime, carry={"nonce": 7})
    world = CapturedWorld(
        sim, runtime, hero, [opponent], seed=23, mix="synthetic", kinds=["synthetic"]
    )
    world._o1_rng = random.Random(23).getstate()
    # evaluate_root starts from the captured selector seam; avoid an evaluator
    # or model by marking this synthetic world as already prepared.
    sim._policy_phase_prepared = True
    observer = ProvenanceObserver(sim)
    _ReactingPolicy.second_actions = []
    result = evaluate_root(world, observer, _ReactingPolicy(), horizon=2, deadline=float("inf"))
    assert result["source_preserved"] is True
    assert set(result["branches"]) == {"1", "2"}
    assert {item for item in _ReactingPolicy.second_actions} >= {(1, 1), (2, 2)}
    assert world.hero.calls == 0
    assert world.opponents[0].carry == {"nonce": 7}
    observer.uninstall()
