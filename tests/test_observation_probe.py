"""Focused contracts for the first-stage raster observation probe."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

import src.evaluation.observation_probe as observation_probe
from src.evaluation.observation_probe import (
    ProbeConfig,
    clone_sim,
    finite_action_returns,
    full_state_digest,
    heading_twin,
    observe_hero,
)
from src.simd_env.batch_sim import BatchSim, BatchSimConfig


def _sim(*, snakes: int = 3, seed: int = 17) -> BatchSim:
    return BatchSim(
        BatchSimConfig(num_envs=1, num_snakes=snakes, initial_food=6, max_food=6),
        seeds=[seed],
        train_mode=True,
        allow_respawn=False,
    )


def _tape(sim: BatchSim, count: int = 16) -> np.ndarray:
    return np.tile(np.array([[[1] * sim.S]], dtype=np.int64), (count, 1, 1))


def test_clone_digest_rng_and_mutable_state_are_independent() -> None:
    sim = _sim()
    before = full_state_digest(sim)
    clone = clone_sim(sim)
    assert full_state_digest(clone) == before
    assert clone.bodies is not sim.bodies
    assert clone.food_cells[0] is not sim.food_cells[0]
    assert clone._rngs[0] is not sim._rngs[0]
    clone.step(np.ones((1, sim.S), dtype=np.int64))
    assert full_state_digest(sim) == before
    assert full_state_digest(clone) != before


def test_digest_includes_random_generator_internal_state_and_next_draw() -> None:
    sim = _sim()
    control = clone_sim(sim)
    before = full_state_digest(sim)
    initial_next_draw = control._rngs[0]._rng.random()
    shifted_rng = random.Random()
    shifted_rng.setstate(sim._rngs[0]._rng.getstate())
    shifted_rng.random()
    sim._rngs[0]._rng.setstate(shifted_rng.getstate())
    assert full_state_digest(sim) != before
    assert sim._rngs[0]._rng.random() != initial_next_draw


def test_digest_rejects_unknown_objects_cycles_and_ambiguous_sequence_shapes() -> None:
    nested = hashlib.sha256()
    flat = hashlib.sha256()
    observation_probe._canonical_hash([[1], 2], nested, set())
    observation_probe._canonical_hash([[1, 2]], flat, set())
    assert nested.hexdigest() != flat.hexdigest()

    sim = _sim()
    sim.unsupported_probe_state = object()
    with pytest.raises(ValueError, match="unsupported"):
        full_state_digest(sim)

    cyclic = _sim()
    objects = np.empty((1,), dtype=object)
    objects[0] = objects
    cyclic.probe_object_array = objects
    with pytest.raises(ValueError, match="cyclic"):
        full_state_digest(cyclic)


def test_natural_step_from_equal_clones_has_equal_state() -> None:
    sim = _sim()
    left, right = clone_sim(sim), clone_sim(sim)
    actions = np.array([[0, 1, 2]], dtype=np.int64)
    left.step(actions)
    right.step(actions)
    assert full_state_digest(left) == full_state_digest(right)


def test_observe_hero_is_v3_copies_and_resolved_mask() -> None:
    sim = _sim()
    observation = observe_hero(sim, 0, ProbeConfig())
    assert observation["tactical_uint8"].shape == (2, 31, 31)
    assert observation["strategic_uint8"].shape == (3, 25, 25)
    assert observation["scalars"].shape == (26,)
    assert np.array_equal(observation["mask"], sim.get_resolved_action_mask()[0, 0])
    observation["mask"][:] = False
    assert sim.get_resolved_action_mask()[0, 0].any()


def test_heading_twin_rejects_enemy_with_neck_and_preserves_source() -> None:
    sim = _sim()
    source = full_state_digest(sim)
    sim.seg_count[0, 1] = 2
    assert heading_twin(sim, 0, 1, 2, ProbeConfig()) is None
    assert full_state_digest(sim) != source  # Test setup, then no further mutation.


def test_heading_twin_rejects_same_heading_and_dead_hero() -> None:
    sim = _sim()
    current = int(sim.direction[0, 1])
    assert heading_twin(sim, 0, 1, current, ProbeConfig()) is None
    sim.alive[0, 0] = False
    assert heading_twin(sim, 0, 1, (current + 1) % 4, ProbeConfig()) is None


def test_heading_twin_rejects_visible_heading_change() -> None:
    sim = _sim()
    # Place a single-segment enemy immediately ahead, inside the tactical view.
    hero_head = sim.heads()[0, 0]
    sim.bodies[0, 1, sim.head_ptr[0, 1]] = hero_head + np.array([0, -2])
    sim.seg_count[0, 1] = 1
    sim.direction[0, 1] = 1
    sim._rebuild_traversed_from_heads(np.ones(1, dtype=bool))
    sim._refresh_action_masks(np.ones(1, dtype=bool))
    assert heading_twin(sim, 0, 1, 3, ProbeConfig()) is None


def test_finite_returns_uses_one_total_step_for_horizon_one_and_does_not_mutate() -> None:
    sim = _sim()
    before = full_state_digest(sim)
    out = finite_action_returns(sim, 0, _tape(sim), (1, 8, 16), ProbeConfig())
    assert np.array_equal(out["mask"], sim.get_resolved_action_mask()[0, 0])
    assert out["source_fingerprint"] == before == full_state_digest(sim)
    for action, record in out["actions"].items():
        if out["mask"][action]:
            assert set(record["discounted_return_by_horizon"]) == {1, 8, 16}
            assert record["discounted_return_by_horizon"][1] == record["first_step"]["reward"]
        else:
            assert record is None


def test_finite_returns_is_action_order_invariant() -> None:
    sim = _sim()
    normal = finite_action_returns(sim, 0, _tape(sim), (1, 4), ProbeConfig())
    reversed_order = finite_action_returns(
        sim, 0, _tape(sim), (1, 4), ProbeConfig(), action_order=(5, 4, 3, 2, 1, 0)
    )
    assert normal["actions"] == reversed_order["actions"]


@pytest.mark.parametrize("hero", (-1, True, 3))
def test_finite_returns_rejects_invalid_hero_without_mutation(hero) -> None:
    sim = _sim()
    before = full_state_digest(sim)
    with pytest.raises(ValueError):
        finite_action_returns(sim, hero, _tape(sim), (1,), ProbeConfig())
    assert full_state_digest(sim) == before


def test_finite_returns_rejects_multi_env_and_invalid_config_before_sim_access() -> None:
    sim = BatchSim(BatchSimConfig(num_envs=2, num_snakes=2), seeds=[1, 2], train_mode=True)
    before = full_state_digest(sim)
    with pytest.raises(ValueError, match="num_envs"):
        finite_action_returns(sim, 0, _tape(sim), (1,), ProbeConfig())
    assert full_state_digest(sim) == before
    with pytest.raises(ValueError, match="gamma"):
        finite_action_returns(_sim(), 0, _tape(_sim()), (1,), ProbeConfig(gamma=0.0))


@pytest.mark.parametrize(
    "action_order",
    [(0, True, 2, 3, 4, 5), (0, 1.0, 2, 3, 4, 5)],
)
def test_finite_returns_rejects_non_integer_action_order_without_mutation(action_order) -> None:
    sim = _sim()
    before = full_state_digest(sim)
    with pytest.raises(ValueError, match="built-in integer"):
        finite_action_returns(sim, 0, _tape(sim), (1,), ProbeConfig(), action_order=action_order)
    assert full_state_digest(sim) == before


def test_finite_returns_rejects_nonfinite_branch_rewards_without_mutation(monkeypatch) -> None:
    sim = _sim()
    before = full_state_digest(sim)
    original = BatchSim.get_reward
    monkeypatch.setattr(
        BatchSim,
        "get_reward",
        lambda branch: np.full_like(original(branch), np.inf, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="non-finite branch rewards"):
        finite_action_returns(sim, 0, _tape(sim), (1,), ProbeConfig())
    assert full_state_digest(sim) == before


def test_finite_returns_rejects_dead_floor_capped_and_gamma_mismatch_without_mutation() -> None:
    sim = _sim(snakes=6)
    before = full_state_digest(sim)
    sim.alive[0, 0] = False
    with pytest.raises(ValueError, match="living hero"):
        finite_action_returns(sim, 0, _tape(sim), (1,), ProbeConfig())
    assert full_state_digest(sim) != before  # Test setup is the only mutation.

    floor = _sim(snakes=6)
    floor.alive[0, 1:] = False
    floor_before = full_state_digest(floor)
    with pytest.raises(ValueError, match="population floor"):
        finite_action_returns(floor, 0, _tape(floor), (1,), ProbeConfig())
    assert full_state_digest(floor) == floor_before

    capped = _sim()
    capped.frame[0] = 5
    capped_before = full_state_digest(capped)
    with pytest.raises(ValueError, match="frame cap"):
        finite_action_returns(capped, 0, _tape(capped), (1,), ProbeConfig(max_frames=5))
    assert full_state_digest(capped) == capped_before

    mismatch = _sim()
    mismatch_before = full_state_digest(mismatch)
    with pytest.raises(ValueError, match="gamma"):
        finite_action_returns(mismatch, 0, _tape(mismatch), (1,), ProbeConfig(gamma=0.9))
    assert full_state_digest(mismatch) == mismatch_before


@pytest.mark.parametrize(
    "tape,horizons",
    [
        (np.zeros((1, 1, 3), dtype=np.float64), (1,)),
        (np.zeros((1, 1, 3), dtype=np.int64), (2,)),
        (np.full((1, 1, 3), 3, dtype=np.int64), (1,)),
        (np.zeros((1, 1, 3), dtype=np.int64), (1, 1)),
    ],
)
def test_finite_returns_rejects_malformed_inputs_without_mutation(tape, horizons) -> None:
    sim = _sim()
    before = full_state_digest(sim)
    with pytest.raises(ValueError):
        finite_action_returns(sim, 0, tape, horizons, ProbeConfig())
    assert full_state_digest(sim) == before


def test_probe_rejects_multi_env_and_invalid_config() -> None:
    sim = BatchSim(BatchSimConfig(num_envs=2, num_snakes=2), seeds=[1, 2], train_mode=True)
    with pytest.raises(ValueError):
        observe_hero(sim, 0, ProbeConfig())
    with pytest.raises(ValueError):
        observe_hero(_sim(), 0, ProbeConfig(gamma=0.0))
