"""Watch action selection observes the same prepared frame as GameState."""

import pickle

import numpy as np
import pytest

from src.simd_env.batch_sim import BatchSim, BatchSimConfig


def _sim() -> BatchSim:
    return BatchSim(BatchSimConfig(num_envs=2, num_snakes=1), seeds=[3, 4], train_mode=False)


def _inactive_snapshot(sim: BatchSim, env: int):
    return {
        "body": sim.bodies[env].copy(),
        "head": sim.head_ptr[env].copy(),
        "length": sim.length[env].copy(),
        "alive": sim.alive[env].copy(),
        "food": list(sim.food_cells[env]),
        "frame": int(sim.frame[env]),
        "rng": sim._rngs[env]._rng.getstate(),
        "reward": sim.get_reward()[env].copy(),
        "done": sim.get_done()[env].copy(),
        "valid": sim.get_transition_valid()[env].copy(),
    }


def test_policy_observes_prepared_frame_and_frozen_inactive_row():
    sim = _sim()
    before_frame = sim.frame.copy()
    before_inactive = _inactive_snapshot(sim, 1)
    seen = []

    def select(world: BatchSim) -> np.ndarray:
        seen.append((world.frame.copy(), world.get_resolved_action_mask().copy()))
        return np.ones((2, 1), dtype=np.int64)

    sim.step_with_policy(select, np.array([True, False], dtype=bool))

    assert len(seen) == 1
    assert seen[0][0].tolist() == [before_frame[0] + 1, before_frame[1]]
    assert seen[0][1].shape == (2, 1, 6)
    after_inactive = _inactive_snapshot(sim, 1)
    assert pickle.dumps(before_inactive) == pickle.dumps(after_inactive)
    assert sim.frame.tolist() == [before_frame[0] + 1, before_frame[1]]


@pytest.mark.parametrize(
    "actions", [np.ones((1, 1), dtype=np.int64), np.ones((2, 1)), np.full((2, 1), 6)]
)
def test_policy_actions_are_strictly_validated(actions):
    sim = _sim()
    with pytest.raises(ValueError):
        sim.step_with_policy(lambda _: actions)


def test_policy_cannot_recursively_step():
    sim = _sim()
    with pytest.raises(RuntimeError, match="recursively"):
        sim.step_with_policy(
            lambda world: (
                world.step(np.zeros((2, 1), dtype=np.int64)),
                np.zeros((2, 1), dtype=np.int64),
            )[1]
        )


def test_policy_and_legacy_array_step_are_equivalent_for_same_actions():
    policy_sim, array_sim = _sim(), _sim()
    actions = np.ones((2, 1), dtype=np.int64)
    policy_sim.step_with_policy(lambda _: actions)
    array_sim.step(actions)
    for name in ("bodies", "head_ptr", "length", "alive", "frame", "direction"):
        assert np.array_equal(getattr(policy_sim, name), getattr(array_sim, name)), name
    assert [rng._rng.getstate() for rng in policy_sim._rngs] == [
        rng._rng.getstate() for rng in array_sim._rngs
    ]


def test_policy_sees_respawned_row_and_can_choose_nonstraight_action():
    sim = _sim()
    sim.alive[0, 0] = False
    sim.respawn_timer[0, 0] = 1
    observed = []

    def select(world: BatchSim) -> np.ndarray:
        observed.append((bool(world.alive[0, 0]), world.get_resolved_action_mask()[0, 0].copy()))
        return np.array([[0], [1]], dtype=np.int64)

    sim.step_with_policy(select)

    assert observed[0][0]
    assert observed[0][1][:3].any()
    assert sim.direction[0, 0] == 0  # right (1), then relative left (0)


def test_callback_failure_cleans_internal_guard_and_regular_step_remains_usable():
    sim = _sim()
    with pytest.raises(RuntimeError, match="boom"):
        sim.step_with_policy(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    assert not hasattr(sim, "_active_env_mask")
    assert not hasattr(sim, "_policy_phase_prepared")
    assert not getattr(sim, "_policy_step_in_progress", False)
    # Lifecycle prep already happened before the callback failure; callers abort
    # that frame, but the simulator is left usable for the next valid step.
    sim.step(np.ones((2, 1), dtype=np.int64))


def test_callback_cannot_recursively_invoke_policy_api():
    sim = _sim()
    with pytest.raises(RuntimeError, match="recursively"):
        sim.step_with_policy(
            lambda world: (
                world.step_with_policy(lambda _: np.ones((2, 1), dtype=np.int64)),
                np.ones((2, 1), dtype=np.int64),
            )[1]
        )
