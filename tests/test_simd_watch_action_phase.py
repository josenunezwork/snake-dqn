"""Watch action selection observes the same prepared frame as GameState."""

import numpy as np
import pytest

from src.simd_env.batch_sim import BatchSim, BatchSimConfig


def _sim() -> BatchSim:
    return BatchSim(BatchSimConfig(num_envs=2, num_snakes=1), seeds=[3, 4], train_mode=False)


def test_policy_observes_prepared_frame_and_frozen_inactive_row():
    sim = _sim()
    before_frame = sim.frame.copy()
    before_body = sim.bodies[1].copy()
    seen = []

    def select(world: BatchSim) -> np.ndarray:
        seen.append((world.frame.copy(), world.get_resolved_action_mask().copy()))
        return np.ones((2, 1), dtype=np.int64)

    sim.step_with_policy(select, np.array([True, False], dtype=bool))

    assert len(seen) == 1
    assert seen[0][0].tolist() == [before_frame[0] + 1, before_frame[1]]
    assert seen[0][1].shape == (2, 1, 6)
    assert np.array_equal(sim.bodies[1], before_body)
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
