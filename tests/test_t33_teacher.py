"""Focused unit coverage for T33's fixed frozen-opponent row contract."""

from __future__ import annotations

import numpy as np

from research.t33_batched_teacher.teacher import BatchedTeacher, SearchConfig


def test_frozen_opponent_forward_keeps_dead_rows_in_its_static_batch() -> None:
    """Dead rows are masked after, rather than compacted before, a frozen forward."""

    class Runtime:
        sel_state = np.array([[[0.0], [1.0]], [[0.0], [2.0]]], dtype=np.float32)
        sel_mask = np.array(
            [
                [
                    [False, False, False, False, False, False],
                    [True, False, False, False, False, False],
                ],
                [
                    [False, False, False, False, False, False],
                    [False, True, False, False, False, False],
                ],
            ]
        )

    class Sim:
        alive = np.array([[True, True], [True, False]])

        @staticmethod
        def get_resolved_action_mask() -> np.ndarray:
            return np.ones((2, 2, 6), dtype=bool)

    class Frozen:
        runtime = Runtime()

        def __init__(self) -> None:
            self.rows: list[float] = []

        def q_values(self, state: np.ndarray) -> np.ndarray:
            self.rows.extend(state[:, 0].tolist())
            return np.tile(np.arange(6, dtype=np.float32), (len(state), 1))

    policy = Frozen()
    actions = np.ones((2, 2), dtype=np.int64)
    BatchedTeacher()._dispatch_opponents(
        Sim(), Runtime(), actions, {policy: np.array([[0, 1], [1, 1]], dtype=np.int64)}
    )
    assert policy.rows == [1.0, 2.0]
    assert actions.tolist() == [[1, 0], [1, 1]]


def test_empty_many_world_call_and_profile_are_deterministic() -> None:
    """The public API has a safe fixed result for an empty scheduler chunk."""
    teacher = BatchedTeacher(SearchConfig(rollouts=1, horizon=1, length_gate=0))
    actions, diagnostics = teacher.choose([])
    assert actions.dtype == np.int64
    assert actions.size == 0
    assert diagnostics == []
    assert teacher.profile()["decisions"] == 0
