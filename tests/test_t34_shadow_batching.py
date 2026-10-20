"""Focused unit contracts for T34 lockstep batching boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from research.t34_exit_b8.policy import PolicyEvaluation
from research.t34_exit_b8.teacher_v2 import _shadow_evaluations


class _TiePolicy:
    def __init__(self) -> None:
        self.batched_calls = 0
        self.serial_calls = 0

    def evaluate_worlds(self, worlds, *, bucket_size):
        self.batched_calls += 1
        assert bucket_size == 64
        return [
            PolicyEvaluation(
                q=np.asarray([100.0, 100.0 + 4.0e-4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                legal=np.asarray([True, True, False, False, False, False]),
            )
            for _ in worlds
        ]

    def evaluate(self, _world):
        self.serial_calls += 1
        return PolicyEvaluation(
            q=np.asarray([101.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            legal=np.asarray([True, True, False, False, False, False]),
        )


def test_shadow_near_ties_restore_the_exact_single_world_action():
    """A scale-aware fp32 margin protects the historical batch-one shadow tie rule."""
    policy = _TiePolicy()
    teacher = SimpleNamespace(device="cpu", policy=policy, evaluate=policy.evaluate)
    evaluations = _shadow_evaluations([object(), object()], teacher)
    assert policy.batched_calls == 1
    assert policy.serial_calls == 2
    assert [evaluation.greedy_action() for evaluation in evaluations] == [0, 0]
