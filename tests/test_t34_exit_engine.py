"""Unit contracts for the deterministic ExIt-B8 teacher controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.t34_exit_b8 import (
    ExitTeacher,
    ExitTeacherConfig,
    PolicyEvaluation,
    TriggerConfig,
    action_grid,
    mix_action,
)


@dataclass
class _Sim:
    lives: dict[int, int]
    alive: np.ndarray
    length: np.ndarray
    S: int = 6
    _policy_phase_prepared: bool = True
    selected: int | None = None

    def get_legal_action_mask(self) -> np.ndarray:
        return np.ones((1, 6, 6), dtype=bool)

    def get_resolved_action_mask(self) -> np.ndarray:
        return np.ones((1, 6, 6), dtype=bool)

    def get_alive(self) -> np.ndarray:
        return self.alive

    def step(self, actions: np.ndarray) -> None:
        action = int(actions[0, 0])
        if self.selected is None:
            self.selected = action
        self.lives.setdefault(self.selected, 3)
        self.lives[self.selected] -= 1
        self.alive[0, 0] = self.lives[self.selected] > 0
        self.length[0, 0] = 10 if self.alive[0, 0] else 0
        self._policy_phase_prepared = False

    def step_with_policy(self, chooser) -> None:
        self.step(chooser(self))


class _Runtime:
    def prepare(self, sim) -> None:
        del sim

    def observe_step(self, sim) -> None:
        del sim


class _FixedPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def actions(self, masks, sim, slots) -> np.ndarray:
        del masks, sim
        self.calls += 1
        return np.ones(len(slots), dtype=np.int64)


class _World:
    def __init__(self, lives: dict[int, int]) -> None:
        self._lives = dict(lives)
        self.sim = _Sim(dict(lives), np.ones((1, 6), dtype=bool), np.full((1, 6), 10))
        self.runtime = _Runtime()
        self.hero, self.opponents = _FixedPolicy(), [_FixedPolicy() for _ in range(5)]

    def clone(self):
        return _World(self._lives)


class _Policy:
    def __init__(self, q: list[float]) -> None:
        self._q = np.asarray(q, dtype=np.float32)

    def evaluate(self, world) -> PolicyEvaluation:
        del world
        return PolicyEvaluation(self._q, np.ones(6, dtype=bool))


def test_verify_pass_never_expands_and_counts_one_active_branch() -> None:
    teacher = ExitTeacher(ExitTeacherConfig(horizon=3), policy=_Policy([0, 5, 1, 2, 3, 4]))
    action, diagnostic = teacher.choose_one(_World({1: 9}))

    assert action == diagnostic.student_action == 1
    assert diagnostic.verified_survives and not diagnostic.expanded
    assert diagnostic.forbidden == ()
    assert diagnostic.outcomes[1].alive_steps == 3
    assert diagnostic.active_row_steps == diagnostic.resident_row_steps == 3


def test_failure_expands_all_alternatives_and_prefers_highest_q_survivor() -> None:
    # Student action 1 dies immediately.  Actions 2 and 4 survive, but action
    # 4 has the larger student Q and therefore becomes the label a-star.
    teacher = ExitTeacher(ExitTeacherConfig(horizon=3), policy=_Policy([0, 9, 2, 1, 3, 0]))
    action, diagnostic = teacher.choose_one(_World({1: 0, 2: 5, 4: 5, 0: 1, 3: 2, 5: 1}))

    assert action == 4
    assert diagnostic.expanded
    assert diagnostic.forbidden == (0, 1, 3, 5)
    assert diagnostic.outcomes[2].survives
    assert diagnostic.outcomes[4].survives
    assert diagnostic.active_row_steps == diagnostic.resident_row_steps
    assert diagnostic.active_row_steps < diagnostic.nominal_row_steps
    assert teacher.profile()["expansions"] == 1


def test_no_survivor_uses_longest_lived_and_student_breaks_ties() -> None:
    teacher = ExitTeacher(ExitTeacherConfig(horizon=4), policy=_Policy([0, 7, 1, 1, 1, 1]))
    action, diagnostic = teacher.choose_one(_World({1: 2, 2: 2, 0: 1, 3: 1, 4: 1, 5: 1}))

    assert action == 1  # student action wins the longest-survival tie
    assert not diagnostic.outcomes[action].survives
    assert diagnostic.forbidden == ()


def test_trigger_and_beta_are_deterministic_and_configured() -> None:
    trigger = TriggerConfig(confirm_interval=8, confirm_min_length=600, low_length_probability=1.0)
    assert trigger.fires(decision=16, length=600, seed=2)
    assert not trigger.fires(decision=17, length=600, seed=2)
    assert trigger.fires(decision=17, length=599, seed=2)
    assert mix_action(1, 4, seed=9, decision=12, beta=1.0) == 4
    assert mix_action(1, 4, seed=9, decision=12, beta=0.0) == 1
    assert mix_action(1, 4, seed=9, decision=12, beta=0.5) == mix_action(
        1, 4, seed=9, decision=12, beta=0.5
    )


def test_forced_action_grid_never_dispatches_the_hero() -> None:
    world = _World({1: 3})
    grid = action_grid(world, 4)

    assert int(grid[0, 0]) == 4
    assert world.hero.calls == 0
    # Separate scripted-policy instances must retain their independent state.
    assert [policy.calls for policy in world.opponents] == [1] * 5


def test_forced_action_grid_batches_only_explicitly_shared_opponents() -> None:
    world = _World({1: 3})
    shared = _FixedPolicy()
    world.opponents = [shared] * 5

    action_grid(world, 2)

    assert shared.calls == 1
