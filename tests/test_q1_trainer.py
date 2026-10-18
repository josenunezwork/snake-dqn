"""Q1 schedule and task-collector registration contracts."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pytest

from research.redesign_m3c_20261008 import task_collection as C
from research.redesign_m3c_20261008 import train_m3c as T


def _args(*extra: str):
    return T.build_parser().parse_args(
        ["--seed", "7", "--init", "x", "--demo-data", ".", "--out", "out", *extra]
    )


def test_mature_schedule_is_constant_at_registered_q1_checkpoints():
    args = _args("--schedule-package", "mature")
    T.apply_schedule_package(args)
    schedule = T.schedules_from_args(args)
    for t in (0, 2_000_000, 8_000_000):
        assert schedule.gamma(t) == pytest.approx(0.998)
        assert schedule.eps(t) == pytest.approx(0.01)
        assert schedule.anchor(t) == pytest.approx(0.1)
        # Reward shaping and the five-step target must use the same registered gamma.
        assert T.mass_term(np.array([100]), np.array([True]), schedule.gamma(t), 1.0)[
            0
        ] == pytest.approx(0.02)
        assert schedule.gamma(t) ** T.N_STEP == pytest.approx(0.990039920079968)


def test_restart_schedule_defaults_are_unchanged_and_explicit_starts_work():
    schedule = T.schedules_from_args(_args())
    assert (schedule.gamma(0), schedule.eps(0), schedule.anchor(0)) == (0.99, 0.05, 1.0)
    explicit = T.schedules_from_args(_args("--eps-start", ".02", "--anchor-start", ".4"))
    assert explicit.eps(0) == pytest.approx(0.02)
    assert explicit.anchor(0) == pytest.approx(0.4)


def test_task_collection_mix_draws_are_stable_isolated_and_uniform_by_probability():
    indexes = np.arange(12_000)
    first = C.equal_mix_assignments(23, indexes)
    assert np.array_equal(first, C.equal_mix_assignments(23, indexes))
    assert set(first) == set(C.MIXES)
    counts = {mix: int((first == mix).sum()) for mix in C.MIXES}
    for count in counts.values():
        assert abs(count / len(first) - 1 / 3) < 0.03
    assert C.fresh_namespace("C", 23).startswith(C.NAMESPACE_PREFIX)
    assert C.mix_for_reset(23, 1, 2, 3) == C.mix_for_reset(23, 1, 2, 3)
    assert C.reset_seed("q1/x", 23, 1, 2, 3) != C.reset_seed("q1/x", 23, 1, 2, 4)


def test_task_collection_terminal_and_horizon_boundaries_are_distinct():
    reset = C.reset_mask_after_step(np.array([True, False, False]), np.array([8, 5_000, 4_999]))
    assert reset.tolist() == [True, True, False]


def test_terminal_nstep_flush_prevents_death_respawn_credit_leakage():
    builder = T.NStepBuilder(1, n_step=5)
    builder.step(
        np.array([0]), np.array([10]), np.array([1]), np.array([2.0]), np.array([True]), 0.9
    )
    first = builder.take()
    assert first[0].tolist() == [10] and first[2].tolist() == pytest.approx([2.0])
    assert first[3].tolist() == [0.0] and first[4].tolist() == [-1]
    builder.step(
        np.array([0]), np.array([20]), np.array([2]), np.array([7.0]), np.array([True]), 0.9
    )
    second = builder.take()
    assert second[0].tolist() == [20] and second[2].tolist() == pytest.approx([7.0])


def test_horizon_truncation_keeps_the_registered_bootstrap_not_a_terminal_discount():
    builder = T.NStepBuilder(1, n_step=5)
    builder.step(
        np.array([0]), np.array([30]), np.array([0]), np.array([3.0]), np.array([False]), 0.998
    )
    builder.step(
        np.array([0]), np.array([31]), np.array([1]), np.array([4.0]), np.array([False]), 0.998
    )
    builder.end_worlds(np.array([0]), 0.998)
    row = builder.take()
    assert row[0].tolist() == [30]
    assert row[2].tolist() == pytest.approx([3.0])
    assert row[3].tolist() == pytest.approx([0.998]) and row[4].tolist() == [31]


@dataclass(frozen=True)
class _Cfg:
    gamma: float


def test_task_reward_gamma_changes_the_live_simulator_contract_at_schedule_boundaries():
    class Sim:
        cfg = _Cfg(0.99)

    sim = Sim()
    for progress in (0, 2_000_000, 8_000_000):
        schedule = T.schedules_from_args(_args())
        T.apply_task_reward_gamma(sim, schedule.gamma(progress))
        assert sim.cfg.gamma == pytest.approx(schedule.gamma(progress))


def test_task_reset_replays_constructor_food_draw_path_for_fresh_gate_seed():
    class Sim:
        E = 2
        _seeds = [1, 2]
        _rngs = [None, None]

        def __init__(self):
            self.calls = []

        def _normalize_reset_env_mask(self, mask):
            return np.asarray(mask, dtype=bool)

        def _validated_reset_rngs(self, selected, seeds):
            return [(seed, f"rng-{seed}") for seed in seeds]

        def _reset_env(self, env, *, constructor_food_draws):
            self.calls.append(("reset", env, constructor_food_draws))

        def _rebuild_traversed_from_heads(self, mask):
            self.calls.append(("traversed", mask.tolist()))

        def _refresh_action_masks(self, mask):
            self.calls.append(("masks", mask.tolist()))

    sim = Sim()
    T.reset_task_worlds(sim, np.array([False, True]), [99])
    assert sim._seeds == [1, 99] and sim._rngs == [None, "rng-99"]
    assert sim.calls == [("reset", 1, True), ("traversed", [False, True]), ("masks", [False, True])]


def test_task_aligned_profile_requires_explicit_n1_and_fresh_namespace():
    with pytest.raises(SystemExit, match="--actors and --learner-seats 1"):
        T.main(
            [
                "--seed",
                "0",
                "--init",
                "x",
                "--demo-data",
                ".",
                "--out",
                "x",
                "--collection-profile",
                C.PROFILE,
            ]
        )
    with pytest.raises(SystemExit, match="fresh q1-task-aligned"):
        T.main(
            [
                "--seed",
                "0",
                "--init",
                "x",
                "--demo-data",
                ".",
                "--out",
                "x",
                "--actors",
                "1",
                "--learner-seats",
                "1",
                "--collection-profile",
                C.PROFILE,
            ]
        )
