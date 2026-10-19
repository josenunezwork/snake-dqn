"""Focused contracts for the Phase-0 exact-search teacher."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.phase0_search_20261019 import teacher


def _world(length: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        sim=SimpleNamespace(frame=np.array([17]), length=np.array([[length]])),
        seed=919,
        mix="scripted",
    )


def test_bare_gate_returns_advisory_b8_action(monkeypatch) -> None:
    world = _world(100)
    monkeypatch.setattr(
        teacher,
        "_q_and_legal",
        lambda _world: (
            np.array([0.0, 1.0, 7.0, 0.0, 0.0, 0.0]),
            np.array([True, True, True, False, False, False]),
            np.array([True, True, True, False, False, False]),
        ),
    )
    action, diag = teacher.SearchTeacher(teacher.SearchConfig(length_gate=600)).choose(world)
    assert action == 2
    assert not diag["searched"]
    assert diag["scores"] == {}
    assert diag["length"] == 100


def test_searches_all_domain_legal_actions_and_breaks_score_ties(monkeypatch) -> None:
    world = _world(700)
    monkeypatch.setattr(
        teacher,
        "_q_and_legal",
        lambda _world: (
            np.arange(6, dtype=np.float32),
            np.array([True, True, False, True, False, False]),
            np.array([True, True, False, True, False, False]),
        ),
    )
    calls = []

    def scores(_world, actions, _config, decision):
        calls.extend((action, rollout, decision) for action in actions for rollout in range(2))
        return {action: 3.0 if action in (0, 1) else 2.0 for action in actions}

    monkeypatch.setattr(teacher, "_batched_scores", scores)
    action, diag = teacher.SearchTeacher(teacher.SearchConfig(rollouts=2)).choose(world)
    assert action == 0
    assert diag["searched"]
    assert set(diag["scores"]) == {0, 1, 3}
    assert calls == [(0, 0, 16), (0, 1, 16), (1, 0, 16), (1, 1, 16), (3, 0, 16), (3, 1, 16)]


def test_rng_is_repeatable_and_action_rollout_specific() -> None:
    first = teacher._rng(7, 9, 1, 2).integers(2**31, size=4)
    second = teacher._rng(7, 9, 1, 2).integers(2**31, size=4)
    changed = teacher._rng(7, 9, 1, 3).integers(2**31, size=4)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, changed)


def test_configure_cpu_overwrites_local_thread_caps(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    teacher.configure_cpu()
    assert teacher.os.environ["OMP_NUM_THREADS"] == "1"
    assert teacher.os.environ["NUMBA_NUM_THREADS"] == "1"
    assert teacher.os.environ["SNAKE_DQN_DEVICE"] == "cpu"


def test_frozen_replicas_share_one_batched_policy_and_independent_runtime_rows() -> None:
    """Frozen network dispatch groups all replica rows by the same policy object."""
    from research.phase0_search_20261019 import bank

    checkpoint = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth"
    )
    if not checkpoint.is_file():
        pytest.skip("local B8 input inventory is unavailable")
    world = teacher.build_world(bank._pool("teacher-test", {"frozen": 1})[0], checkpoint)
    replica = teacher._replicate(world, 3)
    assert replica.sim.E == 3
    assert replica.runtime.controlled.shape[0] == 3
    assert not np.shares_memory(replica.runtime.controlled[0], replica.runtime.controlled[1])
    for slot in range(len(world.opponents)):
        assert len({id(row[slot]) for row in replica.opponents}) == 1
        assert getattr(replica.opponents[0][slot], "forward", "batched") == "batched"

    from src.simd_env.vector61_policy import Vector61SimdPolicy

    calls = []
    original = Vector61SimdPolicy.actions

    def traced(self, masks, sim, slots):
        calls.append(len(slots))
        return original(self, masks, sim, slots)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Vector61SimdPolicy, "actions", traced)
    source_frame = int(world.sim.frame[0])
    try:
        teacher.step_world(
            world, teacher.SearchTeacher(teacher.SearchConfig(rollouts=1, horizon=2, length_gate=0))
        )
    finally:
        monkeypatch.undo()
    assert 3 in calls
    # Search replicas consume their prepared selector directly; only the outer
    # live step advances the source world once.
    assert int(world.sim.frame[0]) == source_frame + 1


def test_batched_frozen_replica_matches_two_scalar_action_tapes() -> None:
    """Replica rows retain exact frozen carry state under a fixed 30-frame tape."""
    from research.phase0_search_20261019 import bank
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    checkpoint = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth"
    )
    if not checkpoint.is_file():
        pytest.skip("local B8 input inventory is unavailable")
    world = teacher.build_world(bank._pool("teacher-oracle", {"frozen": 1})[0], checkpoint)
    observed = {}

    def selector(prepared):
        world.runtime.prepare(prepared)
        source_digest = gid.env_digest(prepared, 0)
        replica = teacher._replicate(world, 2)
        scalar = [world.clone(), world.clone()]
        first = np.ones((2, prepared.S), dtype=np.int64)
        replica.sim.step(first)
        replica.runtime.observe_step(replica.sim)
        for branch in scalar:
            branch.sim.step(first[:1])
            branch.runtime.observe_step(branch.sim)
        for _ in range(29):
            tape = np.ones((2, prepared.S), dtype=np.int64)
            replica.sim.step_with_policy(
                lambda current: (replica.runtime.prepare(current), tape)[1]
            )
            replica.runtime.observe_step(replica.sim)
            for branch in scalar:
                branch.sim.step_with_policy(
                    lambda current, branch=branch: (branch.runtime.prepare(current), tape[:1])[1]
                )
                branch.runtime.observe_step(branch.sim)
        for env, branch in enumerate(scalar):
            assert gid.env_digest(replica.sim, env) == gid.env_digest(branch.sim, 0)
            for name, value in vars(replica.runtime).items():
                if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == 2:
                    np.testing.assert_array_equal(value[env], getattr(branch.runtime, name)[0])
            for name, value in vars(replica.runtime.memory).items():
                if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == 2:
                    np.testing.assert_array_equal(
                        value[env], getattr(branch.runtime.memory, name)[0]
                    )
            assert (
                int(replica.sim.frame[env])
                == int(branch.sim.frame[0])
                # The first forced direct step consumes the already-prepared
                # decision; only the remaining 29 selections increment frame.
                == int(prepared.frame[0]) + 29
            )
        assert gid.env_digest(prepared, 0) == source_digest
        observed["done"] = True
        return first[:1]

    world.sim.step_with_policy(selector)
    world.runtime.observe_step(world.sim)
    assert observed == {"done": True}


def test_actual_frozen_search_is_repeatable() -> None:
    """The same strict world, branch RNG keys, and search config choose identically."""
    from research.phase0_search_20261019 import bank

    checkpoint = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth"
    )
    if not checkpoint.is_file():
        pytest.skip("local B8 input inventory is unavailable")
    row = bank._pool("teacher-repeat", {"frozen": 1})[0]
    config = teacher.SearchConfig(rollouts=1, horizon=4, length_gate=0)
    first = teacher.step_world(teacher.build_world(row, checkpoint), teacher.SearchTeacher(config))
    second = teacher.step_world(teacher.build_world(row, checkpoint), teacher.SearchTeacher(config))
    assert first["action"] == second["action"]
    np.testing.assert_array_equal(first["q"], second["q"])
    assert first["scores"] == second["scores"]
