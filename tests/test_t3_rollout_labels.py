"""Focused contracts for the T3 batched-rollout public API."""

from types import SimpleNamespace

import numpy as np
import pytest

from research.t1_commitment_20261015.replay import build_replay, configure_cpu
from research.t1_commitment_20261015.study import rollout
from research.t3_rollout_labels_20261015.engine import batched_rollout, extract_prepared_env
from research.t3_rollout_labels_20261015.trigger import capped_triggers


def test_batched_rollout_rejects_duplicate_candidates_before_simulation() -> None:
    prepared = SimpleNamespace(sim=SimpleNamespace(E=1))
    with pytest.raises(ValueError, match="unique"):
        batched_rollout(prepared, [1, 1], decision=0)


def test_batched_rollout_rejects_nonpositive_depth_before_simulation() -> None:
    prepared = SimpleNamespace(sim=SimpleNamespace(E=1))
    with pytest.raises(ValueError, match="positive"):
        batched_rollout(prepared, [1], decision=0, depth=0)


def test_extract_prepared_env_checks_index_before_cloning() -> None:
    prepared = SimpleNamespace(sim=SimpleNamespace(E=2))
    with pytest.raises(IndexError, match="outside"):
        extract_prepared_env(prepared, 2)


def test_real_prepared_batch_matches_independent_continuations() -> None:
    """Exercise extraction, different first actions and clone isolation together."""
    configure_cpu()
    replay = build_replay("b8_s0", [0, 1])
    captured = []

    def observe(current):
        captured.append(extract_prepared_env(current, 1))

    replay.step(observer=observe)
    prepared = captured[0]
    original = prepared.sim.bodies.copy()
    actions = np.flatnonzero(prepared.sim.get_advisory_action_mask()[0, 0]).tolist()
    outcomes = batched_rollout(prepared, actions, decision=0)
    for action in actions:
        expected = rollout(prepared, action, decision=0)
        for key in ("survives", "frames_survived", "frames_executed", "death_decision"):
            assert outcomes[action][key] == expected[key]
    np.testing.assert_array_equal(prepared.sim.bodies, original)
    assert prepared.seeds == [replay.seeds[1]]
    assert outcomes["_batch_receipt"]["env_frames"] == sum(
        outcomes[action]["frames_executed"] for action in actions
    )


def test_trigger_cap_uses_decision_distance_and_resets_for_each_world() -> None:
    frames = np.array([0, 1, 4, 5, 6, 10, 17])
    expected = np.array([True, False, False, True, False, True, True])
    np.testing.assert_array_equal(capped_triggers(frames, np.ones(7, bool)), expected)
    np.testing.assert_array_equal(capped_triggers(frames, np.ones(7, bool)), expected)
    np.testing.assert_array_equal(
        capped_triggers(frames, np.array([False, True, True, True, True, True, True])),
        np.array([False, True, False, False, True, False, True]),
    )
