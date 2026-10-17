"""Focused invariants for the opt-in T19 mixed-seat collector."""

import numpy as np

from research.redesign_m3c_20261008.train_m3c import (
    NStepBuilder,
    NStepBuilderRef,
    learner_seat_mask,
    mixed_stream_ids,
)


def test_legacy_seat_counts_do_not_consume_permutation_rng() -> None:
    assert np.array_equal(learner_seat_mask(7, 9, 4, 1), np.array([[1, 0, 0, 0, 0, 0]] * 4, bool))
    assert learner_seat_mask(7, 9, 4, 6).all()


def test_mixed_seat_permutation_is_seeded_segment_pinned_and_balanced() -> None:
    first = learner_seat_mask(12, 3, 32, 3)
    assert np.array_equal(first, learner_seat_mask(12, 3, 32, 3))
    assert not np.array_equal(first, learner_seat_mask(12, 4, 32, 3))
    assert np.array_equal(first.sum(axis=1), np.full(32, 3))


def test_mixed_stream_ids_are_dense_per_world_despite_shuffled_physical_seats() -> None:
    seats = np.array([[False, True, False, True, True, False], [True, False, True, False, False, True]])
    rows = np.argwhere(seats)
    assert np.array_equal(mixed_stream_ids(rows, seats), np.arange(6))


def test_mixed_life_streams_match_scalar_oracle_with_death_and_segment_tail() -> None:
    # Dense mixed streams are keyed by world*N + live-seat rank.  Death flushes only
    # the affected life; the final tail preserves the current life bootstrap.
    fast, ref = NStepBuilder(6, n_step=3), NStepBuilderRef(6, n_step=3)
    worlds = np.array([0, 1, 3, 4], dtype=np.int64)
    ids = np.array([10, 11, 12, 13], dtype=np.int64)
    acts = np.array([0, 1, 2, 3], dtype=np.int64)
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 99.0, 98.0], dtype=np.float32)
    done = np.array([False, False, False, False, False, False])
    fast.step(worlds, ids, acts, rewards, done, 0.9)
    ref.step(worlds, ids, acts, rewards, done, 0.9)
    done[3] = True  # one seat dies; the other mixed seats keep their own windows.
    ids += 20
    fast.step(worlds, ids, acts, rewards, done, 0.9)
    ref.step(worlds, ids, acts, rewards, done, 0.9)
    fast.end_segment(0.9)
    ref.end_segment(0.9)
    got = fast.take()
    want = ref.take()
    for a, b in zip(got, want):
        assert np.array_equal(a, b)
