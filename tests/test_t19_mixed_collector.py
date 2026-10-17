"""Focused invariants for the opt-in T19 mixed-seat collector."""

import numpy as np
import pytest
from pathlib import Path
import importlib.util
import subprocess
import sys

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


def test_real_short_n3_actor_dispatches_only_live_rows(tmp_path, monkeypatch) -> None:
    """Bounded CPU collector exercise: real pinned pool, no learner/update loop."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_m6_reward import _run_actor

    ckpt_dir = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
    init = ckpt_dir / "b8_s3_t7_20261016.pth"
    monkeypatch.setenv("SNAKE_CKPT_DIR", str(ckpt_dir))
    messages, _ = _run_actor(
        __import__("research.redesign_m3c_20261008.train_m3c", fromlist=["actor_main"]),
        tmp_path,
        init,
        8,
        (
            "--learner-seats", "3", "--obs-version", "bv8", "--onpolicy-labels",
            "--label-frac", "1.0", "--anchor-mode", "v8margin",
            "--league-b8-checkpoint", str(init),
        ),
    )
    assert len(messages) == 8
    assert all(sum(m["live_rows_by_policy"].values()) <= 36 for m in messages)
    assert all(m["live_rows_by_policy"]["learner"] == m["decisions"] for m in messages)
    assert all(len(m["lab_ids"]) == m["decisions"] for m in messages)
    assert any(any(key != "learner" for key in m["live_rows_by_policy"]) for m in messages)


@pytest.mark.parametrize("extra", [(), ("--all-snakes-learner",)], ids=["n1", "n6"])
def test_real_legacy_actor_matches_pre_league_collector(tmp_path, extra) -> None:
    """N=1 and N=6 preserve pre-league hero/all-slot frames under fixed RNG."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_m6_reward import _init, _run_actor
    from research.redesign_m3c_20261008 import train_m3c as current

    source = subprocess.run(
        ["git", "show", "1331b346:research/redesign_m3c_20261008/train_m3c.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    path = tmp_path / "baseline_train_m3c.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("t19_baseline_train_m3c", path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    assert spec.loader is not None
    spec.loader.exec_module(baseline)
    init = _init(tmp_path)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old, _ = _run_actor(baseline, old_dir, init, 8, extra)
    new, _ = _run_actor(current, new_dir, init, 8, extra)
    for old_msg, new_msg in zip(old, new):
        for key, value in old_msg.items():
            if key in {"env_s", "timing", "j"}:
                continue
            other = new_msg[key]
            if isinstance(value, np.ndarray):
                assert np.array_equal(value, other), key
            else:
                assert value == other, key
