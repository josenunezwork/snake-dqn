"""T17 learner bookkeeping and bounded actor-shutdown contracts."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from collections import deque

import numpy as np
import pytest

from research.redesign_m3c_20261008 import train_m3c as T


def _queue_producer(tx, stop_evt, ready_evt) -> None:
    """Fill a tiny queue repeatedly; exit normally once the learner stops."""
    item = b"x" * 131_072
    assert T._put_until_stopped(tx, item, stop_evt, timeout=0.01)
    ready_evt.set()
    while T._put_until_stopped(tx, item, stop_evt, timeout=0.01):
        pass


def test_action_window_is_identical_to_deque_across_wrapped_messages() -> None:
    rng = np.random.default_rng(17)
    actual = T.ActionWindow(capacity=31)
    expected = deque(maxlen=31)
    for rows in (1, 7, 29, 3, 65, 2, 31, 4):
        actions = rng.integers(0, 6, size=rows, dtype=np.int8)
        actual.extend(actions)
        expected.extend(actions.tolist())
        assert actual.size == len(expected)
        np.testing.assert_array_equal(
            actual.counts, np.bincount(np.asarray(expected), minlength=6), strict=True
        )
    assert actual.full


def test_action_window_rejects_no_actions_without_changing_counts() -> None:
    window = T.ActionWindow(capacity=4)
    window.extend(np.array([0, 1], dtype=np.int8))
    before = window.counts.copy()
    window.extend(np.array([], dtype=np.int8))
    np.testing.assert_array_equal(window.counts, before)
    assert window.size == 2


def test_action_window_preserves_collapse_decision_at_every_message_boundary() -> None:
    rng = np.random.default_rng(8)
    capacity = 101
    expected = deque(maxlen=capacity)
    actual = T.ActionWindow(capacity=capacity)
    # Variable message lengths force partial evictions and circular writes before a
    # deliberately collapsed tail.  The old loop evaluated precisely at these boundaries.
    for rows, action in [(7, None), (93, None), (11, None), (40, 4), (1, 4), (99, 4)]:
        actions = (
            rng.integers(0, 6, size=rows, dtype=np.int8)
            if action is None
            else np.full(rows, action, dtype=np.int8)
        )
        expected.extend(actions.tolist())
        actual.extend(actions)
        old_halts = len(expected) == capacity and max(
            np.bincount(np.asarray(expected), minlength=6)
        ) > 0.9 * len(expected)
        new_halts = actual.full and actual.counts.max() > 0.9 * actual.counts.sum()
        assert new_halts == old_halts


def test_stop_aware_put_returns_when_full_queue_is_stopped() -> None:
    tx: queue.Queue[int] = queue.Queue(maxsize=1)
    tx.put(1)
    stop = threading.Event()
    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(T._put_until_stopped(tx, 2, stop, timeout=0.01))
    )
    worker.start()
    time.sleep(0.03)
    stop.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert result == [False]


def test_tiny_actor_queue_drains_to_normal_exitcode() -> None:
    """A maxsize-one queue cannot strand an actor after a normal wall-stop."""
    ctx = mp.get_context("spawn")
    tx = ctx.Queue(maxsize=1)
    stop_evt = ctx.Event()
    ready_evt = ctx.Event()
    proc = ctx.Process(target=_queue_producer, args=(tx, stop_evt, ready_evt))
    proc.start()
    try:
        assert ready_evt.wait(timeout=5), "producer never filled the one-frame queue"
        stop_evt.set()
        T._drain_queues_while_joining([proc], [tx], deadline_seconds=2.0)
        proc.join(timeout=1)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1)
        tx.close()


def test_drain_queues_while_joining_consumes_messages_until_worker_exits() -> None:
    class Producer:
        def __init__(self) -> None:
            self.messages = 4

        def is_alive(self) -> bool:
            return self.messages > 0

        def join(self, timeout: float) -> None:
            del timeout
            self.messages -= 1

    class Feed:
        def __init__(self) -> None:
            self.items = [1, 2, 3]

        def get_nowait(self) -> int:
            if not self.items:
                raise queue.Empty
            return self.items.pop(0)

    proc, tx = Producer(), Feed()
    T._drain_queues_while_joining([proc], [tx], deadline_seconds=1.0)
    assert proc.messages == 0 and tx.items == []


@pytest.mark.parametrize("value", ["-1", "-2"])
def test_gather_threads_rejects_negative_values(value: str) -> None:
    with pytest.raises(SystemExit, match="gather-threads"):
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
                "--gather-threads",
                value,
            ]
        )


def test_gather_threads_requires_the_fast_multi_actor_path() -> None:
    base = ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "x"]
    with pytest.raises(SystemExit, match="multi-actor"):
        T.main([*base, "--gather-threads", "2"])
    with pytest.raises(SystemExit, match="needs --learner-fast"):
        T.main([*base, "--actors", "2", "--gather-threads", "2"])


def test_gather_threads_is_recorded_only_when_enabled() -> None:
    parser = T.build_parser()
    base = parser.parse_args(["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "x"])
    enabled = parser.parse_args(
        [
            "--seed",
            "0",
            "--init",
            "x",
            "--demo-data",
            ".",
            "--out",
            "x",
            "--gather-threads",
            "3",
        ]
    )
    assert "gather_threads" not in T.m3c_flags(base)
    assert T.m3c_flags(enabled)["gather_threads"] == 3
