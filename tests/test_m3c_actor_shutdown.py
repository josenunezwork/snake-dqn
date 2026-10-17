"""The multi-actor trainer's shutdown drains the actor queues until the actors have exited."""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from research.redesign_m3c_20261008 import train_m3c as T


def _actor(tx, stop_evt, full_evt) -> None:
    """A back-pressured actor: it keeps offering one large message until told to stop."""
    import queue

    msg = {"s": np.zeros(768, np.int64), "acts": np.zeros(768, np.int8), "R": np.zeros(768)}
    while not stop_evt.is_set():
        try:
            tx.put(msg, timeout=1.0)
        except queue.Full:
            full_evt.set()
            continue


def _spawn(n_actors: int, maxsize: int = 8):
    ctx = mp.get_context("spawn")
    stop_evt = ctx.Event()
    txs = [ctx.Queue(maxsize=maxsize) for _ in range(n_actors)]
    full_events = [ctx.Event() for _ in range(n_actors)]
    procs = [
        ctx.Process(target=_actor, args=(txs[j], stop_evt, full_events[j]), daemon=True)
        for j in range(n_actors)
    ]
    for p in procs:
        p.start()
    # Queue.qsize() is not implemented on macOS. An actual timed-out put proves
    # back-pressure on every platform without depending on semaphore introspection.
    assert all(event.wait(timeout=15) for event in full_events)
    return stop_evt, txs, procs


def test_stop_actors_lets_back_pressured_actors_exit_cleanly():
    stop_evt, txs, procs = _spawn(2)
    t0 = time.perf_counter()
    T.stop_actors(stop_evt, txs, procs, join_seconds=20.0)
    assert [p.exitcode for p in procs] == [0, 0]
    assert time.perf_counter() - t0 < 15.0
    for q in txs:
        q.close()


def test_stop_actors_terminates_an_actor_that_never_exits():
    ctx = mp.get_context("spawn")
    stop_evt = ctx.Event()
    p = ctx.Process(target=time.sleep, args=(60,), daemon=True)
    p.start()
    T.stop_actors(stop_evt, [], [p], join_seconds=0.5, kill_seconds=10.0)
    assert p.exitcode is not None and p.exitcode != 0
