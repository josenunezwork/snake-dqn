"""Synthetic P1 host-gather micro-benchmark; it is development evidence, not a gate.

Example (the smoke-03-sized learner batch):
    ./venv/bin/python research/redesign_m3c_20261008/bench_gather.py --packed --actors 2
"""

from __future__ import annotations

import argparse
import mmap
import queue
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m3c_20261008 import learner_m3c as lm
from research.redesign_m3c_20261008 import obs_pack as op
from research.redesign_m3c_20261008 import train_m3c as m3c


class _Buf:
    def __init__(self, n: int) -> None:
        # Anonymous mmap keeps the synthetic gather's backing-store behavior close to
        # the process-shared actor rings without creating files or touching tmux runs.
        self.buf = mmap.mmap(-1, n)


def _stats():
    return {
        "updates": 0,
        "halt": None,
        "td": [],
        "anc": [],
        "qmax": 0.0,
        "learn_s": 0.0,
        "ring_overrun_rows": 0,
        "label_rows": [],
        "anchor_updates": 0,
    }


def _demos(rng, nscal, pack, n):
    demos = object.__new__(m3c.DemoStream)
    local = np.zeros((n,) + m3c.LOCAL, dtype=np.uint8)
    global_ = np.zeros((n,) + m3c.GLOBAL, dtype=np.uint8)
    demos.buf = {
        "scalars": rng.standard_normal((n, nscal)).astype(np.float32),
        "q_teacher": rng.standard_normal((n, 6)).astype(np.float32),
        "a_v8": rng.integers(0, 6, n),
        "mask_resolved": rng.random((n, 6)) < 0.7,
    }
    if pack is None:
        demos.buf.update({"local": local, "global": global_})
    else:
        demos.pack = pack
        demos.buf["packed"] = op.pack(local, global_, pack)
    demos.store, demos.size = None, n
    demos._q, demos._thread = queue.Queue(), None
    return demos


def _live_obs(n, offset=0):
    """Deterministic non-dead rows that exercise the packed decoder's real fields."""
    local = np.zeros((n,) + m3c.LOCAL, dtype=np.uint8)
    global_ = np.zeros((n,) + m3c.GLOBAL, dtype=np.uint8)
    row = np.arange(n) + offset
    local[np.arange(n), 0, row % 31, (row * 3) % 31] = 17 + (row % 200)
    local[np.arange(n), 2, (row * 5) % 31, (row * 7) % 31] = 255
    local[np.arange(n), 3, (row * 11) % 31, (row * 13) % 31] = 128
    local[np.arange(n), 4, (row * 17) % 31, (row * 19) % 31] = 255
    local[np.arange(n), 5, (row * 23) % 31, (row * 29) % 31] = row % 16
    local[np.arange(n), 6, (row * 2) % 31, (row * 5) % 31] = 128
    global_[:, 3] = 255
    global_[:, 3, 8:19, 9:28] = 0
    global_[np.arange(n), 0, 8 + row % 11, 9 + (row * 3) % 19] = 71
    return local, global_


def build(*, packed: bool, actors: int, threads: int, seed: int):
    """A fixed 768-agent + 256-anchor staging benchmark with synthetic multi-rings."""
    rng, nscal, cap = np.random.default_rng(seed), 30, 4096
    pack = op.layout_for_arena(145, 83, 6) if packed else None
    rings = []
    for actor in range(actors):
        ring = m3c.SharedObsRing(_Buf(m3c.ring_nbytes(cap, nscal, pack)), cap, 32, nscal, pack=pack)
        local, global_ = _live_obs(cap, actor * cap)
        ring.add(
            {
                "local": local,
                "global": global_,
                "scalars": rng.standard_normal((cap, nscal)).astype(np.float32),
            },
            rng.random((cap, 6)) < 0.7,
        )
        rings.append(ring)
    multi = m3c.MultiRing(rings)
    replay = m3c.Replay(8192)
    n = 6000
    s = rng.integers(32, cap, n) * actors + rng.integers(0, actors, n)
    nxt = np.where(
        rng.random(n) < 0.1, -1, rng.integers(32, cap, n) * actors + rng.integers(0, actors, n)
    )
    replay.add_many(s, rng.integers(0, 6, n), rng.standard_normal(n), rng.random(n), nxt)
    labels = m3c.LabelBuffer(m3c.LABEL_ANCHOR_MIN_ROWS + 256)
    label_ids = np.resize(s, m3c.LABEL_ANCHOR_MIN_ROWS + 128)
    labels.add(label_ids, rng.integers(0, 6, len(label_ids)))
    args = m3c.default_args()
    args.actor_sync_updates = 100
    args.label_frac = 1.0
    opts = lm.LearnerOpts(fast=True, fused_adam=False, gather_threads=threads, batch_scale=2)
    return lm.Learner(
        torch,
        online=None,
        target=None,
        opt=None,
        replay=replay,
        multi=multi,
        demos=None,
        labels=labels,
        rng=np.random.default_rng([seed, 73]),
        st=_stats(),
        args=args,
        nscal=nscal,
        margin=0.3,
        device="cpu",
        opts=opts,
    )


def run(args) -> None:
    print("threads  median_ms/update  mean_ms/update  samples")
    for threads in (0, 2, 3, 4):
        learner = build(packed=args.packed, actors=args.actors, threads=threads, seed=args.seed)
        try:
            for _ in range(args.warmup):
                assert learner._sample_fast() is not None
            elapsed = []
            for _ in range(args.updates):
                t0 = time.perf_counter()
                assert learner._sample_fast() is not None
                elapsed.append((time.perf_counter() - t0) * 1000.0)
            print(
                f"{threads:7d}  {np.median(elapsed):16.3f}  {np.mean(elapsed):14.3f}  {len(elapsed):7d}"
            )
        finally:
            learner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packed", action="store_true", help="use obs-pack rings (smoke-03 representative)"
    )
    parser.add_argument("--actors", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.actors < 1 or args.warmup < 0 or args.updates < 1:
        parser.error("actors must be positive, warmup non-negative, and updates positive")
    run(args)


if __name__ == "__main__":
    main()
