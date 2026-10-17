"""P1 parallel replay gather keeps serial staging bytes and RNG draws exactly unchanged."""

from __future__ import annotations

import queue

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.redesign_m3c_20261008 import learner_m3c as lm
from research.redesign_m3c_20261008 import obs_pack as op
from research.redesign_m3c_20261008 import train_m3c as m3c


class _Buf:
    def __init__(self, n: int) -> None:
        self.buf = bytearray(n)


class _FlakyMulti(m3c.MultiRing):
    """Deterministically invalidate some rows only at the post-copy checks."""

    def __init__(self, rings) -> None:
        super().__init__(rings)
        self.calls = 0

    def valid(self, ids):
        self.calls += 1
        return super().valid(ids) & ((np.asarray(ids) + self.calls) % 13 != 0)


def _st():
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


def _live_obs(n, offset=0):
    """Small deterministic live rows accepted by obs-pack, with per-row variation."""
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


def _make_demos(rng, nscal, pack):
    demos = object.__new__(m3c.DemoStream)
    n = 80
    local, global_ = _live_obs(n, 1000)
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
    demos.size, demos.store = n, None
    demos._q, demos._thread = queue.Queue(), None
    return demos


def _learner(*, packed: bool, actors: int, gather_threads: int):
    rng = np.random.default_rng(710)
    nscal, cap = 30, 96
    pack = op.layout_for_arena(145, 83, 6) if packed else None
    rings = []
    for actor in range(actors):
        ring = m3c.SharedObsRing(_Buf(m3c.ring_nbytes(cap, nscal, pack)), cap, 8, nscal, pack=pack)
        # Planes, scalars, and masks all identify the source ring and row, catching wrong
        # actor routing and real packed-unpack scatter errors.
        rows = np.arange(cap, dtype=np.float32)[:, None]
        local, global_ = _live_obs(cap, actor * cap)
        ring.add(
            {
                "local": local,
                "global": global_,
                "scalars": np.repeat(rows + actor * 1000, nscal, axis=1),
            },
            (np.arange(cap)[:, None] + np.arange(6) + actor) % 3 != 0,
        )
        rings.append(ring)
    multi = _FlakyMulti(rings)
    replay = m3c.Replay(400)
    # Repeated source and successor ids force rings to be split and make compaction useful.
    s = rng.integers(8, cap, 250, dtype=np.int64) * actors + rng.integers(0, actors, 250)
    nxt = np.where(
        rng.random(250) < 0.2, -1, rng.integers(8, cap, 250) * actors + rng.integers(0, actors, 250)
    )
    replay.add_many(s, rng.integers(0, 6, 250), rng.standard_normal(250), rng.random(250), nxt)
    labels = m3c.LabelBuffer(m3c.LABEL_ANCHOR_MIN_ROWS + 40)
    label_ids = np.resize(s, m3c.LABEL_ANCHOR_MIN_ROWS + 20)
    labels.add(label_ids, rng.integers(0, 6, len(label_ids)))
    args = m3c.default_args()
    args.actor_sync_updates = 10
    args.label_frac = 0.5
    opts = lm.LearnerOpts(fast=True, fused_adam=False, gather_threads=gather_threads)
    return lm.Learner(
        torch,
        online=None,
        target=None,
        opt=None,
        replay=replay,
        multi=multi,
        demos=_make_demos(rng, nscal, pack),
        labels=labels,
        rng=np.random.default_rng([12, 17]),
        st=_st(),
        args=args,
        nscal=nscal,
        margin=0.3,
        device="cpu",
        opts=opts,
        agent_rows=48,
        anchor_rows=16,
        min_rows=8,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("actors", [2, 3])
def test_parallel_gather_staging_is_serial_byte_identical(packed, actors):
    serial = _learner(packed=packed, actors=actors, gather_threads=0)
    parallel = _learner(packed=packed, actors=actors, gather_threads=3)
    try:
        for _ in range(3):
            # Complete staging is compared, including unused tails. This removes any
            # ambiguity about rows following a compaction.
            for learner in (serial, parallel):
                for values in learner._stage:
                    for value in values.v.values():
                        value.fill(0)
            a, b = serial._sample_fast(), parallel._sample_fast()
            assert a is not None and b is not None
            assert {k: v for k, v in a.items() if k != "stage"} == {
                k: v for k, v in b.items() if k != "stage"
            }
            assert serial.rng.bit_generator.state == parallel.rng.bit_generator.state
            assert serial.multi.calls == parallel.multi.calls
            assert serial.st["ring_overrun_rows"] == parallel.st["ring_overrun_rows"]
            assert serial.st["label_rows"] == parallel.st["label_rows"]
            for stage_a, stage_b in zip(serial._stage, parallel._stage):
                for key in stage_a.v:
                    assert stage_a.v[key].tobytes() == stage_b.v[key].tobytes(), key
    finally:
        serial.close()
        parallel.close()


def test_gather_threads_validation_and_close_are_idempotent():
    with pytest.raises(ValueError, match="gather-threads"):
        lm.LearnerOpts(gather_threads=-1)
    learner = _learner(packed=True, actors=2, gather_threads=2)
    learner.close()
    learner.close()
