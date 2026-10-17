"""P1 parallel replay gather keeps serial staging bytes and RNG draws exactly unchanged."""

from __future__ import annotations

import queue

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.redesign_m3c_20261008 import learner_m3c as lm  # noqa: E402
from research.redesign_m3c_20261008 import obs_pack as op  # noqa: E402
from research.redesign_m3c_20261008 import train_m3c as m3c  # noqa: E402


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


def _legacy_gather(multi, ids, out, rows, mask_out=None):
    """Frozen pre-T17 gather oracle, intentionally independent of Learner helpers."""
    local, global_, scalars = out["local"][rows], out["global"][rows], out["scalars"][rows]
    actor, source_rows = ids % multi.k, ids // multi.k
    for actor_index, ring in enumerate(multi.rings):
        selected = np.flatnonzero(actor == actor_index)
        if not len(selected):
            continue
        pos, arrays = source_rows[selected] % ring.cap, ring.arrays
        if ring.pack is None:
            local[selected] = arrays["local"][pos]
            global_[selected] = arrays["global"][pos]
        else:
            op.unpack_into(arrays["packed"], pos, local, global_, selected, ring.pack)
        scalars[selected] = arrays["scalars"][pos]
        if mask_out is not None:
            mask_out[selected] = arrays["mask"][pos]


def _legacy_sample_fast(learner):
    """Frozen pre-T17 ``_sample_fast`` for the fixture's demo + label configuration."""
    st, rng, multi, replay = learner.st, learner.rng, learner.multi, learner.replay
    demos, labels, args = learner.demos, learner.labels, learner.args
    idx = replay.sample(rng, learner.a_rows, multi)
    if len(idx) < learner.min_rows:
        return None
    stage, v = learner._stage[learner._which], learner._stage[learner._which].v
    n0, s_ids, nxt = len(idx), replay.s[idx], replay.nxt[idx]
    _legacy_gather(multi, np.where(nxt >= 0, nxt, s_ids), v, slice(0, n0), v["nmask"][:n0])
    _legacy_gather(multi, s_ids, v, slice(n0, 2 * n0))
    ok = multi.valid(s_ids) & ((nxt < 0) | multi.valid(nxt))
    n = n0
    if not ok.all():
        st["ring_overrun_rows"] += int((~ok).sum())
        keep = np.flatnonzero(ok)
        idx, n = idx[keep], len(keep)
        if n < learner.min_rows:
            return None
        for key in ("local", "global", "scalars"):
            v[key][:n] = v[key][:n0][keep]
            v[key][n : 2 * n] = v[key][n0 : 2 * n0][keep]
        v["nmask"][:n] = v["nmask"][:n0][keep]
    v["a"][:n], v["R"][:n], v["disc"][:n] = replay.a[idx], replay.R[idx], replay.disc[idx]
    n_lab = int(round(learner.anchor_rows * args.label_frac))
    base = 2 * n
    n_demo = learner.anchor_rows - n_lab
    di = rng.integers(0, demos.size, size=n_demo)
    demos.gather_into(di, v, base)
    li = labels.sample(rng, n_lab, multi)
    lids, m0, lo = labels.ids[li], len(li), base + n_demo
    _legacy_gather(multi, lids, v, slice(lo, lo + m0), v["mask_resolved"][n_demo : n_demo + m0])
    keep = np.flatnonzero(multi.valid(lids))
    if len(keep) < m0:
        for key in ("local", "global", "scalars"):
            v[key][lo : lo + len(keep)] = v[key][lo : lo + m0][keep]
        masks = v["mask_resolved"]
        masks[n_demo : n_demo + len(keep)] = masks[n_demo : n_demo + m0][keep]
    v["a_v8"][n_demo : n_demo + len(keep)] = labels.a[li][keep]
    n_lab = len(keep)
    st["label_rows"].append(n_lab)
    return {"stage": stage, "n": n, "m": n_demo + n_lab, "n_demo": n_demo}


def _record_sample_indices(learner):
    """Capture every stochastic replay/label selection without changing the calls."""
    calls = []
    replay_sample, label_sample = learner.replay.sample, learner.labels.sample

    def replay_spy(*args):
        value = replay_sample(*args)
        calls.append(("replay", value.copy()))
        return value

    def label_spy(*args):
        value = label_sample(*args)
        calls.append(("label", value.copy()))
        return value

    learner.replay.sample, learner.labels.sample = replay_spy, label_spy
    return calls


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
        # Publish a second full pass so sampled ids cross the physical-ring wrap.
        local, global_ = _live_obs(cap, (actor + actors) * cap)
        ring.add(
            {
                "local": local,
                "global": global_,
                "scalars": np.repeat(rows + (actor + actors) * 1000, nscal, axis=1),
            },
            (np.arange(cap)[:, None] + np.arange(6) + actor + 1) % 3 != 0,
        )
        rings.append(ring)
    multi = _FlakyMulti(rings)
    replay = m3c.Replay(400)
    # Repeated source and successor ids force rings to be split and make compaction useful.
    s = rng.integers(cap + 8, 2 * cap, 250, dtype=np.int64) * actors + rng.integers(0, actors, 250)
    nxt = np.where(
        rng.random(250) < 0.2,
        -1,
        rng.integers(cap + 8, 2 * cap, 250) * actors + rng.integers(0, actors, 250),
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
@pytest.mark.parametrize("actors", [1, 2, 3])
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


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("actors", [1, 2, 3])
def test_parallel_gather_matches_frozen_legacy_oracle(packed, actors):
    legacy = _learner(packed=packed, actors=actors, gather_threads=0)
    parallel = _learner(packed=packed, actors=actors, gather_threads=4)
    try:
        ids = legacy.replay.s[:48]
        for learner in (legacy, parallel):
            for stage in learner._stage:
                for value in stage.v.values():
                    value.fill(0)
        legacy_v, parallel_v = legacy._stage[0].v, parallel._stage[0].v
        _legacy_gather(
            legacy.multi, ids, legacy_v, slice(0, len(ids)), legacy_v["nmask"][: len(ids)]
        )
        parallel._gather(ids, parallel_v, slice(0, len(ids)), parallel_v["nmask"][: len(ids)])
        for key in legacy_v:
            assert legacy_v[key].tobytes() == parallel_v[key].tobytes(), key
    finally:
        legacy.close()
        parallel.close()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("actors", [1, 2, 3])
@pytest.mark.parametrize("threads", [2, 3, 4])
def test_parallel_sample_matches_frozen_pre_t17_complete_staging(packed, actors, threads):
    """Hard P1 invariant: old sampling/gather and threaded gather produce identical bytes."""
    legacy = _learner(packed=packed, actors=actors, gather_threads=0)
    parallel = _learner(packed=packed, actors=actors, gather_threads=threads)
    try:
        for learner in (legacy, parallel):
            for stage in learner._stage:
                for value in stage.v.values():
                    value.fill(0)
        legacy_calls, parallel_calls = _record_sample_indices(legacy), _record_sample_indices(
            parallel
        )
        old, new = _legacy_sample_fast(legacy), parallel._sample_fast()
        assert old is not None and new is not None
        assert {key: value for key, value in old.items() if key != "stage"} == {
            key: value for key, value in new.items() if key != "stage"
        }
        assert len(legacy_calls) == len(parallel_calls)
        for (old_name, old_ids), (new_name, new_ids) in zip(legacy_calls, parallel_calls):
            assert old_name == new_name
            assert old_ids.tobytes() == new_ids.tobytes()
        assert legacy.rng.bit_generator.state == parallel.rng.bit_generator.state
        assert legacy.st["ring_overrun_rows"] == parallel.st["ring_overrun_rows"]
        assert legacy.st["label_rows"] == parallel.st["label_rows"]
        for old_stage, new_stage in zip(legacy._stage, parallel._stage):
            for key in old_stage.v:
                assert old_stage.v[key].tobytes() == new_stage.v[key].tobytes(), key
    finally:
        legacy.close()
        parallel.close()


def test_gather_waits_for_an_interrupted_future_before_reraising():
    """An interrupt must not leave a submitted staging writer running in the background."""

    class _InterruptedFuture:
        def __init__(self):
            self.calls = 0
            self.finished = False

        def done(self):
            return self.finished

        def result(self):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            self.finished = True

    future = _InterruptedFuture()
    with pytest.raises(KeyboardInterrupt):
        lm.Learner._wait_for_gather_jobs([future])
    assert future.finished and future.calls >= 2
