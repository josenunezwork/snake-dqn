"""perf-actor (2026-10-13): the vectorized actor bookkeeping equals the reference loops.

* ``NStepBuilder`` (vectorized) == ``NStepBuilderRef`` (per-world deques): the same
  transitions in the same order with bit-identical ``R`` / ``disc``, on random streams with
  deaths, missing rows, varying gamma, segment ends and several n.
* ``sample_legal`` == ``sample_legal_ref`` (per-row ``rng.choice``): the same actions AND
  the same generator state afterwards.
* ``Replay.add_many`` == a loop of ``Replay.add`` (including wrap-around).
* End to end: ``actor_main`` with ``--actor-impl reference`` and ``vectorized`` (and the
  numba vector61 featurizer) emits identical frame messages on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from research.redesign_m3c_20261008 import train_m3c as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8_nb import HAVE_CKPT  # noqa: E402


def _assert_take_equal(a, b, ctx):
    for x, y, name in zip(a, b, ("s", "a", "R", "disc", "nxt")):
        assert x.dtype == y.dtype, (ctx, name, x.dtype, y.dtype)
        assert x.shape == y.shape, (ctx, name, x.shape, y.shape)
        if x.dtype == np.float32:
            assert np.array_equal(x.view(np.uint32), y.view(np.uint32)), (ctx, name)
        else:
            assert np.array_equal(x, y), (ctx, name)


@pytest.mark.parametrize("n_step", [1, 2, 5, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nstep_vectorized_matches_reference(n_step, seed):
    rng = np.random.default_rng([seed, n_step])
    E = 23
    total = 0
    deaths_seen = 0
    for segment in range(4):
        ref, vec = T.NStepBuilderRef(E, n_step), T.NStepBuilder(E, n_step)
        frames = int(rng.integers(1, 60))
        next_id = 1000 * segment
        for frame in range(frames):
            present = rng.random(E) < 0.85  # some worlds have no hero row this frame
            worlds = np.flatnonzero(present)
            ids = next_id + np.arange(len(worlds), dtype=np.int64)
            next_id += len(worlds)
            acts = rng.integers(0, 6, size=len(worlds))
            if frame % 2:
                reward = rng.normal(size=E).astype(np.float32)
            else:
                reward = rng.normal(size=E) * 10.0 ** rng.integers(-3, 3)
            done = rng.random(E) < 0.08
            gamma = float(rng.choice([0.99, 0.995, 0.9921875, rng.uniform(0.9, 1.0)]))
            d_ref = ref.step(worlds, ids, acts, reward, done, gamma)
            d_vec = vec.step(worlds, ids, acts, reward, done, gamma)
            assert d_ref == d_vec
            deaths_seen += d_ref
            if frame == frames - 1:
                g_end = float(rng.uniform(0.9, 1.0))
                ref.end_segment(g_end)
                vec.end_segment(g_end)
            a, b = ref.take(), vec.take()
            total += len(a[0])
            _assert_take_equal(a, b, (seed, n_step, segment, frame))
    assert total > 300 and deaths_seen > 10


def test_nstep_vectorized_empty_and_last_obs():
    b = T.NStepBuilder(4)
    assert b.step(np.zeros(0, np.int64), [], [], np.zeros(4), np.zeros(4, bool), 0.99) == 0
    out = b.take()
    assert all(len(x) == 0 for x in out)
    b.step(np.array([1, 3]), [10, 11], [0, 1], np.ones(4), np.zeros(4, bool), 0.99)
    assert b.last_obs.tolist() == [-1, 10, -1, 11]
    b.end_segment(0.99)
    assert all(len(x) == 0 for x in b.take())  # the only row per world has no successor


@pytest.mark.parametrize("seed", range(6))
def test_sample_legal_matches_choice_loop(seed):
    g = np.random.default_rng(seed)
    for n in [1, 2, 3, 7, 64, 257]:
        mm = g.random((n, 6)) < g.uniform(0.05, 0.95)
        mm[~mm.any(axis=1), :3] = True
        mm[0] = False
        mm[0, int(g.integers(0, 6))] = True  # a single legal action
        if n > 1:
            mm[1] = True
        r1 = np.random.default_rng([seed, n, 17])
        r2 = np.random.default_rng([seed, n, 17])
        a = T.sample_legal_ref(r1, mm)
        b = T.sample_legal(r2, mm)
        assert np.array_equal(a, b)
        assert b.dtype == np.int64 and mm[np.arange(n), b].all()
        assert r1.bit_generator.state == r2.bit_generator.state
        assert r1.random() == r2.random()  # the streams continue identically


def test_replay_add_many_matches_add_loop():
    g = np.random.default_rng(0)
    r1, r2 = T.Replay(50), T.Replay(50)
    for n in [0, 7, 13, 49, 3, 120, 1]:
        s = g.integers(0, 1000, size=n)
        a = g.integers(0, 6, size=n)
        R = g.normal(size=n).astype(np.float32)
        d = g.random(n).astype(np.float32)
        nx = np.where(g.random(n) < 0.2, -1, g.integers(0, 1000, size=n))
        for i in range(n):
            r1.add(int(s[i]), int(a[i]), float(R[i]), float(d[i]), int(nx[i]))
        r2.add_many(s, a, R, d, nx)
        for k in ("s", "a", "R", "disc", "nxt"):
            assert np.array_equal(getattr(r1, k), getattr(r2, k)), (n, k)
        assert (r1.ptr, r1.size, r1.added) == (r2.ptr, r2.size, r2.added)


# ----------------------------------------------------------------------------- end to end
class _Stop:
    def __init__(self) -> None:
        self.flag = False

    def is_set(self) -> bool:
        return self.flag

    def set(self) -> None:
        self.flag = True


class _Collect:
    def __init__(self, frames: int, stop: _Stop) -> None:
        self.frames, self.stop, self.msgs = frames, stop, []

    def put(self, msg, timeout: float = 0.0) -> None:
        self.msgs.append(msg)
        if len(self.msgs) >= self.frames:
            self.stop.set()


def _run_actor(tmp_path, init, impl, v61, frames, extra=()):
    argv = [
        "--seed",
        "5",
        "--init",
        str(init),
        "--demo-data",
        ".",
        "--out",
        str(tmp_path / "unused"),
        "--actors",
        "1",
        "--worlds",
        "6",
        "--frozen-frac",
        "0.34",
        "--mixed-frac",
        "0.33",
        "--segment-frames",
        "45",
        "--device",
        "cpu",
        "--actor-impl",
        impl,
        "--v61-backend",
        v61,
        *extra,
    ]
    args = T.build_parser().parse_args(argv)
    nscal = T.nscal_for(args)
    margin = args.worlds * 16
    ring_cap = args.worlds * (frames + 4) + margin
    net = T.load_init(args, "cpu")
    spec = T._weights_spec(net)
    n_weights = int(sum(n for _, _, n in spec))
    blocks = [
        T.Block.create(T.ring_nbytes(ring_cap, nscal), str(tmp_path), "ring0"),
        T.Block.create(8 + 8 * (n_weights + 1), str(tmp_path), "weights"),
        T.Block.create(32, str(tmp_path), "progress"),
    ]
    names = {"ring0": blocks[0].desc, "weights": blocks[1].desc, "progress": blocks[2].desc}
    wver = np.ndarray((1,), dtype=np.int64, buffer=blocks[1].buf, offset=0)
    wver[0] = 0
    wbuf = np.ndarray((n_weights + 1,), dtype=np.float64, buffer=blocks[1].buf, offset=8)
    prog = np.ndarray((4,), dtype=np.int64, buffer=blocks[2].buf, offset=0)
    prog[:] = 0
    prog[0] = 9_000_000  # past --shield-full-until: p_shield < 1 (exercises its RNG draws)
    T._publish_weights(net, wbuf, wver)
    cfg = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "K": 1,
        "ring_cap": ring_cap,
        "ring_margin": margin,
        "n_weights": n_weights,
        "actor_device": "cpu",
    }
    stop = _Stop()
    sink = _Collect(frames, stop)
    T.actor_main(0, cfg, names, sink, stop)
    rows = T.SharedObsRing(blocks[0], ring_cap, margin, nscal)
    ring = {k: v.copy() for k, v in rows.arrays.items()}
    del wver, wbuf, prog, rows
    for b in blocks:
        b.unlink()
        b.close()
    return sink.msgs, ring


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
@pytest.mark.parametrize(
    "extra",
    [
        (),
        (
            "--shield",
            "v8",
            "--shield-full-until",
            "8000000",
            "--shield-off-at",
            "10000000",
            "--onpolicy-labels",
            "--anchor-mode",
            "v8margin",
            "--v8-backend",
            "numba",
        ),
    ],
    ids=["plain", "shield-labels"],
)
def test_actor_messages_identical_across_impls(tmp_path, monkeypatch, extra):
    import torch

    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    # eps 0.3 (the default schedule's 0.01..0.05 explores too rarely to pin the RNG stream)
    monkeypatch.setattr(T.Schedules, "eps", lambda self, t: 0.3)

    torch.manual_seed(0)
    init = tmp_path / "init.pth"
    save_ego2s_checkpoint(init, Ego2sNet(7, 30))
    frames = 100  # crosses two segment ends (45 frames each)
    runs = {}
    for impl, v61 in (("reference", "python"), ("vectorized", "python"), ("vectorized", "numba")):
        d = tmp_path / f"{impl}-{v61}"
        d.mkdir()
        runs[(impl, v61)] = _run_actor(d, init, impl, v61, frames, extra)
    ref_msgs, ref_ring = runs[("reference", "python")]
    assert len(ref_msgs) == frames
    assert sum(len(m["s"]) for m in ref_msgs) > 300 and sum(m["deaths"] for m in ref_msgs) >= 0
    skip = {"env_s", "timing", "j"}
    for key, (msgs, ring) in runs.items():
        assert len(msgs) == len(ref_msgs), key
        for f, (a, b) in enumerate(zip(ref_msgs, msgs)):
            assert a.keys() == b.keys(), (key, f)
            for k in a:
                if k in skip:
                    continue
                x, y = a[k], b[k]
                if isinstance(x, np.ndarray):
                    assert x.dtype == y.dtype and np.array_equal(x, y), (key, f, k)
                else:
                    assert x == y, (key, f, k)
        for k in ref_ring:
            assert np.array_equal(ref_ring[k], ring[k]), (key, k)
