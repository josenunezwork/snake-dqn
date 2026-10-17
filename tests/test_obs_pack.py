"""perf-ram (2026-10-13): ``--obs-pack`` and ``--demo-mmap`` change storage, not data.

* :mod:`obs_pack`: packing is lossless on played GridBatchSim worlds (dead rows included)
  and on random rows that respect the plane invariants; numba == NumPy reference; a row
  that breaks an invariant raises instead of being stored; the round-trip check fails on a
  deliberately broken unpack (so the parity assertions here are not vacuous).
* The learner: packed rings (and packed / store-backed demonstration buffers) give
  byte-identical gathered batches and therefore bitwise-identical CPU training (reference
  and T1 fast learner), including the ring-overrun compaction path.
* :class:`DemoStream`: raw, packed, store (raw) and store (packed) buffers hold the same
  rows in the same slots through ``fill_initial`` / ``prefetch`` / ``swap_if_ready`` and
  return identical ``sample`` / ``gather_into`` bytes with identical RNG streams.
* Both parity checks are shown to FAIL on deliberate breaks (planes scattered into the wrong
  staging rows on either learner path; an off-by-one store row index; packed demonstration
  planes gathered from the wrong slots).
* End to end: ``actor_main`` writes the same observations into a packed ring as into a raw
  one (slow; needs the pinned checkpoints).
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path

import numpy as np
import pytest

from research.redesign_m3c_20261008 import obs_pack as op
from research.redesign_m3c_20261008 import train_m3c as m3c

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8_nb import HAVE_CKPT  # noqa: E402

LAYOUT = op.layout_for_arena(145, 83, 6)


def test_train_layout_without_checkpoint_pool(tmp_path, monkeypatch):
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    monkeypatch.setattr(gid, "CHECKPOINT_DIR", tmp_path / "absent-pool")

    def no_evaluation_context(*args, **kwargs):
        raise AssertionError("layout must not build an evaluation context")

    monkeypatch.setattr(gid, "_context", no_evaluation_context)
    layout = op.train_layout()
    assert layout == LAYOUT
    assert layout.nbytes == 3064


def test_train_layout_matches_checkpoint_verified_legacy_layout():
    from research.apex_safety_20260926 import dev_screen as ds
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    if not all((gid.CHECKPOINT_DIR / name).is_file() for name, _ in (*ds.POOL, gid.FRP3_S12)):
        pytest.skip("needs the complete pinned checkpoint pool")
    world = gid._context(1)["profile"].world
    legacy = op.layout_for_arena(
        world.width // world.segment_size, world.height // world.segment_size, world.num_snakes
    )
    actual = op.train_layout()
    assert actual == legacy
    assert actual.consts() == legacy.consts()
    local, global_ = valid_rows(np.random.default_rng(17), 16, legacy)
    assert op.pack(local, global_, actual).tobytes() == op.pack(local, global_, legacy).tobytes()


# ----------------------------------------------------------------------------- data
def valid_rows(rng: np.random.Generator, n: int, layout=LAYOUT, dead_every: int = 7):
    """Random ``(local, global)`` rows that satisfy every packing invariant (extremes too)."""
    L = np.zeros((n,) + op.LOCAL_SHAPE, dtype=np.uint8)
    G = np.zeros((n,) + op.GLOBAL_SHAPE, dtype=np.uint8)
    for i in range(n):
        if dead_every and i % dead_every == 0:
            continue  # a dead row: every plane zero
        ttl = rng.integers(0, 256, (31, 31), dtype=np.uint8)
        occ = rng.random((31, 31)) < 0.5
        own = rng.random((31, 31)) < 0.4
        L[i, 0] = np.where(occ & own, ttl, 0)
        L[i, 1] = np.where(occ & ~own, ttl, 0)
        k = int(rng.integers(0, layout.head_cap + 1))
        pix = rng.choice(961, size=k, replace=False)
        L[i, 2].reshape(-1)[pix] = rng.integers(1, 256, k, dtype=np.uint8)
        L[i, 3] = rng.choice(np.array([0, 128, 255], np.uint8), (31, 31))
        L[i, 4] = rng.choice(np.array([0, 255], np.uint8), (31, 31))
        L[i, 5] = rng.integers(0, op.REACH_MAX + 1, (31, 31), dtype=np.uint8)
        L[i, 6] = rng.choice(np.array([0, 128, 255], np.uint8), (31, 31))
        h = int(rng.integers(0, 38))
        w = int(rng.integers(0, min(37, layout.rect_cells // max(h, 1)) + 1))
        r0, c0 = int(rng.integers(0, 38 - h)), int(rng.integers(0, 38 - w))
        G[i, 3] = 255
        G[i, 3, r0 : r0 + h, c0 : c0 + w] = 0
        G[i, :3, r0 : r0 + h, c0 : c0 + w] = rng.integers(0, 256, (3, h, w), dtype=np.uint8)
    return L, G


def _played_rows(mechanics: int, frames: int = 50):
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.grid_scenarios import GreedySafePolicy, inject_serpentines
    from src.simd_env.grid_sim import GridBatchSim

    cfg = BatchSimConfig(num_envs=3, num_snakes=6, mechanics_version=mechanics, max_capacity=1200)
    sim = GridBatchSim(cfg, seeds=[31, 32, 33], train_mode=True, allow_respawn=True)
    assert op.layout_for_arena(sim.grid_w, sim.grid_h, sim.S) == LAYOUT
    inject_serpentines([sim], [800, 400, 150, 60, 20, 5], box_height=8)
    pol = GreedySafePolicy(5, boost_prob=0.4)
    B = EgoRasterConfig(version="b")
    locs, globs = [], []
    for _ in range(frames):
        sim.step(pol.actions(sim, sim.get_action_mask()))
        obs = build_ego_raster(sim, B, backend="numba")
        locs.append(obs["local"].reshape((-1,) + op.LOCAL_SHAPE))
        globs.append(obs["global"].reshape((-1,) + op.GLOBAL_SHAPE))
    return np.concatenate(locs), np.concatenate(globs)


def _round_trip(L, G, layout=LAYOUT):
    p = op.pack(L, G, layout)
    np.testing.assert_array_equal(p, op.pack_ref(L, G, layout))
    L2, G2 = op.unpack(p, np.arange(len(p)), layout)
    L3, G3 = op.unpack_ref(p, layout)
    return p, (L2, G2), (L3, G3)


# ----------------------------------------------------------------------------- obs_pack
def test_training_arena_layout():
    assert (LAYOUT.rect_cells, LAYOUT.head_cap, LAYOUT.nbytes) == (19 * 11, 8, 3064)
    assert op.RAW_OBS_BYTES == 12203
    from src.simd_env.ego_raster import EgoRasterConfig

    with pytest.raises(ValueError, match="5 bits"):
        op.layout_for_arena(145, 83, 6, EgoRasterConfig(version="b", reach_steps=40))


def test_buffer_bytes_per_row():
    """The RAM model's inputs (research/perf_ram_20261013): bytes per stored row."""
    assert m3c.ring_nbytes(1, 30) - 8 == 12203 + 120 + 6
    assert m3c.ring_nbytes(1, 30, LAYOUT) - 8 == 3064 + 120 + 6
    assert m3c.ring_nbytes(1, 60, LAYOUT) - 8 == 3064 + 240 + 6
    assert m3c.demo_row_nbytes() == 12203 + 120 + 24 + 8 + 6
    assert m3c.demo_row_nbytes(LAYOUT) == 3064 + 120 + 24 + 8 + 6
    # production rings: 2 actors x 377,048 rows at replay 500k / 64 worlds
    assert m3c.ring_capacity(500_000, 2, 64, 8) == (375_000 + 64 * 16 + 64 * 16, 64 * 16)
    s = object.__new__(m3c.DemoStream)
    assert s.pack is None and s.store is None  # class defaults (object.__new__ users)


@pytest.mark.parametrize("mechanics", [1, 2])
def test_lossless_on_played_worlds(mechanics):
    L, G = _played_rows(mechanics)
    assert (~L.reshape(len(L), -1).any(axis=1)).sum() > 0  # some dead (all-zero) rows
    assert (L[:, 2] > 0).sum() > 0 and (L[:, 6] == 255).sum() > 0 and (L[:, 0] == 255).any()
    _, (L2, G2), (L3, G3) = _round_trip(L, G)
    for a, b in ((L2, L), (G2, G), (L3, L), (G3, G)):
        assert a.dtype == np.uint8 and np.array_equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lossless_on_random_valid_rows(seed):
    rng = np.random.default_rng(seed)
    L, G = valid_rows(rng, 300)
    big = op.PackLayout(rect_cells=37 * 37, head_cap=20)  # full-window rectangles too
    Lb, Gb = valid_rows(rng, 100, layout=big)
    for lay, (x, y) in ((LAYOUT, (L, G)), (big, (Lb, Gb))):
        _, (L2, G2), (L3, G3) = _round_trip(x, y, lay)
        assert np.array_equal(L2, x) and np.array_equal(G2, y)
        assert np.array_equal(L3, x) and np.array_equal(G3, y)


def _break(code: int, L, G):
    if code == 1:
        L[0, 0, 3, 3], L[0, 1, 3, 3] = 5, 6
    elif code == 2:
        L[0, 3, 1, 1] = 7
    elif code == 3:
        L[0, 4, 1, 1] = 128
    elif code == 4:
        L[0, 5, 1, 1] = 32
    elif code == 5:
        L[0, 6, 1, 1] = 64
    elif code == 6:
        L[0, 2].reshape(-1)[: LAYOUT.head_cap + 1] = 9
    elif code == 7:
        G[0, 3, 0, 0] = 1
    elif code == 8:
        G[0, 3, 0, 0] = 0  # a stray in-arena cell far from the rectangle
    elif code == 9:
        G[0, 1, 0, 0] = 3  # mass on an outside cell
    elif code == 10:
        G[0, 3, 2:30, 2:30] = 0  # rectangle bigger than rect_cells


@pytest.mark.parametrize("code", sorted(op.VIOLATIONS))
@pytest.mark.parametrize("impl", ["numba", "numpy"])
def test_violations_raise(code, impl):
    L = np.zeros((1,) + op.LOCAL_SHAPE, dtype=np.uint8)
    G = np.full((1,) + op.GLOBAL_SHAPE, 0, dtype=np.uint8)
    G[0, 3] = 255
    G[0, 3, 10:21, 9:28] = 0  # the 11 x 19 in-arena rectangle
    G[0, 0, 12, 12] = 4
    op.pack(L, G, LAYOUT)  # valid before the break
    _break(code, L, G)
    fn = op.pack if impl == "numba" else op.pack_ref
    with pytest.raises(ValueError, match="cannot be packed losslessly"):
        fn(L, G, LAYOUT)
    if impl == "numba":
        with pytest.raises(ValueError, match=op.VIOLATIONS[code].split(" ")[0]):
            fn(L, G, LAYOUT)


def test_round_trip_check_fails_on_a_deliberately_broken_unpack(monkeypatch):
    """The equality checks used in this file detect a broken unpack (not vacuous)."""
    L, G = valid_rows(np.random.default_rng(5), 40)
    p = op.pack(L, G, LAYOUT)
    # 1) one flipped bit in one stored row
    q = p.copy()
    q[3, LAYOUT.off_c + 100] ^= 1
    assert not np.array_equal(op.unpack(q, np.arange(len(q)), LAYOUT)[0], L)
    # 2) a wrong value table inside the numba kernel (food / enemy_next codes)
    monkeypatch.setattr(op, "CODES", np.array([0, 128, 254, 0], dtype=np.uint8))
    L2, G2 = op.unpack(p, np.arange(len(p)), LAYOUT)
    assert np.array_equal(G2, G) and not np.array_equal(L2, L)


def test_unpack_into_scatters_rows_in_place():
    L, G = valid_rows(np.random.default_rng(9), 30)
    p = op.pack(L, G, LAYOUT)
    outL = np.full((50,) + op.LOCAL_SHAPE, 7, dtype=np.uint8)
    outG = np.full((50,) + op.GLOBAL_SHAPE, 7, dtype=np.uint8)
    src = np.array([4, 0, 29, 4, 17])
    dst = np.array([39, 3, 0, 10, 11])  # rows of the slice view outL[10:]
    op.unpack_into(p, src, outL[10:], outG[10:], dst, LAYOUT)
    for s, d in zip(src, dst):
        assert np.array_equal(outL[10 + d], L[s]) and np.array_equal(outG[10 + d], G[s])
    untouched = np.setdiff1d(np.arange(50), 10 + dst)
    assert (outL[untouched] == 7).all() and (outG[untouched] == 7).all()
    with pytest.raises(ValueError):
        op.unpack_into(p, src, outL[:, :, ::2], outG, dst, LAYOUT)


# ----------------------------------------------------------------------------- learner
CAP = 1500
K_ACTORS = 2


class _Buf:
    def __init__(self, n: int) -> None:
        self.buf = bytearray(n)


class FlakyMulti(m3c.MultiRing):
    """valid() also drops a call-dependent id subset: exercises the post-gather re-checks."""

    def __init__(self, rings) -> None:
        super().__init__(rings)
        self.calls = 0

    def valid(self, ids):
        self.calls += 1
        ok = super().valid(ids)
        return ok & ((np.asarray(ids) + self.calls) % 11 != 0)


def _demo_rows(r, size):
    L, G = valid_rows(r, size)
    return {
        "local": L,
        "global": G,
        "scalars": r.standard_normal((size, 30)).astype(np.float32),
        "q_teacher": r.standard_normal((size, 6)).astype(np.float32),
        "a_v8": r.integers(0, 6, size),
        "mask_resolved": r.random((size, 6)) < 0.7,
    }


def _world(mode: str, packed: bool, seed: int = 0):
    nscal = 60 if mode == "bv8" else 30
    pack = LAYOUT if packed else None
    r = np.random.default_rng(seed)
    rings = []
    for _ in range(K_ACTORS):
        ring = m3c.SharedObsRing(_Buf(m3c.ring_nbytes(CAP, nscal, pack)), CAP, 16, nscal, pack=pack)
        L, G = valid_rows(r, CAP)
        mask = r.random((CAP, 6)) < 0.7
        scal = r.standard_normal((CAP, nscal)).astype(np.float32)
        ring.add({"local": L, "global": G, "scalars": scal}, mask)
        assert ring.count == CAP
        rings.append(ring)
    multi = FlakyMulti(rings)
    replay = m3c.Replay(4000)
    _ingest(replay, r, 2000)
    labels = demos = None
    if mode == "bv8":
        labels = m3c.LabelBuffer(m3c.LABEL_ANCHOR_MIN_ROWS + 500)
        n = m3c.LABEL_ANCHOR_MIN_ROWS + 100
        labels.add(r.integers(0, CAP * K_ACTORS, n), r.integers(0, 6, n))
    else:
        rows = _demo_rows(r, 600)
        demos = object.__new__(m3c.DemoStream)
        if packed:
            demos.pack = pack
            rows["packed"] = op.pack(rows.pop("local"), rows.pop("global"), pack)
        demos.buf = rows
        demos.size = 600
        demos._q = queue.Queue()
        demos._thread = None
        demos.prefetch = lambda: None
    return nscal, multi, replay, labels, demos


def _ingest(replay, r, n):
    s = r.integers(0, CAP * K_ACTORS, n)
    nxt = np.where(r.random(n) < 0.1, -1, r.integers(0, CAP * K_ACTORS, n))
    replay.add_many(s, r.integers(0, 6, n), r.standard_normal(n), r.random(n) * 0.9, nxt)


def _run(mode, packed, fast, n_updates=6):
    import torch

    from research.redesign_m3c_20261008 import learner_m3c as lm
    from src.model.ego2s_network import Ego2sNet

    torch.manual_seed(3)
    nscal, multi, replay, labels, demos = _world(mode, packed)
    args = m3c.default_args()
    args.actor_sync_updates = 25
    if mode == "bv8":
        args.obs_version, args.onpolicy_labels = "bv8", True
        args.label_frac, args.anchor_mode = 1.0, "v8margin"
    online = Ego2sNet(7, nscal).train()
    target = Ego2sNet(7, nscal).eval()
    target.load_state_dict(online.state_dict())
    opts = lm.LearnerOpts(fast=fast, fused_adam=False)
    st = {
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
    rng = np.random.default_rng([7, 17])
    learner = lm.Learner(
        torch,
        online=online,
        target=target,
        opt=lm.make_optimizer(torch, online, opts, "cpu"),
        replay=replay,
        multi=multi,
        demos=demos,
        labels=labels,
        rng=rng,
        st=st,
        args=args,
        nscal=nscal,
        margin=0.3,
        device="cpu",
        opts=opts,
        agent_rows=24,
        anchor_rows=8,
        min_rows=4,
    )
    feed = np.random.default_rng(99)
    staged = []
    for _ in range(n_updates):
        learner.update(0.5)
        if fast:
            staged.append({k: v.copy() for k, v in learner._stage[learner._which ^ 1].v.items()})
        _ingest(replay, feed, 40)
    learner.flush()
    return online, st, rng, multi, staged


def _assert_same_training(mode, fast):
    """Raw vs packed world: same weights, stats, RNG state, ring checks and staged batches."""
    import torch

    raw = _run(mode, packed=False, fast=fast)
    pk = _run(mode, packed=True, fast=fast)
    assert raw[1]["updates"] == 6 and raw[1]["ring_overrun_rows"] > 0
    for (k, a), (_, b) in zip(raw[0].state_dict().items(), pk[0].state_dict().items()):
        assert torch.equal(a, b), k
    for k in ("updates", "td", "anc", "qmax", "label_rows", "ring_overrun_rows"):
        assert raw[1][k] == pk[1][k], k
    assert raw[2].bit_generator.state == pk[2].bit_generator.state
    assert raw[3].calls == pk[3].calls
    for s_raw, s_pk in zip(raw[4], pk[4]):  # T1: every staged batch, byte for byte
        for k in s_raw:
            assert np.array_equal(s_raw[k], s_pk[k]), k


@pytest.mark.parametrize("mode", ["b", "bv8"])
@pytest.mark.parametrize("fast", [False, True])
def test_packed_world_trains_bitwise_identically(mode, fast):
    _assert_same_training(mode, fast)


@pytest.mark.parametrize("fast", [False, True])
def test_training_parity_fails_on_a_deliberately_broken_unpack(monkeypatch, fast):
    """The parity check above is not vacuous: unpacking the right rows into the wrong
    staging rows (reference path: ``ring.get``; T1 path: ``Learner._gather``) is caught."""
    good = op.unpack_into

    def scattered_wrong(src, src_rows, out_l, out_g, out_rows, layout):
        good(src, src_rows, out_l, out_g, np.asarray(out_rows)[::-1], layout)

    monkeypatch.setattr(op, "unpack_into", scattered_wrong)
    with pytest.raises(AssertionError):
        _assert_same_training("b", fast)


def test_multiring_get_identical():
    _, m_raw, *_ = _world("b", packed=False, seed=4)
    _, m_pk, *_ = _world("b", packed=True, seed=4)
    ids = np.random.default_rng(1).integers(0, CAP * K_ACTORS, 500)
    for a, b in zip(m_raw.get(ids), m_pk.get(ids)):
        assert a.dtype == b.dtype and np.array_equal(a, b)


def test_packed_ring_add_refuses_an_unpackable_row_before_publishing():
    ring = m3c.SharedObsRing(_Buf(m3c.ring_nbytes(10, 30, LAYOUT)), 10, 2, 30, pack=LAYOUT)
    L, G = valid_rows(np.random.default_rng(0), 3)
    L[1, 4, 0, 0] = 1
    mask = np.zeros((3, 6), bool)
    with pytest.raises(ValueError, match="row 1"):
        ring.add({"local": L, "global": G, "scalars": np.zeros((3, 30), np.float32)}, mask)
    assert ring.count == 0


# ----------------------------------------------------------------------------- demonstrations
def _shards(root: Path, seed: int = 0):
    r = np.random.default_rng(seed)
    for rnd in (10, 11):
        d = root / f"round-{rnd}"
        d.mkdir(parents=True)
        for c in range(3):
            n = int(r.integers(30, 60))
            rows = _demo_rows(r, n)
            rows["a_v8"] = rows["a_v8"].astype(np.int8)
            np.savez(
                d / f"chunk-{c:04d}-part-00.npz",
                world_seed=r.integers(0, 1000, n).astype(np.uint32),
                **rows,
            )
    return root


def _streams(tmp_path, capacity=150):
    from research.redesign_m3c_20261008.demo_store import SharedDemoStore

    data = _shards(tmp_path / "data")
    s_raw = SharedDemoStore.open_or_build(tmp_path / "raw.store", data, [10, 11], None, log=None)
    s_pk = SharedDemoStore.open_or_build(tmp_path / "pk.store", data, [10, 11], LAYOUT, log=None)
    return data, {
        "raw": m3c.DemoStream(data, [10, 11], capacity, 3),
        "packed": m3c.DemoStream(data, [10, 11], capacity, 3, pack=LAYOUT),
        "store_raw": m3c.DemoStream(data, [10, 11], capacity, 3, store=s_raw),
        "store_packed": m3c.DemoStream(data, [10, 11], capacity, 3, pack=LAYOUT, store=s_pk),
    }


def _same_dicts(a, b, ctx):
    assert list(a) == list(b), ctx
    for k in a:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), (ctx, k)


def _assert_demo_modes_identical(tmp_path):
    _, streams = _streams(tmp_path)
    for s in streams.values():
        s.fill_initial(2)
    ref = streams["raw"]
    assert ref.size == ref.cap or ref.size > 60
    for step in range(6):
        if step:  # rotate one shard per step (the private buffer: wait for its loader)
            for s in streams.values():
                s.prefetch()
                if s._thread is not None:
                    s._thread.join()
                assert s.swap_if_ready()
        rngs = {k: np.random.default_rng([step, 1]) for k in streams}
        out = {k: s.sample(rngs[k], 64) for k, s in streams.items()}
        for k in streams:
            _same_dicts(out["raw"], out[k], (step, k))
            assert rngs[k].bit_generator.state == rngs["raw"].bit_generator.state
            assert streams[k].q_teacher_absmax() == ref.q_teacher_absmax()
            assert (streams[k].size, streams[k].write) == (ref.size, ref.write)
        di = np.random.default_rng(step).integers(0, ref.size, 40)
        staged = {}
        for k, s in streams.items():
            v = {
                "local": np.zeros((50,) + op.LOCAL_SHAPE, np.uint8),
                "global": np.zeros((50,) + op.GLOBAL_SHAPE, np.uint8),
                "scalars": np.zeros((50, 30), np.float32),
                "a_v8": np.zeros(40, np.int64),
                "mask_resolved": np.zeros((40, 6), bool),
                "q_teacher": np.zeros((40, 6), np.float32),
            }
            s.gather_into(di, v, 7)
            staged[k] = v
        for k in streams:
            _same_dicts(staged["raw"], staged[k], ("gather", step, k))
    assert ref.loaded == 7 and ref.distinct > ref.cap  # the buffer wrapped


def test_demo_stream_modes_identical(tmp_path):
    _assert_demo_modes_identical(tmp_path)


@pytest.mark.parametrize("bug", ["store_index", "packed_planes"])
def test_demo_parity_fails_on_a_deliberate_break(tmp_path, monkeypatch, bug):
    """The demonstration parity check is not vacuous: an off-by-one store row index, or
    packed planes gathered from the wrong slots, is caught."""
    from research.redesign_m3c_20261008.demo_store import SharedDemoStore

    if bug == "store_index":
        good = SharedDemoStore.shard_range

        def shifted(self, path):
            start, n = good(self, path)
            return (start + 1, n) if start + n < self.rows else (start - 1, n)

        monkeypatch.setattr(SharedDemoStore, "shard_range", shifted)
    else:
        good = m3c.DemoStream._planes_into

        def wrong_slots(self, src, rows, out_l, out_g):
            rows = np.asarray(rows)
            good(self, src, rows[::-1] if "packed" in src else rows, out_l, out_g)

        monkeypatch.setattr(m3c.DemoStream, "_planes_into", wrong_slots)
    with pytest.raises(AssertionError):
        _assert_demo_modes_identical(tmp_path)


def test_demo_store_reuse_and_drift_check(tmp_path):
    from research.redesign_m3c_20261008.demo_store import SharedDemoStore, store_nbytes

    data = _shards(tmp_path / "data")
    path = tmp_path / "s.store"
    a = SharedDemoStore.open_or_build(path, data, [10, 11], LAYOUT, log=None)
    mtime = path.stat().st_mtime_ns
    b = SharedDemoStore.open_or_build(path, data, [10, 11], None, log=None)  # reused as built
    assert path.stat().st_mtime_ns == mtime and b.pack == LAYOUT
    assert a.rows == b.rows and a.nbytes >= store_nbytes(a.rows, LAYOUT)
    assert not b.arrays["packed"].flags.writeable
    shard = sorted((data / "round-11").glob("*.npz"))[0]
    with shard.open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(SystemExit, match="missing or changed"):
        SharedDemoStore.open_or_build(path, data, [10, 11], LAYOUT, log=None)
    with pytest.raises(SystemExit, match="missing or changed"):
        SharedDemoStore.open_or_build(path, data, [10, 11, 12], LAYOUT, log=None)


def test_flag_validation(tmp_path):
    base = ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="multi-actor"):
        m3c.main(base + ["--obs-pack"])
    with pytest.raises(SystemExit, match="multi-actor"):
        m3c.main(base + ["--demo-mmap", str(tmp_path / "s")])
    bv8 = ["--obs-version", "bv8", "--onpolicy-labels", "--label-frac", "1.0"]
    bv8 += ["--anchor-mode", "v8margin", "--actors", "1", "--demo-mmap", str(tmp_path / "s")]
    with pytest.raises(SystemExit, match="no demonstration buffer"):
        m3c.main(base + bv8)
    d = m3c.default_args()
    assert d.obs_pack is False and d.demo_mmap is None and m3c.m3c_flags(d) == {}


# ----------------------------------------------------------------------------- actor
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


def _run_actor(tmp_path, init, pack, frames):
    argv = ["--seed", "5", "--init", str(init), "--demo-data", ".", "--out", str(tmp_path / "u")]
    argv += ["--actors", "1", "--worlds", "6", "--frozen-frac", "0.34", "--mixed-frac", "0.33"]
    argv += ["--segment-frames", "45", "--device", "cpu", "--v61-backend", "numba"]
    args = m3c.build_parser().parse_args(argv)
    margin = args.worlds * 16
    ring_cap = args.worlds * (frames + 4) + margin
    net = m3c.load_init(args, "cpu")
    spec = m3c._weights_spec(net)
    n_weights = int(sum(n for _, _, n in spec))
    blocks = [
        m3c.Block.create(m3c.ring_nbytes(ring_cap, 30, pack), str(tmp_path), "ring0"),
        m3c.Block.create(8 + 8 * (n_weights + 1), str(tmp_path), "weights"),
        m3c.Block.create(32, str(tmp_path), "progress"),
    ]
    names = {"ring0": blocks[0].desc, "weights": blocks[1].desc, "progress": blocks[2].desc}
    wver = np.ndarray((1,), dtype=np.int64, buffer=blocks[1].buf, offset=0)
    wver[0] = 0
    wbuf = np.ndarray((n_weights + 1,), dtype=np.float64, buffer=blocks[1].buf, offset=8)
    prog = np.ndarray((4,), dtype=np.int64, buffer=blocks[2].buf, offset=0)
    prog[:] = 0
    m3c._publish_weights(net, wbuf, wver)
    cfg = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "K": 1,
        "ring_cap": ring_cap,
        "ring_margin": margin,
        "n_weights": n_weights,
        "actor_device": "cpu",
    }
    if pack is not None:
        cfg["obs_pack"] = pack.to_dict()
    stop = _Stop()
    sink = _Collect(frames, stop)
    m3c.actor_main(0, cfg, names, sink, stop)
    ring = m3c.SharedObsRing(blocks[0], ring_cap, margin, 30, pack=pack)
    n = ring.count
    obs = ring.get(np.arange(n))
    out = [x.copy() for x in obs], n
    del wver, wbuf, prog, ring, obs
    for b in blocks:
        b.unlink()
        b.close()
    return sink.msgs, out


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_actor_packed_ring_holds_the_same_observations(tmp_path):
    import torch

    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    torch.manual_seed(0)
    init = tmp_path / "init.pth"
    save_ego2s_checkpoint(init, Ego2sNet(7, 30))
    assert op.train_layout() == LAYOUT
    runs = {}
    for name, pack in (("raw", None), ("packed", LAYOUT)):
        d = tmp_path / name
        d.mkdir()
        runs[name] = _run_actor(d, init, pack, frames=60)
    (m_raw, (o_raw, n_raw)), (m_pk, (o_pk, n_pk)) = runs["raw"], runs["packed"]
    assert n_raw == n_pk > 300
    for a, b in zip(o_raw, o_pk):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    for a, b in zip(m_raw, m_pk):
        for k in ("s", "a", "R", "disc", "nxt", "acts"):
            assert np.array_equal(a[k], b[k]), k
