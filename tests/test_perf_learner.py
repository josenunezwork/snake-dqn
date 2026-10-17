"""perf-learner (2026-10-13): the T1 fast learner is the reference update (bitwise on CPU with
plain Adam; ~ulp with fused Adam), including ring-overrun compaction, demonstration / label
anchor rows and the non-finite-loss halt; T2 / T3 options are wired and validated."""

from __future__ import annotations

import queue

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.redesign_m3c_20261008 import learner_m3c as lm  # noqa: E402
from research.redesign_m3c_20261008 import train_m3c as m3c  # noqa: E402
from src.model.ego2s_network import Ego2sNet  # noqa: E402

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


def _world(mode: str, seed: int = 0):
    """Rings + replay (+ labels for bv8, + demonstrations for b), deterministic."""
    nscal = 60 if mode == "bv8" else 30
    r = np.random.default_rng(seed)
    rings = []
    for _ in range(K_ACTORS):
        ring = m3c.SharedObsRing(_Buf(m3c.ring_nbytes(CAP, nscal)), CAP, 16, nscal)
        ring.arrays["local"][:] = r.integers(0, 256, ring.arrays["local"].shape, dtype=np.uint8)
        ring.arrays["global"][:] = r.integers(0, 256, ring.arrays["global"].shape, dtype=np.uint8)
        ring.arrays["scalars"][:] = r.standard_normal(ring.arrays["scalars"].shape)
        ring.arrays["mask"][:] = r.random(ring.arrays["mask"].shape) < 0.7
        ring._count[0] = CAP
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
        demos = object.__new__(m3c.DemoStream)
        size = 600
        demos.buf = {
            "local": r.integers(0, 256, (size,) + m3c.LOCAL, dtype=np.uint8),
            "global": r.integers(0, 256, (size,) + m3c.GLOBAL, dtype=np.uint8),
            "scalars": r.standard_normal((size, 30)).astype(np.float32),
            "q_teacher": r.standard_normal((size, 6)).astype(np.float32),
            "a_v8": r.integers(0, 6, size),
            "mask_resolved": r.random((size, 6)) < 0.7,
        }
        demos.size = size
        demos._q = queue.Queue()
        demos._thread = None
        demos.prefetch = lambda: None  # no shard files here (swap_if_ready: empty queue)
    return nscal, multi, replay, labels, demos


def _ingest(replay, r, n):
    s = r.integers(0, CAP * K_ACTORS, n)
    nxt = np.where(r.random(n) < 0.1, -1, r.integers(0, CAP * K_ACTORS, n))
    replay.add_many(s, r.integers(0, 6, n), r.standard_normal(n), r.random(n) * 0.9, nxt)


def _args(mode: str, **kw):
    a = m3c.default_args()
    a.actor_sync_updates = 25
    if mode == "bv8":
        a.obs_version, a.onpolicy_labels, a.label_frac, a.anchor_mode = "bv8", True, 1.0, "v8margin"
    for k, v in kw.items():
        setattr(a, k, v)
    return a


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


class _ReturnBank:
    """Fixed large-target rows: exposes separate return-delta behavior in a tiny learner test."""

    gamma = 0.99

    def __init__(self, *, nan_return: bool = False) -> None:
        self.nan_return = nan_return
        self.draws: list[np.ndarray] = []

    def sample(self, rng, n, *, kappa):
        del kappa
        draw = rng.integers(0, 2**31, size=n, dtype=np.int64)
        self.draws.append(draw.copy())
        realised = np.full(n, np.nan if self.nan_return else 100.0, dtype=np.float32)
        if not self.nan_return:
            realised += (draw % 17).astype(np.float32)
        return {
            "local": np.zeros((n,) + m3c.LOCAL, dtype=np.uint8),
            "global": np.zeros((n,) + m3c.GLOBAL, dtype=np.uint8),
            "scalars": np.zeros((n, 60), dtype=np.float32),
            "action": np.zeros(n, dtype=np.int64),
            "T": np.ones(n, dtype=np.int32),
            "died": np.ones(n, dtype=bool),
            "realised": realised,
            "terminal_local": np.zeros((n,) + m3c.LOCAL, dtype=np.uint8),
            "terminal_global": np.zeros((n,) + m3c.GLOBAL, dtype=np.uint8),
            "terminal_scalars": np.zeros((n, 60), dtype=np.float32),
            "terminal_mask": np.zeros((n, 6), dtype=bool),
            "weight": np.ones(n, dtype=np.float32),
        }


def _run(mode, opts, n_updates=8, poison_at=None, net_seed=3, return_bank=None, **argkw):
    torch.manual_seed(net_seed)
    nscal, multi, replay, labels, demos = _world(mode)
    online = Ego2sNet(7, nscal).train()
    target = Ego2sNet(7, nscal).eval()
    target.load_state_dict(online.state_dict())
    opt = lm.make_optimizer(torch, online, opts, "cpu")
    st = _st()
    rng = np.random.default_rng([7, 17])
    learner = lm.Learner(
        torch,
        online=online,
        target=target,
        opt=opt,
        replay=replay,
        multi=multi,
        demos=demos,
        labels=labels,
        rng=rng,
        st=st,
        args=_args(mode, **argkw),
        nscal=nscal,
        margin=0.3,
        device="cpu",
        opts=opts,
        agent_rows=24,
        anchor_rows=8,
        min_rows=4,
        return_bank=return_bank,
    )
    learner.target_every = 3  # exercise target syncs inside the short run
    feed = np.random.default_rng(99)
    for i in range(n_updates):
        if poison_at is not None and i == poison_at:
            replay.R[:] = np.nan
        learner.update(0.5)
        if st["halt"]:
            break
        _ingest(replay, feed, 40)  # transitions arrive between updates, as in main_multi
    learner.flush()
    return online, target, st, rng, multi


def _assert_same(r1, r2, exact=True):
    (o1, t1, s1, g1, m1), (o2, t2, s2, g2, m2) = r1, r2
    for net1, net2 in ((o1, o2), (t1, t2)):
        for (k, a), (_, b) in zip(net1.state_dict().items(), net2.state_dict().items()):
            if exact:
                assert torch.equal(a, b), k
            else:
                assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), k
    for k in ("updates", "halt", "anchor_updates", "label_rows", "ring_overrun_rows"):
        assert s1[k] == s2[k], k
    if exact:
        assert s1["td"] == s2["td"] and s1["anc"] == s2["anc"] and s1["qmax"] == s2["qmax"]
    else:
        np.testing.assert_allclose(s1["td"], s2["td"], rtol=1e-4)
    assert g1.bit_generator.state == g2.bit_generator.state
    assert m1.calls == m2.calls


@pytest.mark.parametrize("mode", ["bv8", "b"])
def test_fast_equals_reference_bitwise_cpu(mode):
    ref = _run(mode, lm.LearnerOpts())
    fast = _run(mode, lm.LearnerOpts(fast=True, fused_adam=False))
    assert ref[2]["updates"] == 8 and ref[2]["ring_overrun_rows"] > 0
    assert ref[2]["anchor_updates"] == 8
    _assert_same(ref, fast, exact=True)


@pytest.mark.parametrize("return_delta", [None, 100.0])
def test_fast_return_regression_matches_reference_and_preserves_td(return_delta):
    """T1 adds the same active return loss as reference; delta 100 remains return-only."""
    kwargs = dict(
        return_frac=0.5,
        return_weight=1.0,
        return_huber_delta=return_delta,
        td_huber_delta=1.0,
        mass_kappa=1.0,
    )
    ref_bank, fast_bank = _ReturnBank(), _ReturnBank()
    ref = _run("bv8", lm.LearnerOpts(fused_adam=False), return_bank=ref_bank, **kwargs)
    fast = _run("bv8", lm.LearnerOpts(fast=True, fused_adam=False), return_bank=fast_bank, **kwargs)
    _assert_same(ref, fast, exact=True)
    assert len(ref_bank.draws) == len(fast_bank.draws) == 8
    for reference, fast_draw in zip(ref_bank.draws, fast_bank.draws):
        assert np.array_equal(reference, fast_draw)
    for result in (ref, fast):
        state = result[2]
        assert state["return_rows"] == [4] * 8
        assert len(state["return_loss"]) == 8
        assert state["return_totals"]["rows"] == 32


def test_return_delta_default_fallback_leaves_td_delta_one() -> None:
    """Changing only the return delta changes return loss, while the first TD loss is fixed."""
    common = dict(return_frac=0.5, return_weight=1.0, td_huber_delta=1.0, mass_kappa=1.0)
    fallback = _run(
        "bv8", lm.LearnerOpts(fused_adam=False), n_updates=1, return_bank=_ReturnBank(), **common
    )
    wide = _run(
        "bv8",
        lm.LearnerOpts(fused_adam=False),
        n_updates=1,
        return_bank=_ReturnBank(),
        return_huber_delta=100.0,
        **common,
    )
    assert fallback[2]["td"] == wide[2]["td"]
    assert fallback[2]["return_loss"][0] != wide[2]["return_loss"][0]


def test_fast_nonfinite_return_halts_without_step_or_successful_telemetry() -> None:
    """A NaN return target is halted before Adam and contributes no return-loss evidence."""
    torch.manual_seed(3)
    initial = Ego2sNet(7, 60).state_dict()
    online, _, state, _, _ = _run(
        "bv8",
        lm.LearnerOpts(fast=True, fused_adam=False),
        n_updates=2,
        return_bank=_ReturnBank(nan_return=True),
        return_frac=0.5,
        return_weight=1.0,
        mass_kappa=1.0,
    )
    assert state["halt"] == "non-finite loss"
    assert state["updates"] == 0
    assert state["return_rows"] == []
    assert state["return_loss"] == []
    assert state["return_totals"] == {"updates": 0, "rows": 0, "loss_sum": 0.0}
    for name, parameter in online.state_dict().items():
        assert torch.equal(parameter, initial[name]), name


def test_fast_fused_adam_close_to_reference():
    ref = _run("bv8", lm.LearnerOpts())
    fused = _run("bv8", lm.LearnerOpts(fast=True))
    assert isinstance(fused[0], Ego2sNet)
    _assert_same(ref, fused, exact=False)


def test_m3_anchor_weighted_path_bitwise():
    ref = _run("b", lm.LearnerOpts(), anchor_boost_weight=3.0)
    fast = _run("b", lm.LearnerOpts(fast=True, fused_adam=False), anchor_boost_weight=3.0)
    _assert_same(ref, fast, exact=True)


@pytest.mark.parametrize("fast", [False, True])
def test_non_finite_halt_never_applies_the_step(fast):
    ref = _run("bv8", lm.LearnerOpts(), poison_at=5)
    other = _run("bv8", lm.LearnerOpts(fast=fast, fused_adam=False), poison_at=5)
    assert ref[2]["halt"] == "non-finite loss" and ref[2]["updates"] == 5
    if not fast:
        _assert_same(ref, other, exact=True)
        return
    # The pipelined learner notices the flag one call later, after sampling that call's
    # batch (the run is over by then): weights and counters match, the RNG has moved on.
    (o1, t1, s1, _, _), (o2, t2, s2, _, _) = ref, other
    for net1, net2 in ((o1, o2), (t1, t2)):
        for a, b in zip(net1.state_dict().values(), net2.state_dict().values()):
            assert torch.equal(a, b)
    for k in ("updates", "halt", "anchor_updates", "td", "anc", "qmax"):
        assert s1[k] == s2[k], k
    assert s2["label_rows"][: len(s1["label_rows"])] == s1["label_rows"]


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_precision_runs_and_stays_close(precision):
    ref = _run("bv8", lm.LearnerOpts(fast=True, fused_adam=False), n_updates=4)
    amp = _run("bv8", lm.LearnerOpts(fast=True, fused_adam=False, precision=precision), 4)
    assert amp[2]["halt"] is None and amp[2]["updates"] == 4
    assert all(np.isfinite(amp[2]["td"]))
    for a, b in zip(ref[0].parameters(), amp[0].parameters()):
        assert torch.allclose(a, b, atol=5e-3)  # 4 Adam steps of lr 1e-4: bounded drift


def test_batch_scale_semantics():
    o = lm.LearnerOpts(batch_scale=4, lr_scale="sqrt")
    assert o.lr == pytest.approx(2e-4)
    assert lm.LearnerOpts(batch_scale=2, lr_scale="linear").lr == pytest.approx(2e-4)
    assert lm.LearnerOpts().lr == 1e-4
    online, _, st, _, _ = _run("bv8", lm.LearnerOpts(fast=True, batch_scale=2), n_updates=3)
    assert st["updates"] == 3
    assert st["label_rows"] == [16, 16, 16] or max(st["label_rows"]) <= 16
    with pytest.raises(ValueError):
        lm.LearnerOpts(precision="fp16")  # needs fast
    with pytest.raises(ValueError):
        lm.LearnerOpts(lr_scale="linear")  # needs batch_scale > 1
    with pytest.raises(ValueError):
        lm.LearnerOpts(batch_scale=0)


def test_batch_scale_intervals():
    nscal, multi, replay, labels, demos = _world("bv8")
    net = Ego2sNet(7, nscal)
    st = _st()
    kw = dict(
        online=net,
        target=net,
        opt=None,
        replay=replay,
        multi=multi,
        demos=None,
        labels=labels,
        rng=np.random.default_rng(0),
        st=st,
        nscal=nscal,
        margin=0.3,
        device="cpu",
    )
    one = lm.Learner(torch, args=_args("bv8"), opts=lm.LearnerOpts(), **kw)
    league = lm.Learner(torch, args=_args("bv8"), opts=lm.LearnerOpts(batch_scale=2), **kw)
    four = lm.Learner(torch, args=_args("bv8"), opts=lm.LearnerOpts(batch_scale=4), **kw)
    assert (one.update_every, one.target_every, one.publish_every) == (128, 1000, 25)
    assert league.update_every == 256  # T19 R0 spends replay rows, independent of N seats.
    assert (four.update_every, four.target_every, four.publish_every) == (512, 250, 6)
    assert (four.a_rows, four.d_rows, four.min_rows) == (1536, 512, 256)
    st["updates"] = 0  # reference keeps the M3-C publish condition (also with no new update)
    assert one.publish_due(0)
    fast = lm.Learner(torch, args=_args("bv8"), opts=lm.LearnerOpts(fast=True), **kw)
    assert not fast.publish_due(0)  # fast: only when an update was applied


def test_cli_validation_and_flags(tmp_path):
    base = ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="multi-actor"):
        m3c.main(base + ["--learner-fast"])
    with pytest.raises(SystemExit, match="needs --learner-fast"):
        m3c.main(base + ["--actors", "2", "--learner-precision", "bf16"])
    args = m3c.build_parser().parse_args(
        base + ["--actors", "2", "--learner-fast", "--batch-scale", "2", "--lr-scale", "sqrt"]
    )
    assert m3c.m3c_flags(args) == {
        "actors": 2,
        "learner_fast": True,
        "batch_scale": 2,
        "lr_scale": "sqrt",
    }
    assert m3c.m3c_flags(m3c.default_args()) == {}


def test_fp16_uses_foreach_adam_and_fast_fp32_fused():
    net = Ego2sNet(7, 60)
    fp16 = lm.make_optimizer(torch, net, lm.LearnerOpts(fast=True, precision="fp16"), "cpu")
    assert not fp16.defaults.get("fused")  # fused + GradScaler applied inf steps on MPS
    fp32 = lm.make_optimizer(torch, net, lm.LearnerOpts(fast=True), "cpu")
    assert fp32.defaults.get("fused")
    assert not lm.make_optimizer(torch, net, lm.LearnerOpts(), "cpu").defaults.get("fused")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_fast_and_fp16_stay_finite_and_close():
    """On MPS: T1 tracks the reference closely; fp16 (GradScaler) never writes inf/NaN."""

    def run(opts, n=6):
        torch.manual_seed(3)
        nscal, multi, replay, labels, demos = _world("bv8")
        online = Ego2sNet(7, nscal).to("mps").train()
        target = Ego2sNet(7, nscal).to("mps").eval()
        target.load_state_dict(online.state_dict())
        st = _st()
        learner = lm.Learner(
            torch,
            online=online,
            target=target,
            opt=lm.make_optimizer(torch, online, opts, "mps"),
            replay=replay,
            multi=multi,
            demos=demos,
            labels=labels,
            rng=np.random.default_rng([7, 17]),
            st=st,
            args=_args("bv8"),
            nscal=nscal,
            margin=0.3,
            device="mps",
            opts=opts,
            agent_rows=48,
            anchor_rows=16,
            min_rows=4,
        )
        if learner.scaler is not None:
            learner.scaler = torch.amp.GradScaler("mps", init_scale=2.0**24)  # force overflows
        feed = np.random.default_rng(99)
        for _ in range(n):
            learner.update(0.5)
            _ingest(replay, feed, 40)
        learner.flush()
        return online, st

    ref, s_ref = run(lm.LearnerOpts())
    fast, s_fast = run(lm.LearnerOpts(fast=True))
    half, s_half = run(lm.LearnerOpts(fast=True, precision="fp16"))
    assert s_ref["updates"] == s_fast["updates"] == s_half["updates"] == 6
    assert s_half["halt"] is None
    for a, b, c in zip(ref.parameters(), fast.parameters(), half.parameters()):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)
        assert torch.isfinite(c).all()


@pytest.mark.slow
def test_trainer_runs_with_fast_learner_and_batch_scale(tmp_path):
    """End to end (2 actors, CPU): the flush points (log, checkpoint, final) keep the run
    whole; the perf-learner flags are recorded in the summary."""
    import json

    from tests.test_m3c_trainer import HAVE_DATA, _run

    if not HAVE_DATA:
        pytest.skip("needs the M2b student and demonstrations")
    out = tmp_path / "fast"
    extra = ["--actors", "2", "--no-guard", "--ring-dir", str(tmp_path), "--learner-fast"]
    r = _run("research/redesign_m3c_20261008/train_m3c.py", out, extra + ["--batch-scale", "2"])
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())
    assert s["halt"] is None and s["actor_exitcodes"] == [0, 0]
    assert s["transitions"] >= 12000 and s["updates"] >= 30
    assert s["m3c"] == {"actors": 2, "learner_fast": True, "batch_scale": 2}
    assert (out / "final.pth").exists() and (out / "ckpt_6k.pth").exists()
