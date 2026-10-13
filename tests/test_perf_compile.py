"""perf-compile (2026-10-13): opt-in ``torch.compile`` for the ego2s trainer.

* Off (the default) is the untouched path: ``torch.compile`` is never called and the T1
  update stays bitwise equal to the reference update.
* On (CPU inductor here; the target is CUDA): compiled = eager up to float32 reassociation,
  NOT bitwise. Measured and bounded below: loss terms / gradients on fixed batches, the
  parameters and Q after N compiled updates, the hero forward (max |dQ|, argmax flips).
* Shapes are static (canonical batches only / hero rows padded to ``--worlds``): one graph
  per shape key, no recompiles when the hero row count changes, non-canonical batches eager.
* Fallbacks: a compile error or a compiled step that disagrees with eager (the first-steps
  check) prints a message and continues eagerly, bitwise equal to an eager run.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.redesign_m3c_20261008 import compile_m3c as cm  # noqa: E402
from research.redesign_m3c_20261008 import learner_m3c as lm  # noqa: E402
from research.redesign_m3c_20261008 import train_m3c as m3c  # noqa: E402
from src.model.ego2s_network import Ego2sNet  # noqa: E402
from tests.test_perf_learner import _args, _assert_same, _ingest  # noqa: E402
from tests.test_perf_learner import _run as _run_reference  # noqa: E402
from tests.test_perf_learner import _st, _world  # noqa: E402

# Tolerances (float32 reassociation; the measured values are far below, see the asserts)
TERM_RTOL = 1e-4  # td / anchor / qmax, one batch
GRAD_REL = 1e-4  # per-tensor relative L2 gradient error, one batch
Q_ATOL = 1e-4  # hero forward, |q| ~ 1


@pytest.fixture(autouse=True)
def _fresh_dynamo(monkeypatch):
    """Isolated dynamo caches / counters; in-process inductor compile (no worker pool)."""
    import torch._inductor.config as icfg
    from torch._dynamo.utils import counters

    monkeypatch.setattr(icfg, "compile_threads", 1)
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()
    counters.clear()


def _learner(mode, opts, *, device="cpu", flaky=False, net_seed=3):
    torch.manual_seed(net_seed)
    nscal, multi, replay, labels, demos = _world(mode)
    if not flaky:  # every gathered row stays valid: canonical batches
        multi = m3c.MultiRing(multi.rings)
    online = Ego2sNet(7, nscal).to(device).train()
    target = Ego2sNet(7, nscal).to(device).eval()
    target.load_state_dict(online.state_dict())
    learner = lm.Learner(
        torch,
        online=online,
        target=target,
        opt=lm.make_optimizer(torch, online, opts, device),
        replay=replay,
        multi=multi,
        demos=demos,
        labels=labels,
        rng=np.random.default_rng([7, 17]),
        st=_st(),
        args=_args(mode),
        nscal=nscal,
        margin=0.3,
        device=device,
        opts=opts,
        agent_rows=24,
        anchor_rows=8,
        min_rows=4,
    )
    learner.target_every = 3
    return learner


def _train(learner, n_updates):
    theta0 = torch.cat([p.detach().reshape(-1).cpu().clone() for p in learner.online.parameters()])
    feed = np.random.default_rng(99)
    for _ in range(n_updates):
        learner.update(0.5)
        _ingest(learner.replay, feed, 40)
    learner.flush()
    return theta0


def _theta(net):
    return torch.cat([p.detach().reshape(-1).cpu() for p in net.parameters()])


@torch.no_grad()
def _q_eval(net, learner, n=64):
    ids = np.arange(n, dtype=np.int64) * 7 % (learner.multi.rings[0].cap * 2)
    L, G, S, _ = learner.multi.get(ids)
    dev = next(net.parameters()).device
    return net(*[torch.from_numpy(x).to(dev) for x in (L, G, S)]).float().cpu()


def _batch(learner):
    """One sampled, uploaded staging batch and the loss_terms keyword arguments."""
    b = learner._sample_fast()
    dv = b["stage"].upload(torch, learner.device, False)
    names = ("local", "global", "scalars", "a", "R", "disc", "nmask", "q_teacher", "a_v8")
    tensors = [dv[k] for k in names + ("mask_resolved",)]
    a = learner.args
    kw = dict(
        n=b["n"],
        m=b["m"],
        n_demo=b["n_demo"],
        amp_dtype=None,
        dev_type=learner.device.type,
        anchor_mode=a.anchor_mode,
        boost_weight=float(a.anchor_boost_weight),
        margin=0.3,
    )
    return tensors, kw


def _terms_and_grads(fn, learner, tensors, kw):
    learner.online.zero_grad(set_to_none=True)
    td, anc, qmax = fn(torch, learner.online, learner.target, *tensors, **kw)
    (td + 0.5 * anc).backward()
    grads = {k: p.grad.detach().clone() for k, p in learner.online.named_parameters()}
    return [float(td.detach()), float(anc.detach()), float(qmax)], grads


# ----------------------------------------------------------------------------- off = untouched
def test_compile_off_never_calls_torch_compile(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("torch.compile called with compile off")

    monkeypatch.setattr(torch, "compile", boom)
    ref = _run_reference("bv8", lm.LearnerOpts())
    fast = _run_reference("bv8", lm.LearnerOpts(fast=True, fused_adam=False))
    _assert_same(ref, fast, exact=True)  # T1 (loss_terms) is still bitwise the reference
    learner = _learner("bv8", lm.LearnerOpts(fast=True))
    assert learner._cstat is None and learner.compile_status() is None
    args = m3c.default_args()
    assert (args.compile_learner, args.compile_learner_scope, args.compile_actor) == (
        "off",
        "loss",
        "off",
    )
    assert lm.LearnerOpts.from_args(args).compile == "off"
    assert m3c.m3c_flags(args) == {} and lm.learner_flags(args) == {}


def test_cli_validation_and_flags(tmp_path):
    base = ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="needs --learner-fast"):
        m3c.main(base + ["--actors", "2", "--compile-learner", "default"])
    with pytest.raises(SystemExit, match="multi-actor"):
        m3c.main(base + ["--learner-fast", "--compile-learner", "default"])
    with pytest.raises(SystemExit, match="multi-actor"):
        m3c.main(base + ["--compile-actor", "default"])
    with pytest.raises(SystemExit, match="needs --compile-learner"):
        m3c.main(base + ["--actors", "2", "--compile-learner-scope", "forward"])
    with pytest.raises(SystemExit):
        m3c.build_parser().parse_args(base + ["--compile-actor", "fast"])  # not a mode
    argv = base + ["--actors", "2", "--learner-fast", "--compile-learner", "reduce-overhead"]
    args = m3c.build_parser().parse_args(argv + ["--compile-actor", "default"])
    assert m3c.m3c_flags(args) == {
        "actors": 2,
        "learner_fast": True,
        "compile_learner": "reduce-overhead",
        "compile_actor": "default",
    }
    assert lm.LearnerOpts.from_args(args).compile == "reduce-overhead"
    assert lm.learner_flags(args)["compile_learner_scope"] == "loss"
    with pytest.raises(ValueError):
        lm.LearnerOpts(compile="default")  # needs fast
    with pytest.raises(ValueError):
        lm.LearnerOpts(fast=True, compile="max-autotune")  # cudagraph mode not offered


# ----------------------------------------------------------------------------- learner
@pytest.mark.parametrize("mode,boost", [("bv8", 1.0), ("b", 1.0), ("b", 3.0)])
def test_compiled_loss_terms_close_on_fixed_batch(mode, boost):
    """td / anchor / qmax and every parameter gradient: compiled vs eager, same batch."""
    learner = _learner(mode, lm.LearnerOpts(fast=True, fused_adam=False))
    learner.args.anchor_boost_weight = boost
    tensors, kw = _batch(learner)
    assert kw["n"] == 24 and kw["m"] == 8  # a canonical batch
    t_e, g_e = _terms_and_grads(lm.loss_terms, learner, tensors, kw)
    compiled = cm.compile_fn(torch, lm.loss_terms, "default")
    t_c, g_c = _terms_and_grads(compiled, learner, tensors, kw)
    np.testing.assert_allclose(t_c, t_e, rtol=TERM_RTOL)
    rel = {k: float((g_c[k] - g_e[k]).norm() / (1e-12 + g_e[k].norm())) for k in g_e}
    worst = max(rel.values())
    assert worst < GRAD_REL, (worst, max(rel, key=rel.get))
    assert cm.dynamo_counters()["graph_breaks"] == 0


@pytest.mark.parametrize("scope", ["loss", "forward"])
def test_compiled_updates_stay_close_to_eager(monkeypatch, scope):
    """N updates, every one compiled (the first-steps check disabled): parameters and Q stay
    within float-reassociation distance of the eager T1 run; one graph per grad mode."""
    monkeypatch.setattr(lm, "VERIFY_CALLS", 0)
    n = 10
    eager = _learner("bv8", lm.LearnerOpts(fast=True, fused_adam=False))
    theta0 = _train(eager, n)
    comp = _learner(
        "bv8", lm.LearnerOpts(fast=True, fused_adam=False, compile="default", compile_scope=scope)
    )
    _train(comp, n)
    st = comp.compile_status()
    assert st["fallback"] is None and st["compiled_calls"] == n and st["eager_calls"] == 0
    assert st["graphs"] <= 2 and st["graph_breaks"] == 0 and not st["recompile_storm"]
    for k in ("updates", "anchor_updates", "label_rows", "halt"):
        assert comp.st[k] == eager.st[k], k
    assert comp.rng.bit_generator.state == eager.rng.bit_generator.state
    te, tc = _theta(eager.online), _theta(comp.online)
    step = float((te - theta0).norm())
    rel = float((tc - te).norm()) / step
    # Adam turns reassociation noise on near-zero gradients into lr-sized steps: compare
    # against the size of the whole update (measured ~1e-3 on CPU)
    assert rel < 2e-2, rel
    assert float((tc - te).abs().max()) < 4 * 1e-4 * n  # far below the lr x steps scale
    qe, qc = _q_eval(eager.online, eager), _q_eval(comp.online, eager)
    assert float((qe - qc).abs().max()) < 1e-3 * (1 + float(qe.abs().max()))
    assert bool((qe.argmax(1) == qc.argmax(1)).all())
    np.testing.assert_allclose(comp.st["td"], eager.st["td"], rtol=1e-3, atol=1e-5)


def test_first_steps_verified_then_compiled():
    learner = _learner("b", lm.LearnerOpts(fast=True, fused_adam=False, compile="default"))
    _train(learner, 6)
    st = learner.compile_status()
    assert st["verify_calls"] == lm.VERIFY_CALLS and st["compiled_calls"] == 6 - lm.VERIFY_CALLS
    assert st["fallback"] is None and st["verify_worst_rel_err"] < 1e-4


def test_noncanonical_batches_run_eager():
    """Stale-row drops (FlakyMulti) make batches non-canonical: those run eagerly."""
    learner = _learner(
        "bv8", lm.LearnerOpts(fast=True, fused_adam=False, compile="default"), flaky=True
    )
    _train(learner, 8)
    st = learner.compile_status()
    assert learner.st["ring_overrun_rows"] > 0 and st["eager_calls"] > 0
    assert st["eager_calls"] + st["compiled_calls"] + st["verify_calls"] == 8
    assert st["graphs"] <= 2 and st["fallback"] is None


def _bad_compile(kind):
    def compile_fn(torch_, fn, mode):
        if kind == "raise":

            def broken(*a, **k):
                raise RuntimeError("inductor backend unavailable (test)")

            return broken

        def skewed(torch_, online, target, *a, **k):  # a wrong "compiled" kernel
            td, anc, qmax = fn(torch_, online, target, *a, **k)
            return td * 1.01, anc, qmax

        return skewed

    return compile_fn


@pytest.mark.parametrize("kind", ["raise", "mismatch"])
def test_learner_falls_back_to_eager_bitwise(monkeypatch, capsys, kind):
    monkeypatch.setattr(cm, "compile_fn", _bad_compile(kind))
    ref = _learner("bv8", lm.LearnerOpts(fast=True, fused_adam=False))
    _train(ref, 6)
    bad = _learner("bv8", lm.LearnerOpts(fast=True, fused_adam=False, compile="default"))
    _train(bad, 6)
    st = bad.compile_status()
    assert st["fallback"] and st["compiled_calls"] == 0
    assert "continuing EAGER" in capsys.readouterr().err
    assert torch.equal(_theta(ref.online), _theta(bad.online))  # the eager run, bitwise
    assert ref.st["td"] == bad.st["td"]


# ----------------------------------------------------------------------------- actor
def _obs(r, n, nscal=30):
    def planes(shape):
        return np.where(r.random(shape) < 0.15, r.integers(1, 256, shape, dtype=np.uint8), 0)

    return {
        "local": planes((n,) + m3c.LOCAL).astype(np.uint8),
        "global": planes((n,) + m3c.GLOBAL).astype(np.uint8),
        "scalars": r.standard_normal((n, nscal)).astype(np.float32),
    }


@pytest.mark.parametrize("nscal", [30, 60])
def test_hero_forward_matches_eager_without_recompiles(nscal):
    torch.manual_seed(1)
    net = Ego2sNet(7, nscal).eval()
    hf = cm.HeroForward(torch, net, "cpu", "default", 8, nscal, m3c.LOCAL, m3c.GLOBAL)
    assert hf.fn is not None and hf.fallback is None
    assert hf.probe["max_abs_diff"] < Q_ATOL and hf.probe["argmax_flip_rate"] == 0.0
    r = np.random.default_rng(2)
    worst = 0.0
    for n in (8, 5, 1, 3, 8, 2):  # the live hero count changes every frame
        obs = _obs(r, n, nscal)
        with torch.no_grad():
            q = hf(obs)
            ref = net(*[torch.from_numpy(obs[k]) for k in ("local", "global", "scalars")])
        assert q.shape == (n, 6)
        worst = max(worst, float((q - ref).abs().max()))
        assert bool((q.argmax(1) == ref.argmax(1)).all())
    assert worst < Q_ATOL, worst
    st = hf.status()
    assert st["calls"] == 6 and st["graphs"] == 1 and st["frames_total"] == 1  # one compile


def test_hero_forward_picks_up_in_place_weight_updates():
    torch.manual_seed(1)
    net = Ego2sNet(7, 30).eval()
    hf = cm.HeroForward(torch, net, "cpu", "default", 4, 30, m3c.LOCAL, m3c.GLOBAL)
    obs = _obs(np.random.default_rng(0), 4)
    torch.manual_seed(9)
    new = Ego2sNet(7, 30)
    net.load_state_dict(new.state_dict())  # what the actor does with --compile-actor
    with torch.no_grad():
        q = hf(obs)
        ref = new(*[torch.from_numpy(obs[k]) for k in ("local", "global", "scalars")])
    assert float((q - ref).abs().max()) < Q_ATOL
    assert hf.status()["graphs"] == 1


@pytest.mark.parametrize("kind", ["raise", "mismatch"])
def test_hero_forward_falls_back(monkeypatch, capsys, kind):
    def compile_fn(torch_, net, mode):
        if kind == "raise":

            def broken(*a):
                raise RuntimeError("inductor backend unavailable (test)")

            return broken
        return lambda *a: net(*a) + 0.5  # a wrong kernel: caught by the probe check

    monkeypatch.setattr(cm, "compile_fn", compile_fn)
    torch.manual_seed(1)
    net = Ego2sNet(7, 30).eval()
    hf = cm.HeroForward(torch, net, "cpu", "default", 4, 30, m3c.LOCAL, m3c.GLOBAL)
    assert hf.fn is None and hf.fallback
    assert "continuing EAGER" in capsys.readouterr().err
    obs = _obs(np.random.default_rng(0), 3)
    with torch.no_grad():
        q = hf(obs)
        ref = net(*[torch.from_numpy(obs[k]) for k in ("local", "global", "scalars")])
    assert torch.equal(q, ref)


# ----------------------------------------------------------------------------- MPS
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_compile_is_checked():
    """torch 2.9.1's MPS inductor: the hero forward matches eager; the learner's compiled
    backward gives wrong conv-bias gradients, so the first-steps check must catch it (or,
    on a fixed torch, pass) - never train on unchecked compiled gradients."""
    learner = _learner("bv8", lm.LearnerOpts(fast=True, compile="default"), device="mps")
    _train(learner, 4)
    st = learner.compile_status()
    assert st["fallback"] is not None or st["verify_worst_rel_err"] < lm.VERIFY_TOL
    if st["fallback"]:
        assert st["compiled_calls"] == 0 and "disagrees with eager" in st["fallback"]
    assert all(bool(torch.isfinite(p).all()) for p in learner.online.parameters())
    torch.manual_seed(1)
    net = Ego2sNet(7, 60).to("mps").eval()
    hf = cm.HeroForward(torch, net, "mps", "default", 16, 60, m3c.LOCAL, m3c.GLOBAL)
    assert hf.fallback is None, hf.fallback
    obs = _obs(np.random.default_rng(3), 11, 60)
    with torch.no_grad():
        q = hf(obs).cpu()
        ref = net(*[torch.from_numpy(obs[k]).to("mps") for k in ("local", "global", "scalars")])
    assert float((q - ref.cpu()).abs().max()) < Q_ATOL
