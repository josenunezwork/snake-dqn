"""perf pass item 5 (2026-10-13): the multi-seed (stacked, vmapped) learner prototype.

What is pinned, and how tightly:

* ``single_update`` (the per-seed reference) IS the ``learner_m3c`` T1 update: bitwise on CPU
  against ``Learner`` itself (plain and fused Adam), batch for batch.
* the stacked loss pieces, per-seed clip and :class:`StackedAdam` are bitwise the
  single-model ones given equal inputs / gradients; a global clip and shared Adam moments
  (deliberate breaks) fail those checks.
* K stacked updates track K separate ``single_update`` runs within a tolerance (grouped
  convolution / bmm reassociate on CPU; Adam amplifies that on near-zero gradients), and every
  deliberate break (batch mix-up, shared Adam moments, target sync from the wrong seed, no
  target sync, global clip, wrong anchor weight) fails the same check by orders of magnitude.
* seeds are isolated bitwise: other seeds' data, or a non-finite loss in another seed, do not
  change a seed's trajectory; the non-finite seed halts with its pre-step weights.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.perf_multiseed_20261013 import stacked_learner as sl  # noqa: E402
from research.redesign_m3c_20261008 import learner_m3c as lm  # noqa: E402
from research.redesign_m3c_20261008 import train_m3c as m3c  # noqa: E402
from src.model.ego2s_network import save_ego2s_checkpoint  # noqa: E402
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint  # noqa: E402

NSCAL = 60
K = 3
N, M = 24, 8  # agent / anchor rows per seed (small: CPU test)
STEPS = 6
TARGET_EVERY = 3
LAM, MARGIN = 0.5, 0.3


# ----------------------------------------------------------------------------- helpers
def _nets(k=K, base_seed=100):
    out = []
    for i in range(k):
        torch.manual_seed(base_seed + i)
        out.append(Ego2sNet(7, NSCAL))
    return out


def _batches(seed=0, steps=STEPS, k=K):
    rng = np.random.default_rng(seed)
    return [[sl.synthetic_batch(rng, N, M, NSCAL) for _ in range(k)] for _ in range(steps)]


def _flat(params):
    return torch.cat([p.detach().reshape(-1).cpu() for p in params])


def _seed_flat(lrn, k, params=None):
    sd = lrn.model.seed_state_dict(k, params)
    return torch.cat([sd[name].reshape(-1) for name in lrn.model.names])


def _reference(nets, batches, clip=sl.CLIP, adam_fused=False):
    """K separate single-model learners (the thing the stacked learner must equal)."""
    out = []
    for k, net in enumerate(nets):
        online = copy.deepcopy(net).train()
        target = copy.deepcopy(net).eval()
        opt = torch.optim.Adam(online.parameters(), lr=lm.LR, fused=adam_fused or None)
        st: dict = {}
        tds = []
        for step in batches:
            o = sl.single_update(
                online,
                target,
                opt,
                sl.to_device(step[k], "cpu"),
                N,
                M,
                LAM,
                MARGIN,
                st,
                clip=clip,
                target_every=TARGET_EVERY,
            )
            tds.append(float(o["td"]))
        out.append({"online": online, "target": target, "td": tds, "st": st})
    return out


def _run_stacked(nets, batches, *, lam=LAM, clip=sl.CLIP, roll=False, setup=None):
    lrn = sl.StackedLearner(nets, device="cpu", margin=MARGIN, target_every=TARGET_EVERY, clip=clip)
    if setup is not None:
        setup(lrn)
    tds = []
    for step in batches:
        bs = step[1:] + step[:1] if roll else step
        out = lrn.update(sl.to_device(sl.stack_batches(bs), "cpu"), N, M, lam)
        tds.append(out["td"].numpy().copy())
    return lrn, np.asarray(tds)


def _parity(nets, ref, lrn, tds, hold):
    """Worst-seed metrics of stacked vs reference (see PARITY_TOL)."""
    worst = {
        "drift_ratio": 0.0,
        "param_max_abs": 0.0,
        "target_max_abs": 0.0,
        "q_max_abs": 0.0,
        "td_rel": 0.0,
        "argmax_agree": 1.0,
    }
    for k, r in enumerate(ref):
        a = _flat(r["online"].parameters())
        b = _seed_flat(lrn, k)
        theta0 = _flat(nets[k].parameters())
        drift = float((a - b).norm() / (a - theta0).norm())
        worst["drift_ratio"] = max(worst["drift_ratio"], drift)
        worst["param_max_abs"] = max(worst["param_max_abs"], float((a - b).abs().max()))
        tgt = (_seed_flat(lrn, k, lrn.tparams) - _flat(r["target"].parameters())).abs().max()
        worst["target_max_abs"] = max(worst["target_max_abs"], float(tgt))
        with torch.no_grad():
            qa = r["online"](*hold)
            qb = lrn.model.to_net(k)(*hold)
        worst["q_max_abs"] = max(worst["q_max_abs"], float((qa - qb).abs().max()))
        agree = float((qa.argmax(1) == qb.argmax(1)).float().mean())
        worst["argmax_agree"] = min(worst["argmax_agree"], agree)
        td_ref = np.asarray(r["td"])
        td_rel = float(np.max(np.abs(td_ref - tds[:, k]) / np.abs(td_ref)))
        worst["td_rel"] = max(worst["td_rel"], td_rel)
    return worst


#: Measured on CPU (2026-10-13). Correct: drift_ratio <= ~1e-3, param / target / Q max abs
#: ~1e-4 (a few Adam sign flips on near-zero gradients, <= 2 lr per step), td_rel ~1e-5.
#: Every deliberate break: drift_ratio >= 0.19, td_rel >= 0.09, argmax agreement <= 0.8.
PARITY_TOL = {
    "drift_ratio": 1e-2,
    "param_max_abs": 2e-3,  # 6 steps x 2 x lr (1e-4) = 1.2e-3 worst case for sign flips
    "target_max_abs": 2e-3,
    "q_max_abs": 2e-3,
    "td_rel": 1e-3,
}


def _within(w) -> bool:
    return all(w[k] < tol for k, tol in PARITY_TOL.items()) and w["argmax_agree"] == 1.0


def _hold():
    b = sl.to_device(sl.synthetic_batch(np.random.default_rng(77), 64, 0, NSCAL), "cpu")
    return b["local"][:64], b["global"][:64], b["scalars"][:64]


@pytest.fixture(scope="module")
def setup_ref():
    nets = _nets()
    batches = _batches()
    return nets, batches, _reference(nets, batches)


# ----------------------------------------------------------------------------- 1. reference
class _Buf:
    def __init__(self, n: int) -> None:
        self.buf = bytearray(n)


class _FlakyMulti(m3c.MultiRing):
    """valid() drops a call-dependent subset (exercises the learner's row compaction)."""

    def __init__(self, rings) -> None:
        super().__init__(rings)
        self.calls = 0

    def valid(self, ids):
        self.calls += 1
        return super().valid(ids) & ((np.asarray(ids) + self.calls) % 11 != 0)


def _learner_world(seed=0, cap=1500):
    r = np.random.default_rng(seed)
    rings = []
    for _ in range(2):
        ring = m3c.SharedObsRing(_Buf(m3c.ring_nbytes(cap, NSCAL)), cap, 16, NSCAL)
        a = ring.arrays
        a["local"][:] = r.integers(0, 256, a["local"].shape, dtype=np.uint8)
        a["global"][:] = r.integers(0, 256, a["global"].shape, dtype=np.uint8)
        a["scalars"][:] = r.standard_normal(a["scalars"].shape)
        a["mask"][:] = r.random(a["mask"].shape) < 0.7
        ring._count[0] = cap
        rings.append(ring)
    multi = _FlakyMulti(rings)
    replay = m3c.Replay(4000)
    s = r.integers(0, 2 * cap, 2000)
    nxt = np.where(r.random(2000) < 0.1, -1, r.integers(0, 2 * cap, 2000))
    replay.add_many(s, r.integers(0, 6, 2000), r.standard_normal(2000), r.random(2000) * 0.9, nxt)
    labels = m3c.LabelBuffer(m3c.LABEL_ANCHOR_MIN_ROWS + 500)
    nl = m3c.LABEL_ANCHOR_MIN_ROWS + 100
    labels.add(r.integers(0, 2 * cap, nl), r.integers(0, 6, nl))
    return multi, replay, labels


@pytest.mark.parametrize("fused", [False, True])
def test_single_update_is_the_learner_fast_update_bitwise(fused):
    """The per-seed reference equals ``learner_m3c.Learner`` (T1, bv8 / v8margin) bitwise on
    the batches the Learner itself sampled (ring-overrun compaction included)."""
    torch.manual_seed(3)
    online = Ego2sNet(7, NSCAL).train()
    target = Ego2sNet(7, NSCAL).eval()
    target.load_state_dict(online.state_dict())
    init = copy.deepcopy(online)
    multi, replay, labels = _learner_world()
    args = m3c.default_args()
    args.actor_sync_updates = 25
    args.obs_version, args.onpolicy_labels, args.label_frac = "bv8", True, 1.0
    args.anchor_mode = "v8margin"
    opts = lm.LearnerOpts(fast=True, fused_adam=fused)
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
    learner = lm.Learner(
        torch,
        online=online,
        target=target,
        opt=lm.make_optimizer(torch, online, opts, "cpu"),
        replay=replay,
        multi=multi,
        demos=None,
        labels=labels,
        rng=np.random.default_rng([7, 17]),
        st=st,
        args=args,
        nscal=NSCAL,
        margin=MARGIN,
        device="cpu",
        opts=opts,
        agent_rows=N,
        anchor_rows=M,
        min_rows=4,
    )
    learner.target_every = TARGET_EVERY
    captured = []
    sample = learner._sample_fast

    def spy():
        out = sample()
        if out is not None:
            v = out["stage"].v
            captured.append(({k: torch.from_numpy(v[k].copy()) for k in v}, out["n"], out["m"]))
        return out

    learner._sample_fast = spy
    for _ in range(7):
        learner.update(LAM)
    learner.flush()
    assert st["updates"] == 7 and st["ring_overrun_rows"] > 0
    assert len({n for _, n, _ in captured}) > 1  # compaction changed n at least once

    ref_on = copy.deepcopy(init).train()
    ref_tg = copy.deepcopy(init).eval()
    ref_opt = torch.optim.Adam(ref_on.parameters(), lr=lm.LR, fused=fused or None)
    rst: dict = {}
    tds = []
    for b, n, m in captured:
        out = sl.single_update(
            ref_on, ref_tg, ref_opt, b, n, m, LAM, MARGIN, rst, target_every=TARGET_EVERY
        )
        tds.append(float(out["td"]))
    for a, b in zip(online.state_dict().values(), ref_on.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(target.state_dict().values(), ref_tg.state_dict().values()):
        assert torch.equal(a, b)
    assert tds == st["td"] and rst["updates"] == 7


# ----------------------------------------------------------------------------- 2. pieces
def test_loss_pieces_match_production_bitwise():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(K, 50, 6, generator=g)
    mask = torch.rand(K, 50, 6, generator=g) < 0.5
    mask[:, :5] = False  # rows with no legal action -> the non-boost fallback
    a = torch.randint(0, 6, (K, 50), generator=g)
    got = sl.masked_argmax(q, mask)
    anc = sl.anchor_margin_stacked(q, a, mask, MARGIN)
    for k in range(K):
        assert torch.equal(got[k], lm._masked_argmax(torch, q[k], mask[k]))
        assert torch.equal(anc[k], m3c.anchor_loss_margin(torch, q[k], a[k], mask[k], MARGIN))


def _stacked_grads(scales, seed=0):
    g = torch.Generator().manual_seed(seed)
    nets = _nets()
    model = sl.StackedEgo2s(nets)
    s = torch.tensor(scales)
    for p in model.params.values():
        p.grad = torch.randn(p.shape, generator=g) * s.view((K,) + (1,) * (p.dim() - 1))
    return model


@pytest.mark.parametrize(
    "max_norm,scales",
    [
        (10.0, (1.0, 1e-3, 1e-6)),  # gradient norms ~1300 / ~1.3 / ~1e-3: clipped, not, not
        (1e-4, (1e-6, 1e-6, 1e-7)),  # all clipped; the 1e-6 in the coefficient matters here
    ],
)
def test_clip_per_seed_is_clip_grad_norm_per_seed_bitwise(max_norm, scales):
    model = _stacked_grads(scales)
    params = list(model.params.values())
    per_seed = []
    for k in range(K):  # reference: clip_grad_norm_ on a separate copy of seed k's grads
        ps = [torch.nn.Parameter(p.detach()[k].clone()) for p in params]
        for p, src in zip(ps, params):
            p.grad = src.grad[k].clone()
        torch.nn.utils.clip_grad_norm_(ps, max_norm)
        per_seed.append([p.grad for p in ps])
    sl.clip_per_seed_(params, K, max_norm)
    for k in range(K):
        for p, ref in zip(params, per_seed[k]):
            assert torch.equal(p.grad[k], ref)
    # deliberate break: one global clip over all seeds is NOT the per-seed clip
    model2 = _stacked_grads(scales)
    params2 = list(model2.params.values())
    torch.nn.utils.clip_grad_norm_(params2, max_norm)
    same = [all(torch.equal(p.grad[k], r) for p, r in zip(params2, per_seed[k])) for k in range(K)]
    assert not all(same)  # (a dominant seed's own norm can equal the global one in fp32)


@pytest.mark.parametrize("impl", ["default", "fused"])
def test_stacked_adam_is_torch_adam_per_seed_bitwise(impl):
    """Equal gradients -> bitwise ``torch.optim.Adam`` per seed, including a step in which
    seed 1 is left out (its state and weights stay put, its bias correction does not
    advance)."""
    nets = _nets()
    model = sl.StackedEgo2s(nets)
    params = list(model.params.values())
    opt = sl.StackedAdam(params, K, impl=impl)
    refs = [copy.deepcopy(n) for n in nets]
    ropts = [
        torch.optim.Adam(r.parameters(), lr=lm.LR, fused=(impl == "fused") or None) for r in refs
    ]
    g = torch.Generator().manual_seed(1)
    plan = [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    for active in plan:
        for p in params:
            p.grad = torch.randn(p.shape, generator=g) * 1e-2
        for k, (r, ro) in enumerate(zip(refs, ropts)):
            for rp, p in zip(r.parameters(), params):
                rp.grad = p.grad[k].clone()
            if active[k]:
                ro.step()
        opt.step([bool(a) for a in active])
    for k, r in enumerate(refs):
        for rp, p in zip(r.parameters(), params):
            assert torch.equal(rp.detach(), p.detach()[k])
        st = ropts[k].state[next(iter(r.parameters()))]
        assert float(st["step"]) == float(opt.steps[k][0]) == sum(a[k] for a in plan)
        assert torch.equal(st["exp_avg_sq"], opt.exp_avg_sq[0][k])

    # deliberate break: seeds sharing one set of moments are not independent Adams
    model2 = sl.StackedEgo2s(nets)
    bad = sl.StackedAdam(list(model2.params.values()), K, impl=impl)
    for k in range(K):
        bad._m[k], bad._v[k] = bad._m[0], bad._v[0]
    g = torch.Generator().manual_seed(1)
    for active in plan:
        for p in model2.params.values():
            p.grad = torch.randn(p.shape, generator=g) * 1e-2
        bad.step([bool(a) for a in active])
    assert not all(
        torch.equal(rp.detach(), p.detach()[2])
        for rp, p in zip(refs[2].parameters(), model2.params.values())
    )


# ----------------------------------------------------------------------------- 3. end to end
def test_stacked_tracks_k_single_updates(setup_ref):
    nets, batches, ref = setup_ref
    lrn, tds = _run_stacked(nets, batches)
    assert (lrn.updates == STEPS).all() and not lrn.halted.any()
    w = _parity(nets, ref, lrn, tds, _hold())
    assert _within(w), w


def _break_shared_adam(lrn):
    for k in range(lrn.K):
        lrn.opt._m[k], lrn.opt._v[k] = lrn.opt._m[0], lrn.opt._v[0]


def _break_target_from_seed0(lrn):
    def bad(due):
        with torch.no_grad():
            for name, p in lrn.model.params.items():
                lrn.tparams[name].copy_(p[0].expand_as(p))

    lrn._sync_targets = bad


def _break_no_target_sync(lrn):
    lrn._sync_targets = lambda due: None


@pytest.mark.parametrize(
    "brk",
    ["batch_roll", "shared_adam", "target_from_seed0", "no_target_sync", "global_clip", "lam"],
)
def test_parity_check_fails_on_deliberate_breaks(setup_ref, brk, monkeypatch):
    nets, batches, ref = setup_ref
    setup = {
        "shared_adam": _break_shared_adam,
        "target_from_seed0": _break_target_from_seed0,
        "no_target_sync": _break_no_target_sync,
    }.get(brk)
    if brk == "global_clip":
        monkeypatch.setattr(
            sl, "clip_per_seed_", lambda ps, k, mx=10.0: torch.nn.utils.clip_grad_norm_(ps, mx)
        )
    lrn, tds = _run_stacked(
        nets, batches, setup=setup, roll=brk == "batch_roll", lam=1.0 if brk == "lam" else LAM
    )
    w = _parity(nets, ref, lrn, tds, _hold())
    assert not _within(w), (brk, w)
    assert w["drift_ratio"] > 10 * PARITY_TOL["drift_ratio"], (brk, w)


# ----------------------------------------------------------------------------- 4. isolation
def test_seeds_are_isolated_bitwise():
    nets = _nets()
    batches = _batches()
    base, _ = _run_stacked(nets, batches)
    other = _batches(seed=5)  # seed 1 sees different data
    mixed = [[b[0], o[1], b[2]] for b, o in zip(batches, other)]
    alt, _ = _run_stacked(nets, mixed)
    for k in (0, 2):
        assert torch.equal(_seed_flat(base, k), _seed_flat(alt, k))
        assert torch.equal(_seed_flat(base, k, base.tparams), _seed_flat(alt, k, alt.tparams))
    assert not torch.equal(_seed_flat(base, 1), _seed_flat(alt, 1))

    # non-finite loss in seed 1 at its third update: seed 1 halts with its weights after two
    # updates; seeds 0 and 2 are bitwise unaffected and keep stepping.
    two, _ = _run_stacked(nets, batches[:2])
    poisoned = copy.deepcopy(batches)
    for step in poisoned[2:3]:
        step[1] = dict(step[1], R=np.full_like(step[1]["R"], np.nan))
    bad, _ = _run_stacked(nets, poisoned)
    assert bad.halted.tolist() == [False, True, False]
    assert bad.updates.tolist() == [STEPS, 2, STEPS]
    assert torch.equal(_seed_flat(bad, 1), _seed_flat(two, 1))
    for k in (0, 2):
        assert torch.equal(_seed_flat(bad, k), _seed_flat(base, k))
    assert all(torch.isfinite(p).all() for p in bad.params.values())


# ----------------------------------------------------------------------------- 5. misc
def test_seed_checkpoints_roundtrip(tmp_path):
    nets = _nets()
    lrn = sl.StackedLearner(nets, device="cpu", margin=MARGIN)
    for k, net in enumerate(nets):
        online, target = lrn.seed_nets(k)
        for a, b in zip(net.state_dict().values(), online.state_dict().values()):
            assert torch.equal(a, b)
        save_ego2s_checkpoint(tmp_path / f"s{k}.pth", online, {"seed": k})
        back = load_ego2s_checkpoint(str(tmp_path / f"s{k}.pth"))
        for a, b in zip(net.state_dict().values(), back.state_dict().values()):
            assert torch.equal(a, b)
    nets[1].local[0].bias.data.add_(1.0)
    lrn.model.load_seed(1, nets[1].state_dict())
    assert torch.equal(lrn.model.seed_state_dict(1)["local.0.bias"], nets[1].local[0].bias)
    assert lrn.opt.seed_state(0)["step"][0] == 0.0
    with pytest.raises(ValueError):
        sl.StackedEgo2s([Ego2sNet(7, 60), Ego2sNet(7, 30)])
    with pytest.raises(ValueError):
        sl.StackedLearner(nets, device="cpu", margin=MARGIN, precision="fp16")


def test_bf16_autocast_runs_and_stays_close():
    """bf16 autocast composes with vmap: losses within bf16 rounding of fp32, steps applied."""
    nets = _nets(2)
    batches = _batches(steps=2, k=2)
    lrn = sl.StackedLearner(nets, device="cpu", margin=MARGIN, precision="bf16")
    ref = sl.StackedLearner(nets, device="cpu", margin=MARGIN)
    b = sl.to_device(sl.stack_batches(batches[0]), "cpu")
    out, want = lrn.update(b, N, M, LAM), ref.update(b, N, M, LAM)  # same weights: compare
    assert out["finite"].all() and out["stepped"].all()
    assert torch.allclose(out["td"], want["td"], rtol=5e-2)
    assert torch.allclose(out["qmax"], want["qmax"], rtol=5e-2)
    assert not torch.equal(out["td"], want["td"])  # it really ran in bf16
    lrn.update(sl.to_device(sl.stack_batches(batches[1]), "cpu"), N, M, LAM)
    assert (lrn.updates == 2).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_stacked_tracks_sequential():
    """torch.func batching works on MPS (torch 2.9.1): stacked == K MPS single updates within
    the CPU tolerance."""
    nets = _nets(2)
    batches = _batches(steps=3, k=2)
    lrn = sl.StackedLearner(nets, device="mps", margin=MARGIN, target_every=2, adam="fused")
    for step in batches:
        out = lrn.update(sl.to_device(sl.stack_batches(step), "mps"), N, M, LAM)
        assert out["finite"].all()
    for k, net in enumerate(nets):
        online = copy.deepcopy(net).to("mps").train()
        target = copy.deepcopy(net).to("mps").eval()
        opt = torch.optim.Adam(online.parameters(), lr=lm.LR, fused=True)
        st: dict = {}
        for step in batches:
            sl.single_update(
                online,
                target,
                opt,
                sl.to_device(step[k], "mps"),
                N,
                M,
                LAM,
                MARGIN,
                st,
                target_every=2,
            )
        a, b = _flat(online.parameters()), _seed_flat(lrn, k)
        ratio = float((a - b).norm() / (a - _flat(net.parameters())).norm())
        assert ratio < PARITY_TOL["drift_ratio"], ratio
