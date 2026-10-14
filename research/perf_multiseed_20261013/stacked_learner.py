"""Multi-seed learner prototype (perf pass item 5, 2026-10-13): K independent ego2s Q-networks
updated in ONE process. Research prototype - not wired into ``train_m3c``.

Each seed k owns its online and target :class:`~src.model.ego2s_network.Ego2sNet`, its Adam
state and its own batch (from its own replay). The K online parameter sets are stacked along a
new leading dimension (:func:`torch.func.stack_module_state`), and the three forwards of the
``learner_m3c`` update run as one ``vmap(functional_call)`` each over the stack. Under vmap,
convolutions lower to K-group convolutions and linears to batched matmuls. The per-seed update
is the fp32 ``learner_m3c`` T1 update: Double DQN n-step TD (smooth L1) toward the target
network's value of the online argmax over the legal next actions, plus ``lam`` times the M4-A
``v8margin`` anchor on the label rows, grad-norm clip 10, Adam (lr 1e-4), and a target sync
every 1000 updates. :func:`single_update` is that update for one model and one staged batch.
It is the per-seed reference, and the tests pin it bitwise to ``learner_m3c.Learner``.

Per-seed independence:

* loss: ``sum_k loss_k`` (non-finite seeds masked to 0) is backpropagated. Seeds share no
  parameter, so each stacked gradient slice ``grad[k]`` is seed k's gradient alone.
* clip: :func:`clip_per_seed_` applies ``torch.nn.utils.get_total_norm`` to seed k's gradient
  slices and then the ``clip_grad_norm_`` coefficient. Given equal gradients it is bitwise the
  reference clip.
* Adam: :class:`StackedAdam` keeps per-seed views of the stacked ``exp_avg`` / ``exp_avg_sq``
  and one step counter per (seed, parameter). It steps them with ``torch.optim.adam.adam``,
  the functional form ``torch.optim.Adam.step`` calls, with the same single-tensor / foreach /
  fused choice. Given equal gradients it is bitwise ``torch.optim.Adam``. A seed whose loss is
  non-finite is left out of the call, so its weights and optimizer state do not move, and it
  is marked halted while the other seeds keep stepping. This is failure isolation in place of
  the single-run halt.
* target sync: per seed, on that seed's own update count.

Forward options (``forward=``): ``"vmap"`` (the default) is ``vmap(functional_call)`` over
the stacked parameters. ``"grouped"`` (:func:`grouped_forward`) writes the same network by
hand: the convolution activations stay in a channel layout ``(N, K*C, H, W)`` with one
``groups=K`` convolution per layer, and the linears are ``baddbmm``. vmap's convolution
batching rule moves the K dimension into the channels and back around every convolution (a
float copy of the activations each time); on MPS that alone made a vmap update ~2x as slow
as the grouped one (``DESIGN.md``).

Numerics: on CPU the stacked forward/backward is NOT bitwise the single-model one. The
grouped convolution / bmm reassociates against convolution / addmm, at about 1e-6 on Q, and
Adam's ``m / sqrt(v)`` can amplify that on near-zero gradients. Multi-step trajectories are
therefore compared with a tolerance (``tests/test_perf_multiseed.py``), and deliberate breaks
are shown to fail it.

Batch layout per seed (``learner_m3c._Staging``): observation rows ``[next (n) | current (n)
| anchor (m)]`` in ``local`` / ``global`` / ``scalars``; ``a``, ``R``, ``disc`` and ``nmask``
have n rows; ``a_v8`` and ``mask_resolved`` have m rows. Stacked batches add a leading K
dimension, with the same n and m for every seed.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, stack_module_state, vmap
from torch.optim.adam import adam as adam_functional

from research.redesign_m3c_20261008.learner_m3c import LR, TARGET_EVERY, _masked_argmax
from research.redesign_m3c_20261008.train_m3c import GLOBAL, LOCAL, anchor_loss_margin
from src.model.ego2s_network import Ego2sNet

CLIP = 10.0
BETAS = (0.9, 0.999)
EPS = 1e-8
ADAM_IMPLS = ("default", "fused")
PRECISIONS = ("fp32", "bf16")
FORWARDS = ("vmap", "grouped")
Batch = Dict[str, torch.Tensor]


# ----------------------------------------------------------------------------- loss pieces
def masked_argmax(q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``learner_m3c._masked_argmax`` without in-place indexing, over any leading dims: a row
    with no legal action falls back to the three non-boost actions. Bitwise the same result
    (boolean algebra + the same ``where`` / ``argmax``)."""
    first3 = torch.arange(q.shape[-1], device=q.device) < 3
    m = m | (~m.any(dim=-1, keepdim=True) & first3)
    return torch.where(m, q, torch.full_like(q, -1e9)).argmax(dim=-1)


def _anchor_one(q: torch.Tensor, a: torch.Tensor, mask: torch.Tensor, margin: float):
    return anchor_loss_margin(torch, q, a, mask, margin)


def anchor_margin_stacked(q: torch.Tensor, a: torch.Tensor, mask: torch.Tensor, margin: float):
    """``train_m3c.anchor_loss_margin`` per seed: ``(K, m, 6)`` -> ``(K,)`` (vmap of the
    production function itself)."""
    return vmap(_anchor_one, in_dims=(0, 0, 0, None))(q, a, mask, margin)


# ----------------------------------------------------------------------------- reference
def single_update(
    online: Ego2sNet,
    target: Ego2sNet,
    opt,
    b: Batch,
    n: int,
    m: int,
    lam: float,
    margin: float,
    state: Dict[str, Any],
    *,
    clip: float = CLIP,
    target_every: int = TARGET_EVERY,
) -> Dict[str, Any]:
    """``learner_m3c.Learner._update_fast`` + ``_finalize`` (fp32, v8margin anchor, no
    pipelining) on one staged batch ``b``: the per-seed reference. A non-finite loss applies no
    step (``state["halt"]`` is set; the caller stops stepping that seed)."""
    L, G, S = b["local"], b["global"], b["scalars"]
    cur, nxt = slice(n, 2 * n + m), slice(0, n)
    q_all = online(L[cur], G[cur], S[cur])
    q, qd = q_all[:n], q_all[n:]
    a = b["a"][:n]
    qa = q.gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():
        qn_online = online(L[nxt], G[nxt], S[nxt])
        qn_target = target(L[nxt], G[nxt], S[nxt])
        a_star = _masked_argmax(torch, qn_online, b["nmask"][:n])
        q_next = qn_target.gather(1, a_star[:, None]).squeeze(1)
        y = b["R"][:n] + b["disc"][:n] * q_next
    td = torch.nn.functional.smooth_l1_loss(qa, y)
    if m:
        anc = anchor_loss_margin(torch, qd, b["a_v8"][:m], b["mask_resolved"][:m], margin)
    else:
        anc = torch.zeros((), device=q.device)
    loss = td + lam * anc
    finite = bool(torch.isfinite(loss))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), clip)
    if finite:
        opt.step()
        state["updates"] = state.get("updates", 0) + 1
        if state["updates"] % target_every == 0:
            target.load_state_dict(online.state_dict())
    else:
        state["halt"] = "non-finite loss"
    return {
        "td": td.detach(),
        "anc": anc.detach(),
        "qmax": q.detach().abs().max(),
        "finite": finite,
    }


# ----------------------------------------------------------------------------- grouped forward
def _grouped_seq(seq: nn.Sequential, prefix: str, x, p: Dict[str, torch.Tensor], K: int, n: int):
    """One ``nn.Sequential`` tower of K stacked seeds without vmap.

    Convolution activations stay in the channel layout ``(n, K*C, H, W)`` (seed k owns channel
    block k; one ``groups=K`` convolution per layer, no per-layer permute); ``Flatten`` turns
    them into ``(K, n, F)``, where linears are ``baddbmm`` and LayerNorm normalizes each row
    and applies seed k's affine."""
    for i, mod in enumerate(seq):
        name = f"{prefix}.{i}"
        if isinstance(mod, nn.Conv2d):
            if mod.padding_mode != "zeros":
                raise TypeError("grouped forward: only zero-padded convolutions")
            w, b = p[name + ".weight"], p.get(name + ".bias")
            x = F.conv2d(
                x,
                w.flatten(0, 1),
                None if b is None else b.flatten(),
                mod.stride,
                mod.padding,
                mod.dilation,
                K * mod.groups,
            )
        elif isinstance(mod, nn.ReLU):
            x = F.relu(x)
        elif isinstance(mod, nn.Flatten):
            x = x.reshape(n, K, -1).transpose(0, 1)
        elif isinstance(mod, nn.Linear):
            x = _grouped_linear(x, p, name)
        elif isinstance(mod, nn.LayerNorm):
            x = F.layer_norm(x, mod.normalized_shape, eps=mod.eps)
            x = x * p[name + ".weight"].unsqueeze(1) + p[name + ".bias"].unsqueeze(1)
        else:
            raise TypeError(f"grouped forward: unsupported module {type(mod).__name__}")
    return x


def _grouped_linear(x, p: Dict[str, torch.Tensor], name: str):
    """``(K, n, in)`` -> ``(K, n, out)`` with seed k's ``weight[k]`` / ``bias[k]``."""
    return torch.baddbmm(p[name + ".bias"].unsqueeze(1), x, p[name + ".weight"].transpose(1, 2))


def grouped_forward(base: Ego2sNet, p: Dict[str, torch.Tensor], local, glob, scalars):
    """``Ego2sNet.forward`` for K stacked seeds as grouped convolutions + batched matmuls (the
    ``forward="grouped"`` option). ``(K, n, ...)`` observations -> ``(K, n, 6)`` Q. The uint8
    planes are permuted once to the channel layout (a uint8 copy, 1/4 of a float one); vmap
    instead permutes the float activations around every convolution."""
    K, n = local.shape[0], local.shape[1]
    lo = local.transpose(0, 1).reshape((n, -1) + tuple(local.shape[3:])).float() * (1.0 / 255.0)
    gl = glob.transpose(0, 1).reshape((n, -1) + tuple(glob.shape[3:])).float() * (1.0 / 255.0)
    hl = _grouped_seq(base.local, "local", lo, p, K, n)
    hg = _grouped_seq(base.glob, "glob", gl, p, K, n)
    h = _grouped_seq(base.fuse, "fuse", torch.cat([hl, hg, scalars.float()], dim=2), p, K, n)
    a = _grouped_linear(h, p, "adv")
    return _grouped_linear(h, p, "value") + a - a.mean(dim=2, keepdim=True)


# ----------------------------------------------------------------------------- stacked model
class StackedEgo2s:
    """K :class:`Ego2sNet` parameter sets as stacked leaf tensors + one batched forward:
    ``forward="vmap"`` (``vmap(functional_call)``, the default) or ``"grouped"``
    (:func:`grouped_forward`, the same math without vmap's per-layer layout permutes)."""

    def __init__(self, nets: Sequence[Ego2sNet], device=None, forward: str = "vmap") -> None:
        nets = list(nets)
        if not nets:
            raise ValueError("need at least one network")
        if forward not in FORWARDS:
            raise ValueError(f"forward must be one of {FORWARDS}")
        self.config = dict(nets[0].config)
        if any(dict(net.config) != self.config for net in nets):
            raise ValueError("all seeds must share one Ego2sNet config")
        self.K = len(nets)
        self.forward_impl = forward
        params, buffers = stack_module_state(nets)
        dev = torch.device(device) if device is not None else next(iter(params.values())).device
        self.device = dev
        self.params = {k: v.detach().to(dev).requires_grad_(True) for k, v in params.items()}
        self.buffers = {k: v.to(dev) for k, v in buffers.items()}  # Ego2sNet has none
        self.names = list(self.params)
        base = copy.deepcopy(nets[0]).to("meta")
        self._base = base

        def fwd(p, bufs, local, glob, scalars):
            return functional_call(base, (p, bufs), (local, glob, scalars))

        self._fwd = vmap(fwd)

    def forward(self, params: Dict[str, torch.Tensor], local, glob, scalars) -> torch.Tensor:
        """``(K, N, ...)`` observations -> ``(K, N, 6)`` Q with parameter set ``params``."""
        if self.forward_impl == "grouped":
            return grouped_forward(self._base, params, local, glob, scalars)
        return self._fwd(params, self.buffers, local, glob, scalars)

    def seed_state_dict(self, k: int, params: Optional[Dict[str, torch.Tensor]] = None):
        """Seed k's ``Ego2sNet`` state dict (CPU copies) - per-seed checkpointing."""
        src = self.params if params is None else params
        return {name: src[name][k].detach().cpu().clone() for name in self.names}

    def to_net(self, k: int, params: Optional[Dict[str, torch.Tensor]] = None) -> Ego2sNet:
        net = Ego2sNet(**self.config)
        net.load_state_dict(self.seed_state_dict(k, params))
        return net

    @torch.no_grad()
    def load_seed(self, k: int, state_dict: Dict[str, torch.Tensor], params=None) -> None:
        dst = self.params if params is None else params
        for name in self.names:
            dst[name][k].copy_(state_dict[name])


# ----------------------------------------------------------------------------- clip + Adam
@torch.no_grad()
def clip_per_seed_(params: Sequence[torch.Tensor], K: int, max_norm: float = CLIP):
    """``clip_grad_norm_(seed k's parameters, max_norm)`` for every seed k of stacked leaves
    (gradients ``(K, ...)``). Returns the ``(K,)`` pre-clip total norms."""
    grads = [p.grad for p in params]
    totals = torch.stack(
        [torch.nn.utils.get_total_norm([g[k] for g in grads], 2.0) for k in range(K)]
    )
    coef = torch.clamp(float(max_norm) / (totals + 1e-6), max=1.0)  # == _clip_grads_with_norm_
    for g in grads:
        g.mul_(coef.view((K,) + (1,) * (g.dim() - 1)))
    return totals


class StackedAdam:
    """Per-seed Adam over stacked leaves, stepped with ``torch.optim.adam.adam``.

    ``impl="default"`` is ``torch.optim.Adam(params, lr)`` (single-tensor on CPU / MPS, foreach
    on CUDA); ``impl="fused"`` is ``torch.optim.Adam(params, lr, fused=True)`` (the
    ``--learner-fast`` optimizer). Every seed has its own step counters, so a seed left out of
    a step keeps its bias correction and moments untouched.
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        K: int,
        lr: float = LR,
        betas=BETAS,
        eps: float = EPS,
        impl: str = "default",
    ) -> None:
        if impl not in ADAM_IMPLS:
            raise ValueError(f"impl must be one of {ADAM_IMPLS}")
        self.params = list(params)
        self.K = int(K)
        self.lr, self.betas, self.eps = float(lr), tuple(betas), float(eps)
        self.fused = impl == "fused"
        dev = self.params[0].device
        self.exp_avg = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]
        self.exp_avg_sq = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]

        def new_step():  # == torch.optim.Adam._init_group (non-capturable)
            if self.fused:
                return torch.zeros((), dtype=torch.float32, device=dev)
            return torch.tensor(0.0, dtype=torch.float32)

        self.steps = [[new_step() for _ in self.params] for _ in range(self.K)]
        self._p = [[p.detach()[k] for p in self.params] for k in range(self.K)]
        self._m = [[t[k] for t in self.exp_avg] for k in range(self.K)]
        self._v = [[t[k] for t in self.exp_avg_sq] for k in range(self.K)]

    @torch.no_grad()
    def step(self, active: Sequence[bool]) -> None:
        P: List[torch.Tensor] = []
        G: List[torch.Tensor] = []
        M: List[torch.Tensor] = []
        V: List[torch.Tensor] = []
        S: List[torch.Tensor] = []
        for k in range(self.K):
            if not active[k]:
                continue
            P += self._p[k]
            G += [p.grad[k] for p in self.params]
            M += self._m[k]
            V += self._v[k]
            S += self.steps[k]
        if not P:
            return
        adam_functional(
            P,
            G,
            M,
            V,
            [],
            S,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=True if self.fused else None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            decoupled_weight_decay=False,
            amsgrad=False,
            beta1=self.betas[0],
            beta2=self.betas[1],
            lr=self.lr,
            weight_decay=0.0,
            eps=self.eps,
            maximize=False,
        )

    def seed_state(self, k: int) -> Dict[str, Any]:
        """Seed k's optimizer state (CPU copies; per-seed checkpointing)."""
        return {
            "step": [float(s) for s in self.steps[k]],
            "exp_avg": [t.detach().cpu().clone() for t in self._m[k]],
            "exp_avg_sq": [t.detach().cpu().clone() for t in self._v[k]],
        }


# ----------------------------------------------------------------------------- learner
class StackedLearner:
    """K independent ``learner_m3c`` updates (one per seed) per :meth:`update` call."""

    def __init__(
        self,
        nets: Sequence[Ego2sNet],
        targets: Optional[Sequence[Ego2sNet]] = None,
        *,
        device,
        margin: float,
        lr: float = LR,
        clip: float = CLIP,
        target_every: int = TARGET_EVERY,
        adam: str = "default",
        precision: str = "fp32",
        forward: str = "vmap",
    ) -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS} (fp16 needs per-seed scalers)")
        self.device = torch.device(device)
        self.model = StackedEgo2s(
            [copy.deepcopy(n).to(self.device) for n in nets], self.device, forward=forward
        )
        self.K = self.model.K
        tnets = nets if targets is None else targets
        tp, _ = stack_module_state([copy.deepcopy(n).to(self.device) for n in tnets])
        self.tparams = {k: v.detach().clone() for k, v in tp.items()}
        self.opt = StackedAdam(list(self.model.params.values()), self.K, lr=lr, impl=adam)
        self.margin, self.clip, self.target_every = float(margin), float(clip), int(target_every)
        self.updates = np.zeros(self.K, dtype=np.int64)
        self.halted = np.zeros(self.K, dtype=bool)
        self._amp = {"bf16": torch.bfloat16}.get(precision)

    @property
    def params(self) -> Dict[str, torch.Tensor]:
        return self.model.params

    def _fwd(self, params, L, G, S) -> torch.Tensor:
        if self._amp is None:
            return self.model.forward(params, L, G, S)
        with torch.autocast(self.device.type, dtype=self._amp):
            q = self.model.forward(params, L, G, S)
        return q.float()

    def update(self, b: Batch, n: int, m: int, lam: Union[float, torch.Tensor]) -> Dict[str, Any]:
        """One update of every non-halted seed on its slice ``b[...][k]``."""
        K, params, tparams = self.K, self.model.params, self.tparams
        L, G, S = b["local"], b["global"], b["scalars"]
        cur, nxt = slice(n, 2 * n + m), slice(0, n)
        q_all = self._fwd(params, L[:, cur], G[:, cur], S[:, cur])
        q, qd = q_all[:, :n], q_all[:, n:]
        qa = q.gather(2, b["a"][:, :n, None]).squeeze(2)
        with torch.no_grad():
            qn_online = self._fwd(params, L[:, nxt], G[:, nxt], S[:, nxt])
            qn_target = self._fwd(tparams, L[:, nxt], G[:, nxt], S[:, nxt])
            a_star = masked_argmax(qn_online, b["nmask"][:, :n])
            q_next = qn_target.gather(2, a_star[..., None]).squeeze(2)
            y = b["R"][:, :n] + b["disc"][:, :n] * q_next
        td = torch.nn.functional.smooth_l1_loss(qa, y, reduction="none").mean(dim=1)
        if m:
            anc = anchor_margin_stacked(
                qd, b["a_v8"][:, :m], b["mask_resolved"][:, :m], self.margin
            )
        else:
            anc = torch.zeros(K, device=q.device)
        loss = td + lam * anc
        finite = torch.isfinite(loss)
        for p in params.values():
            p.grad = None
        torch.where(finite, loss, torch.zeros_like(loss)).sum().backward()
        clip_per_seed_(list(params.values()), K, self.clip)
        fin = finite.cpu().numpy()  # the one host sync for all K seeds
        self.halted |= ~fin
        ok = fin & ~self.halted
        self.opt.step(ok)
        self.updates[ok] += 1
        due = ok & (self.updates % self.target_every == 0)
        if due.any():
            self._sync_targets(due)
        return {
            "td": td.detach(),
            "anc": anc.detach(),
            "qmax": q.detach().abs().amax(dim=(1, 2)),
            "finite": fin,
            "stepped": ok,
        }

    @torch.no_grad()
    def _sync_targets(self, due: np.ndarray) -> None:
        for name, p in self.model.params.items():
            if due.all():
                self.tparams[name].copy_(p)
            else:
                for k in np.flatnonzero(due):
                    self.tparams[name][k].copy_(p[k])

    def seed_nets(self, k: int):
        """Seed k's (online, target) as plain ``Ego2sNet`` (CPU) - per-seed checkpoints."""
        return self.model.to_net(k), self.model.to_net(k, self.tparams)


# ----------------------------------------------------------------------------- batches
def synthetic_batch(rng: np.random.Generator, n: int, m: int, nscal: int) -> Dict[str, np.ndarray]:
    """A random staged batch in the ``_Staging`` layout (sparse uint8 planes like rasters)."""
    rows = 2 * n + m

    def planes(shape):
        dense = rng.integers(1, 256, (rows,) + shape, dtype=np.uint8)
        return np.where(rng.random((rows,) + shape, dtype=np.float32) < 0.15, dense, 0).astype(
            np.uint8
        )

    return {
        "local": planes(LOCAL),
        "global": planes(GLOBAL),
        "scalars": rng.standard_normal((rows, nscal), dtype=np.float32),
        "a": rng.integers(0, 6, n).astype(np.int64),
        "R": (rng.standard_normal(n) * 0.3).astype(np.float32),
        "disc": np.where(rng.random(n) < 0.05, 0.0, 0.97**5).astype(np.float32),
        "nmask": rng.random((n, 6)) < 0.8,
        "a_v8": rng.integers(0, 6, m).astype(np.int64),
        "mask_resolved": rng.random((m, 6)) < 0.8,
    }


def to_device(b: Dict[str, np.ndarray], device) -> Batch:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in b.items()}


def stack_batches(batches: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.stack([b[k] for b in batches]) for k in batches[0]}
