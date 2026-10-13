"""The multi-actor learner update of ``train_m3c.main_multi`` (perf-learner, 2026-10-13).

:class:`Learner` owns one Double DQN update (n-step TD toward the target network's value of
the online argmax, plus the anchor term on the demonstration / on-policy label rows, Adam,
grad-norm clip 10, target sync every 1000 updates). Three opt-in tiers; the defaults are the
pre-perf-learner update, moved here verbatim (``_update_reference``):

* **T1 ``fast``** (``--learner-fast``): the same math. Batches are gathered straight into a
  double-buffered (pinned on CUDA) staging block and uploaded in ONE host->device copy, then
  sliced on the device; the online and target "next" forwards share one upload; training
  statistics (TD, anchor, max |Q|) stay on the device until :meth:`Learner.flush` (log /
  checkpoint time); and the update is *pipelined*: the call for update ``n+1`` samples and
  gathers its batch on the CPU while the device still runs update ``n``'s forward/backward,
  then waits for ``n``'s finite-loss flag, applies ``n``'s optimizer step and enqueues
  ``n+1``. Sampling still happens after the transitions ingested since update ``n`` (same
  RNG stream, same replay state), so a fixed-seed run is bitwise identical to the reference
  on CPU with ``fused_adam=False`` (``tests/test_perf_learner.py``). ``--learner-fast`` also
  switches Adam to its fused kernel (one kernel per step; float-reassociation-level
  differences, ~1 ulp per step, also tested). Halting on a non-finite loss keeps its
  semantics: the offending step is never applied.
* **T2 ``precision``** (``--learner-precision fp16|bf16``; needs ``fast``): autocast for the
  three forwards (the loss is computed in fp32 from the cast-up Q), with a dynamic
  ``GradScaler`` for fp16 (an overflowed step is skipped but still counts as an update;
  fp16 uses the foreach Adam, see :func:`make_optimizer`). Numerics change; the drift is
  measured by ``bench_learner.py --drift``.
* **T3 ``batch_scale K``** (``--batch-scale K``): K x the agent (384) and anchor (128) rows
  per update and one update per K x 128 ingested transitions (the replay ratio is
  unchanged); the target sync (1000 updates) and the actor weight publication
  (``--actor-sync-updates``) intervals are divided by K so they stay fixed in *transitions*;
  ``--lr-scale linear|sqrt`` multiplies Adam's lr by K or sqrt(K). This CHANGES the training
  dynamics (K x fewer, larger-batch optimizer steps for the same data): it is a new recipe,
  not a speed-up of the old one, and needs its own evaluation.

Not done (measured on MPS, 2026-10-13, see ``research/perf_learner_20261013``): one forward
for online(next) and online(current + anchor) - the backward would then run over the 384
no-grad rows too, which costs more than the saved forward; ``torch.func.vmap`` over the
stacked online/target parameters (grouped convolutions: ~4x slower on MPS);
``channels_last`` (no gain on MPS; offered to the bench only, via ``channels_last``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from research.redesign_m3c_20261008.train_m3c import (
    GLOBAL,
    LABEL_ANCHOR_MIN_ROWS,
    LOCAL,
    UPDATE_EVERY,
    anchor_loss,
    anchor_loss_margin,
    anchor_loss_weighted,
)

PRECISIONS = ("fp32", "fp16", "bf16")
LR_SCALES = ("none", "linear", "sqrt")
AGENT_ROWS = 384
ANCHOR_ROWS = 128
MIN_ROWS = 64
TARGET_EVERY = 1000
LR = 1e-4


@dataclass(frozen=True)
class LearnerOpts:
    fast: bool = False  # T1
    fused_adam: bool = True  # T1 only (tests turn it off for the bitwise CPU comparison)
    precision: str = "fp32"  # T2 (needs fast)
    batch_scale: int = 1  # T3
    lr_scale: str = "none"  # T3
    channels_last: bool = False  # bench only

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if self.lr_scale not in LR_SCALES:
            raise ValueError(f"lr_scale must be one of {LR_SCALES}")
        if self.precision != "fp32" and not self.fast:
            raise ValueError("--learner-precision needs --learner-fast")
        if int(self.batch_scale) < 1:
            raise ValueError("--batch-scale must be >= 1")
        if self.lr_scale != "none" and self.batch_scale == 1:
            raise ValueError("--lr-scale needs --batch-scale > 1")

    @classmethod
    def from_args(cls, args) -> "LearnerOpts":
        return cls(
            fast=bool(getattr(args, "learner_fast", False)),
            precision=getattr(args, "learner_precision", "fp32"),
            batch_scale=int(getattr(args, "batch_scale", 1)),
            lr_scale=getattr(args, "lr_scale", "none"),
        )

    @property
    def lr(self) -> float:
        k = int(self.batch_scale)
        return LR * {"none": 1.0, "linear": float(k), "sqrt": math.sqrt(k)}[self.lr_scale]


def make_optimizer(torch, net, opts: LearnerOpts, device):
    """Adam(lr 1e-4 x the T3 lr scale); fused under T1 where the device supports it.

    Not with fp16: on MPS (torch 2.9) the fused step ignored the GradScaler's ``found_inf``
    after ``unscale_`` and applied overflowed (inf) gradients; the foreach Adam is used then
    (``GradScaler.step`` reads ``found_inf`` on the host: one extra sync per update)."""
    params = list(net.parameters())
    fused_ok = opts.fast and opts.fused_adam and opts.precision != "fp16"
    if fused_ok and torch.device(device).type in ("cuda", "mps", "cpu"):
        return torch.optim.Adam(params, lr=opts.lr, fused=True)
    return torch.optim.Adam(params, lr=opts.lr)


def _masked_argmax(torch, q, m):
    m = m.clone()
    m[~m.any(dim=1), :3] = True
    return torch.where(m, q, torch.full_like(q, -1e9)).argmax(dim=1)


class _Staging:
    """One flat host block holding a whole batch, sliced into typed numpy views.

    Layout (fixed offsets for the maximum row counts, 8-byte aligned): obs rows are ordered
    ``[next (n) | current (n) | anchor (m)]`` so ``current + anchor`` (the gradient forward,
    in the reference row order) and ``next`` are each one contiguous slice.
    """

    def __init__(self, torch, a_max: int, d_max: int, nscal: int, pin: bool) -> None:
        self.rows = 2 * a_max + d_max
        segs = [
            ("local", (self.rows,) + LOCAL, np.uint8),
            ("global", (self.rows,) + GLOBAL, np.uint8),
            ("scalars", (self.rows, nscal), np.float32),
            ("a", (a_max,), np.int64),
            ("R", (a_max,), np.float32),
            ("disc", (a_max,), np.float32),
            ("nmask", (a_max, 6), np.bool_),
            ("a_v8", (d_max,), np.int64),
            ("mask_resolved", (d_max, 6), np.bool_),
            ("q_teacher", (d_max, 6), np.float32),
        ]
        self.layout = []
        off = 0
        for name, shape, dtype in segs:
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.layout.append((name, shape, dtype, off, nbytes))
            off += -(-nbytes // 8) * 8
        self.nbytes = off
        if pin:
            self.host_t = torch.empty(off, dtype=torch.uint8, pin_memory=True)
            self.flat = self.host_t.numpy()
        else:
            self.flat = np.zeros(off, dtype=np.uint8)
            self.host_t = torch.from_numpy(self.flat)
        self.v = {
            name: np.ndarray(shape, dtype=dtype, buffer=self.flat, offset=o)
            for name, shape, dtype, o, _ in self.layout
        }

    def upload(self, torch, device, non_blocking: bool):
        dev = self.host_t.to(device, non_blocking=non_blocking)
        tdt = {
            np.uint8: torch.uint8,
            np.float32: torch.float32,
            np.int64: torch.int64,
            np.bool_: torch.bool,
        }
        out = {}
        for name, shape, dtype, o, nbytes in self.layout:
            seg = dev[o : o + nbytes]
            if dtype is not np.uint8:
                seg = seg.view(tdt[dtype])
            out[name] = seg.view(shape)
        return out


class Learner:
    """One learner update per call (see the module docstring for the three tiers)."""

    def __init__(
        self,
        torch,
        *,
        online,
        target,
        opt,
        replay,
        multi,
        demos,
        labels,
        rng: np.random.Generator,
        st: dict,
        args,
        nscal: int,
        margin: float,
        device,
        opts: Optional[LearnerOpts] = None,
        agent_rows: int = AGENT_ROWS,
        anchor_rows: int = ANCHOR_ROWS,
        min_rows: int = MIN_ROWS,
    ) -> None:
        self.torch = torch
        self.online, self.target, self.opt = online, target, opt
        self.replay, self.multi, self.demos, self.labels = replay, multi, demos, labels
        self.rng, self.st, self.args = rng, st, args
        self.nscal, self.margin = int(nscal), margin
        self.device = torch.device(device)
        self.opts = opts or LearnerOpts()
        k = int(self.opts.batch_scale)
        self.a_rows, self.d_rows, self.min_rows = agent_rows * k, anchor_rows * k, min_rows * k
        self.target_every = max(1, round(TARGET_EVERY / k))
        self.publish_every = max(1, round(int(args.actor_sync_updates) / k))
        self.update_every = UPDATE_EVERY * k
        if self.opts.channels_last:
            for net in (online, target):
                net.to(memory_format=torch.channels_last)
        # T1 state
        self._pending: Optional[dict] = None
        self._td: list = []
        self._anc: list = []
        self._qmax = None
        self.amp_skipped = 0
        if self.opts.fast:
            pin = self.device.type == "cuda"
            self._stage = [
                _Staging(torch, self.a_rows, self.d_rows, self.nscal, pin) for _ in range(2)
            ]
            self._which = 0
            self._non_blocking = pin
        self.scaler = None
        if self.opts.precision == "fp16":
            # init 2**10 (not 2**16): fewer skipped (overflowed) steps at the start
            self.scaler = torch.amp.GradScaler(self.device.type, init_scale=2.0**10)
        self._amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.opts.precision)

    # ------------------------------------------------------------------ public
    def update(self, lam: float) -> None:
        if self.opts.fast:
            self._update_fast(lam)
        else:
            self._update_reference(lam)

    def publish_due(self, before: int) -> bool:
        """Whether the main loop publishes weights after this call (reference: the M3-C
        condition, unchanged; fast: also only when the update counter moved)."""
        n = self.st["updates"]
        if self.opts.fast and n == before:
            return False
        return n % self.publish_every == 0

    def flush(self) -> None:
        """Apply a pending (pipelined) update and move device statistics into ``st``."""
        if not self.opts.fast:
            return
        self._finalize()
        torch = self.torch
        if self._td:
            self.st["td"].extend(torch.stack(self._td).cpu().tolist())
            self._td = []
        if self._anc:
            self.st["anc"].extend(torch.stack(self._anc).cpu().tolist())
            self._anc = []
        if self._qmax is not None:
            self.st["qmax"] = max(self.st["qmax"], float(self._qmax))
            self._qmax = None

    # ------------------------------------------------------------------ reference (M3-C)
    def _update_reference(self, lam: float) -> None:
        torch = self.torch
        st, rng, multi, replay = self.st, self.rng, self.multi, self.replay
        demos, labels, args, nscal = self.demos, self.labels, self.args, self.nscal
        online, target, opt, margin = self.online, self.target, self.opt, self.margin
        device = self.device

        def to_dev(x):
            return torch.from_numpy(np.ascontiguousarray(x)).to(device)

        def masked_argmax(q, m):
            return _masked_argmax(torch, q, m)

        t0 = time.perf_counter()
        idx = replay.sample(rng, self.a_rows, multi)
        if len(idx) < self.min_rows:
            return
        sl, sg, ss, _ = multi.get(replay.s[idx])
        nxt = replay.nxt[idx]
        nl, ng, ns, nm = multi.get(np.where(nxt >= 0, nxt, replay.s[idx]))
        # Re-check after copying: an overwrite during the gather invalidates the row.
        ok = multi.valid(replay.s[idx]) & ((nxt < 0) | multi.valid(nxt))
        if not ok.all():
            st["ring_overrun_rows"] += int((~ok).sum())
            keep = np.flatnonzero(ok)
            idx, sl, sg, ss = idx[keep], sl[keep], sg[keep], ss[keep]
            nl, ng, ns, nm = nl[keep], ng[keep], ns[keep], nm[keep]
            if len(idx) < self.min_rows:
                return
        a = to_dev(replay.a[idx])
        R = to_dev(replay.R[idx])
        disc = to_dev(replay.disc[idx])
        n_lab = 0
        if labels is not None and labels.size >= LABEL_ANCHOR_MIN_ROWS:
            n_lab = int(round(self.d_rows * args.label_frac))
        if demos is None:  # bv8: label rows only (TD only until LABEL_ANCHOR_MIN_ROWS labels)
            d = {
                "local": np.zeros((0,) + LOCAL, dtype=np.uint8),
                "global": np.zeros((0,) + GLOBAL, dtype=np.uint8),
                "scalars": np.zeros((0, nscal), dtype=np.float32),
                "a_v8": np.zeros(0, dtype=np.int64),
                "mask_resolved": np.zeros((0, 6), dtype=bool),
                "q_teacher": np.zeros((0, 6), dtype=np.float32),
            }
        else:
            d = demos.sample(rng, self.d_rows - n_lab)
        if n_lab:
            li = labels.sample(rng, n_lab, multi)
            lids = labels.ids[li]
            ll, lg, ls_, lm = multi.get(lids)
            keep = np.flatnonzero(multi.valid(lids))  # re-check after the gather
            ll, lg, ls_, lm, la = ll[keep], lg[keep], ls_[keep], lm[keep], labels.a[li][keep]
            n_lab = len(keep)
            d = {
                "local": np.concatenate([d["local"], ll]),
                "global": np.concatenate([d["global"], lg]),
                "scalars": np.concatenate([d["scalars"], ls_]),
                "a_v8": np.concatenate([d["a_v8"], la]),
                "mask_resolved": np.concatenate([d["mask_resolved"], lm]),
                "q_teacher": d["q_teacher"],  # demo rows only (unused by v8margin)
            }
        st["label_rows"].append(n_lab)
        n_agent = len(idx)
        q_all = online(
            to_dev(np.concatenate([sl, d["local"]])),
            to_dev(np.concatenate([sg, d["global"]])),
            to_dev(np.concatenate([ss, d["scalars"]])),
        )
        q, qd = q_all[:n_agent], q_all[n_agent:]
        qa = q.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            nlt, ngt, nst, nmt = to_dev(nl), to_dev(ng), to_dev(ns), to_dev(nm)
            a_star = masked_argmax(online(nlt, ngt, nst), nmt)
            q_next = target(nlt, ngt, nst).gather(1, a_star[:, None]).squeeze(1)
            y = R + disc * q_next
        td = torch.nn.functional.smooth_l1_loss(qa, y)
        db = {k: to_dev(d[k]) for k in ("q_teacher", "a_v8", "mask_resolved")}
        has_anchor = qd.shape[0] > 0
        if not has_anchor:  # bv8 before enough on-policy labels: TD only
            anc = torch.zeros((), device=q.device)
        elif args.anchor_mode == "v8margin":
            anc = anchor_loss_margin(torch, qd, db["a_v8"], db["mask_resolved"], margin)
        elif args.anchor_boost_weight == 1.0:
            anc = anchor_loss(torch, qd, db, margin)
        else:
            anc = anchor_loss_weighted(torch, qd, db, margin, args.anchor_boost_weight)
        loss = td + lam * anc
        if not torch.isfinite(loss):
            st["halt"] = "non-finite loss"
            return
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        st["updates"] += 1
        st["td"].append(float(td.detach()))
        if has_anchor:
            st["anc"].append(float(anc.detach()))
            st["anchor_updates"] += 1
        st["qmax"] = max(st["qmax"], float(q.detach().abs().max()))
        if st["updates"] % self.target_every == 0:
            target.load_state_dict(online.state_dict())
            if demos is not None:
                demos.prefetch()
        if demos is not None:
            demos.swap_if_ready()
        st["learn_s"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ T1 fast path
    def _gather(self, ids: np.ndarray, out: dict, rows: slice, mask_out=None) -> None:
        """``multi.get(ids)`` written straight into staging rows ``rows``."""
        multi = self.multi
        L, G, S = out["local"][rows], out["global"][rows], out["scalars"][rows]
        actor, local = ids % multi.k, ids // multi.k
        for j, ring in enumerate(multi.rings):
            sel = np.flatnonzero(actor == j)
            if not len(sel):
                continue
            pos = local[sel] % ring.cap
            arr = ring.arrays
            L[sel] = arr["local"][pos]
            G[sel] = arr["global"][pos]
            S[sel] = arr["scalars"][pos]
            if mask_out is not None:
                mask_out[sel] = arr["mask"][pos]

    @staticmethod
    def _compact(out: dict, start: int, n: int, keep: np.ndarray) -> None:
        for name in ("local", "global", "scalars"):
            seg = out[name]
            seg[start : start + len(keep)] = seg[start : start + n][keep]

    def _sample_fast(self) -> Optional[dict]:
        """The reference sampling (same RNG calls, same order, same rows) into staging."""
        st, rng, multi, replay = self.st, self.rng, self.multi, self.replay
        demos, labels, args = self.demos, self.labels, self.args
        idx = replay.sample(rng, self.a_rows, multi)
        if len(idx) < self.min_rows:
            return None
        stage = self._stage[self._which]
        v = stage.v
        n0 = len(idx)
        s_ids = replay.s[idx]
        nxt = replay.nxt[idx]
        # rows: [next 0:n0 | current n0:2n0] (compacted below if an overwrite drops rows)
        self._gather(np.where(nxt >= 0, nxt, s_ids), v, slice(0, n0), v["nmask"][:n0])
        self._gather(s_ids, v, slice(n0, 2 * n0))
        ok = multi.valid(s_ids) & ((nxt < 0) | multi.valid(nxt))
        n = n0
        if not ok.all():
            st["ring_overrun_rows"] += int((~ok).sum())
            keep = np.flatnonzero(ok)
            idx = idx[keep]
            n = len(keep)
            if n < self.min_rows:
                return None
            for k in ("local", "global", "scalars"):  # (fancy-indexed RHS: copied first)
                v[k][:n] = v[k][:n0][keep]
                v[k][n : 2 * n] = v[k][n0 : 2 * n0][keep]
            v["nmask"][:n] = v["nmask"][:n0][keep]
        v["a"][:n] = replay.a[idx]
        v["R"][:n] = replay.R[idx]
        v["disc"][:n] = replay.disc[idx]
        n_lab = 0
        if labels is not None and labels.size >= LABEL_ANCHOR_MIN_ROWS:
            n_lab = int(round(self.d_rows * args.label_frac))
        base = 2 * n
        n_demo = 0
        if demos is not None:
            n_demo = self.d_rows - n_lab
            di = rng.integers(0, demos.size, size=n_demo)  # == DemoStream.sample
            buf = demos.buf
            v["local"][base : base + n_demo] = buf["local"][di]
            v["global"][base : base + n_demo] = buf["global"][di]
            v["scalars"][base : base + n_demo] = buf["scalars"][di]
            v["a_v8"][:n_demo] = buf["a_v8"][di]
            v["mask_resolved"][:n_demo] = buf["mask_resolved"][di]
            v["q_teacher"][:n_demo] = buf["q_teacher"][di]
        if n_lab:
            li = labels.sample(rng, n_lab, multi)
            lids = labels.ids[li]
            m0 = len(lids)
            lo = base + n_demo
            self._gather(lids, v, slice(lo, lo + m0), v["mask_resolved"][n_demo : n_demo + m0])
            keep = np.flatnonzero(multi.valid(lids))  # re-check after the gather
            if len(keep) < m0:
                self._compact(v, lo, m0, keep)
                mr = v["mask_resolved"]
                mr[n_demo : n_demo + len(keep)] = mr[n_demo : n_demo + m0][keep]
            v["a_v8"][n_demo : n_demo + len(keep)] = labels.a[li][keep]
            n_lab = len(keep)
        st["label_rows"].append(n_lab)
        return {"stage": stage, "n": n, "m": n_demo + n_lab, "n_demo": n_demo}

    def _finalize(self) -> None:
        """Wait for the pending update's finite-loss flag, then apply its step (reference
        order: step, counters, target sync)."""
        p, self._pending = self._pending, None
        if p is None:
            return
        torch, st, opt = self.torch, self.st, self.opt
        if not bool(p["finite"]):
            st["halt"] = "non-finite loss"
            return
        if self.scaler is not None:
            self.scaler.step(opt)
            self.scaler.update()
        else:
            opt.step()
        st["updates"] += 1
        self._td.append(p["td"])
        if p["has_anchor"]:
            self._anc.append(p["anc"])
            st["anchor_updates"] += 1
        qm = p["qmax"]
        self._qmax = qm if self._qmax is None else torch.maximum(self._qmax, qm)
        if st["updates"] % self.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
            if self.demos is not None:
                self.demos.prefetch()
        if len(self._td) >= 4096:  # bound the device-side lists between log rows
            self.flush_stats_only()

    def flush_stats_only(self) -> None:
        torch = self.torch
        if self._td:
            self.st["td"].extend(torch.stack(self._td).cpu().tolist())
            self._td = []
        if self._anc:
            self.st["anc"].extend(torch.stack(self._anc).cpu().tolist())
            self._anc = []

    def _update_fast(self, lam: float) -> None:
        torch, st, args = self.torch, self.st, self.args
        online, target, opt, margin = self.online, self.target, self.opt, self.margin
        t0 = time.perf_counter()
        batch = self._sample_fast()  # CPU work, overlaps the device's pending update
        self._finalize()
        if batch is None or st["halt"] is not None:
            st["learn_s"] += time.perf_counter() - t0
            return
        self._which ^= 1  # double buffer: the next gather never touches an in-flight upload
        n, m, n_demo = batch["n"], batch["m"], batch["n_demo"]
        dv = batch["stage"].upload(torch, self.device, self._non_blocking)
        L, G, S = dv["local"], dv["global"], dv["scalars"]
        a, R, disc = dv["a"][:n], dv["R"][:n], dv["disc"][:n]
        cur = slice(n, 2 * n + m)
        nxt = slice(0, n)
        amp = self._amp_dtype is not None
        with torch.autocast(self.device.type, dtype=self._amp_dtype, enabled=amp):
            q_all = online(L[cur], G[cur], S[cur])
        if amp:
            q_all = q_all.float()
        q, qd = q_all[:n], q_all[n:]
        qa = q.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            with torch.autocast(self.device.type, dtype=self._amp_dtype, enabled=amp):
                qn_online = online(L[nxt], G[nxt], S[nxt])
                qn_target = target(L[nxt], G[nxt], S[nxt])
            if amp:
                qn_online, qn_target = qn_online.float(), qn_target.float()
            a_star = _masked_argmax(torch, qn_online, dv["nmask"][:n])
            q_next = qn_target.gather(1, a_star[:, None]).squeeze(1)
            y = R + disc * q_next
        td = torch.nn.functional.smooth_l1_loss(qa, y)
        db = {
            "q_teacher": dv["q_teacher"][:n_demo],
            "a_v8": dv["a_v8"][:m],
            "mask_resolved": dv["mask_resolved"][:m],
        }
        has_anchor = m > 0
        if not has_anchor:
            anc = torch.zeros((), device=q.device)
        elif args.anchor_mode == "v8margin":
            anc = anchor_loss_margin(torch, qd, db["a_v8"], db["mask_resolved"], margin)
        elif args.anchor_boost_weight == 1.0:
            anc = anchor_loss(torch, qd, db, margin)
        else:
            anc = anchor_loss_weighted(torch, qd, db, margin, args.anchor_boost_weight)
        loss = td + lam * anc
        finite = torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(opt)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        self._pending = {
            "finite": finite,
            "td": td.detach(),
            "anc": anc.detach(),
            "has_anchor": has_anchor,
            "qmax": q.detach().abs().max(),
        }
        if self.demos is not None:
            self.demos.swap_if_ready()
        st["learn_s"] += time.perf_counter() - t0


def learner_flags(args) -> dict:
    """The non-default perf-learner options (for ``m3c_flags`` / summaries)."""
    out: dict[str, Any] = {}
    if getattr(args, "learner_fast", False):
        out["learner_fast"] = True
    if getattr(args, "learner_precision", "fp32") != "fp32":
        out["learner_precision"] = args.learner_precision
    if int(getattr(args, "batch_scale", 1)) != 1:
        out["batch_scale"] = int(args.batch_scale)
    if getattr(args, "lr_scale", "none") != "none":
        out["lr_scale"] = args.lr_scale
    return out
