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
* **T4 ``update_every M``** (``--update-every M``): one same-sized learner batch per
  M x K x 128 ingested transitions.  Target synchronization and actor publication retain
  their existing K-adjusted, update-count intervals; M changes the training recipe, but not
  the definition of an update.

* **perf-compile** (``--compile-learner MODE``, needs ``fast``; 2026-10-13): the T1 update's
  forwards and losses (:func:`loss_terms`, scope ``loss``) or only the network forwards
  (scope ``forward``) under ``torch.compile(mode=MODE, dynamic=False)``, for *canonical*
  batches (all agent rows, empty or full anchor block); other batches run :func:`loss_terms`
  eagerly. Float reassociation only (not bitwise; bounded in ``tests/test_perf_compile.py``).
  See :mod:`research.redesign_m3c_20261008.compile_m3c`.

Not done (measured on MPS, 2026-10-13, see ``research/perf_learner_20261013``): one forward
for online(next) and online(current + anchor) - the backward would then run over the 384
no-grad rows too, which costs more than the saved forward; ``torch.func.vmap`` over the
stacked online/target parameters (grouped convolutions: ~4x slower on MPS);
``channels_last`` (no gain on MPS; offered to the bench only, via ``channels_last``).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from research.redesign_m3c_20261008.compile_m3c import COMPILE_MODES, LEARNER_SCOPES
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
COMPILE_SCOPES = LEARNER_SCOPES
#: perf-compile: the first VERIFY_CALLS compiled steps per batch shape are checked against
#: eager (loss terms and per-parameter gradients, relative L2 error); above the tolerance
#: the learner falls back to eager. fp32: VERIFY_TOL (reassociation is ~1e-6). fp16 / bf16:
#: compiled and eager autocast round differently (inductor keeps fused intermediates in
#: fp32), so a gradient may differ by AMP_FLOOR_FACTOR x the precision's own rounding error
#: on that batch (eager autocast vs eager fp32; measured ratio <= ~2 on CPU bf16 except
#: near-zero-floor tensors), and at least VERIFY_TOL_AMP.
VERIFY_CALLS = 3
VERIFY_TOL = 1e-3
VERIFY_TOL_AMP = 5e-2
AMP_FLOOR_FACTOR = 4.0
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
    update_every: int = 1  # T4: cadence multiplier, batches remain unchanged
    channels_last: bool = False  # bench only
    compile: str = "off"  # perf-compile (needs fast): a torch.compile mode
    compile_scope: str = "loss"  # perf-compile: "loss" (forwards + losses) or "forward"

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if self.lr_scale not in LR_SCALES:
            raise ValueError(f"lr_scale must be one of {LR_SCALES}")
        if self.precision != "fp32" and not self.fast:
            raise ValueError("--learner-precision needs --learner-fast")
        if int(self.batch_scale) < 1:
            raise ValueError("--batch-scale must be >= 1")
        if int(self.update_every) < 1:
            raise ValueError("--update-every must be >= 1")
        if self.lr_scale != "none" and self.batch_scale == 1:
            raise ValueError("--lr-scale needs --batch-scale > 1")
        if self.compile not in COMPILE_MODES:
            raise ValueError(f"--compile-learner must be one of {COMPILE_MODES}")
        if self.compile_scope not in COMPILE_SCOPES:
            raise ValueError(f"--compile-learner-scope must be one of {COMPILE_SCOPES}")
        if self.compile != "off" and not self.fast:
            raise ValueError("--compile-learner needs --learner-fast")

    @classmethod
    def from_args(cls, args) -> "LearnerOpts":
        return cls(
            fast=bool(getattr(args, "learner_fast", False)),
            precision=getattr(args, "learner_precision", "fp32"),
            batch_scale=int(getattr(args, "batch_scale", 1)),
            lr_scale=getattr(args, "lr_scale", "none"),
            update_every=int(getattr(args, "update_every", 1)),
            compile=getattr(args, "compile_learner", "off"),
            compile_scope=getattr(args, "compile_learner_scope", "loss"),
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


def return_regression_loss(
    torch,
    online,
    target,
    rows: dict,
    *,
    gamma: float,
    delta: float,
    mode: str = "level",
    contrast_epsilon: float = 0.0,
    details: bool = False,
):
    """Weighted return loss for recorded branch returns.

    The terminal bootstrap is detached Double DQN, using precisely the learner's
    empty-mask fallback through :func:`_masked_argmax`.
    """
    if mode not in ("level", "contrast") or contrast_epsilon < 0.0:
        raise ValueError(f"unknown return regression mode: {mode}")
    q0 = online(rows["local"], rows["global"], rows["scalars"])
    qa = q0.gather(1, rows["action"].long()[:, None]).squeeze(1)
    with torch.no_grad():
        q_online = online(rows["terminal_local"], rows["terminal_global"], rows["terminal_scalars"])
        a_star = _masked_argmax(torch, q_online, rows["terminal_mask"])
        q_terminal = (
            target(rows["terminal_local"], rows["terminal_global"], rows["terminal_scalars"])
            .gather(1, a_star[:, None])
            .squeeze(1)
        )
        y = rows["realised"].to(dtype=qa.dtype)
        bootstrap = (gamma ** rows["T"].long()) * q_terminal
        y = y + torch.where(rows["died"].bool(), torch.zeros_like(bootstrap), bootstrap)
    if mode == "level":
        per_row = torch.nn.functional.huber_loss(qa, y, delta=delta, reduction="none")
        loss = (per_row * rows["weight"]).mean()
        telemetry = {
            "contrast_rows": 0,
            "contrast_active_fraction": 0.0,
            "mean_abs_target_contrast": 0.0,
            "mean_abs_contrast_error": 0.0,
        }
    else:
        group_sizes = rows.get("group_sizes")
        selected = rows.get("selected")
        if group_sizes is None or selected is None:
            raise ValueError("contrast return regression needs complete source-state branch groups")
        group_sizes = [int(size) for size in group_sizes]
        if (
            not group_sizes
            or min(group_sizes) < 1
            or max(group_sizes) > 6
            or sum(group_sizes) != len(y)
            or len(selected) != len(y)
        ):
            raise ValueError("invalid return-bank source-state branch groups")
        # Each sampled source state's branches are contiguous.  Padded target rows avoid
        # one MPS synchronization per source group while selecting dynamic a_ref.
        starts = np.cumsum([0, *group_sizes[:-1]], dtype=np.int64)
        width = max(group_sizes)
        padded = np.zeros((len(group_sizes), width), dtype=np.int64)
        valid = np.zeros((len(group_sizes), width), dtype=bool)
        for row, (start, size) in enumerate(zip(starts, group_sizes)):
            padded[row, :size] = np.arange(start, start + size)
            valid[row, :size] = True
        padded_t = torch.as_tensor(padded, device=y.device, dtype=torch.long)
        valid_t = torch.as_tensor(valid, device=y.device)
        ref_indices = padded_t.gather(
            1, y[padded_t].masked_fill(~valid_t, -torch.inf).argmax(dim=1, keepdim=True)
        ).squeeze(1)
        row_refs = torch.repeat_interleave(
            ref_indices, torch.as_tensor(group_sizes, device=y.device)
        )
        non_ref = torch.arange(len(y), device=y.device) != row_refs
        eligible = non_ref & selected.bool()
        target_contrast = y - y[row_refs]
        contrast_error = (
            qa - q0.gather(1, rows["action"].long()[:, None]).squeeze(1)[row_refs].detach()
        ) - target_contrast
        excess = (contrast_error.abs() - contrast_epsilon).clamp_min(0.0)
        per_row = torch.nn.functional.huber_loss(
            excess, torch.zeros_like(excess), delta=delta, reduction="none"
        )
        per_row = per_row * eligible.to(dtype=per_row.dtype)
        loss = per_row.sum() / eligible.sum().clamp_min(1)
        active = eligible & (excess > 0)
        telemetry = {
            "contrast_rows": int(active.sum().detach().cpu()),
            "contrast_active_fraction": float(
                (active.sum() / eligible.sum().clamp_min(1)).detach().cpu()
            ),
            "mean_abs_target_contrast": (
                float(target_contrast.detach().abs()[eligible].mean().cpu())
                if eligible.any()
                else 0.0
            ),
            "mean_abs_contrast_error": (
                float(contrast_error.detach().abs()[eligible].mean().cpu())
                if eligible.any()
                else 0.0
            ),
        }
    if details:
        return loss, y, telemetry
    return loss, y


def _bank_side_agreement(q, surviving, action, counter):
    """Existing preferred-set agreement plus T4a's two raw-six-action side metrics."""
    prediction = q.argmax(dim=1)
    counter = counter.bool()
    decisive = ~counter
    decisive_ok = surviving.gather(1, prediction[:, None]).squeeze(1)
    counter_ok = prediction == action
    overall = decisive_ok.float().mean()
    return (
        overall,
        decisive_ok[decisive].float().mean() if decisive.any() else None,
        counter_ok[counter].float().mean() if counter.any() else None,
    )


def _bank_family_agreement(q, surviving, action, family):
    """Agreement telemetry for T4c's decisive, in-window, and outside row families."""
    prediction = q.argmax(dim=1)
    family = family.long()
    decisive_ok = surviving.gather(1, prediction[:, None]).squeeze(1)
    action_ok = prediction == action
    values = []
    for code in (0, 1, 2):
        selected = family == code
        if not selected.any():
            values.append(None)
        elif code == 0:
            values.append(decisive_ok[selected].float().mean())
        else:
            values.append(action_ok[selected].float().mean())
    return tuple(values)


def _bank_anchor_terms(
    torch,
    qd,
    db,
    n_bank: int,
    margin: float,
    scale: float,
    counter_mode: str = "inert",
    row_family=None,
    outside_margin_scale: float = 1.0,
    primary_rows: int | None = None,
):
    """Combine regular anchors with T3 or optional two-sided bank anchors."""
    from research.t3_rollout_labels_20261015.bank import surviving_margin

    regular = len(qd) - n_bank
    qb = qd[regular:]
    if row_family is not None:
        if counter_mode != "single-action":
            raise ValueError("outside label-bank rows need single-action counter mode")
        if primary_rows is None or primary_rows <= 0:
            raise ValueError("outside label-bank rows need a positive primary row count")
        family = row_family.long()
        decisive, inwindow, outside = family == 0, family == 1, family == 2
        terms = []
        if decisive.any():
            survive = db["surviving"][decisive].bool()
            qd_ = qb[decisive]
            per_row = (qd_ + (~survive).to(qd_.dtype) * (margin * scale)).max(1).values
            per_row -= torch.where(survive, qd_, torch.full_like(qd_, -float("inf"))).max(1).values
            terms.append((per_row * db["weight"][decisive]).sum())
        for family_mask, family_scale in ((inwindow, scale), (outside, outside_margin_scale)):
            if family_mask.any():
                qf, action = qb[family_mask], db["action"][family_mask]
                onehot = torch.nn.functional.one_hot(action.long(), 6).bool()
                per_row = (qf + (~onehot).to(qf.dtype) * (margin * family_scale)).max(1).values
                per_row -= qf.gather(1, action[:, None]).squeeze(1)
                terms.append((per_row * db["weight"][family_mask]).sum())
        bank_loss = sum(terms) / primary_rows
        agree = db["surviving"].gather(1, qb.argmax(dim=1)[:, None]).float().mean()
    elif counter_mode == "inert":
        bank_loss = surviving_margin(torch, qb, db["surviving"], db["weight"], margin * scale)
        agree = db["surviving"].gather(1, qb.argmax(dim=1)[:, None]).float().mean()
    elif counter_mode == "single-action":
        counter = db["counter"].bool()
        decisive = ~counter
        pieces, weights = [], []
        if decisive.any():
            pieces.append(
                surviving_margin(
                    torch,
                    qb[decisive],
                    db["surviving"][decisive],
                    db["weight"][decisive],
                    margin * scale,
                )
            )
            weights.append(decisive.sum())
        if counter.any():
            # Counter rows retain B8's original chosen action as the sole expert
            # action.  All six actions compete, irrespective of advisory legality.
            qc, action = qb[counter], db["action"][counter]
            onehot = torch.nn.functional.one_hot(action.long(), 6).bool()
            per_row = (qc + (~onehot).to(qc.dtype) * (margin * scale)).max(1).values
            per_row = per_row - qc.gather(1, action[:, None]).squeeze(1)
            pieces.append((per_row * db["weight"][counter]).mean())
            weights.append(counter.sum())
        bank_loss = sum(piece * weight for piece, weight in zip(pieces, weights)) / n_bank
        agree = db["surviving"].gather(1, qb.argmax(dim=1)[:, None]).float().mean()
    else:
        raise ValueError(f"unknown label-bank counter mode: {counter_mode}")
    if regular:
        ordinary = anchor_loss_margin(
            torch, qd[:regular], db["a_v8"][:regular], db["mask_resolved"][:regular], margin
        )
        if row_family is not None:
            anchor = (regular * ordinary + primary_rows * bank_loss) / (regular + primary_rows)
        else:
            anchor = (regular * ordinary + n_bank * bank_loss) / len(qd)
    else:
        anchor = bank_loss
    return anchor, bank_loss, agree


def loss_terms(
    torch,
    online,
    target,
    L,
    G,
    S,
    a,
    R,
    disc,
    nmask,
    q_teacher,
    a_v8,
    mask_resolved,
    *,
    n: int,
    m: int,
    n_demo: int,
    amp_dtype,
    dev_type: str,
    anchor_mode: str,
    boost_weight: float,
    margin: float,
    td_delta: float = 1.0,
    n_bank: int = 0,
    bank_surviving=None,
    bank_weight=None,
    bank_action=None,
    bank_counter=None,
    bank_row_family=None,
    bank_margin_scale: float = 1.0,
    bank_counter_mode: str = "inert",
    bank_outside_margin_scale: float = 1.0,
    bank_primary_rows: int | None = None,
    bank_metrics=None,
):
    """The T1 update's forwards and loss terms on one uploaded staging batch.

    ``L``/``G``/``S`` hold the obs rows ``[next (n) | current (n) | anchor (m)]``; ``a``,
    ``R``, ``disc``, ``nmask`` the agent rows (first ``n``), ``a_v8``/``mask_resolved`` the
    anchor rows (first ``m``) and ``q_teacher`` the demonstration rows (first ``n_demo``).
    ``online`` / ``target`` are the networks or (``--compile-learner-scope forward``) their
    compiled forwards. Returns ``(td, anc, qmax)``. The eager T1 update calls exactly this;
    ``--compile-learner`` (scope ``loss``) calls ``torch.compile(loss_terms)``.
    """
    a, R, disc = a[:n], R[:n], disc[:n]
    cur = slice(n, 2 * n + m)
    nxt = slice(0, n)
    amp = amp_dtype is not None
    with torch.autocast(dev_type, dtype=amp_dtype, enabled=amp):
        q_all = online(L[cur], G[cur], S[cur])
    if amp:
        q_all = q_all.float()
    q, qd = q_all[:n], q_all[n:]
    qa = q.gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():
        with torch.autocast(dev_type, dtype=amp_dtype, enabled=amp):
            qn_online = online(L[nxt], G[nxt], S[nxt])
            qn_target = target(L[nxt], G[nxt], S[nxt])
        if amp:
            qn_online, qn_target = qn_online.float(), qn_target.float()
        a_star = _masked_argmax(torch, qn_online, nmask[:n])
        q_next = qn_target.gather(1, a_star[:, None]).squeeze(1)
        y = R + disc * q_next
    td = torch.nn.functional.huber_loss(qa, y, delta=td_delta)
    db = {
        "q_teacher": q_teacher[:n_demo],
        "a_v8": a_v8[:m],
        "mask_resolved": mask_resolved[:m],
    }
    if n_bank:
        db.update(
            surviving=bank_surviving,
            weight=bank_weight,
            action=bank_action,
            counter=bank_counter,
        )
        anc, bank_loss, bank_agree = _bank_anchor_terms(
            torch,
            qd,
            db,
            n_bank,
            margin,
            bank_margin_scale,
            bank_counter_mode,
            bank_row_family,
            bank_outside_margin_scale,
            bank_primary_rows,
        )
        if bank_metrics is not None:
            if bank_row_family is None:
                _, decisive_agree, counter_agree = _bank_side_agreement(
                    qd[-n_bank:], bank_surviving, bank_action, bank_counter
                )
                outside_agree = None
            else:
                decisive_agree, counter_agree, outside_agree = _bank_family_agreement(
                    qd[-n_bank:], bank_surviving, bank_action, bank_row_family
                )
            bank_metrics.update(
                loss=bank_loss.detach(),
                agree=bank_agree.detach(),
                decisive_agree=None if decisive_agree is None else decisive_agree.detach(),
                counter_agree=None if counter_agree is None else counter_agree.detach(),
                outside_agree=None if outside_agree is None else outside_agree.detach(),
            )
    elif m == 0:
        anc = torch.zeros((), device=q.device)
    elif anchor_mode == "v8margin":
        anc = anchor_loss_margin(torch, qd, db["a_v8"], db["mask_resolved"], margin)
    elif boost_weight == 1.0:
        anc = anchor_loss(torch, qd, db, margin)
    else:
        anc = anchor_loss_weighted(torch, qd, db, margin, boost_weight)
    return td, anc, q.detach().abs().max()


def grad_check(g_c, g_e, g_32, scale: float, tol: float):
    """``(error, allowed)`` for one parameter's compiled gradient ``g_c`` against eager
    ``g_e`` (relative L2; NaN -> inf). ``g_32``: the fp32 eager gradient when ``g_e`` came
    from autocast (``g_e / scale`` vs ``g_32`` is that precision's own rounding error,
    ``scale`` the GradScaler scale): allowed ``max(tol, AMP_FLOOR_FACTOR x`` that error``)``;
    else ``tol``."""
    if g_c is None or g_e is None:
        return (0.0 if g_c is None and g_e is None else float("inf")), tol
    err = float((g_c - g_e).norm()) / (1e-12 + float(g_e.norm()))
    allowed = tol
    if g_32 is not None:
        floor = float((g_e / scale - g_32).norm()) / (1e-12 + float(g_32.norm()))
        allowed = max(tol, AMP_FLOOR_FACTOR * floor)
    return (err if err == err else float("inf")), allowed


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
        bank=None,
        outside_bank=None,
        return_bank=None,
    ) -> None:
        self.torch = torch
        self.online, self.target, self.opt = online, target, opt
        self.replay, self.multi, self.demos, self.labels = replay, multi, demos, labels
        self.rng, self.st, self.args = rng, st, args
        self.nscal, self.margin = int(nscal), margin
        self.device = torch.device(device)
        self.opts = opts or LearnerOpts()
        self.bank = bank if float(getattr(args, "label_bank_frac", 0.0)) > 0 else None
        self.outside_bank = outside_bank if self.bank is not None else None
        self.return_bank = return_bank if float(getattr(args, "return_frac", 0.0)) > 0 else None
        if float(getattr(args, "label_bank_frac", 0.0)) > 0 and self.bank is None:
            raise ValueError("--label-bank-frac needs a loaded label bank")
        if self.bank is not None:
            if args.anchor_mode != "v8margin":
                raise ValueError("--label-bank needs --anchor-mode v8margin")
            for key in (
                "bank_rows",
                "bank_margin_loss",
                "bank_agree",
                "bank_agree_decisive",
                "bank_agree_counter",
            ):
                st.setdefault(key, [])
            if self.outside_bank is not None:
                st.setdefault("bank_agree_outside", [])
            self._bank_loss = []
            self._bank_agree = []
            self._bank_agree_decisive = []
            self._bank_agree_counter = []
            self._bank_agree_outside = []
        if float(getattr(args, "return_frac", 0.0)) > 0 and self.return_bank is None:
            raise ValueError("--return-frac needs a loaded return bank")
        if self.return_bank is not None:
            st.setdefault("return_rows", [])
            st.setdefault("return_loss", [])
            st.setdefault("return_contrast_rows", [])
            st.setdefault("return_contrast_active_fraction", [])
            st.setdefault("return_mean_abs_target_contrast", [])
            st.setdefault("return_mean_abs_contrast_error", [])
            st.setdefault("return_totals", {"updates": 0, "rows": 0, "loss_sum": 0.0})
        self.return_gradient_telemetry = bool(getattr(args, "return_gradient_telemetry", False))
        self.last_return_gradient_telemetry: Optional[dict] = None
        if self.return_gradient_telemetry:
            st.setdefault("return_gradient_telemetry", [])
        k = int(self.opts.batch_scale)
        self.a_rows, self.anchor_rows, self.min_rows = agent_rows * k, anchor_rows * k, min_rows * k
        primary_rows = int(round(self.anchor_rows * args.label_bank_frac)) if self.bank else 0
        self.outside_rows = primary_rows // 2 if self.outside_bank is not None else 0
        self.d_rows = self.anchor_rows + self.outside_rows
        self.target_every = max(1, round(TARGET_EVERY / k))
        self.publish_every = max(1, round(int(args.actor_sync_updates) / k))
        self.update_every = UPDATE_EVERY * k * int(self.opts.update_every)
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
        # perf-compile state (compile "off": never touched)
        self._loss_c = None
        self._fwd_c = None
        self._cstat: Optional[dict] = None
        if self.opts.compile != "off":
            self._setup_compile()

    # ------------------------------------------------------------------ perf-compile
    def _setup_compile(self) -> None:
        from research.redesign_m3c_20261008 import compile_m3c as cm

        torch, mode = self.torch, self.opts.compile
        self._cudagraphs = cm.uses_cudagraphs(mode, self.device)
        self._cstat = {
            "mode": mode,
            "scope": self.opts.compile_scope,
            "device": str(self.device),
            "compiled_calls": 0,
            "verify_calls": 0,
            "eager_calls": 0,
            "fallback": None,
            "first_call_s": None,
            "verify_worst_rel_err": None,
            "verify_worst_ratio": None,  # max over checks of error / allowed (fails > 1)
        }
        self._verified: dict = {}
        self._gcount = cm.GraphCount()
        if self.opts.compile_scope == "loss":
            self._loss_c = cm.compile_fn(torch, loss_terms, mode)
        else:
            self._fwd_c = (
                cm.compile_fn(torch, self.online, mode),
                cm.compile_fn(torch, self.target, mode),
            )

    def _canonical(self, n: int, m: int) -> bool:
        """A batch with every agent row and an empty or full anchor block (static shapes)."""
        return n == self.a_rows and m in (0, self.d_rows)

    def _compile_fallback(self, exc: BaseException) -> None:
        from research.redesign_m3c_20261008 import compile_m3c as cm

        self._cstat["fallback"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        print(cm.fallback_message("learner", self.device, self.opts.compile, exc), file=sys.stderr)
        self._loss_c = None
        self._fwd_c = None

    def compile_status(self) -> Optional[dict]:
        """Compile counters for telemetry (``None`` when compile is off); ``graphs`` are the
        graphs THIS learner compiled (dynamo's process-wide counters: ``dynamo_process``)."""
        if self._cstat is None:
            return None
        return {**self._cstat, **self._gcount.status()}

    # ------------------------------------------------------------------ public
    def update(self, lam: float) -> None:
        if self.opts.fast:
            self._update_fast(lam)
        else:
            self._update_reference(lam)

    def _return_gradient_measurement(
        self, td, ret, return_weight: float, *, n: int, m: int, n_return: int, qmax
    ) -> dict:
        """Measure same-update, pre-clip gradients on parameters shared by both terms."""
        torch = self.torch
        params = tuple(self.online.parameters())
        td_grads = torch.autograd.grad(td, params, retain_graph=True, allow_unused=True)
        ret_grads = torch.autograd.grad(
            return_weight * ret, params, retain_graph=True, allow_unused=True
        )
        td_sq = torch.zeros((), device=self.device)
        ret_sq = torch.zeros((), device=self.device)
        for td_grad, ret_grad in zip(td_grads, ret_grads):
            if td_grad is not None and ret_grad is not None:
                td_sq = td_sq + td_grad.detach().float().square().sum()
                ret_sq = ret_sq + ret_grad.detach().float().square().sum()
        td_norm, ret_norm = td_sq.sqrt(), ret_sq.sqrt()
        td_norm_value, ret_norm_value = float(td_norm.cpu()), float(ret_norm.cpu())
        return {
            "weighted_return_grad_norm": ret_norm_value,
            "td_grad_norm": td_norm_value,
            "return_to_td_grad_ratio": (ret_norm_value / td_norm_value if td_norm_value else None),
            "max_abs_q": float(qmax.detach().cpu()),
            "anchor_rows": int(m),
            "return_rows": int(n_return),
            "agent_td_rows": int(n),
        }

    def _record_return_gradient_telemetry(self, measurement: Optional[dict]) -> None:
        if measurement is not None:
            self.last_return_gradient_telemetry = measurement
            self.st["return_gradient_telemetry"].append(measurement)

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
        if self.bank is not None:
            self._flush_bank_stats()

    def _flush_bank_stats(self) -> None:
        """Move optional bank telemetry to CPU only at existing flush boundaries."""
        if self._bank_loss:
            self.st["bank_margin_loss"].extend(self.torch.stack(self._bank_loss).cpu().tolist())
            self.st["bank_agree"].extend(self.torch.stack(self._bank_agree).cpu().tolist())
            if self._bank_agree_decisive:
                self.st["bank_agree_decisive"].extend(
                    self.torch.stack(self._bank_agree_decisive).cpu().tolist()
                )
            if self._bank_agree_counter:
                self.st["bank_agree_counter"].extend(
                    self.torch.stack(self._bank_agree_counter).cpu().tolist()
                )
            if self._bank_agree_outside:
                self.st["bank_agree_outside"].extend(
                    self.torch.stack(self._bank_agree_outside).cpu().tolist()
                )
            self._bank_loss = []
            self._bank_agree = []
            self._bank_agree_decisive = []
            self._bank_agree_counter = []
            self._bank_agree_outside = []

    def _sample_bank_rows(self, n_primary: int) -> dict:
        """Draw the primary T4a rows, then append one outside row per decisive primary row."""
        rows = self.bank.sample(
            self.rng, n_primary, getattr(self.args, "label_bank_counter_ratio", None)
        )
        if self.outside_bank is None:
            return rows
        n_outside = int((rows["row_family"] == 0).sum())
        if n_outside != self.outside_rows:
            raise ValueError("outside bank sampling requires an exact 1:1 primary counter ratio")
        outside = self.outside_bank.sample(self.rng, n_outside)
        outside["counter"] = np.ones(n_outside, dtype=bool)
        outside["row_family"] = np.full(n_outside, 2, dtype=np.int8)
        for key in (
            "local",
            "global",
            "scalars",
            "mask",
            "surviving",
            "action",
            "weight",
            "counter",
            "row_family",
        ):
            rows[key] = np.concatenate([rows[key], outside[key]])
        return rows

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
        n_primary_bank = int(round(self.anchor_rows * args.label_bank_frac)) if self.bank else 0
        n_bank = n_primary_bank + self.outside_rows
        n_lab = 0
        if labels is not None and labels.size >= LABEL_ANCHOR_MIN_ROWS:
            n_lab = int(round(self.anchor_rows * args.label_frac))
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
            d = demos.sample(rng, self.anchor_rows - n_lab - n_primary_bank)
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
        if n_bank:
            bank_rows = self._sample_bank_rows(n_primary_bank)
            for key in ("local", "global", "scalars"):
                d[key] = np.concatenate([d[key], bank_rows[key]])
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
        td_delta = float(getattr(args, "td_huber_delta", 1.0))
        return_delta = getattr(args, "return_huber_delta", None)
        return_delta = td_delta if return_delta is None else float(return_delta)
        td = torch.nn.functional.huber_loss(qa, y, delta=td_delta)
        n_return = int(round(self.anchor_rows * float(getattr(args, "return_frac", 0.0))))
        if self.return_bank is not None:
            n_return = max(1, n_return)
        ret = None
        if n_return:
            return_rows = self.return_bank.sample(
                rng,
                n_return,
                kappa=float(args.mass_kappa),
                grouped=getattr(args, "return_mode", "level") == "contrast",
            )
            n_return = int(return_rows["selected"].sum()) if "selected" in return_rows else n_return
            return_rows = {
                key: (value if key == "group_sizes" else to_dev(value))
                for key, value in return_rows.items()
            }
            ret, _, return_telemetry = return_regression_loss(
                torch,
                online,
                target,
                return_rows,
                gamma=self.return_bank.gamma,
                delta=return_delta,
                mode=getattr(args, "return_mode", "level"),
                contrast_epsilon=float(getattr(args, "return_contrast_epsilon", 0.0)),
                details=True,
            )
            if getattr(args, "return_mode", "level") == "contrast":
                st.setdefault(
                    "return_initial_mean_abs_contrast_error",
                    return_telemetry["mean_abs_contrast_error"],
                )
        db = {k: to_dev(d[k]) for k in ("q_teacher", "a_v8", "mask_resolved")}
        has_anchor = qd.shape[0] > 0
        if n_bank:
            db.update(
                surviving=to_dev(bank_rows["surviving"]),
                weight=to_dev(bank_rows["weight"]),
                action=to_dev(bank_rows["action"]),
                counter=to_dev(bank_rows["counter"]),
                row_family=to_dev(bank_rows["row_family"]),
            )
            anc, bank_loss, bank_agree = _bank_anchor_terms(
                torch,
                qd,
                db,
                n_bank,
                margin,
                args.label_bank_margin_scale,
                getattr(args, "label_bank_counter_mode", "inert"),
                db.get("row_family") if self.outside_bank is not None else None,
                getattr(args, "label_bank_outside_margin_scale", 1.0),
                n_primary_bank or None,
            )
            if self.outside_bank is None:
                _, decisive_agree, counter_agree = _bank_side_agreement(
                    qd[-n_bank:], db["surviving"], db["action"], db["counter"]
                )
                outside_agree = None
            else:
                decisive_agree, counter_agree, outside_agree = _bank_family_agreement(
                    qd[-n_bank:], db["surviving"], db["action"], db["row_family"]
                )
        elif not has_anchor:  # bv8 before enough on-policy labels: TD only
            anc = torch.zeros((), device=q.device)
        elif args.anchor_mode == "v8margin":
            anc = anchor_loss_margin(torch, qd, db["a_v8"], db["mask_resolved"], margin)
        elif args.anchor_boost_weight == 1.0:
            anc = anchor_loss(torch, qd, db, margin)
        else:
            anc = anchor_loss_weighted(torch, qd, db, margin, args.anchor_boost_weight)
        loss = td + lam * anc
        if ret is not None:
            loss = loss + float(getattr(args, "return_weight", 0.0)) * ret
        if not torch.isfinite(loss):
            st["halt"] = "non-finite loss"
            return
        gradient_telemetry = None
        if self.return_gradient_telemetry and ret is not None:
            gradient_telemetry = self._return_gradient_measurement(
                td,
                ret,
                float(getattr(args, "return_weight", 0.0)),
                n=n_agent,
                m=len(d["local"]),
                n_return=n_return,
                qmax=q.detach().abs().max(),
            )
            gradient_telemetry.update(
                return_contrast_active_fraction=return_telemetry["contrast_active_fraction"],
                return_mean_abs_contrast_error=return_telemetry["mean_abs_contrast_error"],
                return_contrast_rows=return_telemetry["contrast_rows"],
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        self._record_return_gradient_telemetry(gradient_telemetry)
        if self.bank is not None:
            st["bank_rows"].append(n_bank)
            st["bank_margin_loss"].append(float(bank_loss.detach()) if n_bank else 0.0)
            st["bank_agree"].append(float(bank_agree.detach()) if n_bank else 0.0)
            if n_bank and decisive_agree is not None:
                st["bank_agree_decisive"].append(float(decisive_agree.detach()))
            if n_bank and counter_agree is not None:
                st["bank_agree_counter"].append(float(counter_agree.detach()))
            if n_bank and outside_agree is not None:
                st["bank_agree_outside"].append(float(outside_agree.detach()))
        if n_return:
            st["return_rows"].append(n_return)
            st["return_loss"].append(float(ret.detach()))
            st["return_contrast_rows"].append(return_telemetry["contrast_rows"])
            st["return_contrast_active_fraction"].append(
                return_telemetry["contrast_active_fraction"]
            )
            st["return_mean_abs_target_contrast"].append(
                return_telemetry["mean_abs_target_contrast"]
            )
            st["return_mean_abs_contrast_error"].append(return_telemetry["mean_abs_contrast_error"])
            st["return_totals"]["updates"] += 1
            st["return_totals"]["rows"] += n_return
            st["return_totals"]["loss_sum"] += float(ret.detach())
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
            if ring.pack is None:
                L[sel] = arr["local"][pos]
                G[sel] = arr["global"][pos]
            else:  # --obs-pack: unpacked straight into the staging rows (byte-identical)
                from research.redesign_m3c_20261008.obs_pack import unpack_into

                unpack_into(arr["packed"], pos, L, G, sel, ring.pack)
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
        n_primary_bank = int(round(self.anchor_rows * args.label_bank_frac)) if self.bank else 0
        n_bank = n_primary_bank + self.outside_rows
        n_lab = 0
        if labels is not None and labels.size >= LABEL_ANCHOR_MIN_ROWS:
            n_lab = int(round(self.anchor_rows * args.label_frac))
        base = 2 * n
        n_demo = 0
        if demos is not None:
            n_demo = self.anchor_rows - n_lab - n_primary_bank
            di = rng.integers(0, demos.size, size=n_demo)  # == DemoStream.sample
            demos.gather_into(di, v, base)
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
        result = {"stage": stage, "n": n, "m": n_demo + n_lab, "n_demo": n_demo}
        if n_bank:
            bank_rows = self._sample_bank_rows(n_primary_bank)
            start = base + n_demo + n_lab
            for key in ("local", "global", "scalars"):
                v[key][start : start + n_bank] = bank_rows[key]
            result.update(
                m=result["m"] + n_bank,
                n_bank=n_bank,
                n_primary_bank=n_primary_bank,
                bank=bank_rows,
            )
        return result

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
        if "bank_rows" in p:
            st["bank_rows"].append(p["bank_rows"])
            self._bank_loss.append(p["bank_loss"])
            self._bank_agree.append(p["bank_agree"])
            if p.get("bank_agree_decisive") is not None:
                self._bank_agree_decisive.append(p["bank_agree_decisive"])
            if p.get("bank_agree_counter") is not None:
                self._bank_agree_counter.append(p["bank_agree_counter"])
            if p.get("bank_agree_outside") is not None:
                self._bank_agree_outside.append(p["bank_agree_outside"])
        if "return_rows" in p:
            st["return_rows"].append(p["return_rows"])
            loss = float(p["return_loss"].cpu())
            st["return_loss"].append(loss)
            st["return_contrast_rows"].append(p["return_contrast_rows"])
            st["return_contrast_active_fraction"].append(p["return_contrast_active_fraction"])
            st["return_mean_abs_target_contrast"].append(p["return_mean_abs_target_contrast"])
            st["return_mean_abs_contrast_error"].append(p["return_mean_abs_contrast_error"])
            st["return_totals"]["updates"] += 1
            st["return_totals"]["rows"] += p["return_rows"]
            st["return_totals"]["loss_sum"] += loss
        self._record_return_gradient_telemetry(p.get("return_gradient_telemetry"))
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
        if self.bank is not None:
            self._flush_bank_stats()

    def _backward(self, td, anc, lam: float, ret=None, return_weight: float = 0.0):
        """Backpropagate TD, anchor, and an optional weighted return-regression loss.

        The default branch retains the former ``td + lam * anc`` expression exactly so
        return-loss support cannot perturb inactive recipes. Returns the device-side
        finite-loss flag.
        """
        loss = td + lam * anc
        if ret is not None:
            loss = loss + return_weight * ret
        finite = self.torch.isfinite(loss)
        self.opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return finite

    def _compiled_step(self, tensors, kw: dict, lam: float):
        """The compiled forward/loss + backward for a canonical batch, else ``None`` (the
        caller then runs the eager update). A compile / run failure falls back to eager for
        the rest of the run (the failing batch is recomputed eagerly by the caller)."""
        torch, cs = self.torch, self._cstat
        if (self._loss_c is None and self._fwd_c is None) or not self._canonical(kw["n"], kw["m"]):
            cs["eager_calls"] += 1
            return None
        t0 = time.perf_counter() if cs["first_call_s"] is None else None
        try:
            with self._gcount:
                if self._cudagraphs:
                    torch.compiler.cudagraph_mark_step_begin()
                if self._loss_c is not None:
                    td, anc, qmax = self._loss_c(torch, self.online, self.target, *tensors, **kw)
                else:
                    td, anc, qmax = loss_terms(torch, *self._fwd_c, *tensors, **kw)
                finite = self._backward(td, anc, lam)
        except Exception as exc:  # noqa: BLE001
            self.opt.zero_grad(set_to_none=True)
            self._compile_fallback(exc)
            cs["eager_calls"] += 1
            return None
        if self._gcount.storm:  # this step is fine; the following ones run eagerly
            self._compile_fallback(self._gcount.storm_error())
        if t0 is not None:
            cs["first_call_s"] = round(time.perf_counter() - t0, 3)
        if self._cudagraphs:  # CUDA-graph outputs are overwritten by the next replay
            td, anc, qmax = td.detach().clone(), anc.detach().clone(), qmax.clone()
            metrics = kw.get("bank_metrics")
            if metrics:
                for name in ("loss", "agree", "decisive_agree", "counter_agree"):
                    if metrics.get(name) is not None:
                        metrics[name] = metrics[name].detach().clone()
        key = (kw["n"], kw["m"], kw["n_demo"])
        if self._verified.get(key, 0) < VERIFY_CALLS:
            return self._verify(key, tensors, kw, lam, (td, anc, qmax))
        cs["compiled_calls"] += 1
        return td, anc, qmax, finite

    def _verify(self, key, tensors, kw: dict, lam: float, compiled_out):
        """Check one compiled step against eager on the same batch: the loss terms and every
        parameter gradient (relative L2 error per tensor; fp32 <= ``VERIFY_TOL``; under
        autocast <= ``max(VERIFY_TOL_AMP, AMP_FLOOR_FACTOR x`` the eager autocast-vs-fp32
        error of that tensor``)``, which costs one extra fp32 forward/backward per checked
        step). The step then uses the EAGER result; a mismatch falls back to eager for the
        rest of the run (torch 2.9.1's MPS inductor, for one, returns wrong conv-bias
        gradients here)."""
        torch, cs = self.torch, self._cstat
        named = list(self.online.named_parameters())
        g_c = [None if p.grad is None else p.grad.detach().float().clone() for _, p in named]
        o_c = [x.detach().float().reshape(()) for x in compiled_out]
        amp = self._amp_dtype is not None
        g_32, scale = None, 1.0
        if amp:  # the precision's own rounding error: fp32 eager gradients, same batch
            td32, anc32, _ = loss_terms(
                torch, self.online, self.target, *tensors, **{**kw, "amp_dtype": None}
            )
            g_32 = torch.autograd.grad(td32 + lam * anc32, [p for _, p in named], allow_unused=True)
            scale = float(self.scaler.get_scale()) if self.scaler is not None else 1.0
        td, anc, qmax = loss_terms(torch, self.online, self.target, *tensors, **kw)
        finite = self._backward(td, anc, lam)  # zeroes the compiled gradients first
        cs["verify_calls"] += 1
        g_e = [None if p.grad is None else p.grad.detach().float() for _, p in named]
        if not all(g is None or bool(torch.isfinite(g).all()) for g in g_e):
            return td, anc, qmax, finite  # fp16 overflow / bad batch: verify a later step
        tol = VERIFY_TOL_AMP if amp else VERIFY_TOL
        checks = []  # (error, allowed, where)
        for name, x, y in zip(("td", "anchor", "qmax"), o_c, (td, anc, qmax)):
            err = float((x - y.detach().float()).abs()) / (1e-6 + float(y.detach().abs()))
            checks.append((err if err == err else float("inf"), tol, name))  # NaN -> inf
        for i, ((name, _), x, y) in enumerate(zip(named, g_c, g_e)):
            ref = None if g_32 is None or g_32[i] is None else g_32[i].detach().float()
            err, allowed = grad_check(x, y, ref, scale, tol)
            checks.append((err, allowed, f"grad {name}"))
        err, allowed, where = max(checks, key=lambda c: c[0] / c[1])
        cs["verify_worst_rel_err"] = max(
            cs.get("verify_worst_rel_err") or 0.0, max(c[0] for c in checks)
        )
        cs["verify_worst_ratio"] = max(cs.get("verify_worst_ratio") or 0.0, err / allowed)
        if not err <= allowed:
            self._compile_fallback(
                RuntimeError(
                    f"compiled step disagrees with eager: {where} relative error {err:.3g} "
                    f"> allowed {allowed:.3g} (shape key {key})"
                )
            )
        else:
            self._verified[key] = self._verified.get(key, 0) + 1
        return td, anc, qmax, finite

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
        tensors = (
            dv["local"],
            dv["global"],
            dv["scalars"],
            dv["a"],
            dv["R"],
            dv["disc"],
            dv["nmask"],
            dv["q_teacher"],
            dv["a_v8"],
            dv["mask_resolved"],
        )
        kw = dict(
            n=n,
            m=m,
            n_demo=n_demo,
            amp_dtype=self._amp_dtype,
            dev_type=self.device.type,
            anchor_mode=args.anchor_mode,
            boost_weight=float(args.anchor_boost_weight),
            margin=margin,
            td_delta=float(getattr(args, "td_huber_delta", 1.0)),
        )
        bank_metrics = {}
        n_bank = batch.get("n_bank", 0)
        if n_bank:
            kw.update(
                n_bank=n_bank,
                bank_surviving=torch.from_numpy(batch["bank"]["surviving"]).to(self.device),
                bank_weight=torch.from_numpy(batch["bank"]["weight"]).to(self.device),
                bank_action=torch.from_numpy(batch["bank"]["action"]).to(self.device),
                bank_counter=torch.from_numpy(batch["bank"]["counter"]).to(self.device),
                bank_margin_scale=float(args.label_bank_margin_scale),
                bank_counter_mode=getattr(args, "label_bank_counter_mode", "inert"),
                bank_metrics=bank_metrics,
            )
            if self.outside_bank is not None:
                kw.update(
                    bank_row_family=torch.from_numpy(batch["bank"]["row_family"]).to(self.device),
                    bank_outside_margin_scale=float(
                        getattr(args, "label_bank_outside_margin_scale", 1.0)
                    ),
                    bank_primary_rows=batch["n_primary_bank"],
                )
        has_anchor = m > 0
        n_return = int(round(self.anchor_rows * float(getattr(args, "return_frac", 0.0))))
        return_rows = None
        if self.return_bank is not None:
            n_return = max(1, n_return)
            sampled = self.return_bank.sample(
                self.rng,
                n_return,
                kappa=float(args.mass_kappa),
                grouped=getattr(args, "return_mode", "level") == "contrast",
            )
            n_return = int(sampled["selected"].sum()) if "selected" in sampled else n_return
            return_rows = {
                key: (
                    value
                    if key == "group_sizes"
                    else torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
                )
                for key, value in sampled.items()
            }
        out = None
        if self._cstat is not None and return_rows is None:
            out = self._compiled_step(tensors, kw, lam)
        if out is None:  # the T1 update (compile off, a non-canonical batch, or a fallback)
            td, anc, qmax = loss_terms(torch, online, target, *tensors, **kw)
            ret = None
            if return_rows is not None:
                return_delta = getattr(args, "return_huber_delta", None)
                return_delta = kw["td_delta"] if return_delta is None else float(return_delta)
                ret, _, return_telemetry = return_regression_loss(
                    torch,
                    online,
                    target,
                    return_rows,
                    gamma=self.return_bank.gamma,
                    delta=return_delta,
                    mode=getattr(args, "return_mode", "level"),
                    contrast_epsilon=float(getattr(args, "return_contrast_epsilon", 0.0)),
                    details=True,
                )
                if getattr(args, "return_mode", "level") == "contrast":
                    st.setdefault(
                        "return_initial_mean_abs_contrast_error",
                        return_telemetry["mean_abs_contrast_error"],
                    )
            gradient_telemetry = None
            if self.return_gradient_telemetry and ret is not None:
                gradient_telemetry = self._return_gradient_measurement(
                    td,
                    ret,
                    float(getattr(args, "return_weight", 0.0)),
                    n=n,
                    m=m,
                    n_return=n_return,
                    qmax=qmax,
                )
                gradient_telemetry.update(
                    return_contrast_active_fraction=return_telemetry["contrast_active_fraction"],
                    return_mean_abs_contrast_error=return_telemetry["mean_abs_contrast_error"],
                    return_contrast_rows=return_telemetry["contrast_rows"],
                )
            finite = self._backward(td, anc, lam, ret, float(getattr(args, "return_weight", 0.0)))
        else:
            td, anc, qmax, finite = out
            ret = None
            gradient_telemetry = None
        if self.scaler is not None:
            self.scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        self._pending = {
            "finite": finite,
            "td": td.detach(),
            "anc": anc.detach(),
            "has_anchor": has_anchor,
            "qmax": qmax,
            "return_gradient_telemetry": gradient_telemetry,
        }
        if ret is not None:
            self._pending.update(
                return_rows=n_return,
                return_loss=ret.detach(),
                return_contrast_rows=return_telemetry["contrast_rows"],
                return_contrast_active_fraction=return_telemetry["contrast_active_fraction"],
                return_mean_abs_target_contrast=return_telemetry["mean_abs_target_contrast"],
                return_mean_abs_contrast_error=return_telemetry["mean_abs_contrast_error"],
            )
        if self.bank is not None:
            zero = torch.zeros((), device=self.device)
            self._pending.update(
                bank_rows=n_bank,
                bank_loss=bank_metrics.get("loss", zero),
                bank_agree=bank_metrics.get("agree", zero),
                bank_agree_decisive=bank_metrics.get("decisive_agree"),
                bank_agree_counter=bank_metrics.get("counter_agree"),
                bank_agree_outside=bank_metrics.get("outside_agree"),
            )
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
    if int(getattr(args, "update_every", 1)) != 1:
        out["update_every"] = int(args.update_every)
    if getattr(args, "compile_learner", "off") != "off":
        out["compile_learner"] = args.compile_learner
        out["compile_learner_scope"] = getattr(args, "compile_learner_scope", "loss")
    return out
