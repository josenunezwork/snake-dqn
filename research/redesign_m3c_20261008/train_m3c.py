#!/usr/bin/env python3
"""M3-C trainer: the M3 vectorized Double DQN fine-tune, plus opt-in M3-C changes.

**Defaults reproduce M3 / M3-B exactly** (``research/redesign_m3_20261008/train_m3.py``; the
equivalence is tested bitwise on checkpoint weights and summary values in
``tests/test_m3c_trainer.py``), with one deliberate exception: a run that HALTs writes its
last weights as ``aborted.pth`` instead of ``final.pth`` (selection reads finals only).
Everything new is behind a flag:

* ``--reward mass --mass-kappa K``: the metric-aligned reward. Each hero decision gains
  ``K * (1 - gamma) * Phi(L_after)`` while the hero is alive after the step (``Phi = L / 10``,
  0 on the death frame), on top of the v2 reward. Over an infinite discounted horizon this
  adds ``K * Phi`` for a snake that keeps its size, so a death costs about ``K * Phi(L)``
  (the mass it destroys) instead of the v2 reward's size-independent 3, and growth has value.
  (Under v2 alone the PBRS potential telescopes away: ``Q = -Phi(L) + U`` with ``U`` = the
  discounted death penalty and kill rewards; see the M3-C design doc, section 2.)
* ``--anchor-boost-weight W``: demonstration rows whose label ``a_v8`` is a boost action get
  row weight ``W`` in the anchor loss (``W = 1``: the M3 anchor, unchanged code path).
* Schedules: ``--gamma-hold/--gamma-anneal-end``, ``--anchor-hold/--anchor-floor-at/
  --anchor-floor``, ``--eps-end-at`` (defaults = M3).
* ``--namespace``: the training-world namespace (default M3's ``redesign-m3-train/v1``).
* ``--mixed-frac F``: a share of the worlds gets the strict *mixed* roster (M3 trains on 75%
  scripted / 25% frozen and never sees "mixed"); with ``--frozen-frac`` and
  ``--segment-frames`` this gives eval-matched worlds (default 0: the M3 world list).
* ``--actors K`` (K >= 1): K actor processes per learner (``actor_main``). Each actor runs
  its own lockstep worlds (segments ``j, j+K, ...``: the same worlds sync mode visits),
  acts with a copy of the online network refreshed every ``--actor-sync-updates`` updates,
  builds the n-step transitions itself and writes observations into its own shared-memory
  ring; the learner keeps the replay index, does one update per 128 ingested transitions
  (replay ratio 4, as M3) and back-pressures the actors through bounded queues. Not
  bit-reproducible (policy lag, scheduling); ``--actors 0`` (default) is the M3 trainer.
* ``--obs-version bv8`` (M4-B; multi-actor only): the ``ego2s-bv8`` observation (``ego2s-b``
  plus the v8 veto's 30 per-action features as scalars 30..59; ``--init`` must be an
  ``ego2s-bv8`` checkpoint, e.g. from ``research/redesign_m4b_20261012/widen_bv8.py``). The
  actor runs the compiled v8 once per frame for the hero rows (``features=True``) and reuses
  it for the shield / label action (``v8_redecide``). The demonstration shards have only 30
  scalars, so it needs ``--onpolicy-labels --label-frac 1.0 --anchor-mode v8margin``: no
  demonstration rows, the anchor term starts once 10k labels are buffered (TD only before),
  and there is no held-out agreement.
* perf-actor (2026-10-13), throughput only, no training-semantics change:
  ``--v61-backend numba`` (or env ``M3C_V61_BACKEND``) featurizes the vector61 opponent /
  hero rows with the compiled :mod:`src.simd_env.vector61_nb` (float32-identical, default
  ``python``); ``--actor-impl`` ``vectorized`` (default) / ``reference`` selects the actor's
  n-step builder and eps sampling (vectorized = transition- and RNG-identical, tested in
  ``tests/test_perf_actor.py``; it also runs the opponents while an asynchronous hero
  forward is in flight). Benchmark: ``research/perf_actor_20261013/bench_actor.py``.

* perf-learner (2026-10-13; multi-actor only; defaults = the previous update, verbatim, in
  :mod:`research.redesign_m3c_20261008.learner_m3c`): ``--learner-fast`` (T1: pipelined
  update, one staged upload, device-side stats, fused Adam - same math, bitwise on CPU with
  plain Adam), ``--learner-precision fp16|bf16`` (T2: autocast, numerics change),
  ``--batch-scale K`` / ``--lr-scale`` (T3: K x larger batches at the same replay ratio -
  CHANGES the training dynamics). Benchmark: ``research/perf_learner_20261013``.

* perf-opp (2026-10-13; multi-actor only; default ``per-group`` = unchanged): ``--opp-forward``
  (or env ``M3C_OPP_FORWARD``) ``merged-cpu`` runs the vector61 opponents (one group per
  pool checkpoint, identical ApexNetwork architecture) as ONE stacked ``baddbmm`` forward per
  frame instead of one call per checkpoint - Q bitwise equal on Apple Accelerate, actor
  messages identical (``tests/test_perf_opp.py``); ``merged-device`` runs that forward on
  ``--actor-device`` (MPS / CUDA floats differ: not same-math). Benchmark and parity check:
  ``research/perf_opp_20261013/bench_opp.py``. Measured on the Mac (noisy): merged-cpu saves
  ~0.2 ms of a ~17 ms 128-world actor frame and nothing at 256; merged-device on MPS is
  slower. Not worth enabling there; CUDA untested.

* perf-compile (2026-10-13; multi-actor only; default ``off`` = no ``torch.compile`` call):
  ``--compile-learner MODE`` (needs ``--learner-fast``; ``--compile-learner-scope
  loss|forward``) and ``--compile-actor MODE`` (the hero forward, padded to ``--worlds``
  rows), MODE in ``default|reduce-overhead|max-autotune-no-cudagraphs``. Static shapes,
  first steps checked against eager, eager fallback with a message on any failure or
  mismatch (torch 2.9.1's MPS inductor returns wrong conv-bias gradients: the learner falls
  back there). Float reassociation, not bitwise. See
  :mod:`research.redesign_m3c_20261008.compile_m3c`.

* perf-ram (2026-10-13; multi-actor only; RAM only, no training-semantics change):
  ``--obs-pack`` stores the observation planes of the actor rings and of the demonstration
  buffer in the lossless packed layout of
  :mod:`research.redesign_m3c_20261008.obs_pack` (3,064 instead of 12,203 bytes per row;
  every invariant is verified when packing, and the gathered batches are byte-identical,
  ``tests/test_obs_pack.py``). ``--demo-mmap PATH`` replaces the private demonstration
  buffer by a slot -> row index into a read-only file of every training demonstration row
  (:mod:`research.redesign_m3c_20261008.demo_store`; built once per host under a lock,
  mapped by every run: the page cache is shared); the sampled rows are identical for the
  same buffer contents, but a rotated-in shard lands before the next batch instead of when
  a loader thread finishes (a scheduling-dependent lag in the default). 12.0 GB per host
  (packed store, rounds 10-12): worth it only from ~12 runs per host. RAM model and
  measurements: ``research/perf_ram_20261013/bench_ram.py`` (per run on the pods: 16.2 GB
  default, 6.6 GB ``--obs-pack``; ``bv8`` 12.4 / 5.5 GB).

* perf-profile (2026-10-13; multi-actor only; off by default): env ``M3C_PROFILE=cprofile``
  or ``pyinstrument`` (plus ``M3C_PROFILE_DIR`` / ``M3C_PROFILE_WARM_S``) makes the learner
  and every actor profile itself and dump ``<role>.*`` at exit
  (:mod:`research.redesign_m3c_20261008.profile_m3c`; analysis and report:
  ``research/perf_profile_20261013``). The actor's messages and ring rows are unchanged by it
  (``tests/test_profile_m3c.py``).

* M6 reward levers (2026-10-14; multi-actor only; every default off = the previous trainer,
  bitwise: ``tests/test_m6_reward.py`` replays the pre-M6 actor from git). Motivation and
  design: the B8 death census (``b8-death-census-20261014``, ``m6_levers.json``).

  - ``--tight-cost c --tight-min-len Lmin`` (needs ``--obs-version bv8``): each hero decision
    with ``L >= Lmin`` that is *tight* gains ``-c * (1 - gamma) * Phi(L)`` (``L`` = the
    decision-time length, ``gamma`` = the segment's gamma, as ``mass_term``), whatever its
    outcome. *Tight* is the census's length-aware definition (:func:`tight_predicate`): the
    best advisory-safe action's v8 tail-aware landing area (v8_nb, capped at v7's
    ``area_cap``; 0 when no action is advisory-safe) is ``< L``. It is read from the
    ``features=True`` v8 result the bv8 actor already holds, on the selection (advisory) mask.
  - ``--death-mass-mult w``: a hero death adds ``-w * Phi(L_before)`` (``L_before`` = the
    length at the fatal decision) on top of the v2 reward (its ``death_value`` unchanged).
  - Telemetry when a lever is on (or ``--tight-telemetry``, bv8, which logs without any
    reward change: the control arm): ``tight_frac`` (tight share of the hero decisions with
    ``L >= Lmin``), ``tight_big_share`` (their share of all decisions), ``tight_frac_1000``
    (the census's ``L >= 1000`` metric) and ``death_len_mean`` in ``log.jsonl``; run totals in
    ``summary.json`` (``m6_totals``).

Outputs as M3 (``log.jsonl`` keys and values unchanged by default) plus, always:
``telemetry.jsonl`` (per log interval: time split, RSS, actor / queue state, held-out Q
calibration) and ``manifest.jsonl`` (one row per completed checkpoint with its sha256, for
incremental pulls). Checkpoints are written to a temporary name and renamed (atomic).
Development run; not evidence.
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import queue  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m3c_20261008.compile_m3c import COMPILE_MODES  # noqa: E402
from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

SCHEMA = "redesign-m3-train/v1"  # unchanged: the default run IS an M3 run
SCHEMA_M3C = "redesign-m3c-train/v1"
TRAIN_NAMESPACE = "redesign-m3-train/v1"
LOCAL = (7, 31, 31)
GLOBAL = (4, 37, 37)
NSCAL = 30  # ego2s-b scalars (the demonstration shards); a run's width: nscal_for(args)
OBS_VERSIONS = ("b", "bv8")
#: M4-B: scalars per observation version (bv8 = b ++ the 30 v8 feature scalars)
OBS_NSCAL = {"b": NSCAL, "bv8": 60}
LABEL_ANCHOR_MIN_ROWS = 10_000  # on-policy labels needed before the anchor term is used
PHI_DIV = 10.0  # Phi(L) = L / 10 (src.core.reward_events.phi)
N_STEP = 5
UPDATE_EVERY = 128


# ----------------------------------------------------------------------------- schedules
@dataclass(frozen=True)
class Schedules:
    gamma_start: float = 0.99
    gamma_end: float = 0.995
    gamma_hold: int = 2_000_000
    gamma_anneal_end: int = 6_000_000
    anchor_hold: int = 8_000_000
    anchor_floor_at: int = 15_000_000
    anchor_floor: float = 0.1
    eps_start: float = 0.05
    eps_end: float = 0.01
    eps_end_at: int = 5_000_000

    def gamma(self, t: int) -> float:
        if t <= self.gamma_hold:
            return self.gamma_start
        if t >= self.gamma_anneal_end:
            return self.gamma_end
        f = (t - self.gamma_hold) / (self.gamma_anneal_end - self.gamma_hold)
        return self.gamma_start + f * (self.gamma_end - self.gamma_start)

    def anchor(self, t: int) -> float:
        if t <= self.anchor_hold:
            return 1.0
        if t >= self.anchor_floor_at:
            return self.anchor_floor
        f = (t - self.anchor_hold) / (self.anchor_floor_at - self.anchor_hold)
        return math.exp(f * math.log(self.anchor_floor))

    def eps(self, t: int) -> float:
        f = min(1.0, t / self.eps_end_at)
        return self.eps_start + f * (self.eps_end - self.eps_start)


def shield_prob(t: int, full_until: int, off_at: int) -> float:
    """M4-A: probability the v8 shield's action is executed (1 until ``full_until``, linear
    to 0 at ``off_at``, 0 after)."""
    if t <= full_until:
        return 1.0
    if t >= off_at:
        return 0.0
    return 1.0 - (t - full_until) / (off_at - full_until)


def schedules_from_args(args) -> Schedules:
    return Schedules(
        gamma_start=getattr(args, "gamma_start", Schedules.gamma_start),
        gamma_end=getattr(args, "gamma_end", Schedules.gamma_end),
        gamma_hold=args.gamma_hold,
        gamma_anneal_end=args.gamma_anneal_end,
        anchor_hold=args.anchor_hold,
        anchor_floor_at=args.anchor_floor_at,
        anchor_floor=args.anchor_floor,
        eps_end_at=args.eps_end_at,
    )


# ----------------------------------------------------------------------------- reward
def nscal_for(args) -> int:
    """The run's scalar width (30 for ``--obs-version b``, 60 for ``bv8``)."""
    return OBS_NSCAL[getattr(args, "obs_version", "b")]


def mass_term(lengths: np.ndarray, alive_after: np.ndarray, gamma: float, kappa: float):
    """``kappa * (1 - gamma) * Phi(L)`` for heroes alive after the step, else 0 (float64)."""
    phi = np.asarray(lengths, dtype=np.float64) / PHI_DIV
    return np.where(np.asarray(alive_after, dtype=bool), kappa * (1.0 - gamma) * phi, 0.0)


# ----------------------------------------------------------------------------- M6 reward levers
TIGHT_MIN_LEN = 600  # --tight-min-len default
TIGHT_REPORT_LEN = 1000  # tight_frac_1000: the census's large-length bin


def tight_predicate(area: np.ndarray, adv: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """M6 *tight* per row: no advisory-safe action lands in a region of at least ``L`` cells.

    Exactly the B8 death census's length-aware test (``best_area < L``; census.py builds
    ``best_area = max(where(adv, area, 0))``):

    * ``area`` ``(N, 6)``: :func:`src.simd_env.v8_nb.v8_decide`'s tail-aware post-move landing
      area per action (``features=True``), capped at v7's ``area_cap = max(1, min(max(32,
      2L), 4096, free))`` (``free`` = cells not covered by another live snake);
    * ``adv`` ``(N, 6)``: the advisory (exactly safe) mask; the flood starts on the new head
      unchecked, so an action that collides at once does not count (no safe action: 0);
    * ``lengths`` ``(N,)``: the decision-time hero length ``L``.

    The cap does not blur ``< L``: capping at ``max(32, 2L) >= L`` or at ``4096 >= L`` keeps
    every value below ``L`` and maps every value ``>= L`` to ``>= L``, and the flood (other
    snakes are walls) cannot exceed ``free`` except by the new head cell itself. So this is
    the uncapped ``area < L`` except when ``L > 4096`` (always tight here) or the arena is
    within one cell of full. In obs terms (``ego2s-bv8`` scalar ``30 + 5a`` = ``area_g =
    min(area, cap) / cap``): when ``area_cap == 2L`` (``16 <= L <= 2048`` with ``2L`` free
    cells) *tight* iff ``max_{a safe} area_g < 0.5`` (bitwise in float32 for these L).
    """
    area = np.asarray(area, dtype=np.int64)
    adv = np.asarray(adv, dtype=bool)
    if area.shape != adv.shape or area.ndim != 2 or area.shape[1] != 6:
        raise ValueError(f"tight: area {area.shape} / mask {adv.shape} must both be (N, 6)")
    if np.any(adv & (area < 0)):
        raise ValueError("tight: a safe action's landing area was not evaluated (features=True)")
    best = np.where(adv, area, 0).max(axis=1, initial=0)
    return best < np.asarray(lengths, dtype=np.int64)


def tight_cost_term(lengths, tight, gamma: float, cost: float, min_len: int) -> np.ndarray:
    """``-cost * (1 - gamma) * Phi(L)`` on tight rows with ``L >= min_len``, else 0 (float64)."""
    L = np.asarray(lengths)
    on = np.asarray(tight, dtype=bool) & (L >= min_len)
    phi = L.astype(np.float64) / PHI_DIV
    return np.where(on, -cost * (1.0 - gamma) * phi, 0.0)


def death_mass_term(lengths_before, died, mult: float) -> np.ndarray:
    """``-mult * Phi(L_before)`` on rows that died this step, else 0 (float64)."""
    phi = np.asarray(lengths_before, dtype=np.float64) / PHI_DIV
    return np.where(np.asarray(died, dtype=bool), -mult * phi, 0.0)


@dataclass(frozen=True)
class M6Levers:
    """The M6 reward levers of a run (:meth:`from_args` is ``None`` when all are off)."""

    tight_cost: float = 0.0
    tight_min_len: int = TIGHT_MIN_LEN
    death_mass_mult: float = 0.0
    tight_telemetry: bool = False

    @classmethod
    def from_args(cls, args) -> Optional["M6Levers"]:
        lv = cls(
            tight_cost=float(getattr(args, "tight_cost", 0.0)),
            tight_min_len=int(getattr(args, "tight_min_len", TIGHT_MIN_LEN)),
            death_mass_mult=float(getattr(args, "death_mass_mult", 0.0)),
            tight_telemetry=bool(getattr(args, "tight_telemetry", False)),
        )
        if lv.tight_cost == 0.0 and lv.death_mass_mult == 0.0 and not lv.tight_telemetry:
            return None
        return lv

    @property
    def tight(self) -> bool:
        """The tight predicate is evaluated (cost or telemetry)."""
        return self.tight_cost > 0.0 or self.tight_telemetry

    def frame(self, lengths, tight, died, gamma: float):
        """One frame's hero rows -> ``(reward delta per row or None, message counters)``.

        ``lengths``: decision-time hero lengths; ``tight``: :func:`tight_predicate` per row
        (``None`` unless :attr:`tight`); ``died``: the row's hero died this step. The delta is
        ``None`` when both reward coefficients are 0 (telemetry only).
        """
        L = np.asarray(lengths, dtype=np.int64)
        died = np.asarray(died, dtype=bool)
        stats = {"m6_death_len": int(L[died].sum())}
        delta = None
        if self.tight:
            t = np.asarray(tight, dtype=bool)
            big, big1k = L >= self.tight_min_len, L >= TIGHT_REPORT_LEN
            stats.update(
                m6_big=int(big.sum()),
                m6_tight=int((t & big).sum()),
                m6_big1k=int(big1k.sum()),
                m6_tight1k=int((t & big1k).sum()),
            )
            if self.tight_cost > 0.0:
                delta = tight_cost_term(L, t, gamma, self.tight_cost, self.tight_min_len)
        if self.death_mass_mult > 0.0:
            d = death_mass_term(L, died, self.death_mass_mult)
            delta = d if delta is None else delta + d
        return delta, stats

    def empty_stats(self) -> Dict[str, int]:
        """The counters of a frame without hero rows."""
        return self.frame(np.zeros(0, np.int64), np.zeros(0, bool), np.zeros(0, bool), 0.0)[1]


def m6_log_fields(acc: Dict[str, int], decisions: int, deaths: int) -> Dict[str, Any]:
    """``log.jsonl`` fields from summed M6 counters (``tight_*`` only when tight is on)."""
    out: Dict[str, Any] = {}
    if "m6_big" in acc:
        big, big1k = acc["m6_big"], acc["m6_big1k"]
        out["tight_frac"] = acc["m6_tight"] / big if big else None
        out["tight_big_share"] = big / max(1, decisions)
        out["tight_frac_1000"] = acc["m6_tight1k"] / big1k if big1k else None
    out["death_len_mean"] = acc.get("m6_death_len", 0) / deaths if deaths else None
    return out


# ----------------------------------------------------------------------------- replay
class ObsRing:
    """uint8 observation ring (+ scalars, resolved mask) addressed by a monotone counter."""

    def __init__(self, capacity: int, nscal: int = NSCAL) -> None:
        self.cap = int(capacity)
        self.local = np.zeros((capacity,) + LOCAL, dtype=np.uint8)
        self.glob = np.zeros((capacity,) + GLOBAL, dtype=np.uint8)
        self.scal = np.zeros((capacity, nscal), dtype=np.float32)
        self.mask = np.zeros((capacity, 6), dtype=bool)
        self.count = 0

    def add(self, obs: Dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
        n = len(mask)
        ids = np.arange(self.count, self.count + n, dtype=np.int64)
        pos = ids % self.cap
        self.local[pos] = obs["local"]
        self.glob[pos] = obs["global"]
        self.scal[pos] = obs["scalars"]
        self.mask[pos] = mask
        self.count += n
        return ids

    def valid(self, ids: np.ndarray) -> np.ndarray:
        return (ids >= 0) & (ids > self.count - self.cap)

    def get(self, ids: np.ndarray):
        pos = ids % self.cap
        return self.local[pos], self.glob[pos], self.scal[pos], self.mask[pos]


class Replay:
    """n-step transitions referencing the observation ring."""

    def __init__(self, capacity: int) -> None:
        self.cap = int(capacity)
        self.s = np.zeros(capacity, dtype=np.int64)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.R = np.zeros(capacity, dtype=np.float32)
        self.disc = np.zeros(capacity, dtype=np.float32)  # gamma**k, 0 when terminal
        self.nxt = np.full(capacity, -1, dtype=np.int64)
        self.size = 0
        self.ptr = 0
        self.added = 0

    def add(self, s: int, a: int, R: float, disc: float, nxt: int) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.R[i], self.disc[i], self.nxt[i] = s, a, R, disc, nxt
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        self.added += 1

    def add_many(self, s, a, R, disc, nxt) -> None:
        """``add`` for each row in order (same slots, same stored values; vectorized)."""
        n = len(s)
        if not n:
            return
        pos = (self.ptr + np.arange(n, dtype=np.int64)) % self.cap
        keep = slice(max(0, n - self.cap), n)  # a wrap within one call: the last write wins
        self.s[pos[keep]] = np.asarray(s, dtype=np.int64)[keep]
        self.a[pos[keep]] = np.asarray(a, dtype=np.int64)[keep]
        self.R[pos[keep]] = np.asarray(R, dtype=np.float64)[keep]
        self.disc[pos[keep]] = np.asarray(disc, dtype=np.float64)[keep]
        self.nxt[pos[keep]] = np.asarray(nxt, dtype=np.int64)[keep]
        self.ptr = int((self.ptr + n) % self.cap)
        self.size = min(self.size + n, self.cap)
        self.added += n

    def sample(self, rng: np.random.Generator, n: int, ring) -> np.ndarray:
        idx = rng.integers(0, self.size, size=n * 2)
        ok = ring.valid(self.s[idx]) & ((self.nxt[idx] < 0) | ring.valid(self.nxt[idx]))
        idx = idx[ok][:n]
        return idx


# ----------------------------------------------------------------------------- multi-actor ring
def ring_fields(nscal: int = NSCAL, pack=None):
    """``(name, per-row shape, dtype)`` of a ring with ``nscal`` scalars per row; with a
    :class:`~research.redesign_m3c_20261008.obs_pack.PackLayout` (``--obs-pack``) the local
    and global planes are one ``packed`` field."""
    planes = (
        (("local", LOCAL, np.uint8), ("global", GLOBAL, np.uint8))
        if pack is None
        else (("packed", (int(pack.nbytes),), np.uint8),)
    )
    return planes + (
        ("scalars", (int(nscal),), np.float32),
        ("mask", (6,), np.bool_),
    )


RING_FIELDS = ring_fields(NSCAL)


class Block:
    """A shared byte block: POSIX shared memory (``shm:<name>``) or a memory-mapped file
    (``file:<path>``; for hosts whose /dev/shm is small, e.g. Docker's 64 MB default)."""

    def __init__(self, desc: str, buf, closer, unlinker) -> None:
        self.desc, self.buf, self._close, self._unlink = desc, buf, closer, unlinker

    @classmethod
    def create(cls, nbytes: int, directory: Optional[str] = None, tag: str = "blk") -> "Block":
        nbytes = int(nbytes)
        if directory:
            import mmap
            import tempfile

            fd, path = tempfile.mkstemp(prefix=f"m3c-{tag}-", suffix=".bin", dir=directory)
            os.ftruncate(fd, nbytes)
            mm = mmap.mmap(fd, nbytes)
            os.close(fd)
            return cls(f"file:{path}", mm, mm.close, lambda: os.unlink(path))
        from multiprocessing import shared_memory

        sm = shared_memory.SharedMemory(create=True, size=nbytes)
        return cls(f"shm:{sm.name}", sm.buf, sm.close, sm.unlink)

    @classmethod
    def attach(cls, desc: str) -> "Block":
        kind, ref = desc.split(":", 1)
        if kind == "file":
            import mmap

            fd = os.open(ref, os.O_RDWR)
            mm = mmap.mmap(fd, os.fstat(fd).st_size)
            os.close(fd)
            return cls(desc, mm, mm.close, lambda: None)
        from multiprocessing import shared_memory

        sm = shared_memory.SharedMemory(name=ref)
        return cls(desc, sm.buf, sm.close, lambda: None)

    def close(self) -> None:
        try:
            self._close()
        except (BufferError, OSError, ValueError):
            pass

    def unlink(self) -> None:
        if getattr(self, "_unlinked", False):
            return
        self._unlinked = True
        try:
            self._unlink()
        except (FileNotFoundError, OSError):
            pass


def ring_capacity(replay: int, actors: int, worlds: int, queue_frames: int) -> tuple[int, int]:
    """``(rows per actor ring, overwrite margin rows)`` for ``main_multi``.

    Each actor holds 1.5x its share of the replay plus the in-flight frames (ingest is
    round-robin, so actor shares stay near 1/K; an overrun only drops rows, counted)."""
    margin_rows = int(worlds) * 16
    cap = int(1.5 * replay / actors) + int(worlds) * (int(queue_frames) + 8) + margin_rows
    return cap, margin_rows


def ring_nbytes(capacity: int, nscal: int = NSCAL, pack=None) -> int:
    total = 8  # the int64 publish counter
    for _, shape, dtype in ring_fields(nscal, pack):
        total += int(capacity) * int(np.prod(shape)) * np.dtype(dtype).itemsize
    return total


class SharedObsRing:
    """One actor's observation ring in a shared block (anything with a ``.buf``).

    The actor writes rows at ``id % cap`` and then publishes ``count``; the learner only
    reads ids it has received in a transition message (so ``id < count``) that are at least
    ``margin`` rows away from being overwritten (``id >= count - cap + margin``).

    ``pack`` (``--obs-pack``): rows hold packed planes (:mod:`obs_pack`); :meth:`add` packs
    (raising before the publish if a row cannot be packed losslessly), :meth:`get` unpacks.
    """

    pack = None

    def __init__(self, shm, capacity: int, margin: int, nscal: int = NSCAL, pack=None) -> None:
        self.shm = shm
        self.cap = int(capacity)
        self.margin = int(margin)
        self.nscal = int(nscal)
        self.pack = pack
        off = 8
        self.arrays: Dict[str, np.ndarray] = {}
        self._count = np.ndarray((1,), dtype=np.int64, buffer=shm.buf, offset=0)
        for name, shape, dtype in ring_fields(self.nscal, pack):
            arr = np.ndarray((self.cap,) + tuple(shape), dtype=dtype, buffer=shm.buf, offset=off)
            self.arrays[name] = arr
            off += arr.nbytes

    @property
    def count(self) -> int:
        return int(self._count[0])

    def add(self, obs: Dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
        n = len(mask)
        c = self.count
        ids = np.arange(c, c + n, dtype=np.int64)
        pos = ids % self.cap
        if self.pack is not None:
            from research.redesign_m3c_20261008.obs_pack import pack as pack_rows

            self.arrays["packed"][pos] = pack_rows(obs["local"], obs["global"], self.pack)
        else:
            self.arrays["local"][pos] = obs["local"]
            self.arrays["global"][pos] = obs["global"]
        self.arrays["scalars"][pos] = obs["scalars"]
        self.arrays["mask"][pos] = mask
        self._count[0] = c + n  # publish after the rows are written
        return ids

    def valid(self, ids: np.ndarray) -> np.ndarray:
        return (ids >= 0) & (ids >= self.count - self.cap + self.margin)

    def get(self, ids: np.ndarray):
        pos = ids % self.cap
        a = self.arrays
        if self.pack is not None:
            from research.redesign_m3c_20261008.obs_pack import unpack

            loc, glo = unpack(a["packed"], pos, self.pack)
            return loc, glo, a["scalars"][pos], a["mask"][pos]
        return a["local"][pos], a["global"][pos], a["scalars"][pos], a["mask"][pos]


class MultiRing:
    """K shared rings behind one id space: global id = local id * K + actor."""

    def __init__(self, rings: Sequence[SharedObsRing]) -> None:
        self.rings = list(rings)
        self.k = len(self.rings)
        self.nscal = self.rings[0].nscal
        if any(r.nscal != self.nscal for r in self.rings):
            raise ValueError("rings disagree on the scalar width")

    def encode(self, actor: int, local_ids: np.ndarray) -> np.ndarray:
        local_ids = np.asarray(local_ids, dtype=np.int64)
        return np.where(local_ids < 0, -1, local_ids * self.k + actor)

    def valid(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        out = np.zeros(ids.shape, dtype=bool)
        actor, local = ids % self.k, ids // self.k
        for j, ring in enumerate(self.rings):
            sel = (actor == j) & (ids >= 0)
            out[sel] = ring.valid(local[sel])
        return out

    def get(self, ids: np.ndarray):
        ids = np.asarray(ids, dtype=np.int64)
        n = len(ids)
        out_l = np.empty((n,) + LOCAL, dtype=np.uint8)
        out_g = np.empty((n,) + GLOBAL, dtype=np.uint8)
        out_s = np.empty((n, self.nscal), dtype=np.float32)
        out_m = np.empty((n, 6), dtype=bool)
        actor, local = ids % self.k, ids // self.k
        for j, ring in enumerate(self.rings):
            sel = np.flatnonzero(actor == j)
            if len(sel):
                l_, g_, s_, m_ = ring.get(local[sel])
                out_l[sel], out_g[sel], out_s[sel], out_m[sel] = l_, g_, s_, m_
        return out_l, out_g, out_s, out_m


class NStepBuilderRef:
    """Per-world n-step windows, arithmetic identical to the M3 sync loop (tested).

    The reference (per-row deque) implementation; :class:`NStepBuilder` is the vectorized
    one, transition-for-transition identical (``tests/test_perf_actor.py``).

    ``step`` takes one frame's hero rows; it returns the completed transitions as arrays
    ``(s, a, R, disc, nxt)`` (``nxt = -1`` and ``disc = 0`` on a terminal flush).
    """

    def __init__(self, n_worlds: int, n_step: int = N_STEP) -> None:
        self.n = int(n_step)
        self.pend: List[deque] = [deque() for _ in range(n_worlds)]
        self.last_obs = np.full(n_worlds, -1, dtype=np.int64)
        self.out: List[tuple] = []

    def _flush(self, e: int, bootstrap: int, gamma_now: float) -> None:
        while self.pend[e]:
            R, g = 0.0, 1.0
            for _, _, r in self.pend[e]:
                R += g * r
                g *= gamma_now
            s0, a0, _ = self.pend[e].popleft()
            self.out.append((s0, a0, R, 0.0 if bootstrap < 0 else g, bootstrap))

    def step(self, worlds, ids, acts, reward, done, gamma_now: float) -> int:
        """Add one frame; returns the number of deaths (terminal flushes)."""
        deaths = 0
        for j, e in enumerate(worlds):
            e = int(e)
            if len(self.pend[e]) == self.n:
                R, g = 0.0, 1.0
                for _, _, r in self.pend[e]:
                    R += g * r
                    g *= gamma_now
                s0, a0, _ = self.pend[e].popleft()
                self.out.append((s0, a0, R, g, int(ids[j])))
            self.pend[e].append((int(ids[j]), int(acts[j]), float(reward[e])))
            self.last_obs[e] = ids[j]
            if done[e]:
                self._flush(e, -1, gamma_now)
                deaths += 1
        return deaths

    def end_segment(self, gamma_end: float) -> None:
        for e in range(len(self.pend)):
            if self.pend[e]:
                self.pend[e].pop()
                self._flush(e, int(self.last_obs[e]), gamma_end)

    def take(self):
        rows, self.out = self.out, []
        if not rows:
            z = np.zeros(0, dtype=np.int64)
            return z, z, np.zeros(0, np.float32), np.zeros(0, np.float32), z
        s, a, R, d, nx = zip(*rows)
        return (
            np.asarray(s, dtype=np.int64),
            np.asarray(a, dtype=np.int64),
            np.asarray(R, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
            np.asarray(nx, dtype=np.int64),
        )


class NStepBuilder:
    """Vectorized :class:`NStepBuilderRef`: the same transitions, in the same order, with
    bit-identical ``R`` / ``disc`` (the same float64 accumulation order, one float32 cast).

    Each world keeps a ring of its last ``n`` hero rows ``(obs id, action, reward)``. Output
    order is the reference's: worlds in row order; within a world the full-window
    transition first, then a terminal flush oldest-first; ``end_segment`` flushes worlds in
    ascending order.
    """

    def __init__(self, n_worlds: int, n_step: int = N_STEP) -> None:
        self.n = int(n_step)
        E = int(n_worlds)
        self.ids = np.zeros((E, self.n), dtype=np.int64)
        self.acts = np.zeros((E, self.n), dtype=np.int64)
        self.rew = np.zeros((E, self.n), dtype=np.float64)
        self.head = np.zeros(E, dtype=np.int64)  # ring slot of the oldest pending row
        self.cnt = np.zeros(E, dtype=np.int64)
        self._chunks: List[tuple] = []  # (s, a, R float64, disc float64, nxt) array chunks

    @property
    def out(self) -> List[tuple]:
        """Pending transitions as ``(s, a, R, disc, nxt)`` Python tuples (float64 R / disc),
        the reference's ``out`` list (for tests; the actor uses :meth:`take`)."""
        rows: List[tuple] = []
        for c in self._chunks:
            rows.extend(zip(*(x.tolist() for x in c)))
        return rows

    @out.setter
    def out(self, value: List[tuple]) -> None:
        if value:
            raise ValueError("NStepBuilder.out can only be cleared")
        self._chunks = []

    @property
    def last_obs(self) -> np.ndarray:
        """Newest pending obs id per world (-1 when nothing is pending)."""
        newest = self.ids[np.arange(len(self.cnt)), (self.head + self.cnt - 1) % self.n]
        return np.where(self.cnt > 0, newest, -1)

    def _returns(self, w: np.ndarray, start: np.ndarray, k: np.ndarray, gamma: float):
        """``(R, g)`` over pending entries ``start .. start+k-1`` (oldest-relative) of worlds
        ``w``, accumulated oldest first like the reference (``R += g * r; g *= gamma``)."""
        R = np.zeros(len(w), dtype=np.float64)
        g = np.ones(len(w), dtype=np.float64)
        for t in range(self.n):
            live = t < k
            if not live.any():
                break
            r = self.rew[w, (self.head[w] + start + t) % self.n]
            R = np.where(live, R + g * r, R)
            g = np.where(live, g * gamma, g)
        return R, g

    def _flush_rows(self, w: np.ndarray, k: np.ndarray, gamma: float, boot: np.ndarray):
        """Transitions for entries ``0..k-1`` of each world (row-major: world, then oldest
        first); ``boot`` < 0 means terminal (``disc = 0``)."""
        rep = np.repeat(np.arange(len(w)), k)
        i = np.arange(len(rep)) - np.repeat(np.cumsum(k) - k, k)
        ww = w[rep]
        R, g = self._returns(ww, i, k[rep] - i, gamma)
        pos = (self.head[ww] + i) % self.n
        b = boot[rep]
        return rep, i, self.ids[ww, pos], self.acts[ww, pos], R, np.where(b < 0, 0.0, g), b

    def step(self, worlds, ids, acts, reward, done, gamma_now: float) -> int:
        """Add one frame; returns the number of deaths (terminal flushes)."""
        w = np.asarray(worlds, dtype=np.int64).reshape(-1)
        m = len(w)
        if not m:
            return 0
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        acts = np.asarray(acts).astype(np.int64).reshape(-1)
        r_new = np.asarray(reward)[w].astype(np.float64)
        done_w = np.asarray(done, dtype=bool)[w]
        n = self.n
        keys_j, keys_sub, parts = [], [], []
        full = np.flatnonzero(self.cnt[w] == n)
        if len(full):
            fw = w[full]
            R, g = self._returns(fw, np.zeros(len(fw), np.int64), np.full(len(fw), n), gamma_now)
            h = self.head[fw]
            keys_j.append(full)
            keys_sub.append(np.zeros(len(full), dtype=np.int64))
            parts.append((self.ids[fw, h], self.acts[fw, h], R, g, ids[full]))
            self.head[fw] = (h + 1) % n
            self.cnt[fw] -= 1
        pos = (self.head[w] + self.cnt[w]) % n
        self.ids[w, pos] = ids
        self.acts[w, pos] = acts
        self.rew[w, pos] = r_new
        self.cnt[w] += 1
        dead = np.flatnonzero(done_w)
        if len(dead):
            dw = w[dead]
            k = self.cnt[dw].copy()
            rep, i, s0, a0, R, d, b = self._flush_rows(dw, k, gamma_now, np.full(len(dw), -1))
            keys_j.append(dead[rep])
            keys_sub.append(1 + i)
            parts.append((s0, a0, R, d, b))
            self.cnt[dw] = 0
            self.head[dw] = 0
        if parts:
            kj = np.concatenate(keys_j)
            ks = np.concatenate(keys_sub)
            order = np.lexsort((ks, kj))
            cols = [np.concatenate([p[c] for p in parts])[order] for c in range(5)]
            self._chunks.append(tuple(cols))
        return int(len(dead))

    def end_segment(self, gamma_end: float) -> None:
        w = np.flatnonzero(self.cnt > 0)
        if not len(w):
            return
        boot = self.ids[w, (self.head[w] + self.cnt[w] - 1) % self.n]  # the newest row
        self.cnt[w] -= 1  # dropped: it has no successor observation
        k = self.cnt[w].copy()
        keep = k > 0
        w, k, boot = w[keep], k[keep], boot[keep]
        if len(w):
            _, _, s0, a0, R, d, b = self._flush_rows(w, k, gamma_end, boot)
            self._chunks.append((s0, a0, R, d, b))
        self.cnt[:] = 0
        self.head[:] = 0

    def take(self):
        rows, self._chunks = self._chunks, []
        if not rows:
            z = np.zeros(0, dtype=np.int64)
            return z, z, np.zeros(0, np.float32), np.zeros(0, np.float32), z
        s, a, R, d, nx = (np.concatenate([r[c] for r in rows]) for c in range(5))
        return (
            s.astype(np.int64),
            a.astype(np.int64),
            R.astype(np.float32),
            d.astype(np.float32),
            nx.astype(np.int64),
        )


def sample_legal_ref(rng: np.random.Generator, mm: np.ndarray) -> np.ndarray:
    """A uniform legal action per row: the reference per-row ``rng.choice`` loop."""
    return np.array([rng.choice(np.flatnonzero(r)) for r in mm])


def sample_legal(rng: np.random.Generator, mm: np.ndarray) -> np.ndarray:
    """Vectorized :func:`sample_legal_ref`: one ``rng.integers(0, counts)`` call draws the
    same variates in the same order as the per-row ``choice`` calls (both are bounded
    ``int64`` draws of ``[0, count)``), so the actions and the generator state after the
    call are identical (pinned by ``tests/test_perf_actor.py``). Rows need >= 1 legal action."""
    mm = np.asarray(mm, dtype=bool)
    if not len(mm):
        return np.zeros(0, dtype=np.int64)
    k = rng.integers(0, mm.sum(axis=1, dtype=np.int64))
    return (np.cumsum(mm, axis=1) <= k[:, None]).sum(axis=1).astype(np.int64)


# ----------------------------------------------------------------------------- demonstrations
DEMO_KEYS = ("local", "global", "scalars", "q_teacher", "a_v8", "mask_resolved")


def demo_key(path: Path) -> str:
    """A demonstration shard's name relative to the data directory (``round-R/chunk-...``)."""
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def demo_fields(pack=None):
    """``(name, per-row shape, dtype)`` of a demonstration buffer row (``DEMO_KEYS`` order);
    with a :class:`~research.redesign_m3c_20261008.obs_pack.PackLayout` the planes are one
    ``packed`` field. Shared by :class:`DemoStream` and the demonstration store."""
    planes = (
        (("local", LOCAL, np.uint8), ("global", GLOBAL, np.uint8))
        if pack is None
        else (("packed", (int(pack.nbytes),), np.uint8),)
    )
    return planes + (
        ("scalars", (NSCAL,), np.float32),
        ("q_teacher", (6,), np.float32),
        ("a_v8", (), np.int64),
        ("mask_resolved", (6,), np.bool_),
    )


def demo_row_nbytes(pack=None) -> int:
    """Bytes per demonstration buffer row (12,361 raw; 3,222 with the training layout)."""
    return int(
        sum(int(np.prod(s, dtype=np.int64)) * np.dtype(d).itemsize for _, s, d in demo_fields(pack))
    )


class DemoStream:
    """Rotating buffer of M2b teacher demonstrations (hold-out worlds excluded).

    Storage modes (perf-ram; the default is the original private raw buffer). Given the same
    load / swap sequence every mode holds the same rows in the same slots and draws the same
    RNG variates, so sampled batches are byte-identical (``tests/test_obs_pack.py``):

    * ``pack`` (``--obs-pack``): the planes as packed rows (:mod:`obs_pack`), unpacked on
      gather; a prefetched shard is packed in the loader thread (~4 us per row).
    * ``store`` (``--demo-mmap``): no private rows, a slot -> row index into a read-only
      :class:`~research.redesign_m3c_20261008.demo_store.SharedDemoStore` (every training
      row, decompressed once per host, page cache shared by every run that maps it). A
      prefetch resolves the next shard's row range at once (no thread), so the new shard is
      swapped in before the next batch is drawn; the private buffer swaps when its loader
      thread finishes, a scheduling-dependent number of updates later (it is not
      reproducible run to run either; packing adds ~40 ms per shard to that thread). That
      swap timing is the only difference in what a run sees.
    """

    pack = None  # class defaults: instances made with object.__new__ (tests, benches)
    store = None

    def __init__(
        self,
        data: Path,
        rounds: List[int],
        capacity: int,
        seed: int,
        pack=None,
        store=None,
    ) -> None:
        paths: List[Path] = []
        for r in rounds:
            paths += sorted((data / f"round-{r}").glob("chunk-*-part-*.npz"))
        if not paths:
            raise SystemExit("no demonstration shards")
        rng = np.random.default_rng([seed, 991])
        self.order = [paths[i] for i in rng.permutation(len(paths))]
        self.cap = int(capacity)
        self.pack = pack
        self.store = store
        if store is not None:
            store.check(paths)
            self.buf = None
            self.index = np.zeros(capacity, dtype=np.int64)
        else:
            self.buf = {
                name: np.zeros((capacity,) + tuple(shape), dtype=dtype)
                for name, shape, dtype in demo_fields(pack)
            }
        self.size = 0
        self.write = 0
        self.next_shard = 0
        self.loaded = 0
        self.distinct = 0
        self._q: "queue.Queue[Optional[Dict[str, np.ndarray]]]" = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def load(path: Path, held_out: bool = False) -> Dict[str, np.ndarray]:
        with np.load(path) as z:
            keep = (z["world_seed"] % 10 == 0) == held_out
            return {k: z[k][keep] for k in DEMO_KEYS}

    def _load_part(self, path: Path) -> Dict[str, Any]:
        """The next shard in this buffer's storage form."""
        if self.store is not None:
            return {"range": self.store.shard_range(path)}
        part = self.load(path)
        if self.pack is not None:
            from research.redesign_m3c_20261008.obs_pack import pack as pack_rows

            part["packed"] = pack_rows(part.pop("local"), part.pop("global"), self.pack)
        return part

    def _put(self, part: Dict[str, np.ndarray]) -> None:
        if self.store is not None:
            start, n = part["range"]
            idx = (self.write + np.arange(n)) % self.cap
            self.index[idx] = start + np.arange(n, dtype=np.int64)
        elif self.pack is not None:
            n = len(part["a_v8"])
            idx = (self.write + np.arange(n)) % self.cap
            for k in self.buf:
                self.buf[k][idx] = part[k]
        else:
            n = len(part["a_v8"])
            idx = (self.write + np.arange(n)) % self.cap
            for k in DEMO_KEYS:
                self.buf[k][idx] = part[k]
        self.write = int((self.write + n) % self.cap)
        self.size = min(self.cap, self.size + n)
        self.loaded += 1
        self.distinct += n

    def fill_initial(self, shards: int) -> None:
        for _ in range(shards):
            self._put(self._load_part(self.order[self.next_shard % len(self.order)]))
            self.next_shard += 1

    def prefetch(self) -> None:
        """Start decompressing the next shard in the background."""
        if self._thread is not None and self._thread.is_alive():
            return
        path = self.order[self.next_shard % len(self.order)]
        self.next_shard += 1
        if self.store is not None:  # nothing to decompress: queued now, swapped next update
            self._q.put(self._load_part(path))
            return
        self._thread = threading.Thread(
            target=lambda: self._q.put(self._load_part(path)), daemon=True
        )
        self._thread.start()

    def swap_if_ready(self) -> bool:
        try:
            part = self._q.get_nowait()
        except queue.Empty:
            return False
        self._put(part)
        return True

    def _source(self, idx: np.ndarray):
        """``(arrays, rows)``: where buffer slots ``idx`` live (private buffer or store)."""
        if self.store is not None:
            return self.store.arrays, self.index[idx]
        return self.buf, idx

    def _planes_into(self, src, rows, out_local, out_global) -> None:
        if "packed" in src:
            from research.redesign_m3c_20261008.obs_pack import unpack_into

            pack = self.store.pack if self.store is not None else self.pack
            n = len(rows)
            unpack_into(
                src["packed"], rows, out_local, out_global, np.arange(n, dtype=np.int64), pack
            )
        else:
            out_local[...] = src["local"][rows]
            out_global[...] = src["global"][rows]

    def sample(self, rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=n)
        if self.store is None and self.pack is None:
            return {k: v[idx] for k, v in self.buf.items()}
        src, rows = self._source(idx)
        out = {
            "local": np.empty((n,) + LOCAL, dtype=np.uint8),
            "global": np.empty((n,) + GLOBAL, dtype=np.uint8),
        }
        self._planes_into(src, rows, out["local"], out["global"])
        for k in DEMO_KEYS[2:]:
            out[k] = src[k][rows]
        return out

    def gather_into(self, di: np.ndarray, v: Dict[str, np.ndarray], base: int) -> None:
        """Rows ``di`` (slots drawn as in :meth:`sample`) into the learner's staging views:
        planes / scalars at rows ``base:base+n``, labels / masks / teacher Q at ``:n``."""
        n = len(di)
        if self.store is None and self.pack is None:
            buf = self.buf
            v["local"][base : base + n] = buf["local"][di]
            v["global"][base : base + n] = buf["global"][di]
            v["scalars"][base : base + n] = buf["scalars"][di]
            v["a_v8"][:n] = buf["a_v8"][di]
            v["mask_resolved"][:n] = buf["mask_resolved"][di]
            v["q_teacher"][:n] = buf["q_teacher"][di]
            return
        src, rows = self._source(di)
        self._planes_into(src, rows, v["local"][base : base + n], v["global"][base : base + n])
        v["scalars"][base : base + n] = src["scalars"][rows]
        v["a_v8"][:n] = src["a_v8"][rows]
        v["mask_resolved"][:n] = src["mask_resolved"][rows]
        v["q_teacher"][:n] = src["q_teacher"][rows]

    def q_teacher_absmax(self) -> float:
        """``max |q_teacher|`` over the filled slots."""
        if self.store is not None:
            return float(np.abs(self.store.arrays["q_teacher"][self.index[: self.size]]).max())
        return float(np.abs(self.buf["q_teacher"][: self.size]).max())


# ----------------------------------------------------------------------------- losses
def anchor_loss(torch, q, b, margin: float):
    """advantage-Huber ×10 + KL τ 0.05 + DQfD margin (the M2b loss without raw-Q)."""
    qt = b["q_teacher"]
    adv_s = q - q.mean(dim=1, keepdim=True)
    adv_t = qt - qt.mean(dim=1, keepdim=True)
    adv = torch.nn.functional.smooth_l1_loss(adv_s, adv_t, beta=0.1)
    legal = b["mask_resolved"].clone()
    legal[~legal.any(dim=1), :3] = True
    neg = torch.full_like(q, -1e9)
    lt = torch.where(legal, qt / 0.05, neg).log_softmax(dim=1)
    ls = torch.where(legal, q / 0.05, neg).log_softmax(dim=1)
    kl = (lt.exp() * (lt - ls)).where(legal, torch.zeros_like(q)).sum(dim=1).mean()
    a = b["a_v8"].long()
    onehot = torch.nn.functional.one_hot(a, 6).bool()
    allowed = b["mask_resolved"] | onehot
    hi = torch.where(allowed, q + (~onehot).float() * margin, neg).max(dim=1).values
    marg = (hi - q.gather(1, a[:, None]).squeeze(1)).mean()
    return 10.0 * adv + kl + marg


def anchor_loss_margin(torch, q, a, mask, margin: float):
    """M4-A ``--anchor-mode v8margin``: the DQfD large-margin term toward the v8 label only.

    No advantage-Huber / KL toward the teacher's *unvetoed* Q (whose ranking conflicts with
    the vetoed label, and whose scale is the teacher's reward, not the student's)."""
    a = a.long()
    neg = torch.full_like(q, -1e9)
    onehot = torch.nn.functional.one_hot(a, 6).bool()
    allowed = mask | onehot
    hi = torch.where(allowed, q + (~onehot).float() * margin, neg).max(dim=1).values
    return (hi - q.gather(1, a[:, None]).squeeze(1)).mean()


class LabelBuffer:
    """M4-A on-policy v8 labels: global observation-ring ids + the v8 action (ring order)."""

    def __init__(self, capacity: int) -> None:
        self.cap = int(capacity)
        self.ids = np.full(capacity, -1, dtype=np.int64)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self.ptr = 0

    def add(self, ids: np.ndarray, a: np.ndarray) -> None:
        n = len(ids)
        pos = (self.ptr + np.arange(n)) % self.cap
        self.ids[pos] = ids
        self.a[pos] = a
        self.ptr = int((self.ptr + n) % self.cap)
        self.size = min(self.cap, self.size + n)

    def sample(self, rng: np.random.Generator, n: int, ring) -> np.ndarray:
        idx = rng.integers(0, self.size, size=n * 2)
        idx = idx[ring.valid(self.ids[idx])][:n]
        return idx


V8_BACKENDS = ("python", "numba")
V8_BACKEND_ENV = "M3C_V8_BACKEND"
#: Vector61Runtime featurizer backends (``src.simd_env.vector61_nb``; float32-identical).
V61_BACKENDS = ("python", "numba")
V61_BACKEND_ENV = "M3C_V61_BACKEND"
#: Actor bookkeeping implementations (``NStepBuilder`` / ``NStepBuilderRef``, eps sampling).
ACTOR_IMPLS = ("vectorized", "reference")
#: perf-opp: how the actor runs the vector61 opponents (``src.simd_env.vector61_merged``).
OPP_FORWARDS = ("per-group", "merged-cpu", "merged-device")
OPP_FORWARD_ENV = "M3C_OPP_FORWARD"


def hero_selection_mask(sim, rows: np.ndarray) -> np.ndarray:
    """The hero's selection (advisory) mask straight from the sim: equal to the vector61
    runtime's ``selection`` mask (0 mismatches over 57,600 training-world hero rows,
    tests/test_m4a_trainer.py), without featurizing the hero in Python."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    return sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)


def load_init(args, device):
    """The ``--init`` network; an ego2s-b checkpoint is widened in memory for ``bv8`` runs
    (zero new fuse columns: Q unchanged, see ``widen_bv8.py``)."""
    from src.model.ego2s_network import OBS_SPEC_B, load_ego2s_checkpoint, spec_for

    net = load_ego2s_checkpoint(args.init, "cpu")
    if getattr(args, "obs_version", "b") == "bv8" and spec_for(net) == OBS_SPEC_B:
        from research.redesign_m4b_20261012.widen_bv8 import widen_net

        net = widen_net(net)
    return net.to(device)


def make_shield(carrier: str, runtime, backend: Optional[str] = None):
    """frp3-s12's v8 veto (lambda 8) applied to externally supplied Q (the online student).

    ``backend`` (default: env ``M3C_V8_BACKEND``, else ``"python"``): ``"python"`` runs the
    live v8 hook per row (the reference); ``"numba"`` runs :func:`src.simd_env.v8_nb.v8_decide`
    on the same Q and the same runtime selection mask, decision-identical (pinned by
    ``tests/test_v8_nb.py``) and ~50-90x faster, but it does not update the live hook's
    per-episode counters (``veto_record`` / ``last_veto``), which the trainer never reads.
    """
    import torch

    from src.simd_env.vector61_policy import Vector61SimdPolicy, check_live_rule_config

    backend = backend or os.environ.get(V8_BACKEND_ENV) or "python"
    if backend not in V8_BACKENDS:
        raise ValueError(f"v8 shield backend must be one of {V8_BACKENDS}, got {backend!r}")

    class InjectedQV8(Vector61SimdPolicy):
        def __init__(self) -> None:
            super().__init__(
                carrier,
                runtime,
                veto_slots=(0,),
                forward="batched",
                veto_variant="v8",
                veto_lambda=8.0,
            )
            self.network = None  # never run: Q comes from the student
            self._q = None
            self.backend = backend

        def act(self, q: np.ndarray, masks, sim, slots) -> np.ndarray:
            if self.backend == "numba":
                return self._act_numba(q, sim, slots)
            self._q = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32))
            try:
                return self.actions(masks, sim, slots)
            finally:
                self._q = None

        def _act_numba(self, q: np.ndarray, sim, slots) -> np.ndarray:
            """``actions()`` with the compiled v8: same mask, base argmax and veto rows."""
            from src.simd_env.v8_nb import v8_decide

            if getattr(self, "_rule_config_checked", None) is not sim:
                check_live_rule_config(sim)
                self._rule_config_checked = sim
            slots = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            mask = hero_selection_mask(sim, slots)
            res = v8_decide(sim, slots, q, mask, self.veto_spec.lam)
            vetoed = np.isin(slots[:, 1], list(self.veto_slots))
            return np.where(vetoed, res.action, res.base).astype(np.int64)

        def q_values(self, states):
            if self._q is None:
                raise RuntimeError("InjectedQV8.q_values outside act()")
            return self._q

    return InjectedQV8()


def anchor_loss_weighted(torch, q, b, margin: float, boost_weight: float):
    """The same three terms with row weight ``boost_weight`` on boost-labelled rows.

    Weighted means (``sum(w * x) / sum(w)``), so ``boost_weight = 1`` equals
    :func:`anchor_loss` up to float summation order; the trainer calls :func:`anchor_loss`
    itself when the weight is 1 (bit-for-bit M3).
    """
    qt = b["q_teacher"]
    a = b["a_v8"].long()
    w = torch.where(a >= 3, torch.full_like(qt[:, 0], boost_weight), torch.ones_like(qt[:, 0]))
    wsum = w.sum()
    adv_s = q - q.mean(dim=1, keepdim=True)
    adv_t = qt - qt.mean(dim=1, keepdim=True)
    adv_rows = torch.nn.functional.smooth_l1_loss(adv_s, adv_t, beta=0.1, reduction="none")
    adv = (adv_rows.mean(dim=1) * w).sum() / wsum
    legal = b["mask_resolved"].clone()
    legal[~legal.any(dim=1), :3] = True
    neg = torch.full_like(q, -1e9)
    lt = torch.where(legal, qt / 0.05, neg).log_softmax(dim=1)
    ls = torch.where(legal, q / 0.05, neg).log_softmax(dim=1)
    kl_rows = (lt.exp() * (lt - ls)).where(legal, torch.zeros_like(q)).sum(dim=1)
    kl = (kl_rows * w).sum() / wsum
    onehot = torch.nn.functional.one_hot(a, 6).bool()
    allowed = b["mask_resolved"] | onehot
    hi = torch.where(allowed, q + (~onehot).float() * margin, neg).max(dim=1).values
    marg = ((hi - q.gather(1, a[:, None]).squeeze(1)) * w).sum() / wsum
    return 10.0 * adv + kl + marg


# ----------------------------------------------------------------------------- worlds
def build_segment(args, ctx, segment: int, gamma: float, hero_v61: bool = False):
    from src.core.world_runtime import WorldRuntimeSpec
    from src.simd_env import eval_engine as ee
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.grid_sim import GridBatchSim
    from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

    E = args.worlds
    from research.apex_safety_20260926 import dev_screen

    namespace = getattr(args, "namespace", TRAIN_NAMESPACE)
    seeds = dev_screen.screen_seeds((segment + 1) * E, namespace, f"seed{args.seed}")[segment * E :]
    base = ee._config_from_game_config(6, gamma, ctx["profile"])
    if getattr(args, "kill_scale", None) is not None:
        base = replace(base, kill_scale=float(args.kill_scale))
    if getattr(args, "death_value", None) is not None:
        base = replace(base, death_value=float(args.death_value))
    cap = WorldRuntimeSpec.source_exact(ctx["profile"]).body_storage_capacity
    sim = GridBatchSim(
        replace(base, num_envs=E, body_storage_capacity=cap),
        seeds=seeds,
        train_mode=False,
        allow_respawn=True,
    )
    n_frozen = int(round(E * args.frozen_frac))
    n_mixed = int(round(E * getattr(args, "mixed_frac", 0.0)))
    if n_frozen + n_mixed > E:
        raise ValueError("frozen + mixed fractions exceed 1")
    mix_of = ["scripted"] * (E - n_frozen - n_mixed) + ["frozen"] * n_frozen + ["mixed"] * n_mixed
    rows = {(int(r["world_seed"]), r["mix"]): r for r in dev_screen._design_rows(seeds)}
    lookup = ctx["lookup"]
    groups: Dict[Any, List[tuple]] = {}
    for e, seed in enumerate(seeds):
        for slot, member in enumerate(rows[(int(seed), mix_of[e])]["slots"], start=1):
            groups.setdefault(lookup[member["member_sha256"]], []).append((e, slot))
    v61 = getattr(args, "v61_backend", None) or os.environ.get(V61_BACKEND_ENV) or "python"
    runtime = Vector61Runtime(backend=v61)
    controlled = np.zeros((E, 6), dtype=bool)
    if hero_v61:  # M4-A shield / labels: the v8 veto reads the hero's vector61 rows
        controlled[:, 0] = True
    policies: Dict[Any, Any] = {}
    for spec, cells in groups.items():
        if spec[0] == "scripted":
            policies[spec] = FastProfileAnchorSimdPolicy(spec[1], seeds)
        else:
            policies[spec] = Vector61SimdPolicy(spec[1], runtime, forward="batched")
            for e, s in cells:
                controlled[e, s] = True
    runtime.bind(sim, controlled)
    groups_arr = {k: np.array(v, dtype=np.int64) for k, v in groups.items()}
    return sim, runtime, policies, groups_arr


# ----------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_checkpoint_atomic(path: Path, net, meta: Dict[str, Any], manifest: Path) -> None:
    """Write to ``<name>.tmp``, rename, then append ``{file, sha256, ...}`` to the manifest.

    The manifest row is the "complete" signal incremental pulls wait for.
    """
    from src.model.ego2s_network import save_ego2s_checkpoint

    tmp = path.with_name(path.name + ".tmp")
    save_ego2s_checkpoint(tmp, net, meta)
    os.replace(tmp, path)
    row = {
        "file": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "transitions": meta.get("transitions"),
        "t": time.time(),
    }
    with manifest.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def rss_gb() -> float:
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9


def resolve_opp_forward_env(args) -> None:
    """Multi-actor runs: a non-default ``M3C_OPP_FORWARD`` becomes ``args.opp_forward``.

    So the mode the actors use is recorded with the run's m3c flags (``merged-device`` is
    not same-math); an explicit ``--opp-forward`` wins, ``per-group`` / unset change nothing.
    """
    env = os.environ.get(OPP_FORWARD_ENV)
    if not getattr(args, "actors", 0) or args.opp_forward is not None or not env:
        return
    if env not in OPP_FORWARDS:
        raise SystemExit(f"{OPP_FORWARD_ENV} must be one of {OPP_FORWARDS}, got {env!r}")
    if env != "per-group":
        args.opp_forward = env


def check_m6_args(args) -> None:
    """Refuse M6 reward-lever settings that would be invalid or silently do nothing."""
    for name in ("tight_cost", "death_mass_mult"):
        v = getattr(args, name)
        if not (math.isfinite(v) and v >= 0.0):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and >= 0, got {v!r}")
    if args.tight_min_len < 0:
        raise SystemExit(f"--tight-min-len must be >= 0, got {args.tight_min_len}")
    tight = args.tight_cost > 0.0 or args.tight_telemetry
    if args.tight_min_len != TIGHT_MIN_LEN and not tight:
        raise SystemExit("--tight-min-len needs --tight-cost or --tight-telemetry")
    if (tight or args.death_mass_mult > 0.0) and not args.actors:
        raise SystemExit("the M6 reward levers need the multi-actor trainer (--actors >= 1)")
    if tight and args.obs_version != "bv8":
        # tightness is read from the bv8 actor's features=True v8 result (no extra flood)
        raise SystemExit("--tight-cost / --tight-telemetry need --obs-version bv8")


def m3c_flags(args) -> Dict[str, Any]:
    """The non-default M3-C options in effect (empty for an M3-identical run)."""
    d = default_args()
    out = {}
    for k in (
        "reward",
        "mass_kappa",
        "anchor_boost_weight",
        "gamma_hold",
        "gamma_anneal_end",
        "anchor_hold",
        "anchor_floor_at",
        "anchor_floor",
        "td_huber_delta",
        "eps_end_at",
        "namespace",
        "mixed_frac",
        "actors",
        "kill_scale",
        "death_value",
        "gamma_start",
        "gamma_end",
        "anchor_mode",
        "shield",
        "shield_full_until",
        "shield_off_at",
        "onpolicy_labels",
        "label_frac",
        "label_bank",
        "label_bank_frac",
        "label_bank_margin_scale",
        "v8_backend",
        "obs_version",
        "v61_backend",
        "actor_impl",
        "learner_fast",
        "learner_precision",
        "batch_scale",
        "lr_scale",
        "obs_pack",
        "demo_mmap",
        "compile_learner",
        "compile_learner_scope",
        "compile_actor",
        "opp_forward",
        "tight_cost",
        "tight_min_len",
        "death_mass_mult",
        "tight_telemetry",
    ):
        if getattr(args, k) != getattr(d, k):
            value = getattr(args, k)
            out[k] = str(value) if k == "label_bank" and value is not None else value
    if (args.tight_cost or args.tight_telemetry) and "tight_min_len" not in out:
        out["tight_min_len"] = args.tight_min_len  # in effect: record it with its lever
    return out


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--demo-data", type=Path, required=True)
    ap.add_argument("--demo-rounds", default="10,11,12")
    ap.add_argument("--transitions", type=int, default=5_000_000)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--frozen-frac", type=float, default=0.25)
    ap.add_argument("--segment-frames", type=int, default=5000)
    ap.add_argument("--replay", type=int, default=500_000)
    ap.add_argument("--demo-capacity", type=int, default=300_000)
    ap.add_argument("--demo-initial-shards", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--checkpoint-every", type=int, default=2_500_000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, required=True)
    # --- M3-C options (defaults = M3) ---
    ap.add_argument("--reward", choices=("v2", "mass"), default="v2")
    ap.add_argument("--mass-kappa", type=float, default=1.0)
    ap.add_argument("--anchor-boost-weight", type=float, default=1.0)
    ap.add_argument("--gamma-hold", type=int, default=Schedules.gamma_hold)
    ap.add_argument("--gamma-anneal-end", type=int, default=Schedules.gamma_anneal_end)
    ap.add_argument("--anchor-hold", type=int, default=Schedules.anchor_hold)
    ap.add_argument("--anchor-floor-at", type=int, default=Schedules.anchor_floor_at)
    ap.add_argument("--anchor-floor", type=float, default=Schedules.anchor_floor)
    ap.add_argument(
        "--td-huber-delta",
        type=float,
        default=1.0,
        help="T2: Huber delta of the TD loss, huber_loss(delta) (1.0 = the M3 loss, identical "
        "to smooth_l1 beta 1). Quadratic 0.5*e^2 below delta (gradients of small errors "
        "unchanged), linear above; ~100 puts a death delta of Phi(L)+3 in the quadratic "
        "regime: an expectation fit, not a median fit. NB smooth_l1(beta) would instead "
        "divide small-error gradients by beta.",
    )
    ap.add_argument("--eps-end-at", type=int, default=Schedules.eps_end_at)
    ap.add_argument("--namespace", default=TRAIN_NAMESPACE)
    ap.add_argument(
        "--mixed-frac", type=float, default=0.0, help="share of worlds with the strict mixed roster"
    )
    ap.add_argument("--actors", type=int, default=0, help="0 = the synchronous M3 trainer")
    ap.add_argument("--actor-device", default=None, help="default: --device")
    ap.add_argument("--actor-sync-updates", type=int, default=25)
    ap.add_argument("--actor-queue-frames", type=int, default=8)
    ap.add_argument(
        "--ring-dir", default=None, help="file-backed observation rings here (default: /dev/shm)"
    )
    # --- M4-A options (defaults = M3-C; multi-actor trainer only) ---
    ap.add_argument("--kill-scale", type=float, default=None, help="reward per victim length")
    ap.add_argument("--death-value", type=float, default=None, help="reward on death")
    ap.add_argument("--gamma-start", type=float, default=Schedules.gamma_start)
    ap.add_argument("--gamma-end", type=float, default=Schedules.gamma_end)
    ap.add_argument("--anchor-mode", choices=("m3", "v8margin"), default="m3")
    ap.add_argument("--shield", choices=("none", "v8"), default="none")
    ap.add_argument("--shield-full-until", type=int, default=8_000_000)
    ap.add_argument("--shield-off-at", type=int, default=16_000_000)
    ap.add_argument(
        "--onpolicy-labels", action="store_true", help="v8 labels on the student's own states"
    )
    ap.add_argument(
        "--v8-backend", choices=("python", "numba"), default=None, help="default: env, else python"
    )
    ap.add_argument(
        "--v61-backend",
        choices=V61_BACKENDS,
        default=None,
        help="opponent/hero vector61 featurizer (default: env M3C_V61_BACKEND, else python)",
    )
    ap.add_argument(
        "--actor-impl",
        choices=ACTOR_IMPLS,
        default="vectorized",
        help="actor-side bookkeeping: vectorized (default; transition- and RNG-identical to "
        "reference, tested) or reference (the per-row loops)",
    )
    ap.add_argument("--label-frac", type=float, default=0.5, help="on-policy share of anchor rows")
    ap.add_argument(
        "--label-bank", type=Path, default=None, help="directory of T3 rollout-label npz shards"
    )
    ap.add_argument("--label-bank-frac", type=float, default=0.0)
    ap.add_argument("--label-bank-margin-scale", type=float, default=1.0)
    ap.add_argument("--label-capacity", type=int, default=300_000)
    ap.add_argument(
        "--obs-version",
        choices=OBS_VERSIONS,
        default="b",
        help="M4-B: bv8 = ego2s-b + 30 v8 feature scalars (needs an ego2s-bv8 --init)",
    )
    # --- perf-learner options (multi-actor trainer only; see learner_m3c.py) ---
    ap.add_argument(
        "--learner-fast",
        action="store_true",
        help="T1: pipelined update, one staged upload, device-side stats, fused Adam "
        "(same math; bitwise on CPU without fused Adam, tested)",
    )
    ap.add_argument(
        "--learner-precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="T2 (needs --learner-fast): autocast forwards; fp16 adds a dynamic GradScaler",
    )
    ap.add_argument(
        "--batch-scale",
        type=int,
        default=1,
        help="T3 (CHANGES training dynamics): K x the batch rows, one update per K x 128 "
        "transitions; target-sync / weight-publish intervals divided by K",
    )
    ap.add_argument(
        "--lr-scale",
        choices=("none", "linear", "sqrt"),
        default="none",
        help="T3: Adam lr x K (linear) or x sqrt(K) with --batch-scale K",
    )
    # --- perf-ram options (multi-actor trainer only; RAM only, same math) ---
    ap.add_argument(
        "--obs-pack",
        action="store_true",
        help="lossless packed observation planes in the actor rings and the demonstration "
        "buffer (~4x less RAM; verified per row, gathered batches byte-identical, tested)",
    )
    ap.add_argument(
        "--demo-mmap",
        default=None,
        help="shared read-only demonstration store at this path (built once per host under a "
        "lock if missing; every run maps it): replaces the private demonstration buffer",
    )
    # --- perf-compile options (multi-actor trainer only; see compile_m3c.py) ---
    ap.add_argument(
        "--compile-learner",
        choices=COMPILE_MODES,
        default="off",
        help="torch.compile mode for the learner update (needs --learner-fast); canonical "
        "batches only, checked against eager on the first steps, eager fallback on failure "
        "(float reassociation, not bitwise)",
    )
    ap.add_argument(
        "--compile-learner-scope",
        choices=("loss", "forward"),
        default="loss",
        help="loss: forwards + TD/anchor losses (and their backward); forward: the networks only",
    )
    ap.add_argument(
        "--compile-actor",
        choices=COMPILE_MODES,
        default="off",
        help="torch.compile mode for the actors' hero forward (padded to --worlds rows; "
        "probe-checked against eager, eager fallback on failure)",
    )
    # --- perf-opp option (multi-actor trainer only; see src/simd_env/vector61_merged.py) ---
    ap.add_argument(
        "--opp-forward",
        choices=OPP_FORWARDS,
        default=None,
        help="vector61 opponents: per-group (one call per checkpoint), merged-cpu (one stacked "
        "forward for all same-architecture checkpoints; Q bitwise = per-group on Accelerate, "
        "tested) or merged-device (that forward on --actor-device; floats differ). Default: "
        "env M3C_OPP_FORWARD, else per-group",
    )
    # --- M6 reward levers (multi-actor trainer only; defaults = off, the previous trainer) ---
    ap.add_argument(
        "--tight-cost",
        type=float,
        default=0.0,
        help="c: each tight hero decision at L >= --tight-min-len gains -c (1 - gamma) Phi(L) "
        "(tight: best advisory-safe v8 landing area < L; needs --obs-version bv8)",
    )
    ap.add_argument(
        "--tight-min-len",
        type=int,
        default=TIGHT_MIN_LEN,
        help="Lmin of the tight cost / telemetry (needs --tight-cost or --tight-telemetry)",
    )
    ap.add_argument(
        "--death-mass-mult",
        type=float,
        default=0.0,
        help="w: a hero death adds -w Phi(L_before) on top of the v2 reward (death_value kept)",
    )
    ap.add_argument(
        "--tight-telemetry",
        action="store_true",
        help="log tight_frac without a reward change (control arms; needs --obs-version bv8)",
    )
    ap.add_argument("--no-guard", action="store_true", help="tests only: skip the Mac guard")
    return ap


def default_args():
    return build_parser().parse_args(
        ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "x"]
    )


# ----------------------------------------------------------------------------- main loop
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.actors < 0:
        raise SystemExit("--actors must be >= 0")
    if args.reward == "v2" and args.mass_kappa != 1.0:
        raise SystemExit("--mass-kappa needs --reward mass")
    m4a = (
        args.shield != "none"
        or args.onpolicy_labels
        or args.anchor_mode != "m3"
        or args.kill_scale is not None
        or args.death_value is not None
        or args.gamma_start != Schedules.gamma_start
        or args.gamma_end != Schedules.gamma_end
        or args.obs_version != "b"
        or args.label_bank_frac > 0.0
    )
    if m4a and not args.actors:
        raise SystemExit("the M4-A options need the multi-actor trainer (--actors >= 1)")
    check_m6_args(args)
    if args.onpolicy_labels and args.anchor_mode != "v8margin":
        raise SystemExit("--onpolicy-labels needs --anchor-mode v8margin")
    if not args.shield_full_until <= args.shield_off_at:
        raise SystemExit("--shield-full-until must be <= --shield-off-at")
    if not math.isfinite(args.label_bank_frac) or not math.isfinite(args.label_bank_margin_scale):
        raise SystemExit("label-bank fractions and scale must be finite")
    if args.label_bank_frac < 0.0 or args.label_bank_frac > 1.0:
        raise SystemExit("--label-bank-frac must be in [0, 1]")
    if args.label_bank_frac and args.label_bank is None:
        raise SystemExit("--label-bank-frac needs --label-bank")
    if args.label_bank_frac and args.anchor_mode != "v8margin":
        raise SystemExit("--label-bank-frac needs --anchor-mode v8margin")
    if args.label_bank_frac and args.obs_version != "bv8":
        raise SystemExit("--label-bank-frac needs --obs-version bv8")
    if args.label_bank_frac and not (math.isfinite(args.label_frac) and 0 <= args.label_frac <= 1):
        raise SystemExit("--label-frac must be finite and in [0, 1] for a label bank")
    if args.label_bank_frac and args.label_bank_margin_scale < 0:
        raise SystemExit("--label-bank-margin-scale must be >= 0")
    if args.label_bank_frac and args.label_frac + args.label_bank_frac > 1.0:
        raise SystemExit("--label-frac + --label-bank-frac must be <= 1")
    bank_only_bv8 = (
        args.label_bank is not None
        and args.label_bank_frac > 0.0
        and args.label_frac + args.label_bank_frac == 1.0
        and (args.label_frac == 0.0 or args.onpolicy_labels)
    )
    if args.obs_version == "bv8" and not (
        (args.onpolicy_labels and args.label_frac == 1.0 and args.anchor_mode == "v8margin")
        or bank_only_bv8
    ):
        # The demonstration shards carry only the 30 ego2s-b scalars: every anchor row must
        # be an on-policy label (observed with the v8 features by the actor).
        raise SystemExit(
            "--obs-version bv8 needs --onpolicy-labels --label-frac 1.0 --anchor-mode v8margin"
        )
    perf_learner = (
        args.learner_fast
        or args.learner_precision != "fp32"
        or args.batch_scale != 1
        or args.lr_scale != "none"
        or args.compile_learner != "off"
    )
    if args.compile_learner_scope != "loss" and args.compile_learner == "off":
        raise SystemExit("--compile-learner-scope needs --compile-learner")
    if args.compile_actor != "off" and not args.actors:
        raise SystemExit("--compile-actor needs the multi-actor trainer (--actors >= 1)")
    if perf_learner and not args.actors:
        raise SystemExit("the perf-learner options need the multi-actor trainer (--actors >= 1)")
    if args.opp_forward is not None and not args.actors:
        raise SystemExit("--opp-forward needs the multi-actor trainer (--actors >= 1)")
    resolve_opp_forward_env(args)
    if perf_learner:
        from research.redesign_m3c_20261008.learner_m3c import LearnerOpts

        try:
            LearnerOpts.from_args(args)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if (args.obs_pack or args.demo_mmap) and not args.actors:
        raise SystemExit("--obs-pack / --demo-mmap need the multi-actor trainer (--actors >= 1)")
    if args.demo_mmap and args.obs_version == "bv8":
        raise SystemExit("--demo-mmap: an --obs-version bv8 run has no demonstration buffer")
    if args.actors:
        return main_multi(args)
    return main_sync(args)


def main_sync(args) -> int:
    import torch

    from research.redesign_m2_20261008.gen_data import guard_ok
    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    if args.no_guard:

        def guard_ok():  # noqa: F811
            return []

    if args.out.exists():
        print(f"refusing: {args.out} exists (create-only)", file=sys.stderr)
        return 2
    problems = guard_ok()
    if problems:
        print(json.dumps({"refused": problems}), flush=True)
        return 3
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng([args.seed, 17])
    device = torch.device(args.device)
    sched = schedules_from_args(args)
    ctx = gid._context(1)
    B = EgoRasterConfig(version="b")

    online = load_ego2s_checkpoint(args.init, device).train()
    target = load_ego2s_checkpoint(args.init, device).eval()
    opt = torch.optim.Adam(online.parameters(), lr=1e-4)
    margin = json.loads((args.demo_data / "margin.json").read_text())["margin"]
    demos = DemoStream(
        args.demo_data, [int(r) for r in args.demo_rounds.split(",")], args.demo_capacity, args.seed
    )
    demos.fill_initial(args.demo_initial_shards)
    demo_qmax = float(np.abs(demos.buf["q_teacher"][: demos.size]).max())
    holdout = DemoStream.load(
        sorted((args.demo_data / "round-12").glob("chunk-*-part-*.npz"))[0], held_out=True
    )
    ring = ObsRing(args.replay + 4 * args.worlds * 8)
    replay = Replay(args.replay)
    mass = args.reward == "mass"
    tel = Telemetry(args.out, args)

    def to_dev(x, dtype=None):
        t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        return t if dtype is None else t.to(dtype)

    def masked_argmax(q, m):
        m = m.clone()
        m[~m.any(dim=1), :3] = True
        return torch.where(m, q, torch.full_like(q, -1e9)).argmax(dim=1)

    @torch.no_grad()
    def holdout_agreement() -> float:
        online.eval()
        q = online(to_dev(holdout["local"]), to_dev(holdout["global"]), to_dev(holdout["scalars"]))
        pred = masked_argmax(q, to_dev(holdout["mask_resolved"])).cpu().numpy()
        online.train()
        tel.calibration(q.cpu().numpy(), holdout)
        return float((pred == holdout["a_v8"]).mean())

    log_path = args.out / "log.jsonl"
    manifest = args.out / "manifest.jsonl"
    state = {
        "transitions": 0,
        "updates": 0,
        "flags": [],
        "halt": None,
        "since_update": 0,
        "td": [],
        "anc": [],
        "qmax": 0.0,
        "env_s": 0.0,
        "learn_s": 0.0,
        "decisions": 0,
        "boost": 0,
        "greedy_boost": 0,
        "deaths": 0,
        "action_window": deque(maxlen=100_000),
    }
    started = time.perf_counter()
    meta_base = {
        "schema": SCHEMA,
        "seed": args.seed,
        "init": args.init,
        "init_sha256": hashlib.sha256(Path(args.init).read_bytes()).hexdigest(),
    }
    flags = m3c_flags(args)
    if flags:
        meta_base.update(schema=SCHEMA_M3C, m3c=flags)

    def update(gamma_now: float, lam: float) -> None:
        t0 = time.perf_counter()
        idx = replay.sample(rng, 384, ring)
        if len(idx) < 64:
            return
        sl, sg, ss, _ = ring.get(replay.s[idx])
        nxt = replay.nxt[idx]
        has_next = nxt >= 0
        nl, ng, ns, nm = ring.get(np.where(has_next, nxt, replay.s[idx]))
        a = to_dev(replay.a[idx])
        R = to_dev(replay.R[idx])
        disc = to_dev(replay.disc[idx])
        d = demos.sample(rng, 128)
        n_agent = len(idx)
        # One online forward for the agent and demonstration rows (shared graph).
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
        td = torch.nn.functional.huber_loss(qa, y, delta=float(args.td_huber_delta))
        db = {k: to_dev(d[k]) for k in ("q_teacher", "a_v8", "mask_resolved")}
        if args.anchor_boost_weight == 1.0:
            anc = anchor_loss(torch, qd, db, margin)
        else:
            anc = anchor_loss_weighted(torch, qd, db, margin, args.anchor_boost_weight)
        loss = td + lam * anc
        if not torch.isfinite(loss):
            state["halt"] = "non-finite loss"
            return
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        state["updates"] += 1
        state["td"].append(float(td.detach()))
        state["anc"].append(float(anc.detach()))
        state["qmax"] = max(state["qmax"], float(q.detach().abs().max()))
        if state["updates"] % 1000 == 0:
            target.load_state_dict(online.state_dict())
            demos.prefetch()
        demos.swap_if_ready()
        state["learn_s"] += time.perf_counter() - t0

    next_log = 100_000
    next_ckpt = args.checkpoint_every
    segment = 0
    n_step = N_STEP
    while state["transitions"] < args.transitions and state["halt"] is None:
        problems = guard_ok()
        if problems:
            state["flags"].append({"t": state["transitions"], "guard": problems})
            time.sleep(60)
            continue
        gamma_seg = sched.gamma(state["transitions"])
        sim, runtime, policies, groups = build_segment(args, ctx, segment, gamma_seg)
        E = sim.E
        pend: List[deque] = [deque() for _ in range(E)]  # per world: (obs_id, a, reward)
        last_obs = np.full(E, -1, dtype=np.int64)

        def flush(e: int, bootstrap: int, gamma_now: float) -> None:
            while pend[e]:
                R, g = 0.0, 1.0
                for _, _, r in pend[e]:
                    R += g * r
                    g *= gamma_now
                s0, a0, _ = pend[e].popleft()
                replay.add(s0, a0, R, 0.0 if bootstrap < 0 else g, bootstrap)
                state["since_update"] += 1

        for _frame in range(args.segment_frames):
            t_env = time.perf_counter()
            t = state["transitions"]
            eps, gamma_now = sched.eps(t), sched.gamma(t)
            chosen: Dict[str, Any] = {}

            def choose(prepared):
                masks = prepared.get_resolved_action_mask()
                alive = prepared.get_alive()
                actions = np.ones((E, prepared.S), dtype=np.int64)
                runtime.prepare(prepared)
                hero = np.argwhere(alive[:, :1])
                if len(hero):
                    t_f = time.perf_counter()
                    obs = build_ego_raster(prepared, B, backend="numba", rows=hero)
                    m = masks[hero[:, 0], 0]
                    t_q = time.perf_counter()
                    with torch.no_grad():
                        q = online(
                            to_dev(obs["local"]), to_dev(obs["global"]), to_dev(obs["scalars"])
                        )
                        greedy = masked_argmax(q, to_dev(m)).cpu().numpy()
                    tel.add("featurizer_s", t_q - t_f)
                    tel.add("forward_s", time.perf_counter() - t_q)
                    mm = m.copy()
                    mm[~mm.any(axis=1), :3] = True
                    rand = np.array([rng.choice(np.flatnonzero(r)) for r in mm])
                    act = np.where(rng.random(len(hero)) < eps, rand, greedy)
                    actions[hero[:, 0], 0] = act
                    chosen.update(hero=hero, obs=obs, mask=m, act=act, greedy=greedy)
                t_o = time.perf_counter()
                for spec, cells in groups.items():
                    live = cells[alive[cells[:, 0], cells[:, 1]]]
                    if len(live):
                        mk = masks[live[:, 0], live[:, 1]]
                        actions[live[:, 0], live[:, 1]] = policies[spec].actions(mk, prepared, live)
                tel.add("opponents_s", time.perf_counter() - t_o)
                return actions

            sim.step_with_policy(choose)
            runtime.observe_step(sim)
            reward = sim.get_reward()[:, 0]
            done = sim.get_done()[:, 0]
            if mass:
                alive_after = sim.get_alive()[:, 0] & ~done
                # the sim's shaping gamma (segment start), so Phi terms cancel exactly
                reward = reward + mass_term(
                    sim.get_lengths()[:, 0], alive_after, gamma_seg, args.mass_kappa
                )
            if chosen:
                hero = chosen["hero"][:, 0]
                ids = ring.add(chosen["obs"], chosen["mask"])
                for j, e in enumerate(hero):
                    e = int(e)
                    if len(pend[e]) == n_step:  # the oldest item's n-step window is complete
                        R, g = 0.0, 1.0
                        for _, _, r in pend[e]:
                            R += g * r
                            g *= gamma_now
                        s0, a0, _ = pend[e].popleft()
                        replay.add(s0, a0, R, g, int(ids[j]))
                        state["since_update"] += 1
                    pend[e].append((int(ids[j]), int(chosen["act"][j]), float(reward[e])))
                    last_obs[e] = ids[j]
                    if done[e]:
                        flush(e, -1, gamma_now)
                        state["deaths"] += 1
                acts = chosen["act"]
                state["decisions"] += len(acts)
                state["boost"] += int((acts >= 3).sum())
                state["greedy_boost"] += int((chosen["greedy"] >= 3).sum())
                state["action_window"].extend(acts.tolist())
                state["transitions"] += len(acts)
            state["env_s"] += time.perf_counter() - t_env
            if replay.size >= args.warmup:
                while state["since_update"] >= 128 and state["halt"] is None:
                    state["since_update"] -= 128
                    update(gamma_now, sched.anchor(state["transitions"]))
            if len(state["action_window"]) == state["action_window"].maxlen:
                counts = np.bincount(np.asarray(state["action_window"]), minlength=6)
                if counts.max() > 0.9 * counts.sum():
                    state["halt"] = f"action collapse {counts.tolist()}"
            if state["halt"] or state["transitions"] >= args.transitions:
                break
            if state["transitions"] >= next_log:
                next_log += 100_000
                wall = time.perf_counter() - started
                row = {
                    "transitions": state["transitions"],
                    "updates": state["updates"],
                    "wall": wall,
                    "rate": state["transitions"] / wall,
                    "env_s": state["env_s"],
                    "learn_s": state["learn_s"],
                    "eps": eps,
                    "gamma": gamma_now,
                    "lambda": sched.anchor(state["transitions"]),
                    "td": float(np.mean(state["td"])) if state["td"] else None,
                    "anchor": float(np.mean(state["anc"])) if state["anc"] else None,
                    "qmax": state["qmax"],
                    "boost_rate": state["boost"] / max(1, state["decisions"]),
                    "greedy_boost_rate": state["greedy_boost"] / max(1, state["decisions"]),
                    "deaths_per_1k": 1000 * state["deaths"] / max(1, state["decisions"]),
                    "demo_shards_loaded": demos.loaded,
                    "demo_distinct": demos.distinct,
                }
                if state["transitions"] % 500_000 < 100_000:
                    row["holdout_agreement"] = holdout_agreement()
                    if row["holdout_agreement"] < 0.6:
                        state["flags"].append(
                            {"t": state["transitions"], "agreement": row["holdout_agreement"]}
                        )
                if state["qmax"] > 5 * demo_qmax:
                    state["flags"].append({"t": state["transitions"], "qmax": state["qmax"]})
                state.update(
                    td=[], anc=[], qmax=0.0, boost=0, greedy_boost=0, decisions=0, deaths=0
                )
                with log_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                tel.write(row, replay_size=replay.size)
            if state["transitions"] >= next_ckpt:
                save_checkpoint_atomic(
                    args.out / f"ckpt_{next_ckpt // 1000}k.pth",
                    online,
                    {**meta_base, "transitions": state["transitions"], "updates": state["updates"]},
                    manifest,
                )
                next_ckpt += args.checkpoint_every
        # Segment end: truncation, bootstrap from each world's last observation.
        for e in range(E):
            if pend[e]:
                # The last decision has no observed successor: drop it; the earlier ones
                # bootstrap from it (truncation).
                pend[e].pop()
                flush(e, int(last_obs[e]), sched.gamma(state["transitions"]))
        segment += 1

    # A HALTed run's last weights are "aborted.pth" (never "final.pth": selection reads
    # final checkpoints only).
    save_checkpoint_atomic(
        args.out / ("aborted.pth" if state["halt"] else "final.pth"),
        online,
        {**meta_base, "transitions": state["transitions"], "updates": state["updates"]},
        manifest,
    )
    summary = {
        "schema": SCHEMA if not flags else SCHEMA_M3C,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "transitions": state["transitions"],
        "updates": state["updates"],
        "wall_seconds": time.perf_counter() - started,
        "rate": state["transitions"] / (time.perf_counter() - started),
        "halt": state["halt"],
        "flags": state["flags"],
        "demo_qmax": demo_qmax,
        "demo_distinct": demos.distinct,
        "holdout_agreement_final": holdout_agreement(),
        "m3c": flags,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("transitions", "rate", "halt", "flags")}))
    return 1 if state["halt"] else 0


# ----------------------------------------------------------------------------- telemetry
class Telemetry:
    """Side-channel diagnostics (``telemetry.jsonl``); never touches the training state."""

    def __init__(self, out: Path, args) -> None:
        self.path = out / "telemetry.jsonl"
        self.kappa = args.mass_kappa if args.reward == "mass" else 0.0
        self.acc: Dict[str, float] = {}
        self.cal: Dict[str, float] = {}
        self.t0 = time.time()

    def add(self, key: str, seconds: float) -> None:
        self.acc[key] = self.acc.get(key, 0.0) + seconds

    def calibration(self, q: np.ndarray, holdout: Dict[str, np.ndarray]) -> None:
        """Held-out Q calibration: greedy Q, U = Q + Phi(L) (the v2 'unshaped' value)."""
        m = holdout["mask_resolved"].copy()
        m[~m.any(axis=1), :3] = True
        qg = np.where(m, q, -np.inf).max(axis=1)
        phi = holdout["scalars"][:, 1].astype(np.float64) * 1000.0 / PHI_DIV
        u = qg + phi
        self.cal = {
            "holdout_q_greedy_mean": float(qg.mean()),
            "holdout_u_mean": float(u.mean()),
            "holdout_u_p99": float(np.quantile(u, 0.99)),
            "holdout_u_minus_kappa_phi_mean": float((u - self.kappa * phi).mean()),
            "holdout_greedy_boost_rate": float(
                (np.where(m, q, -np.inf).argmax(axis=1) >= 3).mean()
            ),
        }

    def write(self, row: Dict[str, Any], **extra: Any) -> None:
        out = {
            "transitions": row["transitions"],
            "t": time.time(),
            "rss_gb_max": rss_gb(),
            **{k: round(v, 3) for k, v in self.acc.items()},
            **self.cal,
            **extra,
        }
        self.cal = {}
        with self.path.open("a") as fh:
            fh.write(json.dumps(out) + "\n")


# ----------------------------------------------------------------------------- multi-actor
def _weights_spec(net):
    sd = net.state_dict()
    return [(k, tuple(v.shape), v.numel()) for k, v in sd.items()]


def _publish_weights(net, wbuf: np.ndarray, wver: np.ndarray) -> None:
    import torch

    flat = torch.cat([v.detach().float().reshape(-1).cpu() for v in net.state_dict().values()])
    wver[0] += 1  # odd: writing
    wbuf[:-1] = flat.numpy()
    wbuf[-1] = float(flat.double().sum())
    wver[0] += 1  # even: stable


def _read_weights(net, spec, wbuf: np.ndarray, wver: np.ndarray, last: int) -> int:
    """Copy published weights into ``net`` if a newer stable version exists (seqlock)."""
    import torch

    v1 = int(wver[0])
    if v1 % 2 or v1 <= last:
        return last
    flat = wbuf[:-1].copy()
    check = float(wbuf[-1])
    if int(wver[0]) != v1 or abs(float(flat.astype(np.float64).sum()) - check) > 1e-3 * (
        1 + abs(check)
    ):
        return last  # torn read: try again next frame
    sd, off = {}, 0
    for k, shape, n in spec:
        sd[k] = torch.from_numpy(flat[off : off + n].reshape(shape))
        off += n
    net.load_state_dict(sd)
    return v1


def _profile_count(meta: Dict[str, Any], msg: Dict[str, Any]) -> None:
    """Per-frame counters of a profiled actor window (``M3C_PROFILE``; never called when off)."""
    meta["frames"] = meta.get("frames", 0) + 1
    meta["decisions"] = meta.get("decisions", 0) + int(msg["decisions"])
    meta["env_s"] = meta.get("env_s", 0.0) + float(msg["env_s"])
    for k, v in msg["timing"].items():
        meta[k] = meta.get(k, 0.0) + float(v)


def actor_main(j: int, cfg: Dict[str, Any], names: Dict[str, str], tx, stop_evt) -> None:
    """One actor process: worlds, acting, n-step windows, observation ring writes."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    torch.set_num_threads(1)
    args = argparse.Namespace(**cfg["args"])
    K = int(cfg["K"])
    from research.redesign_m3c_20261008.profile_m3c import ProcessProfile

    prof = ProcessProfile.from_env(f"actor{j}", Path(args.out))  # None unless M3C_PROFILE
    shm_ring = Block.attach(names[f"ring{j}"])
    shm_w = Block.attach(names["weights"])
    shm_p = Block.attach(names["progress"])
    try:
        bv8 = getattr(args, "obs_version", "b") == "bv8"
        pack = None
        if cfg.get("obs_pack") is not None:
            from research.redesign_m3c_20261008.obs_pack import (
                PackLayout,
                layout_for_arena,
            )

            pack = PackLayout.from_dict(cfg["obs_pack"])
        ring = SharedObsRing(
            shm_ring, cfg["ring_cap"], cfg["ring_margin"], nscal_for(args), pack=pack
        )
        nw = cfg["n_weights"]
        wver = np.ndarray((1,), dtype=np.int64, buffer=shm_w.buf, offset=0)
        wbuf = np.ndarray((nw + 1,), dtype=np.float64, buffer=shm_w.buf, offset=8)
        progress = np.ndarray((4,), dtype=np.int64, buffer=shm_p.buf, offset=0)
        device = torch.device(cfg["actor_device"])
        net = load_init(args, "cpu").eval()
        spec = _weights_spec(net)
        last = _read_weights(net, spec, wbuf, wver, -1)
        net = net.to(device)
        hero_fwd = None  # perf-compile (--compile-actor): the compiled, padded hero forward
        if getattr(args, "compile_actor", "off") != "off":
            from research.redesign_m3c_20261008.compile_m3c import (
                HeroForward,
                limit_compile_workers,
            )

            inductor = limit_compile_workers("actor")
            hero_fwd = HeroForward(
                torch, net, device, args.compile_actor, args.worlds, nscal_for(args), LOCAL, GLOBAL
            )
            msg = {"actor": j, "compile_actor": hero_fwd.status(), "inductor": inductor}
            print(json.dumps(msg), flush=True)
        rng = np.random.default_rng([args.seed, 17, 1000 + j])
        sched = schedules_from_args(args)
        ctx = gid._context(1)
        B = EgoRasterConfig(version="b")
        mass = args.reward == "mass"
        m6 = M6Levers.from_args(args)  # None: every M6 lever off (the previous actor)
        if m6 is not None and m6.tight and not bv8:
            raise ValueError("the M6 tight predicate needs --obs-version bv8")
        use_v8 = args.shield != "none" or args.onpolicy_labels
        carrier = None
        if use_v8:
            from research.redesign_m2_20261008 import ni_spec

            carrier = ctx["lookup"][ni_spec.FRP3_S12[1]][1]
        fast = getattr(args, "actor_impl", "vectorized") == "vectorized"
        sample = sample_legal if fast else sample_legal_ref
        opp_forward = (
            getattr(args, "opp_forward", None) or os.environ.get(OPP_FORWARD_ENV) or "per-group"
        )
        if opp_forward not in OPP_FORWARDS:
            raise ValueError(f"opp forward must be one of {OPP_FORWARDS}, got {opp_forward!r}")
        segment = j
        while not stop_evt.is_set():
            t_glob = int(progress[0])
            gamma_seg = sched.gamma(t_glob)
            # The Python v8 reads the hero's vector61 rows; the compiled one (numba shield,
            # bv8 features) needs only the sim, so the hero is not featurized for it.
            python_v8 = (
                use_v8
                and not bv8
                and ((args.v8_backend or os.environ.get(V8_BACKEND_ENV) or "python") == "python")
            )
            sim, runtime, policies, groups = build_segment(
                args, ctx, segment, gamma_seg, hero_v61=python_v8
            )
            if pack is not None and layout_for_arena(sim.grid_w, sim.grid_h, sim.S) != pack:
                raise RuntimeError(f"--obs-pack layout {pack} does not match this arena")
            # bv8: one compiled v8 evaluation per frame (features=True) serves both the
            # observation's feature scalars and the shield / label action (v8_redecide).
            shield = (
                make_shield(carrier, runtime, backend=args.v8_backend)
                if use_v8 and not bv8
                else None
            )
            if bv8:
                from src.simd_env.ego_raster_bv8 import v8_scalars
                from src.simd_env.v8_nb import v8_redecide
                from src.simd_env.vector61_policy import check_live_rule_config

                check_live_rule_config(sim)
            # perf-opp (opt-in): one stacked forward for every same-architecture vector61
            # opponent checkpoint (merged-*); the rest (scripted anchors) stay per group.
            merged = None
            opp_groups = groups
            if opp_forward != "per-group":
                from src.simd_env.vector61_merged import MergedVector61Opponents

                merged = MergedVector61Opponents(
                    policies,
                    groups,
                    runtime,
                    device="cpu" if opp_forward == "merged-cpu" else device,
                )
                opp_groups = {k: v for k, v in groups.items() if k not in merged.specs}
            E = sim.E
            nsb = NStepBuilder(E) if fast else NStepBuilderRef(E)
            for _frame in range(args.segment_frames):
                if stop_evt.is_set():
                    break
                if prof is not None:
                    prof.tick()
                while progress[1] and not stop_evt.is_set():  # learner paused (Mac guard)
                    time.sleep(1.0)
                if int(wver[0]) > last:
                    if hero_fwd is not None:  # in place: the compiled forward keeps its storage
                        last = _read_weights(net, spec, wbuf, wver, last)
                    else:
                        cpu_net = net.to("cpu")
                        last = _read_weights(cpu_net, spec, wbuf, wver, last)
                        net = cpu_net.to(device)
                t0 = time.perf_counter()
                t_glob = int(progress[0])
                eps, gamma_now = sched.eps(t_glob), sched.gamma(t_glob)
                chosen: Dict[str, Any] = {}
                timing = {"featurizer_s": 0.0, "forward_s": 0.0, "opponents_s": 0.0, "v8_s": 0.0}
                p_shield = (
                    shield_prob(t_glob, args.shield_full_until, args.shield_off_at)
                    if args.shield != "none"
                    else 0.0
                )

                def opponents(prepared, masks, alive, actions):
                    t_o = time.perf_counter()
                    # merged: enqueue first (async on MPS / CUDA), host groups, then collect;
                    # the opponents' cells are disjoint and none of them draws actor RNG.
                    pending = merged.launch(alive) if merged is not None else None
                    for spec_, cells in opp_groups.items():
                        live = cells[alive[cells[:, 0], cells[:, 1]]]
                        if len(live):
                            mk = masks[live[:, 0], live[:, 1]]
                            actions[live[:, 0], live[:, 1]] = policies[spec_].actions(
                                mk, prepared, live
                            )
                    if pending is not None:
                        merged.finish(pending, actions)
                    timing["opponents_s"] += time.perf_counter() - t_o

                def choose(prepared):
                    masks = prepared.get_resolved_action_mask()
                    alive = prepared.get_alive()
                    actions = np.ones((E, prepared.S), dtype=np.int64)
                    runtime.prepare(prepared)
                    hero = np.argwhere(alive[:, :1])
                    # vectorized: the opponents (no actor RNG, read-only runtime state, disjoint
                    # action cells) run while an asynchronous (MPS / CUDA) hero forward is in
                    # flight; the hero's decisions and RNG draws are unchanged.
                    overlap = fast and len(hero) > 0
                    if len(hero):
                        t_f = time.perf_counter()
                        obs = build_ego_raster(prepared, B, backend="numba", rows=hero)
                        m = masks[hero[:, 0], 0]
                        mm = m.copy()
                        mm[~mm.any(axis=1), :3] = True  # the normal-only fallback
                        dt_v8 = 0.0
                        if bv8:
                            # Every stored hero observation gets its v8 features here, from
                            # the same selection mask the shield / labels decide with.
                            t_v = time.perf_counter()
                            sel_mask = hero_selection_mask(prepared, hero)
                            v8s, v8res = v8_scalars(prepared, hero, sel_mask)
                            if v8res is None:
                                raise RuntimeError("bv8: v8 features need live hero rows")
                            obs["scalars"] = np.concatenate([obs["scalars"], v8s], axis=1)
                            dt_v8 = time.perf_counter() - t_v
                            timing["v8_s"] += dt_v8
                        t_q = time.perf_counter()
                        with torch.no_grad():
                            if hero_fwd is not None:
                                q = hero_fwd(obs)
                            else:
                                q = net(
                                    *[
                                        torch.from_numpy(np.ascontiguousarray(obs[k])).to(device)
                                        for k in ("local", "global", "scalars")
                                    ]
                                )
                            if fast:  # the same mask, built on the host
                                mt = torch.from_numpy(mm).to(device)
                            else:
                                mt = torch.from_numpy(m.copy()).to(device)
                                mt[~mt.any(dim=1), :3] = True
                            greedy_t = torch.where(mt, q, torch.full_like(q, -1e9)).argmax(dim=1)
                            dt_opp = 0.0
                            if overlap:
                                t_o = time.perf_counter()
                                opponents(prepared, masks, alive, actions)
                                dt_opp = time.perf_counter() - t_o
                            greedy = greedy_t.cpu().numpy()
                        timing["featurizer_s"] += t_q - t_f - dt_v8
                        timing["forward_s"] += time.perf_counter() - t_q - dt_opp
                        rand = sample(rng, mm)
                        explore = rng.random(len(hero)) < eps
                        policy_act = greedy
                        if shield is not None or bv8:
                            t_v = time.perf_counter()
                            qn = q.float().cpu().numpy()
                            if bv8:  # the hero is slot 0, a veto slot: the v8 action
                                a_v8 = v8_redecide(v8res, qn, sel_mask).action
                            else:
                                a_v8 = shield.act(qn, m, prepared, hero)
                            timing["v8_s"] += time.perf_counter() - t_v
                            chosen["a_v8"] = a_v8
                            if p_shield > 0.0:
                                shielded = rng.random(len(hero)) < p_shield
                                policy_act = np.where(shielded, a_v8, greedy)
                        act = np.where(explore, rand, policy_act)
                        actions[hero[:, 0], 0] = act
                        chosen.update(hero=hero, obs=obs, mask=m, act=act, greedy=greedy)
                        if m6 is not None:  # M6: decision-time length (and tightness)
                            L_dec = np.asarray(prepared.length[hero[:, 0], 0], dtype=np.int64)
                            chosen["L"] = L_dec
                            if m6.tight:
                                chosen["tight"] = tight_predicate(v8res.area, sel_mask, L_dec)
                    if not overlap:
                        opponents(prepared, masks, alive, actions)
                    return actions

                sim.step_with_policy(choose)
                runtime.observe_step(sim)
                reward = sim.get_reward()[:, 0]
                done = sim.get_done()[:, 0]
                if mass:
                    alive_after = sim.get_alive()[:, 0] & ~done
                    reward = reward + mass_term(
                        sim.get_lengths()[:, 0], alive_after, gamma_seg, args.mass_kappa
                    )
                m6_stats: Dict[str, int] = {}
                if m6 is not None:
                    if chosen:
                        hw = chosen["hero"][:, 0]
                        delta, m6_stats = m6.frame(
                            chosen["L"], chosen.get("tight"), done[hw], gamma_seg
                        )
                        if delta is not None:
                            reward = np.array(reward, dtype=np.float64)
                            reward[hw] = reward[hw] + delta
                    else:
                        m6_stats = m6.empty_stats()
                deaths = 0
                stats = {"decisions": 0, "boost": 0, "greedy_boost": 0, "v8_override": 0}
                acts = np.zeros(0, dtype=np.int64)
                lab_ids = np.zeros(0, dtype=np.int64)
                lab_a = np.zeros(0, dtype=np.int8)
                if chosen:
                    ids = ring.add(chosen["obs"], chosen["mask"])
                    if "a_v8" in chosen and args.onpolicy_labels:
                        lab_ids, lab_a = ids, chosen["a_v8"].astype(np.int8)
                    deaths = nsb.step(
                        chosen["hero"][:, 0], ids, chosen["act"], reward, done, gamma_now
                    )
                    acts = chosen["act"]
                    stats = {
                        "decisions": int(len(acts)),
                        "boost": int((acts >= 3).sum()),
                        "greedy_boost": int((chosen["greedy"] >= 3).sum()),
                        "v8_override": (
                            int((chosen["a_v8"] != chosen["greedy"]).sum())
                            if "a_v8" in chosen
                            else 0
                        ),
                    }
                last_frame = _frame == args.segment_frames - 1
                if last_frame:
                    nsb.end_segment(sched.gamma(int(progress[0])))
                s, a, R, d, nx = nsb.take()
                msg = {
                    "j": j,
                    "s": s,
                    "a": a,
                    "R": R,
                    "disc": d,
                    "nxt": nx,
                    "acts": acts.astype(np.int8),
                    "lab_ids": lab_ids,
                    "lab_a": lab_a,
                    "deaths": deaths,
                    "env_s": time.perf_counter() - t0,
                    "timing": timing,
                    "weights_version": last,
                    **stats,
                    **m6_stats,
                }
                if prof is not None and prof.running:
                    _profile_count(prof.meta, msg)
                while not stop_evt.is_set():
                    try:
                        tx.put(msg, timeout=1.0)  # bounded: back-pressure from the learner
                        break
                    except queue.Full:
                        continue
            segment += K
    finally:
        if prof is not None:
            prof.safe_dump({"worlds": int(args.worlds), "actor": j})
        shm_ring.close()
        shm_w.close()
        shm_p.close()


def _learner_counters(st: Dict[str, Any]) -> Dict[str, Any]:
    """Learner counters bracketing a profiled window (``M3C_PROFILE``; never called when off)."""
    keys = ("transitions", "updates", "learn_s", "wait_s", "env_s")
    return {**{k: st[k] for k in keys}, "t": time.perf_counter()}


def main_multi(args) -> int:
    """Learner process with ``args.actors`` actor processes (see the module docstring)."""
    import multiprocessing as mp

    import torch

    from research.redesign_m2_20261008.gen_data import guard_ok

    if args.no_guard:

        def guard_ok():  # noqa: F811
            return []

    # Validate the immutable bank before output allocation, shared memory, or actors.
    # Keep this whole branch absent for default runs so their startup/RNG path is unchanged.
    bank = None
    if args.label_bank_frac > 0.0:
        from research.t3_rollout_labels_20261015.bank import LabelBank

        bank_pack = None
        if args.obs_pack:
            from research.redesign_m3c_20261008.obs_pack import train_layout

            bank_pack = train_layout()
        bank = LabelBank(Path(args.label_bank), nscal_for(args), pack=bank_pack)
    if args.out.exists():
        print(f"refusing: {args.out} exists (create-only)", file=sys.stderr)
        return 2
    problems = guard_ok()
    if problems:
        print(json.dumps({"refused": problems}), flush=True)
        return 3
    from research.redesign_m3c_20261008.profile_m3c import ProcessProfile

    # None unless M3C_PROFILE; a bad value fails here, before any shared memory / actor exists
    prof = ProcessProfile.from_env("learner", args.out)
    args.out.mkdir(parents=True)
    K = int(args.actors)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng([args.seed, 17])
    device = torch.device(args.device)
    sched = schedules_from_args(args)

    from src.model.ego2s_network import OBS_SPEC_B, OBS_SPEC_BV8, spec_for

    bv8 = args.obs_version == "bv8"
    nscal = nscal_for(args)
    online = load_init(args, device).train()
    want_spec = OBS_SPEC_BV8 if bv8 else OBS_SPEC_B
    if spec_for(online) != want_spec:
        raise SystemExit(
            f"--obs-version {args.obs_version} needs a {want_spec} --init, got "
            f"{spec_for(online)} (widen with research/redesign_m4b_20261012/widen_bv8.py)"
        )
    target = load_init(args, device).eval()
    from research.redesign_m3c_20261008.learner_m3c import (
        Learner,
        LearnerOpts,
        make_optimizer,
    )

    lopts = LearnerOpts.from_args(args)
    if lopts.compile != "off":
        from research.redesign_m3c_20261008.compile_m3c import limit_compile_workers

        msg = {"compile_learner_inductor": limit_compile_workers("learner")}
        print(json.dumps(msg), flush=True)
    opt = make_optimizer(torch, online, lopts, device)  # default: Adam(lr=1e-4), as before
    margin = json.loads((args.demo_data / "margin.json").read_text())["margin"]
    holdout = DemoStream.load(
        sorted((args.demo_data / "round-12").glob("chunk-*-part-*.npz"))[0], held_out=True
    )
    pack = None
    if args.obs_pack:  # perf-ram: the training arena's packed layout (rings + demo buffer)
        from research.redesign_m3c_20261008.obs_pack import train_layout

        pack = train_layout()
    if bv8:
        # The shards only have the 30 ego2s-b scalars: no demonstration rows (every anchor row
        # is an on-policy label), no held-out agreement; the Q-scale flag uses the hold-out.
        demos = None
        demo_qmax = float(np.abs(holdout["q_teacher"]).max())
    else:
        rounds = [int(r) for r in args.demo_rounds.split(",")]
        store = None
        if args.demo_mmap:
            from research.redesign_m3c_20261008.demo_store import SharedDemoStore

            store = SharedDemoStore.open_or_build(
                Path(args.demo_mmap), args.demo_data, rounds, pack
            )
        demos = DemoStream(
            args.demo_data,
            rounds,
            args.demo_capacity,
            args.seed,
            pack=pack,
            store=store,
        )
        demos.fill_initial(args.demo_initial_shards)
        demo_qmax = demos.q_teacher_absmax()
    replay = Replay(args.replay)
    labels = LabelBuffer(args.label_capacity) if args.onpolicy_labels else None
    tel = Telemetry(args.out, args)
    # Rings (ring_capacity): with the defaults (replay 500k, K = 2, 64 worlds) 2 x 377k rows x
    # 12.3 KB = 9.3 GB per run (--obs-pack: 3.2 KB per row, 2.4 GB). research/perf_ram_20261013
    ring_cap, margin_rows = ring_capacity(args.replay, K, args.worlds, args.actor_queue_frames)
    spec = _weights_spec(online)
    n_weights = int(sum(n for _, _, n in spec))
    shms: List[Any] = []

    def new_shm(nbytes: int, directory: Optional[str] = None, tag: str = "blk"):
        s = Block.create(nbytes, directory, tag)
        shms.append(s)
        return s

    names: Dict[str, str] = {}
    rings = []
    for j in range(K):
        s = new_shm(ring_nbytes(ring_cap, nscal, pack), args.ring_dir, f"ring{j}")
        names[f"ring{j}"] = s.desc
        rings.append(SharedObsRing(s, ring_cap, margin_rows, nscal, pack=pack))
    s_w = new_shm(8 + 8 * (n_weights + 1), args.ring_dir, "weights")
    names["weights"] = s_w.desc
    wver = np.ndarray((1,), dtype=np.int64, buffer=s_w.buf, offset=0)
    wver[0] = 0
    wbuf = np.ndarray((n_weights + 1,), dtype=np.float64, buffer=s_w.buf, offset=8)
    s_p = new_shm(32, args.ring_dir, "progress")
    names["progress"] = s_p.desc
    progress = np.ndarray((4,), dtype=np.int64, buffer=s_p.buf, offset=0)
    progress[:] = 0
    _publish_weights(online, wbuf, wver)  # version 2 = publish #1
    multi = MultiRing(rings)

    ctx_mp = mp.get_context("spawn")
    stop_evt = ctx_mp.Event()
    txs = [ctx_mp.Queue(maxsize=args.actor_queue_frames) for _ in range(K)]
    cfg = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "K": K,
        "ring_cap": ring_cap,
        "ring_margin": margin_rows,
        "n_weights": n_weights,
        "actor_device": args.actor_device or args.device,
    }
    if pack is not None:
        cfg["obs_pack"] = pack.to_dict()
    cfg["args"]["demo_data"] = str(args.demo_data)
    procs = [
        ctx_mp.Process(target=actor_main, args=(j, cfg, names, txs[j], stop_evt), daemon=True)
        for j in range(K)
    ]
    for p in procs:
        p.start()

    def to_dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    def masked_argmax(q, m):
        m = m.clone()
        m[~m.any(dim=1), :3] = True
        return torch.where(m, q, torch.full_like(q, -1e9)).argmax(dim=1)

    @torch.no_grad()
    def holdout_agreement() -> Optional[float]:
        if bv8:
            return None  # the hold-out shard has no v8 feature scalars
        online.eval()
        q = online(to_dev(holdout["local"]), to_dev(holdout["global"]), to_dev(holdout["scalars"]))
        pred = masked_argmax(q, to_dev(holdout["mask_resolved"])).cpu().numpy()
        online.train()
        tel.calibration(q.cpu().numpy(), holdout)
        return float((pred == holdout["a_v8"]).mean())

    log_path = args.out / "log.jsonl"
    manifest = args.out / "manifest.jsonl"
    st = {
        "transitions": 0,
        "updates": 0,
        "flags": [],
        "halt": None,
        "since_update": 0,
        "td": [],
        "anc": [],
        "qmax": 0.0,
        "env_s": 0.0,
        "learn_s": 0.0,
        "wait_s": 0.0,
        "ring_overrun_rows": 0,
        "decisions": 0,
        "boost": 0,
        "greedy_boost": 0,
        "deaths": 0,
        "v8_override": 0,
        "label_rows": [],
        "anchor_updates": 0,
        "lag": [],
        "action_window": deque(maxlen=100_000),
    }
    if bank is not None:
        st.update(bank_rows=[], bank_margin_loss=[], bank_agree=[])
    m6 = M6Levers.from_args(args)  # None: every M6 lever off (no M6 keys anywhere)
    if m6 is not None:
        st["m6"], st["m6_total"] = {}, {"decisions": 0, "deaths": 0}
    meta_base = {
        "schema": SCHEMA_M3C,
        "seed": args.seed,
        "init": args.init,
        "init_sha256": hashlib.sha256(Path(args.init).read_bytes()).hexdigest(),
        "m3c": m3c_flags(args),
    }
    started = time.perf_counter()
    next_log, next_ckpt, next_guard = 100_000, args.checkpoint_every, 0
    published = 1  # the initial publish (weights version 2 * published)

    learner = Learner(
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
        args=args,
        nscal=nscal,
        margin=margin,
        device=device,
        opts=lopts,
        **({"bank": bank} if bank is not None else {}),
    )

    code = 0
    rr = 0
    attached: set = set()
    try:
        while st["transitions"] < args.transitions and st["halt"] is None:
            if prof is not None and prof.tick():
                prof.meta["start"] = _learner_counters(st)
            if st["transitions"] >= next_guard:
                next_guard += 320_000
                problems = guard_ok()
                progress[1] = 1 if problems else 0
                if problems:
                    st["flags"].append({"t": st["transitions"], "guard": problems})
                    time.sleep(60)
                    next_guard = st["transitions"]
                    continue
            for p in procs:
                if not p.is_alive():
                    raise RuntimeError(f"actor {procs.index(p)} died ({p.exitcode})")
            got = False
            t_w = time.perf_counter()
            first, rr = rr, (rr + 1) % K  # fair round-robin start
            for j in [(first + i) % K for i in range(K)]:
                try:
                    msg = txs[j].get(timeout=0.05 if not got else 0.0)
                except queue.Empty:
                    continue
                got = True
                if j not in attached:
                    attached.add(j)
                    if len(attached) == K:  # every actor mapped its blocks: unlink the names
                        for blk in shms:  # (the mappings live on; nothing leaks on a crash)
                            blk.unlink()
                gs = multi.encode(j, msg["s"])
                gn = multi.encode(j, msg["nxt"])
                replay.add_many(gs, msg["a"], msg["R"], msg["disc"], gn)
                st["since_update"] += len(gs)
                if labels is not None and len(msg["lab_ids"]):
                    labels.add(multi.encode(j, msg["lab_ids"]), msg["lab_a"].astype(np.int64))
                st["v8_override"] += msg["v8_override"]
                st["transitions"] += msg["decisions"]
                st["decisions"] += msg["decisions"]
                st["boost"] += msg["boost"]
                st["greedy_boost"] += msg["greedy_boost"]
                st["deaths"] += msg["deaths"]
                st["env_s"] += msg["env_s"]
                st["lag"].append(published - msg["weights_version"] // 2)
                st["action_window"].extend(msg["acts"].tolist())
                for k_, v_ in msg["timing"].items():
                    tel.add(k_, v_)
                if m6 is not None:
                    tot = st["m6_total"]
                    tot["decisions"] += msg["decisions"]
                    tot["deaths"] += msg["deaths"]
                    for k_, v_ in msg.items():
                        if k_.startswith("m6_"):
                            st["m6"][k_] = st["m6"].get(k_, 0) + v_
                            tot[k_] = tot.get(k_, 0) + v_
            st["wait_s"] += time.perf_counter() - t_w
            progress[0] = st["transitions"]
            if replay.size >= args.warmup:
                while st["since_update"] >= learner.update_every and st["halt"] is None:
                    st["since_update"] -= learner.update_every
                    before = st["updates"]
                    learner.update(sched.anchor(st["transitions"]))
                    if learner.publish_due(before):
                        _publish_weights(online, wbuf, wver)
                        published += 1
            if len(st["action_window"]) == st["action_window"].maxlen:
                counts = np.bincount(np.asarray(st["action_window"]), minlength=6)
                if counts.max() > 0.9 * counts.sum():
                    st["halt"] = f"action collapse {counts.tolist()}"
            if st["transitions"] >= next_log:
                next_log += 100_000
                learner.flush()  # fast learner: apply the pending step, sync device stats
                wall = time.perf_counter() - started
                t = st["transitions"]
                row = {
                    "transitions": t,
                    "updates": st["updates"],
                    "wall": wall,
                    "rate": t / wall,
                    "env_s": st["env_s"],
                    "learn_s": st["learn_s"],
                    "eps": sched.eps(t),
                    "gamma": sched.gamma(t),
                    "lambda": sched.anchor(t),
                    "td": float(np.mean(st["td"])) if st["td"] else None,
                    "anchor": float(np.mean(st["anc"])) if st["anc"] else None,
                    "qmax": st["qmax"],
                    "boost_rate": st["boost"] / max(1, st["decisions"]),
                    "greedy_boost_rate": st["greedy_boost"] / max(1, st["decisions"]),
                    "deaths_per_1k": 1000 * st["deaths"] / max(1, st["decisions"]),
                    "demo_shards_loaded": demos.loaded if demos is not None else 0,
                    "demo_distinct": demos.distinct if demos is not None else 0,
                    "actors": K,
                }
                if m6 is not None:
                    row.update(m6_log_fields(st["m6"], st["decisions"], st["deaths"]))
                    st["m6"] = {}
                if bv8:
                    row["obs_version"] = "bv8"
                    row["label_buffer"] = labels.size if labels is not None else 0
                if bank is not None:
                    row["bank_rows_mean"] = (
                        float(np.mean(st["bank_rows"])) if st["bank_rows"] else 0.0
                    )
                    row["bank_margin_loss"] = (
                        float(np.mean(st["bank_margin_loss"])) if st["bank_margin_loss"] else 0.0
                    )
                    row["bank_agree"] = (
                        float(np.mean(st["bank_agree"])) if st["bank_agree"] else 0.0
                    )
                if args.shield != "none" or labels is not None:
                    row["v8_override_rate"] = st["v8_override"] / max(1, st["decisions"])
                    row["shield_p"] = (
                        shield_prob(t, args.shield_full_until, args.shield_off_at)
                        if args.shield != "none"
                        else 0.0
                    )
                    row["label_rows_mean"] = (
                        float(np.mean(st["label_rows"])) if st["label_rows"] else 0.0
                    )
                if t % 500_000 < 100_000 and not bv8:
                    row["holdout_agreement"] = holdout_agreement()
                    if row["holdout_agreement"] < 0.6:
                        st["flags"].append({"t": t, "agreement": row["holdout_agreement"]})
                if st["qmax"] > 5 * demo_qmax:
                    st["flags"].append({"t": t, "qmax": st["qmax"]})
                lag = st["lag"]
                tel.write(
                    row,
                    replay_size=replay.size,
                    learner_wait_s=round(st["wait_s"], 3),
                    policy_lag_mean=float(np.mean(lag)) if lag else None,
                    ring_overrun_rows=st["ring_overrun_rows"],
                    queue_depth=[q_.qsize() if sys.platform != "darwin" else None for q_ in txs],
                    ring_counts=[r.count for r in rings],
                    **(
                        {"compile_learner": learner.compile_status()}
                        if lopts.compile != "off"
                        else {}
                    ),
                )
                st.update(td=[], anc=[], qmax=0.0, boost=0, greedy_boost=0, decisions=0, deaths=0)
                st.update(v8_override=0, label_rows=[])
                if bank is not None:
                    st.update(bank_rows=[], bank_margin_loss=[], bank_agree=[])
                st["lag"] = []
                with log_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if st["transitions"] >= next_ckpt:
                learner.flush()
                save_checkpoint_atomic(
                    args.out / f"ckpt_{next_ckpt // 1000}k.pth",
                    online,
                    {**meta_base, "transitions": st["transitions"], "updates": st["updates"]},
                    manifest,
                )
                next_ckpt += args.checkpoint_every
    except BaseException as exc:  # noqa: BLE001
        st["halt"] = st["halt"] or f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        if prof is not None and prof.running:
            prof.meta["end"] = _learner_counters(st)
            prof.stop()
        stop_evt.set()
        for q_ in txs:  # drain so blocked actors can exit
            try:
                while True:
                    q_.get_nowait()
            except Exception:  # noqa: BLE001
                pass
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        for blk in shms:  # idempotent; covers a crash before every actor attached
            blk.unlink()
    try:
        learner.flush()  # fast learner: the last pipelined step belongs to the final weights
    except Exception as exc:  # noqa: BLE001
        st["halt"] = st["halt"] or f"flush: {type(exc).__name__}: {exc}"
        code = 1
    save_checkpoint_atomic(
        args.out / ("aborted.pth" if st["halt"] else "final.pth"),
        online,
        {**meta_base, "transitions": st["transitions"], "updates": st["updates"]},
        manifest,
    )
    summary = {
        "schema": SCHEMA_M3C,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "transitions": st["transitions"],
        "updates": st["updates"],
        "wall_seconds": time.perf_counter() - started,
        "rate": st["transitions"] / (time.perf_counter() - started),
        "halt": st["halt"],
        "flags": st["flags"][:200],
        "demo_qmax": demo_qmax,
        "demo_distinct": demos.distinct if demos is not None else 0,
        "holdout_agreement_final": holdout_agreement(),
        "m3c": m3c_flags(args),
        "obs_version": args.obs_version,
        "actors": K,
        "actor_exitcodes": [p.exitcode for p in procs],
        "ring_overrun_rows": st["ring_overrun_rows"],
        "anchor_updates": st["anchor_updates"],
        "label_buffer": labels.size if labels is not None else 0,
    }
    if lopts.compile != "off":
        summary["compile_learner"] = learner.compile_status()
    if m6 is not None:
        tot = st["m6_total"]
        summary["m6_totals"] = {**tot, **m6_log_fields(tot, tot["decisions"], tot["deaths"])}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if prof is not None:
        prof.safe_dump({"actors": K, "worlds": int(args.worlds), "device": str(device)})
    print(json.dumps({k: summary[k] for k in ("transitions", "rate", "halt", "flags")}))
    for s in shms:  # views may still exist: close() tolerates that (names already unlinked)
        s.close()
    return 1 if (st["halt"] or code) else 0


if __name__ == "__main__":
    raise SystemExit(main())
