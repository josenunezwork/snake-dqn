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
  (or ``--update-every M`` x ``--batch-scale K`` x 128) and back-pressures the actors through
  bounded queues. Target synchronization remains counted in learner updates. Not
  bit-reproducible (policy lag, scheduling); ``--actors 0`` (default) is the M3 trainer.
* ``--all-snakes-learner`` (multi-actor only): default-off self-play throughput mode.  The
  actor's learner-network copy drives every live slot and contributes one transition per
  driven slot, using one batched raster forward per frame.  The ordinary hero-only roster,
  serving, and all defaults are unchanged.
* ``--max-wall-seconds`` (multi-actor only): default-off graceful smoke bound.  It stops the
  learner loop at the requested elapsed wall time, then writes the normal final checkpoint
  and summary with ``stopped_for_time: true``.  It is a weights-only restart boundary, not an
  exact optimizer/replay/actor-state resume unless full-state checkpointing is enabled.
* ``--full-state-every N``: opt-in full training state, with local serialization and durable
  copying in separate workers. ``--resume-full-state DIR`` restores a verified manifest;
  see ``CHECKPOINTS.md`` for replay size, refill and measured capture latency limits.
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
  ``--update-every M`` (T4: one unchanged-size update per M x K x 128 ingested transitions)
  is a separate, default-off recipe lever; target synchronization retains its existing
  K-adjusted update interval.

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
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import queue  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
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
    anchor_start: float = 1.0
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
            return self.anchor_start
        if t >= self.anchor_floor_at:
            return self.anchor_floor
        f = (t - self.anchor_hold) / (self.anchor_floor_at - self.anchor_hold)
        return math.exp((1.0 - f) * math.log(self.anchor_start) + f * math.log(self.anchor_floor))

    def eps(self, t: int) -> float:
        # Constant schedules retain a positive denominator: Q1's mature package
        # is explicit configuration, never a division-by-zero special case.
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
        anchor_start=getattr(args, "anchor_start", Schedules.anchor_start),
        anchor_floor=args.anchor_floor,
        eps_start=getattr(args, "eps_start", Schedules.eps_start),
        eps_end=getattr(args, "eps_end", Schedules.eps_end),
        eps_end_at=args.eps_end_at,
    )


def apply_schedule_package(args) -> None:
    """Resolve Q1's named schedule package before any actor/learner is built.

    ``restart`` leaves every historical argument untouched.  ``mature`` is a
    package rather than an inferred continuation: it starts at B8's finished
    gamma, epsilon and anchor values while retaining positive schedule spans for
    telemetry and for code paths that calculate interpolation fractions.
    """
    if getattr(args, "schedule_package", "restart") != "mature":
        return
    args.gamma_start = args.gamma_end = 0.998
    args.eps_start = args.eps_end = 0.01
    args.anchor_start = args.anchor_floor = 0.1


def apply_task_reward_gamma(sim, gamma: float) -> None:
    """Set the Q1 task collector's current reward gamma on its live simulator.

    Legacy actors bind one segment gamma as before.  Task-aligned N=1 has
    independent episode clocks and cannot let an implementation segment freeze
    the reward/PBRS gamma while its target schedule advances.
    """
    sim.cfg = replace(sim.cfg, gamma=float(gamma))


def reset_task_worlds(sim, mask: np.ndarray, seeds: Sequence[int]) -> None:
    """Reset terminal task worlds with constructor-identical seed initialization.

    ``BatchSim.reset_envs`` intentionally skips the constructor-only discarded
    food draws.  That is correct for respawns, but Q1 samples a *fresh gate
    world* at each terminal hero reset, so it must replay those initial draws.
    The simulator exposes the validated reset-RNG construction; this small
    adapter preserves its validation while selecting the constructor reset path.
    """
    selected = np.flatnonzero(sim._normalize_reset_env_mask(mask))
    replacements = sim._validated_reset_rngs(selected, seeds)
    if replacements is None:
        raise ValueError("task-world reset requires replacement seeds")
    for e, (seed, rng) in zip(selected, replacements):
        sim._seeds[int(e)] = seed
        sim._rngs[int(e)] = rng
    for e in selected:
        sim._reset_env(int(e), constructor_food_draws=True)
    task_mask = np.zeros(sim.E, dtype=bool)
    task_mask[selected] = True
    sim._rebuild_traversed_from_heads(task_mask)
    sim._refresh_action_masks(task_mask)


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
BOOST_LENGTH_BUCKETS = ("lt_400", "400_700", "700_1000", "ge_1000")


def boost_bucket_counts(
    lengths: np.ndarray, greedy_actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return decision and greedy-boost counts for the pre-registered length buckets."""
    buckets = np.digitize(np.asarray(lengths, dtype=np.int64), (400, 700, 1000), right=False)
    decisions = np.bincount(buckets, minlength=len(BOOST_LENGTH_BUCKETS))[:4].astype(np.int64)
    boosts = np.bincount(
        buckets, weights=(np.asarray(greedy_actions) >= 3), minlength=len(BOOST_LENGTH_BUCKETS)
    )[:4].astype(np.int64)
    return decisions, boosts


def boost_bucket_rates(decisions: np.ndarray, boosts: np.ndarray) -> dict[str, float]:
    return {
        name: float(boosts[i] / max(1, decisions[i])) for i, name in enumerate(BOOST_LENGTH_BUCKETS)
    }


def per_1k_or_none(numerator: int, denominator: int) -> float | None:
    """Return a rate per 1,000, retaining an empty denominator as undefined."""
    return 1000 * numerator / denominator if denominator else None


def boost_halt_reason(args, row: dict, transitions: int) -> str | None:
    """Apply the independent overall and below-700 greedy-boost tripwires."""
    if args.boost_halt_rate is not None and row["greedy_boost_rate"] > args.boost_halt_rate:
        return (
            f"overall greedy boost rate {row['greedy_boost_rate']:.6f} exceeded "
            f"--boost-halt-rate {args.boost_halt_rate:.6f} at {transitions} transitions"
        )
    low = row["greedy_boost_rate_by_length"]["lt_400"]
    middle = row["greedy_boost_rate_by_length"]["400_700"]
    low_rate = (
        low * row["greedy_boost_length_decisions"]["lt_400"]
        + middle * row["greedy_boost_length_decisions"]["400_700"]
    ) / max(
        1,
        row["greedy_boost_length_decisions"]["lt_400"]
        + row["greedy_boost_length_decisions"]["400_700"],
    )
    row["greedy_boost_rate_lt_700"] = float(low_rate)
    if args.boost_halt_low_length_rate is not None and low_rate > args.boost_halt_low_length_rate:
        return (
            f"greedy boost rate below length 700 {low_rate:.6f} exceeded "
            f"--boost-halt-low-length-rate {args.boost_halt_low_length_rate:.6f} "
            f"at {transitions} transitions"
        )
    return None


def arm_halt_reason(args) -> str | None:
    """Observe the optional shared RETURN_C stop receipt without touching other processes."""
    path = getattr(args, "arm_halt_file", None)
    if path is not None and Path(path).exists():
        return f"shared arm halt receipt exists: {path}"
    return None


def write_arm_halt_receipt(args, reason: str, transitions: int) -> None:
    """Create the shared receipt once; another lane only observes it on its own 100k row."""
    path = getattr(args, "arm_halt_file", None)
    if path is None:
        return
    try:
        with Path(path).open("x") as stream:
            json.dump({"reason": reason, "transitions": transitions}, stream, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        pass


def run_return_early_warning(args, checkpoint: Path, transition: int, log_path: Path) -> dict:
    """Run the T4f binding evaluator after its checkpoint; evaluator failure is fail-closed."""
    config_path = getattr(args, "return_early_warning_config", None)
    if config_path is None or transition not in (500_000, 1_000_000):
        return {"status": "SKIPPED"}
    try:
        config = json.loads(Path(config_path).read_text())
        output_dir = Path(config.get("output_dir", args.out))
        result_path = output_dir / f"early_warning_{transition // 1000}k.json"
        command = [
            sys.executable,
            "-m",
            "research.t4f_return_20261016.early_warning",
            "--checkpoint",
            str(checkpoint),
            "--initial-checkpoint",
            str(config.get("initial_checkpoint", args.init)),
            "--holdout-bank",
            str(config["holdout_bank"]),
            "--outside-bank",
            str(config["outside_bank"]),
            "--telemetry-jsonl",
            str(log_path),
            "--transitions",
            str(transition),
            "--out",
            str(result_path),
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "KILL", "reason": f"invalid --return-early-warning-config: {exc}"}
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    try:
        result = json.loads(result_path.read_text())
    except (OSError, ValueError) as exc:
        return {"status": "KILL", "reason": f"early warning produced no valid receipt: {exc}"}
    if completed.returncode != 0 and result.get("status") != "KILL":
        result = {**result, "status": "KILL", "reason": "early warning evaluator failed closed"}
    return result


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
        self.end_worlds(np.arange(len(self.pend)), gamma_end)

    def end_worlds(self, worlds, gamma_end: float) -> None:
        """Flush selected worlds at their individual segment-clock boundary."""
        for e in np.asarray(worlds, dtype=np.int64):
            e = int(e)
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
        self.end_worlds(np.arange(len(self.cnt)), gamma_end)

    def end_worlds(self, worlds, gamma_end: float) -> None:
        """Flush selected worlds without disturbing windows in other worlds."""
        requested = np.asarray(worlds, dtype=np.int64).reshape(-1)
        w = requested
        w = w[self.cnt[w] > 0]
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
        # A one-row window yields no transition after its terminal successor is
        # dropped, but it must still be cleared at this world's clock boundary.
        self.cnt[requested] = 0
        self.head[requested] = 0

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
    if getattr(args, "collection_profile", "legacy") == "task-aligned-n1":
        # Q1's C/D worlds sample the historical gate mixes uniformly at reset.
        # ``_design_rows`` below remains the single roster/wrapper authority.
        from research.redesign_m3c_20261008.task_collection import equal_mix_assignments

        mix_of = equal_mix_assignments(args.seed, segment * E + np.arange(E)).tolist()
    else:
        mix_of = (
            ["scripted"] * (E - n_frozen - n_mixed) + ["frozen"] * n_frozen + ["mixed"] * n_mixed
        )
    v61 = getattr(args, "v61_backend", None) or os.environ.get(V61_BACKEND_ENV) or "python"
    runtime = Vector61Runtime(backend=v61)
    controlled = np.zeros((E, 6), dtype=bool)
    all_snakes = learner_seats(args) == 6
    if hero_v61:  # M4-A shield / labels: the v8 veto reads the hero's vector61 rows
        controlled[:, 0] = True
    policies: Dict[Any, Any] = {}
    groups: Dict[Any, List[tuple]] = {}
    # N=3 supplies every frozen action through ``StaticPool`` below.  Do not even
    # construct the historical roster policies: doing so would spend forwards and
    # could accidentally reintroduce an unpinned opponent path.
    if not all_snakes and learner_seats(args) != 3:
        rows = {(int(r["world_seed"]), r["mix"]): r for r in dev_screen._design_rows(seeds)}
        lookup = ctx["lookup"]
        for e, seed in enumerate(seeds):
            for slot, member in enumerate(rows[(int(seed), mix_of[e])]["slots"], start=1):
                groups.setdefault(lookup[member["member_sha256"]], []).append((e, slot))
        for spec, cells in groups.items():
            if spec[0] == "scripted":
                policies[spec] = FastProfileAnchorSimdPolicy(spec[1], seeds)
            else:
                policies[spec] = Vector61SimdPolicy(
                    spec[1],
                    runtime,
                    forward=(
                        "rowwise"
                        if getattr(args, "collection_profile", "legacy") == "task-aligned-n1"
                        else "batched"
                    ),
                )
                for e, s in cells:
                    controlled[e, s] = True
    # The N=3 pool can place a vector61 policy in any non-learner seat, including
    # slot zero.  Bind precisely those rows before its dispatcher is constructed;
    # its static seat map is recomputed from the same isolated segment seed below.
    if learner_seats(args) == 3:
        controlled[~learner_seat_mask(args.seed, segment, E, 3)] = True
    runtime.bind(sim, controlled)
    if getattr(args, "collection_profile", "legacy") == "task-aligned-n1":
        sim._task_mixes = np.asarray(mix_of, dtype=object)
        sim._task_reset_ordinals = np.zeros(E, dtype=np.int64)
    groups_arr = {k: np.array(v, dtype=np.int64) for k, v in groups.items()}
    # T4 is deliberately entirely default-off: do not allocate, import, or draw a
    # restart RNG in the ordinary M3-C path.
    if getattr(args, "restart_frac", 0.0) > 0.0:
        from research.t4_snapshot_restart_20261016.snapshots import SnapshotStore

        store = getattr(args, "_restart_store", None)
        if store is None:
            store = SnapshotStore(args.restart_bank)
            store.validate_segment(args.segment_frames)
            args._restart_store = store
        policies, groups_arr, restarted = store.apply(
            sim, runtime, policies, groups_arr, args.restart_frac, args.seed, segment
        )
        sim._restart_worlds = restarted
    return sim, runtime, policies, groups_arr


def learner_seat_mask(seed: int, segment: int, worlds: int, seats: int) -> np.ndarray:
    """Return the segment-pinned live-seat mask for the opt-in league collector.

    One learner seat deliberately remains the historical hero (slot zero), and six
    seats deliberately remains the historical all-slot collector.  Those two cases
    must not consume a new RNG stream.  The mixed case uses an isolated seed sequence
    so changing the collector cannot perturb the environment or action RNG streams.
    """
    if seats not in (1, 3, 6):
        raise ValueError("learner seats must be one of 1, 3, 6")
    mask = np.zeros((int(worlds), 6), dtype=bool)
    if seats == 1:
        mask[:, 0] = True
        return mask
    if seats == 6:
        mask[:] = True
        return mask
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 19_017, int(segment)]))
    for world in range(int(worlds)):
        mask[world, rng.permutation(6)[:seats]] = True
    return mask


def learner_seats(args) -> int:
    """Effective learner-seat count, preserving both pre-league collector paths."""
    requested = getattr(args, "learner_seats", None)
    if requested is None:
        return 6 if getattr(args, "all_snakes_learner", False) else 1
    return int(requested)


def refresh_task_aligned_rosters(args, ctx, sim, runtime, policies, groups, mixes) -> None:
    """Rebuild N=1's exact gate seats after one or more terminal-world resets.

    Every environment keeps its own current seed and sampled mix.  Rebuilding
    from ``dev_screen._design_rows`` is intentionally less clever than caching
    a synthetic training pool: it preserves the historical frozen/scripted/
    mixed construction, including its wrappers and slot rules.
    """
    from research.apex_safety_20260926 import dev_screen
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    rows = {(int(r["world_seed"]), r["mix"]): r for r in dev_screen._design_rows(sim._seeds)}
    cells_by_spec: Dict[Any, List[tuple]] = {}
    for e, seed in enumerate(sim._seeds):
        for slot, member in enumerate(rows[(int(seed), str(mixes[e]))]["slots"], start=1):
            spec = ctx["lookup"][member["member_sha256"]]
            cells_by_spec.setdefault(spec, []).append((e, slot))
    for spec in cells_by_spec:
        if spec in policies:
            continue
        if spec[0] == "scripted":
            policies[spec] = FastProfileAnchorSimdPolicy(spec[1], sim._seeds)
        else:
            policies[spec] = Vector61SimdPolicy(spec[1], runtime, forward="rowwise")
    # Random-safe depends on the current world identity.  Its policy is otherwise
    # stateless, so rebinding this tuple cannot carry a prior episode into a new one.
    for policy in policies.values():
        if isinstance(policy, FastProfileAnchorSimdPolicy):
            policy._world_seeds = tuple(int(seed) for seed in sim._seeds)
    groups.clear()
    groups.update(
        {spec: np.asarray(cells, dtype=np.int64) for spec, cells in cells_by_spec.items()}
    )
    controlled = np.zeros((sim.E, sim.S), dtype=bool)
    for spec, cells in groups.items():
        if spec[0] != "scripted":
            controlled[cells[:, 0], cells[:, 1]] = True
    runtime.controlled[...] = controlled


def mixed_stream_ids(rows: np.ndarray, live_seats: np.ndarray, seats: int = 3) -> np.ndarray:
    """Map ``(world, physical-seat)`` learner rows to dense per-life n-step streams."""
    rows = np.asarray(rows, dtype=np.int64)
    if not np.asarray(live_seats, dtype=bool)[rows[:, 0], rows[:, 1]].all():
        raise ValueError("mixed stream requested for a frozen seat")
    ranks = np.cumsum(np.asarray(live_seats, dtype=np.int64), axis=1) - 1
    stream = rows[:, 0] * int(seats) + ranks[rows[:, 0], rows[:, 1]]
    return stream.astype(np.int64, copy=False)


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


# ---------------------------------------------------------------------- full-state checkpoints
# These helpers use ordinary dicts of arrays. The asynchronous writer owns
# serialization and durability; this module owns the training-state contract. In particular,
# a replay row is only meaningful with its observation ring(s), because ``s`` and ``nxt`` are
# monotone ring ids rather than observations.
FULL_STATE_SCHEMA = "m3c-full-state/v1"


def full_state_recipe(args) -> Dict[str, str]:
    """Math/layout identity, excluding migration paths and checkpoint infrastructure."""
    ignored = {
        "out",
        "init",
        "demo_data",
        "device",
        "actor_device",
        "ckpt_local_dir",
        "ckpt_durable_dir",
        "resume_full_state",
        "full_state_every",
        "full_state_replay",
        "transitions",
        "checkpoint_every",
        "no_guard",
        "ring_dir",
        "demo_mmap",
        "max_wall_seconds",
    }
    return {
        key: str(value)
        for key, value in vars(args).items()
        if key not in ignored and not key.startswith("_")
    }


def _array_state(obj: Any, names: Sequence[str], *, copy: bool = True) -> Dict[str, Any]:
    """Return copies of named ndarray/scalar attributes suitable for a checkpoint."""
    out: Dict[str, Any] = {}
    for name in names:
        value = getattr(obj, name)
        out[name] = value.copy() if copy and isinstance(value, np.ndarray) else value
    return out


def _restore_array_state(obj: Any, state: Dict[str, Any]) -> None:
    for name, value in state.items():
        old = getattr(obj, name)
        if isinstance(old, np.ndarray):
            if old.shape != value.shape or old.dtype != value.dtype:
                raise ValueError(f"full-state {name} has incompatible shape/dtype")
            old[...] = value
        else:
            setattr(obj, name, value)


def replay_state(replay: Replay, *, copy: bool = True) -> Dict[str, Any]:
    """Copy a replay's transition ring.  Its observation ring is saved separately."""
    return _array_state(replay, ("s", "a", "R", "disc", "nxt", "size", "ptr", "added"), copy=copy)


def restore_replay_state(replay: Replay, state: Dict[str, Any]) -> None:
    _restore_array_state(replay, state)


def obs_ring_state(ring: Any, *, copy: bool = True) -> Dict[str, Any]:
    """Copy an ObsRing or SharedObsRing including its monotone publish counter."""
    if isinstance(ring, SharedObsRing):
        return {
            "shared": True,
            "count": ring.count,
            "arrays": {
                name: array.copy() if copy else array for name, array in ring.arrays.items()
            },
        }
    return {
        "shared": False,
        "count": ring.count,
        "arrays": {
            "local": ring.local.copy() if copy else ring.local,
            "glob": ring.glob.copy() if copy else ring.glob,
            "scal": ring.scal.copy() if copy else ring.scal,
            "mask": ring.mask.copy() if copy else ring.mask,
        },
    }


def restore_obs_ring_state(ring: Any, state: Dict[str, Any]) -> None:
    arrays = state["arrays"]
    if isinstance(ring, SharedObsRing):
        for name, value in arrays.items():
            if name not in ring.arrays or ring.arrays[name].shape != value.shape:
                raise ValueError("full-state shared observation ring is incompatible")
            ring.arrays[name][...] = value
        ring._count[0] = int(state["count"])
        return
    mapping = {"local": ring.local, "glob": ring.glob, "scal": ring.scal, "mask": ring.mask}
    for name, value in arrays.items():
        if mapping[name].shape != value.shape:
            raise ValueError("full-state observation ring is incompatible")
        mapping[name][...] = value
    ring.count = int(state["count"])


def label_buffer_state(
    labels: Optional[LabelBuffer], *, copy: bool = True
) -> Optional[Dict[str, Any]]:
    return None if labels is None else _array_state(labels, ("ids", "a", "size", "ptr"), copy=copy)


def restore_label_buffer_state(
    labels: Optional[LabelBuffer], state: Optional[Dict[str, Any]]
) -> None:
    if state is not None:
        if labels is None:
            raise ValueError("full-state contains labels but labels are disabled")
        _restore_array_state(labels, state)


def _torch_rng_state(torch) -> Dict[str, Any]:
    state: Dict[str, Any] = {"cpu": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_torch_rng_state(torch, state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def full_state_payload(
    torch,
    online,
    target,
    optimizer,
    rng: np.random.Generator,
    counters: Dict[str, Any],
    schedules: Schedules,
    replay: Replay,
    labels: Optional[LabelBuffer],
    rings: Sequence[Any],
    replay_mode: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the detached logical snapshot passed to ``AsyncCheckpointer``.

    ``replay_mode=none`` intentionally omits replay and rings.  Resuming it is a
    warm-up continuation, never an exact-resume claim.  ``full`` can be many GB: the
    default multi-actor un-packed rings are about 9.3 GB before serializer overhead.
    """
    payload: Dict[str, Any] = {
        "schema": FULL_STATE_SCHEMA,
        "online": online.state_dict(),
        "target": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "counters": counters,
        "schedules": schedules.__dict__.copy(),
        "rng": {
            "learner": rng.bit_generator.state,
            "numpy": np.random.get_state(),
            "torch": _torch_rng_state(torch),
        },
        "replay_mode": replay_mode,
        # ``capture_state`` performs the single detaching copy after this logical payload is
        # assembled.  Do not pre-copy multi-GB rings here.
        "labels": label_buffer_state(labels, copy=False),
        "extra": extra or {},
    }
    if replay_mode == "full":
        payload["replay"] = replay_state(replay, copy=False)
        payload["rings"] = [obs_ring_state(ring, copy=False) for ring in rings]
    return payload


def restore_full_state(
    torch,
    payload: Dict[str, Any],
    online,
    target,
    optimizer,
    rng: np.random.Generator,
    replay: Replay,
    labels: Optional[LabelBuffer],
    rings: Sequence[Any],
) -> bool:
    """Restore learner state and return whether replay/rings were restored exactly."""
    if payload.get("schema") != FULL_STATE_SCHEMA:
        raise ValueError("not an M3-C full-state snapshot")
    online.load_state_dict(payload["online"])
    target.load_state_dict(payload["target"])
    optimizer.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["rng"]["learner"]
    np.random.set_state(payload["rng"]["numpy"])
    _restore_torch_rng_state(torch, payload["rng"]["torch"])
    restore_label_buffer_state(labels, payload.get("labels"))
    if payload.get("replay_mode") != "full":
        return False
    restore_replay_state(replay, payload["replay"])
    saved = payload.get("rings", [])
    if len(saved) != len(rings):
        raise ValueError("full-state ring count does not match this run")
    for ring, state in zip(rings, saved):
        restore_obs_ring_state(ring, state)
    return True


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
        "anchor_start",
        "anchor_floor",
        "eps_start",
        "eps_end",
        "td_huber_delta",
        "eps_end_at",
        "schedule_package",
        "namespace",
        "collection_profile",
        "mixed_frac",
        "actors",
        "all_snakes_learner",
        "learner_seats",
        "league_checkpoint_dir",
        "league_b8_checkpoint",
        "max_wall_seconds",
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
        "label_bank_counter_mode",
        "label_bank_counter_ratio",
        "label_bank_outside",
        "label_bank_outside_margin_scale",
        "return_bank",
        "return_frac",
        "return_weight",
        "return_huber_delta",
        "return_gradient_telemetry",
        "return_mode",
        "return_contrast_epsilon",
        "boost_halt_rate",
        "boost_halt_low_length_rate",
        "arm_halt_file",
        "return_early_warning_config",
        "v8_backend",
        "obs_version",
        "v61_backend",
        "actor_impl",
        "learner_fast",
        "learner_precision",
        "batch_scale",
        "lr_scale",
        "update_every",
        "gather_threads",
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
            out[k] = str(value) if isinstance(value, Path) else value
    if args.restart_frac > 0.0:
        out["restart_bank"] = str(args.restart_bank)
        out["restart_frac"] = args.restart_frac
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
    ap.add_argument(
        "--full-state-every",
        type=int,
        default=0,
        metavar="N",
        help="full-state cadence in learner transitions (0 disables; persistence is asynchronous)",
    )
    ap.add_argument(
        "--full-state-replay",
        choices=("none", "full"),
        default="none",
        help="include replay and observation rings (full may be many GB)",
    )
    ap.add_argument(
        "--ckpt-local-dir",
        type=Path,
        default=None,
        help="local snapshot directory (required with --full-state-every)",
    )
    ap.add_argument(
        "--ckpt-durable-dir",
        type=Path,
        default=None,
        help="optional durable mirror (e.g. Google Drive); copying runs in a separate worker",
    )
    ap.add_argument(
        "--resume-full-state",
        type=Path,
        default=None,
        metavar="DIR",
        help="resume from the newest verified complete full-state manifest in DIR",
    )
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
    ap.add_argument("--anchor-start", type=float, default=Schedules.anchor_start)
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
    ap.add_argument("--eps-start", type=float, default=Schedules.eps_start)
    ap.add_argument("--eps-end", type=float, default=Schedules.eps_end)
    ap.add_argument("--eps-end-at", type=int, default=Schedules.eps_end_at)
    ap.add_argument(
        "--schedule-package",
        choices=("restart", "mature"),
        default="restart",
        help="Q1 package: restart preserves historical schedules; mature fixes gamma=.998, "
        "epsilon=.01, and anchor=.1 from learner decision zero",
    )
    ap.add_argument("--namespace", default=TRAIN_NAMESPACE)
    ap.add_argument(
        "--collection-profile",
        choices=("legacy", "task-aligned-n1"),
        default="legacy",
        help="Q1 task-aligned N=1 collector; uses exact gate rosters and terminal hero worlds",
    )
    ap.add_argument(
        "--mixed-frac", type=float, default=0.0, help="share of worlds with the strict mixed roster"
    )
    ap.add_argument("--actors", type=int, default=0, help="0 = the synchronous M3 trainer")
    ap.add_argument(
        "--all-snakes-learner",
        action="store_true",
        help="default-off: learner network drives every live snake and collects every transition",
    )
    ap.add_argument(
        "--learner-seats",
        type=int,
        choices=(1, 3, 6),
        default=None,
        help="T19 opt-in static league: live learner seats per six-snake world; 1 and 6 "
        "preserve the historical hero-only and --all-snakes-learner collectors respectively",
    )
    ap.add_argument(
        "--league-checkpoint-dir",
        default=None,
        help="directory containing SHA-pinned static-pool checkpoints (default: SNAKE_CKPT_DIR)",
    )
    ap.add_argument(
        "--league-b8-checkpoint",
        default=None,
        help="explicit SHA-pinned B8-s3 pool checkpoint (defaults to --league-checkpoint-dir)",
    )
    ap.add_argument(
        "--max-wall-seconds",
        type=float,
        default=0.0,
        help="default 0: no limit; multi-actor smoke stops gracefully after this wall time",
    )
    ap.add_argument(
        "--restart-bank",
        type=Path,
        default=None,
        help="T4 snapshot bank; used only with --restart-frac > 0 by multi-actor runs",
    )
    ap.add_argument(
        "--restart-frac",
        type=float,
        default=0.0,
        help="T4 share of actor worlds restored from --restart-bank at each segment (default: 0)",
    )
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
    ap.add_argument(
        "--label-bank-counter-mode", choices=("inert", "single-action"), default="inert"
    )
    ap.add_argument(
        "--label-bank-counter-ratio",
        type=float,
        default=None,
        help="counter:decisive bank rows; default preserves the bank's stored mixture",
    )
    ap.add_argument("--label-bank-outside", type=Path, default=None)
    ap.add_argument("--label-bank-outside-margin-scale", type=float, default=1.0)
    ap.add_argument(
        "--return-bank", type=Path, default=None, help="T4d immutable rolled-out branch-return bank"
    )
    ap.add_argument(
        "--return-frac", type=float, default=0.0, help="T4d return rows per 128 anchor rows"
    )
    ap.add_argument(
        "--return-weight", type=float, default=1.0, help="multiplier for the T4d return Huber loss"
    )
    ap.add_argument(
        "--return-huber-delta",
        type=float,
        default=None,
        help="Huber delta for the T4d return loss (default: --td-huber-delta). "
        "Use 100 for branch-return magnitude regression while leaving TD at 1.",
    )
    ap.add_argument(
        "--return-gradient-telemetry",
        action="store_true",
        help="T4e calibration: measure same-update pre-clip weighted-return/TD gradients",
    )
    ap.add_argument(
        "--return-mode",
        choices=("level", "contrast"),
        default="level",
        help="return regression target: legacy absolute level or max-return-reference contrast",
    )
    ap.add_argument(
        "--return-contrast-epsilon",
        type=float,
        default=0.0,
        help="contrast dead zone in Q units before the return Huber loss (RETURN_C: 2)",
    )
    ap.add_argument(
        "--boost-halt-rate",
        type=float,
        default=None,
        help="halt after a 100k log row if greedy boost rate is strictly greater than this value",
    )
    ap.add_argument(
        "--boost-halt-low-length-rate",
        type=float,
        default=None,
        help="halt after a 100k log row if greedy boost rate below length 700 exceeds this rate",
    )
    ap.add_argument(
        "--arm-halt-file",
        type=Path,
        default=None,
        help="optional shared create-only RETURN_C KILL receipt; existing receipt halts this lane",
    )
    ap.add_argument(
        "--return-early-warning-config",
        type=Path,
        default=None,
        help="JSON paths for the binding 0.5M/1M T4f return early-warning evaluator",
    )
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
    ap.add_argument(
        "--update-every",
        type=int,
        default=1,
        help="T4 (CHANGES training dynamics): one unchanged-size learner update per M x "
        "--batch-scale x 128 ingested transitions; target sync remains counted in updates",
    )
    ap.add_argument(
        "--gather-threads",
        type=int,
        default=0,
        help="T17: persistent replay-gather worker threads (0 keeps the serial path)",
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
    apply_schedule_package(args)
    if args.full_state_every < 0:
        raise SystemExit("--full-state-every must be >= 0")
    if args.full_state_every and args.ckpt_local_dir is None:
        raise SystemExit("--full-state-every needs --ckpt-local-dir")
    if args.resume_full_state is not None and args.ckpt_local_dir is None:
        # Continuing a long run should retain a new local recovery point without making a
        # surprising Drive write: the resume source may itself be mounted Drive.
        args.ckpt_local_dir = args.out / "full_state"
    if args.actors < 0:
        raise SystemExit("--actors must be >= 0")
    if args.learner_seats is not None and args.all_snakes_learner:
        raise SystemExit("--learner-seats and --all-snakes-learner are mutually exclusive")
    if args.learner_seats is not None and not args.actors:
        raise SystemExit("--learner-seats needs the multi-actor trainer")
    if args.collection_profile == "task-aligned-n1":
        from research.redesign_m3c_20261008.task_collection import HORIZON, NAMESPACE_PREFIX

        if not args.actors or learner_seats(args) != 1:
            raise SystemExit(
                "--collection-profile task-aligned-n1 needs --actors and --learner-seats 1"
            )
        if not str(args.namespace).startswith(NAMESPACE_PREFIX):
            raise SystemExit("task-aligned-n1 needs a fresh q1-task-aligned-n1-20261017 namespace")
        if args.segment_frames != HORIZON:
            raise SystemExit(
                "task-aligned-n1 needs --segment-frames 5000 (registered H5000 horizon)"
            )
        if args.opp_forward not in (None, "per-group"):
            raise SystemExit(
                "task-aligned-n1 needs --opp-forward per-group for reset roster refresh"
            )
    if args.gather_threads < 0:
        raise SystemExit("--gather-threads must be >= 0")
    if args.max_wall_seconds < 0.0 or not math.isfinite(args.max_wall_seconds):
        raise SystemExit("--max-wall-seconds must be finite and >= 0")
    if (args.all_snakes_learner or args.max_wall_seconds > 0.0) and not args.actors:
        raise SystemExit("--all-snakes-learner / --max-wall-seconds need the multi-actor trainer")
    if args.all_snakes_learner and args.obs_version != "bv8":
        raise SystemExit("--all-snakes-learner needs --obs-version bv8 (the B8 self-play path)")
    if args.learner_seats in (3, 6) and args.obs_version != "bv8":
        raise SystemExit("--learner-seats 3/6 needs --obs-version bv8 (the B8 league path)")
    if args.all_snakes_learner and args.restart_frac > 0.0:
        raise SystemExit("--all-snakes-learner does not support --restart-frac")
    if args.learner_seats is not None and args.restart_frac > 0.0:
        raise SystemExit("--learner-seats does not support --restart-frac")
    if not math.isfinite(args.restart_frac) or not 0.0 <= args.restart_frac <= 1.0:
        raise SystemExit("--restart-frac must be finite and in [0, 1]")
    if args.restart_frac > 0.0:
        if not args.actors:
            raise SystemExit("--restart-frac needs the multi-actor trainer (--actors >= 1)")
        if args.restart_bank is None:
            raise SystemExit("--restart-frac needs --restart-bank")
        # Validate the immutable bank before shared-memory blocks or actors exist.
        from research.t4_snapshot_restart_20261016.snapshots import SnapshotStore

        SnapshotStore(args.restart_bank).validate_segment(args.segment_frames)
    elif args.restart_bank is not None:
        raise SystemExit("--restart-bank needs --restart-frac > 0")
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
    if args.label_bank_counter_ratio is not None and (
        not math.isfinite(args.label_bank_counter_ratio) or args.label_bank_counter_ratio < 0
    ):
        raise SystemExit("--label-bank-counter-ratio must be finite and >= 0")
    if args.label_bank_frac < 0.0 or args.label_bank_frac > 1.0:
        raise SystemExit("--label-bank-frac must be in [0, 1]")
    if args.label_bank_frac and args.label_bank is None:
        raise SystemExit("--label-bank-frac needs --label-bank")
    if args.label_bank_counter_mode != "inert" and not args.label_bank_frac:
        raise SystemExit("--label-bank-counter-mode needs --label-bank-frac > 0")
    if args.label_bank_counter_ratio is not None and not args.label_bank_frac:
        raise SystemExit("--label-bank-counter-ratio needs --label-bank-frac > 0")
    if args.label_bank_outside is not None:
        if not args.label_bank_frac:
            raise SystemExit("--label-bank-outside needs --label-bank-frac > 0")
        if args.label_bank_counter_mode != "single-action":
            raise SystemExit("--label-bank-outside needs --label-bank-counter-mode single-action")
        if args.label_bank_counter_ratio != 1.0:
            raise SystemExit("--label-bank-outside needs --label-bank-counter-ratio 1")
        if (
            not math.isfinite(args.label_bank_outside_margin_scale)
            or args.label_bank_outside_margin_scale < 0
        ):
            raise SystemExit("--label-bank-outside-margin-scale must be finite and >= 0")
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
    if not math.isfinite(args.return_frac) or not 0.0 <= args.return_frac <= 1.0:
        raise SystemExit("--return-frac must be finite and in [0, 1]")
    if not math.isfinite(args.return_weight) or args.return_weight < 0.0:
        raise SystemExit("--return-weight must be finite and >= 0")
    if args.return_huber_delta is not None and (
        not math.isfinite(args.return_huber_delta) or args.return_huber_delta <= 0.0
    ):
        raise SystemExit("--return-huber-delta must be finite and > 0")
    if args.boost_halt_rate is not None and (
        not math.isfinite(args.boost_halt_rate) or not 0.0 <= args.boost_halt_rate <= 1.0
    ):
        raise SystemExit("--boost-halt-rate must be finite and in [0, 1]")
    if args.boost_halt_low_length_rate is not None and (
        not math.isfinite(args.boost_halt_low_length_rate)
        or not 0.0 <= args.boost_halt_low_length_rate <= 1.0
    ):
        raise SystemExit("--boost-halt-low-length-rate must be finite and in [0, 1]")
    if not math.isfinite(args.return_contrast_epsilon) or args.return_contrast_epsilon < 0.0:
        raise SystemExit("--return-contrast-epsilon must be finite and >= 0")
    if args.return_frac > 0.0:
        if args.return_bank is None:
            raise SystemExit("--return-frac needs --return-bank")
        if not args.actors:
            raise SystemExit("--return-frac needs the multi-actor trainer (--actors >= 1)")
        if args.obs_version != "bv8":
            raise SystemExit("--return-frac needs --obs-version bv8")
        if args.reward != "mass" or args.kill_scale != 0.0 or args.death_value not in (None, -3.0):
            raise SystemExit("--return-frac needs --reward mass --kill-scale 0 and death value -3")
        if args.tight_cost or args.death_mass_mult:
            raise SystemExit("--return-frac is incompatible with M6 reward overlays")
        if args.compile_learner != "off":
            raise SystemExit("--return-frac currently needs no --compile-learner")
    elif args.return_mode != "level":
        raise SystemExit("--return-mode contrast needs --return-frac > 0")
    elif args.return_bank is not None:
        raise SystemExit("--return-bank needs --return-frac > 0")
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
        or args.update_every != 1
        or args.gather_threads != 0
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
    if (reason := arm_halt_reason(args)) is not None:
        print(f"not launching: {reason}", file=sys.stderr)
        return 2
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
        "greedy_boost_length_decisions": np.zeros(4, dtype=np.int64),
        "greedy_boost_length_boosts": np.zeros(4, dtype=np.int64),
        "deaths": 0,
        "action_window": deque(maxlen=100_000),
    }
    state["halt"] = arm_halt_reason(args)
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
    sync_cp = None
    sync_controls: Dict[str, Any] = {}
    if args.full_state_every or args.resume_full_state is not None:
        from research.redesign_m3c_20261008.sync_checkpoint import SyncCheckpointSession

        sync_cp = SyncCheckpointSession(
            args, torch, online, target, opt, rng, state, sched, replay, ring, demos, log_path
        )
        sync_controls = dict(sync_cp.controls)

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

    next_log = sync_controls.get("next_log", 100_000)
    next_ckpt = sync_controls.get("next_ckpt", args.checkpoint_every)
    segment = sync_controls.get("segment", 0)
    n_step = N_STEP
    while state["transitions"] < args.transitions and state["halt"] is None:
        problems = guard_ok()
        if problems:
            state["flags"].append({"t": state["transitions"], "guard": problems})
            time.sleep(60)
            continue
        gamma_seg = sync_controls.pop("gamma_seg", sched.gamma(state["transitions"]))
        sim, runtime, policies, groups = build_segment(args, ctx, segment, gamma_seg)
        E = sim.E
        pend: List[deque] = [deque() for _ in range(E)]  # per world: (obs_id, a, reward)
        last_obs = np.full(E, -1, dtype=np.int64)
        if sync_cp is not None:
            pend, last_obs = sync_cp.restore_segment(sim, runtime, pend, last_obs, policies)
        frame_start = int(sync_controls.pop("next_frame", 0)) if sync_controls else 0

        def flush(e: int, bootstrap: int, gamma_now: float) -> None:
            while pend[e]:
                R, g = 0.0, 1.0
                for _, _, r in pend[e]:
                    R += g * r
                    g *= gamma_now
                s0, a0, _ = pend[e].popleft()
                replay.add(s0, a0, R, 0.0 if bootstrap < 0 else g, bootstrap)
                state["since_update"] += 1

        for _frame in range(frame_start, args.segment_frames):
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
                    chosen.update(
                        hero=hero,
                        obs=obs,
                        mask=m,
                        act=act,
                        greedy=greedy,
                        length=np.asarray(prepared.length[hero[:, 0], 0], dtype=np.int64),
                    )
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
                bucket_decisions, bucket_boosts = boost_bucket_counts(
                    chosen["length"], chosen["greedy"]
                )
                state["greedy_boost_length_decisions"] += bucket_decisions
                state["greedy_boost_length_boosts"] += bucket_boosts
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
            if sync_cp is not None:
                sync_cp.capture(
                    sim,
                    runtime,
                    pend,
                    last_obs,
                    segment,
                    _frame + 1,
                    gamma_seg,
                    next_log,
                    next_ckpt,
                    force=state["halt"] is not None or state["transitions"] >= args.transitions,
                    policies=policies,
                )
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
                    "greedy_boost_length_decisions": dict(
                        zip(BOOST_LENGTH_BUCKETS, state["greedy_boost_length_decisions"].tolist())
                    ),
                    "greedy_boost_rate_by_length": boost_bucket_rates(
                        state["greedy_boost_length_decisions"], state["greedy_boost_length_boosts"]
                    ),
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
                if (reason := boost_halt_reason(args, row, state["transitions"])) is None:
                    reason = arm_halt_reason(args)
                if reason is not None:
                    state["halt"] = reason
                    row["halt"] = state["halt"]
                    write_arm_halt_receipt(args, reason, state["transitions"])
                state.update(
                    td=[], anc=[], qmax=0.0, boost=0, greedy_boost=0, decisions=0, deaths=0
                )
                state["greedy_boost_length_decisions"] = np.zeros(4, dtype=np.int64)
                state["greedy_boost_length_boosts"] = np.zeros(4, dtype=np.int64)
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

    if sync_cp is not None:
        sync_cp.close()

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
    if sync_cp is not None:
        summary.update(sync_cp.summary())
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
            **{
                key: row[key]
                for key in (
                    "learner_decisions",
                    "replay_rows",
                    "updates",
                    "per_policy_live_rows",
                    "policy_lag_mean",
                    "policy_lag_p95",
                    "frozen_forward_gpu_s",
                )
                if key in row
            },
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


class ActionWindow:
    """Fixed-size action history with O(message-size) running action counts.

    The old learner loop used ``deque.extend(acts.tolist())`` followed by a full
    ``np.asarray``/``bincount`` on every pass.  This stores the same last
    ``capacity`` actions, but retains the six counts as rows arrive.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.values = np.zeros(capacity, dtype=np.int8)
        self.counts = np.zeros(6, dtype=np.int64)
        self.size = 0
        self.start = 0

    def extend(self, acts: np.ndarray) -> None:
        acts = np.asarray(acts, dtype=np.int8).reshape(-1)
        if not len(acts):
            return
        if len(acts) >= self.capacity:
            acts = acts[-self.capacity :]
            self.values[:] = acts
            self.counts[:] = np.bincount(acts, minlength=6)
            self.size = self.capacity
            self.start = 0
            return
        evict = max(0, self.size + len(acts) - self.capacity)
        if evict:
            old = (self.start + np.arange(evict)) % self.capacity
            self.counts -= np.bincount(self.values[old], minlength=6)
            self.start = (self.start + evict) % self.capacity
            self.size -= evict
        put = (self.start + self.size + np.arange(len(acts))) % self.capacity
        self.values[put] = acts
        self.counts += np.bincount(acts, minlength=6)
        self.size += len(acts)

    @property
    def full(self) -> bool:
        return self.size == self.capacity


def _put_until_stopped(tx, msg: Dict[str, Any], stop_evt, timeout: float = 0.1) -> bool:
    """Put one actor message, returning promptly once the learner requests shutdown."""
    while not stop_evt.is_set():
        try:
            tx.put(msg, timeout=timeout)
            return True
        except queue.Full:
            pass
    return False


def _drain_queues_while_joining(procs, txs, deadline_seconds: float = 30.0) -> None:
    """Let stopped actors finish even when bounded queues were full at shutdown."""
    deadline = time.monotonic() + deadline_seconds
    while True:
        for tx in txs:
            try:
                # A live producer can refill faster than get_nowait().  Bound one pass so
                # deadline checks and joins still make progress.
                for _ in range(64):
                    tx.get_nowait()
            except queue.Empty:
                pass
        alive = [proc for proc in procs if proc.is_alive()]
        if not alive or time.monotonic() >= deadline:
            return
        wait = min(0.05, max(0.0, deadline - time.monotonic()) / len(alive))
        for proc in alive:
            proc.join(timeout=wait)


def actor_main(
    j: int, cfg: Dict[str, Any], names: Dict[str, str], tx, stop_evt, control=None
) -> None:
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
            from research.redesign_m3c_20261008.obs_pack import PackLayout, layout_for_arena

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
        league_n = learner_seats(args)
        task_aligned_n1 = getattr(args, "collection_profile", "legacy") == "task-aligned-n1"
        # Keep the old paths literally selected for N=1 and N=6.  In particular the
        # latter must retain its old observation/ring/action RNG ordering.
        all_snakes = league_n == 6
        mixed_seats = league_n == 3
        hero_fwd = None  # perf-compile (--compile-actor): the compiled, padded hero forward
        if getattr(args, "compile_actor", "off") != "off":
            from research.redesign_m3c_20261008.compile_m3c import (
                HeroForward,
                limit_compile_workers,
            )

            inductor = limit_compile_workers("actor")
            hero_fwd = HeroForward(
                torch,
                net,
                device,
                args.compile_actor,
                args.worlds * (6 if all_snakes else league_n),
                nscal_for(args),
                LOCAL,
                GLOBAL,
            )
            msg = {"actor": j, "compile_actor": hero_fwd.status(), "inductor": inductor}
            print(json.dumps(msg), flush=True)
        rng = np.random.default_rng([args.seed, 17, 1000 + j])
        sched = schedules_from_args(args)
        ctx = gid._context(1)
        B = EgoRasterConfig(version="b")
        mass = args.reward == "mass"
        restart_enabled = getattr(args, "restart_frac", 0.0) > 0.0
        restart_telemetry = None
        if restart_enabled:
            from research.t4_snapshot_restart_20261016.actor import RestartActorTelemetry

            restart_telemetry = RestartActorTelemetry()
        m6 = M6Levers.from_args(args)  # None: every M6 lever off (the previous actor)
        if m6 is not None and m6.tight and not bv8:
            raise ValueError("the M6 tight predicate needs --obs-version bv8")
        use_v8 = args.shield != "none" or args.onpolicy_labels
        # bv8 obtains its advisory mask and features directly from GridBatchSim; the all-slot
        # learner path has no vector61 opponents, so it need not featurize unused runtime rows.
        uses_runtime = not all_snakes or (use_v8 and not bv8)
        carrier = None
        if use_v8 and not bv8:
            from research.redesign_m2_20261008 import ni_spec

            carrier = ctx["lookup"][ni_spec.FRP3_S12[1]][1]
        fast = getattr(args, "actor_impl", "vectorized") == "vectorized"
        sample = sample_legal if fast else sample_legal_ref
        opp_forward = (
            getattr(args, "opp_forward", None) or os.environ.get(OPP_FORWARD_ENV) or "per-group"
        )
        if opp_forward not in OPP_FORWARDS:
            raise ValueError(f"opp forward must be one of {OPP_FORWARDS}, got {opp_forward!r}")
        actor_resume = cfg.get("resume_actor_states", {}).get(str(j))
        actor_resume_extra = {}
        if actor_resume is not None:
            from research.redesign_m3c_20261008.actor_snapshot import actor_snapshot_extra

            actor_resume_extra = actor_snapshot_extra(actor_resume)
        segment = int(actor_resume["segment"]) if actor_resume is not None else j
        resume_applied = False
        while not stop_evt.is_set():
            t_glob = int(progress[0])
            gamma_seg = (
                actor_resume_extra.get("gamma_seg", sched.gamma(t_glob))
                if actor_resume is not None and not resume_applied
                else sched.gamma(t_glob)
            )
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
            task_pending = None
            if task_aligned_n1:
                task_pending = {
                    "world_resets": {
                        mix: int((sim._task_mixes == mix).sum())
                        for mix in ("scripted", "frozen", "mixed")
                    },
                    "death_resets": {mix: 0 for mix in ("scripted", "frozen", "mixed")},
                    "horizon_resets": {mix: 0 for mix in ("scripted", "frozen", "mixed")},
                    "world_frames": {mix: 0 for mix in ("scripted", "frozen", "mixed")},
                    "learner_decisions": {mix: 0 for mix in ("scripted", "frozen", "mixed")},
                }
            live_seats = learner_seat_mask(args.seed, segment, sim.E, league_n)
            pool = None
            pool_identity_records = None
            if mixed_seats:
                # The static pool owns checkpoint identity validation and frozen dispatch.
                # It is constructed after the segment runtime exists, but before the first
                # frame, so seat ownership remains pinned through respawns.
                from research.redesign_m3c_20261008.league_pool import build_default_static_pool

                pool = build_default_static_pool(args, ctx, sim, runtime, device)
                pool_identity_records = pool.identity_records()
                frozen = np.empty((sim.E, sim.S), dtype=object)
                policy_ids = tuple(pool.policy_ids)
                if not policy_ids:
                    raise RuntimeError("static pool has no eligible frozen policies")
                pool_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, 19_018, segment])
                )
                for e in range(sim.E):
                    for s in range(sim.S):
                        if not live_seats[e, s]:
                            frozen[e, s] = policy_ids[int(pool_rng.integers(len(policy_ids)))]
                pool_seats = frozen
                # Only vector61 pool rows need the carried state/trend runtime.
                # B8 builds its raster straight from the maintained grid and the
                # anchors use only grid/mask inputs.  The seat map is immutable
                # within this segment, including through respawns.
                vector_ids = tuple(
                    identity.policy_id
                    for identity in pool.identities
                    if identity.kind == "vector61"
                )
                runtime.controlled[...] = np.isin(pool_seats, vector_ids)
            else:
                pool_seats = None
            restarted_worlds = getattr(sim, "_restart_worlds", None)
            if restart_enabled:
                if restarted_worlds is None:
                    raise RuntimeError("restart-enabled segment did not provide its world mask")
                restarted_worlds = np.asarray(restarted_worlds, dtype=bool)
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
            # Default hero-only rows preserve the previous E-world builder exactly.  The
            # opt-in self-play mode keys the same builder by flattened (env, slot) rows so
            # each driven snake owns an independent n-step window.
            n_driven = E * sim.S if all_snakes else E * league_n
            nsb = NStepBuilder(n_driven) if fast else NStepBuilderRef(n_driven)
            frame_start = 0
            if actor_resume is not None and not resume_applied:
                from research.redesign_m3c_20261008.actor_snapshot import restore_actor_state

                restored = restore_actor_state(actor_resume, sim, runtime, nsb, rng, net)
                if int(restored["segment"]) != segment:
                    raise RuntimeError("actor snapshot segment mismatch")
                frame_start = int(restored["frame"])
                last = int(restored["last"])
                if mixed_seats:
                    live_seats = restored["live_seats"]
                    pool_seats = restored["pool_seats"]
                    extra = restored.get("extra", {})
                    if extra.get("pool_identities") not in (None, pool_identity_records):
                        raise RuntimeError("actor snapshot static-pool identities differ")
                from research.redesign_m3c_20261008.execution_snapshot import (
                    restore_actor_execution,
                    restore_policy_execution,
                )

                extra = restored["extra"]
                restore_actor_execution(hero_fwd, extra.get("execution"))
                restore_policy_execution(policies, extra.get("policies", {}))
                restore_policy_execution({"shield": shield}, extra.get("shield", {"shield": {}}))
                if pool is not None:
                    restore_policy_execution(pool.policies, extra["pool_policies"])
                    pool_rng.bit_generator.state = extra["pool_rng"]
                    vector_ids = tuple(
                        identity.policy_id
                        for identity in pool.identities
                        if identity.kind == "vector61"
                    )
                    runtime.controlled[...] = np.isin(pool_seats, vector_ids)
                if restart_telemetry is not None and extra.get("restart_telemetry") is not None:
                    vars(restart_telemetry).update(extra["restart_telemetry"])
                if not cfg.get("resume_with_full_replay", True):
                    # Pending n-step IDs refer to deliberately omitted ring rows.
                    nsb = NStepBuilder(n_driven) if fast else NStepBuilderRef(n_driven)
                resume_applied = True
            # The task-aligned collector has per-world H5000 clocks.  Its actor
            # segment is only an implementation boundary, never an episode
            # boundary: a world that reset after a terminal hero death must get
            # a complete new H5000 horizon.
            frame_iter = (
                itertools.count(frame_start)
                if task_aligned_n1
                else range(frame_start, args.segment_frames)
            )
            for _frame in frame_iter:
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
                # Fresh worlds begin at frame 0. Restored worlds share their source
                # clock and stop at this segment's common absolute frame boundary.
                active = (
                    np.asarray(sim.frame < args.segment_frames, dtype=bool)
                    if restart_enabled
                    else None
                )
                if restart_enabled and not active.any():
                    break
                t_glob = int(progress[0])
                eps, gamma_now = sched.eps(t_glob), sched.gamma(t_glob)
                if task_aligned_n1:
                    apply_task_reward_gamma(sim, gamma_now)
                chosen: Dict[str, Any] = {}
                timing = {"featurizer_s": 0.0, "forward_s": 0.0, "opponents_s": 0.0, "v8_s": 0.0}
                frozen_live_rows: Dict[str, int] = {}
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
                    nonlocal frozen_live_rows
                    masks = prepared.get_resolved_action_mask()
                    alive = prepared.get_alive()
                    if active is not None:
                        alive = alive.copy()
                        alive[~active] = False
                    actions = np.ones((E, prepared.S), dtype=np.int64)
                    if uses_runtime:
                        runtime.prepare(prepared)
                    hero = (
                        np.argwhere(alive)
                        if all_snakes
                        else (
                            np.argwhere(alive & live_seats)
                            if mixed_seats
                            else np.argwhere(alive[:, :1])
                        )
                    )
                    # vectorized: the opponents (no actor RNG, read-only runtime state, disjoint
                    # action cells) run while an asynchronous (MPS / CUDA) hero forward is in
                    # flight; the hero's decisions and RNG draws are unchanged.
                    overlap = fast and len(hero) > 0 and not all_snakes and not mixed_seats
                    if mixed_seats and pool is not None and pool_seats is not None and len(hero):
                        t_shared_raster = time.perf_counter()
                        pooled_hero_obs = pool.prime_b8_with_learner_rows(
                            prepared, hero, alive, pool_seats
                        )
                        # This replaces the learner's raster gather below, so it
                        # belongs to the existing actor featurizer accounting.
                        timing["featurizer_s"] += time.perf_counter() - t_shared_raster
                    else:
                        pooled_hero_obs = None
                    if len(hero):
                        t_f = time.perf_counter()
                        obs = (
                            pooled_hero_obs
                            if pooled_hero_obs is not None
                            else build_ego_raster(prepared, B, backend="numba", rows=hero)
                        )
                        m = masks[hero[:, 0], hero[:, 1]]
                        mm = m.copy()
                        mm[~mm.any(axis=1), :3] = True  # the normal-only fallback
                        dt_v8 = 0.0
                        if bv8:
                            # Every stored hero observation gets its v8 features here, from
                            # the same selection mask the shield / labels decide with.
                            t_v = time.perf_counter()
                            sel_mask = hero_selection_mask(prepared, hero)
                            if mixed_seats and pool is not None and pool_seats is not None:
                                v8s, v8res = pool.v8_with_learner_rows(
                                    prepared, hero, sel_mask, alive, pool_seats
                                )
                            else:
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
                            if bv8:  # on-policy v8 labels for every learner-driven row
                                a_v8 = v8_redecide(v8res, qn, sel_mask).action
                            else:
                                a_v8 = shield.act(qn, m, prepared, hero)
                            timing["v8_s"] += time.perf_counter() - t_v
                            chosen["a_v8"] = a_v8
                            if p_shield > 0.0:
                                shielded = rng.random(len(hero)) < p_shield
                                policy_act = np.where(shielded, a_v8, greedy)
                        act = np.where(explore, rand, policy_act)
                        actions[hero[:, 0], hero[:, 1]] = act
                        chosen.update(
                            hero=hero,
                            obs=obs,
                            mask=m,
                            act=act,
                            greedy=greedy,
                            length=np.asarray(
                                prepared.length[hero[:, 0], hero[:, 1]], dtype=np.int64
                            ),
                        )
                        if m6 is not None:  # M6: decision-time length (and tightness)
                            L_dec = np.asarray(
                                prepared.length[hero[:, 0], hero[:, 1]], dtype=np.int64
                            )
                            chosen["L"] = L_dec
                            if m6.tight:
                                chosen["tight"] = tight_predicate(v8res.area, sel_mask, L_dec)
                    if mixed_seats:
                        if pool is None or pool_seats is None:
                            raise RuntimeError("mixed seats require a static frozen pool")
                        pool_timing = pool.dispatch(prepared, masks, alive, actions, pool_seats)
                        frozen_live_rows = dict(pool_timing.pop("frozen_live_rows", {}))
                        timing.update(pool_timing)
                    elif not overlap and not all_snakes:
                        opponents(prepared, masks, alive, actions)
                    return actions

                if restart_enabled:
                    sim.step_with_policy(choose, active_env_mask=active)
                else:
                    sim.step_with_policy(choose)
                if uses_runtime:
                    runtime.observe_step(sim)
                reward = sim.get_reward()
                done = sim.get_done()
                if mass:
                    alive_after = sim.get_alive() & ~done
                    reward = reward + mass_term(
                        sim.get_lengths(),
                        alive_after,
                        gamma_now if task_aligned_n1 else gamma_seg,
                        args.mass_kappa,
                    )
                m6_stats: Dict[str, int] = {}
                if m6 is not None:
                    if chosen:
                        hw = chosen["hero"][:, 0]
                        hs = chosen["hero"][:, 1]
                        delta, m6_stats = m6.frame(
                            chosen["L"], chosen.get("tight"), done[hw, hs], gamma_seg
                        )
                        if delta is not None:
                            reward = np.array(reward, dtype=np.float64)
                            reward[hw, hs] = reward[hw, hs] + delta
                    else:
                        m6_stats = m6.empty_stats()
                deaths = 0
                stats = {
                    "decisions": 0,
                    "boost": 0,
                    "greedy_boost": 0,
                    "greedy_boost_length_decisions": np.zeros(4, dtype=np.int64),
                    "greedy_boost_length_boosts": np.zeros(4, dtype=np.int64),
                    "v8_override": 0,
                }
                acts = np.zeros(0, dtype=np.int64)
                lab_ids = np.zeros(0, dtype=np.int64)
                lab_a = np.zeros(0, dtype=np.int8)
                if chosen:
                    ids = ring.add(chosen["obs"], chosen["mask"])
                    if "a_v8" in chosen and args.onpolicy_labels:
                        lab_ids, lab_a = ids, chosen["a_v8"].astype(np.int8)
                    rows = chosen["hero"]
                    driven = (
                        rows[:, 0] * sim.S + rows[:, 1]
                        if all_snakes
                        else (
                            mixed_stream_ids(rows, live_seats, league_n)
                            if mixed_seats
                            else rows[:, 0]
                        )
                    )
                    flat_reward = reward.reshape(-1) if all_snakes else reward[:, 0]
                    flat_done = done.reshape(-1) if all_snakes else done[:, 0]
                    # NStepBuilder indexes rewards/dones by its supplied stream ids.  Mixed
                    # seat streams therefore receive a dense E*N view rather than pool rows.
                    if mixed_seats:
                        dense_reward = reward[live_seats]
                        dense_done = done[live_seats]
                    else:
                        dense_reward = flat_reward
                        dense_done = flat_done
                    deaths = nsb.step(
                        driven, ids, chosen["act"], dense_reward, dense_done, gamma_now
                    )
                    if restart_telemetry is not None:
                        restart_telemetry.record(
                            chosen["hero"][:, 0],
                            chosen["length"],
                            done[chosen["hero"][:, 0], chosen["hero"][:, 1]],
                            restarted_worlds,
                        )
                    acts = chosen["act"]
                    stats = {
                        "decisions": int(len(acts)),
                        "boost": int((acts >= 3).sum()),
                        "greedy_boost": int((chosen["greedy"] >= 3).sum()),
                        "greedy_boost_length_decisions": boost_bucket_counts(
                            chosen["length"], chosen["greedy"]
                        )[0],
                        "greedy_boost_length_boosts": boost_bucket_counts(
                            chosen["length"], chosen["greedy"]
                        )[1],
                        "v8_override": (
                            int((chosen["a_v8"] != chosen["greedy"]).sum())
                            if "a_v8" in chosen
                            else 0
                        ),
                    }
                    if task_pending is not None:
                        for mix in sim._task_mixes[rows[:, 0]]:
                            task_pending["learner_decisions"][str(mix)] += 1
                if task_pending is not None:
                    for mix in sim._task_mixes:
                        task_pending["world_frames"][str(mix)] += 1
                last_frame = (not task_aligned_n1) and _frame == args.segment_frames - 1
                if restart_enabled:
                    expired = active & (sim.frame >= args.segment_frames)
                    if expired.any():
                        expired_rows = np.flatnonzero(expired)
                        if all_snakes:
                            expired_rows = (
                                expired_rows[:, None] * sim.S + np.arange(sim.S)
                            ).reshape(-1)
                        nsb.end_worlds(expired_rows, sched.gamma(int(progress[0])))
                elif last_frame:
                    nsb.end_segment(sched.gamma(int(progress[0])))
                if task_aligned_n1:
                    # ``sim.frame`` is per environment and was incremented by
                    # the just-completed step.  A live hero at H5000 gets the
                    # established truncation bootstrap, unlike a death which
                    # was terminal-flushed by ``step`` above.
                    horizon_worlds = np.asarray(sim.frame >= 5_000, dtype=bool)
                    if horizon_worlds.any():
                        nsb.end_worlds(np.flatnonzero(horizon_worlds), gamma_now)
                s, a, R, d, nx = nsb.take()
                live_rows_by_policy = {"learner": int(len(acts)), **frozen_live_rows}
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
                if getattr(args, "learner_seats", None) is not None:
                    msg.update(
                        replay_rows=int(len(s)),
                        live_rows_by_policy=live_rows_by_policy,
                        pool_identities=pool_identity_records,
                    )
                if restart_telemetry is not None:
                    msg.update(restart_telemetry.take())
                    msg["restart_frac_actual"] = float(restarted_worlds.mean())
                if prof is not None and prof.running:
                    _profile_count(prof.meta, msg)
                if task_aligned_n1:
                    # A hero death ends its episode.  The n-step builder has already
                    # terminal-flushed that stream above; only then may this whole
                    # six-snake world reset.  Non-death worlds continue to H5000,
                    # where the existing end-segment bootstrap remains authoritative.
                    reset_worlds = np.asarray(done[:, 0], dtype=bool) | np.asarray(
                        sim.frame >= 5_000, dtype=bool
                    )
                    if reset_worlds.any():
                        from research.redesign_m3c_20261008.task_collection import (
                            mix_for_reset,
                            reset_seed,
                        )

                        selected = np.flatnonzero(reset_worlds)
                        death_worlds = np.asarray(done[:, 0], dtype=bool)
                        horizon_only = reset_worlds & ~death_worlds
                        for e in selected:
                            mix = str(sim._task_mixes[e])
                            task_pending["world_resets"][mix] += 1
                            task_pending["death_resets" if death_worlds[e] else "horizon_resets"][
                                mix
                            ] += 1
                        sim._task_reset_ordinals[selected] += 1
                        replacement_seeds = [
                            reset_seed(
                                args.namespace,
                                args.seed,
                                segment,
                                int(e),
                                int(sim._task_reset_ordinals[e]),
                            )
                            for e in selected
                        ]
                        for e in selected:
                            sim._task_mixes[e] = mix_for_reset(
                                args.seed,
                                segment,
                                int(e),
                                int(sim._task_reset_ordinals[e]),
                            )
                        reset_task_worlds(sim, reset_worlds, replacement_seeds)
                        reset_rows = np.repeat(reset_worlds[:, None], sim.S, axis=1)
                        runtime.memory.reset_where(reset_rows)
                        runtime._carry_valid[reset_worlds] = False
                        runtime._alive_post[reset_worlds] = False
                        refresh_task_aligned_rosters(
                            args, ctx, sim, runtime, policies, groups, sim._task_mixes
                        )
                if task_pending is not None:
                    msg["task_collection"] = task_pending
                    task_pending = {
                        key: {mix: 0 for mix in ("scripted", "frozen", "mixed")}
                        for key in task_pending
                    }
                _put_until_stopped(tx, msg, stop_evt)  # bounded: back-pressure from the learner
                # A FIFO marker is emitted *after* the ordinary frame message.  The learner
                # drains through one marker from every actor before copying shared rings,
                # so replay ids and actor n-step state describe the same boundary.
                request = None
                if control is not None:
                    try:
                        request = control.get_nowait()
                    except queue.Empty:
                        pass
                if request is not None and request.get("op") == "snapshot":
                    from research.redesign_m3c_20261008.actor_snapshot import capture_actor_state
                    from research.redesign_m3c_20261008.execution_snapshot import (
                        capture_actor_execution,
                        capture_policy_execution,
                    )

                    snapshot = capture_actor_state(
                        sim,
                        runtime,
                        nsb,
                        rng,
                        net,
                        segment,
                        _frame + 1,
                        last,
                        live_seats=live_seats if mixed_seats else None,
                        pool_seats=pool_seats if mixed_seats else None,
                        extra={
                            "pool_identities": pool_identity_records,
                            "weights_version": last,
                            "gamma_seg": gamma_seg,
                            "execution": capture_actor_execution(hero_fwd),
                            "policies": capture_policy_execution(policies),
                            "shield": capture_policy_execution({"shield": shield}),
                            "pool_policies": (
                                capture_policy_execution(pool.policies) if pool else None
                            ),
                            "pool_rng": pool_rng.bit_generator.state if pool else None,
                            "restart_telemetry": (
                                vars(restart_telemetry) if restart_telemetry else None
                            ),
                        },
                    )
                    _put_until_stopped(
                        tx,
                        {"kind": "snapshot", "j": j, "id": request["id"], "state": snapshot},
                        stop_evt,
                    )
                    # Hold this actor at the frame boundary.  The learner drains all FIFO
                    # markers, copies the shared rings, then releases every actor.  This is
                    # deliberately a checkpoint-only barrier; Drive serialization happens
                    # after release on its own worker.
                    while not stop_evt.is_set():
                        try:
                            release = control.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if release.get("op") == "release" and release.get("id") == request["id"]:
                            break
            segment += K
    finally:
        if prof is not None:
            prof.safe_dump({"worlds": int(args.worlds), "actor": j})
        shm_ring.close()
        shm_w.close()
        shm_p.close()


def stop_actors(
    stop_evt, txs, procs, join_seconds: float = 30.0, kill_seconds: float = 10.0
) -> None:
    """Set ``stop_evt`` and keep draining the actor queues until every actor has exited.

    A single drain is not enough: an actor whose queue is full (back-pressure, the normal
    state of a learner-bound run) hands its last message to the queue's feeder thread, and
    ``multiprocessing`` joins that thread at process exit, so the actor blocks until the
    learner reads again. With one drain before ``join`` the actor was terminated after the
    timeout and reported as not having exited cleanly (``actor_exitcodes`` -15).
    """
    stop_evt.set()
    _drain_queues_while_joining(procs, txs, deadline_seconds=join_seconds)
    # Signal all stragglers before joining under one shared termination deadline.
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    deadline = time.monotonic() + float(kill_seconds)
    for proc in procs:
        proc.join(timeout=max(0.0, deadline - time.monotonic()))


def _learner_counters(st: Dict[str, Any]) -> Dict[str, Any]:
    """Learner counters bracketing a profiled window (``M3C_PROFILE``; never called when off)."""
    keys = ("transitions", "updates", "learn_s", "wait_s", "env_s")
    return {**{k: st[k] for k in keys}, "t": time.perf_counter()}


def _tick_learner_profile(prof, st: Dict[str, Any]) -> None:
    """Bracket the active profile only, including a duration-triggered stop."""
    was_running = prof.running
    if prof.tick():
        prof.meta["start"] = _learner_counters(st)
    elif was_running and not prof.running:
        prof.meta["end"] = _learner_counters(st)


def main_multi(args) -> int:
    """Learner process with ``args.actors`` actor processes (see the module docstring)."""
    import multiprocessing as mp

    import torch

    from research.redesign_m2_20261008.gen_data import guard_ok

    resume_payload = None
    resume_manifest = None
    if args.resume_full_state is not None:
        from research.redesign_m3c_20261008.async_checkpoint import load_latest_snapshot

        # Verification happens before allocating GPUs/shared memory or launching actors.
        resume_payload, resume_manifest = load_latest_snapshot(args.resume_full_state)
        if resume_payload.get("extra", {}).get("recipe") != full_state_recipe(args):
            raise ValueError("full-state multi-actor training recipe differs")
        if resume_payload.get("extra", {}).get("trainer") != "multi":
            raise ValueError("multi-actor resume requires a multi-actor snapshot")

    if args.no_guard:

        def guard_ok():  # noqa: F811
            return []

    # Validate the immutable bank before output allocation, shared memory, or actors.
    # Keep this whole branch absent for default runs so their startup/RNG path is unchanged.
    bank = None
    outside_bank = None
    return_bank = None
    if args.label_bank_frac > 0.0:
        from research.t3_rollout_labels_20261015.bank import LabelBank

        bank_pack = None
        if args.obs_pack:
            from research.redesign_m3c_20261008.obs_pack import train_layout

            bank_pack = train_layout()
        strict_shards = args.label_bank_outside is not None
        bank = LabelBank(
            Path(args.label_bank), nscal_for(args), pack=bank_pack, strict_shards=strict_shards
        )
        if args.label_bank_counter_ratio is not None:
            bank.validate_counter_ratio(args.label_bank_counter_ratio)
        if args.label_bank_outside is not None:
            outside_bank = LabelBank(
                Path(args.label_bank_outside),
                nscal_for(args),
                pack=bank_pack,
                strict_shards=True,
            )
    if args.return_frac > 0.0:
        from research.t4d_return_regression_20261016.bank import ReturnBank

        return_bank = ReturnBank(Path(args.return_bank), nscal_for(args))
        if args.gamma_start != return_bank.gamma or args.gamma_end != return_bank.gamma:
            raise ValueError("return-bank gamma must equal both --gamma-start and --gamma-end")
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
    if resume_payload is not None and resume_payload.get("schedules") != sched.__dict__:
        raise ValueError("full-state schedule recipe differs")

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
    from research.redesign_m3c_20261008.learner_m3c import Learner, LearnerOpts, make_optimizer

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
    if resume_payload is not None and demos is not None:
        from research.redesign_m3c_20261008.demo_snapshot import restore_demo_state

        demo_state = resume_payload.get("extra", {}).get("demos")
        if demo_state is not None:
            restore_demo_state(demos, demo_state)
    replay = Replay(args.replay)
    labels = LabelBuffer(args.label_capacity) if args.onpolicy_labels else None
    tel = Telemetry(args.out, args)
    # Rings (ring_capacity): with the defaults (replay 500k, K = 2, 64 worlds) 2 x 377k rows x
    # 12.3 KB = 9.3 GB per run (--obs-pack: 3.2 KB per row, 2.4 GB). research/perf_ram_20261013
    driven_worlds = args.worlds * learner_seats(args)
    ring_cap, margin_rows = ring_capacity(args.replay, K, driven_worlds, args.actor_queue_frames)
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
    resumed_with_replay = False
    if resume_payload is not None:
        resumed_with_replay = restore_full_state(
            torch,
            resume_payload,
            online,
            target,
            opt,
            rng,
            replay,
            labels,
            rings,
        )
        if not resumed_with_replay and labels is not None:
            # Label ids point into omitted observation rings.  Treat the continuation as
            # a fresh replay warm-up rather than sampling dangling ids.
            labels.ids.fill(-1)
            labels.size = labels.ptr = 0
    if resume_payload is not None:
        saved_extra = resume_payload.get("extra", {})
        saved_wbuf = saved_extra.get("wbuf")
        if saved_wbuf is not None:
            if saved_wbuf.shape != wbuf.shape:
                raise ValueError("full-state shared weight buffer is incompatible")
            wbuf[...] = saved_wbuf
            wver[0] = int(saved_extra["weights_version"])
        else:
            _publish_weights(online, wbuf, wver)
        if "progress" in saved_extra:
            progress[...] = saved_extra["progress"]
    else:
        _publish_weights(online, wbuf, wver)  # version 2 = publish #1
    multi = MultiRing(rings)

    ctx_mp = mp.get_context("spawn")
    stop_evt = ctx_mp.Event()
    txs = [ctx_mp.Queue(maxsize=args.actor_queue_frames) for _ in range(K)]
    # The protocol permits only a release plus the next request per actor. An
    # unbounded control FIFO avoids partial fan-out when a release is still queued.
    controls = [ctx_mp.Queue() for _ in range(K)] if args.full_state_every else [None] * K
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
    if resume_payload is not None:
        cfg["resume_actor_states"] = {
            str(j): state for j, state in resume_payload.get("extra", {}).get("actors", {}).items()
        }
        cfg["resume_with_full_replay"] = resumed_with_replay
    cfg["args"]["demo_data"] = str(args.demo_data)
    procs = [
        ctx_mp.Process(
            target=actor_main, args=(j, cfg, names, txs[j], stop_evt, controls[j]), daemon=True
        )
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
        "replay_rows": 0,
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
        "greedy_boost_length_decisions": np.zeros(4, dtype=np.int64),
        "greedy_boost_length_boosts": np.zeros(4, dtype=np.int64),
        "deaths": 0,
        "v8_override": 0,
        "label_rows": [],
        "anchor_updates": 0,
        "lag": [],
        "live_rows_by_policy": {},
        "frozen_forward_gpu_s": 0.0,
        "pool_identities": None,
        "task_collection": {
            key: {mix: 0 for mix in ("scripted", "frozen", "mixed")}
            for key in (
                "world_resets",
                "death_resets",
                "horizon_resets",
                "world_frames",
                "learner_decisions",
            )
        },
        "action_window": ActionWindow(),
    }
    resume_event = None
    if resume_payload is not None:
        saved_counters = resume_payload.get("counters", {})
        # Counters are mutable telemetry state, so restore only keys the current trainer owns.
        for key, value in saved_counters.items():
            if key in st:
                st[key] = value
        resume_event = {
            "event": "resume-full-state",
            "source": str(args.resume_full_state),
            "snapshot_transitions": int(resume_manifest.get("transitions", st["transitions"])),
            "replay": "full" if resumed_with_replay else "none",
            "resumed_without_replay": not resumed_with_replay,
        }
        st["flags"] = list(st["flags"]) + [resume_event]
        with log_path.open("a") as stream:
            stream.write(json.dumps(resume_event) + "\n")
        if not resumed_with_replay:
            st["since_update"] = 0
    st["halt"] = arm_halt_reason(args)
    if bank is not None:
        st.update(
            bank_rows=[],
            bank_margin_loss=[],
            bank_agree=[],
            bank_agree_decisive=[],
            bank_agree_counter=[],
            bank_agree_outside=[],
        )
    if return_bank is not None:
        st.update(
            return_rows=[],
            return_loss=[],
            return_contrast_rows=[],
            return_contrast_active_fraction=[],
            return_mean_abs_target_contrast=[],
            return_mean_abs_contrast_error=[],
        )
    if args.restart_frac > 0.0:
        st["restart"] = {"restart_frac_actual": []}
        st["restart_total"] = {
            "restart_decisions": 0,
            "fresh_decisions": 0,
            "restart_decisions_ge800": 0,
            "fresh_decisions_ge800": 0,
            "restart_deaths": 0,
            "fresh_deaths": 0,
            "restart_deaths_ge800": 0,
            "fresh_deaths_ge800": 0,
            "restart_length_sum": 0,
            "fresh_length_sum": 0,
            "restart_death_length_sum": 0,
            "fresh_death_length_sum": 0,
        }
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
    resume_extra = resume_payload.get("extra", {}) if resume_payload is not None else {}
    next_log = int(resume_extra.get("next_log", 100_000))
    next_ckpt = int(resume_extra.get("next_ckpt", args.checkpoint_every))
    next_guard = int(resume_extra.get("next_guard", 0))
    published = int(resume_extra.get("published", 1))  # weights version 2 * published

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
        **({"bank": bank, "outside_bank": outside_bank} if bank is not None else {}),
        **({"return_bank": return_bank} if return_bank is not None else {}),
    )
    if resume_payload is not None:
        from research.redesign_m3c_20261008.execution_snapshot import restore_learner_execution

        restore_learner_execution(learner, resume_payload["extra"]["learner"])

        # Learner/options initialize optional telemetry groups; install every saved
        # counter only after those constructors, retaining its shared st reference.
        st.clear()
        st.update(resume_payload["counters"])
        st["flags"] = list(st["flags"]) + [resume_event]
        if not resumed_with_replay:
            st["since_update"] = 0
        np.random.set_state(resume_payload["rng"]["numpy"])
        _restore_torch_rng_state(torch, resume_payload["rng"]["torch"])

    checkpointer = None
    next_full_state = 0
    full_state_events: List[Dict[str, Any]] = []

    def record_full_state(event: Dict[str, Any]) -> None:
        full_state_events.append(event)
        with log_path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def poll_full_state() -> None:
        for event in checkpointer.poll_events():
            record_full_state(event)

    if args.full_state_every:
        from research.redesign_m3c_20261008.async_checkpoint import AsyncCheckpointer, capture_state

        checkpointer = AsyncCheckpointer(args.ckpt_local_dir, args.ckpt_durable_dir, keep=2)
        next_full_state = ((st["transitions"] // args.full_state_every) + 1) * args.full_state_every

    def submit_full_state(force: bool = False) -> None:
        """Detach at an update boundary then hand off nonblocking persistence.

        A full replay copies the shared observation rings.  That cost scales with the
        configured ring footprint and is recorded rather than hidden behind a universal
        stall claim; model-only / replay-none snapshots are normally small.
        """
        nonlocal next_full_state, actor_snapshot_serial, pending_actor_snapshot
        nonlocal actor_barrier_started, last_snapshot_counters
        if checkpointer is None or (not force and st["transitions"] < next_full_state):
            return
        if not force and demos is not None and getattr(demos, "_thread", None) is not None:
            if demos._thread.is_alive():
                record_full_state(
                    {"event": "full-state-snapshot-deferred", "reason": "demo-prefetch"}
                )
                return
        # The background serializer owns one immutable snapshot at a time.  A terminal
        # checkpoint waits only here, after actors are stopped; normal learner snapshots
        # always skip rather than block.
        if checkpointer.busy:
            if not force or not checkpointer.wait_local(timeout=30):
                record_full_state(
                    {"event": "full-state-snapshot-skipped", "reason": "serializer-busy"}
                )
                return
        if pending_actor_snapshot is None:
            actor_snapshot_serial += 1
            pending_actor_snapshot = actor_snapshot_serial
            actor_barrier_started = time.perf_counter()
            for control in controls:
                control.put_nowait({"op": "snapshot", "id": pending_actor_snapshot})
            return
        frozen = {
            j: msg["state"]
            for j, msg in actor_snapshots.items()
            if msg.get("id") == pending_actor_snapshot
        }
        if len(frozen) != K:
            return
        from research.redesign_m3c_20261008.execution_snapshot import capture_learner_execution

        begin = time.perf_counter()
        learner.flush()  # finish any AMP/pipelined learner update before copying optimizer state
        demo_snapshot = None
        if demos is not None:
            from research.redesign_m3c_20261008.demo_snapshot import capture_demo_state

            demo_snapshot = capture_demo_state(demos)
        payload = full_state_payload(
            torch,
            online,
            target,
            opt,
            rng,
            st.copy(),
            sched,
            replay,
            labels,
            rings,
            args.full_state_replay,
            {
                "trainer": "multi",
                "recipe": full_state_recipe(args),
                "published": published,
                "rr": rr,
                "next_log": next_log,
                "next_ckpt": next_ckpt,
                "next_guard": next_guard,
                "weights_version": int(wver[0]),
                "wbuf": wbuf,
                "progress": progress.copy(),
                "demos": demo_snapshot,
                "actors": frozen,
                "learner": capture_learner_execution(learner),
            },
        )
        detached = capture_state(payload)
        stall_ms = 1000.0 * (time.perf_counter() - begin)
        accepted = checkpointer.submit(
            detached,
            int(st["transitions"]),
            {
                "schema": FULL_STATE_SCHEMA,
                "capture_stall_ms": stall_ms,
                "replay": args.full_state_replay,
            },
        )
        event = {
            "event": "full-state-snapshot",
            "transitions": int(st["transitions"]),
            "capture_stall_ms": stall_ms,
            "accepted": accepted,
            "actor_barrier_ms": 1000.0 * (begin - actor_barrier_started),
            "replay": args.full_state_replay,
        }
        record_full_state(event)
        tel.add("full_state_capture_stall_ms", stall_ms)
        for control in controls:
            control.put({"op": "release", "id": pending_actor_snapshot})
        pending_actor_snapshot = None
        if accepted:
            last_snapshot_counters = (int(st["transitions"]), int(st["updates"]))
        next_full_state = ((st["transitions"] // args.full_state_every) + 1) * args.full_state_every

    code = 0
    rr = int(resume_extra.get("rr", 0))
    attached: set = set()
    stopped_for_time = False
    full_state_complete: Optional[bool] = None
    actor_snapshots: Dict[int, Dict[str, Any]] = {}
    actor_snapshot_serial = 0
    pending_actor_snapshot: Optional[int] = None
    actor_barrier_started = 0.0
    last_snapshot_counters = None
    try:
        while (
            not stopped_for_time and st["transitions"] < args.transitions and st["halt"] is None
        ) or pending_actor_snapshot is not None:
            if (
                getattr(args, "max_wall_seconds", 0.0)
                and time.perf_counter() - started >= args.max_wall_seconds
            ):
                stopped_for_time = True
                if checkpointer is None:
                    break
            if prof is not None:
                _tick_learner_profile(prof, st)
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
                if msg.get("kind") == "snapshot":
                    actor_snapshots[int(msg["j"])] = msg
                    continue
                if j not in attached:
                    attached.add(j)
                    if len(attached) == K:  # every actor mapped its blocks: unlink the names
                        for blk in shms:  # (the mappings live on; nothing leaks on a crash)
                            blk.unlink()
                gs = multi.encode(j, msg["s"])
                gn = multi.encode(j, msg["nxt"])
                replay.add_many(gs, msg["a"], msg["R"], msg["disc"], gn)
                st["since_update"] += len(gs)
                st["replay_rows"] += len(gs)
                if "task_collection" in msg:
                    for key, by_mix in msg["task_collection"].items():
                        for mix, value in by_mix.items():
                            st["task_collection"][key][mix] += int(value)
                live_counts = msg.get("live_rows_by_policy", {"learner": msg["decisions"]})
                for policy_id, count in live_counts.items():
                    st["live_rows_by_policy"][policy_id] = st["live_rows_by_policy"].get(
                        policy_id, 0
                    ) + int(count)
                if msg.get("pool_identities") is not None:
                    if st["pool_identities"] is None:
                        st["pool_identities"] = msg["pool_identities"]
                    elif st["pool_identities"] != msg["pool_identities"]:
                        raise RuntimeError("actor pool identity records disagree")
                if labels is not None and len(msg["lab_ids"]):
                    labels.add(multi.encode(j, msg["lab_ids"]), msg["lab_a"].astype(np.int64))
                st["v8_override"] += msg["v8_override"]
                st["transitions"] += msg["decisions"]
                st["decisions"] += msg["decisions"]
                st["boost"] += msg["boost"]
                st["greedy_boost"] += msg["greedy_boost"]
                st["greedy_boost_length_decisions"] += msg["greedy_boost_length_decisions"]
                st["greedy_boost_length_boosts"] += msg["greedy_boost_length_boosts"]
                st["deaths"] += msg["deaths"]
                if args.restart_frac > 0.0:
                    rst = st["restart"]
                    rst["restart_frac_actual"].append(msg["restart_frac_actual"])
                    for key in st["restart_total"]:
                        rst[key] = rst.get(key, 0) + msg[key]
                        st["restart_total"][key] += msg[key]
                st["env_s"] += msg["env_s"]
                st["lag"].append(published - msg["weights_version"] // 2)
                st["action_window"].extend(msg["acts"])
                for k_, v_ in msg["timing"].items():
                    tel.add(k_, v_)
                    if k_ == "frozen_forward_gpu_s":
                        st["frozen_forward_gpu_s"] += float(v_)
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
            if st["action_window"].full:
                counts = st["action_window"].counts
                if counts.max() > 0.9 * counts.sum():
                    st["halt"] = f"action collapse {counts.tolist()}"
            submit_full_state(
                force=stopped_for_time
                or st["transitions"] >= args.transitions
                or st["halt"] is not None
            )
            if checkpointer is not None:
                poll_full_state()
            if st["transitions"] >= next_log:
                next_log += 100_000
                learner.flush()  # fast learner: apply the pending step, sync device stats
                wall = time.perf_counter() - started
                t = st["transitions"]
                row = {
                    "transitions": t,
                    "updates": st["updates"],
                    "learner_decisions": t,
                    "replay_rows": st["replay_rows"],
                    "replay_rows_per_update": st["replay_rows"] / max(1, st["updates"]),
                    "live_rows_by_policy": dict(st["live_rows_by_policy"]),
                    "per_policy_live_rows": dict(st["live_rows_by_policy"]),
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
                    "greedy_boost_length_decisions": dict(
                        zip(BOOST_LENGTH_BUCKETS, st["greedy_boost_length_decisions"].tolist())
                    ),
                    "greedy_boost_rate_by_length": boost_bucket_rates(
                        st["greedy_boost_length_decisions"], st["greedy_boost_length_boosts"]
                    ),
                    "deaths_per_1k": 1000 * st["deaths"] / max(1, st["decisions"]),
                    "demo_shards_loaded": demos.loaded if demos is not None else 0,
                    "demo_distinct": demos.distinct if demos is not None else 0,
                    "actors": K,
                }
                if m6 is not None:
                    row.update(m6_log_fields(st["m6"], st["decisions"], st["deaths"]))
                    st["m6"] = {}
                if args.restart_frac > 0.0:
                    rst = st["restart"]
                    row.update(
                        restart_frac_actual=(
                            float(np.mean(rst["restart_frac_actual"]))
                            if rst["restart_frac_actual"]
                            else 0.0
                        ),
                        restart_decisions=rst.get("restart_decisions", 0),
                        fresh_decisions=rst.get("fresh_decisions", 0),
                        restart_decisions_ge800=rst.get("restart_decisions_ge800", 0),
                        fresh_decisions_ge800=rst.get("fresh_decisions_ge800", 0),
                        restart_deaths=rst.get("restart_deaths", 0),
                        fresh_deaths=rst.get("fresh_deaths", 0),
                        restart_deaths_ge800=rst.get("restart_deaths_ge800", 0),
                        fresh_deaths_ge800=rst.get("fresh_deaths_ge800", 0),
                        restart_deaths_per_1k_ge800=(
                            per_1k_or_none(
                                rst.get("restart_deaths_ge800", 0),
                                rst.get("restart_decisions_ge800", 0),
                            )
                        ),
                        fresh_deaths_per_1k_ge800=(
                            per_1k_or_none(
                                rst.get("fresh_deaths_ge800", 0),
                                rst.get("fresh_decisions_ge800", 0),
                            )
                        ),
                        restart_length_mean=(
                            rst.get("restart_length_sum", 0)
                            / max(1, rst.get("restart_decisions", 0))
                        ),
                        fresh_length_mean=(
                            rst.get("fresh_length_sum", 0) / max(1, rst.get("fresh_decisions", 0))
                        ),
                        restart_death_length_mean=(
                            rst.get("restart_death_length_sum", 0)
                            / max(1, rst.get("restart_deaths", 0))
                        ),
                        fresh_death_length_mean=(
                            rst.get("fresh_death_length_sum", 0)
                            / max(1, rst.get("fresh_deaths", 0))
                        ),
                    )
                    st["restart"] = {"restart_frac_actual": []}
                if bv8:
                    row["obs_version"] = "bv8"
                    row["label_buffer"] = labels.size if labels is not None else 0
                if lopts.update_every != 1:
                    row["update_every"] = lopts.update_every
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
                    row["bank_agree_decisive"] = (
                        float(np.mean(st["bank_agree_decisive"]))
                        if st["bank_agree_decisive"]
                        else None
                    )
                    row["bank_agree_counter"] = (
                        float(np.mean(st["bank_agree_counter"]))
                        if st["bank_agree_counter"]
                        else None
                    )
                    row["bank_agree_outside"] = (
                        float(np.mean(st["bank_agree_outside"]))
                        if st["bank_agree_outside"]
                        else None
                    )
                if return_bank is not None:
                    row["return_rows_mean"] = (
                        float(np.mean(st["return_rows"])) if st["return_rows"] else 0.0
                    )
                    row["return_loss"] = (
                        float(np.mean(st["return_loss"])) if st["return_loss"] else 0.0
                    )
                    if args.return_mode == "contrast":
                        row["return_contrast_rows_mean"] = (
                            float(np.mean(st["return_contrast_rows"]))
                            if st["return_contrast_rows"]
                            else 0.0
                        )
                        row["return_contrast_active_fraction"] = (
                            float(np.mean(st["return_contrast_active_fraction"]))
                            if st["return_contrast_active_fraction"]
                            else 0.0
                        )
                        row["return_mean_abs_target_contrast"] = (
                            float(np.mean(st["return_mean_abs_target_contrast"]))
                            if st["return_mean_abs_target_contrast"]
                            else 0.0
                        )
                        row["return_mean_abs_contrast_error"] = (
                            float(np.mean(st["return_mean_abs_contrast_error"]))
                            if st["return_mean_abs_contrast_error"]
                            else 0.0
                        )
                        row["return_initial_mean_abs_contrast_error"] = st.get(
                            "return_initial_mean_abs_contrast_error"
                        )
                if args.return_gradient_telemetry:
                    grad_rows = st["return_gradient_telemetry"]
                    ratios = [x["return_to_td_grad_ratio"] for x in grad_rows]
                    row.update(
                        return_to_td_grad_ratio=(
                            float(np.mean(ratios))
                            if ratios and all(x is not None for x in ratios)
                            else None
                        ),
                        weighted_return_grad_norm=(
                            float(np.mean([x["weighted_return_grad_norm"] for x in grad_rows]))
                            if grad_rows
                            else None
                        ),
                        td_grad_norm=(
                            float(np.mean([x["td_grad_norm"] for x in grad_rows]))
                            if grad_rows
                            else None
                        ),
                        anchor_rows=(
                            int(round(np.mean([x["anchor_rows"] for x in grad_rows])))
                            if grad_rows
                            else 0
                        ),
                        return_rows=(
                            int(round(np.mean([x["return_rows"] for x in grad_rows])))
                            if grad_rows
                            else 0
                        ),
                        agent_td_rows=(
                            int(round(np.mean([x["agent_td_rows"] for x in grad_rows])))
                            if grad_rows
                            else 0
                        ),
                        max_abs_q=(
                            float(max(x["max_abs_q"] for x in grad_rows)) if grad_rows else 0.0
                        ),
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
                if (reason := boost_halt_reason(args, row, t)) is None:
                    reason = arm_halt_reason(args)
                if reason is not None:
                    st["halt"] = reason
                    row["halt"] = st["halt"]
                    write_arm_halt_receipt(args, reason, t)
                lag = st["lag"]
                row.update(
                    policy_lag_mean=float(np.mean(lag)) if lag else None,
                    policy_lag_p95=float(np.quantile(lag, 0.95)) if lag else None,
                    frozen_forward_gpu_s=st["frozen_forward_gpu_s"],
                )
                if args.collection_profile == "task-aligned-n1":
                    row["task_collection"] = st["task_collection"]
                tel.write(
                    row,
                    replay_size=replay.size,
                    learner_wait_s=round(st["wait_s"], 3),
                    policy_lag_mean=float(np.mean(lag)) if lag else None,
                    policy_lag_p95=float(np.quantile(lag, 0.95)) if lag else None,
                    policy_lag_histogram=(
                        {str(int(v)): int(c) for v, c in zip(*np.unique(lag, return_counts=True))}
                        if lag
                        else {}
                    ),
                    frozen_forward_gpu_s=st["frozen_forward_gpu_s"],
                    ring_overrun_rows=st["ring_overrun_rows"],
                    queue_depth=[q_.qsize() if sys.platform != "darwin" else None for q_ in txs],
                    ring_counts=[r.count for r in rings],
                    **(
                        {"compile_learner": learner.compile_status()}
                        if lopts.compile != "off"
                        else {}
                    ),
                )
                st.update(
                    td=[],
                    anc=[],
                    qmax=0.0,
                    boost=0,
                    greedy_boost=0,
                    decisions=0,
                    deaths=0,
                )
                st["greedy_boost_length_decisions"] = np.zeros(4, dtype=np.int64)
                st["greedy_boost_length_boosts"] = np.zeros(4, dtype=np.int64)
                st.update(v8_override=0, label_rows=[])
                if bank is not None:
                    st.update(
                        bank_rows=[],
                        bank_margin_loss=[],
                        bank_agree=[],
                        bank_agree_decisive=[],
                        bank_agree_counter=[],
                        bank_agree_outside=[],
                    )
                if return_bank is not None:
                    st.update(
                        return_rows=[],
                        return_loss=[],
                        return_contrast_rows=[],
                        return_contrast_active_fraction=[],
                        return_mean_abs_target_contrast=[],
                        return_mean_abs_contrast_error=[],
                    )
                if args.return_gradient_telemetry:
                    st["return_gradient_telemetry"] = []
                st["lag"] = []
                with log_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                if st["halt"] is not None:
                    break
            if st["transitions"] >= next_ckpt:
                learner.flush()
                checkpoint_path = args.out / f"ckpt_{next_ckpt // 1000}k.pth"
                save_checkpoint_atomic(
                    checkpoint_path,
                    online,
                    {
                        **meta_base,
                        "transitions": st["transitions"],
                        "updates": st["updates"],
                        "pool_identities": st["pool_identities"],
                    },
                    manifest,
                )
                early = run_return_early_warning(args, checkpoint_path, next_ckpt, log_path)
                if early.get("status") != "SKIPPED" and early.get("status") != "PASS":
                    st["halt"] = f"return early warning: {early.get('reason', early.get('status'))}"
                    write_arm_halt_receipt(args, st["halt"], st["transitions"])
                next_ckpt += args.checkpoint_every
    except BaseException as exc:  # noqa: BLE001
        st["halt"] = st["halt"] or f"{type(exc).__name__}: {exc}"
        code = 1
    finally:
        if prof is not None and prof.running:
            prof.meta["end"] = _learner_counters(st)
            prof.stop()
        if checkpointer is not None:
            if not checkpointer.wait_local(timeout=30):
                st["flags"].append({"event": "full-state-final-local-incomplete"})
        stop_actors(stop_evt, txs, procs)
        for blk in shms:  # idempotent; covers a crash before every actor attached
            blk.unlink()
        if checkpointer is not None:
            # A bounded close joins serializer/copy workers; a durable-copy failure is
            # surfaced in the run receipt but must never have interrupted the learner.
            persisted = checkpointer.close(timeout=30)
            # Training is over. Keep this process (and its daemon writers) alive until
            # pending local/Drive I/O finishes; a large final upload may take minutes.
            while checkpointer.workers_alive:
                poll_full_state()
                persisted = checkpointer.close(timeout=30)
            full_state_complete = bool(persisted) and (
                last_snapshot_counters == (int(st["transitions"]), int(st["updates"]))
            )
            if not full_state_complete:
                st["flags"].append({"event": "full-state-close-incomplete"})
            poll_full_state()
    try:
        learner.flush()  # fast learner: the last pipelined step belongs to the final weights
    except Exception as exc:  # noqa: BLE001
        st["halt"] = st["halt"] or f"flush: {type(exc).__name__}: {exc}"
        code = 1
    closer = getattr(learner, "close", None)
    if closer is not None:
        closer()
    save_checkpoint_atomic(
        args.out / ("aborted.pth" if st["halt"] else "final.pth"),
        online,
        {
            **meta_base,
            "transitions": st["transitions"],
            "updates": st["updates"],
            "pool_identities": st["pool_identities"],
        },
        manifest,
    )
    summary = {
        "schema": SCHEMA_M3C,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "transitions": st["transitions"],
        "learner_decisions": st["transitions"],
        "replay_rows": replay.added,
        # ``replay_rows`` is the resident ring size and may be capped.  Q1's
        # endpoint instead reconciles emitted rows against learner-owned action
        # decisions, and records bounded actor-message overshoot explicitly.
        "emitted_replay_rows": st["replay_rows"],
        "learner_decisions_accounted": st["decisions"],
        "pending_update_debt": int(st["since_update"]),
        "decision_replay_delta": int(st["decisions"] - st["replay_rows"]),
        "transition_budget_overshoot": max(0, int(st["transitions"] - args.transitions)),
        "task_collection": (
            st["task_collection"] if args.collection_profile == "task-aligned-n1" else None
        ),
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
        "stopped_for_time": stopped_for_time,
        "ring_overrun_rows": st["ring_overrun_rows"],
        "anchor_updates": st["anchor_updates"],
        "label_buffer": labels.size if labels is not None else 0,
        "pool_identities": st["pool_identities"],
        "full_state_events": full_state_events[-200:],
        "full_state_complete": full_state_complete,
        "resumed_with_full_replay": resumed_with_replay if resume_payload is not None else None,
        "resumed_without_replay": bool(resume_payload is not None and not resumed_with_replay),
    }
    if return_bank is not None:
        totals = st["return_totals"]
        summary["return_regression"] = {
            **totals,
            "mean_loss": totals["loss_sum"] / max(1, totals["updates"]),
        }
    if lopts.compile != "off":
        summary["compile_learner"] = learner.compile_status()
    if m6 is not None:
        tot = st["m6_total"]
        summary["m6_totals"] = {**tot, **m6_log_fields(tot, tot["decisions"], tot["deaths"])}
    if args.restart_frac > 0.0:
        rst = st["restart_total"]
        summary["restart_totals"] = {
            **rst,
            "restart_deaths_per_1k_ge800": per_1k_or_none(
                rst["restart_deaths_ge800"], rst["restart_decisions_ge800"]
            ),
            "fresh_deaths_per_1k_ge800": per_1k_or_none(
                rst["fresh_deaths_ge800"], rst["fresh_decisions_ge800"]
            ),
            "restart_length_mean": rst["restart_length_sum"] / max(1, rst["restart_decisions"]),
            "fresh_length_mean": rst["fresh_length_sum"] / max(1, rst["fresh_decisions"]),
            "restart_death_length_mean": (
                rst["restart_death_length_sum"] / max(1, rst["restart_deaths"])
            ),
            "fresh_death_length_mean": (
                rst["fresh_death_length_sum"] / max(1, rst["fresh_deaths"])
            ),
            "restart_frac_actual": int(round(args.worlds * args.restart_frac)) / args.worlds,
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if prof is not None:
        prof.safe_dump({"actors": K, "worlds": int(args.worlds), "device": str(device)})
    print(json.dumps({k: summary[k] for k in ("transitions", "rate", "halt", "flags")}))
    for s in shms:  # views may still exist: close() tolerates that (names already unlinked)
        s.close()
    return 1 if (st["halt"] or code) else 0


if __name__ == "__main__":
    raise SystemExit(main())
