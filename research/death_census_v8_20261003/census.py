#!/usr/bin/env python3
"""TIER-0 DIAGNOSTIC: death census of the Apex champion + released v8 veto (SIMD engine).

Not a screen, not gate evidence, no candidate and no decision rule. It plays the released
Watch configuration (champion + ``free-space-veto/v8-space-and-head(lambda=8.0)``) on the
SIMD engine (``run_simd_eval(vector61=True, hero_safety_veto="v8",
hero_safety_veto_lambda=8.0, vector61_forward="rowwise")``, H5000 parity 24/24 identical
with the live hook) at H5000, profile ``promotion-v2-watch-rect``, strict balanced rosters,
all three mixes, on the fresh namespace :data:`DOMAIN`, and asks what still kills or
limits v8. It mirrors the v5 census (``research/trap_horizon_20261001/diagnose_live_v5.py``,
``docs/research/death_census_v5_2026-10-02.md``) and reuses its analysis unchanged.

Capture (read-only; nothing the hero sees or does changes):

* **Decisions.** ``Vector61SimdPolicy._apply_veto`` is wrapped (original first, its actions
  returned unchanged). For every hero decision it snapshots, in cell units, the hero body,
  length, heading, boost counter, every other live snake's cells, the food, v2's one-step
  counts/need, the masked Q row, the mask, the base and final action, and v8's decision
  reason, derived from the env's live ``SpaceAndHeadVeto`` counter deltas around the call
  (:func:`reason_from_deltas`; one hero row per env per call). The per-decision wall cost
  is the hook's own ``apply_seconds_total`` delta. A ring of :data:`WINDOW` snapshots per
  env bounds memory; per-decision reason/latency/changed flags are kept for every decision.
* **Post-move world.** ``BatchSim._resolve_collisions`` is wrapped: after the original
  (detection only; deaths are applied later), for each env whose hero dies this frame it
  keeps every snake's post-move, pre-collision cells, traversed head cells, heading and
  logical length plus the food. That is the fatal frame.
* **Trajectory.** ``run_simd_eval``'s ``frame_observer`` seam (detached hero-only facts)
  gives per frame the hero mass, alive flag, boost, food eaten, kills, frames since food,
  nearest-food Manhattan distance and food count.

Analysis (offline, pure; per death, CPU-time capped at :data:`DEATH_CPU_SECONDS`, after
which every remaining search is ``unknown``, never guessed):

* every death: the v5 census's exact one-frame counterfactual
  (``diagnose_live_v5.fatal_frame_analysis``) against the captured post-move world;
* self deaths: ``diagnose.analyze_death`` unchanged (count-or-depth PNR, depth-only and
  count-only sensitivities, depth 40, 50,000-node budget), plus v8's view at the PNR
  (reason, v2 counts, offline boost landing counts) and the v5 categories.

Compute (strict): only on AC power with the lid open, both checked at launch and polled
by a monitor thread for the whole run (simulation pauses inside the frame observer and
analysis workers are SIGSTOPped while either is false; a pause longer than
``--max-pause-seconds`` aborts). One shard per CPU slot (pool 3, 2 threads each), each
batch admitted by ``research/compute/thermal_guard.py``. Outputs are create-only.

Usage (one tmux window per shard, under caffeinate)::

  caffeinate -dimsu env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
      ./venv/bin/python research/death_census_v8_20261003/census.py run --shard 1
  ... run --shard 2 ; ... run --shard 3
  ./venv/bin/python research/death_census_v8_20261003/census.py merge

Smoke (smoke namespace, short horizon, scratch output)::

  ... census.py run --smoke --out /tmp/x --shards 1 --shard 1 --worlds-per-mix 2 \\
      --frames 600 --mixes frozen,scripted
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
REQUIRED_ENV = {"SNAKE_DQN_DEVICE": "cpu", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import (  # noqa: E402
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.trap_horizon_20261001 import diagnose as base  # noqa: E402
from research.trap_horizon_20261001 import diagnose_live_v5 as d5  # noqa: E402

SCHEMA = "death-census-v8-simd/v1"
AUTHORITY = "tier-0 development diagnostic (not a screen, not gate evidence)"
DOMAIN = "apex-veto-v8-census-v1"
SMOKE_DOMAIN = "apex-veto-v8-census-smoke-v1"
NAMESPACE = "worlds"
WORLDS_PER_MIX = 60
SHARDS = 3
BATCH_WORLDS = 5  # envs per run_simd_eval call (one thermal-guard admission each)
VARIANT = "v8"
LAMBDA = 8.0
VECTOR61_FORWARD = "rowwise"
WINDOW = base.WINDOW  # 120 decisions kept per env
DEPTH = base.DEPTH  # 40
NODE_BUDGET = base.NODE_BUDGET  # 50,000
DEATH_CPU_SECONDS = 300.0  # per-death analysis CPU cap; then every search is unknown
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
DEFAULT_ROOT = ARTIFACTS / "death-census-v8-20261003" / "run-v1"
SLOT_ROOT = ARTIFACTS / "pqn-followup-20260909"
SLOT_POOL = 3
EXCLUSION_PREFIX = 1000
CHECKPOINT_FRAMES = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)
STARVE_FRAMES = (100, 300)  # frames-since-food thresholds reported
FAR_FOOD_CELLS = 20  # nearest food farther than this (Manhattan, cells) = "far"
NEAR_DEATH_WINDOWS = (1, 10, 40, 120)
MONITOR_POLL_SECONDS = 15.0
CAUSES = ("self", "head_on", "enemy_body", "wall")

# v8 decision reasons (the last layer that set the action), integer codes in the windows.
REASONS = (
    "kept",
    "no_spacious",
    "v2_rule",
    "landing_same_direction_normal",
    "landing_v2_rule",
    "landing_no_eligible",
    "v7_rerank",
    "head_veto",
)
# Extra domains to exclude beyond the v8 serving lane's seed report.
WEB_PURPOSES = ("watch", "play", "parity", "worlds")
EXTRA_DOMAINS: Dict[str, Tuple[str, ...]] = {
    "apex-veto-v8-web-serving-v1": WEB_PURPOSES,
    "apex-veto-v8-web-serving-smoke-v1": WEB_PURPOSES,
    "trap-horizon-v5-dev-v1": ("worlds",),
    "trap-horizon-v5-dev-smoke-v1": ("worlds",),
    "trap-horizon-dev-v1": ("worlds",),
    "trap-horizon-dev-smoke-v1": ("worlds",),
}
V8_SERVING_INTENT = ARTIFACTS / "apex-veto-v8-serving-20261003" / "run-v1" / "intent.json"

Cell = Tuple[int, int]


# ---------------------------------------------------------------------------
# Power / lid safety
# ---------------------------------------------------------------------------
def on_ac_power() -> bool:
    """``pmset -g batt`` says AC Power (fails closed)."""
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "'AC Power'" in out


def parse_clamshell(text: str) -> Optional[bool]:
    """``True`` lid open, ``False`` closed, ``None`` unparseable (callers fail closed)."""
    states = []
    for line in str(text or "").splitlines():
        if "AppleClamshellState" in line and "=" in line:
            value = line.split("=", 1)[1].strip()
            if value in ("Yes", "No"):
                states.append(value == "No")
            else:
                return None
    if not states:
        return None
    return all(states)


def lid_open() -> bool:
    """The lid is open per ``ioreg`` (fails closed)."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return parse_clamshell(out) is True


class SafetyMonitor:
    """Polls AC power and the lid in a daemon thread; callers pause while unsafe.

    ``wait_if_unsafe`` blocks while the latest reading is unsafe and raises
    ``RuntimeError`` once a single pause exceeds ``max_pause``. Readers are injectable.
    """

    def __init__(
        self,
        ac_reader: Callable[[], bool] = on_ac_power,
        lid_reader: Callable[[], bool] = lid_open,
        poll: float = MONITOR_POLL_SECONDS,
        max_pause: float = 3600.0,
        log: Optional[Callable[[Dict[str, Any]], None]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.ac_reader, self.lid_reader = ac_reader, lid_reader
        self.poll, self.max_pause = float(poll), float(max_pause)
        self.log = log or (lambda event: None)
        self.sleep = sleep
        self.pauses = 0
        self.paused_seconds = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.state = self.read()

    def read(self) -> Dict[str, Any]:
        try:
            ac = bool(self.ac_reader())
        except Exception:  # fail closed
            ac = False
        try:
            lid = bool(self.lid_reader())
        except Exception:
            lid = False
        return {"ac_power": ac, "lid_open": lid, "safe": ac and lid, "t": time.time()}

    @property
    def safe(self) -> bool:
        return bool(self.state["safe"])

    def start(self) -> "SafetyMonitor":
        def loop() -> None:
            while not self._stop.wait(self.poll):
                previous = self.state["safe"]
                self.state = self.read()
                if self.state["safe"] != previous:
                    self.log({"event": "safety_change", **self.state})

        self._thread = threading.Thread(target=loop, name="safety-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def wait_if_unsafe(self) -> None:
        if self.safe:
            return
        started = time.monotonic()
        self.pauses += 1
        self.log({"event": "safety_pause", **self.state})
        while not self.safe:
            waited = time.monotonic() - started
            if waited > self.max_pause:
                self.log({"event": "safety_abort", "waited_seconds": waited, **self.state})
                raise RuntimeError(f"safety: unsafe for {waited:.0f} s ({self.state})")
            self.sleep(min(2.0, self.poll))
            if self._thread is None:  # no poller: read synchronously
                self.state = self.read()
        self.paused_seconds += time.monotonic() - started
        self.log({"event": "safety_resume", "paused_seconds": time.monotonic() - started})


# ---------------------------------------------------------------------------
# v8 decision reasons from live-hook counter deltas (pure)
# ---------------------------------------------------------------------------
_V8_FIELDS = (
    "decisions",
    "no_spacious",
    "head_risky_decisions",
    "head_risky_vetoes",
    "head_risky_kept_no_alternative",
    "head_risk_waived_hero_wins",
    "apply_seconds_total",
    "head_seconds_total",
)
_V5_FIELDS = (
    "vetoes_applied",
    "base_landing_failed",
    "boost_landing_no_eligible",
    "boost_to_normal_same_direction",
)


def hook_snapshot(hook: Any) -> Dict[str, float]:
    """The counters :func:`reason_from_deltas` reads (zeros for a hook not yet built)."""
    out: Dict[str, float] = {f: 0 for f in _V8_FIELDS}
    out.update({f"v5_{f}": 0 for f in _V5_FIELDS})
    out["v7_rerank_changes"] = 0
    if hook is None:
        return out
    for f in _V8_FIELDS:
        out[f] = getattr(hook.v8, f)
    out["v7_rerank_changes"] = hook.v7.v7.rerank_changes
    for f in _V5_FIELDS:
        out[f"v5_{f}"] = getattr(hook.v7.v7.v5, f)
    return out


def reason_from_deltas(before: Mapping[str, float], after: Mapping[str, float]) -> Dict[str, Any]:
    """One decision's reason code, head flags and costs from two :func:`hook_snapshot`s.

    Reason = the last layer that set the action: v5's ``landing_no_eligible`` (a boost
    whose landing failed with no eligible alternative, which can co-occur with
    ``no_spacious``); else ``no_spacious``; else ``head_veto``; else ``v7_rerank``; else
    v5's other landing reasons; else v5's ``v2_rule`` veto; else ``kept``.
    """
    d = {k: after[k] - before[k] for k in after}
    if d["decisions"] != 1:
        raise RuntimeError(f"expected exactly one v8 decision per env per call, got {d}")
    if d["v5_base_landing_failed"] and d["v5_boost_landing_no_eligible"]:
        reason = "landing_no_eligible"
    elif d["no_spacious"]:
        reason = "no_spacious"
    elif d["head_risky_vetoes"]:
        reason = "head_veto"
    elif d["v7_rerank_changes"]:
        reason = "v7_rerank"
    elif d["v5_base_landing_failed"]:
        if d["v5_boost_landing_no_eligible"]:
            reason = "landing_no_eligible"
        elif d["v5_boost_to_normal_same_direction"]:
            reason = "landing_same_direction_normal"
        else:
            reason = "landing_v2_rule"
    elif d["v5_vetoes_applied"]:
        reason = "v2_rule"
    else:
        reason = "kept"
    return {
        "reason": REASONS.index(reason),
        "head_risky": bool(d["head_risky_decisions"]),
        "head_kept_no_alternative": bool(d["head_risky_kept_no_alternative"]),
        "head_waived": bool(d["head_risk_waived_hero_wins"]),
        "apply_seconds": float(d["apply_seconds_total"]),
        "head_seconds": float(d["head_seconds_total"]),
    }


# ---------------------------------------------------------------------------
# SIMD capture
# ---------------------------------------------------------------------------
def _ordered_cells(sim: Any, env: int, slot: int) -> np.ndarray:
    return base._ordered_cells(sim, env, slot)


class CensusRecorder:
    """Per-env decision rings, per-decision series, fatal post-move worlds, trajectories."""

    def __init__(
        self,
        envs: int,
        horizon: int,
        window: int = WINDOW,
        monitor: Optional[SafetyMonitor] = None,
    ) -> None:
        self.envs, self.horizon, self.window = int(envs), int(horizon), int(window)
        self.monitor = monitor
        self.sim: Any = None
        self.world_meta: Optional[Dict[str, Any]] = None
        self.buffers: Dict[int, Deque[Dict[str, Any]]] = {}
        self.decisions: Dict[int, Dict[str, List[Any]]] = {}
        self.posts: Dict[int, Dict[str, Any]] = {}
        self.frames_observed = 0
        H = self.horizon
        self.traj: Dict[str, np.ndarray] = {
            "alive_pre": np.zeros((envs, H), bool),
            "alive_post": np.zeros((envs, H), bool),
            "mass_post": np.zeros((envs, H), np.int32),
            "boosted": np.zeros((envs, H), bool),
            "ate": np.zeros((envs, H), np.int16),
            "kills": np.zeros((envs, H), np.int16),
            "frames_since_food": np.zeros((envs, H), np.int32),
            "nearest_food": np.full((envs, H), -1, np.int32),
            "food_count": np.zeros((envs, H), np.int32),
        }

    # -- decisions ----------------------------------------------------------
    def capture_decisions(
        self,
        sim: Any,
        last_veto: Mapping[str, Any],
        q_rows: np.ndarray,
        masks: np.ndarray,
        befores: Mapping[int, Mapping[str, float]],
        afters: Mapping[int, Mapping[str, float]],
    ) -> None:
        if self.sim is None:
            self.sim = sim
            self.world_meta = {
                "grid_width": int(sim.grid_w),
                "grid_height": int(sim.grid_h),
                "min_boost_length": int(sim.cfg.min_boost_length),
                "boost_cost_frames": int(sim.cfg.boost_length_cost_frames),
                "trail_food": bool(sim.v2),
                "segment_size": int(sim.s),
                "mechanics_version": int(sim.cfg.mechanics_version),
            }
        elif sim is not self.sim:
            raise RuntimeError("one recorder per run_simd_eval call")
        rows = np.asarray(last_veto["rows"])
        for i, (env, slot) in enumerate(rows):
            env, slot = int(env), int(slot)
            info = reason_from_deltas(befores[env], afters[env])
            others = [
                _ordered_cells(sim, env, s)
                for s in range(int(sim.S))
                if s != slot and bool(sim.alive[env, s])
            ]
            base_action, final = int(last_veto["base"][i]), int(last_veto["final"][i])
            snap = {
                "frame": int(sim.frame[env]),
                "body": _ordered_cells(sim, env, slot),
                "length": int(sim.length[env, slot]),
                "direction": int(sim.direction[env, slot]),
                "boost_frames": int(sim.boost_frames[env, slot]),
                "others": (
                    np.concatenate(others).astype(np.int16)
                    if others
                    else np.zeros((0, 2), np.int16)
                ),
                "food": np.asarray(list(sim.food_cells[env]), dtype=np.int16).reshape(-1, 2),
                "v2_counts": np.asarray(last_veto["counts"][i], dtype=np.int64),
                "v2_need": int(last_veto["need"][i]),
                "base": base_action,
                "final": final,
                "q": np.asarray(q_rows[i], dtype=np.float32),
                "mask": np.asarray(masks[i], dtype=bool),
                "v8_reason": info["reason"],
                "head_risky": info["head_risky"],
                "head_kept_no_alternative": info["head_kept_no_alternative"],
                "head_waived": info["head_waived"],
                "apply_seconds": info["apply_seconds"],
            }
            self.buffers.setdefault(env, deque(maxlen=self.window)).append(snap)
            series = self.decisions.setdefault(
                env,
                {k: [] for k in ("frame", "reason", "changed", "boost", "head_risky", "seconds")},
            )
            series["frame"].append(snap["frame"])
            series["reason"].append(info["reason"])
            series["changed"].append(final != base_action)
            series["boost"].append(final >= base.NUM_DIRECTIONS)
            series["head_risky"].append(info["head_risky"])
            series["seconds"].append(info["apply_seconds"])

    def decision_count(self, env: int) -> int:
        return len(self.decisions.get(int(env), {}).get("frame", []))

    # -- fatal post-move world ------------------------------------------------
    def capture_post(self, sim: Any, env: int) -> None:
        rows = []
        for slot in range(int(sim.S)):
            rows.append(
                {
                    "is_hero": slot == 0,
                    "id": slot,
                    "alive": bool(sim.alive[env, slot]),
                    "cells": [tuple(int(v) for v in c) for c in _ordered_cells(sim, env, slot)],
                    "traversed": [
                        tuple(int(v) for v in c) for c in sim._traversed_cells(env, slot)
                    ],
                    "direction": int(sim.direction[env, slot]),
                    "logical_length": max(1, int(sim.length[env, slot])),
                }
            )
        self.posts[int(env)] = {
            "frame": int(sim.frame[env]),
            "snakes": rows,
            "food": [tuple(int(v) for v in c) for c in sim.get_food(env)],
        }

    # -- per-frame trajectory -----------------------------------------------
    def observe_frame(self, payload: Mapping[str, Any]) -> None:
        t = int(payload["frame"])
        pre, post = payload["pre"], payload["post"]
        tr = self.traj
        tr["alive_pre"][:, t] = pre["alive"]
        tr["alive_post"][:, t] = post["alive"]
        tr["mass_post"][:, t] = post["logical_mass"]
        tr["boosted"][:, t] = np.asarray(post["boosted"], bool) & np.asarray(
            post["transition_valid"], bool
        )
        tr["ate"][:, t] = np.where(post["transition_valid"], post["food_ate"], 0)
        tr["kills"][:, t] = np.where(post["transition_valid"], post["kills"], 0)
        tr["frames_since_food"][:, t] = post["frames_since_food"]
        heads = np.asarray(pre["heads"])
        for env, cells in enumerate(pre["food_cells"]):
            tr["food_count"][env, t] = len(cells)
            if pre["alive"][env] and cells:
                food = np.asarray(cells, dtype=np.int64)
                tr["nearest_food"][env, t] = int(np.abs(food - heads[env]).sum(axis=1).min())
        self.frames_observed += 1
        if self.monitor is not None:
            self.monitor.wait_if_unsafe()


_ACTIVE: List[CensusRecorder] = []


@contextmanager
def census_capture(recorder: CensusRecorder) -> Iterator[CensusRecorder]:
    """Wrap ``Vector61SimdPolicy._apply_veto`` and ``BatchSim._resolve_collisions``.

    Both originals run first and their results are returned unchanged.
    """
    from src.simd_env.batch_sim import BatchSim
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    original_apply = Vector61SimdPolicy._apply_veto
    original_resolve = BatchSim.__dict__["_resolve_collisions"]

    def apply(self, sim, slots, masked_q, mask, actions):  # type: ignore[no-untyped-def]
        if not self.veto_slots:
            return original_apply(self, sim, slots, masked_q, mask, actions)
        if self.veto_variant != VARIANT:
            raise RuntimeError(f"census expects the {VARIANT} hero veto, got {self.veto_variant}")
        slots_arr = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        pick = np.flatnonzero(np.isin(slots_arr[:, 1], list(self.veto_slots)))
        envs = [int(slots_arr[i, 0]) for i in pick]
        befores = {e: hook_snapshot(self.live_vetoes.get(e)) for e in envs}
        out = original_apply(self, sim, slots, masked_q, mask, actions)
        if len(pick):
            afters = {e: hook_snapshot(self.live_vetoes.get(e)) for e in envs}
            q_rows = masked_q.cpu().numpy()[pick]
            recorder.capture_decisions(
                sim, self.last_veto, q_rows, np.asarray(mask)[pick], befores, afters
            )
        return out

    def resolve(self):  # type: ignore[no-untyped-def]
        result = original_resolve(self)
        if self is recorder.sim:
            cause = result[0]
            active = self._active_envs()
            for env in np.flatnonzero(self.alive[:, 0] & (cause[:, 0] != 0) & active):
                recorder.capture_post(self, int(env))
        return result

    if _ACTIVE:
        raise RuntimeError("census capture is not re-entrant")
    _ACTIVE.append(recorder)
    Vector61SimdPolicy._apply_veto = apply
    BatchSim._resolve_collisions = resolve
    try:
        yield recorder
    finally:
        Vector61SimdPolicy._apply_veto = original_apply
        BatchSim._resolve_collisions = original_resolve
        _ACTIVE.clear()


# ---------------------------------------------------------------------------
# Windows (diagnose.pack_window plus v8 fields)
# ---------------------------------------------------------------------------
_EXTRA_INT = ("v8_reason",)
_EXTRA_BOOL = ("head_risky", "head_kept_no_alternative", "head_waived")


def pack_census_window(snaps: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    out = base.pack_window(snaps)
    for key in _EXTRA_INT:
        out[key] = np.asarray([int(s[key]) for s in snaps], dtype=np.int64)
    for key in _EXTRA_BOOL:
        out[key] = np.asarray([bool(s[key]) for s in snaps], dtype=bool)
    out["apply_seconds"] = np.asarray([float(s["apply_seconds"]) for s in snaps], np.float64)
    return out


def unpack_census_window(packed: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    snaps = base.unpack_window(packed)
    for t, snap in enumerate(snaps):
        for key in _EXTRA_INT:
            snap[key] = int(packed[key][t])
        for key in _EXTRA_BOOL:
            snap[key] = bool(packed[key][t])
        snap["apply_seconds"] = float(packed["apply_seconds"][t])
    return snaps


# ---------------------------------------------------------------------------
# Per-episode metrics (pure)
# ---------------------------------------------------------------------------
def _stats(values: Sequence[float]) -> Optional[Dict[str, float]]:
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return None
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def trajectory_metrics(tr: Mapping[str, np.ndarray], horizon: int) -> Dict[str, Any]:
    """Mass trajectory, growth, boost, food-seeking and kill metrics of one episode."""
    alive = np.asarray(tr["alive_post"], bool)
    alive_pre = np.asarray(tr["alive_pre"], bool)
    mass = np.asarray(tr["mass_post"], np.int64)
    acted = alive_pre  # the hero acted this frame
    n_alive = int(alive.sum())
    died = bool((alive_pre & ~alive).any())
    death_t = int(np.flatnonzero(alive_pre & ~alive)[0]) if died else None
    live_mass = mass[alive]
    first_frame_mass = int(mass[0]) if alive[0] else 0
    final_mass = int(live_mass[-1]) if n_alive else 0
    deltas = np.diff(mass)
    both_alive = alive[1:] & alive[:-1]
    shrink = int(-deltas[both_alive & (deltas < 0)].sum()) if len(deltas) else 0
    fsf = np.asarray(tr["frames_since_food"], np.int64)[alive]
    near = np.asarray(tr["nearest_food"], np.int64)
    near_alive = near[acted & (near >= 0)]
    eaten = int(np.asarray(tr["ate"], np.int64)[acted].sum())
    boost_frames = int(np.asarray(tr["boosted"], bool)[acted].sum())
    # Longest run of alive frames without food (frames_since_food is the run length).
    out: Dict[str, Any] = {
        "horizon": int(horizon),
        "frames_alive": n_alive,
        "died": died,
        "death_frame_index": death_t,
        "mass_integral": float((mass * alive).sum()) / float(horizon),
        "first_frame_mass": first_frame_mass,
        "final_alive_mass": final_mass,
        "peak_mass": int(live_mass.max()) if n_alive else 0,
        "mass_at": {
            str(f): (int(mass[f - 1]) if f - 1 < len(mass) and alive[f - 1] else 0)
            for f in CHECKPOINT_FRAMES
            if f <= horizon
        },
        "food_eaten": eaten,
        "food_per_1000_alive": 1000.0 * eaten / n_alive if n_alive else None,
        "net_growth_per_1000_alive": (
            1000.0 * (final_mass - first_frame_mass) / n_alive if n_alive else None
        ),
        "mass_lost_while_alive": shrink,
        "boost_frames": boost_frames,
        "boost_fraction": boost_frames / n_alive if n_alive else None,
        "kills": int(np.asarray(tr["kills"], np.int64).sum()),
        "max_frames_since_food": int(fsf.max()) if fsf.size else 0,
        "frames_since_food_over": {
            str(k): (float((fsf > k).mean()) if fsf.size else None) for k in STARVE_FRAMES
        },
        "nearest_food_mean": float(near_alive.mean()) if near_alive.size else None,
        "nearest_food_median": float(np.median(near_alive)) if near_alive.size else None,
        "far_food_fraction": (
            float((near_alive > FAR_FOOD_CELLS).mean()) if near_alive.size else None
        ),
        "no_food_frames": int((acted & (near < 0)).sum()),
        "food_count_mean": (
            float(np.asarray(tr["food_count"], np.float64)[acted].mean()) if acted.any() else None
        ),
    }
    half = horizon // 2
    if n_alive and alive[horizon - 1] and alive[half - 1]:
        out["late_growth_per_frame"] = float(mass[horizon - 1] - mass[half - 1]) / float(
            horizon - half
        )
    else:
        out["late_growth_per_frame"] = None
    return out


def decision_metrics(series: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
    """Per-episode decision tallies (by reason) and latency totals."""
    reasons = np.asarray(series.get("reason", []), dtype=np.int64)
    changed = np.asarray(series.get("changed", []), dtype=bool)
    seconds = np.asarray(series.get("seconds", []), dtype=np.float64)
    return {
        "decisions": int(reasons.size),
        "changed": int(changed.sum()),
        "by_reason": {name: int((reasons == i).sum()) for i, name in enumerate(REASONS)},
        "changed_by_reason": {
            name: int(((reasons == i) & changed).sum()) for i, name in enumerate(REASONS)
        },
        "head_risky": int(np.asarray(series.get("head_risky", []), dtype=bool).sum()),
        "boost_decisions": int(np.asarray(series.get("boost", []), dtype=bool).sum()),
        "apply_seconds_total": float(seconds.sum()),
        "apply_seconds_max": float(seconds.max()) if seconds.size else 0.0,
    }


def near_death_veto(snaps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Veto activity in the last decisions before a death (the window ends at the fatal one)."""
    n = len(snaps)
    out: Dict[str, Any] = {}
    for k in NEAR_DEATH_WINDOWS:
        tail = snaps[max(0, n - k) :]
        changed = [s for s in tail if int(s["final"]) != int(s["base"])]
        out[f"last_{k}"] = {
            "decisions": len(tail),
            "changed": len(changed),
            "changed_reasons": base._tally(REASONS[int(s["v8_reason"])] for s in changed),
            "no_spacious": sum(REASONS[int(s["v8_reason"])] == "no_spacious" for s in tail),
            "head_risky": sum(bool(s["head_risky"]) for s in tail),
            "boost_final": sum(int(s["final"]) >= base.NUM_DIRECTIONS for s in tail),
        }
    last_change = [i for i, s in enumerate(snaps) if int(s["final"]) != int(s["base"])]
    out["decisions_since_last_change"] = (n - 1 - last_change[-1]) if last_change else None
    fatal = snaps[-1]
    out["fatal_reason"] = REASONS[int(fatal["v8_reason"])]
    out["fatal_changed"] = int(fatal["final"]) != int(fatal["base"])
    out["fatal_boost"] = int(fatal["final"]) >= base.NUM_DIRECTIONS
    out["fatal_boost_frames_counter"] = int(fatal["boost_frames"])
    return out


# ---------------------------------------------------------------------------
# Offline analysis of one death (pure, CPU-capped)
# ---------------------------------------------------------------------------
class DeadlineEscapeSearch(base.EscapeSearch):
    """``EscapeSearch`` that reports ``unknown`` once the process CPU deadline passes."""

    deadline: Optional[float] = None
    hits = 0

    def _visit(self, state, step):  # type: ignore[no-untyped-def]
        cls = DeadlineEscapeSearch
        if cls.deadline is not None and (self.nodes & 255) == 0:
            if time.process_time() > cls.deadline:
                cls.hits += 1
                raise base.BudgetExhausted
        return super()._visit(state, step)


@contextmanager
def cpu_deadline(seconds: Optional[float]) -> Iterator[None]:
    """Route every escape search through :class:`DeadlineEscapeSearch` for one death."""
    original = base.EscapeSearch
    DeadlineEscapeSearch.deadline = None if seconds is None else time.process_time() + seconds
    DeadlineEscapeSearch.hits = 0
    base.EscapeSearch = DeadlineEscapeSearch  # type: ignore[misc]
    try:
        yield
    finally:
        base.EscapeSearch = original  # type: ignore[misc]
        DeadlineEscapeSearch.deadline = None


def v8_view(snap: Mapping[str, Any], meta: Mapping[str, Any]) -> Dict[str, Any]:
    """v2's one-step view, v8's reason/flags and offline boost landing counts at a snapshot."""
    from src.evaluation.safety_veto import free_space_threshold

    view = base._v2_view(snap)
    world, state = base.world_and_state(snap, meta)
    cap, _need = free_space_threshold(state.length, max(1, state.length))
    view["v8_reason"] = REASONS[int(snap["v8_reason"])]
    view["head_risky"] = bool(snap["head_risky"])
    view["head_kept_no_alternative"] = bool(snap["head_kept_no_alternative"])
    view["landing_counts"] = [
        d5.static_landing_count(world, state, d, cap) for d in range(base.NUM_DIRECTIONS)
    ]
    view["cap"] = int(cap)
    return view


def analyze_census_death(
    snaps: Sequence[Mapping[str, Any]],
    post: Mapping[str, Any],
    meta: Mapping[str, Any],
    cause: str,
    depth: int = DEPTH,
    budget: int = NODE_BUDGET,
    cpu_seconds: Optional[float] = DEATH_CPU_SECONDS,
) -> Dict[str, Any]:
    """Fatal-frame counterfactual for every cause; PNR walk and v8 view for self deaths."""
    started, cpu0 = time.perf_counter(), time.process_time()
    with cpu_deadline(cpu_seconds):
        out: Dict[str, Any] = {
            "fatal": d5.fatal_frame_analysis(snaps[-1], post, meta, cause, depth, budget),
            "fatal_v8": v8_view(snaps[-1], meta),
            "fatal_length": int(snaps[-1]["length"]),
            "fatal_frame": int(snaps[-1]["frame"]),
            "window_frames": len(snaps),
            "model_check": base.model_replay_check(snaps, meta),
            "near_death": near_death_veto(snaps),
        }
        if cause == "self":
            pnr = base.analyze_death(snaps, meta, depth, budget)
            pnr.pop("model_check", None)
            pnr.pop("evaluated_frames", None)  # bulky; the PNR rows are kept
            out.update(pnr)
            walk = out["walk"]
            out["pnr_v8"] = (
                v8_view(snaps[walk["pnr_index"]], meta) if walk["pnr_index"] is not None else None
            )
            count_walk = out["sensitivity"][base.COUNT_ONLY]["walk"]
            out["count_pnr_v8"] = (
                v8_view(snaps[count_walk["pnr_index"]], meta)
                if count_walk["pnr_index"] is not None
                else None
            )
            if walk["pnr_index"] is not None and walk["taken_action"] is not None:
                taken = int(walk["taken_action"])
                row = out["pnr_row"].get(base.COUNT_OR_DEPTH, {})
                out["pnr_same_direction_normal_status"] = (
                    row.get(str(taken % base.NUM_DIRECTIONS))
                    if taken >= base.NUM_DIRECTIONS
                    else None
                )
                mask = [bool(x) for x in snaps[walk["pnr_index"]]["mask"]]
                out["pnr_escaping_masked_legal"] = [a for a in walk["escaping_actions"] if mask[a]]
            out["category"] = d5.classify_self_death(out)
            out["enclosed_early"] = d5.enclosed_early(out)
            out["pnr_escape_kinds"] = d5.pnr_escape_kinds(out)
            out["trap_class"] = trap_class(out)
        out["deadline_hits"] = int(DeadlineEscapeSearch.hits)
    out["deadline_hit"] = out["deadline_hits"] > 0
    out["cpu_seconds"] = time.process_time() - cpu0
    out["seconds_total"] = time.perf_counter() - started
    return out


def trap_class(death: Mapping[str, Any]) -> str:
    """Where a self death's trap sits relative to v8's one-step look-ahead.

    ``beyond_lookahead``: at the count-only PNR (the last frame from which a spacious
    state was still reachable) the taken direction looked spacious to v2's one-step count,
    so the veto had no signal (enterable but doomed further out); ``inside_lookahead``:
    the taken direction was already below ``need`` there (v8 saw it, but no spacious
    alternative or the policy/v7 still chose it); ``no_count_pnr``: the count-only walk
    found no PNR in the window; ``beyond_window``: every window frame was proven without a
    count-only escape (enclosed before the window); ``unresolved``: the count-only walk
    or the taken action's status there is unknown (never guessed).
    """
    walk = death["sensitivity"][base.COUNT_ONLY]["walk"]
    if walk["status"] == "beyond_window":
        return "beyond_window"
    if walk["status"] != "exact" or walk["taken_status"] != base.NO_ESCAPE:
        return "unresolved"
    view = death.get("count_pnr_v8")
    if view is None:
        return "no_count_pnr"
    if view["taken_direction_spacious"]:
        return "beyond_lookahead"
    return "inside_lookahead"


def analyze_death_file(args: Tuple[str, str, Mapping[str, Any], str, int, int, float]):
    """Pool worker: load one saved death (window npz + post JSON) and analyze it."""
    window_path, post_path, meta, cause, depth, budget, cpu_seconds = args
    try:
        with np.load(window_path) as data:
            packed = {k: data[k] for k in data.files}
        post = json.loads(Path(post_path).read_text())
        return analyze_census_death(
            unpack_census_window(packed), post, meta, cause, depth, budget, cpu_seconds
        )
    except Exception as exc:  # recorded, never fatal: the simulation is already saved
        import traceback

        return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Headroom estimates (pure)
# ---------------------------------------------------------------------------
def death_headroom(
    death_index: int, mass_at_death: float, horizon: int, growth_per_frame: float
) -> Dict[str, float]:
    """Mass-integral gain if this death had not happened.

    ``hold``: the hero keeps its mass at death for the rest of the horizon (a lower-style
    estimate: no growth, no later death). ``grow``: plus linear growth at
    ``growth_per_frame`` (the mix's median late survivor growth), still without a later
    death (an upper-style estimate).
    """
    remaining = max(0, int(horizon) - int(death_index))  # frames t..H-1 are dead
    hold = float(mass_at_death) * remaining / float(horizon)
    grow = hold + max(0.0, float(growth_per_frame)) * remaining * remaining / (2.0 * horizon)
    return {"hold": hold, "grow": grow, "remaining_frames": remaining}


def death_mode(episode: Mapping[str, Any], death: Optional[Mapping[str, Any]]) -> str:
    """Failure-mode label of one episode (``survived`` when the hero lived)."""
    cause = episode.get("death_cause")
    if not episode.get("died"):
        return "survived"
    if death is None and cause in ("self", "head_on"):
        return f"{cause}:unanalyzed"
    if cause == "self":
        return f"self:{death['category']}"
    if cause == "head_on" and death is not None:
        fatal = death["fatal"]
        if fatal["any_legal_alternative_avoids_with_escape"]:
            return "head_on:avoidable"
        return "head_on:unavoidable"
    return str(cause)


# ---------------------------------------------------------------------------
# Summary (pure)
# ---------------------------------------------------------------------------
def _percentiles(values: np.ndarray) -> Optional[Dict[str, float]]:
    if values.size == 0:
        return None
    qs = np.quantile(values, [0.5, 0.9, 0.99, 0.999])
    return {
        "n": int(values.size),
        "mean_ms": float(values.mean() * 1e3),
        "p50_ms": float(qs[0] * 1e3),
        "p90_ms": float(qs[1] * 1e3),
        "p99_ms": float(qs[2] * 1e3),
        "p999_ms": float(qs[3] * 1e3),
        "max_ms": float(values.max() * 1e3),
    }


def latency_summary(seconds: np.ndarray, reasons: np.ndarray, risky: np.ndarray) -> Dict:
    """Per-decision v8 hook wall cost overall, by reason and for head-risky decisions."""
    seconds = np.asarray(seconds, np.float64)
    reasons = np.asarray(reasons, np.int64)
    risky = np.asarray(risky, bool)
    return {
        "all": _percentiles(seconds),
        "by_reason": {
            name: _percentiles(seconds[reasons == i])
            for i, name in enumerate(REASONS)
            if (reasons == i).any()
        },
        "head_risky": _percentiles(seconds[risky]),
        "over_16ms": int((seconds > 0.016).sum()),
        "over_100ms": int((seconds > 0.1).sum()),
    }


def summarize_census(
    episodes: Sequence[Mapping[str, Any]],
    deaths: Sequence[Mapping[str, Any]],
    latency: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Outcomes, fatal-frame and PNR tables, near-death veto, trajectories and headroom."""
    horizon = int(episodes[0]["trajectory"]["horizon"]) if episodes else 0
    mixes = sorted({ep["mix"] for ep in episodes})
    errors = [d for d in deaths if "error" in d]
    deaths = [d for d in deaths if "error" not in d]
    by_key = {(d["mix"], int(d["world_seed"])): d for d in deaths}
    outcomes: Dict[str, Dict[str, int]] = {}
    for ep in episodes:
        label = ep["death_cause"] if ep["died"] else "survived"
        cell = outcomes.setdefault(ep["mix"], {})
        cell[label] = cell.get(label, 0) + 1
    pooled: Dict[str, int] = {}
    for cell in outcomes.values():
        for key, value in cell.items():
            pooled[key] = pooled.get(key, 0) + value
    n = len(episodes)

    def mix_stat(fn: Callable[[Mapping[str, Any]], Any]) -> Dict[str, Any]:
        out = {
            mix: _stats(
                [v for ep in episodes if ep["mix"] == mix for v in [fn(ep)] if v is not None]
            )
            for mix in mixes
        }
        out["pooled"] = _stats([v for ep in episodes for v in [fn(ep)] if v is not None])
        return out

    # Growth used by the headroom upper estimate: median late growth of survivors per mix.
    growth = {}
    for mix in mixes:
        rates = [
            ep["trajectory"]["late_growth_per_frame"]
            for ep in episodes
            if ep["mix"] == mix and ep["trajectory"]["late_growth_per_frame"] is not None
        ]
        growth[mix] = float(np.median(rates)) if rates else 0.0

    modes: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        death = by_key.get((ep["mix"], int(ep["world_seed"])))
        mode = death_mode(ep, death)
        row = modes.setdefault(
            mode, {"episodes": 0, "by_mix": {}, "hold": 0.0, "grow": 0.0, "mass_integral": []}
        )
        row["episodes"] += 1
        row["by_mix"][ep["mix"]] = row["by_mix"].get(ep["mix"], 0) + 1
        row["mass_integral"].append(float(ep["mass_integral"]))
        if ep["died"]:
            h = death_headroom(
                ep["trajectory"]["death_frame_index"],
                ep["mass_at_death"],
                horizon,
                growth[ep["mix"]],
            )
            row["hold"] += h["hold"]
            row["grow"] += h["grow"]
    headroom = {
        mode: {
            "episodes": row["episodes"],
            "share": row["episodes"] / n if n else None,
            "by_mix": dict(sorted(row["by_mix"].items())),
            "mean_mass_integral": float(np.mean(row["mass_integral"])),
            "pooled_gain_hold": row["hold"] / n if n else None,
            "pooled_gain_grow": row["grow"] / n if n else None,
        }
        for mode, row in sorted(modes.items())
    }

    by_cause: Dict[str, Dict[str, Any]] = {}
    for cause in sorted({d["cause"] for d in deaths}):
        rows = [d for d in deaths if d["cause"] == cause]
        by_cause[cause] = {
            "deaths": len(rows),
            "model_reproduces_cause": sum(bool(d["fatal"]["model_reproduces_cause"]) for d in rows),
            "any_legal_alternative_avoids": sum(
                bool(d["fatal"]["any_legal_alternative_avoids"]) for d in rows
            ),
            "any_legal_alternative_avoids_with_escape": sum(
                bool(d["fatal"]["any_legal_alternative_avoids_with_escape"]) for d in rows
            ),
            "taken_hit_static_pre_move": base._tally(
                d["fatal"]["taken_hit_static_pre_move"] for d in rows
            ),
            "taken_boost": sum(int(d["fatal"]["taken"]) >= base.NUM_DIRECTIONS for d in rows),
            "fatal_v8_reason": base._tally(d["fatal_v8"]["v8_reason"] for d in rows),
            "fatal_v2_outcome": base._tally(d["fatal_v8"]["outcome"] for d in rows),
            "fatal_head_risky": sum(bool(d["fatal_v8"]["head_risky"]) for d in rows),
            "by_mix": base._tally(d["mix"] for d in rows),
            "fatal_lengths": sorted(int(d["fatal_length"]) for d in rows),
            "fatal_frames": sorted(int(d["fatal_frame"]) for d in rows),
            "veto_changed_last_1": sum(d["near_death"]["last_1"]["changed"] > 0 for d in rows),
            "veto_changed_last_10": sum(d["near_death"]["last_10"]["changed"] > 0 for d in rows),
            "veto_changed_last_40": sum(d["near_death"]["last_40"]["changed"] > 0 for d in rows),
            "no_spacious_in_last_10": sum(
                d["near_death"]["last_10"]["no_spacious"] > 0 for d in rows
            ),
            "head_risky_in_last_10": sum(
                d["near_death"]["last_10"]["head_risky"] > 0 for d in rows
            ),
            "deadline_hit": sum(bool(d["deadline_hit"]) for d in rows),
        }

    self_rows = [d for d in deaths if d["cause"] == "self"]
    summary: Dict[str, Any] = {
        "episodes": n,
        "horizon": horizon,
        "episode_outcomes_by_mix": dict(sorted(outcomes.items())),
        "episode_outcomes_pooled": dict(sorted(pooled.items())),
        "outcome_shares_pooled": {k: v / n for k, v in sorted(pooled.items())} if n else {},
        "mass_integral": mix_stat(lambda ep: ep["mass_integral"]),
        "survival_fraction": mix_stat(lambda ep: ep["survival_fraction"]),
        "mass_at_death": mix_stat(lambda ep: ep["mass_at_death"] if ep["died"] else None),
        "frames_survived_dead": mix_stat(
            lambda ep: ep["trajectory"]["frames_alive"] if ep["died"] else None
        ),
        "final_mass_survivors": mix_stat(
            lambda ep: None if ep["died"] else ep["trajectory"]["final_alive_mass"]
        ),
        "trajectory": {
            "mass_at": {
                str(f): mix_stat(lambda ep, f=f: ep["trajectory"]["mass_at"].get(str(f)))
                for f in CHECKPOINT_FRAMES
                if f <= horizon
            },
            "food_per_1000_alive": mix_stat(lambda ep: ep["trajectory"]["food_per_1000_alive"]),
            "net_growth_per_1000_alive": mix_stat(
                lambda ep: ep["trajectory"]["net_growth_per_1000_alive"]
            ),
            "late_growth_per_frame_survivors": mix_stat(
                lambda ep: ep["trajectory"]["late_growth_per_frame"]
            ),
            "boost_fraction": mix_stat(lambda ep: ep["trajectory"]["boost_fraction"]),
            "mass_lost_while_alive": mix_stat(lambda ep: ep["trajectory"]["mass_lost_while_alive"]),
            "kills": mix_stat(lambda ep: ep["trajectory"]["kills"]),
            "max_frames_since_food": mix_stat(lambda ep: ep["trajectory"]["max_frames_since_food"]),
            "frames_since_food_over_100": mix_stat(
                lambda ep: ep["trajectory"]["frames_since_food_over"]["100"]
            ),
            "frames_since_food_over_300": mix_stat(
                lambda ep: ep["trajectory"]["frames_since_food_over"]["300"]
            ),
            "nearest_food_mean": mix_stat(lambda ep: ep["trajectory"]["nearest_food_mean"]),
            "far_food_fraction": mix_stat(lambda ep: ep["trajectory"]["far_food_fraction"]),
            "food_count_mean": mix_stat(lambda ep: ep["trajectory"]["food_count_mean"]),
        },
        "kills_total": int(sum(ep["trajectory"]["kills"] for ep in episodes)),
        "decisions": {
            "total": int(sum(ep["decision_metrics"]["decisions"] for ep in episodes)),
            "changed": int(sum(ep["decision_metrics"]["changed"] for ep in episodes)),
            "by_reason": {
                r: int(sum(ep["decision_metrics"]["by_reason"][r] for ep in episodes))
                for r in REASONS
            },
            "changed_by_reason": {
                r: int(sum(ep["decision_metrics"]["changed_by_reason"][r] for ep in episodes))
                for r in REASONS
            },
            "head_risky": int(sum(ep["decision_metrics"]["head_risky"] for ep in episodes)),
        },
        "growth_per_frame_used_for_headroom": growth,
        "headroom_by_mode": headroom,
        "fatal_frame_by_cause": by_cause,
        "model_check": {
            "transitions_compared": sum(d["model_check"]["transitions_compared"] for d in deaths),
            "mismatches": sum(d["model_check"]["mismatches"] for d in deaths),
            "fatal_action_static_model_outcome_self_deaths": base._tally(
                d["model_check"]["fatal_action_model_outcome"] for d in self_rows
            ),
        },
        "analysis": {
            "errors": [
                {"mix": d["mix"], "world_seed": d["world_seed"], "error": d["error"]}
                for d in errors
            ],
            "deaths": len(deaths),
            "deadline_hit": sum(bool(d["deadline_hit"]) for d in deaths),
            "cpu_seconds": _stats([d["cpu_seconds"] for d in deaths]),
        },
    }
    if latency is not None:
        summary["veto_latency"] = latency_summary(
            latency["seconds"], latency["reason"], latency["head_risky"]
        )
    if self_rows:
        found = [d for d in self_rows if d["walk"]["pnr_index"] is not None]
        summary["self"] = {
            "pnr": base.summarize_walks(self_rows),
            "pnr_sensitivity": {
                c: base.summarize_walks(
                    [{"mix": d["mix"], "walk": d["sensitivity"][c]["walk"]} for d in self_rows]
                )
                for c in (base.DEPTH_ONLY, base.COUNT_ONLY)
            },
            "categories": base._tally(d["category"] for d in self_rows),
            "categories_by_mix": {
                mix: base._tally(d["category"] for d in self_rows if d["mix"] == mix)
                for mix in sorted({d["mix"] for d in self_rows})
            },
            "enclosed_early": base._tally(d["enclosed_early"] for d in self_rows),
            "category_by_enclosed_early": base._tally(
                f"{d['category']}|enclosed_early={d['enclosed_early']}" for d in self_rows
            ),
            "trap_class": base._tally(d["trap_class"] for d in self_rows),
            "trap_class_by_category": base._tally(
                f"{d['category']}|{d['trap_class']}" for d in self_rows
            ),
            "pnr_escape_kinds": base._tally(d["pnr_escape_kinds"] for d in found),
            "pnr_v8_reason": base._tally(d["pnr_v8"]["v8_reason"] for d in found),
            "pnr_v2_outcome": base._tally(d["pnr_v8"]["outcome"] for d in found),
            "pnr_taken_boost": sum(
                int(d["walk"]["taken_action"]) >= base.NUM_DIRECTIONS for d in found
            ),
            "boost_fatal_choice_same_direction_normal_status": base._tally(
                d.get("pnr_same_direction_normal_status")
                for d in self_rows
                if d["category"] == "boost_fatal_choice"
            ),
            "boost_fatal_choice_landing_counts": [
                {
                    "taken": int(d["walk"]["taken_action"]),
                    "landing": d["pnr_v8"]["landing_counts"][int(d["walk"]["taken_action"]) % 3],
                    "need": d["pnr_v8"]["need"],
                    "pnr_v8_reason": d["pnr_v8"]["v8_reason"],
                }
                for d in self_rows
                if d["category"] == "boost_fatal_choice"
            ],
            "normal_fatal_choice_pnr_v2_outcome": base._tally(
                d["pnr_v8"]["outcome"] for d in self_rows if d["category"] == "normal_fatal_choice"
            ),
            "escaping_actions_all_masked_legal": all(
                d.get("pnr_escaping_masked_legal") == d["walk"]["escaping_actions"] for d in found
            ),
            "fatal_lengths": sorted(int(d["fatal_length"]) for d in self_rows),
        }
    return summary


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def world_seeds(domain: str, count: int) -> List[int]:
    from research.apex_safety_20260926 import dev_screen as ds

    return ds.screen_seeds(count, domain, NAMESPACE)


def _ints_under_seed_keys(value: Any, out: set, ok: bool = False) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ints_under_seed_keys(item, out, ok or "seed" in str(key))
    elif isinstance(value, list):
        for item in value:
            _ints_under_seed_keys(item, out, ok)
    elif ok and isinstance(value, int) and not isinstance(value, bool):
        out.add(int(value))


def extra_exclusions(domain: str) -> Dict[str, List[int]]:
    """Domains beyond the v8 serving lane's report (v5/v2 census, v8 web serving, own pair)."""
    from research.apex_safety_20260926 import dev_screen as ds

    domains: Dict[str, Sequence[str]] = dict(d5.earlier_domains("__none__"))
    domains.update(EXTRA_DOMAINS)
    for other in (DOMAIN, SMOKE_DOMAIN):
        if other != domain:
            domains[other] = (NAMESPACE,)
    domains.pop(domain, None)
    out = {
        f"{name}/{purpose}[0:{EXCLUSION_PREFIX}]": [
            ds.uint32_seed(name, purpose, i) for i in range(EXCLUSION_PREFIX)
        ]
        for name, purposes in sorted(domains.items())
        for purpose in purposes
    }
    if not V8_SERVING_INTENT.is_file():
        raise FileNotFoundError(f"v8 serving intent missing: {V8_SERVING_INTENT}")
    seeds: set = set()
    _ints_under_seed_keys(json.loads(V8_SERVING_INTENT.read_text()), seeds)
    if not seeds:
        raise ValueError("v8 serving intent has no seeds")
    out[f"observed:{V8_SERVING_INTENT}"] = sorted(seeds)
    return out


def disjointness(seeds: Sequence[int], domain: str) -> Dict[str, Any]:
    """The v8 serving lane's fail-closed seed report (superset of every earlier bank) plus
    :func:`extra_exclusions`."""
    from research.apex_veto_v8_serving_20261003 import serving_run as v8serve

    report = v8serve.seed_report({NAMESPACE: list(seeds)})
    mine = set(int(s) for s in seeds)
    extra = {name: sorted(mine & set(values)) for name, values in extra_exclusions(domain).items()}
    overlaps = {name: hits for name, hits in extra.items() if hits}
    return {
        "domain": domain,
        "count": len(mine),
        "v8_serving_seed_report": report,
        "extra_checked": sorted(extra),
        "extra_overlaps": overlaps,
        "disjoint": bool(report["disjoint"]) and not overlaps and len(mine) == len(seeds),
    }


# ---------------------------------------------------------------------------
# Run (one shard per CPU slot)
# ---------------------------------------------------------------------------
def _git() -> Dict[str, Any]:
    return base._git()


def write_new_json(path: Path, value: Any) -> None:
    base.write_new_json(path, value)


def shard_rows(rows: Sequence[Mapping[str, Any]], shard: int, shards: int) -> List[Mapping]:
    """This shard's design rows: world index ``i`` with ``i % shards == shard - 1``."""
    return [row for i, row in enumerate(rows) if i % int(shards) == int(shard) - 1]


def chunks(items: Sequence[Any], size: int) -> List[List[Any]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, event: Mapping[str, Any]) -> None:
        line = json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **event}, default=str)
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="simulate + analyze one shard")
    run.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run.add_argument("--out", type=Path, default=None, help="smoke only: shard output dir")
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--shards", type=int, default=SHARDS)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--worlds-per-mix", type=int, default=WORLDS_PER_MIX)
    run.add_argument("--frames", type=int, default=None, help="smoke only")
    run.add_argument("--mixes", default="frozen,scripted,mixed")
    run.add_argument("--batch-worlds", type=int, default=BATCH_WORLDS)
    run.add_argument("--window", type=int, default=WINDOW)
    run.add_argument("--depth", type=int, default=DEPTH)
    run.add_argument("--budget", type=int, default=NODE_BUDGET)
    run.add_argument("--death-cpu-seconds", type=float, default=DEATH_CPU_SECONDS)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--slot-timeout", type=float, default=600.0)
    run.add_argument("--max-pause-seconds", type=float, default=3600.0)
    run.add_argument("--thermal-backoff-seconds", type=float, default=60.0)
    run.add_argument("--thermal-max-backoffs", type=int, default=5)
    merge = sub.add_parser("merge", help="merge shard outputs into merged/summary.json")
    merge.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    merge.add_argument("--shards", type=int, default=SHARDS)
    merge.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "merge":
        return merge_main(args)
    return run_main(args, argv)


def run_refusal(args: argparse.Namespace) -> Optional[str]:
    """Pre-flight refusals (no side effects)."""
    if not 1 <= int(args.shard) <= int(args.shards):
        return f"--shard must be in 1..{args.shards}"
    if not args.smoke:
        if args.out is not None:
            return "--out is smoke-only (real shards write <root>/shard-<k>)"
        if (
            args.frames is not None
            or args.worlds_per_mix != WORLDS_PER_MIX
            or args.shards != SHARDS
            or args.mixes != "frozen,scripted,mixed"
            or args.depth != DEPTH
            or args.budget != NODE_BUDGET
            or args.window != WINDOW
            or args.death_cpu_seconds != DEATH_CPU_SECONDS
            or args.batch_worlds != BATCH_WORLDS
        ):
            return (
                "--frames/--worlds-per-mix/--shards/--mixes/--depth/--budget/--window/"
                "--death-cpu-seconds/--batch-worlds overrides are smoke-only"
            )
        if int(args.workers) < 2:
            return "real shards use --workers >= 2 (the slot's 2 threads)"
        if base._git()["dirty_paths"]:
            return "working tree is dirty (real shards run from a clean commit)"
    elif args.out is None:
        return "--smoke needs --out (scratch)"
    wrong = {k: os.environ.get(k) for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v}
    if wrong:
        return f"environment must have {REQUIRED_ENV}, got {wrong}"
    if not on_ac_power():
        return "`pmset -g batt` does not show 'AC Power'"
    if not lid_open():
        return "lid is closed (or AppleClamshellState unreadable)"
    return None


def shard_dir(args: argparse.Namespace) -> Path:
    return (args.out if args.smoke else args.root / f"shard-{int(args.shard)}").resolve()


def run_main(args: argparse.Namespace, argv: Sequence[str] | None) -> int:
    refusal = run_refusal(args)
    if refusal:
        print(f"refusing: {refusal}", file=sys.stderr)
        return 2
    out = shard_dir(args)
    if out.exists():
        print(f"refusing: {out} already exists (create-only)", file=sys.stderr)
        return 2
    from research.apex_safety_20260926 import dev_screen as ds

    try:
        slots = ds.acquire_cpu_slots(SLOT_ROOT, 1, float(args.slot_timeout), pool=SLOT_POOL)
    except (FileNotFoundError, TimeoutError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    try:
        return _run_shard(args, out, argv, Path(slots[0].name).name)
    finally:
        ds.release_cpu_slots(slots)


def _run_shard(args: argparse.Namespace, out: Path, argv: Sequence[str] | None, slot: str) -> int:
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    from research.apex_safety_20260926 import dev_screen as ds
    from research.compute.thermal_guard import ThermalGuard, admit_next_episode
    from src.core.config_loader import load_and_initialize_config
    from src.evaluation import safety_veto_v8
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts.tournament_eval import evaluation_profile_for_name
    from src.simd_env.eval_engine import run_simd_eval
    from src.simd_env.vector61_policy import vector61_provenance
    from web.backend import safety_veto_serving as serving

    started_total = time.monotonic()
    if serving.VARIANT_RELEASED_DEFAULT != VARIANT or float(serving.V8_LAMBDA) != LAMBDA:
        raise SystemExit("released Watch veto is not v8(lambda=8); census identity mismatch")
    domain = SMOKE_DOMAIN if args.smoke else DOMAIN
    horizon = int(args.frames) if (args.smoke and args.frames) else ds.HORIZON
    mixes = [m for m in args.mixes.split(",") if m]
    if any(m not in ds.MIXES for m in mixes):
        raise SystemExit(f"unknown mix in {mixes}")
    seeds = world_seeds(domain, int(args.worlds_per_mix))
    report = disjointness(seeds, domain)
    if not report["disjoint"]:
        raise SystemExit(f"world seeds overlap an earlier namespace: {report['extra_overlaps']}")
    if ds.sha256_file(ds.DEFAULT_CONFIG) != ds.CONFIG_SHA256:
        raise SystemExit("deployment config bytes differ from dev_screen.CONFIG_SHA256")
    load_and_initialize_config(str(ds.DEFAULT_CONFIG))
    profile = evaluation_profile_for_name(ds.PROFILE_NAME, ds.HORIZON)
    if horizon != ds.HORIZON:
        from dataclasses import replace

        profile = replace(profile, scored_horizon=horizon)
    elif profile.digest != ds.PROFILE_DIGEST:
        raise SystemExit("resolved profile digest differs from dev_screen.PROFILE_DIGEST")
    method = safety_veto_v8.method_for(LAMBDA)
    provenance = vector61_provenance(VECTOR61_FORWARD, VARIANT, LAMBDA)
    sources = {
        name: ds.sha256_file(REPO / "src" / "evaluation" / name)
        for name in (
            "safety_veto.py",
            "safety_veto_v3.py",
            "safety_veto_v5.py",
            "safety_veto_v6.py",
            "safety_veto_v7.py",
            "safety_veto_v8.py",
        )
    }
    design = ds._design_rows(seeds)
    rows_by_mix = {
        mix: shard_rows([r for r in design if r["mix"] == mix], args.shard, args.shards)
        for mix in mixes
    }

    out.mkdir(parents=True)
    for sub in ("records", "windows", "posts", "trajectories", "deaths"):
        (out / sub).mkdir()
    log = EventLog(out / "events.jsonl")
    monitor = SafetyMonitor(max_pause=float(args.max_pause_seconds), log=log).start()
    guard = ThermalGuard(ac_reader=on_ac_power)
    snapshots = ds.snapshot_checkpoints(ds.DEFAULT_CHECKPOINT_DIR, out)
    lookup = ds.agent_lookup(snapshots)
    hero = lookup[ds.CHAMPION[1]]
    write_new_json(
        out / "intent.json",
        {
            "schema_version": SCHEMA,
            "authority": AUTHORITY,
            "argv": list(sys.argv if argv is None else argv),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git": _git(),
            "smoke": bool(args.smoke),
            "shard": int(args.shard),
            "shards": int(args.shards),
            "worlds": {
                "domain": domain,
                "namespace": NAMESPACE,
                "recipe": "uint32 big-endian prefix of sha256('<domain>|<namespace>|<i>')",
                "seeds": seeds,
                "worlds_per_mix": int(args.worlds_per_mix),
                "shard_rule": "world index i with i % shards == shard - 1",
                "shard_seeds": {m: [int(r["world_seed"]) for r in rows_by_mix[m]] for m in mixes},
                "shared_by_mixes": mixes,
                "disjointness": report,
            },
            "config": {"path": str(ds.DEFAULT_CONFIG), "sha256": ds.CONFIG_SHA256},
            "profile": {"name": ds.PROFILE_NAME, "digest": profile.digest, "horizon": horizon},
            "checkpoint_snapshots": snapshots,
            "hero": {"name": ds.CHAMPION[0], "sha256": ds.CHAMPION[1]},
            "veto": {
                "variant": VARIANT,
                "lambda": LAMBDA,
                "method": method,
                "released_default": serving.VARIANT_RELEASED_DEFAULT,
                "source_sha256": sources,
            },
            "engine": {
                "engine": "simd",
                "runner": "run_simd_eval(vector61=True, hero_safety_veto='v8', "
                "hero_safety_veto_lambda=8.0)",
                "vector61_forward": VECTOR61_FORWARD,
                "provenance": provenance,
                "batch_worlds": int(args.batch_worlds),
            },
            "capture": {
                "window_decisions": int(args.window),
                "decision_point": "Vector61SimdPolicy._apply_veto (read-only wrapper)",
                "reason": "live SpaceAndHeadVeto counter deltas per env per call",
                "post_move_point": "BatchSim._resolve_collisions (read-only wrapper)",
                "trajectory": "run_simd_eval frame_observer",
            },
            "search": {
                "depth": int(args.depth),
                "node_budget_per_action": int(args.budget),
                "death_cpu_seconds": float(args.death_cpu_seconds),
                "pnr": "diagnose.analyze_death (unchanged), self deaths only",
                "fatal_frame": "diagnose_live_v5.fatal_frame_analysis (unchanged)",
            },
            "compute": {
                "slot": slot,
                "slot_pool": SLOT_POOL,
                "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                "torch_intraop": torch.get_num_threads(),
                "torch_interop": torch.get_num_interop_threads(),
                "analysis_workers": int(args.workers),
                "thermal_guard": guard.config(),
                "ac_power_at_start": True,
                "lid_open_at_start": True,
                "monitor_poll_seconds": MONITOR_POLL_SECONDS,
                "max_pause_seconds": float(args.max_pause_seconds),
            },
        },
    )
    log({"event": "start", "slot": slot, **monitor.state})

    def admit(stage: str) -> None:
        monitor.wait_if_unsafe()
        ok, reason, counts = admit_next_episode(
            guard,
            backoff_seconds=float(args.thermal_backoff_seconds),
            max_backoffs=int(args.thermal_max_backoffs),
            seconds_left=lambda: math.inf,
            budget_seconds=0.0,
            log=log,
            sleep=time.sleep,
        )
        if not ok:
            log({"event": "stop", "stage": stage, "reason": reason})
            raise RuntimeError(f"{stage}: {reason}")

    episodes: List[Dict[str, Any]] = []
    jobs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    sim_batches: List[Dict[str, Any]] = []
    lat_parts: Dict[str, List[np.ndarray]] = {"seconds": [], "reason": [], "head_risky": []}
    for mix in mixes:
        for batch in chunks(rows_by_mix[mix], int(args.batch_worlds)):
            admit(f"sim {mix}")
            batch_seeds = [int(r["world_seed"]) for r in batch]
            rosters = {
                int(r["world_seed"]): [lookup[s["member_sha256"]] for s in r["slots"]]
                for r in batch
            }
            identities = {int(r["world_seed"]): _expected_world_identity(r) for r in batch}
            recorder = CensusRecorder(len(batch), horizon, int(args.window), monitor)
            t0 = time.monotonic()
            with census_capture(recorder):
                records = run_simd_eval(
                    hero,
                    rosters[batch_seeds[0]],
                    horizon,
                    batch_seeds,
                    profile=profile,
                    opponent_specs_by_world=rosters,
                    world_identities=identities,
                    mix_id=mix,
                    frame_observer=recorder.observe_frame,
                    vector61=True,
                    hero_safety_veto=VARIANT,
                    vector61_forward=VECTOR61_FORWARD,
                    hero_safety_veto_lambda=LAMBDA,
                )
            wall = time.monotonic() - t0
            guard.record_episode(mix, wall / max(1, len(batch)))
            if recorder.frames_observed != horizon:
                raise RuntimeError(f"observer saw {recorder.frames_observed} of {horizon} frames")
            batch_deaths = 0
            for env, (row, record) in enumerate(zip(batch, records)):
                entry, job = _episode(
                    out, args, mix, row, env, record, recorder, horizon, method, provenance
                )
                episodes.append(entry)
                series = recorder.decisions.get(env, {})
                lat_parts["seconds"].append(np.asarray(series.get("seconds", []), np.float64))
                lat_parts["reason"].append(np.asarray(series.get("reason", []), np.int64))
                lat_parts["head_risky"].append(np.asarray(series.get("head_risky", []), bool))
                if job is not None:
                    jobs.append((job, entry))
                    batch_deaths += 1
                print(
                    json.dumps(
                        {
                            "mix": mix,
                            "seed": entry["world_seed"],
                            "cause": entry["death_cause"],
                            "mass": round(entry["mass_integral"], 1),
                        }
                    ),
                    flush=True,
                )
            sim_batches.append(
                {"mix": mix, "seeds": batch_seeds, "deaths": batch_deaths, "wall_seconds": wall}
            )
            log({"event": "batch_done", "mix": mix, "seeds": batch_seeds, "wall_seconds": wall})
    sim_seconds = time.monotonic() - started_total

    write_new_json(out / "episodes.json", episodes)
    write_new_json(out / "jobs.json", [{"job": list(job), "entry": entry} for job, entry in jobs])
    t_an = time.monotonic()
    results = _analyze(jobs, int(args.workers), monitor, admit, log)
    analysis_seconds = time.monotonic() - t_an
    death_rows: List[Dict[str, Any]] = []
    for (job, entry), result in zip(jobs, results):
        row = {**entry, "cause": entry["death_cause"], **result}
        row["window_file"] = Path(job[0]).name
        write_new_json(out / "deaths" / f"{entry['mix']}-{entry['world_seed']}.json", row)
        death_rows.append(row)

    latency = {k: np.concatenate(v) if v else np.zeros(0) for k, v in lat_parts.items()}
    summary = summarize_census(episodes, death_rows, latency)
    summary["timing"] = {
        "total_wall_seconds": time.monotonic() - started_total,
        "simulation_wall_seconds": sim_seconds,
        "analysis_wall_seconds": analysis_seconds,
        "simulation_batches": sim_batches,
    }
    summary["compute"] = {
        "slot": slot,
        "safety_pauses": monitor.pauses,
        "safety_paused_seconds": monitor.paused_seconds,
    }
    summary.update(
        {
            "schema_version": SCHEMA,
            "authority": AUTHORITY,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "smoke": bool(args.smoke),
            "shard": int(args.shard),
        }
    )
    for sha, path in snapshots.items():
        if ds.sha256_file(Path(path)) != sha:
            raise RuntimeError(f"checkpoint snapshot {path} changed during the run")
    write_new_json(out / "summary.json", summary)
    monitor.stop()
    log({"event": "done"})
    print(json.dumps({k: summary[k] for k in ("episode_outcomes_by_mix", "timing")}, indent=2))
    return 0


def _episode(
    out: Path,
    args: argparse.Namespace,
    mix: str,
    row: Mapping[str, Any],
    env: int,
    record: Mapping[str, Any],
    recorder: CensusRecorder,
    horizon: int,
    method: str,
    provenance: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Optional[Tuple[Any, ...]]]:
    """Validate one env's record against the capture; save it; return (entry, death job)."""
    seed = int(row["world_seed"])
    if int(record["seed"]) != seed:
        raise RuntimeError(f"record seed {record['seed']} != {seed}")
    probe = record["probes"]["safety_veto"]
    if probe["method"] != method:
        raise RuntimeError(f"hero veto is not v8(8): {probe['method']}")
    if record.get("vector61_policy") != dict(provenance):
        raise RuntimeError(f"unexpected SIMD provenance {record.get('vector61_policy')}")
    tr = {k: v[env].copy() for k, v in recorder.traj.items()}
    traj = trajectory_metrics(tr, horizon)
    if abs(traj["mass_integral"] - float(record["mass_integral"])) > 1e-9:
        raise RuntimeError(f"trajectory mass integral differs from the record: {seed}")
    died = float(record["deaths"]) >= 1
    if died != traj["died"]:
        raise RuntimeError(f"trajectory death flag differs from the record: {seed}")
    cause = record["probes"].get("death_cause")
    series = recorder.decisions.get(env, {})
    decisions = recorder.decision_count(env)
    if decisions != int(probe["counters"]["decisions"]):
        raise RuntimeError(f"capture {decisions} != probe decisions for {seed}")
    entry: Dict[str, Any] = {
        "mix": mix,
        "world_seed": seed,
        "shard": int(args.shard),
        "died": died,
        "deaths": record["deaths"],
        "death_cause": cause,
        "survival_fraction": record["survival_fraction"],
        "mass_integral": record["mass_integral"],
        "kills_record": record.get("kills"),
        "mass_at_death": traj["final_alive_mass"] if died else None,
        "veto_decisions": int(probe["counters"]["decisions"]),
        "captured_decisions": decisions,
        "v8_diagnostics": record.get("veto_diagnostics"),
        "trajectory": traj,
        "decision_metrics": decision_metrics(series),
    }
    write_new_json(out / "records" / f"{mix}-{seed}.json", {**entry, "record": record})
    np.savez_compressed(
        out / "trajectories" / f"{mix}-{seed}.npz",
        **tr,
        **{f"dec_{k}": np.asarray(v) for k, v in series.items()},
    )
    if not died:
        return entry, None
    snaps = list(recorder.buffers[env])
    post = recorder.posts.get(env)
    if post is None or post["frame"] != snaps[-1]["frame"]:
        raise RuntimeError(f"post-move capture is not the fatal frame: {mix}-{seed}")
    window = out / "windows" / f"{mix}-{seed}.npz"
    np.savez_compressed(window, **pack_census_window(snaps))
    post_path = out / "posts" / f"{mix}-{seed}.json"
    write_new_json(post_path, post)
    meta = dict(recorder.world_meta or {})
    job = (
        str(window),
        str(post_path),
        meta,
        str(cause),
        int(args.depth),
        int(args.budget),
        float(args.death_cpu_seconds),
    )
    return entry, job


def _analyze(
    jobs: Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]],
    workers: int,
    monitor: SafetyMonitor,
    admit: Callable[[str], None],
    log: Callable[[Dict[str, Any]], None],
) -> List[Dict[str, Any]]:
    """Analyze deaths with ``workers`` processes; admission-gated; SIGSTOP while unsafe."""
    work = [job for job, _ in jobs]
    if not work:
        return []
    if workers <= 1:
        results = []
        for job in work:
            admit("analysis")
            results.append(analyze_death_file(job))  # type: ignore[arg-type]
        return results
    import multiprocessing as mp

    results_by_index: Dict[int, Any] = {}
    pending: Dict[int, Any] = {}
    queue = list(enumerate(work))
    pool = mp.get_context("spawn").Pool(int(workers))
    pids = [p.pid for p in pool._pool]  # type: ignore[attr-defined]

    def signal_all(sig: int) -> None:
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def gate(stage: str) -> None:
        """Workers are stopped while any wait (safety or thermal admission) runs."""
        if monitor.safe and not pending:
            admit(stage)
            return
        signal_all(signal.SIGSTOP)
        log({"event": "analysis_sigstop", "stage": stage, **monitor.state})
        try:
            admit(stage)  # waits while unsafe or thermally not ok; raises on abort
        finally:
            signal_all(signal.SIGCONT)
            log({"event": "analysis_sigcont", "stage": stage, **monitor.state})

    try:
        while queue or pending:
            if not monitor.safe:
                gate("analysis-pause")
            while queue and len(pending) < int(workers):
                gate("analysis")
                index, job = queue.pop(0)
                pending[index] = pool.apply_async(analyze_death_file, (job,))
            for index in list(pending):
                if pending[index].ready():
                    results_by_index[index] = pending.pop(index).get()
            time.sleep(1.0)
        pool.close()
    except BaseException:
        signal_all(signal.SIGCONT)
        pool.terminate()
        raise
    finally:
        pool.join()
    return [results_by_index[i] for i in range(len(work))]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_refusal(intents: Sequence[Mapping[str, Any]], shards: int, smoke: bool) -> Optional[str]:
    if len(intents) != shards:
        return f"expected {shards} shard intents, found {len(intents)}"
    keys = (
        "worlds.domain",
        "worlds.seeds",
        "worlds.shared_by_mixes",
        "git.commit",
        "profile",
        "veto",
        "engine",
        "search",
        "capture",
        "config",
        "hero",
    )

    def pick(intent: Mapping[str, Any], dotted: str) -> Any:
        value: Any = intent
        for part in dotted.split("."):
            value = value[part]
        return value

    for key in keys:
        values = [json.dumps(pick(i, key), sort_keys=True) for i in intents]
        if len(set(values)) != 1:
            return f"shard intents disagree on {key}"
    if sorted(int(i["shard"]) for i in intents) != list(range(1, shards + 1)):
        return "shard numbers are not 1..shards"
    if any(bool(i["smoke"]) != bool(smoke) for i in intents):
        return "smoke flag mismatch"
    if not smoke and intents[0]["worlds"]["domain"] != DOMAIN:
        return "real merge needs the census domain"
    if any(i["git"]["dirty_paths"] for i in intents) and not smoke:
        return "a shard ran from a dirty tree"
    return None


def merge_main(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    shards = int(args.shards)
    dirs = [root / f"shard-{k}" for k in range(1, shards + 1)]
    missing = [str(d) for d in dirs if not (d / "summary.json").is_file()]
    if missing:
        print(f"refusing: incomplete shards {missing}", file=sys.stderr)
        return 2
    intents = [json.loads((d / "intent.json").read_text()) for d in dirs]
    refusal = merge_refusal(intents, shards, bool(args.smoke))
    if refusal:
        print(f"refusing: {refusal}", file=sys.stderr)
        return 2
    out = root / "merged"
    if out.exists():
        print(f"refusing: {out} already exists (create-only)", file=sys.stderr)
        return 2
    episodes: List[Dict[str, Any]] = []
    deaths: List[Dict[str, Any]] = []
    lat: Dict[str, List[np.ndarray]] = {"seconds": [], "reason": [], "head_risky": []}
    timing = []
    for d in dirs:
        episodes.extend(json.loads((d / "episodes.json").read_text()))
        for path in sorted((d / "deaths").glob("*.json")):
            deaths.append(json.loads(path.read_text()))
        for path in sorted((d / "trajectories").glob("*.npz")):
            with np.load(path) as data:
                lat["seconds"].append(data["dec_seconds"] if "dec_seconds" in data else np.zeros(0))
                lat["reason"].append(data["dec_reason"] if "dec_reason" in data else np.zeros(0))
                lat["head_risky"].append(
                    data["dec_head_risky"] if "dec_head_risky" in data else np.zeros(0, bool)
                )
        shard_summary = json.loads((d / "summary.json").read_text())
        timing.append({"shard": d.name, **shard_summary["timing"], **shard_summary["compute"]})
    seeds = intents[0]["worlds"]["seeds"]
    mixes = intents[0]["worlds"]["shared_by_mixes"]
    expected = {(m, int(s)) for m in mixes for s in seeds}
    got = [(e["mix"], int(e["world_seed"])) for e in episodes]
    if set(got) != expected or len(got) != len(expected):
        print("refusing: merged episodes do not cover the design exactly once", file=sys.stderr)
        return 2
    died = {(e["mix"], int(e["world_seed"])) for e in episodes if e["died"]}
    if {(d["mix"], int(d["world_seed"])) for d in deaths} != died or len(deaths) != len(died):
        print("refusing: death files do not match the episodes' deaths", file=sys.stderr)
        return 2
    latency = {k: np.concatenate(v) if v else np.zeros(0) for k, v in lat.items()}
    summary = summarize_census(episodes, deaths, latency)
    summary.update(
        {
            "schema_version": SCHEMA,
            "authority": AUTHORITY,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "smoke": bool(args.smoke),
            "shards": timing,
            "intent_git": intents[0]["git"],
            "domain": intents[0]["worlds"]["domain"],
        }
    )
    out.mkdir()
    write_new_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("episode_outcomes_by_mix", "analysis")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
