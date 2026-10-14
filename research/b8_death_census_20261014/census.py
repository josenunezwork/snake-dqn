#!/usr/bin/env python3
"""DEVELOPMENT DIAGNOSTIC: death census of the M4 B8 no-veto student vs SCRIPTED opponents.

Not a screen, not gate evidence, no candidate and no decision rule. It asks why the B8
student (``ego2s-bv8``, no decision-time veto) dies more often than champion_a5 + v8 against
scripted opponents (M4 Phase R clause c6: scripted survival -0.156), and whether the v8 veto,
run in the shadow on the student's own Q, would have acted near those deaths.

Engine: exactly ``research/redesign_m2_20261008/dev_probe.py``'s student arm on the
dev-probe worlds (``round_index`` 52, the first 32 distillation-namespace worlds, mix
``scripted``, H5000, ``promotion-v2-watch-rect``, grid engine, ``ego2s_forward="rowwise"``,
``ego2s_mask="advisory"``, no veto), with the opt-in ``vector61_backend="numba"`` and
``scripted_anchors="fast"``. Those two are record-identical by claim
(``tests/test_eval_v61_backend_identity.py``); the ``identity`` subcommand re-checks the
claim (whole records and per-frame world digests) on 2 worlds before the census uses them.
The references (frp3-s12 + v8 = the incumbent, champion_a5 + v8) run the same worlds.

Capture (read-only; checked: the census records equal the existing dev-probe rows world
by world, so the instrumentation changes nothing the hero does):

* **Shadow v8** (B8 arms): the hero policy is :class:`ShadowEgo2sPolicy`, a drop-in for
  ``Ego2sSimdPolicy(mask_mode="advisory")`` that returns the same actions and also runs
  ``src.simd_env.v8_nb.v8_decide(sim, rows, q, advisory_mask, 8.0, features=True)`` on the
  student's own Q (the M3-C/M4 ``make_shield`` numba path; never executed). Its ``base``
  must equal the student's action (counted in ``checks.base_mismatch``).
* **Decision-time world** (every arm): the eval sim class is subclassed. ``step`` (called
  after action selection, on the prepared world) records the hero's action and the v8
  per-action diagnostics (tail-aware landing area, v2 counts / need, head risk; these do not
  depend on Q, so the references are measured the same way) and keeps a deep copy of the
  prepared world. ``_resolve_collisions`` records the hero's collision partners (slot, kind,
  offset of the hit cell in the partner's post-move body) before deaths are applied.
* **One-frame counterfactual** at every fatal frame: the copied world is stepped once per
  hero action (0..5) with the opponents' actual actions. The actual action must reproduce the
  actual post-step world digest (``grid_h5000_identity.env_digest``), checked per death.

Definitions (per death; ``t`` = fatal decision, the last decision before death):

* ``enclosed`` at a decision: every hero action's v8 tail-aware landing area is below v2's
  ``need`` (``min(L, cap)``, cap ``min(160, max(32, 2L))``). ``trap_run``: consecutive
  enclosed decisions ending at ``t``. An **enclosure death** has ``trap_run >= 3`` (the trap
  had formed at least two decisions before the fatal one); ``trap_run`` 0-2 is an immediate
  collision.
* ``forced``: no hero action survives the fatal frame (counterfactual, opponents' actual
  moves). ``v8_action_fatal``: v8's shadow action at ``t`` dies in the counterfactual.
* ``v8_differs_within_k``: v8's shadow action differs from the hero's at any of the last
  ``k`` decisions (``k = 1`` is the fatal decision only).
* ``boost_involved``: the hero executed a boost on the fatal decision or on any of the 4
  decisions before it.
* ``opponent_head_involved``: head-on death, or an enemy-body hit within 2 cells of the
  partner's new head (offset <= 2: cut off), or the taken action was v8 head-risky.
* Trap onset (two-frame counterfactual): at every enclosure entry (enclosed at ``t0``, open
  at ``t0 - 1``) the world of decision ``t0 - 1`` is stepped once per hero action with the
  opponents' actual moves and enclosure is re-tested on the result. ``avoidable_1step``:
  some action survives and is not enclosed afterwards (an "escape"); one step only, so an
  escape is necessary, not sufficient, for avoiding the trap. v8's landing area is only
  meaningful for an action that does not collide at once, so every enclosure test uses
  advisory-safe actions only (no such action: enclosed).

Usage::

  ./venv/bin/python research/b8_death_census_20261014/census.py identity --worlds 2
  ./venv/bin/python research/b8_death_census_20261014/census.py run --arm b8_s0
  ./venv/bin/python research/b8_death_census_20261014/census.py summarize
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, deque  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple  # noqa

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
OUT_ROOT = ARTIFACTS / "b8-death-census-20261014"
B8_DIR = ARTIFACTS / "redesign-m4b-mac-20261013"
PROBE_B8 = ARTIFACTS / "redesign-m4-mac-probes-20261013"
PROBE_INCUMBENT = ARTIFACTS / "redesign-m3c-probes-20261008" / "incumbent__dev__v8.jsonl"

MIX = "scripted"
ROUND_INDEX = 52
WORLDS = 32
HORIZON = 5000
LAM = 8.0
WINDOW = 30  # detailed decision snapshots kept per env
HISTORY = 20  # hero actions reported per death
K_MAX = 10
ENCLOSURE_MIN = 3
BOOST_WINDOW = 5
B8_ARMS = tuple(f"b8_s{i}" for i in range(5))
REF_ARMS = ("incumbent", "champion")
ARMS = B8_ARMS + REF_ARMS
CAUSE_LABEL = {0: None, 1: "wall", 2: "self", 3: "head_on", 4: "enemy_body"}
CAUSES = ("wall", "self", "head_on", "enemy_body")
OUTCOME_LABEL = {0: "kept", 1: "vetoed", 2: "no_spacious", -1: None}
V5_REASON = {0: "kept", 1: "v2", 2: "same_dir", 3: "landing_v2", 4: "landing_none"}


# ----------------------------------------------------------------------------- context
def load_context(worlds: int = WORLDS) -> Tuple[Dict[str, Any], List[int], Dict[int, list]]:
    """(gid context, the first ``worlds`` dev-probe seeds, their scripted rosters)."""
    from research.redesign_m2_20261008 import dev_probe
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    ctx = gid._context(1)
    seeds = dev_probe.probe_seeds(WORLDS, ROUND_INDEX)
    rows = {
        int(r["world_seed"]): [ctx["lookup"][s["member_sha256"]] for s in r["slots"]]
        for r in ctx["ds"]._design_rows(seeds)
        if r["mix"] == MIX
    }
    seeds = seeds[: int(worlds)]
    return ctx, seeds, {s: rows[s] for s in seeds}


def b8_path(arm: str) -> Path:
    return B8_DIR / f"B8_s{int(arm[-1])}" / "final.pth"


def arm_call(arm: str, ctx: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """(hero_spec, run_simd_eval kwargs): dev_probe's student / baseline arms + champion."""
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_m2_20261008 import ni_spec

    lookup = ctx["lookup"]
    if arm in B8_ARMS:
        return lookup[ni_spec.FRP3_S12[1]], {
            "hero_ego2s": str(b8_path(arm)),
            "ego2s_forward": "rowwise",
            "ego2s_mask": "advisory",
        }
    veto = {"hero_safety_veto": "v8", "hero_safety_veto_lambda": LAM}
    if arm == "incumbent":
        return lookup[ni_spec.FRP3_S12[1]], veto
    if arm == "champion":
        return lookup[dev_screen.CHAMPION[1]], veto
    raise SystemExit(f"unknown arm {arm}")


def run_eval(
    ctx: Mapping[str, Any],
    seeds: Sequence[int],
    rosters: Mapping[int, list],
    arm: str,
    frames: int,
    backend: str,
    anchors: str,
    frame_observer=None,
) -> List[Dict[str, Any]]:
    from src.simd_env import eval_engine as ee

    profile = ctx["profile"]
    if frames != profile.scored_horizon:
        profile = dataclasses.replace(profile, scored_horizon=frames)
    hero, kwargs = arm_call(arm, ctx)
    return ee.run_simd_eval(
        hero,
        rosters[seeds[0]],
        frames,
        list(seeds),
        profile=profile,
        opponent_specs_by_world=dict(rosters),
        mix_id=MIX,
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        vector61_backend=backend,
        scripted_anchors=anchors,
        frame_observer=frame_observer,
        **kwargs,
    )


def spec_kind(spec: Sequence[Any]) -> str:
    kind, ref = spec[0], spec[1]
    return f"{kind}:{Path(str(ref)).name if kind == 'checkpoint' else ref}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git() -> Dict[str, Any]:
    def run(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


# ----------------------------------------------------------------------------- sim cloning
def clone_sim(sim: Any) -> Any:
    """Deep copy of a (Grid)BatchSim that steps identically (views re-established)."""
    memo: Dict[int, Any] = {}
    kernels = getattr(sim, "_kernels", None)
    if kernels is not None:
        memo[id(kernels)] = kernels
    c = copy.deepcopy(sim, memo)
    if getattr(c, "_grids_ready", False) and hasattr(c, "_owner_pad"):
        P, H, W = c._grid_pad, c.grid_h, c.grid_w
        c._owner = c._owner_pad[:, P : P + H, P : P + W]
        c._slot = c._slot_pad[:, P : P + H, P : P + W]
        c._food = c._food_pad[:, P : P + H, P : P + W]
    return c


def ordered_cells(sim: Any, e: int, s: int) -> np.ndarray:
    n = int(sim.seg_count[e, s])
    idx = (int(sim.head_ptr[e, s]) - np.arange(n)) % int(sim.cap)
    return np.asarray(sim.bodies[e, s, idx], dtype=np.int64)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


# ----------------------------------------------------------------------------- recorder
class Recorder:
    """Per-env decision series and snapshot rings, collision partners, death records."""

    def __init__(
        self,
        arm: str,
        seeds: Sequence[int],
        kinds: Sequence[Sequence[str]],
        horizon: int,
        shadow: bool,
    ) -> None:
        self.arm, self.seeds, self.kinds = arm, [int(s) for s in seeds], [list(k) for k in kinds]
        self.E, self.H, self.shadow_on = len(self.seeds), int(horizon), bool(shadow)
        self.sim: Any = None
        self.meta: Dict[str, Any] = {}
        self.t = 0
        E, H = self.E, self.H
        self.series: Dict[str, np.ndarray] = {
            "action": np.full((E, H), -1, np.int8),
            "v8_action": np.full((E, H), -1, np.int8),
            "v8_outcome": np.full((E, H), -1, np.int8),
            "length": np.zeros((E, H), np.int32),
            "need": np.zeros((E, H), np.int32),
            "best_area": np.full((E, H), -1, np.int32),
            "taken_area": np.full((E, H), -1, np.int32),
            "adv_any": np.zeros((E, H), bool),
            "taken_adv_safe": np.zeros((E, H), bool),
            "boost_exec": np.zeros((E, H), bool),
        }
        self.ring: Dict[int, Deque[Dict[str, Any]]] = {e: deque(maxlen=WINDOW) for e in range(E)}
        self.pending: Dict[int, Dict[str, Any]] = {}
        self.pre: Any = None
        self.prev_pre: Any = None  # the previous decision's prepared world (trap onsets)
        self.prev_actions: Optional[np.ndarray] = None
        self.base_step: Any = None  # the uninstrumented sim step (set by instrumented())
        self.onsets: Dict[int, Dict[int, Dict[str, Any]]] = {e: {} for e in range(E)}
        self.pre_alive = np.zeros(E, bool)
        self.actions: Optional[np.ndarray] = None
        self.collisions: Dict[int, Dict[str, Any]] = {}
        self.deaths: Dict[int, Dict[str, Any]] = {}
        self.checks: Counter = Counter()
        self.seconds: Counter = Counter()

    # -- B8 shadow (inside the hero policy's actions()) ------------------------------
    def shadow(self, sim: Any, rows: np.ndarray, q: np.ndarray, adv: np.ndarray, act) -> None:
        from src.simd_env.v8_nb import v8_decide

        t0 = time.perf_counter()
        res = v8_decide(sim, rows, q, adv, LAM, features=True)
        for i, (e, _) in enumerate(rows):
            self.pending[int(e)] = self._diag(res, i, adv[i], q=q[i], act=int(act[i]))
            self.checks["shadow_decisions"] += 1
            if int(res.base[i]) != int(act[i]):
                self.checks["base_mismatch"] += 1
        self.seconds["shadow"] += time.perf_counter() - t0

    @staticmethod
    def _diag(res, i: int, adv, q=None, act=None) -> Dict[str, Any]:
        d = {
            "adv": np.asarray(adv, bool).copy(),
            "outcome": int(res.outcome[i]),
            "v5_reason": int(res.v5_reason[i]),
            "counts": res.counts[i].copy(),
            "need": int(res.need[i]),
            "cap": int(res.cap[i]),
            "area_cap": int(res.area_cap[i]),
            "area": res.area[i].copy(),
            "head_risky": res.head_risky[i].copy(),
            "head_waived": res.head_waived[i].copy(),
            "landing": res.landing[i].copy(),
        }
        if q is not None:
            d.update(
                q=np.asarray(q, np.float32).copy(),
                act=act,
                v8_action=int(res.action[i]),
                v8_base=int(res.base[i]),
                v5_action=int(res.v5_action[i]),
                v7_action=int(res.v7_action[i]),
            )
        return d

    # -- decision-time world (inside the sim's step, before the transition) ----------
    def pre_step(self, sim: Any, actions: np.ndarray) -> None:
        from src.simd_env.v8_nb import v8_decide

        t0 = time.perf_counter()
        if self.sim is None:
            self.sim = sim
            self.meta = {
                "grid_w": int(sim.grid_w),
                "grid_h": int(sim.grid_h),
                "num_snakes": int(sim.S),
                "min_boost_length": int(sim.cfg.min_boost_length),
                "boost_cost_frames": int(sim.cfg.boost_length_cost_frames),
                "mechanics_version": int(sim.cfg.mechanics_version),
            }
        elif sim is not self.sim:
            raise RuntimeError("one recorder per run_simd_eval call")
        if self.t >= self.H:
            raise RuntimeError("more steps than the horizon")
        t = self.t
        actions = np.asarray(actions, dtype=np.int64)
        alive = np.asarray(sim.alive[:, 0], bool).copy()
        envs = np.flatnonzero(alive)
        if len(envs):
            adv_all = sim.get_advisory_action_mask()[envs, 0].astype(bool)
            if self.shadow_on:
                missing = [int(e) for e in envs if int(e) not in self.pending]
                if missing:
                    raise RuntimeError(f"no shadow decision for alive hero envs {missing}")
            else:
                rows = np.stack([envs, np.zeros_like(envs)], axis=1)
                q0 = np.zeros((len(envs), 6), np.float32)
                res = v8_decide(sim, rows, q0, adv_all, LAM, features=True)
                for i, e in enumerate(envs):
                    self.pending[int(e)] = self._diag(res, i, adv_all[i])
            min_boost = self.meta["min_boost_length"]
            resolved_all = sim.get_resolved_action_mask()
            S = int(sim.S)
            for i, e in enumerate(envs):
                e = int(e)
                d = self.pending[e]
                a = int(actions[e, 0])
                if not np.array_equal(d["adv"], adv_all[i]):
                    self.checks["adv_mismatch"] += 1
                if self.shadow_on and d["act"] != a:
                    self.checks["shadow_action_mismatch"] += 1
                hx, hy = (int(v) for v in sim.bodies[e, 0, int(sim.head_ptr[e, 0])])
                opps = []
                for j in range(1, S):
                    if not sim.alive[e, j]:
                        continue
                    ox, oy = (int(v) for v in sim.bodies[e, j, int(sim.head_ptr[e, j])])
                    opps.append(
                        {
                            "slot": j,
                            "kind": self.kinds[e][j - 1],
                            "head": [ox, oy],
                            "dist": abs(ox - hx) + abs(oy - hy),
                            "length": int(sim.length[e, j]),
                        }
                    )
                length = int(sim.length[e, 0])
                snap = {
                    "t": t,
                    "frame": int(sim.frame[e]),
                    "action": a,
                    "length": length,
                    "seg_count": int(sim.seg_count[e, 0]),
                    "boost_frames": int(sim.boost_frames[e, 0]),
                    "direction": int(sim.direction[e, 0]),
                    "head": [hx, hy],
                    "boost_legal": length >= min_boost,
                    "resolved": np.asarray(resolved_all[e, 0], bool).copy(),
                    "opponents": opps,
                    **d,
                }
                self.ring[e].append(snap)
                s = self.series
                s["action"][e, t] = a
                s["v8_action"][e, t] = d.get("v8_action", -1)
                s["v8_outcome"][e, t] = d["outcome"]
                s["length"][e, t] = length
                s["need"][e, t] = d["need"]
                # v8's landing area is only meaningful for an action that does not
                # collide at once (the flood starts on the new head cell unchecked), so
                # the enclosure test uses advisory-safe actions only (none: 0).
                safe_area = np.where(d["adv"], d["area"], 0)
                s["best_area"][e, t] = int(safe_area.max())
                s["taken_area"][e, t] = int(d["area"][a]) if d["adv"][a] else 0
                s["adv_any"][e, t] = bool(d["adv"].any())
                s["taken_adv_safe"][e, t] = bool(d["adv"][a])
            if t >= 1 and self.prev_pre is not None:
                t2 = time.perf_counter()
                s = self.series
                for e in envs:
                    e = int(e)
                    entered = s["best_area"][e, t] < s["need"][e, t]
                    was_open = s["action"][e, t - 1] >= 0 and not (
                        s["best_area"][e, t - 1] < s["need"][e, t - 1]
                    )
                    if entered and was_open:
                        self.onsets[e][t] = self._onset_counterfactual(e, t)
                self.seconds["onset"] += time.perf_counter() - t2
        self.pending = {}
        self.pre_alive = alive
        self.actions = actions.copy()
        self.seconds["pre"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        self.pre = clone_sim(sim) if len(envs) else None
        self.seconds["clone"] += time.perf_counter() - t1

    # -- collision partners (inside _resolve_collisions, before deaths apply) ---------
    def capture_collisions(self, sim: Any) -> None:
        det = getattr(sim, "_det", None)
        if det is None:
            return
        for e in np.flatnonzero(np.asarray(sim.alive[:, 0], bool) & det["event_env"]):
            e = int(e)
            head_on = [int(j) for j in np.flatnonzero(det["head_on"][e, 0])]
            body = [int(j) for j in np.flatnonzero(det["body"][e, 0])]
            wall, selfhit = bool(det["wall"][e, 0]), bool(det["self"][e, 0])
            if not (wall or selfhit or head_on or body):
                continue
            trav = [
                [int(v) for v in sim._trav[e, 0, k]]
                for k in range(2)
                if bool(sim._trav_valid[e, 0, k])
            ]
            partners = []
            for j in sorted(set(head_on) | set(body)):
                cells = ordered_cells(sim, e, j)
                offsets = sorted(
                    {
                        int(i)
                        for c in trav
                        for i in np.flatnonzero((cells[:, 0] == c[0]) & (cells[:, 1] == c[1]))
                    }
                )
                partners.append(
                    {
                        "slot": j,
                        "kind": self.kinds[e][j - 1],
                        "type": "head_on" if j in head_on else "body",
                        "length": int(sim.length[e, j]),
                        "boost_move": bool(sim._trav_valid[e, j, 1]),
                        "hit_offsets": offsets,
                    }
                )
            self_offsets: List[int] = []
            if selfhit:
                cells = ordered_cells(sim, e, 0)
                self_offsets = sorted(
                    {
                        int(i)
                        for c in trav
                        for i in np.flatnonzero((cells[:, 0] == c[0]) & (cells[:, 1] == c[1]))
                        if i > 0
                    }
                )
            self.collisions[e] = {
                "wall": wall,
                "self": selfhit,
                "self_hit_offsets": self_offsets,
                "traversed": trav,
                "partners": partners,
                "hero_length_post_move": int(sim.length[e, 0]),
            }

    # -- after the transition ---------------------------------------------------------
    def post_step(self, sim: Any, base_step) -> None:
        t0 = time.perf_counter()
        t = self.t
        cause = np.asarray(sim.get_death_cause()[:, 0])
        boosted = np.asarray(sim.get_boosted_this_step()[:, 0], bool)
        self.series["boost_exec"][:, t] = boosted & self.pre_alive
        died = self.pre_alive & (cause != 0)
        for e in np.flatnonzero(died):
            e = int(e)
            self.deaths[e] = self._death(sim, e, int(cause[e]), base_step)
        self.collisions = {}
        self.prev_pre, self.prev_actions = self.pre, self.actions
        self.pre = None
        self.t += 1
        self.seconds["post"] += time.perf_counter() - t0

    def _onset_counterfactual(self, e: int, t: int) -> Dict[str, Any]:
        """At an enclosure entry ``t`` (enclosed at ``t``, open at ``t - 1``): step the
        world of decision ``t - 1`` once per hero action (opponents' actual moves) and
        re-test enclosure on the resulting world (``escape`` = survives and not enclosed)."""
        from src.simd_env.v8_nb import v8_decide

        assert self.prev_actions is not None
        out: Dict[int, Dict[str, Any]] = {}
        for a in range(6):
            c = clone_sim(self.prev_pre)
            acts = self.prev_actions.copy()
            acts[e, 0] = a
            self.base_step(c, acts)
            if not bool(c.alive[e, 0]):
                out[a] = {"survives": False, "enclosed": None, "best_safe_area": None}
                continue
            adv = c.get_advisory_action_mask()[e, 0].astype(bool)
            res = v8_decide(
                c,
                np.array([[e, 0]], dtype=np.int64),
                np.zeros((1, 6), np.float32),
                adv[None],
                LAM,
                features=True,
            )
            best = int(np.where(adv, res.area[0], 0).max())
            out[a] = {
                "survives": True,
                "best_safe_area": best,
                "need": int(res.need[0]),
                "enclosed": best < int(res.need[0]),
            }
        taken = int(self.prev_actions[e, 0])
        if out[taken]["enclosed"]:
            self.checks["onset_taken_enclosed_ok"] += 1
        else:
            self.checks["onset_taken_enclosed_bad"] += 1
        escapes = [a for a in range(6) if out[a]["survives"] and not out[a]["enclosed"]]
        v8a = int(self.series["v8_action"][e, t - 1]) if self.shadow_on else None
        prev = self.ring[e][-2] if len(self.ring[e]) >= 2 else None
        return {
            "t0": t,
            "t_pre": t - 1,
            "length": int(self.series["length"][e, t - 1]),
            "taken_action": taken,
            "v8_action": v8a,
            "adv_safe_pre": (
                [int(x) for x in np.flatnonzero(prev["adv"])]
                if prev is not None and prev["t"] == t - 1
                else None
            ),
            "actions": out,
            "escape_actions": escapes,
            "v8_escapes": (v8a in escapes) if self.shadow_on else None,
            "only_boost_escape": bool(escapes) and all(a >= 3 for a in escapes),
        }

    def _counterfactual(self, sim: Any, e: int, base_step) -> Dict[str, Any]:
        from research.redesign_scope_20261007 import grid_h5000_identity as gid

        assert self.actions is not None and self.pre is not None
        actual = int(self.actions[e, 0])
        outcome = {}
        digest_ok = None
        for a in range(6):
            c = clone_sim(self.pre)
            acts = self.actions.copy()
            acts[e, 0] = a
            base_step(c, acts)
            code = int(c.get_death_cause()[e, 0])
            outcome[a] = {"survives": bool(c.alive[e, 0]) and code == 0, "cause": CAUSE_LABEL[code]}
            if a == actual:
                digest_ok = gid.env_digest(c, e) == gid.env_digest(sim, e)
        self.checks["cf_digest_ok" if digest_ok else "cf_digest_bad"] += 1
        return {"actions": outcome, "actual_reproduced": bool(digest_ok)}

    def _death(self, sim: Any, e: int, code: int, base_step) -> Dict[str, Any]:
        ring = list(self.ring[e])
        fatal = ring[-1]
        if fatal["t"] != self.t:
            raise RuntimeError("fatal snapshot is not the current decision")
        cf = self._counterfactual(sim, e, base_step)
        if cf["actions"][fatal["action"]]["cause"] != CAUSE_LABEL[code]:
            self.checks["cf_cause_bad"] += 1
        else:
            self.checks["cf_cause_ok"] += 1
        return {
            "fatal": fatal,
            "history": ring[-HISTORY:],
            "window": ring,
            "cause": CAUSE_LABEL[code],
            "collision": self.collisions.get(e),
            "counterfactual": cf,
            "onsets": dict(self.onsets[e]),
        }

    # -- frame observer (after each transition) ---------------------------------------
    def observe(self, payload: Mapping[str, Any]) -> None:
        t = int(payload["frame"])
        if t != self.t - 1:
            self.checks["observer_frame_mismatch"] += 1
        pre = payload["pre"]
        alive = np.asarray(pre["alive"], bool)
        acts = np.asarray(pre["actions"])
        mine = self.series["action"][:, t]
        bad = alive & (acts != mine)
        self.checks["observer_action_mismatch"] += int(bad.sum())


@contextmanager
def instrumented(recorder: Recorder) -> Iterator[Recorder]:
    """Patch the eval sim class (step / _resolve_collisions) and, for B8, the hero policy."""
    from src.simd_env import ego2s_policy as ep
    from src.simd_env import eval_engine as ee

    orig_cls_fn = ee._terminal_hero_sim_class
    base = orig_cls_fn("grid")
    base_step = base.step
    recorder.base_step = base_step

    class CensusGridSim(base):  # type: ignore[misc, valid-type]
        def step(self, actions, active_env_mask=None):  # type: ignore[no-untyped-def]
            if not getattr(self, "_policy_phase_prepared", False):
                return base_step(self, actions, active_env_mask)
            if active_env_mask is not None and not np.all(active_env_mask):
                raise RuntimeError("census expects every env active")
            recorder.pre_step(self, actions)
            out = base_step(self, actions, active_env_mask)
            recorder.post_step(self, base_step)
            return out

        def _resolve_collisions(self):  # type: ignore[no-untyped-def]
            if self is recorder.sim:
                recorder.capture_collisions(self)
            return super()._resolve_collisions()

    orig_policy = ep.Ego2sSimdPolicy

    class ShadowEgo2sPolicy(orig_policy):  # type: ignore[misc, valid-type]
        """``Ego2sSimdPolicy`` (advisory) with the shadow v8 on the same Q."""

        def actions(self, masks, sim, slots):  # type: ignore[no-untyped-def]
            if self.mask_mode != "advisory":
                raise RuntimeError("the census replicates the advisory selection only")
            rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            q = ep.student_q(self.net, sim, rows, self.forward).numpy()
            m = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
            adv = m.copy()
            m[~m.any(axis=1), :3] = True
            act = np.where(m, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int64)
            recorder.shadow(sim, rows, q, adv, act)
            return act

    ee._terminal_hero_sim_class = lambda engine: (
        CensusGridSim if engine == "grid" else orig_cls_fn(engine)
    )
    if recorder.shadow_on:
        ep.Ego2sSimdPolicy = ShadowEgo2sPolicy
    try:
        yield recorder
    finally:
        ee._terminal_hero_sim_class = orig_cls_fn
        ep.Ego2sSimdPolicy = orig_policy


# ----------------------------------------------------------------------------- analysis
def trailing_run(flags: Sequence[bool]) -> int:
    """Number of consecutive True values at the end of ``flags``."""
    n = 0
    for f in reversed(list(flags)):
        if not f:
            break
        n += 1
    return n


def within_k(differs: Sequence[bool], k_max: int = K_MAX) -> Dict[int, bool]:
    """``{k: any(differs[-k:])}`` for k = 1..k_max (``differs`` ends at the fatal decision)."""
    d = list(differs)
    return {k: bool(any(d[-k:])) for k in range(1, k_max + 1)}


def analyze_death(
    raw: Mapping[str, Any],
    series: Mapping[str, np.ndarray],
    e: int,
    shadow: bool,
    min_boost: int,
    arena_cells: int = 1,
) -> Dict[str, Any]:
    """The per-death census row (JSON-ready) from the raw capture and the env's series."""
    fatal = raw["fatal"]
    t = int(fatal["t"])
    act = series["action"][e, : t + 1].astype(int)
    v8a = series["v8_action"][e, : t + 1].astype(int)
    best = series["best_area"][e, : t + 1]
    need = series["need"][e, : t + 1]
    outcome = series["v8_outcome"][e, : t + 1]
    boost = series["boost_exec"][e, : t + 1]
    enclosed = best < need
    trap_run = trailing_run(enclosed)
    ns_run = trailing_run(outcome == 2)
    cf = raw["counterfactual"]["actions"]
    survivors = [a for a in range(6) if cf[a]["survives"]]
    a_fatal = int(fatal["action"])
    area = np.asarray(fatal["area"])
    row: Dict[str, Any] = {
        "fatal_t": t,
        "frame": int(fatal["frame"]),
        "frames_alive": t + 1,
        "cause": raw["cause"],
        "length": int(fatal["length"]),
        "boost_frames": int(fatal["boost_frames"]),
        "fatal_action": a_fatal,
        "fatal_action_is_boost": a_fatal >= 3,
        "boost_exec_fatal": bool(boost[t]),
        "boost_exec_last5": bool(boost[max(0, t - BOOST_WINDOW + 1) :].any()),
        "boost_exec_last20": bool(boost[max(0, t - 19) :].any()),
        "boost_involved": bool(boost[max(0, t - BOOST_WINDOW + 1) :].any()),
        "last_actions": [int(x) for x in act[-HISTORY:]],
        "taken_adv_safe": bool(np.asarray(fatal["adv"])[a_fatal]),
        "adv_any": bool(np.asarray(fatal["adv"]).any()),
        "adv_safe_actions": [int(a) for a in np.flatnonzero(np.asarray(fatal["adv"]))],
        "need": int(fatal["need"]),
        "area": [int(x) for x in area],
        "area_cap": int(fatal["area_cap"]),
        "counts": [int(x) for x in fatal["counts"]],
        "best_safe_area_fatal": int(best[-1]),
        "taken_area_fatal": int(series["taken_area"][e, t]),
        "arena_cells": int(arena_cells),
        "hero_fill_frac": float(int(fatal["length"]) / arena_cells),
        "occupied_frac": float(
            (int(fatal["length"]) + sum(o["length"] for o in fatal["opponents"])) / arena_cells
        ),
        "enclosed_fatal": bool(enclosed[-1]),
        "trap_run": trap_run,
        "trap_run_censored": trap_run == t + 1,
        "enclosure": trap_run >= ENCLOSURE_MIN,
        "no_spacious_run": ns_run,
        "taken_head_risky": bool(np.asarray(fatal["head_risky"])[a_fatal]),
        "head_risky": [bool(x) for x in fatal["head_risky"]],
        "nearest_opp_head_dist": min((o["dist"] for o in fatal["opponents"]), default=None),
        "opponents": fatal["opponents"],
        "cf_survivors": survivors,
        "forced": not survivors,
        "cf_survivor_spacious": [a for a in survivors if area[a] >= int(fatal["need"])],
        "cf_actual_reproduced": bool(raw["counterfactual"]["actual_reproduced"]),
        "cf_causes": {str(a): cf[a]["cause"] for a in range(6)},
    }
    col = raw.get("collision") or {}
    partners = col.get("partners", [])
    killer = None
    if raw["cause"] == "wall":
        killer = {"type": "wall"}
    elif raw["cause"] == "self":
        killer = {"type": "own_body", "hit_offsets": col.get("self_hit_offsets", [])}
    else:
        want = "head_on" if raw["cause"] == "head_on" else "body"
        cands = [p for p in partners if p["type"] == want] or partners
        if cands:
            p = cands[0]
            region = (
                "head"
                if p["type"] == "head_on"
                else ("neck" if p["hit_offsets"] and min(p["hit_offsets"]) <= 2 else "body")
            )
            killer = {
                "type": f"opponent_{region}",
                "kind": p["kind"],
                "slot": p["slot"],
                "length": p["length"],
                "boost_move": p["boost_move"],
                "hit_offsets": p["hit_offsets"],
                "n_partners": len(partners),
            }
    row["killer"] = killer
    row["opponent_head_involved"] = bool(
        raw["cause"] == "head_on"
        or (killer is not None and killer.get("type") == "opponent_neck")
        or row["taken_head_risky"]
    )
    row["opponent_head_components"] = {
        "head_on": raw["cause"] == "head_on",
        "cut_off_neck_hit": killer is not None and killer.get("type") == "opponent_neck",
        "taken_head_risky": row["taken_head_risky"],
    }
    # trap onset: the entry into the final enclosed run (first enclosed decision t0)
    onset: Optional[Dict[str, Any]] = None
    if trap_run >= 1:
        t0 = t - trap_run + 1
        o = (raw.get("onsets") or {}).get(t0)
        if o is not None:
            onset = {
                "t0": t0,
                "decisions_before_death": t - t0 + 1,
                "length": o["length"],
                "taken_action": o["taken_action"],
                "v8_action": o["v8_action"],
                "adv_safe_pre": o["adv_safe_pre"],
                "escape_actions": o["escape_actions"],
                "v8_escapes": o["v8_escapes"],
                "only_boost_escape": o["only_boost_escape"],
                "kind": "avoidable_1step" if o["escape_actions"] else "unavoidable_1step",
                "actions": o["actions"],
            }
    row["trap_onset"] = onset
    if shadow:
        differs = act != v8a
        row.update(
            {
                "last_v8_actions": [int(x) for x in v8a[-HISTORY:]],
                "v8_action_fatal_decision": int(fatal["v8_action"]),
                "v8_outcome_fatal": OUTCOME_LABEL[int(fatal["outcome"])],
                "v8_v5_reason_fatal": V5_REASON.get(int(fatal["v5_reason"])),
                "v8_differs_fatal": bool(differs[-1]),
                "v8_differs_within_k": {str(k): v for k, v in within_k(differs).items()},
                "v8_differs_count_last10": int(differs[-10:].sum()),
                "v8_differs_any_in_trap_run": (
                    bool(differs[t - trap_run + 1 :].any()) if trap_run else False
                ),
                "v8_action_survives": bool(cf[int(fatal["v8_action"])]["survives"]),
                "v8_action_also_fatal": not cf[int(fatal["v8_action"])]["survives"],
                "q_fatal": [float(x) for x in fatal["q"]],
            }
        )
    return row


LENGTH_BUCKETS = (0, 100, 200, 400, 600, 800, 1200, 10**9)


def length_hazard(
    length: np.ndarray, alive: np.ndarray, death_lengths: Sequence[int]
) -> Dict[str, Dict[str, Any]]:
    """Decisions and deaths per hero-length bucket (deaths per 1000 decisions)."""
    out = {}
    dl = np.asarray(death_lengths, dtype=np.int64)
    for lo, hi in zip(LENGTH_BUCKETS[:-1], LENGTH_BUCKETS[1:]):
        n = int((alive & (length >= lo) & (length < hi)).sum())
        k = int(((dl >= lo) & (dl < hi)).sum())
        key = f"{lo}-{hi if hi < 10**9 else 'inf'}"
        out[key] = {"decisions": n, "deaths": k, "per_1000": (1000.0 * k / n) if n else None}
    return out


def episode_rows(records: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> List[Dict]:
    out = []
    for r, seed in zip(records, seeds):
        if int(r["seed"]) != int(seed):
            raise RuntimeError("record order does not match seeds")
        out.append(
            {
                "world_seed": int(seed),
                "mass_integral": float(r["mass_integral"]),
                "deaths": int(r["deaths"]),
                "survived": int(r["deaths"]) == 0,
                "death_cause": r["probes"]["death_cause"],
                "survival_fraction": float(r["survival_fraction"]),
                "max_mass": float(r["max_mass"]),
                "boost_frame_fraction": r["probes"]["boost_frame_fraction"],
            }
        )
    return out


def reference_rows(arm: str) -> Optional[Dict[str, Any]]:
    """The existing dev-probe row (scripted mix) for ``arm`` (none for the champion)."""
    if arm in B8_ARMS:
        path = PROBE_B8 / f"B8_s{int(arm[-1])}__final__none.jsonl"
    elif arm == "incumbent":
        path = PROBE_INCUMBENT
    else:
        return None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["mix"] == MIX and int(row["round_index"]) == ROUND_INDEX:
            return {"path": str(path), "row": row}
    return None


def compare_reference(arm: str, episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ref = reference_rows(arm)
    if ref is None:
        return {"available": False}
    row = ref["row"]
    if arm in B8_ARMS and row.get("student_sha256") != sha256_file(b8_path(arm)):
        return {"available": True, "path": ref["path"], "match": False, "reason": "sha"}
    mismatch = []
    for ep in episodes:
        key = str(ep["world_seed"])
        if key not in row["mass"]:
            mismatch.append({"world_seed": ep["world_seed"], "reason": "missing"})
            continue
        if (
            float(row["mass"][key]) != ep["mass_integral"]
            or int(row["deaths"][key]) != ep["deaths"]
        ):
            mismatch.append(
                {
                    "world_seed": ep["world_seed"],
                    "probe": [row["mass"][key], row["deaths"][key]],
                    "census": [ep["mass_integral"], ep["deaths"]],
                }
            )
    return {
        "available": True,
        "path": ref["path"],
        "worlds_compared": len(episodes),
        "match": not mismatch,
        "mismatches": mismatch,
    }


# ----------------------------------------------------------------------------- commands
def cmd_run(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(1)
    ctx, seeds, rosters = load_context(args.worlds)
    frames = int(args.frames or HORIZON)
    kinds = [[spec_kind(s) for s in rosters[seed]] for seed in seeds]
    shadow = args.arm in B8_ARMS
    rec = Recorder(args.arm, seeds, kinds, frames, shadow)
    t0 = time.perf_counter()
    with instrumented(rec):
        records = run_eval(
            ctx, seeds, rosters, args.arm, frames, args.v61_backend, args.anchors, rec.observe
        )
    wall = time.perf_counter() - t0
    episodes = episode_rows(records, seeds)
    min_boost = int(rec.meta["min_boost_length"])
    deaths = []
    for e, raw in sorted(rec.deaths.items()):
        row = analyze_death(
            raw, rec.series, e, shadow, min_boost, rec.meta["grid_w"] * rec.meta["grid_h"]
        )
        row.update({"arm": args.arm, "env": e, "world_seed": seeds[e]})
        deaths.append(row)
    for ep in episodes:
        dead = [d for d in deaths if d["world_seed"] == ep["world_seed"]]
        if bool(dead) == ep["survived"]:
            rec.checks["episode_death_mismatch"] += 1
        if dead and dead[0]["cause"] != ep["death_cause"]:
            rec.checks["episode_cause_mismatch"] += 1
    s = rec.series
    alive = s["action"] >= 0
    frame_stats: Dict[str, Any] = {
        "decisions": int(alive.sum()),
        "enclosed_decision_frac": float(((s["best_area"] < s["need"]) & alive).sum() / alive.sum()),
        "no_spacious_decision_frac": float(((s["v8_outcome"] == 2) & alive).sum() / alive.sum()),
        "boost_exec_frac": float((s["boost_exec"] & alive).sum() / alive.sum()),
        "length_hazard": length_hazard(s["length"], alive, [d["length"] for d in deaths]),
    }
    if shadow:
        differs = (s["action"] != s["v8_action"]) & alive
        p = float(differs.sum() / alive.sum())
        frame_stats.update(
            {
                "v8_differs_decision_frac": p,
                "v8_differs_within_k_independent_baseline": {
                    str(k): 1.0 - (1.0 - p) ** k for k in range(1, K_MAX + 1)
                },
                "v8_differs_frac_by_outcome": {
                    OUTCOME_LABEL[o]: float(
                        (differs & (s["v8_outcome"] == o)).sum()
                        / max(1, (alive & (s["v8_outcome"] == o)).sum())
                    )
                    for o in (0, 1, 2)
                },
                "v8_outcome_frac": {
                    OUTCOME_LABEL[o]: float((alive & (s["v8_outcome"] == o)).sum() / alive.sum())
                    for o in (0, 1, 2)
                },
            }
        )
    out = {
        "schema": "b8-death-census/v1",
        "authority": "development diagnostic only (not a screen, not gate evidence)",
        "arm": args.arm,
        "checkpoint": (
            {"path": str(b8_path(args.arm)), "sha256": sha256_file(b8_path(args.arm))}
            if shadow
            else None
        ),
        "mix": MIX,
        "round_index": ROUND_INDEX,
        "frames": frames,
        "seeds": seeds,
        "rosters": {str(seed): kinds[i] for i, seed in enumerate(seeds)},
        "engine": {
            "sim_engine": "grid",
            "vector61_forward": "rowwise",
            "v61_backend": args.v61_backend,
            "anchors": args.anchors,
            "ego2s_forward": "rowwise" if shadow else None,
            "ego2s_mask": "advisory" if shadow else None,
            "veto": None if shadow else "v8(lambda=8.0)",
            "shadow_v8": (
                "v8_nb.v8_decide(lambda=8.0, features=True) on the student's Q" if shadow else None
            ),
        },
        "git": _git(),
        "wall_seconds": wall,
        "instrument_seconds": dict(rec.seconds),
        "meta": rec.meta,
        "checks": dict(rec.checks),
        "reference_match": compare_reference(args.arm, episodes) if frames == HORIZON else None,
        "episodes": episodes,
        "frame_stats": frame_stats,
        "deaths": deaths,
    }
    out_dir = Path(args.out) / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.arm}.json"
    path.write_text(json.dumps(_jsonable(out), indent=1, sort_keys=True))
    np.savez_compressed(out_dir / f"{args.arm}_series.npz", seeds=np.asarray(seeds), **s)
    surv = sum(ep["survived"] for ep in episodes) / len(episodes)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "wall": round(wall, 1),
                "survival": surv,
                "mass": float(np.mean([ep["mass_integral"] for ep in episodes])),
                "deaths": len(deaths),
                "checks": dict(rec.checks),
                "reference_match": (out["reference_match"] or {}).get("match"),
                "out": str(path),
            }
        )
    )
    return 0


def cmd_identity(args: argparse.Namespace) -> int:
    """python/python vs numba/fast: whole records + per-frame world digests, 2 worlds."""
    import torch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env import eval_engine as ee

    torch.set_num_threads(1)
    ctx, seeds, rosters = load_context(args.worlds)
    frames = int(args.frames or HORIZON)
    result: Dict[str, Any] = {
        "schema": "b8-death-census-identity/v1",
        "worlds": seeds,
        "frames": frames,
        "git": _git(),
        "arms": {},
    }
    ok = True
    for arm in args.arms.split(","):
        runs = {}
        for backend, anchors in (("python", "python"), ("numba", "fast")):
            sim_class = ee._terminal_hero_sim_class("grid")
            recorder = gid.DigestRecorder()
            original = recorder.install(sim_class)
            t0 = time.perf_counter()
            try:
                recs = run_eval(ctx, seeds, rosters, arm, frames, backend, anchors)
            finally:
                sim_class.step_with_policy = original
            runs[f"{backend}/{anchors}"] = {
                "wall": time.perf_counter() - t0,
                "records": [gid.normalize(gid.strip_timing(r)) for r in recs],
                "digests": recorder.per_env,
            }
        a, b = runs["python/python"], runs["numba/fast"]
        first_diff = None
        for e in range(len(seeds)):
            for i, (x, y) in enumerate(zip(a["digests"][e], b["digests"][e])):
                if x != y:
                    first_diff = {"env": e, "frame_index": i}
                    break
            if first_diff:
                break
        same = a["records"] == b["records"] and a["digests"] == b["digests"]
        ok = ok and same
        result["arms"][arm] = {
            "records_identical": a["records"] == b["records"],
            "digests_identical": a["digests"] == b["digests"],
            "frame_digests_compared": sum(len(d) for d in a["digests"]),
            "first_digest_diff": first_diff,
            "wall_python_python": a["wall"],
            "wall_numba_fast": b["wall"],
            "mass": [r["mass_integral"] for r in a["records"]],
            "deaths": [r["deaths"] for r in a["records"]],
        }
        print(json.dumps({arm: {k: v for k, v in result["arms"][arm].items()}}))
    result["identical"] = ok
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    return 0 if ok else 1


def _frac(n: int, d: int) -> Optional[float]:
    return None if d == 0 else n / d


def onset_ok(d: Mapping[str, Any]) -> bool:
    return bool(d["enclosure"] and d["trap_onset"])


def summarize_deaths(deaths: Sequence[Mapping[str, Any]], shadow: bool) -> Dict[str, Any]:
    n = len(deaths)
    out: Dict[str, Any] = {
        "deaths": n,
        "by_cause": {c: sum(d["cause"] == c for d in deaths) for c in CAUSES},
        "boost_involved_frac": _frac(sum(d["boost_involved"] for d in deaths), n),
        "boost_exec_fatal_frac": _frac(sum(d["boost_exec_fatal"] for d in deaths), n),
        "boost_exec_last20_frac": _frac(sum(d["boost_exec_last20"] for d in deaths), n),
        "opponent_head_involved_frac": _frac(sum(d["opponent_head_involved"] for d in deaths), n),
        "opponent_head_components": {
            k: _frac(sum(d["opponent_head_components"][k] for d in deaths), n)
            for k in ("head_on", "cut_off_neck_hit", "taken_head_risky")
        },
        # broader, descriptive: an opponent head next to the hero at the fatal decision,
        # and enemy-body hits within 8 cells of the partner's head
        "opp_head_within_2_at_fatal_frac": _frac(
            sum(
                d["nearest_opp_head_dist"] is not None and d["nearest_opp_head_dist"] <= 2
                for d in deaths
            ),
            n,
        ),
        "enemy_body_hit_offset_le8_frac": _frac(
            sum(
                d["cause"] == "enemy_body"
                and bool((d["killer"] or {}).get("hit_offsets"))
                and min(d["killer"]["hit_offsets"]) <= 8
                for d in deaths
            ),
            n,
        ),
        "enclosure_frac": _frac(sum(d["enclosure"] for d in deaths), n),
        "enclosed_at_fatal_frac": _frac(sum(d["enclosed_fatal"] for d in deaths), n),
        "trap_run_hist": {
            "0": sum(d["trap_run"] == 0 for d in deaths),
            "1-2": sum(1 <= d["trap_run"] <= 2 for d in deaths),
            "3-9": sum(3 <= d["trap_run"] <= 9 for d in deaths),
            "10-29": sum(10 <= d["trap_run"] <= 29 for d in deaths),
            ">=30": sum(d["trap_run"] >= 30 for d in deaths),
        },
        "trap_run_median": float(np.median([d["trap_run"] for d in deaths])) if n else None,
        "trap_onset_kind": dict(Counter(d["trap_onset"]["kind"] for d in deaths if onset_ok(d))),
        "trap_onset_available": sum(1 for d in deaths if onset_ok(d)),
        "trap_onset_only_boost_escape": sum(
            bool(d["trap_onset"]["only_boost_escape"]) for d in deaths if onset_ok(d)
        ),
        "forced_frac": _frac(sum(d["forced"] for d in deaths), n),
        "survivor_existed_frac": _frac(sum(bool(d["cf_survivors"]) for d in deaths), n),
        "spacious_survivor_existed_frac": _frac(
            sum(bool(d["cf_survivor_spacious"]) for d in deaths), n
        ),
        "taken_adv_safe_frac": _frac(sum(d["taken_adv_safe"] for d in deaths), n),
        "no_adv_safe_action_frac": _frac(sum(not d["adv_any"] for d in deaths), n),
        "killer_type": dict(Counter((d["killer"] or {}).get("type") for d in deaths)),
        "killer_kind": dict(Counter((d["killer"] or {}).get("kind", "-") for d in deaths)),
        "killer_boost_move_frac": _frac(
            sum(bool((d["killer"] or {}).get("boost_move")) for d in deaths), n
        ),
        "length_median": float(np.median([d["length"] for d in deaths])) if n else None,
        "length_quartiles": (
            [float(x) for x in np.percentile([d["length"] for d in deaths], [25, 50, 75])]
            if n
            else None
        ),
        "fatal_frame_median": float(np.median([d["frame"] for d in deaths])) if n else None,
        "fatal_frame_quartiles": (
            [float(x) for x in np.percentile([d["frame"] for d in deaths], [25, 50, 75])]
            if n
            else None
        ),
        "early_deaths_frame_lt_1000": sum(d["frame"] < 1000 for d in deaths),
        "cf_actual_reproduced_frac": _frac(sum(d["cf_actual_reproduced"] for d in deaths), n),
    }
    if shadow:
        out.update(
            {
                "v8_differs_fatal_frac": _frac(sum(d["v8_differs_fatal"] for d in deaths), n),
                "v8_differs_within_k_frac": {
                    str(k): _frac(sum(d["v8_differs_within_k"][str(k)] for d in deaths), n)
                    for k in range(1, K_MAX + 1)
                },
                "v8_action_also_fatal_frac": _frac(
                    sum(d["v8_action_also_fatal"] for d in deaths), n
                ),
                "v8_action_survives_frac": _frac(sum(d["v8_action_survives"] for d in deaths), n),
                "v8_action_also_fatal_given_differs_frac": _frac(
                    sum(d["v8_action_also_fatal"] for d in deaths if d["v8_differs_fatal"]),
                    sum(d["v8_differs_fatal"] for d in deaths),
                ),
                "v8_outcome_fatal": dict(Counter(d["v8_outcome_fatal"] for d in deaths)),
                "v8_differs_in_trap_run_frac": _frac(
                    sum(d["v8_differs_any_in_trap_run"] for d in deaths if d["enclosure"]),
                    sum(d["enclosure"] for d in deaths),
                ),
                "trap_onset_v8_escapes_frac": _frac(
                    sum(bool(d["trap_onset"]["v8_escapes"]) for d in deaths if onset_ok(d)),
                    sum(1 for d in deaths if onset_ok(d)),
                ),
            }
        )
    return out


LARGE = 800


def large_length(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Share of decisions and of deaths at hero length >= LARGE."""
    dec = big = 0
    for r in group:
        for key, v in r["frame_stats"]["length_hazard"].items():
            dec += v["decisions"]
            if int(key.split("-")[0]) >= LARGE:
                big += v["decisions"]
    deaths = [d for r in group for d in r["deaths"]]
    eps = [e for r in group for e in r["episodes"]]
    return {
        "threshold": LARGE,
        "decision_share": big / dec if dec else None,
        "deaths": len(deaths),
        "deaths_at_large": sum(d["length"] >= LARGE for d in deaths),
        "episodes": len(eps),
        "episodes_reaching_large": sum(e["max_mass"] >= LARGE for e in eps),
        "deaths_below_large_per_episode": sum(d["length"] < LARGE for d in deaths) / len(eps),
        "deaths_at_large_per_episode_reaching": (
            sum(d["length"] >= LARGE for d in deaths)
            / max(1, sum(e["max_mass"] >= LARGE for e in eps))
        ),
        "max_mass_median": float(np.median([e["max_mass"] for e in eps])),
        "death_hero_fill_median": (
            float(np.median([d["hero_fill_frac"] for d in deaths])) if deaths else None
        ),
        "death_occupied_median": (
            float(np.median([d["occupied_frac"] for d in deaths])) if deaths else None
        ),
    }


def enclosure_episodes(root: Path, arms: Sequence[str]) -> Dict[str, Any]:
    """Maximal enclosed runs (from the saved series): how many end in death vs reopen."""
    runs_total = ended_death = reopened = censored = 0
    lengths: List[int] = []
    for arm in arms:
        z = np.load(root / "runs" / f"{arm}_series.npz")
        alive = z["action"] >= 0
        enc = (z["best_area"] < z["need"]) & alive
        for e in range(enc.shape[0]):
            row, al = enc[e], alive[e]
            t = 0
            H = len(row)
            while t < H:
                if not row[t]:
                    t += 1
                    continue
                start = t
                while t < H and row[t]:
                    t += 1
                runs_total += 1
                lengths.append(t - start)
                if t >= H:
                    censored += 1
                elif not al[t]:
                    ended_death += 1
                else:
                    reopened += 1
    return {
        "enclosed_runs": runs_total,
        "ended_in_death": ended_death,
        "reopened": reopened,
        "censored_at_horizon": censored,
        "run_length_median": float(np.median(lengths)) if lengths else None,
    }


def cmd_summarize(args: argparse.Namespace) -> int:
    root = Path(args.out)
    runs = {}
    for arm in ARMS:
        path = root / "runs" / f"{arm}.json"
        if path.exists():
            runs[arm] = json.loads(path.read_text())
    if not runs:
        raise SystemExit("no runs")
    b8 = [runs[a] for a in B8_ARMS if a in runs]
    b8_deaths = [d for r in b8 for d in r["deaths"]]
    summary: Dict[str, Any] = {
        "schema": "b8-death-census-summary/v1",
        "authority": "development diagnostic only (not a screen, not gate evidence)",
        "definitions": {
            "enclosure": f"trap_run >= {ENCLOSURE_MIN}: consecutive decisions ending at the "
            "fatal one where every action's v8 tail-aware landing area < v2 need",
            "boost_involved": f"boost executed on the fatal decision or the {BOOST_WINDOW - 1} "
            "before it",
            "opponent_head_involved": "head_on, or enemy-body hit at offset <= 2 of the "
            "partner's post-move body (cut off at the neck), or the taken action was v8 "
            "head-risky",
            "v8_differs_within_k": "shadow v8 action != hero action at any of the last k "
            "decisions (k=1: fatal decision only)",
            "v8_action_also_fatal": "v8's shadow action at the fatal decision dies in the "
            "one-frame counterfactual (opponents' actual moves); includes agreements",
            "forced": "no hero action survives the fatal frame (one-frame counterfactual)",
        },
        "worlds_per_seed": len(b8[0]["seeds"]) if b8 else None,
        "frames": next(iter(runs.values()))["frames"],
        "checks": {arm: r["checks"] for arm, r in runs.items()},
        "reference_match": {
            arm: (r.get("reference_match") or {}).get("match") for arm, r in runs.items()
        },
        "arms": {},
    }

    def pooled_hazard(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        keys = list(group[0]["frame_stats"]["length_hazard"])
        out = {}
        for key in keys:
            n = sum(r["frame_stats"]["length_hazard"][key]["decisions"] for r in group)
            k = sum(r["frame_stats"]["length_hazard"][key]["deaths"] for r in group)
            out[key] = {"decisions": n, "deaths": k, "per_1000": (1000.0 * k / n) if n else None}
        return out

    summary["length_hazard"] = {
        "b8_pooled": pooled_hazard(b8) if b8 else None,
        **{arm: runs[arm]["frame_stats"]["length_hazard"] for arm in REF_ARMS if arm in runs},
    }
    for arm, r in runs.items():
        eps = r["episodes"]
        summary["arms"][arm] = {
            "survival": sum(e["survived"] for e in eps) / len(eps),
            "mass": float(np.mean([e["mass_integral"] for e in eps])),
            "frame_stats": r["frame_stats"],
            **summarize_deaths(r["deaths"], arm in B8_ARMS),
        }
    if b8:
        pooled = summarize_deaths(b8_deaths, True)
        eps = [e for r in b8 for e in r["episodes"]]
        pooled["survival"] = sum(e["survived"] for e in eps) / len(eps)
        pooled["mass"] = float(np.mean([e["mass_integral"] for e in eps]))
        p = float(np.mean([r["frame_stats"]["v8_differs_decision_frac"] for r in b8]))
        pooled["v8_differs_decision_frac_mean"] = p
        pooled["v8_differs_within_k_independent_baseline"] = {
            str(k): 1.0 - (1.0 - p) ** k for k in range(1, K_MAX + 1)
        }
        summary["b8_pooled"] = pooled
        # paired by world vs the references
        paired = {}
        for ref in REF_ARMS:
            if ref not in runs:
                continue
            ref_surv = {e["world_seed"]: e["survived"] for e in runs[ref]["episodes"]}
            ref_cause = {e["world_seed"]: e["death_cause"] for e in runs[ref]["episodes"]}
            rows = Counter()
            causes_ref_survived = Counter()
            for d in b8_deaths:
                rows["b8_died_ref_survived" if ref_surv[d["world_seed"]] else "both_died"] += 1
                if ref_surv[d["world_seed"]]:
                    causes_ref_survived[d["cause"]] += 1
            survived_b8 = [e for r in b8 for e in r["episodes"] if e["survived"]]
            rows["b8_survived_ref_died"] = sum(not ref_surv[e["world_seed"]] for e in survived_b8)
            rows["both_survived"] = sum(ref_surv[e["world_seed"]] for e in survived_b8)
            paired[ref] = {
                "episode_pairs": dict(rows),
                "b8_death_causes_where_ref_survived": dict(causes_ref_survived),
                "ref_causes": dict(Counter(c for c in ref_cause.values() if c)),
            }
        summary["paired_vs_reference"] = paired
        # per world: how many B8 seeds die there (hard worlds?)
        per_world = Counter(d["world_seed"] for d in b8_deaths)
        summary["b8_deaths_per_world_hist"] = dict(Counter(per_world.values()))
        summary["b8_worlds_with_no_b8_death"] = len(b8[0]["seeds"]) - len(per_world)
    summary["enclosure_episodes"] = {
        "b8_pooled": enclosure_episodes(root, [a for a in B8_ARMS if a in runs]),
        **{arm: enclosure_episodes(root, [arm]) for arm in REF_ARMS if arm in runs},
    }
    summary["large_length"] = {
        "b8_pooled": large_length(b8),
        **{arm: large_length([runs[arm]]) for arm in REF_ARMS if arm in runs},
    }
    first = next(iter(runs.values()))
    summary["rosters"] = dict(Counter(",".join(k) for k in first["rosters"].values()))
    summary["arena_cells"] = first["meta"]["grid_w"] * first["meta"]["grid_h"]
    path = root / "results.json"
    path.write_text(json.dumps(_jsonable(summary), indent=1, sort_keys=True))
    print(json.dumps(_jsonable(summary.get("b8_pooled", {})), indent=1)[:4000])
    print(str(path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--out", default=str(OUT_ROOT))
    r.add_argument("--worlds", type=int, default=WORLDS)
    r.add_argument("--frames", type=int, default=None, help="default H5000 (smoke only)")
    r.add_argument("--v61-backend", choices=("python", "numba"), default="numba")
    r.add_argument("--anchors", choices=("python", "fast"), default="fast")
    i = sub.add_parser("identity")
    i.add_argument("--out", default=str(OUT_ROOT))
    i.add_argument("--worlds", type=int, default=2)
    i.add_argument("--frames", type=int, default=None)
    i.add_argument("--arms", default="b8_s0,incumbent")
    s = sub.add_parser("summarize")
    s.add_argument("--out", default=str(OUT_ROOT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "identity": cmd_identity, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
