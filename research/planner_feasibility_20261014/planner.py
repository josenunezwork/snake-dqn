#!/usr/bin/env python3
"""DEVELOPMENT FEASIBILITY STUDY (research track 3): a learned-value lookahead planner
for the B8 no-veto student. Not a screen, not gate evidence, no candidate, no serving change.

Question: B8 (ego2s-bv8, advisory-mask greedy, no veto) dies by self-enclosure at length >= 800,
with the trap formed a median 31 decisions before death (``b8-death-census-20261014``). A
one-step check (v8) cannot see that. Could a planner that forks the GridBatchSim world and
rolls each candidate action forward d decisions (the student greedy afterwards, opponents on
their own policies) see it, and what would it cost per decision?

Pieces:

* :func:`fork_env`: a ``k``-env copy of ONE env of a (Grid)BatchSim (every per-env array,
  list, RNG and grid sliced; grid views re-established). Lighter than the census's
  whole-batch ``copy.deepcopy`` and the natural shape for a beam: ``k`` candidate copies of
  the same world stepped in lockstep. :func:`fork_runtime` does the same for the shared
  vector61 featurizer runtime (frozen / mixed opponents).
* :class:`RolloutEngine`: lockstep rollouts. Decision 1 plays each candidate's hero action
  (opponents: their own policies on the decision-time world); decisions 2..d play the
  student greedily (``Ego2sSimdPolicy`` advisory rule) and the opponents on their own
  policies (``scripted:`` = the compiled anchors, decision-identical; vector61 checkpoints =
  ``Vector61SimdPolicy`` rowwise on the forked runtime). ``opp_model="student"`` replaces
  every opponent by the student (identity-blind model) and ``food_rng="private"`` re-seeds
  the forked env RNGs (food / respawn draws no longer the real ones).
* ``bench``: (a) fork cost, (b) one depth-d rollout, (c) a full beam of the legal actions
  x depth d, on dev-probe worlds at several frames / hero lengths, plus a k=1 identity check
  (the rollout of the action actually taken reproduces the real world digest d decisions
  later).
* ``probe``: (d) on the B8 census deaths: at the trap-entry decision (one before the
  enclosure onset) and at each "first-sight" decision ``T - d + 1`` (the first decision at
  which a depth-d rollout of the greedy action reaches the death at ``T``), roll every legal
  action out; does some alternative survive d while the taken one dies? Optionally a
  closed-loop local test (``--cl-depths``): from ``T - lead`` the hero is driven by a sparse
  planner (greedy unless its own rollout dies within d; then the highest-Q action whose
  rollout survives d, else the longest-surviving one) in a forked world, to see whether the
  death is avoided or only delayed.

Engine and worlds: exactly the census / dev-probe student arm (``round_index`` 52, the first
32 distillation-namespace worlds, H5000, ``promotion-v2-watch-rect``, grid engine,
``ego2s_forward="rowwise"``, ``ego2s_mask="advisory"``, ``--v61-backend numba --anchors
fast``).

Usage::

  nice -n 10 ./venv/bin/python research/planner_feasibility_20261014/planner.py bench \
      --arm b8_s0 --mix scripted --at 500,1500,2500,3500,4500
  nice -n 10 ./venv/bin/python research/planner_feasibility_20261014/planner.py probe \
      --arm b8_s0 --cl-depths 16,32
  ./venv/bin/python research/planner_feasibility_20261014/planner.py summarize
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
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
OUT_ROOT = ARTIFACTS / "research-tracks-20261014" / "planner"
CENSUS_RUNS = ARTIFACTS / "b8-death-census-20261014" / "runs"
B8_DIR = ARTIFACTS / "redesign-m4b-mac-20261013"

ROUND_INDEX = 52
WORLDS = 32
HORIZON = 5000
LAM = 8.0
B8_ARMS = tuple(f"b8_s{i}" for i in range(5))
MIXES = ("scripted", "frozen", "mixed")
CAUSE_LABEL = {0: None, 1: "wall", 2: "self", 3: "head_on", 4: "enemy_body"}
OPP_MODELS = ("oracle", "student")
FOOD_RNG = ("oracle", "private")


# ----------------------------------------------------------------------------- forking
#: Per-env Python containers of a BatchSim (index ``e``).
_ENV_LISTS = ("_rngs", "_seeds", "food_cells", "corpse_cells", "food_set")
#: GridBatchSim interior views of the padded grids (re-derived after slicing).
_VIEWS = ("_owner", "_slot", "_food")
#: Arrays that are not per-env (copied whole).
_NON_ENV_ARRAYS = ("_snake_index",)
#: Transient attributes of an in-progress step (never forked).
_TRANSIENT = ("_active_env_mask", "_policy_step_in_progress")
_PLAIN = (int, float, bool, str, type(None))


def fork_env(sim: Any, e: int, k: int = 1, *, prepared: Optional[bool] = None) -> Any:
    """``k`` independent copies of env ``e`` of a (Grid)BatchSim, as a ``k``-env sim.

    Every per-env array is sliced (fancy index: a copy), per-env lists / sets / RNGs are
    copied per fork, the grid views are re-established on the sliced pads, and the compiled
    kernels module is shared. Fails closed on an attribute it does not know. Each fork env
    steps exactly like env ``e`` of ``sim`` (checked by ``tests/test_planner_feasibility.py``
    and by the ``bench`` / ``probe`` identity counters).

    ``prepared``: ``None`` copies ``_policy_phase_prepared`` as is (a world captured inside
    ``step`` on the prepared path); ``True`` marks the fork prepared (a world captured inside
    a ``step_with_policy`` selector, where the flag is not yet set).
    """
    E, S, k, e = int(sim.E), int(sim.S), int(k), int(e)
    if not 0 <= e < E or k < 1:
        raise ValueError(f"bad fork (e={e}, k={k}, E={E})")
    idx = np.full(k, e, dtype=np.int64)
    d: Dict[str, Any] = {}
    for name, v in sim.__dict__.items():
        if name in _VIEWS or name in _TRANSIENT:
            continue
        if name == "E":
            d[name] = k
        elif name == "cfg":
            d[name] = dataclasses.replace(v, num_envs=k)
        elif name == "_env_index":
            d[name] = np.arange(k)[:, None]
        elif name == "_kernels":
            d[name] = v
        elif name == "_last_kill_victim_len":
            arr = np.empty((k, S), dtype=object)
            for i in range(k):
                for s in range(S):
                    arr[i, s] = list(v[e, s])
            d[name] = arr
        elif name == "_rngs":
            d[name] = [copy.deepcopy(v[e]) for _ in range(k)]
        elif name == "_seeds":
            d[name] = [v[e]] * k
        elif name == "food_cells":
            d[name] = [list(v[e]) for _ in range(k)]
        elif name in ("corpse_cells", "food_set"):
            d[name] = [set(v[e]) for _ in range(k)]
        elif name == "_det":
            d[name] = (
                None
                if v is None
                else {
                    kk: (
                        vv[idx]
                        if isinstance(vv, np.ndarray) and vv.shape[:1] == (E,)
                        else copy.deepcopy(vv)
                    )
                    for kk, vv in v.items()
                }
            )
        elif isinstance(v, np.ndarray):
            if name in _NON_ENV_ARRAYS:
                d[name] = v.copy()
            elif v.ndim and v.shape[0] == E:
                d[name] = v[idx]
            else:
                raise TypeError(f"fork_env: array {name!r} {v.shape} is not per-env")
        elif isinstance(v, _PLAIN):
            d[name] = v
        else:
            raise TypeError(f"fork_env: unclassified attribute {name!r} ({type(v).__name__})")
    # A hooked eval class (see ``hooked``) names its un-hooked base: forks never re-enter hooks.
    cls = getattr(type(sim), "_planner_base", type(sim))
    new = object.__new__(cls)
    new.__dict__.update(d)
    if prepared is True:
        new._policy_phase_prepared = True
    elif prepared is False:
        new.__dict__.pop("_policy_phase_prepared", None)
    if d.get("_grids_ready") and "_owner_pad" in d:
        P, H, W = new._grid_pad, new.grid_h, new.grid_w
        new._owner = new._owner_pad[:, P : P + H, P : P + W]
        new._slot = new._slot_pad[:, P : P + H, P : P + W]
        new._food = new._food_pad[:, P : P + H, P : P + W]
    return new


_RUNTIME_ARRAYS = (
    "controlled",
    "_carry_state",
    "_carry_mask",
    "_carry_valid",
    "_alive_post",
    "sel_state",
    "sel_mask",
    "sel_ready",
    "sel_fresh",
)


def fork_runtime(rt: Any, e: int, k: int = 1) -> Any:
    """``k`` copies of env ``e``'s rows of a bound :class:`Vector61Runtime` (or ``None``)."""
    if rt is None:
        return None
    from src.simd_env.vector61_featurizer import Vector61Memory

    idx = np.full(int(k), int(e), dtype=np.int64)
    new = copy.copy(rt)
    new.trace = None
    # Own featurizer objects: the numba featurizer caches food arrays keyed by id(sim).
    new.featurizer = copy.copy(rt.featurizer)
    new.free_space_featurizer = (
        new.featurizer
        if rt.free_space_featurizer is rt.featurizer
        else copy.copy(rt.free_space_featurizer)
    )
    S = rt.memory.prev_dist.shape[1]
    mem = Vector61Memory(int(k), S)
    mem.prev_dist = rt.memory.prev_dist[idx]
    mem.prev_id = rt.memory.prev_id[idx]
    new.memory = mem
    for name in _RUNTIME_ARRAYS:
        setattr(new, name, np.asarray(getattr(rt, name))[idx])
    return new


def reseed_private(w: Any, salt: str) -> None:
    """Replace every env RNG state of fork ``w`` by a private one (hash of ``salt``, env)."""
    for i, r in enumerate(w._rngs):
        h = hashlib.sha256(f"planner-private|{salt}|{i}".encode()).digest()
        r._rng.seed(int.from_bytes(h[:8], "little"))


# ----------------------------------------------------------------------------- rollouts
@dataclasses.dataclass
class RolloutResult:
    """One candidate's rollout (``survived`` = decisions survived, ``depth`` if all)."""

    action: int
    survived: int
    died: bool
    cause: Optional[str]
    length_start: int
    length_end: int
    mass_sum: int
    checkpoints: Dict[int, Dict[str, Any]]
    final_world: Any = None

    def survives(self, d: int) -> bool:
        return self.survived >= int(d)

    def as_json(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "survived": self.survived,
            "died": self.died,
            "cause": self.cause,
            "length_start": self.length_start,
            "length_end": self.length_end,
            "mass_sum": self.mass_sum,
            "checkpoints": {str(k): v for k, v in self.checkpoints.items()},
        }


class RolloutEngine:
    """Lockstep rollouts of candidate hero actions on forked worlds (see module doc)."""

    def __init__(
        self,
        student_path: str,
        forward: str = "rowwise",
        opp_model: str = "oracle",
        food_rng: str = "oracle",
        opp_forward: str = "rowwise",
    ) -> None:
        from src.model.ego2s_network import load_ego2s_checkpoint

        if opp_model not in OPP_MODELS or food_rng not in FOOD_RNG:
            raise ValueError("bad opp_model / food_rng")
        self.net = load_ego2s_checkpoint(student_path, "cpu")
        self.forward = forward
        self.opp_model = opp_model
        self.food_rng = food_rng
        self.opp_forward = opp_forward
        self._v61: Dict[str, Any] = {}
        self._anchors: Dict[Any, Any] = {}
        self.sec: Counter = Counter()
        self.count: Counter = Counter()

    # -- policies --------------------------------------------------------------------
    def hero_q(self, w: Any, rows: np.ndarray, forward: Optional[str] = None) -> np.ndarray:
        """Student Q for ``(N, 2)`` rows: ``ego2s_policy.student_q``, timed in two parts."""
        from src.simd_env.ego2s_policy import ego_config_for, forward_q
        from src.simd_env.ego_raster import build_ego_raster

        t0 = time.perf_counter()
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        obs = build_ego_raster(w, ego_config_for(self.net), backend="numba", rows=rows)
        t1 = time.perf_counter()
        q = forward_q(self.net, obs, forward or self.forward).numpy()
        t2 = time.perf_counter()
        self.sec["hero_featurize"] += t1 - t0
        self.sec["hero_forward"] += t2 - t1
        self.sec["hero"] += t2 - t0
        self.count["hero_rows"] += len(rows)
        self.count["hero_calls"] += 1
        return q

    def greedy(self, w: Any, rows: np.ndarray, forward: Optional[str] = None) -> np.ndarray:
        """The advisory-mask greedy student action (``Ego2sSimdPolicy`` advisory rule)."""
        q = self.hero_q(w, rows, forward)
        m = w.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
        m[~m.any(axis=1), :3] = True
        return np.where(m, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int64)

    def _v61_template(self, path: str) -> Any:
        if path not in self._v61:
            from src.simd_env.vector61_policy import Vector61SimdPolicy, checkpoint_is_vector61

            if not checkpoint_is_vector61(path):
                raise ValueError(f"opponent {path} is not a vector61 checkpoint")
            self._v61[path] = Vector61SimdPolicy(path, None, forward=self.opp_forward)
        return self._v61[path]

    def _anchor(self, kind: str, world_seed: int, k: int) -> Any:
        key = (kind, int(world_seed), int(k))
        if key not in self._anchors:
            from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

            self._anchors[key] = FastProfileAnchorSimdPolicy(kind, [int(world_seed)] * int(k))
        return self._anchors[key]

    def opp_actions(
        self,
        w: Any,
        rt: Any,
        roster: Sequence[Sequence[Any]],
        world_seed: int,
        actions: np.ndarray,
        envs: np.ndarray,
    ) -> None:
        """Fill ``actions[envs, 1:]`` for living opponents (in place)."""
        t0 = time.perf_counter()
        alive = w.alive
        groups: Dict[Any, List] = defaultdict(list)
        for j, spec in enumerate(roster, start=1):
            key = ("student",) if self.opp_model == "student" else tuple(spec)
            for i in envs:
                if alive[i, j]:
                    groups[key].append((int(i), j))
        masks = None
        for key, rows in groups.items():
            rows_a = np.asarray(rows, dtype=np.int64)
            if key[0] == "student":  # identity-blind model: opp_forward (not bit-pinned)
                actions[rows_a[:, 0], rows_a[:, 1]] = self.greedy(w, rows_a, self.opp_forward)
                continue
            if masks is None:
                masks = w.get_resolved_action_mask()
            if key[0] == "scripted":
                pol = self._anchor(str(key[1]), world_seed, int(w.E))
            elif key[0] == "checkpoint":
                if rt is None:
                    raise RuntimeError("vector61 opponents need the forked runtime")
                pol = copy.copy(self._v61_template(str(key[1])))
                pol.runtime = rt
            else:
                raise ValueError(f"unsupported opponent spec {key}")
            sel = masks[rows_a[:, 0], rows_a[:, 1]]
            actions[rows_a[:, 0], rows_a[:, 1]] = pol.actions(sel, w, rows_a)
        self.sec["opp"] += time.perf_counter() - t0

    def leaf(self, w: Any, envs: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Census enclosure test on the post-step world for hero rows ``envs``."""
        from src.simd_env.v8_nb import v8_decide

        t0 = time.perf_counter()
        out: Dict[int, Dict[str, Any]] = {}
        if len(envs):
            rows = np.stack([envs, np.zeros_like(envs)], axis=1).astype(np.int64)
            adv = w.get_advisory_action_mask()[envs, 0].astype(bool)
            res = v8_decide(w, rows, np.zeros((len(envs), 6), np.float32), adv, LAM, features=True)
            for n, i in enumerate(envs):
                best = int(np.where(adv[n], res.area[n], 0).max())
                need = int(res.need[n])
                out[int(i)] = {
                    "best_safe_area": best,
                    "need": need,
                    "enclosed": best < need,
                    "length": int(w.length[i, 0]),
                }
        self.sec["leaf"] += time.perf_counter() - t0
        return out

    # -- one lockstep batch ----------------------------------------------------------
    def rollout(
        self,
        sim: Any,
        e: int,
        first: Sequence[int],
        depth: int,
        roster: Sequence[Sequence[Any]],
        world_seed: int,
        *,
        runtime: Any = None,
        opp_first: Optional[np.ndarray] = None,
        checkpoints: Sequence[int] = (),
        prepared: Optional[bool] = None,
        keep_world: bool = False,
        salt: str = "",
    ) -> List[RolloutResult]:
        """Roll ``len(first)`` copies of env ``e`` (a decision-time, prepared world) out.

        Decision 1 plays ``first[i]`` in copy ``i`` (opponents: ``opp_first`` if given, else
        their policies on the decision-time world); decisions 2..``depth`` play the greedy
        student and the opponents' policies. A copy whose hero dies is frozen
        (``active_env_mask``). ``checkpoints``: decision counts at which the census
        enclosure test runs on the surviving copies (post-step world, as the census onset
        counterfactual does).
        """
        k, S = len(first), int(sim.S)
        t0 = time.perf_counter()
        w = fork_env(sim, e, k, prepared=prepared)
        rt = fork_runtime(runtime, e, k)
        if self.food_rng == "private":
            reseed_private(w, f"{world_seed}|{int(sim.frame[e])}|{salt}")
        self.sec["fork"] += time.perf_counter() - t0
        self.count["forks"] += 1
        envs = np.arange(k)
        length_start = int(sim.length[e, 0])
        acts = np.ones((k, S), dtype=np.int64)
        if opp_first is not None and self.opp_model == "oracle":
            acts[:, 1:] = np.asarray(opp_first, dtype=np.int64)[None, :]
        else:
            self.opp_actions(w, rt, roster, world_seed, acts, envs)
        acts[:, 0] = np.asarray(first, dtype=np.int64)
        self._step(w, rt, acts, None)
        dead = np.zeros(k, bool)
        survived = np.zeros(k, np.int64)
        cause = np.zeros(k, np.int64)
        mass = np.zeros(k, np.int64)
        cps: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        wanted = set(int(c) for c in checkpoints)

        def after(step: int, live: np.ndarray) -> None:
            alive0 = w.alive[:, 0]
            c = w.get_death_cause()[:, 0]
            died_now = live & ~alive0
            cause[died_now] = c[died_now]
            dead[died_now] = True
            ok = live & alive0
            survived[ok] = step
            mass[ok] += w.length[ok, 0]
            if step in wanted:
                for i, info in self.leaf(w, np.flatnonzero(ok)).items():
                    cps[i][step] = info

        after(1, np.ones(k, bool))
        for step in range(2, int(depth) + 1):
            live = ~dead
            if not live.any():
                break
            live_envs = np.flatnonzero(live)

            def choose(p: Any, live_envs: np.ndarray = live_envs) -> np.ndarray:
                t1 = time.perf_counter()
                a = np.ones((k, S), dtype=np.int64)
                if rt is not None:
                    rt.prepare(p)
                    self.sec["v61_prepare"] += time.perf_counter() - t1
                rows = np.stack([live_envs, np.zeros_like(live_envs)], axis=1)
                a[live_envs, 0] = self.greedy(p, rows)
                self.opp_actions(p, rt, roster, world_seed, a, live_envs)
                self.sec["select"] += time.perf_counter() - t1
                return a

            self._step(w, rt, None, choose, active=live)
            after(step, live)
        out = []
        for i in range(k):
            out.append(
                RolloutResult(
                    action=int(first[i]),
                    survived=int(survived[i]),
                    died=bool(dead[i]),
                    cause=CAUSE_LABEL[int(cause[i])] if dead[i] else None,
                    length_start=length_start,
                    length_end=int(w.length[i, 0]),
                    mass_sum=int(mass[i]),
                    checkpoints=dict(cps.get(i, {})),
                    final_world=w if keep_world else None,
                )
            )
        return out

    def _step(
        self,
        w: Any,
        rt: Any,
        acts: Optional[np.ndarray],
        choose: Optional[Callable],
        active: Optional[np.ndarray] = None,
    ) -> None:
        t0 = time.perf_counter()
        if choose is None:
            w.step(acts)
        else:
            w.step_with_policy(choose, active_env_mask=active)
        self.sec["step_total"] += time.perf_counter() - t0
        self.count["steps"] += 1
        self.count["env_steps"] += int(w.E if active is None else np.sum(active))
        if rt is not None:
            t1 = time.perf_counter()
            rt.observe_step(w)
            self.sec["v61_observe"] += time.perf_counter() - t1

    def reset_timers(self) -> None:
        self.sec.clear()
        self.count.clear()

    def timers(self) -> Dict[str, Any]:
        sec = dict(self.sec)
        # step_total includes the selector (hero / opp / v61 prepare inside choose)
        sec["sim_only"] = max(0.0, sec.get("step_total", 0.0) - sec.get("select", 0.0))
        return {"seconds": sec, "counts": dict(self.count)}


# ----------------------------------------------------------------------------- planner rule
def planner_choice(
    engine: RolloutEngine,
    p: Any,
    e: int,
    depth: int,
    roster: Sequence[Sequence[Any]],
    world_seed: int,
    runtime: Any = None,
    prepared: Optional[bool] = None,
    salt: str = "",
) -> Dict[str, Any]:
    """The sparse planner rule at a prepared decision world (hero row ``(e, 0)``).

    Greedy action ``g``; roll ``g`` out ``depth`` decisions. If it survives: play ``g``.
    Otherwise roll every other legal action out; play the highest-Q one that survives
    ``depth`` (ties: lowest index), else the one surviving longest (ties: highest Q; ``g``
    kept when it is among the longest).
    """
    rows = np.array([[e, 0]], dtype=np.int64)
    q = engine.hero_q(p, rows)[0]
    m = p.get_advisory_action_mask()[e, 0].astype(bool)
    adv = m.copy()
    if not m.any():
        m[:3] = True
    g = int(np.where(m, q.astype(np.float64), -np.inf).argmax())
    r_g = engine.rollout(
        p, e, [g], depth, roster, world_seed, runtime=runtime, prepared=prepared, salt=salt
    )[0]
    if r_g.survives(depth):
        return {"action": g, "greedy": g, "fired": False, "switched": False}
    legal = p.get_legal_action_mask()[e, 0].astype(bool)
    alts = [a for a in range(6) if legal[a] and a != g]
    res = {g: r_g}
    if alts:
        for r in engine.rollout(
            p, e, alts, depth, roster, world_seed, runtime=runtime, prepared=prepared, salt=salt
        ):
            res[r.action] = r
    surv = [a for a in res if res[a].survives(depth)]
    if surv:
        best = max(surv, key=lambda a: (float(q[a]), -a))
        reason = "survivor"
    else:
        longest = max(r.survived for r in res.values())
        pool = [a for a in res if res[a].survived == longest]
        best = g if g in pool else max(pool, key=lambda a: (float(q[a]), -a))
        reason = "longest"
    return {
        "action": int(best),
        "greedy": g,
        "fired": True,
        "switched": int(best) != g,
        "reason": reason,
        "survived": {int(a): int(r.survived) for a, r in res.items()},
        "adv_safe": [int(a) for a in np.flatnonzero(adv)],
    }


# ----------------------------------------------------------------------------- eval driver
def load_context(mix: str, worlds: int = WORLDS):
    """(gid context, the first ``worlds`` dev-probe seeds, their ``mix`` rosters)."""
    from research.redesign_m2_20261008 import dev_probe
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    ctx = gid._context(1)
    seeds = dev_probe.probe_seeds(WORLDS, ROUND_INDEX)
    rows = {
        int(r["world_seed"]): [ctx["lookup"][s["member_sha256"]] for s in r["slots"]]
        for r in ctx["ds"]._design_rows(seeds)
        if r["mix"] == mix
    }
    seeds = seeds[: int(worlds)]
    return ctx, seeds, {s: rows[s] for s in seeds}


def b8_path(arm: str) -> Path:
    return B8_DIR / f"B8_s{int(arm[-1])}" / "final.pth"


@contextmanager
def hooked(
    pre_hook: Callable[[Any, np.ndarray, int, Any], None],
    post_hook: Optional[Callable[[Any, int], None]] = None,
) -> Iterator[Dict[str, Any]]:
    """Patch the eval's grid sim class (decision-time / post-step hooks) and capture the
    vector61 runtime. The hooks only read the live world (forks are copies)."""
    from src.simd_env import eval_engine as ee

    orig_cls_fn = ee._terminal_hero_sim_class
    base = orig_cls_fn("grid")
    base_step = base.step
    state: Dict[str, Any] = {"t": 0, "runtime": None, "hook_seconds": 0.0}

    class HookedGridSim(base):  # type: ignore[misc, valid-type]
        _planner_base = base

        def step(self, actions, active_env_mask=None):  # type: ignore[no-untyped-def]
            if not getattr(self, "_policy_phase_prepared", False):
                return base_step(self, actions, active_env_mask)
            t = state["t"]
            t0 = time.perf_counter()
            pre_hook(self, np.asarray(actions, dtype=np.int64).copy(), t, state["runtime"])
            state["hook_seconds"] += time.perf_counter() - t0
            out = base_step(self, actions, active_env_mask)
            if post_hook is not None:
                t0 = time.perf_counter()
                post_hook(self, t)
                state["hook_seconds"] += time.perf_counter() - t0
            state["t"] += 1
            return out

    orig_rt = ee.Vector61Runtime

    class CapturingRuntime(orig_rt):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            state["runtime"] = self

    ee._terminal_hero_sim_class = lambda engine: (
        HookedGridSim if engine == "grid" else orig_cls_fn(engine)
    )
    ee.Vector61Runtime = CapturingRuntime
    try:
        yield state
    finally:
        ee._terminal_hero_sim_class = orig_cls_fn
        ee.Vector61Runtime = orig_rt


def run_eval(ctx, seeds, rosters, arm: str, mix: str, frames: int) -> List[Dict[str, Any]]:
    from research.redesign_m2_20261008 import ni_spec
    from src.simd_env import eval_engine as ee

    profile = ctx["profile"]
    if frames != profile.scored_horizon:
        profile = dataclasses.replace(profile, scored_horizon=frames)
    if arm not in B8_ARMS:
        raise SystemExit(f"unknown arm {arm}")
    return ee.run_simd_eval(
        ctx["lookup"][ni_spec.FRP3_S12[1]],
        rosters[seeds[0]],
        frames,
        list(seeds),
        profile=profile,
        opponent_specs_by_world=dict(rosters),
        mix_id=mix,
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        vector61_backend="numba",
        scripted_anchors="fast",
        hero_ego2s=str(b8_path(arm)),
        ego2s_forward="rowwise",
        ego2s_mask="advisory",
    )


def _git() -> Dict[str, Any]:
    def run(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _setup_threads() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _median_ms(fn: Callable[[], Any], reps: int) -> Dict[str, float]:
    times = []
    for _ in range(max(1, int(reps))):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return {
        "median_ms": float(statistics.median(times)),
        "min_ms": float(min(times)),
        "n": len(times),
    }


# ----------------------------------------------------------------------------- bench
def cmd_bench(args: argparse.Namespace) -> int:
    """(a) fork cost, (b) one depth-d rollout, (c) a full legal-action beam (see module)."""
    from research.b8_death_census_20261014 import census
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    ctx, seeds, rosters = load_context(args.mix, args.worlds)
    _setup_threads()
    at = sorted(int(x) for x in args.at.split(","))
    depths = [int(x) for x in args.depths.split(",")]
    frames = int(args.frames or (max(at) + max(depths) + 2))
    engines = {
        fw: RolloutEngine(str(b8_path(args.arm)), forward=fw) for fw in ("rowwise", "batched")
    }
    rows: List[Dict[str, Any]] = []
    identity: List[Dict[str, Any]] = []
    want_digest: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    def pick_envs(sim: Any) -> List[int]:
        alive = np.flatnonzero(sim.alive[:, 0])
        if not len(alive):
            return []
        lens = sim.length[alive, 0]
        order = alive[np.argsort(lens, kind="stable")]
        picks = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
        return list(dict.fromkeys(picks))[: args.envs_per_frame]

    def pre(sim: Any, actions: np.ndarray, t: int, rt: Any) -> None:
        if t not in at:
            return
        if rt is not None and not bool(np.asarray(rt.controlled).any()):
            rt = None  # no vector61 rows (scripted mix): nothing to fork or prepare
        E = int(sim.E)
        full = _median_ms(lambda: census.clone_sim(sim), args.reps_full)
        rows.append({"t": t, "op": "clone_full_batch", "E": E, **full})
        for e in pick_envs(sim):
            seed = int(seeds[e])
            roster = rosters[seed]
            L = int(sim.length[e, 0])
            base = {"t": t, "env": e, "world_seed": seed, "hero_length": L, "E": E}
            for k in (1, 3, 6):
                r = _median_ms(lambda k=k: fork_env(sim, e, k), args.reps_fork)
                rows.append({**base, "op": "fork_env", "k": k, **r})
            r = _median_ms(lambda: fork_runtime(rt, e, 6), args.reps_fork) if rt else None
            if r:
                rows.append({**base, "op": "fork_runtime", "k": 6, **r})
            legal = [int(a) for a in np.flatnonzero(sim.get_legal_action_mask()[e, 0])]
            taken = int(actions[e, 0])
            for d in depths:
                # (b) one rollout of the action actually taken (k=1; forward modes agree at 1)
                eng = engines["rowwise"]
                eng.reset_timers()
                t0 = time.perf_counter()
                res = eng.rollout(sim, e, [taken], d, roster, seed, runtime=rt, keep_world=True)
                ms = (time.perf_counter() - t0) * 1e3
                rows.append(
                    {
                        **base,
                        "op": "rollout",
                        "k": 1,
                        "d": d,
                        "forward": "rowwise",
                        "ms": ms,
                        "survived": res[0].survived,
                        **eng.timers(),
                    }
                )
                if res[0].survived == d:
                    want_digest[(e, t + d - 1)].append(
                        {"t": t, "env": e, "d": d, "digest": gid.env_digest(res[0].final_world, 0)}
                    )
                else:
                    want_digest[(e, t + res[0].survived)].append(
                        {"t": t, "env": e, "d": d, "dies_at_step": res[0].survived + 1}
                    )
                # (c) the full beam: every legal action, and the 3 normal-speed actions
                for label, cands in (("legal", legal), ("normal3", [a for a in legal if a < 3])):
                    for fw in ("batched", "rowwise"):
                        eng = engines[fw]
                        times = []
                        for _ in range(args.reps_beam):
                            eng.reset_timers()
                            t0 = time.perf_counter()
                            eng.rollout(sim, e, cands, d, roster, seed, runtime=rt)
                            times.append((time.perf_counter() - t0) * 1e3)
                        rows.append(
                            {
                                **base,
                                "op": "beam",
                                "cands": label,
                                "k": len(cands),
                                "d": d,
                                "forward": fw,
                                "ms": float(statistics.median(times)),
                                **eng.timers(),
                            }
                        )
        print(f"[bench] t={t} done ({len(rows)} rows)", flush=True)

    def post(sim: Any, t: int) -> None:
        for e in range(int(sim.E)):
            for item in want_digest.pop((e, t), []):
                if "digest" in item:
                    ok = gid.env_digest(sim, e) == item["digest"]
                else:
                    ok = bool(not sim.alive[e, 0])
                identity.append({**item, "match": ok})

    t0 = time.perf_counter()
    with hooked(pre, post) as st:
        recs = run_eval(ctx, seeds, rosters, args.arm, args.mix, frames)
    out = {
        "schema": "planner-bench/v1",
        "authority": "development feasibility study only (not a screen, not gate evidence)",
        "arm": args.arm,
        "mix": args.mix,
        "worlds": len(seeds),
        "frames": frames,
        "at": at,
        "depths": depths,
        "rows": rows,
        "identity": identity,
        "identity_ok": sum(1 for i in identity if i["match"]),
        "identity_n": len(identity),
        "hook_seconds": st["hook_seconds"],
        "wall_seconds": time.perf_counter() - t0,
        "mass": {int(r["seed"]): float(r["mass_integral"]) for r in recs},
        "git": _git(),
        "threads": {"torch": 1, "omp": os.environ.get("OMP_NUM_THREADS")},
        "host_load": os.getloadavg(),
    }
    path = Path(args.out) / f"bench_{args.arm}_{args.mix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(out), indent=1))
    print(f"[bench] wrote {path} identity {out['identity_ok']}/{out['identity_n']}", flush=True)
    return 0


# ----------------------------------------------------------------------------- probe
def census_deaths(arm: str) -> List[Dict[str, Any]]:
    return json.loads((CENSUS_RUNS / f"{arm}.json").read_text())["deaths"]


def cmd_probe(args: argparse.Namespace) -> int:
    """(d) quality probe on the census deaths (+ optional closed-loop local test)."""
    ctx, seeds, rosters = load_context("scripted", WORLDS)
    _setup_threads()
    deaths = census_deaths(args.arm)
    onset_depths = [int(x) for x in args.onset_depths.split(",")]
    sight_depths = [int(x) for x in args.sight_depths.split(",")]
    cl_depths = [int(x) for x in args.cl_depths.split(",")] if args.cl_depths else []
    engine = RolloutEngine(
        str(b8_path(args.arm)),
        forward="rowwise",
        opp_model=args.opp_model,
        food_rng=args.food_rng,
        opp_forward=args.opp_forward,
    )
    # The closed-loop world itself always plays the real opponents and food; only the
    # planner's rollouts use ``engine``'s (possibly identity-blind) model.
    truth = engine
    if (args.opp_model, args.food_rng) != ("oracle", "oracle"):
        truth = RolloutEngine(str(b8_path(args.arm)), forward="rowwise")
    targets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for di, dr in enumerate(deaths):
        e, T = int(dr["env"]), int(dr["fatal_t"])
        onset = dr.get("trap_onset")
        if onset is not None and int(onset["t0"]) >= 1:
            targets[int(onset["t0"]) - 1].append(
                {
                    "kind": "onset",
                    "death": di,
                    "env": e,
                    "depth": max(onset_depths),
                    "checkpoints": onset_depths,
                }
            )
        for d in sight_depths:
            tau = T - d + 1
            if tau >= 0:
                targets[tau].append(
                    {"kind": "sight", "death": di, "env": e, "depth": d, "checkpoints": [d]}
                )
        for d in cl_depths:
            tau = max(0, T - int(args.cl_lead))
            targets[tau].append({"kind": "closed_loop", "death": di, "env": e, "depth": d})
        if cl_depths and args.cl_control:
            tau = max(0, T - int(args.cl_lead))
            targets[tau].append({"kind": "closed_loop", "death": di, "env": e, "depth": 0})
    results: List[Dict[str, Any]] = []
    checks: Counter = Counter()

    def pre(sim: Any, actions: np.ndarray, t: int, rt: Any) -> None:
        for tg in targets.get(t, []):
            e = tg["env"]
            dr = deaths[tg["death"]]
            seed = int(seeds[e])
            if seed != int(dr["world_seed"]):
                raise RuntimeError("census env / seed mismatch")
            if not sim.alive[e, 0]:
                checks["target_hero_dead"] += 1
                continue
            t0 = time.perf_counter()
            if tg["kind"] == "closed_loop":
                row = closed_loop(
                    engine, truth, sim, e, t, tg["depth"], rosters[seed], seed, dr, args
                )
            else:
                row = probe_target(engine, sim, actions, e, t, tg, rosters[seed], seed, dr, checks)
            row.update(
                {
                    "kind": tg["kind"],
                    "death": tg["death"],
                    "env": e,
                    "world_seed": seed,
                    "t": t,
                    "seconds": time.perf_counter() - t0,
                }
            )
            results.append(row)
        if t in targets:
            print(f"[probe {args.arm}] t={t} n={len(results)}", flush=True)

    t0 = time.perf_counter()
    with hooked(pre) as st:
        recs = run_eval(ctx, seeds, rosters, args.arm, "scripted", int(args.frames))
    census_mass = {
        int(r["world_seed"]): float(r["mass_integral"])
        for r in json.loads((CENSUS_RUNS / f"{args.arm}.json").read_text())["episodes"]
    }
    mass = {int(r["seed"]): float(r["mass_integral"]) for r in recs}
    if int(args.frames) == HORIZON:
        checks["episodes_match_census"] = sum(
            1 for s in seeds if abs(mass[int(s)] - census_mass[int(s)]) < 1e-9
        )
    out = {
        "schema": "planner-probe/v1",
        "authority": "development feasibility study only (not a screen, not gate evidence)",
        "arm": args.arm,
        "opp_model": args.opp_model,
        "food_rng": args.food_rng,
        "onset_depths": onset_depths,
        "sight_depths": sight_depths,
        "cl_depths": cl_depths,
        "cl_lead": args.cl_lead,
        "cl_post": args.cl_post,
        "deaths": len(deaths),
        "results": results,
        "checks": dict(checks),
        "timers": engine.timers(),
        "hook_seconds": st["hook_seconds"],
        "wall_seconds": time.perf_counter() - t0,
        "git": _git(),
    }
    tag = (
        ""
        if (args.opp_model, args.food_rng) == ("oracle", "oracle")
        else (f"_{args.opp_model}_{args.food_rng}")
    )
    path = Path(args.out) / f"probe_{args.arm}{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(out), indent=1))
    print(f"[probe] wrote {path} checks {dict(checks)}", flush=True)
    return 0


def probe_target(
    engine: RolloutEngine,
    sim: Any,
    actions: np.ndarray,
    e: int,
    t: int,
    tg: Mapping[str, Any],
    roster: Sequence[Sequence[Any]],
    seed: int,
    dr: Mapping[str, Any],
    checks: Counter,
) -> Dict[str, Any]:
    """Roll every legal hero action out ``tg['depth']`` decisions from decision ``t``."""
    T = int(dr["fatal_t"])
    taken = int(actions[e, 0])
    legal = [int(a) for a in np.flatnonzero(sim.get_legal_action_mask()[e, 0])]
    adv = sim.get_advisory_action_mask()[e, 0].astype(bool)
    q = engine.hero_q(sim, np.array([[e, 0]], dtype=np.int64))[0]
    # The opponents' first-decision actions from their own policies must equal the real ones.
    if engine.opp_model == "oracle":
        probe_acts = np.ones((1, int(sim.S)), dtype=np.int64)
        w1 = fork_env(sim, e, 1)
        engine.opp_actions(w1, None, roster, seed, probe_acts, np.array([0]))
        live = sim.alive[e, 1:]
        same = np.array_equal(probe_acts[0, 1:][live], actions[e, 1:][live])
        checks["opp_first_match" if same else "opp_first_mismatch"] += 1
    res = engine.rollout(
        sim,
        e,
        legal,
        int(tg["depth"]),
        roster,
        seed,
        opp_first=actions[e, 1:],
        checkpoints=tg["checkpoints"],
        salt=f"{t}",
    )
    by_a = {r.action: r for r in res}
    # Identity: the taken action's rollout is the real future (oracle model only).
    expect = T - t + 1  # the death happens at rollout decision ``expect``
    r_taken = by_a[taken]
    if engine.opp_model == "oracle" and engine.food_rng == "oracle":
        if expect <= int(tg["depth"]):
            ok = r_taken.died and r_taken.survived == expect - 1 and r_taken.cause == dr["cause"]
        else:
            ok = not r_taken.died
        checks["taken_reproduces" if ok else "taken_diverges"] += 1
    return {
        "fatal_t": T,
        "trap_run": int(dr["trap_run"]),
        "enclosure": bool(dr["enclosure"]),
        "cause": dr["cause"],
        "length": int(sim.length[e, 0]),
        "length_at_death": int(dr["length"]),
        "decisions_to_death": expect,
        "taken": taken,
        "legal": legal,
        "adv_safe": [int(a) for a in np.flatnonzero(adv)],
        "q": [float(x) for x in q],
        "depth": int(tg["depth"]),
        "checkpoints": list(tg["checkpoints"]),
        "rollouts": {str(a): r.as_json() for a, r in by_a.items()},
    }


def closed_loop(
    engine: RolloutEngine,
    truth: RolloutEngine,
    sim: Any,
    e: int,
    t: int,
    depth: int,
    roster: Sequence[Sequence[Any]],
    seed: int,
    dr: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """From decision ``t`` drive the hero of a forked copy of env ``e`` with the sparse
    planner (``depth`` 0: pure greedy, the identity control) until the original death
    decision + ``cl_post`` (or the horizon), opponents on their own policies."""
    T = int(dr["fatal_t"])
    end = min(HORIZON - 1, T + int(args.cl_post))
    w = fork_env(sim, e, 1)
    S = int(sim.S)
    log: Dict[str, Any] = {"fires": 0, "switches": 0, "reasons": Counter(), "fire_ts": []}
    plan_sec = 0.0
    decisions = 0
    death_t: Optional[int] = None
    death_cause: Optional[str] = None
    lengths: List[int] = []

    def decide(p: Any, tt: int, prepared: Optional[bool]) -> int:
        nonlocal plan_sec
        t0 = time.perf_counter()
        if depth <= 0:
            a = int(truth.greedy(p, np.array([[0, 0]], dtype=np.int64))[0])
        else:
            ch = planner_choice(
                engine, p, 0, depth, roster, seed, prepared=prepared, salt=f"cl{tt}"
            )
            a = int(ch["action"])
            if ch["fired"]:
                log["fires"] += 1
                log["fire_ts"].append(tt)
                log["reasons"][ch["reason"]] += 1
                log["switches"] += int(ch["switched"])
        plan_sec += time.perf_counter() - t0
        return a

    # decision t: the world is prepared (captured inside step)
    acts = np.ones((1, S), dtype=np.int64)
    truth.opp_actions(w, None, roster, seed, acts, np.array([0]))
    acts[0, 0] = decide(w, t, None)
    w.step(acts)
    decisions = 1
    tt = t
    while True:
        if not w.alive[0, 0]:
            death_t = tt
            death_cause = CAUSE_LABEL[int(w.get_death_cause()[0, 0])]
            break
        lengths.append(int(w.length[0, 0]))
        if tt >= end:
            break
        tt += 1

        def choose(p: Any, tt: int = tt) -> np.ndarray:
            a = np.ones((1, S), dtype=np.int64)
            a[0, 0] = decide(p, tt, True)
            truth.opp_actions(p, None, roster, seed, a, np.array([0]))
            return a

        w.step_with_policy(choose)
        decisions += 1
    reasons = dict(log["reasons"])
    return {
        "depth": depth,
        "start_t": t,
        "fatal_t": T,
        "end_t": end,
        "trap_run": int(dr["trap_run"]),
        "enclosure": bool(dr["enclosure"]),
        "length_at_death": int(dr["length"]),
        "death_t": death_t,
        "death_cause": death_cause,
        "survived_past_T": death_t is None or death_t > T,
        "extra_decisions": (end if death_t is None else death_t) - T,
        "censored": death_t is None,
        "reproduces_greedy": (depth <= 0 and death_t == T and death_cause == dr["cause"]),
        "decisions": decisions,
        "fires": log["fires"],
        "switches": log["switches"],
        "fire_ts": log["fire_ts"][:50],
        "reasons": reasons,
        "planner_ms_per_decision": 1e3 * plan_sec / max(1, decisions),
        "length_end": lengths[-1] if lengths else None,
        "max_length": max(lengths) if lengths else None,
    }


# ----------------------------------------------------------------------------- summarize
def _frac(n: int, d: int) -> Optional[float]:
    return (n / d) if d else None


def summarize_probe(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    out: Dict[str, Any] = {
        "arm": data["arm"],
        "opp_model": data["opp_model"],
        "food_rng": data["food_rng"],
        "checks": data["checks"],
    }
    res = data["results"]
    # (1) trap-entry decision (one before the enclosure onset), per checkpoint depth
    onset = [r for r in res if r["kind"] == "onset"]
    for d in data["onset_depths"]:
        out[f"onset_d{d}"] = _detect(onset, d)
    for d in data["sight_depths"]:
        out[f"sight_d{d}"] = _detect(
            [r for r in res if r["kind"] == "sight" and r["depth"] == d], d
        )
    for d in [0] + list(data["cl_depths"]):
        cl = [r for r in res if r["kind"] == "closed_loop" and r["depth"] == d]
        if not cl:
            continue
        ext = sorted(r["extra_decisions"] for r in cl)
        out[f"closed_loop_d{d}"] = {
            "n": len(cl),
            "survived_past_T": sum(r["survived_past_T"] for r in cl),
            "censored_alive_at_end": sum(r["censored"] for r in cl),
            "reproduces_greedy": sum(r["reproduces_greedy"] for r in cl) if d == 0 else None,
            "median_extra_decisions": statistics.median(ext) if ext else None,
            "fires_total": sum(r["fires"] for r in cl),
            "switches_total": sum(r["switches"] for r in cl),
            "planner_ms_per_decision_median": statistics.median(
                r["planner_ms_per_decision"] for r in cl
            ),
        }
    return out


def _detect(rows: Sequence[Mapping[str, Any]], d: int) -> Dict[str, Any]:
    """Per death: does the taken action die within ``d`` and some alternative survive ``d``
    (and, stricter, survive ``d`` with an open leaf: not enclosed by the census test)?"""
    n = len(rows)
    taken_dies = alt_surv = alt_open = alt_surv_given_dies = alt_open_given_dies = 0
    alt_open_adv = 0
    by_enc = Counter()
    for r in rows:
        ro = r["rollouts"]
        tk = ro[str(r["taken"])]
        dies = tk["survived"] < d
        taken_dies += dies
        alts = [ro[a] for a in ro if int(a) != r["taken"]]
        surv = [a for a in alts if a["survived"] >= d]
        opened = [
            a
            for a in surv
            if a["checkpoints"].get(str(d)) is not None and not a["checkpoints"][str(d)]["enclosed"]
        ]
        alt_surv += bool(surv)
        alt_open += bool(opened)
        alt_open_adv += bool([a for a in opened if a["action"] in r["adv_safe"]])
        if dies:
            alt_surv_given_dies += bool(surv)
            alt_open_given_dies += bool(opened)
            by_enc[("enclosure" if r["enclosure"] else "immediate", bool(surv))] += 1
    return {
        "n": n,
        "taken_dies_within_d": taken_dies,
        "some_alt_survives_d": alt_surv,
        "some_alt_survives_d_open_leaf": alt_open,
        "some_adv_safe_alt_survives_d_open_leaf": alt_open_adv,
        "given_taken_dies": {
            "n": taken_dies,
            "some_alt_survives_d": alt_surv_given_dies,
            "frac": _frac(alt_surv_given_dies, taken_dies),
            "some_alt_survives_d_open_leaf": alt_open_given_dies,
            "frac_open_leaf": _frac(alt_open_given_dies, taken_dies),
        },
        "by_kind_given_taken_dies": {f"{k[0]}|alt_survives={k[1]}": v for k, v in by_enc.items()},
    }


def summarize_bench(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    rows = data["rows"]
    out: Dict[str, Any] = {
        "arm": data["arm"],
        "mix": data["mix"],
        "identity": f"{data['identity_ok']}/{data['identity_n']}",
    }
    full = [r["median_ms"] for r in rows if r["op"] == "clone_full_batch"]
    out["clone_full_batch_ms_median"] = statistics.median(full) if full else None
    for k in (1, 3, 6):
        v = [r["median_ms"] for r in rows if r["op"] == "fork_env" and r["k"] == k]
        out[f"fork_env_k{k}_ms_median"] = statistics.median(v) if v else None
    v = [r["median_ms"] for r in rows if r["op"] == "fork_runtime"]
    out["fork_runtime_k6_ms_median"] = statistics.median(v) if v else None
    for d in data["depths"]:
        v = [r["ms"] for r in rows if r["op"] == "rollout" and r["d"] == d and r["survived"] == d]
        out[f"rollout_k1_d{d}_ms_median"] = statistics.median(v) if v else None
        for lab in ("legal", "normal3"):
            for fw in ("batched", "rowwise"):
                v = [
                    r
                    for r in rows
                    if r["op"] == "beam"
                    and r["d"] == d
                    and r["cands"] == lab
                    and r["forward"] == fw
                ]
                if v:
                    out[f"beam_{lab}_d{d}_{fw}_ms_median"] = statistics.median(x["ms"] for x in v)
                    out[f"beam_{lab}_d{d}_{fw}_k_median"] = statistics.median(x["k"] for x in v)
    return out


def cmd_summarize(args: argparse.Namespace) -> int:
    root = Path(args.out)
    summary: Dict[str, Any] = {"bench": {}, "probe": {}}
    for p in sorted(root.glob("bench_*.json")):
        summary["bench"][p.stem] = summarize_bench(p)
    for p in sorted(root.glob("probe_*.json")):
        summary["probe"][p.stem] = summarize_probe(p)
    (root / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1))
    print(json.dumps(_jsonable(summary), indent=1))
    return 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--arm", choices=B8_ARMS, default="b8_s0")
    b.add_argument("--mix", choices=MIXES, default="scripted")
    b.add_argument("--worlds", type=int, default=WORLDS)
    b.add_argument("--at", default="500,1500,2500,3500,4500")
    b.add_argument("--depths", default="8,16,32")
    b.add_argument("--frames", type=int, default=None)
    b.add_argument("--envs-per-frame", type=int, default=3)
    b.add_argument("--reps-full", type=int, default=3)
    b.add_argument("--reps-fork", type=int, default=15)
    b.add_argument("--reps-beam", type=int, default=2)
    b.add_argument("--out", default=str(OUT_ROOT))
    p = sub.add_parser("probe")
    p.add_argument("--arm", choices=B8_ARMS, required=True)
    p.add_argument("--onset-depths", default="16,32,64")
    p.add_argument("--sight-depths", default="8,16,32,64")
    p.add_argument("--cl-depths", default="")
    p.add_argument("--cl-lead", type=int, default=64)
    p.add_argument("--cl-post", type=int, default=200)
    p.add_argument("--cl-control", action="store_true")
    p.add_argument("--opp-model", choices=OPP_MODELS, default="oracle")
    p.add_argument("--food-rng", choices=FOOD_RNG, default="oracle")
    p.add_argument(
        "--opp-forward",
        choices=("rowwise", "batched"),
        default="rowwise",
        help="opponent forward (vector61 oracle: rowwise = the eval's; student model: either)",
    )
    p.add_argument("--frames", type=int, default=HORIZON, help="default H5000 (smoke only)")
    p.add_argument("--out", default=str(OUT_ROOT))
    s = sub.add_parser("summarize")
    s.add_argument("--out", default=str(OUT_ROOT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {"bench": cmd_bench, "probe": cmd_probe, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
