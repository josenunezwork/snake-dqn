#!/usr/bin/env python3
"""DEVELOPMENT FEASIBILITY STUDY (research track 3, part 2): the sparse lookahead planner with
an incrementally maintained "shadow" world, run as the hero for whole H5000 episodes.

Not a screen, not gate evidence, no candidate, no serving change. ORACLE MODEL ONLY: the
rollouts use the real opponents' policies and the real env RNG (future food spawns), so the
results are an upper bound on what a d-decision lookahead with the student's own continuation
could buy, not a deployable agent.

Rule (identical to :func:`planner.planner_choice`): at every decision play the student's
greedy action ``g`` unless ``g``'s depth-d rollout (student greedy afterwards, opponents on
their own policies) dies; then play the highest-Q legal action whose rollout survives d, else
the longest-surviving one (``g`` kept when it is among the longest).

Why a shadow: with an exact model, the depth-d rollout of the greedy action at decision t+1
is the rollout at t shifted by one decision. So each env keeps ONE forked world ``d``
decisions ahead of the real one on the planned trajectory; per decision it advances one step
(amortized cost: one world step + one student forward), and the beam of alternatives runs
only when the shadow hero dies. After a switch the chosen alternative's rollout world becomes
the shadow. ``--check-naive`` recomputes :func:`planner.planner_choice` at every decision and
counts disagreements (must be 0).

Usage::

  nice -n 10 ./venv/bin/python research/planner_feasibility_20261014/shadow.py episode \
      --arm b8_s0 --depth 32
  nice -n 10 ./venv/bin/python research/planner_feasibility_20261014/shadow.py episode \
      --arm b8_s0 --depth 16 --worlds 4 --frames 1500 --check-naive
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.planner_feasibility_20261014 import planner as P  # noqa: E402


class ShadowPlanner:
    """Per-env shadow worlds + the sparse rule (oracle model, runtime-free mixes only)."""

    def __init__(
        self,
        engine: P.RolloutEngine,
        depth: int,
        rosters: Mapping[int, Sequence[Sequence[Any]]],
        seeds: Sequence[int],
        check_naive: Optional[P.RolloutEngine] = None,
    ) -> None:
        if (engine.opp_model, engine.food_rng) != ("oracle", "oracle"):
            raise ValueError("the shadow is exact only for the oracle model")
        self.engine = engine
        self.depth = int(depth)
        self.rosters = rosters
        self.seeds = [int(s) for s in seeds]
        self.state: Dict[int, Dict[str, Any]] = {}
        self.stats: Counter = Counter()
        self.sec: Counter = Counter()
        self.fires: List[Dict[str, Any]] = []
        self.check_naive = check_naive

    def _roster(self, e: int) -> Sequence[Sequence[Any]]:
        return self.rosters[self.seeds[e]]

    def _state_from(self, r: P.RolloutResult, i: int, t: int) -> Dict[str, Any]:
        w = r.final_world
        world = w if int(w.E) == 1 else P.fork_env(w, i, 1)
        if r.survives(self.depth):
            return {"world": world, "dec": t + self.depth - 1, "death": None}
        death = t + r.survived
        return {"world": world, "dec": death, "death": death}

    def _advance(self, st: Dict[str, Any], e: int, target: int) -> None:
        """Step the shadow on the greedy trajectory until decision ``target`` (or death)."""
        w = st["world"]
        roster, seed = self._roster(e), self.seeds[e]
        eng = self.engine
        while st["death"] is None and st["dec"] < target:

            def choose(p: Any) -> np.ndarray:
                a = np.ones((1, int(p.S)), dtype=np.int64)
                a[0, 0] = int(eng.greedy(p, np.array([[0, 0]], dtype=np.int64))[0])
                eng.opp_actions(p, None, roster, seed, a, np.array([0]))
                return a

            w.step_with_policy(choose)
            st["dec"] += 1
            self.stats["shadow_steps"] += 1
            if not w.alive[0, 0]:
                st["death"] = st["dec"]

    def decide(self, sim: Any, e: int, t: int, g: int, q: np.ndarray) -> int:
        """The planner's action for hero ``(e, 0)`` at decision ``t`` (``g``: greedy)."""
        t0 = time.perf_counter()
        d, eng = self.depth, self.engine
        roster, seed = self._roster(e), self.seeds[e]
        st = self.state.get(e)
        if st is None:
            r = eng.rollout(sim, e, [g], d, roster, seed, prepared=True, keep_world=True)[0]
            st = self._state_from(r, 0, t)
            self.state[e] = st
            self.stats["shadow_inits"] += 1
        else:
            self._advance(st, e, t + d - 1)
        action = int(g)
        if st["death"] is not None:
            self.stats["fires"] += 1
            legal = sim.get_legal_action_mask()[e, 0].astype(bool)
            alts = [a for a in range(6) if legal[a] and a != g]
            survived = {int(g): int(st["death"] - t)}
            res: Dict[int, Any] = {}
            if alts:
                rs = eng.rollout(sim, e, alts, d, roster, seed, prepared=True, keep_world=True)
                for i, r in enumerate(rs):
                    res[r.action] = (i, r)
                    survived[r.action] = int(r.survived)
            surv = [a for a in survived if survived[a] >= d]
            if surv:
                best = max(surv, key=lambda a: (float(q[a]), -a))
                reason = "survivor"
            else:
                longest = max(survived.values())
                pool = [a for a in survived if survived[a] == longest]
                best = int(g) if g in pool else max(pool, key=lambda a: (float(q[a]), -a))
                reason = "longest"
            if best != g:
                i, r = res[best]
                self.state[e] = self._state_from(r, i, t)
                self.stats["switches"] += 1
            self.stats[f"reason_{reason}"] += 1
            action = int(best)
            self.fires.append(
                {
                    "env": e,
                    "t": t,
                    "length": int(sim.length[e, 0]),
                    "greedy": int(g),
                    "action": action,
                    "reason": reason,
                    "survived": survived,
                }
            )
        self.sec["planner"] += time.perf_counter() - t0
        self.stats["decisions"] += 1
        if self.check_naive is not None:
            ch = P.planner_choice(self.check_naive, sim, e, d, roster, seed, prepared=True)
            self.stats["naive_agree" if ch["action"] == action else "naive_disagree"] += 1
        return action

    def drop_dead(self, alive_envs: Sequence[int]) -> None:
        keep = set(int(e) for e in alive_envs)
        for e in list(self.state):
            if e not in keep:
                del self.state[e]


@contextmanager
def planner_hero(planner: ShadowPlanner) -> Iterator[Dict[str, Any]]:
    """Patch ``Ego2sSimdPolicy`` (advisory) so the eval's hero plays the shadow planner."""
    from src.simd_env import ego2s_policy as ep

    orig = ep.Ego2sSimdPolicy
    state: Dict[str, Any] = {"t": 0, "base_sec": 0.0}

    class PlannerEgo2sPolicy(orig):  # type: ignore[misc, valid-type]
        def actions(self, masks, sim, slots):  # type: ignore[no-untyped-def]
            if self.mask_mode != "advisory":
                raise RuntimeError("the planner wraps the advisory selection only")
            t0 = time.perf_counter()
            rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            q = ep.student_q(self.net, sim, rows, self.forward).numpy()
            m = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
            m[~m.any(axis=1), :3] = True
            g = np.where(m, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int64)
            state["base_sec"] += time.perf_counter() - t0
            t = state["t"]
            planner.drop_dead(rows[:, 0])
            out = g.copy()
            for i, (e, _s) in enumerate(rows):
                out[i] = planner.decide(sim, int(e), t, int(g[i]), q[i])
            state["t"] += 1
            return out

    ep.Ego2sSimdPolicy = PlannerEgo2sPolicy
    try:
        yield state
    finally:
        ep.Ego2sSimdPolicy = orig


def cmd_episode(args: argparse.Namespace) -> int:
    ctx, seeds, rosters = P.load_context("scripted", args.worlds)
    P._setup_threads()
    path = str(P.b8_path(args.arm))
    engine = P.RolloutEngine(path, forward="rowwise")
    naive = P.RolloutEngine(path, forward="rowwise") if args.check_naive else None
    planner = ShadowPlanner(engine, args.depth, rosters, seeds, check_naive=naive)
    t0 = time.perf_counter()
    with planner_hero(planner) as st:
        recs = P.run_eval(ctx, seeds, rosters, args.arm, "scripted", int(args.frames))
    wall = time.perf_counter() - t0
    census = json.loads((P.CENSUS_RUNS / f"{args.arm}.json").read_text())
    base = {int(r["world_seed"]): r for r in census["episodes"]}
    worlds = []
    for r in recs:
        s = int(r["seed"])
        worlds.append(
            {
                "world_seed": s,
                "mass_integral": float(r["mass_integral"]),
                "deaths": int(r["deaths"]),
                "survival_fraction": float(r["survival_fraction"]),
                "max_mass": float(r["max_mass"]),
                "death_cause": (r.get("probes") or {}).get("death_cause"),
                "base_mass_integral": float(base[s]["mass_integral"]),
                "base_deaths": int(base[s]["deaths"]),
            }
        )
    n = len(worlds)
    full = int(args.frames) == P.HORIZON
    summary = {
        "worlds": n,
        "survival": sum(w["deaths"] == 0 for w in worlds) / n,
        "mass": float(np.mean([w["mass_integral"] for w in worlds])),
        "base_survival": sum(w["base_deaths"] == 0 for w in worlds) / n if full else None,
        "base_mass": float(np.mean([w["base_mass_integral"] for w in worlds])) if full else None,
        "paired_mass_delta": (
            float(np.mean([w["mass_integral"] - w["base_mass_integral"] for w in worlds]))
            if full
            else None
        ),
        "deaths_avoided": (
            sum(w["base_deaths"] > 0 and w["deaths"] == 0 for w in worlds) if full else None
        ),
        "deaths_new": (
            sum(w["base_deaths"] == 0 and w["deaths"] > 0 for w in worlds) if full else None
        ),
        "decisions": planner.stats["decisions"],
        "fires": planner.stats["fires"],
        "switches": planner.stats["switches"],
        "fire_rate": planner.stats["fires"] / max(1, planner.stats["decisions"]),
        "planner_ms_per_decision": 1e3
        * planner.sec["planner"]
        / max(1, planner.stats["decisions"]),
        "base_policy_ms_per_decision": 1e3 * st["base_sec"] / max(1, planner.stats["decisions"]),
    }
    out = {
        "schema": "planner-shadow-episode/v1",
        "authority": "development feasibility study only (not a screen, not gate evidence); "
        "ORACLE model (real opponent policies + real env RNG)",
        "arm": args.arm,
        "depth": args.depth,
        "frames": int(args.frames),
        "summary": summary,
        "stats": dict(planner.stats),
        "engine_timers": engine.timers(),
        "fires": planner.fires,
        "worlds": worlds,
        "wall_seconds": wall,
        "git": P._git(),
        "host_load": os.getloadavg(),
    }
    tag = f"episode_{args.arm}_d{args.depth}" + ("_check" if args.check_naive else "")
    if int(args.frames) != P.HORIZON or int(args.worlds) != P.WORLDS:
        tag += f"_w{args.worlds}_f{args.frames}"
    path_out = Path(args.out) / f"{tag}.json"
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text(json.dumps(P._jsonable(out), indent=1))
    print(f"[episode] wrote {path_out}: {json.dumps(P._jsonable(summary))}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("episode")
    e.add_argument("--arm", choices=P.B8_ARMS, required=True)
    e.add_argument("--depth", type=int, default=32)
    e.add_argument("--worlds", type=int, default=P.WORLDS)
    e.add_argument("--frames", type=int, default=P.HORIZON)
    e.add_argument("--check-naive", action="store_true")
    e.add_argument("--out", default=str(P.OUT_ROOT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {"episode": cmd_episode}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
