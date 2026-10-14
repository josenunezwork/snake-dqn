#!/usr/bin/env python3
"""DEVELOPMENT DIAGNOSTIC (M7 study A): per-decision feature capture for the escape-feature AUC.

Not a screen, not gate evidence. Re-runs the B8 death-census worlds (``round_index`` 52,
the first 32 dev-probe worlds, mix ``scripted``, H5000, ``promotion-v2-watch-rect``, grid
engine, ``ego2s_forward="rowwise"``, ``ego2s_mask="advisory"``, no veto,
``--v61-backend numba --anchors fast``: record-identical by the census ``identity`` check)
for one ego2s-bv8 checkpoint and logs, per hero decision:

* everything the student observes that is scalar: the 60 ``ego2s-bv8`` scalars (incl. the
  30 v8 per-action features and the ego2s-b region scalars), a few summaries of its local
  and global planes, its own Q and the advisory mask (all decisions);
* the candidate escape features of :mod:`research.m7_feature_auc_20261014.escape_nb`
  (decisions with hero length >= ``--min-len``), with kernel timings;
* the hero's death frame and cause (frame observer).

The hero policy is a drop-in subclass of ``Ego2sSimdPolicy`` (advisory mode) that computes
the observation once, exactly as ``student_q`` does, and selects the same action; episodes
are checked against the census / probe rows world by world (mass and deaths).

Usage::

  ./venv/bin/python research/m7_feature_auc_20261014/collect.py --arm B8_s0 \
      --checkpoint <A>/redesign-m4b-mac-20261013/B8_s0/final.pth
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

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

from research.b8_death_census_20261014 import census  # noqa: E402

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
OUT_ROOT = ARTIFACTS / "m7-diagnostics-20261014" / "feature_auc"
CENSUS_RUNS = ARTIFACTS / "b8-death-census-20261014" / "runs"
PROBE_DIRS = (
    ARTIFACTS / "redesign-m4-mac-probes-20261013",
    ARTIFACTS / "redesign-m6-mac-probes-20261014",
)
CAUSE = {0: None, 1: "wall", 2: "self", 3: "head_on", 4: "enemy_body"}
PLANE_NAMES = (
    "P_reach_cnt",
    "P_reach_le12",
    "P_own_cnt",
    "P_own_255",
    "P_own_min",
    "P_enemy_cnt",
    "P_wall_cnt",
    "G_open_blocks",
    "G_own_near",
    "G_mass_near",
)


def plane_summaries(obs: Mapping[str, np.ndarray]) -> np.ndarray:
    """``(N, 10)`` float32 summaries of the planes the student sees (baseline extras)."""
    loc = obs["local"]
    glo = obs["global"].astype(np.int64)
    n = loc.shape[0]
    out = np.zeros((n, len(PLANE_NAMES)), np.float32)
    if not n:
        return out
    area = float(loc.shape[-1] * loc.shape[-2])
    reach = loc[:, 5]
    own = loc[:, 0]
    out[:, 0] = (reach > 0).sum(axis=(1, 2)) / area
    out[:, 1] = ((reach > 0) & (reach <= 12)).sum(axis=(1, 2)) / area
    out[:, 2] = (own > 0).sum(axis=(1, 2)) / area
    out[:, 3] = (own == 255).sum(axis=(1, 2)) / area
    big = np.where(own > 0, own.astype(np.int64), 256)
    out[:, 4] = np.minimum(big.min(axis=(1, 2)), 255) / 255.0
    out[:, 5] = (loc[:, 1] > 0).sum(axis=(1, 2)) / area
    out[:, 6] = (loc[:, 4] > 0).sum(axis=(1, 2)) / area
    inside = glo[:, 3] == 0
    open_ = inside & (glo[:, 0] == 0) & (glo[:, 1] == 0)
    out[:, 7] = open_.sum(axis=(1, 2)) / np.maximum(inside.sum(axis=(1, 2)), 1)
    c = glo.shape[-1] // 2
    out[:, 8] = glo[:, 0, c - 2 : c + 3, c - 2 : c + 3].sum(axis=(1, 2)) / (25 * 64.0)
    m = glo[:, 0] + glo[:, 1]
    out[:, 9] = m[:, c - 4 : c + 5, c - 4 : c + 5].sum(axis=(1, 2)) / (81 * 64.0)
    return out


class Collector:
    def __init__(self, seeds: Sequence[int], min_len: int, time_every: int) -> None:
        self.seeds = [int(s) for s in seeds]
        self.E = len(self.seeds)
        self.min_len = int(min_len)
        self.time_every = int(time_every)
        self.frame = 0
        self.dec: Dict[str, List[np.ndarray]] = {
            k: [] for k in ("env", "frame", "length", "action", "scalars", "planes", "q", "adv")
        }
        self.cand: Dict[str, List[np.ndarray]] = {k: [] for k in ("dec_index", "act", "state")}
        self.n_dec = 0
        self.death_frame = np.full(self.E, -1, np.int64)
        self.death_cause = np.zeros(self.E, np.int64)
        self.my_actions: Dict[int, np.ndarray] = {}
        self.checks: Counter = Counter()
        self.seconds: Counter = Counter()
        self.timed_rows = 0

    def record(self, sim, rows: np.ndarray, obs, q: np.ndarray, adv: np.ndarray, act) -> None:
        from research.m7_feature_auc_20261014 import escape_nb as ef

        t0 = time.perf_counter()
        envs = rows[:, 0].astype(np.int64)
        if np.any(rows[:, 1] != 0):
            raise RuntimeError("hero rows only")
        n = len(envs)
        L = np.asarray(sim.length[envs, 0], np.int64)
        d = self.dec
        d["env"].append(envs.astype(np.int16))
        d["frame"].append(np.full(n, self.frame, np.int16))
        d["length"].append(L.astype(np.int32))
        d["action"].append(np.asarray(act, np.int8))
        d["scalars"].append(np.asarray(obs["scalars"], np.float32))
        d["planes"].append(plane_summaries(obs))
        d["q"].append(np.asarray(q, np.float32))
        d["adv"].append(np.asarray(adv, bool))
        sel = np.flatnonzero(L >= self.min_len)
        if len(sel):
            t1 = time.perf_counter()
            if self.frame % self.time_every == 0:
                a, s, sec = ef.escape_features_timed(sim, envs[sel])
                for k, v in sec.items():
                    self.seconds[f"kernel_{k}"] += v
                self.timed_rows += len(sel)
            else:
                a, s = ef.escape_features(sim, envs[sel])
            self.seconds["candidates"] += time.perf_counter() - t1
            self.cand["dec_index"].append((self.n_dec + sel).astype(np.int64))
            self.cand["act"].append(a.astype(np.float32))
            self.cand["state"].append(s.astype(np.float32))
        self.my_actions[self.frame] = (envs, np.asarray(act, np.int64))
        self.n_dec += n
        self.seconds["record"] += time.perf_counter() - t0

    def observe(self, payload: Mapping[str, Any]) -> None:
        f = int(payload["frame"])
        if f != self.frame:
            self.checks["observer_frame_mismatch"] += 1
        pre, post = payload["pre"], payload["post"]
        alive = np.asarray(pre["alive"], bool)
        acts = np.asarray(pre["actions"])
        mine = self.my_actions.pop(f, None)
        if mine is not None:
            envs, a = mine
            if not np.array_equal(np.flatnonzero(alive), np.sort(envs)):
                self.checks["observer_rows_mismatch"] += 1
            self.checks["observer_action_mismatch"] += int((acts[envs] != a).sum())
        elif alive.any():
            self.checks["observer_missing_decisions"] += 1
        done = np.asarray(post["done"], bool) & alive
        cause = np.asarray(post["death_cause"])
        for e in np.flatnonzero(done & (cause != 0)):
            if self.death_frame[e] >= 0:
                self.checks["double_death"] += 1
            self.death_frame[e] = f
            self.death_cause[e] = int(cause[e])
        self.frame = f + 1

    def arrays(self) -> Dict[str, np.ndarray]:
        out = {k: np.concatenate(v) for k, v in self.dec.items()}
        if self.cand["dec_index"]:
            out.update({f"cand_{k}": np.concatenate(v) for k, v in self.cand.items()})
        return out


@contextmanager
def instrumented(col: Collector) -> Iterator[Collector]:
    from src.simd_env import ego2s_policy as ep
    from src.simd_env.ego_raster import build_ego_raster

    orig = ep.Ego2sSimdPolicy

    class CollectEgo2sPolicy(orig):  # type: ignore[misc, valid-type]
        """``Ego2sSimdPolicy`` (advisory) that also hands its observation to the collector."""

        def actions(self, masks, sim, slots):  # type: ignore[no-untyped-def]
            if self.mask_mode != "advisory":
                raise RuntimeError("the collector replicates the advisory selection only")
            rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            if not len(rows):
                return np.zeros(0, np.int64)
            obs = build_ego_raster(sim, ep.ego_config_for(self.net), backend="numba", rows=rows)
            q = ep.forward_q(self.net, obs, self.forward).numpy()
            m = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
            adv = m.copy()
            m[~m.any(axis=1), :3] = True
            act = np.where(m, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int64)
            col.record(sim, rows, obs, q, adv, act)
            return act

    ep.Ego2sSimdPolicy = CollectEgo2sPolicy
    try:
        yield col
    finally:
        ep.Ego2sSimdPolicy = orig


def reference_episodes(arm: str, sha: str) -> Optional[Dict[str, Any]]:
    """Mass / deaths per world from the census run (B8) or the dev-probe row (scripted)."""
    census_json = CENSUS_RUNS / f"b8_s{arm[-1]}.json" if arm.startswith("B8_s") else None
    if census_json is not None and census_json.exists():
        r = json.loads(census_json.read_text())
        if r["checkpoint"]["sha256"] == sha:
            return {
                "source": str(census_json),
                "mass": {str(e["world_seed"]): e["mass_integral"] for e in r["episodes"]},
                "deaths": {str(e["world_seed"]): e["deaths"] for e in r["episodes"]},
            }
    for d in PROBE_DIRS:
        p = d / f"{arm}__final__none.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            row = json.loads(line)
            if (
                row["mix"] == census.MIX
                and int(row["round_index"]) == census.ROUND_INDEX
                and row.get("student_sha256") == sha
            ):
                return {"source": str(p), "mass": row["mass"], "deaths": row["deaths"]}
    return None


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    from research.m7_feature_auc_20261014 import escape_nb as ef
    from research.redesign_m2_20261008 import ni_spec
    from src.simd_env import eval_engine as ee

    torch.set_num_threads(1)
    ckpt = Path(args.checkpoint)
    sha = census.sha256_file(ckpt)
    ctx, seeds, rosters = census.load_context(args.worlds)
    frames = int(args.frames)
    profile = ctx["profile"]
    if frames != profile.scored_horizon:
        import dataclasses

        profile = dataclasses.replace(profile, scored_horizon=frames)
    # JIT warm-up outside the timed region
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.grid_sim import GridBatchSim

    warm = GridBatchSim(BatchSimConfig(num_envs=1, num_snakes=2), seeds=[0], train_mode=False)
    ef.escape_features(warm, np.array([0]))
    col = Collector(seeds, args.min_len, args.time_every)
    t0 = time.perf_counter()
    with instrumented(col):
        records = ee.run_simd_eval(
            ctx["lookup"][ni_spec.FRP3_S12[1]],
            rosters[seeds[0]],
            frames,
            list(seeds),
            profile=profile,
            opponent_specs_by_world=dict(rosters),
            mix_id=census.MIX,
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            vector61_backend=args.v61_backend,
            scripted_anchors=args.anchors,
            frame_observer=col.observe,
            hero_ego2s=str(ckpt),
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
        )
    wall = time.perf_counter() - t0
    episodes = census.episode_rows(records, seeds)
    for e, ep_ in enumerate(episodes):
        died = col.death_frame[e] >= 0
        if died == ep_["survived"]:
            col.checks["episode_death_mismatch"] += 1
        if died and CAUSE[int(col.death_cause[e])] != ep_["death_cause"]:
            col.checks["episode_cause_mismatch"] += 1
    ref = reference_episodes(args.arm, sha) if frames == census.HORIZON else None
    match = None
    if ref is not None:
        bad = [
            ep_["world_seed"]
            for ep_ in episodes
            if float(ref["mass"][str(ep_["world_seed"])]) != ep_["mass_integral"]
            or int(ref["deaths"][str(ep_["world_seed"])]) != ep_["deaths"]
        ]
        match = {"source": ref["source"], "match": not bad, "mismatched_worlds": bad}
    arrays = col.arrays()
    out = Path(args.out) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{args.arm}.npz",
        seeds=np.asarray(seeds),
        death_frame=col.death_frame,
        death_cause=col.death_cause,
        **arrays,
    )
    n_cand = int(len(arrays.get("cand_dec_index", [])))
    meta = {
        "schema": "m7-feature-auc-collect/v1",
        "authority": "development diagnostic only (not a screen, not gate evidence)",
        "arm": args.arm,
        "checkpoint": {"path": str(ckpt), "sha256": sha},
        "mix": census.MIX,
        "round_index": census.ROUND_INDEX,
        "frames": frames,
        "seeds": seeds,
        "min_len": args.min_len,
        "engine": {
            "sim_engine": "grid",
            "v61_backend": args.v61_backend,
            "anchors": args.anchors,
            "ego2s_forward": "rowwise",
            "ego2s_mask": "advisory",
            "veto": None,
        },
        "git": census._git(),
        "wall_seconds": wall,
        "seconds": dict(col.seconds),
        "decisions": int(col.n_dec),
        "candidate_decisions": n_cand,
        "timed_rows": col.timed_rows,
        "per_decision_ms": {
            "candidates_all_families_incl_python": (
                1000.0 * col.seconds["candidates"] / max(n_cand, 1)
            ),
            "kernel_static_bcd": 1000.0 * col.seconds["kernel_static_bcd"] / max(col.timed_rows, 1),
            "kernel_full_abcd": 1000.0 * col.seconds["kernel_full"] / max(col.timed_rows, 1),
        },
        "checks": dict(col.checks),
        "reference_match": match,
        "episodes": episodes,
        "deaths": {
            str(seeds[e]): {
                "frame": int(col.death_frame[e]),
                "cause": CAUSE[int(col.death_cause[e])],
            }
            for e in range(len(seeds))
            if col.death_frame[e] >= 0
        },
    }
    (out / f"{args.arm}.json").write_text(json.dumps(census._jsonable(meta), indent=1))
    print(
        json.dumps(
            {
                "arm": args.arm,
                "wall": round(wall, 1),
                "survival": sum(e["survived"] for e in episodes) / len(episodes),
                "mass": float(np.mean([e["mass_integral"] for e in episodes])),
                "decisions": col.n_dec,
                "candidate_decisions": n_cand,
                "per_decision_ms": meta["per_decision_ms"],
                "checks": dict(col.checks),
                "reference_match": (match or {}).get("match"),
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--worlds", type=int, default=census.WORLDS)
    ap.add_argument("--frames", type=int, default=census.HORIZON)
    ap.add_argument("--min-len", type=int, default=750)
    ap.add_argument("--time-every", type=int, default=10)
    ap.add_argument("--v61-backend", choices=("python", "numba"), default="numba")
    ap.add_argument("--anchors", choices=("python", "fast"), default="fast")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    return cmd_run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
