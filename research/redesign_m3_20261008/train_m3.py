#!/usr/bin/env python3
"""M3: synchronous vectorized Double DQN fine-tune of the ego2s-b student (pre-registered).

Implements ``PREREGISTRATION_M3.md`` §3–§4 in ONE process (doc §6): ``GridBatchSim`` worlds in
lockstep on the CPU (one thread), every network forward and update on the Mac GPU (MPS).

* **Worlds:** E worlds per 5000-frame segment (gate world: pinned deployment config,
  ``promotion-v2-watch-rect`` world, Watch food, everyone respawns), 75% with the strict
  scripted roster (``FastProfileAnchorSimdPolicy``), 25% with the strict frozen roster
  (vector61 pool, batched forwards). Seeds: namespace ``redesign-m3-train/v1``, key
  ``seed<s>``, offset ``segment * E``. The sim's reward γ is the learner's γ at segment start.
* **Hero (slot 0 of every world):** ego2s-b observation (numba), Q on MPS, ε-greedy over the
  resolved mask (normal-only fallback). Death is terminal for TD; the hero respawns into a
  new episode; a segment end is a truncation (bootstrap from the final observation).
* **Learner:** n-step 5 Double DQN, dueling ``Ego2sNet``, Huber TD, Adam 1e-4, batch 512 =
  384 agent + 128 demonstration samples, grad-norm clip 10, target sync every 1000 updates,
  one update per 128 new agent transitions (replay ratio 4), 500k agent replay.
* **Anchor:** on demonstration samples only (no TD: they have no reward / next state):
  ``λ(t) · (advantage-Huber ×10 + KL τ 0.05 to the teacher Q + DQfD margin to the v8 action)``
  (the M2b loss without its raw-Q term). Demonstrations stream from the M2b shards (hold-out
  worlds ``world_seed % 10 == 0`` excluded) into a 300k ring, one 10k shard swapped per 1000
  updates.
* **Schedules (transitions):** γ 0.99 to 2M, linear to 0.995 at 6M; λ 1 to 8M, exponential
  to 0.1 at 15M; ε 0.05 → 0.01 linear to 5M.
* **Tripwires:** HALT on NaN/inf or action collapse (> 90% of a 100k-decision window);
  FLAG max |Q| > 5× the demonstrations' max |Q| and held-out v8 agreement < 0.6.

Writes ``<out>/log.jsonl`` (every 100k transitions), checkpoints every ``--checkpoint-every``
transitions, ``<out>/final.pth`` and ``<out>/summary.json``. Development run; not evidence.
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
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
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

SCHEMA = "redesign-m3-train/v1"
TRAIN_NAMESPACE = "redesign-m3-train/v1"
LOCAL = (7, 31, 31)
GLOBAL = (4, 37, 37)
NSCAL = 30


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


# ----------------------------------------------------------------------------- replay
class ObsRing:
    """uint8 observation ring (+ scalars, resolved mask) addressed by a monotone counter."""

    def __init__(self, capacity: int) -> None:
        self.cap = int(capacity)
        self.local = np.zeros((capacity,) + LOCAL, dtype=np.uint8)
        self.glob = np.zeros((capacity,) + GLOBAL, dtype=np.uint8)
        self.scal = np.zeros((capacity, NSCAL), dtype=np.float32)
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

    def sample(self, rng: np.random.Generator, n: int, ring: ObsRing) -> np.ndarray:
        idx = rng.integers(0, self.size, size=n * 2)
        ok = ring.valid(self.s[idx]) & ((self.nxt[idx] < 0) | ring.valid(self.nxt[idx]))
        idx = idx[ok][:n]
        return idx


# ----------------------------------------------------------------------------- demonstrations
DEMO_KEYS = ("local", "global", "scalars", "q_teacher", "a_v8", "mask_resolved")


class DemoStream:
    """Rotating buffer of M2b teacher demonstrations (hold-out worlds excluded)."""

    def __init__(self, data: Path, rounds: List[int], capacity: int, seed: int) -> None:
        paths: List[Path] = []
        for r in rounds:
            paths += sorted((data / f"round-{r}").glob("chunk-*-part-*.npz"))
        if not paths:
            raise SystemExit("no demonstration shards")
        rng = np.random.default_rng([seed, 991])
        self.order = [paths[i] for i in rng.permutation(len(paths))]
        self.cap = int(capacity)
        self.buf = {
            "local": np.zeros((capacity,) + LOCAL, dtype=np.uint8),
            "global": np.zeros((capacity,) + GLOBAL, dtype=np.uint8),
            "scalars": np.zeros((capacity, NSCAL), dtype=np.float32),
            "q_teacher": np.zeros((capacity, 6), dtype=np.float32),
            "a_v8": np.zeros(capacity, dtype=np.int64),
            "mask_resolved": np.zeros((capacity, 6), dtype=bool),
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

    def _put(self, part: Dict[str, np.ndarray]) -> None:
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
            self._put(self.load(self.order[self.next_shard % len(self.order)]))
            self.next_shard += 1

    def prefetch(self) -> None:
        """Start decompressing the next shard in the background."""
        if self._thread is not None and self._thread.is_alive():
            return
        path = self.order[self.next_shard % len(self.order)]
        self.next_shard += 1
        self._thread = threading.Thread(target=lambda: self._q.put(self.load(path)), daemon=True)
        self._thread.start()

    def swap_if_ready(self) -> bool:
        try:
            part = self._q.get_nowait()
        except queue.Empty:
            return False
        self._put(part)
        return True

    def sample(self, rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=n)
        return {k: v[idx] for k, v in self.buf.items()}


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


# ----------------------------------------------------------------------------- worlds
def build_segment(args, ctx, segment: int, gamma: float):
    from src.core.world_runtime import WorldRuntimeSpec
    from src.simd_env import eval_engine as ee
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.grid_sim import GridBatchSim
    from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

    E = args.worlds
    from research.apex_safety_20260926 import dev_screen

    seeds = dev_screen.screen_seeds((segment + 1) * E, TRAIN_NAMESPACE, f"seed{args.seed}")[
        segment * E :
    ]
    base = ee._config_from_game_config(6, gamma, ctx["profile"])
    cap = WorldRuntimeSpec.source_exact(ctx["profile"]).body_storage_capacity
    sim = GridBatchSim(
        replace(base, num_envs=E, body_storage_capacity=cap),
        seeds=seeds,
        train_mode=False,
        allow_respawn=True,
    )
    n_frozen = int(round(E * args.frozen_frac))
    mix_of = ["scripted"] * (E - n_frozen) + ["frozen"] * n_frozen
    rows = {(int(r["world_seed"]), r["mix"]): r for r in dev_screen._design_rows(seeds)}
    lookup = ctx["lookup"]
    groups: Dict[Any, List[tuple]] = {}
    for e, seed in enumerate(seeds):
        for slot, member in enumerate(rows[(int(seed), mix_of[e])]["slots"], start=1):
            groups.setdefault(lookup[member["member_sha256"]], []).append((e, slot))
    runtime = Vector61Runtime()
    controlled = np.zeros((E, 6), dtype=bool)
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


# ----------------------------------------------------------------------------- main loop
def main() -> int:
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
    args = ap.parse_args()

    import torch

    from research.redesign_m2_20261008.gen_data import guard_ok
    from src.model.ego2s_network import load_ego2s_checkpoint, save_ego2s_checkpoint
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

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
    sched = Schedules()
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
        return float((pred == holdout["a_v8"]).mean())

    log_path = args.out / "log.jsonl"
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
        "init_sha256": __import__("hashlib").sha256(Path(args.init).read_bytes()).hexdigest(),
    }

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
        td = torch.nn.functional.smooth_l1_loss(qa, y)
        db = {k: to_dev(d[k]) for k in ("q_teacher", "a_v8", "mask_resolved")}
        anc = anchor_loss(torch, qd, db, margin)
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
    n_step = 5
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
                    obs = build_ego_raster(prepared, B, backend="numba", rows=hero)
                    m = masks[hero[:, 0], 0]
                    with torch.no_grad():
                        q = online(
                            to_dev(obs["local"]), to_dev(obs["global"]), to_dev(obs["scalars"])
                        )
                        greedy = masked_argmax(q, to_dev(m)).cpu().numpy()
                    mm = m.copy()
                    mm[~mm.any(axis=1), :3] = True
                    rand = np.array([rng.choice(np.flatnonzero(r)) for r in mm])
                    act = np.where(rng.random(len(hero)) < eps, rand, greedy)
                    actions[hero[:, 0], 0] = act
                    chosen.update(hero=hero, obs=obs, mask=m, act=act, greedy=greedy)
                for spec, cells in groups.items():
                    live = cells[alive[cells[:, 0], cells[:, 1]]]
                    if len(live):
                        mk = masks[live[:, 0], live[:, 1]]
                        actions[live[:, 0], live[:, 1]] = policies[spec].actions(mk, prepared, live)
                return actions

            sim.step_with_policy(choose)
            runtime.observe_step(sim)
            reward = sim.get_reward()[:, 0]
            done = sim.get_done()[:, 0]
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
            if state["transitions"] >= next_ckpt:
                save_ego2s_checkpoint(
                    args.out / f"ckpt_{next_ckpt // 1000}k.pth",
                    online,
                    {**meta_base, "transitions": state["transitions"], "updates": state["updates"]},
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

    save_ego2s_checkpoint(
        args.out / "final.pth",
        online,
        {**meta_base, "transitions": state["transitions"], "updates": state["updates"]},
    )
    summary = {
        "schema": SCHEMA,
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
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("transitions", "rate", "halt", "flags")}))
    return 1 if state["halt"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
