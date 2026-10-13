#!/usr/bin/env python3
"""M3-C sizing benchmarks (Mac, one thread; development only).

* ``actor``: the M3 environment loop without learning (``train_m3.build_segment`` worlds, 75%
  scripted / 25% frozen-pool, hero = the M2b student acting greedily on ``--device``);
  reports hero transitions/s and the split between sim+opponents, featurizer and forward.
* ``learner``: ``update()``-equivalent steps of the M3 learner (384 replay + 128 demo rows,
  Double DQN target, anchor loss, Adam) from a replay filled with real observations;
  reports updates/s and the split between host-side batch assembly and device compute.

  nice -n 10 ./venv/bin/python research/redesign_m3c_20261008/bench_m3c.py actor --worlds 64
  nice -n 10 ./venv/bin/python research/redesign_m3c_20261008/bench_m3c.py learner --device mps
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STUDENT = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008/student_m2b_final.pth"
)
DEMO = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008/data")


def bench_actor(args) -> dict:
    import torch

    from research.redesign_m3_20261008 import train_m3 as tm
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    torch.set_num_threads(1)
    device = torch.device(args.device)
    net = load_ego2s_checkpoint(args.student, device).eval()
    ctx = gid._context(1)
    B = EgoRasterConfig(version="b")
    ns = argparse.Namespace(worlds=args.worlds, frozen_frac=0.25, seed=args.seed)
    sim, runtime, policies, groups = tm.build_segment(ns, ctx, 0, 0.99)
    E = sim.E
    t = {"feat": 0.0, "fwd": 0.0, "opp": 0.0}
    n = [0]

    def choose(prepared):
        masks = prepared.get_resolved_action_mask()
        alive = prepared.get_alive()
        actions = np.ones((E, prepared.S), dtype=np.int64)
        runtime.prepare(prepared)
        hero = np.argwhere(alive[:, :1])
        if len(hero):
            t0 = time.perf_counter()
            obs = build_ego_raster(prepared, B, backend="numba", rows=hero)
            t1 = time.perf_counter()
            with torch.no_grad():
                q = net(
                    *[torch.from_numpy(obs[k]).to(device) for k in ("local", "global", "scalars")]
                )
                g = q.argmax(dim=1).cpu().numpy()
            t2 = time.perf_counter()
            t["feat"] += t1 - t0
            t["fwd"] += t2 - t1
            actions[hero[:, 0], 0] = g
            n[0] += len(hero)
        t3 = time.perf_counter()
        for spec, cells in groups.items():
            live = cells[alive[cells[:, 0], cells[:, 1]]]
            if len(live):
                mk = masks[live[:, 0], live[:, 1]]
                actions[live[:, 0], live[:, 1]] = policies[spec].actions(mk, prepared, live)
        t["opp"] += time.perf_counter() - t3
        return actions

    for _ in range(args.warm_frames):
        sim.step_with_policy(choose)
        runtime.observe_step(sim)
    for k in t:
        t[k] = 0.0
    n[0] = 0
    t0 = time.perf_counter()
    for _ in range(args.frames):
        sim.step_with_policy(choose)
        runtime.observe_step(sim)
    wall = time.perf_counter() - t0
    return {
        "mode": "actor",
        "device": args.device,
        "worlds": E,
        "frames": args.frames,
        "transitions": n[0],
        "wall": wall,
        "rate": n[0] / wall,
        "featurizer_s": t["feat"],
        "forward_s": t["fwd"],
        "opponents_s": t["opp"],
        "sim_other_s": wall - t["feat"] - t["fwd"] - t["opp"],
    }


def bench_learner(args) -> dict:
    import torch

    from research.redesign_m3_20261008 import train_m3 as tm
    from src.model.ego2s_network import load_ego2s_checkpoint

    torch.set_num_threads(1)
    device = torch.device(args.device)
    online = load_ego2s_checkpoint(args.student, device).train()
    target = load_ego2s_checkpoint(args.student, device).eval()
    opt = torch.optim.Adam(online.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    # Real observations: held-in demonstration rows stand in for agent observations.
    part = tm.DemoStream.load(sorted((DEMO / "round-12").glob("chunk-*-part-*.npz"))[0])
    n = len(part["a_v8"])
    ring = tm.ObsRing(args.replay + 1024)
    reps = (args.replay // n) + 1
    for _ in range(reps):
        ring.add(
            {"local": part["local"], "global": part["global"], "scalars": part["scalars"]},
            part["mask_resolved"],
        )
    replay = tm.Replay(args.replay)
    lo = ring.count - min(ring.cap - 1024, args.replay)
    for i in range(args.replay):
        s = lo + int(rng.integers(0, ring.count - lo - 5))
        replay.add(s, int(rng.integers(0, 6)), float(rng.normal()), 0.97, s + 5)
    demos = tm.DemoStream(DEMO, [12], 60_000, 0)
    demos.fill_initial(6)
    margin = json.loads((DEMO / "margin.json").read_text())["margin"]

    def to_dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    def masked_argmax(q, m):
        m = m.clone()
        m[~m.any(dim=1), :3] = True
        return torch.where(m, q, torch.full_like(q, -1e9)).argmax(dim=1)

    host = dev = 0.0
    for i in range(args.warm + args.updates):
        if i == args.warm:
            host = dev = 0.0
            if device.type == "mps":
                torch.mps.synchronize()
            t_start = time.perf_counter()
        t0 = time.perf_counter()
        idx = replay.sample(rng, 384, ring)
        sl, sg, ss, _ = ring.get(replay.s[idx])
        nl, ng, ns, nm = ring.get(replay.nxt[idx])
        d = demos.sample(rng, 128)
        xl = np.concatenate([sl, d["local"]])
        xg = np.concatenate([sg, d["global"]])
        xs = np.concatenate([ss, d["scalars"]])
        t1 = time.perf_counter()
        q_all = online(to_dev(xl), to_dev(xg), to_dev(xs))
        q, qd = q_all[: len(idx)], q_all[len(idx) :]
        a = to_dev(replay.a[idx])
        qa = q.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            nlt, ngt, nst, nmt = to_dev(nl), to_dev(ng), to_dev(ns), to_dev(nm)
            a_star = masked_argmax(online(nlt, ngt, nst), nmt)
            q_next = target(nlt, ngt, nst).gather(1, a_star[:, None]).squeeze(1)
            y = to_dev(replay.R[idx]) + to_dev(replay.disc[idx]) * q_next
        td = torch.nn.functional.smooth_l1_loss(qa, y)
        db = {k: to_dev(d[k]) for k in ("q_teacher", "a_v8", "mask_resolved")}
        loss = td + tm.anchor_loss(torch, qd, db, margin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        opt.step()
        float(loss.detach())  # synchronize (the trainer reads the loss every update)
        t2 = time.perf_counter()
        host += t1 - t0
        dev += t2 - t1
    wall = time.perf_counter() - t_start
    return {
        "mode": "learner",
        "device": args.device,
        "updates": args.updates,
        "wall": wall,
        "ms_per_update": 1000 * wall / args.updates,
        "host_batch_ms": 1000 * host / args.updates,
        "device_ms": 1000 * dev / args.updates,
        "transitions_per_s_at_rr4": 128 * args.updates / wall,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("actor", "learner"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=90)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warm-frames", type=int, default=50)
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--replay", type=int, default=50_000)
    ap.add_argument("--student", default=STUDENT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    row = bench_actor(args) if args.mode == "actor" else bench_learner(args)
    print(json.dumps(row))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
