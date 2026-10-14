#!/usr/bin/env python3
"""Actor hero forward: four ``Tensor.to`` uploads (as ``actor_main``) vs one packed upload.

The real-run profile (``REPORT.md``) puts the largest single actor component in the hero's
host->device copies (``Tensor.to`` inside ``choose``: local, global, scalars, mask). This
times one actor frame's hero step on synthetic bv8 observations (128 rows by default):

* ``four``: ``torch.from_numpy(x).to(dev)`` for local / global / scalars and the mask, then
  the forward, masked argmax and ``.cpu()`` (the shipped actor path);
* ``packed``: the four arrays copied into one preallocated uint8 host block, ONE ``.to``,
  typed views on the device (the learner's ``_Staging`` idea), then the same forward;
* ``packed_nb``: ``packed`` with ``non_blocking=True`` (no host wait at the upload; the only
  host sync left is the final ``.cpu()``). Safe for this block: it is refilled only by the
  next call, after this call ended with ``.cpu()`` (all of its queued work done).

All variants compute the same Q from the same bytes (asserted bitwise on every observation
set). Interleaved rounds, median reported. Bench only: the actor is unchanged.

Usage (noisy when other jobs share the GPU)::

    ./venv/bin/python research/perf_profile_20261013/bench_hero_h2d.py --device mps
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from research.perf_learner_20261013 import bench_learner as bl  # noqa: E402

LOCAL, GLOBAL, NSCAL = (7, 31, 31), (4, 37, 37), 60


def sync(dev: str) -> None:
    bl.sync(dev)


def make_obs(rows: int, seed: int):
    r = np.random.default_rng(seed)
    loc = np.where(r.random((rows,) + LOCAL) < 0.15, r.integers(1, 256, (rows,) + LOCAL), 0)
    glo = np.where(r.random((rows,) + GLOBAL) < 0.15, r.integers(1, 256, (rows,) + GLOBAL), 0)
    return {
        "local": loc.astype(np.uint8),
        "global": glo.astype(np.uint8),
        "scalars": r.standard_normal((rows, NSCAL)).astype(np.float32),
        "mask": r.random((rows, 6)) < 0.8,
    }


class Packed:
    def __init__(self, rows: int) -> None:
        segs = [("local", (rows,) + LOCAL, np.uint8), ("global", (rows,) + GLOBAL, np.uint8)]
        segs += [("scalars", (rows, NSCAL), np.float32), ("mask", (rows, 6), np.bool_)]
        self.layout, off = [], 0
        for name, shape, dt in segs:
            nb = int(np.prod(shape)) * np.dtype(dt).itemsize
            self.layout.append((name, shape, dt, off, nb))
            off += -(-nb // 8) * 8
        self.flat = np.zeros(off, dtype=np.uint8)
        self.v = {
            n: np.ndarray(s, dtype=d, buffer=self.flat, offset=o) for n, s, d, o, _ in self.layout
        }
        self.host = torch.from_numpy(self.flat)

    def upload(self, obs, dev, non_blocking: bool = False):
        for k in ("local", "global", "scalars", "mask"):
            self.v[k][...] = obs[k]
        d = self.host.to(dev, non_blocking=non_blocking)
        tdt = {np.float32: torch.float32, np.bool_: torch.bool}
        out = {}
        for name, shape, dt, o, nb in self.layout:
            seg = d[o : o + nb]
            out[name] = (seg if dt is np.uint8 else seg.view(tdt[dt])).view(shape)
        return out


@torch.no_grad()
def step_four(net, obs, dev):
    x = [
        torch.from_numpy(np.ascontiguousarray(obs[k])).to(dev)
        for k in ("local", "global", "scalars")
    ]
    q = net(*x)
    mt = torch.from_numpy(obs["mask"]).to(dev)
    g = torch.where(mt, q, torch.full_like(q, -1e9)).argmax(dim=1)
    return g.cpu().numpy(), q


@torch.no_grad()
def step_packed(net, obs, dev, packed, non_blocking: bool = False):
    t = packed.upload(obs, dev, non_blocking)
    q = net(t["local"], t["global"], t["scalars"])
    g = torch.where(t["mask"], q, torch.full_like(q, -1e9)).argmax(dim=1)
    return g.cpu().numpy(), q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    ns = ap.parse_args(argv)
    dev = ns.device
    net = bl.init_net(NSCAL).to(dev).eval()
    obs = [make_obs(ns.rows, s) for s in range(8)]
    packed = Packed(ns.rows)
    fns = {
        "four": lambda o: step_four(net, o, dev),
        "packed": lambda o: step_packed(net, o, dev, packed),
        "packed_nb": lambda o: step_packed(net, o, dev, packed, True),
    }
    same = True
    for o in obs:  # every variant: the same greedy actions and the same Q, bit for bit
        g0, q0 = fns["four"](o)
        for name in ("packed", "packed_nb"):
            g, q = fns[name](o)
            same &= bool(np.array_equal(g0, g) and torch.equal(q0.cpu(), q.cpu()))
    times = {k: [] for k in fns}
    names = list(fns)
    for rnd in range(ns.rounds):
        for name in names[rnd % len(names) :] + names[: rnd % len(names)]:
            sync(dev)
            t0 = time.perf_counter()
            for i in range(ns.iters):
                fns[name](obs[i % len(obs)])
            sync(dev)
            times[name].append(1e3 * (time.perf_counter() - t0) / ns.iters)
    res = {
        "device": dev,
        "rows": ns.rows,
        "same_q_bitwise": same,
        "loadavg": os.getloadavg(),
        "note": "NOISY: other trainings shared the CPU and the MPS GPU",
        "ms_per_frame_median": {k: round(statistics.median(v), 3) for k, v in times.items()},
        "ms_per_frame_rounds": {k: [round(x, 3) for x in v] for k, v in times.items()},
    }
    print(json.dumps(res))
    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
