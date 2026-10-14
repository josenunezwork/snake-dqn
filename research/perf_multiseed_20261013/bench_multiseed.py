#!/usr/bin/env python3
"""Multi-seed learner throughput: K seeds stacked (vmap) vs K sequential models (perf item 5).

Times K independent ego2s learner updates (the ``learner_m3c`` T1 update: 384 agent rows,
128 anchor (label) rows per seed, uint8 planes, bv8 scalars by default) four ways:

* ``stacked``     one :class:`StackedLearner` call updates all K seeds (vmap over stacked
                  parameters; one upload, one host sync per call).
* ``grouped``     the same with ``forward="grouped"`` (hand-written grouped convolutions +
                  batched matmuls instead of vmap; same math, no per-layer permutes).
* ``sequential``  K separate models + optimizers updated one after another in ONE process
                  (``single_update``: one upload and one host sync per seed).
* ``procs``       K separate processes, each updating one model, all sharing the device: the
                  status quo of K training runs on one GPU. Opt in with ``--arms ...,procs``
                  (``--procs-ks`` limits its K values). The workers start once; each round
                  releases them together on a barrier (interleaved with the in-process
                  arms); throughput = all updates / (last end - first start). Starts K
                  extra processes: on a shared Mac keep K <= 2.

Every call uploads its batch from host memory (pinned on CUDA) and reads the finite-loss
flag(s) back, as the trainer does. Rounds are interleaved (arm order rotated each round) and
the median round is reported. ``seed_updates_per_s`` counts per-seed updates (K per stacked
call); ``transitions_per_s`` = seed updates/s x 128, the summed ingest rate the learner can
sustain over all K runs; ``achieved_tflops`` = seed updates/s x ``gflop_per_seed_update``
(FlopCounterMode on one ``single_update``), the device's useful fp32 work rate.

Numerics (default on; ``--no-numerics`` skips): the same 3 batches go through each stacked
arm and through ``sequential`` from equal initial weights on each device (and through
``sequential`` on the CPU for a GPU device); reported are the worst-seed parameter drift ratio
``|theta_stacked - theta_seq| / |theta_seq - theta_0|``, Q max |diff| and argmax agreement on
512 held-out rows. On CPU this is float reassociation only; on a GPU it adds that device's
grouped-vs-plain convolution difference. ``stacked_vs_grouped_params`` compares the two
stacked arms' final weights (bitwise equal on MPS, torch 2.9.1).

Usage (numbers taken while other jobs share the CPU / GPU are NOISY; re-run quiet)::

    ./venv/bin/python research/perf_multiseed_20261013/bench_multiseed.py \\
        --devices mps --ks 1,2,4 --arms stacked,grouped,sequential,procs --procs-ks 1,2 \\
        --out /tmp/bench_multiseed_mps.json
    ./venv/bin/python research/perf_multiseed_20261013/bench_multiseed.py \\
        --devices cpu --ks 1,2,4 --arms stacked,grouped,sequential --rounds-cpu 3
    # CUDA pod (quiet; the go/no-go run, see DESIGN.md), with the K-processes status quo:
    ./venv/bin/python research/perf_multiseed_20261013/bench_multiseed.py \\
        --devices cuda --ks 1,2,4,8 --arms stacked,grouped,sequential,procs --calls 50 \\
        --out bench_multiseed_cuda.json
    ./venv/bin/python research/perf_multiseed_20261013/bench_multiseed.py \\
        --devices cuda --ks 1,2,4,8 --arms stacked,grouped --precision bf16 --calls 50
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from research.perf_multiseed_20261013 import stacked_learner as sl  # noqa: E402
from research.redesign_m3c_20261008 import learner_m3c as lm  # noqa: E402
from src.model.ego2s_network import Ego2sNet  # noqa: E402

NSCALS = {"b": 30, "bv8": 60}  # train_m3c --obs-version -> scalars (bv8: the M5 recipe)
N_HOST_BATCHES = 3  # distinct batches per seed, cycled
STACKED_ARMS = {"stacked": "vmap", "grouped": "grouped"}  # arm -> StackedLearner forward
LAM, MARGIN = 0.5, 0.3


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def make_nets(k: int, nscal: int) -> List[Ego2sNet]:
    nets = []
    for i in range(k):
        torch.manual_seed(1000 + i)
        nets.append(Ego2sNet(7, nscal))
    return nets


def host_batches(k: int, n: int, m: int, nscal: int, pin: bool, seed: int = 0):
    """``[batch][seed] -> dict of host tensors`` (pinned on CUDA)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_HOST_BATCHES):
        row = []
        for _ in range(k):
            b = {kk: torch.from_numpy(v) for kk, v in sl.synthetic_batch(rng, n, m, nscal).items()}
            row.append({kk: (v.pin_memory() if pin else v) for kk, v in b.items()})
        out.append(row)
    return out


def stack_host(batches, pin: bool):
    out = []
    for row in batches:
        st = {kk: torch.stack([b[kk] for b in row]) for kk in row[0]}
        out.append({kk: (v.pin_memory() if pin else v) for kk, v in st.items()})
    return out


def upload(b: Dict[str, torch.Tensor], device: torch.device, nb: bool):
    return {k: v.to(device, non_blocking=nb) for k, v in b.items()}


class StackedArm:
    def __init__(self, k, device, n, m, nscal, adam, precision, pin, forward="vmap"):
        self.k, self.device, self.n, self.m, self.nb = k, device, n, m, pin
        self.lrn = sl.StackedLearner(
            make_nets(k, nscal),
            device=device,
            margin=MARGIN,
            adam=adam,
            precision=precision,
            forward=forward,
        )
        self.host = stack_host(host_batches(k, n, m, nscal, pin), pin)
        self.i = 0

    def call(self) -> None:
        b = upload(self.host[self.i % N_HOST_BATCHES], self.device, self.nb)
        self.i += 1
        self.lrn.update(b, self.n, self.m, LAM)  # reads the K finite flags (one sync)


class SequentialArm:
    def __init__(self, k, device, n, m, nscal, adam, precision, pin):
        if precision != "fp32":
            raise ValueError("sequential arm: fp32 only (use the stacked arm for bf16)")
        self.k, self.device, self.n, self.m, self.nb = k, device, n, m, pin
        self.seeds = []
        for net in make_nets(k, nscal):
            online = copy.deepcopy(net).to(device).train()
            target = copy.deepcopy(net).to(device).eval()
            opt = torch.optim.Adam(online.parameters(), lr=lm.LR, fused=(adam == "fused") or None)
            self.seeds.append((online, target, opt, {}))
        self.host = host_batches(k, n, m, nscal, pin)
        self.i = 0

    def call(self) -> None:
        row = self.host[self.i % N_HOST_BATCHES]
        self.i += 1
        for (online, target, opt, st), hb in zip(self.seeds, row):
            sl.single_update(
                online,
                target,
                opt,
                upload(hb, self.device, self.nb),
                self.n,
                self.m,
                LAM,
                MARGIN,
                st,
            )  # bool(finite): one sync per seed


def time_calls(arm, calls: int, device: torch.device) -> float:
    t0 = time.perf_counter()
    for _ in range(calls):
        arm.call()
    sync(device)
    return time.perf_counter() - t0


# ----------------------------------------------------------------------------- procs arm
def _proc_worker(rank, dev, n, m, nscal, adam, threads, calls, warm, rounds, barrier, q):
    torch.set_num_threads(threads)
    device = torch.device(dev)
    arm = SequentialArm(1, device, n, m, nscal, adam, "fp32", device.type == "cuda")
    time_calls(arm, warm, device)
    q.put(("ready", rank))
    for _ in range(rounds):
        barrier.wait()  # released by the parent (interleaved with the in-process arms)
        t0 = time.perf_counter()
        for _ in range(calls):
            arm.call()
        sync(device)
        q.put((rank, t0, time.perf_counter()))


class ProcsGroup:
    """K worker processes (one model each); :meth:`round` runs one timed round in all of them
    at once. ``time.perf_counter`` is a system-wide monotonic clock on macOS and Linux."""

    def __init__(self, k, dev, ns, calls, warm, rounds) -> None:
        ctx = mp.get_context("spawn")
        self.k = k
        self.barrier = ctx.Barrier(k + 1)
        self.q = ctx.Queue()
        args = (dev, ns.agent_rows, ns.anchor_rows, NSCALS[ns.obs_version], ns.adam, ns.threads)
        args += (calls, warm, rounds)
        self.procs = [
            ctx.Process(target=_proc_worker, args=(r,) + args + (self.barrier, self.q))
            for r in range(k)
        ]
        for p in self.procs:
            p.start()
        for _ in range(k):
            assert self.q.get(timeout=3600)[0] == "ready"

    def round(self) -> float:
        self.barrier.wait()
        rows = [self.q.get(timeout=3600) for _ in range(self.k)]
        return max(r[2] for r in rows) - min(r[1] for r in rows)

    def close(self) -> None:
        for p in self.procs:
            p.join(timeout=60)


# ----------------------------------------------------------------------------- numerics
def numerics(k: int, device: torch.device, ns, arms: List[str]) -> Dict[str, Any]:
    """Each stacked arm vs sequential on ``device`` (and vs CPU sequential) from equal
    weights."""
    n, m, nscal = ns.agent_rows, ns.anchor_rows, NSCALS[ns.obs_version]
    rows = host_batches(k, n, m, nscal, False, seed=11)
    hold = sl.synthetic_batch(np.random.default_rng(12), 512, 0, nscal)
    hold_t = [torch.from_numpy(hold[x][:512]) for x in ("local", "global", "scalars")]
    nets = make_nets(k, nscal)
    theta0 = [torch.cat([p.detach().reshape(-1) for p in net.parameters()]) for net in nets]

    def seq(dev):
        out = []
        for i, net in enumerate(nets):
            online = copy.deepcopy(net).to(dev).train()
            target = copy.deepcopy(net).to(dev).eval()
            opt = torch.optim.Adam(
                online.parameters(), lr=lm.LR, fused=(ns.adam == "fused") or None
            )
            st: dict = {}
            for row in rows:
                sl.single_update(
                    online, target, opt, upload(row[i], dev, False), n, m, LAM, MARGIN, st
                )
            out.append(online.cpu())
        return out

    res: Dict[str, Any] = {"steps": len(rows)}
    refs = {"sequential_same_device": seq(device)}
    if device.type != "cpu":
        refs["sequential_cpu"] = seq(torch.device("cpu"))
    final: Dict[str, torch.Tensor] = {}
    for arm in arms:
        lrn = sl.StackedLearner(
            nets,
            device=device,
            margin=MARGIN,
            adam=ns.adam,
            precision=ns.precision,
            forward=STACKED_ARMS[arm],
        )
        for row in rows:
            stacked_row = {kk: torch.stack([b[kk] for b in row]) for kk in row[0]}
            lrn.update(upload(stacked_row, device, False), n, m, LAM)
        stacked = [lrn.model.to_net(i) for i in range(k)]
        final[arm] = torch.cat([p.detach().reshape(-1).cpu() for p in lrn.params.values()])
        for name, ref in refs.items():
            worst = {"drift_ratio": 0.0, "q_max_abs": 0.0, "argmax_agree": 1.0, "q_scale": 0.0}
            for i in range(k):
                a = torch.cat([p.detach().reshape(-1) for p in ref[i].parameters()])
                b = torch.cat([p.detach().reshape(-1) for p in stacked[i].parameters()])
                worst["drift_ratio"] = max(
                    worst["drift_ratio"], float((a - b).norm() / (a - theta0[i]).norm())
                )
                with torch.no_grad():
                    qa, qb = ref[i](*hold_t), stacked[i](*hold_t)
                worst["q_max_abs"] = max(worst["q_max_abs"], float((qa - qb).abs().max()))
                worst["q_scale"] = max(worst["q_scale"], float(qa.abs().mean()))
                agree = float((qa.argmax(1) == qb.argmax(1)).float().mean())
                worst["argmax_agree"] = min(worst["argmax_agree"], agree)
            res[f"{arm}_vs_{name}"] = worst
    if "stacked" in final and "grouped" in final:  # vmap vs hand-grouped: same math
        d = final["stacked"] - final["grouped"]
        res["stacked_vs_grouped_params"] = {
            "bitwise_equal": bool(torch.equal(final["stacked"], final["grouped"])),
            "max_abs": float(d.abs().max()),
        }
    return res


def gflop_per_seed_update(ns) -> float:
    """FLOPs of one ``single_update`` at the bench's row counts (CPU, FlopCounterMode)."""
    from torch.utils.flop_counter import FlopCounterMode

    nscal = NSCALS[ns.obs_version]
    online, target = make_nets(2, nscal)
    opt = torch.optim.Adam(online.parameters(), lr=lm.LR)
    rng = np.random.default_rng(0)
    b = sl.to_device(sl.synthetic_batch(rng, ns.agent_rows, ns.anchor_rows, nscal), "cpu")
    with FlopCounterMode(display=False) as fc:
        sl.single_update(online, target, opt, b, ns.agent_rows, ns.anchor_rows, LAM, MARGIN, {})
    return round(fc.get_total_flops() / 1e9, 2)


# ----------------------------------------------------------------------------- main bench
def bench(ns) -> Dict[str, Any]:
    gflop = gflop_per_seed_update(ns)
    out: Dict[str, Any] = {"gflop_per_seed_update": gflop, "cells": [], "numerics": []}
    ks = [int(x) for x in ns.ks.split(",")]
    arms = ns.arms.split(",")
    unknown = set(arms) - set(STACKED_ARMS) - {"sequential", "procs"}
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")
    for dev in ns.devices.split(","):
        device = torch.device(dev)
        torch.set_num_threads(ns.threads)
        cpu = device.type == "cpu"
        calls = ns.calls_cpu if cpu else ns.calls
        rounds = ns.rounds_cpu if cpu else ns.rounds
        warm = ns.warm_cpu if cpu else ns.warm
        pin = device.type == "cuda"
        built = {}
        nscal = NSCALS[ns.obs_version]
        for k in ks:
            for arm in arms:
                if arm == "procs" or (arm == "sequential" and ns.precision != "fp32"):
                    continue
                shape = (k, device, ns.agent_rows, ns.anchor_rows, nscal, ns.adam, ns.precision)
                if arm in STACKED_ARMS:
                    a = StackedArm(*shape, pin, forward=STACKED_ARMS[arm])
                else:
                    a = SequentialArm(*shape, pin)
                time_calls(a, warm, device)
                built[(arm, k)] = a
        times: Dict[Any, List[float]] = {key: [] for key in built}
        names = list(built)
        groups = {}
        if "procs" in arms:
            for k in [int(x) for x in (ns.procs_ks or ns.ks).split(",")]:
                groups[k] = ProcsGroup(k, dev, ns, calls, warm, rounds)
                times[("procs", k)] = []
        for r in range(rounds):
            order = names[r % len(names) :] + names[: r % len(names)]
            for key in order:
                times[key].append(time_calls(built[key], calls, device))
            for k, grp in groups.items():
                times[("procs", k)].append(grp.round())
        for grp in groups.values():
            grp.close()
        del built
        per_k = {}
        for (arm, k), ts in times.items():
            med = statistics.median(ts)
            sups = k * calls / med
            cell = {
                "device": dev,
                "arm": arm,
                "K": k,
                "precision": ns.precision,
                "adam": ns.adam,
                "calls_per_round": calls,
                "rounds": rounds,
                "round_seconds": [round(t, 4) for t in ts],
                "ms_per_call_median": round(1000 * med / calls, 2),
                "ms_per_seed_update": round(1000 * med / (calls * k), 2),
                "seed_updates_per_s": round(sups, 3),
                "transitions_per_s": round(sups * lm.UPDATE_EVERY, 1),
                "achieved_tflops": round(sups * gflop / 1000, 3),
            }
            per_k[(arm, k)] = cell
            out["cells"].append(cell)
        base1 = per_k.get(("sequential", 1)) or per_k.get(("stacked", 1))
        for (arm, k), cell in per_k.items():
            seq = per_k.get(("sequential", k))
            if seq is not None and arm != "sequential":
                cell["speedup_vs_sequential_same_K"] = round(
                    cell["seed_updates_per_s"] / seq["seed_updates_per_s"], 3
                )
            if base1 is not None:
                cell["throughput_vs_one_seed"] = round(
                    cell["seed_updates_per_s"] / base1["seed_updates_per_s"], 3
                )
            print(
                json.dumps({k_: v for k_, v in cell.items() if k_ != "round_seconds"}), flush=True
            )
        num_arms = [a for a in arms if a in STACKED_ARMS]
        if ns.numerics and num_arms:
            for k in [x for x in ks if x > 1][:1] or ks[:1]:
                row = {"device": dev, "K": k, **numerics(k, device, ns, num_arms)}
                out["numerics"].append(row)
                print(json.dumps(row), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devices", default="mps,cpu")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument(
        "--arms", default="stacked,grouped,sequential", help="stacked,grouped,sequential,procs"
    )
    ap.add_argument("--procs-ks", default=None, help="K values for the procs arm (default --ks)")
    ap.add_argument("--agent-rows", type=int, default=lm.AGENT_ROWS)
    ap.add_argument("--anchor-rows", type=int, default=lm.ANCHOR_ROWS)
    ap.add_argument("--calls", type=int, default=10, help="calls per round (GPU)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--calls-cpu", type=int, default=1)
    ap.add_argument("--rounds-cpu", type=int, default=2)
    ap.add_argument("--warm-cpu", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="torch threads (trainer learner: 1)")
    ap.add_argument("--adam", choices=sl.ADAM_IMPLS, default="fused", help="fused = T1")
    ap.add_argument("--precision", choices=sl.PRECISIONS, default="fp32")
    ap.add_argument("--obs-version", choices=sorted(NSCALS), default="bv8")
    ap.add_argument("--no-numerics", dest="numerics", action="store_false")
    ap.add_argument(
        "--note", default="NOISY: measured while other training jobs shared the CPU and the MPS GPU"
    )
    ap.add_argument("--out", type=Path, default=None)
    ns = ap.parse_args(argv)
    meta = {
        "schema": "perf-multiseed-bench/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loadavg_start": os.getloadavg(),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "rows_per_seed_update": {"agent": ns.agent_rows, "anchor": ns.anchor_rows},
        "obs_version": ns.obs_version,
        "note": ns.note,
    }
    res = bench(ns)
    res = {**meta, **res, "loadavg_end": os.getloadavg()}
    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
