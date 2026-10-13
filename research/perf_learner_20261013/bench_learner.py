#!/usr/bin/env python3
"""Learner-update throughput benchmark for ``train_m3c``'s multi-actor learner (perf-learner).

Builds a synthetic multi-actor world in-process - two shared observation rings of random
uint8 planes (real ego2s shapes), a replay index, an on-policy label buffer (``bv8``: 384
agent + 128 label rows per update) or a demonstration buffer (``b``) - and times
:class:`research.redesign_m3c_20261008.learner_m3c.Learner` updates, with ``update_every``
fresh transitions ingested between updates as in ``main_multi``. Configs:

* ``baseline``  the reference update (pre-perf-learner code path)
* ``t1``        ``--learner-fast``
* ``t1_fp16`` / ``t1_bf16``  ``--learner-fast --learner-precision fp16|bf16``
* ``t1_k2`` / ``t1_k4``      ``--learner-fast --batch-scale 2|4`` (changes the training
  dynamics; reported per ingested transition, which is what bounds the trainer)
* ``t1_bf16_k2`` / ``t1_fp16_k2``, ``t1_cl`` (channels_last; bench only)

Rounds are interleaved (config order rotated each round) and the median round is reported,
because the Mac GPU is usually shared: treat single numbers as noisy. ``updates_per_s`` is
learner updates per second; ``transitions_per_s`` = updates/s x ``update_every`` (128 x K),
the ingest rate the learner can sustain.

``--drift`` instead runs a fixed batch sequence (same seed, same world, same ingestion) for
``--drift-updates`` updates per precision from the same initial weights and reports the
parameter / Q drift relative to fp32 (and the fp32 reference-vs-fast difference, i.e. the
reassociation noise floor).

Usage (re-run on a quiet machine for clean numbers)::

    ./venv/bin/python research/perf_learner_20261013/bench_learner.py \\
        --devices mps,cpu --out research/perf_learner_20261013/results/bench_learner.json
    ./venv/bin/python research/perf_learner_20261013/bench_learner.py --drift --devices mps \\
        --out research/perf_learner_20261013/results/drift.json

On a CUDA host (e.g. the pod bench session, ``research/perf_podbench_20261013/pod_bench.py``)
pass ``--devices cuda``; a CUDA device on a host without CUDA is skipped (``skipped_devices``).
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from research.redesign_m3c_20261008 import learner_m3c as lm  # noqa: E402
from research.redesign_m3c_20261008 import train_m3c as m3c  # noqa: E402
from src.model.ego2s_network import Ego2sNet  # noqa: E402

STUDENT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008/student_m2b_final.pth"
)
CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "t1": {"fast": True},
    "t1_fp16": {"fast": True, "precision": "fp16"},
    "t1_bf16": {"fast": True, "precision": "bf16"},
    "t1_k2": {"fast": True, "batch_scale": 2},
    "t1_k4": {"fast": True, "batch_scale": 4},
    "t1_fp16_k2": {"fast": True, "precision": "fp16", "batch_scale": 2},
    "t1_bf16_k2": {"fast": True, "precision": "bf16", "batch_scale": 2},
    "t1_cl": {"fast": True, "channels_last": True},
}
DEFAULT_CONFIGS = "baseline,t1,t1_fp16,t1_bf16,t1_k2,t1_k4"
NOISY_NOTE = "NOISY: measured while other training jobs shared the CPU and the MPS GPU"
N_ACTORS = 2


class _Buf:
    def __init__(self, n: int) -> None:
        self.buf = bytearray(n)


def build_rings(nscal: int, rows: int, seed: int) -> m3c.MultiRing:
    r = np.random.default_rng(seed)
    rings = []
    for _ in range(N_ACTORS):
        ring = m3c.SharedObsRing(_Buf(m3c.ring_nbytes(rows, nscal)), rows, 1024, nscal)
        a = ring.arrays
        # sparse-ish planes (like rasters), generated in chunks to bound the temporaries
        for lo in range(0, rows, 4096):
            hi = min(rows, lo + 4096)
            for k in ("local", "global"):
                shape = (hi - lo,) + a[k].shape[1:]
                a[k][lo:hi] = np.where(
                    r.random(shape, dtype=np.float32) < 0.15,
                    r.integers(1, 256, shape, dtype=np.uint8),
                    0,
                )
        a["scalars"][:] = r.standard_normal(a["scalars"].shape, dtype=np.float32)
        a["mask"][:] = r.random(a["mask"].shape) < 0.8
        ring._count[0] = rows
        rings.append(ring)
    return m3c.MultiRing(rings)


def ingest(replay: m3c.Replay, r: np.random.Generator, n: int, id_hi: int) -> None:
    s = r.integers(0, id_hi, n)
    nxt = np.where(r.random(n) < 0.05, -1, r.integers(0, id_hi, n))
    R = r.standard_normal(n) * 0.3
    replay.add_many(s, r.integers(0, 6, n), R, np.full(n, 0.97**5), nxt)


def build_world(mode: str, nscal: int, id_hi: int, seed: int):
    r = np.random.default_rng([seed, 1])
    replay = m3c.Replay(200_000)
    ingest(replay, r, 200_000, id_hi)
    labels = demos = None
    if mode == "bv8":
        labels = m3c.LabelBuffer(50_000)
        labels.add(r.integers(0, id_hi, 50_000), r.integers(0, 6, 50_000))
    else:
        import queue

        size = 20_000
        demos = object.__new__(m3c.DemoStream)
        demos.buf = {
            "local": r.integers(0, 256, (size,) + m3c.LOCAL, dtype=np.uint8),
            "global": r.integers(0, 256, (size,) + m3c.GLOBAL, dtype=np.uint8),
            "scalars": r.standard_normal((size, 30)).astype(np.float32),
            "q_teacher": r.standard_normal((size, 6)).astype(np.float32),
            "a_v8": r.integers(0, 6, size),
            "mask_resolved": r.random((size, 6)) < 0.8,
        }
        demos.size = size
        demos._q = queue.Queue()
        demos._thread = None
        demos.prefetch = lambda: None
    return replay, labels, demos


def init_path(init: Optional[str] = None) -> Optional[Path]:
    """``--init`` if given (it must exist), else the M2b student if present, else None."""
    if init:
        path = Path(init)
        if not path.exists():
            raise SystemExit(f"--init {init} does not exist")
        return path
    return STUDENT if STUDENT.exists() else None


def init_net(nscal: int, init: Optional[str] = None) -> Ego2sNet:
    path = init_path(init)
    if path is not None:
        args = m3c.default_args()
        args.init = str(path)
        args.obs_version = "bv8" if nscal == 60 else "b"
        return m3c.load_init(args, "cpu")
    torch.manual_seed(0)
    return Ego2sNet(7, nscal)


def device_available(device: str) -> bool:
    """False for a CUDA device on a host without CUDA (the device is then skipped)."""
    if device.startswith("cuda"):
        return bool(torch.cuda.is_available())
    return True


def trainer_args(mode: str):
    a = m3c.default_args()
    a.actor_sync_updates = 25
    if mode == "bv8":
        a.obs_version, a.onpolicy_labels, a.label_frac, a.anchor_mode = "bv8", True, 1.0, "v8margin"
    return a


def new_st() -> Dict[str, Any]:
    return {
        "updates": 0,
        "halt": None,
        "td": [],
        "anc": [],
        "qmax": 0.0,
        "learn_s": 0.0,
        "ring_overrun_rows": 0,
        "label_rows": [],
        "anchor_updates": 0,
    }


def make_learner(cfg, device, mode, nscal, net0, multi, world, seed):
    opts = lm.LearnerOpts(**cfg)
    online = copy.deepcopy(net0).to(device).train()
    target = copy.deepcopy(net0).to(device).eval()
    opt = lm.make_optimizer(torch, online, opts, device)
    replay, labels, demos = world
    st = new_st()
    learner = lm.Learner(
        torch,
        online=online,
        target=target,
        opt=opt,
        replay=replay,
        multi=multi,
        demos=demos,
        labels=labels,
        rng=np.random.default_rng([seed, 17]),
        st=st,
        args=trainer_args(mode),
        nscal=nscal,
        margin=0.3,
        device=device,
        opts=opts,
    )
    return learner


def sync(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def run_updates(learner, n: int, feed: np.random.Generator, id_hi: int, device: str) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        ingest(learner.replay, feed, learner.update_every, id_hi)
        learner.update(0.5)
    learner.flush()
    sync(device)
    return time.perf_counter() - t0


def bench(ns) -> Dict[str, Any]:
    nscal = 60 if ns.mode == "bv8" else 30
    multi = build_rings(nscal, ns.ring_rows, 0)
    id_hi = ns.ring_rows * N_ACTORS
    net0 = init_net(nscal, ns.init)
    out: Dict[str, Any] = {"cells": []}
    for device in ns.devices.split(","):
        if not device_available(device):
            print(f"{device} unavailable: skipped", flush=True)
            out.setdefault("skipped_devices", []).append(device)
            continue
        torch.set_num_threads(ns.threads)
        names = ns.configs.split(",")
        upd = ns.updates_cpu if device == "cpu" else ns.updates
        rounds = ns.rounds_cpu if device == "cpu" else ns.rounds
        learners = {}
        feeds = {}
        for name in names:
            world = build_world(ns.mode, nscal, id_hi, 0)
            learners[name] = make_learner(
                CONFIGS[name], device, ns.mode, nscal, net0, multi, world, 0
            )
            feeds[name] = np.random.default_rng(5)
            warm = ns.warm_cpu if device == "cpu" else ns.warm
            run_updates(learners[name], warm, feeds[name], id_hi, device)  # warm-up
        times: Dict[str, List[float]] = {n: [] for n in names}
        for rnd in range(rounds):
            order = names[rnd % len(names) :] + names[: rnd % len(names)]
            for name in order:
                times[name].append(run_updates(learners[name], upd, feeds[name], id_hi, device))
        base = None
        for name in names:
            lr = learners[name]
            med = statistics.median(times[name])
            ups = upd / med
            cell = {
                "device": device,
                "config": name,
                "flags": CONFIGS[name],
                "updates_per_round": upd,
                "rounds": rounds,
                "round_seconds": [round(t, 4) for t in times[name]],
                "ms_per_update_median": round(1000 * med / upd, 2),
                "updates_per_s": round(ups, 3),
                "update_every": lr.update_every,
                "transitions_per_s": round(ups * lr.update_every, 1),
                "rows_per_update": {"agent": lr.a_rows, "anchor": lr.d_rows},
                "halt": lr.st["halt"],
            }
            if lr.scaler is not None:
                cell["grad_scale_final"] = float(lr.scaler.get_scale())
            if name == "baseline":
                base = cell["transitions_per_s"]
            if base:
                cell["speedup_vs_baseline"] = round(cell["transitions_per_s"] / base, 3)
            out["cells"].append(cell)
            print(json.dumps({k: cell[k] for k in cell if k != "round_seconds"}), flush=True)
        del learners
    return out


@torch.no_grad()
def q_eval(net, multi, device, n=512):
    ids = np.arange(n, dtype=np.int64) * 7 % (multi.rings[0].cap * N_ACTORS)
    L, G, S, _ = multi.get(ids)
    net.eval()
    q = net(
        torch.from_numpy(L).to(device),
        torch.from_numpy(G).to(device),
        torch.from_numpy(S).to(device),
    ).float()
    net.train()
    return q.cpu()


def drift(ns) -> Dict[str, Any]:
    nscal = 60 if ns.mode == "bv8" else 30
    multi = build_rings(nscal, ns.ring_rows, 0)
    id_hi = ns.ring_rows * N_ACTORS
    net0 = init_net(nscal, ns.init)
    out: Dict[str, Any] = {"drift": []}
    for device in ns.devices.split(","):
        if not device_available(device):
            print(f"{device} unavailable: skipped", flush=True)
            out.setdefault("skipped_devices", []).append(device)
            continue
        runs = {}
        variants = {
            "reference_fp32": {},
            "t1_fp32": {"fast": True},
            "t1_fp32_plain_adam": {"fast": True, "fused_adam": False},
            "t1_bf16": {"fast": True, "precision": "bf16"},
            "t1_fp16": {"fast": True, "precision": "fp16"},
        }
        theta0 = torch.cat([p.detach().reshape(-1) for p in net0.parameters()])
        for name, cfg in variants.items():
            world = build_world(ns.mode, nscal, id_hi, 0)
            lr = make_learner(cfg, device, ns.mode, nscal, net0, multi, world, 0)
            run_updates(lr, ns.drift_updates, np.random.default_rng(5), id_hi, device)
            runs[name] = {
                "theta": torch.cat([p.detach().reshape(-1).cpu() for p in lr.online.parameters()]),
                "q": q_eval(lr.online, multi, device),
                "td": np.asarray(lr.st["td"]),
                "updates": lr.st["updates"],
                "halt": lr.st["halt"],
                "grad_scale": float(lr.scaler.get_scale()) if lr.scaler is not None else None,
            }
        ref = runs["t1_fp32"]
        step = float((ref["theta"] - theta0).norm())
        for name, r in runs.items():
            dq = (r["q"] - ref["q"]).abs()
            row = {
                "device": device,
                "variant": name,
                "vs": "t1_fp32",
                "updates": r["updates"],
                "halt": r["halt"],
                "param_max_abs_diff": float((r["theta"] - ref["theta"]).abs().max()),
                "param_diff_over_update_norm": float((r["theta"] - ref["theta"]).norm()) / step,
                "q_mean_abs_diff": float(dq.mean()),
                "q_max_abs_diff": float(dq.max()),
                "q_scale_mean_abs": float(ref["q"].abs().mean()),
                "greedy_agreement": float((r["q"].argmax(1) == ref["q"].argmax(1)).float().mean()),
                "td_mean_rel_diff": float(
                    np.mean(np.abs(r["td"] - ref["td"])) / max(1e-12, np.mean(np.abs(ref["td"])))
                ),
                "grad_scale_final": r["grad_scale"],
                "bitwise_equal_to_reference_fp32": bool(
                    torch.equal(r["theta"], runs["reference_fp32"]["theta"])
                ),
            }
            out["drift"].append(row)
            print(json.dumps(row), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devices", default="mps,cpu")
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--mode", choices=("bv8", "b"), default="bv8")
    ap.add_argument("--ring-rows", type=int, default=40_000, help="rows per actor ring")
    ap.add_argument("--updates", type=int, default=30, help="updates per round (GPU)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--updates-cpu", type=int, default=2)
    ap.add_argument("--rounds-cpu", type=int, default=1)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--warm-cpu", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="torch threads (trainer: 1)")
    ap.add_argument(
        "--init",
        default=None,
        help="ego2s checkpoint to start from (default: the M2b student if present, else a "
        "random net; the forward cost is the same, drift is more realistic from the student)",
    )
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--drift-updates", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--note", default=NOISY_NOTE, help="the result's 'note' (host conditions)")
    ns = ap.parse_args(argv)
    if any(n not in CONFIGS for n in ns.configs.split(",")):
        ap.error(f"configs: {sorted(CONFIGS)}")
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "loadavg_start": os.getloadavg(),
        "argv": sys.argv[1:],
        "init": str(init_path(ns.init) or "random"),
        "mode": ns.mode,
        "note": ns.note,
    }
    res = drift(ns) if ns.drift else bench(ns)
    res = {**meta, **res, "loadavg_end": os.getloadavg()}
    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
