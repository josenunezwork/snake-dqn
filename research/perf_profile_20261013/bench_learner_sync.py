#!/usr/bin/env python3
"""Does ``_masked_argmax``'s host sync defeat the ``--learner-fast`` pipeline? (perf-profile)

The real-run profile (``REPORT.md``) shows the T1 learner blocking in two places: the
designed one (``_finalize`` reads the finite-loss flag) and ``_masked_argmax``, whose
``m[~m.any(dim=1), :3] = True`` is a boolean-mask ``index_put`` - the device must finish the
upload and the three forwards before the host learns how many rows to write. While the host
waits there it cannot enqueue the loss / backward / clip, and it cannot start sampling the
next batch (the overlap T1 was built for).

:func:`masked_argmax_nosync` builds the same mask with element-wise ops only
(``m | (~m.any(1, keepdim=True) & first3)``): the same booleans, so the same argmax, bit for
bit (``tests/test_profile_m3c.py`` pins it, including all-false rows). This script times the
T1 learner (``bench_learner.py``'s synthetic bv8 world) with the shipped function and with the
sync-free one patched in, in interleaved rounds. Bench only: the trainer is unchanged.

Usage (noisy unless the GPU is otherwise idle)::

    ./venv/bin/python research/perf_profile_20261013/bench_learner_sync.py --devices mps \\
        --out research/perf_profile_20261013/results/learner_sync_mps_noisy.json
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from research.perf_learner_20261013 import bench_learner as bl  # noqa: E402
from research.redesign_m3c_20261008 import learner_m3c as lm  # noqa: E402

_FIRST3 = {}


def masked_argmax_nosync(torch_mod, q, m):
    """``learner_m3c._masked_argmax`` without the boolean-mask index (no host sync)."""
    key = (m.device.type, m.device.index)
    first3 = _FIRST3.get(key)
    if first3 is None:
        first3 = torch_mod.tensor([True, True, True, False, False, False], device=m.device)
        _FIRST3[key] = first3
    m = m | (~m.any(dim=1, keepdim=True) & first3)
    return torch_mod.where(m, q, torch_mod.full_like(q, -1e9)).argmax(dim=1)


VARIANTS = {"shipped": lm._masked_argmax, "nosync": masked_argmax_nosync}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devices", default="mps")
    ap.add_argument("--configs", default="t1,t1_k2", help="bench_learner config names")
    ap.add_argument("--ring-rows", type=int, default=40_000)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ns = ap.parse_args(argv)
    nscal = 60
    multi = bl.build_rings(nscal, ns.ring_rows, 0)
    id_hi = ns.ring_rows * bl.N_ACTORS
    net0 = bl.init_net(nscal)
    cells = []
    for device in ns.devices.split(","):
        torch.set_num_threads(1)
        learners, feeds = {}, {}
        for cfg in ns.configs.split(","):
            for var in VARIANTS:
                world = bl.build_world("bv8", nscal, id_hi, 0)
                key = (cfg, var)
                learners[key] = bl.make_learner(
                    bl.CONFIGS[cfg], device, "bv8", nscal, net0, multi, world, 0
                )
                feeds[key] = np.random.default_rng(5)
                lm._masked_argmax = VARIANTS[var]
                bl.run_updates(learners[key], ns.warm, feeds[key], id_hi, device)
        keys = list(learners)
        times = {k: [] for k in keys}
        for rnd in range(ns.rounds):
            for key in keys[rnd % len(keys) :] + keys[: rnd % len(keys)]:
                lm._masked_argmax = VARIANTS[key[1]]
                times[key].append(
                    bl.run_updates(learners[key], ns.updates, feeds[key], id_hi, device)
                )
        lm._masked_argmax = VARIANTS["shipped"]
        for cfg in ns.configs.split(","):
            base = None
            for var in VARIANTS:
                lr = learners[(cfg, var)]
                med = statistics.median(times[(cfg, var)])
                cell = {
                    "device": device,
                    "config": cfg,
                    "masked_argmax": var,
                    "round_seconds": [round(t, 4) for t in times[(cfg, var)]],
                    "ms_per_update_median": round(1e3 * med / ns.updates, 2),
                    "transitions_per_s": round(ns.updates / med * lr.update_every, 1),
                    "halt": lr.st["halt"],
                }
                base = base or cell["transitions_per_s"]
                cell["speedup_vs_shipped"] = round(cell["transitions_per_s"] / base, 3)
                cells.append(cell)
                print(json.dumps({k: v for k, v in cell.items() if k != "round_seconds"}))
        # same math: the two variants' learners saw the same batches -> same weights?
        for cfg in ns.configs.split(","):
            a = torch.cat(
                [
                    p.detach().reshape(-1).cpu()
                    for p in learners[(cfg, "shipped")].online.parameters()
                ]
            )
            b = torch.cat(
                [
                    p.detach().reshape(-1).cpu()
                    for p in learners[(cfg, "nosync")].online.parameters()
                ]
            )
            row = {
                "device": device,
                "config": cfg,
                "weights_max_abs_diff_after_bench": float((a - b).abs().max()),
                "bitwise_equal": bool(torch.equal(a, b)),
            }
            cells.append(row)
            print(json.dumps(row))
    res = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "loadavg": os.getloadavg(),
        "argv": sys.argv[1:],
        "note": "NOISY: other trainings shared the CPU and the MPS GPU",
        "cells": cells,
    }
    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
