#!/usr/bin/env python3
"""Opponent-forward census, timing and parity for ``train_m3c --opp-forward`` (perf-opp).

Builds real training segments (``train_m3c.build_segment``, the M5 recipe's world mix by
default: 25% frozen, 25% mixed, the rest scripted; numba vector61 featurizer), drives them
with a random legal hero and the per-group opponents (the default actor path), and on every
timed frame also runs the merged opponents on the SAME prepared world:

* census: opponent groups with live rows, distinct vector61 checkpoints, rows per group;
* time per frame: per-group vector61 calls, scripted groups, ``merged-cpu`` (one stacked
  forward), ``merged-device`` (launch = host-side enqueue, total = with the sync);
* parity vs per-group, per row: Q bitwise equal, max |dQ|, raw-argmax and selected-action
  (masked argmax, what the opponent plays) flips.

Each worlds value runs in its own subprocess (one CPU process at a time, plus the device).
Numbers taken while other jobs run are NOISY; re-run on a quiet machine / the CUDA pod::

    ./venv/bin/python research/perf_opp_20261013/bench_opp.py \\
        --worlds 64,128,256 --frames 200 --segments 2 --device auto \\
        --out research/perf_opp_20261013/results/bench_opp.json

    # parity only (many frames, no timing), e.g. on an x86 pod before trusting merged-cpu
    ./venv/bin/python research/perf_opp_20261013/bench_opp.py --parity \\
        --worlds 128 --frames 400 --segments 3

End to end (the whole actor) use ``research/perf_actor_20261013/bench_actor.py
--opp-forwards per-group,merged-cpu,merged-device``.
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pick_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "none"


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run_cell(a) -> Dict[str, Any]:
    import torch

    from research.redesign_m3c_20261008 import train_m3c as T
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env.vector61_merged import MergedVector61Opponents

    torch.set_num_threads(1)
    device = pick_device(a.device)
    args = T.build_parser().parse_args(
        [
            "--seed",
            str(a.seed),
            "--init",
            "unused",
            "--demo-data",
            ".",
            "--out",
            "unused",
            "--actors",
            "1",
            "--worlds",
            str(a.worlds),
            "--frozen-frac",
            str(a.frozen_frac),
            "--mixed-frac",
            str(a.mixed_frac),
            "--v61-backend",
            a.v61_backend,
        ]
    )
    ctx = gid._context(1)
    rng = np.random.default_rng([a.seed, 99])
    modes = ["merged-cpu"] + ([] if device == "none" else ["merged-device"])
    t_keys = ["per_group_v61", "scripted", "merged-cpu", "merged-device_launch", "merged-device"]
    tsum = {k: 0.0 for k in t_keys}
    par = {
        m: {
            "rows": 0,
            "q_bitwise_rows": 0,
            "max_abs_dq": 0.0,
            "argmax_flips": 0,
            "action_flips": 0,
            # rows whose reference masked top-2 Q gap is below 1e-4 (the device |dQ| scale):
            # the rows a float difference could flip; and the smallest such gap seen
            "near_tie_rows_1e-4": 0,
            "min_masked_top2_gap": float("inf"),
        }
        for m in modes
    }
    census: Dict[str, List[float]] = {
        "groups_live": [],
        "v61_checkpoints_live": [],
        "v61_rows": [],
        "v61_rows_min_group": [],
        "v61_rows_max_group": [],
        "scripted_rows": [],
    }
    timed = 0
    for segment in range(a.segments):
        sim, runtime, policies, groups = T.build_segment(args, ctx, segment, 0.99, hero_v61=True)
        merged = {"merged-cpu": MergedVector61Opponents(policies, groups, runtime, "cpu")}
        if "merged-device" in modes:
            merged["merged-device"] = MergedVector61Opponents(policies, groups, runtime, device)
        v61_specs = [s for p in merged["merged-cpu"].parts for s in p.specs]
        scripted = {k: v for k, v in groups.items() if k not in merged["merged-cpu"].specs}
        frame = [0]

        def per_group(specs, prepared, masks, alive, actions):
            for spec_ in specs:
                cells = groups[spec_]
                live = cells[alive[cells[:, 0], cells[:, 1]]]
                if len(live):
                    mk = masks[live[:, 0], live[:, 1]]
                    actions[live[:, 0], live[:, 1]] = policies[spec_].actions(mk, prepared, live)

        def choose(prepared):
            nonlocal timed
            masks = prepared.get_resolved_action_mask()
            alive = prepared.get_alive()
            actions = np.ones((sim.E, prepared.S), dtype=np.int64)
            runtime.prepare(prepared)
            hero = np.argwhere(alive[:, :1])
            if len(hero):
                m = masks[hero[:, 0], 0].copy()
                m[~m.any(axis=1), :3] = True
                actions[hero[:, 0], 0] = T.sample_legal(rng, m)
            on = frame[0] >= a.warm
            frame[0] += 1
            # the default actor path drives the sim
            order = ["ref", *modes] if frame[0] % 2 else [*modes[::-1], "ref"]
            outs: Dict[str, Any] = {}
            for name in order:
                if name == "ref":
                    t0 = time.perf_counter()
                    per_group(v61_specs, prepared, masks, alive, actions)
                    t1 = time.perf_counter()
                    per_group(list(scripted), prepared, masks, alive, actions)
                    t2 = time.perf_counter()
                    if on and not a.parity:
                        tsum["per_group_v61"] += t1 - t0
                        tsum["scripted"] += t2 - t1
                    continue
                out = np.full_like(actions, -1)
                _sync(device)
                t0 = time.perf_counter()
                pend = merged[name].launch(alive, with_q=a.parity or on)
                t1 = time.perf_counter()
                merged[name].finish(pend, out)
                t2 = time.perf_counter()
                if on and not a.parity:
                    if name == "merged-device":
                        tsum["merged-device_launch"] += t1 - t0
                    tsum[name] += t2 - t0
                outs[name] = (out, pend)
            if not on:
                return actions
            timed += 1
            live_v61 = [groups[s][alive[groups[s][:, 0], groups[s][:, 1]]] for s in v61_specs]
            counts = [len(x) for x in live_v61]
            n_scr = sum(int(alive[c[:, 0], c[:, 1]].sum()) for c in scripted.values())
            census["groups_live"].append(
                sum(c > 0 for c in counts)
                + sum(bool(alive[c[:, 0], c[:, 1]].any()) for c in scripted.values())
            )
            census["v61_checkpoints_live"].append(sum(c > 0 for c in counts))
            census["v61_rows"].append(sum(counts))
            census["v61_rows_min_group"].append(min(counts) if counts else 0)
            census["v61_rows_max_group"].append(max(counts) if counts else 0)
            census["scripted_rows"].append(n_scr)
            # reference Q (per-group forwards on the same rows, part order)
            ref_q = []
            for part in merged["merged-cpu"].parts:
                for s in part.specs:
                    c = groups[s]
                    live = c[alive[c[:, 0], c[:, 1]]]
                    if len(live):
                        st, _ = runtime.selection(live)
                        ref_q.append(policies[s].q_values(st).cpu())
            if not ref_q:
                return actions
            ref_q = torch.cat(ref_q)
            for name, (out, pend) in outs.items():
                q = torch.cat([p["q"].float().cpu() for p in pend])
                rows = np.concatenate([p["rows"] for p in pend])
                mask = np.concatenate([p["mask"] for p in pend])
                p = par[name]
                p["rows"] += len(rows)
                p["q_bitwise_rows"] += int(
                    (q.numpy().view(np.uint32) == ref_q.numpy().view(np.uint32)).all(axis=1).sum()
                )
                p["max_abs_dq"] = max(p["max_abs_dq"], float((q - ref_q).abs().max()))
                p["argmax_flips"] += int((q.argmax(1) != ref_q.argmax(1)).sum())
                p["action_flips"] += int(
                    (out[rows[:, 0], rows[:, 1]] != actions[rows[:, 0], rows[:, 1]]).sum()
                )
                ref_m = torch.where(
                    torch.from_numpy(mask), ref_q, torch.full_like(ref_q, -float("inf"))
                )
                top2 = ref_m.topk(2, dim=1).values
                gap = (top2[:, 0] - top2[:, 1]).nan_to_num(nan=float("inf"))
                p["near_tie_rows_1e-4"] += int((gap < 1e-4).sum())
                p["min_masked_top2_gap"] = min(p["min_masked_top2_gap"], float(gap.min()))
            return actions

        for _ in range(a.warm + a.frames):
            sim.step_with_policy(choose)
            runtime.observe_step(sim)
        frame[0] = 0
    row: Dict[str, Any] = {
        "worlds": a.worlds,
        "frozen_frac": a.frozen_frac,
        "mixed_frac": a.mixed_frac,
        "segments": a.segments,
        "timed_frames": timed,
        "device": device,
        "census_mean": {k: float(np.mean(v)) if v else 0.0 for k, v in census.items()},
        "census_max": {k: float(np.max(v)) if v else 0.0 for k, v in census.items()},
        "parity": {
            m: {
                **p,
                "q_bitwise_frac": p["q_bitwise_rows"] / max(p["rows"], 1),
                "action_flip_rate": p["action_flips"] / max(p["rows"], 1),
                "argmax_flip_rate": p["argmax_flips"] / max(p["rows"], 1),
            }
            for m, p in par.items()
        },
    }
    if not a.parity and timed:
        row["ms_per_frame"] = {
            k: 1e3 * v / timed for k, v in tsum.items() if v or k in ("per_group_v61",)
        }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", default="64,128,256")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--frozen-frac", type=float, default=0.25)
    ap.add_argument("--mixed-frac", type=float, default=0.25, help="M5 recipe: 0.25")
    ap.add_argument("--v61-backend", default="numba")
    ap.add_argument("--device", default="auto", help="merged-device target: auto|cpu|mps|cuda|none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parity", action="store_true", help="parity only (no timing)")
    ap.add_argument("--label", default="noisy", help="recorded with the result (quiet / noisy)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)  # one worlds value
    a = ap.parse_args()
    if a.cell:
        a.worlds = int(a.worlds)
        print("CELL " + json.dumps(run_cell(a)), flush=True)
        return 0
    rows: List[Dict[str, Any]] = []
    for w in [int(x) for x in a.worlds.split(",")]:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--cell", "--worlds", str(w)]
        for k in ("frames", "warm", "segments", "frozen_frac", "mixed_frac", "v61_backend"):
            cmd += [f"--{k.replace('_', '-')}", str(getattr(a, k))]
        cmd += ["--device", a.device, "--seed", str(a.seed)]
        if a.parity:
            cmd.append("--parity")
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("CELL ")]
        if r.returncode or not line:
            row = {"worlds": w, "error": True, "stderr": r.stderr[-2000:]}
        else:
            row = json.loads(line[-1][5:])
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "schema": "perf-opp-bench/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "machine": platform.machine(),
        "label": a.label,
        "note": "opponents only (random hero); one CPU process; noisy if the host is busy",
        "rows": rows,
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
