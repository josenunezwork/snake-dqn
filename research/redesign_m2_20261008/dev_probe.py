#!/usr/bin/env python3
"""Development probe (NOT the pre-registered check): student vs frp3-s12, both no veto.

Same engine and profile as ``ni_check.py`` but on DISTILLATION-namespace worlds reserved
for probing (``ni_spec.distill_seeds(..., round_index=50)``; no data round uses index 50)
so the pre-registered NI worlds stay untouched. Used to decide whether a student is ready
for the check; never reported as the check's result.

  nice -n 10 ./venv/bin/python research/redesign_m2_20261008/dev_probe.py \\
      --student S.pth --mix scripted --worlds 8 --arm student --out probe.jsonl
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m2_20261008 import ni_spec  # noqa: E402
from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

PROBE_ROUND_INDEX = 50


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", default=None)
    ap.add_argument("--arm", choices=("student", "baseline"), required=True)
    ap.add_argument("--mix", choices=ni_spec.MIXES, required=True)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch

    from src.simd_env import eval_engine as ee

    torch.set_num_threads(1)
    ctx = gid._context(1)
    seeds = ni_spec.distill_seeds(args.worlds, round_index=PROBE_ROUND_INDEX)
    assert not set(seeds) & set(ni_spec.ni_seeds())
    rows = {
        int(r["world_seed"]): [ctx["lookup"][s["member_sha256"]] for s in r["slots"]]
        for r in ctx["ds"]._design_rows(seeds)
        if r["mix"] == args.mix
    }
    kwargs = {"hero_ego2s": args.student} if args.arm == "student" else {}
    t0 = time.perf_counter()
    recs = ee.run_simd_eval(
        ctx["lookup"][ni_spec.FRP3_S12[1]],
        rows[seeds[0]],
        ni_spec.HORIZON,
        seeds,
        profile=ctx["profile"],
        opponent_specs_by_world=rows,
        mix_id=args.mix,
        vector61=True,
        sim_engine="grid",
        **kwargs,
    )
    row = {
        "arm": args.arm,
        "student": args.student,
        "mix": args.mix,
        "wall": time.perf_counter() - t0,
        "mass": {int(r["seed"]): r["mass_integral"] for r in recs},
        "deaths": {int(r["seed"]): r["deaths"] for r in recs},
        "causes": {int(r["seed"]): r.get("death_cause") for r in recs},
    }
    with args.out.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    m = list(row["mass"].values())
    print(json.dumps({"arm": args.arm, "mix": args.mix, "mean": sum(m) / len(m)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
