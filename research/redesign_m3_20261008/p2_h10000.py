#!/usr/bin/env python3
"""P2 at H10000 (PREREGISTRATION_M3.md section 3): 1 world per mix, grid vs live, student + v8.

The pinned ``student_record_parity.py`` (unchanged) runs H5000 only. This wrapper reuses its
``run_live`` (with ``dev_screen.HORIZON`` set to 10000 for the call) and its ``compare``; the
grid side mirrors its ``run_sim`` exactly except for the horizon argument. Worlds: the pinned
tool's namespace and selection (``screen_seeds(1, DOMAIN, "worlds")``), i.e. the first H5000
world of each mix, replayed at the longer horizon.

  nice -n 10 ./venv/bin/python research/redesign_m3_20261008/p2_h10000.py \\
      --student S.pth --out OUT.json
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m3_20261008 import student_record_parity as p2  # noqa: E402
from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

HORIZON = 10_000
# promotion-v2-watch-rect with scored_horizon 10000 (LH-1 / FRP-v5 pinned digest)
PROFILE_DIGEST_H10000 = "c882e07fb7179c01feed5380aeae228113975442bb10ab8840bc58a0011f5b15"
SCHEMA = "redesign-m3-student-record-parity-h10000/v1"


def run_sim(rows, ctx, student: str) -> Dict[int, Any]:
    """``p2.run_sim`` with arm v8 and the horizon set to HORIZON (nothing else differs)."""
    from src.simd_env import eval_engine as ee

    lookup = ctx["lookup"]
    seeds = [int(r["world_seed"]) for r in rows]
    rosters = {int(r["world_seed"]): [lookup[s["member_sha256"]] for s in r["slots"]] for r in rows}
    recs = ee.run_simd_eval(
        lookup[gid.FRP3_S12[1]],
        rosters[seeds[0]],
        HORIZON,
        seeds,
        profile=ctx["profile"],
        opponent_specs_by_world=rosters,
        mix_id=rows[0]["mix"],
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        hero_ego2s=student,
        ego2s_forward="rowwise",
        hero_safety_veto="v8",
        hero_safety_veto_lambda=8.0,
    )
    return dict(zip(seeds, recs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        print(f"refusing: {args.out} exists", file=sys.stderr)
        return 2
    import torch

    torch.set_num_threads(1)
    from research.apex_safety_20260926 import dev_screen

    import dataclasses

    ctx = gid._context(1)
    ctx["profile"] = dataclasses.replace(ctx["profile"], scored_horizon=HORIZON)
    if ctx["profile"].digest != PROFILE_DIGEST_H10000:
        raise SystemExit("H10000 profile digest differs from the pinned c882e07f... digest")
    seeds = dev_screen.screen_seeds(1, p2.DOMAIN, "worlds")
    rows_all = list(dev_screen._design_rows(seeds))
    assert {r["mix"] for r in rows_all} >= set(p2.MIXES), "one world per mix expected"
    results: List[Dict[str, Any]] = []
    original = dev_screen.HORIZON
    for mix in p2.MIXES:
        row = [r for r in rows_all if r["mix"] == mix][0]
        sim = run_sim([row], ctx, args.student)
        dev_screen.HORIZON = HORIZON
        try:
            live = p2.run_live(row, ctx, args.student, "v8")
        finally:
            dev_screen.HORIZON = original
        out = p2.compare(live, sim[int(row["world_seed"])])
        out.update({"mix": mix, "world_seed": int(row["world_seed"]), "horizon": HORIZON})
        results.append(out)
        print(json.dumps({k: out[k] for k in ("mix", "world_seed", "identical", "mass_integral")}))
    summary = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "student": {"path": args.student, "sha256": dev_screen.sha256_file(Path(args.student))},
        "arm": "v8",
        "horizon": HORIZON,
        "profile_digest": ctx["profile"].digest,
        "compared": len(results),
        "identical": sum(r["identical"] for r in results),
        "pass": bool(results) and all(r["identical"] for r in results),
        "worlds": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(summary, stream, indent=1, sort_keys=True)
    print(json.dumps({k: summary[k] for k in ("arm", "horizon", "compared", "identical", "pass")}))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
