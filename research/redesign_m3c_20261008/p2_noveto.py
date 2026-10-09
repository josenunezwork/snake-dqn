#!/usr/bin/env python3
"""M3-C prerequisite P2-NV: grid == live records for the student WITHOUT a veto.

M3's P2 found the no-veto student differs grid vs live on 6/9 worlds: the SIMD
``Ego2sSimdPolicy`` acted on the resolved mask (last resort: the legal mask, boost included)
while the live ``AISnake`` uses its advisory mask with a normal-only fallback. M3-C adds
``ego2s_mask="advisory"`` to the SIMD hero (the live selection rule). This wrapper runs the
pinned ``student_record_parity.py`` unchanged except that its grid side passes
``ego2s_mask="advisory"`` for the no-veto arm (``--arm none`` only). Same worlds
(``redesign-m3-p2/v1``), same comparison, H5000. Development prerequisite, not evidence.

  nice -n 10 ./venv/bin/python research/redesign_m3c_20261008/p2_noveto.py \\
      --student S.pth --arm none --worlds 3 --out OUT.json
  nice -n 10 ./venv/bin/python research/redesign_m3c_20261008/p2_noveto.py h10000 \\
      --student S.pth --out OUT.json     # 1 world per mix at H10000 (as M3's p2_h10000.py)
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m3_20261008 import student_record_parity as p2  # noqa: E402
from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

p2.SCHEMA = "redesign-m3c-student-record-parity-noveto/v1"


def run_sim(rows, ctx, student: str, arm: str) -> Dict[int, Any]:
    """``p2.run_sim`` for ``arm == "none"`` with ``ego2s_mask="advisory"`` (nothing else)."""
    from src.simd_env import eval_engine as ee

    if arm != "none":
        raise SystemExit("p2_noveto.py is for --arm none (use the pinned tool for v8)")
    lookup = ctx["lookup"]
    seeds = [int(r["world_seed"]) for r in rows]
    rosters = {int(r["world_seed"]): [lookup[s["member_sha256"]] for s in r["slots"]] for r in rows}
    recs = ee.run_simd_eval(
        lookup[gid.FRP3_S12[1]],
        rosters[seeds[0]],
        5000,
        seeds,
        profile=ctx["profile"],
        opponent_specs_by_world=rosters,
        mix_id=rows[0]["mix"],
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        hero_ego2s=student,
        ego2s_forward="rowwise",
        ego2s_mask="advisory",
    )
    return dict(zip(seeds, recs))


p2.run_sim = run_sim


def main_h10000(argv) -> int:
    """M3's ``p2_h10000.py`` for the no-veto arm: its live side with arm "none", the grid side
    with ``ego2s_mask="advisory"`` (nothing else differs)."""
    import argparse
    import dataclasses
    import json
    from datetime import datetime, timezone

    import torch

    from research.apex_safety_20260926 import dev_screen
    from research.redesign_m3_20261008 import p2_h10000 as h10
    from src.simd_env import eval_engine as ee

    ap = argparse.ArgumentParser(description="P2-NV at H10000")
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"refusing: {args.out} exists", file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    ctx = gid._context(1)
    ctx["profile"] = dataclasses.replace(ctx["profile"], scored_horizon=h10.HORIZON)
    if ctx["profile"].digest != h10.PROFILE_DIGEST_H10000:
        raise SystemExit("H10000 profile digest differs from the pinned digest")
    seeds = dev_screen.screen_seeds(1, p2.DOMAIN, "worlds")
    rows_all = list(dev_screen._design_rows(seeds))
    results = []
    original = dev_screen.HORIZON
    lookup = ctx["lookup"]
    for mix in p2.MIXES:
        row = [r for r in rows_all if r["mix"] == mix][0]
        w = int(row["world_seed"])
        roster = [lookup[s["member_sha256"]] for s in row["slots"]]
        rec = ee.run_simd_eval(
            lookup[gid.FRP3_S12[1]],
            roster,
            h10.HORIZON,
            [w],
            profile=ctx["profile"],
            opponent_specs_by_world={w: roster},
            mix_id=mix,
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            hero_ego2s=args.student,
            ego2s_forward="rowwise",
            ego2s_mask="advisory",
        )[0]
        dev_screen.HORIZON = h10.HORIZON
        try:
            live = p2.run_live(row, ctx, args.student, "none")
        finally:
            dev_screen.HORIZON = original
        out = p2.compare(live, rec)
        out.update({"mix": mix, "world_seed": w, "horizon": h10.HORIZON})
        results.append(out)
        print(json.dumps({k: out[k] for k in ("mix", "world_seed", "identical", "mass_integral")}))
    summary = {
        "schema_version": "redesign-m3c-student-record-parity-noveto-h10000/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "student": {"path": args.student, "sha256": dev_screen.sha256_file(Path(args.student))},
        "arm": "none",
        "horizon": h10.HORIZON,
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
    if sys.argv[1:2] == ["h10000"]:
        raise SystemExit(main_h10000(sys.argv[2:]))
    raise SystemExit(p2.main())
