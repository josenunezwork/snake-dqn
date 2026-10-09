#!/usr/bin/env python3
"""M3 prerequisite P2: ego2s student H5000 records, SIMD grid engine == live engine.

M1/M2b showed the ego2s OBSERVATION is identical live vs sim with the teacher acting. This
checks whole RECORDS with the student acting, the setting Phase R would use: the pinned
deployment config, ``promotion-v2-watch-rect`` at H5000, strict balanced rosters, the student
as the hero

* live: ``tournament_eval.rollout`` with the student attached through ``_attach_agent``
  (``Ego2sServingPolicy(forward="rowwise")``) and, for the ``v8`` arm, the served v8 veto
  (lambda 8) installed by the dev-screen hook, exactly as the perf-sim / strict harnesses do;
* sim: ``run_simd_eval(sim_engine="grid", vector61=True, vector61_forward="rowwise",
  hero_ego2s=student, ego2s_forward="rowwise")`` (the v8 arm through ``Ego2sV8Policy``).

PASS needs every world's record (minus the SIMD-only provenance keys) and, for v8, the veto
diagnostics (minus wall-clock keys) to be identical. Worlds come from their own namespace
``redesign-m3-p2/v1``. Development check (an M3 prerequisite), not gate evidence.

  nice -n 10 ./venv/bin/python research/redesign_m3_20261008/student_record_parity.py \\
      --student S.pth --arm v8 --worlds 3 --out OUT.json
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

SCHEMA = "redesign-m3-student-record-parity/v1"
DOMAIN = "redesign-m3-p2/v1"
SIMD_ONLY = (
    "world_runtime_spec",
    "world_runtime_spec_digest",
    "vector61_policy",
    "veto_diagnostics",
    "ego2s_hero",
)
MIXES = ("frozen", "scripted", "mixed")


def run_live(row, ctx, student: str, arm: str) -> Dict[str, Any]:
    from research.apex_safety_20260926 import dev_screen
    from src.evaluation.safety_veto_v8 import install_space_and_head_veto
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts.tournament_eval import rollout

    lookup = ctx["lookup"]
    opponents = [lookup[slot["member_sha256"]] for slot in row["slots"]]
    install = (lambda h: install_space_and_head_veto(h, 8.0)) if arm == "v8" else None
    t0 = time.perf_counter()
    with dev_screen.hero_veto_installer(install) as installed:
        record = rollout(
            ("checkpoint", student),
            opponents,
            dev_screen.HORIZON,
            int(row["world_seed"]),
            profile=ctx["profile"],
            world_identity=_expected_world_identity(row),
            mix_id=row["mix"],
            hero_safety_veto=arm == "v8",
        )
    diag = installed[0].diagnostics_record() if installed else None
    return {"record": record, "veto_diagnostics": diag, "wall": time.perf_counter() - t0}


def run_sim(rows, ctx, student: str, arm: str) -> Dict[int, Any]:
    from src.simd_env import eval_engine as ee

    lookup = ctx["lookup"]
    seeds = [int(r["world_seed"]) for r in rows]
    rosters = {int(r["world_seed"]): [lookup[s["member_sha256"]] for s in r["slots"]] for r in rows}
    kwargs = {"hero_safety_veto": "v8", "hero_safety_veto_lambda": 8.0} if arm == "v8" else {}
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
        **kwargs,
    )
    return dict(zip(seeds, recs))


def compare(live: Dict[str, Any], sim_record: Dict[str, Any]) -> Dict[str, Any]:
    sim = gid.normalize(sim_record)
    sim_diag = gid.strip_timing(sim.get("veto_diagnostics"))
    view = {k: v for k, v in sim.items() if k not in SIMD_ONLY}
    rec = gid.normalize(live["record"])
    diffs = gid.diff_paths(rec, view)
    live_diag = gid.strip_timing(gid.normalize(live["veto_diagnostics"]))
    diag_diffs = gid.diff_paths(live_diag, sim_diag) if live_diag is not None else []
    return {
        "identical": not diffs and not diag_diffs,
        "record_differences": [{"path": p, "live": a, "sim": b} for p, a, b in diffs[:15]],
        "diagnostics_differences": [
            {"path": p, "live": a, "sim": b} for p, a, b in diag_diffs[:15]
        ],
        "mass_integral": rec.get("mass_integral"),
        "deaths": rec.get("deaths"),
        "live_wall": live["wall"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--arm", choices=("v8", "none"), required=True)
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        print(f"refusing: {args.out} exists", file=sys.stderr)
        return 2
    import torch

    torch.set_num_threads(1)
    from research.apex_safety_20260926 import dev_screen

    ctx = gid._context(1)
    seeds = dev_screen.screen_seeds(args.worlds, DOMAIN, "worlds")
    rows_all = [r for r in dev_screen._design_rows(seeds)]
    results: List[Dict[str, Any]] = []
    sim_wall = 0.0
    for mix in MIXES:
        rows = [r for r in rows_all if r["mix"] == mix]
        t0 = time.perf_counter()
        sim = run_sim(rows, ctx, args.student, args.arm)
        sim_wall += time.perf_counter() - t0
        for row in rows:
            live = run_live(row, ctx, args.student, args.arm)
            out = compare(live, sim[int(row["world_seed"])])
            out.update({"mix": mix, "world_seed": int(row["world_seed"])})
            results.append(out)
            print(
                json.dumps({k: out[k] for k in ("mix", "world_seed", "identical", "mass_integral")})
            )
    summary = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "student": {"path": args.student, "sha256": dev_screen.sha256_file(Path(args.student))},
        "arm": args.arm,
        "compared": len(results),
        "identical": sum(r["identical"] for r in results),
        "pass": bool(results) and all(r["identical"] for r in results),
        "sim_wall_seconds": sim_wall,
        "worlds": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(summary, stream, indent=1, sort_keys=True)
    print(json.dumps({k: summary[k] for k in ("arm", "compared", "identical", "pass")}))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
