#!/usr/bin/env python3
"""Operating characteristics of the M4 Phase R rule (3 seeds; governance condition 7).

The sequential Phase R amendment requires a study whose design departs from the simulated
families (here: 3 seeds instead of 5, clause (4) = 3 of 3) to re-run ``simulate.py`` for its
own rule before ratification. This runs the method's own simulator
(``research.sequential_phase_r.simulate``: variance pool, data model, vectorised evaluator,
FRP-v4 scenarios plus +80 / +100) on ``rule_spec_m4.primary_rule()`` (the secondary rule
differs only in the hero label), for the fixed design, the sequential design and the
sequential design with NO_GO(50) (the pre-registered one). Reference: the 5-seed v4 family
in ``research/sequential_phase_r/oc_results_20261007.json``.

  nice -n 10 ./venv/bin/python research/redesign_m4_20261013/oc_m4.py --reps 10000 \\
      --out research/redesign_m4_20261013/results/oc_m4.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m4_20261013 import rule_spec_m4 as rs  # noqa: E402
from research.sequential_phase_r import simulate as sim  # noqa: E402
from src.evaluation.sequential_phase_r import make_plan  # noqa: E402

EXTRA_EFFECTS = (80.0, 100.0)


def scenarios() -> List[Dict[str, Any]]:
    rows = sim.scenarios("v4")
    for tau2 in (0.0, 500.0):
        rows += [{"family": "m4", "effect": e, "tau2": tau2} for e in EXTRA_EFFECTS]
    return rows


def run(reps: int, seed: int) -> Dict[str, Any]:
    pool = sim.load_pool()
    rule = rs.primary_rule()
    base_rule = dict(rule, no_go=None)
    base = make_plan(rs.N_WORLDS, base_rule)
    with_no_go = make_plan(rs.N_WORLDS, rule)
    designs = [
        sim.fixed_plan(base.rule, rs.N_WORLDS),
        sim.sequential_design(base, "sequential"),
        sim.sequential_design(with_no_go, f"sequential+NO_GO({rs.NO_GO_G:g})"),
    ]
    labels = [
        rule["statuses"]["go"],
        *[o["status"] for o in rule["final_outcomes"]],
        rule["statuses"]["kill"],
        "NO_GO_EARLY",
        rule["statuses"]["partial"],
    ]
    rows = []
    for index, scenario in enumerate(scenarios()):
        started = time.time()
        rng = np.random.default_rng([seed, index, 4])
        parts: Dict[str, list] = {d["name"]: [] for d in designs}
        for start in range(0, reps, sim.CHUNK):
            data = sim.draw(
                pool, base.rule, scenario, rs.N_WORLDS, min(sim.CHUNK, reps - start), rng
            )
            for design in designs:  # common random numbers across designs
                parts[design["name"]].append(sim.evaluate(design, data))
        merged = {
            name: {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}
            for name, ps in parts.items()
        }
        fixed = merged["fixed"]
        for design in designs:
            result = merged[design["name"]]
            paired = {}
            for label in (rule["statuses"]["go"], rule["statuses"]["kill"]):
                diff = (result["status"] == label).astype(float) - (
                    fixed["status"] == label
                ).astype(float)
                paired[label] = {
                    "diff": float(diff.mean()),
                    "paired_se": float(diff.std(ddof=1) / math.sqrt(len(diff))),
                }
            rows.append(
                {
                    "scenario": dict(scenario),
                    "design": design["name"],
                    **sim.summarise(result, labels),
                    "vs_fixed": paired,
                }
            )
        print(
            f"{scenario.get('label', '')} effect {scenario['effect']:+g} tau2 "
            f"{scenario['tau2']:g}: {time.time() - started:.1f}s",
            flush=True,
        )
    return {
        "schema": "sequential-phase-r-oc/v1",
        "study": "redesign-m4 (3 seeds, clause 4 = 3 of 3)",
        "rule_sha256": rs.PRIMARY_RULE_SHA256,
        "plan_sha256": with_no_go.sha256(),
        "reps": reps,
        "seed": seed,
        "n_worlds": rs.N_WORLDS,
        "no_go_threshold": rs.NO_GO_G,
        "pool": sim.POOL.name,
        "rows": rows,
    }


def table(result: Dict[str, Any]) -> str:
    """Markdown: P(GO_R), P(KILL), P(NO_GO), expected fraction, per scenario (NO_GO design)."""
    lines = [
        "| scenario | tau^2 | GO_R | UNGATEABLE | KILL_V4 | NO_GO | E[fraction] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        if not r["design"].startswith("sequential+NO_GO"):
            continue
        s = r["scenario"]
        name = s.get("label") or f"effect {s['effect']:+g}"
        lines.append(
            f"| {name} | {s['tau2']:g} | {r['GO_R']['p']:.3f} | {r['GO_R_UNGATEABLE']['p']:.3f} "
            f"| {r['KILL_V4']['p']:.3f} | {r['NO_GO_EARLY']['p']:.3f} "
            f"| {r['expected_fraction']['mean']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists (create-only)")
    result = run(args.reps, args.seed)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(table(result))


if __name__ == "__main__":
    main()
