#!/usr/bin/env python3
"""Summarize the M3-A smoke: rate, learning-curve diagnostics, probe deltas, slope diagnostic.

Reads ``<run>/log.jsonl`` + ``<run>/summary.json`` and the probe files
``m3a_probes_{0M,2.5M,5M}.jsonl`` (dev_probe rows: arm, veto, mix, per-world mass). For each
veto setting it reports per-checkpoint means, the world-paired delta vs 0M (mean, SE), and
the single-seed OLS slope of the world-paired mix-stratified means on transitions (per 1M)
with its world-bootstrap SE -- a DIAGNOSTIC of the pre-registered M3-B slope statistic,
not the gate (which needs 5 seeds).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

POINTS = (("0M", 0.0), ("2.5M", 2.5), ("5M", 5.0))
MIXES = ("frozen", "scripted", "mixed")


def load_probe(path: Path, veto: str):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = {}
    for r in rows:
        if r.get("veto", "none") == veto:
            out[r["mix"]] = {int(k): float(v) for k, v in r["mass"].items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True, help="dir with m3a_probes_*.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    log = [json.loads(line) for line in (args.run / "log.jsonl").read_text().splitlines()]
    summary = json.loads((args.run / "summary.json").read_text())
    keys = (
        "td",
        "anchor",
        "qmax",
        "greedy_boost_rate",
        "boost_rate",
        "deaths_per_1k",
        "holdout_agreement",
    )
    curve = [{k: row.get(k) for k in ("transitions",) + keys} for row in log]
    report = {
        "rate_transitions_per_s": summary["rate"],
        "wall_seconds": summary["wall_seconds"],
        "transitions": summary["transitions"],
        "updates": summary["updates"],
        "halt": summary["halt"],
        "flags": summary["flags"],
        "holdout_agreement_final": summary.get("holdout_agreement_final"),
        "curve_first_last": [curve[0], curve[-1]] if curve else [],
        "probes": {},
    }
    rng = np.random.default_rng(0)
    for veto in ("v8", "none"):
        per = {}
        for name, _ in POINTS:
            p = args.probes / f"m3a_probes_{name}.jsonl"
            if p.exists():
                per[name] = load_probe(p, veto)
        if len(per) < 2:
            continue
        worlds = sorted(per["0M"]["frozen"].keys())
        # (worlds, points) matrix of mix-stratified means
        xs = [x for name, x in POINTS if name in per]
        mat = np.array(
            [
                [np.mean([per[name][m][w] for m in MIXES]) for name, _ in POINTS if name in per]
                for w in worlds
            ]
        )
        means = mat.mean(axis=0)
        deltas = mat - mat[:, :1]
        x = np.array(xs)

        def slope(m):
            y = m.mean(axis=0)
            return float(np.polyfit(x, y, 1)[0])

        s = slope(mat)
        boots = [slope(mat[rng.integers(0, len(worlds), len(worlds))]) for _ in range(2000)]
        by_mix = {
            name: {m: float(np.mean(list(per[name][m].values()))) for m in MIXES} for name in per
        }
        report["probes"][veto] = {
            "means": {name: float(v) for name, v in zip([n for n, _ in POINTS if n in per], means)},
            "by_mix": by_mix,
            "delta_vs_0M": {
                name: {
                    "mean": float(deltas[:, i].mean()),
                    "se": float(deltas[:, i].std(ddof=1) / math.sqrt(len(worlds))),
                }
                for i, name in enumerate([n for n, _ in POINTS if n in per])
                if i > 0
            },
            "slope_per_1M": s,
            "slope_bootstrap_se": float(np.std(boots, ddof=1)),
            "worlds": len(worlds),
        }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
