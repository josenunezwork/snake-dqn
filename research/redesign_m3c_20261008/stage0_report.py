#!/usr/bin/env python3
"""M3-C stage 0 report (PREREGISTRATION_M3C section 4): boost screen B0 + reference re-probes.

B0 (amended before ratification, 2026-10-08): the 5 M3-B finals with the boost bias 0.035
(``results/diag/boost035_seed<s>_<veto>.jsonl``) vs as is (no veto:
``results/diag/noveto_seed<s>_20000k.jsonl``; v8: M3's ``results/m3b/probe_seed<s>_20M.jsonl``),
the same 16 round-52 worlds per mix and the same probe settings as M3-B's probes (batched
forward, resolved mask). Rule: per seed, the world-paired mix-stratified delta (boost - as
is); across the 5 seeds the one-sided 90% t lower bound (df 4). **Arm E is added iff LB > 0
for BOTH no-veto and v8.**

References (descriptive, not launch-blocking): ``<probes>/<run>__final__<veto>.jsonl`` for
``M2b_s0`` and ``M3B_s<s>`` probed with Phase R's hero modes on 32 worlds per mix, plus
``incumbent__dev__v8.jsonl``; missing files are reported as missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

from research.redesign_m3c_20261008.m3c_select import MIXES, T90, load_probe, stratified

SEEDS = (0, 1, 2, 3, 4)


def paired_delta(a, b) -> float:
    means = []
    for mix in MIXES:
        ws = sorted(set(a.get(mix, {})) & set(b.get(mix, {})))
        if not ws:
            raise SystemExit(f"no common worlds in {mix}")
        means.append(sum(a[mix][w][0] - b[mix][w][0] for w in ws) / len(ws))
    return sum(means) / len(means)


def report(probes: Path) -> Dict[str, object]:
    here = Path(__file__).resolve().parent
    diag = here / "results/diag"
    m3b = here.parent / "redesign_m3_20261008/results/m3b"
    out: Dict[str, object] = {"boost_screen": {}, "references": {}}
    for veto in ("none", "v8"):
        deltas = []
        for s in SEEDS:
            a = load_probe(diag / f"boost035_seed{s}_{veto}.jsonl")
            base = (
                diag / f"noveto_seed{s}_20000k.jsonl"
                if veto == "none"
                else m3b / f"probe_seed{s}_20M.jsonl"
            )
            deltas.append(paired_delta(a, load_probe(base)))
        m = sum(deltas) / len(deltas)
        se = math.sqrt(sum((d - m) ** 2 for d in deltas) / (len(deltas) - 1) / len(deltas))
        out["boost_screen"][veto] = {
            "per_seed": deltas,
            "mean": m,
            "se": se,
            "lb90": m - T90[4] * se,
        }
    out["add_arm_E"] = all(out["boost_screen"][v]["lb90"] > 0 for v in ("none", "v8"))
    inc = load_probe(probes / "incumbent__dev__v8.jsonl")
    worlds = {m: sorted(inc.get(m, {})) for m in MIXES}
    for run in ["M2b_s0"] + [f"M3B_s{s}" for s in SEEDS]:
        row = {}
        for veto in ("none", "v8"):
            p = load_probe(probes / f"{run}__final__{veto}.jsonl")
            if not p:
                row[veto] = None
                continue
            row[veto] = {
                "stratified": stratified(p, worlds),
                "deaths": sum(v[1] for mix in p.values() for v in mix.values()),
                "worlds": sum(len(v) for v in p.values()),
            }
        out["references"][run] = row
    out["incumbent_v8"] = {
        "stratified": stratified(inc, worlds) if inc else None,
        "deaths": sum(v[1] for mix in inc.values() for v in mix.values()),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = report(args.probes)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"add_arm_E": out["add_arm_E"], "boost": out["boost_screen"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
