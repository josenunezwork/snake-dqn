#!/usr/bin/env python3
"""Summarize M3-B no-veto dev probes (round 52, 16 worlds/mix) vs the 0M student (M3-A file)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
M3A = HERE.parents[0] / "redesign_m3_20261008/results/m3a/m3a_probes_0M.jsonl"
MIXES = ("frozen", "scripted", "mixed")


def load(path, veto="none"):
    d = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r.get("veto", veto) == veto:
            for w, m in r["mass"].items():
                d[(r["mix"], w)] = (m, r["deaths"][w])
    return d


def main() -> int:
    base = load(M3A)
    out = {
        "0M": {
            "mean": sum(v[0] for v in base.values()) / len(base),
            "deaths": sum(v[1] for v in base.values()),
            "n": len(base),
        }
    }
    for ck in ("10000k", "20000k"):
        seeds = {}
        for s in range(5):
            f = HERE / f"results/diag/noveto_seed{s}_{ck}.jsonl"
            if not f.exists():
                continue
            d = load(f)
            if set(d) != set(base):
                continue
            diffs = [d[k][0] - base[k][0] for k in base]
            m = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
            seeds[s] = {
                "mean": sum(v[0] for v in d.values()) / len(d),
                "delta": m,
                "se": sd / math.sqrt(len(diffs)),
                "deaths": sum(v[1] for v in d.values()),
                "per_mix": {mx: sum(d[k][0] for k in d if k[0] == mx) / 16 for mx in MIXES},
            }
        if seeds:
            ds = [v["delta"] for v in seeds.values()]
            mm = sum(ds) / len(ds)
            se = (
                math.sqrt(sum((x - mm) ** 2 for x in ds) / (len(ds) - 1)) / math.sqrt(len(ds))
                if len(ds) > 1
                else None
            )
            out[ck] = {"per_seed": seeds, "pooled_delta": mm, "seed_se": se}
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
