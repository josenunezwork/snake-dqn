#!/usr/bin/env python3
"""Summarize dev-probe jsonl files: mass, survival per mix (development information only).

  ./venv/bin/python research/ensemble_20261014/summarize.py FILE [FILE ...]

Survival = fraction of worlds with zero hero deaths (the probe hero is terminal, so deaths is
0 or 1). Paired deltas vs ``--ref`` (default: the M3-C incumbent file) use the shared worlds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

DEFAULT_REF = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/"
    "redesign-m3c-probes-20261008-recovery/incumbent__dev__v8.jsonl"
)
MIXES = ("frozen", "scripted", "mixed")


def load(path: Path) -> Dict[str, dict]:
    """Last row per mix (later rows of the same mix supersede earlier ones)."""
    rows: Dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["mix"]] = r
    return rows


def stats(rows: Dict[str, dict]) -> dict:
    out: dict = {"per_mix": {}}
    all_m: List[float] = []
    for mix in MIXES:
        if mix not in rows:
            continue
        r = rows[mix]
        m = [float(v) for v in r["mass"].values()]
        d = [int(v) for v in r["deaths"].values()]
        all_m += m
        out["per_mix"][mix] = {
            "n": len(m),
            "mass": sum(m) / len(m),
            "survival": sum(1 for x in d if x == 0) / len(d),
        }
    out["mass"] = sum(all_m) / len(all_m) if all_m else float("nan")
    return out


def paired(rows: Dict[str, dict], ref: Dict[str, dict]) -> dict:
    deltas: List[float] = []
    for mix in MIXES:
        if mix not in rows or mix not in ref:
            continue
        for seed, v in rows[mix]["mass"].items():
            if seed in ref[mix]["mass"]:
                deltas.append(float(v) - float(ref[mix]["mass"][seed]))
    if len(deltas) < 2:
        return {"n": len(deltas)}
    mean = sum(deltas) / len(deltas)
    sd = math.sqrt(sum((x - mean) ** 2 for x in deltas) / (len(deltas) - 1))
    se = sd / math.sqrt(len(deltas))
    return {"n": len(deltas), "delta": mean, "lo95": mean - 1.985 * se, "hi95": mean + 1.985 * se}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    ref = load(args.ref)
    table = []
    for f in args.files:
        rows = load(f)
        s = stats(rows)
        s["file"] = str(f)
        s["vs_ref"] = paired(rows, ref)
        s["keys"] = (
            {
                k: rows[next(iter(rows))].get(k)
                for k in (
                    "arm",
                    "veto",
                    "round_index",
                    "ego2s_forward",
                    "ego2s_mask",
                    "v61_backend",
                    "anchors",
                    "wall",
                )
            }
            if rows
            else {}
        )
        table.append(s)
    if args.json:
        print(json.dumps(table, indent=1))
        return 0
    for s in table:
        pm = s["per_mix"]
        surv = " ".join(f"{m[:4]}={pm[m]['survival']:.3f}" for m in MIXES if m in pm)
        mm = " ".join(f"{m[:4]}={pm[m]['mass']:.1f}" for m in MIXES if m in pm)
        d = s["vs_ref"].get("delta")
        print(
            f"{Path(s['file']).name:44s} mass={s['mass']:7.1f} "
            f"d_ref={d if d is None else round(d, 1)} | surv {surv} | {mm} | {s['keys']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
