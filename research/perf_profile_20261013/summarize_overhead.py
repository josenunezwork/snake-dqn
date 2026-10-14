#!/usr/bin/env python3
"""Summarize ``bench_overhead.sh`` output: ms/frame unprofiled vs pyinstrument vs cProfile."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    out = Path(sys.argv[1])
    cells = {"off": [], "pyinstrument": [], "cprofile": []}
    for f in sorted(out.glob("off_r*.json")):
        cells["off"].append(json.loads(f.read_text())["rows"][0])
    for mode in ("pyinstrument", "cprofile"):
        for d in sorted(out.glob(f"{mode}_r*")):
            cells[mode].append(json.loads((d / "actor0.meta.json").read_text()))
    base = None
    rows = {}
    for mode, rs in cells.items():
        if not rs:
            continue
        ms = statistics.median(r["ms_per_frame"] for r in rs)
        keys = rs[0]["timing_ms_per_frame"]
        split = {
            k: round(statistics.median(r["timing_ms_per_frame"][k] for r in rs), 2) for k in keys
        }
        base = base or ms
        rows[mode] = {"ms_per_frame_median": round(ms, 2), "vs_off": round(ms / base, 3)}
        rows[mode].update(split)
        rows[mode]["runs"] = [round(r["ms_per_frame"], 2) for r in rs]
        print(mode, json.dumps(rows[mode]))
    (out / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
