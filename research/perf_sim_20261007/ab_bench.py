"""Paired A/B throughput bench: two source trees play the same gate-world episodes at once.

Each tree runs ``episode_harness.py --mode time`` in its own process holding one shared CPU
slot (``with_slots.py``), started together so both see the same machine load (the Mac's
P/E-core scheduling and other users' processes make sequential timings noisy). Reports per
episode wall time, the A/B ratio, mass-integral equality (a cheap identity tripwire, not the
identity harness) and episodes/hour/core.

Usage::

  ./venv/bin/python research/perf_sim_20261007/ab_bench.py --a ROOT_A --b ROOT_B \
      --worlds 2 --out-dir DIR [--swap]   # --swap: run a second round with roles swapped
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run_pair(a: Path, b: Path, out_dir: Path, worlds: int, offset: int, tag: str) -> dict:
    procs = []
    outs = {}
    for name, root in (("A", a), ("B", b)):
        out = out_dir / f"{tag}-{name}.jsonl"
        outs[name] = out
        cmd = [
            PY,
            str(HERE / "with_slots.py"),
            "--slots",
            "1",
            "--",
            PY,
            str(HERE / "episode_harness.py"),
            "--root",
            str(root),
            "--mode",
            "time",
            "--out",
            str(out),
            "--worlds",
            str(worlds),
            "--world-offset",
            str(offset),
            "--namespace",
            "bench",
        ]
        log = open(out_dir / f"{tag}-{name}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"bench child failed: {codes}")
    rows = {k: [json.loads(line) for line in v.open()] for k, v in outs.items()}
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--swap", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rounds = [("r1", args.a, args.b, False)]
    if args.swap:
        rounds.append(("r2", args.b, args.a, True))
    a_wall, b_wall, ratios, same = [], [], [], True
    for tag, first, second, swapped in rounds:
        rows = run_pair(first, second, args.out_dir, args.worlds, args.offset, tag)
        ra, rb = (rows["B"], rows["A"]) if swapped else (rows["A"], rows["B"])
        for x, y in zip(ra, rb):
            a_wall.append(x["wall_seconds"])
            b_wall.append(y["wall_seconds"])
            ratios.append(x["wall_seconds"] / y["wall_seconds"])
            same &= x["mass_integral"] == y["mass_integral"]
    summary = {
        "a": str(args.a),
        "b": str(args.b),
        "episodes_per_arm": len(a_wall),
        "a_total_wall_s": round(sum(a_wall), 2),
        "b_total_wall_s": round(sum(b_wall), 2),
        "speedup_total": round(sum(a_wall) / sum(b_wall), 3),
        "speedup_episode_median": round(statistics.median(ratios), 3),
        "speedup_episode_min_max": [round(min(ratios), 3), round(max(ratios), 3)],
        "a_episodes_per_hour_per_core": round(3600 * len(a_wall) / sum(a_wall), 1),
        "b_episodes_per_hour_per_core": round(3600 * len(b_wall) / sum(b_wall), 1),
        "mass_integrals_equal": bool(same),
    }
    print(json.dumps(summary, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
