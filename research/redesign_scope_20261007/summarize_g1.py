"""Tabulate results/bench_g1_20261007.jsonl (one row per bench_g1.py cell)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent / "results" / "bench_g1_20261007.jsonl"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PATH
    head = "| opponents | scenario | E | featurizer | anchors | sim jit | hero fwd | hero steps/s"
    print(head + " (median, rounds) | us/world-frame: sim / hero obs / hero fwd / opponents |")
    print("|" + "---|" * 9)
    for line in path.read_text().splitlines():
        d = json.loads(line)
        rates = [round(r["hero_steps_per_s"]) for r in d["rounds"]]
        parts = d["rounds"][0]["us_per_world_frame"]
        opp = sum(v for k, v in parts.items() if k.startswith("opp_"))
        print(
            f"| {d['opponents']} | {d['scenario']} | {d['worlds']} | {d['featurizer']} | "
            f"{d['anchors']} | {d.get('sim_jit', 'on')} | {d['hero_forward']} | "
            f"{round(d['median_hero_steps_per_s']):,} {rates} | "
            f"{parts.get('sim_step', 0):.1f} / {parts.get('hero_obs', 0):.1f} / "
            f"{parts.get('hero_forward', 0):.1f} / {opp:.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
