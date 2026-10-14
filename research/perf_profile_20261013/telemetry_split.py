#!/usr/bin/env python3
"""Throughput split of a (running or finished) multi-actor ``train_m3c`` run, read-only.

From ``log.jsonl`` / ``telemetry.jsonl`` deltas per ``--step`` transitions: the actors'
timed frame cost (``env_s`` per frame of 128 worlds, and its ``featurizer`` = ego raster,
``forward`` = hero transfers + forward, ``opponents``, ``v8`` split; ``untimed`` = the rest of
``env_s``: sim step, vector61 featurization, ring / n-step), the learner's busy share
(``learn_s / wall``) and the transition rate. A learner busy share near 1 means the run is
learner-bound: faster actors would not raise the rate.

Usage::

    ./venv/bin/python research/perf_profile_20261013/telemetry_split.py RUN_DIR [RUN_DIR ...] \\
        [--step 2000000] [--worlds 128]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

KEYS = ("featurizer_s", "forward_s", "opponents_s", "v8_s")


def split(run: Path, step: int, worlds: int) -> List[dict]:
    tel = [json.loads(x) for x in (run / "telemetry.jsonl").read_text().splitlines() if x]
    logs = {}
    for x in (run / "log.jsonl").read_text().splitlines():
        if x:
            r = json.loads(x)
            logs[r["transitions"]] = r
    out, prev = [], None
    for r in tel:
        t = r["transitions"]
        if prev is None:
            prev = r
            continue
        if t // step == prev["transitions"] // step and r is not tel[-1]:
            continue
        lg, lp = logs.get(t), logs.get(prev["transitions"])
        if lg is None or lp is None or not all(k in r and k in prev for k in KEYS):
            prev = r
            continue
        frames = (t - prev["transitions"]) / float(worlds)
        d = {k[:-2]: (r[k] - prev[k]) * 1e3 / frames for k in KEYS}
        env = (lg["env_s"] - lp["env_s"]) * 1e3 / frames
        wall = lg["wall"] - lp["wall"]
        out.append(
            {
                "from": prev["transitions"],
                "to": t,
                "env_ms_per_frame": round(env, 2),
                **{f"{k}_ms": round(v, 2) for k, v in d.items()},
                "untimed_ms": round(env - sum(d.values()), 2),
                "learn_over_wall": round((lg["learn_s"] - lp["learn_s"]) / wall, 3),
                "rate": round((t - prev["transitions"]) / wall, 1),
            }
        )
        prev = r
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--step", type=int, default=2_000_000)
    ap.add_argument("--worlds", type=int, default=128)
    a = ap.parse_args(argv)
    for run in a.runs:
        print(f"## {run}")
        for row in split(run, a.step, a.worlds):
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
