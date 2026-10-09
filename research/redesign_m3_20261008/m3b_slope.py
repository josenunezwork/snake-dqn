#!/usr/bin/env python3
"""M3-B learning-slope gate (PREREGISTRATION_M3.md section 6): probes + the decision.

Probes (student + v8, the pinned ``dev_probe.py``, round index 52, 16 worlds per mix = the
same 48 worlds at every point) at 0 (the M2b student, shared by all seeds), 5, 10, 15 and
20M (``final.pth``) for seeds 0-4. Per seed: the OLS slope (per 1M transitions) of the
world-paired, mix-stratified means. Gate: proceed to Phase R iff the one-sided 90% t lower
bound across the 5 seed slopes (df 4) is > 0.

  nice -n 10 ./venv/bin/python research/redesign_m3_20261008/m3b_slope.py probe \\
      --runs <run-dir>/attemptNN/out --probes <dir> --procs 2
  ./venv/bin/python research/redesign_m3_20261008/m3b_slope.py report --probes <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEEDS = (0, 1, 2, 3, 4)
POINTS = (("0M", 0.0), ("5M", 5.0), ("10M", 10.0), ("15M", 15.0), ("20M", 20.0))
CKPT = {"5M": "ckpt_5000k.pth", "10M": "ckpt_10000k.pth", "15M": "ckpt_15000k.pth"}
MIXES = ("frozen", "scripted", "mixed")
WORLDS = 16
ROUND_INDEX = 52
T90_DF4 = 1.5332062740589443  # scipy.stats.t.ppf(0.90, 4)
STUDENT0 = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008/student_m2b_final.pth"
)


def ckpt_path(runs: Path, seed: int, point: str) -> Path:
    if point == "0M":
        return STUDENT0
    d = runs / f"m3b_seed{seed}"
    return d / ("final.pth" if point == "20M" else CKPT[point])


def jobs(runs: Path, probes: Path) -> List[Tuple[str, Path, str, Path]]:
    out = []
    for mix in MIXES:
        out.append(("0M", STUDENT0, mix, probes / "probe_0M.jsonl"))
    for seed in SEEDS:
        for point, _ in POINTS[1:]:
            for mix in MIXES:
                out.append(
                    (
                        point,
                        ckpt_path(runs, seed, point),
                        mix,
                        probes / f"probe_seed{seed}_{point}.jsonl",
                    )
                )
    return out


def done(path: Path, mix: str) -> bool:
    if not path.exists():
        return False
    return any(json.loads(line).get("mix") == mix for line in path.read_text().splitlines() if line)


def probe(args) -> int:
    from research.redesign_m2_20261008.gen_data import guard_ok

    args.probes.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs(args.runs, args.probes) if not done(j[3], j[2])]
    missing = sorted({str(j[1]) for j in todo if not j[1].exists()})
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", SNAKE_DQN_DEVICE="cpu")
    running: List[subprocess.Popen] = []
    while todo or running:
        running = [p for p in running if p.poll() is None]
        if todo and len(running) < args.procs:
            problems = guard_ok()
            if problems:
                print(json.dumps({"waiting": problems}), flush=True)
                time.sleep(120)
                continue
            point, ckpt, mix, out = todo.pop(0)
            argv = [
                "nice", "-n", "10", sys.executable,
                str(REPO / "research/redesign_m2_20261008/dev_probe.py"),
                "--student", str(ckpt), "--arm", "student", "--mix", mix,
                "--worlds", str(WORLDS), "--round-index", str(ROUND_INDEX),
                "--veto", "v8", "--out", str(out),
            ]  # fmt: skip
            running.append(subprocess.Popen(argv, cwd=REPO, env=env))
            print(json.dumps({"started": [point, str(ckpt), mix], "left": len(todo)}), flush=True)
            continue
        time.sleep(5)
    return 0


def load(path: Path) -> Dict[str, Dict[int, float]]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("veto") == "v8":
                out[r["mix"]] = {int(k): float(v) for k, v in r["mass"].items()}
    return out


def stratified_mean(probe_rows: Dict[str, Dict[int, float]], worlds) -> float:
    """Equal weight per mix over the SAME worlds at every point (world-paired)."""
    return sum(sum(probe_rows[m][w] for w in worlds[m]) / len(worlds[m]) for m in MIXES) / len(
        MIXES
    )


def ols_slope(xs, ys) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def gate(slopes: List[float]) -> Dict[str, float]:
    n = len(slopes)
    mean = sum(slopes) / n
    sd = math.sqrt(sum((s - mean) ** 2 for s in slopes) / (n - 1))
    lb = mean - T90_DF4 * sd / math.sqrt(n)
    return {"mean_slope": mean, "sd": sd, "se": sd / math.sqrt(n), "t90_lower": lb, "pass": lb > 0}


def report(args) -> int:
    base = load(args.probes / "probe_0M.jsonl")
    worlds = {m: sorted(base[m]) for m in MIXES}
    per_seed = {}
    for seed in SEEDS:
        means = {"0M": stratified_mean(base, worlds)}
        for point, _ in POINTS[1:]:
            rows = load(args.probes / f"probe_seed{seed}_{point}.jsonl")
            assert all(sorted(rows[m]) == worlds[m] for m in MIXES), "worlds differ"
            means[point] = stratified_mean(rows, worlds)
        xs = [x for _, x in POINTS]
        per_seed[seed] = {
            "means": means,
            "slope_per_M": ols_slope(xs, [means[p] for p, _ in POINTS]),
        }
    out = {
        "statistic": "OLS slope per 1M of world-paired mix-stratified means (student + v8)",
        "points": [p for p, _ in POINTS],
        "worlds_per_mix": WORLDS,
        "round_index": ROUND_INDEX,
        "per_seed": per_seed,
        "gate": gate([per_seed[s]["slope_per_M"] for s in SEEDS]),
        "decision": None,
    }
    out["decision"] = "PROCEED to Phase R" if out["gate"]["pass"] else "STOP (slope gate fails)"
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("probe", "report"))
    ap.add_argument("--runs", type=Path, help="dir holding m3b_seed{0..4}/")
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.procs > 2:
        raise SystemExit("compute rule: at most 2 processes")
    return probe(args) if args.cmd == "probe" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
