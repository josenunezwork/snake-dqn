#!/usr/bin/env python3
"""Mac-side dev probes that start as each checkpoint lands (M3-C; development only).

Watches ``landed.jsonl`` (written by ``gpu_run_v2.py`` after each sha-checked pull) and/or
local run directories (``--watch-dir``, rows of each run's ``manifest.jsonl``), and runs
``research/redesign_m2_20261008/dev_probe.py`` for every scheduled checkpoint x veto x mix
not yet probed: the no-veto arm first (the M3-C primary metric), then v8. Each probe is one
single-thread process under ``nice -n 10`` after the AC / lid / thermal guard; with
``--slots`` it also holds one shared CPU slot lock (compute policy). Results:
``<out>/<run>__<ckpt>__<veto>.jsonl`` (one row per mix, dev_probe's format). Idempotent:
re-running skips (run, ckpt, veto, mix) rows already written.

  ./venv/bin/python research/redesign_m3c_20261008/probe_watch.py \\
      --landed RUN_DIR/landed.jsonl --out RESULTS --worlds 16 --vetoes none --slots --procs 2
  (M3-C runs two watchers: intermediates ``--schedule 5000k,10000k,15000k --worlds 16
  --vetoes none`` and finals ``--schedule final --worlds 32 --vetoes none,v8``; they write
  different files because the checkpoint stem is part of the name.)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEV_PROBE = REPO / "research/redesign_m2_20261008/dev_probe.py"
LOCK_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909")
MIXES = ("frozen", "scripted", "mixed")
MAX_FAILURES = 2  # a probe that failed twice is logged and skipped

Job = Tuple[str, str, str, str, str]  # run, ckpt stem, path, veto, mix


def ckpt_stem(file: str) -> str:
    return Path(file).stem


def scheduled(file: str, schedule: Sequence[str]) -> bool:
    stem = ckpt_stem(file)
    return stem in schedule or stem.replace("ckpt_", "") in schedule


def landed_rows(landed: Optional[Path], watch_dirs: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if landed and landed.exists():
        for line in landed.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows.append({"run": r["run"], "file": r["file"], "path": r["path"]})
    for d in watch_dirs:
        for man in sorted(Path(d).glob("*/manifest.jsonl")):
            for line in man.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                rows.append(
                    {"run": man.parent.name, "file": r["file"], "path": str(man.parent / r["file"])}
                )
    return rows


def result_path(out: Path, run: str, stem: str, veto: str) -> Path:
    return out / f"{run}__{stem}__{veto}.jsonl"


def done_mixes(path: Path) -> set:
    got = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                got.add(json.loads(line)["mix"])
            except (ValueError, KeyError):
                continue
    return got


def pending_jobs(
    rows: Sequence[Dict[str, str]],
    out: Path,
    schedule: Sequence[str],
    vetoes: Sequence[str],
    running: set,
) -> List[Job]:
    """Every scheduled (run, ckpt, veto, mix) not yet written or running; no-veto first."""
    seen = set()
    jobs: List[Job] = []
    for veto in vetoes:
        for r in rows:
            if not scheduled(r["file"], schedule):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", r["run"]):
                continue
            stem = ckpt_stem(r["file"])
            have = done_mixes(result_path(out, r["run"], stem, veto))
            for mix in MIXES:
                key = (r["run"], stem, veto, mix)
                if mix in have or key in running or key in seen:
                    continue
                seen.add(key)
                jobs.append((r["run"], stem, r["path"], veto, mix))
    return jobs


def probe_argv(job: Job, out: Path, worlds: int, round_index: int, slots: bool) -> List[str]:
    run, stem, path, veto, mix = job
    argv = [
        sys.executable,
        str(DEV_PROBE),
        "--student",
        path,
        "--arm",
        "student",
        "--mix",
        mix,
        "--worlds",
        str(worlds),
        "--round-index",
        str(round_index),
        "--veto",
        veto,
        "--ego2s-forward",
        "rowwise",  # Phase R's forward mode
        "--ego2s-mask",
        "advisory",  # used by dev_probe for the no-veto hero only (Phase R's rule)
        "--out",
        str(result_path(out, run, stem, veto)),
    ]
    if slots:
        return [sys.executable, str(REPO / "research/redesign_m3_20261008/slotrun.py"), "--"] + argv
    return ["nice", "-n", "10"] + argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--landed", type=Path, default=None)
    ap.add_argument("--watch-dir", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--schedule", default="5000k,10000k,15000k")
    ap.add_argument("--vetoes", default="none,v8")
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--round-index", type=int, default=52)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--slots", action="store_true", help="hold a shared CPU slot lock per probe")
    ap.add_argument("--until", type=Path, default=None, help="exit when this file exists and idle")
    ap.add_argument("--once", action="store_true", help="one pass, then wait for the jobs, exit")
    args = ap.parse_args(argv)
    from research.redesign_m2_20261008.gen_data import guard_ok

    args.out.mkdir(parents=True, exist_ok=True)
    schedule = [s.strip() for s in args.schedule.split(",") if s.strip()]
    vetoes = [v.strip() for v in args.vetoes.split(",") if v.strip()]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", SNAKE_DQN_DEVICE="cpu")
    live: Dict[tuple, subprocess.Popen] = {}
    failures: Dict[tuple, int] = {}
    log = args.out / "probe_watch.log"
    while True:
        for key, proc in list(live.items()):
            if proc.poll() is not None:
                if proc.returncode != 0:
                    failures[key] = failures.get(key, 0) + 1
                with log.open("a") as fh:
                    fh.write(
                        json.dumps({"done": key, "rc": proc.returncode, "t": time.time()}) + "\n"
                    )
                del live[key]
        rows = landed_rows(args.landed, args.watch_dir)
        given_up = {k for k, n in failures.items() if n >= MAX_FAILURES}
        jobs = pending_jobs(rows, args.out, schedule, vetoes, set(live) | given_up)
        while jobs and len(live) < args.procs:
            problems = [] if args.slots else guard_ok()  # slotrun.py checks it itself
            if problems:
                with log.open("a") as fh:
                    fh.write(json.dumps({"waiting": problems, "t": time.time()}) + "\n")
                break
            job = jobs.pop(0)
            key = job[:2] + job[3:]
            argv_ = probe_argv(job, args.out, args.worlds, args.round_index, args.slots)
            live[key] = subprocess.Popen(argv_, cwd=REPO, env=env, stdout=subprocess.DEVNULL)
            with log.open("a") as fh:
                fh.write(json.dumps({"started": key, "t": time.time()}) + "\n")
        idle = not live and not jobs
        if idle and (args.once or (args.until and args.until.exists())):
            return 0
        time.sleep(5 if args.once else 60)


if __name__ == "__main__":
    raise SystemExit(main())
