#!/usr/bin/env python3
"""Run ONE command while holding ONE shared CPU slot lock (compute policy: max 3 slots).

Waits for a free slot among ``cpu-slot-{3,1,2}.lock`` (never creates lock files), checks the
AC / lid / thermal guard before starting, runs the command under ``nice -n 10`` with one
thread, then releases the slot. Exit code = the command's.

  ./venv/bin/python research/redesign_m3_20261008/slotrun.py -- CMD ARGS...
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOCK_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909")


def main() -> int:
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_m2_20261008.gen_data import guard_ok

    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        raise SystemExit("usage: slotrun.py -- CMD ARGS...")
    while True:
        try:
            handles = dev_screen.acquire_cpu_slots(LOCK_ROOT, 1, timeout=60.0, pool=3)
            break
        except TimeoutError:
            continue
    try:
        while True:
            problems = guard_ok()
            if not problems:
                break
            print(json.dumps({"waiting": problems}), flush=True)
            time.sleep(120)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        return subprocess.call(["nice", "-n", "10", *argv], cwd=REPO, env=env)
    finally:
        dev_screen.release_cpu_slots(handles)


if __name__ == "__main__":
    raise SystemExit(main())
