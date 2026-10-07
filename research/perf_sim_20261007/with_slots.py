"""Run a command while holding shared CPU slot locks (compute policy), AC + lid checked.

Usage: ``./venv/bin/python research/perf_sim_20261007/with_slots.py --slots 2 -- CMD ...``

Uses ``dev_screen.acquire_cpu_slots`` with the dev pool (slot 3 first, then 1, 2), so strict
slots are taken only when the dev slot is busy. Every child gets ``OMP_NUM_THREADS=1`` and
``SNAKE_DQN_DEVICE=cpu``. Refuses on battery or with the lid closed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.apex_safety_20260926 import dev_screen  # noqa: E402


def lid_open() -> bool:
    out = subprocess.run(
        ["ioreg", "-r", "-k", "AppleClamshellState"], capture_output=True, text=True
    ).stdout
    return '"AppleClamshellState" = No' in out


def on_ac() -> bool:
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "AC Power" in out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not (on_ac() and lid_open()):
        print("refusing: need AC power and lid open", file=sys.stderr)
        return 2
    handles = dev_screen.acquire_cpu_slots(
        dev_screen.DEFAULT_SLOT_LOCK_ROOT, args.slots, args.timeout, pool=3
    )
    print(f"holding {[h.name for h in handles]}", file=sys.stderr, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", SNAKE_DQN_DEVICE="cpu", MKL_NUM_THREADS="1")
    try:
        return subprocess.call(cmd, env=env)
    finally:
        dev_screen.release_cpu_slots(handles)


if __name__ == "__main__":
    raise SystemExit(main())
