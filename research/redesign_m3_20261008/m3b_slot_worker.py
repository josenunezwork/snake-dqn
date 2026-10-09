#!/usr/bin/env python3
"""Extra M3-B probe worker under the shared CPU slot locks (compute policy: max 3 slots).

Takes all three slot locks (``cpu-slot-{3,1,2}.lock``, non-blocking; aborts if any is busy):
two are held on behalf of the already-running unlocked ``m3b_slope.py probe`` driver (2
processes), one is used here. Phase 1 (v8, the gate's probes): run jobs from the END of the
driver's job list, skipping any job already written or already started by the driver (its
log's "started" lines); the two meet in the middle (at worst one identical, deterministic
duplicate row). Phase 2 (descriptive, after every v8 job is written): the no-veto probes
with up to 3 processes under the same 3 locks. Guard (AC / lid / thermal) before each job.

  ./venv/bin/python research/redesign_m3_20261008/m3b_slot_worker.py \\
      --runs <out> --probes <dir> --driver-log <dir>/probe_driver.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m3_20261008 import m3b_slope as ms  # noqa: E402

LOCK_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909")


def driver_started(log: Path) -> set:
    out = set()
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "started" in row:
                point, ckpt, mix = row["started"]
                out.add((ckpt, mix))
    return out


def run(point, ckpt: Path, mix: str, out: Path, veto: str) -> subprocess.Popen:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", SNAKE_DQN_DEVICE="cpu")
    argv = [
        "nice", "-n", "10", sys.executable,
        str(REPO / "research/redesign_m2_20261008/dev_probe.py"),
        "--student", str(ckpt), "--arm", "student", "--mix", mix,
        "--worlds", str(ms.WORLDS), "--round-index", str(ms.ROUND_INDEX),
        "--veto", veto, "--out", str(out),
    ]  # fmt: skip
    print(json.dumps({"started": [point, str(ckpt), mix, veto]}), flush=True)
    return subprocess.Popen(argv, cwd=REPO, env=env)


def wait_guard() -> None:
    from research.redesign_m2_20261008.gen_data import guard_ok

    while True:
        problems = guard_ok()
        if not problems:
            return
        print(json.dumps({"waiting": problems}), flush=True)
        time.sleep(120)


def main() -> int:
    from research.apex_safety_20260926 import dev_screen

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--driver-log", type=Path, required=True)
    ap.add_argument("--driver-pid", type=int, required=True)
    args = ap.parse_args()
    handles = dev_screen.acquire_cpu_slots(LOCK_ROOT, 3, timeout=30.0, pool=3)
    if len(handles) < 3:
        dev_screen.release_cpu_slots(handles)
        raise SystemExit("refusing: not all 3 CPU slots are free")
    print(json.dumps({"slots": 3, "lock_root": str(LOCK_ROOT)}), flush=True)
    try:
        # Phase 1: v8 gate probes, 1 process here (+ the driver's 2).
        for point, ckpt, mix, out in reversed(ms.jobs(args.runs, args.probes)):
            if ms.done(out, mix) or (str(ckpt), mix) in driver_started(args.driver_log):
                continue
            wait_guard()
            run(point, ckpt, mix, out, "v8").wait()
        while not all(ms.done(out, mix) for _, _, mix, out in ms.jobs(args.runs, args.probes)):
            time.sleep(30)
        print(json.dumps({"phase": "v8 complete"}), flush=True)
        # The driver's queue was fixed at its start and it does not re-check written jobs,
        # so it redoes this worker's jobs (identical, deterministic rows). Wait for it to
        # exit before phase 2 so at most 3 probe processes ever run.
        import psutil

        while psutil.pid_exists(args.driver_pid):
            time.sleep(30)
        # Phase 2: descriptive no-veto probes, up to 3 processes.
        todo = []
        for point, ckpt, mix, out in ms.jobs(args.runs, args.probes):
            nov = out.with_name(out.name.replace("probe_", "probe_noveto_"))
            if not ms.done(nov, mix):
                todo.append((point, ckpt, mix, nov))
        running = []
        while todo or running:
            running = [p for p in running if p.poll() is None]
            if todo and len(running) < 3:
                wait_guard()
                running.append(run(*todo.pop(0), "none"))
                continue
            time.sleep(5)
        print(json.dumps({"phase": "no-veto complete"}), flush=True)
    finally:
        dev_screen.release_cpu_slots(handles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
