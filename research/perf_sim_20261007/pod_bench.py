"""One-pod CPU throughput sweep: episodes/hour vs concurrent single-thread processes.

PREPARED, NOT LAUNCHED. Answers, on one cheap pod, (1) does one episode per vCPU (i.e.
using both SMT siblings of every physical core) beat one per physical core, (2) the
per-process slowdown curve as processes are added, and (3) the measured episodes/hour per
dollar for the pod class, so cpu3c vs cpu5c can be compared on the same script.

What it runs: ``episode_harness.py --mode time`` (frp3-s12 + v8 hero, gate world, H5000,
strict rosters, its own seed namespace), one process per slot, ``OMP_NUM_THREADS=1``,
``torch.set_num_threads(1)``. For each process count P in the sweep, P processes start
together and each plays ``--episodes-per-proc`` episodes (distinct worlds, all three mixes);
the step's throughput is ``P * episodes / max(process wall)``.

It also records the platform (lscpu, model name, SMT layout, Python/torch versions), the
``exact_radius_sq`` fast-path checks on this libm (the code falls back to the original float
path when they fail, but the record should say which path ran) and every episode's mass
integral, so the run doubles as a Mac <-> pod determinism spot check
(``--reference-jsonl``: a Mac ``episode_harness --mode time`` output for the same worlds).

Setup on the pod (by the operator, not this script): a ``git archive`` of the perf-sim
commit at ``--root``, the venv (``requirements.txt``), and the five checkpoints in
``--ckpt-dir`` either by file name or as ``<sha256>.pth`` (the fan-out runner's layout).

Usage::

  SNAKE_PERF_CKPT_DIR=/r/ckpt python research/perf_sim_20261007/pod_bench.py \\
      --root /r/repo --sweep auto --episodes-per-proc 3 --price-per-vcpu-hr 0.030 \\
      --out /r/out/pod_bench.json

``--sweep auto`` = 1, physical cores, all vCPUs (and 1.5x vCPUs to show oversubscription).
Budget guide: a perf-sim H5000 episode is ~15 s on an M5 P-core; a cpu3c vCPU is ~3.6x
slower (~55 s alone, ~90 s with its SMT sibling busy). A 16-vCPU pod with 3 episodes/proc and
the auto sweep (1, 8, 16, 24 procs) is ~3 + 3 + 5 + 7 = ~18 min plus setup, about $0.15-0.20
on cpu3c-16 ($0.48/h). Owner approval is required before creating any pod.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent


def _cmd(args: List[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def platform_record() -> Dict[str, Any]:
    lscpu = _cmd(["lscpu"])
    fields = {}
    for line in lscpu.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    logical = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        affinity = logical
    threads_per_core = int(fields.get("Thread(s) per core", "1") or 1)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model_name": fields.get("Model name")
        or _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
        "logical_cpus": logical,
        "usable_cpus": affinity,
        "threads_per_core": threads_per_core,
        "physical_cores_est": max(1, affinity // max(1, threads_per_core)),
        "lscpu": fields,
        "cgroup_cpu_max": (
            Path("/sys/fs/cgroup/cpu.max").read_text().strip()
            if Path("/sys/fs/cgroup/cpu.max").exists()
            else None
        ),
    }


def fast_path_record(root: Path) -> Dict[str, Any]:
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from src.game.game_logic import exact_radius_sq\n"
        "import json, torch\n"
        "print(json.dumps({'torch': torch.__version__, "
        "'r10': exact_radius_sq(10), 'r30': exact_radius_sq(30), 'r300': exact_radius_sq(300)}))\n"
    ) % str(root)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": out.stderr[-2000:]}


def sweep_counts(spec: str, plat: Dict[str, Any]) -> List[int]:
    if spec != "auto":
        return [int(x) for x in spec.split(",") if x]
    usable = int(plat["usable_cpus"])
    phys = int(plat["physical_cores_est"])
    counts = sorted({1, phys, usable, max(usable + usable // 2, usable + 1)})
    return counts


def run_step(root: Path, procs: int, episodes: int, offset: int, out_dir: Path) -> Dict[str, Any]:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", SNAKE_DQN_DEVICE="cpu")
    running = []
    started = time.monotonic()
    for i in range(procs):
        out = out_dir / f"p{procs}-proc{i}.jsonl"
        worlds = max(1, (episodes + 2) // 3)
        cmd = [
            sys.executable,
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
            str(offset + i * worlds),
            "--namespace",
            "bench",
        ]
        log = open(out_dir / f"p{procs}-proc{i}.log", "w")
        running.append((out, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
    codes = [p.wait() for _, p in running]
    wall = time.monotonic() - started
    rows = []
    for out, _ in running:
        if out.exists():
            rows.extend(json.loads(line) for line in out.open())
    eps = len(rows)
    return {
        "procs": procs,
        "exit_codes": codes,
        "episodes": eps,
        "step_wall_s": round(wall, 2),
        "episodes_per_hour": round(3600 * eps / wall, 1) if wall > 0 else None,
        "mean_episode_wall_s": (
            round(sum(r["wall_seconds"] for r in rows) / eps, 2) if eps else None
        ),
        "records": [
            {
                "mix": r["mix"],
                "world_seed": r["world_seed"],
                "mass_integral": r["mass_integral"],
                "wall_seconds": round(r["wall_seconds"], 2),
            }
            for r in rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sweep", default="auto", help="'auto' or comma list of process counts")
    parser.add_argument("--episodes-per-proc", type=int, default=3)
    parser.add_argument("--price-per-vcpu-hr", type=float, default=None)
    parser.add_argument("--reference-jsonl", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out.parent / (args.out.stem + "-runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    plat = platform_record()
    result: Dict[str, Any] = {
        "schema": "perf-sim-pod-bench/v1",
        "platform": plat,
        "fast_path": fast_path_record(args.root),
        "steps": [],
    }
    for procs in sweep_counts(args.sweep, plat):
        # Every step replays the same worlds (offset 0) so steps differ only in load.
        step = run_step(args.root, procs, args.episodes_per_proc, 0, out_dir)
        if args.price_per_vcpu_hr:
            pod_cost_hr = args.price_per_vcpu_hr * int(plat["usable_cpus"])
            step["episodes_per_dollar"] = (
                round(step["episodes_per_hour"] / pod_cost_hr, 1)
                if step["episodes_per_hour"]
                else None
            )
        result["steps"].append(step)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in step.items() if k != "records"}), flush=True)
    if args.reference_jsonl and args.reference_jsonl.exists():
        ref = {
            (r["mix"], r["world_seed"]): r["mass_integral"]
            for r in map(json.loads, args.reference_jsonl.open())
        }
        mismatches = [
            rec
            for step in result["steps"]
            for rec in step["records"]
            if (rec["mix"], rec["world_seed"]) in ref
            and ref[(rec["mix"], rec["world_seed"])] != rec["mass_integral"]
        ]
        result["reference_check"] = {
            "compared": sum(
                1
                for s in result["steps"]
                for r in s["records"]
                if (r["mix"], r["world_seed"]) in ref
            ),
            "mismatches": mismatches[:20],
        }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
