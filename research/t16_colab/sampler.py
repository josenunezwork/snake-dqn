"""Low-rate GPU/CPU utilization sampler for the T16 throughput smoke.

The runner starts one sampler per timed configuration and stops it afterwards. It
appends one JSON line every ``--interval`` seconds with the GPU utilization and memory
reported by ``nvidia-smi`` (when present), the whole-machine CPU utilization from
``psutil``, and the one-minute load average. The summary tells an actor-bound run
(high CPU, low GPU) from a GPU-bound one, which decides the Colab runtime for the
league lane. It never touches the trainer's processes or files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def gpu_sample() -> dict | None:
    """One ``nvidia-smi`` query for GPU 0, or ``None`` when the tool is missing or fails."""
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return None
    try:
        out = subprocess.run(
            [
                tool,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        util, used, total = (float(part.strip()) for part in out.strip().split(",")[:3])
    except (subprocess.SubprocessError, ValueError, OSError):
        return None
    return {"gpu_util_pct": util, "gpu_mem_used_mib": used, "gpu_mem_total_mib": total}


def sample(previous_cpu: float | None = None) -> dict:
    """One row: wall clock, CPU percent since the previous call, load average, GPU fields."""
    import os

    import psutil

    row: dict[str, Any] = {
        "time": time.time(),
        "cpu_pct": psutil.cpu_percent(interval=None),
        "load1": os.getloadavg()[0],
        "cpu_count": os.cpu_count(),
    }
    gpu = gpu_sample()
    if gpu is not None:
        row.update(gpu)
    return row


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Mean and maximum of each utilization field over the sampled rows (skipping the first
    CPU sample, which ``psutil`` reports as 0 before any interval has elapsed)."""

    def stats(key: str, values: list[float]) -> dict:
        if not values:
            return {}
        return {f"{key}_mean": sum(values) / len(values), f"{key}_max": max(values)}

    cpu = [float(r["cpu_pct"]) for r in rows[1:] if r.get("cpu_pct") is not None]
    gpu = [float(r["gpu_util_pct"]) for r in rows if r.get("gpu_util_pct") is not None]
    mem = [float(r["gpu_mem_used_mib"]) for r in rows if r.get("gpu_mem_used_mib") is not None]
    load = [float(r["load1"]) for r in rows if r.get("load1") is not None]
    summary = {"samples": len(rows), **stats("cpu_pct", cpu), **stats("gpu_util_pct", gpu)}
    summary.update(stats("gpu_mem_used_mib", mem))
    summary.update(stats("load1", load))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Append samples to ``--out`` until SIGTERM/SIGINT; exit 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args(argv)
    stop = {"flag": False}

    def request_stop(signum: int, frame: object) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample()  # prime psutil's CPU counter so the next reading covers a real interval
    with args.out.open("a") as stream:
        while not stop["flag"]:
            stream.write(json.dumps(sample()) + "\n")
            stream.flush()
            deadline = time.monotonic() + max(0.5, args.interval)
            while not stop["flag"] and time.monotonic() < deadline:
                time.sleep(0.2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
