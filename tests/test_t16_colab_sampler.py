"""Utilization sampler summary and process contract."""

import json
import os
import signal
import subprocess
import sys
import time

from research.t16_colab import sampler


def test_summarize_skips_primed_cpu_sample_and_handles_missing_gpu():
    rows = [
        {"time": 0, "cpu_pct": 0.0, "load1": 1.0},
        {"time": 5, "cpu_pct": 50.0, "load1": 2.0, "gpu_util_pct": 10.0, "gpu_mem_used_mib": 100},
        {"time": 10, "cpu_pct": 90.0, "load1": 3.0, "gpu_util_pct": 30.0, "gpu_mem_used_mib": 300},
    ]
    summary = sampler.summarize(rows)
    assert summary["samples"] == 3
    assert summary["cpu_pct_mean"] == 70.0 and summary["cpu_pct_max"] == 90.0
    assert summary["gpu_util_pct_mean"] == 20.0 and summary["gpu_mem_used_mib_max"] == 300
    assert sampler.summarize([]) == {"samples": 0}


def test_sampler_process_writes_rows_and_stops_on_sigterm(tmp_path):
    out = tmp_path / "resources.jsonl"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "research.t16_colab.sampler",
            "--out",
            str(out),
            "--interval",
            "0.5",
        ],
        cwd=sampler.__file__.rsplit("/research/", 1)[0],
        start_new_session=True,
    )
    try:
        for _ in range(100):
            if out.exists() and out.read_text().count("\n") >= 2:
                break
            time.sleep(0.1)
        os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) >= 2 and all("cpu_pct" in row and "load1" in row for row in rows)
