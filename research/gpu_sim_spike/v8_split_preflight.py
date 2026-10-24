"""Run the real V8-SPLIT-01 CPU smoke without allocating a runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from research.gpu_sim_spike import v8_split_pack, v8_split_units
from research.gpu_sim_spike.make_manifests import checkpoint_inputs, sha256_file


def preflight(root: Path, commit: str) -> dict:
    if platform.system() != "Darwin":
        raise ValueError("V8 split preflight requires the Mac")
    v8_split_pack.require_source_pin(commit)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    for item in checkpoint_inputs():
        target = inputs / item["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["path"], target)
    ledger = inputs / "inputs/verified-ledger.json"
    v8_split_units.publish_json(
        ledger,
        v8_split_units.empty_ledger(
            v8_split_units.namespace(commit, smoke=True), v8_split_units.allocation(smoke=True)
        ),
    )
    env = dict(os.environ)
    env.update(
        {
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "V8_SPLIT_BENCH_DEVICE": "cpu",
            "V8_SPLIT_BENCH_SECONDS": "300",
            "V8_SPLIT_BENCH_SMOKE": "1",
            "V8_SPLIT_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
            "COLAB_CLI_ROOT": str(root / "runtime"),
            "COLAB_INPUT_DIR": str(inputs),
            "SNAKE_DQN_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        }
    )
    started = time.perf_counter()
    # Exercise the same adversarial oracle inputs on the Mac before any packet
    # can reach the native A100 gate (which deliberately selects CUDA tests).
    with (root / "cpu-parity.log").open("x") as stream:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "-p",
                "no:xdist",
                "research/gpu_sim_spike/test_v8_split_parity.py::"
                "test_cpu_v8_independent_diagnostics_masks_order_and_planes",
            ],
            cwd=v8_split_pack.ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=True,
        )
    log = root / "entrypoint.log"
    with log.open("x") as stream:
        child = subprocess.Popen(
            ["bash", str(v8_split_pack.ROOT / v8_split_pack.SCRIPT)],
            cwd=v8_split_pack.ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=330)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            raise TimeoutError("V8 split Mac preflight exceeded 330 seconds") from None
    if code:
        raise RuntimeError(f"V8 split real entrypoint failed: {code}")
    receipt = root / "runtime/output/v8-split-bench.json"
    v8_split_units._verify(receipt)
    body = json.loads(receipt.read_text())
    expected = list(v8_split_units.allocation(smoke=True))
    if body.get("status") != "PREFLIGHT" or body.get("completed_units") != expected:
        raise ValueError("V8 split preflight did not complete three E1 rosters")
    result = {
        "status": "PREFLIGHT",
        "commit": commit,
        "device": "cpu",
        "entrypoint_run_wall_seconds": time.perf_counter() - started,
        "receipt_path": str(receipt),
        "receipt_sha256": sha256_file(receipt),
        "completed_units": body["completed_units"],
        "source_sha256": body["source_sha256"],
    }
    v8_split_units.publish_json(root / "mac-preflight.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.root, args.commit), sort_keys=True))


if __name__ == "__main__":
    main()
