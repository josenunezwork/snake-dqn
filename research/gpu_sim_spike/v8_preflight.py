"""Mac-only real V8 entrypoint smoke: all three E1 roster cells, no allocation."""

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

from research.gpu_sim_spike import v8_pack, v8_units
from research.gpu_sim_spike.make_manifests import checkpoint_inputs, sha256_file


def preflight(root: Path, commit: str) -> dict:
    if platform.system() != "Darwin":
        raise ValueError("V8 preflight requires the Mac")
    v8_pack.require_source_pin(commit)
    if v8_pack.git_head() != commit:
        raise ValueError("preflight requires exact source pin")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    for row in checkpoint_inputs():
        target = inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    ledger = inputs / "inputs/verified-ledger.json"
    v8_units.publish_json(
        ledger,
        v8_units.empty_ledger(
            v8_units.namespace(commit, smoke=True), v8_units.allocation(smoke=True)
        ),
    )
    env = dict(os.environ)
    env.update(
        {
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "V8_BENCH_DEVICE": "cpu",
            "V8_BENCH_SECONDS": "300",
            "V8_BENCH_SMOKE": "1",
            "V8_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
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
    log = root / "entrypoint.log"
    with log.open("x") as stream:
        process = subprocess.Popen(
            ["bash", str(v8_pack.ROOT / v8_pack.SCRIPT)],
            cwd=v8_pack.ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=330)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise TimeoutError("V8 Mac preflight exceeded 330 seconds") from None
    if code:
        raise RuntimeError(f"V8 real entrypoint failed: {code}")
    receipt = root / "runtime/output/v8-bench.json"
    v8_units._verify(receipt)
    body = json.loads(receipt.read_text())
    if body.get("status") != "PREFLIGHT" or body.get("completed_units") != list(
        v8_units.allocation(smoke=True)
    ):
        raise ValueError("V8 preflight did not complete three E1 rosters")
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
    v8_units.publish_json(root / "mac-preflight.json", result)
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--commit", required=True)
    a = p.parse_args()
    preflight(a.root, a.commit)


if __name__ == "__main__":
    main()
