"""Run a tiny real sparse-benchmark entrypoint locally; no provider calls."""

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

from research.gpu_sim_spike import sparse_pack, sparse_units
from research.gpu_sim_spike.make_manifests import checkpoint_inputs, git_head, sha256_file


def preflight(root: Path, commit: str, exactness_fixture: Path | None = None) -> dict:
    if platform.system() != "Darwin":
        raise ValueError("sparse preflight requires the Mac")
    sparse_pack.require_source_pin(commit)
    if git_head() != commit:
        raise ValueError("preflight must use the exact published source pin")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    for row in checkpoint_inputs():
        target = inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    ledger = inputs / "inputs/verified-ledger.json"
    sparse_units.publish_json(
        ledger,
        sparse_units.empty_ledger(
            sparse_units.namespace(commit, smoke=True), sparse_units.allocation(smoke=True)
        ),
    )
    env = dict(os.environ)
    env.update(
        {
            "GPU_SIM_SPIKE_ENTRYPOINT": "sparse-bench",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_SPIKE_DEVICE": "cpu",
            "GPU_SIM_SPIKE_SECONDS": "300",
            "GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR": "10",
            "SPARSE_BENCH_DEVICE": "cpu",
            "SPARSE_BENCH_SECONDS": "300",
            "SPARSE_BENCH_SMOKE": "1",
            "SPARSE_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
            "COLAB_CLI_ROOT": str(root / "runtime"),
            "COLAB_INPUT_DIR": str(inputs),
            "SNAKE_DQN_DEVICE": "cpu",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        }
    )
    if exactness_fixture is not None:
        target = inputs / "exactness/failed-a100-state.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(exactness_fixture, target)
        env["SPARSE_EXACTNESS_FIXTURE"] = "exactness/failed-a100-state.npz"
    began = time.perf_counter()
    log = root / "entrypoint.log"
    with log.open("x") as stream:
        process = subprocess.Popen(
            ["bash", str(sparse_pack.ROOT / sparse_pack.SCRIPT)],
            cwd=sparse_pack.ROOT,
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
            raise TimeoutError(
                "sparse Mac preflight exceeded its 330-second process bound"
            ) from None
    if code:
        raise RuntimeError(f"sparse real entrypoint failed: {code}")
    receipt_path = root / "runtime/output/sparse-bench.json"
    sparse_units._verify(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "PREFLIGHT" or receipt.get("completed_units") != list(
        sparse_units.allocation(smoke=True)
    ):
        raise ValueError("sparse preflight did not complete all tiny rosters")
    result = {
        "status": "PREFLIGHT",
        "commit": commit,
        "device": "cpu",
        "entrypoint_run_wall_seconds": time.perf_counter() - began,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "completed_units": receipt["completed_units"],
        "source_sha256": receipt["source_sha256"],
    }
    if exactness_fixture is not None:
        probe_path = root / "runtime/output/vector-exactness.json"
        sparse_units._verify(probe_path)
        probe = json.loads(probe_path.read_text())
        if probe.get("status") != "PASS" or probe.get("fixture_sha256") != sha256_file(
            exactness_fixture
        ):
            raise ValueError("exactness fixture/probe differs from preflight input")
        result["exactness_fixture_sha256"] = sha256_file(exactness_fixture)
        result["exactness_probe"] = {"path": str(probe_path), "sha256": sha256_file(probe_path)}
    sparse_units.publish_json(root / "mac-preflight.json", result)
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--exactness-fixture", type=Path)
    args = parser.parse_args()
    preflight(args.root, args.commit, args.exactness_fixture)


if __name__ == "__main__":
    main()
