"""Run the T4A CPU tiny matrix through its real packet entrypoint; never allocates CUDA."""

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
from typing import Any

from research.gpu_sim_spike import r2_pack, t4a_pack, t4a_units
from research.gpu_sim_spike.make_manifests import checkpoint_inputs, sha256_file
from research.metadata_archive import bundle_inputs


def preflight(root: Path, commit: str) -> dict[str, Any]:
    """Create-only current-pin CPU preflight over the three tiny roster units."""
    if platform.system() != "Darwin":
        raise ValueError("T4A preflight requires the Mac")
    t4a_pack.require_source_pin(commit)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    for row in checkpoint_inputs():
        target = inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    ledger = root / "metadata-source" / "verified-ledger.json"
    t4a_units.publish_json(
        ledger,
        t4a_units.empty_ledger(
            t4a_units.namespace(commit, smoke=True), t4a_units.allocation(smoke=True)
        ),
    )
    rows = bundle_inputs(
        r2_pack._ledger_inputs(ledger, "inputs") + t4a_pack._bindings(), inputs / "metadata.tar"
    )
    archive_hash = next(row["sha256"] for row in rows if row["dest"] == "metadata.tar")
    env = dict(os.environ)
    env.pop("GPU_SIM_R2_DEBUG", None)
    env.update(
        {
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "GPU_SIM_VECTOR_GEOMETRY": "native",
            "COLAB_CLI_REQUIREMENTS": f"{t4a_pack.PACKAGE}/integration_requirements.txt",
            "GPU_SIM_V8_INDEXED": "0",
            "T4A_BENCH_DEVICE": "cpu",
            "T4A_BENCH_SECONDS": "600",
            "T4A_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
            "T4A_BENCH_SMOKE": "1",
            "T4A_BENCH_CANARY": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "COLAB_CLI_ROOT": str(root / "runtime"),
            "COLAB_INPUT_DIR": str(inputs),
            "METADATA_ARCHIVE_SHA256": archive_hash,
            "SNAKE_DQN_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        }
    )
    started = time.perf_counter()
    log = root / "entrypoint.log"
    with log.open("x") as output:
        child = subprocess.Popen(
            ["bash", str(t4a_pack.ROOT / t4a_pack.SCRIPT)],
            cwd=t4a_pack.ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=660)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            raise TimeoutError("T4A Mac preflight exceeded 660 seconds") from None
    if code:
        raise RuntimeError(f"T4A real entrypoint failed: {code}")
    receipt = root / "runtime/output/t4a-bench.json"
    t4a_units._verify(receipt)
    body = json.loads(receipt.read_text())
    expected = list(t4a_units.allocation(smoke=True))
    if body.get("completed_units") != expected:
        raise ValueError("T4A preflight did not complete the smoke allocation")
    published = {
        "status": "PREFLIGHT",
        "commit": commit,
        "device": "cpu",
        "completed_units": expected,
        "entrypoint_run_wall_seconds": time.perf_counter() - started,
        "receipt_path": str(receipt),
        "receipt_sha256": sha256_file(receipt),
        "source_sha256": body.get("source_sha256"),
    }
    t4a_units.publish_json(root / "preflight.json", published)
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.root, args.commit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
