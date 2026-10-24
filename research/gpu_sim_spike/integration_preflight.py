"""Run the I0 CPU smoke through the actual runtime entrypoint; never allocate CUDA."""

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

from research.gpu_sim_spike import integration_pack, integration_units
from research.gpu_sim_spike.make_manifests import checkpoint_inputs, sha256_file
from research.metadata_archive import bundle_inputs


def preflight(root: Path, commit: str) -> dict:
    """Publish a create-only CPU receipt after a small all-roster real run."""
    if platform.system() != "Darwin":
        raise ValueError("I0 preflight requires the Mac")
    integration_pack.require_source_pin(commit)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    for row in checkpoint_inputs():
        target = inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    ledger = root / "metadata-source" / "verified-ledger.json"
    integration_units.publish_json(
        ledger,
        integration_units.empty_ledger(
            integration_units.namespace(commit, smoke=True),
            integration_units.allocation(smoke=True),
        ),
    )
    archive_rows = bundle_inputs(
        integration_pack._ledger_inputs(ledger, "inputs") + integration_pack._bindings(),
        inputs / "metadata.tar",
    )
    archive_hash = next(row["sha256"] for row in archive_rows if row["dest"] == "metadata.tar")
    env = dict(os.environ)
    env.update(
        {
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "GPU_SIM_VECTOR_GEOMETRY": "native",
            "COLAB_CLI_REQUIREMENTS": f"{integration_pack.PACKAGE}/integration_requirements.txt",
            "GPU_SIM_V8_INDEXED": "0",
            "INTEGRATION_BENCH_DEVICE": "cpu",
            "INTEGRATION_BENCH_SECONDS": "300",
            "INTEGRATION_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
            "INTEGRATION_BENCH_SMOKE": "1",
            "INTEGRATION_BENCH_CANARY": "0",
            "COLAB_CLI_ROOT": str(root / "runtime"),
            "COLAB_INPUT_DIR": str(inputs),
            "METADATA_ARCHIVE_SHA256": archive_hash,
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
    with log.open("x") as output:
        child = subprocess.Popen(
            ["bash", str(integration_pack.ROOT / integration_pack.SCRIPT)],
            cwd=integration_pack.ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=360)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            raise TimeoutError("I0 Mac preflight exceeded 360 seconds") from None
    if code:
        raise RuntimeError(f"I0 real entrypoint failed: {code}")
    receipt = root / "runtime/output/integration-bench.json"
    integration_units._verify(receipt)
    body = json.loads(receipt.read_text())
    expected = list(integration_units.allocation(smoke=True))
    if body.get("completed_units") != expected:
        raise ValueError("I0 preflight did not complete the smoke allocation")
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
    integration_units.publish_json(root / "preflight.json", published)
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    print(
        json.dumps(
            preflight(parser.parse_args().root, parser.parse_args().commit),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
