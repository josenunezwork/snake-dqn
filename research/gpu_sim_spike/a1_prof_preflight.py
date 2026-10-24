"""Mac CPU preflight through the real Colab Bash entrypoint; no provider calls."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import a1_prof_pack, a1_prof_units
from research.gpu_sim_spike.make_manifests import checkpoint_inputs, git_head


def preflight(root: Path, commit: str) -> dict[str, Any]:
    """Stage portable input bytes and time three real ticks for every tiny roster."""
    if platform.system() != "Darwin":
        raise ValueError("Mac preflight requires Darwin")
    a1_prof_pack.require_source_pin(commit)
    if git_head() != commit:
        raise ValueError("Mac preflight must run the exact published source pin")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    for row in checkpoint_inputs():
        target = inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    ledger = inputs / "inputs/verified-ledger.json"
    a1_prof_units.publish_json(ledger, a1_prof_pack.empty_ledger(commit, smoke=True))
    env = dict(os.environ)
    env.update(
        {
            "GPU_SIM_SPIKE_ENTRYPOINT": "a1-prof",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_SPIKE_DEVICE": "cpu",
            "GPU_SIM_SPIKE_SECONDS": "300",
            "GPU_SIM_SPIKE_THREADS": "1",
            "GPU_SIM_SPIKE_CPU_WORKERS": "1",
            "GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR": "10",
            "A1_PROF_DEVICE": "cpu",
            "A1_PROF_SMOKE": "1",
            "A1_PROF_CANARY": "0",
            "A1_PROF_SECONDS": "300",
            "COLAB_CLI_ROOT": str(root / "runtime"),
            "COLAB_INPUT_DIR": str(inputs),
            "A1_PROF_LEDGER_PATH": "inputs/verified-ledger.json",
            "BL_PUBLIC_INPUTS": "0",
            "SNAKE_DQN_DEVICE": "cpu",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PATH": str(Path(__import__("sys").executable).parent)
            + os.pathsep
            + env.get("PATH", ""),
        }
    )
    command = ["bash", str(a1_prof_pack.ROOT / a1_prof_pack.SCRIPT)]
    started = time.perf_counter()
    # The log receives real runner progress immediately. Bound the entire owned
    # process group, including silent imports/native operations and descendants.
    with (root / "entrypoint.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=a1_prof_pack.ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
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
            raise TimeoutError("Mac preflight exceeded its 330-second process bound") from None
    print((root / "entrypoint.log").read_text(), end="", flush=True)
    if code:
        raise RuntimeError(f"A1-PROF real entrypoint failed: {code}")
    elapsed = time.perf_counter() - started
    output = root / "runtime/output/a1-prof.json"
    a1_prof_units._verify_sidecars(output)
    receipt = json.loads(output.read_text())
    if receipt["status"] != "PREFLIGHT" or len(receipt["completed_units"]) != 3:
        raise ValueError("Mac preflight did not complete all three tiny rosters")
    result = {
        "status": "PREFLIGHT",
        "commit": commit,
        "device": "cpu",
        "entrypoint_command": command,
        "entrypoint_run_wall_seconds": elapsed,
        "receipt_path": str(output),
        "receipt_sha256": a1_prof_pack.sha256_file(output),
        "units": receipt["completed_units"],
        "source_sha256": receipt["source_sha256"],
    }
    a1_prof_units.publish_json(root / "mac-preflight.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    preflight(args.root, args.commit)


if __name__ == "__main__":
    main()
