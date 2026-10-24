"""Serial nice-15 CPU smoke through MOVE's actual shell entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from research.gpu_sim_spike import a1_prof_pack, move_units
from research.gpu_sim_spike.make_manifests import checkpoint_inputs, git_head


def run(root: Path, commit: str) -> dict:
    """Retain an immutable INVALID receipt as well as successful preflights."""
    if platform.system() != "Darwin":
        raise ValueError("MOVE preflight requires Darwin")
    if git_head() != commit:
        raise ValueError("preflight must use exact source pin")
    a1_prof_pack.require_source_pin(commit)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    for row in checkpoint_inputs():
        target = inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    ledger = inputs / "inputs/verified-ledger.json"
    move_units.publish_json(
        ledger,
        move_units.empty_ledger(
            move_units.namespace(commit, smoke=True), move_units.allocation(smoke=True)
        ),
    )
    env = {
        **os.environ,
        "COLAB_CLI_ROOT": str(root / "runtime"),
        "COLAB_INPUT_DIR": str(inputs),
        "MOVE_BENCH_SECONDS": "300",
        "MOVE_BENCH_DEVICE": "cpu",
        "MOVE_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
        "MOVE_BENCH_MODE": "--smoke",
        "GPU_SIM_SPIKE_COMMIT": commit,
        "SNAKE_DQN_DEVICE": "cpu",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    command = ["nice", "-n", "15", "bash", "research/gpu_sim_spike/move_bench.sh"]
    started = time.perf_counter()
    with (root / "entrypoint.log").open("x") as log:
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=330,
            check=False,
        )
    receipt = root / "runtime/output/move-bench.json"
    digest = move_units._verify(receipt)
    body = json.loads(receipt.read_text())
    passed = process.returncode == 0 and body.get("status") == "COMPLETE"
    value = {
        "status": "PREFLIGHT" if passed else "INVALID",
        "commit": commit,
        "device": "cpu",
        "entrypoint_command": command,
        "entrypoint_run_wall_seconds": time.perf_counter() - started,
        "exit_code": process.returncode,
        "completed_units": body.get("completed_units", []),
        "receipt_path": str(receipt),
        "receipt_sha256": digest,
        "serial": True,
        "nice": 15,
        "reason": body.get("reason"),
        "cpu_parity": str(receipt.parent / "cpu-parity.json"),
    }
    move_units.publish_json(root / "mac-preflight.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    result = run(args.root, args.commit)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PREFLIGHT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
