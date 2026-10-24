"""Mac CPU entrypoint proof for MOVE2; classification is diagnostic, never CPU PASS."""

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

from research.gpu_sim_spike import move2_pack, move2_units
from research.gpu_sim_spike.make_manifests import checkpoint_inputs


def run(root: Path, commit: str) -> dict:
    if platform.system() != "Darwin":
        raise ValueError("MOVE2 preflight requires Darwin")
    move2_pack.require_source_pin(commit)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    for row in checkpoint_inputs():
        target = inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    ledger = inputs / "inputs/verified-ledger.json"
    move2_units.publish_json(
        ledger,
        move2_units.empty_ledger(
            move2_units.namespace(commit, smoke=True), move2_units.allocation(smoke=True)
        ),
    )
    env = dict(os.environ)
    env.update(
        {
            "COLAB_CLI_ROOT": str(root / "runtime"),
            "COLAB_INPUT_DIR": str(inputs),
            "MOVE2_BENCH_SECONDS": "300",
            "MOVE2_BENCH_DEVICE": "cpu",
            "MOVE2_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
            "MOVE2_BENCH_MODE": "--smoke",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "SNAKE_DQN_DEVICE": "cpu",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        }
    )
    command = ["nice", "-n", "15", "bash", "research/gpu_sim_spike/move2_bench.sh"]
    started = time.perf_counter()
    with (root / "entrypoint.log").open("x") as log:
        process = subprocess.run(
            command,
            cwd=move2_pack.ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=330,
            check=False,
        )
    output = root / "runtime/output"
    receipt, diagnostic, register = (
        output / "move2-bench.json",
        output / "cpu-discrepancies.json",
        output / "discrepancy-register.json",
    )
    if not receipt.is_file():
        result = {
            "status": "INVALID",
            "failure_class": "MECHANICAL_ENTRYPOINT",
            "reason": "entrypoint did not publish a complete benchmark receipt",
            "commit": commit,
            "device": "cpu",
            "exit_code": process.returncode,
            "entrypoint_command": command,
            "entrypoint_run_wall_seconds": time.perf_counter() - started,
            "log_path": str(root / "entrypoint.log"),
            "native_cuda": "NOT_RUN",
            "gpu_units_spent": 0,
        }
        move2_units.publish_json(root / "mac-preflight.json", result)
        return result
    receipt_sha = move2_units._verify(receipt)
    diagnostic_sha = move2_units._verify(diagnostic)
    register_sha = move2_units._verify(register)
    body, discrepancies, register_value = (
        json.loads(receipt.read_text()),
        json.loads(diagnostic.read_text()),
        json.loads(register.read_text()),
    )
    ok = (
        process.returncode == 0
        and body.get("status") == "PASS"
        and discrepancies.get("status") == "DIAGNOSTIC"
        and register_value.get("ruling_sha256") == move2_pack.RULING_SHA256
    )
    result = {
        "status": "PREFLIGHT" if ok else "INVALID",
        "commit": commit,
        "device": "cpu",
        "entrypoint_command": command,
        "entrypoint_run_wall_seconds": time.perf_counter() - started,
        "exit_code": process.returncode,
        "receipt_path": str(receipt),
        "receipt_sha256": receipt_sha,
        "cpu_discrepancies_path": str(diagnostic),
        "cpu_discrepancies_sha256": diagnostic_sha,
        "discrepancy_register_path": str(register),
        "discrepancy_register_sha256": register_sha,
        "completed_units": body.get("completed_units", []),
        "ruling_sha256": move2_pack.RULING_SHA256,
        "serial": True,
        "nice": 15,
    }
    move2_units.publish_json(root / "mac-preflight.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--commit", required=True)
    a = p.parse_args()
    result = run(a.root, a.commit)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PREFLIGHT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
