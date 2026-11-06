"""Run the actual bank job shell locally on one excluded H300 fixture, then read it."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from research.watch_replay.prepare import release_bundle
from research.watch_replay.run import ROOT, identity
from research.watch_replay.run_l1_os import allocation
from research.watch_replay.seed_audit import L1_OS_SMOKE_NAMESPACE, l1_os_bank_disjointness
from research.watch_replay.uploads import METADATA, pack_inputs
from tools.ops.package_spec import Input, empty_ledger, publish_json, stage_inputs
from web.backend.watch_replay import Recording, validate_frame


def smoke(out: Path, horizon: int = 300) -> dict:
    """Exercise public-input staging, job.sh, generation, and the product frame reader.

    No allocator is involved. The shell downloads the two hash-pinned public release
    assets through the existing staging path. It never changes the served bank.
    """
    if not 1 <= horizon <= 300:
        raise ValueError("local smoke horizon must be 1..300")
    audit = l1_os_bank_disjointness()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    publish_json(out / "seed-audit.json", audit)
    contract = {**allocation(L1_OS_SMOKE_NAMESPACE, 1, horizon=horizon), "commit": commit}
    inputs = []
    for label in ("main", "canary"):
        publish_json(out / f"{label}-contract.json", contract)
        publish_json(
            out / f"{label}-ledger.json",
            empty_ledger(contract["namespace"], "sessions", identity(contract)),
        )
        inputs += [
            Input(out / f"{label}-{kind}.json", f"{label}-{kind}.json")
            for kind in ("contract", "ledger")
        ]
    pins = json.loads((ROOT / "web/backend/served_checkpoint_pins.json").read_text())
    bundle = out / "release-evidence.tar.gz"
    release_bundle(bundle, pins)
    inputs += [
        Input(bundle, bundle.name),
        Input(ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json", "public-inputs.json"),
    ]
    packed = pack_inputs(stage_inputs(out / "staged", inputs), out / METADATA)
    runtime = out / "runtime"
    runtime.mkdir()
    (runtime / "repo").symlink_to(ROOT, target_is_directory=True)
    (runtime / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    uploaded = out / "inputs"
    uploaded.mkdir()
    shutil.copyfile(out / METADATA, uploaded / METADATA)
    output = out / "output"
    output.mkdir()
    env = {
        **os.environ,
        "COLAB_CLI_ROOT": str(runtime),
        "COLAB_INPUT_DIR": str(uploaded),
        "COLAB_OUTPUT_DIR": str(output),
        "WATCH_REPLAY_METADATA_SHA256": packed[0]["sha256"],
        "WATCH_REPLAY_WORKERS": "1",
        "WATCH_REPLAY_CHECKPOINT_SECONDS": "15",
        "WATCH_REPLAY_SESSION_SECONDS": "600",
        "SNAKE_COLAB_CANARY": "0",
        "PYTHONUNBUFFERED": "1",
    }
    job = ROOT / "research/watch_replay/job_l1_os.sh"
    command = ["bash", str(job)]
    if sys.platform == "darwin":
        # Like conftest.py: route torch/sklearn to one OpenMP dylib. Set it inside
        # bash because macOS strips DYLD_* when starting system executables.
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            env["WATCH_REPLAY_MAC_LIBOMP"] = str(Path(spec.origin).parent / "lib")
            command = [
                "bash",
                "-c",
                'export DYLD_LIBRARY_PATH="$WATCH_REPLAY_MAC_LIBOMP"; source "$1"',
                "watch-replay-smoke",
                str(job),
            ]
    started = time.perf_counter()
    with (out / "job.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    shell_seconds = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"job shell failed ({result.returncode}); see {out / 'job.log'}")
    path = output / "sessions/000000.watch.gz"
    read_started = time.perf_counter()
    record = Recording(path, product=False)
    try:
        for frame in range(1, horizon + 1):
            validate_frame(record.frame, frame)
            if frame < horizon and not record.step():
                raise ValueError("smoke recording ended early")
        if record.step():
            raise ValueError("smoke recording contains extra frames")
        report = {
            "status": "PASS",
            "commit": commit,
            "namespace": contract["namespace"],
            "horizon": horizon,
            "validated_frames": horizon,
            "sha256": record.sha256,
            "replay": str(path),
            "compressed_bytes": path.stat().st_size,
            "controller": record.header["controller"],
            "stats": record.header["stats"],
            "shell_seconds": shell_seconds,
            "recording_validation_seconds": time.perf_counter() - read_started,
            "operational": json.loads((output / "sessions/operational.json").read_text()),
            "scope": "Mac CPU H300 fixture, actual job shell; no Colab or release qualification",
        }
    finally:
        record.close()
    publish_json(out / "smoke.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(smoke(args.out, args.horizon), indent=2))


if __name__ == "__main__":
    main()
