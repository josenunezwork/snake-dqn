"""Mac preflight of the real entrypoint on the pack's probe (tiny) input.

Runs ``colab.sh`` exactly as the family probe would: the same metadata tar
bytes, probe control (first parent per stream, horizon 3), the real-code
counter self-check, ``session_budget`` against a supervisor-shaped spec/state,
one worker, one numerical thread, nice 15. Checkpoints are copied from the
local catalogue instead of downloaded. It never touches the F0 factory,
its leases or ``exp/mac-lease.json*``, and allocates nothing.
"""

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

from research.bl_labels.exports import publish_json

from . import job, pack

TIMEOUT_SECONDS = 1200


def preflight(probe_manifest: Path, archive: Path, root: Path, b8: Path, frp3: Path) -> dict:
    if platform.system() != "Darwin":
        raise ValueError("step-ruler preflight is the Mac preflight")
    manifest = json.loads(probe_manifest.read_text())
    if manifest.get("env", {}).get("STEP_RULER_TRANSPORT_PROBE") != "1":
        raise ValueError("preflight runs the excluded probe manifest only")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    for row in pack.pool_inputs(b8, frp3):
        dest = "public-checkpoints/" + row["dest"]
        target = inputs / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    shutil.copyfile(archive, inputs / "metadata.tar")
    runtime = root / "runtime"
    (runtime / "output").mkdir(parents=True)
    env_values = manifest["env"]
    session = float(env_values["F0_SESSION_SECONDS"])
    job_id = manifest["job_id"] + "-mac-preflight"
    (runtime / "spec.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "generation": 1,
                "root": str(runtime),
                "env": {
                    "F0_SESSION_SECONDS": str(int(session)),
                    "F0_FINALIZE_SECONDS": env_values["F0_FINALIZE_SECONDS"],
                },
                "timeout_seconds": session,
            }
        )
    )
    (runtime / "state.json").write_text(
        json.dumps({"job_id": job_id, "generation": 1, "started_at": time.time()})
    )
    env = dict(os.environ)
    for key in (
        "BL_PUBLIC_INPUTS",
        "BL_PUBLIC_INPUT_ALIASES",
        "SNAKE_COLAB_CANARY",
        "SNAKE_COLAB_CANARY_PACKAGE",
        "COLAB_CLI_REQUIREMENTS",
    ):
        env.pop(key, None)
    env.update(
        {
            k: v
            for k, v in env_values.items()
            if k
            not in {
                "BL_PUBLIC_INPUTS",
                "BL_PUBLIC_INPUT_ALIASES",
                "SNAKE_COLAB_CANARY",
                "SNAKE_COLAB_CANARY_PACKAGE",
                "COLAB_CLI_REQUIREMENTS",
            }
        }
    )
    env.update(
        {
            "COLAB_CLI_ROOT": str(runtime),
            "COLAB_INPUT_DIR": str(inputs),
            "SNAKE_CKPT_DIR": str(inputs / "public-checkpoints/checkpoints"),
            "STEP_RULER_WORKERS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
            **{key: "1" for key in job.THREAD_ENV},
        }
    )
    started = time.perf_counter()
    log = root / "entrypoint.log"
    with log.open("x") as output:
        child = subprocess.Popen(
            ["nice", "-n", "15", "bash", str(pack.ROOT / pack.SCRIPT)],
            cwd=pack.ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            raise TimeoutError("step-ruler Mac preflight exceeded its bound") from None
    wall = time.perf_counter() - started
    out = runtime / "output"
    failures: list[str] = []
    if code:
        failures.append(f"entrypoint exit {code}")
    check: dict[str, Any] = {}
    result: dict[str, Any] = {}
    units: list[str] = []
    try:
        job.verify_unit(out / "self-check.json")
        check = json.loads((out / "self-check.json").read_text())
        job.verify_unit(out / "runtime-result.json")
        result = json.loads((out / "runtime-result.json").read_text())
        for path in sorted((out / job.UNIT_DIR).glob("*.json")):
            job.verify_unit(path)
            value = json.loads(path.read_text())
            if any(key in value for key in job.SEALED_COUNTERS) or "rows" in value:
                failures.append(f"unit carries outcome-bearing fields: {path.name}")
            units.append(path.stem)
    except (OSError, ValueError) as error:
        failures.append(str(error))
    if check.get("status") != "PASS":
        failures.append("counter self-check did not PASS")
    if result.get("status") != "COMPLETE" or len(units) != len(pack.ALLOCATION):
        failures.append("probe allocation did not complete one unit per stream")
    receipt = {
        "schema": "bl-step-ruler-mac-preflight/v1",
        "status": "PREFLIGHT" if not failures else "FAIL",
        "failures": failures,
        "commit": env_values.get("F0_COMMIT"),
        "probe_manifest": str(probe_manifest),
        "probe_manifest_sha256": pack._sha(probe_manifest),
        "metadata_sha256": pack._sha(archive),
        "source_sha256": job.source_sha256(pack.ROOT),
        "entrypoint_wall_seconds": wall,
        "workers": 1,
        "nice": 15,
        "self_check": check,
        "units": units,
        "runtime_status": result.get("status"),
    }
    publish_json(root / "preflight.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    args = parser.parse_args()
    receipt = preflight(args.probe_manifest, args.archive, args.root, args.b8, args.frp3)
    print(json.dumps({k: receipt[k] for k in ("status", "failures", "entrypoint_wall_seconds")}))
    if receipt["status"] != "PREFLIGHT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
