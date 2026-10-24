"""Tiny real-entrypoint Mac rehearsal in a scrubbed, pinned, excluded scratch tree."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from research.r33_verification.contract import (
    ALLOCATION,
    NAMESPACE,
    SOURCE_PATHS,
    environment_identity,
)
from tools.ops.package_spec import (
    Input,
    digest,
    empty_ledger,
    publish_json,
    require_published_commit,
    stage_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


def owned_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Bound and reap this command's entire process group, including on alarm."""
    from research.compute.factory_interim_launcher import _cleanup

    timeout = kwargs.pop("timeout")
    check = kwargs.pop("check", False)
    kwargs.pop("capture_output", None)
    kwargs.pop("text", None)
    child = subprocess.Popen(
        command,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kwargs,
    )
    try:
        stdout, stderr = child.communicate(timeout=timeout)
        result = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
    finally:
        _cleanup(child, time.monotonic() + 4)


def source_inventory(commit: str) -> dict[str, Any]:
    archive = subprocess.check_output(
        ["git", "archive", commit, "--", *SOURCE_PATHS], cwd=ROOT, timeout=30
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        files = {
            member.name: {
                "sha256": hashlib.sha256(stream.extractfile(member).read()).hexdigest(),
                "mode": member.mode,
            }
            for member in stream.getmembers()
            if member.isfile()
        }
    return files


def source_closure(commit: str) -> str:
    return hashlib.sha256(json.dumps(source_inventory(commit), sort_keys=True).encode()).hexdigest()


def run(commit: str, out: Path) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise RuntimeError("Mac preflight requires the native Mac; Linux CI is separate")
    commit = require_published_commit(ROOT, commit)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="r33-scrubbed-") as directory:
        work = Path(directory)
        source = work / "source"
        source.mkdir()
        archive = subprocess.check_output(
            ["git", "archive", commit, "--", *SOURCE_PATHS], cwd=ROOT, timeout=30
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(source, filter="data")
        env = {
            "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
            "UV_NO_CONFIG": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "SNAKE_DQN_DEVICE": "cpu",
        }
        for key in (
            "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[key] = "1"
        owned_run(
            ["uv", "--no-config", "venv", "--python", sys.executable, str(work / "venv")],
            env=env,
            check=True,
            timeout=35,
        )
        python = work / "venv/bin/python"
        owned_run(
            [
                "uv",
                "--no-config",
                "pip",
                "install",
                "--python",
                str(python),
                "-r",
                str(source / "research/r33_verification/requirements.txt"),
            ],
            env=env,
            check=True,
            timeout=35,
        )
        identity = {
            "pin": commit,
            "environment_sha256": environment_identity(source),
            "packet": digest(source / "research/r33_verification/contract_inventory.json"),
            "source_closure_sha256": source_closure(commit),
        }
        initial = work / "initial-ledger.json"
        publish_json(initial, empty_ledger(NAMESPACE, ALLOCATION, identity))
        staged = work / "runtime/inputs"
        binding = work / "source-inventory.json"
        publish_json(
            binding,
            {
                "schema": "r33-source-inventory/v1",
                "commit": commit,
                "files": source_inventory(commit),
            },
        )
        stage_inputs(
            staged,
            [Input(initial, "verified-ledger.json"), Input(binding, "source-inventory.json")],
        )
        env.update(
            R33_NAMESPACE=NAMESPACE,
            R33_ALLOCATION=ALLOCATION,
            R33_PIN=commit,
            R33_ENVIRONMENT_SHA256=identity["environment_sha256"],
            R33_PACKET=identity["packet"],
            R33_SOURCE_CLOSURE=identity["source_closure_sha256"],
            COLAB_INPUT_DIR=str(staged),
            COLAB_OUTPUT_DIR=str(work / "runtime/output"),
        )
        env["PATH"] = f"{python.parent}:/usr/bin:/bin"
        temporary_files = work / "tmp"
        temporary_files.mkdir()
        env["TMPDIR"] = str(temporary_files)
        # Dependency installation precedes the rehearsal. The actual producer
        # and consumer cannot read credentials/user artifacts or access network.
        profile = (
            "(version 1)(allow default)(deny network*)"
            '(deny file-read* (subpath "/Users") (subpath "/Volumes"))'
            '(deny file-write* (subpath "/Users") (subpath "/Volumes"))'
        )
        sandbox = ["/usr/bin/sandbox-exec", "-p", profile]
        result = owned_run(
            [*sandbox, "/bin/bash", "research/r33_verification/colab.sh"],
            cwd=source,
            env=env,
            capture_output=True,
            text=True,
            timeout=75,
        )
        if result.returncode:
            raise RuntimeError(
                "real entrypoint preflight failed:\n" + result.stdout + result.stderr
            )
        # Import the actual pinned consumer in a separate, isolated interpreter.
        consumer = (
            "import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
            "from research.r33_verification.verify import union,decide; "
            "from tools.ops.package_spec import unit_id; "
            "ns,al,identity=sys.argv[3],sys.argv[4],json.loads(sys.argv[5]); "
            'root=Path(sys.argv[2]); output=root/"output"; '
            "expected={unit_id(ns,al,i) for i in range(3)}; "
            'union([output],root/"union.json",ns,al,identity,expected); '
            'decide([output],root/"union.json",root/"decision.json",ns,al,identity)'
        )
        owned_run(
            [
                *sandbox,
                str(python),
                "-I",
                "-c",
                consumer,
                str(source),
                str(work / "runtime"),
                NAMESPACE,
                ALLOCATION,
                json.dumps(identity),
            ],
            env=env,
            check=True,
            timeout=10,
        )
        stdout = out / "entrypoint.stdout.txt"
        stdout.write_text(result.stdout)
        decision = json.loads((work / "runtime/decision.json").read_text())
        measured = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        per_unit = [
            row["elapsed_seconds"] for row in measured if row.get("stage") == "verified-shard"
        ]
        report = {
            "schema": "r33-mac-preflight/v1",
            "status": "PASS",
            "commit": commit,
            "source_closure_sha256": source_closure(commit),
            "identity": identity,
            "units": 3,
            "per_unit_seconds": per_unit,
            "checks_passed": decision["checks_passed"],
            "elapsed_seconds": time.monotonic() - started,
            "effective_nice": os.getpriority(os.PRIO_PROCESS, 0),
            "platform": platform.system(),
            "machine": platform.machine(),
            "entrypoint": ["/bin/bash", "research/r33_verification/colab.sh"],
            "stdout_sha256": digest(stdout),
            "native_pending": decision["native_pending"],
            "scrubbed": True,
            "isolation": "seatbelt-no-users-volumes-network",
            "excluded": True,
            "allocation_attempted": False,
        }
        publish_json(out / "receipt.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from research.compute.factory_interim_launcher import (
        _elapsed_alarm,
        _nice_floor,
        serial_test_slot,
    )
    from research.mac_lease import held

    _nice_floor()  # Lower priority if needed; never raise inherited priority.
    with _elapsed_alarm(time.monotonic() + 120):
        with serial_test_slot():
            with held("r33-tiny-preflight", "R33 excluded three-shard preflight", seconds=120):
                print(json.dumps(run(args.commit, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
