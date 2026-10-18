#!/usr/bin/env bash
# Host fixtures only. This launcher never calls the guarded Q0/Q1 dispatcher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# The worker puts the already prepared venv first on PATH. No setup/install here.
exec python3 - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

out = Path(os.environ["QUEUE_OUTPUT_DIR"])
if not out.is_absolute() or not out.is_dir():
    raise SystemExit("QUEUE_OUTPUT_DIR must be an existing absolute job-private directory")
# Requeues must use a fresh output directory rather than overwrite prior evidence.
if any((out / name).exists() for name in (
    "receipt.json", "host-contracts.xml", "stdout.log", "stderr.log",
    "pytest-cache", "pytest-tmp",
)):
    raise SystemExit("host-contract outputs already exist; use a fresh attempt directory")

command = [
    "timeout", "--signal=TERM", "--kill-after=30s", "900s",
    "/content/t16-colab-venv/bin/python", "-m", "pytest", "-q", "-ra", "--durations=15",
    "-o", f"cache_dir={out / 'pytest-cache'}",
    f"--basetemp={out / 'pytest-tmp'}", f"--junitxml={out / 'host-contracts.xml'}",
    "tests/test_q1_trainer.py",
    "tests/test_q1_collector_oracles.py",
    "tests/test_q1_qualification.py",
    "tests/test_q0_audits.py",
    "tests/test_q0_episode_metrics.py",
    "tests/test_t22_random_safe.py",
    "tests/test_t24_v8_bodymap.py",
]

def capture(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=10).strip()

versions = {}
for package in ("torch", "numpy", "numba", "llvmlite"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
identity = {
    "commit": capture(["git", "rev-parse", "HEAD"]),
    "git_status": capture(["git", "status", "--porcelain", "--untracked-files=no"]),
    "launcher_sha256": hashlib.sha256(Path(
        "research/q0q1_20261017/run_host_contracts.sh"
    ).read_bytes()).hexdigest(),
    "python": sys.version,
    "python_executable": sys.executable,
    "packages": versions,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "cpu_count": os.cpu_count(),
    "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    "environment": {key: os.environ[key] for key in (
        "SNAKE_DQN_DEVICE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
    )},
}
try:
    identity["gpu"] = capture([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader",
    ])
except (OSError, subprocess.SubprocessError) as exc:
    identity["gpu"] = {"unavailable": str(exc)}
started_at = datetime.now(timezone.utc).isoformat()
start = time.monotonic()
with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
    try:
        status = subprocess.run(command, stdout=stdout, stderr=stderr).returncode
    except OSError as exc:
        stderr.write(str(exc) + "\n")
        status = 127
elapsed = time.monotonic() - start
skips = None
junit_error = None
try:
    root = ET.parse(out / "host-contracts.xml").getroot()
    skips = len(root.findall(".//testcase/skipped"))
    if not root.findall(".//testcase"):
        junit_error = "JUnit contains no test cases"
except (OSError, ET.ParseError) as exc:
    junit_error = str(exc)
receipt = {
    "schema": "overnight-host-contracts/v1",
    "status": "PASS" if status == 0 and skips == 0 and junit_error is None else "INCOMPLETE",
    "identity": identity,
    "command": command,
    "started_at": started_at,
    "exit_code": status,
    "elapsed_seconds": elapsed,
    "skip_count": skips,
    "junit_error": junit_error,
    "artifacts_sha256": {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ("stdout.log", "stderr.log", "host-contracts.xml") if (out / name).is_file()
    },
}
receipt_path = out / "receipt.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(out / "receipt.json.sha256").write_text(
    hashlib.sha256(receipt_path.read_bytes()).hexdigest() + "  receipt.json\n"
)
# Match shell signal status as well as ordinary pytest/timeout exit codes.
sys.exit(status if status >= 0 else 128 - status)
PY
