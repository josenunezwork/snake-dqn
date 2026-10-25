"""Owner-scoped serial preflight that never reads or mutates Mac lease files.

The explicit 2026-10-25 handoff authorizes nice-15 single-thread tests while
making exp/mac-lease.json* untouchable. The historical local_check route takes
that lease and is unsuitable for this handoff. Keep the shared serial slot,
120-second bound, immutable source evidence and synthetic-only scope here.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path

from research.bl_ppo.exports import export_json
from research.bl_round30.local_check import NODES, ROOT, source_identity

SLOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/factory-companions/test-slot.lock"
)


def run(output: Path) -> int:
    """Execute one source-bound current suite; no live process/lease operations."""
    if os.getpriority(os.PRIO_PROCESS, 0) < 15:
        raise ValueError("nice >=15 required")
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if os.environ.get(key) != "1":
            raise ValueError(f"{key}=1 required")
    if os.environ.get("SNAKE_DQN_DEVICE") != "cpu":
        raise ValueError("CPU required")
    output.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    nodes = list(NODES)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
    ).strip()
    command = [
        str(ROOT / "venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:xdist",
        "-p",
        "research.compute.factory_companion_pytest",
        *nodes,
    ]
    started = time.monotonic()
    SLOT.parent.mkdir(parents=True, exist_ok=True)
    with SLOT.open("a") as slot:
        fcntl.flock(slot.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / "check.log").open("x") as log:
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=115,
                    check=False,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
    current = source_identity()
    export_json(
        output / "preflight.json",
        {
            "schema": "r30-mac-preflight/v1",
            "status": "PASS" if code == 0 and identity == current else "HOLD",
            "returncode": code,
            "source_identity": identity,
            "source_commit": commit,
            "elapsed_seconds": time.monotonic() - started,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "numerical_threads": 1,
            "max_elapsed_seconds": 120,
            "command": command,
            "authorization": "owner explicit 2026-10-25 implementation handoff",
            "lease_files_accessed": False,
            "shared_serial_slot": str(SLOT),
        },
    )
    return code if identity == current else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output.resolve()))


if __name__ == "__main__":
    main()
