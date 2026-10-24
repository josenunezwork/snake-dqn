"""Tiny launcher checks with real scheduling observation; no agent or factory run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "agent_nice.sh"
CAPS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PYTHONDONTWRITEBYTECODE",
)


@pytest.mark.parametrize("inherited_nice", [0, 15, 19])
def test_launcher_caps_environment_and_child_priority(inherited_nice: int) -> None:
    probe = """
import json, os, subprocess, sys
nice = int(subprocess.check_output(['/bin/ps', '-o', 'nice=', '-p', str(os.getpid())]))
child = subprocess.check_output([sys.executable, '-c',
    'import os; print(os.getpriority(os.PRIO_PROCESS, 0))'], text=True).strip()
print(json.dumps({'nice': nice, 'child_nice': int(child),
    'env': {k: os.environ[k] for k in sys.argv[1:-1]}, 'arg': sys.argv[-1]}))
"""
    # Lowering priority is permitted; never try to raise this test runner's priority.
    parent_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if inherited_nice < parent_nice:
        pytest.skip("cannot raise inherited scheduler priority")
    result = subprocess.run(
        [
            "/usr/bin/nice",
            "-n",
            str(inherited_nice - parent_nice),
            str(LAUNCHER),
            sys.executable,
            "-c",
            probe,
            *CAPS,
            "space ; $literal argument",
        ],
        env={**os.environ, **dict.fromkeys(CAPS, "8")},
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    value = json.loads(result.stdout)
    assert value == {
        "nice": max(15, inherited_nice),
        "child_nice": max(15, inherited_nice),
        "env": dict.fromkeys(CAPS, "1"),
        "arg": "space ; $literal argument",
    }


def test_launcher_requires_command_and_preserves_exit_status() -> None:
    usage = subprocess.run([str(LAUNCHER)], capture_output=True, text=True, timeout=10)
    assert usage.returncode == 2
    assert "usage:" in usage.stderr
    result = subprocess.run([str(LAUNCHER), "/bin/sh", "-c", "exit 37"], timeout=10)
    assert result.returncode == 37
