"""Real child scheduling/settings and non-bypassable shell entrypoint checks."""

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
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PYTHONDONTWRITEBYTECODE",
)


@pytest.mark.parametrize("inherited_nice", [15, 19])
def test_launcher_caps_environment_and_child_priority(inherited_nice: int) -> None:
    # Exercise the production spawn implementation in a separate interpreter;
    # no admission locks are replaced in the shell entrypoint.
    probe = """
import json, os, subprocess, sys
from research.compute.factory_interim_launcher import _spawn
code = '''import json, os, sys, numpy
print(json.dumps({'nice': os.getpriority(os.PRIO_PROCESS, 0),
    'env': {k: os.environ[k] for k in sys.argv[1:]}, 'device': os.environ['SNAKE_DQN_DEVICE']}))'''
child = _spawn([sys.executable, '-c', code, *sys.argv[1:]])
raise SystemExit(child.wait())
"""
    parent_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if inherited_nice < parent_nice:
        pytest.skip("cannot raise inherited scheduler priority")
    result = subprocess.run(
        [
            "/usr/bin/nice",
            "-n",
            str(inherited_nice - parent_nice),
            sys.executable,
            "-c",
            probe,
            *CAPS,
        ],
        env={**os.environ, **dict.fromkeys(CAPS, "8"), "SNAKE_DQN_DEVICE": "mps"},
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "nice": inherited_nice,
        "env": dict.fromkeys(CAPS, "1"),
        "device": "cpu",
    }


def test_launcher_requires_class_and_refuses_unreviewed_compute() -> None:
    usage = subprocess.run([str(LAUNCHER)], capture_output=True, text=True, timeout=10)
    assert usage.returncode == 2
    assert "usage:" in usage.stderr
    for args in (
        [sys.executable, "-m", "pytest"],
        ["--class", "light", "--", "/bin/bash", "-c", "echo bypass"],
        ["--class", "test", "--", sys.executable, "-m", "pytest"],
    ):
        result = subprocess.run([str(LAUNCHER), *args], capture_output=True, text=True, timeout=10)
        assert result.returncode == 2
        assert "bypass" not in result.stdout


def test_actual_reviewed_test_settings_are_cpu_serial() -> None:
    import torch

    assert os.getpriority(os.PRIO_PROCESS, 0) >= 15
    assert os.environ["SNAKE_DQN_DEVICE"] == "cpu"
    assert {name: os.environ[name] for name in CAPS} == dict.fromkeys(CAPS, "1")
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1


def test_light_shell_route_preserves_command_exit_status() -> None:
    result = subprocess.run(
        [str(LAUNCHER), "--class", "light", "--", "cat", "/missing-interim-fixture-input"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
