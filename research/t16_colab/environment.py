"""Create the smoke's Python 3.12 environment independently of the Colab kernel."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable


def create_environment(venv: Path, run_owned: Callable) -> tuple[str, dict]:
    """Bootstrap uv from the kernel, then select and verify its managed Python 3.12.

    Every command uses the notebook's existing deadline and owned-process cleanup.
    The kernel is used only to bootstrap/invoke uv, never for the pinned workload.
    """
    run_owned([sys.executable, "-m", "pip", "install", "uv"])
    uv = [sys.executable, "-m", "uv"]
    run_owned([*uv, "python", "install", "3.12"])
    run_owned([*uv, "venv", "--python", "3.12", "--managed-python", "--seed", str(venv)])
    python = str(venv / "bin" / "python")
    probe = run_owned(
        [
            python,
            "-c",
            "import json, platform, sys; "
            "print(json.dumps({'python': platform.python_version(), "
            "'python_executable': sys.executable, 'isolated': sys.prefix != sys.base_prefix}))",
        ],
        capture_output=True,
    )
    info = json.loads(probe.stdout)
    if not info["python"].startswith("3.12.") or not info["isolated"]:
        raise RuntimeError(f"Expected isolated uv Python 3.12, got {info}")
    return python, info
