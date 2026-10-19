"""Start macOS test processes with one OpenMP runtime before test collection."""

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest


def pytest_sessionstart(session: pytest.Session) -> None:
    """Route both torch and sklearn to torch's libomp, only in pytest children.

    macOS wheels bundle separate libomp copies. DYLD_LIBRARY_PATH must be set
    before Python starts; changing it after importing either package is too late.
    Relaunching also covers serial pytest and subprocesses started by tests.
    No installed wheel, research source, or caller environment is modified.
    """
    if sys.platform != "darwin" or hasattr(session.config, "workerinput"):
        return
    spec = find_spec("torch")
    if spec is None or spec.origin is None:
        return
    library_dir = Path(spec.origin).parent / "lib"
    if not (library_dir / "libomp.dylib").is_file():
        return
    current = os.environ.get("DYLD_LIBRARY_PATH", "")
    if current.split(os.pathsep)[0] == str(library_dir):
        return
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(library_dir) + (os.pathsep + current if current else "")
    print("macOS pytest: restarting with a single OpenMP runtime", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *session.config.invocation_params.args],
        cwd=session.config.invocation_params.dir,
        env=env,
        check=False,
    )
    pytest.exit("macOS OpenMP-isolated pytest finished", returncode=result.returncode)
