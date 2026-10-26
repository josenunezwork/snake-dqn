"""Run only the synthetic annex tests at nice 15 with one process/math thread."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def main() -> None:
    """Pin resources before importing numeric libraries; exec preserves the PID."""
    os.setpriority(os.PRIO_PROCESS, 0, 15)
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    library = str(Path(importlib.util.find_spec("torch").origin).parent / "lib")
    if sys.platform == "darwin" and os.environ.get("DYLD_LIBRARY_PATH") != library:
        environment = dict(os.environ, DYLD_LIBRARY_PATH=library)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], environment)

    import pytest
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (
        os.getpriority(os.PRIO_PROCESS, 0),
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    ) != (15, 1, 1):
        raise RuntimeError("Refuse tests without exact nice15 / intra1 / inter1")
    print(
        "Resource proof: nice=15; torch intra/inter-op=1/1; serial pytest; one Python process",
        flush=True,
    )
    raise SystemExit(
        pytest.main(
            [
                "-q",
                "-o",
                "addopts=",
                "-p",
                "no:xdist",
                "tests/test_bl_residual_f5_r10a_annex.py",
            ]
        )
    )


if __name__ == "__main__":
    main()
