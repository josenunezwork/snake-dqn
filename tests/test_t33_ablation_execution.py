"""Real CLI proof that calibration worker scheduling preserves semantic rows."""

import json
import os
import subprocess
import sys

import pytest

from research.t33_batched_teacher.ablation import B8


def semantic(value):
    """Discard only elapsed measurements; retain decisions, scores and settings."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def test_real_ablation_serial_and_two_worker_rows_match(tmp_path) -> None:
    if not B8.is_file():
        pytest.skip("hash-pinned local B8 inventory unavailable")
    outputs = []
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"
    for workers in (1, 2):
        path = tmp_path / f"workers-{workers}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.t33_batched_teacher.ablation",
                "--smoke",
                "--decisions",
                "3",
                "--workers",
                str(workers),
                "--horizons",
                "1",
                "2",
                "--rollouts",
                "1",
                "--reference-horizon",
                "2",
                "--reference-rollouts",
                "1",
                "--out",
                str(path),
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        outputs.append(json.loads(path.read_text()))
    assert semantic(outputs[0]["rows"]) == semantic(outputs[1]["rows"])
