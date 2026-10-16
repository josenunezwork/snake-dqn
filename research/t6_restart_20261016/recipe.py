"""Fixed T6 command construction; importing this module never starts a run."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
RUNS = ARTIFACTS / "redesign-t6-mac-20261016"
STORE = ARTIFACTS / "t4b-snapshot-restart-20261016/full"


def command(seed: int, *, smoke: bool = False) -> list[str]:
    """Return the registered MPS recipe with only the smoke budget/output changed."""
    if seed not in range(10, 15):
        raise ValueError("T6 seeds are 10 through 14")
    return [
        str(REPO / "venv/bin/python"),
        str(REPO / "research/redesign_m3c_20261008/train_m3c.py"),
        "--seed",
        str(seed),
        "--init",
        str(ARTIFACTS / f"redesign-m4b-mac-20261013/B8_s{seed - 10}/final.pth"),
        "--out",
        str(RUNS / (f"smoke_200k_s{seed}" if smoke else f"RESTART_s{seed}")),
        "--demo-data",
        str(ARTIFACTS / "redesign-m2b-20261008/data"),
        "--transitions",
        "200000" if smoke else "8000000",
        "--device",
        "mps",
        "--actors",
        "2",
        "--worlds",
        "128",
        "--segment-frames",
        "5000",
        "--reward",
        "mass",
        "--mass-kappa",
        "1",
        "--kill-scale",
        "0",
        "--gamma-start",
        "0.998",
        "--gamma-end",
        "0.998",
        "--gamma-hold",
        "0",
        "--gamma-anneal-end",
        "1",
        "--eps-end-at",
        "1",
        "--mixed-frac",
        "0.25",
        "--anchor-mode",
        "v8margin",
        "--anchor-hold",
        "0",
        "--anchor-floor-at",
        "1",
        "--anchor-floor",
        "0.1",
        "--onpolicy-labels",
        "--label-frac",
        "1",
        "--td-huber-delta",
        "1.0",
        "--obs-version",
        "bv8",
        "--v61-backend",
        "numba",
        "--opp-forward",
        "per-group",
        "--learner-fast",
        "--batch-scale",
        "2",
        "--obs-pack",
        "--tight-telemetry",
        "--checkpoint-every",
        "2000000",
        "--restart-bank",
        str(STORE),
        "--restart-frac",
        "0.2",
        "--boost-halt-rate",
        "0.02",
        "--boost-halt-low-length-rate",
        "0.01",
    ]


def environment() -> dict[str, str]:
    """Pin numerical threads and MPS, clearing optional inherited trainer overrides."""
    import os

    env = {key: value for key, value in os.environ.items() if not key.startswith("M3C_")}
    env.update(PYTHONPATH=str(REPO), SNAKE_DQN_DEVICE="mps", PYTHONUNBUFFERED="1")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    return env
