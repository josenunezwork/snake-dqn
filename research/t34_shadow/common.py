"""Small create-only receipts and hard run bounds shared by measurements."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019")
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def sha(path: Path) -> str:
    """Hash an input without reading an entire checkpoint into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, value: Any) -> None:
    """Publish a receipt without overwriting earlier evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class Budget:
    """Fail closed at the <=30-minute wall bound, including setup."""

    def __init__(self, seconds: float) -> None:
        if not 0 < seconds <= 1800:
            raise ValueError("wall budget must be in (0, 1800]")
        self.started = time.monotonic()
        self.seconds = seconds

    def check(self) -> None:
        """Reject admission of any further step after the deadline."""
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError("measurement wall budget expired")


def provenance() -> dict[str, Any]:
    """Record Mac execution, pinned source closure, and host contention."""
    if platform.system() != "Darwin":
        raise RuntimeError("T34 measurements require the Mac")
    if any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise RuntimeError("all numerical thread limits must explicitly equal one")
    if os.environ.get("SNAKE_DQN_DEVICE") != "cpu":
        raise RuntimeError("T34 requires explicit CPU device")
    batt = subprocess.check_output(["pmset", "-g", "batt"], text=True)
    lid = subprocess.check_output(["ioreg", "-r", "-k", "AppleClamshellState"], text=True)
    if "AC Power" not in batt or '"AppleClamshellState" = No' not in lid:
        raise RuntimeError("T34 requires AC power and open lid")
    paths = list((ROOT / "research/t34_shadow").glob("*.py"))
    paths += [
        ROOT / "docs/research/specs/2026-10-19_t34_shadow_measurements.md",
        ROOT / "docs/research/final_run_20261019/DESIGN_V2_SHADOW.md",
    ]
    for directory in (
        "src",
        "web/backend",
        "research/b8_death_census_20261014",
        "research/redesign_scope_20261007",
        "research/t15_mixed_roster_20261017",
        "research/apex_veto_serving_20261001",
        "research/apex_safety_20260926",
        "research/redesign_m2_20261008",
        "research/t1_commitment_20261015",
        "research/t3_rollout_labels_20261015",
    ):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend((ROOT / "configs").rglob("*.yaml"))
    paths.extend((ROOT / "research/apex_safety_20260926").glob("*.yaml"))
    paths.append(ROOT / "web/backend/served_checkpoint_pins.json")
    return {
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.version,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        "threads": {name: os.environ[name] for name in THREAD_VARS},
        "load_average": os.getloadavg(),
        "host_contention": "Phase 0 and other owner jobs may run",
        "thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True),
        "power": batt,
    }
