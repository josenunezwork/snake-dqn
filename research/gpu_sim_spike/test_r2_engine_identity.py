"""Engine-level R2 byte identity on long and wrapped fixtures, all three rosters.

The old engine is the pinned I0 runtime source itself (``968f0276``), executed
in a subprocess from a checkout of that commit (default: the read-only
qualified tree; override with ``R2_PINNED_TREE``).  The candidate runs this
tree with R2 enabled in debug mode, so every cached value is also re-derived
with the pinned full-capacity code.  Per tick, the persistent state digest
(``integration_engine.snapshot``: every simulator field including ring padding,
vector carry, policy and collector) and the ordered digests of every observable
output must be identical.  Runs single-threaded on the Mac CPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIN = "968f02765f7c86ca27d2947347896e2fe4e6ce23"
PINNED_TREE = Path(
    os.environ.get(
        "R2_PINNED_TREE", "/Users/josenunez/Projects/ml/snake-dqn-gpuint-qualified-03.noindex"
    )
)
SAVED = Path(
    os.environ.get("R2_SAVED_SNAKES", "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
)
CHECKPOINT = SAVED / "b8_s3_t7_20261016.pth"
SCRIPT = Path(__file__).with_name("r2_identity.py")
FIXTURES = ((), ("--wrap",), ("--exceptions", "--wrap"))

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.is_file() or sys.platform != "darwin",
    reason="Mac CPU identity check needs the primary-checkout checkpoints",
)


def _pinned_head(tree: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()


def _record(tree: Path, mix: str, mode: str, flags: tuple[str, ...], tmp: Path) -> dict:
    output = tmp / f"{tree.name}-{mix}-{mode}-{'-'.join(flags) or 'plain'}.json"
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        SNAKE_DQN_DEVICE="cpu",
        SNAKE_CKPT_DIR=str(SAVED),
        GPU_SIM_VECTOR_FLOOD="static",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.pop("GPU_SIM_R2_DEBUG", None)
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--root",
            str(tree),
            "--checkpoint",
            str(CHECKPOINT),
            "--mix",
            mix,
            "--worlds",
            "3",  # exception fixture: world 2 holds the forced body death
            "--mode",
            mode,
            "--ticks",
            "3",
            "--output",
            str(output),
            *flags,
        ],
        cwd=tmp,
        env=env,
        check=True,
        timeout=600,
    )
    return json.loads(output.read_text())


def _comparable(value: dict) -> list:
    return [(tick["state_sha256"], tick["calls"]) for tick in value["ticks"]]


@pytest.mark.parametrize("flags", FIXTURES, ids=("long", "wrapped", "exceptions-wrapped"))
@pytest.mark.parametrize("mix", ("frozen", "mixed", "scripted"))
def test_r2_engine_is_byte_identical_to_pinned_source(mix, flags, tmp_path):
    if not (PINNED_TREE / "research/gpu_sim_spike/device_sim.py").is_file():
        pytest.skip("pinned I0 runtime checkout is not available")
    assert _pinned_head(PINNED_TREE) == PIN
    old = _record(PINNED_TREE, mix, "pin", flags, tmp_path)
    new = _record(ROOT, mix, "r2-debug", flags, tmp_path)
    assert len(old["ticks"]) == 3 and _comparable(old) == _comparable(new)
    assert new["r2_stats"]["debug_checks"] > 0 and new["r2_stats"]["hits"] > 0
    if "--exceptions" in flags:
        assert sum(tick["deaths"] for tick in old["ticks"]) > 0


@pytest.mark.parametrize("mix", ("frozen", "mixed", "scripted"))
def test_r2_disabled_tree_is_byte_identical_to_pinned_source(mix, tmp_path):
    """The I arm of the R2 packet (this tree, R2 off) is the pinned engine."""
    if not (PINNED_TREE / "research/gpu_sim_spike/device_sim.py").is_file():
        pytest.skip("pinned I0 runtime checkout is not available")
    flags = ("--exceptions", "--wrap")
    old = _record(PINNED_TREE, mix, "pin", flags, tmp_path)
    new = _record(ROOT, mix, "off", flags, tmp_path)
    assert _comparable(old) == _comparable(new)
