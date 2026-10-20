"""A1/A2 checks for the v2 teacher's portable world exports."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research.t34_exit_b8 import replay
from research.t34_exit_b8.exports import load_world


def _checkpoint() -> Path:
    inventory = json.loads(Path("research/t34_exit_b8/INPUTS_REQUIRED.json").read_text())
    for row in inventory["inputs"]:
        path = Path(row["local_path"])
        if row["name"] == "B8_s3_final.pth" and path.is_file():
            return path
    pytest.skip("the pinned B8/frozen checkpoint inventory is unavailable")


def _export_files(root: Path) -> dict[str, bytes]:
    """Return every world payload and immutable receipt sidecar by relative name."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.glob("world-*.json*"))
        if path.is_file()
    }


def _run_in_child(output: Path, checkpoint: Path) -> None:
    """Run the v2 kernel once in a fresh interpreter with the Mac CPU cap."""
    program = r"""
import json
import os
import sys
from pathlib import Path

os.environ["SNAKE_DQN_DEVICE"] = "cpu"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from research.t34_exit_b8.teacher_v2 import run_worlds

result = run_worlds(
    {
        "checkpoint": sys.argv[2],
        "output_dir": Path(sys.argv[1]),
        "namespace": "t34-v2-a1-process-identity/v1",
        "mixes": ("scripted", "frozen", "mixed"),
        "max_worlds": 3,
        "worlds_per_process": 3,
        "frames": 2,
        "horizon": 300,
        "shadow_depth": 300,
        "beta": 1.0,
        "device": "cpu",
        "seconds": 180,
    }
)
assert result["complete"], result
assert result["worlds"] == 3, result
assert result["shadow_depth"] == 300, result
assert result["label_rule"] == "teacher-stride10-transition-refine5", result
Path(sys.argv[1], "summary.json").write_text(json.dumps(result, sort_keys=True))
"""
    child_env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        child_env[name] = "1"
    child_env["SNAKE_DQN_DEVICE"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-c", program, str(output), str(checkpoint)],
        check=False,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode:
        raise AssertionError(
            "v2 child failed with exit "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def test_v2_exports_are_byte_identical_across_spawned_processes_and_replay(
    tmp_path: Path,
) -> None:
    """A1 compares payloads/receipts; A2 replays every exported mix."""
    checkpoint = _checkpoint()
    roots = [tmp_path / "process-0", tmp_path / "process-1"]
    for root in roots:
        root.mkdir()
        _run_in_child(root, checkpoint)

    first = _export_files(roots[0])
    second = _export_files(roots[1])
    assert sorted(first) == sorted(second)
    assert first == second
    payloads = [path for path in sorted(first) if path.endswith(".json")]
    assert len(payloads) == 3

    for relative in payloads:
        path = roots[0] / relative
        exported = load_world(path)
        assert exported["teacher_v2"]["kernel"] == "shadow-lad-v2"
        assert exported["teacher_v2"]["shadow_depth"] == 300
        assert exported["teacher_v2"]["label_rule"] == "teacher-stride10-transition-refine5"
        assert exported["teacher_v2"]["action_trace_width"] == 6
        assert exported["teacher_v2"]["hash_interval"] == 250
        assert exported["rng_stream_ids"]
        result = replay.replay_world(exported, checkpoint)
        assert result.accepted, f"{relative}: {result.mismatch}"

    # Keep the digest comparison explicit in the test receipt so a future
    # failure identifies a changed byte stream rather than only a row count.
    assert hashlib.sha256(first[payloads[0]]).hexdigest() == hashlib.sha256(
        second[payloads[0]]
    ).hexdigest()
