"""A1: actual GridBatchSim exports repeat across runs and spawned processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _checkpoint() -> str:
    inventory = json.loads(Path("research/t34_exit_b8/INPUTS_REQUIRED.json").read_text())
    if not all(Path(row["local_path"]).is_file() for row in inventory["inputs"]):
        pytest.skip("the pinned B8/frozen checkpoint inventory is unavailable")
    return next(row["local_path"] for row in inventory["inputs"] if row["name"] == "B8_s3_final.pth")


def test_real_three_mix_exports_match_two_runs_in_two_processes(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    program = r'''
import sys
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from research.t34_exit_b8.trajectory import run_worlds
root = Path(sys.argv[1])
for attempt in range(2):
    result = run_worlds({
        'checkpoint': sys.argv[2], 'output_dir': root / str(attempt),
        'namespace': 't34-exit-a1-process-identity/v1',
        'mixes': ('scripted', 'frozen', 'mixed'), 'max_worlds': 3,
        'worlds_per_process': 3, 'frames': 6, 'horizon': 4, 'seconds': 120,
        'triggers': {'low_length_probability': 1.0, 'max_rewinds': 0},
    })
    assert result['complete']
    assert result['search_batch_peak'] == 3
    assert result['teacher_profile']['peak_active_rows'] >= 3
'''
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["SNAKE_DQN_DEVICE"] = "cpu"
    for process in range(2):
        subprocess.run([sys.executable, "-c", program, str(tmp_path / str(process)), checkpoint],
                       env=env, check=True, timeout=180, capture_output=True, text=True)
    groups = [[path.read_bytes() for path in sorted((tmp_path / str(process) / str(run)).glob("world-*.json"))]
              for process in range(2) for run in range(2)]
    assert len(groups[0]) == 3
    assert groups[1:] == [groups[0]] * 3


def test_real_compaction_keeps_source_action_and_rng_identity() -> None:
    import numpy as np
    import torch

    from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig
    from research.t34_exit_b8.exports import state_hash
    from research.t34_exit_b8.policy import B8Policy
    from research.t34_exit_b8.trajectory import build_world, make_world_rows, prepare_world

    torch.set_num_threads(1)

    class StraightPolicy(B8Policy):
        def _forward(self, net, obs):
            return np.tile(np.array([0, 10, 0, 0, 0, 0], dtype=np.float32), (len(obs["scalars"]), 1))

    checkpoint = _checkpoint()
    rows = make_world_rows({"namespace": "t34-exit-compaction-fixture/v1",
                            "mixes": ("scripted", "frozen", "mixed")}, 3)
    worlds = [build_world(row, checkpoint) for row in rows]
    for world in worlds:
        sim = world.sim
        # A single-segment hero at the top-left corner: left and straight die
        # on their first step; right remains available for compacted rollout.
        sim.bodies[0, 0, int(sim.head_ptr[0, 0])] = [0, 0]
        sim.direction[0, 0] = 0
        sim._rebuild_env_grids(0, food=False)
        prepare_world(world)
    before = [(state_hash(w.sim), w.runtime.sel_state.tobytes()) for w in worlds]
    teacher = ExitTeacher(ExitTeacherConfig(horizon=12), policy=StraightPolicy())
    outcomes = teacher._rollout_merged(worlds, [[0, 1, 2]] * 3)
    for result in outcomes:
        assert result[0].row_steps == result[1].row_steps == 1
        assert not result[0].survives and not result[1].survives
        assert result[2].row_steps > 1
    assert teacher.profile()["compactions"] >= 1
    assert teacher.profile()["peak_active_rows"] == 9
    assert before == [(state_hash(w.sim), w.runtime.sel_state.tobytes()) for w in worlds]
    assert all(not getattr(policy, "live_vetoes", {}) for w in worlds for policy in w.opponents)
