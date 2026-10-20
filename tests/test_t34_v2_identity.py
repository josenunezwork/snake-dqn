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


def _run_in_child(output: Path, checkpoint: Path, *, searched: bool = False) -> None:
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

searched = sys.argv[3] == "searched"
for repeat in range(2):
    output = Path(sys.argv[1]) / f"repeat-{repeat}"
    result = run_worlds(
        {
            "checkpoint": sys.argv[2],
            "output_dir": output,
            "namespace": (
                "t34-v2-amend2-preflight/benchmark/03"
                if searched else "t34-v2-a1-process-identity/v1"
            ),
            "mixes": ("scripted", "frozen", "mixed"),
            "max_worlds": 2 if searched else 3,
            "worlds_per_process": 2 if searched else 3,
            "frames": 500 if searched else 2,
            "horizon": 300,
            "shadow_depth": 300,
            "beta": 1.0,
            "device": "cpu",
            "seconds": 180,
        }
    )
    assert result["complete"], result
    assert result["worlds"] == (2 if searched else 3), result
    if searched:
        assert sum(mix['searched'] for mix in result['mixes'].values()) >= 1, result
    assert result["shadow_depth"] == 300, result
    assert result["label_rule"] == "bisection-radius3-prepared-cache", result
    output.joinpath("summary.json").write_text(json.dumps(result, sort_keys=True))
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
        [
            sys.executable,
            "-c",
            program,
            str(output),
            str(checkpoint),
            "searched" if searched else "anchors",
        ],
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

    export_sets = [
        _export_files(root / f"repeat-{repeat}") for root in roots for repeat in range(2)
    ]
    first = export_sets[0]
    payloads = [path for path in sorted(first) if path.endswith(".json")]
    assert len(payloads) == 3
    assert all(len(path_set) == len(first) for path_set in export_sets)
    assert all(path_set == first for path_set in export_sets[1:])

    for repeat, root in enumerate(roots):
        for attempt in range(2):
            export_root = root / f"repeat-{attempt}"
            for relative in payloads:
                path = export_root / relative
                exported = load_world(path)
                assert len(exported["labels"]) >= 2, f"{relative}: labels were not regenerated"
                assert exported["teacher_v2"]["kernel"] == "shadow-lad-v2"
                assert exported["teacher_v2"]["shadow_depth"] == 300
                assert exported["teacher_v2"]["label_rule"] == "bisection-radius3-prepared-cache"
                assert exported["teacher_v2"]["work"]["reconstruction_row_steps"] == 0
                assert exported["teacher_v2"]["action_trace_width"] == 6
                assert exported["teacher_v2"]["hash_interval"] == 250
                assert exported["rng_stream_ids"]
                for label in exported["labels"]:
                    if label.get("searched"):
                        assert label["ranked_actions"]
                        assert label["outcome_kind"] in {
                            "bare_survives",
                            "horizon_rescue",
                            "all_die_delay",
                        }
                        assert label["horizon"] == 300
                        if label["outcome_kind"] == "all_die_delay":
                            assert label["delay_only"] is True
                            assert label["decisive"] is False
                            assert label["label_weight"] == pytest.approx(0.0)
                result = replay.replay_world(exported, checkpoint)
                assert (
                    result.accepted
                ), f"process={repeat} repeat={attempt} {relative}: {result.mismatch}"

    # Keep the digest comparison explicit in the test receipt so a future
    # failure identifies a changed byte stream rather than only a row count.
    assert (
        hashlib.sha256(first[payloads[0]]).hexdigest()
        == hashlib.sha256(export_sets[-1][payloads[0]]).hexdigest()
    )


def test_h300_searched_exports_match_across_processes_and_replay(tmp_path: Path) -> None:
    """Repeat the reused two-world development smoke that reaches a real doom.

    This verifies the complete H300 search metadata, not an efficacy estimate.
    The bank is deliberately reused and excluded from campaign inference.
    """
    checkpoint = _checkpoint()
    roots = [tmp_path / "search-process-0", tmp_path / "search-process-1"]
    for root in roots:
        root.mkdir()
        _run_in_child(root, checkpoint, searched=True)
    sets = [_export_files(root / f"repeat-{i}") for root in roots for i in range(2)]
    assert all(files == sets[0] for files in sets[1:])
    assert len([path for path in sets[0] if path.endswith(".json")]) == 2
    for root in roots:
        for repeat in range(2):
            exports = [
                load_world(p) for p in sorted((root / f"repeat-{repeat}").glob("world-*.json"))
            ]
            assert sum(d["teacher_v2"]["shadow_searches"] for d in exports) >= 1
            assert any(row.get("searched") for d in exports for row in d["labels"])
            for exported in exports:
                result = replay.replay_world(exported, checkpoint)
                assert result.accepted, result.mismatch


def test_real_doom_batch_keeps_frozen_carry_and_source_identity() -> None:
    """Two real frozen/mixed source worlds share one compacted branch bucket."""
    import random
    from collections import defaultdict

    import numpy as np

    from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig
    from research.t34_exit_b8.policy import B8Policy, PolicyEvaluation
    from research.t34_exit_b8.teacher_v2 import (
        _build_doom_cache,
        _search_dooms,
        _shadow_stepper,
        _step,
    )
    from research.t34_exit_b8.trajectory import build_world, make_world_rows, prepare_world
    from research.t34_shadow.shadow import ShadowTrajectory, full_digest

    class StraightPolicy(B8Policy):
        def evaluate(self, world):
            return PolicyEvaluation(
                q=np.array([0, 10, 0, 0, 0, 0], dtype=np.float32),
                legal=np.asarray(world.sim.get_legal_action_mask()[0, 0], dtype=bool),
            )

        def evaluate_rows(self, sim, rows, net):
            return np.tile(np.array([0, 10, 0, 0, 0, 0], dtype=np.float32), (len(rows), 1))

    teacher = ExitTeacher(ExitTeacherConfig(horizon=4), policy=StraightPolicy("cpu"))
    rows = make_world_rows({"namespace": "t34-v2-real-doom/v1", "mixes": ("frozen", "mixed")}, 2)
    states = []
    for row in rows:
        world = build_world(row, _checkpoint())
        for _ in range(2):
            _step(world, teacher, 1)
        sim = world.sim
        sim.bodies[0, 0, int(sim.head_ptr[0, 0])] = [0, 0]
        sim.seg_count[0, 0] = sim.length[0, 0] = 1
        sim.direction[0, 0] = 0
        sim._rebuild_env_grids(0, food=False)
        prepare_world(world)
        assert any(getattr(policy, "live_vetoes", {}) for policy in world.opponents)
        rng = random.Random(row["world_seed"]).getstate()
        shadow = ShadowTrajectory(4, checkpoint="fixture", stepper=_shadow_stepper(teacher))
        shadow.fork(world, global_rng_state=rng)
        shadow.catch_up()
        assert len(shadow.frames) == 1 and not shadow.frames[-1].hero_alive
        states.append(
            {
                "world": world,
                "decision": 0,
                "seed": row["world_seed"],
                "shadow_id": 0,
                "shadow": shadow,
                "rng": rng,
                "work": defaultdict(int),
                "stages": defaultdict(float),
                "pending_labels": {},
                "pending_actions": {},
                "searches": 0,
                "search_bucket_peak": 0,
            }
        )
    before = [full_digest(state["world"], checkpoint="fixture") for state in states]
    for state in states:
        state["doom_cache"] = _build_doom_cache(state, teacher, float("inf"))
    _search_dooms(states, teacher, branch_bucket_size=12)
    assert before == [full_digest(state["world"], checkpoint="fixture") for state in states]
    for state in states:
        assert state["pending_actions"] == {0: 2}
        row = state["pending_labels"][0]
        assert row["decisive"] and row["a_star"] == 2
        assert 1 in row["F"] and 2 not in row["F"]
        assert len(row["per_action"]) >= 3
        assert row["per_action"]["1"]["row_steps"] == 0
        assert row["per_action"]["1"]["cached_control"]
        assert all(value["horizon"] == 4 for value in row["per_action"].values())
        assert state["work"]["reconstruction_row_steps"] == 1
    assert teacher.profile()["peak_active_rows"] >= 4
