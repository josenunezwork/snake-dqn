"""T34 shadow batching identity checks against the frozen serial teacher.

The fixture modules are loaded through the hash-pinned independent loader in
``profile_shadow_batching``.  Candidate and oracle runs receive different
world objects and policy instances; only the portable input rows are shared.
World payloads and their completion/checksum sidecars are compared as bytes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pytest

from research.t34_exit_b8 import replay
from research.t34_exit_b8 import teacher_v2 as candidate_teacher
from research.t34_exit_b8.exports import load_world
from research.t34_exit_b8.profile_shadow_batching import serial_module
from research.t34_exit_b8.trajectory import build_world, make_world_rows, prepare_world


def _checkpoint() -> Path:
    """Return the pinned B8 checkpoint or skip when test inputs are absent."""
    import json

    inventory = json.loads(Path("research/t34_exit_b8/INPUTS_REQUIRED.json").read_text())
    for row in inventory["inputs"]:
        path = Path(row["local_path"])
        if row["name"] == "B8_s3_final.pth" and path.is_file():
            return path
    pytest.skip("the pinned B8/frozen checkpoint inventory is unavailable")


def _world_bytes(root: Path) -> dict[str, bytes]:
    """Read every immutable world payload and sidecar by relative filename."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.glob("world-*.json*"))
        if path.is_file()
    }


def _assert_export_identity(serial_root: Path, candidate_root: Path, expected: int) -> None:
    """Compare payload and completion/checksum bytes, with explicit digests."""
    serial = _world_bytes(serial_root)
    candidate = _world_bytes(candidate_root)
    assert len([name for name in serial if name.endswith(".json")]) == expected
    assert candidate == serial
    for name, payload in serial.items():
        assert hashlib.sha256(payload).hexdigest() == hashlib.sha256(candidate[name]).hexdigest()


def _assert_replay_clean(root: Path, checkpoint: Path) -> None:
    """Replay every normal anchor export and require zero mismatches."""
    for path in sorted(root.glob("world-*.json")):
        result = replay.replay_world(load_world(path), checkpoint)
        assert result.accepted, f"{path.name}: {result.mismatch}"


def _run_pair(
    tmp_path: Path,
    checkpoint: Path,
    rows: list[dict[str, Any]],
    *,
    frames: int,
    horizon: int,
    shadow_depth: int,
    policy_factory: Callable[[ModuleType], Any] | None = None,
    world_builder_factory: Callable[[ModuleType, Any], Callable[[Any, Path], Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the frozen oracle and candidate with otherwise identical config."""
    oracle = serial_module()
    serial_policy = policy_factory(oracle) if policy_factory is not None else None
    serial_teacher = oracle.ExitTeacher(
        oracle.ExitTeacherConfig(horizon=horizon),
        device="cpu",
        policy=serial_policy,
    )
    serial_builder = (
        world_builder_factory(oracle, serial_teacher) if world_builder_factory else None
    )
    serial_config = {
        "checkpoint": checkpoint,
        "output_dir": tmp_path / "serial",
        "rows": rows,
        "worlds_per_process": len(rows),
        "frames": frames,
        "horizon": horizon,
        "shadow_depth": shadow_depth,
        "seconds": 180,
        "device": "cpu",
        "teacher": serial_teacher,
    }
    if serial_builder is not None:
        serial_config["world_builder"] = serial_builder
    serial_result = oracle.run_worlds(serial_config)

    candidate_policy = policy_factory(candidate_teacher) if policy_factory is not None else None
    candidate_instance = candidate_teacher.ExitTeacher(
        candidate_teacher.ExitTeacherConfig(horizon=horizon),
        device="cpu",
        policy=candidate_policy,
    )
    candidate_builder = (
        world_builder_factory(candidate_teacher, candidate_instance)
        if world_builder_factory
        else None
    )
    candidate_config = {
        "checkpoint": checkpoint,
        "output_dir": tmp_path / "candidate",
        "rows": rows,
        "worlds_per_process": len(rows),
        "frames": frames,
        "horizon": horizon,
        "shadow_depth": shadow_depth,
        "seconds": 180,
        "device": "cpu",
        "teacher": candidate_instance,
    }
    if candidate_builder is not None:
        candidate_config["world_builder"] = candidate_builder
    candidate_result = candidate_teacher.run_worlds(candidate_config)
    return serial_result, candidate_result


def test_anchor_exports_match_frozen_serial_for_all_three_mixes(tmp_path: Path) -> None:
    """Scripted, frozen, and mixed roots preserve every exported byte."""
    checkpoint = _checkpoint()
    rows = make_world_rows(
        {
            "namespace": "t34-v2-a1-process-identity/v1",
            "mixes": ("scripted", "frozen", "mixed"),
        },
        3,
    )
    serial_result, candidate_result = _run_pair(
        tmp_path,
        checkpoint,
        rows,
        frames=2,
        horizon=300,
        shadow_depth=300,
    )

    assert serial_result["complete"] and candidate_result["complete"]
    assert serial_result["worlds"] == candidate_result["worlds"] == 3
    assert {row["mix"] for row in rows} == {"scripted", "frozen", "mixed"}
    assert serial_result["mixes"] == candidate_result["mixes"]
    _assert_export_identity(tmp_path / "serial", tmp_path / "candidate", expected=3)
    _assert_replay_clean(tmp_path / "serial", checkpoint)
    _assert_replay_clean(tmp_path / "candidate", checkpoint)


def _straight_policy_factory(module: ModuleType) -> Any:
    """Build a constant fp32 policy through the module under test/oracle."""
    if module is candidate_teacher:
        from research.t34_exit_b8 import policy as policy_module
    else:
        policy_module = sys.modules["_t34_frozen_policy"]
    policy_base = policy_module.B8Policy
    evaluation_type = policy_module.PolicyEvaluation
    q = np.asarray([0, 10, 10, 0, 10, 0], dtype=np.float32)

    class StraightPolicy(policy_base):
        def evaluate(self, world: Any) -> Any:
            legal = np.asarray(world.sim.get_legal_action_mask()[0, 0], dtype=bool)
            return evaluation_type(q.copy(), legal)

        def evaluate_rows(self, sim: Any, rows: np.ndarray, net: Any) -> np.ndarray:
            del sim, net
            return np.tile(q, (len(rows), 1))

        def evaluate_worlds(self, worlds: list[Any], *, bucket_size: int = 64) -> list[Any]:
            del bucket_size
            return [self.evaluate(world) for world in worlds]

    return StraightPolicy("cpu")


def _doom_world_builder(module: ModuleType, teacher: Any) -> Callable[[Any, Path], Any]:
    """Prepare the same real GridBatchSim corner doom used by the v2 identity test."""

    def build(row: Any, checkpoint: Path) -> Any:
        world = build_world(row, checkpoint)
        for _ in range(2):
            module._step(world, teacher, 1)
        sim = world.sim
        sim.bodies[0, 0, int(sim.head_ptr[0, 0])] = [0, 0]
        sim.seg_count[0, 0] = sim.length[0, 0] = 1
        sim.direction[0, 0] = 0
        sim._rebuild_env_grids(0, food=False)
        prepare_world(world)
        return world

    return build


def test_h300_doom_search_exports_match_serial_with_tied_rescue(tmp_path: Path) -> None:
    """Real-source H300 search preserves ties, rescue ranking, and all bytes."""
    checkpoint = _checkpoint()
    rows = make_world_rows(
        {
            "namespace": "t34-v2-amend2-preflight/benchmark/03",
            "mixes": ("scripted", "frozen", "mixed"),
        },
        2,
    )

    def policy_factory(module: ModuleType) -> Any:
        return _straight_policy_factory(module)

    def builder_factory(module: ModuleType, teacher: Any) -> Callable[[Any, Path], Any]:
        return _doom_world_builder(module, teacher)

    serial_result, candidate_result = _run_pair(
        tmp_path,
        checkpoint,
        rows,
        frames=500,
        horizon=300,
        shadow_depth=300,
        policy_factory=policy_factory,
        world_builder_factory=builder_factory,
    )

    assert serial_result["complete"] and candidate_result["complete"]
    assert serial_result["worlds"] == candidate_result["worlds"] == 2
    assert sum(item["searched"] for item in serial_result["mixes"].values()) >= 1
    assert serial_result["mixes"] == candidate_result["mixes"]
    _assert_export_identity(tmp_path / "serial", tmp_path / "candidate", expected=2)

    labels = []
    for path in sorted((tmp_path / "candidate").glob("world-*.json")):
        import json

        labels.extend(json.loads(path.read_text())["labels"])
    searched = [item for item in labels if item.get("searched")]
    assert searched, "the H300 fixture must exercise the doom/search path"
    assert any(item.get("outcome_kind") == "horizon_rescue" for item in searched)
    rescue = next(item for item in searched if item.get("outcome_kind") == "horizon_rescue")
    assert rescue["ranked_actions"][0] == 2
    assert rescue["a_star"] == 2
