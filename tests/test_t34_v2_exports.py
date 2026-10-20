"""V2 metadata survives the original immutable world/replay envelope."""

from pathlib import Path

import pytest

from research.t34_exit_b8.exports import load_world, write_world
from tests.test_t34_exit_replay import _checkpoint, _result


def test_v2_label_roles_and_lad_provenance_roundtrip(tmp_path: Path) -> None:
    result = _result(_checkpoint(tmp_path))
    result["teacher_v2"] = {"rule": "teacher-stride10-transition-refine5", "shadow_depth": 300}
    result["labels"][0].update(
        label_role="decisive",
        shadow_id="seed17/fork0",
        shadow_confirmed=True,
        t_star=0,
        a_star=2,
        F=[0, 3],
        q=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        lad={"doom": 20, "global_last_avoidable_proven": False},
        searched=True,
        decisive=True,
        kl_anchor_only=False,
    )
    path = tmp_path / "world.json"
    write_world(path, result)
    loaded = load_world(path)
    assert loaded["teacher_v2"] == result["teacher_v2"]
    for key in (
        "label_role",
        "shadow_id",
        "shadow_confirmed",
        "t_star",
        "a_star",
        "F",
        "q",
        "lad",
        "searched",
        "decisive",
        "kl_anchor_only",
    ):
        assert loaded["labels"][0][key] == result["labels"][0][key]


def test_v2_metadata_cannot_leak_host_paths(tmp_path: Path) -> None:
    result = _result(_checkpoint(tmp_path))
    result["teacher_v2"] = {"source": "/Users/private/checkpoint"}
    with pytest.raises(ValueError, match="filesystem paths"):
        write_world(tmp_path / "world.json", result)
