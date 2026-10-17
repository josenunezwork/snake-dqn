"""Fast contracts for T4 v2 bank and expert-iteration preparation tools."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.t3_rollout_labels_20261015 import collect as t3_collect
from research.t3_rollout_labels_20261015.bank import LabelBank
from research.t3_rollout_labels_20261015.bank_engine import (
    label_set,
    outcome_arrays,
    sha256,
    write_world_rows,
)
from research.t3_rollout_labels_20261015.relabel_v2 import relabel
from research.t4_expert_iteration_20261015 import collect_for_student as expert


def test_seed_audit_rejects_generated_exclusion_without_concrete_hit(monkeypatch) -> None:
    namespace = "test-generated-exclusion"
    first = int.from_bytes(hashlib.sha256(f"{namespace}:0".encode()).digest()[:4], "big")
    monkeypatch.setattr(expert, "excluded_seeds", lambda count: {"ni": {first}})
    monkeypatch.setattr(expert, "concrete_seed_hits", lambda *_: set())
    selected, audit = expert._seeds(namespace, 2, (Path("/tmp"),))
    assert first not in selected
    assert audit["rejected"] == [{"seed": first, "roots": ["generated:ni"]}]
    assert audit["excluded_namespace_overlap"] == {"ni": []}


def _source_bank(path: Path) -> None:
    np.savez_compressed(
        path / "bank-00000.npz",
        local=np.zeros((2, 7, 31, 31), np.uint8),
        **{"global": np.zeros((2, 4, 37, 37), np.uint8)},
        scalars=np.zeros((2, 60), np.float32),
        mask=np.ones((2, 6), bool),
        surviving=np.array(
            [[True, False, False, True, True, False], [False, False, False, True, True, False]]
        ),
        action=np.array([0, 3], np.int64),
        weight=np.ones(2, np.float32),
    )


def test_prefer_nonboost_preserves_boost_only_escape() -> None:
    np.testing.assert_array_equal(
        label_set(np.array([True, False, False, True, True, False]), "prefer-nonboost"),
        np.array([True, False, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        label_set(np.array([False, False, False, True, True, False]), "prefer-nonboost"),
        np.array([False, False, False, True, True, False]),
    )


def test_outcome_arrays_encode_null_death_and_branch_endpoints() -> None:
    arrays = outcome_arrays(
        {
            0: {"survives": True, "death_decision": None, "final_length": 701},
            4: {"survives": False, "death_decision": 42, "final_length": 0},
        }
    )
    assert arrays["outcome_survived"].tolist() == [True, False, False, False, False, False]
    assert arrays["outcome_death_frame"].tolist() == [-1, -1, -1, -1, 42, -1]
    assert arrays["outcome_final_length"].tolist() == [701, 0, 0, 0, 0, 0]


def test_outcome_recorded_reaches_merged_bank(tmp_path: Path) -> None:
    outcomes = {
        0: {"survives": True, "death_decision": None, "final_length": 701},
        3: {"survives": False, "death_decision": 18, "final_length": 0},
    }
    row = t3_collect._row(
        {
            "local": np.zeros((1, 7, 31, 31), np.uint8),
            "global": np.zeros((1, 4, 37, 37), np.uint8),
            "scalars": np.zeros((1, 60), np.float32),
            "mask": np.ones((1, 6), bool),
        },
        0,
        np.array([True, False, False, False, False, False]),
        True,
        10,
        11,
        700,
        np.zeros(6, np.float32),
        outcomes=outcomes,
    )
    job, plan, identity = (
        {"arm": "x", "mix": "scripted", "cycle": 0, "index": 0},
        {"seed": 7},
        {"x": 1},
    )
    key = t3_collect._key(job, plan)
    world_path = tmp_path / "worlds" / f"{key}.npz"
    write_world_rows(world_path, [row])
    receipt = {
        "status": "PASS",
        "key": key,
        "batches": 1,
        "dense_candidate_batches": 1,
        "prefilter_skipped_batches": 0,
        "_resume_kind": "current",
        "run_identity": identity,
        "rows_sha256": sha256(world_path),
    }
    t3_collect.merge_world_rows(tmp_path, [receipt], identity, {})
    with np.load(tmp_path / "bank-00000.npz") as bank:
        assert bank["outcome_recorded"].tolist() == [[True, False, False, True, False, False]]
        assert bank["outcome_final_length"].tolist() == [[701, 0, 0, 0, 0, 0]]


def test_relabel_derives_a_new_hash_bound_bank_without_touching_source(tmp_path: Path) -> None:
    source, derived = tmp_path / "source", tmp_path / "derived"
    source.mkdir()
    _source_bank(source)
    before = (source / "bank-00000.npz").read_bytes()
    result = relabel(source, derived)
    assert result["rows"] == 2 and result["rows_changed"] == 1
    assert result["boost_share_after"] < result["boost_share_before"]
    assert (source / "bank-00000.npz").read_bytes() == before
    with np.load(derived / "bank-00000.npz") as shard:
        assert shard["surviving"].tolist() == [
            [True, False, False, False, False, False],
            [False, False, False, True, True, False],
        ]
    assert json.loads((derived / "manifest.json").read_text())["source"]["tree_sha256"]
    assert LabelBank(derived, 60).size == 2


def test_expert_plan_binds_student_as_played_and_continuation(monkeypatch, tmp_path: Path) -> None:
    student = tmp_path / "student.pth"
    student.write_bytes(b"checkpoint")
    # Namespace freshness is covered independently; avoid scanning the developer's
    # artifact volume when this test only exercises student/roster binding.
    monkeypatch.setattr(expert, "fixed_string_hits", lambda _root, _needles: {})
    monkeypatch.setattr(expert, "concrete_seed_hits", lambda _root, _seeds: set())
    monkeypatch.setattr(
        expert,
        "planned_rosters",
        lambda _context, seeds, mixes: [
            {"seed": seed, "mix": mix, "specs": [], "world_index": index, "member_sha256s": []}
            for index, (seed, mix) in enumerate(zip(seeds, mixes))
        ],
    )
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    monkeypatch.setattr(gid, "_context", lambda _arg: {})
    round_plan = expert.plan(
        student, tmp_path / "round", f"t4-test/{tmp_path.name}", 2, ["scripted"]
    )
    assert round_plan["played_policy"] == round_plan["continuation_policy"] == "student"
    assert round_plan["student"]["sha256"]
    assert len(round_plan["seed_audit"]["selected"]) == 2


def test_expert_collect_routes_student_to_play_and_continuation(
    monkeypatch, tmp_path: Path
) -> None:
    student = tmp_path / "student.pth"
    student.write_bytes(b"checkpoint")
    round_plan = {
        "namespace": "t4-test/call-through",
        "student": {"path": str(student), "sha256": sha256(student)},
        "jobs": [{"arm": "student", "mix": "scripted", "cycle": 0, "index": 0}],
        "plans": [{"seed": 4, "mix": "scripted", "specs": []}],
    }
    calls = []

    def worker(job, roster, out, identity, **kwargs):
        calls.append((job, roster, out, identity, kwargs))
        return {"key": "student-scripted-c0-i0-s4", "batches": 0}

    monkeypatch.setattr(expert, "collect_world_death_window", worker)
    monkeypatch.setattr(expert, "merge_world_rows", lambda *_args, **_kwargs: {"rows": 0})
    monkeypatch.setattr(expert, "manifest", lambda *_args, **_kwargs: {})
    result = expert.collect(
        round_plan, tmp_path, record_outcomes=True, label_policy="prefer-nonboost", max_seconds=1
    )
    assert result["receipt_keys"] == ["student-scripted-c0-i0-s4"]
    assert calls[0][3]["student_sha256"] == sha256(student)
    assert calls[0][4] == {
        "record_outcomes": True,
        "label_policy": "prefer-nonboost",
        "hero_checkpoint": student,
    }


def test_expert_collect_rejects_changed_checkpoint_before_worker(
    monkeypatch, tmp_path: Path
) -> None:
    student = tmp_path / "student.pth"
    student.write_bytes(b"old")
    round_plan = {
        "student": {"path": str(student), "sha256": sha256(student)},
        "jobs": [],
        "plans": [],
    }
    student.write_bytes(b"new")
    monkeypatch.setattr(
        expert, "collect_world_death_window", lambda *_args, **_kwargs: pytest.fail("worker")
    )
    with pytest.raises(RuntimeError, match="before collection"):
        expert.collect(
            round_plan, tmp_path, record_outcomes=False, label_policy="surviving-set", max_seconds=1
        )
