"""Small, hermetic contracts for the T23 evaluator wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.t23_r0_screen_20261017 import evaluate as subject


def _world(seed: int, mix: str) -> dict:
    return {
        "training_seed": 7101,
        "bank_phase": "initial",
        "world_seed": seed,
        "mix": mix,
        "world_semantic_sha256": f"{seed:064x}",
        "roster_semantic_sha256": f"{seed + 1:064x}",
        "opponents": [["scripted", "greedy_food"]],
    }


def test_bank_worlds_accepts_t23_nested_bank_shape() -> None:
    rows = [
        _world(index + 1, mix)
        for index, mix in enumerate(
            mix for mix, count in subject.MIX_COUNTS.items() for _ in range(count)
        )
    ]
    bank = {"worlds": {"7101": {"initial": rows}}}
    selected = subject._bank_worlds(bank, 7101, "initial")
    assert len(selected) == 128
    assert {row["mix"] for row in selected} == set(subject.MIX_COUNTS)


def test_shard_parser_assigns_deterministic_disjoint_worlds() -> None:
    worlds = [{"index": index} for index in range(10)]
    assert [row["index"] for row in subject._shard_worlds(worlds, 0, 3)] == [0, 3, 6, 9]
    assert [row["index"] for row in subject._shard_worlds(worlds, 1, 3)] == [1, 4, 7]
    assert [row["index"] for row in subject._shard_worlds(worlds, 2, 3)] == [2, 5, 8]


def test_completed_shard_requires_both_paired_arms(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    identity = {"world_count": 2}
    path.write_text(
        json.dumps(
            {
                "schema": "t23-eval-shard/v1",
                "status": "completed",
                "clean_exit": True,
                "identity": identity,
                "rows": [{}, {}],
            }
        )
    )
    assert not subject.valid_completed_shard(path, identity)
    complete_rows = [{"schema": "t23-eval-row/v1"}] * 4
    path.write_text(
        json.dumps(
            {
                "schema": "t23-eval-shard/v1",
                "status": "completed",
                "clean_exit": True,
                "identity": identity,
                "rows": complete_rows,
                "rows_sha256": __import__("hashlib")
                .sha256(subject.canonical(complete_rows).encode())
                .hexdigest(),
            }
        )
    )
    assert subject.valid_completed_shard(path, identity)


def test_checkpoint_metadata_rejects_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"t23")
    assert subject.checkpoint_metadata(checkpoint)["bytes"] == 3
    try:
        subject.checkpoint_metadata(checkpoint, required_sha256="0" * 64)
    except ValueError as error:
        assert "SHA mismatch" in str(error)
    else:
        raise AssertionError("wrong checkpoint hash was accepted")


def _training_fixture(tmp_path: Path, **summary_overrides: object) -> Path:
    root, seed_dir = tmp_path / "drive", tmp_path / "drive" / "seed-7101"
    seed_dir.mkdir(parents=True)
    summary = {
        "args": {"seed": "7101"},
        "transitions": 8_000_256,
        "halt": None,
        "stopped_for_time": False,
        "actor_exitcodes": [0] * 6,
    }
    summary.update(summary_overrides)
    (seed_dir / "summary.json").write_text(json.dumps(summary))
    rows = [
        {
            "transitions": step,
            "td": 1.0,
            "anchor": 1.0,
            "qmax": 1.0,
            "greedy_boost_rate": 0.0,
            "greedy_boost_rate_lt_700": 0.0,
        }
        for step in range(100_000, 8_000_001, 100_000)
    ]
    (seed_dir / "log.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (seed_dir / "manifest.jsonl").write_text("{}\n")
    (root / "run.log").write_text(
        "commit a049e92 gpu NVIDIA A100-SXM4-80GB start now\n"
        "seed 7101 start now\nseed 7101 exit 0 now\n"
    )
    return root


def test_training_provenance_accepts_complete_actual_style_windows(tmp_path: Path) -> None:
    receipt = subject.training_provenance(_training_fixture(tmp_path), 7101)
    assert receipt["clean"] and not receipt["tripwire_triggered"]


@pytest.mark.parametrize(
    "override", [{"transitions": 100}, {"actor_exitcodes": [0] * 5}, {"stopped_for_time": True}]
)
def test_training_provenance_rejects_forged_or_incomplete_summary(
    tmp_path: Path, override: dict
) -> None:
    receipt = subject.training_provenance(_training_fixture(tmp_path, **override), 7101)
    assert not receipt["clean"]


def test_training_provenance_tripwire_is_typed_not_infrastructure_invalid(tmp_path: Path) -> None:
    receipt = subject.training_provenance(_training_fixture(tmp_path, halt="action collapse"), 7101)
    assert receipt["clean"] and receipt["tripwire_triggered"]


def test_training_provenance_rejects_bad_gpu_nan_and_missing_window(tmp_path: Path) -> None:
    root = _training_fixture(tmp_path)
    (root / "run.log").write_text(
        "commit x gpu L4 start now\nseed 7101 start now\nseed 7101 exit 0 now\n"
    )
    telemetry = root / "seed-7101" / "log.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "transitions": 100_000,
                "td": float("nan"),
                "anchor": 1.0,
                "qmax": 1.0,
                "greedy_boost_rate": 0.0,
                "greedy_boost_rate_lt_700": 0.0,
            }
        )
        + "\n"
    )
    receipt = subject.training_provenance(root, 7101)
    assert not receipt["clean"] and receipt["tripwire_triggered"]
