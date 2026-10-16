"""Launch-receipt contracts for the addendum 2 macro-bank replacement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.t4f_return_20261016.launch_check import MACRO_POLICY, verify


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bank(root: Path, *, policy: str = MACRO_POLICY) -> Path:
    root.mkdir(parents=True)
    shard = root / "world-00000.npz"
    np.savez_compressed(shard, rows=np.array([1], dtype=np.int64))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "collection_schema": "t4f-return-bank/v3",
                "schema_version": 2,
                "macro": {
                    "depth": 300,
                    "boost_cost_period_frames": 3,
                    "policy": policy,
                },
                "shards": {shard.name: _sha256(shard)},
                "publication": {"rows": 1},
            }
        )
    )
    return root


def _outside(root: Path) -> Path:
    root.mkdir()
    shard = root / "observations.npz"
    np.savez_compressed(shard, rows=np.array([1], dtype=np.int64))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "t4f-heldout-outside-observations/v1",
                "states": 1,
                "shards": {shard.name: _sha256(shard)},
            }
        )
    )
    return root


def _binding(tmp_path: Path, *, policy: str = MACRO_POLICY) -> Path:
    bank = _bank(tmp_path / "bank", policy=policy)
    outside = _outside(tmp_path / "outside")
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema": "t4f-return-launch-inputs/v1",
                "files": {},
                "training_return_bank": {"path": str(bank), "publication_rows": 1},
                "holdout_return_bank": {"path": str(bank), "publication_rows": 1},
                "heldout_outside": {"path": str(outside), "states": 1},
                "return_weight": 0.0005,
            }
        )
    )
    return binding


def test_launch_check_accepts_v3_direction_preserving_macro(tmp_path: Path) -> None:
    result = verify(_binding(tmp_path))

    assert result["training_rows"] == 1
    assert result["holdout_rows"] == 1


def test_launch_check_rejects_repeated_turn_macro(tmp_path: Path) -> None:
    binding = _binding(tmp_path, policy="repeat_forced_boost_action_k_frames_then_B8_masked_greedy")

    with pytest.raises(ValueError, match="addendum 2 boost geometry"):
        verify(binding)
