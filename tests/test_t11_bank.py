"""Focused invariant tests for T11 LOCAL30 derived-bank production."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.t3_rollout_labels_20261015.bank import LabelBank
from research.t11_local30_20261017 import bank


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, *, counters: int = 4) -> Path:
    """Write a tiny hash-bound v2 source and its aligned original receipts."""
    source = tmp_path / "v2"
    original = tmp_path / "original"
    worlds = original / "worlds"
    worlds.mkdir(parents=True)
    n = 8
    action = np.zeros(n, np.int64)
    original_surviving = np.zeros((n, 6), bool)
    original_surviving[:, 1] = True
    original_surviving[:counters, 0] = True
    surviving = original_surviving.copy()
    # Model the v2 prefer-nonboost relabel: B8 action 0 survived originally,
    # but the training target excludes it.  Counter identity must still use
    # surviving_original rather than this selected-label set.
    surviving[:counters, 0] = False
    # Rows 0/4, 1/5, 2/6 and 3/7 share a raw-Q stratum so that the
    # deterministic pairing contract has one counter for each decisive row.
    q = np.repeat(np.array([1, 2, 3, 4, 1, 2, 3, 4], np.float32)[:, None], 6, axis=1)
    arrays = {
        "scalars": np.zeros((n, 60), np.float32),
        "mask": np.ones((n, 6), bool),
        "surviving": surviving,
        "surviving_original": original_surviving,
        "action": action,
        "weight": np.ones(n, np.float32),
        "q": q,
        "length": np.array([100, 100, 450, 450, 100, 100, 450, 450], np.int64),
        "packed": np.zeros((n, 3064), np.uint8),
    }
    source.mkdir()
    np.savez_compressed(source / "bank-00000.npz", **arrays)
    rows = []
    files: dict[str, str] = {}
    for row in range(n):
        key = f"world-{row}"
        receipt = worlds / f"{key}.json"
        receipt.write_text(json.dumps({"death_decision": 20 if row >= counters else None}))
        files[f"worlds/{key}.json"] = _digest(receipt)
        rows.append({"row": row, "source_key": key, "decision": 0, "decisive": row >= counters})
    provenance = original / "bank-00000.provenance.jsonl"
    provenance.write_text("".join(json.dumps(row) + "\n" for row in rows))
    files[provenance.name] = _digest(provenance)
    manifest = {
        "schema": "t4-derived-label-bank/v2",
        "shards": {"bank-00000.npz": _digest(source / "bank-00000.npz")},
        "source": {"path": str(original), "files": files},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    return source


def _refresh_world_hashes(source: Path) -> None:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original = Path(manifest["source"]["path"])
    for receipt in (original / "worlds").glob("*.json"):
        manifest["source"]["files"][f"worlds/{receipt.name}"] = _digest(receipt)
    manifest_path.write_text(json.dumps(manifest))


def test_derive_selects_local30_and_matches_stable_strata(tmp_path: Path) -> None:
    source = _source(tmp_path)
    out = tmp_path / "out"
    receipt = bank.derive(source, out)
    assert receipt["family_counts"] == {"decisive": 4, "counter": 4}
    assert receipt["selection"]["selected_decisive_source_rows"] == [4, 5, 6, 7]
    assert receipt["selection"]["selected_counter_source_rows"] == [0, 1, 2, 3]
    with np.load(out / "bank-00000.npz", allow_pickle=False) as payload:
        assert payload["source_row"].tolist() == [4, 5, 6, 7, 0, 1, 2, 3]
        assert payload["row_family"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        assert payload["decision"].tolist() == [0] * 8
        assert payload["death_decision"].tolist() == [20] * 4 + [-1] * 4
    from research.redesign_m3c_20261008.obs_pack import train_layout

    loaded = LabelBank(out, 60, pack=train_layout(), strict_shards=True)
    assert loaded.size == 8
    assert loaded._decisive_indices.size == loaded._counter_indices.size == 4
    assert not loaded.surviving[4:, 0].any()
    assert loaded.surviving_original[4:, 0].all()


def test_derive_uses_inclusive_zero_to_thirty_gap_and_original_counter_side(tmp_path: Path) -> None:
    source = _source(tmp_path)
    original = Path(json.loads((source / "manifest.json").read_text())["source"]["path"])
    for row, death in {4: 0, 5: 30, 6: 31, 7: 20}.items():
        (original / f"worlds/world-{row}.json").write_text(json.dumps({"death_decision": death}))
    _refresh_world_hashes(source)
    receipt = bank.derive(source, tmp_path / "out")
    assert receipt["family_counts"] == {"decisive": 3, "counter": 3}
    assert receipt["selection"]["selected_decisive_source_rows"] == [4, 5, 7]
    with np.load(tmp_path / "out/bank-00000.npz", allow_pickle=False) as payload:
        assert payload["row_family"].tolist() == [0, 0, 0, 1, 1, 1]
        assert payload["source_row"].tolist() == [4, 5, 7, 0, 1, 3]


def test_derive_fails_closed_when_a_joint_stratum_lacks_counter_support(tmp_path: Path) -> None:
    source = _source(tmp_path, counters=3)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="counter support shortage"):
        bank.derive(source, out)
    assert not out.exists()


def test_derive_rejects_changed_original_world_receipt(tmp_path: Path) -> None:
    source = _source(tmp_path)
    original = Path(json.loads((source / "manifest.json").read_text())["source"]["path"])
    (original / "worlds/world-0.json").write_text('{"death_decision": 1}')
    with pytest.raises(ValueError, match="world receipt hash"):
        bank.derive(source, tmp_path / "out")
