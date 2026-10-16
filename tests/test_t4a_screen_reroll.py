"""Focused contracts for the T4a derived-bank re-roll boundary."""

import json

import numpy as np
import pytest

from research.t4a_screen_20261016 import reroll_t4a


def _bank(tmp_path):
    source = tmp_path / "source"
    derived = tmp_path / "derived"
    source.mkdir()
    derived.mkdir()
    n = 3
    original = np.array(
        [
            [False, True, False, False, False, False],
            [False, False, True, False, False, False],
            [False, False, False, True, False, False],
        ]
    )
    preferred = np.array(
        [
            [False, True, False, False, False, False],
            [False, False, True, False, False, False],
            [False, False, False, False, True, False],
        ]
    )
    actions = np.array([0, 2, 3], np.int64)
    common = {
        "local": np.zeros((n, 7, 31, 31), np.uint8),
        "global": np.zeros((n, 4, 37, 37), np.uint8),
        "scalars": np.zeros((n, 60), np.float32),
        "mask": np.ones((n, 6), bool),
        "action": actions,
    }
    np.savez_compressed(source / "bank-00000.npz", **common, surviving=original)
    np.savez_compressed(
        derived / "bank-00000.npz",
        **common,
        surviving=preferred,
        surviving_original=original,
    )
    provenance = source / "bank-00000.provenance.jsonl"
    provenance.write_text(
        "\n".join(
            json.dumps({"row": row, "source_key": f"w{row}", "decision": row, "frame": row})
            for row in range(n)
        )
        + "\n"
    )
    (source / "manifest.json").write_text(json.dumps({"schema": "t3-label-bank/v1"}))
    files = {
        path.name: reroll_t4a.sha256(path)
        for path in (source / "manifest.json", source / "bank-00000.npz", provenance)
    }
    (derived / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "t4-derived-label-bank/v2",
                "label_policy": "prefer-nonboost",
                "shards": {"bank-00000.npz": reroll_t4a.sha256(derived / "bank-00000.npz")},
                "source": {"path": str(source), "files": files, "tree_sha256": "bound"},
            }
        )
    )
    return derived


def test_sample_binds_v2_and_source_manifests_and_preserves_two_sides(tmp_path) -> None:
    bank = _bank(tmp_path)
    sample = reroll_t4a.prepare_sample(bank, tmp_path / "sample.json", size=3, seed=8)

    assert sample["schema"] == "t4a-reroll-sample/v1"
    by_row = {entry["row"]: entry for entry in sample["entries"]}
    assert by_row[0]["side"] == "decisive"
    assert by_row[0]["target_actions"] == [1]
    assert by_row[1]["side"] == "counter"
    assert by_row[1]["target_actions"] == [2]
    assert by_row[2]["side"] == "counter"
    assert by_row[2]["target_actions"] == [3]


def test_sample_rejects_preferred_label_drift(tmp_path) -> None:
    bank = _bank(tmp_path)
    sample = tmp_path / "sample.json"
    reroll_t4a.prepare_sample(bank, sample, size=2)
    derived = bank / "bank-00000.npz"
    with np.load(derived, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["surviving"][0] = np.array([True, False, False, False, False, False])
    np.savez_compressed(derived, **arrays)
    with pytest.raises(ValueError, match="shard hash mismatch"):
        reroll_t4a.prepare_sample(bank, sample, size=2)


def test_v2_rejects_source_observation_drift(tmp_path) -> None:
    bank = _bank(tmp_path)
    source = tmp_path / "source" / "bank-00000.npz"
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["scalars"][1, 0] = 1.0
    np.savez_compressed(source, **arrays)
    with pytest.raises(ValueError, match="source bank hash mismatch"):
        reroll_t4a.prepare_sample(bank, tmp_path / "sample.json", size=2)
