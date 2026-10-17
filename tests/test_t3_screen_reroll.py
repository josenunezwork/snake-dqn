"""Focused contracts for the T3 post-hoc reroll sample boundary."""

import json

import numpy as np
import pytest

from research.t3_screen_20261015 import reroll_t3
from tests.t14_fixtures import install_train_layout_context


def _bank(tmp_path, decisive=3, packed=False):
    bank = tmp_path / "bank"
    bank.mkdir()
    n = decisive
    arrays = {
        "local": np.zeros((n, 7, 31, 31), np.uint8),
        "global": np.zeros((n, 4, 37, 37), np.uint8),
        "scalars": np.zeros((n, 60), np.float32),
        "mask": np.ones((n, 6), bool),
        "surviving": np.tile(np.array([False, True, False, False, False, False]), (n, 1)),
        "action": np.zeros(n, np.int64),
    }
    if packed:
        from research.redesign_m3c_20261008.obs_pack import train_layout

        arrays.pop("local")
        arrays.pop("global")
        arrays["packed"] = np.zeros((n, train_layout().nbytes), np.uint8)
    np.savez_compressed(bank / "bank-00000.npz", **arrays)
    (bank / "bank-00000.provenance.jsonl").write_text(
        "\n".join(
            json.dumps({"row": i, "source_key": "w", "decision": i, "frame": i, "decisive": True})
            for i in range(n)
        )
        + "\n"
    )
    shard = bank / "bank-00000.npz"
    provenance = bank / "bank-00000.provenance.jsonl"
    (bank / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "t3-label-bank/v1",
                "shards": {shard.name: reroll_t3.sha256(shard)},
                "files": {
                    shard.name: reroll_t3.sha256(shard),
                    provenance.name: reroll_t3.sha256(provenance),
                },
            }
        )
    )
    return bank


def test_sample_is_seeded_and_reusable_with_manifest_binding(tmp_path) -> None:
    bank = _bank(tmp_path)
    sample = tmp_path / "sample.json"
    first = reroll_t3.prepare_sample(bank, sample, size=2, seed=9)
    second = reroll_t3.prepare_sample(bank, sample, size=2, seed=99)
    assert first == second
    assert len(first["entries"]) == 2
    assert {entry["source_key"] for entry in first["entries"]} == {"w"}


def test_sample_rejects_bank_manifest_drift(tmp_path) -> None:
    bank = _bank(tmp_path)
    sample = tmp_path / "sample.json"
    reroll_t3.prepare_sample(bank, sample, size=2)
    (bank / "manifest.json").write_text(json.dumps({"schema": "t3-label-bank/v1", "changed": True}))
    with pytest.raises(ValueError, match="bank manifest"):
        reroll_t3.prepare_sample(bank, sample, size=2)


def test_sample_requires_enough_decisive_exported_rows(tmp_path) -> None:
    with pytest.raises(ValueError, match="decisive exported"):
        reroll_t3.prepare_sample(_bank(tmp_path, decisive=1), tmp_path / "sample.json", size=2)


def test_packed_export_is_unpacked_with_the_canonical_layout(monkeypatch, tmp_path) -> None:
    install_train_layout_context(monkeypatch)
    bank = _bank(tmp_path, packed=True)
    sample = reroll_t3.prepare_sample(bank, tmp_path / "sample.json", size=2)
    assert len(sample["entries"]) == 2
