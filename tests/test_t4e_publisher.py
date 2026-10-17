"""T4e disk-backed return-bank publication contracts."""

import json

import numpy as np
import pytest

from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout
from research.t4d_return_regression_20261016.bank import ReturnBank
from research.t4e_return_20261016 import publisher
from tests.t14_fixtures import install_train_layout_context


@pytest.fixture(autouse=True)
def _portable_train_layout(monkeypatch):
    install_train_layout_context(monkeypatch)


def _terminal(value: int):
    return {
        "local": np.zeros((1, 7, 31, 31), np.uint8),
        "global": np.zeros((1, 4, 37, 37), np.uint8),
        "scalars": np.full((1, 60), value, np.float32),
    }


def _record(decision: int, *, decisive: bool):
    layout = train_layout()
    state_packed = pack_ref(
        np.zeros((1, 7, 31, 31), np.uint8),
        np.zeros((1, 4, 37, 37), np.uint8),
        layout,
    )[0]
    return {
        "seed": 1234,
        "source_decision": decision,
        "legal": np.array([0, 2], np.int64),
        "state_packed": state_packed,
        "state_scalars": np.full(60, decision, np.float32),
        "death_distance": 99 if decisive else -1,
        "branches": {
            0: {"T": 2, "died": True, "S": 5.0, "L0": 70, "L_T": 0, "terminal": None},
            2: {
                "T": 3,
                "died": False,
                "S": 12.0,
                "L0": 70,
                "L_T": 75,
                "terminal": _terminal(decision + 1),
                "terminal_mask": np.array([True, False, True, False, False, True]),
            },
        },
    }


def _job(index: int):
    return {"plan_index": index, "arm": "B8_s0", "mix": "scripted", "seed": 8000 + index}


def _provenance():
    return {"round": 68, "status": "COLLECTION", "reference": "fixed-fixture"}


def test_publish_per_world_shards_share_exact_embedded_manifest_and_load(tmp_path):
    root = tmp_path / "bank"
    first = publisher.publish_world(root, _job(0), [_record(10, decisive=True)], [], _provenance())
    second = publisher.publish_world(
        root, _job(1), [], [_record(20, decisive=False)], _provenance()
    )
    aggregate = publisher.publish_manifest(root, _provenance())
    assert first["decisive_rows"] == second["outside_rows"] == 2
    assert aggregate["publication"]["rows"] == 4
    assert aggregate["publication"]["decisive_rows"] == 2
    assert aggregate["publication"]["outside_rows"] == 2
    shards = sorted(root.glob("*.npz"))
    with (
        np.load(shards[0], allow_pickle=False) as left,
        np.load(shards[1], allow_pickle=False) as right,
    ):
        assert left["manifest_json"].item() == right["manifest_json"].item()
        assert left["row_weight"].tolist() == [1.0, 1.0]
        assert right["row_weight"].tolist() == [0.5, 0.5]
        assert not left["terminal_mask"][0].any()
    bank = ReturnBank(root, 60)
    assert bank.size == 4
    assert set(aggregate["shards"]) == {path.name for path in shards}


def test_aggregate_refuses_tampered_shard_and_create_only_receipts(tmp_path):
    root = tmp_path / "bank"
    publisher.publish_world(root, _job(0), [_record(10, decisive=True)], [], _provenance())
    with pytest.raises(FileExistsError, match="overwrite"):
        publisher.publish_world(root, _job(0), [_record(10, decisive=True)], [], _provenance())
    shard = next(root.glob("*.npz"))
    with shard.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        publisher.publish_manifest(root, _provenance())


def test_aggregate_rejects_a_different_embedded_provenance(tmp_path):
    root = tmp_path / "bank"
    publisher.publish_world(root, _job(0), [_record(10, decisive=True)], [], _provenance())
    other = {**_provenance(), "round": 69}
    with pytest.raises(ValueError, match="embedded manifest"):
        publisher.publish_manifest(root, other)
    receipt = next((root / "worlds").glob("*.receipt.json"))
    assert json.loads(receipt.read_text())["status"] == "COMPLETE"


def test_spool_round_trips_empty_and_replay_metadata(tmp_path):
    empty = tmp_path / "empty.npz"
    publisher._candidate_spool(empty, [])
    assert publisher.load_spool(empty) == []
    record = _record(10, decisive=True)
    record.update(
        {
            "plan_index": 2,
            "length": 700,
            "q": np.arange(6, dtype=np.float32),
            "fingerprint": "f" * 64,
            "holdout": True,
            "b8_action": 2,
        }
    )
    path = tmp_path / "state.npz"
    publisher._candidate_spool(path, [record])
    loaded = publisher.load_spool(path)
    assert loaded[0]["legal"].tolist() == [0, 2]
    assert loaded[0]["death_distance"] == 99
    assert loaded[0]["fingerprint"] == "f" * 64


def test_selected_staged_publication_makes_a_valid_final_train_bank(tmp_path):
    root, staging = tmp_path / "collection", tmp_path / "collection" / "staging"
    job, record = {**_job(0), "holdout": False}, _record(10, decisive=True)
    staged = publisher.publish_world_family(staging, job, [record], 0, _provenance())
    metadata = {
        **record,
        "plan_index": 0,
        "length": 700,
        "q": np.zeros(6, np.float32),
        "fingerprint": "a" * 64,
        "holdout": False,
        "b8_action": 0,
    }
    publisher._candidate_spool(
        staging / "decisive" / "world-00000-B8_s0-scripted-s8000.npz", [metadata]
    )
    publisher._candidate_spool(staging / "candidates" / "world-00000-B8_s0-scripted-s8000.npz", [])
    phase = {
        "job": job,
        "published": staged,
        "candidate_spool": "candidates/world-00000-B8_s0-scripted-s8000.npz",
    }
    publisher._atomic_json(staging / "phase-a" / "world-00000-B8_s0-scripted-s8000.json", phase)
    published = publisher.publish_selected_staged(
        root, {0: job}, {(0, 10)}, set(), _provenance(), workers=1
    )
    assert publisher.validate_completed_banks(root, published)["train_rows"] == 2


def test_progress_receipts_use_a_create_only_batch_sequence(tmp_path):
    first = publisher.write_progress(tmp_path, [{"key": "a"}], phase="phase_a")
    second = publisher.write_progress(tmp_path, [{"key": "b"}], phase="phase_a")
    paths = sorted((tmp_path / "progress").glob("phase_a-*.json"))
    assert first["keys"] == ["a"]
    assert second["keys"] == ["b"]
    assert [path.name for path in paths] == ["phase_a-00000.json", "phase_a-00001.json"]
