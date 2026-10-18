import hashlib
import json
from pathlib import Path

from research.q0q1_20261017 import q1_evaluate


def test_q1_completed_shard_is_single_identity_and_digest_bound(tmp_path) -> None:
    path = tmp_path / "x.json"
    identity = {"world_count": 1}
    rows = [{"schema": "q1-eval-row/v1"}]
    path.write_text(
        json.dumps(
            {
                "schema": "q1-eval-shard/v1",
                "status": "completed",
                "clean_exit": True,
                "identity": identity,
                "rows": rows,
                "rows_sha256": hashlib.sha256(q1_evaluate.canonical(rows).encode()).hexdigest(),
            }
        )
    )
    assert q1_evaluate.valid_completed_shard(path, identity)
    assert not q1_evaluate.valid_completed_shard(path, {"world_count": 2})


def test_q1_shards_are_disjoint() -> None:
    worlds = [{"mix": "scripted", "world_seed": n} for n in range(10)]
    assert [x["world_seed"] for x in q1_evaluate.shard_worlds(worlds, 0, 3)] == [0, 3, 6, 9]


def test_q1_candidate_provenance_fails_closed_on_missing_parity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final.pth"
    checkpoint.write_bytes(b"final")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema": "q1-training-provenance/v1",
                "arm": "A",
                "training_seed": 261701,
                "endpoint": 8_000_000,
                "clean_exit": True,
                "checkpoint_sha256": q1_evaluate.sha256(checkpoint),
                "tripwire_triggered": False,
                "environment_sha256": "e" * 64,
                "runtime_parity_receipt_sha256": "p" * 64,
            }
        )
    )
    import pytest

    with pytest.raises(ValueError, match="provenance"):
        q1_evaluate.validate_candidate_provenance(provenance, "A", 261701, checkpoint)
