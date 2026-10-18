import hashlib, json
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
