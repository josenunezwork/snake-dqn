import hashlib
import json
import time
from pathlib import Path

from research.q0q1_20261017 import q1_evaluate


def _pool_initializer() -> None:
    """Pickleable spawn initializer for deterministic-pool coverage."""


def _delayed_row(world: dict[str, int]) -> dict[str, object]:
    """Finish deliberately out of order without changing the deterministic row."""
    time.sleep(0.01 * (3 - world["world_seed"]))
    return {"world_seed": world["world_seed"], "mass_integral": world["world_seed"] * 1.25}


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


def test_q1_pool_merges_spawn_results_in_serial_plan_order_and_resumes(tmp_path: Path) -> None:
    """The fixed merge makes serial and two spawned workers receipt-byte identical."""
    worlds = [
        {"world_seed": number, "world_semantic_sha256": f"{number:064x}"} for number in range(4)
    ]
    binding = {
        "identity": {"test": "fixed-order"},
        "world_hashes": [x["world_semantic_sha256"] for x in worlds],
    }
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial = q1_evaluate._execute_ordered(
        worlds,
        worker=_delayed_row,
        initializer=_pool_initializer,
        initargs=(),
        workers=1,
        partials_dir=serial_dir,
        binding=binding,
        progress_seconds=60,
    )
    parallel = q1_evaluate._execute_ordered(
        worlds,
        worker=_delayed_row,
        initializer=_pool_initializer,
        initargs=(),
        workers=2,
        partials_dir=parallel_dir,
        binding=binding,
        progress_seconds=60,
    )
    assert q1_evaluate.canonical(parallel) == q1_evaluate.canonical(serial)
    assert [row["world_seed"] for row in parallel] == [0, 1, 2, 3]
    assert (parallel_dir / "manifest.json").read_bytes() == (
        serial_dir / "manifest.json"
    ).read_bytes()

    resumed = q1_evaluate._execute_ordered(
        worlds,
        worker=_delayed_row,
        initializer=_pool_initializer,
        initargs=(),
        workers=2,
        partials_dir=parallel_dir,
        binding=binding,
        progress_seconds=60,
    )
    assert q1_evaluate.canonical(resumed) == q1_evaluate.canonical(serial)


def test_resume_keeps_the_complete_manifest_without_rewriting_existing_units(tmp_path, monkeypatch):
    worlds = [{"world_seed": number} for number in range(3)]
    directory = tmp_path / "partials"
    binding = {"test": "atomic-resume"}
    q1_evaluate._execute_ordered(
        worlds,
        worker=_delayed_row,
        initializer=_pool_initializer,
        initargs=(),
        workers=1,
        partials_dir=directory,
        binding=binding,
    )
    original = (directory / "manifest.json").read_bytes()

    def reject_write(*args, **kwargs):
        raise AssertionError("a fully resumed manifest must never be truncated or rewritten")

    monkeypatch.setattr(q1_evaluate, "_write", reject_write)
    q1_evaluate._execute_ordered(
        worlds,
        worker=reject_write,
        initializer=_pool_initializer,
        initargs=(),
        workers=1,
        partials_dir=directory,
        binding=binding,
    )
    assert (directory / "manifest.json").read_bytes() == original


def test_completed_worlds_export_to_precreated_durable_directory(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    local = tmp_path / "local"
    rows = q1_evaluate._execute_ordered(
        [{"world_seed": 2}],
        worker=_delayed_row,
        initializer=_pool_initializer,
        initargs=(),
        workers=1,
        partials_dir=local,
        binding={"test": "durable"},
        export_dir=export,
    )
    assert rows[0]["world_seed"] == 2
    assert (local / "manifest.json").read_bytes() == (export / "manifest.json").read_bytes()
    assert (local / "world-0000.json").read_bytes() == (export / "world-0000.json").read_bytes()


def test_resume_exports_all_retained_units_before_fresh_work(tmp_path):
    worlds = [{"world_seed": number} for number in range(3)]
    local = tmp_path / "local"
    binding = {"test": "retained-export"}
    q1_evaluate._execute_ordered(
        worlds,
        worker=_delayed_row,
        initializer=_pool_initializer,
        initargs=(),
        workers=1,
        partials_dir=local,
        binding=binding,
    )
    export = tmp_path / "new-export"
    export.mkdir()
    rows = q1_evaluate._execute_ordered(
        worlds,
        worker=_delayed_row,
        initializer=_pool_initializer,
        initargs=(),
        workers=1,
        partials_dir=local,
        binding=binding,
        export_dir=export,
    )
    assert q1_evaluate._load_partials(export / "manifest.json", binding, 3) == dict(enumerate(rows))


def test_default_workers_reserves_one_core_and_caps_at_ten(monkeypatch):
    for cores, expected in ((None, 1), (1, 1), (8, 7), (12, 10), (18, 10)):
        monkeypatch.setattr(q1_evaluate.os, "cpu_count", lambda: cores)
        assert q1_evaluate.default_workers() == expected
