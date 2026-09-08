"""Tests for copy-first SQLite replay metadata migration."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from src.scripts import migrate_replay_metadata as migration_module
from src.scripts.migrate_replay_metadata import migrate_replay_metadata, sqlite_file_hashes


def _open_wal_fixture(path: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(
        "CREATE TABLE memories_standard "
        "(id INTEGER PRIMARY KEY, state BLOB, done INTEGER, next_action_mask INTEGER)"
    )
    writer.execute("CREATE TABLE replay_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    writer.execute(
        "INSERT INTO replay_metadata VALUES (?, ?)",
        ("generation.gamma", json.dumps(0.99)),
    )
    writer.execute("INSERT INTO memories_standard VALUES (1, ?, 0, NULL)", (b"first",))
    writer.commit()

    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    assert reader.execute("SELECT COUNT(*) FROM memories_standard").fetchone()[0] == 1

    writer.execute("INSERT INTO memories_standard VALUES (2, ?, 1, 63)", (b"wal",))
    writer.commit()
    return writer, reader


def _metadata(path: Path) -> dict:
    conn = sqlite3.connect(path)
    try:
        return {
            key: json.loads(value)
            for key, value in conn.execute("SELECT key, value FROM replay_metadata")
        }
    finally:
        conn.close()


def test_backup_includes_committed_wal_rows_with_active_reader(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "migrated.db"
    writer, reader = _open_wal_fixture(source)
    try:
        result = migrate_replay_metadata(source, destination)

        migrated = sqlite3.connect(destination)
        try:
            assert migrated.execute("SELECT COUNT(*) FROM memories_standard").fetchone()[0] == 2
            assert migrated.execute(
                "SELECT id, state FROM memories_standard ORDER BY id"
            ).fetchall() == [(1, b"first"), (2, b"wal")]
            assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            migrated.close()
        assert reader.execute("SELECT COUNT(*) FROM memories_standard").fetchone()[0] == 1
        assert result["experience_counts"] == {"memories_standard": 2}
        assert result["fallback_mask_count"] == 1
        assert result["destination_sha256"]
        assert result["backup_snapshot_file_hashes"]["main"]
        assert result["source_file_hashes_at_snapshot_validation"] == result["source_file_hashes"]
        metadata = _metadata(destination)
        assert metadata["replay.verification"]["status"] == "unverified_legacy"
        assert metadata["replay.verification"]["fallback_mask_count"] == 1
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize(("done", "expected_fallback"), [(False, 1), (True, 0)])
def test_migration_selects_populated_legacy_table_when_standard_is_empty(
    tmp_path: Path, done: bool, expected_fallback: int
) -> None:
    source = tmp_path / "interrupted_legacy_migration.db"
    destination = tmp_path / "migrated.db"
    writer = sqlite3.connect(source)
    writer.execute("CREATE TABLE memories_standard (id INTEGER PRIMARY KEY)")
    writer.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, done INTEGER, next_action_mask INTEGER)"
    )
    writer.execute("INSERT INTO memories VALUES (1, ?, NULL)", (int(done),))
    writer.commit()
    writer.close()

    result = migrate_replay_metadata(source, destination)

    assert result["experience_counts"] == {"memories_standard": 0, "memories": 1}
    assert result["fallback_mask_count"] == expected_fallback
    metadata = _metadata(destination)
    assert metadata["replay.verification"]["fallback_mask_count"] == expected_fallback
    migrated = sqlite3.connect(destination)
    try:
        assert migrated.execute("SELECT COUNT(*) FROM memories_standard").fetchone()[0] == 0
        assert migrated.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    finally:
        migrated.close()


def test_migration_rejects_multiple_populated_replay_tables(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.db"
    destination = tmp_path / "migrated.db"
    writer = sqlite3.connect(source)
    writer.execute(
        "CREATE TABLE memories_standard (id INTEGER PRIMARY KEY, done INTEGER, "
        "next_action_mask INTEGER)"
    )
    writer.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY, done INTEGER, next_action_mask INTEGER)"
    )
    writer.execute("INSERT INTO memories_standard VALUES (1, 1, NULL)")
    writer.execute("INSERT INTO memories VALUES (1, 1, NULL)")
    writer.commit()
    writer.close()

    with pytest.raises(RuntimeError, match="multiple populated"):
        migrate_replay_metadata(source, destination)

    assert not destination.exists()


def test_migration_prefers_standard_over_archival_legacy_copy(tmp_path: Path) -> None:
    source = tmp_path / "auto_migrated.db"
    destination = tmp_path / "migrated.db"
    writer = sqlite3.connect(source)
    writer.execute(
        "CREATE TABLE memories_standard (id INTEGER PRIMARY KEY, done INTEGER, "
        "next_action_mask INTEGER)"
    )
    writer.execute(
        "CREATE TABLE memories_legacy (id INTEGER PRIMARY KEY, done INTEGER, "
        "next_action_mask INTEGER)"
    )
    writer.execute("INSERT INTO memories_standard VALUES (1, 0, NULL)")
    writer.execute("INSERT INTO memories_legacy VALUES (1, 0, NULL)")
    writer.commit()
    writer.close()

    result = migrate_replay_metadata(source, destination)

    assert result["fallback_mask_count"] == 1
    assert result["experience_counts"] == {
        "memories_standard": 1,
        "memories_legacy": 1,
    }


def test_dry_run_writes_nothing_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "migrated.db"
    writer, reader = _open_wal_fixture(source)
    try:
        before = sqlite_file_hashes(source)
        result = migrate_replay_metadata(source, destination, dry_run=True)

        assert result["status"] == "dry_run"
        assert not destination.exists()
        assert sqlite_file_hashes(source) == before
    finally:
        reader.close()
        writer.close()


def test_failure_keeps_source_and_prior_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "migrated.db"
    writer, reader = _open_wal_fixture(source)
    destination.write_bytes(b"prior-valid-destination")
    source_hashes = sqlite_file_hashes(source)
    prior_destination = destination.read_bytes()

    try:
        with pytest.raises(FileExistsError, match="already exist"):
            migrate_replay_metadata(source, destination)

        assert destination.read_bytes() == prior_destination
        assert sqlite_file_hashes(source) == source_hashes
        assert not list(tmp_path.glob(".migrated.db.*.tmp"))
    finally:
        reader.close()
        writer.close()


def test_prepublish_failure_and_directory_fsync_failure_leave_no_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.db"
    writer, reader = _open_wal_fixture(source)
    try:
        prepublish_destination = tmp_path / "prepublish.db"

        def fail_before_publish(_temp_path: Path) -> None:
            raise RuntimeError("injected prepublish failure")

        with pytest.raises(RuntimeError, match="prepublish"):
            migrate_replay_metadata(
                source,
                prepublish_destination,
                before_publish=fail_before_publish,
            )
        assert not prepublish_destination.exists()

        fsync_destination = tmp_path / "fsync.db"
        real_fsync_directory = migration_module._fsync_directory
        calls = 0

        def fail_first_directory_fsync(path: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected directory fsync failure")
            real_fsync_directory(path)

        monkeypatch.setattr(migration_module, "_fsync_directory", fail_first_directory_fsync)
        with pytest.raises(OSError, match="directory fsync"):
            migrate_replay_metadata(source, fsync_destination)
        assert calls == 2
        assert not fsync_destination.exists()
    finally:
        reader.close()
        writer.close()


def test_concurrent_source_writer_is_detected_before_publication(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "migrated.db"
    writer, reader = _open_wal_fixture(source)

    def append_after_backup(_temp_path: Path) -> None:
        writer.execute("INSERT INTO memories_standard VALUES (3, ?, 0, 63)", (b"late",))
        writer.commit()

    try:
        with pytest.raises(RuntimeError, match="source changed"):
            migrate_replay_metadata(source, destination, before_publish=append_after_backup)
        assert not destination.exists()
        assert writer.execute("SELECT COUNT(*) FROM memories_standard").fetchone()[0] == 3
    finally:
        reader.close()
        writer.close()


def test_refuses_real_wal_destination_and_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    source_writer, source_reader = _open_wal_fixture(source)
    destination = tmp_path / "destination.db"
    destination_writer = sqlite3.connect(destination)
    destination_writer.execute("PRAGMA journal_mode=WAL")
    destination_writer.execute("CREATE TABLE keep_me (value TEXT)")
    destination_writer.execute("INSERT INTO keep_me VALUES ('original')")
    destination_writer.commit()
    destination_writer.execute("BEGIN IMMEDIATE")
    before = sqlite_file_hashes(destination)
    try:
        assert Path(f"{destination}-wal").exists()
        with pytest.raises(FileExistsError, match="destination already exists"):
            migrate_replay_metadata(source, destination)
        assert sqlite_file_hashes(destination) == before
        assert destination_writer.execute("SELECT value FROM keep_me").fetchone()[0] == "original"
    finally:
        destination_writer.rollback()
        destination_writer.close()
        source_reader.close()
        source_writer.close()


def test_asserted_facts_remain_unverified(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "migrated.db"
    writer, reader = _open_wal_fixture(source)
    try:
        migrate_replay_metadata(
            source,
            destination,
            asserted_facts={"target.n_step": 3, "mask.schema": "claimed"},
        )
        metadata = _metadata(destination)

        assert metadata["replay.verification"]["status"] == "unverified_legacy"
        assert "target.n_step" in metadata["replay.verification"]["missing_fields"]
        assert metadata["replay.legacy_asserted_facts"] == {
            "target.n_step": 3,
            "mask.schema": "claimed",
        }
    finally:
        reader.close()
        writer.close()


def test_refuses_in_place_or_implicit_destination_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    sqlite3.connect(source).close()
    destination = tmp_path / "destination.db"
    destination.write_bytes(b"keep")

    with pytest.raises(ValueError, match="different"):
        migrate_replay_metadata(source, source)
    with pytest.raises(FileExistsError, match="already exists"):
        migrate_replay_metadata(source, destination)


def test_direct_cli_help_and_dry_run_work_without_pythonpath(tmp_path: Path) -> None:
    script = Path(migration_module.__file__).resolve()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr

    source = tmp_path / "source.db"
    writer, reader = _open_wal_fixture(source)
    destination = tmp_path / "destination.db"
    try:
        dry_run = subprocess.run(
            [sys.executable, str(script), str(source), str(destination), "--dry-run"],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert dry_run.returncode == 0, dry_run.stderr
        assert json.loads(dry_run.stdout)["status"] == "dry_run"
        assert not destination.exists()
    finally:
        reader.close()
        writer.close()
