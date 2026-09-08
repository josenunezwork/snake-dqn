#!/usr/bin/env python3
"""Copy a legacy replay DB and attach explicit unverified provenance."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote

# Direct ``python src/scripts/...`` execution starts with ``src/scripts`` on
# sys.path. Add the repository root before importing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.runtime_contract import canonical_digest  # noqa: E402
from src.data.replay_contract import (  # noqa: E402
    REPLAY_CONTRACT_DIGEST_KEY,
    REPLAY_CONTRACT_KEY,
    replay_sqlite_hashes,
    unverified_legacy_metadata,
)


def sqlite_file_hashes(path: Path) -> dict[str, str]:
    """Hash durable SQLite content files (main and current WAL) separately.

    The SHM file contains reader-lock state and changes when a read-only backup
    connection attaches. It is neither durable data nor an immutable input.
    """
    return replay_sqlite_hashes(path)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _experience_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    counts = {}
    for table in ("memories_standard", "memories", "memories_legacy"):
        if table in tables:
            counts[table] = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    return counts


def _populated_replay_table(experience_counts: Mapping[str, int]) -> str:
    standard_count = experience_counts.get("memories_standard", 0)
    unmigrated_count = experience_counts.get("memories", 0)
    archive_count = experience_counts.get("memories_legacy", 0)
    if standard_count > 0 and unmigrated_count > 0:
        raise RuntimeError(
            "Source has multiple populated active replay experience tables: "
            "memories_standard, memories"
        )
    if standard_count > 0:
        return "memories_standard"
    if unmigrated_count > 0 and archive_count > 0:
        raise RuntimeError(
            "Source has multiple populated legacy replay experience tables: "
            "memories, memories_legacy"
        )
    if unmigrated_count > 0:
        return "memories"
    if archive_count > 0:
        return "memories_legacy"
    if not any(experience_counts.values()):
        raise RuntimeError("Source has no replay experience rows")
    raise RuntimeError("Source has no recognized populated replay experience table")


def _fallback_mask_count(conn: sqlite3.Connection, table: str) -> int:
    columns = _table_columns(conn, table)
    if "next_action_mask" not in columns:
        predicate = "done = 0" if "done" in columns else "1=1"
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {predicate}').fetchone()[0])
    predicate = "next_action_mask IS NULL"
    if "done" in columns:
        predicate += " AND done = 0"
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {predicate}').fetchone()[0])


def _read_metadata(conn: sqlite3.Connection) -> dict[str, Any]:
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='replay_metadata'"
    ).fetchone()
    if table is None:
        return {}
    return {
        str(key): json.loads(value)
        for key, value in conn.execute("SELECT key, value FROM replay_metadata")
    }


def _write_metadata(conn: sqlite3.Connection, metadata: Mapping[str, Any]) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS replay_metadata " "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT OR REPLACE INTO replay_metadata (key, value) VALUES (?, ?)",
        [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_temporary_sqlite_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


def _destination_sqlite_files(path: Path) -> list[Path]:
    """Return existing destination main/WAL/SHM paths."""
    return [
        candidate
        for suffix in ("", "-wal", "-shm")
        if (candidate := Path(f"{path}{suffix}")).exists()
    ]


def migrate_replay_metadata(
    source: str | Path,
    destination: str | Path,
    *,
    dry_run: bool = False,
    asserted_facts: Mapping[str, Any] | None = None,
    before_publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Backup ``source`` to ``destination`` and tag it unverified.

    The source is opened read-only and never modified. The backup, metadata
    transaction, integrity check and row-count comparison all happen on a
    temporary sibling. Only a fully fsynced copy is atomically published.
    ``before_publish`` is a test hook used to prove failure isolation.
    """
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("source and destination must be different paths")
    if not source_path.is_file():
        raise FileNotFoundError(f"Replay source does not exist: {source_path}")
    existing_destination_files = _destination_sqlite_files(destination_path)
    if existing_destination_files:
        names = ", ".join(path.name for path in existing_destination_files)
        raise FileExistsError(f"Replay destination already exists: {names}")

    facts = {} if asserted_facts is None else dict(asserted_facts)
    canonical_digest(facts)
    source_hashes_before = sqlite_file_hashes(source_path)
    source_conn = _connect_read_only(source_path)
    try:
        source_conn.execute("BEGIN")
        source_counts = _experience_counts(source_conn)
        if not source_counts:
            raise RuntimeError("Source has no recognized replay experience table")
        replay_table = _populated_replay_table(source_counts)
        fallback_count = _fallback_mask_count(source_conn, replay_table)
        source_metadata = _read_metadata(source_conn)
        if REPLAY_CONTRACT_KEY in source_metadata or REPLAY_CONTRACT_DIGEST_KEY in source_metadata:
            raise RuntimeError("Source already contains replay.contract metadata")
        legacy_metadata = unverified_legacy_metadata(
            source_metadata,
            fallback_mask_count=fallback_count,
            asserted_facts=facts,
        )
        result: dict[str, Any] = {
            "status": "dry_run" if dry_run else "migrated",
            "source": str(source_path),
            "destination": str(destination_path),
            "source_file_hashes": source_hashes_before,
            "experience_counts": source_counts,
            "fallback_mask_count": legacy_metadata["replay.verification"]["fallback_mask_count"],
            "verification": legacy_metadata["replay.verification"],
            "asserted_facts_digest": legacy_metadata["replay.legacy_asserted_facts_digest"],
        }
        if dry_run:
            if sqlite_file_hashes(source_path) != source_hashes_before:
                raise RuntimeError("Dry-run changed source SQLite files")
            return result

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temp_handle = tempfile.NamedTemporaryFile(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
            delete=False,
        )
        temp_path = Path(temp_handle.name)
        temp_handle.close()
        try:
            destination_conn = sqlite3.connect(temp_path)
            try:
                source_conn.backup(destination_conn)
                destination_conn.execute("PRAGMA journal_mode=DELETE")
            finally:
                destination_conn.close()
            backup_snapshot_hashes = sqlite_file_hashes(temp_path)
            destination_conn = sqlite3.connect(temp_path)
            try:
                with destination_conn:
                    _write_metadata(destination_conn, legacy_metadata)
                    _write_metadata(
                        destination_conn,
                        {
                            "replay.migration": {
                                "schema_version": 1,
                                "method": "sqlite_backup_copy_first",
                                "source_file_hashes_at_open": source_hashes_before,
                                "backup_snapshot_file_hashes": backup_snapshot_hashes,
                                "source_experience_counts": source_counts,
                            }
                        },
                    )
                integrity = destination_conn.execute("PRAGMA integrity_check").fetchone()[0]
                if integrity != "ok":
                    raise RuntimeError(f"Migrated replay integrity check failed: {integrity}")
                destination_counts = _experience_counts(destination_conn)
                if destination_counts != source_counts:
                    raise RuntimeError(
                        "Migrated replay row counts changed: "
                        f"source={source_counts}, destination={destination_counts}"
                    )
            finally:
                destination_conn.close()
            _fsync_file(temp_path)
            if before_publish is not None:
                before_publish(temp_path)
            source_hashes_at_validation = sqlite_file_hashes(source_path)
            if source_hashes_at_validation != source_hashes_before:
                raise RuntimeError("Replay source changed while its backup snapshot was captured")
            if _destination_sqlite_files(destination_path):
                raise FileExistsError(f"Replay destination files appeared: {destination_path}")

            published = False
            try:
                # A hard-link create is atomic and fails if a destination wins
                # the race after the explicit main/WAL/SHM checks.
                os.link(temp_path, destination_path)
                published = True
                temp_path.unlink()
                _fsync_directory(destination_path.parent)
            except Exception:
                if published and destination_path.exists():
                    destination_path.unlink()
                    _fsync_directory(destination_path.parent)
                raise
            result["source_file_hashes_at_snapshot_validation"] = source_hashes_at_validation
            result["backup_snapshot_file_hashes"] = backup_snapshot_hashes
        except Exception:
            _remove_temporary_sqlite_files(temp_path)
            raise
    finally:
        source_conn.close()

    result["destination_sha256"] = replay_sqlite_hashes(destination_path)["main"]
    return result


def _load_asserted_facts(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("asserted facts JSON must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy legacy replay and attach explicit unverified metadata"
    )
    parser.add_argument("source", help="Existing SQLite replay (opened read-only)")
    parser.add_argument("destination", help="Separate destination SQLite path")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without writing")
    parser.add_argument(
        "--asserted-facts-json",
        help="Optional JSON facts retained as assertions; they never confer verified status",
    )
    args = parser.parse_args()
    result = migrate_replay_metadata(
        args.source,
        args.destination,
        dry_run=args.dry_run,
        asserted_facts=_load_asserted_facts(args.asserted_facts_json),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
