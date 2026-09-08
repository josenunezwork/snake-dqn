"""End-to-end persistence checks for replay successor-mask semantics."""

import sqlite3
import struct

import pytest
import torch

from src.data.memory_db_handler import MemoryDBHandler, validate_replay_quality_gates
from src.training.multistep_buffer import MultiStepBuffer
from src.training.replay_buffer import restore_replay_memories
from src.training.td_targets import (
    MASK_MODE_DATASET_VECTOR_ADVISORY_V1,
    MASK_MODE_LEGACY_ADVISORY,
    MASK_MODE_RASTER_RESOLVED_V3,
    MASK_MODE_TERMINAL_NO_SUCCESSOR,
)
from src.utils.tensor_utils import memories_to_dicts


def _state(value: float, *, boost_available: bool = False) -> torch.Tensor:
    state = torch.full((58,), value, dtype=torch.float32)
    state[57] = float(boost_available)
    return state


def _rows_for_restore(columns: tuple[list, ...]) -> list[tuple]:
    return list(zip(*columns))


def test_multistep_export_sqlite_restore_sample_preserves_mask_modes(temp_db):
    """Actual public persistence paths retain mixed row-wise mask authority."""
    source = MultiStepBuffer(capacity=8, n_step=1)
    source.add(
        _state(0.0),
        0,
        1.0,
        _state(1.0),
        False,
        next_action_mask=[False, False, False, True, True, True],
        next_action_mask_mode=MASK_MODE_LEGACY_ADVISORY,
    )
    source.add(
        _state(2.0),
        1,
        2.0,
        _state(3.0),
        False,
        next_action_mask=[False, True, False, False, False, False],
        next_action_mask_mode=MASK_MODE_RASTER_RESOLVED_V3,
    )
    source.add(
        _state(4.0),
        2,
        3.0,
        _state(5.0),
        False,
        next_action_mask=[True, False, False, False, False, False],
        next_action_mask_mode=MASK_MODE_DATASET_VECTOR_ADVISORY_V1,
    )
    source.add(
        _state(6.0),
        2,
        4.0,
        _state(5.0),
        True,
        next_action_mask=None,
        next_action_mask_mode=MASK_MODE_TERMINAL_NO_SUCCESSOR,
    )

    inline_checkpoint_rows = source.get_all_memories()
    inline_restored = MultiStepBuffer(capacity=8, n_step=3)
    restore_replay_memories(
        inline_restored,
        inline_checkpoint_rows,
        torch.device("cpu"),
    )
    inline_batch, _, _ = inline_restored.sample(4, torch.device("cpu"))
    assert set(inline_batch["next_action_mask_modes"].tolist()) == {
        MASK_MODE_LEGACY_ADVISORY,
        MASK_MODE_RASTER_RESOLVED_V3,
        MASK_MODE_DATASET_VECTOR_ADVISORY_V1,
        MASK_MODE_TERMINAL_NO_SUCCESSOR,
    }

    exported = memories_to_dicts(inline_checkpoint_rows)
    handler = MemoryDBHandler(temp_db)
    try:
        handler.save_memories(snake_id=11, memories=exported)
        loaded = handler.load_memories_for_policy(
            "apex",
            limit=None,
            order_by="id",
            include_action_masks=True,
            include_action_mask_modes=True,
            include_snake_ids=True,
        )
    finally:
        handler.close()

    assert loaded[8] == [
        MASK_MODE_LEGACY_ADVISORY,
        MASK_MODE_RASTER_RESOLVED_V3,
        MASK_MODE_DATASET_VECTOR_ADVISORY_V1,
        MASK_MODE_TERMINAL_NO_SUCCESSOR,
    ]
    restored = MultiStepBuffer(capacity=8, n_step=3)
    assert (
        restore_replay_memories(
            restored,
            _rows_for_restore(loaded),
            torch.device("cpu"),
        )
        == 4
    )

    batch, _, _ = restored.sample(4, torch.device("cpu"))
    assert set(batch["next_action_mask_modes"].tolist()) == {
        MASK_MODE_LEGACY_ADVISORY,
        MASK_MODE_RASTER_RESOLVED_V3,
        MASK_MODE_DATASET_VECTOR_ADVISORY_V1,
        MASK_MODE_TERMINAL_NO_SUCCESSOR,
    }
    terminal_rows = batch["next_action_mask_modes"] == MASK_MODE_TERMINAL_NO_SUCCESSOR
    assert bool(batch["dones"][terminal_rows].all())
    assert not bool(batch["next_action_masks"][terminal_rows].any())


def test_homogeneous_terminal_rows_keep_mode_without_synthesizing_masks(temp_db):
    """Terminal-only persistence keeps mode metadata while masks remain absent."""
    source = MultiStepBuffer(capacity=4, n_step=1)
    source.add(
        _state(0.0),
        0,
        -1.0,
        _state(0.0),
        True,
        next_action_mask_mode=MASK_MODE_TERMINAL_NO_SUCCESSOR,
    )

    handler = MemoryDBHandler(temp_db)
    try:
        handler.save_memories(0, memories_to_dicts(source.get_all_memories()))
        loaded = handler.load_memories_for_policy(
            "apex",
            limit=None,
            order_by="id",
            include_action_masks=True,
            include_action_mask_modes=True,
            include_snake_ids=True,
        )
    finally:
        handler.close()

    assert loaded[7] == [None]
    assert loaded[8] == [MASK_MODE_TERMINAL_NO_SUCCESSOR]
    restored = MultiStepBuffer(capacity=4, n_step=3)
    restore_replay_memories(restored, _rows_for_restore(loaded), torch.device("cpu"))
    restored_row = restored.get_all_memories()[0]
    assert restored_row[7] is None
    assert restored_row[8] == MASK_MODE_TERMINAL_NO_SUCCESSOR


def test_read_only_old_database_synthesizes_legacy_advisory_mode(temp_db):
    """An unmigrated database stays read-only and cannot claim exact mask authority."""
    state_blob = struct.pack("<58f", *([0.0] * 58))
    conn = sqlite3.connect(temp_db)
    conn.execute("""
        CREATE TABLE memories_standard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snake_id INTEGER,
            policy_type TEXT,
            state BLOB,
            action INTEGER,
            reward REAL,
            next_state BLOB,
            done INTEGER,
            priority REAL DEFAULT 1.0,
            bootstrap_steps INTEGER DEFAULT 1,
            next_action_mask INTEGER
        )
        """)
    conn.execute(
        """
        INSERT INTO memories_standard
            (snake_id, policy_type, state, action, reward, next_state, done,
             priority, bootstrap_steps, next_action_mask)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (0, "apex", state_blob, 0, 0.0, state_blob, 0, 1.0, 1, 1),
    )
    conn.commit()
    conn.close()

    handler = MemoryDBHandler(temp_db, read_only=True)
    try:
        loaded = handler.load_memories_for_policy(
            "apex",
            limit=None,
            order_by="id",
            include_action_masks=True,
            include_action_mask_modes=True,
        )
        quality = handler.get_replay_quality_stats(policy_type="apex")
        columns = handler._table_columns("memories_standard")
    finally:
        handler.close()

    assert loaded[7] == [(True, False, False, False, False, False)]
    assert loaded[8] == [MASK_MODE_LEGACY_ADVISORY]
    assert quality["mask_count"] == 1
    assert quality["exact_mask_count"] == 0
    assert quality["nonterminal_exact_mask_fraction"] == 0.0
    assert "next_action_mask_mode" not in columns


def test_sql_quality_counts_and_rejects_corrupt_unknown_mask_mode(temp_db):
    """A corrupted stored mode remains present evidence but never exact evidence."""
    handler = MemoryDBHandler(temp_db)
    try:
        handler.save_memories(
            snake_id=0,
            memories=[
                {
                    "state": _state(0.0),
                    "action": 0,
                    "reward": 0.0,
                    "next_state": _state(1.0),
                    "done": False,
                    "priority": 1.0,
                    "bootstrap_steps": 1,
                    "next_action_mask": [True, False, False, False, False, False],
                    "next_action_mask_mode": MASK_MODE_LEGACY_ADVISORY,
                }
            ],
        )
        handler.cursor.execute("PRAGMA ignore_check_constraints = ON")
        handler.cursor.execute("UPDATE memories_standard SET next_action_mask_mode = 99")
        handler.conn.commit()

        stats = handler.get_replay_quality_stats(policy_type="apex")
    finally:
        handler.close()

    assert stats["mask_count"] == 1
    assert stats["exact_mask_count"] == 0
    assert stats["invalid_action_mask_mode_count"] == 1
    with pytest.raises(RuntimeError, match="invalid next-action mask modes"):
        validate_replay_quality_gates(stats)
