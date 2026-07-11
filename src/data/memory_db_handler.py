"""SQLite replay-database handler for Apex-DQN experience replay.

The stateless replay-quality analytics (validation/coercion/state-blob codec/
quality stats) live in replay_quality.py. This module is the DB I/O layer; it
re-exports the analytics it (and external callers) use so the public import path
``from src.data.memory_db_handler import X`` is preserved.
"""

import json
import logging
import sqlite3
import struct

from src.data.replay_quality import (
    ACTION_DANGER_COLLISION_THRESHOLD,
    ACTION_SIZE,
    BOOST_AVAILABLE_INDEX,
    PER_ACTION_DANGER_START,
    REPLAY_QUALITY_GATE_ORDER,
    REPLAY_QUALITY_GATE_PRESETS,
    STATE_BLOB_FORMAT,
    STATE_BLOB_SIZE,
    STATE_SIZE,
    _action_diversity_stats,
    _coerce_action,
    _coerce_action_mask,
    _coerce_bootstrap_steps,
    _coerce_done,
    _coerce_finite_float,
    _coerce_priority,
    _current_action_invalid_from_state,
    _decode_action_mask,
    _decode_state_blob,
    _encode_state_blob,
    _exact_mask_state_disagreement,
    _mask_all_actions_invalid,
    _state_has_malformed_direction,
    _state_has_malformed_semantic_features,
    _state_has_out_of_range_features,
    build_replay_quality_stats,
    current_action_invalid_from_state,
    format_replay_quality_stats,
    format_replay_quality_warnings,
    resolve_min_row_count,
    resolve_replay_quality_fraction,
    resolve_replay_quality_gate_values,
    validate_min_row_count,
    validate_replay_metadata_contract,
    validate_replay_quality_gates,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryDBHandler",
    "REPLAY_QUALITY_GATE_ORDER",
    "REPLAY_QUALITY_GATE_PRESETS",
    "STATE_SIZE",
    "build_replay_quality_stats",
    "current_action_invalid_from_state",
    "format_replay_quality_stats",
    "format_replay_quality_warnings",
    "resolve_min_row_count",
    "resolve_replay_quality_fraction",
    "resolve_replay_quality_gate_values",
    "validate_min_row_count",
    "validate_replay_metadata_contract",
    "validate_replay_quality_gates",
]


class MemoryDBHandler:
    """Database handler for Apex-DQN experience replay storage."""

    def __init__(self, db_name="snake_memories.db"):
        """
        Initialize the database connection and create tables.

        Args:
            db_name: Path to the SQLite database file.
        """
        self.conn = sqlite3.connect(db_name)
        # Performance pragmas for faster bulk inserts/reads
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA temp_store=MEMORY;")
            self.conn.execute("PRAGMA cache_size=-64000")
        except sqlite3.OperationalError:
            # Some PRAGMA settings may not be supported on all SQLite versions
            pass
        self.cursor = self.conn.cursor()
        try:
            self._create_tables()
            self._migrate_legacy_table()
        except Exception:
            # Avoid leaking the open connection if schema setup/migration fails
            # during half-completed construction (__del__/close may not run).
            try:
                self.conn.close()
            except Exception:
                pass
            raise

    def _create_tables(self):
        """Create the memories table if it doesn't exist."""
        # Standard memories table for Apex-DQN transitions
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories_standard (
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

        # Create index for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_standard_policy
            ON memories_standard(policy_type)
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS replay_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        self.conn.commit()
        self._migrate_standard_columns()

    def _migrate_standard_columns(self):
        """Add columns introduced after the original standard memory schema."""
        self.cursor.execute("PRAGMA table_info(memories_standard)")
        columns = {row[1] for row in self.cursor.fetchall()}

        if "bootstrap_steps" not in columns:
            self.cursor.execute(
                "ALTER TABLE memories_standard " "ADD COLUMN bootstrap_steps INTEGER DEFAULT 1"
            )
            self.conn.commit()
        if "next_action_mask" not in columns:
            self.cursor.execute("ALTER TABLE memories_standard ADD COLUMN next_action_mask INTEGER")
            self.conn.commit()

    def _migrate_legacy_table(self):
        """Migrate old 'memories' table to 'memories_standard' if it exists."""
        try:
            # Check if legacy table exists
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
            )
            if self.cursor.fetchone():
                # Count legacy memories
                self.cursor.execute("SELECT COUNT(*) FROM memories")
                count = self.cursor.fetchone()[0]

                if count > 0:
                    logger.info("Migrating %d legacy memories to new schema...", count)

                    # Copy data to new table (convert to apex format)
                    self.cursor.execute("""
                        INSERT INTO memories_standard
                            (
                                snake_id, policy_type, state, action,
                                reward, next_state, done, priority, bootstrap_steps
                            )
                        SELECT
                            snake_id, 'apex', state, action, reward, next_state,
                            done, priority, 1
                        FROM memories
                    """)

                    # Rename legacy table
                    self.cursor.execute("ALTER TABLE memories RENAME TO memories_legacy")
                    self.conn.commit()
                    logger.info("Migration complete. Legacy table renamed to 'memories_legacy'")
        except sqlite3.OperationalError:
            # Table doesn't exist or already migrated
            pass

    # ========================
    # UNIFIED INTERFACE
    # ========================

    def save_memories(self, snake_id: int, memories: list, policy_type: str = "apex"):
        """
        Save Apex-DQN experience memories to the database.

        Args:
            snake_id: Snake identifier.
            memories: List of memory dicts from policy.prepare_memories_for_saving().
                     Each dict should contain: state, action, reward, next_state, done, priority.
            policy_type: Policy type string (defaults to 'apex').
        """
        if not memories:
            return

        self._save_standard_memories(snake_id, memories, policy_type)

    def set_metadata(self, key: str, value) -> None:
        """Store JSON-serializable replay database metadata by key."""
        key = str(key).strip()
        if not key:
            raise ValueError("metadata key must not be empty")

        encoded = json.dumps(value, sort_keys=True)
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO replay_metadata (key, value)
            VALUES (?, ?)
        """,
            (key, encoded),
        )
        self.conn.commit()

    def update_metadata(self, metadata: dict) -> None:
        """Store multiple JSON-serializable replay metadata values."""
        if not metadata:
            return

        rows = []
        for key, value in metadata.items():
            key = str(key).strip()
            if not key:
                raise ValueError("metadata key must not be empty")
            rows.append((key, json.dumps(value, sort_keys=True)))

        self.cursor.executemany(
            """
            INSERT OR REPLACE INTO replay_metadata (key, value)
            VALUES (?, ?)
        """,
            rows,
        )
        self.conn.commit()

    def get_metadata(self, key: str | None = None) -> dict | object | None:
        """Return replay database metadata, or one decoded value when key is given."""
        if key is not None:
            key = str(key).strip()
            if not key:
                raise ValueError("metadata key must not be empty")
            self.cursor.execute("SELECT value FROM replay_metadata WHERE key = ?", (key,))
            row = self.cursor.fetchone()
            return json.loads(row[0]) if row else None

        self.cursor.execute("SELECT key, value FROM replay_metadata ORDER BY key ASC")
        return {row[0]: json.loads(row[1]) for row in self.cursor.fetchall()}

    def load_memories_for_policy(
        self,
        policy_type: str,
        snake_id: int = None,
        limit: int = 4000,
        order_by: str = "priority",
        include_action_masks: bool = False,
        include_snake_ids: bool = False,
    ):
        """
        Load memories for the Apex-DQN policy.

        Args:
            policy_type: Policy type string (typically 'apex').
            snake_id: Optional snake ID filter.
            limit: Maximum number of memories to load. Use None to load all rows.
            order_by: Row ordering strategy: 'priority', 'id', or 'id_uniform'.
            include_action_masks: If True, append optional next_action_masks
                to the returned tuple.
            include_snake_ids: If True, append row snake IDs to the returned tuple.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, priorities,
            bootstrap_steps) plus optional next_action_masks and snake_ids.
        """
        return self._load_standard_memories(
            policy_type,
            snake_id,
            limit,
            order_by=order_by,
            include_action_masks=include_action_masks,
            include_snake_ids=include_snake_ids,
        )

    # ========================
    # STANDARD MEMORIES
    # ========================

    def _save_standard_memories(self, snake_id: int, memories: list, policy_type: str):
        """
        Save individual transition memories.

        Args:
            snake_id: Snake identifier.
            memories: List of transition dicts.
            policy_type: Policy type string.
        """
        rows = []
        for memory in memories:
            action = _coerce_action(memory["action"])
            reward = _coerce_finite_float(memory["reward"], "reward")
            done = _coerce_done(memory["done"])
            priority = _coerce_priority(memory.get("priority", 1.0))
            bootstrap_steps = _coerce_bootstrap_steps(memory.get("bootstrap_steps", 1))
            next_action_mask = _coerce_action_mask(memory.get("next_action_mask"))

            state_bytes = _encode_state_blob(memory["state"], "state")
            next_state_bytes = _encode_state_blob(memory["next_state"], "next_state")

            rows.append(
                (
                    snake_id,
                    policy_type,
                    state_bytes,
                    action,
                    reward,
                    next_state_bytes,
                    int(done),
                    priority,
                    bootstrap_steps,
                    next_action_mask,
                )
            )

        self.cursor.executemany(
            """
            INSERT INTO memories_standard
                (
                    snake_id, policy_type, state, action, reward,
                    next_state, done, priority, bootstrap_steps, next_action_mask
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            rows,
        )
        self.conn.commit()

    def _load_standard_memories(
        self,
        policy_type: str = None,
        snake_id: int = None,
        limit: int = 4000,
        order_by: str = "priority",
        include_action_masks: bool = False,
        include_snake_ids: bool = False,
    ):
        """
        Load standard format memories.

        Args:
            policy_type: Optional policy type filter.
            snake_id: Optional snake ID filter.
            limit: Maximum number of memories to load. Use None to load all rows.
            order_by: Row ordering strategy: 'priority', 'id', or 'id_uniform'.
                'id_uniform' preserves insertion order while selecting an evenly
                spaced subset when limit is smaller than the matching row count.
            include_action_masks: Whether to append optional next_action_masks.
            include_snake_ids: Whether to append row snake IDs.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, priorities,
            bootstrap_steps) plus optional next_action_masks and snake_ids.
        """
        if order_by not in {"priority", "id", "id_uniform"}:
            raise ValueError("order_by must be one of 'priority', 'id', or 'id_uniform'")

        select_clause = (
            "SELECT id, snake_id, policy_type, state, action, reward, next_state, "
            "done, priority, bootstrap_steps, next_action_mask "
            "FROM memories_standard"
        )
        where_clause = " WHERE 1=1"
        params = []

        if policy_type:
            where_clause += " AND policy_type = ?"
            params.append(policy_type)

        if snake_id is not None:
            where_clause += " AND snake_id = ?"
            params.append(snake_id)

        if order_by == "id_uniform" and limit is not None:
            selected_ids = self._select_uniform_memory_ids(where_clause, params, int(limit))
            if selected_ids:
                placeholders = ", ".join("?" for _ in selected_ids)
                query = f"{select_clause}{where_clause} AND id IN ({placeholders}) ORDER BY id ASC"
                self.cursor.execute(query, [*params, *selected_ids])
                rows = self.cursor.fetchall()
            else:
                rows = []
        else:
            query = f"{select_clause}{where_clause}"
            if order_by == "priority":
                query += " ORDER BY priority DESC, id ASC"
            else:
                query += " ORDER BY id ASC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(int(limit))

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        priorities = []
        bootstrap_steps = []
        next_action_masks = []
        snake_ids = []

        for row in rows:
            try:
                # Row format:
                # id, snake_id, policy_type, state, action, reward,
                # next_state, done, priority, bootstrap_steps, next_action_mask
                state_data = _decode_state_blob(row[3], "state")
                next_state_data = _decode_state_blob(row[6], "next_state")
                action = _coerce_action(row[4])
                reward = _coerce_finite_float(row[5], "reward")
                done = _coerce_done(row[7])
                priority = _coerce_priority(row[8])
                steps = _coerce_bootstrap_steps(row[9])
                next_action_mask = _decode_action_mask(row[10])

                states.append(state_data)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state_data)
                dones.append(done)
                priorities.append(priority)
                bootstrap_steps.append(steps)
                next_action_masks.append(next_action_mask)
                snake_ids.append(int(row[1]))
            except ValueError as e:
                raise ValueError(f"Stored replay row {row[0]} is invalid: {e}") from e

        logger.info("Loaded %d memories for policy '%s'", len(states), policy_type)
        result = (
            states,
            actions,
            rewards,
            next_states,
            dones,
            priorities,
            bootstrap_steps,
        )
        if include_action_masks:
            result = (*result, next_action_masks)
        if include_snake_ids:
            result = (*result, snake_ids)
        return result

    def _select_uniform_memory_ids(
        self,
        where_clause: str,
        params: list,
        limit: int,
    ) -> list:
        """Return deterministic, evenly spaced row IDs for capped replay prefill."""
        if limit <= 0:
            return []

        query = f"SELECT id FROM memories_standard{where_clause} ORDER BY id ASC"
        self.cursor.execute(query, params)
        ids = [row[0] for row in self.cursor.fetchall()]

        row_count = len(ids)
        if row_count <= limit:
            return ids
        if limit == 1:
            return [ids[0]]

        last_index = row_count - 1
        return [ids[round(index * last_index / (limit - 1))] for index in range(limit)]

    # ========================
    # LEGACY COMPATIBILITY
    # ========================

    def load_memories(self, snake_id=None):
        """
        Legacy load method - loads from standard table.

        For backward compatibility with existing code.

        Args:
            snake_id: Optional snake ID filter.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, priorities,
            bootstrap_steps).
        """
        return self._load_standard_memories(policy_type=None, snake_id=snake_id)

    # ========================
    # UTILITY METHODS
    # ========================

    def clear_memories(self, snake_id: int = None, policy_type: str = None):
        """
        Clear memories from the database.

        Args:
            snake_id: Optional snake ID filter. If None, clears all snakes.
            policy_type: Optional policy type filter. If None, clears all policies.
        """
        if snake_id is None and policy_type is None:
            self.cursor.execute("DELETE FROM memories_standard")
        elif policy_type and snake_id is not None:
            self.cursor.execute(
                "DELETE FROM memories_standard WHERE policy_type = ? AND snake_id = ?",
                (policy_type, snake_id),
            )
        elif policy_type:
            self.cursor.execute(
                "DELETE FROM memories_standard WHERE policy_type = ?", (policy_type,)
            )
        elif snake_id is not None:
            self.cursor.execute("DELETE FROM memories_standard WHERE snake_id = ?", (snake_id,))

        self.conn.commit()

    def get_memory_count(self, policy_type: str = None, snake_id: int = None) -> int:
        """
        Get the number of memories stored.

        Args:
            policy_type: Optional policy type filter.
            snake_id: Optional snake ID filter.

        Returns:
            Number of stored memories matching the filters.
        """
        query = "SELECT COUNT(*) FROM memories_standard WHERE 1=1"
        params = []

        if policy_type:
            query += " AND policy_type = ?"
            params.append(policy_type)
        if snake_id is not None:
            query += " AND snake_id = ?"
            params.append(snake_id)

        self.cursor.execute(query, params)
        return self.cursor.fetchone()[0]

    def get_memory_stats(self) -> dict:
        """
        Get statistics about stored memories by policy type.

        Returns:
            Dictionary mapping policy_type to count and type info.
        """
        stats = {}

        self.cursor.execute("""
            SELECT policy_type, COUNT(*)
            FROM memories_standard
            GROUP BY policy_type
        """)
        for row in self.cursor.fetchall():
            stats[row[0]] = {"count": row[1], "type": "standard"}

        return stats

    def get_replay_quality_stats(self, policy_type: str = None, snake_id: int = None) -> dict:
        """Return replay diagnostics that help catch unlearnable generated datasets.

        NOT a duplicate of replay_quality.build_replay_quality_stats — do not merge.
        This reads the TRUSTED SQLite DB (SQL aggregates, exact ``reward = 0.0``,
        decodes the DB's packed-integer mask), so it computes a subset and skips the
        invalid-field coercion that build_* does for UNTRUSTED in-memory rows. The two
        intentionally disagree on mask/reward-zero keys; only the per-row state-feature
        checks (the atomic _state_has_* helpers) are genuinely shared.
        """
        where_clause = "WHERE 1=1"
        params = []

        if policy_type:
            where_clause += " AND policy_type = ?"
            params.append(policy_type)
        if snake_id is not None:
            where_clause += " AND snake_id = ?"
            params.append(snake_id)

        self.cursor.execute(
            f"""
            SELECT
                COUNT(*),
                COALESCE(SUM(CASE WHEN done = 1 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN done = 0 THEN 1 ELSE 0 END), 0),
                COALESCE(MIN(reward), 0.0),
                COALESCE(AVG(reward), 0.0),
                COALESCE(MAX(reward), 0.0),
                COALESCE(SUM(CASE WHEN reward < 0.0 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN reward = 0.0 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN reward > 0.0 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE
                    WHEN done = 1 AND reward < 0.0 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE
                    WHEN done = 1 AND reward = 0.0 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE
                    WHEN done = 1 AND reward > 0.0 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE
                    WHEN done = 1 AND COALESCE(bootstrap_steps, 1) <= 1
                    THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE
                    WHEN done = 1 AND COALESCE(bootstrap_steps, 1) > 1
                    THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE
                    WHEN done = 1 AND COALESCE(bootstrap_steps, 1) <= 1
                         AND reward >= 0.0
                    THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE
                    WHEN done = 1 AND COALESCE(bootstrap_steps, 1) > 1
                         AND reward >= 0.0
                    THEN 1 ELSE 0 END), 0),
                COALESCE(MIN(priority), 0.0),
                COALESCE(AVG(priority), 0.0),
                COALESCE(MAX(priority), 0.0),
                COALESCE(MIN(bootstrap_steps), 0),
                COALESCE(AVG(bootstrap_steps), 0.0),
                COALESCE(MAX(bootstrap_steps), 0),
                COALESCE(SUM(CASE WHEN bootstrap_steps > 1 THEN 1 ELSE 0 END), 0),
                COUNT(DISTINCT snake_id)
            FROM memories_standard
            {where_clause}
            """,
            params,
        )
        (
            count,
            done_count,
            nonterminal_count,
            reward_min,
            reward_avg,
            reward_max,
            reward_negative_count,
            reward_zero_count,
            reward_positive_count,
            terminal_reward_negative_count,
            terminal_reward_zero_count,
            terminal_reward_positive_count,
            terminal_immediate_count,
            terminal_multistep_count,
            terminal_immediate_nonnegative_reward_count,
            terminal_multistep_nonnegative_reward_count,
            priority_min,
            priority_avg,
            priority_max,
            bootstrap_steps_min,
            bootstrap_steps_avg,
            bootstrap_steps_max,
            multistep_count,
            snake_count,
        ) = self.cursor.fetchone()
        terminal_nonnegative_reward_count = int(terminal_reward_zero_count) + int(
            terminal_reward_positive_count
        )
        immediate_terminal_fraction = float(terminal_immediate_count) / count if count else 0.0
        terminal_immediate_fraction = (
            float(terminal_immediate_count) / done_count if done_count else 0.0
        )
        terminal_multistep_fraction = (
            float(terminal_multistep_count) / done_count if done_count else 0.0
        )
        terminal_immediate_nonnegative_reward_fraction = (
            float(terminal_immediate_nonnegative_reward_count) / terminal_immediate_count
            if terminal_immediate_count
            else 0.0
        )
        terminal_multistep_nonnegative_reward_fraction = (
            float(terminal_multistep_nonnegative_reward_count) / terminal_multistep_count
            if terminal_multistep_count
            else 0.0
        )

        self.cursor.execute(
            f"""
            SELECT action, COUNT(*)
            FROM memories_standard
            {where_clause}
            GROUP BY action
            ORDER BY action
            """,
            params,
        )
        action_counts = {int(action): int(action_count) for action, action_count in self.cursor}
        action_diversity = _action_diversity_stats(action_counts, int(count))
        self.cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM memories_standard
            {where_clause}
            GROUP BY snake_id
            """,
            params,
        )
        snake_row_counts = [int(row_count) for (row_count,) in self.cursor]
        if snake_row_counts:
            snake_rows_min = min(snake_row_counts)
            snake_rows_max = max(snake_row_counts)
            snake_rows_avg = sum(snake_row_counts) / len(snake_row_counts)
            dominant_snake_fraction = snake_rows_max / count if count else 0.0
        else:
            snake_rows_min = 0
            snake_rows_max = 0
            snake_rows_avg = 0.0
            dominant_snake_fraction = 0.0
        boost_available_count = 0
        malformed_boost_count = 0
        trapped_state_count = 0
        nonterminal_trapped_state_count = 0
        trapped_next_state_count = 0
        nonterminal_trapped_next_state_count = 0
        malformed_danger_count = 0
        malformed_next_danger_count = 0
        exact_mask_state_comparison_count = 0
        exact_mask_state_mismatch_count = 0
        exact_mask_unsafe_normal_count = 0
        exact_mask_blocked_safe_normal_count = 0
        valid_state_count = 0
        valid_next_state_count = 0
        malformed_state_range_count = 0
        malformed_direction_count = 0
        malformed_state_feature_count = 0
        malformed_next_state_range_count = 0
        malformed_next_direction_count = 0
        malformed_next_state_feature_count = 0
        invalid_state_feature_count = 0
        invalid_next_state_feature_count = 0
        invalid_action_mask_count = 0
        current_action_comparison_count = 0
        invalid_current_action_count = 0
        invalid_current_normal_action_count = 0
        invalid_current_boost_action_count = 0
        terminal_invalid_current_action_count = 0
        nonterminal_current_action_comparison_count = 0
        nonterminal_invalid_current_action_count = 0
        nonterminal_invalid_current_normal_action_count = 0
        nonterminal_invalid_current_boost_action_count = 0
        mask_count = 0
        nonterminal_mask_count = 0
        boost_mask_count = 0
        if count:
            self.cursor.execute(
                f"""
                SELECT state, action, next_state, done, next_action_mask
                FROM memories_standard
                {where_clause}
                """,
                params,
            )
            for state_blob, action, next_state_blob, done, next_action_mask in self.cursor:
                if not isinstance(state_blob, (bytes, bytearray, memoryview)):
                    state_blob = None
                if state_blob is not None and len(state_blob) == STATE_BLOB_SIZE:
                    valid_state_count += 1
                    state_values = struct.unpack(STATE_BLOB_FORMAT, state_blob)
                    if _state_has_out_of_range_features(state_values):
                        malformed_state_range_count += 1
                    if _state_has_malformed_direction(state_values):
                        malformed_direction_count += 1
                    if _state_has_malformed_semantic_features(state_values):
                        malformed_state_feature_count += 1
                    boost_value = struct.unpack_from(
                        "<f",
                        state_blob,
                        BOOST_AVAILABLE_INDEX * struct.calcsize("<f"),
                    )[0]
                    if boost_value < 0.0 or boost_value > 1.0:
                        malformed_boost_count += 1
                    if boost_value >= 0.5:
                        boost_available_count += 1
                    danger_values = struct.unpack_from(
                        "<3f",
                        state_blob,
                        PER_ACTION_DANGER_START * struct.calcsize("<f"),
                    )
                    if any(value < 0.0 or value > 1.0 for value in danger_values):
                        malformed_danger_count += 1
                    if all(value >= ACTION_DANGER_COLLISION_THRESHOLD for value in danger_values):
                        trapped_state_count += 1
                        if not bool(done):
                            nonterminal_trapped_state_count += 1
                    current_action_comparison_count += 1
                    if not bool(done):
                        nonterminal_current_action_comparison_count += 1
                    invalid_action, invalid_normal, invalid_boost = (
                        _current_action_invalid_from_state(int(action), state_values)
                    )
                    if invalid_action:
                        invalid_current_action_count += 1
                        if bool(done):
                            terminal_invalid_current_action_count += 1
                        else:
                            nonterminal_invalid_current_action_count += 1
                    if invalid_normal:
                        invalid_current_normal_action_count += 1
                        if not bool(done):
                            nonterminal_invalid_current_normal_action_count += 1
                    if invalid_boost:
                        invalid_current_boost_action_count += 1
                        if not bool(done):
                            nonterminal_invalid_current_boost_action_count += 1
                else:
                    invalid_state_feature_count += 1
                decoded_action_mask = None
                if next_action_mask is not None:
                    try:
                        decoded_action_mask = _decode_action_mask(next_action_mask)
                    except (TypeError, ValueError, OverflowError):
                        invalid_action_mask_count += 1
                    else:
                        mask_count += 1
                        if not bool(done):
                            nonterminal_mask_count += 1
                            if any(decoded_action_mask[3:ACTION_SIZE]):
                                boost_mask_count += 1

                if not bool(done):
                    exact_trapped = None
                    if decoded_action_mask is not None:
                        exact_trapped = _mask_all_actions_invalid(decoded_action_mask)
                    if (
                        isinstance(next_state_blob, (bytes, bytearray, memoryview))
                        and len(next_state_blob) == STATE_BLOB_SIZE
                    ):
                        valid_next_state_count += 1
                        next_state_values = struct.unpack(STATE_BLOB_FORMAT, next_state_blob)
                        if _state_has_out_of_range_features(next_state_values):
                            malformed_next_state_range_count += 1
                        if _state_has_malformed_direction(next_state_values):
                            malformed_next_direction_count += 1
                        if _state_has_malformed_semantic_features(next_state_values):
                            malformed_next_state_feature_count += 1
                        next_danger_values = struct.unpack_from(
                            "<3f",
                            next_state_blob,
                            PER_ACTION_DANGER_START * struct.calcsize("<f"),
                        )
                        if any(value < 0.0 or value > 1.0 for value in next_danger_values):
                            malformed_next_danger_count += 1
                        if decoded_action_mask is not None:
                            exact_mask_state_comparison_count += 1
                            mismatch, unsafe_allowed, safe_blocked = _exact_mask_state_disagreement(
                                decoded_action_mask,
                                next_danger_values,
                            )
                            if mismatch:
                                exact_mask_state_mismatch_count += 1
                            if unsafe_allowed:
                                exact_mask_unsafe_normal_count += 1
                            if safe_blocked:
                                exact_mask_blocked_safe_normal_count += 1
                        state_trapped = all(
                            value >= ACTION_DANGER_COLLISION_THRESHOLD
                            for value in next_danger_values
                        )
                    else:
                        invalid_next_state_feature_count += 1
                        state_trapped = False
                    if exact_trapped if exact_trapped is not None else state_trapped:
                        trapped_next_state_count += 1
                        nonterminal_trapped_next_state_count += 1
        total_valid_state_feature_count = (
            valid_state_count
            + valid_next_state_count
            + invalid_state_feature_count
            + invalid_next_state_feature_count
        )
        total_malformed_state_feature_count = (
            malformed_state_feature_count
            + malformed_next_state_feature_count
            + invalid_state_feature_count
            + invalid_next_state_feature_count
        )

        return {
            "count": int(count),
            "done_count": int(done_count),
            "terminal_fraction": (float(done_count) / count) if count else 0.0,
            "nonterminal_count": int(nonterminal_count),
            "mask_count": int(mask_count),
            "mask_fraction": (float(mask_count) / count) if count else 0.0,
            "nonterminal_mask_count": int(nonterminal_mask_count),
            "nonterminal_mask_fraction": (
                float(nonterminal_mask_count) / nonterminal_count if nonterminal_count else 0.0
            ),
            "boost_mask_count": int(boost_mask_count),
            "boost_mask_fraction": (
                float(boost_mask_count) / nonterminal_count if nonterminal_count else 0.0
            ),
            "reward_min": float(reward_min),
            "reward_avg": float(reward_avg),
            "reward_max": float(reward_max),
            "reward_negative_count": int(reward_negative_count),
            "reward_zero_count": int(reward_zero_count),
            "reward_positive_count": int(reward_positive_count),
            "terminal_reward_negative_count": int(terminal_reward_negative_count),
            "terminal_reward_zero_count": int(terminal_reward_zero_count),
            "terminal_reward_positive_count": int(terminal_reward_positive_count),
            "terminal_nonnegative_reward_count": int(terminal_nonnegative_reward_count),
            "terminal_nonnegative_reward_fraction": (
                float(terminal_nonnegative_reward_count) / done_count if done_count else 0.0
            ),
            "terminal_immediate_count": int(terminal_immediate_count),
            "immediate_terminal_fraction": immediate_terminal_fraction,
            "terminal_immediate_fraction": terminal_immediate_fraction,
            "terminal_multistep_count": int(terminal_multistep_count),
            "terminal_multistep_fraction": terminal_multistep_fraction,
            "terminal_immediate_nonnegative_reward_count": int(
                terminal_immediate_nonnegative_reward_count
            ),
            "terminal_immediate_nonnegative_reward_fraction": (
                terminal_immediate_nonnegative_reward_fraction
            ),
            "terminal_multistep_nonnegative_reward_count": int(
                terminal_multistep_nonnegative_reward_count
            ),
            "terminal_multistep_nonnegative_reward_fraction": (
                terminal_multistep_nonnegative_reward_fraction
            ),
            "priority_min": float(priority_min),
            "priority_avg": float(priority_avg),
            "priority_max": float(priority_max),
            "bootstrap_steps_min": int(bootstrap_steps_min),
            "bootstrap_steps_avg": float(bootstrap_steps_avg),
            "bootstrap_steps_max": int(bootstrap_steps_max),
            "multistep_count": int(multistep_count),
            "multistep_fraction": (float(multistep_count) / count) if count else 0.0,
            "snake_count": int(snake_count),
            "snake_rows_min": int(snake_rows_min),
            "snake_rows_avg": float(snake_rows_avg),
            "snake_rows_max": int(snake_rows_max),
            "dominant_snake_fraction": float(dominant_snake_fraction),
            "action_counts": action_counts,
            **action_diversity,
            "boost_available_count": int(boost_available_count),
            "boost_available_fraction": (float(boost_available_count) / count) if count else 0.0,
            "malformed_boost_feature_count": int(malformed_boost_count),
            "trapped_state_count": int(trapped_state_count),
            "trapped_state_fraction": (float(trapped_state_count) / count) if count else 0.0,
            "nonterminal_trapped_state_count": int(nonterminal_trapped_state_count),
            "nonterminal_trapped_state_fraction": (
                float(nonterminal_trapped_state_count) / nonterminal_count
                if nonterminal_count
                else 0.0
            ),
            "malformed_per_action_danger_count": int(malformed_danger_count),
            "valid_state_feature_count": int(valid_state_count),
            "malformed_state_range_count": int(malformed_state_range_count),
            "malformed_direction_feature_count": int(malformed_direction_count),
            "malformed_state_feature_count": int(malformed_state_feature_count),
            "invalid_state_feature_count": int(invalid_state_feature_count),
            "current_action_state_comparison_count": int(current_action_comparison_count),
            "invalid_current_action_count": int(invalid_current_action_count),
            "invalid_current_action_fraction": (
                float(invalid_current_action_count) / current_action_comparison_count
                if current_action_comparison_count
                else 0.0
            ),
            "invalid_current_normal_action_count": int(invalid_current_normal_action_count),
            "invalid_current_boost_action_count": int(invalid_current_boost_action_count),
            "terminal_invalid_current_action_count": int(terminal_invalid_current_action_count),
            "nonterminal_current_action_state_comparison_count": int(
                nonterminal_current_action_comparison_count
            ),
            "nonterminal_invalid_current_action_count": int(
                nonterminal_invalid_current_action_count
            ),
            "nonterminal_invalid_current_action_fraction": (
                float(nonterminal_invalid_current_action_count)
                / nonterminal_current_action_comparison_count
                if nonterminal_current_action_comparison_count
                else 0.0
            ),
            "nonterminal_invalid_current_normal_action_count": int(
                nonterminal_invalid_current_normal_action_count
            ),
            "nonterminal_invalid_current_boost_action_count": int(
                nonterminal_invalid_current_boost_action_count
            ),
            "invalid_action_mask_count": int(invalid_action_mask_count),
            "invalid_action_mask_fraction": (
                float(invalid_action_mask_count) / count if count else 0.0
            ),
            "trapped_next_state_count": int(trapped_next_state_count),
            "trapped_next_state_fraction": (
                float(trapped_next_state_count) / nonterminal_count if nonterminal_count else 0.0
            ),
            "nonterminal_trapped_next_state_count": int(nonterminal_trapped_next_state_count),
            "nonterminal_trapped_next_state_fraction": (
                float(nonterminal_trapped_next_state_count) / nonterminal_count
                if nonterminal_count
                else 0.0
            ),
            "malformed_next_per_action_danger_count": int(malformed_next_danger_count),
            "valid_next_state_feature_count": int(valid_next_state_count),
            "malformed_next_state_range_count": int(malformed_next_state_range_count),
            "malformed_next_direction_feature_count": int(malformed_next_direction_count),
            "malformed_next_state_feature_count": int(malformed_next_state_feature_count),
            "invalid_next_state_feature_count": int(invalid_next_state_feature_count),
            "malformed_state_feature_fraction": (
                float(total_malformed_state_feature_count) / total_valid_state_feature_count
                if total_valid_state_feature_count
                else 0.0
            ),
            "exact_mask_state_comparison_count": int(exact_mask_state_comparison_count),
            "exact_mask_state_mismatch_count": int(exact_mask_state_mismatch_count),
            "exact_mask_state_mismatch_fraction": (
                float(exact_mask_state_mismatch_count) / exact_mask_state_comparison_count
                if exact_mask_state_comparison_count
                else 0.0
            ),
            "exact_mask_unsafe_normal_count": int(exact_mask_unsafe_normal_count),
            "exact_mask_blocked_safe_normal_count": int(exact_mask_blocked_safe_normal_count),
        }

    def close(self):
        """Close database connection."""
        self.conn.close()
