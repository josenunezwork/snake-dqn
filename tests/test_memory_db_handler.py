"""Tests for memory_db_handler module."""

import struct

import pytest
import torch

from src.data.memory_db_handler import (
    STATE_BLOB_SIZE,
    STATE_SIZE,
    MemoryDBHandler,
    build_replay_quality_stats,
    format_replay_quality_stats,
    format_replay_quality_warnings,
    resolve_replay_quality_fraction,
    validate_replay_quality_gates,
)
from src.utils.tensor_utils import memories_to_dicts


def make_state(
    value: float = 0.0,
    boost_available: float = 0.0,
    per_action_danger: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Create a deterministic 58-float state vector."""
    state = [float(value + idx) for idx in range(58)]
    state[54:57] = [float(danger) for danger in per_action_danger]
    state[57] = boost_available
    return state


def make_memory(**overrides: object) -> dict:
    """Create a valid replay row with optional field overrides."""
    memory = {
        "state": make_state(0.0),
        "action": 1,
        "reward": 1.0,
        "next_state": make_state(100.0),
        "done": False,
        "priority": 1.0,
        "bootstrap_steps": 1,
    }
    memory.update(overrides)
    return memory


def assert_rejects_memory(temp_db, memory: dict, match: str) -> None:
    """Assert invalid replay rows fail before anything is committed."""
    handler = MemoryDBHandler(temp_db)
    try:
        with pytest.raises(ValueError, match=match):
            handler.save_memories(snake_id=0, memories=[memory])
        assert handler.get_memory_count(policy_type="apex") == 0
    finally:
        handler.close()


class TestMemoryDBHandler:
    """Test suite for MemoryDBHandler class."""

    def test_initialization(self, temp_db):
        """Test database handler initializes correctly."""
        handler = MemoryDBHandler(temp_db)
        handler.close()

        assert True  # If no exception, initialization succeeded

    def test_replay_metadata_round_trip(self, temp_db):
        """Replay DB metadata should persist JSON-serializable generation contracts."""
        handler = MemoryDBHandler(temp_db)

        handler.set_metadata("generation.state_size", 58)
        handler.update_metadata(
            {
                "generation.mode": "single",
                "generation.quality_gates": {"min_terminal_fraction": 0.005},
            }
        )

        assert handler.get_metadata("generation.state_size") == 58
        assert handler.get_metadata("generation.mode") == "single"
        assert handler.get_metadata() == {
            "generation.mode": "single",
            "generation.quality_gates": {"min_terminal_fraction": 0.005},
            "generation.state_size": 58,
        }
        handler.close()

    def test_replay_metadata_rejects_empty_keys(self, temp_db):
        """Metadata keys should be explicit so callers cannot hide malformed entries."""
        handler = MemoryDBHandler(temp_db)
        try:
            with pytest.raises(ValueError, match="metadata key"):
                handler.set_metadata("  ", "bad")
            with pytest.raises(ValueError, match="metadata key"):
                handler.get_metadata("")
        finally:
            handler.close()

    def test_save_and_load_memories(self, temp_db):
        """Test saving and loading memories."""
        handler = MemoryDBHandler(temp_db)

        # Create sample memories
        memories = [
            {
                "state": make_state(0.0),
                "action": 1,
                "reward": 10.5,
                "next_state": make_state(100.0),
                "done": False,
                "priority": 1.0,
                "bootstrap_steps": 3,
            },
            {
                "state": make_state(200.0),
                "action": 2,
                "reward": -5.0,
                "next_state": make_state(300.0),
                "done": True,
                "priority": 0.5,
            },
        ]

        handler.save_memories(snake_id=0, memories=memories)

        # Load memories
        states, actions, rewards, next_states, dones, priorities, bootstrap_steps = (
            handler.load_memories(snake_id=0)
        )

        assert len(states) == 2
        assert len(actions) == 2
        assert actions[0] == 1
        assert rewards[0] == 10.5
        assert dones[1] is True
        assert bootstrap_steps == [3, 1]

        handler.close()

    def test_save_and_load_next_action_mask_when_requested(self, temp_db):
        """Exact next-action masks should persist for offline training targets."""
        handler = MemoryDBHandler(temp_db)
        mask = [False, True, False, False, False, False]
        memory = make_memory(next_action_mask=mask)

        handler.save_memories(snake_id=0, memories=[memory])
        default_rows = handler.load_memories(snake_id=0)
        rows_with_masks = handler.load_memories_for_policy(
            policy_type="apex",
            snake_id=0,
            limit=None,
            order_by="id",
            include_action_masks=True,
        )

        assert len(default_rows) == 7
        assert len(rows_with_masks) == 8
        assert rows_with_masks[7] == [tuple(mask)]

        handler.cursor.execute("SELECT next_action_mask FROM memories_standard")
        assert handler.cursor.fetchone()[0] == 2
        handler.close()

    def test_save_and_load_empty_exact_next_action_mask_when_requested(self, temp_db):
        """All-false exact masks should persist as trapped next states."""
        handler = MemoryDBHandler(temp_db)
        mask = [False, False, False, False, False, False]
        memory = make_memory(next_action_mask=mask)

        handler.save_memories(snake_id=0, memories=[memory])
        rows_with_masks = handler.load_memories_for_policy(
            policy_type="apex",
            snake_id=0,
            limit=None,
            order_by="id",
            include_action_masks=True,
        )

        assert rows_with_masks[7] == [tuple(mask)]

        handler.cursor.execute("SELECT next_action_mask FROM memories_standard")
        assert handler.cursor.fetchone()[0] == 0
        handler.close()

    def test_replay_quality_uses_empty_exact_mask_for_trapped_next_state(self, temp_db):
        """Exact empty masks should mark trapped targets even when state features look safe."""
        handler = MemoryDBHandler(temp_db)
        memories = [
            make_memory(
                next_state=make_state(100.0, per_action_danger=(0.0, 0.0, 0.0)),
                next_action_mask=[False, False, False, False, False, False],
            )
        ]
        handler.save_memories(snake_id=0, memories=memories)

        stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=0)

        assert stats["nonterminal_trapped_next_state_count"] == 1
        assert stats["nonterminal_trapped_next_state_fraction"] == pytest.approx(1.0)
        handler.close()

    def test_replay_quality_uses_nonempty_exact_mask_over_trapped_state_features(self, temp_db):
        """Exact non-empty masks should not be reported as trapped due to approximate features."""
        handler = MemoryDBHandler(temp_db)
        memories = [
            make_memory(
                next_state=make_state(100.0, per_action_danger=(1.0, 1.0, 1.0)),
                next_action_mask=[False, True, False, False, False, False],
            )
        ]
        handler.save_memories(snake_id=0, memories=memories)

        stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=0)

        assert stats["nonterminal_trapped_next_state_count"] == 0
        assert stats["nonterminal_trapped_next_state_fraction"] == pytest.approx(0.0)
        handler.close()

    def test_replay_quality_rejects_stored_exact_masks_with_unknown_bits(self, temp_db):
        """Corrupt DB masks should not inflate exact-mask or boost-mask coverage."""
        handler = MemoryDBHandler(temp_db)
        handler.save_memories(
            snake_id=0,
            memories=[
                make_memory(
                    next_state=make_state(100.0, per_action_danger=(0.0, 0.0, 0.0)),
                    next_action_mask=[False, True, False, False, False, False],
                )
            ],
        )
        handler.cursor.execute("UPDATE memories_standard SET next_action_mask = ?", (1 << 6,))
        handler.conn.commit()

        stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=0)

        assert stats["invalid_action_mask_count"] == 1
        assert stats["mask_count"] == 0
        assert stats["nonterminal_mask_count"] == 0
        assert stats["boost_mask_count"] == 0
        assert any(
            "Invalid exact next-action masks: 1/1" in line
            for line in format_replay_quality_stats(stats)
        )
        assert any(
            "invalid exact next-action masks" in warning
            for warning in format_replay_quality_warnings(stats)
        )
        with pytest.raises(RuntimeError, match="invalid exact next-action masks"):
            validate_replay_quality_gates(stats)
        handler.close()

    def test_replay_quality_stats_summarize_learning_signal(self, temp_db):
        """Replay diagnostics should expose action, reward, terminal, and mask coverage."""
        handler = MemoryDBHandler(temp_db)

        memories = [
            make_memory(action=0, reward=-1.0, done=True, priority=0.5, bootstrap_steps=1),
            make_memory(
                action=1,
                reward=0.0,
                done=False,
                priority=1.0,
                bootstrap_steps=2,
                next_action_mask=[False, True, False, False, False, False],
            ),
            make_memory(
                action=1,
                reward=3.0,
                done=False,
                priority=2.0,
                bootstrap_steps=3,
                next_action_mask=[True, True, False, False, False, False],
            ),
        ]
        handler.save_memories(snake_id=7, memories=memories)

        stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=7)

        assert stats["count"] == 3
        assert stats["done_count"] == 1
        assert stats["terminal_fraction"] == pytest.approx(1 / 3)
        assert stats["nonterminal_count"] == 2
        assert stats["mask_count"] == 2
        assert stats["mask_fraction"] == pytest.approx(2 / 3)
        assert stats["nonterminal_mask_count"] == 2
        assert stats["nonterminal_mask_fraction"] == pytest.approx(1.0)
        assert stats["boost_mask_count"] == 0
        assert stats["boost_mask_fraction"] == pytest.approx(0.0)
        assert stats["reward_min"] == pytest.approx(-1.0)
        assert stats["reward_avg"] == pytest.approx(2 / 3)
        assert stats["reward_max"] == pytest.approx(3.0)
        assert stats["reward_negative_count"] == 1
        assert stats["reward_zero_count"] == 1
        assert stats["reward_positive_count"] == 1
        assert stats["terminal_reward_negative_count"] == 1
        assert stats["terminal_reward_zero_count"] == 0
        assert stats["terminal_reward_positive_count"] == 0
        assert stats["terminal_nonnegative_reward_count"] == 0
        assert stats["terminal_nonnegative_reward_fraction"] == pytest.approx(0.0)
        assert stats["terminal_immediate_count"] == 1
        assert stats["immediate_terminal_fraction"] == pytest.approx(1 / 3)
        assert stats["terminal_immediate_fraction"] == pytest.approx(1.0)
        assert stats["terminal_multistep_count"] == 0
        assert stats["terminal_multistep_fraction"] == pytest.approx(0.0)
        assert stats["terminal_immediate_nonnegative_reward_count"] == 0
        assert stats["terminal_immediate_nonnegative_reward_fraction"] == pytest.approx(0.0)
        assert stats["terminal_multistep_nonnegative_reward_count"] == 0
        assert stats["priority_min"] == pytest.approx(0.5)
        assert stats["priority_avg"] == pytest.approx(7 / 6)
        assert stats["priority_max"] == pytest.approx(2.0)
        assert stats["bootstrap_steps_min"] == 1
        assert stats["bootstrap_steps_avg"] == pytest.approx(2.0)
        assert stats["bootstrap_steps_max"] == 3
        assert stats["snake_count"] == 1
        assert stats["snake_rows_min"] == 3
        assert stats["snake_rows_avg"] == pytest.approx(3.0)
        assert stats["snake_rows_max"] == 3
        assert stats["dominant_snake_fraction"] == pytest.approx(1.0)
        assert stats["action_counts"] == {0: 1, 1: 2}
        assert stats["active_action_count"] == 2
        assert stats["dominant_action"] == 1
        assert stats["dominant_action_fraction"] == pytest.approx(2 / 3)
        assert stats["normalized_action_entropy"] == pytest.approx(0.355245, abs=1e-6)
        assert stats["boost_available_count"] == 0
        assert stats["boost_available_fraction"] == pytest.approx(0.0)
        assert stats["malformed_boost_feature_count"] == 0

        lines = format_replay_quality_stats(stats)

        assert any("Rows: 3" in line for line in lines)
        assert any("Nonterminal exact masks: 2/2 (100.0%)" in line for line in lines)
        assert any("Actions: 0:1, 1:2" in line for line in lines)
        assert any(
            "Action coverage: 2/6 | dominant: 1 (66.7%) | entropy: 35.5%" in line for line in lines
        )
        assert any("Reward signs neg/zero/pos: 1/1/1" in line for line in lines)
        assert any(
            "Terminal reward signs neg/zero/pos: 1/0/0; nonnegative=0/1 (0.0%); "
            "one_step_bad=0/1 (0.0%); n_step_nonnegative=0/0 (0.0%)" in line
            for line in lines
        )
        assert any("Rows per snake_id min/avg/max: 3/3.00/3" in line for line in lines)
        assert any("Boost available states: 0 (0.0%)" in line for line in lines)
        assert any("Exact masks allowing boost: 0 (0.0%)" in line for line in lines)
        handler.close()

    def test_replay_quality_rejects_nonnegative_terminal_rewards_from_db(self, temp_db):
        """DB-backed audits should fail when terminal rows do not teach collision loss."""
        handler = MemoryDBHandler(temp_db)
        memories = [
            make_memory(action=0, reward=0.0, done=True),
            make_memory(action=1, reward=1.0, done=True),
            make_memory(action=2, reward=-0.1, done=False),
            make_memory(action=3, reward=0.5, done=False),
        ]
        handler.save_memories(snake_id=7, memories=memories)

        stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=7)

        assert stats["terminal_reward_negative_count"] == 0
        assert stats["terminal_reward_zero_count"] == 1
        assert stats["terminal_reward_positive_count"] == 1
        assert stats["terminal_nonnegative_reward_count"] == 2
        assert stats["terminal_nonnegative_reward_fraction"] == pytest.approx(1.0)
        assert stats["terminal_immediate_count"] == 2
        assert stats["terminal_multistep_count"] == 0
        assert stats["terminal_immediate_nonnegative_reward_count"] == 2
        assert stats["terminal_immediate_nonnegative_reward_fraction"] == pytest.approx(1.0)
        with pytest.raises(RuntimeError, match="one-step terminal rows .* non-negative rewards"):
            validate_replay_quality_gates(stats)
        handler.close()

    def test_replay_quality_rejects_nonnegative_multistep_terminal_returns_from_db(self, temp_db):
        """DB-backed gates should reject aggregated returns that make death non-negative."""
        handler = MemoryDBHandler(temp_db)
        memories = [
            make_memory(
                action=1,
                reward=3.0 + 0.99 * -3.0,
                done=True,
                bootstrap_steps=2,
            )
        ]
        handler.save_memories(snake_id=7, memories=memories)

        stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=7)

        assert stats["terminal_reward_positive_count"] == 1
        assert stats["terminal_nonnegative_reward_count"] == 1
        assert stats["terminal_immediate_count"] == 0
        assert stats["immediate_terminal_fraction"] == pytest.approx(0.0)
        assert stats["terminal_immediate_fraction"] == pytest.approx(0.0)
        assert stats["terminal_multistep_count"] == 1
        assert stats["terminal_multistep_fraction"] == pytest.approx(1.0)
        assert stats["terminal_immediate_nonnegative_reward_count"] == 0
        assert stats["terminal_multistep_nonnegative_reward_count"] == 1
        assert stats["terminal_multistep_nonnegative_reward_fraction"] == pytest.approx(1.0)
        with pytest.raises(RuntimeError, match="n-step terminal rows .* non-negative returns"):
            validate_replay_quality_gates(stats)
        handler.close()

    def test_db_replay_quality_stats_count_current_state_invalid_actions(self, temp_db):
        """DB audits should expose replay actions contradicted by current-state features."""
        handler = MemoryDBHandler(temp_db)
        memories = [
            make_memory(
                state=make_state(0.0, per_action_danger=(0.0, 0.0, 0.0)),
                action=0,
            ),
            make_memory(
                state=make_state(1.0, per_action_danger=(0.0, 1.0, 0.0)),
                action=1,
                reward=-1.0,
                done=True,
            ),
            make_memory(
                state=make_state(2.0, boost_available=0.0, per_action_danger=(0.0, 0.0, 0.0)),
                action=4,
                reward=-1.0,
                done=True,
            ),
            make_memory(
                state=make_state(3.0, per_action_danger=(0.0, 1.0, 0.0)),
                action=1,
                reward=-0.5,
                done=False,
            ),
        ]

        handler.save_memories(snake_id=7, memories=memories)
        stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=7)
        lines = format_replay_quality_stats(stats)

        assert stats["current_action_state_comparison_count"] == 4
        assert stats["invalid_current_action_count"] == 3
        assert stats["invalid_current_action_fraction"] == pytest.approx(3 / 4)
        assert stats["invalid_current_normal_action_count"] == 2
        assert stats["invalid_current_boost_action_count"] == 1
        assert stats["terminal_invalid_current_action_count"] == 2
        assert stats["nonterminal_current_action_state_comparison_count"] == 2
        assert stats["nonterminal_invalid_current_action_count"] == 1
        assert stats["nonterminal_invalid_current_action_fraction"] == pytest.approx(1 / 2)
        assert any(
            "Current actions invalid by state: 3/4 (75.0%); nonterminal=1/2 (50.0%)" in line
            for line in lines
        )

        handler.close()

    def test_replay_quality_stats_count_non_finite_stored_state_as_malformed(self, temp_db):
        """Replay audits should flag old finite-width blobs containing NaN features."""
        handler = MemoryDBHandler(temp_db)
        state = [0.0] * STATE_SIZE
        state[1] = 1.0
        next_state = list(state)

        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    make_memory(
                        state=state,
                        next_state=next_state,
                        next_action_mask=[True, True, True, False, False, False],
                    )
                ],
            )
            corrupted_state = list(state)
            corrupted_state[5] = float("nan")
            handler.cursor.execute(
                "UPDATE memories_standard SET state = ?",
                (struct.pack(f"<{STATE_SIZE}f", *corrupted_state),),
            )
            handler.conn.commit()

            stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=0)

            assert stats["malformed_state_range_count"] == 1
            assert stats["malformed_state_feature_count"] == 1
            assert stats["malformed_state_feature_fraction"] == pytest.approx(0.5)
        finally:
            handler.close()

    def test_replay_quality_stats_count_invalid_stored_state_blob_as_malformed(self, temp_db):
        """Strict audits should reject corrupted blobs before offline loading fails."""
        handler = MemoryDBHandler(temp_db)
        state = [0.0] * STATE_SIZE
        state[1] = 1.0

        try:
            handler.save_memories(
                snake_id=0,
                memories=[
                    make_memory(
                        state=state,
                        next_state=list(state),
                        next_action_mask=[True, True, True, False, False, False],
                    )
                ],
            )
            handler.cursor.execute(
                "UPDATE memories_standard SET state = ?, next_state = ?",
                (struct.pack("<f", 0.0), struct.pack("<ff", 0.0, 1.0)),
            )
            handler.conn.commit()

            stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=0)

            assert stats["invalid_state_feature_count"] == 1
            assert stats["invalid_next_state_feature_count"] == 1
            assert stats["malformed_state_feature_fraction"] == pytest.approx(1.0)
            assert any(
                "invalid current-state" in warning
                for warning in format_replay_quality_warnings(stats)
            )
            assert any(
                "invalid next-state" in warning for warning in format_replay_quality_warnings(stats)
            )
            with pytest.raises(RuntimeError, match="malformed state-feature fraction"):
                validate_replay_quality_gates(
                    stats,
                    max_malformed_state_feature_fraction=0.0,
                )
        finally:
            handler.close()

    def test_replay_quality_stats_summarize_per_snake_distribution(self, temp_db):
        """Replay diagnostics should expose whether rows are balanced across snake ids."""
        handler = MemoryDBHandler(temp_db)
        try:
            handler.save_memories(
                snake_id=0,
                memories=[make_memory(reward=1.0) for _ in range(3)],
            )
            handler.save_memories(
                snake_id=1,
                memories=[make_memory(reward=-1.0) for _ in range(1)],
            )

            stats = handler.get_replay_quality_stats(policy_type="apex")

            assert stats["snake_count"] == 2
            assert stats["snake_rows_min"] == 1
            assert stats["snake_rows_avg"] == pytest.approx(2.0)
            assert stats["snake_rows_max"] == 3
            assert stats["dominant_snake_fraction"] == pytest.approx(0.75)
        finally:
            handler.close()

    def test_replay_quality_stats_count_boost_state_and_exact_masks(self, temp_db):
        """Replay diagnostics should show whether boost state/mask coverage is represented."""
        handler = MemoryDBHandler(temp_db)
        memories = [
            make_memory(
                state=make_state(0.0, boost_available=1.0),
                next_state=make_state(100.0, per_action_danger=(1.0, 1.0, 1.0)),
                action=0,
                next_action_mask=[True, False, False, False, True, False],
            ),
            make_memory(
                state=make_state(
                    10.0,
                    boost_available=1.0,
                    per_action_danger=(1.0, 1.0, 1.0),
                ),
                action=1,
                next_action_mask=[False, True, False, False, False, False],
            ),
            make_memory(
                state=make_state(20.0, boost_available=0.0),
                next_state=make_state(120.0, per_action_danger=(0.0, 2.0, 0.0)),
                action=4,
                next_action_mask=[False, False, True, False, False, True],
            ),
            make_memory(
                state=make_state(
                    30.0,
                    boost_available=2.0,
                    per_action_danger=(0.0, 2.0, 0.0),
                ),
                action=2,
                next_action_mask=None,
            ),
        ]
        handler.save_memories(snake_id=3, memories=memories)

        stats = handler.get_replay_quality_stats(policy_type="apex", snake_id=3)

        assert stats["mask_count"] == 3
        assert stats["nonterminal_mask_count"] == 3
        assert stats["boost_mask_count"] == 2
        assert stats["boost_mask_fraction"] == pytest.approx(0.5)
        assert stats["boost_available_count"] == 3
        assert stats["boost_available_fraction"] == pytest.approx(0.75)
        assert stats["malformed_boost_feature_count"] == 1
        assert stats["trapped_state_count"] == 1
        assert stats["trapped_state_fraction"] == pytest.approx(0.25)
        assert stats["nonterminal_trapped_state_count"] == 1
        assert stats["nonterminal_trapped_state_fraction"] == pytest.approx(0.25)
        assert stats["malformed_per_action_danger_count"] == 1
        assert stats["trapped_next_state_count"] == 0
        assert stats["trapped_next_state_fraction"] == pytest.approx(0.0)
        assert stats["nonterminal_trapped_next_state_count"] == 0
        assert stats["nonterminal_trapped_next_state_fraction"] == pytest.approx(0.0)
        assert stats["malformed_next_per_action_danger_count"] == 1
        handler.close()

    def test_replay_quality_warnings_flag_suspicious_datasets(self):
        """Replay warnings should point at issues that can make learning misleading."""
        stats = {
            "count": 256,
            "done_count": 0,
            "nonterminal_count": 256,
            "mask_count": 128,
            "nonterminal_mask_count": 128,
            "reward_min": 0.0,
            "reward_max": 0.0,
            "reward_negative_count": 0,
            "reward_positive_count": 0,
            "priority_min": 1.0,
            "priority_max": 1.0,
            "snake_count": 2,
            "snake_rows_min": 16,
            "snake_rows_max": 240,
            "dominant_snake_fraction": 240 / 256,
            "action_counts": {0: 256},
            "boost_available_count": 200,
            "boost_mask_count": 64,
            "malformed_boost_feature_count": 4,
            "malformed_per_action_danger_count": 3,
            "malformed_next_per_action_danger_count": 2,
            "nonterminal_trapped_next_state_count": 40,
            "nonterminal_trapped_next_state_fraction": 40 / 256,
        }

        warnings = format_replay_quality_warnings(stats)

        assert any("lack exact next-action masks" in warning for warning in warnings)
        assert any("normal action(s) 1, 2" in warning for warning in warnings)
        assert any("No terminal rows" in warning for warning in warnings)
        assert any("exact next-action masks allow boost" in warning for warning in warnings)
        assert any("rows mark boost available" in warning for warning in warnings)
        assert any("boost-available state feature outside" in warning for warning in warnings)
        assert any("per-action danger features outside" in warning for warning in warnings)
        assert any(
            "next-state per-action danger features outside" in warning for warning in warnings
        )
        assert any("No positive rewards" in warning for warning in warnings)
        assert any("No negative rewards" in warning for warning in warnings)
        assert any("One snake_id contributes" in warning for warning in warnings)
        assert any("Reward signal is flat" in warning for warning in warnings)
        assert any("Replay priorities are flat" in warning for warning in warnings)

    def test_replay_quality_warnings_flag_dominant_action_distribution(self):
        """Action imbalance should be visible even when every action appears."""
        stats = {
            "count": 600,
            "done_count": 10,
            "terminal_fraction": 10 / 600,
            "nonterminal_count": 590,
            "mask_count": 590,
            "nonterminal_mask_count": 590,
            "reward_min": -1.0,
            "reward_max": 1.0,
            "reward_negative_count": 10,
            "reward_positive_count": 20,
            "priority_min": 0.5,
            "priority_max": 1.0,
            "action_counts": {0: 480, 1: 24, 2: 24, 3: 24, 4: 24, 5: 24},
            "boost_available_count": 100,
            "boost_mask_count": 60,
        }

        lines = format_replay_quality_stats(stats)
        warnings = format_replay_quality_warnings(stats)

        assert any("Action coverage: 6/6 | dominant: 0 (80.0%)" in line for line in lines)
        assert any("Action 0 accounts for 80.0%" in warning for warning in warnings)

    def test_replay_quality_warnings_flag_low_terminal_fraction(self):
        """Sparse terminal samples should be visible before they disappear entirely."""
        stats = {
            "count": 1000,
            "done_count": 2,
            "terminal_fraction": 0.002,
            "nonterminal_count": 998,
            "mask_count": 998,
            "nonterminal_mask_count": 998,
            "reward_min": -1.0,
            "reward_max": 1.0,
            "reward_negative_count": 2,
            "reward_positive_count": 5,
            "priority_min": 0.5,
            "priority_max": 1.0,
            "action_counts": {0: 334, 1: 333, 2: 333},
            "boost_available_count": 0,
            "boost_mask_count": 0,
        }

        warnings = format_replay_quality_warnings(stats)

        assert any("Only 2/1,000 rows" in warning for warning in warnings)

    def test_replay_quality_warnings_flag_frequent_nonterminal_trapped_next_states(self):
        """Too many nonterminal dead-end targets should be visible in replay diagnostics."""
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "mask_count": 990,
            "nonterminal_mask_count": 990,
            "reward_min": -1.0,
            "reward_max": 1.0,
            "reward_negative_count": 10,
            "reward_positive_count": 20,
            "priority_min": 0.5,
            "priority_max": 1.0,
            "action_counts": {0: 334, 1: 333, 2: 333},
            "boost_available_count": 0,
            "boost_mask_count": 0,
            "nonterminal_trapped_next_state_count": 100,
            "nonterminal_trapped_next_state_fraction": 100 / 990,
        }

        warnings = format_replay_quality_warnings(stats)

        assert any("100/990 nonterminal next-state targets" in warning for warning in warnings)

    def test_replay_quality_fraction_accepts_valid_values(self):
        """All replay entry points share the same optional fraction gate parsing."""
        assert resolve_replay_quality_fraction(None, "gate") == pytest.approx(0.0)
        assert resolve_replay_quality_fraction(0.25, "gate") == pytest.approx(0.25)
        assert resolve_replay_quality_fraction("1.0", "gate") == pytest.approx(1.0)

    @pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), "bad"])
    def test_replay_quality_fraction_rejects_invalid_values(self, value):
        with pytest.raises(ValueError, match=r"gate must be finite and in \[0, 1\]"):
            resolve_replay_quality_fraction(value, "gate")

    def test_replay_quality_terminal_gate_raises_with_context(self):
        stats = {
            "count": 1000,
            "done_count": 2,
            "terminal_fraction": 0.002,
            "nonterminal_count": 998,
            "nonterminal_mask_count": 998,
            "nonterminal_mask_fraction": 1.0,
        }

        with pytest.raises(RuntimeError, match="Shared replay terminal fraction"):
            validate_replay_quality_gates(
                stats,
                min_terminal_fraction=0.005,
                context="Shared replay",
            )

    def test_replay_quality_exact_mask_gate_raises_with_context(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 100,
            "nonterminal_mask_fraction": 100 / 990,
        }

        with pytest.raises(RuntimeError, match="Shared replay exact-mask fraction"):
            validate_replay_quality_gates(
                stats,
                min_exact_mask_fraction=0.5,
                context="Shared replay",
            )

    def test_replay_quality_trapped_next_gate_raises_with_context(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "nonterminal_trapped_next_state_count": 100,
            "nonterminal_trapped_next_state_fraction": 100 / 990,
        }

        with pytest.raises(RuntimeError, match="Shared replay trapped-next-state fraction"):
            validate_replay_quality_gates(
                stats,
                max_nonterminal_trapped_next_fraction=0.05,
                context="Shared replay",
            )

    def test_replay_quality_action_coverage_gate_raises_with_context(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "action_counts": {0: 500, 1: 500},
        }

        with pytest.raises(RuntimeError, match="Shared replay action coverage"):
            validate_replay_quality_gates(
                stats,
                min_action_coverage_fraction=0.5,
                context="Shared replay",
            )

    def test_replay_quality_dominant_action_gate_raises_with_context(self):
        stats = {
            "count": 1000,
            "done_count": 10,
            "terminal_fraction": 0.01,
            "nonterminal_count": 990,
            "nonterminal_mask_count": 990,
            "nonterminal_mask_fraction": 1.0,
            "action_counts": {0: 850, 1: 50, 2: 50, 3: 25, 4: 15, 5: 10},
        }

        with pytest.raises(RuntimeError, match="Shared replay dominant-action fraction"):
            validate_replay_quality_gates(
                stats,
                max_dominant_action_fraction=0.8,
                context="Shared replay",
            )

    def test_replay_quality_warnings_ignore_missing_terminal_masks(self):
        """Terminal rows do not need next-action masks because they do not bootstrap."""
        stats = {
            "count": 4,
            "done_count": 2,
            "nonterminal_count": 2,
            "mask_count": 2,
            "nonterminal_mask_count": 2,
            "reward_min": -1.0,
            "reward_max": 1.0,
            "reward_negative_count": 1,
            "reward_positive_count": 1,
            "priority_min": 0.5,
            "priority_max": 1.0,
            "action_counts": {0: 2, 1: 1, 2: 1},
            "boost_available_count": 0,
            "boost_mask_count": 0,
        }

        warnings = format_replay_quality_warnings(stats)

        assert not any("lack exact next-action masks" in warning for warning in warnings)

    def test_replay_quality_warnings_flag_boost_action_state_mismatch(self):
        """Boost actions without boost-available states indicate action/state drift."""
        stats = {
            "count": 4,
            "done_count": 1,
            "nonterminal_count": 3,
            "mask_count": 4,
            "nonterminal_mask_count": 3,
            "reward_min": -1.0,
            "reward_max": 1.0,
            "reward_negative_count": 1,
            "reward_positive_count": 1,
            "priority_min": 0.5,
            "priority_max": 1.0,
            "action_counts": {4: 4},
            "boost_available_count": 0,
            "boost_mask_count": 0,
        }

        warnings = format_replay_quality_warnings(stats)

        assert any("boost-action rows exist" in warning for warning in warnings)

    def test_loaded_replay_quality_stats_count_boost_feature_rows(self):
        """Loaded-subset diagnostics should include boost availability from states."""
        states = [
            make_state(0.0, boost_available=0.0),
            make_state(1.0, boost_available=1.0, per_action_danger=(1.0, 1.0, 1.0)),
            make_state(2.0, boost_available=3.0, per_action_danger=(0.0, 2.0, 0.0)),
        ]
        next_states = [
            make_state(10.0, per_action_danger=(0.0, 2.0, 0.0)),
            make_state(11.0, per_action_danger=(1.0, 1.0, 1.0)),
            make_state(12.0, per_action_danger=(1.0, 1.0, 1.0)),
        ]

        stats = build_replay_quality_stats(
            actions=[0, 1, 2],
            rewards=[0.0, 1.0, -1.0],
            dones=[False, False, True],
            priorities=[1.0, 2.0, 3.0],
            bootstrap_steps=[1, 2, 3],
            next_action_masks=[
                [True, False, False, False, False, False],
                [False, False, False, True, False, False],
                None,
            ],
            states=states,
            next_states=next_states,
        )

        assert stats["mask_count"] == 2
        assert stats["terminal_fraction"] == pytest.approx(1 / 3)
        assert stats["nonterminal_count"] == 2
        assert stats["nonterminal_mask_count"] == 2
        assert stats["nonterminal_mask_fraction"] == pytest.approx(1.0)
        assert stats["boost_mask_count"] == 1
        assert stats["boost_mask_fraction"] == pytest.approx(1 / 2)
        assert stats["reward_negative_count"] == 1
        assert stats["reward_zero_count"] == 1
        assert stats["reward_positive_count"] == 1
        assert stats["boost_available_count"] == 2
        assert stats["boost_available_fraction"] == pytest.approx(2 / 3)
        assert stats["malformed_boost_feature_count"] == 1
        assert stats["trapped_state_count"] == 1
        assert stats["trapped_state_fraction"] == pytest.approx(1 / 3)
        assert stats["nonterminal_trapped_state_count"] == 1
        assert stats["nonterminal_trapped_state_fraction"] == pytest.approx(1 / 2)
        assert stats["malformed_per_action_danger_count"] == 1
        assert stats["trapped_next_state_count"] == 0
        assert stats["trapped_next_state_fraction"] == pytest.approx(0.0)
        assert stats["nonterminal_trapped_next_state_count"] == 0
        assert stats["nonterminal_trapped_next_state_fraction"] == pytest.approx(0.0)

    def test_loaded_replay_quality_uses_exact_masks_for_trapped_next_states(self):
        """Loaded-subset diagnostics should match learner target-mask semantics."""
        states = [
            make_state(0.0),
            make_state(1.0),
        ]
        next_states = [
            make_state(10.0, per_action_danger=(0.0, 0.0, 0.0)),
            make_state(11.0, per_action_danger=(1.0, 1.0, 1.0)),
        ]

        stats = build_replay_quality_stats(
            actions=[0, 1],
            rewards=[0.0, 1.0],
            dones=[False, False],
            priorities=[1.0, 2.0],
            bootstrap_steps=[1, 1],
            next_action_masks=[
                [False, False, False, False, False, False],
                [False, True, False, False, False, False],
            ],
            states=states,
            next_states=next_states,
        )

        assert stats["trapped_next_state_count"] == 1
        assert stats["trapped_next_state_fraction"] == pytest.approx(0.5)
        assert stats["nonterminal_trapped_next_state_count"] == 1
        assert stats["nonterminal_trapped_next_state_fraction"] == pytest.approx(0.5)
        assert stats["malformed_next_per_action_danger_count"] == 0
        assert stats["valid_next_state_feature_count"] == 2

    def test_loaded_replay_quality_stats_summarize_snake_ids(self):
        """Loaded-subset diagnostics should include producer coverage when IDs are loaded."""
        stats = build_replay_quality_stats(
            actions=[0, 1, 2, 0],
            rewards=[0.0, 1.0, -1.0, 0.5],
            dones=[False, False, True, False],
            priorities=[1.0, 2.0, 3.0, 4.0],
            bootstrap_steps=[1, 2, 3, 1],
            snake_ids=[0, 1, 1, 1],
        )

        assert stats["snake_count"] == 2
        assert stats["snake_rows_min"] == 1
        assert stats["snake_rows_avg"] == pytest.approx(2.0)
        assert stats["snake_rows_max"] == 3
        assert stats["dominant_snake_fraction"] == pytest.approx(0.75)

    def test_save_migrates_existing_db_with_next_action_mask_column(self, temp_db):
        """Older databases should gain the nullable next_action_mask column."""
        import sqlite3

        conn = sqlite3.connect(temp_db)
        conn.execute(
            """
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
                bootstrap_steps INTEGER DEFAULT 1
            )
            """
        )
        conn.commit()
        conn.close()

        handler = MemoryDBHandler(temp_db)
        try:
            handler.cursor.execute("PRAGMA table_info(memories_standard)")
            columns = {row[1] for row in handler.cursor.fetchall()}
            assert "next_action_mask" in columns
        finally:
            handler.close()

    def test_load_limited_memories(self, temp_db):
        """Test that loading memories respects the limit."""
        handler = MemoryDBHandler(temp_db)

        # Save many memories
        memories = [
            {
                "state": make_state(float(i)),
                "action": i % 6,
                "reward": float(i),
                "next_state": make_state(float(i + 1)),
                "done": False,
                "priority": 1.0,
            }
            for i in range(5000)
        ]

        handler.save_memories(snake_id=0, memories=memories)

        # Load memories (should be limited to 4000)
        states, _, _, _, _, _, _ = handler.load_memories(snake_id=0)

        assert len(states) <= 4000

        handler.close()

    def test_load_all_memories_with_no_limit(self, temp_db):
        """Test loading all memories when limit=None."""
        handler = MemoryDBHandler(temp_db)

        memories = [
            {
                "state": make_state(float(i)),
                "action": i % 6,
                "reward": float(i),
                "next_state": make_state(float(i + 1)),
                "done": False,
                "priority": 1.0,
            }
            for i in range(5000)
        ]

        handler.save_memories(snake_id=0, memories=memories)

        states, _, _, _, _, _, _ = handler.load_memories_for_policy(
            policy_type="apex", snake_id=0, limit=None
        )

        assert len(states) == 5000

        handler.close()

    def test_clear_memories(self, temp_db):
        """Test clearing memories."""
        handler = MemoryDBHandler(temp_db)

        # Save memories
        memories = [
            {
                "state": make_state(0.0),
                "action": 0,
                "reward": 1.0,
                "next_state": make_state(1.0),
                "done": False,
                "priority": 1.0,
            }
        ]

        handler.save_memories(snake_id=0, memories=memories)
        handler.clear_memories(snake_id=0)

        # Try to load
        states, _, _, _, _, _, _ = handler.load_memories(snake_id=0)

        assert len(states) == 0

        handler.close()

    def test_clear_all_memories(self, temp_db):
        """Test clearing all memories from database."""
        handler = MemoryDBHandler(temp_db)

        # Save memories for multiple snakes
        memory = {
            "state": make_state(0.0),
            "action": 0,
            "reward": 1.0,
            "next_state": make_state(1.0),
            "done": False,
            "priority": 1.0,
        }

        handler.save_memories(0, [memory])
        handler.save_memories(1, [memory])

        handler.clear_memories()  # Clear all

        # Try to load from both
        states1, _, _, _, _, _, _ = handler.load_memories(snake_id=0)
        states2, _, _, _, _, _, _ = handler.load_memories(snake_id=1)

        assert len(states1) == 0
        assert len(states2) == 0

        handler.close()

    def test_load_memories_by_priority(self, temp_db):
        """Test that memories are loaded ordered by priority."""
        handler = MemoryDBHandler(temp_db)

        memories = [
            {
                "state": make_state(float(i)),
                "action": i % 6,
                "reward": 1.0,
                "next_state": make_state(float(i + 1)),
                "done": False,
                "priority": float(i + 1),  # Increasing priority
            }
            for i in range(10)
        ]

        handler.save_memories(snake_id=0, memories=memories)

        # Load memories
        _, actions, _, _, _, priorities, _ = handler.load_memories(snake_id=0)

        # Check that priorities are in descending order
        assert priorities[0] >= priorities[-1]

        handler.close()

    def test_load_memories_by_id_preserves_insertion_order(self, temp_db):
        """ID ordering loads the oldest rows exactly for merge/debug flows."""
        handler = MemoryDBHandler(temp_db)

        memories = [
            {
                "state": [float(i)] * 58,
                "action": i,
                "reward": 1.0,
                "next_state": [float(i + 1)] * 58,
                "done": False,
                "priority": float(i + 1),
            }
            for i in range(5)
        ]

        handler.save_memories(snake_id=0, memories=memories)

        _, actions, _, _, _, _, _ = handler.load_memories_for_policy(
            policy_type="apex",
            snake_id=0,
            limit=3,
            order_by="id",
        )

        assert actions == [0, 1, 2]

        handler.close()

    def test_load_memories_by_uniform_id_spreads_capped_load(self, temp_db):
        """Uniform ID ordering samples across a capped generated replay database."""
        handler = MemoryDBHandler(temp_db)

        memories = [
            {
                "state": [float(i)] * 58,
                "action": i % 6,
                "reward": 1.0,
                "next_state": [float(i + 1)] * 58,
                "done": False,
                "priority": 1.0,
            }
            for i in range(10)
        ]

        handler.save_memories(snake_id=0, memories=memories)

        states, actions, _, _, _, _, _ = handler.load_memories_for_policy(
            policy_type="apex",
            snake_id=0,
            limit=4,
            order_by="id_uniform",
        )

        assert [int(state[0]) for state in states] == [0, 3, 6, 9]
        assert actions == [0, 3, 0, 3]

        handler.close()

    def test_save_coerces_state_blobs_to_float32(self, temp_db):
        """List observations should save as fixed-size float32 state blobs."""
        handler = MemoryDBHandler(temp_db)

        state = [float(idx) for idx in range(58)]
        next_state = [idx + 0.5 for idx in state]
        memory = {
            "state": state,
            "action": 1,
            "reward": 1.0,
            "next_state": next_state,
            "done": False,
            "priority": 1.0,
        }

        handler.save_memories(snake_id=0, memories=[memory])
        handler.cursor.execute("SELECT length(state), length(next_state) FROM memories_standard")
        state_blob_size, next_state_blob_size = handler.cursor.fetchone()
        states, _, _, next_states, _, _, _ = handler.load_memories_for_policy(
            policy_type="apex",
            snake_id=0,
            limit=None,
            order_by="id",
        )

        assert state_blob_size == STATE_BLOB_SIZE
        assert next_state_blob_size == STATE_BLOB_SIZE
        assert isinstance(states[0], tuple)
        assert len(states[0]) == 58
        assert isinstance(next_states[0], tuple)
        assert len(next_states[0]) == 58
        assert states[0] == pytest.approx(state)
        assert next_states[0] == pytest.approx(next_state)

        handler.close()

    def test_save_rejects_wrong_state_size(self, temp_db):
        """Replay rows with the wrong observation width should fail before insert."""
        handler = MemoryDBHandler(temp_db)

        memory = {
            "state": [0.0] * 57,
            "action": 1,
            "reward": 1.0,
            "next_state": [0.0] * 58,
            "done": False,
            "priority": 1.0,
        }

        try:
            handler.save_memories(snake_id=0, memories=[memory])
        except ValueError as exc:
            assert "state" in str(exc)
            assert "58" in str(exc)
        else:
            raise AssertionError("Expected ValueError for malformed replay state")
        finally:
            assert handler.get_memory_count(policy_type="apex") == 0
            handler.close()

    @pytest.mark.parametrize("action", [-1, 6, 1.5, True, "1"])
    def test_save_rejects_invalid_action(self, temp_db, action):
        """Stored actions must match the six-output policy head."""
        assert_rejects_memory(temp_db, make_memory(action=action), "action")

    @pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
    def test_save_rejects_non_finite_reward(self, temp_db, reward):
        """Replay rewards must be finite before training sees them."""
        assert_rejects_memory(temp_db, make_memory(reward=reward), "reward")

    @pytest.mark.parametrize("done", [2, -1, 0.5, "False", float("nan")])
    def test_save_rejects_invalid_done(self, temp_db, done):
        """Terminal flags must be explicit bool/0/1 values."""
        assert_rejects_memory(temp_db, make_memory(done=done), "done")

    @pytest.mark.parametrize("priority", [0.0, -1.0, float("nan"), float("inf")])
    def test_save_rejects_invalid_priority(self, temp_db, priority):
        """Prioritized replay cannot train from non-positive or non-finite priority."""
        assert_rejects_memory(temp_db, make_memory(priority=priority), "priority")

    @pytest.mark.parametrize("bootstrap_steps", [0, -1, 1.5, True, "2"])
    def test_save_rejects_invalid_bootstrap_steps(self, temp_db, bootstrap_steps):
        """N-step metadata should be valid instead of silently rewritten."""
        assert_rejects_memory(
            temp_db,
            make_memory(bootstrap_steps=bootstrap_steps),
            "bootstrap_steps",
        )

    @pytest.mark.parametrize(
        "next_action_mask",
        [
            [True] * 5,
            [True] * 7,
            [True, False, False, False, False, 2],
            [True, False, False, False, False, "yes"],
        ],
    )
    def test_save_rejects_invalid_next_action_mask(self, temp_db, next_action_mask):
        """Persisted exact masks must match the six-action output head."""
        assert_rejects_memory(
            temp_db,
            make_memory(next_action_mask=next_action_mask),
            "next_action_mask",
        )

    def test_save_rejects_nested_next_action_mask_with_six_values(self, temp_db):
        """Persisted exact masks should be one flat six-action vector."""
        assert_rejects_memory(
            temp_db,
            make_memory(next_action_mask=[[True, False, False], [False, False, False]]),
            "next_action_mask must have shape",
        )

    def test_save_rejects_non_finite_state_value(self, temp_db):
        """NaN or infinite state features would poison learner tensors."""
        state = make_state(0.0)
        state[10] = float("nan")

        assert_rejects_memory(temp_db, make_memory(state=state), "state")

    def test_load_rejects_corrupt_stored_state_blob(self, temp_db):
        """Old/corrupt databases should fail loudly instead of dropping rows."""
        handler = MemoryDBHandler(temp_db)
        try:
            handler.save_memories(snake_id=0, memories=[make_memory()])
            handler.cursor.execute("UPDATE memories_standard SET state = ?", (b"too-short",))
            handler.conn.commit()

            with pytest.raises(ValueError, match="Stored replay row.*state"):
                handler.load_memories_for_policy("apex", limit=None, order_by="id")
        finally:
            handler.close()

    def test_load_rejects_non_finite_stored_state_blob(self, temp_db):
        """Correctly sized old blobs still fail if they contain NaN or infinity."""
        handler = MemoryDBHandler(temp_db)
        try:
            handler.save_memories(snake_id=0, memories=[make_memory()])
            values = [0.0] * STATE_SIZE
            values[3] = float("nan")
            handler.cursor.execute(
                "UPDATE memories_standard SET state = ?",
                (struct.pack(f"<{STATE_SIZE}f", *values),),
            )
            handler.conn.commit()

            with pytest.raises(ValueError, match="Stored replay row.*state.*finite"):
                handler.load_memories_for_policy("apex", limit=None, order_by="id")
        finally:
            handler.close()

    @pytest.mark.parametrize(
        ("column", "value", "match"),
        [
            ("action", 6, "action"),
            ("reward", float("inf"), "reward"),
            ("done", 2, "done"),
            ("priority", 0.0, "priority"),
            ("bootstrap_steps", 0, "bootstrap_steps"),
            ("next_action_mask", 64, "next_action_mask"),
        ],
    )
    def test_load_rejects_invalid_stored_scalars(self, temp_db, column, value, match):
        """Reader-side validation catches rows created before write checks existed."""
        handler = MemoryDBHandler(temp_db)
        try:
            handler.save_memories(snake_id=0, memories=[make_memory()])
            handler.cursor.execute(f"UPDATE memories_standard SET {column} = ?", (value,))
            handler.conn.commit()

            with pytest.raises(ValueError, match=f"Stored replay row.*{match}"):
                handler.load_memories_for_policy("apex", limit=None, order_by="id")
        finally:
            handler.close()

    def test_memories_to_dicts_preserves_next_action_mask(self):
        """Policy replay export should keep exact masks for SQLite persistence."""
        mask = torch.tensor([False, True, False, False, False, False])
        raw_tuple = (
            torch.zeros(STATE_SIZE),
            1,
            1.0,
            torch.ones(STATE_SIZE),
            False,
            1.0,
            2,
            mask,
        )

        memory = memories_to_dicts([raw_tuple])[0]

        assert memory["next_action_mask"].tolist() == mask.tolist()

    def test_memories_to_dicts_preserves_stream_snake_id(self):
        """Shared-policy replay export should keep the producing snake id."""
        raw_tuple = (
            torch.zeros(STATE_SIZE),
            1,
            1.0,
            torch.ones(STATE_SIZE),
            False,
            1.0,
            2,
            None,
            7,
        )

        memory = memories_to_dicts([raw_tuple])[0]

        assert memory["stream_id"] == 7
        assert memory["snake_id"] == 7

    def test_load_memories_can_include_snake_ids(self, temp_db):
        """Parallel merge can preserve per-row snake ownership from worker DBs."""
        handler = MemoryDBHandler(temp_db)
        try:
            handler.save_memories(snake_id=2, memories=[make_memory(action=0)])
            handler.save_memories(snake_id=5, memories=[make_memory(action=1)])

            loaded_rows = handler.load_memories_for_policy(
                "apex",
                limit=None,
                order_by="id",
                include_action_masks=True,
                include_snake_ids=True,
            )

            assert len(loaded_rows) == 9
            assert loaded_rows[-1] == [2, 5]
        finally:
            handler.close()

    def test_invalid_memory_order_raises(self, temp_db):
        """Unknown order modes should fail loudly instead of changing sampling silently."""
        handler = MemoryDBHandler(temp_db)

        try:
            handler.load_memories_for_policy("apex", order_by="random-ish")
        except ValueError as exc:
            assert "order_by" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid order_by")
        finally:
            handler.close()
