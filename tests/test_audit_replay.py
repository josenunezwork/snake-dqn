"""Tests for the Torch-free replay audit CLI."""

import pytest

from src.core.game_config import GameConfig
from src.core.reward_contract import current_reward_contract
from src.data.memory_db_handler import (
    MemoryDBHandler,
    resolve_min_row_count,
    validate_min_row_count,
)
from src.scripts.audit_replay import (
    AUDIT_GATE_PRESETS,
    audit_replay_database,
    build_parser,
    format_gate_args,
    format_generation_metadata,
    format_offline_train_command,
    format_reusable_gate_args,
    main,
    resolve_audit_gate_values,
)


def make_state(value: float) -> list[float]:
    """Return a valid 58-feature replay state."""
    state = [0.0] * 58
    state[1] = 1.0
    state[4] = min(max(float(value) / 100.0, 0.0), 1.0)
    state[5] = 0.25
    state[6] = -0.25
    state[7] = 0.25
    state[44] = 0.1
    state[45] = -0.1
    state[46] = 0.1
    state[47] = 1.0
    state[49] = -0.5
    state[50] = -0.1
    state[51] = 0.1
    state[52] = 0.1
    state[57] = 1.0
    return state


def write_replay_db(db_path, rewards=None) -> None:
    """Create a small replay database with deterministic quality stats."""
    rewards = rewards if rewards is not None else [1.0, 0.0, 0.5, -1.0]
    handler = MemoryDBHandler(str(db_path))
    try:
        handler.save_memories(
            snake_id=2,
            memories=[
                {
                    "state": make_state(idx),
                    "action": idx % 4,
                    "reward": reward,
                    "next_state": make_state(idx + 10),
                    "done": idx == len(rewards) - 1,
                    "priority": 1.0 + idx,
                    "bootstrap_steps": 2 if idx < 2 else 1,
                    "next_action_mask": [mask_idx < 3 for mask_idx in range(6)],
                }
                for idx, reward in enumerate(rewards)
            ],
        )
    finally:
        handler.close()


def current_replay_contract_metadata(**overrides) -> dict:
    """Return generated-replay metadata matching the current training contract."""
    metadata = {
        "generation.policy_type": "apex",
        "generation.state_size": GameConfig.INPUT_SIZE,
        "generation.action_size": GameConfig.OUTPUT_SIZE,
        "generation.gamma": GameConfig.APEX_GAMMA,
        "generation.apex_n_step": GameConfig.APEX_N_STEP,
        "generation.reward_contract": current_reward_contract(),
        "generation.reward_death": GameConfig.REWARD_DEATH,
        "generation.reward_food_base": GameConfig.REWARD_FOOD_BASE,
    }
    metadata.update(overrides)
    return metadata


def test_format_generation_metadata_returns_compact_lines():
    lines = format_generation_metadata(
        {
            "generation.mode": "single",
            "generation.episodes": 10,
            "generation.frame_limit": 500,
            "generation.num_snakes": 4,
            "generation.board_width": 725,
            "generation.board_height": 415,
            "generation.state_size": 58,
            "generation.action_size": 6,
            "generation.gamma": 0.99,
            "generation.apex_n_step": 3,
            "generation.epsilon_min": 0.02,
            "generation.epsilon_max": 0.4,
            "generation.boost_exploration_rate": 0.25,
            "generation.danger_exploration_rate": 0.02,
            "generation.model_loaded": True,
            "generation.replay_quality": {
                "action_counts": [3, 0, 0, 1, 0, 0],
                "count": 4,
                "invalid_current_action_fraction": 0.25,
                "nonterminal_invalid_current_action_fraction": 0.25,
                "nonterminal_mask_fraction": 0.75,
                "reward_negative_count": 1,
                "reward_positive_count": 2,
                "reward_zero_count": 1,
                "terminal_fraction": 0.25,
            },
        }
    )

    assert lines == [
        (
            "Generation: mode=single | episodes=10 | frame_limit=500 | snakes=4 | "
            "board=725x415 | state=58 | actions=6 | gamma=0.99 | n_step=3"
        ),
        (
            "Generation exploration: epsilon=0.02->0.4 | boost=0.25 | danger=0.02 | "
            "model_loaded=True"
        ),
        (
            "Generation replay quality: rows=4 | terminal=25.00% | exact_masks=75.0% | "
            "actions=0:3, 1:0, 2:0, 3:1, 4:0, 5:0 | reward neg/zero/pos=1/1/2 | "
            "invalid_nonterminal_actions=25.0%"
        ),
    ]


def test_format_gate_args_prints_only_active_gates():
    gates = dict(AUDIT_GATE_PRESETS["none"])
    gates["min_terminal_fraction"] = 0.005
    gates["min_immediate_terminal_fraction"] = 0.001
    gates["min_boost_mask_fraction"] = 0.05
    gates["max_dominant_action_fraction"] = 0.75
    gates["max_invalid_current_action_fraction"] = 0.2
    gates["max_exact_mask_state_mismatch_fraction"] = 0.1
    gates["max_malformed_state_feature_fraction"] = 0.0

    assert (
        format_gate_args(gates) == "--min-terminal-fraction 0.005 "
        "--min-immediate-terminal-fraction 0.001 --min-boost-mask-fraction 0.05 "
        "--max-dominant-action-fraction 0.75 "
        "--max-invalid-current-action-fraction 0.2 "
        "--max-exact-mask-state-mismatch-fraction 0.1 "
        "--max-malformed-state-feature-fraction 0"
    )


def test_format_reusable_gate_args_prints_preset_with_only_overrides():
    gates = dict(AUDIT_GATE_PRESETS["training"])
    gates["min_boost_mask_fraction"] = 0.02
    gates["min_action_coverage_fraction"] = 0.5
    gates["max_dominant_action_fraction"] = 1.0
    gates["max_invalid_current_action_fraction"] = 0.2

    assert (
        format_reusable_gate_args("training", gates)
        == "--replay-quality-preset training --min-boost-mask-fraction 0.02 "
        "--min-action-coverage-fraction 0.5 "
        "--max-dominant-action-fraction 1 --max-invalid-current-action-fraction 0.2"
    )


def test_format_reusable_gate_args_preserves_warning_only_preset():
    gates = dict(AUDIT_GATE_PRESETS["none"])
    gates["min_terminal_fraction"] = 0.01

    assert (
        format_reusable_gate_args("none", gates)
        == "--replay-quality-preset none --min-terminal-fraction 0.01"
    )


def test_format_offline_train_command_quotes_db_path_with_spaces():
    command = format_offline_train_command(
        "/tmp/replay db.sqlite",
        "--replay-quality-preset training",
        min_row_count=100,
    )

    assert (
        command == "python src/scripts/offline_train.py --db '/tmp/replay db.sqlite' "
        "--min-row-count 100 "
        "--replay-quality-preset training"
    )


def test_format_offline_train_command_preserves_config_path():
    command = format_offline_train_command(
        "/tmp/replay.db",
        "--replay-quality-preset training",
        min_row_count=100,
        config_path="configs/training_fast.yaml",
    )

    assert (
        command == "python src/scripts/offline_train.py --db /tmp/replay.db "
        "--config configs/training_fast.yaml --min-row-count 100 "
        "--replay-quality-preset training"
    )


def test_resolve_min_row_count_rejects_negative_values():
    assert resolve_min_row_count(None) == 0
    assert resolve_min_row_count(0) == 0
    assert resolve_min_row_count(128) == 128
    with pytest.raises(ValueError, match="min-row-count"):
        resolve_min_row_count(-1)


def test_validate_min_row_count_raises_when_replay_is_too_small():
    with pytest.raises(RuntimeError, match="has 4 rows.*at least 8"):
        validate_min_row_count({"count": 4}, min_row_count=8, context="Replay audit")


def test_resolve_audit_gate_values_applies_preset_and_explicit_overrides():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--preset",
            "training",
            "--min-positive-reward-fraction",
            "0.2",
            "--min-negative-reward-fraction",
            "0.3",
            "--max-dominant-action-fraction",
            "0.9",
            "--max-malformed-state-feature-fraction",
            "0.05",
        ]
    )

    gates = resolve_audit_gate_values(args)

    assert gates["min_terminal_fraction"] == pytest.approx(0.005)
    assert gates["min_immediate_terminal_fraction"] == pytest.approx(0.001)
    assert gates["min_positive_reward_fraction"] == pytest.approx(0.2)
    assert gates["min_negative_reward_fraction"] == pytest.approx(0.3)
    assert gates["max_dominant_action_fraction"] == pytest.approx(0.9)
    assert gates["max_malformed_state_feature_fraction"] == pytest.approx(0.05)


def test_audit_replay_database_returns_quality_and_warnings(tmp_path):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path)

    quality, warnings = audit_replay_database(str(db_path))

    assert quality["count"] == 4
    assert quality["reward_positive_count"] == 2
    assert quality["multistep_count"] == 2
    assert isinstance(warnings, list)


def test_audit_replay_database_applies_quality_gates(tmp_path):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path, rewards=[0.0, 0.0, 1.0, -1.0])
    gates = dict(AUDIT_GATE_PRESETS["none"])
    gates["min_positive_reward_fraction"] = 0.5

    with pytest.raises(RuntimeError, match="positive-reward fraction"):
        audit_replay_database(str(db_path), gates=gates)


def test_audit_replay_database_rejects_incompatible_generation_metadata(tmp_path):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path)
    handler = MemoryDBHandler(str(db_path))
    try:
        handler.update_metadata(
            {
                "generation.policy_type": "apex",
                "generation.state_size": 57,
                "generation.action_size": 6,
            }
        )
    finally:
        handler.close()

    with pytest.raises(RuntimeError, match="generation.state_size"):
        audit_replay_database(str(db_path))


def test_audit_replay_database_rejects_expected_gamma_mismatch(tmp_path):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path)
    handler = MemoryDBHandler(str(db_path))
    try:
        handler.update_metadata(
            {
                "generation.policy_type": "apex",
                "generation.state_size": 58,
                "generation.action_size": 6,
                "generation.gamma": 0.95,
            }
        )
    finally:
        handler.close()

    with pytest.raises(RuntimeError, match="generation.gamma"):
        audit_replay_database(str(db_path), expected_gamma=0.99)


def test_audit_replay_database_rejects_gamma_mismatch_by_default(tmp_path):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path)
    handler = MemoryDBHandler(str(db_path))
    try:
        handler.update_metadata(current_replay_contract_metadata(**{"generation.gamma": 0.95}))
    finally:
        handler.close()

    with pytest.raises(RuntimeError, match="generation.gamma"):
        audit_replay_database(str(db_path))


def test_audit_replay_database_rejects_expected_n_step_mismatch(tmp_path):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path)
    handler = MemoryDBHandler(str(db_path))
    try:
        handler.update_metadata(
            {
                "generation.policy_type": "apex",
                "generation.state_size": 58,
                "generation.action_size": 6,
                "generation.apex_n_step": 1,
            }
        )
    finally:
        handler.close()

    with pytest.raises(RuntimeError, match="generation.apex_n_step"):
        audit_replay_database(str(db_path), expected_n_step=3)


def test_audit_replay_database_rejects_full_reward_contract_mismatch_by_default(tmp_path):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path)
    stale_contract = current_reward_contract()
    stale_contract["survival"] = float(stale_contract["survival"]) + 1.0
    handler = MemoryDBHandler(str(db_path))
    try:
        handler.update_metadata(
            current_replay_contract_metadata(**{"generation.reward_contract": stale_contract})
        )
    finally:
        handler.close()

    with pytest.raises(RuntimeError, match="generation.reward_contract.survival"):
        audit_replay_database(str(db_path))


def test_audit_replay_database_applies_negative_reward_gate(tmp_path):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path, rewards=[0.0, 0.0, 0.0, -1.0])
    gates = dict(AUDIT_GATE_PRESETS["none"])
    gates["min_negative_reward_fraction"] = 0.5

    with pytest.raises(RuntimeError, match="negative-reward fraction"):
        audit_replay_database(str(db_path), gates=gates)


def test_audit_replay_database_applies_min_row_count(tmp_path):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path)

    with pytest.raises(RuntimeError, match="has 4 rows.*at least 5"):
        audit_replay_database(str(db_path), min_row_count=5)


def test_main_prints_report_and_gate_args(tmp_path, capsys):
    db_path = tmp_path / "replay.db"
    write_replay_db(db_path)
    handler = MemoryDBHandler(str(db_path))
    try:
        handler.update_metadata(
            {
                **current_replay_contract_metadata(),
                "generation.mode": "single",
                "generation.episodes": 4,
                "generation.frame_limit": 100,
                "generation.num_snakes": 2,
                "generation.board_width": 400,
                "generation.board_height": 300,
                "generation.epsilon_min": 0.1,
                "generation.epsilon_max": 0.4,
                "generation.boost_exploration_rate": 0.25,
                "generation.danger_exploration_rate": 0.02,
                "generation.model_loaded": False,
            }
        )
    finally:
        handler.close()

    main(
        [
            "--db",
            str(db_path),
            "--preset",
            "training",
            "--print-gate-args",
            "--min-boost-mask-fraction",
            "0.0",
            "--min-action-coverage-fraction",
            "0.0",
            "--max-dominant-action-fraction",
            "1.0",
            "--min-row-count",
            "4",
        ]
    )
    output = capsys.readouterr().out

    assert f"Replay audit: db={db_path} | policy=apex | snake_id=all" in output
    assert "Replay quality:" in output
    assert "Replay metadata:" in output
    assert "Generation: mode=single | episodes=4 | frame_limit=100" in output
    assert "Rows: 4" in output
    assert "Active gate args:" in output
    assert "Row count gate: >= 4" in output
    assert "Reusable preset gate args:" in output
    assert "Offline train command:" in output
    assert f"python src/scripts/offline_train.py --db {db_path}" in output
    assert "--min-row-count 4" in output
    assert "--replay-quality-preset training" in output
    assert "--min-boost-mask-fraction 0" in output
    assert "--min-action-coverage-fraction 0" in output
    assert "--max-dominant-action-fraction 1" in output
    assert "--max-exact-mask-state-mismatch-fraction 0" in output
    assert "--max-malformed-state-feature-fraction 0" in output
    assert "--min-positive-reward-fraction 0.005" in output
    assert "--min-negative-reward-fraction 0.005" in output


def test_main_rejects_missing_database(tmp_path):
    missing_db = tmp_path / "missing.db"

    with pytest.raises(FileNotFoundError, match="Replay database not found"):
        main(["--db", str(missing_db)])
