#!/usr/bin/env python3
"""Inspect generated replay quality before offline or Ape-X training."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

# Add project root to path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import load_and_initialize_config  # noqa: E402
from src.core.game_config import GameConfig  # noqa: E402
from src.core.reward_contract import current_reward_contract  # noqa: E402
from src.data.memory_db_handler import (  # noqa: E402
    REPLAY_QUALITY_GATE_ORDER,
    REPLAY_QUALITY_GATE_PRESETS,
    MemoryDBHandler,
    format_replay_quality_stats,
    format_replay_quality_warnings,
    resolve_min_row_count,
    resolve_replay_quality_gate_values,
    validate_min_row_count,
    validate_replay_metadata_contract,
    validate_replay_quality_gates,
)

AUDIT_GATE_ORDER = REPLAY_QUALITY_GATE_ORDER

AUDIT_GATE_FLAGS = {
    "min_terminal_fraction": "--min-terminal-fraction",
    "min_immediate_terminal_fraction": "--min-immediate-terminal-fraction",
    "min_exact_mask_fraction": "--min-exact-mask-fraction",
    "min_boost_mask_fraction": "--min-boost-mask-fraction",
    "min_action_coverage_fraction": "--min-action-coverage-fraction",
    "min_positive_reward_fraction": "--min-positive-reward-fraction",
    "min_negative_reward_fraction": "--min-negative-reward-fraction",
    "min_multistep_fraction": "--min-multistep-fraction",
    "max_dominant_action_fraction": "--max-dominant-action-fraction",
    "max_invalid_current_action_fraction": "--max-invalid-current-action-fraction",
    "max_nonterminal_trapped_next_fraction": "--max-nonterminal-trapped-next-fraction",
    "max_exact_mask_state_mismatch_fraction": "--max-exact-mask-state-mismatch-fraction",
    "max_malformed_state_feature_fraction": "--max-malformed-state-feature-fraction",
}

AUDIT_GATE_PRESETS = REPLAY_QUALITY_GATE_PRESETS


def is_gate_active(name: str, value: float) -> bool:
    """Return whether a replay-quality gate is stricter than the disabled default."""
    if name.startswith("max_"):
        return value < 1.0
    return value > 0.0


def format_gate_args(gates: dict[str, float]) -> str:
    """Return CLI arguments that can be reused for generation or offline training."""
    args = []
    for name in AUDIT_GATE_ORDER:
        value = float(gates[name])
        if is_gate_active(name, value):
            args.extend([AUDIT_GATE_FLAGS[name], f"{value:g}"])
    return " ".join(args)


def format_reusable_gate_args(preset: str, gates: dict[str, float]) -> str:
    """Return gate args that preserve the selected preset across CLIs."""
    if preset not in AUDIT_GATE_PRESETS:
        raise ValueError(f"unknown replay quality preset: {preset}")

    baseline = AUDIT_GATE_PRESETS[preset]
    args = []
    args.extend(["--replay-quality-preset", preset])

    for name in AUDIT_GATE_ORDER:
        value = float(gates[name])
        if preset == "none":
            if is_gate_active(name, value):
                args.extend([AUDIT_GATE_FLAGS[name], f"{value:g}"])
            continue

        if value != float(baseline[name]):
            args.extend([AUDIT_GATE_FLAGS[name], f"{value:g}"])

    return " ".join(args)


def format_offline_train_command(
    db_path: str,
    gate_args: str = "",
    min_row_count: int = 0,
    config_path: str | None = None,
) -> str:
    """Return a pasteable offline training command for an audited replay database."""
    command = [
        "python",
        "src/scripts/offline_train.py",
        "--db",
        shlex.quote(db_path),
    ]
    if config_path:
        command.extend(["--config", shlex.quote(config_path)])
    min_row_count = resolve_min_row_count(min_row_count)
    if min_row_count > 0:
        command.extend(["--min-row-count", str(min_row_count)])
    if gate_args:
        command.append(gate_args)
    return " ".join(command)


def resolve_audit_gate_values(args: argparse.Namespace) -> dict[str, float]:
    """Resolve explicit replay-quality gate overrides on top of a preset."""
    return resolve_replay_quality_gate_values(
        preset=args.preset,
        overrides={name: getattr(args, name) for name in AUDIT_GATE_ORDER},
    )


def load_replay_quality(db_path: str, policy_type: str, snake_id: int | None = None) -> dict:
    """Load replay quality diagnostics from a SQLite replay database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Replay database not found: {db_path}")

    db_handler = MemoryDBHandler(db_name=db_path)
    try:
        return db_handler.get_replay_quality_stats(policy_type=policy_type, snake_id=snake_id)
    finally:
        db_handler.close()


def load_replay_metadata(db_path: str) -> dict:
    """Load durable replay metadata from a SQLite replay database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Replay database not found: {db_path}")

    db_handler = MemoryDBHandler(db_name=db_path)
    try:
        return db_handler.get_metadata()
    finally:
        db_handler.close()


def format_generation_metadata(metadata: dict) -> list[str]:
    """Return compact generation metadata lines for replay audit reports."""
    if not metadata or not any(key.startswith("generation.") for key in metadata):
        return []

    lines = [
        (
            "Generation: "
            f"mode={metadata.get('generation.mode')} | "
            f"episodes={metadata.get('generation.episodes')} | "
            f"frame_limit={metadata.get('generation.frame_limit')} | "
            f"snakes={metadata.get('generation.num_snakes')} | "
            f"board={metadata.get('generation.board_width')}x"
            f"{metadata.get('generation.board_height')} | "
            f"state={metadata.get('generation.state_size')} | "
            f"actions={metadata.get('generation.action_size')} | "
            f"gamma={metadata.get('generation.gamma')} | "
            f"n_step={metadata.get('generation.apex_n_step')}"
        ),
        (
            "Generation exploration: "
            f"epsilon={metadata.get('generation.epsilon_min')}->"
            f"{metadata.get('generation.epsilon_max')} | "
            f"boost={metadata.get('generation.boost_exploration_rate')} | "
            f"danger={metadata.get('generation.danger_exploration_rate')} | "
            f"model_loaded={metadata.get('generation.model_loaded')}"
        ),
    ]
    replay_quality = metadata.get("generation.replay_quality")
    if isinstance(replay_quality, dict):
        count = int(replay_quality.get("count", 0))
        action_counts = replay_quality.get("action_counts", [])
        if isinstance(action_counts, dict):
            action_counts = [
                int(action_counts.get(str(action), action_counts.get(action, 0)))
                for action in range(6)
            ]
        actions = ", ".join(
            f"{action}:{int(action_counts[action]) if action < len(action_counts) else 0}"
            for action in range(6)
        )
        lines.append(
            "Generation replay quality: "
            f"rows={count:,} | "
            f"terminal={float(replay_quality.get('terminal_fraction', 0.0)):.2%} | "
            f"exact_masks={float(replay_quality.get('nonterminal_mask_fraction', 0.0)):.1%} | "
            f"actions={actions} | "
            f"reward neg/zero/pos="
            f"{int(replay_quality.get('reward_negative_count', 0)):,}/"
            f"{int(replay_quality.get('reward_zero_count', 0)):,}/"
            f"{int(replay_quality.get('reward_positive_count', 0)):,} | "
            f"invalid_nonterminal_actions="
            f"{float(replay_quality.get('nonterminal_invalid_current_action_fraction', 0.0)):.1%}"
        )
    return lines


def format_audit_report(
    db_path: str,
    policy_type: str,
    snake_id: int | None,
    quality: dict,
    warnings: list[str],
    gates: dict[str, float],
    metadata: dict | None = None,
    print_gate_args: bool = False,
    preset: str = "none",
    min_row_count: int = 0,
    config_path: str | None = None,
) -> list[str]:
    """Return the human-readable replay audit report."""
    snake_label = "all" if snake_id is None else str(snake_id)
    lines = [
        f"Replay audit: db={db_path} | policy={policy_type} | snake_id={snake_label}",
        "Replay quality:",
    ]
    metadata_lines = format_generation_metadata(metadata or {})
    if metadata_lines:
        lines.append("Replay metadata:")
        lines.extend(f"   {line}" for line in metadata_lines)
    lines.extend(format_replay_quality_stats(quality))

    if warnings:
        lines.append("Replay quality warnings:")
        lines.extend(warnings)
    else:
        lines.append("Replay quality warnings: none")

    min_row_count = resolve_min_row_count(min_row_count)
    if min_row_count > 0:
        lines.append(f"Row count gate: >= {min_row_count:,}")

    active_gate_args = format_gate_args(gates)
    if print_gate_args:
        if active_gate_args:
            lines.append("Active gate args:")
            lines.append(f"   {active_gate_args}")
        else:
            lines.append("Active gate args: none")

        reusable_gate_args = format_reusable_gate_args(preset, gates)
        if reusable_gate_args:
            lines.append("Reusable preset gate args:")
            lines.append(f"   {reusable_gate_args}")
        lines.append("Offline train command:")
        offline_command = format_offline_train_command(
            db_path,
            reusable_gate_args,
            min_row_count,
            config_path,
        )
        lines.append(f"   {offline_command}")
    return lines


def audit_replay_database(
    db_path: str,
    policy_type: str = "apex",
    snake_id: int | None = None,
    gates: dict[str, float] | None = None,
    min_row_count: int = 0,
    expected_gamma: float | None = None,
    expected_n_step: int | None = None,
    fail_on_warnings: bool = False,
) -> tuple[dict, list[str]]:
    """Load replay diagnostics, apply optional gates, and return stats plus warnings."""
    metadata = load_replay_metadata(db_path)
    validate_replay_metadata_contract(
        metadata,
        db_path,
        policy_type=policy_type,
        expected_state_size=GameConfig.INPUT_SIZE,
        expected_action_size=GameConfig.OUTPUT_SIZE,
        expected_gamma=GameConfig.APEX_GAMMA if expected_gamma is None else expected_gamma,
        expected_n_step=GameConfig.APEX_N_STEP if expected_n_step is None else expected_n_step,
        expected_reward_contract=current_reward_contract(),
        expected_reward_death=GameConfig.REWARD_DEATH,
        expected_reward_food_base=GameConfig.REWARD_FOOD_BASE,
        state_size_name="INPUT_SIZE",
        action_size_name="OUTPUT_SIZE",
        gamma_name="expected_gamma",
        n_step_name="expected_n_step",
        reward_death_name="REWARD_DEATH",
        reward_food_base_name="REWARD_FOOD_BASE",
    )
    quality = load_replay_quality(db_path, policy_type=policy_type, snake_id=snake_id)
    warnings = format_replay_quality_warnings(quality)
    active_gates = dict(AUDIT_GATE_PRESETS["none"] if gates is None else gates)
    validate_min_row_count(quality, min_row_count=min_row_count, context="Replay audit")
    validate_replay_quality_gates(quality, context="Replay audit", **active_gates)
    if fail_on_warnings and warnings:
        raise RuntimeError("Replay audit warnings present")
    return quality, warnings


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Audit generated Apex replay quality")
    parser.add_argument("--db", default="snake_memories.db", help="SQLite replay database path")
    parser.add_argument("--policy-type", default="apex", help="Policy type to inspect")
    parser.add_argument("--snake-id", type=int, default=None, help="Optional snake_id filter")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config used to validate replay contract metadata",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(AUDIT_GATE_PRESETS),
        default="none",
        help="Optional gate preset to apply before training",
    )
    parser.add_argument(
        "--print-gate-args",
        action="store_true",
        help="Print active gate args for generate_experiences.py/offline_train.py",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit with an error when replay-quality warnings are present",
    )
    parser.add_argument(
        "--min-row-count",
        type=int,
        default=0,
        help="Optional absolute replay row-count gate before offline training",
    )
    parser.add_argument(
        "--expected-gamma",
        type=float,
        default=None,
        help="Optional expected discount factor for generated n-step replay metadata",
    )
    parser.add_argument(
        "--expected-n-step",
        type=int,
        default=None,
        help="Optional expected n-step horizon for generated replay metadata",
    )
    parser.add_argument("--min-terminal-fraction", type=float, default=None)
    parser.add_argument("--min-immediate-terminal-fraction", type=float, default=None)
    parser.add_argument("--min-exact-mask-fraction", type=float, default=None)
    parser.add_argument("--min-boost-mask-fraction", type=float, default=None)
    parser.add_argument("--min-action-coverage-fraction", type=float, default=None)
    parser.add_argument("--min-positive-reward-fraction", type=float, default=None)
    parser.add_argument("--min-negative-reward-fraction", type=float, default=None)
    parser.add_argument("--min-multistep-fraction", type=float, default=None)
    parser.add_argument("--max-dominant-action-fraction", type=float, default=None)
    parser.add_argument("--max-invalid-current-action-fraction", type=float, default=None)
    parser.add_argument("--max-nonterminal-trapped-next-fraction", type=float, default=None)
    parser.add_argument("--max-exact-mask-state-mismatch-fraction", type=float, default=None)
    parser.add_argument("--max-malformed-state-feature-fraction", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the replay audit CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        load_and_initialize_config(args.config)
    gates = resolve_audit_gate_values(args)
    quality, warnings = audit_replay_database(
        db_path=args.db,
        policy_type=args.policy_type,
        snake_id=args.snake_id,
        gates=gates,
        min_row_count=args.min_row_count,
        expected_gamma=args.expected_gamma,
        expected_n_step=args.expected_n_step,
        fail_on_warnings=args.fail_on_warnings,
    )
    metadata = load_replay_metadata(args.db)
    for line in format_audit_report(
        db_path=args.db,
        policy_type=args.policy_type,
        snake_id=args.snake_id,
        quality=quality,
        warnings=warnings,
        gates=gates,
        metadata=metadata,
        print_gate_args=args.print_gate_args,
        preset=args.preset,
        min_row_count=args.min_row_count,
        config_path=args.config,
    ):
        print(line)


if __name__ == "__main__":
    main()
