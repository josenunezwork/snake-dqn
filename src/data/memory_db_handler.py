"""
Memory Database Handler for Apex-DQN policy.

Provides persistent storage for experience replay memories using SQLite.
Optimized for distributed Apex-DQN training with prioritized experience replay.
"""

import json
import math
import sqlite3
import struct
from collections.abc import Iterable, Mapping

ACTION_SIZE = 6
STATE_SIZE = 58
PER_ACTION_DANGER_START = 54
PER_ACTION_DANGER_END = 57
BOOST_AVAILABLE_INDEX = 57
BOOST_ACTION_MASK_BITS = sum(1 << action for action in range(3, ACTION_SIZE))
ACTION_DANGER_COLLISION_THRESHOLD = 1.0
STATE_RANGE_EPSILON = 1e-5
STATE_BLOB_FORMAT = f"<{STATE_SIZE}f"
STATE_BLOB_SIZE = struct.calcsize(STATE_BLOB_FORMAT)
ACTION_MASK_SIZE = ACTION_SIZE
STATE_FEATURE_BOUNDS = (
    *([(0.0, 1.0)] * 5),
    *([(-1.0, 1.0)] * 2),
    (0.0, 1.0),
    *([(0.0, 1.0)] * 36),
    *([(-1.0, 1.0)] * 2),
    (0.0, 1.0),
    *([(-1.0, 1.0)] * 5),
    (0.0, 1.0),
    *([(0.0, 1.0)] * 5),
)
assert len(STATE_FEATURE_BOUNDS) == STATE_SIZE
REPLAY_QUALITY_GATE_ORDER = (
    "min_terminal_fraction",
    "min_immediate_terminal_fraction",
    "min_exact_mask_fraction",
    "min_boost_mask_fraction",
    "min_action_coverage_fraction",
    "min_positive_reward_fraction",
    "min_negative_reward_fraction",
    "min_multistep_fraction",
    "max_dominant_action_fraction",
    "max_invalid_current_action_fraction",
    "max_nonterminal_trapped_next_fraction",
    "max_exact_mask_state_mismatch_fraction",
    "max_malformed_state_feature_fraction",
)
REPLAY_QUALITY_GATE_PRESETS = {
    "none": {
        "min_terminal_fraction": 0.0,
        "min_immediate_terminal_fraction": 0.0,
        "min_exact_mask_fraction": 0.0,
        "min_boost_mask_fraction": 0.0,
        "min_action_coverage_fraction": 0.0,
        "min_positive_reward_fraction": 0.0,
        "min_negative_reward_fraction": 0.0,
        "min_multistep_fraction": 0.0,
        "max_dominant_action_fraction": 1.0,
        "max_invalid_current_action_fraction": 1.0,
        "max_nonterminal_trapped_next_fraction": 1.0,
        "max_exact_mask_state_mismatch_fraction": 1.0,
        "max_malformed_state_feature_fraction": 1.0,
    },
    "training": {
        "min_terminal_fraction": 0.005,
        "min_immediate_terminal_fraction": 0.001,
        "min_exact_mask_fraction": 0.8,
        "min_boost_mask_fraction": 0.05,
        "min_action_coverage_fraction": 1.0,
        "min_positive_reward_fraction": 0.005,
        "min_negative_reward_fraction": 0.005,
        "min_multistep_fraction": 0.5,
        "max_dominant_action_fraction": 0.75,
        "max_invalid_current_action_fraction": 0.0,
        "max_nonterminal_trapped_next_fraction": 0.05,
        "max_exact_mask_state_mismatch_fraction": 0.0,
        "max_malformed_state_feature_fraction": 0.0,
    },
}


def resolve_replay_quality_fraction(value=None, field_name: str = "quality_fraction") -> float:
    """Resolve an optional replay-quality fraction gate in [0, 1]."""
    if value is None:
        return 0.0
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and in [0, 1]") from exc
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return fraction


def resolve_replay_quality_gate_values(
    preset: str = "none",
    overrides: dict[str, object] | None = None,
) -> dict[str, float]:
    """Resolve replay-quality gates from a named preset plus optional overrides."""
    if preset not in REPLAY_QUALITY_GATE_PRESETS:
        choices = ", ".join(sorted(REPLAY_QUALITY_GATE_PRESETS))
        raise ValueError(f"replay quality preset must be one of: {choices}")

    gates = dict(REPLAY_QUALITY_GATE_PRESETS[preset])
    for name, value in (overrides or {}).items():
        if name not in REPLAY_QUALITY_GATE_ORDER:
            raise ValueError(f"unknown replay quality gate: {name}")
        if value is not None:
            gates[name] = resolve_replay_quality_fraction(value, name)
    return gates


def resolve_min_row_count(value: int | None) -> int:
    """Resolve an optional absolute replay row-count gate."""
    if value is None:
        return 0
    row_count = int(value)
    if row_count < 0:
        raise ValueError("min-row-count must be non-negative")
    return row_count


def _validate_metadata_integer(
    metadata: dict,
    key: str,
    expected: int,
    expected_name: str,
    db_path: str,
) -> None:
    """Validate integer generation metadata without accepting lossy coercions."""
    if key not in metadata or metadata[key] is None:
        return

    value = metadata[key]
    if isinstance(value, (bool, str, bytes, bytearray, memoryview)):
        raise RuntimeError(f"Replay metadata {key} in {db_path} must be an integer")
    try:
        integer = int(value)
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Replay metadata {key} in {db_path} must be an integer") from exc
    if not math.isfinite(number) or number != integer:
        raise RuntimeError(f"Replay metadata {key} in {db_path} must be an integer")
    if integer != expected:
        raise RuntimeError(
            f"Replay metadata {key}={integer} does not match current "
            f"{expected_name}={expected} for {db_path}"
        )


def _validate_metadata_float(
    metadata: dict,
    key: str,
    expected: float,
    expected_name: str,
    db_path: str,
    required: bool = False,
) -> None:
    """Validate floating-point generation metadata against a consumer value."""
    if key not in metadata or metadata[key] is None:
        if required:
            raise RuntimeError(
                f"Replay metadata missing required {key} for {db_path}; "
                f"regenerate replay with current {expected_name}={float(expected):g}"
            )
        return

    value = metadata[key]
    if isinstance(value, (bool, str, bytes, bytearray, memoryview)):
        raise RuntimeError(f"Replay metadata {key} in {db_path} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Replay metadata {key} in {db_path} must be finite") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"Replay metadata {key} in {db_path} must be finite")
    if not math.isclose(number, float(expected), rel_tol=1e-7, abs_tol=1e-9):
        raise RuntimeError(
            f"Replay metadata {key}={number:g} does not match current "
            f"{expected_name}={float(expected):g} for {db_path}"
        )


def _validate_reward_contract(
    metadata: dict,
    expected_contract: Mapping[str, object],
    db_path: str,
) -> None:
    """Validate the full reward settings snapshot that produced replay rewards."""
    raw_contract = metadata.get("generation.reward_contract")
    if raw_contract is None:
        raise RuntimeError(
            f"Replay metadata missing required generation.reward_contract for {db_path}; "
            "regenerate replay with the current reward settings"
        )
    if not isinstance(raw_contract, Mapping):
        raise RuntimeError(
            f"Replay metadata generation.reward_contract in {db_path} must be an object"
        )

    for key, expected in expected_contract.items():
        metadata_key = f"generation.reward_contract.{key}"
        if key not in raw_contract or raw_contract[key] is None:
            raise RuntimeError(
                f"Replay metadata missing required {metadata_key} for {db_path}; "
                "regenerate replay with the current reward settings"
            )
        value = raw_contract[key]
        if isinstance(value, (bool, str, bytes, bytearray, memoryview)):
            raise RuntimeError(f"Replay metadata {metadata_key} in {db_path} must be finite")
        try:
            number = float(value)
            expected_number = float(expected)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Replay metadata {metadata_key} in {db_path} must be finite"
            ) from exc
        if not math.isfinite(number) or not math.isfinite(expected_number):
            raise RuntimeError(f"Replay metadata {metadata_key} in {db_path} must be finite")
        if not math.isclose(number, expected_number, rel_tol=1e-7, abs_tol=1e-9):
            raise RuntimeError(
                f"Replay metadata {metadata_key}={number:g} does not match current "
                f"RewardSettings.{key}={expected_number:g} for {db_path}"
            )


def validate_replay_metadata_contract(
    metadata: dict,
    db_path: str,
    policy_type: str = "apex",
    expected_state_size: int = STATE_SIZE,
    expected_action_size: int = ACTION_SIZE,
    expected_gamma: float | None = None,
    expected_n_step: int | None = None,
    expected_reward_contract: Mapping[str, object] | None = None,
    expected_reward_death: float | None = None,
    expected_reward_food_base: float | None = None,
    state_size_name: str = "STATE_SIZE",
    action_size_name: str = "ACTION_SIZE",
    gamma_name: str = "gamma",
    n_step_name: str = "n_step",
    reward_death_name: str = "reward_death",
    reward_food_base_name: str = "reward_food_base",
) -> None:
    """Validate durable replay-generation metadata against the current consumer."""
    if not metadata:
        return

    generated_policy_type = metadata.get("generation.policy_type")
    if generated_policy_type is not None and generated_policy_type != policy_type:
        raise RuntimeError(
            "Replay metadata generation.policy_type="
            f"{generated_policy_type!r} does not match policy_type={policy_type!r} for {db_path}"
        )

    _validate_metadata_integer(
        metadata,
        "generation.state_size",
        int(expected_state_size),
        state_size_name,
        db_path,
    )
    _validate_metadata_integer(
        metadata,
        "generation.action_size",
        int(expected_action_size),
        action_size_name,
        db_path,
    )
    if expected_gamma is not None:
        _validate_metadata_float(
            metadata,
            "generation.gamma",
            float(expected_gamma),
            gamma_name,
            db_path,
        )
    if expected_n_step is not None:
        _validate_metadata_integer(
            metadata,
            "generation.apex_n_step",
            int(expected_n_step),
            n_step_name,
            db_path,
        )
    if expected_reward_contract is not None:
        _validate_reward_contract(metadata, expected_reward_contract, db_path)
    if expected_reward_death is not None:
        _validate_metadata_float(
            metadata,
            "generation.reward_death",
            float(expected_reward_death),
            reward_death_name,
            db_path,
            required=True,
        )
    if expected_reward_food_base is not None:
        _validate_metadata_float(
            metadata,
            "generation.reward_food_base",
            float(expected_reward_food_base),
            reward_food_base_name,
            db_path,
            required=True,
        )


def validate_min_row_count(
    replay_quality: dict,
    min_row_count: int = 0,
    context: str = "Replay",
) -> None:
    """Fail fast when replay has fewer rows than an absolute row-count floor."""
    min_row_count = resolve_min_row_count(min_row_count)
    count = int(replay_quality.get("count", 0))
    if min_row_count > 0 and count < min_row_count:
        raise RuntimeError(f"{context} has {count:,} rows but needs at least {min_row_count:,}")


def validate_replay_quality_gates(
    replay_quality: dict,
    min_terminal_fraction: float = 0.0,
    min_immediate_terminal_fraction: float = 0.0,
    min_exact_mask_fraction: float = 0.0,
    min_boost_mask_fraction: float = 0.0,
    min_action_coverage_fraction: float = 0.0,
    min_positive_reward_fraction: float = 0.0,
    min_negative_reward_fraction: float = 0.0,
    min_multistep_fraction: float = 0.0,
    max_dominant_action_fraction: float = 1.0,
    max_invalid_current_action_fraction: float = 1.0,
    max_nonterminal_trapped_next_fraction: float = 1.0,
    max_exact_mask_state_mismatch_fraction: float = 1.0,
    max_malformed_state_feature_fraction: float = 1.0,
    context: str = "Replay",
) -> None:
    """Fail fast when replay misses requested learning-signal floors."""
    count = int(replay_quality.get("count", 0))
    done_count = int(replay_quality.get("done_count", 0))
    terminal_fraction = float(replay_quality.get("terminal_fraction", 0.0))
    nonterminal_count = int(replay_quality.get("nonterminal_count", max(count - done_count, 0)))
    nonterminal_mask_count = int(replay_quality.get("nonterminal_mask_count", 0))
    exact_mask_fraction = float(replay_quality.get("nonterminal_mask_fraction", 0.0))
    boost_mask_count = int(replay_quality.get("boost_mask_count", 0))
    boost_mask_fraction = float(replay_quality.get("boost_mask_fraction", 0.0))
    action_counts = replay_quality.get("action_counts", {})
    action_diversity = _action_diversity_stats(action_counts, count)
    active_action_count = int(
        replay_quality.get("active_action_count", action_diversity["active_action_count"])
    )
    action_coverage_fraction = active_action_count / ACTION_SIZE if ACTION_SIZE else 0.0
    dominant_action = replay_quality.get("dominant_action", action_diversity["dominant_action"])
    dominant_action_fraction = float(
        replay_quality.get(
            "dominant_action_fraction",
            action_diversity["dominant_action_fraction"],
        )
    )
    trapped_next_count = int(replay_quality.get("nonterminal_trapped_next_state_count", 0))
    trapped_next_fraction = float(
        replay_quality.get("nonterminal_trapped_next_state_fraction", 0.0)
    )
    exact_mask_state_comparison_count = int(
        replay_quality.get("exact_mask_state_comparison_count", 0)
    )
    exact_mask_state_mismatch_count = int(replay_quality.get("exact_mask_state_mismatch_count", 0))
    exact_mask_state_mismatch_fraction = float(
        replay_quality.get("exact_mask_state_mismatch_fraction", 0.0)
    )
    malformed_state_feature_count = int(replay_quality.get("malformed_state_feature_count", 0))
    malformed_next_state_feature_count = int(
        replay_quality.get("malformed_next_state_feature_count", 0)
    )
    invalid_state_feature_count = int(replay_quality.get("invalid_state_feature_count", 0))
    invalid_next_state_feature_count = int(
        replay_quality.get("invalid_next_state_feature_count", 0)
    )
    total_malformed_state_feature_count = (
        malformed_state_feature_count
        + malformed_next_state_feature_count
        + invalid_state_feature_count
        + invalid_next_state_feature_count
    )
    valid_state_feature_count = int(replay_quality.get("valid_state_feature_count", 0))
    valid_next_state_feature_count = int(replay_quality.get("valid_next_state_feature_count", 0))
    total_valid_state_feature_count = (
        valid_state_feature_count
        + valid_next_state_feature_count
        + invalid_state_feature_count
        + invalid_next_state_feature_count
    )
    malformed_state_feature_fraction = float(
        replay_quality.get(
            "malformed_state_feature_fraction",
            (
                total_malformed_state_feature_count / total_valid_state_feature_count
                if total_valid_state_feature_count
                else 0.0
            ),
        )
    )
    positive_reward_count = int(replay_quality.get("reward_positive_count", 0))
    positive_reward_fraction = positive_reward_count / count if count > 0 else 0.0
    negative_reward_count = int(replay_quality.get("reward_negative_count", 0))
    negative_reward_fraction = negative_reward_count / count if count > 0 else 0.0
    terminal_nonnegative_reward_count = int(
        replay_quality.get("terminal_nonnegative_reward_count", 0)
    )
    terminal_immediate_count = int(replay_quality.get("terminal_immediate_count", done_count))
    immediate_terminal_fraction = float(
        replay_quality.get(
            "immediate_terminal_fraction",
            terminal_immediate_count / count if count else 0.0,
        )
    )
    terminal_immediate_nonnegative_reward_count = int(
        replay_quality.get(
            "terminal_immediate_nonnegative_reward_count",
            terminal_nonnegative_reward_count,
        )
    )
    terminal_immediate_nonnegative_reward_fraction = float(
        replay_quality.get(
            "terminal_immediate_nonnegative_reward_fraction",
            (
                terminal_immediate_nonnegative_reward_count / terminal_immediate_count
                if terminal_immediate_count
                else 0.0
            ),
        )
    )
    terminal_multistep_count = int(
        replay_quality.get(
            "terminal_multistep_count",
            max(done_count - terminal_immediate_count, 0),
        )
    )
    terminal_multistep_nonnegative_reward_count = int(
        replay_quality.get(
            "terminal_multistep_nonnegative_reward_count",
            max(
                terminal_nonnegative_reward_count - terminal_immediate_nonnegative_reward_count,
                0,
            ),
        )
    )
    terminal_multistep_nonnegative_reward_fraction = float(
        replay_quality.get(
            "terminal_multistep_nonnegative_reward_fraction",
            (
                terminal_multistep_nonnegative_reward_count / terminal_multistep_count
                if terminal_multistep_count
                else 0.0
            ),
        )
    )
    multistep_count = int(replay_quality.get("multistep_count", 0))
    multistep_fraction = float(replay_quality.get("multistep_fraction", 0.0))
    current_action_comparison_count = int(
        replay_quality.get("current_action_state_comparison_count", 0)
    )
    invalid_current_action_count = int(replay_quality.get("invalid_current_action_count", 0))
    invalid_current_action_fraction = float(
        replay_quality.get("invalid_current_action_fraction", 0.0)
    )
    nonterminal_current_action_comparison_count = int(
        replay_quality.get(
            "nonterminal_current_action_state_comparison_count",
            current_action_comparison_count,
        )
    )
    nonterminal_invalid_current_action_count = int(
        replay_quality.get(
            "nonterminal_invalid_current_action_count",
            invalid_current_action_count,
        )
    )
    nonterminal_invalid_current_action_fraction = float(
        replay_quality.get(
            "nonterminal_invalid_current_action_fraction",
            invalid_current_action_fraction,
        )
    )
    invalid_scalar_row_count = int(replay_quality.get("invalid_scalar_row_count", 0))
    invalid_scalar_field_count = int(replay_quality.get("invalid_scalar_field_count", 0))
    invalid_action_id_count = int(replay_quality.get("invalid_action_id_count", 0))
    invalid_reward_count = int(replay_quality.get("invalid_reward_count", 0))
    invalid_priority_count = int(replay_quality.get("invalid_priority_count", 0))
    invalid_bootstrap_steps_count = int(replay_quality.get("invalid_bootstrap_steps_count", 0))
    invalid_done_count = int(replay_quality.get("invalid_done_count", 0))
    invalid_action_mask_count = int(replay_quality.get("invalid_action_mask_count", 0))

    if invalid_scalar_row_count > 0:
        raise RuntimeError(
            f"{context} has {invalid_scalar_row_count:,}/{count:,} rows with invalid scalar "
            f"replay fields ({invalid_scalar_field_count:,} total: "
            f"actions={invalid_action_id_count:,}, rewards={invalid_reward_count:,}, "
            f"priorities={invalid_priority_count:,}, "
            f"bootstrap_steps={invalid_bootstrap_steps_count:,}, dones={invalid_done_count:,})"
        )
    if invalid_action_mask_count > 0:
        raise RuntimeError(
            f"{context} has {invalid_action_mask_count:,}/{count:,} rows with invalid exact "
            "next-action masks"
        )
    if terminal_immediate_nonnegative_reward_count > 0:
        raise RuntimeError(
            f"{context} has {terminal_immediate_nonnegative_reward_count:,}/"
            f"{terminal_immediate_count:,} one-step terminal rows "
            f"({terminal_immediate_nonnegative_reward_fraction:.2%}) with non-negative rewards; "
            "immediate collision terminal rows must carry a death penalty"
        )
    if terminal_multistep_nonnegative_reward_count > 0:
        raise RuntimeError(
            f"{context} has {terminal_multistep_nonnegative_reward_count:,}/"
            f"{terminal_multistep_count:,} n-step terminal rows "
            f"({terminal_multistep_nonnegative_reward_fraction:.2%}) with non-negative returns; "
            "terminal n-step rows must stay negative so death remains a net penalty"
        )
    if min_terminal_fraction > 0.0 and (count <= 0 or terminal_fraction < min_terminal_fraction):
        raise RuntimeError(
            f"{context} terminal fraction "
            f"{terminal_fraction:.2%} ({done_count:,}/{count:,}) is below the requested "
            f"minimum {min_terminal_fraction:.2%}"
        )
    if min_immediate_terminal_fraction > 0.0 and (
        count <= 0 or immediate_terminal_fraction < min_immediate_terminal_fraction
    ):
        raise RuntimeError(
            f"{context} immediate-terminal fraction "
            f"{immediate_terminal_fraction:.2%} ({terminal_immediate_count:,}/{count:,} rows) "
            "is below the requested minimum "
            f"{min_immediate_terminal_fraction:.2%}; one-step collision terminal rows are "
            "required so the death penalty is not only blended into n-step returns"
        )
    if min_exact_mask_fraction > 0.0 and (
        nonterminal_count <= 0 or exact_mask_fraction < min_exact_mask_fraction
    ):
        raise RuntimeError(
            f"{context} exact-mask fraction "
            f"{exact_mask_fraction:.2%} ({nonterminal_mask_count:,}/{nonterminal_count:,} "
            "nonterminal rows) is below the requested minimum "
            f"{min_exact_mask_fraction:.2%}"
        )
    if min_boost_mask_fraction > 0.0 and (
        nonterminal_count <= 0 or boost_mask_fraction < min_boost_mask_fraction
    ):
        raise RuntimeError(
            f"{context} boost-mask fraction "
            f"{boost_mask_fraction:.2%} ({boost_mask_count:,}/{nonterminal_count:,} "
            "nonterminal rows allow at least one boost action) is below the requested "
            f"minimum {min_boost_mask_fraction:.2%}"
        )
    if min_action_coverage_fraction > 0.0 and (
        count <= 0 or action_coverage_fraction < min_action_coverage_fraction
    ):
        raise RuntimeError(
            f"{context} action coverage "
            f"{action_coverage_fraction:.2%} ({active_action_count}/{ACTION_SIZE} actions) "
            f"is below the requested minimum {min_action_coverage_fraction:.2%}"
        )
    if min_positive_reward_fraction > 0.0 and (
        count <= 0 or positive_reward_fraction < min_positive_reward_fraction
    ):
        raise RuntimeError(
            f"{context} positive-reward fraction "
            f"{positive_reward_fraction:.2%} ({positive_reward_count:,}/{count:,} rows) "
            f"is below the requested minimum {min_positive_reward_fraction:.2%}"
        )
    if min_negative_reward_fraction > 0.0 and (
        count <= 0 or negative_reward_fraction < min_negative_reward_fraction
    ):
        raise RuntimeError(
            f"{context} negative-reward fraction "
            f"{negative_reward_fraction:.2%} ({negative_reward_count:,}/{count:,} rows) "
            f"is below the requested minimum {min_negative_reward_fraction:.2%}"
        )
    if min_multistep_fraction > 0.0 and (count <= 0 or multistep_fraction < min_multistep_fraction):
        raise RuntimeError(
            f"{context} multi-step fraction "
            f"{multistep_fraction:.2%} ({multistep_count:,}/{count:,} rows) "
            f"is below the requested minimum {min_multistep_fraction:.2%}"
        )
    if max_dominant_action_fraction < 1.0 and (
        count <= 0 or dominant_action_fraction > max_dominant_action_fraction
    ):
        raise RuntimeError(
            f"{context} dominant-action fraction "
            f"{dominant_action_fraction:.2%} (action {dominant_action}) exceeds the requested "
            f"maximum {max_dominant_action_fraction:.2%}"
        )
    if max_invalid_current_action_fraction < 1.0:
        if count > 0 and current_action_comparison_count <= 0:
            raise RuntimeError(
                f"{context} current-action validity cannot be checked because replay "
                "does not include current-state observations"
            )
        if nonterminal_invalid_current_action_fraction > max_invalid_current_action_fraction:
            raise RuntimeError(
                f"{context} nonterminal invalid current-action fraction "
                f"{nonterminal_invalid_current_action_fraction:.2%} "
                f"({nonterminal_invalid_current_action_count:,}/"
                f"{nonterminal_current_action_comparison_count:,} "
                "nonterminal state-compared rows) exceeds the requested maximum "
                f"{max_invalid_current_action_fraction:.2%}"
            )
    if (
        max_nonterminal_trapped_next_fraction < 1.0
        and nonterminal_count > 0
        and trapped_next_fraction > max_nonterminal_trapped_next_fraction
    ):
        raise RuntimeError(
            f"{context} trapped-next-state fraction "
            f"{trapped_next_fraction:.2%} ({trapped_next_count:,}/{nonterminal_count:,} "
            "nonterminal targets) exceeds the requested maximum "
            f"{max_nonterminal_trapped_next_fraction:.2%}"
        )
    if (
        max_exact_mask_state_mismatch_fraction < 1.0
        and exact_mask_state_comparison_count > 0
        and exact_mask_state_mismatch_fraction > max_exact_mask_state_mismatch_fraction
    ):
        raise RuntimeError(
            f"{context} exact-mask/state mismatch fraction "
            f"{exact_mask_state_mismatch_fraction:.2%} "
            f"({exact_mask_state_mismatch_count:,}/{exact_mask_state_comparison_count:,} "
            "exact next-action masks) exceeds the requested maximum "
            f"{max_exact_mask_state_mismatch_fraction:.2%}"
        )
    if (
        max_malformed_state_feature_fraction < 1.0
        and total_valid_state_feature_count > 0
        and malformed_state_feature_fraction > max_malformed_state_feature_fraction
    ):
        raise RuntimeError(
            f"{context} malformed state-feature fraction "
            f"{malformed_state_feature_fraction:.2%} "
            f"({total_malformed_state_feature_count:,}/{total_valid_state_feature_count:,} "
            "current/next observations) exceeds the requested maximum "
            f"{max_malformed_state_feature_fraction:.2%}"
        )


def _flatten_state_values(value, field_name: str) -> list[float]:
    """Convert a tensor/array/list observation into a flat Python float list."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "flatten") and hasattr(value, "tolist"):
        value = value.flatten().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    values = []

    def visit(item) -> None:
        if isinstance(item, (str, bytes, bytearray, memoryview)):
            raise ValueError(f"{field_name} must be numeric, got {type(item).__name__}")
        if isinstance(item, Iterable):
            for child in item:
                visit(child)
            return
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must contain only finite values")
        values.append(number)

    visit(value)
    return values


def _coerce_finite_float(value, field_name: str) -> float:
    """Return a finite float or raise a replay-quality error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _coerce_action(value: object) -> int:
    """Validate that an action matches the six-output relative action space."""
    if isinstance(value, (bool, str, bytes, bytearray, memoryview)):
        raise ValueError(f"action must be an integer in [0, {ACTION_SIZE})")
    try:
        action = int(value)
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action must be an integer in [0, {ACTION_SIZE})") from exc
    if not math.isfinite(numeric_value) or numeric_value != action:
        raise ValueError(f"action must be an integer in [0, {ACTION_SIZE})")
    if action < 0 or action >= ACTION_SIZE:
        raise ValueError(f"action must be an integer in [0, {ACTION_SIZE})")
    return action


def _coerce_priority(value: object) -> float:
    """Validate replay priority before it reaches prioritized sampling."""
    priority = _coerce_finite_float(value, "priority")
    if priority <= 0.0:
        raise ValueError("priority must be positive")
    return priority


def _coerce_bootstrap_steps(value: object) -> int:
    """Validate n-step bootstrap metadata instead of silently rewriting it."""
    if isinstance(value, (bool, str, bytes, bytearray, memoryview)):
        raise ValueError("bootstrap_steps must be an integer >= 1")
    try:
        steps = int(value)
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bootstrap_steps must be an integer >= 1") from exc
    if not math.isfinite(numeric_value) or numeric_value != steps or steps < 1:
        raise ValueError("bootstrap_steps must be an integer >= 1")
    return steps


def _coerce_done(value: object) -> bool:
    """Validate a replay terminal flag without broad truthiness coercion."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ValueError("done must be bool/0/1")
    try:
        numeric_value = float(value)
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("done must be bool/0/1") from exc
    if not math.isfinite(numeric_value) or numeric_value != integer or integer not in (0, 1):
        raise ValueError("done must be bool/0/1")
    return bool(integer)


def _coerce_action_mask(value: object) -> int | None:
    """Encode an optional six-action bool mask into compact replay bits.

    A zero-bit mask is valid and means the simulator proved the next state has
    no legal actions. It is distinct from None, which means no exact mask was
    captured for the row.
    """
    if value is None:
        return None
    expected_shape = (ACTION_MASK_SIZE,)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        value_shape = tuple(int(dim) for dim in value.shape)
        if value_shape != expected_shape:
            flattened = value.flatten().tolist() if hasattr(value, "flatten") else value.tolist()
            flattened_count = len(flattened) if isinstance(flattened, list) else 1
            if flattened_count != ACTION_MASK_SIZE:
                raise ValueError(
                    f"next_action_mask must contain {ACTION_MASK_SIZE} values, "
                    f"got {flattened_count}"
                )
            raise ValueError(
                f"next_action_mask must have shape {expected_shape}, got {value_shape}"
            )
        value = value.tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    values = []

    def visit(item) -> None:
        if isinstance(item, (str, bytes, bytearray, memoryview)):
            raise ValueError("next_action_mask must contain boolean/integer values")
        if isinstance(item, Iterable):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool):
            values.append(bool(item))
            return
        try:
            numeric = float(item)
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("next_action_mask must contain boolean/integer values") from exc
        if not math.isfinite(numeric) or numeric != integer or integer not in (0, 1):
            raise ValueError("next_action_mask values must be 0/1 or bool")
        values.append(bool(integer))

    visit(value)
    if len(values) != ACTION_MASK_SIZE:
        raise ValueError(
            f"next_action_mask must contain {ACTION_MASK_SIZE} values, got {len(values)}"
        )
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        nested = any(
            isinstance(item, Iterable) and not isinstance(item, (str, bytes, bytearray, memoryview))
            for item in value
        )
        if nested:
            raise ValueError(f"next_action_mask must have shape {expected_shape}, got nested")
    mask_bits = 0
    for idx, allowed in enumerate(values):
        if allowed:
            mask_bits |= 1 << idx
    return mask_bits


def _decode_action_mask(value: object) -> tuple[bool, ...] | None:
    """Decode nullable six-action replay mask bits from SQLite."""
    if value is None:
        return None
    if isinstance(value, (bool, str, bytes, bytearray, memoryview)):
        raise ValueError("next_action_mask must be an integer bit mask")
    try:
        mask_bits = int(value)
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("next_action_mask must be an integer bit mask") from exc
    max_mask = (1 << ACTION_MASK_SIZE) - 1
    if not math.isfinite(numeric_value) or numeric_value != mask_bits:
        raise ValueError("next_action_mask must be an integer bit mask")
    if mask_bits < 0 or mask_bits > max_mask:
        raise ValueError("next_action_mask contains unknown bits")
    return tuple(bool(mask_bits & (1 << idx)) for idx in range(ACTION_MASK_SIZE))


def next_action_mask_has_valid_action(mask: object) -> bool | None:
    """Return whether an exact next-action mask contains any legal action.

    None means no exact mask was captured, which is different from a captured
    all-false mask that proves the sampled next state cannot bootstrap.
    """
    mask_bits = _coerce_action_mask(mask)
    if mask_bits is None:
        return None
    return mask_bits != 0


def _encode_state_blob(value, field_name: str) -> bytes:
    """Encode an observation as the replay database float32 blob format."""
    values = _flatten_state_values(value, field_name)
    if len(values) != STATE_SIZE:
        raise ValueError(f"{field_name} must contain {STATE_SIZE} values, got {len(values)}")
    return struct.pack(STATE_BLOB_FORMAT, *values)


def _decode_state_blob(blob: bytes, field_name: str) -> tuple[float, ...]:
    """Read a fixed-size float32 observation from a replay database blob."""
    if len(blob) != STATE_BLOB_SIZE:
        value_count = len(blob) // 4
        raise ValueError(f"{field_name} blob contains {value_count} values, expected {STATE_SIZE}")
    values = struct.unpack(STATE_BLOB_FORMAT, blob)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{field_name} blob must contain only finite values")
    return values


def _state_has_out_of_range_features(values: list[float] | tuple[float, ...]) -> bool:
    """Return whether a 58-feature state violates documented feature ranges."""
    return any(
        not math.isfinite(value)
        or value < lower - STATE_RANGE_EPSILON
        or value > upper + STATE_RANGE_EPSILON
        for value, (lower, upper) in zip(values, STATE_FEATURE_BOUNDS)
    )


def _state_has_malformed_direction(values: list[float] | tuple[float, ...]) -> bool:
    """Return whether direction features are not a clean one-hot vector."""
    direction_values = values[:4]
    return not (
        sum(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=STATE_RANGE_EPSILON)
            for value in direction_values
        )
        == 1
        and all(
            math.isclose(value, 0.0, rel_tol=0.0, abs_tol=STATE_RANGE_EPSILON)
            or math.isclose(value, 1.0, rel_tol=0.0, abs_tol=STATE_RANGE_EPSILON)
            for value in direction_values
        )
    )


def _state_has_malformed_semantic_features(values: list[float] | tuple[float, ...]) -> bool:
    """Return whether a decoded state violates trainable semantic feature contracts."""
    boost_value = float(values[BOOST_AVAILABLE_INDEX])
    danger_values = [
        float(value) for value in values[PER_ACTION_DANGER_START:PER_ACTION_DANGER_END]
    ]
    return (
        _state_has_out_of_range_features(values)
        or _state_has_malformed_direction(values)
        or boost_value < 0.0
        or boost_value > 1.0
        or any(value < 0.0 or value > 1.0 for value in danger_values)
    )


def _state_feature_stats_from_rows(
    rows: Iterable,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    """Count replay-quality state features from state/done rows."""
    boost_available_count = 0
    malformed_boost_count = 0
    trapped_state_count = 0
    nonterminal_trapped_state_count = 0
    malformed_danger_count = 0
    valid_state_count = 0
    malformed_state_range_count = 0
    malformed_direction_count = 0
    malformed_state_feature_count = 0
    invalid_state_feature_count = 0

    for state, done in rows:
        try:
            values = _flatten_state_values(state, "state")
        except (TypeError, ValueError, OverflowError):
            invalid_state_feature_count += 1
            continue
        if len(values) != STATE_SIZE:
            invalid_state_feature_count += 1
            continue
        valid_state_count += 1
        if _state_has_out_of_range_features(values):
            malformed_state_range_count += 1
        if _state_has_malformed_direction(values):
            malformed_direction_count += 1
        if _state_has_malformed_semantic_features(values):
            malformed_state_feature_count += 1
        boost_value = float(values[BOOST_AVAILABLE_INDEX])
        if boost_value < 0.0 or boost_value > 1.0:
            malformed_boost_count += 1
        if boost_value >= 0.5:
            boost_available_count += 1

        danger_values = [
            float(value) for value in values[PER_ACTION_DANGER_START:PER_ACTION_DANGER_END]
        ]
        if any(value < 0.0 or value > 1.0 for value in danger_values):
            malformed_danger_count += 1
        if all(value >= ACTION_DANGER_COLLISION_THRESHOLD for value in danger_values):
            trapped_state_count += 1
            if not bool(done):
                nonterminal_trapped_state_count += 1

    return (
        boost_available_count,
        malformed_boost_count,
        trapped_state_count,
        nonterminal_trapped_state_count,
        malformed_danger_count,
        valid_state_count,
        malformed_state_range_count,
        malformed_direction_count,
        malformed_state_feature_count,
        invalid_state_feature_count,
    )


def _state_feature_stats_from_states(
    states: Iterable,
    dones: Iterable | None = None,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Count replay-quality state features from loaded states and optional dones."""
    if dones is None:
        rows = ((state, False) for state in states)
    else:
        rows = zip(states, dones)
    return _state_feature_stats_from_rows(rows)


def _boost_mask_count_from_masks(
    next_action_masks: Iterable | None,
    dones: Iterable | None = None,
) -> int:
    """Count exact next-action masks that allow at least one boost action."""
    if next_action_masks is None:
        return 0

    boost_mask_count = 0
    done_values = list(dones) if dones is not None else None
    for idx, mask in enumerate(next_action_masks):
        if mask is None:
            continue
        if done_values is not None and bool(done_values[idx]):
            continue
        values = list(mask)
        if any(bool(value) for value in values[3:ACTION_SIZE]):
            boost_mask_count += 1

    return boost_mask_count


def _coerce_quality_action_masks(
    next_action_masks: list | None,
) -> tuple[list | None, int, set[int]]:
    """Validate loaded exact masks before replay-quality stats trust them."""
    if next_action_masks is None:
        return None, 0, set()

    valid_masks = []
    invalid_rows: set[int] = set()
    for idx, mask in enumerate(next_action_masks):
        if mask is None:
            valid_masks.append(None)
            continue
        try:
            mask_bits = _coerce_action_mask(mask)
            valid_masks.append(_decode_action_mask(mask_bits))
        except (TypeError, ValueError, OverflowError):
            valid_masks.append(None)
            invalid_rows.add(idx)
    return valid_masks, len(invalid_rows), invalid_rows


def _mask_all_actions_invalid(mask: Iterable | None) -> bool | None:
    """Return whether an exact action mask proves a trapped next state."""
    if mask is None:
        return None
    values = list(mask)
    return not any(bool(value) for value in values[:ACTION_MASK_SIZE])


def _normal_action_mask_from_danger_values(danger_values: Iterable) -> tuple[bool, bool, bool]:
    """Return normal-action validity inferred from per-action danger features."""
    values = [float(value) for value in danger_values]
    return tuple(
        math.isfinite(value) and value < ACTION_DANGER_COLLISION_THRESHOLD for value in values[:3]
    )


def _current_action_invalid_from_state(
    action: int,
    values: list[float] | tuple[float, ...],
) -> tuple[bool, bool, bool]:
    """Return current-action invalidity inferred from compact current-state features.

    The state vector can prove normal actions unsafe from indices 54-56 and can
    prove boost unavailable from index 57. It cannot prove every boosted path is
    safe, so this only counts definite state/action contradictions.
    """
    if action < 0 or action >= ACTION_SIZE:
        return True, False, False

    relative_action = action % 3
    danger_values = values[PER_ACTION_DANGER_START:PER_ACTION_DANGER_END]
    normal_valid = _normal_action_mask_from_danger_values(danger_values)[relative_action]
    if action < 3:
        return not normal_valid, not normal_valid, False

    boost_available = math.isfinite(values[BOOST_AVAILABLE_INDEX]) and (
        values[BOOST_AVAILABLE_INDEX] >= 0.5
    )
    invalid_boost = (not boost_available) or (not normal_valid)
    return invalid_boost, False, invalid_boost


def current_action_invalid_from_state(action: int, state) -> tuple[bool, bool, bool]:
    """Return current-action invalidity for one replay state/action pair."""
    action_idx = _coerce_action(action)
    values = _flatten_state_values(state, "state")
    if len(values) != STATE_SIZE:
        raise ValueError(f"state must contain {STATE_SIZE} values, got {len(values)}")
    return _current_action_invalid_from_state(action_idx, values)


def _exact_mask_state_disagreement(
    mask: Iterable | None,
    danger_values: Iterable,
) -> tuple[bool, bool, bool]:
    """Compare exact normal-action mask bits to state-derived danger bits."""
    if mask is None:
        return False, False, False
    exact_normal = tuple(bool(value) for value in list(mask)[:3])
    state_normal = _normal_action_mask_from_danger_values(danger_values)
    mismatch = exact_normal != state_normal
    unsafe_allowed = any(exact and not state for exact, state in zip(exact_normal, state_normal))
    safe_blocked = any(state and not exact for exact, state in zip(exact_normal, state_normal))
    return mismatch, unsafe_allowed, safe_blocked


def _action_diversity_stats(action_counts: dict, count: int) -> dict:
    """Summarize whether replay covers the six-action policy output space."""
    if count <= 0:
        return {
            "active_action_count": 0,
            "dominant_action": None,
            "dominant_action_fraction": 0.0,
            "action_entropy": 0.0,
            "normalized_action_entropy": 0.0,
        }

    counts = {action: int(action_counts.get(action, 0)) for action in range(ACTION_SIZE)}
    active_action_count = sum(action_count > 0 for action_count in counts.values())
    dominant_action = max(counts, key=lambda action: counts[action])
    dominant_action_count = counts[dominant_action]
    action_entropy = 0.0
    for action_count in counts.values():
        if action_count <= 0:
            continue
        probability = action_count / count
        action_entropy -= probability * math.log(probability)
    normalized_entropy = action_entropy / math.log(ACTION_SIZE) if ACTION_SIZE > 1 else 0.0

    return {
        "active_action_count": active_action_count,
        "dominant_action": dominant_action,
        "dominant_action_fraction": dominant_action_count / count,
        "action_entropy": action_entropy,
        "normalized_action_entropy": normalized_entropy,
    }


def _coerce_quality_values(values: list, coerce_fn) -> tuple[list, int, set[int]]:
    """Return valid quality values plus invalid field/row counts for diagnostics."""
    valid_values = []
    invalid_rows: set[int] = set()
    invalid_count = 0
    for idx, value in enumerate(values):
        try:
            valid_values.append(coerce_fn(value))
        except (TypeError, ValueError, OverflowError):
            invalid_count += 1
            invalid_rows.add(idx)
    return valid_values, invalid_count, invalid_rows


def _validate_replay_quality_field_lengths(actions: list, **fields) -> int:
    """Return row count after ensuring replay-quality fields are row-aligned."""
    row_count = len(actions)
    mismatched = {
        name: len(values)
        for name, values in fields.items()
        if values is not None and len(values) != row_count
    }
    if mismatched:
        details = ", ".join(f"{name}={count}" for name, count in sorted(mismatched.items()))
        raise ValueError(f"Replay quality fields are misaligned: actions={row_count}, {details}")
    return row_count


def _quality_min_avg_max(values: list[float | int]) -> tuple[float, float, float]:
    """Return min/avg/max for already validated quality values."""
    if not values:
        return 0.0, 0.0, 0.0
    float_values = [float(value) for value in values]
    return min(float_values), sum(float_values) / len(float_values), max(float_values)


def format_replay_quality_stats(stats: dict, indent: str = "   ") -> list[str]:
    """Return compact, human-readable replay quality diagnostics."""
    count = int(stats.get("count", 0))
    if count <= 0:
        return [f"{indent}No replay rows found"]

    action_counts = stats.get("action_counts", {})
    action_diversity = _action_diversity_stats(action_counts, count)
    actions = ", ".join(
        f"{action}:{int(action_counts.get(action, 0))}" for action in range(ACTION_SIZE)
    )
    done_count = int(stats.get("done_count", 0))
    terminal_fraction = float(stats.get("terminal_fraction", done_count / count if count else 0.0))
    nonterminal_count = int(stats.get("nonterminal_count", count - done_count))
    mask_count = int(stats.get("mask_count", 0))
    nonterminal_mask_count = int(stats.get("nonterminal_mask_count", mask_count))
    reward_min = float(stats.get("reward_min", 0.0))
    reward_avg = float(stats.get("reward_avg", 0.0))
    reward_max = float(stats.get("reward_max", 0.0))
    reward_negative_count = int(stats.get("reward_negative_count", 0))
    reward_zero_count = int(stats.get("reward_zero_count", 0))
    reward_positive_count = int(stats.get("reward_positive_count", 0))
    terminal_reward_negative_count = int(stats.get("terminal_reward_negative_count", 0))
    terminal_reward_zero_count = int(stats.get("terminal_reward_zero_count", 0))
    terminal_reward_positive_count = int(stats.get("terminal_reward_positive_count", 0))
    terminal_nonnegative_reward_count = int(stats.get("terminal_nonnegative_reward_count", 0))
    terminal_nonnegative_reward_fraction = float(
        stats.get("terminal_nonnegative_reward_fraction", 0.0)
    )
    terminal_immediate_count = int(stats.get("terminal_immediate_count", done_count))
    immediate_terminal_fraction = float(
        stats.get(
            "immediate_terminal_fraction",
            terminal_immediate_count / count if count else 0.0,
        )
    )
    terminal_immediate_fraction = float(
        stats.get(
            "terminal_immediate_fraction",
            terminal_immediate_count / done_count if done_count else 0.0,
        )
    )
    terminal_multistep_count = int(stats.get("terminal_multistep_count", 0))
    terminal_immediate_nonnegative_reward_count = int(
        stats.get("terminal_immediate_nonnegative_reward_count", terminal_nonnegative_reward_count)
    )
    terminal_immediate_nonnegative_reward_fraction = float(
        stats.get("terminal_immediate_nonnegative_reward_fraction", 0.0)
    )
    terminal_multistep_nonnegative_reward_count = int(
        stats.get("terminal_multistep_nonnegative_reward_count", 0)
    )
    terminal_multistep_nonnegative_reward_fraction = float(
        stats.get("terminal_multistep_nonnegative_reward_fraction", 0.0)
    )
    priority_min = float(stats.get("priority_min", 0.0))
    priority_avg = float(stats.get("priority_avg", 0.0))
    priority_max = float(stats.get("priority_max", 0.0))
    steps_min = int(stats.get("bootstrap_steps_min", 0))
    steps_avg = float(stats.get("bootstrap_steps_avg", 0.0))
    steps_max = int(stats.get("bootstrap_steps_max", 0))
    multistep_count = int(stats.get("multistep_count", 0))
    multistep_fraction = float(stats.get("multistep_fraction", 0.0))
    invalid_scalar_row_count = int(stats.get("invalid_scalar_row_count", 0))
    invalid_scalar_field_count = int(stats.get("invalid_scalar_field_count", 0))
    invalid_action_id_count = int(stats.get("invalid_action_id_count", 0))
    invalid_reward_count = int(stats.get("invalid_reward_count", 0))
    invalid_priority_count = int(stats.get("invalid_priority_count", 0))
    invalid_bootstrap_steps_count = int(stats.get("invalid_bootstrap_steps_count", 0))
    invalid_done_count = int(stats.get("invalid_done_count", 0))
    invalid_action_mask_count = int(stats.get("invalid_action_mask_count", 0))
    snake_count = int(stats.get("snake_count", 0))
    snake_rows_min = stats.get("snake_rows_min")
    snake_rows_avg = stats.get("snake_rows_avg")
    snake_rows_max = stats.get("snake_rows_max")
    boost_available_count = stats.get("boost_available_count")
    boost_mask_count = stats.get("boost_mask_count")
    trapped_state_count = stats.get("trapped_state_count")
    nonterminal_trapped_state_count = stats.get("nonterminal_trapped_state_count")
    trapped_next_state_count = stats.get("trapped_next_state_count")
    nonterminal_trapped_next_state_count = stats.get("nonterminal_trapped_next_state_count")
    malformed_state_range_count = stats.get("malformed_state_range_count")
    malformed_next_state_range_count = stats.get("malformed_next_state_range_count")
    malformed_direction_count = stats.get("malformed_direction_feature_count")
    malformed_next_direction_count = stats.get("malformed_next_direction_feature_count")
    malformed_state_feature_count = stats.get("malformed_state_feature_count")
    malformed_next_state_feature_count = stats.get("malformed_next_state_feature_count")
    invalid_state_feature_count = stats.get("invalid_state_feature_count")
    invalid_next_state_feature_count = stats.get("invalid_next_state_feature_count")
    exact_mask_state_comparison_count = stats.get("exact_mask_state_comparison_count")
    exact_mask_state_mismatch_count = stats.get("exact_mask_state_mismatch_count")
    current_action_comparison_count = stats.get("current_action_state_comparison_count")
    invalid_current_action_count = stats.get("invalid_current_action_count")
    active_action_count = int(
        stats.get("active_action_count", action_diversity["active_action_count"])
    )
    dominant_action = stats.get("dominant_action", action_diversity["dominant_action"])
    dominant_action_fraction = float(
        stats.get("dominant_action_fraction", action_diversity["dominant_action_fraction"])
    )
    normalized_action_entropy = float(
        stats.get("normalized_action_entropy", action_diversity["normalized_action_entropy"])
    )

    lines = [
        f"{indent}Rows: {count:,} | terminal: {done_count:,} "
        f"({terminal_fraction:.1%}) | exact masks: {mask_count:,}",
        f"{indent}Nonterminal exact masks: {nonterminal_mask_count:,}/{nonterminal_count:,} "
        f"({float(stats.get('nonterminal_mask_fraction', 0.0)):.1%})",
        f"{indent}Actions: {actions}",
        f"{indent}Action coverage: {active_action_count}/{ACTION_SIZE} | "
        f"dominant: {dominant_action} ({dominant_action_fraction:.1%}) | "
        f"entropy: {normalized_action_entropy:.1%}",
        f"{indent}Rewards min/avg/max: {reward_min:.3f}/{reward_avg:.3f}/{reward_max:.3f}",
        f"{indent}Reward signs neg/zero/pos: "
        f"{reward_negative_count:,}/{reward_zero_count:,}/{reward_positive_count:,}",
        f"{indent}Terminal reward signs neg/zero/pos: "
        f"{terminal_reward_negative_count:,}/{terminal_reward_zero_count:,}/"
        f"{terminal_reward_positive_count:,}; nonnegative="
        f"{terminal_nonnegative_reward_count:,}/{done_count:,} "
        f"({terminal_nonnegative_reward_fraction:.1%}); one_step_bad="
        f"{terminal_immediate_nonnegative_reward_count:,}/{terminal_immediate_count:,} "
        f"({terminal_immediate_nonnegative_reward_fraction:.1%}); n_step_nonnegative="
        f"{terminal_multistep_nonnegative_reward_count:,}/{terminal_multistep_count:,} "
        f"({terminal_multistep_nonnegative_reward_fraction:.1%})",
        f"{indent}One-step terminal rows: {terminal_immediate_count:,}/{done_count:,} "
        f"terminal ({terminal_immediate_fraction:.1%}); "
        f"{terminal_immediate_count:,}/{count:,} rows ({immediate_terminal_fraction:.1%})",
        f"{indent}Priorities min/avg/max: {priority_min:.6f}/{priority_avg:.6f}/{priority_max:.6f}",
        f"{indent}Bootstrap steps min/avg/max: {steps_min}/{steps_avg:.2f}/{steps_max}",
        f"{indent}Multi-step rows: {multistep_count:,}/{count:,} ({multistep_fraction:.1%})",
    ]
    if invalid_scalar_row_count:
        lines.append(
            f"{indent}Invalid scalar rows: {invalid_scalar_row_count:,}/{count:,} "
            f"({float(stats.get('invalid_scalar_fraction', 0.0)):.1%}); "
            f"fields={invalid_scalar_field_count:,}, actions={invalid_action_id_count:,}, "
            f"rewards={invalid_reward_count:,}, priorities={invalid_priority_count:,}, "
            f"bootstrap_steps={invalid_bootstrap_steps_count:,}, dones={invalid_done_count:,}"
        )
    if invalid_action_mask_count:
        lines.append(
            f"{indent}Invalid exact next-action masks: {invalid_action_mask_count:,}/{count:,} "
            f"({float(stats.get('invalid_action_mask_fraction', 0.0)):.1%})"
        )
    if snake_count > 0 and snake_rows_min is not None:
        lines.append(
            f"{indent}Rows per snake_id min/avg/max: "
            f"{int(snake_rows_min):,}/{float(snake_rows_avg):.2f}/{int(snake_rows_max):,}"
        )
    if boost_available_count is not None:
        boost_fraction = float(stats.get("boost_available_fraction", 0.0))
        lines.append(
            f"{indent}Boost available states: {int(boost_available_count):,} "
            f"({boost_fraction:.1%})"
        )
    if boost_mask_count is not None:
        boost_mask_fraction = float(stats.get("boost_mask_fraction", 0.0))
        lines.append(
            f"{indent}Exact masks allowing boost: {int(boost_mask_count):,} "
            f"({boost_mask_fraction:.1%})"
        )
    if current_action_comparison_count is not None:
        comparison_count = int(current_action_comparison_count or 0)
        invalid_count = int(invalid_current_action_count or 0)
        invalid_fraction = float(stats.get("invalid_current_action_fraction", 0.0))
        invalid_normal_count = int(stats.get("invalid_current_normal_action_count", 0))
        invalid_boost_count = int(stats.get("invalid_current_boost_action_count", 0))
        nonterminal_comparison_count = int(
            stats.get(
                "nonterminal_current_action_state_comparison_count",
                comparison_count,
            )
        )
        nonterminal_invalid_count = int(
            stats.get("nonterminal_invalid_current_action_count", invalid_count)
        )
        nonterminal_invalid_fraction = float(
            stats.get("nonterminal_invalid_current_action_fraction", invalid_fraction)
        )
        lines.append(
            f"{indent}Current actions invalid by state: "
            f"{invalid_count:,}/{comparison_count:,} ({invalid_fraction:.1%}); "
            f"nonterminal={nonterminal_invalid_count:,}/{nonterminal_comparison_count:,} "
            f"({nonterminal_invalid_fraction:.1%}); normal={invalid_normal_count:,}, "
            f"boost={invalid_boost_count:,}"
        )
    if trapped_state_count is not None:
        trapped_fraction = float(stats.get("trapped_state_fraction", 0.0))
        nonterminal_trapped_fraction = float(stats.get("nonterminal_trapped_state_fraction", 0.0))
        lines.append(
            f"{indent}Current trapped states: {int(trapped_state_count):,} "
            f"({trapped_fraction:.1%}); nonterminal: "
            f"{int(nonterminal_trapped_state_count or 0):,}/{nonterminal_count:,} "
            f"({nonterminal_trapped_fraction:.1%})"
        )
    if trapped_next_state_count is not None:
        trapped_next_fraction = float(stats.get("trapped_next_state_fraction", 0.0))
        nonterminal_trapped_next_fraction = float(
            stats.get("nonterminal_trapped_next_state_fraction", 0.0)
        )
        lines.append(
            f"{indent}Nonterminal trapped next states: {int(trapped_next_state_count):,} "
            f"({trapped_next_fraction:.1%}); nonterminal targets: "
            f"{int(nonterminal_trapped_next_state_count or 0):,}/{nonterminal_count:,} "
            f"({nonterminal_trapped_next_fraction:.1%})"
        )
    if malformed_state_range_count is not None or malformed_next_state_range_count is not None:
        current_range_count = int(malformed_state_range_count or 0)
        next_range_count = int(malformed_next_state_range_count or 0)
        lines.append(
            f"{indent}Malformed state ranges current/next: "
            f"{current_range_count:,}/{next_range_count:,}"
        )
    if malformed_direction_count is not None or malformed_next_direction_count is not None:
        current_direction_count = int(malformed_direction_count or 0)
        next_direction_count = int(malformed_next_direction_count or 0)
        lines.append(
            f"{indent}Malformed direction one-hot current/next: "
            f"{current_direction_count:,}/{next_direction_count:,}"
        )
    if malformed_state_feature_count is not None or malformed_next_state_feature_count is not None:
        current_malformed_count = int(malformed_state_feature_count or 0)
        next_malformed_count = int(malformed_next_state_feature_count or 0)
        malformed_fraction = float(stats.get("malformed_state_feature_fraction", 0.0))
        lines.append(
            f"{indent}Malformed semantic state features current/next: "
            f"{current_malformed_count:,}/{next_malformed_count:,} "
            f"({malformed_fraction:.1%})"
        )
    if invalid_state_feature_count is not None or invalid_next_state_feature_count is not None:
        current_invalid_count = int(invalid_state_feature_count or 0)
        next_invalid_count = int(invalid_next_state_feature_count or 0)
        lines.append(
            f"{indent}Invalid state observations current/next: "
            f"{current_invalid_count:,}/{next_invalid_count:,}"
        )
    if exact_mask_state_comparison_count is not None:
        comparison_count = int(exact_mask_state_comparison_count or 0)
        mismatch_count = int(exact_mask_state_mismatch_count or 0)
        mismatch_fraction = float(stats.get("exact_mask_state_mismatch_fraction", 0.0))
        unsafe_normal_count = int(stats.get("exact_mask_unsafe_normal_count", 0))
        blocked_safe_count = int(stats.get("exact_mask_blocked_safe_normal_count", 0))
        lines.append(
            f"{indent}Exact mask/state normal-action mismatches: "
            f"{mismatch_count:,}/{comparison_count:,} ({mismatch_fraction:.1%}); "
            f"unsafe_allowed={unsafe_normal_count:,}, safe_blocked={blocked_safe_count:,}"
        )
    return lines


def format_replay_quality_warnings(stats: dict, indent: str = "   ") -> list[str]:
    """Return actionable warnings for replay datasets that may train poorly."""
    count = int(stats.get("count", 0))
    if count <= 0:
        return [f"{indent}No replay rows are available for training"]

    warnings = []
    action_counts = stats.get("action_counts", {})
    action_diversity = _action_diversity_stats(action_counts, count)
    active_action_count = int(
        stats.get("active_action_count", action_diversity["active_action_count"])
    )
    dominant_action = stats.get("dominant_action", action_diversity["dominant_action"])
    dominant_action_fraction = float(
        stats.get("dominant_action_fraction", action_diversity["dominant_action_fraction"])
    )
    done_count = int(stats.get("done_count", 0))
    terminal_fraction = float(stats.get("terminal_fraction", done_count / count if count else 0.0))
    nonterminal_count = int(stats.get("nonterminal_count", count - done_count))
    mask_count = int(stats.get("mask_count", 0))
    nonterminal_mask_count = int(stats.get("nonterminal_mask_count", mask_count))
    reward_min = float(stats.get("reward_min", 0.0))
    reward_max = float(stats.get("reward_max", 0.0))
    reward_negative_count = int(stats.get("reward_negative_count", 0))
    reward_positive_count = int(stats.get("reward_positive_count", 0))
    terminal_nonnegative_reward_count = int(stats.get("terminal_nonnegative_reward_count", 0))
    terminal_immediate_count = int(stats.get("terminal_immediate_count", done_count))
    immediate_terminal_fraction = float(
        stats.get(
            "immediate_terminal_fraction",
            terminal_immediate_count / count if count else 0.0,
        )
    )
    terminal_multistep_count = int(stats.get("terminal_multistep_count", 0))
    terminal_immediate_nonnegative_reward_count = int(
        stats.get("terminal_immediate_nonnegative_reward_count", terminal_nonnegative_reward_count)
    )
    terminal_immediate_nonnegative_reward_fraction = float(
        stats.get("terminal_immediate_nonnegative_reward_fraction", 0.0)
    )
    terminal_multistep_nonnegative_reward_count = int(
        stats.get("terminal_multistep_nonnegative_reward_count", 0)
    )
    terminal_multistep_nonnegative_reward_fraction = float(
        stats.get("terminal_multistep_nonnegative_reward_fraction", 0.0)
    )
    priority_min = float(stats.get("priority_min", 0.0))
    priority_max = float(stats.get("priority_max", 0.0))
    bootstrap_steps_max = int(stats.get("bootstrap_steps_max", 0))
    invalid_scalar_row_count = int(stats.get("invalid_scalar_row_count", 0))
    invalid_scalar_field_count = int(stats.get("invalid_scalar_field_count", 0))
    invalid_action_id_count = int(stats.get("invalid_action_id_count", 0))
    invalid_reward_count = int(stats.get("invalid_reward_count", 0))
    invalid_priority_count = int(stats.get("invalid_priority_count", 0))
    invalid_bootstrap_steps_count = int(stats.get("invalid_bootstrap_steps_count", 0))
    invalid_done_count = int(stats.get("invalid_done_count", 0))
    invalid_action_mask_count = int(stats.get("invalid_action_mask_count", 0))
    snake_count = int(stats.get("snake_count", 0))
    snake_rows_min = int(stats.get("snake_rows_min", 0))
    snake_rows_max = int(stats.get("snake_rows_max", 0))
    dominant_snake_fraction = float(stats.get("dominant_snake_fraction", 0.0))
    boost_available_count = stats.get("boost_available_count")
    boost_mask_count = stats.get("boost_mask_count")
    malformed_boost_count = int(stats.get("malformed_boost_feature_count", 0))
    malformed_danger_count = int(stats.get("malformed_per_action_danger_count", 0))
    malformed_next_danger_count = int(stats.get("malformed_next_per_action_danger_count", 0))
    malformed_state_range_count = int(stats.get("malformed_state_range_count", 0))
    malformed_next_state_range_count = int(stats.get("malformed_next_state_range_count", 0))
    malformed_direction_count = int(stats.get("malformed_direction_feature_count", 0))
    malformed_next_direction_count = int(stats.get("malformed_next_direction_feature_count", 0))
    malformed_state_feature_count = int(stats.get("malformed_state_feature_count", 0))
    malformed_next_state_feature_count = int(stats.get("malformed_next_state_feature_count", 0))
    invalid_state_feature_count = int(stats.get("invalid_state_feature_count", 0))
    invalid_next_state_feature_count = int(stats.get("invalid_next_state_feature_count", 0))
    exact_mask_state_mismatch_count = int(stats.get("exact_mask_state_mismatch_count", 0))
    exact_mask_unsafe_normal_count = int(stats.get("exact_mask_unsafe_normal_count", 0))
    exact_mask_blocked_safe_normal_count = int(stats.get("exact_mask_blocked_safe_normal_count", 0))
    current_action_comparison_count = int(stats.get("current_action_state_comparison_count", 0))
    invalid_current_action_count = int(stats.get("invalid_current_action_count", 0))
    invalid_current_action_fraction = float(stats.get("invalid_current_action_fraction", 0.0))
    invalid_current_normal_action_count = int(stats.get("invalid_current_normal_action_count", 0))
    invalid_current_boost_action_count = int(stats.get("invalid_current_boost_action_count", 0))
    nonterminal_current_action_comparison_count = int(
        stats.get(
            "nonterminal_current_action_state_comparison_count",
            current_action_comparison_count,
        )
    )
    nonterminal_invalid_current_action_count = int(
        stats.get("nonterminal_invalid_current_action_count", invalid_current_action_count)
    )
    nonterminal_invalid_current_action_fraction = float(
        stats.get(
            "nonterminal_invalid_current_action_fraction",
            invalid_current_action_fraction,
        )
    )
    nonterminal_invalid_current_normal_action_count = int(
        stats.get(
            "nonterminal_invalid_current_normal_action_count",
            invalid_current_normal_action_count,
        )
    )
    nonterminal_invalid_current_boost_action_count = int(
        stats.get(
            "nonterminal_invalid_current_boost_action_count",
            invalid_current_boost_action_count,
        )
    )
    has_next_trapped_stats = "nonterminal_trapped_next_state_fraction" in stats
    if has_next_trapped_stats:
        nonterminal_trapped_state_count = int(stats.get("nonterminal_trapped_next_state_count", 0))
        nonterminal_trapped_state_fraction = float(
            stats.get("nonterminal_trapped_next_state_fraction", 0.0)
        )
    else:
        nonterminal_trapped_state_count = int(stats.get("nonterminal_trapped_state_count", 0))
        nonterminal_trapped_state_fraction = float(
            stats.get("nonterminal_trapped_state_fraction", 0.0)
        )

    if invalid_scalar_row_count:
        warnings.append(
            f"{indent}{invalid_scalar_row_count:,} replay rows contain invalid scalar fields "
            f"({invalid_scalar_field_count:,} total: actions={invalid_action_id_count:,}, "
            f"rewards={invalid_reward_count:,}, priorities={invalid_priority_count:,}, "
            f"bootstrap_steps={invalid_bootstrap_steps_count:,}, dones={invalid_done_count:,}); "
            "training gates will reject this replay"
        )
    if invalid_action_mask_count:
        warnings.append(
            f"{indent}{invalid_action_mask_count:,} replay rows contain invalid exact "
            "next-action masks; training gates will reject this replay"
        )

    missing_nonterminal_masks = max(nonterminal_count - nonterminal_mask_count, 0)
    if missing_nonterminal_masks:
        warnings.append(
            f"{indent}{missing_nonterminal_masks:,} nonterminal rows lack exact "
            "next-action masks; "
            "target masking will fall back to state-derived normal actions"
        )

    missing_normal_actions = [
        str(action) for action in range(3) if int(action_counts.get(action, 0)) == 0
    ]
    if missing_normal_actions:
        warnings.append(
            f"{indent}No rows for normal action(s) {', '.join(missing_normal_actions)}; "
            "policy targets may be directionally biased"
        )
    if count >= 128 and active_action_count <= 2:
        warnings.append(
            f"{indent}Only {active_action_count}/{ACTION_SIZE} actions appear in replay; "
            "exploration coverage may be too narrow"
        )
    elif count >= 512 and dominant_action_fraction >= 0.8:
        warnings.append(
            f"{indent}Action {dominant_action} accounts for {dominant_action_fraction:.1%} "
            "of replay rows; policy updates may overfit one behavior"
        )

    if done_count == 0:
        warnings.append(
            f"{indent}No terminal rows in {count:,} transitions; "
            "death/collision learning may be weak"
        )
    elif count >= 512 and terminal_fraction < 0.005:
        warnings.append(
            f"{indent}Only {done_count:,}/{count:,} rows ({terminal_fraction:.2%}) "
            "are terminal; death/collision learning may be weak"
        )
    if done_count > 0 and terminal_immediate_count == 0:
        warnings.append(
            f"{indent}No one-step terminal rows among {done_count:,} terminal samples; "
            f"immediate-terminal coverage is {immediate_terminal_fraction:.1%} of rows; "
            "training gates may reject replay whose death penalty only appears inside "
            "n-step returns"
        )
    if terminal_immediate_nonnegative_reward_count:
        warnings.append(
            f"{indent}{terminal_immediate_nonnegative_reward_count:,}/"
            f"{terminal_immediate_count:,} one-step terminal rows "
            f"({terminal_immediate_nonnegative_reward_fraction:.1%}) have non-negative rewards; "
            "training gates will reject replay that teaches immediate collision terminals "
            "as neutral or good"
        )
    if terminal_multistep_nonnegative_reward_count:
        warnings.append(
            f"{indent}{terminal_multistep_nonnegative_reward_count:,}/"
            f"{terminal_multistep_count:,} n-step terminal rows "
            f"({terminal_multistep_nonnegative_reward_fraction:.1%}) have non-negative returns; "
            "verify this comes from earlier food/kill rewards rather than missing death penalties"
        )

    boost_count = sum(int(action_counts.get(action, 0)) for action in range(3, ACTION_SIZE))
    if boost_mask_count is not None:
        boost_mask_count = int(boost_mask_count)
        if boost_count == 0 and boost_mask_count > 0:
            warnings.append(
                f"{indent}{boost_mask_count:,} exact next-action masks allow boost but no "
                "boost actions were recorded; exploration may under-sample boost"
            )
        elif boost_count > 0 and boost_mask_count == 0:
            warnings.append(
                f"{indent}{boost_count:,} boost-action rows were recorded but no nonterminal "
                "exact next-action masks allow boost; boost target values may be undertrained"
            )
    if boost_available_count is not None:
        boost_available_count = int(boost_available_count)
        if boost_count == 0 and count >= 128 and boost_available_count > 0:
            warnings.append(
                f"{indent}{boost_available_count:,} rows mark boost available but no boost "
                "actions were recorded; boost behavior may remain untrained"
            )
        elif boost_count > 0 and boost_available_count == 0:
            warnings.append(
                f"{indent}{boost_count:,} boost-action rows exist but no current states mark "
                "boost available; check action/state alignment"
            )
    elif count >= 128 and boost_count == 0:
        warnings.append(
            f"{indent}No boost-action rows in {count:,} transitions; "
            "boost policy may remain untrained"
        )

    if nonterminal_invalid_current_action_count:
        if nonterminal_invalid_current_action_fraction >= 0.05 or count < 128:
            warnings.append(
                f"{indent}{nonterminal_invalid_current_action_count:,}/"
                f"{nonterminal_current_action_comparison_count:,} nonterminal stored actions "
                f"({nonterminal_invalid_current_action_fraction:.1%}) are invalid under their "
                "current-state danger/boost features; tune danger exploration or check "
                "action/state alignment"
            )
        elif nonterminal_invalid_current_boost_action_count:
            warnings.append(
                f"{indent}{nonterminal_invalid_current_boost_action_count:,} nonterminal "
                "boost-action rows are invalid under current-state danger/boost features"
            )
        elif nonterminal_invalid_current_normal_action_count:
            warnings.append(
                f"{indent}{nonterminal_invalid_current_normal_action_count:,} nonterminal "
                "normal-action rows are invalid under current-state danger features"
            )

    if malformed_boost_count:
        warnings.append(
            f"{indent}{malformed_boost_count:,} rows have boost-available state feature "
            "outside [0, 1]"
        )
    if malformed_danger_count:
        warnings.append(
            f"{indent}{malformed_danger_count:,} rows have per-action danger features "
            "outside [0, 1]"
        )
    if malformed_next_danger_count:
        warnings.append(
            f"{indent}{malformed_next_danger_count:,} rows have next-state per-action "
            "danger features outside [0, 1]"
        )
    if malformed_state_range_count:
        warnings.append(
            f"{indent}{malformed_state_range_count:,} rows have state features outside "
            "documented ranges; observation scaling may be corrupt"
        )
    if malformed_next_state_range_count:
        warnings.append(
            f"{indent}{malformed_next_state_range_count:,} rows have next-state features "
            "outside documented ranges; target observations may be corrupt"
        )
    if malformed_direction_count:
        warnings.append(
            f"{indent}{malformed_direction_count:,} rows have direction features that are "
            "not one-hot"
        )
    if malformed_next_direction_count:
        warnings.append(
            f"{indent}{malformed_next_direction_count:,} rows have next-state direction "
            "features that are not one-hot"
        )
    if invalid_state_feature_count:
        warnings.append(
            f"{indent}{invalid_state_feature_count:,} rows have invalid current-state "
            "observations; replay loading will reject them"
        )
    if invalid_next_state_feature_count:
        warnings.append(
            f"{indent}{invalid_next_state_feature_count:,} rows have invalid next-state "
            "observations; replay loading will reject them"
        )
    total_malformed_state_feature_count = (
        malformed_state_feature_count
        + malformed_next_state_feature_count
        + invalid_state_feature_count
        + invalid_next_state_feature_count
    )
    if total_malformed_state_feature_count:
        warnings.append(
            f"{indent}{total_malformed_state_feature_count:,} current/next observations "
            "have malformed semantic features or invalid shapes; "
            "strict training gates should reject this replay"
        )
    if exact_mask_unsafe_normal_count:
        warnings.append(
            f"{indent}{exact_mask_unsafe_normal_count:,} exact next-action masks allow "
            "normal actions that next-state danger marks as collisions; "
            "state/action-mask alignment may be corrupt"
        )
    if exact_mask_blocked_safe_normal_count:
        warnings.append(
            f"{indent}{exact_mask_blocked_safe_normal_count:,} exact next-action masks block "
            "normal actions that next-state danger marks safe; "
            "state/action-mask alignment may be drifting"
        )
    elif exact_mask_state_mismatch_count:
        warnings.append(
            f"{indent}{exact_mask_state_mismatch_count:,} exact next-action masks disagree "
            "with next-state per-action danger features"
        )
    if count >= 512 and nonterminal_trapped_state_fraction >= 0.05:
        target_label = "next-state targets" if has_next_trapped_stats else "rows"
        warnings.append(
            f"{indent}{nonterminal_trapped_state_count:,}/{nonterminal_count:,} "
            f"nonterminal {target_label} ({nonterminal_trapped_state_fraction:.1%}) have no "
            "state-derived valid next actions; replay may over-represent dead ends"
        )

    if count >= 128 and snake_count > 1:
        if dominant_snake_fraction >= 0.9:
            warnings.append(
                f"{indent}One snake_id contributes {dominant_snake_fraction:.1%} of replay rows; "
                "multi-snake coverage may be imbalanced"
            )
        elif snake_rows_min > 0 and snake_rows_max / snake_rows_min >= 10.0:
            warnings.append(
                f"{indent}Replay rows per snake_id are highly imbalanced "
                f"({snake_rows_min:,} min vs {snake_rows_max:,} max)"
            )

    if count >= 128 and reward_positive_count == 0:
        warnings.append(
            f"{indent}No positive rewards in {count:,} transitions; "
            "food/kill reward learning may be absent"
        )

    if count >= 128 and reward_negative_count == 0:
        warnings.append(
            f"{indent}No negative rewards in {count:,} transitions; "
            "death/danger avoidance learning may be weak"
        )

    if math.isclose(reward_min, reward_max, rel_tol=0.0, abs_tol=1e-9):
        warnings.append(f"{indent}Reward signal is flat at {reward_min:.3f}")

    if math.isclose(priority_min, priority_max, rel_tol=0.0, abs_tol=1e-9):
        warnings.append(f"{indent}Replay priorities are flat at {priority_min:.6f}")

    if count >= 128 and bootstrap_steps_max <= 1:
        warnings.append(
            f"{indent}All replay rows use bootstrap_steps=1; " "n-step return signal may be absent"
        )

    return warnings


def build_replay_quality_stats(
    actions: list,
    rewards: list,
    dones: list,
    priorities: list,
    bootstrap_steps: list,
    next_action_masks: list | None = None,
    states: list | None = None,
    next_states: list | None = None,
    snake_ids: list | None = None,
) -> dict:
    """Build replay quality diagnostics from already-loaded replay rows."""
    count = _validate_replay_quality_field_lengths(
        actions,
        rewards=rewards,
        dones=dones,
        priorities=priorities,
        bootstrap_steps=bootstrap_steps,
        next_action_masks=next_action_masks,
        states=states,
        next_states=next_states,
        snake_ids=snake_ids,
    )
    if count == 0:
        return {
            "count": 0,
            "done_count": 0,
            "terminal_fraction": 0.0,
            "nonterminal_count": 0,
            "mask_count": 0,
            "mask_fraction": 0.0,
            "nonterminal_mask_count": 0,
            "nonterminal_mask_fraction": 0.0,
            **_action_diversity_stats({}, 0),
            "boost_mask_count": 0,
            "boost_mask_fraction": 0.0,
            "trapped_state_count": 0,
            "trapped_state_fraction": 0.0,
            "nonterminal_trapped_state_count": 0,
            "nonterminal_trapped_state_fraction": 0.0,
            "current_action_state_comparison_count": 0,
            "invalid_current_action_count": 0,
            "invalid_current_action_fraction": 0.0,
            "invalid_current_normal_action_count": 0,
            "invalid_current_boost_action_count": 0,
            "terminal_invalid_current_action_count": 0,
            "nonterminal_current_action_state_comparison_count": 0,
            "nonterminal_invalid_current_action_count": 0,
            "nonterminal_invalid_current_action_fraction": 0.0,
            "nonterminal_invalid_current_normal_action_count": 0,
            "nonterminal_invalid_current_boost_action_count": 0,
            "invalid_scalar_row_count": 0,
            "invalid_scalar_field_count": 0,
            "invalid_scalar_fraction": 0.0,
            "invalid_action_id_count": 0,
            "invalid_reward_count": 0,
            "invalid_priority_count": 0,
            "invalid_bootstrap_steps_count": 0,
            "invalid_done_count": 0,
            "invalid_action_mask_count": 0,
            "invalid_action_mask_fraction": 0.0,
            "malformed_per_action_danger_count": 0,
            "trapped_next_state_count": 0,
            "trapped_next_state_fraction": 0.0,
            "nonterminal_trapped_next_state_count": 0,
            "nonterminal_trapped_next_state_fraction": 0.0,
            "malformed_next_per_action_danger_count": 0,
            "valid_next_state_feature_count": 0,
            "malformed_state_feature_count": 0,
            "malformed_next_state_feature_count": 0,
            "invalid_state_feature_count": 0,
            "invalid_next_state_feature_count": 0,
            "malformed_state_feature_fraction": 0.0,
            "reward_min": 0.0,
            "reward_avg": 0.0,
            "reward_max": 0.0,
            "reward_negative_count": 0,
            "reward_zero_count": 0,
            "reward_positive_count": 0,
            "terminal_reward_negative_count": 0,
            "terminal_reward_zero_count": 0,
            "terminal_reward_positive_count": 0,
            "terminal_nonnegative_reward_count": 0,
            "terminal_nonnegative_reward_fraction": 0.0,
            "terminal_immediate_count": 0,
            "immediate_terminal_fraction": 0.0,
            "terminal_immediate_fraction": 0.0,
            "terminal_multistep_count": 0,
            "terminal_multistep_fraction": 0.0,
            "terminal_immediate_nonnegative_reward_count": 0,
            "terminal_immediate_nonnegative_reward_fraction": 0.0,
            "terminal_multistep_nonnegative_reward_count": 0,
            "terminal_multistep_nonnegative_reward_fraction": 0.0,
            "priority_min": 0.0,
            "priority_avg": 0.0,
            "priority_max": 0.0,
            "bootstrap_steps_min": 0,
            "bootstrap_steps_avg": 0.0,
            "bootstrap_steps_max": 0,
            "multistep_count": 0,
            "multistep_fraction": 0.0,
            "snake_count": 0,
            "snake_rows_min": 0,
            "snake_rows_avg": 0.0,
            "snake_rows_max": 0,
            "dominant_snake_fraction": 0.0,
            "action_counts": {},
        }

    invalid_scalar_rows: set[int] = set()
    reward_values, invalid_reward_count, invalid_rows = _coerce_quality_values(
        rewards,
        lambda value: _coerce_finite_float(value, "reward"),
    )
    invalid_scalar_rows.update(invalid_rows)
    reward_negative_count = sum(reward < 0.0 for reward in reward_values)
    reward_zero_count = sum(
        math.isclose(reward, 0.0, rel_tol=0.0, abs_tol=1e-9) for reward in reward_values
    )
    reward_positive_count = sum(reward > 0.0 for reward in reward_values)
    priority_values, invalid_priority_count, invalid_rows = _coerce_quality_values(
        priorities,
        _coerce_priority,
    )
    invalid_scalar_rows.update(invalid_rows)
    bootstrap_values, invalid_bootstrap_steps_count, invalid_rows = _coerce_quality_values(
        bootstrap_steps,
        _coerce_bootstrap_steps,
    )
    invalid_scalar_rows.update(invalid_rows)
    done_values, invalid_done_count, invalid_rows = _coerce_quality_values(
        dones,
        _coerce_done,
    )
    invalid_scalar_rows.update(invalid_rows)
    terminal_reward_negative_count = 0
    terminal_reward_zero_count = 0
    terminal_reward_positive_count = 0
    terminal_immediate_count = 0
    terminal_multistep_count = 0
    terminal_immediate_nonnegative_reward_count = 0
    terminal_multistep_nonnegative_reward_count = 0
    for reward, done, steps in zip(rewards, dones, bootstrap_steps):
        try:
            reward_value = _coerce_finite_float(reward, "reward")
            done_value = _coerce_done(done)
            bootstrap_step_count = _coerce_bootstrap_steps(steps)
        except (TypeError, ValueError, OverflowError):
            continue
        if not done_value:
            continue
        is_immediate_terminal = bootstrap_step_count <= 1
        if is_immediate_terminal:
            terminal_immediate_count += 1
        else:
            terminal_multistep_count += 1
        if reward_value < 0.0:
            terminal_reward_negative_count += 1
        elif math.isclose(reward_value, 0.0, rel_tol=0.0, abs_tol=1e-9):
            terminal_reward_zero_count += 1
            if is_immediate_terminal:
                terminal_immediate_nonnegative_reward_count += 1
            else:
                terminal_multistep_nonnegative_reward_count += 1
        else:
            terminal_reward_positive_count += 1
            if is_immediate_terminal:
                terminal_immediate_nonnegative_reward_count += 1
            else:
                terminal_multistep_nonnegative_reward_count += 1
    multistep_count = sum(steps > 1 for steps in bootstrap_values)
    action_counts: dict[int, int] = {}
    action_values, invalid_action_id_count, invalid_rows = _coerce_quality_values(
        actions,
        _coerce_action,
    )
    invalid_scalar_rows.update(invalid_rows)
    for action_idx in action_values:
        action_counts[action_idx] = action_counts.get(action_idx, 0) + 1
    invalid_scalar_field_count = (
        invalid_reward_count
        + invalid_priority_count
        + invalid_bootstrap_steps_count
        + invalid_done_count
        + invalid_action_id_count
    )
    invalid_scalar_row_count = len(invalid_scalar_rows)
    invalid_scalar_fraction = invalid_scalar_row_count / count
    action_diversity = _action_diversity_stats(action_counts, count)
    done_count = sum(done_values)
    nonterminal_count = count - done_count
    terminal_nonnegative_reward_count = terminal_reward_zero_count + terminal_reward_positive_count
    terminal_nonnegative_reward_fraction = (
        terminal_nonnegative_reward_count / done_count if done_count else 0.0
    )
    immediate_terminal_fraction = terminal_immediate_count / count if count else 0.0
    terminal_immediate_fraction = terminal_immediate_count / done_count if done_count else 0.0
    terminal_multistep_fraction = terminal_multistep_count / done_count if done_count else 0.0
    terminal_immediate_nonnegative_reward_fraction = (
        terminal_immediate_nonnegative_reward_count / terminal_immediate_count
        if terminal_immediate_count
        else 0.0
    )
    terminal_multistep_nonnegative_reward_fraction = (
        terminal_multistep_nonnegative_reward_count / terminal_multistep_count
        if terminal_multistep_count
        else 0.0
    )
    valid_action_masks, invalid_action_mask_count, _ = _coerce_quality_action_masks(
        next_action_masks
    )
    mask_count = (
        sum(mask is not None for mask in valid_action_masks)
        if valid_action_masks is not None
        else 0
    )
    nonterminal_mask_count = (
        sum(mask is not None and not done for mask, done in zip(valid_action_masks, done_values))
        if valid_action_masks is not None
        else 0
    )
    boost_mask_count = _boost_mask_count_from_masks(valid_action_masks, done_values)
    boost_available_count = None
    malformed_boost_count = 0
    trapped_state_count = 0
    nonterminal_trapped_state_count = 0
    malformed_danger_count = 0
    valid_state_count = 0
    malformed_state_range_count = 0
    malformed_direction_count = 0
    malformed_state_feature_count = 0
    current_action_comparison_count = 0
    invalid_current_action_count = 0
    invalid_current_normal_action_count = 0
    invalid_current_boost_action_count = 0
    terminal_invalid_current_action_count = 0
    nonterminal_current_action_comparison_count = 0
    nonterminal_invalid_current_action_count = 0
    nonterminal_invalid_current_normal_action_count = 0
    nonterminal_invalid_current_boost_action_count = 0
    trapped_next_state_count = 0
    nonterminal_trapped_next_state_count = 0
    malformed_next_danger_count = 0
    exact_mask_state_comparison_count = 0
    exact_mask_state_mismatch_count = 0
    exact_mask_unsafe_normal_count = 0
    exact_mask_blocked_safe_normal_count = 0
    valid_next_state_count = 0
    malformed_next_state_range_count = 0
    malformed_next_direction_count = 0
    malformed_next_state_feature_count = 0
    invalid_state_feature_count = 0
    invalid_next_state_feature_count = 0
    if states is not None:
        (
            boost_available_count,
            malformed_boost_count,
            trapped_state_count,
            nonterminal_trapped_state_count,
            malformed_danger_count,
            valid_state_count,
            malformed_state_range_count,
            malformed_direction_count,
            malformed_state_feature_count,
            invalid_state_feature_count,
        ) = _state_feature_stats_from_states(states, done_values)
        for state, action, done in zip(states, actions, dones):
            try:
                values = _flatten_state_values(state, "state")
            except (TypeError, ValueError):
                continue
            if len(values) != STATE_SIZE:
                continue
            try:
                action_idx = _coerce_action(action)
            except ValueError:
                continue
            try:
                done_value = _coerce_done(done)
            except ValueError:
                done_value = False
            current_action_comparison_count += 1
            if not done_value:
                nonterminal_current_action_comparison_count += 1
            invalid_action, invalid_normal, invalid_boost = _current_action_invalid_from_state(
                action_idx,
                values,
            )
            if invalid_action:
                invalid_current_action_count += 1
                if done_value:
                    terminal_invalid_current_action_count += 1
                else:
                    nonterminal_invalid_current_action_count += 1
            if invalid_normal:
                invalid_current_normal_action_count += 1
                if not done_value:
                    nonterminal_invalid_current_normal_action_count += 1
            if invalid_boost:
                invalid_current_boost_action_count += 1
                if not done_value:
                    nonterminal_invalid_current_boost_action_count += 1
    if next_states is not None:
        masks = valid_action_masks if valid_action_masks is not None else [None] * count
        for next_state, done, mask in zip(next_states, done_values, masks):
            if done:
                continue

            exact_trapped = _mask_all_actions_invalid(mask)
            try:
                values = _flatten_state_values(next_state, "next_state")
            except (TypeError, ValueError):
                invalid_next_state_feature_count += 1
                values = []
            if len(values) == STATE_SIZE:
                valid_next_state_count += 1
                if _state_has_out_of_range_features(values):
                    malformed_next_state_range_count += 1
                if _state_has_malformed_direction(values):
                    malformed_next_direction_count += 1
                if _state_has_malformed_semantic_features(values):
                    malformed_next_state_feature_count += 1
                next_danger_values = [
                    float(value) for value in values[PER_ACTION_DANGER_START:PER_ACTION_DANGER_END]
                ]
                if any(value < 0.0 or value > 1.0 for value in next_danger_values):
                    malformed_next_danger_count += 1
                if mask is not None:
                    exact_mask_state_comparison_count += 1
                    mismatch, unsafe_allowed, safe_blocked = _exact_mask_state_disagreement(
                        mask,
                        next_danger_values,
                    )
                    if mismatch:
                        exact_mask_state_mismatch_count += 1
                    if unsafe_allowed:
                        exact_mask_unsafe_normal_count += 1
                    if safe_blocked:
                        exact_mask_blocked_safe_normal_count += 1

                state_trapped = all(
                    value >= ACTION_DANGER_COLLISION_THRESHOLD for value in next_danger_values
                )
            else:
                if values:
                    invalid_next_state_feature_count += 1
                state_trapped = False

            if exact_trapped if exact_trapped is not None else state_trapped:
                trapped_next_state_count += 1
                nonterminal_trapped_next_state_count += 1
    snake_row_counts = []
    if snake_ids is not None:
        rows_by_snake_id: dict[int, int] = {}
        for snake_id in snake_ids:
            producer_id = int(snake_id)
            rows_by_snake_id[producer_id] = rows_by_snake_id.get(producer_id, 0) + 1
        snake_row_counts = list(rows_by_snake_id.values())
    if snake_row_counts:
        snake_count = len(snake_row_counts)
        snake_rows_min = min(snake_row_counts)
        snake_rows_max = max(snake_row_counts)
        snake_rows_avg = sum(snake_row_counts) / snake_count
        dominant_snake_fraction = snake_rows_max / count
    else:
        snake_count = 0
        snake_rows_min = 0
        snake_rows_max = 0
        snake_rows_avg = 0.0
        dominant_snake_fraction = 0.0
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
    reward_min, reward_avg, reward_max = _quality_min_avg_max(reward_values)
    priority_min, priority_avg, priority_max = _quality_min_avg_max(priority_values)
    bootstrap_steps_min, bootstrap_steps_avg, bootstrap_steps_max = _quality_min_avg_max(
        bootstrap_values
    )

    stats = {
        "count": count,
        "done_count": done_count,
        "terminal_fraction": done_count / count,
        "nonterminal_count": nonterminal_count,
        "mask_count": mask_count,
        "mask_fraction": mask_count / count,
        "nonterminal_mask_count": nonterminal_mask_count,
        "nonterminal_mask_fraction": (
            nonterminal_mask_count / nonterminal_count if nonterminal_count else 0.0
        ),
        "boost_mask_count": boost_mask_count,
        "boost_mask_fraction": boost_mask_count / nonterminal_count if nonterminal_count else 0.0,
        "reward_min": reward_min,
        "reward_avg": reward_avg,
        "reward_max": reward_max,
        "reward_negative_count": reward_negative_count,
        "reward_zero_count": reward_zero_count,
        "reward_positive_count": reward_positive_count,
        "terminal_reward_negative_count": terminal_reward_negative_count,
        "terminal_reward_zero_count": terminal_reward_zero_count,
        "terminal_reward_positive_count": terminal_reward_positive_count,
        "terminal_nonnegative_reward_count": terminal_nonnegative_reward_count,
        "terminal_nonnegative_reward_fraction": terminal_nonnegative_reward_fraction,
        "terminal_immediate_count": terminal_immediate_count,
        "immediate_terminal_fraction": immediate_terminal_fraction,
        "terminal_immediate_fraction": terminal_immediate_fraction,
        "terminal_multistep_count": terminal_multistep_count,
        "terminal_multistep_fraction": terminal_multistep_fraction,
        "terminal_immediate_nonnegative_reward_count": (
            terminal_immediate_nonnegative_reward_count
        ),
        "terminal_immediate_nonnegative_reward_fraction": (
            terminal_immediate_nonnegative_reward_fraction
        ),
        "terminal_multistep_nonnegative_reward_count": (
            terminal_multistep_nonnegative_reward_count
        ),
        "terminal_multistep_nonnegative_reward_fraction": (
            terminal_multistep_nonnegative_reward_fraction
        ),
        "priority_min": priority_min,
        "priority_avg": priority_avg,
        "priority_max": priority_max,
        "bootstrap_steps_min": int(bootstrap_steps_min),
        "bootstrap_steps_avg": bootstrap_steps_avg,
        "bootstrap_steps_max": int(bootstrap_steps_max),
        "multistep_count": multistep_count,
        "multistep_fraction": multistep_count / count,
        "snake_count": snake_count,
        "snake_rows_min": snake_rows_min,
        "snake_rows_avg": snake_rows_avg,
        "snake_rows_max": snake_rows_max,
        "dominant_snake_fraction": dominant_snake_fraction,
        "invalid_scalar_row_count": invalid_scalar_row_count,
        "invalid_scalar_field_count": invalid_scalar_field_count,
        "invalid_scalar_fraction": invalid_scalar_fraction,
        "invalid_action_id_count": invalid_action_id_count,
        "invalid_reward_count": invalid_reward_count,
        "invalid_priority_count": invalid_priority_count,
        "invalid_bootstrap_steps_count": invalid_bootstrap_steps_count,
        "invalid_done_count": invalid_done_count,
        "invalid_action_mask_count": invalid_action_mask_count,
        "invalid_action_mask_fraction": invalid_action_mask_count / count,
        "action_counts": action_counts,
        **action_diversity,
    }
    if boost_available_count is not None:
        stats["boost_available_count"] = boost_available_count
        stats["boost_available_fraction"] = boost_available_count / count
        stats["malformed_boost_feature_count"] = malformed_boost_count
        stats["trapped_state_count"] = trapped_state_count
        stats["trapped_state_fraction"] = trapped_state_count / count
        stats["nonterminal_trapped_state_count"] = nonterminal_trapped_state_count
        stats["nonterminal_trapped_state_fraction"] = (
            nonterminal_trapped_state_count / nonterminal_count if nonterminal_count else 0.0
        )
        stats["malformed_per_action_danger_count"] = malformed_danger_count
        stats["valid_state_feature_count"] = valid_state_count
        stats["malformed_state_range_count"] = malformed_state_range_count
        stats["malformed_direction_feature_count"] = malformed_direction_count
        stats["malformed_state_feature_count"] = malformed_state_feature_count
        stats["invalid_state_feature_count"] = invalid_state_feature_count
        stats["current_action_state_comparison_count"] = current_action_comparison_count
        stats["invalid_current_action_count"] = invalid_current_action_count
        stats["invalid_current_action_fraction"] = (
            invalid_current_action_count / current_action_comparison_count
            if current_action_comparison_count
            else 0.0
        )
        stats["invalid_current_normal_action_count"] = invalid_current_normal_action_count
        stats["invalid_current_boost_action_count"] = invalid_current_boost_action_count
        stats["terminal_invalid_current_action_count"] = terminal_invalid_current_action_count
        stats["nonterminal_current_action_state_comparison_count"] = (
            nonterminal_current_action_comparison_count
        )
        stats["nonterminal_invalid_current_action_count"] = nonterminal_invalid_current_action_count
        stats["nonterminal_invalid_current_action_fraction"] = (
            nonterminal_invalid_current_action_count / nonterminal_current_action_comparison_count
            if nonterminal_current_action_comparison_count
            else 0.0
        )
        stats["nonterminal_invalid_current_normal_action_count"] = (
            nonterminal_invalid_current_normal_action_count
        )
        stats["nonterminal_invalid_current_boost_action_count"] = (
            nonterminal_invalid_current_boost_action_count
        )
        stats["malformed_state_feature_fraction"] = (
            total_malformed_state_feature_count / total_valid_state_feature_count
            if total_valid_state_feature_count
            else 0.0
        )
    if next_states is not None:
        stats["trapped_next_state_count"] = trapped_next_state_count
        stats["trapped_next_state_fraction"] = (
            trapped_next_state_count / nonterminal_count if nonterminal_count else 0.0
        )
        stats["nonterminal_trapped_next_state_count"] = nonterminal_trapped_next_state_count
        stats["nonterminal_trapped_next_state_fraction"] = (
            nonterminal_trapped_next_state_count / nonterminal_count if nonterminal_count else 0.0
        )
        stats["malformed_next_per_action_danger_count"] = malformed_next_danger_count
        stats["valid_next_state_feature_count"] = valid_next_state_count
        stats["malformed_next_state_range_count"] = malformed_next_state_range_count
        stats["malformed_next_direction_feature_count"] = malformed_next_direction_count
        stats["malformed_next_state_feature_count"] = malformed_next_state_feature_count
        stats["invalid_next_state_feature_count"] = invalid_next_state_feature_count
        stats["exact_mask_state_comparison_count"] = exact_mask_state_comparison_count
        stats["exact_mask_state_mismatch_count"] = exact_mask_state_mismatch_count
        stats["exact_mask_state_mismatch_fraction"] = (
            exact_mask_state_mismatch_count / exact_mask_state_comparison_count
            if exact_mask_state_comparison_count
            else 0.0
        )
        stats["exact_mask_unsafe_normal_count"] = exact_mask_unsafe_normal_count
        stats["exact_mask_blocked_safe_normal_count"] = exact_mask_blocked_safe_normal_count
    return stats


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
        self._create_tables()
        self._migrate_legacy_table()

    def _create_tables(self):
        """Create the memories table if it doesn't exist."""
        # Standard memories table for Apex-DQN transitions
        self.cursor.execute(
            """
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
        """
        )

        # Create index for faster queries
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_standard_policy
            ON memories_standard(policy_type)
        """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS replay_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """
        )

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
                    print(f"Migrating {count} legacy memories to new schema...")

                    # Copy data to new table (convert to apex format)
                    self.cursor.execute(
                        """
                        INSERT INTO memories_standard
                            (
                                snake_id, policy_type, state, action,
                                reward, next_state, done, priority, bootstrap_steps
                            )
                        SELECT
                            snake_id, 'apex', state, action, reward, next_state,
                            done, priority, 1
                        FROM memories
                    """
                    )

                    # Rename legacy table
                    self.cursor.execute("ALTER TABLE memories RENAME TO memories_legacy")
                    self.conn.commit()
                    print("Migration complete. Legacy table renamed to 'memories_legacy'")
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

        print(f"Loaded {len(states)} memories for policy '{policy_type}'")
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

        self.cursor.execute(
            """
            SELECT policy_type, COUNT(*)
            FROM memories_standard
            GROUP BY policy_type
        """
        )
        for row in self.cursor.fetchall():
            stats[row[0]] = {"count": row[1], "type": "standard"}

        return stats

    def get_replay_quality_stats(self, policy_type: str = None, snake_id: int = None) -> dict:
        """Return replay diagnostics that help catch unlearnable generated datasets."""
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
                COALESCE(SUM(CASE WHEN next_action_mask IS NOT NULL THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE
                    WHEN done = 0 AND next_action_mask IS NOT NULL
                    THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE
                    WHEN done = 0
                         AND next_action_mask IS NOT NULL
                         AND (next_action_mask & {BOOST_ACTION_MASK_BITS}) != 0
                    THEN 1 ELSE 0 END), 0),
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
            mask_count,
            nonterminal_mask_count,
            boost_mask_count,
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
