"""Pure replay quality tests that avoid importing the Torch runtime."""

import pytest

from src.data.memory_db_handler import (
    build_replay_quality_stats,
    format_replay_quality_stats,
    format_replay_quality_warnings,
    resolve_replay_quality_gate_values,
    validate_replay_quality_gates,
)


def make_semantically_valid_state() -> list[float]:
    """Return a 58-feature state inside documented replay ranges."""
    state = [0.0] * 58
    state[1] = 1.0
    state[4] = 0.2
    state[5] = 0.25
    state[6] = -0.25
    state[7] = 0.4
    state[44] = 0.5
    state[45] = -0.5
    state[46] = 0.1
    state[47] = 1.0
    state[49] = -1.0
    state[50] = -0.2
    state[51] = 0.2
    state[52] = 0.1
    return state


def test_replay_quality_stats_reject_misaligned_required_fields():
    """Replay audits should not compute row fractions from truncated scalar fields."""
    with pytest.raises(ValueError, match="Replay quality fields are misaligned"):
        build_replay_quality_stats(
            actions=[0, 1, 2],
            rewards=[1.0],
            dones=[False],
            priorities=[1.0],
            bootstrap_steps=[3],
        )


def test_replay_quality_stats_reject_misaligned_optional_fields():
    """Optional masks/states must still describe the same replay rows."""
    state = make_semantically_valid_state()

    with pytest.raises(ValueError, match="next_action_masks=1.*states=2"):
        build_replay_quality_stats(
            actions=[0, 1, 2],
            rewards=[0.0, 1.0, -1.0],
            dones=[False, False, True],
            priorities=[1.0, 1.0, 1.0],
            bootstrap_steps=[1, 2, 3],
            next_action_masks=[[True, False, False, False, False, False]],
            states=[state, state],
            next_states=[state, state, state],
        )


def test_replay_quality_stats_flag_current_actions_invalid_under_state_features():
    """Replay audits should reveal actions contradicted by their current-state features."""
    safe_normal = make_semantically_valid_state()
    unsafe_normal = make_semantically_valid_state()
    unsafe_normal[54:57] = [0.0, 1.0, 0.0]
    unsafe_boost = make_semantically_valid_state()
    unsafe_boost[57] = 0.0
    safe_boost = make_semantically_valid_state()
    safe_boost[57] = 1.0

    stats = build_replay_quality_stats(
        actions=[0, 1, 4, 4, 1],
        rewards=[0.0, -1.0, -1.0, 1.0, -0.5],
        dones=[False, True, True, False, False],
        priorities=[1.0, 1.0, 1.0, 1.0, 1.0],
        bootstrap_steps=[1, 1, 1, 1, 1],
        states=[safe_normal, unsafe_normal, unsafe_boost, safe_boost, unsafe_normal],
    )

    assert stats["current_action_state_comparison_count"] == 5
    assert stats["invalid_current_action_count"] == 3
    assert stats["invalid_current_action_fraction"] == pytest.approx(3 / 5)
    assert stats["invalid_current_normal_action_count"] == 2
    assert stats["invalid_current_boost_action_count"] == 1
    assert stats["terminal_invalid_current_action_count"] == 2
    assert stats["nonterminal_current_action_state_comparison_count"] == 3
    assert stats["nonterminal_invalid_current_action_count"] == 1
    assert stats["nonterminal_invalid_current_action_fraction"] == pytest.approx(1 / 3)
    assert stats["nonterminal_invalid_current_normal_action_count"] == 1
    assert stats["nonterminal_invalid_current_boost_action_count"] == 0
    assert any(
        "Current actions invalid by state: 3/5 (60.0%); nonterminal=1/3 (33.3%)" in line
        for line in format_replay_quality_stats(stats)
    )
    assert any(
        "nonterminal stored actions" in warning for warning in format_replay_quality_warnings(stats)
    )

    with pytest.raises(RuntimeError, match="invalid current-action fraction"):
        validate_replay_quality_gates(stats, max_invalid_current_action_fraction=0.25)


def test_replay_quality_gate_allows_terminal_trapped_current_actions():
    """Terminal collision labels should not be rejected as unsafe continuing-policy data."""
    trapped = make_semantically_valid_state()
    trapped[54:57] = [1.0, 1.0, 1.0]

    stats = build_replay_quality_stats(
        actions=[0],
        rewards=[-3.0],
        dones=[True],
        priorities=[1.0],
        bootstrap_steps=[1],
        states=[trapped],
    )

    assert stats["invalid_current_action_count"] == 1
    assert stats["terminal_invalid_current_action_count"] == 1
    assert stats["nonterminal_invalid_current_action_count"] == 0
    assert stats["nonterminal_invalid_current_action_fraction"] == pytest.approx(0.0)
    warnings = format_replay_quality_warnings(stats)
    assert not any("invalid under their current-state" in warning for warning in warnings)

    validate_replay_quality_gates(stats, max_invalid_current_action_fraction=0.0)


def test_replay_quality_gate_rejects_nonterminal_current_action_contradictions():
    """A continuing row should fail when its action contradicts current-state danger bits."""
    unsafe = make_semantically_valid_state()
    unsafe[54:57] = [0.0, 1.0, 0.0]

    stats = build_replay_quality_stats(
        actions=[1],
        rewards=[0.0],
        dones=[False],
        priorities=[1.0],
        bootstrap_steps=[1],
        states=[unsafe],
    )

    assert stats["invalid_current_action_count"] == 1
    assert stats["terminal_invalid_current_action_count"] == 0
    assert stats["nonterminal_invalid_current_action_count"] == 1

    with pytest.raises(RuntimeError, match="nonterminal invalid current-action fraction"):
        validate_replay_quality_gates(stats, max_invalid_current_action_fraction=0.0)


def test_replay_quality_warnings_flag_missing_n_step_signal():
    """A large all-1-step replay set should not masquerade as n-step Ape-X data."""
    stats = {
        "count": 256,
        "done_count": 4,
        "nonterminal_count": 252,
        "mask_count": 256,
        "nonterminal_mask_count": 252,
        "reward_min": -1.0,
        "reward_max": 1.0,
        "reward_negative_count": 4,
        "reward_positive_count": 8,
        "priority_min": 0.5,
        "priority_max": 1.0,
        "bootstrap_steps_max": 1,
        "action_counts": {0: 86, 1: 85, 2: 85},
        "boost_available_count": 0,
        "boost_mask_count": 0,
    }

    warnings = format_replay_quality_warnings(stats)

    assert any("bootstrap_steps=1" in warning for warning in warnings)


def test_replay_quality_warnings_flag_missing_terminal_signal_in_small_smokes():
    """Even short replay smokes should surface missing death/collision targets."""
    stats = {
        "count": 20,
        "done_count": 0,
        "terminal_fraction": 0.0,
        "nonterminal_count": 20,
        "mask_count": 20,
        "nonterminal_mask_count": 20,
        "reward_min": -0.2,
        "reward_max": 0.1,
        "reward_negative_count": 16,
        "reward_positive_count": 4,
        "priority_min": 0.1,
        "priority_max": 1.0,
        "bootstrap_steps_max": 3,
        "action_counts": {1: 18, 2: 2},
        "boost_available_count": 0,
        "boost_mask_count": 0,
    }

    warnings = format_replay_quality_warnings(stats)

    assert any("No terminal rows in 20 transitions" in warning for warning in warnings)


def test_replay_quality_warnings_flag_boost_actions_without_boost_targets():
    """Replay with boost actions still needs boost-valid bootstrap masks."""
    stats = {
        "count": 256,
        "done_count": 4,
        "nonterminal_count": 252,
        "mask_count": 256,
        "nonterminal_mask_count": 252,
        "reward_min": -1.0,
        "reward_max": 1.0,
        "reward_negative_count": 4,
        "reward_positive_count": 8,
        "priority_min": 0.5,
        "priority_max": 1.0,
        "bootstrap_steps_max": 3,
        "action_counts": {0: 50, 1: 50, 2: 50, 3: 50, 4: 28, 5: 28},
        "boost_available_count": 128,
        "boost_mask_count": 0,
    }

    warnings = format_replay_quality_warnings(stats)

    assert any("boost-action rows were recorded" in warning for warning in warnings)


def test_replay_quality_stats_report_multistep_fraction():
    """Replay summaries should expose whether n-step return rows are present."""
    stats = build_replay_quality_stats(
        actions=[0, 1, 2, 1],
        rewards=[0.0, 1.0, -1.0, 0.5],
        dones=[False, False, True, False],
        priorities=[1.0, 2.0, 3.0, 4.0],
        bootstrap_steps=[1, 2, 3, 1],
    )

    assert stats["multistep_count"] == 2
    assert stats["multistep_fraction"] == pytest.approx(0.5)
    assert any(
        "Multi-step rows: 2/4 (50.0%)" in line for line in format_replay_quality_stats(stats)
    )


def test_replay_quality_gate_raises_when_multistep_fraction_is_too_low():
    """Datasets claiming n-step Ape-X training can fail fast when rows are all 1-step."""
    stats = {
        "count": 100,
        "done_count": 0,
        "terminal_fraction": 0.0,
        "nonterminal_count": 100,
        "nonterminal_mask_count": 100,
        "nonterminal_mask_fraction": 1.0,
        "multistep_count": 10,
        "multistep_fraction": 0.1,
    }

    with pytest.raises(RuntimeError, match="multi-step fraction"):
        validate_replay_quality_gates(stats, min_multistep_fraction=0.5)


def test_replay_quality_gate_preset_resolves_with_overrides():
    """Shared presets should keep audit, generation, and offline gates aligned."""
    gates = resolve_replay_quality_gate_values(
        preset="training",
        overrides={
            "min_positive_reward_fraction": 0.2,
            "max_dominant_action_fraction": 0.9,
        },
    )

    assert gates["min_terminal_fraction"] == pytest.approx(0.005)
    assert gates["min_immediate_terminal_fraction"] == pytest.approx(0.001)
    assert gates["min_exact_mask_fraction"] == pytest.approx(0.8)
    assert gates["min_boost_mask_fraction"] == pytest.approx(0.05)
    assert gates["min_positive_reward_fraction"] == pytest.approx(0.2)
    assert gates["min_negative_reward_fraction"] == pytest.approx(0.005)
    assert gates["max_dominant_action_fraction"] == pytest.approx(0.9)
    assert gates["max_exact_mask_state_mismatch_fraction"] == pytest.approx(0.0)
    assert gates["max_malformed_state_feature_fraction"] == pytest.approx(0.0)


def test_replay_quality_gate_raises_when_positive_reward_fraction_is_too_low():
    """Generated replay should be able to require at least some success signal."""
    stats = {
        "count": 100,
        "done_count": 0,
        "terminal_fraction": 0.0,
        "nonterminal_count": 100,
        "nonterminal_mask_count": 100,
        "nonterminal_mask_fraction": 1.0,
        "reward_positive_count": 2,
    }

    with pytest.raises(RuntimeError, match="positive-reward fraction"):
        validate_replay_quality_gates(stats, min_positive_reward_fraction=0.05)


def test_replay_quality_gate_raises_when_boost_mask_fraction_is_too_low():
    """Training can require next-state targets that keep boost learnable."""
    stats = {
        "count": 100,
        "done_count": 0,
        "terminal_fraction": 0.0,
        "nonterminal_count": 100,
        "nonterminal_mask_count": 100,
        "nonterminal_mask_fraction": 1.0,
        "boost_mask_count": 2,
        "boost_mask_fraction": 0.02,
    }

    with pytest.raises(RuntimeError, match="boost-mask fraction"):
        validate_replay_quality_gates(stats, min_boost_mask_fraction=0.05)


def test_replay_quality_gate_raises_when_negative_reward_fraction_is_too_low():
    """Generated replay should be able to require death/danger learning signal."""
    stats = {
        "count": 100,
        "done_count": 0,
        "terminal_fraction": 0.0,
        "nonterminal_count": 100,
        "nonterminal_mask_count": 100,
        "nonterminal_mask_fraction": 1.0,
        "reward_negative_count": 1,
    }

    with pytest.raises(RuntimeError, match="negative-reward fraction"):
        validate_replay_quality_gates(stats, min_negative_reward_fraction=0.05)


def test_replay_quality_gates_reject_nonnegative_terminal_rewards():
    """Terminal coverage is not useful if death rows are neutral or positive targets."""
    stats = build_replay_quality_stats(
        actions=[0, 1, 2, 3],
        rewards=[0.0, 1.0, -0.1, 0.5],
        dones=[True, True, False, False],
        priorities=[1.0, 1.0, 1.0, 1.0],
        bootstrap_steps=[1, 1, 1, 1],
    )

    assert stats["done_count"] == 2
    assert stats["terminal_reward_negative_count"] == 0
    assert stats["terminal_reward_zero_count"] == 1
    assert stats["terminal_reward_positive_count"] == 1
    assert stats["terminal_nonnegative_reward_count"] == 2
    assert stats["terminal_nonnegative_reward_fraction"] == pytest.approx(1.0)
    assert stats["terminal_immediate_count"] == 2
    assert stats["terminal_multistep_count"] == 0
    assert stats["terminal_immediate_nonnegative_reward_count"] == 2
    assert stats["terminal_immediate_nonnegative_reward_fraction"] == pytest.approx(1.0)
    assert stats["terminal_multistep_nonnegative_reward_count"] == 0
    assert any(
        "Terminal reward signs neg/zero/pos: 0/1/1; nonnegative=2/2 (100.0%); "
        "one_step_bad=2/2 (100.0%); n_step_nonnegative=0/0 (0.0%)" in line
        for line in format_replay_quality_stats(stats)
    )
    assert any(
        "one-step terminal rows" in warning and "non-negative rewards" in warning
        for warning in format_replay_quality_warnings(stats)
    )

    with pytest.raises(RuntimeError, match="one-step terminal rows .* non-negative rewards"):
        validate_replay_quality_gates(stats)


def test_replay_quality_gates_reject_nonnegative_multistep_terminal_returns():
    """N-step terminal rows should not turn death into a neutral or positive target."""
    stats = build_replay_quality_stats(
        actions=[1],
        rewards=[3.0 + 0.99 * -3.0],
        dones=[True],
        priorities=[1.0],
        bootstrap_steps=[2],
    )

    assert stats["terminal_reward_positive_count"] == 1
    assert stats["terminal_nonnegative_reward_count"] == 1
    assert stats["terminal_immediate_count"] == 0
    assert stats["terminal_multistep_count"] == 1
    assert stats["terminal_immediate_nonnegative_reward_count"] == 0
    assert stats["terminal_multistep_nonnegative_reward_count"] == 1
    assert stats["terminal_multistep_nonnegative_reward_fraction"] == pytest.approx(1.0)
    assert any(
        "n-step terminal rows" in warning and "non-negative returns" in warning
        for warning in format_replay_quality_warnings(stats)
    )

    with pytest.raises(RuntimeError, match="n-step terminal rows .* non-negative returns"):
        validate_replay_quality_gates(stats)


def test_replay_quality_gates_require_immediate_terminal_rows_when_requested():
    """Training replay should not pass with terminal returns but no direct death samples."""
    stats = build_replay_quality_stats(
        actions=[1, 2],
        rewards=[-0.5, -0.1],
        dones=[True, False],
        priorities=[1.0, 1.0],
        bootstrap_steps=[2, 1],
    )

    assert stats["terminal_immediate_count"] == 0
    assert stats["terminal_multistep_count"] == 1
    assert stats["immediate_terminal_fraction"] == pytest.approx(0.0)
    assert any(
        "No one-step terminal rows" in line for line in format_replay_quality_warnings(stats)
    )

    with pytest.raises(RuntimeError, match="immediate-terminal fraction"):
        validate_replay_quality_gates(stats, min_immediate_terminal_fraction=0.001)


def test_replay_quality_stats_reject_invalid_scalar_fields():
    """Replay audits should not hide NaN rewards or unsampleable priorities."""
    stats = build_replay_quality_stats(
        actions=[0, 9, 1],
        rewards=[0.0, float("nan"), 1.0],
        dones=[False, "bad", True],
        priorities=[1.0, float("inf"), 0.0],
        bootstrap_steps=[1, 0, "x"],
    )

    assert stats["invalid_scalar_row_count"] == 2
    assert stats["invalid_scalar_field_count"] == 7
    assert stats["invalid_action_id_count"] == 1
    assert stats["invalid_reward_count"] == 1
    assert stats["invalid_priority_count"] == 2
    assert stats["invalid_bootstrap_steps_count"] == 2
    assert stats["invalid_done_count"] == 1
    assert stats["reward_avg"] == pytest.approx(0.5)
    assert stats["priority_min"] == pytest.approx(1.0)
    assert stats["priority_avg"] == pytest.approx(1.0)
    assert stats["action_counts"] == {0: 1, 1: 1}
    assert any("Invalid scalar rows: 2/3" in line for line in format_replay_quality_stats(stats))
    assert any(
        "invalid scalar fields" in warning for warning in format_replay_quality_warnings(stats)
    )

    with pytest.raises(RuntimeError, match="invalid scalar replay fields"):
        validate_replay_quality_gates(stats)


def test_replay_quality_stats_count_overflowing_scalar_fields_as_invalid():
    """Overflowing persisted scalars should be reported before training gates run."""
    stats = build_replay_quality_stats(
        actions=[0],
        rewards=[10**10000],
        dones=[False],
        priorities=[1.0],
        bootstrap_steps=[1],
    )

    assert stats["invalid_scalar_row_count"] == 1
    assert stats["invalid_scalar_field_count"] == 1
    assert stats["invalid_reward_count"] == 1
    assert stats["reward_avg"] == pytest.approx(0.0)
    assert any("Invalid scalar rows: 1/1" in line for line in format_replay_quality_stats(stats))

    with pytest.raises(RuntimeError, match="invalid scalar replay fields"):
        validate_replay_quality_gates(stats)


def test_replay_quality_stats_reject_invalid_exact_next_action_masks():
    """Malformed exact masks should not be counted as trustworthy target masks."""
    state = make_semantically_valid_state()

    stats = build_replay_quality_stats(
        actions=[0],
        rewards=[0.0],
        dones=[False],
        priorities=[1.0],
        bootstrap_steps=[1],
        next_action_masks=[[0, 1, 0, 0, 2, 0]],
        states=[state],
        next_states=[state],
    )

    assert stats["invalid_action_mask_count"] == 1
    assert stats["mask_count"] == 0
    assert stats["nonterminal_mask_count"] == 0
    assert stats["boost_mask_count"] == 0
    assert stats["exact_mask_state_comparison_count"] == 0
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


def test_replay_quality_stats_flag_semantically_malformed_states():
    """Replay audits should catch 58-wide states with invalid feature semantics."""
    valid_state = make_semantically_valid_state()
    bad_range_state = make_semantically_valid_state()
    bad_range_state[5] = 1.5
    bad_direction_state = make_semantically_valid_state()
    bad_direction_state[:4] = [0.5, 0.5, 0.0, 0.0]
    bad_next_range_state = make_semantically_valid_state()
    bad_next_range_state[44] = -1.5
    bad_next_direction_state = make_semantically_valid_state()
    bad_next_direction_state[:4] = [0.0, 0.0, 0.0, 0.0]

    stats = build_replay_quality_stats(
        actions=[0, 1, 2],
        rewards=[0.0, 1.0, -1.0],
        dones=[False, False, False],
        priorities=[1.0, 2.0, 3.0],
        bootstrap_steps=[1, 2, 1],
        states=[valid_state, bad_range_state, bad_direction_state],
        next_states=[valid_state, bad_next_range_state, bad_next_direction_state],
    )

    assert stats["malformed_state_range_count"] == 1
    assert stats["malformed_direction_feature_count"] == 1
    assert stats["malformed_next_state_range_count"] == 1
    assert stats["malformed_next_direction_feature_count"] == 1
    assert stats["malformed_state_feature_count"] == 2
    assert stats["malformed_next_state_feature_count"] == 2
    assert stats["malformed_state_feature_fraction"] == pytest.approx(4 / 6)
    assert any(
        "Malformed state ranges current/next: 1/1" in line
        for line in format_replay_quality_stats(stats)
    )
    assert any(
        "Malformed semantic state features current/next: 2/2 (66.7%)" in line
        for line in format_replay_quality_stats(stats)
    )
    warnings = format_replay_quality_warnings(stats)
    assert any("state features outside documented ranges" in warning for warning in warnings)
    assert any("next-state features outside documented ranges" in warning for warning in warnings)
    assert any("direction features that are not one-hot" in warning for warning in warnings)
    assert any(
        "next-state direction features that are not one-hot" in warning for warning in warnings
    )
    assert any("malformed semantic features" in warning for warning in warnings)

    with pytest.raises(RuntimeError, match="malformed state-feature fraction"):
        validate_replay_quality_gates(stats, max_malformed_state_feature_fraction=0.0)


def test_replay_quality_stats_report_malformed_fraction_for_current_states_only():
    """Current-state-only diagnostics should not hide malformed observation rows."""
    valid_state = make_semantically_valid_state()
    bad_state = make_semantically_valid_state()
    bad_state[:4] = [0.0, 0.0, 0.0, 0.0]

    stats = build_replay_quality_stats(
        actions=[0, 1],
        rewards=[0.0, -1.0],
        dones=[False, False],
        priorities=[1.0, 2.0],
        bootstrap_steps=[1, 1],
        states=[valid_state, bad_state],
    )

    assert stats["malformed_state_feature_count"] == 1
    assert stats["malformed_state_feature_fraction"] == pytest.approx(0.5)
    assert any(
        "Malformed semantic state features current/next: 1/0 (50.0%)" in line
        for line in format_replay_quality_stats(stats)
    )

    with pytest.raises(RuntimeError, match="malformed state-feature fraction"):
        validate_replay_quality_gates(stats, max_malformed_state_feature_fraction=0.0)


def test_replay_quality_stats_count_invalid_state_shapes_as_malformed():
    """Invalid-width observations should fail strict replay quality gates."""
    valid_state = make_semantically_valid_state()

    stats = build_replay_quality_stats(
        actions=[0],
        rewards=[0.0],
        dones=[False],
        priorities=[1.0],
        bootstrap_steps=[1],
        states=[valid_state[:-1]],
        next_states=[valid_state[:-2]],
    )

    assert stats["invalid_state_feature_count"] == 1
    assert stats["invalid_next_state_feature_count"] == 1
    assert stats["malformed_state_feature_fraction"] == pytest.approx(1.0)
    assert any(
        "invalid current-state" in warning for warning in format_replay_quality_warnings(stats)
    )
    assert any("invalid next-state" in warning for warning in format_replay_quality_warnings(stats))

    with pytest.raises(RuntimeError, match="malformed state-feature fraction"):
        validate_replay_quality_gates(stats, max_malformed_state_feature_fraction=0.0)


def test_replay_quality_stats_flag_exact_mask_state_danger_disagreements():
    """Exact next-action masks should be audited against next-state danger features."""
    state = make_semantically_valid_state()
    mismatch_next_state = make_semantically_valid_state()
    mismatch_next_state[54:57] = [0.0, 1.0, 0.0]
    trapped_next_state = make_semantically_valid_state()
    trapped_next_state[54:57] = [1.0, 1.0, 1.0]

    stats = build_replay_quality_stats(
        actions=[0, 1],
        rewards=[0.0, -1.0],
        dones=[False, False],
        priorities=[1.0, 2.0],
        bootstrap_steps=[1, 1],
        next_action_masks=[
            [True, True, False, False, False, False],
            [False, False, False, False, False, False],
        ],
        states=[state, state],
        next_states=[mismatch_next_state, trapped_next_state],
    )

    assert stats["exact_mask_state_comparison_count"] == 2
    assert stats["exact_mask_state_mismatch_count"] == 1
    assert stats["exact_mask_state_mismatch_fraction"] == pytest.approx(0.5)
    assert stats["exact_mask_unsafe_normal_count"] == 1
    assert stats["exact_mask_blocked_safe_normal_count"] == 1

    with pytest.raises(RuntimeError, match="exact-mask/state mismatch fraction"):
        validate_replay_quality_gates(stats, max_exact_mask_state_mismatch_fraction=0.0)

    assert any(
        "Exact mask/state normal-action mismatches: 1/2 (50.0%)" in line
        for line in format_replay_quality_stats(stats)
    )
    warnings = format_replay_quality_warnings(stats)
    assert any(
        "allow normal actions that next-state danger marks as collisions" in warning
        for warning in warnings
    )
    assert any(
        "block normal actions that next-state danger marks safe" in warning for warning in warnings
    )
