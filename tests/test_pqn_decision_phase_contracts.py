"""Focused provenance checks for the opt-in PQN Watch decision phase."""

from __future__ import annotations

import pytest

from src.core.config_loader import PQNSettingsSchema
from src.core.runtime_contract import canonical_digest
from src.training.pqn_lifecycle import validate_pqn_episode_lifecycle_metadata
from tests.test_pqn_lifecycle import (
    _native_metadata,
    _pre_lifecycle_metadata,
    _provenance,
)


def _full_native_metadata() -> dict:
    metadata = _native_metadata()
    metadata["target_contract"].update(
        {
            "gamma": 0.997,
            "lambda": 0.65,
            "death": "actual_done_reward_only",
            "trapped": "unsupported_alive_empty_resolved_mask_fails",
            "empty_successor_bootstrap": "error_for_valid_alive_row",
            "truncation": "masked_max_q_of_successor",
            "lambda_carry": "next_in_rollout_valid_transition_including_death",
            "validity": "env_transition_valid_and_active_episode_env",
            "inactive_worlds": "active_env_mask_freezes_world_rng_and_events",
            "reset": "selected_envs_at_next_rollout_boundary",
            "bootstrap_network": "rollout_frozen_online_network",
            "loss_eligibility": "valid_and_episode_assigned_hero",
            "reward_digest": "reward",
            "action_mask": {"version": "fixture"},
        }
    )
    metadata["target_contract_digest"] = canonical_digest(metadata["target_contract"])
    _provenance(metadata)
    return metadata


def _watch_metadata() -> dict:
    metadata = _full_native_metadata()
    metadata["target_contract"].update(
        {
            "version": "pqn-qlambda-corrected-v3-lifecycle-decision-v1",
            "decision_phase": "watch_pre_move_after_frame_food_maintenance_v1",
            "successor_phase": "next_watch_pre_move_or_terminal_post_transition_v1",
            "rollout_edge_bootstrap": "deepcopy_selector_capture_discards_prepared_clone_v1",
        }
    )
    metadata["decision_phase_mode"] = "watch_pre_move_v1"
    metadata["target_contract_digest"] = canonical_digest(metadata["target_contract"])
    _provenance(metadata)
    return metadata


def test_yaml_schema_accepts_only_the_two_declared_modes() -> None:
    assert PQNSettingsSchema(decision_phase_mode="watch_pre_move_v1").decision_phase_mode == (
        "watch_pre_move_v1"
    )
    with pytest.raises(Exception):
        PQNSettingsSchema(decision_phase_mode="watch_after_move_v1")


def test_native_watch_target_crosslinks_top_level_mode_and_all_phase_facts() -> None:
    metadata = _watch_metadata()
    validated = validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=False)
    assert validated.compatibility is None

    metadata["target_contract"].pop("successor_phase")
    metadata["target_contract_digest"] = canonical_digest(metadata["target_contract"])
    _provenance(metadata)
    with pytest.raises(ValueError, match="successor_phase"):
        validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=False)


def test_legacy_native_target_remains_valid_without_decision_phase_mode() -> None:
    validate_pqn_episode_lifecycle_metadata(
        _full_native_metadata(), allow_corrected_v3_adapter=False
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda target: target.__setitem__("fourth_phase", "forged"), "invalid key set"),
        (lambda target: target.pop("successor_phase"), "invalid key set"),
        (lambda target: target.__setitem__("decision_phase", "wrong"), "decision_phase"),
    ],
)
def test_watch_target_rejects_resigned_phase_forgery(mutation, error: str) -> None:
    metadata = _watch_metadata()
    mutation(metadata["target_contract"])
    metadata["target_contract_digest"] = canonical_digest(metadata["target_contract"])
    _provenance(metadata)
    with pytest.raises(ValueError, match=error):
        validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=False)


def test_legacy_target_rejects_watch_phase_keys_and_legacy_adapter_rejects_mode_marker() -> None:
    native = _full_native_metadata()
    native["target_contract"]["decision_phase"] = "watch_pre_move_after_frame_food_maintenance_v1"
    native["target_contract_digest"] = canonical_digest(native["target_contract"])
    _provenance(native)
    with pytest.raises(ValueError, match="invalid key set"):
        validate_pqn_episode_lifecycle_metadata(native, allow_corrected_v3_adapter=False)

    pre_lifecycle = _pre_lifecycle_metadata()
    pre_lifecycle["decision_phase_mode"] = "watch_pre_move_v1"
    with pytest.raises(ValueError, match="partial native"):
        validate_pqn_episode_lifecycle_metadata(pre_lifecycle, allow_corrected_v3_adapter=True)
