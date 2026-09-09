"""Semantic tamper and adapter coverage for the Torch-free PQN lifecycle seam."""

from __future__ import annotations

from typing import Any

import pytest

from src.core.runtime_contract import RunProvenance, RuntimeModeContract, canonical_digest
from src.training.pqn_lifecycle import (
    EPISODE_RESET_PER_ENV_AUTORESET,
    EPISODE_SEED_DERIVED,
    POOL_ADMISSION_DISABLED,
    build_pqn_episode_lifecycle_contract,
    validate_pqn_episode_lifecycle_metadata,
)


def _provenance(metadata: dict[str, Any]) -> None:
    runtime = metadata["runtime_contract"]
    provenance = RunProvenance(
        effective_seed=7,
        observation_digest="observation",
        world_digest="world",
        runtime_digest=canonical_digest(runtime),
        reward_digest="reward",
        target_digest=canonical_digest(metadata["target_contract"]),
        sampler_digest=canonical_digest(metadata["sampler_contract"]),
        optimizer_digest="optimizer",
        model_head_digest="model-head",
        source_revision="test",
    )
    metadata.update(provenance.to_metadata())


def _native_metadata() -> dict[str, Any]:
    lifecycle = build_pqn_episode_lifecycle_contract(
        EPISODE_RESET_PER_ENV_AUTORESET, EPISODE_SEED_DERIVED
    )
    lifecycle_digest = canonical_digest(lifecycle)
    policy_source = {
        "schema_version": "pqn-rollout-policy-source/v1",
        "rollout_policy_mode": "snapshot_pool",
        "fixed_policy_identity": None,
        "assignment_lifetime": "environment_episode",
        "lease_lifetime": "environment_episode",
        "pool_admission_mode": POOL_ADMISSION_DISABLED,
        "initial_opponent_checkpoint_sha256": None,
        "initial_opponent_model_head_digest": None,
        "initial_opponent_snapshot_state_sha256": None,
        "episode_lifecycle_contract_digest": lifecycle_digest,
    }
    policy_digest = canonical_digest(policy_source)
    runtime = RuntimeModeContract(
        mode="pqn_train",
        training=True,
        respawn=False,
        hero_terminal=True,
        population_floor=True,
        reset_strategy="per_env_rollout_boundary",
    )
    metadata: dict[str, Any] = {
        "recipe": "corrected-v3",
        "episode_lifecycle_contract": lifecycle,
        "episode_lifecycle_contract_digest": lifecycle_digest,
        "policy_source_contract": policy_source,
        "policy_source_contract_digest": policy_digest,
        "episode_reset_mode": EPISODE_RESET_PER_ENV_AUTORESET,
        "episode_seed_mode": EPISODE_SEED_DERIVED,
        "pool_admission_mode": POOL_ADMISSION_DISABLED,
        "initial_opponent_checkpoint_sha256": None,
        "runtime_contract": runtime.__dict__,
        "runtime_contract_digest": runtime.digest,
        "target_contract": {
            "version": "pqn-qlambda-corrected-v3-lifecycle-v1",
            "episode_lifecycle_contract_digest": lifecycle_digest,
            "population_floor": True,
        },
        "sampler_contract": {
            "version": "pqn-sampler-corrected-v3-lifecycle-v1",
            "episode_lifecycle_contract_digest": lifecycle_digest,
            "policy_source_contract_digest": policy_digest,
            "assignment_lifetime": "environment_episode",
        },
        "rollout_policy_source": {
            "mode": "snapshot_pool",
            "identity": "episode-assigned",
            "policy_source_contract_digest": policy_digest,
        },
    }
    metadata["target_contract_digest"] = canonical_digest(metadata["target_contract"])
    metadata["sampler_contract_digest"] = canonical_digest(metadata["sampler_contract"])
    _provenance(metadata)
    return metadata


def _pre_lifecycle_metadata() -> dict[str, Any]:
    runtime = RuntimeModeContract(
        mode="pqn_train",
        training=True,
        respawn=False,
        hero_terminal=True,
        population_floor=True,
        reset_strategy="batch_episode",
    )
    source = {"mode": "snapshot_pool", "identity": "episode-assigned"}
    target = {
        "version": "pqn-qlambda-corrected-v3",
        "gamma": 0.99,
        "lambda": 0.95,
        "death": "actual_done_reward_only",
        "trapped": "unsupported_alive_empty_resolved_mask_fails",
        "empty_successor_bootstrap": "error_for_valid_alive_row",
        "truncation": "masked_max_q_of_successor",
        "lambda_carry": "next_in_rollout_valid_transition_including_death",
        "validity": "env_transition_valid_and_active_episode_env",
        "inactive_worlds": "active_env_mask_freezes_world_rng_and_events",
        "reset": "whole_batch_at_rollout_boundary_after_all_floor_or_frame_cap",
        "population_floor": True,
        "bootstrap_network": "rollout_frozen_online_network",
        "loss_eligibility": "valid_and_episode_assigned_hero",
        "reward_digest": "a" * 64,
        "action_mask": {"version": "fixture"},
    }
    sampler = {
        "version": "pqn-sampler-corrected-v3",
        "mode": "minibatch",
        "minibatches": 1,
        "minibatch_size": 1,
        "sgd_epochs": None,
        "pad_sgd_batches": False,
        "sgd_seed": None,
        "sampling": "independent_permutation_prefix_per_batch",
        "eligible": "valid_transitions_of_rollout_assigned_hero_slots",
        "flip_augment": False,
        "augmentation": "disabled",
        "sgd_rng": "shared_with_rollout",
        "num_envs": 1,
        "num_snakes": 2,
        "rollout_len": 2,
        "hero_frac": 1.0,
        "pool_capacity": 2,
        "pool_add_interval": 1,
        "requested_pool_capacity": None,
        "requested_pool_add_interval": None,
        "policy_assignment": "episode_pinned_bernoulli_hero_else_immutable_snapshot_pool",
        "forced_hero_slot": 0,
        "empty_pool": "all_heroes",
        "snapshot_identity": "immutable_content_hash_stable_id",
        "assignment_lifetime": "batch_episode",
        "episode_pinning": True,
        "pool_mutation": "admission_deferred_when_all_snapshots_pinned",
        "rollout_policy_source": source,
        "snapshot_admission": "after_sgd_positive_update_index_divisible_by_interval",
        "exploration": {
            "policy": "hero_only_epsilon_greedy_constant_within_rollout",
            "clock": "valid_hero_agent_steps",
            "start": 1.0,
            "end": 0.1,
            "decay_steps": 10,
        },
    }
    metadata: dict[str, Any] = {
        "recipe": "corrected-v3",
        "gamma": 0.99,
        "lambda": 0.95,
        "reward_contract": {"version": "fixture-reward"},
        "action_mask_contract": {"version": "fixture"},
        "runtime_contract": runtime.__dict__,
        "runtime_contract_digest": runtime.digest,
        "target_contract": target,
        "target_contract_digest": canonical_digest(target),
        "sampler_contract": sampler,
        "sampler_contract_digest": canonical_digest(sampler),
        "rollout_policy_source": source,
        "sgd_epochs": None,
        "pad_sgd_batches": False,
        "sgd_seed": None,
        "eps_start": 1.0,
        "eps_end": 0.1,
        "eps_decay_steps": 10,
    }
    target["reward_digest"] = canonical_digest(metadata["reward_contract"])
    target["action_mask"] = dict(metadata["action_mask_contract"])
    metadata["target_contract_digest"] = canonical_digest(target)
    metadata["reward_contract_digest"] = canonical_digest(metadata["reward_contract"])
    metadata["action_mask_contract_digest"] = canonical_digest(metadata["action_mask_contract"])
    _provenance(metadata)
    return metadata


def test_per_environment_lifecycle_requires_derived_seed_stream() -> None:
    """The closed mode mapping rejects a cross-mode autoreset descriptor."""
    with pytest.raises(ValueError, match="requires derived_env_episode_v1"):
        build_pqn_episode_lifecycle_contract("per_env_autoreset_v1", "continuous_env_rng_v1")


def test_native_metadata_crosslinks_runtime_provenance_and_policy_source() -> None:
    metadata = _native_metadata()
    result = validate_pqn_episode_lifecycle_metadata(
        metadata, allow_corrected_v3_adapter=False
    )

    assert result.compatibility is None
    assert result.descriptor["episode_reset_mode"] == EPISODE_RESET_PER_ENV_AUTORESET

    metadata["episode_seed_mode"] = "continuous_env_rng_v1"
    with pytest.raises(ValueError, match="top-level episode_seed_mode"):
        validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=False)


def test_native_metadata_rejects_runtime_substitution_with_recomputed_digest() -> None:
    metadata = _native_metadata()
    forged = dict(metadata["runtime_contract"])
    forged["reset_strategy"] = "batch_episode"
    metadata["runtime_contract"] = forged
    metadata["runtime_contract_digest"] = canonical_digest(forged)
    _provenance(metadata)

    with pytest.raises(ValueError, match="runtime_contract conflicts"):
        validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=False)


def test_complete_corrected_v3_pre_lifecycle_metadata_adapts_without_rewriting_source() -> None:
    metadata = _pre_lifecycle_metadata()
    result = validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=True)

    assert result.compatibility is not None
    assert result.compatibility["source_schema"] == "corrected-v3-pre-lifecycle"
    assert result.descriptor["episode_reset_mode"] == "batch_barrier_v1"

    metadata["target_contract"]["reset"] = "forged"
    metadata["target_contract_digest"] = canonical_digest(metadata["target_contract"])
    _provenance(metadata)
    with pytest.raises(ValueError, match="known corrected-v3"):
        validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=True)


def test_pre_lifecycle_hero_only_pool_can_have_zero_snapshot_capacity() -> None:
    metadata = _pre_lifecycle_metadata()
    metadata["sampler_contract"]["pool_capacity"] = 0
    metadata["sampler_contract_digest"] = canonical_digest(metadata["sampler_contract"])
    _provenance(metadata)

    result = validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=True)

    assert result.compatibility is not None


def test_pre_lifecycle_adapter_rejects_resigned_snapshot_identity() -> None:
    metadata = _pre_lifecycle_metadata()
    metadata["rollout_policy_source"]["identity"] = "forged"
    metadata["sampler_contract"]["rollout_policy_source"] = metadata["rollout_policy_source"]
    metadata["sampler_contract_digest"] = canonical_digest(metadata["sampler_contract"])
    _provenance(metadata)

    with pytest.raises(ValueError, match="snapshot-pool identity"):
        validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=True)


def test_partial_native_metadata_cannot_downgrade_to_the_legacy_adapter() -> None:
    metadata = _pre_lifecycle_metadata()
    metadata["episode_seed_mode"] = "continuous_env_rng_v1"

    with pytest.raises(ValueError, match="partial native"):
        validate_pqn_episode_lifecycle_metadata(metadata, allow_corrected_v3_adapter=True)


def test_native_returned_descriptor_does_not_share_nested_mutable_state() -> None:
    metadata = _native_metadata()
    result = validate_pqn_episode_lifecycle_metadata(
        metadata, allow_corrected_v3_adapter=False
    )
    metadata["episode_lifecycle_contract"]["completion"]["frame_cap"] = "forged"

    assert result.descriptor["completion"]["frame_cap"] == "done_false_final_successor_bootstrap"
    with pytest.raises(TypeError):
        result.descriptor["completion"]["frame_cap"] = "forged"
