"""Shared complete lifecycle metadata fixtures for serving-consumer tests."""

from __future__ import annotations

from typing import Any, Mapping

from src.core.runtime_contract import (
    RunProvenance,
    RuntimeModeContract,
    canonical_digest,
)


def corrected_v3_pre_lifecycle_metadata(
    *,
    runtime: RuntimeModeContract,
    observation_digest: str,
    world_digest: str,
    model_head_digest: str,
    action_mask_contract: Mapping[str, Any],
    source_revision: str = "test",
) -> dict[str, Any]:
    """Build the complete known pre-lifecycle schema accepted by the C adapter."""
    if runtime.reset_strategy != "batch_episode":
        raise ValueError("the known pre-lifecycle schema used batch_episode")
    source = {"mode": "snapshot_pool", "identity": "episode-assigned"}
    reward = {"version": "fixture-reward"}
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
        "population_floor": runtime.population_floor,
        "bootstrap_network": "rollout_frozen_online_network",
        "loss_eligibility": "valid_and_episode_assigned_hero",
        "reward_digest": canonical_digest(reward),
        "action_mask": dict(action_mask_contract),
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
        "reward_contract": reward,
        "reward_contract_digest": canonical_digest(reward),
        "action_mask_contract": dict(action_mask_contract),
        "action_mask_contract_digest": canonical_digest(action_mask_contract),
        "runtime_contract": dict(runtime.__dict__),
        "runtime_contract_digest": runtime.digest,
        "target_contract": target,
        "target_contract_digest": canonical_digest(target),
        "sampler_contract": sampler,
        "sampler_contract_digest": canonical_digest(sampler),
        "rollout_policy_source": source,
        "minibatches": 1,
        "minibatch_size": 1,
        "sgd_epochs": None,
        "pad_sgd_batches": False,
        "sgd_seed": None,
    }
    provenance = RunProvenance(
        effective_seed=7,
        observation_digest=observation_digest,
        world_digest=world_digest,
        runtime_digest=runtime.digest,
        reward_digest=metadata["reward_contract_digest"],
        target_digest=metadata["target_contract_digest"],
        sampler_digest=metadata["sampler_contract_digest"],
        optimizer_digest="fixture-optimizer",
        model_head_digest=model_head_digest,
        source_revision=source_revision,
    )
    metadata.update(provenance.to_metadata())
    return metadata
