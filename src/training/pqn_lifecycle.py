"""Torch-free lifecycle metadata contracts for corrected PQN checkpoints.

The helpers in this module validate serialized semantic facts before a trainer,
serving session, or strict-artifact consumer constructs a world.  They do not
load weights, select a device, or mutate the checkpoint mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional

from src.core.runtime_contract import (
    PQN_TRAIN_RESET_STRATEGIES,
    RESET_STRATEGY_BATCH_EPISODE,
    RESET_STRATEGY_PER_ENV_ROLLOUT_BOUNDARY,
    RunProvenance,
    canonical_digest,
)


EPISODE_RESET_BATCH_BARRIER = "batch_barrier_v1"
EPISODE_RESET_PER_ENV_AUTORESET = "per_env_autoreset_v1"
EPISODE_SEED_CONTINUOUS = "continuous_env_rng_v1"
EPISODE_SEED_DERIVED = "derived_env_episode_v1"
POOL_ADMISSION_SCHEDULED = "scheduled_v1"
POOL_ADMISSION_DISABLED = "disabled_v1"

_LIFECYCLE_SCHEMA_VERSION = "pqn-episode-lifecycle/v1"
_POLICY_SOURCE_SCHEMA_VERSION = "pqn-rollout-policy-source/v1"
_ADAPTER_SCHEMA_VERSION = "pqn-episode-lifecycle-legacy-adapter/v1"
_ADAPTER_POLICY_SOURCE_SCHEMA_VERSION = "pqn-rollout-policy-source-legacy-adapter/v1"
_SHA256_LENGTH = 64

_RESET_TO_RUNTIME = {
    EPISODE_RESET_BATCH_BARRIER: RESET_STRATEGY_BATCH_EPISODE,
    EPISODE_RESET_PER_ENV_AUTORESET: RESET_STRATEGY_PER_ENV_ROLLOUT_BOUNDARY,
}
_LIFECYCLE_KEYS = frozenset(
    {
        "schema_version",
        "episode_reset_mode",
        "episode_seed_mode",
        "reset_timing",
        "completion",
        "invalid_transition_carry",
        "assignment_lifetime",
        "lease_lifetime",
        "world_seed_stream",
        "assignment_seed_stream",
        "action_seed_stream",
    }
)
_POLICY_SOURCE_KEYS = frozenset(
    {
        "schema_version",
        "rollout_policy_mode",
        "fixed_policy_identity",
        "assignment_lifetime",
        "lease_lifetime",
        "pool_admission_mode",
        "initial_opponent_checkpoint_sha256",
        "initial_opponent_model_head_digest",
        "initial_opponent_snapshot_state_sha256",
        "episode_lifecycle_contract_digest",
    }
)
_NATIVE_MARKERS = frozenset(
    {
        "episode_lifecycle_contract",
        "episode_lifecycle_contract_digest",
        "policy_source_contract",
        "policy_source_contract_digest",
        "episode_reset_mode",
        "episode_seed_mode",
        "pool_admission_mode",
        "initial_opponent_checkpoint_sha256",
    }
)
_OLD_TARGET_KEYS = frozenset(
    {
        "version",
        "gamma",
        "lambda",
        "death",
        "trapped",
        "empty_successor_bootstrap",
        "truncation",
        "lambda_carry",
        "validity",
        "inactive_worlds",
        "reset",
        "population_floor",
        "bootstrap_network",
        "loss_eligibility",
        "reward_digest",
        "action_mask",
    }
)
_OLD_SAMPLER_KEYS = frozenset(
    {
        "version",
        "mode",
        "minibatches",
        "minibatch_size",
        "sgd_epochs",
        "pad_sgd_batches",
        "sgd_seed",
        "sampling",
        "eligible",
        "flip_augment",
        "augmentation",
        "sgd_rng",
        "num_envs",
        "num_snakes",
        "rollout_len",
        "hero_frac",
        "pool_capacity",
        "pool_add_interval",
        "requested_pool_capacity",
        "requested_pool_add_interval",
        "policy_assignment",
        "forced_hero_slot",
        "empty_pool",
        "snapshot_identity",
        "assignment_lifetime",
        "episode_pinning",
        "pool_mutation",
        "rollout_policy_source",
        "snapshot_admission",
        "exploration",
    }
)


@dataclass(frozen=True)
class ValidatedPQNLifecycle:
    """Verified effective lifecycle facts for a corrected PQN checkpoint."""

    descriptor: Mapping[str, Any]
    digest: str
    policy_source_descriptor: Mapping[str, Any]
    policy_source_digest: str
    compatibility: Optional[Mapping[str, Any]]


def _immutable(value: Any) -> Any:
    """Copy a descriptor into a read-only mapping for read-only validation."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(item) for item in value)
    return value


def _exact_mapping(value: object, keys: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ValueError(f"{name} has an invalid key set ({'; '.join(details)})")
    return dict(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_digest(metadata: Mapping[str, Any], name: str, descriptor: Mapping[str, Any]) -> str:
    value = metadata.get(name)
    try:
        expected = canonical_digest(descriptor)
    except ValueError as exc:
        raise ValueError(f"{name.removesuffix('_digest')} contains invalid semantic data") from exc
    if not _is_sha256(value) or value != expected:
        raise ValueError(f"{name} does not match its descriptor")
    return value


def _finite_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite probability")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def build_pqn_episode_lifecycle_contract(
    episode_reset_mode: str, episode_seed_mode: str
) -> dict[str, Any]:
    """Build the sole supported lifecycle descriptor for a PQN mode pair."""
    if episode_reset_mode not in _RESET_TO_RUNTIME:
        raise ValueError(f"unsupported PQN episode_reset_mode {episode_reset_mode!r}")
    if episode_seed_mode not in {EPISODE_SEED_CONTINUOUS, EPISODE_SEED_DERIVED}:
        raise ValueError(f"unsupported PQN episode_seed_mode {episode_seed_mode!r}")
    if episode_reset_mode == EPISODE_RESET_PER_ENV_AUTORESET and (
        episode_seed_mode != EPISODE_SEED_DERIVED
    ):
        raise ValueError("per_env_autoreset_v1 requires derived_env_episode_v1")

    per_environment = episode_reset_mode == EPISODE_RESET_PER_ENV_AUTORESET
    derived = episode_seed_mode == EPISODE_SEED_DERIVED
    return {
        "schema_version": _LIFECYCLE_SCHEMA_VERSION,
        "episode_reset_mode": episode_reset_mode,
        "episode_seed_mode": episode_seed_mode,
        "reset_timing": (
            "selected_envs_at_next_rollout_boundary"
            if per_environment
            else "shared_batch_boundary"
        ),
        "completion": {
            "collision_death": "done_true_reward_only",
            "population_floor": "done_false_final_successor_bootstrap",
            "frame_cap": "done_false_final_successor_bootstrap",
        },
        "invalid_transition_carry": "break",
        "assignment_lifetime": "environment_episode" if per_environment else "batch_episode",
        "lease_lifetime": "environment_episode" if per_environment else "batch_episode",
        "world_seed_stream": (
            "pqn/world/env/{e}/episode/{k}" if derived else "config_seed_plus_env_continuous"
        ),
        "assignment_seed_stream": (
            "pqn/assignment/env/{e}/episode/{k}" if derived else "shared_rollout_rng"
        ),
        "action_seed_stream": (
            "pqn/action/env/{e}/episode/{k}" if derived else "shared_rollout_rng"
        ),
    }


def _validate_lifecycle_descriptor(value: object) -> tuple[dict[str, Any], str]:
    descriptor = _exact_mapping(value, _LIFECYCLE_KEYS, "episode_lifecycle_contract")
    if descriptor.get("schema_version") != _LIFECYCLE_SCHEMA_VERSION:
        raise ValueError("unsupported episode_lifecycle_contract schema_version")
    expected = build_pqn_episode_lifecycle_contract(
        descriptor.get("episode_reset_mode"), descriptor.get("episode_seed_mode")
    )
    if descriptor != expected:
        raise ValueError("episode_lifecycle_contract does not match the closed PQN lifecycle")
    return descriptor, canonical_digest(descriptor)


def _validate_policy_source(
    value: object, lifecycle: Mapping[str, Any], lifecycle_digest: str
) -> tuple[dict[str, Any], str]:
    descriptor = _exact_mapping(value, _POLICY_SOURCE_KEYS, "policy_source_contract")
    if descriptor.get("schema_version") != _POLICY_SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported policy_source_contract schema_version")
    if descriptor["episode_lifecycle_contract_digest"] != lifecycle_digest:
        raise ValueError("policy_source lifecycle digest does not match lifecycle contract")
    if descriptor["rollout_policy_mode"] not in {"snapshot_pool", "fixed"}:
        raise ValueError("policy_source has unsupported rollout_policy_mode")
    if descriptor["pool_admission_mode"] not in {
        POOL_ADMISSION_SCHEDULED,
        POOL_ADMISSION_DISABLED,
    }:
        raise ValueError("policy_source has unsupported pool_admission_mode")
    for name in ("assignment_lifetime", "lease_lifetime"):
        if descriptor[name] != lifecycle[name]:
            raise ValueError(f"policy_source {name} does not match lifecycle contract")
    identity = descriptor["fixed_policy_identity"]
    if descriptor["rollout_policy_mode"] == "fixed":
        if not isinstance(identity, str) or not identity:
            raise ValueError("fixed policy_source requires a non-empty fixed_policy_identity")
    elif identity is not None:
        raise ValueError("snapshot_pool policy_source requires fixed_policy_identity=null")
    hash_fields = (
        "initial_opponent_checkpoint_sha256",
        "initial_opponent_model_head_digest",
        "initial_opponent_snapshot_state_sha256",
    )
    for name in hash_fields:
        item = descriptor[name]
        if item is not None and not _is_sha256(item):
            raise ValueError(f"policy_source {name} must be lowercase SHA-256 or null")
    checkpoint_hash = descriptor["initial_opponent_checkpoint_sha256"]
    companion_hashes = [descriptor[name] for name in hash_fields[1:]]
    if checkpoint_hash is None and any(item is not None for item in companion_hashes):
        raise ValueError("policy_source initial snapshot hashes require checkpoint identity")
    if checkpoint_hash is not None and (
        descriptor["rollout_policy_mode"] != "snapshot_pool"
        or any(item is None for item in companion_hashes)
    ):
        raise ValueError("policy_source initial snapshot identity is incomplete or incompatible")
    return descriptor, canonical_digest(descriptor)


def _validate_common_crosslinks(
    metadata: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    lifecycle_digest: str,
    policy_source: Mapping[str, Any],
    policy_source_digest: str,
) -> None:
    runtime = metadata.get("runtime_contract")
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime_contract must be a mapping")
    expected_runtime_digest = _required_digest(metadata, "runtime_contract_digest", runtime)
    try:
        parsed_runtime = dict(runtime)
        if set(parsed_runtime) != {
            "mode", "training", "respawn", "hero_terminal", "population_floor", "reset_strategy"
        }:
            raise ValueError("runtime_contract has an invalid key set")
        if (
            parsed_runtime["mode"] != "pqn_train"
            or parsed_runtime["training"] is not True
            or parsed_runtime["respawn"] is not False
            or parsed_runtime["hero_terminal"] is not True
            or not isinstance(parsed_runtime["population_floor"], bool)
            or parsed_runtime["reset_strategy"] not in PQN_TRAIN_RESET_STRATEGIES
        ):
            raise ValueError("runtime_contract is not a PQN training runtime")
        if parsed_runtime["reset_strategy"] != _RESET_TO_RUNTIME[lifecycle["episode_reset_mode"]]:
            raise ValueError("runtime reset_strategy does not match lifecycle contract")
    except (KeyError, ValueError) as exc:
        raise ValueError("runtime_contract conflicts with lifecycle contract") from exc

    target = metadata.get("target_contract")
    sampler = metadata.get("sampler_contract")
    if not isinstance(target, Mapping) or not isinstance(sampler, Mapping):
        raise ValueError("target_contract and sampler_contract must be mappings")
    target_digest = _required_digest(metadata, "target_contract_digest", target)
    sampler_digest = _required_digest(metadata, "sampler_contract_digest", sampler)
    if target.get("episode_lifecycle_contract_digest") != lifecycle_digest:
        raise ValueError("target_contract lifecycle digest does not match")
    if sampler.get("episode_lifecycle_contract_digest") != lifecycle_digest:
        raise ValueError("sampler_contract lifecycle digest does not match")
    if sampler.get("policy_source_contract_digest") != policy_source_digest:
        raise ValueError("sampler_contract policy-source digest does not match")
    if target.get("version") != "pqn-qlambda-corrected-v3-lifecycle-v1":
        raise ValueError("unsupported lifecycle target contract version")
    if sampler.get("version") != "pqn-sampler-corrected-v3-lifecycle-v1":
        raise ValueError("unsupported lifecycle sampler contract version")
    if target.get("population_floor") is not parsed_runtime["population_floor"]:
        raise ValueError("target_contract population_floor does not match runtime contract")
    if sampler.get("assignment_lifetime") != lifecycle["assignment_lifetime"]:
        raise ValueError("sampler_contract assignment lifetime does not match lifecycle contract")
    for name in ("episode_reset_mode", "episode_seed_mode"):
        if metadata.get(name) != lifecycle[name]:
            raise ValueError(f"top-level {name} does not match episode_lifecycle_contract")
    for name in ("pool_admission_mode", "initial_opponent_checkpoint_sha256"):
        if metadata.get(name) != policy_source[name]:
            raise ValueError(f"top-level {name} does not match policy_source_contract")
    realized = metadata.get("rollout_policy_source")
    if not isinstance(realized, Mapping) or set(realized) != {
        "mode",
        "identity",
        "policy_source_contract_digest",
    }:
        raise ValueError("rollout_policy_source has an invalid key set")
    expected_identity = (
        policy_source["fixed_policy_identity"]
        if policy_source["rollout_policy_mode"] == "fixed"
        else "episode-assigned"
    )
    if (
        realized.get("mode") != policy_source["rollout_policy_mode"]
        or realized.get("identity") != expected_identity
        or realized.get("policy_source_contract_digest") != policy_source_digest
    ):
        raise ValueError("rollout_policy_source does not bind policy_source_contract")
    try:
        provenance = RunProvenance.from_metadata(metadata)
    except ValueError as exc:
        raise ValueError("invalid run_provenance metadata") from exc
    if (
        provenance.runtime_digest != expected_runtime_digest
        or provenance.target_digest != target_digest
        or provenance.sampler_digest != sampler_digest
    ):
        raise ValueError("run_provenance does not agree with lifecycle descriptors")


def _validate_native(metadata: Mapping[str, Any]) -> ValidatedPQNLifecycle:
    if metadata.get("recipe") != "corrected-v3":
        raise ValueError("native PQN lifecycle metadata requires recipe='corrected-v3'")
    lifecycle, expected_digest = _validate_lifecycle_descriptor(
        metadata.get("episode_lifecycle_contract")
    )
    digest = _required_digest(metadata, "episode_lifecycle_contract_digest", lifecycle)
    if digest != expected_digest:
        raise ValueError("episode_lifecycle_contract_digest does not match lifecycle contract")
    policy_source, expected_policy_digest = _validate_policy_source(
        metadata.get("policy_source_contract"), lifecycle, digest
    )
    policy_digest = _required_digest(metadata, "policy_source_contract_digest", policy_source)
    if policy_digest != expected_policy_digest:
        raise ValueError("policy_source_contract_digest does not match policy source")
    _validate_common_crosslinks(metadata, lifecycle, digest, policy_source, policy_digest)
    return ValidatedPQNLifecycle(
        descriptor=_immutable(lifecycle),
        digest=digest,
        policy_source_descriptor=_immutable(policy_source),
        policy_source_digest=policy_digest,
        compatibility=None,
    )


def _validate_legacy_adapter(metadata: Mapping[str, Any]) -> ValidatedPQNLifecycle:
    """Adapt only a complete verified pre-lifecycle corrected-v3 checkpoint."""
    runtime = metadata.get("runtime_contract")
    target = metadata.get("target_contract")
    sampler = metadata.get("sampler_contract")
    source = metadata.get("rollout_policy_source")
    if not all(isinstance(item, Mapping) for item in (runtime, target, sampler, source)):
        raise ValueError("pre-lifecycle corrected-v3 checkpoint has incomplete descriptors")
    runtime_digest = _required_digest(metadata, "runtime_contract_digest", runtime)
    target_digest = _required_digest(metadata, "target_contract_digest", target)
    sampler_digest = _required_digest(metadata, "sampler_contract_digest", sampler)
    if set(runtime) != {
        "mode", "training", "respawn", "hero_terminal", "population_floor", "reset_strategy"
    }:
        raise ValueError("pre-lifecycle runtime_contract has an invalid key set")
    if (
        runtime.get("mode") != "pqn_train"
        or runtime.get("training") is not True
        or runtime.get("respawn") is not False
        or runtime.get("hero_terminal") is not True
        or not isinstance(runtime.get("population_floor"), bool)
        or runtime.get("reset_strategy") != RESET_STRATEGY_BATCH_EPISODE
        or target.get("version") != "pqn-qlambda-corrected-v3"
        or sampler.get("version") != "pqn-sampler-corrected-v3"
    ):
        raise ValueError("unsupported pre-lifecycle corrected-v3 descriptors")
    _exact_mapping(target, _OLD_TARGET_KEYS, "pre-lifecycle target_contract")
    _exact_mapping(sampler, _OLD_SAMPLER_KEYS, "pre-lifecycle sampler_contract")
    if metadata.get("recipe") != "corrected-v3":
        raise ValueError("lifecycle adapter is only valid for corrected-v3 checkpoints")
    if set(source) != {"mode", "identity"} or source.get("mode") not in {"snapshot_pool", "fixed"}:
        raise ValueError("pre-lifecycle rollout_policy_source is invalid")
    if not isinstance(source.get("identity"), str) or not source["identity"]:
        raise ValueError("pre-lifecycle rollout_policy_source identity is invalid")
    expected_target_values = {
        "death": "actual_done_reward_only",
        "trapped": "unsupported_alive_empty_resolved_mask_fails",
        "empty_successor_bootstrap": "error_for_valid_alive_row",
        "truncation": "masked_max_q_of_successor",
        "lambda_carry": "next_in_rollout_valid_transition_including_death",
        "validity": "env_transition_valid_and_active_episode_env",
        "inactive_worlds": "active_env_mask_freezes_world_rng_and_events",
        "reset": "whole_batch_at_rollout_boundary_after_all_floor_or_frame_cap",
        "bootstrap_network": "rollout_frozen_online_network",
        "loss_eligibility": "valid_and_episode_assigned_hero",
    }
    if any(target[name] != value for name, value in expected_target_values.items()):
        raise ValueError("pre-lifecycle target_contract is not the known corrected-v3 contract")
    if not isinstance(target["population_floor"], bool):
        raise ValueError("pre-lifecycle target_contract population_floor is invalid")
    if target["population_floor"] is not runtime["population_floor"]:
        raise ValueError("pre-lifecycle target_contract population floor conflicts with runtime")
    gamma = _finite_probability(target["gamma"], "pre-lifecycle target gamma")
    lambda_ = _finite_probability(target["lambda"], "pre-lifecycle target lambda")
    if metadata.get("gamma") != gamma or metadata.get("lambda") != lambda_:
        raise ValueError("pre-lifecycle target gamma/lambda conflict with top-level metadata")
    reward = metadata.get("reward_contract")
    mask = metadata.get("action_mask_contract")
    if (
        not isinstance(reward, Mapping)
        or not isinstance(mask, Mapping)
        or not _is_sha256(target["reward_digest"])
        or target["reward_digest"] != canonical_digest(reward)
        or target["action_mask"] != dict(mask)
        or metadata.get("reward_contract_digest") != canonical_digest(reward)
        or metadata.get("action_mask_contract_digest") != canonical_digest(mask)
    ):
        raise ValueError("pre-lifecycle target_contract is incomplete")
    if sampler["assignment_lifetime"] != "batch_episode":
        raise ValueError("pre-lifecycle sampler assignment lifetime is invalid")
    if sampler["rollout_policy_source"] != dict(source):
        raise ValueError("pre-lifecycle sampler policy source does not match realized source")
    if source["mode"] == "fixed":
        expected_sampler = {
            "pool_capacity": 0,
            "pool_add_interval": None,
            "empty_pool": "not_applicable_fixed_source",
            "snapshot_identity": "not_applicable_fixed_source",
            "episode_pinning": False,
            "pool_mutation": "not_applicable_fixed_source",
            "snapshot_admission": "disabled_fixed_source",
            "policy_assignment": "common_fixed_policy_for_nonhero_slots",
            "requested_pool_capacity": sampler["requested_pool_capacity"],
            "requested_pool_add_interval": sampler["requested_pool_add_interval"],
        }
    else:
        expected_sampler = {
            "empty_pool": "all_heroes",
            "snapshot_identity": "immutable_content_hash_stable_id",
            "episode_pinning": True,
            "pool_mutation": "admission_deferred_when_all_snapshots_pinned",
            "snapshot_admission": "after_sgd_positive_update_index_divisible_by_interval",
            "policy_assignment": "episode_pinned_bernoulli_hero_else_immutable_snapshot_pool",
            "requested_pool_capacity": None,
            "requested_pool_add_interval": None,
        }
    if any(sampler[name] != value for name, value in expected_sampler.items()):
        raise ValueError("pre-lifecycle sampler_contract is not the known corrected-v3 contract")
    if (
        sampler["mode"]
        != ("minibatch" if sampler["sgd_epochs"] is None else "exact_coverage")
        or sampler["sampling"]
        != (
            "independent_permutation_prefix_per_batch"
            if sampler["sgd_epochs"] is None
            else "per_epoch_permutation_balanced_array_split"
        )
        or sampler["eligible"] != "valid_transitions_of_rollout_assigned_hero_slots"
        or sampler["augmentation"]
        != (
            "disabled"
            if sampler["flip_augment"] is False
            else "legacy_horizontal_flip_per_minibatch_probability_0.5"
        )
        or sampler["sgd_rng"]
        != ("shared_with_rollout" if sampler["sgd_seed"] is None else "independent_seed")
        or not isinstance(sampler["flip_augment"], bool)
        or not isinstance(sampler["pad_sgd_batches"], bool)
        or any(
            isinstance(sampler[name], bool)
            or not isinstance(sampler[name], int)
            or sampler[name] <= 0
            for name in ("minibatches", "minibatch_size", "num_envs", "num_snakes", "rollout_len")
        )
        or not isinstance(sampler["exploration"], Mapping)
    ):
        raise ValueError("pre-lifecycle sampler_contract has invalid known semantics")
    _finite_probability(sampler["hero_frac"], "pre-lifecycle sampler hero_frac")
    if sampler["forced_hero_slot"] != 0:
        raise ValueError("pre-lifecycle sampler forced hero slot is invalid")
    if source["mode"] == "snapshot_pool" and (
        isinstance(sampler["pool_capacity"], bool)
        or not isinstance(sampler["pool_capacity"], int)
        or sampler["pool_capacity"] < 1
        or isinstance(sampler["pool_add_interval"], bool)
        or not isinstance(sampler["pool_add_interval"], int)
        or sampler["pool_add_interval"] < 1
    ):
        raise ValueError("pre-lifecycle snapshot pool parameters are invalid")
    expected_exploration_keys = {"policy", "clock", "start", "end", "decay_steps"}
    if set(sampler["exploration"]) != expected_exploration_keys:
        raise ValueError("pre-lifecycle exploration descriptor has an invalid key set")
    if (
        sampler["exploration"]["policy"]
        != "hero_only_epsilon_greedy_constant_within_rollout"
        or sampler["exploration"]["clock"] != "valid_hero_agent_steps"
        or _finite_probability(sampler["exploration"]["start"], "pre-lifecycle epsilon start")
        != sampler["exploration"]["start"]
        or _finite_probability(sampler["exploration"]["end"], "pre-lifecycle epsilon end")
        != sampler["exploration"]["end"]
        or isinstance(sampler["exploration"]["decay_steps"], bool)
        or not isinstance(sampler["exploration"]["decay_steps"], int)
        or sampler["exploration"]["decay_steps"] <= 0
    ):
        raise ValueError("pre-lifecycle exploration descriptor is invalid")
    for name in ("sgd_epochs", "sgd_seed"):
        value = sampler[name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"pre-lifecycle sampler {name} is invalid")
    for name in ("minibatches", "minibatch_size", "sgd_epochs", "pad_sgd_batches", "sgd_seed"):
        if metadata.get(name) != sampler[name]:
            raise ValueError(f"pre-lifecycle sampler {name} conflicts with top-level metadata")
    source_digest = canonical_digest(source)
    try:
        provenance = RunProvenance.from_metadata(metadata)
    except ValueError as exc:
        raise ValueError("invalid pre-lifecycle run_provenance metadata") from exc
    if (
        provenance.runtime_digest != runtime_digest
        or provenance.target_digest != target_digest
        or provenance.sampler_digest != sampler_digest
    ):
        raise ValueError("pre-lifecycle run_provenance does not agree with descriptors")
    lifecycle = build_pqn_episode_lifecycle_contract(
        EPISODE_RESET_BATCH_BARRIER, EPISODE_SEED_CONTINUOUS
    )
    lifecycle_digest = canonical_digest(lifecycle)
    compatibility = {
        "schema_version": _ADAPTER_SCHEMA_VERSION,
        "source_schema": "corrected-v3-pre-lifecycle",
        "original_runtime_contract_digest": runtime_digest,
        "original_target_contract_digest": target_digest,
        "original_sampler_contract_digest": sampler_digest,
    }
    policy_source = {
        "schema_version": _ADAPTER_POLICY_SOURCE_SCHEMA_VERSION,
        "source_schema": "corrected-v3-pre-lifecycle",
        "source_rollout_policy_source": dict(source),
        "source_rollout_policy_source_digest": source_digest,
        "source_sampler_contract_digest": sampler_digest,
        "effective_episode_lifecycle_contract_digest": lifecycle_digest,
    }
    return ValidatedPQNLifecycle(
        descriptor=_immutable(lifecycle),
        digest=lifecycle_digest,
        policy_source_descriptor=_immutable(policy_source),
        policy_source_digest=canonical_digest(policy_source),
        compatibility=_immutable(compatibility),
    )


def validate_pqn_episode_lifecycle_metadata(
    metadata: Mapping[str, Any], *, allow_corrected_v3_adapter: bool
) -> ValidatedPQNLifecycle:
    """Validate native lifecycle metadata or a narrow corrected-v3 adapter.

    Generic legacy checkpoints are intentionally outside this helper: their
    historical continuation compatibility path remains its existing authority.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("PQN lifecycle metadata must be a mapping")
    has_lifecycle = "episode_lifecycle_contract" in metadata
    if has_lifecycle:
        return _validate_native(metadata)
    if _NATIVE_MARKERS & set(metadata):
        raise ValueError("partial native PQN lifecycle metadata cannot use the legacy adapter")
    if not allow_corrected_v3_adapter:
        raise ValueError("PQN checkpoint lacks native episode lifecycle metadata")
    return _validate_legacy_adapter(metadata)
