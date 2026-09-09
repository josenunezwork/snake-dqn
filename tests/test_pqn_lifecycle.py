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
        },
        "sampler_contract": {
            "version": "pqn-sampler-corrected-v3-lifecycle-v1",
            "episode_lifecycle_contract_digest": lifecycle_digest,
            "policy_source_contract_digest": policy_digest,
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
