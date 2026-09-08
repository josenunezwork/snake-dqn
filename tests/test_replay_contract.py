"""Tests for versioned replay provenance and trust decisions."""

from dataclasses import replace

import pytest

from src.data.replay_contract import (
    LEGACY_PROVENANCE_KEYS,
    REPLAY_CONTRACT_DIGEST_KEY,
    REPLAY_CONTRACT_KEY,
    REPLAY_VERIFICATION_KEY,
    ReplayContract,
    missing_legacy_fields,
    unverified_legacy_metadata,
    validate_replay_provenance,
)


def replay_contract() -> ReplayContract:
    return ReplayContract(
        policy_type="apex",
        observation={
            "obs_spec": "vector61",
            "input_size": 61,
            "use_free_space": True,
            "use_boundary_as_danger": True,
            "semantic_digest": "obs-digest",
        },
        action={"count": 6, "interpretation": "relative6"},
        mask={
            "schema": "vector_advisory_v1",
            "role": "collision_avoidance_advice",
            "authority": "not_legal_or_terminal_oracle",
            "action_count": 6,
            "encoding": "sqlite_integer_lsb_action_index",
            "presence_rule": "required_nonterminal_nullable_terminal",
            "missing_row_fallback": "state_vector_danger_v1",
        },
        world={"arena_type": "rectangular", "mechanics_version": 2, "width": 400},
        target={"gamma": 0.99, "n_step": 3},
        reward={"version": 2, "digest": "reward-digest", "contract": {"death": -11.0}},
        episode={
            "train_mode": True,
            "allow_respawn": False,
            "population_floor": True,
            "runtime_digest": "runtime-digest",
        },
        seed={"scope": "per_generation_run", "manifest_key": "replay.runs"},
        generator={"identity": "test", "version": 1},
    )


def test_verified_contract_round_trip() -> None:
    contract = replay_contract()

    validation = validate_replay_provenance(
        contract.to_metadata(), "replay.db", expected=contract, exact=True
    )

    assert validation.status == "verified"
    assert validation.contract_digest == contract.digest
    assert validation.missing_fields == ()


@pytest.mark.parametrize(
    ("section", "replacement", "path"),
    [
        ("world", {"arena_type": "circular", "mechanics_version": 2}, "world.arena_type"),
        ("world", {"arena_type": "rectangular", "mechanics_version": 1}, "mechanics_version"),
        ("target", {"gamma": 0.99, "n_step": 1}, "target.n_step"),
        ("mask", {"schema": "legal_only", "action_count": 6}, "mask.schema"),
        (
            "mask",
            {
                "schema": "vector_advisory_v1",
                "role": "collision_avoidance_advice",
                "authority": "not_legal_or_terminal_oracle",
                "action_count": 6,
                "encoding": "sqlite_integer_msb_action_index",
                "presence_rule": "required_nonterminal_nullable_terminal",
                "missing_row_fallback": "state_vector_danger_v1",
            },
            "mask.encoding",
        ),
    ],
)
def test_load_rejects_semantic_mismatch(section: str, replacement: dict, path: str) -> None:
    expected = replay_contract()
    actual = replace(expected, **{section: replacement})

    with pytest.raises(RuntimeError, match=path):
        validate_replay_provenance(actual.to_metadata(), "replay.db", expected=expected)


def test_tampered_contract_digest_is_rejected() -> None:
    metadata = replay_contract().to_metadata()
    metadata[REPLAY_CONTRACT_KEY]["target"]["gamma"] = 0.5

    with pytest.raises(RuntimeError, match="digest"):
        validate_replay_provenance(metadata, "replay.db")
    with pytest.raises(RuntimeError, match="corrupt"):
        validate_replay_provenance(metadata, "replay.db", allow_unverified_legacy=True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("missing_fields", ["target.gamma"]),
        ("missing_fields", None),
        ("fallback_mask_count", 1),
        ("fallback_mask_count", 0.0),
        ("fallback_mask_count", True),
    ],
)
def test_verified_contract_rejects_contradictory_verification_details(
    key: str, value: object
) -> None:
    metadata = replay_contract().to_metadata()
    metadata[REPLAY_VERIFICATION_KEY][key] = value

    with pytest.raises(RuntimeError, match="corrupt"):
        validate_replay_provenance(metadata, "replay.db")


def test_empty_metadata_requires_explicit_legacy_opt_in() -> None:
    with pytest.raises(RuntimeError, match="explicitly allow"):
        validate_replay_provenance({}, "legacy.db")

    validation = validate_replay_provenance(
        {}, "legacy.db", allow_unverified_legacy=True, fallback_mask_count=7
    )

    assert validation.status == "unverified_legacy"
    assert validation.contract_digest is None
    assert "target.gamma" in validation.missing_fields
    assert "target.n_step" in validation.missing_fields
    assert validation.fallback_mask_count == 7
    assert set(validation.missing_fields) == set(LEGACY_PROVENANCE_KEYS)


def test_partial_legacy_metadata_only_removes_directly_recorded_facts() -> None:
    metadata = {
        "generation.state_size": 61,
        "generation.action_size": 6,
        "generation.gamma": 0.99,
        "generation.max_capacity": None,
    }

    missing = missing_legacy_fields(metadata)

    assert "observation.input_size" not in missing
    assert "action.count" not in missing
    assert "target.gamma" not in missing
    assert "world.max_capacity" not in missing
    assert "observation.use_free_space" in missing
    assert "mask.action_count" in missing
    assert "reward.version" in missing
    assert "episode.population_floor" in missing


def test_null_required_legacy_facts_remain_unknown() -> None:
    missing = missing_legacy_fields(
        {
            "generation.gamma": None,
            "generation.apex_n_step": None,
            "generation.circular_geometry": None,
            "generation.max_capacity": None,
        }
    )

    assert "target.gamma" in missing
    assert "target.n_step" in missing
    assert "world.circular_geometry" not in missing
    assert "world.max_capacity" not in missing


def test_asserted_legacy_facts_never_become_verified() -> None:
    metadata = unverified_legacy_metadata(
        {"generation.gamma": 0.99},
        fallback_mask_count=4,
        asserted_facts={"target.gamma": 0.99, "target.n_step": 3},
    )

    validation = validate_replay_provenance(metadata, "legacy.db", allow_unverified_legacy=True)

    assert REPLAY_CONTRACT_KEY not in metadata
    assert REPLAY_CONTRACT_DIGEST_KEY not in metadata
    assert metadata[REPLAY_VERIFICATION_KEY]["status"] == "unverified_legacy"
    assert validation.status == "unverified_legacy"
    assert "target.n_step" in validation.missing_fields


def test_contract_requires_every_semantic_section() -> None:
    metadata = replay_contract().to_metadata()
    del metadata[REPLAY_CONTRACT_KEY]["mask"]

    with pytest.raises(RuntimeError, match="Invalid replay.contract"):
        validate_replay_provenance(metadata, "replay.db")
