"""Versioned provenance and compatibility checks for persisted replay rows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.runtime_contract import canonical_digest

REPLAY_CONTRACT_KEY = "replay.contract"
REPLAY_CONTRACT_DIGEST_KEY = "replay.contract_digest"
REPLAY_VERIFICATION_KEY = "replay.verification"
REPLAY_SCHEMA_VERSION = 1

_REQUIRED_SECTIONS = (
    "observation",
    "action",
    "mask",
    "world",
    "target",
    "reward",
    "episode",
    "seed",
    "generator",
)

# Training may deliberately use a denser/scaled world than deployment. These
# fields still have to match the learner because they change row interpretation
# or the Bellman target. The complete world is retained for lineage.
REPLAY_LOAD_COMPATIBILITY_PATHS = (
    "policy_type",
    "observation.obs_spec",
    "observation.input_size",
    "observation.use_free_space",
    "observation.use_boundary_as_danger",
    "observation.semantic_digest",
    "action.count",
    "action.interpretation",
    "mask.schema",
    "mask.action_count",
    "mask.encoding",
    "mask.presence_rule",
    "mask.missing_row_fallback",
    "world.arena_type",
    "world.mechanics_version",
    "target.gamma",
    "target.n_step",
    "reward.version",
    "reward.digest",
    "episode.train_mode",
    "episode.allow_respawn",
    "episode.population_floor",
    "episode.runtime_digest",
)

# Every semantic fact emitted by the Apex replay generator has a stable path in
# this inventory.  A legacy key is listed only when the old database recorded
# that fact directly; ``None`` deliberately means "unknown".  In particular,
# row shapes and operator assertions cannot establish target, mask, episode, or
# normalization semantics after the fact.
LEGACY_PROVENANCE_KEYS: dict[str, str | None] = {
    "policy_type": "generation.policy_type",
    "observation.obs_spec": "generation.obs_spec",
    "observation.input_size": "generation.state_size",
    "observation.use_free_space": "generation.use_free_space",
    "observation.use_boundary_as_danger": "generation.use_boundary_as_danger",
    "observation.num_sectors": "generation.num_sectors",
    "observation.danger_max_distance": "generation.danger_max_distance",
    "observation.state_layout_version": "generation.state_layout_version",
    "observation.dtype": "generation.observation_dtype",
    "observation.direction_order": "generation.direction_order",
    "observation.game_width": "generation.observation_game_width",
    "observation.game_height": "generation.observation_game_height",
    "observation.segment_size": "generation.observation_segment_size",
    "observation.food_capacity": "generation.observation_food_capacity",
    "observation.arena_type": "generation.observation_arena_type",
    "observation.circular_geometry": "generation.observation_circular_geometry",
    "observation.min_boost_length": "generation.observation_min_boost_length",
    "observation.free_space_bfs_cap": "generation.free_space_bfs_cap",
    "observation.free_space_min_cap": "generation.free_space_min_cap",
    "observation.free_space_length_multiplier": "generation.free_space_length_multiplier",
    "observation.enemy_trend_update_rule": "generation.enemy_trend_update_rule",
    "observation.per_action_tail_vacancy_rule": "generation.per_action_tail_vacancy_rule",
    "observation.per_action_danger_rule": "generation.per_action_danger_rule",
    "observation.semantic_digest": "generation.observation_digest",
    "action.count": "generation.action_size",
    "action.interpretation": "generation.action_interpretation",
    "mask.schema": "generation.mask_schema",
    "mask.action_count": "generation.mask_action_count",
    "mask.encoding": "generation.mask_encoding",
    "mask.presence_rule": "generation.mask_presence_rule",
    "mask.missing_row_fallback": "generation.mask_missing_row_fallback",
    "world.schema_version": "generation.world_schema_version",
    "world.engine": "generation.world_engine",
    "world.width": "generation.board_width",
    "world.height": "generation.board_height",
    "world.segment_size": "generation.segment_size",
    "world.wall_thickness": "generation.wall_thickness",
    "world.arena_type": "generation.arena_type",
    "world.mechanics_version": "generation.mechanics_version",
    "world.num_snakes": "generation.num_snakes",
    "world.max_frames": "generation.frame_limit",
    "world.initial_food": "generation.initial_food",
    "world.max_food": "generation.max_food",
    "world.min_boost_length": "generation.min_boost_length",
    "world.boost_length_cost_frames": "generation.boost_length_cost_frames",
    "world.frame_rate": "generation.frame_rate",
    "world.max_length": "generation.max_length",
    "world.starvation_max_frames": "generation.starvation_max_frames",
    "world.circular_geometry": "generation.circular_geometry",
    "world.max_capacity": "generation.max_capacity",
    "world.kill_scale": "generation.kill_scale",
    "world.death_value": "generation.death_value",
    "world.normalization": "generation.normalization",
    "world.digest": "generation.world_digest",
    "target.gamma": "generation.gamma",
    "target.n_step": "generation.apex_n_step",
    "reward.version": "generation.rewards_version",
    "reward.contract": "generation.reward_contract",
    "reward.digest": "generation.reward_digest",
    "episode.mode": "generation.episode_mode",
    "episode.train_mode": "generation.train_mode",
    "episode.allow_respawn": "generation.allow_respawn",
    "episode.hero_terminal": "generation.hero_terminal",
    "episode.population_floor": "generation.population_floor",
    "episode.reset_strategy": "generation.reset_strategy",
    "episode.runtime_digest": "generation.runtime_digest",
    "seed.effective_seed": "generation.effective_seed",
    "seed.scope": "generation.seed_scope",
    "seed.derivation": "generation.seed_derivation",
    "seed.manifest_key": "generation.seed_manifest_key",
    "generator.identity": "generation.generator_identity",
    "generator.version": "generation.generator_version",
}


def replay_sqlite_hashes(path: str | Path) -> dict[str, str]:
    """Hash durable SQLite content files while excluding mutable SHM locks."""
    source = Path(path)
    hashes = {}
    for suffix, label in (("", "main"), ("-wal", "wal")):
        candidate = Path(f"{source}{suffix}")
        if not candidate.exists():
            continue
        # A read-only connection to a WAL-mode database may create an empty WAL
        # plus SHM locks. Zero bytes contain no durable frames and are equivalent
        # to an absent WAL for source-integrity comparisons.
        if label == "wal" and candidate.stat().st_size == 0:
            continue
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[label] = digest.hexdigest()
    return hashes


def _require_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty object")
    return dict(value)


def _value_at_path(payload: Mapping[str, Any], path: str) -> object:
    value: object = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Replay contract is missing {path}")
        value = value[part]
    return value


@dataclass(frozen=True)
class ReplayContract:
    """Complete semantics needed to append or consume replay as verified data."""

    policy_type: str
    observation: Mapping[str, Any]
    action: Mapping[str, Any]
    mask: Mapping[str, Any]
    world: Mapping[str, Any]
    target: Mapping[str, Any]
    reward: Mapping[str, Any]
    episode: Mapping[str, Any]
    seed: Mapping[str, Any]
    generator: Mapping[str, Any]
    schema_version: int = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported replay schema_version {self.schema_version}")
        if not isinstance(self.policy_type, str) or not self.policy_type:
            raise ValueError("policy_type must be a non-empty string")
        for section in _REQUIRED_SECTIONS:
            object.__setattr__(
                self,
                section,
                _require_mapping(getattr(self, section), f"replay.contract.{section}"),
            )
        # Validation is intentionally delegated to the shared finite-JSON
        # canonicalizer; NaN, infinity and unsupported types cannot be hashed.
        canonical_digest(self.semantic_dict())

    def semantic_dict(self) -> dict[str, Any]:
        """Return a plain JSON-ready mapping whose hash is the row contract."""
        return {
            "schema_version": self.schema_version,
            "policy_type": self.policy_type,
            **{section: dict(getattr(self, section)) for section in _REQUIRED_SECTIONS},
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.semantic_dict())

    def to_metadata(self) -> dict[str, Any]:
        """Serialize a verified replay contract and its status independently."""
        return {
            REPLAY_CONTRACT_KEY: self.semantic_dict(),
            REPLAY_CONTRACT_DIGEST_KEY: self.digest,
            REPLAY_VERIFICATION_KEY: {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "status": "verified",
                "contract_digest": self.digest,
                "missing_fields": [],
                "fallback_mask_count": 0,
            },
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ReplayContract":
        raw = metadata.get(REPLAY_CONTRACT_KEY)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Replay metadata missing required {REPLAY_CONTRACT_KEY}")
        try:
            contract = cls(**dict(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid replay.contract metadata") from exc
        digest = metadata.get(REPLAY_CONTRACT_DIGEST_KEY)
        if not isinstance(digest, str) or digest != contract.digest:
            raise ValueError("replay.contract_digest does not match replay.contract")
        verification = metadata.get(REPLAY_VERIFICATION_KEY)
        if (
            not isinstance(verification, Mapping)
            or verification.get("schema_version") != REPLAY_SCHEMA_VERSION
            or verification.get("status") != "verified"
            or verification.get("contract_digest") != contract.digest
        ):
            raise ValueError("Replay contract is not marked verified")
        missing_fields = verification.get("missing_fields")
        fallback_mask_count = verification.get("fallback_mask_count")
        if (
            not isinstance(missing_fields, Sequence)
            or isinstance(missing_fields, (str, bytes, bytearray))
            or len(missing_fields) != 0
            or not isinstance(fallback_mask_count, int)
            or isinstance(fallback_mask_count, bool)
            or fallback_mask_count != 0
        ):
            raise ValueError("Verified replay has contradictory verification details")
        return contract


@dataclass(frozen=True)
class ReplayValidation:
    """Trust decision attached to a replay load and any resulting checkpoint."""

    status: str
    contract_digest: str | None
    missing_fields: tuple[str, ...]
    fallback_mask_count: int

    def __post_init__(self) -> None:
        if self.status not in {"verified", "unverified_legacy"}:
            raise ValueError("Unknown replay verification status")
        if not isinstance(self.fallback_mask_count, int) or isinstance(
            self.fallback_mask_count, bool
        ):
            raise ValueError("fallback_mask_count must be an integer")
        if self.fallback_mask_count < 0:
            raise ValueError("fallback_mask_count must be non-negative")
        if self.status == "verified" and (
            self.missing_fields or not self.contract_digest or self.fallback_mask_count != 0
        ):
            raise ValueError(
                "Verified replay requires a digest, no missing fields, and no mask fallback"
            )
        if self.status == "unverified_legacy" and self.contract_digest is not None:
            raise ValueError("Legacy replay cannot carry a verified contract digest")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "status": self.status,
            "contract_digest": self.contract_digest,
            "missing_fields": list(self.missing_fields),
            "fallback_mask_count": self.fallback_mask_count,
        }


def missing_legacy_fields(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Return unknown legacy facts without guessing from replay row shape."""
    return tuple(
        path for path, key in LEGACY_PROVENANCE_KEYS.items() if key is None or key not in metadata
    )


def validate_replay_provenance(
    metadata: Mapping[str, Any],
    db_path: str,
    *,
    expected: ReplayContract | None = None,
    allow_unverified_legacy: bool = False,
    fallback_mask_count: int = 0,
    exact: bool = False,
) -> ReplayValidation:
    """Validate a replay contract or return an explicit legacy trust decision."""
    try:
        contract = ReplayContract.from_metadata(metadata)
    except ValueError as exc:
        claimed_contract = REPLAY_CONTRACT_KEY in metadata or REPLAY_CONTRACT_DIGEST_KEY in metadata
        if claimed_contract:
            raise RuntimeError(f"Replay provenance for {db_path} is corrupt: {exc}") from exc
        if not allow_unverified_legacy:
            raise RuntimeError(
                f"Replay provenance for {db_path} is incomplete or unverified: {exc}. "
                "Regenerate/migrate it or explicitly allow unverified legacy replay."
            ) from exc
        missing = missing_legacy_fields(metadata)
        recorded = metadata.get(REPLAY_VERIFICATION_KEY)
        if isinstance(recorded, Mapping):
            recorded_missing = recorded.get("missing_fields", ())
            if isinstance(recorded_missing, Sequence) and not isinstance(recorded_missing, str):
                missing = tuple(sorted(set(missing) | {str(item) for item in recorded_missing}))
            recorded_count = recorded.get("fallback_mask_count")
            if isinstance(recorded_count, int) and not isinstance(recorded_count, bool):
                fallback_mask_count = max(fallback_mask_count, recorded_count)
        return ReplayValidation(
            status="unverified_legacy",
            contract_digest=None,
            missing_fields=missing,
            fallback_mask_count=fallback_mask_count,
        )

    if expected is not None:
        paths = tuple(
            path
            for path in (
                contract.semantic_dict().keys() if exact else REPLAY_LOAD_COMPATIBILITY_PATHS
            )
        )
        if exact:
            if contract.digest != expected.digest:
                raise RuntimeError(
                    f"Replay contract mismatch for {db_path}: existing digest "
                    f"{contract.digest} != expected {expected.digest}"
                )
        else:
            actual_payload = contract.semantic_dict()
            expected_payload = expected.semantic_dict()
            for path in paths:
                actual = _value_at_path(actual_payload, path)
                wanted = _value_at_path(expected_payload, path)
                if actual != wanted:
                    raise RuntimeError(
                        f"Replay contract mismatch for {db_path}: {path}={actual!r} "
                        f"does not match expected {wanted!r}"
                    )
    if fallback_mask_count:
        raise RuntimeError(
            f"Verified replay {db_path} contains {fallback_mask_count} nonterminal "
            "row(s) without an exact next-action mask"
        )
    return ReplayValidation("verified", contract.digest, (), 0)


def unverified_legacy_metadata(
    metadata: Mapping[str, Any],
    *,
    fallback_mask_count: int,
    asserted_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build permanent legacy lineage without promoting assertions to verification."""
    missing = missing_legacy_fields(metadata)
    facts = {} if asserted_facts is None else dict(asserted_facts)
    canonical_digest(facts)
    return {
        REPLAY_VERIFICATION_KEY: ReplayValidation(
            status="unverified_legacy",
            contract_digest=None,
            missing_fields=missing,
            fallback_mask_count=fallback_mask_count,
        ).to_metadata(),
        "replay.legacy_asserted_facts": facts,
        "replay.legacy_asserted_facts_digest": canonical_digest(facts),
    }
