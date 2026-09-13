"""Immutable, separately digested storage choices for evaluation worlds.

``EffectiveWorldConfig.max_capacity`` remains a source-world fact: it is part
of checkpoint and evaluation-profile identity.  This module names a *runtime*
body-storage allocation separately, so a finite-horizon, fresh-reset SIMD
evaluation can allocate enough ring-buffer space without rewriting history.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping

from src.core.runtime_contract import canonical_digest

if TYPE_CHECKING:
    from src.evaluation.protocol import EvaluationProfile


WORLD_RUNTIME_SCHEMA_V1 = "world-runtime/v1"
ENGINE_SIMD = "simd"
STORAGE_FIXED_RING = "fixed_ring"
CAPACITY_POLICY_SOURCE_EXACT_V1 = "source_exact_v1"
CAPACITY_POLICY_FRESH_RESET_HORIZON_BOUND_V1 = "fresh_reset_horizon_bound_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class WorldRuntimeSpec:
    """Closed runtime body-storage contract for one evaluation execution.

    The descriptor is deliberately separate from checkpoint provenance and
    :class:`EvaluationProfile` identity.  Call :meth:`validate_for_profile`
    before allocating a simulator.
    """

    engine: str
    source_world_digest: str
    body_storage_model: str
    source_body_capacity: int
    evaluation_profile_digest: str
    body_storage_capacity: int
    capacity_policy: str
    scored_horizon: int
    schema_version: str = WORLD_RUNTIME_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != WORLD_RUNTIME_SCHEMA_V1:
            raise ValueError(f"unsupported world runtime schema {self.schema_version!r}")
        if self.engine != ENGINE_SIMD:
            raise ValueError(f"unsupported runtime engine {self.engine!r}")
        if not isinstance(self.source_world_digest, str) or not _SHA256.fullmatch(
            self.source_world_digest
        ):
            raise ValueError("source_world_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.evaluation_profile_digest, str) or not _SHA256.fullmatch(
            self.evaluation_profile_digest
        ):
            raise ValueError("evaluation_profile_digest must be a lowercase SHA-256 digest")
        _positive_int(self.scored_horizon, "scored_horizon")

        if self.body_storage_model != STORAGE_FIXED_RING:
            raise ValueError("SIMD capacity policies require fixed_ring storage")
        source = _positive_int(self.source_body_capacity, "source_body_capacity")
        effective = _positive_int(self.body_storage_capacity, "body_storage_capacity")
        if effective < source:
            raise ValueError("body_storage_capacity must not downgrade source_body_capacity")
        if self.capacity_policy == CAPACITY_POLICY_SOURCE_EXACT_V1:
            if effective != source:
                raise ValueError(
                    "source_exact_v1 requires body_storage_capacity == source capacity"
                )
        elif self.capacity_policy == CAPACITY_POLICY_FRESH_RESET_HORIZON_BOUND_V1:
            expected = max(source, self.scored_horizon + 2)
            if effective != expected:
                raise ValueError("horizon-bound capacity must equal its source and horizon bound")
        else:
            raise ValueError(f"unsupported capacity policy {self.capacity_policy!r}")

    @classmethod
    def from_profile(
        cls,
        profile: "EvaluationProfile",
        *,
        capacity_policy: str = CAPACITY_POLICY_SOURCE_EXACT_V1,
    ) -> "WorldRuntimeSpec":
        """Build one closed runtime choice from an already frozen profile."""
        cls._require_profile(profile)
        source = profile.world.max_capacity
        if capacity_policy == CAPACITY_POLICY_SOURCE_EXACT_V1:
            effective = source
        elif capacity_policy == CAPACITY_POLICY_FRESH_RESET_HORIZON_BOUND_V1:
            if profile.legacy_diagnostic or profile.runtime.reset_strategy != "manual":
                raise ValueError(
                    "horizon-bound SIMD storage requires a non-legacy fresh-reset profile"
                )
            effective = max(source, profile.scored_horizon + 2)
        else:
            raise ValueError(f"unsupported capacity policy {capacity_policy!r}")
        return cls(
            engine=ENGINE_SIMD,
            evaluation_profile_digest=profile.digest,
            source_world_digest=profile.world.digest,
            body_storage_model=STORAGE_FIXED_RING,
            source_body_capacity=source,
            body_storage_capacity=effective,
            capacity_policy=capacity_policy,
            scored_horizon=profile.scored_horizon,
        )

    @classmethod
    def source_exact(cls, profile: "EvaluationProfile") -> "WorldRuntimeSpec":
        """Bind source-ring allocation exactly to the frozen profile."""
        return cls.from_profile(profile, capacity_policy=CAPACITY_POLICY_SOURCE_EXACT_V1)

    @classmethod
    def fresh_reset_horizon_bound(cls, profile: "EvaluationProfile") -> "WorldRuntimeSpec":
        """Allocate the proven fresh-reset bound, ``max(source, horizon + 2)``."""
        return cls.from_profile(
            profile, capacity_policy=CAPACITY_POLICY_FRESH_RESET_HORIZON_BOUND_V1
        )

    @staticmethod
    def _require_profile(profile: object) -> None:
        required = ("world", "runtime", "scored_horizon", "legacy_diagnostic")
        if any(not hasattr(profile, name) for name in required):
            raise ValueError("world runtime spec requires an EvaluationProfile")

    def validate_for_profile(self, profile: "EvaluationProfile") -> None:
        """Reject profile drift before policies or a simulator are allocated."""
        self._require_profile(profile)
        if self.evaluation_profile_digest != profile.digest:
            raise ValueError("world runtime profile digest does not match evaluation profile")
        if self.source_world_digest != profile.world.digest:
            raise ValueError("world runtime source digest does not match evaluation profile")
        if self.scored_horizon != profile.scored_horizon:
            raise ValueError("world runtime scored horizon does not match evaluation profile")
        if self.source_body_capacity != profile.world.max_capacity:
            raise ValueError("world runtime source capacity does not match evaluation profile")
        if self.capacity_policy == CAPACITY_POLICY_FRESH_RESET_HORIZON_BOUND_V1:
            if profile.legacy_diagnostic or profile.runtime.reset_strategy != "manual":
                raise ValueError(
                    "horizon-bound SIMD storage requires a non-legacy fresh-reset profile"
                )
            expected = max(profile.world.max_capacity, profile.scored_horizon + 2)
            if self.body_storage_capacity != expected:
                raise ValueError("horizon-bound effective capacity does not match profile bound")

    def descriptor(self) -> dict[str, Any]:
        """Return the exact JSON-ready storage identity, excluding its digest."""
        return asdict(self)

    @property
    def digest(self) -> str:
        """Return the stable digest of this detached runtime identity."""
        return canonical_digest(self.descriptor())

    @classmethod
    def from_descriptor(
        cls, raw: Mapping[str, Any], *, expected_digest: str | None = None
    ) -> "WorldRuntimeSpec":
        """Rebuild an exact descriptor and optionally verify its detached digest."""
        if not isinstance(raw, Mapping):
            raise ValueError("world runtime descriptor must be a mapping")
        expected = {
            "engine",
            "evaluation_profile_digest",
            "source_world_digest",
            "body_storage_model",
            "source_body_capacity",
            "body_storage_capacity",
            "capacity_policy",
            "scored_horizon",
            "schema_version",
        }
        if set(raw) != expected:
            missing = sorted(expected - set(raw))
            extra = sorted(set(raw) - expected)
            raise ValueError(f"world runtime descriptor has missing={missing!r}, extra={extra!r}")
        spec = cls(**dict(raw))
        if expected_digest is not None:
            if not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest):
                raise ValueError("world runtime expected digest must be a lowercase SHA-256 digest")
            if spec.digest != expected_digest:
                raise ValueError("world runtime descriptor digest does not match detached content")
        return spec
