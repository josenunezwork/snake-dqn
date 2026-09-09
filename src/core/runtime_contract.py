"""Stable, serializable contracts shared by training, evaluation, and serving.

These value objects describe a run without selecting an implementation.  They
are deliberately independent of PyTorch and the game engine so a checkpoint,
replay and evaluation receipt can agree on their semantics before a loader is
allowed to consume them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


# These are deliberately closed source-runtime values.  Training lifecycle
# descriptors map their reset implementation onto this existing six-field
# runtime contract; adding a field here would change historic digests.
RESET_STRATEGY_BATCH_EPISODE = "batch_episode"
RESET_STRATEGY_PER_ENV_ROLLOUT_BOUNDARY = "per_env_rollout_boundary"
PQN_TRAIN_RESET_STRATEGIES = frozenset(
    {RESET_STRATEGY_BATCH_EPISODE, RESET_STRATEGY_PER_ENV_ROLLOUT_BOUNDARY}
)


def _canonical_value(value: Any) -> Any:
    """Validate values accepted in a semantic digest and normalize mappings."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Contract values must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Contract mapping keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    raise ValueError(f"Unsupported contract value {type(value).__name__}")


def canonical_digest(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for finite semantic data.

    Mapping order is intentionally erased while every list/channel order remains
    part of the identity.
    """
    payload = json.dumps(
        _canonical_value(value), separators=(",", ":"), ensure_ascii=True, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ObservationContract:
    """Complete ordered observation semantics for a named specification."""

    obs_spec: str
    tactical_shape: tuple[int, int, int] | None = None
    strategic_shape: tuple[int, int, int] | None = None
    scalar_count: int | None = None
    tactical_channels: tuple[str, ...] = ()
    strategic_channels: tuple[str, ...] = ()
    scalars: tuple[str, ...] = ()
    coordinate_frame: str = "world"
    tactical_origin: tuple[int, int] | None = None
    strategic_origin: tuple[int, int] | None = None
    strategic_cell_size: int | None = None
    rotation: str = "none"
    paint_rule: str = "legacy"
    same_type_tie_rule: str = "legacy"
    out_of_world_prediction: str = "legacy"
    action_interpretation: str = "relative6"
    supported_arena_types: tuple[str, ...] = ("rectangular",)
    value_scales: Mapping[str, float] = field(default_factory=dict)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.obs_spec or not self.action_interpretation:
            raise ValueError("Observation contracts require named semantics")
        if self.tactical_shape is not None and (
            len(self.tactical_shape) != 3 or any(size <= 0 for size in self.tactical_shape)
        ):
            raise ValueError("tactical_shape must be (channels, height, width)")
        if self.tactical_shape is not None and self.tactical_shape[0] != len(
            self.tactical_channels
        ):
            raise ValueError("tactical_shape channels must match tactical_channels")
        if self.strategic_shape is not None and (
            len(self.strategic_shape) != 3 or any(size <= 0 for size in self.strategic_shape)
        ):
            raise ValueError("strategic_shape must be (channels, height, width)")
        if self.strategic_shape is not None and self.strategic_shape[0] != len(
            self.strategic_channels
        ):
            raise ValueError("strategic_shape channels must match strategic_channels")
        if self.scalar_count is not None and self.scalar_count != len(self.scalars):
            raise ValueError("scalar_count must match the ordered scalar descriptor")
        descriptors = (self.tactical_channels, self.strategic_channels, self.scalars)
        if any(len(set(items)) != len(items) for items in descriptors):
            raise ValueError(
                "Observation channel and scalar names must be unique per ordered input"
            )
        if self.strategic_cell_size is not None and self.strategic_cell_size <= 0:
            raise ValueError("strategic_cell_size must be positive")
        _canonical_value(dict(self.value_scales))
        object.__setattr__(self, "value_scales", MappingProxyType(dict(self.value_scales)))
        object.__setattr__(self, "digest", canonical_digest(self.semantic_dict()))

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "obs_spec": self.obs_spec,
            "tactical_shape": self.tactical_shape,
            "strategic_shape": self.strategic_shape,
            "scalar_count": self.scalar_count,
            "tactical_channels": self.tactical_channels,
            "strategic_channels": self.strategic_channels,
            "scalars": self.scalars,
            "coordinate_frame": self.coordinate_frame,
            "tactical_origin": self.tactical_origin,
            "strategic_origin": self.strategic_origin,
            "strategic_cell_size": self.strategic_cell_size,
            "rotation": self.rotation,
            "paint_rule": self.paint_rule,
            "same_type_tie_rule": self.same_type_tie_rule,
            "out_of_world_prediction": self.out_of_world_prediction,
            "action_interpretation": self.action_interpretation,
            "supported_arena_types": self.supported_arena_types,
            "value_scales": dict(self.value_scales),
        }

    def to_metadata(self) -> dict[str, Any]:
        """Return complete metadata needed to validate a corrected observation."""
        return {
            "obs_spec": self.obs_spec,
            "obs_contract": self.semantic_dict(),
            "obs_contract_digest": self.digest,
        }


@dataclass(frozen=True)
class EffectiveWorldConfig:
    """Resolved world values that affect dynamics or normalization."""

    width: int
    height: int
    segment_size: int
    wall_thickness: int
    arena_type: str
    mechanics_version: int
    num_snakes: int
    max_frames: int
    initial_food: int
    max_food: int
    min_boost_length: int
    boost_length_cost_frames: int
    frame_rate: int = 1
    max_length: int = 100
    starvation_max_frames: int = 500
    arena_radius: int = 400
    arena_center_x: int = 0
    arena_center_y: int = 0
    max_capacity: int = 400
    kill_scale: float = 0.0
    death_value: float = 0.0
    normalization: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        integer_fields = (
            self.width,
            self.height,
            self.segment_size,
            self.wall_thickness,
            self.mechanics_version,
            self.num_snakes,
            self.max_frames,
            self.initial_food,
            self.max_food,
            self.min_boost_length,
            self.boost_length_cost_frames,
            self.frame_rate,
            self.max_length,
            self.starvation_max_frames,
            self.max_capacity,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise ValueError("World configuration fields must be integers")
        if self.width <= 0 or self.height <= 0 or self.segment_size <= 0:
            raise ValueError("World dimensions and segment_size must be positive")
        if self.width % self.segment_size or self.height % self.segment_size:
            raise ValueError("World width and height must align to segment_size")
        if self.wall_thickness < 0 or self.wall_thickness % self.segment_size:
            raise ValueError("wall_thickness must align to segment_size")
        if self.arena_type not in {"rectangular", "circular"}:
            raise ValueError("Unsupported arena type")
        if min(self.num_snakes, self.max_frames, self.max_capacity) <= 0:
            raise ValueError("World population, horizon, and capacity must be positive")
        if (
            min(
                self.initial_food,
                self.max_food,
                self.min_boost_length,
                self.boost_length_cost_frames,
            )
            < 0
        ):
            raise ValueError("World food and boost settings must be non-negative")
        if self.max_food < self.initial_food:
            raise ValueError("max_food must be at least initial_food")
        if self.mechanics_version not in {1, 2}:
            raise ValueError("mechanics_version must be 1 or 2")
        if (
            min(
                self.min_boost_length,
                self.boost_length_cost_frames,
                self.frame_rate,
                self.max_length,
                self.starvation_max_frames,
            )
            <= 0
        ):
            raise ValueError("World cadence, normalization, and boost thresholds must be positive")
        if self.arena_type == "circular":
            circular_fields = (self.arena_radius, self.arena_center_x, self.arena_center_y)
            if any(
                isinstance(value, bool) or not isinstance(value, int) for value in circular_fields
            ):
                raise ValueError("Circular arena geometry must use integers")
            if self.arena_radius <= 0 or not (
                self.arena_radius <= self.arena_center_x <= self.width - self.arena_radius
                and self.arena_radius <= self.arena_center_y <= self.height - self.arena_radius
            ):
                raise ValueError("Circular arena geometry must fit within the world")
        _canonical_value({"kill_scale": self.kill_scale, "death_value": self.death_value})
        _canonical_value(dict(self.normalization))
        object.__setattr__(self, "normalization", MappingProxyType(dict(self.normalization)))

    @property
    def digest(self) -> str:
        circular_geometry = (
            {
                "arena_radius": self.arena_radius,
                "arena_center_x": self.arena_center_x,
                "arena_center_y": self.arena_center_y,
            }
            if self.arena_type == "circular"
            else None
        )
        return canonical_digest(
            {
                "width": self.width,
                "height": self.height,
                "segment_size": self.segment_size,
                "wall_thickness": self.wall_thickness,
                "arena_type": self.arena_type,
                "mechanics_version": self.mechanics_version,
                "num_snakes": self.num_snakes,
                "max_frames": self.max_frames,
                "initial_food": self.initial_food,
                "max_food": self.max_food,
                "min_boost_length": self.min_boost_length,
                "boost_length_cost_frames": self.boost_length_cost_frames,
                "frame_rate": self.frame_rate,
                "max_length": self.max_length,
                "starvation_max_frames": self.starvation_max_frames,
                "circular_geometry": circular_geometry,
                "max_capacity": self.max_capacity,
                "kill_scale": self.kill_scale,
                "death_value": self.death_value,
                "normalization": dict(self.normalization),
            }
        )


@dataclass(frozen=True)
class RuntimeModeContract:
    """Episode lifecycle switches whose changes invalidate direct comparisons."""

    mode: str
    training: bool
    respawn: bool
    hero_terminal: bool
    population_floor: bool
    reset_strategy: str = "episode"

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class ActionMaskSet:
    """Legal and advisory masks with a row-local safe resolution rule."""

    legal: np.ndarray
    advisory: np.ndarray
    dead: np.ndarray | None = None

    def __post_init__(self) -> None:
        legal = np.asarray(self.legal)
        advisory = np.asarray(self.advisory)
        if legal.shape != advisory.shape or legal.ndim < 1:
            raise ValueError(
                "legal and advisory masks must have the same shape with an action axis"
            )
        if legal.shape[-1] == 0:
            raise ValueError("action-mask action axis must be non-empty")
        if legal.dtype != np.bool_ or advisory.dtype != np.bool_:
            raise ValueError("legal and advisory masks must have boolean dtype")
        rows_shape = legal.shape[:-1]
        dead = np.zeros(rows_shape, dtype=bool) if self.dead is None else np.asarray(self.dead)
        if dead.dtype != np.bool_:
            raise ValueError("dead mask must have boolean dtype")
        if dead.shape != rows_shape:
            raise ValueError("dead must have one flag for every leading mask position")
        object.__setattr__(self, "legal", legal)
        object.__setattr__(self, "advisory", advisory)
        object.__setattr__(self, "dead", dead)

    def resolved(self) -> np.ndarray:
        """Use legal ∩ advisory per live row, falling back to legal per empty row."""
        legal = self.legal.copy()
        advisory = self.advisory
        intersection = legal & advisory
        resolved = np.where(intersection.any(axis=-1, keepdims=True), intersection, legal)
        return np.where(np.expand_dims(self.dead, axis=-1), False, resolved)


@dataclass(frozen=True)
class RunProvenance:
    """Versioned run identity persisted beside checkpoints and experiment receipts."""

    effective_seed: int
    observation_digest: str
    world_digest: str
    runtime_digest: str
    reward_digest: str
    target_digest: str
    sampler_digest: str
    optimizer_digest: str
    model_head_digest: str
    source_revision: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.effective_seed, bool)
            or not isinstance(self.effective_seed, int)
            or not 0 <= self.effective_seed < 2**64
        ):
            raise ValueError("Run provenance effective_seed must be an unsigned 64-bit integer")
        if not isinstance(self.model_head_digest, str) or not self.model_head_digest:
            raise ValueError("Run provenance model_head_digest must be a non-empty string")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError("Run provenance schema_version must be an integer")
        if self.schema_version <= 0:
            raise ValueError("Run provenance schema_version must be positive")
        if self.schema_version != 1:
            raise ValueError(f"Unsupported run provenance schema_version {self.schema_version}")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def to_metadata(self) -> dict[str, Any]:
        """Serialize a verified provenance record and its semantic digest."""
        return {
            "run_provenance": asdict(self),
            "run_provenance_digest": self.digest,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "RunProvenance":
        """Deserialize only a supported, versioned, digest-verified record."""
        raw = metadata.get("run_provenance", metadata)
        if not isinstance(raw, Mapping) or "schema_version" not in raw:
            raise ValueError("Verified run provenance requires schema_version")
        try:
            provenance = cls(**dict(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid run provenance metadata") from exc
        digest = metadata.get("run_provenance_digest")
        if not isinstance(digest, str) or digest != provenance.digest:
            raise ValueError("run_provenance_digest does not match provenance metadata")
        return provenance


@dataclass(frozen=True)
class ModelHeadContract:
    """Closed model-head descriptor, independent from observation identity."""

    algorithm: str
    head: str
    action_count: int
    critic_outputs: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.action_count, bool) or not isinstance(self.action_count, int):
            raise ValueError("action_count must be an integer")
        if isinstance(self.critic_outputs, bool) or not isinstance(self.critic_outputs, int):
            raise ValueError("critic_outputs must be an integer")
        expected = MODEL_HEAD_REGISTRY.get((self.algorithm, self.head))
        if expected is None:
            raise ValueError(f"Unknown model head {self.algorithm!r}/{self.head!r}")
        if (self.action_count, self.critic_outputs) != expected:
            raise ValueError(
                f"{self.algorithm}/{self.head} expects action_count={expected[0]}, "
                f"critic_outputs={expected[1]}"
            )

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def to_metadata(self) -> dict[str, Any]:
        return {"model_head": asdict(self), "model_head_digest": self.digest}


# Recognition is intentionally separate from serving support. S1 and L1 opt in
# to respective loaders after their implementations are qualified.
MODEL_HEAD_REGISTRY: dict[tuple[str, str], tuple[int, int]] = {
    ("apex", "dueling_q"): (6, 0),
    ("pqn", "dueling_q"): (6, 0),
    ("ppo", "categorical_actor_critic"): (6, 1),
}


def validate_model_head_contract(
    metadata: Mapping[str, Any], *, require_digest: bool = False
) -> ModelHeadContract:
    """Validate serialized model-head metadata through the one shared registry."""
    raw = metadata.get("model_head", metadata)
    if not isinstance(raw, Mapping):
        raise ValueError("model_head metadata must be a mapping")
    try:
        algorithm = raw["algorithm"]
        head = raw["head"]
        action_count = raw["action_count"]
        critic_outputs = raw.get("critic_outputs", 0)
        if not isinstance(algorithm, str) or not isinstance(head, str):
            raise ValueError("model_head algorithm and head must be strings")
        contract = ModelHeadContract(
            algorithm=algorithm,
            head=head,
            action_count=action_count,
            critic_outputs=critic_outputs,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid model_head metadata") from exc
    recorded_digest = metadata.get("model_head_digest")
    if require_digest and not isinstance(recorded_digest, str):
        raise ValueError("model_head_digest is required")
    if recorded_digest is not None and recorded_digest != contract.digest:
        raise ValueError("model_head_digest does not match model_head metadata")
    return contract
