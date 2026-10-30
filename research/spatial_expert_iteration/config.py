"""Round allocation and learning recipe, frozen before any round data."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from research.sandbox_a_resident.ct_mechanics import derive_63bit_key
from research.spatial_student_pilot.config import PilotConfig

DOMAIN = "t34-spatial-expert-iteration/v1"


def identity(value: Any) -> str:
    """Canonical JSON identity, rejecting nonfinite registration values."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RoundConfig:
    """Defaults are a proposal, never a ratified scientific allocation.

    UnitStore uses this structural interface to preserve the pilot's publication,
    accounting, retry and full-state trajectory rules without editing the pilot.
    """

    program: str = "candidate-01"
    implementation_sha256: str = ""
    round_index: int = 1
    allocation_index: int = 0
    tiny: bool = False
    seed: int = 1901
    worlds: int = 1024
    horizon: int = 5000
    anchor_worlds: int = 0
    anchor_horizon: int = 512
    body_capacity: int = 10400
    world_batch_size: int = 1024
    logical_reservoir_size: int = 128
    branch_horizon: int = 300
    branch_batch_size: int = 1024
    search_chunk: int = 128
    candidate_roots: int = 2048
    reservoir_roots: int = 0
    reservoir_max_roots: int = 8192
    reservoir_max_bytes: int = 512 * 1024**2
    min_mature_roots: int = 512
    min_mature_parents: int = 256
    min_assessment_mature_parents: int = 64
    demos: int = 32768
    roots_per_parent: int = 2
    root_every: int = 32
    demo_every: int = 16
    mature_length: int = 600
    assessment_worlds: int = 192
    assessment_roots: int = 128
    gameplay_worlds: int = 192
    updates: int = 2000
    batch_size: int = 64
    microbatch: int = 8
    checkpoint_every: int = 250
    query_top_fraction: float = 0.25
    query_floor: float = 0.125
    soft_weight: float = 0.0
    advantage_weight: float = 0.0
    temperature: float = 1.0
    observation_cache: str = "sparse"
    cache_max_bytes: int = 8 * 1024**3
    bc_weight: float = 1.0
    base_timeout_seconds: float = 10800.0
    timeout_seconds: float = 21600.0
    selection_suffixes: tuple[int, ...] = (0, 1)
    scoring_suffixes: tuple[int, ...] = (4, 5)

    def __post_init__(self) -> None:
        import re

        if not isinstance(self.implementation_sha256, str) or (
            self.implementation_sha256
            and not re.fullmatch(r"[0-9a-f]{64}", self.implementation_sha256)
        ):
            raise ValueError("invalid source closure identity")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", self.program):
            raise ValueError("invalid program identity")
        for key in (
            "round_index",
            "worlds",
            "horizon",
            "anchor_horizon",
            "body_capacity",
            "world_batch_size",
            "logical_reservoir_size",
            "branch_horizon",
            "branch_batch_size",
            "search_chunk",
            "candidate_roots",
            "reservoir_max_roots",
            "reservoir_max_bytes",
            "demos",
            "roots_per_parent",
            "root_every",
            "demo_every",
            "mature_length",
            "assessment_worlds",
            "assessment_roots",
            "min_mature_roots",
            "min_mature_parents",
            "min_assessment_mature_parents",
            "gameplay_worlds",
            "updates",
            "batch_size",
            "microbatch",
            "checkpoint_every",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"positive integer required: {key}")
        for key in ("allocation_index", "anchor_worlds", "reservoir_roots", "seed"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"nonnegative integer required: {key}")
        if (
            type(self.tiny) is not bool
            or self.anchor_worlds > self.worlds
            or self.assessment_worlds > self.worlds
        ):
            raise ValueError("invalid tiny/anchor allocation")
        if (
            self.observation_cache not in {"sparse", "integer"}
            or type(self.cache_max_bytes) is not int
            or self.cache_max_bytes <= 0
        ):
            raise ValueError("invalid frozen cache mode/byte cap")
        if self.gameplay_worlds < 3:
            raise ValueError("paired gameplay requires all three roster mixes")
        if (
            self.world_batch_size % self.logical_reservoir_size
            or self.anchor_worlds % self.logical_reservoir_size
        ):
            raise ValueError("physics/anchor batches must preserve whole logical reservoirs")
        if self.reservoir_roots > self.candidate_roots:
            raise ValueError("reservoir quota exceeds candidate population")
        if self.batch_size % 2 or self.microbatch > self.batch_size:
            raise ValueError("even batch and bounded microbatch required")
        for key in (
            "query_top_fraction",
            "query_floor",
            "soft_weight",
            "advantage_weight",
            "temperature",
            "bc_weight",
            "base_timeout_seconds",
            "timeout_seconds",
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"finite scalar required: {key}")
        if not 0 <= self.query_top_fraction <= 1 or not 0 < self.query_floor <= 1:
            raise ValueError("query gate requires a positive random floor")
        if (
            min(self.soft_weight, self.advantage_weight) < 0
            or min(self.bc_weight, self.temperature) <= 0
        ):
            raise ValueError("invalid learning weights/temperature")
        if self.base_timeout_seconds <= 0 or self.timeout_seconds != 2 * self.base_timeout_seconds:
            raise ValueError("freeze base and effective 2x timeout")
        if self.selection_suffixes != (0, 1) or self.scoring_suffixes != (4, 5):
            raise ValueError("selection/scoring suffixes must remain independent")

    @property
    def namespace(self) -> str:
        mode = "smoke" if self.tiny else "registered"
        return f"{DOMAIN}/{self.program}/{mode}/round-{self.round_index:04d}"

    @property
    def mode(self) -> str:
        return "preflight" if self.tiny else "pilot"

    @property
    def sha256(self) -> str:
        return identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoundConfig":
        values = dict(value)
        for key in ("selection_suffixes", "scoring_suffixes"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    # Pilot unit path capacity interface. Assessment collection shares collect
    # units; the last collect unit is the fresh assessment population.
    @property
    def train_late(self) -> int:
        return self.candidate_roots

    @property
    def train_ordinary(self) -> int:
        return 0

    @property
    def assessment_late(self) -> int:
        return self.assessment_roots

    @property
    def assessment_ordinary(self) -> int:
        return 0

    @property
    def bc_updates(self) -> int:
        return 0

    @property
    def improvement_updates(self) -> int:
        return self.updates

    def sim_config(self, worlds: int) -> Any:
        from dataclasses import replace

        base = PilotConfig().development("preflight") if self.tiny else PilotConfig()
        return replace(
            base.sim_config(worlds),
            max_capacity=self.body_capacity,
            body_storage_capacity=self.body_capacity,
        )

    def world_seed(self, world_id: int, phase: str = "collection") -> int:
        return derive_63bit_key(self.namespace, self.allocation_index, phase, world_id)

    def opp_seed(self, world_id: int, phase: str = "collection") -> int:
        return derive_63bit_key(self.namespace, self.allocation_index, phase, "opponent", world_id)

    @classmethod
    def smoke(cls) -> "RoundConfig":
        return cls(
            tiny=True,
            worlds=4,
            horizon=4,
            anchor_worlds=0,
            anchor_horizon=2,
            body_capacity=64,
            world_batch_size=4,
            logical_reservoir_size=2,
            branch_horizon=2,
            branch_batch_size=24,
            search_chunk=2,
            candidate_roots=2,
            reservoir_roots=0,
            min_mature_roots=1,
            min_mature_parents=1,
            min_assessment_mature_parents=1,
            demos=4,
            roots_per_parent=1,
            root_every=1,
            demo_every=1,
            mature_length=1,
            assessment_worlds=2,
            assessment_roots=2,
            gameplay_worlds=3,
            updates=2,
            batch_size=2,
            microbatch=1,
            checkpoint_every=1,
            query_top_fraction=1.0,
            base_timeout_seconds=60,
            timeout_seconds=120,
        )
