"""Frozen pilot allocation; development modes are separate scientific namespaces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from research.sandbox_a_resident.ct_mechanics import derive_63bit_key
from research.sandbox_a_resident.training_rosters import training_rosters
from src.simd_env.batch_sim import BatchSimConfig

NAMESPACE = "t34-spatial-student-20261030/v1"


@dataclass(frozen=True)
class PilotConfig:
    """Science values are immutable outside explicitly separated tiny fixtures."""

    namespace: str = NAMESPACE
    allocation_index: int = 0
    seed: int = 1901
    worlds: int = 1024
    horizon: int = 5000
    body_capacity: int = 10400
    branch_horizon: int = 300
    branch_batch_size: int = 1024
    world_batch_size: int = 1024
    collection_unit_worlds: int = 128
    collection_checkpoint_every: int = 500
    gameplay_worlds: int = 192
    demos: int = 32768
    train_late: int = 512
    train_ordinary: int = 512
    assessment_late: int = 128
    assessment_ordinary: int = 256
    bc_updates: int = 4000
    improvement_updates: int = 4000
    batch_size: int = 64
    microbatch: int = 8
    checkpoint_every: int = 250
    search_chunk: int = 256
    min_decisions: int = 1048576
    selection_suffixes: tuple[int, ...] = (0, 1)
    scoring_suffixes: tuple[int, ...] = (4, 5)
    tiny: bool = False
    mode: str = "pilot"

    def __post_init__(self) -> None:
        if self.mode not in {"pilot", "ruler", "preflight"}:
            raise ValueError("foreign pilot mode")
        expected = NAMESPACE if self.mode == "pilot" else f"{NAMESPACE}/{self.mode}"
        if self.namespace != expected or type(self.allocation_index) is not int:
            raise ValueError("foreign namespace/allocation")
        if self.allocation_index != 0:
            raise ValueError("only frozen allocation zero is registered")
        for name in ("worlds", "horizon", "world_batch_size", "branch_batch_size", "microbatch"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if self.mode == "pilot" and self != PilotConfig.__new_defaults__():
            raise ValueError("scientific config differs from registration")
        if self.tiny and self.mode != "preflight":
            raise ValueError("tiny configuration is Mac preflight only")

    @classmethod
    def __new_defaults__(cls) -> "PilotConfig":
        # Avoid recursively constructing the reference during __post_init__.
        from dataclasses import fields

        obj = object.__new__(cls)
        for field in fields(cls):
            object.__setattr__(obj, field.name, field.default)
        return obj

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PilotConfig":
        values = dict(payload)
        for key in ("selection_suffixes", "scoring_suffixes"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def development(self, mode: str) -> "PilotConfig":
        if mode == "ruler":
            return replace(self, mode=mode, namespace=f"{NAMESPACE}/{mode}")
        if mode != "preflight":
            raise ValueError("unknown development mode")
        return replace(
            self,
            mode=mode,
            namespace=f"{NAMESPACE}/{mode}",
            tiny=True,
            worlds=6,
            horizon=6,
            body_capacity=64,
            branch_horizon=2,
            branch_batch_size=12,
            world_batch_size=6,
            collection_unit_worlds=3,
            collection_checkpoint_every=2,
            gameplay_worlds=3,
            demos=6,
            train_late=1,
            train_ordinary=2,
            assessment_late=1,
            assessment_ordinary=2,
            bc_updates=1,
            improvement_updates=1,
            batch_size=2,
            microbatch=1,
            checkpoint_every=1,
            search_chunk=2,
            min_decisions=1,
        )

    def sim_config(self, worlds: int) -> BatchSimConfig:
        return BatchSimConfig(
            num_envs=worlds,
            num_snakes=6,
            max_capacity=self.body_capacity,
            body_storage_capacity=self.body_capacity,
            mechanics_version=1,
            **(
                dict(game_width=240, game_height=240, initial_food=10, max_food=18)
                if self.tiny
                else {}
            ),
        )

    def world_seed(self, world_id: int, phase: str = "collection") -> int:
        if phase not in {"collection", "gameplay"} or world_id < 0:
            raise ValueError("foreign seed domain")
        return derive_63bit_key(self.namespace, self.allocation_index, phase, world_id)

    def opp_seed(self, world_id: int, phase: str = "collection") -> int:
        return derive_63bit_key(self.namespace, self.allocation_index, phase, "opponent", world_id)

    def partitions(self) -> tuple[str, ...]:
        ranks: dict[str, int] = {}
        out = []
        rosters = training_rosters(self.worlds)
        for roster in rosters:
            rank = ranks.get(roster.mix, 0)
            out.append("assessment" if (rank + 1) % 4 == 0 else "train")
            ranks[roster.mix] = rank + 1
        # Three rounded per-mix counts sum to 255 at N1024. Assign the residual
        # to the last world of the largest mix; never choose based on outcomes.
        target = max(1, self.worlds // 4)
        for idx in range(len(out) - 1, -1, -1):
            if out.count("assessment") >= target:
                break
            if out[idx] == "train":
                out[idx] = "assessment"
        return tuple(out)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def audit_seeds(repo: Path, cfg: PilotConfig | None = None) -> dict[str, Any]:
    """Audit complete effective allocations and retained explicit seed inventories.

    Namespace isolation is necessary but not a collision proof. Scan JSON seed
    inventory fields in the repository; never inspect outcomes or select seeds.
    """
    cfg = cfg or PilotConfig()
    keys = []
    for mode in ("pilot", "ruler", "preflight"):
        current = cfg if mode == "pilot" else cfg.development(mode)
        for phase, count in (("collection", current.worlds), ("gameplay", current.gameplay_worlds)):
            keys.extend(current.world_seed(i, phase) for i in range(count))
            keys.extend(current.opp_seed(i, phase) for i in range(count))
    if len(keys) != len(set(keys)):
        raise ValueError("within-allocation effective seed collision")
    # The existing protected registry is the maintained authoritative source.
    from research.sandbox_a_resident.ct_task import train01_protected_eval_seeds

    protected = {int(x) for x in train01_protected_eval_seeds()}
    registry_path = repo / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json"
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != "a100-sandbox-input-registry/v1":
        raise ValueError("foreign exclusion registry")
    protected.update(int(seed) & 0x7FFF_FFFF_FFFF_FFFF for seed in registry["excluded_seeds"])
    inventories = []
    for base in (repo / "research", repo / "docs"):
        for path in sorted(base.rglob("*seed*.json")):
            if "spatial_student_pilot" in path.parts or path.stat().st_size > 2_000_000:
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue

            def visit(value: Any, field: str = "") -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        visit(item, key)
                elif isinstance(value, list):
                    for item in value:
                        visit(item, field)
                elif type(value) is int and "seed" in field.lower():
                    protected.add(value & 0x7FFF_FFFF_FFFF_FFFF)

            visit(data)
            inventories.append(str(path.relative_to(repo)))
    if set(keys) & protected:
        raise ValueError("protected effective seed collision; reject allocation")
    return {
        "schema": "spatial-seed-audit/v1",
        "namespace": cfg.namespace,
        "allocation_index": cfg.allocation_index,
        "config_sha256": cfg.sha256,
        "effective_keys": keys,
        "protected_count": len(protected),
        "inventories": inventories,
        "collision_count": 0,
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "scope": (
            "repository JSON inventories, canonical exclusion registry and TRAIN01 protected keys; "
            "not an exhaustive external historical inventory"
        ),
    }
