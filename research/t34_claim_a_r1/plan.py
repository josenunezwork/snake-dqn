"""The frozen R1 labels-only plan: 800 training + 200 whole-world held-out worlds.

Shape (JUDGMENT §7c, amendment spec 2026-10-21): bare B8 s3 + teacher_v2, beta 1,
confirm interval 8, ratified v2 shadow-LAD kernel, gate-matched 50/25/25 mixes.
Training and held-out worlds come from *different* namespaces, so the split is
fixed by seed allocation before any label exists (never chosen on label content).

Chunks are 100 worlds (about 1.75 h of steady L4 labelling at the A5 rate of
57 worlds/h on K=12 processes, inside the ratified 170-minute label wall).  Each
chunk keeps the ratified label manifest's own per-job wall/unit bound; the lane
cap is enforced across chunks by :func:`admissible`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

LANE_CAP_UNITS = 60.0
CHUNK_WORLDS = 100
TRAIN_SHARDS = 8  # 800 worlds
HELDOUT_SHARDS = 2  # 200 worlds
SPLITS = {"train": TRAIN_SHARDS, "heldout": HELDOUT_SHARDS}
AUDITED_ATTEMPTS = (0, 1)  # a1 only for a judge-approved mechanical rerun
# Held-out support minimum (JUDGMENT §7c); below it R1 is PRECONDITION_FAILED.
MIN_HELDOUT_DECISIVE_WORLDS = 60
MIN_HELDOUT_DECISIVE_ROWS = 300


@dataclass(frozen=True)
class Chunk:
    split: str
    shard: int
    attempt: int = 0

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"unknown R1 split {self.split!r}")
        if not 0 <= self.shard < SPLITS[self.split]:
            raise ValueError(f"{self.split} shard {self.shard} outside the frozen plan")
        if self.attempt not in AUDITED_ATTEMPTS:
            raise ValueError("attempt namespace is not seed-audited")

    @property
    def namespace(self) -> str:
        return f"t34-claim-a/r1-labels-v4/{self.split}/s{self.shard}/a{self.attempt}"

    @property
    def job_id(self) -> str:
        return f"t34-claim-a-r1-{self.split}-s{self.shard}-a{self.attempt}"

    @property
    def worlds(self) -> int:
        return CHUNK_WORLDS


def plan(attempt: int = 0) -> list[Chunk]:
    return [Chunk(split, shard, attempt) for split, n in SPLITS.items() for shard in range(n)]


def audited_namespaces() -> dict[str, int]:
    return {c.namespace: c.worlds for a in AUDITED_ATTEMPTS for c in plan(a)}


def admissible(unit_cap: float, settled_units: Iterable[float], open_caps: Iterable[float]) -> bool:
    """Admit a chunk only if settled spend + every open chunk's cap + this cap <= lane cap."""
    return sum(settled_units) + sum(open_caps) + float(unit_cap) <= LANE_CAP_UNITS + 1e-9


def heldout_support(per_world_decisive: Mapping[str, int]) -> dict[str, object]:
    """Held-out support check on sealed held-out worlds (decisive = horizon_rescue)."""
    bearing = sum(1 for n in per_world_decisive.values() if n > 0)
    rows = sum(per_world_decisive.values())
    ok = bearing >= MIN_HELDOUT_DECISIVE_WORLDS and rows >= MIN_HELDOUT_DECISIVE_ROWS
    return {
        "decisive_bearing_worlds": bearing,
        "decisive_rows": rows,
        "min_worlds": MIN_HELDOUT_DECISIVE_WORLDS,
        "min_rows": MIN_HELDOUT_DECISIVE_ROWS,
        "status": "SUPPORTED" if ok else "PRECONDITION_FAILED",
    }
