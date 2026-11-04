"""Protocol constants and deterministic Watch6-v2 world allocation for Stage 0 probe."""

from __future__ import annotations

import hashlib
from typing import Any

NAMESPACE = "t34-hero-probe-20261103-v1"
COUNT = 48
HORIZON = 6000
LOOKBACK_FRAMES = 300

CAUSE_CLASSES: tuple[str, ...] = (
    "self-body enclosure",
    "opponent head-on",
    "opponent body",
    "wall",
)

SIM_CODE_TO_CAUSE_CLASS: dict[int, str] = {
    1: "wall",
    2: "self-body enclosure",
    3: "opponent head-on",
    4: "opponent body",
}

LIVE_CAUSE_TO_CAUSE_CLASS: dict[str, str] = {
    "wall": "wall",
    "self": "self-body enclosure",
    "head_on": "opponent head-on",
    "enemy_body": "opponent body",
    "body": "opponent body",
}


def seed_for(namespace: str, index: int) -> int:
    """Derive a deterministic 32-bit unsigned seed from ``namespace`` and ``index``."""
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if type(index) is not int or index < 0:
        raise ValueError("index must be a non-negative int")
    digest = hashlib.sha256(f"{namespace}/{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def allocation(namespace: str = NAMESPACE, count: int = COUNT) -> dict[str, Any]:
    """Return the 48-world Watch6-v2 H=6,000 probe allocation contract."""
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive int")
    worlds = [
        {
            "world_index": idx,
            "seed": seed_for(namespace, idx),
            "horizon": HORIZON,
        }
        for idx in range(count)
    ]
    seeds = [w["seed"] for w in worlds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("internal seed collision in hero_stage0 allocation")
    seeds_digest = hashlib.sha256(",".join(str(s) for s in seeds).encode("utf-8")).hexdigest()
    return {
        "schema": "t34-hero-probe-allocation/v1",
        "namespace": namespace,
        "count": count,
        "horizon": HORIZON,
        "lookback_frames": LOOKBACK_FRAMES,
        "training_excluded": True,
        "world_contract": {
            "constructor": "research.watch_replay.product.open_watch",
            "mechanics_version": 2,
            "continuous_respawn": True,
            "hero_slot": 0,
            "hero_checkpoint": "b8-s3",
            "opponent_slots": [1, 2, 3, 4, 5],
            "opponent_checkpoint": "frp3-s12",
            "opponent_veto": "v8",
            "opponent_veto_lambda": 8.0,
            "live_grid_mirror_audited": True,
        },
        "worlds": worlds,
        "seeds": seeds,
        "seeds_sha256": seeds_digest,
    }


def training_exclusions(namespace: str = NAMESPACE, count: int = COUNT) -> list[int]:
    """Return the sorted list of unique integer probe seeds excluded from all training."""
    seeds = allocation(namespace, count)["seeds"]
    return sorted(int(s) for s in seeds)
