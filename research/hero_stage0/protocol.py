"""Protocol constants and deterministic Watch6-v2 world allocation for Stage 0 probe."""

from __future__ import annotations

import hashlib
from typing import Any

NAMESPACE = "t34-hero-probe-20261103-v1"
COUNT = 48
HORIZON = 6000
LOOKBACK_FRAMES = 300

STAGE1A_CONTROLLER = "exact-stop-h48-nb"
STAGE1A_ARMS: tuple[str, ...] = ("bare", STAGE1A_CONTROLLER)
STAGE1A_BRANCH_HORIZON = 48
STAGE1A_MIN_LENGTH = 1
STAGE1A_INTERVAL = 5
STAGE1A_SELECTION_SUFFIXES: tuple[int, ...] = (0, 1)
STAGE1A_SCORING_SUFFIXES: tuple[int, ...] = (4, 5, 6, 7)

STAGE1A_BAR_L_LT_600_DEATHS_REMOVED_PCT_MIN = 70.0
STAGE1A_BAR_BOOST_SHARE_DELTA_PP_MAX = 0.5
STAGE1A_BAR_HARMFUL_OVERRIDE_PCT_MAX = 1.0

OPENSTOP_CONTROLLER = "open-stop-h32-nb"
OPENSTOP_K1_ARM = "open-stop-h32-nb-k1"
OPENSTOP_K2_ARM = "open-stop-h32-nb-k2"
OPENSTOP_ARMS: tuple[str, ...] = ("bare", OPENSTOP_K1_ARM, OPENSTOP_K2_ARM)
OPENSTOP_BRANCH_HORIZON = 32
OPENSTOP_MIN_LENGTH = 1
OPENSTOP_INTERVAL = 5
OPENSTOP_K1_SELECTION_SUFFIXES: tuple[int, ...] = (0,)
OPENSTOP_K2_SELECTION_SUFFIXES: tuple[int, ...] = (0, 1)
OPENSTOP_SCORING_SUFFIXES: tuple[int, ...] = (4, 5, 6, 7)
OPENSTOP_REGION_FRAC_MIN = 0.25
OPENSTOP_POCKET_MIN_LENGTH = 160
OPENSTOP_V8_NEED_CAP = 160
OPENSTOP_V8_NEED_FLOOR = 4

OPENSTOP_DEV_BAR_OVERRIDE_RATE_PCT_MAX = 0.5
OPENSTOP_DEV_BAR_SELF_ENCLOSURE_REMOVED_PCT_MIN = 55.0
OPENSTOP_DEV_BAR_L_LT_600_DEATHS_REMOVED_PCT_MIN = 70.0

STAGE1A_ARM_SPECS: dict[str, dict[str, Any]] = {
    "bare": {
        "min_length": None,
        "interval": STAGE1A_INTERVAL,
        "branch_horizon": None,
        "selection_suffixes": (),
        "scoring_suffixes": (),
        "prefer_nonboost": False,
        "require_terminal_open": False,
        "label": "bare (b8-s3)",
    },
    STAGE1A_CONTROLLER: {
        "min_length": STAGE1A_MIN_LENGTH,
        "interval": STAGE1A_INTERVAL,
        "branch_horizon": STAGE1A_BRANCH_HORIZON,
        "selection_suffixes": STAGE1A_SELECTION_SUFFIXES,
        "scoring_suffixes": STAGE1A_SCORING_SUFFIXES,
        "prefer_nonboost": True,
        "require_terminal_open": False,
        "label": "exact-stop-h48-nb (L>=1, interval=5, H=48, K=2 sel, non-boost first)",
    },
    OPENSTOP_K1_ARM: {
        "min_length": OPENSTOP_MIN_LENGTH,
        "interval": OPENSTOP_INTERVAL,
        "branch_horizon": OPENSTOP_BRANCH_HORIZON,
        "selection_suffixes": OPENSTOP_K1_SELECTION_SUFFIXES,
        "scoring_suffixes": OPENSTOP_SCORING_SUFFIXES,
        "prefer_nonboost": True,
        "require_terminal_open": True,
        "label": (
            "open-stop-h32-nb K=1 "
            "(L>=1, interval=5, H=32, K=1 sel, alive+un-enclosed, non-boost first)"
        ),
    },
    OPENSTOP_K2_ARM: {
        "min_length": OPENSTOP_MIN_LENGTH,
        "interval": OPENSTOP_INTERVAL,
        "branch_horizon": OPENSTOP_BRANCH_HORIZON,
        "selection_suffixes": OPENSTOP_K2_SELECTION_SUFFIXES,
        "scoring_suffixes": OPENSTOP_SCORING_SUFFIXES,
        "prefer_nonboost": True,
        "require_terminal_open": True,
        "label": (
            "open-stop-h32-nb K=2 "
            "(L>=1, interval=5, H=32, K=2 sel, alive+un-enclosed, non-boost first)"
        ),
    },
}

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
