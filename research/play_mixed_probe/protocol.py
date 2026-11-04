"""Protocol constants and deterministic seed allocation for the Play mixed-field probe."""

from __future__ import annotations

import hashlib
from typing import Any

NAMESPACE = "t34-play-mixed-probe-20261104-v1"
HORIZON = 6000

# Opponent counts: UI presets N in {3, 5, 8} (>=96 paired runs each),
# plus N=1 and N=11 smoke tests (>=16 paired runs each).
# N=5 (the primary decision preset) occupies indices 0..95.
OPPONENT_COUNTS_ORDER: tuple[int, ...] = (5, 3, 8, 1, 11)
RUNS_PER_OPPONENT_COUNT: dict[int, int] = {
    5: 96,
    3: 96,
    8: 96,
    1: 16,
    11: 16,
}
TOTAL_COUNT = sum(RUNS_PER_OPPONENT_COUNT.values())  # 320

ROSTER_ARMS: tuple[str, ...] = ("all_b8", "mixed")

STAND_IN_SPECS: dict[str, dict[str, Any]] = {
    "a_frp3_v8": {
        "id": "a_frp3_v8",
        "code": "(a)",
        "label": "(a) frp3-s12 + v8 (strong)",
        "kind": "frp3_v8",
    },
    "b_b8_eps01_delay3_6": {
        "id": "b_b8_eps01_delay3_6",
        "code": "(b, eps=0.1)",
        "label": "(b) bare b8-s3 (eps=0.1, delay 3-6f)",
        "kind": "b8_noisy_delayed",
        "epsilon": 0.1,
        "delay_min": 3,
        "delay_max": 6,
    },
    "b_b8_eps03_delay3_6": {
        "id": "b_b8_eps03_delay3_6",
        "code": "(b, eps=0.3)",
        "label": "(b) bare b8-s3 (eps=0.3, delay 3-6f)",
        "kind": "b8_noisy_delayed",
        "epsilon": 0.3,
        "delay_min": 3,
        "delay_max": 6,
    },
    "c_greedy_food": {
        "id": "c_greedy_food",
        "code": "(c)",
        "label": "(c) scripted:greedy_food (novice)",
        "kind": "greedy_food",
    },
}

STAND_IN_ORDER: tuple[str, ...] = (
    "a_frp3_v8",
    "b_b8_eps01_delay3_6",
    "b_b8_eps03_delay3_6",
    "c_greedy_food",
)

# Decision rule (OPUS_CRITIQUE.md §4 / CAMPAIGN_LOG.md 2026-11-04 PT 19:00):
# Mixed becomes a candidate Play option iff no stand-in shows paired median score
# or survival worse by more than 15% (LB95 >= -15.0%) at N=5.
RULE_OPPONENT_COUNT = 5
RULE_MAX_REL_LOSS_PCT = -15.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261104


def seed_for(namespace: str, index: int) -> int:
    """Derive a deterministic 32-bit unsigned seed from ``namespace`` and ``index``."""
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if type(index) is not int or index < 0:
        raise ValueError("index must be a non-negative int")
    digest = hashlib.sha256(f"{namespace}/{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def allocation(namespace: str = NAMESPACE) -> dict[str, Any]:
    """Return the deterministic Play mixed-field probe world allocation contract."""
    worlds: list[dict[str, Any]] = []
    by_opponents: dict[str, list[dict[str, Any]]] = {}
    global_idx = 0
    for n_opp in OPPONENT_COUNTS_ORDER:
        count = RUNS_PER_OPPONENT_COUNT[n_opp]
        bucket: list[dict[str, Any]] = []
        for run_idx in range(count):
            s = seed_for(namespace, global_idx)
            entry = {
                "global_index": global_idx,
                "opponent_count": n_opp,
                "num_snakes": n_opp + 1,
                "run_index": run_idx,
                "seed": s,
                "horizon_cap": HORIZON,
                "is_smoke": n_opp in (1, 11),
            }
            worlds.append(entry)
            bucket.append(entry)
            global_idx += 1
        by_opponents[str(n_opp)] = bucket

    seeds = [w["seed"] for w in worlds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("internal seed collision in play_mixed_probe allocation")
    seeds_digest = hashlib.sha256(",".join(str(s) for s in seeds).encode("utf-8")).hexdigest()
    return {
        "schema": "t34-play-mixed-probe-allocation/v1",
        "namespace": namespace,
        "total_count": len(worlds),
        "horizon_cap": HORIZON,
        "training_excluded": True,
        "opponent_counts_order": list(OPPONENT_COUNTS_ORDER),
        "runs_per_opponent_count": {str(k): v for k, v in RUNS_PER_OPPONENT_COUNT.items()},
        "roster_arms": list(ROSTER_ARMS),
        "stand_ins": list(STAND_IN_ORDER),
        "world_contract": {
            "mode": "play",
            "config": "configs/mechanics_v2.yaml",
            "mechanics_version": 2,
            "slot_0_auto_respawn": False,
            "opponent_auto_respawn": True,
            "all_b8_roster": "slot 0 stand-in + N bare b8-s3",
            "mixed_roster": (
                "slot 0 stand-in + 1 bare b8-s3 (slot 1) + (N-1) frp3-s12+v8 (slots 2..N)"
            ),
            "b8_checkpoint": "b8-s3",
            "frp3_checkpoint": "frp3-s12",
            "frp3_veto_variant": "v8",
            "frp3_veto_lambda": 8.0,
        },
        "by_opponents": by_opponents,
        "worlds": worlds,
        "seeds": seeds,
        "seeds_sha256": seeds_digest,
    }


def training_exclusions(namespace: str = NAMESPACE) -> list[int]:
    """Return the sorted list of unique integer probe seeds excluded from all training."""
    seeds = allocation(namespace)["seeds"]
    return sorted(int(s) for s in seeds)
