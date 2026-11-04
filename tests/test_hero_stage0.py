"""Unit tests for Stage 0 probe allocation, seed disjointness audit, and death audit aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from research.hero_stage0.death_audit import aggregate_death_audit
from research.hero_stage0.protocol import (
    CAUSE_CLASSES,
    COUNT,
    HORIZON,
    LOOKBACK_FRAMES,
    NAMESPACE,
    allocation,
    seed_for,
    training_exclusions,
)
from research.hero_stage0.seed_audit import (
    collect_c1_and_committed_bank_seeds,
    collect_watch6_alllen_screen_seeds,
    collect_watch6_l1_v1_seeds,
    collect_watch_replay_seeds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_allocation_and_training_exclusions() -> None:
    alloc = allocation()
    assert alloc["namespace"] == NAMESPACE == "t34-hero-probe-20261103-v1"
    assert alloc["count"] == COUNT == 48
    assert alloc["horizon"] == HORIZON == 6000
    assert alloc["lookback_frames"] == LOOKBACK_FRAMES == 300
    assert alloc["training_excluded"] is True
    assert len(alloc["worlds"]) == 48
    assert len(alloc["seeds"]) == 48
    assert len(set(alloc["seeds"])) == 48
    assert alloc["seeds"][0] == seed_for(NAMESPACE, 0)

    excl = training_exclusions()
    assert len(excl) == 48
    assert excl == sorted(alloc["seeds"])


def test_seed_disjointness_against_all_banks() -> None:
    seeds = set(allocation()["seeds"])
    assert len(seeds & collect_watch6_alllen_screen_seeds()["seeds"]) == 0
    assert len(seeds & collect_watch6_l1_v1_seeds()["seeds"]) == 0
    assert len(seeds & collect_watch_replay_seeds()["seeds"]) == 0
    assert len(seeds & collect_c1_and_committed_bank_seeds(REPO_ROOT)["seeds"]) == 0


def test_committed_audit_files_exist_and_match() -> None:
    pkg_dir = REPO_ROOT / "research" / "hero_stage0"
    excl_path = pkg_dir / "training_exclusions.json"
    bank_path = pkg_dir / "BANK_HERO_PROBE.json"
    audit_path = pkg_dir / "SEED_AUDIT.json"

    assert excl_path.is_file()
    assert bank_path.is_file()
    assert audit_path.is_file()

    excl = json.loads(excl_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert excl == training_exclusions()
    assert bank["seeds"] == allocation()["seeds"]
    assert audit["status"] == "CLEAR"
    assert audit["pass"] is True
    assert audit["collisions"] == []
    assert audit["training_exclusions"]["seeds"] == excl
    assert audit["training_exclusions"]["count"] == 48


def test_aggregate_death_audit_schema_and_regimes() -> None:
    fake_death_lt600 = {
        "world_index": 0,
        "seed": 123,
        "death_index_in_world": 0,
        "life": 1,
        "t_death": 400,
        "life_frames_alive": 400,
        "length": 250,
        "length_regime": "L<600",
        "sim_cause_code": 2,
        "live_cause": "self",
        "cause_class": "self-body enclosure",
        "topological_cause_class": "self-body enclosure",
        "fatal_bare_action": 1,
        "fatal_bare_is_boost": False,
        "fatal_v8_action": 1,
        "fatal_v8_outcome": 2,
        "fatal_advisory_any": True,
        "fatal_advisory_count": 1,
        "fatal_best_safe_area": 10,
        "fatal_taken_area": 10,
        "fatal_need": 160,
        "fatal_enclosed": True,
        "v8_trap_run": 15,
        "v8_differs_within_k": {str(k): (k >= 5) for k in range(1, 11)},
        "lookback_window_available": 300,
        "frames_before_fatal_last_surviving": 14,
        "steps_to_death_from_last_surviving": 15,
        "surviving_actions_at_last_surviving": [0],
        "v8_survives_at_last_surviving": True,
        "frames_before_fatal_last_open_surviving": 18,
        "steps_to_death_from_last_open_surviving": 19,
        "open_surviving_actions_at_last_open": [0],
        "v8_survives_at_last_open_surviving": True,
        "proximity_at_death": {
            "min_opp_head_l1": 20,
            "min_opp_body_l1": 14,
            "opp_head_within_k": {str(k): (20 <= k) for k in (1, 2, 3, 4, 6, 8, 10, 12, 16)},
            "opp_body_within_k": {str(k): (14 <= k) for k in (1, 2, 3, 4, 6, 8, 10, 12, 16)},
        },
        "proximity_at_last_surviving": {
            "min_opp_head_l1": 22,
            "min_opp_body_l1": 15,
            "opp_head_within_k": {str(k): (22 <= k) for k in (1, 2, 3, 4, 6, 8, 10, 12, 16)},
            "opp_body_within_k": {str(k): (15 <= k) for k in (1, 2, 3, 4, 6, 8, 10, 12, 16)},
        },
        "proximity_at_last_open_surviving": {
            "min_opp_head_l1": 24,
            "min_opp_body_l1": 16,
            "opp_head_within_k": {str(k): (24 <= k) for k in (1, 2, 3, 4, 6, 8, 10, 12, 16)},
            "opp_body_within_k": {str(k): (16 <= k) for k in (1, 2, 3, 4, 6, 8, 10, 12, 16)},
        },
        "pocket_boundary_at_death": {
            "dominant_snake_blocker": "self_body",
            "self_body_boundary_cells": 25,
            "opp_head_boundary_cells": 0,
            "opp_body_boundary_cells": 0,
        },
        "pocket_boundary_at_last_surviving": {"dominant_snake_blocker": "self_body"},
        "pocket_boundary_at_last_open_surviving": {"dominant_snake_blocker": "self_body"},
        "lookback_records": {},
        "branch_steps_evaluated": 120,
        "analysis_wall_seconds": 0.2,
    }
    world_res = {
        "world_index": 0,
        "seed": 123,
        "horizon": 6000,
        "mirror_identity_frames": 6000,
        "mean_mass": 512.0,
        "l_lt_600_alive_frames": 3000,
        "l_ge_600_alive_frames": 2990,
        "dead_frames": 10,
        "total_deaths": 1,
        "l_lt_600_deaths": 1,
        "l_ge_600_deaths": 0,
        "wall_seconds": 10.0,
        "deaths": [fake_death_lt600],
    }
    agg = aggregate_death_audit([world_res], wall_seconds=10.0, workers=1)
    assert agg["total_deaths"] == 1
    assert agg["l_lt_600_deaths"] == 1
    assert agg["l_ge_600_deaths"] == 0
    assert agg["premise_evaluation"]["premise_holds"] is False
    for c in CAUSE_CLASSES:
        assert c in agg["regimes"]["L<600"]["cause_class_distribution"]


def test_committed_death_audit_receipt() -> None:
    death_audit_path = REPO_ROOT / "research" / "hero_stage0" / "DEATH_AUDIT.json"
    assert death_audit_path.is_file()
    data = json.loads(death_audit_path.read_text(encoding="utf-8"))
    assert data["namespace"] == NAMESPACE
    assert data["world_count"] == 48
    assert data["horizon_per_world"] == 6000
    assert data["total_frames_simulated"] == 288000
    assert data["total_mirror_identity_frames"] == 288000
    assert data["total_deaths"] == 78
    assert data["l_lt_600_deaths"] == 45
    assert data["l_ge_600_deaths"] == 33
    assert data["premise_evaluation"]["premise_holds"] is False
    assert len(data["deaths"]) == 78
    assert len(data["per_world_summary"]) == 48
