"""Unit tests for Stage 0 probe allocation, seed disjointness audit, and death audit aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.bl_served_search import ceiling, exact_stop
from research.hero_stage0.death_audit import aggregate_death_audit
from research.hero_stage0.protocol import (
    CAUSE_CLASSES,
    COUNT,
    HORIZON,
    LOOKBACK_FRAMES,
    NAMESPACE,
    STAGE1A_BRANCH_HORIZON,
    STAGE1A_CONTROLLER,
    STAGE1A_INTERVAL,
    STAGE1A_MIN_LENGTH,
    STAGE1A_SCORING_SUFFIXES,
    STAGE1A_SELECTION_SUFFIXES,
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
from research.hero_stage0.stage1a import (
    aggregate_stage1a,
    branch_suffix_seed,
    evaluation_order_nb,
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


def test_stage1a_evaluation_order_nb_and_suffix_separation() -> None:
    assert STAGE1A_CONTROLLER == "exact-stop-h48-nb"
    assert STAGE1A_BRANCH_HORIZON == 48
    assert STAGE1A_MIN_LENGTH == 1
    assert STAGE1A_INTERVAL == 5
    assert STAGE1A_SELECTION_SUFFIXES == (0, 1)
    assert STAGE1A_SCORING_SUFFIXES == (4, 5, 6, 7)
    assert set(STAGE1A_SELECTION_SUFFIXES).isdisjoint(set(STAGE1A_SCORING_SUFFIXES))

    # Even when boost actions (3..5) have higher Q than non-boost alternatives (0..2),
    # evaluation_order_nb must place bare first, then all remaining non-boost actions
    # in Q-order before any boost action in Q-order.
    q = np.array([1.0, 3.0, 2.0, 9.0, 8.0, 7.0], dtype=np.float32)
    order = evaluation_order_nb([0, 1, 2, 3, 4, 5], bare=3, q=q)
    assert order == [3, 1, 2, 0, 4, 5]

    # When bare is a non-boost action (1), remaining non-boost (2, 0) still precede boost (3, 4, 5)
    order2 = evaluation_order_nb([0, 1, 2, 3, 4, 5], bare=1, q=q)
    assert order2 == [1, 2, 0, 3, 4, 5]

    # Suffix seed is deterministic and distinct across suffixes
    s0 = branch_suffix_seed(NAMESPACE, 12345, 10, 0)
    s1 = branch_suffix_seed(NAMESPACE, 12345, 10, 1)
    s4 = branch_suffix_seed(NAMESPACE, 12345, 10, 4)
    assert s0 == branch_suffix_seed(NAMESPACE, 12345, 10, 0)
    assert len({s0, s1, s4}) == 3


def test_stage1a_exact_stop_nb_k2_selection_semantics() -> None:
    q = np.array([1.0, 3.0, 2.0, 9.0, 8.0, 7.0], dtype=np.float32)
    legal = [0, 1, 2, 3, 4, 5]
    bare = 3

    # Case 1: bare survives both selection suffixes -> stops at bare immediately
    eval_calls_1: list[int] = []

    def eval_bare_survives(a: int) -> dict:
        eval_calls_1.append(a)
        return {"survives": True, "frames_survived": 48}

    with patch.object(exact_stop, "evaluation_order", evaluation_order_nb):
        res1 = exact_stop.run(legal, bare, q, eval_bare_survives, ceiling._choose)
    assert res1["chosen"] == 3
    assert eval_calls_1 == [3]

    # Case 2: bare fails, boost 4 would survive, but non-boost 2 also survives ->
    # must pick non-boost 2 before boost 4 is ever tested!
    eval_calls_2: list[int] = []
    surv_map = {3: False, 1: False, 2: True, 0: False, 4: True, 5: True}

    def eval_nonboost_wins(a: int) -> dict:
        eval_calls_2.append(a)
        return {"survives": surv_map[a], "frames_survived": 48 if surv_map[a] else 10}

    with patch.object(exact_stop, "evaluation_order", evaluation_order_nb):
        res2 = exact_stop.run(legal, bare, q, eval_nonboost_wins, ceiling._choose)
    assert res2["chosen"] == 2
    assert eval_calls_2 == [3, 1, 2]


def test_aggregate_stage1a_bars_and_verdict() -> None:
    fake_override = {
        "world_index": 0,
        "seed": 123,
        "frame": 25,
        "decision_index": 24,
        "life": 1,
        "length": 120,
        "length_regime": "L<600",
        "bare_action": 1,
        "chosen_action": 0,
        "bare_is_boost": False,
        "chosen_is_boost": False,
        "q": [1.0, 2.0, 0.5, 0.1, 0.1, 0.1],
        "advisory": [True, True, True, True, True, True],
        "evaluation_order": [1, 0, 2, 3, 4, 5],
        "branches_evaluated": 2,
        "chosen_survived_selection_both": True,
        "bare_selection_survives": [False],
        "chosen_selection_survives": [True, True],
        "scoring": {
            "scoring_suffixes": [4, 5, 6, 7],
            "bare_action": 1,
            "chosen_action": 0,
            "bare_survives_vec": [False, False, False, False],
            "chosen_survives_vec": [True, True, True, True],
            "override_dies_where_b8_survives_vec": [False, False, False, False],
            "n_scoring_suffixes": 4,
            "n_bare_survives": 0,
            "n_chosen_survives": 4,
            "n_chosen_dies": 0,
            "n_override_dies_where_b8_survives": 0,
            "s_score_bare": 0.0,
            "s_score_chosen": 1.0,
            "harmful": False,
            "harmful_conditional_majority": False,
            "harmful_majority_of_4": False,
            "harmful_ge2_of_4": False,
            "harmful_ge1_of_4": False,
            "bare_outcomes": {},
            "chosen_outcomes": {},
            "scoring_rollouts_evaluated": 8,
            "scoring_env_frames": 384,
            "scoring_wall_seconds": 0.05,
        },
    }
    bare_arm = {
        "world_index": 0,
        "seed": 123,
        "arm": "bare",
        "min_length": None,
        "interval": 5,
        "frames": 6000,
        "alive_frames": 5900,
        "alive_frames_lt600": 4500,
        "alive_frames_ge600": 1400,
        "dead_frames": 100,
        "mass_sum": 2400000,
        "mass_integral": 400.0,
        "deaths": 4,
        "deaths_per_1k": 4.0 * 1000.0 / 6000.0,
        "short_body_deaths_lt600": 4,
        "long_body_deaths_ge600": 0,
        "late_large_body_deaths_ge800": 0,
        "deaths_lt600_per_1k_frames": 4.0 * 1000.0 / 6000.0,
        "deaths_ge600_per_1k_frames": 0.0,
        "deaths_lt600_per_1k_alive_lt600": 4.0 * 1000.0 / 4500.0,
        "deaths_ge600_per_1k_alive_ge600": 0.0,
        "lives": 5,
        "decisions": 5900,
        "decisions_lt600": 4500,
        "decisions_ge600": 1400,
        "boost_decisions": 10,
        "boost_decisions_lt600": 8,
        "boost_decisions_ge600": 2,
        "boost_share": 10.0 / 5900.0,
        "max_length": 700,
        "alive_length_median": 380.0,
        "alive_length_p95": 650.0,
        "death_lengths": [100, 200, 300, 400],
        "death_records": [
            {"length": 100, "cause_class": "self-body enclosure"},
            {"length": 200, "cause_class": "opponent body"},
            {"length": 300, "cause_class": "self-body enclosure"},
            {"length": 400, "cause_class": "opponent head-on"},
        ],
        "search_roots": 0,
        "search_roots_lt600": 0,
        "search_roots_ge600": 0,
        "bare_survived_both_roots": 0,
        "all_die_roots": 0,
        "interventions": 0,
        "interventions_lt600": 0,
        "interventions_ge600": 0,
        "interventions_to_nonboost": 0,
        "interventions_to_boost": 0,
        "branches_evaluated": 0,
        "branches_skipped": 0,
        "selection_rollouts_evaluated": 0,
        "selection_env_frames": 0,
        "selection_wall_seconds": 0.0,
        "scoring_rollouts_evaluated": 0,
        "scoring_env_frames": 0,
        "scoring_wall_seconds": 0.0,
        "native_v8_kernel_rows": 0,
        "overrides": [],
        "respawns_per_slot": [4, 2, 2, 2, 2, 2],
        "mirror_identity_frames": 6000,
        "wall_seconds": 30.0,
        "roster": {},
    }
    cand_arm = {
        **bare_arm,
        "arm": STAGE1A_CONTROLLER,
        "min_length": 1,
        "mass_sum": 3300000,
        "mass_integral": 550.0,
        "deaths": 1,
        "deaths_per_1k": 1.0 * 1000.0 / 6000.0,
        "short_body_deaths_lt600": 1,
        "deaths_lt600_per_1k_frames": 1.0 * 1000.0 / 6000.0,
        "death_lengths": [200],
        "death_records": [{"length": 200, "cause_class": "self-body enclosure"}],
        "search_roots": 1180,
        "search_roots_lt600": 800,
        "search_roots_ge600": 380,
        "bare_survived_both_roots": 1179,
        "interventions": 1,
        "interventions_lt600": 1,
        "interventions_to_nonboost": 1,
        "branches_evaluated": 1181,
        "branches_skipped": 4,
        "selection_rollouts_evaluated": 2361,
        "selection_env_frames": 113300,
        "selection_wall_seconds": 120.0,
        "scoring_rollouts_evaluated": 8,
        "scoring_env_frames": 384,
        "scoring_wall_seconds": 0.05,
        "overrides": [fake_override],
        "wall_seconds": 150.0,
    }
    pair_row = {
        "schema": "t34-hero-stage1a-paired-world/v1",
        "namespace": NAMESPACE,
        "world_index": 0,
        "seed": 123,
        "horizon": 6000,
        "arm_execution_order": ["bare", STAGE1A_CONTROLLER],
        "arms": {"bare": bare_arm, STAGE1A_CONTROLLER: cand_arm},
        "paired_delta": {
            "mass_integral": 150.0,
            "deaths": -3,
            "deaths_lt600": -3,
            "deaths_ge600": 0,
            "deaths_per_1k": -3.0 * 1000.0 / 6000.0,
            "boost_share_pp": 0.0,
        },
        "wall_seconds": 180.0,
    }
    agg = aggregate_stage1a([pair_row], wall_seconds=180.0, workers=1)
    assert agg["verdict"] == "PASS"
    assert agg["pass"] is True
    assert agg["bars"]["l_lt_600_deaths_removed_pct"]["observed_pct"] == 75.0
    assert agg["bars"]["boost_share_change_pp"]["observed_pooled_pp"] == 0.0
    assert agg["bars"]["harmful_overrides_pct"]["observed_conditional_majority_pct"] == 0.0


def test_committed_stage1a_readout_receipt() -> None:
    readout_path = REPO_ROOT / "research" / "hero_stage0" / "STAGE1A_READOUT.json"
    assert readout_path.is_file()
    data = json.loads(readout_path.read_text(encoding="utf-8"))
    assert data["namespace"] == NAMESPACE
    assert data["world_count"] == 48
    assert data["horizon_per_world"] == 6000
    assert data["total_frames_per_arm"] == 288000
    assert data["mirror_identity_frames_bare"] == 288000
    assert data["mirror_identity_frames_exact_stop_h48_nb"] == 288000
    assert data["arms"]["bare"]["deaths"]["lt600"] == 45
    assert data["arms"]["bare"]["deaths"]["ge600"] == 33
    assert data["arms"]["bare"]["deaths"]["total"] == 78
    assert data["arms"][STAGE1A_CONTROLLER]["deaths"]["lt600"] == 16
    assert data["arms"][STAGE1A_CONTROLLER]["deaths"]["ge600"] == 24
    assert data["arms"][STAGE1A_CONTROLLER]["deaths"]["total"] == 40
    assert data["override_scoring"]["total_overrides_scored"] == 343
    assert data["verdict"] == "FAIL"
    assert data["pass"] is False
    assert len(data["per_world_summary"]) == 48
