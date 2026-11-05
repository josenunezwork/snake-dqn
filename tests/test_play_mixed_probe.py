"""Tests for the Post-T34 Play mixed-field option probe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.play_mixed_probe import run_probe
from research.play_mixed_probe.protocol import (
    HORIZON,
    NAMESPACE,
    OPPONENT_COUNTS_ORDER,
    RUNS_PER_OPPONENT_COUNT,
    STAND_IN_ORDER,
    TOTAL_COUNT,
    allocation,
    training_exclusions,
)
from research.play_mixed_probe.run_probe import build_play_world, run_single_episode
from src.game.ai_snake import AISnake
from src.game.scripted_snake import ScriptedSnake
from src.training.apex_policy import ApexPolicy
from web.backend.ego2s_policy import Ego2sServingPolicy
from web.backend.safety_veto_serving import V8_STRICT_RECEIPT_METHOD

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO_ROOT / "research" / "play_mixed_probe"


@pytest.fixture(scope="module", autouse=True)
def _cleanup_probe_worker_cache():
    yield
    run_probe._WORKER_CACHE.clear()


def test_seed_allocation_and_audit_contract() -> None:
    assert NAMESPACE == "t34-play-mixed-probe-20261104-v1"
    assert HORIZON == 6000
    assert OPPONENT_COUNTS_ORDER == (5, 3, 8, 1, 11)
    assert RUNS_PER_OPPONENT_COUNT == {5: 96, 3: 96, 8: 96, 1: 16, 11: 16}
    assert TOTAL_COUNT == 320

    alloc = allocation()
    assert alloc["namespace"] == NAMESPACE
    assert alloc["total_count"] == 320
    assert len(alloc["seeds"]) == 320
    assert len(set(alloc["seeds"])) == 320
    for s in alloc["seeds"]:
        assert 0 <= int(s) < (1 << 32)

    excl = training_exclusions()
    assert len(excl) == 320
    assert excl == sorted(alloc["seeds"])

    audit_path = PROBE_DIR / "SEED_AUDIT.json"
    bank_path = PROBE_DIR / "BANK_PLAY_MIXED_PROBE.json"
    excl_path = PROBE_DIR / "training_exclusions.json"
    assert audit_path.is_file()
    assert bank_path.is_file()
    assert excl_path.is_file()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "CLEAR"
    assert audit["pass"] is True
    assert audit["candidate_count"] == 320
    assert audit["seeds_sha256"] == alloc["seeds_sha256"]
    assert audit["collisions"] == []


def test_build_play_world_rosters_and_initial_parity() -> None:
    seed = 424242
    baseline_spawns = None
    baseline_foods = None

    for stand_in_id in STAND_IN_ORDER:
        for arm in ("all_b8", "mixed"):
            game, meta = build_play_world(
                seed=seed,
                opponent_count=5,
                roster_arm=arm,
                stand_in_id=stand_in_id,
            )
            assert len(game.snakes) == 6
            assert meta["num_snakes"] == 6

            slot0 = game.snakes[0]
            assert slot0.id == 0
            assert slot0.auto_respawn is False
            if stand_in_id == "c_greedy_food":
                assert isinstance(slot0, ScriptedSnake)
            else:
                assert isinstance(slot0, AISnake)

            if stand_in_id == "a_frp3_v8":
                assert isinstance(slot0.policy, ApexPolicy)
                assert getattr(slot0.safety_veto, "method", None) == V8_STRICT_RECEIPT_METHOD
            elif stand_in_id.startswith("b_b8_"):
                assert isinstance(slot0.policy, Ego2sServingPolicy)
                assert slot0.policy is not game.snakes[1].policy
                assert slot0.safety_veto is None
                assert callable(slot0.action_transform)

            # Slot 1 is always bare b8-s3 in both arms
            assert game.snakes[1].auto_respawn is True
            assert isinstance(game.snakes[1].policy, Ego2sServingPolicy)
            assert game.snakes[1].safety_veto is None

            if arm == "all_b8":
                for opp in game.snakes[2:]:
                    assert opp.auto_respawn is True
                    assert isinstance(opp.policy, Ego2sServingPolicy)
                    assert opp.safety_veto is None
            else:
                for opp in game.snakes[2:]:
                    assert opp.auto_respawn is True
                    assert isinstance(opp.policy, ApexPolicy)
                    assert getattr(opp.safety_veto, "method", None) == V8_STRICT_RECEIPT_METHOD
                    assert getattr(opp.safety_veto, "head_avoidance", None) is True

            spawns = [(int(s.id), list(s.segments), tuple(s.direction)) for s in game.snakes]
            foods = [tuple(f) for f in game.food]
            if baseline_spawns is None:
                baseline_spawns = spawns
                baseline_foods = foods
            else:
                assert spawns == baseline_spawns
                assert foods == baseline_foods


def test_single_episode_determinism_and_food_accounting() -> None:
    r1 = run_single_episode(
        seed=424242,
        opponent_count=3,
        roster_arm="mixed",
        stand_in_id="b_b8_eps01_delay3_6",
        horizon_cap=45,
    )
    r2 = run_single_episode(
        seed=424242,
        opponent_count=3,
        roster_arm="mixed",
        stand_in_id="b_b8_eps01_delay3_6",
        horizon_cap=45,
    )
    assert r1 == r2
    assert r1["food_eaten"] == r1["corpse_pellets_eaten"] + r1["ambient_pellets_eaten"]

    r3 = run_single_episode(
        seed=424242,
        opponent_count=1,
        roster_arm="all_b8",
        stand_in_id="c_greedy_food",
        horizon_cap=45,
    )
    r4 = run_single_episode(
        seed=424242,
        opponent_count=1,
        roster_arm="mixed",
        stand_in_id="c_greedy_food",
        horizon_cap=45,
    )
    # At N=1 opponent, mixed has 1 bare b8-s3 and 0 frp3-s12+v8 opponents,
    # so all_b8 and mixed are identical!
    assert r3["score"] == r4["score"]
    assert r3["run_frames"] == r4["run_frames"]
    assert r3["final_length"] == r4["final_length"]
