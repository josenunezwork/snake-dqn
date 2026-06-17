"""Tests for greedy checkpoint evaluation helpers."""

import argparse

import pytest

from src.scripts.evaluate_checkpoints import (
    checkpoint_rank_key,
    evaluate_checkpoints,
    parse_seed_list,
    summarize_rollouts,
)


def _stats(reward=0.0, food=0, deaths=0, length=100, kills=0):
    return {
        "episode": {
            "reward": reward,
            "food_eaten": food,
            "deaths": deaths,
            "length": length,
            "kills": kills,
        }
    }


def test_parse_seed_list_rejects_empty_values():
    assert parse_seed_list("0, 2,5") == [0, 2, 5]

    with pytest.raises(argparse.ArgumentTypeError):
        parse_seed_list(" , ")


def test_summarize_rollouts_aggregates_eval_metrics():
    summary = summarize_rollouts(
        "checkpoint.pth",
        [
            _stats(reward=10.0, food=4, deaths=0, length=100),
            _stats(reward=2.0, food=1, deaths=1, length=50),
        ],
    )

    assert summary["checkpoint"] == "checkpoint.pth"
    assert summary["avg_reward"] == pytest.approx(6.0)
    assert summary["avg_food"] == pytest.approx(2.5)
    assert summary["avg_deaths"] == pytest.approx(0.5)
    assert summary["avg_length"] == pytest.approx(75.0)


def test_checkpoint_rank_key_prefers_reward_then_food_then_survival():
    high_reward = {"avg_reward": 2.0, "avg_food": 0.0, "avg_deaths": 5.0, "avg_length": 10.0}
    low_reward = {"avg_reward": 1.0, "avg_food": 100.0, "avg_deaths": 0.0, "avg_length": 1000.0}
    more_food = {"avg_reward": 2.0, "avg_food": 4.0, "avg_deaths": 1.0, "avg_length": 10.0}

    assert checkpoint_rank_key(high_reward) > checkpoint_rank_key(low_reward)
    assert checkpoint_rank_key(more_food) > checkpoint_rank_key(high_reward)


def test_evaluate_checkpoints_runs_greedy_no_save_rollouts_and_sorts():
    outputs = {
        ("a.pth", 0): _stats(reward=3.0, food=2),
        ("a.pth", 1): _stats(reward=5.0, food=4),
        ("b.pth", 0): _stats(reward=8.0, food=1),
        ("b.pth", 1): _stats(reward=9.0, food=1),
    }
    calls = []

    def fake_smoke_runner(**kwargs):
        calls.append(kwargs)
        checkpoint = kwargs["checkpoint_path"]
        seed_index = sum(1 for call in calls if call["checkpoint_path"] == checkpoint) - 1
        return outputs[(checkpoint, seed_index)]

    summaries = evaluate_checkpoints(
        ["a.pth", "b.pth"],
        frames=123,
        seeds=[0, 1],
        smoke_runner=fake_smoke_runner,
    )

    assert [summary["checkpoint"] for summary in summaries] == ["b.pth", "a.pth"]
    assert summaries[0]["avg_reward"] == pytest.approx(8.5)
    assert all(call["max_frames"] == 123 for call in calls)
    assert all(call["checkpoint_filename"] is None for call in calls)
    assert all(call["eval_mode"] is True for call in calls)
