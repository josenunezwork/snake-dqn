"""Focused live contracts for the T13b count-parameterized Watch harness."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from research.t12_b8_serving_20261017 import episodes as t12
from research.t13b_six_snake_20261017 import episodes

HAS_B8 = Path("saved_snakes/b8_s3_t7_20261016.pth").is_file()


@pytest.fixture(autouse=True)
def _historical_all_b8_watch_roster(monkeypatch):
    """Keep the T13b evidence harness on its recorded all-B8 product posture."""
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "all_b8")


@pytest.mark.skipif(not HAS_B8, reason="checkout lacks the locally pinned B8 serving artifact")
def test_default_six_h100_matches_original_t12_trace():
    """T13b's default-six instrumentation preserves the T12 serving trace."""
    seed = 2624486211
    with contextlib.redirect_stderr(io.StringIO()):
        baseline = t12.run_watch(seed, horizon=100)
        observed = episodes.run_watch("all_b8", seed, horizon=100, snakes=6)
    assert observed["initial_world"]["effective_num_snakes"] == 6
    assert observed["hero"]["decisions"] == baseline["hero"]["decisions"]
    assert observed["hero"]["deaths"] == baseline["hero"]["deaths"]
    assert {int(k): v for k, v in observed["hero"]["length_histogram"].items()} == baseline["hero"][
        "decision_length_histogram"
    ]


@pytest.mark.skipif(not HAS_B8, reason="checkout lacks the locally pinned B8 serving artifact")
def test_supplemental_eight_has_exact_count_and_unwrapped_b8_scope():
    with contextlib.redirect_stderr(io.StringIO()):
        record = episodes.run_watch("all_b8", 2624486211, horizon=1, snakes=8)
    assert record["requested_snake_count"] == 8
    assert record["initial_world"]["effective_num_snakes"] == 8
    assert len(record["per_snake"]) == 8
    assert record["served"]["veto"]["wrapped_snake_ids"] == []


@pytest.mark.skipif(not HAS_B8, reason="checkout lacks the locally pinned B8 serving artifact")
def test_mixed_six_keeps_b8_hero_and_wraps_each_opponent_only():
    with contextlib.redirect_stderr(io.StringIO()):
        record = episodes.run_watch("mixed", 2624486211, horizon=1, snakes=6)
    assert record["served"]["registry"]["name"] == "b8-s3"
    assert record["served"]["veto"]["wrapped_snake_ids"] == []
    assert record["mixed_policy_identities"]["opponent_ids"] == list(range(1, 6))
    assert record["mixed_b8_dispatch_ids"] == [record["hero_id"]]


def test_only_all_b8_can_request_supplemental_eight():
    with pytest.raises(ValueError, match="all_b8 only"):
        episodes.run_watch("mixed", 1, horizon=1, snakes=8)
