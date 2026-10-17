"""Focused contracts for T15's real mixed Watch measurement helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from research.t15_mixed_roster_20261017 import episodes, measurements

HAS_B8 = Path("saved_snakes/b8_s3_t7_20261016.pth").is_file()


def test_mixed_environment_is_opt_in_and_restores(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_CHECKPOINT", "champion")
    monkeypatch.setenv("SNAKE_SERVE_VETO_VARIANT", "v7")
    with episodes.mixed_environment():
        assert os.environ["SNAKE_SERVE_WATCH_ROSTER"] == "mixed"
        assert "SNAKE_SERVE_CHECKPOINT" not in os.environ
        assert "SNAKE_SERVE_VETO_VARIANT" not in os.environ
    assert os.environ["SNAKE_SERVE_CHECKPOINT"] == "champion"
    assert os.environ["SNAKE_SERVE_VETO_VARIANT"] == "v7"


def test_distribution_reports_raw_percentiles():
    assert measurements._distribution([1, 2, 3, 4])["p95"] == pytest.approx(3.85)


@pytest.mark.skipif(not HAS_B8, reason="checkout lacks B8 artifact")
def test_mixed_watch_short_smoke_has_exact_scope_and_hero_dispatch():
    record = episodes.run_watch(2624486211, horizon=10, num_snakes=6)
    assert record["status"] == "complete"
    assert record["roster"]["watch_roster"] == "mixed"
    assert record["roster"]["served"]["name"] == "b8-s3"
    assert record["roster"]["opponent_checkpoint"]["name"] == "frp3-s12"
    assert record["final_veto"]["scope"] == "watch_opponents"
    assert record["final_veto"]["wrapped_snake_ids"] == [1, 2, 3, 4, 5]
    assert record["dispatch"]["hero_only"]


@pytest.mark.skipif(not HAS_B8, reason="checkout lacks B8 artifact")
def test_forced_dead_hero_keeps_empty_observation_until_live_respawn():
    """A real Watch respawn delay must not be mistaken for an opponent row."""
    session = episodes.new_session(2624486211, num_snakes=6)
    hero = next(snake for snake in session.game.snakes if int(snake.id) == int(session.hero_id))
    hero.die()
    with episodes._HeroDispatchEvidence(session) as evidence:
        # This is the real serving policy's pre-respawn snapshot: no live B8
        # hero means no B8 observation row.  The following Watch steps cover
        # its ordinary countdown and live respawn.
        session.policy.prepare_frame()
        for _ in range(20):
            session.step()
            session.snapshot()
    assert any(not alive for alive in evidence.observation_alive)
    assert all(
        ids == ([int(hero.id)] if alive else [])
        for ids, alive in zip(evidence.observation_ids, evidence.observation_alive)
    )
