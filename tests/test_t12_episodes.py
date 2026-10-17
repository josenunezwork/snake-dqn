"""Focused smoke coverage for the T12 B8 serving-episode evidence helpers."""

from __future__ import annotations

import os

import pytest

from research.t12_b8_serving_20261017 import episodes


def _b8_available() -> bool:
    """A checkout with the intentionally-unfilled pin cannot run even a B8 smoke."""
    from web.backend.served_checkpoint import resolve_served_checkpoint

    old = os.environ.get("SNAKE_SERVE_CHECKPOINT")
    try:
        os.environ["SNAKE_SERVE_CHECKPOINT"] = "b8-s3"
        return resolve_served_checkpoint().name == "b8-s3"
    finally:
        if old is None:
            os.environ.pop("SNAKE_SERVE_CHECKPOINT", None)
        else:
            os.environ["SNAKE_SERVE_CHECKPOINT"] = old


def test_serving_env_selects_only_b8_and_restores(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_CHECKPOINT", "champion")
    monkeypatch.setenv("SNAKE_SERVE_VETO_VARIANT", "v7")
    with episodes.serving_env():
        assert os.environ["SNAKE_SERVE_CHECKPOINT"] == "b8-s3"
        assert "SNAKE_SERVE_VETO_VARIANT" not in os.environ
        assert "SNAKE_SERVE_VETO_WATCH_HERO" not in os.environ
        assert "SNAKE_SERVE_VETO_PLAY_AI" not in os.environ
    assert os.environ["SNAKE_SERVE_CHECKPOINT"] == "champion"
    assert os.environ["SNAKE_SERVE_VETO_VARIANT"] == "v7"


def test_length_quantiles_keep_actual_decision_samples():
    assert episodes._quantiles([])["p50"] is None
    assert episodes._quantiles([3, 1, 2, 100]) == {
        "p50": 3.0,
        "p95": 100.0,
        "min": 1.0,
        "max": 100.0,
    }


@pytest.mark.skipif(not _b8_available(), reason="b8-s3 strict pin is intentionally unfilled")
def test_watch_smoke_keeps_its_original_all_b8_unwrapped_contract(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "all_b8")
    record = episodes.run_watch(918273, horizon=10)
    assert record["status"] == "complete"
    assert record["served"]["registry"]["name"] == "b8-s3"
    assert record["served"]["veto"]["active"] is False
    assert record["served"]["veto"]["wrapped_snake_ids"] == []
    assert record["frames_stepped"] == 10
    assert record["hero"]["decisions"] >= 1


@pytest.mark.skipif(not _b8_available(), reason="b8-s3 strict pin is intentionally unfilled")
def test_controls_smoke_uses_real_finalized_score_path():
    record = episodes.run_controls(817263)
    assert record["status"] == "complete"
    assert record["play_opponents"] == {"1": 1, "5": 5, "11": 11}
    assert record["train_refusal"]["mode"] == "watch"
    assert record["train_refusal"]["policy_training"] is False
    assert len(record["train_refusals"]) == 2
    assert record["train_refusals"][0]["last_error"] == record["train_refusals"][1]["last_error"]
    assert record["score_path"]["submit_status"] == 200
    assert record["score_path"]["server_last_run"]["served_checkpoint"] == "b8-s3"
    assert record["score_path"]["recent"]["recent"][0]["served_checkpoint"] == "b8-s3"
