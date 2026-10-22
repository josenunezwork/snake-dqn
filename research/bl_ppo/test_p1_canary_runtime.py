"""Contracts for the isolated operational P1 family canary."""

from __future__ import annotations

import json

import pytest

from research.bl_ppo.p1_canary_runtime import load_config


def _config(**changes):
    value = {
        "worlds": 2,
        "processes": 1,
        "rollout_steps": 16,
        "pause_at": 32,
    }
    return json.dumps(value | changes)


def test_tiny_canary_config_accepts_only_the_frozen_operational_shape() -> None:
    assert (
        load_config(_config(), control_sha256="a" * 64, excluded_seeds=[11, 12])["pause_at"] == 32
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"scientific_admission": True},
        {"worlds": 128},
        {"actor_enabled": True},
        {"seed": 20261021},
        {"worlds": True},
        {"pause_at": 32.0},
    ],
)
def test_canary_config_refuses_scientific_or_production_shape(changes) -> None:
    with pytest.raises(ValueError):
        load_config(_config(**changes), control_sha256="a" * 64, excluded_seeds=[11, 12])


def test_canary_config_refuses_the_fixed_seed_in_a_scientific_inventory() -> None:
    with pytest.raises(ValueError, match="seed"):
        load_config(_config(), control_sha256="a" * 64, excluded_seeds=[2026102203])


def test_execute_refuses_direct_use_without_the_runtime_canary_switch(
    monkeypatch, tmp_path
) -> None:
    from research.bl_ppo.p1_canary_runtime import execute

    monkeypatch.delenv("SNAKE_COLAB_CANARY", raising=False)
    with pytest.raises(ValueError, match="SNAKE_COLAB_CANARY"):
        execute(
            control={}, control_path=tmp_path / "control.json", output=tmp_path, checkpoint=tmp_path
        )


def test_execute_requires_the_research_package_guard(monkeypatch, tmp_path) -> None:
    from research.bl_ppo.p1_canary_runtime import execute

    monkeypatch.setenv("SNAKE_COLAB_CANARY", "1")
    monkeypatch.delenv("SNAKE_COLAB_CANARY_PACKAGE", raising=False)
    with pytest.raises(ValueError, match="SNAKE_COLAB_CANARY_PACKAGE"):
        execute(
            control={}, control_path=tmp_path / "control.json", output=tmp_path, checkpoint=tmp_path
        )
