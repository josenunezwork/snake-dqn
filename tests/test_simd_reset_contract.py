"""Contract tests for BatchSim reset, masks, and active-world stepping."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.core.game_config import get_config, initialize_config
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.parity import _install_v2_config


def _cfg(**overrides: object) -> BatchSimConfig:
    base = dict(
        num_envs=2,
        num_snakes=2,
        game_width=200,
        game_height=160,
        segment_size=10,
        wall_thickness=10,
        initial_food=8,
        max_food=8,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
        max_capacity=64,
    )
    base.update(overrides)
    return BatchSimConfig(**base)


def _live_snapshot(game) -> tuple:
    """Read actual GameState reset output without using the parity helper."""
    return (
        tuple(
            tuple((pos[0] // 10, pos[1] // 10) for pos in snake.segments) for snake in game.snakes
        ),
        tuple((pos[0] // 10, pos[1] // 10) for pos in game.food),
    )


def _batch_snapshot(sim: BatchSim, env: int) -> tuple:
    return (
        tuple(tuple(sim.get_bodies(env, snake)) for snake in range(sim.S)),
        tuple(sim.get_food(env)),
    )


@pytest.mark.parametrize("seed", [41, 314159])
def test_repeated_resets_match_actual_game_state_rng_order(seed: int) -> None:
    """Constructor food draws occur once, while GameState soft resets do not repeat them."""
    from src.game.game_state import GameState

    cfg = _cfg(num_envs=1)
    saved = get_config()
    _install_v2_config(cfg)
    try:
        random.seed(seed)
        game = GameState(headless=True, num_snakes=cfg.num_snakes)
        sim = BatchSim(cfg, seeds=[seed], train_mode=True)
        for _ in range(3):
            assert _batch_snapshot(sim, 0) == _live_snapshot(game)
            game.reset()
            sim.reset()
    finally:
        initialize_config(saved)


def test_inactive_environment_is_frozen_and_next_reset_keeps_its_rng_stream() -> None:
    """Advancing env 0 cannot mutate env 1 or consume env 1's future spawn draws."""
    cfg = _cfg()
    sim = BatchSim(cfg, seeds=[41, 314159], train_mode=True)
    control = BatchSim(cfg, seeds=[41, 314159], train_mode=True)
    before = _batch_snapshot(sim, 1)
    before_frame = sim.frame[1].copy()
    before_rng = sim._rngs[1]._rng.getstate()

    actions = np.array([[1, 1], [4, 4]], dtype=np.int64)
    sim.step(actions, active_env_mask=np.array([True, False], dtype=bool))
    control.step(actions, active_env_mask=np.array([False, False], dtype=bool))

    assert _batch_snapshot(sim, 1) == before
    assert sim.frame[1] == before_frame
    assert sim._rngs[1]._rng.getstate() == before_rng
    assert not sim.get_transition_valid()[1].any()
    assert sim.get_transition_valid()[0].all()
    assert not sim.get_step_events()["transition_valid"][1].any()

    sim.reset()
    control.reset()
    assert _batch_snapshot(sim, 1) == _batch_snapshot(control, 1)


def test_masks_keep_legacy_advisory_meaning_and_resolve_against_domain_legality() -> None:
    cfg = _cfg(num_envs=1, num_snakes=1, initial_food=0, max_food=0)
    sim = BatchSim(cfg, seeds=[7], train_mode=True)

    # A living length-one snake may issue all normal actions but no boost.
    legal = sim.get_legal_action_mask()[0, 0]
    assert legal.tolist() == [True, True, True, False, False, False]
    assert np.array_equal(sim.get_action_mask(), sim.get_advisory_action_mask())
    assert not sim.get_resolved_action_mask()[0, 0, 3:].any()

    # At the left wall straight-left is advisory-fatal, while a turn can remain
    # safe. Resolution preserves that advisory veto and never enables boost.
    sim.bodies[0, 0, 0] = (0, 3)
    sim.direction[0, 0] = 3
    sim._rebuild_traversed_from_heads()
    sim._refresh_action_masks()
    advisory = sim.get_advisory_action_mask()[0, 0]
    assert not advisory[1]
    assert sim.get_resolved_action_mask()[0, 0, 1] == advisory[1]
    assert not sim.get_resolved_action_mask()[0, 0, 3:].any()

    sim.alive[0, 0] = False
    sim._refresh_action_masks()
    assert not sim.get_legal_action_mask()[0, 0].any()
    assert not sim.get_resolved_action_mask()[0, 0].any()


def test_active_env_mask_rejects_non_boolean_or_wrong_shape() -> None:
    sim = BatchSim(_cfg(), seeds=[1, 2], train_mode=True)
    actions = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="boolean dtype"):
        sim.step(actions, active_env_mask=np.array([1, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="shape"):
        sim.step(actions, active_env_mask=np.array([[True, False]], dtype=bool))
    with pytest.raises(ValueError, match="actions must have shape"):
        sim.step(np.zeros((2, 1), dtype=np.int64))
