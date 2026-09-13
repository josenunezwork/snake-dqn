"""Regression coverage for the opt-in canonical Watch scripted adapter."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.anchors import ScriptedAnchor, watch_anchor_frame
from src.model.obs_spec import RASTER31V3
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.eval_engine import _ProfileAnchorSimdPolicy
from src.training.pqn_trainer import (
    DECISION_PHASE_WATCH_PRE_MOVE_V1,
    PQNConfig,
    PQNTrainer,
)
from src.training.rollout_policies import canonical_watch_scripted_source


def _sim() -> BatchSim:
    return BatchSim(
        BatchSimConfig(num_envs=3, num_snakes=3, initial_food=4, max_food=4),
        seeds=[17, (1 << 63) + 23, 41],
        train_mode=False,
    )


@pytest.mark.parametrize("kind", ["random_safe", "greedy_food"])
def test_canonical_policy_matches_profile_wrapper_across_watch_lanes_and_reset(kind: str) -> None:
    """Training's sparse rows use the same contexts/actions as profiled SIMD."""
    sim = _sim()
    source = canonical_watch_scripted_source(kind)
    profile = _ProfileAnchorSimdPolicy(kind, sim.get_world_seeds())
    slots = np.asarray([[2, 2], [0, 1], [1, 2], [2, 1]], dtype=np.int64)
    observations: list[tuple[tuple[int, ...], list[int], list[int], list[int]]] = []
    training_contexts = []
    original = ScriptedAnchor.action

    def capture(self: ScriptedAnchor, context):
        if self is source.policy._anchor:
            training_contexts.append(context)
        return original(self, context)

    # Both wrappers call the shared anchor, but the assertions below examine
    # the training adapter's contexts directly rather than only comparing two
    # resulting action vectors.
    setattr(ScriptedAnchor, "action", capture)

    try:

        def select(prepared: BatchSim) -> np.ndarray:
            masks = prepared.get_resolved_action_mask()[slots[:, 0], slots[:, 1]]
            training_actions = source.actions(masks, prepared, slots)
            profile_actions = profile.actions(masks, prepared, slots)
            observations.append(
                (
                    prepared.get_world_seeds(),
                    prepared.frame.tolist(),
                    training_actions.tolist(),
                    profile_actions.tolist(),
                )
            )
            assert np.array_equal(training_actions, profile_actions)
            full = np.ones((prepared.E, prepared.S), dtype=np.int64)
            full[slots[:, 0], slots[:, 1]] = training_actions
            return full

        # First and continuing prepared Watch decisions retain sparse caller order.
        sim.step_with_policy(select)
        sim.step_with_policy(select)
        assert observations[0][1] == [1, 1, 1]
        assert observations[1][1] == [2, 2, 2]

        replacement = (1 << 63) + 1_000_003
        sim.reset_envs(np.asarray([False, True, False], dtype=bool), seeds=[replacement])
        assert sim.get_world_seeds() == (17, replacement, 41)
        # Production profiles own their declared seed assignment. Rebuild its
        # adapter after this selected reset; the training source reads the same
        # current lane assignment directly from BatchSim.
        profile = _ProfileAnchorSimdPolicy(kind, sim.get_world_seeds())
        sim.step_with_policy(select)
    finally:
        setattr(ScriptedAnchor, "action", original)

    seeds, frames, training_actions, profile_actions = observations[-1]
    assert seeds == (17, replacement, 41)
    assert frames == [3, 1, 3]
    assert training_actions == profile_actions
    assert source.identity == f"scripted-anchor/v1:{kind}"
    assert [(context.world_seed, context.slot, context.frame) for context in training_contexts] == [
        (41, 2, 0),
        (17, 1, 0),
        ((1 << 63) + 23, 2, 0),
        (41, 1, 0),
        (41, 2, 1),
        (17, 1, 1),
        ((1 << 63) + 23, 2, 1),
        (41, 1, 1),
        (41, 2, 2),
        (17, 1, 2),
        ((1 << 63) + 1_000_003, 2, 0),
        (41, 1, 2),
    ]


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1"])
def test_watch_anchor_frame_rejects_non_prepared_values(value: object) -> None:
    with pytest.raises(ValueError, match="prepared_frame"):
        watch_anchor_frame(value)  # type: ignore[arg-type]


def test_profiled_live_wrapper_uses_the_shared_watch_anchor_clock(
    monkeypatch: pytest.MonkeyPatch, setup_config: None
) -> None:
    """The live wrapper and training adapter share the public frame conversion."""
    from src.evaluation.protocol import promotion_v2_watch_rect
    from src.game.game_state_factory import create_training_game_state
    from src.scripts.tournament_eval import _attach_agent, _evaluation_world_from_config
    from src.simd_env.parity import _install_v2_config

    # The production Watch profile is deliberately v2-only. Keep this test's
    # global world aligned with the compact controlled-parity setup rather than
    # depending on the default (legacy mechanics-v1) test configuration.
    _install_v2_config(
        BatchSimConfig(
            num_envs=1,
            num_snakes=3,
            game_width=400,
            game_height=300,
            initial_food=30,
            max_food=35,
            mechanics_version=2,
            frame_rate=1,
            max_capacity=64,
        )
    )

    captured = []
    original = ScriptedAnchor.action

    def capture(self: ScriptedAnchor, context):
        captured.append(context)
        return original(self, context)

    monkeypatch.setattr(ScriptedAnchor, "action", capture)
    game = create_training_game_state(eval_mode=False)
    profile = promotion_v2_watch_rect(_evaluation_world_from_config())
    world_seed = (1 << 63) + 79
    try:
        game.frame = 1
        _attach_agent(game, 1, ("scripted", "random_safe"), world_seed, {}, profile)
        opponent = game.snakes[1]
        opponent._choose_action(
            [snake for index, snake in enumerate(game.snakes) if index != 1], game.food
        )
    finally:
        game.full_cleanup()

    assert len(captured) == 1
    context = captured[0]
    assert (context.world_seed, context.slot, context.frame) == (world_seed, 1, 0)


def test_fixed_watch_source_is_recorded_in_a_tiny_native_rollout_checkpoint() -> None:
    """The opt-in identity reaches real Watch rollout and checkpoint provenance."""
    source = canonical_watch_scripted_source("random_safe")
    config = PQNConfig(
        num_envs=1,
        num_snakes=2,
        rollout_len=1,
        max_frames=1,
        initial_food=4,
        max_food=4,
        hero_frac=0.0,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
        rollout_policy_mode="fixed",
        fixed_policy_identity=source.identity,
        decision_phase_mode=DECISION_PHASE_WATCH_PRE_MOVE_V1,
    )
    trainer = PQNTrainer(config, fixed_policy=source)
    rollout = trainer._rollout()
    checkpoint = trainer.checkpoint_state()

    assert rollout["actions"].shape == (1, 1, 2)
    assert checkpoint["decision_phase_mode"] == DECISION_PHASE_WATCH_PRE_MOVE_V1
    assert checkpoint["rollout_policy_source"] == {
        "mode": "fixed",
        "identity": source.identity,
        "policy_source_contract_digest": checkpoint["policy_source_contract_digest"],
    }
    assert checkpoint["policy_source_contract"]["fixed_policy_identity"] == source.identity
    assert checkpoint["sampler_contract"]["rollout_policy_source"] == {
        "mode": "fixed",
        "identity": source.identity,
    }
