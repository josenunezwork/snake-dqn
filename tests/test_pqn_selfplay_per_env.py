"""Oracles for per-environment PQN assignment and action RNG seams."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.core.seeding import derive_seed
from src.model.raster_network import RasterDuelingNetwork
from src.training.pqn_selfplay import (
    HERO_POLICY_ID,
    OpponentPool,
    PerEnvExplorationDecisions,
    PinnedOpponentPool,
    assign_policy_ids_per_env,
    batched_act,
    sample_per_env_exploration,
)


def _action_rngs(run_seed: int, episode_ids: np.ndarray) -> dict[int, np.random.Generator]:
    return {
        env: np.random.default_rng(
            derive_seed(run_seed, f"pqn/action/env/{env}/episode/{int(episode_ids[env])}")
        )
        for env in range(len(episode_ids))
    }


def _policy_grid(envs: int = 2, slots: int = 3) -> np.ndarray:
    return np.full((envs, slots), HERO_POLICY_ID, dtype=np.int64)


def _biased_network(action: int) -> RasterDuelingNetwork:
    network = RasterDuelingNetwork().eval()
    with torch.no_grad():
        network.advantage_stream[-1].bias.zero_()
        network.advantage_stream[-1].bias[action] = 100.0
        network.value_stream[-1].bias.zero_()
    return network


def test_selected_assignment_rows_are_independent_and_replayable() -> None:
    """Changing env 0's episode cannot redraw continuing env 1's policy row."""
    seed = 73
    episode_ids = np.array([4, 9], dtype=np.uint64)
    initial = _policy_grid()
    both = assign_policy_ids_per_env(
        initial,
        env_indices=[1, 0],
        episode_ids=episode_ids,
        pool_ids=[5, 8],
        hero_frac=0.25,
        run_seed=seed,
    )
    changed_env0 = assign_policy_ids_per_env(
        both,
        env_indices=[0],
        episode_ids=np.array([5, 9], dtype=np.uint64),
        pool_ids=[5, 8],
        hero_frac=0.25,
        run_seed=seed,
    )
    replay = assign_policy_ids_per_env(
        initial,
        env_indices=[0, 1],
        episode_ids=episode_ids,
        pool_ids=[5, 8],
        hero_frac=0.25,
        run_seed=seed,
    )

    assert np.array_equal(changed_env0[1], both[1])
    assert np.array_equal(replay, both)
    assert both[:, 0].tolist() == [HERO_POLICY_ID, HERO_POLICY_ID]
    assert np.any(both[:, 1:] != HERO_POLICY_ID)


def test_empty_pool_and_empty_selection_have_no_hidden_assignment_dependency() -> None:
    """All-hero and no-op rows are explicit, including empty pool behavior."""
    initial = np.array([[7, 8], [9, 10]], dtype=np.int64)
    episode_ids = np.array([0, 1], dtype=np.uint64)
    untouched = assign_policy_ids_per_env(
        initial,
        env_indices=[],
        episode_ids=episode_ids,
        pool_ids=[],
        hero_frac=0.8,
        run_seed=1,
    )
    all_hero = assign_policy_ids_per_env(
        initial,
        env_indices=[0, 1],
        episode_ids=episode_ids,
        pool_ids=[],
        hero_frac=0.0,
        run_seed=1,
    )

    assert np.array_equal(untouched, initial) and untouched is not initial
    assert np.all(all_hero == HERO_POLICY_ID)


def test_action_streams_are_lane_local_and_replayable() -> None:
    """Extra eligible actions in env 0 cannot advance env 1's generator."""
    seed = 99
    episode_ids = np.array([2, 3], dtype=np.uint64)
    policy_ids = _policy_grid(slots=3)
    masks = np.ones((2, 3, 6), dtype=bool)
    all_eligible = np.ones((2, 3), dtype=bool)
    env0_inactive = all_eligible.copy()
    env0_inactive[0] = False

    full = sample_per_env_exploration(
        policy_ids,
        masks,
        all_eligible,
        1.0,
        _action_rngs(seed, episode_ids),
    )
    env0_short = sample_per_env_exploration(
        policy_ids,
        masks,
        env0_inactive,
        1.0,
        _action_rngs(seed, episode_ids),
    )
    replay = sample_per_env_exploration(
        policy_ids,
        masks,
        all_eligible,
        1.0,
        _action_rngs(seed, episode_ids),
    )

    assert np.array_equal(full.explore[1], env0_short.explore[1])
    assert np.array_equal(full.random_actions[1], env0_short.random_actions[1])
    assert np.array_equal(full.explore, replay.explore)
    assert np.array_equal(full.random_actions, replay.random_actions)


def test_epsilon_zero_and_invalid_request_do_not_advance_action_rngs() -> None:
    """Validation and epsilon zero happen before any persistent RNG advance."""
    policy_ids = _policy_grid()
    masks = np.ones((2, 3, 6), dtype=bool)
    eligible = np.ones((2, 3), dtype=bool)
    episode_ids = np.array([0, 0], dtype=np.uint64)
    rngs = _action_rngs(41, episode_ids)
    before = copy.deepcopy(rngs[1].bit_generator.state)

    zero = sample_per_env_exploration(policy_ids, masks, eligible, 0.0, rngs)
    assert not zero.explore.any()
    assert np.all(zero.random_actions == -1)
    assert rngs[1].bit_generator.state == before

    with pytest.raises(ValueError, match="valid_masks"):
        sample_per_env_exploration(policy_ids, masks.astype(np.uint8), eligible, 1.0, rngs)
    assert rngs[1].bit_generator.state == before


def test_precomputed_decisions_apply_without_shared_rng_or_payload_mutation() -> None:
    """The acting seam accepts a legal precomputed hero action exactly once."""
    hero = _biased_network(1)
    pool = OpponentPool(capacity=0)
    policy_ids = np.array([[HERO_POLICY_ID]], dtype=np.int64)
    mask = torch.ones((1, 1, 6), dtype=torch.bool)
    obs = {
        "tactical": torch.zeros((1, 1, 9, 31, 31)),
        "strategic": torch.zeros((1, 1, 3, 25, 25)),
        "scalars": torch.zeros((1, 1, 26)),
    }
    decisions = PerEnvExplorationDecisions(
        epsilon=1.0,
        eligible=np.array([[True]], dtype=bool),
        explore=np.array([[True]], dtype=bool),
        random_actions=np.array([[4]], dtype=np.int64),
    )
    before = copy.deepcopy(decisions.random_actions)

    actions, _ = batched_act(
        hero, pool, policy_ids, obs, mask, 1.0, None, torch.device("cpu"), decisions
    )

    assert actions.tolist() == [[4]]
    assert np.array_equal(decisions.random_actions, before)


def test_precomputed_decisions_reject_shared_rng_and_illegal_payload_before_forward() -> None:
    """The opt-in path cannot silently fall back to shared RNG or illegal actions."""
    policy_ids = np.array([[HERO_POLICY_ID]], dtype=np.int64)
    decisions = PerEnvExplorationDecisions(
        epsilon=1.0,
        eligible=np.array([[True]], dtype=bool),
        explore=np.array([[True]], dtype=bool),
        random_actions=np.array([[5]], dtype=np.int64),
    )
    mask = torch.tensor([[[True, False, False, False, False, False]]], dtype=torch.bool)
    obs = {
        "tactical": torch.zeros((1, 1, 9, 31, 31)),
        "strategic": torch.zeros((1, 1, 3, 25, 25)),
        "scalars": torch.zeros((1, 1, 26)),
    }

    with pytest.raises(ValueError, match="rng must be None"):
        batched_act(
            _biased_network(0),
            OpponentPool(capacity=0),
            policy_ids,
            obs,
            mask,
            1.0,
            np.random.default_rng(0),
            torch.device("cpu"),
            decisions,
        )
    with pytest.raises(ValueError, match="legal"):
        batched_act(
            _biased_network(0),
            OpponentPool(capacity=0),
            policy_ids,
            obs,
            mask,
            1.0,
            None,
            torch.device("cpu"),
            decisions,
        )


def test_active_pin_count_tracks_real_leases_and_idempotent_close() -> None:
    """B3 can audit cleanup through a public read-only pool surface."""
    pool = PinnedOpponentPool(capacity=1)
    snapshot = pool.add_snapshot(_biased_network(2))
    assert pool.active_pin_count == 0
    lease = pool.acquire([snapshot, snapshot])
    assert pool.active_pin_count == 1
    lease.close()
    lease.close()
    assert pool.active_pin_count == 0
