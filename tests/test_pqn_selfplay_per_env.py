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


def _counting_network(action: int) -> tuple[RasterDuelingNetwork, dict[str, int]]:
    """Return a real network with a forward-call counter for acting oracles."""
    network = _biased_network(action)
    calls = {"count": 0}
    original_forward = network.forward

    def counted_forward(*args):
        calls["count"] += 1
        return original_forward(*args)

    network.forward = counted_forward  # type: ignore[method-assign]
    return network, calls


class _OneFrozenPool:
    """Small PolicyGetter fixture whose net retains its forward-call counter."""

    def __init__(self, network: RasterDuelingNetwork) -> None:
        self.network = network

    def get(self, policy_id: int) -> RasterDuelingNetwork:
        assert policy_id == 0
        return self.network


def test_selected_assignment_rows_are_independent_and_replayable() -> None:
    """Changing env 0's episode cannot redraw continuing env 1's policy row."""
    seed = 73
    episode_ids = np.array([4, 9], dtype=np.uint64)
    initial = _policy_grid()
    initial_before = initial.copy()
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
    assert np.array_equal(initial, initial_before)
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


def test_sampling_does_not_mutate_caller_payloads_or_rng_mapping() -> None:
    """Only generator state advances; caller arrays and lane map stay owned by B3."""
    policy_ids = _policy_grid()
    masks = np.ones((2, 3, 6), dtype=bool)
    eligible = np.ones((2, 3), dtype=bool)
    episode_ids = np.array([0, 1], dtype=np.uint64)
    rngs = _action_rngs(33, episode_ids)
    mapping_ids = {index: id(generator) for index, generator in rngs.items()}
    before_policy_ids, before_masks, before_eligible = (
        policy_ids.copy(),
        masks.copy(),
        eligible.copy(),
    )

    sample_per_env_exploration(policy_ids, masks, eligible, 1.0, rngs)

    assert np.array_equal(policy_ids, before_policy_ids)
    assert np.array_equal(masks, before_masks)
    assert np.array_equal(eligible, before_eligible)
    assert {index: id(generator) for index, generator in rngs.items()} == mapping_ids


def test_effective_eligibility_excludes_unselected_and_frozen_slots() -> None:
    """Endpoint rules apply only to rows this helper is allowed to explore."""
    policy_ids = np.array([[HERO_POLICY_ID, 4], [HERO_POLICY_ID, HERO_POLICY_ID]], dtype=np.int64)
    decisions = sample_per_env_exploration(
        policy_ids,
        np.ones((2, 2, 6), dtype=bool),
        np.ones((2, 2), dtype=bool),
        1.0,
        _action_rngs(9, np.array([0, 0], dtype=np.uint64)),
        env_indices=[0],
    )

    assert decisions.eligible.tolist() == [[True, False], [False, False]]
    assert np.array_equal(decisions.explore, decisions.eligible)
    assert np.all(decisions.random_actions[~decisions.explore] == -1)


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


def test_alias_action_rngs_and_negative_pool_ids_fail_before_any_draw() -> None:
    """Lane-local streams cannot alias and frozen ids cannot use Python indexing."""
    policy_ids = _policy_grid()
    masks = np.ones((2, 3, 6), dtype=bool)
    eligible = np.ones((2, 3), dtype=bool)
    shared = np.random.default_rng(12)
    before = copy.deepcopy(shared.bit_generator.state)

    with pytest.raises(ValueError, match="share Generator"):
        sample_per_env_exploration(
            policy_ids,
            masks,
            eligible,
            1.0,
            {0: shared, 1: shared},
        )
    assert shared.bit_generator.state == before

    bit_generator = np.random.PCG64(44)
    first_wrapper = np.random.Generator(bit_generator)
    second_wrapper = np.random.Generator(bit_generator)
    wrapped_before = copy.deepcopy(bit_generator.state)
    with pytest.raises(ValueError, match="bit-generator"):
        sample_per_env_exploration(
            policy_ids,
            masks,
            eligible,
            1.0,
            {0: first_wrapper, 1: second_wrapper},
        )
    assert bit_generator.state == wrapped_before

    first = np.random.default_rng(45)
    second = np.random.default_rng(46)
    first_before = copy.deepcopy(first.bit_generator.state)
    with pytest.raises(ValueError, match="keys must be integer"):
        sample_per_env_exploration(
            policy_ids,
            masks,
            eligible,
            1.0,
            {False: first, True: second},
        )
    assert first.bit_generator.state == first_before
    with pytest.raises(ValueError, match="keys must be integer"):
        sample_per_env_exploration(
            policy_ids,
            masks,
            eligible,
            1.0,
            {0.0: first, 1.0: second},
        )
    assert first.bit_generator.state == first_before

    with pytest.raises(ValueError, match="fit the policy_ids"):
        assign_policy_ids_per_env(
            policy_ids,
            env_indices=[0],
            episode_ids=np.array([0, 0], dtype=np.uint64),
            pool_ids=[-2],
            hero_frac=0.0,
            run_seed=12,
        )
    with pytest.raises(ValueError, match="fit the policy_ids"):
        assign_policy_ids_per_env(
            policy_ids,
            env_indices=[0],
            episode_ids=np.array([0, 0], dtype=np.uint64),
            pool_ids=[2**63],
            hero_frac=0.0,
            run_seed=12,
        )
    with pytest.raises(ValueError, match="fit the policy_ids"):
        assign_policy_ids_per_env(
            np.full((1, 2), HERO_POLICY_ID, dtype=np.int8),
            env_indices=[0],
            episode_ids=np.array([0], dtype=np.uint64),
            pool_ids=[200],
            hero_frac=0.0,
            run_seed=12,
        )

    with pytest.raises(ValueError, match="HERO_POLICY_ID"):
        batched_act(
            object(),
            object(),
            np.array([[-2]], dtype=np.int64),
            {},
            torch.ones((1, 1, 6), dtype=torch.bool),
            0.0,
            np.random.default_rng(0),
            torch.device("cpu"),
        )


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
    before_decisions = (
        decisions.eligible.copy(),
        decisions.explore.copy(),
        decisions.random_actions.copy(),
    )
    before_policy_ids = policy_ids.copy()
    before_mask = mask.clone()
    before_obs = {name: value.clone() for name, value in obs.items()}

    actions, _ = batched_act(
        hero, pool, policy_ids, obs, mask, 1.0, None, torch.device("cpu"), decisions
    )

    assert actions.tolist() == [[4]]
    assert np.array_equal(decisions.eligible, before_decisions[0])
    assert np.array_equal(decisions.explore, before_decisions[1])
    assert np.array_equal(decisions.random_actions, before_decisions[2])
    assert np.array_equal(policy_ids, before_policy_ids)
    assert torch.equal(mask, before_mask)
    assert all(torch.equal(obs[name], before_obs[name]) for name in obs)


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


def test_zero_epsilon_rejects_precomputed_exploration_before_hero_forward() -> None:
    """A forged zero-epsilon payload cannot override the greedy action."""
    decisions = PerEnvExplorationDecisions(
        epsilon=0.0,
        eligible=np.array([[True]], dtype=bool),
        explore=np.array([[True]], dtype=bool),
        random_actions=np.array([[4]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="epsilon is zero"):
        batched_act(
            object(),
            object(),
            np.array([[HERO_POLICY_ID]], dtype=np.int64),
            {},
            torch.ones((1, 1, 6), dtype=torch.bool),
            0.0,
            None,
            torch.device("cpu"),
            decisions,
        )


def test_epsilon_one_rejects_incomplete_precomputed_exploration_before_hero_forward() -> None:
    """A forged endpoint payload cannot silently leave eligible rows greedy."""
    decisions = PerEnvExplorationDecisions(
        epsilon=1.0,
        eligible=np.array([[True]], dtype=bool),
        explore=np.array([[False]], dtype=bool),
        random_actions=np.array([[-1]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="epsilon-one"):
        batched_act(
            object(),
            object(),
            np.array([[HERO_POLICY_ID]], dtype=np.int64),
            {},
            torch.ones((1, 1, 6), dtype=torch.bool),
            1.0,
            None,
            torch.device("cpu"),
            decisions,
        )


@pytest.mark.parametrize(
    ("policy_id", "epsilon", "decision_epsilon", "eligible", "explore", "random_actions", "match"),
    [
        (HERO_POLICY_ID, 0.4, 0.5, [[True]], [[False]], [[-1]], "epsilon must match"),
        (HERO_POLICY_ID, 0.4, 0.4, [[1]], [[False]], [[-1]], "eligible must be a bool"),
        (HERO_POLICY_ID, 0.4, 0.4, [[True]], [[False, False]], [[-1]], "explore must be a bool"),
        (HERO_POLICY_ID, 0.4, 0.4, [[True]], [[False]], [[0]], "-1 sentinel"),
        (HERO_POLICY_ID, 0.4, 0.4, [[True]], [[True]], [[6]], "action range"),
        (0, 0.4, 0.4, [[True]], [[False]], [[-1]], "only name hero"),
    ],
)
def test_precomputed_payload_contract_rejects_all_invalid_shapes_and_membership(
    policy_id: int,
    epsilon: float,
    decision_epsilon: float,
    eligible: list[list[int]],
    explore: list[list[int]],
    random_actions: list[list[int]],
    match: str,
) -> None:
    """Invalid precomputed payloads fail before accessing a hero network."""
    decisions = PerEnvExplorationDecisions(
        epsilon=decision_epsilon,
        eligible=np.asarray(eligible, dtype=np.uint8 if match.startswith("eligible") else bool),
        explore=np.asarray(explore, dtype=bool),
        random_actions=np.asarray(random_actions, dtype=np.int64),
    )
    with pytest.raises(ValueError, match=match):
        batched_act(
            object(),
            object(),
            np.array([[policy_id]], dtype=np.int64),
            {},
            torch.ones((1, 1, 6), dtype=torch.bool),
            epsilon,
            None,
            torch.device("cpu"),
            decisions,
        )


def test_legacy_nonzero_epsilon_draw_order_actions_and_forwards_are_unchanged() -> None:
    """The default path keeps the original shared-RNG stream and grouped forwards."""
    hero, hero_calls = _counting_network(1)
    frozen, frozen_calls = _counting_network(4)
    policy_ids = np.array([[HERO_POLICY_ID, 0, HERO_POLICY_ID]], dtype=np.int64)
    mask = torch.tensor(
        [
            [
                [True, False, True, False, True, False],
                [True] * 6,
                [False, True, False, True, False, False],
            ]
        ],
        dtype=torch.bool,
    )
    obs = {
        "tactical": torch.zeros((1, 3, 9, 31, 31)),
        "strategic": torch.zeros((1, 3, 3, 25, 25)),
        "scalars": torch.zeros((1, 3, 26)),
    }
    epsilon = 0.8
    rng = np.random.default_rng(7)
    reference = np.random.default_rng(7)
    hero_indices = np.array([0, 2])
    explore = reference.random(len(hero_indices)) < epsilon
    assert explore.tolist() == [True, False]
    random_actions = np.empty(len(hero_indices), dtype=np.int64)
    if np.any(explore):
        for index, flat_index in enumerate(hero_indices):
            options = np.flatnonzero(mask.reshape(-1, 6)[flat_index].numpy())
            random_actions[index] = (
                int(options[reference.integers(options.size)])
                if options.size
                else int(reference.integers(6))
            )
    # The hero's biased action 1 is masked out in slot 0, so masked greedy
    # falls back to the first legal tied action 0 there.
    expected = np.array([[0, 4, 1]], dtype=np.int64)
    expected.reshape(-1)[hero_indices[explore]] = random_actions[explore]

    actions, _ = batched_act(
        hero,
        _OneFrozenPool(frozen),
        policy_ids,
        obs,
        mask,
        epsilon,
        rng,
        torch.device("cpu"),
    )

    assert np.array_equal(actions, expected)
    assert rng.bit_generator.state == reference.bit_generator.state
    assert hero_calls["count"] == 1
    assert frozen_calls["count"] == 1


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
