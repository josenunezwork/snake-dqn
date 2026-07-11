"""Tests for the PQN Q(lambda) trainer + pool self-play (blueprint P3)."""

from __future__ import annotations

import numpy as np
import torch

from src.model.raster_network import (
    SCALARS_DIM,
    STRATEGIC_SHAPE,
    TACTICAL_SHAPE,
    RasterDuelingNetwork,
)
from src.training.pqn_selfplay import (
    HERO_POLICY_ID,
    OpponentPool,
    assign_policy_ids,
    batched_act,
)
from src.training.pqn_trainer import PQNConfig, PQNTrainer, TripwireError, flip_augment


# ---------------------------------------------------------------------------
# Q(lambda) target math on a tiny hand-built rollout
# ---------------------------------------------------------------------------
def _make_trainer(**overrides):
    params = dict(
        num_envs=1,
        num_snakes=1,
        rollout_len=3,
        gamma=0.9,
        lambda_=0.5,
        death_value=-7.0,
    )
    params.update(overrides)
    return PQNTrainer(PQNConfig(**params))


def _zeros_obs(T, E, S):
    return {
        "tactical": torch.zeros((T, E, S, *TACTICAL_SHAPE)),
        "strategic": torch.zeros((T, E, S, *STRATEGIC_SHAPE)),
        "scalars": torch.zeros((T, E, S, SCALARS_DIM)),
    }


def _patch_boot(trainer, boot_values):
    """Force the network's masked-max Q(s') to a known constant per step.

    We monkeypatch the trainer's network forward so ``_compute_targets`` sees a
    deterministic bootstrap. ``boot_values`` is a list length T of the intended
    masked-max value; the net returns that value in EVERY (valid) action slot so
    the masked-max equals it.
    """
    call = {"i": 0}
    # _compute_targets calls network(...) once per step t (t=0..T-1) in order.
    orig = trainer.network

    def fake_forward(tac, strat, scal):
        # returns (N, 6) constant = boot_values[i]
        n = tac.shape[0]
        i = call["i"]
        call["i"] += 1
        val = boot_values[i]
        return torch.full((n, 6), float(val))

    trainer.network = type("Fake", (), {"__call__": staticmethod(fake_forward)})()
    return orig


def test_qlambda_death_is_reward_only():
    """A death transition's return is the reward alone (no bootstrap)."""
    tr = _make_trainer()
    T, E, S = 3, 1, 1
    roll = {
        "tactical": _zeros_obs(T, E, S)["tactical"],
        "strategic": _zeros_obs(T, E, S)["strategic"],
        "scalars": _zeros_obs(T, E, S)["scalars"],
        "rewards": np.array([[[1.0]], [[2.0]], [[5.0]]]),  # (T,E,S)
        "dones": np.array([[[False]], [[False]], [[True]]]),
        "trapped": np.zeros((T, E, S), dtype=bool),
        "valid": np.ones((T, E, S), dtype=bool),
        "next_mask": torch.ones((T, E, S, 6), dtype=torch.bool),
        "final_obs": {
            "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
            "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
            "scalars": torch.zeros((E, S, SCALARS_DIM)),
        },
    }
    _patch_boot(tr, [0.0, 0.0, 0.0])
    targets = tr._compute_targets(roll)
    # Step 2 dies -> G = r = 5.0
    assert targets[2, 0, 0].item() == 5.0


def test_qlambda_trapped_bootstraps_to_death_value():
    """A trapped (non-terminal) transition bootstraps to the DEATH value."""
    tr = _make_trainer()
    T, E, S = 3, 1, 1
    roll = {
        "tactical": _zeros_obs(T, E, S)["tactical"],
        "strategic": _zeros_obs(T, E, S)["strategic"],
        "scalars": _zeros_obs(T, E, S)["scalars"],
        "rewards": np.array([[[1.0]], [[0.0]], [[0.0]]]),
        "dones": np.zeros((T, E, S), dtype=bool),
        "trapped": np.array([[[True]], [[False]], [[False]]]),
        "valid": np.ones((T, E, S), dtype=bool),
        # trapped step -> no valid next action.
        "next_mask": torch.ones((T, E, S, 6), dtype=torch.bool),
        "final_obs": {
            "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
            "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
            "scalars": torch.zeros((E, S, SCALARS_DIM)),
        },
    }
    _patch_boot(tr, [3.0, 3.0, 3.0])
    targets = tr._compute_targets(roll)
    # Step 0 trapped -> G = r + gamma * death_value = 1 + 0.9 * (-7) = -5.3
    assert abs(targets[0, 0, 0].item() - (1.0 + 0.9 * -7.0)) < 1e-5


def test_qlambda_truncation_bootstraps_masked_max():
    """The final rollout step (truncation) bootstraps from masked-max Q(s')."""
    tr = _make_trainer()
    T, E, S = 3, 1, 1
    roll = {
        "tactical": _zeros_obs(T, E, S)["tactical"],
        "strategic": _zeros_obs(T, E, S)["strategic"],
        "scalars": _zeros_obs(T, E, S)["scalars"],
        "rewards": np.array([[[0.0]], [[0.0]], [[1.0]]]),
        "dones": np.zeros((T, E, S), dtype=bool),
        "trapped": np.zeros((T, E, S), dtype=bool),
        "valid": np.ones((T, E, S), dtype=bool),
        "next_mask": torch.ones((T, E, S, 6), dtype=torch.bool),
        "final_obs": {
            "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
            "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
            "scalars": torch.zeros((E, S, SCALARS_DIM)),
        },
    }
    boot = 4.0
    _patch_boot(tr, [boot, boot, boot])
    targets = tr._compute_targets(roll)
    # Last step is truncation: next_G == boot, so
    # G = r + gamma*((1-lam)*boot + lam*boot) = r + gamma*boot.
    expected = 1.0 + 0.9 * boot
    assert abs(targets[2, 0, 0].item() - expected) < 1e-5


def test_qlambda_masked_max_ignores_invalid_actions():
    """Masked-max only considers valid next actions."""
    tr = _make_trainer(rollout_len=1)
    T, E, S = 1, 1, 1
    # Make the net return distinct per-action Q so the mask matters.
    q_row = torch.tensor([[10.0, -1.0, -1.0, -1.0, -1.0, -1.0]])

    def fake_forward(tac, strat, scal):
        n = tac.shape[0]
        return q_row.repeat(n, 1)

    tr.network = type("Fake", (), {"__call__": staticmethod(fake_forward)})()
    # Mask out action 0 (the max); valid actions all == -1.
    mask = torch.zeros((T, E, S, 6), dtype=torch.bool)
    mask[..., 1:] = True
    roll = {
        "tactical": _zeros_obs(T, E, S)["tactical"],
        "strategic": _zeros_obs(T, E, S)["strategic"],
        "scalars": _zeros_obs(T, E, S)["scalars"],
        "rewards": np.array([[[0.0]]]),
        "dones": np.zeros((T, E, S), dtype=bool),
        "trapped": np.zeros((T, E, S), dtype=bool),
        "valid": np.ones((T, E, S), dtype=bool),
        "next_mask": mask,
        "final_obs": {
            "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
            "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
            "scalars": torch.zeros((E, S, SCALARS_DIM)),
        },
    }
    targets = tr._compute_targets(roll)
    # masked-max = -1 (action 0 excluded). G = 0 + 0.9 * -1 = -0.9.
    assert abs(targets[0, 0, 0].item() - (0.9 * -1.0)) < 1e-5


def test_qlambda_interior_recursion():
    """An interior alive step mixes bootstrap and forward return by lambda."""
    tr = _make_trainer()  # gamma=0.9, lambda=0.5
    T, E, S = 3, 1, 1
    roll = {
        "tactical": _zeros_obs(T, E, S)["tactical"],
        "strategic": _zeros_obs(T, E, S)["strategic"],
        "scalars": _zeros_obs(T, E, S)["scalars"],
        "rewards": np.array([[[1.0]], [[1.0]], [[1.0]]]),
        "dones": np.zeros((T, E, S), dtype=bool),
        "trapped": np.zeros((T, E, S), dtype=bool),
        "valid": np.ones((T, E, S), dtype=bool),
        "next_mask": torch.ones((T, E, S, 6), dtype=torch.bool),
        "final_obs": {
            "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
            "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
            "scalars": torch.zeros((E, S, SCALARS_DIM)),
        },
    }
    boot = 2.0
    _patch_boot(tr, [boot, boot, boot])
    targets = tr._compute_targets(roll)
    g, lam = 0.9, 0.5
    # Backward: G2 = 1 + g*boot (truncation) = 1 + 1.8 = 2.8
    g2 = 1.0 + g * boot
    # G1 = 1 + g*((1-lam)*boot + lam*G2)
    g1 = 1.0 + g * ((1 - lam) * boot + lam * g2)
    # G0 = 1 + g*((1-lam)*boot + lam*G1)
    g0 = 1.0 + g * ((1 - lam) * boot + lam * g1)
    assert abs(targets[2, 0, 0].item() - g2) < 1e-5
    assert abs(targets[1, 0, 0].item() - g1) < 1e-5
    assert abs(targets[0, 0, 0].item() - g0) < 1e-5


def test_qlambda_death_return_propagates_one_step_back():
    """A mid-rollout death's terminal return flows one step back via lambda.

    Step 1 dies (G1 = r1, reward alone). Step 1 is a REAL transition, so step 0
    mixes G1 into its lambda channel (blueprint per-agent backward scan). Step 2
    is a post-death zombie (``valid`` False): its bogus return must NOT seed the
    carry, and it is excluded from training.
    """
    tr = _make_trainer()
    T, E, S = 3, 1, 1
    roll = {
        "tactical": _zeros_obs(T, E, S)["tactical"],
        "strategic": _zeros_obs(T, E, S)["strategic"],
        "scalars": _zeros_obs(T, E, S)["scalars"],
        "rewards": np.array([[[1.0]], [[9.0]], [[1.0]]]),
        # Step 1 is a death; step 2 is a post-death zombie.
        "dones": np.array([[[False]], [[True]], [[False]]]),
        "trapped": np.zeros((T, E, S), dtype=bool),
        # Alive at entry for steps 0 and 1 (death step included); zombie at 2.
        "valid": np.array([[[True]], [[True]], [[False]]]),
        "next_mask": torch.ones((T, E, S, 6), dtype=torch.bool),
        "final_obs": {
            "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
            "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
            "scalars": torch.zeros((E, S, SCALARS_DIM)),
        },
    }
    boot = 2.0
    _patch_boot(tr, [boot, boot, boot])
    targets = tr._compute_targets(roll)
    g, lam = 0.9, 0.5
    # Step 1 death -> G1 = r1 = 9.0 (reward alone).
    g1 = 9.0
    assert abs(targets[1, 0, 0].item() - g1) < 1e-5
    # Step 0: successor (step 1) is a REAL death transition -> carry G1 via lambda.
    # G0 = r0 + g*((1-lam)*boot + lam*G1).
    g0 = 1.0 + g * ((1 - lam) * boot + lam * g1)
    assert abs(targets[0, 0, 0].item() - g0) < 1e-5


def test_sgd_excludes_zombie_hero_steps():
    """Post-death (``valid`` False) hero steps are dropped from the loss.

    Build a 1-slot hero rollout where step 0 is alive and steps 1-2 are zombie
    steps (dead-at-entry). The hero index set that :meth:`_sgd` trains on must
    contain only the alive-at-entry step, so exactly one hero transition trains
    regardless of ``minibatch_size``.
    """
    tr = _make_trainer(rollout_len=3, minibatches=1, minibatch_size=64, flip_augment=False)
    T, E, S = 3, 1, 1
    roll = {
        "tactical": torch.zeros((T, E, S, *TACTICAL_SHAPE)),
        "strategic": torch.zeros((T, E, S, *STRATEGIC_SHAPE)),
        "scalars": torch.zeros((T, E, S, SCALARS_DIM)),
        "actions": np.zeros((T, E, S), dtype=np.int64),
        # Slot is the hero for all t, but alive only at step 0.
        "policy_ids": np.array([[HERO_POLICY_ID]]),
        "valid": np.array([[[True]], [[False]], [[False]]]),
    }
    hero_es = roll["policy_ids"] == HERO_POLICY_ID
    hero_mask_tes = (np.broadcast_to(hero_es[None], (T, E, S)) & roll["valid"]).reshape(-1)
    assert int(hero_mask_tes.sum()) == 1  # only the alive-at-entry step trains
    targets = torch.zeros((T, E, S), dtype=torch.float32)
    # _sgd must run without touching the two zombie steps and stay finite.
    loss, gnorm, mean_abs_q, max_abs_q, entropy = tr._sgd(roll, targets)
    assert np.isfinite(loss) and np.isfinite(gnorm) and np.isfinite(max_abs_q)


def test_trapped_default_death_value_is_death_reward():
    """The default ``death_value`` equals the sim's terminal DEATH_REWARD."""
    from src.core.reward_events import DEATH_REWARD

    tr = PQNTrainer(PQNConfig(num_envs=1, num_snakes=1, rollout_len=1))
    assert tr.cfg.death_value == DEATH_REWARD
    T, E, S = 1, 1, 1
    roll = {
        "tactical": torch.zeros((T, E, S, *TACTICAL_SHAPE)),
        "strategic": torch.zeros((T, E, S, *STRATEGIC_SHAPE)),
        "scalars": torch.zeros((T, E, S, SCALARS_DIM)),
        "rewards": np.array([[[0.5]]]),
        "dones": np.zeros((T, E, S), dtype=bool),
        "trapped": np.array([[[True]]]),
        "valid": np.ones((T, E, S), dtype=bool),
        "next_mask": torch.zeros((T, E, S, 6), dtype=torch.bool),  # no valid next
        "final_obs": {
            "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
            "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
            "scalars": torch.zeros((E, S, SCALARS_DIM)),
        },
    }
    _patch_boot(tr, [0.0])
    targets = tr._compute_targets(roll)
    # G = r + gamma * DEATH_REWARD, reflecting the impending forced death.
    expected = 0.5 + tr.cfg.gamma * DEATH_REWARD
    assert abs(targets[0, 0, 0].item() - expected) < 1e-5


# ---------------------------------------------------------------------------
# Flip augmentation
# ---------------------------------------------------------------------------
def test_flip_action_swap():
    """Flip remaps actions 0<->2 and 3<->5, leaves 1 and 4."""
    tac = torch.zeros((6, *TACTICAL_SHAPE))
    strat = torch.zeros((6, *STRATEGIC_SHAPE))
    scal = torch.zeros((6, SCALARS_DIM))
    actions = torch.tensor([0, 1, 2, 3, 4, 5])
    _, _, _, flipped = flip_augment(tac, strat, scal, actions)
    assert flipped.tolist() == [2, 1, 0, 5, 4, 3]


def test_flip_columns_and_lateral_scalars():
    """Flip mirrors raster columns, swaps wall L/R, negates lateral scalars."""
    tac = torch.arange(9 * 31 * 31, dtype=torch.float32).reshape(1, *TACTICAL_SHAPE)
    strat = torch.arange(3 * 25 * 25, dtype=torch.float32).reshape(1, *STRATEGIC_SHAPE)
    scal = torch.zeros((1, SCALARS_DIM))
    # Wall right (9) and left (11) distinct; lateral scalar (12) nonzero.
    scal[0, 9] = 0.3
    scal[0, 11] = 0.7
    scal[0, 12] = 0.5
    scal[0, 15] = -0.2
    ftac, fstrat, fscal, _ = flip_augment(tac, strat, scal, torch.tensor([1]))
    # Columns (last axis) reversed.
    assert torch.equal(ftac, torch.flip(tac, dims=[-1]))
    assert torch.equal(fstrat, torch.flip(strat, dims=[-1]))
    # Wall L/R swapped.
    assert abs(fscal[0, 9].item() - 0.7) < 1e-6
    assert abs(fscal[0, 11].item() - 0.3) < 1e-6
    # Lateral scalars negated.
    assert abs(fscal[0, 12].item() - -0.5) < 1e-6
    assert abs(fscal[0, 15].item() - 0.2) < 1e-6


def test_flip_is_involution_on_raster():
    """Flipping twice restores the raster and actions."""
    tac = torch.randn(4, *TACTICAL_SHAPE)
    strat = torch.randn(4, *STRATEGIC_SHAPE)
    scal = torch.randn(4, SCALARS_DIM)
    actions = torch.tensor([0, 3, 2, 5])
    t1, g1, s1, a1 = flip_augment(tac, strat, scal, actions)
    t2, g2, s2, a2 = flip_augment(t1, g1, s1, a1)
    assert torch.allclose(t2, tac)
    assert torch.allclose(g2, strat)
    assert torch.allclose(s2, scal)
    assert torch.equal(a2, actions)


# ---------------------------------------------------------------------------
# Self-play: K+1 batched forward
# ---------------------------------------------------------------------------
def test_assign_policy_ids_empty_pool_all_hero():
    ids = assign_policy_ids(3, 4, [], hero_frac=0.8, rng=np.random.default_rng(0))
    assert np.all(ids == HERO_POLICY_ID)


def test_assign_policy_ids_slot0_forced_hero():
    ids = assign_policy_ids(
        3, 4, [0, 1], hero_frac=0.0, rng=np.random.default_rng(0), hero_slot0=True
    )
    assert np.all(ids[:, 0] == HERO_POLICY_ID)
    # With hero_frac=0 and non-empty pool, the other slots are frozen ids.
    assert np.all(ids[:, 1:] >= 0)


def test_batched_act_2policy_pool_correct_per_slot():
    """K+1 forward routes each slot through the net its policy_id names.

    Build a hero and one frozen net with hand-set weights so each returns a
    distinct constant-argmax action; verify per-slot actions match the policy
    assignment (greedy, epsilon=0 so no exploration noise).
    """
    device = torch.device("cpu")
    hero = RasterDuelingNetwork().eval()
    pool = OpponentPool(capacity=2, device=device)
    pool.add_snapshot(hero)  # frozen id 0 (a copy of hero at this point)

    # Distinguish hero from frozen: bias the hero's advantage output toward
    # action 5, the frozen (already snapshotted) stays at its init argmax.
    with torch.no_grad():
        hero.advantage_stream[-1].bias.zero_()
        hero.advantage_stream[-1].bias[5] = 100.0
        hero.value_stream[-1].bias.zero_()

    E, S = 1, 3
    # Slot 0 hero, slot 1 frozen(0), slot 2 hero.
    policy_ids = np.array([[HERO_POLICY_ID, 0, HERO_POLICY_ID]])
    obs = {
        "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
        "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
        "scalars": torch.zeros((E, S, SCALARS_DIM)),
    }
    mask = torch.ones((E, S, 6), dtype=torch.bool)
    actions, hero_q = batched_act(
        hero,
        pool,
        policy_ids,
        obs,
        mask,
        epsilon=0.0,
        rng=np.random.default_rng(0),
        device=device,
    )
    assert actions.shape == (E, S)
    assert hero_q.shape == (E, S, 6)
    # Hero slots pick action 5 (biased). Frozen slot != 5 (frozen kept init).
    assert actions[0, 0] == 5
    assert actions[0, 2] == 5
    assert actions[0, 1] != 5


def test_batched_act_multi_frozen_matches_per_slot_reference():
    """Deferred-sync frozen path routes every slot through the net its id names.

    Exercises >1 distinct frozen policy id (the path the single-sync accumulation
    optimizes) and checks the chosen action for every slot against an independent
    per-slot greedy-masked recomputation. epsilon=0 makes hero slots greedy too,
    so the whole output is deterministic and must match exactly.
    """
    from src.training.pqn_selfplay import _greedy_masked_actions

    device = torch.device("cpu")
    hero = RasterDuelingNetwork().eval()
    pool = OpponentPool(capacity=4, device=device)
    torch.manual_seed(0)
    for _ in range(3):  # frozen ids 0,1,2, each a distinct perturbed snapshot
        with torch.no_grad():
            for p in hero.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        pool.add_snapshot(hero)

    E, S = 2, 6
    policy_ids = np.array(
        [HERO_POLICY_ID, 0, 1, 2, 0, HERO_POLICY_ID, 1, 2, HERO_POLICY_ID, 0, 1, 2]
    ).reshape(E, S)
    torch.manual_seed(1)
    obs = {
        "tactical": torch.rand((E, S, *TACTICAL_SHAPE)),
        "strategic": torch.rand((E, S, *STRATEGIC_SHAPE)),
        "scalars": torch.rand((E, S, SCALARS_DIM)),
    }
    mask = torch.ones((E, S, 6), dtype=torch.bool)
    mask[0, 1, 0] = False  # a couple masked bits; valid actions still remain
    mask[1, 3, 2] = False

    actions, _ = batched_act(
        hero, pool, policy_ids, obs, mask, epsilon=0.0,
        rng=np.random.default_rng(0), device=device,
    )

    # Independent per-slot reference: run each slot through its named net.
    flat_ids = policy_ids.reshape(-1)
    tac = obs["tactical"].reshape(E * S, *hero.tactical_shape)
    strat = obs["strategic"].reshape(E * S, *hero.strategic_shape)
    scal = obs["scalars"].reshape(E * S, hero.scalars_dim)
    mask_flat = mask.reshape(E * S, 6)
    ref = np.zeros(E * S, dtype=np.int64)
    for i in range(E * S):
        pid = int(flat_ids[i])
        net = hero if pid == HERO_POLICY_ID else pool.get(pid)
        with torch.no_grad():
            q = net(tac[i : i + 1], strat[i : i + 1], scal[i : i + 1])
        ref[i] = int(_greedy_masked_actions(q, mask_flat[i : i + 1]).item())

    assert np.array_equal(actions.reshape(-1), ref)


def test_batched_act_respects_mask():
    """Greedy acting never picks a masked-out action for hero slots."""
    device = torch.device("cpu")
    hero = RasterDuelingNetwork().eval()
    with torch.no_grad():
        hero.advantage_stream[-1].bias.zero_()
        hero.advantage_stream[-1].bias[0] = 100.0  # would pick action 0
    pool = OpponentPool(capacity=0, device=device)
    E, S = 1, 1
    policy_ids = np.array([[HERO_POLICY_ID]])
    obs = {
        "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
        "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
        "scalars": torch.zeros((E, S, SCALARS_DIM)),
    }
    mask = torch.ones((E, S, 6), dtype=torch.bool)
    mask[..., 0] = False  # forbid the biased action
    actions, _ = batched_act(
        hero,
        pool,
        policy_ids,
        obs,
        mask,
        epsilon=0.0,
        rng=np.random.default_rng(0),
        device=device,
    )
    assert actions[0, 0] != 0


def test_pool_fifo_eviction_and_no_grad():
    hero = RasterDuelingNetwork()
    pool = OpponentPool(capacity=2)
    for _ in range(3):
        pool.add_snapshot(hero)
    assert len(pool) == 2
    # Frozen nets carry no grad.
    for pid in pool.policy_ids():
        net = pool.get(pid)
        assert all(not p.requires_grad for p in net.parameters())


# ---------------------------------------------------------------------------
# End-to-end smoke: learns a trivial signal without NaN
# ---------------------------------------------------------------------------
def test_smoke_end_to_end_finite_and_learns():
    cfg = PQNConfig(
        num_envs=2,
        num_snakes=4,
        rollout_len=8,
        minibatches=3,
        minibatch_size=48,
        eps_start=0.5,
        eps_end=0.02,
        eps_decay_steps=1500,
        pool_capacity=3,
        pool_add_interval=5,
        seed=0,
    )
    tr = PQNTrainer(cfg)
    tels = tr.train(30)
    assert all(np.isfinite(t.loss) for t in tels)
    assert all(np.isfinite(t.max_abs_q) for t in tels)
    # Loss should trend down: mean of the last third below the first third.
    losses = np.array([t.loss for t in tels])
    first = losses[:10].mean()
    last = losses[-10:].mean()
    assert last < first, f"loss did not decrease: first={first:.4f} last={last:.4f}"
    # Agent-step accounting advanced.
    assert tr.agent_steps > 0


def test_tripwire_fires_on_nonfinite():
    tr = _make_trainer(rollout_len=2)
    from src.training.pqn_trainer import PQNTelemetry

    tel = PQNTelemetry(
        update=0,
        agent_steps=0,
        loss=float("nan"),
        grad_norm=1.0,
        mean_abs_q=1.0,
        max_abs_q=1.0,
        epsilon=0.1,
        mean_reward=0.0,
        action_entropy=1.0,
        kills_per_ep=0.0,
        boost_fraction=0.0,
        pool_size=0,
    )
    try:
        tr._check_tripwires(tel)
        assert False, "expected TripwireError"
    except TripwireError:
        pass


def test_checkpoint_metadata():
    tr = _make_trainer()
    state = tr.checkpoint_state()
    assert state["obs_spec"] == "raster31v2"
    assert state["algo"] == "pqn"
    # Canonical flat shape keys (coordinated with src.model.obs_spec).
    assert state["tactical_channels"] == TACTICAL_SHAPE[0]
    assert state["tactical_size"] == TACTICAL_SHAPE[1]
    assert state["strategic_channels"] == STRATEGIC_SHAPE[0]
    assert state["strategic_size"] == STRATEGIC_SHAPE[1]
    assert state["scalars"] == SCALARS_DIM
    assert state["gamma"] == tr.cfg.gamma
    assert state["lambda"] == tr.cfg.lambda_
    assert "dqn_state_dict" in state


def test_checkpoint_loads_via_inference_agent(tmp_path):
    """A PQN checkpoint round-trips through InferenceAgent.from_checkpoint."""
    from src.model.inference_agent import InferenceAgent

    tr = _make_trainer()
    path = str(tmp_path / "pqn.pth")
    tr.save_checkpoint(path)
    agent = InferenceAgent.from_checkpoint(path, device=torch.device("cpu"))
    assert agent.obs_spec == "raster31v2"
    assert agent.output_size == 6
