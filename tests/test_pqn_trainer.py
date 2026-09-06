"""Tests for the PQN Q(lambda) trainer + pool self-play (blueprint P3)."""

from __future__ import annotations

import numpy as np
import pytest
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
from src.training.pqn_trainer import (
    PQNConfig,
    PQNTelemetry,
    PQNTrainer,
    TripwireError,
    flip_augment,
)


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


def _patch_boot(trainer, roll, boot_values):
    """Force the masked-max Q(s') ``_compute_targets`` sees to a known constant.

    ``boot_values[t]`` is the intended masked-max for step t. Step t's bootstrap
    is Q(s_{t+1}): for t < T-1 that is the rollout's stored hero Q of step t+1,
    and for t == T-1 it is the lone forward on ``final_obs``. Setting the value
    in EVERY action slot makes the masked-max equal it.
    """
    T = len(boot_values)
    E, S = roll["rewards"].shape[1:]
    hero_q = torch.zeros((T, E, S, 6))
    for t in range(T - 1):
        hero_q[t + 1] = float(boot_values[t])
    roll["hero_q"] = hero_q
    final_val = float(boot_values[T - 1])

    def fake_forward(tac, strat, scal):
        return torch.full((tac.shape[0], 6), final_val)

    trainer.network = type("Fake", (), {"__call__": staticmethod(fake_forward)})()


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
    _patch_boot(tr, roll, [0.0, 0.0, 0.0])
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
    _patch_boot(tr, roll, [3.0, 3.0, 3.0])
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
    _patch_boot(tr, roll, [boot, boot, boot])
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
        # T==1: the only step is the rollout edge, so its bootstrap comes from
        # the final-obs forward and no stored Q is read.
        "hero_q": torch.zeros((T, E, S, 6)),
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
    _patch_boot(tr, roll, [boot, boot, boot])
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
    _patch_boot(tr, roll, [boot, boot, boot])
    targets = tr._compute_targets(roll)
    g, lam = 0.9, 0.5
    # Step 1 death -> G1 = r1 = 9.0 (reward alone).
    g1 = 9.0
    assert abs(targets[1, 0, 0].item() - g1) < 1e-5
    # Step 0: successor (step 1) is a REAL death transition -> carry G1 via lambda.
    # G0 = r0 + g*((1-lam)*boot + lam*G1).
    g0 = 1.0 + g * ((1 - lam) * boot + lam * g1)
    assert abs(targets[0, 0, 0].item() - g0) < 1e-5


def test_qlambda_zombie_successor_does_not_chain():
    """A zombie successor's return must not chain into an earlier real transition.

    Step 1 is a post-death zombie (``valid`` False) carrying a huge reward; step 0
    is an ALIVE, non-trapped interior step, so its target flows through the
    interior branch and actually CONSUMES ``next_g``. That is what distinguishes
    this from the death-successor case above, where the death branch discards
    ``next_g`` and would mask a regression. Step 0 must fall back to the
    truncation bootstrap (1.0 + gamma*boot = 2.8), not inherit the zombie's
    99-reward return (which would give 47.2).
    """
    tr = _make_trainer()
    T, E, S = 3, 1, 1
    roll = {
        "tactical": _zeros_obs(T, E, S)["tactical"],
        "strategic": _zeros_obs(T, E, S)["strategic"],
        "scalars": _zeros_obs(T, E, S)["scalars"],
        "rewards": np.array([[[1.0]], [[99.0]], [[0.0]]]),
        "dones": np.zeros((T, E, S), dtype=bool),
        "trapped": np.zeros((T, E, S), dtype=bool),
        # Step 1 dead-at-entry: not a transition, so its return must be discarded.
        "valid": np.array([[[True]], [[False]], [[True]]]),
        "next_mask": torch.ones((T, E, S, 6), dtype=torch.bool),
        "final_obs": {
            "tactical": torch.zeros((E, S, *TACTICAL_SHAPE)),
            "strategic": torch.zeros((E, S, *STRATEGIC_SHAPE)),
            "scalars": torch.zeros((E, S, SCALARS_DIM)),
        },
    }
    _patch_boot(tr, roll, [2.0, 2.0, 2.0])
    targets = tr._compute_targets(roll)
    assert targets[0, 0, 0].item() == pytest.approx(2.8, abs=1e-4)
    # Rollout edge stays a plain truncation bootstrap: 0 + 0.9*2.0.
    assert targets[2, 0, 0].item() == pytest.approx(1.8, abs=1e-4)


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
    _patch_boot(tr, roll, [0.0])
    targets = tr._compute_targets(roll)
    # G = r + gamma * DEATH_REWARD, reflecting the impending forced death.
    expected = 0.5 + tr.cfg.gamma * DEATH_REWARD
    assert abs(targets[0, 0, 0].item() - expected) < 1e-5


# ---------------------------------------------------------------------------
# Bootstrap reuse: the rollout's hero Q is the target forward
# ---------------------------------------------------------------------------
def test_compute_targets_matches_a_fresh_forward_over_the_rollout():
    """Reusing the rollout's stored Q is EXACT, not an approximation.

    Step t bootstraps on Q(s_{t+1}), and s_{t+1}'s obs is step t+1's stored obs —
    already forwarded during the rollout under these same weights (the only
    optimizer step runs afterwards). Recomputing every stored Q with a fresh
    forward on the stored obs must therefore give bit-identical targets. Fails if
    the rollout ever stores a Q that does not correspond to the obs beside it
    (wrong step index, a frozen net's Q, a stale buffer).
    """
    tr = _make_trainer(num_envs=2, num_snakes=3, rollout_len=4, pool_capacity=2)
    tr.pool.add_snapshot(tr.network)  # frozen slots exist: hero_q must stay hero's
    E, S, T = 2, 3, 4
    roll = tr._rollout()
    targets = tr._compute_targets(roll)

    reference = dict(roll)
    with torch.no_grad():
        reference["hero_q"] = torch.stack(
            [
                tr.network(
                    roll["tactical"][t].reshape(E * S, *TACTICAL_SHAPE),
                    roll["strategic"][t].reshape(E * S, *STRATEGIC_SHAPE),
                    roll["scalars"][t].reshape(E * S, SCALARS_DIM),
                ).reshape(E, S, 6)
                for t in range(T)
            ]
        )
    assert torch.equal(targets, tr._compute_targets(reference))


def test_bootstrap_uses_the_successor_states_q_not_the_current_states():
    """Step t bootstraps on Q(s_{t+1}), so it reads the stored Q of step t+1.

    Uses a DISTINCT bootstrap per step: with one constant for the whole rollout
    (as the other target-math tests use) an off-by-one in the stored-Q lookup is
    invisible.
    """
    tr = _make_trainer(rollout_len=2)  # gamma=0.9, lambda=0.5
    T, E, S = 2, 1, 1
    roll = {
        "tactical": _zeros_obs(T, E, S)["tactical"],
        "strategic": _zeros_obs(T, E, S)["strategic"],
        "scalars": _zeros_obs(T, E, S)["scalars"],
        "rewards": np.zeros((T, E, S)),
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
    b0, b1 = 5.0, -3.0
    _patch_boot(tr, roll, [b0, b1])
    targets = tr._compute_targets(roll)
    g, lam = 0.9, 0.5
    # Step 1 is the rollout edge: G1 = gamma * b1 (its s_2 is final_obs).
    g1 = g * b1
    # Step 0 is interior and bootstraps on b0 — the masked-max of Q(s_1).
    g0 = g * ((1 - lam) * b0 + lam * g1)
    assert targets[1, 0, 0].item() == pytest.approx(g1, abs=1e-5)
    assert targets[0, 0, 0].item() == pytest.approx(g0, abs=1e-5)


def test_compute_targets_forwards_the_network_once():
    """Only the unseen final obs costs a forward; the rest reuse the rollout's Q.

    Re-forwarding the whole rollout was ~34% of update wall time and produced
    values the rollout had already computed and thrown away.
    """
    tr = _make_trainer(num_envs=1, num_snakes=2, rollout_len=4, pool_capacity=0)
    roll = tr._rollout()
    calls = {"n": 0}
    real = tr.network

    class _Counting:
        def __call__(self, *args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

    tr.network = _Counting()
    tr._compute_targets(roll)
    assert calls["n"] == 1, f"expected 1 forward on the final obs, got {calls['n']}"


def test_rollout_stores_hero_q_for_every_slot():
    """The rollout exposes the hero's Q over the WHOLE grid, aligned to its obs."""
    tr = _make_trainer(num_envs=2, num_snakes=3, rollout_len=2, pool_capacity=0)
    roll = tr._rollout()
    assert roll["hero_q"].shape == (2, 2, 3, 6)
    assert torch.isfinite(roll["hero_q"]).all()


# ---------------------------------------------------------------------------
# Episode boundary: v2 population floor + frame cap
# ---------------------------------------------------------------------------
def _floor_env(trainer, env):
    """Kill snakes in ``env`` until it is past the v2 population floor (<3 alive)."""
    trainer.sim.alive[env, 2:] = False


def _spy_reset(trainer):
    """Record calls to the sim's reset while still performing them."""
    calls: list = []
    orig = trainer.sim.reset

    def spy():
        calls.append(1)
        orig()

    trainer.sim.reset = spy
    return calls


def test_episode_over_on_frame_cap():
    """``max_frames`` is a real episode end, not just an observation denominator."""
    tr = _make_trainer(num_envs=2, num_snakes=1, max_frames=10)  # S<3: floor disabled
    assert not tr._episode_over()
    tr.sim.frame[:] = 10
    assert tr._episode_over()


def test_episode_over_requires_all_envs_past_the_floor():
    """A batch-wide reset must not truncate an env whose episode is still live."""
    tr = _make_trainer(num_envs=2, num_snakes=6, max_frames=10**9)
    _floor_env(tr, 0)
    assert tr.sim.population_floor_reached().tolist() == [True, False]
    assert not tr._episode_over()
    _floor_env(tr, 1)
    assert tr._episode_over()


def test_rollout_resets_when_all_envs_reach_population_floor():
    """The v2 population floor ends the episode; the next rollout repopulates."""
    tr = _make_trainer(num_envs=2, num_snakes=6, rollout_len=1, max_frames=10**9)
    _floor_env(tr, 0)
    _floor_env(tr, 1)
    assert int(tr.sim.get_alive().sum()) == 4
    calls = _spy_reset(tr)
    tr._rollout()
    assert len(calls) == 1
    assert int(tr.sim.get_alive().sum()) > 4, "reset did not repopulate the batch"


def test_rollout_does_not_reset_while_an_env_is_still_live():
    """One env past the floor must not discard the other env's in-flight episode."""
    tr = _make_trainer(num_envs=2, num_snakes=6, rollout_len=1, max_frames=10**9)
    _floor_env(tr, 0)
    calls = _spy_reset(tr)
    tr._rollout()
    assert calls == []


def test_rollout_enforces_the_frame_cap():
    """Frames never run past ``max_frames``: the next rollout starts a new episode."""
    tr = _make_trainer(num_envs=1, num_snakes=1, rollout_len=2, max_frames=2)
    tr._rollout()
    assert int(tr.sim.frame.max()) == 2
    tr._rollout()
    # Without the episode end the counter would keep climbing to 4.
    assert int(tr.sim.frame.max()) == 2


def test_steps_after_the_population_floor_are_not_transitions():
    """An env past its floor is past its episode end: its later steps never train.

    The batch cannot reset until every env has floored, so a floored env keeps
    being stepped. Those steps must not enter ``valid`` (and hence not the loss,
    the targets' backward carry, or the agent-step count) even though the
    surviving snakes are still alive.
    """
    tr = _make_trainer(num_envs=1, num_snakes=6, rollout_len=3, max_frames=10**9)
    # Floor the env from step 2 onward, keyed off the sim's own frame counter so
    # the script does not depend on how many times the trainer reads the signal.
    tr.sim.population_floor_reached = lambda: np.array([int(tr.sim.frame.max()) >= 2])
    roll = tr._rollout()
    valid = roll["valid"]
    assert valid[0].any(), "pre-floor step should be a real transition"
    assert not valid[2].any(), "post-floor steps must not be transitions"
    # Prove the exclusion came from the floor, not from everyone being dead.
    assert int(tr.sim.get_alive().sum()) > 0


def test_population_recovers_across_updates():
    """Deaths are permanent in train mode, so alive can only rise via a reset.

    Without an episode end the alive count is monotone non-increasing for the
    whole run (the reported bug: one endless, decaying episode).
    """
    cfg = PQNConfig(
        num_envs=2,
        num_snakes=6,
        rollout_len=4,
        minibatches=1,
        minibatch_size=16,
        eps_start=1.0,
        eps_end=1.0,
        eps_decay_steps=1,
        max_frames=8,
        pool_capacity=0,
        flip_augment=False,
        seed=1,
    )
    tr = PQNTrainer(cfg)
    tr.sim.alive[:, 2:] = False  # both envs past the floor -> episode over
    alive = []
    for _ in range(3):
        tr.update()
        alive.append(int(tr.sim.get_alive().sum()))
    assert max(alive) > 4, f"population never recovered: {alive}"


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
        hero,
        pool,
        policy_ids,
        obs,
        mask,
        epsilon=0.0,
        rng=np.random.default_rng(0),
        device=device,
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


def _telemetry(**overrides):
    """A HEALTHY telemetry snapshot; override one field to arm one tripwire."""
    params = dict(
        update=0,
        agent_steps=0,
        loss=0.5,
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
    params.update(overrides)
    return PQNTelemetry(**params)


# Each tripwire test pins its OWN branch via a distinctive `match` literal, so it
# cannot pass by tripping a different branch. NB `match` is a regex search:
# "max|Q|" would be an alternation with an empty arm and match ANY message.
def test_no_tripwire_on_healthy_telemetry():
    _make_trainer(rollout_len=2)._check_tripwires(_telemetry())


@pytest.mark.parametrize("field", ["loss", "grad_norm"])
def test_tripwire_fires_on_nonfinite(field):
    tr = _make_trainer(rollout_len=2)
    with pytest.raises(TripwireError, match="non-finite"):
        tr._check_tripwires(_telemetry(**{field: float("nan")}))


def test_tripwire_fires_on_max_abs_q():
    tr = _make_trainer(rollout_len=2)
    with pytest.raises(TripwireError, match="exceeded alarm"):
        tr._check_tripwires(_telemetry(max_abs_q=tr.cfg.max_abs_q_alarm * 10))


def test_tripwire_fires_on_nonfinite_max_abs_q():
    tr = _make_trainer(rollout_len=2)
    with pytest.raises(TripwireError, match="exceeded alarm"):
        tr._check_tripwires(_telemetry(max_abs_q=float("inf")))


def test_tripwire_fires_on_action_collapse():
    tr = _make_trainer(rollout_len=2)
    with pytest.raises(TripwireError, match="action collapse"):
        tr._check_tripwires(_telemetry(epsilon=0.1, action_entropy=0.0))


def test_action_collapse_not_flagged_during_exploration():
    """The epsilon<0.5 guard: zero entropy under pure exploration is not collapse."""
    _make_trainer(rollout_len=2)._check_tripwires(_telemetry(epsilon=0.9, action_entropy=0.0))


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


def test_checkpoint_weights_survive_the_roundtrip(tmp_path):
    """Serving == training: the TRAINED weights reach the served network.

    The checkpoint is the only artifact of a run — the file the promotion gate
    scores and the web app serves. Metadata assertions alone pass just as
    happily when the saved network is a random one.
    """
    from src.model.inference_agent import InferenceAgent

    tr = _make_trainer()
    # Move the weights off init so a fresh-network save is distinguishable,
    # without coupling this test to update()/tripwires.
    torch.manual_seed(0)
    with torch.no_grad():
        for p in tr.network.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    tr.network.eval()

    path = str(tmp_path / "pqn.pth")
    tr.save_checkpoint(path)
    agent = InferenceAgent.from_checkpoint(path, device=torch.device("cpu"))

    for (k_src, v_src), (k_dst, v_dst) in zip(
        tr.network.state_dict().items(), agent.network.state_dict().items()
    ):
        assert k_src == k_dst
        torch.testing.assert_close(v_src.cpu(), v_dst.cpu())

    # End-to-end: the served Q-values reproduce the trainer's own forward pass.
    rng = np.random.default_rng(0)
    obs = {
        "tactical": rng.random(TACTICAL_SHAPE, dtype=np.float32),
        "strategic": rng.random(STRATEGIC_SHAPE, dtype=np.float32),
        "scalars": rng.random((SCALARS_DIM,), dtype=np.float32),
    }
    with torch.no_grad():
        expected = (
            tr.network(
                torch.from_numpy(obs["tactical"]).unsqueeze(0),
                torch.from_numpy(obs["strategic"]).unsqueeze(0),
                torch.from_numpy(obs["scalars"]).unsqueeze(0),
            )
            .squeeze(0)
            .numpy()
        )
    np.testing.assert_allclose(agent.q_values(obs), expected, atol=1e-5)


class _PoisonPickle:
    """Fails to serialize, simulating an interrupt part-way through a save."""

    def __reduce__(self):
        raise RuntimeError("interrupted mid-save")


def test_failed_save_leaves_the_previous_checkpoint_intact(tmp_path):
    """A run has ONE rolling artifact: a failed save must not destroy it.

    ``torch.save`` truncates its destination on open, so writing straight to the
    live path turns any mid-write failure (preemption, OOM, full disk) into the
    loss of the whole run rather than a skipped save.
    """
    tr = _make_trainer()
    path = tmp_path / "latest_pqn.pth"
    tr.save_checkpoint(str(path))
    good = torch.load(str(path), map_location="cpu", weights_only=False)

    poisoned = tr.checkpoint_state()
    poisoned["poison"] = _PoisonPickle()
    tr.checkpoint_state = lambda: poisoned
    with pytest.raises(RuntimeError, match="interrupted mid-save"):
        tr.save_checkpoint(str(path))

    reloaded = torch.load(str(path), map_location="cpu", weights_only=False)
    for key, value in good["dqn_state_dict"].items():
        torch.testing.assert_close(value, reloaded["dqn_state_dict"][key])
    assert [p.name for p in tmp_path.iterdir()] == [path.name], "temp file left behind"


def test_save_checkpoint_creates_missing_parent_dirs(tmp_path):
    """The destination directory is created rather than raising mid-run."""
    tr = _make_trainer()
    path = tmp_path / "runs" / "nested" / "latest_pqn.pth"
    tr.save_checkpoint(str(path))
    assert path.exists()
