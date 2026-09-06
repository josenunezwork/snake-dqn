"""Pin double_dqn_next_q / n_step_td_target to the exact inline math they replace.

The learner and the local policy both computed the Double-DQN n-step target inline.
Those copies were extracted into shared helpers. A training-math change would NOT
be caught by the inference eval, so this test reproduces the ORIGINAL inline code
and asserts the helpers match it bit-for-bit on random inputs.
"""

import torch

from src.training.action_mask import has_valid_actions, mask_invalid_q_values
from src.training.td_targets import double_dqn_next_q, n_step_td_target


def _fixture(seed: int):
    torch.manual_seed(seed)
    b, a = 24, 6
    next_q_online = torch.randn(b, a)
    next_q_target = torch.randn(b, a)
    rewards = torch.randn(b)
    dones = (torch.rand(b) > 0.7).float()
    masks = torch.rand(b, a) > 0.3
    bootstrap = torch.randint(1, 4, (b,)).float()
    return next_q_online, next_q_target, rewards, dones, masks, bootstrap


def _extreme_fixture(seed: int):
    """Like ``_fixture`` but guaranteed to straddle the clamp and trapped rows.

    ``_fixture`` draws rewards from ``randn``, so |td_target| stays near ~3 and the
    ±q_clip clamp can never bind; and a fully-invalid row has probability 0.3**6, so
    the trapped->0 branch never fires. Both branches would then be deletable with the
    bit-for-bit pins still green.
    """
    next_q_online, next_q_target, rewards, dones, masks, bootstrap = _fixture(seed)
    rewards = rewards * 60.0
    masks[0] = False
    masks[1] = False
    # A trapped row must not be terminal, or (1 - done) would zero the bootstrap
    # term anyway and hide whether next_q was zeroed.
    dones[0] = 0.0
    dones[1] = 0.0
    # Degenerate argmax over an all -1e9 row returns index 0, so index 0 of the
    # target net must be non-zero for the trapped->0 branch to be observable.
    next_q_target[0, 0] = 9.0
    next_q_target[1, 0] = -7.0
    return next_q_online, next_q_target, rewards, dones, masks, bootstrap


def test_matches_learner_inline():
    nqo, nqt, rewards, dones, masks, bootstrap = _fixture(0)
    gamma, n_step, q_clip = 0.99, 3, 100.0

    # --- original learner inline (apex_learner.compute_td_targets) ---
    masked = mask_invalid_q_values(nqo, None, action_masks=masks)
    valid = has_valid_actions(nqo, None, action_masks=masks)
    next_actions = masked.argmax(dim=1, keepdim=True)
    next_q = nqt.gather(1, next_actions).squeeze(1)
    next_q = torch.where(valid, next_q, torch.zeros_like(next_q))
    discounts = torch.pow(torch.full_like(rewards, gamma), bootstrap.to(rewards.device))
    old = torch.clamp(rewards + (1.0 - dones) * discounts * next_q, min=-q_clip, max=q_clip)

    # --- helpers ---
    new_next_q = double_dqn_next_q(nqo, nqt, None, masks)
    new = n_step_td_target(rewards, dones, new_next_q, bootstrap, gamma, n_step, q_clip)
    assert torch.equal(old, new)


def test_matches_policy_inline():
    nqo, nqt, rewards, dones, masks, bootstrap = _fixture(1)
    gamma, n_step, q_clip = 0.99, 3, 50.0

    # --- original policy inline (apex_policy._compute_double_dqn_loss): argmax
    # without keepdim, then gather(unsqueeze(1)); clamp ±50 ---
    masked = mask_invalid_q_values(nqo, None, action_masks=masks)
    valid = has_valid_actions(nqo, None, action_masks=masks)
    next_actions = masked.argmax(dim=1)
    next_q = nqt.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    next_q = torch.where(valid, next_q, torch.zeros_like(next_q))
    discounts = torch.pow(torch.full_like(rewards, gamma), bootstrap.to(rewards.device))
    old = torch.clamp(rewards + (1 - dones) * discounts * next_q, min=-q_clip, max=q_clip)

    new_next_q = double_dqn_next_q(nqo, nqt, None, masks)
    new = n_step_td_target(rewards, dones, new_next_q, bootstrap, gamma, n_step, q_clip)
    assert torch.equal(old, new)


def test_bootstrap_none_defaults_to_n_step():
    nqo, nqt, rewards, dones, masks, _ = _fixture(2)
    next_q = double_dqn_next_q(nqo, nqt, None, masks)
    explicit = n_step_td_target(
        rewards, dones, next_q, torch.full_like(rewards, 3.0), 0.99, 3, 50.0
    )
    defaulted = n_step_td_target(rewards, dones, next_q, None, 0.99, 3, 50.0)
    assert torch.equal(explicit, defaulted)


def test_matches_inline_on_clamping_and_trapped_inputs():
    """The bit-for-bit pin, on inputs that actually reach the clamp and trapped rows."""
    nqo, nqt, rewards, dones, masks, bootstrap = _extreme_fixture(0)
    gamma, n_step, q_clip = 0.99, 3, 50.0

    # --- original inline (apex_learner.compute_td_targets) ---
    masked = mask_invalid_q_values(nqo, None, action_masks=masks)
    valid = has_valid_actions(nqo, None, action_masks=masks)
    next_actions = masked.argmax(dim=1, keepdim=True)
    next_q = nqt.gather(1, next_actions).squeeze(1)
    next_q = torch.where(valid, next_q, torch.zeros_like(next_q))
    discounts = torch.pow(torch.full_like(rewards, gamma), bootstrap.to(rewards.device))
    unclamped = rewards + (1.0 - dones) * discounts * next_q
    old = torch.clamp(unclamped, min=-q_clip, max=q_clip)

    # Pin that this fixture reaches both branches, so the comparison below cannot
    # silently go vacuous again.
    assert bool((unclamped.abs() > q_clip).any())
    assert int((~valid).sum()) >= 2

    new_next_q = double_dqn_next_q(nqo, nqt, None, masks)
    new = n_step_td_target(rewards, dones, new_next_q, bootstrap, gamma, n_step, q_clip)
    assert torch.equal(old, new)


def test_clamp_binds_at_q_clip():
    rewards = torch.tensor([500.0, -500.0])
    out = n_step_td_target(rewards, torch.zeros(2), torch.zeros(2), None, 0.99, 3, 50.0)
    assert torch.equal(out, torch.tensor([50.0, -50.0]))


def test_fully_masked_row_contributes_zero():
    """A trapped next-state must not bootstrap off the degenerate -1e9 argmax."""
    # Online net selects (argmax at index 4); target net evaluates.
    nqo = torch.tensor([[1.0, 2.0, 3.0, 4.0, 9.0, 6.0], [1.0, 2.0, 3.0, 4.0, 9.0, 6.0]])
    # Index 0 is non-zero: argmax over an all-invalid row degenerates to 0, so a
    # zero there would make this pass with or without the trapped->0 branch.
    nqt = torch.tensor([[8.0, 5.0, 0.0, 0.0, 50.0, 0.0], [8.0, 5.0, 0.0, 0.0, 50.0, 0.0]])
    masks = torch.ones(2, 6, dtype=torch.bool)
    masks[0] = False  # row 0 trapped, row 1 is the untrapped control

    next_q = double_dqn_next_q(nqo, nqt, None, masks)
    assert next_q[0].item() == 0.0
    assert next_q[1].item() == 50.0  # control: a valid row still bootstraps

    rewards = torch.tensor([2.5, 2.5])
    for done in (0.0, 1.0):
        dones = torch.tensor([done, done])
        out = n_step_td_target(rewards, dones, next_q, None, 0.99, 3, 50.0)
        assert out[0].item() == 2.5  # reward alone, no bootstrap leak
