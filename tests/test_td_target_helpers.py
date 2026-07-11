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
