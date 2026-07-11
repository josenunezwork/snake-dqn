"""Shared Double-DQN n-step TD-target helpers (live training path).

Extracted from the old apex_priorities module (whose other contents were dead
reference code). Used by BOTH apex_learner and apex_policy so the core TD-target
computation lives in one place; pinned bit-for-bit by tests/test_td_target_helpers.py.
"""

from typing import Optional

import torch

from .action_mask import has_valid_actions, mask_invalid_q_values


def double_dqn_next_q(
    next_q_online: torch.Tensor,
    next_q_target: torch.Tensor,
    next_states: Optional[torch.Tensor],
    next_action_masks: Optional[torch.Tensor],
) -> torch.Tensor:
    """Double-DQN next-state value: online net selects, target net evaluates.

    Invalid actions are masked for selection; states with no valid action
    contribute 0. Shared by the learner and the local policy's batch training so
    the core TD computation lives in one place. Mirrors those sites exactly:
    ``has_valid_actions`` reads the RAW online Q-values (not the masked copy).
    """
    masked_online = mask_invalid_q_values(
        next_q_online, next_states, action_masks=next_action_masks
    )
    valid_next_actions = has_valid_actions(
        next_q_online, next_states, action_masks=next_action_masks
    )
    next_actions = masked_online.argmax(dim=1, keepdim=True)
    next_q = next_q_target.gather(1, next_actions).squeeze(1)
    return torch.where(valid_next_actions, next_q, torch.zeros_like(next_q))


def n_step_td_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q: torch.Tensor,
    bootstrap_steps: Optional[torch.Tensor],
    gamma: float,
    n_step_default: int,
    q_clip: float,
) -> torch.Tensor:
    """``rewards + (1 - dones) * gamma**bootstrap * next_q``, clamped to ±q_clip.

    Per-sample bootstrap horizons (partial tail flushes / offline replay can be
    shorter than ``n_step_default``). Shared by the learner and the local policy.
    """
    if bootstrap_steps is None:
        bootstrap_steps = torch.full_like(rewards, float(n_step_default))
    discounts = torch.pow(torch.full_like(rewards, gamma), bootstrap_steps.to(rewards.device))
    td_targets = rewards + (1.0 - dones) * discounts * next_q
    return torch.clamp(td_targets, min=-q_clip, max=q_clip)
