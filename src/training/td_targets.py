"""Shared Double-DQN n-step TD-target helpers (live training path).

Extracted from the old apex_priorities module (whose other contents were dead
reference code). Used by BOTH apex_learner and apex_policy so the core TD-target
computation lives in one place; pinned bit-for-bit by tests/test_td_target_helpers.py.
"""

from typing import Optional

import torch

from .action_mask import has_valid_actions, mask_invalid_q_values

# Stored replay/IPC values.  A mask's presence never says whether it is advice
# or a resolved simulator fact; that distinction is carried by this field.
MASK_MODE_LEGACY_ADVISORY = 0
MASK_MODE_RASTER_RESOLVED_V3 = 1
MASK_MODE_TERMINAL_NO_SUCCESSOR = 2
MASK_MODE_DATASET_VECTOR_ADVISORY_V1 = 3
KNOWN_MASK_MODES = frozenset({0, 1, 2, 3})


def validate_mask_mode(mode: int) -> int:
    """Validate a closed persisted mask-mode value."""
    if isinstance(mode, bool) or not isinstance(mode, int) or mode not in KNOWN_MASK_MODES:
        raise ValueError(f"unknown next_action_mask_mode: {mode!r}")
    return mode


def domain_legal_action_mask(next_states: torch.Tensor) -> torch.Tensor:
    """Return executable controls, independent of collision-danger advice.

    Every alive vector row can make each normal turn. Boost controls additionally
    require the successor state's explicit availability feature (index 57).
    """
    if next_states.ndim < 1:
        raise ValueError("next_states requires an action-independent state axis")
    shape = (*next_states.shape[:-1], 6)
    legal = torch.ones(shape, dtype=torch.bool, device=next_states.device)
    if next_states.shape[-1] >= 58:
        boost_available = torch.isfinite(next_states[..., 57]) & (next_states[..., 57] >= 0.5)
        legal[..., 3:] = boost_available.unsqueeze(-1)
    return legal


def resolve_bootstrap_action_masks(
    next_states: torch.Tensor,
    next_action_masks: Optional[torch.Tensor],
    next_action_mask_modes: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Resolve saved masks row-wise under their recorded semantics."""
    if next_action_masks is None:
        if next_action_mask_modes is not None:
            raise ValueError("next_action_mask_modes requires next_action_masks")
        return None
    if next_action_masks.ndim != 2 or next_action_masks.shape[-1] != 6:
        raise ValueError("next_action_masks must have shape (batch, 6)")
    if next_action_mask_modes is None:
        modes = torch.full((next_action_masks.shape[0],), MASK_MODE_LEGACY_ADVISORY,
                           dtype=torch.long, device=next_action_masks.device)
    else:
        modes = next_action_mask_modes.to(device=next_action_masks.device)
        if modes.ndim != 1 or modes.shape[0] != next_action_masks.shape[0]:
            raise ValueError("next_action_mask_modes must have one entry per batch row")
        known = torch.tensor(sorted(KNOWN_MASK_MODES), device=modes.device)
        if not bool(torch.isin(modes, known).all()):
            raise ValueError("unknown next_action_mask_mode in batch")
    masks = next_action_masks.to(dtype=torch.bool)
    legal = domain_legal_action_mask(next_states)
    advisory = (modes == MASK_MODE_LEGACY_ADVISORY) | (modes == MASK_MODE_DATASET_VECTOR_ADVISORY_V1)
    intersection = legal & masks
    advisory_resolved = torch.where(intersection.any(dim=1, keepdim=True), intersection, legal)
    exact = modes == MASK_MODE_RASTER_RESOLVED_V3
    if bool((masks[exact] & ~legal[exact]).any()):
        raise ValueError("resolved next_action_mask includes domain-illegal action")
    resolved = torch.where(advisory.unsqueeze(1), advisory_resolved, masks)
    return torch.where((modes == MASK_MODE_TERMINAL_NO_SUCCESSOR).unsqueeze(1), False, resolved)


def double_dqn_next_q(
    next_q_online: torch.Tensor,
    next_q_target: torch.Tensor,
    next_states: Optional[torch.Tensor],
    next_action_masks: Optional[torch.Tensor],
    next_action_mask_modes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Double-DQN next-state value: online net selects, target net evaluates.

    Invalid actions are masked for selection; states with no valid action
    contribute 0. Shared by the learner and the local policy's batch training so
    the core TD computation lives in one place. Mirrors those sites exactly:
    ``has_valid_actions`` reads the RAW online Q-values (not the masked copy).
    """
    resolved_masks = (
        resolve_bootstrap_action_masks(next_states, next_action_masks, next_action_mask_modes)
        if next_states is not None
        else next_action_masks
    )
    masked_online = mask_invalid_q_values(
        next_q_online, next_states, action_masks=resolved_masks
    )
    valid_next_actions = has_valid_actions(
        next_q_online, next_states, action_masks=resolved_masks
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
