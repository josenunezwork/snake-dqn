"""Action-validity helpers derived from the compact snake state vector."""

import torch

from src.core.game_config import GameConfig, StateIndices

INVALID_Q_VALUE = -1.0e9
ACTION_DANGER_COLLISION_THRESHOLD = 1.0


def action_mask_from_safe_actions(
    safe_actions: list[int],
    device: torch.device | None = None,
    allow_fallback: bool = False,
) -> torch.Tensor:
    """Build a boolean action mask from simulator-validated safe action ids."""
    mask = torch.zeros(GameConfig.OUTPUT_SIZE, dtype=torch.bool, device=device)
    for action in safe_actions:
        action_idx = int(action)
        if 0 <= action_idx < GameConfig.OUTPUT_SIZE:
            mask[action_idx] = True

    if allow_fallback and not bool(mask.any()):
        mask[:3] = True
    return mask


def valid_action_mask_from_states(states: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask of executable actions inferred from state rows.

    The 58D state encodes immediate danger for relative left/straight/right at
    indices 54-56. Normal actions are valid when their direction is not an
    immediate collision.

    Boost actions move two steps and the compact state cannot prove their
    destination is safe. Only simulator-provided exact masks should allow boost
    targets, so this state-derived fallback keeps boost actions invalid.

    For smaller synthetic test states or non-standard action counts, this
    helper returns an all-valid mask so legacy tests and alternate toy networks
    keep their old behavior.
    """
    output_size = GameConfig.OUTPUT_SIZE
    mask_shape = (*states.shape[:-1], output_size)
    if states.shape[-1] <= StateIndices.BOOST_AVAILABLE or output_size != 6:
        return torch.ones(mask_shape, dtype=torch.bool, device=states.device)

    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    directional_danger = states[..., danger_start:danger_end]
    normal_actions = torch.isfinite(directional_danger) & (
        directional_danger < ACTION_DANGER_COLLISION_THRESHOLD
    )

    boost_actions = torch.zeros_like(normal_actions, dtype=torch.bool)
    mask = torch.cat((normal_actions, boost_actions), dim=-1)

    return mask


def coerce_action_mask(action_masks: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
    """Return a boolean exact action mask aligned with Q-values."""
    mask = action_masks.to(device=q_values.device)
    if mask.dim() == 1 and q_values.dim() > 1 and mask.shape[0] == q_values.shape[-1]:
        mask = mask.reshape((1,) * (q_values.dim() - 1) + (mask.shape[0],))
    if mask.dtype == torch.bool:
        coerced = mask
    else:
        if torch.is_complex(mask):
            raise ValueError("action_mask values must be finite 0/1 or bool")
        if torch.is_floating_point(mask) and not bool(torch.isfinite(mask).all()):
            raise ValueError("action_mask values must be finite 0/1 or bool")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("action_mask values must be 0/1 or bool")
        coerced = mask.to(dtype=torch.bool)

    if coerced.shape != q_values.shape:
        raise ValueError(
            "action_mask shape must match q_values shape "
            f"(got {tuple(coerced.shape)}, expected {tuple(q_values.shape)})"
        )
    return coerced


def resolve_action_mask(
    q_values: torch.Tensor,
    states: torch.Tensor | None,
    action_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a boolean action mask aligned to q_values."""
    if action_masks is not None:
        return coerce_action_mask(action_masks, q_values)
    elif states is None:
        return torch.ones_like(q_values, dtype=torch.bool)
    else:
        state_mask = valid_action_mask_from_states(states)
        if state_mask.shape == q_values.shape:
            return state_mask
    return torch.ones_like(q_values, dtype=torch.bool)


def has_valid_actions(
    q_values: torch.Tensor,
    states: torch.Tensor | None,
    action_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return whether each row/time-step has at least one valid action."""
    return resolve_action_mask(q_values, states, action_masks=action_masks).any(dim=-1)


def summarize_next_action_quality(
    next_states: torch.Tensor,
    output_size: int,
    next_action_masks: torch.Tensor | None = None,
    next_action_mask_present: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize target-action coverage for replay rows that can bootstrap."""
    with torch.no_grad():
        q_probe = torch.empty(
            (*next_states.shape[:-1], output_size),
            dtype=torch.float32,
            device=next_states.device,
        )
        valid_next_actions = has_valid_actions(
            q_probe,
            next_states,
            action_masks=next_action_masks,
        )
        if sample_mask is not None:
            metric_mask = sample_mask.to(device=valid_next_actions.device, dtype=torch.bool)
            valid_next_actions = valid_next_actions[metric_mask]
            if not bool(valid_next_actions.numel()):
                return {
                    "valid_next_action_fraction": 0.0,
                    "trapped_next_state_fraction": 0.0,
                    "exact_next_action_mask_fraction": 0.0,
                }

        valid_fraction = valid_next_actions.float().mean().item()
        if next_action_mask_present is None:
            exact_mask_fraction = 0.0
        else:
            exact_mask_present = next_action_mask_present.to(device=next_states.device)
            if sample_mask is not None:
                exact_mask_present = exact_mask_present[metric_mask]
            exact_mask_fraction = exact_mask_present.float().view(-1).mean().item()

    return {
        "valid_next_action_fraction": float(valid_fraction),
        "trapped_next_state_fraction": float(1.0 - valid_fraction),
        "exact_next_action_mask_fraction": float(exact_mask_fraction),
    }


def mask_invalid_q_values(
    q_values: torch.Tensor,
    states: torch.Tensor | None,
    action_masks: torch.Tensor | None = None,
    invalid_value: float = INVALID_Q_VALUE,
) -> torch.Tensor:
    """Replace Q-values for invalid actions before argmax target selection."""
    mask = resolve_action_mask(q_values, states, action_masks=action_masks)
    invalid_fill = torch.full_like(q_values, invalid_value)
    return torch.where(mask, q_values, invalid_fill)
