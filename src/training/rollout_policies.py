"""Immutable fixed-policy interface for corrected-v3 rollouts.

This module deliberately does not import the evaluation engine.  Its protocol
matches the simulator policy surface while keeping training's fixed opponents
explicitly identified for checkpoint and telemetry consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from src.evaluation.anchors import (
    AnchorContext,
    SCRIPTED_ANCHOR_KINDS,
    SCRIPTED_ANCHOR_VERSION,
    ScriptedAnchor,
    watch_anchor_frame,
)
from src.simd_env.batch_sim import BatchSim

__all__ = [
    "CanonicalWatchScriptedPolicy",
    "FixedPolicySource",
    "RolloutPolicy",
    "canonical_watch_scripted_source",
    "fixed_policy_actions",
]


@runtime_checkable
class RolloutPolicy(Protocol):
    """Immutable batched action policy supplied as a fixed rollout opponent."""

    @property
    def identity(self) -> str:
        """Stable immutable descriptor recorded with the rollout."""

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        """Return one action in ``[0, 5]`` for each sparse ``(env, slot)`` row."""


@dataclass(frozen=True)
class FixedPolicySource:
    """A fixed common opponent source with a mode and immutable identity."""

    policy: RolloutPolicy
    identity: str
    mode: str = "fixed"

    def __post_init__(self) -> None:
        if self.mode != "fixed":
            raise ValueError("FixedPolicySource mode must be 'fixed'")
        if not self.identity or self.identity != self.policy.identity:
            raise ValueError("fixed policy identity must be a non-empty policy identity")

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        """Validate and return actions for exactly the requested sparse rows."""
        if self.policy.identity != self.identity:
            raise ValueError("fixed policy identity changed after source construction")
        return fixed_policy_actions(self.policy, masks, sim, slots)


class CanonicalWatchScriptedPolicy:
    """Opt-in fixed opponent using the shared Watch scripted-anchor/v1 rule.

    The policy deliberately reads each lane's *current* seed from ``BatchSim``
    on every call.  That makes sparse selection and per-environment reseeding
    deterministic without coupling the trainer to simulator internals.
    """

    def __init__(self, kind: str) -> None:
        if kind not in SCRIPTED_ANCHOR_KINDS:
            raise ValueError(f"unknown scripted anchor kind {kind!r}")
        self._anchor = ScriptedAnchor(kind)
        self.kind = kind

    @property
    def identity(self) -> str:
        """Return the established immutable scripted-anchor identity."""
        return f"{SCRIPTED_ANCHOR_VERSION}:{self.kind}"

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        """Select exact shared-anchor actions for sparse prepared Watch rows."""
        if not isinstance(sim, BatchSim):
            raise ValueError("canonical Watch scripted policy requires a BatchSim")
        masks = np.asarray(masks, dtype=bool)
        slots = np.asarray(slots, dtype=np.int64)
        if masks.ndim != 2 or masks.shape[1] != 6:
            raise ValueError("masks must have shape (N, 6)")
        if slots.shape != (masks.shape[0], 2):
            raise ValueError("slots must have shape (N, 2) matching masks")
        if np.any(slots[:, 0] < 0) or np.any(slots[:, 0] >= sim.E):
            raise ValueError("slot environment indices must be in BatchSim bounds")
        if np.any(slots[:, 1] < 0) or np.any(slots[:, 1] >= sim.S):
            raise ValueError("slot snake indices must be in BatchSim bounds")

        world_seeds = sim.get_world_seeds()
        heads = sim.get_heads()
        headings = sim.get_direction_vectors()
        out = np.ones(len(slots), dtype=np.int64)
        for row, (env, slot) in enumerate(slots):
            env_i, slot_i = int(env), int(slot)
            out[row] = self._anchor.action(
                AnchorContext(
                    world_seed=world_seeds[env_i],
                    slot=slot_i,
                    frame=watch_anchor_frame(int(sim.frame[env_i])),
                    head_cell=tuple(int(value) for value in heads[env_i, slot_i]),
                    heading=tuple(int(value) for value in headings[env_i, slot_i]),
                    food_cells=tuple(
                        tuple(int(value) for value in cell) for cell in sim.get_food(env_i)
                    ),
                    allowed_mask=masks[row],
                )
            )
        return out


def canonical_watch_scripted_source(kind: str) -> FixedPolicySource:
    """Build an opt-in fixed source for the canonical Watch scripted anchor."""
    policy = CanonicalWatchScriptedPolicy(kind)
    return FixedPolicySource(policy, policy.identity)


def fixed_policy_actions(
    policy: RolloutPolicy, masks: np.ndarray, sim: BatchSim, slots: np.ndarray
) -> np.ndarray:
    """Call a fixed policy and validate its scatter-ready action vector.

    The caller owns scattering into the full ``(E, S)`` action grid; this helper
    retains sparse row order exactly, preventing roster-index assumptions.
    """
    masks = np.asarray(masks, dtype=bool)
    slots = np.asarray(slots, dtype=np.int64)
    if masks.ndim != 2 or masks.shape[1] != 6:
        raise ValueError("masks must have shape (N, 6)")
    if slots.shape != (masks.shape[0], 2):
        raise ValueError("slots must have shape (N, 2) matching masks")
    raw_actions = np.asarray(policy.actions(masks, sim, slots))
    if raw_actions.shape != (masks.shape[0],):
        raise ValueError("fixed policy actions must have shape (N,)")
    if np.issubdtype(raw_actions.dtype, np.bool_):
        raise ValueError("fixed policy actions must be non-boolean integers")
    if not (
        np.issubdtype(raw_actions.dtype, np.integer)
        or np.issubdtype(raw_actions.dtype, np.floating)
    ):
        raise ValueError("fixed policy actions must be numeric integers")
    if not np.all(np.isfinite(raw_actions)):
        raise ValueError("fixed policy actions must be finite")
    if np.issubdtype(raw_actions.dtype, np.floating) and not np.all(
        raw_actions == np.floor(raw_actions)
    ):
        raise ValueError("fixed policy actions must be integral")
    actions = raw_actions.astype(np.int64)
    if np.any((actions < 0) | (actions >= 6)):
        raise ValueError("fixed policy actions must be in [0, 5]")
    return actions
